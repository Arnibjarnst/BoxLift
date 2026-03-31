import argparse
import os
import sys
from datetime import datetime
import json
import logging
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import threading
import onnxruntime as ort

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


ARM_IDX = 0

# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--onnx_model_path", type=str)
parser.add_argument("--reference_trajectory_path", type=str)
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
log_dir = os.path.join(os.path.dirname(os.path.dirname(args.onnx_model_path)), "ur_rtde_logs")
os.makedirs(log_dir, exist_ok=True)

# ---------------------------
# Logging Setup
# ---------------------------

logger = logging.getLogger("trajectory_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | step=%(step)s | %(message)s"
)

date_t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = os.path.join(log_dir, f"trajectory_debug_{date_t}.log")
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ---------------------------
# Load model and trajectory
# ---------------------------

logger.info("Loading model", extra={"step": -1})
# 'CUDAExecutionProvider' uses the GPU; 'CPUExecutionProvider' is the fallback
session = ort.InferenceSession(args.onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 2. Identify input and output names (Isaac Lab usually uses "obs" and "mu" or "action")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

logger.info("Loading reference trajectory file", extra={"step": -1})
traj = np.load(args.reference_trajectory_path)

joints_l        = traj["joints_l"]
joints_r        = traj["joints_r"]
joint_vel_l     = traj["joint_vel_l"]
joint_vel_r     = traj["joint_vel_r"]
joints_target_l = traj["joints_target_l"]
joints_target_r = traj["joints_target_r"]


logger.info(
    f"Trajectory loaded. Total frames: {len(joints_l)}",
    extra={"step": -1}
)


# ---------------------------
# Robot connection
# ---------------------------

import psutil
os_used = sys.platform
process = psutil.Process(os.getpid())
if os_used == "win32":  # Windows (either 32-bit or 64-bit)
    process.nice(psutil.REALTIME_PRIORITY_CLASS)

if args.real_robot:
    # Single Robot
    robot_ip = "192.168.1.100"
else:
    if os_used == "win32":
        robot_ip = "172.29.144.1"
    else:
        robot_ip = "192.168.56.1"


logger.info(f"Connecting to robot at {robot_ip}", extra={"step": -1})

rtde_frequency = 500
dt = 1 / rtde_frequency
policy_decimation = 10
max_steps = (len(joints_l) - 1) * policy_decimation

rtde_r = RTDEReceive(robot_ip, rtde_frequency)
print("RTDE_r connected successfully")
rtde_c = RTDEControl(robot_ip, rtde_frequency, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)

# UR5e max torques
max_torque = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

logger.info("Connection established", extra={"step": -1})

# Attempt to clear errors and reset robot state
logger.info("Resetting robot state", extra={"step": -1})
rtde_c.reuploadScript()

rtde_c.setPayload(0.025, [0.0, 0.0, 0.0])

# ---------------------------
# CSV logging
# ---------------------------

csv_path = os.path.join(log_dir, f"trajectory_debug_{date_t}.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow(
    [
        "arm_idx",
        "step",
        "actual_q",
        "expected_q",
        "loop_time",
        "robot_mode",
        "safety_mode",
    ]
)

np.set_printoptions(suppress=True, precision=8)

def log(i):
    actual_q = np.array(rtde_r.getActualQ())

    if i < 0:
        expected_q = joints_l[0]
    else:
        expected_i = i // policy_decimation
        if expected_i == len(joints_l) - 1:
            expected_q = joints_l[expected_i]
        else:
            expected_alpha = i / policy_decimation - expected_i
            expected_q = (1 - expected_alpha) * joints_l[expected_i] + expected_alpha * joints_l[expected_i+1]

    tracking_error = np.linalg.norm(actual_q - expected_q)

    logger.info(
        f"Tracking error {tracking_error:.6f}. Actual: {actual_q}. Expected: {expected_q}",
        extra={"step": i},
    )

    robot_mode = rtde_r.getRobotMode()
    safety_mode = rtde_r.getSafetyMode()

    if safety_mode != 1:  # 1 = NORMAL
        logger.error(
            f"Robot left NORMAL safety mode. safety_mode={safety_mode}",
            extra={"step": i},
        )
        raise RuntimeError("Robot safety event")

    if robot_mode != 7:  # 7 = RUNNING
        logger.error(
            f"Robot not running. robot_mode={robot_mode}",
            extra={"step": i},
        )
        raise RuntimeError("Robot stopped")

    # try:
    #     external_torques = rtde_c.getJointTorques() # External Torque I think
    #     target_moments = rtde_r.getTargetMoment() # What should happen?
    #     raw_wrench = rtde_r.getFtRawWrench() # What is happening?
    #     # current_as_torque = rtde_r.getActualCurrentAsTorque() # Doesn't exist?
        
    #     logger.info(
    #         f"external_torques {external_torques}",
    #         extra={"step": i},
    #     )

    #     logger.info(
    #         f"target_moments {target_moments}",
    #         extra={"step": i},
    #     )

    #     logger.info(
    #         f"raw_wrench {raw_wrench}",
    #         extra={"step": i},
    #     )
    # except:
    #     pass

    # CSV logging
    csv_writer.writerow(
        [
            ARM_IDX,
            i,
            actual_q.tolist(),
            expected_q.tolist(),
            dt,
            robot_mode,
            safety_mode,
        ]
    )


# ---------------------------
# Control parameters
# ---------------------------

velocity = 0.5 # Not Used
acceleration = 0.5 # Not Used
lookahead_time = 0.05
gain = 100


# ---------------------------
# Move To Start
# ---------------------------
try:
    logger.info(
        f"moveJ to initial position {joints_l[0]}",
        extra={"step": -1},
    )

    last_time = time.perf_counter()

    success = rtde_c.moveJ(joints_l[0], 0.5, 1.0, True)
    while not success:
        curr_time = time.perf_counter()
        success = rtde_c.moveJ(joints_l[0], 0.5, 1.0, True)
        if curr_time - last_time > 5:
            raise TimeoutError("Ran out of time getting to initial position")
        
        time.sleep(dt)

    last_time = time.perf_counter()

    while rtde_c.isSteady() == False:
        log(-1)

        # No need to be accurate
        time.sleep(dt)

    logger.info(
        f"moveJ finished to initial position {joints_l[0]}",
        extra={"step": -1},
    )
    print(np.array(rtde_r.getActualQ()))
except KeyboardInterrupt:

    logger.warning(
        "Execution interrupted by user",
        extra={"step": -1},
    )

finally:
    rtde_c.stopJ()

data_lock = threading.Lock()
current_target_q = np.array(rtde_r.getActualQ())
previous_target_q = np.copy(current_target_q)
current_target_i = 0
run_policy_event = threading.Event()  # Trigger for the policy
new_data_event = threading.Event()



def policy_thread():
    global current_target_q, previous_target_q, current_target_i
    trajectory_step = 0
    while True:
        run_policy_event.wait()
        run_policy_event.clear()

        q_l = rtde_r.getActualQ()
        relative_q_l = q_l - joints_l[trajectory_step]
        relative_q_r = np.zeros_like(relative_q_l)
        relative_q = np.concatenate((relative_q_l, relative_q_r))

        q_l_vel = rtde_r.getActualQd()
        q_r_vel = joint_vel_r[trajectory_step]
        q_vel = np.concatenate((q_l_vel, q_r_vel))

        phase = np.array([trajectory_step / (len(joints_l) - 1)])
        
        obs = np.concatenate((relative_q, q_vel, phase))[None, ...].astype(np.float32)

        output = session.run([output_name], {input_name: obs})[0][0]

        # TODO: SHOULD NOT BE HARDCODED HERE
        ACTION_SCALE = 0.2

        # 2. RL Inference
        new_joint_targets = joints_target_l[trajectory_step] + ACTION_SCALE * output[:6]
        # new_q_goal = joints_target_l[trajectory_step]
        
        with data_lock:
            previous_target_q = current_target_q
            current_target_q = new_joint_targets
            current_target_i = trajectory_step
            new_data_event.set()

        trajectory_step += 1


# Kp = 100
# Kd = 10
# def PD(q_target):
#     try:
#         actual_q = np.array(rtde_r.getActualQ())
#         actual_q_vel = np.array(rtde_r.getActualQd())
#         # TODO: add payload contribution maybe? might already be there (check real robot)
#         mass_matrix = np.array(rtde_c.getMassMatrix()).reshape((6,6))
#         cc_matrix = np.array(rtde_c.getCoriolisAndCentrifugalTorques())

#         acc = Kp * (q_target - actual_q) + Kd * (-actual_q_vel)

#         torques = mass_matrix @ acc + cc_matrix

#         torques = np.clip(torques, -max_torque, max_torque)

#         return torques
#     except Exception as e:
#         print(e)

#     return np.zeros(6)

def control_thread():
    step_counter = 0
    alpha = 0
    try:
        while True:
            t_start = rtde_c.initPeriod()

            if step_counter % policy_decimation == 0:
                run_policy_event.set()

            if new_data_event.is_set():
                new_data_event.clear()

                # Reset interpolation alpha
                alpha = 1.0 / policy_decimation

            with data_lock:
                interp_q = (1 - alpha) * previous_target_q + alpha * current_target_q

                # 4. Command the robot
                # TODO: Change to torque
                rtde_c.servoJ(
                    interp_q,
                    velocity,
                    acceleration,
                    dt,
                    lookahead_time,
                    gain,
                )

            alpha = min(alpha + 1.0 / policy_decimation, 1.0)
            step_counter += 1

            log(step_counter)

            if step_counter > max_steps:
                break

            # Should be at the end to ensure correct timing
            rtde_c.waitPeriod(t_start)

    except Exception as e:

        logger.error(
            f"servoJ exception: {str(e)}",
            extra={"step": step_counter},
        )

        raise

    finally:

        logger.info("Stopping servo", extra={"step": -1})
        rtde_c.servoStop()
        rtde_c.stopScript()

        csv_file.close()

        logger.info(
            f"Execution finished. Logs written to {log_path} and {csv_path}",
            extra={"step": -1},
        )


t1 = threading.Thread(target=policy_thread, daemon=True)
t2 = threading.Thread(target=control_thread, daemon=True)

t1.start()
t2.start()

# Keep main thread alive and when either thread dies we stop and exit
while t1.is_alive() and t2.is_alive():
    time.sleep(0.02)

rtde_c.stopJ()
rtde_c.servoStop()
rtde_c.stopScript()
