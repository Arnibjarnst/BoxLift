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
parser.add_argument("--duration", default=10.0, type=float)
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

# ---------------------------
# Logging Setup
# ---------------------------

log_dir = "./logs/ur_rtde/"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("trajectory_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | step=%(step)s | %(message)s"
)

date_t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
real_or_sim_string = "real" if args.real_robot else "sim"
filename = f"trajectory_{args.duration}_{real_or_sim_string}"
log_path = os.path.join(log_dir, f"{filename}.log")
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

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

rtde_c = RTDEControl(robot_ip, rtde_frequency, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)
rtde_r = RTDEReceive(robot_ip, rtde_frequency)

logger.info("Connection established", extra={"step": -1})

# Attempt to clear errors and reset robot state
logger.info("Resetting robot state", extra={"step": -1})
rtde_c.reuploadScript()

rtde_c.setPayload(0.0, [0.0, 0.0, 0.0])

# ---------------------------
# CSV logging
# ---------------------------

csv_path = os.path.join(log_dir, f"{filename}.csv")
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

np.set_printoptions(suppress=True, precision=6)


# ---------------------------
# Create Trajectory
# ---------------------------
joints_0 = np.array([0.0, -np.pi/2, 0.0, 0.0, 0.0, 0.0])
delay = 0.5
amplitude = np.deg2rad(15)

steps = int(args.duration * rtde_frequency)
phase_shift = delay / args.duration * 2 * np.pi
t = np.linspace(0, 2*np.pi, steps) + phase_shift
joints = joints_0[None, ...] + amplitude * np.sin(t)[:, None]


def log(i):
    actual_q = np.array(rtde_r.getActualQ())


    expected_q = joints[i] if i >= 0 else joints_0
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
lookahead_time = 0.03
gain = 100


# ---------------------------
# Move To Start
# ---------------------------
try:
    logger.info(
        f"moveJ to initial position {joints_0}",
        extra={"step": -1},
    )

    last_time = time.perf_counter()

    success = rtde_c.moveJ(joints_0, 0.5, 1.0, True)
    while not success:
        curr_time = time.perf_counter()
        success = rtde_c.moveJ(joints_0, 0.5, 1.0, True)
        if curr_time - last_time > 5:
            raise TimeoutError("Ran out of time getting to initial position")

    last_time = time.perf_counter()

    while rtde_c.isSteady() == False:
        # Get actual joint positions (6 floats)
        curr_time = time.perf_counter()
        loop_time = curr_time - last_time
        last_time = curr_time

        log(-1)

        # No need to be accurate
        time.sleep(dt)
except KeyboardInterrupt:

    logger.warning(
        "Execution interrupted by user",
        extra={"step": -1},
    )

finally:
    rtde_c.stopJ()


first_command_time = None
first_command_delay = None

def control_thread():
    step_counter = 0
    global first_command_time
    try:
        while True:
            t_start = rtde_c.initPeriod()
                
            if first_command_time is None:
                first_command_time = time.perf_counter()
            
            # 4. Command the robot
            # TODO: Change to torque
            rtde_c.servoJ(
                joints[step_counter],
                velocity,
                acceleration,
                dt,
                lookahead_time,
                gain,
            )

            print("Exec time:", rtde_r.getActualExecutionTime())

            step_counter += 1

            log(step_counter)

            if step_counter >= len(joints) - 1:
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


def get_movement_start_t(delays):
    threshold = 1e-3
    detected = np.zeros(6)

    while not np.all(detected):
        curr_time = time.perf_counter()
        curr_joints = np.array(rtde_r.getActualQ())
        joint_error = np.abs(curr_joints - joints_0)

        for j in range(6):
            if not detected[j] and joint_error[j] > threshold:
                print(curr_time, first_command_time)
                delays[j] = curr_time - first_command_time
                detected[j] = True

delays = np.zeros(6)
t0 = threading.Thread(target=get_movement_start_t, args=(delays,), daemon=True)

t0.start()
time.sleep(0.001)

t1 = threading.Thread(target=control_thread, daemon=True)
t1.start()

while t0.is_alive() or t1.is_alive():
    time.sleep(0.02)

print(f"Motor Delay: {delays}")
