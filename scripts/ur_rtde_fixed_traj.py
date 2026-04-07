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
parser.add_argument("--delay", default=0.0, type=float)
parser.add_argument("--lookahead", default=0.05, type=float)
parser.add_argument("--gain", default=100, type=int)
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
args = parser.parse_args()

assert args.delay >= 0 and args.delay <= args.duration
assert args.duration >= 3

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
filename = f"trajectory_dur={args.duration}_delay={args.delay}_gain={args.gain}_lh={args.lookahead}_{real_or_sim_string}"
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

rtde_c = RTDEControl(robot_ip, rtde_frequency)
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
        "actual_qd",
        "target_q",
        "expected_q",
        "applied_torque",
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
delay = args.delay
amplitude = np.deg2rad(20)

steps = int(args.duration * rtde_frequency)
phase_shift = delay / args.duration * 2 * np.pi
t = np.linspace(0, 2*np.pi, steps) + phase_shift
joints = joints_0[None, ...] + amplitude * np.sin(t)[:, None]


torques=np.zeros_like(joints)

def log(i):
    actual_q = np.array(rtde_r.getActualQ())
    actual_qd = np.array(rtde_r.getActualQd())
    applied_torque = np.array(rtde_r.getTargetMoment())

    if i >= 0:
        torques[i] = applied_torque

    expected_q = joints[i] if i >= 0 else joints_0
    tracking_error = np.linalg.norm(actual_q - expected_q)

    logger.info(
        f"Tracking error {tracking_error:.6f}. Actual: {actual_q}. Expected: {expected_q}",
        extra={"step": i},
    )

    logger.info(
        f"Applied Torque: {applied_torque}",
        extra={"step": i},
    )

    logger.info(
        f"Joint Velocity: {actual_qd}",
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
            actual_qd.tolist(),
            expected_q.tolist(),
            expected_q.tolist(),
            applied_torque.tolist(),
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
lookahead_time = args.lookahead
gain = args.gain


max_torque = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])
Kp = 100
Kd = 10
def PD(q_target):
    try:
        actual_q = np.array(rtde_r.getActualQ())
        actual_q_vel = np.array(rtde_r.getActualQd())

        acc = Kp * (q_target - actual_q) + Kd * (-actual_q_vel)

        # mass_matrix = np.array(rtde_c.getMassMatrix()).reshape((6,6))
        # cc_matrix = np.array(rtde_c.getCoriolisAndCentrifugalTorques())
        # torques = mass_matrix @ acc + cc_matrix

        torques = acc

        torques = np.clip(torques, -max_torque, max_torque)

        return torques
    except Exception as e:
        print(e)

    return np.zeros(6)


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
        
        time.sleep(0.02)

    last_time = time.perf_counter()

    while rtde_c.isSteady() == False:
        # Get actual joint positions (6 floats)
        log(-1)

        # No need to be accurate
        time.sleep(dt)
except KeyboardInterrupt:

    logger.warning(
        "Execution interrupted by user",
        extra={"step": -1},
    )

logger.info(
    f"Finished moveJ to initial position {joints_0}",
    extra={"step": -1},
)

first_command_time = None
first_command_delay = None

pd_torques = []

def control_thread():
    step_counter = 0
    global first_command_time
    try:
        while True:
            t_start = rtde_c.initPeriod()
                
            if first_command_time is None:
                first_command_time = rtde_c.initPeriod().total_seconds()
            
            # torques = PD(joints[step_counter])
            # pd_torques.append(torques)

            # dq = np.array(rtde_r.getActualQd())
            # kd = 10
            # damping = kd * dq
            # tau_desired = np.array([0.0, 0.0, 0.0, 4.0, 0.0, 0.0])
            # # clamp damping so it only ever opposes tau_desired, never amplifies it
            # damping = np.clip(damping, -np.abs(tau_desired), np.abs(tau_desired))
            # torques = tau_desired - damping
            # rtde_c.directTorque(torques.tolist(), friction_comp=True)
            
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

times = []
vels = []
currs = []

def get_movement_start_t(delays):
    # shouldn't be lower than 2e-4
    threshold = 1e-4
    detected = np.zeros(6)

    while not np.all(detected):
        curr_time = rtde_c.initPeriod().total_seconds()
        curr_vel = np.array(rtde_r.getActualQd())
        current = np.array(rtde_r.getActualCurrent())

        times.append(curr_time)
        vels.append(curr_vel)
        currs.append(current)

        for j in range(6):
            if not detected[j] and curr_vel[j] > threshold and first_command_time is not None:
                delays[j] = curr_time - first_command_time
                detected[j] = True

        busy_wait_s = 0.05 / 1000 #0.05ms
        end = time.perf_counter() + busy_wait_s
        while time.perf_counter() < end:
            pass

delays = np.zeros(6)
t0 = threading.Thread(target=get_movement_start_t, args=(delays,), daemon=True)

t0.start()
time.sleep(0.005)

t1 = threading.Thread(target=control_thread, daemon=True)
t1.start()

while t0.is_alive() or t1.is_alive():
    time.sleep(0.02)


print(f"Motor Delay (joint error):       {delays * 1000}ms")

times = (np.array(times) - first_command_time) * 1000
vels = np.array(vels)
currs = np.array(currs)

import matplotlib.pyplot as plt
import numpy as np

arrays = {"Velocities": vels, "Currents": currs}

for name, arr in arrays.items():
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle(name, fontsize=14)
    axes = axes.flatten()

    for i in range(6):
        ax = axes[i]
        ax.plot(times, arr[:, i])
        ax.set_title(f"Dim {i+1}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylim(arr[:, i].min(), arr[:, i].max())

    plt.tight_layout()
    plt.show()


pd_torques = np.array(pd_torques)
plt.plot(torques)
plt.show()

plt.plot(pd_torques)
plt.show()