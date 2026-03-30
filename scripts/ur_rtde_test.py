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

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

ARM_IDX = 0

# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("joint_target_file", type=str)
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
log_dir = os.path.dirname(args.joint_target_file)

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
# Load trajectory data
# ---------------------------

logger.info("Loading joint trajectory file", extra={"step": -1})

with open(args.joint_target_file) as f:
    data = json.load(f)

joint_targets = np.array(data["joint_targets_log"])
joint_pos = np.array(data["joint_positions_log"])
joint_torques = np.array(data["joint_torques_log"])

assert np.all(joint_pos <= 2*np.pi) and np.all(joint_pos >= -2*np.pi)
assert np.all(joint_targets <= 2*np.pi) and np.all(joint_targets >= -2*np.pi)

def split_lr(arr):
    arr_l = arr[..., :6]
    arr_r = arr[..., 6:]

    return arr_l, arr_r

joint_pos = split_lr(joint_pos)
joint_targets = split_lr(joint_targets)
joint_torques = split_lr(joint_torques)

def upsample_linear(matrix, scale_factor=10):
    N, K = matrix.shape
    if N < 2:
        return matrix # Cannot interpolate a single row
    
    # Define original row indices: [0, 1, 2, ..., N-1]
    original_indices = np.arange(N)
    
    # Define new indices. 
    # Example: If steps_per_segment is 10, indices will be [0, 0.1, 0.2, ..., N-1]
    num_samples = (N - 1) * scale_factor + 1
    new_indices = np.linspace(0, N - 1, num_samples)
    
    # Interpolate each column independently across the new row indices
    upsampled_matrix = np.apply_along_axis(
        lambda col: np.interp(new_indices, original_indices, col), 
        axis=0, 
        arr=matrix
    )
    
    return upsampled_matrix

# 50hz -> 500hz
joint_qs = upsample_linear(joint_pos[ARM_IDX], 10)
joint_target_qs = upsample_linear(joint_targets[ARM_IDX], 10)
joint_torques = upsample_linear(joint_torques[ARM_IDX], 10)

logger.info(
    f"Trajectory loaded. Total frames: {len(joint_target_qs)}",
    extra={"step": -1}
)

import psutil
os_used = sys.platform
process = psutil.Process(os.getpid())
if os_used == "win32":  # Windows (either 32-bit or 64-bit)
    process.nice(psutil.REALTIME_PRIORITY_CLASS)

# ---------------------------
# Robot connection
# ---------------------------

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

# UR5e max torques
max_torque = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

logger.info("Connection established", extra={"step": -1})

# Attempt to clear errors and reset robot state
logger.info("Resetting robot state", extra={"step": -1})
rtde_c.reuploadScript()

rtde_c.setPayload(0.025, [0.0, 0.0, 0.0])

# ---------------------------
# Control parameters
# ---------------------------

velocity = 0.5 # Not Used
acceleration = 0.5 # Not Used
lookahead_time = 0.2
gain = 300

logger.info(
    f"Control parameters: dt={dt}, velocity={velocity}, accel={acceleration}, "
    f"lookahead={lookahead_time}, gain={gain}",
    extra={"step": -1},
)

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
        "target_q",
        "actual_q",
        "loop_time",
        "robot_mode",
        "safety_mode",
    ]
)

np.set_printoptions(suppress=True, precision=3)

target_errors = np.zeros(len(joint_target_qs))
tracking_errors = np.zeros(len(joint_target_qs))

def log(i, dt):
    actual_q = np.array(rtde_r.getActualQ())
    target_q = joint_target_qs[i-1] if i >= 1 else joint_qs[0]
    expected_q = joint_qs[i]  if i >= 0 else joint_qs[0]

    tracking_error = np.linalg.norm(actual_q - expected_q)
    target_error = np.linalg.norm(target_q - actual_q)

    logger.info(
        f"Target error   {target_error:.6f}. Actual: {actual_q}. Target:   {target_q}",
        extra={"step": i},
    )

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
            target_q.tolist(),
            actual_q.tolist(),
            dt,
            robot_mode,
            safety_mode,
        ]
    )

    if i >= 0:
        target_errors[i] = target_error
        tracking_errors[i] = tracking_error
    
        expected_torque = joint_torques[i]
        external_torques = rtde_c.getJointTorques() # External Torque I think
        target_moments = rtde_r.getTargetMoment() # What should happen?
        raw_wrench = rtde_r.getFtRawWrench() # What is happening?
        # current_as_torque = rtde_r.getActualCurrentAsTorque() # Doesn't exist?
        
        logger.info(
            f"external_torques {external_torques}",
            extra={"step": i},
        )

        logger.info(
            f"target_moments {target_moments}",
            extra={"step": i},
        )

        logger.info(
            f"raw_wrench {raw_wrench}",
            extra={"step": i},
        )

        logger.info(
            f"Expected Torque: {expected_torque}",
            extra={"step": i}
        )

# ---------------------------
# Trajectory execution
# ---------------------------
try:
    logger.info(
        f"Starting trajectory for arm {ARM_IDX}",
        extra={"step": -1},
    )

    # Move to start
    logger.info(
        f"moveJ to initial position {joint_qs[0]}",
        extra={"step": -1},
    )

    last_time = time.perf_counter()

    success = rtde_c.moveJ(joint_qs[0], 0.5, 1.0, True)
    while not success:
        curr_time = time.perf_counter()
        success = rtde_c.moveJ(joint_qs[0], 0.5, 1.0, True)
        if curr_time - last_time > 5:
            raise TimeoutError("Ran out of time getting to initial position")

    last_time = time.perf_counter()

    while rtde_c.isSteady() == False:
        # Get actual joint positions (6 floats)
        curr_time = time.perf_counter()
        loop_time = curr_time - last_time
        last_time = curr_time

        log(-1, loop_time)

        # No need to check on this more often
        time.sleep(0.01)


    for i, target_q in enumerate(joint_target_qs):
        try:
            t_start = rtde_c.initPeriod()

            # send command
            # rtde_c.servoJ(
            #     target_q,
            #     velocity,
            #     acceleration,
            #     dt,
            #     lookahead_time,
            #     gain,
            # )

            torque = joint_torques[i]

            # rtde_c.directTorque(torque)
            rtde_c.directTorque([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            log(i, dt)

            # Should be at the end to ensure correct timing
            rtde_c.waitPeriod(t_start)

        except Exception as e:

            logger.error(
                f"servoJ exception: {str(e)}",
                extra={"step": i},
            )

            raise
    
    # Plot here
    plt.figure(figsize=(10, 6))
    plt.plot(tracking_errors, label='Tracking Error (Actual vs Expected)', alpha=0.7)
    plt.plot(target_errors, label='Target Error (Actual vs Target)', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Error (L2 Norm)')
    plt.title(f'Trajectory Errors - Arm {ARM_IDX}')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(log_dir, f"error_plot_arm_{ARM_IDX}_{date_t}.png")
    plt.savefig(plot_path)
    logger.info(f"Plot saved to {plot_path}", extra={"step": -1})
    plt.show()

except KeyboardInterrupt:

    logger.warning(
        "Execution interrupted by user",
        extra={"step": -1},
    )


finally:

    logger.info("Stopping servo", extra={"step": -1})
    rtde_c.servoStop()
    rtde_c.stopScript()

    csv_file.close()

    logger.info(
        f"Execution finished. Logs written to {log_path} and {csv_path}",
        extra={"step": -1},
    )
