import argparse
import os
from datetime import datetime
import json
import logging
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

import rtde_control
import rtde_receive

# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("joint_target_file", type=str)
args = parser.parse_args()
log_dir = os.path.dirname(args.joint_target_file)

# ---------------------------
# Logging Setup
# ---------------------------

logger = logging.getLogger("trajectory_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | segment=%(segment)s step=%(step)s | %(message)s"
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

logger.info("Loading joint trajectory file", extra={"segment": -1, "step": -1})

with open(args.joint_target_file) as f:
    data = json.load(f)

joint_targets = np.array(data["joint_targets_log"])
joint_pos = np.array(data["joint_positions_log"])


joint_pos = np.clip(joint_pos, -2*np.pi, 2*np.pi)
joint_targets = np.clip(joint_targets, -2*np.pi, 2*np.pi)

joint_pos_l = joint_pos[:, :6]
joint_pos_r = joint_pos[:, 6:]
joint_pos = (joint_pos_l, joint_pos_r)

joint_target_l = joint_targets[:, :6]
joint_target_r = joint_targets[:, 6:]
joint_targets = (joint_target_l, joint_target_r)


logger.info(
    f"Trajectory loaded. Total frames: {len(joint_target_l)}",
    extra={"segment": -1, "step": -1},
)


# ---------------------------
# Robot connection
# ---------------------------

robot_ip = "192.168.56.1"

logger.info("Connecting to robot", extra={"segment": -1, "step": -1})

rtde_c = rtde_control.RTDEControlInterface(robot_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

logger.info("Connection established", extra={"segment": -1, "step": -1})

# Attempt to clear errors and reset robot state
logger.info("Resetting robot state", extra={"segment": -1, "step": -1})
rtde_c.reuploadScript()

rtde_c.setPayload(0.025, [0.0, 0.0, 0.0])

# ---------------------------
# Control parameters
# ---------------------------

velocity = 0.5
acceleration = 0.5
dt = 1.0 / 500
lookahead_time = 0.15
gain = 100

logger.info(
    f"Control parameters: dt={dt}, velocity={velocity}, accel={acceleration}, "
    f"lookahead={lookahead_time}, gain={gain}",
    extra={"segment": -1, "step": -1},
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
        "segment",
        "interp_step",
        "interp_t",
        "cmd_q",
        "actual_q",
        "tracking_error",
        "loop_time",
        "robot_mode",
        "safety_mode",
    ]
)

np.set_printoptions(suppress=True, precision=3)


# ---------------------------
# Trajectory execution
# ---------------------------
upsample_factor = 1
sub_steps = int(10 * upsample_factor)
try:
    for arm_idx in [0]:
        joint_qs = joint_pos[arm_idx]
        target_qs = joint_targets[arm_idx]
        

        logger.info(
            f"Starting trajectory for arm {arm_idx}",
            extra={"segment": -1, "step": -1},
        )

        # Move to start
        logger.info(
            f"moveJ to initial position {joint_qs[0]}",
            extra={"segment": -1, "step": -1},
        )

        success = rtde_c.moveJ(joint_qs[0])
        start_time = time.perf_counter()
        while not success:
            curr_time = time.perf_counter()
            if curr_time - start_time > 3:
                raise TimeoutError()
            success = rtde_c.moveJ(joint_qs[0])

        if success:
            # Move to start
            logger.info(
                f"Successfully moved to initial position",
                extra={"segment": -1, "step": -1},
            )
        else:
            logger.error(
                f"Failed to reach initial position",
                extra={"segment": -1, "step": -1},
            )
            raise RuntimeError()
        
        tracking_errors = []
        target_errors = []

        for i in range(len(target_qs)-1):
            joint_q_prev = joint_qs[i]
            joint_q_next = joint_qs[i + 1]

            target_q_prev = target_qs[i]
            target_q_next = target_qs[i + 1]

            for j in range(sub_steps):

                interp_t = j / sub_steps
                next_interp_t = (j + 1) / sub_steps
                target_q = target_q_prev * (1 - interp_t) + target_q_next * interp_t
                # expected joint configuration after step
                joint_q = joint_q_prev * (1 - next_interp_t) + joint_q_next * next_interp_t

                try:
                    loop_start = time.perf_counter()

                    t_start = rtde_c.initPeriod()

                    # send command
                    rtde_c.servoJ(
                        target_q,
                        velocity,
                        acceleration,
                        dt,
                        lookahead_time,
                        gain,
                    )

                    rtde_c.waitPeriod(t_start)

                    loop_time = time.perf_counter() - loop_start

                    # read robot state
                    actual_q = np.array(rtde_r.getActualQ())

                    robot_mode = rtde_r.getRobotMode()
                    safety_mode = rtde_r.getSafetyMode()

                    # timing violation detection
                    if loop_time > dt + 1e-5:
                        logger.warning(
                            f"Loop overrun {loop_time:.6f}s > {dt}",
                            extra={"segment": i, "step": j},
                        )

                    tracking_error = np.linalg.norm(actual_q - joint_q)
                    target_error = np.linalg.norm(target_q - actual_q)

                    tracking_errors.append(tracking_error)
                    target_errors.append(target_error)

                    logger.info(
                        f"Target error   {target_error:.6f}. Actual: {actual_q}. Target:   {target_q}",
                        extra={"segment": i, "step": j},
                    )

                    logger.info(
                        f"Tracking error {tracking_error:.6f}. Actual: {actual_q}. Expected: {joint_q}",
                        extra={"segment": i, "step": j},
                    )

                    logger.debug(
                        f"interp={interp_t:.3f} "
                        f"cmd_q={joint_q} "
                        f"actual_q={actual_q} "
                        f"err={tracking_error:.6f}",
                        extra={"segment": i, "step": j},
                    )

                    robot_mode = rtde_r.getRobotMode()
                    safety_mode = rtde_r.getSafetyMode()

                    if safety_mode != 1:  # 1 = NORMAL
                        logger.error(
                            f"Robot left NORMAL safety mode. safety_mode={safety_mode}",
                            extra={"segment": i, "step": j},
                        )
                        raise RuntimeError("Robot safety event")

                    if robot_mode != 7:  # 7 = RUNNING
                        logger.error(
                            f"Robot not running. robot_mode={robot_mode}",
                            extra={"segment": i, "step": j},
                        )
                        raise RuntimeError("Robot stopped")

                    # CSV logging
                    csv_writer.writerow(
                        [
                            arm_idx,
                            i,
                            j,
                            interp_t,
                            joint_q.tolist(),
                            actual_q.tolist(),
                            tracking_error,
                            loop_time,
                            robot_mode,
                            safety_mode,
                        ]
                    )

                except Exception as e:

                    logger.error(
                        f"servoJ exception: {str(e)}",
                        extra={"segment": i, "step": j},
                    )

                    raise
        
        # Plot here
        plt.figure(figsize=(10, 6))
        plt.plot(tracking_errors, label='Tracking Error (Actual vs Expected)', alpha=0.7)
        plt.plot(target_errors, label='Target Error (Actual vs Target)', alpha=0.7)
        plt.xlabel('Step')
        plt.ylabel('Error (L2 Norm)')
        plt.title(f'Trajectory Errors - Arm {arm_idx}')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(log_dir, f"error_plot_arm_{arm_idx}_{date_t}.png")
        plt.savefig(plot_path)
        logger.info(f"Plot saved to {plot_path}", extra={"segment": -1, "step": -1})
        plt.show()


except KeyboardInterrupt:

    logger.warning(
        "Execution interrupted by user",
        extra={"segment": -1, "step": -1},
    )


finally:

    logger.info("Stopping servo", extra={"segment": -1, "step": -1})
    rtde_c.servoStop()
    rtde_c.stopScript()

    csv_file.close()

    logger.info(
        "Execution finished. Logs written to trajectory_debug.log and trajectory_debug.csv",
        extra={"segment": -1, "step": -1},
    )
