import argparse
import json
import logging
import csv
import time
import numpy as np

import rtde_control
import rtde_receive


# ---------------------------
# Logging Setup
# ---------------------------

logger = logging.getLogger("trajectory_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | segment=%(segment)s step=%(step)s | %(message)s"
)

file_handler = logging.FileHandler("trajectory_debug.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("joint_target_file", type=str)
args = parser.parse_args()


# ---------------------------
# Load trajectory data
# ---------------------------

logger.info("Loading joint trajectory file", extra={"segment": -1, "step": -1})

with open(args.joint_target_file) as f:
    data = json.load(f)

joint_targets = np.array(data["joint_targets_log"])
joint_pos = np.array(data["joint_positions_log"])

def clamp_to_2pi(q):
    two_pi = 2 * np.pi
    while np.any(q > two_pi):
        q[q > two_pi] -= two_pi
    while np.any(q < -two_pi):
        q[q < -two_pi] += two_pi
    return q

joint_pos = clamp_to_2pi(joint_pos)
joint_targets = clamp_to_2pi(joint_targets)

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
rtde_c.re_initialize()
rtde_c.unlockProtectiveStop()

rtde_c.setPayload(0.01, [0.0, 0.0, 0.0])

# ---------------------------
# Control parameters
# ---------------------------

velocity = 0.5
acceleration = 0.5
dt = 1.0 / 500
lookahead_time = 0.1
gain = 300

logger.info(
    f"Control parameters: dt={dt}, velocity={velocity}, accel={acceleration}, "
    f"lookahead={lookahead_time}, gain={gain}",
    extra={"segment": -1, "step": -1},
)


# ---------------------------
# CSV logging
# ---------------------------

csv_file = open("trajectory_debug.csv", "w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow(
    [
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


# ---------------------------
# Trajectory execution
# ---------------------------

try:

    for arm_idx in [0, 1]:
        joint_qs = joint_pos[arm_idx]
        target_qs = joint_targets[arm_idx]
        

        logger.info(
            f"Starting trajectory for arm {arm_idx}",
            extra={"segment": -1, "step": -1},
        )

        # Move to start
        logger.info(
            f"moveJ to initial position {joint_qs[0].tolist()}",
            extra={"segment": -1, "step": -1},
        )

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

        for i in range(len(target_qs)-1):
            joint_q_prev = joint_qs[i]
            joint_q_next = joint_qs[i + 1]

            target_q_prev = target_qs[i]
            target_q_next = target_qs[i + 1]

            for j in range(10):

                interp_t = j / 10.0
                next_interp_t = (j + 1) / 10.0
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

                    tracking_error = np.linalg.norm(actual_q - joint_q)

                    robot_mode = rtde_r.getRobotMode()
                    safety_mode = rtde_r.getSafetyMode()

                    # timing violation detection
                    if loop_time > dt + 1e-5:
                        logger.warning(
                            f"Loop overrun {loop_time:.6f}s > {dt}",
                            extra={"segment": i, "step": j},
                        )

                    # tracking error warning
                    if tracking_error > 0.05:
                        logger.warning(
                            f"Large tracking error {tracking_error:.6f}. Actual: {actual_q}. Expected: {joint_q}",
                            extra={"segment": i, "step": j},
                        )

                    logger.debug(
                        f"interp={interp_t:.3f} "
                        f"cmd_q={joint_q.tolist()} "
                        f"actual_q={actual_q.tolist()} "
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
