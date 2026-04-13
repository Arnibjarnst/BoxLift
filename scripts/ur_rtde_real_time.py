import argparse
import os
import sys
from datetime import datetime
import json
import logging
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
import threading
import onnxruntime as ort

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive


# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory (e.g. logs/rsl_rl/boxpush/2026-04-08_14-42-08)")
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
parser.add_argument("--gain", type=float, default=None, help="servoJ gain (overrides default)")
parser.add_argument("--lookahead", type=float, default=None, help="servoJ lookahead_time (overrides default)")
parser.add_argument("--action_scale", type=float, default=None, help="Override action scale (use 0 for pure trajectory)")
args = parser.parse_args()

# Load training config
with open(os.path.join(args.run_dir, "params", "env.yaml"), "r") as f:
    env_cfg = yaml.unsafe_load(f)

reference_trajectory_path = env_cfg["trajectory_path"]
action_scale = args.action_scale if args.action_scale is not None else env_cfg["action_scale"]
action_mode = env_cfg.get("action_mode", "A")  # default A for backward compat with older runs
obs_history_steps = int(env_cfg.get("obs_history_steps", 1))

export_dir = os.path.join(args.run_dir, "exported")
onnx_model_path = os.path.join(export_dir, "policy.onnx")
if not os.path.exists(onnx_model_path):
    raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}. Run training first (exports automatically).")

log_dir = os.path.join(args.run_dir, "ur_rtde_logs")
os.makedirs(log_dir, exist_ok=True)

# Resolve servo parameters early for file naming
_gain = args.gain if args.gain is not None else 100
_lookahead = args.lookahead if args.lookahead is not None else 0.05
date_t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_tag = f"{date_t}_gain{_gain}_la{_lookahead}" + (f"_as{action_scale}" if args.action_scale is not None else "")

# ---------------------------
# Logging Setup
# ---------------------------

logger = logging.getLogger("trajectory_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | step=%(step)s | %(message)s"
)

log_path = os.path.join(log_dir, f"{run_tag}.log")
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

logger.info(f"Run dir: {args.run_dir}, action_scale: {action_scale}, action_mode: {action_mode}, obs_history_steps: {obs_history_steps}, trajectory: {reference_trajectory_path}", extra={"step": -1})
logger.info("Loading model", extra={"step": -1})
# 'CUDAExecutionProvider' uses the GPU; 'CPUExecutionProvider' is the fallback
session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# 2. Identify input and output names (Isaac Lab usually uses "obs" and "mu" or "action")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

logger.info("Loading reference trajectory file", extra={"step": -1})
traj = np.load(reference_trajectory_path)

# Detect single vs dual arm trajectory
dual_arm = "joints_l" in traj

if dual_arm:
    joints        = traj["joints_l"]
    joint_vel_ref = traj["joint_vel_l"]
    joints_target = traj["joints_target_l"]
    joints_r      = traj["joints_r"]
    joint_vel_r   = traj["joint_vel_r"]
else:
    joints        = traj["joints"]
    joint_vel_ref = traj["joint_vel"]
    joints_target = traj["joints_target"]

logger.info(
    f"Trajectory loaded ({'dual' if dual_arm else 'single'} arm). Total frames: {len(joints)}",
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
max_steps = (len(joints) - 1) * policy_decimation

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

np.set_printoptions(suppress=True, precision=8)

# Per-timestep tracking data for analysis
tracking_log = []  # list of (step, actual_q, expected_q, target_q, tracking_error)

def log(i, target_q):
    actual_q = np.array(rtde_r.getActualQ())

    if i < 0:
        expected_q = joints[0]
    else:
        expected_i = i // policy_decimation
        if expected_i == len(joints) - 1:
            expected_q = joints[expected_i]
        else:
            expected_alpha = i / policy_decimation - expected_i
            print(expected_i, expected_alpha)
            expected_q = (1 - expected_alpha) * joints[expected_i] + expected_alpha * joints[expected_i+1]

    tracking_error = np.linalg.norm(actual_q - expected_q)

    if i >= 0:
        tracking_log.append({
            "step": i,
            "actual_q": actual_q.copy(),
            "expected_q": expected_q.copy(),
            "target_q": np.array(target_q).copy(),
        })

    logger.info(
        f"Tracking error {tracking_error:.6f}. Actual: {actual_q}. Expected: {expected_q}. Target: {target_q}",
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

    # Safety: check tracking error
    if i >= 0 and tracking_error > max_tracking_error:
        logger.error(
            f"Tracking error {tracking_error:.4f} exceeds limit {max_tracking_error}",
            extra={"step": i},
        )
        raise RuntimeError("Tracking error too large")

    # Safety: check joint velocities
    actual_qd = np.array(rtde_r.getActualQd())
    max_vel = np.max(np.abs(actual_qd))
    if max_vel > max_joint_velocity:
        logger.error(
            f"Joint velocity {max_vel:.4f} exceeds limit {max_joint_velocity}. Qd: {actual_qd}",
            extra={"step": i},
        )
        raise RuntimeError("Joint velocity too high")


# ---------------------------
# Control parameters
# ---------------------------

velocity = 0.5 # Not Used
acceleration = 0.5 # Not Used
lookahead_time = _lookahead
gain = _gain

# ---------------------------
# Safety limits
# ---------------------------

# Max allowed deviation of command from current position (rad)
max_target_delta = 0.5
# Max allowed tracking error before stopping (rad)
max_tracking_error = 10.0
# Max allowed joint velocity before stopping (rad/s)
max_joint_velocity = 10000
# UR5e joint limits (rad) [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
joint_limits_upper = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

# ---------------------------
# Move To Start
# ---------------------------
try:
    logger.info(
        f"moveJ to initial position {joints[0]}",
        extra={"step": -1},
    )

    last_time = time.perf_counter()

    success = rtde_c.moveJ(joints[0], 0.5, 1.0, True)
    while not success:
        curr_time = time.perf_counter()
        success = rtde_c.moveJ(joints[0], 0.5, 1.0, True)
        if curr_time - last_time > 5:
            raise TimeoutError("Ran out of time getting to initial position")
        
        time.sleep(dt)

    last_time = time.perf_counter()

    while rtde_c.isSteady() == False:
        log(-1, joints[0])

        # No need to be accurate
        time.sleep(dt)

    logger.info(
        f"moveJ finished to initial position {joints[0]}",
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
    # Observation history buffer: (history_steps, per_step_features). Zero-initialized to
    # match sim reset behavior.
    per_step_features = 12 if not dual_arm else 24
    obs_history = np.zeros((obs_history_steps, per_step_features), dtype=np.float32)
    while True:
        run_policy_event.wait()
        run_policy_event.clear()

        actual_q = np.array(rtde_r.getActualQ())
        actual_qd = np.array(rtde_r.getActualQd())
        phase = np.array([trajectory_step / (len(joints) - 1)])

        if dual_arm:
            relative_q_l = actual_q - joints[trajectory_step]
            relative_q_r = np.zeros(6)
            relative_qd_l = actual_qd - joint_vel_ref[trajectory_step]
            relative_qd_r = np.zeros(6)
            current_features = np.concatenate((relative_q_l, relative_q_r, relative_qd_l, relative_qd_r))
        else:
            relative_q = actual_q - joints[trajectory_step]
            relative_qd = actual_qd - joint_vel_ref[trajectory_step]
            current_features = np.concatenate((relative_q, relative_qd))

        # Shift history and append newest
        obs_history = np.roll(obs_history, shift=-1, axis=0)
        obs_history[-1] = current_features

        obs = np.concatenate((obs_history.flatten(), phase))
        obs = obs[None, ...].astype(np.float32)
        output = session.run([output_name], {input_name: obs})[0][0]

        residual = action_scale * output[:6]
        if action_mode == "A":
            new_joint_targets = joints_target[trajectory_step] + residual
        elif action_mode == "B":
            new_joint_targets = joints[trajectory_step] + residual
        elif action_mode == "C":
            new_joint_targets = actual_q + residual
        elif action_mode == "D":
            planner_pd_error = joints_target[trajectory_step] - joints[trajectory_step]
            new_joint_targets = actual_q + planner_pd_error + residual
        else:
            raise ValueError(f"Unknown action_mode: {action_mode!r}")

        # Safety: clamp to joint limits, then clamp to be near current position
        new_joint_targets = np.clip(new_joint_targets, joint_limits_lower, joint_limits_upper)
        new_joint_targets = np.clip(new_joint_targets, actual_q - max_target_delta, actual_q + max_target_delta)

        with data_lock:
            previous_target_q = current_target_q
            current_target_q = new_joint_targets
            current_target_i = trajectory_step
            new_data_event.set()

        trajectory_step += 1


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
                # TODO: try interp_q = (1 - alpha) * interp_q + alpha * current_target_q
                # It is smoother but might be less accurate
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

            log(step_counter, interp_q)

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

        # Save per-timestep tracking data for analysis
        if tracking_log:
            tracking_npz_path = os.path.join(log_dir, f"{run_tag}.npz")
            np.savez(
                tracking_npz_path,
                steps=np.array([d["step"] for d in tracking_log]),
                actual_q=np.array([d["actual_q"] for d in tracking_log]),
                expected_q=np.array([d["expected_q"] for d in tracking_log]),
                target_q=np.array([d["target_q"] for d in tracking_log]),
                gain=gain,
                lookahead_time=lookahead_time,
                action_scale=action_scale,
            )
            logger.info(f"Tracking data saved to {tracking_npz_path}", extra={"step": -1})

        logger.info(
            f"Execution finished. Log written to {log_path}",
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
