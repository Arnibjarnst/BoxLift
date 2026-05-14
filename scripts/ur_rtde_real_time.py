import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
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

# Make tag_pose_estimation importable regardless of pip-install state.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TAG_POSE_DIR = _REPO_ROOT / "tag_pose_estimation"
sys.path.insert(0, str(_TAG_POSE_DIR))
from tag_pose_estimation.board_pose_listener import BoardPoseListener  # noqa: E402

# ---------------------------
# Pose-estimation streaming (hardcoded for now; promote to flag once we
# support multiple objects).
# ---------------------------
ENABLE_POSE_ESTIMATION = True
POSE_ESTIMATION_SCRIPT = _TAG_POSE_DIR / "scripts" / "run_pose_estimation.py"
# Path is relative to _TAG_POSE_DIR (which we set as the subprocess cwd).
POSE_ESTIMATION_CONFIG = (
    "config/pose_estimation_configs/bigbox_pose_estimation_config.json"
)
POSE_ESTIMATION_PORT = 5555
BOX_BOARD_ID = "0"
POSE_FIRST_POSE_TIMEOUT_S = 30.0


# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory (e.g. logs/rsl_rl/boxpush/2026-04-08_14-42-08)")
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
parser.add_argument("--gain", type=float, default=None, help="servoJ gain (overrides default)")
parser.add_argument("--lookahead", type=float, default=None, help="servoJ lookahead_time (overrides default)")
parser.add_argument("--action_scale", type=float, default=None, help="Override action scale (use 0 for pure trajectory)")
parser.add_argument("--isaacsim", action="store_true",
                    help="Run the policy against IsaacSim instead of URSim/RTDE. Used to diagnose "
                         "whether oscillation observed on URSim is caused by deployment-script bugs "
                         "(would persist in IsaacSim too) or URSim-specific behavior (would not). "
                         "Skips RTDE/pose-listener setup; runs single-threaded.")
# Lazy import of AppLauncher only if we may need it — keeps the rtde-only path light.
import sys as _sys
if any(a == "--isaacsim" for a in _sys.argv):
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# If running in IsaacSim mode, launch the omniverse app immediately so subsequent
# IsaacLab imports work. Disable pose estimation (the cube
# state comes from sim, not a tracker).
_isaacsim_app = None
if args.isaacsim:
    # Respect user's --headless choice (defaults to False so the viewer opens). Pass
    # --headless explicitly for batch / headless runs.
    _isaacsim_app = AppLauncher(args).app
    ENABLE_POSE_ESTIMATION = False
    print(f"[INFO] --isaacsim: running policy against IsaacSim backend "
          f"(headless={getattr(args, 'headless', False)}, no RTDE, no tracker, single-threaded).")

# Load training config
with open(os.path.join(args.run_dir, "params", "env.yaml"), "r") as f:
    env_cfg = yaml.unsafe_load(f)

reference_trajectory_path = env_cfg["trajectory_path"]
action_scale_cfg = args.action_scale if args.action_scale is not None else env_cfg["action_scale"]
# action_scale can be a scalar or a per-joint list in newer configs.
action_scale = np.asarray(action_scale_cfg, dtype=np.float32) if isinstance(action_scale_cfg, (list, tuple)) else float(action_scale_cfg)
action_mode = env_cfg.get("action_mode", "A")  # default A for backward compat with older runs
obs_history_steps = int(env_cfg.get("obs_history_steps", 1))
# New observation flags (defaults match legacy runs that predate these features).
include_object_obs = bool(env_cfg.get("include_object_obs", False))
# Tracker delay used during training (env steps). Box obs are computed against the trajectory
# reference from this many steps ago, matching the temporal alignment the policy was trained
# on. Default 13 ≈ 260ms at 20ms env step.
obs_obj_delay_steps = int(env_cfg.get("obs_obj_delay_steps", 13))
future_obs_steps = tuple(env_cfg.get("future_obs_steps", ()) or ())
include_prev_actions = bool(env_cfg.get("include_prev_actions", False))
include_absolute_obs = bool(env_cfg.get("include_absolute_obs", False))
# Thresholded EE↔cube contact bool. Real-side: derive from getActualTCPForce() with a
# baseline-offset model (the raw force has gripper-gravity bias that varies with pose
# but is roughly constant at the trajectory start, so we capture a baseline once at
# the rest pose and threshold ||delta|| each step). Sim-side (--isaacsim): read from
# the env's ee_contact_sensor directly.
include_contact_obs = bool(env_cfg.get("include_contact_obs", False))
contact_threshold = float(env_cfg.get("contact_threshold", 0.5))
contact_obs_delay_steps = int(env_cfg.get("contact_obs_delay_steps", 0))
# Bit-flip noise from training — not applied at deployment (we want clean readings) but
# kept here so the obs schema is identical. flip_prob is read but not used.
contact_obs_flip_prob = float(env_cfg.get("contact_obs_flip_prob", 0.0))
enable_phase_slowdown = bool(env_cfg.get("enable_phase_slowdown", False))
# Phase-slowdown / mode-D curriculum settings. Sentinel <0 on force_alpha disables the
# override; 1.0 ≈ "training completed warmup" which is what we usually want at deploy.
dphase_min = float(env_cfg.get("dphase_min", 0.5))
action_alpha_floor = float(env_cfg.get("action_alpha_floor", 0.1))
force_alpha = float(env_cfg.get("force_alpha", -1.0))
eff_alpha = float(force_alpha) if 0.0 <= force_alpha <= 1.0 else 1.0

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
robot_name = "real" if args.real_robot else "sim"
run_tag = f"{date_t}_gain{_gain}_la{_lookahead}_{robot_name}" + (f"_as{action_scale}" if args.action_scale is not None else "")

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

logger.info(
    f"Run dir: {args.run_dir}, action_scale: {action_scale_cfg}, action_mode: {action_mode}, "
    f"obs_history_steps: {obs_history_steps}, include_object_obs: {include_object_obs}, "
    f"future_obs_steps: {future_obs_steps}, include_prev_actions: {include_prev_actions}, "
    f"include_absolute_obs: {include_absolute_obs}, "
    f"enable_phase_slowdown: {enable_phase_slowdown}, dphase_min: {dphase_min}, "
    f"eff_alpha: {eff_alpha:.3f} (force_alpha={force_alpha}, "
    f"action_alpha_floor={action_alpha_floor}), "
    f"trajectory: {reference_trajectory_path}",
    extra={"step": -1},
)
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

# Reference object pose trajectory. Assumed to always be present and the same length as
# `joints` (i.e. one entry per trajectory step).
obj_poses_ref = traj["obj_poses"]


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz convention (matches isaaclab.utils.math.quat_mul)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float32)


def quat_inv_np(q: np.ndarray) -> np.ndarray:
    """Conjugate for unit quaternion (wxyz)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def interp_np(traj: np.ndarray, phase: float) -> np.ndarray:
    """Linear interpolation along the trajectory time axis (matches env._interp)."""
    T = traj.shape[0]
    p = float(np.clip(phase, 0.0, T - 1 - 1e-6))
    i0 = int(np.floor(p))
    i1 = min(i0 + 1, T - 1)
    a = p - i0
    return ((1.0 - a) * traj[i0] + a * traj[i1]).astype(np.float32)


def nlerp_np(traj_quat: np.ndarray, phase: float) -> np.ndarray:
    """Normalized lerp for unit quaternions (wxyz, hemisphere-corrected; matches env._nlerp)."""
    T = traj_quat.shape[0]
    p = float(np.clip(phase, 0.0, T - 1 - 1e-6))
    i0 = int(np.floor(p))
    i1 = min(i0 + 1, T - 1)
    a = p - i0
    q0 = traj_quat[i0].astype(np.float32)
    q1 = traj_quat[i1].astype(np.float32)
    if float((q0 * q1).sum()) < 0.0:
        q1 = -q1
    q = (1.0 - a) * q0 + a * q1
    return (q / max(float(np.linalg.norm(q)), 1e-8)).astype(np.float32)


IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

logger.info(
    f"Trajectory loaded ({'dual' if dual_arm else 'single'} arm). Total frames: {len(joints)}",
    extra={"step": -1}
)


# ---------------------------
# Launch pose estimation subprocess + listener
# ---------------------------
pose_proc = None
pose_listener = None
if ENABLE_POSE_ESTIMATION:
    logger.info(
        f"Launching pose estimation subprocess: {POSE_ESTIMATION_SCRIPT} "
        f"--config {POSE_ESTIMATION_CONFIG} (cwd={_TAG_POSE_DIR})",
        extra={"step": -1},
    )
    _pose_env = os.environ.copy()
    _pose_env["PYTHONPATH"] = (
        str(_TAG_POSE_DIR) + os.pathsep + _pose_env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    pose_proc = subprocess.Popen(
        [
            sys.executable,
            str(POSE_ESTIMATION_SCRIPT),
            "--config",
            POSE_ESTIMATION_CONFIG,
        ],
        cwd=str(_TAG_POSE_DIR),
        env=_pose_env,
        stdout=subprocess.DEVNULL,
    )
    pose_listener = BoardPoseListener(
        box_pose_socket_address=f"tcp://localhost:{POSE_ESTIMATION_PORT}",
        update_rate=0.01,
    )
    if not pose_listener.start():
        logger.error("Failed to start BoardPoseListener", extra={"step": -1})
        pose_proc.terminate()
        pose_proc = None
        pose_listener = None


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


rtde_frequency = 500
dt = 1 / rtde_frequency
policy_decimation = 10
max_steps = (len(joints) - 2) * policy_decimation
max_wallclock_steps = max_steps * 2

if not args.isaacsim:
    logger.info(f"Connecting to robot at {robot_ip}", extra={"step": -1})
    rtde_r = RTDEReceive(robot_ip, rtde_frequency)
    print("RTDE_r connected successfully")
    rtde_c = RTDEControl(robot_ip, rtde_frequency, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)
else:
    rtde_r = None
    rtde_c = None

# UR5e max torques
max_torque = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])

logger.info("Connection established", extra={"step": -1})

if not args.isaacsim:
    # Attempt to clear errors and reset robot state
    logger.info("Resetting robot state", extra={"step": -1})
    rtde_c.reuploadScript()
    rtde_c.setPayload(0.025, [0.0, 0.0, 0.0])

np.set_printoptions(suppress=True, precision=8)

# Per-timestep tracking data for analysis
tracking_log = []  # list of (step, actual_q, expected_q, target_q, tracking_error)


def save_tracking_npz():
    """Snapshot tracking_log and write the .npz. Safe to call multiple times — the
    write is idempotent (overwrite). Used by both the normal-exit path and the
    KeyboardInterrupt handler in main."""
    snapshot = list(tracking_log)  # shallow snapshot in case threads still append
    if not snapshot:
        return
    tracking_npz_path = os.path.join(log_dir, f"{run_tag}.npz")
    np.savez(
        tracking_npz_path,
        steps=np.array([d["step"] for d in snapshot]),
        phase=np.array([d["phase"] for d in snapshot]),
        actual_q=np.array([d["actual_q"] for d in snapshot]),
        expected_q=np.array([d["expected_q"] for d in snapshot]),
        target_q=np.array([d["target_q"] for d in snapshot]),
        actual_obj_pos=np.array([d["actual_obj_pos"] for d in snapshot]),
        actual_obj_quat=np.array([d["actual_obj_quat"] for d in snapshot]),
        # Source sample period (s) so downstream visualizers don't have to assume 500Hz.
        src_dt=np.float64(dt),
        gain=gain,
        lookahead_time=lookahead_time,
        action_scale=action_scale,
    )
    logger.info(f"Tracking data saved to {tracking_npz_path} ({len(snapshot)} steps)",
                extra={"step": -1})

def log(i, target_q, phase):
    actual_q = np.array(rtde_r.getActualQ())

    # Reference is the trajectory interpolated at the exact phase the policy thread
    # produced for this servo tick (control thread linearly blends previous→current).
    expected_q = interp_np(joints, phase)

    tracking_error = np.linalg.norm(actual_q - expected_q)

    # Latest estimated box pose from the pose-estimation stream, shifted into the robot
    # frame. NaN-filled when the listener isn't running or hasn't received a pose yet.
    box_pose = pose_listener.get_pose(BOX_BOARD_ID) if pose_listener is not None else None
    if box_pose is None:
        actual_obj_pos = np.full(3, np.nan, dtype=np.float32)
        actual_obj_quat = np.full(4, np.nan, dtype=np.float32)
    else:
        actual_obj_pos = box_pose[:3].astype(np.float32)
        actual_obj_quat = box_pose[3:].astype(np.float32)

    if i >= 0:
        tracking_log.append({
            "step": i,
            "phase": float(phase),
            "actual_q": actual_q.copy(),
            "expected_q": expected_q.copy(),
            "target_q": np.array(target_q).copy(),
            "actual_obj_pos": actual_obj_pos.copy(),
            "actual_obj_quat": actual_obj_quat.copy(),
        })

    logger.info(
        f"phase={phase:.4f}. Tracking error {tracking_error:.6f}. "
        f"Actual: {actual_q}. Expected: {expected_q}. Target: {target_q}. "
        f"obj_pos={actual_obj_pos}, obj_quat_wxyz={actual_obj_quat}",
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

# Max allowed tracking error before stopping (rad)
max_tracking_error = 10.0
# Max allowed joint velocity before stopping (rad/s)
max_joint_velocity = 10000
# UR5e joint limits (rad) [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
joint_limits_lower = np.array([-2*np.pi, -2*np.pi, -np.pi, -2*np.pi, -2*np.pi, -2*np.pi])
joint_limits_upper = np.array([ 2*np.pi,  2*np.pi,  np.pi,  2*np.pi,  2*np.pi,  2*np.pi])

# ---------------------------
# Move To Start (RTDE-only; IsaacSim mode starts at phase 0 via env reset)
# ---------------------------
if not args.isaacsim:
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
            log(-1, joints[0], 0.0)

            # No need to be accurate
            time.sleep(dt)

        logger.info(
            f"moveJ finished to initial position {joints[0]}",
            extra={"step": -1},
        )
    except KeyboardInterrupt:

        logger.warning(
            "Execution interrupted by user",
            extra={"step": -1},
        )

    finally:
        rtde_c.stopJ()

data_lock = threading.Lock()
if not args.isaacsim:
    current_target_q = np.array(rtde_r.getActualQ())
else:
    current_target_q = np.asarray(joints[0], dtype=np.float64)
previous_target_q = np.copy(current_target_q)

# Capture a baseline TCP wrench while the robot is at rest at the trajectory start
# pose. The raw getActualTCPForce() has a static gripper-gravity bias that varies with
# arm pose but is roughly constant at this initial pose; subtracting this baseline
# leaves ||delta|| as a usable proxy for external contact force.
# Only meaningful when include_contact_obs is set and RTDE is connected.
tcp_force_baseline = np.zeros(6, dtype=np.float64)
if include_contact_obs and not args.isaacsim and rtde_r is not None:
    logger.info("Sampling TCP force baseline (50 readings)...", extra={"step": -1})
    samples = np.array([rtde_r.getActualTCPForce() for _ in range(50)])
    tcp_force_baseline = samples.mean(axis=0)
    logger.info(
        f"TCP force baseline (Fxyz, Txyz) = {tcp_force_baseline.round(3).tolist()}",
        extra={"step": -1},
    )
# Float trajectory phase (matches the policy thread's `phase` variable). Updated under
# data_lock by the policy thread; the control thread blends previous→current the same
# way as target_q so we can compute the exact expected reference at any servo tick.
current_phase = 0.0
previous_phase = 0.0
run_policy_event = threading.Event()  # Trigger for the policy
new_data_event = threading.Event()

# Wait for the first box pose to arrive before starting threads. This is the
# point of the test — make sure the streaming pipeline is healthy.
if pose_listener is not None:
    logger.info(
        f"Waiting up to {POSE_FIRST_POSE_TIMEOUT_S:.0f}s for first box pose...",
        extra={"step": -1},
    )
    _start = time.perf_counter()
    while pose_listener.get_pose(BOX_BOARD_ID) is None:
        if pose_proc is not None and pose_proc.poll() is not None:
            logger.error(
                f"Pose-estimation subprocess exited early "
                f"(returncode={pose_proc.returncode})",
                extra={"step": -1},
            )
            break
        if time.perf_counter() - _start > POSE_FIRST_POSE_TIMEOUT_S:
            logger.warning(
                "Timed out waiting for first box pose; continuing without it",
                extra={"step": -1},
            )
            break
        time.sleep(0.1)
    else:
        logger.info(
            f"First box pose received: {pose_listener.get_pose(BOX_BOARD_ID)}",
            extra={"step": -1},
        )


def policy_thread():
    global current_target_q, previous_target_q, current_phase, previous_phase
    # Float phase to support enable_phase_slowdown. Indexed via floor() into trajectory
    # arrays. For legacy (no-slowdown) policies dphase=1 each step → phase increments by 1
    # → indexing matches the old integer trajectory_step behavior exactly.
    phase = 0.0
    # Per-step feature dim must match training: joints (12 or 24 for dual-arm)
    # [+ 13 for object state (rel_pos 3 + rel_quat 4 + rel_vel 6) if include_object_obs]
    # [+ same again (absolute versions) if include_absolute_obs].
    if dual_arm:
        per_step_features = 24
        if include_object_obs:
            raise NotImplementedError("include_object_obs with dual-arm is not supported in the real-robot script yet.")
        if include_absolute_obs:
            raise NotImplementedError("include_absolute_obs with dual-arm is not supported in the real-robot script yet.")
    else:
        # 12 = relative joint pos (6) + relative joint vel (6); always present.
        # +7 (pos+quat) if include_object_obs.
        per_step_features = 12 + (7 if include_object_obs else 0)
        if include_absolute_obs:
            per_step_features *= 2
        if include_contact_obs:
            per_step_features += 1
    # Observation history buffer: (history_steps, per_step_features). Zero-initialized to
    # match sim reset behavior.
    obs_history = np.zeros((obs_history_steps, per_step_features), dtype=np.float32)
    # Previous raw policy action (pre-scale), fed back into obs when include_prev_actions=True.
    prev_action = np.zeros(6, dtype=np.float32)
    max_traj_idx = len(joints) - 1
    # Phase history (length delay+1), newest-last. Each policy iteration appends the current
    # phase; the box obs uses index 0 (oldest) as the phase that was current `obs_obj_delay_steps`
    # iterations ago — matching the env's obj_phase_delay_buf so the rel pose comparison is
    # temporally aligned with the (already-delayed) tracker measurement.
    phase_history = np.zeros(obs_obj_delay_steps + 1, dtype=np.float32)
    # Note: no software contact-delay buffer here. The training-side
    # contact_obs_delay_steps exists to simulate the real TCP force sensor's intrinsic
    # latency (~10-30ms) — at deployment that latency is already physically present in
    # getActualTCPForce(), so applying additional software delay would double-shift it.
    # The --isaacsim path (which has no physical delay) does apply the buffer.
    while True:
        run_policy_event.wait()
        run_policy_event.clear()

        actual_q = np.array(rtde_r.getActualQ())
        actual_qd = np.array(rtde_r.getActualQd())
        traj_idx = min(int(np.floor(phase)), max_traj_idx)  # integer floor, used only for logging
        phase_obs = np.array([phase / max_traj_idx])

        # Roll phase history newest-last and stash the current phase. We do this BEFORE
        # building obs so phase_history[0] is the phase from delay_steps iterations ago.
        phase_history = np.roll(phase_history, -1)
        phase_history[-1] = phase
        delayed_phase = float(phase_history[0])

        # Interpolate trajectory references at the current float phase, matching the env's
        # _interp / _nlerp so the policy sees consistent conventions in sim and real.
        joints_at_phase        = interp_np(joints,        phase)
        joints_target_at_phase = interp_np(joints_target, phase)
        joint_vel_at_phase     = interp_np(joint_vel_ref, phase)
        obj_pos_at_phase       = interp_np(obj_poses_ref[:, :3], phase)
        obj_quat_at_phase      = nlerp_np(obj_poses_ref[:, 3:], phase)
        # Reference at the *delayed* phase — temporally aligned with the (delayed) tracker
        # measurement. Used to compute relative box obs in the same way the trained env did.
        obj_pos_at_delayed     = interp_np(obj_poses_ref[:, :3], delayed_phase)
        obj_quat_at_delayed    = nlerp_np(obj_poses_ref[:, 3:], delayed_phase)

        if dual_arm:
            relative_q_l = actual_q - joints_at_phase
            relative_q_r = np.zeros(6)
            relative_qd_l = actual_qd - joint_vel_at_phase
            relative_qd_r = np.zeros(6)
            current_features = np.concatenate((relative_q_l, relative_q_r, relative_qd_l, relative_qd_r))
        else:
            relative_q = actual_q - joints_at_phase
            relative_qd = actual_qd - joint_vel_at_phase
            feature_parts = [relative_q, relative_qd]
            # Latest box pose, shifted into the robot frame. None if listener is missing
            # or hasn't received a pose yet. Only fetched when include_object_obs is on,
            # since that's the only branch that consumes it (absolute obs reuses it too,
            # but only when include_object_obs is also true).
            actual_obj_pos = None
            actual_obj_quat = None
            if include_object_obs:
                box_pose = pose_listener.get_pose(BOX_BOARD_ID) if pose_listener is not None else None
                if box_pose is not None:
                    actual_obj_pos = box_pose[:3].astype(np.float32)
                    actual_obj_quat = box_pose[3:].astype(np.float32)

                # On the real robot, feed measured pose into the policy. In sim (or if the
                # pose stream is dead) fall back to zeros/identity (i.e. "actual == reference").
                use_actual = args.real_robot and actual_obj_pos is not None
                if actual_obj_pos is None:
                    logger.warning("No box pose available from listener; falling back to zeros",
                                   extra={"step": traj_idx})
                if use_actual:
                    # Compare actual (already lagged by tracker latency) against the reference
                    # at the same lag — matches sim training, where rel = delayed_actual minus
                    # reference_at_delayed_phase.
                    rel_obj_pos = actual_obj_pos - obj_pos_at_delayed
                    rel_obj_quat = quat_mul_np(obj_quat_at_delayed, quat_inv_np(actual_obj_quat))
                    logger.info(
                        f"Box obs: rel_pos={rel_obj_pos}, rel_quat_wxyz={rel_obj_quat} "
                        f"(actual_pos={actual_obj_pos}, ref_pos@delayed={obj_pos_at_delayed}, "
                        f"actual_quat={actual_obj_quat}, ref_quat@delayed={obj_quat_at_delayed}, "
                        f"delayed_phase={delayed_phase:.2f}, current_phase={phase:.2f})",
                        extra={"step": traj_idx},
                    )
                    feature_parts.append(rel_obj_pos.astype(np.float32))
                    feature_parts.append(rel_obj_quat.astype(np.float32))
                else:
                    feature_parts.append(np.zeros(3, dtype=np.float32))             # relative obj pos
                    feature_parts.append(IDENTITY_QUAT_WXYZ)                        # relative obj quat
            if include_absolute_obs:
                # Absolute joint state is directly observable from the robot.
                feature_parts.append(actual_q.astype(np.float32))
                feature_parts.append(actual_qd.astype(np.float32))
                if include_object_obs:
                    use_actual = args.real_robot and actual_obj_pos is not None
                    if use_actual:
                        feature_parts.append(actual_obj_pos)
                        feature_parts.append(actual_obj_quat)
                    else:
                        # Sim / no-tracker fallback: reference pose at current phase.
                        feature_parts.append(obj_pos_at_phase)
                        feature_parts.append(obj_quat_at_phase)
            if include_contact_obs:
                # Real-side contact bool: thresholded ||current_tcp_force - baseline||.
                # Force-only magnitude (first 3 dims); torque component is ignored —
                # contact with the cube manifests primarily as a translational push.
                # The sensor's intrinsic latency (~10-30ms) is what training's
                # contact_obs_delay_steps was modeling, so we don't add software delay
                # here — that would double-shift the signal.
                current_tcp_wrench = np.array(rtde_r.getActualTCPForce())
                delta_force = current_tcp_wrench[:3] - tcp_force_baseline[:3]
                force_mag = float(np.linalg.norm(delta_force))
                in_contact = 1.0 if force_mag > contact_threshold else 0.0
                feature_parts.append(np.array([in_contact], dtype=np.float32))
                logger.debug(
                    f"contact: force_mag={force_mag:.3f} N (thr={contact_threshold:.2f}), "
                    f"in_contact={in_contact}",
                    extra={"step": traj_idx},
                )
            current_features = np.concatenate(feature_parts).astype(np.float32)

        # Shift history and append newest
        obs_history = np.roll(obs_history, shift=-1, axis=0)
        obs_history[-1] = current_features

        obs_parts = [obs_history.flatten().astype(np.float32), phase_obs.astype(np.float32)]

        # Future reference obj pose look-ahead (pos delta + world-frame quat delta,
        # plus absolute future pos/quat if include_absolute_obs).
        if future_obs_steps:
            inv_cur_quat = quat_inv_np(obj_quat_at_phase)
            futures = []
            for k in future_obs_steps:
                fut_phase = phase + float(k)  # interp_np / nlerp_np clamp internally
                fut_pos = interp_np(obj_poses_ref[:, :3], fut_phase)
                fut_quat = nlerp_np(obj_poses_ref[:, 3:], fut_phase)
                futures.append(fut_pos - obj_pos_at_phase)
                futures.append(quat_mul_np(fut_quat, inv_cur_quat))
                if include_absolute_obs:
                    futures.append(fut_pos)
                    futures.append(fut_quat)
            obs_parts.append(np.concatenate(futures).astype(np.float32))

        # Previous raw residual action.
        if include_prev_actions:
            obs_parts.append(prev_action)

        obs = np.concatenate(obs_parts)[None, ...].astype(np.float32)
        output = session.run([output_name], {input_name: obs})[0][0]

        raw_action = output[:6].astype(np.float32)
        # Phase-slowdown action: when enabled the policy emits a 7th dim controlling dphase.
        # dphase = (1 + (1 - dphase_min) * tanh(action[6])).clamp(dphase_min, 1).
        if enable_phase_slowdown and len(output) >= 7:
            dphase = 1.0 + (1.0 - dphase_min) * float(np.tanh(output[6]))
            dphase = float(np.clip(dphase, dphase_min, 1.0))
        else:
            dphase = 1.0

        # Mode D applies the curriculum α to blend planner feedforward with the residual.
        # eff_alpha is read from saved env.yaml (force_alpha if set, else 1.0). At α=1 mode D
        # collapses to mode C. For modes A/B/C the formulas are α-independent.
        if action_mode == "A":
            new_joint_targets = joints_target_at_phase + action_scale * raw_action
        elif action_mode == "B":
            new_joint_targets = joints_at_phase + action_scale * raw_action
        elif action_mode == "C":
            new_joint_targets = actual_q + action_scale * raw_action
        elif action_mode == "D":
            eps = action_alpha_floor
            action_gain = eff_alpha + eps * (1.0 - eff_alpha)
            planner_pd_error = joints_target_at_phase - joints_at_phase
            new_joint_targets = (
                actual_q
                + (1.0 - eff_alpha) * planner_pd_error
                + action_gain * action_scale * raw_action
            )
        else:
            raise ValueError(f"Unknown action_mode: {action_mode!r}")

        # Safety: clamp to joint limits, then clamp to be near current position
        new_joint_targets = np.clip(new_joint_targets, joint_limits_lower, joint_limits_upper)

        # Stream box pose for logging only — not used in obs yet.
        if pose_listener is not None:
            box_pose = pose_listener.get_pose(BOX_BOARD_ID)
            if box_pose is not None:
                logger.debug(
                    f"box_pose [pos={box_pose[:3]}, quat_wxyz={box_pose[3:]}]",
                    extra={"step": traj_idx},
                )

        with data_lock:
            previous_target_q = current_target_q
            current_target_q = new_joint_targets
            previous_phase = current_phase
            current_phase = phase
            new_data_event.set()

        prev_action = raw_action
        # Advance phase by dphase (≤ 1; deepens slowdown when policy commands a pause).
        # Clamp at max_traj_idx so we don't index past the end of the trajectory.
        phase = min(phase + dphase, float(max_traj_idx))


def control_thread():
    step_counter = 0
    try:
        while True:
            t_start = rtde_c.initPeriod()

            if step_counter % policy_decimation == 0:
                run_policy_event.set()

            if new_data_event.is_set():
                new_data_event.clear()

            with data_lock:
                # ZOH: send the policy-thread-computed target unchanged. Training now
                # caches the target once per policy step (see _pre_physics_step in
                # boxhinge/boxpush envs) — so a single fixed target per 20ms window with
                # the low-level controller chasing it at 500Hz is a clean 1:1 match.
                interp_q = np.copy(current_target_q)
                interp_phase = current_phase

                # 4. Command the robot
                rtde_c.servoJ(
                    interp_q,
                    velocity,
                    acceleration,
                    dt,
                    lookahead_time,
                    gain,
                )

            step_counter += 1

            print(rtde_r.getActualTCPPose())
            log(step_counter, interp_q, interp_phase)

            if interp_phase * policy_decimation > max_steps:
                break

            if step_counter > max_wallclock_steps:
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

        logger.info(
            f"Execution finished. Log written to {log_path}",
            extra={"step": -1},
        )


if args.isaacsim:
    # =====================================================================
    # IsaacSim backend: single-threaded loop that reuses the same obs
    # construction logic as policy_thread but reads/writes IsaacSim state.
    # =====================================================================
    import gymnasium as gym
    import torch
    import BoxLift.tasks  # noqa: F401, registers env
    from BoxLift.tasks.direct.boxhinge.boxhinge_env_cfg import BoxhingeEnvCfg

    # Build env cfg using the same saved env.yaml fields that play.py / record.py restore.
    isaac_cfg = BoxhingeEnvCfg()
    isaac_cfg.scene.num_envs = 1
    isaac_cfg.episode_length_s = 1e6  # let us run as long as we want
    isaac_cfg.trajectory_path = env_cfg["trajectory_path"]
    # Restore obs/action layout
    for _f in ("obs_history_steps", "action_mode", "action_scale", "include_object_obs",
               "include_absolute_obs", "future_obs_steps", "include_prev_actions",
               "enable_phase_slowdown", "dphase_min", "dphase_max", "phase_mapping",
               "observation_space", "kp", "kd", "effort_limit", "velocity_limit",
               "actuator_type"):
        if _f in env_cfg and hasattr(isaac_cfg, _f):
            setattr(isaac_cfg, _f, env_cfg[_f])
    # Physics at 500Hz (matching URSim/real RTDE rate). Target is computed once per
    # policy step (before the substep loop) — matching the env's new behavior where
    # _pre_physics_step caches the target and _apply_action just re-sends it. Same
    # target held across all 10 substeps, low-level PD chases it.
    isaac_cfg.physics_dt = dt
    isaac_cfg.decimation = policy_decimation

    # Clean test: disable all noise / DR / perturbations / VOC / reset noise.
    isaac_cfg.obs_obj_pos_noise = 0.0
    isaac_cfg.obs_obj_ori_noise = 0.0
    isaac_cfg.obs_obj_pos_bias_std = 0.0
    isaac_cfg.obs_obj_ori_bias_std = 0.0
    isaac_cfg.obs_obj_delay_steps = 0
    isaac_cfg.obs_obj_update_period = 1
    isaac_cfg.voc_enabled = False
    isaac_cfg.perturbation_probability = 0.0
    isaac_cfg.reset_joint_pos_noise = [0.0] * 6
    isaac_cfg.reset_joint_vel_noise = [0.0] * 6
    isaac_cfg.reset_obj_pos_xy_noise = 0.0
    isaac_cfg.reset_obj_ori_noise = 0.0
    isaac_cfg.reset_obj_lin_vel_xy_noise = 0.0
    isaac_cfg.reset_obj_ang_vel_noise = 0.0
    isaac_cfg.events.actuator_gains = None
    isaac_cfg.events.object_physics_material = None
    isaac_cfg.events.table_physics_material = None
    isaac_cfg.events.object_mass = None
    if hasattr(isaac_cfg.events, "object_com"):
        isaac_cfg.events.object_com = None
    if hasattr(isaac_cfg.events, "reset_gravity"):
        isaac_cfg.events.reset_gravity = None

    isaac_env = gym.make("Template-Boxhinge-Direct-v0", cfg=isaac_cfg)
    _direct_env = isaac_env.unwrapped
    ur5e_art = _direct_env.ur5e
    isaac_sim = _direct_env.sim
    isaac_device = _direct_env.device
    # Force VOC off (matches play.py default)
    if hasattr(_direct_env, "voc_kp_pos"):
        _direct_env.voc_kp_pos = 0.0
        _direct_env.voc_kp_rot = 0.0
        _direct_env.voc_kv_pos = 0.0
        _direct_env.voc_kv_rot = 0.0
    # Hide visualization markers (cube_marker, ee_markers). They're normally driven by
    # _pre_physics_step which we bypass — so they'd sit at their default origin pose
    # (visible as a phantom "white box" inside the robot). The actual physics cube is
    # still drawn correctly.
    if hasattr(_direct_env, "cube_marker"):
        _direct_env.cube_marker.set_visibility(False)
    if hasattr(_direct_env, "ee_markers"):
        _direct_env.ee_markers.set_visibility(False)
    # Force start at phase 0. Match play.py / record.py: just call _reset_idx — calling
    # env.reset() after this would re-randomize and undo the phase=0 override.
    _direct_env._reset_idx(None, 0)

    # State for the loop (mirrors policy_thread's locals)
    isaac_phase = 0.0
    isaac_prev_action = np.zeros(6, dtype=np.float32)
    isaac_contact_history = np.zeros(contact_obs_delay_steps + 1, dtype=np.float32)
    isaac_obs_history = np.zeros((obs_history_steps, per_step_features), dtype=np.float32) \
        if "per_step_features" in globals() else None
    # If per_step_features wasn't set yet (it's defined inside policy_thread), compute here.
    if isaac_obs_history is None:
        _psf = 12 + (7 if include_object_obs else 0)
        if include_absolute_obs:
            _psf *= 2
        if include_contact_obs:
            _psf += 1
        isaac_obs_history = np.zeros((obs_history_steps, _psf), dtype=np.float32)

    _max_traj_idx = len(joints) - 1
    _isaac_step = 0
    _isaac_max_steps = (len(joints) - 2) * policy_decimation

    print("[INFO] IsaacSim loop starting...")
    try:
        while _isaac_step < _isaac_max_steps and isaac_phase < _max_traj_idx - 1e-3:
            actual_q  = ur5e_art.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
            actual_qd = ur5e_art.data.joint_vel[0].detach().cpu().numpy().astype(np.float64)

            joints_at_phase_l        = interp_np(joints,         isaac_phase)
            joints_target_at_phase_l = interp_np(joints_target,  isaac_phase)
            joint_vel_at_phase_l     = interp_np(joint_vel_ref,  isaac_phase)
            obj_pos_at_phase_l       = interp_np(obj_poses_ref[:, :3], isaac_phase)
            obj_quat_at_phase_l      = nlerp_np(obj_poses_ref[:, 3:], isaac_phase)

            phase_obs_l = np.array([isaac_phase / _max_traj_idx])

            relative_q_l  = (actual_q  - joints_at_phase_l).astype(np.float32)
            relative_qd_l = (actual_qd - joint_vel_at_phase_l).astype(np.float32)
            feature_parts_l = [relative_q_l, relative_qd_l]

            # Match URSim mode's no-tracker fallback: feed the policy "perfect tracking"
            # obs (rel = zeros/identity, abs = reference pose at current phase). Identical
            # obs construction to URSim mode, so this is a pure dynamics-only swap. Reading
            # the live cube pose here would be OOD — training used a delayed/noisy obj obs
            # path, not a clean instantaneous one.
            if include_object_obs:
                feature_parts_l.append(np.zeros(3, dtype=np.float32))
                feature_parts_l.append(IDENTITY_QUAT_WXYZ)

            if include_absolute_obs:
                feature_parts_l.append(actual_q.astype(np.float32))
                feature_parts_l.append(actual_qd.astype(np.float32))
                if include_object_obs:
                    feature_parts_l.append(obj_pos_at_phase_l.astype(np.float32))
                    feature_parts_l.append(obj_quat_at_phase_l.astype(np.float32))

            if include_contact_obs:
                # Read the env's ee_contact_sensor directly — same code path as the env
                # uses in _get_observations. force_matrix_w shape (1, n_bodies, n_filt, 3);
                # sum magnitudes across body/filter pairs to get total contact force.
                if hasattr(_direct_env, "ee_contact_sensor"):
                    fmw = _direct_env.ee_contact_sensor.data.force_matrix_w
                    isaac_force_mag = float(fmw.norm(dim=-1).sum().item())
                else:
                    isaac_force_mag = 0.0
                isaac_in_contact = 1.0 if isaac_force_mag > contact_threshold else 0.0
                isaac_contact_history = np.roll(isaac_contact_history, -1)
                isaac_contact_history[-1] = isaac_in_contact
                feature_parts_l.append(np.array([isaac_contact_history[0]], dtype=np.float32))

            curr_features = np.concatenate(feature_parts_l).astype(np.float32)
            isaac_obs_history = np.roll(isaac_obs_history, shift=-1, axis=0)
            isaac_obs_history[-1] = curr_features

            obs_parts_l = [isaac_obs_history.flatten().astype(np.float32), phase_obs_l.astype(np.float32)]
            if future_obs_steps:
                inv_cur_q = quat_inv_np(obj_quat_at_phase_l)
                fl = []
                for k in future_obs_steps:
                    fp = isaac_phase + float(k)
                    fpos = interp_np(obj_poses_ref[:, :3], fp)
                    fquat = nlerp_np(obj_poses_ref[:, 3:], fp)
                    fl.append((fpos - obj_pos_at_phase_l).astype(np.float32))
                    fl.append(quat_mul_np(fquat, inv_cur_q).astype(np.float32))
                    if include_absolute_obs:
                        fl.append(fpos.astype(np.float32))
                        fl.append(fquat.astype(np.float32))
                obs_parts_l.append(np.concatenate(fl).astype(np.float32))
            if include_prev_actions:
                obs_parts_l.append(isaac_prev_action)

            obs_l = np.concatenate(obs_parts_l)[None, ...].astype(np.float32)
            out_l = session.run([output_name], {input_name: obs_l})[0][0]
            raw_action_l = out_l[:6].astype(np.float32)

            if enable_phase_slowdown and len(out_l) >= 7:
                raw_dp = float(out_l[6])
                if getattr(isaac_cfg, "phase_mapping", "tanh") == "cubic_bidir":
                    scale_d = max(1.0 - dphase_min, isaac_cfg.dphase_max - 1.0)
                    dphase_l = float(np.clip(1.0 + scale_d * raw_dp ** 3, dphase_min, isaac_cfg.dphase_max))
                else:
                    dphase_l = float(np.clip(1.0 + (1.0 - dphase_min) * np.tanh(raw_dp), dphase_min, 1.0))
            else:
                dphase_l = 1.0

            # Compute the target once per policy step using q at the start of the step,
            # matching the env's _pre_physics_step caching behavior. Held fixed for all
            # decimation substeps.
            if action_mode == "A":
                policy_target = joints_target_at_phase_l + action_scale * raw_action_l
            elif action_mode == "B":
                policy_target = joints_at_phase_l + action_scale * raw_action_l
            elif action_mode == "C":
                policy_target = actual_q + action_scale * raw_action_l
            elif action_mode == "D":
                _eps = action_alpha_floor
                _gain = eff_alpha + _eps * (1.0 - eff_alpha)
                planner_pd = joints_target_at_phase_l - joints_at_phase_l
                policy_target = (
                    actual_q
                    + (1.0 - eff_alpha) * planner_pd
                    + _gain * action_scale * raw_action_l
                )
            else:
                raise ValueError(f"Unknown action_mode: {action_mode!r}")
            policy_target = np.clip(policy_target, joint_limits_lower, joint_limits_upper)
            target_t = torch.from_numpy(policy_target.astype(np.float32)).unsqueeze(0).to(isaac_device)
            ur5e_art.set_joint_position_target(target_t)
            ur5e_art.write_data_to_sim()

            _render_substep = not getattr(args, "headless", True)
            for _i in range(isaac_cfg.decimation):
                isaac_sim.step(render=(_render_substep and _i == isaac_cfg.decimation - 1))
                ur5e_art.update(isaac_cfg.physics_dt)
                if hasattr(_direct_env, "object"):
                    _direct_env.object.update(isaac_cfg.physics_dt)
                # Read fresh state AFTER the physics step for logging.
                substep_actual_q = ur5e_art.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
                substep_actual_obj_pos = (
                    _direct_env.object.data.root_pos_w[0] - _direct_env.scene.env_origins[0]
                ).detach().cpu().numpy().astype(np.float32)
                substep_actual_obj_quat = _direct_env.object.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
                substep_expected_q = joints_at_phase_l.copy()
                tracking_log.append({
                    "step": _isaac_step * isaac_cfg.decimation + _i,
                    "phase": float(isaac_phase),
                    "actual_q": substep_actual_q,
                    "expected_q": substep_expected_q,
                    "target_q": policy_target.copy(),
                    "actual_obj_pos": substep_actual_obj_pos,
                    "actual_obj_quat": substep_actual_obj_quat,
                })

            isaac_phase = float(min(isaac_phase + dphase_l, _max_traj_idx))
            isaac_prev_action = raw_action_l
            _isaac_step += 1
            if _isaac_step % 100 == 0:
                print(f"  step={_isaac_step}, phase={isaac_phase:.2f}, "
                      f"joint_err={np.linalg.norm(actual_q - joints_at_phase_l):.4f}")
    finally:
        save_tracking_npz()
        try:
            isaac_env.close()
        except Exception:
            pass
        if _isaacsim_app is not None:
            _isaacsim_app.close()
    sys.exit(0)


t1 = threading.Thread(target=policy_thread, daemon=True)
t2 = threading.Thread(target=control_thread, daemon=True)

t1.start()
t2.start()

try:
    # Keep main thread alive and when either thread dies we stop and exit
    while t1.is_alive() and t2.is_alive():
        time.sleep(0.02)
except KeyboardInterrupt:
    logger.warning("KeyboardInterrupt — stopping robot and saving tracking data",
                   extra={"step": -1})
finally:
    # Stop the robot first (safety priority), then save, then tear down pose streaming.
    # Each step is wrapped so a failure in one doesn't prevent the others.
    try:
        rtde_c.stopJ()
        rtde_c.servoStop()
        rtde_c.stopScript()
    except Exception as e:
        logger.error(f"Error stopping robot: {e}", extra={"step": -1})

    try:
        save_tracking_npz()
    except Exception as e:
        logger.error(f"Error saving tracking npz: {e}", extra={"step": -1})

    if pose_listener is not None:
        pose_listener.stop()
    if pose_proc is not None and pose_proc.poll() is None:
        pose_proc.terminate()
        try:
            pose_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pose_proc.kill()
