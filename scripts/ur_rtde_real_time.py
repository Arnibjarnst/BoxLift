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


class RTDEArm:
    """Per-arm RTDE connections and trajectory arrays."""
    def __init__(self, name: str, ip: str, joints_ref, joints_target, joint_vel_ref, frequency: int):
        self.name = name
        self.ip = ip
        self.joints_ref = joints_ref
        self.joints_target = joints_target
        self.joint_vel_ref = joint_vel_ref
        self.rtde_r = RTDEReceive(ip, frequency)
        self.rtde_c = RTDEControl(ip, frequency, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)

    def get_q(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualQ())

    def get_qd(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualQd())

    def stop(self):
        for fn in (self.rtde_c.servoStop, self.rtde_c.stopScript, self.rtde_c.stopJ):
            try:
                fn()
            except Exception:
                pass

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
POSE_FIRST_POSE_TIMEOUT_S = 5.0


# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True, help="Path to training run directory (e.g. logs/rsl_rl/boxpush/2026-04-08_14-42-08)")
parser.add_argument("--real_robot", action=argparse.BooleanOptionalAction)
parser.add_argument("--gain", type=float, default=None, help="servoJ gain (overrides default)")
parser.add_argument("--lookahead", type=float, default=None, help="servoJ lookahead_time (overrides default)")
parser.add_argument("--action_scale", type=float, default=None, help="Override action scale (use 0 for pure trajectory)")
parser.add_argument("--use_ref", action="store_true",
                    help="Bypass the policy and command the reference trajectory (joints_target) "
                         "directly. Use this for baseline / pure-planner runs instead of "
                         "--action_scale 0, which only zeros the policy in action modes A/B and "
                         "leaves modes C/D still tied to actual_q.")
parser.add_argument("--guard_max_obj_dist", type=float, default=None,
                    help="OOD guard: abort the run if ||box_pos - reference|| exceeds this (m). "
                         "Default: env.yaml max_obj_dist_from_traj (the training envelope).")
parser.add_argument("--guard_max_obj_angle", type=float, default=None,
                    help="OOD guard: abort if box orientation error exceeds this (rad). "
                         "Default: env.yaml max_obj_angle_from_traj, or 1.0 rad if that is "
                         "a disabled-in-training sentinel (>10).")
parser.add_argument("--no_guard", action="store_true",
                    help="Disable the OOD protective guard entirely.")
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
use_reference_obs = bool(env_cfg.get("use_reference_obs", True))
enable_phase_slowdown = bool(env_cfg.get("enable_phase_slowdown", False))
# Phase-slowdown / mode-D curriculum settings. Sentinel <0 on force_alpha disables the
# override; 1.0 ≈ "training completed warmup" which is what we usually want at deploy.
dphase_min = float(env_cfg.get("dphase_min", 0.5))
action_alpha_floor = float(env_cfg.get("action_alpha_floor", 0.1))
force_alpha = float(env_cfg.get("force_alpha", -1.0))
eff_alpha = float(force_alpha) if 0.0 <= force_alpha <= 1.0 else 1.0

# OOD protective guard thresholds. The training envelope (env's task-failure thresholds)
# is the principled OOD boundary: the policy was never trained on box states beyond it,
# so beyond it its output is meaningless. Default to the env.yaml values; the angle
# threshold is often a disabled-in-training sentinel (e.g. 10000), which is not a useful
# bound, so fall back to 1.0 rad in that case (a 180° box flip — the failure mode this
# guards against — is ~π rad, well past 1.0).
_env_guard_dist = float(env_cfg.get("max_obj_dist_from_traj", 0.2))
_env_guard_angle = float(env_cfg.get("max_obj_angle_from_traj", 1.0))
guard_max_obj_dist = (args.guard_max_obj_dist
                      if args.guard_max_obj_dist is not None else _env_guard_dist)
if args.guard_max_obj_angle is not None:
    guard_max_obj_angle = args.guard_max_obj_angle
elif _env_guard_angle <= 10.0:
    guard_max_obj_angle = _env_guard_angle
else:
    guard_max_obj_angle = 1.0
# Guard is only meaningful when we actually have a measured box pose to check.
guard_enabled = (not args.no_guard) and include_object_obs
if guard_enabled:
    print(f"[INFO] OOD guard: |box_pos-ref| > {guard_max_obj_dist:.3f} m OR "
          f"box angle error > {guard_max_obj_angle:.3f} rad → abort (servoStop via "
          f"control-thread teardown).")
else:
    print("[INFO] OOD guard disabled.")

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
robot_name = "real" if args.real_robot else "isaac" if args.isaacsim else "ursim"
run_tag = (
    f"{date_t}_gain{_gain}_la{_lookahead}_{robot_name}"
    + (f"_as{action_scale}" if args.action_scale is not None else "")
    + ("_useref" if args.use_ref else "")
)

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
    f"use_ref: {args.use_ref}, "
    f"obs_history_steps: {obs_history_steps}, include_object_obs: {include_object_obs}, "
    f"future_obs_steps: {future_obs_steps}, include_prev_actions: {include_prev_actions}, "
    f"include_absolute_obs: {include_absolute_obs}, "
    f"enable_phase_slowdown: {enable_phase_slowdown}, dphase_min: {dphase_min}, "
    f"eff_alpha: {eff_alpha:.3f} (force_alpha={force_alpha}, "
    f"action_alpha_floor={action_alpha_floor}), "
    f"trajectory: {reference_trajectory_path}",
    extra={"step": -1},
)
if args.use_ref:
    logger.info(
        "[--use_ref] Policy bypassed: commanding joints_target directly. "
        "ONNX inference still runs (output discarded) so obs construction "
        "stays exercised, but action_scale and action_mode have no effect.",
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

# Camera-world -> IsaacLab sim / UR-base frame is a 180° rotation about z (no
# translation). Pose-estimation listener output is in camera-world; the policy was
# trained against sim coordinates. Apply this to the pose before feeding it to the
# policy and before logging, so saved obj poses are directly comparable to the
# reference trajectory in `obj_poses`.
REAL_TO_SIM_R = np.array(
    [[-1.0, 0.0, 0.0],
     [ 0.0,-1.0, 0.0],
     [ 0.0, 0.0, 1.0]], dtype=np.float32,
)
# 180° about z in (w, x, y, z); used as q_R * q_in_world = q_in_sim.
REAL_TO_SIM_Q_WXYZ = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def real_to_sim_pose(pose):
    """Map a 7-vec [pos, quat_wxyz] from camera-world to sim / UR-base frame.
    Hemisphere-corrects the result to w >= 0 so the network sees the same sign
    convention it was trained on. Returns None unchanged so call sites can stay
    nan-safe."""
    if pose is None:
        return None
    pose = np.asarray(pose, dtype=np.float32)
    pos = REAL_TO_SIM_R @ pose[:3]
    quat = quat_mul_np(REAL_TO_SIM_Q_WXYZ, pose[3:])
    if quat[0] < 0.0:
        quat = -quat
    return np.concatenate([pos, quat]).astype(np.float32)


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
    robot_ip_l = "192.168.1.100"
    if dual_arm:
        robot_ip_l = "192.168.1.33"
        robot_ip_r = "192.168.1.66"
elif os_used == "win32":
    robot_ip_l = "172.29.144.1"
    robot_ip_r = "172.29.144.2"
else:
    robot_ip_l = "192.168.56.101"
    robot_ip_r = "192.168.56.102"

rtde_frequency = 500
dt = 1 / rtde_frequency
policy_decimation = 10
max_steps = (len(joints) - 2) * policy_decimation
max_wallclock_steps = max_steps * 2

# Global arms list (1 or 2 RTDEArm objects). All non-IsaacSim control goes through this.
arms: list[RTDEArm] = []

if not args.isaacsim:
    logger.info(f"Connecting to left arm at {robot_ip_l}", extra={"step": -1})
    arm_l = RTDEArm("left", robot_ip_l, joints, joints_target, joint_vel_ref, rtde_frequency)
    arms.append(arm_l)
    logger.info("Left arm connected", extra={"step": -1})

    if dual_arm:
        logger.info(f"Connecting to right arm at {robot_ip_r}", extra={"step": -1})
        arm_r = RTDEArm("right", robot_ip_r, joints_r, traj["joints_target_r"], joint_vel_r, rtde_frequency)
        arms.append(arm_r)
        logger.info("Right arm connected", extra={"step": -1})

    # Convenience aliases kept for the IsaacSim-legacy path and log() calls.
    rtde_r = arms[0].rtde_r
    rtde_c = arms[0].rtde_c
else:
    rtde_r = None
    rtde_c = None

logger.info("Connection established", extra={"step": -1})

if not args.isaacsim:
    logger.info("Resetting robot state", extra={"step": -1})
    for arm in arms:
        arm.rtde_c.reuploadScript()

np.set_printoptions(suppress=True, precision=8)

# Per-timestep tracking data for analysis. Two parallel logs:
#   - tracking_log: one entry per 500 Hz control tick (RTDE-side state, includes
#     actual_q, actual_qd, target_q, box pose snapshot, tcp_force).
#   - policy_log: one entry per 50 Hz policy iteration (obs vector fed to ONNX,
#     raw policy output, dphase). Saved with `policy_` prefix so users don't
#     confuse the two rates during analysis. Each list is appended from a single
#     thread (control / policy respectively) so no extra lock is needed beyond
#     CPython's GIL — same pattern save_tracking_npz already uses.
tracking_log = []
policy_log = []

# Shared between the 50 Hz policy thread and the 500 Hz control thread. The policy
# thread caches the latest box pose here so log() can reuse it instead of polling
# the listener on every servoJ tick. NaN-filled until the first policy iteration.
data_lock = threading.Lock()
current_actual_obj_pos = np.full(3, np.nan, dtype=np.float32)
current_actual_obj_quat = np.full(4, np.nan, dtype=np.float32)


print("hieohfioeh")

def save_tracking_npz():
    """Snapshot tracking_log + policy_log and write the .npz. Safe to call multiple
    times — the write is idempotent (overwrite). Used by both the normal-exit path
    and the KeyboardInterrupt handler in main.

    Schema (different first-dim lengths are fine in np.savez; the prefix tells the
    consumer which rate the array is at):
      - 500 Hz keys (no prefix): steps, phase, actual_q, actual_qd, expected_q,
        target_q, actual_obj_pos, actual_obj_quat, tcp_force, target_moment,
        current_as_torque.
      - 50 Hz keys (`policy_` prefix): policy_iter, policy_phase, policy_obs,
        policy_raw_output (6 or 7 dims), policy_dphase.
      - Scalar metadata: src_dt, gain, lookahead_time, action_scale,
        policy_decimation (so consumers can map iter → control-step index).
    """
    # Shallow snapshots — CPython list.append + list(...) are GIL-atomic, so we
    # don't need the data_lock here.
    snapshot = list(tracking_log)
    policy_snapshot = list(policy_log)
    if not snapshot and not policy_snapshot:
        return
    tracking_npz_path = os.path.join(log_dir, f"{run_tag}.npz")

    arrays = {}
    if snapshot:
        arrays.update(
            steps=np.array([d["step"] for d in snapshot]),
            phase=np.array([d["phase"] for d in snapshot]),
            actual_q=np.array([d["actual_q"] for d in snapshot]),
            actual_qd=np.array([
                d.get("actual_qd", np.full(6, np.nan, dtype=np.float32)) for d in snapshot
            ]),
            expected_q=np.array([d["expected_q"] for d in snapshot]),
            target_q=np.array([d["target_q"] for d in snapshot]),
            actual_obj_pos=np.array([d["actual_obj_pos"] for d in snapshot]),
            actual_obj_quat=np.array([d["actual_obj_quat"] for d in snapshot]),
            # External TCP wrench [Fx, Fy, Fz, Tx, Ty, Tz] in UR base frame.
            # NaN-filled for ticks that didn't capture it (e.g. IsaacSim backend).
            tcp_force=np.array([
                d.get("tcp_force", np.full(6, np.nan, dtype=np.float32)) for d in snapshot
            ]),
            # Controller's commanded per-joint torque [Nm] and the motor-current-
            # derived torque, both from the receive interface. NaN-filled when
            # unavailable.
            target_moment=np.array([
                d.get("target_moment", np.full(6, np.nan, dtype=np.float32)) for d in snapshot
            ]),
            current_as_torque=np.array([
                d.get("current_as_torque", np.full(6, np.nan, dtype=np.float32)) for d in snapshot
            ]),
        )
    if policy_snapshot:
        arrays.update(
            policy_iter=np.array([d["iter"] for d in policy_snapshot]),
            policy_phase=np.array([d["phase"] for d in policy_snapshot]),
            # Stack obs vectors into a (N_policy, obs_dim) array; all rows share the
            # same dim since obs construction is deterministic for a given env.yaml.
            policy_obs=np.stack([d["obs"] for d in policy_snapshot]),
            policy_raw_output=np.stack([d["raw_output"] for d in policy_snapshot]),
            policy_dphase=np.array([d["dphase"] for d in policy_snapshot]),
        )

    # Arm base poses from the reference trajectory — required by visualize_traj.py to
    # place the robot bases in IsaacSim. Save whichever keys exist in the traj file.
    arm_pose_arrays = {}
    if "arm_l_pose" in traj:
        arm_pose_arrays["arm_l_pose"] = np.asarray(traj["arm_l_pose"])
    if "arm_r_pose" in traj:
        arm_pose_arrays["arm_r_pose"] = np.asarray(traj["arm_r_pose"])
    if "arm_pose" in traj:
        arm_pose_arrays["arm_pose"] = np.asarray(traj["arm_pose"])

    np.savez(
        tracking_npz_path,
        src_dt=np.float64(dt),
        policy_decimation=np.int32(policy_decimation),
        dual_arm=np.bool_(dual_arm),
        gain=gain,
        lookahead_time=lookahead_time,
        action_scale=action_scale,
        **arm_pose_arrays,
        **arrays,
    )
    logger.info(
        f"Tracking data saved to {tracking_npz_path} "
        f"({len(snapshot)} ctrl steps, {len(policy_snapshot)} policy steps)",
        extra={"step": -1},
    )

def log(i, target_q, phase):
    # Concatenate state from all arms (6 dims each → 6 or 12 dims total).
    actual_q  = np.concatenate([arm.get_q()  for arm in arms])
    actual_qd = np.concatenate([arm.get_qd() for arm in arms])

    tcp_force = np.concatenate([
        np.array(arm.rtde_r.getActualTCPForce(), dtype=np.float32) for arm in arms
    ])
    target_moment = np.concatenate([
        np.array(arm.rtde_r.getTargetMoment(), dtype=np.float32) for arm in arms
    ])
    try:
        current_as_torque = np.concatenate([
            np.array(arm.rtde_r.getActualCurrentAsTorque(), dtype=np.float32) for arm in arms
        ])
    except Exception:
        current_as_torque = np.full(len(arms) * 6, np.nan, dtype=np.float32)

    # Reference trajectory at the current phase (concatenated across arms).
    expected_q = np.concatenate([interp_np(arm.joints_ref, phase) for arm in arms])

    with data_lock:
        actual_obj_pos  = current_actual_obj_pos.copy()
        actual_obj_quat = current_actual_obj_quat.copy()

    if i >= 0:
        tracking_log.append({
            "step": i,
            "phase": float(phase),
            "actual_q":  actual_q.copy(),
            "actual_qd": actual_qd.copy(),
            "expected_q": expected_q.copy(),
            "target_q": np.array(target_q).copy(),
            "actual_obj_pos":  actual_obj_pos.copy(),
            "actual_obj_quat": actual_obj_quat.copy(),
            "tcp_force":        tcp_force.copy(),
            "target_moment":    target_moment.copy(),
            "current_as_torque": current_as_torque.copy(),
        })

    tracking_error = np.linalg.norm(actual_q - expected_q)

    for arm in arms:
        safety_mode = arm.rtde_r.getSafetyMode()
        robot_mode  = arm.rtde_r.getRobotMode()
        if safety_mode != 1:
            logger.error(
                f"[{arm.name}] Robot left NORMAL safety mode. safety_mode={safety_mode}",
                extra={"step": i},
            )
            raise RuntimeError("Robot safety event")
        if robot_mode != 7:
            logger.error(
                f"[{arm.name}] Robot not running. robot_mode={robot_mode}",
                extra={"step": i},
            )
            raise RuntimeError("Robot stopped")

    if i >= 0 and tracking_error > max_tracking_error:
        logger.error(
            f"Tracking error {tracking_error:.4f} exceeds limit {max_tracking_error}",
            extra={"step": i},
        )
        raise RuntimeError("Tracking error too large")

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
        for arm in arms:
            logger.info(f"[{arm.name}] moveJ to initial position {arm.joints_ref[0]}", extra={"step": -1})
        # Command all arms to start simultaneously (non-blocking async=True), then wait.
        for arm in arms:
            last_t = time.perf_counter()
            success = arm.rtde_c.moveJ(arm.joints_ref[0], 0.5, 1.0, True)
            while not success:
                if time.perf_counter() - last_t > 5:
                    raise TimeoutError(f"[{arm.name}] Timed out moving to initial position")
                success = arm.rtde_c.moveJ(arm.joints_ref[0], 0.5, 1.0, True)
                time.sleep(dt)

        # Wait until all arms are steady.
        while any(not arm.rtde_c.isSteady() for arm in arms):
            log(-1, np.concatenate([arm.joints_ref[0] for arm in arms]), 0.0)
            time.sleep(dt)

        logger.info("moveJ finished to initial positions", extra={"step": -1})
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user", extra={"step": -1})
    finally:
        for arm in arms:
            try:
                arm.rtde_c.stopJ()
            except Exception:
                pass

if not args.isaacsim:
    current_target_q = np.concatenate([arm.get_q() for arm in arms])
else:
    current_target_q = np.asarray(joints[0], dtype=np.float64)
previous_target_q = np.copy(current_target_q)

# Capture a baseline TCP wrench while the robot is at rest at the trajectory start
# pose. The raw getActualTCPForce() has a static gripper-gravity bias that varies with
# arm pose but is roughly constant at this initial pose; subtracting this baseline
# leaves ||delta|| as a usable proxy for external contact force.
# Only meaningful when include_contact_obs is set and RTDE is connected.
# Per-arm TCP force baselines (6-dim each). Indexed as tcp_force_baselines[arm_idx].
tcp_force_baselines = [np.zeros(6, dtype=np.float64) for _ in arms]
if include_contact_obs and not args.isaacsim and arms:
    for idx, arm in enumerate(arms):
        logger.info(f"[{arm.name}] Sampling TCP force baseline (50 readings)...", extra={"step": -1})
        samples = np.array([arm.rtde_r.getActualTCPForce() for _ in range(50)])
        tcp_force_baselines[idx] = samples.mean(axis=0)
        logger.info(
            f"[{arm.name}] TCP force baseline = {tcp_force_baselines[idx].round(3).tolist()}",
            extra={"step": -1},
        )
# Single-arm legacy alias.
tcp_force_baseline = tcp_force_baselines[0] if tcp_force_baselines else np.zeros(6, dtype=np.float64)
# Float trajectory phase (matches the policy thread's `phase` variable). Updated under
# data_lock by the policy thread; the control thread blends previous→current the same
# way as target_q so we can compute the exact expected reference at any servo tick.
current_phase = 0.0
previous_phase = 0.0
run_policy_event = threading.Event()  # Trigger for the policy
new_data_event = threading.Event()
# OOD protective guard: the policy thread sets ood_reason then ood_event when the
# measured box pose leaves the training envelope. The control thread checks the event
# each tick and raises, so its except/finally (servoStop + stopScript) and the main
# teardown (stopJ) run — a controlled halt with no further policy commands.
ood_event = threading.Event()
ood_reason = ""

# Wait for the first box pose to arrive before starting threads. With
# include_object_obs the tracker is a hard requirement (the policy consumes it),
# so missing/timed-out poses abort. Without it we still want box poses for the
# npz log but can keep going if the tracker is dead — log() will NaN-fill.
if pose_listener is not None:
    logger.info(
        f"Waiting up to {POSE_FIRST_POSE_TIMEOUT_S:.0f}s for first box pose...",
        extra={"step": -1},
    )
    _start = time.perf_counter()
    _first_pose = None
    _failure_reason = None
    while True:
        _first_pose = pose_listener.get_pose(BOX_BOARD_ID)
        if _first_pose is not None:
            break
        if pose_proc is not None and pose_proc.poll() is not None:
            _failure_reason = (
                f"Pose-estimation subprocess exited early "
                f"(returncode={pose_proc.returncode})"
            )
            break
        if time.perf_counter() - _start > POSE_FIRST_POSE_TIMEOUT_S:
            _failure_reason = (
                f"Timed out after {POSE_FIRST_POSE_TIMEOUT_S:.0f}s waiting for "
                f"first box pose"
            )
            break
        time.sleep(0.1)

    if _first_pose is not None:
        logger.info(f"First box pose received: {_first_pose}", extra={"step": -1})
    elif include_object_obs and args.real_robot:
        # Real-robot path: the tracker is the only source of box pose, so we can't
        # continue without it.
        logger.error(
            f"{_failure_reason}; include_object_obs=True on the real robot requires a "
            f"live tracker, aborting.",
            extra={"step": -1},
        )
        # Tear down anything we already brought up so we don't leave the RTDE
        # script or pose subprocess hanging.
        try:
            pose_listener.stop()
        except Exception:
            pass
        if pose_proc is not None and pose_proc.poll() is None:
            try:
                pose_proc.terminate()
                pose_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pose_proc.kill()
            except Exception:
                pass
        if rtde_c is not None:
            try:
                rtde_c.stopJ()
                rtde_c.stopScript()
            except Exception:
                pass
        sys.exit(1)
    elif include_object_obs:
        # URSim (or IsaacSim if it reached this branch) — fall back to the "no tracker"
        # path that policy_thread already handles: rel = zeros/identity, abs = reference
        # pose at current phase. Clear pose_listener / pose_proc so the obs path takes
        # that fallback branch and we don't keep a dead subprocess around.
        logger.warning(
            f"{_failure_reason}; include_object_obs=True but not on a real robot — "
            f"falling back to perfect tracking (rel=0/identity, abs=reference at phase).",
            extra={"step": -1},
        )
        try:
            pose_listener.stop()
        except Exception:
            pass
        pose_listener = None
        if pose_proc is not None and pose_proc.poll() is None:
            try:
                pose_proc.terminate()
                pose_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pose_proc.kill()
            except Exception:
                pass
            pose_proc = None
    else:
        logger.warning(
            f"{_failure_reason}; include_object_obs=False so continuing without it "
            f"(box pose will be NaN-filled in the npz log).",
            extra={"step": -1},
        )


def policy_thread():
    global current_target_q, previous_target_q, current_phase, previous_phase
    global current_actual_obj_pos, current_actual_obj_quat
    global ood_reason

    phase = 0.0

    # Compute per-step obs dim to match the env's __post_init__ logic.
    base_pose = 7  # 3 pos + 4 quat
    n_joints = len(arms) * 6  # 6 or 12
    if dual_arm:
        if use_reference_obs:
            per_step_features = (n_joints + n_joints + base_pose   # rel_q + rel_qd + rel_obj
                                 + n_joints + n_joints + base_pose  # abs_q + abs_qd + abs_obj
                                 + 2)                                # contact (one per arm)
        else:
            per_step_features = (n_joints + n_joints + base_pose   # abs_q + abs_qd + abs_obj
                                 + 2)                                # contact
    else:
        per_step_features = 12 + (7 if include_object_obs else 0)
        if include_absolute_obs:
            per_step_features *= 2
        if include_contact_obs:
            per_step_features += 1

    obs_history = np.zeros((obs_history_steps, per_step_features), dtype=np.float32)
    prev_action = np.zeros(n_joints, dtype=np.float32)
    max_traj_idx = len(arms[0].joints_ref) - 1
    phase_history = np.zeros(obs_obj_delay_steps + 1, dtype=np.float32)
    policy_iter = 0

    while True:
        run_policy_event.wait()
        run_policy_event.clear()

        # Read all arm states simultaneously (serial reads are fast local RTDE packet reads).
        qs  = [arm.get_q()  for arm in arms]
        qds = [arm.get_qd() for arm in arms]

        traj_idx = min(int(np.floor(phase)), max_traj_idx)
        phase_obs = np.array([phase / max_traj_idx])

        actual_obj_pos = None
        actual_obj_quat = None
        if pose_listener is not None:
            box_pose = real_to_sim_pose(pose_listener.get_pose(BOX_BOARD_ID))
            if box_pose is not None:
                actual_obj_pos  = box_pose[:3].astype(np.float32)
                actual_obj_quat = box_pose[3:].astype(np.float32)

        phase_history = np.roll(phase_history, -1)
        phase_history[-1] = phase
        delayed_phase = float(phase_history[0])

        obj_pos_at_phase   = interp_np(obj_poses_ref[:, :3], phase)
        obj_quat_at_phase  = nlerp_np(obj_poses_ref[:, 3:],  phase)
        obj_pos_at_delayed = interp_np(obj_poses_ref[:, :3], delayed_phase)
        obj_quat_at_delayed = nlerp_np(obj_poses_ref[:, 3:], delayed_phase)

        # Per-arm reference interpolations.
        ref_q      = [interp_np(arm.joints_ref,    phase) for arm in arms]
        ref_target = [interp_np(arm.joints_target, phase) for arm in arms]
        ref_qd     = [interp_np(arm.joint_vel_ref, phase) for arm in arms]

        # Build object obs (shared across single/dual arm).
        use_actual_obj = actual_obj_pos is not None
        if use_actual_obj:
            rel_obj_pos  = actual_obj_pos - obj_pos_at_delayed
            rel_obj_quat = quat_mul_np(obj_quat_at_delayed, quat_inv_np(actual_obj_quat))
            abs_obj_pos  = actual_obj_pos
            abs_obj_quat = actual_obj_quat
            if guard_enabled and not ood_event.is_set():
                _pos_err = float(np.linalg.norm(rel_obj_pos))
                _w = min(1.0, abs(float(rel_obj_quat[0])))
                _ang_err = 2.0 * float(np.arccos(_w))
                if _pos_err > guard_max_obj_dist or _ang_err > guard_max_obj_angle:
                    ood_reason = (
                        f"box OOD: pos_err={_pos_err:.4f}m (limit {guard_max_obj_dist:.3f}), "
                        f"ang_err={_ang_err:.4f}rad (limit {guard_max_obj_angle:.3f}) "
                        f"at phase={phase:.1f}"
                    )
                    logger.error(ood_reason, extra={"step": traj_idx})
                    ood_event.set()
        else:
            if include_object_obs:
                logger.warning("No box pose available; falling back to zeros", extra={"step": traj_idx})
            rel_obj_pos  = np.zeros(3, dtype=np.float32)
            rel_obj_quat = IDENTITY_QUAT_WXYZ
            abs_obj_pos  = obj_pos_at_phase
            abs_obj_quat = obj_quat_at_phase

        if dual_arm:
            # Contact bools: one per arm (2 total).
            contact_dims = []
            if include_contact_obs:
                for idx, arm in enumerate(arms):
                    wrench = np.array(arm.rtde_r.getActualTCPForce())
                    delta  = wrench[:3] - tcp_force_baselines[idx][:3]
                    contact_dims.append(1.0 if np.linalg.norm(delta) > contact_threshold else 0.0)
            else:
                contact_dims = [0.0, 0.0]
            contact_arr = np.array(contact_dims, dtype=np.float32)

            abs_q  = np.concatenate([q.astype(np.float32)  for q  in qs])
            abs_qd = np.concatenate([qd.astype(np.float32) for qd in qds])
            rel_q  = np.concatenate([(qs[i]  - ref_q[i]).astype(np.float32)  for i in range(len(arms))])
            rel_qd = np.concatenate([(qds[i] - ref_qd[i]).astype(np.float32) for i in range(len(arms))])

            if use_reference_obs:
                current_features = np.concatenate([
                    rel_q, rel_qd, rel_obj_pos, rel_obj_quat,   # relative state
                    abs_q, abs_qd, abs_obj_pos, abs_obj_quat,   # absolute state
                    contact_arr,
                ]).astype(np.float32)
            else:
                current_features = np.concatenate([
                    abs_q, abs_qd, abs_obj_pos, abs_obj_quat,
                    contact_arr,
                ]).astype(np.float32)
        else:
            actual_q  = qs[0]
            actual_qd = qds[0]
            relative_q  = (actual_q  - ref_q[0]).astype(np.float32)
            relative_qd = (actual_qd - ref_qd[0]).astype(np.float32)
            feature_parts = [relative_q, relative_qd]
            if include_object_obs:
                feature_parts.append(rel_obj_pos.astype(np.float32))
                feature_parts.append(rel_obj_quat.astype(np.float32))
            if include_absolute_obs:
                feature_parts.append(actual_q.astype(np.float32))
                feature_parts.append(actual_qd.astype(np.float32))
                if include_object_obs:
                    feature_parts.append(abs_obj_pos.astype(np.float32))
                    feature_parts.append(abs_obj_quat.astype(np.float32))
            if include_contact_obs:
                wrench = np.array(arms[0].rtde_r.getActualTCPForce())
                delta  = wrench[:3] - tcp_force_baselines[0][:3]
                force_mag = float(np.linalg.norm(delta))
                in_contact = 1.0 if force_mag > contact_threshold else 0.0
                feature_parts.append(np.array([in_contact], dtype=np.float32))
                logger.debug(
                    f"contact: force_mag={force_mag:.3f} N (thr={contact_threshold:.2f}), "
                    f"in_contact={in_contact}",
                    extra={"step": traj_idx},
                )
            current_features = np.concatenate(feature_parts).astype(np.float32)

        obs_history = np.roll(obs_history, shift=-1, axis=0)
        obs_history[-1] = current_features

        # Phase dim present only in reference mode (matches env's __post_init__ logic).
        obs_parts = [obs_history.flatten().astype(np.float32)]
        if use_reference_obs or not dual_arm:
            obs_parts.append(phase_obs.astype(np.float32))

        if future_obs_steps:
            inv_cur_quat = quat_inv_np(obj_quat_at_phase)
            futures = []
            for k in future_obs_steps:
                fut_phase = phase + float(k)
                fut_pos   = interp_np(obj_poses_ref[:, :3], fut_phase)
                fut_quat  = nlerp_np(obj_poses_ref[:, 3:],  fut_phase)
                futures.append((fut_pos - obj_pos_at_phase).astype(np.float32))
                futures.append(quat_mul_np(fut_quat, inv_cur_quat).astype(np.float32))
                if include_absolute_obs or dual_arm:
                    futures.append(fut_pos.astype(np.float32))
                    futures.append(fut_quat.astype(np.float32))
            obs_parts.append(np.concatenate(futures).astype(np.float32))

        # New dual-arm boxlift env always appends prev_actions (hardcoded + 12 in actor_dim).
        # Single-arm legacy envs use the include_prev_actions flag from env.yaml.
        if dual_arm or include_prev_actions:
            obs_parts.append(prev_action)

        obs    = np.concatenate(obs_parts)[None, ...].astype(np.float32)
        output = session.run([output_name], {input_name: obs})[0][0]

        raw_action = np.clip(output[:n_joints], -2.0, 2.0).astype(np.float32)  # clamp before applying; prevents runaway via prev_action feedback

        if args.use_ref:
            dphase = 1.0
        elif enable_phase_slowdown and len(output) > n_joints:
            dphase = 1.0 + (1.0 - dphase_min) * float(np.tanh(output[n_joints]))
            dphase = float(np.clip(dphase, dphase_min, 1.0))
        else:
            dphase = 1.0

        # Build per-arm targets and concatenate into a single (n_joints,) vector.
        new_joint_targets_list = []
        for i, arm in enumerate(arms):
            a_slice = slice(i * 6, (i + 1) * 6)
            arm_action = raw_action[a_slice]
            q_i        = qs[i]
            if args.use_ref:
                t_i = ref_target[i]
            elif action_mode == "A":
                t_i = ref_target[i] + action_scale * arm_action
            elif action_mode == "B":
                t_i = ref_q[i] + action_scale * arm_action
            elif action_mode == "C":
                t_i = q_i + action_scale * arm_action
            elif action_mode == "D":
                eps  = action_alpha_floor
                gain = eff_alpha + eps * (1.0 - eff_alpha)
                pd   = ref_target[i] - ref_q[i]
                t_i  = q_i + (1.0 - eff_alpha) * pd + gain * action_scale * arm_action
            elif action_mode == "BC":
                # B→C curriculum blend: (1-α)·q_ref + α·q_curr + scale·a
                # At α=0 → mode B (residual on reference), at α=1 → mode C (residual on current).
                # eff_alpha=1.0 when force_alpha<0 (training converged), collapsing to mode C.
                t_i = (1.0 - eff_alpha) * ref_q[i] + eff_alpha * q_i + action_scale * arm_action
            else:
                raise ValueError(f"Unknown action_mode: {action_mode!r}")
            t_i = np.clip(t_i, joint_limits_lower, joint_limits_upper)
            new_joint_targets_list.append(t_i)
        new_joint_targets = np.concatenate(new_joint_targets_list)

        with data_lock:
            current_target_q = new_joint_targets
            current_phase    = phase
            if actual_obj_pos is not None:
                current_actual_obj_pos  = actual_obj_pos.copy()
                current_actual_obj_quat = actual_obj_quat.copy()
            else:
                current_actual_obj_pos  = np.full(3, np.nan, dtype=np.float32)
                current_actual_obj_quat = np.full(4, np.nan, dtype=np.float32)
            new_data_event.set()

        prev_action = raw_action

        policy_log.append({
            "iter":       policy_iter,
            "phase":      float(phase),
            "obs":        obs[0].copy(),
            "raw_output": np.array(output, dtype=np.float32),
            "dphase":     float(dphase),
        })
        policy_iter += 1

        phase = min(phase + dphase, float(max_traj_idx))


def control_thread():
    step_counter = 0
    try:
        # Wait for first target
        run_policy_event.set()
        while not new_data_event.is_set():
            pass
        new_data_event.clear()

        while True:
            t_start = rtde_c.initPeriod()
            t1 = time.perf_counter()

            # OOD guard tripped by the policy thread — raise before sending any further
            # command. except→raise runs this thread's finally (servoStop, stopScript);
            # the thread dying drops the main keep-alive loop, whose finally runs stopJ.
            if ood_event.is_set():
                raise RuntimeError(f"OOD guard breached: {ood_reason}")

            if step_counter % policy_decimation == 0:
                run_policy_event.set()

            if new_data_event.is_set():
                new_data_event.clear()

            with data_lock:
                target_q    = current_target_q.copy()
                target_phase = current_phase

                for i, arm in enumerate(arms):
                    arm.rtde_c.servoJ(
                        target_q[i * 6:(i + 1) * 6],
                        velocity, acceleration, dt, lookahead_time, gain,
                    )
            log(step_counter, target_q, target_phase)

            step_counter += 1

            if target_phase * policy_decimation > max_steps:
                break

            if step_counter > max_wallclock_steps:
                break

            # Should be at the end to ensure correct timing
            t2 = time.perf_counter()
            if (t2 - t1) > (1 / rtde_frequency):
                raise Exception(f"too slow: {t2-t1}")
            rtde_c.waitPeriod(t_start)

    except Exception as e:

        logger.error(
            f"servoJ exception: {str(e)}",
            extra={"step": step_counter},
        )

        raise

    finally:
        logger.info("Stopping servo", extra={"step": -1})
        for arm in arms:
            arm.stop()
        logger.info(f"Execution finished. Log written to {log_path}", extra={"step": -1})


if args.isaacsim:
    # =====================================================================
    # IsaacSim backend: single-threaded loop that reuses the same obs
    # construction logic as policy_thread but reads/writes IsaacSim state.
    # =====================================================================
    import gymnasium as gym
    import torch
    import BoxLift.tasks  # noqa: F401, registers env
    from BoxLift.tasks.direct.boxhinge.boxhinge_env_cfg import BoxhingeEnvCfg

    kp_override = 30.0
    kd_override = None

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
    
    if kp_override is not None:
        setattr(isaac_cfg, "kp", kp_override)
    if kd_override is not None:
        setattr(isaac_cfg, "kd", kd_override)

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

            if args.use_ref:
                dphase_l = 1.0
            elif enable_phase_slowdown and len(out_l) >= 7:
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
            if args.use_ref:
                policy_target = joints_target_at_phase_l
            elif action_mode == "A":
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
            elif action_mode == "BC":
                policy_target = (1.0 - eff_alpha) * joints_at_phase_l + eff_alpha * actual_q + action_scale * raw_action_l
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
                    # No equivalent of getActualTCPForce in this IsaacSim path.
                    "tcp_force": np.full(6, np.nan, dtype=np.float32),
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
    for arm in arms:
        try:
            arm.stop()
        except Exception as e:
            logger.error(f"Error stopping {arm.name} arm: {e}", extra={"step": -1})

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
