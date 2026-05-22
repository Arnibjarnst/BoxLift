"""Per-env success analysis + initial→final error scatter for multi-env rollouts.

For each env in a record.py rollout we compute:
  - success / failure (based on early-termination AND terminal-task-reward thresholds)
  - cube pose at first frame and at the last VALID frame (= done_step - 1, since the
    death-step state is NaN'd by record.py)
  - reference cube pose at the same trajectory phases (from the .npz traj file)
  - three error metrics, computed both initially and finally:
      * pos  — ||p_actual − p_ref||                                (m)
      * quat — smallest rotation angle between the two quaternions  (rad)
      * pose — mean Euclidean distance over the 8 cube corners      (m)

Typical use from the notebook:

    from scripts.rollout_summary import compute_rollout_summary, print_summary, plot_initial_vs_final
    s = compute_rollout_summary("logs/.../rollout/output.npz", min_task_reward=0.05)
    print_summary(s, title="some run")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plot_initial_vs_final(s, mode="pos",  ax=axes[0])
    plot_initial_vs_final(s, mode="quat", ax=axes[1])
    plot_initial_vs_final(s, mode="pose", ax=axes[2])
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import yaml


# ---------- small math helpers (wxyz quaternion convention) ----------

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """wxyz quat → 3×3 rotation matrix. q: (..., 4)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=-2)


def _quat_angle_diff(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Smallest angle between two unit quaternions (rad). q_*: (..., 4) wxyz."""
    dot = np.clip(np.abs(np.sum(q_a * q_b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _corner_distance(p_a, q_a, p_b, q_b, box_dims) -> np.ndarray:
    """Mean Euclidean distance over the 8 cube corners after applying each pose.
    p_*: (..., 3), q_*: (..., 4) wxyz, box_dims: 3-tuple [Lx, Ly, Lz]."""
    lx, ly, lz = (float(x) for x in box_dims)
    corners_local = np.array([
        [+lx / 2, +ly / 2, +lz / 2], [+lx / 2, +ly / 2, -lz / 2],
        [+lx / 2, -ly / 2, +lz / 2], [+lx / 2, -ly / 2, -lz / 2],
        [-lx / 2, +ly / 2, +lz / 2], [-lx / 2, +ly / 2, -lz / 2],
        [-lx / 2, -ly / 2, +lz / 2], [-lx / 2, -ly / 2, -lz / 2],
    ])  # (8, 3)
    R_a = _quat_to_rotmat(q_a)   # (..., 3, 3)
    R_b = _quat_to_rotmat(q_b)
    # einsum: rotate each local corner by R, then add center pose
    corners_a = p_a[..., None, :] + np.einsum("...ij,kj->...ki", R_a, corners_local)
    corners_b = p_b[..., None, :] + np.einsum("...ij,kj->...ki", R_b, corners_local)
    return np.linalg.norm(corners_a - corners_b, axis=-1).mean(axis=-1)


# ---------- I/O helper to discover the trajectory + box dims for a rollout ----------

def _load_traj_for_rollout(rollout_path: Path) -> tuple[dict, tuple[float, float, float]]:
    """Mirror the rest of the notebook: pull `trajectory_path` from <run>/params/env.yaml,
    resolve relative to repo root, and load. Also derive box dims (object_dims from the
    traj if present, else cube_cfg.spawn.size from env.yaml)."""
    run_dir = rollout_path.parent.parent
    env_yaml = run_dir / "params" / "env.yaml"
    with open(env_yaml, "r") as f:
        env_cfg = yaml.unsafe_load(f)
    traj_path = env_cfg["trajectory_path"]
    if not Path(traj_path).is_absolute():
        # rollout sits at <repo>/logs/rsl_rl/<exp>/<run>/rollout/output.npz → 4 parents up.
        traj_path = run_dir.parent.parent.parent.parent / traj_path
    traj = np.load(traj_path)
    if "object_dims" in traj.files:
        box_dims = tuple(float(x) for x in traj["object_dims"])
    else:
        box_dims = tuple(float(x) for x in env_cfg["cube_cfg"]["spawn"]["size"])
    return traj, box_dims


# ---------- main preprocessing ----------

def compute_rollout_summary(
    rollout_path,
    *,
    min_task_reward: float = 0.05,
    early_step_threshold: int | None = None,
    task_reward_key: str = "extras_per_env__Rewards_task__total",
    box_dims: tuple[float, float, float] | None = None,
    traj=None,
) -> dict:
    """Compute per-env summary metrics for a multi-env rollout NPZ.

    Args:
        rollout_path: path to record.py output.npz (must be multi-env: has `num_envs`).
        min_task_reward: terminal task reward below which an env is counted as a failure.
            Default 0.05 matches "the policy basically achieved nothing at the end."
        early_step_threshold: done_step strictly LESS than this counts as early
            termination. None → T - 5 (anything not at trajectory-end timeout).
        task_reward_key: per-env extras key holding the task reward. Defaults to the
            new `extras_per_env__Rewards_task__total` field (record.py with
            `emit_per_env_extras=True`). Override if you want a different aggregate.
        box_dims, traj: usually auto-discovered via env.yaml; override if you want.

    Returns:
        dict with per-env arrays (init/final pose, errors, success, reason) and aggregate
        config so `print_summary` / `plot_initial_vs_final` can be called on it.
    """
    rp = Path(rollout_path)
    d = np.load(rp)
    if "num_envs" not in d.files:
        raise ValueError(f"{rp} is not a multi-env rollout (no `num_envs` field).")
    N = int(d["num_envs"])
    T = len(d["steps"])

    if traj is None:
        traj, _box_dims_default = _load_traj_for_rollout(rp)
        if box_dims is None:
            box_dims = _box_dims_default
    elif box_dims is None:
        box_dims = (0.235, 0.34, 0.27)   # last-ditch fallback (boxhinge-ish)

    done_step = d["done_step"].astype(np.int64)         # (N,)
    early_thresh = (T - 5) if early_step_threshold is None else int(early_step_threshold)

    # --- terminal task reward per env ---
    # `extras_per_env__*` values are NaN'd by record.py at done_step (along with the
    # state arrays — both reflect post-reset garbage there). Last VALID per-env extras
    # step is therefore done_step - 1. NOTE: the raw `rewards` field is NaN'd one step
    # later (record.py keeps the death-step reward as the valid terminal reward) — but
    # we want the task component specifically, which only lives in per-env extras, so
    # the done_step-1 lookup is the right move here.
    if task_reward_key not in d.files:
        raise KeyError(
            f"`{task_reward_key}` not in rollout npz. Re-record with "
            f"`emit_per_env_extras=True` (record.py flips this for you), or pass "
            f"`task_reward_key=` to point at an available per-env field."
        )
    task_reward = d[task_reward_key]                     # (T, N), NaN at done_step+
    env_idx = np.arange(N)
    ds_for_metric = np.clip(done_step - 1, 0, T - 1).astype(np.int64)
    final_task_reward = task_reward[ds_for_metric, env_idx]    # (N,)

    # --- cube pose at first frame and last VALID frame ---
    # Per-env state arrays are NaN'd at done_step (post-reset garbage), so the last
    # valid step is done_step - 1.
    actual_pos = d["actual_obj_pos"]                     # (T, N, 3)
    actual_quat = d["actual_obj_quat"]                   # (T, N, 4)
    init_pos = actual_pos[0]                             # (N, 3)
    init_quat = actual_quat[0]                           # (N, 4)
    final_step = np.clip(done_step - 1, 0, T - 1).astype(np.int64)
    final_pos = actual_pos[final_step, env_idx]
    final_quat = actual_quat[final_step, env_idx]

    # --- reference cube pose at the same phases ---
    phase = d["phase"]                                   # (T, N) float
    obj_poses_traj = traj["obj_poses"]                   # (T_traj, 7) [pos, quat-wxyz]
    T_traj = obj_poses_traj.shape[0]
    init_phase_idx = np.clip(phase[0],                    0, T_traj - 1).astype(np.int64)
    final_phase_idx = np.clip(phase[final_step, env_idx], 0, T_traj - 1).astype(np.int64)
    ref_init_pos  = obj_poses_traj[init_phase_idx,  :3]
    ref_init_quat = obj_poses_traj[init_phase_idx,  3:]
    ref_final_pos  = obj_poses_traj[final_phase_idx, :3]
    ref_final_quat = obj_poses_traj[final_phase_idx, 3:]

    # --- three error metrics, initial and final ---
    init_err_pos  = np.linalg.norm(init_pos  - ref_init_pos,  axis=-1)
    init_err_quat = _quat_angle_diff(init_quat,  ref_init_quat)
    init_err_pose = _corner_distance(init_pos,  init_quat,  ref_init_pos,  ref_init_quat,  box_dims)

    final_err_pos  = np.linalg.norm(final_pos - ref_final_pos, axis=-1)
    final_err_quat = _quat_angle_diff(final_quat, ref_final_quat)
    final_err_pose = _corner_distance(final_pos, final_quat, ref_final_pos, ref_final_quat, box_dims)

    # --- success/failure classification ---
    early_term = done_step < early_thresh
    low_reward = final_task_reward < min_task_reward
    success = ~early_term & ~low_reward

    reason = np.full(N, "success", dtype=object)
    reason[early_term & ~low_reward] = "early_term"
    reason[~early_term & low_reward] = "low_reward"
    reason[early_term & low_reward]  = "early_term+low_reward"

    return {
        # config echo
        "num_envs": N, "T": T, "box_dims": box_dims,
        "early_step_threshold": early_thresh, "min_task_reward": min_task_reward,
        "task_reward_key": task_reward_key,
        # per-env outcomes
        "done_step": done_step, "final_task_reward": final_task_reward,
        "success": success, "reason": reason,
        # per-env poses
        "init_pos":  init_pos,  "init_quat":  init_quat,
        "final_pos": final_pos, "final_quat": final_quat,
        "ref_init_pos":  ref_init_pos,  "ref_init_quat":  ref_init_quat,
        "ref_final_pos": ref_final_pos, "ref_final_quat": ref_final_quat,
        # per-env errors
        "init_err_pos":  init_err_pos,  "final_err_pos":  final_err_pos,
        "init_err_quat": init_err_quat, "final_err_quat": final_err_quat,
        "init_err_pose": init_err_pose, "final_err_pose": final_err_pose,
    }


# ---------- summary printer ----------

def print_summary(s: dict, title: str = "") -> None:
    """Print success % + mean final error over successes only."""
    N = s["num_envs"]
    succ = s["success"]
    n_succ = int(succ.sum())
    n_fail = N - n_succ
    print(f"=== Rollout summary {title} ===")
    print(f"  N={N}, early_step_threshold={s['early_step_threshold']}, "
          f"min_task_reward={s['min_task_reward']}")
    print(f"  Success: {n_succ}/{N} ({100 * n_succ / N:.1f}%)")
    if n_fail > 0:
        c = Counter(s["reason"][~succ])
        print(f"  Failure breakdown: {dict(c)}")
    if n_succ == 0:
        print("  (no successes → error stats skipped)")
        return
    print(f"\n  Mean final error (successes only, n={n_succ}):")
    print(f"    pos:  {s['final_err_pos'][succ].mean()  * 1000:6.1f} mm")
    print(f"    quat: {np.degrees(s['final_err_quat'][succ]).mean():6.2f} deg")
    print(f"    pose: {s['final_err_pose'][succ].mean() * 1000:6.1f} mm "
          f"(mean over 8 corners)")
    print(f"\n  Final task reward (successes): "
          f"mean={s['final_task_reward'][succ].mean():.3f}, "
          f"median={np.median(s['final_task_reward'][succ]):.3f}, "
          f"min={s['final_task_reward'][succ].min():.3f}")


# ---------- scatter plot ----------

def plot_initial_vs_final(
    s: dict,
    mode: str = "pose",
    ax=None,
    *,
    color_success: str = "tab:blue",
    color_fail: str = "tab:red",
    marker_size: int = 18,
    title: str | None = None,
):
    """Scatter initial error (x) vs final error (y) per env, colored by success.

    Args:
        s: dict from compute_rollout_summary.
        mode: 'pos' | 'quat' | 'pose'. Units: m for pos / pose, rad for quat.
        ax: optional matplotlib axis to draw into.
        title: optional override.

    Returns the axis. y=x reference line shows "no improvement"; points below it have
    been corrected toward the reference, above it have drifted further.
    """
    import matplotlib.pyplot as plt
    if mode not in ("pos", "quat", "pose"):
        raise ValueError(f"mode must be 'pos' | 'quat' | 'pose', got {mode!r}")
    x = s[f"init_err_{mode}"]
    y = s[f"final_err_{mode}"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    succ = s["success"]
    ax.scatter(x[succ], y[succ],  s=marker_size, color=color_success,
               alpha=0.65, label=f"success ({int(succ.sum())})")
    ax.scatter(x[~succ], y[~succ], s=marker_size, color=color_fail,
               alpha=0.65, marker="x", label=f"fail ({int((~succ).sum())})")

    # Square axes around the data, then add y=x reference (= no improvement).
    finite_x = x[np.isfinite(x)]
    finite_y = y[np.isfinite(y)]
    if finite_x.size and finite_y.size:
        lo = min(finite_x.min(), finite_y.min())
        hi = max(finite_x.max(), finite_y.max())
        ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", alpha=0.5,
                label="y = x (no improvement)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    units = "m" if mode in ("pos", "pose") else "rad"
    suffix = " (mean corner distance)" if mode == "pose" else ""
    ax.set_xlabel(f"initial {mode} error ({units}){suffix}")
    ax.set_ylabel(f"final {mode} error ({units}){suffix}")
    ax.set_title(title or f"initial → final cube {mode} error")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    return ax
