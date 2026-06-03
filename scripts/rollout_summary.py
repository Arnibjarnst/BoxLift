"""Per-env success analysis for multi-env rollouts.

For each env in a record.py rollout we compute:
  - success / failure (based on early-termination AND terminal-task-reward thresholds)
  - cube pose at first frame and at the last VALID frame (= done_step - 1, since the
    death-step state is NaN'd by record.py)
  - reference cube pose at the same trajectory phases (from the .npz traj file)
  - three error metrics, each in two reference frames:
      * pos  — ||p_actual − p_ref||                                (m)
      * xy   — ||(p_actual − p_ref)[:2]||                          (m)   in-plane only
      * quat — smallest rotation angle between the two quaternions  (rad)
      * pose — mean Euclidean distance over the 8 cube corners      (m)
    Final errors are computed BOTH vs the env's last-reached phase
    (``final_err_*_at_phase``) AND vs the trajectory's goal pose (``final_err_*``).

`to_records(summary)` flattens the per-env arrays into the same per-rollout dict
schema produced by ``eval_rollout.evaluate``, so the plotters in
``rollout_plots.py`` work on either source.

Typical use from the notebook:

    from scripts.rollout_summary import compute_rollout_summary, print_summary, to_records
    from scripts.rollout_plots import plot_init_vs_final, plot_error
    s = compute_rollout_summary("logs/.../rollout/output.npz", min_task_reward=0.05)
    print_summary(s, title="some run")
    plot_init_vs_final(to_records(s), x="pose", y="pose", metric="pos")
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

def _load_traj_for_rollout(rollout_path: Path) -> tuple[dict | None, tuple[float, float, float]]:
    """Pull `trajectory_path` from <run>/params/env.yaml, resolve relative to repo root,
    and load. Also derive box dims (object_dims from the traj if present, else
    cube_cfg.spawn.size from env.yaml).

    Dataset-mode rollouts (boxtracker etc.) have an empty `trajectory_path` and a
    populated `dataset_path` instead — in that case we return (None, box_dims) and
    the per-env reference lives in the rollout npz itself as `ref_obj_poses`.
    """
    run_dir = rollout_path.parent.parent
    env_yaml = run_dir / "params" / "env.yaml"
    with open(env_yaml, "r") as f:
        env_cfg = yaml.unsafe_load(f)
    traj_path = env_cfg.get("trajectory_path", "") or ""
    if not traj_path:
        # Dataset mode — reference comes from the rollout npz (ref_obj_poses), not a file.
        box_dims = tuple(float(x) for x in env_cfg["cube_cfg"]["spawn"]["size"])
        return None, box_dims
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
    # Early-termination threshold. Three branches, in priority order:
    #   1. User passed an explicit value → scalar, applies uniformly.
    #   2. Dataset-mode rollout (env_traj_length saved per env) → per-env threshold
    #      `env_traj_length[i] - 5`. This is critical: short-segment envs reach their
    #      natural timeout at done_step ≈ env_traj_length, which would be FAR below a
    #      global `T - 5` derived from the longest segment. Comparing per-env catches
    #      "actually-failed" without false-positives on legitimately-short successes.
    #   3. Single-trajectory rollout → scalar `T - 5` (legacy behavior).
    if early_step_threshold is not None:
        early_thresh = int(early_step_threshold)
    elif "env_traj_length" in d.files:
        # 5-step tolerance window matches single-traj's `T - 5` semantics.
        early_thresh = np.maximum(d["env_traj_length"].astype(np.int64) - 5, 0)
    else:
        early_thresh = T - 5

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
    # Single-traj mode: reference is shared, indexed by phase only — `obj_poses_traj`
    # has shape (T_traj, 7) and lookup is `obj_poses_traj[phase_idx]`.
    # Dataset mode: reference is per-env (each env may have used a different segment).
    # Pulled from the rollout npz's `ref_obj_poses` field, shape (N, T_max, 7). Lookup
    # is `ref_obj_poses[env_idx, phase_idx]`. Both branches produce the same (N, ...)
    # output arrays so the rest of the code is shape-uniform.
    phase = d["phase"]                                   # (T, N) float
    init_phase_arr  = phase[0]                            # (N,) — should be 0 in record.py
    final_phase_arr = phase[final_step, env_idx]          # (N,) — last valid phase
    if "ref_obj_poses" in d.files:
        ref_per_env = d["ref_obj_poses"]                  # (N, T_max, 7) [pos, quat-wxyz]
        T_ref = ref_per_env.shape[1]
        init_phase_idx  = np.clip(init_phase_arr,  0, T_ref - 1).astype(np.int64)
        final_phase_idx = np.clip(final_phase_arr, 0, T_ref - 1).astype(np.int64)
        ref_init_pose  = ref_per_env[env_idx, init_phase_idx]    # (N, 7)
        ref_final_pose = ref_per_env[env_idx, final_phase_idx]   # (N, 7)
        # Per-env goal: the last VALID frame of each env's segment. If env_traj_length
        # was saved use it directly; else fall back to T_max-1 (boxtracker right-pads
        # with the last valid frame, so the last index is fine but the explicit length
        # is cleaner).
        if "env_traj_length" in d.files:
            goal_idx = np.clip(d["env_traj_length"].astype(np.int64) - 1, 0, T_ref - 1)
        else:
            goal_idx = np.full(N, T_ref - 1, dtype=np.int64)
        ref_goal_pose = ref_per_env[env_idx, goal_idx]           # (N, 7)
        ref_init_pos,  ref_init_quat  = ref_init_pose[:, :3],  ref_init_pose[:, 3:]
        ref_final_pos, ref_final_quat = ref_final_pose[:, :3], ref_final_pose[:, 3:]
        goal_pos_per_env  = ref_goal_pose[:, :3]                 # (N, 3)
        goal_quat_per_env = ref_goal_pose[:, 3:]                 # (N, 4)
    else:
        obj_poses_traj = traj["obj_poses"]                # (T_traj, 7) shared across all envs
        T_ref = obj_poses_traj.shape[0]
        init_phase_idx  = np.clip(init_phase_arr,  0, T_ref - 1).astype(np.int64)
        final_phase_idx = np.clip(final_phase_arr, 0, T_ref - 1).astype(np.int64)
        ref_init_pos   = obj_poses_traj[init_phase_idx,  :3]
        ref_init_quat  = obj_poses_traj[init_phase_idx,  3:]
        ref_final_pos  = obj_poses_traj[final_phase_idx, :3]
        ref_final_quat = obj_poses_traj[final_phase_idx, 3:]
        # Broadcast the shared goal up to (N, ...) so downstream code is shape-uniform
        # with the dataset-mode branch above.
        goal_pos_per_env  = np.broadcast_to(obj_poses_traj[-1, :3], (N, 3)).copy()
        goal_quat_per_env = np.broadcast_to(obj_poses_traj[-1, 3:], (N, 4)).copy()

    # --- error metrics ---
    # initial: actual vs reference at the SAME phase (phase[0] for every env)
    init_pos_delta   = init_pos - ref_init_pos                       # (N, 3)
    init_err_pos     = np.linalg.norm(init_pos_delta, axis=-1)       # 3D norm
    init_err_xy      = np.linalg.norm(init_pos_delta[..., :2], axis=-1)
    init_err_quat    = _quat_angle_diff(init_quat, ref_init_quat)
    init_err_pose    = _corner_distance(init_pos,  init_quat,  ref_init_pos,  ref_init_quat,  box_dims)

    # final-at-phase: actual vs reference at the env's LAST VALID phase
    fp_delta         = final_pos - ref_final_pos
    final_err_pos_at_phase   = np.linalg.norm(fp_delta, axis=-1)
    final_err_xy_at_phase    = np.linalg.norm(fp_delta[..., :2], axis=-1)
    final_err_quat_at_phase  = _quat_angle_diff(final_quat, ref_final_quat)
    final_err_pose_at_phase  = _corner_distance(final_pos, final_quat, ref_final_pos, ref_final_quat, box_dims)

    # final-vs-goal: actual at last valid step vs the trajectory's GOAL pose
    # (per-env in dataset mode; broadcast from a single goal in single-traj mode —
    # the goal arrays are already (N, ...) thanks to the block above).
    fg_delta  = final_pos - goal_pos_per_env
    final_err_pos    = np.linalg.norm(fg_delta, axis=-1)
    final_err_xy     = np.linalg.norm(fg_delta[..., :2], axis=-1)
    final_err_quat   = _quat_angle_diff(final_quat, goal_quat_per_env)
    final_err_pose   = _corner_distance(final_pos, final_quat,
                                        goal_pos_per_env, goal_quat_per_env, box_dims)

    # --- success/failure classification ---
    early_term = done_step < early_thresh
    low_reward = final_task_reward < min_task_reward
    success = ~early_term & ~low_reward

    reason = np.full(N, "success", dtype=object)
    reason[early_term & ~low_reward] = "early_term"
    reason[~early_term & low_reward] = "low_reward"
    reason[early_term & low_reward]  = "early_term+low_reward"

    # Per-env trajectory assignment from dataset-mode rollouts (None for single-traj).
    # `print_summary` uses this to emit a per-trajectory breakdown when present.
    env_traj_idx = d["env_traj_idx"].astype(np.int64) if "env_traj_idx" in d.files else None

    return {
        # config echo
        "num_envs": N, "T": T, "box_dims": box_dims,
        "early_step_threshold": early_thresh, "min_task_reward": min_task_reward,
        "task_reward_key": task_reward_key,
        "rollout_path": str(Path(rollout_path)),
        # per-env outcomes
        "done_step": done_step, "final_task_reward": final_task_reward,
        "success": success, "reason": reason,
        "env_traj_idx": env_traj_idx,
        "init_phase": init_phase_arr, "final_phase": final_phase_arr,
        # per-env poses
        "init_pos":  init_pos,  "init_quat":  init_quat,
        "final_pos": final_pos, "final_quat": final_quat,
        "ref_init_pos":  ref_init_pos,  "ref_init_quat":  ref_init_quat,
        "ref_final_pos": ref_final_pos, "ref_final_quat": ref_final_quat,
        # Per-env goal arrays (always (N, ...)) so single-traj and dataset-mode rollouts
        # share the schema. In single-traj mode every row is identical.
        "goal_pos":  goal_pos_per_env,  "goal_quat":  goal_quat_per_env,
        # per-env errors — three reference frames × three metrics each
        "init_pos_delta": init_pos_delta,
        "init_err_pos":  init_err_pos,  "init_err_xy":  init_err_xy,
        "init_err_quat": init_err_quat, "init_err_pose": init_err_pose,
        "final_err_pos":  final_err_pos,  "final_err_xy":  final_err_xy,
        "final_err_quat": final_err_quat, "final_err_pose": final_err_pose,
        "final_err_pos_at_phase":  final_err_pos_at_phase,
        "final_err_xy_at_phase":   final_err_xy_at_phase,
        "final_err_quat_at_phase": final_err_quat_at_phase,
        "final_err_pose_at_phase": final_err_pose_at_phase,
    }


def to_records(summary: dict) -> list[dict]:
    """Explode a per-env summary dict (from `compute_rollout_summary`) into a list
    of per-env flat dicts in the schema produced by ``eval_rollout.evaluate``.

    Each record is one env's worth of metrics. Lets the plotters in
    ``rollout_plots.py`` (originally written against eval_rollout output) work on
    multi-env sim rollouts without modification. Synthetic ``npz`` names look
    like ``<basename>__env{i:04d}`` so envs from multiple rollouts don't collide
    if you concat several summaries.

    Mapping to the eval_rollout schema:
      - eval_rollout's ``ori_err_deg``      ← degrees(summary['final_err_quat'])
      - eval_rollout's ``pose_err_m``       ← summary['final_err_pose']           (vs GOAL)
      - eval_rollout's ``pos_err_m``        ← summary['final_err_pos']            (vs GOAL, 3D)
      - eval_rollout's ``xy_err_m``         ← summary['final_err_xy']             (vs GOAL, 2D)
      - eval_rollout's ``*_at_phase_*``     ← summary['final_err_*_at_phase']     (vs phase)
      - eval_rollout's ``completed``        ← summary['success'] (task-level "done it")
    A success-only ``use_ref=False`` is stamped on every record (sim rollouts
    don't have the real-robot use_ref concept).
    """
    rp_stem = Path(summary["rollout_path"]).stem if "rollout_path" in summary else "rollout"
    N = int(summary["num_envs"])
    init_pd = summary["init_pos_delta"]                    # (N, 3)
    # 3D delta of final vs goal (recompute for the record; could stash in summary too)
    final_pd_goal = summary["final_pos"] - np.broadcast_to(summary["goal_pos"], summary["final_pos"].shape)
    records: list[dict] = []
    for i in range(N):
        records.append({
            "npz": f"{rp_stem}__env{i:04d}",
            "use_ref": False,                              # sim rollouts have no use_ref
            "gain": None, "lookahead": None,               # n/a for sim
            "init_phase": float(summary["init_phase"][i]),
            "init_pos_err_m": float(summary["init_err_pos"][i]),
            "init_xy_err_m":  float(summary["init_err_xy"][i]),
            "init_pos_delta_m": init_pd[i].astype(float),
            "init_ori_err_deg": float(np.degrees(summary["init_err_quat"][i])),
            "init_pose_err_m":  float(summary["init_err_pose"][i]),
            "final_phase": float(summary["final_phase"][i]),
            "completed": bool(summary["success"][i]),
            "pos_err_m": float(summary["final_err_pos"][i]),
            "xy_err_m":  float(summary["final_err_xy"][i]),
            "pos_delta_m": final_pd_goal[i].astype(float),
            "ori_err_deg": float(np.degrees(summary["final_err_quat"][i])),
            "pose_err_m":  float(summary["final_err_pose"][i]),
            "pos_err_at_phase_m": float(summary["final_err_pos_at_phase"][i]),
            "xy_err_at_phase_m":  float(summary["final_err_xy_at_phase"][i]),
            "ori_err_at_phase_deg": float(np.degrees(summary["final_err_quat_at_phase"][i])),
            "pose_err_at_phase_m":  float(summary["final_err_pose_at_phase"][i]),
            # Sim-rollout-specific extras — harmless to non-eval-rollout consumers.
            "env_idx": i,
            "done_step": int(summary["done_step"][i]),
            "reason": str(summary["reason"][i]),
            "final_task_reward": float(summary["final_task_reward"][i]),
        })
    return records


# ---------- summary printer ----------

def print_summary(s: dict, title: str = "", *, error_frame: str = "phase") -> None:
    """Print success % + mean final error over successes only.

    Args:
        s: dict from compute_rollout_summary.
        title: optional title to print.
        error_frame: which "final error" to report on the mean line —
            "phase" (default) = vs reference at the env's last reached phase;
            "goal"  = vs the trajectory's goal pose (last frame of obj_poses).
            For success-only envs that completed the trajectory these are equal.
    """
    if error_frame not in ("phase", "goal"):
        raise ValueError(f"error_frame must be 'phase' or 'goal', got {error_frame!r}")
    suffix = "_at_phase" if error_frame == "phase" else ""

    N = s["num_envs"]
    succ = s["success"]
    n_succ = int(succ.sum())
    n_fail = N - n_succ
    print(f"=== Rollout summary {title} ===")
    # early_step_threshold is scalar in single-traj mode, (N,) in dataset mode — print
    # a compact range summary in the array case so the line stays readable.
    _et = s["early_step_threshold"]
    et_str = f"{int(_et)}" if np.ndim(_et) == 0 else f"per-env [{int(np.min(_et))}..{int(np.max(_et))}]"
    print(f"  N={N}, early_step_threshold={et_str}, "
          f"min_task_reward={s['min_task_reward']}")
    print(f"  Success: {n_succ}/{N} ({100 * n_succ / N:.1f}%)")
    if n_fail > 0:
        c = Counter(s["reason"][~succ])
        print(f"  Failure breakdown: {dict(c)}")
    if n_succ == 0:
        print("  (no successes → error stats skipped)")
        return
    print(f"\n  Mean final error vs reference at {error_frame} (successes only, n={n_succ}):")
    print(f"    pos (3D): {s[f'final_err_pos{suffix}'][succ].mean()  * 1000:6.1f} mm")
    print(f"    xy:       {s[f'final_err_xy{suffix}'][succ].mean()   * 1000:6.1f} mm")
    print(f"    quat:     {np.degrees(s[f'final_err_quat{suffix}'][succ]).mean():6.2f} deg")
    print(f"    pose:     {s[f'final_err_pose{suffix}'][succ].mean() * 1000:6.1f} mm "
          f"(mean over 8 corners)")
    print(f"\n  Final task reward (successes): "
          f"mean={s['final_task_reward'][succ].mean():.3f}, "
          f"median={np.median(s['final_task_reward'][succ]):.3f}, "
          f"min={s['final_task_reward'][succ].min():.3f}")

    # --- Per-trajectory breakdown (dataset-mode rollouts only) ---
    # When env_traj_idx is present and the rollout spans more than one trajectory,
    # print a per-trajectory table: success rate, failure reasons, mean error metric
    # on that trajectory's successes. Useful for "which reference trajectories does
    # the policy fail on?". Sorted by trajectory index for stable output (use the
    # ad-hoc recipe in the docstring if you want sort-by-success-rate).
    traj_idx = s.get("env_traj_idx")
    if traj_idx is not None and np.unique(traj_idx).size > 1:
        err_field = f"final_err_pose{suffix}"   # one error metric on the table to keep it readable
        err_per_env = s[err_field]
        uniq = np.unique(traj_idx)
        print(f"\n  Per-trajectory breakdown (n_traj={len(uniq)}, sorted by traj_idx):")
        print(f"    {'traj':>5}  {'n':>5}  {'succ':>5}  {'rate':>6}  "
              f"{'pose_err_mm':>11}  failures")
        for t in uniq:
            mask = traj_idx == t
            n = int(mask.sum())
            s_mask = mask & succ
            n_s = int(s_mask.sum())
            rate = 100.0 * n_s / max(n, 1)
            pose_err_mm = (err_per_env[s_mask].mean() * 1000) if n_s else float("nan")
            fail_reasons = Counter(s["reason"][mask & ~succ])
            # Compact failure reason string: omit "success" entry; truncate to a few cats.
            fail_str = ", ".join(f"{k}={v}" for k, v in fail_reasons.items()) or "-"
            print(f"    {int(t):>5d}  {n:>5d}  {n_s:>5d}  {rate:5.1f}%  "
                  f"{pose_err_mm:>11.1f}  {fail_str}")


# ---------- CLI entry ----------

def _main():
    import argparse
    p = argparse.ArgumentParser(
        description="Summarize a multi-env record.py rollout: success %, failure "
                    "breakdown, mean final pose error over successes.",
    )
    p.add_argument("npz", type=Path,
                   help="Path to record.py output.npz (must be multi-env — has `num_envs` field).")
    p.add_argument("--min-task-reward", type=float, default=0.05,
                   help="Terminal task reward below which an env counts as a failure. Default 0.05.")
    p.add_argument("--early-step-threshold", type=int, default=None,
                   help="done_step strictly less than this counts as early termination. "
                        "Default: T - 5.")
    p.add_argument("--task-reward-key", type=str,
                   default="extras_per_env__Rewards_task__total",
                   help="Per-env extras key holding the task reward (default: "
                        "extras_per_env__Rewards_task__total — needs emit_per_env_extras=True).")
    p.add_argument("--error-frame", choices=("phase", "goal"), default="phase",
                   help="Which 'final error' to report: 'phase' (vs ref at last reached "
                        "phase, default) or 'goal' (vs trajectory's last frame).")
    p.add_argument("--title", type=str, default="",
                   help="Optional title to include in the printed summary header.")
    args = p.parse_args()

    summary = compute_rollout_summary(
        args.npz,
        min_task_reward=args.min_task_reward,
        early_step_threshold=args.early_step_threshold,
        task_reward_key=args.task_reward_key,
    )
    print_summary(summary, title=args.title or str(args.npz), error_frame=args.error_frame)


if __name__ == "__main__":
    _main()
