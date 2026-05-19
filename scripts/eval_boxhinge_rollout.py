"""Evaluate the final-pose quality of a boxhinge rollout.

Takes the rollout npz written by ur_rtde_real_time.py, finds the last valid box
pose estimate, and compares it to the reference trajectory's goal pose (the end
of obj_poses). Reports:

  - xy position error: ||(p_final_actual - p_goal_ref)[:2]||  (m), plus the
    per-axis xy delta. z is ignored (the boxhinge task is evaluated in-plane).
  - orientation error: geodesic angle between the two quaternions (deg)

The orientation metric is convention-agnostic: it only uses the 4D dot product,
so as long as the rollout's actual_obj_quat and the trajectory's obj_poses[:,3:]
are stored in the same component order (they are — ur_rtde_real_time.py feeds
both through the same wxyz path), the angle is correct without us having to know
whether that order is wxyz or xyzw.

Pure numpy + yaml: intentionally no matplotlib, so it runs even with a broken
plotting stack.

Usage:
    python scripts/eval_boxhinge_rollout.py <rollout.npz> [<rollout2.npz> ...]
"""

import argparse
from pathlib import Path

import numpy as np
import yaml


def _resolve_trajectory_path(npz_path: Path) -> Path:
    """Mirror analyze_rollout.ipynb: the rollout npz lives at
    <run_dir>/ur_rtde_logs/<file>.npz; env.yaml at <run_dir>/params/env.yaml
    holds trajectory_path, relative to the repo root if not absolute."""
    run_dir = npz_path.parent.parent
    env_yaml = run_dir / "params" / "env.yaml"
    if not env_yaml.is_file():
        raise FileNotFoundError(f"env.yaml not found at {env_yaml}")
    with open(env_yaml, "r") as f:
        env_cfg = yaml.unsafe_load(f)
    traj_path = Path(env_cfg["trajectory_path"])
    if not traj_path.is_absolute():
        # run_dir is logs/rsl_rl/<exp>/<run>; repo root is 4 levels up.
        traj_path = run_dir.parent.parent.parent.parent / traj_path
    if not traj_path.is_file():
        raise FileNotFoundError(f"trajectory file not found at {traj_path}")
    return traj_path


def _quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle (deg) between two quaternions. Normalizes first (interp /
    measurement can leave them slightly off-unit). |dot| handles the double-cover
    (q and -q are the same rotation)."""
    q_a = q_a / max(np.linalg.norm(q_a), 1e-12)
    q_b = q_b / max(np.linalg.norm(q_b), 1e-12)
    dot = float(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def evaluate(npz_path: Path) -> dict | None:
    d = np.load(npz_path)
    for k in ("actual_obj_pos", "actual_obj_quat", "phase"):
        if k not in d.files:
            print(f"[{npz_path.name}] missing '{k}' in npz — cannot evaluate.")
            return None

    actual_pos = d["actual_obj_pos"]    # (N, 3), NaN where no pose
    actual_quat = d["actual_obj_quat"]  # (N, 4)
    phase = d["phase"]                  # (N,) float trajectory-index units

    valid = ~np.isnan(actual_pos[:, 0])
    if not valid.any():
        print(f"[{npz_path.name}] no valid box pose samples — cannot evaluate.")
        return None
    i_last = int(np.flatnonzero(valid)[-1])  # last non-NaN estimate

    traj = np.load(_resolve_trajectory_path(npz_path))
    obj_poses = traj["obj_poses"]          # (T, 7): pos[:3], quat[3:]
    T = obj_poses.shape[0]
    goal_pos = obj_poses[-1, :3]
    goal_quat = obj_poses[-1, 3:]

    final_pos = actual_pos[i_last]
    final_quat = actual_quat[i_last]
    final_phase = float(phase[i_last])

    pos_delta = final_pos - goal_pos
    xy_err = float(np.linalg.norm(pos_delta[:2]))  # in-plane only; z ignored
    ori_err_deg = _quat_angle_deg(final_quat, goal_quat)

    # Completion context: comparing to the goal only means "task success" if the
    # rollout actually reached the end of the reference. If it stopped early,
    # also report the reference pose at the phase the estimate was taken, so a
    # large goal error isn't misread as bad tracking when it's just incomplete.
    completed = final_phase >= (T - 1) - 1e-3
    ref_at_phase_pos = np.array([
        np.interp(final_phase, np.arange(T), obj_poses[:, j]) for j in range(3)
    ])
    ref_at_phase_quat = np.array([
        np.interp(final_phase, np.arange(T), obj_poses[:, 3 + j]) for j in range(4)
    ])
    xy_err_at_phase = float(np.linalg.norm((final_pos - ref_at_phase_pos)[:2]))
    ori_err_at_phase = _quat_angle_deg(final_quat, ref_at_phase_quat)

    print(f"[{npz_path.name}]")
    print(f"  final estimate: step idx {i_last}, phase {final_phase:.2f} / {T - 1} "
          f"({'completed' if completed else 'INCOMPLETE — stopped early'})")
    print(f"  vs reference GOAL (end of trajectory):")
    print(f"    xy position err: {xy_err * 1000:8.2f} mm   "
          f"(Δxy = [{pos_delta[0] * 1000:+.1f}, {pos_delta[1] * 1000:+.1f}] mm)")
    print(f"    orientation err: {ori_err_deg:8.2f} deg")
    if not completed:
        print(f"  vs reference AT FINAL PHASE (tracking, not task success):")
        print(f"    xy position err: {xy_err_at_phase * 1000:8.2f} mm")
        print(f"    orientation err: {ori_err_at_phase:8.2f} deg")

    return {
        "npz": npz_path.name,
        "final_phase": final_phase,
        "completed": completed,
        "xy_err_m": xy_err,
        "pos_delta_m": pos_delta,
        "ori_err_deg": ori_err_deg,
        "xy_err_at_phase_m": xy_err_at_phase,
        "ori_err_at_phase_deg": ori_err_at_phase,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_path", nargs="+", type=Path,
                        help="One or more rollout npz files from ur_rtde_real_time.py.")
    args = parser.parse_args()

    results = []
    for p in args.npz_path:
        try:
            r = evaluate(p)
            if r is not None:
                results.append(r)
        except (FileNotFoundError, KeyError) as e:
            print(f"[{p.name}] error: {e}")

    # Summary table when evaluating more than one rollout.
    if len(results) > 1:
        print("\n=== summary (vs goal) ===")
        print(f"{'rollout':<48} {'phase':>8} {'xy(mm)':>9} {'ori(deg)':>9}  done")
        for r in results:
            print(f"{r['npz']:<48} {r['final_phase']:>8.1f} "
                  f"{r['xy_err_m'] * 1000:>9.1f} {r['ori_err_deg']:>9.2f}  "
                  f"{'Y' if r['completed'] else 'N'}")


if __name__ == "__main__":
    main()
