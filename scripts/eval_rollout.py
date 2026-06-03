"""Evaluate the final-pose quality of a rollout (boxhinge / boxlift / any task).

Takes the rollout npz written by ur_rtde_real_time.py (real) or record.py (sim),
finds the last valid box pose estimate, and compares it to the reference
trajectory's goal pose (the end of ``obj_poses``). Reports:

  - 3D position error:  ||p_final_actual - p_goal_ref||         (m)   ``pos_err_m``
  - xy  position error: ||(p_final_actual - p_goal_ref)[:2]||   (m)   ``xy_err_m``
    plus the per-axis xy delta. Both are kept so tasks that move in-plane
    (boxhinge) can use xy_err while tasks where z matters (boxlift) can use
    pos_err. ``init_*`` and ``*_at_phase_*`` variants follow the same pattern.
  - orientation error: geodesic angle between the two quaternions (deg)
  - mean corner pose error: mean Euclidean distance between corresponding box
    corners under actual vs reference pose (m). One scalar that combines
    translation and rotation, expressed in metres; needs the box dimensions
    (read from the trajectory's ``object_dims`` if present, else env.yaml's
    ``cube_cfg.spawn.size``, else a boxhinge-shaped fallback).

The orientation metric is convention-agnostic: it only uses the 4D dot product,
so as long as the rollout's actual_obj_quat and the trajectory's obj_poses[:,3:]
are stored in the same component order (they are — ur_rtde_real_time.py feeds
both through the same wxyz path), the angle is correct without us having to know
whether that order is wxyz or xyzw. The corner-distance metric *does* assume
wxyz (matching the rest of the pipeline; see analyze_rollout.ipynb::analyze_box).
If the convention is ever flipped, the angle stays correct but the corner
distance does not.

Pure numpy + yaml: intentionally no matplotlib, so it runs even with a broken
plotting stack.

Accepts both individual rollout npz files and directories. A directory is
scanned for real-robot rollouts only (the ones ur_rtde_real_time.py tags with
the 'real' robot token; sim rollouts — 'isaac'/'ursim' — are skipped). You can
point at:
  - a policy/run folder      (logs/rsl_rl/<exp>/<run>/ — has ur_rtde_logs/)
  - its ur_rtde_logs/ folder directly
  - an experiment folder     (logs/rsl_rl/<exp>/ — every run under it is scanned)
Explicitly named npz files are always evaluated, real or not.

Companion modules:
  - rollout_summary.py: multi-env sim rollouts. Same record schema via
    ``to_records()`` so the plotters below work on either source.
  - rollout_plots.py:   matplotlib plot functions consuming the record schema.

Usage:
    python scripts/eval_rollout.py <rollout.npz | folder> [more ...]
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Output uses Δ and an em-dash; the default Windows console codec (cp1252)
# can't encode them and would crash mid-batch. Degrade gracefully instead.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


def _is_real_rollout(npz_path: Path) -> bool:
    """ur_rtde_real_time.py names rollouts
    <date>_<time>_gain<g>_la<l>_<robot>[_as<s>][_useref], where <robot> is one
    of real|isaac|ursim. Real-robot rollouts carry the 'real' token; sim ones
    are skipped when scanning a folder."""
    return "real" in npz_path.stem.split("_")


def _uses_ref(npz_path: Path) -> bool:
    """True for a --use_ref rollout (nominal trajectory replayed, policy bypassed)
    vs. a normal policy rollout. ur_rtde_real_time.py only records this in the
    filename (the trailing '_useref' token), not in the npz, so we read it from
    the same stem tokens as _is_real_rollout."""
    return "useref" in npz_path.stem.split("_")


def _npz_scalar(d, key: str) -> float | None:
    """Read an optional scalar from the npz as float. Older real rollouts may
    not carry gain/lookahead_time, so missing → None rather than KeyError."""
    return float(d[key]) if key in d.files else None


def _parse_timestamp(npz_path: Path) -> datetime | None:
    """Pull the rollout's wall-clock datetime out of the filename. ur_rtde_real_time.py
    names rollouts <YYYYMMDD>_<HHMMSS>_..., so split on '_' and parse the first
    two tokens. Returned as a datetime so pandas can filter on time ranges via
    `pd.to_datetime(df['datetime'])`. None for unrecognized stems."""
    tokens = npz_path.stem.split("_")
    if len(tokens) < 2:
        return None
    try:
        return datetime.strptime(f"{tokens[0]}_{tokens[1]}", "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def _collect_rollouts(path: Path) -> list[Path]:
    """A file resolves to itself (evaluated regardless of real/sim — the user
    asked for it explicitly). A directory is scanned for *real* rollouts only:
    if it has an ur_rtde_logs/ subdir we look there, otherwise we recurse, so
    pointing at a run folder, its ur_rtde_logs/, or a whole experiment folder
    all work. npz parent structure is preserved so _resolve_trajectory_path
    (which walks npz.parent.parent) still finds each run's env.yaml."""
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"not a file or directory: {path}")
    log_dir = path / "ur_rtde_logs"
    search_root = log_dir if log_dir.is_dir() else path
    return sorted(p for p in search_root.rglob("*.npz") if _is_real_rollout(p))


def _quat_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Geodesic angle (deg) between two quaternions. Normalizes first (interp /
    measurement can leave them slightly off-unit). |dot| handles the double-cover
    (q and -q are the same rotation)."""
    q_a = q_a / max(np.linalg.norm(q_a), 1e-12)
    q_b = q_b / max(np.linalg.norm(q_b), 1e-12)
    dot = float(np.clip(abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """wxyz quaternion -> 3x3 rotation matrix. Same convention as the rest of
    this pipeline (ur_rtde_real_time.py / analyze_rollout.ipynb). Normalizes to
    absorb minor numerical drift from interp / pose-listener noise."""
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _box_corners(pos: np.ndarray, quat: np.ndarray,
                 dims: tuple[float, float, float]) -> np.ndarray:
    """World-frame positions of the 8 corners of a box centred at ``pos`` with
    orientation ``quat`` (wxyz) and full side lengths ``dims = (dx, dy, dz)``.
    Returns an (8, 3) array."""
    dx, dy, dz = float(dims[0]), float(dims[1]), float(dims[2])
    local = np.array([(sx * dx / 2, sy * dy / 2, sz * dz / 2)
                      for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0)
                      for sz in (-1.0, 1.0)])
    return local @ _quat_to_rotmat(quat).T + pos


def _mean_corner_dist(pos_a: np.ndarray, quat_a: np.ndarray,
                      pos_b: np.ndarray, quat_b: np.ndarray,
                      dims: tuple[float, float, float]) -> float:
    """Mean Euclidean distance between corresponding box corners under pose A
    vs pose B. A single scalar that bakes both translation and rotation error
    into a metric in metres: for identical poses it's 0; for a pure translation
    Δ it's ||Δ||; for a pure rotation around the box centre it grows with both
    the angle and the box extents."""
    a = _box_corners(pos_a, quat_a, dims)
    b = _box_corners(pos_b, quat_b, dims)
    return float(np.linalg.norm(a - b, axis=1).mean())


def _resolve_box_dims(npz_path: Path, traj) -> tuple[float, float, float]:
    """Box (dx, dy, dz) full side lengths. Tries the trajectory npz's
    ``object_dims`` first (set by recent IK pipelines), then the run's env.yaml
    ``cube_cfg.spawn.size``, then a boxhinge-shaped fallback. Mirrors the chain
    used in notebooks/analyze_rollout.ipynb::analyze_box, so the corner-distance
    metric stays consistent with what the notebook would draw."""
    if "object_dims" in traj.files:
        d = np.asarray(traj["object_dims"], dtype=float).reshape(-1)
        return float(d[0]), float(d[1]), float(d[2])
    env_yaml = npz_path.parent.parent / "params" / "env.yaml"
    if env_yaml.is_file():
        try:
            with open(env_yaml, "r") as f:
                env_cfg = yaml.unsafe_load(f)
            size = env_cfg["cube_cfg"]["spawn"]["size"]
            return float(size[0]), float(size[1]), float(size[2])
        except Exception:
            pass
    return (0.235, 0.34, 0.27)


def _ref_at_phase(obj_poses: np.ndarray, ph: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference object pose linearly interpolated at a (possibly fractional)
    trajectory-index phase. Returns (pos[3], quat[4])."""
    T = obj_poses.shape[0]
    idx = np.arange(T)
    pos = np.array([np.interp(ph, idx, obj_poses[:, j]) for j in range(3)])
    quat = np.array([np.interp(ph, idx, obj_poses[:, 3 + j]) for j in range(4)])
    return pos, quat


def evaluate(npz_path: Path, verbose: bool = True) -> dict | None:
    """Evaluate a single rollout npz. Returns a result dict (or None if the npz
    has no usable box pose). Set verbose=False to suppress the printed report
    and only get the dict back (useful from a notebook)."""
    npz_path = Path(npz_path)
    d = np.load(npz_path)
    for k in ("actual_obj_pos", "actual_obj_quat", "phase"):
        if k not in d.files:
            if verbose:
                print(f"[{npz_path.name}] missing '{k}' in npz — cannot evaluate.")
            return None

    actual_pos = d["actual_obj_pos"]    # (N, 3), NaN where no pose
    actual_quat = d["actual_obj_quat"]  # (N, 4)
    phase = d["phase"]                  # (N,) float trajectory-index units

    valid = ~np.isnan(actual_pos[:, 0])
    if not valid.any():
        if verbose:
            print(f"[{npz_path.name}] no valid box pose samples — cannot evaluate.")
        return None
    i_first = int(np.flatnonzero(valid)[0])  # first non-NaN estimate
    i_last = int(np.flatnonzero(valid)[-1])  # last non-NaN estimate

    traj = np.load(_resolve_trajectory_path(npz_path))
    obj_poses = traj["obj_poses"]          # (T, 7): pos[:3], quat[3:]
    T = obj_poses.shape[0]
    goal_pos = obj_poses[-1, :3]
    goal_quat = obj_poses[-1, 3:]

    # Initial box error: where the box actually started vs where the reference
    # expected it at that same phase. Captures placement / reset offset, which
    # the policy can't fix and which inflates the goal error downstream.
    init_pos = actual_pos[i_first]
    init_quat = actual_quat[i_first]
    init_phase = float(phase[i_first])
    init_ref_pos, init_ref_quat = _ref_at_phase(obj_poses, init_phase)
    init_pos_delta = init_pos - init_ref_pos
    init_pos_err = float(np.linalg.norm(init_pos_delta))      # 3D norm
    init_xy_err  = float(np.linalg.norm(init_pos_delta[:2]))  # in-plane only
    init_ori_err_deg = _quat_angle_deg(init_quat, init_ref_quat)

    final_pos = actual_pos[i_last]
    final_quat = actual_quat[i_last]
    final_phase = float(phase[i_last])

    pos_delta = final_pos - goal_pos
    pos_err   = float(np.linalg.norm(pos_delta))         # 3D norm — use for boxlift
    xy_err    = float(np.linalg.norm(pos_delta[:2]))     # in-plane only — boxhinge
    ori_err_deg = _quat_angle_deg(final_quat, goal_quat)

    # Completion context: comparing to the goal only means "task success" if the
    # rollout actually reached the end of the reference. If it stopped early,
    # also report the reference pose at the phase the estimate was taken, so a
    # large goal error isn't misread as bad tracking when it's just incomplete.
    completed = final_phase >= (T - 1) - 1e-3
    ref_at_phase_pos, ref_at_phase_quat = _ref_at_phase(obj_poses, final_phase)
    pos_delta_at_phase = final_pos - ref_at_phase_pos
    pos_err_at_phase = float(np.linalg.norm(pos_delta_at_phase))
    xy_err_at_phase  = float(np.linalg.norm(pos_delta_at_phase[:2]))
    ori_err_at_phase = _quat_angle_deg(final_quat, ref_at_phase_quat)

    # Mean corner distance — pos+ori in a single metres-valued scalar. Reported
    # at all three reference frames the other metrics use (initial-phase, goal,
    # final-phase) so it slots in alongside them.
    box_dims = _resolve_box_dims(npz_path, traj)
    init_pose_err = _mean_corner_dist(
        init_pos, init_quat, init_ref_pos, init_ref_quat, box_dims)
    pose_err = _mean_corner_dist(
        final_pos, final_quat, goal_pos, goal_quat, box_dims)
    pose_err_at_phase = _mean_corner_dist(
        final_pos, final_quat, ref_at_phase_pos, ref_at_phase_quat, box_dims)

    if verbose:
        print(f"[{npz_path.name}]")
        print(f"  initial estimate: step idx {i_first}, phase {init_phase:.2f} / {T - 1} "
              f"(box placement / reset offset)")
        print(f"  vs reference AT INITIAL PHASE:")
        print(f"    pos err (3D):    {init_pos_err * 1000:8.2f} mm   "
              f"(Δ = [{init_pos_delta[0] * 1000:+.1f}, "
              f"{init_pos_delta[1] * 1000:+.1f}, "
              f"{init_pos_delta[2] * 1000:+.1f}] mm)")
        print(f"    xy position err: {init_xy_err * 1000:8.2f} mm   (in-plane only)")
        print(f"    orientation err: {init_ori_err_deg:8.2f} deg")
        print(f"    mean corner err: {init_pose_err * 1000:8.2f} mm   (pos+ori combined)")
        print(f"  final estimate: step idx {i_last}, phase {final_phase:.2f} / {T - 1} "
              f"({'completed' if completed else 'INCOMPLETE — stopped early'})")
        print(f"  vs reference GOAL (end of trajectory):")
        print(f"    pos err (3D):    {pos_err * 1000:8.2f} mm   "
              f"(Δ = [{pos_delta[0] * 1000:+.1f}, {pos_delta[1] * 1000:+.1f}, "
              f"{pos_delta[2] * 1000:+.1f}] mm)")
        print(f"    xy position err: {xy_err * 1000:8.2f} mm")
        print(f"    orientation err: {ori_err_deg:8.2f} deg")
        print(f"    mean corner err: {pose_err * 1000:8.2f} mm")
        if not completed:
            print(f"  vs reference AT FINAL PHASE (tracking, not task success):")
            print(f"    pos err (3D):    {pos_err_at_phase * 1000:8.2f} mm")
            print(f"    xy position err: {xy_err_at_phase * 1000:8.2f} mm")
            print(f"    orientation err: {ori_err_at_phase:8.2f} deg")
            print(f"    mean corner err: {pose_err_at_phase * 1000:8.2f} mm")

    return {
        "npz": npz_path.name,
        "datetime": _parse_timestamp(npz_path),
        "use_ref": _uses_ref(npz_path),
        "gain": _npz_scalar(d, "gain"),
        "lookahead": _npz_scalar(d, "lookahead_time"),
        "init_phase": init_phase,
        # Initial errors (vs reference at the FIRST valid pose's phase)
        "init_pos_err_m": init_pos_err,        # 3D
        "init_xy_err_m": init_xy_err,          # in-plane only
        "init_pos_delta_m": init_pos_delta,
        "init_ori_err_deg": init_ori_err_deg,
        "init_pose_err_m": init_pose_err,
        # Final errors vs the GOAL pose (end of obj_poses)
        "final_phase": final_phase,
        "completed": completed,
        "pos_err_m": pos_err,                  # 3D
        "xy_err_m": xy_err,                    # in-plane only
        "pos_delta_m": pos_delta,
        "ori_err_deg": ori_err_deg,
        "pose_err_m": pose_err,
        # Final errors vs reference AT the final reached phase (fairer for incompletes)
        "pos_err_at_phase_m": pos_err_at_phase,
        "xy_err_at_phase_m": xy_err_at_phase,
        "ori_err_at_phase_deg": ori_err_at_phase,
        "pose_err_at_phase_m": pose_err_at_phase,
    }


def expand_rollouts(paths, verbose: bool = True) -> list[Path]:
    """Turn a path or an iterable of paths (files and/or folders) into a
    deduped, order-preserving list of rollout npz Paths. Folders are scanned
    for real-robot rollouts only (see _collect_rollouts); explicitly named
    files are kept as-is."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    rollouts: list[Path] = []
    seen: set[Path] = set()
    for arg in paths:
        arg = Path(arg)
        try:
            collected = _collect_rollouts(arg)
        except FileNotFoundError as e:
            if verbose:
                print(f"[{arg}] error: {e}")
            continue
        if arg.is_dir() and not collected and verbose:
            print(f"[{arg}] no real-robot rollouts found.")
        for p in collected:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                rollouts.append(p)
    return rollouts


def analyze(paths, verbose: bool = True) -> list[dict]:
    """Evaluate every real-robot rollout under `paths` and return a list of
    result dicts — one per rollout, ready for ``pandas.DataFrame(results)``.

    `paths` is a file/folder path or an iterable of them; a folder (run dir,
    its ur_rtde_logs/, or a whole experiment dir) is scanned for real-robot
    rollouts only. Set verbose=False for a silent return (notebook use):

        from scripts.eval_boxhinge_rollout import analyze
        results = analyze("logs/rsl_rl/boxhinge/<run>", verbose=False)
    """
    results = []
    for p in expand_rollouts(paths, verbose=verbose):
        try:
            r = evaluate(p, verbose=verbose)
            if r is not None:
                results.append(r)
        except (FileNotFoundError, KeyError) as e:
            if verbose:
                print(f"[{p.name}] error: {e}")
    return results


def print_summary(results: list[dict]) -> None:
    """Print the multi-rollout summary table for the dicts `analyze` returns."""
    if not results:
        return
    print("\n=== summary (init = vs ref at start; goal = vs ref end) ===")
    print(f"{'rollout':<48} {'phase':>8} {'init_xy(mm)':>11} "
          f"{'init_ori':>8} {'xy(mm)':>9} {'ori(deg)':>9}  done")
    for r in results:
        print(f"{r['npz']:<48} {r['final_phase']:>8.1f} "
              f"{r['init_xy_err_m'] * 1000:>11.1f} "
              f"{r['init_ori_err_deg']:>8.2f} "
              f"{r['xy_err_m'] * 1000:>9.1f} {r['ori_err_deg']:>9.2f}  "
              f"{'Y' if r['completed'] else 'N'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_path", nargs="+", type=Path,
                        help="Rollout npz files and/or folders. Folders are "
                             "scanned for real-robot rollouts only.")
    args = parser.parse_args()

    results = analyze(args.npz_path, verbose=True)
    if len(results) > 1:
        print_summary(results)


if __name__ == "__main__":
    main()
