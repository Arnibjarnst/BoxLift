#!/usr/bin/env python
"""Record one rollout per trajectory file in a dataset.

Spawns `scripts/rsl_rl/record.py` once per .npz under --dataset, each time
overriding `env.trajectory_path` to point at that file. Per-trajectory outputs
land in <ckpt_log_dir>/rollout_dataset/<traj_stem>.npz (or --output_dir).

Works with any env that already accepts env.trajectory_path as a Hydra override:
    boxhinge  — native single-trajectory env.
    boxtracker — single-traj mode added via the trajectory_path branch in
                 _load_segment_pool; pool degenerates to size 1.
    boxlift   — same as boxhinge.

The use case is "non-generalized policy evaluated across a multi-trajectory
dataset" — record.py running on each reference one at a time, so you can
compute per-trajectory success rates for a policy that was only trained on a
subset (or one) of them.

After this finishes, run rollout_summary on each file:
    for f in <log_dir>/rollout_dataset/*.npz; do
        python scripts/rollout_summary.py "$f" --title "$(basename "$f" .npz)"
    done

Or use the analyze_rollout notebook to load the whole folder and faceted-plot.

PERFORMANCE: Each subprocess pays ~30-60s IsaacSim startup. For N trajectories
that's N × startup overhead. For one-off analysis this is fine; if you record
datasets often, an in-process version that swaps trajectory tensors without
recreating the env would be substantially faster (~30× for short rollouts) but
non-trivial because each env derives RSI / segment / episode-length state from
the trajectory at _setup_scene time.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(
        description="Run record.py once per trajectory in a dataset folder.",
        # Unknown args (e.g. Hydra overrides) are passed through to record.py.
    )
    p.add_argument("--dataset", required=True, type=Path,
                   help="Folder containing per-trajectory .npz files. Each file is "
                        "passed to one record.py invocation as env.trajectory_path.")
    p.add_argument("--checkpoint", required=True, type=str,
                   help="Path to model_<iter>.pt — forwarded to record.py.")
    p.add_argument("--task", required=True, type=str,
                   help="IsaacLab task name — forwarded to record.py.")
    p.add_argument("--num_envs", type=int, default=8,
                   help="Parallel envs per rollout (per trajectory). Default 8 — bump up "
                        "if you want tighter per-trajectory success-rate confidence intervals.")
    p.add_argument("--output_dir", type=Path, default=None,
                   help="Where to write the per-trajectory output npz files. Default: "
                        "<ckpt_dir>/rollout_dataset/. Created if missing.")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only process the first N trajectories (for smoke-testing).")
    p.add_argument("--pattern", type=str, default="*.npz",
                   help="Glob pattern within --dataset to select trajectory files. "
                        "Default '*.npz' — narrow to e.g. 'IK_*.npz' or 'traj_*_cubic.npz' if "
                        "your dataset folder mixes file types.")
    p.add_argument("--skip-existing", action="store_true", default=False,
                   help="Skip trajectories whose output file already exists. Useful for resuming "
                        "an interrupted run.")
    p.add_argument("--dry-run", action="store_true", default=False,
                   help="Print the commands that would be run without executing them.")
    args, hydra_passthrough = p.parse_known_args()

    # ---- Find trajectory files ---------------------------------------------------
    traj_files = sorted(glob.glob(str(args.dataset / args.pattern)))
    if not traj_files:
        sys.exit(f"[record_dataset] No {args.pattern!r} under {args.dataset}")
    if args.limit:
        traj_files = traj_files[: args.limit]

    # ---- Decide output directory -------------------------------------------------
    # Default: <ckpt_dir>/rollout_dataset/. ckpt path is what record.py uses as log_dir,
    # so this puts dataset outputs alongside the regular `rollout/` folder.
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    out_dir = args.output_dir or Path(ckpt_dir) / "rollout_dataset"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[record_dataset] {len(traj_files)} trajectories from {args.dataset}")
    print(f"[record_dataset] output dir: {out_dir}")

    record_script = str(Path(__file__).parent / "rsl_rl" / "record.py")

    # ---- Loop over trajectories --------------------------------------------------
    started = time.time()
    failures = []
    for i, tf in enumerate(traj_files):
        traj_stem = Path(tf).stem
        out_path = out_dir / f"{traj_stem}.npz"
        if args.skip_existing and out_path.exists():
            print(f"[{i + 1}/{len(traj_files)}] SKIP existing: {out_path}")
            continue

        # Per-trajectory cmd. Hydra passthrough comes BEFORE our env.trajectory_path
        # override so the latter wins (Hydra: rightmost override beats earlier ones).
        cmd = [
            sys.executable, record_script,
            "--task", args.task,
            "--checkpoint", args.checkpoint,
            "--num_envs", str(args.num_envs),
            "--rollout_path", str(out_path),
            "--headless",
            *hydra_passthrough,
            f"env.trajectory_path={tf}",
        ]
        elapsed = time.time() - started
        print(f"\n[{i + 1}/{len(traj_files)}] {traj_stem}  →  {out_path}  "
              f"(elapsed {elapsed:.0f}s)")
        if args.dry_run:
            print("  (dry-run)  " + " ".join(cmd))
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[record_dataset] FAILED on {traj_stem}: returncode={e.returncode}")
            failures.append((traj_stem, e.returncode))
            # Continue to next trajectory — one bad reference shouldn't kill the batch.

    total = time.time() - started
    print(f"\n[record_dataset] done. {len(traj_files) - len(failures)} ok, "
          f"{len(failures)} failed, total {total:.0f}s.")
    if failures:
        print("[record_dataset] failures:")
        for stem, rc in failures:
            print(f"  {stem} (returncode={rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()
