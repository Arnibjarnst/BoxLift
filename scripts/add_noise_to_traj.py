#!/usr/bin/env python
"""Add Gaussian noise to the joint reference fields of a trajectory .npz.

Use case: test how much reference noise the policy can absorb. Train one policy on
the clean trajectory, then train others on noised copies at increasing strengths,
and measure where success rate collapses. Box reference (`obj_poses`, `obj_vel`,
`EE_poses*`, `arm_pose*`, `dt`, `object_*`) is left untouched so the task spec is
constant across the sweep — only the joint side of the reference drifts.

By default, position-like joint fields are noised (`joints`, `joints_target`, and
their `_l` / `_r` variants for dual-arm). Velocity-like fields are kept clean
unless `--vel_noise_std` is set explicitly.

Two noise temporal structures:
  --smooth_tau 0      → i.i.d. per-step (default): every (t, joint, field) is an
                        independent Gaussian draw. High-frequency, unsmooth jitter —
                        most pessimistic test of "policy robustness to reference noise."
  --smooth_tau >0     → OU-smoothed: per-channel Ornstein-Uhlenbeck process with
                        correlation time `tau` (seconds). Models a slow drift /
                        calibration error rather than per-step jitter — closer to
                        what a real planner's reference error looks like. Stationary
                        marginal std equals --noise_std regardless of tau (the
                        smoothing renormalizes by √(1-α²)).

Independence across CHANNELS is preserved either way — joints, joints_target, and
each per-joint dim get independent noise sequences. Smoothing only correlates across
time within a single channel.

Usage:
    # i.i.d. (per-step jitter) — the original behavior
    python scripts/add_noise_to_traj.py --input <traj.npz> --noise_std 0.05

    # smoothed (slow drift) — tau=0.5s ≈ 25 steps at dt=0.02s
    python scripts/add_noise_to_traj.py --input <traj.npz> --noise_std 0.05 --smooth_tau 0.5

    # both pos + vel noise, custom output, fixed seed
    python scripts/add_noise_to_traj.py --input <traj.npz> --noise_std 0.05 \
        --vel_noise_std 0.1 --smooth_tau 0.2 --output /tmp/noisy.npz --seed 7
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


# Position-like joint fields (units: rad). Noised by default.
POS_FIELDS = (
    "joints", "joints_l", "joints_r",
    "joints_target", "joints_target_l", "joints_target_r",
)
# Velocity-like joint fields (units: rad/s). Noised only when --vel_noise_std > 0.
VEL_FIELDS = (
    "joint_vel", "joint_vel_l", "joint_vel_r",
    "joint_target_vel", "joint_target_vel_l", "joint_target_vel_r",
)


def _gen_noise(shape: tuple, std: float, dt: float, tau: float, rng: np.random.Generator) -> np.ndarray:
    """Generate Gaussian noise of given `shape` with stationary marginal std = `std`.

    shape[0] is the time dimension. All other dims are independent channels.

    - tau == 0: i.i.d. per-step Gaussian (white noise), each entry ~ N(0, std²).
    - tau >  0: per-channel Ornstein-Uhlenbeck (AR(1)) process. Stationary marginal
      remains N(0, std²); autocorrelation E[x_t x_{t+k}] = std² · α^k where
      α = exp(-dt/tau). Implemented as:
          x_0 = std · z_0
          x_t = α · x_{t-1} + std·√(1-α²) · z_t       (z_t ~ N(0, I))
      The √(1-α²) renormalization is what keeps the stationary variance = std²
      independent of tau — so sweeping tau at fixed std is a clean ablation of
      "noise temporal structure" without confounding the marginal magnitude.
    """
    if tau <= 0.0:
        return std * rng.standard_normal(shape)
    if dt <= 0.0:
        raise ValueError(f"smoothing requires a positive dt; got dt={dt}")
    alpha = float(np.exp(-dt / tau))
    z = rng.standard_normal(shape)
    out = np.empty(shape, dtype=np.float64)
    out[0] = std * z[0]
    drive = std * np.sqrt(max(1.0 - alpha * alpha, 0.0))
    for t in range(1, shape[0]):
        out[t] = alpha * out[t - 1] + drive * z[t]
    return out


def main():
    p = argparse.ArgumentParser(
        description="Add i.i.d. Gaussian noise to joint reference fields of a .npz trajectory.",
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Path to input trajectory .npz.")
    p.add_argument("--noise_std", required=True, type=float,
                   help="Std of i.i.d. Gaussian noise on joint POSITIONS (rad). "
                        "Per-step, per-joint, per-field independent draws.")
    p.add_argument("--vel_noise_std", type=float, default=0.0,
                   help="Std of i.i.d. Gaussian noise on joint VELOCITIES (rad/s). "
                        "Default 0.0 → vel fields left untouched. Set to >0 to noise them "
                        "(uses the same --smooth_tau timescale as the position fields).")
    p.add_argument("--smooth_tau", type=float, default=0.0,
                   help="OU correlation time in seconds. Default 0 → i.i.d. per-step noise. "
                        "Positive values produce smoothed (low-frequency) drift; stationary "
                        "marginal std remains --noise_std regardless of tau. Reasonable values: "
                        "0.1-1.0 s for 'slow drift' tests at typical 0.02 s dt.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output path. Default: <input_stem>_noise<std>.npz next to input.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility. Different seeds → different noise "
                        "realizations of the same std (use this to ablate noise samples "
                        "from noise magnitude).")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"input not found: {args.input}")
    # Output filename encodes both std and tau when smoothing is on, so a noise sweep
    # over multiple (std, tau) combos doesn't clobber files.
    tau_tag = f"_tau{args.smooth_tau:g}" if args.smooth_tau > 0 else ""
    out_path = args.output or args.input.with_name(
        f"{args.input.stem}_noise{args.noise_std:g}{tau_tag}.npz"
    )
    if out_path.resolve() == args.input.resolve():
        raise SystemExit(f"refusing to overwrite input: {args.input}")

    rng = np.random.default_rng(args.seed)
    d = np.load(args.input)
    # Copy everything; we'll overwrite the joint fields below.
    out = {k: d[k] for k in d.files}

    # dt comes from the trajectory file (set by IK / planning scripts). Fall back to
    # 0.02 with a warning — boxhinge / boxlift default physics_dt × decimation.
    if "dt" in d.files:
        dt = float(d["dt"])
    else:
        dt = 0.02
        if args.smooth_tau > 0:
            print(f"[warn] no 'dt' key in input; assuming dt={dt} for OU smoothing.")

    noised_pos, noised_vel, skipped = [], [], []
    for k in POS_FIELDS:
        if k in d.files:
            arr = d[k].astype(np.float64, copy=True)
            arr += _gen_noise(arr.shape, args.noise_std, dt, args.smooth_tau, rng)
            out[k] = arr
            noised_pos.append((k, arr.shape))
    if args.vel_noise_std > 0.0:
        for k in VEL_FIELDS:
            if k in d.files:
                arr = d[k].astype(np.float64, copy=True)
                arr += _gen_noise(arr.shape, args.vel_noise_std, dt, args.smooth_tau, rng)
                out[k] = arr
                noised_vel.append((k, arr.shape))
    else:
        for k in VEL_FIELDS:
            if k in d.files:
                skipped.append(k)

    if not noised_pos and not noised_vel:
        raise SystemExit(
            f"No joint fields found in {args.input}. Expected at least one of: "
            f"{POS_FIELDS + VEL_FIELDS}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    print(f"Input:  {args.input}  ({len(d.files)} keys)")
    print(f"Output: {out_path}")
    print(f"Seed:   {args.seed}  dt={dt}")
    if args.smooth_tau > 0:
        alpha = float(np.exp(-dt / args.smooth_tau))
        print(f"Smoothing: OU process, tau={args.smooth_tau}s, "
              f"α=exp(-dt/tau)={alpha:.4f} (effective horizon ≈ {args.smooth_tau / dt:.1f} steps)")
    else:
        print("Smoothing: none (i.i.d. per-step noise)")
    print(f"Noised joint POSITIONS (std={args.noise_std} rad):")
    for k, sh in noised_pos:
        print(f"  {k:25s} shape={sh}")
    if noised_vel:
        print(f"Noised joint VELOCITIES (std={args.vel_noise_std} rad/s):")
        for k, sh in noised_vel:
            print(f"  {k:25s} shape={sh}")
    elif skipped:
        print(f"Velocity fields left clean ({len(skipped)} skipped). "
              f"Pass --vel_noise_std >0 to noise them too.")


if __name__ == "__main__":
    main()
