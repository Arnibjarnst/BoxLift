"""Plot per-step rewards / errors / phase metrics from a record.py rollout NPZ.

Usage:
    python scripts/plot_rollout_rewards.py logs/rsl_rl/boxpush/<run>/rollout/output.npz
    python scripts/plot_rollout_rewards.py <path> --prefix Rewards_task --prefix Error
    python scripts/plot_rollout_rewards.py <path> --out reward_plot.png
    python scripts/plot_rollout_rewards.py <path> --hide-zero        # drop always-zero metrics

Default excludes Curriculum/* and RSI/* (constant during a rollout) and VOC/seg*_*_mean
(episode-aggregate, not per-step). Override with --exclude '' to see everything.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Per-step extras metrics are stored in the NPZ under keys like
# "extras__Rewards_task__obj_pos" — the original "Rewards_task/obj_pos" with '/' replaced
# by '__' (NPZ keys can't contain '/'). Decode to the original metric name.
EXTRAS_PREFIX = "extras__"
# Per-env variant: same metric name, but values are (T, N) instead of (T,). Only present
# for envs that emit `extras["log_per_env"]` (boxlift does). Lets --env_idx slice each
# reward component to a single env's trace rather than always showing the env-mean.
PER_ENV_EXTRAS_PREFIX = "extras_per_env__"

# Hide by default — these are constant or trailing-window-aggregate metrics that aren't
# meaningful as per-step rollout curves. Override with --exclude '' to keep everything.
#   Curriculum/alpha     : pinned via force_alpha in record.py setup; flat line.
#   RSI/seg*_focus_prob  : derived from kp_pos, which doesn't decay during a rollout; flat.
#   VOC/seg*_kp_pos      : doesn't change during a rollout (no decay-check fires); flat.
#   VOC/seg*_task_mean   : trailing-window mean of COMPLETED episodes' normalized rewards.
#   VOC/seg*_track_mean    Updated only on env reset, so in a rollout context the curve
#                          spikes at each env's done step rather than tracking per-step state.
DEFAULT_EXCLUDE = ("Curriculum/", "RSI/", "VOC/")


def _decode(key: str) -> str:
    return key[len(EXTRAS_PREFIX):].replace("__", "/")


def _decode_per_env(key: str) -> str:
    return key[len(PER_ENV_EXTRAS_PREFIX):].replace("__", "/")


def _step_mean_reward(data):
    """Per-step total reward, averaged over alive envs when the multi-env schema is in use.
    Order of preference:
      1) `rewards_mean_alive` (precomputed by record.py for multi-env runs)
      2) alive-masked mean of `rewards`+`alive_mask` if both are present and 2D
      3) raw `rewards` for legacy single-env (T,) data
    """
    if "rewards_mean_alive" in data.files:
        return np.asarray(data["rewards_mean_alive"], dtype=np.float32)
    if "rewards" not in data.files:
        return np.zeros(0, dtype=np.float32)
    r = np.asarray(data["rewards"], dtype=np.float32)
    if r.ndim == 1:
        return r  # legacy single-env
    # Multi-env without precomputed mean: try alive_mask, else plain mean.
    if "alive_mask" in data.files:
        m = np.asarray(data["alive_mask"], dtype=np.float32)
        n = m.sum(axis=1)
        return np.where(n > 0, (r * m).sum(axis=1) / np.maximum(n, 1), np.float32("nan"))
    return r.mean(axis=1)


def _step_reward_single_env(data, env_idx):
    """Per-step raw reward for a single env. Errors if the npz is legacy single-env or
    if env_idx is out of range."""
    if "rewards" not in data.files:
        raise SystemExit("No 'rewards' field in the rollout npz.")
    r = np.asarray(data["rewards"], dtype=np.float32)
    if r.ndim == 1:
        raise SystemExit(
            f"--env_idx={env_idx} requires a multi-env rollout (rewards shape (T, N)); "
            f"this npz has legacy single-env rewards shape {r.shape}."
        )
    n_envs = r.shape[1]
    if env_idx < 0 or env_idx >= n_envs:
        raise SystemExit(f"--env_idx={env_idx} out of range; num_envs={n_envs}.")
    return r[:, env_idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout", help="Path to output.npz from record.py")
    ap.add_argument("--out", default=None, help="Save figure to this path instead of showing.")
    ap.add_argument(
        "--prefix",
        action="append",
        default=None,
        help="Only plot keys starting with this prefix (can repeat). Default: all groups.",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=None,
        help=f"Drop metric keys whose name starts with any of these prefixes (repeatable). "
             f"Default = {DEFAULT_EXCLUDE} — see header comment for rationale. "
             f"Pass --exclude '' to disable all default excludes and keep everything.",
    )
    ap.add_argument(
        "--include-total-reward",
        action="store_true",
        help="Add a panel with the per-step total reward (alive-masked mean over envs, "
             "or env_idx's single-env trace if --env_idx is set).",
    )
    ap.add_argument(
        "--env_idx",
        type=int,
        default=None,
        help="Restrict the total-reward panel to a single env's raw per-step reward. "
             "Extras (Rewards_task/*, Error/*, etc.) remain env-mean — the env writes them "
             "with .mean() inside _get_rewards, so they can't be split per-env from the "
             "rollout npz. Use this to inspect a specific env's reward time-series "
             "(e.g. to correlate a force spike with the total reward).",
    )
    ap.add_argument(
        "--hide-zero",
        action="store_true",
        help="Drop metrics whose values are essentially zero everywhere (max|y| < --zero-tol). "
             "Useful for filtering out reward terms with weight=0 that still get logged.",
    )
    ap.add_argument(
        "--zero-tol",
        type=float,
        default=1e-9,
        help="Threshold for --hide-zero: a metric is hidden if max(|y|) < this value. Default 1e-9.",
    )
    args = ap.parse_args()

    data = np.load(Path(args.rollout))
    extras_keys = sorted(k for k in data.files if k.startswith(EXTRAS_PREFIX)
                         and not k.startswith(PER_ENV_EXTRAS_PREFIX))
    per_env_keys = sorted(k for k in data.files if k.startswith(PER_ENV_EXTRAS_PREFIX))

    if not extras_keys and not per_env_keys:
        print("No extras logged in rollout.")
        return

    # Choose source. With --env_idx set AND per-env extras present, slice each (T, N)
    # array to (T,) for the requested env so every component panel shows that env's
    # contribution rather than the cohort mean. Components missing from the per-env
    # stream fall back to the env-mean version.
    metrics = {}
    extras_source = "env-mean"
    if args.env_idx is not None and per_env_keys:
        n_envs = int(data["num_envs"]) if "num_envs" in data.files else None
        if n_envs is None:
            raise SystemExit("--env_idx provided but rollout has no `num_envs` field.")
        if args.env_idx < 0 or args.env_idx >= n_envs:
            raise SystemExit(f"--env_idx={args.env_idx} out of range; num_envs={n_envs}.")
        for k in per_env_keys:
            metrics[_decode_per_env(k)] = data[k][:, args.env_idx]
        # Fill in any keys that only exist as env-mean (e.g. VOC/seg* — though those are
        # in the default excludes anyway).
        for k in extras_keys:
            name = _decode(k)
            if name not in metrics:
                metrics[name] = data[k]
        extras_source = f"per-env (env_idx={args.env_idx})"
    else:
        for k in extras_keys:
            metrics[_decode(k)] = data[k]
        if args.env_idx is not None:
            print(f"[warn] --env_idx={args.env_idx} requested but rollout has no per-env "
                  f"extras (env didn't emit `log_per_env`). Falling back to env-mean for "
                  f"all components; only the total-reward panel will reflect env_idx.")
    print(f"[plot] extras source: {extras_source}")
    if args.prefix:
        metrics = {k: v for k, v in metrics.items() if any(k.startswith(p) for p in args.prefix)}

    # Apply exclude filter — defaults drop constant / non-per-step metrics. A user-supplied
    # empty string disables the defaults so everything is kept.
    excludes = args.exclude if args.exclude is not None else list(DEFAULT_EXCLUDE)
    excludes = [e for e in excludes if e]  # filter out '' which signals "no excludes"
    if excludes:
        before = len(metrics)
        metrics = {k: v for k, v in metrics.items() if not any(k.startswith(e) for e in excludes)}
        if before != len(metrics):
            print(f"[exclude] dropped {before - len(metrics)} of {before} metric(s) "
                  f"matching prefixes {excludes} — pass --exclude '' to keep them.")

    # Drop metrics that are essentially zero for every step, if requested. Reports the
    # count so it's obvious when --hide-zero hid something.
    if args.hide_zero:
        before = len(metrics)
        kept, hidden = {}, []
        for k, v in metrics.items():
            finite = np.asarray(v, dtype=np.float64)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0 or np.abs(finite).max() < args.zero_tol:
                hidden.append(k)
            else:
                kept[k] = v
        metrics = kept
        if hidden:
            print(f"[hide-zero] dropped {len(hidden)} of {before} metric(s) "
                  f"(|y| < {args.zero_tol}): {', '.join(hidden)}")

    if not metrics:
        print("No keys matched the requested filters.")
        return

    # Group by everything before the first '/'.
    groups = defaultdict(list)
    for k in metrics:
        group = k.split("/", 1)[0] if "/" in k else "_misc"
        groups[group].append(k)
    group_order = sorted(groups.keys())

    n_steps = len(next(iter(metrics.values())))
    t = np.arange(n_steps)

    n_panels = len(group_order) + (1 if args.include_total_reward else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.6 * n_panels), sharex=True, squeeze=False)
    axes = axes.flat

    for ax, group in zip(axes, group_order):
        for k in groups[group]:
            ys = metrics[k]
            label = k.split("/", 1)[1] if "/" in k else k
            ax.plot(t, ys, label=label, lw=1)
        ax.set_title(group)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="x-small", ncol=2)

    if args.include_total_reward:
        ax = axes[len(group_order)]
        if args.env_idx is not None:
            rewards = _step_reward_single_env(data, args.env_idx)
            ax.set_title(f"total reward per step (env_idx={args.env_idx}, raw)")
        else:
            rewards = _step_mean_reward(data)
            ax.set_title("total reward per step (alive-masked mean over envs)")
        ax.plot(np.arange(len(rewards)), rewards, lw=1, color="black")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("step")
    n_envs = int(data["num_envs"]) if "num_envs" in data.files else 1
    suptitle = f"Rollout: {args.rollout}  (num_envs={n_envs})"
    if args.env_idx is not None:
        suptitle += f"   [env_idx={args.env_idx}, extras source: {extras_source}]"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
