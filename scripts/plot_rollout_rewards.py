"""Plot per-step rewards / errors / phase metrics from a record.py rollout NPZ.

Usage:
    python scripts/plot_rollout_rewards.py logs/rsl_rl/boxpush/<run>/rollout/output.npz
    python scripts/plot_rollout_rewards.py <path> --prefix Rewards_task --prefix Error
    python scripts/plot_rollout_rewards.py <path> --out reward_plot.png
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


def _decode(key: str) -> str:
    return key[len(EXTRAS_PREFIX):].replace("__", "/")


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
        "--include-total-reward",
        action="store_true",
        help="Add a panel with rewards (total reward per step).",
    )
    args = ap.parse_args()

    data = np.load(Path(args.rollout))
    extras_keys = sorted(k for k in data.files if k.startswith(EXTRAS_PREFIX))
    if not extras_keys:
        print("No extras logged in rollout.")
        return

    # Map decoded metric name → (N,) array.
    metrics = {_decode(k): data[k] for k in extras_keys}
    if args.prefix:
        metrics = {k: v for k, v in metrics.items() if any(k.startswith(p) for p in args.prefix)}
    if not metrics:
        print("No keys matched the requested prefixes.")
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
        rewards = data["rewards"] if "rewards" in data.files else np.zeros(0)
        ax.plot(np.arange(len(rewards)), rewards, lw=1, color="black")
        ax.set_title("total reward per step")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.suptitle(f"Rollout: {args.rollout}", fontsize=10)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=120)
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
