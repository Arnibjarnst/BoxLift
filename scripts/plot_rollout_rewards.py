"""Plot per-step rewards / errors / phase metrics from a record.py rollout JSON.

Usage:
    python scripts/plot_rollout_rewards.py logs/rsl_rl/boxpush/<run>/rollout/output.json
    python scripts/plot_rollout_rewards.py <path> --prefix Rewards_task --prefix Error
    python scripts/plot_rollout_rewards.py <path> --out reward_plot.png
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout", help="Path to output.json from record.py")
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
        help="Add a panel with output['rewards'] (total reward per step).",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.rollout).read_text())
    extras = data.get("extras") or []
    if not extras:
        print("No extras logged in rollout.")
        return

    # Collect keys, optionally filtered by prefix.
    all_keys = sorted({k for e in extras for k in e.keys()})
    if args.prefix:
        all_keys = [k for k in all_keys if any(k.startswith(p) for p in args.prefix)]
    if not all_keys:
        print("No keys matched the requested prefixes.")
        return

    # Group by everything before the first '/'.
    groups = defaultdict(list)
    for k in all_keys:
        group = k.split("/", 1)[0] if "/" in k else "_misc"
        groups[group].append(k)
    group_order = sorted(groups.keys())

    n_steps = len(extras)
    t = list(range(n_steps))

    n_panels = len(group_order) + (1 if args.include_total_reward else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.6 * n_panels), sharex=True, squeeze=False)
    axes = axes.flat

    for ax, group in zip(axes, group_order):
        for k in groups[group]:
            ys = [e.get(k, float("nan")) for e in extras]
            label = k.split("/", 1)[1] if "/" in k else k
            ax.plot(t, ys, label=label, lw=1)
        ax.set_title(group)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize="x-small", ncol=2)

    if args.include_total_reward:
        ax = axes[len(group_order)]
        rewards = data.get("rewards") or []
        ax.plot(range(len(rewards)), rewards, lw=1, color="black")
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
