#!/bin/bash
# obs_history_steps at mode A (baseline action formulation)

TRAJECTORY="./reference_trajectories/box_push_ur5e/traj_full_smoothed_gaussian_20260412_180220_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=1000

# --- Sweep 1: obs_history_steps at mode A ---
HISTORY_STEPS=(1 3 5 10)

for n in "${HISTORY_STEPS[@]}"; do
    run_name="history_${n}_mode_A"
    echo "=== Training: obs_history_steps=$n action_mode=A ==="
    python ./scripts/rsl_rl/train.py \
        --task=$TASK \
        --trajectory_path=$TRAJECTORY \
        --headless \
        --max_iterations=$MAX_ITER \
        env.obs_history_steps=$n \
        env.action_mode=A \
        --run_name=$run_name
done
