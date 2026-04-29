TRAJECTORY="./reference_trajectories/box_push_ur5e/traj_full_smoothed_gaussian_20260412_180220_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=1000


ACTION_MODES=(A B C D)

for mode in "${ACTION_MODES[@]}"; do
    run_name="history_1_mode_${mode}"
    echo "=== Training: obs_history_steps=1 action_mode=$mode ==="
    python ./scripts/rsl_rl/train.py \
        --task=$TASK \
        --trajectory_path=$TRAJECTORY \
        --headless \
        --max_iterations=$MAX_ITER \
        env.action_mode=$mode \
        --run_name=$run_name
done

echo "=== Sweep complete ==="
