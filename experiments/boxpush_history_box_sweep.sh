#!/bin/bash
# Sweep obs_history_steps in {1, 3} x include_object_obs in {False, True} on the rotate reference.

TRAJECTORY="./reference_trajectories/box_rotate_ur5e/traj_full_refined_20260417_134041_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=5000

HISTORY_STEPS=(1 3)
BOX_OBS=(True False)

for n in "${HISTORY_STEPS[@]}"; do
    for b in "${BOX_OBS[@]}"; do
        box_tag=$([ "$b" = "True" ] && echo "box" || echo "nobox")
        run_name="history_${n}_${box_tag}"
        echo "=== Training: obs_history_steps=$n include_object_obs=$b ==="
        python ./scripts/rsl_rl/train.py \
            --task=$TASK \
            --trajectory_path=$TRAJECTORY \
            --headless \
            --max_iterations=$MAX_ITER \
            env.obs_history_steps=$n \
            env.include_object_obs=$b \
            --run_name=$run_name
    done
done
