#!/bin/bash
# Focused sweep: stricter reset angle × quat kernel shape × failure resampling on/off.
# Hypothesis: tighter termination thresholds are more useful when paired with failure-aware
# resampling (hard segments get exercised more, keeping the policy from being starved of gradient
# in the high-precision regime).
#
# Grid: 3 angle thresholds × 4 quat-sigma configs × 2 failresample states = 24 runs.
# Budget: 1000 iters ≈ 5 min/run → ~2 hours total.

FAILED_RUNS=()

TRAJECTORY="./reference_trajectories/box_rotate_ur5e/traj_full_refined_20260417_134041_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=1000

# --- Axis 1: stricter angle reset threshold (env default 0.5 rad). ---
# Skip 0.5 (already covered by prior sweeps / env default).
ANGLE_THRESHOLDS=(0.3 0.2 0.1)

# --- Axis 2: quat kernel sigma — 2 single + 2 double. ---
# Single kernels cover the "simpler" baseline; double kernels test whether the
# wide+narrow combination still pays off under stricter termination.
declare -A SIGMA_CONFIGS=(
    [single_wide]='[0.1]'
    [single_narrow]='[0.05]'
    [double_baseline]='[0.15,0.05]'
    [double_tight]='[0.1,0.03]'
)
SIGMA_NAMES=(single_wide single_narrow double_baseline double_tight)

# --- Axis 3: failure-aware phase resampling on/off. ---
FAILRESAMPLE=(False True)

tag_num() { echo "$1" | tr -d '.'; }
tag_failres() { [ "$1" = "True" ] && echo "fr" || echo "nofr"; }

run() {
    local tag=$1; shift
    echo "=== $tag ==="
    if ! python ./scripts/rsl_rl/train.py \
        --task=$TASK \
        --trajectory_path=$TRAJECTORY \
        --headless \
        --max_iterations=$MAX_ITER \
        --run_name=$tag \
        "$@"; then
        echo "!!! FAILED: $tag (exit $?) — continuing"
        FAILED_RUNS+=("$tag")
    fi
}

for ang in "${ANGLE_THRESHOLDS[@]}"; do
    for sig_name in "${SIGMA_NAMES[@]}"; do
        for fr in "${FAILRESAMPLE[@]}"; do
            fr_tag=$(tag_failres $fr)
            run "SR_ang$(tag_num $ang)_${sig_name}_${fr_tag}" \
                env.max_obj_angle_from_traj=$ang \
                env.sigma_obj_quat="${SIGMA_CONFIGS[$sig_name]}" \
                env.enable_failure_resampling=$fr
        done
    done
done

echo "=== Sweep complete ==="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Failed runs (${#FAILED_RUNS[@]}):"
    printf '  %s\n' "${FAILED_RUNS[@]}"
else
    echo "All runs succeeded."
fi
