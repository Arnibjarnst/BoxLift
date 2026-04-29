#!/bin/bash
# Multi-axis sweep on rotate trajectory. One variable at a time against a fixed baseline.
# Each axis is self-contained — comment out sections to narrow the sweep.
#
# Budget: ~25 min per 5000-iter run. 50 runs = ~21 hours. Total axes below = 25 configs × 2 box variants = 50.
# Individual run failures are logged to FAILED_RUNS and do not abort the sweep.

FAILED_RUNS=()

TRAJECTORY="./reference_trajectories/box_rotate_ur5e/traj_full_refined_20260417_134041_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=5000

# --- Baseline (applied to every run unless overridden) ---
# Match the h3_box winner from the last sweep: window=3, box-on side is toggled per run.
BASE_WINDOW=3
BASE_W_TASK=1.0
BASE_W_TRACK=0.0
BASE_W_REG=0.2
BASE_W_OBJ_POS=0.5
BASE_W_OBJ_QUAT=0.5
BASE_SIGMA_OBJ_QUAT='[0.2,0.05]'   # hydra list syntax
BASE_ACTION_SCALE=0.05
BASE_INIT_STD=1.0
BASE_ENT_COEF=0.005

BOX_OBS=(False True)

run() {
    local tag=$1; shift
    echo "=== $tag ==="
    if ! python ./scripts/rsl_rl/train.py \
        --task=$TASK \
        --trajectory_path=$TRAJECTORY \
        --headless \
        --max_iterations=$MAX_ITER \
        --run_name=$tag \
        env.obs_history_steps=$BASE_WINDOW \
        env.w_task=$BASE_W_TASK \
        env.w_track=$BASE_W_TRACK \
        env.w_regularization=$BASE_W_REG \
        env.w_obj_pos=$BASE_W_OBJ_POS \
        env.w_obj_quat=$BASE_W_OBJ_QUAT \
        env.sigma_obj_quat=$BASE_SIGMA_OBJ_QUAT \
        env.action_scale=$BASE_ACTION_SCALE \
        agent.policy.init_noise_std=$BASE_INIT_STD \
        agent.algorithm.entropy_coef=$BASE_ENT_COEF \
        "$@"; then
        echo "!!! FAILED: $tag (exit $?) — continuing"
        FAILED_RUNS+=("$tag")
    fi
}

tag_box() { [ "$1" = "True" ] && echo "box" || echo "nobox"; }

# ================= Axis 1: obj_pos / obj_quat ratio (sum=1) — 4 values × 2 = 8 runs =================
# Hypothesis: rotation is underweighted. Test pushing the ratio toward quat.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for rw in "0.7,0.3" "0.5,0.5" "0.3,0.7" "0.2,0.8"; do
        IFS=',' read -r wp wq <<< "$rw"
        run "pq_${wp}_${wq}_${bt}" env.include_object_obs=$b env.w_obj_pos=$wp env.w_obj_quat=$wq
    done
done

# ================= Axis 2: narrow sigma magnitude — 3 values × 2 = 6 runs =================
# Hypothesis: the narrow-kernel σ controls how hard the policy is pulled in close.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for ns in "[0.2,0.03]" "[0.2,0.05]" "[0.2,0.1]"; do
        tag_s=$(echo $ns | tr -d '[]' | tr ',' '_')
        run "sigq_${tag_s}_${bt}" env.include_object_obs=$b env.sigma_obj_quat=$ns
    done
done

# ================= Axis 3: action_scale — 3 values × 2 = 6 runs =================
# Hypothesis: residual lacks authority for fine rotation control.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for a in 0.05 0.1 0.2; do
        run "actscl_${a}_${bt}" env.include_object_obs=$b env.action_scale=$a
    done
done

# ================= Axis 4: w_regularization — 3 values × 2 = 6 runs =================
# Hypothesis: current 0.2 may be too loose (action_rate spikes) or too tight (over-smoothing).
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for wr in 0.1 0.2 0.4; do
        run "reg_${wr}_${bt}" env.include_object_obs=$b env.w_regularization=$wr
    done
done

# ================= Axis 5: init_noise_std — 3 values × 2 = 6 runs =================
# Hypothesis: 1.0 causes catastrophic first updates; lower may stabilize and help collapse.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for s in 0.5 1.0 1.5; do
        run "istd_${s}_${bt}" env.include_object_obs=$b agent.policy.init_noise_std=$s
    done
done

# ================= Axis 6: entropy_coef — 3 values × 2 = 6 runs =================
# Hypothesis: mean_noise_std stuck at 0.3; lower coef lets it collapse further.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for e in 0.001 0.005 0.01; do
        run "entcoef_${e}_${bt}" env.include_object_obs=$b agent.algorithm.entropy_coef=$e
    done
done

# ================= Axis 7: obs history window — 3 values × 2 = 6 runs =================
# Re-run of prior sweep with the new reward shape, now including window=5.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for n in 1 3 5; do
        run "win_${n}_${bt}" env.include_object_obs=$b env.obs_history_steps=$n
    done
done

# ================= Axis 8: task/track ratio (sum=1) — 2 values × 2 = 4 runs =================
# Hypothesis: a small track weight helps keep EE near nominal without squashing rotation learning.
for b in "${BOX_OBS[@]}"; do
    bt=$(tag_box $b)
    for rw in "1.0,0.0" "0.9,0.1" "0.8,0.2"; do
        IFS=',' read -r wt wk <<< "$rw"
        run "tt_${wt}_${wk}_${bt}" env.include_object_obs=$b env.w_task=$wt env.w_track=$wk
    done
done

echo "=== Sweep complete ==="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Failed runs (${#FAILED_RUNS[@]}):"
    printf '  %s\n' "${FAILED_RUNS[@]}"
else
    echo "All runs succeeded."
fi
