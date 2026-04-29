#!/bin/bash
# New-features + hyperparameter sweep on the rotate trajectory.
# Structure:
#   1) Two plain baselines, no new features (box on/off).
#   2) Two future-obs baselines (box on/off) — used as the reference config for all later axes.
#   3) One axis per new feature/knob, each testing box on/off while other features stay at
#      the future-obs baseline.
#
# Budget: 1000 iters ≈ 5 min/run. Flip RUN_* flags to skip sections.

FAILED_RUNS=()

TRAJECTORY="./reference_trajectories/box_rotate_ur5e/traj_full_refined_20260417_134041_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=1000

# --- Per-axis run toggles (edit to skip sections) ---
RUN_PLAIN_BASELINE=true
RUN_FUTURE_BASELINE=true
RUN_PREV_ACTIONS=true
RUN_CURRICULUM=true
RUN_PHASE_SLOWDOWN=true      # sweeps w_phase_speed ∈ {0.0, 0.1}
RUN_FAILURE_RESAMPLE=true
RUN_PER_JOINT_SCALE=true
RUN_WINDOW_5=true
RUN_REWARD_GRID=true         # w_task/w_track × w_regularization (10 configs × box)
RUN_PPO_SEARCH=true          # lr / entropy_coef / max_grad_norm variants

# --- Shared baseline (applied to every run unless overridden) ---
BASE_WINDOW=1
BASE_W_TASK=0.7
BASE_W_TRACK=0.3
BASE_W_REG=0.4
BASE_W_OBJ_POS=0.5
BASE_W_OBJ_QUAT=0.5
BASE_SIGMA_OBJ_QUAT='[0.15,0.05]'
BASE_ACTION_SCALE=0.05
BASE_INIT_STD=1.0
BASE_ENT_COEF=0.005

# Future-obs set used as the "effective baseline" for non-baseline axes.
FUTURE_OBS='[1,2,4,8,16]'

# Curriculum warmup. 24 policy steps/iter → 1000 iters = 24000 env steps.
# Default: half of training.
WARMUP_STEPS=12000

BOX_OBS=(False True)
tag_box() { [ "$1" = "True" ] && echo "box" || echo "nobox"; }
# Bash string cleanup for run tags: "1e-4" → "1e-4", "0.001" → "0001", "0.5" → "05"
tag_num() { echo "$1" | tr -d '.'; }

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

# ================= Section 1: plain baselines (no new features) — 2 runs =================
# Explicitly disable future_obs_steps — env default is now (1,2,4,8,16), so without this
# override the "plain" baseline would actually include future look-ahead.
if $RUN_PLAIN_BASELINE; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_baseline_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps='[]'
    done
fi

# ================= Section 2: future-obs baseline (new effective baseline) — 2 runs =================
if $RUN_FUTURE_BASELINE; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_future_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS"
    done
fi

# ================= Section 3: previous raw actions in obs — 2 runs =================
if $RUN_PREV_ACTIONS; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_prevact_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.include_prev_actions=True
    done
fi

# ================= Section 4: task/track weight curriculum — 2 runs =================
if $RUN_CURRICULUM; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_curriculum_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.w_task_warmup_steps=$WARMUP_STEPS
    done
fi

# ================= Section 5: phase slowdown — 4 runs (w_phase_speed ∈ {0, 0.1} × box) =================
if $RUN_PHASE_SLOWDOWN; then
    for wps in 0.0 0.1; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_phase_wps$(tag_num $wps)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                env.enable_phase_slowdown=True \
                env.w_phase_speed=$wps
        done
    done
fi

# ================= Section 6: failure-aware phase resampling — 2 runs =================
if $RUN_FAILURE_RESAMPLE; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_failresample_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.enable_failure_resampling=True \
            env.phase_segment_s=1.0 \
            env.phase_resample_alpha=0.1 \
            env.phase_resample_beta=1.0
    done
fi

# ================= Section 7: per-joint action scale — 6 runs (3 profiles × box) =================
# mild:        moderate wrist-heavy, roughly inverse to link-length gain
# aggressive:  strong wrist-heavy, shoulder barely moves
# frozen:      zero shoulder — diagnostic for whether shoulder residual matters
if $RUN_PER_JOINT_SCALE; then
    declare -A PERJOINT_PROFILES=(
        [mild]='[0.02,0.03,0.04,0.08,0.10,0.12]'
        [aggressive]='[0.01,0.01,0.02,0.10,0.15,0.20]'
        [frozen]='[0.0,0.0,0.02,0.10,0.15,0.20]'
    )
    for name in mild aggressive frozen; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_perjoint_${name}_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                env.action_scale="${PERJOINT_PROFILES[$name]}"
        done
    done
fi

# ================= Section 8: obs history window = 5 — 2 runs =================
if $RUN_WINDOW_5; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_window5_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.obs_history_steps=5
    done
fi

# ================= Section 9: task/track ratio × regularization grid — 20 runs =================
# 5 ratios × 2 regs × 2 box = 20. Good coverage of the reward-landscape / stability trade-off.
if $RUN_REWARD_GRID; then
    for rw in "0.5,0.5" "0.7,0.3" "0.8,0.2" "0.9,0.1" "1.0,0.0"; do
        IFS=',' read -r wt wk <<< "$rw"
        for reg in 0.2 0.4; do
            for b in "${BOX_OBS[@]}"; do
                bt=$(tag_box $b)
                run "NF_rew_t$(tag_num $wt)_k$(tag_num $wk)_r$(tag_num $reg)_${bt}" \
                    env.include_object_obs=$b \
                    env.future_obs_steps="$FUTURE_OBS" \
                    env.w_task=$wt \
                    env.w_track=$wk \
                    env.w_regularization=$reg
            done
        done
    done
fi

# ================= Section 10: PPO hyperparameter search — 10 runs =================
# Small exploration around lr / entropy_coef / max_grad_norm. One knob varied at a time.
if $RUN_PPO_SEARCH; then
    # Learning rate: default 3e-4, test 1e-4 (more stable) and 1e-3 (faster convergence as in RobotDancing)
    for lr in 1e-4 1e-3; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_lr_${lr}_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                agent.algorithm.learning_rate=$lr
        done
    done
    # Entropy coef: default 0.005, test 0.001 (let std collapse faster) and 0.01 (more exploration)
    for e in 0.001 0.01; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_entcoef_$(tag_num $e)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                agent.algorithm.entropy_coef=$e
        done
    done
    # Max grad norm: default 1.0, test 0.5 (tighter clipping → less catastrophic updates)
    for g in 0.5; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_gradnorm_$(tag_num $g)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                agent.algorithm.max_grad_norm=$g
        done
    done
fi

# ============================================================================================
# Additional exploratory sections — toggle independently. Placed last so the core sweep runs
# first and these only fire if you let the script run long enough.
# ============================================================================================

RUN_INIT_STD=true        # init_noise_std ∈ {0.3, 0.5, 0.7} × box = 6
RUN_EXTRA_PPO=true       # desired_kl and entropy=0 variants × box = 8
RUN_SIGMA=true           # wider / sharper multi-sigma variants × box = 6
RUN_ROLLOUT=true         # num_steps_per_env / num_mini_batches × box = 4
RUN_NET_ARCH=true        # wider / deeper actor-critic × box = 4
RUN_RESET_NOISE=true     # tighter / looser reset noise × box = 4
RUN_PERTURB=true         # turn perturbations on at two probabilities × box = 4
RUN_KITCHEN=true         # combined configs (kitchen sink variants) × box = 8
RUN_FUTURE_VARIANTS=true # alternate future_obs_steps densities/spacings × box = 6

# ================= Section 11: init_noise_std — 6 runs =================
# Default 1.0 may be too high (see earlier analysis — vf spikes in first iters).
if $RUN_INIT_STD; then
    for s in 0.3 0.5 0.7; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_iniSTD_$(tag_num $s)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                agent.policy.init_noise_std=$s
        done
    done
fi

# ================= Section 12: extra PPO knobs — 8 runs =================
# desired_kl = 0.005 (tighter LR adaptation), 0.02 (looser)
# entropy_coef = 0.0 (no bonus → let std collapse fully), 0.02 (more exploration)
if $RUN_EXTRA_PPO; then
    for kl in 0.005 0.02; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_kl_$(tag_num $kl)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                agent.algorithm.desired_kl=$kl
        done
    done
    for e in 0.0 0.02; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_ent_$(tag_num $e)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                agent.algorithm.entropy_coef=$e
        done
    done
fi

# ================= Section 13: sigma variants on obj rewards — 6 runs =================
# Tests whether the (0.15, 0.05) multi-sigma is a good pick vs alternatives.
if $RUN_SIGMA; then
    declare -A SIGMA_PROFILES=(
        [wide]='[0.3,0.1]'           # broader gradient, less precision pull
        [sharp]='[0.1,0.03]'         # sharper — more precision but less early signal
        [triple]='[0.2,0.08,0.02]'   # three-scale kernel
    )
    for name in wide sharp triple; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_sigmaQ_${name}_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                env.sigma_obj_quat="${SIGMA_PROFILES[$name]}"
        done
    done
fi

# ================= Section 14: rollout buffer / mini-batches — 4 runs =================
# Larger num_steps_per_env = better advantage estimates (less variance), more memory.
# Smaller num_mini_batches = bigger mini-batch = more stable updates.
if $RUN_ROLLOUT; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_steps48_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            agent.num_steps_per_env=48
        run "NF_mb8_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            agent.algorithm.num_mini_batches=8
    done
fi

# ================= Section 15: network architecture — 4 runs =================
# Current [256, 256]. Try RobotDancing's [512, 256, 128] and a wider [512, 512].
if $RUN_NET_ARCH; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_arch_512_256_128_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            agent.policy.actor_hidden_dims='[512,256,128]' \
            agent.policy.critic_hidden_dims='[512,256,128]'
        run "NF_arch_512_512_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            agent.policy.actor_hidden_dims='[512,512]' \
            agent.policy.critic_hidden_dims='[512,512]'
    done
fi

# ================= Section 16: reset noise — 4 runs =================
# Increase (0.1) for broader initial state coverage, decrease (0.01) for cleaner starts.
if $RUN_RESET_NOISE; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        run "NF_resetn_01_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.reset_joint_pos_noise=0.1 \
            env.reset_joint_vel_noise=0.1
        run "NF_resetn_001_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.reset_joint_pos_noise=0.01 \
            env.reset_joint_vel_noise=0.01
    done
fi

# ================= Section 17: perturbations — 4 runs =================
# Currently perturbation_probability=0.0. Test at 0.05 and 0.1 for robustness.
if $RUN_PERTURB; then
    for p in 0.05 0.1; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_perturb_$(tag_num $p)_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="$FUTURE_OBS" \
                env.perturbation_probability=$p
        done
    done
fi

# ================= Section 18: kitchen sink combos — 8 runs =================
# Promising feature combinations to see if stacking helps.
if $RUN_KITCHEN; then
    for b in "${BOX_OBS[@]}"; do
        bt=$(tag_box $b)
        # prev actions + obs history + action scale
        run "NF_all_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.include_prev_actions=True \
            env.obs_history_steps=5 \
            env.action_scale='[0.02,0.03,0.04,0.08,0.10,0.12]'
        # prev actions + obs history
        run "NF_combo_future_curr_perjoint_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.include_prev_actions=True \
            env.obs_history_steps=5 \
        # prev actions + action scale
        run "NF_combo_future_phase_failres_${bt}" \
            env.include_object_obs=$b \
            env.future_obs_steps="$FUTURE_OBS" \
            env.include_prev_actions=True \
            env.action_scale='[0.02,0.03,0.04,0.08,0.10,0.12]'
    done
fi

# ================= Section 19: future_obs_steps variants — 6 runs =================
# Fix step count comparable across variants to isolate spacing vs density effects.
# linear_sparse:  5 evenly-spaced offsets up to ~300 ms look-ahead
# dense_short:    5 consecutive steps (100 ms look-ahead, no gaps)
# dense_long:     10 consecutive steps (200 ms look-ahead, no gaps)
# Baseline (exponential sparse [1,2,4,8,16]) already covered by Section 2.
if $RUN_FUTURE_VARIANTS; then
    declare -A FUTURE_VARIANTS=(
        [linearSparse]='[3,6,9,12,15]'
        [denseShort]='[1,2,3,4,5]'
        [denseMedium]='[1,2,3,4,5,6,7,8,9,10]'
        [denseLong]='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]'
    )
    for name in linearSparse denseShort denseLong; do
        for b in "${BOX_OBS[@]}"; do
            bt=$(tag_box $b)
            run "NF_future_${name}_${bt}" \
                env.include_object_obs=$b \
                env.future_obs_steps="${FUTURE_VARIANTS[$name]}"
        done
    done
fi

echo "=== Sweep complete ==="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Failed runs (${#FAILED_RUNS[@]}):"
    printf '  %s\n' "${FAILED_RUNS[@]}"
else
    echo "All runs succeeded."
fi
