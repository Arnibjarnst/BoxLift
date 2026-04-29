#!/bin/bash
# Feature ablation — turn one feature off at a time vs the current working baseline.
# Each run is a single policy training; compare tensorboard to see which features matter.
#
# Baseline: all features enabled (matches current env_cfg defaults — action_mode=D with
# alpha_warmup curriculum, include_absolute_obs, EE-box-rel gated tracking, obj vel
# rewards, failure resampling).
#
# Ablations (one per run):
#   1) no_abs_obs         — include_absolute_obs=False
#   2) no_obj_vel         — w_obj_lin_vel=w_obj_ang_vel=0, pos/quat scaled to keep Σ=1
#   3) no_fail_resample   — enable_failure_resampling=False
#   4) action_A_no_curr   — action_mode=A, alpha_warmup_steps=0
#   5) action_B_no_curr   — action_mode=B, alpha_warmup_steps=0
#   6) action_C_no_curr   — action_mode=C, alpha_warmup_steps=0
#   7) action_D_no_curr   — action_mode=D, alpha_warmup_steps=0
#   8) no_eef_box_rel     — disable gated rel tracking; absolute trackers active always
#                           (achieved by setting eef_box_gate_obj_vel_eps huge, which
#                            zeroes the gate mask → abs_gate=1 everywhere).
#   9) no_dphase          — enable_phase_slowdown=False; action_space drops to 6 (no
#                           dphase action), dphase=1 always, no pause penalty, wall-clock
#                           episode cap = nominal trajectory duration.
#  10) single_sigma_quat  — sigma_obj_quat = 0.05 / 0.2 (narrow only + wide only).
#                           Tests whether the multi-sigma trick was actually responsible
#                           for the 8× quat_err improvement.
#
# Kept constant across all runs: reset noise, kernel sigmas, all other reward weights.

FAILED_RUNS=()

TRAJECTORY="./reference_trajectories/box_rotate_ur5e/traj_full_refined_20260427_175616_cubic.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=2000

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

# --- 0. Baseline (all features on, current cfg defaults) ---
run "ABL_baseline"

# --- 1. Absolute obs off ---
run "ABL_no_abs_obs" \
    env.include_absolute_obs=False

# --- 2. Object velocity rewards off, rebalance pos/quat to keep Σ(task sub-weights)=1 ---
# Current: pos=0.3, quat=0.4, lin_vel=0.15, ang_vel=0.15 (sum=1.0, ratio pos:quat = 3:4)
# Scaled up to fill the 0.3 vel budget while preserving the 3:4 ratio:
#   new_pos  = 0.3 / 0.7 = 0.4286
#   new_quat = 0.4 / 0.7 = 0.5714
run "ABL_no_obj_vel" \
    env.w_obj_lin_vel=0.0 \
    env.w_obj_ang_vel=0.0 \
    env.w_obj_pos=0.4286 \
    env.w_obj_quat=0.5714

# # --- 3. Failure-aware phase resampling off (uniform start sampling instead) ---
# run "ABL_no_fail_resample" \
#     env.enable_failure_resampling=False

# --- 4–7. Action modes without curriculum ---
# Mode A/B/C/D each tested as a "pure" run (no α ramp, action blend and reward weights
# locked at their end-of-curriculum values). Good comparison against curriculum-on D.
run "ABL_action_A_no_curr" \
    env.action_mode=A \
    env.alpha_warmup_steps=0

run "ABL_action_B_no_curr" \
    env.action_mode=B \
    env.alpha_warmup_steps=0

run "ABL_action_C_no_curr" \
    env.action_mode=C \
    env.alpha_warmup_steps=0

# Pure mode D (original formulation: q_curr + planner_pd + scale·a). Achieved via
# force_alpha=0 (full planner_pd) + action_alpha_floor=1 (full residual scale). Reward
# weights pinned to end values via w_task_start/w_track_start overrides so we test the
# action formulation in isolation — without the track-heavy α=0 reward shape.
# (Note: the previous "ABL_action_D_no_curr" with alpha_warmup_steps=0 collapsed to
#  mode C exactly because mode D at α=1 is mode C by construction — that's what the
#  curriculum endpoint was designed to be.)
run "ABL_action_D_pure" \
    env.action_mode=D \
    env.force_alpha=0.0 \
    env.action_alpha_floor=1.0 \
    env.w_task_start=0.8 \
    env.w_track_start=0.2

# --- 8. EE-box-rel reward + gating off ---
# Disable the rel reward AND force the gate off everywhere so absolute trackers
# (w_eef_pos / w_joint_pos) are active during contact phases too. Setting the velocity
# threshold absurdly high makes no reference step register as "moving", so the dilated
# mask stays all-False → gate=0 → abs_gate=1 throughout the trajectory.
run "ABL_no_eef_box_rel" \
    env.w_eef_box_rel_pos=0.0 \
    env.w_eef_box_rel_quat=0.0 \
    env.eef_box_gate_obj_vel_eps=1e10

# --- 9. Phase-slowdown off (no dphase action, no pause mechanism) ---
# action_space recomputes to 6 in __post_init__; dphase stays 1.0 throughout the episode,
# the cumulative-quadratic pause penalty never fires, and episode length caps at the
# nominal trajectory duration instead of 3× it.
run "ABL_no_dphase" \
    env.enable_phase_slowdown=False

# --- 10. Single-sigma quat kernel (drops the wide kernel half of the multi-sigma) ---
# Tests whether the multi-sigma (0.05, 0.2) was actually responsible for the 8× quat_err
# improvement we saw, or whether single-sigma 0.05 would have converged anyway.
run "ABL_single_small_sigma_quat" \
    env.sigma_obj_quat='[0.05]'

run "ABL_single_large_sigma_quat" \
    env.sigma_obj_quat='[0.2]'

echo "=== Ablation sweep complete ==="
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Failed runs (${#FAILED_RUNS[@]}):"
    printf '  %s\n' "${FAILED_RUNS[@]}"
else
    echo "All runs succeeded."
fi
