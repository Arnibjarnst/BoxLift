#!/bin/bash
# Sweep: train RL models with different kp/kd/action_scale,
# then evaluate each in URSim with different servoJ gain/lookahead.

TRAJECTORY="./reference_trajectories/box_push_ur5e/IK_20260408_110350.npz"
TASK="Template-Boxpush-Direct-v0"
MAX_ITER=300

# --- Training sweep parameters ---
KPS=(100.0 150.0 200.0 300.0 400.0)
KD_RATIOS=(0.1 0.15 0.2 0.25)
ACTION_SCALES=(0.05 0.1 0.15 0.2 0.25)

# --- URSim evaluation parameters ---
GAINS=(100.0 200.0 400.0)
LOOKAHEADS=(0.05 0.1 0.2)

LOG_ROOT="logs/rsl_rl/boxpush"

# ===================================
# Phase 0: Baseline (pure trajectory)
# ===================================
echo "=== Phase 3: Baseline (action_scale=0) ==="

# Use any existing run dir for the ONNX model (output is zeroed anyway)
baseline_run_dir=$(ls -dt "$LOG_ROOT"/*_kp* 2>/dev/null | head -1)
if [ -z "$baseline_run_dir" ] || [ ! -f "$baseline_run_dir/exported/policy.onnx" ]; then
    echo "No trained run found for baseline — skipping"
else
    for servo_gain in "${GAINS[@]}"; do
        for servo_lookahead in "${LOOKAHEADS[@]}"; do
            eval_npz=$(ls "$baseline_run_dir"/ur_rtde_logs/*_gain${servo_gain}_la${servo_lookahead}_as0*.npz 2>/dev/null | head -1)
            if [ -n "$eval_npz" ]; then
                echo "Skipping baseline gain=$servo_gain lookahead=$servo_lookahead (npz exists)"
                continue
            fi

            echo "Baseline: gain=$servo_gain lookahead=$servo_lookahead"
            python ./scripts/ur_rtde_real_time.py \
                --run_dir="$baseline_run_dir" \
                --action_scale=0 \
                --gain=$servo_gain \
                --lookahead=$servo_lookahead \
                --no-real_robot
        done
    done
fi

# ===================================
# Phase 1: Train models
# ===================================
echo "=== Phase 1: Training ==="

for kp in "${KPS[@]}"; do
    for kd_ratio in "${KD_RATIOS[@]}"; do
        kd=$(echo "$kp * $kd_ratio" | bc)

        for scale in "${ACTION_SCALES[@]}"; do
            run_name="kp${kp}_kd${kd}_scale${scale}"

            # Skip training if ONNX already exists for this run
            existing_dir=$(ls -dt "$LOG_ROOT"/*_${run_name} 2>/dev/null | head -1)
            if [ -n "$existing_dir" ] && [ -f "$existing_dir/exported/policy.onnx" ]; then
                echo "Skipping training $run_name (ONNX already exists)"
                continue
            fi

            echo "Training: $run_name"
            python ./scripts/rsl_rl/train.py \
                --task=$TASK \
                --trajectory_path=$TRAJECTORY \
                --headless \
                --max_iterations=$MAX_ITER \
                env.kp=$kp \
                env.kd=$kd \
                env.action_scale=$scale \
                --run_name=$run_name
        done
    done
done

# ===================================
# Phase 2: Evaluate in URSim
# ===================================
echo "=== Phase 2: URSim Evaluation ==="

for kp in "${KPS[@]}"; do
    for kd_ratio in "${KD_RATIOS[@]}"; do
        kd=$(echo "$kp * $kd_ratio" | bc)
        for scale in "${ACTION_SCALES[@]}"; do
            run_name="kp${kp}_kd${kd}_scale${scale}"

            # Find the run directory matching this run name (take latest if multiple)
            run_dir=$(ls -dt "$LOG_ROOT"/*_${run_name} 2>/dev/null | head -1)
            if [ -z "$run_dir" ] || [ ! -f "$run_dir/exported/policy.onnx" ]; then
                echo "Skipping $run_name (not found or no ONNX)"
                continue
            fi

            for servo_gain in "${GAINS[@]}"; do
                for servo_lookahead in "${LOOKAHEADS[@]}"; do
                    # Skip if result .npz already exists for this gain/lookahead combo
                    eval_npz=$(ls "$run_dir"/ur_rtde_logs/*_gain${servo_gain}_la${servo_lookahead}.npz 2>/dev/null | head -1)
                    if [ -n "$eval_npz" ]; then
                        echo "Skipping eval $run_name gain=$servo_gain lookahead=$servo_lookahead (npz exists)"
                        continue
                    fi

                    echo "Eval: $run_name gain=$servo_gain lookahead=$servo_lookahead"
                    python ./scripts/ur_rtde_real_time.py \
                        --run_dir="$run_dir" \
                        --gain=$servo_gain \
                        --lookahead=$servo_lookahead \
                        --no-real_robot
                done
            done
        done
    done
done



echo "=== Sweep complete ==="
echo "Results in each run's ur_rtde_logs/ directory"
echo "Baseline files have _as0 suffix"
