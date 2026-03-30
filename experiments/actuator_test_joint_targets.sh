#!/bin/bash

ACTUATORS=("IdealPD" "DelayedPD" "Implicit")
KPS=(100.0 200.0 400.0)
KP_KD_RATIOS=(0.05 0.1 0.2)
SCALES=(0.1 0.2 0.3)

for actuator in "${ACTUATORS[@]}"
do
  for kp in "${KPS[@]}"
  do
    # Loop through each ratio to calculate KD dynamically
    for ratio in "${KP_KD_RATIOS[@]}"
    do
      # Use 'bc' for floating point multiplication
      kd=$(echo "$kp * $ratio" | bc -l)

      for scale in "${SCALES[@]}"
      do
        # We sanitize the run_name because dots (e.g., 0.05) can mess up some loggers
        run_name="${actuator}_kp${kp}_kd${kd}_scale${scale}"

        echo "-------------------------------------------------------"
        echo "Running: $run_name (KD calculated as $kd)"
        echo "-------------------------------------------------------"

        python ./scripts/rsl_rl/train.py \
          --task=Follow-Joint-Targets \
          --trajectory_path=/home/arni/planning_through_contact/RL_data/box_lift_ur5e/IK_20260328_155357.npz \
          --headless \
          env.actuator_type="$actuator" \
          env.kp="$kp" \
          env.kd="$kd" \
          env.action_scale="$scale" \
          --run_name="$run_name"
      done
    done
  done
done