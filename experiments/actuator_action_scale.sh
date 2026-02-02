#!/bin/bash

ACTUATORS=("IdealPD" "Implicit")
KPS=(100 200 400 800 1600 3200)
SCALES=(0.025 0.05 0.1 0.25 0.5)

for actuator in "${ACTUATORS[@]}"
do
  for kp in "${KPS[@]}"
  do
    kd=$((kp / 20))

    for scale in "${SCALES[@]}"
    do
      run_name="${actuator}_kp${kp}_kd${kd}_scale${scale}"

      python ./scripts/rsl_rl/train.py \
        --task=Template-Boxlift-Direct-v0 \
        --trajectory_path=../planning_through_contact/IK_data/box_lift/IK_20260127_175758.npz \
        --headless \
        env.actuator_type=$actuator \
        env.kp=$kp \
        env.kd=$kd \
        env.action_scale=$scale \
        --run_name=$run_name
    done
  done
done