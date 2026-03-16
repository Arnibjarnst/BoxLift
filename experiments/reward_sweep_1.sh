#!/bin/bash


# WTASKS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
WTASKS=(0.7 0.8 0.9 1.0)
WREG=(0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5)

for wtask in "${WTASKS[@]}"
do
  for wreg in "${WREG[@]}"
  do
    wtrack=$(awk "BEGIN {printf \"%.1f\", 1 - $wtask}")

    run_name="wtask${wtask}_wtrack${wtrack}_wreg${wreg}"

    python ./scripts/rsl_rl/train.py \
    --task=Template-Boxlift-Direct-v0 \
    --trajectory_path=../planning_through_contact/IK_data/box_lift/IK_20260127_175758.npz \
    --headless \
    env.w_task=$wtask \
    env.w_track=$wtrack \
    env.w_regularization=$wreg \
    --run_name=$run_name
  done
done