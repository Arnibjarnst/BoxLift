#!/bin/bash

make_layers() {
  local n=$1
  local d=$2
  printf "%s," $(yes "$d" | head -n "$n")
}

for layers in 2 3; do
  for dim in 16 32 64 128 256; do
    layer_str=$(make_layers "$layers" "$dim")
    layer_str="${layer_str%,}"   # strip trailing comma

    run_name="hidden_dims=[$layer_str]"

    python ./scripts/rsl_rl/train.py \
      --task=Template-Boxlift-Direct-v0 \
      --trajectory_path=../planning_through_contact/IK_data/box_lift/IK_20260127_175758.npz \
      --headless \
      agent.policy.actor_hidden_dims=[$layer_str] \
      agent.policy.critic_hidden_dims=[$layer_str] \
      --run_name=$run_name
  done
done