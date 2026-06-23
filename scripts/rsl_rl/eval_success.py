# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-env eval: runs trained policy with max_episode_steps=-1 (full trajectory),
counts completions vs terminations across many auto-reset episodes, dumps a JSON
summary to the run dir. Used by the autonomous training loop."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Eval RL agent (multi-env success rate).")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--eval_steps", type=int, default=4000, help="Total policy steps to run.")
parser.add_argument("--warmup_steps", type=int, default=200, help="Discard first N steps from error stats.")
parser.add_argument("--trajectory_path", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import json
import os
import re
import torch
import yaml

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import BoxLift.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.max_episode_steps = -1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    with open(os.path.join(log_dir, "params", "env.yaml"), "r") as f:
        saved_env_cfg = yaml.unsafe_load(f)
    env_cfg.trajectory_path = args_cli.trajectory_path or saved_env_cfg["trajectory_path"]
    if "obs_history_steps" in saved_env_cfg:
        env_cfg.obs_history_steps = int(saved_env_cfg["obs_history_steps"])
    if "action_mode" in saved_env_cfg:
        env_cfg.action_mode = saved_env_cfg["action_mode"]
    if "observation_space" in saved_env_cfg:
        env_cfg.observation_space = int(saved_env_cfg["observation_space"])

    m = re.search(r"model_(\d+)\.pt", os.path.basename(resume_path))
    ckpt_iter = int(m.group(1)) if m else 0
    num_steps_per_env = int(getattr(agent_cfg, "num_steps_per_env", 24))
    warmup = int(saved_env_cfg.get("alpha_warmup_steps", 0) or 0)
    if warmup > 0:
        env_cfg.force_alpha = min(1.0, ckpt_iter * num_steps_per_env / warmup)
    else:
        env_cfg.force_alpha = 1.0

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    direct_env = env.env.unwrapped

    completed = 0
    terminated = 0
    obj_pos_err_sum = 0.0
    obj_quat_err_sum = 0.0
    n_err_samples = 0
    max_obj_pos_err = 0.0
    max_obj_quat_err = 0.0

    obs = env.get_observations()
    for step in range(args_cli.eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)

            rt = direct_env.reset_terminated.detach().cpu()
            to = direct_env.reset_time_outs.detach().cpu()
            completed  += int(to.sum().item())
            terminated += int(rt.sum().item())

            if step >= args_cli.warmup_steps:
                pos_err = direct_env._get_obj_pos_error()
                quat_err = direct_env._get_obj_quat_error()
                obj_pos_err_sum += float(pos_err.mean().item())
                obj_quat_err_sum += float(quat_err.mean().item())
                n_err_samples += 1
                max_obj_pos_err = max(max_obj_pos_err, float(pos_err.max().item()))
                max_obj_quat_err = max(max_obj_quat_err, float(quat_err.max().item()))

    total_eps = completed + terminated
    success_rate = completed / total_eps if total_eps > 0 else 0.0
    mean_obj_pos_err = obj_pos_err_sum / max(n_err_samples, 1)
    mean_obj_quat_err = obj_quat_err_sum / max(n_err_samples, 1)

    summary = {
        "checkpoint": os.path.basename(resume_path),
        "checkpoint_iter": ckpt_iter,
        "num_envs": args_cli.num_envs,
        "eval_steps": args_cli.eval_steps,
        "trajectory_path": env_cfg.trajectory_path,
        "trajectory_length": int(direct_env.obj_poses.shape[0]),
        "force_alpha": float(env_cfg.force_alpha),
        "completed_episodes": completed,
        "terminated_episodes": terminated,
        "total_episodes": total_eps,
        "success_rate": success_rate,
        "mean_obj_pos_error": mean_obj_pos_err,
        "mean_obj_quat_error": mean_obj_quat_err,
        "max_obj_pos_error": max_obj_pos_err,
        "max_obj_quat_error": max_obj_quat_err,
    }
    out_path = os.path.join(log_dir, "eval_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[EVAL] success_rate={success_rate:.3f}  ({completed}/{total_eps})  "
          f"obj_pos_err={mean_obj_pos_err:.4f}  obj_quat_err={mean_obj_quat_err:.4f}")
    print(f"[EVAL] summary -> {out_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
