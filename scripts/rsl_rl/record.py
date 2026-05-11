# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--trajectory_path", type=str, default=None, help="Override trajectory path (default: read from run's env.yaml)")
parser.add_argument("--keep_voc", action="store_true", default=False,
                    help="Record with the VOC gains saved at the end of training (loaded from <log_dir>/voc_state.npz). "
                         "If unset (default), VOC is forcibly disabled so the rollout matches deployment (kp=kv=0).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import re
import time
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
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import BoxLift.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.max_episode_steps = -1

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # Load saved training config so obs/action layout matches the checkpoint
    with open(os.path.join(log_dir, "params", "env.yaml"), "r") as f:
        saved_env_cfg = yaml.unsafe_load(f)

    if args_cli.trajectory_path is not None:
        env_cfg.trajectory_path = args_cli.trajectory_path
    else:
        env_cfg.trajectory_path = saved_env_cfg["trajectory_path"]
    print(f"[INFO] Using trajectory: {env_cfg.trajectory_path}")

    # Restore fields that affect obs/action layout from the trained run
    if "obs_history_steps" in saved_env_cfg:
        env_cfg.obs_history_steps = int(saved_env_cfg["obs_history_steps"])
    if "action_mode" in saved_env_cfg:
        env_cfg.action_mode = saved_env_cfg["action_mode"]
    if "observation_space" in saved_env_cfg:
        env_cfg.observation_space = int(saved_env_cfg["observation_space"])
    # Phase slowdown determines action_space (7 if enabled, 6 otherwise) — must be set
    # before env creation so __post_init__ allocates the correct action head shape, or
    # checkpoint loading will fail with a size mismatch on actor.6.weight/bias.
    if "enable_phase_slowdown" in saved_env_cfg:
        env_cfg.enable_phase_slowdown = bool(saved_env_cfg["enable_phase_slowdown"])
    # Phase-mapping variants (only meaningful if enable_phase_slowdown is True). These
    # control how action[6] is interpreted; mismatch between train and eval would make
    # the policy's commanded dphase wrong even if shapes match.
    for _phase_field in ("phase_mapping", "dphase_max", "dphase_min",
                         "task_scale_by_dphase", "track_scale_by_dphase"):
        if _phase_field in saved_env_cfg and hasattr(env_cfg, _phase_field):
            setattr(env_cfg, _phase_field, saved_env_cfg[_phase_field])
    print(f"[INFO] obs_history_steps={getattr(env_cfg, 'obs_history_steps', None)}, "
          f"action_mode={getattr(env_cfg, 'action_mode', None)}, "
          f"observation_space={env_cfg.observation_space}, "
          f"enable_phase_slowdown={getattr(env_cfg, 'enable_phase_slowdown', None)}")

    # Pin curriculum α to the value the policy was trained at (see play.py for rationale).
    m = re.search(r"model_(\d+)\.pt", os.path.basename(resume_path))
    ckpt_iter = int(m.group(1)) if m else 0
    num_steps_per_env = int(getattr(agent_cfg, "num_steps_per_env", 24))
    warmup = int(saved_env_cfg.get("alpha_warmup_steps", 0) or 0)
    if warmup > 0:
        env_cfg.force_alpha = min(1.0, ckpt_iter * num_steps_per_env / warmup)
    else:
        env_cfg.force_alpha = 1.0
    print(f"[INFO] Pinning curriculum α = {env_cfg.force_alpha:.3f} "
          f"(ckpt_iter={ckpt_iter}, num_steps_per_env={num_steps_per_env}, warmup={warmup})")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # === Virtual Object Controller override at record time ===
    # See play.py for the rationale — VOC's runtime decay state is saved separately
    # to <log_dir>/voc_state.npz and is NOT in the checkpoint or env.yaml. Default
    # behavior here is to disable VOC entirely so the recorded rollout reflects
    # deployment conditions; --keep_voc loads the training-end gains instead.
    _direct_env = env.unwrapped
    if hasattr(_direct_env, "voc_kp_pos"):
        if args_cli.keep_voc:
            _voc_state_path = os.path.join(log_dir, "voc_state.npz")
            if os.path.exists(_voc_state_path):
                _voc_state = np.load(_voc_state_path)
                _direct_env.voc_kp_pos = float(_voc_state["voc_kp_pos"])
                _direct_env.voc_kp_rot = float(_voc_state["voc_kp_rot"])
                _direct_env.voc_kv_pos = float(_voc_state["voc_kv_pos"])
                _direct_env.voc_kv_rot = float(_voc_state["voc_kv_rot"])
                print(f"[INFO] --keep_voc: loaded VOC state from {_voc_state_path} "
                      f"(kp_pos={_direct_env.voc_kp_pos:.3f}, kp_rot={_direct_env.voc_kp_rot:.3f})")
            else:
                print(f"[WARN] --keep_voc requested but {_voc_state_path} not found; "
                      f"VOC will use the cfg's initial values (kp_pos={_direct_env.voc_kp_pos}). "
                      f"This is likely wrong — re-train with VOC state saving, or omit --keep_voc.")
        else:
            _direct_env.voc_kp_pos = 0.0
            _direct_env.voc_kp_rot = 0.0
            _direct_env.voc_kv_pos = 0.0
            _direct_env.voc_kv_rot = 0.0
            print("[INFO] VOC disabled for record (kp=kv=0). Use --keep_voc to load training-end gains.")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")


    dt = env.unwrapped.step_dt

    direct_env = env.env.unwrapped
    direct_env._reset_idx(None, 0)

    # Detect single-arm (boxpush) vs dual-arm (boxlift) layout
    dual_arm = hasattr(direct_env, "ur5e_l") and hasattr(direct_env, "ur5e_r")

    def _flatten(x):
        """Concatenate tuples of per-arm tensors into a flat (12,) vector; pass through (6,) tensors."""
        if isinstance(x, (tuple, list)):
            return torch.cat([t[0] for t in x]).cpu().tolist()
        return x[0].cpu().tolist()

    def _joint_positions():
        if dual_arm:
            return torch.cat((direct_env.ur5e_l.data.joint_pos[0], direct_env.ur5e_r.data.joint_pos[0])).cpu().tolist()
        return direct_env.ur5e.data.joint_pos[0].cpu().tolist()

    def _applied_torques():
        if dual_arm:
            return torch.cat((direct_env.ur5e_l.data.applied_torque[0], direct_env.ur5e_r.data.applied_torque[0])).cpu().tolist()
        return direct_env.ur5e.data.applied_torque[0].cpu().tolist()

    def _expected_q():
        """Trajectory joints at the current phase. Boxpush uses float phase + _interp;
        boxlift uses integer episode_length_buf with separate joints_l/joints_r."""
        if dual_arm:
            idx = int(direct_env.episode_length_buf[0].item())
            idx = min(idx, direct_env.joints_l.shape[0] - 1)
            return torch.cat([direct_env.joints_l[idx], direct_env.joints_r[idx]]).cpu().tolist()
        return direct_env._interp(direct_env.joints)[0].cpu().tolist()

    def _phase_value():
        """Float phase index. Boxlift exposes only episode_length_buf (integer)."""
        if hasattr(direct_env, "phase"):
            return float(direct_env.phase[0].item())
        return float(direct_env.episode_length_buf[0].item())

    def _obj_pose():
        """Box pose in env-frame. Returns (pos_xyz, quat_wxyz) as flat lists."""
        pos = (direct_env.object.data.root_pos_w[0] - direct_env.scene.env_origins[0]).cpu().tolist()
        quat = direct_env.object.data.root_quat_w[0].cpu().tolist()
        return pos, quat

    # Per-step rollout buffers. Schema matches ur_rtde_real_time.py's tracking_log so the
    # same downstream tooling (visualize_traj.py, plot_rollout_rewards.py, analyze_*.ipynb,
    # ur_rtde_test.py) works on both real and sim rollouts.
    rollout = {
        "steps": [],
        "phase": [],
        "actual_q": [],          # post-step joint position (matches real-rollout convention)
        "expected_q": [],        # trajectory joints at current phase
        "target_q": [],          # joint targets sent to the actuator (was joint_targets)
        "actual_obj_pos": [],
        "actual_obj_quat": [],
        "joint_torques": [],     # not in real npz; kept for analyze_policy_residual.ipynb
        "rewards": [],
    }
    extras_log = []  # list of dicts of {metric_key: float}

    def _to_py(v):
        """Convert a torch tensor / numpy scalar / python scalar to a python float."""
        if hasattr(v, "detach"):
            v = v.detach()
        if hasattr(v, "cpu"):
            v = v.cpu()
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)

            target_q  = _flatten(direct_env.get_joint_targets())
            expected  = _expected_q()
            phase_val = _phase_value()

            obs, rewards, dones, extras = env.step(actions)

            obj_pos, obj_quat = _obj_pose()
            rollout["steps"].append(timestep)
            rollout["phase"].append(phase_val)
            rollout["actual_q"].append(_joint_positions())
            rollout["expected_q"].append(expected)
            rollout["target_q"].append(target_q)
            rollout["actual_obj_pos"].append(obj_pos)
            rollout["actual_obj_quat"].append(obj_quat)
            rollout["joint_torques"].append(_applied_torques())
            rollout["rewards"].append(float(rewards[0].item()))

            log_entry = {}
            raw_log = extras.get("log", {}) if isinstance(extras, dict) else {}
            for k, v in raw_log.items():
                log_entry[k] = _to_py(v)
            extras_log.append(log_entry)

            timestep += 1
            if torch.any(dones):
                break

        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    rollout_path = os.path.join(log_dir, "rollout", "output.npz")
    os.makedirs(os.path.dirname(rollout_path), exist_ok=True)

    # Per-step extras: collect every metric key seen, stack into (N,) arrays. NPZ keys can't
    # contain '/', so encode "Rewards_task/obj_pos" → "extras__Rewards_task__obj_pos". The
    # "extras__" prefix lets loaders enumerate them without false positives.
    all_extras_keys = sorted({k for e in extras_log for k in e.keys()})
    extras_arrays = {
        f"extras__{k.replace('/', '__')}": np.asarray(
            [e.get(k, float("nan")) for e in extras_log], dtype=np.float32
        )
        for k in all_extras_keys
    }

    arm_pose_kwargs = (
        {"arm_l_pose": direct_env.arm_l_pose.cpu().numpy(),
         "arm_r_pose": direct_env.arm_r_pose.cpu().numpy()}
        if dual_arm else
        {"arm_pose": direct_env.arm_pose.cpu().numpy()}
    )

    np.savez(
        rollout_path,
        steps=np.asarray(rollout["steps"], dtype=np.int64),
        phase=np.asarray(rollout["phase"], dtype=np.float32),
        actual_q=np.asarray(rollout["actual_q"], dtype=np.float32),
        expected_q=np.asarray(rollout["expected_q"], dtype=np.float32),
        target_q=np.asarray(rollout["target_q"], dtype=np.float32),
        actual_obj_pos=np.asarray(rollout["actual_obj_pos"], dtype=np.float32),
        actual_obj_quat=np.asarray(rollout["actual_obj_quat"], dtype=np.float32),
        joint_torques=np.asarray(rollout["joint_torques"], dtype=np.float32),
        rewards=np.asarray(rollout["rewards"], dtype=np.float32),
        # Sample period of the rollout (env step). Real-rollout npz uses 1/500;
        # consumers should read this rather than hardcode 500Hz.
        src_dt=np.float64(dt),
        # Real-rollout-only fields, NaN here so the schema matches.
        gain=np.float64("nan"),
        lookahead_time=np.float64("nan"),
        # Metadata.
        action_scale=np.asarray(getattr(env_cfg, "action_scale", 0.0)),
        action_mode=np.asarray(str(getattr(env_cfg, "action_mode", ""))),
        trajectory_path=np.asarray(str(env_cfg.trajectory_path)),
        obs_history_steps=np.int64(getattr(env_cfg, "obs_history_steps", 1)),
        dual_arm=np.bool_(dual_arm),
        **arm_pose_kwargs,
        **extras_arrays,
    )
    print(f"[INFO] Rollout saved to: {rollout_path}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
