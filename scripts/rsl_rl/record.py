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
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel rollouts to record. All start at phase=0. The loop "
                         "continues until every env has hit `done`; finished envs are masked out "
                         "of the running mean and their per-env slots are flagged in `alive_mask` "
                         "so downstream code can ignore the post-reset garbage. Per-env arrays in "
                         "the npz are (T, N, ...), even for N=1, so the schema is uniform.")
parser.add_argument("--trajectory_path", type=str, default=None, help="Override trajectory path (default: read from run's env.yaml)")
parser.add_argument("--rollout_path", type=str, default=None,
                    help="Full output path for the rollout npz. Default: <ckpt_log_dir>/rollout/output.npz. "
                         "Used by scripts/record_dataset.py to direct per-trajectory outputs into "
                         "separate files within one shared folder.")
parser.add_argument("--voc_kp_pos", type=float, default=None,
                    help="Record-time override: set a single VOC kp_pos and derive kp_rot, kv_pos, kv_rot "
                         "from the env's init formulas (proportional + critical damping). Lookup the value "
                         "at the iter of interest from W&B (VOC/kp_pos_mean) and pass it here to recreate "
                         "that training-time VOC strength. Ignored if --keep_voc is set. Pass 0 to disable.")
parser.add_argument("--keep_voc", action="store_true", default=False,
                    help="Record with the VOC gains saved at the end of training (loaded from <log_dir>/voc_state.npz). "
                         "If unset (default), VOC is forcibly disabled so the rollout matches deployment (kp=kv=0).")
parser.add_argument("--zero_policy", action="store_true", default=False,
                    help="Skip checkpoint loading and use zero actions (pure nominal controller baseline). "
                         "Requires trajectory to be set via --trajectory_path or env.trajectory_path=. "
                         "No log_dir or saved env.yaml is needed.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


def _hydra_overridden_env_fields(hydra_args: list[str]) -> set[str]:
    """Extract the set of top-level `env.<X>` keys the user set on the command line.
    Used to gate the saved-cfg restoration block in main() — fields the user explicitly
    overrode via Hydra must NOT be clobbered back to the trained-run's saved value.
    Handles +env.X=... (add) and ~env.X (delete) prefixes. Nested keys like
    `env.scene.num_envs` collapse to the top-level (`scene`) — fine for our use since
    the restoration block only touches top-level env fields."""
    out = set()
    for arg in hydra_args:
        a = arg.lstrip("+~")
        if not a.startswith("env."):
            continue
        rest = a[len("env."):]
        key = rest.split("=", 1)[0]
        out.add(key.split(".")[0])
    return out


_env_overrides = _hydra_overridden_env_fields(hydra_args)
if _env_overrides:
    print(f"[INFO] Hydra env overrides detected, will NOT restore from saved env.yaml: {sorted(_env_overrides)}")

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
import sys
import time
import torch
import yaml
from pathlib import Path

# isaaclab_rl, isaaclab_tasks and rsl_rl live alongside isaaclab in the source tree.
# When running with plain `python` (not isaaclab.sh) they may not be on sys.path yet.
# Derive their location from the isaaclab package that AppLauncher already loaded.
import isaaclab as _isaaclab_pkg
_isaaclab_src_root = Path(_isaaclab_pkg.__file__).resolve().parent.parent.parent
for _pkg_name in ("isaaclab_rl", "isaaclab_tasks", "rsl_rl"):
    _pkg_path = str(_isaaclab_src_root / _pkg_name)
    if Path(_pkg_path).exists() and _pkg_path not in sys.path:
        sys.path.insert(0, _pkg_path)

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
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.max_episode_steps = -1

    # Ask the env to also stash per-env reward components in extras["log_per_env"] so
    # we can save per-env arrays alongside the env-mean ones. Off during training to
    # avoid runner overhead; harmless on envs that don't recognize the flag.
    if hasattr(env_cfg, "emit_per_env_extras"):
        env_cfg.emit_per_env_extras = True

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.zero_policy:
        resume_path = None
        log_dir = None
        if args_cli.trajectory_path is not None:
            env_cfg.trajectory_path = args_cli.trajectory_path
        print(f"[INFO] --zero_policy: skipping checkpoint. Trajectory: {getattr(env_cfg, 'trajectory_path', '(from Hydra)')}")
        env_cfg.force_alpha = 1.0
    else:
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

        # User-explicit CLI flag for trajectory_path wins over saved value (legacy non-Hydra arg).
        if args_cli.trajectory_path is not None:
            env_cfg.trajectory_path = args_cli.trajectory_path
        elif "trajectory_path" not in _env_overrides:
            env_cfg.trajectory_path = saved_env_cfg["trajectory_path"]
        print(f"[INFO] Using trajectory: {env_cfg.trajectory_path}")

        # All restorations below: only apply if the user didn't already set the field via a
        # Hydra `env.<field>=...` override. Otherwise we'd silently clobber the override.
        if "dataset_path" in saved_env_cfg and "dataset_path" not in _env_overrides:
            env_cfg.dataset_path = saved_env_cfg["dataset_path"]
        if getattr(env_cfg, "dataset_path", ""):
            print(f"[INFO] Using dataset: {env_cfg.dataset_path}")

        # Restore fields that affect obs/action layout from the trained run
        if "obs_history_steps" in saved_env_cfg and "obs_history_steps" not in _env_overrides:
            env_cfg.obs_history_steps = int(saved_env_cfg["obs_history_steps"])
        if "action_mode" in saved_env_cfg and "action_mode" not in _env_overrides:
            env_cfg.action_mode = saved_env_cfg["action_mode"]
        if "observation_space" in saved_env_cfg and "observation_space" not in _env_overrides:
            obs_space = saved_env_cfg["observation_space"]
            env_cfg.observation_space = obs_space if isinstance(obs_space, dict) else int(obs_space)
        # Phase slowdown determines action_space (7 if enabled, 6 otherwise) — must be set
        # before env creation so __post_init__ allocates the correct action head shape, or
        # checkpoint loading will fail with a size mismatch on actor.6.weight/bias.
        if "enable_phase_slowdown" in saved_env_cfg and "enable_phase_slowdown" not in _env_overrides:
            env_cfg.enable_phase_slowdown = bool(saved_env_cfg["enable_phase_slowdown"])
        # Phase-mapping variants (only meaningful if enable_phase_slowdown is True). These
        # control how action[6] is interpreted; mismatch between train and eval would make
        # the policy's commanded dphase wrong even if shapes match.
        for _phase_field in ("phase_mapping", "dphase_max", "dphase_min",
                             "task_scale_by_dphase", "track_scale_by_dphase"):
            if (_phase_field in saved_env_cfg and hasattr(env_cfg, _phase_field)
                    and _phase_field not in _env_overrides):
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
                # Per-segment training saves (S,) arrays; single-VOC saves scalars.
                # apply_voc_state handles both transparently.
                if hasattr(_direct_env, "apply_voc_state"):
                    _direct_env.apply_voc_state(_voc_state)
                else:
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
        elif args_cli.voc_kp_pos is not None:
            # User-specified kp_pos. Env derives kp_rot / kv_pos / kv_rot via the
            # init-time formulas (proportional + critical damping). Single-scalar
            # approximation — per-segment differences during training are flattened.
            if hasattr(_direct_env, "set_voc_strength"):
                _direct_env.set_voc_strength(float(args_cli.voc_kp_pos))
            else:
                _direct_env.voc_kp_pos = float(args_cli.voc_kp_pos)
                print(f"[WARN] env has no set_voc_strength method; only kp_pos overridden "
                      f"(kp_rot / kv_* unchanged). Update the env to derive them properly.")
        else:
            _direct_env.voc_kp_pos = 0.0
            _direct_env.voc_kp_rot = 0.0
            _direct_env.voc_kv_pos = 0.0
            _direct_env.voc_kv_rot = 0.0
            print("[INFO] VOC disabled for record (kp=kv=0). Use --keep_voc to load training-end "
                  "gains, or --voc_kp_pos <value> to recreate a specific iter's strength.")

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

    if args_cli.zero_policy:
        print("[INFO] --zero_policy: using zero-action baseline (no checkpoint).")
        _action_shape = env.action_space.shape
        _device = env.unwrapped.device
        policy = lambda obs: torch.zeros(_action_shape, device=_device)  # noqa: E731
    else:
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
    num_envs = int(direct_env.num_envs)
    device = direct_env.device

    # Detect single-arm (boxpush) vs dual-arm (boxlift) layout
    dual_arm = hasattr(direct_env, "ur5e_l") and hasattr(direct_env, "ur5e_r")

    # All helpers below return per-env arrays of shape (N, ...). For dual-arm,
    # joint-related quantities are concatenated as [left (6), right (6)] = (N, 12).
    def _to_np_per_env(t):
        """(N, D) torch tensor → (N, D) numpy array."""
        return t.detach().cpu().numpy()

    def _flatten_targets(x):
        """get_joint_targets() returns either a single (N, 6) tensor (boxpush) or a
        (q_l, q_r) tuple of (N, 6) tensors (boxlift). Concatenate to (N, 12 or 6)."""
        if isinstance(x, (tuple, list)):
            return _to_np_per_env(torch.cat(x, dim=-1))
        return _to_np_per_env(x)

    def _joint_positions_all():
        if dual_arm:
            return _to_np_per_env(torch.cat((direct_env.ur5e_l.data.joint_pos,
                                             direct_env.ur5e_r.data.joint_pos), dim=-1))
        return _to_np_per_env(direct_env.ur5e.data.joint_pos)

    def _applied_torques_all():
        if dual_arm:
            return _to_np_per_env(torch.cat((direct_env.ur5e_l.data.applied_torque,
                                             direct_env.ur5e_r.data.applied_torque), dim=-1))
        return _to_np_per_env(direct_env.ur5e.data.applied_torque)

    def _expected_q_all():
        """Trajectory joints at each env's current phase. Boxlift indexes per-env via
        integer episode_length_buf with separate joints_l/joints_r; boxpush uses
        float `phase` + `_interp`."""
        if dual_arm:
            T_traj = direct_env.joints_l.shape[0]
            idx = direct_env.episode_length_buf.clamp(max=T_traj - 1).long()
            q_l = direct_env.joints_l[idx]  # (N, 6)
            q_r = direct_env.joints_r[idx]
            return _to_np_per_env(torch.cat([q_l, q_r], dim=-1))
        return _to_np_per_env(direct_env._interp(direct_env.joints))

    def _phase_all():
        """Per-env float phase. Boxlift only exposes integer episode_length_buf."""
        if hasattr(direct_env, "phase"):
            return direct_env.phase.detach().cpu().numpy().astype(np.float32)
        return direct_env.episode_length_buf.detach().cpu().numpy().astype(np.float32)

    def _obj_pose_all():
        """Per-env box pose in env-frame. Returns (pos_xyz (N, 3), quat_wxyz (N, 4))."""
        pos = (direct_env.object.data.root_pos_w - direct_env.scene.env_origins)
        quat = direct_env.object.data.root_quat_w
        return _to_np_per_env(pos), _to_np_per_env(quat)

    # Per-step rollout buffers. All per-env fields are (N, ...) at append time; we
    # stack to (T, N, ...) at save time. Schema matches ur_rtde_real_time.py modulo
    # the new N dim (downstream notebooks can .squeeze(1) for N=1 if needed).
    rollout = {
        "steps": [],
        "phase": [],             # (T, N) float
        "actual_q": [],          # (T, N, J) post-step joint position
        "expected_q": [],        # (T, N, J) trajectory joints at current phase
        "target_q": [],          # (T, N, J) joint targets sent to the actuator
        "actual_obj_pos": [],    # (T, N, 3)
        "actual_obj_quat": [],   # (T, N, 4)
        "joint_torques": [],     # (T, N, J)
        "ee_force": [],          # (T, N, S, 3) net contact-force vector per EE sensor (world frame)
        "ee_force_mag": [],      # (T, N, S)   scalar total contact magnitude per EE sensor
        "rewards": [],           # (T, N) per-env reward; mean over alive envs derived at save
        "alive_mask": [],        # (T, N) bool: True if env was alive going INTO this step
        "done_this_step": [],    # (T, N) bool: True iff env's `done` fired at this step
    }

    # EE contact sensors (boxlift/boxhinge/boxtracker have them; absent on simpler envs).
    _ee_sensors = getattr(direct_env, "ee_contact_sensors", [])

    def _ee_forces_all():
        """Returns (N, S, 3) net force and (N, S) magnitude; zeros if no sensors."""
        if not _ee_sensors:
            return (np.zeros((num_envs, 0, 3), dtype=np.float32),
                    np.zeros((num_envs, 0),    dtype=np.float32))
        vecs = []
        mags = []
        for sensor in _ee_sensors:
            f = sensor.data.force_matrix_w          # (N, n_bodies, n_filter, 3)
            vecs.append(f.sum(dim=(1, 2)).detach().cpu().numpy())           # (N, 3)
            mags.append(f.norm(dim=-1).sum(dim=(-1, -2)).detach().cpu().numpy())  # (N,)
        return (np.stack(vecs, axis=1).astype(np.float32),   # (N, S, 3)
                np.stack(mags, axis=1).astype(np.float32))   # (N, S)
    extras_log = []          # per-step dicts of {metric_key: float} — already env-mean
    extras_per_env_log = []  # per-step dicts of {metric_key: (N,) numpy array} — when the env
                             # exposes `extras["log_per_env"]` (boxlift does). Lets downstream
                             # tools split reward components per-env (e.g. why env 12 spiked).

    def _to_py(v):
        """Convert torch / numpy / python scalar to a python float (or list)."""
        if hasattr(v, "detach"):
            v = v.detach()
        if hasattr(v, "cpu"):
            v = v.cpu()
        if hasattr(v, "tolist"):
            return v.tolist()
        return v

    def _to_np_1d(v):
        """Convert a torch (N,) / numpy (N,) tensor to a 1-D numpy float32 array. Scalars
        are broadcast to (num_envs,) NaN so the resulting stack stays shape-uniform."""
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v)
        if v.ndim == 0:
            return np.full(num_envs, np.nan, dtype=np.float32)
        return v.astype(np.float32, copy=False)

    # Track per-env aliveness and the step at which each env terminated. Once an env
    # hits done, IsaacLab auto-resets it and the data flowing through env.step is from
    # the next episode — alive_mask is what downstream code uses to ignore those slots.
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    done_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    # reset environment
    obs = env.get_observations()

    # === Dataset-mode capture (boxtracker / any env with a per-env segment pool) ======
    # Single-trajectory envs (boxhinge / boxlift) share one trajectory across all envs
    # and the per-env reference is implicit. Dataset-mode envs assign a (potentially
    # different) trajectory to each env on reset; the active per-env reference lives at
    # `env.unwrapped.obj_poses` with shape (N, T_max, 7). Capture it RIGHT NOW (after
    # the initial reset, before any step has fired) so rollout_summary can do per-env
    # reference lookups without needing to re-load the dataset.
    _u = env.unwrapped
    _dataset_kwargs = {}
    if hasattr(_u, "obj_poses") and _u.obj_poses.dim() == 3:
        # (N, T_max, 7) — store as float32 to keep the npz small.
        _dataset_kwargs["ref_obj_poses"] = _u.obj_poses.detach().cpu().numpy().astype(np.float32)
        if hasattr(_u, "cur_seg_idx"):
            _dataset_kwargs["env_traj_idx"] = _u.cur_seg_idx.detach().cpu().numpy().astype(np.int64)
        if hasattr(_u, "segment_length"):
            # Per-env length of the active (variable-length) segment. Lets rollout_summary
            # pick the right "goal" frame (= last valid frame) instead of relying on the
            # right-padded sentinel — padding usually replicates the last valid frame so
            # `ref_obj_poses[env, -1]` would also work, but explicit lengths are safer.
            _dataset_kwargs["env_traj_length"] = _u.segment_length.detach().cpu().numpy().astype(np.int64)
        print(f"[INFO] dataset-mode capture: ref_obj_poses shape {_dataset_kwargs['ref_obj_poses'].shape}, "
              f"unique env_traj_idx={len(np.unique(_dataset_kwargs.get('env_traj_idx', np.zeros(1))))}")

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)

            target_q  = _flatten_targets(direct_env.get_joint_targets())
            expected  = _expected_q_all()
            phase_val = _phase_all()
            alive_now = alive.detach().cpu().numpy().copy()

            obs, rewards, dones, extras = env.step(actions)
            # rsl_rl's VecEnvWrapper returns `dones` as torch.long (it's used as an
            # index into the rollout-storage tensors inside the runner). For our
            # bitwise-mask logic below we need it as bool — otherwise `alive & ~dones`
            # silently coerces `alive` to int, `~alive` becomes bitwise-NOT (= -2 for
            # value 1, -1 for value 0), and NumPy interprets the resulting int array
            # as FANCY INDICES rather than a bool mask in `arr[dead_after] = nan`.
            # The visible symptom is only envs N-2 (= ~int(1)) and N-1 (= ~int(0))
            # ever getting NaN'd, regardless of which env actually terminated.
            dones = dones.bool()

            # Per-env data captured AFTER the step. For envs that just terminated,
            # IsaacLab has already auto-reset them, so their slots reflect the new
            # episode's reset state — alive_mask flags this so consumers ignore them.
            obj_pos_all, obj_quat_all = _obj_pose_all()
            rollout["steps"].append(timestep)
            rollout["phase"].append(phase_val)
            rollout["actual_q"].append(_joint_positions_all())
            rollout["expected_q"].append(expected)
            rollout["target_q"].append(target_q)
            rollout["actual_obj_pos"].append(obj_pos_all)
            rollout["actual_obj_quat"].append(obj_quat_all)
            rollout["joint_torques"].append(_applied_torques_all())
            _ef_vec, _ef_mag = _ee_forces_all()
            rollout["ee_force"].append(_ef_vec)
            rollout["ee_force_mag"].append(_ef_mag)
            rollout["rewards"].append(rewards.detach().cpu().numpy().astype(np.float32))
            rollout["alive_mask"].append(alive_now)
            newly_done = (dones & alive)
            rollout["done_this_step"].append(newly_done.detach().cpu().numpy())

            log_entry = {}
            raw_log = extras.get("log", {}) if isinstance(extras, dict) else {}
            for k, v in raw_log.items():
                log_entry[k] = _to_py(v)
            extras_log.append(log_entry)

            # Per-env metrics — only populated when the env exposes log_per_env.
            log_pe_entry = {}
            raw_log_pe = extras.get("log_per_env", {}) if isinstance(extras, dict) else {}
            for k, v in raw_log_pe.items():
                log_pe_entry[k] = _to_np_1d(v)
            extras_per_env_log.append(log_pe_entry)

            # Mark envs that just finished; stop when every env has terminated.
            done_step[newly_done] = timestep
            alive = alive & ~dones

            # Once an env hits done, IsaacLab has already auto-reset it inside env.step
            # — every per-env state quantity we just captured for that env is therefore
            # the FIRST FRAME of a new episode, not the terminal state of the old one.
            # NaN those slots so downstream tools (plot_rollout_rewards.py,
            # visualize_traj.py, success-metric notebooks) naturally skip them instead
            # of plotting/rendering the auto-reset garbage as if it were real behavior.
            #
            # Reward is the exception: IsaacLab computes rewards BEFORE the auto-reset,
            # so the death-step reward IS the valid terminal reward — keep it. NaN
            # rewards only for envs that were ALREADY dead going into this step (their
            # reward at this step is the next-episode reward we want to ignore).
            dead_after = (~alive).detach().cpu().numpy()    # died this step OR earlier
            already_dead = ~alive_now                        # were already dead at start
            if dead_after.any():
                for k in ("phase", "actual_q", "expected_q", "target_q",
                          "actual_obj_pos", "actual_obj_quat", "joint_torques",
                          "ee_force", "ee_force_mag"):
                    arr = rollout[k][-1]
                    arr[dead_after, ...] = np.nan
                if log_pe_entry:
                    for v in log_pe_entry.values():
                        v[dead_after] = np.nan
            if already_dead.any():
                rollout["rewards"][-1][already_dead] = np.nan

            timestep += 1

            if not bool(alive.any().item()):
                print(f"[INFO] All {num_envs} env(s) terminated by step {timestep}.")
                break

        if args_cli.video and timestep >= args_cli.video_length:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    rollout_path = args_cli.rollout_path or os.path.join(log_dir, "rollout", "output.npz")
    os.makedirs(os.path.dirname(rollout_path), exist_ok=True)

    # Per-step extras (env-mean): collect every metric key seen, stack into (T,) arrays.
    # NPZ keys can't contain '/', so encode "Rewards_task/obj_pos" →
    # "extras__Rewards_task__obj_pos". The "extras__" prefix lets loaders enumerate them
    # without false positives.
    all_extras_keys = sorted({k for e in extras_log for k in e.keys()})
    extras_arrays = {
        f"extras__{k.replace('/', '__')}": np.asarray(
            [e.get(k, float("nan")) for e in extras_log], dtype=np.float32
        )
        for k in all_extras_keys
    }

    # Per-step per-env extras: parallel to the above but with an N dim. Missing keys at any
    # step (env-bound emission switched off mid-run) fill with NaN-(N,) so the (T, N) stack
    # stays shape-uniform. Encoded with "extras_per_env__" prefix; downstream tools should
    # prefer these over the env-mean `extras__*` when slicing to a specific env.
    all_per_env_keys = sorted({k for e in extras_per_env_log for k in e.keys()})
    nan_n = np.full(num_envs, np.nan, dtype=np.float32)
    extras_per_env_arrays = {
        f"extras_per_env__{k.replace('/', '__')}": np.stack(
            [e.get(k, nan_n) for e in extras_per_env_log], axis=0
        )
        for k in all_per_env_keys
    }

    arm_pose_kwargs = (
        {"arm_l_pose": direct_env.arm_l_pose.cpu().numpy(),
         "arm_r_pose": direct_env.arm_r_pose.cpu().numpy()}
        if dual_arm else
        {"arm_pose": direct_env.arm_pose.cpu().numpy()}
    )

    # Stack per-step lists into (T, N, ...) arrays.
    phase_arr           = np.stack(rollout["phase"],          axis=0)  # (T, N)
    actual_q_arr        = np.stack(rollout["actual_q"],       axis=0)  # (T, N, J)
    expected_q_arr      = np.stack(rollout["expected_q"],     axis=0)
    target_q_arr        = np.stack(rollout["target_q"],       axis=0)
    actual_obj_pos_arr  = np.stack(rollout["actual_obj_pos"], axis=0)  # (T, N, 3)
    actual_obj_quat_arr = np.stack(rollout["actual_obj_quat"],axis=0)  # (T, N, 4)
    joint_torques_arr   = np.stack(rollout["joint_torques"],  axis=0)
    ee_force_arr        = np.stack(rollout["ee_force"],       axis=0)   # (T, N, S, 3)
    ee_force_mag_arr    = np.stack(rollout["ee_force_mag"],   axis=0)   # (T, N, S)
    rewards_arr         = np.stack(rollout["rewards"],        axis=0).astype(np.float32)  # (T, N)
    alive_mask_arr      = np.stack(rollout["alive_mask"],     axis=0).astype(np.bool_)    # (T, N)
    done_this_step_arr  = np.stack(rollout["done_this_step"], axis=0).astype(np.bool_)

    # Convenience: alive-masked per-step mean reward. Computed once at save time so
    # downstream notebooks don't have to redo the mask math. NaN-safe — already-dead
    # envs have NaN rewards, which would otherwise propagate through the multiply
    # (0 * NaN == NaN). Use `valid_mask = alive_mask & finite(rewards)` so dead envs
    # are excluded from both the numerator and the denominator. Steps where no env
    # has a valid reward (shouldn't happen post-loop) get NaN.
    finite_mask = np.isfinite(rewards_arr)
    valid_mask = alive_mask_arr & finite_mask
    valid_f = valid_mask.astype(np.float32)
    n_valid_per_step = valid_f.sum(axis=1)
    r_safe = np.where(finite_mask, rewards_arr, 0.0)
    rewards_mean_alive = np.where(
        n_valid_per_step > 0,
        (r_safe * valid_f).sum(axis=1) / np.maximum(n_valid_per_step, 1),
        np.float32("nan"),
    ).astype(np.float32)

    done_step_arr = done_step.detach().cpu().numpy().astype(np.int64)  # (N,) -1 if never done

    np.savez(
        rollout_path,
        steps=np.asarray(rollout["steps"], dtype=np.int64),
        # Per-env, per-step. Shapes (T, N, ...); for N=1 do .squeeze(1) downstream.
        phase=phase_arr,
        actual_q=actual_q_arr,
        expected_q=expected_q_arr,
        target_q=target_q_arr,
        actual_obj_pos=actual_obj_pos_arr,
        actual_obj_quat=actual_obj_quat_arr,
        joint_torques=joint_torques_arr,
        ee_force=ee_force_arr,         # (T, N, S, 3) net contact-force vector per EE, world frame
        ee_force_mag=ee_force_mag_arr, # (T, N, S)   scalar total contact magnitude per EE
        rewards=rewards_arr,                  # per-env raw rewards
        rewards_mean_alive=rewards_mean_alive, # convenience: alive-masked mean per step
        alive_mask=alive_mask_arr,             # True iff env was alive going INTO that step
        done_this_step=done_this_step_arr,     # True iff env's `done` fired at that step
        done_step=done_step_arr,               # (N,) step idx where each env terminated; -1 if never
        num_envs=np.int64(num_envs),
        # Sample period of the rollout (env step). Real-rollout npz uses 1/500;
        # consumers should read this rather than hardcode 500Hz.
        src_dt=np.float64(dt),
        # Real-rollout-only fields, NaN here so the schema matches.
        gain=np.float64("nan"),
        lookahead_time=np.float64("nan"),
        # Metadata.
        action_scale=np.asarray(getattr(env_cfg, "action_scale", 0.0)),
        action_mode=np.asarray(str(getattr(env_cfg, "action_mode", ""))),
        trajectory_path=np.asarray(str(getattr(env_cfg, "trajectory_path", ""))),
        # Dataset-mode metadata (empty string when env runs a single trajectory).
        dataset_path=np.asarray(str(getattr(env_cfg, "dataset_path", ""))),
        obs_history_steps=np.int64(getattr(env_cfg, "obs_history_steps", 1)),
        dual_arm=np.bool_(dual_arm),
        **arm_pose_kwargs,
        **extras_arrays,
        **extras_per_env_arrays,
        **_dataset_kwargs,
    )
    print(f"[INFO] Rollout saved to: {rollout_path}  "
          f"(T={len(rollout['steps'])}, N={num_envs}, "
          f"done_step range=[{int(done_step_arr.min())}, {int(done_step_arr.max())}])")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
