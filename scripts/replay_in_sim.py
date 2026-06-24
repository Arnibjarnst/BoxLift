"""Replay a real-robot rollout NPZ in Isaac Sim.

Feeds the target_q values from the real-robot log directly into the sim's PD
controller (bypassing the policy), then logs joint positions and EE contact
forces.  Useful for comparing sim vs real contact dynamics on the same motion.

Usage:
    ${ISAACLAB_PATH}/isaaclab.sh -p scripts/replay_in_sim.py \\
        --rollout_path logs/rsl_rl/boxlift/<run>/ur_rtde_logs/<file>.npz \\
        --headless

The output NPZ is saved alongside the input as <stem>_sim_replay.npz.
Override with --output_path.
"""

import argparse
import sys
import traceback
from pathlib import Path

# Add source/BoxLift to sys.path so `import BoxLift.tasks` works whether or not
# the package was installed via `pip install -e source/BoxLift`.
_repo_root = Path(__file__).resolve().parent.parent
_boxlift_src = _repo_root / "source" / "BoxLift"
if str(_boxlift_src) not in sys.path:
    sys.path.insert(0, str(_boxlift_src))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay a real-robot rollout in simulation.")
parser.add_argument("--rollout_path", required=True,
                    help="Path to the real-robot .npz (from ur_rtde_real_time.py)")
parser.add_argument("--trajectory_path", default=None,
                    help="Override trajectory path. Auto-detected from params/env.yaml if omitted.")
parser.add_argument("--output_path", default=None,
                    help="Output .npz path (default: <rollout_stem>_sim_replay.npz next to input)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Strip custom args so Hydra / gymnasium don't choke on them
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Everything below runs inside the running sim ─────────────────────────────

import numpy as np
import torch
import gymnasium as gym

import BoxLift.tasks  # noqa: F401  registers "Template-Boxlift-Direct-v0"
from BoxLift.tasks.direct.boxlift.boxlift_env_cfg import BoxliftEnvCfg


def main():
    # ── 1. Load rollout ──────────────────────────────────────────────────────
    rollout_path = Path(args_cli.rollout_path)
    d = np.load(rollout_path)

    target_q_500hz = d["target_q"]   # (T_ctrl, 12) at 500 Hz
    src_dt     = float(d["src_dt"]) if "src_dt" in d.files else 1.0 / 500.0
    dual_arm   = bool(d["dual_arm"]) if "dual_arm" in d.files else True
    decimation = int(d["policy_decimation"]) if "policy_decimation" in d.files else 10

    # Down-sample to policy rate (50 Hz = one env step)
    target_q = target_q_500hz[::decimation].astype(np.float32)   # (T, 12)
    T = len(target_q)
    ctrl_hz = 1.0 / src_dt
    policy_hz = ctrl_hz / decimation
    print(f"[replay] {rollout_path.name}")
    print(f"         {len(target_q_500hz)} steps @ {ctrl_hz:.0f} Hz  →  {T} steps @ {policy_hz:.0f} Hz")

    # ── 2. Configure env ─────────────────────────────────────────────────────
    env_cfg = BoxliftEnvCfg()
    env_cfg.scene.num_envs = 1

    # Resolve trajectory path:
    #   1. --trajectory_path CLI arg
    #   2. trajectory_path field stored in the rollout NPZ
    #   3. params/env.yaml in the run directory (rollout is at <run>/ur_rtde_logs/<file>.npz)
    traj_path = args_cli.trajectory_path
    if not traj_path and "trajectory_path" in d.files:
        traj_path = str(d["trajectory_path"]).strip()
    if not traj_path:
        env_yaml = rollout_path.parent.parent / "params" / "env.yaml"
        if env_yaml.exists():
            import yaml
            with open(env_yaml) as f:
                saved_cfg = yaml.unsafe_load(f)
            traj_path = saved_cfg.get("trajectory_path", "")
    if not traj_path:
        raise ValueError(
            "Cannot determine trajectory path. Pass --trajectory_path or ensure "
            "params/env.yaml exists in the run directory."
        )
    # Resolve relative paths against the repo root
    traj_path = Path(traj_path)
    if not traj_path.is_absolute():
        traj_path = _repo_root / traj_path
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_path}")
    env_cfg.trajectory_path = str(traj_path)
    print(f"[replay] trajectory: {traj_path}")

    # Disable task-failure termination — let the replay run to completion
    env_cfg.max_obj_dist_from_traj = 1e6
    env_cfg.max_obj_angle_from_traj = 1e6

    env_cfg.force_alpha = 1.0   # no curriculum annealing

    # ── 3. Create env and reset to phase 0 ──────────────────────────────────
    env = gym.make("Template-Boxlift-Direct-v0", cfg=env_cfg)
    de = env.unwrapped
    device = de.device

    # Standard gymnasium reset to initialise internal buffers
    env.reset()
    # Override to start at phase 0 (trajectory start)
    de._reset_idx(None, fixed_value=0)

    # ── 4. Patch _pre_physics_step to use rollout targets ────────────────────
    # _apply_action reads de._cached_joint_target which is normally written by
    # _pre_physics_step.  We overwrite it here with the desired replay targets.
    target_q_t = torch.from_numpy(target_q).to(device)   # (T, 12)

    def _replay_pre_physics(actions: torch.Tensor) -> None:
        i = int(de.episode_length_buf[0])
        if i >= T:
            return
        tq = target_q_t[i].unsqueeze(0)          # (1, 12)
        q_l = tq[:, :6].clamp(de.ur5e_l.data.joint_pos_limits[..., 0],
                               de.ur5e_l.data.joint_pos_limits[..., 1])
        q_r = tq[:, 6:].clamp(de.ur5e_r.data.joint_pos_limits[..., 0],
                               de.ur5e_r.data.joint_pos_limits[..., 1])
        de._cached_joint_target = (q_l, q_r)

    de._pre_physics_step = _replay_pre_physics

    # ── 5. Allocate log arrays ───────────────────────────────────────────────
    n_joints = 12 if dual_arm else 6
    log_target_q   = np.empty((T, n_joints), dtype=np.float32)
    log_actual_q   = np.empty((T, n_joints), dtype=np.float32)
    log_ee_force_l = np.zeros((T, 3), dtype=np.float32)  # net contact-force vector (world frame)
    log_ee_force_r = np.zeros((T, 3), dtype=np.float32)
    log_ee_mag_l   = np.zeros(T, dtype=np.float32)        # scalar total contact magnitude
    log_ee_mag_r   = np.zeros(T, dtype=np.float32)
    log_obj_pos    = np.zeros((T, 3), dtype=np.float32)
    log_obj_quat   = np.zeros((T, 4), dtype=np.float32)

    dummy_action = torch.zeros(1, env.action_space.shape[0], device=device)

    # ── 6. Replay loop ───────────────────────────────────────────────────────
    step = 0
    while simulation_app.is_running() and step < T:
        with torch.inference_mode():
            de.step(dummy_action)   # advances physics + updates sensors

        # The target applied this step was target_q[step]
        log_target_q[step] = target_q[step]

        # Joint positions (after physics)
        q_l = de.ur5e_l.data.joint_pos[0].cpu().numpy()
        q_r = de.ur5e_r.data.joint_pos[0].cpu().numpy() if dual_arm else np.zeros(6, np.float32)
        log_actual_q[step] = np.concatenate([q_l, q_r]) if dual_arm else q_l

        # EE contact forces — force_matrix_w: (N, n_bodies, n_filter_prims, 3)
        f_l = de.ee_contact_sensors[0].data.force_matrix_w   # (1, 1, 2, 3) typically
        log_ee_force_l[step] = f_l[0].sum(dim=(0, 1)).cpu().numpy()           # net vec
        log_ee_mag_l[step]   = float(f_l.norm(dim=-1).sum(dim=(-1, -2))[0])   # scalar

        if dual_arm and len(de.ee_contact_sensors) > 1:
            f_r = de.ee_contact_sensors[1].data.force_matrix_w
            log_ee_force_r[step] = f_r[0].sum(dim=(0, 1)).cpu().numpy()
            log_ee_mag_r[step]   = float(f_r.norm(dim=-1).sum(dim=(-1, -2))[0])

        # Box pose in env-local frame
        log_obj_pos[step]  = (de.object.data.root_pos_w - de.scene.env_origins)[0].cpu().numpy()
        log_obj_quat[step] = de.object.data.root_quat_w[0].cpu().numpy()

        step += 1
        if step % 50 == 0 or step == T:
            print(f"  step {step:4d}/{T}  "
                  f"EE_mag L={log_ee_mag_l[step-1]:.2f} N  R={log_ee_mag_r[step-1]:.2f} N  "
                  f"obj_z={log_obj_pos[step-1, 2]:.3f} m")

    print(f"[replay] done — {step} steps")

    # ── 7. Save ──────────────────────────────────────────────────────────────
    out_path = args_cli.output_path or str(
        rollout_path.parent / (rollout_path.stem + "_sim_replay.npz")
    )
    np.savez(
        out_path,
        target_q      = log_target_q,      # (T, 12) what was commanded
        actual_q      = log_actual_q,      # (T, 12) what sim achieved
        ee_force_l    = log_ee_force_l,    # (T, 3)  net EE contact force, world frame
        ee_force_r    = log_ee_force_r,    # (T, 3)
        ee_force_l_mag= log_ee_mag_l,      # (T,)    scalar total contact magnitude
        ee_force_r_mag= log_ee_mag_r,      # (T,)
        actual_obj_pos = log_obj_pos,      # (T, 3)
        actual_obj_quat= log_obj_quat,     # (T, 4)
        src_dt        = np.float64(de.step_dt),   # 0.02 s at 50 Hz
        dual_arm      = np.bool_(dual_arm),
        rollout_path  = np.asarray(str(rollout_path)),
    )
    print(f"[replay] saved → {out_path}")

    env.close()


if __name__ == "__main__":
    _log_file = Path(args_cli.rollout_path).parent / "replay_in_sim.log"
    try:
        main()
    except Exception:
        msg = traceback.format_exc()
        print(msg)
        _log_file.write_text(msg)
        print(f"[replay] error written to {_log_file}")
    finally:
        simulation_app.close()
