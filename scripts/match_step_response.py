"""Free-running step-response match: does IsaacSim's PD reproduce URSim's servoJ
transient for a single small step from rest?

Companion to scripts/ursim_step_response.py. That script captures, on URSim/real, the
first ~20ms of joint motion after a fixed target step from a settled pose. This script:

  1. Writes IsaacSim to the captured initial pose with ZERO velocity (at rest).
  2. Sets the captured (fixed) joint target.
  3. Free-runs physics for `steps` ticks at the captured dt — NO state re-sync, so the
     PD dynamics evolve on their own (unlike match_ursim_dynamics.py's teacher-forced
     one-step prediction). The accumulated transient is what reveals kp early and kd
     over the rest of the window.
  4. Compares per-joint to the URSim transient, prints RMS, optionally plots.

Tune kp/kd (per-joint) until the IsaacSim curve overlays the URSim curve:
    python scripts/match_step_response.py <step_npz> --kp 300 --kd 45 --plot m.png
    python scripts/match_step_response.py <step_npz> --kp 300,300,300,28,28,28 \
        --kd 45,45,45,4.2,4.2,4.2 --plot m.png
"""

import argparse
import glob
import os

import numpy as np

from isaaclab.app import AppLauncher

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
JOINT_LABELS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("step_npz", type=str, help="Capture from ursim_step_response.py")
parser.add_argument("--task", type=str, default="Template-Boxhinge-Direct-v0")
parser.add_argument("--kp", type=str, default=None,
                    help="kp scalar or 6 comma-separated per-joint values (overrides cfg)")
parser.add_argument("--kd", type=str, default=None,
                    help="kd scalar or 6 comma-separated per-joint values (overrides cfg)")
parser.add_argument("--effort_limit", type=str, default=None,
                    help="effort_limit scalar or 6 comma-separated values (overrides cfg)")
parser.add_argument("--velocity_limit", type=float, default=None)
parser.add_argument("--actuator_type", type=str, default=None, help='"Implicit" or "IdealPD"')
parser.add_argument("--plot", type=str, default=None, help="Output PNG for the per-joint overlay.")
parser.add_argument("--out", type=str, default=None, help="Output npz with the comparison arrays.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import BoxLift.tasks  # noqa: F401, registers env


def _parse_gain(s, name):
    if s is None:
        return None
    parts = s.split(",")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != 6:
        raise ValueError(f"{name} list must have 1 or 6 entries, got {len(parts)}")
    return {jn: float(v) for jn, v in zip(UR5E_JOINT_NAMES, parts)}


# ---------- Load capture ----------

d = np.load(args.step_npz, allow_pickle=True)
init_q = np.asarray(d["init_q"], dtype=np.float64)
target_q = np.asarray(d["target_q"], dtype=np.float64)
ursim_q = np.asarray(d["actual_q"], dtype=np.float64)        # (steps, 6)
ursim_qd = np.asarray(d["actual_qd"], dtype=np.float64)       # (steps, 6)
src_dt = float(d["dt"]) if "dt" in d.files else 1.0 / 500.0
steps = int(ursim_q.shape[0])
print(f"[INFO] Capture: {steps} ticks @ {1/src_dt:.0f}Hz, "
      f"gain={float(d['gain'])}, lookahead={float(d['lookahead'])}")

kp_override = _parse_gain(args.kp, "kp")
kd_override = _parse_gain(args.kd, "kd")
effort_override = _parse_gain(args.effort_limit, "effort_limit")

# ---------- Configure env (mirror match_ursim_dynamics.py) ----------

from BoxLift.tasks.direct.boxhinge.boxhinge_env_cfg import BoxhingeEnvCfg

env_cfg: BoxhingeEnvCfg = BoxhingeEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.physics_dt = src_dt
env_cfg.decimation = 1
env_cfg.episode_length_s = steps * src_dt * 4

env_cfg.obs_obj_pos_noise = 0.0
env_cfg.obs_obj_ori_noise = 0.0
env_cfg.obs_obj_pos_bias_std = 0.0
env_cfg.obs_obj_ori_bias_std = 0.0
env_cfg.obs_obj_delay_steps = 0
env_cfg.obs_obj_update_period = 1
env_cfg.voc_enabled = False
env_cfg.enable_phase_slowdown = False
env_cfg.perturbation_probability = 0.0
env_cfg.events.actuator_gains = None
env_cfg.events.object_physics_material = None
env_cfg.events.table_physics_material = None
env_cfg.events.object_mass = None
if hasattr(env_cfg.events, "object_com"):
    env_cfg.events.object_com = None
if hasattr(env_cfg.events, "reset_gravity"):
    env_cfg.events.reset_gravity = None

if kp_override is not None:
    env_cfg.kp = kp_override
if kd_override is not None:
    env_cfg.kd = kd_override
if effort_override is not None:
    env_cfg.effort_limit = effort_override
if args.velocity_limit is not None:
    env_cfg.velocity_limit = args.velocity_limit
if args.actuator_type is not None:
    env_cfg.actuator_type = args.actuator_type

traj_candidates = sorted(glob.glob("reference_trajectories/box_hinge_ur5e/*.npz"))
if not traj_candidates:
    raise FileNotFoundError("No reference trajectory found in reference_trajectories/box_hinge_ur5e/")
env_cfg.trajectory_path = traj_candidates[0]
print(f"[INFO] kp = {env_cfg.kp}")
print(f"[INFO] kd = {env_cfg.kd}")

# ---------- Free-run ----------

env = gym.make(args.task, cfg=env_cfg)
direct_env = env.unwrapped
ur5e = direct_env.ur5e
sim = direct_env.sim
device = direct_env.device
env.reset()

init_q_t = torch.tensor(init_q[None, :], dtype=torch.float32, device=device)
zero_qd_t = torch.zeros((1, 6), dtype=torch.float32, device=device)
target_q_t = torch.tensor(target_q[None, :], dtype=torch.float32, device=device)
env_ids = torch.tensor([0], dtype=torch.long, device=device)

# Start exactly where URSim started: init pose, fully at rest.
ur5e.write_joint_state_to_sim(init_q_t, zero_qd_t, env_ids=env_ids)

sim_q = np.zeros((steps, 6), dtype=np.float64)
sim_qd = np.zeros((steps, 6), dtype=np.float64)

for i in range(steps):
    # Record at the start of the tick so sim_q[0] == init (rest), matching the URSim
    # capture convention (sample 0 = resting state before the command takes effect).
    sim_q[i] = ur5e.data.joint_pos[0].detach().cpu().numpy()
    sim_qd[i] = ur5e.data.joint_vel[0].detach().cpu().numpy()
    # Re-assert the fixed target every tick (servoJ does the same on URSim). Free-run:
    # no write_joint_state here, so the PD dynamics accumulate.
    ur5e.set_joint_position_target(target_q_t)

    ur5e.write_data_to_sim()
    sim.step(render=False)
    ur5e.update(src_dt)

# ---------- Metrics ----------

q_err = sim_q - ursim_q
rms_q_per_joint = np.sqrt(np.mean(q_err ** 2, axis=0))
final_err_per_joint = np.abs(q_err[-1])

print()
print("=" * 70)
print(f"Step-response match over {steps} ticks ({steps*src_dt*1e3:.0f} ms):")
print(f"{'joint':<14} {'RMS err (rad)':>14} {'final err (rad)':>16}")
for jn, r, fe in zip(JOINT_LABELS, rms_q_per_joint, final_err_per_joint):
    print(f"  {jn:<12} {r:>14.6f} {fe:>16.6f}")
print(f"  {'TOTAL':<12} {np.sqrt(np.mean(q_err**2)):>14.6f}")
print("=" * 70)

# ---------- Save comparison npz (before the possibly-blocking plot) ----------

if args.out:
    np.savez(args.out, init_q=init_q, target_q=target_q,
             ursim_q=ursim_q, ursim_qd=ursim_qd, sim_q=sim_q, sim_qd=sim_qd,
             src_dt=src_dt, rms_q_per_joint=rms_q_per_joint)
    print(f"[INFO] Comparison saved → {args.out}")

# ---------- Plot ----------
# Always build the figure. With --plot: save headless (Agg, no display needed).
# Without --plot: show it interactively (needs a display / $DISPLAY).

import matplotlib
if args.plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_ms = np.arange(steps) * src_dt * 1e3
fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
for i, (ax, name) in enumerate(zip(axes.flat, JOINT_LABELS)):
    ax.plot(t_ms, ursim_q[:, i], "o-", ms=3, label="URSim actual", color="tab:blue")
    ax.plot(t_ms, sim_q[:, i], "s-", ms=3, label="IsaacSim actual", color="tab:orange")
    ax.axhline(init_q[i], color="gray", ls="--", alpha=0.6, label="init")
    ax.axhline(target_q[i], color="tab:green", ls=":", alpha=0.8, label="target")
    ax.set_title(f"{name}  (RMS {rms_q_per_joint[i]:.4f} rad)", fontsize=10)
    ax.set_ylabel("position (rad)")
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8, loc="best")
axes[-1, 0].set_xlabel("time (ms)")
axes[-1, 1].set_xlabel("time (ms)")
fig.suptitle(f"Step-response match — {os.path.basename(args.step_npz)}\n"
             f"kp={env_cfg.kp}  kd={env_cfg.kd}", fontsize=10)
plt.tight_layout()

if args.plot:
    plt.savefig(args.plot, dpi=110)
    print(f"[INFO] Plot saved → {args.plot}")
else:
    print("[INFO] No --plot given: showing plot interactively (not saving). "
          "Close the window to exit.")
    plt.show()

simulation_app.close()
