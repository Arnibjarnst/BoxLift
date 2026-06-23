"""One-step prediction error: how closely does IsaacSim's PD match URSim/real dynamics?

For each step t in a URSim/real rollout (saved by ur_rtde_real_time.py):
  1. Force IsaacSim joints to (actual_q[t], actual_qd[t]) — same starting state as real.
  2. Set joint target to target_q[t] — same command as real.
  3. Step physics by one src_dt step.
  4. Compare IsaacSim's resulting joint pos/vel to URSim's actual_q[t+1] / actual_qd[t+1].

The per-step error is the residual between IsaacSim's local PD response and URSim's
under matched conditions. Summed across the trajectory, this directly measures how
well IsaacSim's (kp, kd) reproduces real closed-loop dynamics. No drift accumulation
since we re-sync the state every step.

Usage:
    python scripts/match_ursim_dynamics.py <ursim_npz> --kp 300 --kd 45
    python scripts/match_ursim_dynamics.py <ursim_npz> --kp 300,300,300,28,28,28 --kd 45,45,45,4.2,4.2,4.2
    python scripts/match_ursim_dynamics.py <ursim_npz> --out comparison.npz

The URSim rollout must contain actual_q and target_q (saved by ur_rtde_real_time.py).
If actual_qd isn't present, it's derived from actual_q via centered finite difference.
"""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("ursim_npz", type=str, help="Path to URSim rollout npz (from ur_rtde_real_time.py)")
parser.add_argument("--task", type=str, default="Template-Boxhinge-Direct-v0")
parser.add_argument("--kp", type=str, default=None,
                    help="kp scalar or comma-separated per-joint list (overrides cfg)")
parser.add_argument("--kd", type=str, default=None,
                    help="kd scalar or comma-separated per-joint list (overrides cfg)")
parser.add_argument("--effort_limit", type=str, default=None,
                    help="effort_limit scalar or comma-separated per-joint list (overrides cfg)")
parser.add_argument("--velocity_limit", type=float, default=None)
parser.add_argument("--actuator_type", type=str, default=None,
                    help='"Implicit" or "IdealPD"; overrides cfg.')
parser.add_argument("--max_steps", type=int, default=None,
                    help="Limit to first N steps of the URSim rollout.")
parser.add_argument("--out", type=str, default=None, help="Output npz path for full comparison data.")
parser.add_argument("--plot", type=str, default=None,
                    help="Output PNG path for joint-trajectory comparison plot. URSim vs IsaacSim vs target vs reference.")
parser.add_argument("--reference_trajectory", type=str, default=None,
                    help="Optional path to planner reference trajectory .npz. Auto-discovers from "
                         "<run_dir>/params/env.yaml when the URSim npz lives under <run_dir>/ur_rtde_logs/.")
# parser.add_argument("--verbose", action="store_true")

# AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force headless — this is a batch comparison, no rendering needed
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Imports that need IsaacSim to be loaded
import gymnasium as gym
import torch

import BoxLift.tasks  # noqa: F401, registers env


# ---------- Load URSim data ----------

d = np.load(args.ursim_npz, allow_pickle=True)
assert "actual_q" in d.files, "URSim npz missing 'actual_q'"
assert "target_q" in d.files, "URSim npz missing 'target_q'"

actual_q = np.asarray(d["actual_q"], dtype=np.float64)
target_q = np.asarray(d["target_q"], dtype=np.float64)
src_dt = float(d["src_dt"]) if "src_dt" in d.files else 1.0 / 500.0

N = len(actual_q)
if args.max_steps is not None:
    N = min(N, args.max_steps)
    actual_q = actual_q[:N]
    target_q = target_q[:N]

print(f"[INFO] Loaded URSim rollout: {N} steps at {1/src_dt:.0f} Hz from {args.ursim_npz}")

# Derive actual_qd via centered finite difference if not stored
if "actual_qd" in d.files and d["actual_qd"].shape == actual_q.shape:
    actual_qd = np.asarray(d["actual_qd"], dtype=np.float64)
    print("[INFO] Using actual_qd from npz.")
else:
    actual_qd = np.zeros_like(actual_q)
    actual_qd[1:-1] = (actual_q[2:] - actual_q[:-2]) / (2.0 * src_dt)
    actual_qd[0]    = (actual_q[1]  - actual_q[0])  / src_dt
    actual_qd[-1]   = (actual_q[-1] - actual_q[-2]) / src_dt
    print("[INFO] actual_qd derived from actual_q via central finite difference.")


# ---------- Load reference trajectory (optional, for plotting) ----------

import os

def _autodiscover_ref_trajectory(input_path: str):
    """Look for the training run's env.yaml two dirs above the URSim npz (matching the
    layout ur_rtde_real_time.py writes: <run_dir>/ur_rtde_logs/<run>.npz alongside
    <run_dir>/params/env.yaml) and pull `trajectory_path` from it."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_path)))
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        return None
    try:
        import yaml as _yaml
        with open(env_yaml, "r") as f:
            env_cfg_yaml = _yaml.unsafe_load(f)
        path = env_cfg_yaml.get("trajectory_path") if isinstance(env_cfg_yaml, dict) else None
    except Exception as e:
        print(f"[WARN] failed to parse {env_yaml}: {e}")
        return None
    if not path or not os.path.isfile(path):
        return None
    return path


reference_q = None  # shape (N, 6) — planner reference joints interp'd at URSim's phase
ref_traj_path = args.reference_trajectory or _autodiscover_ref_trajectory(args.ursim_npz)
if ref_traj_path is not None and "phase" in d.files:
    print(f"[INFO] Loading reference trajectory from {ref_traj_path}")
    ref_data = np.load(ref_traj_path)
    ref_joints = np.asarray(ref_data["joints"], dtype=np.float64)  # (T_traj, 6)
    phase = np.asarray(d["phase"], dtype=np.float64)[:N]
    T_traj = ref_joints.shape[0]
    idx = np.clip(phase, 0.0, T_traj - 1 - 1e-6)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, T_traj - 1)
    a = (idx - i0)[:, None]
    reference_q = (1.0 - a) * ref_joints[i0] + a * ref_joints[i1]
elif args.plot and "phase" not in d.files:
    print("[WARN] URSim npz has no 'phase'; reference trajectory cannot be aligned.")


# ---------- Parse kp/kd overrides ----------

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def _parse_gain(s, name):
    """Parse a kp/kd CLI value. Returns:
       - None if s is None,
       - float if s is a single value (uniform across joints),
       - dict {joint_name: float} for per-joint (6 comma-separated values).
    IsaacLab actuator cfg accepts float (uniform) or dict (per-joint regex → value).
    """
    if s is None:
        return None
    parts = s.split(",")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) != 6:
        raise ValueError(f"{name} list must have 1 or 6 entries, got {len(parts)}")
    return {name_: float(v) for name_, v in zip(UR5E_JOINT_NAMES, parts)}

kp_override = _parse_gain(args.kp, "kp")
kd_override = _parse_gain(args.kd, "kd")
effort_override = _parse_gain(args.effort_limit, "effort_limit")


# ---------- Configure env ----------

from BoxLift.tasks.direct.boxhinge.boxhinge_env_cfg import BoxhingeEnvCfg

env_cfg: BoxhingeEnvCfg = BoxhingeEnvCfg()
env_cfg.scene.num_envs = 1
env_cfg.physics_dt = src_dt
env_cfg.decimation = 1
env_cfg.episode_length_s = N * src_dt * 2  # plenty of headroom

# Disable everything that would interfere or randomize
env_cfg.obs_obj_pos_noise = 0.0
env_cfg.obs_obj_ori_noise = 0.0
env_cfg.obs_obj_pos_bias_std = 0.0
env_cfg.obs_obj_ori_bias_std = 0.0
env_cfg.obs_obj_delay_steps = 0
env_cfg.obs_obj_update_period = 1
env_cfg.voc_enabled = False
env_cfg.enable_phase_slowdown = False
env_cfg.perturbation_probability = 0.0
# Disable actuator-gain randomization so we test the kp/kd we set, not a perturbed version.
env_cfg.events.actuator_gains = None
env_cfg.events.object_physics_material = None
env_cfg.events.table_physics_material = None
env_cfg.events.object_mass = None
if hasattr(env_cfg.events, "object_com"):
    env_cfg.events.object_com = None
if hasattr(env_cfg.events, "reset_gravity"):
    env_cfg.events.reset_gravity = None

# Apply overrides
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

# Some trajectory is required for env init; just pick any existing one
import os
import glob
traj_candidates = sorted(glob.glob("reference_trajectories/box_hinge_ur5e/*.npz"))
if not traj_candidates:
    raise FileNotFoundError("No reference trajectory found in reference_trajectories/box_hinge_ur5e/")
env_cfg.trajectory_path = traj_candidates[0]
print(f"[INFO] Using placeholder trajectory: {env_cfg.trajectory_path}")
print(f"[INFO] kp = {env_cfg.kp}")
print(f"[INFO] kd = {env_cfg.kd}")


# ---------- Create env and run comparison ----------

env = gym.make(args.task, cfg=env_cfg)
direct_env = env.unwrapped
ur5e = direct_env.ur5e
sim = direct_env.sim
device = direct_env.device

# Reset once to put env in a valid state
env.reset()

# Tensors on device
actual_q_t  = torch.tensor(actual_q,  dtype=torch.float32, device=device)
actual_qd_t = torch.tensor(actual_qd, dtype=torch.float32, device=device)
target_q_t  = torch.tensor(target_q,  dtype=torch.float32, device=device)
env_ids = torch.tensor([0], dtype=torch.long, device=device)

sim_q_next  = np.zeros((N - 1, 6), dtype=np.float64)
sim_qd_next = np.zeros((N - 1, 6), dtype=np.float64)

print(f"[INFO] Running one-step prediction over {N-1} steps...")

for t in range(N - 1):
    # Force IsaacSim to URSim's state at time t (kinematic write — bypasses dynamics).
    ur5e.write_joint_state_to_sim(
        actual_q_t[t:t+1], actual_qd_t[t:t+1], env_ids=env_ids
    )
    # Set the joint position target the same way ur_rtde_real_time.py's servoJ command did.
    ur5e.set_joint_position_target(target_q_t[t:t+1])
    # Push target buffer to PhysX.
    ur5e.write_data_to_sim()
    # Advance physics one src_dt step.
    sim.step(render=False)
    # Refresh articulation data buffers from PhysX.
    ur5e.update(src_dt)
    # Record the result.
    sim_q_next[t]  = ur5e.data.joint_pos[0].detach().cpu().numpy()
    sim_qd_next[t] = ur5e.data.joint_vel[0].detach().cpu().numpy()

    if args.verbose and t % 500 == 0:
        err = np.linalg.norm(sim_q_next[t] - actual_q[t + 1])
        print(f"  t={t}/{N-1}, step q-error={err:.5f}")


# ---------- Metrics ----------

ursim_q_next  = actual_q[1:N]
ursim_qd_next = actual_qd[1:N]

q_err  = sim_q_next  - ursim_q_next
qd_err = sim_qd_next - ursim_qd_next

rms_q_per_joint  = np.sqrt(np.mean(q_err ** 2, axis=0))
rms_qd_per_joint = np.sqrt(np.mean(qd_err ** 2, axis=0))
rms_q_total      = float(np.sqrt(np.mean(q_err ** 2)))
rms_qd_total     = float(np.sqrt(np.mean(qd_err ** 2)))

joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

print()
print("=" * 70)
print(f"One-step RMS pos error per joint (rad):")
for n, e in zip(joint_names, rms_q_per_joint):
    print(f"  {n:<14}: {e:.6f}")
print(f"  TOTAL          : {rms_q_total:.6f}")
print()
print(f"One-step RMS vel error per joint (rad/s):")
for n, e in zip(joint_names, rms_qd_per_joint):
    print(f"  {n:<14}: {e:.6f}")
print(f"  TOTAL          : {rms_qd_total:.6f}")
print("=" * 70)


# ---------- Plot ----------

if args.plot:
    # Use Agg backend so this works headless (no GUI dependency).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Time axis. We compare URSim's actual_q[t+1] to IsaacSim's prediction-from-t.
    # Both are plotted at time (t+1)*src_dt — the "next state" timeline.
    t_next = np.arange(1, N) * src_dt          # length N-1
    t_full = np.arange(N) * src_dt             # length N (for full URSim trace + target)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for j in range(6):
        ax = axes[j]
        # Planner reference joints[phase[t]] — what the trained policy was nominally
        # trying to track. Drawn first so the actual/predicted lines sit on top.
        if reference_q is not None:
            ax.plot(t_full, reference_q[:, j], color="tab:green", lw=1.0, ls=":",
                    alpha=0.85, label="reference (planner joints)")
        # URSim actual trajectory (full).
        ax.plot(t_full, actual_q[:, j], color="tab:blue", lw=1.2,
                label="URSim actual")
        # IsaacSim one-step predicted state. Plot at "next-state" timeline so it
        # overlays URSim's actual_q[1:N] for direct comparison.
        ax.plot(t_next, sim_q_next[:, j], color="tab:red", lw=1.0, alpha=0.85,
                label="IsaacSim 1-step prediction")
        # Target command (what servoJ asked for). On the "command" timeline (the
        # state at t+1 was driven by target at t — but we plot target at t to show
        # what the controller saw).
        ax.plot(t_full, target_q[:N, j], color="black", lw=0.8, ls="--", alpha=0.5,
                label="target_q (cmd)")
        ax.set_title(f"joint {j}: {joint_names[j]}  (RMS = {rms_q_per_joint[j]:.5f} rad)",
                     fontsize=10)
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("time (s)")
    axes[-2].set_xlabel("time (s)")

    kp_str = str(env_cfg.kp) if not isinstance(env_cfg.kp, dict) else \
        "{" + ", ".join(f"{k}={v}" for k, v in env_cfg.kp.items()) + "}"
    kd_str = str(env_cfg.kd) if not isinstance(env_cfg.kd, dict) else \
        "{" + ", ".join(f"{k}={v}" for k, v in env_cfg.kd.items()) + "}"
    fig.suptitle(f"URSim vs IsaacSim (1-step prediction)\n"
                 f"kp = {kp_str}\nkd = {kd_str}\n"
                 f"total RMS q = {rms_q_total:.5f} rad", fontsize=10)
    plt.tight_layout()
    fig.savefig(args.plot, dpi=120)
    plt.close(fig)
    print(f"[INFO] Saved comparison plot to {args.plot}")


# ---------- Save ----------

if args.out:
    np.savez(
        args.out,
        sim_q_next=sim_q_next,
        sim_qd_next=sim_qd_next,
        ursim_q_next=ursim_q_next,
        ursim_qd_next=ursim_qd_next,
        target_q=target_q[:N],
        src_dt=src_dt,
        kp=np.asarray(env_cfg.kp if not isinstance(env_cfg.kp, dict) else list(env_cfg.kp.values())),
        kd=np.asarray(env_cfg.kd if not isinstance(env_cfg.kd, dict) else list(env_cfg.kd.values())),
        rms_q_per_joint=rms_q_per_joint,
        rms_qd_per_joint=rms_qd_per_joint,
        rms_q_total=rms_q_total,
        rms_qd_total=rms_qd_total,
        joint_names=np.asarray(joint_names),
    )
    print(f"[INFO] Saved comparison data to {args.out}")


env.close()
simulation_app.close()
