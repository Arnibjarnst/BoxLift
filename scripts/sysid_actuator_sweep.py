"""Actuator / gain system identification sweep.

Reads a real-robot CSV (from ur_rtde_fixed_traj.py or ur_rtde_test.py),
replays the joint targets through IsaacLab with many actuator / gain
combinations **all in parallel**, and reports RMSE vs real joint positions.

Architecture: one sim, one scene.  Each env contains multiple robots
(one per actuator-type group: Implicit, IdealPD, DelayedPD variants).
Envs are cloned for the gain sweep — env_i gets gain combo i applied to
every robot.  All robots in all envs step in a single GPU pass.

Usage:
    python scripts/sysid_actuator_sweep.py \
        --csv logs/ur_rtde/trajectory_dur=10.0_delay=0.5_ampl=30.0_sim.csv \
        --headless
"""

import argparse
import ast
import csv
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep actuator parameters against real robot data.")
parser.add_argument("--csv", type=str, required=True, help="Path to real-robot CSV file.")
parser.add_argument("--proportion", type=float, default=1.0, help="Proportion of trajectory to simulate")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.actuators.actuator_base_cfg import ActuatorBaseCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


# ---------------------------------------------------------------------------
# Custom actuator: LookaheadP — mimics servoJ lookahead behaviour
# ---------------------------------------------------------------------------


class LookaheadPActuator(ActuatorBase):
    """P-controller on a lookahead position: q_predicted = q + qd * lookahead_time.

    torque = gain * (q_target - q_predicted)

    gain is stored in ``self.stiffness``.
    dampint is stored in ``self.damping``.
    """

    def reset(self, env_ids):
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        q_predicted = joint_pos + joint_vel * self.cfg.lookahead_time
        error_pos = control_action.joint_positions - q_predicted
        error_vel = control_action.joint_velocities - joint_vel

        self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts

        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class LookaheadPActuatorCfg(ActuatorBaseCfg):
    """Config for a P-only actuator that controls on a velocity-extrapolated position."""

    class_type: type = LookaheadPActuator
    lookahead_time: float = 0.0
    """Time (s) to extrapolate current joint velocity forward."""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_USD = "./robots/ur5e.usd"

PHYSICS_DT = 1.0 / 500.0
DECIMATION = 1
CONTROL_DT = PHYSICS_DT * DECIMATION  # 0.02 s = 50 Hz

CSV_HZ = 500
CONTROL_HZ = int(1.0 / CONTROL_DT)
DOWNSAMPLE_RATIO = CSV_HZ // CONTROL_HZ  # 10

UR5E_EFFORT_LIMITS = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

# UR5E_EFFORT_LIMITS = 10000000
UR5E_VELOCITY_LIMITS = np.pi

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
NUM_JOINTS = 6

_act_kwargs = dict(joint_names_expr=[".*"], velocity_limit=UR5E_VELOCITY_LIMITS, velocity_limit_sim=UR5E_VELOCITY_LIMITS, effort_limit=UR5E_EFFORT_LIMITS, effort_limit_sim=UR5E_EFFORT_LIMITS)


# ---------------------------------------------------------------------------
# Actuator-type groups — each becomes a separate robot in every env
# ---------------------------------------------------------------------------

# Each group key maps to (actuator_cfg_factory, list_of_gain_configs).
# Gains vary per env; the actuator type is fixed per robot.

GROUPS: dict[str, dict] = {
    "Implicit": {"factory": lambda kp, kd, **_: ImplicitActuatorCfg(stiffness=kp, damping=kd, **_act_kwargs)},
    # "IdealPD":  {"factory": lambda kp, kd, **_: IdealPDActuatorCfg(stiffness=kp, damping=kd, **_act_kwargs)},
    # "DelayedPD_0_1": {"factory": lambda kp, kd, **_: DelayedPDActuatorCfg(stiffness=kp, damping=kd, min_delay=0, max_delay=1, **_act_kwargs)},
    # "DelayedPD_1_2": {"factory": lambda kp, kd, **_: DelayedPDActuatorCfg(stiffness=kp, damping=kd, min_delay=1, max_delay=2, **_act_kwargs)},
    # "DelayedPD_1_3": {"factory": lambda kp, kd, **_: DelayedPDActuatorCfg(stiffness=kp, damping=kd, min_delay=1, max_delay=3, **_act_kwargs)},
    # "LookaheadP_30": {"factory": lambda kp, kd, **_: LookaheadPActuatorCfg(stiffness=kp, damping=kd, lookahead_time=0.03, **_act_kwargs)},
    # "LookaheadP_50": {"factory": lambda kp, kd, **_: LookaheadPActuatorCfg(stiffness=kp, damping=kd, lookahead_time=0.05, **_act_kwargs)},
    # "LookaheadP_100": {"factory": lambda kp, kd, **_: LookaheadPActuatorCfg(stiffness=kp,damping=kd, lookahead_time=0.1,  **_act_kwargs)},
}


def build_sweep_configs() -> dict[str, list[dict]]:
    """Build gain configs, grouped by actuator-type key."""
    out: dict[str, list[dict]] = defaultdict(list)

    # Gain grid shared across all types
    uniform_gains = []
    for kp in [28, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 10000]:
        for kd_ratio in [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]:
            uniform_gains.append({"kp": float(kp), "kd": kp * kd_ratio})

    # Per-joint gains (UR5e torque ratio: 150 Nm shoulder/elbow, 28 Nm wrist)
    perjoint_gains = []
    for kp_mult in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 30.0]:
        kp_vals = [150.0 * kp_mult] * 3 + [28.0 * kp_mult] * 3
        for kd_ratio in [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]:
            kd_vals = [v * kd_ratio for v in kp_vals]
            perjoint_gains.append({"kp": kp_vals, "kd": kd_vals})

    # All types get both uniform and per-joint gains
    all_types = list(GROUPS.keys())

    for type_key in all_types:
        for g in uniform_gains:
            kp_label = g["kp"] if isinstance(g["kp"], (int, float)) else "pj"
            kd_val = g["kd"] if isinstance(g["kd"], (int, float)) else g["kd"][0]
            out[type_key].append({
                "name": f"{type_key}_kp={kp_label}_kd={kd_val:.1f}",
                **g,
            })
        for g in perjoint_gains:
            out[type_key].append({
                "name": f"{type_key}_pj_m={g['kp'][0]/150:.1f}_kr={g['kd'][0]/g['kp'][0]:.2f}",
                **g,
            })

    return dict(out)


# ---------------------------------------------------------------------------
# Parse CSV
# ---------------------------------------------------------------------------


def parse_csv(path: str):
    """Parse CSV and downsample 500 Hz → 50 Hz."""
    rows = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            if step < 0:
                continue
            target_q = np.array(ast.literal_eval(row["target_q"]))
            actual_q = np.array(ast.literal_eval(row["actual_q"]))
            actual_qd = np.array(ast.literal_eval(row["actual_qd"]))
            applied_torque = np.array(ast.literal_eval(row["applied_torque"]))
            rows.append((step, target_q, actual_q, actual_qd, applied_torque))

    rows.sort(key=lambda x: x[0])
    rows_to_keep = int(len(rows) * args_cli.proportion)
    rows = rows[:rows_to_keep]

    target_q = np.array([r[1] for r in rows])
    actual_q = np.array([r[2] for r in rows])
    actual_qd = np.array([r[3] for r in rows])
    applied_torque = np.array([r[4] for r in rows])
    return target_q[::DOWNSAMPLE_RATIO], actual_q[::DOWNSAMPLE_RATIO], actual_qd[::DOWNSAMPLE_RATIO], applied_torque[::DOWNSAMPLE_RATIO]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg_to_gain_row(cfg: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert one config's kp/kd to (6,) tensors."""
    def _to_1d(val):
        if isinstance(val, (int, float)):
            return torch.full((NUM_JOINTS,), float(val), device=device)
        if isinstance(val, list):
            return torch.tensor(val, dtype=torch.float32, device=device)
        if isinstance(val, dict):
            return torch.tensor(list(val.values()), dtype=torch.float32, device=device)
        return torch.tensor(val, dtype=torch.float32, device=device).reshape(NUM_JOINTS)
    return _to_1d(cfg["kp"]), _to_1d(cfg["kd"])


def _make_art_cfg(prim_name: str, actuator_cfg, x_offset: float = 0.0) -> ArticulationCfg:
    """Build ArticulationCfg for a robot named `prim_name` inside each env."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=10.0, enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=4,
                solver_velocity_iteration_count=0, sleep_threshold=0.005,
                stabilization_threshold=0.001, fix_root_link=True,
            ),
            copy_from_source=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(x_offset, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={"joints": actuator_cfg},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print(f"\n{'=' * 80}")
    print("Actuator System Identification Sweep")
    print(f"{'=' * 80}")

    joint_targets, joint_pos, joint_vel, applied_torques = parse_csv(args_cli.csv)
    T = len(joint_targets)
    print(f"CSV: {args_cli.csv}")
    print(f"  Downsampled {CSV_HZ} Hz -> {CONTROL_HZ} Hz (ratio {DOWNSAMPLE_RATIO})")
    print(f"  Control steps: {T}  |  Duration: {T * CONTROL_DT:.2f} s")
    print(f"  Sim: physics_dt={PHYSICS_DT}  decimation={DECIMATION}  control_dt={CONTROL_DT}")

    group_configs = build_sweep_configs()

    # Number of envs = number of gain configs (same grid for all groups)
    N = len(next(iter(group_configs.values())))
    total_configs = N * len(group_configs)
    print(f"  Actuator groups: {len(group_configs)} ({', '.join(group_configs.keys())})")
    print(f"  Total configs: {total_configs}  |  Envs (parallel): {N}")
    print(f"  Robots per env: {len(group_configs)}\n")

    device = args_cli.device if args_cli.device else "cuda:0"

    # --- Build scene config dynamically with one robot per group ---
    scene_fields = {}
    group_art_cfgs = {}  # group_key -> (prim_name, art_cfg)

    # Space robots apart within each env so they don't collide
    ROBOT_SPACING = 2.0  # meters between robots within an env
    active_groups = [k for k in GROUPS if k in group_configs]

    for idx, group_key in enumerate(active_groups):
        group_info = GROUPS[group_key]
        first_gain = group_configs[group_key][0]
        act_cfg = group_info["factory"](kp=first_gain["kp"], kd=first_gain["kd"])
        prim_name = f"Robot_{group_key}"
        x_offset = idx * ROBOT_SPACING
        art_cfg = _make_art_cfg(prim_name, act_cfg, x_offset=x_offset)
        scene_fields[group_key] = art_cfg
        group_art_cfgs[group_key] = prim_name

    # Torque replay baseline robot (IdealPD with kp=0, kd=0 → pure effort passthrough)
    TORQUE_REPLAY_KEY = "TorqueReplay"
    torque_act_cfg = IdealPDActuatorCfg(stiffness=0.0, damping=0.0, **_act_kwargs)
    torque_x_offset = len(active_groups) * ROBOT_SPACING
    scene_fields[TORQUE_REPLAY_KEY] = _make_art_cfg(
        f"Robot_{TORQUE_REPLAY_KEY}", torque_act_cfg, x_offset=torque_x_offset,
    )

    # Dynamically create the scene config class with one ArticulationCfg field per group
    SysidSceneCfg = configclass(
        type("SysidSceneCfg", (InteractiveSceneCfg,), {
            "__annotations__": {k: ArticulationCfg for k in scene_fields},
            **scene_fields,
        })
    )

    # --- Create sim + scene ---
    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=device, render_interval=DECIMATION)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.0, 0.0, 2.5], [0.0, 0.0, 0.0])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(size=(300,300)))
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    env_spacing = len(active_groups) * ROBOT_SPACING + 2.0  # enough room for all robots + margin
    scene = InteractiveScene(SysidSceneCfg(num_envs=N, env_spacing=env_spacing, replicate_physics=True))
    sim.reset()

    # --- Per-env gain setup + initial state for each robot ---
    init_pos_1 = torch.tensor(joint_pos[0], dtype=torch.float32, device=device)
    init_pos = init_pos_1.unsqueeze(0).expand(N, -1).contiguous()
    init_vel_1 = torch.tensor(joint_vel[0], dtype=torch.float32, device=device)
    init_vel = init_vel_1.unsqueeze(0).expand(N, -1).contiguous()

    targets_t = torch.tensor(joint_targets, dtype=torch.float32, device=device)
    targets_t = targets_t.unsqueeze(0).expand(N, -1, -1)  # (N, T, 6)

    real_pos_t = torch.tensor(joint_pos, dtype=torch.float32, device=device)
    real_pos_t = real_pos_t.unsqueeze(0).expand(N, -1, -1)  # (N, T, 6)

    real_vel_t = torch.tensor(joint_vel, dtype=torch.float32, device=device)
    real_vel_t = real_vel_t.unsqueeze(0).expand(N, -1, -1)  # (N, T, 6)

    torques_t = torch.tensor(applied_torques, dtype=torch.float32, device=device)
    torques_t = torques_t.unsqueeze(0).expand(N, -1, -1)  # (N, T, 6)

    # --- Torque replay baseline robot setup ---
    torque_robot: Articulation = scene.articulations[TORQUE_REPLAY_KEY]
    torque_robot.write_joint_state_to_sim(init_pos, init_vel)
    torque_robot.reset()

    robots: dict[str, Articulation] = {}
    for group_key in group_art_cfgs:
        robot = scene.articulations[group_key]
        robots[group_key] = robot

        cfgs = group_configs[group_key]
        actuator = robot.actuators["joints"]
        # Build per-env gain tensors
        kp_all = torch.zeros(N, NUM_JOINTS, device=device)
        kd_all = torch.zeros(N, NUM_JOINTS, device=device)
        for i, cfg in enumerate(cfgs):
            kp_all[i], kd_all[i] = _cfg_to_gain_row(cfg, device)

        actuator.stiffness[:] = kp_all
        actuator.damping[:] = kd_all

        # Implicit pushes gains to PhysX
        if group_key == "Implicit":
            robot.write_joint_stiffness_to_sim(kp_all, joint_ids=actuator.joint_indices)
            robot.write_joint_damping_to_sim(kd_all, joint_ids=actuator.joint_indices)

        robot.write_joint_state_to_sim(init_pos, init_vel)

        robot.reset()

    NUM_EPISODES = 1
    print(f"Scene ready. Stepping trajectory x{NUM_EPISODES} episodes...\n")

    # --- Accumulate squared errors across episodes ---
    # Running sums: (N, T, 6) for per-joint, scalar accumulators for total/max
    all_robot_keys = list(robots.keys()) + [TORQUE_REPLAY_KEY]
    sum_sq_pos_error: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in all_robot_keys}
    sum_sq_vel_error: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in all_robot_keys}
    sum_pos_error: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in all_robot_keys}
    sum_vel_error: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in all_robot_keys}
    max_pos_error_acc: dict[str, torch.Tensor] = {k: torch.zeros(N, device=device) for k in all_robot_keys}
    max_vel_error_acc: dict[str, torch.Tensor] = {k: torch.zeros(N, device=device) for k in all_robot_keys}

    # Need to average for delayed_PD
    for episode in range(NUM_EPISODES):
        # Reset all robots to initial state
        env_ids = torch.arange(N, device=device)
        for group_key, robot in robots.items():
            robot.write_joint_state_to_sim(init_pos, init_vel, env_ids=env_ids)
            robot.reset(env_ids)
        torque_robot.write_joint_state_to_sim(init_pos, init_vel, env_ids=env_ids)
        torque_robot.reset(env_ids)

        sim_positions: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in all_robot_keys}
        sim_velocities: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in all_robot_keys}

        for t in range(T):
            for k, robot in robots.items():
                sim_positions[k][:, t, :] = robot.data.joint_pos
                sim_velocities[k][:, t, :] = robot.data.joint_vel
            sim_positions[TORQUE_REPLAY_KEY][:, t, :] = torque_robot.data.joint_pos
            sim_velocities[TORQUE_REPLAY_KEY][:, t, :] = torque_robot.data.joint_vel

            for robot in robots.values():
                robot.set_joint_position_target(targets_t[:, t, :])
            torque_robot.set_joint_effort_target(torques_t[:, t, :])

            for _ in range(DECIMATION):
                for robot in robots.values():
                    robot.write_data_to_sim()
                torque_robot.write_data_to_sim()
                sim.step(render=False)
                for robot in robots.values():
                    robot.update(PHYSICS_DT)
                torque_robot.update(PHYSICS_DT)

            sim.render()

        # Accumulate errors for this episode
        for group_key in all_robot_keys:
            sp = sim_positions[group_key]
            sv = sim_velocities[group_key]
            pos_error = sp - real_pos_t  # (N, T, 6)
            sum_sq_pos_error[group_key] += pos_error**2
            sum_pos_error[group_key] += pos_error
            vel_error = sv - real_vel_t  # (N, T, 6)
            sum_sq_vel_error[group_key] += vel_error**2
            sum_vel_error[group_key] += vel_error

            max_pos_error_acc[group_key] = torch.maximum(max_pos_error_acc[group_key], pos_error.abs().amax(dim=(1, 2)))
            max_vel_error_acc[group_key] = torch.maximum(max_vel_error_acc[group_key], vel_error.abs().amax(dim=(1, 2)))

        print(f"  Episode {episode + 1}/{NUM_EPISODES} done")
    
    # --- Torque replay baseline metrics (env 0 only, all envs are identical) ---
    tr_pos_mse = sum_sq_pos_error[TORQUE_REPLAY_KEY][0] / NUM_EPISODES  # (T, 6)
    tr_vel_mse = sum_sq_vel_error[TORQUE_REPLAY_KEY][0] / NUM_EPISODES
    torque_baseline = {
        "name": "TorqueReplay (baseline)",
        "total_pos_rmse": float(torch.sqrt(tr_pos_mse.mean())),
        "total_vel_rmse": float(torch.sqrt(tr_vel_mse.mean())),
        "max_pos_error": float(max_pos_error_acc[TORQUE_REPLAY_KEY][0]),
        "max_vel_error": float(max_vel_error_acc[TORQUE_REPLAY_KEY][0]),
        "per_joint_pos_rmse": torch.sqrt(tr_pos_mse.mean(dim=0)).cpu().numpy(),
        "per_joint_vel_rmse": torch.sqrt(tr_vel_mse.mean(dim=0)).cpu().numpy(),
        "pos_rmse_over_time": torch.sqrt(tr_pos_mse.mean(dim=1)).cpu().numpy(),
        "vel_rmse_over_time": torch.sqrt(tr_vel_mse.mean(dim=1)).cpu().numpy(),
        "pos_error_over_time": (sum_pos_error[TORQUE_REPLAY_KEY][0] / NUM_EPISODES).cpu().numpy(),
        "vel_error_over_time": (sum_vel_error[TORQUE_REPLAY_KEY][0] / NUM_EPISODES).cpu().numpy(),
    }

    print(f"\n{'=' * 100}")
    print(f"{'TORQUE REPLAY BASELINE':^100}")
    print(f"{'=' * 100}")
    print(f"  Pos RMSE: {torque_baseline['total_pos_rmse']:.4f}    Vel RMSE: {torque_baseline['total_vel_rmse']:.4f}"
          f"    Max Pos: {torque_baseline['max_pos_error']:.4f}    Max Vel: {torque_baseline['max_vel_error']:.4f}")
    print(f"  Per-joint Pos RMSE: " + "  ".join(f"{n}={v:.4f}" for n, v in zip(JOINT_NAMES, torque_baseline["per_joint_pos_rmse"])))
    print(f"  Per-joint Vel RMSE: " + "  ".join(f"{n}={v:.4f}" for n, v in zip(JOINT_NAMES, torque_baseline["per_joint_vel_rmse"])))

    # --- Compute metrics (averaged over episodes) ---
    all_results = [torque_baseline]
    for group_key, cfgs in group_configs.items():
        pos_mse = sum_sq_pos_error[group_key] / NUM_EPISODES  # (N, T, 6)
        vel_mse = sum_sq_vel_error[group_key] / NUM_EPISODES  # (N, T, 6)

        per_joint_pos_rmse = torch.sqrt(pos_mse.mean(dim=1))  # (N, 6)
        per_joint_vel_rmse = torch.sqrt(vel_mse.mean(dim=1))  # (N, 6)
        total_pos_rmse = torch.sqrt(pos_mse.mean(dim=(1, 2)))  # (N,)
        total_vel_rmse = torch.sqrt(vel_mse.mean(dim=(1, 2)))  # (N,)
        max_pos_error = max_pos_error_acc[group_key]  # (N,)
        max_vel_error = max_vel_error_acc[group_key]  # (N,)

        # Per-timestep RMSE across joints: sqrt(mean over joints of mse) -> (N, T)
        pos_rmse_t = torch.sqrt(pos_mse.mean(dim=2))  # (N, T)
        vel_rmse_t = torch.sqrt(vel_mse.mean(dim=2))  # (N, T)
        # Mean signed error per joint over time: (N, T, 6)
        pos_error_t = sum_pos_error[group_key] / NUM_EPISODES
        vel_error_t = sum_vel_error[group_key] / NUM_EPISODES

        for i, cfg in enumerate(cfgs):
            all_results.append({
                "name": cfg["name"],
                "total_pos_rmse": float(total_pos_rmse[i]),
                "total_vel_rmse": float(total_vel_rmse[i]),
                "max_pos_error": float(max_pos_error[i]),
                "max_vel_error": float(max_vel_error[i]),
                "per_joint_pos_rmse": per_joint_pos_rmse[i].cpu().numpy(),
                "per_joint_vel_rmse": per_joint_vel_rmse[i].cpu().numpy(),
                "pos_rmse_over_time": pos_rmse_t[i].cpu().numpy(),
                "vel_rmse_over_time": vel_rmse_t[i].cpu().numpy(),
                "pos_error_over_time": pos_error_t[i].cpu().numpy(),  # (T, 6)
                "vel_error_over_time": vel_error_t[i].cpu().numpy(),  # (T, 6)
            })

    # --- Print results ---
    sort_key = "max_vel_error"
    all_results.sort(key=lambda x: x[sort_key])

    print(f"\n{'=' * 100}")
    print(f"{f'RESULTS (sorted by {sort_key})':^100}")
    print(f"{'=' * 100}")
    print(f"{'Rank':<5} {'Config':<55} {'Pos RMSE':<12} {'Vel RMSE':<12} {'Max Pos':<12} {'Max Vel':<12}")
    print(f"{'-' * 110}")
    for rank, r in enumerate(all_results[:300], 1):
        print(f"{rank:<5} {r['name']:<55} {r['total_pos_rmse']:<12.3f} {r['total_vel_rmse']:<12.3f} {r['max_pos_error']:<12.3f} {r['max_vel_error']:<12.3f}")

    print(f"\n{'=' * 140}")
    print("Top 10 — Per-joint Pos RMSE:")
    print(f"{'=' * 140}")
    header = f"{'Config':<55} " + " ".join(f"{j:<14}" for j in JOINT_NAMES)
    print(header)
    print("-" * 140)
    for r in all_results[:10]:
        pj = r["per_joint_pos_rmse"]
        row = f"{r['name']:<55} " + " ".join(f"{v:<14.3f}" for v in pj)
        print(row)

    print(f"\n{'=' * 140}")
    print("Top 10 — Per-joint Vel RMSE:")
    print(f"{'=' * 140}")
    print(header)
    print("-" * 140)
    for r in all_results[:10]:
        pj = r["per_joint_vel_rmse"]
        row = f"{r['name']:<55} " + " ".join(f"{v:<14.3f}" for v in pj)
        print(row)
    print()

    # --- Interactive plotting ---
    import matplotlib.pyplot as plt

    time_axis = np.arange(T) * CONTROL_DT

    while True:
        try:
            print("Commands:")
            print("  <ranks>        — RMSE over time  (e.g. '1 3 5')")
            print("  j <rank>       — per-joint error over time for one config")
            print("  q              — quit")
            sel = input("> ").strip()
        except EOFError:
            break
        if sel.lower() == "q":
            break

        per_joint_mode = sel.lower().startswith("j ")
        tokens = sel.split()[1:] if per_joint_mode else sel.split()

        try:
            ranks = [int(x) for x in tokens]
        except ValueError:
            print("Invalid input.")
            continue
        if any(r < 1 or r > len(all_results) for r in ranks):
            print(f"Ranks must be between 1 and {len(all_results)}.")
            continue

        if per_joint_mode:
            rank = ranks[0]
            r = all_results[rank - 1]
            _, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            for j_idx, j_name in enumerate(JOINT_NAMES):
                axes[0].plot(time_axis, r["pos_error_over_time"][:, j_idx], label=j_name)
                axes[1].plot(time_axis, r["vel_error_over_time"][:, j_idx], label=j_name)

            axes[0].set_ylabel("Pos error (rad)")
            axes[0].set_title(f"#{rank} {r['name']} — per-joint position error")
            axes[0].legend()
            axes[0].grid(True)

            axes[1].set_ylabel("Vel error (rad/s)")
            axes[1].set_xlabel("Time (s)")
            axes[1].set_title(f"#{rank} {r['name']} — per-joint velocity error")
            axes[1].legend()
            axes[1].grid(True)
        else:
            _, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
            for rank in ranks:
                r = all_results[rank - 1]
                ax_pos.plot(time_axis, r["pos_rmse_over_time"], label=f"#{rank} {r['name']}")
                ax_vel.plot(time_axis, r["vel_rmse_over_time"], label=f"#{rank} {r['name']}")

            ax_pos.set_ylabel("Pos RMSE (rad)")
            ax_pos.set_title("Position RMSE over time")
            ax_pos.legend(fontsize=7)
            ax_pos.grid(True)

            ax_vel.set_ylabel("Vel RMSE (rad/s)")
            ax_vel.set_xlabel("Time (s)")
            ax_vel.set_title("Velocity RMSE over time")
            ax_vel.legend(fontsize=7)
            ax_vel.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
    simulation_app.close()
    import sys
    sys.exit(0)
