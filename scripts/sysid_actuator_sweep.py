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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROBOT_USD = "./robots/ur5e_sphere.usd"

PHYSICS_DT = 1.0 / 100.0
DECIMATION = 2
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

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
NUM_JOINTS = 6

_act_kwargs = dict(joint_names_expr=[".*"], velocity_limit=100.0, effort_limit=UR5E_EFFORT_LIMITS)


# ---------------------------------------------------------------------------
# Actuator-type groups — each becomes a separate robot in every env
# ---------------------------------------------------------------------------

# Each group key maps to (actuator_cfg_factory, list_of_gain_configs).
# Gains vary per env; the actuator type is fixed per robot.

GROUPS: dict[str, dict] = {
    "Implicit": {"factory": lambda kp, kd, **_: ImplicitActuatorCfg(stiffness=kp, damping=kd, **_act_kwargs)},
    "IdealPD":  {"factory": lambda kp, kd, **_: IdealPDActuatorCfg(stiffness=kp, damping=kd, **_act_kwargs)},
    "DelayedPD_0_1": {"factory": lambda kp, kd, **_: DelayedPDActuatorCfg(stiffness=kp, damping=kd, min_delay=0, max_delay=1, **_act_kwargs)},
    "DelayedPD_1_2": {"factory": lambda kp, kd, **_: DelayedPDActuatorCfg(stiffness=kp, damping=kd, min_delay=1, max_delay=2, **_act_kwargs)},
    "DelayedPD_1_3": {"factory": lambda kp, kd, **_: DelayedPDActuatorCfg(stiffness=kp, damping=kd, min_delay=1, max_delay=3, **_act_kwargs)},
}


def build_sweep_configs() -> dict[str, list[dict]]:
    """Build gain configs, grouped by actuator-type key."""
    out: dict[str, list[dict]] = defaultdict(list)

    # Gain grid shared across all types
    uniform_gains = []
    for kp in [28, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1000]:
        for kd_ratio in [0.025, 0.05, 0.1, 0.15, 0.2]:
            uniform_gains.append({"kp": float(kp), "kd": kp * kd_ratio})

    # Per-joint gains (UR5e torque ratio: 150 Nm shoulder/elbow, 28 Nm wrist)
    perjoint_gains = []
    for kp_mult in [0.5, 1.0, 1.5, 2.0]:
        kp_vals = [150.0 * kp_mult] * 3 + [28.0 * kp_mult] * 3
        for kd_ratio in [0.1, 0.15, 0.2]:
            kd_vals = [v * kd_ratio for v in kp_vals]
            perjoint_gains.append({"kp": kp_vals, "kd": kd_vals})

    # All types get both uniform and per-joint gains
    all_types = ["Implicit", "IdealPD", "DelayedPD_0_1", "DelayedPD_1_2", "DelayedPD_1_3"]

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
            target_q = np.array(ast.literal_eval(row["expected_q"]))
            actual_q = np.array(ast.literal_eval(row["actual_q"]))
            rows.append((step, target_q, actual_q))

    rows.sort(key=lambda x: x[0])
    targets = np.array([r[1] for r in rows])
    actuals = np.array([r[2] for r in rows])
    return targets[::DOWNSAMPLE_RATIO], actuals[::DOWNSAMPLE_RATIO]


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

    joint_targets_np, real_actual_np = parse_csv(args_cli.csv)
    T = len(joint_targets_np)
    print(f"CSV: {args_cli.csv}")
    print(f"  Downsampled {CSV_HZ} Hz -> {CONTROL_HZ} Hz (ratio {DOWNSAMPLE_RATIO})")
    print(f"  Control steps: {T}  |  Duration: {T * CONTROL_DT:.2f} s")
    print(f"  Sim: physics_dt={PHYSICS_DT}  decimation={DECIMATION}  control_dt={CONTROL_DT}")

    group_configs = build_sweep_configs()

    # Number of envs = max gain configs across groups (pad shorter groups)
    N = max(len(cfgs) for cfgs in group_configs.values())
    total_configs = sum(len(cfgs) for cfgs in group_configs.values())
    print(f"  Actuator groups: {len(group_configs)} ({', '.join(f'{k}={len(v)}' for k, v in group_configs.items())})")
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

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0))

    env_spacing = len(active_groups) * ROBOT_SPACING + 2.0  # enough room for all robots + margin
    scene = InteractiveScene(SysidSceneCfg(num_envs=N, env_spacing=env_spacing, replicate_physics=True))
    sim.reset()

    # --- Per-env gain setup + initial state for each robot ---
    init_pos_1 = torch.tensor(real_actual_np[0], dtype=torch.float32, device=device)
    init_pos = init_pos_1.unsqueeze(0).expand(N, -1).contiguous()
    init_vel = torch.zeros_like(init_pos)

    targets_t = torch.tensor(joint_targets_np, dtype=torch.float32, device=device)
    targets_t = targets_t.unsqueeze(0).expand(N, -1, -1)  # (N, T, 6)

    real_t = torch.tensor(real_actual_np, dtype=torch.float32, device=device)
    real_t = real_t.unsqueeze(0).expand(N, -1, -1)  # (N, T, 6)

    robots: dict[str, Articulation] = {}
    for group_key in group_art_cfgs:
        robot = scene.articulations[group_key]
        robots[group_key] = robot

        cfgs = group_configs[group_key]
        actuator = robot.actuators["joints"]
        n_cfgs = len(cfgs)

        # Build per-env gain tensors
        kp_all = torch.zeros(N, NUM_JOINTS, device=device)
        kd_all = torch.zeros(N, NUM_JOINTS, device=device)
        for i, cfg in enumerate(cfgs):
            kp_all[i], kd_all[i] = _cfg_to_gain_row(cfg, device)
        # Pad remaining envs with last config's gains (they'll be ignored in results)
        if n_cfgs < N:
            kp_all[n_cfgs:] = kp_all[n_cfgs - 1]
            kd_all[n_cfgs:] = kd_all[n_cfgs - 1]

        actuator.stiffness[:] = kp_all
        actuator.damping[:] = kd_all

        # Implicit pushes gains to PhysX
        if group_key == "Implicit":
            robot.write_joint_stiffness_to_sim(kp_all, joint_ids=actuator.joint_indices)
            robot.write_joint_damping_to_sim(kd_all, joint_ids=actuator.joint_indices)

        robot.write_joint_state_to_sim(init_pos, init_vel)

        robot.reset()

    print("Scene ready. Stepping trajectory...\n")

    # --- Step all robots in parallel ---
    sim_positions: dict[str, torch.Tensor] = {k: torch.zeros(N, T, NUM_JOINTS, device=device) for k in robots}

    for t in range(T):
        for k, robot in robots.items():
            sim_positions[k][:, t, :] = robot.data.joint_pos

        for robot in robots.values():
            robot.set_joint_position_target(targets_t[:, t, :])

        for _ in range(DECIMATION):
            for robot in robots.values():
                robot.write_data_to_sim()
            sim.step(render=False)
            for robot in robots.values():
                robot.update(PHYSICS_DT)

        sim.render()

    # --- Compute metrics ---
    all_results = []
    for group_key, cfgs in group_configs.items():
        sp = sim_positions[group_key]
        n_cfgs = len(cfgs)

        error = sp[:n_cfgs] - real_t[:n_cfgs]  # (n_cfgs, T, 6)

        per_joint_rmse = torch.sqrt((error**2).mean(dim=1))  # (n_cfgs, 6)
        total_rmse = torch.sqrt((error**2).mean(dim=(1, 2)))  # (n_cfgs,)
        max_error = error.abs().amax(dim=(1, 2))  # (n_cfgs,)

        for i, cfg in enumerate(cfgs):
            all_results.append({
                "name": cfg["name"],
                "total_rmse": float(total_rmse[i]),
                "max_error": float(max_error[i]),
                "per_joint_rmse": per_joint_rmse[i].cpu().numpy(),
            })

    # --- Print results ---
    all_results.sort(key=lambda x: x["total_rmse"])

    print(f"\n{'=' * 90}")
    print(f"{'RESULTS (sorted by total RMSE)':^90}")
    print(f"{'=' * 90}")
    print(f"{'Rank':<5} {'Config':<55} {'RMSE':<12} {'Max':<12}")
    print(f"{'-' * 90}")
    for rank, r in enumerate(all_results, 1):
        print(f"{rank:<5} {r['name']:<55} {r['total_rmse']:<12.3f} {r['max_error']:<12.3f}")

    print(f"\n{'=' * 140}")
    print("Top 10 — Per-joint RMSE:")
    print(f"{'=' * 140}")
    header = f"{'Config':<55} " + " ".join(f"{j:<14}" for j in JOINT_NAMES)
    print(header)
    print("-" * 140)
    for r in all_results[:10]:
        pj = r["per_joint_rmse"]
        row = f"{r['name']:<55} " + " ".join(f"{v:<14.3f}" for v in pj)
        print(row)
    print()


if __name__ == "__main__":
    main()
    # simulation_app.close()
    import sys
    sys.exit(0)
