# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors.contact_sensor import ContactSensorCfg


ROBOT_PATH = "./robots/ur5e.usd"
ENV_REGEX = "/World/envs/env_.*"


CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.235, 0.34, 0.27),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=4.4),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35,
            dynamic_friction=0.35,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
)

TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.CuboidCfg(
        size=(1.5, 1.5, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35,
            dynamic_friction=0.35,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5 + 0.018))
)

@configclass
class EventCfg:
    object_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.3, 0.4),
            "dynamic_friction_range": (0.3, 0.4),
            "restitution_range": (0.2, 0.3),
            "num_buckets": 250,
        },
    )
    table_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.3, 0.4),
            "dynamic_friction_range": (0.3, 0.4),
            "restitution_range": (0.4, 0.5),
            "num_buckets": 250,
        },
    )
    object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )
    reset_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

@configclass
class BoxpushEnvCfg(DirectRLEnvCfg):
    trajectory_path = ""
    # physics_dt * decimation must match dt from the reference trajectory planner/IK
    physics_dt = 1.0 / 100.0
    decimation = 2
    episode_length_s = 3.0
    # action_space is recomputed in __post_init__: 6 residual + 1 phase if enable_phase_slowdown
    action_space = 6
    obs_history_steps = 3
    include_object_obs = True
    include_absolute_obs = True
    future_obs_steps = (1,2,3,4,5)
    include_prev_actions = True
    observation_space = {"policy": 13, "privileged": 85}  # recomputed in __post_init__
    state_space = 0

    enable_phase_slowdown = True
    dphase_min = 0.0
    max_slowdown_multiplier = 3.0

    # Hard cap on per-episode steps; t0 sampling is restricted to [0, T-1-L] so episodes can
    # always complete. -1 disables.
    max_episode_steps: int = -1

    enable_failure_resampling = False
    phase_segment_s = 1.0
    phase_resample_alpha = 0.05
    phase_resample_beta = 1.0
    phase_resample_clamp = (0.1, 0.9)

    @property
    def per_step_feature_dim(self) -> int:
        dim = 12 + (7 if self.include_object_obs else 0)
        if self.include_absolute_obs:
            dim *= 2
        return dim

    def __post_init__(self):
        self.action_space = 7 if self.enable_phase_slowdown else 6
        actor_dim = self.per_step_feature_dim * self.obs_history_steps + 1
        future_dim = 14 if self.include_absolute_obs else 7
        actor_dim += future_dim * len(self.future_obs_steps)
        if self.include_prev_actions:
            actor_dim += 6
        self.observation_space = {"policy": actor_dim, "privileged": 85}

    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))

    events: EventCfg = EventCfg()

    ur5e_prim_path = f"{ENV_REGEX}/ur5e"

    kp = 300.0
    kd = 50
    actuator_type = "Implicit"  # or "IdealPD"
    velocity_limit = 3.14
    effort_limit = {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
    }

    action_scale: float | list = 0.1

    # A: residual on planner target; B: residual on traj position; C: residual on current joints;
    # D: planner PD error from current position + residual
    action_mode = "D"

    cube_cfg = CUBE_CFG
    table_cfg = TABLE_CFG

    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=replicate_physics)

    reset_joint_pos_noise = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    reset_joint_vel_noise = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    reset_obj_pos_xy_noise = 0.02        # m, std on box x,y position (z unchanged)
    reset_obj_lin_vel_xy_noise = 0.05    # m/s, std on box linear x,y velocity (z unchanged)
    reset_obj_ori_noise = 0.1            # rad, axis-angle std for small orientation perturbation
    reset_obj_ang_vel_noise = 0.1        # rad/s, std on box angular velocity (z axis)

    # obs noise applied only to observation path; rewards see clean ground truth
    obs_obj_pos_noise = 0.01             # m
    obs_obj_ori_noise = 0.02             # rad

    # tracker delay/rate model: fixed latency + sub-50Hz update; between fires policy reuses last sample
    obs_obj_delay_steps = 5              # 100ms at 20ms env step
    obs_obj_update_period = 2            # 25Hz

    perturbation_force_std = 10.0       # N
    perturbation_torque_std = 2.0       # Nm
    perturbation_probability = 0.0
    perturbation_duration_steps = 5

    w_task = 0.5
    w_track = 1 - w_task
    w_regularization = 0.1

    # α ramps 0→1 over alpha_warmup_steps, interpolating reward weights and (mode D) action blend
    alpha_warmup_steps = 24 * 750
    w_task_start = 0.2
    w_track_start = 0.8
    action_alpha_floor = 0.1
    # pin α for eval (play.py/record.py set this from checkpoint); -1 = use schedule
    force_alpha: float = -1

    w_obj_pos = 0.2
    sigma_obj_pos = 0.05
    tol_obj_pos = 0.0

    w_obj_quat = 0.7
    sigma_obj_quat = (0.05, 0.15)
    tol_obj_quat = 0.0

    w_obj_lin_vel = 0.05
    sigma_obj_lin_vel = 0.08
    tol_obj_lin_vel = 0.0

    w_obj_ang_vel = 0.05
    sigma_obj_ang_vel = 0.2
    tol_obj_ang_vel = 0.0

    w_eef_pos = 1.0
    sigma_eef_pos = 0.1
    tol_eef_pos = 0.0

    w_eef_quat = 0.0
    sigma_eef_quat = 0.5
    tol_eef_quat = 0.1

    w_joint_pos = 0.0
    sigma_joint_pos = 0.1
    tol_joint_pos = 0.0

    # EE-in-box-frame tracking, active only when reference box is moving (gated by obj_vel_ref)
    w_eef_box_rel_pos = 0.7
    sigma_eef_box_rel_pos = 0.05
    tol_eef_box_rel_pos = 0.0

    w_eef_box_rel_quat = 0.3
    sigma_eef_box_rel_quat = 0.5
    tol_eef_box_rel_quat = 0.0

    eef_box_gate_obj_vel_eps = 1e-3
    eef_box_gate_dilation_steps = 1e7

    # RSI excludes mid-contact resets (EE snaps into box after joint noise); independent of gate dilation
    rsi_contact_dilation_steps = 5

    w_joint_acc = 1e-3
    tol_joint_acc = 0.0

    w_joint_torque = 1e-3
    tol_joint_torque = 0.0

    w_action_rate = 1.0
    tol_action_rate = 0.0

    w_action_norm = 0.0
    tol_action_norm = 0.0

    w_total_slowdown = 0.025

    w_completion = 0.0

    w_joint_limit = 1e3
    joint_limit_eps = 0.05

    w_proximity_to_contact = 0.5
    max_proximity = 0.05

    w_illegal_contact = 50.0
    min_contact_force = 0
    max_contact_force = 20

    w_flange_forearm_dist = 1.0
    max_flange_forearm_distance = 0.028 + 0.0375

    non_ee_link_names = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
    ]

    cube_contact_filter = []
    for link_name in non_ee_link_names:
        cube_contact_filter.append(f"{ur5e_prim_path}/{link_name}/")
    table_contact_filter = cube_contact_filter
    table_contact_filter.append(f"{ur5e_prim_path}/wrist_3_link/")

    illegal_contact_sensor_cfgs = {
        "table": ContactSensorCfg(
            prim_path=table_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=table_contact_filter
        ),
    }

    max_obj_dist_from_traj = 0.1
    max_obj_angle_from_traj = 1.0


def get_ur5e_cfg(
    prim_path,
    init_pose,
    cfg: BoxpushEnvCfg,
):
    actuator_kwargs = dict(
        joint_names_expr=[".*"],
        stiffness=cfg.kp,
        damping=cfg.kd,
        velocity_limit=cfg.velocity_limit,
        effort_limit=cfg.effort_limit,
    )

    if cfg.actuator_type == "IdealPD":
        actuator_cfg = IdealPDActuatorCfg(**actuator_kwargs)
    elif cfg.actuator_type == "Implicit":
        actuator_cfg = ImplicitActuatorCfg(**actuator_kwargs)
    else:
        raise ValueError(f"Unknown actuator type: {cfg.actuator_type}")

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
                fix_root_link=True
            ),
            copy_from_source=False,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(init_pose[:3]),
            rot=tuple(init_pose[3:])
        ),
        actuators={
            "joints": actuator_cfg
        }
    )
