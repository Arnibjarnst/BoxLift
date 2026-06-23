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
        size=(0.4, 0.6, 0.06),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
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
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5))
)

@configclass
class EventCfg:
    ur5e_l_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_left", body_names="wrist_3_link"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.1, 0.3),
            "num_buckets": 250,
        },
    )
    ur5e_r_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_right", body_names="wrist_3_link"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.1, 0.3),
            "num_buckets": 250,
        },
    )
    ur5e_l_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_left"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    ur5e_r_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_right"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    table_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform"
        },
    )
    # object_scale = EventTermCfg(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "scale_range": (0.975, 1.025)
    #     },
    # )
    # object_com = EventTermCfg(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "com_range": {
    #             "x": (-0.1, 0.1), # not relative to scale but absolute addition uniformly sampled
    #             "y": (-0.1, 0.1),
    #         }
    #     },
    # )
    reset_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    
@configclass
class BoxliftEnvCfg(DirectRLEnvCfg):
    trajectory_path = ""
    # physics_dt * decimation must match trajectory dt (0.02 s)
    physics_dt = 1.0 / 100.0
    decimation = 2
    episode_length_s = 3.0
    action_space = 12
    observation_space = {"policy": 275, "privileged": 136}  # placeholder; __post_init__ overwrites
    state_space = 0

    use_reference_obs: bool = True
    obs_history_steps: int = 3
    future_obs_steps: tuple = (1, 2, 3, 4, 5)

    contact_threshold: float = 0.5
    contact_obs_delay_steps: int = 0
    contact_obs_flip_prob: float = 0.0

    # Simulated tracker: latches cube pose every obs_obj_update_period steps, with obs_obj_delay_steps lag.
    obs_obj_delay_steps: int = 5
    obs_obj_update_period: int = 2
    obs_obj_pos_noise: float = 0.001     # m, std per fire
    obs_obj_ori_noise: float = 0.001     # rad, std per fire
    obs_obj_pos_bias_std: float = 0.005  # m, per-episode constant offset std
    obs_obj_ori_bias_std: float = 0.005  # rad, per-episode constant offset std

    obs_joint_pos_noise_std: float = 0.005   # rad
    obs_joint_vel_noise_std: float = 0.05    # rad/s

    def __post_init__(self) -> None:
        base_pose = 3 + 4                       # obj pos + quat
        if self.use_reference_obs:
            per_step = 12 + 12 + base_pose      # rel_q + rel_qd + rel_obj
            per_step += 12 + 12 + base_pose     # abs_q + abs_qd + abs_obj
            per_step += 2                       # contact bools
            phase_dim = 1
        else:
            per_step = 12 + 12 + base_pose      # abs_q + abs_qd + abs_obj only
            per_step += 2                       # contact bools
            phase_dim = 0                       # no trajectory clock
        self.per_step_feature_dim = per_step

        future_dim_per_offset = 2 * base_pose   # (rel pos+quat) + (abs pos+quat) per offset
        actor_dim = (
            per_step * self.obs_history_steps
            + phase_dim
            + future_dim_per_offset * len(self.future_obs_steps)
            + 12                                         # prev_actions
        )

        # Privileged-extras layout (concatenated by _get_privileged_obs in this order):
        #   13  clean obj state           (pos 3 + quat 4 + lin_vel 3 + ang_vel 3)
        #   28  DR samples                (obj_mass 1 + obj_friction 3 + stiff_L 6 +
        #                                  stiff_R 6 + damp_L 6 + damp_R 6)
        #   75  reference state @ phase   (ref_obj_pos 3 + quat 4 + lin_vel 3 + ang_vel 3
        #                                  + ref_joints_L 6 + R 6 + ref_joint_vels_L 6 + R 6
        #                                  + ref_joints_target_L 6 + R 6
        #                                  + planner_pd_err_L 6 + R 6
        #                                  + ref_EE_pose_L 7 + R 7)
        #   12  force/contact             (EE force mag L 1 + R 1 + EE force dir L 3 + R 3
        #                                  + illegal_contact cube 1 + table 1
        #                                  + flange-forearm dist L 1 + R 1)
        #    4  eef-box rel scalars       (pos_err_L + quat_err_L + pos_err_R + quat_err_R)
        #    4  voc / curriculum context  (current_seg kp_pos + kp_rot + alpha + seg_idx)
        priv_dim = 13 + 28 + 75 + 12 + 4 + 4

        self.observation_space = {"policy": actor_dim, "privileged": priv_dim}

    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))
    events: EventCfg = EventCfg()

    ur5e_l_prim_path = f"{ENV_REGEX}/ur5e_l"
    ur5e_r_prim_path = f"{ENV_REGEX}/ur5e_r"

    kp = 150.0
    kd = 22.5
    actuator_type = "Implicit"  # or "IdealPD"
    disable_robot_gravity: bool = False
    velocity_limit = 3.14
    effort_limit = {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
    }

    cube_cfg = CUBE_CFG
    table_cfg = TABLE_CFG
    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=replicate_physics)

    # Action mode: A=residual on planner target, B=on trajectory pos, C=on current pos,
    # BC=(1-α)·traj+α·current, D=current+(1-α)·PD_error+(α+ε(1-α))·scale·action
    action_mode = "A"

    force_alpha: float = -1           # pin α to fixed value (-1 = use schedule)
    action_alpha_floor: float = 1.0   # mode D: residual authority floor at α=0
    alpha_warmup_steps: int = 0       # linear ramp when curriculum disabled

    alpha_curriculum_enabled: bool = True
    alpha_decay_phi: float = 0.95
    alpha_min_support: float = 0.01
    alpha_threshold_task: float = 0.7
    alpha_reward_window_size: int = 100
    alpha_decay_check_interval: int = 240
    alpha_decay_warmup_steps: int = 0

    action_scale: float | list = 0.1

    reset_joint_pos_noise = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
    reset_joint_vel_noise = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    reset_obj_pos_xy_noise = 0.01
    reset_obj_lin_vel_xy_noise = 0.0
    reset_obj_ori_noise = 0.03
    reset_obj_ang_vel_noise = 0.0

    robot_pos_randomization_xyz_std: float = 0.01   # m
    robot_ori_randomization_yaw_std: float = 0.02   # rad

    # Boost above ~1/T default so phase=0 (deployment start) gets adequate training.
    reset_to_zero_prob: float = 0.1

    w_task = 0.7
    w_track = 1 - w_task
    w_regularization = 1.0

    w_obj_pos = 0.5
    sigma_obj_pos = 0.05
    tol_obj_pos = 0.0

    w_obj_quat = 0.5
    sigma_obj_quat = 0.15
    tol_obj_quat = 0.0

    w_obj_vel = 0.0
    sigma_obj_vel = 0.0
    tol_obj_vel = 0.0

    w_eef_pos = 1.0
    sigma_eef_pos = 0.075
    tol_eef_pos = 0.0

    w_eef_quat = 0.0
    sigma_eef_quat = 0.2
    tol_eef_quat = 0.1

    w_joint_pos = 0.0
    sigma_joint_pos = 0.1
    tol_joint_pos = 0.0

    w_eef_box_rel_pos = 0.7
    sigma_eef_box_rel_pos = 0.075
    tol_eef_box_rel_pos = 0.0

    w_eef_box_rel_quat = 0.3
    sigma_eef_box_rel_quat = 0.5
    tol_eef_box_rel_quat = 0.0

    # Large dilation_steps effectively disables the contact gate.
    eef_box_gate_obj_vel_eps = 1e-3
    eef_box_gate_dilation_steps = 1e7

    w_joint_acc = 1e-5
    tol_joint_acc = 0.0

    w_joint_torque = 5e-5
    tol_joint_torque = 0.0

    w_action_rate = 1e-1
    tol_action_rate = 0.0

    w_action_norm = 1e-3
    tol_action_norm = 0.0

    w_joint_limit = 5e2
    joint_limit_eps = 0.05

    w_proximity_to_contact = 0.5
    max_proximity = 0.05

    w_illegal_contact = 0.1
    min_contact_force = 1
    max_contact_force = 20

    w_flange_forearm_dist = 0.5
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
    for robot_prim_path in [ur5e_l_prim_path, ur5e_r_prim_path]:
        for link_name in non_ee_link_names:
            cube_contact_filter.append(f"{robot_prim_path}/{link_name}/")
    table_contact_filter = list(cube_contact_filter)
    for robot_prim_path in [ur5e_l_prim_path, ur5e_r_prim_path]:
        table_contact_filter.append(f"{robot_prim_path}/wrist_3_link/")

    ee_contact_sensors = [
        ContactSensorCfg(
            prim_path=f"{ur5e_prim_path}/wrist_3_link",
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=0,
            filter_prim_paths_expr=[CUBE_CFG.prim_path, TABLE_CFG.prim_path]
        )
        for ur5e_prim_path in [ur5e_l_prim_path, ur5e_r_prim_path]
    ]

    illegal_contact_sensor_cfgs = {
        "cube": ContactSensorCfg(
            prim_path=cube_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=cube_contact_filter,
        ),
        "table": ContactSensorCfg(
            prim_path=table_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=table_contact_filter,
        ),
    }

    max_obj_dist_from_traj = 0.2
    max_obj_angle_from_traj = 10000

    # VOC: virtual PD wrench (DexMachina) drives cube along reference while policy learns.
    # Gain decays per-segment as reward thresholds are met. See _apply_voc, _voc_decay_check.
    voc_enabled: bool = True
    voc_kp_pos: float = 1000.0    # N/m
    voc_kp_rot: float = 100.0     # Nm/rad
    voc_kp_min: float = 0.1       # snap to zero below this
    # kv = scale · sqrt(kp · inertia) — critically damped
    voc_kv_pos_scale: float = 2.0
    voc_kv_rot_scale: float = 2.0
    voc_decay_phi_p: float = 0.98
    voc_decay_phi_v: float = 0.98
    voc_decay_check_interval: int = 100
    voc_reward_window_size: int = 100
    voc_threshold_task: float = 0.7
    voc_threshold_track: float = 0.0
    voc_decay_warmup_steps: int = 0

    emit_per_env_extras: bool = False

    # Per-segment VOC: "none" | "uniform" (voc_n_uniform_segments bins) | "contact" (gate transitions)
    voc_segmentation: str = "contact"
    voc_n_uniform_segments: int = 5
    # Independent of eef_box_gate_dilation_steps (reward path); this smooths segmentation boundaries.
    voc_segment_dilation_steps: int = 10

    # Bias RSI toward segments still under VOC assist; weight = kp[s]^segment_focus_beta.
    reset_segment_focus_prob: float = 0.7
    segment_focus_beta: float = 1.0


def get_ur5e_cfg(
    prim_path,
    init_pose,
    box_lift_cfg: BoxliftEnvCfg,
):
    actuator_kwargs = dict(
        joint_names_expr=[".*"],
        stiffness=box_lift_cfg.kp,
        damping=box_lift_cfg.kd,
        velocity_limit=box_lift_cfg.velocity_limit,
        effort_limit=box_lift_cfg.effort_limit,
    )

    if box_lift_cfg.actuator_type == "IdealPD":
        actuator_cfg = IdealPDActuatorCfg(**actuator_kwargs)
    elif box_lift_cfg.actuator_type == "Implicit":
        actuator_cfg = ImplicitActuatorCfg(**actuator_kwargs)
    else:
        raise ValueError(f"Unknown actuator type: {box_lift_cfg.actuator_type}")

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=box_lift_cfg.disable_robot_gravity,
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
