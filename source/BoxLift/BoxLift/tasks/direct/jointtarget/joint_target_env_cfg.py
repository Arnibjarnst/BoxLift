# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors.contact_sensor import ContactSensorCfg


ROBOT_PATH = "./robots/ur5e_sphere.usd"
ENV_REGEX = "/World/envs/env_.*"

TABLE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Table",
    spawn=sim_utils.CuboidCfg(
        size=(1.5, 1.5, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5)),
)

@configclass
class EventCfg:
    ur5_l_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("UR5_left", body_names="Sphere"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.7, 0.9),
            "num_buckets": 250,
        },
    )
    ur5_r_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("UR5_right", body_names="Sphere"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.7, 0.9),
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
class JointTargetEnvCfg(DirectRLEnvCfg):
    # Trajectory file path
    trajectory_path = ""
    # env
    # physics_dt * decimations needs to match dt from planner/IK
    physics_dt = 1.0 / 100.0
    decimation = 2
    episode_length_s = 3.0
    # - spaces definition
    action_space = 12
    observation_space = 38
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))

    # Domain Randomization
    events: EventCfg = EventCfg()

    ur5_l_prim_path = f"{ENV_REGEX}/ur5_l"
    ur5_r_prim_path = f"{ENV_REGEX}/ur5_r"

    # Arm Actuator parameters
    kp = 100.0
    kd = 10.0
    actuator_type = "Implicit"  # "Implicit", "IdealPD" or "DelayedPD"
    velocity_limit = 3.14
    effort_limit= {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
    }
    actuator_min_delay = 1 # For DelayedPDActuatorCfg
    actuator_max_delay = 2 # For DelayedPDActuatorCfg

    # table
    table_cfg = TABLE_CFG

    # scene
    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=replicate_physics)

    # Action scale
    action_scale = 0.25

    # Reward parameteres
    w_track = 1.0
    w_regularization = 0.25

    # Track reward parameters
    w_eef_pos = 1.0
    sigma_eef_pos = 0.05
    tol_eef_pos = 0.0

    # TODO: Use or delete?
    w_eef_quat = 0.0
    sigma_eef_quat = 0.5
    tol_eef_quat = 0.1

    w_joint_pos = 0.0
    sigma_joint_pos = 0.1
    tol_joint_pos = 0.0

    # Regularization reward parameters
    w_joint_acc = 1e-6
    tol_joint_acc = 0.0

    w_joint_torque = 1e-5
    tol_joint_torque = 0.0

    w_action_rate = 1e-1
    tol_action_rate = 0.0

    w_proximity_to_contact = 1.0
    max_proximity = 0.01

    w_illegal_contact = 0.2
    min_contact_force = 0
    max_contact_force = 20

    # Contact Sensors
    non_ee_link_names = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ]

    filter_prim_paths_expr = []
    for robot_prim_path in [ur5_l_prim_path, ur5_r_prim_path]:
        for link_name in non_ee_link_names:
            filter_prim_paths_expr.append(f"{robot_prim_path}/{link_name}/")

    ee_contact_sensors = [
        ContactSensorCfg(
            prim_path=f"{ur5_prim_path}/Sphere",
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=0,
            filter_prim_paths_expr=[TABLE_CFG.prim_path]
        )
        for ur5_prim_path in [ur5_l_prim_path, ur5_r_prim_path]
    ]

    wrist_3_contact_sensors = [
        ContactSensorCfg(
            prim_path=f"{ur5_prim_path}/wrist_3_link",
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=0,
            track_contact_points=True,
            track_air_time=True,
            # track_pose=True,
            filter_prim_paths_expr=[TABLE_CFG.prim_path]
        )
        for ur5_prim_path in [ur5_l_prim_path, ur5_r_prim_path]
    ]

    illegal_contact_sensor_cfgs = {
        "table": ContactSensorCfg(
            prim_path=table_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            track_air_time=True,
            filter_prim_paths_expr=filter_prim_paths_expr
        ),
    }

    # - reset conditions
    max_obj_dist_from_traj = 0.1
    max_obj_angle_from_traj = 0.5


def get_ur5_cfg(
    prim_path,
    init_pose,
    joint_target_cfg: JointTargetEnvCfg,
):
    actuator_kwargs = dict(
        joint_names_expr=[".*"],
        stiffness=joint_target_cfg.kp,
        damping=joint_target_cfg.kd,
        velocity_limit=joint_target_cfg.velocity_limit,
        effort_limit=joint_target_cfg.effort_limit,
    )

    if joint_target_cfg.actuator_type == "DelayedPD":
        actuator_kwargs["min_delay"] = joint_target_cfg.actuator_min_delay
        actuator_kwargs["max_delay"] = joint_target_cfg.actuator_max_delay
        actuator_cfg = DelayedPDActuatorCfg(**actuator_kwargs)
    elif joint_target_cfg.actuator_type == "IdealPD":
        actuator_cfg = IdealPDActuatorCfg(**actuator_kwargs)
    elif joint_target_cfg.actuator_type == "Implicit":
        actuator_cfg = ImplicitActuatorCfg(**actuator_kwargs)
    else:
        raise ValueError(f"Unknown actuator type: {joint_target_cfg.actuator_type}")
    

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01
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
