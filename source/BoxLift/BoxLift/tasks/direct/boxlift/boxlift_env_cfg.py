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


ROBOT_PATH = "./robots/ur5_sphere_1.0_2.usd"
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
            restitution=0.2
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
            restitution=0.2
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5))
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
            "mass_distribution_params": (0.5, 2.0),
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
    # Trajectory file path
    trajectory_path = ""
    # env
    # physics_dt * decimations needs to match dt from planner/IK
    physics_dt = 1.0 / 100.0
    decimation = 2
    episode_length_s = 3.0
    # - spaces definition
    action_space = 12
    observation_space = 37
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))

    # Domain Randomization
    events: EventCfg = EventCfg()

    ur5_l_prim_path = f"{ENV_REGEX}/ur5_l"
    ur5_r_prim_path = f"{ENV_REGEX}/ur5_r"

    # Arm Actuator parameters
    kp = 100
    kd = 5
    actuator_type = "Implicit"  # or "IdealPD"
    velocity_limit=50.0
    effort_limit=87.0

    # object (cube)
    cube_cfg = CUBE_CFG

    # table
    table_cfg = TABLE_CFG

    # scene
    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=replicate_physics)

    # Action scale
    action_scale = 0.25

    # Reward parameteres
    w_task = 0.8
    w_track = 1 - w_task
    w_regularization = 0.15

    # Task reward parameteres
    w_obj_pos = 0.6
    sigma_obj_pos = 0.02
    tol_obj_pos = 0.0

    w_obj_quat = 0.4
    sigma_obj_quat = 0.1
    tol_obj_quat = 0.0

    # TODO: Use or delete?
    w_obj_vel = 0.0
    sigma_obj_vel = 0.0
    tol_obj_vel = 0.0

    # Track reward parameters
    w_eef_pos = 0.5
    sigma_eef_pos = 0.04
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
    tol_joint_acc = 0.1

    w_joint_torque = 1e-5
    tol_joint_torque = 0.1

    w_action_rate = 1e-2
    tol_action_rate = 0.1

    w_illegal_contact = 0.1
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

    illegal_contact_sensor_cfgs = {
        "cube": ContactSensorCfg(
            prim_path=cube_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            filter_prim_paths_expr=filter_prim_paths_expr
        ),
        "table": ContactSensorCfg(
            prim_path=table_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            filter_prim_paths_expr=filter_prim_paths_expr
        ),
    }

    # - reset conditions
    max_obj_dist_from_traj = 0.1
    max_obj_angle_from_traj = 0.15


def get_ur5_cfg(
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
