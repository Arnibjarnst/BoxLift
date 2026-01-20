# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def get_ur5_cfg(prim_path):
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"./robots/ur5_sphere_0.5.usd",
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
        ),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness={
                    ".*": 4000.0,
                },
                damping={
                    ".*": 150.0,
                },
            ),
        }
    )

UR5_CFG_LEFT = get_ur5_cfg("/World/envs/env_.*/UR5_left")
UR5_CFG_RIGHT = get_ur5_cfg("/World/envs/env_.*/UR5_right")

TABLE_CFG = sim_utils.CuboidCfg(
    size=(0.4, 0.6, 1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), metallic=0.2),
)

CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.4, 0.6, 0.06),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    ),
)
    
@configclass
class BoxliftEnvCfg(DirectRLEnvCfg):
    # Trajectory file path
    trajectory_path = ""
    # env
    decimation = 2
    episode_length_s = 3.0
    # - spaces definition
    action_space = 12
    observation_space = 64
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    ur5_l_cfg: ArticulationCfg = UR5_CFG_LEFT
    ur5_r_cfg: ArticulationCfg = UR5_CFG_RIGHT

    # table
    table_cfg = TABLE_CFG

    # object (cube)
    cube_cfg = CUBE_CFG

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # - reward scales
    rew_scale_obj_pos = 2.0
    rew_sigma_obj_pos = 1.0
    rew_tol_obj_pos = 0.0

    rew_scale_obj_quat = 1.0
    rew_sigma_obj_quat = 1.0
    rew_tol_obj_quat = 0.0

    rew_scale_obj_vel = 0.0
    rew_sigma_obj_vel = 0.0
    rew_tol_obj_vel = 0.0

    rew_scale_eef_pos = 1.5
    rew_sigma_eef_pos = 1.0
    rew_tol_eef_pos = 0.0

    rew_scale_eef_quat = 0.5
    rew_sigma_eef_quat = 1.0
    rew_tol_eef_quat = 0.0

    rew_scale_joint_pos = 1.0
    rew_sigma_joint_pos = 1.0
    rew_tol_joint_pos = 0.0

    # - reset states/conditions
    max_obj_dist_from_traj = 0.05
    max_obj_angle_from_traj = 0.1
