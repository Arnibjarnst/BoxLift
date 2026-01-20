# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from BoxLift.tasks.direct.boxlift.boxlift_env_cfg import *

from isaaclab.utils.math import quat_apply, quat_mul, quat_inv, quat_error_magnitude


def convert_pose_to_isaac(poses: torch.Tensor):
    """
    Converts [... , 7] from [qw, qx, qy, qz, x, y, z] to [x, y, z, qw, qx, qy, qz]
    Works for shapes (7,), (N, 7), or (B, N, 7).
    """
    return torch.cat([poses[..., 4:], poses[..., :4]], dim=-1)

class BoxliftEnv(DirectRLEnv):
    cfg: BoxliftEnvCfg

    def __init__(self, cfg: BoxliftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
       
        self.EE_link_idx = self.ur5_r.body_names.index("wrist_3_link")
        self.EE_local_offset = torch.tensor([0.0, 0.0, 0.02], device=self.device).repeat(self.num_envs, 1)


    def _setup_scene(self):
        # Load the trajectory file
        traj = np.load(self.cfg.trajectory_path)

        # Store initial positions and joint states
        self.obj_poses          = convert_pose_to_isaac(torch.from_numpy(traj["obj_poses"]).float().to(self.device))
        self.arm_l_pose         = convert_pose_to_isaac(torch.from_numpy(traj["arm_l_pose"]).float().to(self.device))
        self.arm_r_pose         = convert_pose_to_isaac(torch.from_numpy(traj["arm_r_pose"]).float().to(self.device))
        self.joints_l           = torch.from_numpy(traj["joints_l"]).float().to(self.device)
        self.joints_r           = torch.from_numpy(traj["joints_r"]).float().to(self.device)
        self.joints_target_l    = torch.from_numpy(traj["joints_target_l"]).float().to(self.device)
        self.joints_target_r    = torch.from_numpy(traj["joints_target_r"]).float().to(self.device)
        self.EE_poses_l         = convert_pose_to_isaac(torch.from_numpy(traj["EE_poses_l"]).float().to(self.device))
        self.EE_poses_r         = convert_pose_to_isaac(torch.from_numpy(traj["EE_poses_r"]).float().to(self.device))
        self.dt                 = float(traj["dt"])

        self.ref_frame_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.ref_frame_idx_int = self.ref_frame_idx.int()
        self.total_frames = self.obj_poses.shape[0]

        self.cfg.ur5_l_cfg.init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.arm_l_pose[:3]),
            rot=tuple(self.arm_l_pose[3:])
        )

        self.cfg.ur5_r_cfg.init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.arm_r_pose[:3]),
            rot=tuple(self.arm_r_pose[3:])
        )

        self.ur5_l = Articulation(self.cfg.ur5_l_cfg)
        self.ur5_r = Articulation(self.cfg.ur5_r_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.0))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)

        # spawn table
        self.cfg.table_cfg.func("/World/envs/env_.*/Table", self.cfg.table_cfg, translation=(0.0, 0.0, -0.5), orientation=(1.0, 0.0, 0.0, 0.0))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["UR5_left"] = self.ur5_l
        self.scene.articulations["UR5_right"] = self.ur5_r
        # add object to the scene
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.ref_frame_idx += self.cfg.sim.dt / self.dt
        self.ref_frame_idx_int = self.ref_frame_idx.int()

    def _apply_action(self) -> None:
        q_l = self.joints_target_l[self.ref_frame_idx_int]
        q_r = self.joints_target_r[self.ref_frame_idx_int]
        dq_l = self.actions[:, :6]
        dq_r = self.actions[:, 6:]

        self.ur5_l.set_joint_position_target(q_l + dq_l)
        self.ur5_r.set_joint_position_target(q_r + dq_r)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._get_joint_pos(relative=True),
                self._get_joint_vel(),
                self._get_EE_pos(relative=True),
                self._get_EE_quat(relative=True),
                self._get_EE_vel(),
                self._get_obj_pos(relative=True),
                self._get_obj_quat(relative=True),
                self._get_obj_vel(),
                self.ref_frame_idx[:, None] / (self.total_frames - 1) # phase variable
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.ref_frame_idx >= self.total_frames - 1

        obj_pos_error = self._get_obj_pos_error()
        obj_quat_error = self._get_obj_quat_error()

        self.reset_terminated = obj_pos_error > self.cfg.max_obj_dist_from_traj
        self.reset_terminated |= obj_quat_error > self.cfg.max_obj_angle_from_traj

        return self.reset_terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            return
        super()._reset_idx(env_ids)

        # TODO: verify this is the correct range
        # self.ref_frame_idx[env_ids] = torch.randint(0, self.total_frames - 1, (len(env_ids),), device=self.device).float()
        self.ref_frame_idx[env_ids] = torch.zeros((len(env_ids),), device=self.device).float()
        self.ref_frame_idx_int[env_ids] = self.ref_frame_idx[env_ids].int()

        initial_joint_pos_l = self.joints_l[self.ref_frame_idx_int[env_ids]]
        initial_joint_vel_l = torch.zeros_like(initial_joint_pos_l)
        self.ur5_l.write_joint_state_to_sim(initial_joint_pos_l, initial_joint_vel_l, env_ids=env_ids)

        initial_joint_pos_r = self.joints_r[self.ref_frame_idx_int[env_ids]]
        initial_joint_vel_r = torch.zeros_like(initial_joint_pos_r)
        self.ur5_r.write_joint_state_to_sim(initial_joint_pos_r, initial_joint_vel_r, env_ids=env_ids)

        # Reset Object
        initial_object_pose = self.obj_poses[self.ref_frame_idx_int[env_ids]]
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = torch.zeros((len(env_ids), 6), device=self.device)

        self.object.write_root_pose_to_sim(initial_object_pose, env_ids)
        self.object.write_root_velocity_to_sim(initial_object_vel, env_ids)

    def _reward_track(self, error, scale, tolerance=0.0):
        error *= error > tolerance
        reward = torch.exp(-scale * error * error)
        return reward

    def _get_rewards(self) -> torch.Tensor:
        obj_pos_error = self._get_obj_pos_error()
        rew_obj_pos = self.cfg.rew_scale_obj_pos * self._reward_track(obj_pos_error, self.cfg.rew_sigma_obj_pos, self.cfg.rew_tol_obj_pos)
        
        obj_quat_error = self._get_obj_quat_error()
        rew_obj_quat = self.cfg.rew_scale_obj_quat * self._reward_track(obj_quat_error, self.cfg.rew_sigma_obj_quat, self.cfg.rew_tol_obj_quat)

        EE_pos_error_l, EE_pos_error_r = self._get_EE_pos_error()
        rew_EE_pos_l = self._reward_track(EE_pos_error_l, self.cfg.rew_sigma_eef_pos, self.cfg.rew_tol_eef_pos)
        rew_EE_pos_r = self._reward_track(EE_pos_error_r, self.cfg.rew_sigma_eef_pos, self.cfg.rew_tol_eef_pos)
        rew_EE_pos = self.cfg.rew_scale_eef_pos * (rew_EE_pos_l + rew_EE_pos_r)

        eef_quat_error_l, eef_quat_error_r = self._get_EE_quat_error()
        rew_EE_quat_l = self._reward_track(eef_quat_error_l, self.cfg.rew_sigma_eef_quat, self.cfg.rew_tol_eef_quat)
        rew_EE_quat_r = self._reward_track(eef_quat_error_r, self.cfg.rew_sigma_eef_quat, self.cfg.rew_tol_eef_quat)
        rew_EE_quat = self.cfg.rew_scale_eef_quat * (rew_EE_quat_l + rew_EE_quat_r)

        joint_pos_error_l, joint_pos_error_r = self._get_joint_pos_error()
        rew_joint_pos_l = self._reward_track(joint_pos_error_l, self.cfg.rew_sigma_joint_pos, self.cfg.rew_tol_joint_pos)
        rew_joint_pos_r = self._reward_track(joint_pos_error_r, self.cfg.rew_sigma_joint_pos, self.cfg.rew_tol_joint_pos)
        rew_joint_pos = self.cfg.rew_scale_joint_pos * (rew_joint_pos_l + rew_joint_pos_r)

        total_reward = rew_obj_pos + rew_obj_quat + rew_EE_pos + rew_EE_quat + rew_joint_pos

        return total_reward

    
    def _get_joint_pos(self, relative=True):
        joint_pos_l = self.ur5_l.data.joint_pos
        joint_pos_r = self.ur5_r.data.joint_pos

        if relative:
            joint_pos_l -= self.joints_l[self.ref_frame_idx_int]
            joint_pos_r -= self.joints_r[self.ref_frame_idx_int]

        return torch.cat((joint_pos_l, joint_pos_r), 1)
    
    def _get_joint_pos_error(self):
        joint_pos_lr = self._get_joint_pos(relative=True)
        joint_pos_l = joint_pos_lr[:, :3]
        joint_pos_r = joint_pos_lr[:, 3:]

        joint_pos_error_l = torch.sum(joint_pos_l, dim=-1)
        joint_pos_error_r = torch.sum(joint_pos_r, dim=-1)
        
        return joint_pos_error_l, joint_pos_error_r
    
    def _get_joint_vel(self):
        ur5_l_joint_vel = self.ur5_l.data.joint_vel
        ur5_r_joint_vel = self.ur5_r.data.joint_vel

        return torch.cat((ur5_l_joint_vel, ur5_r_joint_vel), 1)
    
    
    def _get_EE_pos(self, relative=True) -> torch.Tensor:
        EE_pos_l = self.ur5_l.data.body_pos_w[:, self.EE_link_idx] - self.scene.env_origins
        EE_pos_r = self.ur5_r.data.body_pos_w[:, self.EE_link_idx] - self.scene.env_origins

        # Find the position of the sphere EE instead of frame origin of link
        EE_quat_lr = self._get_EE_quat(relative=False)
        EE_pos_l += quat_apply(EE_quat_lr[:,:4], self.EE_local_offset)
        EE_pos_r += quat_apply(EE_quat_lr[:,4:], self.EE_local_offset)

        if relative:
            EE_pos_l -= self.EE_poses_l[self.ref_frame_idx_int, :3]
            EE_pos_r -= self.EE_poses_r[self.ref_frame_idx_int, :3]

        return torch.cat((EE_pos_l, EE_pos_r), 1)
    
    def _get_EE_pos_error(self):
        EE_pos_lr = self._get_EE_pos(relative=True)
        EE_pos_l = EE_pos_lr[:, :3]
        EE_pos_r = EE_pos_lr[:, 3:]

        EE_pos_error_l = torch.norm(EE_pos_l, dim=-1)
        EE_pos_error_r = torch.norm(EE_pos_r, dim=-1)
        
        return EE_pos_error_l, EE_pos_error_r
    
    def _get_EE_quat(self, relative=True) -> torch.Tensor:
        EE_quat_l = self.ur5_l.data.body_quat_w[:, self.EE_link_idx]
        EE_quat_r = self.ur5_r.data.body_quat_w[:, self.EE_link_idx]

        if relative:
            desired_quat_l = self.EE_poses_l[self.ref_frame_idx_int, 3:]
            EE_quat_l = quat_mul(desired_quat_l, quat_inv(EE_quat_l))
            desired_quat_r = self.EE_poses_r[self.ref_frame_idx_int, 3:]
            EE_quat_r = quat_mul(desired_quat_r, quat_inv(EE_quat_r))

        return torch.cat((EE_quat_l, EE_quat_r), 1)
    
    def _get_EE_quat_error(self):
        EE_quat_l = self.ur5_l.data.body_quat_w[:, self.EE_link_idx]
        EE_quat_r = self.ur5_r.data.body_quat_w[:, self.EE_link_idx]

        desired_quat_l = self.EE_poses_l[self.ref_frame_idx_int, 3:]
        desired_quat_r = self.EE_poses_r[self.ref_frame_idx_int, 3:]

        error_l = torch.abs(quat_error_magnitude(EE_quat_l, desired_quat_l))
        error_r = torch.abs(quat_error_magnitude(EE_quat_r, desired_quat_r))

        return error_l, error_r
    
    def _get_EE_vel(self) -> torch.Tensor:
        EE_vel_l = self.ur5_l.data.body_vel_w[:, self.EE_link_idx]
        EE_vel_r = self.ur5_r.data.body_vel_w[:, self.EE_link_idx]

        return torch.cat((EE_vel_l, EE_vel_r), 1)
    
    def _get_obj_pos(self, relative=True):
        obj_pos = self.object.data.root_pos_w - self.scene.env_origins

        if relative:
            desired_obj_pos = self.obj_poses[self.ref_frame_idx_int, :3]
            obj_pos -= desired_obj_pos

        return obj_pos
    
    def _get_obj_pos_error(self):
        return torch.norm(self._get_obj_pos(relative=True), dim=-1)
    
    def _get_obj_quat(self, relative=True):
        obj_quat = self.object.data.root_quat_w

        if relative:
            desired_obj_quat = self.obj_poses[self.ref_frame_idx_int, 3:]
            obj_quat = quat_mul(desired_obj_quat, quat_inv(obj_quat))

        return obj_quat
    
    def _get_obj_quat_error(self):
        obj_quat = self.object.data.root_quat_w
        desired_obj_quat = self.obj_poses[self.ref_frame_idx_int, 3:]
        return torch.abs(quat_error_magnitude(obj_quat, desired_obj_quat))

    def _get_obj_vel(self):
        return self.object.data.root_vel_w

