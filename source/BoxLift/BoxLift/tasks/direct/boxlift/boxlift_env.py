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
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
from isaaclab.sensors.contact_sensor import ContactSensor

from BoxLift.tasks.direct.boxlift.boxlift_env_cfg import *

from isaaclab.utils.math import quat_apply, quat_mul, quat_inv, quat_error_magnitude

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
        self.obj_poses          = torch.from_numpy(traj["obj_poses"]).float().to(self.device)
        self.arm_l_pose         = torch.from_numpy(traj["arm_l_pose"]).float().to(self.device)
        self.arm_r_pose         = torch.from_numpy(traj["arm_r_pose"]).float().to(self.device)
        self.joints_l           = torch.from_numpy(traj["joints_l"]).float().to(self.device)
        self.joints_r           = torch.from_numpy(traj["joints_r"]).float().to(self.device)
        self.joints_target_l    = torch.from_numpy(traj["joints_target_l"]).float().to(self.device)
        self.joints_target_r    = torch.from_numpy(traj["joints_target_r"]).float().to(self.device)
        self.EE_poses_l         = torch.from_numpy(traj["EE_poses_l"]).float().to(self.device)
        self.EE_poses_r         = torch.from_numpy(traj["EE_poses_r"]).float().to(self.device)
        self.dt                 = float(traj["dt"])

        # TODO: Support last trajectory point
        self.cfg.episode_length_s = self.dt * (self.obj_poses.shape[0] - 1)

        ur5_l_cfg = get_ur5_cfg(self.cfg.ur5_l_prim_path, self.arm_l_pose, self.cfg)
        ur5_r_cfg = get_ur5_cfg(self.cfg.ur5_r_prim_path, self.arm_r_pose, self.cfg)

        self.ur5_l = Articulation(ur5_l_cfg)
        self.ur5_r = Articulation(ur5_r_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)

        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}

        # Regularization stuff
        self.prev_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 12), device=self.device)

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
        # add table to the scene
        self.scene.rigid_objects["table"] = self.table
        # add sensors to the scene
        for name, sensor in self.illegal_contact_sensors.items():
            self.scene.sensors[f"illegal_contact_sensor_{name}"] = sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        q_l = self.joints_target_l[self.episode_length_buf]
        q_r = self.joints_target_r[self.episode_length_buf]
        dq_l = self.cfg.action_scale * self.actions[:, :6]
        dq_r = self.cfg.action_scale * self.actions[:, 6:]

        self.ur5_l.set_joint_position_target(q_l + dq_l)
        self.ur5_r.set_joint_position_target(q_r + dq_r)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._get_joint_pos(relative=True),
                self._get_joint_vel(),
                self._get_obj_pos(relative=True),
                self._get_obj_quat(relative=True),
                self._get_obj_vel(),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        obj_pos_error = self._get_obj_pos_error()
        obj_quat_error = self._get_obj_quat_error()

        self.reset_terminated = obj_pos_error > self.cfg.max_obj_dist_from_traj
        self.reset_terminated |= obj_quat_error > self.cfg.max_obj_angle_from_traj

        return self.reset_terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None): # type: ignore
        if env_ids is None:
            env_ids : Sequence[int] = self.scene._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)

        # TODO: verify this is the correct range
        # self.episode_length_buf[env_ids] = torch.randint(0, self.max_episode_length - 2, (len(env_ids),), device=self.device, dtype=torch.long)
        self.episode_length_buf[env_ids] = torch.zeros((len(env_ids),), device=self.device, dtype=torch.long)

        initial_joint_pos_l = self.joints_l[self.episode_length_buf[env_ids]]
        initial_joint_vel_l = torch.zeros_like(initial_joint_pos_l)
        self.ur5_l.write_joint_state_to_sim(initial_joint_pos_l, initial_joint_vel_l, env_ids=env_ids)

        initial_joint_pos_r = self.joints_r[self.episode_length_buf[env_ids]]
        initial_joint_vel_r = torch.zeros_like(initial_joint_pos_r)
        self.ur5_r.write_joint_state_to_sim(initial_joint_pos_r, initial_joint_vel_r, env_ids=env_ids)

        # Reset Object
        initial_object_pose = self.obj_poses[self.episode_length_buf[env_ids]]
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = torch.zeros((len(env_ids), 6), device=self.device)

        self.object.write_root_pose_to_sim(initial_object_pose, env_ids)
        self.object.write_root_velocity_to_sim(initial_object_vel, env_ids)

        # Reset prev variables
        self.prev_actions[env_ids, :6] = initial_joint_pos_l
        self.prev_actions[env_ids, 6:] = initial_joint_pos_r
        self.prev_joint_vel[env_ids, :6] = initial_joint_vel_l
        self.prev_joint_vel[env_ids, 6:] = initial_joint_vel_r

    def _reward_track(self, error, sigma, tolerance=0.0):
        error *= error > tolerance
        reward = torch.exp(-error / (sigma ** 2))
        return reward

    def _get_rewards(self) -> torch.Tensor:
        # Task Reward
        obj_pos_error = self._get_obj_pos_error()
        rew_obj_pos = self.cfg.w_obj_pos * self._reward_track(obj_pos_error ** 2, self.cfg.sigma_obj_pos, self.cfg.tol_obj_pos)
        
        obj_quat_error = self._get_obj_quat_error()
        rew_obj_quat = self.cfg.w_obj_quat * self._reward_track(obj_quat_error ** 2, self.cfg.sigma_obj_quat, self.cfg.tol_obj_quat)

        rew_task = self.cfg.w_task * (rew_obj_pos + rew_obj_quat)

        # Tracking Reward
        EE_pos_error_l, EE_pos_error_r = self._get_EE_pos_error()
        rew_EE_pos_l = self._reward_track(EE_pos_error_l ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)
        rew_EE_pos_r = self._reward_track(EE_pos_error_r ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)
        rew_EE_pos = self.cfg.w_eef_pos * (rew_EE_pos_l + rew_EE_pos_r)

        eef_quat_error_l, eef_quat_error_r = self._get_EE_quat_error()
        rew_EE_quat_l = self._reward_track(eef_quat_error_l ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)
        rew_EE_quat_r = self._reward_track(eef_quat_error_r ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)
        rew_EE_quat = self.cfg.w_eef_quat * (rew_EE_quat_l + rew_EE_quat_r)

        joint_pos_lr2 = self._get_joint_pos(relative=True) ** 2
        joint_pos_l2, joint_pos_r2 = joint_pos_lr2[:,:6], joint_pos_lr2[:,6:]
        joint_pos_error_l = joint_pos_l2.sum(dim=-1)
        joint_pos_error_r = joint_pos_r2.sum(dim=-1)
        rew_joint_pos_l = self._reward_track(joint_pos_error_l, self.cfg.sigma_joint_pos, self.cfg.tol_joint_pos)
        rew_joint_pos_r = self._reward_track(joint_pos_error_r, self.cfg.sigma_joint_pos, self.cfg.tol_joint_pos)
        rew_joint_pos = self.cfg.w_joint_pos * (rew_joint_pos_l + rew_joint_pos_r)

        rew_track = self.cfg.w_track * (rew_EE_pos + rew_EE_quat + rew_joint_pos)

        # Regularization Reward
        joint_acc = (self._get_joint_vel() - self.prev_joint_vel) / self.dt
        joint_acc *= joint_acc > self.cfg.tol_joint_acc
        joint_acc_penalty = joint_acc.square().sum(dim=-1)
        rew_joint_acc = self.cfg.w_joint_acc * joint_acc_penalty

        torque = torch.cat((self.ur5_l.data.applied_torque, self.ur5_r.data.applied_torque), 1)
        torque *= torque > self.cfg.tol_joint_torque
        torque_penalty = torque.square().sum(dim=-1)
        rew_torque = self.cfg.w_joint_torque * torque_penalty

        action_rate_error = (self.actions - self.prev_actions)
        action_rate_error *= action_rate_error > self.cfg.tol_action_rate
        action_rate_penalty = action_rate_error.square().sum(dim=-1)
        rew_action_rate = self.cfg.w_action_rate * action_rate_penalty


        total_illegal_force = torch.zeros((self.num_envs,), device=self.device)
        for sensor in self.illegal_contact_sensors.values():
            f_abs = sensor.data.force_matrix_w.norm(dim=-1)
            f_abs_clamped = f_abs.clamp(max=self.cfg.max_contact_force)
            total_illegal_force += f_abs_clamped.flatten(1).sum(dim=-1)

        rew_illegal_contact = self.cfg.w_illegal_contact * total_illegal_force

        rew_reqularization = self.cfg.w_regularization * (rew_joint_acc + rew_torque + rew_action_rate + rew_illegal_contact)

        self.extras["log"] = {
            "Rewards_task/obj_pos": rew_obj_pos.mean(),
            "Rewards_task/obj_quat": rew_obj_quat.mean(),
            "Rewards_track/eef_pos": rew_EE_pos.mean(),
            "Rewards_track/eef_quat": rew_EE_quat.mean(),
            "Rewards_track/joint_pos": rew_joint_pos.mean(),
            "Rewards_task/total": rew_task.mean(),
            "Rewards_track/total": rew_track.mean(),
            "Error/obj_pos_error": obj_pos_error.mean(),
            "Error/obj_quat_error": obj_quat_error.mean(),
            "Error/EE_pos_error": (EE_pos_error_l + EE_pos_error_r).mean() / 2.0,
            "Rewards_regularization/total": rew_reqularization.mean(),
            "Rewards_regularization/joint_acceleration": rew_joint_acc.mean(), 
            "Rewards_regularization/torque": rew_torque.mean(),
            "Rewards_regularization/action_rate": rew_action_rate.mean(),
            "Rewards_regularization/illegal_contact": rew_illegal_contact.mean(),
        }
        
        total_reward = rew_task + rew_track - rew_reqularization

        # Can update prev_action/joint_vel now
        self.prev_actions[:] = self.actions[:]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

    
    def _get_joint_pos(self, relative=True):
        joint_pos_l = self.ur5_l.data.joint_pos
        joint_pos_r = self.ur5_r.data.joint_pos

        if relative:
            joint_pos_l -= self.joints_l[self.episode_length_buf]
            joint_pos_r -= self.joints_r[self.episode_length_buf]

        return torch.cat((joint_pos_l, joint_pos_r), 1)
    
    
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
            EE_pos_l -= self.EE_poses_l[self.episode_length_buf, :3]
            EE_pos_r -= self.EE_poses_r[self.episode_length_buf, :3]

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
            desired_quat_l = self.EE_poses_l[self.episode_length_buf, 3:]
            EE_quat_l = quat_mul(desired_quat_l, quat_inv(EE_quat_l))
            desired_quat_r = self.EE_poses_r[self.episode_length_buf, 3:]
            EE_quat_r = quat_mul(desired_quat_r, quat_inv(EE_quat_r))

        return torch.cat((EE_quat_l, EE_quat_r), 1)
    
    def _get_EE_quat_error(self):
        EE_quat_l = self.ur5_l.data.body_quat_w[:, self.EE_link_idx]
        EE_quat_r = self.ur5_r.data.body_quat_w[:, self.EE_link_idx]

        desired_quat_l = self.EE_poses_l[self.episode_length_buf, 3:]
        desired_quat_r = self.EE_poses_r[self.episode_length_buf, 3:]

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
            desired_obj_pos = self.obj_poses[self.episode_length_buf, :3]
            obj_pos -= desired_obj_pos

        return obj_pos
    
    def _get_obj_pos_error(self):
        return torch.norm(self._get_obj_pos(relative=True), dim=-1)
    
    def _get_obj_quat(self, relative=True):
        obj_quat = self.object.data.root_quat_w

        if relative:
            desired_obj_quat = self.obj_poses[self.episode_length_buf, 3:]
            obj_quat = quat_mul(desired_obj_quat, quat_inv(obj_quat))

        return obj_quat
    
    def _get_obj_quat_error(self):
        obj_quat = self.object.data.root_quat_w
        desired_obj_quat = self.obj_poses[self.episode_length_buf, 3:]
        return torch.abs(quat_error_magnitude(obj_quat, desired_obj_quat))

    def _get_obj_vel(self):
        return self.object.data.root_vel_w

