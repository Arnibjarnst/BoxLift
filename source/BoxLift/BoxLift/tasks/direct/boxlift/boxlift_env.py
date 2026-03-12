# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import omni.usd
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from BoxLift.tasks.direct.boxlift.boxlift_env_cfg import *

from isaaclab.utils.math import quat_apply, quat_mul, quat_inv, quat_error_magnitude

class BoxliftEnv(DirectRLEnv):
    cfg: BoxliftEnvCfg

    def __init__(self, cfg: BoxliftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
       
        self.EE_link_idx = self.ur5e_r.body_names.index("Sphere")
        self.flange_idx = self.ur5e_r.body_names.index("wrist_3_link")
        self.forearm_link_idx = self.ur5e_r.body_names.index("forearm_link")

    def _setup_scene(self):
        # Load the trajectory file
        traj = np.load(self.cfg.trajectory_path)

        # Store initial positions and joint states
        self.obj_poses          = torch.from_numpy(traj["obj_poses"]).float().to(self.device)
        self.obj_vel            = torch.from_numpy(traj["obj_vel"]).float().to(self.device)
        self.arm_l_pose         = torch.from_numpy(traj["arm_l_pose"]).float().to(self.device)
        self.arm_r_pose         = torch.from_numpy(traj["arm_r_pose"]).float().to(self.device)
        self.joints_l           = torch.from_numpy(traj["joints_l"]).float().to(self.device)
        self.joints_r           = torch.from_numpy(traj["joints_r"]).float().to(self.device)
        self.joint_vel_l        = torch.from_numpy(traj["joint_vel_l"]).float().to(self.device)
        self.joint_vel_r        = torch.from_numpy(traj["joint_vel_r"]).float().to(self.device)
        self.joints_target_l    = torch.from_numpy(traj["joints_target_l"]).float().to(self.device)
        self.joints_target_r    = torch.from_numpy(traj["joints_target_r"]).float().to(self.device)
        self.EE_poses_l         = torch.from_numpy(traj["EE_poses_l"]).float().to(self.device)
        self.EE_poses_r         = torch.from_numpy(traj["EE_poses_r"]).float().to(self.device)
        self.dt                 = float(traj["dt"])

        # TODO: Support last trajectory point
        self.cfg.episode_length_s = self.dt * (self.obj_poses.shape[0] - 1)

        ur5e_l_cfg = get_ur5e_cfg(self.cfg.ur5e_l_prim_path, self.arm_l_pose, self.cfg)
        ur5e_r_cfg = get_ur5e_cfg(self.cfg.ur5e_r_prim_path, self.arm_r_pose, self.cfg)

        self.ur5e_l = Articulation(ur5e_l_cfg)
        self.ur5e_r = Articulation(ur5e_r_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)

        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}

        self.ee_contact_sensors = [ContactSensor(cfg) for cfg in self.cfg.ee_contact_sensors]

        # Regularization stuff
        self.prev_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 12), device=self.device)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["ur5e_left"] = self.ur5e_l
        self.scene.articulations["ur5e_right"] = self.ur5e_r
        # add object to the scene
        self.scene.rigid_objects["object"] = self.object
        # add table to the scene
        self.scene.rigid_objects["table"] = self.table
        # add sensors to the scene
        for name, sensor in self.illegal_contact_sensors.items():
            self.scene.sensors[f"illegal_contact_sensor_{name}"] = sensor
        for i, sensor in enumerate(self.ee_contact_sensors):
            self.scene.sensors[f"ee_contact_sensor_{i}"] = sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.ee_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/eeMarkers",
                markers={
                    "ee_l": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "ee_r": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
        )

        self.forearm_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/forearmMarkers",
                markers={
                    "cyl_p1": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "cyl_p2": sim_utils.SphereCfg(
                        radius=0.05,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            )
        )

        self.cube_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/myMarkers",
                markers={
                    "cube": sim_utils.CuboidCfg(
                        size=(0.4, 0.6, 0.06),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.7, 0.7, 0.7),
                        ),                
                    ),
                },
            )
        )

        self.cube_marker.set_visibility(False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        EE_pos_l = self.EE_poses_l[self.episode_length_buf, :3] + self.scene.env_origins
        EE_pos_r = self.EE_poses_r[self.episode_length_buf, :3] + self.scene.env_origins

        # Visualize EE markers
        ee_marker_pos = torch.stack([EE_pos_l, EE_pos_r], dim=1).view(-1, 3)
        self.ee_markers.visualize(translations=ee_marker_pos)

        closest_point_l = self._get_closest_point_on_forearm(self.ur5e_l)
        closest_point_r = self._get_closest_point_on_forearm(self.ur5e_r)

        # Visualize Forearm markers
        forearm_marker_pos = torch.stack([closest_point_l, closest_point_r], dim=1).view(-1, 3)
        self.forearm_markers.visualize(translations=forearm_marker_pos)

        obj_pos = self.obj_poses[self.episode_length_buf, :3] + self.scene.env_origins
        obj_quat = self.obj_poses[self.episode_length_buf, 3:]

        self.cube_marker.visualize(translations=obj_pos, orientations=obj_quat)

    def get_joint_targets(self):
        q_l = self.joints_target_l[self.episode_length_buf] + self.cfg.action_scale * self.actions[:, :6]
        q_r = self.joints_target_r[self.episode_length_buf] + self.cfg.action_scale * self.actions[:, 6:]

        return q_l, q_r


    def _apply_action(self) -> None:
        q_l, q_r = self.get_joint_targets()

        self.ur5e_l.set_joint_position_target(q_l)
        self.ur5e_r.set_joint_position_target(q_r)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._get_joint_pos(relative=True),
                self._get_joint_vel(),
                self._get_obj_pos(relative=True),
                self._get_obj_quat(relative=True),
                self._get_obj_vel(),
                self.episode_length_buf[:, None] / self.max_episode_length,
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
        # self.episode_length_buf[env_ids] = torch.randint(0, 40, (len(env_ids),), device=self.device, dtype=torch.long)

        initial_joint_pos_l = self.joints_l[self.episode_length_buf[env_ids]]
        initial_joint_vel_l = self.joint_vel_l[self.episode_length_buf[env_ids]]
        self.ur5e_l.write_joint_state_to_sim(initial_joint_pos_l, initial_joint_vel_l, env_ids=env_ids)

        initial_joint_pos_r = self.joints_r[self.episode_length_buf[env_ids]]
        initial_joint_vel_r = self.joint_vel_r[self.episode_length_buf[env_ids]]
        self.ur5e_r.write_joint_state_to_sim(initial_joint_pos_r, initial_joint_vel_r, env_ids=env_ids)

        # Reset Object
        initial_object_pose = self.obj_poses[self.episode_length_buf[env_ids]].clone()
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = self.obj_vel[self.episode_length_buf[env_ids]]

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
        joint_acc *= torch.abs(joint_acc) > self.cfg.tol_joint_acc
        joint_acc_penalty = joint_acc.square().sum(dim=-1)
        rew_joint_acc = self.cfg.w_joint_acc * joint_acc_penalty

        torque = torch.cat((self.ur5e_l.data.applied_torque.clone(), self.ur5e_r.data.applied_torque.clone()), 1)
        torque *= torch.abs(torque) > self.cfg.tol_joint_torque
        torque_penalty = torque.square().sum(dim=-1)
        rew_torque = self.cfg.w_joint_torque * torque_penalty        

        action_rate_error = (self.actions - self.prev_actions)
        action_rate_error *= torch.abs(action_rate_error) > self.cfg.tol_action_rate
        action_rate_penalty = action_rate_error.square().sum(dim=-1)
        rew_action_rate = self.cfg.w_action_rate * action_rate_penalty

        total_illegal_force = torch.zeros((self.num_envs,), device=self.device)
        for sensor in self.illegal_contact_sensors.values():
            f_abs = sensor.data.force_matrix_w.norm(dim=-1)
            f_abs_clamped = f_abs.clamp(max=self.cfg.max_contact_force)
            total_illegal_force += f_abs_clamped.sum(dim=-1).flatten()

        mean_ee_force = torch.zeros((self.num_envs,), device=self.device)
        for sensor in self.ee_contact_sensors:
            mean_ee_force += sensor.data.force_matrix_w.norm(dim=-1).sum(dim=-1).flatten() / len(self.ee_contact_sensors)

        flange_to_forearm_dist_l = self._get_flange_to_forearm_distance(self.ur5e_l)
        flange_to_forearm_dist_r = self._get_flange_to_forearm_distance(self.ur5e_r)

        # Penalize if distance is less than the threshold
        # Using a quadratic penalty for distances below the threshold
        penalty_dist_l = torch.clamp(self.cfg.max_flange_forearm_distance - flange_to_forearm_dist_l, min=0.0).square()
        penalty_dist_r = torch.clamp(self.cfg.max_flange_forearm_distance - flange_to_forearm_dist_r, min=0.0).square()
        rew_flange_forearm_collision = self.cfg.w_flange_forearm_collision * (penalty_dist_l + penalty_dist_r)

        rew_illegal_contact = self.cfg.w_illegal_contact * total_illegal_force

        rew_regularization = self.cfg.w_regularization * (
            rew_joint_acc + rew_torque + rew_action_rate + rew_illegal_contact + rew_flange_forearm_collision
        )

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
            "Rewards_regularization/total": rew_regularization.mean(),
            "Rewards_regularization/joint_acceleration": rew_joint_acc.mean(), 
            "Rewards_regularization/torque": rew_torque.mean(),
            "Rewards_regularization/action_rate": rew_action_rate.mean(),
            "Rewards_regularization/illegal_contact": rew_illegal_contact.mean(),
            "Rewards_regularization/flange_forearm_collision": rew_flange_forearm_collision.mean(),
            "Extra/mean_EE_force": mean_ee_force.mean(),
        }
        
        total_reward = rew_task + rew_track - rew_regularization

        # Can update prev_action/joint_vel now
        self.prev_actions[:] = self.actions[:]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

    
    def _get_joint_pos(self, relative=True):
        joint_pos_l = self.ur5e_l.data.joint_pos.clone()
        joint_pos_r = self.ur5e_r.data.joint_pos.clone()

        if relative:
            joint_pos_l -= self.joints_l[self.episode_length_buf]
            joint_pos_r -= self.joints_r[self.episode_length_buf]

        return torch.cat((joint_pos_l, joint_pos_r), 1)
    
    
    def _get_joint_vel(self):
        ur5e_l_joint_vel = self.ur5e_l.data.joint_vel.clone()
        ur5e_r_joint_vel = self.ur5e_r.data.joint_vel.clone()

        return torch.cat((ur5e_l_joint_vel, ur5e_r_joint_vel), 1)
    
    def _get_forearm_endpoints(self, robot: Articulation):
        # Height of the forearm cylinder is ~0.4225.
        forearm_length = 0.4225
        # Local offset of the Cylinder center relative to forearm_link
        p2_local = torch.tensor([-forearm_length, 0.0, 0.0], device=self.device)

        forearm_pos = robot.data.body_pos_w[:, self.forearm_link_idx]
        forearm_quat = robot.data.body_quat_w[:, self.forearm_link_idx]

        # Map link frame to world frame
        p2 = forearm_pos + quat_apply(forearm_quat, p2_local.repeat(self.num_envs, 1))
        return forearm_pos, p2
    
    def _get_closest_point_on_forearm(self, robot: Articulation):
        flange_pos = robot.data.body_pos_w[:, self.flange_idx]
        p1, p2 = self._get_forearm_endpoints(robot)

        # Calculate distance from flange_pos to line segment [p1, p2]
        line_vec = p2 - p1
        p1_to_flange = flange_pos - p1
        
        line_len_sq = torch.sum(line_vec**2, dim=-1)
        # Project flange_pos onto the line segment, clamped between 0 and 1
        t = torch.sum(p1_to_flange * line_vec, dim=-1) / line_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        
        closest_point = p1 + t.unsqueeze(-1) * line_vec

        return closest_point

    def _get_flange_to_forearm_distance(self, robot: Articulation):
        flange_pos = robot.data.body_pos_w[:, self.flange_idx]
        p1, p2 = self._get_forearm_endpoints(robot)

        # Calculate distance from flange_pos to line segment [p1, p2]
        line_vec = p2 - p1
        p1_to_flange = flange_pos - p1
        
        line_len_sq = torch.sum(line_vec**2, dim=-1)
        # Project flange_pos onto the line segment, clamped between 0 and 1
        t = torch.sum(p1_to_flange * line_vec, dim=-1) / line_len_sq
        t = torch.clamp(t, 0.0, 1.0)
        
        closest_point = p1 + t.unsqueeze(-1) * line_vec
        distance = torch.norm(flange_pos - closest_point, dim=-1)
        
        return distance
    
    def _get_EE_pos(self, relative=True) -> torch.Tensor:
        EE_pos_l = self.ur5e_l.data.body_pos_w[:, self.EE_link_idx].clone() - self.scene.env_origins
        EE_pos_r = self.ur5e_r.data.body_pos_w[:, self.EE_link_idx].clone() - self.scene.env_origins

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
        # TODO: check orientation in usd file. might not match the one from the planner
        EE_quat_l = self.ur5e_l.data.body_quat_w[:, self.EE_link_idx].clone()
        EE_quat_r = self.ur5e_r.data.body_quat_w[:, self.EE_link_idx].clone()

        if relative:
            desired_quat_l = self.EE_poses_l[self.episode_length_buf, 3:]
            EE_quat_l = quat_mul(desired_quat_l, quat_inv(EE_quat_l))
            desired_quat_r = self.EE_poses_r[self.episode_length_buf, 3:]
            EE_quat_r = quat_mul(desired_quat_r, quat_inv(EE_quat_r))

        return torch.cat((EE_quat_l, EE_quat_r), 1)
    
    def _get_EE_quat_error(self):
        # TODO: check orientation in usd file. might not match the one from the planner
        EE_quat_l = self.ur5e_l.data.body_quat_w[:, self.EE_link_idx].clone()
        EE_quat_r = self.ur5e_r.data.body_quat_w[:, self.EE_link_idx].clone()

        desired_quat_l = self.EE_poses_l[self.episode_length_buf, 3:]
        desired_quat_r = self.EE_poses_r[self.episode_length_buf, 3:]

        error_l = torch.abs(quat_error_magnitude(EE_quat_l, desired_quat_l))
        error_r = torch.abs(quat_error_magnitude(EE_quat_r, desired_quat_r))

        return error_l, error_r
    
    def _get_EE_vel(self) -> torch.Tensor:
        EE_vel_l = self.ur5e_l.data.body_vel_w[:, self.EE_link_idx].clone()
        EE_vel_r = self.ur5e_r.data.body_vel_w[:, self.EE_link_idx].clone()

        return torch.cat((EE_vel_l, EE_vel_r), 1)
    
    def _get_obj_pos(self, relative=True):
        obj_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins

        if relative:
            desired_obj_pos = self.obj_poses[self.episode_length_buf, :3]
            obj_pos -= desired_obj_pos

        return obj_pos
    
    def _get_obj_pos_error(self):
        return torch.norm(self._get_obj_pos(relative=True), dim=-1)
    
    def _get_obj_quat(self, relative=True):
        obj_quat = self.object.data.root_quat_w.clone()

        if relative:
            desired_obj_quat = self.obj_poses[self.episode_length_buf, 3:]
            obj_quat = quat_mul(desired_obj_quat, quat_inv(obj_quat))

        return obj_quat
    
    def _get_obj_quat_error(self):
        obj_quat = self.object.data.root_quat_w.clone()
        desired_obj_quat = self.obj_poses[self.episode_length_buf, 3:]
        return torch.abs(quat_error_magnitude(obj_quat, desired_obj_quat))

    def _get_obj_vel(self):
        return self.object.data.root_vel_w.clone()

