# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch
import numpy as np
import omni.usd
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.sim.schemas import modify_collision_properties
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from BoxLift.tasks.direct.boxhinge.boxhinge_env_cfg import *

from isaaclab.utils.math import quat_apply, quat_mul, quat_inv, quat_error_magnitude

class BoxhingeEnv(DirectRLEnv):
    cfg: BoxhingeEnvCfg

    def __init__(self, cfg: BoxhingeEnvCfg, render_mode: str | None = None, **kwargs):
        # __post_init__ must run before super().__init__ so cfg.observation_space reflects
        # any Hydra CLI overrides before DirectRLEnv allocates observation buffers.
        cfg.__post_init__()
        super().__init__(cfg, render_mode, **kwargs)

        self.EE_link_idx = self.ur5e.body_names.index("wrist_3_link")
        self.flange_idx = self.ur5e.body_names.index("wrist_3_link")
        self.forearm_link_idx = self.ur5e.body_names.index("forearm_link")

        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.dphase = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_phase = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_slowdown = torch.zeros(self.num_envs, device=self.device)
        self._err_task_prev = torch.zeros(self.num_envs, device=self.device)
        self._err_track_prev = torch.zeros(self.num_envs, device=self.device)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)

        T = self.obj_poses.shape[0]
        self._segment_size = max(1, int(round(self.cfg.phase_segment_s / self.dt)))
        self._num_segments = max(1, (T - 1 + self._segment_size - 1) // self._segment_size)
        self.segment_scores = torch.full(
            (self._num_segments,), 0.5, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        traj = np.load(self.cfg.trajectory_path)

        self.obj_poses          = torch.from_numpy(traj["obj_poses"]).float().to(self.device)
        self.obj_vel            = torch.from_numpy(traj["obj_vel"]).float().to(self.device)
        self.arm_pose           = torch.from_numpy(traj["arm_pose"]).float().to(self.device)
        self.joints             = torch.from_numpy(traj["joints"]).float().to(self.device)
        self.joint_vel          = torch.from_numpy(traj["joint_vel"]).float().to(self.device)
        self.joints_target      = torch.from_numpy(traj["joints_target"]).float().to(self.device)
        self.EE_poses           = torch.from_numpy(traj["EE_poses"]).float().to(self.device)
        self.dt                 = float(traj["dt"])

        if "object_dims" in traj:
            self.cfg.object_dims = tuple(traj["object_dims"].tolist())
            self.cfg.cube_cfg.spawn.size = self.cfg.object_dims
        if "object_mass" in traj:
            self.cfg.object_mass = float(traj["object_mass"])
            self.cfg.cube_cfg.spawn.mass_props.mass = self.cfg.object_mass

        traj_duration = self.dt * (self.obj_poses.shape[0] - 1)
        if self.cfg.max_episode_steps > 0:
            self.cfg.episode_length_s = self.cfg.max_episode_steps * self.dt
        elif self.cfg.enable_phase_slowdown:
            self.cfg.episode_length_s = traj_duration * self.cfg.max_slowdown_multiplier
        else:
            self.cfg.episode_length_s = traj_duration
        self.cfg.episode_length_s += self.cfg.post_traj_hold_s

        ur5e_cfg = get_ur5e_cfg(self.cfg.ur5e_prim_path, self.arm_pose, self.cfg)
        self.ur5e = Articulation(ur5e_cfg)

        # Ground is raised 1.8cm for the box surface; disable base_link collision to avoid
        # the robot base clipping into the raised area near the box.
        modify_collision_properties(
            "/World/envs/env_0/ur5e/base_link",
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)
        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}
        self.ee_contact_sensor = ContactSensor(self.cfg.ee_contact_sensor_cfg)

        # Gate: which trajectory steps have active box motion (and dilated margins).
        obj_vel_mag = self.obj_vel[:, :3].norm(dim=-1) + self.obj_vel[:, 3:].norm(dim=-1)
        moving = (obj_vel_mag > self.cfg.eef_box_gate_obj_vel_eps).float()
        if self.cfg.eef_box_gate_dilation_steps > 0:
            k = 2 * int(self.cfg.eef_box_gate_dilation_steps) + 1
            moving = torch.nn.functional.max_pool1d(
                moving.view(1, 1, -1), kernel_size=k, stride=1,
                padding=int(self.cfg.eef_box_gate_dilation_steps),
            ).view(-1)
        self.eef_box_gate_mask = moving.bool()

        rsi_contact = (obj_vel_mag > self.cfg.eef_box_gate_obj_vel_eps).float()
        if self.cfg.rsi_contact_dilation_steps > 0:
            k = 2 * int(self.cfg.rsi_contact_dilation_steps) + 1
            rsi_contact = torch.nn.functional.max_pool1d(
                rsi_contact.view(1, 1, -1), kernel_size=k, stride=1,
                padding=int(self.cfg.rsi_contact_dilation_steps),
            ).view(-1)
        self.rsi_valid_phases = torch.nonzero(~rsi_contact.bool(), as_tuple=False).squeeze(-1)

        self.prev_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self._dr_obj_mass = torch.zeros((self.num_envs, 1), device=self.device)
        self._dr_obj_friction = torch.zeros((self.num_envs, 3), device=self.device)

        self.obs_history = torch.zeros(
            (self.num_envs, self.cfg.obs_history_steps, self.cfg.per_step_feature_dim),
            device=self.device,
        )

        self.obj_pose_delay_buf = torch.zeros(
            (self.num_envs, self.cfg.obs_obj_delay_steps + 1, 7),
            device=self.device,
        )
        self.obj_phase_delay_buf = torch.zeros(
            (self.num_envs, self.cfg.obs_obj_delay_steps + 1),
            device=self.device,
        )
        self.obj_obs_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.obj_obs_last_pose = torch.zeros(self.num_envs, 7, device=self.device)
        self.obj_obs_last_rel = torch.zeros(self.num_envs, 7, device=self.device)
        self.obj_obs_bias_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_obs_bias_ori_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.obj_obs_bias_ori_quat[:, 0] = 1.0

        self.ee_contact_delay_buf = torch.zeros(
            (self.num_envs, self.cfg.contact_obs_delay_steps + 1),
            device=self.device,
        )

        self.post_traj_step_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.post_traj_hold_steps = int(round(self.cfg.post_traj_hold_s / self.dt))

        # VOC gains. kv derived from critical-damping estimate (I ≈ m·d_max²/12).
        mass = float(self.cfg.cube_cfg.spawn.mass_props.mass)
        d_max = float(max(self.cfg.cube_cfg.spawn.size))
        inertia_est = mass * (d_max ** 2) / 12.0
        self.voc_kp_pos = float(self.cfg.voc_kp_pos)
        self.voc_kp_rot = float(self.cfg.voc_kp_rot)
        self.voc_kv_pos = self.cfg.voc_kv_pos_scale * (self.voc_kp_pos * mass) ** 0.5
        self.voc_kv_rot = self.cfg.voc_kv_rot_scale * (self.voc_kp_rot * inertia_est) ** 0.5
        self._voc_ep_rew_task = torch.zeros(self.num_envs, device=self.device)
        self._voc_ep_rew_track = torch.zeros(self.num_envs, device=self.device)
        self._voc_ep_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._voc_buf_task = torch.full(
            (self.cfg.voc_reward_window_size,), float("nan"), device=self.device
        )
        self._voc_buf_track = torch.full(
            (self.cfg.voc_reward_window_size,), float("nan"), device=self.device
        )
        self._voc_buf_idx = 0
        self._voc_decay_step_counter = 0

        self.perturbation_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.perturbation_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.perturbation_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["ur5e"] = self.ur5e
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        for name, sensor in self.illegal_contact_sensors.items():
            self.scene.sensors[f"illegal_contact_sensor_{name}"] = sensor
        self.scene.sensors["ee_contact_sensor"] = self.ee_contact_sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.ee_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/eeMarkers",
                markers={
                    "ee": sim_utils.SphereCfg(
                        radius=0.02,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                },
            )
        )

        self.cube_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/myMarkers",
                markers={
                    "cube": sim_utils.CuboidCfg(
                        size=self.cfg.cube_cfg.spawn.size,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.7, 0.7, 0.7),
                        ),
                    ),
                },
            )
        )

        self.cube_marker.set_visibility(True)

    def _apply_perturbations(self):
        self.perturbation_counter = torch.clamp(self.perturbation_counter - 1, min=0)

        new_perturbation = torch.rand(self.num_envs, device=self.device) < self.cfg.perturbation_probability
        new_perturbation &= self.perturbation_counter == 0
        new_ids = new_perturbation.nonzero(as_tuple=False).squeeze(-1)

        if len(new_ids) > 0:
            self.perturbation_counter[new_ids] = self.cfg.perturbation_duration_steps
            self.perturbation_forces[new_ids, 0] = (
                torch.randn(len(new_ids), 3, device=self.device)
                * self.cfg.perturbation_force_std
            )
            self.perturbation_torques[new_ids, 0] = (
                torch.randn(len(new_ids), 3, device=self.device)
                * self.cfg.perturbation_torque_std
            )

        expired = self.perturbation_counter == 0
        self.perturbation_forces[expired] = 0.0
        self.perturbation_torques[expired] = 0.0

        self.ur5e.set_external_force_and_torque(
            self.perturbation_forces,
            self.perturbation_torques,
            body_ids=[self.forearm_link_idx],
            is_global=True,
        )

    def _interp(self, traj: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        """Linear interpolation along trajectory time axis. traj: (T, D), returns (num_envs, D)."""
        if phase is None:
            phase = self.phase
        T = traj.shape[0]
        p = phase.clamp(0.0, T - 1 - 1e-6)
        i0 = p.floor().long()
        i1 = (i0 + 1).clamp(max=T - 1)
        a = (p - i0.float()).unsqueeze(-1)
        return (1.0 - a) * traj[i0] + a * traj[i1]

    def _nlerp(self, traj_quat: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized lerp for unit quaternions (wxyz). Hemisphere-corrected for short path."""
        if phase is None:
            phase = self.phase
        T = traj_quat.shape[0]
        p = phase.clamp(0.0, T - 1 - 1e-6)
        i0 = p.floor().long()
        i1 = (i0 + 1).clamp(max=T - 1)
        a = (p - i0.float()).unsqueeze(-1)
        q0 = traj_quat[i0]
        q1 = traj_quat[i1]
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        q = (1.0 - a) * q0 + a * q1
        return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        if self.cfg.enable_phase_slowdown:
            raw = self.actions[:, 6]
            if self.cfg.phase_mapping == "cubic_bidir":
                scale = max(1.0 - self.cfg.dphase_min, self.cfg.dphase_max - 1.0)
                self.dphase = (1.0 + scale * raw ** 3).clamp(
                    min=self.cfg.dphase_min, max=self.cfg.dphase_max
                )
            else:
                self.dphase = (1.0 + (1.0 - self.cfg.dphase_min) * torch.tanh(raw)).clamp(
                    min=self.cfg.dphase_min, max=1.0
                )
        else:
            self.dphase = torch.ones_like(self.phase)

        self._apply_perturbations()
        self._apply_voc()

        EE_pos_ref   = self._interp(self.EE_poses[:, :3])
        obj_pos_ref  = self._interp(self.obj_poses[:, :3])
        obj_quat_ref = self._nlerp(self.obj_poses[:, 3:])
        rel_ref_pos_box = quat_apply(quat_inv(obj_quat_ref), EE_pos_ref - obj_pos_ref)
        obj_pos_actual  = self._get_obj_pos(relative=False)
        obj_quat_actual = self._get_obj_quat(relative=False)
        target_EE_pos_world = obj_pos_actual + quat_apply(obj_quat_actual, rel_ref_pos_box)
        self.ee_markers.visualize(translations=target_EE_pos_world + self.scene.env_origins)

        obj_pos = self._interp(self.obj_poses[:, :3]) + self.scene.env_origins
        obj_quat = self._nlerp(self.obj_poses[:, 3:])
        self.cube_marker.visualize(translations=obj_pos, orientations=obj_quat)

        # Cache once per policy step so substeps in the decimation window all chase the
        # same ZOH target (modes C/D depend on q_current; recomputing would cause drift).
        self._cached_joint_target = self.get_joint_targets().clamp(
            self.ur5e.data.joint_pos_limits[..., 0],
            self.ur5e.data.joint_pos_limits[..., 1],
        )

    def _apply_voc(self):
        """Virtual Object Controller (DexMachina, Mandi et al. 2025): 6-DoF PD wrench on
        the cube driving it toward the reference. Gains decay as the policy improves."""
        if not self.cfg.voc_enabled or self.voc_kp_pos <= 0.0:
            n = self.num_envs
            self.object.set_external_force_and_torque(
                torch.zeros(n, 1, 3, device=self.device),
                torch.zeros(n, 1, 3, device=self.device),
                is_global=True,
            )
            return

        ref_pos = self._interp(self.obj_poses[:, :3])
        ref_quat = self._nlerp(self.obj_poses[:, 3:])
        ref_vel = self._interp(self.obj_vel)

        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        obj_quat = self.object.data.root_quat_w
        obj_vel = self.object.data.root_vel_w

        pos_err = ref_pos - obj_pos
        lin_vel_err = obj_vel[:, :3] - ref_vel[:, :3]
        force = self.voc_kp_pos * pos_err - self.voc_kv_pos * lin_vel_err

        # Small-angle axis-angle from quaternion error.
        q_err = quat_mul(ref_quat, quat_inv(obj_quat))
        sign_w = torch.where(q_err[:, 0:1] >= 0,
                             torch.ones_like(q_err[:, 0:1]),
                             -torch.ones_like(q_err[:, 0:1]))
        rot_err = 2.0 * sign_w * q_err[:, 1:]
        ang_vel_err = obj_vel[:, 3:] - ref_vel[:, 3:]
        torque = self.voc_kp_rot * rot_err - self.voc_kv_rot * ang_vel_err

        self.object.set_external_force_and_torque(
            force.unsqueeze(1), torque.unsqueeze(1), is_global=True,
        )


    def get_joint_targets_A(self):
        """Residual on planner targets. Planner feedforward is baked into the action."""
        return self._interp(self.joints_target) + self._action_scale * self.actions[:, :6]

    def get_joint_targets_B(self):
        """Residual on trajectory positions. Pair with feedforward obs."""
        return self._interp(self.joints) + self._action_scale * self.actions[:, :6]

    def get_joint_targets_C(self):
        """Residual on current joint positions. Pair with feedforward obs."""
        return self._get_joint_pos() + self._action_scale * self.actions[:, :6]

    def get_joint_targets_D(self):
        """Blended residual: (1-α)·planner_feedforward + (α + ε(1-α))·scale·action from current q."""
        planner_pd_error = self._interp(self.joints_target) - self._interp(self.joints)
        alpha = self._curriculum_alpha()
        eps = float(self.cfg.action_alpha_floor)
        action_gain = alpha + eps * (1.0 - alpha)
        policy_pd_error = (1.0 - alpha) * planner_pd_error + action_gain * self._action_scale * self.actions[:, :6]
        return self._get_joint_pos() + policy_pd_error

    def get_joint_targets(self):
        mode = self.cfg.action_mode
        if mode == "A":
            return self.get_joint_targets_A()
        if mode == "B":
            return self.get_joint_targets_B()
        if mode == "C":
            return self.get_joint_targets_C()
        if mode == "D":
            return self.get_joint_targets_D()
        raise ValueError(f"Unknown action_mode: {mode!r}")

    def _apply_action(self) -> None:
        self.ur5e.set_joint_position_target(self._cached_joint_target)

    def _get_feedforward(self):
        """Planner's intended feedforward: joints_target - joints (proportional to desired PD force)."""
        return self._interp(self.joints_target) - self._interp(self.joints)

    def _get_noisy_obj_obs(self):
        """Simulated tracker: fixed delay, sub-rate updates, per-episode bias, per-fire noise."""
        obj_pos_now  = self.object.data.root_pos_w.clone() - self.scene.env_origins
        obj_quat_now = self.object.data.root_quat_w.clone()

        pose_now = torch.cat([obj_pos_now, obj_quat_now], dim=-1)
        self.obj_pose_delay_buf = torch.roll(self.obj_pose_delay_buf, shifts=-1, dims=1)
        self.obj_pose_delay_buf[:, -1] = pose_now
        self.obj_phase_delay_buf = torch.roll(self.obj_phase_delay_buf, shifts=-1, dims=1)
        self.obj_phase_delay_buf[:, -1] = self.phase

        self.obj_obs_counter += 1
        fires = self.obj_obs_counter >= self.cfg.obs_obj_update_period
        fires_idx = fires.nonzero(as_tuple=False).squeeze(-1)

        if fires_idx.numel() > 0:
            n_f = fires_idx.numel()
            sampled = self.obj_pose_delay_buf[fires_idx, 0]
            sampled_phase = self.obj_phase_delay_buf[fires_idx, 0]
            sampled_pos  = sampled[:, :3]
            sampled_quat = sampled[:, 3:]
            sampled_pos = sampled_pos + self.obj_obs_bias_pos[fires_idx]
            sampled_quat = quat_mul(self.obj_obs_bias_ori_quat[fires_idx], sampled_quat)
            sampled_pos += self.cfg.obs_obj_pos_noise * torch.randn(n_f, 3, device=self.device)
            if self.cfg.obs_obj_ori_noise > 0:
                aa = self.cfg.obs_obj_ori_noise * torch.randn(n_f, 3, device=self.device)
                delta_quat = torch.cat([torch.ones(n_f, 1, device=self.device), 0.5 * aa], dim=-1)
                delta_quat = delta_quat / delta_quat.norm(dim=-1, keepdim=True)
                sampled_quat = quat_mul(delta_quat, sampled_quat)
            sampled_quat = sampled_quat / sampled_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            past_ref_pos  = self._interp(self.obj_poses[:, :3], phase=sampled_phase)
            past_ref_quat = self._nlerp(self.obj_poses[:, 3:], phase=sampled_phase)
            rel_pos_fire  = sampled_pos - past_ref_pos
            rel_quat_fire = quat_mul(past_ref_quat, quat_inv(sampled_quat))
            self.obj_obs_last_pose[fires_idx, :3] = sampled_pos
            self.obj_obs_last_pose[fires_idx, 3:] = sampled_quat
            self.obj_obs_last_rel[fires_idx, :3]  = rel_pos_fire
            self.obj_obs_last_rel[fires_idx, 3:]  = rel_quat_fire
            self.obj_obs_counter[fires_idx] = 0

        obj_pos  = self.obj_obs_last_pose[:, :3]
        obj_quat = self.obj_obs_last_pose[:, 3:]
        rel_pos  = self.obj_obs_last_rel[:, :3]
        rel_quat = self.obj_obs_last_rel[:, 3:]

        return rel_pos, rel_quat, obj_pos, obj_quat

    def _get_observations(self) -> dict:
        T = self.obj_poses.shape[0]
        self.phase = (self.phase + self.dphase).clamp(0.0, T - 1 - 1e-6)

        feature_parts = [self._get_joint_pos(relative=True), self._get_joint_vel(relative=True)]

        need_obj = self.cfg.include_object_obs
        if need_obj:
            obj_rel_pos, obj_rel_quat, obj_abs_pos, obj_abs_quat = self._get_noisy_obj_obs()
            feature_parts.extend([obj_rel_pos, obj_rel_quat])

        if self.cfg.include_absolute_obs:
            feature_parts.extend([
                self._get_joint_pos(relative=False),
                self._get_joint_vel(relative=False),
            ])
            if need_obj:
                feature_parts.extend([obj_abs_pos, obj_abs_quat])

        if self.cfg.include_contact_obs:
            ee_force_mag = self.ee_contact_sensor.data.force_matrix_w.norm(dim=-1)
            total_force_mag = ee_force_mag.sum(dim=(-1, -2))
            in_contact = (total_force_mag > self.cfg.contact_threshold).float()
            self.ee_contact_delay_buf = torch.roll(self.ee_contact_delay_buf, shifts=-1, dims=1)
            self.ee_contact_delay_buf[:, -1] = in_contact
            delayed = self.ee_contact_delay_buf[:, :1]
            if self.cfg.contact_obs_flip_prob > 0.0:
                flip_mask = torch.rand_like(delayed) < self.cfg.contact_obs_flip_prob
                delayed = torch.where(flip_mask, 1.0 - delayed, delayed)
            feature_parts.append(delayed)

        current_features = torch.cat(feature_parts, dim=-1)

        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = current_features

        phase_obs = (self.phase / (T - 1)).unsqueeze(-1)
        obs_parts = [self.obs_history.flatten(start_dim=1), phase_obs]

        # Future reference obj pose look-ahead: (future_ref - current_ref) for each configured offset,
        # plus the absolute future ref pose if include_absolute_obs.
        if self.cfg.future_obs_steps:
            cur_pos = self._interp(self.obj_poses[:, :3])
            cur_quat = self._nlerp(self.obj_poses[:, 3:])
            inv_cur_quat = quat_inv(cur_quat)
            futures = []
            for k in self.cfg.future_obs_steps:
                fut_phase = self.phase + float(k)
                fut_pos = self._interp(self.obj_poses[:, :3], phase=fut_phase)
                fut_quat = self._nlerp(self.obj_poses[:, 3:], phase=fut_phase)
                futures.append(fut_pos - cur_pos)
                futures.append(quat_mul(fut_quat, inv_cur_quat))
                if self.cfg.include_absolute_obs:
                    futures.append(fut_pos)
                    futures.append(fut_quat)
            obs_parts.append(torch.cat(futures, dim=-1))

        if self.cfg.include_prev_actions:
            obs_parts.append(self.prev_actions)

        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs, "privileged": self._get_privileged_obs()}

    def _get_privileged_obs(self) -> torch.Tensor:
        T = self.obj_poses.shape[0]

        clean_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins
        clean_quat = self.object.data.root_quat_w.clone()
        clean_vel = self.object.data.root_vel_w.clone()
        obj_block = torch.cat([clean_pos, clean_quat, clean_vel], dim=-1)

        stiff = self.ur5e.data.joint_stiffness
        damp  = self.ur5e.data.joint_damping
        dr_block = torch.cat([self._dr_obj_mass, self._dr_obj_friction, stiff, damp], dim=-1)

        ref_obj_pos  = self._interp(self.obj_poses[:, :3])
        ref_obj_quat = self._nlerp(self.obj_poses[:, 3:])
        ref_obj_vel  = self._interp(self.obj_vel)
        ref_joints      = self._interp(self.joints)
        ref_joint_vel   = self._interp(self.joint_vel)
        ref_joints_tgt  = self._interp(self.joints_target)
        planner_pd      = ref_joints_tgt - ref_joints
        ref_EE          = torch.cat([self._interp(self.EE_poses[:, :3]), self._nlerp(self.EE_poses[:, 3:])], dim=-1)
        ref_block = torch.cat([ref_obj_pos, ref_obj_quat, ref_obj_vel,
                                ref_joints, ref_joint_vel, ref_joints_tgt, planner_pd,
                                ref_EE], dim=-1)

        f_mat = self.ee_contact_sensor.data.force_matrix_w.sum(dim=(1, 2))
        ee_f_mag = f_mat.norm(dim=-1, keepdim=True)
        ee_f_dir = f_mat / ee_f_mag.clamp(min=1e-6)
        illegal = torch.zeros((self.num_envs, 1), device=self.device)
        for sensor in self.illegal_contact_sensors.values():
            illegal[:, 0] += sensor.data.force_matrix_w.norm(dim=-1).sum(dim=-1).flatten()
        fl = self._get_flange_to_forearm_distance(self.ur5e).unsqueeze(-1)
        force_block = torch.cat([ee_f_mag, ee_f_dir, illegal, fl], dim=-1)

        pos_err, quat_err = self._compute_eef_box_rel_errors()
        eef_block = torch.stack([pos_err, quat_err], dim=-1)

        kp_pos_t = torch.full((self.num_envs, 1), self.voc_kp_pos, device=self.device)
        kp_rot_t = torch.full((self.num_envs, 1), self.voc_kp_rot, device=self.device)
        alpha_t  = torch.full((self.num_envs, 1), self._curriculum_alpha(), device=self.device)
        phase_norm = (self.phase / max(T - 1, 1)).unsqueeze(-1)
        voc_block = torch.cat([kp_pos_t, kp_rot_t, alpha_t, phase_norm], dim=-1)

        return torch.cat([obj_block, dr_block, ref_block, force_block, eef_block, voc_block], dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        at_end = self.phase >= (self.obj_poses.shape[0] - 1) - 1e-3
        self.post_traj_step_counter = torch.where(
            at_end,
            self.post_traj_step_counter + 1,
            torch.zeros_like(self.post_traj_step_counter),
        )
        time_out = at_end & (self.post_traj_step_counter > self.post_traj_hold_steps)
        if self.cfg.max_episode_steps > 0:
            time_out = time_out | (self.episode_length_buf >= self.cfg.max_episode_steps - 1)

        obj_pos_error = self._get_obj_pos_error()
        obj_quat_error = self._get_obj_quat_error()

        self.reset_terminated = obj_pos_error > self.cfg.max_obj_dist_from_traj
        self.reset_terminated |= obj_quat_error > self.cfg.max_obj_angle_from_traj

        return self.reset_terminated, time_out

    def _update_segment_scores(self, env_ids):
        """EMA update of per-segment failure rate. Credits all traversed segments with the terminal outcome."""
        f = self.reset_terminated[env_ids].float()
        start_seg = (self.episode_start_phase[env_ids].long() // self._segment_size).clamp(max=self._num_segments - 1)
        end_seg   = (self.phase[env_ids].long()              // self._segment_size).clamp(max=self._num_segments - 1)

        seg_idx = torch.arange(self._num_segments, device=self.device).unsqueeze(0)
        traversed = (seg_idx >= start_seg.unsqueeze(1)) & (seg_idx <= end_seg.unsqueeze(1))

        count = traversed.float().sum(dim=0)                                     # (S,)
        fails = (traversed.float() * f.unsqueeze(1)).sum(dim=0)                  # (S,)

        fail_rate = fails / count.clamp(min=1)
        update_w = (count > 0).float() * self.cfg.phase_resample_alpha
        self.segment_scores = (1.0 - update_w) * self.segment_scores + update_w * fail_rate
        lo, hi = self.cfg.phase_resample_clamp
        self.segment_scores.clamp_(lo, hi)

    def _max_start_phase(self, T: int) -> float:
        """Upper bound on RSI starting phase. Leaves room for max_episode_steps steps at full speed."""
        if self.cfg.max_episode_steps > 0:
            return float(max(1, T - 1 - self.cfg.max_episode_steps))
        return float(T - 2)

    def _sample_phase_failure_weighted(self, n: int, T: int) -> torch.Tensor:
        """Sample n starting phases weighted by segment failure rate. Uniform within each segment."""
        probs = self.segment_scores ** self.cfg.phase_resample_beta
        probs = probs / probs.sum()
        segs = torch.multinomial(probs, num_samples=n, replacement=True)
        within = torch.randint(0, self._segment_size, (n,), device=self.device)
        phase = (segs * self._segment_size + within).float()
        if self.cfg.enable_phase_slowdown:
            phase = phase + torch.rand(n, device=self.device)
        return phase.clamp(max=self._max_start_phase(T))

    def _reset_idx(self, env_ids: Sequence[int] | None, fixed_value: int = None): # type: ignore
        if env_ids is None:
            env_ids : Sequence[int] = self.scene._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)

        T = self.obj_poses.shape[0]

        # Must run before we overwrite self.phase[env_ids] — it still holds the final phase.
        if self.cfg.enable_failure_resampling and hasattr(self, "reset_terminated"):
            self._update_segment_scores(env_ids)

        if fixed_value is not None:
            self.phase[env_ids] = float(fixed_value) * torch.ones(len(env_ids), device=self.device)
        elif self.cfg.enable_failure_resampling:
            self.phase[env_ids] = self._sample_phase_failure_weighted(len(env_ids), T)
        else:
            upper_float = self._max_start_phase(T)
            max_int = int(upper_float)
            valid = self.rsi_valid_phases[self.rsi_valid_phases <= max_int]
            if valid.numel() == 0:
                valid = torch.arange(0, max(1, max_int + 1), device=self.device)
            picks = valid[torch.randint(0, valid.numel(), (len(env_ids),), device=self.device)]
            base = picks.float()
            if self.cfg.enable_phase_slowdown:
                base = (base + torch.rand(len(env_ids), device=self.device)).clamp(max=upper_float)
            self.phase[env_ids] = base

        # Boost phase=0 exposure: RSI under-samples the trajectory start relative to deployment.
        if fixed_value is None and self.cfg.reset_to_zero_prob > 0:
            zero_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.reset_to_zero_prob
            if zero_mask.any():
                self.phase[env_ids[zero_mask]] = 0.0
        self.episode_start_phase[env_ids] = self.phase[env_ids]
        # dphase=0 so the post-reset _get_observations advance is a no-op.
        self.dphase[env_ids] = 0.0

        idx = self.phase[env_ids].floor().long().clamp(max=T - 1)
        initial_joint_pos = self.joints[idx].clone()
        initial_joint_vel = self.joint_vel[idx].clone()
        pos_noise_std = torch.as_tensor(self.cfg.reset_joint_pos_noise, device=self.device, dtype=torch.float32)
        vel_noise_std = torch.as_tensor(self.cfg.reset_joint_vel_noise, device=self.device, dtype=torch.float32)
        initial_joint_pos += pos_noise_std * torch.randn_like(initial_joint_pos)
        initial_joint_vel += vel_noise_std * torch.randn_like(initial_joint_vel)
        self.ur5e.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel, env_ids=env_ids)

        initial_object_pose = self.obj_poses[idx].clone()
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = self.obj_vel[idx].clone()

        n = len(env_ids)
        initial_object_pose[:, 0:2] += self.cfg.reset_obj_pos_xy_noise * torch.randn(n, 2, device=self.device)
        # Yaw-only orientation noise keeps the box flat on the surface.
        yaw = self.cfg.reset_obj_ori_noise * torch.randn(n, device=self.device)
        half = 0.5 * yaw
        zeros = torch.zeros_like(half)
        delta_quat = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        initial_object_pose[:, 3:7] = quat_mul(delta_quat, initial_object_pose[:, 3:7])
        initial_object_vel[:, 0:2] += self.cfg.reset_obj_lin_vel_xy_noise * torch.randn(n, 2, device=self.device)
        initial_object_vel[:, 5]   += self.cfg.reset_obj_ang_vel_noise * torch.randn(n, device=self.device)

        self.object.write_root_pose_to_sim(initial_object_pose, env_ids)
        self.object.write_root_velocity_to_sim(initial_object_vel, env_ids)

        init_pose_env = torch.cat([
            initial_object_pose[:, :3] - self.scene.env_origins[env_ids],
            initial_object_pose[:, 3:7],
        ], dim=-1)
        self.obj_pose_delay_buf[env_ids]  = init_pose_env.unsqueeze(1)
        self.obj_phase_delay_buf[env_ids] = self.phase[env_ids].unsqueeze(1)
        self.ee_contact_delay_buf[env_ids] = 0.0
        self.post_traj_step_counter[env_ids] = 0
        self.obj_obs_last_pose[env_ids]   = init_pose_env
        self.obj_obs_last_rel[env_ids, :3] = 0.0
        self.obj_obs_last_rel[env_ids, 3:] = 0.0
        self.obj_obs_last_rel[env_ids, 3]  = 1.0  # identity quat w
        self.obj_obs_counter[env_ids] = self.cfg.obs_obj_update_period

        n_b = len(env_ids)
        if self.cfg.obs_obj_pos_bias_std > 0:
            self.obj_obs_bias_pos[env_ids] = (
                self.cfg.obs_obj_pos_bias_std * torch.randn(n_b, 3, device=self.device)
            )
        else:
            self.obj_obs_bias_pos[env_ids] = 0.0
        if self.cfg.obs_obj_ori_bias_std > 0:
            aa = self.cfg.obs_obj_ori_bias_std * torch.randn(n_b, 3, device=self.device)
            bias_quat = torch.cat([torch.ones(n_b, 1, device=self.device), 0.5 * aa], dim=-1)
            bias_quat = bias_quat / bias_quat.norm(dim=-1, keepdim=True)
            self.obj_obs_bias_ori_quat[env_ids] = bias_quat
        else:
            self.obj_obs_bias_ori_quat[env_ids, 0] = 1.0
            self.obj_obs_bias_ori_quat[env_ids, 1:] = 0.0

        n = len(env_ids)
        ep_steps = self._voc_ep_steps[env_ids].clamp(min=1).float()
        norm_task = self._voc_ep_rew_task[env_ids] / ep_steps
        norm_track = self._voc_ep_rew_track[env_ids] / ep_steps
        W = self.cfg.voc_reward_window_size
        slots = (self._voc_buf_idx + torch.arange(n, device=self.device)) % W
        self._voc_buf_task[slots] = norm_task
        self._voc_buf_track[slots] = norm_track
        self._voc_buf_idx = int((self._voc_buf_idx + n) % W)
        self._voc_ep_rew_task[env_ids] = 0.0
        self._voc_ep_rew_track[env_ids] = 0.0
        self._voc_ep_steps[env_ids] = 0

        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = initial_joint_vel
        self.obs_history[env_ids] = 0.0
        self.perturbation_counter[env_ids] = 0
        self.perturbation_forces[env_ids] = 0.0
        self.perturbation_torques[env_ids] = 0.0
        self.cumulative_slowdown[env_ids] = 0.0
        self._err_task_prev[env_ids] = 0.0
        self._err_track_prev[env_ids] = 0.0

        try:
            masses = self.object.root_physx_view.get_masses().to(self.device)
            self._dr_obj_mass[env_ids] = masses[env_ids]
            mat = self.object.root_physx_view.get_material_properties().to(self.device)
            self._dr_obj_friction[env_ids] = mat[env_ids, 0, :]
        except Exception as e:
            if not getattr(self, "_dr_readback_warned", False):
                print(f"[boxhinge] _reset_idx: DR cache readback failed: {e!r}.")
                self._dr_readback_warned = True

    def _curriculum_alpha(self) -> float:
        """α ∈ [0,1] curriculum schedule. force_alpha overrides for eval; 0 warmup steps → always 1."""
        if 0.0 <= self.cfg.force_alpha <= 1.0:
            return float(self.cfg.force_alpha)
        if self.cfg.alpha_warmup_steps > 0:
            return min(1.0, self.common_step_counter / self.cfg.alpha_warmup_steps)
        return 1.0

    def _reward_track(self, error, sigma, tolerance=0.0):
        error = error * (error > tolerance)
        if isinstance(sigma, (tuple, list)):
            sigmas = torch.tensor(sigma, device=error.device, dtype=error.dtype)
            kernels = torch.exp(-error.unsqueeze(-1) / (sigmas ** 2))  # (N, K)
            return kernels.mean(dim=-1)
        return torch.exp(-error / (sigma ** 2))

    def _get_rewards(self) -> torch.Tensor:
        obj_pos_error = self._get_obj_pos_error()
        obj_quat_error = self._get_obj_quat_error()

        if self.cfg.task_reward_form == "product":
            r_pos = torch.exp(-self.cfg.task_beta_pos * obj_pos_error)
            r_rot = torch.exp(-self.cfg.task_beta_rot * obj_quat_error)
            rew_obj_pos = r_pos
            rew_obj_quat = r_rot
            rew_obj_lin_vel = torch.zeros_like(r_pos)
            rew_obj_ang_vel = torch.zeros_like(r_pos)
            rew_obj_vel = rew_obj_lin_vel + rew_obj_ang_vel
            rew_task_unweighted = r_pos * r_rot
        else:
            rew_obj_pos = self.cfg.w_obj_pos * self._reward_track(
                obj_pos_error ** 2, self.cfg.sigma_obj_pos, self.cfg.tol_obj_pos)
            rew_obj_quat = self.cfg.w_obj_quat * self._reward_track(
                obj_quat_error ** 2, self.cfg.sigma_obj_quat, self.cfg.tol_obj_quat)
            obj_vel_rel = self._get_obj_vel(relative=True)
            obj_lin_vel_error = obj_vel_rel[:, :3].norm(dim=-1)
            obj_ang_vel_error = obj_vel_rel[:, 3:].norm(dim=-1)
            rew_obj_lin_vel = self.cfg.w_obj_lin_vel * self._reward_track(
                obj_lin_vel_error ** 2, self.cfg.sigma_obj_lin_vel, self.cfg.tol_obj_lin_vel)
            rew_obj_ang_vel = self.cfg.w_obj_ang_vel * self._reward_track(
                obj_ang_vel_error ** 2, self.cfg.sigma_obj_ang_vel, self.cfg.tol_obj_ang_vel)
            rew_obj_vel = rew_obj_lin_vel + rew_obj_ang_vel
            rew_task_unweighted = rew_obj_pos + rew_obj_quat + rew_obj_vel

        alpha = self._curriculum_alpha()
        w_task_eff  = self.cfg.w_task_start  + (self.cfg.w_task  - self.cfg.w_task_start)  * alpha
        w_track_eff = self.cfg.w_track_start + (self.cfg.w_track - self.cfg.w_track_start) * alpha

        rew_task = w_task_eff * rew_task_unweighted
        if self.cfg.task_scale_by_dphase:
            rew_task = self.dphase * rew_task

        T = self.eef_box_gate_mask.shape[0]
        phase_idx = self.phase.floor().long().clamp(max=T - 1)
        gate = self.eef_box_gate_mask[phase_idx].float()
        abs_gate = 1.0 - gate

        EE_pos_error = self._get_EE_pos_error()
        rew_EE_pos = abs_gate * self.cfg.w_eef_pos * self._reward_track(EE_pos_error ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)

        eef_quat_error = self._get_EE_quat_error()
        rew_EE_quat = abs_gate * self.cfg.w_eef_quat * self._reward_track(eef_quat_error ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)

        joint_pos_err_per_joint = self._get_joint_pos(relative=True) ** 2
        joint_pos_kernels = self._reward_track(
            joint_pos_err_per_joint, self.cfg.sigma_joint_pos, self.cfg.tol_joint_pos
        )
        rew_joint_pos = abs_gate * self.cfg.w_joint_pos * joint_pos_kernels.mean(dim=-1)

        eef_box_rel_pos_err, eef_box_rel_quat_err = self._compute_eef_box_rel_errors()
        rew_eef_box_rel_pos = gate * self.cfg.w_eef_box_rel_pos * self._reward_track(
            eef_box_rel_pos_err ** 2, self.cfg.sigma_eef_box_rel_pos, self.cfg.tol_eef_box_rel_pos)
        rew_eef_box_rel_quat = gate * self.cfg.w_eef_box_rel_quat * self._reward_track(
            eef_box_rel_quat_err ** 2, self.cfg.sigma_eef_box_rel_quat, self.cfg.tol_eef_box_rel_quat)

        rew_track = w_track_eff * (rew_EE_pos + rew_EE_quat + rew_joint_pos + rew_eef_box_rel_pos + rew_eef_box_rel_quat)
        if self.cfg.track_scale_by_dphase:
            rew_track = self.dphase * rew_track

        joint_acc = (self._get_joint_vel() - self.prev_joint_vel) / self.dt
        joint_acc *= torch.abs(joint_acc) > self.cfg.tol_joint_acc
        joint_acc_penalty = joint_acc.square().sum(dim=-1)
        rew_joint_acc = self.cfg.w_joint_acc * joint_acc_penalty

        torque = self.ur5e.data.applied_torque.clone()
        torque *= torch.abs(torque) > self.cfg.tol_joint_torque
        torque_penalty = torque.square().sum(dim=-1)
        rew_torque = self.cfg.w_joint_torque * torque_penalty

        eps = float(self.cfg.action_alpha_floor)
        alpha_scale = alpha + eps * (1.0 - alpha)
        rate_reg_scale = alpha_scale if self.cfg.action_mode == "D" else 1.0
        norm_reg_scale = alpha_scale

        action_rate_error = (self.actions[:, :6] - self.prev_actions)
        action_rate_error *= torch.abs(action_rate_error) > self.cfg.tol_action_rate
        action_rate_penalty = action_rate_error.square().sum(dim=-1)
        rew_action_rate = rate_reg_scale * self.cfg.w_action_rate * action_rate_penalty

        action_norm_error = self.actions[:, :6].clone()
        action_norm_error *= torch.abs(action_norm_error) > self.cfg.tol_action_norm
        action_norm_penalty = action_norm_error.square().sum(dim=-1)
        rew_action_norm = norm_reg_scale * self.cfg.w_action_norm * action_norm_penalty

        if self.cfg.enable_phase_slowdown:
            slowdown_step = (1.0 - self.dphase).clamp(min=0.0)
            self.cumulative_slowdown += slowdown_step  # kept for logging + optional (b)

            # r_task_norm ∈ [0,1] for the gate.
            if self.cfg.task_reward_form == "product":
                r_task_norm = rew_task_unweighted.clamp(0.0, 1.0)
            else:
                task_w_sum = (self.cfg.w_obj_pos + self.cfg.w_obj_quat
                              + self.cfg.w_obj_lin_vel + self.cfg.w_obj_ang_vel)
                r_task_norm = (rew_task_unweighted / max(task_w_sum, 1e-6)).clamp(0.0, 1.0)

            # r_track_norm ∈ [0,1]: phase-exclusive (abs gate=0 OR rel gate=1), so
            # divide by the active weight sum.
            abs_w = self.cfg.w_eef_pos + self.cfg.w_eef_quat + self.cfg.w_joint_pos
            rel_w = self.cfg.w_eef_box_rel_pos + self.cfg.w_eef_box_rel_quat
            track_w_active = (abs_gate * abs_w + gate * rel_w).clamp(min=1e-6)
            rew_track_components = (rew_EE_pos + rew_EE_quat + rew_joint_pos
                                    + rew_eef_box_rel_pos + rew_eef_box_rel_quat)
            r_track_norm = (rew_track_components / track_w_active).clamp(0.0, 1.0)

            slowdown_health = torch.minimum(r_task_norm, r_track_norm)
            rew_slowdown_gated = self.cfg.w_slowdown_gated * slowdown_step * slowdown_health

            err_task = (obj_pos_error  / self.cfg.sigma_obj_pos
                        + obj_quat_error / self.cfg.sigma_obj_quat)
            err_track_abs = (EE_pos_error   / self.cfg.sigma_eef_pos
                             + eef_quat_error / self.cfg.sigma_eef_quat)
            err_track_rel = (eef_box_rel_pos_err  / self.cfg.sigma_eef_box_rel_pos
                             + eef_box_rel_quat_err / self.cfg.sigma_eef_box_rel_quat)
            err_track = abs_gate * err_track_abs + gate * err_track_rel

            delta_err_task  = (self._err_task_prev  - err_task ).clamp(min=0.0)
            delta_err_track = (self._err_track_prev - err_track).clamp(min=0.0)
            slowdown_improvement = delta_err_task + delta_err_track
            improvement_factor = torch.exp(-self.cfg.slowdown_improvement_beta * slowdown_improvement)
            rew_slowdown_improvement = (self.cfg.w_slowdown_improvement
                                        * slowdown_step * improvement_factor)

            self._err_task_prev = err_task.detach()
            self._err_track_prev = err_track.detach()

            rew_total_slowdown = self.cfg.w_total_slowdown * self.cumulative_slowdown * slowdown_step
        else:
            rew_slowdown_gated = torch.zeros(self.num_envs, device=self.device)
            rew_slowdown_improvement = torch.zeros(self.num_envs, device=self.device)
            rew_total_slowdown = torch.zeros(self.num_envs, device=self.device)
            r_task_norm = torch.zeros(self.num_envs, device=self.device)
            r_track_norm = torch.zeros(self.num_envs, device=self.device)
            slowdown_health = torch.zeros(self.num_envs, device=self.device)
            slowdown_improvement = torch.zeros(self.num_envs, device=self.device)
            improvement_factor = torch.ones(self.num_envs, device=self.device)

        q_pos = self._get_joint_pos()
        q_limits = self.ur5e.data.joint_pos_limits
        q_min = q_limits[..., 0] + self.cfg.joint_limit_eps
        q_max = q_limits[..., 1] - self.cfg.joint_limit_eps
        limit_violation = torch.clamp(q_min - q_pos, min=0.0) + torch.clamp(q_pos - q_max, min=0.0)
        rew_joint_limit = self.cfg.w_joint_limit * limit_violation.square().sum(dim=-1)

        total_illegal_force = torch.zeros((self.num_envs,), device=self.device)
        for sensor in self.illegal_contact_sensors.values():
            f_abs = sensor.data.force_matrix_w.norm(dim=-1)
            f_abs_clamped = f_abs.clamp(max=self.cfg.max_contact_force)
            total_illegal_force += f_abs_clamped.sum(dim=-1).flatten()

        rew_illegal_contact = self.cfg.w_illegal_contact * total_illegal_force

        rew_proximity = self._compute_proximity_penalty()

        flange_to_forearm_dist = self._get_flange_to_forearm_distance(self.ur5e)
        is_too_close = (flange_to_forearm_dist < self.cfg.max_flange_forearm_distance).float()
        rew_flange_forearm_dist = self.cfg.w_flange_forearm_dist * is_too_close

        rew_regularization = self.cfg.w_regularization * (
            rew_joint_acc + rew_torque + rew_action_rate + rew_action_norm + rew_joint_limit + rew_illegal_contact + rew_proximity + rew_flange_forearm_dist + rew_total_slowdown + rew_slowdown_gated + rew_slowdown_improvement
        )

        rew_completion = self.cfg.w_completion * self.reset_time_outs.float()

        self.extras["log"] = {
            "Curriculum/alpha": torch.tensor(alpha, device=self.device),
            "Curriculum/rate_reg_scale": torch.tensor(rate_reg_scale, device=self.device),
            "Curriculum/norm_reg_scale": torch.tensor(norm_reg_scale, device=self.device),
            "Rewards_task/obj_pos": rew_obj_pos.mean(),
            "Rewards_task/obj_quat": rew_obj_quat.mean(),
            "Rewards_task/obj_lin_vel": rew_obj_lin_vel.mean(),
            "Rewards_task/obj_ang_vel": rew_obj_ang_vel.mean(),
            "Rewards_track/eef_pos": rew_EE_pos.mean(),
            "Rewards_track/eef_quat": rew_EE_quat.mean(),
            "Rewards_track/joint_pos": rew_joint_pos.mean(),
            "Rewards_task/total": rew_task.mean(),
            "Rewards_track/total": rew_track.mean(),
            "Rewards/completion_bonus": rew_completion.mean(),
            "Rewards/completion_fraction": self.reset_time_outs.float().mean(),
            "Rewards/termination_fraction": self.reset_terminated.float().mean(),
            "Error/obj_pos_error": obj_pos_error.mean(),
            "Error/obj_quat_error": obj_quat_error.mean(),
            "Error/obj_lin_vel_error": obj_lin_vel_error.mean(),
            "Error/obj_ang_vel_error": obj_ang_vel_error.mean(),
            "Error/EE_pos_error": EE_pos_error.mean(),
            "Error/EE_quat_error": eef_quat_error.mean(),
            "Error/eef_box_rel_pos_active": (eef_box_rel_pos_err * gate).sum() / gate.sum().clamp(min=1),
            "Error/eef_box_rel_quat_active": (eef_box_rel_quat_err * gate).sum() / gate.sum().clamp(min=1),
            "Error/eef_box_rel_pos_raw": eef_box_rel_pos_err.mean(),
            "Error/eef_box_rel_quat_raw": eef_box_rel_quat_err.mean(),
            "Rewards_regularization/total": rew_regularization.mean(),
            "Rewards_regularization/joint_acceleration": rew_joint_acc.mean(),
            "Rewards_regularization/torque": rew_torque.mean(),
            "Rewards_regularization/action_rate": rew_action_rate.mean(),
            "Rewards_regularization/action_norm": rew_action_norm.mean(),
            "Rewards_regularization/joint_limit": rew_joint_limit.mean(),
            "Rewards_regularization/illegal_contact": rew_illegal_contact.mean(),
            "Rewards_regularization/illegal_proximity": rew_proximity.mean(),
            "Rewards_regularization/flange_forearm_distance": rew_flange_forearm_dist.mean(),
            "Rewards_regularization/total_slowdown": rew_total_slowdown.mean(),
            "Rewards_regularization/slowdown_gated": rew_slowdown_gated.mean(),
            "Rewards_regularization/slowdown_improvement": rew_slowdown_improvement.mean(),
            "Rewards_track/eef_box_rel_pos": rew_eef_box_rel_pos.mean(),
            "Rewards_track/eef_box_rel_quat": rew_eef_box_rel_quat.mean(),
            "Rewards_track/eef_box_gate_frac": gate.mean(),
            "Phase/mean_phase_norm": (self.phase / max(self.obj_poses.shape[0] - 1, 1)).mean(),
            "Phase/mean_dphase": self.dphase.mean(),
            "Phase/min_dphase": self.dphase.min(),
            "Phase/frac_slowed": (self.dphase < 0.9).float().mean(),
            "Phase/frac_deep_paused": (self.dphase < 0.2).float().mean(),
            "Phase/mean_cumulative_slowdown": self.cumulative_slowdown.mean(),
            "Phase/max_cumulative_slowdown": self.cumulative_slowdown.max(),
            "Phase/r_task_norm": r_task_norm.mean(),
            "Phase/r_track_norm": r_track_norm.mean(),
            "Phase/slowdown_health": slowdown_health.mean(),
            "Phase/slowdown_improvement": slowdown_improvement.mean(),
            "Phase/slowdown_improvement_factor": improvement_factor.mean(),
        }

        if getattr(self.cfg, "emit_per_env_extras", False):
            self.extras["log_per_env"] = {
                "Rewards_task/obj_pos": rew_obj_pos,
                "Rewards_task/obj_quat": rew_obj_quat,
                "Rewards_task/obj_lin_vel": rew_obj_lin_vel,
                "Rewards_task/obj_ang_vel": rew_obj_ang_vel,
                "Rewards_task/total": rew_task,
                "Rewards_track/eef_pos": rew_EE_pos,
                "Rewards_track/eef_quat": rew_EE_quat,
                "Rewards_track/joint_pos": rew_joint_pos,
                "Rewards_track/eef_box_rel_pos": rew_eef_box_rel_pos,
                "Rewards_track/eef_box_rel_quat": rew_eef_box_rel_quat,
                "Rewards_track/total": rew_track,
                "Rewards/completion_bonus": rew_completion,
                "Rewards_regularization/total": rew_regularization,
                "Rewards_regularization/joint_acceleration": rew_joint_acc,
                "Rewards_regularization/torque": rew_torque,
                "Rewards_regularization/action_rate": rew_action_rate,
                "Rewards_regularization/action_norm": rew_action_norm,
                "Rewards_regularization/joint_limit": rew_joint_limit,
                "Rewards_regularization/illegal_contact": rew_illegal_contact,
                "Rewards_regularization/illegal_proximity": rew_proximity,
                "Rewards_regularization/flange_forearm_distance": rew_flange_forearm_dist,
                "Rewards_regularization/total_slowdown": rew_total_slowdown,
                "Rewards_regularization/slowdown_gated": rew_slowdown_gated,
                "Rewards_regularization/slowdown_improvement": rew_slowdown_improvement,
                "Error/obj_pos_error": obj_pos_error,
                "Error/obj_quat_error": obj_quat_error,
                # obj_lin_vel_error / obj_ang_vel_error are only defined when
                # cfg.task_reward_form != "sum"; omit to keep this block branch-safe
                # (and matches boxhinge's existing extras["log"] which also omits them).
                "Error/EE_pos_error": EE_pos_error,
                "Error/EE_quat_error": eef_quat_error,
                "Error/eef_box_rel_pos": eef_box_rel_pos_err,
                "Error/eef_box_rel_quat": eef_box_rel_quat_err,
            }

        if self.cfg.enable_failure_resampling:
            probs = self.segment_scores ** self.cfg.phase_resample_beta
            probs = probs / probs.sum()
            entropy = -(probs * (probs + 1e-8).log()).sum()
            self.extras["log"]["PhaseResample/mean_failure_rate"] = self.segment_scores.mean()
            self.extras["log"]["PhaseResample/max_failure_rate"] = self.segment_scores.max()
            self.extras["log"]["PhaseResample/min_failure_rate"] = self.segment_scores.min()
            self.extras["log"]["PhaseResample/sample_entropy"] = entropy

        total_reward = rew_task + rew_track + rew_completion - rew_regularization

        rew_track_unweighted_per_step = (
            rew_EE_pos + rew_EE_quat + rew_joint_pos + rew_eef_box_rel_pos + rew_eef_box_rel_quat
        )
        self._voc_ep_rew_task += rew_task_unweighted
        self._voc_ep_rew_track += rew_track_unweighted_per_step
        self._voc_ep_steps += 1
        self._voc_decay_step_counter += 1
        if (self.cfg.voc_enabled
                and self.voc_kp_pos > 0.0
                and self._voc_decay_step_counter >= self.cfg.voc_decay_check_interval):
            self._voc_decay_step_counter = 0
            self._voc_decay_check()

        self.extras["log"]["VOC/kp_pos"] = torch.tensor(self.voc_kp_pos, device=self.device)
        self.extras["log"]["VOC/kp_rot"] = torch.tensor(self.voc_kp_rot, device=self.device)
        self.extras["log"]["VOC/kv_pos"] = torch.tensor(self.voc_kv_pos, device=self.device)
        self.extras["log"]["VOC/kv_rot"] = torch.tensor(self.voc_kv_rot, device=self.device)
        valid_t = self._voc_buf_task[~torch.isnan(self._voc_buf_task)]
        valid_k = self._voc_buf_track[~torch.isnan(self._voc_buf_track)]
        self.extras["log"]["VOC/recent_task_mean"] = (
            valid_t.mean() if valid_t.numel() else torch.tensor(float("nan"), device=self.device)
        )
        self.extras["log"]["VOC/recent_track_mean"] = (
            valid_k.mean() if valid_k.numel() else torch.tensor(float("nan"), device=self.device)
        )

        self.prev_actions[:] = self.actions[:, :6]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

    def _voc_decay_check(self):
        """Decay VOC gains when trailing reward means exceed thresholds (DexMachina Algorithm 1)."""
        if self.common_step_counter < self.cfg.voc_decay_warmup_steps:
            return
        valid_task = self._voc_buf_task[~torch.isnan(self._voc_buf_task)]
        valid_track = self._voc_buf_track[~torch.isnan(self._voc_buf_track)]
        min_samples = self.cfg.voc_reward_window_size // 2
        if valid_task.numel() < min_samples or valid_track.numel() < min_samples:
            return
        if (valid_task.mean() < self.cfg.voc_threshold_task or
                valid_track.mean() < self.cfg.voc_threshold_track):
            return
        self.voc_kp_pos *= self.cfg.voc_decay_phi_p
        self.voc_kp_rot *= self.cfg.voc_decay_phi_p
        self.voc_kv_pos *= self.cfg.voc_decay_phi_v
        self.voc_kv_rot *= self.cfg.voc_decay_phi_v
        if self.voc_kp_pos < self.cfg.voc_kp_min:
            self.voc_kp_pos = 0.0
            self.voc_kp_rot = 0.0
            self.voc_kv_pos = 0.0
            self.voc_kv_rot = 0.0
        self._save_voc_state()

    def _save_voc_state(self):
        """Persist current VOC gains to <log_dir>/voc_state.npz for eval resumption. Best-effort."""
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir:
            return
        try:
            np.savez(
                os.path.join(log_dir, "voc_state.npz"),
                voc_kp_pos=self.voc_kp_pos,
                voc_kp_rot=self.voc_kp_rot,
                voc_kv_pos=self.voc_kv_pos,
                voc_kv_rot=self.voc_kv_rot,
            )
        except OSError:
            pass

    def _get_joint_pos(self, relative=False):
        joint_pos = self.ur5e.data.joint_pos.clone()
        if relative:
            joint_pos -= self._interp(self.joints)
        return joint_pos

    def _get_joint_vel(self, relative=False):
        joint_vel = self.ur5e.data.joint_vel.clone()
        if relative:
            joint_vel -= self._interp(self.joint_vel)
        return joint_vel

    def _get_forearm_endpoints(self, robot: Articulation):
        forearm_length = 0.4225
        p2_local = torch.tensor([-forearm_length, 0.0, 0.0], device=self.device)

        forearm_pos = robot.data.body_pos_w[:, self.forearm_link_idx]
        forearm_quat = robot.data.body_quat_w[:, self.forearm_link_idx]

        p2 = forearm_pos + quat_apply(forearm_quat, p2_local.repeat(self.num_envs, 1))
        return forearm_pos, p2

    def _get_closest_point_on_forearm(self, robot: Articulation):
        flange_pos = robot.data.body_pos_w[:, self.flange_idx]
        p1, p2 = self._get_forearm_endpoints(robot)

        line_vec = p2 - p1
        p1_to_flange = flange_pos - p1

        line_len_sq = torch.sum(line_vec**2, dim=-1)
        t = torch.sum(p1_to_flange * line_vec, dim=-1) / line_len_sq
        t = torch.clamp(t, 0.0, 1.0)

        closest_point = p1 + t.unsqueeze(-1) * line_vec
        return closest_point

    def _get_flange_to_forearm_distance(self, robot: Articulation):
        flange_pos = robot.data.body_pos_w[:, self.flange_idx]
        p1, p2 = self._get_forearm_endpoints(robot)

        line_vec = p2 - p1
        p1_to_flange = flange_pos - p1

        line_len_sq = torch.sum(line_vec**2, dim=-1)
        t = torch.sum(p1_to_flange * line_vec, dim=-1) / line_len_sq
        t = torch.clamp(t, 0.0, 1.0)

        closest_point = p1 + t.unsqueeze(-1) * line_vec
        distance = torch.norm(flange_pos - closest_point, dim=-1)
        return distance

    def _compute_eef_box_rel_errors(self):
        """EE pose error in the box's frame. Returns (pos_err [m], quat_err [rad]), shape (num_envs,)."""
        EE_pos_ref   = self._interp(self.EE_poses[:, :3])
        EE_quat_ref  = self._nlerp(self.EE_poses[:, 3:])
        obj_pos_ref  = self._interp(self.obj_poses[:, :3])
        obj_quat_ref = self._nlerp(self.obj_poses[:, 3:])

        inv_obj_quat_ref = quat_inv(obj_quat_ref)
        rel_ref_pos_box  = quat_apply(inv_obj_quat_ref, EE_pos_ref - obj_pos_ref)
        rel_ref_quat     = quat_mul(inv_obj_quat_ref, EE_quat_ref)

        EE_pos_actual   = self._get_EE_pos(relative=False)
        EE_quat_actual  = self._get_EE_quat(relative=False)
        obj_pos_actual  = self._get_obj_pos(relative=False)
        obj_quat_actual = self._get_obj_quat(relative=False)

        inv_obj_quat_actual = quat_inv(obj_quat_actual)
        rel_act_pos_box  = quat_apply(inv_obj_quat_actual, EE_pos_actual - obj_pos_actual)
        rel_act_quat     = quat_mul(inv_obj_quat_actual, EE_quat_actual)

        pos_err  = (rel_act_pos_box - rel_ref_pos_box).norm(dim=-1)
        quat_err = torch.abs(quat_error_magnitude(rel_act_quat, rel_ref_quat))
        return pos_err, quat_err

    def _compute_proximity_penalty(self) -> torch.Tensor:
        penalty = torch.zeros(self.num_envs, device=self.device)
        for sensor in self.illegal_contact_sensors.values():
            _, _, _, separation, contact_count_per_link, _ = sensor.contact_physx_view.get_contact_data(self.dt)
            total_count = contact_count_per_link.sum().item()
            if total_count == 0:
                continue
            separation = separation[:total_count, 0]
            contact_count_per_env = contact_count_per_link.sum(dim=-1)
            env_ids = torch.repeat_interleave(
                torch.arange(self.num_envs, device=self.device), contact_count_per_env
            )
            min_sep = torch.full((self.num_envs,), self.cfg.max_proximity * 2, device=self.device)
            min_sep.index_reduce_(0, env_ids, separation, reduce='amin', include_self=True)
            proximity = torch.clamp(1.0 - min_sep / self.cfg.max_proximity, min=0.0)
            penalty += proximity.square()
        return self.cfg.w_proximity_to_contact * penalty

    def _get_EE_pos(self, relative=True) -> torch.Tensor:
        EE_pos = self.ur5e.data.body_pos_w[:, self.EE_link_idx].clone() - self.scene.env_origins
        if relative:
            EE_pos -= self._interp(self.EE_poses[:, :3])
        return EE_pos

    def _get_EE_pos_error(self):
        return torch.norm(self._get_EE_pos(relative=True), dim=-1)

    def _get_EE_quat(self, relative=True) -> torch.Tensor:
        EE_quat = self.ur5e.data.body_quat_w[:, self.EE_link_idx].clone()
        if relative:
            desired_quat = self._nlerp(self.EE_poses[:, 3:])
            EE_quat = quat_mul(desired_quat, quat_inv(EE_quat))
        return EE_quat

    def _get_EE_quat_error(self):
        EE_quat = self.ur5e.data.body_quat_w[:, self.EE_link_idx].clone()
        desired_quat = self._nlerp(self.EE_poses[:, 3:])
        return torch.abs(quat_error_magnitude(EE_quat, desired_quat))

    def _get_EE_vel(self) -> torch.Tensor:
        return self.ur5e.data.body_vel_w[:, self.EE_link_idx].clone()

    def _get_obj_pos(self, relative=True):
        obj_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins
        if relative:
            desired_obj_pos = self._interp(self.obj_poses[:, :3])
            obj_pos -= desired_obj_pos
        return obj_pos

    def _get_obj_pos_error(self):
        return torch.norm(self._get_obj_pos(relative=True), dim=-1)

    def _get_obj_quat(self, relative=True):
        obj_quat = self.object.data.root_quat_w.clone()
        if relative:
            desired_obj_quat = self._nlerp(self.obj_poses[:, 3:])
            obj_quat = quat_mul(desired_obj_quat, quat_inv(obj_quat))
        return obj_quat

    def _get_obj_quat_error(self):
        obj_quat = self.object.data.root_quat_w.clone()
        desired_obj_quat = self._nlerp(self.obj_poses[:, 3:])
        return torch.abs(quat_error_magnitude(obj_quat, desired_obj_quat))

    def _get_obj_vel(self, relative=True):
        obj_vel = self.object.data.root_vel_w.clone()
        if relative:
            obj_vel -= self._interp(self.obj_vel)
        return obj_vel
