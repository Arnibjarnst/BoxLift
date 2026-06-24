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
        # cfg.observation_space is read in super().__init__, so __post_init__ must run first
        # to pick up Hydra CLI overrides (obs_history_steps etc. affect observation_space).
        cfg.__post_init__()
        super().__init__(cfg, render_mode, **kwargs)

        self.EE_link_idx = self.ur5e_r.body_names.index("wrist_3_link")
        self.flange_idx = self.ur5e_r.body_names.index("wrist_3_link")
        self.forearm_link_idx = self.ur5e_r.body_names.index("forearm_link")

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)

    def _setup_scene(self):
        traj = np.load(self.cfg.trajectory_path)

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

        if "object_dims" in traj:
            self.cfg.object_dims = tuple(traj["object_dims"].tolist())
            self.cfg.cube_cfg.spawn.size = self.cfg.object_dims
        if "object_mass" in traj:
            self.cfg.object_mass = float(traj["object_mass"])
            self.cfg.cube_cfg.spawn.mass_props.mass = self.cfg.object_mass

        # TODO: Support last trajectory point
        self.cfg.episode_length_s = self.dt * (self.obj_poses.shape[0] - 1)

        ur5e_l_cfg = get_ur5e_cfg(self.cfg.ur5e_l_prim_path, self.arm_l_pose, self.cfg)
        ur5e_r_cfg = get_ur5e_cfg(self.cfg.ur5e_r_prim_path, self.arm_r_pose, self.cfg)

        self.ur5e_l = Articulation(ur5e_l_cfg)
        self.ur5e_r = Articulation(ur5e_r_cfg)
        
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)

        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}

        self.ee_contact_sensors = [ContactSensor(cfg) for cfg in self.cfg.ee_contact_sensors]

        obj_vel_mag = self.obj_vel[:, :3].norm(dim=-1) + self.obj_vel[:, 3:].norm(dim=-1)
        moving = (obj_vel_mag > self.cfg.eef_box_gate_obj_vel_eps).float()
        if self.cfg.eef_box_gate_dilation_steps > 0:
            k = 2 * int(self.cfg.eef_box_gate_dilation_steps) + 1
            moving = torch.nn.functional.max_pool1d(
                moving.view(1, 1, -1), kernel_size=k, stride=1,
                padding=int(self.cfg.eef_box_gate_dilation_steps),
            ).view(-1)
        self.eef_box_gate_mask = moving.bool()  # (T,)

        self.prev_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 12), device=self.device)

        _pos_std = self.cfg.robot_pos_randomization_xyz_std
        _yaw_std = self.cfg.robot_ori_randomization_yaw_std
        self.robot_base_offset_l = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_base_offset_r = torch.zeros(self.num_envs, 3, device=self.device)
        self.robot_yaw_quat_l = torch.zeros(self.num_envs, 4, device=self.device)
        self.robot_yaw_quat_l[:, 0] = 1.0
        self.robot_yaw_quat_r = torch.zeros(self.num_envs, 4, device=self.device)
        self.robot_yaw_quat_r[:, 0] = 1.0
        if _pos_std > 0.0:
            self.robot_base_offset_l = _pos_std * torch.randn(self.num_envs, 3, device=self.device)
            self.robot_base_offset_r = _pos_std * torch.randn(self.num_envs, 3, device=self.device)
        if _yaw_std > 0.0:
            _yaw_l = _yaw_std * torch.randn(self.num_envs, device=self.device)
            _yaw_r = _yaw_std * torch.randn(self.num_envs, device=self.device)
            _zeros = torch.zeros(self.num_envs, device=self.device)
            self.robot_yaw_quat_l = torch.stack(
                [torch.cos(0.5 * _yaw_l), _zeros, _zeros, torch.sin(0.5 * _yaw_l)], dim=-1
            )
            self.robot_yaw_quat_r = torch.stack(
                [torch.cos(0.5 * _yaw_r), _zeros, _zeros, torch.sin(0.5 * _yaw_r)], dim=-1
            )

        # newest-last; [:, 0] is the delayed read; delay_steps=0 → length 1
        self.ee_contact_delay_buf = torch.zeros(
            (self.num_envs, self.cfg.contact_obs_delay_steps + 1, 2),
            device=self.device,
        )

        self.obs_history = torch.zeros(
            (self.num_envs, self.cfg.obs_history_steps, self.cfg.per_step_feature_dim),
            device=self.device,
        )

        # Simulated cube tracker: newest-last delay ring + sub-rate update counter.
        # obj_pose_delay_buf[:, 0] holds the pose from obs_obj_delay_steps ago.
        # obj_obs_counter fires when it reaches obs_obj_update_period, latching a fresh
        # delayed sample into obj_obs_last_pose / obj_obs_last_rel.
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
        self.obj_obs_last_pose[:, 3] = 1.0  # identity quat (wxyz)
        self.obj_obs_last_rel = torch.zeros(self.num_envs, 7, device=self.device)
        self.obj_obs_last_rel[:, 3] = 1.0
        self.obj_obs_bias_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_obs_bias_ori_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.obj_obs_bias_ori_quat[:, 0] = 1.0

        # DR cache: read from PhysX at reset; joint stiff/damp are live in .data.
        # obj_friction is (static, dynamic, restitution)
        self._dr_obj_mass = torch.zeros((self.num_envs, 1), device=self.device)
        self._dr_obj_friction = torch.zeros((self.num_envs, 3), device=self.device)

        mass = float(self.cfg.cube_cfg.spawn.mass_props.mass)
        d_max = float(max(self.cfg.cube_cfg.spawn.size))
        inertia_est = mass * (d_max ** 2) / 12.0
        kp_pos_init = float(self.cfg.voc_kp_pos)
        kp_rot_init = float(self.cfg.voc_kp_rot)
        kv_pos_init = self.cfg.voc_kv_pos_scale * (kp_pos_init * mass) ** 0.5
        kv_rot_init = self.cfg.voc_kv_rot_scale * (kp_rot_init * inertia_est) ** 0.5

        T = self.obj_poses.shape[0]
        seg_mode = self.cfg.voc_segmentation
        if seg_mode == "none":
            self.phase_to_segment = torch.zeros(T, dtype=torch.long, device=self.device)
            self._voc_n_segments = 1
            self._voc_seg_boundaries = torch.tensor([0, T], device=self.device, dtype=torch.long)
        elif seg_mode == "uniform":
            N = int(self.cfg.voc_n_uniform_segments)
            bin_size = max(1, (T + N - 1) // N)
            self.phase_to_segment = (torch.arange(T, device=self.device) // bin_size).clamp(max=N - 1).long()
            self._voc_n_segments = N
            self._voc_seg_boundaries = torch.minimum(
                torch.arange(N + 1, device=self.device, dtype=torch.long) * bin_size,
                torch.tensor(T, device=self.device, dtype=torch.long),
            )
        elif seg_mode == "contact":
            obj_vel_mag = self.obj_vel[:, :3].norm(dim=-1) + self.obj_vel[:, 3:].norm(dim=-1)
            seg_moving = (obj_vel_mag > self.cfg.eef_box_gate_obj_vel_eps).float()
            dil = int(self.cfg.voc_segment_dilation_steps)
            if dil > 0:
                k = 2 * dil + 1
                seg_moving = torch.nn.functional.max_pool1d(
                    seg_moving.view(1, 1, -1), kernel_size=k, stride=1, padding=dil,
                ).view(-1)
            seg_gate = seg_moving.bool()  # (T,)

            transitions = (seg_gate[1:] != seg_gate[:-1]).nonzero(as_tuple=True)[0] + 1
            boundaries = torch.cat([
                torch.zeros(1, device=self.device, dtype=torch.long),
                transitions.to(torch.long),
                torch.tensor([T], device=self.device, dtype=torch.long),
            ])
            self._voc_n_segments = boundaries.numel() - 1
            self._voc_seg_boundaries = boundaries
            self.phase_to_segment = torch.zeros(T, dtype=torch.long, device=self.device)
            for s in range(self._voc_n_segments):
                self.phase_to_segment[boundaries[s]:boundaries[s + 1]] = s
            print(f"[VOC] contact segmentation → {self._voc_n_segments} segments, "
                  f"boundaries (frames) = {boundaries.tolist()}, "
                  f"gate@boundary = {seg_gate[boundaries[:-1]].tolist()}  "
                  f"(seg_dilation={dil}, eps={self.cfg.eef_box_gate_obj_vel_eps})")
        else:
            raise ValueError(f"Unknown voc_segmentation: {seg_mode!r} (expected 'none'|'uniform'|'contact')")

        N_s = self._voc_n_segments
        self._voc_kp_pos_seg = torch.full((N_s,), kp_pos_init, device=self.device)
        self._voc_kp_rot_seg = torch.full((N_s,), kp_rot_init, device=self.device)
        self._voc_kv_pos_seg = torch.full((N_s,), kv_pos_init, device=self.device)
        self._voc_kv_rot_seg = torch.full((N_s,), kv_rot_init, device=self.device)

        self._voc_ep_rew_task_seg  = torch.zeros((self.num_envs, N_s), device=self.device)
        self._voc_ep_rew_track_seg = torch.zeros((self.num_envs, N_s), device=self.device)
        self._voc_ep_steps_seg     = torch.zeros((self.num_envs, N_s), dtype=torch.long, device=self.device)

        W = self.cfg.voc_reward_window_size
        self._voc_buf_task_seg  = torch.full((N_s, W), float("nan"), device=self.device)
        self._voc_buf_track_seg = torch.full((N_s, W), float("nan"), device=self.device)
        self._voc_buf_idx_seg   = torch.zeros(N_s, dtype=torch.long, device=self.device)
        self._voc_decay_step_counter = 0

        self._alpha_value: float = 0.0
        W_a = self.cfg.alpha_reward_window_size
        self._alpha_buf_task   = torch.full((W_a,), float("nan"), device=self.device)
        self._alpha_buf_idx:  int = 0
        self._alpha_ep_rew_task = torch.zeros(self.num_envs, device=self.device)
        self._alpha_ep_steps    = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._alpha_check_counter: int = 0

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["ur5e_left"] = self.ur5e_l
        self.scene.articulations["ur5e_right"] = self.ur5e_r
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        for name, sensor in self.illegal_contact_sensors.items():
            self.scene.sensors[f"illegal_contact_sensor_{name}"] = sensor
        for i, sensor in enumerate(self.ee_contact_sensors):
            self.scene.sensors[f"ee_contact_sensor_{i}"] = sensor
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

        self.cube_marker.set_visibility(False)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

        self._apply_voc()

        EE_pos_l = self._robot_ee_ref_pos(self.EE_poses_l[self.episode_length_buf, :3], "l") + self.scene.env_origins
        EE_pos_r = self._robot_ee_ref_pos(self.EE_poses_r[self.episode_length_buf, :3], "r") + self.scene.env_origins

        ee_marker_pos = torch.stack([EE_pos_l, EE_pos_r], dim=1).view(-1, 3)
        self.ee_markers.visualize(translations=ee_marker_pos)

        obj_pos = self.obj_poses[self.episode_length_buf, :3] + self.scene.env_origins
        obj_quat = self.obj_poses[self.episode_length_buf, 3:]

        self.cube_marker.visualize(translations=obj_pos, orientations=obj_quat)

        # Cache once per policy step; modes C/D depend on q_current and must not recompute
        # each decimation substep (target would drift within the substep window).
        q_l, q_r = self.get_joint_targets()
        q_l = q_l.clamp(self.ur5e_l.data.joint_pos_limits[..., 0],
                        self.ur5e_l.data.joint_pos_limits[..., 1])
        q_r = q_r.clamp(self.ur5e_r.data.joint_pos_limits[..., 0],
                        self.ur5e_r.data.joint_pos_limits[..., 1])
        self._cached_joint_target = (q_l, q_r)

    def get_joint_targets_A(self):
        """Residual on planner targets."""
        q_l = self.joints_target_l[self.episode_length_buf] + self._scaled_action()[:, :6]
        q_r = self.joints_target_r[self.episode_length_buf] + self._scaled_action()[:, 6:]
        return q_l, q_r

    def get_joint_targets_B(self):
        """Residual on trajectory positions."""
        q_l = self.joints_l[self.episode_length_buf] + self._scaled_action()[:, :6]
        q_r = self.joints_r[self.episode_length_buf] + self._scaled_action()[:, 6:]
        return q_l, q_r

    def get_joint_targets_C(self):
        """Residual on current joint positions."""
        q_curr = self._get_joint_pos()
        q_l = q_curr[:, :6] + self._scaled_action()[:, :6]
        q_r = q_curr[:, 6:] + self._scaled_action()[:, 6:]
        return q_l, q_r

    def get_joint_targets_D(self):
        """Planner PD error from current position + scaled residual, blended by curriculum α."""
        alpha = self._curriculum_alpha()
        eps = float(self.cfg.action_alpha_floor)
        action_gain = alpha + eps * (1.0 - alpha)
        idx = self.episode_length_buf
        planner_pd_l = self.joints_target_l[idx] - self.joints_l[idx]
        planner_pd_r = self.joints_target_r[idx] - self.joints_r[idx]
        q_curr = self._get_joint_pos()
        scaled = self._scaled_action()
        q_l = q_curr[:, :6] + (1.0 - alpha) * planner_pd_l + action_gain * scaled[:, :6]
        q_r = q_curr[:, 6:] + (1.0 - alpha) * planner_pd_r + action_gain * scaled[:, 6:]
        return q_l, q_r

    def get_joint_targets_BC(self):
        """Linear blend between mode B (ref position) and mode C (current position) via curriculum α."""
        alpha = self._curriculum_alpha()
        q_curr = self._get_joint_pos()
        q_ref_l = self.joints_l[self.episode_length_buf]
        q_ref_r = self.joints_r[self.episode_length_buf]
        scaled = self._scaled_action()
        q_l = (1.0 - alpha) * q_ref_l + alpha * q_curr[:, :6] + scaled[:, :6]
        q_r = (1.0 - alpha) * q_ref_r + alpha * q_curr[:, 6:] + scaled[:, 6:]
        return q_l, q_r

    def _curriculum_alpha(self) -> float:
        if 0.0 <= self.cfg.force_alpha <= 1.0:
            return float(self.cfg.force_alpha)
        if self.cfg.alpha_curriculum_enabled:
            return self._alpha_value
        if self.cfg.alpha_warmup_steps > 0:
            return min(1.0, self.common_step_counter / self.cfg.alpha_warmup_steps)
        return 1.0

    def _scaled_action(self) -> torch.Tensor:
        return self._action_scale * self.actions

    def get_joint_targets(self):
        mode = self.cfg.action_mode
        if mode == "A":
            return self.get_joint_targets_A()
        if mode == "B":
            return self.get_joint_targets_B()
        if mode == "C":
            return self.get_joint_targets_C()
        if mode == "BC":
            return self.get_joint_targets_BC()
        if mode == "D":
            return self.get_joint_targets_D()
        raise ValueError(f"Unknown action_mode: {mode!r}")

    def _apply_action(self) -> None:
        q_l, q_r = self._cached_joint_target
        self.ur5e_l.set_joint_position_target(q_l)
        self.ur5e_r.set_joint_position_target(q_r)

    def _apply_voc(self):
        """PD wrench on cube toward reference: F=kp·pos_err-kv·vel_err, T=kp·rot_err-kv·ang_vel_err.
        rot_err = 2·sign(w)·xyz from quat_mul(ref, inv(obj)). (DexMachina, Mandi et al. 2025)
        """
        if not self.cfg.voc_enabled or float(self._voc_kp_pos_seg.max().item()) <= 0.0:
            n = self.num_envs
            self.object.set_external_force_and_torque(
                torch.zeros(n, 1, 3, device=self.device),
                torch.zeros(n, 1, 3, device=self.device),
                is_global=True,
            )
            return

        idx = self.episode_length_buf
        ref_pos = self.obj_poses[idx, :3]                                # (N, 3) env-frame
        ref_quat = self.obj_poses[idx, 3:]                               # (N, 4) wxyz
        ref_vel = self.obj_vel[idx]                                      # (N, 6) lin+ang

        obj_pos = self.object.data.root_pos_w - self.scene.env_origins   # (N, 3) env-frame
        obj_quat = self.object.data.root_quat_w                          # (N, 4) wxyz
        obj_vel = self.object.data.root_vel_w                            # (N, 6)

        seg_idx = self.phase_to_segment[idx.clamp(max=self.phase_to_segment.shape[0] - 1)]
        kp_pos = self._voc_kp_pos_seg[seg_idx].unsqueeze(-1)
        kv_pos = self._voc_kv_pos_seg[seg_idx].unsqueeze(-1)
        kp_rot = self._voc_kp_rot_seg[seg_idx].unsqueeze(-1)
        kv_rot = self._voc_kv_rot_seg[seg_idx].unsqueeze(-1)

        pos_err = ref_pos - obj_pos
        lin_vel_err = obj_vel[:, :3] - ref_vel[:, :3]
        force = kp_pos * pos_err - kv_pos * lin_vel_err

        q_err = quat_mul(ref_quat, quat_inv(obj_quat))
        sign_w = torch.where(q_err[:, 0:1] >= 0,
                             torch.ones_like(q_err[:, 0:1]),
                             -torch.ones_like(q_err[:, 0:1]))
        rot_err = 2.0 * sign_w * q_err[:, 1:]
        ang_vel_err = obj_vel[:, 3:] - ref_vel[:, 3:]
        torque = kp_rot * rot_err - kv_rot * ang_vel_err

        self.object.set_external_force_and_torque(
            force.unsqueeze(1), torque.unsqueeze(1), is_global=True,
        )

    def _get_noisy_obj_obs(self):
        """Simulated cube-tracker: fixed delay + sub-rate update.
        Returns (rel_pos, rel_quat, abs_pos, abs_quat) from the same latched sample.
        """
        obj_pos_now  = self.object.data.root_pos_w.clone() - self.scene.env_origins
        obj_quat_now = self.object.data.root_quat_w.clone()

        pose_now = torch.cat([obj_pos_now, obj_quat_now], dim=-1)
        self.obj_pose_delay_buf = torch.roll(self.obj_pose_delay_buf, shifts=-1, dims=1)
        self.obj_pose_delay_buf[:, -1] = pose_now
        self.obj_phase_delay_buf = torch.roll(self.obj_phase_delay_buf, shifts=-1, dims=1)
        self.obj_phase_delay_buf[:, -1] = self.episode_length_buf.float()

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
            if self.cfg.obs_obj_pos_noise > 0.0:
                sampled_pos = sampled_pos + (
                    self.cfg.obs_obj_pos_noise * torch.randn(n_f, 3, device=self.device)
                )
            if self.cfg.obs_obj_ori_noise > 0.0:
                aa = self.cfg.obs_obj_ori_noise * torch.randn(n_f, 3, device=self.device)
                delta_quat = torch.cat([torch.ones(n_f, 1, device=self.device), 0.5 * aa], dim=-1)
                delta_quat = delta_quat / delta_quat.norm(dim=-1, keepdim=True)
                sampled_quat = quat_mul(delta_quat, sampled_quat)
            sampled_quat = sampled_quat / sampled_quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            past_idx = sampled_phase.long().clamp(max=self.obj_poses.shape[0] - 1)
            past_ref_pos  = self.obj_poses[past_idx, :3]
            past_ref_quat = self.obj_poses[past_idx, 3:]
            rel_pos_fire  = sampled_pos - past_ref_pos
            rel_quat_fire = quat_mul(past_ref_quat, quat_inv(sampled_quat))  # desired * inv(actual)
            self.obj_obs_last_pose[fires_idx, :3] = sampled_pos
            self.obj_obs_last_pose[fires_idx, 3:] = sampled_quat
            self.obj_obs_last_rel[fires_idx, :3]  = rel_pos_fire
            self.obj_obs_last_rel[fires_idx, 3:]  = rel_quat_fire
            self.obj_obs_counter[fires_idx] = 0

        return (
            self.obj_obs_last_rel[:, :3],
            self.obj_obs_last_rel[:, 3:],
            self.obj_obs_last_pose[:, :3],
            self.obj_obs_last_pose[:, 3:],
        )

    def _get_observations(self) -> dict:
        contact_bools = []
        for sensor in self.ee_contact_sensors:
            f_mag = sensor.data.force_matrix_w.norm(dim=-1)
            total_mag = f_mag.sum(dim=(-1, -2))
            contact_bools.append((total_mag > self.cfg.contact_threshold).float())
        in_contact = torch.stack(contact_bools, dim=-1)              # (N, 2) [L, R]
        self.ee_contact_delay_buf = torch.roll(self.ee_contact_delay_buf, shifts=-1, dims=1)
        self.ee_contact_delay_buf[:, -1] = in_contact
        delayed_contact = self.ee_contact_delay_buf[:, 0]            # (N, 2)
        if self.cfg.contact_obs_flip_prob > 0.0:
            flip_mask = torch.rand_like(delayed_contact) < self.cfg.contact_obs_flip_prob
            delayed_contact = torch.where(flip_mask, 1.0 - delayed_contact, delayed_contact)

        # Always computed: side-effects update the delay buffer regardless of obs mode.
        obj_rel_pos, obj_rel_quat, obj_abs_pos, obj_abs_quat = self._get_noisy_obj_obs()

        _q_l  = self.ur5e_l.data.joint_pos.clone()
        _q_r  = self.ur5e_r.data.joint_pos.clone()
        _qd_l = self.ur5e_l.data.joint_vel.clone()
        _qd_r = self.ur5e_r.data.joint_vel.clone()
        if self.cfg.obs_joint_pos_noise_std > 0.0:
            _q_l  += self.cfg.obs_joint_pos_noise_std * torch.randn_like(_q_l)
            _q_r  += self.cfg.obs_joint_pos_noise_std * torch.randn_like(_q_r)
        if self.cfg.obs_joint_vel_noise_std > 0.0:
            _qd_l += self.cfg.obs_joint_vel_noise_std * torch.randn_like(_qd_l)
            _qd_r += self.cfg.obs_joint_vel_noise_std * torch.randn_like(_qd_r)
        _eidx  = self.episode_length_buf
        _abs_q  = torch.cat([_q_l,  _q_r],  dim=1)
        _abs_qd = torch.cat([_qd_l, _qd_r], dim=1)
        _rel_q  = _abs_q  - torch.cat([self.joints_l[_eidx],    self.joints_r[_eidx]],    dim=1)
        _rel_qd = _abs_qd - torch.cat([self.joint_vel_l[_eidx], self.joint_vel_r[_eidx]], dim=1)

        if self.cfg.use_reference_obs:
            current_features = torch.cat([
                _rel_q, _rel_qd, obj_rel_pos, obj_rel_quat,
                _abs_q, _abs_qd, obj_abs_pos, obj_abs_quat,
                delayed_contact,
            ], dim=-1)
        else:
            current_features = torch.cat([
                _abs_q, _abs_qd, obj_abs_pos, obj_abs_quat, delayed_contact,
            ], dim=-1)

        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = current_features

        T = self.obj_poses.shape[0]
        obs_parts = [self.obs_history.flatten(start_dim=1)]

        if self.cfg.use_reference_obs:
            phase_obs = (self.episode_length_buf.float() / max(T - 1, 1)).unsqueeze(-1)
            obs_parts.append(phase_obs)

        if self.cfg.future_obs_steps:
            idx_now = self.episode_length_buf.clamp(max=T - 1)
            cur_pos  = self.obj_poses[idx_now, :3]
            cur_quat = self.obj_poses[idx_now, 3:]
            inv_cur_quat = quat_inv(cur_quat)
            futures = []
            for k in self.cfg.future_obs_steps:
                fut_idx = (idx_now + int(k)).clamp(max=T - 1)
                fut_pos  = self.obj_poses[fut_idx, :3]
                fut_quat = self.obj_poses[fut_idx, 3:]
                futures.append(fut_pos - cur_pos)
                futures.append(quat_mul(fut_quat, inv_cur_quat))
                futures.append(fut_pos)
                futures.append(fut_quat)
            obs_parts.append(torch.cat(futures, dim=-1))

        obs_parts.append(self.prev_actions)

        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs, "privileged": self._get_privileged_obs()}

    def _get_privileged_obs(self) -> torch.Tensor:
        idx = self.episode_length_buf
        T = self.obj_poses.shape[0]
        idx_clamped = idx.clamp(max=T - 1)

        clean_obj_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins   # (N, 3)
        clean_obj_quat = self.object.data.root_quat_w.clone()                           # (N, 4)
        clean_obj_vel = self.object.data.root_vel_w.clone()                             # (N, 6) lin+ang
        clean_obj_block = torch.cat([clean_obj_pos, clean_obj_quat, clean_obj_vel], dim=-1)  # (N, 13)

        stiff_l = self.ur5e_l.data.joint_stiffness
        stiff_r = self.ur5e_r.data.joint_stiffness
        damp_l  = self.ur5e_l.data.joint_damping
        damp_r  = self.ur5e_r.data.joint_damping
        dr_block = torch.cat([
            self._dr_obj_mass, self._dr_obj_friction,
            stiff_l, stiff_r, damp_l, damp_r,
        ], dim=-1)

        ref_obj_pos = self.obj_poses[idx_clamped, :3]
        ref_obj_quat = self.obj_poses[idx_clamped, 3:]
        ref_obj_vel = self.obj_vel[idx_clamped]
        ref_joints_l = self.joints_l[idx_clamped]
        ref_joints_r = self.joints_r[idx_clamped]
        ref_joint_vels_l = self.joint_vel_l[idx_clamped]
        ref_joint_vels_r = self.joint_vel_r[idx_clamped]
        ref_joints_target_l = self.joints_target_l[idx_clamped]
        ref_joints_target_r = self.joints_target_r[idx_clamped]
        planner_pd_l = ref_joints_target_l - ref_joints_l
        planner_pd_r = ref_joints_target_r - ref_joints_r
        ref_EE_pose_l = torch.cat([
            self._robot_ee_ref_pos(self.EE_poses_l[idx_clamped, :3], "l"),
            self._robot_ee_ref_quat(self.EE_poses_l[idx_clamped, 3:], "l"),
        ], dim=-1)
        ref_EE_pose_r = torch.cat([
            self._robot_ee_ref_pos(self.EE_poses_r[idx_clamped, :3], "r"),
            self._robot_ee_ref_quat(self.EE_poses_r[idx_clamped, 3:], "r"),
        ], dim=-1)
        ref_block = torch.cat([
            ref_obj_pos, ref_obj_quat, ref_obj_vel,
            ref_joints_l, ref_joints_r,
            ref_joint_vels_l, ref_joint_vels_r,
            ref_joints_target_l, ref_joints_target_r,
            planner_pd_l, planner_pd_r,
            ref_EE_pose_l, ref_EE_pose_r,
        ], dim=-1)

        ee_force_vecs = []
        ee_force_mags = []
        for sensor in self.ee_contact_sensors:
            f = sensor.data.force_matrix_w.sum(dim=(1, 2))                              # (N, 3) net force vec
            mag = f.norm(dim=-1, keepdim=True)                                          # (N, 1)
            dir_ = f / mag.clamp(min=1e-6)                                              # (N, 3) unit (or 0)
            ee_force_mags.append(mag)
            ee_force_vecs.append(dir_)
        ee_force_mag_l, ee_force_mag_r = ee_force_mags[0], ee_force_mags[1]
        ee_force_dir_l, ee_force_dir_r = ee_force_vecs[0], ee_force_vecs[1]

        illegal_cube = torch.zeros((self.num_envs, 1), device=self.device)
        illegal_table = torch.zeros((self.num_envs, 1), device=self.device)
        if "cube" in self.illegal_contact_sensors:
            illegal_cube[:, 0] = (
                self.illegal_contact_sensors["cube"].data.force_matrix_w
                .norm(dim=-1).sum(dim=-1).flatten()
            )
        if "table" in self.illegal_contact_sensors:
            illegal_table[:, 0] = (
                self.illegal_contact_sensors["table"].data.force_matrix_w
                .norm(dim=-1).sum(dim=-1).flatten()
            )

        fl = self._get_flange_to_forearm_distance(self.ur5e_l).unsqueeze(-1)
        fr = self._get_flange_to_forearm_distance(self.ur5e_r).unsqueeze(-1)

        force_block = torch.cat([
            ee_force_mag_l, ee_force_mag_r,
            ee_force_dir_l, ee_force_dir_r,
            illegal_cube, illegal_table,
            fl, fr,
        ], dim=-1)

        pos_err_l, quat_err_l, pos_err_r, quat_err_r = self._compute_eef_box_rel_errors()
        eef_box_block = torch.stack([pos_err_l, quat_err_l, pos_err_r, quat_err_r], dim=-1)

        seg_idx_per_env = self.phase_to_segment[
            idx_clamped.clamp(max=self.phase_to_segment.shape[0] - 1)
        ]
        cur_kp_pos = self._voc_kp_pos_seg[seg_idx_per_env].unsqueeze(-1)
        cur_kp_rot = self._voc_kp_rot_seg[seg_idx_per_env].unsqueeze(-1)
        alpha_scalar = float(self._curriculum_alpha())
        alpha_t = torch.full((self.num_envs, 1), alpha_scalar, device=self.device)
        seg_t = seg_idx_per_env.float().unsqueeze(-1)
        voc_block = torch.cat([cur_kp_pos, cur_kp_rot, alpha_t, seg_t], dim=-1)

        return torch.cat([
            clean_obj_block, dr_block, ref_block, force_block, eef_box_block, voc_block
        ], dim=-1)                                                                       # (N, 136)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        obj_pos_error = self._get_obj_pos_error()
        obj_quat_error = self._get_obj_quat_error()

        self.reset_terminated = obj_pos_error > self.cfg.max_obj_dist_from_traj
        self.reset_terminated |= obj_quat_error > self.cfg.max_obj_angle_from_traj

        return self.reset_terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None, fixed_value: int = None): # type: ignore
        if env_ids is None:
            env_ids : Sequence[int] = self.scene._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)

        # TODO: verify this is the correct range
        if fixed_value is not None:
            self.episode_length_buf[env_ids] = fixed_value * torch.ones((len(env_ids),), device=self.device, dtype=torch.long)
        else:
            self.episode_length_buf[env_ids] = torch.randint(0, self.max_episode_length - 2, (len(env_ids),), device=self.device, dtype=torch.long)

        # Optional override: force a fraction of resets to start at phase=0. Pure RSI
        # samples phase=0 with probability ~1/T (often <1%); for multi-phase tasks the
        # early phase (lift) ends up severely under-trained. Only applies on the
        # non-fixed_value path so eval (which passes fixed_value) is untouched.
        if fixed_value is None and self.cfg.reset_to_zero_prob > 0:
            eids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            zero_mask = torch.rand(eids_t.numel(), device=self.device) < self.cfg.reset_to_zero_prob
            if zero_mask.any():
                self.episode_length_buf[eids_t[zero_mask]] = 0

        # Focused RSI: bias a fraction of (non-zero) resets toward segments still under
        # VOC assist (high kp). Pure `kp^beta` weighting — decayed segments get 0
        # focused exposure and rely on the (1 - focus_prob) uniform-RSI fraction below
        # for forgetting protection. If every segment has fully decayed, fall back to
        # uniform within the focused budget (degenerate but harmless). Skipped when
        # only one segment exists (no asymmetry to exploit).
        if (fixed_value is None
                and self.cfg.reset_segment_focus_prob > 0
                and self._voc_n_segments > 1):
            eids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            # Only refocus envs that weren't already snapped to phase=0 above.
            not_zero = self.episode_length_buf[eids_t] != 0
            focus_mask = (torch.rand(eids_t.numel(), device=self.device)
                          < self.cfg.reset_segment_focus_prob) & not_zero
            if focus_mask.any():
                w = self._voc_kp_pos_seg.clamp(min=0) ** float(self.cfg.segment_focus_beta)
                if w.sum().item() <= 0.0:
                    w = torch.ones_like(w)   # all decayed → uniform across segments
                probs = w / w.sum()
                n_focus = int(focus_mask.sum().item())
                seg_picks = torch.multinomial(probs, num_samples=n_focus, replacement=True)
                seg_start = self._voc_seg_boundaries[seg_picks]
                seg_end   = self._voc_seg_boundaries[seg_picks + 1]
                # Uniform within the picked segment; clamp so the episode has at least
                # a few steps of trajectory left before the wall-clock cap.
                rand_off = (torch.rand(n_focus, device=self.device) * (seg_end - seg_start).float()).long()
                new_phase = (seg_start + rand_off).clamp(max=self.max_episode_length - 2)
                self.episode_length_buf[eids_t[focus_mask]] = new_phase

        idx = self.episode_length_buf[env_ids]
        n = len(env_ids)
        pos_noise_std = torch.as_tensor(self.cfg.reset_joint_pos_noise, device=self.device, dtype=torch.float32)
        vel_noise_std = torch.as_tensor(self.cfg.reset_joint_vel_noise, device=self.device, dtype=torch.float32)

        initial_joint_pos_l = self.joints_l[idx].clone()
        initial_joint_vel_l = self.joint_vel_l[idx].clone()
        initial_joint_pos_l += pos_noise_std * torch.randn_like(initial_joint_pos_l)
        initial_joint_vel_l += vel_noise_std * torch.randn_like(initial_joint_vel_l)
        self.ur5e_l.write_joint_state_to_sim(initial_joint_pos_l, initial_joint_vel_l, env_ids=env_ids)

        initial_joint_pos_r = self.joints_r[idx].clone()
        initial_joint_vel_r = self.joint_vel_r[idx].clone()
        initial_joint_pos_r += pos_noise_std * torch.randn_like(initial_joint_pos_r)
        initial_joint_vel_r += vel_noise_std * torch.randn_like(initial_joint_vel_r)
        self.ur5e_r.write_joint_state_to_sim(initial_joint_pos_r, initial_joint_vel_r, env_ids=env_ids)

        # Teleport robot bases to their per-env offset positions. Needed every reset because
        # super()._reset_idx snaps articulations back to init_state (no offset).
        base_pos_l = (self.arm_l_pose[:3].unsqueeze(0).expand(n, -1)
                      + self.scene.env_origins[env_ids]
                      + self.robot_base_offset_l[env_ids])
        base_quat_l = quat_mul(self.robot_yaw_quat_l[env_ids],
                               self.arm_l_pose[3:7].unsqueeze(0).expand(n, -1))
        self.ur5e_l.write_root_pose_to_sim(
            torch.cat([base_pos_l, base_quat_l], dim=-1), env_ids=env_ids
        )
        base_pos_r = (self.arm_r_pose[:3].unsqueeze(0).expand(n, -1)
                      + self.scene.env_origins[env_ids]
                      + self.robot_base_offset_r[env_ids])
        base_quat_r = quat_mul(self.robot_yaw_quat_r[env_ids],
                               self.arm_r_pose[3:7].unsqueeze(0).expand(n, -1))
        self.ur5e_r.write_root_pose_to_sim(
            torch.cat([base_pos_r, base_quat_r], dim=-1), env_ids=env_ids
        )

        # Reset Object — base pose + per-axis noise (xy pos, yaw rot, xy lin_vel, z ang_vel).
        initial_object_pose = self.obj_poses[idx].clone()
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = self.obj_vel[idx].clone()

        initial_object_pose[:, 0:2] += self.cfg.reset_obj_pos_xy_noise * torch.randn(n, 2, device=self.device)
        # Yaw-only orientation noise so the box stays flat on the table.
        yaw = self.cfg.reset_obj_ori_noise * torch.randn(n, device=self.device)
        half = 0.5 * yaw
        zeros = torch.zeros_like(half)
        delta_quat = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        initial_object_pose[:, 3:7] = quat_mul(delta_quat, initial_object_pose[:, 3:7])
        initial_object_vel[:, 0:2] += self.cfg.reset_obj_lin_vel_xy_noise * torch.randn(n, 2, device=self.device)
        initial_object_vel[:, 3:6] += self.cfg.reset_obj_ang_vel_noise * torch.randn(n, 3, device=self.device)

        self.object.write_root_pose_to_sim(initial_object_pose, env_ids)
        self.object.write_root_velocity_to_sim(initial_object_vel, env_ids)

        # Must run BEFORE we zero the per-segment trackers below.
        W = self.cfg.voc_reward_window_size
        eids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for s in range(self._voc_n_segments):
            steps_s = self._voc_ep_steps_seg[eids_t, s]
            mask = steps_s > 0
            if not mask.any():
                continue
            n_s = int(mask.sum().item())
            norm_task_s  = self._voc_ep_rew_task_seg[eids_t[mask], s]  / steps_s[mask].float()
            norm_track_s = self._voc_ep_rew_track_seg[eids_t[mask], s] / steps_s[mask].float()
            slots = (self._voc_buf_idx_seg[s] + torch.arange(n_s, device=self.device)) % W
            self._voc_buf_task_seg[s, slots]  = norm_task_s
            self._voc_buf_track_seg[s, slots] = norm_track_s
            self._voc_buf_idx_seg[s] = (self._voc_buf_idx_seg[s] + n_s) % W
        self._voc_ep_rew_task_seg[eids_t]  = 0.0
        self._voc_ep_rew_track_seg[eids_t] = 0.0
        self._voc_ep_steps_seg[eids_t]     = 0

        steps_a = self._alpha_ep_steps[eids_t].clamp(min=1).float()
        norm_task_a = self._alpha_ep_rew_task[eids_t] / steps_a
        n_a = eids_t.numel()
        W_a = self.cfg.alpha_reward_window_size
        slots_a = (self._alpha_buf_idx + torch.arange(n_a, device=self.device)) % W_a
        self._alpha_buf_task[slots_a] = norm_task_a
        self._alpha_buf_idx = int((self._alpha_buf_idx + n_a) % W_a)
        self._alpha_ep_rew_task[eids_t] = 0.0
        self._alpha_ep_steps[eids_t]    = 0

        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids, :6] = initial_joint_vel_l
        self.prev_joint_vel[env_ids, 6:] = initial_joint_vel_r

        self.ee_contact_delay_buf[env_ids] = 0.0

        # Warm delay buffers with post-reset pose; counter=period triggers a fire on step 1.
        # Subtract env_origins: obj_poses is env-frame, but initial_object_pose has origins added.
        init_pose_env = initial_object_pose.clone()
        init_pose_env[:, :3] -= self.scene.env_origins[env_ids]
        self.obj_pose_delay_buf[env_ids]  = init_pose_env.unsqueeze(1)
        self.obj_phase_delay_buf[env_ids] = self.episode_length_buf[env_ids].float().unsqueeze(1)
        self.obj_obs_last_pose[env_ids]   = init_pose_env
        # rel ≈ 0 at reset (actual = ref + small reset noise; ref(start_phase) = trajectory[start_phase]).
        self.obj_obs_last_rel[env_ids, :3] = 0.0
        self.obj_obs_last_rel[env_ids, 3:] = 0.0
        self.obj_obs_last_rel[env_ids, 3]  = 1.0  # identity quat (wxyz)
        self.obj_obs_counter[env_ids] = self.cfg.obs_obj_update_period

        n_b = len(env_ids)
        if self.cfg.obs_obj_pos_bias_std > 0.0:
            self.obj_obs_bias_pos[env_ids] = (
                self.cfg.obs_obj_pos_bias_std * torch.randn(n_b, 3, device=self.device)
            )
        else:
            self.obj_obs_bias_pos[env_ids] = 0.0
        if self.cfg.obs_obj_ori_bias_std > 0.0:
            aa = self.cfg.obs_obj_ori_bias_std * torch.randn(n_b, 3, device=self.device)
            bias_quat = torch.cat([torch.ones(n_b, 1, device=self.device), 0.5 * aa], dim=-1)
            bias_quat = bias_quat / bias_quat.norm(dim=-1, keepdim=True)
            self.obj_obs_bias_ori_quat[env_ids] = bias_quat
        else:
            self.obj_obs_bias_ori_quat[env_ids, 0] = 1.0
            self.obj_obs_bias_ori_quat[env_ids, 1:] = 0.0

        self.obs_history[env_ids] = 0.0

        try:
            masses = self.object.root_physx_view.get_masses().to(self.device)
            self._dr_obj_mass[env_ids] = masses[env_ids]
            mat = self.object.root_physx_view.get_material_properties().to(self.device)
            self._dr_obj_friction[env_ids] = mat[env_ids, 0, :]
        except Exception as e:
            # Defensive: if a future IsaacLab version changes the PhysX view API,
            # log once and continue with stale (zeros / last good) cache values rather
            # than crash. The critic will see slightly stale DR info but training continues.
            if not getattr(self, "_dr_readback_warned", False):
                print(f"[boxlift] DR cache readback failed: {e!r}. Critic sees stale DR samples.")
                self._dr_readback_warned = True

    def _reward_track(self, error, sigma, tolerance=0.0):
        error *= error > tolerance
        reward = torch.exp(-error / (sigma ** 2))
        return reward

    def _get_rewards(self) -> torch.Tensor:
        obj_pos_error = self._get_obj_pos_error()
        rew_obj_pos = self.cfg.w_obj_pos * self._reward_track(obj_pos_error ** 2, self.cfg.sigma_obj_pos, self.cfg.tol_obj_pos)

        obj_quat_error = self._get_obj_quat_error()
        rew_obj_quat = self.cfg.w_obj_quat * self._reward_track(obj_quat_error ** 2, self.cfg.sigma_obj_quat, self.cfg.tol_obj_quat)

        rew_task_unweighted = rew_obj_pos + rew_obj_quat
        rew_task = self.cfg.w_task * rew_task_unweighted

        # Accumulate for BC alpha curriculum
        self._alpha_ep_rew_task += rew_task_unweighted
        self._alpha_ep_steps    += 1

        T = self.eef_box_gate_mask.shape[0]
        phase_idx = self.episode_length_buf.clamp(max=T - 1)
        gate = self.eef_box_gate_mask[phase_idx].float()
        abs_gate = 1.0 - gate

        # Averaged (not summed) across arms: keeps reward in [0, w_eef_pos] matching single-arm scale.
        EE_pos_error_l, EE_pos_error_r = self._get_EE_pos_error()
        rew_EE_pos_l = self._reward_track(EE_pos_error_l ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)
        rew_EE_pos_r = self._reward_track(EE_pos_error_r ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)
        rew_EE_pos = abs_gate * self.cfg.w_eef_pos * 0.5 * (rew_EE_pos_l + rew_EE_pos_r)

        eef_quat_error_l, eef_quat_error_r = self._get_EE_quat_error()
        rew_EE_quat_l = self._reward_track(eef_quat_error_l ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)
        rew_EE_quat_r = self._reward_track(eef_quat_error_r ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)
        rew_EE_quat = abs_gate * self.cfg.w_eef_quat * 0.5 * (rew_EE_quat_l + rew_EE_quat_r)

        # Per-joint kernel then mean: one bad joint can't hide behind the others.
        joint_pos_err_per_joint = self._get_joint_pos(relative=True) ** 2  # (N, 12)
        joint_pos_kernels = self._reward_track(
            joint_pos_err_per_joint, self.cfg.sigma_joint_pos, self.cfg.tol_joint_pos
        )  # (N, 12) — broadcasts over the trailing axis
        rew_joint_pos = abs_gate * self.cfg.w_joint_pos * joint_pos_kernels.mean(dim=-1)

        eef_box_rel_pos_err_l, eef_box_rel_quat_err_l, eef_box_rel_pos_err_r, eef_box_rel_quat_err_r = (
            self._compute_eef_box_rel_errors()
        )
        # L+R averaged (not summed) — same rationale as rew_EE_pos above.
        rew_eef_box_rel_pos = gate * self.cfg.w_eef_box_rel_pos * 0.5 * (
            self._reward_track(eef_box_rel_pos_err_l ** 2, self.cfg.sigma_eef_box_rel_pos, self.cfg.tol_eef_box_rel_pos)
            + self._reward_track(eef_box_rel_pos_err_r ** 2, self.cfg.sigma_eef_box_rel_pos, self.cfg.tol_eef_box_rel_pos)
        )
        rew_eef_box_rel_quat = gate * self.cfg.w_eef_box_rel_quat * 0.5 * (
            self._reward_track(eef_box_rel_quat_err_l ** 2, self.cfg.sigma_eef_box_rel_quat, self.cfg.tol_eef_box_rel_quat)
            + self._reward_track(eef_box_rel_quat_err_r ** 2, self.cfg.sigma_eef_box_rel_quat, self.cfg.tol_eef_box_rel_quat)
        )

        rew_track = self.cfg.w_track * (rew_EE_pos + rew_EE_quat + rew_joint_pos + rew_eef_box_rel_pos + rew_eef_box_rel_quat)

        joint_acc = (self._get_joint_vel() - self.prev_joint_vel) / self.dt
        joint_acc *= torch.abs(joint_acc) > self.cfg.tol_joint_acc
        joint_acc_penalty = joint_acc.square().sum(dim=-1)
        rew_joint_acc = self.cfg.w_joint_acc * joint_acc_penalty

        torque = torch.cat((self.ur5e_l.data.applied_torque.clone(), self.ur5e_r.data.applied_torque.clone()), 1)
        torque *= torch.abs(torque) > self.cfg.tol_joint_torque
        torque_penalty = torque.square().sum(dim=-1)
        rew_torque = self.cfg.w_joint_torque * torque_penalty

        # Mode D: scale rate/norm penalties by actual policy authority (α + ε(1-α)).
        if self.cfg.action_mode == "D":
            alpha = self._curriculum_alpha()
            eps = float(self.cfg.action_alpha_floor)
            action_authority_scale = alpha + eps * (1.0 - alpha)
        else:
            action_authority_scale = 1.0

        action_rate_error = (self.actions - self.prev_actions)
        action_rate_error *= torch.abs(action_rate_error) > self.cfg.tol_action_rate
        action_rate_penalty = action_rate_error.square().sum(dim=-1)
        rew_action_rate = action_authority_scale * self.cfg.w_action_rate * action_rate_penalty

        action_norm_error = self.actions.clone()
        action_norm_error *= torch.abs(action_norm_error) > self.cfg.tol_action_norm
        action_norm_penalty = action_norm_error.square().sum(dim=-1)
        rew_action_norm = action_authority_scale * self.cfg.w_action_norm * action_norm_penalty

        q_l, q_r = self._cached_joint_target
        q_limits_l = self.ur5e_l.data.joint_pos_limits
        q_limits_r = self.ur5e_r.data.joint_pos_limits
        eps = self.cfg.joint_limit_eps
        limit_violation_l = (
            torch.clamp((q_limits_l[..., 0] + eps) - q_l, min=0.0)
            + torch.clamp(q_l - (q_limits_l[..., 1] - eps), min=0.0)
        )
        limit_violation_r = (
            torch.clamp((q_limits_r[..., 0] + eps) - q_r, min=0.0)
            + torch.clamp(q_r - (q_limits_r[..., 1] - eps), min=0.0)
        )
        rew_joint_limit = self.cfg.w_joint_limit * (
            limit_violation_l.square().sum(dim=-1) + limit_violation_r.square().sum(dim=-1)
        )

        total_illegal_force = torch.zeros((self.num_envs,), device=self.device)
        for sensor in self.illegal_contact_sensors.values():
            f_abs = sensor.data.force_matrix_w.norm(dim=-1)
            f_abs_clamped = f_abs.clamp(max=self.cfg.max_contact_force)
            total_illegal_force += f_abs_clamped.sum(dim=-1).flatten()

        rew_illegal_contact = self.cfg.w_illegal_contact * total_illegal_force

        rew_proximity = self._compute_proximity_penalty()

        mean_ee_force = torch.zeros((self.num_envs,), device=self.device)
        for sensor in self.ee_contact_sensors:
            mean_ee_force += sensor.data.force_matrix_w.norm(dim=-1).sum(dim=-1).flatten() / len(self.ee_contact_sensors)

        flange_to_forearm_dist_l = self._get_flange_to_forearm_distance(self.ur5e_l)
        flange_to_forearm_dist_r = self._get_flange_to_forearm_distance(self.ur5e_r)

        is_too_close_l = (flange_to_forearm_dist_l < self.cfg.max_flange_forearm_distance).float()
        is_too_close_r = (flange_to_forearm_dist_r < self.cfg.max_flange_forearm_distance).float()

        rew_flange_forearm_dist = self.cfg.w_flange_forearm_dist * (is_too_close_l + is_too_close_r)

        rew_regularization = self.cfg.w_regularization * (
            rew_joint_acc + rew_torque + rew_action_rate + rew_action_norm
            + rew_joint_limit + rew_illegal_contact + rew_proximity + rew_flange_forearm_dist
        )

        per_env_log = {
            "Rewards_task/obj_pos": rew_obj_pos,
            "Rewards_task/obj_quat": rew_obj_quat,
            "Rewards_track/eef_pos": rew_EE_pos,
            "Rewards_track/eef_quat": rew_EE_quat,
            "Rewards_track/joint_pos": rew_joint_pos,
            "Rewards_track/eef_box_rel_pos": rew_eef_box_rel_pos,
            "Rewards_track/eef_box_rel_quat": rew_eef_box_rel_quat,
            "Rewards_task/total": rew_task,
            "Rewards_track/total": rew_track,
            "Error/obj_pos_error": obj_pos_error,
            "Error/obj_quat_error": obj_quat_error,
            "Error/EE_pos_error": (EE_pos_error_l + EE_pos_error_r) / 2.0,
            "Rewards_regularization/total": rew_regularization,
            "Rewards_regularization/joint_acceleration": rew_joint_acc,
            "Rewards_regularization/torque": rew_torque,
            "Rewards_regularization/action_rate": rew_action_rate,
            "Rewards_regularization/action_norm": rew_action_norm,
            "Rewards_regularization/joint_limit": rew_joint_limit,
            "Rewards_regularization/illegal_contact": rew_illegal_contact,
            "Rewards_regularization/illegal_proximity": rew_proximity,
            "Rewards_regularization/flange_forearm_distance": rew_flange_forearm_dist,
            "Extra/mean_EE_force": mean_ee_force,
        }
        self.extras["log"] = {k: v.mean() for k, v in per_env_log.items()}
        if getattr(self.cfg, "emit_per_env_extras", False):
            self.extras["log_per_env"] = per_env_log

        total_reward = rew_task + rew_track - rew_regularization

        rew_track_unweighted_per_step = (
            rew_EE_pos + rew_EE_quat + rew_joint_pos + rew_eef_box_rel_pos + rew_eef_box_rel_quat
        )
        env_arange = torch.arange(self.num_envs, device=self.device)
        seg_idx = self.phase_to_segment[
            self.episode_length_buf.clamp(max=self.phase_to_segment.shape[0] - 1)
        ]
        self._voc_ep_rew_task_seg[env_arange, seg_idx]  += rew_task_unweighted
        self._voc_ep_rew_track_seg[env_arange, seg_idx] += rew_track_unweighted_per_step
        self._voc_ep_steps_seg[env_arange, seg_idx]     += 1

        # Rate-limited; uses max() not mean() so we keep checking until the last segment decays.
        self._voc_decay_step_counter += 1
        if (self.cfg.voc_enabled
                and float(self._voc_kp_pos_seg.max().item()) > 0.0
                and self._voc_decay_step_counter >= self.cfg.voc_decay_check_interval):
            self._voc_decay_step_counter = 0
            self._voc_decay_check()

        # BC alpha curriculum: rate-limited check, only when VOC is fully off
        self._alpha_check_counter += 1
        if (self.cfg.alpha_curriculum_enabled
                and self._alpha_value < 1.0
                and float(self._voc_kp_pos_seg.max().item()) <= 0.0
                and self._alpha_check_counter >= self.cfg.alpha_decay_check_interval):
            self._alpha_check_counter = 0
            self._alpha_increase_check()

        self.extras["log"]["VOC/kp_pos_mean"] = self._voc_kp_pos_seg.mean()
        self.extras["log"]["VOC/kp_pos_min"]  = self._voc_kp_pos_seg.min()
        self.extras["log"]["VOC/kp_pos_max"]  = self._voc_kp_pos_seg.max()
        self.extras["log"]["VOC/kp_rot_mean"] = self._voc_kp_rot_seg.mean()
        self.extras["log"]["VOC/n_active_segments"] = (self._voc_kp_pos_seg > 0.0).sum().float()
        self.extras["log"]["Curriculum/alpha"] = torch.tensor(self._curriculum_alpha(), device=self.device)
        valid_a = ~torch.isnan(self._alpha_buf_task)
        self.extras["log"]["Curriculum/alpha_buf_mean"] = (
            self._alpha_buf_task[valid_a].mean() if valid_a.any() else torch.tensor(0.0, device=self.device)
        )

        valid_ts = ~torch.isnan(self._voc_buf_task_seg)
        valid_ks = ~torch.isnan(self._voc_buf_track_seg)
        cnt_ts = valid_ts.sum(dim=-1).clamp(min=1).float()
        cnt_ks = valid_ks.sum(dim=-1).clamp(min=1).float()
        sum_ts = torch.where(valid_ts, self._voc_buf_task_seg,  torch.zeros_like(self._voc_buf_task_seg)).sum(dim=-1)
        sum_ks = torch.where(valid_ks, self._voc_buf_track_seg, torch.zeros_like(self._voc_buf_track_seg)).sum(dim=-1)
        means_ts = sum_ts / cnt_ts
        means_ks = sum_ks / cnt_ks
        for s in range(self._voc_n_segments):
            self.extras["log"][f"VOC/seg{s}_kp_pos"]     = self._voc_kp_pos_seg[s]
            self.extras["log"][f"VOC/seg{s}_task_mean"]  = means_ts[s]
            self.extras["log"][f"VOC/seg{s}_track_mean"] = means_ks[s]

        if self.cfg.reset_segment_focus_prob > 0 and self._voc_n_segments > 1:
            w_log = self._voc_kp_pos_seg.clamp(min=0) ** float(self.cfg.segment_focus_beta)
            if w_log.sum().item() <= 0.0:
                w_log = torch.ones_like(w_log)
            probs_log = w_log / w_log.sum()
            for s in range(self._voc_n_segments):
                self.extras["log"][f"RSI/seg{s}_focus_prob"] = probs_log[s]

        self.prev_actions[:] = self.actions[:]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

    def _voc_decay_check(self):
        if self.common_step_counter < self.cfg.voc_decay_warmup_steps:
            return
        min_samples = self.cfg.voc_reward_window_size // 2
        phi_p = self.cfg.voc_decay_phi_p
        phi_v = self.cfg.voc_decay_phi_v
        kp_min = self.cfg.voc_kp_min

        valid_t = ~torch.isnan(self._voc_buf_task_seg)      # (N_s, W)
        valid_k = ~torch.isnan(self._voc_buf_track_seg)
        cnt_t = valid_t.sum(dim=-1)                          # (N_s,)
        cnt_k = valid_k.sum(dim=-1)
        sum_t = torch.where(valid_t, self._voc_buf_task_seg,  torch.zeros_like(self._voc_buf_task_seg)).sum(dim=-1)
        sum_k = torch.where(valid_k, self._voc_buf_track_seg, torch.zeros_like(self._voc_buf_track_seg)).sum(dim=-1)
        means_t = sum_t / cnt_t.float().clamp(min=1)
        means_k = sum_k / cnt_k.float().clamp(min=1)

        ready = (cnt_t >= min_samples) & (cnt_k >= min_samples)
        passing = ready & (means_t >= self.cfg.voc_threshold_task) & (means_k >= self.cfg.voc_threshold_track)
        if not bool(passing.any().item()):
            return

        kp_scale = torch.where(passing, torch.full_like(self._voc_kp_pos_seg, phi_p),
                               torch.ones_like(self._voc_kp_pos_seg))
        kv_scale = torch.where(passing, torch.full_like(self._voc_kv_pos_seg, phi_v),
                               torch.ones_like(self._voc_kv_pos_seg))
        self._voc_kp_pos_seg.mul_(kp_scale)
        self._voc_kp_rot_seg.mul_(kp_scale)
        self._voc_kv_pos_seg.mul_(kv_scale)
        self._voc_kv_rot_seg.mul_(kv_scale)
        # Snap below floor to zero; _apply_voc short-circuits per-env wrench when segment is zero.
        below = self._voc_kp_pos_seg < kp_min
        if bool(below.any().item()):
            self._voc_kp_pos_seg[below] = 0.0
            self._voc_kp_rot_seg[below] = 0.0
            self._voc_kv_pos_seg[below] = 0.0
            self._voc_kv_rot_seg[below] = 0.0
        self._save_voc_state()

    def _alpha_increase_check(self):
        """(1-alpha) *= phi when task reward exceeds threshold; snaps to 1 below alpha_min_support."""
        if self.common_step_counter < self.cfg.alpha_decay_warmup_steps:
            return
        valid = ~torch.isnan(self._alpha_buf_task)
        if int(valid.sum().item()) < self.cfg.alpha_reward_window_size // 2:
            return
        mean_task = float(self._alpha_buf_task[valid].mean().item())
        if mean_task < self.cfg.alpha_threshold_task:
            return
        support = (1.0 - self._alpha_value) * self.cfg.alpha_decay_phi
        self._alpha_value = 1.0 if support < self.cfg.alpha_min_support else float(1.0 - support)
        self._alpha_buf_task.fill_(float("nan"))
        self._alpha_buf_idx = 0
        print(f"[BC-alpha] alpha → {self._alpha_value:.4f}  (mean_task={mean_task:.3f})")

    def _save_voc_state(self):
        """Best-effort write of per-segment VOC gains to voc_state.npz for --keep_voc eval."""
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir:
            return
        try:
            np.savez(
                os.path.join(log_dir, "voc_state.npz"),
                voc_kp_pos=self._voc_kp_pos_seg.detach().cpu().numpy(),
                voc_kp_rot=self._voc_kp_rot_seg.detach().cpu().numpy(),
                voc_kv_pos=self._voc_kv_pos_seg.detach().cpu().numpy(),
                voc_kv_rot=self._voc_kv_rot_seg.detach().cpu().numpy(),
            )
        except OSError:
            pass

    def set_voc_strength(self, kp_pos: float) -> None:
        """Set uniform VOC kp_pos and derive kp_rot, kv_pos, kv_rot (critically damped). 0.0 disables."""
        mass = float(self.cfg.cube_cfg.spawn.mass_props.mass)
        d_max = float(max(self.cfg.cube_cfg.spawn.size))
        inertia_est = mass * (d_max ** 2) / 12.0
        kp_pos_init = float(self.cfg.voc_kp_pos)
        kp_rot_init = float(self.cfg.voc_kp_rot)
        scale = (kp_pos / kp_pos_init) if kp_pos_init > 0 else 0.0
        kp_rot = kp_rot_init * scale
        kv_pos = float(self.cfg.voc_kv_pos_scale) * max(kp_pos * mass, 0.0) ** 0.5
        kv_rot = float(self.cfg.voc_kv_rot_scale) * max(kp_rot * inertia_est, 0.0) ** 0.5
        self._voc_kp_pos_seg.fill_(float(kp_pos))
        self._voc_kp_rot_seg.fill_(float(kp_rot))
        self._voc_kv_pos_seg.fill_(float(kv_pos))
        self._voc_kv_rot_seg.fill_(float(kv_rot))
        print(f"[INFO] set_voc_strength: kp_pos={kp_pos:.3g}, kp_rot={kp_rot:.3g}, "
              f"kv_pos={kv_pos:.3g}, kv_rot={kv_rot:.3g} "
              f"(uniform across {self._voc_n_segments} segments)")

    def apply_voc_state(self, state) -> None:
        """Restore VOC gains from voc_state.npz. Accepts scalar (broadcast) or (N_s,) array."""
        for key, dst in (
            ("voc_kp_pos", self._voc_kp_pos_seg),
            ("voc_kp_rot", self._voc_kp_rot_seg),
            ("voc_kv_pos", self._voc_kv_pos_seg),
            ("voc_kv_rot", self._voc_kv_rot_seg),
        ):
            v = np.asarray(state[key])
            if v.ndim == 0:
                dst[:] = float(v)
            elif v.shape == (dst.numel(),):
                dst[:] = torch.as_tensor(v, device=self.device, dtype=dst.dtype)
            else:
                print(f"[WARN] apply_voc_state: '{key}' shape {v.shape} != ({dst.numel()},); "
                      f"falling back to mean broadcast.")
                dst[:] = float(v.mean())

    def _compute_eef_box_rel_errors(self):
        """EE error vs reference expressed in the box frame. Returns (pos_l, quat_l, pos_r, quat_r)."""
        idx = self.episode_length_buf
        obj_pos_ref  = self.obj_poses[idx, :3]
        obj_quat_ref = self.obj_poses[idx, 3:]
        inv_obj_quat_ref = quat_inv(obj_quat_ref)

        obj_pos_actual  = self._get_obj_pos(relative=False)
        obj_quat_actual = self._get_obj_quat(relative=False)
        inv_obj_quat_actual = quat_inv(obj_quat_actual)

        EE_pos_lr = self._get_EE_pos(relative=False)        # (N, 6)
        EE_quat_lr = self._get_EE_quat(relative=False)      # (N, 8)

        def _per_arm(EE_pos_ref, EE_quat_ref, EE_pos_actual, EE_quat_actual):
            rel_ref_pos_box = quat_apply(inv_obj_quat_ref,    EE_pos_ref    - obj_pos_ref)
            rel_ref_quat    = quat_mul(  inv_obj_quat_ref,    EE_quat_ref)
            rel_act_pos_box = quat_apply(inv_obj_quat_actual, EE_pos_actual - obj_pos_actual)
            rel_act_quat    = quat_mul(  inv_obj_quat_actual, EE_quat_actual)
            pos_err  = (rel_act_pos_box - rel_ref_pos_box).norm(dim=-1)
            quat_err = torch.abs(quat_error_magnitude(rel_act_quat, rel_ref_quat))
            return pos_err, quat_err

        pos_err_l, quat_err_l = _per_arm(
            self._robot_ee_ref_pos(self.EE_poses_l[idx, :3], "l"),
            self._robot_ee_ref_quat(self.EE_poses_l[idx, 3:], "l"),
            EE_pos_lr[:, :3], EE_quat_lr[:, :4],
        )
        pos_err_r, quat_err_r = _per_arm(
            self._robot_ee_ref_pos(self.EE_poses_r[idx, :3], "r"),
            self._robot_ee_ref_quat(self.EE_poses_r[idx, 3:], "r"),
            EE_pos_lr[:, 3:], EE_quat_lr[:, 4:],
        )
        return pos_err_l, quat_err_l, pos_err_r, quat_err_r


    @property
    def voc_kp_pos(self) -> float:
        return float(self._voc_kp_pos_seg.mean().item())

    @voc_kp_pos.setter
    def voc_kp_pos(self, value: float) -> None:
        self._voc_kp_pos_seg[:] = float(value)

    @property
    def voc_kp_rot(self) -> float:
        return float(self._voc_kp_rot_seg.mean().item())

    @voc_kp_rot.setter
    def voc_kp_rot(self, value: float) -> None:
        self._voc_kp_rot_seg[:] = float(value)

    @property
    def voc_kv_pos(self) -> float:
        return float(self._voc_kv_pos_seg.mean().item())

    @voc_kv_pos.setter
    def voc_kv_pos(self, value: float) -> None:
        self._voc_kv_pos_seg[:] = float(value)

    @property
    def voc_kv_rot(self) -> float:
        return float(self._voc_kv_rot_seg.mean().item())

    @voc_kv_rot.setter
    def voc_kv_rot(self, value: float) -> None:
        self._voc_kv_rot_seg[:] = float(value)

    def _get_joint_pos(self, relative=False):
        joint_pos_l = self.ur5e_l.data.joint_pos.clone()
        joint_pos_r = self.ur5e_r.data.joint_pos.clone()

        if relative:
            joint_pos_l -= self.joints_l[self.episode_length_buf]
            joint_pos_r -= self.joints_r[self.episode_length_buf]

        return torch.cat((joint_pos_l, joint_pos_r), 1)
    
    
    def _get_joint_vel(self, relative=False):
        ur5e_l_joint_vel = self.ur5e_l.data.joint_vel.clone()
        ur5e_r_joint_vel = self.ur5e_r.data.joint_vel.clone()

        if relative:
            ur5e_l_joint_vel -= self.joint_vel_l[self.episode_length_buf]
            ur5e_r_joint_vel -= self.joint_vel_r[self.episode_length_buf]

        return torch.cat((ur5e_l_joint_vel, ur5e_r_joint_vel), 1)
    
    def _get_forearm_endpoints(self, robot: Articulation):
        forearm_length = 0.4225  # forearm cylinder height in USD
        p2_local = torch.tensor([-forearm_length, 0.0, 0.0], device=self.device)
        forearm_pos = robot.data.body_pos_w[:, self.forearm_link_idx]
        forearm_quat = robot.data.body_quat_w[:, self.forearm_link_idx]
        p2 = forearm_pos + quat_apply(forearm_quat, p2_local.repeat(self.num_envs, 1))
        return forearm_pos, p2
    
    def _get_closest_point_on_forearm(self, robot: Articulation):
        flange_pos = robot.data.body_pos_w[:, self.flange_idx]
        p1, p2 = self._get_forearm_endpoints(robot)
        line_vec = p2 - p1
        t = torch.clamp(torch.sum((flange_pos - p1) * line_vec, dim=-1) / torch.sum(line_vec**2, dim=-1), 0.0, 1.0)
        return p1 + t.unsqueeze(-1) * line_vec

    def _get_flange_to_forearm_distance(self, robot: Articulation):
        flange_pos = robot.data.body_pos_w[:, self.flange_idx]
        p1, p2 = self._get_forearm_endpoints(robot)
        line_vec = p2 - p1
        t = torch.clamp(torch.sum((flange_pos - p1) * line_vec, dim=-1) / torch.sum(line_vec**2, dim=-1), 0.0, 1.0)
        return torch.norm(flange_pos - (p1 + t.unsqueeze(-1) * line_vec), dim=-1)
    
    def _compute_proximity_penalty(self) -> torch.Tensor:
        """PhysX separation-distance proximity penalty; silently clips overflow contacts."""
        penalty = torch.zeros(self.num_envs, device=self.device)
        for name, sensor in self.illegal_contact_sensors.items():
            _, _, _, separation, contact_count_per_link, _ = sensor.contact_physx_view.get_contact_data(self.dt)
            total_count = int(contact_count_per_link.sum().item())
            if total_count == 0:
                continue
            n_buf = separation.shape[0]
            if total_count > n_buf:
                # Throttle the warning so it doesn't spam every step under sustained overflow.
                if not getattr(self, "_proximity_overflow_warned", False):
                    print(f"[proximity-overflow] {name}: {total_count} contacts > buffer {n_buf}; overflow dropped.")
                    self._proximity_overflow_warned = True
                total_count = n_buf
            separation = separation[:total_count, 0]
            contact_count_per_env = contact_count_per_link.sum(dim=-1)
            env_ids = torch.repeat_interleave(
                torch.arange(self.num_envs, device=self.device), contact_count_per_env
            )
            if env_ids.shape[0] > total_count:
                env_ids = env_ids[:total_count]
            min_sep = torch.full((self.num_envs,), self.cfg.max_proximity * 2, device=self.device)
            min_sep.index_reduce_(0, env_ids, separation, reduce='amin', include_self=True)
            proximity = torch.clamp(1.0 - min_sep / self.cfg.max_proximity, min=0.0)
            penalty += proximity.square()
        return self.cfg.w_proximity_to_contact * penalty

    def _robot_ee_ref_pos(self, nominal_pos: torch.Tensor, arm: str) -> torch.Tensor:
        """Apply per-env robot base offset+yaw to a batch of nominal EE positions (env-frame).
        EE_ref = base_pos + offset + R_yaw * (nominal_pos - base_pos)
        """
        base_pos = self.arm_l_pose[:3] if arm == "l" else self.arm_r_pose[:3]
        offset   = self.robot_base_offset_l if arm == "l" else self.robot_base_offset_r
        yaw_q    = self.robot_yaw_quat_l    if arm == "l" else self.robot_yaw_quat_r
        return base_pos + offset + quat_apply(yaw_q, nominal_pos - base_pos)

    def _robot_ee_ref_quat(self, nominal_quat: torch.Tensor, arm: str) -> torch.Tensor:
        """Rotate nominal EE quats by the per-env robot yaw."""
        yaw_q = self.robot_yaw_quat_l if arm == "l" else self.robot_yaw_quat_r
        return quat_mul(yaw_q, nominal_quat)

    def _get_EE_pos(self, relative=True) -> torch.Tensor:
        EE_pos_l = self.ur5e_l.data.body_pos_w[:, self.EE_link_idx].clone() - self.scene.env_origins
        EE_pos_r = self.ur5e_r.data.body_pos_w[:, self.EE_link_idx].clone() - self.scene.env_origins

        if relative:
            EE_pos_l -= self._robot_ee_ref_pos(self.EE_poses_l[self.episode_length_buf, :3], "l")
            EE_pos_r -= self._robot_ee_ref_pos(self.EE_poses_r[self.episode_length_buf, :3], "r")

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
            desired_quat_l = self._robot_ee_ref_quat(self.EE_poses_l[self.episode_length_buf, 3:], "l")
            EE_quat_l = quat_mul(desired_quat_l, quat_inv(EE_quat_l))
            desired_quat_r = self._robot_ee_ref_quat(self.EE_poses_r[self.episode_length_buf, 3:], "r")
            EE_quat_r = quat_mul(desired_quat_r, quat_inv(EE_quat_r))

        return torch.cat((EE_quat_l, EE_quat_r), 1)
    
    def _get_EE_quat_error(self):
        # TODO: check orientation in usd file. might not match the one from the planner
        EE_quat_l = self.ur5e_l.data.body_quat_w[:, self.EE_link_idx].clone()
        EE_quat_r = self.ur5e_r.data.body_quat_w[:, self.EE_link_idx].clone()

        desired_quat_l = self._robot_ee_ref_quat(self.EE_poses_l[self.episode_length_buf, 3:], "l")
        desired_quat_r = self._robot_ee_ref_quat(self.EE_poses_r[self.episode_length_buf, 3:], "r")

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

