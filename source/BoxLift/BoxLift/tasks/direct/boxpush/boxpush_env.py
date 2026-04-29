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
from isaaclab.sim.schemas import modify_collision_properties
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import spawn_ground_plane, GroundPlaneCfg
from isaaclab.sensors.contact_sensor import ContactSensor
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from BoxLift.tasks.direct.boxpush.boxpush_env_cfg import *

from isaaclab.utils.math import quat_apply, quat_mul, quat_inv, quat_error_magnitude

class BoxpushEnv(DirectRLEnv):
    cfg: BoxpushEnvCfg

    def __init__(self, cfg: BoxpushEnvCfg, render_mode: str | None = None, **kwargs):
        # Re-run __post_init__ to pick up fields that were set by hydra CLI overrides
        # after the cfg dataclass was originally constructed (e.g. enable_phase_slowdown,
        # future_obs_steps, include_prev_actions). DirectRLEnv reads cfg.action_space /
        # cfg.observation_space in super().__init__ to allocate buffers, so this must
        # happen first.
        cfg.__post_init__()
        super().__init__(cfg, render_mode, **kwargs)

        self.EE_link_idx = self.ur5e.body_names.index("wrist_3_link")
        self.flange_idx = self.ur5e.body_names.index("wrist_3_link")
        self.forearm_link_idx = self.ur5e.body_names.index("forearm_link")

        # Continuous trajectory phase index (float). All trajectory lookups go through
        # _interp/_nlerp using this. dphase is the per-step advance rate (1 = nominal);
        # initialized to 0 so the first _get_observations after env init doesn't advance.
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.dphase = torch.zeros(self.num_envs, device=self.device)
        # Starting phase of the current episode (set in _reset_idx). Used by the
        # failure-aware resampler to credit every segment the episode traversed with the
        # terminal outcome (RobotDancing-style stability score).
        self.episode_start_phase = torch.zeros(self.num_envs, device=self.device)
        # Cumulative slowdown this episode: Σ (1 - dphase). Drives the quadratic pause penalty.
        self.cumulative_slowdown = torch.zeros(self.num_envs, device=self.device)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)

        # Failure-aware phase resampling state (allocated unconditionally so runtime toggles
        # don't crash; only updated when enable_failure_resampling=True).
        # Derive segment size (in trajectory steps) from configured duration and trajectory dt.
        T = self.obj_poses.shape[0]
        self._segment_size = max(1, int(round(self.cfg.phase_segment_s / self.dt)))
        # Number of segments that fit in the valid start range [0, T-1). Last segment may be
        # smaller than _segment_size if (T-1) isn't a multiple — _sample_phase_failure_weighted
        # clamps within-segment offsets accordingly.
        self._num_segments = max(1, (T - 1 + self._segment_size - 1) // self._segment_size)
        # Init at 0.5 (unknown): nothing is blocked from sampling on the first iter.
        self.segment_scores = torch.full(
            (self._num_segments,), 0.5, device=self.device, dtype=torch.float32
        )

    def _setup_scene(self):
        # Load the trajectory file
        traj = np.load(self.cfg.trajectory_path)

        # Store initial positions and joint states
        self.obj_poses          = torch.from_numpy(traj["obj_poses"]).float().to(self.device)
        self.obj_vel            = torch.from_numpy(traj["obj_vel"]).float().to(self.device)
        self.arm_pose           = torch.from_numpy(traj["arm_pose"]).float().to(self.device)
        self.joints             = torch.from_numpy(traj["joints"]).float().to(self.device)
        self.joint_vel          = torch.from_numpy(traj["joint_vel"]).float().to(self.device)
        self.joints_target      = torch.from_numpy(traj["joints_target"]).float().to(self.device)
        self.EE_poses           = torch.from_numpy(traj["EE_poses"]).float().to(self.device)
        self.dt                 = float(traj["dt"])

        # Set scene params from trajectory
        if "object_dims" in traj:
            self.cfg.object_dims = tuple(traj["object_dims"].tolist())
            self.cfg.cube_cfg.spawn.size = self.cfg.object_dims
        if "object_mass" in traj:
            self.cfg.object_mass = float(traj["object_mass"])
            self.cfg.cube_cfg.spawn.mass_props.mass = self.cfg.object_mass

        # TODO: Support last trajectory point
        # When phase slowdown is enabled, the wall-clock cap is a fixed multiple of nominal
        # trajectory duration (max_slowdown_multiplier) — this lets dphase_min go to 0 without
        # dividing by zero and bounds total episode time regardless of pause strategy.
        traj_duration = self.dt * (self.obj_poses.shape[0] - 1)
        if self.cfg.enable_phase_slowdown:
            self.cfg.episode_length_s = traj_duration * self.cfg.max_slowdown_multiplier
        else:
            self.cfg.episode_length_s = traj_duration

        ur5e_cfg = get_ur5e_cfg(self.cfg.ur5e_prim_path, self.arm_pose, self.cfg)
        self.ur5e = Articulation(ur5e_cfg)

        # Disable collision on base_link (ground is raised 1.8cm for the box pushing surface,
        # but the robot base sits at the original level — only the area near the box is elevated)
        modify_collision_properties(
            "/World/envs/env_0/ur5e/base_link",
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)
        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}

        # EE-box-relative reward gate: mark reference steps where the box is moving
        # (contact/near-contact phases), then dilate by ±dilation_steps so the window also
        # covers brief pre-contact approach and post-release follow-through. Precomputed
        # once from the reference trajectory — looked up at runtime by phase.floor().
        obj_vel_mag = self.obj_vel[:, :3].norm(dim=-1) + self.obj_vel[:, 3:].norm(dim=-1)
        moving = (obj_vel_mag > self.cfg.eef_box_gate_obj_vel_eps).float()
        if self.cfg.eef_box_gate_dilation_steps > 0:
            k = 2 * int(self.cfg.eef_box_gate_dilation_steps) + 1
            moving = torch.nn.functional.max_pool1d(
                moving.view(1, 1, -1), kernel_size=k, stride=1,
                padding=int(self.cfg.eef_box_gate_dilation_steps),
            ).view(-1)
        self.eef_box_gate_mask = moving.bool()  # (T,)

        # Regularization stuff
        self.prev_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 6), device=self.device)

        # Observation history buffer: (num_envs, history_steps, per_step_feature_dim).
        # per_step_feature_dim = 12 (rel_q + rel_qd) [+ 13 (obj_pos_rel + obj_quat_rel + obj_vel_rel) if include_object_obs]
        #                        [+ same again (absolute versions) if include_absolute_obs].
        # Index 0 = oldest, -1 = newest. Flattened in _get_observations.
        self.obs_history = torch.zeros(
            (self.num_envs, self.cfg.obs_history_steps, self.cfg.per_step_feature_dim),
            device=self.device,
        )

        # Perturbation state
        self.perturbation_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.perturbation_forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.perturbation_torques = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["ur5e"] = self.ur5e
        # add object to the scene
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        # add sensors to the scene
        for name, sensor in self.illegal_contact_sensors.items():
            self.scene.sensors[f"illegal_contact_sensor_{name}"] = sensor
        # add lights
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
        """Apply random external forces to the forearm to improve robustness."""
        # Decrement active perturbation counters
        self.perturbation_counter = torch.clamp(self.perturbation_counter - 1, min=0)

        # Start new perturbations with some probability
        new_perturbation = torch.rand(self.num_envs, device=self.device) < self.cfg.perturbation_probability
        new_perturbation &= self.perturbation_counter == 0  # don't overlap
        new_ids = new_perturbation.nonzero(as_tuple=False).squeeze(-1)

        if len(new_ids) > 0:
            self.perturbation_counter[new_ids] = self.cfg.perturbation_duration_steps
            # Random force direction and magnitude on forearm
            self.perturbation_forces[new_ids, 0] = (
                torch.randn(len(new_ids), 3, device=self.device)
                * self.cfg.perturbation_force_std
            )
            self.perturbation_torques[new_ids, 0] = (
                torch.randn(len(new_ids), 3, device=self.device)
                * self.cfg.perturbation_torque_std
            )

        # Clear expired perturbations
        expired = self.perturbation_counter == 0
        self.perturbation_forces[expired] = 0.0
        self.perturbation_torques[expired] = 0.0

        # Apply to forearm link
        self.ur5e.set_external_force_and_torque(
            self.perturbation_forces,
            self.perturbation_torques,
            body_ids=[self.forearm_link_idx],
            is_global=True,
        )

    def _interp(self, traj: torch.Tensor, phase: torch.Tensor | None = None) -> torch.Tensor:
        """Linear interpolation along the trajectory time axis.

        traj: (T, D) reference. phase: (num_envs,) float index, defaults to self.phase.
        Returns (num_envs, D).
        """
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

        # Set per-step phase advance rate. When the phase variable is enabled, the policy's
        # 7th action controls it; otherwise we advance by 1 nominal step. The actual phase
        # update happens in _get_observations (post-physics), so trajectory lookups in
        # _apply_action below still see the current step's reference.
        if self.cfg.enable_phase_slowdown:
            raw = self.actions[:, 6]
            # Slowdown-only: tanh(raw) ∈ [-1, 1] → dphase ∈ [2*dphase_min - 1, 1] before
            # the clamp, then clamped to [dphase_min, 1]. Net mapping:
            #   raw <= 0: dphase ∈ [dphase_min, 1] (smooth slowdown)
            #   raw >  0: dphase = 1 (deadzone — no speedup possible)
            # raw=0 is the neutral point (dphase=1, no slowdown).
            self.dphase = (1.0 + (1.0 - self.cfg.dphase_min) * torch.tanh(raw)).clamp(
                min=self.cfg.dphase_min, max=1.0
            )
        else:
            self.dphase = torch.ones_like(self.phase)

        self._apply_perturbations()

        EE_pos = self._interp(self.EE_poses[:, :3]) + self.scene.env_origins
        self.ee_markers.visualize(translations=EE_pos)

        obj_pos = self._interp(self.obj_poses[:, :3]) + self.scene.env_origins
        obj_quat = self._nlerp(self.obj_poses[:, 3:])
        self.cube_marker.visualize(translations=obj_pos, orientations=obj_quat)


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
        """Residual on current position shifted by planner's intended PD error, blended
        with a curriculum α ∈ [0, 1]:
            q_target = q_curr + (1-α)·(ref_target - ref_pos) + (α + ε(1-α))·scale·a
        At α=0 the command is pure planner PD feedforward with a small residual floor ε;
        at α=1 it collapses to mode C (pure residual from current position)."""
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
        q = self.get_joint_targets()
        q = q.clamp(self.ur5e.data.joint_pos_limits[...,0], self.ur5e.data.joint_pos_limits[..., 1])
        self.ur5e.set_joint_position_target(q)

    def _get_feedforward(self):
        """Planner's intended feedforward: joints_target - joints (proportional to desired PD force)."""
        return self._interp(self.joints_target) - self._interp(self.joints)

    def _get_noisy_obj_obs(self):
        """Sample observation-time noise on the box state once and return both the
        relative (vs reference) and absolute (env-frame) views. Same noise tensor used
        for both views so the policy sees mutually consistent readings, as a single
        real-world sensor would. Reward path is NOT affected — it calls _get_obj_pos /
        _get_obj_quat / _get_obj_vel directly which read clean ground truth.

        Velocity branch is computed only when include_obj_vel_obs=True; otherwise the
        rel_vel / abs_vel slots are returned as None.
        """
        obj_pos  = self.object.data.root_pos_w.clone() - self.scene.env_origins
        obj_quat = self.object.data.root_quat_w.clone()

        n = self.num_envs
        if self.cfg.obs_obj_pos_noise > 0:
            obj_pos = obj_pos + self.cfg.obs_obj_pos_noise * torch.randn(n, 3, device=self.device)
        if self.cfg.obs_obj_ori_noise > 0:
            aa = self.cfg.obs_obj_ori_noise * torch.randn(n, 3, device=self.device)
            delta_quat = torch.cat([torch.ones(n, 1, device=self.device), 0.5 * aa], dim=-1)
            delta_quat = delta_quat / delta_quat.norm(dim=-1, keepdim=True)
            obj_quat = quat_mul(delta_quat, obj_quat)

        rel_pos  = obj_pos - self._interp(self.obj_poses[:, :3])
        rel_quat = quat_mul(self._nlerp(self.obj_poses[:, 3:]), quat_inv(obj_quat))

        if self.cfg.include_obj_vel_obs:
            obj_vel = self.object.data.root_vel_w.clone()
            if self.cfg.obs_obj_lin_vel_noise > 0:
                obj_vel[:, :3] = obj_vel[:, :3] + self.cfg.obs_obj_lin_vel_noise * torch.randn(n, 3, device=self.device)
            if self.cfg.obs_obj_ang_vel_noise > 0:
                obj_vel[:, 3:] = obj_vel[:, 3:] + self.cfg.obs_obj_ang_vel_noise * torch.randn(n, 3, device=self.device)
            rel_vel = obj_vel - self._interp(self.obj_vel)
        else:
            obj_vel = None
            rel_vel = None
        return rel_pos, rel_quat, rel_vel, obj_pos, obj_quat, obj_vel

    def _get_observations(self) -> dict:
        # Advance phase post-physics so obs/rewards reference the new step. For envs that
        # were just reset, dphase=0 (set in _reset_idx) makes this a no-op.
        T = self.obj_poses.shape[0]
        self.phase = (self.phase + self.dphase).clamp(0.0, T - 1 - 1e-6)

        feature_parts = [self._get_joint_pos(relative=True), self._get_joint_vel(relative=True)]

        need_obj = self.cfg.include_object_obs
        if need_obj:
            obj_rel_pos, obj_rel_quat, obj_rel_vel, obj_abs_pos, obj_abs_quat, obj_abs_vel = (
                self._get_noisy_obj_obs()
            )
            feature_parts.extend([obj_rel_pos, obj_rel_quat])
            if self.cfg.include_obj_vel_obs:
                feature_parts.append(obj_rel_vel)

        if self.cfg.include_absolute_obs:
            feature_parts.extend([
                self._get_joint_pos(relative=False),
                self._get_joint_vel(relative=False),
            ])
            if need_obj:
                feature_parts.extend([obj_abs_pos, obj_abs_quat])
                if self.cfg.include_obj_vel_obs:
                    feature_parts.append(obj_abs_vel)

        current_features = torch.cat(feature_parts, dim=-1)  # (num_envs, per_step_feature_dim)

        # Shift history: oldest entry drops out, current becomes newest
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
                # World-frame delta (matches _get_obj_quat's `desired * inv(actual)` convention).
                futures.append(quat_mul(fut_quat, inv_cur_quat))
                if self.cfg.include_absolute_obs:
                    futures.append(fut_pos)
                    futures.append(fut_quat)
            obs_parts.append(torch.cat(futures, dim=-1))

        # Previous raw policy action (pre-scale).
        if self.cfg.include_prev_actions:
            obs_parts.append(self.prev_actions)

        obs = torch.cat(obs_parts, dim=-1)
        observations = {"policy": obs}
        return observations

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Trajectory completion. episode_length_s is sized in _setup_scene so the wall-clock
        # can't trigger before phase reaches T-1, even at sustained worst-case slowdown.
        time_out = self.phase >= (self.obj_poses.shape[0] - 1) - 1e-3

        obj_pos_error = self._get_obj_pos_error()
        obj_quat_error = self._get_obj_quat_error()

        self.reset_terminated = obj_pos_error > self.cfg.max_obj_dist_from_traj
        self.reset_terminated |= obj_quat_error > self.cfg.max_obj_angle_from_traj

        return self.reset_terminated, time_out

    def _update_segment_scores(self, env_ids):
        """EMA update of per-segment stability score using the episode's terminal outcome.
        Every segment traversed from the start phase through the end phase is credited with
        the same binary outcome: 1 if the episode failed (reset_terminated), 0 if it timed
        out (reached trajectory end). Segments successfully passed accumulate success
        signal every episode; the hard segment that breaks the episode accumulates failure
        signal. Score → resampling weight via probs ∝ score^beta."""
        f = self.reset_terminated[env_ids].float()                               # (N,) 1=fail, 0=timeout
        start_seg = (self.episode_start_phase[env_ids].long() // self._segment_size).clamp(max=self._num_segments - 1)
        end_seg   = (self.phase[env_ids].long()              // self._segment_size).clamp(max=self._num_segments - 1)

        # Traversed mask: (N, S) True where seg_idx ∈ [start_seg, end_seg]
        seg_idx = torch.arange(self._num_segments, device=self.device).unsqueeze(0)  # (1, S)
        traversed = (seg_idx >= start_seg.unsqueeze(1)) & (seg_idx <= end_seg.unsqueeze(1))

        count = traversed.float().sum(dim=0)                                     # (S,)
        fails = (traversed.float() * f.unsqueeze(1)).sum(dim=0)                  # (S,)

        fail_rate = fails / count.clamp(min=1)
        update_w = (count > 0).float() * self.cfg.phase_resample_alpha
        self.segment_scores = (1.0 - update_w) * self.segment_scores + update_w * fail_rate
        lo, hi = self.cfg.phase_resample_clamp
        self.segment_scores.clamp_(lo, hi)

    def _sample_phase_failure_weighted(self, n: int, T: int) -> torch.Tensor:
        """Sample n starting phases weighted by segment failure rate. Uniform within each segment."""
        probs = self.segment_scores ** self.cfg.phase_resample_beta
        probs = probs / probs.sum()
        segs = torch.multinomial(probs, num_samples=n, replacement=True)
        within = torch.randint(0, self._segment_size, (n,), device=self.device)
        phase = (segs * self._segment_size + within).float()
        if self.cfg.enable_phase_slowdown:
            phase = phase + torch.rand(n, device=self.device)
        return phase.clamp(max=T - 2)

    def _reset_idx(self, env_ids: Sequence[int] | None, fixed_value: int = None): # type: ignore
        if env_ids is None:
            env_ids : Sequence[int] = self.scene._ALL_INDICES  # type: ignore
        super()._reset_idx(env_ids)

        T = self.obj_poses.shape[0]

        # Failure-aware resampling: credit the failure segment (and partially the preceding
        # segment, if failure was early in its segment) with this episode's outcome.
        # Must run BEFORE we overwrite self.phase[env_ids] — that still holds the final phase.
        if self.cfg.enable_failure_resampling and hasattr(self, "reset_terminated"):
            self._update_segment_scores(env_ids)

        if fixed_value is not None:
            # Deterministic start (eval/play).
            self.phase[env_ids] = float(fixed_value) * torch.ones(len(env_ids), device=self.device)
        elif self.cfg.enable_failure_resampling:
            # Sample segment from failure-weighted distribution, then uniform within segment.
            self.phase[env_ids] = self._sample_phase_failure_weighted(len(env_ids), T)
        elif self.cfg.enable_phase_slowdown:
            # Random float starting phase.
            self.phase[env_ids] = torch.rand(len(env_ids), device=self.device) * (T - 2)
        else:
            # Random integer-valued starting phase. With dphase=1 each step the phase
            # stays integer, so _interp reproduces the original integer-indexing behavior.
            self.phase[env_ids] = torch.randint(0, T - 2, (len(env_ids),), device=self.device).float()
        # Remember the start phase so the next _update_segment_scores can credit every
        # segment traversed from start → end with the terminal outcome.
        self.episode_start_phase[env_ids] = self.phase[env_ids]
        # Set dphase=0 so the immediately-following _get_observations advance is a no-op
        # (the policy hasn't acted yet for this freshly-reset env). _pre_physics_step will
        # overwrite dphase from the next action.
        self.dphase[env_ids] = 0.0


        # Floor phase for integer indexing into trajectory tensors (sim state writes don't
        # need fractional precision; the env settles in a step).
        idx = self.phase[env_ids].floor().long().clamp(max=T - 1)
        initial_joint_pos = self.joints[idx].clone()
        initial_joint_vel = self.joint_vel[idx].clone()
        initial_joint_pos += self.cfg.reset_joint_pos_noise * torch.randn_like(initial_joint_pos)
        initial_joint_vel += self.cfg.reset_joint_vel_noise * torch.randn_like(initial_joint_vel)
        self.ur5e.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel, env_ids=env_ids)

        # Reset Object
        initial_object_pose = self.obj_poses[idx].clone()
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = self.obj_vel[idx].clone()

        n = len(env_ids)
        # Position noise (xy only)
        initial_object_pose[:, 0:2] += self.cfg.reset_obj_pos_xy_noise * torch.randn(n, 2, device=self.device)
        # Orientation noise: yaw-only (z-axis) so the box stays flat and doesn't intersect the ground.
        yaw = self.cfg.reset_obj_ori_noise * torch.randn(n, device=self.device)
        half = 0.5 * yaw
        zeros = torch.zeros_like(half)
        delta_quat = torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)
        initial_object_pose[:, 3:7] = quat_mul(delta_quat, initial_object_pose[:, 3:7])
        # Linear velocity noise (xy only) + angular velocity noise (z-axis only, consistent with yaw-only ori noise).
        initial_object_vel[:, 0:2] += self.cfg.reset_obj_lin_vel_xy_noise * torch.randn(n, 2, device=self.device)
        initial_object_vel[:, 5]   += self.cfg.reset_obj_ang_vel_noise * torch.randn(n, device=self.device)

        self.object.write_root_pose_to_sim(initial_object_pose, env_ids)
        self.object.write_root_velocity_to_sim(initial_object_vel, env_ids)

        # Reset prev variables. prev_actions tracks the previous raw residual action
        # (units: policy output, ≈ [-1, 1]), NOT joint positions — reset to 0 so the
        # first-step action_rate penalty isn't a giant spike from the unit mismatch.
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_vel[env_ids] = initial_joint_vel

        # Reset observation history
        self.obs_history[env_ids] = 0.0

        # Clear perturbations
        self.perturbation_counter[env_ids] = 0
        self.perturbation_forces[env_ids] = 0.0
        self.perturbation_torques[env_ids] = 0.0

        # Reset cumulative slowdown for the new episode.
        self.cumulative_slowdown[env_ids] = 0.0

    def _curriculum_alpha(self) -> float:
        """α ∈ [0, 1] schedule used by the reward curriculum, the mode-D action blend, and
        the policy-authored regularization scaling. alpha_warmup_steps=0 disables (α=1).
        force_alpha in [0, 1] short-circuits the schedule (sentinel < 0 disables) — used
        at eval time so a frozen policy runs at the same α it was trained at, regardless
        of step counter."""
        if 0.0 <= self.cfg.force_alpha <= 1.0:
            return float(self.cfg.force_alpha)
        if self.cfg.alpha_warmup_steps > 0:
            return min(1.0, self.common_step_counter / self.cfg.alpha_warmup_steps)
        return 1.0

    def _reward_track(self, error, sigma, tolerance=0.0):
        # sigma can be a scalar or an iterable of scalars. With multiple sigmas the kernels
        # are averaged so max reward stays in [0, 1].
        error = error * (error > tolerance)
        if isinstance(sigma, (tuple, list)):
            sigmas = torch.tensor(sigma, device=error.device, dtype=error.dtype)
            kernels = torch.exp(-error.unsqueeze(-1) / (sigmas ** 2))  # (N, K)
            return kernels.mean(dim=-1)
        return torch.exp(-error / (sigma ** 2))

    def _get_rewards(self) -> torch.Tensor:
        # Task Reward
        obj_pos_error = self._get_obj_pos_error()
        rew_obj_pos = self.cfg.w_obj_pos * self._reward_track(obj_pos_error ** 2, self.cfg.sigma_obj_pos, self.cfg.tol_obj_pos)

        obj_quat_error = self._get_obj_quat_error()
        rew_obj_quat = self.cfg.w_obj_quat * self._reward_track(obj_quat_error ** 2, self.cfg.sigma_obj_quat, self.cfg.tol_obj_quat)

        # Object velocity tracking, split linear vs angular to keep kernel sigmas matched
        # to each quantity's natural scale. Instantaneous signal that catches "policy
        # stopped pushing" before obj_pos_error integrates up to termination threshold.
        obj_vel_rel = self._get_obj_vel(relative=True)
        obj_lin_vel_error = obj_vel_rel[:, :3].norm(dim=-1)
        obj_ang_vel_error = obj_vel_rel[:, 3:].norm(dim=-1)
        rew_obj_lin_vel = self.cfg.w_obj_lin_vel * self._reward_track(
            obj_lin_vel_error ** 2, self.cfg.sigma_obj_lin_vel, self.cfg.tol_obj_lin_vel)
        rew_obj_ang_vel = self.cfg.w_obj_ang_vel * self._reward_track(
            obj_ang_vel_error ** 2, self.cfg.sigma_obj_ang_vel, self.cfg.tol_obj_ang_vel)
        rew_obj_vel = rew_obj_lin_vel + rew_obj_ang_vel

        # Curriculum α drives (1) the reward-weight ramp, (2) the mode-D action blend, and
        # (3) the mode-D policy-regularization scaling. common_step_counter increments once
        # per env step and is maintained by DirectRLEnv.
        alpha = self._curriculum_alpha()
        w_task_eff  = self.cfg.w_task_start  + (self.cfg.w_task  - self.cfg.w_task_start)  * alpha
        w_track_eff = self.cfg.w_track_start + (self.cfg.w_track - self.cfg.w_track_start) * alpha

        rew_task = w_task_eff * (rew_obj_pos + rew_obj_quat + rew_obj_vel)

        # Tracking Reward — phase-exclusive: relative EE-in-box-frame when the reference
        # expects contact (gate=1), absolute EE / joint tracking elsewhere (gate=0). One
        # active tracker at a time, no gradient competition between them. Gate is derived
        # from the precomputed eef_box_gate_mask at the current integer phase.
        T = self.eef_box_gate_mask.shape[0]
        phase_idx = self.phase.floor().long().clamp(max=T - 1)
        gate = self.eef_box_gate_mask[phase_idx].float()
        abs_gate = 1.0 - gate

        EE_pos_error = self._get_EE_pos_error()
        rew_EE_pos = abs_gate * self.cfg.w_eef_pos * self._reward_track(EE_pos_error ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)

        eef_quat_error = self._get_EE_quat_error()
        rew_EE_quat = abs_gate * self.cfg.w_eef_quat * self._reward_track(eef_quat_error ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)

        joint_pos_error = (self._get_joint_pos(relative=True) ** 2).sum(dim=-1)
        rew_joint_pos = abs_gate * self.cfg.w_joint_pos * self._reward_track(joint_pos_error, self.cfg.sigma_joint_pos, self.cfg.tol_joint_pos)

        # Relative EE-in-box-frame tracking (gate=1). Computes its own gate internally —
        # same mask, same phase — so the pair is exactly mutually exclusive.
        rew_eef_box_rel_pos, rew_eef_box_rel_quat = self._compute_eef_box_rel_rewards()

        rew_track = w_track_eff * (rew_EE_pos + rew_EE_quat + rew_joint_pos + rew_eef_box_rel_pos + rew_eef_box_rel_quat)

        # Regularization Reward
        joint_acc = (self._get_joint_vel() - self.prev_joint_vel) / self.dt
        joint_acc *= torch.abs(joint_acc) > self.cfg.tol_joint_acc
        joint_acc_penalty = joint_acc.square().sum(dim=-1)
        rew_joint_acc = self.cfg.w_joint_acc * joint_acc_penalty

        torque = self.ur5e.data.applied_torque.clone()
        torque *= torch.abs(torque) > self.cfg.tol_joint_torque
        torque_penalty = torque.square().sum(dim=-1)
        rew_torque = self.cfg.w_joint_torque * torque_penalty

        # Policy-authored penalty scaling.
        #   action_rate (mode D only): scaled by (α + ε(1-α)) — same gain as the action
        #     itself — so the penalty tracks the action's actual impact on the env.
        #   action_norm (all modes): scaled by (α + ε(1-α)) as a regularization curriculum.
        #     Early training (low α) doesn't penalize large residuals, letting the policy
        #     explore non-zero actions without being pushed toward a=0. Late training (α=1)
        #     applies the full penalty to refine toward "quiet unless necessary."
        eps = float(self.cfg.action_alpha_floor)
        alpha_scale = alpha + eps * (1.0 - alpha)
        rate_reg_scale = alpha_scale if self.cfg.action_mode == "D" else 1.0
        norm_reg_scale = alpha_scale

        # Residual action rate (first 6 dims). Phase action (dim 6) handled separately.
        action_rate_error = (self.actions[:, :6] - self.prev_actions)
        action_rate_error *= torch.abs(action_rate_error) > self.cfg.tol_action_rate
        action_rate_penalty = action_rate_error.square().sum(dim=-1)
        rew_action_rate = rate_reg_scale * self.cfg.w_action_rate * action_rate_penalty

        # Residual action magnitude: bias toward zero residual when the nominal plan is good enough.
        action_norm_error = self.actions[:, :6].clone()
        action_norm_error *= torch.abs(action_norm_error) > self.cfg.tol_action_norm
        action_norm_penalty = action_norm_error.square().sum(dim=-1)
        rew_action_norm = norm_reg_scale * self.cfg.w_action_norm * action_norm_penalty

        # Cumulative-quadratic pause penalty. Running (dphase=1) gives zero cost; per-step
        # cost = w_total_slowdown · cum_slowdown · (1 - dphase), so integrating over a pause
        # of k steps ≈ w_total_slowdown · k². Resets per-episode via _reset_idx.
        if self.cfg.enable_phase_slowdown:
            slowdown_step = (1.0 - self.dphase).clamp(min=0.0)
            self.cumulative_slowdown += slowdown_step
            rew_total_slowdown = self.cfg.w_total_slowdown * self.cumulative_slowdown * slowdown_step
        else:
            rew_total_slowdown = torch.zeros(self.num_envs, device=self.device)

        # Joint limit penalty
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
            rew_joint_acc + rew_torque + rew_action_rate + rew_action_norm + rew_joint_limit + rew_illegal_contact + rew_proximity + rew_flange_forearm_dist + rew_total_slowdown
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
            # Per-step end-of-episode rates. Logged as fractions of total step-instances:
            #   completion_fraction = fraction that timed out (= reached end of trajectory)
            #   termination_fraction = fraction that failed (= reset_terminated)
            # Aggregated success_rate over a window = mean(cf) / (mean(cf) + mean(tf)).
            # Per-step end-of-episode rates. Compute success rate offline from these as
            # mean(cf) / (mean(cf) + mean(tf)) over a window — the ratio must be done on
            # iter-aggregated values, not per-step (which gives wrong weights when steps
            # have varying numbers of endings).
            "Rewards/completion_fraction": self.reset_time_outs.float().mean(),
            "Rewards/termination_fraction": self.reset_terminated.float().mean(),
            "Error/obj_pos_error": obj_pos_error.mean(),
            "Error/obj_quat_error": obj_quat_error.mean(),
            "Error/obj_lin_vel_error": obj_lin_vel_error.mean(),
            "Error/obj_ang_vel_error": obj_ang_vel_error.mean(),
            "Error/EE_pos_error": EE_pos_error.mean(),
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
        }

        # Failure-aware phase resampling diagnostics.
        if self.cfg.enable_failure_resampling:
            probs = self.segment_scores ** self.cfg.phase_resample_beta
            probs = probs / probs.sum()
            entropy = -(probs * (probs + 1e-8).log()).sum()
            self.extras["log"]["PhaseResample/mean_failure_rate"] = self.segment_scores.mean()
            self.extras["log"]["PhaseResample/max_failure_rate"] = self.segment_scores.max()
            self.extras["log"]["PhaseResample/min_failure_rate"] = self.segment_scores.min()
            self.extras["log"]["PhaseResample/sample_entropy"] = entropy

        # Task/track rewards are NOT scaled by dphase — sustained pauses are discouraged
        # via the cumulative-quadratic `rew_total_slowdown` inside rew_regularization. This
        # keeps short pause-to-recover strategies positive-EV while bounding total pause cost.
        total_reward = rew_task + rew_track + rew_completion - rew_regularization

        # Update prev residual action / joint vel (first 6 dims of self.actions are residuals).
        self.prev_actions[:] = self.actions[:, :6]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

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

    def _compute_eef_box_rel_rewards(self):
        """Rewards the policy for matching the reference's EE pose expressed in the box's
        frame (i.e., "keep the EE at the same offset from the box as the planner expected"),
        gated by the reference EE↔box distance so the term only contributes when contact is
        expected. During regrasp/approach (EE far from box by design) the gate ≈ 0 and this
        reward doesn't fight the joint/EE trackers.

        Returns (rew_pos, rew_quat) each shaped (num_envs,).
        """
        EE_pos_ref   = self._interp(self.EE_poses[:, :3])
        EE_quat_ref  = self._nlerp(self.EE_poses[:, 3:])
        obj_pos_ref  = self._interp(self.obj_poses[:, :3])
        obj_quat_ref = self._nlerp(self.obj_poses[:, 3:])

        inv_obj_quat_ref = quat_inv(obj_quat_ref)
        rel_ref_pos_box  = quat_apply(inv_obj_quat_ref, EE_pos_ref - obj_pos_ref)
        rel_ref_quat     = quat_mul(inv_obj_quat_ref, EE_quat_ref)

        self._get_EE_pos()

        EE_pos_actual   = self._get_EE_pos(relative=False)
        EE_quat_actual  = self._get_EE_quat(relative=False)
        obj_pos_actual  = self._get_obj_pos(relative=False)
        obj_quat_actual = self._get_obj_quat(relative=False)

        inv_obj_quat_actual = quat_inv(obj_quat_actual)
        rel_act_pos_box  = quat_apply(inv_obj_quat_actual, EE_pos_actual - obj_pos_actual)
        rel_act_quat     = quat_mul(inv_obj_quat_actual, EE_quat_actual)

        pos_err_sq  = (rel_act_pos_box - rel_ref_pos_box).square().sum(dim=-1)
        quat_err_sq = quat_error_magnitude(rel_act_quat, rel_ref_quat).square()

        # Precomputed boolean gate from |obj_vel_ref| (dilated). Active during contact /
        # near-contact phases; zero during pure approach/regrasp.
        T = self.eef_box_gate_mask.shape[0]
        idx = self.phase.floor().long().clamp(max=T - 1)
        gate = self.eef_box_gate_mask[idx].float()

        rew_pos  = gate * self.cfg.w_eef_box_rel_pos  * self._reward_track(
            pos_err_sq,  self.cfg.sigma_eef_box_rel_pos,  self.cfg.tol_eef_box_rel_pos)
        rew_quat = gate * self.cfg.w_eef_box_rel_quat * self._reward_track(
            quat_err_sq, self.cfg.sigma_eef_box_rel_quat, self.cfg.tol_eef_box_rel_quat)
        
        return rew_pos, rew_quat

    def _compute_proximity_penalty(self) -> torch.Tensor:
        """Penalize links approaching illegal contact surfaces based on PhysX separation distance."""
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
