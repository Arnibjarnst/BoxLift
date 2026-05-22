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
        super().__init__(cfg, render_mode, **kwargs)

        # With the ur5e.usd asset (no separate sphere attachment) the EE *is* wrist_3_link
        # — same convention as boxhinge. flange_idx and EE_link_idx both point at it.
        self.EE_link_idx = self.ur5e_r.body_names.index("wrist_3_link")
        self.flange_idx = self.ur5e_r.body_names.index("wrist_3_link")
        self.forearm_link_idx = self.ur5e_r.body_names.index("forearm_link")

        # Action scale as a tensor so per-joint lists (length 12) and scalars both
        # broadcast correctly against (N, 12) actions in get_joint_targets_*.
        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)

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

        # Set scene params from trajectory
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
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)

        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}

        self.ee_contact_sensors = [ContactSensor(cfg) for cfg in self.cfg.ee_contact_sensors]

        # EE-box-relative reward gate: mark reference steps where the box is moving
        # (contact/near-contact phases); dilated by ±dilation_steps so it covers brief
        # pre/post-contact margin. Precomputed once; looked up at runtime by phase.
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
        self.prev_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 12), device=self.device)

        # === Virtual Object Controller (VOC) state ===
        # Per-segment gains: each segment of the trajectory has its OWN kp/kv that
        # decays independently from its own trailing-mean ring buffer. The legacy
        # scalar attributes `voc_kp_pos`, `voc_kp_rot`, `voc_kv_pos`, `voc_kv_rot` are
        # exposed as @property (getter = mean of per-seg tensor, setter = broadcast)
        # so play.py / record.py / ur_rtde scripts that assign or read them keep working
        # unchanged.
        #
        # Critical-damping defaults derived from box mass and a rough inertia estimate
        # (uniform cube: I ≈ m·d_max²/12). Mass/dims sourced from cube_cfg.spawn — already
        # overridden from the trajectory file above if those keys were present.
        mass = float(self.cfg.cube_cfg.spawn.mass_props.mass)
        d_max = float(max(self.cfg.cube_cfg.spawn.size))
        inertia_est = mass * (d_max ** 2) / 12.0
        kp_pos_init = float(self.cfg.voc_kp_pos)
        kp_rot_init = float(self.cfg.voc_kp_rot)
        kv_pos_init = self.cfg.voc_kv_pos_scale * (kp_pos_init * mass) ** 0.5
        kv_rot_init = self.cfg.voc_kv_rot_scale * (kp_rot_init * inertia_est) ** 0.5

        # Build phase_to_segment: (T,) long tensor mapping each trajectory frame to its
        # segment id. Number of segments depends on segmentation mode.
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
            # Boundaries: [0, bin_size, 2*bin_size, ..., T]. Final may be < bin_size.
            self._voc_seg_boundaries = torch.minimum(
                torch.arange(N + 1, device=self.device, dtype=torch.long) * bin_size,
                torch.tensor(T, device=self.device, dtype=torch.long),
            )
        elif seg_mode == "contact":
            # Recompute the contact gate from RAW obj_vel for segmentation purposes —
            # `self.eef_box_gate_mask` is the *reward* gate, which often uses a huge
            # dilation (e.g. 1e7 = "always-on") so the policy gets eef_box_rel reward
            # everywhere. That dilation collapses the gate to all-True and kills the
            # segment transitions. Here we apply a much smaller dilation that smooths
            # single-frame flickers without losing the lift / rotate / place breaks.
            obj_vel_mag = self.obj_vel[:, :3].norm(dim=-1) + self.obj_vel[:, 3:].norm(dim=-1)
            seg_moving = (obj_vel_mag > self.cfg.eef_box_gate_obj_vel_eps).float()
            dil = int(self.cfg.voc_segment_dilation_steps)
            if dil > 0:
                k = 2 * dil + 1
                seg_moving = torch.nn.functional.max_pool1d(
                    seg_moving.view(1, 1, -1), kernel_size=k, stride=1, padding=dil,
                ).view(-1)
            seg_gate = seg_moving.bool()  # (T,)

            # Segment boundaries at every transition of the segmentation gate. First
            # segment runs [0, t1); subsequent ones span [t_i, t_{i+1}); final ends at T.
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
            # One-time print so the segment layout is visible in the training log.
            print(f"[VOC] contact segmentation → {self._voc_n_segments} segments, "
                  f"boundaries (frames) = {boundaries.tolist()}, "
                  f"gate@boundary = {seg_gate[boundaries[:-1]].tolist()}  "
                  f"(seg_dilation={dil}, eps={self.cfg.eef_box_gate_obj_vel_eps})")
        else:
            raise ValueError(f"Unknown voc_segmentation: {seg_mode!r} (expected 'none'|'uniform'|'contact')")

        N_s = self._voc_n_segments
        # Per-segment gain tensors — canonical VOC state. Internal env code uses these
        # directly; external scripts go through the legacy scalar properties.
        self._voc_kp_pos_seg = torch.full((N_s,), kp_pos_init, device=self.device)
        self._voc_kp_rot_seg = torch.full((N_s,), kp_rot_init, device=self.device)
        self._voc_kv_pos_seg = torch.full((N_s,), kv_pos_init, device=self.device)
        self._voc_kv_rot_seg = torch.full((N_s,), kv_rot_init, device=self.device)

        # Per-env per-segment episode-cumulative rewards. An env spans multiple segments
        # over its episode (phase advances every step), so each (env, seg) tracks an
        # independent partial sum. Means are computed at reset (only for seg with steps>0)
        # and pushed to the per-seg ring buffer below.
        self._voc_ep_rew_task_seg  = torch.zeros((self.num_envs, N_s), device=self.device)
        self._voc_ep_rew_track_seg = torch.zeros((self.num_envs, N_s), device=self.device)
        self._voc_ep_steps_seg     = torch.zeros((self.num_envs, N_s), dtype=torch.long, device=self.device)

        # Per-segment ring buffer of recent completed-episode normalized means. NaN
        # init so partially-filled rings correctly ignore empty slots. Each segment's
        # write pointer advances independently (envs differ in which segments they span).
        W = self.cfg.voc_reward_window_size
        self._voc_buf_task_seg  = torch.full((N_s, W), float("nan"), device=self.device)
        self._voc_buf_track_seg = torch.full((N_s, W), float("nan"), device=self.device)
        self._voc_buf_idx_seg   = torch.zeros(N_s, dtype=torch.long, device=self.device)
        self._voc_decay_step_counter = 0

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

        # Apply VOC wrench on the cube (zero short-circuit when disabled/decayed).
        self._apply_voc()

        EE_pos_l = self.EE_poses_l[self.episode_length_buf, :3] + self.scene.env_origins
        EE_pos_r = self.EE_poses_r[self.episode_length_buf, :3] + self.scene.env_origins

        # Visualize EE markers
        ee_marker_pos = torch.stack([EE_pos_l, EE_pos_r], dim=1).view(-1, 3)
        self.ee_markers.visualize(translations=ee_marker_pos)

        obj_pos = self.obj_poses[self.episode_length_buf, :3] + self.scene.env_origins
        obj_quat = self.obj_poses[self.episode_length_buf, 3:]

        self.cube_marker.visualize(translations=obj_pos, orientations=obj_quat)

        # Cache joint targets once per policy step. Modes C/D depend on q_current — without
        # caching, _apply_action would recompute the target every decimation substep and
        # the target would drift with the joint inside the substep window. Caching here
        # makes the target a plain ZOH-from-q_at_policy_time, matching standard deployment
        # chains where a high-rate low-level controller chases a fixed target between
        # policy updates. Modes A/B don't depend on q_current so caching is a no-op for them.
        q_l, q_r = self.get_joint_targets()
        q_l = q_l.clamp(self.ur5e_l.data.joint_pos_limits[..., 0],
                        self.ur5e_l.data.joint_pos_limits[..., 1])
        q_r = q_r.clamp(self.ur5e_r.data.joint_pos_limits[..., 0],
                        self.ur5e_r.data.joint_pos_limits[..., 1])
        self._cached_joint_target = (q_l, q_r)

    def get_joint_targets_A(self):
        """Residual on planner targets. Planner feedforward is baked into the action."""
        q_l = self.joints_target_l[self.episode_length_buf] + self._scaled_action()[:, :6]
        q_r = self.joints_target_r[self.episode_length_buf] + self._scaled_action()[:, 6:]
        return q_l, q_r

    def get_joint_targets_B(self):
        """Residual on trajectory positions. Pair with planner-FF observation (no FF baked in)."""
        q_l = self.joints_l[self.episode_length_buf] + self._scaled_action()[:, :6]
        q_r = self.joints_r[self.episode_length_buf] + self._scaled_action()[:, 6:]
        return q_l, q_r

    def get_joint_targets_C(self):
        """Residual on current joint positions."""
        q_curr = self._get_joint_pos()
        q_l = q_curr[:, :6] + self._scaled_action()[:, :6]
        q_r = q_curr[:, 6:] + self._scaled_action()[:, 6:]
        return q_l, q_r

    def get_joint_targets_BC(self):
        """B→C curriculum: q_target = (1-α)·q_ref + α·q_curr + scale·a.

        Equivalent form:  q_curr + (1-α)·(q_ref - q_curr) + scale·a — a decaying spring
        toward the reference whose effective stiffness is kp·(1-α). At α=0 → mode B,
        at α=1 → mode C. Action authority is constant (full scale·a) throughout the
        curriculum; only the spring magnitude decays.
        """
        alpha = self._curriculum_alpha()
        q_curr = self._get_joint_pos()
        q_ref_l = self.joints_l[self.episode_length_buf]
        q_ref_r = self.joints_r[self.episode_length_buf]
        scaled = self._scaled_action()
        q_l = (1.0 - alpha) * q_ref_l + alpha * q_curr[:, :6] + scaled[:, :6]
        q_r = (1.0 - alpha) * q_ref_r + alpha * q_curr[:, 6:] + scaled[:, 6:]
        return q_l, q_r

    def _curriculum_alpha(self) -> float:
        """α ∈ [0, 1] schedule used by mode BC (and mode D when implemented).
        `alpha_warmup_steps=0` disables the curriculum (α=1 always, equivalent to mode C).
        `force_alpha ∈ [0, 1]` short-circuits the schedule — used at eval to pin α to
        the training-end value so the policy isn't re-exposed to the assist on a fresh
        common_step_counter."""
        if 0.0 <= self.cfg.force_alpha <= 1.0:
            return float(self.cfg.force_alpha)
        if self.cfg.alpha_warmup_steps > 0:
            return min(1.0, self.common_step_counter / self.cfg.alpha_warmup_steps)
        return 1.0

    def _scaled_action(self) -> torch.Tensor:
        """Returns action_scale * actions, broadcasting scalar or per-joint scale across (N, 12)."""
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
            raise NotImplementedError(
                "Mode D (planner-PD-error blend with curriculum α) is not yet ported to "
                "boxlift. Use mode B (no curriculum) or BC (decaying spring) instead."
            )
        raise ValueError(f"Unknown action_mode: {mode!r}")

    def _apply_action(self) -> None:
        q_l, q_r = self._cached_joint_target
        self.ur5e_l.set_joint_position_target(q_l)
        self.ur5e_r.set_joint_position_target(q_r)

    def _apply_voc(self):
        """Virtual Object Controller (DexMachina, Mandi et al. 2025).

        Applies a 6-DoF PD wrench on the cube driving it toward the reference trajectory:
          F = kp·(ref_pos  - obj_pos) - kv·(obj_lin_vel - ref_lin_vel)
          T = kp·rot_err_axisangle    - kv·(obj_ang_vel - ref_ang_vel)
        rot_err is computed from quat_mul(ref, inv(obj)) as the small-angle axis-angle
        vector (2·sign(w)·xyz). Gain decays in `_voc_decay_check` as the policy meets
        reward thresholds. Wrench is set every env step and held across decimation substeps.
        """
        # Short-circuit when VOC is disabled OR all segments have fully decayed.
        # In per-segment mode, a single segment can keep VOC active even after others
        # zero out — we only skip the wrench buffer write when the entire pool is zero.
        # Per-env gains below will be 0 for envs whose current segment has decayed,
        # so they naturally contribute zero wrench without needing a per-env branch.
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

        # Per-env current segment via phase_to_segment lookup. Each env's gain is the
        # gain of the segment it's currently in — gather as (N,) then unsqueeze for
        # broadcast against the (N, 3) error vectors.
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

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._get_joint_pos(relative=True),
                self._get_joint_vel(relative=True),
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
        initial_object_vel[:, 5]   += self.cfg.reset_obj_ang_vel_noise * torch.randn(n, device=self.device)

        self.object.write_root_pose_to_sim(initial_object_pose, env_ids)
        self.object.write_root_velocity_to_sim(initial_object_vel, env_ids)

        # === Per-segment VOC ring-buffer push ===
        # For each completing env, push the per-segment normalized reward mean (where
        # steps>0) to that segment's ring buffer. Different envs visited different
        # segment sets (RSI sample + episode length determine coverage), so per-segment
        # write pointers advance independently. Must run BEFORE we zero the trackers.
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
        # Zero per-env per-segment accumulators for the new episode.
        self._voc_ep_rew_task_seg[eids_t]  = 0.0
        self._voc_ep_rew_track_seg[eids_t] = 0.0
        self._voc_ep_steps_seg[eids_t]     = 0

        # Reset prev variables. prev_actions tracks the previous RAW residual action
        # (units: policy output, ≈ [-1, 1]), NOT joint positions — reset to 0 so the
        # first-step action_rate penalty isn't a giant spike from the unit mismatch.
        self.prev_actions[env_ids] = 0.0
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

        rew_task_unweighted = rew_obj_pos + rew_obj_quat
        rew_task = self.cfg.w_task * rew_task_unweighted

        # Tracking — phase-exclusive: absolute (EE/joint) when the reference is OUT of
        # contact (abs_gate=1), relative EE-in-box-frame when IN contact (gate=1). No
        # gradient competition between trackers, mirrors boxhinge.
        T = self.eef_box_gate_mask.shape[0]
        phase_idx = self.episode_length_buf.clamp(max=T - 1)
        gate = self.eef_box_gate_mask[phase_idx].float()
        abs_gate = 1.0 - gate

        # L+R tracking kernels are averaged (not summed) so the reward is bounded in
        # [0, w_eef_pos] regardless of arm count — matches boxhinge's per-arm scale,
        # restores threshold calibration, and avoids the dual-arm 2× scale that was
        # making voc_threshold_track effectively a quarter-saturation gate.
        EE_pos_error_l, EE_pos_error_r = self._get_EE_pos_error()
        rew_EE_pos_l = self._reward_track(EE_pos_error_l ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)
        rew_EE_pos_r = self._reward_track(EE_pos_error_r ** 2, self.cfg.sigma_eef_pos, self.cfg.tol_eef_pos)
        rew_EE_pos = abs_gate * self.cfg.w_eef_pos * 0.5 * (rew_EE_pos_l + rew_EE_pos_r)

        eef_quat_error_l, eef_quat_error_r = self._get_EE_quat_error()
        rew_EE_quat_l = self._reward_track(eef_quat_error_l ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)
        rew_EE_quat_r = self._reward_track(eef_quat_error_r ** 2, self.cfg.sigma_eef_quat, self.cfg.tol_eef_quat)
        rew_EE_quat = abs_gate * self.cfg.w_eef_quat * 0.5 * (rew_EE_quat_l + rew_EE_quat_r)

        # BC-style joint tracking (DexMachina r_bc): per-joint kernel, then mean across
        # all 12 joints (both arms). Each joint's deviation enters its own exp; bounded
        # in [0, 1]. Different from the legacy "sum-then-kernel" — that hid one bad joint
        # behind the others.
        joint_pos_err_per_joint = self._get_joint_pos(relative=True) ** 2  # (N, 12)
        joint_pos_kernels = self._reward_track(
            joint_pos_err_per_joint, self.cfg.sigma_joint_pos, self.cfg.tol_joint_pos
        )  # (N, 12) — broadcasts over the trailing axis
        rew_joint_pos = abs_gate * self.cfg.w_joint_pos * joint_pos_kernels.mean(dim=-1)

        # Relative EE-in-box-frame tracking, summed across arms (mutually exclusive with
        # the absolute trackers above via the gate).
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

        # Residual action magnitude penalty: bias toward zero residual when the nominal
        # plan is already good (important for VOC since the VOC wrench is the actor early).
        action_norm_error = self.actions.clone()
        action_norm_error *= torch.abs(action_norm_error) > self.cfg.tol_action_norm
        action_norm_penalty = action_norm_error.square().sum(dim=-1)
        rew_action_norm = self.cfg.w_action_norm * action_norm_penalty

        # Joint limit penalty — read from the cached (already-clamped) joint target to
        # match what _apply_action commanded. joint_limit_eps insets the soft band slightly
        # so the penalty starts before the hard limit and the limit clamp doesn't mask it.
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

        # Boolean (binary) penalty for breaching the safety zone
        is_too_close_l = (flange_to_forearm_dist_l < self.cfg.max_flange_forearm_distance).float()
        is_too_close_r = (flange_to_forearm_dist_r < self.cfg.max_flange_forearm_distance).float()

        rew_flange_forearm_dist = self.cfg.w_flange_forearm_dist * (is_too_close_l + is_too_close_r)

        rew_regularization = self.cfg.w_regularization * (
            rew_joint_acc + rew_torque + rew_action_rate + rew_action_norm
            + rew_joint_limit + rew_illegal_contact + rew_proximity + rew_flange_forearm_dist
        )

        # Build a per-env dict of (num_envs,) tensors first, then derive the env-mean log
        # the runner consumes for W&B / TensorBoard. The per-env dict is stashed under
        # `extras["log_per_env"]` so record.py can collect it for per-env post-processing
        # (e.g. "what did env 12's joint_torque penalty look like at the spike?"). The
        # training logger ignores it — it only reads `extras["log"]`.
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
        # Per-env tensors are opt-in via cfg.emit_per_env_extras (default False). Keeps the
        # training-time `extras` payload identical to the pre-refactor layout so the runner
        # can't accidentally pick the dict up or pay extra GPU→CPU transfer costs.
        if getattr(self.cfg, "emit_per_env_extras", False):
            self.extras["log_per_env"] = per_env_log

        total_reward = rew_task + rew_track - rew_regularization

        # === VOC: per-segment accumulation for the decay check ===
        # Track the *unweighted* task/tracking signals so thresholds correspond directly
        # to per-step kernel values in [0, 1] rather than to weighted sums whose scale
        # would shift with w_task/w_track etc. Route this step's per-env contribution
        # into the segment corresponding to the current trajectory phase — each
        # (env, segment) tracks an independent partial sum.
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

        # Decay check rate-limited to avoid hammering it every step. Uses the tensor
        # max() directly (not the property mean) so we keep checking until the LAST
        # segment decays — in per-segment mode, the mean drops as easy segments zero
        # out but hard segments may still be active.
        self._voc_decay_step_counter += 1
        if (self.cfg.voc_enabled
                and float(self._voc_kp_pos_seg.max().item()) > 0.0
                and self._voc_decay_step_counter >= self.cfg.voc_decay_check_interval):
            self._voc_decay_step_counter = 0
            self._voc_decay_check()

        # VOC logging — headline values are cross-segment mean / min / max so you can
        # see the distribution at a glance, plus per-segment kp and recent task/track
        # means for spotting which segment is the bottleneck.
        self.extras["log"]["VOC/kp_pos_mean"] = self._voc_kp_pos_seg.mean()
        self.extras["log"]["VOC/kp_pos_min"]  = self._voc_kp_pos_seg.min()
        self.extras["log"]["VOC/kp_pos_max"]  = self._voc_kp_pos_seg.max()
        self.extras["log"]["VOC/kp_rot_mean"] = self._voc_kp_rot_seg.mean()
        self.extras["log"]["VOC/n_active_segments"] = (self._voc_kp_pos_seg > 0.0).sum().float()
        # Curriculum α (used by mode BC; 1.0 if disabled).
        self.extras["log"]["Curriculum/alpha"] = torch.tensor(self._curriculum_alpha(), device=self.device)

        # Per-segment recent trailing means and current gain. If seg2_task_mean plateaus
        # low while others are at 0.8, that segment is the bottleneck — address with
        # kernel widening, focused RSI, etc. without touching global thresholds.
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

        # Focused-RSI sampling distribution (computed identically to _reset_idx so the
        # logged probs match what's actually sampled). Useful for watching the curriculum
        # shift: lift segment's prob should start uniform and grow toward 1.0 as other
        # segments decay (their weights drop and lift dominates by relative growth).
        if self.cfg.reset_segment_focus_prob > 0 and self._voc_n_segments > 1:
            w_log = self._voc_kp_pos_seg.clamp(min=0) ** float(self.cfg.segment_focus_beta)
            if w_log.sum().item() <= 0.0:
                w_log = torch.ones_like(w_log)
            probs_log = w_log / w_log.sum()
            for s in range(self._voc_n_segments):
                self.extras["log"][f"RSI/seg{s}_focus_prob"] = probs_log[s]

        # Update prev variables (action_rate already read above)
        self.prev_actions[:] = self.actions[:]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

    def _voc_decay_check(self):
        """Per-segment VOC decay. Each segment passes its OWN sample-count + threshold
        gates and decays its own gains independently. Easy segments (free motion, no
        contact) clear thresholds early and lose assist quickly; hard segments (contact)
        keep assist until they're actually learned. Below `voc_kp_min` a segment's gains
        snap to zero — _apply_voc then short-circuits the per-env wrench wherever that
        env's current segment is zero."""
        if self.common_step_counter < self.cfg.voc_decay_warmup_steps:
            return
        min_samples = self.cfg.voc_reward_window_size // 2
        phi_p = self.cfg.voc_decay_phi_p
        phi_v = self.cfg.voc_decay_phi_v
        kp_min = self.cfg.voc_kp_min

        # Per-segment independent decay. Each segment passes its OWN sample-count gate
        # and its own threshold gate. Segments that pass get their gain multiplied by
        # phi; others are unchanged. Below kp_min the segment's gains snap to zero.
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
        # Snap per-segment kp below the floor to zero (and zero the matching kv); `_apply_voc`
        # then short-circuits the per-env wrench wherever that env's current segment is zero.
        below = self._voc_kp_pos_seg < kp_min
        if bool(below.any().item()):
            self._voc_kp_pos_seg[below] = 0.0
            self._voc_kp_rot_seg[below] = 0.0
            self._voc_kv_pos_seg[below] = 0.0
            self._voc_kv_rot_seg[below] = 0.0
        self._save_voc_state()

    def _save_voc_state(self):
        """Persist current per-segment VOC runtime gains to <log_dir>/voc_state.npz so
        play.py / record.py can resume eval at the trained-end VOC level (via --keep_voc)
        instead of the cfg's initial values. Saves as (N_s,) arrays — apply_voc_state on
        load handles both array and scalar legacy formats. Best-effort write."""
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

    def apply_voc_state(self, state) -> None:
        """Restore VOC gains from a saved `voc_state.npz` (dict-like with `voc_kp_pos`,
        `voc_kp_rot`, `voc_kv_pos`, `voc_kv_rot`). Accepts both formats:
          - scalar (legacy single-VOC) → broadcast across all per-seg entries;
          - (N_s,) array (per-segment) → assign element-wise. Pool size at restore time
            must match saved N_s; otherwise the array is broadcast via mean with a warning
            so loading doesn't hard-fail across re-segmentations.
        Called by play.py / record.py when --keep_voc is passed."""
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
        """Per-arm error between actual and reference EE pose expressed in the box's frame
        ("keep the EE at the same offset from the box as the planner expected"). Caller
        applies the gate and reward kernel.

        Returns (pos_err_l, quat_err_l, pos_err_r, quat_err_r), each (num_envs,).
        """
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
            self.EE_poses_l[idx, :3], self.EE_poses_l[idx, 3:],
            EE_pos_lr[:, :3], EE_quat_lr[:, :4],
        )
        pos_err_r, quat_err_r = _per_arm(
            self.EE_poses_r[idx, :3], self.EE_poses_r[idx, 3:],
            EE_pos_lr[:, 3:], EE_quat_lr[:, 4:],
        )
        return pos_err_l, quat_err_l, pos_err_r, quat_err_r

    
    # === VOC gain properties (back-compat shim over the per-seg tensors) ===
    # Getter returns the mean across segments — equals the legacy scalar when all
    # segments share a value (e.g. at init). Setter broadcasts a scalar to every
    # segment, so existing external code like `env.voc_kp_pos = 0.0` (play.py /
    # record.py / ur_rtde_real_time.py) keeps working unchanged.
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
    
    def _compute_proximity_penalty(self) -> torch.Tensor:
        """Penalize links approaching illegal contact surfaces based on PhysX separation distance.

        Buffer-overflow handling: PhysX caps the per-prim contact buffer at
        `max_contact_data_count_per_prim` (set in `illegal_contact_sensor_cfgs`). When
        more contacts occur than fit, `separation` is truncated to the buffer size but
        `contact_count_per_link` still reports the *real* count. The resulting length
        mismatch crashes `index_reduce_` (expected indices == source length). We clamp
        `env_ids` to the actual separation length so the overflow contacts are silently
        dropped from the penalty — the alternative is to crash mid-training. If you see
        the `[proximity-overflow]` warning a lot, bump `max_contact_data_count_per_prim`.
        """
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
                    print(f"[proximity-overflow] {name} sensor: {total_count} contacts > "
                          f"buffer {n_buf} (max_contact_data_count_per_prim*num_envs). "
                          f"Overflow contacts dropped from the proximity penalty. Bump "
                          f"`max_contact_data_count_per_prim` if you want full coverage.")
                    self._proximity_overflow_warned = True
                total_count = n_buf
            separation = separation[:total_count, 0]
            contact_count_per_env = contact_count_per_link.sum(dim=-1)
            env_ids = torch.repeat_interleave(
                torch.arange(self.num_envs, device=self.device), contact_count_per_env
            )
            # If the buffer overflowed, env_ids reflects the real per-env counts and is
            # longer than the truncated separation. Drop the tail of env_ids to match.
            if env_ids.shape[0] > total_count:
                env_ids = env_ids[:total_count]
            min_sep = torch.full((self.num_envs,), self.cfg.max_proximity * 2, device=self.device)
            min_sep.index_reduce_(0, env_ids, separation, reduce='amin', include_self=True)
            proximity = torch.clamp(1.0 - min_sep / self.cfg.max_proximity, min=0.0)
            penalty += proximity.square()
        return self.cfg.w_proximity_to_contact * penalty

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

