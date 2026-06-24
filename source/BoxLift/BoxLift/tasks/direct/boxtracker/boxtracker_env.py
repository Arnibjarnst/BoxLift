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

from BoxLift.tasks.direct.boxtracker.boxtracker_env_cfg import *

from isaaclab.utils.math import quat_apply, quat_mul, quat_inv, quat_error_magnitude

class BoxtrackerEnv(DirectRLEnv):
    cfg: BoxtrackerEnvCfg

    def __init__(self, cfg: BoxtrackerEnvCfg, render_mode: str | None = None, **kwargs):
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

        # initialized to 0 so the first _get_observations after env init doesn't advance
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.dphase = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_phase = torch.zeros(self.num_envs, device=self.device)
        self.cumulative_slowdown = torch.zeros(self.num_envs, device=self.device)
        # init to 0 so step-1 improvement-gate delta clamps to 0 (no spurious discount)
        self._err_task_prev = torch.zeros(self.num_envs, device=self.device)
        self._err_track_prev = torch.zeros(self.num_envs, device=self.device)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)

        T = self._T_max
        self._segment_size = max(1, int(round(self.cfg.phase_segment_s / self.dt)))
        self._num_segments = max(1, (T - 1 + self._segment_size - 1) // self._segment_size)
        self.segment_scores = torch.full(
            (self._num_segments,), 0.5, device=self.device, dtype=torch.float32
        )

    def _load_segment_pool(self):
        """Load trajectory segments from cfg.trajectory_path (single file) or cfg.dataset_path (folder).
        Variable-length segments are right-padded; robot base / object props come from the first segment."""
        import glob
        traj_path    = (self.cfg.trajectory_path or "").strip()
        dataset_path = (self.cfg.dataset_path    or "").strip()
        if traj_path and dataset_path:
            print(f"[boxtracker] both trajectory_path and dataset_path are set; "
                  f"using trajectory_path={traj_path!r} (dataset_path ignored).")
        if traj_path:
            seg_files = [traj_path]
            if not os.path.isfile(traj_path):
                raise FileNotFoundError(
                    f"trajectory_path={traj_path!r} does not exist or is not a file.")
        else:
            seg_files = sorted(glob.glob(os.path.join(dataset_path, "*.npz")))
            if not seg_files:
                raise FileNotFoundError(
                    f"No *.npz found under dataset_path={dataset_path!r}. "
                    f"Pass env.dataset_path=<folder> or env.trajectory_path=<file>.")

        keys_ts = ["obj_poses", "obj_vel", "joints", "joint_vel",
                   "joints_target", "EE_poses"]
        raw = []
        lengths = []
        has_contact = True
        for fp in seg_files:
            d = np.load(fp)
            T = d["obj_poses"].shape[0]
            lengths.append(T)
            entry = {k: d[k].astype(np.float32) for k in keys_ts}
            entry["in_contact"] = (d["in_contact"].astype(np.float32)
                                   if "in_contact" in d.files else None)
            if entry["in_contact"] is None:
                has_contact = False
            raw.append(entry)
        S = len(raw)
        T_max = int(max(lengths))
        self._T_max = T_max
        self._n_pool = S
        self.segment_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)

        dims = {"obj_poses": 7, "obj_vel": 6, "joints": 6, "joint_vel": 6,
                "joints_target": 6, "EE_poses": 7}
        pool = {k: torch.zeros((S, T_max, dims[k]), device=self.device) for k in keys_ts}
        gate = torch.zeros((S, T_max), device=self.device)  # 1 where box in contact/moving
        for s, entry in enumerate(raw):
            L = lengths[s]
            for k in keys_ts:
                a = torch.from_numpy(entry[k]).to(self.device)
                pool[k][s, :L] = a
                if L < T_max:
                    pool[k][s, L:] = a[-1]                       # replicate last frame
            if has_contact:
                ic = torch.from_numpy(entry["in_contact"]).to(self.device)
                gate[s, :L] = ic
                if L < T_max:
                    gate[s, L:] = ic[-1]
            else:
                v = entry  # derive from obj_vel like boxhinge
                ov = pool["obj_vel"][s]
                gate[s] = ((ov[:, :3].norm(dim=-1) + ov[:, 3:].norm(dim=-1))
                           > self.cfg.eef_box_gate_obj_vel_eps).float()

        if self.cfg.eef_box_gate_dilation_steps > 0:
            kk = 2 * int(self.cfg.eef_box_gate_dilation_steps) + 1
            gate = torch.nn.functional.max_pool1d(
                gate.unsqueeze(1), kernel_size=kk, stride=1,
                padding=int(self.cfg.eef_box_gate_dilation_steps),
            ).squeeze(1)
        self._pool = pool
        self._pool_gate = gate.bool()                            # (S, T_max)

        d0 = np.load(seg_files[0])
        self.arm_pose = torch.from_numpy(d0["arm_pose"]).float().to(self.device)
        self.dt = float(d0["dt"])
        if "object_dims" in d0.files:
            self.cfg.object_dims = tuple(d0["object_dims"].tolist())
            self.cfg.cube_cfg.spawn.size = self.cfg.object_dims
        if "object_mass" in d0.files:
            self.cfg.object_mass = float(d0["object_mass"])
            self.cfg.cube_cfg.spawn.mass_props.mass = self.cfg.object_mass

    def _setup_scene(self):
        self._load_segment_pool()

        self.cur_seg_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.segment_length = self.segment_lengths[self.cur_seg_idx].clone()
        self.obj_poses     = self._pool["obj_poses"][self.cur_seg_idx].clone()
        self.obj_vel       = self._pool["obj_vel"][self.cur_seg_idx].clone()
        self.joints        = self._pool["joints"][self.cur_seg_idx].clone()
        self.joint_vel     = self._pool["joint_vel"][self.cur_seg_idx].clone()
        self.joints_target = self._pool["joints_target"][self.cur_seg_idx].clone()
        self.EE_poses      = self._pool["EE_poses"][self.cur_seg_idx].clone()
        self.eef_box_gate_mask = self._pool_gate[self.cur_seg_idx].clone()  # (N, T_max)

        traj_duration = self.dt * (self._T_max - 1)
        if self.cfg.max_episode_steps > 0:
            self.cfg.episode_length_s = self.cfg.max_episode_steps * self.dt
        elif self.cfg.enable_phase_slowdown:
            self.cfg.episode_length_s = traj_duration * self.cfg.max_slowdown_multiplier
        else:
            self.cfg.episode_length_s = traj_duration
        self.cfg.episode_length_s += self.cfg.post_traj_hold_s

        ur5e_cfg = get_ur5e_cfg(self.cfg.ur5e_prim_path, self.arm_pose, self.cfg)
        self.ur5e = Articulation(ur5e_cfg)

        # Disable collision on base_link (ground is raised 1.8cm for the box pushing surface,
        # but the robot base sits at the original level — only the area near the box is elevated)
        modify_collision_properties(
            "/World/envs/env_0/ur5e/base_link",
            sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        )

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0,0,-0.5))

        self.object = RigidObject(cfg=self.cfg.cube_cfg)
        self.table = RigidObject(cfg=self.cfg.table_cfg)

        self.illegal_contact_sensors = {name: ContactSensor(cfg) for name, cfg in self.cfg.illegal_contact_sensor_cfgs.items()}
        self.ee_contact_sensor = ContactSensor(self.cfg.ee_contact_sensor_cfg)

        # RSI not used; always starts from segment beginning
        self.rsi_valid_phases = torch.zeros(1, dtype=torch.long, device=self.device)

        self.prev_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self.prev_joint_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self._dr_obj_mass = torch.zeros((self.num_envs, 1), device=self.device)
        self._dr_obj_friction = torch.zeros((self.num_envs, 3), device=self.device)

        self.obs_history = torch.zeros(
            (self.num_envs, self.cfg.obs_history_steps, self.cfg.per_step_feature_dim),
            device=self.device,
        )

        # delay ring: index 0 = oldest (exactly delay_steps ago); fires latch into last_*
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

        mass = float(self.cfg.cube_cfg.spawn.mass_props.mass)
        d_max = float(max(self.cfg.cube_cfg.spawn.size))
        inertia_est = mass * (d_max ** 2) / 12.0
        kp_pos_init = float(self.cfg.voc_kp_pos)
        kp_rot_init = float(self.cfg.voc_kp_rot)
        kv_pos_init = self.cfg.voc_kv_pos_scale * (kp_pos_init * mass) ** 0.5
        kv_rot_init = self.cfg.voc_kv_rot_scale * (kp_rot_init * inertia_est) ** 0.5
        S = self._n_pool
        self._voc_kp_pos_seg = torch.full((S,), kp_pos_init, device=self.device)
        self._voc_kp_rot_seg = torch.full((S,), kp_rot_init, device=self.device)
        self._voc_kv_pos_seg = torch.full((S,), kv_pos_init, device=self.device)
        self._voc_kv_rot_seg = torch.full((S,), kv_rot_init, device=self.device)
        self._voc_ep_rew_task = torch.zeros(self.num_envs, device=self.device)
        self._voc_ep_rew_track = torch.zeros(self.num_envs, device=self.device)
        self._voc_ep_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        W = self.cfg.voc_reward_window_size
        if self.cfg.voc_per_segment:
            self._voc_buf_task_seg = torch.full((S, W), float("nan"), device=self.device)
            self._voc_buf_track_seg = torch.full((S, W), float("nan"), device=self.device)
            self._voc_buf_idx_seg = torch.zeros(S, dtype=torch.long, device=self.device)
        else:
            self._voc_buf_task = torch.full((W,), float("nan"), device=self.device)
            self._voc_buf_track = torch.full((W,), float("nan"), device=self.device)
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
        """Apply random external forces to the forearm to improve robustness."""
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

    def _seg_idx(self, phase: torch.Tensor, env_sel: torch.Tensor | None = None):
        """Per-env (i0, i1, a) for the active per-env trajectory pool.

        phase: (M,) float index into each env's own segment. env_sel: optional (M,)
        env indices when phase covers a subset of envs. Clamps to each env's own
        length (segment_length-1) so padded frames past the segment end are never
        read. Returns i0,i1 (M,) long and a (M,1) lerp weight.
        """
        seg_len = self.segment_length if env_sel is None else self.segment_length[env_sel]
        last = (seg_len - 1).clamp(min=0)                          # (M,) last valid idx
        p = torch.minimum(phase.clamp(min=0.0), last.float())      # (M,)
        i0 = p.floor().long().clamp(min=0)
        i0 = torch.minimum(i0, last)
        i1 = torch.minimum(i0 + 1, last)
        a = (p - i0.float()).unsqueeze(-1)                         # (M,1)
        return i0, i1, a

    def _interp(self, traj: torch.Tensor, phase: torch.Tensor | None = None,
                env_sel: torch.Tensor | None = None) -> torch.Tensor:
        """Linear interpolation along each env's own segment.

        traj: per-env pool view (num_envs, T_max, D). phase: (M,) float index
        (defaults to self.phase, M=num_envs). env_sel: optional (M,) env indices
        when phase covers only a subset of envs (e.g. tracker fires). Returns (M, D).
        """
        if phase is None:
            phase = self.phase
        i0, i1, a = self._seg_idx(phase, env_sel)
        rows = torch.arange(traj.shape[0], device=traj.device) if env_sel is None else env_sel
        return (1.0 - a) * traj[rows, i0] + a * traj[rows, i1]

    def _nlerp(self, traj_quat: torch.Tensor, phase: torch.Tensor | None = None,
               env_sel: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized lerp for unit quaternions (wxyz), per-env segment.
        Hemisphere-corrected for short path."""
        if phase is None:
            phase = self.phase
        i0, i1, a = self._seg_idx(phase, env_sel)
        rows = torch.arange(traj_quat.shape[0], device=traj_quat.device) if env_sel is None else env_sel
        q0 = traj_quat[rows, i0]
        q1 = traj_quat[rows, i1]
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

        EE_pos_ref   = self._interp(self.EE_poses[..., :3])
        obj_pos_ref  = self._interp(self.obj_poses[..., :3])
        obj_quat_ref = self._nlerp(self.obj_poses[..., 3:])
        rel_ref_pos_box = quat_apply(quat_inv(obj_quat_ref), EE_pos_ref - obj_pos_ref)
        obj_pos_actual  = self._get_obj_pos(relative=False)
        obj_quat_actual = self._get_obj_quat(relative=False)
        target_EE_pos_world = obj_pos_actual + quat_apply(obj_quat_actual, rel_ref_pos_box)
        self.ee_markers.visualize(translations=target_EE_pos_world + self.scene.env_origins)

        obj_pos = self._interp(self.obj_poses[..., :3]) + self.scene.env_origins
        obj_quat = self._nlerp(self.obj_poses[..., 3:])
        self.cube_marker.visualize(translations=obj_pos, orientations=obj_quat)

        # Cache once per policy step so C/D modes use q_current at decision time, not per-substep.
        self._cached_joint_target = self.get_joint_targets().clamp(
            self.ur5e.data.joint_pos_limits[..., 0],
            self.ur5e.data.joint_pos_limits[..., 1],
        )

    def _apply_voc(self):
        """DexMachina VOC: PD wrench drives the box along the reference; gains decay as the policy improves."""
        if not self.cfg.voc_enabled or float(self._voc_kp_pos_seg.max().item()) <= 0.0:
            n = self.num_envs
            self.object.set_external_force_and_torque(
                torch.zeros(n, 1, 3, device=self.device),
                torch.zeros(n, 1, 3, device=self.device),
                is_global=True,
            )
            return

        ref_pos = self._interp(self.obj_poses[..., :3])
        ref_quat = self._nlerp(self.obj_poses[..., 3:])
        ref_vel = self._interp(self.obj_vel)

        obj_pos = self.object.data.root_pos_w - self.scene.env_origins
        obj_quat = self.object.data.root_quat_w
        obj_vel = self.object.data.root_vel_w

        kp_pos = self._voc_kp_pos_seg[self.cur_seg_idx].unsqueeze(-1)
        kv_pos = self._voc_kv_pos_seg[self.cur_seg_idx].unsqueeze(-1)
        kp_rot = self._voc_kp_rot_seg[self.cur_seg_idx].unsqueeze(-1)
        kv_rot = self._voc_kv_rot_seg[self.cur_seg_idx].unsqueeze(-1)

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


    def get_joint_targets_A(self):
        """Residual on planner targets. Planner feedforward is baked into the action."""
        return self._interp(self.joints_target) + self._action_scale * self.actions[:, :6]

    def get_joint_targets_B(self):
        """Residual on trajectory positions. Pair with feedforward obs."""
        return self._interp(self.joints) + self._action_scale * self.actions[:, :6]

    def get_joint_targets_C(self):
        """Residual on current joint positions. Pair with feedforward obs."""
        return self._get_joint_pos() + self._action_scale * self.actions[:, :6]

    def get_joint_targets_BC(self):
        """Linear blend between mode B (ref position) and mode C (current position) via curriculum α."""
        alpha = self._bc_alpha()
        ref_pos = self._interp(self.joints)
        cur_pos = self._get_joint_pos()
        blend = (1.0 - alpha) * ref_pos + alpha * cur_pos
        return blend + self._action_scale * self.actions[:, :6]

    def get_joint_targets_D(self):
        """Planner PD error from current position + scaled residual, blended by curriculum α."""
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
        if mode == "BC":
            return self.get_joint_targets_BC()
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
        """Tracker reading with fixed delay, sub-50Hz update rate, per-fire noise, and per-episode bias."""
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
            past_ref_pos  = self._interp(self.obj_poses[..., :3], phase=sampled_phase, env_sel=fires_idx)
            past_ref_quat = self._nlerp(self.obj_poses[..., 3:], phase=sampled_phase, env_sel=fires_idx)
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

    def _get_observations_agnostic(self) -> dict:
        """Reference-agnostic obs: absolute-only per-step features + goal conditioning.

        Drops all reference-relative terms (relative_q, relative_obj, future waypoints).
        Appends a 14-dim goal obs: rel_pos(3) + rel_quat(4) + abs_pos(3) + abs_quat(4),
        where the goal is the trajectory's final object pose for each env.
        """
        feature_parts = [
            self._get_joint_pos(relative=False),
            self._get_joint_vel(relative=False),
        ]

        obj_abs_pos = obj_abs_quat = None
        if self.cfg.include_object_obs:
            _, _, obj_abs_pos, obj_abs_quat = self._get_noisy_obj_obs()
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

        if self.cfg.include_ee_box_obs:
            EE_pos_a  = self._get_EE_pos(relative=False)
            EE_quat_a = self._get_EE_quat(relative=False)
            if obj_abs_pos is not None:
                cur_obj_pos, cur_obj_quat = obj_abs_pos, obj_abs_quat
            else:
                cur_obj_pos  = self._get_obj_pos(relative=False)
                cur_obj_quat = self._get_obj_quat(relative=False)
            inv_obj = quat_inv(cur_obj_quat)
            feature_parts.append(quat_apply(inv_obj, EE_pos_a - cur_obj_pos))  # abs EE-in-box pos (3)
            feature_parts.append(quat_mul(inv_obj, EE_quat_a))                 # abs EE-in-box quat (4)

        current_features = torch.cat(feature_parts, dim=-1)
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = current_features
        obs_parts = [self.obs_history.flatten(start_dim=1)]

        # Goal conditioning: each env's trajectory end-pose, relative to current obj + absolute.
        env_arange = torch.arange(self.num_envs, device=self.device)
        goal_idx  = (self.segment_length - 1).clamp(min=0)
        goal_pos  = self.obj_poses[env_arange, goal_idx, :3]   # (N, 3) env-frame
        goal_quat = self.obj_poses[env_arange, goal_idx, 3:]   # (N, 4) wxyz
        if obj_abs_pos is not None:
            cur_pos, cur_quat = obj_abs_pos, obj_abs_quat
        else:
            cur_pos  = self._get_obj_pos(relative=False)
            cur_quat = self._get_obj_quat(relative=False)
        obs_parts.extend([
            goal_pos - cur_pos,                        # rel goal pos  (3)
            quat_mul(goal_quat, quat_inv(cur_quat)),   # rel goal quat (4)
            goal_pos,                                  # abs goal pos  (3)
            goal_quat,                                 # abs goal quat (4)
        ])

        if self.cfg.include_prev_actions:
            obs_parts.append(self.prev_actions)

        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs, "privileged": self._get_privileged_obs()}

    def _get_observations(self) -> dict:
        last = (self.segment_length - 1).clamp(min=0).float()
        self.phase = torch.minimum((self.phase + self.dphase).clamp(min=0.0), last)

        if self.cfg.reference_agnostic:
            return self._get_observations_agnostic()

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

        if self.cfg.include_ee_box_obs:
            EE_pos_a  = self._get_EE_pos(relative=False)
            EE_quat_a = self._get_EE_quat(relative=False)
            obj_pos_a = self._get_obj_pos(relative=False)
            obj_quat_a = self._get_obj_quat(relative=False)
            inv_obj_a = quat_inv(obj_quat_a)
            act_pos_box  = quat_apply(inv_obj_a, EE_pos_a - obj_pos_a)
            act_quat_box = quat_mul(inv_obj_a, EE_quat_a)

            EE_pos_r  = self._interp(self.EE_poses[..., :3])
            EE_quat_r = self._nlerp(self.EE_poses[..., 3:])
            obj_pos_r = self._interp(self.obj_poses[..., :3])
            obj_quat_r = self._nlerp(self.obj_poses[..., 3:])
            inv_obj_r = quat_inv(obj_quat_r)
            ref_pos_box  = quat_apply(inv_obj_r, EE_pos_r - obj_pos_r)
            ref_quat_box = quat_mul(inv_obj_r, EE_quat_r)

            feature_parts.append(act_pos_box - ref_pos_box)
            feature_parts.append(quat_mul(ref_quat_box, quat_inv(act_quat_box)))
            if self.cfg.include_absolute_obs:
                feature_parts.append(act_pos_box)
                feature_parts.append(act_quat_box)

        current_features = torch.cat(feature_parts, dim=-1)

        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1] = current_features

        obs_parts = [self.obs_history.flatten(start_dim=1)]

        if self.cfg.future_obs_steps:
            cur_pos = self._interp(self.obj_poses[..., :3])
            cur_quat = self._nlerp(self.obj_poses[..., 3:])
            inv_cur_quat = quat_inv(cur_quat)
            cur_joints = self._interp(self.joints) if self.cfg.future_obs_joints else None
            futures = []
            for k in self.cfg.future_obs_steps:
                fut_phase = self.phase + float(k)
                fut_pos = self._interp(self.obj_poses[..., :3], phase=fut_phase)
                fut_quat = self._nlerp(self.obj_poses[..., 3:], phase=fut_phase)
                futures.append(fut_pos - cur_pos)
                futures.append(quat_mul(fut_quat, inv_cur_quat))
                if self.cfg.future_obs_absolute:
                    futures.append(fut_pos)
                    futures.append(fut_quat)
                if self.cfg.future_obs_joints:
                    fut_joints = self._interp(self.joints, phase=fut_phase)
                    futures.append(fut_joints - cur_joints)
                    if self.cfg.future_obs_absolute:
                        futures.append(fut_joints)
            obs_parts.append(torch.cat(futures, dim=-1))

        if self.cfg.include_prev_actions:
            obs_parts.append(self.prev_actions)

        obs = torch.cat(obs_parts, dim=-1)
        return {"policy": obs, "privileged": self._get_privileged_obs()}

    def _get_privileged_obs(self) -> torch.Tensor:
        clean_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins
        clean_quat = self.object.data.root_quat_w.clone()
        clean_vel = self.object.data.root_vel_w.clone()
        obj_block = torch.cat([clean_pos, clean_quat, clean_vel], dim=-1)

        stiff = self.ur5e.data.joint_stiffness
        damp  = self.ur5e.data.joint_damping
        dr_block = torch.cat([self._dr_obj_mass, self._dr_obj_friction, stiff, damp], dim=-1)

        ref_obj_pos  = self._interp(self.obj_poses[..., :3])
        ref_obj_quat = self._nlerp(self.obj_poses[..., 3:])
        ref_obj_vel  = self._interp(self.obj_vel)
        ref_joints      = self._interp(self.joints)
        ref_joint_vel   = self._interp(self.joint_vel)
        ref_joints_tgt  = self._interp(self.joints_target)
        planner_pd      = ref_joints_tgt - ref_joints
        ref_EE = torch.cat([self._interp(self.EE_poses[..., :3]), self._nlerp(self.EE_poses[..., 3:])], dim=-1)
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

        kp_pos_t = self._voc_kp_pos_seg[self.cur_seg_idx].unsqueeze(-1)
        kp_rot_t = self._voc_kp_rot_seg[self.cur_seg_idx].unsqueeze(-1)
        alpha_t  = torch.full((self.num_envs, 1), self._curriculum_alpha(), device=self.device)
        phase_norm = (self.phase / max(self._T_max - 1, 1)).unsqueeze(-1)
        voc_block = torch.cat([kp_pos_t, kp_rot_t, alpha_t, phase_norm], dim=-1)

        base = torch.cat([obj_block, dr_block, ref_block, force_block, eef_block, voc_block], dim=-1)

        if not self.cfg.future_obs_steps or self.cfg.reference_agnostic or not self.cfg.future_obs_in_privileged:
            return base

        cur_pos = self._interp(self.obj_poses[..., :3])
        cur_quat = self._nlerp(self.obj_poses[..., 3:])
        inv_cur_quat = quat_inv(cur_quat)
        cur_joints = self._interp(self.joints) if self.cfg.future_obs_joints else None
        fut_parts = []
        for k in self.cfg.future_obs_steps:
            fut_phase = self.phase + float(k)
            fut_pos = self._interp(self.obj_poses[..., :3], phase=fut_phase)
            fut_quat = self._nlerp(self.obj_poses[..., 3:], phase=fut_phase)
            fut_parts.append(fut_pos - cur_pos)
            fut_parts.append(quat_mul(fut_quat, inv_cur_quat))
            if self.cfg.future_obs_absolute:
                fut_parts.append(fut_pos)
                fut_parts.append(fut_quat)
            if self.cfg.future_obs_joints:
                fut_joints = self._interp(self.joints, phase=fut_phase)
                fut_parts.append(fut_joints - cur_joints)
                if self.cfg.future_obs_absolute:
                    fut_parts.append(fut_joints)
        return torch.cat([base, *fut_parts], dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        at_end = self.phase >= (self.segment_length - 1).clamp(min=0).float() - 1e-3
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
        """EMA failure-rate update per traversed segment; feeds into weighted phase resampling."""
        f = self.reset_terminated[env_ids].float()
        start_seg = (self.episode_start_phase[env_ids].long() // self._segment_size).clamp(max=self._num_segments - 1)
        end_seg   = (self.phase[env_ids].long()              // self._segment_size).clamp(max=self._num_segments - 1)

        seg_idx = torch.arange(self._num_segments, device=self.device).unsqueeze(0)
        traversed = (seg_idx >= start_seg.unsqueeze(1)) & (seg_idx <= end_seg.unsqueeze(1))

        count = traversed.float().sum(dim=0)
        fails = (traversed.float() * f.unsqueeze(1)).sum(dim=0)

        fail_rate = fails / count.clamp(min=1)
        update_w = (count > 0).float() * self.cfg.phase_resample_alpha
        self.segment_scores = (1.0 - update_w) * self.segment_scores + update_w * fail_rate
        lo, hi = self.cfg.phase_resample_clamp
        self.segment_scores.clamp_(lo, hi)

    def _max_start_phase(self, T: int) -> float:
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

        eids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = eids.numel()

        old_seg = self.cur_seg_idx[eids].clone()

        # Sample a new random segment per resetting env and refresh that env's per-env
        # trajectory views + length + reward gate from the pool.
        new_seg = torch.randint(0, self._n_pool, (n,), device=self.device)
        self.cur_seg_idx[eids]        = new_seg
        self.segment_length[eids]     = self.segment_lengths[new_seg]
        self.obj_poses[eids]          = self._pool["obj_poses"][new_seg]
        self.obj_vel[eids]            = self._pool["obj_vel"][new_seg]
        self.joints[eids]             = self._pool["joints"][new_seg]
        self.joint_vel[eids]          = self._pool["joint_vel"][new_seg]
        self.joints_target[eids]      = self._pool["joints_target"][new_seg]
        self.EE_poses[eids]           = self._pool["EE_poses"][new_seg]
        self.eef_box_gate_mask[eids]  = self._pool_gate[new_seg]

        if fixed_value is not None:
            self.phase[eids] = float(fixed_value)
        else:
            T_per_env = self.segment_length[eids].float()            # (n,) per-env lengths
            if self.cfg.max_episode_steps > 0:
                max_phase = (T_per_env - 1 - self.cfg.max_episode_steps).clamp(min=1.0)
            else:
                max_phase = (T_per_env - 2).clamp(min=0.0)
            sampled = (torch.rand(n, device=self.device) * max_phase).floor()
            if self.cfg.enable_failure_resampling:
                T_scalar = int(T_per_env.float().mean().item())
                sampled = self._sample_phase_failure_weighted(n, T_scalar)
            if self.cfg.reset_to_zero_prob > 0.0:
                force_zero = torch.rand(n, device=self.device) < self.cfg.reset_to_zero_prob
                sampled = torch.where(force_zero, torch.zeros_like(sampled), sampled)
            self.phase[eids] = sampled
        self.episode_start_phase[eids] = self.phase[eids]
        self.dphase[eids] = 0.0

        idx = self.phase[eids].floor().long()
        idx = torch.minimum(idx, (self.segment_length[eids] - 1).clamp(min=0))
        initial_joint_pos = self.joints[eids, idx].clone()
        initial_joint_vel = self.joint_vel[eids, idx].clone()
        pos_noise_std = torch.as_tensor(self.cfg.reset_joint_pos_noise, device=self.device, dtype=torch.float32)
        vel_noise_std = torch.as_tensor(self.cfg.reset_joint_vel_noise, device=self.device, dtype=torch.float32)
        initial_joint_pos += pos_noise_std * torch.randn_like(initial_joint_pos)
        initial_joint_vel += vel_noise_std * torch.randn_like(initial_joint_vel)
        self.ur5e.write_joint_state_to_sim(initial_joint_pos, initial_joint_vel, env_ids=env_ids)

        initial_object_pose = self.obj_poses[eids, idx].clone()
        initial_object_pose[:, :3] += self.scene.env_origins[env_ids]
        initial_object_vel = self.obj_vel[eids, idx].clone()

        n = len(env_ids)
        initial_object_pose[:, 0:2] += self.cfg.reset_obj_pos_xy_noise * torch.randn(n, 2, device=self.device)
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
        self.obj_obs_last_rel[env_ids, 3]  = 1.0
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

        # Must run BEFORE we zero the per-env episode trackers below.
        n = len(env_ids)
        ep_steps = self._voc_ep_steps[env_ids].clamp(min=1).float()
        norm_task = self._voc_ep_rew_task[env_ids] / ep_steps
        norm_track = self._voc_ep_rew_track[env_ids] / ep_steps
        W = self.cfg.voc_reward_window_size
        if self.cfg.voc_per_segment:
            S = self._n_pool
            seg_one_hot = torch.zeros((n, S), device=self.device)
            seg_one_hot[torch.arange(n, device=self.device), old_seg] = 1.0
            within_idx = (seg_one_hot.cumsum(dim=0) - 1.0)[
                torch.arange(n, device=self.device), old_seg
            ].long()
            slots = (self._voc_buf_idx_seg[old_seg] + within_idx) % W
            self._voc_buf_task_seg[old_seg, slots] = norm_task
            self._voc_buf_track_seg[old_seg, slots] = norm_track
            inc = torch.zeros(S, dtype=torch.long, device=self.device)
            inc.scatter_add_(0, old_seg, torch.ones(n, dtype=torch.long, device=self.device))
            self._voc_buf_idx_seg = (self._voc_buf_idx_seg + inc) % W
        else:
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
                print(f"[boxtracker] _reset_idx: DR cache readback failed: {e!r}.")
                self._dr_readback_warned = True

    def _bc_alpha(self) -> float:
        if self.cfg.bc_end_steps <= 0:
            return 0.0
        t = self.common_step_counter
        if t <= self.cfg.bc_start_steps:
            return 0.0
        if t >= self.cfg.bc_end_steps:
            return 1.0
        return (t - self.cfg.bc_start_steps) / (self.cfg.bc_end_steps - self.cfg.bc_start_steps)

    def _curriculum_alpha(self) -> float:
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
        if self.cfg.action_mode == "BC":
            reward_alpha = self._bc_alpha()
        else:
            reward_alpha = alpha
        w_task_eff  = self.cfg.w_task_start  + (self.cfg.w_task  - self.cfg.w_task_start)  * reward_alpha
        w_track_eff = self.cfg.w_track_start + (self.cfg.w_track - self.cfg.w_track_start) * reward_alpha

        rew_task = w_task_eff * rew_task_unweighted
        if self.cfg.task_scale_by_dphase:
            rew_task = self.dphase * rew_task

        N = self.eef_box_gate_mask.shape[0]
        last = (self.segment_length - 1).clamp(min=0)
        phase_idx = torch.minimum(self.phase.floor().long().clamp(min=0), last)
        gate = self.eef_box_gate_mask[torch.arange(N, device=self.device), phase_idx].float()
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

        # Three slowdown penalties (min-gate / improvement-gate / cumulative); all inside
        # w_regularization, zero at dphase=1. Formulas + constraints in env_cfg.
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

            # Improvement gate uses raw sigma-normalized error (not kernel: kernel Δr→0
            # far from ref, exactly when we need the signal). err_track phase-exclusive.
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

            # Store this step's errors as "prev" for next step's Δ.
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
            rew_joint_acc + rew_torque + rew_action_rate + rew_action_norm + rew_joint_limit + rew_illegal_contact + rew_proximity + rew_flange_forearm_dist + rew_total_slowdown + rew_slowdown_gated + rew_slowdown_improvement
        )

        rew_completion = self.cfg.w_completion * self.reset_time_outs.float()

        self.extras["log"] = {
            "Curriculum/alpha": torch.tensor(alpha, device=self.device),
            "Curriculum/bc_alpha": torch.tensor(self._bc_alpha(), device=self.device),
            "Curriculum/reward_alpha": torch.tensor(reward_alpha, device=self.device),
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
            "Error/EE_quat_error": eef_quat_error.mean(),
            # Relative EE-in-box-frame errors. Two views: gated mean (denominator = active
            # envs only) is the meaningful "tracking error during contact" signal; raw mean
            # includes non-gated envs (where this term is irrelevant) and is mostly there to
            # confirm the gate isn't always 0.
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
            "Phase/mean_phase_norm": (self.phase / (self.segment_length - 1).clamp(min=1).float()).mean(),
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

        # Per-env extras for record.py / rollout_summary. Only populated when
        # cfg.emit_per_env_extras=True (record.py flips this on). Skipped during normal
        # training to keep extras["log"] identical to the pre-refactor payload — RSL-RL
        # only consumes that, so a missing log_per_env is a no-op. Keys mirror the
        # per-component reward / error names from extras["log"] above; the (N,) tensors
        # are the un-.mean()'d sources of those scalars. Curriculum / Phase / aggregate-
        # fraction fields aren't per-env-meaningful and are deliberately omitted.
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
                "Error/obj_lin_vel_error": obj_lin_vel_error,
                "Error/obj_ang_vel_error": obj_ang_vel_error,
                "Error/EE_pos_error": EE_pos_error,
                "Error/EE_quat_error": eef_quat_error,
                "Error/eef_box_rel_pos": eef_box_rel_pos_err,
                "Error/eef_box_rel_quat": eef_box_rel_quat_err,
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

        total_reward = rew_task + rew_track + rew_completion - rew_regularization

        # === VOC: accumulate per-env episode rewards for the decay check ===
        # We track the *unweighted* task and tracking signals so thresholds correspond
        # directly to per-step kernel values (in [0, 1]) rather than to weighted sums whose
        # scale would shift with α / w_task etc.
        rew_track_unweighted_per_step = (
            rew_EE_pos + rew_EE_quat + rew_joint_pos + rew_eef_box_rel_pos + rew_eef_box_rel_quat
        )
        self._voc_ep_rew_task += rew_task_unweighted
        self._voc_ep_rew_track += rew_track_unweighted_per_step
        self._voc_ep_steps += 1
        # Decay check is rate-limited to avoid hammering it every step. Read max() of
        # the per-seg tensor directly (not the property mean) so we keep checking until
        # the LAST segment decays — in per-segment mode, the mean drops as easy segs
        # zero out, but hard segs may still be active.
        self._voc_decay_step_counter += 1
        if (self.cfg.voc_enabled
                and float(self._voc_kp_pos_seg.max().item()) > 0.0
                and self._voc_decay_step_counter >= self.cfg.voc_decay_check_interval):
            self._voc_decay_step_counter = 0
            self._voc_decay_check()

        # Logs (extras["log"] was assigned by the block above; just append VOC entries).
        # Headline VOC/kp_pos etc. = mean across segments (= the single scalar in
        # single-VOC mode). Per-segment distribution stats are added below when relevant.
        self.extras["log"]["VOC/kp_pos"] = self._voc_kp_pos_seg.mean()
        self.extras["log"]["VOC/kp_rot"] = self._voc_kp_rot_seg.mean()
        self.extras["log"]["VOC/kv_pos"] = self._voc_kv_pos_seg.mean()
        self.extras["log"]["VOC/kv_rot"] = self._voc_kv_rot_seg.mean()
        if self.cfg.voc_per_segment:
            # Per-segment distribution of the gain; tracks how many segments still have
            # any assist active and how the curriculum has spread across difficulty.
            self.extras["log"]["VOC/kp_pos_min"] = self._voc_kp_pos_seg.min()
            self.extras["log"]["VOC/kp_pos_max"] = self._voc_kp_pos_seg.max()
            self.extras["log"]["VOC/kp_rot_min"] = self._voc_kp_rot_seg.min()
            self.extras["log"]["VOC/kp_rot_max"] = self._voc_kp_rot_seg.max()
            self.extras["log"]["VOC/n_active_segments"] = (
                (self._voc_kp_pos_seg > 0.0).sum().float()
            )
        # Recent-episode means used by the threshold check (NaN if buffer is empty).
        if self.cfg.voc_per_segment:
            buf_t, buf_k = self._voc_buf_task_seg, self._voc_buf_track_seg
        else:
            buf_t, buf_k = self._voc_buf_task, self._voc_buf_track
        valid_t = buf_t[~torch.isnan(buf_t)]
        valid_k = buf_k[~torch.isnan(buf_k)]
        self.extras["log"]["VOC/recent_task_mean"] = (
            valid_t.mean() if valid_t.numel() else torch.tensor(float("nan"), device=self.device)
        )
        self.extras["log"]["VOC/recent_track_mean"] = (
            valid_k.mean() if valid_k.numel() else torch.tensor(float("nan"), device=self.device)
        )

        # Update prev residual action / joint vel (first 6 dims of self.actions are residuals).
        self.prev_actions[:] = self.actions[:, :6]
        self.prev_joint_vel[:] = self._get_joint_vel()

        return total_reward

    def _voc_decay_check(self):
        """Decay VOC gains if all tracked reward-category trailing means exceed thresholds.

        Mirrors DexMachina Algorithm 1: deque-based mean of normalized cumulative rewards;
        decay only when ALL tracked categories pass their thresholds; below `voc_kp_min`
        the controller is fully zeroed out. Buffer is filled in `_reset_idx` whenever an
        env's episode ends.

        Warmup gate: no decay during the first `voc_decay_warmup_steps` env steps. Without
        this gate, the trailing-mean buffer fills with high values quickly (because
        VOC + controller tracking together produce high task/track reward even when the
        policy has done nothing), and decay starts firing every check interval — driving
        kp to a small fraction of initial before the policy has learned to compensate.
        """
        if self.common_step_counter < self.cfg.voc_decay_warmup_steps:
            return
        min_samples = self.cfg.voc_reward_window_size // 2
        phi_p = self.cfg.voc_decay_phi_p
        phi_v = self.cfg.voc_decay_phi_v
        kp_min = self.cfg.voc_kp_min

        if self.cfg.voc_per_segment:
            # Per-segment decay: each segment passes its own thresholds independently.
            # Compute per-row valid counts and masked means over the (S, W) buffers, then
            # build a (S,) `passing` mask gating which segments get multiplied by phi.
            buf_t, buf_k = self._voc_buf_task_seg, self._voc_buf_track_seg
            valid_t = ~torch.isnan(buf_t)
            valid_k = ~torch.isnan(buf_k)
            cnt_t = valid_t.sum(dim=1)
            cnt_k = valid_k.sum(dim=1)
            mean_t = torch.where(valid_t, buf_t, torch.zeros_like(buf_t)).sum(dim=1) / cnt_t.clamp(min=1).float()
            mean_k = torch.where(valid_k, buf_k, torch.zeros_like(buf_k)).sum(dim=1) / cnt_k.clamp(min=1).float()
            ready = (cnt_t >= min_samples) & (cnt_k >= min_samples)
            passing = ready & (mean_t >= self.cfg.voc_threshold_task) & (mean_k >= self.cfg.voc_threshold_track)
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
            # Snap per-segment kp below the floor to zero (and zero the matching kv);
            # `_apply_voc` then short-circuits per env via the kp[s]=0 entry.
            below = self._voc_kp_pos_seg < kp_min
            if bool(below.any().item()):
                self._voc_kp_pos_seg[below] = 0.0
                self._voc_kp_rot_seg[below] = 0.0
                self._voc_kv_pos_seg[below] = 0.0
                self._voc_kv_rot_seg[below] = 0.0
        else:
            # Single-VOC mode: global mean across the (W,) buffer; decay broadcasts
            # uniformly to every entry of the per-seg tensor (kept uniform by design).
            valid_task = self._voc_buf_task[~torch.isnan(self._voc_buf_task)]
            valid_track = self._voc_buf_track[~torch.isnan(self._voc_buf_track)]
            if valid_task.numel() < min_samples or valid_track.numel() < min_samples:
                return
            if (valid_task.mean() < self.cfg.voc_threshold_task or
                    valid_track.mean() < self.cfg.voc_threshold_track):
                return
            self._voc_kp_pos_seg.mul_(phi_p)
            self._voc_kp_rot_seg.mul_(phi_p)
            self._voc_kv_pos_seg.mul_(phi_v)
            self._voc_kv_rot_seg.mul_(phi_v)
            if float(self._voc_kp_pos_seg[0].item()) < kp_min:
                # Snap to zero so `_apply_voc` short-circuits cleanly.
                self._voc_kp_pos_seg.zero_()
                self._voc_kp_rot_seg.zero_()
                self._voc_kv_pos_seg.zero_()
                self._voc_kv_rot_seg.zero_()
        self._save_voc_state()

    def _save_voc_state(self):
        """Persist the current VOC runtime gains to <log_dir>/voc_state.npz so play.py /
        record.py can resume eval at the trained-end VOC level (via --keep_voc) instead
        of the cfg's initial values. Called only on decay events (so during play, when
        VOC is overridden to 0, no decay fires and no overwrite happens). Best-effort —
        a write failure shouldn't crash training."""
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir:
            return
        try:
            if self.cfg.voc_per_segment:
                # Per-segment arrays: play.py / record.py detect ndim>0 and write back
                # into the per-seg tensors via `apply_voc_state` instead of broadcasting
                # a scalar through the property setter.
                np.savez(
                    os.path.join(log_dir, "voc_state.npz"),
                    voc_kp_pos=self._voc_kp_pos_seg.detach().cpu().numpy(),
                    voc_kp_rot=self._voc_kp_rot_seg.detach().cpu().numpy(),
                    voc_kv_pos=self._voc_kv_pos_seg.detach().cpu().numpy(),
                    voc_kv_rot=self._voc_kv_rot_seg.detach().cpu().numpy(),
                )
            else:
                np.savez(
                    os.path.join(log_dir, "voc_state.npz"),
                    voc_kp_pos=self.voc_kp_pos,
                    voc_kp_rot=self.voc_kp_rot,
                    voc_kv_pos=self.voc_kv_pos,
                    voc_kv_rot=self.voc_kv_rot,
                )
        except OSError:
            pass

    def apply_voc_state(self, state) -> None:
        """Restore VOC gains from a saved `voc_state.npz` (dict-like with `voc_kp_pos`,
        `voc_kp_rot`, `voc_kv_pos`, `voc_kv_rot`). Accepts both formats:
          - scalar (single-VOC training) → broadcast across all per-seg tensors;
          - array of shape (S,) (per-segment training) → assign element-wise. The pool
            size at restore time must match the saved S; otherwise the array is broadcast
            via mean (with a warning) to avoid a hard failure.
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

    # === VOC gain properties (back-compat shim over the per-seg tensors) ===
    # Getter returns the mean across segments; in single-VOC mode all entries are equal
    # so this is exactly the legacy scalar. Setter broadcasts a scalar to every segment,
    # so existing external code like `env.voc_kp_pos = 0.0` (play.py / record.py /
    # ur_rtde_real_time.py) keeps working unchanged in both modes.
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
        """Errors between actual and reference EE pose expressed in the box's frame
        ("keep the EE at the same offset from the box as the planner expected"). Caller
        applies the gate and reward kernel.

        Returns (pos_err, quat_err) in meters / radians, both shaped (num_envs,).
        """
        EE_pos_ref   = self._interp(self.EE_poses[..., :3])
        EE_quat_ref  = self._nlerp(self.EE_poses[..., 3:])
        obj_pos_ref  = self._interp(self.obj_poses[..., :3])
        obj_quat_ref = self._nlerp(self.obj_poses[..., 3:])

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
            EE_pos -= self._interp(self.EE_poses[..., :3])
        return EE_pos

    def _get_EE_pos_error(self):
        return torch.norm(self._get_EE_pos(relative=True), dim=-1)

    def _get_EE_quat(self, relative=True) -> torch.Tensor:
        EE_quat = self.ur5e.data.body_quat_w[:, self.EE_link_idx].clone()
        if relative:
            desired_quat = self._nlerp(self.EE_poses[..., 3:])
            EE_quat = quat_mul(desired_quat, quat_inv(EE_quat))
        return EE_quat

    def _get_EE_quat_error(self):
        EE_quat = self.ur5e.data.body_quat_w[:, self.EE_link_idx].clone()
        desired_quat = self._nlerp(self.EE_poses[..., 3:])
        return torch.abs(quat_error_magnitude(EE_quat, desired_quat))

    def _get_EE_vel(self) -> torch.Tensor:
        return self.ur5e.data.body_vel_w[:, self.EE_link_idx].clone()

    def _get_obj_pos(self, relative=True):
        obj_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins
        if relative:
            desired_obj_pos = self._interp(self.obj_poses[..., :3])
            obj_pos -= desired_obj_pos
        return obj_pos

    def _get_obj_pos_error(self):
        return torch.norm(self._get_obj_pos(relative=True), dim=-1)

    def _get_obj_quat(self, relative=True):
        obj_quat = self.object.data.root_quat_w.clone()
        if relative:
            desired_obj_quat = self._nlerp(self.obj_poses[..., 3:])
            obj_quat = quat_mul(desired_obj_quat, quat_inv(obj_quat))
        return obj_quat

    def _get_obj_quat_error(self):
        obj_quat = self.object.data.root_quat_w.clone()
        desired_obj_quat = self._nlerp(self.obj_poses[..., 3:])
        return torch.abs(quat_error_magnitude(obj_quat, desired_obj_quat))

    def _get_obj_vel(self, relative=True):
        obj_vel = self.object.data.root_vel_w.clone()
        if relative:
            obj_vel -= self._interp(self.obj_vel)
        return obj_vel
