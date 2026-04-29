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


ROBOT_PATH = "./robots/ur5e.usd"
ENV_REGEX = "/World/envs/env_.*"


CUBE_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.235, 0.34, 0.27),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=4.4),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        activate_contact_sensors=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.35,
            dynamic_friction=0.35,
            restitution=0.2,
            friction_combine_mode="multiply"
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
            static_friction=0.35,
            dynamic_friction=0.35,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5 + 0.018))
)

@configclass
class EventCfg:
    object_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.3, 0.4),
            "dynamic_friction_range": (0.3, 0.4),
            "restitution_range": (0.2, 0.3),
            "num_buckets": 250,
        },
    )
    table_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.3, 0.4),
            "dynamic_friction_range": (0.3, 0.4),
            "restitution_range": (0.4, 0.5),
            "num_buckets": 250,
        },
    )
    object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )
    reset_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

@configclass
class BoxpushEnvCfg(DirectRLEnvCfg):
    # Trajectory file path
    trajectory_path = ""
    # env
    # physics_dt * decimations needs to match dt from planner/IK
    physics_dt = 1.0 / 100.0
    decimation = 2
    episode_length_s = 3.0
    # - spaces definition (single arm: 6 actions, reduced obs)
    # action_space is recomputed in __post_init__: 6 residual + 1 phase if enable_phase_slowdown.
    action_space = 6
    # obs = stacked per-step features * obs_history_steps + phase (1)
    #       [+ (7 or 14) * len(future_obs_steps)  future ref obj pose deltas (+ absolute fut ref if include_absolute_obs)]
    #       [+ 6                                  previous raw action]
    # per-step features = relative_q (6) + relative_qd (6)  [+ relative_obj_pos (3) + relative_obj_quat (4) + relative_obj_vel (6) if include_object_obs]
    #                     [+ absolute_q (6) + absolute_qd (6)  [+ absolute_obj_pos (3) + absolute_obj_quat (4) + absolute_obj_vel (6) if include_object_obs] if include_absolute_obs]
    obs_history_steps = 3
    # Toggle object (box) state in observation history. When True, adds the 7-dim pose
    # block (pos 3 + quat 4) per history step, plus the 6-dim velocity block if
    # include_obj_vel_obs is also True.
    include_object_obs = True
    # Include box velocity (lin 3 + ang 3 = 6 dims) in obj observations. Drop for sim2real
    # deployments where only pose is observable from the tracker — the policy can recover
    # implicit velocity from the position history. Reward path is unaffected; w_obj_lin_vel
    # and w_obj_ang_vel still see ground-truth simulator velocity.
    include_obj_vel_obs = False
    # Include absolute (world/env-frame) state alongside the relative (error) obs. Doubles the
    # per-step feature dim and adds (pos (3) + quat (4)) per future_obs_steps entry for the
    # absolute future reference obj pose.
    include_absolute_obs = True
    # Future reference obj pose look-ahead: list of phase offsets (in env steps) to include
    # as (pos_delta (3) + quat_delta (4)) relative to the reference at the current phase.
    # If include_absolute_obs, also appends absolute (pos (3) + quat (4)) per offset.
    # Empty tuple = disabled.
    future_obs_steps = (1,2,3,4,5)
    # Include previous raw residual action (6 dims) in the observation.
    include_prev_actions = True
    observation_space = 13  # recomputed in __post_init__
    state_space = 0

    # Continuous phase variable with slowdown-only semantics. When True, action[6] controls
    # the per-step trajectory advance rate dphase ∈ [dphase_min, 1.0] (the policy can slow
    # down but never speed up); reference values are interpolated at fractional phase indices.
    # When False, the env behaves as before: integer indexing, dphase=1.
    #
    # Reward accounting: task/track rewards are NOT scaled by dphase. Instead, sustained
    # pauses are discouraged by a cumulative-quadratic penalty — `w_total_slowdown` times
    # the running sum of (1 - dphase), gated by (1 - dphase) so running (dphase=1) is free.
    # Total episode pause cost ≈ w_total_slowdown · (total_slowdown)².
    enable_phase_slowdown = True
    # Lower bound on dphase. dphase = (1 + (1 - dphase_min) * tanh(action[6])).clamp(dphase_min, 1).
    # raw=0 → dphase=1 (no slowdown, neutral); raw<0 → progressive slowdown to dphase_min;
    # raw>0 → clamped at 1 (deadzone). dphase_min=0.0 is allowed (full pause).
    dphase_min = 0.0
    # Wall-clock cap as multiple of nominal trajectory duration (replaces the old
    # traj_duration / dphase_min formula, which divides by zero at dphase_min=0).
    max_slowdown_multiplier = 3.0

    # Failure-aware phase resampling: biases episode start phases toward segments with
    # high historical failure rate. Credits are assigned to the segment each episode
    # STARTED in (not where it failed), which matches the RSI lever we actually control.
    enable_failure_resampling = False
    phase_segment_s = 1.0                    # segment duration in seconds; num segments = ceil((T-1)*dt / phase_segment_s)
    phase_resample_alpha = 0.05              # per-event EMA weight
    phase_resample_beta = 1.0                # temperature: p_s ∝ r_s^beta
    phase_resample_clamp = (0.1, 0.9)        # (low, high) bounds on each r_s

    @property
    def per_step_feature_dim(self) -> int:
        # 12: relative joint pos (6) + relative joint vel (6) — always present.
        # If include_object_obs: + 7 (obj pos 3 + obj quat 4), + 6 if include_obj_vel_obs.
        # If include_absolute_obs: doubles the whole thing (joint and obj parts mirrored).
        obj_dim = 0
        if self.include_object_obs:
            obj_dim = 7 + (6 if self.include_obj_vel_obs else 0)
        dim = 12 + obj_dim
        if self.include_absolute_obs:
            dim *= 2
        return dim

    def __post_init__(self):
        # Idempotent: always reset action_space/observation_space to base before recomputing.
        # Must be safe to call more than once (e.g. after hydra overrides).
        self.action_space = 7 if self.enable_phase_slowdown else 6
        self.observation_space = self.per_step_feature_dim * self.obs_history_steps + 1
        future_dim = 14 if self.include_absolute_obs else 7
        self.observation_space += future_dim * len(self.future_obs_steps)
        if self.include_prev_actions:
            self.observation_space += 6

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))

    # Domain Randomization
    events: EventCfg = EventCfg()

    ur5e_prim_path = f"{ENV_REGEX}/ur5e"

    # Arm Actuator parameters
    # kp = 150.0
    # kd = 22.5
    kp = {
        "shoulder_pan_joint": 800.0,
        "shoulder_lift_joint": 600.0,
        "elbow_joint": 300.0,
        "wrist_1_joint": 200.0,
        "wrist_2_joint": 100.0,
        "wrist_3_joint": 100.0,
    }
    kd = {joint_name: joint_kp * 0.15 for joint_name, joint_kp in kp.items()}
    actuator_type = "Implicit"  # or "IdealPD"
    velocity_limit = 3.14
    effort_limit = {
        "shoulder_pan_joint": 150.0,
        "shoulder_lift_joint": 150.0,
        "elbow_joint": 150.0,
        "wrist_1_joint": 28.0,
        "wrist_2_joint": 28.0,
        "wrist_3_joint": 28.0,
    }

    # Action scale
    action_scale: float | list = 0.1

    # Action formulation. One of:
    #   "A" — joints_target[t] + action_scale * action
    #         Residual on the planner's absolute target.
    #   "B" — joints[t] + action_scale * action
    #         Residual on the planner's trajectory position (planner feedforward not applied).
    #   "C" — curr_joints + action_scale * action
    #         Residual on the robot's current joint position (no planner info).
    #   "D" — curr_joints + (joints_target[t] - joints[t]) + action_scale * action
    #         Planner's intended PD error (force direction) applied from current position,
    #         plus learned residual. Effective PD error is independent of tracking state.
    action_mode = "D"

    # object (cube)
    cube_cfg = CUBE_CFG

    # table
    table_cfg = TABLE_CFG

    # scene
    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=replicate_physics)

    # Reset noise (for sim-to-real robustness)
    reset_joint_pos_noise = 0.05    # rad, std of Gaussian noise added to joint positions on reset
    reset_joint_vel_noise = 0.05     # rad/s, std of Gaussian noise added to joint velocities on reset

    # Box reset noise: xy-plane for translation / linear vel; small 3-axis rotation + ang vel.
    reset_obj_pos_xy_noise = 0.01        # m, std on box x,y position (z unchanged)
    reset_obj_lin_vel_xy_noise = 0.05    # m/s, std on box linear x,y velocity (z unchanged)
    reset_obj_ori_noise = 0.05           # rad, axis-angle std for small orientation perturbation
    reset_obj_ang_vel_noise = 0.1        # rad/s, std on box angular velocity (all axes)

    # Box observation noise (sim2real). Sampled fresh each step, applied ONLY to the
    # observation path — rewards still see the clean ground-truth box state. Same noise
    # tensor is used for both the relative and absolute obj views in a given step so the
    # policy sees mutually consistent readings (a single noisy "sensor" produced both).
    obs_obj_pos_noise = 0.005            # m, per-step Gaussian noise on box position
    obs_obj_ori_noise = 0.02             # rad, axis-angle per-step noise on box orientation
    obs_obj_lin_vel_noise = 0.02         # m/s, per-step Gaussian noise on box linear velocity
    obs_obj_ang_vel_noise = 0.05         # rad/s, per-step Gaussian noise on box angular velocity

    # Perturbation forces (for sim-to-real robustness)
    perturbation_force_std = 10.0       # N, std of per-axis Gaussian force
    perturbation_torque_std = 2.0       # Nm, std of per-axis Gaussian torque
    perturbation_probability = 0.05     # probability of applying a perturbation each step
    perturbation_duration_steps = 5     # how many steps the perturbation lasts

    # Reward parameteres. Final (end-of-curriculum) values.
    w_task = 0.8
    w_track = 1 - w_task
    w_regularization = 0.4

    # Curriculum ("α schedule"). α ramps linearly from 0 to 1 over alpha_warmup_steps env
    # steps; 0 disables the curriculum (α=1 always). Drives three coupled shifts:
    #   1) (w_task, w_track) interpolate (w_task_start, w_track_start) → (w_task, w_track).
    #   2) In action_mode == "D" only: the action command blends
    #        q_curr + (1-α)·(ref_target - ref_pos) + (α + ε(1-α))·scale·a
    #      where ε = action_alpha_floor keeps a minimum action authority at α=0.
    #   3) In action_mode == "D" only: the policy-authored regularization terms
    #      (action_rate, action_norm) are scaled by (α + ε(1-α)) so the penalty
    #      tracks the action's actual effect on the env. Safety penalties (joint_limit,
    #      illegal_contact, flange_forearm, proximity, joint_acc, torque) stay unscaled.
    alpha_warmup_steps = 24 * 1500
    w_task_start = 0.2
    w_track_start = 0.8
    action_alpha_floor = 0.1
    # Optional fixed-α override. When set (≥ 0), bypasses the schedule and uses this value
    # everywhere _curriculum_alpha() is consulted. Intended for eval: play.py / record.py
    # compute the training-final α from the checkpoint and pin it so the frozen policy runs
    # in the same regime it was trained at (critical for mode D, where α controls the
    # action blend between planner feedforward and learned residual).
    force_alpha: float = -1

    # Task reward parameteresr
    w_obj_pos = 0.3
    sigma_obj_pos = 0.05
    tol_obj_pos = 0.0

    w_obj_quat = 0.4
    # Multi-sigma: wide kernel (0.2) keeps gradient alive at moderate errors,
    # narrow kernel (0.05) pulls precision in close. Kernels are averaged in _reward_track.
    sigma_obj_quat = (0.05, 0.2)
    tol_obj_quat = 0.0

    # Object velocity tracking — split into linear (m/s) and angular (rad/s) to avoid
    # unit-mixing in the norm. The rotation task has reference ang-vel up to ~1.2 rad/s
    # vs linear up to ~0.15 m/s, so the kernels have very different scales.
    # Instantaneous signal that catches "policy stopped pushing" before obj_pos_error
    # integrates up to the termination threshold.
    w_obj_lin_vel = 0.15
    sigma_obj_lin_vel = 0.08      # m/s — covers reference vel range with decent gradient
    tol_obj_lin_vel = 0.0

    w_obj_ang_vel = 0.15
    sigma_obj_ang_vel = 0.4       # rad/s — wider to match ~10× larger magnitude
    tol_obj_ang_vel = 0.0

    # Track reward parameters
    w_eef_pos = 1.0
    sigma_eef_pos = 0.1
    tol_eef_pos = 0.0

    w_eef_quat = 0.0
    sigma_eef_quat = 0.5
    tol_eef_quat = 0.1

    w_joint_pos = 0.0
    sigma_joint_pos = 0.2
    tol_joint_pos = 0.0

    # Relative EE-box tracking: rewards matching the reference's EE-position-in-box-frame
    # (and optionally quat). Only active during reference-trajectory segments where the
    # planner expects the box to be moving (contact/near-contact phases), since enforcing
    # a specific EE-in-box-frame offset during regrasp/approach is brittle and
    # unachievable if the box is flipped from the reference. The gate is a precomputed
    # boolean mask derived from |obj_vel_ref| > eps, dilated by ±dilation_steps so it
    # captures brief pre-contact approach and post-release follow-through.
    w_eef_box_rel_pos = 1.0
    sigma_eef_box_rel_pos = 0.05
    tol_eef_box_rel_pos = 0.0

    w_eef_box_rel_quat = 0.0
    sigma_eef_box_rel_quat = 0.3
    tol_eef_box_rel_quat = 0.0

    # Gate parameters: a reference step is "active" if ||obj_vel_lin|| + ||obj_vel_ang||
    # > eps, then dilated by ±dilation_steps (in policy steps).
    eef_box_gate_obj_vel_eps = 1e-3
    eef_box_gate_dilation_steps = 50 # 1s of approach and leaving

    # Regularization reward parameters
    w_joint_acc = 1e-3
    tol_joint_acc = 0.0

    w_joint_torque = 1e-3
    tol_joint_torque = 0.0

    w_action_rate = 2e-1
    tol_action_rate = 0.0

    w_action_norm = 1e-2
    tol_action_norm = 0.0

    # Cumulative-quadratic pause penalty for enable_phase_slowdown=True.
    # Per-step: w_total_slowdown · cumulative_slowdown · (1 - dphase), which is zero
    # while running (dphase=1) and grows with accumulated pause time while paused.
    # Episode-total: w_total_slowdown · (Σ(1-dphase))². Trades off "pause to recover from a
    # miss" against "don't pause forever": short pauses are cheap, sustained pauses scale
    # quadratically and eventually dominate any per-step reward gain.
    w_total_slowdown = 0.05

    # Completion bonus — one-shot terminal reward when the episode times out (reached the
    # end of the trajectory without reset_terminated). Counters stall-dominance in the
    # value function: without it, V(stall-until-pos_error-threshold) can exceed V(attempt-
    # and-fail) because both end in termination and stalling simply survives longer. With
    # this bonus, completing the trajectory is a dedicated positive terminal state.
    # TODO (follow-up): scale by final obj_pos/quat reward so quality of completion matters.
    # TODO: is this even good to have?
    w_completion = 0.0

    w_joint_limit = 500.0
    joint_limit_eps = 0.05

    w_proximity_to_contact = 0.5
    max_proximity = 0.05

    w_illegal_contact = 50.0
    min_contact_force = 0
    max_contact_force = 20

    w_flange_forearm_dist = 1.0
    max_flange_forearm_distance = 0.028 + 0.0375

    # Contact Sensors
    non_ee_link_names = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
    ]

    cube_contact_filter = []
    for link_name in non_ee_link_names:
        cube_contact_filter.append(f"{ur5e_prim_path}/{link_name}/")
    table_contact_filter = cube_contact_filter
    table_contact_filter.append(f"{ur5e_prim_path}/wrist_3_link/")

    illegal_contact_sensor_cfgs = {
        # "cube": ContactSensorCfg(
        #     prim_path=cube_cfg.prim_path,
        #     update_period=0.0,
        #     history_length=0,
        #     debug_vis=True,
        #     force_threshold=min_contact_force,
        #     max_contact_data_count_per_prim=16,
        #     filter_prim_paths_expr=cube_contact_filter
        # ),
        "table": ContactSensorCfg(
            prim_path=table_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=table_contact_filter
        ),
    }

    # - reset conditions
    max_obj_dist_from_traj = 0.1
    max_obj_angle_from_traj = 1.0


def get_ur5e_cfg(
    prim_path,
    init_pose,
    cfg: BoxpushEnvCfg,
):
    actuator_kwargs = dict(
        joint_names_expr=[".*"],
        stiffness=cfg.kp,
        damping=cfg.kd,
        velocity_limit=cfg.velocity_limit,
        effort_limit=cfg.effort_limit,
    )

    if cfg.actuator_type == "IdealPD":
        actuator_cfg = IdealPDActuatorCfg(**actuator_kwargs)
    elif cfg.actuator_type == "Implicit":
        actuator_cfg = ImplicitActuatorCfg(**actuator_kwargs)
    else:
        raise ValueError(f"Unknown actuator type: {cfg.actuator_type}")

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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
