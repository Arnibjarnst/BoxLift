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
            static_friction=0.5,
            dynamic_friction=0.5,
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
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5 + 0.018))
)

@configclass
class EventCfg:
    # Step 2 of sim2real: re-widened DR ranges. Actuator gains widened MORE than the
    # other fields (0.7-1.3 vs ±20% on others) because the real UR5e's internal
    # controller is impossible to match in sim — wide gain DR forces the policy to
    # work across closed-loop dynamics it can't predict, which is what deployment
    # actually looks like.
    object_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.4, 0.6),
            "dynamic_friction_range": (0.4, 0.6),
            "restitution_range": (0.2, 0.3),
            "num_buckets": 250,
        },
    )
    table_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.7, 1.0),
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.1, 0.3),
            "num_buckets": 250,
        },
    )
    object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform"
        },
    )
    # CoM offset randomization (sim2real). com_range values are absolute meters added
    # to the geometric-center CoM. For a 0.235x0.34x0.27m box, ±4cm in xy is ~25-34% of
    # the half-dimension — captures meaningful asymmetric content / hollow interior,
    # while staying inside the box geometry (so the contact point is still physically
    # reasonable). ±2cm in z because the box is shortest in that dim.
    #
    # IMPORTANT: mode="startup" (NOT "reset"). The IsaacLab implementation does
    # `coms += rand_samples`, not `coms = baseline + rand_samples`, so calling on every
    # reset accumulates the offset (random walk with variance growing in N). The official
    # docstring recommends one-shot use — startup-mode samples once per env at training
    # start and holds it. Diversity comes from the many envs, not per-episode resampling.
    object_com = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
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
    # Wider than the other fields (±30% vs ±20%) because real UR5e closed-loop
    # dynamics don't match sim's PD; this is the dominant sim2real mismatch.
    actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

@configclass
class BoxhingeEnvCfg(DirectRLEnvCfg):
    # Trajectory file path
    trajectory_path = ""

    emit_per_env_extras: bool = False
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
    # per-step features = relative_q (6) + relative_qd (6)  [+ relative_obj_pos (3) + relative_obj_quat (4) if include_object_obs]
    #                     [+ absolute_q (6) + absolute_qd (6)  [+ absolute_obj_pos (3) + absolute_obj_quat (4) if include_object_obs] if include_absolute_obs]
    obs_history_steps = 3
    # Toggle object (box) state (pose only — pos 3 + quat 4 = 7 dims) in the observation
    # history. The policy is expected to recover implicit velocity from the pose history;
    # there is no longer a velocity-observation option (only the tracker exposes pose on
    # the real robot). Reward path uses ground-truth velocity via _get_obj_vel and is
    # unaffected.
    include_object_obs = True
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
    # Include a single thresholded contact bool (EE ↔ cube) in the per-step observation.
    # 1.0 when the EE contact sensor reports |force| > contact_threshold, else 0.0.
    # Real-side analog: thresholded delta of getActualTCPForce() vs a baseline.
    include_contact_obs = True
    # Force magnitude threshold (N) for the contact bool. ~0.5–2N is reasonable for the
    # sphere EE on the cube; tune by inspecting force-magnitude histograms in obvious
    # contact vs free motion.
    contact_threshold = 0.5
    # DR on the contact bool (default off — start with a clean threshold, add these only
    # if sim2real shows the policy is brittle to real-side flakiness):
    # - delay_steps: shift the bool through a rolling buffer so the obs sees a value from
    #   N steps ago. Models the ~10–30ms estimator latency of getActualTCPForce.
    # - flip_prob: per-step Bernoulli bit-flip rate. Models false positives (inertia
    #   spikes) and false negatives (transient drops below threshold).
    contact_obs_delay_steps = 0
    contact_obs_flip_prob = 0.0
    observation_space = {"policy": 13, "privileged": 85}  # recomputed in __post_init__
    state_space = 0

    # True: action[6] sets per-step advance dphase ∈ [dphase_min, 1]; ref interpolated at
    # fractional phase. False: integer indexing, dphase=1. Disabled — phase slowdown is a
    # speculative timing hedge; CoM/size DR prioritized for sim2real first.
    enable_phase_slowdown = False
    # tanh mapping: dphase = (1 + (1-dphase_min)*tanh(action[6])).clamp(dphase_min, 1).
    # raw=0→dphase=1, raw<0→slowdown, raw>0→clamped (deadzone). 0.0 = full pause allowed.
    dphase_min = 0.0
    dphase_max = 1.0  # only used by phase_mapping="cubic_bidir"; tanh clamps to 1
    # "tanh" = slowdown-only. "cubic_bidir" = clamp(1 + scale*raw**3, dphase_min, dphase_max),
    # scale = max(1-dphase_min, dphase_max-1); flat near raw=0, must commit to deviate.
    phase_mapping: str = "tanh"
    max_slowdown_multiplier = 3.0  # wall-clock cap = this × nominal traj duration
    # Multiply task reward by dphase: removes the positive incentive to stall (task→0 at
    # dphase=0). Complements w_slowdown_gated which makes stalling actively costly.
    task_scale_by_dphase: bool = True
    # Also scale track by dphase → all per-step reward vanishes at dphase=0.
    track_scale_by_dphase: bool = True

    # Hard cap on per-episode wall-clock length, in simulator steps. When set (>0), the
    # episode terminates by time_out after this many steps regardless of phase progress,
    # AND t0 sampling in _reset_idx is restricted to [0, T-1-L] so an episode at full
    # speed can run for L steps without falling off the trajectory. The combination gives
    # roughly uniform state visitation across the trajectory (each state t is visited
    # with probability ~ min(L, t+1) / (T-L+1) instead of (t+1)/(T-1) under the default).
    # -1 disables the cap (use full trajectory length, the behavior before this flag).
    max_episode_steps: int = -1

    # Continue the episode for this many seconds after the trajectory phase reaches its
    # end. During the hold, phase stays clamped at max so the reference targets stay at
    # the trajectory's final pose, and the policy keeps receiving rewards for maintaining
    # that final pose. Increases the training signal for endpoint stability — particularly
    # useful for box-lift / box-hinge tasks where the final pose must be held (otherwise
    # the policy can succeed at the end and let the box drop the next step without
    # penalty). 0.0 disables the hold (legacy behavior — terminate as soon as phase ends).
    post_traj_hold_s: float = 2.0

    # Probability that a reset overrides RSI/failure-resampling and starts at phase=0
    # instead. Pure RSI samples phase=0 with probability ~1/T (where T is trajectory
    # length, often <1%), but deployment ALWAYS starts at 0 — so the start of the
    # trajectory is heavily under-trained. Setting this to 0.05 gives the start ~5×
    # the training exposure it would otherwise get. Useful when the policy oscillates
    # on the real robot during initial-approach motion. 0.0 disables the override.
    reset_to_zero_prob: float = 0.05

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
        # If include_object_obs: + 7 (obj pos 3 + obj quat 4).
        # If include_absolute_obs: doubles the whole thing (joint and obj parts mirrored).
        # If include_contact_obs: + 1 (EE-cube thresholded contact bool, not mirrored).
        dim = 12 + (7 if self.include_object_obs else 0)
        if self.include_absolute_obs:
            dim *= 2
        if self.include_contact_obs:
            dim += 1
        return dim

    def __post_init__(self):
        # Idempotent: always reset action_space/observation_space to base before recomputing.
        # Must be safe to call more than once (e.g. after hydra overrides).
        self.action_space = 7 if self.enable_phase_slowdown else 6
        actor_dim = self.per_step_feature_dim * self.obs_history_steps + 1
        future_dim = 14 if self.include_absolute_obs else 7
        actor_dim += future_dim * len(self.future_obs_steps)
        if self.include_prev_actions:
            actor_dim += 6
        self.observation_space = {"policy": actor_dim, "privileged": 85}

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))

    # Domain Randomization
    events: EventCfg = EventCfg()

    ur5e_prim_path = f"{ENV_REGEX}/ur5e"

    # Arm Actuator parameters
    kp = 150.0
    kd = 22.5
    # kp = {
    #     "shoulder_pan_joint": 150.0,
    #     "shoulder_lift_joint": 150.0,
    #     "elbow_joint": 150.0,
    #     "wrist_1_joint": 28.0,
    #     "wrist_2_joint": 28.0,
    #     "wrist_3_joint": 28.0,
    # }
    # kd = {joint_name: joint_kp * 0.15 for joint_name, joint_kp in kp.items()}
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
    # Mode A chosen after observing mode B + VOC (kp=1000) struggle to learn: tracking
    # peaked at 0.72 then regressed below the action_scale=0 baseline (0.65). Mode B's
    # learning problem is "discover FF from scratch under weak gradient signal", which
    # PPO didn't solve in this setup. Mode A's planner FF gives good baseline tracking
    # at residual=0; the policy only needs small corrections on top — a friendlier RL
    # problem. The earlier mode A + VOC failure (erratic motion) was driven by the
    # combination of high reset noise + VOC yanking the cube; with reset_joint_pos_noise
    # halved that interaction is much milder.
    action_mode = "B"

    # Action scale
    action_scale: float | list = 0.05

    # object (cube)
    cube_cfg = CUBE_CFG

    # table
    table_cfg = TABLE_CFG

    # scene
    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=replicate_physics)

    # Reset noise (for sim-to-real robustness). Per-joint (length 6, ordered shoulder_pan,
    # shoulder_lift, elbow, wrist_1, wrist_2, wrist_3). Wrist joints have small Jacobians on
    # EE position, so they can absorb more noise without dragging the EE far from the
    # trajectory. Scalar still works (broadcasts across all 6 joints).
    # Halved for VOC training. Mode B without FF needs the controller (kp=300) to recover
    # from initial offset before the trajectory diverges; large initial noise compounds
    # with the FF-lag problem. Re-widen once the policy is reliably tracking.
    # Restored to original values (Step 2 sim2real). Halved during VOC-curriculum
    # bring-up because mode B without FF couldn't recover from large initial joint
    # offsets fast enough; with the policy now trained and VOC mostly decayed, the
    # original spread (which simulates a wider band of real-robot startup states) is
    # appropriate.
    reset_joint_pos_noise = [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]
    reset_joint_vel_noise = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Box reset noise: xy-plane for translation / linear vel; small 3-axis rotation + ang vel.
    reset_obj_pos_xy_noise = 0.02        # m, std on box x,y position (z unchanged)
    reset_obj_lin_vel_xy_noise = 0.05    # m/s, std on box linear x,y velocity (z unchanged)
    reset_obj_ori_noise = 0.1            # rad, axis-angle std for small orientation perturbation
    reset_obj_ang_vel_noise = 0.1        # rad/s, std on box angular velocity (z axis)

    # Box observation noise (sim2real). Sampled fresh each step, applied ONLY to the
    # observation path — rewards still see the clean ground-truth box state. Same noise
    # tensor is used for both the relative and absolute obj views in a given step so the
    # policy sees mutually consistent readings (a single noisy "sensor" produced both).
    # Per-fire noise: small per-tracker-frame jitter on top of the per-episode bias below.
    # Magnitudes reduced from 0.005 / 0.01 since the bias term now provides the dominant
    # systematic offset; per-fire noise just adds residual detection jitter.
    obs_obj_pos_noise = 0.001            # m, per-FRESH-SAMPLE Gaussian noise on box position
    obs_obj_ori_noise = 0.001            # rad, axis-angle per-FRESH-SAMPLE noise on box orientation

    # Per-episode bias on the box obs (sim2real). Sampled fresh on each reset, then held
    # constant for the entire episode. Models systematic calibration error in the
    # camera-to-robot transform — the kind of offset that doesn't average out across
    # frames and forces the policy to learn behaviors robust to a constant displacement.
    # Total effective noise per step ≈ sqrt(bias_std² + per_fire_noise²) ≈ 0.01m / 0.02rad,
    # matching the previous magnitudes but redistributing weight to the temporally-
    # correlated component (which is more realistic for ArUco-style trackers).
    obs_obj_pos_bias_std = 0.005          # m, per-EPISODE Gaussian std on constant box-position offset
    obs_obj_ori_bias_std = 0.005          # rad, per-EPISODE Gaussian std on constant box-orientation offset

    # Box pose tracker model (sim2real). Approximates the UR5e vision tracker:
    #   - Fixed latency: every fresh sample reads exactly `obs_obj_delay_steps` env steps
    #     into the past. At env_dt = 20ms, 13 steps = 260ms (covers the typical ~250ms lag).
    #     Fixed (not random) so consecutive measurements are time-ordered: a fire at step t
    #     observes physical time (t-delay)*dt, the next fire at t+period observes
    #     (t+period-delay)*dt — strictly increasing.
    #   - Sub-50Hz update rate: a fresh tracker frame fires every `obs_obj_update_period`
    #     env steps. Default 2 → 25Hz. Between fires the policy reuses the last sample
    #     (real trackers hold the previous frame until a new one arrives).
    # Per-fire noise (obs_obj_pos_noise / obs_obj_ori_noise) is applied only when a fresh
    # sample fires; held samples don't re-jitter.
    # Reference path (planner trajectory) is NOT affected. Reward path reads clean ground
    # truth via _get_obj_pos / _get_obj_quat. To disable: set obs_obj_delay_steps=0 and
    # obs_obj_update_period=1.
    # Re-enabled (Step 1 of sim2real) — matches the real tracker: ~260ms latency, 25Hz update.
    obs_obj_delay_steps = 5              # 100ms at 20ms env step
    obs_obj_update_period = 2            # 25Hz

    # Perturbation forces (for sim-to-real robustness)
    perturbation_force_std = 10.0       # N, std of per-axis Gaussian force
    perturbation_torque_std = 2.0       # Nm, std of per-axis Gaussian torque
    perturbation_probability = 0.05     # probability of applying a perturbation each step
    perturbation_duration_steps = 5     # how many steps the perturbation lasts

    # Reward parameteres. Final (end-of-curriculum) values.
    # Inverted from 0.2 / 0.8 (track-dominated) to 0.7 / 0.3 (task-dominated). Following
    # DexMachina §4.2: "λ_task is the largest weight" — box-state reward should dominate;
    # tracking exists as guidance. Under the old weighting, the policy maxed track (cheap
    # kinematic following) at the expense of task, and VOC could decay past the policy's
    # actual contact-mechanics capability because the threshold check still saw high track.
    w_task = 0.9
    w_track = 1 - w_task
    w_regularization = 1.0

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
    alpha_warmup_steps = 24 * 0
    w_task_start = 0.7
    w_track_start = 0.3
    action_alpha_floor = 0.1
    # Optional fixed-α override. When set (≥ 0), bypasses the schedule and uses this value
    # everywhere _curriculum_alpha() is consulted. Intended for eval: play.py / record.py
    # compute the training-final α from the checkpoint and pin it so the frozen policy runs
    # in the same regime it was trained at (critical for mode D, where α controls the
    # action blend between planner feedforward and learned residual).
    force_alpha: float = -1

    # === Task reward aggregation form ===
    # "sum"     — legacy: rew_task = w_pos·exp(-d²/σ²) + w_quat·exp(-d²/σ²) + vel terms.
    #             Policy can compensate one bad axis with another good one.
    # "product" — DexMachina: rew_task = exp(-β_pos·d_pos) · exp(-β_rot·d_rot). All axes
    #             must be small for high reward; matches the paper's r_task = r_pos·r_rot.
    #             Distances (not squared errors) — different falloff than sum form.
    task_reward_form: str = "sum"
    task_beta_pos: float = 50.0   # 1/m; ~2cm error → exp(-1)
    task_beta_rot: float = 3.0    # 1/rad; ~20° error → exp(-1)

    # Task reward parameters (used by sum form; still computed in product form for logging).
    w_obj_pos = 0.5
    sigma_obj_pos = 0.075
    tol_obj_pos = 0.0

    w_obj_quat = 0.5
    sigma_obj_quat = 0.15
    tol_obj_quat = 0.0

    # Object velocity tracking left disabled — the policy reached a working solution
    # without this signal in the prior runs, and obj_pos + obj_quat already capture
    # trajectory matching. Adds reward-balance complexity without clear benefit.
    w_obj_lin_vel = 0.0
    sigma_obj_lin_vel = 0.08
    tol_obj_lin_vel = 0.0

    w_obj_ang_vel = 0.0
    sigma_obj_ang_vel = 0.2
    tol_obj_ang_vel = 0.0

    # Track reward parameters. Sigmas tightened (was 0.1 / 0.5) so the gradient between
    # "rough tracking" and "precise tracking" is steep enough to overcome PPO's entropy
    # noise. With the previous wide kernels, going from 5cm error → 1cm gave only 0.21
    # additional reward, which gets buried under per-sample noise.
    w_eef_pos = 1.0
    sigma_eef_pos = 0.03    # 1cm err: 0.89; 3cm err: 0.37
    tol_eef_pos = 0.0

    w_eef_quat = 0.0
    sigma_eef_quat = 0.2    # 10° err: 0.79; 25° err: 0.20
    tol_eef_quat = 0.1

    w_joint_pos = 0.0
    # Tuned for the per-joint averaged kernel (r_bc form). At σ=0.1, a single joint with
    # 0.1 rad error gives kernel = exp(-0.01/0.01) ≈ 0.37; 0.2 rad gives ≈ 0.018. Was 0.2
    # under the old sum-then-kernel form; not equivalent — this is the per-joint scale.
    sigma_joint_pos = 0.1
    tol_joint_pos = 0.0

    # Relative EE-box tracking: rewards matching the reference's EE-position-in-box-frame
    # (and optionally quat). Only active during reference-trajectory segments where the
    # planner expects the box to be moving (contact/near-contact phases), since enforcing
    # a specific EE-in-box-frame offset during regrasp/approach is brittle and
    # unachievable if the box is flipped from the reference. The gate is a precomputed
    # boolean mask derived from |obj_vel_ref| > eps, dilated by ±dilation_steps so it
    # captures brief pre-contact approach and post-release follow-through.
    w_eef_box_rel_pos = 0.7
    sigma_eef_box_rel_pos = 0.05
    tol_eef_box_rel_pos = 0.0

    w_eef_box_rel_quat = 0.3
    sigma_eef_box_rel_quat = 0.5
    tol_eef_box_rel_quat = 0.0

    # Gate parameters: a reference step is "active" if ||obj_vel_lin|| + ||obj_vel_ang||
    # > eps, then dilated by ±dilation_steps (in policy steps).
    eef_box_gate_obj_vel_eps = 1e-3
    eef_box_gate_dilation_steps = 1e7 # everything

    # RSI (random start init) contact exclusion. The set of trajectory phases that count
    # as "in contact" for RSI sampling is the raw |obj_vel_ref| > eef_box_gate_obj_vel_eps
    # mask, dilated by this many integer steps in each direction. INDEPENDENT of
    # eef_box_gate_dilation_steps — that one controls reward shaping, this one controls
    # which start phases the env will reset to.
    #
    # Resetting mid-contact tends to put the EE inside the box (after joint reset noise)
    # and snap-resolves into weird states, so we forbid it. 0 = use the raw boolean mask.
    rsi_contact_dilation_steps = 5

    # Regularization reward parameters
    w_joint_acc = 2e-4
    tol_joint_acc = 0.0

    w_joint_torque = 1e-3
    tol_joint_torque = 0.0

    # Reduced from 1e-1 (default) to 1e-2 for VOC + mode B. At 0.0 the policy was free
    # to output wildly different residuals on consecutive steps, jittering the joint
    # command faster than kp=300 could track (track ↓, reg ↑ from torque/contact spikes).
    # At 0.1 the rate penalty dominated and pushed the policy toward useless-constant.
    # 0.01 keeps the command coherent across steps without preventing the time-varying
    # residuals that mode B needs for FF compensation during the lift.
    w_action_rate = 5e-1
    tol_action_rate = 0.0

    w_action_norm = 5e-3
    tol_action_norm = 0.0

    # === Slowdown penalties (require enable_phase_slowdown=True) ===
    # Min-gate: penalty = w_slowdown_gated · (1-dphase) · min(r_task_norm, r_track_norm),
    # r_*_norm ∈ [0,1] (cheap when either channel suffers, expensive when both healthy).
    # CONSTRAINT: w_reg · w_slowdown_gated < min(w_task, w_track), else the gradient on the
    # smaller-weighted channel inverts (policy paid to make it worse). w_reg=0.25,
    # w_task=0.7, w_track=0.3 → keep w_slowdown_gated < 1.2.
    w_slowdown_gated: float = 0.0

    # Improvement-gate: penalty = w_slowdown_improvement · (1-dphase) · exp(-β·improvement),
    # improvement = max(0,Δerr_task)+max(0,Δerr_track), Δerr in sigma units (err/σ summed
    # over pos+quat). Raw error not Δreward: the kernel saturates far from ref so Δr≈0
    # during recovery exactly when we need the gate to license slowdown. β dimensionless
    # (per σ); 2.0 → Δerr=0.5σ gives ~63% discount.
    w_slowdown_improvement: float = 0.0
    slowdown_improvement_beta: float = 2.0

    # Cumulative-quadratic pause penalty (legacy safety net): per-step
    # w_total_slowdown·cumulative_slowdown·(1-dphase), episode ≈ w_total_slowdown·(Σ(1-dphase))².
    # Off by default; re-enable if the gates let the policy park without completing.
    w_total_slowdown = 0.0

    # One-shot terminal bonus on timeout (trajectory completed, not reset_terminated);
    # counters stall-dominance in the value function. INCIDENT: 50.0 caused bimodal
    # returns (+50 full vs +0 failed) → value loss blew 70→4000; keep small (~5) if used.
    # TODO: scale by final obj_pos/quat reward; reconsider whether it's needed at all.
    w_completion = 0.0

    w_joint_limit = 1e3
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

    # EE↔cube contact sensor. Reports forces on the cube filtered to the EE link
    # (wrist_3_link, since this is the ur5e.usd asset with no separate sphere). Used by
    # _get_observations to produce a thresholded contact bool when include_contact_obs
    # is set. Lightweight (max_contact_data_count_per_prim=4) since we only consume the
    # net magnitude, not per-contact data.
    ee_contact_sensor_cfg = ContactSensorCfg(
        prim_path=cube_cfg.prim_path,
        update_period=0.0,
        history_length=0,
        debug_vis=False,
        force_threshold=min_contact_force,
        max_contact_data_count_per_prim=4,
        filter_prim_paths_expr=[f"{ur5e_prim_path}/wrist_3_link/"],
    )

    # - reset conditions. Loosened (was 0.1m / 1.0rad) for VOC training: with the virtual
    # controller active the policy may briefly diverge before the controller pulls the box
    # back; tight thresholds would terminate episodes before that recovery completes.
    max_obj_dist_from_traj = 0.2
    max_obj_angle_from_traj = 10000 # just continue if we fail lifting until end

    # === Virtual Object Controller (VOC) curriculum ===
    # DexMachina-style assist: a virtual PD controller drives the cube along its reference
    # trajectory while the policy learns the contact pattern; gain decays exponentially as
    # the policy meets reward thresholds, eventually handing off full control. See
    # `_apply_voc` and `_voc_decay_check` in boxhinge_env.py for the runtime logic.
    voc_enabled: bool = True
    # Translational stiffness chosen so the controller can overpower gravity at typical
    # tracking offsets. For a 4.4 kg box, gravity ≈ 43 N; at 1000 N/m a 5 cm error gives
    # 50 N — comfortably above gravity, so the VOC dominates object dynamics during the
    # high-gain phase. Tune up if the box still droops; tune down if the controller
    # overshoots / oscillates against contact.
    voc_kp_pos: float = 1000.0    # N/m, initial translational stiffness
    voc_kp_rot: float = 100.0     # Nm/rad, initial rotational stiffness
    voc_kp_min: float = 10        # absolute floor; below this kp/kv set to zero
    # Critical-damping multipliers: kv = scale · sqrt(kp · effective_inertia).
    voc_kv_pos_scale: float = 2.0
    voc_kv_rot_scale: float = 2.0
    voc_decay_phi_p: float = 0.99  # multiplicative decay factor on kp per decay event
    voc_decay_phi_v: float = 0.99  # multiplicative decay factor on kv per decay event
    # How often (in env steps) to check the decay condition. Decay only fires when ALL
    # tracked reward means exceed their thresholds. Increased from 24 → 100 so decay
    # can fire at most every ~4 iters (vs every iter); combined with phi=0.99 this gives
    # a smooth, slow decay that doesn't outpace the policy's ability to adapt.
    voc_decay_check_interval: int = 100
    # Trailing window of completed-episode normalized rewards. Smaller window = decay
    # reacts faster to current performance (less lag behind reality), at the cost of
    # noisier mean estimates.
    voc_reward_window_size: int = 100
    # Per-category normalized-reward thresholds. Decay triggers only when the trailing
    # mean of every listed category exceeds its threshold. Higher thresholds = decay
    # only fires once the policy is genuinely tracking well, not just when VOC alone
    # produces high reward.
    # Raised from 0.5 → 0.65 (Step 2 sim2real): with wider DR the trailing means stay
    # comfortably above 0.5 even when 5–10% of episodes terminate early (the successful
    # majority dominate the mean). 0.65 is calibrated so a small completion-rate dip
    # (≈95%) brings the mean below threshold and pauses decay, giving the policy time
    # to adapt to the kp level before VOC weakens further.
    voc_threshold_task: float = 0.8     # task reward (obj_pos · obj_quat or sum form)
    voc_threshold_track: float = 0.0    # tracking reward (eef_box_rel + eef_pos + ...)
    # Warmup period (in env steps via common_step_counter) before any decay can fire.
    # common_step_counter increments by 1 per env step (not per env*step), so this
    # divides by num_steps_per_env=24 to get "iterations": 5000 / 24 ≈ 208 iters.
    # Chosen so decay starts while the policy still has meaningful exploration
    # (mean_noise_std > 0.3 for the boxhinge run), not after it has frozen at ~0.08.
    # If decay starts when the policy is already deterministic, it can't adapt to
    # each kp level — adaptation requires exploring new residual patterns.
    voc_decay_warmup_steps: int = 0


def get_ur5e_cfg(
    prim_path,
    init_pose,
    cfg: BoxhingeEnvCfg,
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
