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
        size=(0.4, 0.6, 0.06),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
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
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.2,
            friction_combine_mode="multiply"
        )
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,-0.5))
)

@configclass
class EventCfg:
    ur5e_l_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            # EE is the wrist_3_link (sphere asset retired); friction DR targets it
            # since it's the body that actually contacts the cube.
            "asset_cfg": SceneEntityCfg("ur5e_left", body_names="wrist_3_link"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            # Lowered from (0.7, 0.9) — bouncy EE was triggering contact chatter against
            # the cube during VOC bring-up; matches boxhinge's calmer-contact setting.
            "restitution_range": (0.1, 0.3),
            "num_buckets": 250,
        },
    )
    ur5e_r_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_right", body_names="wrist_3_link"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.1, 0.3),
            "num_buckets": 250,
        },
    )
    # Actuator-gain DR — widest of all per-cfg DR (±50%) because real UR5e closed-loop
    # dynamics don't match sim's PD; this is the dominant sim2real mismatch. Duplicated
    # per arm because each is a separate articulation.
    ur5e_l_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_left"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    ur5e_r_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ur5e_right"),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    table_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.3, 0.7),
            "dynamic_friction_range": (0.3, 0.7),
            "restitution_range": (0.0, 0.4),
            "num_buckets": 250,
        },
    )
    object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            # Narrowed from (0.5, 2.0) for VOC bring-up — the critical-damping kv defaults
            # are derived from nominal mass, so a 0.5x cube has 2x ω_n and the VOC PD
            # becomes sampled-data unstable. Re-widen after VOC has fully decayed.
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform"
        },
    )
    # object_scale = EventTermCfg(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "scale_range": (0.975, 1.025)
    #     },
    # )
    # object_com = EventTermCfg(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object"),
    #         "com_range": {
    #             "x": (-0.1, 0.1), # not relative to scale but absolute addition uniformly sampled
    #             "y": (-0.1, 0.1),
    #         }
    #     },
    # )
    reset_gravity = EventTermCfg(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.1]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    
@configclass
class BoxliftEnvCfg(DirectRLEnvCfg):
    # Trajectory file path
    trajectory_path = ""
    # env
    # physics_dt * decimations needs to match dt from planner/IK
    physics_dt = 1.0 / 100.0
    decimation = 2
    episode_length_s = 3.0
    # === Observation layout ============================================================
    # use_reference_obs=True  (reference mode):
    #   per-step = rel_q(12) + rel_qd(12) + rel_obj(7) + abs_q(12) + abs_qd(12) + abs_obj(7) + contact(2) = 64
    #   full obs = 64×history + phase(1) + future_box(14×offsets) + prev_actions(12)
    #
    # use_reference_obs=False (goal-conditioned / reference-free):
    #   per-step = abs_q(12) + abs_qd(12) + abs_obj(7) + contact(2) = 33
    #   full obs = 33×history + future_box(14×offsets) + prev_actions(12)   [no phase]
    #   future_box offsets serve as goal waypoints; privileged critic still has full ref.
    #
    # action_space and observation_space recomputed in __post_init__.
    action_space = 12
    observation_space = {"policy": 275, "privileged": 136}  # placeholder; __post_init__ overwrites
    state_space = 0

    # Toggle reference observations. True = current behaviour. False = goal-conditioned:
    # drops relative joint/obj obs and phase; keeps future box waypoints.
    use_reference_obs: bool = True

    # History of per-step features (newest-last, flattened into obs).
    obs_history_steps: int = 3
    # Phase offsets (env steps) for future reference box pose look-ahead.
    # In reference-free mode these become goal waypoints (still meaningful at deployment
    # if the trajectory planner can provide k-step box pose predictions).
    future_obs_steps: tuple = (1, 2, 3, 4, 5)

    # Force magnitude threshold (N) for the per-arm contact bool. ~0.5–2 N is reasonable
    # for the UR5e wrist on the cube; tune by inspecting per-arm force-magnitude
    # histograms in obvious-contact vs free-motion segments.
    contact_threshold: float = 0.5
    # Optional DR on the contact bool: shift the bool through a rolling buffer so the
    # obs sees a value from N env steps ago (models force-estimator latency), and apply
    # a per-step bit-flip rate (models false positives/negatives). 0/0.0 = clean.
    contact_obs_delay_steps: int = 0
    contact_obs_flip_prob: float = 0.0

    # === Simulated tracker for the box pose ==========================================
    # Both the relative AND absolute box obs are read from a buffer that latches the
    # clean cube pose only every obs_obj_update_period env steps, and the latched read
    # is taken from obs_obj_delay_steps ago — models a slow + delayed tracker (e.g. a
    # 25 Hz vision tracker with ~100 ms latency at 50 Hz policy). Reward path keeps
    # using clean ground truth via _get_obj_pos / _get_obj_quat / _get_obj_vel.
    obs_obj_delay_steps: int = 5
    obs_obj_update_period: int = 2
    # Per-FRESH-SAMPLE Gaussian noise applied on every "fire" (when the latched value
    # refreshes). Held readings between fires do NOT re-jitter — same held value goes
    # into the obs until the next fire. 0 disables. Defaults match boxhinge.
    obs_obj_pos_noise: float = 0.001     # m, std on position per fire
    obs_obj_ori_noise: float = 0.001     # rad, axis-angle std on orientation per fire
    # Per-EPISODE constant bias sampled in _reset_idx. Held for the entire episode
    # so the policy can't average it out across frames — models systematic calibration
    # error of the tracker. Applied BEFORE per-fire noise (so bias is the dominant
    # systematic component, noise is residual detection jitter on top). 0 disables.
    obs_obj_pos_bias_std: float = 0.005  # m, std on per-episode constant pos offset
    obs_obj_ori_bias_std: float = 0.005  # rad, std on per-episode constant ori offset

    # Per-step Gaussian noise on joint position/velocity observations. The same noise
    # sample is used for both the absolute and relative obs so the difference (rel = abs -
    # ref) reflects a single consistent noisy reading, not two independent samples.
    # Privileged obs and reward paths read raw .data values and are unaffected.
    obs_joint_pos_noise_std: float = 0.005   # rad
    obs_joint_vel_noise_std: float = 0.05   # rad/s

    def __post_init__(self) -> None:
        # Idempotent: must be safe to call twice (the env's __init__ calls it again
        # after Hydra CLI overrides land on the cfg).
        base_pose = 3 + 4                       # obj pos + quat
        if self.use_reference_obs:
            per_step = 12 + 12 + base_pose      # rel_q + rel_qd + rel_obj
            per_step += 12 + 12 + base_pose     # abs_q + abs_qd + abs_obj
            per_step += 2                       # contact bools
            phase_dim = 1
        else:
            per_step = 12 + 12 + base_pose      # abs_q + abs_qd + abs_obj only
            per_step += 2                       # contact bools
            phase_dim = 0                       # no trajectory clock
        self.per_step_feature_dim = per_step

        future_dim_per_offset = 2 * base_pose   # (rel pos+quat) + (abs pos+quat) per offset
        actor_dim = (
            per_step * self.obs_history_steps
            + phase_dim
            + future_dim_per_offset * len(self.future_obs_steps)
            + 12                                         # prev_actions
        )

        # Privileged-extras layout (concatenated by _get_privileged_obs in this order):
        #   13  clean obj state           (pos 3 + quat 4 + lin_vel 3 + ang_vel 3)
        #   28  DR samples                (obj_mass 1 + obj_friction 3 + stiff_L 6 +
        #                                  stiff_R 6 + damp_L 6 + damp_R 6)
        #   75  reference state @ phase   (ref_obj_pos 3 + quat 4 + lin_vel 3 + ang_vel 3
        #                                  + ref_joints_L 6 + R 6 + ref_joint_vels_L 6 + R 6
        #                                  + ref_joints_target_L 6 + R 6
        #                                  + planner_pd_err_L 6 + R 6
        #                                  + ref_EE_pose_L 7 + R 7)
        #   12  force/contact             (EE force mag L 1 + R 1 + EE force dir L 3 + R 3
        #                                  + illegal_contact cube 1 + table 1
        #                                  + flange-forearm dist L 1 + R 1)
        #    4  eef-box rel scalars       (pos_err_L + quat_err_L + pos_err_R + quat_err_R)
        #    4  voc / curriculum context  (current_seg kp_pos + kp_rot + alpha + seg_idx)
        priv_dim = 13 + 28 + 75 + 12 + 4 + 4

        self.observation_space = {"policy": actor_dim, "privileged": priv_dim}

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=physics_dt, render_interval=decimation, gravity=(0,0,-9.8))

    # Domain Randomization
    events: EventCfg = EventCfg()

    ur5e_l_prim_path = f"{ENV_REGEX}/ur5e_l"
    ur5e_r_prim_path = f"{ENV_REGEX}/ur5e_r"

    # Arm Actuator parameters. Mode B drops the planner feedforward, so the PD has to
    # hold against gravity alone — bumped from kp=100/kd=20 to kp=150/kd=22.5 to track
    # the planner trajectory accurately without the FF term.
    kp = 150.0
    kd = 22.5
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

    # object (cube)
    cube_cfg = CUBE_CFG

    # table
    table_cfg = TABLE_CFG

    # scene
    replicate_physics = bool(np.all([event["mode"] != "prestartup" and event["mode"] != "startup" for event in events.to_dict().values()])) # type: ignore
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=replicate_physics)

    # Action formulation. One of:
    #   "A"  — joints_target[t] + action_scale * action   (residual on planner FF target)
    #   "B"  — joints[t]        + action_scale * action   (residual on trajectory positions;
    #          planner FF not applied — PD has to do the work; use higher kp/kd)
    #   "C"  — q_current        + action_scale * action   (residual on current joint position)
    #   "BC" — (1-α)·joints[t] + α·q_current + action_scale * action
    #          ≡ q_current + (1-α)·(joints[t] - q_current) + action_scale * action.
    #          A decaying spring toward the trajectory reference. At α=0 → mode B
    #          (full spring); at α=1 → mode C (no spring, pure residual on current).
    #          Action has full authority throughout — only the assist decays. Equivalent
    #          to a per-joint virtual controller with kp_vrc = (1-α)·kp, kd_vrc = 0.
    #   "D"  — q_current + (1-α)·(joints_target[t] - joints[t]) + (α + ε(1-α))·scale·a
    #          Planner-PD-error feedforward transplanted to the CURRENT actual joint
    #          position, blended with a residual whose authority is gated by α. At α=0
    #          → mostly planner PD with a small ε·scale residual floor (≡ mode A
    #          transplanted to current position, with limited policy authority). At α=1
    #          → mode C (pure residual from current position, no FF). Difference vs BC:
    #          BC decays the spring magnitude while keeping full action authority; D
    #          decays the FF authority AND gates action authority together.
    action_mode = "A"

    # Curriculum α schedule. Used by modes BC and D.
    # force_alpha ∈ [0,1]: pin α to a fixed value (eval override; bypasses all schedules).
    force_alpha: float = -1
    # Mode D only: residual authority floor at α=0. The action gain is α + ε·(1-α).
    action_alpha_floor: float = 1.0
    # alpha_warmup_steps > 0: linear ramp (fallback when alpha_curriculum_enabled=False).
    alpha_warmup_steps: int = 0

    # Reward-based BC alpha curriculum (sequential: only activates after VOC reaches 0).
    # (1-α) decays by phi each time mean task reward >= threshold; snaps to α=1 when
    # (1-α) < alpha_min_support. Mirrors VOC gain decay exactly.
    alpha_curriculum_enabled: bool = True
    alpha_decay_phi: float = 0.95          # (1-alpha) *= phi per passing check
    alpha_min_support: float = 0.01        # snap alpha=1 when (1-alpha) < this
    alpha_threshold_task: float = 0.7      # mean task reward needed to trigger increase
    alpha_reward_window_size: int = 100    # ring buffer capacity (completed episodes)
    alpha_decay_check_interval: int = 240  # env steps between checks
    alpha_decay_warmup_steps: int = 0      # env steps to wait before any increase

    # Action scale (scalar or per-joint list of length 12; broadcast against actions)
    action_scale: float | list = 0.1

    # Reset noise (sim-to-real robustness). Per-joint (length 6, ordered shoulder_pan,
    # shoulder_lift, elbow, wrist_1, wrist_2, wrist_3) — same per-joint std applied to both
    # arms independently. Wrist joints have small Jacobian on EE position so they can
    # absorb more noise without dragging the EE far from the trajectory. Scalar also OK.
    # Halved for BC-curriculum bring-up. At α=0 the PD pulls hard toward q_ref, and 0.2 rad
    # of wrist noise creates a ~30 N·m first-step impulse (kp=150) that spikes joint_acc /
    # action_rate before the controller settles. Re-widen once the policy is reliably
    # tracking (Step 2 sim2real, after VOC has fully decayed).
    reset_joint_pos_noise = [0.05, 0.05, 0.05, 0.1, 0.1, 0.1]
    reset_joint_vel_noise = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    # Box reset noise: xy-plane for translation / linear vel; yaw-only orientation noise
    # so the box stays flat; z-axis-only angular vel for consistency with the yaw noise.
    reset_obj_pos_xy_noise = 0.01        # m, std on box x,y position
    reset_obj_lin_vel_xy_noise = 0.0     # m/s, std on box linear x,y velocity
    reset_obj_ori_noise = 0.03           # rad, axis-angle std for small yaw perturbation
    reset_obj_ang_vel_noise = 0.0        # rad/s, std on box angular velocity (z axis)

    # Per-env robot base pose randomization, sampled once at startup and held fixed for the
    # entire training run. Models real-robot placement uncertainty; the RL residual compensates.
    # With 1024 envs a 1-2 cm XYZ std and 2-5° yaw std give broad coverage.
    robot_pos_randomization_xyz_std: float = 0.01   # m, std on XYZ base position offset (independent per arm)
    robot_ori_randomization_yaw_std: float = 0.02   # rad, std on yaw (Z-axis rotation) offset (independent per arm)

    # Probability that a reset overrides random RSI and starts the episode at phase=0
    # instead. Pure RSI samples phase=0 with prob ~1/T (often <1%) — far too low for
    # multi-phase tasks where the early phase (lift) is critical but short. Deployment
    # always starts at 0, so the first frames are heavily under-trained otherwise.
    # Use ~0.5 during bring-up if the early phase isn't learning; drop to ~0.05 once
    # it does. 0.0 disables.
    reset_to_zero_prob: float = 0.1

    # Reward parameteres (aligned with boxhinge — DexMachina §4.2: λ_task is the largest
    # weight. Track is guidance; regularization budget high enough to actually shape policy.)
    w_task = 0.7
    w_track = 1 - w_task
    w_regularization = 1.0

    # Task reward parameteres (boxhinge values)
    w_obj_pos = 0.5
    sigma_obj_pos = 0.05
    tol_obj_pos = 0.0

    w_obj_quat = 0.5
    sigma_obj_quat = 0.15
    tol_obj_quat = 0.0

    # TODO: Use or delete? (env doesn't currently compute an obj-vel reward term;
    # boxhinge keeps them at w=0 with a velocity-split form — kept here for parity.)
    w_obj_vel = 0.0
    sigma_obj_vel = 0.0
    tol_obj_vel = 0.0

    # Track reward parameters. Sigmas tightened to match boxhinge (was 0.1 / 0.5) so the
    # gradient between "rough" and "precise" tracking is steep enough to overcome PPO's
    # entropy noise — going 5cm err → 1cm now yields a much bigger reward delta.
    w_eef_pos = 1.0
    sigma_eef_pos = 0.075     # 1cm err: 0.89; 3cm err: 0.37
    tol_eef_pos = 0.0

    # TODO: Use or delete?
    w_eef_quat = 0.0
    sigma_eef_quat = 0.2     # 10° err: 0.79; 25° err: 0.20
    tol_eef_quat = 0.1

    w_joint_pos = 0.0
    # Tuned for the per-joint averaged kernel (DexMachina r_bc form). At σ=0.1, a single
    # joint with 0.1 rad error gives kernel exp(-0.01/0.01) ≈ 0.37; 0.2 rad gives ≈ 0.018.
    # The mean is taken across ALL 12 joints (both arms), giving a [0, 1] bounded reward.
    sigma_joint_pos = 0.1
    tol_joint_pos = 0.0

    # Relative EE-in-box-frame tracking: rewards matching the reference's EE-in-box-frame
    # offset. Gated on |obj_vel_ref| > eps, dilated by ±dilation_steps so it captures brief
    # pre-contact approach and post-release follow-through. Mutually exclusive with the
    # absolute EE/quat/joint trackers (those use abs_gate = 1 - gate).
    w_eef_box_rel_pos = 0.7
    sigma_eef_box_rel_pos = 0.075
    tol_eef_box_rel_pos = 0.0

    w_eef_box_rel_quat = 0.3
    sigma_eef_box_rel_quat = 0.5
    tol_eef_box_rel_quat = 0.0

    # Gate: a reference step is "active" if ||obj_vel_lin|| + ||obj_vel_ang|| > eps, then
    # dilated by ±dilation_steps. Large dilation_steps effectively disables gating.
    eef_box_gate_obj_vel_eps = 1e-3
    eef_box_gate_dilation_steps = 1e7

    # Regularization reward parameters.
    w_joint_acc = 1e-5
    tol_joint_acc = 0.0

    w_joint_torque = 5e-5
    tol_joint_torque = 0.0

    # Rate penalty: primary anti-jitter term. Needs to be high enough that consecutive
    # step action differences are penalised, but low enough that the task reward (~1.0
    # scale) still dominates.  2.5e-1 collapsed training (reg>>reward); 5e-2 was too
    # weak to damp deployment oscillation. 1e-1 is the working compromise.
    w_action_rate = 1e-1
    tol_action_rate = 0.0

    w_action_norm = 1e-3
    tol_action_norm = 0.0

    w_joint_limit = 5e2             # boxhinge 1e3 / 2 (sums over 12 joints × 2 arms)
    # Margin inset from the hard joint limit before the penalty starts firing.
    joint_limit_eps = 0.05

    w_proximity_to_contact = 0.5
    max_proximity = 0.05

    # Lowered from boxhinge's 50.0 for dual-arm: with two arms wrapping a 0.4×0.6 m cube,
    # forearms and wrists naturally pass within cm of the cube during the grip, so the
    # number of brush-contacts firing per step is 2–3× boxhinge's single-arm regime.
    # At 50 the penalty hits 100–140 (=2–2.8 N total), drowning the task signal entirely
    # (w_task·R_task tops out near 0.9 while w_reg·rew_illegal_contact is ≥ 100).
    w_illegal_contact = 0.1
    # Reject sensor reports below 1 N — filters out persistent micro-contact / sensor
    # noise that isn't a real safety violation. Real impacts (5+ N) still register.
    min_contact_force = 1
    max_contact_force = 20

    # Halved from boxhinge's 1.0 because env sums (is_too_close_l + is_too_close_r) —
    # range [0, 2] vs boxhinge's [0, 1]. Keeps per-arm pressure identical.
    w_flange_forearm_dist = 0.5
    max_flange_forearm_distance = 0.028 + 0.0375

    # Contact Sensors. wrist_3_link is now the EE (no separate sphere asset), so it's
    # ALLOWED to contact the cube (that's the grip) but still not the table. Split filters
    # follow boxhinge: cube_contact_filter excludes wrist_3_link; table_contact_filter
    # includes it.
    non_ee_link_names = [
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "forearm_link",
        "wrist_1_link",
        "wrist_2_link",
    ]

    cube_contact_filter = []
    for robot_prim_path in [ur5e_l_prim_path, ur5e_r_prim_path]:
        for link_name in non_ee_link_names:
            cube_contact_filter.append(f"{robot_prim_path}/{link_name}/")
    table_contact_filter = list(cube_contact_filter)
    for robot_prim_path in [ur5e_l_prim_path, ur5e_r_prim_path]:
        table_contact_filter.append(f"{robot_prim_path}/wrist_3_link/")

    ee_contact_sensors = [
        ContactSensorCfg(
            prim_path=f"{ur5e_prim_path}/wrist_3_link",
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=0,
            filter_prim_paths_expr=[CUBE_CFG.prim_path, TABLE_CFG.prim_path]
        )
        for ur5e_prim_path in [ur5e_l_prim_path, ur5e_r_prim_path]
    ]

    illegal_contact_sensor_cfgs = {
        "cube": ContactSensorCfg(
            prim_path=cube_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=cube_contact_filter,
        ),
        "table": ContactSensorCfg(
            prim_path=table_cfg.prim_path,
            update_period=0.0,
            history_length=0,
            debug_vis=True,
            force_threshold=min_contact_force,
            max_contact_data_count_per_prim=16,
            filter_prim_paths_expr=table_contact_filter,
        ),
    }

    # - reset conditions. Loosened from 0.1 → 0.2 for VOC training: the virtual controller
    # may briefly pull the box past the old threshold before recovering, and we don't want
    # to terminate during that recovery. Angle effectively disabled (matches boxhinge) —
    # the box-lift task allows large yaw deviations while the policy recovers; obj_pos
    # termination is sufficient.
    max_obj_dist_from_traj = 0.2
    max_obj_angle_from_traj = 10000

    # === Virtual Object Controller (VOC) curriculum ===
    # DexMachina-style assist: a virtual PD wrench drives the cube along its reference
    # trajectory while the policy learns the contact pattern; gain decays exponentially as
    # the policy meets reward thresholds, eventually handing off full control. See
    # `_apply_voc` and `_voc_decay_check` in boxlift_env.py for the runtime logic.
    voc_enabled: bool = True
    # Translational stiffness chosen so the controller can overpower gravity at typical
    # tracking offsets. For a 1 kg cube, gravity ≈ 10 N; at 1000 N/m a 1 cm error gives
    # 10 N — comfortably above gravity. ω_n = √(kp/m) ≈ 31 rad/s, ω·h = 0.63 at 50 Hz —
    # within the sampled-data stability boundary.
    voc_kp_pos: float = 1000.0    # N/m, initial translational stiffness
    voc_kp_rot: float = 100.0     # Nm/rad, initial rotational stiffness
    voc_kp_min: float = 0.1       # absolute floor; below this kp/kv set to zero
    # Critical-damping multipliers: kv = scale · sqrt(kp · effective_inertia).
    voc_kv_pos_scale: float = 2.0
    voc_kv_rot_scale: float = 2.0
    voc_decay_phi_p: float = 0.98  # multiplicative decay factor on kp per decay event
    voc_decay_phi_v: float = 0.98  # multiplicative decay factor on kv per decay event
    # How often (in env steps) to check the decay condition. Decay fires only when ALL
    # tracked reward means exceed their thresholds; rate-limited so decay can fire at most
    # every ~4 iters (vs every iter), giving the policy time to adapt to each kp level.
    voc_decay_check_interval: int = 100
    # Trailing window of completed-episode normalized rewards. Smaller window = decay
    # reacts faster to current performance, at the cost of noisier mean estimates.
    voc_reward_window_size: int = 100
    # Per-category normalized-reward thresholds (in [0,1]). Decay only triggers once the
    # trailing mean of every category exceeds its threshold — ensures the policy is
    # genuinely tracking well before VOC weakens further.
    voc_threshold_task: float = 0.7     # task reward (obj_pos + obj_quat), max 1.0
    # L+R kernels are now averaged in _get_rewards (rew_EE_* and rew_eef_box_rel_*),
    # so rew_track_unweighted_per_step is back in [0, 1.0] — same scale as boxhinge.
    voc_threshold_track: float = 0.0    # tracking reward (mean of L+R kernels, max 1.0)
    # Warmup period (in env steps) before any decay can fire. Without this gate, decay
    # would start firing while VOC is doing all the work and the policy hasn't learned
    # to compensate; 0 means warmup-off (decay can fire as soon as the buffer fills).
    voc_decay_warmup_steps: int = 0

    # When True, the env additionally stashes per-env (num_envs,) reward-component tensors
    # under `extras["log_per_env"]` so record.py can save them for per-env post-processing
    # (e.g. inspecting one env's torque/action_rate trace). Off by default — leaving these
    # tensors in the per-step extras during training risks the runner traversing them and
    # logging garbage, plus the extra GPU→CPU transfers add overhead. record.py flips this
    # on after loading the cfg.
    emit_per_env_extras: bool = False

    # === Phase-segmented VOC ===
    # The trajectory is split into segments and each gets its OWN gain that decays
    # independently. Easy segments (e.g. free-motion approach) clear thresholds early
    # and lose assist quickly, getting lots of solo-practice time → sim2real-ready.
    # Hard segments (lift, contact phases) keep assist until they're actually learned.
    #
    # voc_segmentation:
    #   "none"    — single global segment (legacy behavior, one gain, one decay decision)
    #   "uniform" — N equal-length bins (set voc_n_uniform_segments). Generic, no semantic
    #               alignment with task phases.
    #   "contact" — segments derived from eef_box_gate_mask transitions (gate True→False
    #               and False→True boundaries). Produces clean approach / contact / retreat
    #               segments. Recommended for box-lift / box-hinge tasks where the contact
    #               phase is the hard one.
    voc_segmentation: str = "contact"
    voc_n_uniform_segments: int = 5     # used only when voc_segmentation == "uniform"
    # Dilation applied to the contact gate BEFORE computing segment boundaries — INDEPENDENT
    # of `eef_box_gate_dilation_steps` (which is for the reward path and often set huge to
    # always reward eef_box_rel). Segmentation needs a small dilation (3–10 frames) so it
    # smooths single-frame flickers without losing the major lift/rotate/place transitions.
    # Used only when voc_segmentation == "contact".
    voc_segment_dilation_steps: int = 10

    # === Focused RSI by segment difficulty ===
    # When > 0, a fraction of resets bias the starting phase toward segments still
    # under VOC assist (high kp). Auto-tracks the bottleneck: as easy segments decay,
    # focus shifts to still-hard segments. Decayed segments (kp=0) get zero focused
    # exposure — they're covered by the (1 - reset_segment_focus_prob) uniform-RSI
    # fraction, which is plenty of forgetting protection at typical settings.
    # Layered on top of `reset_to_zero_prob` — envs already snapped to phase=0 are
    # not re-focused.
    reset_segment_focus_prob: float = 0.7
    # Weight = kp[s]^beta. beta=1 → linear proportional; 2 → aggressive (squared
    # difference between active and decayed segments); 0.5 → gentler. When all
    # segments have fully decayed, focused sampling falls back to uniform within
    # the focused budget (same as the uniform-RSI fraction).
    segment_focus_beta: float = 1.0


def get_ur5e_cfg(
    prim_path,
    init_pose,
    box_lift_cfg: BoxliftEnvCfg,
):
    actuator_kwargs = dict(
        joint_names_expr=[".*"],
        stiffness=box_lift_cfg.kp,
        damping=box_lift_cfg.kd,
        velocity_limit=box_lift_cfg.velocity_limit,
        effort_limit=box_lift_cfg.effort_limit,
    )

    if box_lift_cfg.actuator_type == "IdealPD":
        actuator_cfg = IdealPDActuatorCfg(**actuator_kwargs)
    elif box_lift_cfg.actuator_type == "Implicit":
        actuator_cfg = ImplicitActuatorCfg(**actuator_kwargs)
    else:
        raise ValueError(f"Unknown actuator type: {box_lift_cfg.actuator_type}")

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
