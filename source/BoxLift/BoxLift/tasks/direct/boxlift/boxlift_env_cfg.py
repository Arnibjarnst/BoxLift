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
    # - spaces definition
    action_space = 12
    observation_space = 38
    state_space = 0

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
    #   "D"  — q_current + (planner PD error) + action_scale * action  (mode-C + planner
    #          FF with curriculum α; not implemented in boxlift yet)
    action_mode = "BC"

    # Curriculum α schedule. Used by mode BC (and mode D when implemented).
    #   α ramps linearly from 0 to 1 over `alpha_warmup_steps` env steps; 0 disables the
    #   curriculum (α=1 always — collapses BC to mode C).
    #   `common_step_counter` increments once per env step, so for a 5000-iter run with
    #   num_steps_per_env=24 you have ~120k env steps total — pick warmup ~50k–80k so the
    #   spring is mostly gone by the end while leaving time for a fully-decayed phase.
    alpha_warmup_steps: int = 24 * 1000
    # Optional fixed-α override (≥ 0). Bypasses the schedule — used at eval to pin α to
    # the value the policy was trained at (otherwise common_step_counter resets to 0
    # during eval and the spring would re-appear).
    force_alpha: float = -1

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
    reset_obj_pos_xy_noise = 0.02        # m, std on box x,y position
    reset_obj_lin_vel_xy_noise = 0.05    # m/s, std on box linear x,y velocity
    reset_obj_ori_noise = 0.1            # rad, axis-angle std for small yaw perturbation
    reset_obj_ang_vel_noise = 0.1        # rad/s, std on box angular velocity (z axis)

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

    # Regularization reward parameters. HALVED from boxhinge's per-joint values because
    # boxlift sums these penalties over 12 joints / 2 arms vs boxhinge's 6 / 1; without
    # halving, the per-joint pressure is 2× what boxhinge tuned for, smothering action
    # freedom and trapping the policy in a small-residual basin around the trajectory.
    # (joint_acc, torque, action_rate, action_norm, joint_limit all sum over the joint
    # axis; the boxhinge value × 0.5 keeps per-joint pressure identical.)
    w_joint_acc = 1e-5              # boxhinge 2e-4 / 2
    tol_joint_acc = 0.0

    w_joint_torque = 5e-5           # boxhinge 1e-3 / 2
    tol_joint_torque = 0.0

    # Rate penalty: at 0 the policy outputs wildly different residuals on consecutive
    # steps, jittering faster than kp can track. Boxhinge picked 0.5 to keep commands
    # coherent without preventing time-varying residuals for FF compensation. Halved here.
    w_action_rate = 2.5e-2          # boxhinge 5e-1 / 2
    tol_action_rate = 0.0

    # Residual-magnitude penalty (separate from rate): biases the policy toward zero
    # residual when the nominal plan is already good. Important during VOC: the policy
    # should output ~0 residual at the start so VOC's wrench is the only "actor".
    w_action_norm = 1e-3          # boxhinge 5e-3 / 2
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
    w_illegal_contact = 1.0
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
    voc_kp_min: float = 10.0      # absolute floor; below this kp/kv set to zero
    # Critical-damping multipliers: kv = scale · sqrt(kp · effective_inertia).
    voc_kv_pos_scale: float = 2.0
    voc_kv_rot_scale: float = 2.0
    voc_decay_phi_p: float = 0.99  # multiplicative decay factor on kp per decay event
    voc_decay_phi_v: float = 0.99  # multiplicative decay factor on kv per decay event
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
    voc_threshold_task: float = 0.6     # task reward (obj_pos + obj_quat), max 1.0
    # L+R kernels are now averaged in _get_rewards (rew_EE_* and rew_eef_box_rel_*),
    # so rew_track_unweighted_per_step is back in [0, 1.0] — same scale as boxhinge.
    voc_threshold_track: float = 0.5    # tracking reward (mean of L+R kernels, max 1.0)
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
