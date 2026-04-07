# CLAUDE.md — BoxLift: Dual-Arm Residual RL for Contact-Rich Manipulation

## What this repo does
This repo trains a **residual RL policy** in NVIDIA IsaacLab that augments a nominal
trajectory controller for **dual-arm box lifting** on UR5e robots. The policy learns joint
position corrections on top of pre-computed IK trajectories (from the `planning_through_contact`
repo) to handle contact dynamics the nominal planner cannot model precisely.

- **Robot:** 2x UR5e with sphere end-effectors (`robots/ur5e_sphere.usd`)
- **Task:** Dual-arm extrinsic manipulation — lift a box off a table using two arms
- **Nominal controller:** Pre-computed IK joint trajectories loaded from `.npz` files
- **RL algorithm:** PPO via RSL-RL (also supports RL-Games, SB3, SKRL)
- **Workflow type:** Direct (`DirectRLEnv`)
- **Real robot interface:** `ur_rtde` library using `servoJ()` at 500 Hz

## Repo structure
```
source/BoxLift/BoxLift/
  tasks/direct/
    boxlift/                    # Main task: dual-arm box lifting with object tracking
      boxlift_env.py            # DirectRLEnv implementation
      boxlift_env_cfg.py        # All configs: scene, actuators, rewards, DR, contacts
      agents/rsl_rl_ppo_cfg.py  # PPO hyperparameters
      __init__.py               # Registers "Template-Boxlift-Direct-v0"
    jointtarget/                # Simpler task: joint target tracking only (no object rewards)
      joint_target_env.py
      joint_target_env_cfg.py   # Per-joint kp/kd, supports DelayedPDActuator
      agents/rsl_rl_ppo_cfg.py
      __init__.py               # Registers "Follow-Joint-Targets"
scripts/
  rsl_rl/
    train.py                    # Training entry point (--task, --trajectory_path, --num_envs)
    play.py                     # Eval + exports ONNX/JIT models
    record.py                   # Record rollout data to JSON
    cli_args.py                 # Shared argument parsing
  ur_rtde_real_time.py          # Deploy ONNX policy on real UR5e via servoJ (main deployment script)
  ur_rtde_fixed_traj.py         # Play back fixed trajectories on real robot (sysid / baseline)
  ur_rtde_test.py               # Harmonic oscillation test for measuring motor delays
  IK.py                         # Generate IK trajectories from planner output (Lula solver)
  visualize_traj.py             # Visualize trajectories in Isaac Sim
  follow_joint_targets.py       # Simple trajectory playback in sim
  zero_agent.py / random_agent.py  # Baseline agents for env testing
robots/                         # USD/URDF robot assets
  ur5e_sphere.usd               # Primary asset (UR5e + sphere EE)
  ur5.yaml                      # Kinematics config for Lula IK solver
reference_trajectories/         # Pre-computed nominal trajectories (.npz)
  box_lift_ur5e/
data/box_lift/                  # Older IK trajectory data
logs/
  rsl_rl/                       # Training checkpoints + TensorBoard (gitignored)
  ur_rtde/                      # Real robot execution logs (CSV + .log)
experiments/                    # Hyperparameter sweep shell scripts
```

## Key dependencies
- **IsaacLab**: `/home/arni/IsaacLab/` (v5.0+)
- **Isaac Sim**: 4.5.0+ / 5.0.0+ / 5.1.0+
- **RSL-RL**: v3.0.1+ (supports `OnPolicyRunner` and `DistillationRunner`)
- **planning_through_contact**: `/home/arni/planning_through_contact/` — generates the nominal
  IK trajectories that the residual policy corrects
- **ur_rtde**: Real robot RTDE SDK (`rtde_control`, `rtde_receive`)
- **ONNX Runtime**: For real-time policy inference on hardware
- **PyTorch + CUDA**
- **W&B**: Training logging (configured as default logger)

## Build & run
```bash
# Install this extension
python -m pip install -e source/BoxLift

# Train
python scripts/rsl_rl/train.py \
  --task Template-Boxlift-Direct-v0 \
  --trajectory_path reference_trajectories/box_lift_ur5e/IK_20260328_155357.npz \
  --num_envs 4096 --headless

# Evaluate (exports ONNX + JIT automatically)
python scripts/rsl_rl/play.py \
  --task Template-Boxlift-Direct-v0 \
  --trajectory_path reference_trajectories/box_lift_ur5e/IK_20260328_155357.npz \
  --num_envs 16

# Deploy on real UR5e
python scripts/ur_rtde_real_time.py \
  --onnx_model_path logs/rsl_rl/boxlift/<run>/exported/policy.onnx \
  --reference_trajectory_path reference_trajectories/box_lift_ur5e/IK_20260328_155357.npz \
  --real_robot
```

## Residual policy architecture
```
q_target = q_nominal(t) + action_scale * pi_residual(obs)
```

- **Action space:** 12D joint position residuals (6 per arm)
- **Action scaling:** `action_scale = 0.25` (multiplied before adding to nominal)
- **Clamping:** Joint targets clamped to articulation joint limits after applying residual
- **Nominal source:** `.npz` trajectory file loaded at env init, indexed by `episode_length_buf`

### Sim controller
- **Actuator:** `ImplicitActuatorCfg` (default) or `IdealPDActuatorCfg`
- **Gains (boxlift):** kp=100, kd=20 (uniform all joints)
- **Gains (jointtarget):** Per-joint: shoulder/elbow kp=150/kd=15, wrist kp=28/kd=2.8
- **Effort limit:** 87 Nm (boxlift) or per-joint 150/28 Nm (jointtarget)
- **Velocity limit:** 50 rad/s (boxlift) or 3.14 rad/s (jointtarget)
- **Actuator options:** `DelayedPDActuatorCfg` available in jointtarget (min_delay=1, max_delay=2)

### Real robot controller
- **Command:** `servoJ()` via `ur_rtde` at **500 Hz** RTDE frequency
- **servoJ params:** lookahead_time=0.03, gain=100
- **Policy decimation:** 10 (policy runs at 50 Hz, servoJ interpolates at 500 Hz)
- **Interpolation:** Linear between previous and current policy targets
- **Threading:** Separate policy inference thread + control thread with data lock
- **Robot IP:** 192.168.1.100 (real), 192.168.56.1 (Linux sim), 172.29.144.1 (Windows sim)

## Simulation parameters

| Parameter | Boxlift | JointTarget |
|-----------|---------|-------------|
| Physics dt | 1/100 s (10 ms) | 1/100 s (10 ms) |
| Decimation | 2 | 2 |
| **Control freq** | **50 Hz** | **50 Hz** |
| Episode length | dynamic (matches trajectory) | 3.0 s |
| Num envs | 4096 | 1024 |
| Env spacing | 4.0 m | 4.0 m |

**Critical constraint:** `physics_dt * decimation` must equal the trajectory dt (0.02 s).

## Observations (38D for boxlift)

| Component | Dims | Details |
|-----------|------|---------|
| Joint pos (relative) | 12 | `q_actual - q_trajectory` for both arms |
| Joint vel | 12 | Absolute joint velocities, both arms |
| Object pos (relative) | 3 | `obj_pos - desired_obj_pos` |
| Object quat (relative) | 4 | `quat_mul(desired, quat_inv(actual))` |
| Object vel | 6 | Linear + angular root velocity |
| Episode phase | 1 | `episode_length_buf / max_episode_length` |

Normalization: Running mean/std via RSL-RL's `actor_obs_normalization=True`.

**Real robot obs construction** (in `ur_rtde_real_time.py`):
- `relative_q` from `rtde_r.getActualQ() - trajectory[step]`
- `q_vel` from `rtde_r.getActualQd()`
- Phase from `step / (total_steps - 1)`
- Right arm obs zeroed out (single-arm deployment currently)

## Rewards
`R_total = R_task + R_track - R_regularization`

### Task reward (w_task=0.7)
| Term | Weight | Sigma | What |
|------|--------|-------|------|
| obj_pos | 0.6 | 0.02 | Gaussian on object position error^2 |
| obj_quat | 0.4 | 0.1 | Gaussian on object orientation error^2 |

### Tracking reward (w_track=0.3)
| Term | Weight | Sigma | What |
|------|--------|-------|------|
| eef_pos | 1.0 | 0.1 | Gaussian on EE position error^2 (both arms) |
| eef_quat | 0.0 | 0.5 | Disabled |
| joint_pos | 0.0 | 0.1 | Disabled |

### Regularization (w_regularization=0.25, subtracted)
| Term | Weight | What |
|------|--------|------|
| joint_acc | 1e-6 | Sum of squared joint accelerations |
| joint_torque | 1e-5 | Sum of squared applied torques |
| action_rate | 1e-1 | Sum of squared action deltas between steps |
| joint_limit | 50.0 | Squared violation of joint limits |
| illegal_contact | 0.1 | Non-EE link contact forces on cube/table (clamped 0-20 N) |
| flange_forearm_dist | 1.0 | Binary penalty if flange < 0.0655m from forearm |

Reward function: `exp(-error / sigma^2)` (Gaussian kernel, not squared sigma in denominator).

## Domain randomization (applied on reset)

| What | Range | Distribution |
|------|-------|-------------|
| Robot EE (Sphere) friction | static/dynamic: (0.7, 1.3) | Uniform, 250 buckets |
| Object friction | static/dynamic: (0.3, 0.7) | Uniform |
| Table friction | static/dynamic: (0.3, 0.7) | Uniform |
| Object mass | scale (0.5, 2.0) × 1.0 kg | Uniform |
| Gravity z-noise | mean=0, std=0.1 m/s^2 | Gaussian additive |
| Object scale | Commented out (was ±2.5%) | — |
| Object CoM | Commented out (was ±0.1m) | — |

## Termination & reset
- **Timeout:** `episode_length_buf >= max_episode_length - 1`
- **Task failure:** Object position error > 0.1 m OR orientation error > 0.5 rad
- **Reset:** Random starting point within trajectory (`randint(0, max_episode_length-2)`)
  - Writes joint pos/vel from trajectory to both arms
  - Writes object pose/vel from trajectory

## PPO configuration (RSL-RL)
| Parameter | Value |
|-----------|-------|
| Network | [256, 256] ELU, actor+critic |
| Learning rate | 3e-4, adaptive KL schedule (desired_kl=0.01) |
| Gamma / Lambda | 0.99 / 0.95 |
| Clip param | 0.2 |
| Entropy coeff | 0.005 |
| Steps per env | 24 |
| Mini-batches | 4 |
| Learning epochs | 5 |
| Max iterations | 5000 |
| Save interval | 100 |
| Init noise std | 1.0 |
| Logger | wandb |

## Trajectory file format (.npz)
Loaded in `_setup_scene()` from `--trajectory_path` argument.

| Key | Shape | Content |
|-----|-------|---------|
| `obj_poses` | (T, 7) | Object [pos_xyz, quat_xyzw] per timestep |
| `obj_vel` | (T, 6) | Object [lin_vel, ang_vel] |
| `arm_l_pose` / `arm_r_pose` | (7,) | Arm base poses [pos_xyz, quat_xyzw] |
| `joints_l` / `joints_r` | (T, 6) | Joint positions per timestep |
| `joint_vel_l` / `joint_vel_r` | (T, 6) | Joint velocities |
| `joints_target_l` / `joints_target_r` | (T, 6) | Nominal joint targets (residual is added to these) |
| `EE_poses_l` / `EE_poses_r` | (T, 7) | End-effector poses [pos_xyz, quat_xyzw] |

Timestep dt = 0.02 s (hardcoded in env, must match `physics_dt * decimation`).

## Real robot deployment details

### ur_rtde_real_time.py (main deployment)
1. Loads ONNX policy + reference trajectory
2. `moveJ()` to initial joint position
3. Spawns two threads:
   - **Policy thread:** Reads `getActualQ()` + `getActualQd()`, runs ONNX inference,
     computes `q_target = joints_target[step] + ACTION_SCALE * output[:6]`
   - **Control thread:** At 500 Hz, interpolates between previous/current targets,
     calls `servoJ(interp_q, vel, acc, dt, lookahead_time=0.03, gain=100)`
4. Logs CSV: actual_q, expected_q, loop_time, robot_mode, safety_mode
5. Safety: Aborts on safety_mode != 1 (NORMAL) or robot_mode != 7 (RUNNING)

**ACTION_SCALE on real robot: 0.2** (hardcoded, differs from sim's 0.25 — TODO in code)

### ur_rtde_fixed_traj.py (sysid / baseline)
- Generates harmonic oscillation trajectory: `joints_0 + amplitude * sin(t + phase)`
- Sends via `servoJ()` at 500 Hz
- Measures per-joint motor delay (time from first command to detected motion > 1e-3 rad)
- Logs all data for system identification

### UR5e max torques (for reference)
`[150.0, 150.0, 150.0, 28.0, 28.0, 28.0]` Nm (shoulder/elbow/wrist)

## PhysX articulation settings
```python
rigid_props:
  disable_gravity: False
  max_depenetration_velocity: 10.0
  enable_gyroscopic_forces: True
articulation_props:
  enabled_self_collisions: True
  solver_position_iteration_count: 4
  solver_velocity_iteration_count: 0
  sleep_threshold: 0.005
  stabilization_threshold: 0.001
  fix_root_link: True
```

## Scene objects
- **Ground plane:** z = -0.5
- **Cube:** 0.4 x 0.6 x 0.06 m, mass 1.0 kg, green, dynamic
- **Table:** 1.5 x 1.5 x 1.0 m, kinematic, gray
- **UR5e left:** Positioned from trajectory `arm_l_pose`
- **UR5e right:** Positioned from trajectory `arm_r_pose`

## Contact sensors
- **EE sensors** (2): On each arm's Sphere body, filtered to cube + table contacts
- **Illegal contact sensors** (2 for boxlift, 1 for jointtarget): On cube/table, filtered
  to non-EE links (base, shoulder, upper_arm, forearm, wrist_1/2/3)
- **Wrist_3 sensors** (jointtarget only): Track contact points and air time with table

## Cross-repo notes
This repo consumes **IK trajectories** from `planning_through_contact`:
- Generated via `scripts/IK.py` using IsaacSim's Lula Kinematics Solver
- Input: Contact planner output (EE positions/orientations)
- Output: `.npz` file saved to `data/box_lift/` or `reference_trajectories/`
- The planner provides the full contact-aware motion plan; this repo's RL policy
  learns residual corrections for sim-to-real transfer

## Experiment sweeps
Shell scripts in `experiments/` run grid searches over:
- Reward weights: w_task, w_track, w_regularization
- Network sizes: uniform hidden dims
- Action scales: 0.1, 0.2, 0.3
- Actuator types and kp/kd combinations

## Conventions
- Direct env pattern: all logic (obs, rewards, resets, actions) in env class methods
- `@configclass` decorator for all config dataclasses
- Robot asset path: relative `./robots/ur5e_sphere.usd`
- `replicate_physics` auto-computed from event modes (True if no prestartup/startup events)
- Checkpoints: `logs/rsl_rl/<experiment_name>/<timestamp>/model_<iter>.pt`
- ONNX/JIT exports: `<checkpoint_dir>/exported/policy.onnx` and `policy.pt`

## What NOT to touch
- Auto-generated USD cache files (typically in `/tmp/IsaacLab/`)
- Vendored/upstream RSL-RL or IsaacLab code
- `logs/` directory contents (gitignored)
- `wandb/` directory (gitignored)

## Known issues / gotchas
- **ACTION_SCALE mismatch:** Sim uses 0.25, real robot script hardcodes 0.2 — reconcile before deployment
- **Single-arm deployment:** `ur_rtde_real_time.py` currently only deploys left arm; right arm obs are zeroed
- **Trajectory dt hardcoded:** `self.dt = 0.02` in env — must match `physics_dt * decimation`
- **Episode length overridden:** `episode_length_s` is set dynamically from trajectory length in `_setup_scene()`
- **Reset to random phase:** Env resets to random point in trajectory, not always t=0
- **EE orientation TODO:** Comments in code note USD orientation may not match planner frame
- **PhysX articulation properties must be set via CPU API** (standard IsaacLab limitation)
