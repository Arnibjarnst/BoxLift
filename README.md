# BoxLift

Residual reinforcement learning for contact-rich dual-arm manipulation in
[NVIDIA IsaacLab](https://github.com/isaac-sim/IsaacLab). A PPO policy learns
joint-space corrections on top of a pre-computed nominal trajectory, enabling a
pair of UR5e robots to reliably lift a box off a table in spite of contact
dynamics the nominal planner cannot model exactly.

## Overview

The core idea is **residual policy learning**: rather than learning a
manipulation policy from scratch, the RL agent outputs small corrections
(`action_scale * π(obs)`) that are added to a reference joint trajectory. 
```
q_target[t] = q_nominal[t] + action_scale * π_residual(obs[t])
```

## Repository structure

```
BoxLift/
├── scripts/
│   ├── rsl_rl/
│   │   ├── train.py              # main training entry point
│   │   ├── play.py               # evaluation; exports ONNX + TorchScript
│   │   ├── record.py             # record rollout data to JSON
│   │   └── cli_args.py           # shared argument parsing
│   ├── ur_rtde_real_time.py           # deploy ONNX policy on real UR5e (main deployment script)
│   ├── ur_rtde_fixed_traj.py          # replay fixed trajectory on real robot (sysid / baseline)
│   ├── ur_rtde_test.py                # harmonic oscillation test for motor delay measurement
│   ├── ur_rtde_estimate_pose_delay.py # estimate latency of the pose-estimation pipeline
│   ├── log_robot_pose.py              # log TCP pose + joints while jogging from teach pendant
│   ├── visualize_traj.py              # visualize a trajectory .npz in Isaac Sim
│   ├── follow_joint_targets.py        # simple trajectory playback in sim
│   ├── record_dataset.py              # run record.py once per .npz in a dataset folder
│   ├── rollout_summary.py             # per-env success analysis for multi-env rollouts
│   ├── rollout_plots.py               # plot rollout summaries (sim + real-robot batches)
│   ├── plot_rollout_rewards.py        # plot per-step rewards / errors from a rollout NPZ
│   ├── eval_rollout.py                # evaluate final-pose quality of a recorded rollout
│   ├── add_noise_to_traj.py           # add Gaussian noise to joint reference fields of a .npz
│   ├── ursim_step_response.py         # capture step-response transient from URSim / real robot
│   ├── match_step_response.py         # check whether IsaacSim PD reproduces URSim step response
│   ├── match_ursim_dynamics.py        # one-step prediction error: IsaacSim vs URSim/real
│   ├── sweep_kp_kd_per_joint.py       # per-joint kp/kd sweep against a URSim rollout (single IsaacSim session)
│   ├── sysid_actuator_sweep.py        # full actuator / gain system-identification sweep
│   └── zero_agent.py / random_agent.py # baseline agents for environment testing
│
├── source/BoxLift/BoxLift/tasks/direct/
│   ├── boxlift/                  # ★ main task: dual-arm box lifting
│   ├── boxhinge/                 # single-arm box hinge manipulation
│   ├── boxpush/                  # single-arm box pushing
│   ├── boxtracker/               # single-arm task with multi-trajectory pool + continuous phase
│   ├── boxmagic/                 # box lifting with modified collision properties
│   └── jointtarget/              # simple joint-target tracking (diagnostics / sysid)
│
├── reference_trajectories/       # pre-computed nominal trajectories (.npz)
│   ├── box_lift_ur5e/
│   ├── box_hinge_ur5e/
│   ├── box_push_ur5e/
│   └── box_rotate_ur5e/
│
├── experiments/                  # hyperparameter sweep shell scripts
├── notebooks/                    # analysis notebooks
├── stubs/                        # type stubs for ur_rtde (rtde_control, rtde_receive, …)
├── robots/                       # USD robot assets (ur5e.usd)
└── tag_pose_estimation/          # AprilTag-based cube pose estimation (separate package)
```

## Environments

All environments follow the IsaacLab `DirectRLEnv` pattern. The main task is
**`Template-Boxlift-Direct-v0`**; the others are either earlier single-arm
variants or experimental environments.

| Task ID | Description |
|---|---|
| `Template-Boxlift-Direct-v0` | **Primary.** Dual-arm UR5e box lifting with VOC curriculum |
| `Template-Boxhinge-Direct-v0` | Single-arm hinge manipulation |
| `Template-Boxpush-Direct-v0` | Single-arm box pushing |
| `Template-Boxtracker-Direct-v0` | Single-arm; multi-traj pool, continuous phase, adaptive playback speed |
| `Template-Boxmagic-Direct-v0` | Dual-arm lift with modified collision properties (experimental) |
| `Follow-Joint-Targets` | Joint-target tracking (diagnostics) |

### boxlift (primary)

The dual-arm environment. Key design points:

- **Action space (12D):** Joint position residuals, 6 per arm. Added to the
  nominal trajectory target after scaling by `action_scale` and clamped to
  joint limits.
- **Action modes:** Several formulations are supported (`A`–`D`, `BC`), ranging
  from residual-on-planner-target to residual-on-current-position, with optional
  curriculum blending between modes.
- **Observations (actor):** Relative and absolute joint positions/velocities,
  box pose error vs reference (delayed + sub-rate tracker model), contact bools,
  trajectory phase, future box waypoints, previous action. A history of the last
  `obs_history_steps` steps is stacked.
- **Privileged observations (critic):** Clean box state, DR sample values
  (friction, mass, actuator gains), full reference state at current phase, EE
  contact forces, EE-in-box-frame errors, VOC/curriculum context.
- **Rewards:** Task (box position + orientation vs reference), tracking
  (EE position, EE-in-box-frame during contact), regularization (joint
  acceleration, torque, action rate/norm, joint limits, illegal contact,
  flange-forearm proximity).
- **VOC curriculum:** A virtual PD wrench drives the box along its reference
  trajectory while the policy learns. The gain decays segment-by-segment as
  reward thresholds are met, handing full control to the policy progressively.
- **Domain randomisation:** EE and object friction, actuator gains (±50%),
  object mass, gravity noise, robot base pose (XYZ + yaw, fixed per env for the
  full training run), joint observation noise.
- **Reset:** Random State Initialisation (RSI) — resets to a random phase of
  the trajectory. Configurable bias toward phase 0 and toward segments still
  under VOC assist.

## Installation

Install [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
first, then:

```bash
# Clone this repo outside the IsaacLab directory
git clone <repo-url> BoxLift
cd BoxLift

# Install in editable mode using the IsaacLab Python environment
python -m pip install -e source/BoxLift
```

Verify the install:

```bash
python scripts/list_envs.py          # should list all registered tasks
```

## Usage

### Recording rollouts

```bash
python scripts/rsl_rl/record.py \
    --task Template-Boxlift-Direct-v0 \
    --trajectory_path <path> \
    --num_envs 16 \
    --checkpoint <path-to-model.pt>
```

### Deployment on real UR5e

```bash
python scripts/ur_rtde_real_time.py \
    --onnx_model_path logs/rsl_rl/boxlift/<run>/exported/policy.onnx \
    --reference_trajectory_path reference_trajectories/box_lift_ur5e/<traj>.npz \
    --real_robot
```

The deployment script runs two threads: a policy thread (50 Hz) and a control
thread (500 Hz) that linearly interpolates between policy targets and sends them
via `servoJ()`. It aborts immediately on any safety or robot-mode fault.

Robot IP defaults: `192.168.1.100` (real), `192.168.56.1` (Linux URSim),
`172.29.144.1` (Windows URSim).

## Trajectory file format

Trajectories are `.npz` files with the following arrays:

| Key | Shape | Description |
|---|---|---|
| `joints_l` / `joints_r` | `(T, 6)` | Joint positions per timestep |
| `joints_target_l` / `joints_target_r` | `(T, 6)` | Planner joint targets (residual is added to these) |
| `joint_vel_l` / `joint_vel_r` | `(T, 6)` | Joint velocities |
| `EE_poses_l` / `EE_poses_r` | `(T, 7)` | End-effector poses `[pos_xyz, quat_xyzw]` |
| `obj_poses` | `(T, 7)` | Object poses `[pos_xyz, quat_xyzw]` |
| `obj_vel` | `(T, 6)` | Object linear + angular velocity |
| `arm_l_pose` / `arm_r_pose` | `(7,)` | Robot base poses `[pos_xyz, quat_xyzw]` |
| `dt` | scalar | Timestep (must equal `physics_dt * decimation = 0.02 s`) |

## Related

`tag_pose_estimation/` is a separate package in this repository for
AprilTag-based cube pose estimation used during real-robot experiments. See its
own [README](tag_pose_estimation/README.md).
