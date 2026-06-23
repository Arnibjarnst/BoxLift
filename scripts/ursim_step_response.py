"""Capture a single short step-response transient from URSim (or the real robot).

The idea: a deployed policy's first action is a small joint target near the current
(stationary) pose. We want IsaacSim's PD (kp/kd) to reproduce URSim's servoJ response
to exactly that kind of command. So:

  1. moveJ to a known initial pose and let it settle (qd ≈ 0).
  2. Sample a target ~ N(init, std), pushed at least `min_delta` per joint away from
     init so it's not trivially close.
  3. servoJ that fixed target and record actual_q / actual_qd for `steps` ticks at
     500 Hz (default 10 ticks = 20 ms — the initial response, kp-dominated from rest).
  4. Save an npz that scripts/match_step_response.py replays in IsaacSim to tune kp/kd.

Usage:
    python scripts/ursim_step_response.py --run_dir logs/rsl_rl/boxhinge/<run>
    python scripts/ursim_step_response.py --init_q 0,-1.57,0,0,0,0 --std 0.1
    python scripts/ursim_step_response.py --run_dir <run> --real_robot --gain 800 --lookahead 0.03

The resulting npz contains: init_q, target_q, actual_q (steps,6), actual_qd (steps,6),
gain, lookahead, dt, src_dt, steps, std, seed.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

UR5E_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--run_dir", type=str, default=None,
                    help="Training run dir; uses the trajectory's joints[0] as the initial "
                         "pose (the deployment-relevant start pose).")
parser.add_argument("--init_q", type=str, default=None,
                    help="Explicit 6 comma-separated initial joint angles (rad). Overrides --run_dir.")
parser.add_argument("--std", type=float, default=0.1,
                    help="Gaussian std (rad) for sampling the target around init.")
parser.add_argument("--min_delta", type=float, default=0.01,
                    help="Minimum per-joint |target-init| (rad). Components sampled smaller "
                         "than this are pushed out to ±min_delta (sign preserved) so the "
                         "step isn't trivially close to the start pose.")
parser.add_argument("--steps", type=int, default=10,
                    help="Number of 500Hz ticks to record (10 = 20ms, the initial transient).")
parser.add_argument("--gain", type=float, default=100, help="servoJ gain.")
parser.add_argument("--lookahead", type=float, default=0.05, help="servoJ lookahead_time (s).")
parser.add_argument("--seed", type=int, default=0, help="RNG seed for the target sample.")
parser.add_argument("--settle_s", type=float, default=1.0,
                    help="Seconds to wait after moveJ for the arm to come to rest.")
parser.add_argument("--out", type=str, default=None, help="Output npz path (auto if omitted).")
parser.add_argument("--real_robot", action="store_true",
                    help="Connect to the real robot IP instead of URSim.")
args = parser.parse_args()

# ---------- Resolve the initial joint configuration ----------

if args.init_q is not None:
    init_q = np.array([float(x) for x in args.init_q.split(",")], dtype=np.float64)
    if init_q.shape != (6,):
        raise ValueError(f"--init_q needs 6 values, got {init_q.shape}")
elif args.run_dir is not None:
    with open(os.path.join(args.run_dir, "params", "env.yaml"), "r") as f:
        env_cfg = yaml.unsafe_load(f)
    traj_path = env_cfg["trajectory_path"]
    if not Path(traj_path).is_absolute():
        # env.yaml stores a repo-root-relative path; resolve against the repo root.
        traj_path = Path(__file__).resolve().parent.parent / traj_path
    traj = np.load(traj_path)
    joints0 = traj["joints"] if "joints" in traj.files else traj["joints_l"]
    init_q = np.asarray(joints0[0], dtype=np.float64)
    print(f"[INFO] init_q from {traj_path} joints[0] = {init_q.round(4).tolist()}")
else:
    raise SystemExit("Provide --run_dir or --init_q for the initial pose.")

# ---------- Sample the target near init ----------

rng = np.random.default_rng(args.seed)
delta = rng.normal(0.0, args.std, size=6)
# Push any component that landed too close to init out to ±min_delta, preserving the
# sampled sign (an exact-zero sample is treated as positive).
sign = np.where(delta >= 0.0, 1.0, -1.0)
too_small = np.abs(delta) < args.min_delta
delta[too_small] = sign[too_small] * args.min_delta
target_q = init_q + delta
print(f"[INFO] target delta (rad) = {delta.round(4).tolist()}")

# ---------- Connect ----------

os_used = sys.platform
if args.real_robot:
    robot_ip = "192.168.1.100"
else:
    robot_ip = "172.29.144.1" if os_used == "win32" else "192.168.56.1"

rtde_frequency = 500
dt = 1.0 / rtde_frequency

print(f"[INFO] Connecting to {'real robot' if args.real_robot else 'URSim'} at {robot_ip}")
rtde_c = RTDEControl(robot_ip, rtde_frequency)
rtde_r = RTDEReceive(robot_ip, rtde_frequency)
rtde_c.reuploadScript()
rtde_c.setPayload(0.0, [0.0, 0.0, 0.0])

actual_q = np.zeros((args.steps, 6), dtype=np.float64)
actual_qd = np.zeros((args.steps, 6), dtype=np.float64)

try:
    # moveJ to the initial pose. Mirror the proven pattern from ur_rtde_fixed_traj.py /
    # ur_rtde_real_time.py: async moveJ (4th arg True) returns a bool for whether the
    # command was *accepted* (not completed), so retry until accepted, then poll
    # isSteady() until the arm has actually arrived AND stopped. The 3-arg blocking
    # moveJ does not wait for arrival in this ur_rtde setup.
    print(f"[INFO] moveJ to init pose {init_q.round(4).tolist()} ...")
    last_time = time.perf_counter()
    success = rtde_c.moveJ(init_q.tolist(), 0.5, 1.0, True)
    while not success:
        if time.perf_counter() - last_time > 5:
            raise TimeoutError("moveJ to init pose was not accepted within 5s")
        time.sleep(0.02)
        success = rtde_c.moveJ(init_q.tolist(), 0.5, 1.0, True)

    while not rtde_c.isSteady():
        time.sleep(dt)

    # Extra settle so getActualQd is genuinely ~0 before the step.
    time.sleep(args.settle_s)

    q_rest = np.array(rtde_r.getActualQ())
    qd_rest = np.array(rtde_r.getActualQd())
    arrival_err = np.linalg.norm(q_rest - init_q)
    if arrival_err > 1e-2:
        print(f"[WARN] arm did not reach init_q: |q_rest - init_q|={arrival_err:.4f} rad. "
              f"The move may have been rejected / clamped (joint limits, protective stop, "
              f"URSim not in remote control). q_rest={q_rest.round(4).tolist()}")
    if np.linalg.norm(qd_rest) > 1e-2:
        print(f"[WARN] arm not fully at rest: |qd|={np.linalg.norm(qd_rest):.4f} rad/s "
              f"(increase --settle_s if this is large)")
    print(f"[INFO] resting q = {q_rest.round(4).tolist()} (target was init_q)")

    # Capture loop: record state at the start of each tick (sample 0 = resting state),
    # then command the fixed target, then wait out the 500Hz period.
    print(f"[INFO] Capturing {args.steps} ticks ({args.steps * dt * 1e3:.0f} ms)...")
    for i in range(args.steps):
        t_start = rtde_c.initPeriod()
        actual_q[i] = rtde_r.getActualQ()
        actual_qd[i] = rtde_r.getActualQd()
        rtde_c.servoJ(target_q.tolist(), 0.5, 0.5, dt, args.lookahead, args.gain)
        rtde_c.waitPeriod(t_start)
finally:
    try:
        rtde_c.servoStop()
        rtde_c.stopScript()
    except Exception as e:
        print(f"[WARN] error stopping robot: {e}")

# ---------- Save ----------

if args.out is not None:
    out_path = Path(args.out)
else:
    log_dir = Path("logs/ur_rtde/step_response")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rb = "real" if args.real_robot else "ursim"
    out_path = log_dir / f"{tag}_{rb}_gain{args.gain}_la{args.lookahead}_seed{args.seed}.npz"

out_path.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    out_path,
    init_q=init_q,
    target_q=target_q,
    actual_q=actual_q,
    actual_qd=actual_qd,
    gain=np.float64(args.gain),
    lookahead=np.float64(args.lookahead),
    dt=np.float64(dt),
    src_dt=np.float64(dt),
    steps=np.int64(args.steps),
    std=np.float64(args.std),
    seed=np.int64(args.seed),
)
print(f"[INFO] Saved step response → {out_path}")
print(f"[INFO] Match in sim: python scripts/match_step_response.py {out_path} --kp 300 --kd 45 --plot out.png")
