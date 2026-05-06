"""Interactively estimate the latency of the pose-estimation pipeline by repeatedly
bumping the EE into the box and measuring the delay between contact (TCP force spike)
and the first visible change in the reported box pose.

Workflow per trial (interactive):
  1. Prompt for axis ("x" / "-x" / "y" / "-y") or "q" to stop.
  2. Read current box pose, compute approach pose offset from the box along that axis.
  3. moveL to approach.
  4. Prompt: press Enter to push (or "a" to abort this trial without pushing).
  5. moveL (asynchronous) into the box at low speed; sample TCP force + box pose.
  6. stopL on contact + pose-change detected (or timeout); retreat to approach.
  7. Print delay for this trial and running mean over all valid trials.

On exit, prints summary stats and shows a histogram of delays.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Make tag_pose_estimation importable regardless of pip-install state.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TAG_POSE_DIR = _REPO_ROOT / "tag_pose_estimation"
sys.path.insert(0, str(_TAG_POSE_DIR))
from tag_pose_estimation.board_pose_listener import BoardPoseListener  # noqa: E402

POSE_ESTIMATION_SCRIPT = _TAG_POSE_DIR / "scripts" / "run_pose_estimation.py"
POSE_ESTIMATION_CONFIG = (
    "config/pose_estimation_configs/bigbox_pose_estimation_config.json"
)
POSE_ESTIMATION_PORT = 5555
BOX_BOARD_ID = "0"
POSE_FIRST_POSE_TIMEOUT_S = 30.0

# Same world→robot offset used by ur_rtde_real_time.py.
WORLD_TO_ROBOT_TRANSLATION = np.array([0.02, 0.508, 0.018], dtype=np.float32)

AXIS_MAP = {"x": (1, 0), "-x": (-1, 0), "y": (0, 1), "-y": (0, -1)}


def start_pose_estimation():
    """Start the pose-estimation subprocess + listener (mirrors ur_rtde_real_time.py)."""
    pose_env = os.environ.copy()
    pose_env["PYTHONPATH"] = (
        str(_TAG_POSE_DIR) + os.pathsep + pose_env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    pose_proc = subprocess.Popen(
        [sys.executable, str(POSE_ESTIMATION_SCRIPT), "--config", POSE_ESTIMATION_CONFIG],
        cwd=str(_TAG_POSE_DIR),
        env=pose_env,
        stdout=subprocess.DEVNULL,
    )
    listener = BoardPoseListener(
        box_pose_socket_address=f"tcp://localhost:{POSE_ESTIMATION_PORT}",
        update_rate=0.01,
    )
    if not listener.start():
        pose_proc.terminate()
        raise RuntimeError("Failed to start BoardPoseListener")
    return pose_proc, listener


def get_box_pos(listener):
    p = listener.get_pose(BOX_BOARD_ID)
    if p is None:
        return None
    return np.asarray(p[:3], dtype=np.float64) + WORLD_TO_ROBOT_TRANSLATION


def prompt_axis():
    """Ask the user for an axis ('x', '-x', 'y', '-y') or quit. Returns the axis
    string, or None to stop trials."""
    while True:
        try:
            s = input("\n[next trial] axis to push (x / -x / y / -y), 'q' to stop: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if s in ("q", "esc", "quit", "exit", ""):
            return None
        if s in AXIS_MAP:
            return s
        print(f"  invalid input '{s}'. Use x, -x, y, -y, or q.")


def confirm_push():
    """Ask the user to press Enter (push) or 'a' (abort this trial). Returns True
    to proceed with the push, False to skip back to axis prompt."""
    try:
        s = input("  press Enter to push, or 'a'+Enter to abort: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return s != "a"


def push_and_measure(rtde_c, rtde_r, listener, args, ax_dir, approach_pose, push_pose):
    """Run the async push, sample force + box-pose, return (delay_s_or_None, samples).
    Caller is responsible for retreating to approach afterwards."""
    initial_box = get_box_pos(listener)
    if initial_box is None:
        print("  ! lost box pose at approach")
        return None, None
    baseline_force = np.asarray(rtde_r.getActualTCPForce())[:3]

    rtde_c.moveL(push_pose, args.push_speed, 0.5, asynchronous=True)

    contact_t = None
    pose_change_t = None
    sample_t, sample_f, sample_dpos = [], [], []

    t_start = time.perf_counter()
    while True:
        t = time.perf_counter() - t_start
        if t > args.max_push_time:
            break

        f_now = np.asarray(rtde_r.getActualTCPForce())[:3] - baseline_force
        f_norm = float(np.linalg.norm(f_now))
        bp = get_box_pos(listener)
        dpos_norm = float(np.linalg.norm(bp - initial_box)) if bp is not None else float("nan")

        sample_t.append(t)
        sample_f.append(f_norm)
        sample_dpos.append(dpos_norm)

        if contact_t is None and f_norm > args.force_threshold:
            contact_t = t
            print(f"  contact at t={t*1000:.1f} ms (||F||={f_norm:.1f} N)")
        if (
            pose_change_t is None
            and not np.isnan(dpos_norm)
            and dpos_norm > args.pose_threshold
        ):
            pose_change_t = t
            print(f"  pose Δ at t={t*1000:.1f} ms (||Δp||={dpos_norm*1000:.1f} mm)")

        if contact_t is not None and pose_change_t is not None and t > pose_change_t + 0.10:
            break
        time.sleep(0.001)

    rtde_c.stopL(0.5)
    time.sleep(0.2)

    samples = {
        "t": np.asarray(sample_t),
        "f_norm": np.asarray(sample_f),
        "dpos_norm": np.asarray(sample_dpos),
        "contact_t": contact_t,
        "pose_change_t": pose_change_t,
    }
    if contact_t is None:
        print("  ! no contact detected")
        return None, samples
    if pose_change_t is None:
        print("  ! contact but no pose change before timeout")
        return float("nan"), samples
    return pose_change_t - contact_t, samples


def compute_targets(box_pos, ax_dir, args, rtde_r):
    """Compute approach + push poses given the current box pose and chosen axis."""
    cur_tcp = np.asarray(rtde_r.getActualTCPPose())
    orientation = cur_tcp[3:]

    approach_xyz = box_pos.copy()
    approach_xyz[0] -= ax_dir[0] * args.approach_offset
    approach_xyz[1] -= ax_dir[1] * args.approach_offset
    if args.fixed_z is not None:
        approach_xyz[2] = args.fixed_z

    push_xyz = approach_xyz.copy()
    push_xyz[0] += ax_dir[0] * (args.approach_offset + args.push_distance)
    push_xyz[1] += ax_dir[1] * (args.approach_offset + args.push_distance)

    return list(approach_xyz) + list(orientation), list(push_xyz) + list(orientation), approach_xyz


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real_robot", action="store_true",
                        help="Connect to the real UR5e (192.168.1.100) instead of sim.")
    parser.add_argument("--approach_offset", type=float, default=0.20,
                        help="EE distance from box CENTER along push_axis at the approach pose (m). "
                             "Should be larger than the box's half-extent along that axis + ~5cm. "
                             "Default 0.20 m is safe for a 0.34m box.")
    parser.add_argument("--push_distance", type=float, default=0.05,
                        help="How far past approach to drive the EE if no contact (m).")
    parser.add_argument("--push_speed", type=float, default=0.02,
                        help="moveL linear speed for the push (m/s).")
    parser.add_argument("--force_threshold", type=float, default=10.0,
                        help="||F||₃ above baseline that counts as contact (N).")
    parser.add_argument("--pose_threshold", type=float, default=0.002,
                        help="Box position delta from initial that counts as pose change (m).")
    parser.add_argument("--max_push_time", type=float, default=2.0,
                        help="Push timeout per trial (s).")
    parser.add_argument("--fixed_z", type=float, default=None,
                        help="Override approach Z (m, robot frame). Default: use box-pose Z.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="If set, save histogram + raw samples here on exit.")
    args = parser.parse_args()

    pose_proc, listener = start_pose_estimation()

    rtde_c = rtde_r = None
    delays = []          # delay per completed trial (NaN if contact-but-no-pose-change)
    all_samples = []     # full sample dict per attempted trial (incl. failed contact)
    trial_axes = []      # which axis was used per trial

    try:
        # Wait for first box pose.
        print("Waiting for first box pose...")
        t0 = time.time()
        while listener.get_pose(BOX_BOARD_ID) is None:
            if time.time() - t0 > POSE_FIRST_POSE_TIMEOUT_S:
                raise RuntimeError("Timed out waiting for first box pose")
            time.sleep(0.1)
        print("OK.")

        robot_ip = "192.168.1.100" if args.real_robot else "192.168.56.1"
        print(f"Connecting to robot at {robot_ip}")
        rtde_r = RTDEReceive(robot_ip, 500)
        rtde_c = RTDEControl(robot_ip, 500, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT)
        rtde_c.reuploadScript()

        trial = 0
        while True:
            axis = prompt_axis()
            if axis is None:
                break
            trial += 1
            ax_dir = AXIS_MAP[axis]
            print(f"--- Trial {trial} (axis={axis}) ---")

            box_pos = get_box_pos(listener)
            if box_pos is None:
                print("  ! no box pose available, skipping")
                continue
            print(f"  box at {box_pos}")

            approach_pose, push_pose, approach_xyz = compute_targets(box_pos, ax_dir, args, rtde_r)
            print(f"  approach: {approach_xyz}")

            # Move to approach.
            try:
                rtde_c.moveL(approach_pose, 0.25, 0.5, asynchronous=False)
            except Exception as e:
                print(f"  ! moveL to approach failed: {e}")
                continue
            time.sleep(0.5)

            # Confirm push.
            if not confirm_push():
                print("  trial aborted before push")
                continue

            # Push + measure.
            d, samples = push_and_measure(rtde_c, rtde_r, listener, args, ax_dir,
                                          approach_pose, push_pose)
            # Retreat back to approach for safety regardless of outcome.
            try:
                rtde_c.moveL(approach_pose, 0.25, 0.5, asynchronous=False)
            except Exception as e:
                print(f"  ! retreat moveL failed: {e}")

            if samples is not None:
                all_samples.append({**samples, "axis": axis, "trial": trial})
                trial_axes.append(axis)

            if d is None:
                continue
            delays.append(d)

            # Per-trial + running stats.
            valid = np.asarray([x for x in delays if not np.isnan(x)])
            d_str = "NaN (no pose change)" if np.isnan(d) else f"{d*1000:.1f} ms"
            mean_str = "—" if len(valid) == 0 else f"{valid.mean()*1000:.1f} ms (n={len(valid)})"
            print(f"  this trial: {d_str}    running mean: {mean_str}")

    finally:
        try:
            if rtde_c is not None:
                rtde_c.stopL(2.0)
                rtde_c.disconnect()
            if rtde_r is not None:
                rtde_r.disconnect()
        except Exception:
            pass
        try:
            listener.stop()
        except Exception:
            pass
        if pose_proc is not None:
            pose_proc.terminate()

    # Final summary + histogram.
    valid = np.asarray([x for x in delays if not np.isnan(x)])
    print(f"\n=== Summary over {len(valid)} valid trials (out of {len(delays)} attempts) ===")
    if len(valid) == 0:
        print("No valid trials — nothing to plot.")
        return
    print(f"  mean   = {valid.mean()*1000:.2f} ms")
    print(f"  median = {np.median(valid)*1000:.2f} ms")
    print(f"  std    = {valid.std()*1000:.2f} ms")
    print(f"  min    = {valid.min()*1000:.2f} ms")
    print(f"  max    = {valid.max()*1000:.2f} ms")

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = max(5, min(20, len(valid)))
    ax.hist(valid * 1000.0, bins=bins, edgecolor="k", alpha=0.85)
    ax.axvline(valid.mean() * 1000.0, color="r", ls="--",
               label=f"mean = {valid.mean()*1000:.1f} ms")
    ax.axvline(np.median(valid) * 1000.0, color="orange", ls="--",
               label=f"median = {np.median(valid)*1000:.1f} ms")
    ax.set_xlabel("pose-estimation delay (ms)")
    ax.set_ylabel("count")
    ax.set_title(f"Box pose-estimation latency over {len(valid)} bumps")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fig_path = out_dir / f"pose_delay_histogram_{stamp}.png"
        data_path = out_dir / f"pose_delay_raw_{stamp}.npz"
        fig.savefig(fig_path, dpi=120)
        np.savez(
            data_path,
            delays=np.asarray(delays, dtype=np.float64),
            axes=np.asarray(trial_axes),
            approach_offset=args.approach_offset,
            push_distance=args.push_distance,
            push_speed=args.push_speed,
            force_threshold=args.force_threshold,
            pose_threshold=args.pose_threshold,
            samples=np.array(all_samples, dtype=object),
        )
        print(f"Saved figure -> {fig_path}")
        print(f"Saved raw    -> {data_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
