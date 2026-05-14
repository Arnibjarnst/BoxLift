"""Log real-robot TCP pose and joint positions while you move the robot from the
teach pendant. Saves time-series to .npz in the format expected by
visualize_traj.py (500 Hz source rate, fields actual_q + expected_q).

Usage:
  python scripts/log_robot_pose.py --real_robot --out logs/manual_pose.npz

Press Ctrl+C in the terminal to stop logging and save.

NPZ contents (compatible with scripts/visualize_traj.py):
  actual_q    (N, 6)  joint positions used by visualize_traj's primary arm
  expected_q  (N, 6)  reference arm — duplicated from actual_q (no target here)
  t           (N,)    seconds since logging started
  qd          (N, 6)  joint velocities (rad/s)
  tcp_pose    (N, 6)  TCP in UR native base frame [x,y,z,rx,ry,rz]
  tcp_speed   (N, 6)  TCP velocity [vx,vy,vz,wx,wy,wz]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from rtde_receive import RTDEReceiveInterface as RTDEReceive

PRINT_HZ = 10.0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--real_robot", action="store_true",
                        help="Connect to 192.168.1.100; otherwise URSim virtual IP.")
    parser.add_argument("--rate", type=float, default=500.0,
                        help="Sample rate (Hz). Default 500 to match visualize_traj.py.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output .npz path.")
    args = parser.parse_args()

    if sys.platform == "win32":
        sim_ip = "172.29.144.1"
    else:
        sim_ip = "192.168.56.1"
    robot_ip = "192.168.1.100" if args.real_robot else sim_ip

    print(f"Connecting to {robot_ip} ...")
    rtde_r = RTDEReceive(robot_ip)
    print("Logging passively. Move the robot via the teach pendant.")
    print(f"Logging at {args.rate:.0f} Hz, printing at {PRINT_HZ:.0f} Hz.")
    print("Press Ctrl+C to stop and save.\n")

    ts, qs, qds, tcps, tcp_speeds = [], [], [], [], []
    period = 1.0 / args.rate
    print_period = 1.0 / PRINT_HZ
    t0 = time.perf_counter()
    next_t = t0
    next_print_t = t0

    np.set_printoptions(precision=4, suppress=True, sign=" ")

    try:
        while True:
            now = time.perf_counter()
            t = now - t0
            q = rtde_r.getActualQ()
            qd = rtde_r.getActualQd()
            tcp = rtde_r.getActualTCPPose()
            tcp_v = rtde_r.getActualTCPSpeed()

            ts.append(t)
            qs.append(q)
            qds.append(qd)
            tcps.append(tcp)
            tcp_speeds.append(tcp_v)

            if now >= next_print_t:
                print(f"[t={t:6.2f}s] tcp={np.asarray(tcp)}  q={np.asarray(q)}")
                next_print_t += print_period
                if next_print_t < now:
                    next_print_t = now + print_period

            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try:
            rtde_r.disconnect()
        except Exception:
            pass

    if not ts:
        print("No samples collected — nothing to save.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    actual_q = np.asarray(qs, dtype=np.float64)
    np.savez(
        out,
        actual_q=actual_q,
        expected_q=actual_q.copy(),
        t=np.asarray(ts, dtype=np.float64),
        qd=np.asarray(qds, dtype=np.float64),
        tcp_pose=np.asarray(tcps, dtype=np.float64),
        tcp_speed=np.asarray(tcp_speeds, dtype=np.float64),
    )
    duration = ts[-1]
    print(f"Saved {len(ts)} samples ({duration:.1f} s, "
          f"{len(ts)/max(duration, 1e-9):.1f} Hz effective) -> {out}")


if __name__ == "__main__":
    main()
