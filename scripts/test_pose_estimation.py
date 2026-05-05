"""
Test the pose estimation pipeline end-to-end.

Spawns run_pose_estimation.py in a subprocess, subscribes to its ZMQ stream
via BoardPoseListener, and prints the latest box pose each time the user
presses Enter. Ctrl+C to quit.

Usage (from BoxLift root):
    python scripts/test_pose_estimation.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TAG_POSE_DIR = REPO_ROOT / "tag_pose_estimation"
POSE_ESTIMATION_SCRIPT = TAG_POSE_DIR / "scripts" / "run_pose_estimation.py"
# Relative to TAG_POSE_DIR (which is the cwd we set for the subprocess).
POSE_ESTIMATION_CONFIG = (
    "config/pose_estimation_configs/bigbox_pose_estimation_config.json"
)
POSE_ESTIMATION_PORT = 5555
BOX_BOARD_ID = "0"
FIRST_POSE_TIMEOUT_S = 30.0

# Make BoardPoseListener importable when tag_pose_estimation isn't pip-installed.
sys.path.insert(0, str(TAG_POSE_DIR))
from tag_pose_estimation.board_pose_listener import BoardPoseListener  # noqa: E402


def main():
    print(
        f"Launching pose estimation:\n"
        f"  cwd={TAG_POSE_DIR}\n"
        f"  {sys.executable} {POSE_ESTIMATION_SCRIPT} --config {POSE_ESTIMATION_CONFIG}"
    )
    # Ensure the tag_pose_estimation package is importable in the subprocess
    # regardless of whether it's pip-installed in this env.
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(TAG_POSE_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    pose_proc = subprocess.Popen(
        [
            sys.executable,
            str(POSE_ESTIMATION_SCRIPT),
            "--config",
            POSE_ESTIMATION_CONFIG,
        ],
        cwd=str(TAG_POSE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
    )

    listener = BoardPoseListener(
        box_pose_socket_address=f"tcp://localhost:{POSE_ESTIMATION_PORT}",
        update_rate=0.01,
    )
    if not listener.start():
        print("Failed to start BoardPoseListener.")
        pose_proc.terminate()
        return

    print(
        f"Listening for board id '{BOX_BOARD_ID}' on tcp://localhost:"
        f"{POSE_ESTIMATION_PORT}. Waiting for first pose..."
    )

    np.set_printoptions(suppress=True, precision=4)

    try:
        # Wait until at least one pose has arrived (or timeout / subprocess died).
        start = time.time()
        while listener.get_pose(BOX_BOARD_ID) is None:
            if pose_proc.poll() is not None:
                print(
                    f"Pose estimation subprocess exited early "
                    f"(returncode={pose_proc.returncode}). Aborting."
                )
                return
            if time.time() - start > FIRST_POSE_TIMEOUT_S:
                print(
                    f"Timed out after {FIRST_POSE_TIMEOUT_S:.0f}s waiting for "
                    f"the first pose. Is the board visible to the camera?"
                )
                return
            time.sleep(0.1)

        print(
            "First pose received. Press Enter to print the current pose, "
            "Ctrl+C to quit."
        )

        while True:
            try:
                input()
            except EOFError:
                break

            if pose_proc.poll() is not None:
                print(
                    f"Pose estimation subprocess exited "
                    f"(returncode={pose_proc.returncode})."
                )
                break

            pose = listener.get_pose(BOX_BOARD_ID)
            if pose is None:
                print("No pose available yet.")
                continue

            pos = pose[:3]
            quat_wxyz = pose[3:]
            confidence = listener.get_confidence(BOX_BOARD_ID)
            stable = listener.is_stable(BOX_BOARD_ID)
            tracked = listener.get_tracked_board_ids()

            print(f"  position [m]    : {pos}")
            print(f"  quat [w x y z]  : {quat_wxyz}")
            print(f"  confidence      : {confidence}")
            print(f"  stable          : {stable}")
            print(f"  tracked boards  : {tracked}")

    except KeyboardInterrupt:
        print("\nInterrupted, shutting down.")
    finally:
        listener.stop()
        if pose_proc.poll() is None:
            pose_proc.terminate()
            try:
                pose_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pose_proc.kill()


if __name__ == "__main__":
    main()
