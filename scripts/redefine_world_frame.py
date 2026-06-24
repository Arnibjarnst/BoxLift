"""
Redefine world frame to the midpoint of two UR5e robot bases.

The extrinsic/pose-estimation pipeline uses an intermediate ArUco board frame:
  camera frame -> (camera extrinsic) -> ArUco world frame
               -> (pose_estimation_frame_transform inverted) -> output frame W'

This script takes the two calibrated robot base poses in the ArUco world frame
and defines a new output world frame W' centred at the midpoint of the two bases,
with the orientation of the RIGHT robot base.

Outputs (saved next to --left in the extrinsic_calibration dir):
  world_midpoint_<ts>.npy          — pose_estimation_frame_transform (T_ArUco_W')
  ur5e_left_in_midpoint_<ts>.npy   — left  robot pose in W'
  ur5e_right_in_midpoint_<ts>.npy  — right robot pose in W'

Usage:
  python scripts/redefine_world_frame.py \\
    --left  tag_pose_estimation/config/extrinsic_calibration/ur5e_left_20260527_160024.npy \\
    --right tag_pose_estimation/config/extrinsic_calibration/ur5e_right_20260527_160806.npy \\
    [--update_config tag_pose_estimation/config/pose_estimation_configs/bigbox_pose_estimation_config.json]
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path


def invert_hom(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--left",  required=True,
                    help="ur5e_left*.npy — 4x4 left robot base pose in ArUco world frame")
parser.add_argument("--right", required=True,
                    help="ur5e_right*.npy — 4x4 right robot base pose in ArUco world frame")
parser.add_argument("--update_config", default=None,
                    help="If given, writes the new frame-transform path into this JSON config")
args = parser.parse_args()

T_ArUco_L = np.load(args.left)
T_ArUco_R = np.load(args.right)

p_L = T_ArUco_L[:3, 3]
p_R = T_ArUco_R[:3, 3]
p_mid = (p_L + p_R) / 2

print("=" * 60)
print(f"Left  robot base (ArUco frame): {p_L}")
print(f"Right robot base (ArUco frame): {p_R}")
print(f"Midpoint W' origin (ArUco):     {p_mid}")
print(f"Left-to-right distance:         {np.linalg.norm(p_R - p_L):.4f} m")
print("=" * 60)

# W' frame: origin at midpoint, orientation = right robot's orientation.
# T_ArUco_W' is the pose of W' expressed in the ArUco world frame.
T_ArUco_W = np.eye(4)
T_ArUco_W[:3, :3] = T_ArUco_R[:3, :3]  # right robot orientation
T_ArUco_W[:3, 3]  = p_mid

# T_W'_ArUco: transforms a point in ArUco coords into W' coords.
# The pose estimator loads T_ArUco_W', inverts it to get T_W'_ArUco,
# and applies it to detected object poses.
T_W_ArUco = invert_hom(T_ArUco_W)

# Robot base poses expressed in W'
T_W_L = T_W_ArUco @ T_ArUco_L
T_W_R = T_W_ArUco @ T_ArUco_R

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(args.left).parent

frame_transform_path = out_dir / f"world_midpoint_{ts}.npy"
left_out_path  = out_dir / f"ur5e_left_in_midpoint_{ts}.npy"
right_out_path = out_dir / f"ur5e_right_in_midpoint_{ts}.npy"

np.save(frame_transform_path, T_ArUco_W)
np.save(left_out_path,  T_W_L)
np.save(right_out_path, T_W_R)

def write_pose_txt(path: Path, robot: str, T: np.ndarray, source_left: str, source_right: str) -> None:
    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    with open(path, "w") as f:
        f.write(f"# Robot base pose for robot {robot} in midpoint world frame W'\n")
        f.write(f"# Generated on {datetime.now().isoformat()}\n")
        f.write(f"# W' origin: midpoint of left and right robot bases\n")
        f.write(f"# W' orientation: right robot base orientation\n")
        f.write(f"# Source left:  {source_left}\n")
        f.write(f"# Source right: {source_right}\n")
        f.write(f"\n# Translation [x, y, z] (m)\n")
        f.write(np.array2string(T[:3, 3], precision=8, suppress_small=True))
        f.write(f"\n\n# Rotation as quaternion [qw, qx, qy, qz]\n")
        f.write(np.array2string(quat_wxyz, precision=8, suppress_small=True))
        f.write(f"\n\n# 4x4 homogeneous transformation matrix\n")
        f.write(np.array2string(T, precision=8, suppress_small=True))
        f.write("\n")

write_pose_txt(left_out_path.with_suffix(".txt"),  "ur5e_left",  T_W_L, args.left, args.right)
write_pose_txt(right_out_path.with_suffix(".txt"), "ur5e_right", T_W_R, args.left, args.right)

print("\nSaved files:")
print(f"  pose_estimation_frame_transform : {frame_transform_path}")
print(f"  Left  robot in W'               : {left_out_path}  (+.txt)")
print(f"  Right robot in W'               : {right_out_path}  (+.txt)")

print("\n--- Left robot in W' (4x4) ---")
print(T_W_L)
print("\n--- Right robot in W' (4x4) ---  (should be near-identity rotation, offset translation)")
print(T_W_R)

if args.update_config:
    config_path = Path(args.update_config)
    with open(config_path) as f:
        cfg = json.load(f)
    # The pose estimator subprocess runs with cwd = tag_pose_estimation root
    # (config is at <root>/config/pose_estimation_configs/<file> — 3 levels up).
    tag_root = config_path.parent.parent.parent.resolve()
    try:
        rel = frame_transform_path.resolve().relative_to(tag_root)
        cfg["pose_estimation_frame_transform"] = str(rel).replace("\\", "/")
    except ValueError:
        cfg["pose_estimation_frame_transform"] = str(frame_transform_path.resolve()).replace("\\", "/")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nUpdated config: {config_path}")
    print(f"  pose_estimation_frame_transform -> {cfg['pose_estimation_frame_transform']}")
