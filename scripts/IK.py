import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("trajectory_path", type=str)
parser.add_argument("--hz", type=int, default=50)
args = parser.parse_args()


from isaacsim import SimulationApp

# instantiate the SimulationApp helper class
simulation_app = SimulationApp({"headless": True})

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import torch

from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.core.api.objects import GroundPlane, DynamicCuboid

from isaaclab.utils.math import quat_slerp


np.set_printoptions(precision=6, suppress=True)


data = np.load(args.trajectory_path)

q_trj = data["q_trj"]
u_trj = data["u_trj"]
q_u_indices_into_x = data["q_u_indices_into_x"]
q_a_indices_into_x = data["q_a_indices_into_x"]


planner_hz = 1 / data["h"]
planner_hz_int = int(round(planner_hz))

# Only support perfect upsampling
assert np.isclose(planner_hz_int, planner_hz)
assert args.hz % planner_hz_int == 0

# linearly upsample q_trj, u_trj
def upsample_linear(x, k):
    N = x.shape[0]

    t_old = np.arange(N)
    t_new = np.linspace(0, N - 1, k * (N - 1) + 1)

    f = interp1d(t_old, x, axis=0, kind="linear")
    return f(t_new)

def upsample_slerp(wxyzs, k):
    out = []
    quats = [torch.Tensor(wxyz) for wxyz in wxyzs]
    
    for i in range(len(quats) - 1):
        for dt in np.linspace(0.0, 1.0, k, endpoint=False):
            quat_interp = quat_slerp(quats[i], quats[i+1], dt)
            out.append(quat_interp.numpy())

    out.append(wxyzs[-1])

    return np.array(out)


upsampling_factor = int(round(args.hz / planner_hz_int))

u_trj = upsample_linear(u_trj, upsampling_factor)

obj_poses = q_trj[:, q_u_indices_into_x]

# We cannot linearly upsample quaternions so upsample seperately
obj_quats = upsample_slerp(obj_poses[:, :4], upsampling_factor)
obj_pos = upsample_linear(obj_poses[:, 4:], upsampling_factor)
obj_poses = np.hstack((obj_quats, obj_pos))

EE_trj = upsample_linear(q_trj[:, q_a_indices_into_x], upsampling_factor)

EE_l = EE_trj[:, :3]
EE_r = EE_trj[:, 3:]
EE_target_l = u_trj[:, :3]
EE_target_r = u_trj[:, 3:]

# Only difference is robot pose
arm_l_pose = np.array([0.0, 0.0, 0.0, 1.0, -0.6, -0.2, 0.0])
arm_r_pose = np.array([1.0, 0.0, 0.0, 0.0,  0.6, -0.2, 0.0])

# from isaacsim.robot_motion.motion_generation.interface_config_loader import load_supported_lula_kinematics_solver_config, get_supported_robots_with_lula_kinematics
# ORIGINAL FOUND WITH: load_supported_lula_kinematics_solver_config("UR5")
robot_description_path = "./robots/ur5.yaml"
urdf_path = "./robots/ur5.urdf"
kinematics_solver = LulaKinematicsSolver(
    robot_description_path=robot_description_path,
    urdf_path=urdf_path
)

# TODO: Change solver so this is supported
# # Add collisions
# ground_plane = GroundPlane("/World/ground")
# succ = kinematics_solver.add_ground_plane(ground_plane)

# cube_prim = DynamicCuboid("/World/cube")
# kinematics_solver.add_cuboid(cube_prim, static=False)

def get_joints(target_pos, target_quat, robot_pose, warm_start = None):
    kinematics_solver.set_robot_base_pose(robot_position=robot_pose[4:], robot_orientation=robot_pose[:4])
    
    # if obj_pose is not None:
    #     cube_prim.set_size(1.0)
    #     cube_prim.set_local_pose(translation=obj_pose[:3], orientation=obj_pose[3:])
    # else:
    #     cube_prim.set_size(0.0)

    kinematics_solver.update_world()

    joint_pos, success = kinematics_solver.compute_inverse_kinematics(
        frame_name="sphere",
        target_position=target_pos,
        target_orientation=target_quat,
        warm_start=warm_start,
    )

    print(target_pos, target_quat)
    print(kinematics_solver.compute_forward_kinematics("sphere", joint_pos))
    print(kinematics_solver.compute_forward_kinematics("wrist_3_link", joint_pos))


    if success:
        return joint_pos
    else:
        raise RuntimeError("IK did not converge to a solution.")

init_joints_l = np.array([
    0.0,
    -0.66 * np.pi,
    -0.5 * np.pi,
    - np.pi,
    0.0,
    0.0
])
init_joints_r = np.array([
    0.0,
    -0.33 * np.pi,
    0.5 * np.pi,
    0.0,
    0.0,
    0.0
])

N = len(EE_trj)
joints_l = np.zeros((N, 6))
joints_r = np.zeros((N, 6))
joints_target_l = np.zeros((N, 6))
joints_target_r = np.zeros((N, 6))

initial_quat_l_xyzw = R.align_vectors(obj_pos[0] - EE_l[0], [0, 0, 1])[0].as_quat()
initial_quat_r_xyzw = R.align_vectors(obj_pos[0] - EE_r[0], [0, 0, 1])[0].as_quat()
initial_quat_l = np.array([initial_quat_l_xyzw[3], initial_quat_l_xyzw[0], initial_quat_l_xyzw[1], initial_quat_l_xyzw[2]])
initial_quat_r = np.array([initial_quat_r_xyzw[3], initial_quat_r_xyzw[0], initial_quat_r_xyzw[1], initial_quat_r_xyzw[2]])

print(initial_quat_l)
print(initial_quat_r)
print(EE_target_l[0], EE_l[0])
print(EE_target_r[0], EE_r[0])


for i in range(N):
    last_joints_l = init_joints_l if i == 0 else joints_l[i-1]
    last_joints_r = init_joints_r if i == 0 else joints_r[i-1]

    desired_quat_l = initial_quat_l if i == 0 else None
    desired_quat_r = initial_quat_r if i == 0 else None

    joints_l[i] = get_joints(EE_l[i], desired_quat_l, arm_l_pose, last_joints_l)
    joints_r[i] = get_joints(EE_r[i], desired_quat_r, arm_r_pose, last_joints_r)
    joints_target_l[i] = get_joints(EE_target_l[i], None, arm_l_pose, joints_l[i])
    joints_target_r[i] = get_joints(EE_target_r[i], None, arm_r_pose, joints_r[i])

        


date_str = os.path.basename(args.trajectory_path)[5:-4]

np.savez_compressed(
    f"data/box_lift/IK_{date_str}.npz",
    obj_poses       = obj_poses,
    EE_l            = EE_l,
    EE_r            = EE_r,
    EE_target_l     = EE_target_l,
    EE_target_r     = EE_target_r,
    joints_l        = joints_l,
    joints_target_l = joints_target_l,
    joints_r        = joints_r,
    joints_target_r = joints_target_r,
    arm_l_pose      = arm_l_pose,
    arm_r_pose      = arm_r_pose,
    dt              = 1.0 / args.hz
)
