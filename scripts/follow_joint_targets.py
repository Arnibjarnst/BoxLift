import numpy as np
import time
import argparse

np.set_printoptions(precision=5, suppress=True)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation, Articulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.controllers import ArticulationController

import carb.input


parser = argparse.ArgumentParser()
parser.add_argument("IK_file_path", type=str)
parser.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

data = np.load(args.IK_file_path)

obj_poses       = data["obj_poses"]
joints_l        = data["joints_l"]
joints_r        = data["joints_r"]
joints_target_l = data["joints_target_l"]
joints_target_r = data["joints_target_r"]
arm_l_pose    = data["arm_l_pose"]
arm_r_pose    = data["arm_r_pose"]
dt              = data["dt"]

N = len(joints_l)


physics_dt = 1.0 / 240.0

world = World(physics_dt=physics_dt)
world

z_position = 0 if args.simulate else -0.5
world.scene.add_default_ground_plane(z_position=z_position)


normal_friction = PhysicsMaterial(
    prim_path="/World/PhysicsMaterials/normal_friction",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.0,
)


prim_path_box = "/World/envs/env_0/box"

# box = DynamicCuboid(
#     prim_path=prim_path_box,
#     name="box",
#     scale=np.array([0.4, 0.6, 0.06]),
#     mass=1.0,
#     physics_material=normal_friction,
# ) if args.simulate else VisualCuboid(
#     prim_path=prim_path_box,
#     name="box",
#     scale=np.array([0.4, 0.6, 0.06]),
# )


usd_path = "./robots/ur5_sphere_1.0.usd"

prim_path_l = "/World/envs/env_0/ur5_left"
prim_path_r = "/World/envs/env_0/ur5_right"
add_reference_to_stage(usd_path, prim_path_l)
add_reference_to_stage(usd_path, prim_path_r)

# Create SingleArticulation wrapper (automatically creates articulation controller)

arm_l = SingleArticulation(prim_path=prim_path_l, name="ur5_left")
arm_r = SingleArticulation(prim_path=prim_path_r, name="ur5_right")

# ghost_prim_path_l = prim_path_l + "_ghost"
# ghost_prim_path_r = prim_path_r + "_ghost"
# add_reference_to_stage(usd_path, ghost_prim_path_l)
# add_reference_to_stage(usd_path, ghost_prim_path_r)

# ghost_arm_l = SingleArticulation(prim_path=ghost_prim_path_l, name="ur5_left_ghost")
# ghost_arm_r = SingleArticulation(prim_path=ghost_prim_path_r, name="ur5_right_ghost")

# from omni.isaac.core.prims import RigidPrim

# ghost_arm_l_prim = RigidPrim(ghost_arm_l.prim_path)
# ghost_arm_l_prim.disable_rigid_body_physics()
# ghost_arm_r_prim = RigidPrim(ghost_arm_r.prim_path)
# ghost_arm_r_prim.disable_rigid_body_physics()

kps = np.ones(6, dtype=np.float32) * 50.0
kds = np.ones(6, dtype=np.float32) * 10.0

def initialize():
    # initialize the world
    world.reset()

    arm_l.initialize()
    arm_r.initialize()

    arm_l.disable_gravity()

    # ghost_arm_l.initialize()
    # ghost_arm_r.initialize()

    arm_l._articulation_controller.set_gains(kps=kps,kds=kds)
    arm_r._articulation_controller.set_gains(kps=kps,kds=kds)

    arm_l.set_joint_positions(joints_l[0])
    arm_l.set_world_pose(position=arm_l_pose[:3], orientation=arm_l_pose[3:])
    arm_r.set_joint_positions(joints_r[0])
    arm_r.set_world_pose(position=arm_r_pose[:3], orientation=arm_r_pose[3:])

    # ghost_arm_l.set_joint_positions(joints_l[0])
    # ghost_arm_l.set_world_pose(position=arm_l_pose[:3], orientation=arm_l_pose[3:])
    # ghost_arm_r.set_joint_positions(joints_r[0])
    # ghost_arm_r.set_world_pose(position=arm_r_pose[:3], orientation=arm_r_pose[3:])

    # box.set_world_pose(orientation=obj_poses[0,3:], position=obj_poses[0, :3])

initialize()


input_iface = carb.input.acquire_input_interface()

def on_keyboard_event(event):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.R:
            print("Restarting simulation")
            initialize()
        

input_iface.subscribe_to_keyboard_events(None, on_keyboard_event)

from isaacsim.core.prims import XFormPrim
ee_l = XFormPrim(f"{prim_path_l}/ur5/wrist_3_link/Sphere")
ee_r = XFormPrim(f"{prim_path_r}/ur5/wrist_3_link/Sphere")

import torch
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

def print_state(i):
    desired_pos_l = data['EE_l'][i] if 'EE_l' in data else data['EE_poses_l'][i, :3]
    desired_pos_r = data['EE_r'][i] if 'EE_r' in data else data['EE_poses_r'][i, :3]

    ee_pos_l, _ = ee_l.get_world_poses()
    ee_pos_r, _ = ee_r.get_world_poses()

    # print(i)
    # print("EE left position: ", ee_pos_l)
    # print(f"Desired EE_l position: {desired_pos_l}")
    # print("EE right position: ", ee_pos_r)
    # print(f"Desired EE_r position: {desired_pos_r}")
    # print(f"Error L: {desired_pos_l - ee_pos_l} {np.linalg.norm(desired_pos_l - ee_pos_l)}")
    # print(f"Error R: {desired_pos_r - ee_pos_r} {np.linalg.norm(desired_pos_r - ee_pos_r)}")
    
    print(f"Desired joint position: {joints_l[i]}")
    print(f"Current joint position: {arm_l.get_joint_positions()}")
    print(f"Joint Position Error: {joints_l[i] - arm_l.get_joint_positions()}")




while simulation_app.is_running():
    world.step(render=True)

    i = min(int(world.current_time // dt), N-1)

    if args.simulate:
        # Create and apply articulation action
        # action_l = ArticulationAction(joint_positions=joints_target_l[i])
        # action_r = ArticulationAction(joint_positions=joints_target_r[i])
        action_l = ArticulationAction(joint_positions=joints_l[i])
        action_r = ArticulationAction(joint_positions=joints_r[i])
        arm_l.apply_action(action_l)
        arm_r.apply_action(action_r)
    else:
        arm_l.set_joint_positions(joints_l[i])
        arm_r.set_joint_positions(joints_r[i])
        # box.set_world_pose(orientation=obj_poses[i,3:], position=obj_poses[i, :3])

    # ghost_arm_l.set_joint_positions(joints_l[i])
    # ghost_arm_r.set_joint_positions(joints_r[i])
    print_state(i)

    time.sleep(physics_dt)
