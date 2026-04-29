import numpy as np
import time
import argparse

np.set_printoptions(precision=5, suppress=True)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.world import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, VisualSphere
from isaacsim.core.api.materials import PhysicsMaterial

import carb.input


parser = argparse.ArgumentParser()
parser.add_argument("IK_file_path", type=str)
parser.add_argument("--simulate", action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()

data = np.load(args.IK_file_path)
files = set(data.files)

# Detect single-arm (box_push) vs dual-arm (box_lift) from trajectory keys
dual_arm = "joints_r" in files

skip_k = 0
obj_poses = data["obj_poses"][skip_k:]

if dual_arm:
    joints_list = [data["joints_l"][skip_k:], data["joints_r"][skip_k:]]
    joints_target_list = [data["joints_target_l"][skip_k:], data["joints_target_r"][skip_k:]]
    arm_poses = [data["arm_l_pose"], data["arm_r_pose"]]
    arm_names = ["ur5e_left", "ur5e_right"]
    ee_poses_list = [
        data["EE_poses_l"][skip_k:] if "EE_poses_l" in files else None,
        data["EE_poses_r"][skip_k:] if "EE_poses_r" in files else None,
    ]
else:
    joints_list = [data["joints"][skip_k:]]
    joints_target_list = [data["joints_target"][skip_k:]]
    arm_poses = [data["arm_pose"]]
    arm_names = ["ur5e"]
    ee_poses_list = [data["EE_poses"][skip_k:] if "EE_poses" in files else None]

dt = float(data["dt"])
N = len(joints_list[0])

# Object properties from npz with defaults for older box_lift trajectories
object_dims = data["object_dims"] if "object_dims" in files else np.array([0.4, 0.6, 0.06])
object_mass = float(data["object_mass"]) if "object_mass" in files else 1.0

print(f"Loaded trajectory: {args.IK_file_path}")
print(f"  {'dual-arm (box_lift)' if dual_arm else 'single-arm (box_push)'}, N={N}, dt={dt}")
print(f"  object_dims={object_dims}, object_mass={object_mass}")


physics_dt = 1.0 / 240.0
world = World(physics_dt=physics_dt)

print(data.keys())

z_position = 0 + 0.018 if args.simulate else -0.5
world.scene.add_default_ground_plane(z_position=z_position)


normal_friction = PhysicsMaterial(
    prim_path="/World/PhysicsMaterials/normal_friction",
    static_friction=0.5,
    dynamic_friction=0.5,
    restitution=0.0,
)


prim_path_box = "/World/envs/env_0/box"

box = DynamicCuboid(
    prim_path=prim_path_box,
    name="box",
    scale=object_dims,
    mass=object_mass,
    physics_material=normal_friction,
) if args.simulate else VisualCuboid(
    prim_path=prim_path_box,
    name="box",
    scale=object_dims,
)

# Desired-box pose marker: a translucent-coloured visual cuboid showing where the
# planner expects the box to be. Only useful in --simulate mode (without simulation,
# the actual box already gets snapped to the reference each step). Tracks obj_poses[i]
# in the main loop.
desired_box_marker = VisualCuboid(
    prim_path="/World/envs/env_0/desired_box",
    name="desired_box",
    scale=object_dims,
    color=np.array([0.0, 1.0, 0.0]),
) if args.simulate else None


usd_path = "./robots/ur5e_sphere.usd" if dual_arm else "./robots/ur5e.usd"

arms = []
for name in arm_names:
    prim_path = f"/World/envs/env_0/{name}"
    add_reference_to_stage(usd_path, prim_path)
    arms.append(SingleArticulation(prim_path=prim_path, name=name))

ee_markers = []
for name, ee_poses in zip(arm_names, ee_poses_list):
    if ee_poses is None:
        ee_markers.append(None)
        continue
    marker = VisualSphere(
        prim_path=f"/World/envs/env_0/{name}_ee_marker",
        name=f"{name}_ee_marker",
        radius=0.013,
        color=np.array([1.0, 0.0, 0.0]),
    )
    ee_markers.append(marker)

# kps = np.ones(6, dtype=np.float32) * 150.0
kps = np.array([800, 600, 300, 200, 100, 100])
kds = np.ones(6, dtype=np.float32) * 22.5
kds = kps * 0.1



def initialize():
    world.reset()
    for arm, joints, arm_pose in zip(arms, joints_list, arm_poses):
        arm.initialize()
        arm._articulation_controller.set_gains(kps=kps, kds=kds)
        arm.set_joint_positions(joints[0])
        arm.set_world_pose(position=arm_pose[:3], orientation=arm_pose[3:])
        arm.set_joint_velocities(np.zeros_like(joints[0]))
        arm.disable_gravity()
    box.set_world_pose(orientation=obj_poses[0, 3:], position=obj_poses[0, :3])
    if desired_box_marker is not None:
        desired_box_marker.set_world_pose(
            position=obj_poses[0, :3], orientation=obj_poses[0, 3:]
        )
    for marker, ee_poses in zip(ee_markers, ee_poses_list):
        if marker is None:
            continue
        marker.set_world_pose(position=ee_poses[0, :3], orientation=ee_poses[0, 3:])


initialize()


input_iface = carb.input.acquire_input_interface()


def on_keyboard_event(event):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.R:
            print("Restarting simulation")
            initialize()


input_iface.subscribe_to_keyboard_events(None, on_keyboard_event)


while simulation_app.is_running():
    world.step(render=True)

    i = min(int(world.current_time // dt), N - 1)
    print(i)

    if args.simulate:
        for arm, joints_target in zip(arms, joints_target_list):
            arm.apply_action(ArticulationAction(joint_positions=joints_target[i]))
        if desired_box_marker is not None:
            desired_box_marker.set_world_pose(
                position=obj_poses[i, :3], orientation=obj_poses[i, 3:]
            )
    else:
        for arm, joints in zip(arms, joints_list):
            arm.set_joint_positions(joints[i])
        box.set_world_pose(orientation=obj_poses[i, 3:], position=obj_poses[i, :3])

    for marker, ee_poses in zip(ee_markers, ee_poses_list):
        if marker is None:
            continue
        marker.set_world_pose(position=ee_poses[i, :3], orientation=ee_poses[i, 3:])

    time.sleep(physics_dt)
