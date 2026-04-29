import numpy as np
import os
import argparse
import json
import time

# ----------------------------
# Parse input JSON file
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("joint_target_file_1", type=str)
parser.add_argument("--hz", type=int, default=50)
args = parser.parse_args()

dt = 1 / args.hz

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    q_joints = np.array(data["joint_positions"])

    return q_joints[:, :6], q_joints[:, 6:], np.zeros_like(q_joints[:, :6]), np.zeros_like(q_joints[:, :6])

def read_npz(file_path):
    # Downsample 500Hz log to args.hz without interpolation
    d = np.load(file_path)
    actual = d["actual_q"]      # (N, 6)
    expected = d["expected_q"]  # (N, 6)

    # Source rate is 500Hz (1/dt = 500)
    src_dt = 1 / 500.0
    stride = max(1, int(round(dt / src_dt)))
    actual_ds = actual[::stride]
    expected_ds = expected[::stride]

    zeros = np.zeros_like(actual_ds)
    return actual_ds, zeros, expected_ds, zeros

extension = args.joint_target_file_1.split(".")[-1]
if extension == "json":
    q_joints_l , q_joints_r, q_joints_gt_l, q_joints_gt_r = read_json(args.joint_target_file_1)
elif extension == "npz":
    q_joints_l , q_joints_r, q_joints_gt_l, q_joints_gt_r = read_npz(args.joint_target_file_1)


np.set_printoptions(precision=5, suppress=True)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.world import World
import carb.input
from omni.isaac.sensor import ContactSensor
from pxr import Usd, UsdPhysics, Sdf, Gf, UsdShade, UsdGeom

world = World()

ground_prim_path = "/World/GroundPlane"
world.scene.add_default_ground_plane(prim_path=ground_prim_path, z_position=0.00)

usd_path = "./robots/ur5e_sphere.usd"

prim_path_1 = "/World/envs/env_0/ur5_1"
add_reference_to_stage(usd_path, prim_path_1)

robot_contact_sensors = [
    ContactSensor(prim_path_1 + f"/{link_name}/ContactSensor")
    for link_name in ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
]

# Create SingleArticulation wrapper (automatically creates articulation controller)
arm_1 = SingleArticulation(prim_path=prim_path_1, name="ur5_1")

prim_path_2 = "/World/envs/env_0/ur5_2"
add_reference_to_stage(usd_path, prim_path_2)
arm_2 = SingleArticulation(prim_path=prim_path_2, name="ur5_2")

# Disable collisions between the robots
arm_1_group_path = "/World/Arm1Group"
arm_2_group_path = "/World/Arm2Group"

arm_1_group = UsdPhysics.CollisionGroup.Define(world.stage, arm_1_group_path)
arm_2_group = UsdPhysics.CollisionGroup.Define(world.stage, arm_2_group_path)

arm_1_col_api = Usd.CollectionAPI.Apply(arm_1_group.GetPrim(), UsdPhysics.Tokens.colliders)
arm_2_col_api = Usd.CollectionAPI.Apply(arm_2_group.GetPrim(), UsdPhysics.Tokens.colliders)

arm_1_col_api.CreateIncludesRel().AddTarget(prim_path_1)
arm_2_col_api.CreateIncludesRel().AddTarget(prim_path_2)

arm_1_group.CreateFilteredGroupsRel().AddTarget(arm_2_group_path)


# Modify arm visuals
material_path = "/World/Looks/arm_2_material"

material = UsdShade.Material.Define(world.stage, material_path)
shader = UsdShade.Shader.Define(world.stage, f"{material_path}/Shader")
shader.CreateIdAttr("UsdPreviewSurface")

shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))

material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

arm_2_prim = world.stage.GetPrimAtPath(prim_path_2)

for prim in Usd.PrimRange(arm_2_prim, Usd.TraverseInstanceProxies()):
    if prim.IsInstanceable():
        prim.SetInstanceable(False)

for prim in Usd.PrimRange(arm_2_prim):
    if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_api.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )



def initialize(robot):
    # initialize the world
    world.reset()

    arm_1.initialize()
    arm_2.initialize()

    # arm_2.set_world_pose(position=[1,0,0])

    init_q = q_joints_l[0] if robot == 0 else q_joints_r[0]
    arm_1.set_joint_positions(init_q)

    init_q_2 = q_joints_gt_l[0] if robot == 0 else q_joints_gt_r[0]
    arm_2.set_joint_positions(init_q_2)


robot = 0
initialize(robot)

input_iface = carb.input.acquire_input_interface()

def on_keyboard_event(event):
    global robot
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        if event.input == carb.input.KeyboardInput.R:
            print("Restarting simulation")
            initialize(robot)
        if event.input == carb.input.KeyboardInput.KEY_0:
            print("Restarting simulation")
            robot = 0
            initialize(robot)
        if event.input == carb.input.KeyboardInput.KEY_1:
            print("Restarting simulation")
            robot = 1
            initialize(robot)
        

input_iface.subscribe_to_keyboard_events(None, on_keyboard_event)

dt = 1 / args.hz
N = len(q_joints_l)

while simulation_app.is_running():
    world.step(render=True)
    
    i = min(int(world.current_time // dt), N-1)

    q_joints_1 = q_joints_l[i] if robot == 0 else q_joints_r[i]
    q_joints_2 = q_joints_gt_l[i] if robot == 0 else q_joints_gt_r[i]

    arm_1.set_joint_positions(q_joints_1)
    arm_2.set_joint_positions(q_joints_2)

    contact_readings = [sensor.get_current_frame() for sensor in robot_contact_sensors]
    in_contact = np.any([r["in_contact"] and r["force"] > 0.0 for r in contact_readings])
    
    if in_contact and world.is_playing():
        print(f"IN CONTACT AT STEP: {i} ")
        print(q_joints_1)
        world.pause()

    time.sleep(dt)
