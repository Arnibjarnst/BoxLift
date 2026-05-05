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
parser.add_argument("--box_dims", type=str, default="0.357,0.259,0.277",
                    help="Box visual dimensions Lx,Ly,Lz in meters.")
parser.add_argument("--show_ref", action="store_true",
                    help="Also spawn a second (green) arm following the reference trajectory.")
args = parser.parse_args()

dt = 1 / args.hz
box_dims = tuple(float(x) for x in args.box_dims.split(","))
assert len(box_dims) == 3, "--box_dims must be 'Lx,Ly,Lz'"

def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    q_joints = np.array(data["joint_positions"])

    return (
        q_joints[:, :6], q_joints[:, 6:],
        np.zeros_like(q_joints[:, :6]), np.zeros_like(q_joints[:, :6]),
        None, None,
    )

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

    # Optional box pose stream (added to npz by ur_rtde_real_time.py).
    obj_pos_ds = d["actual_obj_pos"][::stride] if "actual_obj_pos" in d.files else None
    obj_quat_ds = d["actual_obj_quat"][::stride] if "actual_obj_quat" in d.files else None

    zeros = np.zeros_like(actual_ds)
    return actual_ds, zeros, expected_ds, zeros, obj_pos_ds, obj_quat_ds

extension = args.joint_target_file_1.split(".")[-1]
if extension == "json":
    q_joints_l, q_joints_r, q_joints_gt_l, q_joints_gt_r, obj_pos, obj_quat = read_json(args.joint_target_file_1)
elif extension == "npz":
    q_joints_l, q_joints_r, q_joints_gt_l, q_joints_gt_r, obj_pos, obj_quat = read_npz(args.joint_target_file_1)


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

usd_path = "./robots/ur5e.usd"

prim_path_1 = "/World/envs/env_0/ur5_1"
add_reference_to_stage(usd_path, prim_path_1)

robot_contact_sensors = [
    ContactSensor(prim_path_1 + f"/{link_name}/ContactSensor")
    for link_name in ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
]

# Create SingleArticulation wrapper (automatically creates articulation controller)
arm_1 = SingleArticulation(prim_path=prim_path_1, name="ur5_1")

# Always spawn the second arm — Isaac Sim's articulation discovery seems to need both
# present for arm_1.initialize() to succeed. When --show_ref is off we just hide its
# meshes (further down) so it isn't visible.
prim_path_2 = "/World/envs/env_0/ur5_2"
add_reference_to_stage(usd_path, prim_path_2)
arm_2 = SingleArticulation(prim_path=prim_path_2, name="ur5_2")

# Optional box visual driven by recorded pose-estimation data (npz only).
box_visual = None
if obj_pos is not None and obj_quat is not None and not np.all(np.isnan(obj_pos)):
    from isaacsim.core.api.objects import VisualCuboid
    box_visual = VisualCuboid(
        prim_path="/World/box_visual",
        name="box_visual",
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
        scale=np.array(box_dims, dtype=float),
        color=np.array([0.7, 0.7, 0.7]),
    )

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

# When the user only wants the real trajectory, hide the reference arm's meshes.
# The articulation still exists in physics (so arm_1.initialize() works), but isn't drawn.
if not args.show_ref:
    UsdGeom.Imageable(arm_2_prim).MakeInvisible()



def initialize(robot):
    # initialize the world
    world.reset()

    arm_1.initialize()
    arm_2.initialize()

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

    # Drive the box visual from recorded pose, skipping frames where pose was missing.
    if box_visual is not None and not np.isnan(obj_pos[i]).any():
        box_visual.set_world_pose(position=obj_pos[i], orientation=obj_quat[i])

    contact_readings = [sensor.get_current_frame() for sensor in robot_contact_sensors]
    in_contact = np.any([r["in_contact"] and r["force"] > 0.0 for r in contact_readings])
    
    if in_contact and world.is_playing():
        print(f"IN CONTACT AT STEP: {i} ")
        print(q_joints_1)
        world.pause()

    time.sleep(dt)
