import numpy as np
import csv
import os
import argparse
import json
import time
import ast

# ----------------------------
# Parse input JSON file
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("joint_target_file_1", type=str)
parser.add_argument("--joint_target_file_2", type=str, default=None)
parser.add_argument("--hz", type=int, default=50)
parser.add_argument("--hz2", type=int, default=50)
args = parser.parse_args()


def read_json(file_path):
    with open(file_path) as f:
        data = json.load(f)

    q_joints = np.array(data["joint_positions_log"])

    return q_joints[:, :6], q_joints[:, 6:]

def read_csv(file_path):
    q_joints_l = []
    q_joints_r = []
    
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            # DictReader uses the first row as keys for each dictionary
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                arm_idx = int(row["arm_idx"])
                q_joint = ast.literal_eval(row["actual_q"])
                if arm_idx == 0:
                    q_joints_l.append(q_joint)
                else:
                    q_joints_r.append(q_joint)

    except FileNotFoundError:
        print("Error: File not found. Check your path!")

    q_joints_l = np.array(q_joints_l)
    q_joints_r = np.array(q_joints_r)

    return q_joints_l, q_joints_r

extension = args.joint_target_file_1.split(".")[-1]
if extension == "json":
    q_joints_l , q_joints_r = read_json(args.joint_target_file_1)
elif extension == "csv":
    q_joints_l , q_joints_r = read_csv(args.joint_target_file_1)

q_joints_l_2 = None
q_joints_r_2 = None
if args.joint_target_file_2 is not None:
    extension_2 = args.joint_target_file_2.split(".")[-1]
    if extension_2 == "json":
        q_joints_l_2, q_joints_r_2 = read_json(args.joint_target_file_2)
    elif extension_2 == "csv":
        q_joints_l_2, q_joints_r_2 = read_csv(args.joint_target_file_2)


np.set_printoptions(precision=5, suppress=True)

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})


from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.world import World
from omni.isaac.core.prims import RigidPrim
import carb.input
from pxr import UsdPhysics, PhysxSchema


world = World()

world.scene.add_default_ground_plane()

usd_path = "./robots/ur5e_sphere.usd"

prim_path_1 = "/World/envs/env_0/ur5_1"
add_reference_to_stage(usd_path, prim_path_1)

stage = get_current_stage()

# Create SingleArticulation wrapper (automatically creates articulation controller)
arm_1 = SingleArticulation(prim_path=prim_path_1, name="ur5_1")

arm_2 = None
if q_joints_l_2 is not None:
    prim_path_2 = "/World/envs/env_0/ur5_2"
    add_reference_to_stage(usd_path, prim_path_2)
    arm_2 = SingleArticulation(prim_path=prim_path_2, name="ur5_2")


def initialize(robot):
    # initialize the world
    world.reset()

    arm_1.initialize()

    init_q = q_joints_l[0] if robot == 0 else q_joints_r[0]
    
    arm_1.set_joint_positions(init_q)

    if arm_2:
        init_q_2 = q_joints_l_2[0] if robot == 0 else q_joints_r_2[0]
        arm_2.initialize()
        arm_2.set_world_pose(position=[1,0,0])
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
dt2 = 1 / args.hz2
N1 = len(q_joints_l) if robot == 0 else len(q_joints_r)
# last_pause_i = -1
while simulation_app.is_running():
    world.step(render=True)

    i = min(int(world.current_time // dt), N1-1)

    q_joints_i = q_joints_l[i] if robot == 0 else q_joints_r[i]


    # if i != N1-1:
    #     print(q_joints_i)
    arm_1.set_joint_positions(q_joints_i)

    if arm_2:
        N2 = len(q_joints_l_2) if robot == 0 else len(q_joints_r_2)
        i2 = min(int(world.current_time // dt2), N2-1)
        q_joints_2_i = q_joints_l_2[i2] if robot == 0 else q_joints_r_2[i2]
        arm_2.set_joint_positions(q_joints_2_i)

    time.sleep(dt)
