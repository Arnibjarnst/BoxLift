import numpy as np
import os
import argparse
import time
import yaml

# ----------------------------
# Parse input JSON file
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("joint_target_file_1", type=str)
parser.add_argument("--hz", type=int, default=50)
parser.add_argument("--box_dims", type=str, default="0.34,0.235,0.27",
                    help="Box visual dimensions Lx,Ly,Lz in meters.")
parser.add_argument("--show_ref", action="store_true",
                    help="Also spawn a second (green) arm following the reference trajectory, "
                         "and (if --ref_trajectory_path is provided) a green box marker tracking "
                         "the reference object pose.")
parser.add_argument("--ref_trajectory_path", type=str, default=None,
                    help="Path to a reference trajectory .npz containing obj_poses (T, 7). "
                         "Used to drive the reference box marker when --show_ref is set. "
                         "If omitted, auto-discovers from <run_dir>/params/env.yaml when the "
                         "input npz lives under <run_dir>/ur_rtde_logs/.")
args = parser.parse_args()


def _autodiscover_ref_trajectory(input_path: str) -> str | None:
    """Look for the training run's env.yaml two dirs above the input npz (matching the layout
    ur_rtde_real_time.py writes: <run_dir>/ur_rtde_logs/<run>.npz alongside <run_dir>/params/env.yaml)
    and pull `trajectory_path` from it. Returns None if anything in the chain is missing."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(input_path)))
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    if not os.path.isfile(env_yaml):
        return None
    try:
        with open(env_yaml, "r") as f:
            env_cfg = yaml.unsafe_load(f)
        path = env_cfg.get("trajectory_path") if isinstance(env_cfg, dict) else None
    except Exception as e:
        print(f"[visualize_traj] failed to parse {env_yaml}: {e}")
        return None
    if not path:
        return None
    if not os.path.isfile(path):
        print(f"[visualize_traj] env.yaml says trajectory_path={path!r} but it doesn't exist; ignoring")
        return None
    return path


if args.show_ref and args.ref_trajectory_path is None:
    args.ref_trajectory_path = _autodiscover_ref_trajectory(args.joint_target_file_1)
    if args.ref_trajectory_path is not None:
        print(f"[visualize_traj] auto-discovered ref trajectory: {args.ref_trajectory_path}")

dt = 1 / args.hz
box_dims = tuple(float(x) for x in args.box_dims.split(","))
assert len(box_dims) == 3, "--box_dims must be 'Lx,Ly,Lz'"

def read_npz(file_path):
    """Read rollout NPZ produced by either ur_rtde_real_time.py (500Hz real-robot log) or
    record.py (50Hz sim rollout). Source rate is read from `src_dt` if present (record.py
    writes it); otherwise defaults to 1/500 for the legacy real-rollout schema. For dual-arm
    rollouts (actual_q has 12 cols) we split into left/right arm vectors."""
    d = np.load(file_path)
    actual = d["actual_q"]      # (N, 6) single-arm, (N, 12) dual-arm
    expected = d["expected_q"]  # same shape

    src_dt = float(d["src_dt"]) if "src_dt" in d.files else (1.0 / 500.0)
    stride = max(1, int(round(dt / src_dt)))
    actual_ds = actual[::stride]
    expected_ds = expected[::stride]

    # Split joints into left/right if dual-arm; otherwise put everything on the left and
    # zero the right (visualize_traj only ever drives one robot from this output).
    if actual_ds.shape[-1] == 12:
        q_l, q_r = actual_ds[:, :6], actual_ds[:, 6:]
        gt_l, gt_r = expected_ds[:, :6], expected_ds[:, 6:]
    else:
        q_l = actual_ds
        q_r = np.zeros_like(actual_ds)
        gt_l = expected_ds
        gt_r = np.zeros_like(expected_ds)

    # Optional box pose stream (present in both real and sim rollouts).
    obj_pos_ds = d["actual_obj_pos"][::stride] if "actual_obj_pos" in d.files else None
    obj_quat_ds = d["actual_obj_quat"][::stride] if "actual_obj_quat" in d.files else None

    # Per-step trajectory phase. Used to drive the reference box marker.
    phase_ds = d["phase"][::stride] if "phase" in d.files else None

    return q_l, q_r, gt_l, gt_r, obj_pos_ds, obj_quat_ds, phase_ds

extension = args.joint_target_file_1.split(".")[-1]
if extension != "npz":
    raise SystemExit(f"Expected an .npz rollout file, got: {args.joint_target_file_1!r}")
q_joints_l, q_joints_r, q_joints_gt_l, q_joints_gt_r, obj_pos, obj_quat, phase = read_npz(args.joint_target_file_1)

# Reference object trajectory (driven by per-step phase). Loaded only if --show_ref AND a
# reference trajectory file is provided. ref_obj_pos/ref_obj_quat are precomputed per step
# by interpolating the ref obj_poses at the logged phase, so the main loop can index by `i`.
ref_obj_pos = None
ref_obj_quat = None
if args.show_ref and args.ref_trajectory_path is not None:
    ref_data = np.load(args.ref_trajectory_path)
    ref_obj_poses = ref_data["obj_poses"]  # (T, 7) [pos_xyz, quat_wxyz]
    T_ref = ref_obj_poses.shape[0]
    if phase is not None:
        # Linear interpolation along time axis. Phase saved by ur_rtde_real_time is the
        # same float index used in the env's _interp.
        idx = np.clip(phase, 0.0, T_ref - 1 - 1e-6)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, T_ref - 1)
        a = (idx - i0)[:, None]
        ref_obj_pos = (1.0 - a) * ref_obj_poses[i0, :3] + a * ref_obj_poses[i1, :3]
        # nlerp for quats with hemisphere correction.
        q0 = ref_obj_poses[i0, 3:]
        q1 = ref_obj_poses[i1, 3:]
        dot = (q0 * q1).sum(axis=-1, keepdims=True)
        q1 = np.where(dot < 0, -q1, q1)
        q = (1.0 - a) * q0 + a * q1
        ref_obj_quat = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-8)
    else:
        # No per-step phase (e.g. JSON input): fall back to step-index → trajectory-index
        # 1:1 mapping, clamped to trajectory length.
        N_log = len(q_joints_l)
        idx = np.minimum(np.arange(N_log, dtype=np.int64), T_ref - 1)
        ref_obj_pos = ref_obj_poses[idx, :3]
        ref_obj_quat = ref_obj_poses[idx, 3:]


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

# Reference box marker (green) driven by the loaded reference trajectory + phase.
ref_box_visual = None
if ref_obj_pos is not None and ref_obj_quat is not None:
    from isaacsim.core.api.objects import VisualCuboid
    ref_box_visual = VisualCuboid(
        prim_path="/World/ref_box_visual",
        name="ref_box_visual",
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz
        scale=np.array(box_dims, dtype=float),
        color=np.array([0.0, 1.0, 0.0]),
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

    # TEMP: arm_1 wrist_3 (TCP) pose in world frame, in ur_rtde getTCPPose format (x,y,z,rx,ry,rz)
    _w3_xf = UsdGeom.Xformable(world.stage.GetPrimAtPath(f"{prim_path_1}/wrist_3_link")).ComputeLocalToWorldTransform(0)
    _w3_pos = _w3_xf.ExtractTranslation()
    _w3_quat = _w3_xf.ExtractRotationQuat()  # (imag_xyz, real_w)
    _qx, _qy, _qz, _qw = _w3_quat.imaginary[0], _w3_quat.imaginary[1], _w3_quat.imaginary[2], _w3_quat.real
    _vnorm = np.sqrt(_qx * _qx + _qy * _qy + _qz * _qz)
    _angle = 2.0 * np.arctan2(_vnorm, _qw)
    if _vnorm < 1e-8:
        _rx = _ry = _rz = 0.0
    else:
        _scale = _angle / _vnorm
        _rx, _ry, _rz = _qx * _scale, _qy * _scale, _qz * _scale
    print(f"arm_1 wrist_3 TCP=({_w3_pos[0]:.4f}, {_w3_pos[1]:.4f}, {_w3_pos[2]:.4f}, {_rx:.4f}, {_ry:.4f}, {_rz:.4f})")

    # Drive the box visual from recorded pose, skipping frames where pose was missing.
    if box_visual is not None and not np.isnan(obj_pos[i]).any():
        box_visual.set_world_pose(position=obj_pos[i], orientation=obj_quat[i])

    # Drive the reference box marker from the loaded ref trajectory at this step's phase.
    if ref_box_visual is not None:
        ref_box_visual.set_world_pose(position=ref_obj_pos[i], orientation=ref_obj_quat[i])

    contact_readings = [sensor.get_current_frame() for sensor in robot_contact_sensors]
    in_contact = np.any([r["in_contact"] and r["force"] > 0.0 for r in contact_readings])
    
    if in_contact and world.is_playing():
        print(f"IN CONTACT AT STEP: {i} ")
        print(q_joints_1)
        world.pause()

    time.sleep(dt)
