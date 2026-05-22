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
parser.add_argument("--box_dims", type=str, default="0.357,0.259,0.277",
                    help="Box visual dimensions Lx,Ly,Lz in meters.")
parser.add_argument("--env_idx", type=int, default=0,
                    help="For multi-env record.py rollouts ((T, N, ...) schema), which env to "
                         "visualize. Default 0 (first env). Ignored for legacy single-env files.")
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
    rollouts (actual_q has 12 cols on the joint axis) the left/right joint streams are
    returned separately so the renderer can drive two arms in parallel.

    record.py with --num_envs>1 saves arrays with an extra N dim ((T, N, ...)). We detect
    this via the `num_envs` field and slice down to args.env_idx so the downstream rendering
    code stays single-env.

    Returns a dict with keys:
      q_l, q_r, gt_l, gt_r  — (T_ds, 6) per-arm joint positions (actual + reference)
      obj_pos, obj_quat     — (T_ds, 3/4) box pose stream (or None)
      phase                 — (T_ds,) per-step trajectory phase (or None)
      dual_arm              — bool, whether the original rollout had 12-DOF actions
      arm_l_pose, arm_r_pose — (7,) base poses for dual-arm rollouts (or None)
      arm_pose              — (7,) base pose for single-arm rollouts (or None)
    """
    d = np.load(file_path)

    n_envs = int(d["num_envs"]) if "num_envs" in d.files else 0
    if n_envs > 0:
        if args.env_idx < 0 or args.env_idx >= n_envs:
            raise SystemExit(f"--env_idx={args.env_idx} out of range; rollout has num_envs={n_envs}")
        print(f"[visualize_traj] multi-env rollout (num_envs={n_envs}); visualizing env {args.env_idx}")
        # `done_step` is (N,) — -1 if never done. Mention how long this env's episode was so the
        # operator knows where the visualization will start showing post-reset garbage.
        if "done_step" in d.files:
            ds = int(d["done_step"][args.env_idx])
            print(f"[visualize_traj] env {args.env_idx} done_step={ds} "
                  f"(-1 means never done — frames past that show post-reset state)")

    def _pick(arr):
        """Slice the per-env axis if this is a multi-env array; passthrough otherwise."""
        if n_envs > 0 and arr.ndim >= 2 and arr.shape[1] == n_envs:
            return arr[:, args.env_idx]
        return arr

    actual   = _pick(d["actual_q"])    # (T, 6) single-arm or (T, 12) dual-arm
    expected = _pick(d["expected_q"])

    src_dt = float(d["src_dt"]) if "src_dt" in d.files else (1.0 / 500.0)
    stride = max(1, int(round(dt / src_dt)))
    actual_ds = actual[::stride]
    expected_ds = expected[::stride]

    # Split joints into left/right if dual-arm; otherwise leave q_r/gt_r as None so the
    # downstream code can branch cleanly on dual_arm rather than always producing zeros.
    dual_arm = actual_ds.shape[-1] == 12 or bool(d["dual_arm"]) if "dual_arm" in d.files else (actual_ds.shape[-1] == 12)
    if actual_ds.shape[-1] == 12:
        q_l, q_r = actual_ds[:, :6], actual_ds[:, 6:]
        gt_l, gt_r = expected_ds[:, :6], expected_ds[:, 6:]
    else:
        q_l = actual_ds
        q_r = None
        gt_l = expected_ds
        gt_r = None

    obj_pos_ds  = _pick(d["actual_obj_pos"])[::stride]  if "actual_obj_pos"  in d.files else None
    obj_quat_ds = _pick(d["actual_obj_quat"])[::stride] if "actual_obj_quat" in d.files else None
    phase_ds    = _pick(d["phase"])[::stride] if "phase" in d.files else None

    # Arm base poses (planner-frame). Real rollouts may not save these.
    arm_l_pose = np.asarray(d["arm_l_pose"]) if "arm_l_pose" in d.files else None
    arm_r_pose = np.asarray(d["arm_r_pose"]) if "arm_r_pose" in d.files else None
    arm_pose   = np.asarray(d["arm_pose"])   if "arm_pose"   in d.files else None

    return {
        "q_l": q_l, "q_r": q_r, "gt_l": gt_l, "gt_r": gt_r,
        "obj_pos": obj_pos_ds, "obj_quat": obj_quat_ds, "phase": phase_ds,
        "dual_arm": dual_arm,
        "arm_l_pose": arm_l_pose, "arm_r_pose": arm_r_pose, "arm_pose": arm_pose,
    }

extension = args.joint_target_file_1.split(".")[-1]
if extension != "npz":
    raise SystemExit(f"Expected an .npz rollout file, got: {args.joint_target_file_1!r}")
_d = read_npz(args.joint_target_file_1)
q_joints_l    = _d["q_l"]
q_joints_r    = _d["q_r"]              # None for single-arm
q_joints_gt_l = _d["gt_l"]
q_joints_gt_r = _d["gt_r"]             # None for single-arm
obj_pos       = _d["obj_pos"]
obj_quat      = _d["obj_quat"]
phase         = _d["phase"]
dual_arm      = _d["dual_arm"]
arm_l_pose    = _d["arm_l_pose"]       # may be None on real-rollout npzs
arm_r_pose    = _d["arm_r_pose"]
arm_pose      = _d["arm_pose"]
print(f"[visualize_traj] dual_arm={dual_arm}")

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


def _bind_green_material(prim_path):
    """Bind a green UsdPreviewSurface material to all Mesh/Subset prims under prim_path.
    Used to recolor the reference arms green so they're visually distinct from the
    'actual' arms driven by the recorded trajectory."""
    safe = prim_path.replace("/", "_").lstrip("_")
    mat_path = f"/World/Looks/{safe}_green"
    material = UsdShade.Material.Define(world.stage, mat_path)
    shader = UsdShade.Shader.Define(world.stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 1.0, 0.0))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    arm_prim = world.stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(arm_prim, Usd.TraverseInstanceProxies()):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    for prim in Usd.PrimRange(arm_prim):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Subset):
            UsdShade.MaterialBindingAPI(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )


def _spawn_arm(prim_path, base_pose=None, green=False, name=None):
    """Spawn a UR5e reference at prim_path, optionally place it at base_pose
    ([pos_xyz, quat_wxyz]) and bind the green reference material. Returns a
    SingleArticulation wrapper.

    Precision note: `ClearXformOpOrder` only empties the op-order metadata, it does NOT
    delete the underlying attribute schemas already on the prim. The UR5e USD wrapper
    ships with double-precision translate/orient attribute schemas (quatd / double3), so
    calling AddTranslateOp/AddOrientOp without specifying precision picks Float and
    collides with the existing typed attribute. We explicitly request PrecisionDouble
    here to match, and use Gf.Vec3d / Gf.Quatd accordingly.
    """
    add_reference_to_stage(usd_path, prim_path)
    if base_pose is not None:
        prim = world.stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(
            float(base_pose[0]), float(base_pose[1]), float(base_pose[2])
        ))
        # arm_*_pose is wxyz (matches the env's convention); Gf.Quatd is (real, imag-Vec3d).
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(
            float(base_pose[3]),
            Gf.Vec3d(float(base_pose[4]), float(base_pose[5]), float(base_pose[6])),
        ))
    if green:
        _bind_green_material(prim_path)
    return SingleArticulation(prim_path=prim_path, name=name or prim_path.split("/")[-1])


# Spawn arms based on the rollout layout.
#   dual_arm: 2 actual arms (left+right) at their planner base poses, optionally 2 ref
#             arms (green) at the same poses for visual comparison.
#   else (single-arm): 1 actual arm + 1 ref arm at the origin (legacy layout). The ref
#             arm always exists because Isaac Sim's articulation discovery occasionally
#             needs more than one to initialize cleanly; when --show_ref is off we just
#             hide its meshes.
if dual_arm:
    if arm_l_pose is None or arm_r_pose is None:
        raise SystemExit("Dual-arm rollout but the npz is missing arm_l_pose / arm_r_pose. "
                         "Re-record with the current record.py (those fields are saved automatically) "
                         "or pass --dual_arm via a single-arm path if you meant single-arm.")
    prim_path_l_actual = "/World/envs/env_0/ur5_l_actual"
    prim_path_r_actual = "/World/envs/env_0/ur5_r_actual"
    arm_l_actual = _spawn_arm(prim_path_l_actual, base_pose=arm_l_pose, name="ur5_l_actual")
    arm_r_actual = _spawn_arm(prim_path_r_actual, base_pose=arm_r_pose, name="ur5_r_actual")
    actual_arms = [(arm_l_actual, prim_path_l_actual), (arm_r_actual, prim_path_r_actual)]

    ref_arms = []
    if args.show_ref:
        prim_path_l_ref = "/World/envs/env_0/ur5_l_ref"
        prim_path_r_ref = "/World/envs/env_0/ur5_r_ref"
        arm_l_ref = _spawn_arm(prim_path_l_ref, base_pose=arm_l_pose, green=True, name="ur5_l_ref")
        arm_r_ref = _spawn_arm(prim_path_r_ref, base_pose=arm_r_pose, green=True, name="ur5_r_ref")
        ref_arms = [(arm_l_ref, prim_path_l_ref), (arm_r_ref, prim_path_r_ref)]

    # Contact sensors / pause-on-contact don't make sense for the dual-arm box lift —
    # contact with the cube is the normal operating mode. Skip the sensor setup.
    robot_contact_sensors = []
else:
    prim_path_1 = "/World/envs/env_0/ur5_1"
    arm_1 = _spawn_arm(prim_path_1, name="ur5_1")
    robot_contact_sensors = [
        ContactSensor(prim_path_1 + f"/{link_name}/ContactSensor")
        for link_name in ["shoulder_link", "upper_arm_link", "forearm_link",
                          "wrist_1_link", "wrist_2_link", "wrist_3_link"]
    ]
    prim_path_2 = "/World/envs/env_0/ur5_2"
    arm_2 = _spawn_arm(prim_path_2, green=True, name="ur5_2")
    # Hide the ref arm's meshes when --show_ref is off (articulation still exists for
    # Isaac Sim's discovery to succeed, just not rendered).
    if not args.show_ref:
        UsdGeom.Imageable(world.stage.GetPrimAtPath(prim_path_2)).MakeInvisible()
    actual_arms = [(arm_1, prim_path_1)]
    ref_arms = [(arm_2, prim_path_2)]

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

# Disable inter-arm collisions. Each arm gets its own collision group, and every group
# is filtered against every other one — so actual_l, actual_r, ref_l, ref_r (whichever
# exist) all pass through each other. The arms are kinematic visualizers driven by
# set_joint_positions; we don't want PhysX trying to resolve overlap with corrective
# forces that would push the rendered pose away from the recorded data.
all_arm_prim_paths = [p for _, p in actual_arms + ref_arms]
_arm_groups = []
for i, p in enumerate(all_arm_prim_paths):
    gpath = f"/World/Arm{i}Group"
    g = UsdPhysics.CollisionGroup.Define(world.stage, gpath)
    Usd.CollectionAPI.Apply(g.GetPrim(), UsdPhysics.Tokens.colliders) \
        .CreateIncludesRel().AddTarget(p)
    _arm_groups.append((g, gpath))
for i, (g_i, _) in enumerate(_arm_groups):
    for j, (_, p_j) in enumerate(_arm_groups):
        if i != j:
            g_i.CreateFilteredGroupsRel().AddTarget(p_j)


def _set_arm_joints(arm, q):
    """Wrap set_joint_positions. No-op if q is None (single-arm right-side) or contains
    NaN (record.py NaNs per-env slots after the env hit `done` — we hold the previous
    pose rather than feeding NaN into the articulation, which would break the sim)."""
    if q is None:
        return
    if np.isnan(q).any():
        return
    arm.set_joint_positions(q)


def initialize(robot=0):
    """(Re)initialize all spawned arms and seed them with their first-frame joint state.
    `robot` only applies to single-arm mode where the key handler can toggle between the
    actual (q_l) and reference (q_r) streams as the driver for arm_1."""
    world.reset()
    for arm, _ in actual_arms + ref_arms:
        arm.initialize()
    if dual_arm:
        _set_arm_joints(actual_arms[0][0], q_joints_l[0])
        _set_arm_joints(actual_arms[1][0], q_joints_r[0])
        if ref_arms:
            _set_arm_joints(ref_arms[0][0], q_joints_gt_l[0])
            _set_arm_joints(ref_arms[1][0], q_joints_gt_r[0])
    else:
        # Single-arm legacy: q_r/gt_r are None, the keyboard toggle picks the stream.
        init_q  = q_joints_l[0]  if robot == 0 or q_joints_r    is None else q_joints_r[0]
        init_q2 = q_joints_gt_l[0] if robot == 0 or q_joints_gt_r is None else q_joints_gt_r[0]
        _set_arm_joints(actual_arms[0][0], init_q)
        _set_arm_joints(ref_arms[0][0],   init_q2)


robot = 0
initialize(robot)

input_iface = carb.input.acquire_input_interface()

def on_keyboard_event(event):
    """R resets the rollout. 0/1 toggle the single-arm driver stream — no-op in dual-arm."""
    global robot
    if event.type != carb.input.KeyboardEventType.KEY_PRESS:
        return
    if event.input == carb.input.KeyboardInput.R:
        print("Restarting simulation")
        initialize(robot)
    elif event.input == carb.input.KeyboardInput.KEY_0 and not dual_arm:
        print("Switch to robot 0")
        robot = 0
        initialize(robot)
    elif event.input == carb.input.KeyboardInput.KEY_1 and not dual_arm:
        print("Switch to robot 1")
        robot = 1
        initialize(robot)


input_iface.subscribe_to_keyboard_events(None, on_keyboard_event)

dt = 1 / args.hz
N = len(q_joints_l)

while simulation_app.is_running():
    world.step(render=True)

    i = min(int(world.current_time // dt), N - 1)

    if dual_arm:
        _set_arm_joints(actual_arms[0][0], q_joints_l[i])
        _set_arm_joints(actual_arms[1][0], q_joints_r[i])
        if ref_arms:
            _set_arm_joints(ref_arms[0][0], q_joints_gt_l[i])
            _set_arm_joints(ref_arms[1][0], q_joints_gt_r[i])
    else:
        q_joints_1 = q_joints_l[i] if robot == 0 or q_joints_r is None else q_joints_r[i]
        q_joints_2 = q_joints_gt_l[i] if robot == 0 or q_joints_gt_r is None else q_joints_gt_r[i]
        _set_arm_joints(actual_arms[0][0], q_joints_1)
        _set_arm_joints(ref_arms[0][0],   q_joints_2)

    # Drive the box visual from recorded pose, skipping frames where pose was missing.
    if box_visual is not None and not np.isnan(obj_pos[i]).any():
        box_visual.set_world_pose(position=obj_pos[i], orientation=obj_quat[i])

    # Drive the reference box marker from the loaded ref trajectory at this step's phase.
    if ref_box_visual is not None:
        ref_box_visual.set_world_pose(position=ref_obj_pos[i], orientation=ref_obj_quat[i])

    # Contact-sensor based pause-on-contact (legacy debug feature) — single-arm only.
    if robot_contact_sensors:
        contact_readings = [s.get_current_frame() for s in robot_contact_sensors]
        in_contact = np.any([r["in_contact"] and r["force"] > 0.0 for r in contact_readings])
        if in_contact and world.is_playing():
            print(f"IN CONTACT AT STEP: {i}")
            print(q_joints_l[i])
            world.pause()

    time.sleep(dt)
