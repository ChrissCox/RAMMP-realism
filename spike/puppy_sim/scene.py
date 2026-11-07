"""The scene, built with MjSpec. World = arm_mount: z up, x forward to the workspace, y left.

Robot (frames from the live URDF/tf in the run files): the Gen3 base tilted +30 deg about y on its
mount, the Robotiq 2F-85 on the end-effector link (fingers open along end-effector x), and the OAK
wrist camera modelled as a small 'face': two light lens rings (eyes) above the gripper (mouth).
Everything else is stylised and APPROXIMATE: the wheelchair, its user (a plain seated mannequin),
the kitchen counter, the round container (placed where the real run closed the gripper) and a
button panel. Kinematic only: poses are written into qpos/mocap, the model is never stepped.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from . import kin, runs

SPIKE_DIR = Path(__file__).resolve().parents[1]
MENAGERIE_DIR = SPIKE_DIR / "third_party" / "mujoco_menagerie"
GEN3_XML = MENAGERIE_DIR / "kinova_gen3" / "gen3.xml"
ROBOTIQ_XML = MENAGERIE_DIR / "robotiq_2f85" / "2f85.xml"
REFERENCE_RUN = "grab_container_20261029_132745"

# ---- robot frames (URDF arm_base_joint, arm_end_effector, arm_tool_frame_joint, wrist_camera tf)
ARM_BASE_PITCH_DEG = 30.0
EE_IN_BRACELET_POS = (0.0, 0.0, -0.061525)
EE_IN_BRACELET_QUAT = (0.0, 1.0, 0.0, 0.0)  # 180 deg about x (wxyz)
TOOL_Z_M = 0.120
EYE_IN_EE = (0.0, 0.065, 0.015)  # OAK optical centre; it looks along end-effector +z
LENS_X_M = (-0.037, 0.038)  # stereo lenses along end-effector x; the colour lens sits at 0
GRIPPER_PREFIX = "g_"
PAD_TIP_IN_PAD = (0.0, -0.0026, 0.0375)  # far end of the menagerie 2F-85 pad box, in the pad body

# ---- OAK housing, stylised so the eyes read at video resolution
OAK_HALF = (0.050, 0.015, 0.0125)  # ~0.10 x 0.03 x 0.025 m
OAK_FACE_Z = 0.018  # front face (end-effector z) of the housing, just ahead of the optical centre
EYE_RING_R, EYE_PUPIL_R, EYE_GLINT_R = 0.0115, 0.0068, 0.0021
EYE_GLINT_OFFSET = (0.0028, 0.0030)  # catchlight, up and to the face's left of the pupil centre
CENTER_LENS_R, CENTER_GLASS_R = 0.0055, 0.0035
LENS_PROUD_M = 0.0016  # how far the lens discs stand off the face

# ---- wheelchair and user: ASSUMED, stylised, approximate (from the fridge keep-out and tf)
FLOOR_Z = -0.75
SEAT_TOP_Z = -0.27
CHAIR_Y = -0.32
BACKREST_X = -0.45
SEAT_FRONT_X = 0.02
SEAT_HALF_WIDTH = 0.235
ARMREST_TOP_Z = -0.05
ARMREST_Y_OFFSET = 0.215  # from the chair centreline
ARMREST_X = (-0.42, -0.08)
WHEEL_R, WHEEL_HALF_W, WHEEL_X, WHEEL_Y_OFFSET = 0.17, 0.03, -0.22, 0.27
CASTER_R, CASTER_X = 0.06, (0.10, -0.52)
HEAD_CENTER = (-0.27, CHAIR_Y, 0.50)
HEAD_R = 0.095

# ---- the container: the run's brim, gripped from the side (grasp.py)
BRIM_RADIUS_M = 0.0843  # outer brim radius
BRIM_WIDTH_M = 0.013
BRIM_THICK_M = 0.003  # the pads' gap at the run's closed knuckle angle
GRASP_DEPTH_M = -0.008  # tool point along the approach from the brim's outer edge (short of it)
CONTAINER_HEIGHT_M = 0.095  # table top to brim top: puts the counter at kitchen height
WALL_TOP_R, WALL_BOTTOM_R, WALL_THICK = 0.0755, 0.064, 0.002
FOOD_R, FOOD_H = 0.058, 0.035
COUNTER_X = (0.55, 1.15)
COUNTER_Y = (-0.62, 0.55)
COUNTER_SLAB_M = 0.035
COUNTER_TOE_KICK = (0.07, 0.10)  # depth, height of the recess at the floor

# ---- button panel (a later 'boop the button' clip), on the counter left of the container
PANEL_CENTER_XY = (0.86, 0.24)
PANEL_YAW_DEG = 15.0  # front face turned toward the arm base
PANEL_HALF = (0.03, 0.12, 0.085)  # depth, width, height
BUTTON_R, BUTTON_PROUD_M = 0.017, 0.008
BUTTON_HEIGHT_M = 0.10  # above the counter top
PRESS_STANDOFF_M = 0.02  # button_press's pre-press STANDOFF_M

# ---- colours (rgba)
ARM_RGBA = (0.87, 0.87, 0.88, 1.0)
CHAIR_RGBA = (0.17, 0.18, 0.20, 1.0)
CUSHION_RGBA = (0.26, 0.27, 0.30, 1.0)
METAL_RGBA = (0.45, 0.46, 0.48, 1.0)
SWEATER_RGBA = (0.44, 0.56, 0.66, 1.0)
PANTS_RGBA = (0.30, 0.33, 0.40, 1.0)
SKIN_RGBA = (0.85, 0.76, 0.68, 1.0)
WOOD_RGBA = (0.80, 0.64, 0.47, 1.0)
CABINET_RGBA = (0.93, 0.92, 0.89, 1.0)
CONTAINER_RGBA = (0.88, 0.95, 1.0, 0.26)
BRIM_RGBA = (0.13, 0.13, 0.15, 1.0)
FOOD_RGBA = (0.98, 0.55, 0.20, 1.0)
PANEL_RGBA = (0.94, 0.91, 0.84, 1.0)
BUTTON_RGBA = (0.89, 0.29, 0.24, 1.0)
OAK_RGBA = (0.11, 0.11, 0.12, 1.0)
EYE_RING_RGBA = (0.93, 0.94, 0.96, 1.0)
EYE_PUPIL_RGBA = (0.04, 0.05, 0.07, 1.0)
GLINT_RGBA = (1.0, 1.0, 1.0, 1.0)
DEFAULT_GEOM_RGBA = [0.5, 0.5, 0.5, 1.0]  # MuJoCo uses the material colour only for this rgba
LATHE_SEGMENTS = 48

# ---- geom groups: what the renderer shows (0-2) and the hidden clearance proxies (3)
VISUAL_GROUP, PROXY_GROUP = 2, 3
# The arm for clearance queries, by part. The base and shoulder (before joint 2) are left out: the
# mount fixes them beside the user. The head part carries the container while it is held.
ARM_PART_BODIES = {  # keys and order: kin.ARM_PARTS
    "upper_arm": ("half_arm_1_link", "half_arm_2_link"),
    "forearm": ("forearm_link",),
    "wrist": ("spherical_wrist_1_link", "spherical_wrist_2_link"),
    "head": ("bracelet_link", "ee"),
}
OFFSCREEN_SIZE = (1920, 1080)


@dataclass(frozen=True)
class Placement:
    """Where the real run closed the gripper, and the container it closed on (world frame)."""

    tool_pos: np.ndarray  # tool frame at closing
    approach: np.ndarray  # tool z at closing
    brim_edge: np.ndarray  # outer brim edge facing the approach (top surface)
    center: np.ndarray  # container axis at the brim top (the container body's origin)
    table_top_z: float


@dataclass
class Scene:
    """The compiled model and what the rest of the package needs to know about it."""

    model: mujoco.MjModel
    placement: Placement
    container_quat: np.ndarray  # resting orientation (wxyz)
    carry_in_tool: tuple  # (pos, quat wxyz) of the container in the tool frame while held
    button_pos: np.ndarray
    button_normal: np.ndarray  # out of the panel
    linkage: dict  # gripper joint angles vs driver angle, for kinematic posing
    groups: dict  # name -> geom ids for clearance queries
    person_head: np.ndarray

    def kin(self):
        """A fresh Kin (its own MjData) on this scene."""
        return kin.Kin(self)

    def press_targets(self, k):
        """Tool poses (p, R) for pressing the button with the closed fingertips along the panel
        normal (the face upright): (pre-press PRESS_STANDOFF_M out, press: fingertips on it)."""
        k.pose(np.zeros(7), grip=1.0)
        p_tool, R_tool = k.site("tool")
        tips = (k.site("g_left_tip")[0] + k.site("g_right_tip")[0]) / 2.0
        reach = float(R_tool[:, 2] @ (tips - p_tool))  # closed fingertips beyond the tool point
        z = -self.button_normal
        y = np.array([0.0, 0.0, 1.0])
        R = np.column_stack([np.cross(y, z), y, z])
        press = self.button_pos + reach * self.button_normal
        return (press + PRESS_STANDOFF_M * self.button_normal, R), (press, R)


def _quat(rot):
    """scipy Rotation -> MuJoCo wxyz."""
    x, y, z, w = rot.as_quat()
    return [w, x, y, z]


def _yaw_quat(deg):
    return _quat(Rotation.from_euler("z", deg, degrees=True))


def _axis_y_quat():
    """Cylinder axis (z) turned onto world y."""
    return _quat(Rotation.from_euler("x", 90, degrees=True))


def _material(spec, name, rgba, specular=0.3, shininess=0.4, reflectance=0.0):
    spec.add_material(name=name, rgba=rgba, specular=specular, shininess=shininess, reflectance=reflectance)
    return name


# ---- motion kinematics: puppy_motion's Chain and clearance geometry (MuJoCo stays the renderer)
@lru_cache(maxsize=4)
def motion_chain(run_name=REFERENCE_RUN):
    """puppy_motion's Chain of the run's own URDF: the kinematics every motion decision uses."""
    from puppy_motion.urdf import Chain

    return Chain.from_urdf(runs.load_run(run_name).urdf)


@lru_cache(maxsize=1)
def person_scene():
    """The mannequin above as puppy_motion's geometry Scene (data/spike_person.json)."""
    from puppy_motion.geometry import Scene as PmScene

    return PmScene.from_param((SPIKE_DIR / "data" / "spike_person.json").read_text())


class MotionKin:
    """The few Kin methods the runs and the camera use (fk, fk_track, clearance_track), computed by
    puppy_motion's Chain and geometry instead of MuJoCo."""

    def __init__(self, chain=None, person=None):
        self.chain = chain or motion_chain()
        self.person = person or person_scene()

    def fk(self, q, site="ee"):
        return self.chain.fk(q, site)

    def fk_track(self, q, site="ee"):
        """(positions (N, 3), rotations (N, 3, 3))."""
        out = [self.chain.fk(qi, site) for qi in np.asarray(q, dtype=float)]
        return np.array([p for p, _ in out]), np.array([R for _, R in out])

    def clearance_track(self, track):
        """(N, 2, 4) clearances (row 0 person, row 1 obstacles; kin.ARM_PARTS columns) along a track."""
        from puppy_motion.geometry import clearances_batch

        out = np.empty((track.n, 2, len(kin.ARM_PARTS)))
        held = np.asarray(track.held, dtype=bool)
        for flag in (False, True):
            rows = held == flag
            if rows.any():
                out[rows] = clearances_batch(self.chain, self.person, np.asarray(track.q)[rows], flag)
        return out


def _geom(body, kind, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), material="", group=1, name="", fromto=None):
    """A visual geom (no contacts: the model is never stepped)."""
    g = body.add_geom(type=kind, size=size, pos=pos, quat=quat, group=group, contype=0, conaffinity=0)
    if fromto is not None:
        g.fromto = fromto
    g.material, g.name = material, name
    return g


def _capsule(body, a, b, r, material, name=""):
    return _geom(body, mujoco.mjtGeom.mjGEOM_CAPSULE, [r, 0, 0], material=material, name=name, fromto=[*a, *b])


def _box(body, center, half, material, quat=(1, 0, 0, 0), name="", group=1):
    box = mujoco.mjtGeom.mjGEOM_BOX
    return _geom(body, box, half, pos=center, quat=quat, material=material, name=name, group=group)


def lathe(profile, segments=LATHE_SEGMENTS):
    """Vertices and faces of the closed solid swept by a counter-clockwise (r, z) profile about z."""
    profile = np.asarray(profile, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    r, z = profile[:, 0], profile[:, 1]
    verts = np.stack(
        [np.outer(np.cos(theta), r).ravel(), np.outer(np.sin(theta), r).ravel(), np.tile(z, segments)], axis=1
    )
    m = len(profile)
    faces = []
    for k in range(segments):
        k1 = (k + 1) % segments
        for j in range(m):
            j1 = (j + 1) % m
            a, b, c, d = k * m + j, k1 * m + j, k1 * m + j1, k * m + j1
            faces += [[a, b, c], [a, c, d]]
    return verts, np.array(faces)


def _add_lathe_mesh(spec, name, profile):
    verts, faces = lathe(profile)
    spec.add_mesh(name=name, uservert=verts.ravel().tolist(), userface=faces.ravel().tolist())
    return name


# ------------------------------------------------------------------------------------------ robot
def _robot_spec():
    """Gen3 on its tilted mount + the 2F-85 + the OAK face. Bodies: ee (= arm_end_effector_link)
    with sites ee, tool and eye."""
    arm = mujoco.MjSpec.from_file(str(GEN3_XML))
    grip = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))
    arm.option.cone, arm.option.impratio = grip.option.cone, grip.option.impratio
    arm.body("base_link").quat = _quat(Rotation.from_euler("y", ARM_BASE_PITCH_DEG, degrees=True))
    _use_bracelet_without_vision(arm)
    arm_mat = _material(arm, "arm_shell", ARM_RGBA, specular=0.45, shininess=0.55)
    for g in arm.geoms:
        if g.group == VISUAL_GROUP:  # the default rgba lets the material's colour through
            g.rgba, g.material = DEFAULT_GEOM_RGBA, arm_mat
    ee = arm.body("bracelet_link").add_body(name="ee", pos=EE_IN_BRACELET_POS, quat=EE_IN_BRACELET_QUAT)
    ee_site = ee.add_site(name="ee")
    ee.add_site(name="tool", pos=(0, 0, TOOL_Z_M))
    ee.add_site(name="eye", pos=EYE_IN_EE)
    _add_face(arm, ee)
    for side in ("left", "right"):  # fingertip sites at the far end of each pad
        grip.body(f"{side}_pad").add_site(name=f"{side}_tip", pos=PAD_TIP_IN_PAD)
    arm.attach(grip, prefix=GRIPPER_PREFIX, site=ee_site)
    return arm


def _use_bracelet_without_vision(arm):
    """The live URDF's bracelet is bracelet_no_vision_link (the OAK replaces Kinova's camera)."""
    arm.add_mesh(name="bracelet_no_vision_link", file="bracelet_no_vision_link.stl")
    for g in arm.body("bracelet_link").geoms:
        g.meshname = "bracelet_no_vision_link"
    arm.delete(arm.camera("wrist"))


def _add_face(spec, ee):
    """The OAK housing (a dark box, its collision proxy too) with two eye rings and a centre lens."""
    m_oak = _material(spec, "oak", OAK_RGBA, specular=0.5, shininess=0.7)
    m_ring = _material(spec, "eye_ring", EYE_RING_RGBA, specular=0.6, shininess=0.8)
    m_pupil = _material(spec, "eye_pupil", EYE_PUPIL_RGBA, specular=1.0, shininess=1.0)
    m_glint = _material(spec, "eye_glint", GLINT_RGBA, specular=0.0)
    cx, cy = EYE_IN_EE[0], EYE_IN_EE[1]
    housing_z = OAK_FACE_Z - OAK_HALF[2]
    _box(ee, (cx, cy, housing_z), OAK_HALF, m_oak, name="oak_housing", group=VISUAL_GROUP)
    _box(ee, (cx, cy - 0.022, housing_z - 0.002), (0.018, 0.009, 0.010), m_oak, group=VISUAL_GROUP)  # bracket
    disc = mujoco.mjtGeom.mjGEOM_CYLINDER

    def lens(x, y, r, proud, material):
        _geom(ee, disc, [r, proud / 2, 0], pos=(x, y, OAK_FACE_Z + proud / 2), material=material, group=VISUAL_GROUP)

    for lx in LENS_X_M:
        lens(lx, cy, EYE_RING_R, LENS_PROUD_M, m_ring)
        lens(lx, cy, EYE_PUPIL_R, LENS_PROUD_M * 1.5, m_pupil)
        lens(lx + EYE_GLINT_OFFSET[0], cy + EYE_GLINT_OFFSET[1], EYE_GLINT_R, LENS_PROUD_M * 2.0, m_glint)
    lens(0.0, cy, CENTER_LENS_R, LENS_PROUD_M, _material(spec, "center_ring", (0.35, 0.36, 0.38, 1)))
    lens(0.0, cy, CENTER_GLASS_R, LENS_PROUD_M * 1.5, m_pupil)


@lru_cache(maxsize=1)
def gripper_linkage():
    """The 2F-85's closed-chain joint angles as functions of the driver angle, by settling the
    menagerie model (alone, no gravity) at a sweep of controls. Returns arrays keyed by joint."""
    spec = mujoco.MjSpec.from_file(str(ROBOTIQ_XML))
    spec.option.gravity = [0, 0, 0]
    m = spec.compile()
    d = mujoco.MjData(m)
    names = ["driver", "coupler", "spring_link", "follower"]
    adr = [m.joint(f"right_{n}_joint").qposadr[0] for n in names]
    rows = []
    for ctrl in np.linspace(0.0, 255.0, 24):
        d.ctrl[0] = ctrl
        mujoco.mj_step(m, d, nstep=1500)
        rows.append(d.qpos[adr].copy())
    table = np.array(rows)
    order = np.argsort(table[:, 0])
    return {n: table[order, i] for i, n in enumerate(names)}


# ------------------------------------------------------------------------------- the placement
@lru_cache(maxsize=4)
def grasp_placement(run_name=REFERENCE_RUN):
    """The tool frame where `run_name` closed the gripper, and the container under it: the brim's
    outer edge sits GRASP_DEPTH_M beyond the tool point along the approach, the axis BRIM_RADIUS_M
    further along the horizontal approach, and the table CONTAINER_HEIGHT_M below the brim."""
    run = runs.load_run(run_name)
    q_close = run.q[runs.closing_index(run)]
    k = kin.Kin.robot_only(_robot_spec().compile())
    p, R = k.fk(q_close, "tool")
    approach = R[:, 2]
    edge = p - GRASP_DEPTH_M * approach
    flat = np.array([approach[0], approach[1], 0.0])
    center = edge + BRIM_RADIUS_M * flat / np.linalg.norm(flat)
    return Placement(p, approach, edge, center, float(edge[2] - CONTAINER_HEIGHT_M))


# ---------------------------------------------------------------------------------- environment
def _add_look(spec):
    """Light background, soft floor, a key light with shadows and a fill."""
    spec.visual.global_.offwidth, spec.visual.global_.offheight = OFFSCREEN_SIZE
    spec.visual.quality.shadowsize = 8192
    spec.visual.quality.offsamples = 8
    spec.visual.quality.numslices, spec.visual.quality.numstacks = 96, 64  # smooth spheres and capsules
    spec.visual.headlight.ambient = [0.42, 0.42, 0.44]
    spec.visual.headlight.diffuse = [0.20, 0.20, 0.20]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.rgba.haze = [0.93, 0.94, 0.96, 1.0]
    spec.stat.center, spec.stat.extent = [0.3, -0.15, -0.1], 1.6
    spec.add_texture(
        name="sky",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.97, 0.975, 0.985],
        rgb2=[0.86, 0.88, 0.91],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="floor_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.87, 0.86, 0.84],
        rgb2=[0.855, 0.845, 0.825],
        width=512,
        height=512,
    )
    floor = spec.add_material(name="floor", reflectance=0.04, texrepeat=[3, 3], texuniform=True, specular=0.1)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor_tex"
    w = spec.worldbody
    _geom(w, mujoco.mjtGeom.mjGEOM_PLANE, [6, 6, 0.05], pos=(0, 0, FLOOR_Z), material="floor", group=0)
    key = w.add_light(name="key", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=True)
    key.dir, key.pos = [-0.45, 0.35, -0.82], [1.5, -1.2, 2.5]
    key.diffuse, key.specular, key.ambient = [0.55, 0.53, 0.50], [0.25, 0.25, 0.25], [0.06, 0.06, 0.06]
    fill = w.add_light(name="fill", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=False)
    fill.dir, fill.pos = [0.55, -0.25, -0.5], [-1.5, 1.0, 2.0]
    fill.diffuse, fill.specular = [0.24, 0.25, 0.27], [0.0, 0.0, 0.0]


def _add_chair(spec):
    """Power wheelchair proxy and the arm's mount column. Returns the chair body."""
    m_frame = _material(spec, "chair", CHAIR_RGBA, specular=0.3, shininess=0.3)
    m_cush = _material(spec, "cushion", CUSHION_RGBA, specular=0.1, shininess=0.1)
    m_metal = _material(spec, "metal", METAL_RGBA, specular=0.6, shininess=0.6)
    b = spec.worldbody.add_body(name="chair")
    cy, top = CHAIR_Y, SEAT_TOP_Z
    seat_cx = (BACKREST_X + SEAT_FRONT_X) / 2
    seat_hx = (SEAT_FRONT_X - BACKREST_X) / 2
    _box(b, (seat_cx, cy, top - 0.04), (seat_hx, SEAT_HALF_WIDTH, 0.04), m_cush, name="seat")
    _box(b, (seat_cx - 0.03, cy, -0.52), (seat_hx + 0.05, SEAT_HALF_WIDTH - 0.02, 0.12), m_frame, name="base")
    _box(b, (BACKREST_X - 0.03, cy, 0.02), (0.03, SEAT_HALF_WIDTH - 0.01, 0.29), m_cush, name="backrest")
    _box(b, (BACKREST_X - 0.01, cy, 0.37), (0.012, 0.02, 0.06), m_metal, name="headrest_post")
    _box(b, (-0.42, cy, 0.50), (0.025, 0.10, 0.065), m_cush, name="headrest")
    arm_hx, arm_cx = (ARMREST_X[1] - ARMREST_X[0]) / 2, sum(ARMREST_X) / 2
    for side in (1, -1):
        y = cy + side * ARMREST_Y_OFFSET
        _box(b, (arm_cx, y, ARMREST_TOP_Z - 0.02), (arm_hx, 0.035, 0.02), m_cush, name=f"armrest_{side}")
        _box(b, (arm_cx - 0.05, y, (top + ARMREST_TOP_Z) / 2 - 0.02), (0.015, 0.012, 0.09), m_metal)
        wheel_y = cy + side * WHEEL_Y_OFFSET
        wz = FLOOR_Z + WHEEL_R
        _geom(
            b,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [WHEEL_R, WHEEL_HALF_W, 0],
            pos=(WHEEL_X, wheel_y, wz),
            quat=_axis_y_quat(),
            material=m_frame,
            name=f"wheel_{side}",
        )
        _geom(
            b,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [WHEEL_R * 0.45, WHEEL_HALF_W + 0.004, 0],
            pos=(WHEEL_X, wheel_y, wz),
            quat=_axis_y_quat(),
            material=m_metal,
        )
        for cx in CASTER_X:
            _geom(
                b,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [CASTER_R, 0.018, 0],
                pos=(cx, cy + side * 0.2, FLOOR_Z + CASTER_R),
                quat=_axis_y_quat(),
                material=m_frame,
            )
    _box(b, (0.19, cy, -0.655), (0.12, 0.17, 0.01), m_frame, name="footrest")
    _box(b, (0.0, cy, -0.47), (0.012, 0.12, 0.17), m_metal)  # footrest hanger, behind the shins
    _add_mount(b, m_metal)
    return b


def _add_mount(body, material):
    """The column and bracket under the arm base (to the chair's side frame)."""
    _geom(
        body,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [0.034, 0.19, 0],
        pos=(0.0, 0.0, -0.215),
        material=material,
        name="mount_column",
    )
    _box(body, (-0.03, -0.06, -0.40), (0.05, 0.07, 0.018), material, name="mount_bracket")


def _add_person(spec):
    """A plain seated mannequin: torso, neck, head (no face), arms on the armrests, legs."""
    sweater = _material(spec, "sweater", SWEATER_RGBA, specular=0.1, shininess=0.1)
    pants = _material(spec, "pants", PANTS_RGBA, specular=0.1, shininess=0.1)
    skin = _material(spec, "skin", SKIN_RGBA, specular=0.15, shininess=0.25)
    shoe = _material(spec, "shoe", CHAIR_RGBA)
    b = spec.worldbody.add_body(name="person")
    cy = CHAIR_Y
    hx, hy, hz = HEAD_CENTER
    _capsule(b, (-0.32, cy, -0.12), (-0.31, cy, 0.22), 0.125, sweater, name="torso")
    _capsule(b, (-0.31, cy - 0.16, 0.255), (-0.31, cy + 0.16, 0.255), 0.07, sweater, name="shoulders")
    _capsule(b, (-0.30, cy, 0.32), (hx - 0.015, cy, hz - 0.08), 0.045, skin, name="neck")
    _geom(b, mujoco.mjtGeom.mjGEOM_SPHERE, [HEAD_R, 0, 0], pos=HEAD_CENTER, material=skin, name="head")
    fore_z = ARMREST_TOP_Z + 0.038
    for side, tag in ((1, "left"), (-1, "right")):
        y = cy + side * ARMREST_Y_OFFSET
        shoulder, elbow = (-0.31, cy + side * 0.2, 0.24), (-0.29, y, fore_z + 0.005)
        wrist = (-0.13, y, fore_z)
        _capsule(b, shoulder, elbow, 0.045, sweater, name=f"upper_arm_{tag}")
        _capsule(b, elbow, wrist, 0.038, sweater, name=f"forearm_{tag}")
        _capsule(b, wrist, (-0.085, y, fore_z - 0.003), 0.03, skin, name=f"hand_{tag}")
        hip, knee = (-0.30, cy + side * 0.10, SEAT_TOP_Z + 0.08), (0.08, cy + side * 0.10, SEAT_TOP_Z + 0.08)
        ankle = (0.13, cy + side * 0.10, -0.58)
        _capsule(b, hip, knee, 0.08, pants, name=f"thigh_{tag}")
        _capsule(b, (0.09, knee[1], knee[2] - 0.02), ankle, 0.055, pants, name=f"shin_{tag}")
        _box(b, (0.18, ankle[1], -0.61), (0.10, 0.045, 0.035), shoe, name=f"foot_{tag}")
    return b


def _add_counter(spec, placement):
    """A kitchen counter whose top is at placement.table_top_z: wood slab on a light cabinet."""
    wood = _material(spec, "wood", WOOD_RGBA, specular=0.2, shininess=0.3)
    cab = _material(spec, "cabinet", CABINET_RGBA, specular=0.15, shininess=0.2)
    b = spec.worldbody.add_body(name="counter")
    top = placement.table_top_z
    (x0, x1), (y0, y1) = COUNTER_X, COUNTER_Y
    cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
    _box(b, (cx, cy, top - COUNTER_SLAB_M / 2), (hx, hy, COUNTER_SLAB_M / 2), wood, name="counter_top")
    kick_d, kick_h = COUNTER_TOE_KICK
    body_bottom = FLOOR_Z + kick_h
    body_top = top - COUNTER_SLAB_M
    _box(
        b,
        (cx + 0.02, cy, (body_top + body_bottom) / 2),
        (hx - 0.02, hy - 0.01, (body_top - body_bottom) / 2),
        cab,
        name="cabinet",
    )
    _box(
        b,
        (cx + kick_d / 2 + 0.02, cy, FLOOR_Z + kick_h / 2),
        (hx - kick_d / 2 - 0.02, hy - 0.02, kick_h / 2),
        _material(spec, "kick", (0.35, 0.33, 0.31, 1)),
    )
    return b


def _add_container(spec, placement):
    """The round container (mocap: posed every frame): a translucent tub, a dark brim, food inside.
    Its origin is on the axis at the brim's top surface; a hidden cylinder is its clearance proxy.
    The proxy is solid out to the brim, so while the open gripper straddles the brim (the final
    approach and the close, PRECISION) its lower finger reads as up to ~4 cm inside it: that is the
    negative obstacle clearance in the check table, not the container sinking into the counter."""
    body = spec.worldbody.add_body(name="container", mocap=True, pos=placement.center)
    ring = _add_lathe_mesh(
        spec,
        "brim_mesh",
        [
            (BRIM_RADIUS_M - BRIM_WIDTH_M, -BRIM_THICK_M),
            (BRIM_RADIUS_M, -BRIM_THICK_M),
            (BRIM_RADIUS_M, 0.0),
            (BRIM_RADIUS_M - BRIM_WIDTH_M, 0.0),
        ],
    )
    h = CONTAINER_HEIGHT_M
    wall = _add_lathe_mesh(
        spec,
        "wall_mesh",
        [
            (WALL_BOTTOM_R - WALL_THICK, -h),
            (WALL_BOTTOM_R, -h),
            (WALL_TOP_R, -BRIM_THICK_M),
            (WALL_TOP_R - WALL_THICK, -BRIM_THICK_M),
        ],
    )
    tub = _material(spec, "tub", CONTAINER_RGBA, specular=0.8, shininess=0.9, reflectance=0.0)
    mesh = mujoco.mjtGeom.mjGEOM_MESH
    g = _geom(body, mesh, [0, 0, 0], material=_material(spec, "brim", BRIM_RGBA, 0.4, 0.5), name="brim")
    g.meshname = ring
    g = _geom(body, mesh, [0, 0, 0], material=tub, name="tub_wall")
    g.meshname = wall
    _geom(
        body,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [WALL_BOTTOM_R, WALL_THICK / 2, 0],
        pos=(0, 0, -h + WALL_THICK / 2),
        material=tub,
    )
    _geom(
        body,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [FOOD_R, FOOD_H / 2, 0],
        pos=(0, 0, -h + WALL_THICK + FOOD_H / 2),
        material=_material(spec, "food", FOOD_RGBA, 0.2, 0.3),
        name="food",
    )
    _geom(
        body,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [BRIM_RADIUS_M, h / 2, 0],
        pos=(0, 0, -h / 2),
        group=PROXY_GROUP,
        name="container_proxy",
    )
    return body


def _add_panel(spec, placement):
    """A small button panel on the counter (front face toward the arm). Returns (button, normal)."""
    yaw = np.radians(180.0 + PANEL_YAW_DEG)
    normal = np.array([np.cos(yaw), np.sin(yaw), 0.0])  # out of the front face
    hd, hw, hh = PANEL_HALF
    top = placement.table_top_z
    center = np.array([*PANEL_CENTER_XY, top + hh])
    body = spec.worldbody.add_body(name="panel", pos=center, quat=_yaw_quat(PANEL_YAW_DEG))
    _box(body, (0, 0, 0), PANEL_HALF, _material(spec, "panel", PANEL_RGBA, 0.2, 0.3), name="panel_box")
    button_z = BUTTON_HEIGHT_M - hh
    facing = _quat(Rotation.from_euler("y", 90, degrees=True))  # cylinder axis along the panel normal
    _geom(
        body,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [BUTTON_R + 0.005, 0.002, 0],
        pos=(-hd - 0.002, 0, button_z),
        quat=facing,
        material=_material(spec, "bezel", METAL_RGBA, 0.6, 0.6),
        name="button_bezel",
    )
    _geom(
        body,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        [BUTTON_R, BUTTON_PROUD_M / 2, 0],
        pos=(-hd - BUTTON_PROUD_M / 2, 0, button_z),
        quat=facing,
        material=_material(spec, "button", BUTTON_RGBA, 0.5, 0.6),
        name="button",
    )
    button = center + _yaw_rot(PANEL_YAW_DEG) @ np.array([-hd - BUTTON_PROUD_M, 0.0, button_z])
    return button, normal


def _yaw_rot(deg):
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


# ---------------------------------------------------------------------------------------- build
def _geoms_where(model, keep):
    """Geom ids whose (body name, geom group) satisfy keep."""
    return np.array(
        [g for g in range(model.ngeom) if keep(model.body(model.geom_bodyid[g]).name, model.geom_group[g])], dtype=int
    )


def _clearance_groups(model):
    """Geom ids for clearance queries: one entry per arm part (collision meshes; the gripper and
    the face count as head) and one per environment object."""

    def part(bodies, with_gripper):
        return lambda body, group: (
            group == PROXY_GROUP and (body in bodies or (with_gripper and body.startswith(GRIPPER_PREFIX)))
        )

    def on(name):
        return lambda body, group: body == name

    groups = {name: _geoms_where(model, part(bodies, name == "head")) for name, bodies in ARM_PART_BODIES.items()}
    groups["head"] = np.append(groups["head"], model.geom("oak_housing").id)
    groups.update(
        person=_geoms_where(model, on("person")),
        chair=_geoms_where(model, on("chair")),
        counter=_geoms_where(model, on("counter")),
        panel=_geoms_where(model, on("panel")),
        container=np.array([model.geom("container_proxy").id]),
    )
    return groups


def _carry_in_tool(placement, k, q_close):
    """The container's pose in the tool frame at closing: held rigidly from then on."""
    p_tool, R_tool = k.fk(q_close, "tool")
    pos = R_tool.T @ (placement.center - p_tool)
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (R_tool.T).ravel())
    return pos, quat


@lru_cache(maxsize=4)
def build(run_name=REFERENCE_RUN):
    """The full scene with the container where `run_name` closed the gripper (run_name 'flatbot':
    the empty flatbot chair instead, see flatbot_scene)."""
    if run_name == "flatbot":
        from . import flatbot_scene

        return flatbot_scene.build()
    placement = grasp_placement(run_name)
    spec = _robot_spec()
    _add_look(spec)
    _add_chair(spec)
    _add_person(spec)
    _add_counter(spec, placement)
    _add_container(spec, placement)
    button, normal = _add_panel(spec, placement)
    model = spec.compile()
    run = runs.load_run(run_name)
    k = kin.Kin.robot_only(model)
    return Scene(
        model=model,
        placement=placement,
        container_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        carry_in_tool=_carry_in_tool(placement, k, run.q[runs.closing_index(run)]),
        button_pos=button,
        button_normal=normal,
        linkage=gripper_linkage(),
        groups=_clearance_groups(model),
        person_head=np.array(HEAD_CENTER),
    )
