"""An approximate dynamic model of the arm under the driver's two execution modes, to see how much
compliance softens Puppy Mode's beats. Offline only: MuJoCo forward dynamics on the menagerie
Gen3, nothing here talks to a robot.

Elsewhere the spike is kinematic: it writes the commanded joints into the model, so every video
shows the command tracked perfectly. Here a composed Track is the reference and the arm follows it
under one of the modes RAMMP's arm API offers (control_mode=POSITION | IMPEDANCE, gains a profile
name or a custom {kq, zeta, torque_limit}). A Puppy Mode beat runs in the task's own mode and gains.

IMPEDANCE ports the driver's JointImpedanceMode::compute (kinova-gen3-driver 2.0.0,
src/joint_impedance_mode.cpp), run at its 1 kHz:

    q_d   <- the trajectory sample, rate-limited to max_ref_speed (the URDF velocity limits)
    D_i    = 2 zeta sqrt(Kq_i M_ii(q))                    damping DERIVED from the model's M
    tau    = g(q) + ramp * (Kq * clamp(q_d - q, +-leash) - D * (qd - qd_ref) + M(q) qdd_ref)
    tau    = clamp(tau, +-torque_limit)

with qd_ref, qdd_ref the planner's profile (clamped to max_ref_speed) when the trajectory carries
one, as cuRobo's plans do; the executor samples it by quintic Hermite (interface::sample_target).
`feedforward=False` is a positions-only trajectory: linear interpolation, qd_ref = qdd_ref = 0.
There is no Coriolis, friction or payload term in the driver's law; its g and M come from its URDF
(the arm with the 2F-85 as a rigid payload). Here the controller's model is the plant's own, minus
the joint armature (a URDF has none) and minus a held container (the driver does not know about it).

POSITION is Kortex's own actuator position servo (firmware, not in the driver), fed the same
reference through JointPositionMode's rate limit (seeded from the URDF limits by the ROS bringup)
and following-error leash. Its servo is modelled as exact: achieved = the rate-limited command,
which for every Puppy Mode track is the kinematic playback the rest of the spike shows.

The plant: the menagerie Gen3 (its inertials) with the scene's 2F-85 welded open (about 1.05 kg)
and the OAK face (0.09 kg), gravity on, contacts off, 0.5 ms steps under a 1 kHz zero-order-hold
torque. Its joint armature, 0.1 kg m^2 on every joint, is what the menagerie added to stabilise
its position actuators (its CHANGELOG, 2026-09-13), not an identified value: on the real arm the
actuator's torque loop on its output torque sensor hides some of the rotor inertia. `Plant` takes
it as a parameter (armature=0: hidden entirely). The menagerie gives the arm joints no damping or
friction; the real arm has friction the driver does not compensate (`Plant.friction_nm`, off by
default). A held container (PAYLOAD_KG, an assumption) rides at the scene's carry pose, and the
counter top is a contact for its base only (nothing else touches anything): while it stands on the
counter the counter carries it, and its weight passes to the arm's springs as the arm lifts it.
"""

import copy
from dataclasses import dataclass, field, replace
from functools import lru_cache

import mujoco
import numpy as np

from . import render, scene
from .kin import VEL_LIMITS
from .timeline import DT, PRECISION, Track

POSITION, IMPEDANCE = "position", "impedance"
VELOCITIES = "velocities"  # Control.feedforward: positions + velocities, no accelerations
CONTROL_HZ = 1000.0  # RtExecutor rate_hz and the Supervisor's sampler_hz
CONTROL_DT = 1.0 / CONTROL_HZ
STEPS_PER_SAMPLE = int(round(DT / CONTROL_DT))  # control cycles per 50 Hz track sample
PHYSICS_SUBSTEPS = 2  # plant steps per control cycle (0.5 ms)
WARMUP_S = 2.0  # the arm settles on the first sample (and the gain ramp runs) before t = 0
GAIN_RAMP_S = 0.1  # JointImpedanceParams::gain_ramp_s: the spring fades in on mode entry
URDF_EFFORT_NM = (39.0,) * 4 + (9.0,) * 3  # JointImpedanceParams torque_limit default = URDF effort
KQ_RANGE, ZETA_RANGE = (1.0, 400.0), (0.05, 2.0)  # interface/gains.h validation bounds
TORQUE_FLOOR_NM = (18.8, 37.6, 14.6, 14.5, 3.7, 3.7, 0.5)  # gains.cpp kTorqueLimitFloor (gravity headroom)

MENAGERIE_ARMATURE = 0.1  # kg m^2, every arm joint (gen3.xml default class)
PAYLOAD_KG = 0.4  # ASSUMED: the container with some food in it
PAYLOAD_COM_M = (0.0, 0.0, -0.06)  # in the container frame (origin at the brim top, z up): food sits low
PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M = 0.07, 0.09  # its inertia: a solid cylinder this size
PAYLOAD_BODY = "payload"
PAYLOAD_FOOT, COUNTER_GEOM = "payload_foot", "counter_top"  # the one contact pair: the container's base on the counter
FOOT_HALF_M = 0.004  # half thickness of the base disc
CONTACT_BIT = 2
CONTACT_SOLREF = (0.004, 1.0)  # a firm counter: time constant 4 ms, critically damped
CONTACT_FRICTION = 0.4  # a tub on a kitchen counter
POINT_SITES = ("eye", "tool")  # tracking error is measured at these head points


# ---------------------------------------------------------------------------------- gains
@dataclass(frozen=True)
class Gains:
    """The gain fields of JointImpedanceParams: Kq (N m/rad), the damping ratio, the per-joint
    torque limit (N m) and the spring leash max_tracking_error (rad)."""

    name: str
    kq: tuple
    zeta: float
    torque_limit: tuple = URDF_EFFORT_NM
    leash: float = 0.35

    @classmethod
    def custom(cls, kq, zeta, torque_limit=URDF_EFFORT_NM):
        """Custom gains as RAMMP's arm API takes them ({kq: 7, zeta, torque_limit: 7}), refused
        outside the driver's validate_custom bounds (it refuses, never clamps). The leash stays
        the session default's (a custom spec does not carry one)."""
        kq, limit = tuple(float(v) for v in kq), tuple(float(v) for v in torque_limit)
        if len(kq) != 7 or len(limit) != 7:
            raise ValueError("custom gains need 7 kq and 7 torque_limit values")
        if not ZETA_RANGE[0] <= zeta <= ZETA_RANGE[1]:
            raise ValueError(f"zeta {zeta} outside {ZETA_RANGE}")
        if not all(KQ_RANGE[0] <= k <= KQ_RANGE[1] for k in kq):
            raise ValueError(f"kq {kq} outside {KQ_RANGE}")
        if not all(f <= v <= c for v, f, c in zip(limit, TORQUE_FLOOR_NM, URDF_EFFORT_NM)):
            raise ValueError(f"torque_limit {limit} outside the gravity floor {TORQUE_FLOOR_NM} .. {URDF_EFFORT_NM}")
        return cls("custom", kq, float(zeta), limit, PROFILES[SESSION_DEFAULT].leash)

    def describe(self):
        return f"Kq {self.kq[0]:g}/{self.kq[-1]:g} N m/rad, zeta {self.zeta:g}"


# interface/gains.cpp profile_params; the session default starts as medium (supervisor.h)
PROFILES = {
    "soft": Gains("soft", (40.0,) * 4 + (15.0,) * 3, 0.4, leash=0.45),
    "medium": Gains("medium", (80.0,) * 4 + (30.0,) * 3, 0.5, leash=0.35),
    "stiff": Gains("stiff", (160.0,) * 4 + (60.0,) * 3, 0.7, leash=0.25),
}
SESSION_DEFAULT = "medium"
PROFILES["default"] = replace(PROFILES[SESSION_DEFAULT], name="default")


@dataclass(frozen=True)
class Control:
    """How the driver executes a track: POSITION, or IMPEDANCE with gains. feedforward: the
    trajectory carries velocities and accelerations (as cuRobo's plans do); False: positions only;
    VELOCITIES: positions and velocities, no accelerations (what the playground's
    execute_joint_trajectory sends: central-difference dq, zero at both ends)."""

    mode: str = POSITION
    gains: Gains | None = None
    feedforward: bool = True

    @property
    def label(self):
        if self.mode == POSITION:
            return POSITION
        return f"{IMPEDANCE}:{self.gains.name}" + _ff_suffix(self.feedforward, " (positions only)", " (positions+velocities)")

    def describe(self):
        """Plain words for a subtitle."""
        if self.mode == POSITION:
            return "position mode (stiff)"
        g = self.gains
        alias = f" = {SESSION_DEFAULT}" if g.name == "default" else ""
        return f"impedance '{g.name}'{alias} ({g.describe()})"


def _ff_suffix(feedforward, none, velocities):
    return velocities if feedforward == VELOCITIES else ("" if feedforward else none)


def control(spec, feedforward=True):
    """A Control from 'position', 'impedance' (the session default), 'impedance:<profile>', a
    Gains, or a custom dict {kq, zeta, torque_limit}."""
    if isinstance(spec, Control):
        return spec
    if isinstance(spec, Gains):
        return Control(IMPEDANCE, spec, feedforward)
    if isinstance(spec, dict):
        return Control(IMPEDANCE, Gains.custom(**spec), feedforward)
    mode, _, profile = str(spec).partition(":")
    if mode == POSITION and not profile:
        return Control(POSITION)
    if mode == IMPEDANCE:
        name = profile or "default"
        if name not in PROFILES:
            raise ValueError(f"unknown gains profile {name!r}; known: {sorted(PROFILES)}")
        return Control(IMPEDANCE, PROFILES[name], feedforward)
    raise ValueError(f"unknown control {spec!r}: 'position' or 'impedance[:{'|'.join(PROFILES)}]'")


@dataclass(frozen=True)
class Plant:
    """The simulated arm. armature: kg m^2 per joint (one value or 7; the menagerie's by
    default, 0 = hidden by the actuators' torque loops); payload_kg: a held container's mass;
    friction_nm: Coulomb friction per joint (one value or 7; the menagerie has none)."""

    armature: float | tuple = MENAGERIE_ARMATURE
    payload_kg: float = PAYLOAD_KG
    friction_nm: float | tuple = 0.0

    def describe(self):
        arm = np.broadcast_to(np.asarray(self.armature, dtype=float), (7,))
        fric = np.broadcast_to(np.asarray(self.friction_nm, dtype=float), (7,))
        parts = [f"armature {arm[0]:g} kg m^2" if np.all(arm == arm[0]) else f"armature {arm.tolist()}"]
        if np.any(fric):
            parts.append(f"friction {fric[0]:g} N m" if np.all(fric == fric[0]) else f"friction {fric.tolist()}")
        parts.append(f"container {self.payload_kg:g} kg")
        return ", ".join(parts)


DEFAULT_PLANT = Plant()


def slug(spec, plant=None):
    """A file-name tag for a control (and a non-default plant): 'impedance_default', ..."""
    ctl = control(spec)
    out = (
        ctl.mode
        if ctl.mode == POSITION
        else f"{IMPEDANCE}_{ctl.gains.name}" + _ff_suffix(ctl.feedforward, "_posonly", "_posvel")
    )
    plant = plant or DEFAULT_PLANT
    if plant.armature != DEFAULT_PLANT.armature:
        out += f"_arm{np.max(plant.armature):g}"
    if np.any(np.asarray(plant.friction_nm)):
        out += f"_fric{np.max(plant.friction_nm):g}"
    if plant.payload_kg != DEFAULT_PLANT.payload_kg:
        out += f"_load{plant.payload_kg:g}"
    return out


# ----------------------------------------------------------------------------------- models
@lru_cache(maxsize=1)
def _base_model():
    """The robot alone with the gripper welded open, no actuators, a payload body on the end
    effector (posed and weighed per simulation) and the counter top (placed per simulation); the
    only contact is the payload's base on the counter, switched on while it is held."""
    spec = scene._robot_spec()
    for group in (spec.equalities, spec.tendons, spec.actuators, spec.excludes, spec.keys):
        for el in list(group):
            spec.delete(el)
    for j in list(spec.joints):
        if j.name.startswith(scene.GRIPPER_PREFIX):
            spec.delete(j)
    for g in spec.geoms:
        g.contype, g.conaffinity = 0, 0
    spec.option.timestep = CONTROL_DT / PHYSICS_SUBSTEPS
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    body = spec.body("ee").add_body(name=PAYLOAD_BODY)
    body.mass, body.inertia, body.ipos, body.explicitinertial = 1.0, [1.0, 1.0, 1.0], PAYLOAD_COM_M, True
    foot = body.add_geom(
        name=PAYLOAD_FOOT,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[scene.WALL_BOTTOM_R, FOOT_HALF_M, 0.0],
        pos=(0.0, 0.0, -scene.CONTAINER_HEIGHT_M + FOOT_HALF_M),
        contype=CONTACT_BIT,  # compiled on: MuJoCo filters contacts by body_contype, fixed at compile time
        conaffinity=0,
        group=scene.PROXY_GROUP,
    )
    (x0, x1), (y0, y1) = scene.COUNTER_X, scene.COUNTER_Y
    top = spec.worldbody.add_geom(
        name=COUNTER_GEOM,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[(x1 - x0) / 2.0, (y1 - y0) / 2.0, scene.COUNTER_SLAB_M / 2.0],
        pos=((x0 + x1) / 2.0, (y0 + y1) / 2.0, 0.0),
        contype=0,
        conaffinity=CONTACT_BIT,
        group=scene.PROXY_GROUP,
    )
    for g in (foot, top):
        g.solref, g.friction = CONTACT_SOLREF, [CONTACT_FRICTION, 0.005, 0.0001]
    model = spec.compile()
    assert model.nq == model.nv == 7, "the dynamics model is the 7 arm joints only"
    return model


def _payload_inertia(mass):
    """Diagonal inertia (kg m^2) of a solid cylinder of PAYLOAD_RADIUS_M x PAYLOAD_HEIGHT_M about its COM."""
    r, h = PAYLOAD_RADIUS_M, PAYLOAD_HEIGHT_M
    side = mass * (3.0 * r * r + h * h) / 12.0
    return np.array([side, side, mass * r * r / 2.0])


def models(sc, plant=DEFAULT_PLANT):
    """(plant model, controller model) for scene `sc`: the controller's has no armature, no
    friction and no payload (the driver's URDF), the plant's carries a held container at sc's carry
    pose (its mass and its contact with the counter set per step by `hold`)."""
    base = _base_model()
    pid = base.body(PAYLOAD_BODY).id
    pos_in_tool, quat_in_tool = sc.carry_in_tool
    plant_m, ctrl_m = copy.copy(base), copy.copy(base)
    for m in (plant_m, ctrl_m):
        m.body_pos[pid] = np.array([0.0, 0.0, scene.TOOL_Z_M]) + pos_in_tool  # tool site = ee + TOOL_Z along z
        m.body_quat[pid] = quat_in_tool
        m.body_mass[pid], m.body_inertia[pid] = 0.0, 0.0
        m.geom_pos[m.geom(COUNTER_GEOM).id, 2] = sc.placement.table_top_z - scene.COUNTER_SLAB_M / 2.0
        m.geom_contype[m.geom(PAYLOAD_FOOT).id] = 0  # hold() switches it on while the container is held
    plant_m.dof_armature[:] = np.broadcast_to(np.asarray(plant.armature, dtype=float), (7,))
    plant_m.dof_frictionloss[:] = np.broadcast_to(np.asarray(plant.friction_nm, dtype=float), (7,))
    ctrl_m.dof_armature[:] = 0.0
    ctrl_m.dof_frictionloss[:] = 0.0
    return plant_m, ctrl_m


def hold(m, plant, held):
    """The plant's payload body: the container (its mass, inertia and its base's contact with the
    counter) while held, nothing otherwise."""
    pid = m.body(PAYLOAD_BODY).id
    mass = plant.payload_kg if held else 0.0
    m.body_mass[pid] = mass
    m.body_inertia[pid] = _payload_inertia(mass)
    m.geom_contype[m.geom(PAYLOAD_FOOT).id] = CONTACT_BIT if held else 0


def controller_terms(ctrl_m, ctrl_d, q):
    """(M (7, 7), g (7,)) of the controller's model at q: what the driver's Dynamics returns."""
    ctrl_d.qpos[:] = q
    mujoco.mj_kinematics(ctrl_m, ctrl_d)
    mujoco.mj_comPos(ctrl_m, ctrl_d)
    mujoco.mj_crb(ctrl_m, ctrl_d)
    M = np.empty((7, 7))
    mujoco.mj_fullM(ctrl_m, ctrl_d, M)
    g = np.empty(7)
    mujoco.mj_rne(ctrl_m, ctrl_d, 0, g)  # qvel stays 0 here: the bias is gravity alone
    return M, g


# ------------------------------------------------------------------------------- reference
def reference(track, feedforward=True):
    """The trajectory sampled at the control rate, as interface::sample_target does from the
    track's 50 Hz waypoints: (q, qd, qdd), each (n_ctrl, 7). With feedforward the waypoints carry
    finite-difference velocities and accelerations and are joined by quintic Hermite; with
    VELOCITIES the waypoints carry central-difference velocities only (zero at both ends, as
    beat.to_points) and are joined by cubic Hermite, qdd = 0; without, by straight lines
    (qd = qdd = 0: no profile)."""
    q = np.asarray(track.q, dtype=float)
    n_ctrl = (len(q) - 1) * STEPS_PER_SAMPLE + 1
    seg = np.minimum(np.arange(n_ctrl) // STEPS_PER_SAMPLE, len(q) - 2)
    u = (np.arange(n_ctrl) - seg * STEPS_PER_SAMPLE)[:, None] / STEPS_PER_SAMPLE
    a, b = q[seg], q[seg + 1]
    if not feedforward:
        qr = a + u * (b - a)
        qr[-1] = q[-1]
        return qr, np.zeros_like(a), np.zeros_like(a)
    if feedforward == VELOCITIES:
        qd_w = np.zeros_like(q)
        qd_w[1:-1] = (q[2:] - q[:-2]) / (2.0 * DT)
        v0, v1 = qd_w[seg] * DT, qd_w[seg + 1] * DT
        d = b - a
        c2, c3 = 3.0 * d - 2.0 * v0 - v1, -2.0 * d + v0 + v1
        qr = a + v0 * u + c2 * u**2 + c3 * u**3
        qdr = (v0 + 2.0 * c2 * u + 3.0 * c3 * u**2) / DT
        qr[-1], qdr[-1] = q[-1], qd_w[-1]
        return qr, qdr, np.zeros_like(a)
    qd_w = np.gradient(q, DT, axis=0)
    qdd_w = np.gradient(qd_w, DT, axis=0)
    span = DT
    v0, v1 = qd_w[seg] * span, qd_w[seg + 1] * span
    a0, a1 = qdd_w[seg] * span * span, qdd_w[seg + 1] * span * span
    d = b - a
    c3 = 10.0 * d - 6.0 * v0 - 4.0 * v1 - 1.5 * a0 + 0.5 * a1
    c4 = -15.0 * d + 8.0 * v0 + 7.0 * v1 + 1.5 * a0 - a1
    c5 = 6.0 * d - 3.0 * v0 - 3.0 * v1 - 0.5 * a0 + 0.5 * a1
    qr = a + v0 * u + 0.5 * a0 * u**2 + c3 * u**3 + c4 * u**4 + c5 * u**5
    qdr = (v0 + a0 * u + 3.0 * c3 * u**2 + 4.0 * c4 * u**3 + 5.0 * c5 * u**4) / span
    qddr = (a0 + 6.0 * c3 * u + 12.0 * c4 * u**2 + 20.0 * c5 * u**3) / span**2
    qr[-1] = q[-1]  # the end of the last segment (u = 1) exactly, as sample_target pins it
    return qr, qdr, qddr


# ------------------------------------------------------------------------------ simulation
def simulate(track, ctl, sc=None, plant=DEFAULT_PLANT):
    """(n_ctrl, 7) achieved joints at the control rate following `track` under Control `ctl`."""
    ctl = control(ctl)
    q_ref, qd_ref, qdd_ref = reference(track, ctl.feedforward)
    if ctl.mode == POSITION:
        return _position_mode(q_ref)
    sc = sc or scene.build()
    return _impedance_mode(q_ref, qd_ref, qdd_ref, np.asarray(track.held, dtype=bool), ctl, sc, plant)


def _position_mode(q_ref):
    """JointPositionMode's reference pipeline (rate limit at the URDF speeds; the following-error
    leash never binds on an exact servo), then Kortex's servo, taken as exact."""
    out, q_d = np.empty_like(q_ref), q_ref[0].copy()
    for i, target in enumerate(q_ref):
        q_d = _rate_limited(q_d, target)
        out[i] = q_d
    return out


def _rate_limited(q_d, target, max_step=VEL_LIMITS * CONTROL_DT):
    """The reference one cycle on toward target, at most max_step per joint (the target itself
    when it is within reach, so an unbound limit reproduces the command exactly)."""
    step = target - q_d
    return np.where(np.abs(step) <= max_step, target, q_d + np.clip(step, -max_step, max_step))


def _impedance_mode(q_ref, qd_ref, qdd_ref, held, ctl, sc, plant):
    """JointImpedanceMode::compute at CONTROL_HZ against the plant (see the module docstring)."""
    g_ = ctl.gains
    kq, limit = np.asarray(g_.kq), np.asarray(g_.torque_limit)
    two_zeta, leash = 2.0 * g_.zeta, g_.leash
    max_qd = VEL_LIMITS
    plant_m, ctrl_m = models(sc, plant)
    d, cd = mujoco.MjData(plant_m), mujoco.MjData(ctrl_m)
    d.qpos[:] = q_ref[0]
    mujoco.mj_forward(plant_m, d)
    n_warm = int(round(WARMUP_S * CONTROL_HZ))
    n = len(q_ref)
    out = np.empty((n, 7))
    q_d = q_ref[0].copy()
    holding, elapsed = None, 0.0
    zero = np.zeros(7)
    for step in range(-n_warm, n):
        i = max(step, 0)
        h = bool(held[min(i // STEPS_PER_SAMPLE, len(held) - 1)])
        if h != holding:
            hold(plant_m, plant, h)
            holding = h
        q, qd = d.qpos.copy(), d.qvel.copy()
        if step >= 0:
            out[step] = q
        target, v_ff, a_ff = (q_ref[i], qd_ref[i], qdd_ref[i]) if step >= 0 else (q_ref[0], zero, zero)
        q_d = _rate_limited(q_d, target)
        M, grav = controller_terms(ctrl_m, cd, q)
        damping = two_zeta * np.sqrt(np.maximum(0.0, kq * np.diag(M)))
        tau = kq * np.clip(q_d - q, -leash, leash) - damping * (qd - np.clip(v_ff, -max_qd, max_qd))
        if ctl.feedforward:
            tau = tau + M @ a_ff
        ramp = min(1.0, elapsed / GAIN_RAMP_S)
        elapsed += CONTROL_DT
        d.qfrc_applied[:] = np.clip(grav + ramp * tau, -limit, limit)
        for _ in range(PHYSICS_SUBSTEPS):
            mujoco.mj_step(plant_m, d)
        if not np.all(np.isfinite(d.qpos)):
            raise FloatingPointError(f"the simulation diverged at control step {step} under {ctl.label}")
    return out


# ------------------------------------------------------------------------------ achieved track
def achieved(track, ctl, sc=None, plant=DEFAULT_PLANT):
    """The achieved Track: `track`'s timeline (zones, gripper, labels) with the joints the arm
    reaches under `ctl`, sampled at the track's own samples. It keeps the command as `commanded`
    and, for a composed track, its functional track achieved the same way as `func` (so a beat's
    legibility compares what the arm does with and without it). `control` names the mode."""
    ctl = control(ctl)
    sc = sc or scene.build()
    q = simulate(track, ctl, sc, plant)[::STEPS_PER_SAMPLE]
    func = achieved(track.func, ctl, sc, plant) if track.func is not None else None
    return replace(
        track,
        q=q,
        func=func,
        commanded=track,
        control=ctl.label,
        log=[*track.log, f"achieved under {ctl.label} ({plant.describe()})"],
    )


# ----------------------------------------------------------------------------------- metrics
SETTLED_MM = 0.5  # the puppy and plain arms agree within this once a beat's motion has settled
STILL_PX = 1.0  # a beat that moves the face less than this is a label (FOCUS), not a motion


@dataclass(frozen=True)
class BeatResponse:
    """One beat, commanded vs achieved: its on-screen travel (max of eye and mouth, px, the
    render.legibility measure: close camera, 1280x720) and the largest tracking error of the
    head points (eye, tool) in it (mm)."""

    beat: str
    t0: float
    commanded_px: float
    achieved_px: float
    error_mm: float

    @property
    def ratio(self):
        return self.achieved_px / self.commanded_px if self.commanded_px >= STILL_PX else float("nan")


@dataclass(frozen=True)
class PrecisionDeviation:
    """The tool point on PRECISION samples: its deviation (mm) from the commanded path for the
    plain task and for the Puppy Mode one, and on the same commanded precise samples (matched by
    functional time) the largest gap between where the two arms put it (`added_mm`: what a beat
    carries into the zone) and how long after a zone starts that gap stays over SETTLED_MM."""

    plain_max_mm: float
    plain_rms_mm: float
    puppy_max_mm: float
    puppy_rms_mm: float
    added_mm: float
    settle_s: float
    samples: int


@dataclass
class Response:
    """What `ctl` does to a composed track: per beat, the peak tracking error over the track
    (joint rad, head points mm), and the PRECISION comparison when the track has a plain task."""

    control: str
    beats: list
    peak_rad: float
    peak_mm: float
    precision: PrecisionDeviation | None
    achieved: Track = field(repr=False, default=None)


def point_error_mm(kin, q_cmd, q_ach, sites=POINT_SITES):
    """(N,) largest distance (mm) between commanded and achieved positions of the head points."""
    worst = np.zeros(len(q_cmd))
    for s in sites:
        worst = np.maximum(worst, np.linalg.norm(kin.fk_track(q_cmd, s)[0] - kin.fk_track(q_ach, s)[0], axis=1))
    return 1000.0 * worst


def _tool_gap_mm(kin, qa, qb):
    return 1000.0 * np.linalg.norm(kin.fk_track(qa, "tool")[0] - kin.fk_track(qb, "tool")[0], axis=1)


def precision_deviation(plain_cmd, plain_ach, puppy_cmd, puppy_ach, kin):
    """PrecisionDeviation of the tool point. Each precise Puppy Mode sample is matched to the plain
    sample with the same functional time tau and the same commanded joints (the zone only retimes)."""

    def deviation(cmd, ach):
        mask = cmd.zone == PRECISION
        return _tool_gap_mm(kin, cmd.q[mask], ach.q[mask]), mask

    plain_err, _ = deviation(plain_cmd, plain_ach)
    puppy_err, mask = deviation(puppy_cmd, puppy_ach)
    idx = np.flatnonzero(mask)
    j = np.clip(np.searchsorted(plain_cmd.tau, puppy_cmd.tau[idx] - 1e-9), 0, plain_cmd.n - 1)
    same = np.abs(plain_cmd.tau[j] - puppy_cmd.tau[idx]) < 1e-6
    same &= np.all(np.abs(plain_cmd.q[j] - puppy_cmd.q[idx]) < 1e-9, axis=1)
    idx, j = idx[same], j[same]
    gap = _tool_gap_mm(kin, puppy_ach.q[idx], plain_ach.q[j])
    settle = 0.0
    for seg in np.split(np.arange(len(idx)), np.flatnonzero(np.diff(idx) > 1) + 1):
        over = np.flatnonzero(gap[seg] >= SETTLED_MM)
        if over.size:
            settle = max(settle, (over[-1] + 1) * DT)

    def stats(err):
        return (float(err.max()), float(np.sqrt(np.mean(err**2)))) if err.size else (0.0, 0.0)

    added = float(gap.max()) if gap.size else 0.0
    return PrecisionDeviation(*stats(plain_err), *stats(puppy_err), added, settle, len(idx))


def response(track, ctl, sc=None, plant=DEFAULT_PLANT, kin=None, plain=None):
    """Response of composed `track` under `ctl`. plain: the plain task for the PRECISION
    comparison (default: the functional track it warped, track.func.source, if it has one)."""
    ctl = control(ctl)
    sc = sc or scene.build()
    kin = kin or sc.kin()
    ach = achieved(track, ctl, sc, plant)
    err = point_error_mm(kin, track.q, ach.q)
    beats = []
    for c, a in zip(render.legibility(track, kin), render.legibility(ach, kin)):
        i0, i1 = int(round(c.t0 / DT)), min(int(round(c.t1 / DT)), track.n - 1)
        beats.append(
            BeatResponse(
                c.beat, c.t0, max(c.eye_px, c.mouth_px), max(a.eye_px, a.mouth_px), float(err[i0 : i1 + 1].max())
            )
        )
    plain = plain if plain is not None else (track.func.source if track.func is not None else None)
    prec = None
    if plain is not None and np.any(track.zone == PRECISION):
        prec = precision_deviation(plain, achieved(plain, ctl, sc, plant), track, ach, kin)
    peak_rad = float(np.abs(ach.q - track.q).max())
    return Response(ctl.label, beats, peak_rad, float(err.max()), prec, ach)


def modes(q, gains, plant=DEFAULT_PLANT, sc=None):
    """Per joint at posture q: (natural frequency Hz, effective damping ratio) of the decoupled
    spring-damper the driver tunes, against the plant's inertia M_ii + armature: the driver sets
    zeta for M_ii alone, so armature it does not model lowers the effective ratio."""
    sc = sc or scene.build()
    plant_m, ctrl_m = models(sc, plant)
    M, _ = controller_terms(ctrl_m, mujoco.MjData(ctrl_m), q)
    m_ctrl = np.diag(M)
    m_plant = m_ctrl + plant_m.dof_armature
    kq = np.asarray(gains.kq)
    return np.sqrt(kq / m_plant) / (2.0 * np.pi), gains.zeta * np.sqrt(m_ctrl / m_plant)


COMPARE = ("position", "impedance:soft", "impedance:default", "impedance:stiff")  # `compliance`'s default


def compliance_named(name, controls=COMPARE, plant=DEFAULT_PLANT, feedforward=True):
    """Worker: the registered clip `name`'s composed tracks (their commands, for an achieved
    clip) under each control; [(track label, Response)] without the achieved tracks."""
    from .clips import registry

    clip = registry()[name]
    sc = scene.build(clip.run)
    kin = sc.kin()
    built = clip.build()
    tracks = built if isinstance(built, tuple) else (built,)
    commands, seen = [], set()
    for t in tracks:
        cmd = t.commanded if t.commanded is not None else t
        if cmd.func is not None and id(cmd) not in seen:
            seen.add(id(cmd))
            commands.append(cmd)
    rows = []
    for k, cmd in enumerate(commands):
        label = name if len(commands) == 1 else f"{name}[{k}]"
        for spec in controls:
            r = response(cmd, control(spec, feedforward), sc, plant, kin)
            r.achieved = None
            rows.append((label, r))
    return rows


def format_responses(rows):
    """A fixed-width table of (track label, Response) rows: per beat commanded vs achieved px and
    the tracking error, the whole track's peak error, and the PRECISION comparison."""
    w = max([len(r.control) for _, r in rows] + [7])
    tw = max([len(label) for label, _ in rows] + [5])
    head = f"{'track':{tw}s} {'control':{w}s} {'beat':22s} {'t0 s':>5s} {'cmd px':>7s} {'ach px':>7s} {'ratio':>6s}"
    lines = [f"{head} {'err mm':>7s}"]
    for label, r in rows:
        for b in r.beats:
            ratio = f"{b.ratio:6.2f}" if np.isfinite(b.ratio) else f"{'-':>6s}"
            lines.append(
                f"{label:{tw}s} {r.control:{w}s} {b.beat:22s} {b.t0:5.1f} {b.commanded_px:7.1f} {b.achieved_px:7.1f} "
                f"{ratio} {b.error_mm:7.1f}"
            )
        lines.append(
            f"{label:{tw}s} {r.control:{w}s} {'whole track':22s} {'':5s} {'':7s} {'':7s} {'':6s} {r.peak_mm:7.1f}"
        )
        if r.precision is not None:
            p = r.precision
            lines.append(
                f"{label:{tw}s} {r.control:{w}s} PRECISION, tool off its command: plain up to {p.plain_max_mm:.1f} mm "
                f"(rms {p.plain_rms_mm:.1f}), Puppy Mode up to {p.puppy_max_mm:.1f} mm (rms {p.puppy_rms_mm:.1f}); "
                f"Puppy Mode vs plain on the same {p.samples} commands: up to {p.added_mm:.2f} mm, "
                f"under {SETTLED_MM} mm after {p.settle_s:.2f} s"
            )
    return "\n".join(lines)


# ------------------------------------------------------------------------- overshoot report
FEEDFORWARDS = (("pos+vel+acc", True), ("pos+vel", VELOCITIES), ("pos only", False))


def _overshoot_tracks(sc, kin):
    """{name: composed Puppy Mode track} for the three beats the overshoot is reported on: the
    happy wag and the bounce at home, and lock_on_view's lean and stare."""
    from . import behaviors as B
    from . import layer, runs
    from .clips.lock_on_view import directed
    from .clips.wag_compare import END_REST_S, REST_GRIP, WAG_AT_S, wag_track
    from .timeline import constant_track

    caps = layer.DEFAULT_CAPS
    rest = constant_track(runs.HOME_JOINTS, REST_GRIP, WAG_AT_S[0] + 3 * B.BOUNCE_HOP_S + END_REST_S + 1.0, name="rest at home")
    runs.mark_near_user(rest, kin)
    ctx = B.Context.of(rest, sc, kin, caps=caps)
    hop = B.bounce(ctx, WAG_AT_S[0], beat="BOUNCE")
    return {
        "wag": wag_track(caps, sc, kin),
        "lock_on": directed("view")[1],
        "bounce": layer.compose(rest, [hop], kin, caps),
    }


def overshoot(track, ctl, sc, kin, plant=DEFAULT_PLANT):
    """How far the arm overshoots the beat's command under `ctl`: over the beats' own windows, the
    largest |achieved - commanded| of any joint (rad), the largest head-point (eye, tool) error (mm),
    and the over-travel: the most any joint's achieved excursion from its pose at the beat's start
    exceeds the commanded excursion on the same side, as a fraction of the commanded excursion
    (joints that move under 0.01 rad are left out)."""
    ach = achieved(track, ctl, sc, plant)
    on = np.zeros(track.n, dtype=bool)
    for b in render.legibility(track, kin):
        on[int(round(b.t0 / DT)) : min(int(round(b.t1 / DT)), track.n - 1) + 1] = True
    err = np.abs(ach.q - track.q)[on]
    ref = track.q[on] - track.q[on][0]
    got = ach.q[on] - track.q[on][0]
    over = 0.0
    for sign in (1.0, -1.0):
        peak = (sign * ref).max(axis=0)
        moved = peak > 0.01
        if moved.any():
            over = max(over, float(((sign * got).max(axis=0)[moved] - peak[moved]).max() / peak[moved].max()))
    mm = point_error_mm(kin, track.q, ach.q)[on].max()
    return {"err_rad": float(err.max()), "err_mm": float(mm), "over_travel": over}


def overshoot_report(profiles=("soft", "default", "stiff"), plant=DEFAULT_PLANT):
    """[(beat, feedforward label, profile, overshoot dict)] for the wag, lock_on and the bounce under
    each trajectory format the driver can be sent: positions+velocities+accelerations, positions and
    central-difference velocities (what execute_joint_trajectory sends), positions only."""
    sc = scene.build()
    kin = sc.kin()
    rows = []
    for name, track in _overshoot_tracks(sc, kin).items():
        for label, ff in FEEDFORWARDS:
            for profile in profiles:
                rows.append((name, label, profile, overshoot(track, control(f"impedance:{profile}", ff), sc, kin, plant)))
    return rows


def format_overshoot(rows):
    lines = [f"{'beat':8s} {'trajectory':12s} {'gains':8s} {'max err rad':>11s} {'max err mm':>10s} {'over-travel':>11s}"]
    for name, label, profile, o in rows:
        lines.append(f"{name:8s} {label:12s} {profile:8s} {o['err_rad']:11.4f} {o['err_mm']:10.1f} {o['over_travel']:11.1%}")
    return "\n".join(lines)
