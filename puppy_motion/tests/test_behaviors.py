"""The vocabulary: every behavior starts and ends at rest at zero offset, fits the caps at home by
default (and after compose's auto-scaling anywhere), is deterministic for a seed, and keeps its
own gating: no wiggle or clack while holding or near the user, no background life in PRECISION,
a level look while holding, a sniff that keeps its standoff, motion craft off the move.
Ported from the spike's test_behaviors.py (the MuJoCo Kin and the checks/render modules are replaced
by Chain + Scene and compose.cap_violations)."""

import copy

import numpy as np
import pytest

from puppy_motion import behaviors as B
from puppy_motion import compose as C
from puppy_motion.behaviors import Context, Targets
from puppy_motion.compose import head_dq
from puppy_motion.geometry import PARTS
from puppy_motion.track import DT, Zone, concat, constant_track, joint_move, joint_speed

FREE, NEAR_USER, PRECISION = Zone.FREE, Zone.NEAR_USER, Zone.PRECISION
REST_GRIP = 0.79
T_ON = 1.0
REST_S = 14.0
STOP_RAD_S = 0.02  # the arm is stopped when every joint is slower than this (between moves)
NEAR_USER_M = 0.35
END_VALUE_SHARE = 0.02  # a behavior's first and last moving samples are this small next to its peak
END_STEP_SHARE = 0.25  # and its second and second-to-last steps this slow next to its fastest
# (a min-jerk start gives ~0.5% and ~16% at 50 Hz even for a 0.25 s ramp; starting with a velocity
# would give ~8% and ~100%)
# The spike scene's container brim centre and button (spike/puppy_sim/scene.py), world points.
CONTAINER = np.array([0.881, -0.178, 0.263])
BUTTON = np.array([0.823, 0.230, 0.268])


def _targets(scene):
    return Targets(scene.person_head(), container=CONTAINER, button=BUTTON)


def _mark_near_user(track, chain, scene):
    """NEAR_USER where a FREE sample brings the head within NEAR_USER_M of the person."""
    head, memo = PARTS.index("head"), {}  # a rest repeats one posture: look each one up once

    def clearance(q, h):
        key = (q.tobytes(), bool(h))
        if key not in memo:
            memo[key] = C.clearances_of(chain, scene, q, h)[0, head]
        return memo[key]

    person = np.array([clearance(q, h) for q, h in zip(track.q, track.held)])
    track.zone[(track.zone == FREE) & (person < NEAR_USER_M)] = NEAR_USER


def _rest(home, chain, scene, held=False, zone=None, seconds=REST_S):
    track = constant_track(home, REST_GRIP, seconds)
    track.held[:] = held
    _mark_near_user(track, chain, scene)
    if zone is not None:
        track.zone[:] = zone
    return track


def _move(home, chain, scene):
    """Rest, a 2 s joint move to the side, rest: a transit for the motion-craft behaviors."""
    side = home + np.radians([-15.0, 6.0, -10.0, 0.0, -12.0, -2.0, 7.0])
    track = concat(
        [
            constant_track(home, REST_GRIP, 2.0),
            joint_move(home, side, 2.0, REST_GRIP),
            constant_track(side, REST_GRIP, 4.0),
        ]
    )
    _mark_near_user(track, chain, scene)
    return track


def _ctx(track, chain, scene, **kw):
    return Context.of(track, chain, scene, _targets(scene), **kw)


AT_REST = {
    "breathe": lambda ctx: B.breathe(ctx, T_ON, T_ON + 6.0),
    "micro_life": lambda ctx: B.micro_life(ctx, T_ON, T_ON + 4.0),
    "gaze_to": lambda ctx: B.gaze_to(ctx, ctx.targets.container, T_ON, T_ON + 1.0),
    "head_tilt": lambda ctx: B.head_tilt(ctx, T_ON, T_ON + 1.0),
    "perk": lambda ctx: B.perk(ctx, T_ON),
    "droop": lambda ctx: B.droop(ctx, T_ON, T_ON + 2.5),
    "puppy_eyes": lambda ctx: B.puppy_eyes(ctx, T_ON, T_ON + 1.0),
    "wiggle": lambda ctx: B.wiggle(ctx, T_ON),
    "shake_off": lambda ctx: B.shake_off(ctx, T_ON),
    "startle": lambda ctx: B.startle(ctx, T_ON),
    "sniff": lambda ctx: B.sniff(ctx, ctx.targets.container, T_ON),
    "double_take": lambda ctx: B.double_take(ctx, ctx.targets.container, ctx.person, T_ON),
    "nod": lambda ctx: B.nod(ctx, T_ON),
    "bounce": lambda ctx: B.bounce(ctx, T_ON),
    "clack": lambda ctx: B.clack(ctx, T_ON),
    "doze_and_wake": lambda ctx: B.doze_and_wake(ctx, T_ON, 3.0),
    "proud": lambda ctx: B.proud(ctx, T_ON, T_ON + 6.0),
    "search": lambda ctx: B.search(ctx, T_ON),
    "look_at_user": lambda ctx: B.look_at_user(ctx, T_ON, T_ON + 1.0),
}
ON_A_MOVE = {  # the transit runs from 2.0 to 4.0 s
    "wind_up": lambda ctx: B.wind_up(ctx, 1.0, 2.0),
    "lead_gaze": lambda ctx: B.lead_gaze(ctx, 1.2, 4.0),
    "follow_through": lambda ctx: B.follow_through(ctx, 4.0),
    "lock_on": lambda ctx: B.lock_on(ctx, ctx.targets.container, 4.2, 6.0),
}


@pytest.fixture(scope="module")
def at_rest(chain, scene, home):
    ctx = _ctx(_rest(home, chain, scene), chain, scene)
    return ctx, {name: make(ctx) for name, make in AT_REST.items()}


@pytest.fixture(scope="module")
def on_a_move(chain, scene, home):
    ctx = _ctx(_move(home, chain, scene), chain, scene)
    return ctx, {name: make(ctx) for name, make in ON_A_MOVE.items()}


def _channels(c):
    return np.column_stack([c.head, c.joints, c.grip])


def _assert_rests_at_both_ends(c):
    """Exactly zero before and after, and it eases in and out (zero velocity at both ends)."""
    x = _channels(c)
    moving = np.nonzero(np.any(x != 0.0, axis=1))[0]
    assert moving.size, f"{c.name} does nothing"
    i0, i1 = moving[0], moving[-1]
    assert i0 > 0 and i1 < len(x) - 1, f"{c.name} is not inside the track"
    assert not np.any(x[:i0]) and not np.any(x[i1 + 1 :])
    size = np.abs(x).max(axis=1)
    step = np.abs(np.diff(x, axis=0)).max(axis=1)
    assert max(size[i0], size[i1]) <= END_VALUE_SHARE * size.max(), f"{c.name} jumps in or out"
    assert max(step[i0], step[i1 - 1]) <= END_STEP_SHARE * step.max(), f"{c.name} starts or stops moving"


@pytest.mark.parametrize("name", sorted(AT_REST))
def test_behavior_starts_and_ends_at_rest(at_rest, name):
    _assert_rests_at_both_ends(at_rest[1][name])


@pytest.mark.parametrize("name", sorted(ON_A_MOVE))
def test_motion_craft_starts_and_ends_at_rest(on_a_move, name):
    _assert_rests_at_both_ends(on_a_move[1][name])


def _composed(ctx, c):
    """compose() on a copy of c (it scales what it is given): (track, the copy, the cap
    violations of the composite, which are none by construction)."""
    c = copy.deepcopy(c)
    out = C.compose(ctx.track, [c], ctx.chain, ctx.scene, ctx.caps)
    bad = C.cap_violations(ctx.chain, ctx.scene, out.func, out.q, out.grip, out.caps)
    return out, c, bad


def _failed(bad):
    return [(v.kind, v.detail) for v in bad]


@pytest.mark.parametrize("name", sorted(AT_REST))
def test_defaults_fit_the_caps_at_home(at_rest, name):
    """Named defaults are within the caps: at home compose has nothing to shrink."""
    ctx, made = at_rest
    out, c, bad = _composed(ctx, made[name])
    # doze_and_wake: the last nodding-off dip catches just as the wake's jolt starts; the strict
    # per-sample acceleration check (the spike used smoothed gradients) trims the pair by 6%.
    assert c.scale >= (0.9 if name == "doze_and_wake" else 1.0), out.log
    assert not _failed(bad)


@pytest.mark.parametrize("name", sorted(ON_A_MOVE))
def test_motion_craft_respects_the_caps(on_a_move, name):
    ctx, made = on_a_move
    assert not _failed(_composed(ctx, made[name])[2])


@pytest.mark.parametrize("name", ["proud", "doze_and_wake", "search", "droop", "double_take"])
def test_caps_hold_after_auto_scaling_at_the_view_pose(chain, scene, functional, name):
    """At the task's view posture (the face turned on its side) the same calls still pass every
    check: whatever does not fit is shrunk by compose and logged."""
    view = functional.q[int(round(5.3 / DT))]
    ctx = _ctx(_rest(view, chain, scene), chain, scene)
    assert not _failed(_composed(ctx, AT_REST[name](ctx))[2])


def test_the_wiggle_sways_visibly_within_the_caps(chain, scene, home, at_rest):
    """At home the sway swings away from the user and back; every joint stays within its own
    caps; the eye travels well over a centimetre (the spike's 15 px in the close camera); snappy's
    wrist wags further, everything else the same."""
    ctx, made = at_rest
    c = made["wiggle"]
    assert c.notes and "sway" in c.notes[0] and "caps 'default'" in c.notes[0]
    out = _composed(ctx, c)[0]
    eye_func = np.array([chain.fk(q, "eye")[0] for q in out.func.q])
    eye_puppy = np.array([chain.fk(q, "eye")[0] for q in out.q])
    assert np.linalg.norm(eye_puppy - eye_func, axis=1).max() > 0.010
    snappy = B.wiggle(_ctx(ctx.track, chain, scene, caps="snappy"), T_ON)
    wrist = slice(4, 7)
    assert np.abs(snappy.joints[:, wrist]).max() > 2.0 * np.abs(c.joints[:, wrist]).max()
    assert np.allclose(snappy.joints[:, :4], c.joints[:, :4])


def test_motion_craft_stays_off_the_move(on_a_move):
    """Boundary beats: the wind-up is back at rest when the move starts, the follow-through starts
    once the arm is still, the lock-on stays in its window."""
    ctx, made = on_a_move
    moving = joint_speed(ctx.track) >= STOP_RAD_S
    for name in ("wind_up", "follow_through", "lock_on"):
        c = made[name]
        assert not np.any(c.active() & moving), name
    assert np.any(made["lead_gaze"].active() & moving)  # the in-motion one rides on it


@pytest.mark.parametrize("make", [B.breathe, B.micro_life])
def test_seeded_noise_is_deterministic(chain, scene, home, make):
    track = _rest(home, chain, scene)
    a = make(_ctx(track, chain, scene, seed=5))
    b = make(_ctx(track, chain, scene, seed=5))
    c = make(_ctx(track, chain, scene, seed=6))
    assert np.array_equal(a.head, b.head)
    assert not np.array_equal(a.head, c.head)


@pytest.mark.parametrize("make", [B.breathe, B.micro_life])
def test_background_life_is_off_in_precision(chain, scene, home, make):
    track = _rest(home, chain, scene)
    track.zone[300:400] = PRECISION
    c = make(_ctx(track, chain, scene))
    assert not np.any(c.head[300:400])
    assert np.any(c.head[:250])  # and on elsewhere


def test_micro_life_stays_tiny(at_rest):
    c = at_rest[1]["micro_life"]
    assert np.linalg.norm(c.head[:, :3], axis=1).max() <= 0.002 + 1e-12
    assert np.linalg.norm(c.head[:, 3:], axis=1).max() <= np.radians(0.5) + 1e-12


@pytest.mark.parametrize("make", [B.wiggle, B.clack])
@pytest.mark.parametrize("where", ["holding", "near the user", "precision"])
def test_wiggle_and_clack_never_fire_while_holding_or_near_the_user(chain, scene, home, make, where):
    track = _rest(
        home,
        chain,
        scene,
        held=where == "holding",
        zone={"near the user": NEAR_USER, "precision": PRECISION}.get(where),
    )
    c = make(_ctx(track, chain, scene), T_ON)
    assert not np.any(_channels(c))


def test_clack_stops_before_the_gripper_takes_hold(chain, scene, home):
    track = _rest(home, chain, scene)
    track.held[int(round(1.6 / DT)) :] = True  # the grip closes partway through the chomps
    c = B.clack(_ctx(track, chain, scene), T_ON)
    assert np.any(c.grip) and not np.any(c.grip[track.held])


def test_clack_chomps_within_the_gripper_speed(at_rest):
    c = at_rest[1]["clack"]
    assert np.abs(c.grip).max() == pytest.approx(B.CLACK_SIZE, rel=1e-3)
    assert np.abs(np.gradient(c.grip, DT)).max() <= C.EXPR_GRIP_SPEED * 1.02  # the 2F-85's 150 mm/s
    assert not np.any(c.head) and not np.any(c.joints)  # clack writes only grip


def test_proud_is_muted_while_holding(chain, scene, functional):
    """Holding the container: no clack, no wiggle, no tilt, a pan-only look: it stays level."""
    i = int(round(14.0 / DT))  # the run's 'done', holding
    track = _rest(functional.q[i], chain, scene, held=True)
    track.grip[:] = functional.grip[i]
    ctx = _ctx(track, chain, scene)
    c = B.proud(ctx, T_ON, T_ON + 4.0)
    assert not np.any(c.grip)
    out = C.compose(track, [c], chain, scene)
    _, R0 = chain.fk(track.q[0], "tool")
    tilts = []
    for q in out.q[::10]:
        _, R = chain.fk(q, "tool")
        tilts.append(np.degrees(np.arccos(np.clip((R @ (R0.T @ [0, 0, 1.0]))[2], -1, 1))))
    assert max(tilts) < 1.0  # the container's up axis stays vertical to within a degree


def test_a_nod_dips_toward_the_floor_even_with_the_face_on_its_side(chain, scene, functional):
    """At the view posture the face's up is world -y: a head-frame pitch would swing the gaze
    sideways; the gravity nod lowers it."""
    q = functional.q[int(round(5.3 / DT))]
    ctx = _ctx(_rest(q, chain, scene), chain, scene)
    c = B.nod(ctx, T_ON, dips=1)
    i = int(np.argmax(np.abs(c.head[:, 3:]).max(axis=1)))
    _, R0 = chain.fk(q, "ee")
    _, R1 = chain.fk(q + head_dq(chain, q, c.head[i]), "ee")
    g0, g1 = R0[:, 2], R1[:, 2]
    assert g1[2] < g0[2] - np.sin(B.NOD_RAD) * 0.4  # lower
    az0, az1 = np.arctan2(g0[1], g0[0]), np.arctan2(g1[1], g1[0])
    assert abs(np.degrees(az1 - az0)) < 1.0  # not sideways


def test_sniff_keeps_its_standoff(chain, at_rest):
    ctx, made = at_rest
    out = _composed(ctx, made["sniff"])[0]
    tool = np.array([chain.fk(q, "tool")[0] for q in out.q])
    assert np.linalg.norm(tool - ctx.targets.container, axis=1).min() >= B.SNIFF_STANDOFF_M


def test_lock_on_is_still_before_the_motion_resumes(on_a_move):
    ctx, made = on_a_move
    c = made["lock_on"]
    resume = int(round(6.0 / DT))
    still = int(round(B.LOCK_STILL_S / DT))
    assert np.any(c.head) and not np.any(c.head[resume - still :])


def test_lead_gaze_looks_ahead(chain, on_a_move):
    """Mid-transit the composed gaze is closer to where the plain gaze will be LEAD_S later."""
    ctx, made = on_a_move
    out = _composed(ctx, made["lead_gaze"])[0]
    i, ahead = int(round(3.0 / DT)), int(round((3.0 + B.LEAD_S) / DT))
    g_func, g_puppy, g_ahead = (chain.fk(q, "ee")[1][:, 2] for q in (ctx.track.q[i], out.q[i], ctx.track.q[ahead]))
    assert g_puppy @ g_ahead > g_func @ g_ahead


def test_wind_up_draws_back_against_the_move(chain, on_a_move):
    ctx, made = on_a_move
    i_back = int(round(1.6 / DT))  # drawn back, before the move starts at 2.0 s
    q_back = ctx.track.q[i_back] + head_dq(chain, ctx.track.q[i_back], made["wind_up"].head[i_back])
    _, R_rest = chain.fk(ctx.track.q[i_back], "eye")
    _, R_back = chain.fk(q_back, "eye")
    _, R_later = chain.fk(ctx.track.q[int(round(3.0 / DT))], "eye")
    g_rest, g_back, g_later = R_rest[:, 2], R_back[:, 2], R_later[:, 2]
    assert g_back @ g_later < g_rest @ g_later  # the gaze turned away from where it is going


# ------------------------------------------------------------------- new in the port
def test_targets_from_scene_sets_only_the_person_head(scene):
    t = Targets.from_scene(scene)
    assert np.array_equal(t.person_head, scene.person_head())
    assert t.container is None and t.button is None


def test_with_no_object_target_sniff_and_lock_on_aim_along_the_gaze(chain, scene, home):
    """On the robot Puppy Mode has no perception: with container None the aim is 0.35 m along the gaze."""
    rest = _rest(home, chain, scene)
    ctx = Context.of(rest, chain, scene, Targets.from_scene(scene))
    eye, R = chain.fk(home, "eye")
    assert np.allclose(ctx.aim_point(None, 0.0), eye + 0.35 * R[:, 2])
    assert np.array_equal(ctx.aim_point(CONTAINER, 0.0), CONTAINER)
    sniff = B.sniff(ctx, None, T_ON)
    assert np.any(sniff.head)
    tool = chain.fk(home, "tool")[0]
    u = sniff.head[np.argmax(np.abs(sniff.head[:, :3]).max(axis=1)), :3]
    assert u @ (eye + 0.35 * R[:, 2] - tool) > 0.0  # leans toward the aimed point
    still = _move(home, chain, scene)
    ctx_move = Context.of(still, chain, scene, Targets.from_scene(scene))
    assert np.any(B.lock_on(ctx_move, None, 4.2, 6.0).head)
    assert np.any(B.search(ctx, T_ON).head)  # and a search with nothing to look around
