"""The vocabulary: every behavior starts and ends at rest at zero offset, fits the caps at home by
default (and after compose's auto-scaling anywhere), is deterministic for a seed, and keeps its
own gating: no wiggle or clack while holding or near the user, no background life in PRECISION,
a level look while holding, a sniff that keeps its standoff."""

import copy

import numpy as np
import pytest

from puppy_sim import behaviors as B
from puppy_sim import checks, layer, runs
from puppy_sim import timeline as tl
from puppy_sim.timeline import DT, NEAR_USER, PRECISION

REST_GRIP = 0.79
T_ON = 1.0
REST_S = 14.0
END_VALUE_SHARE = 0.02  # a behavior's first and last moving samples are this small next to its peak
END_STEP_SHARE = 0.25  # and its second and second-to-last steps this slow next to its fastest
# (a min-jerk start gives ~0.5% and ~16% at 50 Hz even for a 0.25 s ramp; starting with a velocity
# would give ~8% and ~100%)


def _rest(home, kin, held=False, zone=None, seconds=REST_S):
    track = tl.constant_track(home, REST_GRIP, seconds)
    track.held[:] = held
    runs.mark_near_user(track, kin)
    if zone is not None:
        track.zone[:] = zone
    return track


def _move(home, kin):
    """Rest, a 2 s joint move to the side, rest: a transit for the motion-craft behaviors."""
    side = home + np.radians([-15.0, 6.0, -10.0, 0.0, -12.0, -2.0, 7.0])
    track = tl.concat(
        [
            tl.constant_track(home, REST_GRIP, 2.0),
            tl.joint_move(home, side, 2.0, REST_GRIP),
            tl.constant_track(side, REST_GRIP, 4.0),
        ]
    )
    runs.mark_near_user(track, kin)
    return track


AT_REST = {
    "breathe": lambda ctx: B.breathe(ctx, T_ON, T_ON + 6.0),
    "micro_life": lambda ctx: B.micro_life(ctx, T_ON, T_ON + 4.0),
    "gaze_to": lambda ctx: B.gaze_to(ctx, ctx.container, T_ON, T_ON + 1.0),
    "head_tilt": lambda ctx: B.head_tilt(ctx, T_ON, T_ON + 1.0),
    "perk": lambda ctx: B.perk(ctx, T_ON),
    "droop": lambda ctx: B.droop(ctx, T_ON, T_ON + 2.5),
    "puppy_eyes": lambda ctx: B.puppy_eyes(ctx, T_ON, T_ON + 1.0),
    "wiggle": lambda ctx: B.wiggle(ctx, T_ON),
    "shake_off": lambda ctx: B.shake_off(ctx, T_ON),
    "startle": lambda ctx: B.startle(ctx, T_ON),
    "sniff": lambda ctx: B.sniff(ctx, ctx.container, T_ON),
    "double_take": lambda ctx: B.double_take(ctx, ctx.container, ctx.person, T_ON),
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
    "lock_on": lambda ctx: B.lock_on(ctx, ctx.container, 4.2, 6.0),
}


@pytest.fixture(scope="module")
def at_rest(sc, home):
    k = sc.kin()
    ctx = B.Context.of(_rest(home, k), sc, k)
    return ctx, {name: make(ctx) for name, make in AT_REST.items()}


@pytest.fixture(scope="module")
def on_a_move(sc, home):
    k = sc.kin()
    ctx = B.Context.of(_move(home, k), sc, k)
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
    """compose() on a copy of c (it scales what it is given): (track, the copy, checks by name)."""
    c = copy.deepcopy(c)
    out = layer.compose(ctx.track, [c], ctx.kin)
    return out, c, {x.name: x for x in checks.check_track(out, ctx.kin)}


def _failed(by_name):
    return [(x.name, x.shown) for x in by_name.values() if not x.ok]


@pytest.mark.parametrize("name", sorted(AT_REST))
def test_defaults_fit_the_caps_at_home(at_rest, name):
    """Named defaults are within the caps: at home compose has nothing to shrink."""
    ctx, made = at_rest
    out, c, by_name = _composed(ctx, made[name])
    assert c.scale == 1.0, out.log
    assert not _failed(by_name)


@pytest.mark.parametrize("name", sorted(ON_A_MOVE))
def test_motion_craft_respects_the_caps(on_a_move, name):
    ctx, made = on_a_move
    assert not _failed(_composed(ctx, made[name])[2])


@pytest.mark.parametrize("name", ["proud", "doze_and_wake", "search", "droop", "double_take"])
def test_caps_hold_after_auto_scaling_at_the_view_pose(sc, functional, name):
    """At the task's view posture (the face turned on its side) the same calls still pass every
    check: whatever does not fit is shrunk by compose and logged."""
    k = sc.kin()
    view = functional.q[int(round(5.3 / DT))]
    ctx = B.Context.of(_rest(view, k), sc, k)
    assert not _failed(_composed(ctx, AT_REST[name](ctx))[2])


def test_a_too_big_wiggle_is_limited_to_the_caps_and_noted(at_rest):
    ctx, made = at_rest
    c = made["wiggle"]
    assert c.notes and "limited" in c.notes[0]  # 2 Hz: the 3 rad/s^2 cap allows only a little
    assert np.abs(c.head[:, 3]).max() < B.WIGGLE_MAX_ROLL_RAD


@pytest.mark.parametrize("make", [B.breathe, B.micro_life])
def test_seeded_noise_is_deterministic(sc, home, make):
    k = sc.kin()
    track = _rest(home, k)
    a = make(B.Context.of(track, sc, k, seed=5))
    b = make(B.Context.of(track, sc, k, seed=5))
    c = make(B.Context.of(track, sc, k, seed=6))
    assert np.array_equal(a.head, b.head)
    assert not np.array_equal(a.head, c.head)


@pytest.mark.parametrize("make", [B.breathe, B.micro_life])
def test_background_life_is_off_in_precision(sc, home, make):
    k = sc.kin()
    track = _rest(home, k)
    track.zone[300:400] = PRECISION
    c = make(B.Context.of(track, sc, k))
    assert not np.any(c.head[300:400])
    assert np.any(c.head[:250])  # and on elsewhere


def test_micro_life_stays_tiny(at_rest):
    c = at_rest[1]["micro_life"]
    assert np.linalg.norm(c.head[:, :3], axis=1).max() <= 0.002 + 1e-12
    assert np.linalg.norm(c.head[:, 3:], axis=1).max() <= np.radians(0.5) + 1e-12


@pytest.mark.parametrize("make", [B.wiggle, B.clack])
@pytest.mark.parametrize("where", ["holding", "near the user", "precision"])
def test_wiggle_and_clack_never_fire_while_holding_or_near_the_user(sc, home, make, where):
    k = sc.kin()
    track = _rest(
        home, k, held=where == "holding", zone={"near the user": NEAR_USER, "precision": PRECISION}.get(where)
    )
    c = make(B.Context.of(track, sc, k), T_ON)
    assert not np.any(_channels(c))


def test_clack_stops_before_the_gripper_takes_hold(sc, home):
    k = sc.kin()
    track = _rest(home, k)
    track.held[int(round(1.6 / DT)) :] = True  # the grip closes partway through the chomps
    c = B.clack(B.Context.of(track, sc, k), T_ON)
    assert np.any(c.grip) and not np.any(c.grip[track.held])


def test_clack_chomps_within_the_gripper_speed(at_rest):
    c = at_rest[1]["clack"]
    assert np.abs(c.grip).max() == pytest.approx(B.CLACK_SIZE, rel=1e-3)
    assert np.abs(np.gradient(c.grip, DT)).max() <= layer.EXPR_GRIP_SPEED * 1.02  # the 2F-85's 150 mm/s


def test_proud_is_muted_while_holding(sc, functional):
    """Holding the container: no clack, no wiggle, no tilt, a pan-only look: it stays level."""
    k = sc.kin()
    i = int(round(14.0 / DT))  # the run's 'done', holding
    track = _rest(functional.q[i], k, held=True)
    track.grip[:] = functional.grip[i]
    ctx = B.Context.of(track, sc, k)
    c = B.proud(ctx, T_ON, T_ON + 4.0)
    assert not np.any(c.grip)
    out = layer.compose(track, [c], k)
    _, R0 = k.fk(track.q[0], "tool")
    tilts = []
    for q in out.q[::10]:
        _, R = k.fk(q, "tool")
        tilts.append(np.degrees(np.arccos(np.clip((R @ (R0.T @ [0, 0, 1.0]))[2], -1, 1))))
    assert max(tilts) < 1.0  # the container's up axis stays vertical to within a degree


def test_a_nod_dips_toward_the_floor_even_with_the_face_on_its_side(sc, functional):
    """At the view posture the face's up is world -y: a head-frame pitch would swing the gaze
    sideways; the gravity nod lowers it."""
    k = sc.kin()
    q = functional.q[int(round(5.3 / DT))]
    track = _rest(q, k)
    ctx = B.Context.of(track, sc, k)
    c = B.nod(ctx, T_ON, dips=1)
    i = int(np.argmax(np.abs(c.head[:, 3:]).max(axis=1)))
    _, R0 = k.fk(q, "ee")
    _, R1 = k.fk(q + k.head_dq(q, c.head[i]), "ee")
    g0, g1 = R0[:, 2], R1[:, 2]
    assert g1[2] < g0[2] - np.sin(B.NOD_RAD) * 0.4  # lower
    az0, az1 = np.arctan2(g0[1], g0[0]), np.arctan2(g1[1], g1[0])
    assert abs(np.degrees(az1 - az0)) < 1.0  # not sideways


def test_sniff_keeps_its_standoff(at_rest):
    ctx, made = at_rest
    out = _composed(ctx, made["sniff"])[0]
    tool = ctx.kin.fk_track(out.q, "tool")[0]
    assert np.linalg.norm(tool - ctx.container, axis=1).min() >= B.SNIFF_STANDOFF_M


def test_lock_on_is_still_before_the_motion_resumes(on_a_move):
    ctx, made = on_a_move
    c = made["lock_on"]
    resume = int(round(6.0 / DT))
    still = int(round(B.LOCK_STILL_S / DT))
    assert np.any(c.head) and not np.any(c.head[resume - still :])


def test_lead_gaze_looks_ahead(on_a_move):
    """Mid-transit the composed gaze is closer to where the plain gaze will be LEAD_S later."""
    ctx, made = on_a_move
    out = _composed(ctx, made["lead_gaze"])[0]
    i, ahead = int(round(3.0 / DT)), int(round((3.0 + B.LEAD_S) / DT))
    g_func, g_puppy, g_ahead = (ctx.kin.fk(q, "ee")[1][:, 2] for q in (ctx.track.q[i], out.q[i], ctx.track.q[ahead]))
    assert g_puppy @ g_ahead > g_func @ g_ahead


def test_wind_up_draws_back_against_the_move(on_a_move):
    ctx, made = on_a_move
    i_back = int(round(1.7 / DT))  # drawn back, before the move starts at 2.0 s
    q_back = ctx.track.q[i_back] + ctx.kin.head_dq(ctx.track.q[i_back], made["wind_up"].head[i_back])
    eye_rest, R_rest = ctx.kin.fk(ctx.track.q[i_back], "eye")
    _, R_back = ctx.kin.fk(q_back, "eye")
    _, R_later = ctx.kin.fk(ctx.track.q[int(round(3.0 / DT))], "eye")
    g_rest, g_back, g_later = R_rest[:, 2], R_back[:, 2], R_later[:, 2]
    assert g_back @ g_later < g_rest @ g_later  # the gaze turned away from where it is going
