"""Composition: PRECISION is untouched, gains ramp outside zones, caps shrink behaviors (tiered by
zone, gentler near the user), the 'snappy' preset only when asked, and near the user nothing moves
toward them."""

import copy

import numpy as np

from puppy_sim import checks, curves, layer
from puppy_sim import timeline as tl
from puppy_sim.kin import ARM_PARTS
from puppy_sim.layer import Contribution
from puppy_sim.timeline import DT, FREE, NEAR_USER, PRECISION


def _big_offset(n):
    c = Contribution.empty("big", n)
    c.head[:] = [0.02, -0.02, 0.03, 0.4, 0.3, -0.5]  # 2-3 cm and ~20-30 deg, everywhere
    c.joints[:] = 0.2
    return c


def test_precision_samples_are_bit_identical_under_a_large_offset(functional, kin):
    track = tl.resample_track(functional, np.arange(260, 560, dtype=float))  # detect .. close
    out = layer.compose(track, [_big_offset(track.n)], kin)
    prec = out.zone == PRECISION
    assert prec.any() and (~prec).any()
    assert np.all(out.gain[prec] == 0.0)
    assert out.q[prec].tobytes() == track.q[prec].tobytes()
    assert out.func.q.tobytes() == track.q.tobytes()  # the functional motion itself is never changed
    assert np.abs(out.q[~prec] - track.q[~prec]).max() > 1e-3  # elsewhere the offset does show


def test_gain_ramps_finish_before_a_lower_zone():
    zone = np.full(200, FREE)
    zone[80:120] = PRECISION
    zone[150:170] = NEAR_USER
    g = layer.zone_gain(zone)
    assert np.all(g[79:121] == 0.0)  # already 0 one sample before the zone
    assert np.all(g[150:170] <= layer.GAINS[NEAR_USER] + 1e-12)
    ramp = int(round(layer.GAIN_RAMP_S / DT))
    assert g[79 - ramp - 1] == 1.0 and 0.0 < g[79 - ramp // 2] < 1.0
    assert np.all(np.diff(g[50:80]) <= 1e-12)  # a smooth monotone ramp down
    assert np.abs(np.diff(g)).max() < 0.15


def test_a_too_fast_behavior_scales_itself_down(home, kin):
    rest = tl.constant_track(home, 0.5, 3.0)
    wiggle = Contribution.empty("wiggle", rest.n)
    wiggle.head[:, 5] = np.radians(25.0) * np.sin(2 * np.pi * 2.0 * rest.t) * np.sin(np.pi * rest.t / 3.0)
    out = layer.compose(rest, [wiggle], kin)
    assert wiggle.scale < 0.5
    assert any(line.startswith("wiggle: amplitude x") for line in out.log)
    by_name = {c.name: c for c in checks.check_track(out, kin)}
    for name in ("free speed", "free accel", "free head", "user clr", "obstacle clr"):
        assert by_name[name].ok, by_name[name]


def test_near_the_user_nothing_moves_toward_them(home, kin, sc):
    rest = tl.constant_track(home, 0.5, 2.0, zone=NEAR_USER)
    toward = Contribution.empty("toward", rest.n)
    direction = sc.person_head - kin.fk(home, "ee")[0]
    toward.head[:, :3] = 0.05 * direction / np.linalg.norm(direction) * np.sin(np.pi * rest.t / 2.0)[:, None]
    out = layer.compose(rest, [toward], kin)
    func = kin.clearance_track(rest)[:, 0]
    comp = kin.clearance_track(out)[:, 0]
    assert np.all(out.gain <= layer.GAINS[NEAR_USER] + 1e-12)
    assert np.all(comp >= func - 1e-3)  # no arm part closer to the person (first order)
    head = ARM_PARTS.index("head")
    assert comp[:, head].max() >= func[:, head].max()


def test_gripper_offsets_stop_while_holding(functional, kin):
    track = tl.resample_track(functional, np.arange(560, 700, dtype=float))  # close .. done
    snap = Contribution.empty("snap", track.n)
    snap.grip[:] = -0.3
    out = layer.compose(track, [snap], kin)
    assert out.held.any()
    assert out.grip[out.held].tobytes() == track.grip[out.held].tobytes()


def test_caps_follow_the_most_restrictive_zone_across_a_ramp():
    """Wherever the gain ramps toward NEAR_USER or PRECISION, the gentle caps already apply."""
    zone = np.full(300, FREE)
    zone[100:140] = NEAR_USER
    zone[220:240] = PRECISION
    limits = layer.sample_caps(zone)
    gain = layer.zone_gain(zone)
    ramping = (gain < 1.0) & (zone == FREE)
    assert ramping.any() and np.all(limits.tier[ramping] != FREE)
    assert np.all(limits.head_speed[ramping] == layer.NEAR_HEAD_SPEED)
    assert np.all(limits.tier[zone == NEAR_USER] == NEAR_USER) and np.all(limits.tier[:60] == FREE)
    free, gentle = layer.CAP_PRESETS["default"][FREE], layer.CAP_PRESETS["default"][NEAR_USER]
    assert np.array_equal(limits.joint_speed[0], free.joint_speed)
    assert np.array_equal(limits.joint_accel[120], gentle.joint_accel)
    assert gentle.head_speed < free.head_speed and max(gentle.joint_accel) < max(free.joint_accel)


def _wave(n, seconds, amplitude):
    t = np.arange(n) * DT
    return amplitude * np.sin(np.pi * t / seconds) ** 2


def test_a_held_offset_carried_into_near_user_moves_under_the_gentle_caps(home, kin):
    """A roll held while the user comes near: the gain ramp alone moves it, already under the
    gentle caps (0.5 rad/s, 1.5 rad/s^2 per joint), which compose shrinks it to and check reports
    in its 'near' columns."""
    rest = tl.constant_track(home, 0.5, 3.0)
    rest.zone[int(round(1.5 / DT)) :] = NEAR_USER
    roll = Contribution.empty("roll", rest.n)
    roll.joints[:, 6] = np.radians(20.0) * curves.min_jerk(rest.t / 1.0)  # eased in, then held
    out = layer.compose(rest, [roll], kin)
    limits = layer.sample_caps(out.zone)
    gentle = limits.tier != FREE
    v = np.abs(np.gradient(out.q - rest.q, DT, axis=0))
    a = np.abs(np.gradient(np.gradient(out.q - rest.q, DT, axis=0), DT, axis=0))
    assert v[gentle].max() > 0.05  # the ramp moves it
    assert np.all(v[gentle] <= limits.joint_speed[gentle] + 1e-9)
    assert np.all(a[gentle] <= limits.joint_accel[gentle] + 1e-9)
    by_name = {c.name: c for c in checks.check_track(out, kin)}
    assert all(c.ok for c in by_name.values()) and by_name["near accel"].shown != "-"


def test_snappy_is_only_used_when_asked_and_only_loosens_the_wrist_in_free(home, kin):
    """'snappy' differs from 'default' only in FREE's wrist acceleration; compose and check use the
    default unless told, and a snappy track is checked against snappy's caps."""
    default, snappy = layer.CAP_PRESETS["default"], layer.CAP_PRESETS["snappy"]
    assert layer.DEFAULT_CAPS == "default"
    assert snappy[NEAR_USER] == default[NEAR_USER] and snappy[PRECISION] == default[PRECISION]
    assert snappy[FREE].joint_speed == default[FREE].joint_speed
    assert snappy[FREE].joint_accel[:4] == default[FREE].joint_accel[:4]
    assert min(snappy[FREE].joint_accel[4:]) == layer.SNAPPY_WRIST_ACCEL > max(default[FREE].joint_accel)
    rest = tl.constant_track(home, 0.5, 2.0)
    wag = Contribution.empty("wag", rest.n)
    wag.joints[:, 6] = _wave(rest.n, 0.8, np.radians(4.0)) * np.sin(2 * np.pi * 1.5 * rest.t)
    quick = layer.compose(rest, [copy.deepcopy(wag)], kin, caps="snappy")
    assert quick.caps == "snappy" and not [c for c in checks.check_track(quick, kin) if not c.ok]
    assert layer.compose(rest, [wag], kin).caps == "default" and wag.scale < 1.0
    quick.caps = "default"  # the same motion held to the default caps fails
    assert "free accel" in {c.name for c in checks.check_track(quick, kin) if not c.ok}
