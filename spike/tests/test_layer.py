"""Composition: PRECISION is untouched, gains ramp outside zones, caps shrink behaviors, and near
the user nothing moves toward them."""

import numpy as np

from puppy_sim import checks, layer
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
    for name in ("expr speed", "expr accel", "head speed", "user clr", "obstacle clr"):
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
