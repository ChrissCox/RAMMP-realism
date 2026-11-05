"""The gallery clips at rest and at the start of a task (curious_tilt, doze_wake, eager_start;
idle_alive is covered by test_clips): registered, they pass every check with nothing shrunk to fit,
each expression moves the face on screen in its own camera, and each does what it says."""

import copy

import numpy as np
import pytest

from puppy_sim import checks, layer, render, runs
from puppy_sim.clips import PairClip, curious_tilt, eager_start, registry
from puppy_sim.timeline import BOUNDARY, DT, PRECISION

GALLERY = ("curious_tilt", "doze_wake", "eager_start")
SHOWN_PX = 25.0  # the gallery's bar: an expression moves the eye or the mouth this far (1280x720)
HOME = np.array(runs.HOME_JOINTS)


@pytest.fixture(scope="module")
def mkin(sc):
    return sc.kin()


@pytest.fixture(scope="module")
def built():
    return {name: registry()[name].build() for name in GALLERY}


def _tracks(built_clip):
    return built_clip if isinstance(built_clip, tuple) else (built_clip,)


def _window(track, beat):
    """(N,) True over a beat's span."""
    return (track.t >= beat.t0) & (track.t < beat.t1)


def _angle(a, b):
    """Degrees between direction rows a (N, 3) and b (N, 3) or (3,)."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def _eye_px(kin, a, b, camera):
    """(N,) on-screen distance (1280x720) between the eye of tracks a and b, in a's camera."""
    cam = render.CAMERAS[camera]
    look = cam.lookats(a, kin)
    pa, pb = kin.fk_track(a.q, "eye")[0], kin.fk_track(b.q, "eye")[0]
    return np.linalg.norm(cam.project(pa, look, render.SINGLE_SIZE) - cam.project(pb, look, render.SINGLE_SIZE), axis=1)


def test_the_gallery_clips_are_registered():
    clips = registry()
    for name in GALLERY:
        assert clips[name].camera in render.CAMERAS
    assert isinstance(clips["eager_start"], PairClip)


@pytest.mark.parametrize("name", GALLERY)
def test_passes_every_check_with_nothing_shrunk(built, mkin, name):
    for track in _tracks(built[name]):
        failed = [(c.name, c.shown, c.detail) for c in checks.check_track(track, mkin) if not c.ok]
        assert not failed, (name, failed)
    log = _tracks(built[name])[-1].log
    assert not [line for line in log if "amplitude x" in line or "still over a cap" in line], log


@pytest.mark.parametrize("name", GALLERY)
def test_every_expression_shows_on_screen(built, mkin, name):
    """Each chip's beat moves the eye or the mouth at least SHOWN_PX in the clip's own camera; the
    recorded move in eager_start not at all."""
    rows = render.legibility(_tracks(built[name])[-1], mkin, registry()[name].camera)
    assert rows
    for r in rows:
        travel = max(r.eye_px, r.mouth_px)
        if r.beat == eager_start.MOVE_CHIP:
            assert travel == 0.0, r
        else:
            assert travel >= SHOWN_PX, r


def test_curious_tilt_looks_tilts_both_ways_and_settles(sc, mkin):
    rest, parts = curious_tilt.rest_and_parts(sc, mkin)
    full = layer.compose(rest, copy.deepcopy(parts), mkin)
    no_tilt = layer.compose(rest, [copy.deepcopy(p) for p in parts if p.name != "double tilt"], mkin)
    assert {b.name for b in full.beats} == {"CURIOUS"}
    curious = _window(full, full.beats[0])
    eye, R = mkin.fk_track(full.q[curious], "eye")
    assert _angle(R[:, :, 2], sc.placement.center - eye).min() < 12.0  # it looks at the container
    _, R_full = mkin.fk_track(full.q, "ee")
    _, R_ref = mkin.fk_track(no_tilt.q, "ee")
    up_ref, up = R_ref[:, :, 1], R_full[:, :, 1]
    roll = np.degrees(np.arcsin(np.clip(np.einsum("ij,ij->i", np.cross(up_ref, up), R_ref[:, :, 2]), -1.0, 1.0)))
    assert roll.max() > 10.0 and roll.min() < -10.0  # one side, then the other, about its gaze
    assert _eye_px(mkin, full, no_tilt, "close").max() >= SHOWN_PX
    assert np.abs(full.q[-1] - HOME).max() < 1e-3  # settled back home


def test_doze_wake_sleeps_low_then_wakes_stretches_and_nods(built, mkin, sc):
    track = built["doze_wake"]
    order = [b.name for b in sorted(track.beats, key=lambda b: b.t0)]
    assert order == ["SLEEPY", "TASK REQUESTED", "AWAKE", "EAGER (on it)"]
    beats = {b.name: b for b in track.beats}
    t_request = beats["TASK REQUESTED"].t0
    phase = next(p for p in track.phases if p.name == "task requested")
    assert phase.t0 == pytest.approx(t_request, abs=DT)  # the request shows on the timeline too
    eye0, R0 = mkin.fk(HOME, "eye")
    eye, R = mkin.fk(track.q[int(round(t_request / DT))], "eye")
    assert eye[2] < eye0[2] - 0.03  # asleep: curled low...
    assert _angle(R[:, 2], R0[:, 2]) > 15.0 and R[2, 2] < R0[2, 2]  # ...looking down
    eyes, R = mkin.fk_track(track.q[_window(track, beats["AWAKE"])], "eye")
    assert eyes[:, 2].max() > eye0[2] + 0.02 and R[:, 2, 2].max() > R0[2, 2] + 0.15  # the stretch: up, chin up
    eyes, R = mkin.fk_track(track.q[_window(track, beats["EAGER (on it)"])], "eye")
    to_user = _angle(R[:, :, 2], sc.person_head - eyes)
    assert to_user.min() < _angle(R0[:, 2], sc.person_head - eye0) - 8.0  # turned toward the user
    assert np.abs(track.q[-1] - HOME).max() < 1e-3  # back at rest, ready


def test_eager_start_plays_every_move_as_recorded(built, mkin, sc):
    plain, puppy = built["eager_start"]
    assert puppy.tier == BOUNDARY
    tier = next(c for c in checks.check_track(puppy, mkin) if c.name == "tier")
    assert tier.ok and tier.shown == "boundary"
    assert puppy.duration > plain.duration
    assert not np.any(plain.zone == PRECISION)  # the window keeps clear of the 10 ms locate phase
    order = [b.name for b in sorted(puppy.beats, key=lambda b: b.t0)]
    assert order == ["TASK REQUESTED", "EAGER", "WIND UP", eager_start.MOVE_CHIP, "FOLLOW THROUGH"]
    beats = {b.name: b for b in puppy.beats}
    move = _window(puppy, beats[eager_start.MOVE_CHIP])
    assert np.array_equal(puppy.q[move], puppy.func.q[move])  # the move itself: untouched
    eager = _window(puppy, beats["EAGER"])
    eyes, R = mkin.fk_track(puppy.q[eager], "eye")
    func_eyes, func_R = mkin.fk_track(puppy.func.q[eager], "eye")
    assert (eyes[:, 2] - func_eyes[:, 2]).max() > 0.015  # perked up
    to_user = _angle(R[:, :, 2], sc.person_head - eyes) - _angle(func_R[:, :, 2], sc.person_head - func_eyes)
    assert to_user.min() < -6.0  # turned toward the user
    line = registry()["eager_start"].subtitle_for((plain, puppy))[1]
    assert f"plain {plain.duration:.1f} s" in line and f"Puppy Mode {puppy.duration:.1f} s" in line
