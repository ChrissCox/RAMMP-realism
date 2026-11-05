"""The reference clips: registered, they pass every safety check, idle_alive does what it says,
the reels' primitives read on screen, the pair clip shows both durations, and the cameras are the
ones the legibility measure assumes."""

import numpy as np
import pytest

from puppy_sim import checks, render, runs
from puppy_sim.clips import PairClip, registry
from puppy_sim.timeline import DT, PRECISION

REFERENCE_CLIPS = ("replay_plain", "replay_pair", "idle_alive", "reel_at_rest", "reel_reactions", "reel_craft")
SUBTLE_BY_DESIGN = {"MICRO LIFE"}


def test_registry_has_the_reference_clips():
    clips = registry()
    for name in REFERENCE_CLIPS:
        assert name in clips and clips[name].camera in render.CAMERAS
    assert "vocab_reel" not in clips and "replay_puppy" not in clips
    assert isinstance(clips["replay_pair"], PairClip)


@pytest.mark.parametrize("name", REFERENCE_CLIPS)
def test_reference_clip_passes_every_check(name):
    for clip, label, _, results, _ in checks.check_named(name):
        failed = [(c.name, c.shown, c.detail) for c in results if not c.ok]
        assert not failed, (clip, label, failed)


def test_idle_alive_glances_tilts_and_looks(kin, sc):
    track = registry()["idle_alive"].build()
    eye0, R0 = kin.fk(track.q[0], "eye")
    beats = {b.name: b for b in track.beats}

    def at(t):
        return kin.fk(track.q[int(round(t / DT))], "eye")

    def angle(a, b):
        return np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0)))

    _, R = at(beats["CURIOUS"].t0)  # glancing at the user: turned toward them
    to_user = sc.person_head - eye0
    assert angle(R[:, 2], R0[:, 2]) > 12.0
    assert R[:, 2] @ to_user > R0[:, 2] @ to_user
    _, R = at((beats["CURIOUS"].t0 + beats["CURIOUS"].t1) / 2)  # the curious tilt rolls the head
    assert angle(R[:, 1], R0[:, 1]) > 12.0
    eye, R = at(beats["INTERESTED"].t1 - 0.8)  # looking at the container
    d = sc.placement.center - eye
    assert angle(R[:, 2], d / np.linalg.norm(d)) < 10.0
    assert np.abs(track.q[-1] - np.array(runs.HOME_JOINTS)).max() < 1e-3  # settled back home
    assert set(beats) == {"HELLO", "CURIOUS", "INTERESTED"}


@pytest.mark.parametrize("name", ["reel_at_rest", "reel_reactions"])
def test_every_primitive_moves_the_face_on_screen(kin, name):
    """In the close camera at 1280x720 each primitive moves the eye or the mouth at least
    LEGIBLE_PX (micro-life is subtle by design)."""
    rows = render.legibility(registry()[name].build(), kin)
    assert len(rows) >= 7
    weak = [
        (r.beat, round(max(r.eye_px, r.mouth_px), 1)) for r in rows if max(r.eye_px, r.mouth_px) < render.LEGIBLE_PX
    ]
    assert {beat for beat, _ in weak} <= SUBTLE_BY_DESIGN, weak


def test_the_pair_clip_states_both_durations_and_renders_side_by_side():
    clip = registry()["replay_pair"]
    reel = render.Reel(clip, fast=True)
    plain, puppy = reel.tracks
    assert puppy.duration > plain.duration
    line = reel.subtitle[1]
    assert f"plain {plain.duration:.1f} s" in line and f"Puppy Mode {puppy.duration:.1f} s" in line
    assert "demo budget" in line
    t, img = reel.frame(reel.n_frames - 1)  # the plain side holds its last frame
    reel.close()
    assert t == pytest.approx(puppy.duration, abs=2 * DT)
    assert img.size == (
        2 * int(render.PANEL_SIZE[0] * render.FAST_SCALE),
        int(render.PANEL_SIZE[1] * render.FAST_SCALE),
    )
    assert render.gif_frame(img, reel.kind).width == render.GIF_WIDTH["side"] == 960


@pytest.mark.parametrize("name", sorted(render.CAMERAS))
def test_the_camera_model_is_mujocos(sc, name):
    """Camera.axes and the eye of Camera.project are what MuJoCo renders with."""
    cam = render.CAMERAS[name]
    stage = render.Stage(sc, cam, (320, 180))
    look = np.array([0.2, -0.1, 0.4])
    stage.frame(np.array(runs.HOME_JOINTS), 0.5, False, look)
    left, right = stage.renderer.scene.camera  # the stereo eyes; a mono render uses their average
    forward, up = cam.axes()
    np.testing.assert_allclose(left.forward, forward, atol=1e-5)
    np.testing.assert_allclose(left.up, up, atol=1e-5)
    np.testing.assert_allclose((np.array(left.pos) + right.pos) / 2.0, look - cam.distance * forward, atol=1e-5)
    stage.close()
    assert cam.project(look, look, (320, 180))[0] == pytest.approx((160.0, 90.0))


def test_checks_catch_a_doctored_track(kin):
    track = registry()["idle_alive"].build()
    track.zone[100:110] = PRECISION  # pretend these samples were precision: they carry offsets
    track.tau[200:] += 0.5  # and a jump forward in functional time
    failed = {c.name for c in checks.check_track(track, kin) if not c.ok}
    assert {"precision", "dtau/dt"} <= failed


def test_check_command_exits_nonzero_on_a_violation(monkeypatch, kin):
    from puppy_sim import __main__ as cli

    track = registry()["idle_alive"].build()
    track.zone[100:110] = PRECISION
    monkeypatch.setattr(
        checks, "check_named", lambda name: [(name, "puppy", track.duration, checks.check_track(track, kin), [])]
    )
    assert cli.main(["check", "idle_alive"]) == 1
