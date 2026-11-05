"""The reaction gallery clips (success_proud, oops_retry, boop_button, wag_compare): they pass every
safety check, play only while the arm is stopped between moves, stay within the 'demo' budget where
they promise it, keep a held container level and never wag or chomp it, read on screen in their own
cameras, face the viewer where the boop needs it, and the snappy wag differs from today's only at
the wrist. Also: a pair clip's side labels default to 'Plain (today)' / 'Puppy Mode'."""

import numpy as np
import pytest
from PIL import Image

from puppy_sim import checks, director, layer, render
from puppy_sim.clips import PairClip, registry
from puppy_sim.clips import wag_compare as W
from puppy_sim.timeline import BOUNDARY, DT, STOP_RAD_S, joint_speed

GALLERY = ("success_proud", "oops_retry", "boop_button", "wag_compare")
BUDGETED = ("success_proud", "boop_button")
STILL_BY_DESIGN = {"FREEZE"}  # stock still: the pause is the expression
LEGIBLE_PX = 25.0
LEVEL_DEG = 0.5  # a held container's up axis may tilt this much at most (numerical)
FACING_DEG = 70.0  # the boop's face points within this of the camera at the stare and the happy look


@pytest.fixture(scope="module")
def built():
    """{clip name: (clip, tracks)} built once."""
    out = {}
    for name in GALLERY:
        clip = registry()[name]
        tracks = clip.build()
        out[name] = (clip, tracks if isinstance(tracks, tuple) else (tracks,))
    return out


def _puppy(built, name):
    return built[name][1][-1]


def test_the_gallery_clips_are_registered_with_their_cameras():
    clips = registry()
    for name in GALLERY:
        assert name in clips and clips[name].camera in render.CAMERAS
    assert isinstance(clips["boop_button"], PairClip) and isinstance(clips["wag_compare"], PairClip)


@pytest.mark.parametrize("name", GALLERY)
def test_gallery_clip_passes_every_check(built, sc, name):
    for track in built[name][1]:
        failed = [(c.name, c.shown, c.detail) for c in checks.check_track(track, sc.kin()) if not c.ok]
        assert not failed, (name, failed)


@pytest.mark.parametrize("name", BUDGETED)
def test_the_pauses_stay_within_the_demo_budget(built, name):
    puppy = _puppy(built, name)
    assert puppy.budget is director.BUDGETS["demo"]
    assert 0.0 < puppy.duration - puppy.plain_duration <= puppy.budget.seconds + 1e-9


@pytest.mark.parametrize("name", ("success_proud", "oops_retry", "boop_button"))
def test_beats_play_only_while_the_arm_is_stopped(built, name):
    """Every sample where the plain task moves plays at full speed, bit for bit the functional pose."""
    puppy = _puppy(built, name)
    assert puppy.tier == BOUNDARY
    src = puppy.func.source
    u = np.interp(puppy.func.tau, src.tau, np.arange(src.n, dtype=float))
    moving = joint_speed(src) >= STOP_RAD_S
    on_move = moving[np.floor(u).astype(int)] | moving[np.ceil(u).astype(int)]
    assert on_move.sum() > 100
    step = np.abs(np.diff(u) - 1.0) < 1e-6
    assert np.all(step[on_move[:-1]]) and np.all(step[on_move[1:]])
    assert puppy.q[on_move].tobytes() == puppy.func.q[on_move].tobytes()


@pytest.mark.parametrize("name", ("success_proud", "oops_retry"))
def test_a_held_container_stays_level_and_is_never_wagged_or_chomped(built, kin, name):
    """While holding: the gripper is the task's, the tool (and the container riding in it) only pans
    about the vertical (every axis keeps its height), and the only beat is the muted proud."""
    puppy = _puppy(built, name)
    held = np.nonzero(puppy.held)[0]
    assert held.size and np.any(puppy.q[held] != puppy.func.q[held])  # it does react while holding
    assert puppy.grip[held].tobytes() == puppy.func.grip[held].tobytes()
    heights = [np.abs(kin.fk(puppy.q[i], "tool")[1][2] - kin.fk(puppy.func.q[i], "tool")[1][2]).max() for i in held]
    assert max(heights) < np.sin(np.radians(LEVEL_DEG))
    held_beats = {b.name for b in puppy.beats if puppy.held[min(int(round(b.t0 / DT)), puppy.n - 1)]}
    assert held_beats == {"PROUD (holding)"}


@pytest.mark.parametrize("name", GALLERY)
def test_every_beat_reads_on_screen_in_its_camera(built, sc, name):
    clip, tracks = built[name]
    for track in tracks:
        if track.func is None:
            continue
        rows = render.legibility(track, sc.kin(), clip.camera)
        assert rows
        weak = [(r.beat, round(max(r.eye_px, r.mouth_px), 1)) for r in rows if max(r.eye_px, r.mouth_px) < LEGIBLE_PX]
        assert {beat for beat, _ in weak} <= STILL_BY_DESIGN, (name, weak)


def test_the_boop_faces_the_camera_at_the_stare_and_the_happy_look(built, kin):
    clip, (_, puppy) = built["boop_button"]
    cam = render.CAMERAS[clip.camera]
    forward, _ = cam.axes()
    look = cam.lookats(puppy, kin)
    for beat in ("LOCK ON", "HAPPY"):
        b = next(b for b in puppy.beats if b.name == beat)
        i = int(round((b.t0 + b.t1) / 2.0 / DT))
        eye, R = kin.fk(puppy.q[i], "eye")
        to_camera = (look[i] - cam.distance * forward) - eye
        angle = np.degrees(np.arccos(R[:, 2] @ to_camera / np.linalg.norm(to_camera)))
        assert angle < FACING_DEG, (beat, angle)


def test_the_snappy_wag_differs_only_at_the_wrist(built):
    clip, (today, snappy) = built["wag_compare"]
    assert (today.caps, snappy.caps) == (layer.DEFAULT_CAPS, "snappy")
    assert W.wrist_swing_deg(snappy) > 2.0 * W.wrist_swing_deg(today)
    arm = slice(0, 4)
    np.testing.assert_allclose(today.q[:, arm], snappy.q[:, arm], atol=1e-6)  # the same sway
    assert clip.headers == W.SIDES
    assert clip.subtitle_for((today, snappy))[0] == clip.subtitle  # no plain vs Puppy Mode durations


def test_a_pair_clip_keeps_the_default_side_labels(functional):
    assert registry()["replay_pair"].headers is None
    img = Image.new("RGB", (2 * render.PANEL_SIZE[0], render.PANEL_SIZE[1]))

    def side(**kw):
        return np.asarray(render.overlay_side(img, "t", "s", functional, functional, 0.0, None, **kw))

    assert np.array_equal(side(), side(headers=render.PAIR_HEADERS))
    assert not np.array_equal(side(), side(headers=("A", "B")))
