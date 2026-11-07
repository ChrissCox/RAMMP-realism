"""The reaction gallery clips (success_proud, oops_retry, boop_button, wag_compare), built on
puppy_motion (the Director for the three tasks, compose for the wag): they pass every safety check,
play only while the arm is stopped between moves, stay within the 'demo' budget (per task) where
they promise it, keep a held container level and never wag or chomp it, read on screen in their own
cameras, face the viewer where the boop needs it, and the snappy wag differs from today's only at
the wrist. Also: a pair clip's side labels default to 'Plain (today)' / 'Puppy Mode'."""

import numpy as np
import pytest
from PIL import Image

from puppy_sim import checks, layer, render
from puppy_sim.clips import PairClip, registry
from puppy_sim.clips import wag_compare as W
from puppy_sim.timeline import DT, STOP_RAD_S, functional_of, joint_speed

GALLERY = ("success_proud", "oops_retry", "boop_button", "wag_compare")
BUDGETED = ("success_proud", "boop_button")
STILL_BY_DESIGN = set()  # (the old freeze beat has no counterpart in the library)
TASKS = {"success_proud": 2, "boop_button": 1}  # tasks the clip shows: each has its own 6 s demo budget
DEMO_S = 6.0
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
def test_gallery_clip_passes_every_check(name):
    for clip, label, _, rows, _ in checks.check_named(name):
        failed = [(c.name, c.shown, c.detail) for c in rows if not c.ok]
        assert not failed, (name, label, failed)


@pytest.mark.parametrize("name", BUDGETED)
def test_the_pauses_stay_within_the_demo_budget(built, name):
    puppy = _puppy(built, name)
    assert puppy.budget.seconds == TASKS[name] * DEMO_S
    assert 0.0 < puppy.duration - puppy.plain_duration <= puppy.budget.seconds + 1e-9


@pytest.mark.parametrize("name", ("success_proud", "oops_retry", "boop_button"))
def test_beats_play_only_while_the_arm_is_stopped(built, name):
    """Every sample outside the beats plays the plain task, and the plain task's moves run at full speed."""
    puppy = _puppy(built, name)
    assert puppy.tier == "boundary"
    func, src = functional_of(puppy), puppy.source
    inside = np.zeros(puppy.n, dtype=bool)
    for sp in puppy.spliced:
        inside[sp.i0 : sp.i0 + len(sp.beat.points)] = True
    assert np.abs(puppy.q - func.q)[~inside].max() < 1e-6
    moving = joint_speed(src) >= STOP_RAD_S
    assert moving.sum() > 100
    u = np.interp(puppy.tau, src.tau, np.arange(src.n, dtype=float))
    on_move = moving[np.floor(u).astype(int)] | moving[np.ceil(u).astype(int)]
    step = np.abs(np.diff(u) - 1.0) < 1e-6
    assert np.all(step[on_move[:-1]]) and np.all(step[on_move[1:]])  # the moves are not retimed
    for sp in puppy.spliced:  # each beat starts where the arm is stopped
        assert np.abs(sp.beat.points[0][1] - sp.q_now).max() < 1e-6


@pytest.mark.parametrize("name", ("success_proud", "oops_retry"))
def test_a_held_container_stays_level_and_is_never_wagged_or_chomped(built, kin, name):
    """While holding: the gripper is the task's, the tool (and the container riding in it) only pans
    about the vertical (every axis keeps its height), and the only beats are the muted proud and the
    attentive look."""
    puppy = _puppy(built, name)
    func = functional_of(puppy)
    held = np.nonzero(puppy.held)[0]
    assert held.size and np.any(puppy.q[held] != func.q[held])  # it does react while holding
    np.testing.assert_allclose(puppy.grip[held], func.grip[held], atol=1e-9)
    heights = [np.abs(kin.fk(puppy.q[i], "tool")[1][2] - kin.fk(func.q[i], "tool")[1][2]).max() for i in held]
    assert max(heights) < np.sin(np.radians(LEVEL_DEG))
    held_beats = {b.name for b in puppy.beats if puppy.held[min(int(round(b.t0 / DT)), puppy.n - 1)]}
    assert held_beats == {"PROUD (holding)", "ATTENTIVE"}


@pytest.mark.parametrize("name", GALLERY)
def test_every_beat_reads_on_screen_in_its_camera(built, sc, name):
    clip, tracks = built[name]
    for track in tracks:
        if track.func is None and track.source is None:
            continue  # the plain track
        rows = render.legibility(track, sc.kin(), clip.camera)
        assert rows
        weak = [(r.beat, round(max(r.eye_px, r.mouth_px), 1)) for r in rows if max(r.eye_px, r.mouth_px) < LEGIBLE_PX]
        assert {beat for beat, _ in weak} <= STILL_BY_DESIGN, (name, weak)


def test_the_boop_faces_the_camera_at_the_stare_and_the_happy_look(built, kin):
    clip, (_, puppy) = built["boop_button"]
    cam = render.CAMERAS[clip.camera]
    forward, _ = cam.axes()
    look = cam.lookats(puppy, kin)
    for beat in ("LOCK ON", "PROUD"):
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
