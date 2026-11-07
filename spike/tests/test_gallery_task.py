"""The task gallery clips (anticipation_pair, lock_on_view, lock_on_standoff, failure_pair and
replay_pair), built on puppy_motion's Director (replay.direct_run / direct_scripted): registered
pair clips, every check passes (the replay checks: each beat valid, the PRECISION samples and the
plain path untouched, the beats' time within the budget), each plays the beats the director decides
for the events it is given, the lock-on modes differ as they should, and the expressions each clip
is about read on screen (at least READS_PX of eye or mouth travel in the clip's own camera at
1280x720)."""

import numpy as np
import pytest

from puppy_sim import checks, render
from puppy_sim.clips import PairClip, registry
from puppy_sim.timeline import DT, PRECISION, functional_of, speed

GALLERY = ("anticipation_pair", "lock_on_view", "lock_on_standoff", "failure_pair", "replay_pair")
READS_PX = 25.0
LEGIBLE = {  # the expressions each clip is about (the counter camera is far: replay_pair's small ones are not)
    "anticipation_pair": {"WIND UP", "FOLLOW THROUGH"},
    "lock_on_view": {"LOCK ON"},
    "lock_on_standoff": {"LOCK ON"},
    "failure_pair": set(),  # a failure in a PRECISION phase plays nothing: the clip shows the robot staying still
    "replay_pair": {"EAGER", "PROUD"},
}
DEMO_S = 6.0


@pytest.fixture(scope="module")
def built():
    """{clip: (plain, puppy)}, each built once."""
    clips = registry()
    return {name: clips[name].build() for name in GALLERY}


def _order(puppy):
    names = []
    for b in sorted(puppy.beats, key=lambda b: b.t0):
        if not names or names[-1] != b.name:
            names.append(b.name)
    return names


def test_the_gallery_clips_are_registered_pair_clips():
    clips = registry()
    for name in GALLERY:
        assert isinstance(clips[name], PairClip) and clips[name].camera in render.CAMERAS


@pytest.mark.parametrize("name", GALLERY)
def test_gallery_clip_passes_every_check(name):
    for clip, label, _, rows, _ in checks.check_named(name):
        failed = [(c.name, c.shown, c.detail) for c in rows if not c.ok]
        assert not failed, (label, failed)


@pytest.mark.parametrize("name", GALLERY)
def test_gallery_expressions_read_on_screen(built, kin, name):
    rows = render.legibility(built[name][1], kin, registry()[name].camera)
    reach = {}
    for r in rows:
        reach[r.beat] = max(reach.get(r.beat, 0.0), r.eye_px, r.mouth_px)
    assert LEGIBLE[name] <= set(reach)
    assert all(reach[beat] >= READS_PX for beat in LEGIBLE[name]), reach


def test_anticipation_winds_up_before_the_move_and_follows_through_after_it(built, kin):
    plain, puppy = built["anticipation_pair"]
    assert _order(puppy) == ["WIND UP", "FOLLOW THROUGH"]
    assert puppy.tier == "boundary"
    wind, follow = (next(b for b in puppy.beats if b.name == n) for n in ("WIND UP", "FOLLOW THROUGH"))
    func = functional_of(puppy)  # the plain move on the puppy's clock
    moving = np.nonzero(np.abs(np.gradient(func.q, DT, axis=0)).max(axis=1) > 0.02)[0]
    assert wind.t1 <= moving[0] * DT <= moving[-1] * DT <= follow.t0 + DT  # before the move, after it


def test_lock_on_at_the_view_pose_stares_after_detect_and_reaches_as_recorded(built, kin):
    plain, puppy = built["lock_on_view"]
    assert _order(puppy) == ["LOCK ON"]
    stare = puppy.beats[0]
    precise = np.nonzero(puppy.zone == PRECISION)[0] * DT
    assert precise[0] < stare.t0 and stare.t1 < precise[precise > stare.t1][0]  # between detect and the reach
    assert puppy.tier == "boundary" and np.all(speed(puppy) >= 0.0)
    assert puppy.duration > plain.duration


def test_lock_on_at_the_standoff_stops_mid_reach_and_says_it_needs_the_call_split(built, kin):
    plain, puppy = built["lock_on_standoff"]
    stare = next(b for b in puppy.beats if b.name == "LOCK ON")
    s = speed(puppy)
    assert np.all(s[int(round(stare.t0 / DT)) : int(round(stare.t1 / DT))] == 0.0)  # stopped while it stares
    assert puppy.tier == "standoff"
    assert "split" in registry()["lock_on_standoff"].subtitle
    view = built["lock_on_view"][1]
    assert puppy.duration > view.duration  # the stop and the creep cost more than the stare at the view pose


def test_failure_pair_plays_nothing_because_the_failure_is_in_a_precise_phase(built, kin):
    plain, puppy = built["failure_pair"]
    assert _order(puppy) == []  # no phase is invented: the last one reported is 'detect' (PRECISION)
    assert puppy.duration == pytest.approx(plain.duration)
    clip = registry()["failure_pair"]
    assert "stays still" in clip.title and "refused" in clip.subtitle and "no sorry" in clip.subtitle


def test_replay_pair_plays_the_demo_plan(built, kin):
    plain, puppy = built["replay_pair"]
    assert _order(puppy) == ["EAGER", "LOCK ON", "PROUD"]
    assert puppy.duration - plain.duration >= 0.85 * DEMO_S
    assert puppy.tier == "boundary"
