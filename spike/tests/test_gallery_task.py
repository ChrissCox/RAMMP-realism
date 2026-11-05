"""The task gallery clips (anticipation_pair, lock_on_view, lock_on_standoff, failure_pair and
replay_pair): registered pair clips, every check passes, each plays the beats it promises in order
and keeps the tier it says, the lock-on modes differ as they should, and the expressions each clip
is about read on screen (at least READS_PX of eye or mouth travel in the clip's own camera at
1280x720)."""

import numpy as np
import pytest

from puppy_sim import checks, director, render
from puppy_sim.clips import PairClip, registry
from puppy_sim.timeline import DT, PRECISION, speed

GALLERY = ("anticipation_pair", "lock_on_view", "lock_on_standoff", "failure_pair", "replay_pair")
READS_PX = 25.0
LEGIBLE = {  # the expressions each clip is about (the counter camera is far: replay_pair's small ones are not)
    "anticipation_pair": {"WIND UP", "FOLLOW THROUGH"},
    "lock_on_view": {"LOCK ON"},
    "lock_on_standoff": {"LOCK ON"},
    "failure_pair": {"LOCK ON", "PUZZLED", "SORRY", "SHAKE IT OFF"},
    "replay_pair": {"EAGER", "PROUD"},
}


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


def _tier(track, kin):
    return next(c for c in checks.check_track(track, kin) if c.name == "tier")


def test_the_gallery_clips_are_registered_pair_clips():
    clips = registry()
    for name in GALLERY:
        assert isinstance(clips[name], PairClip) and clips[name].camera in render.CAMERAS


@pytest.mark.parametrize("name", GALLERY)
def test_gallery_clip_passes_every_check(built, kin, name):
    for track in built[name]:
        failed = [(c.name, c.shown, c.detail) for c in checks.check_track(track, kin) if not c.ok]
        assert not failed


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
    assert puppy.tier == director.BOUNDARY and _tier(puppy, kin).shown == "boundary"
    moving = np.nonzero(np.abs(np.gradient(plain.q, DT, axis=0)).max(axis=1) > 0.02)[0]
    wind, follow = (next(b for b in puppy.beats if b.name == n) for n in ("WIND UP", "FOLLOW THROUGH"))
    assert wind.t1 <= moving[0] * DT <= moving[-1] * DT <= follow.t0  # before the move, after it


def test_lock_on_at_the_view_pose_stares_after_detect_and_reaches_as_recorded(built, kin):
    plain, puppy = built["lock_on_view"]
    assert _order(puppy) == ["FOCUS", "LOCK ON", "FOCUS"]
    stare = next(b for b in puppy.beats if b.name == "LOCK ON")
    precise = np.nonzero(puppy.zone == PRECISION)[0] * DT
    assert precise[0] < stare.t0 and stare.t1 < precise[precise > stare.t1][0]  # between detect and the reach
    assert _tier(puppy, kin).shown == "boundary" and np.all(speed(puppy) >= 0.0)
    assert puppy.duration > plain.duration


def test_lock_on_at_the_standoff_stops_mid_reach_and_says_it_needs_the_call_split(built, kin):
    plain, puppy = built["lock_on_standoff"]
    stare = next(b for b in puppy.beats if b.name == "LOCK ON")
    s = speed(puppy)
    assert np.all(s[int(round(stare.t0 / DT)) : int(round(stare.t1 / DT))] == 0.0)  # stopped while it stares
    assert puppy.tier == director.STANDOFF and "retimed" in _tier(puppy, kin).shown
    assert "split" in registry()["lock_on_standoff"].subtitle
    view = built["lock_on_view"][1]
    assert puppy.duration > view.duration  # the stop and the creep cost more than the stare at the view pose


def test_failure_pair_plays_the_whole_apology_within_the_demo_budget(built, kin):
    plain, puppy = built["failure_pair"]
    assert _order(puppy) == ["CURIOUS", "LOCK ON", "FOCUS", "PUZZLED", "SORRY", "SHAKE IT OFF"]
    assert 0.0 < puppy.duration - plain.duration <= director.BUDGETS["demo"].seconds
    assert _tier(puppy, kin).shown == "boundary"
    assert "refused" in registry()["failure_pair"].subtitle


def test_replay_pair_plays_the_demo_plan(built, kin):
    plain, puppy = built["replay_pair"]
    assert _order(puppy) == ["EAGER", "CURIOUS", "FOCUS", "LOCK ON", "WIND UP", "FOCUS", "PROUD"]
    assert puppy.duration - plain.duration >= 0.85 * director.BUDGETS["demo"].seconds
    assert _tier(puppy, kin).shown == "boundary"
