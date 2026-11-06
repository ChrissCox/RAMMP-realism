"""The idle director: determinism, moods, requests, validity at rest and at a stage pose, held
muting, and small beats that never come closer to the person."""

import numpy as np
import pytest

from puppy_motion.beat import Beat
from puppy_motion.behaviors import Targets
from puppy_motion.flatbot import chair_scene
from puppy_motion.geometry import clearances
from puppy_motion.idle import EXPRESSIONS, LOOK_TARGET, MOODS, IdleDirector
from puppy_motion.stage import find_stage_poses
from puppy_motion.track import Zone
from puppy_motion.validate import check_beat


@pytest.fixture(scope="module")
def room():
    return chair_scene()


@pytest.fixture(scope="module")
def stage_q(chain, room):
    poses = find_stage_poses(chain, room, 1, seed=1)
    assert poses
    return poses[0]


def _director(chain, room, seed=3, size=0.5):
    return IdleDirector(chain, room, Targets(person_head=LOOK_TARGET), size, seed)


@pytest.fixture
def cheap(monkeypatch):
    """Sequencing tests do not compose (a composition takes seconds): _make returns an empty beat."""
    monkeypatch.setattr(IdleDirector, "_make", lambda self, name, mood, q, held: Beat(name, mood, [], Zone.FREE))


def _run(d, home, n, held=False, step=5.0):
    t, out = 0.0, []
    for _ in range(n):
        beat, pause = d.next(home, held, t)
        out.append((beat, pause, d.mood, t))
        t += pause + step
    return out


def test_same_seed_same_sequence(cheap, chain, room, home):
    a = [(b.name, p) for b, p, _, _ in _run(_director(chain, room), home, 50)]
    b = [(b.name, p) for b, p, _, _ in _run(_director(chain, room), home, 50)]
    assert a == b
    assert all(1.0 <= p <= 6.0 for _, p in a)


def test_moods_visit_all_four_over_a_long_run(cheap, chain, room, home):
    run = _run(_director(chain, room, seed=5), home, 120, step=6.0)
    assert {m for _, _, m, _ in run} == set(MOODS)
    assert min(t for _, _, m, t in run if m == "sleepy") >= 180.0
    names = [b.name for b, _, m, _ in run if m == "sleepy"]
    assert names[:2] == ["droop", "doze"]


def test_request_resets_the_sleepy_clock_and_plays_the_named_beat(cheap, chain, room, home):
    d = _director(chain, room)
    d.next(home, False, 0.0)
    beat = d.request("head_tilt", home, False, 170.0)
    assert beat is not None and beat.name == "head_tilt"
    # 130 s after the request the sleepy mood is still not offered
    for k in range(20):
        d.next(home, False, 300.0 + k * 0.1)
        assert d.mood != "sleepy"
    d.mood, d._mood_end = "calm", 0.0
    assert d.request("nope", home, False, 400.0) is None and "nope" in d.log[-1]


def test_held_never_returns_wiggle_or_bounce(cheap, chain, room, home):
    d = _director(chain, room, seed=9)
    names = {b.name for b, _, _, _ in _run(d, home, 120, held=True, step=6.0)}
    assert names and not names & {"wag", "bounce", "shake_off"}


def test_held_wiggle_and_bounce_requests_are_refused(chain, room, home):
    d = _director(chain, room)  # refused before any composition
    for name in ("wag", "bounce", "shake_off"):
        assert d.request(name, home, True, 0.0) is None
        assert name in d.log[-1]


def test_every_returned_beat_validates_at_rest_and_at_a_stage_pose(chain, room, home, stage_q):
    for q, name in ((home, "startle"), (stage_q, "glance")):
        for held in (False, True):
            beat = _director(chain, room, size=1.0).request(name, q, held, 0.0)
            assert beat is not None, (name, held)
            assert check_beat(beat.points, q, chain, room, beat.zone, held=held) == []
            assert not np.array(beat.points[0][2]).any() and not np.array(beat.points[-1][2]).any()
            assert max(np.abs(p[1] - q).max() for p in beat.points) > 1e-3


def test_at_rest_beats_are_small_and_never_closer_to_the_person(chain, room, home):
    start = clearances(chain, room, home, False)[0]
    beat = _director(chain, room, size=1.0).request("startle", home, False, 0.0)
    assert beat is not None and beat.zone == Zone.NEAR_USER
    for _, q, _ in beat.points[::2]:
        assert (clearances(chain, room, q, False)[0] >= start - 0.005).all()
    assert max(np.abs(p[1] - home).max() for p in beat.points) < 0.35
