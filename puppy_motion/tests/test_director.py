"""The online director: causality, the budgets, validity at q_now, the zone and held gates, and the
rules each capability gets (box_opening only at the task's boundaries)."""

import numpy as np
import pytest

from puppy_motion.behaviors import Targets
from puppy_motion.director import (
    BEATS,
    BUDGETS,
    AfterMove,
    BeforeMove,
    Director,
    Failed,
    Phase,
    Succeeded,
    TaskRequested,
    load_rules,
)
from puppy_motion.beat import zone_at
from puppy_motion.track import Zone
from puppy_motion.validate import check_beat

LOG_LEN = 8


@pytest.fixture(scope="module")
def targets(scene):
    return Targets(person_head=scene.person_head(), container=np.array([0.5, 0.3, 0.2]), button=np.array([0.5, 0.0, 0.5]))


def make(chain, scene, targets, capability="grab_container", preset="demo", **kw):
    return Director(capability, chain, scene, targets, preset, **kw)


def random_log(rng):
    pool = [
        lambda: TaskRequested(),
        lambda: Phase(str(rng.choice(["view", "home", "detect", "raise", "done", "renamed"])), ""),
        lambda: BeforeMove(bool(rng.integers(2))),
        lambda: AfterMove(),
        lambda: Succeeded(),
        lambda: Failed(bool(rng.integers(2))),
    ]
    return [pool[rng.integers(len(pool))]() for _ in range(LOG_LEN)]


def play(d, events, q, held=False):
    return [[b.name for b in d.on_event(e, q, held)] for e in events]


def test_causal_prefix(chain, scene, targets, home):
    rng = np.random.default_rng(1)
    for _ in range(5):
        events = random_log(rng)
        full = play(make(chain, scene, targets, seed=2), events, home)
        for k in range(1, len(events) + 1):
            assert play(make(chain, scene, targets, seed=2), events[:k], home) == full[:k]


@pytest.mark.parametrize("preset", sorted(BUDGETS))
def test_budget_never_exceeded(chain, scene, targets, home, preset):
    rng = np.random.default_rng(7)
    for seed in range(200):
        d = make(chain, scene, targets, preset=preset, seed=seed % 3)  # composed beats are cached per seed
        beats = [b for e in random_log(rng) for b in d.on_event(e, home, False)]
        assert d.spent_s <= BUDGETS[preset] + 1e-9
        assert abs(d.spent_s - sum(b.duration for b in beats)) < 1e-9


def test_every_beat_starts_and_ends_at_q_now_and_validates(chain, scene, targets, home):
    seen = set()
    for held in (False, True):
        for cap in ("grab_container", "button_press", "box_opening"):
            events = [TaskRequested(), Phase("view", ""), BeforeMove(True), BeforeMove(False), AfterMove(),
                      Phase("home", ""), AfterMove(), Failed(True)]
            d = make(chain, scene, targets, cap, seed=3)
            for e in events:
                for b in d.on_event(e, home, held):
                    seen.add(b.name)
                    assert np.array_equal(b.points[0][1], home) and np.array_equal(b.points[-1][1], home)
                    assert not np.any(b.points[0][2]) and not np.any(b.points[-1][2])
                    assert check_beat(b.points, home, chain, scene, b.zone, held=held) == []
            d = make(chain, scene, targets, cap, seed=3)
            for e in [Succeeded()]:
                for b in d.on_event(e, home, held):
                    seen.add(b.name)
                    assert check_beat(b.points, home, chain, scene, b.zone, held=held) == []
    assert {"eager", "proud", "sorry"} <= seen


def test_every_beat_version_composes_or_is_refused_cleanly(chain, scene, targets, home):
    d = make(chain, scene, targets)
    for name in BEATS:
        for short in (False, True):
            beat = d._beat(name, short, home, False, zone_at(chain, scene, home, False), None)
            if beat is not None:
                assert check_beat(beat.points, home, chain, scene, beat.zone) == []


def test_non_recoverable_failure_gets_nothing(chain, scene, targets, home):
    assert make(chain, scene, targets).on_event(Failed(False), home, False) == []


def test_unknown_phase_gets_nothing_and_logs(chain, scene, targets, home):
    d = make(chain, scene, targets)
    assert d.on_event(Phase("renamed", ""), home, False) == [] and "renamed" in d.log[-1]


def test_subtle_keeps_only_priority_six_and_up(chain, scene, targets, home):
    keep = {r.beat: r.keep for r in load_rules()["grab_container"]}
    d = make(chain, scene, targets, preset="subtle")
    names = play(d, [TaskRequested(), BeforeMove(False), AfterMove(), Succeeded()], home)
    played = {n for step in names for n in step}
    assert played == {"proud"} and all(keep[n] >= 6 for n in played)
    assert d.spent_s <= 2.5 + 1e-9
    f = make(chain, scene, targets, preset="subtle")
    assert set(play(f, [Failed(True)], home)[0]) == {"sorry"}


@pytest.mark.parametrize("capability", ["grab_container", "box_opening"])
def test_subtle_plays_the_success_reaction(chain, scene, targets, home, capability):
    d = make(chain, scene, targets, capability, preset="subtle")
    assert [b.name for b in d.on_event(Succeeded(), home, False)] == ["proud"]


def test_short_proud_fits_the_subtle_budget(chain, scene, targets, home):
    d = make(chain, scene, targets)
    beat = d._beat("proud", True, home, False, zone_at(chain, scene, home, False), None)
    assert beat is not None and beat.duration <= 2.0


def test_subtle_lock_on_never_starves_proud(chain, scene, targets, home):
    d = make(chain, scene, targets, preset="subtle")
    for _ in range(3):
        d.on_event(BeforeMove(True), home, False)
    assert [b.name for b in d.on_event(Succeeded(), home, False)] == ["proud"]


def test_lock_on_fires_before_the_precise_approach(chain, scene, targets, home):
    d = make(chain, scene, targets)
    assert [b.name for b in d.on_event(BeforeMove(True), home, False)] == ["lock_on"]


def test_ending_reaction_is_not_starved_by_earlier_beats(chain, scene, targets, home):
    d = make(chain, scene, targets)
    for e in [TaskRequested(), BeforeMove(False), AfterMove(), BeforeMove(False), AfterMove(), BeforeMove(False)]:
        d.on_event(e, home, False)
    assert "proud" in [b.name for b in d.on_event(Succeeded(), home, False)]


def test_two_precise_moves_do_not_spend_the_success_reaction(chain, scene, targets, home):
    d = make(chain, scene, targets)
    first = play(d, [TaskRequested(), BeforeMove(True), BeforeMove(True)], home)
    assert first[0] == ["eager"]
    assert "proud" in [b.name for b in d.on_event(Succeeded(), home, False)]


def test_a_retry_after_a_recoverable_failure_keeps_the_success_reserve(chain, scene, targets, home):
    d = make(chain, scene, targets)
    d.on_event(Failed(True), home, False)
    d.spent_s = 0.0  # the failure reaction used the budget; the reserve logic is what is under test
    for _ in range(4):
        d.on_event(BeforeMove(True), home, False)
    assert "proud" in [b.name for b in d.on_event(Succeeded(), home, False)]


def test_a_bad_q_now_gets_nothing_and_logs(chain, scene, targets):
    d = make(chain, scene, targets)
    assert d.on_event(Succeeded(), np.zeros(3), False) == [] and "bad q_now" in d.log[-1]
    assert d.on_event(Succeeded(), "nope", False) == [] and "bad q_now" in d.log[-1]


def test_a_returned_beat_is_a_copy_of_the_cached_one(chain, scene, targets, home):
    a = make(chain, scene, targets).on_event(Succeeded(), home, False)[0]
    a.points[0][1][:] += 1.0
    a.points.clear()
    b = make(chain, scene, targets).on_event(Succeeded(), home, False)[0]
    assert np.array_equal(b.points[0][1], home)


def test_recoverable_failure_in_a_precision_phase_gets_nothing(chain, scene, targets, home):
    for cap, phase in (("grab_container", "detect"), ("button_press", "press"), ("box_opening", "grip")):
        d = make(chain, scene, targets, cap)
        d.on_event(Phase(phase, "x"), home, False)
        assert d.on_event(Failed(True), home, False) == [] and "PRECISION" in d.log[-1]
    d = make(chain, scene, targets)
    d.on_event(Phase("aborted", ""), home, False)  # FREE: the arm holds where it stopped
    assert {b.name for b in d.on_event(Failed(True), home, False)} >= {"sorry"}


def test_precision_phase_and_moves_inside_it_get_nothing(chain, scene, targets, home):
    d = make(chain, scene, targets)
    assert d.on_event(Phase("detect", ""), home, False) == []
    assert d.on_event(BeforeMove(False), home, False) == []
    assert d.on_event(AfterMove(), home, False) == []


def test_after_a_precise_move_the_arm_stays_settled(chain, scene, targets, home):
    d = make(chain, scene, targets)
    d.on_event(BeforeMove(True), home, False)
    assert d.on_event(AfterMove(), home, False) == []


def test_held_near_the_person_gets_nothing(chain, scene, targets, home):
    near = Targets(person_head=scene.person_head(), container=targets.container)
    q = _q_near_person(chain, scene, home)
    assert zone_at(chain, scene, q, True) is Zone.NEAR_USER
    d = make(chain, scene, near)
    assert d.on_event(Succeeded(), q, True) == [] and "near the person" in d.log[-1]
    assert d.on_event(TaskRequested(), q, True) == []
    assert d.on_event(Failed(True), q, True) == []


def _q_near_person(chain, scene, home):
    """A posture with the head under 0.35 m from the person: home, shoulder swung toward them."""
    for j in range(7):
        for step in np.linspace(-1.5, 1.5, 31):
            q = home.copy()
            q[j] += step
            if zone_at(chain, scene, q, True) is Zone.NEAR_USER:
                return q
    pytest.skip("no near-user posture found by single-joint sweeps")


def test_unknown_capability_or_preset_is_an_error(chain, scene, targets):
    with pytest.raises(ValueError):
        make(chain, scene, targets, "fridge")
    with pytest.raises(ValueError):
        make(chain, scene, targets, preset="loud")
    with pytest.raises(ValueError):
        make(chain, scene, targets, lock_on="never")


def test_button_press_uses_the_same_rows():
    rules = load_rules()
    assert rules["button_press"] == rules["grab_container"]


def test_grab_container_rows():
    rows = [(r.event, r.beat, r.keep) for r in load_rules()["grab_container"]]
    assert rows == [
        ("TaskRequested", "eager", 5), ("AfterMove", "curious", 4), ("BeforeMove", "lock_on", 7),
        ("BeforeMove", "wind_up", 2), ("AfterMove", "follow_through", 1), ("Succeeded", "proud", 6),
        ("Succeeded", "attentive", 1), ("Failed", "search", 3), ("Failed", "sorry", 6), ("Failed", "shake_off", 4),
    ]


def test_box_opening_rows_and_zones():
    from puppy_motion.zones import zone_of

    rows = [(r.event, r.beat, r.keep, r.recoverable) for r in load_rules()["box_opening"]]
    assert rows == [
        ("TaskRequested", "eager", 5, None), ("Succeeded", "proud", 6, None),
        ("Failed", "sorry", 6, True), ("Failed", "shake_off", 4, True),
    ]
    for phase in ("locate", "look", "press", "grip", "carry", "release", "home"):
        assert zone_of("box_opening", phase, "anything").zone is Zone.PRECISION
    assert zone_of("box_opening", "renamed", "anything").zone is Zone.PRECISION


def test_box_opening_yields_only_boundary_beats(chain, scene, targets, home):
    d = make(chain, scene, targets, "box_opening")
    assert [b.name for b in d.on_event(TaskRequested(), home, False)] == ["eager"]
    for phase in ("locate", "look", "press", "grip", "carry", "release", "home"):
        assert d.on_event(Phase(phase, "x"), home, False) == []
        assert d.on_event(BeforeMove(False), home, False) == []
        assert d.on_event(BeforeMove(True), home, False) == []
        assert d.on_event(AfterMove(), home, False) == []
    assert [b.name for b in d.on_event(Succeeded(), home, False)] == ["proud"]
    f = make(chain, scene, targets, "box_opening")
    assert set(b.name for b in f.on_event(Failed(True), home, False)) <= {"sorry", "shake_off"}
    assert f.on_event(Failed(False), home, False) == []


def test_box_opening_moves_get_nothing_even_without_a_phase(chain, scene, targets, home):
    d = make(chain, scene, targets, "box_opening")
    assert d.on_event(BeforeMove(False), home, False) == []
    assert d.on_event(AfterMove(), home, False) == []


def test_box_opening_aborted_home_gets_no_sorry_and_a_precision_phase_still_gets_proud(chain, scene, targets, home):
    f = make(chain, scene, targets, "box_opening")
    f.on_event(Phase("home", "going home: stopped"), home, False)
    assert f.on_event(Failed(True), home, False) == []
    d = make(chain, scene, targets, "box_opening")
    d.on_event(Phase("release", "x"), home, False)
    assert [b.name for b in d.on_event(Succeeded(), home, False)][:1] == ["proud"]


@pytest.mark.parametrize("preset", ["demo", "subtle"])
def test_box_opening_success_is_a_single_proud_under_every_preset(chain, scene, targets, home, preset):
    """The hook defers TaskRequested to the first move, which box_opening never makes: after a
    success the only beat is one proud, whatever budget is left."""
    d = make(chain, scene, targets, "box_opening", preset=preset)
    for phase in ("locate", "look", "press", "grip", "carry", "release", "home"):
        assert d.on_event(Phase(phase, "x"), home, False) == []
    assert [b.name for b in d.on_event(Succeeded(), home, False)] == ["proud"]
