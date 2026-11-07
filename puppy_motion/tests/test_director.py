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


def _still_tail_s(beat):
    """Seconds the beat stands still at its end posture."""
    q = np.array([p[1] for p in beat.points])
    moving = np.flatnonzero(np.abs(q - q[-1]).max(axis=1) > 1e-6)
    return beat.duration - beat.points[moving[-1] + 1][0] if moving.size else beat.duration


def _first_beat(d, q, held):
    """The beat the director would build after a move (AfterMove: curious), whatever the budget
    would let it play: the budget is tested separately."""
    beat = d._beat("curious", False, q, held, Zone.FREE, None, False)
    assert beat is not None, d.log
    return beat


@pytest.mark.parametrize(
    "control, settle",
    [
        ((1, "soft"), 2.3),
        ((1, "default"), 1.1),
        ((1, "medium"), 1.1),
        ((1, "stiff"), 0.1),
        (("impedance", "soft"), 2.3),
        ((1, None), 1.1),  # no gains given: the arm's session default (medium)
        ((1, {"kq": [80.0] * 7, "zeta": 0.5}), 2.3),  # custom gains: the longest
        ((7, "stiff"), 2.3),  # a mode it does not know: the longest
        (("position", None), 0.0),
        ((0, "soft"), 0.0),
    ],
)
def test_settle_before_precision_while_holding_under_impedance(chain, scene, targets, home, control, settle):
    base = _still_tail_s(_first_beat(make(chain, scene, targets, seed=3), home, True))
    d = make(chain, scene, targets, seed=3, control=control)
    beat = _first_beat(d, home, True)
    tail = _still_tail_s(beat)
    assert tail >= settle - 0.03
    assert tail >= base - 1e-9  # never shorter than the usual tail
    assert check_beat(beat.points, home, chain, scene, beat.zone, d.caps, held=True) == []
    if settle <= 0.2:  # position mode or a stiff arm: the beat is the usual one
        assert abs(tail - base) < 1e-9
    # not holding: the usual tail whatever the mode
    free = _first_beat(make(chain, scene, targets, seed=3, control=control), home, False)
    assert abs(_still_tail_s(free) - _still_tail_s(_first_beat(make(chain, scene, targets, seed=3), home, False))) < 1e-9


def test_set_control_changes_the_next_beat_and_is_default_safe(chain, scene, targets, home):
    d = make(chain, scene, targets, seed=3)
    assert d.control == ("position", None) and d.settle_s(True) == 0.0
    d.set_control(1, "soft")
    assert d.settle_s(True) == 2.3 and d.settle_s(False) == 0.0
    d.set_control(object(), [])  # unhashable or odd values never raise: the longest
    assert d.settle_s(True) == 2.3


@pytest.mark.parametrize("gains", ["soft", "default", None])
def test_ending_reactions_keep_the_usual_tail_so_subtle_still_plays_proud_held(chain, scene, targets, home, gains):
    d = make(chain, scene, targets, preset="subtle", control=(1, gains))
    beats = d.on_event(Succeeded(), home, True)
    assert [b.name for b in beats] == ["proud"]
    ref = make(chain, scene, targets, preset="subtle").on_event(Succeeded(), home, True)
    assert abs(beats[0].duration - ref[0].duration) < 1e-9


def test_a_held_soft_arm_in_demo_still_keeps_eager(chain, scene, targets, home):
    d = make(chain, scene, targets, preset="demo", control=(1, "soft"))
    names = [b.name for e in (TaskRequested(), BeforeMove(False), AfterMove()) for b in d.on_event(e, home, True)]
    names += [b.name for b in d.on_event(Succeeded(), home, True)]
    assert "eager" in names and "proud" in names, (names, d.log)
    assert d.spent_s <= d.budget_s + 1e-9


def test_beats_before_a_move_keep_the_usual_tail(chain, scene, targets, home):
    for event in (TaskRequested(), BeforeMove(False)):
        soft = make(chain, scene, targets, seed=3, control=(1, "soft")).on_event(event, home, True)
        usual = make(chain, scene, targets, seed=3).on_event(event, home, True)
        assert [b.name for b in soft] == [b.name for b in usual]
        assert all(abs(a.duration - u.duration) < 1e-9 for a, u in zip(soft, usual))


def test_only_after_move_and_precise_approach_beats_ask_for_the_settle_tail(chain, scene, targets, home):
    asked = {}
    for label, events in (
        ("plain", (TaskRequested(), BeforeMove(False), AfterMove(), Succeeded())),
        ("precise", (BeforeMove(True),)),
    ):
        d = make(chain, scene, targets, control=(1, "soft"))
        real = d._beat

        def spy(name, short, q, held, zone, tag, usual_tail=False, real=real, label=label):
            if tag:
                asked[label, tag.split()[0]] = usual_tail
            return real(name, short, q, held, zone, tag, usual_tail)

        d._beat = spy
        d.on_event(Phase("view", ""), home, True)
        for e in events:
            d.on_event(e, home, True)
    assert asked["plain", "TaskRequested"] is True and asked["plain", "Succeeded"] is True
    assert asked.get(("plain", "BeforeMove"), True) is True
    assert asked["plain", "AfterMove"] is False
    assert asked["precise", "BeforeMove"] is False  # lock_on: the settle tail (it may then be dropped for budget)
