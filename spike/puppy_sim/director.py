"""The director: maps a task's events to Puppy Mode beats, event by event, as the robot would.

RULES is the whole mapping, one readable table per capability: a task event, the beat that plays
at it, its mood chip, its tier, its priority and the time held back for it. Several rules may share
an event; they play in table order. find_events() reads the stages and the events off the
functional track.

Tiers. A BOUNDARY beat plays only while the functional motion is fully stopped between two moves
(a stage: every joint slower than STOP_RAD_S for at least STOP_MIN_S, or across a phase change,
outside PRECISION) and starts and ends there at rest. On the robot it is a short self-contained
motion between the capability's arm calls, which the driver plans and runs untouched: a pause
(timeline.Hold) only ever goes inside a stage, so every moving sample plays exactly as recorded, and
a fence zeroes every boundary beat wherever the arm moves. IN_MOTION beats (lead_gaze) ride on a
moving trajectory and need Puppy Mode to plan and run the move itself, a later phase: opt in with
tiers=(BOUNDARY, IN_MOTION).

The lock-on. grab_container's approach (view -> pre-grasp -> grasp) is one blended driver call. By
default (lock_on='view') the stare plays at the view pose once detect is done, while the gripper
opens, before the approach starts (and, where the arm waits at the view pose before detect anyway,
there too: a free beat). lock_on='standoff' stops at the pre-grasp standoff instead and creeps in at
x0.75: that needs the capability to split its approach into two calls.

Budgets (BUDGETS: seconds per task, and the least priority that plays at all). Each event is decided
when it arrives, from what has happened so far. A beat is priced at what its stop would then add as
far as the director knows the stop: as if the arm moved on right after now into something precise
(it cannot know when or what); exactly once the arm is about to move (a BEFORE beat: the stop's whole
length is known then); and once the task has ended, as if its next command came NEXT_COMMAND_S
later (an assumption about the user; a replay keeps 2 s past the end). A version fits if its stop,
with the short versions of the more important beats still to decide at this event, fits the time
left less what is held back (reserve_s: about what they usually add) for more important beats at
later events (of the two endings, success and failure, only the larger: one of them plays). A beat
plays in full if that fits at its full version's priority (full_keep: the extra length is worth
less), else in short at its own, else not at all. When the arm moves on the stop is charged what it
really added, its ramps included, so the budget is never exceeded, and what a stop did not use goes
to the beats after it. A BEFORE beat starts no earlier than the last task event in its stop (the
director learns then what comes next).

A phase the table does not know is PRECISION (runs.zone_of) and gets no beat. Every beat is fenced:
faded to exactly zero before any PRECISION sample (noted in the log if that ever changes it). The
layer zeroes PRECISION on top of that.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from . import behaviors as B
from . import layer, runs
from . import scene as scene_mod
from .layer import DEFAULT_CAPS
from .timeline import (
    BOUNDARY,
    DT,
    IN_MOTION,
    PRECISION,
    STOP_RAD_S,
    Budget,
    Hold,
    Label,
    Slow,
    Track,
    constant_track,
    hold_ramp_s,
    joint_speed,
    output_time,
    still_span,
    warp,
    warp_indices,
)

STANDOFF = "standoff"  # a directed track's tier with lock_on='standoff': boundary beats, the approach split
AFTER, BEFORE, SPAN = "after", "before", "span"  # a beat starts at its event / ends at it / spans it
FULL, SHORT = "full", "short"
VERSIONS = (FULL, SHORT)  # tried in this order
SUCCESS, FAILURE = "success", "failure"  # a task's two endings: only one of them plays
LOCK_ON_MODES = ("view", "standoff")


@dataclass(frozen=True)
class Allowance(Budget):
    """A director's budget: `seconds` a task may gain, and min_keep, the least priority (Rule.keep)
    a beat needs to play at all under it (a subtle budget keeps only the beats that matter most)."""

    min_keep: int = 0


BUDGETS = {"demo": Allowance("demo", 6.0), "subtle": Allowance("subtle", 2.5, min_keep=6)}
DEFAULT_BUDGET = "demo"
NEXT_COMMAND_S = 1.9  # once a task has ended its next command is assumed no sooner than this (a replay
# keeps runs.TAIL_S = 2 s past the end): an ending reaction is charged only for what runs past it
EPS = 1e-9

STOP_MIN_S = 0.2  # ... for at least this long (or across a phase change): between two moves
STILL_MARGIN_S = 0.1  # boundary beats start this long after the arm stops and end this long before it moves
PRECISION_CLEAR_S = 0.34  # ... and keep this far from PRECISION (the gain ramp and its guard)
BOUNDARY_FADE_S = 0.08  # the boundary fence: beats ease to exactly zero this fast before the arm moves
HOLD_RAMP_MIN_S = 0.06  # a pause in a stage eases the (stopped) motion over at least this
HOLD_EDGE = 3  # samples a pause keeps from a stage's ends (its ramps stay inside the stage)
STANDOFF_RAMP_MIN_S = 0.3  # standoff: the stop mid-approach eases over at least this
TRANSIT_MIN_M = 0.05  # a move of the eye shorter than this is not a transit
PHASE_REACH_S = 3 * DT  # a phase that starts this little before a stage (as PRECISION ends) is in it
TERMINAL_PHASES = ("done", "aborted", "canceled")  # once one comes, no more events will: nothing is held back
LOCK_MARGIN_S = 0.06  # standoff: the stop is this long before the first precise sample (on a FREE one)
MEASURE_PAD_S = 0.5  # a self-timed beat is measured on a still track from this time
MEASURE_S = 40.0
BACKGROUND_SEED = 3
QUIET_BEATS = ("wind_up", "lock_on")  # no breathing or micro-life while these play

# beat timings (the vocabulary's own defaults are in behaviors). Under the arm's own acceleration a
# look and back takes about 2 s, so a beat is one or two of those: a budget is a handful of beats.
EAGER_GLANCE_RAD = np.radians(14.0)
EAGER_LIFT_M = B.PERK_LIFT_M  # the full eager rises this much as it glances at the user
EAGER_HOLD_S = 0.2  # the glance holds this long
EAGER_PERK_HOLD_S = 0.2  # the short eager is the perk alone, held this long
CURIOUS_TILT_AT_S = 0.1
CURIOUS_TILT_RAD = np.radians(12.0)
CURIOUS_TILT_SIDE = 1
CURIOUS_TILT_HOLD_S = 0.1  # the short curious holds the tilt this long; the full one sniffs instead
CURIOUS_PECKS = 2
CURIOUS_LEAN_M = 0.01  # the full curious leans in this little before it pecks
WIND_UP_SHORT = 0.6  # the short wind-up draws back this share as far (and quicker: sqrt of it)
FOLLOW_SHORT_S = 0.9  # the short follow-through settles this fast
LOCK_STARE_S = 0.6  # the lock-on's stillness once it has leaned in (short: LOCK_STARE_SHORT_S)
LOCK_STARE_SHORT_S = 0.2
LOCK_LEAN_M = 0.025  # the full lock-on leans in this far toward the container (short: B.LOCK_LEAN_M)
PROUD_HOLD_S = 0.2  # proud looks at the user this long (in full a bob, or a tilt and a flourish)
PROUD_FLOURISH_S = 1.2  # ... or, in full, this long
PROUD_LOOK_MAX_RAD = np.radians(12.0)
ATTENTIVE_HOLD_S = 0.8  # the full proud keeps looking at the user this much longer: the attentive tail
SEARCH_SPOTS = B.SEARCH_SPOTS[:2]
SEARCH_SHORT_SPOTS = B.SEARCH_SPOTS[:1]
SORRY_QUICK = 0.8  # the full sorry sags this much quicker than a droop's design (paced to the caps)
SORRY_EYES_AFTER_S = 0.7  # the sheepish look up at the user starts as the droop arrives
SORRY_EYES_HOLD_S = 0.3  # held this long; then the eyes come down as the droop lifts (one move back)
SORRY_LINGER_S = 0.3  # the short sorry stays drooped this long
SORRY_SHORT_SAG = 0.5  # the short sorry: a droop this share as deep (and quicker), held SORRY_LINGER_S
SHAKE_GAP_S = 0.05  # the shake-off starts this long after the sorry
SHAKE_CYCLES, SHAKE_SHORT_CYCLES = 3.0, 2.0


@dataclass(frozen=True)
class Rule:
    """event: a task event (find_events). beat: what plays (BEATS). mood: the chip it shows. tier:
    BOUNDARY or IN_MOTION. keep: its priority (a tight budget shrinks and drops lower ones first);
    full_keep: its full version's (None: keep), so a beat can matter in short and its extra length
    little. reserve_s: held back for it from less important beats until its event comes (about what
    it usually adds). outcome: SUCCESS or FAILURE for an ending reaction (only one ending plays, so
    only the larger outcome is held back). free: it plays only where it adds no time (the arm waits
    there anyway). lock_on: it plays only under that lock_on mode (None: always). pause_s, slow: a
    rule on a SPAN event that stops the task at it for pause_s (short: half) and then runs the span
    at `slow` (the standoff lock-on)."""

    event: str
    beat: str
    mood: str | None = None
    tier: str = BOUNDARY
    keep: int = 0
    full_keep: int | None = None
    reserve_s: float = 0.0
    outcome: str | None = None
    free: bool = False
    lock_on: str | None = None
    pause_s: float = 0.0
    slow: float = 1.0

    def level(self, version):
        """The priority a version of this rule plays at."""
        return self.keep if version == SHORT or self.full_keep is None else self.full_keep


RULES = {
    # Priorities: the stare (7); the ending, success or failure (6); the eager perk and the puzzled
    # look (5); the curious tilt (4); the shake-off (4); the fillers, a wind-up or a follow-through
    # (1-2). A full version is worth less than its short one (full_keep), so a beat shrinks first.
    "grab_container": (
        Rule("requested", "eager", "EAGER", keep=5, full_keep=2),  # a perk; full: a glance at the user, rising
        Rule("transit_start", "wind_up", "WIND UP", keep=1),  # a draw-back against the coming move, before it
        Rule("transit", "lead_gaze", tier=IN_MOTION),  # look where it is going (rides on the move)
        Rule("arrived", "follow_through", "FOLLOW THROUGH", keep=2, full_keep=1, reserve_s=0.3),  # carry on, settle
        Rule("arrived_view", "curious", "CURIOUS", keep=4, full_keep=2, reserve_s=2.5),  # a tilt; full: and sniffs
        Rule("precision", "stillness", "FOCUS"),  # detect, the final approach, close, raise
        Rule("precise_next", "lock_on", "LOCK ON", keep=5, free=True, lock_on="view"),  # the stare, waiting to detect
        Rule("phase:open", "lock_on", "LOCK ON", keep=7, reserve_s=0.6, lock_on="view"),  # the stare, at the view pose
        Rule(  # stop at the standoff, stare, creep in (needs the approach in two calls)
            "final_approach", "lock_on", "LOCK ON", keep=7, reserve_s=1.6, lock_on="standoff", pause_s=1.6, slow=0.75
        ),
        Rule("raised", "proud", "PROUD", keep=6, full_keep=3, reserve_s=1.0, outcome=SUCCESS),  # full: a bob, attentive
        Rule("phase:aborted", "search", "PUZZLED", keep=5, full_keep=2, outcome=FAILURE),  # a look around the counter
        Rule("phase:aborted", "sorry", keep=6, full_keep=5, reserve_s=0.9, outcome=FAILURE),  # a droop; full: eyes up
        Rule("phase:aborted", "shake_off", "SHAKE IT OFF", keep=4, reserve_s=0.3, outcome=FAILURE),  # then shake off
    ),
}


@dataclass(frozen=True)
class Stage:
    """Samples i0..i1 where the functional motion is stopped between moves, outside PRECISION;
    before / after: what borders it ('move', 'precision', or the track's 'start' / 'end')."""

    i0: int
    i1: int
    before: str
    after: str

    @property
    def t0(self):
        return self.i0 * DT

    @property
    def t1(self):
        return self.i1 * DT

    @property
    def start_margin(self):
        return PRECISION_CLEAR_S if self.before == "precision" else STILL_MARGIN_S

    @property
    def end_margin(self):
        return PRECISION_CLEAR_S if self.after == "precision" else STILL_MARGIN_S


@dataclass(frozen=True)
class Event:
    """A task event at functional time t: AFTER (a beat starts there, in `stage`), BEFORE (a beat
    ends there and the arm then moves, in `stage`) or SPAN (over [t, t_end]: a transit, a precise
    run, the standoff stop)."""

    name: str
    t: float
    kind: str
    t_end: float = 0.0
    stage: int = -1


# ------------------------------------------------------------------------------------- events
def _index(t):
    return int(round(t / DT))


def _runs(mask):
    """[(i0, i1)] inclusive index runs where mask is set."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]])))
    return list(zip(edges[::2], edges[1::2] - 1))


def stages(track):
    """[Stage] of a track: runs where every joint is slower than STOP_RAD_S outside PRECISION, at
    least STOP_MIN_S long or across a phase change (a call boundary), and long enough for a pause."""
    still = (joint_speed(track) < STOP_RAD_S) & (track.zone != PRECISION)
    starts = [_index(p.t0) for p in track.phases]
    out = []
    for i0, i1 in _runs(still):
        long_enough = (i1 - i0 + 1) * DT >= STOP_MIN_S or any(i0 <= k <= i1 for k in starts)
        if not long_enough or i1 - i0 < 2 * HOLD_EDGE:
            continue
        before = "start" if i0 == 0 else ("precision" if track.zone[i0 - 1] == PRECISION else "move")
        after = "end" if i1 == track.n - 1 else ("precision" if track.zone[i1 + 1] == PRECISION else "move")
        out.append(Stage(int(i0), int(i1), before, after))
    return out


def stage_mask(track):
    """(N,) True on the samples of the track's stages."""
    mask = np.zeros(track.n, dtype=bool)
    for s in stages(track):
        mask[s.i0 : s.i1 + 1] = True
    return mask


def _phase(track, name):
    return next((p for p in track.phases if p.name == name), None)


def find_events(track, kin):
    """(stages, events in time order) of a functional track."""
    st = stages(track)
    in_stage = np.full(track.n, -1)
    for k, s in enumerate(st):
        in_stage[s.i0 : s.i1 + 1] = k
    events = []
    first = next((p for p in track.phases if p.name != "idle"), None)
    for k, s in enumerate(st):
        if first is not None and s.t0 <= first.t0 <= s.t1:
            events.append(Event("requested", first.t0, AFTER, stage=k))
        reach = s.t0 - PHASE_REACH_S if s.before == "precision" else s.t0  # a phase that starts as PRECISION ends
        events += [
            Event(f"phase:{p.name}", max(p.t0, s.t0), AFTER, stage=k) for p in track.phases if reach <= p.t0 <= s.t1
        ]
        if s.after == "precision":  # the stop ends as something precise starts (detect, at the view pose)
            events.append(Event("precise_next", s.t1, BEFORE, stage=k))
    eye = kin.fk_track(track.q, "eye")[0]
    for a, b in _runs((in_stage < 0) & (track.zone != PRECISION)):
        if np.linalg.norm(eye[a : b + 1] - eye[a], axis=1).max() >= TRANSIT_MIN_M:
            events += _transit_events(track, in_stage, a, b)
    events += _raised(track, st)
    events += [Event("precision", a * DT, SPAN, (b + 1) * DT) for a, b in _runs(track.zone == PRECISION)]
    return st, sorted(events, key=lambda e: (e.t, (AFTER, BEFORE, SPAN).index(e.kind)))


def _transit_events(track, in_stage, a, b):
    """A move (samples a..b, between stages or PRECISION) that is a transit: the transit, its
    transit_start in the stage before it and arrived (+ arrived_view in 'view') in the stage after
    it, or, when it runs into PRECISION inside 'approach', the final_approach stop for the standoff."""
    out = [Event("transit", a * DT, SPAN, (b + 1) * DT)]
    if a > 0 and in_stage[a - 1] >= 0:
        out.append(Event("transit_start", a * DT, BEFORE, stage=int(in_stage[a - 1])))
    if b + 1 < track.n and in_stage[b + 1] >= 0:
        k = int(in_stage[b + 1])
        out.append(Event("arrived", (b + 1) * DT, AFTER, stage=k))
        view = _phase(track, "view")
        if view is not None and view.t0 <= b * DT < view.t1:
            out.append(Event("arrived_view", (b + 1) * DT, AFTER, stage=k))
    approach = _phase(track, "approach")
    into_precision = b + 1 < track.n and track.zone[b + 1] == PRECISION
    if into_precision and approach is not None and approach.t0 <= (b + 1) * DT < approach.t1:
        precise_end = b + 1 + int(np.argmax(track.zone[b + 1 :] != PRECISION))
        out.append(Event("final_approach", (b + 1) * DT - LOCK_MARGIN_S, SPAN, precise_end * DT))
    return out


def _raised(track, st):
    """raised: in the first stage after 'raise' (once the gain is back up after PRECISION)."""
    lift = _phase(track, "raise")
    k = None if lift is None else next((k for k, s in enumerate(st) if s.t0 >= lift.t1 - 2 * DT), None)
    return [] if k is None else [Event("raised", st[k].t0, AFTER, stage=k)]


# --------------------------------------------------------------------------------- the beats
def _eager(ctx, t0, t1, short=False):
    """A perk; in full, a glance at the user that rises as it turns (one move); then back."""
    holding = bool(ctx.track.held[ctx.index(t0)])
    if short:
        return [B.perk(ctx, t0, t0 + B.paced_perk_s(ctx, t0, holding) + EAGER_PERK_HOLD_S, level=holding)]
    t_back = t0 + B.user_turn_s(ctx, t0, holding, EAGER_GLANCE_RAD, EAGER_LIFT_M) + EAGER_HOLD_S
    return [B.look_at_user(ctx, t0, t_back, holding, EAGER_GLANCE_RAD, lift_m=EAGER_LIFT_M)]


def _wind_kwargs(short):
    """A short wind-up draws back WIND_UP_SHORT as far, in sqrt(WIND_UP_SHORT) of the time."""
    share = WIND_UP_SHORT if short else 1.0
    quick = np.sqrt(share)
    return dict(
        shift_m=B.WIND_UP_SHIFT_M * share,
        turn_rad=B.WIND_UP_TURN_RAD * share,
        draw_s=B.WIND_UP_S * quick,
        release_s=B.WIND_UP_RELEASE_S * quick,
    )


def _wind_up(ctx, t0, t1, short=False):
    return [B.wind_up(ctx, t0, t1, **_wind_kwargs(short))]


def _wind_up_s(ctx, ev, short=False):
    return B.wind_up_s(ctx, ev.t, ev.t, **_wind_kwargs(short))


def _lead_gaze(ctx, t0, t1, short=False):
    """The lead joins while the arm is still at rest (it is exactly zero until LEAD_S before the
    move), so its fade-in adds nothing."""
    return [B.lead_gaze(ctx, t0 - B.LEAD_S - B.LEAD_FADE_S, t1)]


def _follow_through(ctx, t0, t1, short=False):
    return [B.follow_through(ctx, t0, settle_s=t1 - t0)]


def _follow_through_s(ctx, ev, short=False):
    return FOLLOW_SHORT_S if short else B.FOLLOW_SETTLE_S


def _curious(ctx, t0, t1, short=False):
    """A curious tilt and, in full, quick sniffs toward the container while tilted; the tilt back."""
    t_tilt = t0 + CURIOUS_TILT_AT_S
    tilt_s = B.paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, CURIOUS_TILT_RAD, 0.0, 0.0], B.TILT_TURN_S, B.TILT_OVERSHOOT)
    extra = []
    t_back = t_tilt + tilt_s + CURIOUS_TILT_HOLD_S
    if not short:
        sniff = B.sniff(ctx, ctx.container, t_tilt + tilt_s, pecks=CURIOUS_PECKS, lean_m=CURIOUS_LEAN_M)
        t_back = _end(sniff, t_tilt + tilt_s)
        extra.append(sniff)
    return [B.head_tilt(ctx, t_tilt, t_back, side=CURIOUS_TILT_SIDE, angle_rad=CURIOUS_TILT_RAD), *extra]


def _stillness(ctx, t0, t1, short=False):
    return []


def _lock_on(ctx, t0, t1, short=False):
    """A lean toward the container and a stare (the full one leans further and stares longer)."""
    return [B.lock_on(ctx, ctx.container, t0, t1, lean_m=B.LOCK_LEAN_M if short else LOCK_LEAN_M)]


def _lock_on_s(ctx, ev, short=False):
    stare, lean = (LOCK_STARE_SHORT_S, B.LOCK_LEAN_M) if short else (LOCK_STARE_S, LOCK_LEAN_M)
    return B.lock_on_s(ctx, ctx.container, ev.t, stare, lean_m=lean)


def _proud(ctx, t0, t1, short=False):
    """Proud (muted while holding) and, in full, the attentive tail: the look at the user held
    ATTENTIVE_HOLD_S longer before it turns back."""
    holding = bool(ctx.track.held[ctx.index(t0)])
    t_look = t0 + B.PROUD_LOOK_AT_S
    t_there = t_look + B.user_turn_s(ctx, t_look, holding, PROUD_LOOK_MAX_RAD, B.PROUD_LIFT_M)
    t_proud = t_there + (PROUD_HOLD_S if short else PROUD_FLOURISH_S)
    look_until = t_proud if short else t_proud + ATTENTIVE_HOLD_S
    flourish = None if short else "clack"
    proud = B.proud(ctx, t0, t_proud, look_until=look_until, flourish=flourish, max_turn_rad=PROUD_LOOK_MAX_RAD)
    if short:
        return [proud]
    return [proud, ctx.empty("attentive", [Label("ATTENTIVE", t_proud, _end(proud, look_until))])]


def _search(ctx, t0, t1, short=False):
    """After a failed detection: a puzzled look around the counter (one spot when short)."""
    return [B.search(ctx, t0, spots=SEARCH_SHORT_SPOTS if short else SEARCH_SPOTS)]


def _sorry(ctx, t0, t1, short=False):
    """After a failure: a droop and, in full, a sheepish look up at the user while drooped (puppy
    eyes), held a moment; then the eyes come down as the droop lifts, one move back."""
    if short:
        sag_m, look_rad = B.DROOP_SAG_M * SORRY_SHORT_SAG, B.DROOP_LOOK_DOWN_RAD * SORRY_SHORT_SAG
        quick = np.sqrt(SORRY_SHORT_SAG)
        sag = B.droop(ctx, t0, t0 + B.DROOP_S * quick + SORRY_LINGER_S, sag_m, look_rad, quick=quick)
        return [sag, ctx.empty("sorry", [Label("SORRY", t0, _end(sag, t0))])]
    t_eyes = t0 + SORRY_EYES_AFTER_S
    turn_s = (_end(B.puppy_eyes(ctx, t_eyes, t_eyes), t_eyes) - t_eyes) / 2.0  # up and straight back
    t_back = t_eyes + turn_s + SORRY_EYES_HOLD_S
    eyes = B.puppy_eyes(ctx, t_eyes, t_back)
    sag = B.droop(ctx, t0, t_back, quick=SORRY_QUICK)
    end = max(_end(eyes, t_back), _end(sag, t_back))
    return [sag, eyes, ctx.empty("sorry", [Label("SORRY", t0, end)])]


def _shake_off(ctx, t0, t1, short=False):
    """Once the sorry is over: shake it off, like a dog shaking off water."""
    return [B.shake_off(ctx, t0 + SHAKE_GAP_S, cycles=SHAKE_SHORT_CYCLES if short else SHAKE_CYCLES)]


def _end(c, default):
    """When a contribution stops moving (default if it never moves)."""
    span = c.span()
    return span[1] if span else default


@dataclass(frozen=True)
class Beat:
    """play(ctx, t0, t1, short) -> [Contribution], within [t0, t1]; length(ctx, event, short) ->
    its seconds, ctx on the functional track (None: self-timed, measured on a still copy)."""

    play: Callable
    length: Callable | None = None


BEATS = {
    "eager": Beat(_eager),
    "wind_up": Beat(_wind_up, _wind_up_s),
    "lead_gaze": Beat(_lead_gaze),
    "follow_through": Beat(_follow_through, _follow_through_s),
    "curious": Beat(_curious),
    "stillness": Beat(_stillness),
    "lock_on": Beat(_lock_on, _lock_on_s),
    "proud": Beat(_proud),
    "search": Beat(_search),
    "sorry": Beat(_sorry),
    "shake_off": Beat(_shake_off),
}


# ------------------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class Slot:
    """A rule at one of its events: the version that plays and its seconds (for a stop-and-creep
    rule, the stop)."""

    rule: Rule
    event: Event
    need_s: float
    short: bool = False


@dataclass
class Ledger:
    """The time budget as the events arrive: what the settled stops really added, which events
    have come, and when the task ended (its terminal phase), once it has."""

    budget: Budget | None
    rules: tuple
    spent: float = 0.0
    seen: set = field(default_factory=set)
    ended: float | None = None
    log: list = field(default_factory=list)

    def held_back(self, level):
        """Seconds kept for rules more important than `level` whose events have not come yet: each
        mid-task rule's reserve, plus the larger of the two endings' (an outcome's rules together:
        only one ending plays)."""
        mid, endings = 0.0, {}
        for r in self.rules:
            if r.keep <= level or r.event in self.seen:
                continue
            if r.outcome is None:
                mid += r.reserve_s
            else:
                endings[r.outcome] = endings.get(r.outcome, 0.0) + r.reserve_s
        return mid + max(endings.values(), default=0.0)

    def decide(self, rule, costs, gains):
        """The version of a beat to play (FULL, SHORT or None: drop it). costs: {version: what its
        stop would then add in all, with the short versions of the more important beats of this
        event still to decide}; gains: {version: what the beat itself adds to that} (both as far as
        the director knows the stop). A version fits if its stop still fits the time left and,
        unless it adds nothing, leaves what is held back for more important beats at later events.
        A free rule plays only a version that adds nothing; a version below the budget's min_keep
        never plays. Logged (with no budget, only a free rule's decision)."""
        budgeted = self.budget is not None
        left = self.budget.seconds - self.spent if budgeted else np.inf
        floor = getattr(self.budget, "min_keep", 0)
        version, adds, kept = None, 0.0, 0.0
        for v in VERSIONS:
            if (rule.free and gains[v] > EPS) or (budgeted and rule.level(v) < floor):
                continue
            kept = self.held_back(rule.level(v)) if budgeted else 0.0
            if costs[v] <= left + EPS and (gains[v] <= EPS or costs[v] + kept <= left + EPS):
                version, adds = v, gains[v]
                break
        if budgeted:
            what = f"{version} (+{adds:.1f} s)" if version else "dropped"
            if version is None and max(rule.keep, rule.level(FULL)) < floor:
                what += f" (the '{self.budget.name}' budget plays priorities {floor} and up)"
            self.log.append(f"{rule.beat} at {rule.event}: {what}; {left:.1f} s left, {kept:.1f} s held back")
        elif rule.free:
            self.log.append(
                f"{rule.beat} at {rule.event}: {version or 'dropped'} (it plays only where it adds no time)"
            )
        return version


@dataclass(frozen=True)
class StagePlan:
    """A stage's beats (AFTER ones in play order, BEFORE ones in table order) and the pause that
    makes room for them: at functional time `at`, hold_s long (0: none)."""

    stage: Stage
    after: tuple
    before: tuple
    at: float
    hold_s: float


@dataclass(frozen=True)
class Plan:
    """What the director decided: per stage, the spanning slots, the time-warp, its log, and the
    seconds the ledger charged for it all (spent)."""

    staged: tuple
    spanned: tuple
    holds: tuple
    slows: tuple
    log: tuple
    spent: float = 0.0


def _measure(beat, ev, func, sc, kin, seed, caps, short):
    """How long a self-timed beat lasts at the event's posture (on a still copy of it, in its zone)."""
    i = min(_index(ev.t), func.n - 1)
    still = constant_track(func.q[i], func.grip[i], MEASURE_S, zone=int(func.zone[i]))
    still.held[:] = func.held[i]
    ctx = B.Context.of(still, sc, kin, seed, caps)
    parts = beat.play(ctx, MEASURE_PAD_S, MEASURE_S, short)
    ends = [p.span()[1] for p in parts if p.span()] + [b.t1 for p in parts for b in p.beats]
    return max(ends, default=MEASURE_PAD_S) - MEASURE_PAD_S + DT


def _added_s(func, holds=(), slows=()):
    """Seconds a time-warp adds to the functional track."""
    return (len(warp_indices(func.n, holds, slows)) - func.n) * DT


def _accel_cap(func, t):
    """(7,) the default preset's joint acceleration cap at functional time t (a pause's ramps)."""
    return layer.caps_at(func.zone, min(_index(t), func.n - 1)).joint_accel


def _standoff(func, ev, rule, stop_s):
    """The stop-and-creep Slow of a standoff rule: stop at ev.t for stop_s, then run its span at rule.slow."""
    ramp = hold_ramp_s(func, ev.t, _accel_cap(func, ev.t), STANDOFF_RAMP_MIN_S)
    return Slow(ev.t, ev.t_end, rule.slow, ramp, stop_s)


def _stage_plan(stage, after, before, last_event=None):
    """Where a stage's beats go and the pause they need: AFTER beats one after another from their
    events (once the arm is still), BEFORE beats one after another up to the move but not before
    the stage's last event (last_event: the director learns then what comes next; the task's
    request, say), and a pause for whatever the stage's own stillness does not cover. The pause
    sits after the last event (so no event moves) and keeps its ramps inside the stage."""
    cursor, last = stage.start_margin, max(stage.t0, stage.t0 if last_event is None else last_event)
    for slot in after:
        cursor = max(cursor, slot.event.t - stage.t0) + slot.need_s
        last = max(last, slot.event.t)
    if before:
        cursor = max(cursor, last - stage.t0)
    need = cursor + sum(s.need_s for s in before) + stage.end_margin
    hold_s = max(0.0, need - (stage.t1 - stage.t0)) if after or before else 0.0
    at = min(max(last, stage.t0 + HOLD_EDGE * DT), stage.t1 - HOLD_EDGE * DT)
    return StagePlan(stage, tuple(after), tuple(before), round(at / DT) * DT, hold_s)


def _hold(func, sp):
    """The pause (timeline.Hold) a stage plan needs, eased over at least HOLD_RAMP_MIN_S."""
    return Hold(sp.at, sp.hold_s, hold_ramp_s(func, sp.at, _accel_cap(func, sp.at), HOLD_RAMP_MIN_S))


def _pause_s(func, hold_s):
    """Seconds a pause of hold_s adds to the task, its ramps included (the arm is stopped in a stage,
    so they take HOLD_RAMP_MIN_S): what the director charges for it."""
    return _added_s(func, holds=[Hold(func.duration / 2.0, hold_s, HOLD_RAMP_MIN_S)]) if hold_s > 0.0 else 0.0


def _until(stage, t, after):
    """The stage as if the arm moved on right after time t (at least its first sample), into `after`."""
    return Stage(stage.i0, max(stage.i0, _index(t)), stage.before, after)


def _known(stage, ev, ended):
    """The stage as far as the director knows it at event ev: all of it once the arm is about to
    move (a BEFORE event); once the task has ended (at `ended`), until its next command, assumed
    NEXT_COMMAND_S later (nothing precise follows); else as if the arm moved on right after now,
    into something precise (the director cannot know when, or what comes next)."""
    if ev.kind == BEFORE:
        return stage
    if ended is not None:
        return _until(stage, max(ev.t, ended + NEXT_COMMAND_S), "end")
    return _until(stage, ev.t, "precision")


@dataclass
class _Stop:
    """The beats decided so far at the stop the arm is in."""

    index: int
    after: list = field(default_factory=list)
    before: list = field(default_factory=list)

    def with_slots(self, slots=()):
        """(after, before) slot lists, with `slots` added, each to its kind."""
        return (
            self.after + [s for s in slots if s.event.kind != BEFORE],
            self.before + [s for s in slots if s.event.kind == BEFORE],
        )


def plan(func, found, rules, sc, kin, seed=0, budget=None, caps=DEFAULT_CAPS):
    """The beats and pauses for a functional track's (stages, events), decided event by event: each
    decision sees only the events so far and the time they have added. A stop is settled (charged
    what it really adds, ramps included) when an event after it arrives: the arm has moved on."""
    stages_, events = found
    ledger = Ledger(budget, tuple(rules))
    func_ctx = B.Context.of(func, sc, kin, seed, caps)
    staged, spanned, slows = [], [], []
    stop = None
    last_event = {}  # stage -> the time of its latest AFTER event so far

    def length(rule, ev, short):
        beat = BEATS[rule.beat]
        return beat.length(func_ctx, ev, short) if beat.length else _measure(beat, ev, func, sc, kin, seed, caps, short)

    def slot_at(cache, rule, ev, version):
        """A rule's slot at event ev in one version (cache: this event's, by (rule, version))."""
        if (rule, version) not in cache:
            cache[rule, version] = Slot(rule, ev, length(rule, ev, version == SHORT), version == SHORT)
        return cache[rule, version]

    def stage_plan(stage, index, after, before):
        ordered = (sorted(s, key=lambda x: _play_order(rules, x)) for s in (after, before))
        return _stage_plan(stage, *ordered, last_event.get(index))

    def adds(stop, stage, slots=()):
        """What the stop adds with its beats (and `slots`) if the stage is `stage`."""
        return _pause_s(func, stage_plan(stage, stop.index, *stop.with_slots(slots)).hold_s)

    def settle(stop):
        if stop is not None and (stop.after or stop.before):
            sp = stage_plan(stages_[stop.index], stop.index, stop.after, stop.before)
            ledger.spent += _added_s(func, holds=[_hold(func, sp)]) if sp.hold_s > 0.0 else 0.0
            staged.append(sp)

    for ev in events:
        if stop is not None and ev.stage != stop.index:
            settle(stop)
            stop = None
        if ev.kind == AFTER and ev.stage >= 0:
            last_event[ev.stage] = max(ev.t, last_event.get(ev.stage, ev.t))
        phase = ev.name.removeprefix("phase:")
        ledger.seen.add(ev.name)  # its rules are decided now, the more important first
        if ev.name.startswith("phase:") and phase in TERMINAL_PHASES and ledger.ended is None:
            ledger.ended = _phase(func, phase).t0  # the task is over: no more events will come
            ledger.seen.update(r.event for r in rules)
        todo = sorted((r for r in rules if r.event == ev.name), key=lambda r: -r.keep)
        cache = {}  # (rule, version) -> its slot at this event
        for k, rule in enumerate(todo):
            if ev.kind == SPAN:
                slot, slow = _span_slot(rule, ev, func, ledger)
                spanned += [slot] if slot else []
                slows += [slow] if slow else []
            elif func.zone[min(_index(ev.t), func.n - 1)] != PRECISION:
                stop = stop or _Stop(ev.stage)
                stage = _known(stages_[ev.stage], ev, ledger.ended)
                options = {v: slot_at(cache, rule, ev, v) for v in VERSIONS}
                costs, gains = {}, {}
                for v, slot in options.items():  # with the short versions of the more important beats to come here
                    ahead = [slot_at(cache, r, ev, SHORT) for r in todo[k + 1 :] if r.keep > rule.level(v)]
                    costs[v] = adds(stop, stage, [slot, *ahead])
                    gains[v] = costs[v] - adds(stop, stage, ahead)
                version = ledger.decide(rule, costs, gains)
                if version is not None:
                    stop.after, stop.before = stop.with_slots([options[version]])
    settle(stop)
    holds = tuple(_hold(func, p) for p in staged if p.hold_s > 0.0)
    return Plan(tuple(staged), tuple(spanned), holds, tuple(slows), tuple(ledger.log), ledger.spent)


def _play_order(rules, slot):
    """A slot's place in play order: its event's time, then its rule's place in the table."""
    return slot.event.t, rules.index(slot.rule)


def _span_slot(rule, ev, func, ledger):
    """(slot or None, Slow or None) for a rule on a SPAN event: a label or a moving beat (no time),
    or a stop-and-creep (decided against the budget, charged at once: its cost is known)."""
    if rule.slow >= 1.0:
        return Slot(rule, ev, 0.0), None
    if func.zone[min(_index(ev.t), func.n - 1)] == PRECISION:
        return None, None
    stops = {FULL: rule.pause_s, SHORT: rule.pause_s / 2.0}
    costs = {v: _added_s(func, slows=[_standoff(func, ev, rule, s)]) for v, s in stops.items()}
    version = ledger.decide(rule, costs, costs)
    if version is None:
        return None, None
    ledger.spent += costs[version]
    return Slot(rule, ev, stops[version], version == SHORT), _standoff(func, ev, rule, stops[version])


# ---------------------------------------------------------------------------------- windows
def windows(planned, warped):
    """[(slot, (t0, t1))] output-time windows on the warped track. A stage's AFTER beats run one
    after another from their events (once the arm is still), its BEFORE beats one after another up
    to the move; a standoff lock-on fills its stop; a label or a moving beat spans its event. Every
    window but a label's ends PRECISION_CLEAR_S before the next precise sample."""
    out = []
    for sp in planned.staged:
        s = sp.stage
        start = output_time(warped, s.t0)
        cursor = start + s.start_margin
        for slot in sp.after:
            cursor = max(cursor, output_time(warped, slot.event.t))
            out.append((slot, (cursor, cursor + slot.need_s)))
            cursor += slot.need_s
        end = output_time(warped, s.t1) - s.end_margin
        for slot in reversed(sp.before):
            out.append((slot, (end - slot.need_s, end)))
            end -= slot.need_s
    for slot in planned.spanned:
        ev = slot.event
        if slot.rule.slow < 1.0:
            span = still_span(warped, ev.t)
            window = span if span is not None else (output_time(warped, ev.t),) * 2
        else:
            window = (output_time(warped, ev.t), output_time(warped, ev.t_end))
        out.append((slot, window if slot.rule.beat == "stillness" else _clipped(warped, window)))
    return out


def _clipped(warped, window):
    """window ended PRECISION_CLEAR_S before the next precise sample after its start."""
    t0, t1 = window
    ahead = np.nonzero((warped.zone == PRECISION) & (warped.t >= t0))[0]
    if ahead.size:
        t1 = min(t1, ahead[0] * DT - PRECISION_CLEAR_S)
    return t0, max(t0, t1)


# --------------------------------------------------------------------------------- directing
def direct(run, sc=None, kin=None, seed=0, **options):
    """(plain, puppy) for a real run: its functional track, and the Puppy Mode track the RULES make
    of it (time-warped, the beats composed on top). options: direct_track's."""
    sc = sc or scene_mod.build()
    kin = kin or sc.kin()
    func = runs.functional_track(run, kin)
    puppy, _ = direct_track(func, run.capability, sc, kin, seed, **options)
    return func, puppy


def direct_track(
    func,
    capability,
    sc,
    kin,
    seed=0,
    tiers=(BOUNDARY,),
    budget=DEFAULT_BUDGET,
    lock_on="view",
    caps=DEFAULT_CAPS,
    skip=(),
):
    """(puppy, contributions) for a functional track of `capability`. tiers: which rules play
    (BOUNDARY; add IN_MOTION to opt in). budget: a BUDGETS name, or None for no limit. lock_on:
    'view' or 'standoff' (see the module doc). caps: the layer.CAP_PRESETS preset. skip: beats
    (BEATS names) whose rules are left out, e.g. to show one beat on its own."""
    if lock_on not in LOCK_ON_MODES:
        raise ValueError(f"lock_on is one of {LOCK_ON_MODES}, not {lock_on!r}")
    unknown = set(skip) - set(BEATS)
    if unknown:
        raise ValueError(f"skip names beats in BEATS, not {sorted(unknown)}")
    rules = tuple(
        r for r in RULES.get(capability, ()) if r.tier in tiers and r.lock_on in (None, lock_on) and r.beat not in skip
    )
    limit = None if budget is None else BUDGETS[budget]
    planned = plan(func, find_events(func, kin), rules, sc, kin, seed, limit, caps)
    warped = warp(func, planned.holds, planned.slows)
    ctx = B.Context.of(warped, sc, kin, seed, caps)
    still = stage_mask(warped)
    precise = B.precision_fence(warped)
    between = np.minimum(precise, B.fence(warped, ~still, BOUNDARY_FADE_S))
    parts, quiet = [], np.zeros(warped.n, dtype=bool)
    for slot, (t0, t1) in windows(planned, warped):
        for c in BEATS[slot.rule.beat].play(ctx, t0, t1, slot.short):
            if slot.rule.tier == IN_MOTION:
                fenced(c, precise, "faded out before PRECISION")
            else:
                fenced(c, between, "faded out where the arm moves or before PRECISION")
            parts.append(c)
        if slot.rule.mood:
            parts.append(ctx.empty(slot.rule.beat, [Label(slot.rule.mood, t0, t1)]))
        if slot.rule.beat in QUIET_BEATS:
            quiet |= (warped.t >= t0) & (warped.t < t1)
    parts += background(B.Context.of(warped, sc, kin, seed + BACKGROUND_SEED, caps), quiet | ~still)
    puppy = layer.compose(warped, parts, kin, caps)
    puppy.budget = limit
    puppy.tier = IN_MOTION if IN_MOTION in tiers else (STANDOFF if lock_on == "standoff" else BOUNDARY)
    added = warped.duration - func.duration
    budget_note = f" of the {limit.seconds:.1f} s '{limit.name}' budget" if limit else ""
    puppy.log = [
        f"time: plain {func.duration:.1f} s -> {warped.duration:.1f} s (+{added:.1f} s{budget_note})",
        *planned.log,
        *puppy.log,
    ]
    return puppy, parts


def excerpt(plain, puppy, t0, t1):
    """(plain, puppy) cut to the functional window [t0, t1] (seconds of the plain task), both
    starting at 0: the plain track's samples there, and the Puppy Mode samples that play them (its
    warped functional track and its source cut alike, so every check still holds; its added time is
    the window's)."""
    i0, i1 = _index(t0), min(_index(t1), plain.n - 1)
    base = i0 * DT
    src = _cut(plain, np.arange(i0, i1 + 1), base, base)
    keep = np.nonzero((puppy.tau >= base - EPS) & (puppy.tau <= i1 * DT + EPS))[0]
    out_base = float(keep[0]) * DT
    func = _cut(puppy.func, keep, out_base, base)
    func.source = src
    cut = _cut(puppy, keep, out_base, base)
    cut.func, cut.gain = func, puppy.gain[keep]
    cut.caps, cut.budget, cut.tier = puppy.caps, puppy.budget, puppy.tier
    cut.log = [f"excerpt: {t0:.1f}-{t1:.1f} s of the task", *puppy.log]
    return src, cut


def _cut(track, idx, t_base, tau_base):
    """Samples idx of a track (contiguous), its clock and labels starting at t_base and its tau at tau_base."""
    duration = (len(idx) - 1) * DT

    def moved(labels):
        out = [Label(b.name, max(0.0, float(b.t0) - t_base), min(duration, float(b.t1) - t_base)) for b in labels]
        return [b for b in out if b.t1 > b.t0]

    return Track(
        q=track.q[idx].copy(),
        grip=track.grip[idx].copy(),
        zone=track.zone[idx].copy(),
        held=track.held[idx].copy(),
        tau=track.tau[idx] - tau_base,
        phases=moved(track.phases),
        beats=moved(track.beats),
        name=track.name,
    )


def fenced(c, fence, note):
    """c faded by `fence` (0 where it must not act, easing to 1 away from there), noted when that
    changes it: the director's guarantee that no beat reaches into PRECISION, and that no boundary
    beat rides on a move."""
    if np.any(c.active() & (fence < 1.0)):
        c.notes.append(note)
    c.head *= fence[:, None]
    c.joints *= fence[:, None]
    c.grip *= fence


def background(ctx, avoid):
    """Breathing and micro-life wherever the arm rests between moves (they fade out before
    PRECISION on their own), except where `avoid` (N,) asks for stillness (a wind-up, a stare, any
    sample outside a stage)."""
    return [B.breathe(ctx, avoid=avoid), B.micro_life(ctx, avoid=avoid)]
