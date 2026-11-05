"""The director: maps a real run's task events to Puppy Mode beats.

RULES is the whole mapping, one readable table per capability. Each rule names a task event, the
beat that plays at it, how long the task pauses for it, an optional slow-in and the mood chip.
find_events() reads the events off the functional track (its phases, zones and joint speed).

The director then plans the pauses: a timeline.Hold where a beat needs stillness that the arm does
not already give, a timeline.Slow (stopping first) for the lock-on. It warps the functional track,
places the beats on the warped clock with the background life (breathe, micro_life), and composes.

A phase the table does not know is PRECISION (runs.zone_of) and gets no beat. Beats only play at
events outside PRECISION, in windows that end PRECISION_CLEAR_S before the next precise sample,
and every beat is fenced (faded to exactly zero before any PRECISION sample, noted in the log if
that ever changes it), so the director never puts a spatial offset on a PRECISION sample. The
layer zeroes PRECISION on top of that.
"""

from dataclasses import dataclass

import numpy as np

from . import behaviors as B
from . import layer, runs
from . import scene as scene_mod
from .layer import EXPR_JOINT_ACCEL
from .timeline import (
    DT,
    FREE,
    PRECISION,
    Hold,
    Label,
    Slow,
    constant_track,
    hold_ramp_s,
    joint_speed,
    output_time,
    quiet_time,
    still_span,
    warp,
)

MOVING_RAD_S = 0.05  # the arm moves when a joint is faster than this
TRANSIT_MIN_M = 0.05  # a move of the eye shorter than this is not a transit
MERGE_GAP_S = 0.3  # moves separated by a shorter pause are one transit
PRECISION_CLEAR_S = 0.4  # beats end this long before a precise sample (the gain ramp, plus margin)
LOCK_MARGIN_S = 0.06  # the lock-on stops this long before the first precise sample (on a FREE one)
QUIET_LOOKBACK_S = 0.3  # a wind-up hold goes at the quietest moment this long before the move
HOLD_RAMP_MIN_S = 0.3
MEASURE_PAD_S = 0.5  # a self-timed beat is measured on a still track from this time
MEASURE_S = 40.0
BACKGROUND_SEED = 3

# beat timings (the vocabulary's own defaults are in behaviors)
EAGER_LOOK_AT_S = 0.3
EAGER_GLANCE_RAD = np.radians(14.0)
EAGER_NODS = 1
CURIOUS_TILT_AT_S = 0.5  # after the arrival (the follow-through is settling)
CURIOUS_TILT_RAD = np.radians(16.0)
CURIOUS_TILT_SIDE = 1
CURIOUS_PECKS = 2
ATTENTIVE_HOLD_S = 1.4
PROUD_S = 3.4
SORRY_GAP_S = 0.2
SORRY_SEARCH_SPOTS = B.SEARCH_SPOTS[:2]
SORRY_EYES_AFTER_S = 1.8  # the sheepish look up starts once the droop is down
SORRY_EYES_HOLD_S = 0.8
SORRY_LINGER_S = 0.4  # drooped a moment longer once the eyes are down again


@dataclass(frozen=True)
class Rule:
    """event: a task event (find_events). beat: what plays (BEATS). pause_s: the stillness it
    plays in, the task held for whatever the arm does not already give (None: as long as the beat
    takes; 0: no pause, it plays over the task's own motion). slow: the task's speed over the
    event's span afterwards (a slow-in). mood: the chip it shows."""

    event: str
    beat: str
    pause_s: float | None = 0.0
    slow: float = 1.0
    mood: str | None = None


RULES = {
    "grab_container": (
        Rule("requested", "eager", pause_s=None, mood="EAGER"),  # perk, glance at the user, a nod
        Rule("transit_start", "wind_up", pause_s=0.8),  # draw back against the coming move
        Rule("transit", "lead_gaze"),  # look where it is going
        Rule("transit_end", "follow_through"),  # carry on a little and settle
        Rule("arrived_view", "curious", pause_s=None, mood="CURIOUS"),  # a tilt, a sniff at the container
        Rule("precision", "stillness", mood="FOCUS"),  # detect, the final approach, close, raise
        Rule("final_approach", "lock_on", pause_s=1.6, slow=0.75, mood="FOCUS"),  # stare, then creep in
        Rule("raised", "proud", pause_s=PROUD_S, mood="PROUD"),  # muted while holding
        Rule("finished", "attentive", pause_s=None, mood="ATTENTIVE"),  # keep looking at the user
        Rule("aborted", "sorry", pause_s=None),  # search, droop + puppy eyes, shake it off
    ),
}
BEFORE = {"transit_start"}  # beats that play before their event (the rest play after it)


@dataclass(frozen=True)
class Event:
    """A task event on the functional track: functional time t, the end of its span (a transit,
    the final approach, a precision run), the stillness the arm already gives for a beat there
    (after t; before it for BEFORE events) and how long a beat waits after t before it starts
    (settle_s: the zone gain still ramps up right after PRECISION)."""

    name: str
    t: float
    t_end: float
    room_s: float = 0.0
    settle_s: float = 0.0


# ------------------------------------------------------------------------------------- events
def find_events(track, kin):
    """The task events on a functional track, in time order."""
    moving = joint_speed(track) > MOVING_RAD_S
    events = []
    first = next((p for p in track.phases if p.name != "idle"), None)
    if first is not None:
        i = _index(first.t0)
        events.append(Event("requested", first.t0, first.t0, _room_after(track, moving, i)))
    for i0, i1 in _transits(track, kin, moving):
        events += _transit_events(track, moving, i0, i1)
    for a, b in _runs(track.zone == PRECISION):
        events.append(Event("precision", a * DT, (b + 1) * DT))
    events += _ending_events(track, moving)
    return sorted(events, key=lambda e: e.t)


def _index(t):
    return int(round(t / DT))


def _runs(mask):
    """[(i0, i1)] inclusive index runs where mask is set."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]])))
    return list(zip(edges[::2], edges[1::2] - 1))


def _phase(track, name):
    return next((p for p in track.phases if p.name == name), None)


def _room_after(track, moving, i):
    """Seconds a beat can play from sample i without a hold: the arm still, and clear of the next
    precise sample by PRECISION_CLEAR_S."""
    ahead = np.nonzero(moving[i:] | (track.zone[i:] == PRECISION))[0]
    if not ahead.size:
        return (track.n - i) * DT
    clear = PRECISION_CLEAR_S if track.zone[i + ahead[0]] == PRECISION else 0.0
    return max(0.0, ahead[0] * DT - clear)


def _room_before(track, moving, i):
    """Seconds the arm has been still, and out of PRECISION, before sample i."""
    behind = np.nonzero((moving[:i] | (track.zone[:i] == PRECISION))[::-1])[0]
    return (behind[0] if behind.size else i) * DT


def _transits(track, kin, moving):
    """[(i0, i1)] runs of motion (short pauses merged) that start in FREE and move the eye at
    least TRANSIT_MIN_M."""
    merged = []
    for a, b in _runs(moving):
        if merged and (a - merged[-1][1]) * DT < MERGE_GAP_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    eye = kin.fk_track(track.q, "eye")[0]
    return [
        (a, b)
        for a, b in merged
        if track.zone[a] == FREE and np.linalg.norm(eye[a : b + 1] - eye[a], axis=1).max() >= TRANSIT_MIN_M
    ]


def _transit_events(track, moving, i0, i1):
    """transit_start, transit and either transit_end (+ arrived_view in 'view') or, when it runs
    into PRECISION inside 'approach', final_approach."""
    t0 = i0 * DT
    precise = np.nonzero(track.zone[i0 : i1 + 1] == PRECISION)[0]
    if precise.size:
        t_boundary = (i0 + precise[0]) * DT
        out = [
            Event("transit_start", t0, t_boundary, _room_before(track, moving, i0)),
            Event("transit", t0, t_boundary),
        ]
        approach = _phase(track, "approach")
        if approach is not None and approach.t0 <= t_boundary < approach.t1:
            out.append(Event("final_approach", t_boundary - LOCK_MARGIN_S, (i1 + 1) * DT))
        return out
    t1 = i1 * DT
    out = [
        Event("transit_start", t0, t1, _room_before(track, moving, i0)),
        Event("transit", t0, t1),
        Event("transit_end", t1, t1, _room_after(track, moving, i1 + 1)),
    ]
    view = _phase(track, "view")
    if view is not None and view.t0 <= t1 < view.t1:
        out.append(Event("arrived_view", t1, t1, _room_after(track, moving, i1 + 1)))
    return out


def _first_free(track, t):
    """The first FREE sample at or after time t, one sample further in when it follows a
    PRECISION one (a pause there stands still within half a sample of it, which must be FREE)."""
    free = np.nonzero((track.t >= t - 1e-9) & (track.zone == FREE))[0]
    if not free.size:
        return track.n - 1
    i = int(free[0])
    return min(i + 1, track.n - 1) if i > 0 and track.zone[i - 1] == PRECISION else i


def _ending_events(track, moving):
    """raised (the first FREE sample after 'raise'), finished ('done': with raised when both),
    aborted ('aborted')."""
    out = []
    lift = _phase(track, "raise")
    i_raised = _first_free(track, lift.t1 - DT) if lift is not None else None
    if i_raised is not None:
        out.append(_resting_event(track, moving, "raised", i_raised))
    for phase, name in (("done", "finished"), ("aborted", "aborted")):
        p = _phase(track, phase)
        if p is None:
            continue
        i = _first_free(track, p.t0)
        if name == "finished" and i_raised is not None:
            i = max(i, i_raised)
        out.append(_resting_event(track, moving, name, i))
    return out


def _resting_event(track, moving, name, i):
    """An event at sample i where the arm rests, waiting out the gain ramp after PRECISION."""
    before = np.nonzero(track.zone[:i] == PRECISION)[0]
    since = (i - before[-1]) * DT if before.size else np.inf
    settle = max(0.0, PRECISION_CLEAR_S - since)
    return Event(name, i * DT, i * DT, max(0.0, _room_after(track, moving, i) - settle), settle)


# --------------------------------------------------------------------------------- the beats
def _eager(ctx, t0, t1):
    """Perk, glance at the user, a nod while looking, look back."""
    holding = bool(ctx.track.held[ctx.index(t0)])
    t_look = t0 + EAGER_LOOK_AT_S
    turn_s = B.user_turn_s(ctx, t_look, holding, EAGER_GLANCE_RAD)
    nod = B.nod(ctx, t_look + turn_s, dips=EAGER_NODS)
    t_back = _end(nod, t_look + turn_s)
    look = B.look_at_user(ctx, t_look, t_back, holding, EAGER_GLANCE_RAD)
    return [B.perk(ctx, t0, t_back, level=holding), look, nod]


def _wind_up(ctx, t0, t1):
    return [B.wind_up(ctx, t0, t1)]


def _lead_gaze(ctx, t0, t1):
    """The lead joins while the arm is still at rest (it is exactly zero until LEAD_S before the
    move), so its fade-in adds nothing."""
    return [B.lead_gaze(ctx, t0 - B.LEAD_S - B.LEAD_FADE_S, t1)]


def _follow_through(ctx, t0, t1):
    return [B.follow_through(ctx, t0)]


def _curious(ctx, t0, t1):
    """A curious tilt, a sniff toward the container while tilted, the tilt back."""
    t_tilt = t0 + CURIOUS_TILT_AT_S
    tilt_s = B.paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, CURIOUS_TILT_RAD, 0.0, 0.0], B.TILT_TURN_S, B.TILT_OVERSHOOT)
    sniff = B.sniff(ctx, ctx.container, t_tilt + tilt_s, pecks=CURIOUS_PECKS)
    tilt = B.head_tilt(ctx, t_tilt, _end(sniff, t_tilt + tilt_s), side=CURIOUS_TILT_SIDE, angle_rad=CURIOUS_TILT_RAD)
    return [tilt, sniff]


def _stillness(ctx, t0, t1):
    return []


def _lock_on(ctx, t0, t1):
    return [B.lock_on(ctx, ctx.container, t0, t1)]


def _proud(ctx, t0, t1):
    """Proud, its look at the user running on to t1 (where 'attentive' takes it over)."""
    return [B.proud(ctx, t0, t1, look_until=t1)]


def _attentive(ctx, t0, t1):
    """Keep looking at the user (it runs on from proud's look), then look back."""
    holding = bool(ctx.track.held[ctx.index(t0)])
    return [B.look_at_user(ctx, t0, t0 + ATTENTIVE_HOLD_S, holding)]


def _sorry(ctx, t0, t1):
    """After a failure: a puzzled search, a droop with a sheepish look up at the user, a shake."""
    look = B.search(ctx, t0, spots=SORRY_SEARCH_SPOTS, beat="PUZZLED")
    t_droop = _end(look, t0) + SORRY_GAP_S
    t_eyes = t_droop + SORRY_EYES_AFTER_S
    eyes = B.puppy_eyes(ctx, t_eyes, t_eyes + SORRY_EYES_HOLD_S)
    t_lift = _end(eyes, t_eyes) + SORRY_LINGER_S
    sag = B.droop(ctx, t_droop, t_lift)
    t_shake = _end(sag, t_lift) + SORRY_GAP_S
    shake = B.shake_off(ctx, t_shake, beat="SHAKE IT OFF")
    sorry = ctx.empty("sorry", [Label("SORRY", t_droop, t_shake)])
    return [look, sag, eyes, sorry, shake]


def _end(c, default):
    """When a contribution stops moving (default if it never moves)."""
    span = c.span()
    return span[1] if span else default


BEATS = {
    "eager": _eager,
    "wind_up": _wind_up,
    "lead_gaze": _lead_gaze,
    "follow_through": _follow_through,
    "curious": _curious,
    "stillness": _stillness,
    "lock_on": _lock_on,
    "proud": _proud,
    "attentive": _attentive,
    "sorry": _sorry,
}


# ------------------------------------------------------------------------------------- plan
@dataclass(frozen=True)
class Slot:
    """A rule at one of its events, with the seconds of stillness its beat gets (0: spanned)."""

    rule: Rule
    event: Event
    need_s: float


def _measure(rule, event, func, sc, kin, seed):
    """How long a self-timed beat lasts at the event's posture (on a still copy of it)."""
    i = min(_index(event.t), func.n - 1)
    still = constant_track(func.q[i], func.grip[i], MEASURE_S)
    still.held[:] = func.held[i]
    ctx = B.Context.of(still, sc, kin, seed)
    parts = BEATS[rule.beat](ctx, MEASURE_PAD_S, MEASURE_S)
    ends = [p.span()[1] for p in parts if p.span()] + [b.t1 for p in parts for b in p.beats]
    return max(ends, default=MEASURE_PAD_S) - MEASURE_PAD_S + DT


def plan(func, events, rules, sc, kin, seed=0):
    """(slots, holds, slows): the rules matched to the events and the pauses they need."""
    by_event = {r.event: r for r in rules}
    slots = []
    for ev in events:
        rule = by_event.get(ev.name)
        if rule is None or not _open(func, ev):
            continue
        need = _measure(rule, ev, func, sc, kin, seed) if rule.pause_s is None else rule.pause_s
        slots.append(Slot(rule, ev, need))
    slows = [
        Slow(s.event.t, s.event.t_end, s.rule.slow, _ramp(func, s.event.t), s.need_s)
        for s in slots
        if s.rule.slow < 1.0
    ]
    holds, free_from = [], 0.0  # functional stillness before free_from is used by an earlier beat
    for t, group in _paused_groups(slots):
        need = group[0].event.settle_s + sum(s.need_s for s in group)
        before = group[0].event.name in BEFORE
        room = min(group[0].event.room_s, max(0.0, t - free_from)) if before else group[0].event.room_s
        if need > room:
            at = quiet_time(func, max(free_from, t - QUIET_LOOKBACK_S), t) if before else t
            holds.append(Hold(at, need - room, _ramp(func, at)))
        free_from = t if before else t + min(need, room)
    return slots, _merged(holds), slows


def _open(func, ev):
    """Whether a beat may play at an event: its sample is not PRECISION (for a BEFORE event, the
    sample before it). Precision labels are always placed: they carry no motion."""
    if ev.name == "precision":
        return True
    i = _index(ev.t) - (1 if ev.name in BEFORE else 0)
    return func.zone[min(max(i, 0), func.n - 1)] != PRECISION


def _ramp(func, t):
    """A pause's ramp at functional time t: long enough to stop the arm within the accel cap."""
    return hold_ramp_s(func, t, EXPR_JOINT_ACCEL, HOLD_RAMP_MIN_S)


def _merged(holds):
    """Holds closer than their ramps merged into one at the first (the arm is still between them:
    both sit at quiet moments, a BEFORE one never ahead of the stillness an earlier beat uses), so
    a second hold never lands after a move has started."""
    out = []
    for h in sorted(holds, key=lambda h: h.at):
        if out and h.at - out[-1].at < out[-1].ramp_s + h.ramp_s:
            last = out.pop()
            h = Hold(last.at, last.hold_s + h.hold_s, max(last.ramp_s, h.ramp_s))
        out.append(h)
    return out


def _paused_groups(slots):
    """[(t, slots)] of the slots that pause the task (not slow-ins), grouped by event time."""
    groups = {}
    for s in slots:
        if s.need_s > 0.0 and s.rule.slow >= 1.0:
            groups.setdefault(round(s.event.t / DT), []).append(s)
    return [(k * DT, g) for k, g in sorted(groups.items())]


def windows(slots, warped):
    """[(slot, (t0, t1))] output-time windows on the warped track. Slots at one event time play one
    after another, in table order, after the event's settle_s; BEFORE slots end where their event
    starts. Every window but a precision label ends PRECISION_CLEAR_S before the warped track's
    next precise sample, and a transit's ends where a lock-on stare in it begins."""
    out, stacked = [], {}
    for slot in slots:
        ev = slot.event
        start = output_time(warped, ev.t)
        if slot.rule.slow < 1.0:
            span = still_span(warped, ev.t)
            window = span if span is not None else (start, start)
        elif ev.name in BEFORE:
            window = (start - slot.need_s, start)
        elif slot.need_s > 0.0:
            key = round(ev.t / DT)
            start += stacked.setdefault(key, ev.settle_s)
            stacked[key] += slot.need_s
            window = (start, start + slot.need_s)
        else:
            window = (start, output_time(warped, ev.t_end))
        out.append((slot, window))
    stares = [w for s, w in out if s.rule.slow < 1.0]
    return [(s, w if s.event.name == "precision" else _clipped(warped, w, stares)) for s, w in out]


def _clipped(warped, window, stares):
    """window ended PRECISION_CLEAR_S before the next precise sample after its start, and at the
    start of a stare that begins inside it."""
    t0, t1 = window
    ahead = np.nonzero((warped.zone == PRECISION) & (warped.t >= t0))[0]
    if ahead.size:
        t1 = min(t1, ahead[0] * DT - PRECISION_CLEAR_S)
    t1 = min([t1] + [a for a, _ in stares if t0 < a < t1])
    return t0, max(t0, t1)


# --------------------------------------------------------------------------------- directing
def direct(run, sc=None, kin=None, seed=0):
    """(plain, puppy) for a real run: its functional track, and the Puppy Mode track the RULES make
    of it (time-warped, with the beats composed on top)."""
    sc = sc or scene_mod.build()
    kin = kin or sc.kin()
    func = runs.functional_track(run, kin)
    puppy, _ = direct_track(func, run.capability, sc, kin, seed)
    return func, puppy


def direct_track(func, capability, sc, kin, seed=0):
    """(puppy, contributions) for a functional track of `capability`: see direct()."""
    rules = RULES.get(capability, ())
    slots, holds, slows = plan(func, find_events(func, kin), rules, sc, kin, seed)
    warped = warp(func, holds, slows)
    ctx = B.Context.of(warped, sc, kin, seed)
    parts, quiet = [], np.zeros(warped.n, dtype=bool)
    for slot, (t0, t1) in windows(slots, warped):
        parts += BEATS[slot.rule.beat](ctx, t0, t1)
        if slot.rule.mood:
            parts.append(ctx.empty(slot.rule.beat, [Label(slot.rule.mood, t0, t1)]))
        if slot.rule.beat in ("wind_up", "lock_on"):
            quiet |= (warped.t >= t0) & (warped.t < t1)
    fence = B.precision_fence(warped)
    for c in parts:
        fenced(c, fence)
    parts += background(B.Context.of(warped, sc, kin, seed + BACKGROUND_SEED), quiet)  # fade out on their own
    return layer.compose(warped, parts, kin), parts


def fenced(c, fence):
    """c faded by `fence` (0 on PRECISION samples, easing to 1 away from them), noted when that
    changes it: the director's guarantee that no beat reaches into PRECISION."""
    if np.any(c.active() & (fence < 1.0)):
        c.notes.append("faded out before PRECISION")
    c.head *= fence[:, None]
    c.joints *= fence[:, None]
    c.grip *= fence


def background(ctx, quiet):
    """Breathing and micro-life wherever the arm rests in FREE (they fade out before PRECISION),
    except where `quiet` (N,) asks for stillness (a wind-up, a stare)."""
    avoid = quiet | (joint_speed(ctx.track) > MOVING_RAD_S)
    return [B.breathe(ctx, avoid=avoid), B.micro_life(ctx, avoid=avoid)]
