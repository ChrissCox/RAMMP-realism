"""The online director: which short expressive beats play at a task's events, within a time budget.

A capability reports its events as they happen (TaskRequested, Phase, BeforeMove, AfterMove,
Succeeded, Failed) and gets back the beats to play right then: the arm is stopped whenever an event
arrives, so every beat is self-contained, starts and ends exactly at the arm's measured posture
q_now, and has passed check_beat. Nothing here talks to a robot, and nothing looks ahead: a decision
uses only the events so far.

data/rules.yaml says which beat each event gets and how much it matters (keep). Each beat has a full
and a short version. A beat costs its duration against the preset's budget (BUDGETS). It plays in
full if that fits what is left less a reserve for what matters more and may still come: the larger of
the short success reaction and the short failure reaction (their rules above this one's keep), plus
the short versions of the other higher-priority rules still possible. Otherwise it plays short, otherwise not at all. The subtle
preset plays only rules with keep >= SUBTLE_MIN_KEEP.

Nothing spatial comes from a Phase the zone table maps to PRECISION (an unknown phase is PRECISION),
from a move made while the current phase is PRECISION, from any event while the arm holds something
and the head is near the person, or from the AfterMove of a move that ended at a precise pose (the
arm must stay settled for the grasp or the press). Failed(False) gets nothing.

Settle time. Under joint impedance a held container makes the arm spring and settle slowly, so a
beat that ends just before a PRECISION move must leave it still for longer: while the arm holds
something, the still tail (the hold after its last motion) of a beat played after a move (AfterMove) is at least SETTLE_S for
the task's control mode and gains (soft 2.3 s, medium/default 1.1 s, stiff 0.1 s), else the usual
(TAIL_S) tail; the tail is never shorter than TAIL_S. Only a beat after a move (AfterMove) or
before a move that ends precise (BeforeMove with precise_end: lock_on) is followed by a PRECISION
move; beats before TaskRequested's or a plain BeforeMove's (non-precise) move and the ending
reactions (Succeeded, Failed: nothing follows) always keep TAIL_S.
The mode and gains are those of the task's LAST move call (set_control is called after each move,
so a beat sees the previous move's control: the settle concerns the move after the beat, which in
practice runs in the same mode; the deferred TaskRequested beat plays before any move, with the
constructor's control). Impedance with gains None means the arm's session default profile
(medium, 1.1 s); a custom gains dict, a gains name this table does not know, or a mode it does not
know, gets the longest. Position mode, or not holding, keeps the tail as before.
"""

from collections import OrderedDict
from dataclasses import dataclass
from importlib import resources
from zlib import crc32

import numpy as np
import yaml

from . import behaviors as B
from . import curves
from .beat import Beat, compose_beat, zone_at  # noqa: F401  (Beat is this module's beat type)
from .caps import CAP_PRESETS, DEFAULT_CAPS
from .track import Label, Zone
from .zones import zone_of

BUDGETS = {"demo": 6.0, "subtle": 2.5}
SUBTLE_MIN_KEEP = 6  # the subtle preset plays only rules at least this important
LOCK_ON_MODES = ("view", "standoff")
USUAL_SETTLE_S = 0.0  # position mode, or nothing held: the usual tail (TAIL_S) stands
SETTLE_S = {"soft": 2.3, "medium": 1.1, "default": 1.1, "stiff": 0.1}  # impedance gains profile -> seconds, while held
LONGEST_SETTLE_S = max(SETTLE_S.values())  # unknown mode or gains
POSITION_MODES, IMPEDANCE_MODES = (0, "position"), (1, "impedance")  # capability_engine.arm.POSITION / IMPEDANCE
EPS = 1e-9
RESERVE_MIN_KEEP = 1  # a rule this unimportant (or less) never reserves time
ENDING_RESERVE_KEEP = 6  # an ending reaction reserves time for its essential rules only (proud, sorry): the
# whole failure reaction (search, sorry, shake_off) would be 5.7 s of a 6 s budget
T_ON = 0.1  # a beat's motion starts this long into its track
TAIL_S = 0.2
CACHE_MAX = 512

EAGER_GLANCE_RAD = np.radians(14.0)
EAGER_LIFT_M = B.PERK_LIFT_M
EAGER_HOLD_S = 0.2
CURIOUS_TILT_AT_S = 0.1
CURIOUS_TILT_RAD = np.radians(12.0)
CURIOUS_TILT_HOLD_S = 0.1
CURIOUS_PECKS = 2
CURIOUS_LEAN_M = 0.01
WIND_UP_SHORT = 0.6  # the short wind-up draws back this share as far (and quicker: sqrt of it)
FOLLOW_SHORT_S = 0.9
LOCK_STARE_S, LOCK_STARE_SHORT_S = 0.6, 0.2
LOCK_LEAN_M = 0.025  # the full lock-on leans in this far (short: B.LOCK_LEAN_M)
PROUD_HOLD_S = 0.2
PROUD_FLOURISH_S = 1.2
PROUD_LOOK_MAX_RAD = np.radians(12.0)
PROUD_SHORT_LOOK_MAX_RAD = np.radians(3.0)  # the short proud's look is small: it fits the subtle budget (<= 2 s)
ATTENTIVE_HOLD_S, ATTENTIVE_SHORT_HOLD_S = 0.8, 0.2
SEARCH_SPOTS = B.SEARCH_SPOTS[:2]
SEARCH_SHORT_SPOTS = B.SEARCH_SPOTS[:1]
SORRY_QUICK = 0.8
SORRY_EYES_AFTER_S = 0.7
SORRY_EYES_HOLD_S = 0.3
SORRY_LINGER_S = 0.3
SORRY_SHORT_SAG = 0.5
SHAKE_CYCLES, SHAKE_SHORT_CYCLES = 3.0, 2.0
NOT_WHEN_HELD = ("shake_off",)


# ---------------------------------------------------------------------------------------- events
@dataclass(frozen=True)
class TaskRequested:
    pass


@dataclass(frozen=True)
class Phase:
    name: str
    message: str


@dataclass(frozen=True)
class BeforeMove:
    precise_end: bool


@dataclass(frozen=True)
class AfterMove:
    pass


@dataclass(frozen=True)
class Succeeded:
    pass


@dataclass(frozen=True)
class Failed:
    recoverable: bool


EVENTS = (TaskRequested, Phase, BeforeMove, AfterMove, Succeeded, Failed)


# ------------------------------------------------------------------------------------------ rules
@dataclass(frozen=True)
class Rule:
    event: str  # an event class name
    beat: str
    keep: int
    phase: str = None
    not_phase: str = None
    precise_end: bool = None
    recoverable: bool = None

    @property
    def ending(self):
        """'success' / 'failure' for an ending reaction, else None."""
        return {"Succeeded": "success", "Failed": "failure"}.get(self.event)

    def matches(self, event, phase):
        if type(event).__name__ != self.event:
            return False
        if self.phase is not None and phase != self.phase:
            return False
        if self.not_phase is not None and phase == self.not_phase:
            return False
        if self.precise_end is not None and getattr(event, "precise_end", None) != self.precise_end:
            return False
        if self.recoverable is not None and getattr(event, "recoverable", None) != self.recoverable:
            return False
        return True


def load_rules():
    """{capability: (Rule, ...)} from data/rules.yaml."""
    text = resources.files("puppy_motion").joinpath("data/rules.yaml").read_text(encoding="utf-8")
    return {cap: tuple(Rule(**row) for row in rows) for cap, rows in yaml.safe_load(text).items()}


# ------------------------------------------------------------------------------------------ beats
def _end(c, default):
    span = c.span()
    return span[1] if span else default


def _toward(ctx, target):
    """(6,) unit shift from the tool toward `target` (zeros if there is none to speak of)."""
    d = ctx.aim_point(target, T_ON) - ctx.chain.fk(ctx.q(T_ON), "tool")[0]
    n = np.linalg.norm(d)
    return np.concatenate([d / n if n > B.MIN_PROBE_M else np.zeros(3), np.zeros(3)])


def _eager(ctx, target, held, short):
    if short:
        return [B.perk(ctx, T_ON, T_ON + B.paced_perk_s(ctx, T_ON, held) + EAGER_HOLD_S, level=held)]
    t_back = T_ON + B.user_turn_s(ctx, T_ON, held, EAGER_GLANCE_RAD, EAGER_LIFT_M) + EAGER_HOLD_S
    return [B.look_at_user(ctx, T_ON, t_back, held, EAGER_GLANCE_RAD, lift_m=EAGER_LIFT_M)]


def _wind_up(ctx, target, held, short):
    """A draw-back against the coming move (toward `target`: the director is not told the move)."""
    share = WIND_UP_SHORT if short else 1.0
    quick = np.sqrt(share)
    back = -B.WIND_UP_SHIFT_M * share * _toward(ctx, target)
    back = back * B.safe_scale(ctx, T_ON, back)
    c = ctx.empty("wind_up", [])
    if not np.any(back):
        return [c]
    draw_s = B.paced_move(ctx, T_ON, back, B.WIND_UP_S * quick)
    release_s = B.paced_move(ctx, T_ON, back, B.WIND_UP_RELEASE_S * quick)
    env = curves.envelope(ctx.track.t, T_ON, T_ON + draw_s + B.WIND_UP_HOLD_S, draw_s, release_s)
    c.head[:] = back * env[:, None]
    c.labels = [Label("wind_up", T_ON, T_ON + draw_s + B.WIND_UP_HOLD_S + release_s)]
    return [c]


def _follow_through(ctx, target, held, short):
    """The head carries on a little toward `target`, then settles with a damped wobble."""
    settle_s = FOLLOW_SHORT_S if short else B.FOLLOW_SETTLE_S
    carry = B.FOLLOW_SHIFT_M * _toward(ctx, target)
    c = ctx.empty("follow_through", [Label("follow_through", T_ON, T_ON + settle_s)])
    if np.any(carry):
        wave = curves.wobble(ctx.track.t, T_ON, B.FOLLOW_HZ, B.FOLLOW_DAMPING, settle_s)
        c.head[:] = B.fit_scale(ctx, T_ON, carry, wave) * carry * wave[:, None]
    return [c]


def _curious(ctx, target, held, short):
    t_tilt = T_ON + CURIOUS_TILT_AT_S
    tilt_s = B.paced_move(ctx, t_tilt, [0.0, 0.0, 0.0, CURIOUS_TILT_RAD, 0.0, 0.0], B.TILT_TURN_S, B.TILT_OVERSHOOT)
    extra, t_back = [], t_tilt + tilt_s + CURIOUS_TILT_HOLD_S
    if not short:
        sniff = B.sniff(ctx, target, t_tilt + tilt_s, pecks=CURIOUS_PECKS, lean_m=CURIOUS_LEAN_M)
        t_back = _end(sniff, t_tilt + tilt_s)
        extra.append(sniff)
    return [B.head_tilt(ctx, t_tilt, t_back, side=1, angle_rad=CURIOUS_TILT_RAD), *extra]


def _lock_on(ctx, target, held, short):
    stare, lean = (LOCK_STARE_SHORT_S, B.LOCK_LEAN_M) if short else (LOCK_STARE_S, LOCK_LEAN_M)
    return [B.lock_on(ctx, target, T_ON, T_ON + B.lock_on_s(ctx, target, T_ON, stare, lean_m=lean), lean_m=lean)]


def _proud(ctx, target, held, short):
    turn = PROUD_SHORT_LOOK_MAX_RAD if short else PROUD_LOOK_MAX_RAD
    t_look = T_ON + B.PROUD_LOOK_AT_S
    t_there = t_look + B.user_turn_s(ctx, t_look, held, turn, B.PROUD_LIFT_M)
    t_end = t_there + (PROUD_HOLD_S if short else PROUD_FLOURISH_S)
    return [B.proud(ctx, T_ON, t_end, flourish=None if short else "clack", max_turn_rad=turn)]


def _attentive(ctx, target, held, short):
    hold = ATTENTIVE_SHORT_HOLD_S if short else ATTENTIVE_HOLD_S
    t_back = T_ON + B.user_turn_s(ctx, T_ON, held, PROUD_LOOK_MAX_RAD, 0.0) + hold
    return [B.look_at_user(ctx, T_ON, t_back, held, PROUD_LOOK_MAX_RAD)]


def _search(ctx, target, held, short):
    return [B.search(ctx, T_ON, spots=SEARCH_SHORT_SPOTS if short else SEARCH_SPOTS)]


def _sorry(ctx, target, held, short):
    if short:
        sag_m, look_rad = B.DROOP_SAG_M * SORRY_SHORT_SAG, B.DROOP_LOOK_DOWN_RAD * SORRY_SHORT_SAG
        quick = np.sqrt(SORRY_SHORT_SAG)
        return [B.droop(ctx, T_ON, T_ON + B.DROOP_S * quick + SORRY_LINGER_S, sag_m, look_rad, quick=quick)]
    t_eyes = T_ON + SORRY_EYES_AFTER_S
    turn_s = (_end(B.puppy_eyes(ctx, t_eyes, t_eyes), t_eyes) - t_eyes) / 2.0
    t_back = t_eyes + turn_s + SORRY_EYES_HOLD_S
    return [B.droop(ctx, T_ON, t_back, quick=SORRY_QUICK), B.puppy_eyes(ctx, t_eyes, t_back)]


def _shake_off(ctx, target, held, short):
    return [B.shake_off(ctx, T_ON, cycles=SHAKE_SHORT_CYCLES if short else SHAKE_CYCLES)]


# name -> (builder(ctx, target, held, short) -> [Contribution], seconds the track must run (full))
BEATS = {
    "eager": (_eager, 6.0),
    "wind_up": (_wind_up, 4.0),
    "follow_through": (_follow_through, 4.0),
    "curious": (_curious, 8.0),
    "lock_on": (_lock_on, 6.0),
    "proud": (_proud, 8.0),
    "attentive": (_attentive, 5.0),
    "search": (_search, 10.0),
    "sorry": (_sorry, 8.0),
    "shake_off": (_shake_off, 8.0),
}
MOODS = {
    "eager": "eager", "wind_up": "anticipating", "follow_through": "settling", "curious": "curious",
    "lock_on": "focused", "proud": "proud", "attentive": "attentive", "search": "puzzled",
    "sorry": "sorry", "shake_off": "shaking it off",
}
TARGET_OF = {"grab_container": "container", "button_press": "button"}

_CACHE = OrderedDict()  # composed beats shared across directors: composing takes a while


def _target_key(targets):
    return tuple(
        None if p is None else np.asarray(p, dtype=float).tobytes()
        for p in (targets.person_head, targets.container, targets.button)
    )


# ----------------------------------------------------------------------------------------- director
class Director:
    def __init__(self, capability, chain, scene, targets, preset="demo", lock_on="view", caps=DEFAULT_CAPS, seed=0,
                 control=("position", None)):
        rules = load_rules()
        if capability not in rules:
            raise ValueError(f"unknown capability {capability!r}; expected one of {sorted(rules)}")
        if preset not in BUDGETS:
            raise ValueError(f"unknown preset {preset!r}; expected one of {sorted(BUDGETS)}")
        if lock_on not in LOCK_ON_MODES:
            raise ValueError(f"unknown lock_on {lock_on!r}; expected one of {LOCK_ON_MODES}")
        if caps not in CAP_PRESETS:
            raise ValueError(f"unknown caps preset {caps!r}; expected one of {sorted(CAP_PRESETS)}")
        self.capability, self.chain, self.scene, self.targets = capability, chain, scene, targets
        self.preset, self.lock_on, self.caps, self.seed = preset, lock_on, caps, int(seed)
        self.control = ("position", None)
        self.set_control(*control)
        # the arm is stopped when an event arrives, so both lock_on modes play the same stare there
        self.rules = rules[capability]
        self.budget_s = BUDGETS[preset]
        self.min_keep = SUBTLE_MIN_KEEP if preset == "subtle" else 0
        self.spent_s = 0.0
        self.log = []
        self._phase = "idle"
        self._entry = None  # the zone table's entry for the current phase (None: no Phase yet, at rest)
        self._precise_move = False
        self._requested = False
        self._done = set()  # indices of rules already decided
        self._over = False  # an ending (success or failure) has come

    # ------------------------------------------------------------------ public
    def set_control(self, mode, gains):
        """Record the control mode and gains of the task's latest move (what a beat now copies)."""
        self.control = (mode, gains)

    def settle_s(self, held):
        """The still tail (s) a beat should leave before the next PRECISION move: see the module
        docstring. Never raises: anything it cannot read is the longest."""
        if not held:
            return USUAL_SETTLE_S
        mode, gains = self.control
        try:
            if isinstance(mode, str):
                mode = mode.strip().lower()
            if not isinstance(mode, bool) and mode in POSITION_MODES:
                return USUAL_SETTLE_S
            if not isinstance(mode, bool) and mode in IMPEDANCE_MODES:
                if gains is None:  # the arm uses its session default profile (medium)
                    return SETTLE_S["default"]
                if isinstance(gains, str):
                    return SETTLE_S.get(gains.strip().lower(), LONGEST_SETTLE_S)
        except Exception:  # unhashable or odd values
            pass
        return LONGEST_SETTLE_S

    def on_event(self, event, q_now, held):
        """The beats to play now, in order: each starts and ends at q_now and passes check_beat."""
        if not isinstance(event, EVENTS):
            raise TypeError(f"not a director event: {event!r}")
        name = type(event).__name__
        try:
            q_now, held = np.asarray(q_now, dtype=float), bool(held)
        except Exception as e:
            self.log.append(f"{name}: nothing (bad q_now: {type(e).__name__}: {e})")
            return []
        if isinstance(event, Phase):
            return self._on_phase(event)
        if isinstance(event, TaskRequested):
            self._requested = True
        if isinstance(event, BeforeMove):
            self._precise_move = bool(event.precise_end)
        if isinstance(event, Succeeded) or isinstance(event, Failed) and not event.recoverable:
            self._over = True
        if isinstance(event, Failed) and not event.recoverable:
            self.log.append(f"{name}(False): nothing")
            return []
        floor = Zone.FREE
        if isinstance(event, Failed) and self._entry is not None and self._entry.zone is Zone.PRECISION:
            self.log.append(f"{name}(True): nothing (phase {self._phase!r} is PRECISION: the arm may be at a contact)")
            return []
        if isinstance(event, (BeforeMove, AfterMove)):
            if self._entry is not None and self._entry.zone is Zone.PRECISION:
                self.log.append(f"{name}: nothing (phase {self._phase!r} is PRECISION)")
                return []
            if isinstance(event, AfterMove) and self._precise_move:
                self._precise_move = False
                self.log.append(f"{name}: nothing (the move ended at a precise pose; the arm stays settled)")
                return []
            floor = self._entry.zone if self._entry is not None else Zone.FREE
        try:
            zone = zone_at(self.chain, self.scene, q_now, held, floor)
        except Exception as e:  # a malformed q_now is refused, not raised
            self.log.append(f"{name}: nothing (bad q_now: {type(e).__name__}: {e})")
            return []
        if held and zone is Zone.NEAR_USER:
            self.log.append(f"{name}: nothing (holding something near the person)")
            return []
        matched = [i for i, r in enumerate(self.rules) if r.matches(event, self._phase)]
        chosen = {}
        for i in sorted(matched, key=lambda i: -self.rules[i].keep):  # most important first
            beat = self._decide(i, name, q_now, held, zone)
            self._done.add(i)
            if beat is not None:
                chosen[i] = beat
                self.spent_s += beat.duration
        if isinstance(event, Failed):  # a recoverable failure may be followed by a retry and its ending
            self._done -= {j for j, r in enumerate(self.rules) if r.ending is not None}
        return [chosen[i] for i in matched if i in chosen]

    # ----------------------------------------------------------------- internals
    def _on_phase(self, event):
        self._phase = event.name
        self._entry = zone_of(self.capability, event.name, event.message)
        if self._entry.zone is Zone.PRECISION:
            self.log.append(f"Phase {event.name!r}: PRECISION (or not in the zone table): nothing")
        else:
            self.log.append(f"Phase {event.name!r}: {self._entry.zone.name}")
        return []

    def _settles(self, event_name):
        """Whether a beat answering this event may be followed by a PRECISION move: after a move, or
        before one that ends at a precise pose (the approach to the grasp)."""
        return event_name == "AfterMove" or event_name == "BeforeMove" and self._precise_move

    def _decide(self, i, event_name, q_now, held, zone):
        rule = self.rules[i]
        tag = f"{event_name} {rule.beat}"
        if rule.keep < self.min_keep:
            self.log.append(f"{tag}: dropped (the {self.preset!r} preset plays keep >= {self.min_keep})")
            return None
        if held and rule.beat in NOT_WHEN_HELD:
            self.log.append(f"{tag}: dropped (not while holding something)")
            return None
        left = self.budget_s - self.spent_s - self._reserve(i, q_now, held, zone)
        if left <= EPS:
            self.log.append(f"{tag}: dropped (no budget left after the reserve)")
            return None
        for short in (False, True):
            beat = self._beat(rule.beat, short, q_now, held, zone, tag, not self._settles(event_name) or rule.ending is not None)
            if beat is not None and beat.duration <= left + EPS:
                self.log.append(f"{tag}: {'short' if short else 'full'} {beat.duration:.2f} s ({left:.2f} s free)")
                return beat
        self.log.append(f"{tag}: dropped (does not fit {left:.2f} s)")
        return None

    def _reserve(self, i, q_now, held, zone):
        """Seconds kept from rule i, in short versions, for what may still come. An ending reaction
        reserves nothing. Any other beat keeps the larger of the success and the failure reaction
        (their essential rules, keep >= ENDING_RESERVE_KEEP, that the preset plays), plus the other
        higher-keep mid-task rules not yet decided."""
        rule = self.rules[i]
        if rule.ending is not None or self._over:
            return 0.0
        mid, endings = 0.0, {}
        for j, o in enumerate(self.rules):
            if j == i or j in self._done or o.keep <= RESERVE_MIN_KEEP or o.keep < self.min_keep:
                continue
            if o.ending is not None:
                if o.keep < ENDING_RESERVE_KEEP:
                    continue
                endings[o.ending] = endings.get(o.ending, 0.0) + self._short_s(o, q_now, held, zone)
            elif o.keep > rule.keep and not (o.event == "TaskRequested" and self._requested):
                mid += self._short_s(o, q_now, held, zone)
        return mid + max(endings.values(), default=0.0)

    def _short_s(self, rule, q_now, held, zone):
        """Seconds of rule's short beat with the usual tail (a reserve is the least that must stay
        affordable; a later mid-task beat's settle is paid from what is left then)."""
        if held and rule.beat in NOT_WHEN_HELD:
            return 0.0
        beat = self._beat(rule.beat, True, q_now, held, zone, None, True)  # the least a reserve must keep: the usual tail
        return 0.0 if beat is None else beat.duration

    def _beat(self, name, short, q_now, held, zone, tag, usual_tail=False):
        """The composed, validated beat (cached), or None (why: in the log when `tag` is given)."""
        tail_s = TAIL_S if usual_tail else max(TAIL_S, self.settle_s(held))
        key = (id(self.chain), id(self.scene), _target_key(self.targets), name, short, held, int(zone),
               self.caps, q_now.tobytes(), self.seed, tail_s)
        hit = _CACHE.get(key)
        if hit is None:
            hit = self._compose(name, short, q_now, held, zone, tail_s) + (self.chain, self.scene)  # keeps ids alive
            _CACHE[key] = hit
            while len(_CACHE) > CACHE_MAX:
                _CACHE.popitem(last=False)
        beat, why = hit[0], hit[1]
        if beat is None:
            if tag:
                self.log.append(f"{tag}: {'short' if short else 'full'} not played ({why})")
            return None
        # a copy: the cache is shared, and a caller may shift or edit the points it is given
        return Beat(beat.name, beat.mood, [(t, q.copy(), dq.copy()) for t, q, dq in beat.points], beat.zone)

    def _compose(self, name, short, q_now, held, zone, tail_s=TAIL_S):
        build, length_s = BEATS[name]
        length_s += max(0.0, tail_s - TAIL_S)  # room for a longer still tail
        t = self.targets
        target = getattr(t, TARGET_OF.get(self.capability, ""), None)
        seed = int(np.random.default_rng([self.seed, crc32(name.encode())]).integers(1 << 30))
        return compose_beat(
            name, MOODS[name], lambda ctx, rng, h: build(ctx, target, h, short), length_s, q_now, held, zone,
            self.chain, self.scene, t, seed, caps=self.caps, tail_s=tail_s,
        )
