"""The idle director for the playground's auto mode: picks the next small expressive beat from a
mood, composes it at the arm's current posture, and refuses (None, logged) what does not validate.

Calm is the hub: curious and playful moods return to it, and sleepy (droop, then doze_and_wake) is
only offered after SLEEPY_AFTER_S without a request. `size` multiplies every contribution's head and
joints before composing (the playground's size slider). Nothing here talks to a robot."""

from collections import deque

import numpy as np

from . import behaviors as B
from .beat import Beat, compose_beat, zone_at

LOOK_TARGET = np.array([1.2, -0.36, 0.9])  # about 1 m in front of the seat front, standing head height
MOODS = ("calm", "curious", "playful", "sleepy")
DWELL_S = {"calm": (20.0, 60.0), "curious": (15.0, 40.0), "playful": (10.0, 25.0)}
PAUSE_S = (1.0, 6.0)
SLEEPY_AFTER_S = 180.0
LOG_MAX = 200  # the director's log keeps the latest entries (auto mode runs unattended for hours)
T_ON = 0.3

MOOD_BEATS = {
    "calm": ("breathe", "glance"),
    "curious": ("look_around", "head_tilt", "sniff"),
    "playful": ("perk", "wag", "bounce", "proud"),
}
NOT_WHEN_HELD = ("wag", "bounce", "shake_off")
# the seconds a track must run for each expression (the beat is trimmed to its motion afterwards)
LENGTH_S = {
    "breathe": 8.0, "glance": 4.0, "look_around": 8.0, "head_tilt": 5.0, "perk": 5.0, "droop": 6.0,
    "puppy_eyes": 7.0, "nod": 5.0, "bounce": 5.0, "wag": 5.0, "sniff": 6.0, "double_take": 7.0,
    "search": 9.0, "startle": 3.5, "shake_off": 6.0, "proud": 7.0, "doze": 16.0, "lock_on": 6.0,
    "look_at_you": 6.0,
}


def _spots(ctx, yaws_deg, dist_m=1.0):
    """World points dist_m along the gaze, turned about vertical by each yaw."""
    eye, R = ctx.chain.fk(ctx.q(T_ON), "eye")
    f = R[:, 2]
    out = []
    for y in np.radians(yaws_deg):
        c, s = np.cos(y), np.sin(y)
        d = np.array([c * f[0] - s * f[1], s * f[0] + c * f[1], f[2]])
        out.append(eye + dist_m * d / np.linalg.norm(d))
    return out


def _glances(ctx, yaws, held, gap_s=1.3):
    mode = "pan" if held else "shortest"
    parts, t = [], T_ON
    for p in _spots(ctx, yaws):
        parts.append(B.gaze_to(ctx, p, t, t + gap_s, mode=mode))
        t += gap_s
    return parts


def _ahead(ctx, dist_m=0.4):
    q = ctx.q(T_ON)
    return ctx.chain.fk(q, "tool")[0] + dist_m * ctx.chain.fk(q, "ee")[1][:, 2]


# name -> builder(ctx, rng, held) -> [Contribution]
BUILDERS = {
    "breathe": lambda c, r, h: [B.breathe(c, 0.0, 7.0)],
    "glance": lambda c, r, h: _glances(c, [float(r.choice([-1, 1])) * r.uniform(8, 15)], h),
    "look_around": lambda c, r, h: _glances(c, [r.uniform(8, 20), -r.uniform(8, 20), 0.0], h),
    "head_tilt": lambda c, r, h: [B.head_tilt(c, T_ON, T_ON + 1.5, side=int(r.choice([-1, 1])))],
    "perk": lambda c, r, h: [B.perk(c, T_ON, T_ON + 1.5, level=h)],
    "droop": lambda c, r, h: [B.droop(c, T_ON, T_ON + 3.0)],
    "puppy_eyes": lambda c, r, h: [B.puppy_eyes(c, T_ON, T_ON + 3.0, target=c.person)],
    "nod": lambda c, r, h: [B.nod(c, T_ON)],
    "bounce": lambda c, r, h: [B.bounce(c, T_ON)],
    "wag": lambda c, r, h: [B.wiggle(c, T_ON)],
    "sniff": lambda c, r, h: [B.sniff(c, _ahead(c), T_ON)],
    "double_take": lambda c, r, h: [B.double_take(c, *_spots(c, [12.0, -18.0]), T_ON)],
    "search": lambda c, r, h: [B.search(c, T_ON)],
    "startle": lambda c, r, h: [B.startle(c, T_ON)],
    "shake_off": lambda c, r, h: [B.shake_off(c, T_ON)],
    "proud": lambda c, r, h: [B.proud(c, T_ON, T_ON + 3.5)],
    "doze": lambda c, r, h: [B.doze_and_wake(c, T_ON, 5.0)],
    "lock_on": lambda c, r, h: [B.lock_on(c, c.person, T_ON, T_ON + 3.0)],
    "look_at_you": lambda c, r, h: [B.look_at_user(c, T_ON, T_ON + 2.5, level=h)],
}
EXPRESSIONS = tuple(BUILDERS)


class IdleDirector:
    def __init__(self, chain, scene, targets, size: float, seed: int):
        self.chain, self.scene, self.targets = chain, scene, targets
        self.size = float(size)
        self.rng = np.random.default_rng(seed)
        self.mood = "calm"
        self.log = deque(maxlen=LOG_MAX)
        self._mood_end = None
        self._last_request = None
        self._last_sleep = -np.inf
        self._queue = []  # the rest of a sleepy sequence

    # ------------------------------------------------------------------ public
    def next(self, q_now, held: bool, t_now: float):
        """(beat or None, pause_s before the next call)."""
        if self._last_request is None:
            self._last_request = t_now
        self._advance_mood(t_now)
        name = self._pick(held)
        beat = self._make(name, self.mood, q_now, held)
        return beat, float(self.rng.uniform(*PAUSE_S))

    def request(self, name: str, q_now, held: bool, t_now: float):
        """The named expression a button asked for; resets the sleepy clock."""
        self._last_request = t_now
        self._queue = []
        self.mood, self._mood_end = "calm", t_now + float(self.rng.uniform(*DWELL_S["calm"]))
        if name not in BUILDERS:
            self.log.append(f"{name}: unknown expression")
            return None
        return self._make(name, "calm", q_now, held)

    # ----------------------------------------------------------------- moods
    def _sleepy_ok(self, t_now):
        return t_now - self._last_request >= SLEEPY_AFTER_S and t_now - self._last_sleep >= SLEEPY_AFTER_S

    def _advance_mood(self, t_now):
        if self._queue:
            return
        if self._mood_end is None:
            self._mood_end = t_now + float(self.rng.uniform(*DWELL_S["calm"]))
        if t_now < self._mood_end:
            return
        if self.mood != "calm":
            self.mood = "calm"
        else:
            options = ["curious", "playful"] + (["sleepy", "sleepy"] if self._sleepy_ok(t_now) else [])
            self.mood = str(self.rng.choice(options))
        if self.mood == "sleepy":
            self._queue = ["droop", "doze"]
            self._last_sleep = t_now
        else:
            self._mood_end = t_now + float(self.rng.uniform(*DWELL_S[self.mood]))

    def _pick(self, held):
        if self._queue:
            name = self._queue.pop(0)
            if not self._queue:
                self._mood_end = 0.0  # the sleepy sequence is done: back to calm on the next call
            return name
        names = [n for n in MOOD_BEATS[self.mood] if not (held and n in NOT_WHEN_HELD)]
        return str(self.rng.choice(names))

    # ------------------------------------------------------------------ beats
    def _make(self, name, mood, q_now, held):
        q_now = np.asarray(q_now, dtype=float)
        held = bool(held)
        if held and name in NOT_WHEN_HELD:
            self.log.append(f"{name}: not while holding something")
            return None
        zone = zone_at(self.chain, self.scene, q_now, held)
        seed = int(self.rng.integers(1 << 30))
        beat, why = compose_beat(
            name, mood, BUILDERS[name], LENGTH_S[name], q_now, held, zone, self.chain, self.scene,
            self.targets, seed, size=self.size,
        )
        if why:
            self.log.append(why)
        return beat
