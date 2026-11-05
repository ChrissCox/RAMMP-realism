"""Puppy Mode at rest: each primitive once at home in the close camera, its name in the chip:
breathe, micro-life, a gaze, a head tilt, a perk, a droop with puppy eyes, a doze and a wake."""

from .. import behaviors as B
from ..timeline import Label
from . import Clip, rest_reel

BREATHE_S = 4.6
MICRO_S = 3.2
GAZE_HOLD_S = 0.4
TILT_HOLD_S = 0.6
DROOP_EYES_AFTER_S = 1.4  # the sheepish look up starts once the droop is down
DROOP_EYES_HOLD_S = 0.8
DOZE_S = 4.0


def _droop_and_eyes(ctx, t):
    """DROOP, then PUPPY EYES while drooped."""
    t_eyes = t + DROOP_EYES_AFTER_S
    eyes = B.puppy_eyes(ctx, t_eyes, t_eyes + DROOP_EYES_HOLD_S)
    sag = B.droop(ctx, t, eyes.span()[1])
    sag.beats = [Label("DROOP", t, t_eyes)]
    eyes.beats = [Label("PUPPY EYES", t_eyes, sag.span()[1])]
    return [sag, eyes]


SLOTS = (
    ("BREATHE", lambda ctx, t: [B.breathe(ctx, t, t + BREATHE_S)]),
    ("MICRO LIFE", lambda ctx, t: [B.micro_life(ctx, t, t + MICRO_S)]),
    ("GAZE TO", lambda ctx, t: [B.gaze_to(ctx, ctx.container, t, t + GAZE_HOLD_S)]),
    ("HEAD TILT", lambda ctx, t: [B.head_tilt(ctx, t, t + TILT_HOLD_S)]),
    ("PERK", lambda ctx, t: [B.perk(ctx, t)]),
    (None, _droop_and_eyes),
    (None, lambda ctx, t: [B.doze_and_wake(ctx, t, DOZE_S, "DOZE", "WAKE")]),
)


def build():
    return rest_reel(SLOTS)


CLIP = Clip(
    name="reel_at_rest",
    title="Puppy Mode at rest",
    subtitle="each primitive once at home, inside the arm's own speed envelope; the chip names it",
    camera="close",
    build=build,
)
