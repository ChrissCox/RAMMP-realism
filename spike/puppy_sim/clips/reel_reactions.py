"""Puppy Mode's reactions: each primitive once at home in the close camera, its name in the chip:
nod, bounce, wiggle, clack, sniff, double take, search, startle, shake off, proud."""

import numpy as np

from puppy_motion import behaviors as B

from . import Clip, rest_reel

DOUBLE_TAKE_SPOT = (-0.02, 0.17, 0.0)  # the glance-away spot on the counter, from the container (m)
PROUD_S = 5.0  # long enough for the look, the tilt and the flourish


def _double_take(ctx, t):
    return [B.double_take(ctx, ctx.targets.container, ctx.targets.container + np.array(DOUBLE_TAKE_SPOT), t)]


SLOTS = (
    ("NOD", lambda ctx, t: [B.nod(ctx, t)]),
    ("BOUNCE", lambda ctx, t: [B.bounce(ctx, t)]),
    ("WIGGLE", lambda ctx, t: [B.wiggle(ctx, t)]),
    ("CLACK", lambda ctx, t: [B.clack(ctx, t)]),
    ("SNIFF", lambda ctx, t: [B.sniff(ctx, ctx.targets.container, t)]),
    ("DOUBLE TAKE", _double_take),
    ("SEARCH", lambda ctx, t: [B.search(ctx, t)]),
    ("STARTLE", lambda ctx, t: [B.startle(ctx, t)]),
    ("SHAKE OFF", lambda ctx, t: [B.shake_off(ctx, t)]),
    ("PROUD", lambda ctx, t: [B.proud(ctx, t, t + PROUD_S)]),
)


def build():
    return rest_reel(SLOTS)


CLIP = Clip(
    name="reel_reactions",
    title="Puppy Mode reactions",
    subtitle="each primitive once at home, inside the arm's own speed envelope; the chip names it",
    camera="close",
    build=build,
)
