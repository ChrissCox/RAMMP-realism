"""The same happy wag twice at home, side by side in the close camera: inside today's envelope (the
default caps: every joint at most 3 rad/s^2, about the task's own peak) and with three times the
wrist acceleration ('snappy': joints 5-7 at 9 rad/s^2 in open space only, a comparison preset that
is never the default). It is the same wiggle() call on both sides; each joint swings as far as its
own caps allow, so at the same tempo the snappy wrist pans and rolls further and faster, while the
arm's sway (joints 1-4, still 3 rad/s^2) is the same. For deciding whether quicker wags are worth
a wider envelope.

The camera is the close one (same distance, angle and lens, so the legibility numbers compare with
the reels'), its aim shifted right so the face and the gripper sit in the narrower pair panel."""

from dataclasses import dataclass

import numpy as np

from puppy_motion import behaviors as B

from .. import lib, render, runs
from ..timeline import constant_track
from . import REST_GRIP, PairClip

SNAPPY = "snappy"
WAG_AT_S = (0.7, 3.5)  # two bursts, the same on both sides
END_REST_S = 0.7
CHIP = "HAPPY (wag)"
SIDES = ("Inside today's envelope", "3x wrist acceleration (not default)")
CAMERA = "wag_close"
AIM_SHIFT_M = (0.07, 0.01)  # the close camera's aim moved this far along the image's right and up


def _close_for_pairs():
    """render.CAMERAS['close'] aimed AIM_SHIFT_M to the right and up."""
    close = render.CAMERAS["close"]
    forward, up = close.axes()
    right, lift = AIM_SHIFT_M
    offset = np.asarray(close.head_offset) + right * np.cross(forward, up) + lift * up
    return render.Camera(None, close.distance, close.azimuth, close.elevation, close.fovy, tuple(offset.tolist()))


render.CAMERAS.setdefault(CAMERA, _close_for_pairs())


def wag_track(caps, sc=None, kin=None):
    """The wags at home, composed under the cap preset `caps` (sc and kin are not needed: lib.env())."""
    e = lib.env()
    duration = WAG_AT_S[-1] + B.WIGGLE_CYCLES / B.WIGGLE_HZ + END_REST_S
    rest = constant_track(runs.HOME_JOINTS, REST_GRIP, duration, name="rest at home")
    runs.mark_near_user(rest, e.kin)
    ctx = lib.context(rest, e, caps)
    return lib.compose(rest, [B.wiggle(ctx, t, beat=CHIP) for t in WAG_AT_S], e, caps)


def build():
    return wag_track("default"), wag_track(SNAPPY)


def wrist_swing_deg(track):
    """The largest swing (deg) a wrist joint (5-7) makes about its functional angle."""
    return float(np.degrees(np.abs(track.q - track.func.q)[:, B.WRIST_JOINTS].max()))


@dataclass(frozen=True)
class CapsPair(PairClip):
    """Two Puppy Mode tracks under different caps: the subtitle and what changes, measured (no
    plain vs Puppy Mode durations: both take the same time)."""

    def subtitle_for(self, tracks):
        left, right = (wrist_swing_deg(t) for t in tracks)
        return [self.subtitle, f"wrist swing up to {left:.1f} deg vs {right:.1f} deg at the same tempo; same arm sway"]


CLIP = CapsPair(
    name="wag_compare",
    title="A happy wag: today's envelope vs a snappier wrist",
    subtitle="wrist joints 5-7 at 3 rad/s^2 (left) vs 9 rad/s^2 (right; open space only, gentle near the user)",
    camera=CAMERA,
    build=build,
    headers=SIDES,
)
