"""Eager to start, plain vs Puppy Mode: the first seconds of the real run
grab_container_20261029_132745, from just before the task is requested until the arm has arrived
at the view pose. Plain: the recorded motion. Puppy Mode: the same moves, each exactly as
recorded, with beats only where the arm is stopped between them, chosen by puppy_motion's
Director from the events the capability would report (replay.direct_run):
- asked to go, it looks at the user and perks up (eager), in the stop where the request arrives.
The demo budget (6 s) keeps nothing else: the eager takes 2.8 s and the director reserves the rest
for what may still come (the lock-on, the success), so the wind-up before the move and the curious
tilt at the view pose are dropped (see the composition log).
Each pause is a time-warp hold inside a stop; `check` holds the track to the recorded path, to
the PRECISION samples and to the beat budget.
"""

from .. import replay, scene
from . import PairClip

# the window on the replay's clock: the request comes at 1.0, the arm is at the view pose at 4.74.
# Its 50 Hz grid skips the run's 10 ms 'locate' phase (PRECISION), which would split the stop.
WINDOW_S = (0.7, 5.1)


def build():
    plain, puppy = replay.direct_run(scene.REFERENCE_RUN, "demo", window=WINDOW_S)
    return plain, replay.display_names(puppy)


CLIP = PairClip(
    name="eager_start",
    title="Eager to start: plain vs Puppy Mode",
    subtitle=(
        "the start of a real run (grab container): asked to go, it looks at you and perks up, "
        "then moves as recorded"
    ),
    camera="close_follow",
    build=build,
    mood="CALM",
)
