"""A real failed run, plain vs Puppy Mode, through the director with the 'demo' budget.

What really happened (grab_container_20261028_191809): the arm went to the view pose and the wrist
camera looked for the container for 4.5 s, but its two views of it disagreed by more than 1 cm, so
the capability refused to grab ("grab failed: frames disagree > 1 cm -- not grasping -- the arm
holds where it is"), a recoverable error, and the arm stayed where it was.

Puppy Mode, through puppy_motion's Director (replay.direct_run, the 'demo' budget): the events the
capability would report go to the director, which plays what fits: with the request before the
window and the budget reserved for the ending, the curious tilt and the wind-up do not fit; on the
failure (a recoverable failure; no phase is invented, the last one the run reported is 'detect', a
PRECISION phase) it plays nothing: the arm stays where it stopped, as the ruling says for a failure at
a precise moment. The Puppy Mode side is the plain run.
"""

from .. import replay
from . import PairClip

FAILURE_RUN = "grab_container_20261028_191809"
WINDOW_S = (5.4, 19.88)  # the plain task's seconds shown: from home, moving to the view pose, to the end


def build():
    """The request came before the window: only the events inside it are reported."""
    plain, puppy = replay.direct_run(FAILURE_RUN, "demo", window=WINDOW_S, drop=("TaskRequested",))
    return plain, replay.display_names(puppy)


CLIP = PairClip(
    name="failure_pair",
    title="When it fails at a precise moment: it stays still",
    subtitle="real run: its two camera views disagreed by over 1 cm in a precise phase, so it refused to grab; Puppy Mode plays no sorry",
    camera="counter",
    build=build,
    mood="CALM",
)
