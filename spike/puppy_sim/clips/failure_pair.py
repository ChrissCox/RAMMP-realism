"""A real failed run, plain vs Puppy Mode, through the director with the 'demo' budget.

What really happened (grab_container_20261028_191809): the arm went to the view pose and the wrist
camera looked for the container for 4.5 s, but its two views of it disagreed by more than 1 cm, so
the capability refused to grab ("grab failed: frames disagree > 1 cm -- not grasping -- the arm
holds where it is"), a recoverable error, and the arm stayed where it was.

Puppy Mode: a curious tilt on arriving at the view pose, a stare at the container while the arm
waits to look anyway (free: it adds no time there), stillness while the camera looks, then on the
failure a puzzled look around the counter, a droop with a sheepish look up toward the user (puppy
eyes), and a shake to shake it off. The eager perk and the fillers (wind-ups, follow-throughs) are
left out here: with its defaults the director spends ~1.6 s on an eager perk before it can know the
task will fail, and then the shake-off no longer fits the 6 s (it plays the perk, the tilt, the
stare, the puzzled look and the sorry: +5.5 s).
"""

from .. import director, runs
from . import PairClip

FAILURE_RUN = "grab_container_20261028_191809"
WINDOW_S = (5.4, 19.88)  # the plain task's seconds shown: from home, moving to the view pose, to the end
SKIP = ("eager", "wind_up", "follow_through")


def build():
    plain, puppy = director.direct(runs.load_run(FAILURE_RUN), skip=SKIP)
    return director.excerpt(plain, puppy, *WINDOW_S)


CLIP = PairClip(
    name="failure_pair",
    title="When it fails: puzzled, sorry, shake it off",
    subtitle="real run: its camera's two views of the container disagreed by over 1 cm, so it refused to grab",
    camera="counter",
    build=build,
    mood="CALM",
)
