"""The real grab_container run, plain and through the director side by side: the same path, with
Puppy Mode's beats only where the arm is stopped between moves (the boundary tier), decided event
by event within the 'demo' budget (6 s), the lock-on stare at the view pose before the reach. On
this run that is an eager perk when asked, a curious tilt at the view pose, the stare while the
gripper opens, a wind-up before the reach and a proud look at the user once it holds the container.
"""

from .. import director, runs, scene
from . import PairClip


def build():
    return director.direct(runs.load_run(scene.REFERENCE_RUN), scene.build())


CLIP = PairClip(
    name="replay_pair",
    title="Grab container: plain vs Puppy Mode",
    subtitle="a real run; beats only while the arm is stopped between moves, chosen as it goes within 6 s",
    camera="counter",
    build=build,
    mood="CALM",
)
