"""The real grab_container run, plain and through the director side by side: the same path, with
Puppy Mode's beats only where the arm is stopped between moves (the boundary tier), decided event
by event within the 'demo' budget (6 s), the lock-on stare at the view pose before the reach. On
this run that is an eager look at the user and a perk when asked, the lock-on stare while the
gripper opens, and a proud look at the user once it holds the container (the wind-up, the curious
tilt and the follow-through do not fit the budget). The events and the director are
puppy_motion's: this is what its hook would do (replay.direct_run).
"""

from .. import replay, scene
from . import PairClip


def build():
    plain, puppy = replay.direct_run(scene.REFERENCE_RUN, "demo")
    return plain, replay.display_names(puppy)


CLIP = PairClip(
    name="replay_pair",
    title="Grab container: plain vs Puppy Mode",
    subtitle="a real run; beats only while the arm is stopped between moves, chosen as it goes within 6 s",
    camera="counter",
    build=build,
    mood="CALM",
)
