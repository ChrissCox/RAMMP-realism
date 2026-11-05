"""The lock-on at the pre-grasp standoff, plain vs Puppy Mode, on the real run: the reach stops
10 cm short of the container, Puppy Mode leans in and stares, and the last 10 cm creep in at 0.75x
speed (timing only: the path is the same). The robot cannot do this today: grab_container's reach
(view -> pre-grasp -> grasp) is one driver call, so it would have to split that call in two
(lock_on='standoff'). Same window and camera as lock_on_view; only the lock-on plays.
"""

from . import PairClip
from .lock_on_view import CAMERA, directed


def build():
    return directed("standoff")


CLIP = PairClip(
    name="lock_on_standoff",
    title="Lock-on at 10 cm: stop, stare, then creep in",
    subtitle="needs grab_container to split its reach into two calls (view -> 10 cm short, then -> grasp)",
    camera=CAMERA,
    build=build,
    mood="CALM",
)
