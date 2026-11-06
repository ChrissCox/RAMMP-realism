"""replay_director with lock_on='standoff': the reach stops 10 cm short of the container for the
lock-on stare and the last 10 cm creep in at 0.75x speed (timing only: the path is the same). The
robot cannot do this today: grab_container's reach is one driver call, so it would have to be split
in two. The 6 s budget is lifted so the stare is not crowded out by the earlier beats.
"""

from .. import replay, scene
from . import PairClip


def build():
    from puppy_motion import director

    keep = director.BUDGETS["demo"]
    director.BUDGETS["demo"] = 30.0
    try:
        return replay.direct_run(scene.REFERENCE_RUN, "demo", lock_on="standoff")
    finally:
        director.BUDGETS["demo"] = keep


CLIP = PairClip(
    name="replay_director_standoff",
    title="Lock-on at 10 cm: stop, stare, then creep in (puppy_motion's director)",
    subtitle=replay.STANDOFF_SUBTITLE,
    camera="counter",
    build=build,
    mood="CALM",
)
