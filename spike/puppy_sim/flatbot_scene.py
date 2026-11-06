"""The flatbot chair scene: the same Gen3 on the same 30 deg mount, an EMPTY chair drawn as
flatbot's own cuRobo layer boxes (puppy_motion.flatbot.chair_scene), no mannequin.

Red boxes are the occupant keep-out (the person layer: where a seated adult may be, plus margin);
grey boxes are the seat and the drive base (the obstacle layers). Both translucent. Kinematic only,
like the rest of the spike. Used by the playground previews (clips/playground_*.py); a clip selects
it with run=FLATBOT."""

from dataclasses import replace
from functools import lru_cache

import mujoco
import numpy as np

from . import scene as sc

FLATBOT = "flatbot"
FLOOR_Z = -0.50  # just under the drive base's lowest box
PERSON_RGBA = (0.90, 0.25, 0.25, 0.25)
OBSTACLE_RGBA = (0.45, 0.47, 0.52, 0.30)
CHAIR_LAYER_SOURCE = "flatbot-empty-world@ec39dd2"
LEGEND = "red: occupant keep-out   grey: seat and drive base"


def _boxes(spec, body_name, boxes, rgba):
    mat = sc._material(spec, f"{body_name}_box", rgba, specular=0.0, shininess=0.0)
    body = spec.worldbody.add_body(name=body_name)
    for k, b in enumerate(boxes):
        sc._box(body, tuple(b.center), tuple(b.half), mat, name=f"{body_name}_{k}")
    return body


def _groups(model, base):
    """Clearance groups: the arm parts as in the full scene, the layer boxes as person / chair."""

    def on(name):
        return np.array([g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name == name], dtype=int)

    groups = {k: base.groups[k] for k in sc.ARM_PART_BODIES}
    groups.update(person=on("person"), chair=on("chair"), counter=np.array([], dtype=int), panel=np.array([], dtype=int))
    return groups


@lru_cache(maxsize=1)
def build():
    from puppy_motion import flatbot

    base = sc.build()  # placement and gripper linkage only; its own model is not used
    chair = flatbot.chair_scene()
    spec = sc._robot_spec()
    sc._add_look(spec)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            g.pos = [0.0, 0.0, FLOOR_Z]
    _boxes(spec, "person", chair.person, PERSON_RGBA)
    _boxes(spec, "chair", chair.obstacles, OBSTACLE_RGBA)
    model = spec.compile()
    return replace(base, model=model, groups=_groups(model, base), person_head=np.array([-0.02, -0.36, 0.88]))
