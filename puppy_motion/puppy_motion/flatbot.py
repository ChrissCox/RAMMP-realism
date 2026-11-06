"""The flatbot chair scene: the occupant, seat and drive base as boxes, from flatbot's own cuRobo
layers (vendored byte-for-byte under data/flatbot_layers, see SOURCE.txt).

Layer boxes are {position: centre, dims: full size}, axis-aligned in arm_mount, which is the frame
of the run URDF's root, so they are used as they are."""

from importlib import resources

import numpy as np
import yaml

from .geometry import Box, Scene

CHAIR_LAYERS_COMMIT = "ec39dd2764d9c1eeabd1633e949c5b5fbb5b9a98"
PERSON_LAYERS = ("layer_mebot_person.yaml",)
OBSTACLE_LAYERS = ("layer_mebot_seat.yaml", "layer_mebot_platform.yaml")


def layer_boxes(yaml_text):
    """The boxes of one layer file: center = position, half = dims / 2."""
    boxes = []
    for ob in yaml.safe_load(yaml_text)["obstacles"]:
        boxes.append(Box(np.asarray(ob["position"], dtype=float), np.asarray(ob["dims"], dtype=float) / 2.0))
    return boxes


def _load(name):
    text = resources.files("puppy_motion").joinpath("data/flatbot_layers/" + name).read_text(encoding="utf-8")
    return layer_boxes(text)


def chair_scene():
    """Person = the person layer; obstacles = the seat and platform layers."""
    return Scene(
        person=[b for n in PERSON_LAYERS for b in _load(n)],
        obstacles=[b for n in OBSTACLE_LAYERS for b in _load(n)],
    )
