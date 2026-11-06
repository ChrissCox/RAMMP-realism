import numpy as np

from puppy_motion.flatbot import CHAIR_LAYERS_COMMIT, chair_scene, layer_boxes
from puppy_motion.geometry import clearances


def test_person_layer_has_its_eight_boxes():
    assert len(chair_scene().person) == 8  # forearms, head, hips, legs, thighs, torso_0..2


def test_head_box_centre_and_half_size():
    text = """obstacles:
  - {name: head, position: [-0.01895, -0.36000, 0.87885], dims: [0.4657, 0.2032, 0.4251]}
"""
    head = layer_boxes(text)[0]
    assert np.allclose(head.center, [-0.01895, -0.36000, 0.87885]) and np.allclose(head.half, [0.23285, 0.1016, 0.21255])
    # and the vendored person layer carries the same box
    assert any(np.allclose(b.center, head.center) and np.allclose(b.half, head.half) for b in chair_scene().person)


def test_rest_pose_is_inside_the_torso_box_as_flatbot_documents(chain, home):
    assert clearances(chain, chair_scene(), home, held=False)[0].min() < 0.0


def test_obstacles_are_seat_plus_platform():
    assert len(chair_scene().obstacles) == 5 + 16


def test_commit_is_pinned():
    assert CHAIR_LAYERS_COMMIT == "ec39dd2764d9c1eeabd1633e949c5b5fbb5b9a98"
