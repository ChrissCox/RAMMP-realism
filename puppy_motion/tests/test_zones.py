import pytest

from puppy_motion.zones import Zone, ZoneEntry, zone_of

APPROACH = "view -> pre-grasp -> grasp, one motion"


def test_unknown_phase_is_precision():
    assert zone_of("grab_container", "renamed", "") == ZoneEntry(Zone.PRECISION, False)


def test_unknown_capability_is_precision():
    assert zone_of("fridge", "view", "").zone is Zone.PRECISION


def test_approach_ends_precise():
    assert zone_of("grab_container", "approach", APPROACH).precise_end


def test_a_new_message_in_a_known_grab_container_phase_is_precision():
    assert zone_of("grab_container", "view", "something new") == ZoneEntry(Zone.PRECISION, False)


@pytest.mark.parametrize(
    "phase,message,zone",
    [
        ("idle", "", Zone.FREE),
        ("home", "", Zone.FREE),
        ("locate", "", Zone.PRECISION),
        ("view", "", Zone.FREE),
        ("detect", "", Zone.PRECISION),
        ("grasp", "opening", Zone.FREE),
        ("grasp", APPROACH, Zone.FREE),
        ("grasp", "closing", Zone.PRECISION),
        ("raise", "", Zone.PRECISION),
        ("approach", "dry run: planning", Zone.PRECISION),
        ("raise", "dry run: planning", Zone.PRECISION),
        ("done", "", Zone.FREE),
        ("aborted", "", Zone.FREE),
    ],
)
def test_grab_container_recorded_pairs(phase, message, zone):
    assert zone_of("grab_container", phase, message).zone is zone


def test_only_the_approach_move_is_precise_ended():
    assert not zone_of("grab_container", "grasp", APPROACH).precise_end
    assert not zone_of("grab_container", "home", "").precise_end


@pytest.mark.parametrize("phase", ["detect", "redetect", "plan", "press", "return"])
def test_button_press_precision_phases_with_the_nodes_dynamic_message(phase):
    assert zone_of("button_press", phase, f"dome: {phase}").zone is Zone.PRECISION


def test_button_press_transit_and_pre_press_move():
    assert zone_of("button_press", "stage", "dome: stage") == ZoneEntry(Zone.FREE, False)
    assert zone_of("button_press", "approach", "dome: approach") == ZoneEntry(Zone.FREE, True)
    assert zone_of("button_press", "done", "done").zone is Zone.FREE


def test_button_press_unknown_phase_is_precision():
    assert zone_of("button_press", "renamed", "dome: renamed").zone is Zone.PRECISION
