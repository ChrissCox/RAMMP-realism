"""check_home.py: the read-only home-pose checker (no live ROS needed; --q and a URDF file)."""

import importlib.util
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest
from fake_driver import FLATBOT_URDF

from puppy_motion.stage import REST_Q
from puppy_motion.urdf import Chain
from puppy_playground import stage_store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_home.py"
PARKED = np.deg2rad([11.7, -7.9, -176.2, -126.5, -17.0, 55.0, 90.0])
INSIDE_OCCUPANT = np.deg2rad([22.9, 114.2, -43.4, -71.2, -15.7, 36.2, -143.6])


@pytest.fixture(scope="module")
def check_home():
    spec = importlib.util.spec_from_file_location("check_home", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def urdf_text():
    return FLATBOT_URDF.read_text()


def test_rest_passes_the_gate_and_finds_a_planner(check_home, urdf_text):
    code, lines, found = check_home.check(REST_Q, urdf_text)
    assert code == 0, "\n".join(lines)
    assert found is not None and found[0] in check_home.PLANNERS
    text = "\n".join(lines)
    assert "PASS" in text and "cuRobo" in text and "pose (deg, signed)" in text


def test_the_tested_park_finds_via(check_home, urdf_text):
    code, lines, found = check_home.check(PARKED, urdf_text)
    assert code == 0, "\n".join(lines)
    assert found[0] == "via"


def test_a_pose_inside_the_occupant_box_fails_the_gate(check_home, urdf_text):
    code, lines, found = check_home.check(INSIDE_OCCUPANT, urdf_text)
    assert code == 2 and found is None
    assert "FAIL" in "\n".join(lines) and "chair occupant" in "\n".join(lines)


def test_ignore_start_gate_says_so_and_still_applies_the_path_rules(check_home, urdf_text):
    code, lines, _ = check_home.check(INSIDE_OCCUPANT, urdf_text, ignore_gate=True)
    text = "\n".join(lines)
    assert "START GATE IGNORED" in text and "offline planners" in text
    assert code in (0, 3)


def test_write_candidate_writes_a_file_stage_store_loads(check_home, urdf_text, tmp_path):
    dest = tmp_path / "stage_path.candidate.json"
    code, lines, found = check_home.check(REST_Q, urdf_text, write_candidate=str(dest))
    assert code == 0, "\n".join(lines)
    chain = Chain.from_urdf(urdf_text)
    stage_q = np.array(json.loads(
        resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text(encoding="utf-8"))["q"])
    sp = stage_store.load(str(dest), urdf_text, stage_q, chain)
    assert json.loads(dest.read_text())["method"] == found[0]
    assert any(l.startswith("sha256 ") and l.split()[1] == sp.file_sha256 for l in lines)


def test_main_exit_codes_and_no_candidate_on_a_gate_failure(check_home, tmp_path, capsys):
    urdf = tmp_path / "flatbot.urdf"
    urdf.write_text(FLATBOT_URDF.read_text())
    rc = check_home.main(["--q", json.dumps(list(INSIDE_OCCUPANT)), "--urdf", str(urdf)])
    assert rc == 2 and "FAIL" in capsys.readouterr().out
