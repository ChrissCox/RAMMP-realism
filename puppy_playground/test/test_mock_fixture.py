"""scripts/make_mock_fixture.py (run by mock_smoke.sh inside the image) writes a stage path the
engine's own store accepts, at the margin the engine checks with."""

import importlib.util
import json
from importlib import resources
from pathlib import Path

import numpy as np
from fake_driver import FLATBOT_URDF

from puppy_motion.urdf import Chain
from puppy_playground import stage_store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_mock_fixture.py"


def test_fixture_loads_at_the_engine_margin(tmp_path):
    spec = importlib.util.spec_from_file_location("make_mock_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(str(FLATBOT_URDF), str(tmp_path))

    urdf = FLATBOT_URDF.read_text()
    text = resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text(encoding="utf-8")
    stage_q = np.array(json.loads(text)["q"])
    sp = stage_store.load(str(tmp_path / "stage_path.json"), urdf, stage_q, Chain.from_urdf(urdf))
    assert sp.margin_m == stage_store.MARGIN_M
    assert len(sp.points) >= 2
