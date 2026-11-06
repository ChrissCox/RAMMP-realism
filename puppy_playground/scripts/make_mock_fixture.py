"""Write an approved stage_path.json for a URDF (run inside the image by scripts/mock_smoke.sh).

    python3 make_mock_fixture.py <urdf> <out_dir>

The path is the engine's own escape planner from the rest pose to the stage pose, checked with the
stage-path rules and written by stage_store.save, so the engine's load checks pass."""

import json
import sys
from importlib import resources
from pathlib import Path

import numpy as np

from puppy_motion import stage as rules
from puppy_motion.escape import escape_path, grown_scene
from puppy_motion.flatbot import chair_scene
from puppy_motion.urdf import Chain
from puppy_playground import stage_store


def main(urdf_path, out_dir):
    text = Path(urdf_path).read_text()
    chain = Chain.from_urdf(text)
    stage_q = np.array(
        json.loads(resources.files("puppy_playground").joinpath("data/stage_pose.json").read_text())["q"]
    )
    points = escape_path(chain, chair_scene(), rules.REST_Q, stage_q)
    bad = rules.path_violations(points, chain, grown_scene(chair_scene(), stage_store.MARGIN_M))
    if bad:
        sys.exit(f"the fixture path breaks the stage rules: {bad[0].rule}: {bad[0].detail}")
    out = Path(out_dir) / "stage_path.json"
    stage_store.save(str(out), points, text, stage_q, method="escape")
    print(f"wrote {out} ({len(points)} points)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
