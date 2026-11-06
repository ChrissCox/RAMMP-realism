"""Write puppy_playground/data/stage_pose.json: the first stage pose find_stage_poses finds
(seed 0), continuous joints unwrapped to the turn nearest rest on the chair scene, for the URDF given as argv[1] (a URDF file; default: the flatbot
fixture used by the CI image).

    python scripts/make_stage_pose.py path/to/robot.urdf
"""
import json
import subprocess
import sys
from pathlib import Path

from puppy_motion.flatbot import chair_scene
from puppy_motion.escape import escape_path, unwrap_to
from puppy_motion.stage import REST_Q, find_stage_poses, path_violations
from puppy_motion.urdf import Chain
import hashlib

urdf = Path(sys.argv[1] if len(sys.argv) > 1 else "/fixtures/urdf/flatbot.urdf").read_text(encoding="utf-8")
chain = Chain.from_urdf(urdf)
# continuous joints are turned to the whole-turn equivalent nearest the rest pose (same posture)
# The first pose (in find_stage_poses order) the escape leg can actually reach from rest: some poses
# have no route that keeps the forearm, wrist and head 0.35 m from the person all the way.
q = None
for cand in find_stage_poses(chain, chair_scene(), n=3, seed=0):
    cand = unwrap_to(chain, REST_Q, cand)
    try:
        path = escape_path(chain, chair_scene(), REST_Q, cand)
    except ValueError as e:
        print("skipped a pose:", e)
        continue
    if path_violations(path, chain, chair_scene()) == []:
        q = cand
        break
if q is None:
    raise SystemExit("no stage pose with a checked escape path")
sha = subprocess.run(
    ["git", "log", "-1", "--format=%h", "--", ":(top)puppy_motion/puppy_motion/stage.py"],
    cwd=Path(__file__).resolve().parent,
    capture_output=True, text=True, check=True,
).stdout.strip()
out = Path(__file__).resolve().parent.parent / "puppy_playground" / "data" / "stage_pose.json"
out.write_text(
    json.dumps({"q": [float(x) for x in q], "urdf_sha256": hashlib.sha256(urdf.encode("utf-8")).hexdigest(), "rules": f"stage.py@{sha}"}, indent=2) + "\n",
    encoding="utf-8",
)
print(out, q)
