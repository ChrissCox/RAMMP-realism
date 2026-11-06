"""The playground's stage pose and the candidate path to it, on flatbot's empty chair.

Rest, then the candidate path (escape.escape_path: climb out of the occupant keep-out, blend to the
stage pose, retimed to the FREE caps), a hold at the stage pose, and the same path back. The path
is a CANDIDATE for the user to approve; nothing here moves a robot.

`python -m puppy_sim.clips.playground_stage` writes media/playground_stage_pose.png (the still) and
puppy_playground/puppy_playground/data/stage_pose.json."""

import hashlib
import json
import subprocess
from dataclasses import dataclass

from PIL import Image, ImageDraw

from .. import render, scene
from . import Clip
from .playground_common import (
    FLATBOT,
    LEGEND,
    REST_GRIP,
    STAGE_POSE_JSON,
    Assembler,
    cached,
    chain,
    chair,
    points_q,
    stage_pose,
    urdf_text,
)

HOLD_S = 2.5
STILL_PATH = render.MEDIA_DIR / "playground_stage_pose.png"
CHIP_PATH = "CANDIDATE PATH, NOT APPROVED"


def plan():
    """(rest q, stage pose, path points rest -> stage): cached on puppy_motion's source."""
    from puppy_motion import stage

    def make():
        ch, sc = chain(), chair()
        q_stage = stage_pose(ch, sc)
        try:
            from puppy_motion import escape

            points, how = escape.escape_path(ch, sc, stage.REST_Q, q_stage), "escape_path"
        except Exception as e:  # no escape module yet, or it found no route: the plain joint-space path
            first = stage.min_jerk_path(stage.REST_Q, q_stage, 8.0)
            points, how = stage.retime_path(first, ch), f"min_jerk_path (escape_path: {type(e).__name__}: {e})"
        found = stage.path_violations(points, ch, sc)
        return stage.REST_Q.copy(), q_stage, points, how, [f"{v.rule}: {v.detail}" for v in found]

    return cached("playground_stage_plan", make)


def build():
    q_rest, q_stage, points, how, found = plan()
    path = points_q(points)
    label = CHIP_PATH
    a = Assembler(q_rest)
    a.hold(1.0, phase="rest")
    a.add(path, chip=label, phase="rest to stage")
    a.hold(HOLD_S, phase="stage pose")
    a.add(path[::-1], chip=label + " (back)", phase="stage to rest")
    a.hold(1.0, phase="rest")
    track = a.track("playground_stage")
    track.log.append(f"path by {how}")
    return track


@dataclass(frozen=True)
class StageClip(Clip):
    def subtitle_for(self, tracks):
        _, _, points, how, found = plan()
        kind = "escape_path" if how == "escape_path" else "min_jerk_path (escape_path found no route)"
        verdict = f"breaks {len(found)} stage-path rule(s): {found[0][:60]}" if found else "meets every stage-path rule"
        return [
            f"stage pose = first of find_stage_poses (seed 0); path by {kind}, {points[-1][0]:.0f} s each way",
            f"{LEGEND}",
            f"Candidate path, not approved: {verdict}",
        ]


CLIP = StageClip(
    name="playground_stage",
    title="Playground stage pose and candidate path",
    subtitle="",
    camera="playground_wide",
    build=build,
    run=FLATBOT,
)


# ------------------------------------------------------------------------------- the still
def clearance_lines(ch, sc, q):
    from puppy_motion.geometry import PARTS, clearances

    c = clearances(ch, sc, q, False)
    person = ", ".join(f"{p} {c[0, i]:.2f} m" for i, p in enumerate(PARTS) if p != "upper_arm")
    return [f"clearance to the occupant keep-out (need 0.35 m): {person}", f"tool to the seat / base (need 0.10 m): {c[1, PARTS.index('head')]:.2f} m"]


def render_still(path=STILL_PATH, size=(1600, 900)):
    ch, sc = chain(), chair()
    _, q_stage, *_ = plan()
    sce = scene.build(FLATBOT)
    stage = render.Stage(sce, render.CAMERAS["playground_still"], size)
    img = Image.fromarray(stage.frame(q_stage, REST_GRIP, False, render.CAMERAS["playground_still"].lookat))
    stage.close()
    layer = Image.new("RGBA", img.size)
    render.draw_title(
        ImageDraw.Draw(layer),
        "Playground stage pose",
        [LEGEND, *clearance_lines(ch, sc, q_stage)],
        img.width / render.SINGLE_SIZE[0],
    )
    out = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path


def write_stage_pose_json(path=STAGE_POSE_JSON):
    _, q_stage, *_ = plan()
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%h", "--", "puppy_motion/puppy_motion/stage.py"],
        capture_output=True, text=True, cwd=STAGE_POSE_JSON.parents[3],
    ).stdout.strip()
    doc = {
        "q": [float(v) for v in q_stage],
        "urdf_sha256": hashlib.sha256(urdf_text().encode("utf-8")).hexdigest(),
        "rules": f"stage.py@{commit}",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


if __name__ == "__main__":
    print(render_still())
    print(write_stage_pose_json())
