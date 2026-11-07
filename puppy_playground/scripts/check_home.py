"""Read-only home-pose checker: can the Puppy Playground plan the stage path from this pose?

    python3 check_home.py [--q "[7 joint angles in rad]"] [--urdf /data/flatbot.urdf]
                          [--write-candidate] [--ignore-start-gate] [--floor-cap 0.02]

Without --q it takes one /joint_states reading (a read-only subscription). It prints the pose, the
plan_stage start gate exactly as the engine computes it, then every offline planner the engine
uses (min_jerk, escape, via) with the engine's own candidate generation and path checks. cuRobo
needs the live planner and is not tried here: Plan stage may still find a cuRobo route.
Nothing moves. Exit 0: the gate passes and a planner passes. 2: the gate fails. 3: no planner passes.
--write-candidate saves /data/stage_path.candidate.json (as plan_stage does) when a planner passes.
--ignore-start-gate skips only the gate (an attended operator's call); every path rule still applies.
--floor-cap 0.02 (only 0.0, the default, or 0.02) checks with the ATTENDED floor cap: each part's person
floor is min(its first sample, 20 mm). For attended tests with NOBODY in the chair. The output names the
rule used, and --write-candidate records the cap in the file (which then only replays under
stage_floor_cap_m:=0.02).
"""

import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np

from puppy_motion.geometry import PARTS
from puppy_motion.joints import shortest_delta
from puppy_motion.urdf import Chain
from puppy_playground import stage_store
from puppy_playground.engine import (
    ATTENDED_FLOOR_CAP_M, CANDIDATE_PATH, START_OBSTACLE_MIN_M, START_PERSON_MIN_M, STILL_RAD_S, PuppyPlayground,
)

PLANNERS = ("min_jerk", "escape", "via")  # curobo needs the live planner
JOINT_NAMES = [f"arm_joint_{i}" for i in range(1, 8)]
READ_TIMEOUT_S = 10.0
DEFAULT_URDF = "/data/flatbot.urdf"

# _candidate_points and _path_violations use `self` only to reach _via_points and the floor cap
def _engine(floor_cap_m=0.0):
    return SimpleNamespace(_via_points=PuppyPlayground._via_points, stage_floor_cap_m=floor_cap_m)


def read_joint_states():
    """One /joint_states reading: arm_joint_1..7 by name. Refuses if the arm moves or no message comes."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("puppy_check_home")
    got = []
    node.create_subscription(JointState, "/joint_states", got.append, qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + READ_TIMEOUT_S
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            for msg in got:
                if all(n in msg.name for n in JOINT_NAMES):
                    idx = [list(msg.name).index(n) for n in JOINT_NAMES]
                    pos = np.array([msg.position[i] for i in idx], dtype=float)
                    vel = np.array([msg.velocity[i] for i in idx], dtype=float) if len(msg.velocity) == len(msg.name) else np.full(7, np.nan)
                    if not (np.abs(vel).max() < STILL_RAD_S):
                        raise SystemExit(f"the arm is moving (or reports no velocity): max |velocity| {np.nanmax(np.abs(vel)) if np.isfinite(vel).any() else float('nan'):.3f} rad/s, needs < {STILL_RAD_S}")
                    return pos
            got.clear()
        raise SystemExit(f"no /joint_states message with {JOINT_NAMES[0]}..{JOINT_NAMES[-1]} in {READ_TIMEOUT_S:.0f} s")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _pose_lines(q):
    deg = np.rad2deg(q)
    signed = (deg + 180.0) % 360.0 - 180.0
    signed = np.where(np.isclose(signed, -180.0), 180.0, signed)
    return [
        "pose (deg, signed):  " + ", ".join(f"{d:.1f}" for d in signed),
        "pose (deg, 0-360):   " + ", ".join(f"{d % 360.0:.1f}" for d in deg),
        "pose (rad):          " + ", ".join(f"{x:.4f}" for x in q),
    ]


def check(q, urdf_text, ignore_gate=False, write_candidate=None, floor_cap_m=0.0):
    """Returns (exit_code, lines, found) where found is (method, points) or None. Moves nothing.
    `write_candidate`: a path to save the first passing planner's path to (stage_store.save).
    `floor_cap_m`: 0.0 (each part's person floor is its first sample) or ATTENDED_FLOOR_CAP_M."""
    if floor_cap_m not in (0.0, ATTENDED_FLOOR_CAP_M):
        raise ValueError(f"floor_cap_m must be 0.0 or {ATTENDED_FLOOR_CAP_M}, not {floor_cap_m!r}")
    engine = _engine(floor_cap_m)
    q = np.asarray(q, dtype=float)
    chain = Chain.from_urdf(urdf_text)
    stage_q = PuppyPlayground._stage_q(chain)
    out = _pose_lines(q)

    out.append("")
    out.append("start gate (the engine's plan_stage gate, on the scene grown by "
               f"{stage_store.MARGIN_M * 1000:.0f} mm):")
    person, obstacle = PuppyPlayground._clearances(chain, q, stage_store.MARGIN_M)
    for i, part in enumerate(PARTS):
        out.append(
            f"  {part:9s} person {person[i] * 1000:7.0f} mm (min {START_PERSON_MIN_M[part] * 1000:.0f})"
            f"   obstacle {obstacle[i] * 1000:7.0f} mm (min {START_OBSTACLE_MIN_M * 1000:.0f})"
        )
    why = PuppyPlayground._bad_start(chain, q)
    gate_ok = why is None
    out.append("  joint limits and guards, clearances: " + ("PASS" if gate_ok else f"FAIL: {why}"))
    if not gate_ok and ignore_gate:
        out.append("*** START GATE IGNORED (--ignore-start-gate): the gate failed and was skipped; every path rule still applies ***")
    elif not gate_ok:
        out.append("Plan stage would refuse this pose (bad_start). Move the arm and check again.")
        return 2, out, None

    target = q + shortest_delta(chain, q, stage_q)  # the stage pose on the turn nearest the pose
    out.append("")
    if floor_cap_m:
        out.append(f"*** ATTENDED FLOOR CAP {floor_cap_m * 1000:.0f} mm: person floor = min(first sample, {floor_cap_m * 1000:.0f} mm) per part; "
                   "for attended tests with NOBODY in the chair ***")
    else:
        out.append("person floor rule: each part may not fall below its first sample (no floor cap)")
    out.append("offline planners (cuRobo needs the live planner and is not tried here; Plan stage may still find a cuRobo route):")
    found = None
    for method in PLANNERS:
        try:
            points = PuppyPlayground._candidate_points(engine, method, chain, q, target, {}, None)
        except ValueError as e:
            out.append(f"  {method:9s} {type(e).__name__}: {e}")
            continue
        if points is None:
            out.append(f"  {method:9s} no plan")
            continue
        bad = [f"{v.rule}: {v.detail}" for v in PuppyPlayground._path_violations(engine, points, chain, stage_store.MARGIN_M)]
        if not bad:
            stub = stage_store.StagePath(points, points[0][1], target, "", 0.0, "")
            try:
                stage_store.check_endpoints(stub, chain, stage_q)
            except stage_store.StageStoreError as e:
                bad = [f"{e.code}: {e}"]
        if bad:
            out.append(f"  {method:9s} FAIL: " + " | ".join(bad[:3]))
            continue
        out.append(f"  {method:9s} PASS, {points[-1][0] - points[0][0]:.1f} s, {len(points)} points")
        found = (method, points)
        break

    if found is None:
        out.append("No offline planner passes from this pose.")
        return 3, out, None
    if write_candidate:
        method, points = found
        stage_store.save(write_candidate, points, urdf_text, stage_q, method=method, person_floor_cap_m=floor_cap_m,
                         start_gate="skipped" if (ignore_gate and not gate_ok) else "passed")
        with open(write_candidate, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        out.append(f"WROTE {write_candidate} (method {method})")
        out.append(f"sha256 {sha}")
    return 0, out, found


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only: can the Puppy Playground plan the stage path from this pose?")
    ap.add_argument("--q", help='seven joint angles in radians, e.g. "[0.2, -0.1, ...]" (default: one /joint_states reading)')
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--write-candidate", action="store_true", help=f"save {CANDIDATE_PATH} when a planner passes")
    ap.add_argument("--ignore-start-gate", action="store_true", help="skip only the start-clearance gate (attended operator's call)")
    ap.add_argument("--floor-cap", type=float, default=0.0, choices=(0.0, ATTENDED_FLOOR_CAP_M),
                    help="0.0 (default, off) or 0.02: the ATTENDED person-floor cap, nobody in the chair")
    args = ap.parse_args(argv)
    if args.q is not None:
        try:
            q = np.array(json.loads(args.q), dtype=float)
        except (ValueError, TypeError) as e:
            sys.exit(f"--q must be a JSON list of 7 numbers: {e}")
        if q.shape != (7,) or not np.isfinite(q).all():
            sys.exit("--q must be 7 finite numbers (radians)")
    else:
        q = read_joint_states()
    with open(args.urdf, encoding="utf-8") as f:
        urdf_text = f.read()
    code, lines, _ = check(q, urdf_text, ignore_gate=args.ignore_start_gate,
                           write_candidate=CANDIDATE_PATH if args.write_candidate else None,
                           floor_cap_m=args.floor_cap)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
