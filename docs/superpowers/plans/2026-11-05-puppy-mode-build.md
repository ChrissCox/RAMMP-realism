# Puppy Mode Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Puppy Mode motion library, rebuild the simulator on it, and build the engine hook
that plugs it into RAMMP capabilities, proven against the fake driver. Nothing moves real hardware.

**Architecture:**
- `puppy_motion` is a pure-Python library (numpy, pyyaml) that holds the safety logic. It covers:
  - URDF kinematics and capsule clearances;
  - the timing-only time-warp and the composer;
  - the 23 behaviors;
  - an *online* director that decides beats event by event;
  - an independent beat validator.
- `sim/` is the MuJoCo preview, moved out of `spike/`. It drives its motion through `puppy_motion` and
  replays recorded runs by feeding events to the director in order.
- `puppy_mode_ros` is an ament_python package. It wraps a capability's `ctx` and `self.arm` in
  proxies while `puppy.enabled` is true, and plays validated beats with
  `self.arm.execute_joint_trajectory`.
- A pinned `capability_engine` gets a small patch (one hook point), kept as small as an upstream PR
  would be.

**Tech Stack:**
- Library: Python 3.10+, numpy (1.26 and 2.x), pyyaml, pytest via `uv`.
- Simulator: MuJoCo 3.10, imageio-ffmpeg, Pillow.
- Hook: ROS 2 Humble (rclpy) in RAMMP's x86 test image, run in GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-05-puppy-mode-design.md` (read it first). The throwaway spike
in `spike/` is the reference implementation for every algorithm ported here; its README explains each
module.

## Global Constraints

**Runtime and environment**
- `puppy_motion` runs on Python 3.10 with numpy 1.26.4 and with numpy ≥ 2. It uses no `scipy`, no
  `mujoco`, no 3.11+ syntax (`Self`, `ExceptionGroup`, `tomllib`), and nothing else at runtime but
  `numpy` and `pyyaml`.
- Kinematics come from the URDF text the driver publishes (`/arm/robot_description`), never from a
  hard-coded mount.
- Nothing commands a real robot. No test, script or tool connects to snorlax or to any robot network,
  and the fake driver never runs on snorlax.
- Every hook test runs inside the x86 test image (GitHub Actions or local Docker).

**Rules that must always hold**
- `puppy.enabled` defaults to `false`. With it false, the driver receives exactly the same goals as
  without the package.
- Never call `/acquire_control`, `/release_control` or `/revoke_control`, and never publish `/estop`.
- Clock: `DT = 0.02` s (50 Hz).
- Zones and gains:
  - FREE 1.0, NEAR_USER 0.35, PRECISION exactly 0.
  - Gain ramps are raised-cosine, 0.3 s long, and finish before a lower zone begins.
  - NEAR_USER means the head (bracelet, face, gripper, held object) is within 0.35 m of the person.
- Inside PRECISION the commanded `q` is bit-identical to the task's `q` at the same task time, and
  `dτ/dt ∈ [0, 1]`.
- A (phase, message) pair the zone table does not know maps to PRECISION.

**Caps for the expressive part** (preset `default`)
- FREE: joint speed `[0.36, 0.85, 0.59, 0.64, 0.85, 0.85, 0.85]` rad/s, joint accel 3.0 rad/s², head
  speed 0.40 m/s at the eye and tool points.
- NEAR_USER: 0.5 rad/s, 1.5 rad/s², 0.16 m/s.
- Preset `snappy` is the same except wrist joints 5-7 get 9.0 rad/s² in FREE. It is never the default.
- The combined motion stays within the URDF velocity limits (1.3963 rad/s for joints 1-4, 1.2218 for
  joints 5-7).
- Joint guards: |q2| ≤ 2.24 rad, |q4| ≤ 144°, |q6| ≤ 115°.

**Clearance floors**
- Person: every arm part keeps at least min(task clearance, 0.35 m) − 0.005 m.
- Obstacles: min(task clearance, 0.03 m).

**Budgets per task, in seconds:** `demo` 6.0, `subtle` 2.5.

**Beats on the robot**
- Beats start and end at the measured posture with zero velocity, and only while every joint is
  slower than 0.02 rad/s.
- They move the arm only; gripper beats stay simulator-only.
- Control mode: copy the most recent move call's `control_mode` and `gains`. In NEAR_USER use
  IMPEDANCE, with those gains or the `default` profile.
- Results: only `SUCCESS` and recoverable `EngineError` get reactions. `Cancelled`, `EStopped`,
  `ControlTaken` and any other exception get none.
- A beat never changes a task's result. Between moves, `Cancelled`, `EStopped` and `ControlTaken`
  propagate; any other refusal turns beats off for the rest of the action.

**Work branch:** `feature/puppy-mode-build` off `main`, merged by PR. No PR to any `rammp-org` repo.

## Review Focus

1. **A capability reports a phase string the zone table doesn't know** (it was renamed). Expected: the
   phase is treated as PRECISION, no beats play in it, and a warning names the string. Tested in
   Task 13.
2. **The arm is still moving or settling when a beat would start.** Expected: the beat is skipped and
   logged; nothing is sent. Tested in Task 12.
3. **A cancel arrives in the middle of a beat.** Expected: `Cancelled` propagates, the arm is left
   still, and no further beats play. Tested in Task 12.
4. **`puppy.enabled` is true but `puppy.person_zone` is empty or malformed.** Expected: Puppy Mode
   refuses to enable, logs why, and the capability runs exactly as with Puppy Mode off. Tested in
   Task 13.
5. **The driver refuses a beat's trajectory, or joint state is stale (`StateTimeout`).** Expected:
   beats turn off for this action, the task continues, and its result is unchanged. Tested in Task 12.

---

## Part A — `puppy_motion` library

Layout (new): `puppy_motion/pyproject.toml`, `puppy_motion/puppy_motion/*.py`,
`puppy_motion/puppy_motion/data/zones.yaml`, and `puppy_motion/tests/`, with test data copied from
`spike/data/runs/*.json`.

The test command, run from `puppy_motion/`, has two variants:

- py310/np1: `uv run --python 3.10 --with-editable . --with numpy==1.26.4 --with pytest pytest -q`
- py312/np2: `uv run --python 3.12 --with-editable . --with "numpy>=2" --with pytest pytest -q`

Both must pass at every commit.

### Task 1: Package, CI and URDF kinematics

**Files:**
- Create: `puppy_motion/pyproject.toml` (name `puppy-motion`, `requires-python = ">=3.10"`,
  dependencies `numpy>=1.24`, `pyyaml>=5.4`).
- Create: `puppy_motion/puppy_motion/__init__.py` and `puppy_motion/puppy_motion/urdf.py`.
- Create: `.github/workflows/puppy_motion.yml`, a matrix over {3.10 + numpy 1.26.4, 3.12 + numpy
  2.x} running the test command.
- Create: `puppy_motion/tests/data/` (copy the three run JSONs) and `puppy_motion/tests/test_urdf.py`.

**Interfaces:**
- Produces: `Chain.from_urdf(urdf_text: str) -> Chain`. The root is the URDF root link; the chain
  covers the 7 revolute or continuous joints whose names end in `joint_1`…`joint_7`.
- Produces: `Chain.joint_names: tuple[str, ...]`, `Chain.vel_limits: np.ndarray (7,)` and
  `Chain.pos_limits: np.ndarray (7, 2)`, with ±inf for continuous joints.
- Produces: `Chain.fk(q, site="ee") -> tuple[np.ndarray (3,), np.ndarray (3, 3)]` and
  `Chain.jacobian(q, site="ee") -> np.ndarray (6, 7)` (linear rows first, world frame).
- Sites:
  - `"ee"`: the link whose name ends in `end_effector_link`;
  - `"tool"`: `ee` + 0.120 m along ee z;
  - `"eye"`: `ee` + (0, 0.065, 0.015) in the ee frame;
  - `"neck"`: the wrist centre, 0.1675 m behind ee along its z.

- [ ] **Step 1: Write the failing tests**

```python
def test_fk_matches_the_recorded_end_effector():   # all three runs, joints interpolated to each /ee_state stamp
    assert max_pos_err_mm < 0.1 and max_rot_err_deg < 0.02   # /ee_state = ee in the URDF root (arm_mount)
def test_tool_is_120_mm_along_ee_z(): ...           # |p_tool - (p_ee + R_ee[:, 2]*0.120)| < 1e-9
def test_jacobian_matches_finite_differences():     # 20 random q, eps 1e-6, every site
    assert np.abs(J - J_fd).max() < 1e-5
def test_guards_and_limits_from_the_urdf():
    assert chain.vel_limits.tolist() == [1.3963]*4 + [1.2218]*3
```

- [ ] **Step 2: Run both test commands; expect an ImportError for `puppy_motion.urdf`.**
- [ ] **Step 3: Implement `urdf.py`** with an `xml.etree` parse, origin rpy → R, and
  product-of-transforms FK. The Jacobian is analytic: z-axis × (p_site − p_joint) for the linear
  rows and z-axis for the angular rows. The verifier's numpy chain in the spike is the reference
  (`spike/README.md`, "What is real").
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): URDF chain, FK and Jacobian validated on recorded runs`.

### Task 2: Track, zones and the time-warp

**Files:**
- Create: `puppy_motion/puppy_motion/track.py` and `puppy_motion/puppy_motion/zones.py`.
- Create: `puppy_motion/puppy_motion/data/zones.yaml`.
- Create: `puppy_motion/tests/test_track.py` and `puppy_motion/tests/test_zones.py`.

**Interfaces:**
- Produces: `DT = 0.02`; `class Zone(IntEnum): FREE = 0; NEAR_USER = 1; PRECISION = 2`.
- Produces: `@dataclass Track` with fields `q (N, 7)`, `grip (N,)`, `zone (N,)`, `held (N,)`,
  `tau (N,)`, `labels: list[Label]` and `source: Track | None`, plus `.t` and `.n`.
- Produces: `Label(name: str, t0: float, t1: float)`.
- Produces: `Hold(at: float, hold_s: float, ramp_s: float)` and
  `Slow(t0: float, t1: float, factor: float, ramp_s: float)`. A `Slow` raises `ValueError` unless
  `0 < factor <= 1`.
- Produces: `warp(track, holds=(), slows=()) -> Track` and `constant_track(q, grip, seconds) -> Track`.
- Produces: `joint_move(q0, q1, seconds, grip) -> Track` (minimum-jerk) and
  `concat(tracks) -> Track`.
- Produces: `ZoneEntry(zone: Zone, precise_end: bool)`.
- Produces: `zone_of(capability: str, phase: str, message: str) -> ZoneEntry`. It loads `zones.yaml`,
  and an unknown pair returns `ZoneEntry(Zone.PRECISION, False)`.
- `zones.yaml` content:
  - `grab_container` gets the recorded pairs (from `spike/puppy_sim/runs.py` `ZONE_TABLES`), plus
    `("approach", "view -> pre-grasp -> grasp, one motion")` → FREE with `precise_end: true`, and
    `("dry run: planning")` → PRECISION.
  - `button_press` gets the pairs read from `ctx.progress` in `feature/4-button-press`
    `button_press/node.py` and `press_button.py`; the press and contact phases are PRECISION.

- [ ] **Step 1: Write the failing tests.** Port `spike/tests/test_timeline.py` (10 tests) and the
  warp half of `spike/tests/test_guarantee.py`:
  - `test_random_pauses_and_slow_ins_change_only_the_timing`;
  - `test_the_time_warp_refuses_events_that_are_not_timing_only`;
  - `test_a_hold_at_the_fastest_moment_adds_at_most_the_accel_cap`.

  Adapt the imports only; keep their assertions. Add:

```python
def test_unknown_phase_is_precision():
    assert zone_of("grab_container", "renamed", "") == ZoneEntry(Zone.PRECISION, False)
def test_unknown_capability_is_precision():
    assert zone_of("fridge", "view", "").zone is Zone.PRECISION
def test_approach_ends_precise():
    assert zone_of("grab_container", "approach", "view -> pre-grasp -> grasp, one motion").precise_end
```

- [ ] **Step 2: Run both test commands; expect FAIL.**
- [ ] **Step 3: Port** `Track`, `Hold`, `Slow`, `warp`, the scripted moves and `concat` from
  `spike/puppy_sim/timeline.py`. Keep the even-sample, round-up ramp fix and `source`.
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): tracks, zone tables and the timing-only time-warp`.

### Task 3: Capsule geometry and clearances

**Files:**
- Create: `puppy_motion/puppy_motion/geometry.py` and `puppy_motion/tests/test_geometry.py`.

**Interfaces:**
- Consumes: `Chain` (Task 1).
- Produces: `Capsule(a, b, r)`, `Sphere(c, r)`, `Box(center, half, R=np.eye(3))` and
  `Scene(person: list[Shape], obstacles: list[Shape])`.
- Produces: `Scene.from_param(text: str) -> Scene`. The text is JSON:
  `{"person": {"spheres": [[x, y, z, r], ...], "capsules": [[ax, ay, az, bx, by, bz, r], ...],
  "boxes": [[cx, cy, cz, hx, hy, hz], ...]}, "obstacles": {...}}`, in the URDF root frame. It raises
  `ValueError` if `person` is missing or empty.
- Produces: `Scene.person_head() -> np.ndarray (3,)`, the centre of the highest person sphere.
- Create: `puppy_motion/tests/data/spike_person.json`. It holds the spike mannequin's head sphere,
  torso and limb capsules, copied from `spike/puppy_sim/scene.py` `_add_person`.
- Produces: `PARTS = ("upper_arm", "forearm", "wrist", "head")`.
- Produces: `arm_capsules(chain, q, held: bool) -> dict[str, list[Capsule]]`. Radii: upper arm 0.06,
  forearm 0.055, wrist 0.05, head 0.06 around ee→tool. A held object adds a 0.09 m sphere 0.17 m
  along ee z.
- Produces: `clearances(chain, scene, q, held) -> np.ndarray (2, 4)`. Row 0 is the distance to the
  person, row 1 to obstacles, and each column is one of `PARTS`. With no obstacles, row 1 is +inf.

- [ ] **Step 1: Write the failing tests**

```python
def test_capsule_sphere_distance_closed_form(): ...      # 6 hand-computed cases, abs err < 1e-12
def test_capsule_box_distance_matches_brute_force():      # 200 random pairs vs dense surface sampling
    assert abs(d - d_brute) < 2e-3
def test_spike_person_clearances_reproduced():            # tests/data/spike_person.json, run 132745
    assert minima_by_part ≈ {"upper_arm": 0.068, "forearm": 0.108, "wrist": 0.246, "head": 0.377} within 0.03 m
def test_scene_param_without_person_raises(): ...
```

- [ ] **Step 2: Run both test commands; expect FAIL.**
- [ ] **Step 3: Implement** the closed-form capsule–sphere distance and the capsule–box distance
  (segment–OBB, iterative closest point). Take the arm link frames from `Chain.fk` on the URDF links.
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): capsule model of the arm and clearance to person and obstacles`.

### Task 4: Caps, head offsets and the composer

**Files:**
- Create: `puppy_motion/puppy_motion/caps.py` and `puppy_motion/puppy_motion/compose.py`.
- Create: `puppy_motion/tests/test_compose.py`.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `CapSet(joint_speed: np.ndarray (7,), joint_accel: np.ndarray (7,), head_speed: float)`,
  `CAP_PRESETS: dict[str, dict[Zone, CapSet]]` with the Global Constraints values, and
  `URDF_GUARDS = {1: 2.24, 3: radians(144), 5: radians(115)}`.
- Produces: `Contribution(name, head (N, 6), joints (N, 7), grip (N,), labels: list[Label])`. Head
  offsets are `[dx, dy, dz` in world metres, `roll, pitch, yaw` in head-frame radians, applied
  yaw→pitch→roll about `neck`]`.
- Produces: `head_dq(chain, q, offset6) -> np.ndarray (7,)`: iterated weighted DLS with weights
  `[25, 25, 15, 8, 1, 1, 1]`, damping 0.03 and 4 iterations. A zero offset gives exactly zero.
- Produces: `compose(warped: Track, parts: list[Contribution], chain, scene, caps="default") ->
  Track`. The result carries `.gain (N,)` and `.log: list[str]`.

- [ ] **Step 1: Write the failing tests.** Port `spike/tests/test_layer.py` (8 tests) and the gain
  half of `test_guarantee.py`:
  - `test_gain_never_leaks_into_precision_for_any_zone_pattern`;
  - `test_precision_samples_are_bit_identical_under_a_large_offset`.

  Also port `spike/tests/test_kin.py::test_head_offset_is_realised_by_the_iterated_dls` and
  `::test_zero_offset_gives_exactly_zero`. Add:

```python
def test_free_caps_stay_inside_the_measured_envelope():   # spike/puppy_sim/envelope.py numbers
    assert (CAP_PRESETS["default"][Zone.FREE].joint_speed <= measured_peak).all()
def test_snappy_only_raises_wrist_accel_in_free():
    assert CAP_PRESETS["snappy"][Zone.FREE].joint_accel.tolist() == [3.0]*4 + [9.0]*3
```

- [ ] **Step 2: Run both test commands; expect FAIL.**
- [ ] **Step 3: Port** `spike/puppy_sim/layer.py` (gains, ramps, cap tiers, fitting by shrinking
  contributions, person and obstacle floors, toward-person projection in NEAR_USER) onto
  `geometry.clearances` instead of MuJoCo distances.
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): zone-tiered caps and the composer with its guarantees`.

### Task 5: Behaviors

**Files:**
- Create: `puppy_motion/puppy_motion/curves.py`, `puppy_motion/puppy_motion/behaviors.py` and
  `puppy_motion/tests/test_behaviors.py`.

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `Context.of(track, chain, scene, targets: Targets, seed: int, caps="default")`, with
  `Targets(person_head, container=None, button=None)` as world points.
- Produces: `Targets.from_scene(scene)` (only `person_head` is set). On the robot Puppy Mode has no
  perception of its own, so behaviors that aim at an object (`sniff`, `lock_on`, `curious`'s look)
  aim along the current gaze, 0.35 m from the eye, whenever that target is `None`. At the view pose
  the wrist camera is already looking at the object.
- Produces the 23 behavior functions with the spike's names and signatures. Each takes `ctx` first
  and returns `Contribution`:
  - at rest: `breathe`, `micro_life`, `gaze_to`, `look_at_user`, `head_tilt`, `perk`, `droop`,
    `puppy_eyes`, `doze_and_wake`;
  - reactions: `nod`, `bounce`, `wiggle`, `clack`, `sniff`, `double_take`, `search`, `startle`,
    `shake_off`, `proud`;
  - motion craft: `wind_up`, `lead_gaze`, `follow_through`, `lock_on`.

  `clack` writes only `grip`.

- [ ] **Step 1: Write the failing tests.** Port `spike/tests/test_behaviors.py` (19 tests) and
  `spike/tests/test_curves.py` (8 tests). The scene comes from `Scene.from_param` with the spike's
  mannequin spheres.
- [ ] **Step 2: Run both test commands; expect FAIL.**
- [ ] **Step 3: Port** `spike/puppy_sim/curves.py` and `behaviors.py`. Replace `kin.*` calls with
  `Chain` + `head_dq`, and keep every named constant and default.
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): the 23 puppy behaviors`.

### Task 6: Beat validator

**Files:**
- Create: `puppy_motion/puppy_motion/validate.py` and `puppy_motion/tests/test_validate.py`.

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: `Violation(rule: str, sample: int, detail: str)`.
- Produces: `check_beat(points: list[tuple[float, np.ndarray, np.ndarray]], q_start, chain, scene,
  zone: Zone, caps="default", task_clearance: np.ndarray (2, 4) | None = None) -> list[Violation]`.
  It writes nothing back to the composer. Rules:
  - `start` / `end`: the first and last points equal `q_start` with zero velocity (1e-9);
  - `dt`: the time step is `DT`;
  - `speed`, `accel`, `head_speed`: under the zone's caps;
  - `urdf_speed`;
  - `limits`;
  - `guards`;
  - `person` and `obstacle`: the clearance floors.

- [ ] **Step 1: Write the failing tests.** One passing beat (a composed `head_tilt` at home). Then one
  case per rule, each a minimal mutation of that beat that breaks exactly that rule, asserting
  `[v.rule for v in check_beat(...)] == ["<rule>"]`.
- [ ] **Step 2: Run both test commands; expect FAIL.**
- [ ] **Step 3: Implement** the checks independently of `compose.py`; do not import its fitting code.
  This mirrors the spike's `checks.py`.
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): independent beat validator`.

### Task 7: Online director

**Files:**
- Create: `puppy_motion/puppy_motion/director.py`, `puppy_motion/puppy_motion/data/rules.yaml` and
  `puppy_motion/tests/test_director.py`.

**Interfaces:**
- Consumes: Tasks 2–6.
- Produces the event types, each a frozen dataclass: `TaskRequested()`,
  `Phase(name: str, message: str)`, `BeforeMove(precise_end: bool)`, `AfterMove()`, `Succeeded()`,
  `Failed(recoverable: bool)`.
- Produces: `Beat(name: str, mood: str, points: list[tuple[float, np.ndarray, np.ndarray]],
  zone: Zone)`.
- Produces: `Director(capability: str, chain, scene, targets, preset="demo" | "subtle",
  lock_on="view" | "standoff", caps="default", seed=0)`.
- Produces: `Director.on_event(event, q_now: np.ndarray, held: bool) -> list[Beat]`. Every beat starts
  and ends at `q_now` and passes `check_beat`.
- Produces: `Director.spent_s: float`, `Director.log: list[str]`, and
  `BUDGETS = {"demo": 6.0, "subtle": 2.5}`.
- `rules.yaml` for `grab_container`, as (event, beat, keep priority):
  - `TaskRequested`: eager 5;
  - `AfterMove` in phase `view`: curious 4;
  - `BeforeMove` with `precise_end`: lock_on 7;
  - `BeforeMove` otherwise: wind_up 2;
  - `AfterMove` otherwise: follow_through 1;
  - `Succeeded`: proud 6, then attentive 1;
  - `Failed(True)`: search 3, sorry 6, shake_off 4.
- `rules.yaml` for `button_press` uses the same rows, with lock_on on the press's `BeforeMove`.
- Pricing: a beat costs its duration. It plays in full if that fits the remaining budget less a
  reserve. The reserve is the larger of (the short success reaction, the short failure reaction) plus
  the short versions of higher-priority rules still possible. Otherwise the beat plays short;
  otherwise it is dropped. `subtle` plays only rules with keep ≥ 6.
- A `Phase` the zone table maps to PRECISION, or any event while `held` and NEAR_USER, yields
  nothing spatial. `Failed(False)` yields nothing.

- [ ] **Step 1: Write the failing tests**

```python
def test_causal_prefix():          # decisions on events[:k] equal the first decisions on the full log, for every k
def test_budget_never_exceeded():   # 200 seeded random event logs per preset
    assert d.spent_s <= BUDGETS[preset] + 1e-9
def test_every_beat_starts_and_ends_at_q_now_and_validates():
    assert check_beat(b.points, q_now, ...) == []
def test_non_recoverable_failure_gets_nothing():
    assert d.on_event(Failed(False), q, False) == []
def test_unknown_phase_gets_nothing_and_logs():
    assert d.on_event(Phase("renamed", ""), q, False) == [] and "renamed" in d.log[-1]
def test_subtle_keeps_only_priority_six_and_up(): ...
def test_lock_on_fires_before_the_precise_approach():
    assert [b.name for b in d.on_event(BeforeMove(True), q, False)] == ["lock_on"]
```

- [ ] **Step 2: Run both test commands; expect FAIL.**
- [ ] **Step 3: Implement.**
  - Each beat composes its behaviors on `constant_track(q_now, grip, length)`, with the zone taken
    from the current phase (NEAR_USER by geometry), via `compose`, then `check_beat`.
  - A beat that fails `check_beat` is dropped and logged, never sent.
  - The full and short variants are the spike's (`spike/puppy_sim/director.py` `BEATS`, `RULES`,
    `Ledger`), minus stage and hold planning: on the robot the arm is stopped whenever an event
    arrives.
- [ ] **Step 4: Run both test commands; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_motion): online director with per-task budgets`.

## Part B — Simulator on the library

### Task 8: Move the simulator to `sim/` and drive it with `puppy_motion`

**Files:**
- Create the `sim/` package from `spike/`: move `scene.py`, `render.py`, `runs.py`, `envelope.py`
  and the `third_party/`, `data/` and `clips/` folders, plus `sim/requirements.txt`, which adds
  `-e ../puppy_motion`.
- Create: `sim/puppy_sim/replay.py`.
- Delete: `spike/` (git history keeps it).
- Test: port `spike/tests/test_scene.py`, `test_runs.py`, `test_fk.py` and `test_clips.py`, plus a
  new `sim/tests/test_replay.py`.

**Interfaces:**
- Consumes: the `puppy_motion` public API.
- Produces: `events_from_run(track, chain) -> list[tuple[float, Event]]`. Phases come from the
  recorded feedback. `BeforeMove` and `AfterMove` come from stops: every joint slower than 0.02 rad/s
  for 0.2 s, as in the spike's `director.stages`. `precise_end` comes from the zone table, and
  `Succeeded` / `Failed(True)` from the terminal status.
- Produces: `direct_run(run_name, preset="demo", lock_on="view", caps="default") -> (plain: Track,
  puppy: Track)`. It feeds events in order to `Director.on_event` with the recorded posture at that
  time, and splices each returned beat in as a time-warp hold of the beat's length at that time.
- With `lock_on="standoff"`, `events_from_run` splits the recorded approach where the tool is 0.10 m
  from the grasp pose. That is what grab_container would do if it split its call. The split adds
  `AfterMove()` and `BeforeMove(precise_end=True)` there, and `direct_run` adds
  `Slow(t_split, t_close, 0.75, ramp)` over the rest of the approach. The `subtitle` must keep saying
  that this option needs the capability change.
- Produces: `sim/puppy_sim/checks.py`, which calls `puppy_motion.validate.check_beat` per beat, plus
  the whole-track precision-identity and path checks.

- [ ] **Step 1: Write the failing tests**

```python
def test_replay_precision_identical():   # all three runs, both presets
    assert np.array_equal(puppy.q[p_mask], plain.q[p_idx])
def test_replay_within_budget():
    assert puppy.duration - plain.duration <= BUDGETS[preset] + DT
def test_failure_run_reacts():            # run 191809, demo
    assert {"sorry"} <= beat_names(puppy)
def test_every_spliced_beat_validates(): ...
```

- [ ] **Step 2: Run** `cd sim && .venv/Scripts/python -m pytest -q`; expect FAIL.
- [ ] **Step 3: Implement** `replay.py` and switch `scene.py` and `render.py` imports to
  `puppy_motion`. The MuJoCo model stays the renderer; kinematics for motion now come from
  `Chain.from_urdf(run["urdf"])`.
- [ ] **Step 4: Run the sim tests; expect PASS.**
- [ ] **Step 5: Commit** `refactor(sim): simulator drives motion through puppy_motion; spike retired`.

### Task 9: Gallery clips on the library

**Files:**
- Modify: `sim/puppy_sim/clips/*.py`, the 17 clips listed in the old spike README.
- Test: port `spike/tests/test_gallery_rest.py`, `test_gallery_task.py` and
  `test_gallery_reactions.py`.

**Interfaces:**
- Consumes: Task 8's `direct_run` and `puppy_motion.behaviors`.

- [ ] **Step 1: Run the ported gallery tests; expect FAIL on imports.**
- [ ] **Step 2: Rewrite each clip's `build()`** on `puppy_motion` (behaviors, `compose`, `warp`) or
  on `direct_run`. Titles, subtitles and cameras stay the same.
- [ ] **Step 3: Run** `python -m puppy_sim check all` (exit 0) and the gallery tests (PASS).
  Legibility thresholds from the ported tests: every key beat ≥ 25 px at 1280×720 in its camera.
- [ ] **Step 4: Run** `python -m puppy_sim render all`, then Read every `_sheet.png`. Compare each
  against the spike's committed sheet (`git show HEAD~N:spike/media/<clip>_sheet.png`) and note any
  clip that reads worse.
- [ ] **Step 5: Commit** `feat(sim): gallery clips rebuilt on puppy_motion`. Commit the MP4s and
  sheets only; GIFs stay ignored.

### Task 10: Impedance model in the simulator

**Files:**
- Modify or create: `sim/puppy_sim/dynamics.py`, ported from the spike's impedance agent work and
  `spike/puppy_sim/dynamics.py` if it exists on `main`.
- Create: `sim/tests/test_dynamics.py`.

**Interfaces:**
- Produces: `achieve(track: Track, mode: "position" | "impedance", gains="default") -> Track`. The
  result has the same timeline and the achieved `q`, using the driver's joint impedance law with the
  gains cited in the module docstring.
- Produces: `precision_deviation(plain, puppy, mode, gains) -> float`, the max |Δq| on PRECISION
  samples between the achieved plain and achieved puppy tracks.

- [ ] **Step 1: Write the failing tests**

```python
def test_holds_still_under_gravity():       # 2 s at home, impedance default
    assert np.abs(err).max() < 1e-3
def test_step_overshoot_matches_zeta(): ...  # within 20 % of the analytic second-order overshoot
def test_puppy_adds_no_precision_deviation():  # runs 132745 and 191809, impedance default, demo
    assert precision_deviation(plain, puppy, "impedance", "default") < 1e-3   # rad
```

- [ ] **Step 2: Run the sim tests; expect FAIL.**
- [ ] **Step 3: Implement or port** the dynamics. If `test_puppy_adds_no_precision_deviation` fails,
  the director must end beats earlier before a PRECISION move so the controller settles. Add a settle
  time per mode to `Director` (a `Director(settle_s=...)` constructor argument, default 0 for
  position) and record the value in the spec.
- [ ] **Step 4: Run the sim tests; expect PASS.**
- [ ] **Step 5: Commit** `feat(sim): driver joint-impedance model and the precision-deviation check`.

## Part C — `puppy_mode_ros` engine hook

The test environment is the x86 image from
`rammp-org/rammp-kitchen-capability-engines@c3989c6:docker/test.Dockerfile`, with `capability_engine`
from that same commit plus our patch.

- Workflow: `.github/workflows/puppy_mode_ros.yml` runs `docker build`, then `pytest
  puppy_mode_ros/test` inside the image, and runs on every push.
- Locally with Docker: `make -C puppy_mode_ros test`.
- Without Docker: push the branch, then `gh run watch`.

### Task 11: Package, test image and the `capability_engine` patch

**Files:**
- Create: `puppy_mode_ros/package.xml`, `puppy_mode_ros/setup.py`, `puppy_mode_ros/Makefile`,
  `puppy_mode_ros/docker/Dockerfile` (`FROM` the built test image; it copies `capability_engine` at
  `c3989c6`, applies the patch, and pip-installs `puppy_motion`), and
  `puppy_mode_ros/patches/capability_engine-puppy-hook.patch`.
- Create: `puppy_mode_ros/puppy_mode_ros/__init__.py`, `puppy_mode_ros/puppy_mode_ros/hook.py` and
  `.github/workflows/puppy_mode_ros.yml`.
- Test: `puppy_mode_ros/test/test_hook_off.py`.

**Interfaces:**
- The patch makes three changes to `capability_engine/engine.py`:
  - `__init__` declares `puppy.enabled` (False), `puppy.preset` ("demo") and `puppy.person_zone` ("").
  - `_on_execute` replaces `self.execute(goal.action_id, params, ctx)` with
    `self._run_execute(goal.action_id, params, ctx)`.
  - The new `_run_execute` returns `self.execute(...)` unless `puppy.enabled` is true, in which case it
    returns `puppy_mode_ros.hook.run(self, action_id, params, ctx)`. The import happens lazily inside
    that branch.
- Produces: `hook.run(engine, action_id, params, ctx) -> dict`. In this task it simply calls
  `engine.execute`; Task 13 fills it in.

- [ ] **Step 1: Write the failing test.** `test_hook_off.py` drives a minimal capability through
  `execute_action` beside a `FakeDriver`, following `capability_engine/test/test_engine_arm.py`. The
  capability makes two `go_to_joint_config` calls (one with `control_mode=IMPEDANCE, gains="soft"`)
  and one `execute_joint_trajectory`, and `puppy.enabled` is `False`. The test writes out the expected
  goals literally, from the arguments the capability passed:

```python
assert [n for n, _ in fake_goal_log] == ["go_to_joint_config", "go_to_joint_config", "execute_joint_trajectory"]
assert goal_fields(fake_goal_log) == expected_fields   # targets, control_mode, gains profile, points: exactly as passed
assert "puppy_mode_ros" not in sys.modules             # the hook is never imported while disabled
```

- [ ] **Step 2: Push and run the workflow; expect FAIL** (no package or patch yet).
- [ ] **Step 3: Implement** the patch, the package scaffold, the Dockerfile and the workflow.
- [ ] **Step 4: Push and run the workflow; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_mode_ros): package, test image and the off-by-default hook point`.

### Task 12: Proxies and the beat player

**Files:**
- Create: `puppy_mode_ros/puppy_mode_ros/proxies.py` and `puppy_mode_ros/puppy_mode_ros/player.py`.
- Test: `puppy_mode_ros/test/test_player.py`.

**Interfaces:**
- Consumes: `Director`, `Beat`, `check_beat` and `zone_of` from `puppy_motion`; the real
  `ArmInterface` and `Context`.
- Produces: `PuppyCtx(ctx, director_feed)`. `progress()` forwards first, then emits `Phase`. Every
  other attribute passes through `__getattr__`.
- Produces: `PuppyArm(arm, player)`. Move calls (`go_to_*`, `execute_joint_trajectory`,
  `move_joints`, `move_cartesian`) emit `BeforeMove` before forwarding unchanged and `AfterMove` after
  a normal return. They also record `control_mode` and `gains` from the call's kwargs; `move_*`
  `impedance=` maps to IMPEDANCE. `stream()` marks a stream open until its context exits, and no
  beats play while a stream is open. Every other attribute passes through `__getattr__`.
- Produces: `Player(arm, ctx, director, chain, scene).play(beats) -> None`. For each beat:
  - call `ctx.checkpoint()`;
  - skip the beat if any |measured joint velocity| ≥ 0.02 rad/s, or on `StateTimeout`;
  - re-validate with `check_beat` against the measured q;
  - call `arm.execute_joint_trajectory(chain.joint_names, beat.points, control_mode=…, gains=…,
    preemption=QUEUE)`. The mode and gains are the recorded ones, or IMPEDANCE with the recorded
    gains (else `"default"`) when `beat.zone is Zone.NEAR_USER`.
  - `Cancelled`, `EStopped` and `ControlTaken` propagate. `Refused` or any other `ArmError` sets
    `player.disabled = True` and returns.
  - Only arm joints are sent. A beat's grip offsets are dropped on the robot, and this is logged once
    per action.

- [ ] **Step 1: Write the failing tests** with `FakeDriver`:

```python
def test_beat_copies_the_tasks_control_mode():    # last move IMPEDANCE "soft" -> beat goal control_mode==1, gains profile soft
def test_near_user_beat_forces_impedance():        # last move POSITION, NEAR_USER beat -> control_mode==1, profile default
def test_moving_arm_skips_the_beat():              # fake joint velocities 0.05 -> no execute_joint_trajectory goal
def test_cancel_mid_beat_propagates_and_stops():   # hold_goals, cancel -> Cancelled raised, no later beat goals
def test_refused_beat_disables_and_task_continues():  # accept("execute_joint_trajectory", False) once
    assert player.disabled and next_task_goal_sent
def test_stale_state_skips(): ...                  # pause_publishing -> StateTimeout -> disabled, no goal
def test_no_beats_while_streaming(): ...
def test_proxies_forward_every_call_unchanged():   # goal fields identical with and without the proxy
```

- [ ] **Step 2: Push and run the workflow; expect FAIL.**
- [ ] **Step 3: Implement** `proxies.py` and `player.py`.
- [ ] **Step 4: Push and run the workflow; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_mode_ros): proxies and the validated beat player`.

### Task 13: `hook.run`, parameters and end-to-end behavior

**Files:**
- Modify: `puppy_mode_ros/puppy_mode_ros/hook.py`.
- Test: `puppy_mode_ros/test/test_hook_e2e.py`.
- Create: `puppy_mode_ros/README.md`. It covers enabling the hook, the `puppy.person_zone` format and
  why hardware runs are out of scope.

**Interfaces:**
- `hook.run(engine, action_id, params, ctx) -> dict` does the following:
  1. Parse `puppy.person_zone` with `Scene.from_param`. On a `ValueError`, log `"puppy mode off:
     <reason>"` and return `engine.execute(...)`.
  2. Build `Chain.from_urdf(engine.arm.robot_description())`. On failure, likewise run without Puppy
     Mode.
  3. Build a `Director(capability=engine.get_name(), chain, scene, Targets.from_scene(scene),
     preset=puppy.preset)`, then swap `engine.arm` for a `PuppyArm` and pass a `PuppyCtx`.
  4. Emit `TaskRequested`, then call `engine.execute`.
  5. On a normal return, emit `Succeeded`, play its beats (they stop at the first exception), and
     return the payload.
  6. On `EngineError(recoverable=True)`, emit `Failed(True)`, play the beats, then re-raise the
     original error. Any other exception propagates with no beats.
  7. Always restore `engine.arm` in a `finally`.

- [ ] **Step 1: Write the failing tests** (a scripted grab_container-like capability that reports the
  recorded phases):

```python
def test_success_plays_eager_lock_on_and_proud_within_budget():
    assert beat_goal_count >= 3 and sum(beat_durations) <= 6.0 + 1e-6
def test_recoverable_failure_reacts_then_reraises():   # result.status FAILED, error_code unchanged, sorry beat sent
def test_non_recoverable_failure_plays_nothing(): ...
def test_cancel_during_task_plays_nothing_after(): ...
def test_estop_plays_nothing():                        # engage_estop() mid-task
def test_empty_person_zone_runs_without_puppy_mode():  # goals == baseline, log has "puppy mode off"
def test_unknown_phase_plays_nothing_in_it():          # progress("renamed") -> no beat goals until the next known phase
def test_arm_attribute_restored(): ...
```

- [ ] **Step 2: Push and run the workflow; expect FAIL.**
- [ ] **Step 3: Implement** `hook.run`.
- [ ] **Step 4: Push and run the workflow; expect PASS**, plus both `puppy_motion` matrix jobs and
  the sim tests locally.
- [ ] **Step 5: Commit** `feat(puppy_mode_ros): end-to-end hook with reactions, fail-safes and docs`.
  Then open the PR `feature/puppy-mode-build → main`. Its body reports the test evidence per part
  (library matrix, simulator `check all`, hook workflow run link) and says plainly that nothing ran on
  hardware.
