# Puppy Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A never-shipped web "playground" that plays Puppy Mode expressions on flatbot's real arm on
demand, including an auto mode that keeps the arm alive whenever it is idle. Every motion is checked
against the chair's no-go boxes before it is sent.

**Architecture:**
- **`puppy_motion` (library additions):**
  - the chair scene loaded from flatbot's cuRobo layer files;
  - an idle director for auto mode;
  - the stage-pose and stage-path rules.
- **`puppy_playground` (one ament_python package), with two nodes:**
  - a `CapabilityEngine` subclass that holds the arm only inside its actions: stage, rest, express,
    combo, auto, recover and plan_stage;
  - a `playground_ui` node that serves a plain web page, talks to the engine through its action and
    control interfaces, and supervises auto mode, yielding to any task and resuming when idle.
- **The simulator** gives previews of the stage pose, the stage path and every expression, with the
  chair boxes loaded.

**Tech Stack:**
- Library: Python 3.10, numpy, pyyaml.
- Robot side: ROS 2 Humble rclpy, `capability_engine` (rammp-kitchen-capability-engines@c3989c6),
  and Python's `http.server`.
- Sim side: MuJoCo (`sim/`).
- Packaging: a Docker image `FROM rammp-base` (arm64, built on snorlax). Tests run in RAMMP's x86 test
  image in GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-06-puppy-playground-design.md`. It builds on
`docs/superpowers/specs/2026-11-05-puppy-mode-design.md`.

**Prerequisite (fast track, user asked for speed on 2026-11-06):** Tasks 1–6 of
`docs/superpowers/plans/2026-11-05-puppy-mode-build.md` are complete. That covers the URDF, track,
geometry, composer, behaviors and validator. The main plan's Tasks 7–13 wait until after this plan.

As a consequence:
- `Beat` is defined here, in `puppy_motion/puppy_motion/beat.py`, not taken from the main plan's
  Task 7.
- The previews use the existing throwaway `spike/` renderer and its `spike/puppy_sim/dynamics.py`
  (Task 4), not `sim/`.

Work continues on branch `feature/puppy-mode-build`.

## Global Constraints

**Scope and access**
- Never shipped: no PR to any `rammp-org` repo, nothing added to a `rammp-deployments` profile.
- No agent starts anything that can move the arm. Agents may use snorlax, telling the user first each
  time, only to:
  - build the image;
  - run it with `mock:=true`;
  - run the plan-only `plan_stage` action.

  Every motion step is a user checklist in `puppy_playground/BRINGUP.md`. The FakeDriver never runs on
  snorlax.

**The chair scene**
- It comes from rammp-deployments `feature/flatbot-empty-world@ec39dd2764d9c1eeabd1633e949c5b5fbb5b9a98`,
  `december_2026/config/robots/flatbot/curobo/`:
  - `layer_mebot_person.yaml` → `Scene.person`;
  - `layer_mebot_seat.yaml` + `layer_mebot_platform.yaml` → `Scene.obstacles`.
- Layer boxes are `{position: centre, dims: full size}`, axis-aligned in `arm_mount`.
- Vendor them under `puppy_motion/puppy_motion/data/flatbot_layers/` with a `SOURCE.txt` naming the
  commit.

**The motion pipeline (every playground motion)**
1. Measure `q` (`self.arm.joint_states()`).
2. Require every |dq| < 0.02 rad/s.
3. Behaviors at `q`, composed by `puppy_motion.compose.compose` with the chair scene and caps
   `"default"`.
   - Size s ∈ [0.25, 1.0] (page default 0.25) multiplies every contribution's `head` and `joints`
     before composing.
   - "Look at you" aims at `LOOK_TARGET = (1.2, -0.36, 0.9)` m in `arm_mount`: about 1 m in front of
     the seat front, on the chair's centreline, at standing head height *(call)*.
4. Points with central-difference velocities, first and last exactly 0.
5. `check_beat(points, q, chain, scene, zone, held=held) == []`.
6. `self.arm.execute_joint_trajectory(chain.joint_names, points, control_mode=IMPEDANCE,
   gains="default", preemption=QUEUE)`.
7. Never send gripper motion.

**Stage pose rules**
- Forearm, wrist and head clearance to the person boxes ≥ 0.35 m.
- Tool (head part) clearance to the seat and platform boxes ≥ 0.10 m.
- The eye's gaze (ee +z) is within 15° of horizontal.
- The face's up (ee +y) is within 20° of world up.
- Every joint is within the guards minus 10°.

**Stage path rules**
- Every part's person clearance stays ≥ its value at the path's first sample.
- Once the forearm, wrist and head each reach 0.35 m, they stay ≥ 0.35 m.
- Obstacle floor: min(first-sample clearance, 0.03) − 0.0005.
- The path is retimed (slowed only) until it meets the FREE caps:
  - joint speed `[0.36, 0.85, 0.59, 0.64, 0.85, 0.85, 0.85]` rad/s;
  - accel 3.0 rad/s²;
  - head speed 0.40 m/s.
- It starts and ends at rest with zero velocity.

**Recover**
- Within 0.10 rad on every joint of the stage pose: a minimum-jerk move to the stage pose, checked by
  `check_beat`.
- Within 0.02 rad of a stored-path sample: follow the path from there back to rest.
- Anywhere else: refuse.

**Auto mode, idle**
- No other engine's `~/status` reports `executing`.
- Every joint has been slower than 0.02 rad/s for 3 s.
- `/control_status.estopped` is false.

**Auto mode, held**
- `0.05 < /gripper_state.position < 0.93` means held.
- A gripper state older than 1 s also counts as held.

**Logging:** one JSON line per request in `$PLAYGROUND_LOG` (default `/data/playground.jsonl`), with
the action, params, peak speeds, clearances, check results, the goal sent and the driver result.

## Review Focus

1. **A task starts while a beat is mid-motion.** The UI supervisor sends the engine's Control CANCEL
   within one 0.1 s tick, then sends nothing until every engine has been idle for 3 s. Tested in
   Task 7.
2. **Two browser tabs, or double clicks, while a goal runs.** Requests are serialized: the UI never
   sends a second goal while one is active (it returns 409 busy), and the page shows "busy". Tested in
   Task 7.
3. **The stored stage path is missing, corrupt, or was planned for a different URDF.** `stage` and
   `rest` refuse with that reason. The file carries a hash of the URDF it was planned with. Tested in
   Task 5.
4. **E-stop mid-beat.** No retry. Status shows "e-stopped", and auto waits until the e-stop clears
   and the arm is idle. Tested in Task 6.
5. **The `/gripper_state` stream stops.** It is treated as held, so beats use the muted versions.
   Tested in Task 6.

---

## Part P1 — Library additions (`puppy_motion`)

### Task 1: Chair scene from flatbot's layers

**Files:**
- Create: `puppy_motion/puppy_motion/flatbot.py`.
- Create: `puppy_motion/puppy_motion/data/flatbot_layers/{layer_mebot_person,layer_mebot_seat,layer_mebot_platform}.yaml`
  and `SOURCE.txt`, copied byte-for-byte at the pinned commit.
- Test: `puppy_motion/tests/test_flatbot.py`.

**Interfaces:**
- Consumes: `geometry.Box`, `geometry.Scene`.
- Produces: `layer_boxes(yaml_text: str) -> list[Box]`, with `half = dims / 2` and
  `center = position`.
- Produces: `chair_scene() -> Scene`. Person = the person layer; obstacles = the seat + platform
  layers.
- Produces: `CHAIR_LAYERS_COMMIT = "ec39dd2764d9c1eeabd1633e949c5b5fbb5b9a98"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_person_layer_has_its_eight_boxes():
    assert len(chair_scene().person) == 8          # forearms, head, hips, legs, thighs, torso_0..2
def test_head_box_centre_and_half_size():
    head = by_name("head")
    assert np.allclose(head.center, [-0.01895, -0.36000, 0.87885]) and np.allclose(head.half, [0.23285, 0.1016, 0.21255])
def test_rest_pose_is_inside_the_torso_box_as_flatbot_documents():   # recorded-run home q
    assert clearances(chain, chair_scene(), HOME, held=False)[0].min() < 0.0
def test_obstacles_are_seat_plus_platform(): ...   # 5 seat + the platform's boxes
```

- [ ] **Step 2: Run** both `puppy_motion` test commands; expect FAIL.
- [ ] **Step 3: Implement** `flatbot.py` with `yaml.safe_load`. Load the files with
  `importlib.resources`, and add them to package-data.
- [ ] **Step 4: Run** both test commands; expect PASS.
- [ ] **Step 5: Commit** `feat(puppy_motion): flatbot chair scene from its pinned cuRobo layers`.

### Task 2: Stage-pose and stage-path rules

**Files:**
- Create: `puppy_motion/puppy_motion/stage.py`.
- Test: `puppy_motion/tests/test_stage.py`.

**Interfaces:**
- Consumes: `Chain`, `geometry.clearances`, `caps.CAP_PRESETS`, `track.DT`,
  `validate.Violation`.
- Produces: `stage_violations(chain, scene, q) -> list[Violation]`, checking the stage-pose rules.
  Rule names: `stage_person`, `stage_obstacle`, `stage_gaze`, `stage_up`, `stage_guard`.
- Produces: `find_stage_poses(chain, scene, n: int, seed: int, tries: int = 20000) -> list[np.ndarray]`.
  Random postures within the guards, refined by DLS toward forward and level, keeping those with no
  violations. Results are sorted by smallest head travel from rest.
- Produces: `retime_path(points, chain, caps="default") -> points`, which only ever slows: it uniformly
  stretches time until the FREE caps are met.
- Produces: `path_violations(points, chain, scene) -> list[Violation]`, checking the stage-path
  rules. Rule names: `path_person_monotone`, `path_person_floor`, `path_obstacle`, `path_caps`,
  `path_ends`.
- Produces: `min_jerk_path(q0, q1, seconds) -> points`, the hand-built fallback path.

- [ ] **Step 1: Write the failing tests**

```python
def test_found_stage_poses_meet_every_rule():
    poses = find_stage_poses(chain, chair_scene(), n=5, seed=0)
    assert poses and all(stage_violations(chain, chair_scene(), q) == [] for q in poses)
def test_rest_is_not_a_stage_pose():
    assert "stage_person" in rules(stage_violations(chain, chair_scene(), HOME))
def test_retime_only_slows_and_meets_the_caps():
    assert slow[-1][0] >= fast[-1][0] and caps_ok(slow)
def test_a_path_that_dips_back_toward_the_person_is_refused():   # rest -> out -> back toward torso -> stage
    assert "path_person_monotone" in rules(path_violations(...))
def test_a_min_jerk_path_from_rest_to_a_found_stage_pose_is_accepted_or_names_its_rule(): ...
```

- [ ] **Step 2: Run** both test commands; expect FAIL.
- [ ] **Step 3: Implement** `stage.py`.
- [ ] **Step 4: Run** both test commands; expect PASS.
- [ ] **Step 5: Commit** `feat(puppy_motion): stage-pose search and stage-path rules`.

### Task 3: Idle director for auto mode

**Files:**
- Create: `puppy_motion/puppy_motion/idle.py`.
- Test: `puppy_motion/tests/test_idle.py`.

**Interfaces:**
- Consumes: `behaviors`, `compose`, `check_beat`, `track.constant_track`.
- Produces: `puppy_motion/puppy_motion/beat.py` with `Beat(name: str, mood: str, points:
  list[tuple[float, np.ndarray, np.ndarray]], zone: Zone)` and
  `to_points(track: Track) -> list[tuple[float, np.ndarray, np.ndarray]]`. Velocities are central
  differences, first and last exactly 0. The main plan's Task 7 reuses `beat.py` later.
- Produces: `MOODS = ("calm", "curious", "playful", "sleepy")`.
- Produces: `IdleDirector(chain, scene, targets, size: float, seed: int)`. The playground passes
  `Targets(person_head=LOOK_TARGET)` (Global Constraints, pipeline step 3).
- Produces: `IdleDirector.next(q_now, held: bool, t_now: float) -> tuple[Beat | None, float]`. It
  returns a beat (or None when the check refused it, logged) and the pause in seconds before the next
  call.
- Produces: `IdleDirector.request(name: str, q_now, held, t_now) -> Beat | None` for an expression a
  button requested. `IdleDirector.mood` and `IdleDirector.log`.
- Moods and dwell:
  - calm: breathe, small glances; 20–60 s.
  - curious: gaze_to, head_tilt, sniff; 15–40 s.
  - playful: perk, wiggle, bounce, proud; 10–25 s.
  - sleepy: droop, then doze_and_wake; only after ≥ 180 s with no request.
- Calm is the hub. Pauses are 1–6 s. `held` → proud/gaze in their muted forms, no wiggle or bounce.
- The playground's expression names map to behaviors. Lead_gaze, follow_through, wind_up and clack are
  excluded.

- [ ] **Step 1: Write the failing tests**

```python
def test_same_seed_same_sequence():          # 50 calls, identical names and pauses
def test_moods_visit_all_four_over_a_long_run():   # t advanced by returned pauses; sleepy appears only after 180 s without request
def test_request_resets_the_sleepy_clock_and_plays_the_named_beat(): ...
def test_every_returned_beat_validates_at_rest_and_at_a_stage_pose():  # check_beat == [] with held False and True
def test_held_never_returns_wiggle_or_bounce(): ...
def test_at_rest_beats_are_small_and_never_closer_to_the_person():   # person clearance >= start - 0.005 everywhere
```

- [ ] **Step 2: Run** both test commands; expect FAIL.
- [ ] **Step 3: Implement** `idle.py`.
- [ ] **Step 4: Run** both test commands; expect PASS.
- [ ] **Step 5: Commit** `feat(puppy_motion): idle director for the playground's auto mode`.

## Part P2 — Simulator previews

### Task 4: Playground previews with the chair boxes

**Files** (fast track: in the existing `spike/`, using its renderer and `dynamics.py`, with
`puppy_motion` installed into `spike/.venv` editable):
- Create: `spike/puppy_sim/flatbot_scene.py`. It is the MuJoCo scene with the chair boxes drawn as
  translucent boxes in place of the mannequin, and an empty chair.
- Create: `spike/puppy_sim/clips/playground_stage.py`, `playground_expressions.py` and
  `playground_auto.py`. Each builds its joint tracks with `puppy_motion` and hands the `q` arrays to
  the spike's Track and renderer.
- Modify: `spike/puppy_sim/dynamics.py`. Add the positions-and-velocities trajectory variant (central
  dq, no accelerations), matching what `execute_joint_trajectory` sends.
- Test: `spike/tests/test_playground_clips.py`.

**Interfaces:**
- Consumes: `flatbot.chair_scene`, `stage.find_stage_poses`, `stage.min_jerk_path`,
  `stage.path_violations`, `idle.IdleDirector`, and the Task 10 driver model (positions + velocities
  variant).
- Produces three clips:
  - `playground_stage`: the chosen stage pose, plus the candidate rest→stage path.
  - `playground_expressions`: a reel of every expression at the stage pose at size 0.25 and 1.0.
  - `playground_auto`: 120 s of auto mode at the stage pose, seed 0.
- Produces `media/playground_stage_pose.png`, a still image of the stage pose for the user to approve.
- Produces `puppy_playground/puppy_playground/data/stage_pose.json`: `{"q": [7 floats], "urdf_sha256":
  ..., "rules": "stage.py@<commit>"}`. It holds the first pose `find_stage_poses` returns with seed 0;
  Task 5's `plan_stage` targets it.

- [ ] **Step 1: Write the failing tests**

```python
def test_clips_registered_and_pass_check(): ...
def test_stage_still_exists_and_pose_meets_the_rules(): ...
def test_driver_model_overshoot_with_positions_and_velocities_is_within_the_checks():
    # every expression at size 1.0, impedance "default": achieved head path stays within check_beat's floors
```

- [ ] **Step 2: Run** `cd spike && .venv/Scripts/python -m pytest -q tests/test_playground_clips.py`;
  expect FAIL.
- [ ] **Step 3: Implement**, render, and Read the sheets: the expressions must be visible at size 1.0.
- [ ] **Step 4: Run** the test again; expect PASS. Then run `python -m puppy_sim check
  playground_stage playground_expressions playground_auto`; it must exit 0.
- [ ] **Step 5: Commit** `feat(sim): playground previews with flatbot's chair boxes`. Commit the MP4s,
  the sheets and the stage still.

## Part P3 — `puppy_playground` package

All ROS tests run in the x86 test image, via the GitHub Actions workflow `ros_tests.yml` created in
Task 5. Push the branch, then `gh run watch`.

### Task 5: Package, test image, and the stage/rest/plan_stage/recover actions

**Files:**
- Create: `puppy_playground/package.xml`, `setup.py` and `setup.cfg`.
- Create: `puppy_playground/puppy_playground/__init__.py`, `engine.py` and `stage_store.py`.
- Create: `docker/ros-test.Dockerfile`. It builds RAMMP's `docker/test.Dockerfile` from
  rammp-kitchen-capability-engines@c3989c6, adds `capability_engine` from that commit, and
  pip-installs `puppy_motion`.
- Create: `.github/workflows/ros_tests.yml`. It runs `pytest puppy_playground/test` in that image.
  The main plan's Task 11 reuses it.
- Test: `puppy_playground/test/test_stage_actions.py`.

**Interfaces:**
- Produces: `class PuppyPlayground(CapabilityEngine)`, with `required_resources = ["arm"]` and
  `actions`: `plan_stage`, `stage`, `rest`, `recover`, `express`, `combo` and `auto`. `express`,
  `combo` and `auto` raise `EngineError("not_yet")` until Task 6.
- Produces: `stage_store.save(path, points, urdf_text, stage_q)` and
  `stage_store.load(path, urdf_text) -> StagePath`. `load` raises `StageStoreError` if the file is
  missing, unparseable, or its `urdf_sha256` differs.
- Parameters:
  - `playground.stage_path`: the approved file, default `/data/stage_path.json`;
  - `playground.candidate_path`: default `/data/stage_path.candidate.json`;
  - `playground.log`.
- `stage_q` is read from the package data file `stage_pose.json` (Task 4).
- `plan_stage`:
  1. calls `self.arm.plan_to_joints(stage_q)` (nothing moves);
  2. runs `retime_path`, then `path_violations`;
  3. on failure, tries `min_jerk_path`;
  4. writes the candidate and returns `{"violations": [...], "candidate": path}`. It never executes.
- `stage` and `rest` load the approved file and refuse unless the measured q is within 0.02 rad of the
  path's start (`stage`) or its end (`rest`). They re-check `path_violations` against `chair_scene()`
  and execute the points, reversed for `rest`.
- `recover` follows the Global Constraints rules.

- [ ] **Step 1: Write the failing tests** with `FakeDriver`, following
  `capability_engine/test/test_engine_arm.py`:

```python
def test_plan_stage_sends_no_motion_goal_and_writes_a_candidate():
    assert "execute_joint_trajectory" not in fake.goals and candidate.exists()
def test_stage_refuses_without_an_approved_file(): ...          # EngineError code "no_stage_path"
def test_stage_refuses_a_file_planned_for_another_urdf(): ...   # code "stale_stage_path"
def test_stage_refuses_when_not_at_the_paths_start(): ...       # fake.set_joints(far) -> "not_at_start"
def test_stage_and_rest_send_the_path_forward_and_reversed():   # goal points equal the stored ones, reversed for rest, IMPEDANCE default
def test_recover_near_stage_sends_a_checked_min_jerk_move(): ...
def test_recover_on_the_path_follows_it_back_to_rest(): ...
def test_recover_elsewhere_refuses(): ...                       # code "recover_by_hand"
```

- [ ] **Step 2: Push and run the workflow; expect FAIL.**
- [ ] **Step 3: Implement** the package, the store, the four actions, the Dockerfile and the workflow.
- [ ] **Step 4: Push and run the workflow; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_playground): engine with checked stage, rest, plan_stage and recover`.

### Task 6: express, combo and auto actions

**Files:**
- Modify: `puppy_playground/puppy_playground/engine.py`.
- Create: `puppy_playground/puppy_playground/pipeline.py` and `playlog.py`.
- Test: `puppy_playground/test/test_motion_actions.py`.

**Interfaces:**
- Produces: `pipeline.play(arm, ctx, chain, scene, beat_builder, held) -> dict` for one motion,
  running the Global Constraints pipeline. It returns the log record. On a failed check it sends
  nothing and raises `EngineError("check_failed", <rules>, recoverable=True)`.
- `express`: params `{"name": str, "size": float}`. It must be within 0.02 rad of the stage pose.
- `combo`: params `{"names": [str], "size": float}`, each in turn, with a `ctx.checkpoint()` between.
- `auto`: params `{"size": float, "seed": int}`. It loops `IdleDirector.next`, playing each beat and
  sleeping the pause in `ctx.checkpoint()`-sized steps. Each iteration it reads `~/request`
  (std_msgs/String, via `create_action_subscription`) and plays `IdleDirector.request(name)` at the
  next boundary. It runs until cancelled.
- Held: from `/gripper_state` via `create_action_subscription`, with the thresholds and 1 s staleness
  from the Global Constraints.
- `EStopped` and `ControlTaken` from a beat end the action, with result FAILED and code `estopped` or
  `control_taken`. No retry.

- [ ] **Step 1: Write the failing tests**

```python
def test_express_sends_one_impedance_default_goal_with_central_velocities(): ...
def test_express_refuses_away_from_stage_or_while_moving(): ...
def test_a_failed_check_sends_nothing_and_logs_the_rules(): ...
def test_combo_plays_in_order_with_checkpoints(): ...
def test_auto_plays_seeded_beats_until_cancelled_and_takes_requests(): ...
def test_gripper_stale_or_mid_range_counts_as_held(): ...
def test_estop_mid_beat_ends_auto_without_retry(): ...   # fake.engage_estop()
def test_no_gripper_goal_is_ever_sent(): ...
```

- [ ] **Step 2: Push and run the workflow; expect FAIL.**
- [ ] **Step 3: Implement** the actions, the pipeline and the log.
- [ ] **Step 4: Push and run the workflow; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_playground): express, combo and auto through the checked pipeline`.

### Task 7: Web page, UI node and auto supervisor

The page is a temporary testing tool (user decision). Keep it minimal: plain unstyled buttons, one
HTML file, no build step, no design polish, and no work toward making it a product UI. The supervisor
and the HTTP endpoints still get full tests, because they decide when the arm moves.

**Files:**
- Create: `puppy_playground/puppy_playground/ui_node.py`, `supervisor.py` and `web/index.html`
  (plain HTML, CSS and JavaScript, no build step).
- Create: `puppy_playground/launch/playground.launch.py`, which starts both nodes with
  `use_orchestrator:=false`.
- Test: `puppy_playground/test/test_supervisor.py` (pure Python) and
  `puppy_playground/test/test_ui_http.py`.

**Interfaces:**
- Produces: `Supervisor(clock)` as a pure state machine. Its states are `OFF`, `PLAYING`,
  `WAITING_FOR_TASK`, `WAITING_IDLE` and `ESTOPPED`.
  - Inputs: `toggle(on)`, `engine_status(engine_id, state)`, `arm_still_for(seconds)`,
    `estopped(bool)`, `goal_done()`, `tick(now)`.
  - Outputs: a list of commands, `SendAuto(size, seed)` / `Cancel()` / `None`.
  - When any other engine reports `executing`, it emits `Cancel()` and goes to `WAITING_FOR_TASK`.
  - When every engine is idle and the arm has been still for 3 s, it emits `SendAuto` and goes to
    `PLAYING`.
- Produces: `ui_node.main()`.
  - An rclpy node with an ActionClient for `/puppy_playground/execute_action` and a client for
    `/puppy_playground/control`.
  - It subscribes to every `/<id>/status` topic it discovers, plus `/joint_states` and
    `/control_status`.
  - It runs the supervisor at a 0.1 s tick.
  - It serves port 8080 (`playground.ui_port`) with `http.server`.
- HTTP endpoints:
  - `GET /`;
  - `GET /status` (JSON: state, mood, last refusal, auto state);
  - `POST /stage`, `/rest`, `/recover`, `/express` `{name, size}`, `/combo` `{names, size}`,
    `/auto` `{on, size}`, `/stop`.
  - A POST while a non-auto goal is active returns 409 `busy`. `/express` while auto is playing
    publishes `/puppy_playground/request` instead of sending a goal.
- The page:
  - buttons for the expressions listed in the spec;
  - a combo builder, a size slider (default 0.25), Stage, Rest, Recover, the Auto toggle and a large
    Stop;
  - a status line;
  - links to the preview clips;
  - the sentence "The hardware e-stop is the safety; Stop only cancels the current motion."

- [ ] **Step 1: Write the failing tests**

```python
def test_task_start_cancels_within_one_tick_and_waits():      # supervisor: PLAYING + engine_status(x, executing) -> [Cancel()], WAITING_FOR_TASK
def test_resumes_only_after_all_idle_and_three_still_seconds(): ...
def test_estop_holds_until_cleared_and_idle(): ...
def test_double_post_returns_409_busy(): ...                  # test_ui_http with a fake action client
def test_express_during_auto_publishes_a_request_not_a_goal(): ...
def test_page_contains_the_e_stop_sentence_and_every_expression_button(): ...
```

- [ ] **Step 2: Push and run the workflow; expect FAIL.**
- [ ] **Step 3: Implement** the UI node, the supervisor, the page and the launch file.
- [ ] **Step 4: Push and run the workflow; expect PASS.**
- [ ] **Step 5: Commit** `feat(puppy_playground): web page, UI node and auto-mode supervisor`.

## Part P4 — Image, mock bring-up and the user's checklist

### Task 8: arm64 image, mock run on snorlax, and BRINGUP.md

**Files:**
- Create: `puppy_playground/Dockerfile` (`FROM rammp-base`, the same base and build pattern as
  rammp-kitchen-capability-engines' capability Dockerfiles; it installs `puppy_motion` and the
  package).
- Create: `puppy_playground/BRINGUP.md`.
- Create: `puppy_playground/scripts/mock_smoke.sh`. It runs the container with `mock:=true` and
  curls every endpoint.

**BRINGUP.md is the user's checklist:**
1. Mock run.
2. With the user's profile switch, `sheppy up flatbot_empty_world`, run the plan-only `plan_stage`.
   Review the candidate in the simulator (`playground_stage` clip, rendered from the candidate file),
   then copy it to `stage_path.json` to approve it.
3. Back on `sheppy up flatbot`: Stage at size 0.25 with a hand on the e-stop. Each expression once.
4. Full size.
5. Auto mode, then a task while auto is on.
6. Rest.

The checklist states where to stand: the chair's right side, by the joystick.

- [ ] **Step 1: Tell the user** that the agent is about to build the image and run mock mode on
  snorlax. Build on snorlax (`docker build` in a checkout under `~/claude-scratch/puppy-playground`),
  with `nice` and nothing touching the sheppy containers.
- [ ] **Step 2: Run** `scripts/mock_smoke.sh` on snorlax with `mock:=true` under the running profile.
  Expected:
  - every endpoint returns 200, or a reasoned refusal;
  - the log has one JSON line per request;
  - the driver receives no goal (mock).
- [ ] **Step 3: Verify the driver-acceptance assumption without motion.** In mock mode, confirm
  `execute_joint_trajectory` goals are built and logged. Record that the real-driver acceptance
  under the `flatbot` profile is BRINGUP step 3's first action at size 0.25.
- [ ] **Step 4: Write BRINGUP.md** with exact commands for every step above.
- [ ] **Step 5: Commit** `feat(puppy_playground): arm64 image, mock smoke test and the bring-up
  checklist`. Then report to the user with the stage still and the checklist link. Every remaining
  step is theirs.
