# Puppy Mode: overnight build summary (2026-11-06 to 2026-11-07)

Branch `feature/puppy-mode-build`, head `7d0d704`. Nothing moved the arm overnight: every robot-side step was a build, an isolated mock run (ROS domain 77, nothing sent) or a read-only joint reading.

## Ready on snorlax

| What | Image | Check |
|---|---|---|
| Puppy Playground | `puppy-playground` (79bb114) | mock run on domain 77: all checks pass, 5 trajectories reached the mock arm, none sent |
| box_opening with Puppy Mode | `box_opening:v2-105f74e-puppy` (7d0d704) | mock run: Puppy Mode on, the goal succeeds with exactly one proud after the run and nothing before or during it; off, no beats |

CI is green on every workflow: puppy_motion (py3.10/numpy 1.26.4 and py3.12/numpy>=2), puppy-mode-ros, ros-tests. The spike suite passes 445 tests.

## Tomorrow

### 1. Playground from your new home pose (`puppy_playground/BRINGUP.md`)

1. Park the arm at the new home. Nobody in the chair.
2. Check it, read-only: "Choosing a home pose". `check_home.py` prints the start-clearance gate and whether an offline planner finds a stage path. It also says when cuRobo, which only runs live, is still worth trying.
3. If the gate passes but no planner does, two things can help:
   - move the home a little; keep the upper arm well clear of the occupant box;
   - or turn on the opt-in floor cap. That is your call, attended, with nobody in the chair.
4. Plan (step 4, or `check_home.py --write-candidate`), render, approve (step 5), then Stage at size 0.25 (step 6).

Measured last night: from poses near Kinova Home, the strict rule ("no part may get closer to the occupant than it started") blocks every route. The upper arm starts about 44 mm from the occupant box, and every route passes it at about 28 mm. With the floor capped at 20 mm, the straight route passes from your near-Home pose.

### 2. box_opening with Puppy Mode (`puppy_mode_ros/BRINGUP.md`)

1. Stop sheppy's box_opening node.
2. Run the `-puppy` image with the launcher command. Start with `puppy.preset:=subtle`.
3. Expect one small "proud" beat after a successful open, at the HOME pose, and nothing before or during the run.
4. Turn it off with `puppy.enabled:=false`, or use the stock image.
5. Restore sheppy.

box_opening runs its motion in a subprocess, so Puppy Mode can only react after the run ends. If the gripper isn't fully open at HOME, the beat is skipped, and the log says why.

## Decisions for you

- **Floor cap:** `stage_floor_cap_m`, off by default. With it on, stage paths may bring parts within 20 mm of the occupant boxes. It is for attended tests with nobody in the chair. It is read-only at run time, shown on the page, and recorded in the approved file.
- **Last night's approved stage path** (`~/puppy-data/stage_path.json`, sha 470461b178e3) starts inside the occupant box, because the start gate was skipped at your call. The rebuilt image refuses it by default. To use it again, start the engine with `accept_skipped_start_gate:=true`.
- **PR:** the plan ends by opening `feature/puppy-mode-build -> main`. I have not opened it; tell me when you want it.

## Follow-ups (not done)

- **Accelerations:** RAMMP's `execute_joint_trajectory` (capability_engine c3989c6) takes `(t, q, qd)` points only. The impedance model shows wag and bounce overshooting 13-14 mm at default gains without accelerations, against 1-4 mm with them. Adding accelerations needs a small, separate capability_engine PR.
- **The `via` planner:** slow constant-speed legs (about 47 s), and it returns the first prescreened waypoint.
- **Simulator:** the `spike/` to `sim/` rename and retiring the old spike director were deferred so tomorrow's render commands stay put.
- **Parked minors:** in each task's review report.

## Every ruling made while building (binding decisions, with the reason and the cost if wrong)

### Puppy Mode build plan

- Ruling: T8 COPIES scene.py, render.py, runs.py, envelope.py, dynamics.py, data/ and third_party/ into sim/ and leaves spike/ untouched; T9 ports every clip into sim/ and deletes spike/ at its end; T10 adapts sim/puppy_sim/dynamics.py (copied in T8) — avoids T10 porting from a deleted file and keeps every clip working between T8 and T9 — cost if wrong: spike/ and sim/ coexist for one task (duplication for a short window).
- Ruling: T8's ported tests are test_scene/test_runs/test_fk plus test_replay; test_clips moves to T9 — clips only exist on the library after T9 — cost if wrong: replay clips untested in sim/ for one task.
- Ruling: T9 ports the 17 gallery clips; T10 also ports the 4 impedance clips (impedance_wag, impedance_lock_on, impedance_profiles, impedance_friction) and the `compliance` command onto sim/ — they were added after the plan and belong with the dynamics — cost if wrong: T10 grows by 5 small files.
- Ruling: implementers may push feature/puppy-mode-build (never main) to origin so the hook workflow runs in GitHub Actions — the user asked to use this repo and there is no local Docker — cost if wrong: work-in-progress commits are visible on the public repo's feature branch.
- Ruling (from the playground design, 2026-11-06): Task 10 also models trajectories carrying positions and velocities only (what execute_joint_trajectory sends, central-difference dq), not just pos+vel+acc — the playground and the hook send exactly that — cost if wrong: one more driver-model variant to test
- Ruling (scope): Part C (Tasks 11-13) + Task 7 are what an arm test needs; Part B (Tasks 8-10: sim move, gallery on the library, impedance pos+vel variant) is parked until after the hardware test — sim-only, no effect on the robot — cost if wrong: the sim keeps using the spike code one more day
- Ruling (hook point, amends Task 11): no capability_engine source patch. puppy_mode_ros.hook.install(engine_cls) wraps the capability subclass's execute() at class level (engine_cls.execute = wrapper calling hook.run(self, action_id, params, ctx, original)) when the engine's puppy.enabled param is true; a launcher (python -m puppy_mode_ros.launch <module>:<Class> <module>:main) installs it and runs the engine's own main. The puppy.* params are declared by install() (declare if undeclared). RAMMP's capability_engine is untouched (repo rule: capability_engine changes are separate PRs), and it works with both KCE c3989c6 (tests) and box_opening's microwave:dev copy — cost if wrong: one extra indirection vs a patch
- Ruling (box_opening): box_opening v2 runs its motion in a subprocess (node.py Run(...)), not through self.arm. The hook supports it with boundary-only beats: TaskRequested → eager, Succeeded → proud, Failed(True) → sorry (+shake_off); every box_opening phase (locate, look, press, grip, carry, release, home) maps to PRECISION (no spatial beat while the subprocess may move); rules.yaml and zones.yaml gain a box_opening entry. Beats play only before execute() and after it returns, never while the run's subprocess exists — cost if wrong: less expression during box opening (by design)
- Ruling (scene on the robot): puppy.person_zone accepts "flatbot" = puppy_motion.flatbot.chair_scene() (the PR #150 chair layers) in addition to the Scene.from_param JSON; default preset on the robot "subtle" for the first test — cost if wrong: none
- Ruling (Task 14, new): robot packaging for the hook: a derived image FROM box_opening:v2-105f74e adding puppy_motion + puppy_mode_ros, run by the launcher; a mock run on snorlax on an isolated ROS domain (box_opening mock + capability_engine mock arm, nothing sent); puppy_mode_ros/BRINGUP.md for the attended test — cost if wrong: none (no RAMMP repo or manifest changes)
- Ruling: Task 12 dispatched in parallel with Task 7 and Task 11's review — disjoint files (proxies.py, player.py, test_player.py); it depends on a director stub, beat.py's Beat and Task 11's package — cost if wrong: a fix round if Task 11's review changes hook.py's interface
- Ruling (Task 12): player requires measured q within 0.002 rad (wrap-aware) of the beat start, align_turns, check_beat on the exact sent points; False return = Refused (disable); PREEMPTED → ControlTaken; tolerance violated → disable + continue; HALTED → EStopped/ControlTaken; no recorded move → IMPEDANCE "default" (box_opening) — cost if wrong: more skipped beats
- Ruling: short proud <= 2.0 s; every non-ending beat keeps the ending reserve max(short success, short failure) + short versions of higher-keep rules; keep<=1 rules never count — the success reaction must always remain possible — cost if wrong: fewer mid-task beats at demo
- Ruling: any failed task move disables the player for the rest of the action, no AfterMove, no failure reaction — a beat must never start from a contact — cost if wrong: fewer failure reactions
- Ruling: per-beat code never raises into the task (only Cancelled/EStopped/ControlTaken propagate); checkpoint + fresh reading (still, 0.002 rad, finite, <=0.1 s old) right before send; warm the action client; bounded settle wait (<=1.0 s, 3 still reads) before composing AfterMove beats and after each beat; control mode by the stricter of beat.zone and geometry
- Ruling (Task 7 addendum): Failed(True) yields nothing spatial in a PRECISION phase
- Ruling (Task 13): box_opening's "home" phase maps FREE (not PRECISION): box_opening has no Phase/Move rules, so nothing plays during the run either way, and a PRECISION last phase would suppress the Succeeded proud after the run has ended at home — cost if wrong: none during the run; proud plays at the end pose (checked by check_beat with the chair scene)
- Ruling (Task 8): keep spike/ (no move to sim/, no delete) — BRINGUP step 5 renders candidates from spike/ tomorrow; implement replay.py/checks.py inside spike/puppy_sim — cost if wrong: the rename happens later
- Ruling: REVERSES "Ruling (Task 13): box_opening's home phase maps FREE" — home stays PRECISION — cost if wrong: none
- Ruling: TaskRequested beats are deferred to the task's first BeforeMove (so a goal the task refuses before moving moves nothing); a task that never moves through self.arm (box_opening) gets no pre-task beat — cost if wrong: box_opening shows only the proud after success
- Ruling: ending beats never change the task's result — Cancelled/EStopped/ControlTaken during proud/sorry are caught and logged; payload returned / original error re-raised unchanged — cost if wrong: a cancel during the proud reports SUCCEEDED (the task did succeed)
- Ruling: drop box_opening's attentive row — one proud after a success under every preset, as documented — cost if wrong: none
- Ruling (Tasks 9, 10): stay in spike/; run in parallel (disjoint: gallery clips vs dynamics.py + director settle + one guarded PuppyArm call); Task 10 adds the pos+vel-only driver variant (2026-11-06 ruling); set_control in PuppyArm guarded so it can never raise into a task's move — cost if wrong: one more image rebuild before hardware
- Ruling: settle tail only on mid-task beats; endings keep TAIL_S; gains None under impedance = session default (medium, 1.1 s); custom/unknown = 2.3 s — cost if wrong: none for unheld box_opening
- Ruling: the settle tail also applies to BeforeMove(precise_end=True) beats (lock_on precedes a PRECISION move); not to TaskRequested or plain BeforeMove — cost if wrong: lock_on dropped under held+soft (safety over expressiveness). Note: the Player's measured settle wait after every beat (Task 12) already keeps control from returning until the arm is still
- Ruling (C1): Succeeded beats play only if the task made a move call through PuppyArm or the payload reports executed is True — cost if wrong: none (box_opening real runs report executed True)
- Ruling (I1): candidates record start_gate ("passed" | "skipped"); stage_store.load re-checks the start gate on the file's first sample; a file that fails it, or records "skipped", is refused unless the engine's read-only param accept_skipped_start_gate is true (default false); the render title says START GATE SKIPPED. Not moving the user's approved file on snorlax: the rebuilt image refuses it by default — cost if wrong: the user must opt in to repeat tonight's override
- Ruling (I3): the sim never invents phases; failure clips show what the robot would do — cost if wrong: fewer sorry beats in the gallery

### Puppy Playground plan and follow-ups

- Ruling: fast track (user asked for speed): Tasks 1+2 dispatched as one; independent tasks may run in parallel when they touch disjoint files; reviews on sonnet except Task 6 (motion pipeline, opus); minors deferred, one re-review per fix round — cost if wrong: less review depth on non-motion code
- Ruling: the stage path leaves rest by an escape leg — puppy_motion/escape.py `escape_path(chain, scene, q_rest, q_stage) -> points`: climb the person-clearance gradient (joint-space steps, obstacle floors kept, never decreasing any part's person clearance) until forearm/wrist/head are >= 0.35 m, then min-jerk to the stage pose, retimed; checked by path_violations — straight min-jerk from rest dips 8-9 cm deeper into the torso box (rest starts inside it) — owned by Task 5 (new file, so Task 2 review fixes can't collide) — cost if wrong: a slightly longer first move out of rest
- Ruling: Task 5 writes puppy_playground/puppy_playground/data/stage_pose.json itself (first find_stage_poses seed 0 pose) so Task 5 runs in parallel with Tasks 3-4; Task 4 renders it — cost if wrong: none
- Ruling: the runtime re-check of the stored stage path is done once per engine start and cached by (path sha256, chair scene commit, urdf sha256) — a full check costs 5-19 s per press — cost if wrong: a scene change inside one process lifetime goes unchecked (the scene is vendored, so it cannot change)
- Ruling: Task 6 plays beats from a precompiled cache keyed by (pose: stage|rest, name, size, held, seed), built at engine start (and re-checked with check_beat at load); at any other pose beats are composed live in a background thread ahead of when they play, with the page showing "composing" — a button press must not wait 10-60 s — cost if wrong: a few seconds of startup and memory for the cache
- Ruling: add a perf task now (parallel): vectorize geometry.clearances and compose's clearance gradients, outputs equal to the current implementation within 1e-9 on random postures (tested old-vs-new) — every beat and the stage-path check pay this cost — cost if wrong: a numerically different but equivalent clearance (guarded by the equality test)
- Ruling: stage_pose.json = the first find_stage_poses(seed 0, n<=10) pose for which escape_path succeeds and passes path_violations (+0.002 m margin), with "search_index" — written by Task 5's generator; sent to Task 5's implementer — cost if wrong: the stage pose is not the "nearest" one
- Ruling: new puppy_motion/joints.py (shortest_delta, align_turns) shared by Tasks 5 and 6; every pose comparison wrap-aware; every sent path aligned to the measured q — cost if wrong: none
- Ruling: plan_stage tries cuRobo → straight min-jerk → escape, writes the first passing (records "method") — a straight 11.6 s path passes for the shipped pose vs the 57.4 s escape — cost if wrong: none (all candidates pass the same rules)
- Ruling: the user approves the EXACT candidate: the playground_stage clip gains a candidate-file mode (PLAYGROUND_CANDIDATE) showing "REVIEW FOR APPROVAL" or "NOT APPROVABLE"; BRINGUP points at it — dispatched (sonnet), with Task 4's two carried review items
- Ruling: auto is anchored to named poses: each auto beat's final measured q within 0.02 rad (wrap-aware) of stage or approved q_rest, else non-recoverable drifted - impedance sag re-bases each beat so unanchored auto creeps - cost if wrong: auto stops more often and needs a Stage press
- Ruling: PREEMPTED -> control_taken, PATH/GOAL_TOLERANCE_VIOLATED -> contact, both non-recoverable - in impedance a tolerance abort usually means contact - cost if wrong: auto stops on spurious aborts
- Ruling: the supervisor turns auto OFF on stop codes (drifted, no_anchor, contact, control_taken, estopped) and shows the reason - otherwise it restarts auto into the same fault - cost if wrong: the user re-toggles auto
- Ruling: a lazy engine-scoped BeatSource keyed with URDF sha + stage-pose bytes supersedes 'built at engine start' - composition <= 1.5 s after the perf work - cost if wrong: first press of each expression waits up to 1.5 s
- Ruling: sizes outside [0.25, 1] and bad seeds refuse (bad_params) instead of clamping - conservative - cost if wrong: a caller sending 0.1 gets a refusal
- Ruling: the mock smoke runs its own container on ROS_DOMAIN_ID=77 + ROS_LOCALHOST_ONLY=1 on a dedicated port - isolation makes motion structurally impossible even if mock were wrong - cost if wrong: none
- Ruling: a plan_only launch arg (8a) + engine gate (Task 5 fixer, who owns engine.py) refusing every action but plan_stage - the plan-stage run happens on the live domain - cost if wrong: one small engine edit
- Ruling: the margin is pinned to MARGIN_M (files with any other margin refused) - path_violations is not monotone in the margin - cost if wrong: none
- Ruling: slow reading-independent work, then checkpoint, then measure; recover re-reads q before send (moved if > 1e-3 rad); the residual window inside arm's own checkpoint is parked as a capability_engine follow-up - cost if wrong: a microsecond window remains
- Ruling: I2 — blend the stored path onto the reading with a min-jerk weight over 1.0 s (end exact, so no accumulation), check the sent points at MARGIN_M, re-read before send — a constant shift would accumulate across Stage/Rest cycles — cost if wrong: ~1.3 s re-check per press
- Ruling: I3+M9 — any e-stop, and any auto goal ending by itself (not our cancel), turns auto OFF with the reason; supersedes the spec's resume-after-e-stop — the user said auto runs until they turn it off; resuming by itself after an e-stop is not what a person expects — cost if wrong: the user re-toggles auto
- Ruling: M8 — CI must test the installed package (pinned pip, conftest no longer shadows it) — cost if wrong: none
- Ruling: parked: M2 recover-along-path retime (BRINGUP note only), M4 0.0.0.0 bind (documented, attended), M5 preview link, M12 early request dropped
- Ruling: N2 — the blended Stage/Rest check anchors the person floor at min(clearance at the reading, clearance at the stored path's start), per part; all other rules unchanged — the blend otherwise refuses on readings a few mrad off; this never lets the path come closer than either the arm's actual pose or the approved start — cost if wrong: up to the reading-vs-start clearance difference (mm)
- Ruling: N5 — turning auto on while e-stopped is refused (stays OFF, reason estopped); N4 — a CANCELLED result of the supervisor's own cancel keeps resume-after-task. Parked: N7 no mood published (page shows the action)
- Ruling: park — Stage after later Rest/Stage cycles can refuse path_obstacle (upper_arm) for j1 +0.005..+0.015 rad offsets (obstacle floor anchored at the reading); the first Stage starts at plan_stage's own measured start (offset ~0). Follow-up: anchor the obstacle floor like the person floor — cost if wrong: a refusal (nothing moves), user presses Recover or re-plans
- Ruling: start gate on the grown scene — person >= +10 mm forearm/wrist/head, >= -10 mm upper arm; obstacles >= 10 mm all parts (reviewer's numbers; passes REST_Q, its ±0.02 window, the park and Kinova Home; refuses all 9 bad starts) — cost if wrong: some parks refuse bad_start and need a small attended nudge
- Ruling: build an OPT-IN attended floor cap (stage_floor_cap_m, only 0.0 or 0.020, default off; recorded in the file; load refuses a mismatch; render and check_home use the same rule) — the user decides tomorrow; off = today's rules bit-for-bit — cost if wrong: a dormant option
- Ruling: stage_floor_cap_m and plan_only are read-only parameters; the page and /status show the cap — cost if wrong: none

