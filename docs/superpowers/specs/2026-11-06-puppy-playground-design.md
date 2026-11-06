# Puppy Playground: Puppy Mode on the real arm, on demand (design)

Date: 2026-11-06. Status: open questions answered, waiting for approval. Builds on the Puppy Mode design
(`2026-11-05-puppy-mode-design.md`) and its `puppy_motion` library.

## Intent

The user loves how expressive Puppy Mode looks in simulation and wants to try it on the real arm as a
fun feature that will never ship. There is no task: the user stands at the e-stop and presses buttons
on a web page to play each expression, or a combination of them. An auto mode lets the arm pick its own
expressions, the way a dog behaves when nobody is asking anything of it.

## Goals

1. Play any single expression, or a chosen sequence, on the real arm with one press.
2. **Auto mode:** the arm runs on its own until stopped, drifting through moods and picking behaviors,
   so it feels alive rather than looping.
3. Every motion meets the same safety rules as Puppy Mode in simulation, checked before it is sent.
4. A safe bring-up path: no motion, then a simulator preview, then a quarter-size real arm, then full
   size.

## Non-goals

- Shipping: not in any demo profile, not in `rammp-deployments`, no PR to `rammp-org`.
- Orchestrator integration. The playground runs standalone.
- Running alongside a task. The capability hook is the main Puppy Mode plan's Part C.
- Gripper chomps on the real arm *(call)*: they add a pinch hazard, so they are left out for now.
- Perception, sound and a person in the chair. The chair is empty during tests.

## Anti-goals

- Any motion that skips the checks, including the moves to and from the stage pose.
- Motion toward the occupant volume, or faster than the arm's own measured envelope.
- A Stop button presented as safety. The hardware e-stop is the safety; Stop only cancels the
  current motion.
- Running unattended.

## Constraints

- **The robot.** Snorlax's arm is *flatbot*: a Gen3 on the MEBot wheelchair, on the chair's left,
  tilted 30° forward (`rammp-deployments/december_2026/config/robots/flatbot/mount.yaml`).
- **The no-go zone (user decision).** It is flatbot's planner layers in `rammp-deployments`, in
  `config/robots/flatbot/curobo/`, all as axis-aligned boxes in `arm_mount`:
  - the **person** is `layer_mebot_person.yaml`: occupant boxes sized to fit from the 5th-percentile
    woman to the 95th-percentile man, plus a margin;
  - **obstacles** are `layer_mebot_seat.yaml` (seat base, backrest, armrest) and
    `layer_mebot_platform.yaml`.

  They are best guesses, not measured on the chair. Read them at the pinned commit of the profile in
  use, never hand-copied.
- **The rest pose is inside the zone.** flatbot's rest pose sits 25 mm inside the occupant's torso box
  and 5 mm inside the backrest (`planner.yaml`'s own note). The chair layers therefore refuse plans
  that start from rest.
- **The planner profile.** Moving out of rest needs `flatbot_empty_world`, whose planner has no
  collision volumes. That profile is rammp-deployments PR #150, still open on 2026-11-06. It is for
  attended use only.
- **RAMMP rules.**
  - `self.arm` works only inside `execute()`.
  - Never call `/acquire_control`, `/release_control` or `/revoke_control`, and never publish
    `/estop`.
  - Every hardware run is attended, with a human on the e-stop.
- **Puppy Mode's rules, unchanged.**
  - Zone gains: FREE 1, NEAR_USER 0.35, PRECISION 0.
  - Caps: FREE joint speed `[0.36, 0.85, 0.59, 0.64, 0.85, 0.85, 0.85]` rad/s, 3.0 rad/s², head speed
    0.40 m/s. NEAR_USER 0.5 rad/s, 1.5 rad/s², 0.16 m/s.
  - URDF velocity limits and the joint guards.
  - Person floor: min(start clearance, 0.35 m) − 0.005 m. Obstacle floor: min(start clearance, 0.03 m).
  - Kinematics come from `/arm/robot_description`.
- **The operator (user decision: "doesn't matter").** The operator's position is not modelled. The
  page and the bring-up checklist tell them to stand outside the arm's reach: the stage poses put the
  tool out to the arm's left, 0.8–1.0 m forward of its base, so the chair's right side, by the
  joystick, is the natural spot. "Look at you" keeps its fixed target.
- **Snorlax access (user decision).** Agents may use snorlax, telling the user first each time, for
  three things only, none of which can move the arm:
  - building the playground image;
  - running it in mock mode;
  - the plan-only `plan_stage` run.

  Every step that can move the arm is a checklist the user runs, with a hand on the e-stop.
- **Dependency.** It needs `puppy_motion` (the Puppy Mode plan's Part A, Tasks 1–7) for behaviors,
  the composer, `check_beat` and the geometry. It also needs the plan's Part B, the simulator on that
  library, for two things:
  - previews on flatbot's mount with the chair boxes;
  - the driver-model check of trajectories that carry positions and velocities only, which Task 10
    is extended to cover.

  So it is built after Part B and before Part C *(call)*.

## Decisions

The user decided:

- the playground exists, with buttons, combos and auto mode;
- the chair is empty during tests and the user stands at the e-stop;
- the no-go zone is the chair's planner layers.

Everything marked *(call)* is the lead's and can be changed.

### Where it runs *(call)*

The playground lives in this repo as one ament_python package, `puppy_playground`, with a Dockerfile
`FROM rammp-base`, the same base as RAMMP's capability images. It is built on snorlax (arm64) and run
with `docker run` next to `sheppy up flatbot_empty_world`, standalone (`use_orchestrator:=false`). The
container holds two nodes:

1. **`puppy_playground` engine**, a `CapabilityEngine` subclass. It holds the arm only inside its
   actions, so cancel and e-stop behave exactly as for every RAMMP capability.
2. **`playground_ui` node.** It serves the web page on port 8080 and sends ExecuteAction goals and
   Control requests to the engine. It uses Python's standard `http.server` with plain HTML and
   JavaScript, so no web framework is added.

### Actions

| Action | Params | What it does |
|---|---|---|
| `stage` | — | Plans rest → stage pose, checks the path, executes it |
| `rest` | — | Plays the stage path backwards (the same checked path) |
| `express` | `name`, `size` 0.25–1.0 | One expression at the stage pose |
| `combo` | `names` (list), `size` | Expressions back to back, each checked |
| `auto` | `size`, `seed`, `max_minutes` (default 10) | Auto mode until cancelled or timed out |

`express`, `combo` and `auto` refuse unless the measured posture is within 0.02 rad of the stage pose
and every joint is slower than 0.02 rad/s.

### The stage pose *(call)*

The stage pose is one joint configuration in front of the chair:

- the forearm, wrist and head are at least 0.35 m from every occupant box, so expressions play as FREE
  at full size. The upper arm is excluded: it starts at joint 2, which barely moves, and stays within
  about 6 cm of the occupant boxes in every posture. A search of 6,000 postures found 34 that meet
  this rule facing forward and level, all with the tool out to the arm's left, away from the chair;
- the tool clears the seat and platform boxes by at least 0.10 m;
- the face points forward and level, and all joint guards are met with margin.

It is chosen in the simulator, with the chair boxes loaded, and shown to the user as a still image for
approval before any hardware run.

### Moving to and from the stage pose

The stage path is planned once, approved, and stored *(call)*.

- A `plan_stage` action asks cuRobo for a plan under the empty-world profile and moves nothing.
- The simulator renders the path with the chair boxes for the user to approve.
- The approved path is stored as a file in the repo.
- Stage and Rest replay that stored path; they never plan fresh.
- If cuRobo's path fails the rules below, a hand-built joint path, checked by the same rules, may be
  used instead.

1. `self.arm.plan_to_joints(stage)` plans under the empty-world profile, in `plan_stage` only.
2. Before the path is stored, and again at runtime before every replay, `puppy_motion` checks it
   against the chair boxes:
   - clearance to the occupant boxes never drops below its value at the start;
   - after the first 0.35 m of clearance is reached, it stays at or above 0.35 m;
   - the seat and platform floors hold;
   - the speeds stay within the arm's own envelope.
3. A path that fails is refused, and the page shows which rule it broke. It is executed with
   `self.arm.execute_joint_trajectory`.
4. `rest` sends the same checked path reversed, so the only approach to the rest pose is the one that
   left it.
5. Stage refuses unless the measured posture is within 0.02 rad of the stored path's start, and Rest
   unless it is within 0.02 rad of its end.

### Stopping and recovering *(call)*

- **Stop** cancels immediately, wherever the arm is. The hardware e-stop stays the real safety.
- **Recover.** A `recover` action, offered on the page whenever the arm is neither at the stage pose
  nor at rest, gets it back:
  - near the stage pose (within 0.10 rad on every joint), a short minimum-jerk joint move back to the
    stage pose, checked like any expression;
  - on the stored stage path, it rejoins the path at the nearest sample and continues back toward the
    rest pose, checked by the stage-path rules;
  - anywhere else it refuses, and the page says to recover by hand under the e-stop.
- **Soft stop in auto mode.** Pressing an expression button while auto mode runs pauses it at the next
  boundary between beats (Control PAUSE), so the arm is still at the stage pose. The UI then cancels
  auto mode and sends the expression. Stop stays a hard cancel.

### Expressions *(call)*

The page offers every Puppy Mode behavior that plays at a still pose: breathe, look around (gaze), head
tilt, perk, droop, puppy eyes, nod, bounce, wag, sniff, double take, search, startle, shake-off, proud,
doze and wake, and lock-on.

Lead gaze and follow-through ride on moves, so they are absent. Chomps are left out, as above.

Each expression runs through the same pipeline:
1. built at the measured stage posture with the chosen size;
2. composed by `puppy_motion.compose` with the chair boxes as person and obstacles;
3. checked by `check_beat`;
4. sent as one `execute_joint_trajectory` goal with positions and velocities.

"Look at you" aims at a point 1 m in front of the chair at standing head height *(call: no
perception, and the operator stands roughly there)*.

### Control mode *(call)*

Every playground motion uses IMPEDANCE with the driver's `default` (medium) profile, because no task is
there to copy and someone is standing nearby.

`execute_joint_trajectory` carries positions and velocities but no accelerations. The simulator's
driver model therefore checks that variant's overshoot before hardware. Every expression must stay
within its checks under it.

### Auto mode *(call)*

A small idle director (`puppy_motion/idle.py`) drifts through four moods:

| Mood | Behaviors | Dwell |
|---|---|---|
| calm | breathing, small glances | 20–60 s |
| curious | look around, head tilts, sniff | 15–40 s |
| playful | perk, wag, bounce, proud | 10–25 s |
| sleepy | droop, then doze | after 3+ min without a press |

How it behaves:

- Transitions are weighted by mood, and calm is the hub. Pauses between behaviors are 1–6 s.
- Everything is seeded, so a session can be replayed.
- Pressing any expression button cancels auto mode and plays that expression.
- Stop cancels.
- `max_minutes` ends the session.
- Each behavior goes through the same pipeline as a button. A behavior that fails its checks is
  skipped and logged, and auto mode carries on.

### The web page

- one button per expression;
- a combo builder: tap to queue, then Play;
- a size slider, starting at 0.25;
- Stage and Rest buttons;
- an Auto toggle;
- a large Stop button.

A status line shows the action, the mood in auto mode, and the last refused check. Each expression
button links its simulator preview clip. The page states plainly that the hardware e-stop is the
safety.

### Bring-up order

1. `mock:=true`: the arm interface moves nothing, while every button, check and log runs.
2. Simulator previews: the same behaviors, the stage pose and the stage path, rendered with the chair
   boxes.
3. Real arm, attended, size 0.25, one expression at a time.
4. Then full size, combos, and auto mode.

### Logging

Every request is written as one JSON line to a log file in the container's mounted volume. Each line
holds:

- the action and its parameters;
- the composed beat's peak speeds and clearances;
- the check results;
- what was sent and the driver's result.

## Left to the builder

- File layout inside the package, page styling, and exact UI wording.
- The idle director's weights and timing within the ranges above.
- The stage pose's exact joint values, within its rules (the user approves the image).
- How the UI node waits on goals (threads or the executor).

## Open questions (not blocking the build)

- Does the driver's trajectory executor build a smooth curve from positions and velocities alone? The
  simulator answers part of this; a slow first hardware run answers the rest.
- PR #150 is unmerged. Until it merges, the profile is run from that branch's checkout on snorlax.
- The chair boxes are best guesses. A measured chair would let the stage pose sit closer.
