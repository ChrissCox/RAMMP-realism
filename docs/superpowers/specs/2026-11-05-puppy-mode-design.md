# Puppy Mode: expressive motion for the RAMMP arm (design)

Date: 2026-11-05. Status: approved for planning; the user delegated the three open questions to the
lead, and they are decided below. Visual companion: the "RAMMP Puppy Mode" doc
(<https://claude.ai/artifact/Rm6oByCsLQMb7nRnFSab7q>) shows every motion as video.

## Intent

The RAMMP Kinova Gen3 arm (Robotiq 2F-85, OAK wrist camera, wheelchair-mounted beside its user) does
its kitchen tasks competently but reads as a machine. Puppy Mode gives it a puppy-like personality on
top of those tasks, so people watching think it is alive, cute and a bit goofy. The reference is Dum-E,
the shop arm in the Iron Man films: no face, yet obviously eager, curious, clumsy and sheepish.

It is an experiment and an add-on. It must work with the capabilities RAMMP already has, it is off
unless someone turns it on, and for now it runs only in simulation.

The user's own framing of precision: there are moments when the arm must be extremely precise, and the
personality should make it *more* precise there, not less. A puppy that wiggles while it waits goes
perfectly still when it stares at the treat.

## Goals

1. Loose moments carry personality: at rest, between moves, after a success or a failure, and when
   attending to the person.
2. Precise moments carry focus: the arm stops, locks on and approaches more slowly. Expression there is
   timing only and never changes the path.
3. It is an add-on: existing capabilities keep working unchanged, with no rewrites, behind one switch
   that defaults to off.
4. Its safety rules hold provably: they are enforced in code and checked by tests and by the simulator
   on every clip.
5. A simulator shows each behavior and real recorded tasks with and without Puppy Mode, so the team
   can judge the feel before anything touches hardware.

## Non-goals (this phase)

- Running on the real arm. Hardware trials are a later, separately approved phase.
- Sound. The robot computer has no speaker or microphone attached yet.
- Learned motion (ACT-style puppeteering). The behavior library is built so learned clips can replace
  hand-designed ones later.
- Perceiving the person. Their position is configured, not tracked.
- Changing any capability's task logic, perception or motion targets.

## Anti-goals (a failure even if it works)

- Any change to where the arm goes during precise work: perception holds, final approaches, grasps,
  contact moves, lifts.
- Moving faster or accelerating harder than the arm's own task motions already do, or moving toward
  the person.
- Constant unexplained fidgeting. ELEGNT found it reads as inattentive.
- Hiding or delaying safety behavior. There are no beats after an e-stop, a lost control, or a cancel,
  and a cute reaction never replaces an error report.
- Unbounded slowdowns, or identical canned loops that grate on the tenth viewing.
- Asking capability authors to restructure their code.

## Constraints from RAMMP

These come from `rammp-kitchen-capability-engines` (`AGENTS.md`, `capability_engine/`):

- `self.arm` works only inside `execute()`. An engine runs one goal at a time, and the orchestrator
  holds the arm lock for that goal. Another sender commanding the arm raises `ControlTaken`. Code
  never calls `/acquire_control`, `/release_control` or `/revoke_control`, and never publishes
  `/estop`.
- Changes to `capability_engine/` are small, additive and in their own PR.
- Hardware runs are attended, with a human on the e-stop. This phase has none.
- Speed parameters: `arm.JOINT_SPEED` is 0.85 rad/s and `arm.CART_SPEED` is 0.16 m/s. Joint guards:
  |joint 4| ≤ 144°, |joint 6| ≤ 115°, |joint 2| ≤ 2.24 rad. The URDF velocity limits are 1.3963 rad/s
  for joints 1-4 and 1.2218 rad/s for joints 5-7.
- Measured envelope: three recorded grab_container runs (2026-10-28/29, arm data only) give the arm's
  own open-space motion as up to 1.12 rad/s, about 3.2 rad/s², and a tool speed of 0.40 m/s.

## Decisions

The user decided:

- the puppy character (Dum-E);
- that it is an experimental add-on for current capabilities;
- focus in precise moments;
- hand-designed motions first;
- simulation only for now;
- the GitHub repo;
- a doc with videos.

Everything else below is the lead's call, marked *(call)*. It can be changed.

### Zones *(call)*

Every 50 Hz sample of a task is in one zone:

| Zone | When | Expressive gain |
| --- | --- | --- |
| PRECISION | perception holds, final approach from the pre-grasp standoff, grasp, contact moves, lift until clear, and any unknown phase | 0. Timing only |
| NEAR_USER | the arm's head (bracelet, face, gripper, held object) is within 35 cm of the person | 0.35, with any component toward the person removed |
| FREE | everything else | 1 |

Gain ramps (raised cosine, about 0.3 s) finish before a lower zone begins. Every arm link keeps a
clearance to the person of at least min(task's own clearance, 0.35 m) − 5 mm.

Zones come from a per-capability table that maps (progress phase, message) to a zone, plus geometric
rules for the final approach and the person. A pair the table does not know maps to PRECISION.

### The timing-only rule

Inside PRECISION the commanded joint path is exactly the task's path at the same task time. Puppy Mode
may pause there or slow down (time-warp speed dτ/dt in [0, 1]), never speed up or deviate. Pauses
decelerate and accelerate smoothly, within the acceleration cap.

### Caps *(call; grounded in the measured envelope)*

These limits apply to the expressive part alone:

| Zone | Joint speed | Joint accel | Head speed (eye and tool points) |
| --- | --- | --- | --- |
| FREE | 0.85 rad/s | 3.0 rad/s² | 0.40 m/s |
| NEAR_USER | 0.5 rad/s | 1.5 rad/s² | 0.16 m/s |

The combined motion stays inside the URDF velocity limits and the joint guards. A behavior that would
break a cap shrinks itself and logs it; the task's own motion is never scaled.

A real puppy-speed tail wag needs about five times the acceleration the arm uses today. Under these caps
the wag becomes a slower happy sway. A `snappy` preset (wrist joints up to 9 rad/s², open space only)
exists for one comparison clip and is off by default.

### Vocabulary *(call)*

There are 23 behaviors. At rest: breathe, micro-life, gaze-to, head tilt, perk, droop, puppy eyes, and
doze-and-wake. Reactions: nod, bounce, wiggle (sway), clack (the gripper as a mouth), sniff,
double-take, search, startle, shake-off, and proud. Motion craft: wind-up (anticipation), lead gaze,
follow-through, and lock-on.

Each behavior is a function of the task's track, its targets (the person, the object, the button) and
a seed. It returns head offsets, base motions, gripper play and the beat labels. All amplitudes and
timings are named constants.

### The director *(call)*

The director is a readable table per capability that maps task events to beats. It works event by
event, the way it must on the robot: each decision uses only the events so far, never the task's
future or total length. The simulator feeds recorded runs to it in order, so the simulator and the
robot behave the same. For grab_container:

| Event | Beats |
| --- | --- |
| task requested | eager: perk, look toward the person, nod |
| arm stopped before a free move | wind-up (a small counter-move that ends still) |
| arm stopped after a free move | follow-through (a small overshoot and settle) |
| arrival at the view pose | curious: tilt, sniff toward the container. Then stillness through detect |
| after detect, before the approach call | lock-on: a still stare at the container from the view pose |
| after the lift is clear | proud, muted while holding (no sway, no clack) |
| at the end | attentive hold toward the person |
| recoverable failure | search, then sorry (droop and puppy eyes), then shake-off |

Beats come in two tiers:

- **Boundary** beats play only while the task's motion is fully stopped between two arm calls. They
  start and end at that posture with zero velocity and never overlap a move. Because the bags have
  no record of arm-call boundaries, the simulator infers them from joint speed and phase changes.
- **In-motion** beats layer onto a moving trajectory: lead gaze, or a stop and slow-in inside one
  blended driver call.

Only boundary beats are possible without Puppy Mode planning and executing moves itself. The first
robot version therefore uses boundary beats only, and the simulator shows that version by default.
In-motion previews are labelled "later".

grab_container's approach (view → pre-grasp → grasp) is a single blended `go_to_ee_poses` call. So in
the boundary tier the lock-on stare happens at the view pose before that call, with no slow-in. The
alternative is to stop at the pre-grasp standoff and creep in at ×0.75. That needs the capability to
split its approach call, and it is shown as a separate preview (open question 2 below).

Time budgets are absolute, because the robot cannot know a task's length in advance: `demo` adds at
most 6 s per task, and `subtle` at most 2.5 s. When a beat would exceed the budget it shrinks first,
then is dropped in a fixed priority order. Lock-on and the success or failure reaction go last.

A beat is priced at what it would add as far as the director knows at that moment:

- A beat before a move is priced exactly, because the move is about to be called.
- A beat after an arrival is priced as if a precise move followed at once.
- After the task ends, it is priced up to an assumed next command 1.9 s later *(call; the robot
  version should measure this)*.

Only the larger of the success and failure endings is held back. In the simulator this spends 5.6 of
the 6 s on the successful run (15.1 s → 20.7 s). A full apology (search, droop with puppy eyes,
shake-off) needs about 6.6 s on its own, so on a failed task that started eagerly the shake-off
drops.

A failure reaction plays only on a recoverable `EngineError`. The recorded failure (run
2026-10-28 19:18:09) is one: "grab failed: frames disagree > 1 cm -- not grasping -- the arm holds
where it is". grab_container raises that refusal as recoverable.

### Architecture on the robot *(call)*

1. **`puppy_motion` library** (pure Python, no ROS): curves, behaviors, the director, the composer and
   the checks. The simulator and the robot share it.
   - It must run inside the Humble capability images: Python 3.10, and numpy 1.x or 2.x.
   - It has no MuJoCo dependency; MuJoCo belongs to the simulator package only.
   - Kinematics come from the driver's URDF (`/arm/robot_description`) at runtime, never from a
     hard-coded mount. The tilt between `arm_mount` and `arm_base_link` differs between RAMMP robots.
2. **Engine hook** (a small additive `capability_engine` change, in its own PR when upstreamed). When
   the ROS parameter `puppy.enabled` is true (default false), `_on_execute` wraps `ctx` and `self.arm`
   in thin proxies:
   - `ctx.progress(...)` forwards, then reports a phase event to the director.
   - `self.arm.go_to_*` and the other move calls forward unchanged. Before and after each call the
     director may play boundary beats.
   - Calls the zone table marks PRECISION, contact moves and streams get only pauses. A move's
     control mode does not change its zone (see "Control mode" below).
   - After `execute()` returns, a success plays the success beats. A recoverable `EngineError` plays
     the failure beats, then re-raises unchanged. A `Cancelled`, `EStopped` or `ControlTaken` result,
     or any other exception, plays nothing.
   - A beat is a short joint trajectory sent with `self.arm.execute_joint_trajectory`. It starts and
     ends at the measured posture, is validated against the caps, the joint guards and the configured
     person zone, and runs after a `ctx.checkpoint()` so pause and cancel still work.
   - On the robot, beats move the arm only. Gripper chomps stay in the simulator until the hardware
     phase *(call)*, because they add a pinch hazard and need the gripper resource.
   - A beat never changes a task's result:
     - Beats after the task stop at their first exception. The task's own result is reported either
       way: its payload, or the original `EngineError`.
     - Between moves, a `Cancelled`, `EStopped` or `ControlTaken` from a beat propagates as it would
       from the capability's next call.
     - Any other refusal from the driver turns beats off for the rest of the action, and the task
       goes on.
3. **Idle engine** (later). A `puppy_idle` capability plays at-rest behaviors when nothing else runs.
   It needs the orchestrator to preempt it instantly for a real task.
4. **In-motion tier** (later). Plan with `plan_to_*`, add offsets in FREE parts, validate, and execute
   with `execute_joint_trajectory`.

### Control mode *(call)*

The driver runs a move in POSITION mode (stiff) or joint IMPEDANCE mode (compliant, with a gain
profile: `default`, `soft`, `medium`, `stiff` or custom `kq`/`zeta`/`torque_limit`).
grab_container, for example, runs every move in impedance when its goal sets `impedance: true`.

- **A beat copies the task.** It is sent with the same `control_mode` and `gains` as the task's most
  recent move call; the proxies see both on that call. The arm therefore never switches controllers
  in the middle of a task, which could cause a jump or a settle.
- **Near the person, always impedance.** In NEAR_USER a beat uses IMPEDANCE with the task's gains,
  or the `default` profile if the task is in POSITION mode. A compliant arm yields if it is bumped.
- **No task move is changed.** The task's own move calls keep whatever mode they asked for.
- **Measured in simulation: impedance does not soften the beats.** The simulator runs the driver's
  joint impedance law from kinova-gen3-driver 2.0.0 (`spike/puppy_sim/dynamics.py`, 1 kHz):
  gravity, spring, damping 2·zeta·sqrt(Kq·M_ii), and velocity and acceleration feedforward.
  - Beats sent as full trajectories (positions, velocities, accelerations) read within −4 % to +10 %
    of position mode, under every profile: medium (the session default), soft and stiff.
  - Two things do change:
    - **Held objects sag.** The driver doesn't compensate payload, so a 0.4 kg container sags about
      6 cm on soft, 3.5 cm on medium and 2 cm on stiff. That is the task's own sag and is the same
      with or without Puppy Mode.
    - **Unmodelled joint friction could halve a wag on soft.** It is unmeasured; measure it in the
      hardware phase.
  - Beats must always send velocities and accelerations. Positions-only overshoots by 6–28 %.
- **Settle before a precise move under impedance while holding.** A beat that ends just before a
  precise move leaves the springy arm still settling. The director keeps a settle time between a
  beat's end and the next PRECISION move while holding something: soft 2.3 s, medium 1.1 s, stiff
  0.1 s. In position mode, or when not holding, it keeps the usual 0.34 s.
- **The timing-only guarantee is about commands.** In impedance mode the achieved path already differs
  slightly from the commanded one for the task itself. Puppy Mode must not add to that inside
  PRECISION: the simulator checks that the achieved path there is the same with and without Puppy
  Mode, so the controller state a beat leaves behind cannot leak into a precise move.

Config is declared in the capability fragments (`tier: experimental`): `puppy.enabled`,
`puppy.preset`, `puppy.person_zone` (a shape in `arm_mount`), and the zone tables, kept in the add-on
package and keyed by capability.

### Simulator *(exists as a throwaway spike in `spike/`)*

The simulator is MuJoCo with the menagerie Gen3 and 2F-85. Forward kinematics matches the recorded
`/ee_state` to 0.07 mm. The scene has:

- a stylized wheelchair and user (positions approximate, taken from the fridge task's chair keep-out);
- a counter, container and button panel;
- real runs replayed from the robot's bags.

It renders MP4s, GIFs and contact sheets with zone bars, and a `check` command enforces every rule
above. The spike proved the design. The plan ports what it learned into `puppy_motion`, with tests,
instead of keeping the spike code as is.

### Capabilities covered in this phase *(call)*

- **grab_container:** code on `feature/12-grab-container`, plus three recorded runs (two successes,
  one recoverable refusal).
- **button_press:** code on `feature/4-button-press`. It has no recordings, so its motion is scripted
  in the simulator from the press poses.

Other capabilities get tables later. Each table must be checked against the capability's current
progress strings; a string the table does not know falls back to PRECISION.

## Decided questions *(the user delegated these to the lead)*

1. **Scope of the build phase.**
   - It covers the `puppy_motion` library, the simulator (rebuilt on that library), and the engine
     hook.
   - The hook is tested against `capability_engine`'s fake driver, so nothing moves.
   - It lives in this repo as its own ROS 2 package (`puppy_mode_ros`). It hooks `capability_engine`
     at a pinned version, and the patch to `capability_engine` is kept as small as it would be in an
     upstream PR.
   - There is no PR to `rammp-org` before the December demos.
   - Why: the hook is where most of the real risk lives (proxies, cancel, e-stop, control mode), and
     the fake driver lets it be proven without hardware or touching the demo code.
2. **Lock-on placement on the robot.**
   - The first robot version stares at the view pose, before the approach call. That needs no change
     to grab_container and adds about 0.9 s.
   - The 10 cm stop-and-creep stays an opt-in option for the simulator, plus a later per-capability
     opt-in (the capability would have to split its approach call).
   - Why: it keeps the anti-goal "no change to a precise motion's own command" absolute.
3. **The person rule.**
   - Keep today's rule: no arm link comes closer to the person than min(the task's own distance,
     35 cm) − 5 mm.
   - Near the person (NEAR_USER), motion toward them is removed and the gain is 0.35.
   - Why: a glance toward the person is most of the charm. The rule never lets any link inside
     35 cm unless the task itself goes there, and impedance is used near the person (see "Control
     mode").

## Left to the builder

- Package layout and names, apart from those above.
- How the proxies intercept calls (wrapper objects or subclass), as long as `puppy.enabled=false` sends
  the driver exactly the same goals as today.
- Logging format: every beat logged with its name, timing, peak speeds and clearances.
- Variety: seeded variation so a repeated task doesn't replay identical beats.
- Test organization.

## Open questions for later phases (not blocking simulation)

- Collision safety of beats on hardware. Puppy Mode has no world model; can cuRobo validate a short
  joint trajectory against its collision world?
- Does the driver apply `speed_scale` as pure time scaling of the same path? If so, a slow-in is
  timing-only on hardware too.
- Can the orchestrator schedule and instantly preempt an idle engine?
- Where the person really is on each wheelchair setup: a measured zone or perception.
- Sound: a USB speaker for chirps, a microphone for reacting to the person's voice.
