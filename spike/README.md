# Puppy Mode: simulation preview (THROWAWAY)

This is a **throwaway preview**. It renders videos so people can judge what hand-designed "Puppy
Mode" motions would look like on RAMMP's Kinova Gen3 arm. It is **not** the add-on: it never
talks to a robot (no ROS, no network, no SSH). It is pure offline Python and MuJoCo, fed by arm
data that was already extracted from real `grab_container` rosbags.

Puppy Mode layers small expressive motions (breathing, glances, a curious head tilt, a perk, a
droop...) on top of what the task does. Inside anything precise (perception holds, the final
approach, the grip, the raise) it may only change timing, never the path. By default it acts only
between the task's moves (the boundary tier): the robot's driver plans and runs each move, so
that is all Puppy Mode could do on the real arm today.

## Run it

From `spike/`:

```sh
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

.venv/Scripts/python -m puppy_sim list                 # the clips
.venv/Scripts/python -m puppy_sim render all           # media/<clip>.mp4, .gif, _sheet.png
.venv/Scripts/python -m puppy_sim render idle_alive --fast   # half size, no GIF, in media/_work/fast/
.venv/Scripts/python -m puppy_sim sheet all            # contact sheets only (quick look)
.venv/Scripts/python -m puppy_sim check all            # safety checks; exit code 1 on a violation
.venv/Scripts/python -m puppy_sim peaks reel_reactions # media/<clip>_peaks.png + each beat's on-screen travel
.venv/Scripts/python -m puppy_sim compliance impedance_wag   # each beat as commanded vs as the arm would execute it
.venv/Scripts/python -m puppy_sim render idle_alive --control impedance:soft   # ... rendered: media/idle_alive__impedance_soft.*
.venv/Scripts/python -m pytest                         # tests
```

`compliance` and `render --control` simulate the execution (see "Compliance" below); both take
`--positions-only` (no velocity/acceleration feedforward), `--armature A`, `--friction N` (one value or
seven) and `--payload KG`, and `compliance` takes `--control` more than once (default: position and
impedance soft/default/stiff).

Outputs go to `media/`, and intermediate files go to `media/_work/`, which git ignores. Each clip
produces an H.264 MP4 (25 fps, 1280x720, or 1600x720 side by side), a GIF for documents (640 px
or 960 px wide, 12.5 fps, at most 8 MB), and a contact sheet of evenly spaced stills with
timestamps. `peaks` adds a sheet with one tile per beat at its peak and prints its legibility: how
far it moves the eye and the mouth (the farther fingertip) on screen in the close camera at
1280x720 (a beat should reach about 15 px).

## Clips

| clip | what it shows |
|---|---|
| `idle_alive` | 9.6 s at home: breathing, hello to the user, a curious tilt, a look at the container, back to forward |
| `curious_tilt` | at home the container catches its eye: a look, two sniffs, a double head tilt, a breath (puppy_motion's behaviors and compose) |
| `doze_wake` | nothing to do: it dozes off; a task request wakes it: startle, stretch, a nod to the user |
| `eager_start` | the start of run 132745, plain vs Puppy Mode, through puppy_motion's Director: eager at the request, then the recorded move (the demo budget keeps nothing else; 4.4 s -> 7.2 s) |
| `anticipation_pair` | a scripted free move, plain vs Puppy Mode (the Director, budget lifted): a wind-up before and a follow-through after, twice the vocabulary's size (+2.8 s) |
| `lock_on_view` | run 132745's approach: a lean-in stare at the view pose before the reach (the first robot version; +1.8 s) |
| `lock_on_standoff` | the same with a stop at the (about) 11 cm standoff and a x0.75 creep (needs the reach split into two calls; +3.0 s) |
| `boop_button` | a scripted button press with a staging stop, plain vs Puppy Mode (the Director): a lean-in stare at the button before the approach, the press, a proud look (+5.5 s, demo budget) |
| `success_proud` | two successes through the Director, a pick and a put-back: proud and attentive while holding (level), proud (with chomps) and attentive empty-handed (+11.9 s, two demo budgets) |
| `oops_retry` | illustrative: a miss (aborted, recoverable failure), then search, sorry, shake-off, a lock-on before the retry and a proud (budget lifted to 30 s, +21.7 s) |
| `wag_compare` | the same wag inside today's envelope vs the 'snappy' wrist (3 vs 9 rad/s^2) |
| `replay_plain` | the real run `grab_container_20261029_132745` replayed with no personality (counter camera) |
| `replay_pair` | that run plain and through the director side by side: beats only between moves, the 'demo' budget, the lock-on at the view pose (15.1 s plain, 21.0 s Puppy Mode) |
| `failure_pair` | the real refusal in run 191809 ("frames disagree > 1 cm"): sorry and a short shake-off (14.5 s -> 20.1 s; the request is before the window) |
| `reel_at_rest` | breathe, micro-life, gaze, head tilt, perk, droop with puppy eyes, doze and wake, each once at home, its name in the chip |
| `reel_reactions` | nod, bounce, wiggle, clack, sniff, double take, search, startle, shake off, proud |
| `reel_craft` | wind-up, follow-through and lock-on between scripted moves; the lead gaze rides on one (labelled 'later: in motion') |
| `impedance_wag` | the happy wag, simulated: position mode vs the driver's impedance mode with its default gains (the wag reads the same: 52 vs 54 px) |
| `impedance_lock_on` | lock_on_view's Puppy Mode track, simulated: position vs impedance default (the lean reads the same; holding the container, the springy arm sags up to 37 mm) |
| `impedance_profiles` | success_proud's Puppy Mode track, simulated under the 'soft' vs 'stiff' gains (holding, the arm sags up to 63 vs 18 mm) |
| `impedance_friction` | what-if: the wag on 'soft' gains with joint friction the driver does not cancel (assumed): it shrinks to about half |

Some clips register their own camera from the clip module (`lock_on`, `boop`, `oops_follow`,
`wag_close`) with `render.CAMERAS.setdefault`.

The overlays are the title, the current beat (mood chip), and a timeline bar colored by zone
(green FREE, amber NEAR_USER, blue PRECISION, labeled "focus: timing only"). A pair clip has a
header per side ('Plain (today)', 'Puppy Mode') with its own clock, which stops at its end while
that side holds its last frame; the subtitle states both durations.

Cameras (`render.CAMERAS`): `wide` (the whole scene, 3/4 front-right), `close` (the forearm and
face, fixed at the first pose), `close_follow` (the same, following the functional head, smoothed
over 0.5 s so expressions never move the camera), `counter` (a reverse angle from beyond the
counter: the face while the arm works there, the user behind it) and `user_pov` (over the user's
left shoulder at head height: what the user sees; from home the face looks at the counter, about
145 deg away from the user, so it shows the back and side of the head).

## What is real and what is assumed

- **Robot: real.** The kinematics are the menagerie Gen3 with its base tilted +30 deg about y on
  `arm_mount` (from the live URDF). Its FK matches the recorded `/ee_state`, which is
  `arm_end_effector_link` in `arm_mount`, to 0.07 mm and 0.01 deg in all three runs (`check`
  prints this). The Robotiq 2F-85 sits on the end-effector link, and its closed-chain linkage is
  posed kinematically. The OAK wrist camera is modelled as a small "face" (two lens rings for eyes,
  above the gripper "mouth").
- **Task motion: real.** The joints are recorded `/joint_states`, resampled to 50 Hz and lightly
  smoothed. Phases come from the capability's progress feedback.
- **Container and counter: placed from the run.** The container sits where the run closed the
  gripper: the brim's outer edge is 8 mm past the tool point along the approach, as in
  `grasp.py`, and the brim sits between the fingertips. The counter top is 9.5 cm below the brim,
  which puts it at kitchen height. Once the grip closes, the container rides rigidly in the
  gripper.
- **User and wheelchair: assumed and approximate.** They are stylized, using the numbers from the
  fridge keep-out and tf: seat at z -0.27, chair centerline y -0.32, head at (-0.27, -0.32, 0.50).
  The mannequin is plain, with no face.
- **Button panel: placed, and the press is scripted.** It stands on the counter. The press poses
  (closed fingertips along the panel normal, 2 cm standoff) are reachable by IK
  (`Scene.press_targets`, tested); `boop_button` scripts the press from them because there are no
  recorded button-press runs.

## The motion model

`q = q_func(tau(t)) + gain(zone) * dq_expressive(t)` on one 50 Hz clock.

- **Zones** come from a per-capability table (phase + message) plus geometric rules. Unknown
  phases, and unknown messages of known phases, are PRECISION. The final approach (tool within
  0.11 m of the grasp pose) is PRECISION.
  NEAR_USER applies when the head (bracelet, face, gripper, anything held) is within 0.35 m of the
  user.
- **Gains** are FREE 1, NEAR_USER 0.35, PRECISION exactly 0. They ramp with raised cosines that
  finish before a lower zone starts. Inside PRECISION, q is copied from the functional track
  (bit-identical).
- **The time-warp** (Hold, Slow) changes timing only. Speed dtau/dt stays within [0, 1], and the
  warped track is always the functional path evaluated at tau (it keeps that track as `source`).
  A Slow refuses a factor outside (0, 1], and ramps round up to whole samples, never shorter than
  asked, so a ramp sized by `hold_ramp_s` keeps its acceleration cap.
- **Head offsets** `[dx, dy, dz, roll, pitch, yaw]` turn into joint offsets by damped least squares
  at the functional posture. The joint weighting favors the wrist, and turns pivot about the
  wrist center (the Gen3 wrist is spherical), so the shoulder and elbow stay put.
- **Caps, inside the arm's own envelope** (`layer.CAP_PRESETS`, zone-tiered). `envelope.py`
  measures what RAMMP's task motions already use (the three runs, FREE samples while moving): joints
  0.36-1.12 rad/s and 0.84-3.22 rad/s^2, the eye 0.45 m/s and the tool point 0.40 m/s. In FREE the
  expressive part stays under 0.85 rad/s per joint and no faster than the task moves that joint
  (joints 1, 3 and 4: 0.36, 0.59, 0.64), 3 rad/s^2, and 0.40 m/s at the eye and tool points; a test
  holds these under the measured envelope. Near the user it is gentler: 0.5 rad/s, 1.5 rad/s^2 and
  RAMMP's 0.16 m/s. A sample takes the caps of the most restrictive zone within a gain ramp's
  reach, so a ramp into NEAR_USER or PRECISION is already under the gentle caps. The gripper offset
  stays under the 2F-85's 150 mm/s. 'snappy' (wrist joints 5-7 at 9 rad/s^2 in FREE) is a
  comparison preset, used only where a clip asks for it (`compose(..., caps="snappy")` with
  `Context.of(..., caps="snappy")`); `check` holds a track to the preset it was composed with.
- The composite stays under the URDF velocity limits and RAMMP's joint guards. Every arm part
  keeps a clearance floor (the fingers as composed, so a chomp cannot reach the counter): to the
  user, min(functional, 0.35 m) - 5 mm; to the counter, panel and chair, min(functional, 3 cm). A
  behavior that would break one shrinks its own amplitude (including a look that compose's own gain
  ramp would move too fast), and the shrink is logged.
- `check` verifies all of this independently, plus continuity, and prints a table: the preset, the
  'free' and 'near' tiers, `tier` ('boundary' when every move of the task plays exactly as
  recorded: same timing, no offset; else how many samples it retimes and layers onto; a track that
  promises the boundary tier must keep it), and for a warped track the path (the same path at tau,
  every precise sample still PRECISION), what the pauses add to the motion (`warp accel`, within
  each tier's acceleration cap) and the time they add (`added s`, within a directed track's
  budget). It also prints the measured envelope and the caps.

## The vocabulary (`behaviors.py`)

Each behavior is `behavior(ctx, [target,] t_on, ..., beat=None) -> layer.Contribution`. `ctx` is a
`behaviors.Context`: the (warped) functional track, a `Kin`, the user's head, the container, the
button, the counter height, a seed and the cap preset it paces to. Every amplitude and timing is a named constant at the top of
the module. Moves pace themselves to the caps (`paced_move`, `paced_bump`), oscillations limit their
own amplitude (`fit_scale`, noted in the log), and big static poses stop short of the clearance
floors (`safe_scale`); compose's auto-scaling is the safety net on top.

- Alive at rest: `breathe`, `micro_life` (at most 2 mm and 0.5 deg), both off in PRECISION.
- Looking: `gaze_to` (eased turn with 6 % overshoot, the lean 0.1 s behind, optionally rising as it
  turns), `look_at_user`, `head_tilt`, `double_take`, `search`, `sniff` (pecks at a 10 cm standoff).
- Moods: `perk`, `droop`, `puppy_eyes`, `startle`, `nod`, `bounce`, `wiggle`, `shake_off`, `clack`
  (the gripper as a mouth), `proud`, `doze_and_wake`.
- Motion craft, self-contained so it can play between arm calls: `wind_up` (a draw-back against
  the coming move and back, at rest when the move starts), `follow_through` (once the arm is still,
  a carry-on and settle), `lock_on` (a lean toward the target and a stare, back before the next
  move). `lead_gaze` (the gaze 0.25 s ahead of the path) rides on a move: the in-motion tier.
- `wiggle` and `shake_off` are joint-space wags: every joint goes as far as its own acceleration
  cap allows, signed so they all push the face the same way; the wrist pans and rolls both ways,
  and the arm sways the head out to the side the clearance floors leave room on and back (beside
  the user it only swings away from them). Its size is noted in the log.

Two kinds of turn: face turns (a tilt) roll about the gaze; gravity turns (a nod, a perk,
a droop, a look around) tip the gaze toward the floor or pan it about the vertical. At the task's
view and grasp postures the face is rolled 90 deg, where a head-frame nod would read as a "no".
Gating is each behavior's own job too: no wiggle or clack while holding or near the user (clack
only in FREE), and while holding a container looks only pan and nothing tilts it.

## The director (`director.py`)

`director.RULES` is the mapping, a table per capability: task event -> beat, its mood chip, its
tier, its priority (`keep`) and the time held back for it (`reserve_s`); several rules may share an
event and play in table order. For `grab_container`: the request gets an eager perk (and a glance
at the user in full); a wind-up before every transit; a follow-through after every arrival; the
arrival at `view` a curious tilt (and sniffs in full); the stop after `detect`, while the gripper
opens, the lock-on stare; after the raise a muted proud (and an attentive tail in full); a failure
a puzzled search and a droop (puppy eyes and a shake-off in full).

- **Boundary tier (default).** A beat plays only in a stage, where every joint is slower than
  0.02 rad/s for 0.2 s (or across a phase change) outside PRECISION, and starts and ends there at
  rest; a fence zeroes it wherever the arm moves. Pauses are time-warp holds inside a stage only,
  so every move plays exactly as recorded (`check`'s `tier` verifies it, tested on all three runs).
  `tiers=(BOUNDARY, IN_MOTION)` adds the lead gaze (Puppy Mode would have to run the move itself).
- **The lock-on.** The approach (view -> pre-grasp -> grasp) is one blended driver call, so by
  default (`lock_on='view'`) the stare plays at the view pose after detect, before the approach,
  which then runs as recorded. `lock_on='standoff'` is the old one: stop at the pre-grasp standoff,
  stare, creep in at x0.75; it needs the capability to split its approach into two calls, and
  `check` reports the samples it retimes.
- **Budgets, event by event.** 'demo' allows 6 s per task, 'subtle' 2.5 s and plays only the most
  important beats (`budget=None`: no limit). Each event is decided when it arrives, from the time
  spent so far: a beat plays in full if it fits the time left less what is held back for more
  important beats still to come (only the larger of the success and failure endings is held back),
  else its short version, else it is dropped. A beat is priced at what its stop would add as far as
  the director knows: a beat before a move exactly (the move's start is known), a beat after an
  arrival as if a precise move followed at once, and after the task ends up to an assumed next
  command 1.9 s later (`NEXT_COMMAND_S`; the replays keep 2 s after the end). The stop is then
  charged what it really added, ramps included; the log lists every decision, and a test holds the
  decisions on any prefix of the events equal to those on the whole run.

Unknown phases are PRECISION and get no beat, and every beat is fenced out of PRECISION, so no
spatial offset ever lands on a precise sample (tested on all three runs).

## Compliance: the driver's execution modes (`dynamics.py`)

Everywhere else the spike is kinematic, so every video shows the command tracked perfectly. On the
arm a capability runs its moves in POSITION mode or in the driver's joint IMPEDANCE mode with gains
(a profile, 'soft' | 'medium' | 'stiff' | the session default, which starts as 'medium', or custom
{kq, zeta, torque_limit}), and a Puppy Mode beat runs in the task's mode and gains. `dynamics.py`
simulates that: MuJoCo forward dynamics of the menagerie Gen3 with gravity, the 2F-85 welded open,
the OAK face, and while held a 0.4 kg container (assumed) whose base stands on the counter until the
arm lifts it (the counter is a contact for it, nothing else touches).

- **IMPEDANCE** ports kinova-gen3-driver 2.0.0's `JointImpedanceMode::compute` at its 1 kHz:
  `tau = g(q) + ramp * (Kq * clamp(q_d - q, +-leash) - D * (qd - qd_ref) + M(q) qdd_ref)`, then
  clamped to the torque limits, with `D = 2 zeta sqrt(Kq M_ii(q))` and the reference sampled from the
  track by quintic Hermite as the trajectory executor does (velocities and accelerations fed forward,
  as cuRobo's plans carry them; `feedforward=False` is a positions-only stream). The profiles are the
  driver's: soft Kq 40/15 N m/rad, zeta 0.4, leash 0.45 rad; medium 80/30, 0.5, 0.35; stiff 160/60,
  0.7, 0.25 (shoulder and elbow / wrist); torque limits 39/9 N m. The driver's g and M come from its
  URDF: here the plant's own model without armature and without the container (it has no payload
  term, no Coriolis, no friction compensation).
- **POSITION** is Kortex's own actuator servo (firmware, not in the driver) fed through the driver's
  rate limit; it is modelled as exact, so it is the kinematic playback.
- **The plant's armature is the big unknown.** The menagerie gives every joint 0.1 kg m^2 (added to
  stabilise its position actuators, not identified), which the driver's damping does not know about:
  the wrist roll's effective damping ratio drops to 0.06. `Plant(armature=0)` is the other bound
  (the actuators' torque loops hide the rotor inertia). The menagerie has no joint friction;
  `Plant(friction_nm=...)` adds some as a what-if (the driver does not compensate friction).
- `dynamics.achieved(track, control)` is the same timeline with the joints the arm reaches (it
  keeps `commanded` and, for a composed track, its functional track achieved too, so a beat's
  legibility compares what the arm does with and without it); `check` holds the command and adds a
  note with how far the arm strays and its margins over the command's clearance floors.
  `dynamics.response` measures each beat commanded vs achieved, the tracking error and, inside
  PRECISION, where the Puppy Mode arm and the plain arm put the tool on the same commands.

What it shows: with the trajectory's feedforward the beats read the same in impedance as in position
mode (within -4 % to +10 %; if anything a little larger, not smaller: the arm's sway modes near 2 Hz sit close
to the 1.25 Hz wag). What compliance does show is the held container: the driver does not model
its weight, so while holding the arm sags 18 mm (stiff), 34-37 mm (default) or 63-65 mm (soft) in
the task itself. Inside PRECISION Puppy Mode adds nothing on lock_on_view (under 0.5 mm), but on
success_proud the proud beat while holding ends 0.46 s before the set-down and its motion has not
settled: the arm is up to 3 mm (default; 6 mm soft) off where the plain task's arm is, for about 1 s
(2.3 s soft). A positions-only stream overshoots every beat by 6-28 %; joint friction (assumed 1 /
0.5 N m) halves the wag on 'soft'.

## Findings so far

- **At home the arm is folded beside the user.** With the assumed seating, the elbow is 7 cm from
  the user's left shoulder and the wrist is 25 cm from the head. That is why NEAR_USER is measured
  from the head: counting every link would mark the whole workspace NEAR_USER. The elbow and wrist
  are guarded by their own clearance floors instead.
- **From home, a glance at the user is about 18 deg.** The head starts 0.38 m from the user and may
  not get closer than 0.345 m. From home the face looks at the counter, about 145 deg away from the
  user: no eye contact is possible from there.
- **Inside the arm's own envelope a 25 deg turn takes about a second.** It is the acceleration
  (3 rad/s^2 on the wrist joint that carries it), not the head speed, that paces it now.
- **A MuJoCo distance glitch is worked around.** `mj_geomDistance` sometimes returns 0 for well
  separated mesh/box pairs. `Kin.geom_distance` detects this and bisects on its reliable "farther
  than D" answer.
- **Quick motions stay small under 3 rad/s^2.** At 1.25 Hz one joint may swing about 2 deg, so a
  wrist-only wag moves the eye about 7 mm. The wag reads (about 50 px) because every joint swings
  at its own limit and the arm sways the head out away from the user and back: beside the user the
  folded wrist has 5 mm of slack toward them. 'snappy' (9 rad/s^2 on the wrist) doubles the wrist's
  part. Overlapping moves stack their accelerations, so compound beats sequence their parts (a
  proud lift rides with its look as one move).
- **Puppy Mode costs time, and the robot cannot know how long it will wait anyway.** Every look and
  back is about 2 s. With 'demo' (6 s): on 132745 an eager perk, the curious tilt, a short lock-on,
  a short wind-up and a short proud (15.1 s -> 20.7 s); on the failure run 191809 the eager perk,
  the curious tilt, a free stare before detect, a one-spot search and a sorry with puppy eyes, with
  no room for the shake-off (19.9 s -> 25.3 s). 'subtle' keeps a short stare and a short proud
  (16.1 s) or a short sorry (20.1 s). A full apology (search, droop with puppy eyes, shake-off)
  needs about 6.6 s on its own, more than the whole demo budget if the task also started eagerly.
- **At the counter the face is turned on its side.** A look at the user from there is a pan (the
  user is behind it): about 1.3 s for 20 deg while holding, where the pan is also what keeps the
  container level. Seen from beyond the counter, a curious tilt at the view pose rolls the face
  about an axis that points across the view, so it moves the eye only about 8 px there (15 px in
  the close camera); the lock-on lean reads at about 13-15 px, the proud look at about 45-90 px.
- **The real runs pass the wrist singularity** (joint 6 near 0) during transits. There a head turn
  needs the shoulder, so `lead_gaze` eases out around it.
- **At home the user's head is below the face**, so puppy eyes from home are a sheepish glance
  toward them rather than up at them, and a droop sags down and away from them: a straight sag
  would bring the folded elbow within the 5 mm person slack.

## Extending it

- **A clip** is a module in `puppy_sim/clips/` exposing `CLIP = Clip(name, title, subtitle,
  camera, build)`, or `PairClip(...)` whose `build()` returns `(plain, puppy)` for side by side.
  `clips.rest_reel(slots)` builds a reel of primitives at a rest pose.
- **A behavior** is a function `(ctx, ..., t_on, ..., beat=None) -> layer.Contribution` on a
  `behaviors.Context`. Pace it with `behaviors.paced_move` / `paced_bump`, limit an oscillation
  with `fit_scale`, keep a static pose clear with `safe_scale` (`safe_joint_scale` for joint
  offsets). If the task must pause for it, add a `timeline.Hold` where the arm is already stopped
  (`director.stages`), sized by `hold_ramp_s`; `timeline.output_time` / `still_span` place a beat
  inside it.
- **A director beat** is a row in `director.RULES` plus a `director.Beat` in `director.BEATS`
  (`play(ctx, t0, t1, short)`, and `length(ctx, event, short)` unless it is self-timed);
  `director.direct(run, sc, tiers=..., budget=..., lock_on=..., caps=...)` returns `(plain, puppy)`.
- **Scripted functional moves** come from `timeline.joint_move`, `kin.cartesian_path`,
  `timeline.concat`, `timeline.constant_track` and `runs.mark_near_user`.

Code: `scene.py` (model), `kin.py` (FK/IK/look-at/gravity turns/clearance), `runs.py` (real runs,
phases, zones), `curves.py`, `timeline.py` (Track, time-warp, scripted moves), `layer.py`
(composition, gains, cap presets), `envelope.py` (the task's own motion envelope), `behaviors.py`
(the vocabulary), `director.py` (task events to beats), `checks.py`, `render.py` (cameras,
overlays, legibility, peak sheets), `dynamics.py` (the driver's execution modes, simulated), `clips/`,
`__main__.py`, and `tests/`.

The menagerie models in `third_party/` are BSD-licensed; see the LICENSE file in each folder.
