# Puppy Mode: simulation preview (THROWAWAY)

This is a **throwaway preview**. It renders videos so people can judge what hand-designed "Puppy
Mode" motions would look like on RAMMP's Kinova Gen3 arm. It is **not** the add-on: it never
talks to a robot (no ROS, no network, no SSH). It is pure offline Python and MuJoCo, fed by arm
data that was already extracted from real `grab_container` rosbags.

Puppy Mode layers small expressive motions (breathing, glances, a curious head tilt, a perk, a
droop...) on top of what the task does. Inside anything precise (perception holds, the final
approach, the grip, the raise) it may only change timing, never the path.

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
.venv/Scripts/python -m pytest                         # tests
```

Outputs go to `media/`, and intermediate files go to `media/_work/`, which git ignores. Each clip
produces an H.264 MP4 (25 fps, 1280x720, or 1600x720 side by side), a GIF for documents (640 px
or 960 px wide, 12.5 fps), and a contact sheet of evenly spaced stills with timestamps.

## Clips

| clip | what it shows |
|---|---|
| `replay_plain` | the real run `grab_container_20261029_132745` replayed with no personality; the timeline bar carries the real phases |
| `idle_alive` | about 10 s at home: breathing, a glance at the user, a curious tilt, a look at the container, then back to forward |
| `vocab_reel` | every primitive of the vocabulary once, at home in the close camera, its name in the chip (103 s); the motion-craft ones ride on short scripted moves |
| `replay_puppy` | the same real run through the director: the beats in its free moments, only timing in its precise ones (34 s instead of 15 s) |

The overlays are the title, the current beat (mood chip), and a timeline bar colored by zone
(green FREE, amber NEAR_USER, blue PRECISION, labeled "focus: timing only").

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
- **Button panel: placed for a later clip.** It stands on the counter. The press poses (closed
  fingertips along the panel normal, 2 cm standoff) are reachable by IK (`Scene.press_targets`,
  tested).

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
- **Caps**: the expressive part stays under 0.85 rad/s per joint, 3 rad/s^2, and 0.16 m/s at the
  head (eye and tool point), and a gripper offset under the 2F-85's 150 mm/s. The composite stays
  under the URDF velocity limits and RAMMP's joint guards. Every arm part keeps a clearance floor
  (the fingers as composed, so a chomp cannot reach the counter): to the user, min(functional,
  0.35 m) - 5 mm; to the counter, panel and chair, min(functional, 3 cm). A behavior that would
  break one shrinks its own amplitude (including a look that compose's own gain ramp would move
  too fast), and the shrink is logged.
- `check` verifies all of this independently, plus continuity, and prints a table. For a warped
  track it also holds the functional motion to the track it warped (`path`: the same path at tau,
  every precise sample still PRECISION) and bounds what the pauses add (`warp accel` <= 3 rad/s^2).

## The vocabulary (`behaviors.py`)

Each behavior is `behavior(ctx, [target,] t_on, ..., beat=None) -> layer.Contribution`. `ctx` is a
`behaviors.Context`: the (warped) functional track, a `Kin`, the user's head, the container, the
button, the counter height and a seed. Every amplitude and timing is a named constant at the top of
the module. Moves pace themselves to the caps (`paced_move`, `paced_bump`), oscillations limit their
own amplitude (`fit_scale`, noted in the log), and big static poses stop short of the clearance
floors (`safe_scale`); compose's auto-scaling is the safety net on top.

- Alive at rest: `breathe`, `micro_life` (at most 2 mm and 0.5 deg), both off in PRECISION.
- Looking: `gaze_to` (eased turn with 8 % overshoot, the lean 0.1 s behind), `look_at_user`,
  `head_tilt`, `double_take`, `search`, `sniff` (pecks at a 10 cm standoff).
- Moods: `perk`, `droop`, `puppy_eyes`, `startle`, `nod`, `bounce`, `wiggle`, `shake_off`, `clack`
  (the gripper as a mouth), `proud`, `doze_and_wake`.
- Motion craft: `wind_up` (anticipation), `lead_gaze` (the gaze runs 0.25 s ahead of the path),
  `follow_through` (overshoot and settle at a stop), `lock_on` (the stare before PRECISION).

Two kinds of turn: face turns (a tilt, a wiggle) roll about the gaze; gravity turns (a nod, a perk,
a droop, a look around) tip the gaze toward the floor or pan it about the vertical. At the task's
view and grasp postures the face is rolled 90 deg, where a head-frame nod would read as a "no".
Gating is each behavior's own job too: no wiggle or clack while holding or near the user (clack
only in FREE), and while holding a container looks only pan and nothing tilts it.

## The director (`director.py`)

`director.RULES` is the mapping, a table per capability: task event -> beat, the pause it needs,
an optional slow-in, the mood chip. For `grab_container`: the request gets an eager beat (perk,
glance, nod); every FREE transit gets a wind-up, a lead gaze and a follow-through; the arrival at
`view` a curious tilt and sniff before `detect`; the final approach a lock-on stare at the
precision boundary and a 0.75 slow-in; after the raise a muted proud and an attentive look at the
user; a failure a search, a droop with puppy eyes and a shake-off. Pauses are time-warp holds,
only as long as the arm does not already wait. Unknown phases are PRECISION and get no beat, and
every beat is fenced out of PRECISION, so no spatial offset ever lands on a precise sample
(tested on all three runs).

## Findings so far

- **At home the arm is folded beside the user.** With the assumed seating, the elbow is 7 cm from
  the user's left shoulder and the wrist is 25 cm from the head. That is why NEAR_USER is measured
  from the head: counting every link would mark the whole workspace NEAR_USER. The elbow and wrist
  are guarded by their own clearance floors instead.
- **From home, a glance at the user is about 18 deg.** The head starts 0.38 m from the user and may
  not get closer than 0.345 m.
- **RAMMP's caps make big head turns slow.** A 30 deg turn takes about 3 s, mostly because of the
  tool point's 0.16 m/s at 0.29 m from the neck. Quick puppy glances have to be small.
- **A MuJoCo distance glitch is worked around.** `mj_geomDistance` sometimes returns 0 for well
  separated mesh/box pairs. `Kin.geom_distance` detects this and bisects on its reliable "farther
  than D" answer.
- **Under the assumed 3 rad/s^2 cap, fast motions are tiny.** At 2 Hz a roll can only be about a
  degree (about a millimetre at the eyes), so `wiggle` and `shake_off` show as a 3-5 mm sideways
  shimmy, invisible from the wide camera. `bounce` is 8 mm hops, `sniff` 8 mm pecks, `nod` 5 deg
  dips. The slow, big poses read clearly: gaze, tilt, perk, droop, doze, wake, proud, search.
- **Puppy Mode costs time.** The holds take the real run from 15 s to 34 s (the failure run from
  20 s to 40 s): eager 3.8 s, curious 5 s, lock-on 1.6 s plus a 0.55 s slow-in, proud and
  attentive 6.2 s. Every beat is paced to RAMMP's caps, so each move takes about a second.
- **At the counter the face is turned on its side.** A look at the user from there is a slow pan
  (the user is behind it): about 2 s for 20 deg while holding, where the pan is also what keeps
  the container level.
- **The real runs pass the wrist singularity** (joint 6 near 0) during transits. There a head turn
  needs the shoulder, so `lead_gaze` eases out around it.
- **At home the user's head is below the face**, so puppy eyes from home are a sheepish glance
  toward them rather than up at them, and a droop sags down and away from them: a straight sag
  would bring the folded elbow within the 5 mm person slack.

## Extending it

- **A clip** is a module in `puppy_sim/clips/` exposing `CLIP = Clip(name, title, subtitle,
  camera, build)`. `build()` returns one Track, or a `(plain, puppy)` pair for side by side.
- **A behavior** is a function `(ctx, ..., t_on, ..., beat=None) -> layer.Contribution` on a
  `behaviors.Context`. Pace it with `behaviors.paced_move` / `paced_bump`, limit an oscillation
  with `fit_scale`, keep a static pose clear with `safe_scale`. If the task must pause for it, add
  a `timeline.Hold` (or a `Slow` with `hold_s`); to pick hold points, use `timeline.quiet_time`
  and `hold_ramp_s`, and `timeline.output_time` / `still_span` to place a beat inside one.
- **A director beat** is a row in `director.RULES` plus a function in `director.BEATS`;
  `director.direct(run)` returns `(plain, puppy)` for a side-by-side clip.
- **Scripted functional moves** come from `timeline.joint_move`, `kin.cartesian_path`,
  `timeline.concat` and `runs.mark_near_user`.

Code: `scene.py` (model), `kin.py` (FK/IK/look-at/gravity turns/clearance), `runs.py` (real runs,
phases, zones), `curves.py`, `timeline.py` (Track, time-warp, scripted moves), `layer.py`
(composition, gains, caps), `behaviors.py` (the vocabulary), `director.py` (task events to beats),
`checks.py`, `render.py`, `clips/`, `__main__.py`, and `tests/`.

The menagerie models in `third_party/` are BSD-licensed; see the LICENSE file in each folder.
