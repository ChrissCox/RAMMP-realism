# puppy_mode_ros

Puppy Mode on a capability engine: small expressive arm motions ("beats") at the boundaries of a
task. It is experimental and off by default. RAMMP's `capability_engine` is not patched: a launcher
wraps the capability class's `execute`, and the wrapper does nothing unless `puppy.enabled` is true.

## Enabling it

Run the engine through the launcher instead of its own entry point:

```
python3 -m puppy_mode_ros.launch box_opening.node:Capability box_opening.node:main \
    --ros-args -p puppy.enabled:=true -p puppy.person_zone:=flatbot -p puppy.preset:=subtle
```

`Capability` is the `CapabilityEngine` subclass in box_opening's `node.py` (its `main()` is
`run(Capability)`). The launcher installs the hook on that class, then calls the
engine's own `main` with the remaining arguments. With `puppy.enabled` false (the default) every goal
runs the original `execute` unchanged.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `puppy.enabled` | `false` | Only the bool `true` turns it on. |
| `puppy.preset` | `subtle` | `subtle` (budget 2.5 s a task, only the essential beats: proud and sorry) or `demo` (6 s, all rules). |
| `puppy.person_zone` | `""` | Where the person is. Empty: Puppy Mode stays off, logging `puppy mode off: no person zone`. |
| `puppy.lock_on` | `view` | `view` or `standoff`; the stare style for the lock-on beat. |

A bad value never fails the task: Puppy Mode logs `puppy mode off: <reason>` and runs the original
action. This holds for an unreadable person zone, an unknown preset, a robot description that cannot
be read, and a capability with no rules (`<name> has no rules`). The capability is looked up by the
engine's name, then by the root module of its class.

### `puppy.person_zone`

- `flatbot`: the PR #150 chair scene (`puppy_motion.flatbot.chair_scene()`): person and obstacle
  layers of the Flatbot chair setup.
- A JSON object: `{"person": {...}, "obstacles": {...}}`, each with optional `spheres`
  (`[x, y, z, r]`), `capsules` (`[ax, ay, az, bx, by, bz, r]`) and `boxes` (`[cx, cy, cz, hx, hy, hz]`),
  in the arm's base frame, metres. `person` needs at least one shape; its head is its highest sphere, else the centre of its highest box.

## What plays, and when

Beats play only at the task's boundaries. The `TaskRequested` beats are held until the task's first
move through `engine.arm` and play just before it (its own checks have passed by then); a task that
never moves through `engine.arm` never plays them, and a goal the task refuses before moving moves
nothing. `Succeeded` beats play after `execute` returns, and on a recoverable `EngineError` the
`Failed` beats play, then the same error is re-raised. Any other exception, a cancel, an e-stop and
a non-recoverable failure play nothing. A beat never changes the task's payload, and the ending
beats never change its result: a cancel, e-stop or takeover that meets one is logged and the task's
own success or error stands. During the task `engine.arm` is a proxy
and `ctx` a wrapper, both restored in a `finally`.

For box_opening the motion runs in a subprocess, so its phases (locate, look, press, grip, carry,
release, home) are all PRECISION and never trigger a beat, and it never moves through `engine.arm`, so
nothing plays before or during the run. After a success exactly one proud plays, under both presets
(it is expected to be tiny at the Flatbot rest pose). A failure plays nothing.

## Safety and the attended-hardware rules

Every beat is checked against the person and obstacle geometry with the same caps preset the director
composed it under, and is sent only if all of these hold at that moment: no task move or stream is in
flight, the joint state is fresh (0.1 s at the send), the arm is still (< 0.02 rad/s) and within
0.002 rad of where the beat starts. Beats are sent with queue preemption over the arm joints only,
with the task's own control mode and gains (box_opening: impedance "default").

Fail-safes: if the gripper state is missing, stale, not present or NaN the container counts as held
and beats near the person are muted; a failed task move, a refused or aborted beat, an unreadable
state, or no trajectory action server within 2 s disables Puppy Mode for the rest of that action,
and the task goes on. An e-stop, a takeover or a cancel meeting a beat before the task or during it
ends the action as it would without Puppy Mode; one meeting an ending beat leaves the task's result.

Rules for running it on the real arm:

- Attended only: someone at the e-stop, within reach, for every run.
- Start with `puppy.preset:=subtle` and the `flatbot` person zone.
- Never connect to or change a robot computer from CI or an agent session; the engine is run by the
  person at the robot.
- Do not run it with the person inside the arm's reach unless the person zone is set, and keep the
  zone current: the checks are only as good as the scene.
- To stop it, restart the engine without `puppy.enabled` (or set it to false); the next goal runs
  the original action.

Task 14 holds the attended checklist.

## Tests

```
python3 -m pytest puppy_mode_ros/test -q
```

They run in the `puppy-mode-ros` workflow against a fake driver, in the same image as the other ROS
tests.
