# Box opening with Puppy Mode: attended test on flatbot

Puppy Mode is experimental. Here it plays at most one small beat around one box-opening run: a
proud beat after a success, under both presets. Nothing plays before or during the run, and nothing
after a failure: box_opening runs its motion in a subprocess, and Puppy Mode waits for it.
Nothing in RAMMP's repos, the box_opening repo, the sheppy manifest or the profiles changes.

## Standing rules

- Nobody in the chair. The beats are checked against the Flatbot chair scene, but the first
  test is with the chair empty.
- E-stop in hand, within reach, for every run, as for any box-opening run. Attended only.
- The box-opening run itself is your existing attended procedure (preflight, container placed,
  `open_box`). This page only changes which image and command start the box_opening node.
- The Puppy Playground must not be running at the same time (both would command the arm).
  Stop its `puppy` container first.
- Start with `preset:=subtle`, then `demo`. If a beat looks wrong, hit the e-stop, then turn
  Puppy Mode off (below).

## What is on snorlax

- Image `box_opening:v2-105f74e-puppy`: the stock `box_opening:v2-105f74e` plus `puppy_motion`
  and `puppy_mode_ros` (nothing else changed; same user, entrypoint and workdir).
- Rebuild, if the repo changes (from a checkout of branch `feature/puppy-mode-build`, repo root;
  `git archive` avoids Windows line endings):

  ```
  DOCKER_BUILDKIT=0 nice -n 19 docker build --cpuset-cpus 8-11 \
      -f puppy_mode_ros/robot/Dockerfile.box_opening -t box_opening:v2-105f74e-puppy .
  ```
- Isolated mock run (own container, ROS domain 77, nothing sent), to check the image any time:

  ```
  puppy_mode_ros/robot/mock_box_opening.sh ~/puppy-data/flatbot.urdf
  ```

  It runs the goal twice, Puppy Mode on and off, and ends with `ALL PASS` (exit status 0).

## 1. Stop sheppy's box_opening node

In the `sheppy` TUI, stop the `box_opening` node (alternative `box_opening_v2`, container
`sheppy-box-opening`) and leave the rest of the profile up (arm, planner, cameras, perception).
The `open_box` node (the goal sender) stays as it is. Check with `docker ps` that
`sheppy-box-opening` is gone before starting the next step; two box_opening nodes on one domain
would both serve `/box_opening/execute_action`.

## 2. Run box_opening with Puppy Mode

The same `docker run` as the manifest's `box_opening_v2`, with the `-puppy` image, the launcher as
the command and the node's params as ROS args. Run from `~/rammp-deployments/december_2026` (the
volumes are relative to it). First run, `subtle`: the proud beat after a success.

```
docker run --rm --name puppy-box-opening --network host --ipc host --user 1000:1000 \
  -e CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml -e ROS_DOMAIN_ID=0 -e HOME=/tmp \
  -v /home/snorlax/.ros/box_opening:/state \
  -v "$PWD/config/dds/cyclonedds.xml":/etc/cyclonedds/config.xml:ro \
  -v "$PWD/config/robots/current":/config/robot:ro \
  box_opening:v2-105f74e-puppy \
  python3 -m puppy_mode_ros.launch box_opening.node:Capability box_opening.node:main \
  --ros-args -p use_orchestrator:=false -p allow_execute:=true \
  -p container:=ankou_pink.yaml -p world:=world_flatbot.yaml -p state_dir:=/state \
  -p video:=true -p viz:=false -p mission_frames:=true -p speed_scale:=1.0 \
  -p puppy.enabled:=true -p puppy.person_zone:=flatbot -p puppy.preset:=subtle
```

It is foreground (Ctrl+C stops it cleanly; STOPSIGNAL is SIGINT). The log shows the node up and,
if Puppy Mode could not engage for a goal (logged when the goal starts), a line `puppy mode off: <reason>` (the run then goes on without
it). Then send the goal as usual (the `open_box` sheppy node, or your own `ros2 action send_goal`).

Second run: the same command with `-p puppy.preset:=demo` (the same single proud after a success).
Only after the first run looked right.

## 3. What to watch

- Nothing before the run starts (box_opening never moves through the arm interface).
- Nothing during the run (locate, look, press, grip, carry, release, home). Any arm motion that
  is not box opening's own is a failure: e-stop.
- After a success: one proud beat at the end pose. It is tiny at the Flatbot rest pose (a few
  millimetres): an almost invisible proud is not a hook failure. After a failure: nothing.
- The log also shows every beat sent (`puppy: sending <name> <s>, zone, mode/gains, start offset`),
  the director's decisions and drops (`puppy: director: ...`, for example `nothing (holding
  something near the person)`) and, at the start of each goal, `puppy mode on: <capability>, preset
  ...`. No `sending` line after a success means the director chose nothing or a gate skipped it:
  the lines say which. A gripper not fully open (> 0.05) at HOME counts as holding something near
  the person: no proud. A goal that moved nothing and did not report `executed` (a dry run) gets no
  success beats; the log says so. Before an ending beat the arm must settle (up to 1 s), else it is
  skipped and logged.
- Log lines starting `puppy:` say what was skipped and why (for example `skip <beat>: ...`);
  `puppy: ...; no more beats this action` means Puppy Mode switched itself off for that run and
  the run was unaffected.

## 4. Turn Puppy Mode off

- Live, without a restart: `ros2 param set /box_opening puppy.enabled false` (read at every goal).
- Or at start: `-p puppy.enabled:=false` in the command above. The hook then runs the original
  `execute` unchanged (the mock check shows zero beats and the same result).
- Or run the stock image with its own command, as sheppy does: restart `box_opening_v2` in the
  `sheppy` TUI.

## 5. Restore sheppy

Stop the `puppy-box-opening` container (Ctrl+C), confirm with `docker ps` that it is gone, then
restart the `box_opening` node in the `sheppy` TUI (the profile converges back to the manifest).
