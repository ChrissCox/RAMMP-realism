# Puppy Playground bring-up checklist

**The first step that can move the arm is step 6.** Steps 1 to 5 move nothing: step 4 starts the
real engine but with `plan_only:=true`, so every action except Plan stage refuses.

**Standing rules for every attended step (4 and 6 to 9):** nobody in the chair, you on the chair's
right side by the joystick, a hand on the e-stop. The page's **Stop** button is a convenience, **not a
safety device**. The e-stop is. The page has no login and listens on every address of snorlax, so
anyone on the robot's network can press its buttons: keep it off any open network.

Replace `<snorlax>` with the robot computer's address. `~/claude-scratch/puppy-playground` is the
checkout the agent builds in. Containers run as root, so files they write in `~/puppy-data` are
root-owned (reading and `cp` as yourself still work).

## 1. Build the image (on snorlax)

```
cd ~/claude-scratch/puppy-playground
nice docker build -f puppy_playground/Dockerfile -t puppy-playground .
mkdir -p ~/puppy-data
```

Build context is the repo root. The image's entrypoint script replaces the base image's. Compare, and
tell the agent if the base sets anything the script does not (environment exports, other setup):

```
docker image inspect ghcr.io/rammp-org/rammp-base:nightly-20261030 --format '{{json .Config.Entrypoint}} {{json .Config.Env}}'
docker image inspect puppy-playground --format '{{json .Config.Entrypoint}}'
```

 The build does not touch the sheppy containers. arm64 only.

## 2. Copy the live stack's ROS settings and save the arm's URDF (nothing moves)

The `flatbot` profile is up (`sheppy up flatbot`). The new container must speak the same DDS as the
sheppy containers, so copy their settings from one of them:

```
docker ps --format '{{.Names}}'                      # pick one sheppy container, call it <name>
docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' <name> \
    | grep -E '^(ROS_|RMW_|CYCLONEDDS|FASTRTPS)' > ~/puppy-env.list
cat ~/puppy-env.list
```

Check the list has `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION` if the sheppy container sets them. On snorlax (checked
2026-11-06) the sheppy containers use `ROS_DOMAIN_ID=0`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and
`CYCLONEDDS_URI=file:///etc/cyclonedds/config.xml`, mounted from
`~/rammp-deployments/december_2026/config/dds/cyclonedds.xml`; the commands below mount that file at
the same in-container path. If `docker inspect <name> --format '{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}'`
shows a different source, use it. The real containers below also use `--ipc host`
(Fast DDS shared memory).

Save the arm's URDF (a read-only subscription to `/arm/robot_description`) for the mock run:

```
mkdir -p ~/puppy-data
docker run --rm --network host --ipc host --env-file ~/puppy-env.list \
    -v ~/rammp-deployments/december_2026/config/dds/cyclonedds.xml:/etc/cyclonedds/config.xml:ro puppy-playground \
    python3 /module_ws/src/puppy_playground/scripts/dump_urdf.py > ~/puppy-data/flatbot.urdf
head -c 200 ~/puppy-data/flatbot.urdf
```

It must start with `<?xml` or `<robot`. If it fails with "no /arm/robot_description", the settings
in `~/puppy-env.list` are wrong.

## 3. Mock run (nothing moves)

```
cd ~/claude-scratch/puppy-playground
puppy_playground/scripts/mock_smoke.sh ~/puppy-data/flatbot.urdf
```

The script starts its own container, `puppy-playground-mock`, and removes it when it ends. It is
isolated from the live stack: `ROS_DOMAIN_ID=77` with `ROS_LOCALHOST_ONLY=1` (the container's engine
and UI nodes, with their action server, services and topics, exist only there), port 8766, a temp
data dir holding an approved `stage_path.json` it makes from your URDF, and no `--ipc host`. It
refuses to run if anything already answers on port 8766 or the container name exists, and only sends
requests once it has checked the container it started. In mock mode (capability_engine `arm.py`
`MOCK_*`) the arm interface makes no ROS entity at all: a motion logs `execute_joint_trajectory: mock:
not sent`, succeeds and becomes the mock state, which starts at the rest pose here. The script
prints one PASS/FAIL line per endpoint and shows, from the container's log, that Stage, an
expression, a combo and Rest each sent a checked trajectory to the arm interface's
`execute_joint_trajectory`.

Expect `ALL PASS` and exit status 0. This shows each action reached `execute_joint_trajectory` with a
checked trajectory; mock mode returns there, before any ROS goal is built. Building the goal is covered
by the CI FakeDriver tests, and the driver accepting a real goal is only checked at step 6.1. A failure prints the reason; do not go on until it passes.

## 4. Plan the stage path (no motion; `plan_only`)

`plan_stage` needs the chair-only scene, so switch profiles first, and only then start the real
container with `plan_only:=true`:

```
cd ~/rammp-deployments/december_2026   # sheppy reads ./sheppy-manifest.yaml
sheppy up flatbot_empty_world          # the chair-only scene (rammp-deployments PR #150)
docker run --rm --network host --ipc host --env-file ~/puppy-env.list -v ~/puppy-data:/data \
    -v ~/rammp-deployments/december_2026/config/dds/cyclonedds.xml:/etc/cyclonedds/config.xml:ro \
    --name puppy puppy-playground \
    ros2 launch puppy_playground playground.launch.py use_orchestrator:=false plan_only:=true
```

Attended: hand on the e-stop, nobody in the chair. Park the arm anywhere still and clear (for example
Kinova Home); the plan starts from wherever it is, and Stage and Rest later use that spot as "rest".
If Plan stage refuses `bad_start`, the arm is outside a joint limit or too close to an obstacle: move
it by hand and try again. Open `http://<snorlax>:8080/` and
press **only "Plan stage (no motion)"**, the button set apart from the motion buttons. Press nothing
else: with `plan_only` the engine refuses every other action, but do not rely on that. The line beside
the button says "candidate written" or shows why none was (`no_candidate` and its message). (Or, on
snorlax: `curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://127.0.0.1:8080/plan_stage`,
then `curl -s http://127.0.0.1:8080/status`.)

It writes `~/puppy-data/stage_path.candidate.json` and never moves the arm. Stop the container
(Ctrl-C), then switch back:

```
sheppy up flatbot
```

## 5. Review the candidate in the simulator, then approve it

On your PC, copy the candidate from snorlax (the data dir is `~/puppy-data`) and render exactly that
file. Git Bash:

```
scp snorlax@10.5.127.121:puppy-data/stage_path.candidate.json C:/Users/CHC785/orca/projects/RAMMP-realism/spike/media/stage_path.candidate.json
cd C:/Users/CHC785/orca/projects/RAMMP-realism/spike
PLAYGROUND_CANDIDATE=media/stage_path.candidate.json .venv/Scripts/python -m puppy_sim render playground_stage_candidate
```

PowerShell (same scp and cd):

```
$env:PLAYGROUND_CANDIDATE="media/stage_path.candidate.json"; .venv/Scripts/python -m puppy_sim render playground_stage_candidate
```

The candidate renders to `media/playground_stage_candidate.*`; the committed `playground_stage` clip is
not overwritten.

The clip plays the candidate file's own points, forward to the stage pose and back, on flatbot's chair
boxes, and re-checks it with the engine's rules (chair scene grown by the file's margin). Approve ONLY
if the title reads "Stage path candidate <12 hex>: <method>, <seconds> s - REVIEW FOR APPROVAL" and no
violations are listed. The 12 hex digits are the first 12 of the candidate file's sha256: before the `cp`
below, run `sha256sum ~/puppy-data/stage_path.candidate.json` on snorlax and check that it starts with
the same 12 digits. If it does not, the file you reviewed is not the file you would approve. A title with "NOT APPROVABLE" (violations, or a file that cannot be read) means
do not approve. Without `PLAYGROUND_CANDIDATE` the `playground_stage` clip shows the simulated plan, labelled "not the
robot's candidate", which is not what the arm will replay.

To approve, on snorlax:

```
cp ~/puppy-data/stage_path.candidate.json ~/puppy-data/stage_path.json
```

## 6. Attended motion: `sheppy up flatbot`, the real container, no `plan_only`

```
docker run --rm --network host --ipc host --env-file ~/puppy-env.list -v ~/puppy-data:/data \
    -v ~/rammp-deployments/december_2026/config/dds/cyclonedds.xml:/etc/cyclonedds/config.xml:ro \
    --name puppy puppy-playground \
    ros2 launch puppy_playground playground.launch.py use_orchestrator:=false
```

Nobody in the chair, you on the chair's right side by the joystick, hand on the e-stop. Open
`http://<snorlax>:8080/`. Keep the size at 0.25 until step 7.

1. **Stage.** This is the first and slowest motion: watch it whole. (Stage has no size.) Press it
   only with the arm where you ran Plan stage (within 0.02 rad of the approved path's start). If the driver rejects the goal, stop here and read `last_refusal` in
   `/status`; this is the first real check of the driver accepting the goal under `flatbot`.
2. **Each expression once at size 0.25**, one at a time, watching the whole beat. Watch **wag**,
   **lock_on** and **bounce** most closely: the driver model predicts up to 31 mm of overshoot on these
   without acceleration feedforward. Stop at any beat that looks wrong.
3. **Rest (back to where Stage started).** The arm goes back along the approved path in reverse. If Rest refuses `not_at_end`
   (the arm sagged off the stage pose), press **Recover**, then Rest.

Continuous joints read in (-pi, pi] from the driver. The engine compares and sends them modulo whole
turns, so a joint that reads wrapped is not a fault.

## 7. Full size

Repeat the expressions at size 1.0 (the size slider), the same way: one each, hand on the e-stop,
wag, lock_on and bounce first. Then **Rest** if the arm is still at stage (if it refuses `not_at_end`, press Recover, then Rest). If the arm is back at
its start, press Stage again first.

## 8. Auto

With the arm at the stage pose, set the size slider to 0.25, then turn **Auto** on. Expect it to wait for discovery, then 3 s
of quiet and a still arm, before the first beat. Watch several beats.

Then start a real task (for example from the orchestrator) while Auto is on. Confirm Auto yields
within about half a second: the arm finishes nothing new and the page shows `waiting_for_task`. When
the task is done and everything has been idle and still for 3 s, Auto resumes by itself only if the
task left the arm within 0.02 rad of the stage pose or of the stored rest. Otherwise (a task moved the
arm elsewhere, or another controller took it) Auto turns itself off and the page says why
(`drifted`: press Recover; `control_taken`: another controller has the arm). An e-stop also turns Auto
off: release it and turn Auto on again.

## 9. Rest

Turn Auto off, press **Rest**, confirm the arm is back where Stage started, then stop the container (Ctrl-C).
If Rest refuses `not_at_end` after the arm sagged, press **Recover**, then Rest.

## If something goes wrong

- Press the e-stop first. Then **Stop** on the page, which also turns Auto off.
- Arm left off the path: press **Recover** (near the stage pose, or within 0.02 rad of a path sample).
  Anywhere else it refuses and you recover by hand. Recover from the middle of the path can take
  minutes (it re-times the rest of the path from standstill): watch it, hand on the e-stop.
- The status line names the action, the error code and the message of every refusal. Hints: `drifted`
  press Recover; `contact` check the arm, then press Recover; `control_taken` another controller has
  the arm; `estopped` release the e-stop, then turn Auto on again.
- Log of every request and motion: `~/puppy-data/playground.jsonl`.
