# Puppy Playground bring-up checklist

Every step from step 4 on moves the real arm. Do them in order, and stop at the first surprise.

**Where to stand:** on the chair's right side, by the joystick, with a hand on the e-stop for every
motion step. The page's **Stop** button is a convenience, **not a safety device**. The e-stop is.

Replace `<snorlax>` with the robot computer's address. `~/claude-scratch/puppy-playground` is the
checkout the agent builds in.

## 1. Build the image (on snorlax)

```
cd ~/claude-scratch/puppy-playground
nice docker build -f puppy_playground/Dockerfile -t puppy-playground .
```

Build context is the repo root. The build does not touch the sheppy containers. arm64 only.

Data directory for the stage path files and the JSONL log:

```
mkdir -p ~/puppy-data
```

## 2. Mock run (nothing moves)

The `flatbot` profile is up (`sheppy up flatbot`). Start the container with `mock:=true`: the arm
interface logs "mock: not sent" and sends no goal.

```
docker run --rm --network host -v ~/puppy-data:/data --name puppy-mock \
    puppy-playground mock:=true
```

(Add `mock_urdf:=/data/flatbot.urdf` after putting a URDF in `~/puppy-data` if you want the actions to
get past "no model"; without it they refuse, which also passes.) In a second shell:

```
cd ~/claude-scratch/puppy-playground
puppy_playground/scripts/mock_smoke.sh localhost 8080
```

Expect `ALL PASS`: every endpoint returns 200 (or a reasoned refusal such as 409 busy), the log
`~/puppy-data/playground.jsonl` has one JSON line per request that reached the engine, and the arm did
not move. Stop with Ctrl-C.

## 3. Plan the stage path (no motion)

`plan_stage` needs the chair-only scene, so switch profiles first:

```
sheppy up flatbot_empty_world      # rammp-deployments PR #150 branch
docker run --rm --network host -v ~/puppy-data:/data --name puppy puppy-playground
```

The arm must be at rest. In the page (`http://<snorlax>:8080/`) press **Plan stage**, or:

```
curl -s -X POST -H 'Content-Type: application/json' -d '{}' http://<snorlax>:8080/plan_stage
curl -s http://<snorlax>:8080/status
```

It writes `~/puppy-data/stage_path.candidate.json` and never moves the arm. Stop the container
(Ctrl-C), then switch back:

```
sheppy up flatbot
```

## 4. Review the candidate in the simulator, then approve it

On your PC, copy the candidate next to the repo and inspect it:

```
scp <snorlax>:puppy-data/stage_path.candidate.json C:/Users/CHC785/orca/projects/RAMMP-realism/spike/media/stage_path.candidate.json
cd C:/Users/CHC785/orca/projects/RAMMP-realism/spike
.venv/Scripts/python -m puppy_sim render playground_stage
```

Note: the `playground_stage` clip plays the repo's own reference path (`escape_path`, cached), not the
candidate file itself. Check the file's `"method"` (`curobo`, `min_jerk` or `escape`) and that it is
the one you expect; the engine has already refused any path that breaks the stage-path rules, and
re-checks the approved file again before every Stage and Rest. If the candidate's method is not
`escape`, the clip is only a stand-in: look at the candidate's first and last samples against the
clip and ask before approving.

To approve, on snorlax:

```
cp ~/puppy-data/stage_path.candidate.json ~/puppy-data/stage_path.json
```

## 5. Attended motion: `sheppy up flatbot` and the container running

```
docker run --rm --network host -v ~/puppy-data:/data --name puppy puppy-playground
```

Stand on the chair's right side, by the joystick, hand on the e-stop. Open `http://<snorlax>:8080/`.
Keep the size at 0.25 until step 6.

1. **Stage.** This is the first and slowest motion: watch it whole. (Stage has no size.) Press it
   only with the arm at rest. If the driver rejects the goal, stop here and read `last_refusal` in
   `/status`; that is the first real check of the driver accepting the goal under `flatbot`.
2. **Each expression once at size 0.25**, one at a time, watching the whole beat. Watch **wag**,
   **lock_on** and **bounce** most closely: the driver model predicts up to 31 mm of overshoot on these
   without acceleration feedforward. Stop at any beat that looks wrong.
3. **Rest.** The arm goes back along the approved path in reverse.

Continuous joints read in (-pi, pi] from the driver. The engine compares and sends them modulo whole
turns, so a joint that reads wrapped is not a fault.

## 6. Full size

Repeat the expressions at size 1.0 (the size slider), the same way: one each, hand on the e-stop,
wag, lock_on and bounce first. Then **Rest** if the arm is still at stage. If the arm is at rest,
press Stage again first.

## 7. Auto

With the arm at the stage pose, turn **Auto** on at size 0.25. Expect it to wait for discovery, then 3 s
of quiet and a still arm, before the first beat. Watch several beats.

Then start a real task (for example from the orchestrator) while Auto is on. Confirm Auto yields
within about half a second: the arm finishes nothing new and the page shows `waiting_for_task`. When
the task is done and everything has been idle and still for 3 s, Auto resumes by itself.

## 8. Rest

Turn Auto off, press **Rest**, confirm the arm is back at rest, then stop the container (Ctrl-C).

## If something goes wrong

- Press the e-stop first. Then **Stop** on the page, which also turns Auto off.
- Arm left off the path: press **Recover** (near the stage pose, or within 0.02 rad of a path sample).
  Anywhere else it refuses and you recover by hand.
- Log of every request and motion: `~/puppy-data/playground.jsonl`.
