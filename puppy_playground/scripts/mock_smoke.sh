#!/usr/bin/env bash
# Mock smoke test. Starts ITS OWN container in mock mode and curls every endpoint of it.
#
#   scripts/mock_smoke.sh <flatbot.urdf>
#
# <flatbot.urdf> is the arm driver's URDF (BRINGUP step 2 saves it with scripts/dump_urdf.py).
# Env: IMAGE (puppy-playground), PORT (8766), NAME (puppy-playground-mock).
#
# Isolation, so nothing here can reach the live stack or the arm:
#   - mock mode: capability_engine's arm interface creates no ROS entity and sends no goal; a
#     motion only logs "<label>: mock: not sent" and becomes the mock state (arm.py, MOCK_*).
#   - its own DDS world: ROS_DOMAIN_ID=77 and ROS_LOCALHOST_ONLY=1, no --ipc host.
#   - its own UI port, which must be free; it never talks to anything but the container it started.
#   - its own temp data dir (an approved stage_path.json fixture made by stage_store.save).
# Exit status is the number of failed checks (non-zero = failure), or 99 for a setup failure.
set -euo pipefail

URDF="${1:-}"
IMAGE="${IMAGE:-puppy-playground}"
PORT="${PORT:-8766}"
NAME="${NAME:-puppy-playground-mock}"
case "$NAME" in puppy-playground-mock*) ;; *) echo "SETUP FAIL: NAME must start with puppy-playground-mock (got $NAME)" >&2; exit 99 ;; esac
BASE="http://127.0.0.1:$PORT"
FAILS=0
TMP=""
STARTED=0  # set only after our own successful docker run: the trap removes nothing else

die() { echo "SETUP FAIL: $*" >&2; exit 99; }

cleanup() {
  [ "$STARTED" = 1 ] && { timeout 60 docker rm -f "$NAME" >/dev/null 2>&1 || true; }
  if [ -n "$TMP" ] && [ -d "$TMP" ]; then
    # the container wrote as root: remove through a container
    timeout 120 docker run --rm -v "$TMP":/d --entrypoint sh "$IMAGE" -c 'rm -rf /d/* /d/.[!.]*' >/dev/null 2>&1 || true
    rm -rf "$TMP" 2>/dev/null || true
  fi
}
trap cleanup EXIT

[ -n "$URDF" ] && [ -f "$URDF" ] || die "usage: $0 <flatbot.urdf> (a readable URDF file)"
command -v docker >/dev/null || die "docker not found"
command -v curl >/dev/null || die "curl not found"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found: build it first (BRINGUP step 1)"
if grep -qx "$NAME" <<<"$(docker ps -a --format '{{.Names}}')"; then
  die "a container named $NAME already exists: remove it with 'docker rm -f $NAME'"
fi
if curl -s -o /dev/null --max-time 2 "$BASE/status"; then
  die "something already answers on port $PORT: refusing to send anything to it"
fi

TMP="$(mktemp -d)"
cp "$URDF" "$TMP/flatbot.urdf"
echo "== making the approved stage-path fixture"
timeout 600 docker run --rm -v "$TMP":/data "$IMAGE" python3 /module_ws/src/puppy_playground/scripts/make_mock_fixture.py \
  /data/flatbot.urdf /data || die "could not make the stage-path fixture"
[ -f "$TMP/stage_path.json" ] || die "no stage_path.json was written"

REST_Q='[0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725]'
echo "== starting $NAME (mock, ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1, port $PORT)"
docker run -d --name "$NAME" --network host -v "$TMP":/data \
  -e ROS_DOMAIN_ID=77 -e ROS_LOCALHOST_ONLY=1 \
  "$IMAGE" ros2 launch puppy_playground playground.launch.py use_orchestrator:=false \
  mock:=true mock_urdf:=/data/flatbot.urdf "mock_joints:=$REST_Q" ui_port:="$PORT" >/dev/null \
  || die "docker run failed"
STARTED=1

# Only talk to it once it is shown to be the container we started.
ENV_DUMP="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME")"
grep -qx 'ROS_DOMAIN_ID=77' <<<"$ENV_DUMP" || die "$NAME is not on ROS_DOMAIN_ID=77"
grep -qx 'ROS_LOCALHOST_ONLY=1' <<<"$ENV_DUMP" || die "$NAME is not ROS_LOCALHOST_ONLY"
[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || die "$NAME is not running"

logs() { docker logs "$NAME" 2>&1; }
has_log() { local l; l="$(logs)"; grep -q -- "$1" <<<"$l"; } # no pipe: pipefail + grep -q would SIGPIPE docker logs
# Ready = the engine logged "up (standalone, mock)" (it builds its model first, so it is the slow one),
# the UI logged its port, and the UI has heard the engine's status ("state": "idle", so its action
# server is discovered). Nothing arm-derived: the mock arm publishes no joint_states.
ready=0
for _ in $(seq 1 120); do
  if has_log "up (standalone, mock)" && has_log "playground UI on port $PORT"      && grep -q '"state": "idle"' <<<"$(curl -s --max-time 2 "$BASE/status" || true)"; then ready=1; break; fi
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || { logs | tail -30; die "$NAME exited"; }
  sleep 1
done
[ "$ready" = 1 ] || { logs | tail -30; die "engine and UI were not both up and connected in 120 s"; }
echo "== $NAME is up and is in mock mode"

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }

post_code() { # path [json]
  curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST -H 'Content-Type: application/json' \
    -d "${2:-{\}}" "$BASE$1" || echo 000
}
check() { # name, wanted codes, method, path, [json]
  local name="$1" want="$2" method="$3" path="$4" body="${5:-}" code
  if [ "$method" = GET ]; then
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$BASE$path" || echo 000)"
  else
    code="$(post_code "$path" "$body")"
  fi
  case " $want " in
    *" $code "*) pass "$name  ($method $path -> $code)" ;;
    *) fail "$name  ($method $path -> $code, wanted $want)" ;;
  esac
}
idle() { # wait for the goal to end (up to 120 s: the first path check takes 5-20 s)
  for _ in $(seq 1 240); do
    grep -q '"busy": false' <<<"$(curl -s --max-time 5 "$BASE/status")" && return 0
    sleep 0.5
  done
  return 1
}
mock_goals() { logs | grep -c 'execute_joint_trajectory: mock: not sent' || true; }
accepted_and_built() { # name, path, json: 200, finishes with no refusal, and a goal reached the mock arm
  local name="$1" path="$2" body="${3:-}" before after
  before="$(mock_goals)"
  check "$name" "200" POST "$path" "$body"
  if ! idle; then fail "$name did not finish in 120 s"; return; fi
  if grep -q '"last_refusal": null' <<<"$(curl -s --max-time 5 "$BASE/status")"; then pass "$name: no refusal"
  else fail "$name was refused: $(curl -s --max-time 5 "$BASE/status")"; fi
  after="$(mock_goals)"
  if [ "$after" -gt "$before" ]; then pass "$name: a driver goal was built and handed to the mock arm ($before -> $after)"
  else fail "$name: no goal reached the mock arm"; fi
}

check "page"      "200" GET /
check "status"    "200" GET /status
check "unknown path" "404" GET /nope
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST -d 'name=nod' "$BASE/express" || echo 000)"
[ "$code" = 415 ] && pass "form POST refused (POST /express -> 415)" || fail "form POST refused (-> $code, wanted 415)"
check "express (bad name)"  "400" POST /express '{"name":"no_such_thing"}'
check "combo (bad)"         "400" POST /combo   '{"names":[]}'

# the mock arm starts at the rest pose with an approved path: these build real goals
accepted_and_built "stage"   /stage
accepted_and_built "express" /express '{"name":"head_tilt","size":0.25}'
accepted_and_built "combo"   /combo   '{"names":["nod","wag"],"size":0.25}'
accepted_and_built "rest"    /rest
check "recover (at rest)"    "200" POST /recover; idle || fail "recover did not finish"
check "plan_stage"           "200" POST /plan_stage; idle || fail "plan_stage did not finish"
check "stop"                 "200 503" POST /stop
check "auto on"              "200" POST /auto '{"on":true,"size":0.25}'
check "auto off"             "200" POST /auto '{"on":false}'
check "stop (again)"         "200 503" POST /stop
check "status (last)"        "200" GET /status

echo "--- /status:"; curl -s --max-time 10 "$BASE/status"; echo
echo "--- mock goals handed to the mock arm: $(mock_goals)"
echo "--- request log lines: $(wc -l < "$TMP/playground.jsonl" 2>/dev/null || echo 0)"
if [ "$FAILS" -eq 0 ]; then echo "ALL PASS"; else echo "$FAILS FAILED"; fi
exit "$FAILS"
