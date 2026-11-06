#!/usr/bin/env bash
# Mock run of box_opening with Puppy Mode, on the robot computer. Starts ITS OWN container (twice,
# one after the other) from the -puppy image and sends one open_box goal to it from inside it.
#
#   mock_box_opening.sh <flatbot.urdf>
#
# Env: IMAGE (box_opening:v2-105f74e-puppy), NAME (puppy-box-opening-mock).
#
# Isolation, so nothing here can reach the live stack or the arm:
#   - mock mode: box_opening plays its phases and moves nothing; capability_engine's arm interface
#     creates no ROS entity and sends no goal (a motion only logs "<label>: mock: not sent").
#     allow_execute is false.
#   - its own DDS world: ROS_DOMAIN_ID=77 and ROS_LOCALHOST_ONLY=1, on docker's default bridge
#     network (no --network host, no --ipc host), so the live domain 0 cannot see it either way.
#   - the goal is sent with docker exec from inside the same container (same domain).
#   - it never touches a container it did not start: the trap removes only $NAME, and only after
#     its own successful docker run.
# Exit status is the number of failed checks (non-zero = failure), or 99 for a setup failure.
set -euo pipefail

URDF="${1:-}"
IMAGE="${IMAGE:-box_opening:v2-105f74e-puppy}"
NAME="${NAME:-puppy-box-opening-mock}"
case "$NAME" in puppy-box-opening-mock*) ;; *) echo "SETUP FAIL: NAME must start with puppy-box-opening-mock (got $NAME)" >&2; exit 99 ;; esac
# a pose that passes check_beat against the flatbot chair scene: the playground's tested park
REST_Q='[0.067, -1.8971, -3.137, -2.2691, -0.211, -0.6831, 1.725]'
FAILS=0
TMP=""
STARTED=0  # set only after our own successful docker run: the trap removes nothing else

die() { echo "SETUP FAIL: $*" >&2; exit 99; }

cleanup() {
  [ "$STARTED" = 1 ] && { timeout 60 docker rm -f "$NAME" >/dev/null 2>&1 || true; }
  [ -n "$TMP" ] && [ -d "$TMP" ] && rm -rf "$TMP" 2>/dev/null || true
}
trap cleanup EXIT

[ -n "$URDF" ] && [ -f "$URDF" ] || die "usage: $0 <flatbot.urdf> (a readable URDF file)"
command -v docker >/dev/null || die "docker not found"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found: build it first (BRINGUP step 1)"
if grep -qx "$NAME" <<<"$(docker ps -a --format '{{.Names}}')"; then
  die "a container named $NAME already exists: remove it with 'docker rm -f $NAME'"
fi

TMP="$(mktemp -d)"
chmod 777 "$TMP"  # the container runs as 1000:1000 like the manifest's box_opening_v2
cp "$URDF" "$TMP/flatbot.urdf"
chmod 644 "$TMP/flatbot.urdf"

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }

start() { # puppy.enabled (true|false)
  local enabled="$1"
  echo "== starting $NAME (mock, puppy.enabled=$enabled, ROS_DOMAIN_ID=77, ROS_LOCALHOST_ONLY=1)"
  # CYCLONEDDS_URI is emptied: the image's points at a file sheppy mounts, which a bridge-network
  # mock run does not have (the default config is right for one container).
  docker run -d --name "$NAME" --user 1000:1000 \
    -e ROS_DOMAIN_ID=77 -e ROS_LOCALHOST_ONLY=1 -e HOME=/tmp -e CYCLONEDDS_URI= \
    -v "$TMP":/data \
    "$IMAGE" python3 -m puppy_mode_ros.launch box_opening.node:Capability box_opening.node:main \
    --ros-args -p use_orchestrator:=false -p mock:=true -p allow_execute:=false \
    -p container:=ankou_pink.yaml -p world:=world_flatbot.yaml -p state_dir:=/data \
    -p video:=false -p viz:=false \
    -p puppy.enabled:="$enabled" -p puppy.person_zone:=flatbot -p puppy.preset:=demo \
    -p arm.mock_urdf:=/data/flatbot.urdf -p "arm.mock_joints:=$REST_Q" >/dev/null \
    || die "docker run failed"
  STARTED=1
  local env_dump
  env_dump="$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$NAME")"
  grep -qx 'ROS_DOMAIN_ID=77' <<<"$env_dump" || die "$NAME is not on ROS_DOMAIN_ID=77"
  grep -qx 'ROS_LOCALHOST_ONLY=1' <<<"$env_dump" || die "$NAME is not ROS_LOCALHOST_ONLY"
  [ "$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$NAME")" != host ] || die "$NAME is on the host network"
  for _ in $(seq 1 120); do
    has_log "box_opening up (standalone, mock)" && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = true ] || { logs | tail -30; die "$NAME exited"; }
    sleep 1
  done
  logs | tail -30; die "the engine was not up in 120 s"
}
logs() { docker logs "$NAME" 2>&1; }
has_log() { local l; l="$(logs)"; grep -q -- "$1" <<<"$l"; } # no pipe: pipefail + grep -q would SIGPIPE docker logs
stop() { timeout 60 docker rm -f "$NAME" >/dev/null 2>&1 || true; STARTED=0; }

send_goal() { # prints the client's output
  timeout 120 docker exec "$NAME" /ros_entrypoint.sh ros2 action send_goal --feedback /box_opening/execute_action \
    rammp_orchestrator_interfaces/action/ExecuteAction \
    "{engine_id: box_opening, action_id: open_box, params_json: '{}'}" 2>&1
}

# line numbers (in the container log) of the first/last progress line and of each beat that reached
# the mock arm. The mock logs one line per phase; its words are in MOCK_PHASES (node.py).
BEAT_PAT='execute_joint_trajectory: mock: not sent'
PHASE_PAT='found the container|looking for the button|pressing the button|holding the lid|setting the lid down|letting go of the lid|going home'

run_case() { # enabled
  local enabled="$1" out lg first last beats before after outside
  start "$enabled"
  out="$(send_goal || true)"
  sleep 2
  lg="$(logs)"
  echo "---- goal output ($enabled):"; echo "$out" | tail -12
  echo "---- node log ($enabled), beat/puppy/phase lines:"
  grep -nE "puppy|$BEAT_PAT|$PHASE_PAT" <<<"$lg" | head -60 || true
  if grep -q 'Goal finished with status: SUCCEEDED' <<<"$out" && grep -q '"mock": *true' <<<"$out"; then
    pass "puppy.enabled=$enabled: the goal SUCCEEDED with the mock payload"
  else
    fail "puppy.enabled=$enabled: the goal did not succeed with the mock payload"
  fi
  first="$(grep -nE "$PHASE_PAT" <<<"$lg" | head -1 | cut -d: -f1 || true)"
  last="$(grep -nE "$PHASE_PAT" <<<"$lg" | tail -1 | cut -d: -f1 || true)"
  beats="$(grep -c -- "$BEAT_PAT" <<<"$lg" || true)"
  if [ "$enabled" = true ]; then
    if grep -q 'puppy mode off' <<<"$lg"; then fail "the hook did not engage: $(grep 'puppy mode off' <<<"$lg" | head -1)"
    else pass "the hook engaged (no 'puppy mode off')"; fi
    if [ "$beats" -ge 1 ]; then pass "$beats beat trajectory(ies) reached the mock arm (mock: not sent)"
    else fail "no beat reached the mock arm"; fi
    if [ -n "$first" ] && [ -n "$last" ]; then
      outside="$(grep -n -- "$BEAT_PAT" <<<"$lg" | cut -d: -f1 | awk -v f="$first" -v l="$last" '$1 > f && $1 < l' | wc -l)"
      after="$(grep -n -- "$BEAT_PAT" <<<"$lg" | cut -d: -f1 | awk -v l="$last" '$1 > l' | wc -l)"
      [ "$outside" -eq 0 ] && pass "no beat between the first and last progress line (lines $first-$last)" \
        || fail "$outside beat(s) between the first and last progress line"
      [ "$after" -ge 1 ] && pass "$after beat(s) after the last progress line (the proud beat after the run)" \
        || fail "no beat after the run's last progress line"
    else
      fail "no progress lines found in the node log"
    fi
  else
    [ "$beats" -eq 0 ] && pass "no beat reached the mock arm with puppy.enabled=false" \
      || fail "$beats beat(s) reached the mock arm with puppy.enabled=false"
    if grep -q 'puppy: ' <<<"$lg"; then fail "puppy log lines with puppy.enabled=false"; else pass "no puppy log lines with puppy.enabled=false"; fi
  fi
  stop
}

run_case true
run_case false

if [ "$FAILS" -eq 0 ]; then echo "ALL PASS"; else echo "$FAILS FAILED"; fi
exit "$FAILS"
