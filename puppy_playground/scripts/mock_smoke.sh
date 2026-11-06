#!/usr/bin/env bash
# Curl every endpoint of a RUNNING puppy-playground container started with mock:=true.
#   scripts/mock_smoke.sh [host] [port]        (default localhost 8080)
# In mock mode nothing is sent to the arm: every motion request ends in the engine's own checks
# (a refusal) or a "mock: not sent" success. Each endpoint prints one PASS/FAIL line.
HOST="${1:-localhost}"
PORT="${2:-8080}"
BASE="http://$HOST:$PORT"
FAILS=0

check() { # name, expected-codes (space separated), method, path, [json body]
  local name="$1" want="$2" method="$3" path="$4" body="${5:-}" code
  if [ "$method" = GET ]; then
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$BASE$path")
  else
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST \
      -H 'Content-Type: application/json' -d "${body:-{\}}" "$BASE$path")
  fi
  case " $want " in
    *" $code "*) echo "PASS  $name  ($method $path -> $code)" ;;
    *) echo "FAIL  $name  ($method $path -> $code, wanted $want)"; FAILS=$((FAILS + 1)) ;;
  esac
}

settle() { sleep 1.5; } # let a refused goal end so the next request is not "busy"

check "page"             "200"     GET  /
check "status"           "200"     GET  /status
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 -X POST -d 'name=nod' "$BASE/express")
if [ "$code" = 415 ]; then echo "PASS  form POST refused  (POST /express -> 415)"; else echo "FAIL  form POST refused  (-> $code, wanted 415)"; FAILS=$((FAILS + 1)); fi
check "unknown path"     "404"     GET  /nope
check "stop"             "200 503" POST /stop
for action in plan_stage stage rest recover; do
  check "$action" "200 409" POST "/$action"; settle
done
check "express"          "200 409" POST /express '{"name":"head_tilt","size":0.25}'; settle
check "express (bad)"    "400"     POST /express '{"name":"no_such_thing"}'
check "combo"            "200 409" POST /combo   '{"names":["nod","wag"],"size":0.25}'; settle
check "combo (bad)"      "400"     POST /combo   '{"names":[]}'
check "auto on"          "200 409" POST /auto    '{"on":true,"size":0.25}'
check "auto off"         "200"     POST /auto    '{"on":false}'
check "stop (again)"     "200 503" POST /stop
check "status (last)"    "200"     GET  /status

echo "--- /status:"; curl -s --max-time 10 "$BASE/status"; echo
echo "--- expected: every goal refused or 'mock: not sent'; the driver received nothing."
[ "$FAILS" -eq 0 ] && echo "ALL PASS" || echo "$FAILS FAILED"
exit "$FAILS"
