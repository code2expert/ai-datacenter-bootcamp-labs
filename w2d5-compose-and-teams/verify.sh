#!/usr/bin/env bash
# Green-check verifier for W2D5.
# Brings the stack up with compose, waits for the healthcheck to report healthy,
# sends one completion, tears down. Prints exactly one line last:
#   GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# Usage:  ./verify.sh            (run from the folder holding compose.yaml and .env)
set -u

SERVICE="${SERVICE:-serving}"
TIMEOUT="${TIMEOUT:-240}"   # first run may pull the image and download the model
# read HOST_PORT from .env if present, else default 8000
PORT="$(grep -E '^HOST_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2)"
PORT="${PORT:-8000}"

# docker compose (v2) or docker-compose (v1)
if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
else
  echo "GREEN CHECK: FAIL (no docker compose available)"; exit 1
fi

fail() { echo "GREEN CHECK: FAIL ($1)"; $DC down >/dev/null 2>&1 || true; exit 1; }

[ -f compose.yaml ] || [ -f docker-compose.yaml ] || fail "no compose.yaml in this folder"
[ -f .env ] || fail "no .env in this folder (cp .env.example .env and edit it)"

# bring it up
if ! $DC up -d >/dev/null 2>&1; then
  fail "compose up failed (bad image ref, port in use, or invalid compose.yaml)"
fi

# wait for the service container to report healthy
echo "waiting for $SERVICE to become healthy (up to ${TIMEOUT}s) ..."
deadline=$(( $(date +%s) + TIMEOUT ))
state="unknown"
while [ "$(date +%s)" -lt "$deadline" ]; do
  cid="$($DC ps -q "$SERVICE" 2>/dev/null | tail -1)"
  if [ -n "$cid" ]; then
    state="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "$cid" 2>/dev/null || echo unknown)"
    if [ "$state" = "healthy" ]; then break; fi
    if [ "$state" = "no-healthcheck" ]; then
      fail "the service has no healthcheck; add one to compose.yaml"
    fi
    # if the container died, surface why
    running="$(docker inspect -f '{{.State.Running}}' "$cid" 2>/dev/null || echo false)"
    if [ "$running" != "true" ] && [ "$state" != "starting" ]; then
      echo "--- logs (tail) ---"; docker logs --tail 20 "$cid" 2>&1 || true
      fail "service container is not running (state: $state)"
    fi
  fi
  sleep 3
done
[ "$state" = "healthy" ] || fail "service did not become healthy within ${TIMEOUT}s (last state: $state)"

# the key the service should be enforcing (step 4)
KEY="$(grep -E '^API_KEY=' .env 2>/dev/null | tail -1 | cut -d= -f2-)"
[ -n "$KEY" ] || fail "no API_KEY in .env; step 4 requires the service to be keyed"

# /health must stay OPEN: k8s probes in week 4 carry no key
hcode="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" 2>/dev/null)"
[ "$hcode" = "200" ] || fail "/health returned $hcode unauthenticated; probes need it open"

# /v1 must be CLOSED without a key
ucode="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/v1/models" 2>/dev/null)"
[ "$ucode" = "401" ] || fail "/v1/models returned $ucode without a key, expected 401; anyone who can reach the port can spend your GPU"

# and OPEN with one
acode="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${KEY}" \
  "http://localhost:${PORT}/v1/models" 2>/dev/null)"
[ "$acode" = "200" ] || fail "/v1/models returned $acode with the key from .env, expected 200"

# one real completion from the host, now keyed
resp="$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${KEY}" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}' 2>/dev/null)"

echo "$resp" | grep -q '"chat.completion"' || fail "/v1/chat/completions did not return a chat.completion"
echo "$resp" | grep -q '"content"' || fail "completion had no content field"

echo "service: healthy"
echo "auth: /health open, /v1 401 without key, 200 with key"
echo "completion: ok"
$DC down >/dev/null 2>&1 || true
echo "GREEN CHECK: PASS"
exit 0
