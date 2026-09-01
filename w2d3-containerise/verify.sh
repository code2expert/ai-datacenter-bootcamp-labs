#!/usr/bin/env bash
# Green-check verifier for W2D3.
# Pulls the image FRESH from the registry (removes any local copy first), runs it
# with the hf-cache volume, polls /health, sends one completion, cleans up.
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# Usage:  IMAGE=<user>/aidc-serving:cpu-v1 ./verify.sh
set -u

IMAGE="${IMAGE:?set IMAGE=<user>/aidc-serving:cpu-v1}"
NAME="aidc-verify-d3"
PORT="${PORT:-8000}"
TIMEOUT="${TIMEOUT:-420}"  # /health wait; first run downloads the model into the volume

fail() { echo "GREEN CHECK: FAIL ($1)"; cleanup; exit 1; }

cleanup() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}

# make sure we start clean
cleanup

# 1. remove any local copy so the run genuinely comes from the registry
docker image rm "$IMAGE" >/dev/null 2>&1 || true

# 2. pull fresh
echo "pulling $IMAGE ..."
if ! docker pull "$IMAGE" >/dev/null 2>&1; then
  fail "docker pull failed (image not on registry, or not logged in for a private repo)"
fi

# 3. run detached with the model cache volume (weights are NOT in the image)
if ! docker run -d --name "$NAME" -p "${PORT}:8000" \
      -v hf-cache:/home/app/.cache/huggingface "$IMAGE" >/dev/null 2>&1; then
  fail "docker run failed (port ${PORT} in use, or the image will not start)"
fi

# 4. poll /health until healthy or timeout
echo "waiting for /health (up to ${TIMEOUT}s) ..."
deadline=$(( $(date +%s) + TIMEOUT ))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then healthy=1; break; fi
  # if the container died, stop waiting and surface its logs reason
  if [ -z "$(docker ps -q -f name=$NAME)" ]; then
    echo "--- container logs (tail) ---"; docker logs --tail 20 "$NAME" 2>&1 || true
    fail "container exited before /health came up"
  fi
  sleep 3
done
[ "$healthy" -eq 1 ] || fail "/health did not return 200 within ${TIMEOUT}s"

# 5. one real completion through /v1
resp=$(curl -s "http://localhost:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}' 2>/dev/null)

echo "$resp" | grep -q '"chat.completion"' || fail "/v1/chat/completions did not return a chat.completion"
echo "$resp" | grep -q '"content"' || fail "completion had no content field"

echo "image: $IMAGE"
echo "health: 200"
echo "completion: ok"
cleanup
echo "GREEN CHECK: PASS"
exit 0
