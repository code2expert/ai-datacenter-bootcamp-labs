#!/usr/bin/env bash
# Green-check verifier for W2D4, the LOCAL half (parts 1 and 2) plus part 3 read
# from the Colab evidence file.
#   part 1: the GPU image builds (or pulls).
#   part 2: the CPU-fallback container answers /health on a GPU-less machine.
#   part 3: the same code showed CUDA on Colab -> gpu_evidence.json has cuda: true.
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# Usage:  IMAGE=<user>/aidc-serving:gpu-v1 ./verify.sh
# Run it WITHOUT --gpus on purpose: the fallback path is the lab.
set -u

IMAGE="${IMAGE:?set IMAGE=<user>/aidc-serving:gpu-v1}"
NAME="aidc-verify-d4"
PORT="${PORT:-8000}"
TIMEOUT="${TIMEOUT:-420}"  # CUDA image is large and the first run may download the model
EVIDENCE="${EVIDENCE:-gpu_evidence.json}"

fail() { echo "GREEN CHECK: FAIL ($1)"; cleanup; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }

cleanup

# part 1: image is available (pull it; if that fails, it must exist locally from a build)
echo "resolving $IMAGE (pull, else expect a local build) ..."
if ! docker pull "$IMAGE" >/dev/null 2>&1; then
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    fail "GPU image $IMAGE neither pulls from the registry nor exists locally (build or push it first)"
  fi
  echo "using local image (not on registry yet)"
fi

# part 2: run WITHOUT --gpus; the CPU fallback must answer /health
if ! docker run -d --name "$NAME" -p "${PORT}:8000" \
      -v hf-cache:/home/app/.cache/huggingface "$IMAGE" >/dev/null 2>&1; then
  fail "docker run failed (port ${PORT} in use, or the image will not start)"
fi

echo "waiting for /health on CPU fallback (up to ${TIMEOUT}s) ..."
deadline=$(( $(date +%s) + TIMEOUT ))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then healthy=1; break; fi
  if [ -z "$(docker ps -q -f name=$NAME)" ]; then
    echo "--- container logs (tail) ---"; docker logs --tail 20 "$NAME" 2>&1 || true
    fail "container exited before /health came up"
  fi
  sleep 3
done
[ "$healthy" -eq 1 ] || fail "/health did not return 200 within ${TIMEOUT}s (CPU fallback)"

# part 3: the Colab evidence must exist and show cuda: true
if [ ! -f "$EVIDENCE" ]; then
  fail "colab evidence missing"
fi
# parse with python (stdlib) so we do not depend on jq
python3 - "$EVIDENCE" <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        e = json.load(f)
except Exception as ex:
    print("PARSE_FAIL:%s" % ex); sys.exit(3)
if e.get("cuda") is not True:
    print("NOT_CUDA"); sys.exit(4)
if not isinstance(e.get("tokens_per_s"), (int, float)) or e.get("tokens_per_s") <= 0:
    print("BAD_TPS"); sys.exit(5)
print("OK:%s:%s" % (e.get("device_name"), e.get("tokens_per_s")))
PY
rc=$?
case "$rc" in
  0) : ;;  # ok
  3) fail "gpu_evidence.json is not valid JSON" ;;
  4) fail "colab evidence shows cuda: false (run the probe on a T4 runtime, not CPU)" ;;
  5) fail "colab evidence has no positive tokens_per_s" ;;
  *) fail "could not read colab evidence" ;;
esac

echo "part 1: GPU image resolved"
echo "part 2: /health 200 on CPU fallback"
echo "part 3: colab evidence shows cuda: true"
cleanup
echo "GREEN CHECK: PASS"
exit 0
