#!/usr/bin/env bash
# Run from the lab directory: bash verify.sh
set -euo pipefail
PF_PID=""
TMPDIR_LAB=""
cleanup() {
  if [ -n "$PF_PID" ]; then
    kill "$PF_PID" 2>/dev/null || true
    wait "$PF_PID" 2>/dev/null || true
  fi
  if [ -n "$TMPDIR_LAB" ]; then rm -r -- "$TMPDIR_LAB"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
fail() { echo "GREEN CHECK: FAIL ($1)"; exit 1; }
for tool in python3 curl; do
  command -v "$tool" >/dev/null || fail "$tool not on PATH"
done

if [ -z "${GRAFANA_URL:-}" ]; then
  command -v kubectl >/dev/null || fail "kubectl not on PATH"
  NAMESPACE=${NAMESPACE:-team}
  GRAFANA_SERVICE=${GRAFANA_SERVICE:-grafana}
  GRAFANA_PORT=${GRAFANA_PORT:-3000}
  kubectl -n "$NAMESPACE" get svc "$GRAFANA_SERVICE" >/dev/null 2>&1 \
    || fail "Grafana Service not found in $NAMESPACE (Step 1)"
  TMPDIR_LAB=$(mktemp -d)
  kubectl -n "$NAMESPACE" port-forward --address=127.0.0.1 \
    "svc/$GRAFANA_SERVICE" ":$GRAFANA_PORT" >"$TMPDIR_LAB/forward.log" 2>&1 &
  PF_PID=$!
  for ((i=0; i<40; i++)); do
    kill -0 "$PF_PID" 2>/dev/null || fail "Grafana port-forward exited; check Service endpoints"
    PORT=$(sed -n 's/^Forwarding from 127.0.0.1:\([0-9]*\) ->.*/\1/p' "$TMPDIR_LAB/forward.log" | head -n 1)
    if [ -n "$PORT" ]; then
      GRAFANA_URL="http://127.0.0.1:$PORT"
      if curl -fsS --max-time 1 "$GRAFANA_URL/api/health" >/dev/null 2>&1; then break; fi
    fi
    sleep 0.5
  done
  [ "$i" -lt 40 ] || fail "Grafana never answered /api/health"
fi
python3 "$(dirname "$0")/verify.py" --url "$GRAFANA_URL"
