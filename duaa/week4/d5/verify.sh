#!/usr/bin/env bash
# Green check for W4D5 (go-live).
# Usage:  bash verify.sh <public-base-url> <bearer-key>
#         bash verify.sh https://t07.aidc.nadir.sh mykey123
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# It checks the posture a consumer meets: /health open, /v1 locked without the
# key and serving with it, a real completion, metrics history accruing in the
# in-cluster Prometheus, and an integration note with nothing left to fill in.
set -u

URL="${1:-}"; KEY="${2:-}"
NOTE="${NOTE:-my-integration-note.md}"
PF_PID=""

fail() { echo "GREEN CHECK: FAIL ($1)"; [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null; exit 1; }

[ -n "$URL" ] && [ -n "$KEY" ] || fail "usage: verify.sh <public-base-url> <bearer-key>"
URL="${URL%/}"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$URL/health") \
  || fail "could not reach $URL at all"
[ "$code" = "200" ] || fail "/health answered $code from outside; probes and the watch need it open"

code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 "$URL/v1/models")
[ "$code" = "401" ] || fail "/v1/models without a key answered $code, expected 401; the door is open"

models=$(curl -s --max-time 15 -H "Authorization: Bearer $KEY" "$URL/v1/models") \
  || fail "/v1/models with the key did not answer"
model_id=$(printf '%s' "$models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[ -n "$model_id" ] || fail "keyed /v1/models carried no model id (got: $(printf '%.80s' "$models"))"

body=$(curl -s --max-time 30 -X POST "$URL/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"go-live green check\"}]}")
printf '%s' "$body" | grep -q '"choices"' \
  || fail "completion from outside failed (got: $(printf '%.100s' "$body"))"

if command -v kubectl >/dev/null 2>&1 && kubectl get svc prometheus >/dev/null 2>&1; then
  kubectl port-forward svc/prometheus 19090:9090 >/dev/null 2>&1 &
  PF_PID=$!
  sleep 3
  # The engine's own series (vllm:... or vllm_..., depending on how the scrape
  # negotiated the name) or the app's aidc_requests_total: either spelling,
  # either engine, counted rather than named.
  total=$(curl -sG --data-urlencode 'query=count({__name__=~"vllm.*|aidc_requests_total"})' \
      'http://127.0.0.1:19090/api/v1/query' \
    | python3 -c "import json,sys; r=json.load(sys.stdin)['data']['result']; print(sum(float(x['value'][1]) for x in r))" 2>/dev/null)
  kill "$PF_PID" 2>/dev/null; PF_PID=""
  case "$total" in
    ''|*[!0-9.]*) fail "Prometheus answered nothing for the serving series; the metrics history is not accruing" ;;
  esac
  awk -v t="$total" 'BEGIN{exit !(t > 0)}' \
    || fail "Prometheus holds no serving series (vllm* or aidc_requests_total); is it in the same namespace as team-serving?"
  echo "metrics history: $total serving series in the store"
else
  echo "note: no in-cluster Prometheus visible from here (running off-cluster?); metrics check skipped"
fi

[ -f "$NOTE" ] || fail "no $NOTE next to this script; copy integration-note.md, fill it, save as $NOTE"
if tr '\n' ' ' < "$NOTE" | grep -qE '<[a-zA-Z][^>]*>'; then
  fail "$NOTE still contains angle-bracket placeholders (they can span lines): $(tr '\n' ' ' < "$NOTE" | grep -oE '<[a-zA-Z][^>]{0,60}>' | head -1)"
fi
grep -q "$model_id" "$NOTE" \
  || fail "$NOTE never names the served model id ($model_id); the note and the endpoint must agree"

echo "outside-in posture verified for $URL serving $model_id"
echo "GREEN CHECK: PASS"
