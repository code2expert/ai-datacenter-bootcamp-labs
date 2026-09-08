#!/usr/bin/env bash
# Green check for W4D1 (first cluster).
# Run next to pod.yaml:  bash verify.sh
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# Truth comes from the cluster, not from your terminal history: the pod must be
# Running AND Ready, and /v1 must answer through a port-forward this script
# opens itself.
set -u

CLUSTER="${CLUSTER:-aidc}"
POD="${POD:-serving}"
PORT=18441
PF_PID=""

fail() { echo "GREEN CHECK: FAIL ($1)"; [ -n "$PF_PID" ] && kill "$PF_PID" 2>/dev/null; exit 1; }

command -v kubectl >/dev/null || fail "kubectl not on PATH"
command -v kind >/dev/null || fail "kind not on PATH"

kind get clusters 2>/dev/null | grep -qx "$CLUSTER" || fail "no kind cluster named '$CLUSTER'"
kubectl config current-context 2>/dev/null | grep -q "kind-$CLUSTER" \
  || fail "kubectl context is not kind-$CLUSTER (kubectl config use-context kind-$CLUSTER)"

phase=$(kubectl get pod "$POD" -o jsonpath='{.status.phase}' 2>/dev/null) \
  || fail "no pod named '$POD' (kubectl apply -f pod.yaml first)"
[ "$phase" = "Running" ] || fail "pod is $phase, not Running (kubectl describe pod $POD, read Events)"

ready=$(kubectl get pod "$POD" -o jsonpath='{.status.containerStatuses[0].ready}')
[ "$ready" = "true" ] || fail "pod is Running but not Ready; /health is still 503 (kubectl logs $POD)"

image=$(kubectl get pod "$POD" -o jsonpath='{.spec.containers[0].image}')
case "$image" in
  *"<your-user>"*) fail "pod.yaml still carries the placeholder image line; edit it to your own image" ;;
esac
echo "pod image: $image"

kubectl port-forward "pod/$POD" "$PORT:8000" >/dev/null 2>&1 &
PF_PID=$!
up=""
for _ in $(seq 1 20); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi
  sleep 0.5
done
[ -n "$up" ] || fail "port-forward opened but /health never answered"

models_json=$(curl -sf "http://127.0.0.1:$PORT/v1/models") \
  || fail "/v1/models did not answer through the cluster"
model_id=$(printf '%s' "$models_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[ -n "$model_id" ] || fail "/v1/models answered but carried no model id"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$model_id\",\"messages\":[{\"role\":\"user\",\"content\":\"green check\"}]}")
case "$code" in
  200|401) : ;;   # 401 means auth is on, which is also a served answer
  *) fail "/v1/chat/completions answered $code through the cluster" ;;
esac

kill "$PF_PID" 2>/dev/null; PF_PID=""

node=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}')
cat > w4d1_evidence.json <<EOF
{"cluster": "$CLUSTER", "pod": "$POD", "image": "$image", "node": "$node", "chat_status": $code}
EOF
echo "evidence written to w4d1_evidence.json"
echo "GREEN CHECK: PASS"
