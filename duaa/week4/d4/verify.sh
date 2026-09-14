#!/usr/bin/env bash
# Green check for W4D4 (Helm + autoscaling).
# Run in this directory:  bash verify.sh
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# The scale event is observed, not assumed: the script starts its own load
# generator against the release and waits for the HPA to raise desired
# replicas above minReplicas. Takes about four minutes on the reference laptop.
set -u

RELEASE="${RELEASE:-team}"
DEPLOY="$RELEASE-serving"
LOADGEN="verify-loadgen-$$"

fail() { echo "GREEN CHECK: FAIL ($1)"; kubectl delete pod "$LOADGEN" --ignore-not-found >/dev/null 2>&1; exit 1; }

command -v kubectl >/dev/null || fail "kubectl not on PATH"
command -v helm >/dev/null || fail "helm not on PATH"

helm status "$RELEASE" >/dev/null 2>&1 || fail "no helm release '$RELEASE' (helm install $RELEASE ./serving-chart ...)"
kubectl get deployment "$DEPLOY" >/dev/null 2>&1 || fail "release exists but deployment $DEPLOY does not"

spec=$(kubectl get deployment "$DEPLOY" -o json)
for want in '"readinessProbe"' '"livenessProbe"' '"preStop"'; do
  echo "$spec" | grep -q "$want" || fail "the chart lost ${want//\"/} on its way from Monday's YAML"
done
maxunavail=$(kubectl get deployment "$DEPLOY" -o jsonpath='{.spec.strategy.rollingUpdate.maxUnavailable}')
[ "$maxunavail" = "0" ] || fail "chart strategy lost maxUnavailable: 0"
image=$(kubectl get deployment "$DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].image}')
case "$image" in *"<your-user>"*) fail "release still runs the placeholder image" ;; esac

kubectl get hpa "$DEPLOY" >/dev/null 2>&1 || fail "no HPA named $DEPLOY (helm upgrade with --set hpa.enabled=true)"
minr=$(kubectl get hpa "$DEPLOY" -o jsonpath='{.spec.minReplicas}')
maxr=$(kubectl get hpa "$DEPLOY" -o jsonpath='{.spec.maxReplicas}')
[ "${maxr:-0}" -gt "${minr:-0}" ] || fail "HPA bounds $minr..$maxr leave no room to scale"

kubectl top pods >/dev/null 2>&1 || fail "kubectl top has no metrics; metrics-server is not serving (kind needs --kubelet-insecure-tls)"

# the release's own model id and key (empty if open) - a changed modelId or an
# enabled key must not turn the load into 400s/401s with a misleading verdict
model_id=$(kubectl get deploy "$DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MODEL_ID")].value}' 2>/dev/null)
[ -n "$model_id" ] || model_id="Qwen/Qwen2.5-0.5B-Instruct"
api_key=$(kubectl get secret serving-keys -o jsonpath='{.data.api-key}' 2>/dev/null | base64 -d 2>/dev/null || true)

kubectl run "$LOADGEN" --image="$image" --restart=Never \
  --env "LG_MODEL=$model_id" --env "LG_KEY=${api_key:-}" --command -- \
  python -c '
import json, os, threading, time, urllib.request
BODY = json.dumps({"model": os.environ["LG_MODEL"], "messages": [
    {"role": "user", "content": "repeat the word load " * 60}], "max_tokens": 200}).encode()
HDRS = {"Content-Type": "application/json"}
if os.environ.get("LG_KEY"):
    HDRS["Authorization"] = "Bearer " + os.environ["LG_KEY"]
def worker():
    end = time.time() + 210
    while time.time() < end:
        try:
            req = urllib.request.Request("http://'"$DEPLOY"':8000/v1/chat/completions",
                                         data=BODY, headers=HDRS)
            urllib.request.urlopen(req, timeout=10).read()
        except Exception:
            time.sleep(0.2)
threads = [threading.Thread(target=worker) for _ in range(24)]
[t.start() for t in threads]
[t.join() for t in threads]
' >/dev/null 2>&1 || fail "could not start the load generator"

echo "load running; waiting for the HPA to move (up to 3 minutes)"
scaled=""
for _ in $(seq 1 36); do
  desired=$(kubectl get hpa "$DEPLOY" -o jsonpath='{.status.desiredReplicas}' 2>/dev/null)
  if [ "${desired:-0}" -gt "${minr:-1}" ]; then scaled="$desired"; break; fi
  sleep 5
done
kubectl delete pod "$LOADGEN" --ignore-not-found >/dev/null 2>&1
[ -n "$scaled" ] || fail "HPA never raised desired replicas above $minr under load; check requests vs real usage"

echo "scale event observed: desired replicas $minr -> $scaled"
echo "GREEN CHECK: PASS"
