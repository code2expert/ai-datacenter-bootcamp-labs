#!/usr/bin/env bash
# Green check for W4D2 (self-healing deployment).
# Run next to deployment.yaml:  bash verify.sh
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# It does not trust a past run: it inspects the live spec (probes, strategy,
# preStop), then performs a rolling update itself with its own in-cluster
# prober watching, and demands zero failed requests. Takes ~90 seconds.
set -u

DEPLOY="${DEPLOY:-serving}"
SVC="${SVC:-serving}"
PROBER="verify-prober-$$"

fail() { echo "GREEN CHECK: FAIL ($1)"; kubectl delete pod "$PROBER" --ignore-not-found >/dev/null 2>&1; exit 1; }

command -v kubectl >/dev/null || fail "kubectl not on PATH"
kubectl get deployment "$DEPLOY" >/dev/null 2>&1 || fail "no deployment '$DEPLOY'"
kubectl get service "$SVC" >/dev/null 2>&1 || fail "no service '$SVC'"

ready=$(kubectl get deployment "$DEPLOY" -o jsonpath='{.status.readyReplicas}')
[ "${ready:-0}" -ge 2 ] || fail "deployment has ${ready:-0} ready replicas, need 2"

spec=$(kubectl get deployment "$DEPLOY" -o json)
echo "$spec" | grep -q '"readinessProbe"' || fail "no readiness probe in the live spec"
echo "$spec" | grep -q '"livenessProbe"' || fail "no liveness probe in the live spec"
echo "$spec" | grep -q '"preStop"' || fail "no preStop hook; the update will drop the endpoint-removal race window"
maxunavail=$(kubectl get deployment "$DEPLOY" -o jsonpath='{.spec.strategy.rollingUpdate.maxUnavailable}')
[ "$maxunavail" = "0" ] || fail "maxUnavailable is '${maxunavail:-<default>}', the zero-downtime bar needs 0"

image=$(kubectl get deployment "$DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].image}')
case "$image" in *"<your-user>"*) fail "deployment still carries the placeholder image" ;; esac

kubectl run "$PROBER" --image="$image" --restart=Never --command -- \
  python -c '
import time, urllib.request
ok = bad = 0
end = time.time() + 45
while time.time() < end:
    try:
        with urllib.request.urlopen("http://'"$SVC"':8000/health", timeout=2) as r:
            ok += (r.status == 200)
    except Exception:
        bad += 1
    time.sleep(0.1)
print(f"PROBE RESULT ok={ok} bad={bad}")
' >/dev/null 2>&1 || fail "could not start the prober pod"

sleep 5
stamp="verify-$(date +%s)"
kubectl set env deployment/"$DEPLOY" VERIFY_STAMP="$stamp" >/dev/null \
  || fail "could not trigger a rolling update"
kubectl rollout status deployment/"$DEPLOY" --timeout=120s >/dev/null \
  || fail "rolling update did not complete"

kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/$PROBER" --timeout=90s >/dev/null 2>&1 \
  || fail "prober did not finish"
line=$(kubectl logs "$PROBER" | tail -1)
kubectl delete pod "$PROBER" >/dev/null 2>&1
echo "$line"
case "$line" in
  "PROBE RESULT ok="*) : ;;
  *) fail "prober produced no result line" ;;
esac
okn=$(printf '%s' "$line" | sed -E 's/.*ok=([0-9]+).*/\1/')
badn=$(printf '%s' "$line" | sed -E 's/.*bad=([0-9]+).*/\1/')
[ "$okn" -ge 200 ] || fail "prober only landed $okn requests; something throttled the watch"
[ "$badn" -eq 0 ] || fail "$badn requests failed during the rolling update; the zero-downtime bar is zero"

echo "rolling update completed with $okn requests served and none dropped"
echo "GREEN CHECK: PASS"
