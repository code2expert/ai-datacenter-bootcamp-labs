#!/usr/bin/env bash
# Green check for W4D3 (GPU scheduling as accounting).
# Run in this directory:  bash verify.sh
# Prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
#
# Three checks. Your own serving deployment's ledger entry (Step 1, in your
# namespace); on a node that has a card, the team's engine in the `team`
# namespace (Step 6: Guaranteed, the card visible from inside, no Service
# named vllm); and the overdraft experiment performed by the script itself,
# because a pod that asks for a card nobody can grant must go Pending with a
# scheduler event that names it. On a laptop's kind cluster the middle check
# is skipped: no card, nothing to hold it. TEAM_NS overrides the namespace.
set -u

DOOMED=""
cleanup() { [ -n "$DOOMED" ] && kubectl delete pod "$DOOMED" --ignore-not-found >/dev/null 2>&1; }
trap cleanup EXIT
fail() { echo "GREEN CHECK: FAIL ($1)"; exit 1; }

command -v kubectl >/dev/null || fail "kubectl not on PATH"

# ---------------------------------------------------- your ledger entry ----
# Step 1, in the namespace you are standing in. Same on a laptop's kind.
DEPLOY="${DEPLOY:-serving}"
kubectl get deployment "$DEPLOY" >/dev/null 2>&1 || fail "no deployment '$DEPLOY' in this namespace (kubectl config view --minify | grep namespace:)"

spec=$(kubectl get deployment "$DEPLOY" -o json)
for want in requests limits; do
  printf '%s' "$spec" | grep -q "\"$want\"" || fail "serving spec has no $want; Step 1 not applied"
done
for res in cpu memory; do
  v=$(kubectl get deployment "$DEPLOY" -o jsonpath="{.spec.template.spec.containers[0].resources.requests.$res}")
  [ -n "$v" ] || fail "serving requests carry no $res"
done
image=$(kubectl get deployment "$DEPLOY" -o jsonpath='{.spec.template.spec.containers[0].image}')

# --------------------------------------------------- the team's engine ----
# On a node with a card (the team pod) the engine must be up in the team
# namespace, holding that card. Every assertion corresponds to a fault
# reproduced on real hardware; see instructor/tier1-pod-runbook.md. On a
# laptop's kind cluster there is no card and this block is skipped.
TEAM_NS="${TEAM_NS:-team}"
gpu=$(kubectl get nodes -o jsonpath='{.items[*].status.allocatable.nvidia\.com/gpu}' 2>/dev/null)
if printf '%s' "$gpu" | grep -q '[1-9]'; then
  ENGINE="${ENGINE:-vllm}"
  kubectl -n "$TEAM_NS" get deployment "$ENGINE" >/dev/null 2>&1 \
    || fail "this node has a card, but no engine '$ENGINE' in namespace '$TEAM_NS'; Step 6 (once per team) has not run"

  # A GPU alone leaves the pod BestEffort: QoS is computed from cpu and memory
  # only. The engine must carry all three or it is first in line to be throttled.
  for res in cpu memory 'nvidia\.com/gpu'; do
    v=$(kubectl -n "$TEAM_NS" get deployment "$ENGINE" -o jsonpath="{.spec.template.spec.containers[0].resources.requests.$res}")
    [ -n "$v" ] || fail "the engine requests no ${res//\\/}; it cannot hold a ledger entry it never made"
  done

  # The NEWEST engine pod: during a rollout restart the old one is still
  # listed while it terminates, and it is the one that has lost the card.
  pod=$(kubectl -n "$TEAM_NS" get pods -l app=vllm --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1:].metadata.name}' 2>/dev/null)
  [ -n "$pod" ] || fail "no pod with label app=vllm in '$TEAM_NS'"

  qos=$(kubectl -n "$TEAM_NS" get pod "$pod" -o jsonpath='{.status.qosClass}')
  [ "$qos" = "Guaranteed" ] \
    || fail "engine QoS is $qos, not Guaranteed; requests must equal limits on cpu and memory"

  # A Service named `vllm` injects VLLM_PORT=tcp://... into every pod in the
  # namespace, which vLLM parses as its own port and dies on. Pods that predate
  # the Service keep running, so this only bites at the next restart.
  kubectl -n "$TEAM_NS" get svc vllm >/dev/null 2>&1 \
    && fail "a Service named 'vllm' exists in '$TEAM_NS'; it sets VLLM_PORT and kills the engine on next restart"

  kubectl -n "$TEAM_NS" get pod "$pod" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q True \
    || fail "engine pod is not Ready; check kubectl -n $TEAM_NS logs $pod"

  # The decisive one: the scheduler debits a GPU whether or not the container
  # ever receives it. Only the container can answer.
  kubectl -n "$TEAM_NS" exec "$pod" -- nvidia-smi -L >/dev/null 2>&1 \
    || fail "the container cannot see the GPU it was charged for; node was not provisioned per the pod runbook"

  echo "team engine: Guaranteed, GPU visible inside the container, Service name safe"
fi

# ------------------------------------------------------- the overdraft ----
# Step 3, performed here: a pod that asks for a card nobody can grant must go
# Pending with a scheduler event that names it. On a laptop there is no card;
# on the pod the engine above holds the only one.
DOOMED="verify-wants-gpu-$$"
cat <<EOF | kubectl apply -f - >/dev/null || fail "could not create the doomed GPU pod"
apiVersion: v1
kind: Pod
metadata:
  name: $DOOMED
spec:
  containers:
    - name: wisher
      image: $image
      command: ["sleep", "60"]
      resources:
        requests: {nvidia.com/gpu: 1}
        limits: {nvidia.com/gpu: 1}
EOF

verdict=""
for _ in $(seq 1 20); do
  phase=$(kubectl get pod "$DOOMED" -o jsonpath='{.status.phase}' 2>/dev/null)
  events=$(kubectl get events --field-selector "involvedObject.name=$DOOMED" -o jsonpath='{.items[*].message}' 2>/dev/null)
  if printf '%s' "$events" | grep -qi 'nvidia.com/gpu'; then verdict=ok; break; fi
  [ "$phase" = "Running" ] && fail "the GPU pod scheduled: a card was free, so nothing is holding it; on the pod that means the team engine is not up"
  sleep 1
done
[ "$verdict" = "ok" ] || fail "no scheduler event naming nvidia.com/gpu appeared for the Pending pod"

phase=$(kubectl get pod "$DOOMED" -o jsonpath='{.status.phase}')
[ "$phase" = "Pending" ] || fail "doomed pod is $phase, expected Pending"

echo "overdraft verified: Pending with 'Insufficient nvidia.com/gpu'"
echo "GREEN CHECK: PASS"
