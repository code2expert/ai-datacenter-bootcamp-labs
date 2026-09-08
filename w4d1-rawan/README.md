# Lab W4D1: first cluster

Start:      Your wk2 image on Docker Hub (the wk-2 Thursday checkpoint), Docker
            running, and the install block below (kind, kubectl, helm, cloudflared).
Objective:  Stand up a local Kubernetes cluster, run your own serving container
            in it as a pod, and reach `/v1` through the cluster instead of
            through Docker.

Time: about 2 and a half hours. Tier 0 throughout: everything here is kind on
your own laptop, CPU only, echo backend. Tier 1 (the team GPU pod) enters on
Tuesday.

The point of today is a translation, not a new system: the container you have
run with `docker run` since week 2 does not change at all. What changes is who
starts it, watches it, and gives it a network. By the end of the afternoon the
answer to all three is "the cluster", and you have seen where the container
lives inside it.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Write these on your card before the first command.

- `kind` runs your "cluster" on this laptop. Where do you expect the cluster's
  node to actually live? (You have been using the answer all week.)
- When the pod starts, `/health` returns 503 before it returns 200. From what
  you know of the app since week 2: what is it waiting for?
- `kubectl logs serving` and `docker logs <container>`: same output or
  different? Commit to a guess.
- Three pods will refuse to run this afternoon. Exactly one refusal comes from
  the scheduler itself. Which status will it wear: `ImagePullBackOff`,
  `Pending`, or `CrashLoopBackOff`?

## The delta

### Step 1: install the four tools of the fortnight (about 15 min)

kind and kubectl for today; helm (Wednesday) and cloudflared (Thursday) now,
while you are already here - Thursday morning is not install time. Versions
per `../../../PINS.md`; macOS (brew) / Windows (winget) equivalents on each
tool's site.

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.31.0/kind-linux-amd64
chmod +x ./kind && sudo mv ./kind /usr/local/bin/
curl -LO "https://dl.k8s.io/release/v1.35.0/bin/linux/amd64/kubectl"
chmod +x ./kubectl && sudo mv ./kubectl /usr/local/bin/
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
curl -Lo ./cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ./cloudflared && sudo mv ./cloudflared /usr/local/bin/
kind version && kubectl version --client && helm version --short && cloudflared --version
```

### Step 2: a cluster on your laptop (about 15 min, first pull is the slow part)

```bash
kind create cluster --name aidc --wait 120s
kubectl cluster-info --context kind-aidc
kubectl get nodes
```

One node, role `control-plane`. Answer to the prediction: the "node" is a
Docker container (`docker ps` shows it, named `aidc-control-plane`). Kubernetes
is not magic hardware; today it is a container that runs your containers.

### Step 3: your image, into the cluster (about 15 min)

kind's node cannot see your laptop's Docker images; it has its own image store.
Two ways in, use the first today:

```bash
# fast, offline, no registry roundtrip
docker pull <your-user>/aidc-serving:cpu-v1     # if it is not already local
kind load docker-image <your-user>/aidc-serving:cpu-v1 --name aidc
```

The second way is the one production uses: let the cluster pull from the
registry itself. That is Thursday's story, when the cluster is not on your
laptop.

### Step 4: the three refusals (about 25 min)

Before you run a healthy pod, meet the three ways a pod refuses to run. The
`broken/` folder next to this README holds three manifests. Edit only pod-c's
image line (same one-line edit as pod.yaml will need); leave the rest alone.

```bash
kubectl apply -f broken/
kubectl get pods
```

Within a minute you have the day's whole taxonomy on one line each:
`ImagePullBackOff`, `Pending`, `CrashLoopBackOff`. The rule of the week: **do
not open the YAML to find out why.** The cluster already knows, and asking it
is the skill. For each pod, in this order:

```bash
kubectl describe pod pod-a | tail -12    # Events: who said no, in words
kubectl describe pod pod-b | tail -12
kubectl logs pod-c                       # empty? then:
kubectl describe pod pod-c | grep -A4 "Last State"
```

Write one line per pod on your card: the cause, and **who** refused - the
node's hands (it cannot get the image), the matchmaker (no node fits the
request), or your own container (it started and died; the exit code is in
Last State). One of the three is the cluster itself saying no; notice which,
because Tuesday's GPU scheduling is the same refusal wearing
`Insufficient nvidia.com/gpu`.

Then clean up. The diagnosis is the deliverable; the broken pods are not yours
to fix:

```bash
kubectl delete -f broken/
```

### Step 5: the pod (about 30 min)

First, read `pod.yaml`'s env block and notice something new: `MODEL_BACKEND:
"echo"`. Your image has carried this switch since week 2 - it is the
serving-stack app's test backend, a deterministic replier that loads in
seconds and never downloads a model. All of week 4 runs on it **on purpose**:
this week's objectives are the cluster's, and a pod that starts in two seconds
lets you watch rollouts and probes instead of download bars. The real model
comes back when the GPU does (tier 1), and Thursday's go-live is explicit
about what that means for your consumer.

Edit `pod.yaml`'s image line to your own image, then:

```bash
kubectl apply -f pod.yaml
kubectl get pods -w          # watch until Running; Ctrl-C to stop watching
```

While it is `-w`atching, read the file: a pod is the smallest thing Kubernetes
schedules, and the `containers:` block is your `docker run` flags as YAML:
image, port, env. Nothing in it is new; only the owner changed.

Now the three verbs you already know, in their kubectl spelling:

```bash
kubectl logs serving                      # docker logs
kubectl exec -it serving -- /bin/sh       # docker exec
kubectl describe pod serving              # docker inspect, but readable
```

`describe`'s Events section at the bottom is where scheduling problems appear
all week. Read it once now while it is boring: Scheduled, Pulled, Created,
Started.

### Step 6: reach it (about 20 min)

The pod has an IP inside the cluster, and your laptop is outside. Today's
bridge is port-forward, the operator's private door:

```bash
kubectl port-forward pod/serving 8000:8000
```

Leave that running; in a second terminal, week 2's own smoke test:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/models
MODEL=$(curl -s localhost:8000/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hello from inside the cluster\"}]}"
```

The reply is the echo backend answering from inside Kubernetes. Same contract,
same app, new operator.

### Step 7: write the evidence (about 10 min)

```bash
bash verify.sh          # writes w4d1_evidence.json next to it, then checks it
```

## Verify (green check)

```bash
bash verify.sh
```

It checks the cluster exists, the pod is `Running` and `Ready`, `/v1/models`
answers through a port-forward it opens itself, and the pod's image is not the
placeholder. Expected final line: `GREEN CHECK: PASS`.

## Stretch

Delete the pod (`kubectl delete pod serving`) and re-apply it. Then time how
long `kubectl apply` to `Ready` takes with the echo backend. Tomorrow a
Deployment does the re-creating for you; today you are the restart policy.

## Failure modes

- **`kind create cluster` hangs then fails on the node image.** Almost always
  the pull: the node image is ~900 MB. Check `docker pull` works at all
  (Hub login from prep week), then retry; the image is cached after the first
  success.
- **Pod stuck `ImagePullBackOff`.** The cluster is trying to pull an image it
  cannot see: the tag is misspelt, the image was never pushed in wk2, or you
  skipped `kind load` and the Hub pull is rate-limited. `kubectl describe pod
  serving` names the exact image string it tried.
- **Pod `Running` but `0/1 READY` for a long time.** With `MODEL_BACKEND=echo`
  readiness arrives in seconds. If you pointed it at the transformers backend,
  it is downloading a model inside the node; that is Tuesday's tier-1 story,
  switch back to echo today.
- **`port-forward` says address already in use.** Something on your laptop
  already owns 8000 (often the compose stack from wk2 Thursday still running).
  `docker compose down` it, or forward `8001:8000` and curl 8001.
- **`curl /health` gives 503.** Honest readiness: the backend has not loaded
  yet. If echo takes more than ~10s to become ready, `kubectl logs serving`
  and read what it is actually doing.
- **You fixed the broken pods.** Tell: pod-a is Running because you corrected
  the tag. The refusals are not yours to repair; the deliverable is the three
  one-line causes. Delete them (`kubectl delete -f broken/`) and move on -
  fixing them teaches YAML editing, diagnosing them teaches operating.
