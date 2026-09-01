# Lab W2D5: compose and teams

Start:      Your pushed `<user>/aidc-serving:cpu-v1` image on Docker Hub.
Objective:  Bring the service up with Docker Compose (image from the registry, env config, model cache volume, a working healthcheck), form your team and claim a progress board username, and pass the checkpoint: a fresh pull plus compose up serves `/v1` end to end.

This is the week-2 wrap. You stop running `docker run` by hand and describe the
service once in `compose.yaml`, so bringing it up is one command and the config
is version-controlled. Then teams form, and the checkpoint tags your first
shipped artifact (badge Shipped).

Quiz 1 already ran in the morning block: see [`../../quiz/quiz-1.md`](../../quiz/quiz-1.md).

## Predict (by hand)

**Submitted before you bring anything up. Credited for submitting, never marked
right or wrong**, so commit to a real answer rather than a safe one.

- After `docker compose up -d`, how long until `docker compose ps` reports the
  service as `healthy`? About _____ seconds (the healthcheck has a start period
  while the model loads).
- If you change `MODEL_ID` in `.env` and run `docker compose up -d` again, does
  compose recreate the container? _____
- The healthcheck runs INSIDE the container. Does the base image have `curl`?
  _____ (this decides which healthcheck form works).
- Your service currently has no key. If you published this port to the internet
  right now, how long until someone else is generating tokens on your GPU?
  _____ (hours / days / weeks). Write a number; you will be asked to defend it.
- After you add a key in step 4, which endpoint must still answer WITHOUT one,
  and why? _____

## The delta

You author two files: `.env` (from the given `.env.example`) and `compose.yaml`.

### Step 1: create .env (about 5 min)

Copy the template and set your image and model.

```bash
cp .env.example .env
# edit .env: IMAGE=<your-user>/aidc-serving:cpu-v1, MODEL_ID, HOST_PORT
```

compose reads `.env` automatically. Do not commit `.env`; commit `.env.example`.
(`.env.example` landed in an upstream template fix this week; if your copy
lacks it, `git pull upstream main` first. `setup.md` wired that remote.)

### Step 2: write compose.yaml (about 45 min)

One service, pulled from the registry (not built here), configured by env, with
the model cache as a named volume, a healthcheck, and a restart policy.

```yaml
services:
  serving:
    image: ${IMAGE}
    ports:
      - "${HOST_PORT}:8000"
    environment:
      MODEL_ID: ${MODEL_ID}
      # step 4 adds these; both must reach the container or its green check
      # fails: without API_KEY inside the process, /v1 answers 200 open
      API_KEY: ${API_KEY}
      MAX_TOKENS: ${MAX_TOKENS}
    volumes:
      - hf-cache:/home/app/.cache/huggingface
    restart: unless-stopped
    healthcheck:
      # curl is NOT in python:3.11-slim; python IS. Use python for the probe.
      test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 120s

volumes:
  hf-cache:
```

Why each part:

- `image: ${IMAGE}` pulls your day-3 image; compose does not rebuild it.
- `environment: MODEL_ID` is how the container learns which model to load; your
  `main.py` already reads `MODEL_ID`.
- the named volume `hf-cache` keeps weights out of the image and alive across
  restarts, exactly as in the `docker run` on day 3.
- the healthcheck probes `/health` from inside the container. The base image has
  no `curl`, so a `curl` healthcheck silently fails and the container never goes
  healthy. The python one-liner works because python is in the image.
- `start_period: 120s` gives the model time to load before failures count.
- `restart: unless-stopped` brings it back if it crashes.

### Step 3: bring it up and read the state (about 20 min)

```bash
docker compose up -d
docker compose logs -f serving      # wait for "model ready", Ctrl-C to stop
docker compose ps                   # STATUS should become "healthy"
```

`docker compose ps` shows the health state. It starts `health: starting` during
the start period, then flips to `healthy` once `/health` answers. Smoke it from
the host:

```bash
curl -s http://localhost:${HOST_PORT:-8000}/health
curl -s http://localhost:${HOST_PORT:-8000}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
```

### Step 4: lock the door (about 25 min)

Right now anyone who can reach your port can spend your GPU. In three weeks the
Agentic AI cohort will call this endpoint as real traffic, and an agent in a
retry loop is indistinguishable from an attack. Unbounded consumption is LLM10
in the OWASP Top 10 for LLM applications, and it is the single most likely way
your service falls over in this course.

Two things to add, both configured the way you configured everything else today.

**A key.** Add `API_KEY` to `.env` and `.env.example` (a fake value in the
example, a real one in `.env`, which is gitignored). Then require it:

- Read it in `main.py` the way `MODEL_ID` is already read at the top:
  `API_KEY = os.environ.get("API_KEY", "")`. Configuration lives in the
  environment, which is exactly why step 2's compose block can deliver it.
- Reject any request to `/v1/*` whose `Authorization` header is not
  `Bearer <your key>`, with **401**.
- Leave `/health` open. Kubernetes probes call it in week 4 and they will not
  carry a key. This is a real design decision, not an oversight: liveness
  endpoints are unauthenticated almost everywhere, which is why they must never
  return anything sensitive.
- If `API_KEY` is unset, run open but **log a loud warning at startup**. Tuesday
  and Wednesday's images have no key and their green checks send none, so a hard
  refusal would break your own earlier work. In a real deployment the rule is
  stricter: refuse to start, because a service that silently runs
  unauthenticated on a missing variable is a standard way this goes wrong.
  Note the difference on your card; week 4 makes the strict version enforceable
  because the key comes from a Kubernetes Secret that either exists or does not.

**A ceiling.** A client can already ask for any `max_tokens` it likes and hold
your GPU for as long as it wants. Clamp it: read `MAX_TOKENS` as a hard upper
bound and silently reduce any larger request down to it. One line, and it caps
the worst case a single caller can cost you.

Rate limiting proper (requests per key per minute) belongs at the gateway in
week 4, not in your app. Name it now as the third leg and move on.

**Ship the change.** Both edits live in the image, and your compose pulls the
day-3 `cpu-v1`, which predates them. Rebuild, tag `cpu-v2`, push, and point
`IMAGE` in `.env` at the new tag; a rebuild on the same tag changes nothing a
registry already holds. Then `docker compose up -d` again.

Prove both:

```bash
# no key -> 401
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:${HOST_PORT:-8000}/v1/models
# with key -> 200
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $API_KEY" \
  http://localhost:${HOST_PORT:-8000}/v1/models
# health stays open -> 200
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:${HOST_PORT:-8000}/health
```

The key itself never leaves your `.env` - not into `teams.md`, not into any
committed file; this morning's slide said secrets never enter git and it meant
this exact moment. `teams.md` records the key **owner** (whose `.env` holds
it); the handover in week 4 happens in person, key to on-call, out of band.

### Step 5: form your team and claim a username (about 15 min)

Open [`teams.md`](teams.md) and fill one row: team, member usernames, your board
username, one emphasis (Inference Platform, RAG Infrastructure, or
Observability Platform), your key owner, and the **modality answer**: message
your paired Agentic AI team today and ask whether their capstone needs an
image model or a voice model. Record their answer verbatim ("text only" or
what they named). After week 3's model lock this question becomes expensive;
today it is one message. Use usernames, not real names. Commit it.

### Step 6: the checkpoint, a clean fresh-pull run (about 15 min)

Prove the artifact ships. Tear down, remove the local image, and bring it up
again so compose pulls fresh from the registry and serves `/v1`.

```bash
docker compose down
docker image rm ${IMAGE}         # force a real pull
docker compose up -d
docker compose ps                # healthy again, from a fresh pull
```

A fresh pull that comes up healthy and serves `/v1` is the badge-Shipped
checkpoint. This is also what the verifier checks.

## Verify (green check)

From the lab folder, with `.env` and `compose.yaml` in place:

```bash
./verify.sh
```

It runs `docker compose up -d`, waits for the healthcheck to report `healthy`,
sends one completion, then `docker compose down`. The last line printed is
exactly one of:

```
GREEN CHECK: PASS
GREEN CHECK: FAIL (<reason>)
```

## Stretch

- Add a second service to `compose.yaml` (for example a tiny static status page)
  and see how compose brings both up together. You will add real sidecars in
  later weeks; this is the shape.
- Set the service to depend on its own health with `depends_on` condition
  `service_healthy` for a future sidecar, and read how compose orders start-up.

## Failure modes

- Container never goes `healthy`. Tell: `docker compose ps` stays
  `health: starting` then `unhealthy`. Cause: a `curl`-based healthcheck in an
  image with no curl. Fix: use the python healthcheck form shown above.
- Volume name versus bind path confusion. Tell: the model re-downloads every
  `up`, or compose complains about the volume. Cause: you wrote a bind mount
  (`./hf-cache:/...`) or misspelled the named volume. Fix: use the named volume
  `hf-cache:/home/app/.cache/huggingface` and declare it under top-level `volumes:`.
- Stale image. Tell: your code change is not reflected after `up`. Cause:
  compose reuses the local image. Fix: `docker compose pull` (or remove the
  local image) to get the registry copy.
- Env not read. Tell: the service loads the wrong model, or `${IMAGE}` is empty.
  Cause: the file is not named exactly `.env`, or you ran compose from another
  directory. Fix: name it `.env`, run compose from the folder that holds it.
- Port already bound. Tell: `port is already allocated` on `up`. Fix: stop
  whatever holds `HOST_PORT`, or change `HOST_PORT` in `.env`.
