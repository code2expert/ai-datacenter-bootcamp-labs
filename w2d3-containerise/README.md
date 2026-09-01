# Lab W2D3: containerise

Start:      Your working `/v1` service from yesterday (in `app/`). It runs on your machine with uvicorn.
Objective:  Package the service into a Docker image, run it with the model cache mounted (weights stay out of the image), slim it, and push it to Docker Hub as `<user>/aidc-serving:cpu-v1`.

Yesterday the service ran from a virtualenv on your machine. Today it becomes an
image anyone can pull and run with one command. The rule that shapes the whole
lab: the weights do NOT go in the image. The image is generic code; the model
arrives at run time into a mounted cache volume. That keeps the image small and
lets the same image serve any model id.

Runs on any laptop with Docker. No GPU today.

Checkpoints for the room: image built and `/v1` answering locally by
**14:30**, the naive-vs-slim size pair on your card by **15:15**, pushed and
fresh-pulled by **16:30**. The push itself can take 20+ minutes on shared
wifi - start it and keep working; that is why Step 4 says the push rides
underneath.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Before you build, predict:

- Your final image size (code plus a CPU torch, no weights baked in): about
  _____ MB.
- If you `COPY . .` before installing requirements, how many of your next ten
  code edits will re-run `pip install`? _____
- After a slim pass (right base, `.dockerignore`, no pip cache), the image will
  shrink from _____ MB to _____ MB.

Write these down, then measure at the end.

## The delta

You will author three files in your repo root: `.dockerignore` first, then
`Dockerfile`, then use them. The image copies your `app/` code and installs the
pinned deps; it does not copy weights or your `.venv`.

### Step 1: author .dockerignore first (about 10 min)

Write `.dockerignore` before the Dockerfile, so your first build already excludes
the junk. If you skip this, `COPY` drags your `.venv`, the HF cache, and git
history into the build context and the image balloons.

```
.venv/
__pycache__/
*.pyc
.git/
.gitignore
*.md
results.json
```

### Step 2: write the Dockerfile step by step (about 45 min)

Build it up in this order; each line has a reason.

1. Base image: `python:3.11-slim` per `../../../PINS.md`. Slim, not full: you do
   not need build toolchains at run time for these wheels.
2. `WORKDIR /app` so paths are predictable.
3. Copy `requirements.txt` and install it BEFORE copying your code. This is the
   layer-cache move: requirements change rarely, code changes constantly. If
   requirements are their own layer, a code edit reuses the cached install
   instead of re-downloading torch every build.
4. Install with `--no-cache-dir` so pip's download cache does not bloat the
   layer.
5. Copy the app code (`app/`).
6. `EXPOSE 8000` (documentation) and `CMD` running uvicorn on `0.0.0.0:8000`.

```dockerfile
FROM python:3.11-slim

# a non-root user whose home holds the model cache; the run command mounts
# a volume exactly here, so the path must exist and be writable
RUN useradd --create-home app
ENV HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# requirements layer FIRST: cached across code edits.
# --index-url makes the CPU wheel index authoritative, so pip takes the CPU
# torch (~180 MB) instead of the default CUDA build (~2.5 GB). Without it this
# image is ~6.5 GB and the push takes over an hour on classroom wifi.
COPY app/requirements.txt .
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
      -r requirements.txt   # versions per ../../../PINS.md

# code SECOND: this is the layer that changes every edit
COPY app/ .

# the cache dir must exist and be owned by app BEFORE USER drops privileges;
# a volume mountpoint Docker auto-creates is root-owned, and the first model
# download dies with a permission error
RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache /app
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This is the same shape as the repo's own `serving-stack/Dockerfile` - after
writing yours by hand once, diff the two; every difference should have a
reason you can say out loud.

Build it and tag with your Docker Hub username:

```bash
docker build -t <user>/aidc-serving:cpu-v1 .
```

### Step 3: run it with the model cache mounted (about 20 min)

Run detached, publish port 8000, and mount the `hf-cache` named volume at the
Hugging Face cache path. The weights are NOT in the image; the first run
downloads them into the volume, and every later run reuses them.

```bash
docker run -d --name serving -p 8000:8000 \
  -v hf-cache:/home/app/.cache/huggingface \
  <user>/aidc-serving:cpu-v1
```

Watch it come up, then hit it:

```bash
docker logs -f serving        # wait for "model ready", Ctrl-C to stop following
curl -s http://localhost:8000/health
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
```

Prove the volume works: stop and remove the container, run it again, and note the
model does NOT download a second time.

```bash
docker rm -f serving
docker run -d --name serving -p 8000:8000 -v hf-cache:/home/app/.cache/huggingface <user>/aidc-serving:cpu-v1
docker logs serving   # loads from cache, no download
```

### Step 4: the before picture, then the push and the measure (about 35 min, the push rides underneath)

Your Step-2 image was born optimised (slim base, no pip cache), so it cannot
show you what the optimisations bought. Build the before picture once, from a
given naive Dockerfile - full base, cached pip - and read the difference off
`docker images`:

```dockerfile
# Dockerfile.naive - the BEFORE picture. Do not push this one.
FROM python:3.11
WORKDIR /app
COPY app/requirements.txt .
RUN pip install -r requirements.txt
COPY app/ .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -f Dockerfile.naive -t aidc-serving:naive .
docker images | grep aidc-serving      # naive vs your slim tag, side by side
```

Write both numbers on your card; the gap (roughly a gigabyte, mostly the full
Debian base plus pip's wheel cache) is what Step 1 and 2's choices were worth.
Delete the naive tag after reading it (`docker rmi aidc-serving:naive`).

Kick `docker push` off in a second terminal NOW - it is the afternoon's long
pole and it needs none of your attention. While it uploads, name what the gap
is made of. The three levers, in impact order:

- Base image choice: `python:3.11-slim` over `python:3.11` saves hundreds of MB.
- `--no-cache-dir` on pip: no download cache baked into the layer.
- `.dockerignore`: keeps `.venv` and the cache out of the build context (both
  builds here used yours, so its saving does not show in this pair - remove it
  and rebuild naive if you want to see it, it is dramatic).

Stretch lever (optional): a multi-stage build that installs into a venv in a
builder stage and copies only the venv into a slim final stage. Note the extra
saving if you try it.

Fill this table in your repo README from your own numbers:

| stage | image size |
|---|---|
| naive build (full base, cached pip) |  |
| your slim build |  |

### Step 5: log in and push (about 15 min)

`docker login` is mandatory: anonymous Docker Hub pulls are rate-limited, and
the verifier pulls fresh. Log in, then push.

```bash
docker login          # your Docker Hub username and a token
docker push <user>/aidc-serving:cpu-v1
```

The namespace in the tag must be YOUR Docker Hub username, or the push is denied.

## Verify (green check)

The verifier pulls your image fresh from the registry (removing any local copy
first), runs it with the cache volume, polls `/health`, sends one completion,
and cleans up. Pass your image reference in the `IMAGE` env var:

```bash
IMAGE=<user>/aidc-serving:cpu-v1 ./verify.sh
```

The last line printed is exactly one of:

```
GREEN CHECK: PASS
GREEN CHECK: FAIL (<reason>)
```

Because it removes the local image and pulls, a pass proves the image runs from
the registry on a clean machine, not just on yours.

## Stretch

- Multi-stage build (the lever above). Measure the extra saving.
- Add a `HEALTHCHECK` instruction to the Dockerfile that curls `/health`, and
  watch `docker ps` show the container as healthy. You will need this exact
  mechanism on Thursday for compose.

## Failure modes

- Every build re-runs `pip install`. Tell: torch downloads on each build even
  for a one-line code change. Cause: you copied code before requirements, so the
  install layer's cache busts every time. Fix: `COPY requirements.txt` and
  install it, THEN `COPY app/ .`.
- The model downloads on every run. Tell: minutes of download each `docker run`.
  Cause: you forgot the `-v hf-cache:/home/app/.cache/huggingface` mount, so the
  cache dies with the container. Fix: add the volume mount.
- Port 8000 already bound. Tell: `port is already allocated`. Fix: stop whatever
  holds it (`docker ps`, then `docker rm -f`), or map a different host port.
- Push denied. Tell: `denied: requested access to the resource is denied`.
  Cause: not logged in, or the tag namespace is not your Docker Hub username.
  Fix: `docker login`; re-tag with `<your-user>/aidc-serving:cpu-v1`.
- Works locally, fails fresh. Tell: your container runs, but the verifier's
  fresh pull fails to start. Cause: a file the container needs was excluded by
  `.dockerignore` (for example you ignored something under `app/`). Fix: narrow
  `.dockerignore`; never exclude code the app imports.
