# Lab W2D4: the GPU image

Start:      Your pushed CPU image and `/v1` service from yesterday. Copy `app/generate_probe.py` from this lab into your repo's `app/`.
Objective:  Build a GPU image on a CUDA base that runs the same code, with a CPU fallback so it also runs on a GPU-less machine; push `:gpu-v1`; then run the same probe on a Colab T4 and compare tokens per second.

This lab is split across two machines by design. Colab cannot run Docker, and
most laptops have no NVIDIA GPU. So the proof is two halves:

- Local half (your laptop, Docker): build the GPU image, run it, and watch it
  answer `/health` on CPU fallback because your laptop has no GPU. Push it.
- Colab half (a T4): run the SAME `app/generate_probe.py` and watch it report
  CUDA and a much higher tokens per second.

One image, one script, correct on both. That is the whole idea: a GPU image that
still runs where there is no GPU is what makes the artifact portable.

The three-part tier-0 green check, stated up front so you know the target:

1. The GPU image builds.
2. The CPU-fallback container answers `/health` (on a machine with no GPU).
3. The same code shows CUDA on Colab (the probe writes `cuda: true`).

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Before you build or run:

- On your GPU-less laptop, the GPU image's container will answer `/health` with
  status _____ (it uses the CPU fallback).
- tokens per second for a 128-token generation: on your laptop CPU about _____;
  on the Colab T4 about _____.
- The ratio of T4 to CPU tokens per second will be roughly _____ x.

## The delta

### Local half

#### Step 1: write Dockerfile.gpu on a CUDA base (about 30 min)

A second Dockerfile, `Dockerfile.gpu`, FROM the CUDA runtime base in
`../../../PINS.md` (`nvidia/cuda:12.4.1-runtime-ubuntu22.04`; the prep-week
verify-env pull pre-seeded it, so this is not a download day). It installs
Python and torch per PINS, copies the same `app/`, and its entrypoint runs the
service. The image is built the same on any machine; whether it uses a GPU is
decided at run time by the code, not baked in.

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# python is not in the CUDA base; install 3.11 explicitly - plain `python3`
# on ubuntu22.04 is 3.10, off the course's python:3.11 baseline
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3-pip python3.11-venv && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# the run command mounts hf-cache at /home/app/.cache/huggingface; HF_HOME must
# point there and the process must run as that user, or the model caches to
# /root and re-downloads on every fresh container
ENV HF_HOME=/home/app/.cache/huggingface
RUN useradd --create-home --uid 10001 app

WORKDIR /app

COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt   # torch per ../../../PINS.md

COPY app/ .

RUN mkdir -p /home/app/.cache/huggingface && chown -R app:app /home/app/.cache /app
USER app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Your `main.py` already auto-detects the device: `cuda` if available, else `cpu`.
That is the CPU-fallback pattern. Do not hard-code the device. Build it:

```bash
docker build -f Dockerfile.gpu -t <user>/aidc-serving:gpu-v1 .
```

The image will be large: torch plus a CUDA base is well over a gigabyte. That
size is expected; it is the cost of shipping CUDA userspace.

#### Step 2: run it and watch the CPU fallback (about 15 min)

Your laptop has no NVIDIA GPU, so do NOT pass `--gpus`. Run it plain; the code
falls back to CPU and the service still comes up.

```bash
docker run -d --name serving-gpu -p 8000:8000 \
  -v hf-cache:/home/app/.cache/huggingface \
  <user>/aidc-serving:gpu-v1
docker logs -f serving-gpu     # wait for "model ready" (on cpu), Ctrl-C to stop
curl -s http://localhost:8000/health
```

`/health` answers 200 even though there is no GPU. This is the point: the GPU
image degrades to CPU instead of failing. [onsite] If you are in the lab on a
machine that does have an NVIDIA GPU and the Container Toolkit, you may add
`--gpus all` and watch it pick CUDA instead; the objective is the same either
way.

If you run `--gpus all` on a machine with no NVIDIA runtime, Docker errors out.
That is expected: the fallback path (running WITHOUT `--gpus`) is the lab.

#### Step 3: push :gpu-v1 (about 10 min)

```bash
docker login
docker push <user>/aidc-serving:gpu-v1
```

### Colab half

#### Step 4: run the same probe on a T4 (about 10 min)

Open a Colab notebook, set the runtime to T4 GPU, and run the SAME
`app/generate_probe.py`. Install the deps, upload or clone the file, run it.

```python
# Colab cell
# versions mirror ../../../PINS.md; if a pin changes, change it there first
!pip -q install "transformers==4.46.*" "accelerate==1.1.*"
# bring app/generate_probe.py into the runtime (clone your repo or upload the file), then:
!python app/generate_probe.py
```

It prints `device: cuda (Tesla T4 ...)`, writes `gpu_evidence.json` with
`cuda: true`, and reports tokens per second. Compare that number to the CPU
number from your container: the T4 is many times faster. Write the two numbers
and the ratio in your notes.

#### Step 5: bring the evidence back (about 5 min)

Download `gpu_evidence.json` from Colab and drop it next to your repo (the same
folder you run the verifier from). Part 3 of the green check reads this file. The
Colab run is the only place part 3 can be proven, because your laptop has no GPU.

## Verify (green check)

The local verifier does parts 1 and 2 itself, and reads `gpu_evidence.json` for
part 3. Pass your GPU image reference:

```bash
IMAGE=<user>/aidc-serving:gpu-v1 ./verify.sh
```

It builds or pulls the GPU image, runs it without `--gpus`, confirms `/health`
answers on CPU fallback, then looks for `gpu_evidence.json` next to the repo and
checks it shows `cuda: true`. If the evidence file is missing it fails with the
reason `colab evidence missing`, which means you have not brought the Colab file
back yet. The Colab side also has a paste-in `verify_cell.py` that validates the
evidence right after you generate it.

The last line printed by either verifier is exactly one of:

```
GREEN CHECK: PASS
GREEN CHECK: FAIL (<reason>)
```

## Stretch

- Compare the `:cpu-v1` and `:gpu-v1` image sizes (`docker images`). Explain the
  difference: the CUDA base and the GPU torch wheel. Note it in your table.
- On Colab, also time the generation at 512 new tokens and see how the tokens
  per second holds up as the generation gets longer.

## Failure modes

- `--gpus all` errors on your laptop. Tell: `could not select device driver ...
  with capabilities: [[gpu]]`. Cause: no NVIDIA runtime on the machine. This is
  expected; the lab is the fallback path. Fix: run WITHOUT `--gpus`.
- The CUDA base tag will not pull. Tell: `manifest unknown` on build. Cause: the
  tag drifted. Fix: use the exact tag in `../../../PINS.md`; if it moved, that is a
  pin update, not a lab edit.
- Image is surprisingly large. Tell: `docker images` shows multiple GB. Cause:
  the CUDA base plus the GPU torch wheel. Expected; do not try to strip CUDA.
- Colab evidence forgotten. Tell: the verifier prints `colab evidence missing`.
  Fix: run `generate_probe.py` on the T4, download `gpu_evidence.json`, put it
  next to the repo, rerun the verifier.
- Colab runtime is on CPU, not T4. Tell: the probe writes `cuda: false`. Fix:
  Runtime, Change runtime type, T4 GPU, rerun the probe.
