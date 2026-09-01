# %% [markdown]
# # Week 3 shared Colab scaffold
#
# This file is the source of truth for the reusable Colab cells every week-3 lab
# uses. It is written in py-percent format: each `# %%` block is one standalone,
# pasteable Colab cell. Copy the cells you need into the day's notebook in the
# order the lab README gives.
#
# Why this scaffold exists: a Colab notebook runs cells one at a time, top to
# bottom, and a cell blocks until it returns. A live inference server does not
# return: it runs until you kill it. So you cannot "start the server" in one cell
# and "watch it" in the next the way you would in a terminal with two panes. The
# pattern here is launch-then-poll: one cell launches the server as a background
# subprocess and returns immediately, and a second cell polls the health endpoint
# until the server answers or a timeout fires. Every long-running piece (the
# server, the nvidia-smi sampler) runs in the background and is watched by a
# short cell that returns.
#
# Pin source: versions come from ../../../PINS.md (course root). The vLLM-on-T4 pin
# is verified on a real free-tier T4 before the cohort starts; the confirmed
# version and date land in PINS.md under "Verification status". Do not invent a
# vLLM version here; read the pin.
#
# Convert to a .ipynb when you want a notebook file (the .py stays the source of
# truth):
#   uvx jupytext --to ipynb colab_scaffold.py

# %%
# PINS block. These mirror ../../../PINS.md (course root, the single source of
# truth). If a pin changes, it changes in PINS.md first, then here. The vLLM pin
# is the load-bearing one: it must be the version confirmed on a real free-tier
# T4 during the pre-cohort verification pass. Read PINS.md before you run this.
#
# PINS (from ../../../PINS.md):
#   VLLM_PIN=0.6.*          # OpenAI server; runs the xformers backend on sm75
#   BITSANDBYTES_PIN=0.49.2 # int8/int4 load path (day 1 profiling); 0.44.* is
#                           # broken on Colab's cu128 torch, see PINS.md
#   AUTOAWQ_PIN=0.2.*       # AWQ weights load path (day 4)
#   TRANSFORMERS_PIN=4.46.* # streaming generation (day 2)
#   ACCELERATE_PIN=1.1.*    # device placement
#   HTTPX_PIN=0.27.*        # async A/B client (day 3)
#   OPENAI_PIN=1.54.*       # the client that proves the /v1 contract

# %%
# Cell: the pins and the installer function. Defines only, installs nothing.
# Paste this on every week-3 day. Then paste ONE of the two install cells below,
# whichever the day's README names. Day 1 profiles with transformers and must
# NOT install vLLM; days 2 to 5 serve, and must.
import subprocess, sys

# Pins mirrored from ../../../PINS.md. Keep these two in sync (PINS.md wins).
VLLM_PIN = "0.6.*"
BITSANDBYTES_PIN = "0.49.2"
AUTOAWQ_PIN = "0.2.*"
TRANSFORMERS_PIN = "4.46.*"
ACCELERATE_PIN = "1.1.*"
HTTPX_PIN = "0.27.*"
OPENAI_PIN = "1.54.*"

def pip_install(*specs):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *specs]
    print("installing:", " ".join(specs))
    subprocess.run(cmd, check=True)

# %%
# INSTALL CELL A: profiling only, NO SERVER. This is day 1.
#
# Day 1 loads the model with transformers and reads the card. It never serves,
# so it must NOT install vLLM. Installing vLLM here would replace Colab's torch
# with vLLM's older build AND downgrade numpy to 1.26, and Colab's preinstalled
# extensions are compiled against numpy 2. The model load then dies with
#   RuntimeError: Failed to import transformers.models.qwen2.modeling_qwen2
#   ... numpy.dtype size changed, Expected 96 from C header, got 88
# Verified on a T4, 2026-07-27. Colab's own torch is the torch today. This
# install is about two minutes, not thirty.
pip_install(
    f"transformers=={TRANSFORMERS_PIN}",
    f"accelerate=={ACCELERATE_PIN}",
    f"bitsandbytes=={BITSANDBYTES_PIN}",
)
print("profiling pins installed (no vLLM today)")

# %%
# INSTALL CELL B: the serving set. This is days 3 to 5 (day 2 is CELL A:
# direct transformers loads crash on vLLM's numpy - verified on T4 2026-08-07). About 30 minutes on a
# cold runtime, and it prints almost nothing for most of it, so start it and go
# and fill in your prediction card. vLLM brings its own torch; do NOT install a
# second one.
#
# transformers and accelerate are NOT optional here, even on days you never call
# them directly. vLLM 0.6.x installs its own torch (2.5.1), which downgrades
# Colab's torch and leaves Colab's preinstalled torchaudio compiled against the
# wrong ABI. Colab's preinstalled transformers imports torchaudio at module load,
# so vLLM then dies during startup with
#   OSError: _torchaudio.abi3.so: undefined symbol: aoti_torch_abi_version
# Pinning transformers to 4.46 removes that import path. Verified on a T4,
# 2026-07-27: without these two lines the server never comes up.
#
# autoawq is only needed on day 4; that README says so and adds it to this call.
pip_install(
    f"vllm=={VLLM_PIN}",
    f"transformers=={TRANSFORMERS_PIN}",
    f"accelerate=={ACCELERATE_PIN}",
    f"httpx=={HTTPX_PIN}",
    f"openai=={OPENAI_PIN}",
)
# NOTE (2026-08-07, verified the hard way on a live T4): do NOT add a
# numpy>=2 pin here - vLLM 0.6.x requires numpy<2 and the install fails
# outright. CELL B as verified 2026-07-27 runs on the numpy vLLM chooses.
print("serving pins installed")

# %%
# Cell: launch the server as a background subprocess.
# vLLM's OpenAI-compatible server runs until killed, so it cannot live in a cell
# that must return. Popen launches it in the background and this cell returns at
# once. stdout and stderr are teed to /content/server.log so the health-poll cell
# and you can read what happened. Flags come from PINS.md canon: --dtype half is
# mandatory on the T4 (sm75 has no bf16 and no FlashAttention, so vLLM uses the
# xformers backend). Edit SERVER_ARGS for the day (day 3 is plain, day 4 adds
# --quantization awq and the tool-call flags).
import os, signal, subprocess

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PORT = 8000
SERVER_LOG = "/content/server.log"

# Args as a dict so a lab can override one value without retyping the line.
SERVER_ARGS = {
    "--model": MODEL,
    "--dtype": "half",                 # sm75: no bf16, no FlashAttention
    "--max-model-len": "4096",
    "--gpu-memory-utilization": "0.85",
    "--port": str(PORT),
}

def build_cmd(args: dict) -> list:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    for k, v in args.items():
        if v is None:            # bare flag, e.g. "--enable-auto-tool-choice": None
            cmd.append(k)
        else:
            cmd += [k, str(v)]
    return cmd

def launch_server(args: dict = None):
    args = SERVER_ARGS if args is None else args
    cmd = build_cmd(args)
    print("launching:", " ".join(cmd))
    logf = open(SERVER_LOG, "wb")
    # start_new_session=True puts the server in its own process group so the
    # shutdown cell can kill the whole group, not just the parent pid.
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
    )
    print(f"server pid {proc.pid}, logging to {SERVER_LOG}")
    return proc

server = launch_server()

# %%
# Cell: health poll.
# The launch cell returned immediately; the server is still loading weights in
# the background. This cell polls GET /v1/models until it answers 200 or the
# timeout fires. First launch on a fresh runtime downloads the model, so the
# first poll can take a while; that is what the 300s timeout is for. On timeout
# it prints the last 30 log lines so you can see why (usually still downloading,
# or an OOM, or a bad flag).
import time, urllib.request, urllib.error

def tail_log(path=SERVER_LOG, n=30):
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n:])
    except FileNotFoundError:
        return "(no log file yet)"

def wait_for_health(port=PORT, timeout_s=300, interval_s=3):
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    waited = int(timeout_s - (deadline - time.time()))
                    print(f"server healthy after about {waited}s: {url} -> 200")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # not up yet
        time.sleep(interval_s)
    print(f"TIMED OUT after {timeout_s}s waiting for {url}")
    print("last 30 log lines:")
    print(tail_log())
    print("server did not come up. common causes: model still downloading "
          "(rerun this cell), OOM at load (lower --gpu-memory-utilization to "
          "0.80), or a bad flag (bf16 on sm75; use --dtype half).")
    return False

healthy = wait_for_health()

# %%
# Cell: nvidia-smi sampler thread.
# A daemon thread samples GPU utilisation and memory every 2s into a CSV. It is a
# thread, not a subprocess, so it stops when the runtime does and never outlives
# the notebook. Start it before a measurement, stop it after. Do NOT start it
# twice: two samplers write interleaved rows and double your entries (a named
# failure mode in the day-1 lab). start_sampler() guards against that.
import csv, threading, time

GPU_SAMPLES = "/content/gpu_samples.csv"
_sampler = {"thread": None, "stop": None}

def _sample_loop(stop_event, path, interval_s):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "util_gpu", "mem_used_mib"])
        t0 = time.time()
        while not stop_event.is_set():
            out = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True,
            ).stdout.strip()
            # e.g. "37, 4210"
            parts = [p.strip() for p in out.split(",")]
            if len(parts) == 2:
                w.writerow([round(time.time() - t0, 2), parts[0], parts[1]])
                fh.flush()
            stop_event.wait(interval_s)

def start_sampler(path=GPU_SAMPLES, interval_s=2):
    if _sampler["thread"] and _sampler["thread"].is_alive():
        print("sampler already running; not starting a second one")
        return
    stop = threading.Event()
    th = threading.Thread(
        target=_sample_loop, args=(stop, path, interval_s), daemon=True,
    )
    th.start()
    _sampler["thread"], _sampler["stop"] = th, stop
    print(f"sampler started -> {path} (every {interval_s}s)")

def stop_sampler():
    if _sampler["stop"]:
        _sampler["stop"].set()
    if _sampler["thread"]:
        _sampler["thread"].join(timeout=5)
    _sampler["thread"], _sampler["stop"] = None, None
    print("sampler stopped")

def read_util_mean(path=GPU_SAMPLES):
    """Mean GPU utilisation over the samples on file. Use it after stop_sampler."""
    vals = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            try:
                vals.append(float(row["util_gpu"]))
            except (KeyError, ValueError):
                pass
    return sum(vals) / len(vals) if vals else 0.0

# %%
# Cell: clean shutdown.
# Terminate the server process group and confirm port 8000 is free again. Run
# this between labs, or before relaunching with different flags. Killing only the
# parent pid can leave a child holding the port; killpg kills the whole group the
# launch cell created with start_new_session=True.
def shutdown_server(proc=None, port=PORT):
    try:
        proc = server if proc is None else proc
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        print(f"sent SIGTERM to process group of pid {proc.pid}")
    except (ProcessLookupError, NameError):
        print("no server process to kill")
    # give it a moment, then confirm the port is free
    time.sleep(3)
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=2):
            print(f"WARNING: port {port} still answering; something is still up")
    except (urllib.error.URLError, ConnectionError, OSError):
        print(f"port {port} is free")

# shutdown_server()   # uncomment to run

# %% [markdown]
# ## RECOVERY: the session-died cell
#
# Free Colab drops runtimes without warning: you lose the GPU, the installed
# packages, and any running server. When that happens you do not re-run the whole
# notebook. Run the ONE cell below. It kills anything left over, reinstalls the
# pins, relaunches the server, and re-polls health. When it prints the healthy
# line you continue from the last step you had finished. Every week-3 lab points
# at this cell at the top for exactly this reason.

# %%
# RECOVERY CELL (self-contained). Runtime died? Run only this cell, then continue
# from your last completed step. It repeats the install, launch, and health-poll
# so you do not have to scroll. Nothing here depends on earlier cells having run.
import os, sys, time, signal, subprocess, urllib.request, urllib.error

RECOVERY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RECOVERY_PORT = 8000
RECOVERY_LOG = "/content/server.log"

# Pins mirrored from ../../../PINS.md (keep in sync; PINS.md wins).
_R_TRANSFORMERS = "4.46.*"   # PINS.md: mandatory beside vLLM, every fresh runtime
_R_ACCELERATE = "1.1.*"
_R_NEED_AWQ = False          # set True on day 4+ if your locked model is AWQ
_R_VLLM = "0.6.*"; _R_HTTPX = "0.27.*"; _R_OPENAI = "1.54.*"

# If a day added --quantization awq or the tool-call flags, add them here too so
# recovery brings the server back the way the lab needs it. Default is the plain
# serving config from canon.
RECOVERY_ARGS = {
    "--model": RECOVERY_MODEL,
    "--dtype": "half",
    "--max-model-len": "4096",
    "--gpu-memory-utilization": "0.85",
    "--port": str(RECOVERY_PORT),
}

# 1) kill any leftover server holding the port
subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"],
               check=False)
time.sleep(2)

# 2) reinstall pins (fresh runtime has nothing)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"vllm=={_R_VLLM}", f"transformers=={_R_TRANSFORMERS}",
                f"accelerate=={_R_ACCELERATE}", f"httpx=={_R_HTTPX}",
                f"openai=={_R_OPENAI}"]
               + (["autoawq==0.2.9"] if _R_NEED_AWQ else []),
               check=True)
print("pins reinstalled")

# 3) relaunch the server in the background
_r_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
for k, v in RECOVERY_ARGS.items():
    _r_cmd += [k] if v is None else [k, str(v)]
_r_logf = open(RECOVERY_LOG, "wb")
server = subprocess.Popen(_r_cmd, stdout=_r_logf, stderr=subprocess.STDOUT,
                          start_new_session=True)
print(f"relaunched server pid {server.pid}, logging to {RECOVERY_LOG}")

# 4) re-poll health (first load re-downloads the model; hence 300s)
_deadline = time.time() + 300
while time.time() < _deadline:
    try:
        with urllib.request.urlopen(
                f"http://localhost:{RECOVERY_PORT}/v1/models", timeout=5) as r:
            if r.status == 200:
                print("RECOVERED: server healthy. continue from your last step.")
                break
    except (urllib.error.URLError, ConnectionError, OSError):
        pass
    time.sleep(3)
else:
    print("recovery timed out. last 30 log lines:")
    try:
        with open(RECOVERY_LOG, errors="replace") as fh:
            print("".join(fh.readlines()[-30:]))
    except FileNotFoundError:
        print("(no log file)")
    print("if it keeps timing out: switch to the Kaggle fallback in the shared "
          "README, or rotate to another team Colab account.")
