# Lab W3D3: the engine swap

Start:      wk2-fastapi checkpoint plus yesterday's baselines.json (the file you
            downloaded on day 2). A fresh Colab T4 runtime. The shared scaffold
            at `../shared/colab_scaffold.py`.
Objective:  Put vLLM behind the same OpenAI /v1 your week-2 client already speaks,
            then A/B it against Monday's hand-rolled numbers under concurrency and
            see where continuous batching pays.

Time: about 3 hours.

If the runtime dies at any point, run the RECOVERY cell (the last cell in
`../shared/colab_scaffold.py`) and continue from the last step you finished. Keep
that cell pasted at the top of today's notebook.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Fill this in before you run anything.

- At concurrency 8, vLLM's throughput (tokens/s across all requests) will be
  about ______ times Monday's static-batch-8 number. Write the multiple down.
- Monday you measured how far static batching scales from batch 1 to batch 8.
  Write that multiple here from your own baselines.json: ______ x.
- vLLM runs the identical queue. Predict how far IT scales from concurrency 1 to
  8: ______ x. This is the number the day turns on, so commit to it.
- Monday's `slot_efficiency` collapsed to about a third once the queue had mixed
  output lengths. Continuous batching does not pay that tax. So you should expect
  vLLM's scaling multiple to be ______ (larger / smaller / the same as) static
  batching's, and roughly ______ x larger.
- Hand in the card.

## The delta

## Cell 0: Create virtual environment
#Updated on September 01, 2026.

#Google updated the version of python on colab to 3.13 which would affect the code belo from running smoothly. To correct this, you should deploy
#Python 3.10 in a virtual environment before running the lab. Thie ensures that your environment has all you need to run the lab smoothly.

import subprocess
import os

# 1. Install python3.10 and venv support on Colab
subprocess.run(["sudo", "apt-get", "update", "-y"], check=True)
subprocess.run(["sudo", "apt-get", "install", "python3.10", "python3.10-venv", "python3.10-dev", "-y"], check=True)

# 2. Create the virtual environment
subprocess.run(["python3.10", "-m", "venv", "/content/venv"], check=True)

# 3. Upgrade pip inside the venv
subprocess.run(["/content/venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"], check=True)

# 4. Install the serving pins + autoawq for Day 4
subprocess.run([
    "/content/venv/bin/python", "-m", "pip", "install", "-q",
    "vllm==0.6.*",
    "transformers==4.46.*",
    "accelerate==1.1.*",
    "autoawq==0.2.*",
    "httpx==0.27.*",
    "openai==1.54.*"
], check=True)

print("Python 3.10 venv ready with AWQ serving pins installed!")

### Cell 1: install pins and launch vLLM (about 35 min, mostly the install)

Paste, in order, from `../shared/colab_scaffold.py`:

1. the **pins and installer** cell, then **INSTALL CELL B** (the base serving
   set is enough today),
2. the **launch server** cell,
3. the **health poll** cell.

The launch cell uses the canon flags from PINS.md: `--dtype half`,
`--max-model-len 4096`, `--gpu-memory-utilization 0.85`, port 8000. The pin
itself lives in PINS.md and is verified on a real T4 before the cohort. Do not
retype the vLLM version here.

Budget the time honestly. The vLLM install alone is about 30 minutes on a fresh
runtime and prints almost nothing for most of it, so it looks hung when it is
working. Start it, then go and fill in your prediction card while it runs. The
server then takes about 4 more minutes to come up, most of that capturing
cudagraphs after the weights have loaded.

Wait for the health poll to print the healthy line. On a fresh runtime the
launch also downloads the 3.1 GB of weights, so the 300s health timeout is
close to the bone; if it expires, poll again rather than assuming a failure.

One line in the startup log looks alarming and is not:
`Cannot use FlashAttention-2 backend for Volta and Turing GPUs`. The T4 is
Turing, FlashAttention needs newer silicon, and vLLM falls back to xformers.
That fallback is the reason `--dtype half` is in the flags.

### Cell 2: the seam test (about 5 min)

Your week-2 client code does not change. The only difference is `base_url` now
points at vLLM. That is the whole lesson of the engine swap: the interface held,
so the engine slid under it.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "In one sentence, what is a GPU?"}],
)
print(r.choices[0].message.content)
```

It just works. Nothing above this line knows or cares that the engine changed.

### Cell 3: restore Monday's baselines (about 5 min)

Upload the baselines.json you downloaded on day 2.

```python
from google.colab import files
uploaded = files.upload()   # pick baselines.json
import json
baseline = json.load(open("baselines.json"))
print("baseline batch tokens/s:", baseline["batch"])
```

If you lost the file, rerun day 2's Cells 1 to 5 to regenerate it, then come
back. If the afternoon cannot afford that, use `baselines.sample.json` from this
folder (reference T4 numbers, clearly marked as such) and note the substitution
in your report. The A/B needs a real Monday number to compare against.

### Cell 4: the async A/B client (about 40 min)

`ab_client.py` (next to this README) sweeps concurrency 1, 4, 8 against the vLLM
server with an async httpx client, a fixed prompt set, and a warm-up that is
excluded from the timing. Paste its contents as a cell, or load it:

```python
# paste the contents of ab_client.py here, then:
prompts = FIXED_PROMPTS            # defined in ab_client.py
vllm_measured = await run_sweep(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompts=prompts,
    concurrencies=[1, 4, 8],
)
for level in vllm_measured:
    print(level)
```

Each level reports tokens/s aggregated across the concurrent requests. Warm-up
requests run first and are dropped so the model-load and cache-warm cost does not
pollute the numbers.

### Cell 5: write ab_report.json (about 20 min)

Combine Monday's baseline with today's vLLM sweep and compute the speedup at each
concurrency.

```python
import json

def tokps_at(level_list, c):
    return next(x["tokens_per_s"] for x in level_list if x["concurrency"] == c)

vllm_by_c = {x["concurrency"]: x["tokens_per_s"] for x in vllm_measured}
# Monday's static batching at 1/4/8 is the baseline curve
base_by_c = {int(k): v for k, v in baseline["batch"].items()}

speedup = {c: round(vllm_by_c[c] / base_by_c[c], 2)
           for c in vllm_by_c if c in base_by_c}

report = {
    "baseline": base_by_c,               # from Monday's baselines.json
    "vllm": vllm_by_c,                    # measured today
    "speedup_by_concurrency": speedup,
    "predicted_speedup": None,           # <- put your prediction-card number here
}
with open("ab_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
```

Set `predicted_speedup` to the multiple you wrote on your card, so the report
holds your prediction next to the measurement.

### Cell 6: reflect and submit (about 15 min)

Compare the two **scaling multiples**, not the per-level speedups. Divide each
engine's concurrency-8 throughput by its own concurrency-1 throughput:

```python
static_scaling = base_by_c[8] / base_by_c[1]
vllm_scaling   = vllm_by_c[8] / vllm_by_c[1]
print(f"static batching scales {static_scaling:.2f}x, vLLM scales {vllm_scaling:.2f}x")
print(f"continuous batching is worth {vllm_scaling / static_scaling:.2f}x of scaling")
```

Static batching should land near 3x and vLLM near 5x. That gap is the whole
lesson and it traces straight back to Monday's `slot_efficiency`: static batching
decoded roughly three token-slots for every one anybody asked for, because short
requests sat finished in a slot they could not release. Continuous batching
evicts a finished sequence and admits a waiting one on the next step, so it does
not pay that tax and its curve keeps climbing where the static one flattens.

Now look at `speedup_by_concurrency` and notice it does **not** rise cleanly with
concurrency. It is a ratio of two curves, so it moves with whichever curve is
steeper at that moment, and with only 24 requests in the sweep vLLM runs short of
queue depth to exploit at the top level. The ratio of the ratios is the robust
number; the per-level speedup is not, and reading a trend into it would be
reading noise. If you predicted a clean rise, this is the correction: write down
which number you would quote to a capacity planner and why.

Submit your concurrency-8 tokens/s to the progress board under your team username.
This is the throughput column's debut. If the progress board is not up yet, put the
number in the shared sheet; the fallback counts the same.

### Cell 7: clean shutdown

Paste the **clean shutdown** cell from the scaffold to free port 8000 before you
close the runtime.

## Verify (green check)

Paste `verify_cell.py` as the last cell and run it. It checks the ab_report.json
schema, that vLLM's concurrency-8 throughput beats Monday's batch-8 baseline, and
that the speedup fields were computed. Expected final line:

```
GREEN CHECK: PASS
```

## Stretch

Add concurrency 16 to the sweep. Watch throughput keep climbing where Monday's
static batching would have fallen over: continuous batching admits the extra
requests instead of padding a fixed batch.

## Failure modes

- **OOM at launch.** Tell: the health poll times out and the log tail shows CUDA
  out of memory during load. Fix: `--gpu-memory-utilization` is too high for the
  fp16 model on a T4 that already has some memory in use; lower it to 0.80 in the
  launch cell's SERVER_ARGS and relaunch.
- **Health poll times out on the first model load.** Tell: first launch on a
  fresh runtime, 300s passes, log shows it still downloading. Fix: this is the
  cold download; just rerun the health-poll cell (the model is on disk now) or
  rerun the launch and poll. The timeout is generous for exactly this.
- **An old process holds port 8000 after a crash.** Tell: the new launch fails
  with address-in-use, or the seam test hits a stale server. Fix: run the
  RECOVERY cell; it kills the leftover vLLM process before relaunching. Or run
  the clean-shutdown cell first.
- **Forgetting --dtype half.** Tell: the server errors on startup about bf16 or
  an unsupported dtype. Fix: the T4 is sm75, no bf16; the scaffold launch cell
  already sets `--dtype half`. If you edited SERVER_ARGS, put it back.
- **base_url missing the /v1 suffix.** Tell: the client 404s or connection
  refused. Fix: the OpenAI client wants `http://localhost:8000/v1`, with the
  `/v1`. The health poll hits `/v1/models`; match it.
