# Week 3 shared Colab scaffold

Every week-3 lab serves a model on a free Colab T4 and measures it. The pieces
that repeat across days (install pins, launch the server, poll health, sample
the GPU, shut down cleanly, recover a dead runtime) live once here in
`colab_scaffold.py` and are pasted into each day's notebook. The `.py` is the
source of truth; the labs never re-print these cells.

## Why a scaffold at all

A Colab notebook runs cells one at a time, and a cell blocks until it returns. A
live inference server never returns: it runs until you kill it. So you cannot
start the server in one cell and watch it in the next the way you would in a
terminal with two panes. The scaffold uses launch-then-poll: one cell launches
the server as a background subprocess and returns immediately, and a second cell
polls the health endpoint until the server answers. The long-running pieces (the
server, the nvidia-smi sampler) run in the background; short cells watch them and
return.

## The cells

`colab_scaffold.py` is py-percent format: each `# %%` block is one Colab cell.
In order:

1. **PINS block (comment).** Mirrors `../../../PINS.md`. Read the pin file before
   you run anything; the vLLM-on-T4 pin is verified on a real free-tier T4
   before the cohort starts and the confirmed version and date live in PINS.md.
2. **Install pins.** Installs vLLM, httpx, openai from the pins. vLLM brings its
   own torch build; do not install a second torch. A cold runtime takes a few
   minutes. Days that need `bitsandbytes` (day 1) or `autoawq` (day 4) add those
   pins; the day's README says so.
3. **Launch server.** Starts vLLM's OpenAI server as a background subprocess and
   returns at once. Flags come from PINS.md canon: `--dtype half` is mandatory
   on the T4. `SERVER_ARGS` is a dict so a lab can override one value; day 4 adds
   `--quantization awq` and the tool-call flags. Logs tee to `/content/server.log`.
4. **Health poll.** Polls `GET /v1/models` until 200 or a 300s timeout. First
   launch downloads the model, which is why the timeout is generous. On timeout
   it prints the last 30 log lines and the likely cause.
5. **nvidia-smi sampler thread.** A daemon thread samples utilisation and memory
   every 2s into `/content/gpu_samples.csv`. `start_sampler()` / `stop_sampler()`
   bracket a measurement; `read_util_mean()` gives the mean afterward. Starting
   it twice is guarded against (double entries are a day-1 failure mode).
6. **Clean shutdown.** Terminates the server process group and confirms port 8000
   is free. Run it between labs or before relaunching with different flags.
7. **RECOVERY cell.** The one cell to run when the runtime dies: it kills
   leftovers, reinstalls pins, relaunches the server, re-polls health. It is
   self-contained (depends on no earlier cell). Every lab points at it up top.

## Paste-in order per lab

Every lab starts the same way and pastes the specifics from its own README after.

- **Day 1 (profile inference):** pins + INSTALL CELL A (no vLLM) -> sampler
  thread cell -> the day's load-and-measure cells (no vLLM server; day 1 loads
  the model directly with transformers/bitsandbytes to profile dtypes).
- **Day 2 (inference anatomy):** pins + INSTALL CELL B -> the day's streaming and static
  batching cells (transformers `TextIteratorStreamer`; again no vLLM server, this
  is the hand-rolled baseline day). Sampler optional.
- **Day 3 (engine swap):** pins + INSTALL CELL B -> launch server -> health poll -> the
  day's A/B client cells -> clean shutdown. RECOVERY referenced at the top.
- **Day 4 (quantise and lock):** pins + INSTALL CELL B (add `autoawq`) -> launch server
  with `--quantization awq` and the model's tool-call flags -> health poll -> the
  smoke test cell -> clean shutdown, relaunch fp16 for the side by side.
- **Day 5 (benchmark harness):** pins + INSTALL CELL B -> launch the locked model ->
  health poll -> run `bench.py` -> clean shutdown.

Keep the RECOVERY cell in every day-3-onward notebook. If the runtime dies
mid-lab, run it and continue from your last completed step.

## When Colab cuts you off

Free Colab gives no guarantee. It will drop your runtime, and on a bad day it
will refuse you a GPU for hours. Two standing moves:

- **Tier-0 fallback: Kaggle Notebooks.** Kaggle gives 30 GPU hours per week on a
  comparable card. The scaffold cells run there unchanged (it is the same
  Jupyter-style environment). If Colab denies a GPU, open the same notebook on
  Kaggle, enable the GPU accelerator, and continue. This is the named tier-0
  fallback for the whole week.
- **Team Colab account rotation (standing practice).** A team has several Google
  accounts. When one account's Colab quota is exhausted, switch to another team
  member's account and keep going. Rotate before you are blocked, not after.

Neither of these changes the lab. The measurements and the green check are the
same on Colab or Kaggle.

## Notebook file when you want one

The `.py` is the source of truth. Generate a `.ipynb` on demand:

```
uvx jupytext --to ipynb colab_scaffold.py
```
