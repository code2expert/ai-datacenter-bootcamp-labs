# Lab W3D2: inference anatomy, by hand

Start:      wk2-fastapi checkpoint. A fresh Colab T4 runtime. The shared
            scaffold at `../shared/colab_scaffold.py`. Yesterday's profile.json
            in your notes for context.
Objective:  Measure the two halves of a response (time to first token, then the
            per-token gap), watch the KV cache grow and check it against the
            arithmetic, and hand-roll static batching to feel where it ceilings.
            Export the baselines and download them, because tomorrow's A/B has to
            survive a lost runtime.

Time: about 3 hours. Checkpoints: install green and the TTFT table by
**14:30**, KV growth measured by **15:30**, baselines.json exported AND
downloaded by **16:30** - the download is the one non-negotiable, tomorrow's
A/B dies without it.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Fill this in before you run anything.

- Time to first token (TTFT) is dominated by prefill (reading the whole prompt).
  A longer prompt makes TTFT go ______ (up / down / no change).
- After the first token, decode emits one token at a time. The mean gap between
  tokens (TPOT) depends mostly on ______ (prompt length / model size and memory
  bandwidth).
- KV cache math for Qwen2.5-1.5B: 28 layers, 2 KV heads, head_dim 128, fp16.
  Per token that is 2 (K and V) x 28 x 2 x 128 x 2 bytes = ______ KB per token.
  So a 4096-token context holds about ______ GB of KV. (Compute it; do not guess.)
- Static batching: if you pad 8 prompts of different lengths and run them as one
  batch, the batch finishes when the ______ prompt finishes.
- Hand in the card.

## The delta

Everything today uses transformers directly (no vLLM server). This is the
hand-rolled baseline: you build the primitives yourself so tomorrow's engine has
something honest to beat.

### Cell 1: install pins (about 4 min)

(If Colab drops you mid-afternoon: the shared RECOVERY cell reinstalls this
exact set; your `baselines.json` is the one artifact D3 needs, so download it
the moment Cell 5 writes it, not at day's end.)

Paste the **pins and installer** cell from `../shared/colab_scaffold.py`, then
**INSTALL CELL A** (the profiling set - same as day 1). Today is hand-rolled
measurement with transformers directly; no server ever starts, and vLLM must
NOT be installed in this runtime:

```python
pip_install(
    f"transformers=={TRANSFORMERS_PIN}",
    f"accelerate=={ACCELERATE_PIN}",
)
```

Why not the serving set, when days 3 to 5 use it? Verified on a live T4,
2026-08-07: installing vLLM drags numpy below 2, and a **direct**
`AutoModelForCausalLM` load in that runtime then dies with
`numpy.dtype size changed` inside `modeling_qwen2` - exactly the crash day 1's
warning describes. vLLM's own server survives it; your hand-rolled cells do
not. Serving days install CELL B because they only talk to the server; today
you touch the model directly, so today is a CELL A day. (This also makes Cell 1
about 4 minutes, not 30.)

Load the model once, fp16:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="cuda")
```

### Cell 2: TTFT and TPOT by streaming (about 30 min)

You cannot separate first-token time from per-token time unless you timestamp
each token as it arrives. `TextIteratorStreamer` yields tokens as they generate;
run generate() on a background thread and read the stream on the main thread,
stamping each yield.

```python
import time, threading
from transformers import TextIteratorStreamer

def prompt_of_len(n_tokens: int) -> str:
    base = "Explain the following in detail.\n"
    filler = "A data center serves many inference requests at once. " * 600
    ids = tok(base + filler)["input_ids"][:n_tokens]
    return tok.decode(ids)

def measure_stream(prompt: str, new_tokens: int = 128):
    enc = tok(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tok, skip_prompt=True,
                                    skip_special_tokens=True)
    kwargs = dict(**enc, max_new_tokens=new_tokens, do_sample=False,
                  streamer=streamer)
    th = threading.Thread(target=model.generate, kwargs=kwargs)
    t0 = time.time()
    th.start()
    stamps = []
    for _ in streamer:
        stamps.append(time.time())
    th.join()
    ttft = stamps[0] - t0
    # mean inter-token gap over the tokens after the first
    if len(stamps) > 1:
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        tpot = sum(gaps) / len(gaps)
    else:
        tpot = 0.0
    total = stamps[-1] - t0
    return {"ttft_s": round(ttft, 4), "tpot_s": round(tpot, 4),
            "total_s": round(total, 4), "n_tokens": len(stamps)}

# Warm-up, and it is not optional. The first generation on a fresh runtime pays
# CUDA context init and kernel autotuning, and all of that lands inside its TTFT.
# Time it and the shortest prompt comes out slowest, which is backwards and would
# tell you prefill does not depend on prompt length. Throw one generation away.
measure_stream(prompt_of_len(128), new_tokens=8)

ttft_by_len = {}
for n in [128, 512, 2048]:
    r = measure_stream(prompt_of_len(n))
    ttft_by_len[str(n)] = r["ttft_s"]
    print(n, r)
```

TTFT climbs with prompt length: prefill reads the whole prompt before the first
token. TPOT stays roughly flat across prompt lengths, because decode does the
same memory-bound step each time regardless of how long the prompt was. If you
measured with streaming off you would get total time and call it TTFT; that is
the classic mistake (see failure modes).

### Cell 3: KV growth versus the formula (about 30 min, by hand)

Measure the VRAM delta across a generation at three context lengths, then compare
against the computed KV size. The prediction is arithmetic, not a guess.

```python
import gc

def kv_formula_kb_per_token(layers=28, kv_heads=2, head_dim=128, dbytes=2):
    return 2 * layers * kv_heads * head_dim * dbytes / 1024  # 28.0 KB

def cache_bytes(pkv):
    """Bytes the KV cache itself holds, read straight off the cache tensors."""
    if hasattr(pkv, "key_cache"):        # transformers returns a Cache object
        tensors = list(pkv.key_cache) + list(pkv.value_cache)
    else:                                # legacy tuple of (k, v) per layer
        tensors = [t for layer in pkv for t in layer]
    return sum(t.numel() * t.element_size() for t in tensors)

def measure_kv(context: int, new_tokens: int = 256):
    torch.cuda.empty_cache(); gc.collect()
    torch.cuda.reset_peak_memory_stats()
    enc = tok(prompt_of_len(context), return_tensors="pt").to("cuda")
    before = torch.cuda.memory_allocated()
    out = model.generate(**enc, max_new_tokens=new_tokens, do_sample=False,
                         use_cache=True, return_dict_in_generate=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    total_tokens = out.sequences.shape[1]  # prompt + generated, the full KV span
    return {
        "context": context,
        "total_tokens": int(total_tokens),
        # what the whole generation cost, cache and activations together
        "peak_kb_per_token": round((peak - before) / total_tokens / 1024, 1),
        # the cache on its own, which is what the formula predicts
        "kv_kb_per_token": round(cache_bytes(out.past_key_values) / total_tokens / 1024, 1),
    }

formula = kv_formula_kb_per_token()
print("formula KB/token:", formula)
kv_rows = [measure_kv(c) for c in [512, 2048, 4096]]
for r in kv_rows:
    print(r, "  vs formula", formula, "KB/token")

# the green check compares the cache itself, not the whole-generation peak
import json
with open("kv_check.json", "w") as f:
    json.dump({"formula_kb_per_token": formula,
               "measured_kb_per_token": kv_rows[-1]["kv_kb_per_token"],
               "peak_kb_per_token": kv_rows[-1]["peak_kb_per_token"]}, f)
```

Read the two numbers against each other, because the gap between them is the
lesson.

`kv_kb_per_token` should land on 28.0, and not approximately: exactly, at every
context length you tried. You predicted it on the card this morning from four
integers and no measurement, and the card was right. That is the payoff of an
arithmetic prediction over an intuition.

`peak_kb_per_token` reads two to three times higher, and climbs with context.
That is not the cache. It is activations and allocator workspace during prefill,
which also scale with sequence length, riding on top of the cache in the same
measurement. Sunday's Cell 9 measured exactly this compound number with
`memory_reserved` and honestly labelled it "KV + activations"; today you have
the instrument to pull the two apart.

Take the discipline, not just the number: the same peak can be read as either
figure, and which one is right depends on the question. "What must I budget per
concurrent user?" wants the cache. "Will this request OOM the card?" wants the
peak. At 4096 the cache alone is about 0.11 GB, small for one request, but
multiply by many concurrent users and it is the region that fills the card.
That region is what PagedAttention exists to manage, which is tomorrow.

If `kv_kb_per_token` is not 28.0, something is genuinely wrong rather than
merely noisy: `use_cache` was false, or the model is not the one the formula
describes (see failure modes).

### Cell 4: hand-rolled static batching (about 30 min)

Pad and batch 1, 4, and 8 prompts through one generate() call. Measure per
request latency and total throughput, and watch the straggler drag the batch.

Two things make this measurement honest, and both matter tomorrow.

It is a **queue**, not one batch. A single batched call has nothing to wait for.
A server has a line of requests and must decide how to group them.

The requests have **mixed output lengths**, and that is where a straggler comes
from. Every sequence in a static batch decodes in lockstep, so a long *prompt*
only wastes prefill, while a long *output* holds the whole batch for the entire
decode. Decode is most of the time, so output length is the tax that matters.

```python
# 24 requests: 18 that want 32 tokens, 6 that want 256. 2112 useful tokens.
QUEUE = [32, 32, 32, 256] * 6

def static_queue(batch: int, prompt: str = "Explain what an inference server does."):
    """A server WITHOUT continuous batching: a batch starts, nothing new joins
    until every member has finished, so it runs until its SLOWEST member."""
    t0 = time.time(); useful = 0; slots = 0
    for i in range(0, len(QUEUE), batch):
        chunk = QUEUE[i:i + batch]
        n = max(chunk)                    # the batch runs until the slowest
        enc = tok([prompt] * len(chunk), return_tensors="pt",
                  padding=True).to("cuda")
        model.generate(**enc, max_new_tokens=n, do_sample=False)
        useful += sum(chunk)              # tokens anyone actually asked for
        slots += n * len(chunk)           # token-slots the GPU actually decoded
        # accounting note: this counts REQUESTED tokens. Greedy decoding on
        # these prompts runs to the max_new_tokens cap, so requested equals
        # generated here; tomorrow's vLLM client counts the server's own
        # completion_tokens, and its README says so. Same convention, stated.
    dt = time.time() - t0
    return {"batch": batch, "wall_s": round(dt, 2),
            "tokens_per_s": round(useful / dt, 1),
            "slot_efficiency": round(useful / slots, 3)}

batch_rows = {}
for n in [1, 4, 8]:
    r = static_queue(n)
    batch_rows[str(n)] = r["tokens_per_s"]
    print(r)
```

Throughput still rises from batch 1 to batch 8, because the GPU does more useful
work per step when sequences share it. Read `slot_efficiency` next to it, because
that is the number the day is about.

At batch 1 it is 1.0: each request runs alone and every decoded token is a token
somebody wanted. The moment you batch mixed lengths it collapses, to roughly a
third, and it does not recover as the batch grows. Two thirds of what the GPU
decodes is short requests sitting finished in a slot they cannot release, waiting
for the 256-token member. That is the straggler tax, it is now a number rather
than an assertion, and it is exactly what continuous batching removes tomorrow.

Notice also what it does to scaling. Uniform-length batching would have taken you
about 5.6x from batch 1 to batch 8; with a realistic mixed queue you should see
roughly half that. The missing half is the ceiling in the morning's "static
batching and its ceiling" slide. Write your own 1-to-8 multiple on the card:
tomorrow you compute vLLM's, and the gap between the two multiples is the
engine's real contribution.

### Cell 5: export baselines.json AND download it (about 15 min)

This is a numbered step, not a remark. Tomorrow's A/B compares vLLM against these
exact numbers, and tomorrow you may be on a fresh runtime. If baselines.json only
lives in this runtime, a Colab drop erases your comparison. Write it, then
download it to your own machine.

```python
import json
baselines = {
    "model": MODEL,
    "dtype": "fp16",
    "ttft_s": ttft_by_len,                    # by prompt length
    "tpot_s": measure_stream(prompt_of_len(512))["tpot_s"],
    "batch": {k: v for k, v in batch_rows.items()},  # tokens_per_s at 1,4,8
}
with open("baselines.json", "w") as f:
    json.dump(baselines, f, indent=2)
print(json.dumps(baselines, indent=2))
```

Now download it. Do not skip this line:

```python
from google.colab import files
files.download("baselines.json")
```

Keep the downloaded file. Tomorrow you upload it back.

## Verify (green check)

Paste `verify_cell.py` as the last cell and run it. It checks the baselines.json
schema, that TTFT is less than the total generation time, that batch-8 throughput
beats batch-1, and that your measured KV is within a factor of two of the 28
KB/token formula. Expected final line:

```
GREEN CHECK: PASS
```

## Stretch

Plot the per-token timestamps from Cell 2 as a strip: a long flat gap at the
start (prefill, before the first token) then a steady comb (decode). You are
looking at prefill and decode with your own eyes.

## Failure modes

- **Measuring TTFT with streaming disabled.** Tell: your "TTFT" equals your total
  generation time and does not change much with prompt length. Fix: TTFT only
  means anything with a streamer; a plain generate() returns after the last
  token. Use `TextIteratorStreamer` and stamp the first yield.
- **Padding inflates batch cost (the static-batching tax).** Tell: batch-8 wall
  time is much larger than one request, and throughput gains are smaller than you
  hoped. This is not a bug; it is the tax. Static batching pads every sequence to
  the longest and makes all wait for the straggler. Name it in your notes; it is
  the thing continuous batching fixes.
- **KV measurement wildly off the formula.** Tell: measured KB/token is a tiny
  fraction of 28, or enormous. Fix: `use_cache=True` must be set, and you must
  `reset_peak_memory_stats()` before each measurement and read
  `max_memory_allocated()` after `torch.cuda.synchronize()`. Without the reset you
  read a stale peak from a previous run.
- **Forgetting the download (the one that hurts tomorrow).** Tell: tomorrow you
  open a fresh runtime and baselines.json is gone. Fix: run `files.download` in
  Cell 5 today and keep the file. If you did lose it, you can regenerate it by
  rerunning this lab, but that costs you tomorrow's morning.
- **Out of memory at batch 8 with long prompts.** Tell: CUDA OOM in Cell 4 at
  n=8. Fix: shorten the straggler prompt (1024 -> 768) or drop new_tokens to 96;
  the point is the straggler effect, not maxing the card.
