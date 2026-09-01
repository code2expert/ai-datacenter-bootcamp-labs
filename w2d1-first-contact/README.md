# Lab W2D1: first contact

Start:      A fresh fork of the `serving-stack` repo and a Colab notebook on a T4 runtime.
Objective:  Measure how model memory actually behaves on a real GPU, at three precisions, and check it against the morning's formula.

This lab runs entirely in a Colab notebook on a T4 GPU. The morning taught one
multiplication: memory is about parameters times bytes per parameter. This
afternoon you load a real model three ways and watch the number move.

Model: `Qwen/Qwen2.5-1.5B-Instruct` (1.5 billion parameters, about 3.1 GB of
weights at fp16). GPU: the free Colab T4, 16 GB total, about 15 GB usable.

Before you touch the GPU, set the runtime type. Runtime menu, Change runtime
type, Hardware accelerator: T4 GPU. If the runtime is CPU, every cell below that
touches CUDA will fail with a plain "no GPU" error.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

**Submit this before you raise a runtime.** Not after Cell 1, not "roughly in
your head", not while the model is loading. Written down, handed in, then you
start.

**How it is marked: you get the credit for submitting it. The numbers are never
marked right or wrong.** That is deliberate and it is not generosity. Guessing
and missing is what makes the measurement stick; a prediction you hedged to
protect a score teaches you nothing. In the study this lab's shape comes from,
roughly three quarters of predictions were wrong and that group still learned
substantially more than students who simply watched the same demonstration.
Students who only watched scored no better than students who never saw it at all.

So commit to a real number. If you are unsure, commit to the number you would
defend out loud, and be ready to say why.

Use only the morning's formula: weights are about parameters times bytes per
parameter, and serving adds 1 to 2 GB of overhead on top.

- fp16 is 2 bytes per parameter. 1.5B times 2 bytes is _____ GB of weights.
- int8 is 1 byte per parameter, so _____ GB of weights.
- int4 is about 0.5 bytes per parameter, so _____ GB of weights.
- Loaded and resident on the GPU, add roughly _____ GB of overhead to each.
- Would this model fit at fp32 (4 bytes per parameter) on a 15 GB T4? _____

Write these down. The point of the lab is the gap between what you wrote and
what the card actually shows.

## The delta

The notebook is ten cells. Paste each one into its own Colab cell, in order,
and run it. Each cell is self-contained given the ones above it.

### Cell 1: install (about 2 minutes)

`transformers` and `accelerate` for loading, `bitsandbytes` for int8 and int4.
Pin per `../../../PINS.md` (the week-3 GPU table lists `bitsandbytes`; the CPU
table lists `transformers` and `accelerate`). Colab already ships a working
torch, so do not install a second one.

```python
# Cell 1
# versions mirror ../../../PINS.md; if a pin changes, change it there first
!pip -q install "transformers==4.46.*" "accelerate==1.1.*" "bitsandbytes==0.49.2"
```

### Cell 2: baseline the T4 (about 1 minute)

Before any model loads, see what is already resident and why. A clean CUDA
context is not zero: the driver and the CUDA runtime reserve a few hundred MB
the moment torch first talks to the GPU.

```python
# Cell 2
import torch, subprocess

assert torch.cuda.is_available(), "no GPU: set Runtime > Change runtime type > T4 GPU"

gpu_name = torch.cuda.get_device_name(0)
total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print("gpu:", gpu_name)
print("total VRAM: %.2f GB" % total_gb)

# nvidia-smi is the ground truth the whole course reads.
print(subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip())
```

Note the baseline used number. Everything you measure later is on top of this.
If `memory.used` is already several GB, someone else's tensors are resident from
an earlier cell; use Runtime, Restart session and rerun from Cell 1.

### Cell 3: the measurement helper (given)

This is the measurement cell. It reports resident VRAM the same way every time,
so your three loads are comparable. Read it once, then reuse it.

```python
# Cell 3
import torch, time, json, gc

def measured_vram_gb():
    """GB held by PyTorch's caching allocator for this process.

    Not the same as the card being full. The CUDA context the driver sets up,
    several hundred MB of it, sits outside this number and inside the one
    nvidia-smi prints. That difference is the point of Cell 2.
    """
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved(0) / 1e9

def free_vram():
    """Hand freed memory back to the driver.

    Delete the model variable yourself first, in the cell, with `del model_x`.
    Python frees an object when the last reference to it goes away, and your
    notebook variable is a reference. No helper can delete a name that lives in
    your cell, so this function only does the second half of the job.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
print("helper ready for", MODEL_ID)
```

### Cell 4: load fp16, measure, explain the gap (about 2 minutes, first run downloads weights)

Load the model in fp16 and measure. The first run downloads about 3.1 GB of
weights; on Colab that is a minute or two. Compare the measured number to your
fp16 prediction and write down why they differ.

```python
# Cell 4
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_ID)

before = measured_vram_gb()
model_fp16 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
after = measured_vram_gb()

fp16_measured = after
print("fp16 resident VRAM: %.2f GB" % fp16_measured)
print("delta from before-load: %.2f GB" % (after - before))
```

Expect roughly 3.2 to 3.6 GB. The weights alone are 3.1 GB, so the gap above
your "just weights" prediction is small, real and nameable: the allocator rounds
up to its block size, and loading leaves working buffers behind. Not the CUDA
context, and not the tokenizer. The context is the driver's and does not appear
in this number; the tokenizer lives in system RAM and costs no VRAM at all.

Now compare this figure with the nvidia-smi reading from Cell 2. That one is
larger, and the extra is the CUDA context you just excluded. Two instruments,
two honest numbers, and knowing which answers which question is the skill.

Write the gap and both causes on your card. This is the "explain the gap" step
from the briefing.

### Cell 5: reload at int8 (about 1 minute)

Free fp16 first, then load int8 through bitsandbytes. Measure.

```python
# Cell 5
from transformers import BitsAndBytesConfig

del model_fp16   # your reference to the weights; without this line nothing frees
free_vram()

model_int8 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="cuda")
int8_measured = measured_vram_gb()
print("int8 resident VRAM: %.2f GB" % int8_measured)
```

### Cell 6: reload at int4, complete the table (about 1 minute)

```python
# Cell 6
del model_int8
free_vram()

model_int4 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="cuda")
int4_measured = measured_vram_gb()
print("int4 resident VRAM: %.2f GB" % int4_measured)
```

You now have three measured numbers. Work out the observed bytes per parameter
for each: measured weight bytes divided by 1.5e9. It will not be exactly 2, 1,
0.5. Note why: the number you measured includes overhead that is roughly fixed,
so the smaller the model the more that fixed part inflates the per-parameter
figure. int4 also keeps some tensors (layernorms, embeddings) at higher
precision. Both are real, not measurement error.

### Cell 7: the generation script (about 1 minute)

Write and save the generation script as a file in your repo. It loads the model
at a chosen dtype, generates a fixed number of tokens, and reports tokens per
second. This file is a graded deliverable, so it must exist on disk.

```python
# Cell 7
generate_src = '''
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

def load(dtype):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if dtype == "fp16":
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    elif dtype == "int8":
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="cuda")
    elif dtype == "int4":
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="cuda")
    else:
        raise ValueError(dtype)
    return tok, m

def tokens_per_s(dtype, new_tokens=128):
    tok, m = load(dtype)
    msgs = [{"role": "user", "content": "Explain what a GPU does, in three sentences."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    m.generate(**{"input_ids": ids}, max_new_tokens=8)  # warm-up, not timed
    torch.cuda.synchronize()
    t0 = time.time()
    out = m.generate(**{"input_ids": ids}, max_new_tokens=new_tokens, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    generated = out.shape[1] - ids.shape[1]
    return generated / dt

if __name__ == "__main__":
    for d in ["fp16", "int8", "int4"]:
        print(d, "%.1f tok/s" % tokens_per_s(d))
'''

with open("generate.py", "w") as f:
    f.write(generate_src)
print("wrote generate.py")
```

### Cell 8: measure tokens per second at each precision (about 2 minutes)

Run the timing for all three. Record the numbers.

```python
# Cell 8
import importlib.util
spec = importlib.util.spec_from_file_location("gen", "generate.py")
gen = importlib.util.module_from_spec(spec)

# free the int4 model from Cell 6 so the script loads cleanly
del model_int4
free_vram()
spec.loader.exec_module(gen)

# Inside tokens_per_s the model is a local variable, so it frees itself when the
# function returns. Here free_vram() alone is enough; there is no name to delete.
fp16_tps = gen.tokens_per_s("fp16"); print("fp16 %.1f tok/s" % fp16_tps)
free_vram()
int8_tps = gen.tokens_per_s("int8"); print("int8 %.1f tok/s" % int8_tps)
free_vram()
int4_tps = gen.tokens_per_s("int4"); print("int4 %.1f tok/s" % int4_tps)
```

Expected and important: int8 and int4 generate SLOWER than fp16 here, not
faster. bitsandbytes quantises to save memory, and its dequantise-on-the-fly
kernels cost time on every step. Quantisation buys memory this week, not speed.
Week 3 swaps in an engine with fused kernels and pays this back. Do not treat
the slowdown as a bug; write it down as expected and park the why.

You should also find that int8 is the slowest of the three, slower than int4,
which looks backwards if you assume more quantisation means more work. It is a
kernel story, not an arithmetic one: the two paths use different bitsandbytes
kernels, and the int8 one carries more per-step overhead on this card. Note the
ordering you measure and carry the question into week 3.

Also note the generation quality: at int4 the answer is usually still coherent
but can drift. That is the quality cost the morning mentioned.

### Cell 9: grow the context, watch it take memory (about 2 minutes)

Load fp16 once, then generate with a growing prompt and watch resident VRAM
climb. Two things grow together here, which is why the print says "KV +
activations". The KV cache holds a slice of GPU memory for every token in the
context: 2 times layers times kv_heads times head_dim times bytes per token,
about 28 KB per token for this model. Activations and allocator workspace grow
alongside it, and at these context lengths they are the larger share of what you
will see.

So do not expect your delta to divide out to 28 KB per token. It will read
several times that, and that is not a measurement error. The formula gives you
the floor the cache alone imposes, not the total the card gives up. This cell
measures the whole process with `memory_reserved`, which is the honest number
for "what fills the card" and the wrong instrument for isolating one term in it.
Week 3 day 2 separates them properly, against `memory_allocated` and the full
token span. Today, watch the direction and the size of the climb.

```python
# Cell 9
free_vram()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
base = measured_vram_gb()
print("weights only: %.2f GB" % base)

for ctx in [256, 1024, 3072]:
    prompt = "word " * ctx
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    m.generate(**{"input_ids": ids}, max_new_tokens=64, do_sample=False)
    peak = torch.cuda.max_memory_reserved(0) / 1e9
    print("ctx ~%d tokens -> peak VRAM %.2f GB (KV + activations: %.2f GB)" % (ctx, peak, peak - base))
    torch.cuda.reset_peak_memory_stats(0)
```

The delta over weights-only grows with context length. That amber region is
exactly what vLLM exists to manage, which is the week-3 story.

### Cell 10: write results.json and fill the table

Write the machine artifact the verifier reads, and fill the human table below in
this README (in your repo copy). Both are graded: the JSON for the green check,
the table for you.

```python
# Cell 10
import json

results = {
    "model": MODEL_ID,
    "gpu": gpu_name,
    "measurements": [
        {"dtype": "fp16", "predicted_gb": None, "measured_gb": round(fp16_measured, 2), "tokens_per_s": round(fp16_tps, 1)},
        {"dtype": "int8", "predicted_gb": None, "measured_gb": round(int8_measured, 2), "tokens_per_s": round(int8_tps, 1)},
        {"dtype": "int4", "predicted_gb": None, "measured_gb": round(int4_measured, 2), "tokens_per_s": round(int4_tps, 1)},
    ],
}

# fill predicted_gb from your prediction card (weights only is fine):
results["measurements"][0]["predicted_gb"] = 3.0   # <- your fp16 estimate
results["measurements"][1]["predicted_gb"] = 1.5   # <- your int8 estimate
results["measurements"][2]["predicted_gb"] = 0.75  # <- your int4 estimate

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
```

Fill this table in your repo copy of the README from your own numbers:

| dtype | predicted GB | measured GB | observed bytes/param | tokens/s |
|---|---|---|---|---|
| fp16 |  |  |  |  |
| int8 |  |  |  |  |
| int4 |  |  |  |  |

Observed bytes per parameter is measured weight bytes divided by 1.5e9. Commit
`results.json`, `generate.py`, and this filled-in README to your repo.

### The second half of the afternoon: the budget solver (about 90 min, by hand)

The measurement above is short by design; the rest of the afternoon is
`../extra-d1-memory-budget/` - promoted from optional to the day's second act.
No GPU, no network: from five real models' config fields, compute which
model+precision fits a 16 GB budget at 4 users, and prove it with that lab's
own `verify.py`. Bring today's measured bytes-per-parameter numbers with you;
they are the constants the solver leans on.

## Verify (green check)

Download `results.json` and `generate.py` from Colab into the same folder, then
paste `verify_cell.py` into a fresh Colab cell (it reads the two files you
wrote). It validates the schema, checks the ranges are sane, and confirms the
generation script exists. The last line it prints is exactly one of:

```
GREEN CHECK: PASS
GREEN CHECK: FAIL (<reason>)
```

An unattempted lab scores zero. The verifier does not care about your prose,
only that the numbers are present and physically sane.

## Stretch

OOM the T4 on purpose, then explain the failure with the morning's formula.

```python
# Cell S (stretch)
del m            # the fp16 model Cell 9 left resident
free_vram()
# fp32 is 4 bytes per parameter: 1.5B x 4 = ~6 GB weights, plus a fat context.
m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32, device_map="cuda")
huge = "word " * 20000
ids = tok(huge, return_tensors="pt").input_ids.to("cuda")
m.generate(**{"input_ids": ids}, max_new_tokens=2000)  # expect CUDA out of memory
```

When it dies, write the one-line explanation: fp32 weights are about 6 GB, and a
20k-token context at fp32 KV is several GB more; the sum crosses 15 GB and the
allocator cannot place the next tensor. The formula predicted this before you
ran it.

## Failure modes

- Runtime is on CPU, not T4. Tell: Cell 2 asserts "no GPU". Fix: Runtime,
  Change runtime type, T4 GPU, then rerun from Cell 1.
- Out of memory on a load you did not expect. Tell: `CUDA out of memory` on
  Cell 4 or 5. Cause: a previous model is still resident. Fix: you skipped the
  `del model_x` line before `free_vram()`. Deleting the variable is the part
  that frees the weights; `free_vram()` only returns already-freed blocks to the
  driver. If you are unsure what is still bound, Runtime, Restart session and
  rerun from Cell 1.
- Cell 1 prints a red `ERROR: pip's dependency resolver...` block about `gradio`
  and `huggingface-hub`. Expected and harmless: pinning transformers pulls an
  older hub than Colab's preinstalled gradio wants, and this lab never imports
  gradio. Keep going.
- `bitsandbytes` import or CUDA error on Cell 5. Tell: an error mentioning
  `bitsandbytes` or a missing CUDA symbol. Cause: install did not finish or a
  stale torch. Fix: rerun Cell 1, restart the session, run in order.
- int8 or int4 slower than fp16 and it looks wrong. It is not wrong. bitsandbytes
  trades speed for memory; this is the expected result and a week-3 hook.
- Colab GPU quota exhausted mid-lab. Tell: "cannot connect to a GPU backend".
  Fix: switch to a teammate's account, or use Kaggle Notebooks (30 GPU hours a
  week) with the same cells; the fallback is named for exactly this.
