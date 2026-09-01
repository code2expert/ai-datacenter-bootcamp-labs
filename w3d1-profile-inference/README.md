# Lab W3D1: profile inference on a real GPU

Start:      wk2-fastapi checkpoint. A fresh Colab T4 runtime. The shared
            scaffold at `../shared/colab_scaffold.py`.
Objective:  Measure how resident VRAM and GPU utilisation actually behave for
            one model across two dtypes and three context lengths, and see why
            "the GPU is busy" is not the same as "the GPU is working hard".

Time: about 3 hours. Steps carry a rough clock so you can pace it.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Fill this in before you run anything. Use only what you know from this morning
and from week 2 (memory is roughly parameters times bytes per parameter; the KV
cache grows with context).

- Qwen2.5-1.5B-Instruct is about 1.5 billion parameters. At fp16 (2 bytes each)
  the weights alone are about ______ GB. At int8 (1 byte each) about ______ GB.
- Resident VRAM at 512 context, fp16: ______ GB. At 4096 context, fp16:
  ______ GB. (Which is larger, and by roughly how much?)
- During a single-request decode (one prompt, generating tokens one at a time),
  GPU utilisation will read about ______ percent.
- Hand in the card before you open Colab.

The last two are the interesting ones. Hold onto your utilisation guess.

## The delta

You will load the model directly with transformers and bitsandbytes and profile
it. There is no vLLM server today; today is about reading the card, not serving.

### Cell 1: install pins and the sampler (about 3 min, mostly download)

Paste the **pins and installer** cell from `../shared/colab_scaffold.py`, then
paste **INSTALL CELL A** from the same file. Cell A is the profiling set:

```python
pip_install(
    f"transformers=={TRANSFORMERS_PIN}",
    f"accelerate=={ACCELERATE_PIN}",
    f"bitsandbytes=={BITSANDBYTES_PIN}",
)
```

Note what is missing: there is no vLLM today, and that is deliberate. Colab
already ships a working torch and today you load the model with transformers,
so Colab's torch is the torch. Installing vLLM here would swap it for vLLM's
older build and drag numpy back to 1.26, and Colab's preinstalled extensions are
compiled against numpy 2. The model load then fails with a `numpy.dtype size
changed` binary-incompatibility error that looks nothing like its cause. Take
Cell A, not Cell B.

Then paste the **nvidia-smi sampler thread** cell. You will use
`start_sampler()`, `stop_sampler()`, and `read_util_mean()`.

Confirm you have a GPU:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# expect: Tesla T4, 15360 MiB
```

If that prints no GPU, you have a CPU runtime: Runtime -> Change runtime type ->
T4 GPU, then rerun. If Colab denies a GPU, use the Kaggle fallback in
`../shared/README.md`.

### Cell 2: the measurement helper (about 10 min to read and paste)

This is the core of the lab. It loads the model at a given dtype, runs one fixed
generation at a given context length with the sampler running, and returns a row.
Read it before you run it; you are being asked to understand what each number
means.

```python
import time, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

def load(dtype: str):
    if dtype == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16, device_map="cuda")
    if dtype == "int8":
        qc = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=qc, device_map="cuda")
    raise ValueError(dtype)

def make_prompt(context_tokens: int) -> str:
    # a filler prompt padded to about context_tokens input tokens
    base = "Summarise the following text in one sentence.\n"
    filler = ("The data center runs many small inference requests all day. " * 400)
    ids = tok(base + filler)["input_ids"][:context_tokens]
    return tok.decode(ids)

def resident_vram_gb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / (1024 ** 3)

def profile(model, dtype: str, context: int, new_tokens: int = 128, batch: int = 1):
    prompt = make_prompt(context)
    prompts = [prompt] * batch
    enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
    # warm-up (compile/allocate), not measured
    _ = model.generate(**enc, max_new_tokens=8, do_sample=False)
    vram = resident_vram_gb()
    start_sampler()
    t0 = time.time()
    out = model.generate(**enc, max_new_tokens=new_tokens, do_sample=False)
    dt = time.time() - t0
    stop_sampler()
    gen_tokens = (out.shape[1] - enc["input_ids"].shape[1]) * batch
    return {
        "dtype": dtype,
        "context": context,
        "vram_gb": round(vram, 3),
        "util_mean": round(read_util_mean(), 1),
        "tokens_per_s": round(gen_tokens / dt, 1),
    }

def free_vram():
    """Hand freed memory back to the driver.

    Delete the model variable yourself first, in the cell, with `del model`.
    Passing it to a helper does not work: the helper deletes its own local name
    while your notebook variable still holds the weights, so nothing is freed
    and empty_cache() has nothing to give back.
    """
    gc.collect()
    torch.cuda.empty_cache()
```

Note `tok.padding_side` and a pad token: if generation complains about padding,
set `tok.pad_token = tok.eos_token` and `tok.padding_side = "left"`.

### Cell 3: the matrix (about 40 min, this is the bulk)

Two dtypes, three context lengths, one row each. Load a dtype once, profile it at
all three contexts, then unload before the next dtype so int8 and fp16 do not sit
in memory together.

```python
rows = []
for dtype in ["fp16", "int8"]:
    model = load(dtype)
    for context in [512, 2048, 4096]:
        row = profile(model, dtype, context)
        print(row)
        rows.append(row)
    del model      # without this the next dtype loads on top of this one
    free_vram()
```

Watch what prints. VRAM should climb with context. fp16 VRAM should sit above
int8 VRAM at the same context. Utilisation during single-request decode will
read lower than you probably guessed: decode generates one token at a time and
spends most of each step waiting on memory, not computing. That gap is the whole
point of the afternoon.

### Cell 4: the utilisation-vs-busy experiment (about 25 min)

Utilisation as nvidia-smi reports it tells you the GPU had work in the queue. It
does not tell you the work was efficient. Prove it: same model, same everything,
batch 1 versus batch 8. Both numbers rise, but nothing like together: watch the
ratio of the ratios. Throughput should multiply several times over while
utilisation climbs by well under half of that.

```python
model = load("fp16")
b1 = profile(model, "fp16", 512, new_tokens=128, batch=1)
b8 = profile(model, "fp16", 512, new_tokens=128, batch=8)
del model
free_vram()
print("batch 1:", b1)
print("batch 8:", b8)
print("tokens/s ratio:", round(b8["tokens_per_s"] / b1["tokens_per_s"], 2))
print("util delta:", round(b8["util_mean"] - b1["util_mean"], 1))
```

Divide one ratio by the other. Batch 8 does several times the work per unit time
for a utilisation reading that is nowhere near several times higher, so the two
numbers are not measuring the same thing. Utilisation counts intervals in which
the GPU had at least one kernel resident. It cannot tell a step that saturated
the card from a step that moved a few bytes and waited, and single-request decode
is mostly the latter. Utilisation is the number to distrust: it says busy, not
productive. Note both ratios on your card, because "we were at 90% utilisation"
is exactly the sentence that will be used on you later to argue a GPU is full. Tuesday's engine swap (day 3) is largely about turning that idle-busy
into real throughput.

Save the two tokens/s numbers so the green check can read them:

```python
import json
with open("batch_check.json", "w") as f:
    json.dump({"batch1_tokens_per_s": b1["tokens_per_s"],
               "batch8_tokens_per_s": b8["tokens_per_s"]}, f, indent=2)
```

### Cell 5: write profile.json (about 10 min)

Write the six matrix rows to `profile.json`.

```python
import json
with open("profile.json", "w") as f:
    json.dump(rows, f, indent=2)
print("wrote", len(rows), "rows to profile.json")
```

Each row is `{dtype, context, vram_gb, util_mean, tokens_per_s}`. Keep the batch
experiment numbers in your notes; the green check reads `profile.json`.

## Verify (green check)

Paste `verify_cell.py` as the last cell and run it. It checks the schema and the
sanity rules (VRAM rises with context; fp16 uses more memory than int8; and the
batch-8 tokens/s you record beats batch-1). Expected final line:

```
GREEN CHECK: PASS
```

If it fails, the parenthesis names the rule that broke. Fix the run, not the
file.

## Stretch

Add int4 (`BitsAndBytesConfig(load_in_4bit=True)`) as a third dtype and a fourth
context. Watch int4 use even less memory than int8, and confirm it does not run
faster: at this stage quantisation buys memory, not speed. That is the hook for
Wednesday, when fused kernels change the story.

## Failure modes

- **Sampler thread left running (doubles your entries).** Tell: `gpu_samples.csv`
  has interleaved rows, `util_mean` looks wrong, or `read_util_mean` returns a
  smeared average. Fix: always `stop_sampler()` before the next `start_sampler()`.
  The scaffold's `start_sampler()` refuses to start a second thread, but if you
  edited it, check for a stray one and restart the runtime if unsure.
- **int8 load fails after a runtime reset.** Tell: `ImportError` or a CUDA error
  from bitsandbytes on the int8 `load()`. Fix: bitsandbytes did not survive the
  reset; rerun Cell 1 to reinstall it, then retry. int8 needs bitsandbytes and
  accelerate present.
- **Context 4096 sits near the fp16 ceiling.** Tell: 4096 fp16 uses noticeably
  more VRAM and the margin to 15 GB looks thin. That is expected tightness, not a
  bug; note the number. The closeness is Sunday's lesson: serving fills the spare
  memory, and 4k contexts eat into it fast.
- **OOM when both dtypes are loaded at once.** Tell: CUDA out of memory on the
  int8 load. Fix: you skipped the `del model` line between dtypes. Deleting the
  variable is what frees the weights; `free_vram()` on its own only returns
  already-freed blocks to the driver. Restart the runtime and run the matrix
  loop as written.
- **Padding error on batch generation.** Tell: generate() raises about a missing
  pad token or wrong padding side. Fix: `tok.pad_token = tok.eos_token` and
  `tok.padding_side = "left"` before the batch profile.

## Before you close the tab

Colab runtimes vanish; artifacts in them vanish too. Last cell, every day:

```python
from google.colab import files
for f_ in ["profile.json", "batch_check.json"]:
    files.download(f_)
```
