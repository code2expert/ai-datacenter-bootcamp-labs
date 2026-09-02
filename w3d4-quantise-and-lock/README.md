# Lab W3D4: quantise and lock the model

Start:      wk3-vllm in progress (vLLM serving from day 3). A fresh Colab T4
            runtime. The shared scaffold at `../shared/colab_scaffold.py`.
Objective:  Serve the AWQ build, compare it against yesterday's fp16 numbers,
            run the function-calling smoke test that gates the choice, and record
            the team's locked model, flags, and smoke score.

Time: about 3 hours.

The model your team locks today is the model you serve for the rest of the
course. After this week, what you serve does not change; only how you operate it
does. So the smoke test is the real gate: an endpoint the agentic cohort cannot
get reliable tool calls from is useless to them.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Fill this in before you run anything.

- AWQ stores weights at 4-bit, roughly a quarter of fp16's bytes. With the same
  `--gpu-memory-utilization 0.85`, `nvidia-smi memory.used` versus yesterday
  will read ______ (much lower / about the same). Commit to one.
- Tokens/s served by vLLM with the AWQ kernels will be
  ______ (faster / slower / about the same) than fp16. Note: this is vLLM with
  fused AWQ kernels, not day-1 bitsandbytes; the speed story is different.
- The FC smoke test fires 10 attempts across 3 prompts: 8 want a tool call,
  2 must stay call-free. Your candidate will return valid parseable tool_calls
  on about ______ of the 8 that want one.
- Hand in the card.

## The delta

### Cell 1: install pins and serve AWQ (about 38 min: ~30 of silent install + the AWQ download - start it, fill your prediction card while it runs)

Paste the **pins and installer** cell from `../shared/colab_scaffold.py`,
then **INSTALL CELL B**, and add
autoawq:

```python
pip_install(
    f"vllm=={VLLM_PIN}",
    f"transformers=={TRANSFORMERS_PIN}",   # required: see the scaffold's note
    f"accelerate=={ACCELERATE_PIN}",
    f"autoawq=={AUTOAWQ_PIN}",
    f"httpx=={HTTPX_PIN}",
    f"openai=={OPENAI_PIN}",
)
```

Paste the **launch server** cell, but serve the AWQ build with the AWQ flag and
the tool-call parser. AWQ has its own weights, so the first launch downloads them
(cold cache). Override SERVER_ARGS before you call launch:

```python
SERVER_ARGS = {
    "--model": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    "--dtype": "half",
    "--max-model-len": "4096",
    "--gpu-memory-utilization": "0.85",
    "--port": "8000",
    "--quantization": "awq",
    "--enable-auto-tool-choice": None,        # bare flag
    "--tool-call-parser": "hermes",           # Qwen2.5 family uses hermes
}
server = launch_server(SERVER_ARGS)
```

Then paste the **health poll** cell and wait for the healthy line.

The parser flag is per model family, from PINS.md canon: Qwen2.5 and Hermes-3 use
`hermes`; Llama-3.1 uses `llama3_json`. Mismatch the parser and tool calls arrive
as prose instead of parsed `tool_calls` (a named failure mode).

### Cell 2: measure AWQ VRAM and tokens/s (about 20 min)

Reuse the async client idea from day 3 (or the OpenAI client) to get a tokens/s
number, and read VRAM from nvidia-smi. Compare against yesterday's fp16 row.

```python
!nvidia-smi --query-gpu=memory.used --format=csv,noheader
# note the number; compare to yesterday's fp16 resident VRAM
```

Most cards predict a big drop here, and most cards are wrong: on the reference
T4 run the AWQ server read **11.7 GB**, within about a gigabyte of fp16. The
weights did shrink (roughly 2.5 GB down to under 1 GB), but you told vLLM
`--gpu-memory-utilization 0.85`, and vLLM spends whatever the weights free up
on more KV-cache blocks up to that fraction. `nvidia-smi` measures the pool,
not the weights. The savings are real; they show up as capacity, not as a
lower number: check `/content/server.log` for the `# GPU blocks` line and
compare it to yesterday's, that is where the freed memory went. With vLLM's
fused AWQ kernels the tokens/s should be in the same range as fp16 or better,
unlike day 1's bitsandbytes path where quantisation cost speed. Record both
numbers, and note which prediction your card got wrong.

### Cell 3: five-prompt quality spot check (about 25 min)

Run the same five prompts against fp16 and AWQ side by side. This is judgment
recorded, not scored: you are looking for obvious degradation, and five prompts
is the minimum to see it (one prompt tells you nothing).

```python
SPOT_PROMPTS = [
    "Write a two-sentence summary of what an inference server does.",
    "A user asks for the weather in Riyadh and the time in Tokyo. "
    "What two tool calls would you make?",
    "Refactor this into a single sentence: The GPU was busy but not "
    "productive, because decode is memory-bound.",
    "List the steps to roll back a bad deployment, in order.",
    "Explain quantisation to a non-technical manager in three sentences.",
]
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
for p in SPOT_PROMPTS:
    r = client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        messages=[{"role": "user", "content": p}], max_tokens=200)
    print("PROMPT:", p[:50], "...")
    print(r.choices[0].message.content, "\n")
```

To compare against fp16, relaunch fp16 (SERVER_ARGS without `--quantization awq`,
model `Qwen/Qwen2.5-1.5B-Instruct`) and run the same five. Record your judgment:
does AWQ hold up, or does it lose the plot on any prompt.

### Cell 4: the function-calling smoke test (about 40 min)

`smoke_test.py` (next to this README) is given in full. You run it; you do not
write it. It fires 3 canonical prompts, k times each for n=10 total attempts,
and scores each attempt's behaviour - a valid `tool_calls` when one is wanted,
a clean refusal when not. The three prompts are:

1. a two-tool task (weather AND calculator) that should produce two tool calls,
2. a single-tool task that should produce one,
3. a distractor that needs NO tool and must NOT call one.

Pass is at least 8 of 10 attempts showing **correct behaviour** (a valid tool
call when one is wanted, no call on the distractor) AND the distractor staying
call-free in the majority of its attempts. A model that always calls a tool
fails the real consumer, so restraint is scored, not just obedience.

Run it against the currently served model:

```python
# paste the contents of smoke_test.py, then:
result = run_smoke(base_url="http://localhost:8000/v1",
                   model="Qwen/Qwen2.5-1.5B-Instruct-AWQ")
print(result)
```

Run it against fp16 too (relaunch fp16, rerun). Record both scores. If your
candidate scores below 8/10, fall back to the pocket known-good model (below).

### Cell 5: fill model-lock.md (about 20 min)

Open `model-lock.md` (the template next to this README) and fill every field:
the model id you lock, the exact vLLM launch flags including the parser, and the
smoke score you measured. This file is your team's record of the locked model and
is what the green check reads.

Write the smoke result to a file for the green check:

```python
import json
with open("smoke_result.json", "w") as f:
    json.dump(result, f, indent=2)
```

### The pocket known-good model

If your candidate fails the smoke test, do not spend the afternoon fighting it.
Lock the pocket known-good model:

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Flags: `--dtype half --max-model-len 4096 --gpu-memory-utilization 0.85
  --enable-auto-tool-choice --tool-call-parser hermes`

It is pre-verified to pass the smoke test. Record it in model-lock.md with its
smoke score and move on; a locked, working model beats a clever one that flakes.

### Cell 6: clean shutdown

Paste the **clean shutdown** cell to free port 8000.

## Verify (green check)

Paste `verify_cell.py` as the last cell and run it. It checks that the smoke
result shows at least 8/10 with distractor compliance, and that model-lock.md is
filled in (no template placeholders left). Expected final line:

```
GREEN CHECK: PASS
```

## Stretch A

Run the smoke test at a higher k (n=20) for a tighter estimate of the tool-call
success rate. A model that passes 8/10 but only 15/20 is riskier for a live
consumer than one that passes 19/20; the wider sample tells you which you have.

## Stretch B: the engine sets the ceiling, not the card (about 45 min)

You have spent the week believing the T4 decides which models you can serve.
That is half true, and the half that is false is worth an afternoon.

**Predict first, and write it down.** A model published this year, with an
architecture vLLM has never heard of, on your T4:

- Will your vLLM serve it? _____ (yes / no)
- If not, is that because the card is too old, or the engine is? _____
- Is there any way to run it on this exact T4 today? _____ (yes / no / don't know)

**Step 1: try it in vLLM and read the error carefully.** Pick a 2026 model whose
`config.json` declares an architecture vLLM does not implement, for example
`Nanbeige/Nanbeige4.2-3B` (`NanbeigeForCausalLM`, Apache 2.0, built for agents).
Check the architecture before you launch anything:

```python
import json, urllib.request
cfg = json.loads(urllib.request.urlopen(
    "https://huggingface.co/Nanbeige/Nanbeige4.2-3B/resolve/main/config.json").read())
print(cfg["architectures"], "| custom code:", "auto_map" in cfg)
```

Then launch it the way you launched everything else this week, adding
`--trust-remote-code`. It will fail, in under a minute, before it downloads a
single weight:

```
ValueError: Model architectures ['NanbeigeForCausalLM'] are not supported for
now. Supported architectures: dict_keys(['AquilaModel', 'AquilaForCausalLM',
'ArcticForCausalLM', 'BaiChuanForCausalLM', ...
```

Read it twice, because it is the most useful error message you will see this
week. There is **nothing in it about your GPU**. Not memory, not compute
capability, not Turing. It is a dictionary lookup that missed. The engine has a
list of architectures it knows how to build, and yours is not on it. Your T4 was
never consulted.

**Step 2: run the same model on the same card, via llama.cpp.** GGUF is not only
the tier-0 CPU format from this morning's slide. llama.cpp gives Turing a
first-class tensor-core path, and it tracks new architectures far faster than
vLLM does, because adding one is a much smaller job. llama.cpp merged native
Nanbeige4.2 support on 2026-07-27, days after the model was published.

Community GGUF builds already exist, so you do not have to convert one:
`Abiray/Nanbeige4.2-3B-GGUF` and `owao/Nanbeige4.2-3B-GGUF` (both verified
present 2026-07-28). Take a Q4 quant, roughly 2.5 GB, which is nothing against
your 15 GB card. Run it with a GPU offload rather than on CPU, so the comparison
is fair.

**Step 3: explain the gap in writing.** Two engines, one card, one model, two
different answers. Note in `model-lock.md`:

- which engine could serve it and which could not,
- whether the limit you hit was hardware or software,
- and what you would actually do if a product owner asked for this model on this
  hardware next week.

The point is a reflex you will use for the rest of your career: **when a model
will not serve, find out whether you are blocked by silicon or by software,
because those have completely different fixes.** Silicon means new hardware or a
different model. Software means a different engine, a newer version, or waiting
for an upstream merge. Engineers waste a great deal of money buying GPUs to solve
software problems.

One honest caveat to carry: llama.cpp is not a drop-in replacement for vLLM here.
You lose continuous batching, PagedAttention and most of what week 3 measured.
It is the right tool for "can this run at all", not for "can this serve a
cohort". Knowing which question you are answering is the skill.

## Failure modes

- **Parser flag mismatched to the model family.** Tell: `tool_calls` is empty and
  the tool request shows up as prose in `message.content`. Fix: Qwen2.5 and
  Hermes-3 need `--tool-call-parser hermes`; Llama-3.1 needs `llama3_json`. Match
  the parser to the model, relaunch.
- **AWQ needs its own weights download.** Tell: first AWQ launch's health poll
  runs long; the log shows it downloading `-AWQ` weights. Fix: this is the cold
  cache; wait it out or rerun the health poll once the download finishes. It is a
  different repo from the fp16 weights.
- **Judging quality on one prompt.** Tell: you declare AWQ fine (or broken) from a
  single output. Fix: the spot check is five prompts for a reason; a model can
  nail one and fumble another. Judge the set.
- **Distractor compliance ignored.** Tell: your candidate scores 8/10 on tool
  calls but also calls a tool on the distractor every time. Fix: that model fails
  the real consumer; it cannot tell when NOT to act. The smoke test counts
  distractor compliance; do not lock a model that fails it.
- **Both models loaded at once (OOM).** Tell: relaunching fp16 while AWQ is still
  up OOMs. Fix: run the clean-shutdown cell (or RECOVERY) before relaunching with
  different flags; one server at a time on the T4.

## Before you close the tab

Colab runtimes vanish; artifacts in them vanish too. Last cell, every day:

```python
from google.colab import files
for f_ in ["smoke_result.json", "model-lock.md"]:
    files.download(f_)
```
