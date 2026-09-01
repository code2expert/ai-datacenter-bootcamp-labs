"""Device probe: the SAME code, run in two places, is the proof.
# Colab-standalone sibling of serving-stack/app/generate_probe.py (that one is
# an argparse module run as `python -m app.generate_probe`); output format differs
# on purpose - this one prints for a notebook, that one for a container log.

Run it inside your CPU-fallback container (it reports cuda: false) and on a
Colab T4 (it reports cuda: true). The point of week 2 day 4 is that one image
and one script behave correctly on both a GPU-less machine and a GPU machine.

It loads the model, reports the device it landed on, times a fixed generation,
and writes gpu_evidence.json:

    {"cuda": true|false, "device_name": "...", "tokens_per_s": <float>}

On Colab, download the gpu_evidence.json it writes and drop it next to your repo
so the local verifier can read it for part 3 of the green check.
"""
from __future__ import annotations

import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

# The CPU-fallback pattern: auto-detect the device. The SAME line picks cuda on a
# GPU box and cpu on a GPU-less box. This is what lets one image run everywhere.
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    device_name = torch.cuda.get_device_name(0)
else:
    device = "cpu"
    dtype = torch.float32
    device_name = "cpu"

print(f"device: {device} ({device_name}), dtype: {dtype}")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype)
model.to(device)
model.eval()

msgs = [{"role": "user", "content": "Explain what a GPU does, in three sentences."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)

# warm-up (not timed)
with torch.no_grad():
    model.generate(ids, max_new_tokens=8)

if device == "cuda":
    torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=128, do_sample=False)
if device == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t0

generated = out.shape[1] - ids.shape[1]
tokens_per_s = generated / dt

evidence = {
    "cuda": device == "cuda",
    "device_name": device_name,
    "tokens_per_s": round(tokens_per_s, 1),
    "model": MODEL_ID,
}

with open("gpu_evidence.json", "w") as f:
    json.dump(evidence, f, indent=2)

print(json.dumps(evidence, indent=2))
print("wrote gpu_evidence.json")
