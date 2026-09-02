# Model lock (team record)

Fill every field. This is your team's record of the model you serve for the rest
of the course. The green check reads this file and refuses template placeholders,
so replace every `FILL:` line with your real value.

## The locked model

- Model id: FILL: the exact Hugging Face id you serve, e.g.
  `Qwen/Qwen2.5-1.5B-Instruct-AWQ` or the pocket known-good
  `Qwen/Qwen2.5-1.5B-Instruct`
- Quantisation: FILL: `awq` / `none`
- Why this one: FILL: one sentence (passed smoke, VRAM headroom, quality held)

## The launch flags

The exact vLLM flags your team runs. Copy them from the SERVER_ARGS you launched
with.

```
FILL: --model ... --dtype half --max-model-len 4096 \
FILL: --gpu-memory-utilization 0.85 \
FILL: --enable-auto-tool-choice --tool-call-parser hermes
```

- Tool-call parser: FILL: `hermes` (Qwen2.5, Hermes-3) or `llama3_json`
  (Llama-3.1)

## The smoke score

- Score (valid behaviours out of 10): FILL: e.g. `9`
- Distractor stayed call-free in the majority: FILL: `yes` / `no`
- Passed the gate (>= 8/10 and distractor majority clean): FILL: `yes` / `no`
- Measured against: FILL: `AWQ` / `fp16` / both, with each score if you ran both

## Quality spot check note

- FILL: one or two sentences on the five-prompt side by side: did the quantised
  build hold up, or did it degrade on any prompt.
