# Integration note: Team t08 (v1, go-live)

Copy this file, fill every angle bracket, and hand it to your paired Agentic AI
team. It is Part A of the cross-cohort runbook
(`../../../week-06-capstone/cross-cohort-runbook.md`); the full operating rules
for the window live there.

- **base_url** (client form, ends in `/v1` - paste into an OpenAI client):
  `https://t08.aidc.nadir.sh/v1`

- **service root** (no `/v1` - the runbook's triage curls and `verify.sh`
  build paths from this): `https://t08.aidc.nadir.sh`

- **model id:** `Qwen/Qwen2.5-1.5B-Instruct-AWQ`

- **auth:** bearer key, handed over in person / DM to their on-call, never in this file

- **modalities:** text in, text out, tool calls per the OpenAI schema.

- **example call:** the exact `curl` from your green check, with the key
  redacted:

```bash
curl -s https://t08.aidc.nadir.sh/v1/chat/completions \
  -H "Authorization: Bearer REDACTED" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct-AWQ","messages":[{"role":"user","content":"hello from outside"}]}'
```

- **SLOs we publish:** availability 99% over the window · TTFT p95 < 300 ms
  (tier 1) · error rate < 1%

- **limits, declared honestly:** max_tokens clamp 256 · concurrency knee ~8
  (from your wk-3 bench) · max total model length 4096 tokens

- **on-call:** Noura · Discord · response within 10 minutes during the window