# Integration note: <team name> (v1, go-live)

Copy this file, fill every angle bracket, and hand it to your paired Agentic AI
team. It is Part A of the cross-cohort runbook
(`../../../week-06-capstone/cross-cohort-runbook.md`); the full operating rules
for the window live there.

- **base_url** (client form, ends in `/v1` - paste into an OpenAI client):
  `https://<your-tunnel-or-pod-host>/v1`
- **service root** (no `/v1` - the runbook's triage curls and `verify.sh`
  build paths from this): `https://<your-tunnel-or-pod-host>`
- **model id:** `<exactly what /v1/models returns>`
- **auth:** bearer key, handed over <how: in person / DM to their on-call, never in this file>
- **modalities:** text in, text out, tool calls per the OpenAI schema. <If your
  pair answered anything other than "text only" at wk-2 team formation, name
  what was agreed and with whose instructor sign-off.>
- **example call:** the exact `curl` from your green check, with the key
  redacted
- **SLOs we publish:** availability <n>% over the window · TTFT p95 < <n> ms
  (tier 1) or e2e p95 < <n> ms (tier 0) · error rate < <n>%
- **limits, declared honestly:** max_tokens clamp <n> · concurrency knee ~<n>
  (from your wk-3 bench) · <anything else a caller will hit>
- **on-call:** <name> · <channel> · response within <n> minutes during the window
