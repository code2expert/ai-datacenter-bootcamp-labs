# Capacity note (team, one page)

Fill every field from your bench_report.json. The green check reads this file and
refuses template placeholders, so replace every `FILL:` line with your value.

## The numbers

- Locked model: FILL: the model id you benchmarked
- Target p95 end-to-end latency (your SLO today): FILL: e.g. `2.0` seconds
- Knee concurrency (highest concurrency whose p95 is still under target):
  FILL: e.g. `8`
- Tokens per second at the knee: FILL: e.g. `410`
- Max sustainable request rate at the target p95: FILL: requests per second you
  can serve at the knee before p95 crosses target, e.g. `3.2 req/s`

## The limiting family

One sentence, using this morning's triage lens (compute vs memory vs overhead):
which family limits this stack at the knee, and the tell that points to it.

- FILL: e.g. "Memory-bound: throughput flattens while GPU utilisation stays
  moderate and p95 climbs, the decode memory-bandwidth ceiling, not compute."

## Why the knee, not the peak

One sentence in your own words on why you report the knee at the SLO rather than
the peak throughput.

- FILL:
