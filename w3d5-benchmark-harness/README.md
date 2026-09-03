# Lab W3D5: the benchmark harness

Start:      wk3-vllm with the model locked (day 4). A fresh Colab T4 runtime. The
            shared scaffold at `../shared/colab_scaffold.py`. The locked model
            and its flags from your model-lock.md.
Objective:  Run the given benchmark harness against your locked model, sweep
            concurrency, find the knee where p95 crosses your target while
            throughput flattens, and write the one-page capacity note. Publish
            tokens/s and p95 to the progress board.

Time: about 3 hours. Quiz 2 is this afternoon (see `../../quiz/quiz-2.md`).

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Fill this in before you run anything.

- As concurrency rises, throughput (tokens/s) climbs, then flattens; p95 latency
  is flat at low concurrency, then climbs. Your knee (where p95 crosses target as
  throughput stops rising) will be at about concurrency ______.
- Pick a target: p95 end-to-end latency of ______ seconds. This is your SLO for
  today.
- Hand in the card.

## The delta

The harness is given: `../../../serving-stack/bench/bench.py`. You do not write it. You
launch your locked model, run the sweep, and read the numbers.

### Cell 1: install pins and launch your locked model (about 38 min, mostly the silent install - prediction card time)

Paste the **pins and installer**, **INSTALL CELL B**, **launch server**, and
**health poll** cells from
`../shared/colab_scaffold.py`. Set SERVER_ARGS to your team's locked flags from
model-lock.md before launching (the AWQ build with the tool-call parser, or the
pocket known-good, whatever you locked yesterday). Wait for the healthy line.

### Cell 2: get the harness and the prompts (about 5 min)

`bench.py` lives in the course repo at `../../../serving-stack/bench/bench.py`. `prompts.txt`
(20 fixed prompts, next to this README) is the input set. Make both available in
the runtime (clone the repo or upload the two files).

The harness CLI contract, exactly:

```
python bench.py \
  --base-url http://localhost:8000 \
  --model <your-locked-model-id> \
  --concurrency 1,2,4,8,16 \
  --requests-per-level 20 \
  --prompt-file prompts.txt \
  --out bench_report.json
```

Output JSON is a document: {"runs": [...]}, each run carrying its "levels" list, each:
`{concurrency, tokens_per_s, ttft_p50_s, ttft_p95_s, latency_p95_s, errors}`.
The harness excludes a warm-up round and appends levels as it finishes them, so a
mid-sweep interruption does not lose the levels already done.

### Cell 3: run the sweep (about 40 min)

```python
!python bench.py \
  --base-url http://localhost:8000 \
  --model "PASTE-YOUR-LOCKED-MODEL-ID" \
  --concurrency 1,2,4,8,16 \
  --requests-per-level 20 \
  --prompt-file prompts.txt \
  --out bench_report.json
```

Watch the levels print. tokens/s should rise then flatten; `latency_p95_s` should
be low and flat at first, then climb once the server saturates. If a level errors,
the harness records the error count for that level rather than crashing the sweep.

### Cell 4: find the knee (about 30 min, by hand)

Load the report and find the knee: the concurrency where `latency_p95_s` crosses
your target while `tokens_per_s` has stopped rising meaningfully.

```python
import json
levels = json.load(open("bench_report.json"))["runs"][-1]["levels"]
for L in levels:
    print(f"c={L['concurrency']:>2}  tok/s={L['tokens_per_s']:>7.1f}  "
          f"ttft_p95={L['ttft_p95_s']:.3f}  lat_p95={L['latency_p95_s']:.3f}  "
          f"errors={L['errors']}")

TARGET_P95_S = 0.0   # <- your SLO from the prediction card
# knee: highest concurrency whose p95 is still under target
under = [L for L in levels if L["latency_p95_s"] <= TARGET_P95_S]
knee = max(under, key=lambda L: L["concurrency"]) if under else None
print("knee:", knee)
```

The knee is your answer, not the peak. The peak throughput happens past the knee,
where p95 has already blown through your target: it flatters the number by
serving requests too slowly to count. The concurrency at the SLO is the capacity
you can actually promise.

Two outcomes are legitimate, and both happened on real T4s:

- **The knee is inside your sweep**: p95 crosses your target somewhere in
  1 to 16. Report that concurrency.
- **The sweep never reaches it**: even level 16 sits under your target with
  throughput still rising (the reference T4 run measured p95 of 2.7 s at
  concurrency 16, under any everyday target). Then your knee is "16,
  sweep-bounded": you can promise at least that, and the stretch's
  `--concurrency 32` is how you find the real edge. Say which case you are in;
  a sweep-bounded knee is a finding, not a failure.

### Cell 5: fill capacity-note.md (about 25 min, by hand)

Open `capacity-note.md` (the template next to this README) and fill it: the knee
concurrency, the max sustainable request rate at your target p95, and one
sentence naming the limiting family (compute, memory, or overhead) using the
triage lens from this morning. This is a one-page note; keep it to the template.

Write the knee to a file for the green check:

```python
with open("knee.json", "w") as f:
    json.dump({"target_p95_s": TARGET_P95_S,
               "knee_concurrency": knee["concurrency"] if knee else None}, f)
```

### Cell 6: publish and shut down (about 10 min)

Publish your tokens/s at the knee and your p95 to the progress board under your team
username (or the shared sheet fallback). Then paste the **clean shutdown** cell.

Checkpoint for the week: vLLM live, numbers published, model locked. That is the
Engine Swap badge.

## Verify (green check)

Paste `verify_cell.py` as the last cell and run it. It checks the bench_report.json
schema, that at least four concurrency levels ran, that errors are zero or
explained, and that capacity-note.md is filled in. Expected final line:

```
GREEN CHECK: PASS
```

## Stretch

Rerun the sweep with `--concurrency 1,2,4,8,16,32`. See whether 32 gives you any
more real throughput or just more p95. On the reference T4 run, 16 was still
scaling almost linearly (364 to 583 tok/s from 8 to 16) - the knee for this
model on this card lives somewhere past 16, and finding where it actually stops
is the point. If your Wednesday self believed a single request measured the
card's limit, your Friday self now knows better by an order of magnitude.

## Failure modes

- **Benchmarking through a cold server.** Tell: the concurrency-1 level shows a
  huge tokens/s dip or a long first latency. Fix: the harness excludes a warm-up
  round, but it cannot warm a model that is still downloading. Wait for the health
  poll's healthy line before you start the sweep.
- **Quota death mid-sweep.** Tell: the runtime drops at, say, concurrency 8 and
  you have levels 1, 2, 4 on file. Fix: run the RECOVERY cell to bring the server
  back, then rerun the harness for only the missing levels
  (`--concurrency 8,16`); the harness appends, so you keep the earlier levels.
- **Reading the peak as capacity.** Tell: you report the highest tokens/s (at
  concurrency 16 or 32) as your capacity. Fix: that point is past the knee, where
  p95 already fails your SLO. The knee at the SLO is the answer; the peak flatters.
- **Target left at zero.** Tell: the knee comes out `None` because no level is
  under a target of 0. Fix: set `TARGET_P95_S` to your real SLO from the
  prediction card before computing the knee.
- **Wrong model id passed to the harness.** Tell: every request errors with a
  model-not-found. Fix: pass the exact id your server serves (from model-lock.md);
  the AWQ build's id ends in `-AWQ`.

## Before you close the tab

Colab runtimes vanish; artifacts in them vanish too. Last cell, every day:

```python
from google.colab import files
for f_ in ["bench_report.json", "capacity-note.md"]:
    files.download(f_)
```
