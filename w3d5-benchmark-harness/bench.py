"""Benchmark harness for the serving stack (week 3 day 5 reference).

Sweeps concurrency levels against an OpenAI-compatible endpoint and reports the
numbers the week-3 lab is graded on: tokens/sec, TTFT p50/p95, end-to-end
latency p95, and error counts per level.

CLI contract (the week-3 lab is written against this exactly):

    python bench.py \
        --base-url http://localhost:8000 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --concurrency 1,2,4,8,16 \
        --requests-per-level 20 \
        --prompt-file prompts.txt \
        --out bench_report.json

Method:
  - Every request uses stream=true so TTFT is the wall-clock time to the first
    SSE content chunk. End-to-end latency is time to the [DONE] sentinel.
  - One warm-up request per level is fired and excluded from the statistics, so
    a cold cache or JIT does not skew the first measured level.
  - Errors are counted per level and never crash the sweep; a level with all
    errors still reports (with null latencies).
  - Output is written to --out as JSON and printed as a readable table. Re-runs
    append to a `runs` array in the same file when it already exists, so an A/B
    (CPU vs vLLM, Monday vs Wednesday) accumulates in one place.

Dependencies: httpx (async). Pin per ../../PINS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx


# --------------------------------------------------------------------------- #
# Per-request measurement                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class RequestResult:
    ok: bool
    ttft_s: Optional[float] = None          # time to first content chunk
    latency_s: Optional[float] = None       # time to [DONE]
    completion_tokens: int = 0              # counted from streamed chunks
    error: Optional[str] = None


async def _one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Fire one streaming completion and measure TTFT and end-to-end latency.

    TTFT is the time from send to the first SSE frame that carries non-empty
    content (the role-announcement frame and empty deltas do not count).
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    start = time.perf_counter()
    ttft: Optional[float] = None
    tokens = 0

    try:
        async with client.stream("POST", url, json=body) as response:
            if response.status_code != 200:
                # Drain so the connection can be reused, then report the error.
                text = (await response.aread()).decode("utf-8", "replace")[:200]
                return RequestResult(
                    ok=False, error=f"HTTP {response.status_code}: {text}"
                )
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens += 1
        latency = time.perf_counter() - start
        return RequestResult(
            ok=True, ttft_s=ttft, latency_s=latency, completion_tokens=tokens
        )
    except Exception as exc:  # network error, timeout, reset: count, do not raise
        return RequestResult(ok=False, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Per-level sweep                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class LevelReport:
    concurrency: int
    tokens_per_s: float
    ttft_p50_s: Optional[float]
    ttft_p95_s: Optional[float]
    latency_p95_s: Optional[float]
    errors: int
    ok: int
    wall_s: float = field(default=0.0)


def _percentile(values: list[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile, robust for tiny samples."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = max(1, int(round(pct / 100.0 * len(ordered))))
    rank = min(rank, len(ordered))
    return round(ordered[rank - 1], 4)


async def _run_level(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    requests_per_level: int,
    max_tokens: int,
) -> LevelReport:
    """Run one concurrency level: one warm-up (excluded), then the measured batch."""
    # Warm-up, excluded from stats. Ignore its result entirely.
    await _one_request(client, base_url, model, prompts[0], max_tokens)

    # Fire `requests_per_level` requests `concurrency` at a time. A semaphore
    # caps in-flight requests so the level actually holds the intended load.
    semaphore = asyncio.Semaphore(concurrency)

    async def _guarded(index: int) -> RequestResult:
        async with semaphore:
            prompt = prompts[index % len(prompts)]
            return await _one_request(client, base_url, model, prompt, max_tokens)

    level_start = time.perf_counter()
    results = await asyncio.gather(
        *(_guarded(i) for i in range(requests_per_level))
    )
    wall = time.perf_counter() - level_start

    ok = [r for r in results if r.ok]
    errors = len(results) - len(ok)
    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    latencies = [r.latency_s for r in ok if r.latency_s is not None]
    total_tokens = sum(r.completion_tokens for r in ok)

    # System throughput: total completion tokens over the level wall-clock, so
    # it rises with concurrency until the server saturates.
    tokens_per_s = round(total_tokens / wall, 2) if wall > 0 else 0.0

    return LevelReport(
        concurrency=concurrency,
        tokens_per_s=tokens_per_s,
        ttft_p50_s=_percentile(ttfts, 50),
        ttft_p95_s=_percentile(ttfts, 95),
        latency_p95_s=_percentile(latencies, 95),
        errors=errors,
        ok=len(ok),
        wall_s=round(wall, 3),
    )


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #

def _load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise SystemExit(f"prompt file {path!r} has no non-empty lines")
    return prompts


def _print_table(levels: list[LevelReport]) -> None:
    header = (
        f"{'conc':>4}  {'tok/s':>8}  {'ttft_p50':>9}  {'ttft_p95':>9}  "
        f"{'lat_p95':>8}  {'ok':>4}  {'err':>4}"
    )
    print(header)
    print("-" * len(header))
    for lv in levels:
        def fmt(value: Optional[float]) -> str:
            return f"{value:.3f}" if value is not None else "  n/a"
        print(
            f"{lv.concurrency:>4}  {lv.tokens_per_s:>8.2f}  "
            f"{fmt(lv.ttft_p50_s):>9}  {fmt(lv.ttft_p95_s):>9}  "
            f"{fmt(lv.latency_p95_s):>8}  {lv.ok:>4}  {lv.errors:>4}"
        )


def _write_report(out_path: str, run_record: dict) -> None:
    """Append this run to `runs` in the JSON file, creating it if absent."""
    document: dict = {"runs": []}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                document = existing
        except (json.JSONDecodeError, OSError):
            # Corrupt or unreadable prior file: start fresh rather than crash.
            document = {"runs": []}
    document["runs"].append(run_record)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)


async def _sweep(args: argparse.Namespace) -> tuple[dict, list[LevelReport]]:
    prompts = _load_prompts(args.prompt_file)
    concurrency_levels = [int(c) for c in args.concurrency.split(",") if c.strip()]

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    limits = httpx.Limits(max_connections=max(concurrency_levels) + 4)
    levels: list[LevelReport] = []

    headers = {}
    if getattr(args, "api_key", ""):
        headers["Authorization"] = "Bearer " + args.api_key
    async with httpx.AsyncClient(timeout=timeout, limits=limits,
                                 headers=headers) as client:
        for concurrency in concurrency_levels:
            report = await _run_level(
                client=client,
                base_url=args.base_url,
                model=args.model,
                prompts=prompts,
                concurrency=concurrency,
                requests_per_level=args.requests_per_level,
                max_tokens=args.max_tokens,
            )
            levels.append(report)
            # Progress line per level so a long sweep is not silent.
            print(
                f"[level {concurrency}] tok/s={report.tokens_per_s} "
                f"ttft_p95={report.ttft_p95_s} errors={report.errors}",
                flush=True,
            )

    return {
        "timestamp": int(time.time()),
        "base_url": args.base_url,
        "model": args.model,
        "requests_per_level": args.requests_per_level,
        "max_tokens": args.max_tokens,
        "prompt_file": args.prompt_file,
        "levels": [vars(lv) for lv in levels],
    }, levels


def main() -> None:
    parser = argparse.ArgumentParser(description="serving-stack benchmark harness")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", default="1,2,4,8,16",
                        help="comma-separated concurrency levels")
    parser.add_argument("--requests-per-level", type=int, default=20)
    parser.add_argument("--prompt-file", default="prompts.sample.txt")
    parser.add_argument("--out", default="bench_report.json")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--api-key", default=os.environ.get("API_KEY", ""),
                        help="bearer key for keyed services (or set API_KEY); "
                             "omit for an open endpoint")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-request timeout in seconds")
    args = parser.parse_args()

    run_record, levels = asyncio.run(_sweep(args))
    print()
    _print_table(levels)
    _write_report(args.out, run_record)
    print(f"\nwrote {args.out} (run appended)")


if __name__ == "__main__":
    main()