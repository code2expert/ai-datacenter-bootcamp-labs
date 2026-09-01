# Async A/B client for Lab W3D3 (engine swap).
# Paste the whole file as one Colab cell (after the vLLM server is healthy), then
# call run_sweep(...) as the day-3 README shows. It fires N concurrent chat
# completions per level with httpx + asyncio, excludes a warm-up round, and
# reports aggregate tokens/s at each concurrency level.
#
# It talks to the OpenAI-compatible /v1 endpoint, so the same client works
# against any team's service. No secrets: the local vLLM server needs no key.

import asyncio
import time

import httpx

# A fixed prompt set so every run measures the same work. Varied lengths, no
# duplicates. Requests cycle through this list.
FIXED_PROMPTS = [
    "In one sentence, what is a GPU?",
    "List three reasons decode is memory-bound.",
    "Explain the KV cache to a new ops engineer in two sentences.",
    "What does continuous batching change versus static batching?",
    "Give a one-line definition of tokens per second.",
    "Why does a longer prompt increase time to first token?",
    "Name two things quantisation trades away for smaller memory.",
    "Summarise what an inference server does in three short bullets.",
]

# Output lengths per request, cycled in order. This list is IDENTICAL to Monday's
# QUEUE in the day-2 lab, and it has to stay that way: the A/B is only honest if
# both engines are asked for exactly the same work. 24 requests, 18 that want 32
# tokens and 6 that want 256, so a long request is always in flight alongside
# short ones.
#
# The mixed lengths are the entire point. Ask every request for the same number
# of tokens and there is no straggler, static batching pays no tax, and
# continuous batching has nothing to win back. You would measure a flat speedup
# across concurrency and conclude, wrongly, that continuous batching does not
# scale.
QUEUE = [32, 32, 32, 256] * 6

# Fallback when a caller does not pass a length.
MAX_TOKENS = 128
# Warm-up requests per level, dropped from the timing.
WARMUP = 4


async def _one_request(client, base_url, model, prompt, max_tokens=MAX_TOKENS):
    """Fire one chat completion, return the count of completion tokens."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    r = await client.post(f"{base_url}/chat/completions", json=payload)
    r.raise_for_status()
    body = r.json()
    usage = body.get("usage", {})
    # completion_tokens is what the server generated; fall back to counting.
    # Accounting note vs Monday: static_queue counted REQUESTED tokens, which
    # equals generated there (greedy decode runs to the cap). vLLM can stop at
    # EOS short of the cap, so counting usage is the honest number for it -
    # any bias this introduces runs AGAINST vLLM, never for it.
    ct = usage.get("completion_tokens")
    if ct is None:
        ct = len(body["choices"][0]["message"]["content"].split())
    return ct


async def _run_level(client, base_url, model, prompts, concurrency, total_requests):
    """Run total_requests requests, at most `concurrency` in flight at once."""
    sem = asyncio.Semaphore(concurrency)
    counts = []

    async def guarded(prompt, max_tokens):
        async with sem:
            return await _one_request(client, base_url, model, prompt, max_tokens)

    # Each request carries its own output length from QUEUE, so the workload
    # matches Monday's static-batching baseline request for request.
    tasks = [asyncio.create_task(guarded(prompts[i % len(prompts)],
                                         QUEUE[i % len(QUEUE)]))
             for i in range(total_requests)]
    t0 = time.time()
    for coro in asyncio.as_completed(tasks):
        counts.append(await coro)
    dt = time.time() - t0
    total_tokens = sum(counts)
    return {
        "concurrency": concurrency,
        "requests": total_requests,
        "tokens_per_s": round(total_tokens / dt, 1),
        "wall_s": round(dt, 3),
    }


async def run_sweep(base_url, model, prompts=FIXED_PROMPTS,
                    concurrencies=(1, 4, 8), requests_per_level=24):
    """Sweep the concurrency levels; return a list of per-level result dicts.

    A warm-up round runs first and is discarded so model-load and cache-warm
    cost stays out of the measured numbers.
    """
    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
        # warm-up: fire WARMUP requests, ignore timing
        await asyncio.gather(*[
            _one_request(client, base_url, model, prompts[i % len(prompts)])
            for i in range(WARMUP)
        ])
        for c in concurrencies:
            level = await _run_level(client, base_url, model, prompts, c,
                                     requests_per_level)
            print("level:", level)
            results.append(level)
    return results
