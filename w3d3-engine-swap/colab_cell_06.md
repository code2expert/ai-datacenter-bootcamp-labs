# Clean up whitespace and write a fresh, valid ab_client.py file
raw_code = '''
import asyncio
import time
import httpx

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

QUEUE = [32, 32, 32, 256] * 6
MAX_TOKENS = 128
WARMUP = 4

async def _one_request(client, base_url, model, prompt, max_tokens=MAX_TOKENS):
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
    ct = usage.get("completion_tokens")
    if ct is None:
        ct = len(body["choices"][0]["message"]["content"].split())
    return ct

async def _run_level(client, base_url, model, prompts, concurrency, total_requests):
    sem = asyncio.Semaphore(concurrency)
    counts = []

    async def guarded(prompt, max_tokens):
        async with sem:
            return await _one_request(client, base_url, model, prompt, max_tokens)

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
    results = []
    async with httpx.AsyncClient(timeout=120.0) as client:
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
'''

# Clean non-breaking spaces and save
cleaned_code = raw_code.replace('\xa0', ' ')
with open("ab_client.py", "w") as f:
    f.write(cleaned_code)

# Execute the sweep
from ab_client import run_sweep, FIXED_PROMPTS

vllm_measured = await run_sweep(
    base_url="http://localhost:8000/v1",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    prompts=FIXED_PROMPTS,
    concurrencies=[1, 4, 8],
)

print("Sweep completed successfully!")