import json

sample_data = {
    "_note": "SAMPLE baseline from the reference T4 run 2026-08-31.",
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype": "fp16",
    "ttft_s": {
        "128": 0.0371,
        "512": 0.0647,
        "2048": 0.3123
    },
    "tpot_s": 0.0341,
    "batch": {
        "1": 34.0,
        "4": 49.8,
        "8": 96.6
    }
}

with open("baselines.sample.json", "w") as f:
    json.dump(sample_data, f, indent=2)

print("baselines.sample.json created successfully!")