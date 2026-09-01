import json

# Load baseline data
baseline = json.load(open("baselines.sample.json"))

# Map measured throughput and baseline numbers by concurrency level
vllm_by_c = {x["concurrency"]: x["tokens_per_s"] for x in vllm_measured}
base_by_c = {int(k): v for k, v in baseline["batch"].items()}

# Calculate speedup ratios
speedup = {c: round(vllm_by_c[c] / base_by_c[c], 2)
           for c in vllm_by_c if c in base_by_c}

report = {
    "baseline": base_by_c,
    "vllm": vllm_by_c,
    "speedup_by_concurrency": speedup,
    "predicted_speedup": None,
}

# Write out report
with open("ab_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("Generated ab_report.json successfully:\n")
print(json.dumps(report, indent=2))