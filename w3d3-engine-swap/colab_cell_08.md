import json
import os

def verify_report():
    report_path = "ab_report.json"
    assert os.path.exists(report_path), f"Error: {report_path} not found!"
    
    with open(report_path, "r") as f:
        report = json.load(f)
        
    # 1. Check schema / required keys
    required_keys = ["baseline", "vllm", "speedup_by_concurrency", "predicted_speedup"]
    for k in required_keys:
        assert k in report, f"Schema Error: Missing key '{k}' in ab_report.json"
        
    # 2. Check that vLLM concurrency-8 throughput beats Monday's batch-8 baseline
    vllm_8 = report["vllm"].get("8") or report["vllm"].get(8)
    base_8 = report["baseline"].get("8") or report["baseline"].get(8)
    
    assert vllm_8 is not None and base_8 is not None, "Concurrency 8 metrics missing from report."
    assert vllm_8 > base_8, f"Performance check failed: vLLM concurrency-8 ({vllm_8}) must exceed baseline ({base_8})."
    
    # 3. Verify speedup fields were computed
    speedups = report["speedup_by_concurrency"]
    assert len(speedups) > 0, "Speedup fields were not computed."
    for level, ratio in speedups.items():
        assert isinstance(ratio, (int, float)) and ratio > 0, f"Invalid speedup ratio for level {level}: {ratio}"

    print("GREEN CHECK: PASS")

verify_report()