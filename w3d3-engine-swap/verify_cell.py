# Green-check verifier for Lab W3D3 (engine swap).
# Paste this as the last cell of your day-3 notebook and run it. It reads
# ab_report.json and checks the schema, that vLLM's concurrency-8 throughput
# beats Monday's batch-8 baseline, and that the speedup fields were computed.
#
# Last line is exactly one of:
#   GREEN CHECK: PASS
#   GREEN CHECK: FAIL (<reason>)
# No interactivity, no arguments; exit code matches.

import json, os


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def fail(reason: str) -> "NoReturn":
    print(f"GREEN CHECK: FAIL ({reason})")
    raise _Stop()


def main() -> None:
    path = "ab_report.json"
    if not os.path.exists(path):
        fail(f"{path} not found; write it in Cell 5")
    try:
        with open(path) as fh:
            report = json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"{path} is not valid JSON: {exc}")

    for key in ("baseline", "vllm", "speedup_by_concurrency"):
        if key not in report:
            fail(f"{path} missing key: {key}")

    baseline = report["baseline"]
    vllm = report["vllm"]
    speedup = report["speedup_by_concurrency"]

    if not isinstance(baseline, dict) or not baseline:
        fail("baseline must be a non-empty object (from Monday's baselines.json)")
    if not isinstance(vllm, dict) or not vllm:
        fail("vllm must be a non-empty object of measured throughput")
    if not isinstance(speedup, dict) or not speedup:
        fail("speedup_by_concurrency must be a non-empty object")

    # keys may be strings or ints depending on how the report was built; normalise
    def get_c(d, c):
        for k, v in d.items():
            if str(k) == str(c):
                return v
        return None

    base8 = get_c(baseline, 8)
    vllm8 = get_c(vllm, 8)
    if base8 is None:
        fail("baseline has no concurrency-8 (batch-8) number")
    if vllm8 is None:
        fail("vllm has no concurrency-8 number")
    if not isinstance(base8, (int, float)) or not isinstance(vllm8, (int, float)):
        fail("concurrency-8 throughput values must be numbers")

    # the headline claim of the day
    if not vllm8 > base8:
        fail(f"vllm concurrency-8 throughput ({vllm8}) not above baseline "
             f"batch-8 ({base8}); the engine swap should win here")

    # speedup fields must be computed (present and numeric for at least c=8)
    s8 = get_c(speedup, 8)
    if s8 is None or not isinstance(s8, (int, float)):
        fail("speedup_by_concurrency has no numeric value at concurrency 8")
    # sanity: the reported speedup should match vllm8/base8 within rounding
    expected = vllm8 / base8
    if abs(s8 - expected) > 0.1:
        fail(f"speedup at 8 ({s8}) does not match vllm/baseline "
             f"({expected:.2f}); recompute it")

    print(f"baseline batch-8: {base8}, vllm concurrency-8: {vllm8}")
    print(f"speedup at 8: {s8}x")
    print("GREEN CHECK: PASS")


try:
    main()
except _Stop:
    # A notebook cell cannot exit nonzero without printing a red traceback over
    # the result line, so only signal by exit code when run as a plain script.
    try:
        get_ipython()  # defined only inside IPython/Colab
    except NameError:
        raise SystemExit(1)
