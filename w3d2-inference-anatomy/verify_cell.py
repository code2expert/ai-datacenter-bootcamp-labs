# Green-check verifier for Lab W3D2 (inference anatomy).
# Paste this as the last cell of your day-2 notebook and run it. It reads
# baselines.json (the export you also downloaded) plus the KV measurement you
# saved to kv_check.json, and checks the schema and the sanity rules.
#
# Last line is exactly one of:
#   GREEN CHECK: PASS
#   GREEN CHECK: FAIL (<reason>)
# No interactivity, no arguments; exit code matches.

import json, os

# Qwen2.5-1.5B KV cache: 2 x 28 layers x 2 kv_heads x 128 head_dim x 2 bytes.
KV_FORMULA_KB_PER_TOKEN = 2 * 28 * 2 * 128 * 2 / 1024  # 28.0


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def fail(reason: str) -> "NoReturn":
    print(f"GREEN CHECK: FAIL ({reason})")
    raise _Stop()


def load_json(path: str):
    if not os.path.exists(path):
        fail(f"{path} not found")
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"{path} is not valid JSON: {exc}")


def main() -> None:
    b = load_json("baselines.json")

    # schema
    for key in ("model", "dtype", "ttft_s", "tpot_s", "batch"):
        if key not in b:
            fail(f"baselines.json missing key: {key}")
    if not isinstance(b["ttft_s"], dict) or not b["ttft_s"]:
        fail("ttft_s must be a non-empty object keyed by prompt length")
    if not isinstance(b["batch"], dict):
        fail("batch must be an object keyed by batch size")
    for size in ("1", "4", "8"):
        if size not in b["batch"]:
            fail(f"batch missing size {size}")

    # sanity 1: TTFT rises with prompt length - the day's actual physics.
    # Prefill reads the whole prompt before the first token, so a 2048-token
    # prompt must pay a visibly larger TTFT than a 128-token one (the reference
    # T4 run measured 0.037 s vs 0.312 s). A flat TTFT means prefill was not
    # measured (cached prompt, wrong timestamps, or a reused stream).
    tpot = b["tpot_s"]
    if not isinstance(tpot, (int, float)) or tpot <= 0:
        fail(f"tpot_s not a positive number: {tpot}")
    for plen, ttft in b["ttft_s"].items():
        if not isinstance(ttft, (int, float)) or ttft <= 0:
            fail(f"ttft_s[{plen}] not a positive number: {ttft}")
    if not b["ttft_s"]["2048"] > b["ttft_s"]["128"]:
        fail(f"TTFT did not rise with prompt length "
             f"(128: {b['ttft_s']['128']}, 2048: {b['ttft_s']['2048']}); "
             "prefill is not being measured")

    # sanity 2: batch-8 throughput beats batch-1
    b1, b8 = b["batch"]["1"], b["batch"]["8"]
    if not (isinstance(b1, (int, float)) and isinstance(b8, (int, float))):
        fail("batch tokens/s values must be numbers")
    if not b8 > b1:
        fail(f"batch-8 throughput ({b8}) not above batch-1 ({b1})")

    # sanity 3: measured KV within a factor of 2 of the formula
    kv = load_json("kv_check.json")
    measured = kv.get("measured_kb_per_token")
    if not isinstance(measured, (int, float)) or measured <= 0:
        fail("kv_check.json needs a positive measured_kb_per_token")
    lo, hi = KV_FORMULA_KB_PER_TOKEN / 2, KV_FORMULA_KB_PER_TOKEN * 2
    if not lo <= measured <= hi:
        fail(f"measured KV {measured} KB/token outside 2x of formula "
             f"{KV_FORMULA_KB_PER_TOKEN} (allowed {lo:.1f} to {hi:.1f})")

    print(f"ttft lengths: {sorted(b['ttft_s'])}, tpot_s: {tpot}")
    print(f"batch tokens/s 1/4/8: {b['batch']['1']}/{b['batch']['4']}/"
          f"{b['batch']['8']}")
    print(f"KV measured {measured} KB/token vs formula "
          f"{KV_FORMULA_KB_PER_TOKEN} KB/token")
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
