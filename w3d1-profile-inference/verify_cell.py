# Green-check verifier for Lab W3D1 (profile inference).
# Paste this as the last cell of your day-1 notebook and run it. It reads
# profile.json (the matrix rows you wrote) and checks the schema and the sanity
# rules. It also reads the batch experiment numbers if you saved them to
# batch_check.json; if that file is absent it asks for the two numbers inline so
# the batch-8 > batch-1 rule can still be checked.
#
# Last line is exactly one of:
#   GREEN CHECK: PASS
#   GREEN CHECK: FAIL (<reason>)
# No interactivity, no arguments; exit code matches.

import json, os

REQUIRED_KEYS = {"dtype", "context", "vram_gb", "util_mean", "tokens_per_s"}


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def fail(reason: str) -> "NoReturn":
    print(f"GREEN CHECK: FAIL ({reason})")
    raise _Stop()


def load_json(path: str):
    if not os.path.exists(path):
        fail(f"{path} not found; write it in the last data cell")
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"{path} is not valid JSON: {exc}")


def main() -> None:
    rows = load_json("profile.json")

    if not isinstance(rows, list) or not rows:
        fail("profile.json must be a non-empty list of rows")

    # schema
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            fail(f"row {i} is not an object")
        missing = REQUIRED_KEYS - set(row)
        if missing:
            fail(f"row {i} missing keys: {sorted(missing)}")

    dtypes = {r["dtype"] for r in rows}
    contexts = sorted({r["context"] for r in rows})
    if "fp16" not in dtypes:
        fail("no fp16 rows; the matrix needs fp16")
    if len(contexts) < 3:
        fail(f"need at least 3 context lengths, found {contexts}")

    # sanity 1: VRAM rises with context (within each dtype)
    for dt in dtypes:
        sub = sorted((r for r in rows if r["dtype"] == dt),
                     key=lambda r: r["context"])
        vrams = [r["vram_gb"] for r in sub]
        if any(b < a - 0.01 for a, b in zip(vrams, vrams[1:])):
            fail(f"{dt} VRAM does not rise with context: {vrams}")

    # sanity 2: fp16 uses more memory than int8 at a shared context
    if "int8" in dtypes:
        shared = None
        for c in contexts:
            has_fp16 = any(r["dtype"] == "fp16" and r["context"] == c for r in rows)
            has_int8 = any(r["dtype"] == "int8" and r["context"] == c for r in rows)
            if has_fp16 and has_int8:
                shared = c
                break
        if shared is None:
            fail("fp16 and int8 share no context length to compare")
        fp16_v = next(r["vram_gb"] for r in rows
                      if r["dtype"] == "fp16" and r["context"] == shared)
        int8_v = next(r["vram_gb"] for r in rows
                      if r["dtype"] == "int8" and r["context"] == shared)
        if not fp16_v > int8_v:
            fail(f"fp16 VRAM ({fp16_v}) not above int8 VRAM ({int8_v}) at "
                 f"context {shared}")

    # sanity 3: batch-8 tokens/s beats batch-1
    b1 = b8 = None
    if os.path.exists("batch_check.json"):
        bc = load_json("batch_check.json")
        b1 = bc.get("batch1_tokens_per_s")
        b8 = bc.get("batch8_tokens_per_s")
    else:
        # allow the two numbers as module-level names set in an earlier cell
        b1 = globals().get("BATCH1_TOKENS_PER_S")
        b8 = globals().get("BATCH8_TOKENS_PER_S")
    if b1 is None or b8 is None:
        fail("batch numbers missing; save batch_check.json with "
             "batch1_tokens_per_s and batch8_tokens_per_s, or set "
             "BATCH1_TOKENS_PER_S / BATCH8_TOKENS_PER_S")
    if not b8 > b1:
        fail(f"batch-8 tokens/s ({b8}) not above batch-1 ({b1})")

    print(f"rows: {len(rows)}, dtypes: {sorted(dtypes)}, contexts: {contexts}")
    print(f"batch-1 tokens/s: {b1}, batch-8 tokens/s: {b8}")
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
