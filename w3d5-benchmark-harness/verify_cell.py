# Green-check verifier for Lab W3D5 (benchmark harness).
# Paste this as the last cell of your day-5 notebook and run it. It reads
# bench_report.json (from the harness) and capacity-note.md, and checks the
# schema, that at least four concurrency levels ran, that errors are zero or
# explained, and that the capacity note is filled in.
#
# Last line is exactly one of:
#   GREEN CHECK: PASS
#   GREEN CHECK: FAIL (<reason>)
# No interactivity, no arguments; exit code matches.

import json, os, re

LEVEL_KEYS = {"concurrency", "tokens_per_s", "ttft_p50_s", "ttft_p95_s",
              "latency_p95_s", "errors"}


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def fail(reason: str) -> "NoReturn":
    print(f"GREEN CHECK: FAIL ({reason})")
    raise _Stop()


def main() -> None:
    # 1) bench report
    if not os.path.exists("bench_report.json"):
        fail("bench_report.json not found; run the harness in Cell 3")
    try:
        with open("bench_report.json") as fh:
            document = json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"bench_report.json is not valid JSON: {exc}")

    # bench.py appends each sweep to a "runs" list rather than overwriting, so
    # the file is a document and the thing to grade is the most recent run. A
    # bare list is also accepted, for a report assembled by hand.
    if isinstance(document, dict) and isinstance(document.get("runs"), list):
        if not document["runs"]:
            fail("bench_report.json has no runs; the harness wrote nothing")
        levels = document["runs"][-1].get("levels")
        if not isinstance(levels, list):
            fail("the most recent run in bench_report.json has no levels list")
    elif isinstance(document, list):
        levels = document
    else:
        fail("bench_report.json must be the harness output ({'runs': [...]}) "
             "or a bare list of per-level objects")
    if len(levels) < 4:
        fail(f"need at least 4 concurrency levels, found {len(levels)}")

    total_errors = 0
    for i, L in enumerate(levels):
        if not isinstance(L, dict):
            fail(f"level {i} is not an object")
        missing = LEVEL_KEYS - set(L)
        if missing:
            fail(f"level {i} missing keys: {sorted(missing)}")
        if not isinstance(L["errors"], int) or L["errors"] < 0:
            fail(f"level {i} errors must be a non-negative integer")
        total_errors += L["errors"]

    # 2) the knee file from Cell 5
    if not os.path.exists("knee.json"):
        fail("knee.json not found; write it in Cell 5")
    try:
        with open("knee.json") as fh:
            knee = json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"knee.json is not valid JSON: {exc}")
    target = knee.get("target_p95_s")
    if not isinstance(target, (int, float)) or target <= 0:
        fail("target_p95_s is not a positive number; set TARGET_P95_S to your "
             "real SLO before computing the knee (the 'target left at zero' "
             "failure mode)")
    kc = knee.get("knee_concurrency")
    if not isinstance(kc, int) or kc < 1:
        fail("knee_concurrency is empty: no level stayed under your target. "
             "Either your SLO is stricter than this stack can serve (explain "
             "that in the note) or the target was never set from the card")

    # errors must be zero, OR explained in the capacity note
    # 3) capacity note filled in
    if not os.path.exists("capacity-note.md"):
        fail("capacity-note.md not found")
    with open("capacity-note.md") as fh:
        note = fh.read()
    remaining = re.findall(r"FILL:", note)
    if remaining:
        fail(f"capacity-note.md has {len(remaining)} unfilled FILL: placeholders")

    if total_errors > 0 and not re.search(r"error", note, re.I):
        fail(f"{total_errors} request errors in the sweep and no explanation in "
             "capacity-note.md; zero errors, or explain them")

    # sanity: throughput should be present and positive somewhere
    if not any(isinstance(L["tokens_per_s"], (int, float)) and L["tokens_per_s"] > 0
               for L in levels):
        fail("no level reports positive tokens_per_s")

    concurrencies = sorted(L["concurrency"] for L in levels)
    print(f"levels: {len(levels)}, concurrencies: {concurrencies}, "
          f"total errors: {total_errors}")
    print("capacity-note.md: all fields filled")
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
