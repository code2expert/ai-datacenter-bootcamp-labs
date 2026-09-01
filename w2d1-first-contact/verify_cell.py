# Paste this whole cell into a fresh Colab cell after you have written
# results.json and generate.py into the working directory.
# It prints exactly one line last: GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
# stdlib only. No arguments, no interactivity.
import json, os


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def _fail(reason):
    print("GREEN CHECK: FAIL (%s)" % reason)
    raise _Stop()


def main():
    # 1. the generation script must exist on disk
    if not os.path.isfile("generate.py"):
        _fail("generate.py not found next to this cell")

    # 2. results.json must exist and parse
    if not os.path.isfile("results.json"):
        _fail("results.json not found; run the results cell first")
    try:
        with open("results.json") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        _fail("results.json is not valid JSON: %s" % e)

    # 3. top-level schema
    for key in ("model", "gpu", "measurements"):
        if key not in data:
            _fail("results.json missing top-level key '%s'" % key)
    rows = data["measurements"]
    if not isinstance(rows, list) or len(rows) != 3:
        _fail("measurements must be a list of 3 rows (fp16, int8, int4)")

    by_dtype = {}
    for row in rows:
        for field in ("dtype", "predicted_gb", "measured_gb", "tokens_per_s"):
            if field not in row:
                _fail("a measurement row is missing '%s'" % field)
        dt = row["dtype"]
        if dt not in ("fp16", "int8", "int4"):
            _fail("unexpected dtype '%s'" % dt)
        for field in ("predicted_gb", "measured_gb", "tokens_per_s"):
            v = row[field]
            if not isinstance(v, (int, float)):
                _fail("%s.%s is not a number (fill it in)" % (dt, field))
            if v <= 0:
                _fail("%s.%s must be positive, got %s" % (dt, field, v))
        by_dtype[dt] = row

    for dt in ("fp16", "int8", "int4"):
        if dt not in by_dtype:
            _fail("missing the %s row" % dt)

    fp16 = by_dtype["fp16"]["measured_gb"]
    int8 = by_dtype["int8"]["measured_gb"]
    int4 = by_dtype["int4"]["measured_gb"]

    # 4. ranges sane: fp16 weights+overhead land in a physical window on a T4
    if not (2.5 <= fp16 <= 6.0):
        _fail("fp16 measured %.2f GB outside sane 2.5-6.0 GB; remeasure" % fp16)

    # 5. ordering on memory: int4 < int8 < fp16
    if not (int4 < int8 < fp16):
        _fail("memory order wrong: expected int4 < int8 < fp16, got %.2f, %.2f, %.2f"
              % (int4, int8, fp16))

    print("model:", data["model"])
    print("gpu:  ", data["gpu"])
    print("fp16 %.2f GB | int8 %.2f GB | int4 %.2f GB" % (fp16, int8, int4))
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
