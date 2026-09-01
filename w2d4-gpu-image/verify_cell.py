# Paste this into a fresh Colab cell AFTER running generate_probe.py on a T4.
# It validates the gpu_evidence.json the probe just wrote and confirms the T4
# actually ran on CUDA (part 3 of the three-part green check).
# stdlib only. Prints exactly one line last:
#   GREEN CHECK: PASS  or  GREEN CHECK: FAIL (<reason>)
import json, os


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def _fail(reason):
    print("GREEN CHECK: FAIL (%s)" % reason)
    raise _Stop()


def main():
    if not os.path.isfile("gpu_evidence.json"):
        _fail("gpu_evidence.json not found; run generate_probe.py first")
    try:
        with open("gpu_evidence.json") as f:
            e = json.load(f)
    except json.JSONDecodeError as ex:
        _fail("gpu_evidence.json is not valid JSON: %s" % ex)

    for key in ("cuda", "device_name", "tokens_per_s"):
        if key not in e:
            _fail("gpu_evidence.json missing '%s'" % key)

    if e["cuda"] is not True:
        _fail("cuda is false: this runtime is not a GPU (set Runtime > T4 GPU and rerun the probe)")

    tps = e["tokens_per_s"]
    if not isinstance(tps, (int, float)) or tps <= 0:
        _fail("tokens_per_s must be positive, got %s" % tps)

    print("device:", e["device_name"])
    print("tokens_per_s:", tps)
    print("this is part 3 of the green check; download gpu_evidence.json next to your repo for parts 1-2")
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
