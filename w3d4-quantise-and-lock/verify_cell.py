# Green-check verifier for Lab W3D4 (quantise and lock).
# Paste this as the last cell of your day-4 notebook and run it. It reads
# smoke_result.json (written from the smoke test) and model-lock.md, and checks
# that the smoke score meets the gate and that the lock file is fully filled in.
#
# Last line is exactly one of:
#   GREEN CHECK: PASS
#   GREEN CHECK: FAIL (<reason>)
# No interactivity, no arguments; exit code matches.

import json, os, re


class _Stop(Exception):
    """Ends the check without killing the notebook kernel."""


def fail(reason: str) -> "NoReturn":
    print(f"GREEN CHECK: FAIL ({reason})")
    raise _Stop()


def main() -> None:
    # 1) smoke result
    if not os.path.exists("smoke_result.json"):
        fail("smoke_result.json not found; write it in Cell 5")
    try:
        with open("smoke_result.json") as fh:
            result = json.load(fh)
    except json.JSONDecodeError as exc:
        fail(f"smoke_result.json is not valid JSON: {exc}")

    for key in ("score", "total_attempts", "distractor_majority_clean", "passed"):
        if key not in result:
            fail(f"smoke_result.json missing key: {key}")

    score = result["score"]
    total = result["total_attempts"]
    if not isinstance(score, int) or not isinstance(total, int):
        fail("score and total_attempts must be integers")
    if total != 10:
        fail(f"total_attempts is {total}, the smoke test defines n=10")
    if score < 8:
        fail(f"smoke score {score}/10 is below the 8/10 gate")
    if not result["distractor_majority_clean"]:
        fail("distractor did not stay call-free in the majority; a model that "
             "always calls a tool fails the real consumer")
    if not result["passed"]:
        fail("smoke test reports passed=false")

    # 2) model-lock.md fully filled in
    if not os.path.exists("model-lock.md"):
        fail("model-lock.md not found")
    with open("model-lock.md") as fh:
        lock = fh.read()
    remaining = re.findall(r"FILL:", lock)
    if remaining:
        fail(f"model-lock.md has {len(remaining)} unfilled FILL: placeholders")
    # require a concrete model id line
    if not re.search(r"Model id:\s*\S+", lock):
        fail("model-lock.md has no concrete Model id")

    print(f"smoke score: {score}/{total}, distractor clean: "
          f"{result['distractor_majority_clean']}")
    print("model-lock.md: all fields filled")
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
