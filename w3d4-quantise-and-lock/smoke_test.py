# Function-calling smoke test for Lab W3D4 (quantise and lock).
# Given in full. You run it; you do not write it. Paste the whole file as one
# Colab cell (after the tool-call-enabled vLLM server is healthy), then call
# run_smoke(base_url=..., model=...).
#
# What it does: fires 3 canonical prompts, k times each for n=10 total
# attempts - 8 that want a tool call, 2 distractors that must NOT call - and
# scores each attempt's BEHAVIOUR: a valid parseable tool_calls when one is
# wanted, or a clean refusal on the distractor.
# Gate: PASS if at least 8 of 10 attempts show correct behaviour AND the
# distractor stays call-free in the majority of its attempts. A model that
# always calls a tool fails the real consumer, so restraint is scored.
#
# It talks to the OpenAI-compatible /v1 endpoint, so the same test works against
# any team's service. No secrets: the local vLLM server needs no key.

from openai import OpenAI

# Two tools the model may call. Shapes match the OpenAI tools schema.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string",
                                   "description": "e.g. 23 * 19"},
                },
                "required": ["expression"],
            },
        },
    },
]

# The 3 canonical prompts. Each carries how many attempts (k) it gets and how many
# tool calls a correct answer makes. n = sum of k = 10.
#   two_tool:   needs BOTH tools (weather + calculator)   -> expect >= 1 call
#   single:     needs ONE tool                            -> expect >= 1 call
#   distractor: needs NO tool, must NOT call one          -> expect 0 calls
CANONICAL = [
    {
        "id": "two_tool",
        "k": 4,
        "wants_call": True,
        "prompt": "What is the weather in Riyadh, and what is 23 multiplied "
                  "by 19? Use your tools.",
    },
    {
        "id": "single",
        "k": 4,
        "wants_call": True,
        "prompt": "What is the weather in Tokyo right now? Use your tools.",
    },
    {
        "id": "distractor",
        "k": 2,
        "wants_call": False,
        "prompt": "In one sentence, explain what a tool call is. Do not call "
                  "any tool; just answer.",
    },
]


def _tool_calls_of(message) -> list:
    """Return the parsed tool_calls list on a response message, or []."""
    tc = getattr(message, "tool_calls", None)
    return list(tc) if tc else []


def _valid_call(call) -> bool:
    """A tool call is valid if it names a known function and its arguments
    parse as JSON with the required field present."""
    import json
    try:
        fn = call.function.name
        if fn not in ("get_weather", "calculate"):
            return False
        args = json.loads(call.function.arguments or "{}")
    except (AttributeError, ValueError):
        return False
    if fn == "get_weather":
        return isinstance(args.get("city"), str) and bool(args["city"])
    if fn == "calculate":
        return isinstance(args.get("expression"), str) and bool(args["expression"])
    return False


def run_smoke(base_url: str, model: str, temperature: float = 0.0) -> dict:
    """Run the smoke test. Returns a result dict with counts and the pass gate."""
    client = OpenAI(base_url=base_url, api_key="not-needed")

    total_attempts = 0
    valid_call_attempts = 0          # attempts that returned >=1 valid tool call
    distractor_attempts = 0
    distractor_call_free = 0         # distractor attempts that made NO tool call
    per_prompt = {}

    for spec in CANONICAL:
        pid, k, wants = spec["id"], spec["k"], spec["wants_call"]
        got_valid = 0
        got_call_free = 0
        for _ in range(k):
            total_attempts += 1
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": spec["prompt"]}],
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=256,
            )
            msg = resp.choices[0].message
            calls = _tool_calls_of(msg)
            any_valid = any(_valid_call(c) for c in calls)

            if wants:
                # a "wants a call" prompt counts toward the 8/10 gate when it
                # returns at least one valid tool call
                if any_valid:
                    valid_call_attempts += 1
                    got_valid += 1
            else:
                # the distractor counts toward the 8/10 gate when it correctly
                # makes NO tool call, and separately toward distractor compliance
                distractor_attempts += 1
                if not calls:
                    valid_call_attempts += 1
                    distractor_call_free += 1
                    got_call_free += 1

        per_prompt[pid] = {"k": k, "wants_call": wants,
                           "valid": got_valid, "call_free": got_call_free}

    # gate: >=8/10 correct behaviours AND distractor call-free in the majority
    distractor_majority = (distractor_call_free * 2 > distractor_attempts) \
        if distractor_attempts else True
    passed = (valid_call_attempts >= 8) and distractor_majority

    return {
        "model": model,
        "total_attempts": total_attempts,       # 10
        "score": valid_call_attempts,           # correct behaviours, out of 10
        "distractor_attempts": distractor_attempts,
        "distractor_call_free": distractor_call_free,
        "distractor_majority_clean": distractor_majority,
        "per_prompt": per_prompt,
        "passed": passed,
    }
