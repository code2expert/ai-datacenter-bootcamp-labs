#!/usr/bin/env python3
"""Green-check verifier for W2D2.

Start your server first:
    uvicorn main:app --host 0.0.0.0 --port 8000
then:
    python verify.py            # defaults to http://localhost:8000
    python verify.py http://localhost:9000   # or point it elsewhere

Checks /health, /v1/models, and a non-streaming /v1/chat/completions. Probes
streaming and reports it, but does not require it. Prints exactly one line last:
    GREEN CHECK: PASS
    GREEN CHECK: FAIL (<reason>)
Exit code matches. Uses httpx if present, else requests, else stdlib urllib.
"""
import json
import sys

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"


# --- one tiny HTTP layer over whatever is installed -------------------------
def _get(url):
    try:
        import httpx
        r = httpx.get(url, timeout=60.0)
        return r.status_code, r.text
    except ImportError:
        pass
    try:
        import requests
        r = requests.get(url, timeout=60.0)
        return r.status_code, r.text
    except ImportError:
        pass
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=60.0) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _post(url, body, stream=False):
    payload = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    try:
        import httpx
        if stream:
            with httpx.stream("POST", url, content=payload, headers=headers, timeout=120.0) as r:
                text = "".join(r.iter_text())
                return r.status_code, text
        r = httpx.post(url, content=payload, headers=headers, timeout=120.0)
        return r.status_code, r.text
    except ImportError:
        pass
    try:
        import requests
        r = requests.post(url, data=payload, headers=headers, timeout=120.0, stream=stream)
        return r.status_code, r.text
    except ImportError:
        pass
    import urllib.request
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _fail(reason):
    print("GREEN CHECK: FAIL (%s)" % reason)
    sys.exit(1)


def main():
    # 1. /health returns 200 and a model id
    try:
        code, body = _get(BASE + "/health")
    except Exception as e:
        _fail("cannot reach %s/health (is the server up?): %s" % (BASE, e))
    if code != 200:
        _fail("/health returned %s, expected 200" % code)
    try:
        health = json.loads(body)
    except json.JSONDecodeError:
        _fail("/health body is not JSON")
    model_id = health.get("model")
    if not model_id:
        _fail("/health body missing 'model'")

    # 2. /v1/models lists the served model id
    code, body = _get(BASE + "/v1/models")
    if code != 200:
        _fail("/v1/models returned %s, expected 200 (route not implemented?)" % code)
    try:
        models = json.loads(body)
    except json.JSONDecodeError:
        _fail("/v1/models body is not JSON")
    ids = [m.get("id") for m in models.get("data", [])]
    if model_id not in ids:
        _fail("/v1/models does not list '%s' (got %s)" % (model_id, ids))

    # 3. non-streaming chat completion is valid
    req = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
    }
    code, body = _post(BASE + "/v1/chat/completions", req)
    if code != 200:
        _fail("/v1/chat/completions returned %s, expected 200 (read the body: %s)"
              % (code, body[:200]))
    try:
        comp = json.loads(body)
    except json.JSONDecodeError:
        _fail("chat completion body is not JSON")

    for field in ("id", "object", "created", "model", "choices", "usage"):
        if field not in comp:
            _fail("completion missing '%s'" % field)
    if comp["object"] != "chat.completion":
        _fail("object is '%s', expected 'chat.completion'" % comp["object"])
    if not comp["choices"]:
        _fail("choices is empty")
    choice = comp["choices"][0]
    msg = choice.get("message", {})
    if msg.get("role") != "assistant":
        _fail("choices[0].message.role is not 'assistant'")
    content = msg.get("content", "")
    if not content or not content.strip():
        _fail("choices[0].message.content is empty")
    if choice.get("finish_reason") not in ("stop", "length"):
        _fail("finish_reason is '%s', expected 'stop' or 'length'" % choice.get("finish_reason"))

    usage = comp["usage"]
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(k), int) or usage.get(k) < 0:
            _fail("usage.%s must be a non-negative int, got %s" % (k, usage.get(k)))
    if usage["completion_tokens"] <= 0:
        _fail("usage.completion_tokens must be positive (nothing was generated)")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        _fail("usage.total_tokens != prompt_tokens + completion_tokens")

    # 4. optional: probe streaming, report but do not require
    stream_note = "streaming: not implemented (optional this week)"
    try:
        req_s = dict(req, stream=True)
        code_s, body_s = _post(BASE + "/v1/chat/completions", req_s, stream=True)
        if code_s == 200 and "data:" in body_s:
            stream_note = "streaming: implemented, saw SSE chunks"
            if "[DONE]" in body_s:
                stream_note += " and [DONE] terminator"
    except Exception:
        pass  # streaming is optional; any failure here is not a green-check failure

    print("model:", model_id)
    print("completion content:", repr(content[:60]))
    print("usage:", usage)
    print(stream_note)
    print("GREEN CHECK: PASS")
    sys.exit(0)


try:
    main()
except SystemExit:
    raise
except Exception as e:                      # a dropped connection mid-run
    print("GREEN CHECK: FAIL (unexpected error mid-check: %s)" % e)
    sys.exit(1)
