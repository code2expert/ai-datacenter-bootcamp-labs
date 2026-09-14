#!/usr/bin/env python3
"""Generate a small, bounded chat workload for the W5D1 dashboard."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def api(base, key, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + route, data=data, headers={
        "Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}: check the endpoint, key, model and request format") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("request timed out or could not connect; check the endpoint") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30800/v1")
    parser.add_argument("--namespace", default=os.environ.get("NAMESPACE", "team"))
    parser.add_argument("--model", help="required if /v1/models advertises more than one model")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        parser.error("use an HTTP(S) base URL without embedded credentials")
    key = os.environ.get("API_KEY", "")
    if not key:
        raw = subprocess.check_output([
            "kubectl", "-n", args.namespace, "get", "secret", "serving-keys",
            "-o", "json"], stderr=subprocess.DEVNULL)
        key = base64.b64decode(json.loads(raw)["data"]["api-key"]).decode().strip()
    if not key:
        raise ValueError("the API key is empty")
    models = [m["id"] for m in api(base, key, "/models")["data"]]
    model = args.model or (models[0] if len(models) == 1 else None)
    if model not in models:
        raise ValueError("choose a served model with --model; inspect /v1/models in your client")

    def request(number):
        started = time.monotonic()
        result = api(base, key, "/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content":
                          f"Lab request {number}: explain why monitoring a model service is useful."}],
            "max_tokens": 64, "stream": False})
        if not result.get("choices") or not result["choices"][0].get("message"):
            raise ValueError("HTTP response had no chat result; check the service API")
        print(f"{stamp()} request={number} duration_s={time.monotonic() - started:.3f}", flush=True)

    print(f"{stamp()} START artificial lab traffic; model={model}; max_tokens=64", flush=True)
    count = 0
    # A batch completes before the next starts. At most four requests are in flight.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, seconds, callers, pause in [
                ("single caller", 60, 1, 5), ("quiet", 30, 0, 0),
                ("four callers", 60, 4, 2), ("wait for scrapes", 30, 0, 0)]:
            print(f"{stamp()} PHASE {name}", flush=True)
            end = time.monotonic() + seconds
            if not callers:
                time.sleep(seconds)
                continue
            while time.monotonic() < end and count < 150:
                batch = min(callers, 150 - count)
                futures = [pool.submit(request, count + n + 1) for n in range(batch)]
                count += batch
                for future in futures:
                    future.result()
                time.sleep(min(pause, max(0, end - time.monotonic())))
    print(f"{stamp()} END requests={count}; observe the graphs as traffic stops", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Traffic stopped by user.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        # Do not print response bodies or subprocess output: either can contain credentials.
        print("Traffic stopped: check connectivity, credentials and the chat API. "
              "No further requests will be started.", file=sys.stderr)
        sys.exit(1)
