import time, urllib.request, urllib.error

def tail_log(path=SERVER_LOG, n=30):
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n:])
    except FileNotFoundError:
        return "(no log file yet)"

def wait_for_health(port=PORT, timeout_s=300, interval_s=3):
    url = f"http://localhost:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    waited = int(timeout_s - (deadline - time.time()))
                    print(f"server healthy after about {waited}s: {url} -> 200")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval_s)
    print(f"TIMED OUT after {timeout_s}s waiting for {url}")
    print("last 30 log lines:")
    print(tail_log())
    return False

healthy = wait_for_health()