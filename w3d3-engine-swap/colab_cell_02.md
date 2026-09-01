#911 for session recovery
import os, sys, time, signal, subprocess, urllib.request, urllib.error

RECOVERY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
RECOVERY_PORT = 8000
RECOVERY_LOG = "/content/server.log"

_R_TRANSFORMERS = "4.46.*"
_R_ACCELERATE = "1.1.*"
_R_NEED_AWQ = False
_R_VLLM = "0.6.*"; _R_HTTPX = "0.27.*"; _R_OPENAI = "1.54.*"

RECOVERY_ARGS = {
    "--model": RECOVERY_MODEL,
    "--dtype": "half",
    "--max-model-len": "4096",
    "--gpu-memory-utilization": "0.85",
    "--port": str(RECOVERY_PORT),
}

subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
time.sleep(2)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                f"vllm=={_R_VLLM}", f"transformers=={_R_TRANSFORMERS}",
                f"accelerate=={_R_ACCELERATE}", f"httpx=={_R_HTTPX}",
                f"openai=={_R_OPENAI}"], check=True)

_r_cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
for k, v in RECOVERY_ARGS.items():
    _r_cmd += [k] if v is None else [k, str(v)]
_r_logf = open(RECOVERY_LOG, "wb")
server = subprocess.Popen(_r_cmd, stdout=_r_logf, stderr=subprocess.STDOUT, start_new_session=True)

_deadline = time.time() + 300
while time.time() < _deadline:
    try:
        with urllib.request.urlopen(f"http://localhost:{RECOVERY_PORT}/v1/models", timeout=5) as r:
            if r.status == 200:
                print("RECOVERED: server healthy. continue from your last step.")
                break
    except (urllib.error.URLError, ConnectionError, OSError):
        pass
    time.sleep(3)