import urllib.request
import os
import subprocess

# 1. Pull the venv deployment script
os.makedirs("w3d3-engine-swap", exist_ok=True)
venv_url = "https://raw.githubusercontent.com/code2expert/ai-datacenter-bootcamp-labs/main/w3d3-engine-swap/deploy_venv_for_310.py"
script_path = "w3d3-engine-swap/deploy_venv_for_310.py"
urllib.request.urlretrieve(venv_url, script_path)

# 2. Read, clean line continuations (\), strip '!', and execute block-by-block
with open(script_path, "r") as f:
    content = f.read()

# Reconstruct commands by joining backslash-continued lines
cleaned_lines = []
current_cmd = ""
for line in content.splitlines():
    stripped = line.strip()
    if stripped.startswith("#") or not stripped:
        continue
    if stripped.startswith("!"):
        stripped = stripped[1:].strip()
    
    if stripped.endswith("\\"):
        current_cmd += stripped[:-1] + " "
    else:
        current_cmd += stripped
        cleaned_lines.append(current_cmd)
        current_cmd = ""

# Run each constructed command via subprocess
for cmd in cleaned_lines:
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# 3. Pull and run the shared scaffold for today's lab
os.makedirs("shared", exist_ok=True)
scaffold_url = "https://raw.githubusercontent.com/code2expert/ai-datacenter-bootcamp-labs/main/shared/colab_scaffold.py"
urllib.request.urlretrieve(scaffold_url, "shared/colab_scaffold.py")

%run shared/colab_scaffold.py