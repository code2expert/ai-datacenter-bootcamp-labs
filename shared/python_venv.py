import subprocess
import os

# 1. Install python3.10 and venv support on Colab
subprocess.run(["sudo", "apt-get", "update", "-y"], check=True)
subprocess.run(["sudo", "apt-get", "install", "python3.10", "python3.10-venv", "python3.10-dev", "-y"], check=True)

# 2. Create the virtual environment
subprocess.run(["python3.10", "-m", "venv", "/content/venv"], check=True)

# 3. Upgrade pip inside the venv
subprocess.run(["/content/venv/bin/python", "-m", "pip", "install", "--upgrade", "pip"], check=True)