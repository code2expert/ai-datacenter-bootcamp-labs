# 1. Install python3.10 and venv support
!sudo apt-get update -y
!sudo apt-get install python3.10 python3.10-venv python3.10-dev -y

# 2. Create an isolated virtual environment
!python3.10 -m venv /content/venv

# 3. Upgrade pip inside the virtual environment
!/content/venv/bin/python -m pip install --upgrade pip

# 4. Install the required serving pins
!/content/venv/bin/python -m pip install \
    "vllm==0.6.*" \
    "transformers==4.46.*" \
    "accelerate==1.1.*" \
    "httpx==0.27.*" \
    "openai==1.54.*"

print("Virtual environment ready with vLLM installed!")