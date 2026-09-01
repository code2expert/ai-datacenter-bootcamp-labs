# Ensure openai is imported from the virtual environment if running in a separate script/cell
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "In one sentence, what is a GPU?"}],
)
print(r.choices[0].message.content)