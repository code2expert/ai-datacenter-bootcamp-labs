"""Prove the contract with the real openai client, not curl.

The whole point of building the OpenAI-compatible shape is that a standard
client works against it with only a base_url swap. Start the server first:

    uvicorn main:app --host 0.0.0.0 --port 8000

then in another shell:

    python client_test.py

It sends one chat completion and prints the reply. If this prints a coherent
answer, your /v1/chat/completions route honours the contract.
"""
from openai import OpenAI

# base_url swap: point the client at the local service instead of api.openai.com.
# The api_key is required by the client but unused by our server this week.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Name three primary colours."},
    ],
    max_tokens=64,
)

print("reply:", resp.choices[0].message.content)
print("finish_reason:", resp.choices[0].finish_reason)
print("usage:", resp.usage)
