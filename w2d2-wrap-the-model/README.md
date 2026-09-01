# Lab W2D2: wrap the model

Start:      Your fork of `serving-stack` from yesterday. Work in `app/`. Copy the five `starter/` files there to begin.
Objective:  Put the model behind an OpenAI-compatible `/v1` API on CPU: `/health`, `/v1/models`, and non-streaming `/v1/chat/completions`.

Yesterday the model ran in a notebook. Today it goes behind an HTTP API in the
OpenAI `/v1` shape. That shape is not decoration: it is the contract the Agentic
AI cohort's agents call in week 4, and the seam week 3 slides a new engine under.
Get it right once and it never changes.

This lab is CPU only. Run it on your laptop, or on a Colab CPU runtime (Runtime,
Change runtime type, CPU). Model: `Qwen/Qwen2.5-0.5B-Instruct`, small enough to
load and generate on a CPU in seconds.

Correctness before speed. The model blocks the event loop while it generates;
that is fine this week. Week 3's engine owns concurrency. Name the limitation,
do not fix it here.

Checkpoints for the room (the honest pace, recovery time included): `/health`
answering by **14:00**, `/v1/models` and a first completion by **15:00**, the
green check by **16:30**. Behind a checkpoint? Say so at the checkpoint, not at
five - the TA triages by these times, and a stuck install is a five-minute fix
at 14:00 and a lost afternoon at 16:00.

## Predict (by hand)

Credited for handing it in, never marked right or wrong - hedged guesses teach nothing, and nothing here is graded for accuracy.

Before you write any route, predict:

- After you send one chat request with a 10-word user message and ask for 32
  tokens back, `usage.prompt_tokens` will be about _____ and
  `usage.completion_tokens` will be about _____.
- Which of the three routes will pass its test first, with the least code? _____
- Will an unmodified `openai` Python client work against your server with only a
  `base_url` change, no other edits? _____ (why or why not)

## The delta

You are given five starter files in `starter/`. Copy them into your repo's
`app/` folder and build from there:

- `schemas.py` complete. The pydantic request and response shapes. Do not weaken
  them; they are the contract's teeth and are given so the teeth are not the
  exercise.
- `main.py` a FastAPI skeleton. `GET /health` is implemented as the worked
  example. The other two routes are `TODO` stubs whose docstrings state the exact
  contract.
- `client_test.py` the openai client pointed at localhost:8000; one call, prints
  the reply.
- `requests.md` a working curl for every route with the expected output shape.
- `requirements.txt` the dependency names and no versions. Pinning each one per
  `../../../PINS.md` is step 1, and it is part of the lab, not setup noise.

Then work the routes in this order. Each step is small.

### Step 1: pin, install, run the skeleton (about 45 min, mostly the install; fill your prediction card while pip runs)

`requirements.txt` ships with package names and no versions. Open
`../../../PINS.md`, find the "serving service (weeks 2, tier 0 CPU path)" table,
and write the version next to every package **before you install anything**.

Do this first because an unpinned install resolves to whatever shipped this
morning, and the failure that produces is the worst kind: the server starts, it
answers `/health`, and it returns 500 on the first real completion. You would
spend the afternoon debugging your route when the bug was in your install.

On Colab, delete the `torch` line rather than pinning it. Colab ships torch
preinstalled and a second one breaks the runtime. Locally, pin it.

Then install and start the server. The skeleton loads the model and serves
`/health` already.

```bash
cd app
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt   # only once every line carries a version
uvicorn main:app --host 0.0.0.0 --port 8000
```

In another shell, confirm the trivial win:

```bash
curl -s http://localhost:8000/health
```

You should get `{"status": "ok", "model": "Qwen/Qwen2.5-0.5B-Instruct"}`. If the
first load is slow, the HF cache was not pre-seeded and weights are downloading;
expect a few minutes once, then it is cached.

### Step 2: implement GET /v1/models (about 20 min)

The smallest real route. Read the docstring in `main.py`. Build a `ModelList`
with one `ModelCard` whose `id` is `MODEL_ID`, and return it. Verify against the
shape in `requests.md`:

```bash
curl -s http://localhost:8000/v1/models
```

The `data[0].id` must equal the served model id. uvicorn reloads on save if you
launched with `--reload`; otherwise restart it.

### Step 3: implement POST /v1/chat/completions, non-streaming (about 60 min)

This is the lab. Follow the numbered steps in the `chat_completions` docstring:
apply the chat template, count `prompt_tokens`, generate, decode only the new
tokens, count `completion_tokens`, set `finish_reason`, and assemble the
`ChatCompletionResponse`. Test with curl from `requests.md`:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":16}'
```

Required to be correct: `choices[0].message.content` non-empty;
`usage` counts positive with `total_tokens == prompt_tokens + completion_tokens`;
`model` echoed back. If pydantic returns a 422, read it; the error names the
field you got wrong.

### Step 4: prove it with the openai client (about 15 min)

Correctness is not "curl worked", it is "a standard client works". Run:

```bash
python client_test.py
```

It swaps `base_url` to your service and sends one completion. A coherent reply
printed means the contract holds end to end. This is the green-check behaviour.

### Step 5 (delta step): streaming (about 30 min, not required for the green check)

Add Server-Sent Events. When `req.stream` is true, return a `StreamingResponse`
that yields `data: {chunk}\n\n` per token with `choices[0].delta.content`, and
ends with the literal line `data: [DONE]`. Test:

```bash
curl -sN http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Count to five."}],"max_tokens":32,"stream":true}'
```

The `-N` disables curl buffering so you watch chunks arrive. The verifier checks
streaming only if you implemented it, and skips it cleanly otherwise.

## Verify (green check)

With the server running on port 8000, from the lab folder:

```bash
python verify.py
```

It checks `/health` returns 200, `/v1/models` lists the model id, and
`/v1/chat/completions` returns a valid completion (fields present, content
non-empty, usage counts positive). It probes streaming and notes whether it is
implemented, but does not require it. The last line printed is exactly one of:

```
GREEN CHECK: PASS
GREEN CHECK: FAIL (<reason>)
```

## Stretch

- Make `temperature: 0` deterministic (greedy) and confirm two identical
  requests return identical content.
- Add a `system` message and observe how it changes the reply. The contract
  already supports the role; no code change needed, just try it.

## Failure modes

- The server starts, `/health` answers, and the first completion returns 500.
  Tell: an `AttributeError` in the traceback, usually on `input_ids.shape`.
  Cause: you installed before pinning, so `transformers` resolved to 5.x, which
  changed what `apply_chat_template` returns. Fix: pin per `../../../PINS.md`
  (`transformers==4.46.*`), delete the venv, recreate it, reinstall. This is the
  failure step 1 exists to prevent, and it is the one that wastes a whole
  afternoon because the server looks healthy the entire time.
- First model load is slow. Tell: the server sits on "loading ..." for minutes.
  Cause: the HF cache was not pre-seeded, so weights download. Fix: wait it out
  once; subsequent loads are cached and fast.
- Port 8000 already bound. Tell: `[Errno 98] address already in use` on
  startup. Fix: stop the other process (`lsof -i :8000` then kill it) or run on
  another port and point the verifier at it.
- pydantic 422 on your chat route. Tell: the response is a 422 with a JSON body.
  Read it: it names the exact field and why (missing, wrong type). Fix that
  field. The 422 is the schema doing its job.
- Empty or garbage content. Tell: `content` is "" or repeats one token. Cause:
  you decoded the whole output including the prompt, or forgot
  `add_generation_prompt=True`. Fix: decode only `out[0][prompt_tokens:]`.
- The request seems to hang under two clients at once. That is generation
  blocking the event loop. Expected this week; week 3's engine owns concurrency.
  Name it in your notes, do not solve it.
