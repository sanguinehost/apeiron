# nq-serve

An OpenAI-compatible `/v1/chat/completions` endpoint over the apeiron engine. It is the embedding
application's listener, so the apeiron crate keeps its rule that `src/` and `examples/` open no
socket (`../SECURITY.md`, "No network listener").

## Goal

This server is the serving path for the goal in [`../GOAL.md`](../GOAL.md): a 7B-class model on the
6.6 GiB laptop at about 30 tok/s CPU decode with a 200,000-token context, leaving at least 2 GiB for
four or five coding-agent sessions, driven by `../intent`. The relevant acceptance criteria are M1
(memory), T1 and T2 (decode and turn latency), C1 and C2 (context state and quality), and I1 and I2
(Intent integration and the DeepSeek replacement share).

The current 8B v3 pair is a comparison artifact, not the goal: it stores 4.05 GB at 3.13 b/w, keeps
the classic f16 KV cache, and decodes at 0.09 tok/s on the laptop (`../bench/cpu/1794690665-nq8b-laptop-serve.md`).
It does not meet Q1, M1, T1 or C1. A pack, a row or a speedup is an artifact of the goal, not the
goal.

This crate is a sibling workspace, not a member of the apeiron workspace: `cargo test --workspace`
at the repository root does not build it, and it has its own `Cargo.lock` and `target/`.

## Build

```sh
cd serve
cargo build --release
```

## Run

8B v3 pack:

```sh
./target/release/nq-serve \
  --model     /home/lucas/models/arelath/nq8b/skeleton.gguf \
  --pack      /home/lucas/models/arelath/nq8b-v3s-r2/model.nqp \
  --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
  --bind 127.0.0.1:8080 --model-name nq-qwen3-8b \
  --max-tokens 32 --kernel q8act \
  --drop-system --system "You are a helpful assistant." --stop-at-sentence
```

0.6B pack (for plumbing):

```sh
./target/release/nq-serve \
  --model     /home/lucas/models/arelath/nq06/skeleton.gguf \
  --pack      /home/lucas/models/arelath/nq06/model.nqp \
  --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
  --bind 127.0.0.1:8080 --model-name nq-qwen3-0.6b \
  --max-tokens 32 --kernel q8act \
  --drop-system --system "You are a helpful assistant." --stop-at-sentence
```

The model pair is mapped read-only; the file-backed pages are reclaimable. `q8act` builds its
per-tensor F16 tiles on the first multi-column prefill, so a run that follows only the one-token
warmup pays that build on its first request; raise the warmup prompt if the first request must not.

## Intent

The `local` provider (`../intent/crates/intent-agent/src/providers/local.rs`) needs a base URL and
an optional key; it sends `POST {base}/chat/completions` and expects SSE deltas and `[DONE]`.

```sh
export INTENT_LOCAL_BASE_URL=http://127.0.0.1:8080/v1
export INTENT_TIMEOUT_SECS=1800   # the default 180 s does not cover a laptop prefill

# one shot through the real loop (stdout is the final assistant message)
intent agent -p "Hello World" --model local/nq-qwen3-8b --work .

# the console, with a model selected at launch
intent --model local/nq-qwen3-8b
```

`--drop-system` and `--stop-at-sentence` are demo rules, not defaults:

- `--drop-system` serves the conversation without Intent's assembled system prompt. Intent sends
  2,672 prompt tokens for a one-word task in an empty directory; the compact request is 25. On this
  CPU the difference between one minute and most of a day.
- `--system TEXT` replaces every system/developer message with that one line, so the compact
  request still carries an instruction. `--drop-system` alone leaves the model without one.
- `--stop-at-sentence` ends the turn at the first `.`, `!`, `?` or newline with
  `finish_reason: "stop"`. The base checkpoint does not emit `<|im_end|>` on its own, and Intent
  treats a `length` cut with no tool calls as `response_truncated`, an error.

Drop both flags for a real coding turn and budget the prefill and KV.

## Measured

Two turns through `intent agent`, same 25-token prompt, `finish_reason: "stop"`, Intent outcome
`complete`; the full machine and commands are in `../bench/cpu/1794696786-nq8b-intent-loop.md`.

| pack | kernel | prefill | decode | output |
|---|---|---:|---:|---|
| 0.6B | `q8act` | 60.095 s | 27.614 s, 9 tok | `Hello World, you're a helpful assistant.` |
| 8B v3 | `q8act` | 457.390 s | 215.983 s, 4 tok | `Hello, world!` |

The 8B turn ran under a `MemoryMax=4600M`, `MemorySwapMax=0` scope at load average 35–49 with
co-tenant builds; it read 23.15 GB from NVMe on a 4.05 GB mapped pair because the page cache was
evicted. The rates are what that shared box produced, not a quiet-machine bound. The 0.6B repeat
measured 84.918 s prefill, so cache state dominates the spread.

## Routes and domain

| route | answer |
|---|---|
| `POST /v1/chat/completions` | greedy completion; `stream` selects SSE or one JSON body |
| `GET /v1/models` | the served id |
| `GET /health` | `ok` |

- Greedy only. `temperature`, `top_p`, `stop` (client) and `tools` are accepted and ignored; the
  server has no sampler and never emits tool calls.
- One request is decoded at a time, on the thread that accepted it.
- The output ceiling is `min(request max_tokens, --max-tokens)` because the KV state is allocated
  for prompt + output up front. The default is 256.
- Prompt + output must fit the engine's KV capacity; a refusal is HTTP 500 with the engine message.
- No authentication and no TLS. The default bind is `127.0.0.1`. Binding a wider address exposes
  an unauthenticated text generator to that network; do not do it without a proxy that adds both.
- ChatML follows the Qwen3 template, including the empty `<think>` block for
  `enable_thinking=False`. `--think` leaves the block out and lets the model reason.
