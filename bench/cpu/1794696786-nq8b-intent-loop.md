# Intent drives the served v3 Qwen3-8B pack end-to-end

`bench/cpu/1794690665-nq8b-laptop-serve.md` closes the engine half of "end-to-end": the full
36-layer pair loaded, a tokenized prompt in, greedy tokens and text out. This receipt closes the
other half. A sibling listener (`serve/`, outside `src/` and `examples/`) exposes
`/v1/chat/completions`; the `local` provider in `../intent` dials it, parses the SSE stream, appends
the turn to its event log, and prints the assistant message. Both a 0.6B plumbing run and the 8B v3
pack completed a turn.

## Machine and artifacts

Intel Core i5-10210U (4 cores / 8 threads), 6.7 GiB RAM, 4 GiB swap, NVMe, Ubuntu, no GPU,
governor `powersave`. The box is shared: the 0.6B turn ran at load average 13–17, the 8B turn at
35–49, with co-tenant `cargo`, Clippy and harness builds. The rates below are what ran, not a quiet
gate.

| artifact | bytes |
|---|---:|
| 8B skeleton `nq8b/skeleton.gguf` | 1,329,625,696 |
| 8B r2 pack `nq8b-v3s-r2/model.nqp` | 2,718,095,872 |
| 8B `nq8b/tokenizer.json` | 4,790,085 |
| 0.6B skeleton `nq06/skeleton.gguf` | 336,814,976 |
| 0.6B pack `nq06/model.nqp` | 63,370,960 |

The 8B file hashes are in `1794690665-nq8b-laptop-serve.md`; the pair is the same one.

## The listener

`serve/` is a separate workspace with its own `Cargo.lock` and `target/`. The apeiron crate still
opens no socket: `src/` and `examples/` are untouched, and `cargo test --workspace` at the root does
not build the listener. It parses `POST /v1/chat/completions`, applies the Qwen3 ChatML template
(the empty `<think>` block for thinking off), runs `Engine::prefill_state` and
`Engine::next_token_state` greedily, and writes SSE deltas, a usage-bearing finish chunk and
`data: [DONE]`. `GET /v1/models` and `GET /health` answer probes.

```
cd serve && cargo build --release

systemd-run --user --scope -p MemoryMax=4600M -p MemorySwapMax=0 -- \
  ./target/release/nq-serve \
    --model     /home/lucas/models/arelath/nq8b/skeleton.gguf \
    --pack      /home/lucas/models/arelath/nq8b-v3s-r2/model.nqp \
    --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
    --bind 127.0.0.1:8080 --model-name nq-qwen3-8b \
    --max-tokens 64 --kernel q8act \
    --drop-system --system "You are a helpful assistant." --stop-at-sentence
```

`cargo test --release` in `serve/` covers the prompt renderer: four tests over the assistant prefix,
the thinking-off compare, content-part arrays and the `developer`→`system` role mapping.

## Intent wiring

The `local` provider needs no default endpoint: `INTENT_LOCAL_BASE_URL` names the base, and the
model reference is `local/<name>`. No credential is required (`source=none` in the provenance row).

```
export INTENT_LOCAL_BASE_URL=http://127.0.0.1:8080/v1
export INTENT_TIMEOUT_SECS=1800      # the 180 s default is below one 8B prefill here

intent agent -p "Hello World" --model local/nq-qwen3-8b --work .
```

The model loop assembled 2,672 prompt tokens for that one-word task in an empty directory (system
prompt, workspace view, the text-tool fallback and the tool registry). The server's `--drop-system`
replaces that block with the one-line system message and never renders the tool array; the request
it tokenizes is 25 ids. `--stop-at-sentence` ends the turn at the first `.`, `!`, `?` or newline
with `finish_reason: "stop"`, because the base checkpoint does not emit `<|im_end|>` on its own and
Intent classifies a `length` cut with no tool calls as `response_truncated`, a turn error.

The two flags are demo rules. They are not the server default, and the 2,672-token prompt was not
run: at the measured prefill rates it would be about 1.8 h on the 0.6B and 13.5 h on the 8B. The
Intent record still labels the turn `prompt_tokens: 2672` from its own assembly; the response's
`in_tokens` is the 25 the engine actually read.

## Measured

One turn each, `finish_reason: "stop"`, Intent `outcome: complete`, one turn.

| model | kernel | prompt | prefill | decode | output | Intent ttft | Intent stream |
|---|---|---:|---|---:|---|---:|---:|
| 0.6B | `q8act` | 25 | 60.095 s (0.42 tok/s) | 27.614 s, 9 tok (0.33 tok/s) | `Hello World, you're a helpful assistant.` | 60.105 s | 87.721 s |
| 8B v3 | `q8act` | 25 | 457.390 s (0.055 tok/s) | 215.983 s, 4 tok (0.019 tok/s) | `Hello, world!` | 460.762 s | 673.487 s |

A repeat of the 0.6B turn before the 8B one measured 84.918 s prefill and 26.078 s decode for the
same 9 tokens; the difference is co-tenant load, not a code change.

The 8B server ran in a `MemoryMax=4600M`, `MemorySwapMax=0` scope. Peak cgroup memory was
2,595,938,304 B (2.42 GiB), current at stop 2,356,359,168 B; warmup (a 7-token prefill) was
108.035 s. The pair and the pack are mapped, so the resident set is reclaimable file pages, not a
fixed 4.05 GB allocation. An `exact`-kernel warmup of the same pair took 184.607 s before the
server was restarted on `q8act`.

The 8B rate is I/O, not arithmetic: `/proc/<pid>/io` read 22.17 GB by late prefill and 23.15 GB by
the end of the turn, on a 4.05 GB pair, because the co-tenant builds evicted the mapped pages and
each token re-faulted them from NVMe. The earlier receipt's 29-token `q8act` chat took 85.9 s
prefill on a quieter box through the copied pair; that comparison is a cache-state difference, not
a kernel regression. The serving receipt's `exact` path remains the documented mode for the 2 GB
footprint; `q8act` fits here because the pair is file-backed.

## What this establishes, and what it does not

`measured`: the OpenAI-compatible SSE surface, the `local` provider adapter, Intent's event log and
its final-message path all ran against the real engine, for both packs. `tested`: the four prompt
renderer tests. The answer text is the base checkpoints' own continuation; it is not a quality or
instruction-following claim.

Not covered: the full assembled prompt, tool calls, multi-turn KV reuse, concurrent requests,
sampling other than greedy, auth or TLS, and the cold KV tier. The server has no sampler and never
emits a tool call. The KV allocated for the 8B turn was 25 + 64 + 8 positions, 57 MB at 589,824
B/token; context-scale KV is the next problem, not this receipt.
