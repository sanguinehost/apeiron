# Qwen3-8B NanoQuant serves through Intent on the laptop

Criterion: **I1** (Intent drives the served pack) and **Q1** (the rate this pack reaches). Grade:
`measured` for every timing and byte count; the PPL is `measured` on another host for the same
bytes, and transfers here by hash equality. Q1 is **not met**: 0.92 tok/s end-to-end against the
30 tok/s target, and the pack's 1.124186 b/w misses Q3's 1.00.

## Result

The upstream `arelath/Qwen3-8B-nanoquant-GGUF` image converts to the pair
`bench/cpu/1794879503-nq8b-1124-ctx2048.md` scored, byte for byte, in 6.162 s. `nq-serve` loads it,
and `intent agent` completes a turn against it. No GPU, no cloud, 2.08 GB peak resident under a
2.6 GiB cgroup.

| artifact | bytes | sha256 |
|---|---:|---|
| `qwen3-8b-nanoquant.gguf` (source) | 2,214,297,824 | `75d3f4f150195e0c6a7ccd2ce3372f919534fb7ec609f6b9ba52eb71af194253` |
| `model.nqp` | 976,049,680 | `1f4cc4b43cd60f70cf52e110884f96e6a2eb2539cd0add5b353cc699cd8d0bac` |
| `skeleton.gguf` | 1,329,625,696 | `3927454ca530ce26afdcd5dafe78f2d29d5ac006ea09151c8387d973a2ceaa28` |

Both output hashes equal the pair in `1794879503-nq8b-1124-ctx2048.md`. That pair's measured quality
is therefore this artifact's: **12.189530 PPL** at ctx 2048 on raw wikitext-2 test, first 2,048
tokens, `--second-half`, against Q4_K_M's 7.587145, a ratio of **1.606603**.

## Conversion

```
python3 tools/nq_convert.py \
  --source ~/models/arelath/nq8b-nq/qwen3-8b-nanoquant.gguf \
  --out-dir ~/models/arelath/nq8b-nq
# pack 976049680 B, 252 linears, 1.124186 b/w on 6945767424 fitted weights
# skeleton 1329625696 B, 147 tensors
# real 0m6.162s
```

The source is the `.nq_*` dialect `nq_convert.py` reads: 36 layers, 1,407 tensors, one fused
`attn_qkv` factor per layer split by output row into q/k/v, `nq_u`/`nq_v` as I32 words, scales as
BF16, the embedding and head as Q8_0. No dense reference and no residual plane.

## Serve

```
systemd-run --user --unit=nq-serve -p MemoryMax=2600M -p MemorySwapMax=0 -- \
  ./serve/target/release/nq-serve \
    --model /home/lucas/models/arelath/nq8b-nq/skeleton.gguf \
    --pack /home/lucas/models/arelath/nq8b-nq/model.nqp \
    --tokenizer /home/lucas/models/arelath/nq8b/tokenizer.json \
    --bind 127.0.0.1:8080 --model-name nq-qwen3-8b \
    --max-tokens 64 --kernel q8act \
    --drop-system --system "You are a helpful assistant." --stop-at-sentence
```

Load 0.300 s, warmup 5.292 s, 36 layers, vocab 151,936, 252 linears at 1.124186 b/w.
`GET /health` → `ok`; `GET /v1/models` → `nq-qwen3-8b`.

## Measured

| path | prompt | prefill | decode | output | wall |
|---|---:|---|---|---:|---:|
| `POST /v1/chat/completions` | 33 tok | 11.801 s (2.80 tok/s) | 6.771 s, 17 tok (2.51 tok/s) | 17 tok | 18.727 s |
| `intent agent` (`local/nq-qwen3-8b`) | 25 tok | 17.487 s (1.43 tok/s) | 0.522 s, 3 tok | `Hello World!` | 18.999 s |

The Intent turn is `complete`, one turn, `finish_reason: "stop"`, credential `provider=local
source=none`, model loop `--work /tmp/opencode/intent-smoke`.

The 17-token completion answers "What is 17 times 23?" with `17 times 23 is **17 multiplied by
23**.` — fluent and wrong. That is the base checkpoint at 1.607× Q4_K_M PPL, not a serving defect.

## Memory and rate

Peak cgroup memory 2,080,563,200 B (1.94 GiB) for the 2,305,675,376 B pair, `MemoryMax=2600M`,
`MemorySwapMax=0`. The files are mapped; the resident set is the touched pages.

A decode token reads 1,637,275,152 B: the 976,049,680 B pack plus the Q8_0 `output.weight` at
661,225,472 B. At the 2.51 tok/s decode row that is 4.1 GB/s, below the laptop's bandwidth, so the
binding term is the 6.95e9 sign terms per token, not the bytes.

## What this does not cover

- The PPL was not re-run here. It transfers because the pack and skeleton are hash-identical to the
  receipted pair; a re-measurement on this host would be a new row.
- One turn and one completion, not a mean over three; the two prefill rates differ by 2.0× on the
  same binary and are not a quiet-machine bound.
- The Intent prompt was compacted by the server's `--drop-system`; a real coding turn at the
  assembled 2,672-token prompt would be hours of prefill at these rates.
- No context beyond a few hundred tokens, no 8k row, no 200k: C1 and C2 are untouched.
- The base checkpoint, not the instruct one. Zero-shot accuracy, the other half of Q2, is not
  measured here.
- The upstream GPU pipeline produced this pack; nothing here exercises `learn::nanoquant`.

## What remains

- Q1: 2.51 tok/s decode against 20 (8.0× short) and 30 at 2k (12.0×).
- Q3: 1.124186 b/w against 1.00, and 1.607× against the 1.5× bar (1.07× over).
- Training speed: the upstream pipeline produced this pack in 1,770 s of llama.cpp-adjacent runtime
  on one GPU; our own fit needs a new algorithm, not a faster form of the current one.
