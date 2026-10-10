# Perplexity check for the published CPU table (Qwen3.5-0.8B Q8_0)

The lucasrosewinters.com CPU table (apeiron `04d2060`, pp200 2.17×, pp512 2.45×, tg64 1.68–1.76× against
llama.cpp CPU) carried no output-quality check. This receipt is that check: perplexity on identical token ids,
apeiron against llama.cpp, same model file.

| field | value |
|---|---|
| model | `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf`, sha256 `37ae482d336108d2…` |
| text | `/home/socol/models/gpu-ppl-corpus.txt`, sha256 `b0ef1a2f195d5393…`, 17150 tokens (no BOS) |
| ids | `llama-tokenize --ids` (llama.cpp `ece963f41`, `build-cuda`), sha256 `84081af6e51213be…` |
| population | first 16 chunks of 512 tokens; each scores tokens 257–511 (255 per chunk, 4080 total) |
| llama.cpp arm | `llama-perplexity -c 512 -b 512 --chunks 16 -ngl 0 -t 32`, `CUDA_VISIBLE_DEVICES=` (pure CPU) |
| apeiron arm | `examples/lm_eval_bridge.rs`, `APEIRON_KERNEL=q8act APEIRON_F32_CARRIER=1` (the published decode path); context = chunk[0..257], continuation = chunk[257..512] |
| machine | i9-14900KF, 32 threads |
| board | `pub-verify-cpu-quality-5-GUlcVBMCjmJC8SAFf9Mqeh` |

## Result

| arm | PPL | vs llama.cpp |
|---|---:|---:|
| llama.cpp CPU | 11.8967 ± 0.5668 | — |
| apeiron `04d2060`, q8act, token-by-token | 11.9199 | +0.19% |
| apeiron `04d2060`, q8act, batched prefill (`APEIRON_BRIDGE_PREFILL=1`) | 11.9199 | +0.19% |
| apeiron main (this commit), both arms | 11.9199 | +0.19% |

- Per-chunk mean NLL difference, apeiron − llama.cpp: −0.0052 to +0.0078 nats.
- Batched prefill and token-by-token output are byte-identical on all 16 chunks (`cmp`, sha256 `d1dbb3a38c6cccd6…`).
- Main and `04d2060` output are byte-identical on all 16 chunks.
- Decision rule (within 1%) holds.
- The `exact` kernel arm is not measured: it exceeded its 280 s budget at 0 of 16 chunks.

## Bridge defect fixed

At `04d2060` and on main before this commit, `lm_eval_bridge` built a `KvState` only. On a hybrid schedule
(Qwen3.5: affine and softmax layers) it exited 1 with `bad config field affine state for an affine layer`.
The bridge now uses `Engine::state`, `step_logits_state` and `prefill_state`. The same patch was applied to the
`04d2060` worktree for the arm above (sha256 `0fd838612c629254…`).

## Not covered

- Speculative decode (published 2.14× code, 1.80× prose): its output is not compared here.
- Depth: 512-token windows only.
- rustc/LLVM codegen and llama.cpp itself are outside the check.

Artifacts: `/home/socol/oc-tmp/pubverify/` (`llama-ppl.log` sha256 `e9fcd12a82999be5…`, `bridge-*.jsonl`,
`main-*.jsonl`, `bridge-in.jsonl` sha256 `4598ad423785ae31…`). Lean: `proof/corpus/Perf/PubVerifyCpu.lean`.
