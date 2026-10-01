# Matched llama.cpp CUDA hybrid prefill baseline

The number the native GPU path must beat. Exact-token comparator, same 200 ids as the retained
Apeiron receipt. Command and output are the record; grade **measured** for the llama.cpp side only.

- Command: `timeout 540 python3 tools/gpu/llama_cmp_gate.py --device gpu` (cwd repository root)
- Result: exit 0, `gate_elapsed_seconds=1.326`, `llama_cmp_gate OK`
- Device: NVIDIA GeForce RTX 5080, compute capability 12.0, 15899 MiB; CUDA0 high priority
- llama.cpp revision: `ece963f41b0b02d7a0d61436ae365762c073a4c8`
- Model: `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf` sha256
  `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- Ids: `tests/fixtures/qwen35-prompt2.ids` sha256
  `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`
- Comparator source: `tools/gpu/llama_cmp.c` sha256
  `d1f25b5e69d368b35d25fc0a48625448e88f70d41580f03872fa5b8d9b9914d0`

## Result

```json
{"model":"/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf","ids":"tests/fixtures/qwen35-prompt2.ids","n_tokens":200,"n_gpu_layers":-1,"n_ctx":1024,"n_batch":512,"n_ubatch":512,"n_threads":16,"warmup_tps":4577.0482,"elapsed_s":0.008716,"tok_per_s":22947.3982,"next_token":12182}
```

`n_gpu_layers=-1` offloads every layer; `next_token=12182` is the same continuation the CPU
comparator produced, so the run is the same computation on the same tokens. `warmup_tps=4577` is the
first (graph-reserve-inclusive) pass; the reported rate is the post-warmup pass.

## Comparison (0.8B Q8_0 hybrid, pp200, same ids)

| path | tok/s | vs CUDA |
|---|---:|---:|
| llama.cpp CUDA prefill (`--device gpu`) | **22947** | 1.00× |
| llama.cpp CPU prefill (`--device cpu`) | 582 | 0.025× |
| Apeiron CPU prefill, HEAD `06e9d6d` default carrier | 1291 | 0.056× |
| Apeiron GPU resident prefill | — | none exists |

Controls (both red, from the gate): a malformed id file is rejected (`llama_cmp: malformed id`,
exit 2); an id count over `n_batch` is refused without truncation (`id count exceeds n_batch`,
exit 2).

## Kernel-level gap

Measured, matched at the ten Qwen3.5-0.8B projection shapes: Apeiron's Q8 SPIR-V primitive is
171.63–506.36 GMAC/s (`primitive-latency.md`), llama.cpp's CUDA MMQ is 6387–52225 GMAC/s
(`cuda-primitive-latency.md`), a factor **30.6×–295.9×**. Apeiron uses `dotPacked4x8EXT`, a four-lane
signed dot per quad; llama.cpp's `tokens>8` prefill routes to `ggml_cuda_mul_mat_q` int8 MMA
(`mmq.cu:85,312-314`), quantizing the f32 activation on-device to `block_q8_1_mmq` before the MMA. The
whole-model ratio (17.8× over Apeiron CPU) and the primitive ratio share the cause that Apeiron does
not use the cooperative-matrix path the device exposes (`coopmat-design.md:31-49`).

The primitive comparison is device-time-for-the-projection on both sides and is not
epilogue-matched: Apeiron computes the exact checked i128 epilogue, CUDA a float epilogue plus a
separate on-device activation quantization. Apeiron's own cost is further dominated by its
per-invocation term (`gpu-occupancy-analysis.md`), so the ratio is not a pure tensor-core effect.

## Boundary (obligation 11)

- llama.cpp whole-model prefill only. It is not a matched timer boundary against Apeiron's device
  timestamps: it includes graph scheduling, attention, the LM head and host transport.
- Apeiron has no resident GPU prefill, so the rows above are not a like-for-like speed claim.
- The comparator shares the token list and continuation token, not the timer or state reset.
