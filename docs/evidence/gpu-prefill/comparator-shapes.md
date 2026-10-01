# Comparator shapes — llama.cpp Q8 prefill vs Apeiron Q8

Read-only analysis. No GPU command ran; the GPU timing lane is another worker.

- llama.cpp: `/home/socol/Workspace/llama.cpp` at `ece963f41b0b02d7a0d61436ae365762c073a4c8`.
- Model: `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf`, sha256
  `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`, GGUF v3, 335 tensors,
  arch `qwen35`, 25 blocks, Q8_0 + F32.
- Apeiron: `src/spirv/q8_tile.comp`, `src/engine/kern/gemm.rs` (`exact_dot`),
  `src/engine/chart/mod.rs` (`quant_act`).
- Derivation and gate: `tools/gpu/gguf_shapes.py`.

## 1 · llama.cpp CUDA Q8 prefill path

`ggml_cuda_mul_mat` (`ggml/src/ggml-cuda/ggml-cuda.cu:1812`) selects the kernel in this order
(lines 1857–1865):

```cpp
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {          // 1857
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst);     // 1858
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {  // 1861
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);         // 1862
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);                 // 1865
```

`ne11` is the token count (src1 `ne[1]`). Decode vs prefill:

- **Decode / thin batch.** `ggml_cuda_should_use_mmvq` (`mmvq.cu:282`) ends
  `return ne11 <= MMVQ_MAX_BATCH_SIZE;` (`mmvq.cu:329`) with `MMVQ_MAX_BATCH_SIZE 8`
  (`mmvq.cuh:3`). `N ≤ 8` → MMVQ (dp4a/MMVQ scalar-vector kernels).
- **Prefill `N > 8`.** MMVQ declines, MMQ is tested. `ggml_cuda_should_use_mmq`
  (`mmq.cu:259`) lists `GGML_TYPE_Q8_0` as supported (`mmq.cu:273`) and returns:

  ```cpp
    if (turing_mma_available(cc)) {
        return true;                                   // mmq.cu:312-314
    }
    ...
    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;  // mmq.cu:325
    }
  ```

  `MMQ_DP4A_MAX_BATCH_SIZE 64` (`mmq.cuh:8`). On Turing or newer the MMA branch returns true for
  every batch, so **all prefill (`N > 8`) Q8_0 projections go through MMQ**. On pre-Turing
  NVIDIA (dp4a, no tensor cores) the dp4a branch caps at `N < 64`; at/above 64 the call reaches
  `ggml_cuda_mul_mat_cublas` (`ggml-cuda.cu:1865`), which dequantizes Q8_0 first
  (`dequantize_block_q8_0_f16`, `convert.cu:44`, selected at `convert.cu:529-533`). There is a
  shared-memory floor: MMQ refused when `smpbo < 48*1024` (`mmq.cu:303-310`).

- **MMQ entry.** `ggml_cuda_mul_mat_q` (`mmq.cu:85`), `GGML_ASSERT(src1->type == GGML_TYPE_F32)`
  (`mmq.cu:87`). Type switch: `case GGML_TYPE_Q8_0: mul_mat_q_case<GGML_TYPE_Q8_0>(...)`
  (`mmq.cu:28-29`). src1 floats are quantized to `block_q8_1_mmq` in
  `quantize_mmq_q8_1_cuda` (`mmq.cu:156`; kernel `quantize.cu:575-603`) with
  `GGML_ASSERT(ne0 % QK8_1_MMQ == 0)` (`quantize.cu:580`), `QK8_1_MMQ = 4*QK8_1 = 128`
  (`mmq.cuh:24`).
- **Tile/kernel.** `mul_mat_q_case` (`mmq.cuh:1552`) picks `J ∈ {8,…,128}` and dispatches
  `launch_mul_mat_q<type, J, fallback>`. Q8_0 util functions (`mmq.cuh:535`) are the dp4a pair
  (`ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a`, `mmq.cuh:577-582`, `VDR_Q8_0_Q8_1_MMQ 8` at
  `vecdotq.cuh:244`) or the mma pair (`ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`, `mmq.cuh:741-746`),
  chosen by `ggml_cuda_mmq_config::use_mma_data_layout` (`mmq.cuh:189-194`):
  `turing_mma_available(cc) || amd_mfma_available(cc) || amd_wmma_available(cc)`.
  On the target NVIDIA prefill, that is the **int8 MMA** kernel.
- **Arithmetic.** MMA: int8 `A·B → int32` tile, epilogue `sum += C.x[l]*dA*dB` in fp32
  (`mmq-vec-dot.cuh:191-197`). dp4a: `d8_0*d8_1*((float)sumi)` where `sumi` is the int32 dot
  (`vecdotq.cuh:246-257`). Either way: **per-block integer dot, then a float scale product,
  accumulated in fp32 across blocks**; `write_back_dp4a`/`write_back_mma` store `sum` unchanged
  (`mmq.cuh:428-465`).

## 2 · Derived shapes

`python3 tools/gpu/gguf_shapes.py --model <gguf> --check` prints these from the file.

Metadata (read, not guessed): `n_embd 1024`, `n_ff 3584`, `n_head 8`, `n_head_kv 2`,
`key_length 256`, `value_length 256`, `n_block 25`, `n_ctx 262144`, `vocab 248320`
(`token_embd.weight.ne[1]`, equals `len(tokenizer.ggml.tokens)`),
`full_attention_interval 4`, `nextn_predict_layers 1`, `ssm.inner_size 2048`.

Layer classes: full-attention at blocks `3, 7, 11, 15, 19, 23` (interval 4) and block `24`
(the `nextn` MTP block, also full attention) — 7 attention blocks; the other 18 are SSM
(linear-attention) blocks.

`q_width = n_head*n_head_dim = 2048`, `kv_width = n_head_kv*n_head_dim = 512`. Note the naive
identity in the task sketch, `n_head*n_head_dim == n_embd`, is **false** for this model:
`8*256 = 2048 = 2*n_embd = attn_output.ne[0]`, and `attn_q` carries the fused per-head gate
(`attn_q.ne[1] = 4096 = 2*q_width`). The checker asserts the true identities against the tensor
table instead.

Q8_0 projection weights (`ne[0]=K` contraction width, `ne[1]=M` rows, `blocks = K/32`); the
attention projections exist in the 7 attention blocks, the SSM projections in the 18 SSM blocks:

| tensor | GGML dims (ne0×ne1 as stored) | K | M | blocks | layers |
|---|---|---|---|---|---|
| `attn_q.weight` | `(1024, 4096)` | 1024 | 4096 | 32 | 7 |
| `attn_k.weight` | `(1024, 512)` | 1024 | 512 | 32 | 7 |
| `attn_v.weight` | `(1024, 512)` | 1024 | 512 | 32 | 7 |
| `attn_output.weight` | `(2048, 1024)` | 2048 | 1024 | 64 | 7 |
| `ffn_gate.weight` | `(1024, 3584)` | 1024 | 3584 | 32 | 25 |
| `ffn_up.weight` | `(1024, 3584)` | 1024 | 3584 | 32 | 25 |
| `ffn_down.weight` | `(3584, 1024)` | 3584 | 1024 | 112 | 25 |
| `attn_qkv.weight` (SSM) | `(1024, 6144)` | 1024 | 6144 | 32 | 18 |
| `attn_gate.weight` (SSM) | `(1024, 2048)` | 1024 | 2048 | 32 | 18 |
| `ssm_out.weight` | `(2048, 1024)` | 2048 | 1024 | 64 | 18 |
| `ssm_alpha.weight` | `(1024, 16)` | 1024 | 16 | 32 | 18 |
| `ssm_beta.weight` | `(1024, 16)` | 1024 | 16 | 32 | 18 |
| `token_embd.weight` (tied LM head) | `(1024, 248320)` | 1024 | 248320 | 32 | 1 |

`blk.24.nextn.eh_proj.weight` is `(2048, 1024)` Q8_0 (K=2048, M=1024, blocks=64).
All `K` values are multiples of 128, so every prefill shape clears the MMQ
`ne0 % QK8_1_MMQ == 0` assert (`quantize.cu:580`). `fallback = ne01 % 128 != 0`
(`mmq.cu:129`) is set for `ssm_alpha`/`ssm_beta` (M=16) and clear for the rest.

## 3 · Matched-input construction

- **Weights.** Both sides read the same GGML Q8_0 payload: `QK8_0 = 32`, 34 bytes per block =
  one f16 scale + 32 int8 codes (`ggml-common.h`; `Q8_0_BLOCK_BYTES = 2 + 32` at
  `src/format/quant.rs:52`). Apeiron's `exact_dot` reads that block directly; the Apeiron GPU
  layout is the repacked `(w, dm)` form (`w` = 8 words of four codes, `dm[r] = (mantissa,
  exponent)` with `f16_split(d) = (m, k)`, `d = m·2^k`). llama.cpp reads the 34-byte block and
  converts the f16 `d` to float. The **codes and the f16 scale are identical inputs**.
- **Activations.** llama.cpp MMQ quantizes per 32 values (`quantize.cu:463`,
  `vals_per_scale = 32` for the Q8_0 D4 layout): `d_inv = 127/amax`, `code = roundf(x*d_inv)`,
  `d = 1/d_inv` (`quantize.cu:515-521`), scale stored f32 (`quantize.cu:551`). Apeiron's
  `quant_act` (`chart/mod.rs:242`) uses a **power-of-two** scale, `S_b = least power of two ≥
  ceil(absmax/127)` (`chart/mod.rs:264`) and `code = floor((2|v| + S)/(2S))` sign-applied
  (`chart/mod.rs:290-301`). Block granularity (32) matches; the scale convention does not.
- **Apeiron `(M, tokens, blocks)` case.** Push constant `Params { M, tokens, blocks }`
  (`q8_tile.comp:16`); `M` = weight rows `ne[1]`, `tokens` = prefill column count, `blocks =
  ne[0]/32`. Each `(row, token)` cell accumulates `Σ_b (m_{r,b}·D_b)·2^{k_{r,b}+e_{t,b}}` in
  signed i128 and emits one Q32.32 word (`q8_tile.comp:71-151`). The attention/FFN shapes above
  fix `(M, blocks)`; `tokens` is the runner's choice. `ssm_alpha`/`ssm_beta` are M=16 with the
  MMQ fallback branch set.
- **To build the same blocks.** Quantize one float activation vector with both quantizers. The
  per-block codes coincide only when Apeiron's power-of-two `S_b` equals llama's float `amax/127`,
  i.e. hand-built activations whose per-block max is a power-of-two multiple of 127. Otherwise
  the dequantized activations differ and only a dequantized-value (tolerance) comparison or a
  llama-side re-quantization to Apeiron's power-of-two scale is valid.
- **Boundaries.** Block size 32 on both sides; weight scale f16 in both; activation scale f32
  in MMQ (f16 in MMVQ's `block_q8_1`) vs Apeiron i64 power-of-two.

## 4 · Arithmetic difference accounting

Comparable **bit-exactly**:

- Per-block integer dot `D_b = Σ_l q_l·xq_l` when the two sides are fed the same weight codes
  and the same activation codes: an int8 dot in both (`mmq-vec-dot.cuh:191`, `exact_dot`
  `gemm.rs:319-322`, `dotPacked4x8EXT` in `q8_tile.comp:107`).
- The weight f16 scale as a Q32.32 value: `d = m·2^k`, `|m| < 2^11`, `k ∈ [8,37]`
  (`gemm.rs:6-7`), identical to `float(d)` only exactly when `d` is representable — `f16 → f32`
  is exact, so `float(d_b)` is exact, but `m·2^k` used as an integer shift is not the same
  arithmetic as a float multiply.

Not bit-exact (needs tolerance or a converted epilogue):

- The block term. Apeiron: `(m·D_b)<<(k+e)` in i128, one `round_shift_32` with the `+2^31` bias
  at the end, then an i64 narrowing check (`q8_tile.comp:122-146`, `gemm.rs:314-330`).
  llama.cpp: `float(d_b)·float(S_b)·D_b` summed in **fp32 over blocks** with no final rounding
  step (`mmq-vec-dot.cuh:197`, `vecdotq.cuh:257`). The scale product and the accumulation are
  both float, and the reduction order is kernel/`J` dependent.
- Cells where the fp32 sum happens to be exact (single block, or exactly representable products)
  can be compared bit-exactly; the rest require an absolute/relative tolerance, or converting
  llama.cpp's per-block float term to Apeiron's shifted-integer term under a stated rounding
  contract.

Apeiron behaviours with **no CUDA counterpart**:

| Apeiron | llama.cpp |
|---|---|
| `REFUSE_I128` on signed i128 accumulation overflow (`q8_tile.comp:124`) | fp32 accumulation, no refusal |
| `REFUSE_I64` on Q32.32 → i64 narrowing (`q8_tile.comp:146`) | result is fp32, no narrowing |
| `REFUSE_NONFINITE` on a non-finite weight scale (`q8_tile.comp:118-119`) | `__half2float` of inf/NaN propagates silently |
| `ZERO_SCALE` marks an absent token/block (`q8_tile.comp:88`) | out-of-bounds rows padded/`fallback`-guarded (`mmq.cu:129`) |

A matched harness must therefore not claim equality on a refusing Apeiron cell, and a mismatch
attributable to fp32-vs-exact must be recorded as a domain difference, not a defect.

## 5 · Open

- No GPU run occurred in this task. Shapes, dispatch and arithmetic only; the timing lane is
  owned by another worker (no `llama-bench`, no GPU command here).
- A matched timing harness still needs: (a) an exact-token runner (tokens from
  `tokenizer.ggml.tokens` / `tokenizer.json`, a fixed token-id list of length `N`); (b) a stated
  timer boundary (whether activation quantization, weight upload/first-touch, dispatch setup and
  the Q32.32 epilogue are inside or outside the timed region, and the same boundary for
  llama.cpp); (c) state reset per run (KV/SSM state, allocator warm state) and warmup iterations;
  (d) the same dispatch on both sides — on Turing+ prefill `N > 8` llama.cpp takes MMQ, on
  pre-Turing `N ≥ 64` it takes cuBLAS (a different arithmetic and a different timing object).
- §10 measurement conditions (named machine, metric, interval, before/after) are not established
  here.
