# Qwen3.5-0.8B Float64 Reference Forward and Parity Evidence

- **Date:** Mon Sep 28 2026
- **Task:** Q35-T0 (Float64 NumPy reference forward for Qwen3.5-0.8B)
- **Model GGUF:** `/home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf`
- **Model SHA256:** `37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f`
- **llama.cpp Commit:** `ece963f41b0b02d7a0d61436ae365762c073a4c8`
- **ApeironWord.lean SHA256:** `2c41dae6ffcac7faceac06a7378c66e897326d9f0b60f7349a29a680829bde71`
- **Reference Script:** `tools/qwen35/reference.py`

---

## 1. Primary Questions & Citations from `llama.cpp`

### (a) RoPE Type and Pairing for `LLM_ARCH_QWEN35`

From `src/llama-model.cpp:2737-2740`:
```cpp
        case LLM_ARCH_QWEN35:
        case LLM_ARCH_QWEN35MOE:
            return LLAMA_ROPE_TYPE_IMROPE;
```

From `ggml/src/ggml-cpu/ops.cpp:6074-6076`:
```cpp
                    case GGML_ROPE_TYPE_NEOX:
                    case GGML_ROPE_TYPE_MROPE:
                    case GGML_ROPE_TYPE_IMROPE:
                        rotate_pairs<T>(n_dims, n_dims/2, cache, src, dst_data);
                        break;
```

From `ggml/src/ggml-cpu/ops.cpp:5934-5948`:
```cpp
template<typename T>
static void rotate_pairs(const int64_t n, const int64_t n_offset, const float * cache, const T * src_data, T * dst_data, const int scale = 2) {
  for (int64_t i0 = 0; i0 < n; i0 += 2) {
    const int64_t ic = i0/scale; // hack for GGML_ROPE_TYPE_NORMAL, where we need ic = i0; for all other cases, ic = i0/2

    const float cos_theta = cache[i0 + 0];
    const float sin_theta = cache[i0 + 1];

    const T * const src = src_data + ic;
    T * dst             = dst_data + ic;

    const float x0 = type_conversion_table<T>::to_f32(src[0]);
    const float x1 = type_conversion_table<T>::to_f32(src[n_offset]);

    dst[0]        = type_conversion_table<T>::from_f32(x0*cos_theta - x1*sin_theta);
    dst[n_offset] = type_conversion_table<T>::from_f32(x0*sin_theta + x1*cos_theta);
  }
}
```

**Semantics:**
- For `n_dims = 64` and `n_offset = n_dims / 2 = 32`, `scale = 2`:
  Index `ic` runs `0..31`. The paired indices are NeoX halves `(ic, ic + 32)`:
  `(0, 32), (1, 33), ..., (31, 63)`.
- When all text position coordinates are equal (`pos_t = pos_h = pos_w = pos_e = pos`), `ggml_mrope_cache_init` (`ggml/src/ggml-cpu/ops.cpp:5862-5929`) initializes `theta = pos` across all sectors. Advancing by `theta_scale = powf(1.0f / freq_base, 2.0f / n_dims)` yields frequency `angle_j = pos · (1e7)^(-2j/64)` for each pair `j ∈ 0..31`.
- Dimensions `64..255` pass through untouched (`ops.cpp:6086-6094`).

---

### (b) RMS Norm Weight Handling

From `conversion/qwen.py:387-388`:
```python
        elif name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
            data_torch = data_torch + 1
```

From `src/llama-graph.cpp:1564, 1578`:
```cpp
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
...
    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);
    }
```

**Semantics:**
- Standard HuggingFace RMSNorm models often store weights as `w - 1` (offsets around 0). `conversion/qwen.py` adds `+1` at conversion time to all `norm.weight` tensors except `linear_attn.norm.weight` (`ssm_norm`), which was already stored as the true multiplicative scale.
- At runtime in `llama.cpp`, `build_norm` applies `rms_norm(x) · mw` without adding `+1`.
- Therefore, all GGUF norm weights are used directly as stored.

---

### (c) Orientation of the State Matrix $S$ in Autoregressive GDN

From `src/models/delta-net-base.cpp:343-366`:
```cpp
    // [1, S_v, H_v, n_seqs]
    ggml_tensor * sk;
    sk = ggml_mul     (ctx0, s, k);
    sk = ggml_sum_rows(ctx0, sk);

    // [S_v, 1, H_v, n_seqs]
    ggml_tensor * d;
    d = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));
    d = ggml_mul(ctx0, d, b);

    // [1, S_v, H_v, n_seqs]
    ggml_tensor * d_t;
    d_t = ggml_transpose(ctx0, d);

    // [S_v, S_v, H_v, n_seqs]
    ggml_tensor * kd;
    k  = ggml_repeat(ctx0, k, s);
    kd = ggml_mul   (ctx0, k, d_t);

    s = ggml_add(ctx0, s, kd);
...
    ggml_tensor * s_q = ggml_mul     (ctx0, s, q);
    ggml_tensor * o   = ggml_sum_rows(ctx0, s_q);
```

**Semantics:**
- In ggml, `s` has dimensions `[ne0, ne1, ne2, ne3] = [S_v, S_v, H_v, n_seqs]` with `S_v = 128`. `ggml_sum_rows` sums along `ne0`.
- Vector `k` has length `S_k = 128` along `ne0`. Thus `k` contracts along `ne0` of `s`.
- When represented as a matrix $S \in \mathbb{R}^{S_k \times S_v}$ where axis 0 is the key/query dimension ($S_k = 128$) and axis 1 is the value dimension ($S_v = 128$):
  - $k$ contracts with the row/key index: $sk = k^\top S = S^\top k \in \mathbb{R}^{S_v}$.
  - $d = \beta \odot (v - sk) \in \mathbb{R}^{S_v}$.
  - State update: $S \leftarrow e^g S + k \otimes d$ (outer product of shape $S_k \times S_v$).
  - Query output: $o = q^\top S = S^\top q \in \mathbb{R}^{S_v}$ (where $q$ is pre-scaled by $1/\sqrt{128}$).

---

### (d) $L_2$ Norm Epsilon Semantics in `ggml_l2_norm`

From `ggml/src/ggml-cpu/ops.cpp:4204`:
```cpp
                const float scale = 1.0f/fmaxf(sqrtf(sum), eps);
```

**Semantics:**
- `ggml_l2_norm` computes $\frac{x}{\max(\|x\|_2, \epsilon)}$ with $\epsilon = 10^{-6}$.
- It does **not** compute $\frac{x}{\sqrt{\|x\|_2^2 + \epsilon}}$.

---

## 2. Parity Verification Against `llama.cpp`

### llama.cpp Execution Commands

Greedy decoding on CPU with no chat template, `-ngl 0`, `--temp 0`, `--top-k 1`, `-s 42`, `-n 32`, `-t 8`, `--no-display-prompt`:

**Prompt 1:**
```bash
/home/socol/Workspace/llama.cpp/build-cuda/bin/llama-completion \
  -m /home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf \
  -p "The capital of France is" \
  --temp 0 --top-k 1 -no-cnv -s 42 -n 32 -t 8 -ngl 0 \
  --no-display-prompt < /dev/null
```

**Prompt 2:**
```bash
/home/socol/Workspace/llama.cpp/build-cuda/bin/llama-completion \
  -m /home/socol/models/qwen3.5-0.8b/Qwen3.5-0.8B-Q8_0.gguf \
  -p "$PROMPT_2_TEXT" \
  --temp 0 --top-k 1 -no-cnv -s 42 -n 32 -t 8 -ngl 0 \
  --no-display-prompt < /dev/null
```
*(where `$PROMPT_2_TEXT` is the 200-token opening of `README.md`)*.

### Generated Token ID Table (32 Tokens)

| Step | Prompt 1 ID | Prompt 1 Piece | Prompt 2 ID | Prompt 2 Piece |
|---|---|---|---|---|
| 0 | 11751 | ` Paris` | 12182 | ` repository` |
| 1 | 13 | `.` | 9251 | `](` |
| 2 | 198 | `\n` | 2349 | `https` |
| 3 | 760 | `The` | 1074 | `://` |
| 4 | 6511 | ` capital` | 5039 | `github` |
| 5 | 314 | ` of` | 877 | `.com` |
| 6 | 9338 | ` France` | 14 | `/` |
| 7 | 369 | ` is` | 1825 | `AG` |
| 8 | 11751 | ` Paris` | 42427 | `ENTS` |
| 9 | 13 | `.` | 14 | `/` |
| 10 | 198 | `\n` | 1825 | `AG` |
| 11 | 760 | `The` | 42427 | `ENTS` |
| 12 | 6511 | ` capital` | 33673 | `/blob` |
| 13 | 314 | ` of` | 14905 | `/main` |
| 14 | 9338 | ` France` | 14 | `/` |
| 15 | 369 | ` is` | 52838 | `README` |
| 16 | 11751 | ` Paris` | 20668 | `.md` |
| 17 | 13 | `.` | 553 | `).` |
| 18 | 198 | `\n` | 271 | `\n\n` |
| 19 | 760 | `The` | 760 | `The` |
| 20 | 6511 | ` capital` | 1510 | ` ` |
| 21 | 314 | ` of` | 49 | `` ` `` |
| 22 | 9338 | ` France` | 8065 | `R` |
| 23 | 369 | ` is` | 9477 | `[x` |
| 24 | 11751 | ` Paris` | 7 | `]/` |
| 25 | 13 | `.` | 87 | `(` |
| 26 | 198 | `\n` | 61 | `x` |
| 27 | 760 | `The` | 17 | `^` |
| 28 | 6511 | ` capital` | 471 | `2` |
| 29 | 314 | ` of` | 19638 | ` -` |
| 30 | 9338 | ` France` | 44646 | ` sigma` |
| 31 | 369 | ` is` | 17189 | `)` |

**Comparison Result:** 32/32 token IDs bit-identical across both prompts.

---

## 3. Negative Control (RoPE Pairing Mutation)

- **Control Mutation:** In `/home/socol/oc-tmp/q35-t0/reference_mutated.py`, changed RoPE rotary pairing from NeoX halves `(j, j + 32)` to adjacent pairs `(2j, 2j + 1)` on dims `0..64`.
- **Command:** `python3 /home/socol/oc-tmp/q35-t0/reference_mutated.py --check`
- **Output:**
```text
=== Running --check verification against llama.cpp greedy decoding ===
Prompt 1: 'The capital of France is' (5 tokens)
Prompt 1 greedy 32 IDs match: True
Prompt 2: README.md opening (200 tokens)
Prompt 2 greedy 32 IDs match: False
  P2 mismatch at step 31: got 45309 expected 17189, top2 gap=0.155004
```
- **Exit Status:** 1 (falsifiable gate confirmed).
- **First Divergence:** Step index 31 of Prompt 2: mutated reference produced `45309` (`' algebra'`) instead of expected `17189` (`' crate'`), with a top-2 logit gap of `0.155004`.

---

## 4. Verification Gate Command & Timing

```bash
python3 tools/qwen35/reference.py --check
```
- **Exit Status:** 0
- **Elapsed Time:** ~27 seconds
- **Line Count:** `tools/qwen35/reference.py` has 379 lines (contract limit: $\le 600$ lines).
- **Files Owned / Modified:**
  - `tools/qwen35/reference.py` (new)
  - `docs/evidence/qwen35-reference.md` (new)
