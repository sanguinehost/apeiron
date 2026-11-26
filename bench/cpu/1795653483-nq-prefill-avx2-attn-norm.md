# nq: AVX2 quantization, the attention pack and score, the fused norm

Criterion: **T2** (turn latency), partial: the cold 1,827-position CPU prefill of the 0.6B pack at
block 1024. Every change here is bit-exact on the PPL stream, so this is not a quality claim. It is
not a 7B result, not a 200k turn and not a decode result.

Grades: `measured` for every wall, paired ratio, PPL and phase split; `tested` for the equality
tests and the mutation falsifiers. No `proved` claim: the changed bodies sit in the extraction
closure (`cols.rs`, `step.rs`, `nq.rs`), but the committed extraction predates them and the Lean
bridge did not run here (Omissions). Machine: OVH box `ubuntu@139.99.223.161` (Sydney; 8 vCPU EPYC
Milan, 8 sockets × 1 core × 1 thread, 31.3 GiB, 256 MiB L3, 2.4 GHz base, no AVX-512, no VNNI), no
GPU. The laptop — the only device a timed claim may name — is **not measured**. Times UTC; every
job under `~/bin/adg run --` (16 GiB cap, 8 jobs, swap zero).

Model: `~/models/qwen3-0.6b-base-nq/nq-pack/skeleton-q8head.gguf`; pack
`~/models/nq-scratch/nq06-v3.nqp` (196 linears, 28 layers, 0.977441 b/w, `k = 0` salient,
`res_layers = 0`); harness `examples/prefix_solve.rs`, `--len 1827 --differ 205 --decode 0 --mb
440`, width 8, `APEIRON_PREFILL_COLS=1024`. PPL: `examples/ppl`, 8,176 tokens of
`~/models/wiki.test.raw`, 16 windows of ≤ 512, `--kernel q8act --head-batch 16`. Every timed build
follows `touch src/lib.rs`.

## What changed

HEAD is `bcefc8ef`. The batch is five dials on the one path, none a second projection function.

### `nq_act`: the AVX2 grouped quantizer

`nq_act` (`src/engine/kern/nq.rs`) is the only NQ quantizer call site. Its body is now four `i64`
lanes wide: the group's magnitude maximum decides the scale, and each code is
`sign(x)·floor((2|x| + S)/(2S))`, which for the power-of-two `S = 2^e` is
`sign(x)·((|x| + S/2) >> e)`. `quant_act_grouped` is the scalar twin and the fallback: the vector
body writes exactly its codes and scales, and a value whose scale or code would leave the twin's
domain returns `None` from both. `the_vector_quantizer_equals_the_scalar_quantizer` compares codes
and scales word for word on the pack's lengths and magnitudes, group boundaries at 32 blocks, a
ragged tail, zero groups, ties at the rounding edge, negative extrema and both `i64` end points.
Mutating the vector shift by one reddens it and restoring the source turns it green.

### `swiglu_piece`: the pooled FFN piece onto the slice path

`swiglu_piece` (`src/engine/chart/ffn.rs`) called `swiglu_one_carrier` once per element. It now
calls `swiglu_slice`, the eight-lane AVX2 carrier loop with its scalar twin, which `ffn_q8act_step`
already used. `swiglu_one_carrier` stays as the test oracle and is `#[cfg(test)]`. The slice body's
equality test and mutation control are the ones committed with `swiglu_slice`.

### `QUERY_TILE`: 32 → 64

`src/engine/step/softmax.rs`: consecutive columns per attention tile on the `KvCarrier::F32` path,
now 64, llama.cpp's `GGML_FA_TILE_Q`. The cold read is
`Σ_t (pos₀ + t·QUERY_TILE + QUERY_TILE)` rows instead of `Σ_c (pos₀ + c + 1)`; the score stage
(`rows · 64` words) stays inside L2.

### The attention pack

`attn_core_tile_avx2_inner` (`src/engine/attn/f32_core.rs`) packed each K row with one scalar store
per element (`kt[d·KV_TILE + t]`) and checked V's finiteness one word at a time — four bounds checks
and a scattered store per element. Eight consecutive K rows now go through one 8×8 transpose
(`transpose8_store`), whose store side is eight contiguous eight-lane runs; V is copied by
`copy_finite`, whose finiteness refusal is `finite8`. Both are `unsafe` AVX2 bodies with the scalar
twin named in their docs — `attn_core_f32_tile_scalar`'s per-element pack — and
`test_attn_core_tiled_equals_scalar_on_tile_edges` and
`test_attn_core_query_tile_agrees_with_the_per_query_core` compare the two bodies word for word at
`hd = 8, 16, 64, 128`, strides across the tile edge and 1, 2 and 4 query heads per KV row.
Changing one `permute2f128` lane selector in `transpose8_store` from `0x31` to `0x20` reddens the
second; restoring the source turns it green.

The padded-key zero fill also narrowed: only the last tile's padded columns are written. A full
tile leaves the previous tile's finite words in place, and the score of a padded column is zeroed
before it is read.

### The score stage and `gemm_tile::<INIT>`

The per-row scale-and-maximum is eight lanes at a time on both arms (a per-key `kscale` vector and
the scalar `scale`), with the lane maximum reduced horizontally; the scalar tail keeps the twin's
`is_finite` refusal and its `>` order. Only finite scores are maxed, so lane order does not change
the value: a `+0.0`/`-0.0` tie reaches `exp` as the same argument.

`gemm_tile<const INIT: bool>` is the same microkernel with the accumulator's initial read removed;
the score GEMM is `::<true>` (the tile overwrites `p`, so `p.fill(0.0)` before it is gone) and the
value GEMM stays `::<false>`. This is a const dial on one kernel, not a sibling.

### The fused norm

`src/engine/step.rs` and `src/engine/step/cols.rs`. `rms_with_gain_slice` was
`rms_norm_fixed` (a fresh normalized `Vec`) then a gain multiply into a second `Vec`;
`norm_cols` then ran that per column, wrote the results into one shared `k × n` zeroed plane and
copied each column out. That is four passes over every column the pool already owned.

The body is now `rms_root` (the `i128` sum of squares, rounded mean, `eps` add and `fixed_rsqrt`,
the same checks as `rms_norm_fixed`) plus `rms_gain_word` per element, which applies the two
roundings `rms_norm_fixed` then the gain apply, in that order, and keeps both refusal kinds. The
allocating entry is one exactly sized `Vec` per column. `norm_cols` runs `NormColsCtx` over
`chunks_mut_global(&mut cols, ⌈k/(2·width)⌉)`: each participant allocates and writes its own
columns, the shared plane and the `to_vec` copies are gone, and `FirstErr<usize, StepError>` keeps
the lowest failing column as the error. `k < 2`, a ragged list and a single-worker pool keep the
serial arm.

`the_fused_norm_gain_equals_the_norm_then_gain_pair` compares the fused slice with
`rms_norm_fixed` + `round_shift_32(·g/2^32)` on 1, 4, 33 and 1,024-word rows with negative gains,
and checks all three refusals (`EmptyRow`, a zero mean with zero `eps`, a width mismatch). Replacing
`base` with `x` in `rms_gain_word`, i.e. dropping the first rounding, reddens it; restoring the
source turns it green.

### Measured neutral and reverted

Two earlier forms of this batch were built and timed, measured neutral, and reverted so the commit
carries only measured wins:

- a norm that fused the gain **into** the shared plane (`rms_with_gain_into`) instead of removing
  the plane — 5.549 s against the control's 5.531 s;
- a `quantized_band_scaled` that pre-scaled and quantized a column in one pass instead of
  materializing `t` — same window, same neutral result.

Both are inside the paired spread and neither is in the diff. The shipped norm form differs from
the first in the mechanism, not the arithmetic: it removes the plane and the copies rather than
writing the plane in place.

## Result

### Cold prefill: HEAD against the batch, five interleaved rounds

| round | final2 (HEAD) s | batch s | pair |
|:---:|---:|---:|---:|
| 1 | 5.807 | 5.178 | 0.892 |
| 2 | 5.949 | 5.197 | 0.874 |
| 3 | 5.809 | 5.266 | 0.906 |
| 4 | 5.839 | 5.149 | 0.882 |
| 5 | 5.862 | 5.207 | 0.888 |
| mean | **5.853** | **5.199** | **0.888** |

Binary md5s: `final2` `18fe984d…`, batch `bb5da211…` (`prefix_solve.qavx6`). The control's spread
is 5.807–5.949 (2.4% of its mean), the batch's 5.149–5.266 (2.3%). Every one of the five pairs
moves the same way and the ratio is outside both spreads. The rate is **314 → 351 tok/s** at width
8.

The committed source, rebuilt after the gate's doc and visibility edits, is md5 `e26e47a6…`
(`prefix_solve.qavx7`) and reads **5.782 → 5.159 s** over three further rounds (pairs 0.894, 0.894,
0.889; mean **0.892**), PPL 53.991898. The ratio reproduces on the exact committed binary; two
earlier windows read 0.888 on `3a4bc66e…` and `bb5da211…`, so the pair is 0.888–0.892 across three
windows while the absolute walls are not.

The sub-arms, each in its own window, pair as:

| arm | control s | change s | pair |
|---|---:|---:|---:|
| quantizer + `swiglu_piece` + `QUERY_TILE` | 5.838 | 5.612 | 0.961 |
| attention pack + score | 5.473 | 5.283 | 0.965 |
| fused norm | 5.362 | 5.137 | 0.958 |
| batch vs HEAD (table above) | 5.710 | 5.071 | 0.888 |

The sub-arm windows are different windows and are not summed; the batch row is the graded number.

### Quality

| arm | PPL | mean nll | tokens |
|---|---:|---:|---:|
| final2 (HEAD) | 53.991898 | 3.988834 | 8,176 |
| batch | **53.991898** | **3.988834** | 8,176 |

Same stream, same 16 windows of ≤ 512, `--kernel q8act`. Every change is word-exact on the values
it touches, so the batch reproduces the PPL exactly rather than approximately. No PPL movement is
claimed or observed.

### Phase split

`--phases`, one cold run of the batch binary, same window.

| phase | ms | share |
|---|---:|---:|
| `ffn` | 1892.369 | 36.7% |
| `attn.core` | 1486.505 | 28.9% |
| `attn.proj_in` | 750.387 | 14.6% |
| `attn.proj_out` | 294.457 | 5.7% |
| `norm` | **160.906** | **3.1%** |
| `residual` | 144.366 | 2.8% |
| `pool.wait` | 276.530 | 5.4% |
| `head` | 5.854 | 0.1% |
| `pass` | 4780.856 | 92.8% |
| cold wall | 5.152 s | |

The leaves sum to 4,779.8 ms. The three projection leaves are 2,937.2 ms (57.0% of the wall), so a
further projection change inherits **`1/(1 − 0.570) = 2.33×`**; `attn.core` is 28.9% and inherits
**`1.41×`**. The norm leaf moved **498.5 → 149.0 ms** in the window that paired the norm change
(the qavx4 split, 5.417 s); the committed split reads 160.9 ms for the same leaf, and at 3.1% it is
no longer a target.

### Against llama.cpp

llama.cpp at `049326a00`, same host and window, `llama-bench -m Qwen3-0.6B-Base.f16.gguf -p 1827
-n 0 -t 8 -r 3`:

| arm | tok/s | wall s |
|---|---:|---:|
| llama.cpp F16 | **463.49 ± 0.67** | **3.942** |
| batch (q8 pack), `qavx6` | 351.5 | 5.199 |
| batch (q8 pack), committed `qavx7` | 354.2 | 5.159 |

The in-window ratio is `5.159 / 3.942 = 1.31×` behind (`5.199 / 3.942 = 1.32×` on the pre-lint
binary). The target is 1×–0.5× (3.942–1.971 s): parity needs the wall to fall a further **1.31×**
(1.22 s) and half a further **2.62×**. An earlier window read 1.30× against llama's 3.894 s. The
batch moved 0.888–0.892 from 1.36×; the remainder is the projection staging (57.0% leaves, of which
the dot family is about a third) and `attn.core` (28.9%), not the norm or the residual.

## Omissions

- The laptop is not measured. The box is shared and its absolute walls are window-specific; the
  interleaved pair is the graded number.
- The extraction and bridge are stale: `proof/regen.sh --check` has not run since `57b4352d`, no
  pinned Aeneas/Charon/Lean is installed on either host, and `norm_cols`' pooled error shape and the
  new AVX2 bodies are in the Charon closure. The changed bodies are therefore `tested`, not
  `proved`; the maintainer gate inherits them.
- `cargo test --workspace --all-features` needs the `glslc` pin and a GPU device; neither host has
  them. On the box, the committed source is green on `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`,
  `tools/check-claims.py`, `tools/book-check.py`, `mdbook build book` and `mdbook test book`;
  `cargo test --workspace --locked` is **684 passed, 1 failed, 54 ignored**, and the failure is the
  documented `gpu::nn::coopmat_gemm::tests::coopmat_spirv_is_the_glsl` byte check, which needs
  `glslc`.
- The phase split is a single run; the batch wall in the split (5.090 s) is a different run from
  the paired mean (5.071 s). No unchanged leaf's movement is attributed.
- The sub-arm rows are four windows; only the batch row is interleaved against HEAD in the graded
  window.
- No decode, turn, 7B, context or GPU claim.
