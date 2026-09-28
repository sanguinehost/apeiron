# Engine throughput to the R0 floor

- **Owner:** lead (Claude track)
- **Status:** W2: T6 tested (`src/engine/chart/{mod,x86}.rs`: `matvec_q8act` == `matvec_fixed ∘ dequant ∘ quant_act` bit for bit on six families; AVX2 == portable; control: nibble-plane swap → 3/7 red); T7 proved (`Numeric/ChartDot.lean`: `qcode_range`, `qcode_err`, `qcode_smul`, `chart_factor`, `row_err_bound`; registered; control: truncation → red); T8 tested (`src/engine/chart/ffn.rs`, `KernelMode` in `step.rs`; control: zeroed `up` activation → red). Parity under `APEIRON_KERNEL=q8act` (`examples/lm_eval_bridge.rs`), same 20+20 arc_easy/piqa slice: chosen answer equals `Exact` 40/40 and Hugging Face 40/40. Max |Δ continuation log-likelihood| vs `Exact` is 0.538 (arc_easy) and 0.741 (piqa). Results: `/home/socol/oc-tmp/lm-eval/engine-q8-*`. The wave wall time of 85 s against 356 s is not a measurement (busy host). T9 needs a quiet host. W1: W1: T1, T3 and T4 done (tested; bit-identical; indicative on a busy host). T1 went 1.43 → 1.91 tok/s. T4 (row-parallel FFN): 2.0 → 8.2 tok/s with identical generated ids; the FFN phase is still 51–66% of a step. T2 proved (`Numeric/MatvecChunk.lean`: `matvec_fixed_rows`, `matvec_fixed_chunk`, `rowOut_join`, `matvec_fixed_chunks`; registered; control: chunk offset +1 → red). Residual: `matvec_fixed_chunks` takes each chunk's row map as the hypothesis `houtmaps`, and the `par.rs` split is not yet linked to the chunk list. T5 (`fused.rs`) is tested and bit-identical.
- **Epic ID:** `APEIRON-TPUT`
- **Parent:** R0 of [`frontier-consumer`](frontier-consumer.md) (coherent generation at ≥ 30 tokens/s)

## Intent

The proved engine answers identically to the Hugging Face reference on the E1 slice (40/40 chosen
answers). It runs at 1.6 tok/s single-stream on Qwen2.5-0.5B `Q4_K_M` (E1, host not quiet), about 20×
below R0.

Cause, read from source:
- every projection and the LM head call `Engine::read_range` (`src/engine/loader.rs:766`), which
  opens the GGUF file, seeks and copies the tensor bytes on every step;
- every step then dequantizes and reduces sequentially on one core of 32.

Changes that do not touch an extracted kernel keep every refinement theorem valid:
1. **Resident weights:** read each tensor's payload once and serve borrowed slices.
2. **Row-parallel projections:** split each matrix's rows into contiguous chunks and call the proved
   `matvec_fixed` per chunk on scoped threads. The concatenation equals the single call (tested now;
   provable later as a chunking lemma over the extracted row loop).

GPU (the SPIR-V DP4A path) and mmap for models larger than RAM are later tasks.

Invariant: logits are bit-identical before and after every change, because these are data-movement
and scheduling changes only.

## Plan

| task | files | gate | control |
|---|---|---|---|
| T1 resident payload, borrowed slices, row-parallel projections | `src/engine/loader.rs`, `src/engine/step.rs` | `cargo test --workspace --all-features --locked` + bit-identical logits on a fixed prompt | perturb one chunk boundary → the bit-identity test red |
| T2 chunking lemma: `matvec_fixed` on the concatenation = concatenation of chunk calls | `proof/corpus/Numeric/MatvecChunk.lean` | module gate | off-by-one chunk start → red |
| T3 quiet-host measurement: tok/s before/after, named machine | none | receipt under `docs/evidence/` | a deliberately slowed build must register slower |

## W2 — integer chart dot, ported from fold

### Intent

The W1 levers are exhausted. Resident weights and row threads went from 1.6 to about 10 tok/s. The
fused block-wise dot (`src/engine/fused.rs`) removed per-element dequantization and measured
little. The remaining cost is the per-element `i128` multiply of a small code by a Q32.32
activation.

`fold` already closed this gap with measured kernels.

| source (read) | what it is | measured there |
|---|---|---|
| `fold/bedrock/src/serve/model/rowdot.rs:52,403,517` | AVX2 `row_dot_q4_avx2`, `row_dot_q2_avx2`, `row_dot_q2sb_avx2`: `vpmaddubsw`+`vpmaddwd` over u8 codes × int8 activations, one i32 per block, deferred horizontal sum | 43.8 GB/s against a measured 66.0 GB/s host read, and llama.cpp at 43.0 on the same file (`fold/results/vs-llamacpp-cpu-27b.md`, i9-14900KF) |
| `fold/bedrock/src/serve/model/q1.rs:60,121` | `row_dot_q1_scalar` / `row_dot_q1_avx2`: sign-code dot with no weight multiply, a transcription of llama.cpp `ggml_vec_dot_q1_0_q8_0` | bit-identical scalar twin (`to_bits` gate) |
| `fold/modelfold/src/simt/dp4a.rs` | SPIR-V `OpSUDot` (DP4A) GEMV, k-packed, with a dynamic per-32 activation scale and an f16-packed (d, lo) block-parameter word | ~890 GB/s on this RTX 5080 (`fold/results/gpu-bandwidth.md`) |
| `fold/results/decode-ceilings.md` | Qwen3-30B-A3B at Q4, CPU | 17.94 tok/s, 90% of its bandwidth ceiling |

These numbers are `fold`'s receipts, not measurements here (a document number is a hypothesis
until re-run). They set the target: decode is bandwidth-bound once the inner dot is an integer
instruction, so tok/s ≈ bandwidth ÷ bytes per token.

### Spec

- **Chart factoring (exact).** For a block `w_j = d·q_j + lo`, over any commutative ring:
  `Σ_j w_j·x_j = d·Σ_j q_j·x_j + lo·Σ_j x_j`. `fused.rs` already uses this on Q32.32 `x`.
- **Activation quantization (the one semantic change).** `quant : Q32.32^n → Q32.32^n`. For each
  32-wide block `b`: `S_b = ceil(absmax_b / 127)` as a Q32.32 raw integer, and
  `xq_j = round(x_j / S_b) ∈ [−127, 127]`, round to nearest with ties away from zero. The result is
  `x̂_j = S_b·xq_j`, which is exact in Q32.32. This is the llama.cpp `q8_0` activation shape. The
  error bound is `|x_j − x̂_j| ≤ S_b/2`.
- **Definition of the fast path:** `matvec_q8act(W, x) := matvec_fixed(W, quant(x))`, **bit for bit**,
  including the `Err` value. The kernel computes it by factoring. Take a group `g` inside
  activation block `b` with `w_j = W_g·q_j − M_g`. Then
  `Σ_j w_j·x̂_j = Σ_b S_b·Σ_{g⊆b} (W_g·Σ_j q_j·xq_j − M_g·Σ_j xq_j)`. The inner sums are exact `i32`
  (`maddubs`/`madd`, DP4A), and the outer products use `i128`. The kernel ends with the same single
  `round_shift_32`. Wherever a bound cannot rule out overflow, it falls back to `matvec_fixed`, as
  `fused.rs` does. The portable, AVX2 and DP4A faces are then all equal to the proved `Exact` path
  on `quant(x)`: there is no float epilogue, and so none of the reduction-order hazard `fold`
  documents at `q1.rs:62`.
- **Mode switch.** `KernelMode::{Exact, Q8Act}`. `Exact` is unchanged. `Q8Act` = `Exact ∘ quant`.
  Its whole deviation from `Exact` is the quantizer, bounded per row by
  `Σ_b (S_b/2)·Σ_{j∈b} |w_j|`.
- **Trust boundary.** The AVX2 intrinsics and the SPIR-V shader are not extracted. Their equality
  with the portable twin is `tested`. The portable twin and the error bound are the `proved`
  targets.

### Plan

`src/engine/` already holds 21 source files, over the 12-per-directory limit. New code goes in
`src/engine/chart/`, and only its one `mod` line enters `src/engine/mod.rs` (lead).

| task | files | gate | control |
|---|---|---|---|
| T6 `quant`, and portable `matvec_q8act` for the `fused.rs` set (Q2_K, Q3_K, Q4_K, Q5_0, Q6_K, Q8_0) with the fallback; AVX2 i32 group cores ported from `fold` `rowdot.rs` behind `is_x86_feature_detected!` | `src/engine/chart/mod.rs`, `src/engine/chart/x86.rs` (new), one `mod chart;` line in `src/engine/mod.rs` | `cargo test --workspace --all-features --locked chart`: `matvec_q8act == matvec_fixed(·, quant(x))` bit for bit on every family, on random, extreme (±127, maximal codes) and ragged-tail inputs, with AVX2 on and forced off | swap the two nibble planes in the AVX2 Q4_K core → red |
| T7 Lean: `quant` spec (range `[−127,127]`, `|x − x̂| ≤ S/2`, idempotence on the grid) and the factoring identity | `proof/corpus/Numeric/ChartDot.lean` (new) | `lake build Numeric.ChartDot` | drop the `/2` in the bound → red |
| T8 `KernelMode` through `par`/`ffn`/`step` (row-chunked `matvec_q8act`) | `src/engine/par.rs`, `src/engine/step.rs`, `src/engine/ffn.rs` | `cargo test`; E1 lm-eval slice still 40/40 under `Q8Act`; greedy-id diff vs `Exact` reported | route `Q8Act` through a zeroed scale → the parity test red |
| T9 quiet-host measurement: tok/s and achieved GB/s vs measured host read bandwidth, `Exact` vs `Q8Act` vs llama.cpp on the same GGUF | none | receipt under `docs/evidence/` | a single-threaded `Q8Act` build must register slower |
| T10 GPU execution: port the `fold` emitter (`modelfold/src/simt/dp4a.rs`, `gemv_dp4a_module_dyn`) and its raw-FFI Vulkan runtime (`modelfold/src/gpu/{vulkan,runtime}.rs`, no `ash` dependency) into a new `src/gpu/`. The emitted words pass `src/spirv/check.rs` before dispatch, and the per-32 `S_b` of `quant` is the kernel's dynamic activation scale. The proof side is already the [`gpu-spv`](gpu-spv.md) epic (`Spirv/Dp4a.lean`: `gpu_dp4a_word_refines_cpu_word`). | own wave, after T8 | GPU `matvec_q8act` == CPU `matvec_q8act` bit for bit on every family | drop the row guard → the `Spirv.Check` gate refuses |
