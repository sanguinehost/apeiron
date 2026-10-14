# Exact scan: a versioned numerical contract for chunked exact accumulation

## Intent

- **Owner:** lead (AI track); bounded workers take the per-wave tasks below
- **Source revision:** `6a639bde533a293e7026fc126a4f47144f5289d7`; working tree dirty (untracked
  `.held-control/`, `.held-scratch/`, `libtest_closure_capture.rlib`, `proof/timeout`,
  `test_closure_pointer`, `test_edition2`)
- **Status:** draft

### Problem

Prefill runs at 0.51× llama.cpp: `docs/evidence/bench-ledger.md#1790620216-ns-r003` measures apeiron pp
median 299.50 tok/s against llama.cpp pp 585.48 (per-round ratio median 0.51, min 0.41, max 0.52) on
`Qwen3.5-0.8B-Q8_0.gguf`, pp 200, 5 quiet rounds; tg ratio is 1.24. The same receipt attributes
1.049 ms/token to `affine.heads` (32.8% of the 3.20 ms/token prefill wall, 90 calls); the share is
29.5% in `1790612485-astra-projection-baseline.md` and 30.5% in
`1790607104-astra-batched-baseline.md`.

`src/engine/mixer.rs` executes the DeltaNet state recurrence one token at a time and rounds to
Q32.32 after every step: `c ← round_shift_32(α · c)` (line 29) with readouts `m = round_shift_32(c · m̃)`,
`d = round_shift_32(β · (v − m))`, `o = round_shift_32(c · õ) + round_shift_32(d · (kᵀ q̃))`
(lines 55–59), refusal at `STATE_BOUND = 1 << 48` (line 81), renormalisation at `RENORM_THRESHOLD`
(lines 31–32, 78). `delta_heads_cols` (line 1050) carries each head slice through all `k` columns in
token order (lines 1131–1148). Per-step rounding is not associative, so the chain cannot be
reassociated into chunk-shaped work; the algebra of the affine monoid is associative
(`face_recurrence_below`).

`src/engine/attn.rs` computes softmax with an exact max shift (`exp_at`, line 127) and rounds every
probability by a round-to-nearest division (`weight_at`, line 136) inside the per-position loop
(lines 502–540). Its state is the materialised `(pos + 1)`-long score/exp/weight vector, so prefill
attention is `O(pos²)` invocations of `fixed_exp` and one division per weight.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Keep per-step rounding (status quo) | Measured 0.51× at pp200; each of C sequential steps is one 128×128 read pass plus one write pass; the measured `affine.heads` share is 29.5–32.8% of a 3.20 ms/token wall |
| Reassociate the chunk but keep Q32.32 rounding after every step | `round_shift_32` is not a ring homomorphism; `ringMul_assoc` holds over the ring, not over the rounded words (`docs/epics/one-executor.md` records "chunked or WY evaluation bit-identical" as not relied on) |
| Replace the fixed-point contract with float/FMA accumulation | Changes the denominator and llama.cpp id parity, and has no in-repo refinement or graded error bound |
| Assert only a tolerance on sampled logits | Unfalsifiable at scale; gives no bound to test and no refusal semantics |
| Carry the state as an unbounded exact rational end to end | State width grows with token count; no bounded overflow refusal and no finite carrier |
| Attention: recompute the full row after each prefill chunk | Retains the `O(pos²)` `fixed_exp`/division count that the online recurrence removes |

### Completion condition

`APEIRON_EXACT_SCAN=1` runs pp200/tg64 on `Qwen3.5-0.8B-Q8_0.gguf` at a median paired prefill ratio
`≥ 1.00` against llama.cpp `0324696b8` over 5 quiet rounds, with greedy ids equal to llama.cpp and
the argmax/logit bound below held; the extraction and corpus builds are fresh.

```sh
cargo test --workspace --all-features --locked
proof/regen.sh --check
cd proof/corpus && lake build
```

and, on the same host, `examples/engine_profile` pp200/tg64 with `APEIRON_EXACT_SCAN=1` against
`llama-bench`.

### Invariants

1. The per-step Q32.32 path stays on the build path, selectable (`APEIRON_EXACT_SCAN=0`), and remains
   the oracle; its word outputs never change.
2. Inside a chunk the affine composition is exact over the chunk carrier; `round_shift_32` runs only
   at a chunk boundary and at readout.
3. Overflow is refused with `StepError::Overflow` before any state word is written; no wrap, clamp or
   saturation is introduced.
4. Existing model parity (`qwen35_on_the_one_executor_decodes_llama_cpp_ids`,
   `qwen35_long_prompt_on_the_one_executor`) keeps passing unchanged for the default path.
5. `sigma` stays a runtime value in the executing signature (`Nous.Model.sigmaValue`,
   `sigmaOfFace`, `dispatchAtFace`); no face is specialised into a separate path.

### Verification hypotheses

| Claim | Grade | Evidence (command or theorem) |
|---|---|---|
| The affine step composes as a monoid and a chunk's steps compose to one step | proved | `Nous.Model.affComp_assoc` (FaceRecurrence.lean:65), `recur_eq_scan` (:158), `scan_split` (:179), `#audit` Remaining.lean:634–638 |
| Chunked evaluation equals sequential evaluation of the composed step | proved | `Nous.Seq.GatedDelta.recurrent_eq_chunked` (GatedDelta.lean:210), `wy_form` (:332), `#audit` Remaining.lean:863–866 |
| The chunk-boundary rounding error is bounded and does not grow with length | proved | `Nous.Seq.ChunkedRounding.chunked_discrepancy_uniform_bound` (ChunkedRounding.lean:91), `witness_delta_net_100_lsb` (:109), `#audit` Remaining.lean:1029–1034 |
| The chunk kernel returns the composed step's state and refuses on width | tested | `cargo test --locked --lib engine::mixer::tests` |
| Online `(m, den, num)` attention is exact softmax | proved (sibling package) | `Substrate.Algebra.Neural.softmax_attention_is_online_spinor`, `oaDen_fuse`, `oaNum_fuse`, `aff_chunked_eq_full` (`../sanguine-ref/proof/Substrate/Algebra/Neural/AttentionSpinor.lean:462,419,424,681`) |
| The fixed-point attention epilogue matches `attention_channel_form` on `attn_domain` | anchor → proved | `Numeric/Engine/Attn.lean` (`attnScoreInt`, `numeratorReal`, `softmaxWeights`), `Numeric/Attn/SoftmaxPerturb.lean:320 attn_output_perturb`; the naive statement is refuted at `not_attn_head_refines` (:92) |
| Prefill ratio ≥ 1.00 vs llama.cpp at pp200 | measured | `examples/engine_profile` + `llama-bench`, i9-14900KF, median paired ratio over 5 quiet rounds |
| Greedy argmax equals llama.cpp at every pp200+tg64 position; max per-logit difference ≤ 256 LSB (1 LSB = 2⁻³²) | tested | oracle test in `src/engine/tests.rs`; the 256 LSB constant is `anchor` until W1-T1 fixes it from the proved bound |

### Non-goals

- No change to the decode path's contract, the model weights, the quantization formats, or the
  definition of llama.cpp id parity.
- No bit-identical claim against the per-step rounded path; only the bounded discrepancy above.
- No GPU/SPIR-V, no new architecture, no new addressing or lookup mechanism.
- No merge of the `../sanguine-ref` and `../sanguine` repositories; their theorems are cited
  pointers until a link is built on this repository's toolchain.
- No new benchmark harness; the existing `examples/engine_profile` and quiet-gate protocol are reused.

### Trust boundary

The measured ratio covers `src/engine` on Q8_0 at pp200/tg64 on one i9-14900KF and does not transfer
to other models, quantizations or hosts. The proofs reach the affine monoid, the chunk composition
and the rounding-error recurrence; they do not reach rustc/LLVM codegen, AVX2 instruction selection,
the pool scheduler, the `unsafe`-free container's allocation behaviour, or the kernel ABI. The
fixed-point refinement covers the extracted body over the stated `attn_domain`; positions outside it
are refused, and `not_attn_head_refines` records that an unrestricted statement is false.

### Open questions

- `CHUNK_MAX` for the state kernel: fixed at 64, or derived per layer from `STATE_BOUND` and the
  decay? Owner: lead; decided in W1-T1 from the width derivation.
- Whether the attention fusion lands as one pass over the prefill columns or per column. Owner: lead;
  decided in W1-T2 by the measured `attn.core` share (0.280 ms/token pp).
- Whether the flag flips default-on at W5 or stays opt-in. Owner: lead; decided by the measured ratio.

## Specification

- **Owner:** lead
- **Intent:** this file, §Intent
- **Source revision:** `6a639bde533a293e7026fc126a4f47144f5289d7`; working tree as above

### Scope and admissible domain

| Item | Value |
|---|---|
| Operation | new `delta_head_chunk` (`src/engine/mixer.rs`); new online attention read (name fixed in W1-T2, `src/engine/attn.rs`) |
| Domain | chunk `C ∈ 1..=CHUNK_MAX`, `CHUNK_MAX` derived in W1-T1; head `q, k` L2-normalised with `‖k‖₂ = 1`; `v`, `β`, `α` finite Q32.32; every state word `|S̃| ≤ STATE_BOUND = 1 << 48`; attention `pos < occupied[layer]` and `head_dim` inside the `fixed_rsqrt` window |
| Totality | `Result<_, StepError>` (`delta_head_chunk`) and `Result<_, AttnError>` (attention read) |
| Preconditions | geometry widths as `delta_heads_cols` checks today (`conv = 2·key_dim + val_dim`, `z = val_dim`, `b = a = n_v_heads`); `β = σ(b) ∈ [0, ONE]`, `α ∈ (0, ONE]` |
| Postcondition | the chunk's exit state equals the single `affApply` of the chunk's composed `AffineCell` to the entry state, rounded once at the boundary; the readout `o` is the rounded sum of the closed-form read; the attention output equals the one-normalisation quotient of the fused `(m, den, num)` state |

No silent domain reduction: an out-of-domain input is a named refusal, never a clamped value.

### Representation relation (Rust to Lean)

| Layer | Symbol | Path |
|---|---|---|
| Production Rust, state | `delta_head_chunk` (new) | `src/engine/mixer.rs` |
| Production Rust, attention | fused online read (new) | `src/engine/attn.rs` |
| Extraction wrapper | `word_delta_head_chunk` | `proof/extract/lib.rs` |
| Extracted body | `engine.mixer.delta_head_chunk` | `proof/ApeironWord.lean` (generated) |
| Canonical definition, monoid | `Nous.Model.affApply`, `affFold` | `proof/corpus/Nous/Model/FaceRecurrence.lean` |
| Canonical definition, chunks | `Nous.Seq.GatedDelta.affFold`, `recurrent_eq_chunked`, `wy_form` | `proof/corpus/Nous/Seq/GatedDelta.lean` |
| Error recurrence | `Nous.Seq.ChunkedRounding.stepDiscrepancy`, `chunked_discrepancy_uniform_bound` | `proof/corpus/Nous/Seq/ChunkedRounding.lean` |
| Refinement theorem | `word_delta_head_chunk_denote` (new) | `proof/corpus/Numeric/ExactScanRefinement.lean` (new), registered in `proof/corpus/WordRefinement.lean` |
| Attention canonical, ring | `channelAttnScan`, `attention_channel_form` | `proof/corpus/Nous/Model/Attention.lean` |
| Attention canonical, online | `oaFuse`, `oaDen_fuse`, `oaNum_fuse`, `softmax_attention_is_online_spinor` | `../sanguine-ref/proof/Substrate/Algebra/Neural/AttentionSpinor.lean` |
| Attention fixed point | `attnScoreInt`, `numeratorReal`, `softmaxWeights`, `denseAttnReal`, `attn_domain` | `proof/corpus/Numeric/Engine/Attn.lean` |

The wrapper fixes a type; it does not restate the formula. The refinement theorem is a link only if
it resolves and the axiom audit passes.

### Where rounding happens

| Stage | Carrier | Rounding |
|---|---|---|
| Chunk-internal step products and sums | `i128` (from `mul_q32` products unrounded), checked | none |
| Chunk boundary (state) | Q32.32 `i64` | one `round_shift_32` per composed word |
| State renormalisation (`c < 2⁻⁸`) | Q32.32 `i64` | one `round_shift_32` per word, unchanged |
| Readout `m`, `d`, `o` | Q32.32 `i64` | as `mixer.rs:55–59` today; the chunk changes the state path, not the readout formula |
| Attention tile `exp` | `fixed_exp` window `[-16·ONE, 0]` | one `fixed_exp` per (score, tile) |
| Attention `den`, `num` | exact `i128` sums of tile values | none inside a chunk |
| Attention output | Q32.32 `i64` | one `weight_at` division per output element after the final `(m, den, num)` |

### Widths and overflow refusal

- State words stay under `STATE_BOUND = 1 << 48` (`src/engine/mixer.rs:81`); the read-pass
  accumulation bound `128 · 2⁸⁰ = 2⁸⁷ < 2¹²⁷ − 1` and the AVX2 lane bound
  `128 · (2⁴⁸ + 2³³) < 2⁶³ − 1` (lines 36–45) are re-derived for the C-step chunk and recorded in the
  function comment; `CHUNK_MAX` is the largest `C` for which both hold at the layer's `head_k`.
- Every product uses checked `i128`; every narrowing to `i64` uses `i64::try_from` and maps failure to
  `StepError::Overflow` before any write.
- Attention `den`, `num` are checked `i128` sums; `total == 0` is `AttnError::ZeroSum`.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `chunk_eq_composition` | for a chunk of C steps, the exit state equals `affApply` of `affFold` over the chunk rounded once | proved | `word_delta_head_chunk_denote`, reduced to `recurrent_eq_chunked` |
| `chunk_matches_oracle` | on the admissible domain, the chunk kernel's outputs equal the C one-step `delta_head_step_ref` outputs under the discrepancy bound | tested + proved | `cargo test --locked --lib engine::mixer::tests`; bound from `chunked_discrepancy_uniform_bound` |
| `error_bound` | `|X_L − x_L| ≤ ε·(1 − α^L)/(1 − α) ≤ ε/(1 − α)` with `ε = 1` LSB, `α` the chunk's maximum decay | proved | `chunked_discrepancy_uniform_bound`, instantiated at the Q32.32 carrier in `Numeric/ExactScanRefinement.lean` |
| `attention_fuse_exact` | `oaDen`/`oaNum` of a fused tile equal the sum of the tiles' den/num; chunked = full | proved | `oaDen_fuse`, `oaNum_fuse`, `softmax_attention_is_online_spinor`, `aff_chunked_eq_full` |
| `attention_fixed_point_denote` | the fixed-point read on `attn_domain` refines `attention_channel_form`'s per-token weighted sum | anchor → proved | `EngineAttnRefinement.attn_domain`; `AttnSoftmaxPerturb.attn_output_perturb`; domain excludes `pos ≥ occupied` (`not_attn_head_refines`) |
| `refuses_on_width` | a state word over `STATE_BOUND` or a non-positive `c` returns `StepError::Overflow` and writes nothing | tested | boundary tests in `src/engine/mixer/tests.rs` |
| `flag_off_is_baseline` | `APEIRON_EXACT_SCAN=0` executes the existing per-step path bit for bit | tested | parity tests at the default |

The error bound is between the exact-scan contract and the per-step contract: the per-step rounded
sequence `x_L` is the reference, the chunked sequence `X_L` the candidate, `ε` the per-boundary
rounding noise and `α` the decay. The bound is uniform in `L`; it is not an equality and not a
bit-parity claim. The state bound for `α = 99/100`, `ε = 1` is 100 LSB (`witness_delta_net_100_lsb`).

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| state word over bound | `StepError::Overflow` | `|S̃| > STATE_BOUND` (1 << 48) |
| deferred decay non-positive | `StepError::Overflow` | `c ≤ 0` |
| chunk length outside `1..=CHUNK_MAX` | `StepError::BadConfig` | geometry check, before any evaluation |
| `i128` product/sum or `i64` narrowing out of range | `StepError::Overflow` | checked arithmetic |
| degenerate `eps`/norm input | `StepError::Norm` / `Overflow` | `l2_normalize` |
| softmax total zero | `AttnError::ZeroSum` | `total == 0` |
| `exp` shift below the window | numerator saturates to 0 | documented, not a refusal (`attn.rs:127`) |
| attention `pos ≥ occupied[layer]` | `AttnError::PositionOutOfRange` | outside `attn_domain`; the unrestricted refinement is false |

### Evidence plan

1. **Positive tests.** `cargo test --workspace --all-features --locked`; new kernel cases in
   `src/engine/mixer/tests.rs` and attention cases in `src/engine/tests.rs`, plus the two existing
   Qwen3.5 parity tests at both flag values.
2. **Proof.** `proof/regen.sh --check` regenerates and audits the extraction; `cd proof/corpus &&
   lake build Numeric.ExactScanRefinement AxiomAudit` checks the refinement headline.
3. **Axiom cone.** `#audit` lines for the new headline in `proof/corpus/AxiomAudit/Remaining.lean`;
   anything outside `{propext, Classical.choice, Quot.sound}` fails the build.
4. **Negative control.** Per task below; each mutates the real source or its wrapper in a throwaway
   copy and shows the named gate red, then restores it and shows green.
5. **Measurement.** `examples/engine_profile` pp200/tg64 with `APEIRON_EXACT_SCAN=0|1`, interleaved
   with `llama-bench` `0324696b8` at 8/16/24/32 threads, 5 quiet rounds, i9-14900KF; metric is the
   median paired prefill ratio, before 0.51.

### Trust boundary

As §Intent. The atomics are the error-bound theorem at the Q32.32 carrier: the field-level bound is
already audited, but the concrete `round_shift_32` model and the `i128` width derivation are this
epic's new proof obligation, not a citation.

## Plan

- **Owner:** lead
- **Intent:** this file, §Intent
- **Spec:** this file, §Specification
- **Source revision:** `6a639bde533a293e7026fc126a4f47144f5289d7`; working tree as above

### Goal and completion condition

Every task gate green with its negative control run; the barrier set below plus the measured ratio
and parity bound.

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
```

Bounded as in AGENTS §6/§10: Lean under `systemd-run --user --scope -p MemoryMax=12G
-p MemorySwapMax=0`, `LEAN_NUM_THREADS=1`, `timeout 540`, serialized on
`flock /tmp/opencode/apeiron-lane.lock`. Shared lanes: `proof/ApeironWord.lean`, `target/`,
`proof/.lake/`, `proof/corpus/.lake/`.

### Non-goals

- No default flip before the measured ratio; the flag stays `APEIRON_EXACT_SCAN=0` until W5.
- No change to `round_shift_32`, `mul_q32`, the quantizers or the readout formulas.
- No new crate, no new benchmark binary, no model file change.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | T1 exact chunked state step | `src/engine/mixer.rs`, `src/engine/mixer/tests.rs` | mixer worker | `cargo test --locked --lib engine::mixer::tests` | round inside the chunk instead of at the boundary → `chunk_matches_oracle` red |
| W1 | T2 online attention fusion | `src/engine/attn.rs`, `src/engine/tests.rs` | attention worker | `cargo test --locked --lib engine::tests` | fuse with the first tile's max only → fused-vs-full test red |
| W2 | T1 prefill wiring and flag | `src/engine/step/cols.rs`, `src/engine/step.rs` | lead | `cargo test --locked --lib engine::step` | force chunk routing at `APEIRON_EXACT_SCAN=0` → baseline bit-parity red |
| W2 | T2 extraction roots for the new body | `proof/extract/lib.rs`, `proof/regen.sh` | lead (shared lane) | `proof/regen.sh --check` | seeded defect in the wrapper → `regen.sh --check` reports drift |
| W3 | T1 Q32.32 refinement and audit | `proof/corpus/Numeric/ExactScanRefinement.lean` (new), `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit/Remaining.lean` | proof worker | `cd proof/corpus && lake build Numeric.ExactScanRefinement AxiomAudit` | rounding index in the wrapper shifted by one word → headline red |
| W4 | T1 oracle parity and bound gate | `src/engine/tests.rs` | lead | `cargo test --locked --lib engine::tests` | drop the boundary rounding → argmax/logit bound red |
| W4 | T2 measurement receipt | `docs/evidence/bench/<ts>-exact-scan.md` (new) | lead | `timeout 540 <bench>` + quiet gate | instrument control: run with a nonzero load, ratio must leave the quiet band |

W1-T1 and W1-T2 touch disjoint files. W2-T1 needs W1-T1 and W1-T2 merged. W2-T2 owns the generated
`proof/ApeironWord.lean` lane. W3-T1 needs W2-T2. W4-T1 needs W2-T1 and W3-T1. W4-T2 needs W4-T1.

### Task blocks

#### W1-T1 · exact chunked DeltaNet state step

- **Scope:** add `delta_head_chunk` computing one head's C-step composition in the chunked
  (WY/UT) form at a fixed intermediate precision `P` fractional bits (`P ≥ 32`, products rounded to
  `2^−P` in checked `i128`, one fixed bracketing), rounding to Q32.32 at the chunk boundary; keep
  `delta_head_step_ref` as the scalar oracle and test the deviation against the bound below.
- **Width note:** exact composition is infeasible: each Q32.32 product adds 32 fractional bits, so
  an exact `i128` composition overflows at `C = 2`. The contract is therefore the fixed bracketing at
  precision `P`, which is deterministic and bit-reproducible. It is graded by an error bound against
  the ℝ denotation (`recur_eq_scan`, `scan_split`) and against the per-step engine, not by
  associativity of the rounded words. `CHUNK_MAX` is chosen for throughput, since width no longer
  forces it; the stop rule below applies only to the bound.
- **Files:** `src/engine/mixer.rs`, `src/engine/mixer/tests.rs`
- **Depends on:** none
- **Grade:** tested (kernel), proved target (link at W3-T1)
- **Gate:** `cargo test --locked --lib engine::mixer::tests`
- **Negative control:** in a throwaway copy, round the composed coefficient before the injection
  update; `chunk_matches_oracle` and the bound test must fail; restore passes.
- **Evidence to record:** sha256 of `src/engine/mixer.rs`, `round_shift_32` identity, the derived
  `CHUNK_MAX`, `cargo test` exit status.

#### W1-T2 · online (m, den, num) attention chunk fusion

- **Scope:** replace the per-position weight vector with the running `(m, den, num)` state fused by
  exact `i128` addition, one `fixed_exp` per (score, tile) and one normalisation at readout; keep the
  existing single-position path as the oracle.
- **Files:** `src/engine/attn.rs`, `src/engine/tests.rs`
- **Depends on:** none
- **Grade:** tested; proved target for the fixed-point denotation
- **Gate:** `cargo test --locked --lib engine::tests`
- **Negative control:** fuse with the first tile's maximum only; the fused output must differ from
  the per-position path on a two-tile input and the test must fail.
- **Evidence to record:** sha256 of `src/engine/attn.rs`, tile count, before/after `attn.core`
  ms/token under the trace flag.

#### W2-T1 · prefill wiring and selectable flag

- **Scope:** route `step/cols.rs` prefill columns through the chunk kernels when
  `APEIRON_EXACT_SCAN=1`, default 0; propagate `StepError`/`AttnError` unchanged.
- **Files:** `src/engine/step/cols.rs`, `src/engine/step.rs`
- **Depends on:** W1-T1, W1-T2
- **Grade:** tested
- **Gate:** `cargo test --locked --lib engine::step`
- **Negative control:** route through the chunk kernel when the flag is 0; the default-path parity
  test must fail.
- **Evidence to record:** sha256 of both files, `APEIRON_EXACT_SCAN` read site, flag default.

#### W2-T2 · extraction roots for the chunk body

- **Scope:** add the `word_delta_head_chunk` wrapper and the extraction root; regenerate the committed
  extraction on the W1/W2 source.
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`; generated `proof/ApeironWord.lean` is the
  shared lane
- **Depends on:** W2-T1
- **Grade:** tested (extraction freshness)
- **Gate:** `proof/regen.sh --check`
- **Negative control:** in a throwaway copy, mutate one constant in the wrapper and run `regen.sh`;
  `--check` must report drift; restore passes.
- **Evidence to record:** Aeneas/Charon revisions, `ApeironWord.lean` sha256, exit status.

#### W3-T1 · Q32.32 refinement and audit

- **Scope:** model `round_shift_32` on the integer carrier, instantiate
  `chunked_discrepancy_uniform_bound` at Q32.32 with explicit rounding, and prove
  `word_delta_head_chunk_denote` against `Nous.Model.affApply`/`affFold`; register the `#audit`.
- **Files:** `proof/corpus/Numeric/ExactScanRefinement.lean` (new), `proof/corpus/WordRefinement.lean`,
  `proof/corpus/AxiomAudit/Remaining.lean`
- **Depends on:** W2-T2
- **Grade:** proved
- **Gate:** `cd proof/corpus && lake build Numeric.ExactScanRefinement AxiomAudit`
- **Negative control:** shift the wrapper's rounding index by one word; the headline must fail; restore
  passes.
- **Evidence to record:** source sha256, generated-body sha256, axiom cone of the headline, job count.

#### W4-T1 · oracle parity and logit bound

- **Scope:** assert greedy argmax equality against llama.cpp at every pp200+tg64 position and the
  per-logit bound (≤ 256 LSB, `anchor` until this task fixes it from W3-T1's ε), at both flag values.
- **Files:** `src/engine/tests.rs`
- **Depends on:** W2-T1, W3-T1
- **Grade:** tested
- **Gate:** `cargo test --locked --lib engine::tests`
- **Negative control:** drop the chunk-boundary rounding; the argmax/bound assertion must fail.
- **Evidence to record:** the fixed bound constant, argmax agreement count, logit maximum, exit status.

#### W4-T2 · measured prefill ratio

- **Scope:** run the pp200/tg64 comparison for `APEIRON_EXACT_SCAN=0|1` against llama.cpp and record
  the receipt; no claim beyond the receipt.
- **Files:** `docs/evidence/bench/<ts>-exact-scan.md` (new)
- **Depends on:** W4-T1
- **Grade:** measured
- **Gate:** `timeout 540 cargo run --release --example engine_profile -- pp200 tg64` plus
  `llama-bench` in the same quiet window
- **Negative control:** one round at busy ≥ 0.05 must fall outside the quiet band and be recorded, not
  averaged in.
- **Evidence to record:** host, governor, model sha256, prompt-fixture sha256, git revision, src
  sha256, busy-core readings, best-of llama.cpp threads, median and per-round paired ratios.

### Barrier checklist

- [ ] Every task gate exit 0, with its negative control run and restored.
- [ ] No task touched a file owned by another task.
- [ ] `cargo test`, `cargo clippy`, `cargo fmt` pass.
- [ ] `proof/regen.sh --check` reports fresh; `cd proof/corpus && lake build` passes.
- [ ] Source and tool digests recorded.
- [ ] Claims registered at their evidence grade; the sibling-repo citations are not called verified.
- [ ] Trust-boundary statement retained.

### Stop and rollback conditions

- Stop if the chunk kernel's width derivation cannot hold `CHUNK_MAX ≥ 2` at a real layer geometry;
  report the layer, `head_k` and the failing bound.
- Stop if the attention fixed-point refinement requires weakening `attn_domain`; the domain change is a
  contract change and goes to a verifier, not into this epic.
- Stop if the measured ratio is below 1.00: record the receipt, keep the flag at 0 and retain the
  per-step path; do not tune the benchmark protocol.
- After three materially different failed approaches to one obligation, report the smallest
  unresolved goal (named declaration, command and observed output) and stop.

```text
SCOPE:      docs/epics/exact-scan-contract.md; draft epic only, no source change
REGISTER:   planning (algebra/refinement/state/test/measurement); no acceptance from this file
CORPUS:     Nous.Model.FaceRecurrence.{affComp_assoc,recur_eq_scan,scan_split,recur_channel_split};
            Nous.Seq.GatedDelta.recurrent_eq_chunked,wy_form;
            Nous.Seq.ChunkedRounding.chunked_discrepancy_uniform_bound,witness_delta_net_100_lsb;
            Nous.Model.Attention.attention_channel_form; Nous.Model.TrichotomyAttention.attnDispatch;
            Numeric.EngineAttnRefinement.attnScoreInt,attn_domain; Numeric.AttnSoftmaxPerturb.not_attn_head_refines;
            Apeiron.Trichotomy.AnyRing.{ringMul_assoc,ringAffineFold_apply,ringAffineFold_append};
            Substrate...AttentionSpinor.{softmax_attention_is_online_spinor,oaDen_fuse,oaNum_fuse,aff_chunked_eq_full}
REFERENCE:  /home/socol/Workspace/apeiron/docs/evidence/bench-ledger.md#1790620216-ns-r003;
            /home/socol/Workspace/apeiron/src/engine/mixer.rs; /home/socol/Workspace/apeiron/src/engine/attn.rs
GATE:       none yet - W1-T1 carries the first gate: `cargo test --locked --lib engine::mixer::tests`, 540 s
GRADE:      planning; corpus declarations read are proved in their packages, sibling ones are pointers
```
