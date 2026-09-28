# One executor: models are schedules over primitives

## Intent

- **Owner:** lead (AI track); the lead writes the primitive layer, workers take bounded moves into existing files
- **Status:** S1–S3 and S5 landed (`hybrid/` removed with the user's go-ahead: 4,828 lines); S4 (head primitive) and S6 (Lean) open

### Problem

`src/engine/hybrid/` serves Qwen3.5-0.8B as a second stack beside `Engine::compose_logits`:

- a second attention wiring (`hybrid/gattn.rs`);
- three payload loaders (`GdnState::open`, `GattnState::open`, the decoder's store);
- a second config and tensor index (`hybrid/mod.rs`);
- duplicate rounding helpers;
- a single-layer DeltaNet state swapped per layer.

The two models differ only in which mixer each layer runs and in four attention flags. Every mathematically identical
operation must have one implementation.

### The primitive set (each one implementation, one Lean spec where proved)

| primitive | instances | today | Lean |
|---|---|---|---|
| `Proj` | every projection, grouped per input | `chart::ffn::projections_q8act*`, `matvec_fixed` | `MatvecChunk`, `DequantRefinement` |
| `Norm` (RMS × gain) | block norms, q/k norm, output norm, the DeltaNet gated norm | `rms_norm_fixed` + `rms_with_gain_slice` | `EngineNormRefinement`, `Sphere.rms_*` |
| `Rotate(dims, θ)` | RoPE on the first `dims` of a head (all dims: Qwen2; 64 of 256: Qwen3.5) | `RopeTable::rotate_one` | `RopeTurns`, `EngineRopeRefinement` |
| `Gate(σ \| SiLU)` | SwiGLU, attention output gate, DeltaNet z-gate, β | `silu_fixed`, `fixed_sigmoid` | `silu_gate_eq_channelWeight` |
| `Mixer::List` | softmax attention over the append-only K/V list | `KvState::{append, attn_head}` | `EngineAttnRefinement`, `LazySoftmax` |
| `Mixer::Affine` | state `S ← A_t S + B_t`, readout `Sᵀq`: DeltaNet `A = α(I − β k kᵀ)`, `B = β k vᵀ`; Mamba-2 `A = αI`; GLA `A = diag α` | `hybrid/gdn.rs` recurrence (to move) | `GatedDelta` (`recurrent_eq_chunked`, `gated_wy_form`); generalise to (A, B) |
| `Conv(k)` | short causal depthwise conv before the affine mixer | `hybrid/gdn.rs` (to move) | — |
| `Route(top-k)` + routed FFN | MoE (OLMoE, Qwen3-Next) | `moe.rs` | `MoeRouter`, `TopKRouter` |
| `Head(dense \| bitplane)` | every LM head, tied or not | `lm_head*`, `kern/bitplane.rs` (to fill) | `PrunedArgmax`, `BitplaneArgmax` |

A block is always `h += Mix(Norm(h)); h += FFN(Norm(h))`. A model is `EngineConfig` plus a per-layer schedule
`Vec<MixerKind>` plus flags: q/k norm, rotary dims, output gate, biases, tied head, FFN kind.

### Completion condition

One `Engine::compose_logits` runs Qwen2.5-0.5B, Bonsai-1.7B and Qwen3.5-0.8B in both kernel modes.

- Qwen2.5 ids equal `/tmp/opencode/l9/qwen-turns.txt`.
- Bonsai ids equal `/tmp/opencode/idle/bonsai-1.txt`.
- Qwen3.5 ids equal llama.cpp's on both prompts (32/32).
- `src/engine/hybrid/` is removed once parity holds, with the user's go-ahead.
- `examples/engine_decode.rs` serves all three models; `examples/hybrid_decode.rs` goes.

### Invariants

1. Qwen2 and Bonsai ids never move during the refactor; they are gated at every step.
2. The fast path always has an exact oracle, reached through the same primitive, not a copy.
3. A new file appears only for a new mathematical object: `Mixer::Affine` (with `Conv`) is the only one here.

## Plan (sequential; each step gated by all three models' ids)

| step | change | files |
|---|---|---|
| S1 | `EngineConfig` gains the schedule and flags; `DecodeVectors` reads q/k norms at `key_length`; `Engine::open` accepts `qwen35` | `loader.rs` |
| S2 | `Rotate` takes `dims`; `Gate` sigmoid on the attention output; the attention branch of `compose_logits` reads the flags | `rope.rs`, `step.rs` |
| S3 | `Mixer::Affine` + `Conv` in one module, holding every affine layer's state, with T7's pooled recurrence moved in; `compose_logits` dispatches on `MixerKind` | `mixer.rs` (new object), `step.rs`, `attn.rs` (layer → KV slot map) |
| S4 | the head is one primitive: the dense path and the bit-plane argmax behind `Head` | `lm_head.rs`, `kern/bitplane.rs` |
| S5 | parity on all three models, then remove `hybrid/` and `hybrid_decode.rs` | — |
| S6 | Lean: `GatedDelta` generalised to the affine step `(A, B)` with `A` block-diagonal in cells `RingCell σ`, one per channel pair: hyperbolic = two real decays (GLA, Gated DeltaNet's `α`), parabolic = the Jordan block `1 + ε` (pure accumulation: linear attention), elliptic = rotation × decay (RetNet, Mamba-3 complex state); DeltaNet's rank-1 correction `I − β k kᵀ` composes on the left | `Nous/Seq/GatedDelta.lean` |

Not relied on (from an external report):
- "`ringMul_assoc` makes chunked or WY evaluation bit-identical." Q32.32 rounds after every product, so
  reordering changes bits. Pooling across heads is bit-identical because heads are independent; chunking in time
  needs exact unrounded accumulation or its own tolerance contract.
- "`I − β k kᵀ` is an idempotent projection." That holds only at `β ∈ {0, 1}` with `‖k‖ = 1`; in general it is a
  contraction with eigenvalue `1 − β` along `k`.

## Results

| step | change | gate | grade |
|---|---|---|---|
| S1 | `EngineConfig.{mixers, rot_dims, attn_gate, affine}` read from metadata and tensor shapes (not the architecture name); `n_layer` excludes MTP blocks; pre-FFN norm alias `post_attention_norm`; `AffineVectors` loaded once | `schedule_is_read_from_metadata_not_architecture_name` | tested |
| S2 | partial `RoPE` (`rotate_head_prefixes`), `split_query_gate`, `apply_sigmoid_gate`, O projection sized `n_head·head_dim`, softmax slot map for `KvState` | Qwen2.5 64/64 and Bonsai 64/64 ids unchanged | tested |
| S3 | `src/engine/mixer.rs`: `AffineState`, `conv_silu`, `l2_normalize`, `delta_head_step`, `delta_heads` (pool over heads through `attn_heads`, no `unsafe`); `Engine::{state, step_state, step_logits_state}`, `DecodeState`; `group_q8act_at` replaces the q/k/v-only group; one `mul_q32` in `value.rs`; `fixed_rsqrt::isqrt` shared | `qwen35_on_the_one_executor_decodes_llama_cpp_ids` (Q8Act 8/8, Exact first token), `qwen35_long_prompt_on_the_one_executor` (32/32 after 200 tokens), Qwen2.5 and Bonsai 64/64 ids unchanged; `cargo test` 524 passed, clippy 0, fmt 0, `regen.sh --check` 0 | tested |

Speed, measured (i9-14900KF, 8 threads, 64 tokens, load 3.0–6.0), interleaved with llama.cpp `0324696b8`
on the same `Qwen3.5-0.8B-Q8_0.gguf`:

| engine | tok/s |
|---|---|
| one executor | 48.1, 48.6, 49.3 |
| llama.cpp | 50.9 ± 0.7, 51.4 ± 0.9 |

The ratio is about 0.95. The retired `hybrid/` stack ran at 18 tok/s on the same machine: its private weight store streamed
FFN and head at about 17 GB/s against the Engine's resident path.

## Performance plan: 2× llama.cpp on Qwen3.5-0.8B

Bytes per token (Q8_0; `gguf-py`, trunk plus tied head, MTP block excluded): 530.7 MB + 270.2 MB = 800.9 MB.

| path | tok/s | GB/s achieved |
|---|---|---|
| ours, Qwen3.5 | 48.7 | 39.0 |
| llama.cpp, Qwen3.5 | 51.1 | 40.9 |
| ours, Qwen2.5 (392 MB/token) | 140.7 | 55.1 |

Host DRAM: 82 GB/s. Both engines leave half the bandwidth unused on this model.

| lever | licence | expected effect (model grade until measured) | task |
|---|---|---|---|
| DeltaNet update at the algebraic minimum: the central decay deferred as one scalar per head; one read pass for both `Sᵀk` and `Sᵀq̃` through `o = Sᵀq̃ + d·(kᵀq̃)`; AVX2 | the decay `αI` is central in the matrix algebra; the rank-1 read identity | ≈ 9 → ≈ 2 ms/token | PERF-P1 |
| Bit-plane head | `BitplaneArgmax.bitplane_argmax_exact` | head bytes 270 → 143 MB | PERF-P2 |
| Projection bandwidth 39 → 55 GB/s on Q8_0 | — | measure the per-projection split first | lead |
| MTP self-speculation: verify k tokens per weight pass | `JacobiDecode.verify_accept_prefix` | amortises every byte over the accepted tokens | after Antigravity F3 (acceptance rate) |

2× llama.cpp is about 102 tok/s, which is 9.8 ms/token. The first three levers target that at full bandwidth; MTP goes beyond it.

### Landed (measured on i9-14900KF, 8 llama.cpp threads, 64 tokens, interleaved on the same Q8_0 file)

| change | licence | effect | control |
|---|---|---|---|
| PERF-P1: DeltaNet update with the central decay deferred as one scalar per head (renormalised at `c < 2⁻⁸`), one read pass through `o = Sᵀq̃ + d·(kᵀq̃)`, AVX2 (`mixer.rs` `delta_head_step_{ref,fast}`) | `αI` is central; the rank-1 read identity | `delta_heads` 207 → 73 µs/layer (≈ 3.7 → 1.3 ms/token); max drift vs the previous rounding 79 units of 2⁻³²; Qwen3.5 ids = llama.cpp on both prompts | worker: drop `d·(kᵀq̃)` → drift 3.6e10; lead: keep `c` undecayed → first token 279 ≠ 11751 |
| Conv + `SiLU` split across the pool (`conv_silu`) | channel independence | 2.75 → 0.47 ms/token | ids unchanged (parity tests) |
| Bit-plane head (`kern/bitplane.rs`, `Engine::bitplane_head`, `next_token_state`) | `BitplaneArgmax.bitplane_argmax_exact`, with rounded comparisons for `argmax_fixed`'s lowest-index ties | exact: lockstep with the dense head on real states, 160 synthetic activations with ties and negative scales; survivors median 18 of 248 320; **3.77 ms vs dense 3.26 ms**: stage 1 is compute-bound, so the path is opt-in (`APEIRON_BITPLANE=1`) until stage 1 is batched | tests `bitplane_argmax_equals_dense_argmax`, `duplicate_rows_resolve_to_the_lower_index`, `qwen35_bitplane_head_equals_dense_head_in_lockstep` |
| Pooled k-column `Q8_0` kernel (`kern/matmat.rs` `pooled_q8_0`, `chart::q8_0_row_kcols_avx2`) | exact integer sums | column-equal to `matvec_q8act` (existing tests); DRAM k-scaling not yet measured on a quiet host | — |

| path | tok/s |
|---|---|
| one executor, before | 48.1–49.3 (llama.cpp 50.9–51.4) |
| + P1 | 52.8–54.7 (llama.cpp 50.2–50.4) |
| + pooled conv, dense head | **58.7–60.4** (llama.cpp 47.1–48.5, load 6–9): **≈ 1.25×** |

Phase profile at 16.7 ms/token (throwaway instrumented copy, load 6.9):

| phase | ms/token |
|---|---|
| head (dense) | 3.92 |
| FFN | 4.32 |
| affine input projections | 3.36 |
| affine heads | 2.20 |
| softmax mixers | 1.47 |
| affine output projection | 0.70 |
| conv | 0.47 |
| norms | 0.18 |

### Multi-token pass and self-speculative decoding

| change | licence | gate | grade |
|---|---|---|---|
| `step/cols.rs` `compose_cols`: the executor's single layer loop over k token columns; `compose_final` is its k = 1 case; k-column projections through the pooled `matmat_q8act` (`Q8_0`: weights read once), quantized once per group | exact integer sums; causal attention per column; affine updates in token order | `qwen35_two_columns_equal_two_single_steps` (2 and 3 columns), `qwen25_two_columns_equal_two_single_steps`: every column and the state left behind equal the single steps bit for bit | tested |
| Pooled k-column `Q8_0` kernel (`matmat.rs` `pooled_q8_0`, `chart::q8_0_row_kcols_avx2`) | exact integer sums | DRAM-streamed 3584×1024 (load 1.0): k = 1 64.5 GB/s; k = 2 1.02× and 0.94× the time of k = 1; k = 3 1.31×; k = 4 1.85–1.99× | measured |
| `step/spec.rs` `generate_speculative`: the MTP block (`blk.24`: eh_proj, then the executor's `softmax_cols` and `ffn_cols`) drafts one token; a 2-column pass verifies it; a reject truncates the KV cache and restores each affine layer (state and deferred decay `c`) from a persistent in-place snapshot | `Nous/Model/SpeculativeRollback.lean`: `spec_round_eq_greedy`, `spec_decode_eq_greedy` (any drafter), `undo_write`, KV truncate lemmas (SPEC-L1; `AxiomAudit` exit 0, 9089 jobs; control: accept on mismatch → red) | `qwen35_speculative_equals_greedy`: ids = llama.cpp with the full draft head (8/8 accepted) and a 64-row draft head (2/13 accepted, so rollback runs on 11 rounds); `qwen35_speculative_long_prompt`: 32/32 after 200 tokens, 12/19 accepted | tested |

End to end, including prefill, 64 tokens, "The capital of France is" (every draft accepted); load 10.8–11.1:

| path | tok/s |
|---|---|
| greedy | 61.0–61.6 |
| speculative, 32 768 draft rows | 72.9–74.2 |
| llama.cpp `tg64` (decode only) | 55.0–55.9 |

Speculative is ≈ 1.3× llama.cpp, with our prefill counted and theirs not.

A round on the 200-token prompt (throwaway profile, load 6.7; 35 rounds, 28 accepted):

| part | ms/round |
|---|---|
| trunk, 2 columns | 23.2, including the snapshot at 2.1 before in-place reuse |
| head, 2 columns | 5.3, against 3.3 for one column |
| MTP block | 1.2 |
| draft head | 1.0 |

Open: the 2-column head, attention's per-column cost at longer contexts, and exact undo (`undo_write`) in place of the
snapshot copy.

### Batched prefill and an affinity bug it exposed

- `Engine::prefill_state` / `prefill_cols` run the prompt through `compose_cols` in blocks of `PREFILL_COLS = 8`, which
  reads the weights once per block. The speculative decoder also fills its MTP cache in blocks. Every Qwen3.5 parity test
  passes, and Qwen2.5 and Bonsai ids are unchanged.
- It exposed a latent bug in `par::row_chunk_budget`:
  - The function cached `std::thread::available_parallelism()` in a process-wide `OnceLock`. That value reports the
    *calling thread's* affinity.
  - `Pool::run` pins the dispatching thread to one core on its first dispatch.
  - So when a pooled dispatch preceded the first budget call (the batched prefill's `quant_act_pooled`), the cache
    read 1, and every chunked projection ran as a single chunk for the life of the process.
  - Measured: Qwen3.5 decode fell from 69 to 19 tok/s after a 5-token prompt, and stayed at 69 after a 1-token one.
  - Fix: `par::online_cpus` (`sysconf(_SC_NPROCESSORS_ONLN)`, affinity-independent). After the fix, 65.1–66.0 tok/s
    for both prompts. Regression test: `online_cpus_ignores_the_calling_threads_affinity`.

Quiet host after the fix (load 2.8 → 7.5; `llama-bench` tg64 45.3, pp200 259–262 in the same window):

| scenario | ours | llama.cpp |
|---|---|---|
| greedy decode, 5-token prompt | 59.8–60.7 tok/s | 45.3: ≈ 1.33× |
| speculative, 5-token prompt, end to end | 66.8–70.3 tok/s | — |
| prefill, 200 tokens | 104–109 tok/s (1.84–1.92 s) | 259–262: 2.5× slower |
| greedy decode after the 200-token prompt | 56.3–57.7 tok/s | — |
| speculative, 200-token prompt, end to end | 20.1–20.8 tok/s (26/37 accepted) | — |
| greedy, 200-token prompt, end to end | 20.9–21.7 tok/s | — |

Next: prefill needs a real int8 GEMM microkernel for `k ≥ 16` columns (llama.cpp's prompt path is GEMM-bound; ours
is the k-column GEMV at k = 8). Speculation at long context loses its margin to per-column attention and the 2-column
head.


## Baseline integrity and exact-kernel continuation

### Intent

Restore source-connected extraction and measurement integrity before evaluating further exact CPU
optimizations. The user authorizes local checkpointing, isolated worker tasks and integration. The
performance objective is practical throughput on limited hardware; this executor reports tokens/s,
not frames/s. Preserve mathematical denotations, rounding order and token parity. Approximation is
not an exact optimization. Protected policy sources and `src/lib.rs` retain their existing bytes.

### Specification

- Convolution checks each i128 addition and the rounding bias before updating its history.
- Half conversion uses an exact cast of a value bounded by the 10-bit mask; no new model axiom.
- External benchmark results require successful execution and complete, finite, positive pp/tg rows.
- Cargo extraction compiles the existing `#[path]` bodies with the locked libc dependency; retain
  all 68 roots, the model-backed stub cleanup and the rejecting axiom audit.
- Gates establish only their native grade. A checkpoint preserves existing work and certifies none
  of it. Missing Lean links, source-size obligations and unmeasured performance remain open.

### Plan

Base before checkpoint: `74c037b1cfc79b0d6be6d3907398fe926d78d12d`. Lead owns integration and every
shared build/measurement lane. Each source task uses its own `worktree-agent-<id>`; workers edit,
the lead runs gates, commits, integrates and removes the worktrees. No worker benchmarks. Each
command and descendants have a 540-second cap; Lean also uses the lane lock, 12 GiB memory limit,
no swap and one thread. The supplied-checkout documentation pass is lead-owned.

| task | owned files | owner | focused gate | rejecting control |
|---|---|---|---|---|
| APEIRON-INFERENCE-W0-T1 | `src/engine/kern/{gemm,pow2}.rs`, `src/engine/step/cols.rs` | lead | full Clippy command from AGENTS | existing seven-diagnostic baseline |
| APEIRON-INFERENCE-W0-T2 | `src/engine/mixer.rs`, `src/engine/mixer/tests.rs` | Luna attention | `cargo test --locked --lib engine::mixer::tests` | unchecked accumulation and rounding return the two failing regressions |
| APEIRON-INFERENCE-W0-T3 | `examples/engine_profile.rs` | Luna heads | `cargo test --locked --example engine_profile` | remove row completeness validation; missing-row fixture fails |
| APEIRON-INFERENCE-W0-T4 | `proof/extract/Cargo.toml`, `proof/extract/Cargo.lock`, `proof/regen.sh` | Luna pow2 | bounded regeneration after conversion task, generated file lead-owned | regenerate a real-source seeded defect and reject freshness |
| APEIRON-INFERENCE-W0-T5 | `src/engine/kern/pow2.rs`, after T1 | lead | `cargo test --locked --lib f16_split_is_f16_to_fixed` | fraction mask `0x3ff` to `0x3fe` fails at `0x0001` |

Move mixer tests without rewriting them so both owned files fit 1,500 lines. The convolution repair
and all three scratch mutation receipts are in `/home/socol/oc-tmp/astra-luna-baseline/receipt.md`;
rerun the focused gates on the integrated source. Regeneration owns `proof/ApeironWord.lean` in a
subsequent serial task. Run the full Rust set and both Lean package builds at the implementation
barrier; any incomplete gate keeps that barrier open. A separate quiet-host baseline uses five
interleaved repetitions against llama.cpp at 8/16/24/32 threads and retains quiet flags. Diagnostic
measurements do not close an implementation barrier. Select the next exact optimization from that
receipt; the d-prime split hoist is a candidate, not an established speedup.

```text
SCOPE:      docs/epics/one-executor.md planning extension; preserve current tree in checkpoint
REGISTER:   planning; no implementation or programme acceptance from checkpoint
CORPUS:     no new theorem claim; existing source and scratch gate receipts read
REFERENCE:  /home/socol/oc-tmp/astra-handoff.md
GATE:       inspect checkpoint paths and unchanged source identity, 60 seconds
GRADE:      planning; implementation, proof and measurement gates remain separate
```


### Exact affine traversal and half-word refinement

Fresh receipts: `docs/evidence/bench/1790607104-astra-batched-baseline.{json,md}` and
`1790607289-astra-core-budget.{json,md}`. Every round passed the quiet gate. The former measures
224.99 prefill and 62.71 greedy tokens/s against llama.cpp's 523.54 and 49.90; affine heads consume
30.5% of traced prefill wall time. The rotated core comparison measures 67.56 greedy tokens/s with
eight performance cores and 47.84 with four. Those are measurements on the i9-14900KF, not a
claim about different low-end hardware. Speculation at 32,768 draft rows measures 37.95 tokens/s
and 24% acceptance on this prompt; it is not the selected fast path.

The affine candidate traverses value-column vectors before independent key rows, loading each
unchanged d-prime vector outside the row loop. Retain the existing multiplication formula,
rounding, bounds and post-update check. A process-level `APEIRON_WRITE_COLUMN_MAJOR=0|1` switch
selects traversal for measurement; default 0 until evidence selects a winner. Extra storage is
constant; no new heap allocation or approximation. The release assembly saved at
`/home/socol/oc-tmp/astra-integration/write-pass-assembly.txt` contains the repeated split inside
the current row loop. This observation is not a speed claim.

The proof task links the actual `word_f16_split` to existing `f16Value` for finite patterns and
proves rejection exactly for exponent 31, with reachable finite/nonfinite witnesses. Do not change
`act_scale`, `finish`, `ChartDot`, a corpus pin or an axiom policy in this task. The audit registry
is first split mechanically because its current 2,373 lines exceed the owned-file bound. Preserve
every existing import, declaration and audit command and keep the same default build root.

| task | owned files | owner | one focused gate | control |
|---|---|---|---|---|
| APEIRON-INFERENCE-W1-T1 | `src/engine/mixer.rs`, `src/engine/mixer/tests.rs` | Luna heads | `cargo test --locked --lib engine::mixer::tests` | corrupt only the candidate coefficient; dual-traversal/scalar equality must fail |
| APEIRON-INFERENCE-W1-T2 | `proof/corpus/AxiomAudit.lean`, `proof/corpus/AxiomAudit/{Core,Remaining}.lean` | Luna attention | `cd proof/corpus && lake build AxiomAudit` | forbidden-axiom sentinel must be rejected in a throwaway copy |
| APEIRON-INFERENCE-W1-T3 | `proof/corpus/Numeric/Pow2Refinement.lean`, then root `proof/corpus/AxiomAudit.lean` after its factoring task | Luna pow2 for theorem, lead for final registration | `cd proof/corpus && lake build Numeric.Pow2Refinement AxiomAudit` | actual fraction-mask mutation plus regeneration must break the named exactness theorem; restore passes |

Tasks use separate worktrees from the committed plan; lead owns commits, merges and all shared
build products. The proof writer may check its proposed file against freshly built main dependencies
under the pinned package's `lake env lean`, lane lock and remaining task budget. That development
check does not replace the final named build/audit. A fresh independent verifier on a different
model reviews the new theorem's domain, attribution, witnesses and audit before acceptance. The
proof has a 540-second work slice and stops after three materially different failed approaches.
All workers stop for the measured window. The lead performs the five-round rotated A/B against
the same binary's original traversal; only a supported result changes the default. A regression
keeps the original traversal and is recorded, not described as a speedup.

### Reproducible benchmark prompt

Lead-owned task `APEIRON-INFERENCE-W1-T4` owns `examples/engine_profile.rs`,
`tests/fixtures/qwen35-prompt2.ids` and this planning entry in its isolated task worktree. Copy the
existing prompt bytes, sha256 `894a166ae31393f33425ed7d1fd0b3eed0801c26a3c7095d90825d66fb240443`,
and use the repository fixture as the default. The explicit `--ids` option remains available.
Gate: byte/hash equality and `cargo test --locked --example engine_profile`, 540 seconds.
Grade: tested input preservation; no arithmetic or performance claim. A semantic mutation control
is inapplicable to a byte-identical relocation; exact hash equality checks the complete input.
