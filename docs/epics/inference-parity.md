# Inference parity: an existing model on the proven stack

Merged epic record: `intent/inference-parity.md`, `spec/inference-parity.md`, `plan/inference-parity.md` (one file per epic; see `docs/README.md`).

## Intent — Inference parity: an existing model on the proven stack

- **Owner:** Apeiron lead
- **Status:** draft for decision (the runtime-seam question below is open)
- **Source material:** the old `fold` engine (working tree at `/home/socol/Workspace/fold`, commits
  `7bdd7a58`, `b20352db`, `24ed7451`; inventory `/tmp/opencode/fold-engine-inventory.md`); llama.cpp
  at `/home/socol/Workspace/llama.cpp` (builds `build-cpu`, `build-cuda`, `build-vulkan`); recorded
  comparison receipts in `/home/socol/Workspace/fold/results/`.

### Problem

Apeiron's canonical object and primitives are proved; the executing face for a standard open-weight
model is not. The programme must run an existing model whose outputs match the float reference
semantics, with its discrete control plane refined to the canonical corpus, and compare favourably
against llama.cpp on a named machine. The architecture track (`Nous`) waits behind this milestone.

The `fold` engine already did the numeric work and measured it: Vulkan/SPIR-V runtime with
hand-emitted shaders, GGUF/SafeTensors loaders, quantizers, KV pool, benchmark harnesses. Recorded on
an RTX 5080 against llama.cpp (`results/vs-llamacpp-27b.md`, `results/h2h-8b-at-the-served-carrier.md`,
`results/vs-llamacpp.md`):

| Model | fold Vulkan | llama.cpp Vulkan | llama.cpp CUDA |
|---|---|---|---|
| Qwen3.8-27B IQ2_M | 47.6 tok/s, 620 GB/s effective | 50.42 ± 0.72 tok/s, 520 GB/s | 67.97 ± 0.18 tok/s, 701 GB/s |
| Qwen3-8B (split-K) | 150.6 tok/s, 524 GB/s | 153.3 tok/s, 476 GB/s | 232.6 tok/s, 722 GB/s |
| Qwen2.5-0.5B | 920.2 tok/s, 284 GB/s | 796.08 ± 13.18 tok/s, 276 GB/s | 944.76 ± 6.13 tok/s, 327 GB/s |
| Bonsai-1.7B ternary | 1281.0 tok/s | 641.4 tok/s | 26.8 tok/s |

At 27B the raw tokens/s is 5.6% behind while effective bandwidth is +19.2%: the deficit is a format
(bytes/token), not a kernel. The receipts are the baseline to reproduce from the apeiron tree.

### Why the alternatives are rejected

- **Wrap llama.cpp or PyTorch through FFI.** No refinement story for any layer; the engine would be a
  dependency, not a subject. Rejected.
- **Rebuild the engine from scratch.** The measured engine exists, is present in full (the 25→12 merge
  losses were restored by `24ed7451`; zero files missing), and already beats llama.cpp Vulkan on
  effective bandwidth. Rebuilding would discard the evidence. Rejected.
- **Treat the missing float model as a stopping condition.** The engine can be ported and run while
  the toolchain work proceeds, but no release accepts an unverified executing component. The gap is
  sequenced, not skipped; see the decision below.
- **Fixed-point rewrite of every kernel now.** Integer-only arithmetic is provable and fast, but
  changes the model semantics: it sacrifices compatibility with standard checkpoints and the recorded
  llama.cpp comparison would no longer be a like-for-like measurement. Rejected as the first step.

### Decision: no unverified source component (user direction)

The bounded runtime seam is rejected, and the Aeneas float fork is **not required for this path**:
the executing engine is integer/fixed-point, which the existing Aeneas integer model already covers.
The float fork is deferred with a trigger — it becomes necessary only if a float artifact must enter
the verified executing path (for example verifying an unquantized float runner or a float reference
simulator as an artifact rather than as a test oracle).

**Executing path.** Integer arithmetic with explicit scaling; every operation in the existing Aeneas
integer model; the transcendental functions (`exp` for softmax/SiLU, `rsqrt` for RMSNorm, `sin`/`cos`
with argument reduction for RoPE) are fixed-point implementations carrying proved error bounds against
the real-valued specification. Per the integer spike (`/tmp/opencode/toolchain-spike-integer-transcendentals.md`):
accuracy is not the binding constraint (the recorded served path already passes its differential gate
with f32 per-op error; a degree-6 fixed-point Taylor `exp` reaches ~1e-8 relative), the proof is; the
pinned Mathlib v4.31.0 ships `Real.exp_bound`, `Complex.exp_bound`, `Real.sin_bound`/`cos_bound`, so
the polynomial remainders are off-the-shelf. `exp` and `rsqrt` are weeks each of Lean by the spike's
anchor estimate (calibrated by the first slice, not accepted as a schedule); the long pole is RoPE's
argument-reduction proof — short Q32 contexts first, full 32768+ contexts later.

**Honest artifact shape.** Per-function bounds do not compose into an unconditional token theorem; the
artifact is a proved ε per function plus a runtime-checked margin on the decision (the existing
`Substrate.StrictWinner` shape). The comparison with llama.cpp is quality and speed within that
bound, not bit-parity.

Exactness target: the integer route targets the proven error bound and quality equivalence; bit-parity
with PyTorch is required by neither route (llama.cpp's quantized paths differ from PyTorch too).

**Platform boundary (irreducible).** Even with every source component modeled, `rustc`/LLVM codegen,
the CPU ISA, and — for the GPU lane — the SPIR-V toolchain, driver and hardware remain outside the
proof, as any verified software stack assumes its compiler and machine. Parity claims name this
boundary; no source-level component is left unverified.

**Platform boundary (irreducible).** Even with every source component modeled, `rustc`/LLVM codegen,
the CPU ISA, and — for the GPU lane — the SPIR-V toolchain, driver and hardware remain outside the
proof, as any verified software stack assumes its compiler and machine. Parity claims name this
boundary; no source-level component is left unverified.

### Invariants

- The float execution path is never presented as proved; the discrete path is never presented as tested
  only once its refinement lands.
- Every claim carries its grade and its command; the comparison names machine, metric, interval and
  both engines' pinned revisions.
- The ported engine sits in the apeiron tree under its gates; the `fold` tree is a migration source,
  not a second live home; its 13 worktrees and 7 dirty files are not migrated.
- Removing or superseding anything in `fold` follows the removal rule: replacement named, consumers
  inventoried, reachability gated.
- One in-progress owner per file; one build lane per shared cache.

### Measurable success condition

1. **Runs.** A chosen open-weight model (smallest first: Qwen2.5-0.5B, then an 8B recipe) loads,
   tokenizes and decodes from the apeiron tree.
2. **Matches.** Differential gate: zero device-vs-host argmax mismatches over the recorded number of
   warm steps; greedy output equals the recorded fold output; logits within the stated tolerance of the
   CPU f64 reference.
3. **Compares.** Fresh receipts on the named machine for decode tokens/s and effective bandwidth
   against the pinned llama.cpp Vulkan and CUDA builds at the same model/recipe/context; favourable
   means at least the recorded relative position (e.g. 620 vs 520 GB/s at 27B, 524 vs 476 at 8B, 1281
   vs 641 tok/s on Bonsai).
4. **Refines.** The discrete-plane modules extract, refine and audit under the closed cone
   `{propext, Classical.choice, Quot.sound}`, with controls that turn red under seeded defects.
5. **Bounds.** The trust-boundary statement names the float runtime, the Vulkan FFI, the model files
   and the tokenizer data as outside the proof.

### Non-goals

- No proof of float kernels and no claim of bit-exact GPU semantics.
- No CUDA backend requirement; Vulkan/SPIR-V is the measured lane.
- No PyTorch runtime dependency (offline reference values only).
- No `Nous` architecture work while parity is the priority; no vocabulary revival from the old tree.
- No publication or external post; no commit without explicit user authorization.

### Exit to the architecture track

When conditions 1–5 hold, the float runtime is a measured, bounded instrument and the discrete plane is
proved; the `Nous` architecture waves resume from `docs/epics/ai-track-reanchor.md` (W4-T2, W4-T3, then
the falsification point), with the fixed-point question revisited there.

## Specification — Inference parity: integer executing path

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/inference-parity.md`](inference-parity.md)
- **Source revision:** `74c037b…`; dirty (`Nous` and `Growable` lanes; see `docs/archive/handoff-openai.md`)
- **Status:** approved (lead); no commit without explicit user authorization

```text
SCOPE:      integer/fixed-point numeric kernels for the parity engine: Rust sources, extraction
            roots, Lean refinement and error-bound proofs; no float executing path
REGISTER:   refinement/measurement; exclusions: float semantics, GPU execution semantics, sampling
CORPUS:     Mathlib v4.31.0 Real.exp_bound, Complex.exp_bound, Real.sin_bound, Real.cos_bound;
            existing Aeneas integer model at target/aeneas-model/lean
REFERENCE:  /tmp/opencode/toolchain-spike-integer-transcendentals.md,
            /tmp/opencode/fold-engine-inventory.md,
            /home/socol/Workspace/fold/results/{vs-llamacpp.md,vs-llamacpp-27b.md,h2h-8b-at-the-served-carrier.md}
GATE:       proof/regen.sh --check; lake build Numeric.ExpRefinement AxiomAudit (lane lock, timeout 540);
            budgets per §10
GRADE:      proved for refinements; tested for the differential harness; measured for the comparison
```

### Scope and admissible domain

Integer/fixed-point numeric kernels for the parity engine: exact integer operations (dot products,
scaling, conversions) and transcendental approximations (`exp`, `rsqrt`, `sin`/`cos` with argument
reduction) carrying proved error bounds against the real-valued specification. Rust functions are
extracted by the pinned Charon/Aeneas (`target/aeneas/`, present and executable) and refined in Lean
under `proof/corpus/Numeric/`.

| Item | Value |
|---|---|
| Operation | fixed-point `exp`, `rsqrt`, `sin`/`cos` + reduction; integer dot/scale |
| Domain | `i64`/`i32` with explicit scale exponent; Q32 first (positions ≤ 2^12), wider context later |
| Totality | total Rust functions returning `Result`; out-of-range or overflow refuses, never wraps |
| Preconditions | documented input window per function; scale invariant on operands |
| Postcondition | `\|toReal (f x) − F (toReal x)\| ≤ ε_F · \|F (toReal x)\|` with proved `ε_F`, or an exact integer law |

### Representation relation (Rust to Lean)

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `fixed_exp` | `src/numeric/fixed_exp.rs` |
| Extraction wrapper | real body, no restatement | `proof/extract/lib.rs` (`#[path]`) |
| Extracted body | `fixed_exp` | `proof/ApeironWord.lean` (generated) |
| Real specification | `expRef` | `proof/corpus/Numeric/ExpRefinement.lean` |
| Refinement theorem | `fixed_exp_refines` | same |

Adding numeric roots **regenerates `proof/ApeironWord.lean`**, moving the identity from
`8e76b917…`; that regeneration is a planned barrier with `proof/regen.sh --check`, a full
`AxiomAudit` re-run, and every digest receipt re-bound. The wrapper fixes types only.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| `fixed_exp_refines` | `\|toReal v − exp (toReal x)\| ≤ 2^-20·exp (toReal x) + 4/q32` on `window` (the Q32.32 floor; pure relative `2^-20` does not follow) | proved | refinement + `Real.exp_bound` remainder |
| `fixed_rsqrt_refines` | `\|toReal v − 1/√(toReal x)\| ≤ 2^-20·(1/√(toReal x)) + 4/q32` on `rsqrt_window` (same additive floor) | proved | `Nat.sqrt` + one division, bound |
| `fixed_sin_cos_refines` | `\|toReal sc.f − f (toReal x)\| ≤ 2^-20·\|f (toReal x)\| + 2^-20` on `trig_window`, both outputs; the complement is refused | proved | `Real.sin_bound`/`cos_bound` + reduction lemma |
| `fixed_dot_exact` | integer dot exact in `i64` on the documented window (no overflow) | proved | extraction + arithmetic |
| `fixed_scale_rounds` | one-rounding bound for fixed-point scaling | proved | extraction + bound |
| `numeric_margin_refuses` | decision refuses when the margin is below the proved `ε` | proved | runtime margin check (`StrictWinner` shape) |
| `gpugate_agreement` | device vs host reference argmax agreement over N warm steps | tested | `gpugate` harness (engine wave) |

Per-function `ε` values are set so that the recorded served-path accuracy is retained; the spike's
`anchor` budget is ≤ ~1e-6 relative per op, with `2^−20` as the proved target. `exp` dominates by call
count; `rsqrt` is cheapest; RoPE reduction is the proof-hard one. Bounds do not compose into an
unconditional token theorem — the decision carries a runtime margin.

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| argument outside the reduction window | refusal (`Result` error) | per-function `hbound` |
| `i64` overflow in accumulation | refusal; no wrap | checked add/mul |
| margin below the proved `ε` | decision refusal | margin predicate |

### Evidence plan

1. **Positive tests:** `#guard` differential tables per function against exact rational/real
   evaluations; the engine-wave `gpugate` harness.
2. **Proof:** `proof/regen.sh --check`; `lake build Numeric.ExpRefinement AxiomAudit` under the lane
   lock, `timeout 540`.
3. **Axiom cone:** every headline registered in `proof/corpus/AxiomAudit.lean`; closed cone
   `{propext, Classical.choice, Quot.sound}`.
4. **Negative control:** perturb one polynomial coefficient in a throwaway copy; the bound goal must
   be unsolved; restore green.
5. **Measurement:** engine wave only — llama.cpp comparison on the RTX 5080, decode tok/s and
   effective bandwidth, interval and prompt set as in the `fold` receipts.

### Trust boundary

No float executing path. The CPU `f64` reference is a test oracle, not a verified artifact. `rustc`/
LLVM codegen, the CPU ISA, and (if a GPU lane is kept) the SPIR-V toolchain, driver and hardware are
the platform boundary. No token or capability theorem; the artifact is the proved `ε` plus the runtime
margin.

### Open questions

- RoPE position range: Q32 (short contexts) first versus 32768+ (Payne–Hanek-scale proof) later.
- Extraction identity move: schedule the `proof/ApeironWord.lean` regeneration barrier once the first
  refinement lands.
- Engine lane: CPU-only receipts first, or integer shader generation for the GPU comparison.

## Plan — Inference parity: integer executing path

- **Owner:** Apeiron lead
- **Intent:** [`docs/epics/inference-parity.md`](inference-parity.md)
- **Spec:** [`docs/epics/inference-parity.md`](inference-parity.md)
- **Source revision:** `74c037b…`; dirty (see `docs/archive/handoff-openai.md`)
- **Status:** approved (lead); no commit or push without explicit user authorization

### Goal and completion condition

The wave-level barrier: every task gate exit 0 with its control restored, the numeric refinements
extract from real Rust, the cone closes over every headline, and the engine wave reproduces the
recorded llama.cpp comparison on the RTX 5080. Barrier commands:

```sh
proof/regen.sh --check
cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Numeric.ExpRefinement Numeric.RsqrtRefinement Numeric.RopeRefinement AxiomAudit'
cargo test --workspace --all-features --locked
```

### Non-goals

- No float executing path, no Aeneas float fork (deferred with trigger; see the intent).
- No proof of GPU execution semantics; the platform boundary is named, not removed.
- No bit-parity claim with PyTorch or llama.cpp; the artifact is proved `ε` plus a runtime margin.
- No `Nous` architecture work until the parity barrier; W4-T2/W4-T3 stay paused.
- No task before its files are named, one owner per file, ≤3 files per task.

### Wave map

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | fixed-point `exp` Rust implementation (precursor) | `src/numeric/mod.rs`, `src/numeric/fixed_exp.rs`, `src/lib.rs` | gemini-flash | `cargo test --workspace --all-features --locked`; clippy; fmt | perturb one Taylor coefficient -> unit test red |
| W1 | fixed-point `rsqrt` Rust implementation (precursor) | `src/numeric/mod.rs`, `src/numeric/fixed_rsqrt.rs` | deepseek-flash | cargo trio; live extraction audit `NO_AXIOMS` | perturb a rounding constant -> unit test red |
| W1 | fixed-point `exp` refinement | `src/numeric/fixed_exp.rs`, `proof/extract/lib.rs`, `proof/corpus/Numeric/ExpRefinement.lean` | gemini-flash | `regen.sh --check`; `lake build Numeric.ExpRefinement AxiomAudit` | perturb one coefficient -> bound goal unsolved |
| W1 | fixed-point `rsqrt` refinement | `src/numeric/fixed_rsqrt.rs`, `proof/extract/lib.rs`, `proof/corpus/Numeric/RsqrtRefinement.lean` | deepseek-flash | `regen.sh --check`; `lake build Numeric.RsqrtRefinement AxiomAudit` | drop the division step -> bound red |
| W2 | RoPE reduction + `sin`/`cos` (Q32) | `src/numeric/fixed_trig.rs`, `proof/extract/lib.rs`, `proof/corpus/Numeric/RopeRefinement.lean` | gemini-flash | `regen.sh --check`; `lake build Numeric.RopeRefinement AxiomAudit` | wrong reduction quadrant -> bound red |
| W2 | extraction identity barrier (lead) | `proof/ApeironWord.lean` (regenerated), `AxiomAudit.lean` | lead | `regen.sh --check`; bare `lake build`; `lake build AxiomAudit` | n/a — barrier |
| W3 | GGUF parser + quant unpack extraction | `src/format/gguf.rs`, `src/format/quant.rs`, `proof/extract/lib.rs` | gemini-flash | `regen.sh --check`; `lake build Numeric.FormatRefinement` | corrupt length field -> refusal proof red |
| W3 | BPE tokenizer extraction | `src/format/bpe.rs`, new Lean module | deepseek-flash | `regen.sh --check`; module gate | byte-boundary mutation -> red |
| W4 | engine port + integer kernels + harness | ported `fold` subset under `src/engine/`, harness bin | lead + workers | `lake build`; harness differential gate | layer-order mutation -> `gpugate` mismatch |
| W4 | comparison receipts | none (measurement) | lead | named machine/metric/interval protocol | n/a — measurement |

Owners are the intended lanes; the lead assigns at dispatch. Shared caches (`proof/.lake`,
`proof/corpus/.lake`, `target/`) are one lane each.

### Task blocks

#### APEIRON-INT-TRANSCENDENTAL-W1-T0 · fixed-point `exp` Rust implementation (precursor)

- **Scope:** the executable Rust body for W1-T1's extraction: `i64` Q32.32 fixed-point `exp` via
  range reduction to `[-ln2/2, ln2/2]` and Horner Taylor, with `Result` refusal outside the documented
  window; unit tests bound the achieved relative error and a mutation fails them. No proof-tree file
  is touched; extraction and the refinement theorem stay in W1-T1.
- **Files:** `src/numeric/mod.rs`, `src/numeric/fixed_exp.rs`, `src/lib.rs`
- **Depends on:** none
- **Grade:** tested
- **Gate:** `cargo test --workspace --all-features --locked`; `cargo clippy --workspace --all-features --all-targets --locked -- -D warnings`; `cargo fmt --all -- --check`
- **Negative control:** in a throwaway copy, perturb one Taylor coefficient (or the `ln2` constant); a
  unit test must fail with an attributable assertion; restore green.
- **Evidence to record:** Rust sha256, gate exits, control receipts, the documented window and the
  achieved error bound, elapsed.
- **Budget:** 45 min task; stop at expiry and report.

#### APEIRON-INT-TRANSCENDENTAL-W1-T2A · fixed-point `rsqrt` Rust implementation (precursor)

- **Scope:** extraction-clean `i64` Q32.32 `rsqrt` over a documented positive window, integer-only
  (scaled integer square root plus division, or Newton with shifts and multiplies if division audits
  unclean), with `Result` refusal, unit tests against an `f64` oracle, and a live re-extraction audit
  that must print no `axiom`. Grade `tested`.
- **Files:** `src/numeric/mod.rs`, `src/numeric/fixed_rsqrt.rs`
- **Depends on:** none
- **Gate:** cargo trio; extraction + audit `NO_AXIOMS`; control in a throwaway copy
- **Budget:** 45 min.

#### APEIRON-INT-TRANSCENDENTAL-W1-T1 · fixed-point `exp` refinement

- **Scope:** fixed-point `exp` via range reduction and Horner Taylor (spike: degree 6 reaches
  ~1e-8 relative), extracted through Charon/Aeneas, with `fixed_exp_refines` bounding the relative
  error against `Real.exp` on a documented window; refusal outside it. No float path.
- **Files:** `src/numeric/fixed_exp.rs`, `proof/extract/lib.rs`, `proof/regen.sh`, `proof/corpus/Numeric/ExpRefinement.lean`
- **Depends on:** W1-T0, W1-T0B
- **Integration note:** `proof/regen.sh` carries the source-digest entry and the `--start-from
  crate::word_fixed_exp` root. It was an undeclared fourth file in the W1-T1 slice; the lead
  reviewed and accepted it (additive registration only, audit logic untouched).
- **Grade:** proved
- **Gate:** `proof/regen.sh --check`; then `cd proof/corpus && flock -w 600 /tmp/opencode/apeiron-lane.lock timeout 540 bash -c 'LEAN_NUM_THREADS=1 lake build Numeric.ExpRefinement AxiomAudit'`
- **Negative control:** in a throwaway copy, perturb one Taylor coefficient; the bound goal must be
  unsolved with an attributable diagnostic; restore green. Logs named in the receipt.
- **Evidence to record:** Rust and Lean sha256, regenerated `ApeironWord.lean` digest, gate exits,
  control receipts, the proved `ε_exp` and window, elapsed.
- **Budget:** 45 min task; the regen step declares its own §10 budget. Stop at expiry and report.

#### APEIRON-INT-TRANSCENDENTAL-W1-T2 · fixed-point `rsqrt` refinement

- **Scope:** fixed-point `rsqrt` from `Nat.sqrt` plus one division (spike: cheapest to tighten),
  extracted, with a proved relative bound and refusal outside the window.
- **Files:** `proof/corpus/Numeric/RsqrtRefinement.lean`
- **Depends on:** W1-T1B, W1-T2A (extraction wired by the lead batch)
- **Grade:** proved
- **Gate:** `lake build Numeric.RsqrtRefinement AxiomAudit`
- **Negative control:** drop the division step in a copy -> bound goal red; restore green.
- **Evidence to record:** digests, gate exits, control receipts, `ε_rsqrt`, elapsed.

#### APEIRON-INT-TRANSCENDENTAL-W2-T1 · RoPE reduction and `sin`/`cos` (Q32)

- **Scope:** argument reduction to a Q32 quadrant plus Taylor `sin`/`cos`, extracted, with the
  reduction exactness lemma and the trig bounds from `Real.sin_bound`/`Real.cos_bound`; position range
  documented (Q32 first).
- **Files:** `proof/corpus/Numeric/RopeRefinement.lean`
- **Depends on:** W2-T1A (extraction wired by the lead batch)
- **Grade:** proved
- **Gate:** `lake build Numeric.RopeRefinement AxiomAudit`
- **Negative control:** wrong reduction quadrant -> bound red; restore green.
- **Evidence to record:** digests, gate exits, control receipts, the reduction lemma and bounds,
  position window, elapsed.

### Barrier checklist

- [ ] Every task gate exit 0, control run and restored.
- [ ] `proof/regen.sh --check` clean at the wave barrier; every digest receipt re-bound after the
      identity move.
- [ ] `AxiomAudit` green with every numeric headline registered inside the closed cone.
- [ ] `cargo test`/`clippy`/`fmt` pass on the Rust additions.
- [ ] The engine wave's `gpugate` differential gate and the llama.cpp comparison receipts recorded
      with machine, metric and interval.
- [ ] Trust boundary restated: platform layers only; no token or capability theorem.

### Stop and rollback conditions

- Any gate at 540 s or task at 45 min: stop, record elapsed and the incomplete stage; a timeout is a
  blocked task, not closure.
- Three materially different failed approaches to one bound: stop and report the smallest unresolved
  goal with the exact goal state.
- A bound needs a new axiom, a premise change, or a grade change: escalate to the lead before edits.
- No commit, push, branch or worktree without explicit user authorization; controls never edit the
  live tree.

### Receipts
- `APEIRON-INT-TRANSCENDENTAL-W1-T0` (deepseek-flash, ~4 min of 45): `src/numeric/fixed_exp.rs`
  `d225d1e8e65ffb68d1b39687bc4ff769c9c5edb0d52d699490e117e384c6ffa3` (196 lines),
  `src/numeric/mod.rs` `c89edded393a002dd632805d2b573652262103108ccf559e805edb23449726e8`
  (8 lines), `src/lib.rs` `bb20d6345c74f577c2b10510165a1382ebf04c05be16a5fd2407c71450f2055d`
  (one added line `pub mod numeric;`; the other `src/lib.rs` diffs are pre-existing Kani-retirement
  dirt and `src/proofs` no longer exists). Grade `tested`: `cargo test` 103 passed / 0 failed, clippy
  and fmt exit 0; control `/tmp/opencode/int-exp-control/` baseline 103/0, red 102/1 panicked at
  `fixed_exp.rs:163` (`x=-25304236032 err=1.4977e-9 tol=9.5895e-10`), restore 103/0. Window
  `[-16·2^32, 21·2^32]` with `Result` refusal; measured max relative error 8.436e-9 for `x ≥ 0`,
  3.431e-7 for `x ≥ -8` (≤ `2^-20`). `i128` intermediates are used; extraction fitness is the
  question of `APEIRON-INT-TRANSCENDENTAL-SPIKE-EXTRACT`. `proof/` untouched, `ApeironWord.lean`
  `8e76b917…` unchanged.
- `APEIRON-INT-TRANSCENDENTAL-SPIKE-EXTRACT` (deepseek-flash, 105 s of the 45-min budget; report
  `/tmp/opencode/int-extract-spike.md`): Charon `0.1.254` and Aeneas `505b6ca-dirty` extract the real
  `fixed_exp` (`#[path]`-included, `i128 → Std.I128`, `Result` modeled) with exit 0, but the as-is
  emission carries three unmodeled axioms — `I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from`,
  `U32.Insts.CoreConvertTryFromI64TryFromIntError.try_from`, `core.num.I64.unsigned_abs` — which
  `proof/regen.sh`'s audit rejects. The smallest change tested — explicit `i128` range checks with
  `as` narrowing, a checked `k >= 64` guard with `k as u32`, and `if k < 0 { (-k) as u64 } else
  { k as u64 }` — emits `NO_AXIOMS` and agrees pointwise (`mismatches=0`) on the dense grid plus
  seven boundaries. That is a sample equivalence, not a proof; W1-T1's refinement closes the gap.
  `APEIRON-INT-TRANSCENDENTAL-W1-T0B` (deepseek-flash, 154 s of 30 min) applied the form:
  `src/numeric/fixed_exp.rs` is now
  `3538878f700ae26cda6eafe6b873ee5352e57a9c84ced587de4105ed48870aaa` (216 lines; the `i64`/`i128`
  range checks carry a reasoned `#[allow]` for cast lints because `RangeInclusive::contains`,
  `TryFrom` and `unsigned_abs` emit unmodeled externals). Gates: cargo trio exit 0 (103 passed),
  live re-extraction exit 0, audit `NO_AXIOMS` (independently re-verified: emitted
  `fixed/FixedExpSpike.lean` has no `axiom` declaration), equivalence `mismatches=0` over the dense
  grid plus boundaries. `proof/` untouched, `ApeironWord.lean` `8e76b917…` unchanged.
- `APEIRON-INT-TRANSCENDENTAL-W1-T2A` (deepseek-flash, ~5 min of 45): `src/numeric/fixed_rsqrt.rs`
  `4181249b7b14a013df391b2171d288ecf3c58795ba08052499a3699030678540` (204 lines),
  `src/numeric/mod.rs` `bda18029d8296f798fb94c1f4678c76ac82907048e650073c7cf5e79425e5444`
  (9 lines). The preferred scaled-root form landed: `s = isqrt(x << 64)` by a bit-by-bit restoring
  root, then `out = ((1 << 80) + s/2) / s`; `i128` division extracts as modeled, so the Newton
  fallback was not needed. Grade `tested`: cargo trio exit 0 (106 passed); extraction spike exit 0
  with `NO_AXIOMS` and `removed 0 model axiom(s)`; measured max relative error **1.1636e-7** against
  the `2^-20` target (8.2× margin), exact identities at `ONE`, `4*ONE`, `ONE/4`; control
  `/tmp/opencode/int-rsqrt-control/` baseline 3/0, red panic at `fixed_rsqrt.rs:150`
  (`x=4096 relative=0.5 target=9.536743e-7`), restore 3/0. The live `proof/extract/lib.rs` root for
  `word_fixed_rsqrt` is W1-T2's step.
- W1-T1 lead setup (before dispatch): `proof/corpus/lakefile.lean`
  `d66a71cc18e7cb5f560916866ab6f5da9bea06eae28508ef5bc20f9fd86d18b6` adds `lean_lib Numeric`
  (`globs := #[.submodules `Numeric]`); `proof/corpus/Numeric/ExpRefinement.lean` skeleton
  `ba1ad45a0756e0ccf995cdd3bfe425a1db2731eda59975971d61ff22993b6370`; `AxiomAudit.lean`
  `318de34046c894cf8554af02a964df053b8b50150e203d810b48bf335eefd870` imports the module (the
  `#audit` line is the lead's step after `fixed_exp_refines` lands). Gate
  `lake build Numeric.ExpRefinement AxiomAudit` exit 0 (3891 jobs).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1` (deepseek-flash, ~12 min of measured work; partial as
  designed): the extraction landed and the identity moved — `proof/ApeironWord.lean`
  `b2841ac7b1e44a05cb2a2b8962dd75433a2574b2f747955a7b411f372d5769a3` (10879 lines; `proof/regen.sh`
  exit 0, 42 s, 1708 jobs; `proof/regen.sh --check` exit 0, independently re-run). Emitted symbols:
  `fixed_exp.{ONE,LN2_Q32,LN2_RECIP_Q32,ExpError,round_div_2p32,fixed_exp}` and
  `word_fixed_exp : Std.I64 → Result (Result Std.I64 ExpError)`. `proof/extract/lib.rs`
  `28da3829dfce2c14232089499bd63b185ac564c3f124677be66fde07bbf6494e` (581 lines) adds the `#[path]`
  module and the wrapper; `proof/regen.sh`
  `6bbc33ac71b53687b6c80a543b2a4d78dd452018aa206e304197314f49b8e156` adds the source-digest entry
  and the `--start-from` root (undeclared fourth file, reviewed and accepted: additive registration
  only, audit logic untouched; it also completes the digest list with `src/table/prepared.rs`, which
  `src/table.rs:215` `include!`s). `proof/corpus/Numeric/ExpRefinement.lean`
  `13995ed598e785f043daeb547905afcb1696c4adf9ae78fab4f4b6427c415c5e` (98 lines) defines `q32`,
  `toReal`, `toReal128`, `window`, `epsExp`, `fixed_exp_refuses` and `fixed_exp_refines` (combined
  bound `2^-20·exp + 4/2^32`; the module doc records why the pure relative bound is false at the
  lower window edge where the Q32.32 grid dominates) and proves `word_fixed_exp_refuses_of_lt`/
  `_of_gt` parameterised by the extracted endpoint constants. Gate
  `lake build Numeric.ExpRefinement AxiomAudit` exit 0 (3891 jobs, 57 s), zero `sorry`; control
  `/tmp/opencode/int-exp-control2/` perturbs `LN2_Q32` by one unit and the re-extracted constant
  differs with no other semantic diff. Lead integration: nine declarations registered in
  `AxiomAudit.lean` `6e3be33e2fd910dc2cab1249496eaf85ad3d8caca6e180d8b8f1f62f58636ecf` (3891 jobs,
  closed cone). Open: the endpoint evaluations (`1#i64 <<< 32#i32 = ok 4294967296#i64`, then
  `LOWER`/`UPPER`) instantiate `fixed_exp_refuses` (`W1-T1B`), and the full bound is its own wave.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1B` (deepseek-flash, ~20 min of 30): `Numeric/ExpRefinement.lean`
  `aa1a30f1f804fc504403966bbd6a790cf56fa71daf87376ad83f8baa8e83ad1e` (166 lines) proves
  `fixed_exp_ONE_eq`, `fixed_exp_LOWER_eq`, `fixed_exp_UPPER_eq` (`I64.ShiftLeft_IScalar_spec` and
  `I64.mul_spec`, not `decide`) and `fixed_exp_refuses_proved : fixed_exp_refuses`. Gate exit 0
  (3891 jobs), zero `sorry`; control flips UPPER to `20·ONE` and the instance goes red. Lead
  integration: four declarations registered in `AxiomAudit.lean`
  `86a7a60368bc205462578235f9e2142b211e8ace1a9abb2aa8b1cdae75b4306c` (3891 jobs, closed cone).
  Note: `fixed_exp_ONE_eq`'s script pins the `step`-generated hypothesis name `z_post` on the pinned
  toolchain.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1A` (deepseek-flash, ~8 min of 45): `src/numeric/fixed_trig.rs`
  `0813a402b38e714bc446e9bd132a3627fdc6a0456ceb6ae297deb5c45977d400` (375 lines),
  `src/numeric/mod.rs` `7123b5499411b0f80ff0fc1d133fdc406428bc20e8ed42eb649c3da55a8f7231` (10 lines):
  `SinCos`-returning `fixed_sin_cos`, `2π` reduction with quadrant fold, degree-8 Taylor. Cargo trio
  exit 0 (110 tests); extraction `NO_AXIOMS`; measured max relative error 4.4735e-7 for `sin` and
  `cos` over 8.4 M samples (≤ `2^-20` with margin); control perturbs `SIN_C1` and the near-value test
  goes red.
- Batch extraction wiring (lead): `proof/extract/lib.rs` and `proof/regen.sh` add the `fixed_rsqrt`
  and `fixed_trig` roots; one regeneration moved the identity to `proof/ApeironWord.lean`
  `b85742bff7161274772d6a4ad8a45d701c77f37fbf1739822210303f94f29916` (11251 lines; `regen.sh` exit
  0; `regen.sh --check` exit 0 after one transient package-build exit 120 that an immediate re-run
  cleared — recorded as an unreproduced transient on those identities). `Numeric/RsqrtRefinement.lean`
  `4262798e…` and `Numeric/RopeRefinement.lean` `784ee3a0…` are roots, imported by `AxiomAudit.lean`
  `87fd8182…` (3893 jobs). `W1-T1C` (exp bound structure) and `W1-T2` (rsqrt refinement) run on this
  identity; `W2-T1` (trig refinement) is queued.
- `APEIRON-INT-TRANSCENDENTAL-W1-T2` (deepseek-flash, 30 min of 45; partial as designed):
  `Numeric/RsqrtRefinement.lean`
  `0fd57c82bf3dccecea899783b42b7db9e4ee61b631afd351a7faf9a8ecbfe676` (334 lines) proves the refusal
  surface (`rsqrt_window`, `fixed_rsqrt_refuses`, the two parameterised refusal lemmas, the endpoint
  evaluations `fixed_rsqrt_ONE_eq`, `LOWER_eq` = `2^12`, `UPPER_eq` = `2^52` via
  `Int.shiftRight_eq_div_pow`, `Int.shiftLeft_eq`/`Int.bmod`, and `fixed_rsqrt_refuses_proved`), states
  `fixed_rsqrt_refines` (combined bound), and proves the bound decomposition:
  `word_fixed_rsqrt_unfold`, `isqrt_real_bounds`, `isqrt_recip_error_bound`, `isqrt_scaled_error`,
  `round_nearest_div_error`, `sqrt_shift`. Gate exit 0 (3893 jobs), zero `sorry`; control flips the
  UPPER constant and the goal goes red at `RsqrtRefinement.lean:132`. Open: the extracted `isqrt` loop
  invariant (`s*s ≤ m < (s+1)^2`) and the composition through the guard and final cast. The receipt
  binds `ExpRefinement.lean` `391be78d…`, which W1-T1C extended concurrently; the module is re-gated
  before the lead registers its declarations.
- Lead step queued: a shared `Numeric/Scale.lean` (`q32`, `toReal`, `toReal128`, `epsExp`) so the
  refinement modules stop importing `ExpRefinement` for scale definitions; scheduled after W1-T1C
  lands, to decouple evidence binding across numeric lanes.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1C` (deepseek-flash, ~14 min of 45; partial):
  `Numeric/ExpRefinement.lean` `683356b0…` (387 lines, pre-refactor) proves
  `word_fixed_exp_window_past_guards` (rewrite past both evaluated endpoint guards, leaving the four
  inner range refusals), the guard side conditions, `fixed_exp_round_div_2p32_HALF_eq` and
  `fixed_exp_round_div_2p32_spec` (the generated `2^31` rounding step), the reduction budget
  `toReal_ln2`, `toReal_ln2_recip` and `ln2_recip_defect` (`6196328019·2977044472 − 2^64 =
  1953109352`), plus `int_shiftRight_nonneg`. Gate exit 0 (3893 jobs), zero `sorry`; controls mutate
  `ln2_recip_defect` and the rounding shift, both red. Open: seven arithmetic refusal goals over
  `x.val ∈ [-16·2^32, 21·2^32]` (listed in `/tmp/opencode/s6c.out`), then the `k`/`r` value
  characterization (`W1-T1D`, running).
- Lead refactor after W1-T1C: `Numeric/Scale.lean`
  `1a215651a72af0baf8269c8a2756b8aac2a63dfc0668c0a41a554a1155fad4b8` carries `q32`, `toReal`,
  `toReal128`, `epsExp`, `q32_pos`, `q32_ne_zero`; `ExpRefinement`, `RsqrtRefinement` and
  `RopeRefinement` import it rather than each other, decoupling evidence binding. Post-refactor
  digests: `ExpRefinement.lean` `8d0fc3ce…`, `RsqrtRefinement.lean` `4a09c1b0…`, `RopeRefinement.lean`
  `a28ef0c7…`; refactor gate exit 0 (3894 jobs).
- Lead registration: `AxiomAudit.lean`
  `00ab17947721702ba5eefc064746deb707920018278f15d71776762e230c96be` registers the W1-T1C and W1-T2
  declarations — all inside the closed cone, 3894 jobs. `W1-T1D` (exp range goals) and `W1-T2B`
  (isqrt invariant and composition) run on these digests; `W2-T1` (trig refinement) is next in queue.
- `APEIRON-INT-TRANSCENDENTAL-W1-T2B` (deepseek-flash, ~50–55 min of a 45-min budget — overrun; no
  work after the stop): `Numeric/RsqrtRefinement.lean`
  `94e816acb03e22f83b3046d347aa9841e04ce8d92edbd961781ca4eb7377c47d` (426 lines) proves the integer
  invariant core of the extracted restoring square root: `IsPow4`, `IsqrtInv` (one-sided coupling
  `res² = 4·bit·(n−rem)`, `rem < 2·res + 4·bit`, terminal `bit = 0 ⇒ res² ≤ n < (res+1)²`),
  `isqrt_inv_init`, `isqrt_inv_final`, the two step-preservation lemmas (`isqrt_step_hit`/`_miss`)
  and the two final-iteration lemmas (`isqrt_final_hit`/`_miss`). Gate exit 0 (3894 jobs), zero
  `sorry`; control mutates the hit result by `+1` and `linarith` fails at
  `RsqrtRefinement.lean:396`. Open: the monadic lift `loop.spec_decr_nat` on `isqrt_loop1` (measure
  `bit.val.toNat`), the `isqrt_loop0` seed reduction and `isqrt_spec`; the guard/cast/composition is
  a separate later slice. Lead registration: eight declarations in `AxiomAudit.lean`
  `473abde3e75d09fc684f141eb4df3a46acf03f1f0faada71a5eb719e89c92a66` (3894 jobs, closed cone).
  `W1-T2C` (same session) runs the lift under a 30/40-min bound.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1D` (deepseek-flash, ~55 min of a 45-min budget — overrun; stopped
  without wiring): `Numeric/ExpRefinement.lean`
  `c2be31b262041bb82b28b3561b358f243b549d614f6e832491d6cbbe3b9a73e9` (508 lines) proves six lemmas
  that discharge the seven side conditions arithmetically: `fixed_exp_shift_recip_shift_bound`
  (`k ∈ [-24, 30]` on the window), `fixed_exp_shift_recip_bound_window`
  (`k·2977044472 ∈ [I64.min, I64.max]`), `fixed_exp_bmod_64_lower`/`_upper` (the `bmod 2^64` range),
  `fixed_exp_bmod_self_64` (`a.bmod 2^64 = a` on the `I64` range) and
  `fixed_exp_reduction_remainder_range` (`r ∈ [-2^63, 2^63-1]`, from
  `6196328019·2977044472 = 2^64 + 1953109352`). Gate exit 0 (3894 jobs), zero `sorry`; control mutates
  the `s ≤ 30` bound and the mismatch is immediate. Open: the five-goal `step*`-context wiring (three
  named local normalization obstacles) and the continuation past `round_div_2p32`. Lead registration:
  six declarations in `AxiomAudit.lean`
  `b0a87ea91f899eafada90248b3aa0f2f647f62baa600705d603a2511da703676` (3894 jobs, closed cone).
  `W1-T1E` (same session) runs the wiring under a 25/35-min bound.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1E` (deepseek-flash, ~40 min of a 25/35 box — overrun; stopped
  after the wiring): `Numeric/ExpRefinement.lean`
  `973d610ba5d7785036c220d3f1f26eadcfad98ba33c200a6e87e56e8ab06b500` (614 lines) adds
  `word_fixed_exp_reduction_arith_safe`: on `window x` the extracted post-guard reduction prefix has
  all its `i128`/`i64` refusal arithmetic safe, discharging the five `step*` goals with the six
  W1-T1D lemmas and resolving the three normalization obstacles (`norm_num [I128.size, I128.numBits,
  Int.shiftLeft_eq, Int.bmod]`; literal-vs-`2^64` normalization; `hsv`/`hkval` built from the `_post`
  coercions). Gate exit 0 (3894 jobs), zero `sorry`; control mutates the `s ≤ 30` bound and the wiring
  goes red. Open: the success-post value characterization and the four guard goals (`W1-T1F`,
  running). Lead registration: `word_fixed_exp_reduction_arith_safe` in `AxiomAudit.lean`
  `b3e4edf5…`.
- `APEIRON-INT-TRANSCENDENTAL-W1-T2C` (deepseek-flash, ~70 min of a 30/40 box — overrun; stopped
  after the lift): `Numeric/RsqrtRefinement.lean` `65896e7e…` (666 lines) proves `isqrt_loop1_spec`
  by `loop.spec_decr_nat` (measure `bit.val.toNat`, invariant `IsqrtInv ∩ IsPow4 bit ∧ bit ≤ 2^116`,
  post `out.val² ≤ nz < (out.val+1)²`), splitting the final iteration from the step case and deriving
  the divisibility facts with `pow4_dvd_four` and `Int.prime_two.dvd_of_dvd_pow`; new pure helpers
  `IsPow4.div_four`, `pow4_dvd_four`, `isqrt_final_hit_inv`, `isqrt_final_miss_inv`. Gate exit 0
  (3894 jobs), zero `sorry`; one bounded local `set_option maxHeartbeats 1600000 in` on
  `isqrt_loop1_spec` (recorded residual: finite, local, not `0`). Control mutates the loop measure and
  the rewrite fails at `RsqrtRefinement.lean:603`. Open: `isqrt_loop0` seed reduction and
  `isqrt_spec` (`W1-T2D`, running), then the guard/cast/composition. Lead registration: five
  declarations in `AxiomAudit.lean` `b3e4edf5…` (3894 jobs, closed cone). Mid-slice the worker found
  `/tmp` at ~285M free and moved `TMPDIR` to `/home/socol/oc-tmp`.
- Environment (lead): `/tmp` is a 32G tmpfs that reached 93% (29G) on finished copy trees
  (`spike-locate` 18G, `verify-w5t15` 18G peak) plus stale wave/control copies. Cleanup reclaimed
  ~26G (now 18G used, 58%) while keeping every `*.log` receipt; the failure is registered in
  `AGENTS.md` §10 with the convention that scratch and control copies live under
  `/home/socol/oc-tmp` and only named logs under `/tmp/opencode`.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1F` (deepseek-flash, ~24 min of a 20/30 box — soft stop exceeded,
  hard stop not breached): `Numeric/ExpRefinement.lean`
  `5b0109326efa06e173b58b9212b3af80ad7ae24f8891cb0b13f9bfaa57517f78` (812 lines) adds
  `word_fixed_exp_reduction_value`: on `window x` the post-guard reduction prefix returns `Ok r` with
  `∃ k, k.val = (x.val·6196328019 + 2^63) >>> 64 ∧ r.val = x.val − k.val·2977044472`; the witness is
  `k := IScalar.cast .I64 shifted`, the four guard branches are refuted by the range lemmas, and the
  post closes by `r_post` + `fixed_exp_bmod_self_64`. Gate exit 0 (3894 jobs), zero `sorry`; control
  mutates the post's `k` multiplier and the mismatch is immediate. Lead registration:
  `word_fixed_exp_reduction_value` in `AxiomAudit.lean` `ef5de704…`. Open: the Horner chain value
  (`W1-T1G`, running), then the scaling branches.
- `APEIRON-INT-TRANSCENDENTAL-W1-T2D` (deepseek-flash, within the 20/30 box):
  `Numeric/RsqrtRefinement.lean`
  `b4ce924deb0cb78244b8e26608d17f54282840f7ccf17e6276a1d9b951b13ec1` (740 lines) proves
  `isqrt_loop0_spec` (`loop.spec_decr_nat`, measure `b.val.toNat`, invariant `IsPow4 b.val ∧ 0 < b.val
  ∧ n.val < 4·b.val`, post `IsPow4 out.val ∧ 0 < out.val ∧ out.val ≤ n.val ∧ n.val < 4·out.val`) and
  `isqrt_spec` (`1 ≤ n.val ≤ 2^116` implies `s.val² ≤ n.val < (s.val+1)²`), with the seed evaluated
  to `2^126 = 4^63`. Gate exit 0 (3894 jobs), zero `sorry`; a second bounded local
  `set_option maxHeartbeats 1600000 in` on `isqrt_spec` (recorded residual: finite and local); control
  mutates the loop0 invariant factor and the step goes red at `RsqrtRefinement.lean:705`. Lead
  registration: two declarations in `AxiomAudit.lean` `2ac66fb9…` (3894 jobs, closed cone). Open: the
  guard/cast/composition (`W1-T2E`, running), after which `fixed_rsqrt_refines` closes.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1G` (deepseek-flash, ~29 min of a 20/30 box — soft exceeded,
  inside the hard stop): `Numeric/ExpRefinement.lean`
  `f91ae652ed9925c459af5923cb53cc48b08b768101a54c7c99199825d8620b72` (869 lines) adds the
  round-value machinery — `fixed_exp.round_q32` (the mathematical `round_div_2p32`) and the value
  spec `fixed_exp_round_div_2p32_val` via `spec_mono` and a sign case — the target nested expression
  `fixed_exp.horner_acc`, and the first chain step `word_fixed_exp_horner_first`
  (`i13.val = round_q32 (852176 * r.val)`) with the `spec_bind`-plus-overflow-premise method the
  rest of the chain reuses. Gate exit 0 (3894 jobs), zero `sorry`; control mutates the first step's
  multiplier and the goal is immediately unsolved. Lead registration: four declarations in
  `AxiomAudit.lean` `fed6eaef…` (3894 jobs, closed cone). Open: the six further stages to `acc6`
  (`W1-T1H`, running), then the scaling branches.
- Guard raises inventory (residuals for the wave barrier; policy forbids unbounded or unreported
  ones): `Numeric/ExpRefinement.lean` carries three bounded, local `set_option maxRecDepth 4000 in`
  (lines 506, 612, 841; default 1000) and `Numeric/RsqrtRefinement.lean` two bounded, local
  `set_option maxHeartbeats 1600000 in` (lines 476, 712; default 200000). None is `maxHeartbeats 0`,
  none is global, each is attached to one theorem, and every gate command stayed inside 540 s.
  W1-T1G's raise was not disclosed in its report — the disclosure gap is noted; later slices must
  report any guard use, and no increase beyond these values is permitted without a lead decision.
  Removing or restructuring the raises is an open obligation at the wave barrier.
- `APEIRON-INT-TRANSCENDENTAL-W1-T2E` (deepseek-flash, at the 20/25 hard stop — not closed):
  `Numeric/RsqrtRefinement.lean`
  `c98fbf7241de0a9c18bc0efcefacc16df9711f7999d06ceb9ba6ce7acc6dec7f` (770 lines) proves three
  real-analysis helpers — `toReal_pos`, `sqrt_toReal` (`√(toReal x) = √x.val / 2^16`) and
  `one_div_sqrt_toReal` (`1/√(toReal x) = (2^48/√x.val)/2^32`, the Q32.32 denominator the root and
  quotient bounds are stated against). Gate exit 0 (3894 jobs), zero `sorry`; control mutates the
  `2^48` exponent and the helper goes red at `RsqrtRefinement.lean:763`. `fixed_rsqrt_refines` remains
  a `def`; the remainder is split — item 1 structural evaluation (`W1-T2F`, running, integer-only),
  then item 2 the real composition. Lead registration: three declarations in `AxiomAudit.lean`
  `20cf1400f4e416be37054452d042cafd3b08cd91a9e1edd2d0fad69a401076e8` (3894 jobs, closed cone).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1H` (deepseek-flash, ~27 min of a 20/30 box — soft exceeded, inside
  the hard stop): `Numeric/ExpRefinement.lean`
  `9a5174bbf94b2a90ce738a0576582ee4cdbd663d6eb9ec4c17cfd0b4cd73e60b` (898 lines) adds
  `fixed_exp.round_q32_abs_le` (`|round_q32 v| ≤ |v|/2^32 + 1`) and `fixed_exp.round_safe`
  (`|x| ≤ 2^100` discharges both `round_div_2p32` overflow premises for Horner intermediates). Gate
  exit 0 (3894 jobs), zero `sorry`; the slice added no `set_option` and reported the guard inventory
  correctly — the no-new-guard rule holds. Control mutates the `+1` bound and the goal is unsolved.
  Blocker: the `5965232#i128` literal's well-formedness `decide` does not reduce under the instances
  transparency after the first `spec_bind`, so `step*` stalls on the continuation; trace
  `/home/socol/oc-tmp/scratchtrace2.out`. `acc6` is not closed. `W1-T1I` (running) tries `@[step]`
  registration of the landed round spec, explicit `I128` add/mul specs, or `IScalar.ofInt` literals.
  Lead registration: two declarations in `AxiomAudit.lean` `4498efe7…` (3894 jobs, closed cone;
  `round_safe` needs only `{propext, Quot.sound}`).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1I` (deepseek-flash, ~32 min of a 20/30 box — overrun; not closed):
  `Numeric/ExpRefinement.lean`
  `730304371b72594826079c8a40c0125e63a7e875a206e70c48414c405bf3eb2e` (908 lines) adds
  `fixed_exp.round_q32_abs_le'` (`|v| ≤ C·2^32 → |round_q32 v| ≤ C + 1`, the parameterised bound each
  stage consumes). Findings: `I128.add_spec`/`mul_spec` are already `@[step]`, so `@[step]`
  registration is not the fix; the transparency failure is triggered by any tactic touching a
  continuation goal containing an `…#i128` literal; the explicit `WP.spec_bind` chain works around it
  (a two-stage instance compiles) but the generated 7-stage proof times out at the default 200000
  heartbeats at four `spec_bind` unifications. Gate exit 0 (3894 jobs), zero `sorry`, no new
  `set_option`; control mutates `+1` to `+0` and the goal is unsolved (`round_safe`'s `2^100` is slack
  and is not a usable control). `W1-T1J` (running) factors the stage into a standalone lemma and
  recomposes. Lead registration: `round_q32_abs_le'` in `AxiomAudit.lean` `6429a481…` (3894 jobs,
  closed cone).
- `APEIRON-INT-TRANSCENDENTAL-W1-T2F` (deepseek-flash, at the soft stop ≈25 min of a 20/30 box):
  `Numeric/RsqrtRefinement.lean`
  `b70b804af41dfedbcc84f90775a0830b82a92c51534377e323bc1691de3bb107` (968 lines) proves
  `word_fixed_rsqrt_success`: on `rsqrt_window x`, `word_fixed_rsqrt x = ok (Ok v)` with `∃ s,
  s² ≤ x.val·2^64 < (s+1)² ∧ 2^38 ≤ s.val ≤ 2^58 ∧ v.val = (2^80 + s.val/2)/s.val ∧ 0 ≤ v.val ≤
  2^61` — the full integer structural evaluation (`scaled`, `s`, `i3`/`i4`/`num`/`out`, the `I64`
  guard false, the cast), closing the side goals that failed in the W1-T2E scratch. Gate exit 0
  (3894 jobs), zero `sorry`, no new `set_option` (compiles at the default heartbeat cap); control
  mutates the claimed numerator `2^80` → `2^79` and the goal is unsolved. Lead registration:
  `word_fixed_rsqrt_success` in `AxiomAudit.lean` `6ba5c5eb…` (3894 jobs, closed cone). Open: item 2,
  the real composition (`W1-T2G`, running), after which `fixed_rsqrt_refines` closes.
- Operational note: the rsqrt gate hit a transient failure from the concurrently edited
  `Numeric/ExpRefinement.lean` (unused-variable warning-as-error) reached through `AxiomAudit`'s
  import; the rsqrt module built alone (exit 0) and the owner fixed it, after which the full gate was
  green. The shared `AxiomAudit` import makes each numeric lane's full gate sensitive to the other's
  mid-edit state; both the transient log and the green run are retained.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1J` (deepseek-flash, ~25 min of a 20/30 box — overrun; `acc6` not
  closed): `Numeric/ExpRefinement.lean`
  `f823b6ec248b44f9b61b892353bafad82d153bf6a1dab63d832900f56d281240` (943 lines) proves the factored
  stage lemma `fixed_exp.horner_step` — for `|acc| ≤ B ≤ 2^68`, `|r| ≤ 2^32`, `|c| ≤ 2^32`,
  `(do i ← acc*r; i1 ← round_div_2p32 i; a ← i1 + c; ok a) ⦃ z => z.val = round_q32 (acc.val·r.val)
  + c.val ⦄`, via `I128.mul_spec`, the round spec, `round_safe` and `round_q32_abs_le'`. Gate exit 0
  (3894 jobs), zero `sorry`, no new guard; control mutates the claimed result and the goal is
  unsolved. Composition fails structurally: `WP.spec_bind` needs the factored block as the head `m`,
  but the extracted body is left-nested (bind associativity is a theorem, not defeq); the site is
  `/home/socol/oc-tmp/gen2.lean:89`. Lead registration: `horner_step` in `AxiomAudit.lean`
  `bb46f9a1fceba7a05a1d729ae39a1d94aaa29c3ffc9379ec2488857e6671e5e4` (3894 jobs, closed cone).
- Lead decision (guard authorization): `W1-T1K` may place **exactly one** bounded
  `set_option maxHeartbeats 1600000 in` on the final `word_fixed_exp_horner_acc` composition theorem
  if the reassociation route does not land in its time-box — the same finite, per-theorem value
  already recorded for the rsqrt lift, with no higher value and no other guard. The raise enters the
  same wave-barrier obligation as the existing five: removal or restructuring before closure.
- **W1-T2 closure** — `APEIRON-INT-TRANSCENDENTAL-W1-T2G` (deepseek-flash, ≈22 min of a 20/30 box —
  closed before the hard stop): `Numeric/RsqrtRefinement.lean`
  `b50c6c59e7f965336e66e68888d3afa80a77350f191d92f1d0cd5169147754a1` (1074 lines) proves
  `isqrt_scaled_real` and `fixed_rsqrt_refines_proved : fixed_rsqrt_refines` — the full real bound
  `|toReal v − 1/√(toReal x)| ≤ epsExp·(1/√(toReal x)) + 4/q32` on `rsqrt_window`, composing
  `word_fixed_rsqrt_success`, `round_nearest_div_error`, `isqrt_scaled_real`, `sqrt_shift` and
  `one_div_sqrt_toReal`, with `fixed_rsqrt_refuses_proved` covering the complement. Gate exit 0
  (3894 jobs), zero `sorry`, no new `set_option` (the two pre-existing bounded rsqrt guards remain as
  recorded residuals); control mutates the final bound `4/q32` → `3/q32` and the proof is unsolved.
  Lead registration: two declarations in `AxiomAudit.lean`
  `a11ddcd3ca48ba98e6f2f1b2af2c30a6454968924568895e3a26d240065b2bab` (3894 jobs, closed cone).
  **The `rsqrt` refinement is closed** (statement, refusal, bound, cone) and is a candidate for the
  independent-verifier pass at the wave barrier. `W2-T1` (trig refinement) starts on the freed lane
  with the refusal surface, the statement and the structural unfold.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1` (deepseek-flash, ≈25 min of a 20/30 box — inside the hard stop):
  `Numeric/RopeRefinement.lean`
  `7e3dcd6d9c8f67a59925646332aecdf0af90773ed3d6ab64e7e5f8e08a2d750c` (410 lines) proves the trig
  refusal surface (`trig_window`, `fixed_sin_cos_refuses`, `fixed_sin_cos_refuses_proved`, the three
  endpoint evaluations) and the structural reduction facts (`word_fixed_sin_cos_window_past_guards`;
  `toReal_two_pi`/`_recip`/`_pi`/`_half_pi`/`_quarter_pi`; `two_pi_recip_defect = 11400346268`;
  `fixed_trig_shift_bound` (`-652 ≤ k ≤ 652`); `fixed_trig_shift_bound_window`;
  `fixed_trig_reduction_remainder_range` (`|r| ≤ 2^36`, with the note that the `ln 2` module's tight
  `2^-32` bound is false here)), and states `fixed_sin_cos_refines` (relative `2^-20` with an
  absolute floor `trig_floor = 2^-20`; the module doc records that the reduction displacement
  `|k|·2^-33 ≤ 7.6e-8` forces a floor and that the Rust test's `5e-7·|f| + 2^-24` is a dense-grid
  measurement, not the universal statement). Gate exit 0 (3894 jobs), zero `sorry`, no `set_option`
  added; control mutates the lower endpoint and the refusal proof goes red. Lead registration:
  eighteen declarations in `AxiomAudit.lean` `29847cb6…` (3894 jobs, closed cone). Open: the reduction
  value characterization (`W2-T1B`, running; a single bounded `maxRecDepth 4000` is pre-authorized if
  needed), then the reduction/Taylor error decomposition — the long pole.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1K` (deepseek-flash, ~29 min of a 20/30 box — overrun; `acc6` not
  closed): `Numeric/ExpRefinement.lean`
  `73a9bcb49176f6ab772033518f54a03ff9febddd0a28a3cddf6200e3e6561323` (1025 lines) proves
  `word_fixed_exp_horner_acc1` — the first two Horner stages at the default heartbeat cap via the
  explicit per-operation route. Tier 1 (reassociation) did not land inside its time-box; Tier 2 at
  exactly 1600000 heartbeats still times out (`whnf`/`isDefEq` at six sites, `omega` at two), so no
  option was landed. Gate exit 0 (3894 jobs), zero `sorry`, the three pre-existing `maxRecDepth 4000`
  guards unchanged; control mutates the claimed coefficient and the goal is unsolved. Blocker: the
  21-operation flat body's `spec_bind` unification cost exceeds 1600000 heartbeats. Lead registration:
  `word_fixed_exp_horner_acc1` appended to `AxiomAudit.lean` (digest `588738df…`), **build pending** —
  the concurrent trig lane's mid-edit `RopeRefinement.lean` (`:495` unsolved goals) breaks the shared
  `AxiomAudit` import; re-verify when that lane settles.
- Lead decision (cap ceiling): `W1-T1L` may test finite caps 2000000/4000000/8000000 in scratch and
  land the minimal closing value on the single `word_fixed_exp_horner_acc` theorem; 8000000 is the
  ceiling, structural reassociation is still preferred, and any landed cap joins the wave-barrier
  removal/restructuring obligation.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1B` (deepseek-flash, ≈8 min of a 20/30 box — well inside):
  `Numeric/RopeRefinement.lean`
  `54ae566624ac097ddabaaeabfd8210c6869f6d106280455e99182e7789aeec61` (631 lines) proves
  `word_fixed_sin_cos_reduction_value`: on `trig_window`, the post-guard prefix returns `Ok r` with
  `k.val = (x.val·683565276 + 2^63) >>> 64` and `r.val = x.val − k.val·26986075409`, the twelve side
  conditions discharged by the landed bounds (`fixed_trig_shift_bound`, `fixed_trig_shift_bound_window`,
  `fixed_trig_reduction_remainder_range`, plus the generic `int_bmod_64_lower/upper/self_64` helpers).
  The authorized `maxRecDepth 4000` was placed, validated as unnecessary in a disposable copy, and
  removed — the live file carries **no** `set_option`. Gate exit 0 (3894 jobs), zero `sorry`; control
  mutates the post's `2^63` shift and the mismatch is immediate. Lead registration:
  `word_fixed_sin_cos_reduction_value` in `AxiomAudit.lean` `04df75ea…` (3894 jobs, closed cone) —
  this run also cleared the pending `word_fixed_exp_horner_acc1` audit line from W1-T1K. Open: the
  quadrant fold (`W2-T1C`, running), then the Taylor/error decomposition — the long pole.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1C` (deepseek-flash, ≈7 min of a 20/30 box — well inside):
  `Numeric/RopeRefinement.lean`
  `ac33bceec42ebd94297fe2147df615d77a040793bf8988f57e3b185ea3431601` (868 lines) proves the pure
  Q32.32 fold functions matching the extracted comparisons (`fixed_trig.fold1`/`fold1_negate`,
  `fold2`/`fold2_swap`), `fixed_trig.fold1_abs_le` (`|r| ≤ 2^36 → |fold1 r| ≤ 2^36`), the no-refusal
  WP specs `fixed_trig_fold1_spec` and `fixed_trig_fold2_spec`, and the composed
  `word_fixed_sin_cos_fold_value`. Gate exit 0 (3894 jobs), zero `sorry`, no `set_option`; control
  mutates `fold2_swap`'s upper branch and the spec goal is unsolved. Lead registration: eight
  declarations in `AxiomAudit.lean`
  `6502f7ddbab7534d4dbcefad0438813d3ec735e2588041bf16fe1dca778a3eef` (3894 jobs, closed cone). Open:
  the reduction→fold bridge and the Taylor value (`W2-T1D`, running); the error decomposition remains
  the long pole.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1D` (deepseek-flash, ≈7.5 min of a 20/30 box — inside), two slices:
  (1) the reduction→fold composition `word_fixed_sin_cos_reduction_fold_value` via the `Result`
  bind/match route, with `fixed_trig.fold_step` naming the extracted block and
  `fixed_trig_fold_step_spec` restating the fold value on it — the prefix yields `Ok r`, then the fold
  yields `Ok (r, s, negate, arg, swap)` with `s = fold1 r`, `negate = fold1_negate r`,
  `arg = fold2 (fold1 r)`, `swap = fold2_swap (fold1 r)`; (2) the Taylor head: the rounding value
  spec `fixed_trig_round_div_2p32_val`, the pure targets `sin_poly`/`cos_poly`, the target
  `fixed_trig_taylor_value`, and `word_fixed_taylor_first`. Gate exit 0 (3894 jobs), zero `sorry`, no
  `set_option`; control mutates the rounding constant `2147483648` → `2147483647` and the value spec
  goes red. Lead registration: twelve declarations in `AxiomAudit.lean`
  `d8f570398f9255174d1bbdb5469e8d414e4f143329502aeeb5943d84994fcca5` (3894 jobs, closed cone). Open:
  the remaining Taylor stages (`W2-T1E`, running), the literal-body bind-law bridge, then the error
  decomposition.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1E` (deepseek-flash, ≈7 min of a 20/30 box — inside): the decisive
  result is a **falsification and correction of the Taylor domain** — as first stated (`|t| ≤ 2^36`)
  the target `fixed_trig_taylor_value` is false: at `|t| = 2^36` the degree-8 polynomial reaches
  `~2^105`, which does not fit the final `i64` cast. It is re-stated at `|t.val| ≤ 2^32` with the
  range analysis in the doc comment (intermediates ≤ `2^67`, outputs ≤ `2^35`); `2^32` is also the
  true post-fold argument bound (`|arg| ≤ π/4 < 2^32` in Q32.32). Landed the per-stage tools
  `fixed_trig.round_q32_abs_le`, `round_q32_abs_le'` and `round_safe`. Gate exit 0 (3894 jobs), zero
  `sorry`, no `set_option`; control mutates the round-1 polynomial term and the first stage goes red.
  Lead registration: three helper declarations in `AxiomAudit.lean`
  `327a4a920c8634d113cfaa4640f53ca103debeeda20619cf4867378a9fe325dd` (3894 jobs, closed cone). Open:
  the nine-stage chain (`W2-T1F`, running), the literal-body bridge, then the error decomposition.
- **W1-T1L closure** — `APEIRON-INT-TRANSCENDENTAL-W1-T1L` (deepseek-flash, ~47 min of a 20/30 box —
  overrun; `acc6` closed): `Numeric/ExpRefinement.lean`
  `74ea5d23cc13f087723e0403d31fdb26334c5b2b0e64b22751bc9c3f22d7c030` (1311 lines) proves
  `word_fixed_exp_horner_acc` — the full chain from `i12` to `acc6`, post
  `acc6.val = fixed_exp.horner_acc r.val` given `|r.val| ≤ 2^32` — by the explicit per-operation route
  with one lead-authorized `set_option maxHeartbeats 8000000 in` on that theorem (2M and 4M tested and
  insufficient; 8M is the minimal closing value; observed compile 167 s, inside the 540 s gate).
  Control mutates one coefficient of the claimed `acc6` and the goal is unsolved (154 s red; restore
  green). Lead registration: `word_fixed_exp_horner_acc` in `AxiomAudit.lean` `5c59dfd2…` (3894 jobs,
  closed cone; the run also cleared the earlier pending line). Open: the scaling branches (`W1-T1M`,
  running), then the scaling/error composition.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1F` (deepseek-flash, ≈10.5 min of a 20/30 box — inside; wall hit):
  the nine-stage Taylor chain runs through stages 1–3 and walls at stage 4 on the default 200000
  heartbeats in `whnf`/`isDefEq`/`nlinarith` (probe sites `RopeRefinement.lean:1231:16`, `:1231:65`,
  `:1235:38`). No `set_option` was added; the file is restored to the green W2-T1E revision
  `4f6f84b9…`. Mitigations tried without clearing the wall: two-sided accumulator bounds, `mul_le_mul`
  instead of `nlinarith` over `abs` products, and reducing `round_safe`'s bound to `2^66`. The
  recommended and dispatched fix is per-stage lemmas so no elaboration sees the whole nested
  continuation (`W2-T1G`, running); a cap bump (≈400k–800k) was judged fragile.
- Guard inventory (updated): `Numeric/ExpRefinement.lean` carries three bounded local
  `set_option maxRecDepth 4000 in` and one bounded local `set_option maxHeartbeats 8000000 in` (acc6,
  167 s compile); `Numeric/RsqrtRefinement.lean` carries two bounded local
  `set_option maxHeartbeats 1600000 in`. All finite, per-theorem and reported; the
  removal/restructuring obligation at the wave barrier now covers six raises.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1M` (deepseek-flash, ~29 min of a 20/30 box — at the hard stop; no
  landing): the file is unchanged (`74ea5d23…`, 1311 lines) and no `set_option` was added. The slice
  established the exact scaling goals — `step*` drives the scaling block and splits both branches
  (manual `if`-reasoning on `k >= 0#i64`/`k >= 64#i64` hits the `#i64` literal instance-transparency
  failure, so the split must come from `step*`); with a value post on the `k ≥ 0` sub-block the
  success path reduces to the guards confining `scaled` to the `i64` range, `fixed_exp_bmod_self_64`
  and the shifted value; the near-complete shift-count lemma `0 ≤ k.val → k.val < 64 → (hcast .U32
  k).val < 128` fails only on numeral normalization (`2^32` vs `4294967296`). Remaining goal: the
  `word_fixed_exp_scale` value characterization for both branches. `W1-T1N` (running) lands the
  shift-count lemma and the branch values under a 20/30 box with at most one 8000000 cap.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1G` (deepseek-flash, ≈7 min of a 20/30 box — inside; wall persists,
  no landing): the nine CPS stage functions (`taylor_u`, `taylor_s1`–`s4`, `taylor_c1`–`c4`) were
  written and compile cleanly; the definitional bridge to the extracted body
  (`fixed_trig.taylor_sin_cos_decomp`, composition = `taylor_sin_cos`) times out in `whnf` at `rfl` on
  the default 200000 heartbeats, and the stage-4 `nlinarith` sits at the same ceiling. No `set_option`
  was added; the file is restored to the green revision `4f6f84b9…`. Lead decision: `W2-T1H` may test
  400000/600000/1000000 in scratch and land the minimal closing value (ceiling **1000000**) on the
  single decomposition theorem; if that fails, a bounded stage-by-stage `change` attempt is allowed.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1N` (deepseek-flash, ~35 min of a 20/30 box — overrun; step 1
  landed): `Numeric/ExpRefinement.lean`
  `80069af62060d513d709a8406ddc7a35787594662196008cdf06d79c09600fdc` (1321 lines) proves
  `fixed_exp_scale_shift_count_lt` (`0 ≤ k.val → k.val < 64 → (hcast .U32 k).val < 128`), fixing the
  numeral-normalization blocker with `norm_num [UScalarTy.numBits]`, `Int.emod_eq_of_lt` and `omega`.
  Module build exit 0 (2224 jobs, 154 s); the full gate is blocked only by the concurrent trig lane's
  mid-edit file. Control note: the `< 128 → < 64` mutation is slack and does not go red — reported so
  it is not mistaken for a control; `< 128 → < 0` goes red. No `set_option` added. Step 2's blocker is
  named: the `IScalar` guard hypotheses (`¬ scaled > cast I128 I64.MAX`, `¬ scaled < cast I128
  I64.MIN`) resist `simp only [IScalar.lt_equiv]` and must be exposed as Int inequalities for `omega`;
  the residual bmod equality closes with `k.val % 2^32 = k.val`. Lead registration:
  `fixed_exp_scale_shift_count_lt` appended to `AxiomAudit.lean` (`46908738…`), **build pending** — the
  trig lane's mid-edit `RopeRefinement.lean` breaks the shared import; re-verify when it settles.
  `W1-T1O` (running) tries the comparison-lemma recipes and lands steps 2–4.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1O` (deepseek-flash, ≈26 min of a 20/30 box — soft stop exceeded,
  hard stop not reached; step 2 landed): `Numeric/ExpRefinement.lean`
  `f7f02a50707a5c7ddefd47c3cafcbeaf1596e8f3e0d924dd361e7b888b222aea` (1379 lines) proves
  `fixed_exp_scale_nonneg` — for `0 ≤ k.val`, `k.val < 64`, the `k ≥ 0` branch returns `Err` on `i64`
  overflow, otherwise `Ok v` with `(v.val : ℤ) = (acc6.val <<< k.val.toNat) mod 2^128`. The W1-T1N
  blocker is resolved with the recipe `rename_i` (name the inaccessible guard hypotheses),
  `rw [gt_iff_lt]` + `rw [IScalar.lt_equiv]` + `not_lt.mp` to expose the Int inequalities, closed
  against the landed cast-value constants; the residual bmod equality via `Int.emod_eq_of_lt`.
  Module build exit 0 (2224 jobs, 160 s); the full gate is blocked only by the concurrent trig lane.
  Control (non-slack): branch value `… bmod 2^128` → `(… + 1) bmod 2^128` goes red (166 s); restore
  cached green. No new `set_option`. Lead registration: `fixed_exp_scale_nonneg` appended to
  `AxiomAudit.lean` (`2575378457a095129c1e714da1b31ea0596b8c6f8ef2fa90c3a38cc2f2884d0b`), **build
  pending** while the trig lane is mid-edit. Open: step 3 (the `k < 0` branch — exact post in the
  task) and step 4 (`word_fixed_exp_scale` composition); `W1-T1P` (running).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1P` (deepseek-flash, ~29 min of a 20/30 box — at the hard stop; no
  landing): the file is unchanged (`f7f02a50…`, 1379 lines); no `set_option` added. The slice reduced
  the `k < 0` core block via `step*` to exactly four goals (underflow `1 ≤ n.val`; `I128.min ≤ acc6 +
  half`; `acc6 + half ≤ I128.max`; the value goal) and named the mechanical fixes: (a) the underflow
  coercion via `show (1:ℤ) ≤ (↑n : ℤ); exact_mod_cast` (`norm_num` leaves the `1#u64` literal
  unnormalized), (b) an explicit `calc` for `2^m < 2^128` before `Int.emod_eq_of_lt`/`bmod_eq_emod`,
  (c) keeping the `half`/`i27_post` hypotheses inside the goal bullets. The value goal's witness chain
  is otherwise fully identified (`cast_val_eq` → `bmod 2^64`, `rounded_post`, `i27_post`,
  `Int.shiftLeft_eq`). No control (no new claim; the W1-T1O control stands). `W1-T1Q` (running) lands
  step 3 with these fixes; step 4 only if the box allows.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1H` (deepseek-flash, ≈15 min of a 20/30 box — inside; cap route
  failed, structural route landed): the authorized cap route is **closed as failed** — `set_option
  maxHeartbeats 1000000 in` on `taylor_sin_cos_decomp` (`rfl`) still times out at `isDefEq` (45 s), so
  the decomposition ceiling is above 1000000 and unauthorized; no `set_option` was landed. The
  structural calc route landed instead: `Numeric/RopeRefinement.lean`
  `ef533655e601713ca89336633224e55e787600506e700490ceedd93f855df84b` (1427 lines) adds the bound
  helpers (`fixed_trig.abs_mul_le`, `mul_range`, `abs_mul_le_z`, `abs_add_const_le`,
  `abs_sub_const_le`, `round_safe66`) and **`word_taylor_spec`** proving the whole nine-stage Taylor
  body total on `|t.val| ≤ 2^32` with every `i128`/`i64` range and cast side condition discharged at
  the table bounds (`u ≤ 2^32+1`, `q1 ≤ 2^26`, `q2 ≤ 2^30`, `q3 ≤ 2^33`, …, `p4 ≤ 2^34`), `nlinarith`
  replaced by small lemmas. Gate exit 0 (3894 jobs), zero `sorry`, no `set_option`; control mutates the
  `round_q32` add constant and the value spec goes red. Lead registration: the seven declarations
  appended to `AxiomAudit.lean`
  (`4dfa9777902283e5c773fe73532484dd2318b6bbcb211485967d9919cbb89261`), **build pending** — both lanes
  are mid-edit (`ExpRefinement.lean:1452`, `RopeRefinement.lean:1421`); re-verify when they settle.
  Open: the value post (`∃ sin_t cos_t, z = (sin_t, cos_t) ∧ sin_t.val = sin_poly t.val ∧ cos_t.val =
  cos_poly t.val`) and `fixed_trig_taylor_value` (`W2-T1I`, running), then the literal-body bind-law
  bridge, then the error decomposition.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1I` (deepseek-flash, ≈13 min of a 20/30 box — inside;
  `fixed_trig_taylor_value` **closed**): `Numeric/RopeRefinement.lean`
  `3ef4b3d676bee2b30d186f8bbba924a3f513f9cd68b4d9d9590e1a9a136c2a13` (1457 lines) carries the value
  post on the calc chain (`word_taylor_spec`: `∃ sin_t cos_t, z = (sin_t,cos_t) ∧ sin_t.val =
  sin_poly t.val ∧ cos_t.val = cos_poly t.val`) and derives `fixed_trig_taylor_value_proved` via
  `WP.spec_equiv_exists`; the final casts are proved equal to their inputs with `IScalar.cast_val_eq`
  + `int_bmod_self_64` from in-context abs bounds. **Defect fixed en route**: `fixed_trig.cos_poly`
  carried one extra `round_q32` — corrected to the extracted four cos roundings; the value fold is
  what caught it. Gate exit 0 (3894 jobs), zero `sorry`, no `set_option`; control mutates the
  `sin_poly` coefficient `35791394 → 35791395` (non-slack) and the `refine_1` sin equality is
  unsolved. Lead registration + clearance: `fixed_trig_taylor_value_proved` added; the full
  `lake build AxiomAudit` is **green** (`27653a4ef4a57d6e97b49262383d8f1a1e26c8c0a84b61731f2f9f59e6ecac44`,
  3894 jobs), clearing every pending line (`fixed_exp_scale_shift_count_lt`, `fixed_exp_scale_nonneg`,
  and the W2-T1H set). Remaining: the literal-body bind-law bridge (defeq route closed above the
  authorized 1000000 ceiling), then the error decomposition. `W2-T1J` (running).
- `APEIRON-INT-TRANSCENDENTAL-W2-T1J` (deepseek-flash, ≈6 min of a 20/30 box — inside; partial): the
  file `f28484f7…` (1466 lines) lands `fixed_trig.fold_arg_abs_le` (`|r| ≤ 2^32 → |fold2 (fold1 r)| ≤
  2^32`), verified in `AxiomAudit.lean` `6d20647b…` (3894 jobs). Two findings: (1) the literal-body
  defeq bridge has no local `change` site — the extracted body is a flat let-chain, so every rewrite
  sees the whole continuation (the same `isDefEq` wall above the authorized 1000000 ceiling), and the
  match-form composition is not `spec_bind`-composable with it; (2) **the loose remainder bound blocks
  the Taylor path**: with `|r| ≤ 2^36`, `fold2 (fold1 r)` reaches ~2^35.8, so `word_taylor_spec`'s
  `|arg| ≤ 2^32` is unmet — the tight remainder bound (`π/4` plus the `k`-scaled reciprocal defect) is
  the argument-reduction long pole and a prerequisite, not the error decomposition. Option (ii) —
  extend the calc chain through `taylor_sin_cos` and the swap/negate block — is the path once the
  bound lands; the global defeq route is closed. Control mutates the new lemma's `≤ 2^32` to `≤ 2^31`
  (non-slack) and `omega` finds the counterexample `3373259427 ≤ r ≤ 4294967296`; no `set_option`.
  `W2-T1K` (running) proves the tight bound.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1R` (deepseek-flash, ≈21 min of a 20/30 box — soft stop slightly
  exceeded; flush and full `k < 0` branch landed): `Numeric/ExpRefinement.lean`
  `e725881b92d7dc660b878bda263d1c147c57968085e5029386ce5f7aa345a0db` (1533 lines) proves
  `fixed_exp_scale_flush` (`64 ≤ n.val → Ok 0`) and `fixed_exp_scale_neg` — the full `k < 0` branch on
  `1 ≤ n.val`, `|acc6.val| ≤ 2^40`, with the `Ok 0`/round value/refusal post (composed by `by_cases` +
  `WP.spec_mono`). Gate exit 0 (3894 jobs, `AxiomAudit` green — the trig lane had settled). Control
  non-slack (flush result `0 → 1`) red (170 s), restore cached. No `set_option`. Lead registration:
  `fixed_exp_scale_flush` and `fixed_exp_scale_neg` verified in `AxiomAudit.lean` `2c44086f…`.
  Remaining: step 4 `word_fixed_exp_scale` (`W1-T1S`, running).
- `APEIRON-INT-TRANSCENDENTAL-W2-T1K` (deepseek-flash, ≈6 min of a 20/30 box — inside; landed with a
  correction): `Numeric/RopeRefinement.lean`
  `74022c67ec896ba1f5f15c45a97a3e323c70196dd8ed9ed42317a8fd3a04af18` (1521 lines) proves
  `fixed_trig_reduction_remainder_range_tight` (`|x − s·26986075409| ≤ 2^34` over the window: `|s| ≤
  652` from `fixed_trig_shift_bound`, reciprocal defect `652·11400346268 < 2^43`, the identity `r·R =
  t − 2^63 − s·D`, and `2^34·R > 2^63 + 2^43`), and relaxes `fold_arg_abs_le` to `|r| ≤ 2^34 ⇒ |arg| ≤
  2^32` so the tight bound feeds `word_taylor_spec`. **Task premise falsified**: `|r| ≤ 2^32` is false
  — `x = HALF_PI_Q32` gives `k = 0`, `r = 6746518852 > 2^32`; the true magnitude is `π·2^32 +
  2e-5·2^32 ≈ 2^33.65`, so `2^34` is the sharp power-of-two bound. The loose `2^36` theorem is
  untouched. Gate exit 0 (3894 jobs); control non-slack (`2^34` → `2^33`) red; no `set_option`. Lead
  registration: both verified in `AxiomAudit.lean` `c4e120e1…` (3894 jobs, closed cone). Remaining:
  the end-to-end value through the swap/negate block (`W2-T1L`, running), then the error decomposition.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1S` (deepseek-flash, ~28 min of a 20/30 box — at the hard stop; no
  landing): the file is unchanged (`e725881b…`, 1533 lines); no `set_option` — the reserved cap was
  correctly judged unnecessary because the wall is composition bookkeeping, not a defeq. The slice
  proved the `k ≥ 0, k < 64` case by `if_pos`/`if_neg` + `spec_mono` and named the two helpers that
  clear the rest: `fixed_exp_scale_abs` (the `n = k.val.natAbs` extraction lemma for the negation/
  `hcast`, `step*`-provable) and `fixed_exp_scale_nonneg_ge64` (`64 ≤ k.val → ok Err`); the `k ≥ 64`
  vacuous-conjunct failure is a `#i64` literal normalization (`0#i64`/`64#i64` atom mismatch under
  `omega`, cleared by `simpa`/`norm_num`). `W1-T1T` (running) lands the two helpers and the
  composition.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1T` (deepseek-flash, ~34 min of a 20/30 box — overrun; helpers
  landed): `Numeric/ExpRefinement.lean`
  `f7c174309e25059fb11c6f5058a142a68b5b55664413fcc38bc081601c121106` (1572 lines) proves
  `fixed_exp_scale_abs` (`k.val < 0` with the `-2^62` floor the `I64.min` negation overflow requires,
  giving `n.val = k.val.natAbs`; via `step*` + `IScalar.hcast_val_eq` + `Int.emod_eq_of_lt` + `omega`)
  and `fixed_exp_scale_nonneg_ge64` (`64 ≤ k.val → ok Err`). Module exit 0 (2224 jobs, 171 s); the full
  gate was blocked only by the concurrent trig lane. Control non-slack (post `natAbs + 1`) red
  (168 s), restore cached. No `set_option`. Lead registration: both verified in `AxiomAudit.lean`
  `79433b26…` (3894 jobs, closed cone). Remaining: the composition `word_fixed_exp_scale` with the
  exact three-case recipe (`W1-T1U`, running).
- `APEIRON-INT-TRANSCENDENTAL-W2-T1M` (deepseek-flash, ≈8 min of a 20/30 box — inside; no landing):
  the file is unchanged (`74022c67…`, 1521 lines); no `set_option`. The explicit-post route cleared the
  `WP.Post`/`I32`-literal friction (selector on `swap.val`), and the composition reduction → fold
  (`fixed_trig_reduction_remainder_range_tight`) → `fixed_trig.fold_arg_abs_le` → `word_taylor_spec`
  type-checked in scratch. The exact smallest open goal is the final negation's overflow side
  conditions (`¬ ↑sin_t = I64.min`, `¬ ↑cos_t = I64.min`); the needed bounds `|sin_t|, |cos_t| ≤ 2^34`
  are proved inside `word_taylor_spec`'s chain but not exposed by its post. `W2-T1N` (running)
  strengthens the post with the two bounds, updates the single `fixed_trig_taylor_value_proved`
  destructuring, and closes the six cases. No control (no new claim; the W2-T1K control stands).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1U` (deepseek-flash, ~22 min of a 20/30 box — soft stop slightly
  exceeded; **`word_fixed_exp_scale` closed**): `Numeric/ExpRefinement.lean` `4949303d…` (1686 lines)
  adds `fixed_exp_scale_abs_raw` (the raw-`if` form `WP.spec_bind` needs; the wrapped
  `fixed_exp_scale_abs` is retained) and `word_fixed_exp_scale` — the full scaling block on `-2^62 ≤
  k.val`, `|acc6.val| ≤ 2^40` with the exact value/refusal post — by the three-case recipe (`spec_mono`
  for both `k ≥ 0` cases; `spec_bind` + `spec_mono` for `k < 0`, flush mapped to the left disjunct).
  Module exit 0 (2224 jobs, 166 s); the full gate was blocked only by the concurrent trig lane. Control
  non-slack (flush `Ok 0` → `Ok 1`) red (164 s), restore cached. No `set_option`; the reserved 8M cap
  was not used. Lead registration: `fixed_exp_scale_abs_raw` and `word_fixed_exp_scale` appended to
  `AxiomAudit.lean` (`e77ed344…`), **build pending** while the trig lane is mid-edit. Remaining: the
  whole-body composition (`W1-T1V`, running), then the real-error decomposition into
  `fixed_exp_refines`.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1V` (deepseek-flash, ≈16 min of a 20/30 box — inside; no landing,
  structural obstacle named): the file is unchanged (`4949303d…`, 1686 lines); no `set_option`. The
  whole-body composition fails at the first `spec_bind`: the landed chunk lemmas are multi-operation
  chunks (`A >>= fun r => ok r`) while the flat extracted body is right-nested (`A >>= fun r => rest
  r`); they are equal by `bind_assoc` + `bind_ok` (`Aeneas/Std/Primitives.lean:259/287`), not
  definitionally — the same associativity obstruction that blocked the earlier `horner_step` factor
  and the trig literal-body bridge. The reserved 8M cap is not indicated (not a heartbeat wall).
  Restructuring named: state `word_fixed_exp_body` in explicitly chunked form and prove the flat body
  equal via a WP-level `bind_assoc`/`bind_ok` rewrite (`conv` descent + `rw [← Aeneas.Std.bind_assoc]`),
  or add a small `spec_bind`-compatible reassociation lemma; then compose with the three chunk lemmas.
  `W1-T1W` (running) lands it. No control (no new claim).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1W` (deepseek-flash, ≈15 min of a 20/30 box — soft stop reached;
  no landing, obstacle sharpened): the file is unchanged (`4949303d…`, 1686 lines); no `set_option`.
  Beyond the associativity mismatch, the composition has a **`k`-scoping barrier**: the extracted body
  binds `k` before `r` and the scaling block consumes both, while `word_fixed_exp_reduction_value`'s
  post existentially quantifies `k` (`∃ k r, z = Ok r ∧ …`), so a continuation obtained from it does
  not receive `k` — the W1-T1V reassociation recipe is not merely non-definitional but unsound as
  stated. Named fix: a pair-returning chunk `word_fixed_exp_reduction_value_pair` (`z.1 = k`,
  `z.2 = r`, adapting the existing proof), reassociation of the flat body to the pair-binding form via
  `bind_assoc`/`bind_ok` (sound because the pair carries `k`), then `spec_bind` composition with
  Horner and scale. `W1-T1X` (running) lands it. No control (no new claim).
- `APEIRON-INT-TRANSCENDENTAL-W2-T1N` (deepseek-flash, ≈12 min of a 20/30 box — inside; no landing):
  the file is unchanged (`74022c67…`, 1521 lines); no `set_option`. The slice validated the strengthened
  `word_taylor_spec` post (`… ∧ |sin_t.val| ≤ 2^34 ∧ |cos_t.val| ≤ 2^34` from the chain's `hi13B`/`hp4B`
  and the `int_bmod_self_64` identities; `fixed_trig_taylor_value_proved` re-derived by a 7-way
  `rcases`) and the full e2e composition to the six selector cases, negation side conditions closing
  via `norm_num [IScalar.min, I64.min, I64.numBits]; omega`. The exact residual is the final `.val`
  equality per case: after `refine`, the post prints with `arg.val`/`swap`/`negate` already unfolded, so
  `simp_all` stalls (it does not orient `hswap`/`hneg` backwards); the directed chain is
  `simp only [← hr, ← harg, ← hswap, ← hneg, hsin, hcos, hnt, h1, sin_out1_post]`. `W2-T1O` (running)
  lands it.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1X` (deepseek-flash, ~28 min of a 20/30 box — at the hard stop; no
  landing, pair chunk close): the file is unchanged (`4949303d…`, 1686 lines); no `set_option`. The
  pair chunk must end `ok (core.result.Result.Ok (k, r))` — keeping the inner `Result` — or the guard
  branches type-clash; with that form, `step*` leaves the two-conjunct goal (`refine ⟨?_, ?_⟩`), closed
  by exposing the cast (`change (↑(IScalar.cast IScalarTy.I64 shifted) : ℤ) = …` or
  `rw [← IScalar.cast_val_eq]`), then `rw [hcast]`, `Int.shiftRight_eq_div_pow` and `norm_num`, with
  the residual bmod identity `(↑shifted).bmod 2^64 = ↑shifted`. Reassociation and composition then
  follow the W1-T1W plan. `W1-T1Y` (running) lands it.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1Y` (deepseek-flash, ≈21 min of a 20/30 box — soft stop reached; no
  landing, bullet 1 closed): the file is unchanged (`4949303d…`, 1686 lines); no `set_option`. The pair
  chunk's first conjunct closes (`hbmod` via `rw [← hcast]; rw [IScalar.cast_val_eq]; norm_num`, then
  `Int.shiftRight_eq_div_pow` on `hsv`); the second fails only on the `bmod`-numeral mismatch
  (`hb : … bmod (2^64) = …` vs the goal's `18446744073709551616`) — fix: `norm_num at hb` first, then
  `rw [hb, hkval]` and the same `hbmod` identity. Steps 2–3 unchanged. `W1-T1Z` (running) lands the
  pair chunk unconditionally and proceeds.
- `APEIRON-INT-TRANSCENDENTAL-W2-T1P` (deepseek-flash, ≈5 min of a 20/30 box — inside;
  **`word_fixed_sin_cos_value` closed**): `Numeric/RopeRefinement.lean` `c05c15f7…` (1682 lines) lands
  the strengthened `word_taylor_spec` post (`|sin_t|, |cos_t| ≤ 2^34`, 7-way `rcases`) and the
  end-to-end value spec of the post-guard body — reduction → fold (`|r| ≤ 2^34`) → `fold_arg_abs_le` →
  `word_taylor_spec` → quadrant swap/negate — with the exact integer output formulas
  (`sin_v.val = (if negate then -1 else 1) * (if swap.val = 1 then cos_poly arg.val else if swap.val =
  2 then -cos_poly arg.val else sin_poly arg.val)`, `cos_v` dual). Closing fixes: `swap.val = 2`
  consumes the two post goals `step*` produced with `_` witnesses and `simp only [← hsin, ← hcos];
  simp_all`; `swap.val = 0/1` use explicit neg witnesses and directed full-`simp` closers. Control
  non-slack (swap=1 output pair swapped) red; restore green; no `set_option`. Lead registration +
  clearance: the full `lake build AxiomAudit` is **green** (`efedc36e…`, 3894 jobs, closed cone),
  verifying `word_fixed_sin_cos_value` and the strengthened `word_taylor_spec`, and clearing the
  pending exp lines (`fixed_exp_scale_abs_raw`, `word_fixed_exp_scale`). **Trig's discrete value
  semantics is complete**; `E2a` (argument displacement + Lipschitz) opens the error decomposition.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1Z` (deepseek-flash, ~19 min of a 20/30 box — soft stop reached;
  **pair chunk landed**): `Numeric/ExpRefinement.lean` `4c3d1db5…` (1891 lines) adds
  `word_fixed_exp_reduction_value_pair` — the reduction ops verbatim, success
  `ok (core.result.Result.Ok (k, r))` (inner `Result` kept), post `∃ k r, z = Ok (k,r) ∧ k.val =
  (x.val·6196328019 + 2^63) >>> 64 ∧ r.val = x.val − k.val·2977044472` — with bullet 2 closed by
  `norm_num at hb` then `rw [r_post, hb, hkval]` and the bmod identity. Control non-slack
  (`· 2977044472` → `· 2977044473`) red; restore cached; no `set_option`. Steps 2–3 blockers found: the
  reassociation lemma is `Aeneas.Std.bind_assoc_eq` (`@[simp]`, `Aeneas/Std/Primitives.lean:287`), used
  as `rw [← Aeneas.Std.bind_assoc_eq]` at the `r` boundary via targeted `conv`; a whole-body `simp
  only [lift, bind_tc_ok]` re-triggers the `#i128` literal failure, so the reassociation must be
  targeted. `W1-T1AA` (running) lands the reassociation + composition. Lead registration: the pair
  chunk verified in the green `AxiomAudit.lean` `d1b2cc2e…` (3894 jobs, closed cone), which also
  re-verified the trig closure lines.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AA` (deepseek-flash, ≈11 min of a 20/30 box — inside; no landing,
  decisive structural finding): the file is unchanged (`4c3d1db5…`, 1891 lines); no `set_option`. The
  reassociation recipe is **unreachable by rewriting on the flat extracted body**: the chunk boundary
  is not a bind node, `bind_assoc_eq` needs a `(e >>= g) >>= h` subterm the flat term does not contain
  at the boundary, `change`/`conv` cannot introduce one (not definitional), and a whole-body `simp`
  re-triggers the `#i128` literal failure. `WP.spec_bind` and its variants require `m` to be the head;
  no prefix/chunk combinator exists. Two viable routes: (1) preferred — state the explicitly chunked
  term, compose the three chunk specs on it, prove `flat = chunked` as a plain `Result` equality via
  `simpa only [Aeneas.Std.bind_assoc_eq, bind_tc_ok]` (scoping the `#i128` issue), transfer;
  (2) a custom spec registration (`#register_spec_info`/`dspec`). `W1-T1AB` (running) takes route 1.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AB` (deepseek-flash, ≈11 min of a 20/30 box — inside; no landing,
  route 1 corrected): the file is unchanged (`4c3d1db5…`, 1891 lines); no `set_option`. Route 1 as
  stated is unsound: the pair chunk returns the **inner** `Result` (`Ok (k, r)`), so a chunked
  continuation must `match` it while the flat body binds `r` directly — the `flat = chunked` equality
  needs the guard branches resolved and the unwrap step, not `bind_assoc_eq`/`bind_tc_ok` alone. Two
  sound routes: (1) preferred — a **success-path-only pair chunk** (no inner guard `if`s; guard
  conditions as hypotheses) returning `ok (k, r)` at the outer level, so the flat body decomposes as
  guards-discharged (landed `fixed_exp_*_test_false` lemmas) → chunk → Horner → scale, equality from
  `bind_assoc_eq`/`bind_tc_ok` + guard-false lemmas; (2) a local `spec_bind_prefix` rule or
  `#register_spec_info`/`dspec` entry — the framework capability the wave has been missing. `W1-T1AC`
  (running) takes route 1.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AC` (deepseek-flash, ~22 min of a 20/30 box — soft stop reached; no
  landing): the file is unchanged (`4c3d1db5…`, 1891 lines); no `set_option`. The success-path-only
  pair chunk's body is well-typed and `step*` drives it to the same seven `i128`/`i64` side conditions
  as the guarded versions; porting the landed proofs fails because guard removal changes the `step*`
  context (three transplant attempts fail at the side-condition bullets). Smallest open goal: a
  **fresh** proof of `word_fixed_exp_reduction_pair_succ` (`simp only [lift, bind_tc_ok]; step*`, side
  conditions closed from `window x` via the `hsv`/`hsr` derivation and the range lemmas, then the
  validated pair post bullet), then the composition: resolve the flat body's four guard `if`s,
  reassociate to `(success chunk) >>= K`, `spec_bind`/`spec_mono` Horner/scale. `W1-T1AD` (running).
- `APEIRON-INT-TRANSCENDENTAL-E2A` (deepseek-flash, ≈10 min of a 20/30 box — inside; **landed**):
  `Numeric/RopeRefinement.lean` `f8227bf9…` (1755 lines) adds `trig_delta = 326/q32` (the displacement
  budget: `|k| ≤ 652` times the half-unit `TWO_PI_Q32` defect), `two_pi_q32_defect` (`|26986075409 −
  2π·q32| ≤ 1/2`, from `Real.pi_gt_d20`/`pi_lt_d20`; the actual defect is `0.044…`, so the half-unit
  is not tight), the exact identity `trig_reduction_displacement` (`|toReal r − (toReal x −
  k.val·2π)| ≤ trig_delta`), and the periodicity + Lipschitz transfers `trig_sin_reduction_transfer` /
  `trig_cos_reduction_transfer` (`|sin/cos (toReal r) − sin/cos (toReal x)| ≤ trig_delta`). Correction
  recorded: the true reduction is `x − k.val·2π` with `k` the number of full turns, not `k·π`. Gate
  exit 0 (3895 jobs; new imports `Trigonometric.Bounds`, `Real.Pi.Bounds`); control non-slack
  (`326/q32 → 200/q32`) red; no `set_option`. Lead registration: the five declarations verified in
  `AxiomAudit.lean` `5514d868…` (closed cone). `E2b` (running) proves the fold-constant displacement and
  the transfer to `arg`; the Taylor remainder and round ledger (`E2c`) and the composition follow.
- `APEIRON-INT-TRANSCENDENTAL-E2B` (deepseek-flash, ≈4 min of a 20/30 box — inside; partial): the file
  `be5c6a88…` (1790 lines) lands the three fold-constant defect lemmas — `pi_q32_defect`,
  `half_pi_q32_defect`, `quarter_pi_q32_defect`, each within a half Q32.32 unit from
  `Real.pi_gt_d20`/`pi_lt_d20` (true defects `0.478`, `−0.26`, `−0.13` units) — and
  `trig_fold_delta := 1/q32` (two fold levels, one half-unit each). The fold displacement and the
  sin/cos transfer did not close; two exact fixes named: wire the two-sided threshold facts
  (`6746518852/q32 < π/2 < 6746518853/q32`, `3373259426/q32 < π/4 < 3373259427/q32`) explicitly, and
  `norm_num`/`push_cast` before the `show … by ring` rewrite after `simp only [if_pos …]`. Control
  non-slack (`6746518852 → 6746518851`) red; no `set_option`. Lead registration: the four declarations
  appended (`56fec175…`), pending while the lane continues. `E2B2` (running) lands the displacements
  and the transfer.
- `APEIRON-INT-TRANSCENDENTAL-E2B2` (deepseek-flash, ≈20 min of a 20/30 box — soft stop; no landing):
  the file was restored to the E2b revision (`be5c6a88…`, 1790 lines); no `set_option`. The attempt had
  the full displacement shape (`fold1T`/`fold2T` with exact `π/2`/`π/4`; the integer thresholds
  coincide with the exact ones) but failed on `/q32`-form normalization with two exact edits named:
  replace `linarith`/`nlinarith` with direct order composition (`lt_of_le_of_lt`/`lt_of_lt_of_le`,
  `not_lt.mpr` forms), and make the negated-π form syntactically uniform with a separate `hr0`/`if_neg`
  before `if_pos hr`. **Environment recurrence**: `/tmp` hit 100% mid-slice and the live-file write
  failed with `ENOSPC` (file restored); the lead pruned ~32G of stale finished-slice trees (`w8-t1`
  through `w8-t5-control`, `w27target`, `w26rc-target`, `w05-target`, `w25-recheck-target`,
  `w20rc-target`, `w05-recheck-target`) — `/tmp` now 329M/32G. Per-slice pruning had not been done;
  recorded as a recurrence of the `AGENTS.md` §10 row. `E2B3` (running) lands the displacements.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AD` (deepseek-flash, ~34 min of a 20/30 box — overrun;
  **success-path chunk landed**): `Numeric/ExpRefinement.lean` `f83cc8e8…` (2029 lines) adds
  `word_fixed_exp_reduction_pair_succ` — the reduction ops without the four inner range guards,
  returning the pair at the outer level, post `z.1.val = (x.val·6196328019 + 2^63) >>> 64 ∧ z.2.val =
  x.val − z.1.val·2977044472`. Written fresh (`step*` + the seven side conditions from `window x` via
  `fixed_exp_shift_recip_bound_window`/`fixed_exp_reduction_remainder_range`); **no guard** — the fresh
  proof compiles with the `maxRecDepth 4000` line dropped. Control non-slack (`· 2977044472` →
  `· 2977044473`) red; restore cached.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AE` (deepseek-flash, ≈10 min of a 20/30 box — inside; no landing,
  decisive negative): the whole-body composition cannot be assembled from the chunk lemmas with the
  available WP rules: the guards are resolved by `step*` semantically (a top-level `rw` has no target),
  reassociation has no bind node at the chunk boundary, `step*` stops inside the Horner chunk (not at
  its head), and whole-body `simp` remains blocked by the `#i128` transparency failure.
  `spec_bind`/`spec_bind'`/`dspec_bind'` require the head; `qimp_spec` only reshapes the continuation.
  The named missing capability is a WP prefix/non-head composition rule (or `#register_spec_info`
  entry). `W1-T1AF` (running) instead transfers the sibling lane's proven technique —
  `word_fixed_sin_cos_value` composed the same reduction → stage → stage → output shape on a flat
  post-guard body — before a new framework rule is designed.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AF` (deepseek-flash, ≈12 min of a 20/30 box — inside; no landing,
  precise technique mismatch): the trig composition works because the **extracted trig body is
  natively chunked** — it contains `let res ← (do <reduction>); match res with | Err e => ok (Err e) |
  Ok r => <stages>` — so `word_fixed_sin_cos_value` composes by `spec_bind` on that subterm without any
  reassociation. The exp extraction is a **single flat do-block** (inline range-guard `if`s, no inner
  `Result`, no `match`), so no chunk is a head sub-computation and the trig technique cannot apply.
  Secondary gap: `word_fixed_exp_horner_acc`'s post does not expose `|acc6.val| ≤ 2^40`, which the
  scale consumes (another "post must expose what the continuation consumes" instance). Decision input
  for the lead: (a) restructure the Rust `fixed_exp` into the trig shape (named helpers or an
  inner-`Result`/`match` reduction), re-extract and re-anchor — an extraction-identity barrier; or
  (b) a local WP prefix-composition kit. No control (no new claim).
- **Decision (owner, option (a))**: `W1-T1AG` restructures `src/numeric/fixed_exp.rs` into the trig
  shape — `exp_reduce` (window + `k`/`r`, `Result<(i64,i64)>`), `exp_horner` (the seven-step
  accumulator), `exp_scale` (the branch scaling), and `fixed_exp` as the short chain of calls — so the
  extraction has call/bind boundaries; then `proof/regen.sh` (lane-locked) and `--check`, with the
  Rust gates. `W1-T1AH` re-anchors `Numeric/ExpRefinement.lean` to the new extraction (the landed
  stage lemmas keep their bodies; names/wrappers adapt), then the composition follows the trig
  technique and the exp error decomposition (`E1`) is the last piece.
- `APEIRON-INT-TRANSCENDENTAL-E2B4` (deepseek-flash, ≈6 min of a 20/30 box — inside; **landed with a
  correction**): `Numeric/RopeRefinement.lean` `ab948c35…` (1987 lines) lands the fold phase identities
  (`sin_fold1T`/`cos_fold1T`, `sin_fold2T`/`cos_fold2T`) and the phase-select lemmas
  (`sin_phase_select`/`cos_phase_select`) identifying the extracted `negate` flag with the `fold1T`
  jump factor and the `swap` selectors with the `fold2T` branches. **Correction**: the value-level
  `|toReal arg − fold2T (fold1T θ)| ≤ trig_delta + trig_fold_delta` is false across a first-fold
  threshold — the jump is compensated by the flag, not in the folded value; the correct chain is the
  same-`r` fold displacement plus the phase selects. Control non-slack (phase sign) red; no
  `set_option`. Lead registration: the six declarations verified in `AxiomAudit.lean` `672238f2…`
  (closed cone). `E2B5` (running) lands the pre-Taylor flagged-output bounds; then the Taylor remainder
  and round ledger (`E2c`) closes `fixed_sin_cos_refines`.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AG` (deepseek-flash + lead, ~30 min of the authorized 30/45 box —
  Rust landed; regen blocked, then fixed by the lead): `src/numeric/fixed_exp.rs` `e3dc208a…` (266
  lines) factors `exp_reduce` (`Result<(i64,i64)>`: window + `k`/`r` + narrowing refusals verbatim),
  `exp_horner` (`r → i128`, the seven steps verbatim) and `exp_scale` (`(k, acc) → Result<i64>`), with
  `fixed_exp` the short chain (explicit `match`; no `?`). Rust gates: fmt 0, clippy 0, test 0 (44 s;
  all `numeric::fixed_exp` tests pass). Control: `852_176 → 700_000` red with an attributable accuracy
  diagnostic (the ±1 mutation is inside tolerance and was reported as slack); restore green.
  **Environment defect and fix**: `proof/regen.sh` first failed exit 2 with `E0514` — Charon's driver
  (pinned rustc `8fa1c96cf`) resolved `std` from the **miri cache** (`~/.cache/miri/…`, std built by
  the default nightly `923c95cdf` 2026-09-16); `RUSTUP_TOOLCHAIN=nightly-2026-08-18` did not change it;
  the lead parked `~/.cache/miri` → `~/.cache/miri.bak-lead` (regenerable) and the regen then exited 0.
  Regenerated `proof/ApeironWord.lean` `328655cd…` (11273 lines; 1708 jobs); `regen.sh --check` exit 0.
  The emitted `fixed_exp.fixed_exp` now has the trig shape (`let r ← exp_reduce x; match r with | Ok kr
  => … | Err e => …`), so `spec_bind` composes at the call boundaries. `W1-T1AH` (running) re-anchors
  `Numeric/ExpRefinement.lean` and composes `word_fixed_exp_body`.
- `APEIRON-INT-TRANSCENDENTAL-E2B5` (deepseek-flash, ≈8.6 min of a 20/30 box — inside; **landed**):
  `Numeric/RopeRefinement.lean` `a0f04645…` (2318 lines) defines the flagged output combinations
  (`flagSin`/`flagCos`) and proves the fold bounds (`flag_sin_fold_bound`/`flag_cos_fold_bound`,
  `≤ trig_fold_delta`) and the reduction bounds (`flag_sin_reduction_bound`/`flag_cos_reduction_bound`,
  `≤ trig_delta + trig_fold_delta = 327/q32 ≈ 7.61e-8` on `trig_window x`). Mechanism: the fold1 jump
  is absorbed by the `negate` factor (matched via `half_pi_q32_floor`), the fold2 threshold by the
  `swap` case split; no false value-level `arg` displacement is used. Module gate exit 0 (2500 jobs);
  the combined gate fails only in the concurrently re-anchored `Numeric/ExpRefinement.lean`
  (authorized fallback). Control non-slack (`swap = 2` sign) red; no `set_option`. Lead registration:
  the six declarations appended (`a7fba17e…`), pending the exp re-anchor. `E2C` (running) proves the
  Taylor remainder + round ledger and closes `fixed_sin_cos_refines` — the last trig piece.
- `APEIRON-INT-TRANSCENDENTAL-E2C` (deepseek-flash, ≈6–10 min of a 25/40 box — inside; no landing,
  residual fully derived): `Numeric/RopeRefinement.lean` `b03bc862…` (2329 lines) lands
  `fixed_trig.fold_arg_abs_le_quarter` (`|fold2 (fold1 r)| ≤ 3373259426 = floor(π/4·2^32)`, so
  `|toReal arg| < π/4`) — the hypothesis that makes the degree-7 remainder fit the `2^-20` floor (the
  old `≤ 2^32` bound leaves `1/9! > 2^-20(|sin|+1)`). The single-rounding ledger
  `|round_q32 v − v/2^32| ≤ 1/2` was attempted twice and reverted on the ℤ→ℝ `2^32` cast
  normalization; the chain ledgers are derived exactly (sin/cos `7δ`, Horner-vs-exact `≤ 1/q32` each,
  `δ = 1/(2q32)`), and Mathlib has only degree-3 `Real.sin_bound`/`cos_bound`, so the degree-7/8
  remainder needs its own Taylor-with-remainder. Totals: sin `≈ 3.9e-7`, cos `≈ 1.0e-7`, both
  `≤ trig_floor = 2^-20 ≈ 9.54e-7` uniformly (no Jordan split). Module gate exit 0 (2500 jobs); control
  non-slack red; no `set_option`. Lead registration: `fold_arg_abs_le_quarter` appended (`016f2360…`),
  pending; **the regenerated extraction retired `word_fixed_exp_window_past_guards`, which
  `AxiomAudit.lean` still audits (line 839)** — the exp re-anchor's return names the replacements and
  the lead updates the audit. `E2C2` (running) lands `round_q32_half` and the chain ledgers; then the
  Taylor remainders, then the composition.
- `APEIRON-INT-TRANSCENDENTAL-E2C2` (deepseek-flash, ≈6 min of a 25/40 box — inside; partial): the file
  `0fd55970…` (2411 lines) lands the single-rounding ledger in both forms — `round_q32_half`
  (`|round_q32 v − v/2^32| ≤ 1/2`, with the `↑(2^32)` cast pitfall solved by rewriting to the numeral
  `4294967296` and `Int.cast_neg` in the negative branch, then `mul_le_mul_iff_of_pos_right` and
  `nlinarith`) and `round_q32_ledger` (`|round_q32 v / q32 − v/q64| ≤ δ = 1/(2q32)`). Module gate exit 0
  (2500 jobs); control non-slack (`1/2 → 1/4`) red; no `set_option`. The Horner chain ledgers did not
  land; the exact structure is derived (`sinHorner`/`cosHorner`, stage errors `2δ, 4δ, 6δ, 7δ`, the
  coefficient-defect bounds `≤ 1/q32`, numeric facts `by norm_num [q32]`) with no new mathematics — the
  `abs_sub_le`/`mul_le_mul` induction. Lead registration: the two ledgers appended (`e811612c…`),
  pending. `E2C3` (running) lands the chain ledgers; then the Taylor remainders, then the composition.
- `APEIRON-INT-TRANSCENDENTAL-E2C3` (deepseek-flash, ≈20 min of a 25/40 box — inside; **landed**): the
  file `604366f6…` (2773 lines) defines `sinHorner`/`cosHorner` and proves the chain ledgers
  (`sin_poly_horner ≤ 15/(2q32)`, `cos_poly_horner ≤ 40/(2q32)`, on `|t.val| ≤ 2^32`). The constants
  are coarser than the requested `7δ` (a `|U| ≤ 2` slack avoids a `D²` term) but `15δ ≈ 1.75e-9` and
  `40δ ≈ 4.7e-9` are ≪ the ledger budget, so the end-to-end bound is unaffected; the `7δ` tightening
  is a constant-only refinement. Module exit 0 (2500 jobs); control non-slack (coefficient `−852176 →
  −852177`) red; no `set_option`. Lead registration: the four declarations verified in `AxiomAudit.lean`
  `141e3dfe…` (closed cone). `E2C4` (running) lands the coefficient-defect bounds and the degree-7/8
  Taylor remainders; then the composition (`E2C5`) closes `fixed_sin_cos_refines`.
- `APEIRON-INT-TRANSCENDENTAL-E2C4` (deepseek-flash, ≈4–5 min of a 25/40 box — inside; **landed**): the
  file `e07bbd74…` (2932 lines) defines `sinTaylor`/`cosTaylor` and proves the coefficient defects
  (`sinHorner_taylor`/`cosHorner_taylor ≤ 1/q32`, from the exact defects `−1/(3q32)`, `−2/(15q32)`,
  `16/(315q32)`, `1/(3q32)`, `16/(45q32)`, `−2/(315q32)`) and the remainders (`sinTaylor_remainder ≤
  |s|^9·(10/3265920)`, `cosTaylor_remainder ≤ |s|^10·(11/36288000)`, on `|s| ≤ 1`). **Route**:
  `Complex.exp_bound` at `Complex.exp (s·I)` with `n = 9/10` — not the degree-3 `Real.sin_bound` —
  with the finite-sum `im`/`re` identities by `norm_num`/`simp`/`ring` (no `decide`); the
  `n.succ·(n!·n)⁻¹` factor makes the constants `(10/9)`/`(11/10)` times `1/n!`, still `≈ 3.5e-7`/
  `2.7e-8` at `|s| ≤ π/4`, inside the floor. Module exit 0 (2500 jobs; new import
  `Mathlib.Analysis.Complex.Exponential`); control non-slack (`10/3265920 → 1/3265920`) red; no
  `set_option`. Lead registration: the six declarations appended (`75038da7…`), pending. Only `E2C5`
  (the composition) remains to close `fixed_sin_cos_refines`.
- `APEIRON-INT-TRANSCENDENTAL-E2C5` (deepseek-flash, ≈8–9 min of a 25/40 box — inside; no landing,
  one-step residual): the file is untouched at the E2C4 revision (`e07bbd74…`, 2932 lines); the probe
  (`/home/socol/oc-tmp/e2c5_probe.lean`) compiles every composition lemma — the poly-close bounds,
  `cos_rem_le`, `close_floor` (`(trig_delta + trig_fold_delta) + 40/(2q32) + 1/q32 +
  |s|^9·(10/3265920) ≤ trig_floor`; the verifier-corrected worst case is 4.92e-7 with `|s| ≤ 4/5`,
  4.29e-7 at `|s| = π/4` — the probe's 5.8e-7 over-estimated), the `negate`-factor split and the
  flagged-output bounds — and both `calc` blocks type-check. The single residual is the first conjunct:
  `word_fixed_sin_cos x` does not reduce definitionally to the do-block transcribed in
  `word_fixed_sin_cos_value` (neither `exact hy.trans …` nor `simpa [word_fixed_sin_cos,
  fixed_trig.fixed_sin_cos]` unifies). Fix: state the value spec on `word_fixed_sin_cos x` directly
  (as `word_taylor_spec` does on `fixed_trig.taylor_sin_cos`) or add the bridge equality; no new
  mathematics. `E2C6` (running) lands it.
- `APEIRON-INT-TRANSCENDENTAL-E2C6` (deepseek-flash, ≈5 min of a 25/40 box — inside; no landing,
  blocker exact): the file is untouched at the E2C4 revision (`e07bbd74…`, 2932 lines). The bridge
  fails because `word_fixed_sin_cos_window_past_guards` (P: the extracted inlined post-guard body) and
  `word_fixed_sin_cos_value` (Q: the transcription with `let res ← <reduction>; match res with | Err e
  => ok (Err e) | Ok r => …`) are not definitionally equal — Q's wrapper is the identity only because
  the reduction returns `ok (Ok r)` on the window, a semantic fact; every `simpa`/`rw`/`exact
  hy.trans …` variant fails on the wrapper. Two bounded fixes: (a) restate the value spec and its
  reduction base on P and replay the composition; (b) derive the reduction's value equality
  (`R = ok (Ok r)` on the window) and rewrite the wrapper away, then reuse the probe's `e2c5_main` as
  written. `E2C7` (running) takes (b) with (a) as fallback.
- `APEIRON-INT-TRANSCENDENTAL-E2C7` (deepseek-flash, ≈2–3 min of commands of a 25/40 box — inside; no
  landing, three-attempt stop): the smallest unresolved goal is the program equality `P = Q`
  (`word_fixed_sin_cos x = <E2B transcription>`); after the available `bind_assoc`/`bind_tc_ok`/
  `fold_step` and tuple/projection normalisations, the only surviving mismatch is the reduction wrapper
  (`let res ← R; match res with | Err e => ok (Err e) | Ok r => …` vs P's inlined continuation), whose
  collapse is the semantic fact that `R` returns `ok (Ok r)` on the window. Recommended and dispatched:
  fix (a) — restate `word_fixed_sin_cos_value` and its `…_reduction_fold_value` base directly on P and
  replay the `spec_bind` composition (`E2C8`, running).
- `APEIRON-INT-TRANSCENDENTAL-E2C8` (deepseek-flash, ≈3–4 min of commands of a 25/40 box — inside; no
  landing, encoding pinned): `simp only [bind_assoc, fixed_trig.fold_step] at hy` **does** eliminate
  the reduction wrapper (the earlier belief it did not is updated); the only survivor is the
  **tuple-binding style** — Q projects over the `fold_step` 5-tuple (`u.2.2.2.1` …) and the
  `I64 × I64` results (`sc.1/sc.2`, `p.1/p.2`), while P uses named tuple-lets `let (s,negate) ← …`.
  `bind_tc_ok` does not fire on `(do …; ok tuple) >>= f`; `dsimp only` makes no progress and hits the
  `#i32` transparency failure; full `simp` leaves the mismatch. Recommended: tuple-elimination `@[simp]`
  equations (bind-push + `ok` bind + literal-tuple projection reduction) for those nestings, or
  restating the value theorem on P's named-let encoding. `E2C9` (running) tries the lemmas first,
  restatement as fallback.
- `APEIRON-INT-TRANSCENDENTAL-E2C9` (deepseek-flash, ≈3 min of commands of a 25/40 box — inside; no
  landing, gap sharpened): the required equality is a **monadic bind-push through `let`-headed
  do-blocks**; `simp only [bind_assoc]` exposes the structure, but `bind_tc_ok` does not fire on a
  `let`-headed bound term, `dsimp only` is unusable (`#i32` transparency), and the pair-bind lemma
  fails higher-order matching for concrete continuations. Recommended: prove the missing lemmas by
  `cases`/structural reasoning with the continuation as a variable (or restate on P). `E2C10` (running)
  builds the local `Result` normalization kit (bind-let push, tuple collapse with variable
  continuations) and closes the headline with the probe's analytic composition.
- `APEIRON-INT-TRANSCENDENTAL-E2C10` (deepseek-flash, ≈3 min of commands of a 25/40 box — inside; no
  landing, kit partially built): `c10_let_push`, `c10_let_prod_push`, `c10_pair_bind` are proved
  standalone, but the fold-collapse lemma does not close — the pattern-let bind inside a monadic
  continuation survives (`c10_let_prod_push` matches nothing; `funext`/`cases` leaves the redundant
  redex `let (s,negate) := (s,negate); …` with `dsimp` excluded by the `#i32` transparency failure).
  Recommended and dispatched: prove the P-value spec directly, bypassing the equality — re-run the
  reduction prefix's `step*` chain with the fold/taylor/quadrant continuation appended (the
  `word_fixed_sin_cos_reduction_value` tactic, lines 515–554), then the probe's analytic composition
  (`E2C11`, running).
- `APEIRON-INT-TRANSCENDENTAL-E2C11` (deepseek-flash, ≈5 min of commands of a 25/40 box — inside; no
  landing, route scoped): the direct-P `step*` replay (`e2c11_probe.lean`) walks the inlined reduction
  and stalls on its overflow/guard block (identical to `word_fixed_sin_cos_reduction_value`'s, lines
  556–653), then descends into the inlined fold/Taylor because P's continuation is flat; finishing it
  means (i) `SinCos` ascriptions, (ii) replaying that reduction case block, (iii) closing the inlined
  fold per branch with the fold defs, (iv) inserting `spec_bind (word_taylor_spec …)` before
  `taylor_sin_cos`, (v) assembling the post — ~150 lines of fragile `step*`/case work. The kit lemmas
  (`c10_let_push`, `c10_let_prod_push`, `c10_pair_bind`) are proven but do not fire on the P↔Q term.
  `E2C12` (running) executes (i)–(v); if it stalls, the fallback is the exp-style trig restructure
  (helper `trig_reduce`/`trig_fold` plus regen), decided by the lead.
- `APEIRON-INT-TRANSCENDENTAL-E2C12` (deepseek-flash, ≈8 min of commands of a 25/40 box — inside; no
  landing, reduction closed in the probe): `e2c12_probe.lean`'s direct-P proof closes every
  reduction/overflow/guard goal (via the E2B reduction case block); two delimited blockers remain in
  the success path — (1) a signed I128→I64 cast lemma (`(IScalar.cast .I64 r_wide).val = r_wide.val`
  from the guard range; `int_bmod_self_64` is for the ℤ form and `r_wide` is signed), (2) fold-spec
  unification after `step*` inlines the cast (`↑HALF_PI_Q32 < (↑r_wide).bmod 2^64` vs the spec's
  `IScalar.cast .I64 r_wide > HALF`) — fix by closing the fold branch-by-branch with the fold defs
  (`word_fixed_sin_cos_fold_value` style) or by naming `r` before `step*`. `E2C13` (running) lands the
  cast lemma and the fold closure, then the Taylor/quadrant assembly (E2B lines 1740–1774) and the
  headline.
- `APEIRON-INT-TRANSCENDENTAL-E2C13` (deepseek-flash, ≈5 min of commands of a 25/40 box — inside; no
  landing, final post assembly): the direct-P proof now closes everything (reduction, signed cast via
  `IScalar.cast_val_eq` + `int_bmod_self_64`, `hcastmod`, both fold specs, `word_taylor_spec`, branch
  casing); the single stall is the tuple-let redex
  `(let (sin_t,cos_t) := (sin_t,cos_t); do … ok (Ok {…})) ⦃ post ⦄`, where `WP.spec_ok` finds no `ok
  ?x` and `simp_all` exceeds `maxRecDepth`; branch conditions are `swap = 1#i32`/`2#i32`, so the E2B
  assembly does not port verbatim. Fix: a local `(let (a,b) := (a,b); K a b) = K a b` lemma or
  `change`, then `IScalar.eq_equiv` bridging and the `refine` witness list. **Escalated to
  `xiaomi-pro`** (`E2C14`) per §9.1 — nine slices of `Result`-monad normalization on the speed lane;
  the Lean kernel arbitrates.
- `APEIRON-INT-TRANSCENDENTAL-E2C14` (xiaomi-pro escalation per §9.1, ~45 min of the resumed 40/60 box
  — overrun; **closed**): `Numeric/RopeRefinement.lean` `3bd26da3…` (3524 lines, +592) proves
  `fixed_sin_cos_refines_proved : fixed_sin_cos_refines` — per output `|toReal sc.f − f (toReal x)| ≤
  epsExp·|f (toReal x)| + trig_floor`, `epsExp = trig_floor = 2^-20`, composed worst case 4.92e-7 (4.29e-7 at `|s| = π/4`) ≤
  `trig_floor` = 9.537e-7 (`trig_floor_close`) — with the direct-P value proof `word_fixed_sin_cos_past_guards_value`
  landing the post assembly. Two defects, not one: the goal's post normalises mid-proof to the
  nested-if form (15-slot `refine` with `hsw`/`hfn` bridges `(swap:ℤ) = 1`,
  `fold1_negate ↑(cast r_wide) = true`), and the tuple-let redex needed no lemma — it falls to `step*`
  once the **outer** `if swap = 1#i32` is resolved (`simp only [if_neg h1, …]`); `WP.spec_ok`/`refine`
  then fire directly. Kit (`c10_let_push`/`c10_let_prod_push`/`c10_pair_bind`) retained; no `dsimp
  only`/`simp_all` in the final proof; zero `set_option`. Gate exit 0 (2500 jobs, 13.6 s); `AxiomAudit`
  green (3896 jobs); control non-slack (`26986075409 → …410` in the direct-P post's
  `r.val = x.val − k.val·…`) red with attributable diagnostics in the direct-P refine and the
  headline; restore byte-identical. Lead registration: the twelve E2C14 declarations appended; audit
  `a235b779…` (closed cone). **Trig's real bound is proved.**
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AJ` (deepseek-flash, ~54 min of the authorized 30/45 box — overrun;
  **`word_fixed_exp_body` landed**): `Numeric/ExpRefinement.lean` `448abb53…` (2037 lines) adds the
  tight remainder bound (`fixed_exp_reduction_remainder_abs_le`, `|x − s·2977044472| ≤ 2^32`), the
  `|horner_acc| ≤ 2^40` bound as a second conjunct of the `exp_horner` spec (cheaper than a pure lemma;
  from `|i13| ≤ 2^36`), and `word_fixed_exp_body` by the trig call-chain technique (`spec_bind` on
  `fixed_exp_exp_reduce_spec` → `word_fixed_exp_horner_acc` → `spec_mono (fixed_exp_exp_scale_spec …)`
  with the `Ok 0` flush witness and the `Err` branch), the post naming the exact whole-body value. Gate
  exit 0 (3895 jobs, 166 s; module 155 s); cap retained single (8M on `word_fixed_exp_horner_acc`);
  control non-slack (tight bound `2^32 → 2^31`) red; restore cached. **Exp's discrete value semantics
  is complete**; only the real-error composition into `fixed_exp_refines` remains (`E1a`, running).
- `APEIRON-INT-TRANSCENDENTAL-E1A` (deepseek-flash, ~35 min of a 25/40 box — soft stop slightly
  exceeded; **landed**): `Numeric/ExpRefinement.lean` `754295f1…` (2103 lines; two Mathlib imports)
  proves `fixed_exp_ln2_q32_defect` (`|2977044472/q32 − log 2| ≤ 10^-9`, from
  `Real.log_two_gt_d9`/`log_two_lt_d9`), `fixed_exp_reduction_displacement` (`|toReal r − (toReal x −
  k.val·log 2)| ≤ deltaRed = 31/10^9`; `|k| ≤ 31` from the shift bound), and
  `fixed_exp_exponential_transfer` (`|exp (toReal r + k.val·log 2) − exp (toReal x)| ≤ exp(toReal x)·
  (2·deltaRed)`, via `Real.abs_exp_sub_one_le`; `2·deltaRed ≈ 6.2e-8 ≪ 2^-20`). Module gate exit 0
  (2445 jobs, 160 s); control non-slack (`deltaRed 31/10^9 → 1/10^9`) red; no new `set_option`. Lead
  registration: the four declarations verified in `AxiomAudit.lean` `9eb66152…` (closed cone). `E1b`
  (running) proves the Horner polynomial error (degree-7 remainder + round ledger + coefficient
  defects); then `E1c` composes into `fixed_exp_refines`.
- **Reboot and reconciliation**: the machine rebooted mid-slice; the trig escalation lane (`E2C14`) and
  the exp `E1B` lane died. On resume: `Numeric/ExpRefinement.lean` was `b38375df…` (2160 lines) — the
  E1B pieces (`epsHorner = 2·10^-7`, `fixed_exp_taylor7_remainder`, eight `fixed_exp_coeff_defect_*`)
  had landed and build green (2445 jobs) but no gate/control ran; `Numeric/RopeRefinement.lean`
  untouched (`e07bbd74…`, 2932 lines); `AxiomAudit.lean` re-verified after registering the E1B pieces
  (`348b1c63…`, closed cone); probes survive under `/home/socol/oc-tmp/` (`e2c11_probe.lean`,
  `e2c12_probe.lean`, `e2c5_probe.lean`); `/tmp` was wiped (logs gone) and recreated. Both lanes
  resumed (`E1B` finish the Horner error; `E2C14` the trig post assembly), with the pristine E1a
  revision preserved at `/home/socol/oc-tmp/int-exp-e1a-control/ExpRefinementControl.lean`
  (`754295f1…`).
- `APEIRON-INT-TRANSCENDENTAL-E1B` (deepseek-flash, ~33 min of a 30/45 box — soft stop exceeded; round
  ledger landed): `Numeric/ExpRefinement.lean` `04f6aeae…` (2200 lines) adds
  `fixed_exp_round_q32_int_error` (`|round_q32 v · 2^32 − v| ≤ 2^31`) and
  `fixed_exp_round_q32_real_error` (`|round_q32 v/q32 − v/q32²| ≤ 1/(2q32)`), on top of the truncation
  remainder (`≤ |t|^8·9/322560`) and the eight coefficient defects. Gate exit 0 (2445 jobs, 178 s);
  control non-slack (`1/(2q32) → 1/(4q32)`) red; no new `set_option`. Residual: assemble the Horner
  error (`hornerReal`, round propagation `≤ 1/q32`, defects `≤ 8/q32`, total `≤ 9/q32 + 2^-8·9/322560 ≈
  1.11e-7 ≤ epsHorner = 2e-7`), then E1c. Lead registration: the two round lemmas verified in
  `AxiomAudit.lean` `0ecd8162…` (closed cone). `E1B2` (running) lands the assembly.
- `APEIRON-INT-TRANSCENDENTAL-E1B2` (deepseek-flash, ~44 min of a 30/45 box — at the hard stop; round
  propagation landed): `Numeric/ExpRefinement.lean` `c20fb4a0…` (2292 lines) adds the exact Horner
  machinery (`hand`, `handReal`, `handRealF`, `horner_acc_eq`, `hornerReal`) and proves the round
  propagation `horner_round_error` (`|(horner_acc r)/q32 − hornerReal t| ≤ 7·(1/(2q32))` for `|t| ≤ 1`)
  from `hand_bound` + `fixed_exp_round_q32_real_error`. Gate exit 0 (2445 jobs, 179 s); control
  non-slack (`7·(1/(2q32)) → 5·(1/(2q32))`) red; no new `set_option`. Residual: the coefficient-defect
  assembly (`taylorReal`, the real stage lemma, total `≤ 8/q32`), the combined Horner error (`≤
  7/(2q32) + 8/q32 + 2^-8·9/322560 ≈ 1.12e-7 ≤ epsHorner = 2e-7`), then E1c. Lead registration: the
  seven declarations verified in `AxiomAudit.lean` `376a6904…` (closed cone). `E1B3` (running).
- `APEIRON-INT-TRANSCENDENTAL-E1B3` (deepseek-flash, ~35 min of a 30/45 box — soft stop exceeded;
  coefficient machinery landed, chain blocked): `Numeric/ExpRefinement.lean` `d404cdc7…` (2347 lines)
  adds `hornerRealF`, `taylorReal`, `hornerReal_eq_F`, `taylorReal_eq_sum` and the real defect stage
  `handRealF_defect`. The defect chain stalls at stages `e5`/`e6` on the default 200k heartbeats
  (`isDefEq`/`whnf` on the deep `handRealF` nesting) — a structural fix, not a cap: reformulate over an
  explicit coefficient list with a recursive `realHorner` and prove the bound by `List` induction
  (small terms per step), connecting by `rfl`. Gate exit 0 (2445 jobs, 174 s); control non-slack
  (`coeff_defect_6 ≤ 1/q32 → ≤ 0`) red; no new `set_option`. Lead registration: the five declarations
  verified in `AxiomAudit.lean` `10892edb…` (closed cone). `E1B4` (running) reformulates and
  assembles; then E1c.
- `APEIRON-INT-TRANSCENDENTAL-E1B4` (deepseek-flash, ~30 min of a 30/45 box — soft stop reached;
  **E1B complete**): `Numeric/ExpRefinement.lean` `7be36e01…` (2467 lines) reformulates the real Horner
  over a coefficient list (`realHorner`, `hornerReal_eq_list`/`taylorReal_eq_list`) and proves the chain
  by `List` induction (`realHorner_step`, `realHorner_pair_le`, `hornerReal_taylor_error ≤ 8/q32`) and
  the combined `horner_acc_error ≤ epsHorner = 2·10^-7` on `|t| ≤ 1/2` — the deep-nesting heartbeat
  wall is gone at default heartbeats. Gate exit 0 (2445 jobs, 168 s); `AxiomAudit` green (3896 jobs);
  control non-slack (`epsHorner 2e-7 → 1e-7`) red; restore byte-identical. Lead registration: the
  E1B4 declarations verified in `AxiomAudit.lean` `eec20a18…` (closed cone). Residual: `E1c` —
  `fixed_exp_refines` via `|r| ≤ 2^31`, the Horner error, the **scale refinement** (`|toReal v −
  2^k·(acc/q32)| ≤ 4/q32` both branches), and the composition with `fixed_exp_exponential_transfer`;
  the relative budget closes at `≈ 3.30e-7 < 2^-20`. `E1C` (running).
- `APEIRON-INT-TRANSCENDENTAL-E1C` (deepseek-flash, ~45 min of a 30/45 box — at the hard stop; steps
  1–2 landed, two spec gaps exposed): `Numeric/ExpRefinement.lean` `6bc9a9ed…` (2526 lines) adds the
  sharpened `fixed_exp_reduction_remainder_abs_le_2p31` (`|x − s·2977044472| ≤ 2^31`, from
  `k ∈ [-24,30]` and `R·D = 2^64 + 1953109352`) and `fixed_exp_reduction_remainder_toReal_half`
  (`|toReal r| ≤ 1/2`, the hypothesis `horner_acc_error` needs — the older `≤ 2^32` bound only gives
  `≤ 1`, where the degree-7 remainder `2.79e-5` blows the budget). Gate exit 0 (2445 jobs, 172 s);
  control non-slack (`2^31 → 2^30`) red; restore byte-identical; no new `set_option`. **Two gaps
  exposed**: (A) `I64.val` is unbounded in this Aeneas model (no `IScalar.val_range`; `omega` refutes
  `v.val ≤ 2^63−1`), so `exp_scale`'s post drops the range and `bmod` can wrap — step 3 needs a new WP
  theorem against the body carrying `v.val = acc.val·2^k` (and the `k < 0` rounding form); (B)
  `word_fixed_exp_body` admits `Err` on the window, so the headline also needs the `Err` branches
  excluded (`|scaled| ≤ I64.max` from `exp (toReal x) ≤ exp 21` and `acc/q32 ≥ e^{−1/2} − epsHorner`).
  Lead registration: the two sharpened lemmas verified in `AxiomAudit.lean` `e8f68776…` (closed cone).
  `E1C2` (running) closes Gap A; then E1C3 composes.
- `APEIRON-INT-TRANSCENDENTAL-E1C2` (deepseek-flash, ~56 min of a 30/45 box — overrun; **Gap A
  discharged**): `Numeric/ExpRefinement.lean` `8f8031b3…` (2758 lines) proves the body-level exact
  values for both `exp_scale` branches under explicit range hypotheses, excluding `Err`:
  `fixed_exp_bmod128_low` (`a.bmod 2^128 = a` on `[0, 2^63−1]`), `fixed_exp_scale_nonneg_exact`
  (k≥0: `v.val = acc·2^k`), `fixed_exp_two_pow_bmod`, `fixed_exp_scale_neg_core_exact` (the `n<64`
  core), `fixed_exp_scale_neg_exact` (k<0: the rounded shift, plus the `natAbs ≥ 64` flush). Gate exit
  0 (2445 jobs, 158 s); control non-slack (`v.val = acc·2^k → …+1`) red at `:2611`; restore
  byte-identical; no new `set_option`. Process note: one broken append produced a doc-comment splice;
  the lane restored the pristine `6bc9a9ed…` copy and re-appended — no broken tree left. Lead
  registration: the five declarations verified in `AxiomAudit.lean` `3a3192f1…` (closed cone).
  Residual (`E1C3`, running): the on-window range budget for the hypotheses, then the composition
  into `fixed_exp_refines` (relative budget `≈ 3.3e-7 < 2^-20`).
- `APEIRON-INT-TRANSCENDENTAL-E1C3` (deepseek-flash, ~43 min of a 35/50 box — soft stop exceeded;
  step-1 checkpoint landed): `Numeric/ExpRefinement.lean` `47938976…` (2848 lines) adds
  `fixed_exp_exp_half_lt` (`exp (1/2) < 7/4` via `Real.exp_bound` at `n = 5`) and
  `fixed_exp_scale_range_budget` — from the window plus the body characterization: `k.val < 64` (the
  `k ≥ 64` refusal never arises on the window), `0 < acc.val`, and the exact range budgets
  `acc.val·2^{k.val} ≤ 2^63−1` (k≥0) / `−2^63 ≤ (acc + 2^{|k|−1}) >>> |k| ≤ 2^63−1` (k<0) that the
  exact scale theorems need. Gate exit 0 (2445 jobs, 158 s); control non-slack (`≤ 2^63−1 →
  ≤ 2^63−2`) red at `:2826`; restore byte-identical; no new `set_option`. Lead registration: the two
  declarations verified in `AxiomAudit.lean` `249fdd0f…` (closed cone). Residual (`E1C4`, running):
  strengthen the `exp_reduce` spec to Ok-only, prove `word_fixed_exp_body_refines` by `spec_bind`
  over it, `word_fixed_exp_horner_acc` and both exact scale theorems (k≥0 exact, k<0 `≤ 1/(2q32)`;
  relative closure `3.30e-7 < 2^-20`; rounding under the `4/q32` floor), then close
  `fixed_exp_refines` via `spec_imp_exists` and `word_fixed_exp =ᵈ fixed_exp.fixed_exp` (refusal
  branch already in `fixed_exp_refuses_proved`).
- `APEIRON-INT-TRANSCENDENTAL-E1C4` (deepseek-flash, ~62 min of a 35/50 box — hard stop exceeded;
  **exp closed**): `Numeric/ExpRefinement.lean` `33c1efc3…` (3160 lines) lands the composition —
  `fixed_exp_exp_reduce_spec_ok` (Ok-only reduce), `fixed_exp_exp_u_bounds`,
  `fixed_exp_scale_abs_rel_pos`/`_neg` (relative budget), `fixed_exp_ediv_self_abs`,
  `fixed_exp_round_neg_abs`, `word_fixed_exp_body_refines` (body WP triple), and
  `fixed_exp_refines_proved : fixed_exp_refines` — `|toReal v − exp (toReal x)| ≤ epsExp·exp (toReal x)
  + 4/q32` with `epsExp = 2^-20`. k≥0 exact, k<0 round-to-nearest absorbed by the `4/q32` floor;
  total ≈ 4.6e-7. Gate exit 0 (2445 jobs, 154 s); `AxiomAudit` green; control non-slack (`4/q32 →
  1/q32`) red at the k<0 step and the headline; restore byte-identical; no new `set_option`. Lead
  registration: the eight declarations verified in `AxiomAudit.lean` `99c83c9e…` (closed cone).
  Overrun note: ℝ/ℤ/natAbs coercion iteration, not a proof failure.
- `APEIRON-INT-TRANSCENDENTAL-E3B` (deepseek-flash, ~10 min of a 30/45 box — inside; **both rsqrt
  guards removed**): `Numeric/RsqrtRefinement.lean` `916ca0ef…` (1072 lines) now elaborates
  `isqrt_loop1_spec` and `isqrt_spec` at the default 200k heartbeat cap — 22 `nlinarith`/`linarith`
  sites in `isqrt_loop1_spec` restricted to `… only [hyps]` (two needed the explicit set), guard-free
  red run exit 1 with the deterministic `isDefEq` timeout at `:592` before the change. Gate exit 0
  (2394 jobs); full audit green, same names and cone; control non-slack (`(2^117)^2 → (2^118)^2`) red
  with pristine-copy positive exit 0; restore byte-identical; no `set_option` remains. The two rsqrt
  openings are discharged.
- `APEIRON-INT-TRANSCENDENTAL-E3A` (deepseek-flash, ~28 min of a 30/45 box — at the soft stop; three
  of four exp guards removed): `Numeric/ExpRefinement.lean` `6c436055…` (3157 lines) drops the three
  `maxRecDepth 4000` raises (file byte-identical minus those lines; guard-free elaboration exit 0,
  163 s — they were redundant at the default depth). The `maxHeartbeats 8000000` on
  `word_fixed_exp_horner_acc` is **kept as a named accepted opening**: the guard-free red run (exit 1,
  12 s) fails first at the theorem *statement* (`whnf` over the ~15-operation `exp_horner` body at
  `:905`, cascading to the `I128.mul_spec` side conditions and the `|acc2.val·r.val| ≤ 2^100` simplex
  call) — the deep-continuation shape, so the E1B4 list reformulation does not transfer; the pattern
  for a future removal slice is factoring the ~15 stages into separate small-term WP lemmas carrying
  the post explicitly. Gate exit 0 (2445 jobs, 163 s); audit green (98 `fixed_exp` lines, same names);
  control non-slack (`≤ 30 → ≤ 29` in `word_fixed_exp_reduction_value`) red; restore byte-identical.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AI` (deepseek-flash, ~47 min of the authorized 30/45 box — overrun;
  **`exp_horner` spec landed**): `Numeric/ExpRefinement.lean` `53182c60…` (1939 lines) rewrites
  `word_fixed_exp_horner_acc` as the spec on `fixed_exp.exp_horner`'s body (`⦃ z => z.val =
  fixed_exp.horner_acc r.val ⦄`), proved fresh against the new chunked extraction's local names; the
  inline copy is retired in place (same name, so the audit line still resolves). Cap status: the single
  `maxHeartbeats 8000000` is retained on that one theorem (the per-operation proof needs it); module
  build 153 s, full gate exit 0 (3895 jobs), `AxiomAudit` green. Control non-slack (final add constant
  `4294967296 → 4294967297`) red; restore cached. One corruption was rebuilt from the pristine W1-T1AH
  copy with a single-pass token rename. Remaining: `word_fixed_exp_body` — the tight `|r| ≤ 2^32`
  lemma, the `|horner_acc| ≤ 2^40` bound, and the composition script (`W1-T1AJ`, running).
- `APEIRON-INT-TRANSCENDENTAL-W1-T1AH` (deepseek-flash, ~45 min of the authorized 30/45 box — at the
  hard stop; first half landed): `Numeric/ExpRefinement.lean` `084ba5cc…` (1966 lines) compiles
  against the regenerated extraction (compile fixes: `exp_reduce.LOWER/UPPER`, refusal proofs unfold
  `exp_reduce`), retires the obsolete `word_fixed_exp_window_past_guards` rewrite and its helpers with
  an in-file note, and lands `fixed_exp_exp_reduce_spec` (the `exp_reduce` pair/refusal post via
  `spec_mono` on the retained pair chunk) and `fixed_exp_exp_scale_spec` (via `word_fixed_exp_scale`).
  Module exit 0 (2224 jobs, 237 s). The attempted `fixed_exp_exp_horner_spec` (`unfold exp_horner;
  exact word_fixed_exp_horner_acc`) doubled the defeq cost and pushed the module past 540 s —
  withdrawn; the 8M cap remains on the retained inline `word_fixed_exp_horner_acc` only. Control
  non-slack (`· 2977044472` → `· 2977044473`) red; restore cached. Remaining: the fresh `exp_horner`
  spec (retire the inline copy; one 8M cap), the exposed `|r| ≤ 2^32`/`|horner| ≤ 2^40` bounds, and
  `word_fixed_exp_body` by the trig technique (`W1-T1AI`, running). Lead fix + clearance:
  `AxiomAudit.lean` line 839 swapped the retired name for the two new specs; the full audit is
  **green** (`86753d52…`, closed cone), clearing every pending registration (including the trig
  E2C/E2C2 ledgers).
- `APEIRON-INT-TRANSCENDENTAL-W2-T1O` (deepseek-flash, ≈16 min of a 20/30 box — inside; no landing,
  residual narrowed): the file is unchanged (`74022c67…`, 1521 lines); no `set_option`. Step 1 (the
  strengthened `word_taylor_spec` post `|sin_t|, |cos_t| ≤ 2^34` + the 7-way `rcases`) compiles and
  gates green in scratch; the e2e type-checks to the six selector cases and the `swap.val = 1` cases
  close (`simp [sin_out1_post, hnt, h1, ← hcos]`; full `simp` so the selector/`neg_one_mul` reductions
  fire). Residual, both in `swap.val = 2/0`: `step*` already splits the following `if negate` (so
  `by_cases hnt` re-splits and `simp only [if_pos hnt]` makes no progress) and names the neg result `x`
  (not `i14`). Fix: do not re-split — close the two post goals `step*` produced with `_` witnesses and
  `simp_all`, or use explicit `split`s for naming. `W2-T1P` (running) lands step 1 unconditionally and
  the e2e with that route.

### Sequence to the numeric barrier (lead, current)

1. ✅ `W1-T1X`/`W2-T1O` landed; exp and trig discrete value semantics complete.
2. ✅ Every headline registered; full `AxiomAudit.lean` green (`99c83c9e…`, closed cone) with the exp
   (`fixed_exp_refines_proved`) and trig (`fixed_sin_cos_refines_proved`) headlines.
3. ✅ `E1` exp real-error decomposition closed: `fixed_exp_refines_proved` (`ExpRefinement.lean`
   `33c1efc3…`, 3160 lines) — `|toReal v − exp (toReal x)| ≤ 2^-20·exp (toReal x) + 4/q32`; scaling
   exact values, Horner error (`epsHorner = 2·10^-7`), on-window range budget, total ≈ 4.6e-7.
4. ✅ `E2` trig real-error decomposition closed: `fixed_sin_cos_refines_proved` (`RopeRefinement.lean`
   `3bd26da3…`, 3524 lines) — `|toReal sc.f − f (toReal x)| ≤ 2^-20·|f (toReal x)| + 2^-20`, composed
   worst case 4.92e-7 (4.29e-7 at `|s| = π/4`); the direct-P post assembly took the xiaomi escalation (`E2C14`).
5. ✅/⚠️ `E3` guard restructuring done: rsqrt's two 1.6M removed (`916ca0ef…`); exp's three
   `maxRecDepth 4000` removed (`6c436055…`); the 8M on `word_fixed_exp_horner_acc` is the one named
   accepted opening (owner acceptance pending), with the failing-goal diagnostic and the future
   factor-the-stages pattern recorded above. Five of six raises discharged.
6. ✅/⚠️ `E4` barrier on the freeze (exp `6c436055…`, trig `3bd26da3…`, rsqrt `916ca0ef…`): machine
   checks green — `proof/regen.sh --check` exit 0 (final re-run; extraction `328655cd…` unchanged),
   `cd proof && lake build` exit 0 (1708 jobs), `cd proof/corpus && lake build` exit 0 (4026 jobs),
   `AxiomAudit` `99c83c9e…` exit 0 (3896 jobs, closed cone), `cargo test/clippy/fmt` exit 0.
   Independent verification (three fresh glm sessions) returned **verified** on all three artifacts
   with independent mutation controls. Remaining: the owner's acceptance of the single kept
   `maxHeartbeats 8000000` raise; commit of the working tree is unauthorized (evidence binds to
   working-tree digests).
- **Independent verification receipts** (fresh glm sessions, non-writer model; read-only on the repo,
  controls in disposable copies):
  - trig `RopeRefinement.lean` `3bd26da3…`: verified; `lake build Numeric.RopeRefinement AxiomAudit`
    exit 0 (3896 jobs); `#print axioms` cone `[propext, Classical.choice, Quot.sound]`; independent
    mutation `trig_delta 326/q32 → 325/q32` red at `trig_reduction_displacement:1817`; restore
    byte-identical. Prose correction applied: the composed worst case is 4.92e-7 (`|s| ≤ 4/5`),
    4.29e-7 at `|s| = π/4` — the earlier `5.8e-7` figure over-estimated (fixed here and in the
    handoff).
  - rsqrt `RsqrtRefinement.lean` `916ca0ef…`: verified; statement-preservation diff vs `b50c6c59…` is
    proof-body-only plus the two removed guard lines (37 declarations, signature lines identical);
    independent domain mutation (window top `2^52 → 2^51`) red at `fixed_rsqrt_refuses_proved:153`;
    cone closed; zero options. Spec drift fixed: `docs/epics/inference-parity.md` now carries the
    exact combined bound (`+ 4/q32` floor).
  - exp `ExpRefinement.lean` `6c436055…`: verified; cache-independent re-elaboration exit 0 (157 s);
    `#print axioms` closed cone; two independent mutations (`deltaRed 31/10^9 → 30/10^9` red at
    `fixed_exp_reduction_displacement:2107`; floor `4/q32 → 1/q32` red at
    `word_fixed_exp_body_refines`); restore byte-identical. Disclosed: the Rust contract's `1e-8` is
    stricter than the proved `2^-20`; the single 8M guard remains the named accepted opening.
- `APEIRON-INT-TRANSCENDENTAL-W1-T1Q` (deepseek-flash, ~32 min of a 20/30 box — overrun; step 3 core
  landed): `Numeric/ExpRefinement.lean`
  `f7f0efd52fe989f9c3b3a0388ffb01bb646998188c5eebce1cc9511fcf22e3ba` (1479 lines) proves
  `fixed_exp_scale_neg_core` — for `1 ≤ n.val`, `n.val < 64`, `|acc6.val| ≤ 2^40`, the extracted
  round-to-nearest right-shift core returns `Err` or `Ok v` with `(v.val : ℤ) = ((acc6.val +
  2^(n−1)) >>> n) mod 2^64`. The three named fixes worked (`exact_mod_cast` underflow; explicit
  `calc` for `2^m < 2^128` with `norm_num [I128.size, I128.numBits]`; per-bullet scope); an
  `unnecessarySeqFocus` linter error that first failed the package build was fixed. Module exit 0
  (2224 jobs, 163 s); control non-slack (`2^(n−1)` → `2^n`) red (160 s), restore cached. No new
  `set_option`. Lead registration: verified in the green `AxiomAudit.lean` `6d07c88b…` (3894 jobs,
  closed cone). Remaining: the flush sub-case (`64 ≤ n.val → Ok 0`), the full `k < 0` branch, and
  `word_fixed_exp_scale` (`W1-T1R`, running).
- `APEIRON-INT-FORMAT-W3-T1`/`W3-T2` (deepseek-flash, ~45 + ~70 min — overruns): the format layer
  lands as `src/format/{gguf,quant,mod}.rs` (GGUF header/tensor-directory parse and K-quant code
  unpack, `Result` refusals, 19 tests; cargo trio green), `proof/extract/lib.rs` wrappers, and the
  lead's two `--start-from`/digest additions to `proof/regen.sh`. The extractor's integer fragment
  accepts `word_q4_k_codes`/`word_q6_k_codes` after the loops were reshaped to while-accumulators,
  the nested rank loop hoisted, and iterators/`TryFrom`/nested borrows removed;
  `proof/ApeironWord.lean` is `e6ebf0f5…` (11585 lines, +312; `regen.sh` and `--check` exit 0;
  corpus 4026 jobs, `AxiomAudit` 3896 jobs green on the new identity). **Named migration items:**
  `word_gguf_parse_header` **translates** (`W3-T4`: the cause was a `&'static str` payload on a direct
  `return Err`, fixed via `ok_or(...)?`; the recursion was not the translation cause). The raw-bits
  metadata landed (`W3-T16`) and the root was re-added: extraction exit 0, but the generated Lean
  failed to compile — Aeneas cannot prove `Cur.value_loop` monotone (recursive call enclosed in a
  loop) — so the root was reverted and the extraction restored (`proof/ApeironWord.lean` `a3ad8c27…`,
  11585 lines; regen/`--check` exit 0). `W3-T20` rewrites `Cur::value` iteratively and must show the
  generated Lean compiles before the root is re-added;
  `word_{q4,q6}_k_dequant` float scaling is outside the integer fragment (consistent route:
  fixed-point dequant rewrite, matching the integer executing-path decision). `W3-T3` refines the two
  code functions in `Numeric/FormatRefinement.lean` (`word_q4_k_codes_ok` landed; Q6_K `W3-T15` in
  flight).
- `APEIRON-INT-ENGINE-W4-T1` (deepseek-flash, ~30 min of a 30/45 box — soft stop met): the engine
  skeleton lands — `src/engine/mod.rs` `b7c203a3…` (350 lines), `src/engine/loader.rs` `feb8590b…`
  (630), `examples/engine_probe.rs` `f37b6c06…` (102), `src/lib.rs` wiring. The probe loads
  `/home/socol/models/gguf/qwen05b-Q4_K_M.gguf`: 290 tensors, Qwen2.5-0.5B config (`n_layer 24`,
  `n_head 14`, `n_head_kv 2`, `head_dim 64`, `n_embd 896`, `n_ff 4864`, `vocab 151936`, `rope_theta
  1e6`); exact Q32.32 token embedding via `f16_to_fixed` (every finite f16 is a multiple of `2^-32`)
  plus `q8_0_codes`; determinism across loads; out-of-vocab refused. Gates: cargo test/clippy/fmt exit
  0; probe exit 0. Control: tensor-offset corruption → EOF refusal red, restore green. The decode-loop
  port map is recorded in `src/engine/mod.rs` (rows 0–8; row 1 landed). Residual: Q5_0 geometry
  missing (132/290 tensors), private K-quant scale accessors, `F32` rows refused, rows 2–8 unported;
  grade `tested`. `W3-T14` (Q5_0 + accessors) in flight.
- `APEIRON-INT-FORMAT-W3-T14` (deepseek-flash, ~19 min of a 30/45 box — inside): `src/format/quant.rs`
  `f55de4cd…` (825 lines) adds `GGML_TYPE_Q5_0` (22-byte block: f16 `d` + 4-byte high-bit plane + 16
  low nibbles), `q5_0_codes` (signed `q − 16`, indexed while loops, extraction-clean), `q5_0_scale`,
  public integer accessors (`q2_k_scales`/`q3_k_scales`/`q6_k_scales`, `q2_k_scale_min`, raw-f16
  `*_d`/`*_dmin`) and fixture tests; `src/engine/loader.rs` `5c5da3b9…` (657 lines) classifies Q5_0.
  Gates: cargo trio exit 0; probe exit 0 and **290/290 tensor sizes resolve** (computed Q5_0 sizes
  equal the directory extents). Controls: high-plane fold and size-field mutations red; restore
  byte-identical. Residual: `dequant_row_fixed` still refuses Q5_0/K-quants (`W4-T2` running) and the
  `src/engine/mod.rs` comments are stale (fixed there).
- `APEIRON-INT-ENGINE-W4-T2` (deepseek-flash, ~20–25 min of a 30/45 box — inside): `src/engine/mod.rs`
  `364fd68f…` (642 lines) lands port-map rows 1–2 — `dequant_row_fixed` for F16/Q8_0/Q5_0/Q4_K/Q6_K/
  Q2_K/Q3_K (exact f16→Q32.32 scales via `f16_to_fixed` plus the public accessors; `value = d·sc·q`,
  K-quants `− dmin·m`; checked arithmetic, no shift/wrap; unsupported/short payloads refuse) and
  `rms_norm_fixed` (i128 mean accumulator, `crate::numeric::fixed_rsqrt`, round-to-nearest epilogue;
  `EmptyRow`/`OutOfRange` refusals). Gates: cargo trio + probe exit 0; 17 engine tests. Control: Q2_K
  slope mutation red; restore green. Residual: `F32` rows refused, `eps_fixed` conversion is a
  row-0/load concern, rows 3–8 pending (`W4-T3` RoPE running).
- `APEIRON-GPU-SPV-W2-T3` (deepseek-flash, ~10 min of a 20/30 box — inside): `src/spirv/check.rs`
  `05cd1a70…` drops 196/197 — `SERVED_OPCODES: [u32; 42]`, the mechanically derived served set; the
  676 captured words contain exactly 42 distinct opcodes and neither 196 nor 197, verified before the
  edit. Gates: focused 6/6, workspace green, clippy green on the shared tree; control (`194 → 196`
  seed) red, restore byte-identical. The Lean side (`Spirv.Fragment.servedOpcodes` → 42,
  `executingOpcodes` → 37, `Inst.uses` forward-reference exemptions) is `W3-T19`.
- `APEIRON-GPU-SPV-W3-T19` (deepseek-flash, ~3 min of gate/control wall time — inside): aligns the
  Lean fragment with the derived set and SPIR-V 1.3 forward references — `Spirv/Fragment.lean`
  `3ee225c3…` (served `42`, executing `37`; guards and `executingOpcodes_eq` re-closed; off-served
  112/124/196/197/198 recorded) and `Spirv/Decode.lean` `35b8bb0b…` (`Inst.uses` returns `[]` for
  `executionMode`/`decorate`/`memberDecorate`; `forwardRefTestModule` accepted,
  `undefinedUseTestModule` still refused; D2 re-proved). Gate `lake build Spirv.Fragment Spirv.Decode`
  exit 0 (2967 jobs); controls: `4452 → 4453` red at the removal guard and `executingOpcodes_eq`;
  exemption removal red at the fixture; restore byte-identical. The Rust checker matches at
  `[u32; 42]` (`W2-T3`).
- `APEIRON-INT-ENGINE-W4-T3` (deepseek-flash, ~4 min of a 30/45 box — inside): `src/engine/mod.rs`
  `9013d446…` (937 lines) lands port-map row 3 — `RopeTable::new` (load-time `theta^(-2j/head_dim)`
  table as Q32.32; the single host-`f64` boundary, declared) and `rope_rotate_fixed` (angle =
  `pos · freq_j`, quadrant reduction inside `fixed_sin_cos`, round-to-nearest `>>32` epilogue, refusals
  past the 4096-rad window). Gates: cargo trio + probe exit 0; four new tests (zero-angle,
  quarter-turn hand case, load boundary, refusals). Control: frequency-exponent mutation red; restore
  green. Residual: no stored table on `Engine` (caller holds it), grade `tested`, wide-context
  Payne–Hanek remains open. `W4-T4` (projections) running.
- `APEIRON-INT-ENGINE-W4-T4` (deepseek-flash, ~4 min of gate/control (≤15 min slice) of a 30/45 box —
  inside): `src/engine/mod.rs` `3ac4cc8c…` (1241 lines) lands port-map row 4 — `matvec_fixed` (one
  kernel over the dequantized rows for Q/K/V/O/FFN/LM-head/embedding shapes) and `row_dot_fixed`:
  exact `i128` accumulation, one round-to-nearest `>>32` per output (ties toward +∞), typed refusals.
  Gates: cargo trio + probe exit 0; 25 engine tests (hand 2×2 tie case, refusals, the real
  `token_embd` row-dot). Control: `round_shift_32` tie mutation red; restore byte-identical. Residual:
  no decode-loop integration yet; `F32` weights refused; grade `tested`. `W4-T5` (attention) running.
- `APEIRON-INT-FORMAT-W3-T20` (deepseek-flash, ~44 min of a 30/45 box — at the hard cap; done): the
  recursion elimination lands — `src/format/gguf.rs` `3ee474ed…` (943 lines): `Cur::value` split into
  `scalar`/`array_header` and a loop-only `value` over a flat post-order arena with open-frame stacks
  and a numeric mode; the depth cap and the `n > remaining` gate are preserved; the generated Lean has
  `value → value_loop0` with no self-call. Verified at both levels: direct Charon+Aeneas exit 0, strip
  leaves zero axioms, and `cd proof && lake env lean <generated>` exit 0 (the check the earlier probes
  missed). Extractor-fragment findings recorded: a loop-carried `bool` selector fails the loop fixed
  point (`u32` works), `Vec::is_empty` emits an unstripped axiom, and `GVal`'s derived `Clone` is
  self-referential (`mem::replace` only). Controls: corrupt-length bound removal red (abort); the
  anti-allocation gate is preserved in source but not independently observable (recorded coverage
  gap). Lead: the root is re-added to `proof/regen.sh`; `proof/ApeironWord.lean` is now `71ccf8e1…`
  (13066 lines, +1481); regen and `--check` exit 0; package build 1708 jobs. Corpus and audit
  re-verification follow the `W3-T22` lane's settle.
- `APEIRON-INT-FORMAT-W3-T22` (deepseek-flash, ~44 min of a 30/45 box — at the hard cap; **Q6_K
  closed**): `Numeric/FormatRefinement.lean` `ae3d48c4…` (1025 lines) proves `word_q6_k_codes_ok :
  210 ≤ Slice.len blk → ∃ v : Array I8 256, word_q6_k_codes blk = ok (.Ok v) ∧ ∀ i < 256,
  (v.val.getD i 0#i8).val = q6CodesRef blk.val i` by the `hspec` WP prefix, `step*` over the
  `QK_K/2`/`QK_K/4` divisions and slice reads, `WP.spec_bind` with `q6_outer_spec` + `q6Inv_repeat`,
  `q6Val_four`, and the `q6QsRef_eq_q6CodesRef` bridge. Gate exit 0 (1707 jobs); control non-slack
  (`take 64 → take 63`) red; restore byte-identical; no `set_option`. Both K-quant unpackers now carry
  exact positive byte-equality and exact refusal. Lead registration:
  `#audit apeiron_word.word_q6_k_codes_ok` appended; corpus and audit re-verification against the
  parser extraction (`71ccf8e1…`) follows the parser-refinement lane's settle.
- `APEIRON-INT-FORMAT-W3-T26` (deepseek-flash, ~10–12 min of a 30/45 box — inside): `src/format/bpe.rs`
  `fe712a65…` (1260 lines) implements `Bpe::from_tokenizer_json` (checked self-contained JSON reader),
  the byte-level GPT-2 alphabet, the cl100k split and lowest-rank/leftmost merging, literal added
  tokens, and `encode`/`decode` with typed refusals. Test vectors match fold's tokenizer on ten inputs
  (`Hello world`, punctuation, newlines, CJK, `<|im_start|>`/`<|im_end|>`, leading space, empty,
  digits) with decode round-trips. Gates: workspace tests 370 passed; clippy/fmt exit 0 (clearing the
  workspace block). Control: byte-map mutation red; restore byte-identical. Residual: NFC not applied
  (fold doesn't either); the algorithm core is extraction-disciplined but the loader (`HashMap`,
  `fs::read`, `f64`) is outside the modeled fragment — extraction/refinement is a later slice.
- `APEIRON-INT-ENGINE-W4-T5` (deepseek-flash, ~22 min of a 30/45 box — inside): `src/engine/mod.rs`
  `f53494f0…` (1819 lines) lands port-map row 5 — `KvState` (Q32.32 position-major, GQA, checked
  capacity) and `attn_head` (raw `row_dot_fixed` + one `fixed_rsqrt(head_dim)` score epilogue,
  softmax via `fixed_exp` with a checked `i128` total and round-to-nearest division, causal prefix,
  second epilogue over `p·v`); typed refusals, no clamping. Gates: cargo trio + probe exit 0; seven
  attention tests (hand 2-key softmax, score-scale pin, GQA, causal, refusals, determinism). Control:
  softmax constant mutation red; restore byte-identical. Residual: exact Q32.32 KV (the fold int8 KV
  cache is not ported), grade `tested`, no decode-loop integration. `W4-T6` (FFN) running.
- `APEIRON-INT-ENGINE-W4-T6` (deepseek-flash, ~25 min of a 30/45 box — inside): `src/engine/mod.rs`
  `8fa8af41…` (2062 lines) lands port-map row 6 — `silu_fixed` (`|v|·2^32/(2^32 + fixed_exp(-v))`,
  ties away from zero, domain `[-21·ONE, 16·ONE]` matching the `fixed_exp` window, refusals) and
  `ffn_fixed` in the fold's exact ordering (the gate is inside the SiLU: `round(silu(gate)·up/2^32)`
  then `W_down`). Gates: cargo trio + probe exit 0; 29 engine tests. Control: round-to-nearest →
  truncation mutation red (`1984696314` vs `1984696315`); restore byte-identical. Residual:
  `ZeroDenominator`/elementwise `Overflow` are defensive guards with no reachable witness (recorded as
  non-vacuity openings); grade `tested`. `W4-T7` (LM head + sampler) running.
- `APEIRON-INT-ENGINE-W4-T7` (deepseek-flash, ~30 min of a 30/45 box — soft stop reached; rows 7–8
  landed): `src/engine/mod.rs` `6ee03602…` (2436 lines) — `lm_head_fixed` over the tied
  `token_embd.weight` (the directory has no `output.weight`; recorded), `argmax_fixed` (ties to the
  lowest index), and `top_p_fixed` (exact-max shift, `fixed_exp`, one round-to-nearest division per
  probability, shortest prefix); token-0 logits argmax = 0, checksum `0x4b39186d63e81773`, determinism
  asserted across independent loads. Gates: workspace tests and probe exit 0; workspace clippy/fmt
  blocked only by the BPE lane's mid-edit `src/format/bpe.rs` (the owned file is clean under the
  package-only clippy). Control: argmax tie mutation red; restore byte-identical. Residual: the
  stochastic draw and temperature are open (need an RNG contract); grade `tested`. `W4-T8`
  (decode-loop integration) running.
- `APEIRON-INT-FORMAT-W3-T3` (deepseek-flash, ~40 min of a 30/45 box — soft stop exceeded; checkpoint
  landed): `Numeric/FormatRefinement.lean` `709b7b62…` (113 lines) defines the exact integer references
  `q4CodesRef`/`q6CodesRef` for the Q4_K/Q6_K code layouts and proves both block-size constants and the
  **refusal correspondence** (`word_q4_k_codes`/`word_q6_k_codes` return the exact `Truncated`
  `QuantError` below 144/210 bytes). Gate exit 0 (1707 jobs); control non-slack (refusal hypothesis
  `144 → 200`) red; zero `set_option`. Lead registration: six declarations audited
  (`AxiomAudit.lean` `bd75bd11…`, 3897 jobs, closed cone). Residual (`W3-T5`, running): the positive
  `i < 256` pointwise equality via per-loop `loop.spec_decr_nat` invariants. `W3-T4` (parser fragment
  fix) still running.

- `APEIRON-ALGEBRA-D1` (deepseek-flash, ~25 min of a 30/45 box — inside): `docs/epics/model-algebra.md`
  (166 lines) and `docs/epics/model-algebra.md` (212) define the governing abstraction — the layer fold
  over the canonical object (`token_step_fold`, `memory_step_model`, the attention/spinor identities
  with `attention_channel_form` named open), the three-class reflection contract (extracted kernels
  source→abstraction; generated/data parameters abstraction→source; tested driver), the composed-margin
  target, the canonical-packing direction, and the proof-cost budget. Path checks clean; the lead
  registered the previously unregistered `Sanguine.AttentionSpinor.softmax_attention_is_online_spinor`
  headline (`AxiomAudit.lean` `001ccab5…`, 3900 jobs).
- `APEIRON-ALGEBRA-D2` (deepseek-flash, ~10 min of a 30/45 box — inside):
  `Nous/Model/ErrorComposition.lean` `1dc48144…` (326 lines) proves the abstract layer-composition
  budget (`Layer.budget` = `Σ_k δ_k·Π_{j<k} L_j`, `norm_execComp_sub_refComp_le`,
  `budget_replicate_le`), the margin lemmas (`strictWinner_perturb`, `strictWinner_perturb_argmax`,
  `decision_margin_preserved`), and the numeric instantiation (`numericScale = epsExp + epsHorner +
  trig_floor < 1e-5`, `numeric_fold_margin_usable`, `numeric_decision_unchanged`). Gate exit 0 (3391
  jobs); control (ε + 1) red; registered in the audit (`8d3d8479…`, 3909 jobs). First model-level error
  theorem: per-op ε compose to a margin-guarded decision; discharging the per-layer hypotheses is the
  engine-refinement wave.
- `APEIRON-ALGEBRA-D3` (deepseek-flash, ~55 min — overran both boxes, reported as a budget failure):
  `src/format/packing.rs` `da8555b6…` (1352 lines) — the canonical packing grammar
  (`width ∈ {2,3,4,6,8} | Trit(L)`, `high_planes`, `block ∈ {32,64,256}`, `group`,
  `scale_kind ∈ {F16, Pow2Exp8}`, `has_min`, `signed_code`, `bias`, `signed_coeff`, plane/nibble order)
  with `decode_block`/`row_decode_fixed` (Q32.32), canonical encode/decode and
  `from_gguf_kind`/`convert_row`; conversion equality with `quant.rs` is bit-exact for
  Q8_0/Q5_0/Q2_K/Q3_K/Q4_K/Q6_K on fixtures and real tensors. Inexpressible: `F32`/`F16`, `IQ*`,
  `Q4_1`/`Q5_1`, Q5_0's native 5-bit (expressed 4+1 plane). Gates: cargo trio exit 0 (188 lib tests).
  Controls: Q2_K block and Q5_0 bias mutations red, restore green (order mutations slack by
  construction). `APEIRON-ALGEBRA-A5` (extraction root) is running.
