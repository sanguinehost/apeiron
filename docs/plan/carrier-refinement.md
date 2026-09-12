# Plan — Carrier and word refinement

- **Owner:** Lucas 'Paperboy' Rose-Winters (operator: `deepseek-flash`)
- **Intent:** `docs/intent/carrier-refinement.md`
- **Spec:** `docs/spec/carrier-refinement.md`
- **Source revision:** `d7121ac0432a4842aff69e63cfc5486d56c4dd73`; working tree clean
- **Status:** approved; all waves complete, W7 committed with the fixed three-cell line and its success/unrepresentable witnesses

## Goal and completion condition

Every task gate is green with its negative control run and restored. The barrier is
the fresh extraction plus the bridge build with a closed axiom cone.

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
```

## Non-goals

- Container storage refinement, rustc/LLVM or GPU evidence, totality of the partial kernels.
- A physical mass law or a water solver; this epic stops at the exact integer readout.
- A corpus native toolchain/manifest rebuild.

## Wave map

The refinement files are shared, so each wave is one task on one lane. A wave opens
only after the previous gate is green.

| Wave | Task | Files | Owner | Gate | Negative control |
|---|---|---|---|---|---|
| W1 | Extraction entry point and denotation | `proof/extract/lib.rs`, `proof/regen.sh`, `proof/Denotation.lean` | Flash | `proof/regen.sh --check` | broad extraction emits external axioms; `--start-from` roots remove them |
| W2 | Word multiplication and norm refinement | `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean` | Flash | `cd proof/corpus && lake build` | dial `1 → 2` in a throwaway `src/cell.rs` breaks `word_mul_denote`/`word_norm_denote` |
| W3 | Carrier point readout refinement | `src/carrier.rs`, `proof/extract/lib.rs`, `proof/corpus/WordRefinement.lean` | Flash | `cd proof/corpus && lake build` | composed from W2; the dial mutation breaks `word_point_density_denote` |
| W4 | Exchange and checked-acceptance refinement | `src/exchange.rs`, `proof/extract/lib.rs`, `proof/corpus/WordRefinement.lean` | Flash | `cd proof/corpus && lake build` | impulse `sub → add` and generated branch inversion turn the exchange/acceptance theorems red |
| W5 | Scalar step, half-flux and finite conservation | `src/carrier.rs`, `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean` | Flash | `cd proof/corpus && lake build` | generated `checked_sub → checked_add` and the face-orientation swap turn the step theorems red |
| W6 | Corpus pin and carrier lab | `proof/corpus/lakefile.lean`, corpus `Carrier*.lean`, `tests/the_carrier_*.rs` | Flash | corpus `lake build`; `cargo test --workspace --all-features --locked` | swapping the two face arguments in `sum_scalarCarrierStep_eq` breaks the telescoping proof |
| W7 | Fixed three-cell diffusive line | `src/carrier.rs`, `proof/extract/lib.rs`, `proof/regen.sh`, `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `tests/the_diffusive_line3_step_is_exact.rs` | Flash | `proof/regen.sh --check`; `cd proof/corpus && lake build`; `cargo test --workspace --all-features --locked` | swapping the generated left cell's shared face breaks `word_diffusive_line3_step_eq_some_iff`; an always-`none` generated body breaks `word_diffusive_line3_step_zero_success`; the odd-difference and unrepresentable fixtures refuse |

## Task blocks

### W1 · Extraction entry point and denotation

- **Scope:** `#[path]`-include the real `src/scalar.rs`, `src/cell.rs`, `src/exchange.rs` and `src/carrier.rs`; add monomorphic roots; define `denote : I64 → ZMod (2^64)` and the wrapping operations.
- **Files:** `proof/extract/lib.rs`, `proof/regen.sh`, `proof/Denotation.lean`
- **Depends on:** none
- **Grade:** proved (denotation); tool gate (freshness)
- **Gate:**

```sh
proof/regen.sh --check
```

- **Negative control:** the earlier whole-module probe emitted unknown external axioms for hashing, `i128::trailing_zeros` and iterator folding; the `--start-from crate::word_*` roots remove them and `regen.sh`'s `axiom|sorry|admit` audit rejects their return.
- **Evidence to record:** source digests; Charon `0.1.254` (`b104e24f…`), Aeneas `nightly-2026.09.09-505b6ca`, Charon Rust `nightly-2026-08-18`, Lean `4.31.0`.

### W2 · Word multiplication and norm refinement

- **Scope:** prove extracted `Cell<i64>::mul`/`norm_form` equal `ringMul`/`ringNormForm` at `ZMod (2^64)` and transfer `ringNormForm_mul`/`ringMul_conj`.
- **Files:** `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `proof/corpus/lakefile.lean`
- **Depends on:** W1
- **Grade:** proved
- **Gate:**

```sh
cd proof/corpus && lake build
```

- **Negative control:** change the dial's hyperbolic value from `1` to `2` in a throwaway copy of `src/cell.rs`, re-extract, rebuild: `sigma_value_eq` and all four multiplication/norm theorems fail (the `rfl`/`ring` cases); restore the source and generated file and the build passes. The production source is not modified.
- **Evidence to record:** `word_mul_denote`, `word_norm_denote`, `word_norm_mul`, `word_mul_conj`; all four `#audit` clean.

### W3 · Carrier point readout refinement

- **Scope:** add the ring-generic `point_density`/`point_flux` production kernels and refine the extracted `i64` readouts to `Apeiron.Physics.CarrierAnyRing`.
- **Files:** `src/carrier.rs`, `proof/extract/lib.rs`, `proof/corpus/WordRefinement.lean`
- **Depends on:** W2
- **Grade:** proved
- **Gate:**

```sh
cd proof/corpus && lake build
```

- **Negative control:** covered by the W2 dial mutation; `word_point_density_denote` composes `word_mul_denote` and `word_norm_denote`, so the same mutation turns it red. Not separately mutated.
- **Evidence to record:** `word_point_density_denote`, `word_point_flux_denote`, `word_point_density_ringOne_independent`, `word_point_flux_ringOne_independent`; `CarrierAnyRing.lean` sha256 `bee50d1d…`.

### W4 · Exchange and checked-acceptance refinement

- **Scope:** extract `exchange_impulse`, `elastic_exchange` and `accept_derived_impulse`; prove the ring-generic `CarrierBodyWorkAnyRing` laws and the acceptance characterization. Replace `bool::then` with an explicit `if` so Charon translates the branch instead of an external combinator.
- **Files:** `src/exchange.rs`, `proof/extract/lib.rs`, `proof/corpus/WordRefinement.lean`
- **Depends on:** W2
- **Grade:** proved
- **Gate:**

```sh
cd proof/corpus && lake build
```

- **Negative control:** change the derived impulse from `p.a.sub(q.a)` to `p.a.add(q.a)` (and `q.b` likewise) in a throwaway `src/exchange.rs`, re-extract, rebuild: `word_exchange_impulse_denote` and `word_elastic_exchange_denote` fail. Invert the generated branch of `exchange.accept_derived_impulse` in a throwaway `ApeironWord.lean`: `word_accept_derived_impulse_accepts`, `_accepts_derived` and `_spec` fail. Change one equality component (`self.a` vs `other.b`) in the generated `CoreCmpPartialEqCell.eq`: `word_cell_eq_eq` and the acceptance theorems fail. Restore each and the build passes.
- **Evidence to record:** `word_exchange_impulse_denote`, `word_elastic_exchange_denote`, `word_elastic_exchange_carrier`/`_body`/`_totalMomentum`/`_normFormLedger`, `word_exchange_impulse_antisymm`, `word_accept_derived_impulse_spec`/`_rejects`/`_rejects_of_ne`/`_accepts`/`_accepts_derived`/`_denote`/`_denote_canonical`; `CarrierBodyWorkAnyRing.lean` sha256 `ea315561…`.

### W5 · Scalar step, half-flux and finite conservation

- **Scope:** extract `conservative_density_step` and `diffusive_flux_half`; prove conditional denotation, exact success preconditions, the `exactHalfFlux` witness and the finite conservative total. Keep the `Option` signature and explicit `match` (`?` has no Aeneas model).
- **Files:** `src/carrier.rs`, `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`
- **Depends on:** W3
- **Grade:** proved
- **Gate:**

```sh
cd proof/corpus && lake build
```

- **Negative control:** change the generated `carrier.diffusive_flux_half` result from `checked_sub` to `checked_add` in a throwaway `ApeironWord.lean`; `word_diffusive_flux_half_success` fails. Swap the two face arguments in `Apeiron.Physics.sum_scalarCarrierStep_eq` in a throwaway `CarrierScalarKernelAnyRing.lean`; the telescoping proof and `scalarCarrierStep_total_mass` fail. Restore each and the build passes.
- **Evidence to record:** `word_conservative_density_step_denote`, `_eq_some_iff`, `_success`; `word_diffusive_flux_half_denote`, `_success`, `_odd_refusal`, `_unrepresentable_refusal`; `word_conservative_density_step_overflow_refusal`; `word_conservative_total_mass`, `_closed`; `CarrierScalarKernelAnyRing.lean` sha256 `a89424a4…` at the `16112030` pin, `45454667…` at `d9aefc93`.

### W6 · Corpus pin and carrier lab

- **Scope:** pin the corpus with Lean `4.31.0` and Mathlib `v4.31.0` (landed at `16112030`, advanced to `d9aefc93` by W7); connect the extracted readout to the canonical carrier definitions and exercise the production path in a carrier lab.
- **Files:** `proof/corpus/lakefile.lean`, corpus `CarrierAnyRing.lean`/`CarrierBodyWorkAnyRing.lean`/`CarrierScalarKernelAnyRing.lean`, `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs`
- **Depends on:** W3, W5
- **Grade:** proved (bridge); tested (lab)
- **Gate:**

```sh
cd proof/corpus && lake build
cargo test --workspace --all-features --locked
```

- **Negative control:** the W2/W5 source and generated mutations; the lab tests also pin the
  refusal and sign edges.
- **Evidence to record:** corpus `16112030781eeec022139cdae07407131acc0915`, committed and
  pushed; module digests in the table below; bridge build exit 0.

### W7 · Fixed three-cell diffusive line

- **Scope:** add the closed three-cell `nu = 1/2` production kernel `diffusive_line3_step` and its `DensityLine3` carrier; extract it as `word_diffusive_line3_step`; prove the success characterization, the per-component `scalarCarrierStep` refinement, the two `exactHalfFlux` face witnesses, the source-only total-mass equation, its zero-source corollary, the odd-difference and unrepresentable refusals, and the all-zero success witness; test the exact values, refusal and transactionality.
- **Files:** `src/carrier.rs`, `proof/extract/lib.rs`, `proof/regen.sh`, `proof/corpus/WordRefinement.lean`, `proof/corpus/AxiomAudit.lean`, `tests/the_diffusive_line3_step_is_exact.rs`
- **Lane:** one task in its own wave; the generated file is shared, so the production, extraction, refinement and test edits land together rather than in parallel lanes.
- **Depends on:** W5, W6
- **Grade:** proved (bridge); tested (Rust)
- **Gate:**

```sh
proof/regen.sh --check
cd proof/corpus && lake build
cargo test --workspace --all-features --locked
```

- **Negative control:** swapping the generated left cell's shared face from `face_left` to `face_right` in a throwaway `ApeironWord.lean` breaks `word_diffusive_line3_step_eq_some_iff` and its consumers; replacing the generated final `ok (some { left, center, right })` with `ok none` breaks `word_diffusive_line3_step_zero_success`; restoring the generated file passes. The odd-difference and unrepresentable fixtures and the transactional-refusal test fail if the kernel rounds or returns a partial line.
- **Evidence to record:** `word_diffusive_line3_step_eq_some_iff`, `_denote_of`, `_denote`, `_total_mass`, `_total_mass_zero_source`, `_odd_refusal`, `_unrepresentable_refusal`, `_zero_success`; corpus `d9aefc93522e2109bc96158526de6dbd52477276`; `src/carrier.rs` `4fcdbeba…`; `ApeironWord.lean` `95542402…`; bridge build 3276 jobs.

## Recorded digests

Extracted working sources (`proof/regen.sh` prints these):

| source | sha256 |
|---|---|
| `src/scalar.rs` | `9199e955304cb3d54fe973ac5881f219d4c6f71b4be001a0d32a6fb16ffde92e` |
| `src/cell.rs` | `8cfc787aa0c6fcec6b039398f54fd7388f7dec8f047b98f8379e7165c0069404` |
| `src/exchange.rs` | `f488773c798c649ff59d39f7b24c4c17b6ff350fa9f7a3b4ccc81cfc479b7621` |
| `src/carrier.rs` | `4fcdbeba5c2d25d10e0a22f2f328e214d0e9884bd9f886a25abd4b6741a8a8e8` |

Canonical corpus modules at `d9aefc93`:

| module | sha256 |
|---|---|
| `Apeiron/Physics/CarrierAnyRing.lean` | `bee50d1df59be11d7a41536e4c56ff284235cb32c16265e55ba0cfd4e7df2b79` |
| `Apeiron/Physics/CarrierBodyWorkAnyRing.lean` | `ea3155611c61b7e4edcc26461ccbb9d238b0e8c24c74d921d39d3e1e8c3b34ac` |
| `Apeiron/Physics/CarrierScalarKernelAnyRing.lean` | `4545466793c6fcc367e5d8367784c9cd9f0f0cbb8bdbc952e6aaac00d2853b29` |
| `Apeiron/Physics/CarrierDissipation.lean` | `8d027a7c673af7f3bb0f7b3d56ad8b63ff94485caaa52e71d07fa152b816177d` |
| `Apeiron/Trichotomy/AnyRing.lean` | `51cf1ceecd283771a4a5ae52be39150cc3d3e2ff3419906822a10a121eafffca` |

Generated `proof/ApeironWord.lean` is `955424020949ca6509ab0c7e71b444d750aa8fadd98ec4c6cfe30ceb978f0bac` at the digests above.

## Barrier checklist

- [x] Every task gate exits 0, with its negative control run and restored. `proof/regen.sh --check` reported `extraction is fresh` (exit 0); `cd proof/corpus && lake build` exited 0.
- [x] No task touched a file owned by another task; the refinement files were one lane.
- [x] `cargo test --workspace --all-features --locked` exit 0 (173 including 9 doc-tests; the nine `tests/the_diffusive_line3_step_is_exact.rs` cases are W7); `cargo clippy … -D warnings` exit 0; `cargo fmt --all -- --check` exit 0; `cargo kani` exit 0 (60 harnesses).
- [x] `proof/regen.sh --check` fresh (1708 jobs); `proof/corpus` `lake build` passed (3276 jobs on the operator run), rebuilt at corpus `d9aefc93`.
- [x] Source and tool digests recorded above.
- [x] Claims registered at their grade; the corpus declarations are cited as links, not called verified by proximity.
- [x] Trust boundary stated in the intent and spec.

## Stop and rollback conditions

- A generated `axiom`/`sorry`/`admit` in the accepted chain stops the wave; `regen.sh` fails on it rather than admitting it.
- A mutation that does not turn its gate red means the check has no weight; report the theorem and stop.
- After three materially different failed approaches to one proof, stop and report the smallest unresolved goal.
