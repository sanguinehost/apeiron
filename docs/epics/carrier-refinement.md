# Carrier and word refinement

Merged epic record: `intent/carrier-refinement.md`, `spec/carrier-refinement.md`, `plan/carrier-refinement.md` (one file per epic; see `docs/README.md`).

## Intent — Carrier and word refinement

- **Owner:** Lucas 'Paperboy' Rose-Winters (operator: `deepseek-flash`)
- **Source revision:** `d7121ac0432a4842aff69e63cfc5486d56c4dd73`; working tree clean
- **Status:** approved; epic complete, evidence in `docs/epics/carrier-refinement.md`
- **Spec:** `docs/epics/carrier-refinement.md`
- **Plan:** `docs/epics/carrier-refinement.md`

### Problem

The Apeiron Rust crate's `Cell<i64>` arithmetic and its carrier readout have no
machine-checked link to the canonical Lean corpus. The word ring `Z/2^64` is not the
real trichotomy the corpus states, so `Apeiron.Trichotomy.AnyRing.ringMul`,
`ringNormForm`, `Apeiron.Physics.CarrierAnyRing.pointDensity`/`pointFlux`,
`Apeiron.Physics.CarrierBodyWorkAnyRing` and `Apeiron.Physics.scalarCarrierStep` do
not transfer to the extracted `i64` bodies by name. Tests and the bounded Kani
harnesses in `src/proofs.rs` execute the production path but prove no full-width
refinement.

### Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Kani at full `i64` width | the degree-four norm-form identity bit-blasts past twenty-five minutes at `i8`; a 4-bit window is not full width |
| `cargo test` against a `std` oracle | executes the path but relates no extracted body to a corpus declaration |
| a hand-written Lean restatement of `mul`/`norm_form` | not the production body; a citation, not a refinement |
| a corpus source rewrite to a new algebra | the corpus already states the ring-generic laws; only the pin and the extraction were missing |
| calling a readout identity a physical law | a readout identity is arithmetic, not a material model |

### Completion condition

The extraction is fresh from the committed sources and the bridge build checks every
headline theorem inside the closed axiom cone.

```sh
proof/regen.sh --check
cd proof/corpus && lake build
```

### Invariants

1. Every wrapper in `proof/extract/lib.rs` calls the real production body; none restates a formula.
2. The partial kernels keep their `Option` signatures and refuse rather than wrap or round.
3. Every headline theorem's transitive axiom cone is within `{propext, Classical.choice, Quot.sound}`.
4. The corpus revision and every extracted source digest are recorded with the evidence.
5. No theorem narrows its domain to close a proof; refusal witnesses show partiality is real.

### Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| extracted `Cell<i64>::mul` = `Apeiron.Trichotomy.AnyRing.ringMul` at `ZMod (2^64)` | proved | `word_mul_denote` |
| extracted `Cell<i64>::norm_form` = `ringNormForm` | proved | `word_norm_denote` |
| norm multiplicativity and conjugation transfer to the word ring | proved | `word_norm_mul`, `word_mul_conj` |
| extracted `point_density`/`point_flux` = `Apeiron.Physics.CarrierAnyRing.pointDensity`/`pointFlux` | proved | `word_point_density_denote`, `word_point_flux_denote` |
| extracted exchange = `CarrierBodyWorkAnyRing.ringExchangeImpulse` and momentum fields | proved | `word_exchange_impulse_denote`, `word_elastic_exchange_denote` |
| checked acceptance accepts exactly the derived impulse | proved | `word_accept_derived_impulse_spec` |
| a successful extracted step denotes `scalarCarrierStep`; the half-flux satisfies `exactHalfFlux` | proved | `word_conservative_density_step_denote`, `word_diffusive_flux_half_denote` |
| the partial kernels really refuse boundary inputs | proved | `word_diffusive_flux_half_odd_refusal`, `word_conservative_density_step_overflow_refusal` |
| the finite conservative total is conserved over the extracted step | proved | `word_conservative_total_mass` |
| a successful extracted three-cell line step denotes `scalarCarrierStep` per cell with exact-half interior faces | proved | `word_diffusive_line3_step_denote` |
| a successful line step changes the denoted total only by the sources; zero source conserves it | proved | `word_diffusive_line3_step_total_mass`, `word_diffusive_line3_step_total_mass_zero_source` |
| odd or unrepresentable line inputs refuse the whole line, not a partial result | proved | `word_diffusive_line3_step_odd_refusal`, `word_diffusive_line3_step_unrepresentable_refusal` |
| the all-zero line succeeds, so the line's success branch is inhabited | proved | `word_diffusive_line3_step_zero_success` |
| nominal and boundary Rust behavior on the production path | tested | `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs`, `tests/the_diffusive_line3_step_is_exact.rs`, `tests/the_exchange_is_a_swap.rs` |
| extraction freshness from the committed sources | tested (tool gate) | `proof/regen.sh --check` |

### Non-goals

- Aeneas refinement of `map`/`table` storage, SIMD, threads or `MaybeUninit`.
- Any property of rustc/LLVM code generation, machine code, the kernel ABI or a GPU backend.
- Totality of `conservative_density_step` or `diffusive_flux_half`.
- A physical mass law, a calibrated viscosity, a time step, a unit or a water solver.
- Transfer of the real-valued nonnegativity, kinetic-energy or classification reading to `Z/2^64`.
- Importing the whole corpus into the bridge; the bridge imports the `AnyRing` closure and the scalar-step links.
- The corpus native toolchain/manifest rebuild; the bridge supplies Mathlib `v4.31.0`.

### Trust boundary

The evidence covers safe, source-level Rust as translated by the pinned Charon/Aeneas,
its Lean denotation and the corpus declarations it refines. It does not cover the
Aeneas `Std` wrapping/equality models beyond their stated theorems, rustc/LLVM
lowering, the generated machine code, the `unsafe` container, the kernel ABI or any
GPU/SPIR-V backend. The corpus sibling path is a development convenience, not a
release pin; release immutability still needs a pinned archive or a vendored corpus.

### Open questions

- Pinned archive or vendored corpus for a reproducible release, owner: corpus owner.
- Stage 5's production integration and float projection remain tested, not refined; the fixed-line
  carrier realization proof is closed by W7, while the live mapping obligations are separate work.
- No energy statement transfers to the word ring; the corpus exact quarter-energy identity is over
  `ℝ` and the any-ring line theorem claims none.

## Specification — Carrier and word refinement

- **Owner:** Lucas 'Paperboy' Rose-Winters (operator: `deepseek-flash`)
- **Intent:** `docs/epics/carrier-refinement.md`
- **Source revision:** `d7121ac0432a4842aff69e63cfc5486d56c4dd73`; working tree clean
- **Status:** approved; epic complete

### Scope and admissible domain

The epic specifies the `i64` instantiation of the crate's ring-generic kernels and
its refinement to the canonical corpus, plus the point readout and the scalar
carrier step. Every operation below is on the production path the library builds.

| Item | Value |
|---|---|
| Operations | `Cell::<i64>::mul`, `Cell::<i64>::norm_form`; `carrier::{point_density, point_flux, conservative_density_step, diffusive_flux_half, diffusive_line3_step}`; `exchange::{exchange_impulse, elastic_exchange, accept_derived_impulse}` |
| Domain | every `Cell<i64>` (`a`, `b` any bit pattern), every `Sigma` (`Elliptic`/`Parabolic`/`Hyperbolic`), every `i64` sample, every `DensityLine3` triple |
| Totality | `mul`, `norm_form`, `point_density`, `point_flux`, `exchange_impulse`, `elastic_exchange` total for a lawful `Ring`; `conservative_density_step`, `diffusive_flux_half`, `diffusive_line3_step`, `accept_derived_impulse` return `Option` |
| Preconditions | none on the total kernels; the partial kernels refuse rather than precondition |
| Postcondition | the extracted body equals its canonical corpus declaration under `denote : I64 → ZMod (2^64)` |

`Ring` is a behavioral contract, not a type-level proof. A partial implementation
such as `Dyadic` can panic when a result is unrepresentable; the generic identities
are therefore conditioned on lawful, successful operations, and only the `i64`
wrapping execution is extracted here.

### Representation relation (Rust to Lean)

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `Cell::<R>::mul`, `norm_form` | `src/cell.rs` |
| Production Rust | `Ring for i64` | `src/scalar.rs` |
| Production Rust | `carrier::{point_density, point_flux}` | `src/carrier.rs` |
| Production Rust | `carrier::{conservative_density_step, diffusive_flux_half, diffusive_line3_step}`, `carrier::DensityLine3` | `src/carrier.rs` |
| Production Rust | `exchange::{exchange_impulse, elastic_exchange, accept_derived_impulse}` | `src/exchange.rs` |
| Extraction wrapper | `word_mul`, `word_norm`, `word_point_density`, `word_point_flux`, `word_conservative_density_step`, `word_diffusive_flux_half`, `word_diffusive_line3_step`, `word_exchange_impulse`, `word_elastic_exchange`, `word_accept_derived_impulse` | `proof/extract/lib.rs` |
| Extracted body | generated `cell`/`carrier`/`exchange` definitions | `proof/ApeironWord.lean` (generated) |
| Denotation | `denote : I64 → ZMod (2^64)` reading the bit pattern; `denoteSigma := denote ∘ sigmaI64` | `proof/Denotation.lean`, `proof/corpus/WordRefinement.lean` |
| Canonical algebra | `Apeiron.Trichotomy.AnyRing.ringMul`, `ringNormForm`, `ringNormForm_mul`, `ringMul_conj` | corpus `Apeiron/Trichotomy/AnyRing.lean` |
| Canonical readout | `Apeiron.Physics.pointDensity`, `pointFlux`, `pointDensity_eq`, `pointFlux_eq` | corpus `Apeiron/Physics/CarrierAnyRing.lean` |
| Canonical exchange | `Apeiron.Physics.ringExchangeImpulse`, `ringCarrierMomentumAfter`, `ringBodyMomentumAfter`, `ringNormFormLedger` | corpus `Apeiron/Physics/CarrierBodyWorkAnyRing.lean` |
| Canonical scalar step | `Apeiron.Physics.scalarCarrierStep`, `exactHalfFlux` | corpus `Apeiron/Physics/CarrierScalarKernelAnyRing.lean` |
| Canonical three-cell line | `Apeiron.Physics.diffusiveUpdate_half_three_anyRing` | corpus `Apeiron/Physics/CarrierScalarKernelAnyRing.lean` |
| Real energy (scope only) | `Apeiron.Physics.dirichletEnergy_halfStep_three`, `dirichletEnergy_halfStep_three_lt` | corpus `Apeiron/Physics/CarrierDissipation.lean` |
| Refinement theorems | `word_*_denote`, `word_*` transfers | `proof/corpus/WordRefinement.lean` |

Each wrapper fixes a concrete type and calls the real body. The generated
`cell`, `carrier` and `exchange` definitions are the production bodies, not
restatements. The corpus pin is `d9aefc93522e2109bc96158526de6dbd52477276`
(`d9aefc93`), committed and pushed, requiring Lean `4.31.0` and Mathlib `v4.31.0`.

### Contracts

| Contract | Statement | Grade target | Evidence |
|---|---|---|---|
| word multiplication | extracted `mul` = `ringMul` at `ZMod (2^64)` for all `Cell`/`Sigma` | proved | `word_mul_denote` |
| word norm form | extracted `norm_form` = `ringNormForm` | proved | `word_norm_denote` |
| multiplicativity transfer | `N(zw) = N(z)N(w)` in the word ring | proved | `word_norm_mul` |
| conjugation transfer | `conj(zw) = conj(w)conj(z)` | proved | `word_mul_conj` |
| point density | extracted `point_density` = `pointDensity` at `denoteSigma s` | proved | `word_point_density_denote` |
| point flux | extracted `point_flux` = `pointFlux` at `denoteSigma s` | proved | `word_point_flux_denote` |
| unit-direction readout | density/flux at the unit direction are dial-independent | proved | `word_point_density_ringOne_independent`, `word_point_flux_ringOne_independent` |
| exchange impulse | extracted `exchange_impulse` = `ringExchangeImpulse` | proved | `word_exchange_impulse_denote` |
| exchange result | extracted `elastic_exchange` = canonical impulse, carrier and body fields | proved | `word_elastic_exchange_denote` |
| exchange conservation | carrier/body swap and total basis momentum unchanged | proved | `word_elastic_exchange_carrier`, `word_elastic_exchange_totalMomentum` |
| ledger conservation | `ringNormFormLedger` unchanged at every symbolic dial | proved | `word_elastic_exchange_normFormLedger` |
| checked acceptance | `accept_derived_impulse` returns `some` exactly when the claim equals the derived impulse | proved | `word_accept_derived_impulse_spec` |
| conservative step | a successful step denotes `scalarCarrierStep`; success iff all three checked intermediates are `some` | proved | `word_conservative_density_step_denote`, `word_conservative_density_step_eq_some_iff` |
| half flux | a successful half-flux satisfies `exactHalfFlux` (`2 * f = left - right`) | proved | `word_diffusive_flux_half_denote`, `word_diffusive_flux_half_success` |
| finite total | the denoted `ZMod (2^64)` sum over `n` successful cells is conserved | proved | `word_conservative_total_mass`, `word_conservative_total_mass_closed` |
| three-cell line | a successful extracted line step has exact-half interior faces and each component is `scalarCarrierStep` at the shared face and closed end | proved | `word_diffusive_line3_step_denote` |
| three-cell total | a successful line step changes the denoted total only by the denoted sources; zero source conserves it | proved | `word_diffusive_line3_step_total_mass`, `word_diffusive_line3_step_total_mass_zero_source` |
| three-cell refusal | an odd or unrepresentable interior difference refuses the whole line, not a partial result | proved | `word_diffusive_line3_step_odd_refusal`, `word_diffusive_line3_step_unrepresentable_refusal` |
| three-cell success witness | the all-zero old/source line returns the all-zero line as `some`, so the partial kernel is not vacuously total | proved | `word_diffusive_line3_step_zero_success` |
| line energy | no energy statement transfers to the word ring; the corpus quarter-energy identity is over `ℝ` | anchor (scope) | `dirichletEnergy_halfStep_three`, `dirichletEnergy_halfStep_three_lt` |
| nominal Rust behavior | readout, step, line and exchange on the production path | tested | `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs`, `tests/the_diffusive_line3_step_is_exact.rs`, `tests/the_exchange_is_a_swap.rs` |

### Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| `outgoing - incoming` unrepresentable | `ok none` | first `checked_sub` in `conservative_density_step` |
| `density - net` unrepresentable | `ok none` | second `checked_sub` |
| `after + source` unrepresentable | `ok none` | final `checked_add` |
| `right - left` unrepresentable | `ok none` | `checked_sub` in `diffusive_flux_half` |
| face difference odd | `ok none` | `checked_rem(2) != 0`; refused, not rounded |
| line interior difference odd | whole line `ok none` | first refusing `diffusive_flux_half` in `diffusive_line3_step`; no partial line |
| line cell intermediate unrepresentable | whole line `ok none` | first refusing `conservative_density_step` in `diffusive_line3_step` |
| claim ≠ derived impulse | `ok none` | `accept_derived_impulse` equality branch |
| `Dyadic` result unrepresentable | panic | `Dyadic` is not a total finite ring; not extracted |

The `?` operator is not used in the two partial carrier kernels because it desugars
through `Try::branch`, which Aeneas has no model for; the source uses explicit
`match` with early `return None`.

### Evidence plan

1. **Positive tests.** `cargo test --workspace --all-features --locked` covers the nominal path and the boundaries above; `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs` and `tests/the_exchange_is_a_swap.rs` name them.
2. **Proof.** `proof/regen.sh --check` re-extracts from the committed sources and fails on drift; `cd proof/corpus && lake build` checks every refinement theorem.
3. **Axiom cone.** `proof/corpus/AxiomAudit.lean` `#audit`s each headline; anything outside `{propext, Classical.choice, Quot.sound}` fails the build.
4. **Negative control.** Each proof family has a mutation in a throwaway copy that must turn the gate red and a restore that must pass; the controls are listed in `docs/epics/carrier-refinement.md` and `proof/README.md`.
5. **Measurement.** Not applicable; this epic makes no performance claim.

### Trust boundary

The specification and its evidence cover safe source-level Rust, its Aeneas
translation and the corpus declarations it refines. They do not cover the Aeneas
`Std` models beyond their stated theorems, rustc/LLVM lowering, generated machine
code, the `unsafe` container, the kernel ABI or any GPU backend. The corpus sibling
path is a development path, not a reproducible release pin.

### Open questions

- Pinned archive or vendored corpus for release immutability; owner: corpus owner.
- Stage 5's production integration of the fixed three-cell line and the live float-projection
  mapping remain open. The fixed-line realization/refinement proof itself is closed by W7 of this
  epic (`word_diffusive_line3_step_*`); the corpus any-ring line theorem and the real quarter-energy
  identity are links and scope, not transferred word claims.

## Plan — Carrier and word refinement

- **Owner:** Lucas 'Paperboy' Rose-Winters (operator: `deepseek-flash`)
- **Intent:** `docs/epics/carrier-refinement.md`
- **Spec:** `docs/epics/carrier-refinement.md`
- **Source revision:** `d7121ac0432a4842aff69e63cfc5486d56c4dd73`; working tree clean
- **Status:** approved; all waves complete, W7 committed with the fixed three-cell line and its success/unrepresentable witnesses

### Goal and completion condition

Every task gate is green with its negative control run and restored. The barrier is
the fresh extraction plus the bridge build with a closed axiom cone.

```sh
cargo test --workspace --all-features --locked
cargo clippy --workspace --all-features --all-targets --locked -- -D warnings
cargo fmt --all -- --check
proof/regen.sh --check
cd proof/corpus && lake build
```

### Non-goals

- Container storage refinement, rustc/LLVM or GPU evidence, totality of the partial kernels.
- A physical mass law or a water solver; this epic stops at the exact integer readout.
- A corpus native toolchain/manifest rebuild.

### Wave map

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

### Task blocks

#### W1 · Extraction entry point and denotation

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

#### W2 · Word multiplication and norm refinement

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

#### W3 · Carrier point readout refinement

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

#### W4 · Exchange and checked-acceptance refinement

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

#### W5 · Scalar step, half-flux and finite conservation

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

#### W6 · Corpus pin and carrier lab

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

#### W7 · Fixed three-cell diffusive line

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

### Recorded digests

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

### Barrier checklist

- [x] Every task gate exits 0, with its negative control run and restored. `proof/regen.sh --check` reported `extraction is fresh` (exit 0); `cd proof/corpus && lake build` exited 0.
- [x] No task touched a file owned by another task; the refinement files were one lane.
- [x] `cargo test --workspace --all-features --locked` exit 0 (173 including 9 doc-tests; the nine `tests/the_diffusive_line3_step_is_exact.rs` cases are W7); `cargo clippy … -D warnings` exit 0; `cargo fmt --all -- --check` exit 0; `cargo kani` exit 0 (60 harnesses).
- [x] `proof/regen.sh --check` fresh (1708 jobs); `proof/corpus` `lake build` passed (3276 jobs on the operator run), rebuilt at corpus `d9aefc93`.
- [x] Source and tool digests recorded above.
- [x] Claims registered at their grade; the corpus declarations are cited as links, not called verified by proximity.
- [x] Trust boundary stated in the intent and spec.

### Stop and rollback conditions

- A generated `axiom`/`sorry`/`admit` in the accepted chain stops the wave; `regen.sh` fails on it rather than admitting it.
- A mutation that does not turn its gate red means the check has no weight; report the theorem and stop.
- After three materially different failed approaches to one proof, stop and report the smallest unresolved goal.
