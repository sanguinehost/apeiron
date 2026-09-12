# Specification — Carrier and word refinement

- **Owner:** Lucas 'Paperboy' Rose-Winters (operator: `deepseek-flash`)
- **Intent:** `docs/intent/carrier-refinement.md`
- **Source revision:** `5ffd2521a128e749568ec4b29b3c93ab55c916f4`; working tree clean
- **Status:** approved; epic complete

## Scope and admissible domain

The epic specifies the `i64` instantiation of the crate's ring-generic kernels and
its refinement to the canonical corpus, plus the point readout and the scalar
carrier step. Every operation below is on the production path the library builds.

| Item | Value |
|---|---|
| Operations | `Cell::<i64>::mul`, `Cell::<i64>::norm_form`; `carrier::{point_density, point_flux, conservative_density_step, diffusive_flux_half}`; `exchange::{exchange_impulse, elastic_exchange, accept_derived_impulse}` |
| Domain | every `Cell<i64>` (`a`, `b` any bit pattern), every `Sigma` (`Elliptic`/`Parabolic`/`Hyperbolic`), every `i64` sample |
| Totality | `mul`, `norm_form`, `point_density`, `point_flux`, `exchange_impulse`, `elastic_exchange` total for a lawful `Ring`; `conservative_density_step`, `diffusive_flux_half`, `accept_derived_impulse` return `Option` |
| Preconditions | none on the total kernels; the partial kernels refuse rather than precondition |
| Postcondition | the extracted body equals its canonical corpus declaration under `denote : I64 → ZMod (2^64)` |

`Ring` is a behavioral contract, not a type-level proof. A partial implementation
such as `Dyadic` can panic when a result is unrepresentable; the generic identities
are therefore conditioned on lawful, successful operations, and only the `i64`
wrapping execution is extracted here.

## Representation relation (Rust to Lean)

| Layer | Symbol | Path |
|---|---|---|
| Production Rust | `Cell::<R>::mul`, `norm_form` | `src/cell.rs` |
| Production Rust | `Ring for i64` | `src/scalar.rs` |
| Production Rust | `carrier::{point_density, point_flux}` | `src/carrier.rs` |
| Production Rust | `carrier::{conservative_density_step, diffusive_flux_half}` | `src/carrier.rs` |
| Production Rust | `exchange::{exchange_impulse, elastic_exchange, accept_derived_impulse}` | `src/exchange.rs` |
| Extraction wrapper | `word_mul`, `word_norm`, `word_point_density`, `word_point_flux`, `word_conservative_density_step`, `word_diffusive_flux_half`, `word_exchange_impulse`, `word_elastic_exchange`, `word_accept_derived_impulse` | `proof/extract/lib.rs` |
| Extracted body | generated `cell`/`carrier`/`exchange` definitions | `proof/ApeironWord.lean` (generated) |
| Denotation | `denote : I64 → ZMod (2^64)` reading the bit pattern; `denoteSigma := denote ∘ sigmaI64` | `proof/Denotation.lean`, `proof/corpus/WordRefinement.lean` |
| Canonical algebra | `Apeiron.Trichotomy.AnyRing.ringMul`, `ringNormForm`, `ringNormForm_mul`, `ringMul_conj` | corpus `Apeiron/Trichotomy/AnyRing.lean` |
| Canonical readout | `Apeiron.Physics.pointDensity`, `pointFlux`, `pointDensity_eq`, `pointFlux_eq` | corpus `Apeiron/Physics/CarrierAnyRing.lean` |
| Canonical exchange | `Apeiron.Physics.ringExchangeImpulse`, `ringCarrierMomentumAfter`, `ringBodyMomentumAfter`, `ringNormFormLedger` | corpus `Apeiron/Physics/CarrierBodyWorkAnyRing.lean` |
| Canonical scalar step | `Apeiron.Physics.scalarCarrierStep`, `exactHalfFlux` | corpus `Apeiron/Physics/CarrierScalarKernelAnyRing.lean` |
| Refinement theorems | `word_*_denote`, `word_*` transfers | `proof/corpus/WordRefinement.lean` |

Each wrapper fixes a concrete type and calls the real body. The generated
`cell`, `carrier` and `exchange` definitions are the production bodies, not
restatements. The corpus pin is `16112030781eeec022139cdae07407131acc0915`
(`16112030`), committed and pushed, requiring Lean `4.31.0` and Mathlib `v4.31.0`.

## Contracts

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
| nominal Rust behavior | readout, step and exchange on the production path | tested | `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs`, `tests/the_exchange_is_a_swap.rs` |

## Failure semantics

| Input class | Result | Named condition |
|---|---|---|
| `outgoing - incoming` unrepresentable | `ok none` | first `checked_sub` in `conservative_density_step` |
| `density - net` unrepresentable | `ok none` | second `checked_sub` |
| `after + source` unrepresentable | `ok none` | final `checked_add` |
| `right - left` unrepresentable | `ok none` | `checked_sub` in `diffusive_flux_half` |
| face difference odd | `ok none` | `checked_rem(2) != 0`; refused, not rounded |
| claim ≠ derived impulse | `ok none` | `accept_derived_impulse` equality branch |
| `Dyadic` result unrepresentable | panic | `Dyadic` is not a total finite ring; not extracted |

The `?` operator is not used in the two partial carrier kernels because it desugars
through `Try::branch`, which Aeneas has no model for; the source uses explicit
`match` with early `return None`.

## Evidence plan

1. **Positive tests.** `cargo test --workspace --all-features --locked` covers the nominal path and the boundaries above; `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs` and `tests/the_exchange_is_a_swap.rs` name them.
2. **Proof.** `proof/regen.sh --check` re-extracts from the committed sources and fails on drift; `cd proof/corpus && lake build` checks every refinement theorem.
3. **Axiom cone.** `proof/corpus/AxiomAudit.lean` `#audit`s each headline; anything outside `{propext, Classical.choice, Quot.sound}` fails the build.
4. **Negative control.** Each proof family has a mutation in a throwaway copy that must turn the gate red and a restore that must pass; the controls are listed in `docs/plan/carrier-refinement.md` and `proof/README.md`.
5. **Measurement.** Not applicable; this epic makes no performance claim.

## Trust boundary

The specification and its evidence cover safe source-level Rust, its Aeneas
translation and the corpus declarations it refines. They do not cover the Aeneas
`Std` models beyond their stated theorems, rustc/LLVM lowering, generated machine
code, the `unsafe` container, the kernel ABI or any GPU backend. The corpus sibling
path is a development path, not a reproducible release pin.

## Open questions

- Pinned archive or vendored corpus for release immutability; owner: corpus owner.
- Stage 5 production integration, the fixed-line carrier realization proof and the live
  float-projection mapping remain open and are not covered by this epic.
