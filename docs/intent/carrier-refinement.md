# Intent — Carrier and word refinement

- **Owner:** Lucas 'Paperboy' Rose-Winters (operator: `deepseek-flash`)
- **Source revision:** `5ffd2521a128e749568ec4b29b3c93ab55c916f4`; working tree clean
- **Status:** approved; epic complete, evidence in `docs/plan/carrier-refinement.md`
- **Spec:** `docs/spec/carrier-refinement.md`
- **Plan:** `docs/plan/carrier-refinement.md`

## Problem

The Apeiron Rust crate's `Cell<i64>` arithmetic and its carrier readout have no
machine-checked link to the canonical Lean corpus. The word ring `Z/2^64` is not the
real trichotomy the corpus states, so `Apeiron.Trichotomy.AnyRing.ringMul`,
`ringNormForm`, `Apeiron.Physics.CarrierAnyRing.pointDensity`/`pointFlux`,
`Apeiron.Physics.CarrierBodyWorkAnyRing` and `Apeiron.Physics.scalarCarrierStep` do
not transfer to the extracted `i64` bodies by name. Tests and the bounded Kani
harnesses in `src/proofs.rs` execute the production path but prove no full-width
refinement.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Kani at full `i64` width | the degree-four norm-form identity bit-blasts past twenty-five minutes at `i8`; a 4-bit window is not full width |
| `cargo test` against a `std` oracle | executes the path but relates no extracted body to a corpus declaration |
| a hand-written Lean restatement of `mul`/`norm_form` | not the production body; a citation, not a refinement |
| a corpus source rewrite to a new algebra | the corpus already states the ring-generic laws; only the pin and the extraction were missing |
| calling a readout identity a physical law | a readout identity is arithmetic, not a material model |

## Completion condition

The extraction is fresh from the committed sources and the bridge build checks every
headline theorem inside the closed axiom cone.

```sh
proof/regen.sh --check
cd proof/corpus && lake build
```

## Invariants

1. Every wrapper in `proof/extract/lib.rs` calls the real production body; none restates a formula.
2. The partial kernels keep their `Option` signatures and refuse rather than wrap or round.
3. Every headline theorem's transitive axiom cone is within `{propext, Classical.choice, Quot.sound}`.
4. The corpus revision and every extracted source digest are recorded with the evidence.
5. No theorem narrows its domain to close a proof; refusal witnesses show partiality is real.

## Verification hypotheses

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
| odd or unrepresentable line inputs refuse the whole line, not a partial result | proved | `word_diffusive_line3_step_odd_refusal` |
| nominal and boundary Rust behavior on the production path | tested | `tests/the_carrier_readout_is_generic.rs`, `tests/the_carrier_step_is_conservative.rs`, `tests/the_diffusive_line3_step_is_exact.rs`, `tests/the_exchange_is_a_swap.rs` |
| extraction freshness from the committed sources | tested (tool gate) | `proof/regen.sh --check` |

## Non-goals

- Aeneas refinement of `map`/`table` storage, SIMD, threads or `MaybeUninit`.
- Any property of rustc/LLVM code generation, machine code, the kernel ABI or a GPU backend.
- Totality of `conservative_density_step` or `diffusive_flux_half`.
- A physical mass law, a calibrated viscosity, a time step, a unit or a water solver.
- Transfer of the real-valued nonnegativity, kinetic-energy or classification reading to `Z/2^64`.
- Importing the whole corpus into the bridge; the bridge imports the `AnyRing` closure and the scalar-step links.
- The corpus native toolchain/manifest rebuild; the bridge supplies Mathlib `v4.31.0`.

## Trust boundary

The evidence covers safe, source-level Rust as translated by the pinned Charon/Aeneas,
its Lean denotation and the corpus declarations it refines. It does not cover the
Aeneas `Std` wrapping/equality models beyond their stated theorems, rustc/LLVM
lowering, the generated machine code, the `unsafe` container, the kernel ABI or any
GPU/SPIR-V backend. The corpus sibling path is a development convenience, not a
release pin; release immutability still needs a pinned archive or a vendored corpus.

## Open questions

- Pinned archive or vendored corpus for a reproducible release, owner: corpus owner.
- Stage 5's production integration and float projection remain tested, not refined; the fixed-line
  carrier realization proof is closed by W7, while the live mapping obligations are separate work.
- No energy statement transfers to the word ring; the corpus exact quarter-energy identity is over
  `ℝ` and the any-ring line theorem claims none.
