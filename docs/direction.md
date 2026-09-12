# Apeiron: Consolidated Direction

2026-09-11. Recommendation following the direction review, OpenCode session
`ses_f7a685f53ffeAg9eprerWoNyHd`, source inspection and upstream fetches.
This is a direction and evidence record, not a claim that the migration is complete.

Execution order and specialist approval gates: [Flash operator handoff](flash-handoff.md).

## Decision

**One Apeiron project: a canonical mathematical specification, its verified Rust
implementation, and independently checked consumers. Penglai is the immediate
application. Song is the sibling language/runtime instantiation, not a prerequisite
for developing the Rust game. Fold consumes the substrate; it does not define it.**

```text
                 Canonical Apeiron Lean corpus
                    /                    \
             Song / Slang          Rust Apeiron + Aeneas
                    <--- shared contracts --->
                      \                  /  \
                       Sovereign AI stack    Penglai -> Lyrium
                             ^
                      Fold / modelfold
```

The old Sanguine corpus is migration input, not a second continuing source of
authority. Re-derive its useful results under the new root, preserving actual
hypotheses and provenance. Do not copy the old import hierarchy or relabel a
split-complex specialization as a theorem about every base ring.

## Repository Convergence

The recommended final home is `apeiron`: Rust sources plus the canonical Lean
corpus and Rust refinement proofs. Keep separate Lake packages for the mathematical
specification and Aeneas implementation proofs, so Song can depend on the former
without importing Rust-generated definitions or Aeneas. Keep `Apeiron.*` as the
shared mathematical namespace. Generic mathematical results belong in the corpus;
Rust representation lemmas belong in the refinement package; game-specific adapters
and physical calibration belong in Penglai.

Do not start with a directory merge. First reconcile the actively developed corpus
and establish a revision-pinned Rust-to-corpus bridge. The follow-up push described
below supplies the previously missing revisions.
Then move the canonical corpus with history/provenance, update Song's dependency
and citation resolver in coordination with its owner, and retire the duplicate
active location. During that transition, two repositories are acceptable; two
independently maintained definitions of the same law are not.

Each release must bind Rust source, corpus revision, Charon/Aeneas revision,
Lean/Mathlib pins, extraction options, supported targets and proof artifacts.
Changing the code or its numerical domain invalidates the corresponding evidence.
Local sibling paths are convenient for development, not a reproducible release pin.

### Freshness Check

All configured remotes for the repositories below were fetched, including the
follow-up Sanguine, Song, Whispering Tides and Fold refresh. No existing worktree was
pulled, rebased, switched, committed or pushed. Existing changes were preserved.

| Repository | Checked state |
| --- | --- |
| `apeiron` | `main` and `origin/main` at `d043ad3`; 69 pre-existing modified files and untracked `docs/kernel-boundary.md`. |
| `song` | Local `gen2` at `50f8e47d`; fetched `origin/gen2` at `857e9434`, 77 commits ahead. Reviewed the fetched source and roadmap without moving the active worktree. |
| `sanguine-apeiron` | Local `apeiron-rebuild` at `c7339cce`; fetched upstream `6213271f`, 13 commits ahead. This is a worktree of the same Git repository as `sanguine`, not an independent remote project. |
| `sanguine` | Local `proof/null-transformer` at `eaf22284`, with pre-existing navigation/audit edits. Fetched `origin/main` at `795c7a8f`. |
| `penglai` | Clean `master` and upstream at `1222477`. |
| `fold` | Local `bench/spatial-binning` at `d497161f`, with pre-existing modelfold work. Fetched GitHub main `aca63cbf`, GitLab main `4cc14471`; the mirrors are not identical. |
| `whispering-tides` | Clean local `master` at `2ab6787`; fetched upstream `e62ae9b`, one commit ahead. Retained as the gen1 design/scientific reference. |
| `song-ref` | Detached reference at `ae688f66`; Song upstream main now at `5596c18f` and gen2 at `857e9434`, without replacing the reference checkout. |
| `ascii-venture` | Cloned `paperboygold/ascii-venture` master `e933813` into `/tmp/opencode/ascii-venture` for inspection. |

**Missing revisions resolved by the follow-up push:** `38afaef1` is now confirmed
as an ancestor of `origin/apeiron-rebuild`, and both `Parser/Expression.lean` and
`Parser/Arithmetic.lean` are published. The latter includes exact-suffix soundness,
finite-fuel completeness, token-derived fuel and node-capacity bounds. The models
explicitly exclude production Slang arena/machine refinement. Song's citation
resolver still searches `../sanguine/proof`, not `../sanguine-apeiron/proof`; pinning
and path convergence remain coordination work, but missing publication is no longer
the blocker. The fetched corpus still uses Lean `4.30.0-rc2`.

The new Song roadmap records 65/65 matching contract ledgers and 65/65 matching
selected face surfaces; statement-level effects, universal production refinement
and source-to-binary guarantees remain open. Its compiler memoization work reports
self-compilation improving from 22.1 s to 6.3 s. These are upstream records, not
benchmarks rerun in this review. The useful shared lessons are source identity,
independent semantic controls, exact resource bounds and refusal semantics.

The follow-up `857e9434` fixes statement ingestion by distinguishing top-level,
guard-arm and else/loop-block modes. This advances the statement-level work without
establishing a universal refinement theorem. Compiler commit `5596c18f` is now
published as well, rather than only described by the gen2 roadmap.

New physics in `6213271f` includes `ClassicalWaves.lean`: amplitude-curvature
residue, elliptic quarter-turn structure and an exact four-sample identity for
`normForm`. Its own scope excludes reconstruction of arbitrary smooth amplitudes.
`WaveTime.lean` supplies time-derivative identities and explicitly does not link
time evolution to the spatial second derivative. These are useful inputs to the
carrier experiment, not its completed temporal/material law. This refresh inspected
the declarations and audit additions but did not rebuild the new Lean snapshot.

Sanguine main's new aliasing/provenance modules have corresponding results already
in Apeiron's `Linear/Basic.lean` and `Capability/Basic.lean`; the latter also proves
the converse reachability direction, while the former addresses slot reuse. Do not
duplicate those migrations. Whispering Tides `e62ae9b` corrects citations only:
`bandCount` never existed, and the relief/rainfall interpretation was stronger than
its cited algebra identity. There is no new runtime weather solver in that commit.

## Verification Direction

Use Aeneas for symbolic refinement of the actual safe Rust kernels. Use Lean to
prove their representation relations to canonical definitions and compose existing
theorems. Retain Kani for bounded, bit-level properties where it is effective;
do not make degree-four full-width bit-blasting the algebra's critical path.

The first proof target remains `Cell<i64>::mul` and `norm_form`, interpreted in
`ZMod (2^64)`, linked to `Apeiron.Trichotomy.AnyRing.ringMul` and `ringNormForm`.
Then transfer `ringNormForm_mul` and `ringMul_conj` through the established link.
Require total success, not merely a property conditional on obtaining a result.

### Checks Performed

- `cargo test --workspace --all-features --locked --quiet` passed on the current
  Apeiron worktree. `git diff --check` passed before this document was added.
- Penglai `proof/`: `lake build` passed, reporting 1707 jobs.
- A scratch extraction root imported the actual `src/scalar.rs` and `src/cell.rs`
  by path and exposed two trivial calling wrappers, not rewritten formulas.
- Whole-module Charon/Aeneas translation emitted unknown external axioms for
  hashing, `i128::trailing_zeros` and iterator folding. Lean checking also failed
  at the dyadic `Ord::max` translation and an `Eq` implementation.
- Restricting Charon with `--start-from crate::word_mul` and
  `--start-from crate::word_norm` generated a file without those axioms;
  `lake env lean /tmp/opencode/Apeiron-word.lean` passed. Aeneas reported
  0.227 seconds for this translation stage alone, not extraction-plus-proof time.
- This establishes extraction/elaboration feasibility. The full-width
  representation-to-`AnyRing` theorem has **not** been proved in this review.

Scratch sources and outputs are in `/tmp/opencode/apeiron-direction-*` and
`/tmp/opencode/Apeiron-word.lean`; they are diagnostics, not release artifacts.
The installed tools are Penglai's documented Aeneas `505b6ca` / Charon toolchain.
The corpus currently pins Lean `4.30.0-rc2`, while that backend uses `4.31.0`.
Align them in a coordinated, checked change, not a blind version bump.

**Stage 1 executed (2026-09-11).** The scratch probe is now a maintained entry point in
`proof/` (`proof/README.md`): `proof/extract/lib.rs` includes the real `src/scalar.rs` and
`src/cell.rs` by `#[path]`, `proof/regen.sh` regenerates and audits `ApeironWord.lean`, and
`proof/regen.sh --check` fails on drift or on an `axiom`/`sorry`/`admit` in the accepted chain.
Ran against source `scalar.rs` `285bf66e…` and `cell.rs` `8cfc787a…` with Aeneas
`nightly-2026.09.09-505b6ca` / Charon `0.1.254` / Lean `4.31.0`: extraction fresh and
`lake build` passed (1707 jobs). `cargo test --workspace --all-features --locked` passed
(134 tests, exit 0). This is extraction and elaboration, **not** the stage-2 refinement theorem.

**Stage 2 executed (2026-09-11).** `proof/corpus/` (package `apeironWordBridge`) now proves the
full-width refinement: `word_mul_denote` and `word_norm_denote` equate the extracted `Cell<i64>`
operations with `Apeiron.Trichotomy.AnyRing.ringMul`/`ringNormForm` instantiated at
`ZMod (2^64)`, and `word_norm_mul`/`word_mul_conj` transfer `ringNormForm_mul`/`ringMul_conj`.
All are total `ok` equalities, no `sorry`, no reduced domain, `lake build` 2609 jobs. The axiom
audit is now the standard `propext`/`Classical.choice`/`Quot.sound` cone alone: `Ring` no longer
requires `Debug` (`src/scalar.rs`), so `Aeneas.Std.core.fmt.Formatter` and the generated debug
instance are gone from the extraction. A negative control that changes the dial's
hyperbolic value from `1` to `2` and re-extracts breaks every theorem (`sigma_value_eq` and the
four multiplication/norm theorems; the `rfl`/`ring` cases fail); restoring the source (`cell.rs`
back to `8cfc787a…`) passes.

The corpus/backend incompatibility is now characterised: the corpus's pinned 4.30.0-rc2 Mathlib
does not build on Lean 4.31.0 (`Cache/IO.lean:506`, `Lake.Glob.ofString?`), but the corpus
**source** compiles cleanly against Mathlib `v4.31.0` (the bridge requires it last so its
transitive versions win). The coordinated change is a corpus Mathlib/toolchain bump, not a source
rewrite. *(Historical at the time of this Stage 2 entry: the bridge read the local corpus worktree
`c7339cce` via a sibling path and `6213271f` was the intended pin. The pin landed at `16112030`
and has since advanced to `d9aefc93`; see Stage 4 below.)*

**Stage 3 slice 1 executed (2026-09-11).** The bounded-dyadic contract is now honest.
`src/scalar.rs` gained `Dyadic::checked_add/checked_sub/checked_mul/checked_halve`, which report
unrepresentable alignment, mantissa, scale and halving results as `None`; `aligned`'s silent
`<<` was replaced by `checked_shl_exact`, which rejects any shift that loses a bit or exceeds the
width. `impl Ring for Dyadic` and `impl Halve for Dyadic` are now explicitly partial and panic
rather than returning a value with the overflow already gone; `README.md` and the type docs were
corrected. The previously reproduced `Dyadic::new(1,127).add(Dyadic::whole(2)) == Dyadic::new(1,127)`
is fixed. `cargo test --workspace --all-features --locked` passes (137 tests, up from 134). The
extraction was regenerated from the new `src/scalar.rs` (`285bf66e…`) and the stage-2 bridge
re-verified (`lake build`, 2609 jobs) — the `Ring for i64` kernel is unchanged. Slice 3
(identity/region contracts) remains open.

**Stage 3 slice 2 executed (2026-09-11).** The false reassociation permission is gone.
`src/merge.rs` no longer implements the associative `Monoid` for `f32`/`f64`: the counterexample
`(1e20 + -1e20) + 1 = 1` vs `1e20 + (-1e20 + 1) = 0` is now a regression test that the exact
integer/bool monoids satisfy and the floats are excluded from. Floats get a separate
`OrderedMonoid` trait and `approx_count`/`approx_count_into` merge functions, documented as
`Species::Ordered`, so a count over them cannot be justified by an associativity contract the
arithmetic does not meet; `count`/`count_into` remain bound to `Monoid`. `cargo test --workspace
--all-features --locked` passes (139 tests). Clippy's pedantic `float_cmp` on the deliberately
exact counterexample is silenced with a reasoned `#[expect]`. This does not affect the extracted
kernel, so the stage-2 evidence is unchanged.

**Stage 3 slice 3 executed (2026-09-11).** The scaled channel map is separated from the idempotent
projection. `src/keystore.rs` keeps `after_storing` for any ring and documents it as a one-shot
scale-2 map `(a, b) -> (a+b, a+b)` that is **not** idempotent, and adds
`after_storing_exact<R: Halve>` using the genuine idempotent `e+ = (1/2, 1/2)`. New regressions in
`tests/three_and_one.rs` pin the word form's doubling `(3,1) -> (4,4) -> (8,8)` and the exact
form's idempotence with the lower lane destroyed. `cargo test --workspace --all-features --locked`
passes (141 tests). Packing round-trips were re-checked and are already covered at the type
extremes in `tests/each_face_gets_the_fold_it_needs.rs`, so no fix was needed there.

**Stage 3 slice 4 executed (2026-09-11) — container scope, canonical order, failure outcomes.**
- **Explicit capacity failure.** `try_reserve` no longer panics on an unrepresentable request:
  `live + additional` is checked and `region_for`'s doubling saturates (with a private
  `region_for_checked` for the fallible path), so an impossible capacity returns
  `AllocFailed { slots: usize::MAX }` with the map untouched. Infallible `reserve`/`with_capacity`
  saturate and fail at the allocation rather than in the arithmetic. Regression:
  `tests/a_map_is_a_map.rs::an_impossible_reserve_reports_instead_of_panicking` (it panicked at
  `src/map.rs:1966` before the fix).
- **Item 5, container claims scoped.** `src/map.rs` now carries a "Verification scope" note, and
  `docs/kernel-boundary.md` separates the Lean-proved algebra and Aeneas-refined `Cell<i64>` kernel
  from the storage layer, which is **tested** against a `std` oracle, under Miri and by bounded Kani
  harnesses, with no Aeneas storage refinement claimed.
- **Item 6, canonical observation.** Raw iteration is placement-dependent (open addressing), so
  replay must observe a canonical order, not the table's own. A test pins sorted-key observation as
  insertion-order/placement independent while documenting that raw iteration is not;
  `examples/determinism.rs` fingerprints the layout across processes and a recorded pepper remains
  the replay recipe.
- **Modular words vs physical magnitudes (direction item 1).** `README.md` no longer claims the
  real trichotomy's *classification* transfers to `i64`: it now states that `Z/2^64` has only the
  idempotents `0`/`1`, a nonzero nilpotent at every dial (a zero divisor on the elliptic face the
  real ring calls a field) and no `Discards` projection at any dial, matching the harnesses in
  `src/proofs.rs` and `Trichotomy/AnyRing.lean:13-46`, while retaining the structural laws that do
  hold. This closes the review's item 1.
- **Identity/region evidence (handoff Stage 3.3).** Each required operation has its own test:
  packing round trips (`each_face_gets_the_fold_it_needs::a_packed_key_round_trips_exactly`),
  negative coordinates handled by sign-extension and spread (`coord_is_injective`),
  narrow-coordinate bounds (`a_narrow_coordinate_refuses_an_out_of_range_value`), collision-safe
  identity (`the_arithmetic_fold_is_not_a_bijection_and_here_is_the_collision`,
  `the_address_is_the_pair`), get/insert/remove and merge order against `std` (`a_map_is_a_map`),
  and capacity failure (`an_impossible_reserve_reports_instead_of_panicking`). Literal Euclidean
  division is absent because the container is addressed by key words, not divided coordinate grids.
- `cargo test --workspace --all-features --locked` passes (143 tests); clippy and rustfmt clean on
  every changed file. Negative-coordinate Euclidean division is not an apeiron operation — the
  container is addressed by key words, not by dividing coordinate grids — so it belongs to the
  carrier's spatial index and is deferred with the physics work.

**Stage 3 complete (2026-09-11).** All four slices above are executed and checked; the review's
correctness items 1–6 are closed (item 1 by the modular-word/README correction, items 2–4 by the
dyadic, reduction and projection slices, items 5–6 by the container-scope and canonical-order
slices). `cargo test --workspace --all-features --locked` passes (143 tests), clippy and rustfmt
are clean on every changed file, and `proof/regen.sh --check` is fresh.

**Debug-from-Ring cone reduction done (2026-09-11).** `Ring` no longer has `Debug` as a supertrait,
so the Aeneas extraction carries no `core::fmt` instance. Regenerated (`src/scalar.rs` now
`9199e955…`) and rebuilt: the bridge reports all six theorems depend only on
`propext`/`Classical.choice`/`Quot.sound`, and the positive/negative mutation control still holds.
`cargo check --workspace --all-features --locked --all-targets` passes.

**Corpus Mathlib pin bump — resolved at `16112030` (the proposal below is retained as the
historical record).** The bump landed in corpus commits `1d7c4aa6` (Lean `4.31.0`, Mathlib
`v4.31.0`) and `33bc96f6` (the `CarrierScalarKernelAnyRing` links). The corpus revision
`16112030781eeec022139cdae07407131acc0915` is committed and pushed; the bridge requires the same
Mathlib `v4.31.0` last, so one revision resolves. The pin has since advanced to `d9aefc93` (see
Stage 4 complete). The `Flow.lean` source break named below was
migrated before the pin landed. The proposal text is historical:

> **Historical proposal (corpus-owner sign-off required).** `../sanguine-apeiron` is
> clean at `c7339cce`. Its `proof/lean-toolchain` is `leanprover/lean4:v4.30.0-rc2` and
> `proof/lake-manifest.json` pins Mathlib `89d44072ff8960893f469c67f75280db4fe4023a` (inputRev
> `master`, with `aesop`/`Qq`/`Cli` at `v4.30.0-rc2`). That manifest does not build on the 4.31.0
> Aeneas backend (`Cache/IO.lean:506`, `Lake.Glob.ofString?` absent). Source compatibility is already
> verified: `proof/corpus/lakefile.lean` requires `mathlib @ v4.31.0` last and the `AnyRing` closure
> plus the bridge build with zero errors. The coordinated change is therefore bookkeeping, not a
> source rewrite:

> 1. `proof/lean-toolchain` -> `leanprover/lean4:v4.31.0`.
> 2. On a network-enabled machine: `cd proof && lake update`, to regenerate `lake-manifest.json`
>    (Mathlib `v4.31.0`, tag commit `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`, and the matching
>    `aesop`/`Qq`/`Cli`/`batteries`/`proofwidgets` revisions).
> 3. `lake build` and reverify the bridge against the new pin.
>
> This is a corpus-repo commit eventually consumed by Song; it is **not** applied to either corpus
> worktree from this Apeiron workstream, and Song's resolver still points at `../sanguine/proof`.
> A first 4.31.0 build of the corpus source found at least one **source** break beyond the manifest:
> `Trichotomy/Flow.lean:310` fails to elaborate, so the bump also needs a migration of the analysis
> files, not just the toolchain line.

**Stage 4 complete.** The corpus pin, the extraction and the bridge are done. The pin advanced from
`16112030` to `d9aefc93` (Lean `4.31.0`, Mathlib `v4.31.0`), which supplies
`Apeiron.Physics.CarrierAnyRing`, `CarrierBodyWorkAnyRing` and `CarrierScalarKernelAnyRing`, and
`proof/corpus/WordRefinement.lean` refines the extracted `i64` point readout, scalar carrier step,
half-flux, exchange and checked acceptance to them. `proof/regen.sh --check` is fresh (1708 jobs)
and `cd proof/corpus && lake build` exits 0 (3276 jobs) with every `#audit` inside
`{propext, Classical.choice, Quot.sound}`. The lifecycle artifacts are
`docs/intent/carrier-refinement.md`, `docs/spec/carrier-refinement.md` and
`docs/plan/carrier-refinement.md`. **Stage 5 is current.** Its corpus conservation/energy results
(`CarrierEvolve.lean`, `CarrierDissipation.lean`, `CarrierVolume.lean`, `CarrierAuthority.lean`,
`CarrierCirculation.lean`, `CarrierBodyWork*.lean`) are Lean-proved. The fixed-line carrier
realization/refinement proof has landed: the closed three-cell `diffusive_line3_step` is extracted
from the real `src/carrier.rs` and refined to `scalarCarrierStep` and `exactHalfFlux`, with the
denoted total changing only by the denoted sources (zero-source conservation corollary); see
`proof/README.md`. Corpus `d9aefc93` adds the whole-line any-ring analogue
`diffusiveUpdate_half_three_anyRing` and the real exact quarter-energy identity
`dirichletEnergy_halfStep_three` with its strict form `dirichletEnergy_halfStep_three_lt`; the
factor `1/4` needs the order of `ℝ`, so the ring theorem and the word bridge claim no energy. The
surrounding Rust production integration, float mapping and live scheduling remain tested, not
refined. Evidence for both is in `docs/autonomous-run-log.md`.

### Correctness Work Before Physics Reuse

1. **Separate modular words from physical magnitudes.** Wrapped `i64` is not
   unbounded integer arithmetic. The real trichotomy's channel classification,
   invertibility and projections do not automatically survive base change.
   The corpus already says this in `Trichotomy/AnyRing.lean:13-46`; several Rust
   comments and the README still overstate what transfers.
2. **Make exact bounded arithmetic honestly partial.** A source-level probe
   reproduced `Dyadic::new(1, 127) + Dyadic::whole(2) == Dyadic::new(1, 127)`:
   alignment silently discards the high bits even in a debug build
   (`src/scalar.rs:145-152`). Checked shifts alone are insufficient: prove the
   shifted value fits. Check products, sums and exponent growth as well. An
   infallible finite `Ring` implementation cannot promise all dyadic operations.
3. **Remove false reordering permissions.** `Monoid` requires associativity, but
   implements floating addition (`src/merge.rs:158-217`). The probe produced
   `(1e20 + -1e20) + 1 = 1`, but `1e20 + (-1e20 + 1) = 0`. Keep an explicitly
   ordered approximate reduction or supply an actually exact accumulator; a fixed
   partition is not permission for arbitrary reassociation.
4. **Distinguish a scaled channel map from a projection.** `after_storing` uses
   `(1,1)`, not `(1/2,1/2)` (`src/keystore.rs:68-83`). It is not idempotent, and
   multiplying by two in the word ring can lose additional information.
5. **Scope container claims.** `map`/`table` use unsafe storage, SIMD and threads.
   They are not proved by extracting `Cell`. Establish a safe executable reference
   and finite-slot abstraction for required operations; independently relate any
   optimized implementation to it. Until then label that implementation as tested,
   not verified. Miri, fuzzing and Kani supplement but do not close that refinement.
6. **Specify replay independently of placement.** Stable world/object identity,
   ordered event inputs and canonical observation order cannot depend on table
   iteration, worker count, cache residency or pepper. Recorded random seeds can
   support replay too; a pepper is not a mathematical HashDoS guarantee. Threat
   models and bounded work must be explicit.

The review memo's generic-`&mut` exclusion is too broad: Aeneas's documented
restriction concerns instantiating a generic type with a mutable reference, not
every function taking `&mut V`. Test the actual reachable calls before redesigning
the merge API. Likewise, extraction returning zero is not sufficient: the broad
probe above produced axioms despite `-warnings-as-errors`.

Aeneas does not verify rustc/LLVM, the generated machine code, Vulkan drivers,
firmware or silicon. Proving an encoder emits valid SPIR-V is also weaker than
proving the shader implements the intended computation under its memory model.
Keep those boundaries in the trust inventory. The eventual bit-level goal needs
translation and execution arguments in addition to source-level refinement.

## Stage 4 Plan: Corpus-Connected Carrier

*(Historical plan and slice record. Stage 4 is complete; the current lifecycle artifacts are
`docs/intent/carrier-refinement.md`, `docs/spec/carrier-refinement.md` and
`docs/plan/carrier-refinement.md`, and the current status is in the Stage 4 complete entry above.)*

**First deliverable (slice 1) — link the extracted kernels to canonical definitions.**
Penglai's `density_exact`/`phase_flux_exact`/`reverse_phase_flux_exact` prove exact integer values
but import no corpus: `PhysicsProofs.lean` states `v.val = r.val * r.val`, not equality with a
corpus declaration. The cited corpus names are in the **old `sanguine` corpus**
(`Substrate.Algebra.ProductMadelung:107` `productDensity`, `ProductPhaseFlux:183`
`productPhaseFluxFromWave_first`), not in canonical `sanguine-apeiron` (`c7339cce`; `6213271f` has
different wave physics and neither has those filenames). Slice 1 is:

- re-derive the minimal density/phase-flux readout under canonical Apeiron
  (`productDensity R x y = R x y * R x y`; `productPhaseFluxFromWave_first R phase x y =
  productDensity R x y * (phase (x+1) y - phase x y)`);
- prove the Aeneas-extracted `physics_math.density`/`phase_flux` equal them at integer samples,
  widening the `u16` amplitude to `ℤ` and retaining the sign of the `i16` phase difference;
- give Penglai's `proof/` a corpus dependency using the bridge's `mathlib @ v4.31.0`-last pattern.

**Decision point.** The re-derivation is generic corpus math and therefore coordination work with
the corpus owner; the link itself is small integration. This is a scoped specialist task under the
handoff template, not another whole-corpus migration.

**Gate (handoff):** a small carrier lab executes the refined operations, unique-face reads and
complete boundary accounting; amplitude balance, density continuity and energy are distinguished;
a readout identity alone is not a water solver.

**Stage 4 slice 1 executed, corpus side (2026-09-11) — the carrier readout is rooted in the
trichotomy.** Added to canonical `sanguine-apeiron`:

- `Apeiron/Physics/Carrier.lean` defines `density σ R u x y := normForm σ (R x y • u)` — the
  trichotomy's own norm form of the phase cell, not a posited `R * R` — and proves
  `density_eq_amplitude_sq` from `Trichotomy.normForm_smul` plus the direction's unit-size
  hypothesis, with `fluxFirst` as the density times the phase's forward difference and the
  integer readouts `wordDensity`/`wordFluxFirst` for the machine-word kernels. Builds at the
  corpus's native Lean 4.30.0-rc2 (2395 jobs, no warnings) **and** at the Aeneas backend's Lean
  4.31.0 (`lake build Apeiron.Physics.Carrier` in `proof/corpus`, 2301 jobs).
- `Apeiron/Physics/CarrierFlow.lean` instantiates the direction at the algebra's flow,
  `density_flow`/`fluxFirst_flow` via `Trichotomy.normForm_flow`; it is on the analysis part of
  the corpus and is not imported by the 4.31.0 bridge.

**New compatibility finding (historical; migrated before the `16112030` pin).** The Aeneas
backend's 4.31.0 could not then build the whole corpus:
`Apeiron/Trichotomy/Flow.lean:310` failed to elaborate (`Prod.snd ∘ flow` vs
`fun u => (flow σ u).2` with a `NormedField.toNormedSpace.toModule` / `Semiring.toModule`
instance mismatch), so the corpus pin bump needed **source** fixes, not just a manifest update.
The trichotomy root used by `Carrier.lean` avoided this; the flow companion did not. The
migration landed in the corpus before `16112030`.

## Fold And Hashtrinity

Retire hashtrinity as a separately evolving trusted primitive. Keep it temporarily
as a compatibility/adaptation boundary for existing Fold consumers, then remove
duplicated implementations as their consumers migrate. Apeiron is not the proposed
home of every HTTP, TLS, codec, macro, reactor or async-runtime replacement.

This is not a package-name substitution. Hashtrinity's ordinary `Table::get`
trusts folded coordinates; its sentinel mapping aliases the top three words
(`hashtrinity/src/lib.rs`, `coord_of`, `get`, `get_exact`). Apeiron distinguishes
packed identities from lossy placement and carries/equality-checks opaque keys.
Insertion/merge APIs, panic behavior, iteration and persisted formats need explicit
adapters and regression fixtures. In-place mutation alone does not guarantee
rollback: Apeiron's `count_into` can leave an identity value after a panic.

The first Fold migration should exercise real stored-key and packed-key consumers
against collisions, sentinel values, removal, growth, ordered merges and exact
counts, followed by complete-operation benchmarks. Preserve measured optimized
paths such as batch access and dense arrays until replacements earn their keep.
GitLab branch `fix/hashtrinity-coord-fold` (`4e8534ca`) already applies an Apeiron
tuple-distribution finding; reuse its regression evidence instead of duplicating it.

Split issue #10 conceptually into substrate migration and platform compatibility.
Issues #11-13 remain optional Bedrock runtime/transport work, not prerequisites for
Apeiron or Penglai. Keep useful product maintenance, security fixes and measured
GPU work moving; do not freeze every Fold application. Do not replace vetted TLS,
QUIC or scheduling semantics merely to meet a dependency-count target. Bedrock is
excluded from the root workspace and currently has no `runtime-tokio` feature on
either fetched main: issue gates need a pinned branch and its actual manifest.

Record licensing before distribution: Apeiron currently carries GPL-3.0, while
Fold's workspace declares MIT/Apache. Common ownership does not itself document
an alternative grant for every contribution. Resolve the intended combined
distribution terms without making licensing an excuse to duplicate the kernel.

## Penglai First

Keep the current combat, creator-facing RON/Luau rules, controller support and sparse
ASCII presentation. Build the first valley around that fun game, not a replacement
engine showcase. Apeiron owns reusable kernels; Penglai owns fiction, materials'
physical interpretation, content, gameplay policies and host integration. Lyrium
extracts reusable engine machinery from demonstrated needs later.

Penglai currently only re-exports Apeiron (`src/substrate.rs:13`). The three
successful Aeneas lemmas prove exact integer formulas; `PhysicsProofs.lean` imports
only `PhysicsMath`, and its Lake package imports Aeneas, not either physics corpus.
The carrier kernels are not the existing cellular water solver. Therefore the
previous session's "milestone 1 done" does not satisfy its own scope's original
gate: the corpus connection, transfer proof and reproducible bootstrap/audit are
still incomplete. Preserve the genuine arithmetic progress without upgrading it.

### Recovered Toy Reference

[`ascii-venture` at `e933813`](https://github.com/paperboygold/ascii-venture/tree/e93381325336d878550aad1705be006ebc0a5846)
contains a Rust overworld with the grass/forest fire, steam, smoke, scorch, shallow
water, wakes, swimming and boats that make it a candidate for the previously
missing toy reference. Its contribution is concrete behavior and presentation to
reuse, not a new competing engine or an already verified material implementation.

- `src/main.rs:2076-2148`: fuel-time burning, downwind-biased hash-based spread,
  water quenching and drifting smoke. `2175-2194` selects flame glyphs/colors and
  scorched ground; `data/art.txt:23-27,46-50` holds their art vocabulary. This is
  the immediate reference for restoring the effects without an art reinvention.
- `src/main.rs:2196-2239`: separate water presentation snapshots, crest/trough
  bands, whitecap classification and depth-dependent underwater colors. Preserve
  the legibility concepts while connecting them to supported Penglai readouts.
- `src/main.rs:1592-1665`: swimmer/vessel wake interactions provide concrete game
  behaviors to test. Their forcing and direct height edits are prototype rules,
  not a proved two-way body/fluid coupling.
- `src/main.rs:532-645`: glyph damage repaint and Kitty shared-memory presentation.
  The dirty rasterization is useful, but the diff still scans the grids and the
  transport writes the whole framebuffer. This is not a measured O(changes)
  end-to-end renderer or a reason to replace Penglai's Bevy presentation.
- `src/bin/pot.rs`: a separate shaded 3-D ASCII rendering experiment with inverse
  depth, reflection and defocus, useful as a possible future examine-view reference.

Do not import its authority model. Water is stored in dense `Vec<f32>` height and
momentum arrays (`main.rs:809-822`), with shallow-water updates at `1943-2073`.
The ordinary HashMaps organize art/rendering rather than implement the fluid law.
Terrain and simulation dimensions follow the rendered grid; `render` calls
`ensure_terrain`, which resets water, fuel and fire on a size change
(`1811-1871,2376-2383`). Fire writes neighbors in-place while scanning, so update
order affects within-tick propagation. Those contracts conflict with persistent,
camera-independent, schedule-controlled Penglai physics.

Use the toy to recover observable effects and bounded reference scenarios. Do not
copy its hardcoded gameplay into Rust over Penglai's RON/Luau policy, claim its
fuel-time model is thermodynamic combustion, or infer floating-point refinement
from its comments citing Sanguine. `cargo check --locked --all-targets` passed;
`cargo test --locked --all-targets --quiet` ran successfully but found zero tests
in both binaries. No visual playtest or runtime performance measurement was done.

### Next Deliverables

1. **One pinned proof chain.** Reconcile corpus revisions with Song; establish the
   full-width word bridge in Apeiron. Re-derive the minimal density/phase/boundary
   closure in the canonical corpus, then connect Penglai's extracted functions to
   those definitions. Check source freshness, axiom cones and a failing mutation.
2. **One constructive carrier experiment.** Alongside the necessary arithmetic and
   region work, specify and implement a nontrivial temporal evolution with actual
   boundary/source inputs. Prove its advertised invariant from the construction,
   rather than assume the residual vanishes. Extend to three spatial factors and
   six-side flux accounting where volumetric gameplay requires them. Settle the
   material dictionary, vortical regimes, dissipation and body reaction/work.
3. **One interactive bank.** Under the supported material regime, connect the
   clearing to physical ground, flowing water and item/body interaction through
   one collision and material authority. Preserve combat and peaceful progression.
   Do not recreate the rejected independent MAC/global-pressure architecture or
   call a renamed mutable water array the seed-derived substrate.
4. **One valley loop.** Settlement, river/lake/outlet, woodland, a few named people;
   fight or negotiate, cut one tree, use its timber, ignite it with an existing
   technique, wet/extinguish it, and revisit the consequences. Fire/weather must
   affect materials and actions, not exist only as shaders. No LLM integration,
   generalized economy, replacement ECS or new art pipeline is prerequisite.
5. **Scale the same system.** Measure increasing active populations and disturbed
   regions, then expand geography. Thousands of entities need stable identities
   and bounded scheduling, not thousands of fully serialized Luau worlds per tick.
   Different activity rates and spatial resolutions require explicit contracts for
   causal effects and transitions, rather than silently freezing distant fires.

Do not finish every historical theorem or all stellar physics before attempting
the carrier experiment. Use the existing cosmos/sun/orbit ladder as bounded parent
inputs, with its actual approximations and calibration visible. Bring across the
dependency closures the next observable needs, progressively retiring old imports.

The key unresolved cost is **history evaluation**, not whether one hash lookup is
fast. Authoritative input is a versioned seed/content baseline plus accepted causal
events. Pure baseline regeneration cannot recover an unrecorded tree cut. Derive
an event residue/summary and prove it sufficient for future observations before
compacting history. Disposable caches or verified summaries are compatible with
this direction; secretly authoritative pressure/velocity caches are not. If bounded
evaluation is not established, report that limit and keep the experiment small.

The existing phase-gradient identity has zero plaquette circulation
(`ProductPhaseFlux.lean:161-175`), and joint stationarity consumes a stationary
history as a premise (`ProductNullWaveStationarity.lean:68-85,125-138`). Neither
result alone supplies arbitrary 3-D vortical, viscous, burning material dynamics.
These are concrete derivation targets, not evidence that the substrate is impossible.

Choose supported island extent from measured active work, memory and cold/revisit
cost, not uniform sub-metre simulation of New Zealand. Benchmark complete disturbed
scenes: kernel, query dependencies, residency, snapshots, scripts, collision and
rendering separately, including p95/p99 and failure budgets. Proofs establish
correctness/work bounds; measurements establish frame time on named hardware.

## Sovereign AI Path

Penglai exercises the same useful primitives as the later AI stack: identity,
regions, reductions, residency, bounded causal updates and backend correspondence.
It does not prove training quality or frontier intelligence. Song's independent
language/runtime path can consume these contracts and counterexamples while Rust
remains the practical application route.

The performance premise has real, narrower evidence. Fold's
`modelfold/src/simt/mod.rs:21-25` records one fp32 GEMV at 895 +/- 5 GB/s against
cuBLAS 902-911 on an RTX 5080. `docs/results/vs-llamacpp.md` records Qwen2.5-0.5B
decode at 920.2 tok/s against CUDA Q4_0 at 944.76, with different bits/weight and a
larger per-byte efficiency gap. These are not arbitrary-model or mobile guarantees,
and were not rebenchmarked here.

The credible proposition is that proven representations and transformations let
us remove work, materialization, transfers and avoidable dependency stages without
guessing about correctness. They do not automatically supply a remaining two
percent. Keep exact arbitrary-model execution and new model/training architectures
as separate acceptance tracks. Later Vulkan/SPIR-V implementations must refine the
same supported semantics, with target features, reduction order and numerical
tolerances specified, not inherit the CPU proof by name.

## Immediate Assignment

Pin the now-published canonical corpus revision and implement the targeted
`Cell<i64>` to `AnyRing` proof chain, correcting the numeric contracts that physics
would otherwise misuse. In the same focused workstream, define the smallest
nontrivial carrier/material experiment and its falsifying fixture. This puts the
hard mathematical and cost questions in front of the next playable interaction,
without blocking ordinary combat/content work on the entire sovereign stack.

## Issue Responses

Posted on 2026-09-11 at the owner's request. No issues were closed, reassigned or
retitled; these responses communicate direction and request scope separation.

- [#10: substrate migration and platform scope](https://github.com/Ptyktos/fold/issues/10#issuecomment-5627767455)
- [#11: runtime semantics and manifest-scoped gates](https://github.com/Ptyktos/fold/issues/11#issuecomment-5627772870)
- [#12: mesh/security acceptance independent of the substrate](https://github.com/Ptyktos/fold/issues/12#issuecomment-5627772891)
- [#13: evaluate tokio-xor as a separate, revision-pinned candidate](https://github.com/Ptyktos/fold/issues/13#issuecomment-5627772901)
