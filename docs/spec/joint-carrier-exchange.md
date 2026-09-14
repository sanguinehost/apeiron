# Joint Carrier and Exchange Profile

This file is the frozen specification of the joint carrier and exchange subprofile for
`APEIRON-PROGRAMME-W19-T2`. It names the word-ring operations, the two-slot cell algebra, the
carrier point readout, the exact scalar update, the exact half-flux, the three-cell diffusive line
and the equal-mass exchange, the eight obligations each row carries, the selected boundary, the
unmatched Song counterparts recorded as blockers, the hazards, the rejecting controls, the
integration order and what is explicitly not covered.

Authority: [`docs/direction.md`](../direction.md#primitive-correspondence) is the programme
authority; [`docs/kernel-boundary.md`](../kernel-boundary.md) owns the Rust representation and trust
boundary; [`proof/README.md`](../../proof/README.md) owns extraction and bridge scope. The canonical
corpus remains at `../sanguine-apeiron/proof/` at Git
`50c94192399add3b9f8e64bd14e36d8cba270705`, Mathlib `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f`.
The Apeiron sources were read at revision `2b5eda334e262f70f7c10ce567c8641c2396626d` and the Song
sources at revision `47869e76f982cfa3085a8618a82b80463e22ef46`; the per-file digests are recorded
under [Pins and source digests](#pins-and-source-digests).

This document specifies obligations. It does not claim the correspondence exists. No row is closed:
every row below is `selected-open` or `blocker`. The machine-readable form is
[`tools/joint/carrier_profile_rows.json`](../../tools/joint/carrier_profile_rows.json); the checker
[`tools/joint/carrier_profile_check.py`](../../tools/joint/carrier_profile_check.py) establishes
specification consistency only, never correspondence.

## Required fields

Every carrier and exchange profile row carries these eight fields. A row that drops one is
incomplete and the checker rejects it.

| Field | Obligation |
|---|---|
| `canonical` | The canonical object and observation relation, with exact theorem targets. |
| `entry_points` | The concrete Rust entry points and source identity, and the Song absence statement. |
| `matched_domain` | The matched admissible domain on which the Rust implementation and the canonical object are compared. |
| `representation_invariant` | The representation invariant established by the actual constructor, plus the state and effects. |
| `admission` | Separate Rust and Song admission predicates; joint success requires both, and an absent implementation is recorded. |
| `behavior` | Success, unsupported-input and resource-exhaustion behaviour, including the raw return encoding and the post-state. |
| `resources` | Separate per-implementation resource accounting; arithmetic ops, probes, nodes, stack entries and input consumption are not interchangeable fuel. |
| `refinement` | A source-connected refinement and a rejecting control. A copied model or a differential suite alone does not close a row. |

## Selected boundary

The selected boundary is the `i64` wrapping word ring, the two-slot cell algebra, the carrier point
readout, the exact scalar update, the exact half-flux, the three-cell line and the equal-mass
exchange. Every selected implementation has a canonical definition in the pinned corpus, an
executing Rust body on the crate path and a bridge refinement in `proof/corpus/WordRefinement.lean`.
Every selected implementation has **no** executing Song counterpart; the missing Song side is
recorded as an unmatched blocker and the joint domain is empty. No interval-propagation substitute
is accepted for a missing Song primitive.

| Row | Canonical object | Rust entry | Song entry | Status |
|---|---|---|---|---|
| K1 | Word wrapping add, sub and mul | `src/scalar.rs::<i64 as Ring>::{add, sub, mul}`, linked by `proof/Denotation.lean::{denote_wrapping_add, denote_wrapping_sub, denote_wrapping_mul}` | NONE | selected-open |
| K2 | Cell multiplication and norm form | `src/cell.rs::{Cell::mul, Cell::norm_form, Sigma::value, Cell::new}` | NONE | selected-open |
| K3 | Point density and flux | `src/carrier.rs::{point_density, point_flux}` | NONE | selected-open |
| K4 | Conservative scalar update | `src/carrier.rs::conservative_density_step` | NONE | selected-open |
| K5 | Exact half-flux | `src/carrier.rs::diffusive_flux_half` | NONE | selected-open |
| K6 | Three-cell diffusive line | `src/carrier.rs::{diffusive_line3_step, DensityLine3}` | NONE | selected-open |
| K7 | Exchange and claimed impulse | `src/exchange.rs::{exchange_impulse, elastic_exchange, accept_derived_impulse, Exchange}` | NONE | selected-open |

Canonical authority for K1 and K2 is `Apeiron/Trichotomy/AnyRing.lean` (`RingCell`, `ringMul`,
`ringNormForm`, `ringNormForm_mul`, `ringMul_comm`, `ringMul_assoc`). Canonical authority for K3 is
`Apeiron/Physics/CarrierAnyRing.lean` (`pointDensity`, `pointFlux`, `pointDensity_eq`,
`pointFlux_eq`). Canonical authority for K4, K5 and K6 is
`Apeiron/Physics/CarrierScalarKernelAnyRing.lean` (`scalarCarrierStep`, `scalarCarrierStep_eq`,
`scalarCarrierStep_continuity`, `DoublesTo`, `exactHalfFlux`, `exactHalfFlux_iff`,
`exactHalfFlux_neg`, `sum_scalarCarrierStep_eq`, `scalarCarrierStep_total_mass`). Canonical
authority for K7 is `Apeiron/Physics/CarrierBodyWorkAnyRing.lean` (`ringExchangeImpulse`,
`ringCarrierMomentumAfter_eq`, `ringBodyMomentumAfter_eq`, `ringTotalMomentum_conserved`,
`ringNormFormLedger_conserved`).

## Rows

### K1 · Word wrapping add, sub and mul

**1 · Canonical object and observation relation.** The `ZMod (2^64)` commutative ring under the
bit-pattern denotation `denote`: addition, subtraction and multiplication are total, pure and
wrapping on every 64-bit pattern. The two-slot algebra's operations are the canonical
`Apeiron.Trichotomy.AnyRing.ringMul` and `ringNormForm`; `ringNormForm_mul`, `ringMul_comm` and
`ringMul_assoc` are the ring laws the wrapping word operations instantiate. No order, no division
and no nonnegativity is used.

**2 · Entry points and source identity.** Rust: `<i64 as Ring>::add`, `<i64 as Ring>::sub` and
`<i64 as Ring>::mul` in `src/scalar.rs`, each a wrapping `i64` instance method. The extracted
`core.num.I64.wrapping_add`, `wrapping_sub` and `wrapping_mul` are linked by
`proof/Denotation.lean::{denote_wrapping_add, denote_wrapping_sub, denote_wrapping_mul}`. Song: NONE
(blocker `BC1`); `song/src/table/table.slang::mergeAcc` is a bounded addition on
`0 <= v <= COORD_CEILING`, not the full-width wrapping ring.

**3 · Matched domain.** Every pair of `i64` words for add, sub and mul: every 64-bit pattern; total
wrapping, no refusal.

**4 · Representation invariant.** The Rust scalar is the `i64` bit pattern; the `Ring for i64`
instance in `src/scalar.rs` is the constructor of the word ring. There is no state.
`denote : I64 -> ZMod (2^64)` reads the bit pattern (unsigned), so the extracted wrapping operations
denote `+`, `-` and `*` in `ZMod (2^64)`. `RingCell (ZMod (2^64))` is the two-slot carrier type.
State and effects: none; the three operations are pure and stateless.

**5 · Admission predicates.** Rust: total; every `(i64, i64)` pair is admitted for add, sub and mul,
with no refusal and no precondition. Song: NONE (blocker `BC1`); there is no executing Song
admission predicate for the full-width wrapping word ring. Joint success requires both admission
predicates; because the Song implementation is absent, the joint domain is empty and no
correspondence is claimed.

**6 · Behaviour.** Success: the extracted wrapping operations return a bare `i64` word for every
input, and the result is the ring operation under `denote`. Unsupported input: none; every 64-bit
pattern is supported. Resource exhaustion: not applicable; a fixed-width wrapping operation cannot
exhaust a resource. Raw return encoding: a bare `i64` word with no success tag, wrapping is the
defined result. Post-state: no state; the operations are pure.

**7 · Resource accounting.** Rust: one wrapping add, sub or mul per call; no probes, nodes or stack
entries; two scalar arguments. Song: NONE (blocker `BC1`); there is no Song implementation to
account. The accounts are separate and are never converted to a common fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/Denotation.lean::{denote_wrapping_add, denote_wrapping_sub, denote_wrapping_mul}` prove the
extracted wrapping operations are the `ZMod (2^64)` ring operations under `denote`. Song: no checked
Song-to-Lean refinement exists. The joint refinement is OPEN. Rejecting control: `C1`.

### K2 · Cell multiplication and norm form

**1 · Canonical object and observation relation.** `ringMul sigma z w =
(z.1 * w.1 + sigma * z.2 * w.2, z.1 * w.2 + z.2 * w.1)` and `ringNormForm sigma z =
z.1^2 - sigma * z.2^2` on `RingCell R = R x R`. `ringNormForm_mul` states the norm form is
multiplicative. The dial `sigma` is exactly `{-1, 0, 1}`: elliptic, parabolic and hyperbolic.

**2 · Entry points and source identity.** Rust: `Cell::mul`, `Cell::norm_form`, `Sigma::value` and
`Cell::new` in `src/cell.rs`. The extracted `word_mul` and `word_norm` call the real
`src/cell.rs::Cell::mul` and `Cell::norm_form` bodies at the `i64` `Ring`, linked by
`proof/corpus/WordRefinement.lean::{word_mul_denote, word_norm_denote, word_norm_mul}`. Song: NONE
(blocker `BC2`); `song/src/prover/ring/prover0.slang` operates on SKI/iota terms.

**3 · Matched domain.** Every pair of `Cell<i64>` (each slot any 64-bit pattern) and every dial in
`{Elliptic, Parabolic, Hyperbolic}`; componentwise, total wrapping.

**4 · Representation invariant.** `Cell::new(a, b)` constructs the ordered pair and `Sigma::value`
maps the dial to `-1`, `0` or `1`. There is no state. `denoteCell` maps a `Cell<i64>` to a
`RingCell (ZMod (2^64))` by denoting each slot, and `denoteSigma` maps the dial by `sigmaI64`. State
and effects: none; both operations are pure and stateless.

**5 · Admission predicates.** Rust: total for a lawful `Ring`; every pair of cells and every dial is
admitted for `mul` and `norm_form`, with no refusal and no precondition. Song: NONE (blocker `BC2`);
there is no executing Song admission predicate for a two-slot algebra with a dial. Joint success
requires both admission predicates; because the Song implementation is absent, the joint domain is
empty and no correspondence is claimed.

**6 · Behaviour.** Success: `mul` returns the product cell and `norm_form` returns the scalar for
every input and every dial. Unsupported input: none on the `i64` wrapping instance; the generic
`Ring` contract is behavioral, so a partial scalar such as `Dyadic` can panic. Resource exhaustion:
not applicable; fixed-width wrapping operations cannot exhaust a resource. Raw return encoding:
`mul` returns a `Cell<i64>` and `norm_form` returns an `i64`, with no success tag. Post-state: no
state; both operations are pure.

**7 · Resource accounting.** Rust: one dial multiply and componentwise adds per `mul`, two squares
and a subtract per `norm_form`; no probes, nodes or stack entries; two cells and one dial for `mul`,
one cell and one dial for `norm_form`. Song: NONE (blocker `BC2`); there is no Song implementation
to account. The accounts are separate and are never converted to a common fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/WordRefinement.lean::{word_mul_denote, word_norm_denote}` prove the extracted cell
product and norm form equal `Apeiron.Trichotomy.AnyRing.ringMul` and `ringNormForm` at
`denoteSigma s`, and `word_norm_mul` transfers `ringNormForm_mul`. Song: no checked Song-to-Lean
refinement exists. The joint refinement is OPEN. Rejecting control: `C2`.

### K3 · Point density and flux

**1 · Canonical object and observation relation.** `pointDensity sigma a u =
ringNormForm sigma (ringMul sigma (a, 0) u)` is the size of the phase cell the amplitude `a`
stretches the direction `u` into; `pointFlux sigma a u p n = pointDensity sigma a u * (n - p)`.
`pointDensity_eq` factors the density through `ringNormForm_mul` and `pointFlux_eq` reads a unit
direction off that factor.

**2 · Entry points and source identity.** Rust: `point_density` and `point_flux` in `src/carrier.rs`,
generic over the `Ring` contract. The extracted `word_point_density` and `word_point_flux` call the
real bodies at `R = i64`, linked by `proof/corpus/WordRefinement.lean::{word_point_density_denote,
word_point_flux_denote}`. Song: NONE (blocker `BC3`); no executing Song primitive denotes the
carrier readout.

**3 · Matched domain.** Every dial, every amplitude `i64`, every direction `Cell<i64>` and every
pair of phase samples `(previous, next)`; total wrapping.

**4 · Representation invariant.** `Cell::new(amplitude, ZERO)` builds the stretched unit;
`point_density` and `point_flux` are the generic `src/carrier.rs` bodies instantiated at `R = i64`.
There is no state. The extracted density denotes `Apeiron.Physics.pointDensity` at `denoteSigma s`,
and the extracted flux denotes `Apeiron.Physics.pointFlux` at `denoteSigma s`, composing the density
with the wrapping sub and mul denotations. State and effects: none; both readouts are pure.

**5 · Admission predicates.** Rust: total for a lawful `Ring`; every amplitude, direction and sample
pair is admitted, with no refusal. Song: NONE (blocker `BC3`); there is no executing Song admission
predicate for the carrier readout. Joint success requires both admission predicates; because the
Song implementation is absent, the joint domain is empty and no correspondence is claimed.

**6 · Behaviour.** Success: `point_density` returns the density and `point_flux` returns the density
times the forward phase difference `next - previous`. Unsupported input: none on the `i64` wrapping
instance. Resource exhaustion: not applicable; fixed-width wrapping operations cannot exhaust a
resource. Raw return encoding: a bare `i64` word with no success tag. Post-state: no state; both
readouts are pure.

**7 · Resource accounting.** Rust: one cell product, one norm form and one multiply per flux; no
probes, nodes or stack entries; one dial, one amplitude and one direction for density, plus two
phase samples for flux. Song: NONE (blocker `BC3`); there is no Song implementation to account. The
accounts are separate and are never converted to a common fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/WordRefinement.lean::{word_point_density_denote, word_point_flux_denote}` prove the
extracted readouts equal `Apeiron.Physics.pointDensity` and `pointFlux` at `denoteSigma s`. Song: no
checked Song-to-Lean refinement exists. The joint refinement is OPEN. Rejecting control: `C3`.

### K4 · Conservative scalar update

**1 · Canonical object and observation relation.** `scalarCarrierStep d outgoing incoming source =
d - (outgoing - incoming) + source` is one exact scalar carrier update; `scalarCarrierStep_continuity`
is the local balance `scalarCarrierStep d outgoing incoming source - d + (outgoing - incoming) =
source`, and `scalarCarrierStep_eq` reads the definition back.

**2 · Entry points and source identity.** Rust: `conservative_density_step` in `src/carrier.rs`,
generic over the `Ring` contract. The extracted `word_conservative_density_step` calls the real
body at `R = i64`, linked by `proof/corpus/WordRefinement.lean::{word_conservative_density_step_denote,
word_conservative_density_step_eq_some_iff}`. Song: NONE (blocker `BC3`); no executing Song
primitive denotes the exact integer carrier update.

**3 · Matched domain.** Signed `i64` density, incoming, outgoing and source; the step admits exactly
the quadruples whose three checked intermediates `outgoing - incoming`, `density - net` and
`after + source` all fit `i64`.

**4 · Representation invariant.** `conservative_density_step` is the generic `src/carrier.rs` body
instantiated at `R = i64`; it has no state and uses explicit `match` rather than the question-mark
operator. On success the extracted step denotes `Apeiron.Physics.scalarCarrierStep` at the
denotation of its four arguments, and `word_conservative_density_step_eq_some_iff` states the exact
generated no-overflow precondition over the three checked intermediates. State and effects: none;
the step is pure and returns `Option<i64>`.

**5 · Admission predicates.** Rust: the step admits exactly when all three checked intermediates are
representable; otherwise it returns `None` and refuses the whole update. Song: NONE (blocker `BC3`);
there is no executing Song admission predicate for the exact update. Joint success requires both
admission predicates; because the Song implementation is absent, the joint domain is empty and no
correspondence is claimed.

**6 · Behaviour.** Success: when all three intermediates fit, the extracted step returns `Some(r)`
and `denote r = scalarCarrierStep (denote density) (denote outgoing) (denote incoming) (denote
source)`. Unsupported input: none; the inputs are signed words and an out-of-range intermediate is a
refusal, not an unsupported input class. Resource exhaustion: an unrepresentable intermediate
returns `None`; this is exact arithmetic refusal, not a resource budget. Raw return encoding:
`Option<i64>`, `Some(word)` on success and `None` on refusal, with no wrapping. Post-state: no
state; the step is pure.

**7 · Resource accounting.** Rust: two checked subtractions and one checked addition; no probes,
nodes or stack entries; four scalar arguments. Song: NONE (blocker `BC3`); there is no Song
implementation to account. The accounts are separate and are never converted to a common fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/WordRefinement.lean::{word_conservative_density_step_denote,
word_conservative_density_step_eq_some_iff}` prove the successful step denotes `scalarCarrierStep`
and state the exact success precondition. Song: no checked Song-to-Lean refinement exists. The joint
refinement is OPEN. Rejecting control: `C4`.

### K5 · Exact half-flux

**1 · Canonical object and observation relation.** `DoublesTo q x := 2 * q = x` and
`exactHalfFlux f left right := DoublesTo f (left - right)`: the exact half `-(right - left) / 2`
written with no division. `exactHalfFlux_iff` reads the definition back and `exactHalfFlux_neg`
reads it as `2 * f = -(right - left)`.

**2 · Entry points and source identity.** Rust: `diffusive_flux_half` in `src/carrier.rs`, generic
over the `Ring` contract. The extracted `word_diffusive_flux_half` calls the real body at `R = i64`,
linked by `proof/corpus/WordRefinement.lean::{word_diffusive_flux_half_denote,
word_diffusive_flux_half_success}`. Song: NONE (blocker `BC3`); no executing Song primitive denotes
the exact half-flux.

**3 · Matched domain.** Signed `i64` left and right; the half-flux exists exactly when `right - left`
fits `i64` and is even; an odd or unrepresentable difference is refused.

**4 · Representation invariant.** `diffusive_flux_half` is the generic `src/carrier.rs` body
instantiated at `R = i64`; it selects the signed quotient `0 - ((right - left) / 2)`. On success the
extracted result satisfies the canonical `Apeiron.Physics.exactHalfFlux` witness `2 * f =
left - right`; the selected output is the signed quotient, which is the extra selection a bare
witness does not determine in `ZMod (2^64)`. State and effects: none; the half-flux is pure.

**5 · Admission predicates.** Rust: the half-flux admits exactly when `right - left` is
representable and even; otherwise it returns `None`. Song: NONE (blocker `BC3`); there is no
executing Song admission predicate for the exact half-flux. Joint success requires both admission
predicates; because the Song implementation is absent, the joint domain is empty and no
correspondence is claimed.

**6 · Behaviour.** Success: `Some(f)` with `exactHalfFlux (denote f) (denote left) (denote right)`,
that is `2 * denote f = denote left - denote right`. Unsupported input: none; left and right are
signed words. Resource exhaustion: an odd or unrepresentable difference returns `None`; this is
exact arithmetic refusal, not a resource budget. Raw return encoding: `Option<i64>`, with no
rounding and no wrapping. Post-state: no state; the half-flux is pure.

**7 · Resource accounting.** Rust: one checked subtraction, one remainder test, one division and one
negation; no probes, nodes or stack entries; two scalar arguments. Song: NONE (blocker `BC3`); there
is no Song implementation to account. The accounts are separate and are never converted to a common
fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/WordRefinement.lean::{word_diffusive_flux_half_denote, word_diffusive_flux_half_success}`
prove a successful half-flux satisfies `exactHalfFlux` and give the representability and evenness
preconditions. Song: no checked Song-to-Lean refinement exists. The joint refinement is OPEN.
Rejecting control: `C5`.

### K6 · Three-cell diffusive line

**1 · Canonical object and observation relation.** The three-cell line is `scalarCarrierStep` on
each of the three cells with the two interior faces the `exactHalfFlux` values `-(center - left) / 2`
and `-(right - center) / 2`, read once each and shared, and the two endpoint faces closed at zero.
The corpus any-ring three-cell theorem `Apeiron.Physics.diffusiveUpdate_half_three_anyRing` is not
used: its hypothesis `DoublesTo nu 1` is unsatisfiable in `ZMod (2^64)` because `2` is not
invertible, so the extracted line carries the face value directly.

**2 · Entry points and source identity.** Rust: `diffusive_line3_step` and `DensityLine3` in
`src/carrier.rs`, generic over the `Ring` contract. The extracted `word_diffusive_line3_step` calls
the real body at `R = i64`, linked by
`proof/corpus/WordRefinement.lean::{word_diffusive_line3_step_eq_some_iff,
word_diffusive_line3_step_denote, word_diffusive_line3_step_total_mass_zero_source}`. Song: NONE
(blocker `BC3`); no executing Song primitive denotes the three-cell line.

**3 · Matched domain.** Every pair of `DensityLine3` triples of signed words and every pair of
sources; the step admits exactly when both interior differences are representable and even and all
three checked cell updates succeed.

**4 · Representation invariant.** `DensityLine3 { left, center, right }` is the state triple;
`diffusive_line3_step` is the generic `src/carrier.rs` body instantiated at `R = i64`, and every
local is bound before the result is constructed. On success the two interior faces are
`exactHalfFlux` witnesses and each component is the canonical `scalarCarrierStep` at the shared face
and closed end; a refusal returns `None` with no partially updated line. State and effects: none;
the step is pure and returns `Option<DensityLine3>`.

**5 · Admission predicates.** Rust: the line admits exactly when both interior faces exist and all
three cell updates succeed; any refusal returns `None` for the whole line. Song: NONE (blocker
`BC3`); there is no executing Song admission predicate for the line. Joint success requires both
admission predicates; because the Song implementation is absent, the joint domain is empty and no
correspondence is claimed.

**6 · Behaviour.** Success: `Some(next)` where each component denotes `scalarCarrierStep` at the
shared face and closed end, and the denoted total changes only by the denoted sources. Unsupported
input: none; the inputs are signed word triples. Resource exhaustion: an odd interior difference or
an unrepresentable intermediate returns `None` for the whole line, with no partial result. Raw
return encoding: `Option<DensityLine3>`, with no rounding and no wrapping. Post-state: no state; the
step is pure.

**7 · Resource accounting.** Rust: two half-flux computations and three cell updates; no probes,
nodes or stack entries; two triples, the old line and the source line. Song: NONE (blocker `BC3`);
there is no Song implementation to account. The accounts are separate and are never converted to a
common fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/WordRefinement.lean::{word_diffusive_line3_step_eq_some_iff,
word_diffusive_line3_step_denote, word_diffusive_line3_step_total_mass_zero_source}` characterise
success, refine each component and conserve the denoted total at zero source. The corpus
`diffusiveUpdate_half_three_anyRing` is not a target because its `DoublesTo nu 1` is uninhabited at
`ZMod (2^64)`. Song: no checked Song-to-Lean refinement exists. The joint refinement is OPEN.
Rejecting control: `C6`.

### K7 · Exchange and claimed impulse

**1 · Canonical object and observation relation.** `ringExchangeImpulse p q = p - q` is the derived
impulse; `ringCarrierMomentumAfter p q = p - ringExchangeImpulse p q` and `ringBodyMomentumAfter p q
= q + ringExchangeImpulse p q`. `ringCarrierMomentumAfter_eq` and `ringBodyMomentumAfter_eq` state
the swap, `ringTotalMomentum_conserved` states the total is unchanged, and
`ringNormFormLedger_conserved` states the `ringNormFormLedger` is unchanged at every dial.

**2 · Entry points and source identity.** Rust: `exchange_impulse`, `elastic_exchange`,
`accept_derived_impulse` and `Exchange` in `src/exchange.rs`, generic over the `Ring` contract. The
extracted `word_exchange_impulse`, `word_elastic_exchange` and `word_accept_derived_impulse` call
the real bodies at `R = i64`, linked by
`proof/corpus/WordRefinement.lean::{word_exchange_impulse_denote, word_elastic_exchange_denote,
word_accept_derived_impulse_spec}`. Song: NONE (blocker `BC4`); no executing Song primitive performs
the equal-mass exchange or the checked-claim acceptance.

**3 · Matched domain.** Every pair of `Cell<i64>` momenta and every claimed `Cell<i64>`; the derived
impulse is total and the claimed-impulse kernel is partial on source-word equality of both slots.

**4 · Representation invariant.** `Exchange { impulse, carrier_after, body_after }` is the result
record; the three functions are the generic `src/exchange.rs` bodies instantiated at `R = i64`. The
extracted impulse denotes `Apeiron.Physics.ringExchangeImpulse` and the extracted exchange denotes
the canonical impulse, carrier and body fields;
`word_accept_derived_impulse_spec` states the checked-claim kernel returns `some` exactly when the
claim equals the derived word impulse, selected by source-word equality. State and effects: none;
the functions are pure and the claim kernel returns `Option<Exchange<i64>>`.

**5 · Admission predicates.** Rust: the derived impulse is total; `accept_derived_impulse` admits
exactly a claim equal to the derived impulse by source-word equality of both slots, otherwise it
returns `None`. Song: NONE (blocker `BC4`); there is no executing Song admission predicate for the
exchange or the claim. Joint success requires both admission predicates; because the Song
implementation is absent, the joint domain is empty and no correspondence is claimed.

**6 · Behaviour.** Success: `exchange_impulse` returns `p - q`, `elastic_exchange` returns the
impulse and the swapped momenta, and `accept_derived_impulse` returns `Some` of the elastic exchange
exactly when the claim equals the derived word impulse. Unsupported input: none at the exchange; a
claim unequal to the derived word impulse is rejected, not unsupported. Resource exhaustion: not
applicable; wrapping arithmetic cannot exhaust a resource. Raw return encoding: `exchange_impulse`
returns a `Cell<i64>`, `elastic_exchange` returns an `Exchange<i64>`, and `accept_derived_impulse`
returns `Option<Exchange<i64>>`. Post-state: no state; the functions are pure.

**7 · Resource accounting.** Rust: one componentwise subtraction for the impulse, plus two
subtractions and two additions for the exchange, and the claim kernel recomputes the exchange; no
probes, nodes or stack entries; two momenta for the exchange, and two momenta and one claim for the
claim kernel. Song: NONE (blocker `BC4`); there is no Song implementation to account. The accounts
are separate and are never converted to a common fuel.

**8 · Source-connected refinement and rejecting control.** Rust:
`proof/corpus/WordRefinement.lean::{word_exchange_impulse_denote, word_elastic_exchange_denote,
word_accept_derived_impulse_spec}` prove the derived impulse, the exchange and the checked-claim
kernel against `Apeiron.Physics.ringExchangeImpulse` and the canonical momentum updates; the claim
branch is selected by source-word equality. Song: no checked Song-to-Lean refinement exists. The
joint refinement is OPEN. Rejecting control: `C7`.

## Unmatched counterparts

These counterparts have no matched primitive on one side. They are recorded as blockers, not
assumed, and the checker requires every one of them to be present. Every blocker is on the Song
side: no executing Song primitive denotes the word ring, the two-slot algebra, the carrier readout
or the exchange, so the joint domain is empty and no row closes. No interval-propagation substitute
is accepted.

| ID | Counterpart | Reason | Status |
|---|---|---|---|
| BC1 | Rust `src/scalar.rs::<i64 as Ring>::{add, sub, mul}` word arithmetic, linked by `proof/Denotation.lean::{denote_wrapping_add, denote_wrapping_sub, denote_wrapping_mul}`; Song side NONE. | `song/src/table/table.slang::mergeAcc` is a bounded machine-integer addition on `0 <= v <= COORD_CEILING`, not the full-width wrapping word ring, and Song has no wrapping subtraction or multiplication. | blocker |
| BC2 | Rust `src/cell.rs::{mul, norm_form}` with `Cell::add`, `Cell::upper`, `Cell::lower` and `Cell::channel` over `Ring`; Song side NONE. | Song has no executing two-slot algebra with a `Sigma` dial; `song/src/prover/ring/prover0.slang` operates on SKI/iota terms, not on `R[x]/(x^2 - sigma)` cells. | blocker |
| BC3 | Rust `src/carrier.rs::{point_density, point_flux, conservative_density_step, diffusive_flux_half, diffusive_line3_step}`; Song side NONE. | No Song executing primitive denotes the carrier readout or the exact integer carrier update; no interval-propagation substitute is accepted. | blocker |
| BC4 | Rust `src/exchange.rs::{exchange_impulse, elastic_exchange, accept_derived_impulse}`; Song side NONE. | No Song executing primitive performs the equal-mass exchange or the checked-claim acceptance. | blocker |

## Hazards

| ID | Hazard | Consequence |
|---|---|---|
| H1 | Two witnesses in `ZMod (2^64)` need not be equal: the doubling witness `2 * f = left - right` does not determine a unique `f` because `2` is a zero divisor in `ZMod (2^64)`. | The canonical `exactHalfFlux` witness is not enough for joint output equality; the Rust kernel selects the signed quotient `0 - ((right - left) / 2)`, and that selection is an additional obligation the refinement records. |
| H2 | `DoublesTo nu 1` is unsatisfiable in `ZMod (2^64)`: reducing `2 * nu = 1` modulo `2` gives `0 = 1`, so the corpus any-ring three-cell theorem `diffusiveUpdate_half_three_anyRing` is vacuous at the bridge's ring. | The three-cell line's canonical targets are `scalarCarrierStep` and `exactHalfFlux`, not `diffusiveUpdate_half_three_anyRing`; the extracted line carries the face value directly. |
| H3 | The conservative step and the exact half-flux are partial: an unrepresentable intermediate or an odd face difference returns `None`. | No theorem claims totality; the refinement is conditional on the generated no-overflow preconditions and the representability and evenness hypotheses, and the refusal witnesses show the partiality is real. |
| H4 | The Rust kernels are generic over the crate's `Ring` contract, which is a behavioral contract and not a type-level proof; a partial scalar such as `Dyadic` can panic. | Only the `i64` wrapping execution is extracted; the generic identities are conditioned on lawful and successful operations, and no claim transfers to `Dyadic`. |
| H5 | Every carrier and exchange row has an unmatched Song counterpart (`BC1` to `BC4`): Song has no executing primitive for the word ring, the two-slot algebra, the carrier readout or the exchange. | There is no matched Song domain and no joint refinement for any row; no interval-propagation substitute is accepted. Every row stays `selected-open`. |

## Rejecting controls

These controls are specification-level obligations. They are not executed by
`carrier_profile_check.py`, which checks consistency only; each names a source-derived mutation
that the executing refinement must reject. The checker's `--controls` and `--self-test` modes
separately mutate a copy of the profile and require each mutation to be rejected.

| ID | Target | Control |
|---|---|---|
| C1 | K1 | Replace the wrapping addition with a subtraction or a checked addition in `proof/Denotation.lean`; `denote_wrapping_add` must fail. |
| C2 | K2 | Change the hyperbolic dial value from `1` to `2` or cross the two output slots in `src/cell.rs`; `word_mul_denote` and `word_norm_mul` must fail. |
| C3 | K3 | Drop the `point_density` factor from `point_flux` or reverse the forward difference `next - previous` in `src/carrier.rs`; `word_point_flux_denote` must fail. |
| C4 | K4 | Swap incoming and outgoing or remove one checked intermediate in `conservative_density_step`; `word_conservative_density_step_denote` and `word_conservative_density_step_eq_some_iff` must fail. |
| C5 | K5 | Round an odd difference instead of refusing it, or reverse the negated half in `diffusive_flux_half`; `word_diffusive_flux_half_success` and `word_diffusive_flux_half_denote` must fail. |
| C6 | K6 | Give the left cell the right shared face or make `diffusive_line3_step` always return `None`; `word_diffusive_line3_step_eq_some_iff`, `word_diffusive_line3_step_denote` and `word_diffusive_line3_step_zero_success` must fail. |
| C7 | K7 | Replace the derived impulse subtraction with addition, compare one cell component, or invert the acceptance branch in `src/exchange.rs`; `word_exchange_impulse_denote`, `word_elastic_exchange_denote` and `word_accept_derived_impulse_spec` must fail. |

## Integration order

1. K1 word wrapping add, sub and mul: the base ring the cell algebra is built over; the denotation
   identities are the first source-connected link.
2. K2 cell multiplication and norm form: the dial-parameterised product and the multiplicative norm
   form; the blocker `BC2` records the missing Song two-slot algebra.
3. K3 point density and flux: the carrier readout over the cell algebra.
4. K4 conservative scalar update and K5 exact half-flux: the partial exact kernels; `H1` records the
   witness non-uniqueness and the signed quotient selection.
5. K6 three-cell diffusive line: the composition of K4 and K5; `H2` records why the corpus any-ring
   three-cell theorem is not the target.
6. K7 exchange and claimed impulse: the arithmetic exchange and the checked-claim kernel.
7. A vertical lift or a Song carrier/exchange implementation only after a matched Song primitive
   exists; no interval-propagation substitute discharges a blocker.

## Not covered

- rustc/LLVM code generation and the generated machine code.
- The kernel ABI, stack and OS.
- GPU / SPIR-V.
- Runtime linkage between Apeiron and Song; the required result is a checked correspondence, not
  linkage.
- Song local registers, ontology and commands.
- Semantic correspondence itself. `tools/joint/carrier_profile_check.py` establishes specification
  consistency only; it never reports correspondence.
- Any interval-propagation substitute for the missing Song carrier or exchange primitive.
- Totality of the conservative step, the exact half-flux or the three-cell line; their partiality is
  recorded.
- Any energy, nonnegativity or ordering claim over the word ring.
- A physical mass or momentum law; the conserved totals are denoted `ZMod (2^64)` sums.

## Pins and source digests

The checker resolves and verifies these pins; a drift in any revision or digest fails `--check`.
Every recorded digest is bound to the blob at its recorded revision.

| Item | Revision |
|---|---|
| Canonical corpus (`../sanguine-apeiron/proof`) | `50c94192399add3b9f8e64bd14e36d8cba270705` |
| Mathlib | `fabf563a7c95a166b8d7b6efca11c8b4dc9d911f` |
| Apeiron sources | `2b5eda334e262f70f7c10ce567c8641c2396626d` |
| Song sources | `47869e76f982cfa3085a8618a82b80463e22ef46` |

| Source | sha256 |
|---|---|
| `src/scalar.rs` | `9199e955304cb3d54fe973ac5881f219d4c6f71b4be001a0d32a6fb16ffde92e` |
| `src/cell.rs` | `8cfc787aa0c6fcec6b039398f54fd7388f7dec8f047b98f8379e7165c0069404` |
| `src/carrier.rs` | `4fcdbeba5c2d25d10e0a22f2f328e214d0e9884bd9f886a25abd4b6741a8a8e8` |
| `src/exchange.rs` | `f488773c798c649ff59d39f7b24c4c17b6ff350fa9f7a3b4ccc81cfc479b7621` |
| `proof/Denotation.lean` | `d55971bea1f696e6d3ac8d7ed56e3be59cb98df68308ee2a159867cd5fbfa999` |
| `proof/corpus/WordRefinement.lean` | `9ceaed87c9a0ea92eb1a0e05e634b9ff18486684418ce4a8d419372e653d9c7a` |
| `../sanguine-apeiron/proof/Apeiron/Trichotomy/AnyRing.lean` | `51cf1ceecd283771a4a5ae52be39150cc3d3e2ff3419906822a10a121eafffca` |
| `../sanguine-apeiron/proof/Apeiron/Physics/CarrierAnyRing.lean` | `bee50d1df59be11d7a41536e4c56ff284235cb32c16265e55ba0cfd4e7df2b79` |
| `../sanguine-apeiron/proof/Apeiron/Physics/CarrierScalarKernelAnyRing.lean` | `4545466793c6fcc367e5d8367784c9cd9f0f0cbb8bdbc952e6aaac00d2853b29` |
| `../sanguine-apeiron/proof/Apeiron/Physics/CarrierBodyWorkAnyRing.lean` | `ea3155611c61b7e4edcc26461ccbb9d238b0e8c24c74d921d39d3e1e8c3b34ac` |

## Checker

`python3 tools/joint/carrier_profile_check.py --check` validates
`tools/joint/carrier_profile_rows.json` against this document and the pinned sources: every row
carries all eight fields and stays `selected-open`; every named canonical target resolves as the
exact fully-qualified declaration in the pinned corpus under its real namespace, so a merely opened
prefix, a reversed namespace or a missing namespace is rejected; every named bridge refinement
resolves in `proof/`; every named Rust entry exists as a concrete executing declaration bound to the
named implementation, so a same-named method on another type is rejected; the corpus Git revision,
the Mathlib revision resolved from a manifest, the Apeiron and Song revisions and per-source sha256
digests match the resolved checkout, and each recorded digest is bound to the blob at the recorded
revision; every implementation present in a row carries its required resource kinds separately and
no generic `fuel` is admitted; every named reference in the structured prose fields and in this
document, including blocker `counterpart`/`reason` entries and control targets, is checked; each
row's Markdown fields are present with nonempty bodies and its Markdown status agrees with the JSON
status; every required unmatched blocker, hazard and rejecting control is present; and this document
carries the actual records and fields, not incidental mentions. It exits nonzero on a missing field,
row, blocker, hazard, control, theorem or source entry, on a declaration-name or namespace mismatch,
on a corpus or digest drift, on an unresolved named reference, on a Markdown field or status that
disagrees with the JSON, or on a row that is not `selected-open`.

`--controls` and `--self-test` mutate a deep copy of the loaded profile and the spec text in memory
and require each mutation to be rejected; `--self-test` additionally requires one rejecting control
per required row obligation. The working tree is never touched. The checker establishes
specification consistency only, never correspondence.
