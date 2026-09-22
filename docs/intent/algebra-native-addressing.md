# Intent — Algebra-native addressing and channel decomposition

- **Owner:** lead (direction); Astra (architecture founding); Sol (ratification)
- **Source revision:** `c450c222905c050f793a23332708c230c658f82c`; working tree dirty
- **Status:** approved for founding — per-face address rings
- **Spec:** `docs/spec/algebra-native-addressing.md` (Astra founding; prime-power split and key-retention corrections applied)
- **Plan:** `docs/plan/algebra-native-addressing.md` (Astra founding; implementation rows are reservations)

## Problem

The growable table places a key by free geometry: the shard is the low byte of the coordinate
(`src/table/state.rs`, `shard`); a region is rounded to a power-of-two group count from an observed
load (`slots_for`, `src/table/state.rs:328`); and absence is filtered by a Bloom summary over
`place >> 8` with a golden-ratio multiplier and a runtime word width (`summary_index`,
`src/table/state.rs:127`). The canonical object is a total function whose semantics leave all of this
unobserved: "an implementation may sort by home address, shard across cores, or take arrival order,
and get the same function" (`proof/corpus` canonical `Apeiron/Table/Basic.lean:11,25-26,71`). Every
geometry value that reaches a shift or an index therefore becomes a partial operation in the Aeneas
rendering with no algebraic statement to discharge it at the use site. The class has recurred: the
unmodeled `Vec::is_empty`/`usize::trailing_zeros` intrinsics (avoided in `src/table/state.rs:13`), the
`summary_index` shift overflow (totalized, which moved the obligation rather than removing it), the
nested-loop return in `spill_read` (single-exit), the borrowed-carrier context-match failure, and the
`summary_index` dynamic-shift bound now blocking `Growable/Search` and `Growable/Relocation`. The
placement is also an adversary-influenceable hash surface, which is exactly the property the
programme claims the primitive removes.

## Why the alternatives were rejected

| Alternative | Why not |
|---|---|
| Patch each Aeneas obstruction in the source | Relocates a free-geometry safety obligation; the next geometry feature regenerates it. The recurrence is the evidence. |
| Prove the dynamic-shift bound with a one-off spec lemma | Same bucket-2 obligation, and leaves the golden-ratio hash of `place` on the proved read path. |
| Type the geometry (const-generic region/summary width) | Removes the shift but keeps arbitrary logical partitioning; the partition and its security properties stay un-derivable from the primitive. |
| Scope the geometry executor as tested/unverified | Renegotiates "every executing face refines the canonical object" and keeps adversary-influenceable placement. |
| Compact hot lookup directory (Astra T1) | Adds locator/coverage/collision contracts on top of the same arbitrary partition; a performance patch over free geometry, not a derivation. |

## Completion condition

Every executing face places a key by the primitive's canonical reduction, with no hash and no free
power-of-two geometry, and the read/write faces refine `Apeiron.Table` under the closed axiom cone.

```sh
cargo test --workspace --all-features --locked \
  && cargo clippy --workspace --all-features --all-targets --locked -- -D warnings \
  && proof/regen.sh --check \
  && (cd proof/corpus && lake build) \
  && cargo run --release --locked --example reg1 -- --check performance --profile full --accept --controls
```

## Invariants

1. The primitive is the Clifford algebra of the rank-one form `q = ⟨σ⟩`, `Cl(σ) ≅ R[x]/(x²−σ)`, with
   norm form `N(a+bx) = a²−σb²`, multiplicative: `N(uv) = N(u)·N(v)`.
2. The address structure is per face, as `src/coord.rs:36-47` already states ("a single fold cannot be
   right for all three"):
   - **elliptic** (σ<0, regenerated): the address determines the key, so the address ring stays
     `Z/2⁶⁴` and the fold is the bijection `pack`. A modulus-based reduction cannot be injective on the
     full key space, so it is excluded here.
   - **parabolic** (σ=0, carried): the key is carried beside the address, so the address ring stays
     `Z/2⁶⁴` with the spreading fold.
   - **hyperbolic** (σ>0, discard): the address does not determine the key, so the address ring is an
     odd prime power `Z/q` (`q = pᵏ`, `p` odd, `2` a unit) and the quadratic idempotents
     `e± = (1 ± x/r)/2` (`r` a unit, `r² = σ`) are the two crosstalk-free channels, `e₊e₋ = 0`. A
     composite `m = ∏ qᵢ` of coprime factors additionally gives scalar CRT channels; the `≥2`-factor
     condition is required for the scalar decomposition only, not for the quadratic split. The full key,
     or a lossless codec regenerating it, is retained: "discard" is an address property, not key
     erasure, because exact-map identity requires the full `(Sigma, K)` witness.
3. The split exists exactly where the algebra supplies the witnesses `2 ∈ R^×` and a unit `r² = σ`.
   `Z/2ᵏ` has nonunit `2`, so the split is unrealizable there for every σ. A single odd prime power
   splits: `(Z/9)[x]/(x²−1) ≅ Z/9 × Z/9` via `r = s = 1`, `h = 5`; locality forbids nontrivial *scalar*
   idempotents, not the quadratic extension's `e±`. σ=0 stays local (`R[ε]/ε²`, a filtration); an
   irreducible `x²−σ` over a prime field is the unsplit field case.
4. On the hyperbolic face the component strata are necessary-condition descriptors: absence of the
   query's projected component implies absence of the query (sound exclusion), while presence of every
   component does not establish presence. Presence requires a common full-key witness on one stored
   representative, so components of different keys cannot be combined into a false positive. On the
   elliptic and parabolic faces presence is decided by the carried or decoded `(Sigma, K)` identity.
5. The `PHI` summary hash over `Z/2⁶⁴` (`summary_index`, `src/table/state.rs:127`) is deleted: it
   filters against free geometry with no algebraic statement and is the source of the dynamic-shift
   class.
6. Full `(Sigma, K)` identity decides presence. Address, channel, level and any fingerprint are
   placement or exclusion filters only; a false negative is forbidden.
7. The canonical semantics is unchanged: `get`/`insertWith`/`build` and the absence default
   (`Apeiron/Table/Basic.lean`); `Cell<i64>` remains the value/key word kernel (`word_mul`,
   `word_norm_denote`).
8. Every HashDOS claim carries a source-linked adversarial worst-case cost theorem. Removing the
   unkeyed hash does not by itself bound adversarial occupancy, and the pepper stays until the theorem
   exists.

## Verification hypotheses

| Claim | Grade | Evidence |
|---|---|---|
| `N` is multiplicative (`N(uv) = N(u)·N(v)`). | proved | `ringNormForm` and its multiplication law. |
| `N` is not additive, so it is not an address homomorphism. | proved | counterexample `N(1+1) = 4 ≠ 2`. |
| `Z/2ᵏ` is local for every σ, with only `0,1` as idempotents; no CRT split. | proved | canonical `Aeneas`-adjacent `AnyRing.ringIdempotent_trivial`; Sol ratified. |
| An odd prime power `Z/pᵏ` (p odd, unit `r² = σ`) splits the quadratic algebra through `e±`; a composite odd `m` additionally gives scalar CRT channels. | proved | `AnyRing.ringUpper_channel`/`ringLower_channel`/`ringComplementary_idem`; `(Z/9)[x]/(x²−1) ≅ Z/9×Z/9`; Sol ratified. |
| Per-face address rings: elliptic/parabolic `Z/2⁶⁴`; hyperbolic `Z/m`. | anchor/open | `Coord`/`Home` contract change; Sol ratification. |
| Hyperbolic absence of a query's projected channel implies absence of the query (sound exclusion). | proved target | projection is an endomorphism and `eᵢeⱼ = 0`; shared-representative obligation. |
| Presence requires a common full-key witness, not the conjunction of components. | proved target | `RepR`/`get_at`; `(Sigma,K)` identity. |
| Adversarial worst-case cost is bounded without an unkeyed hash. | anchor/open | source-linked worst-case cost theorem; not implied by CRT or by removing the hash. |
| Parity without the `PHI` summary. | measured target | strict full-profile command, named machine and interval. |

## Non-goals

- No threshold relaxation, historical envelope, aggregate pass, or reduced matrix.
- No change to the canonical object: `Apeiron.Table` stays a function.
- No GPU/SIMD/ISA specialization.
- No discharge of `W7-SearchCommitBoundary` from performance evidence.
- No new arbitrary partition (shards by low bits, power-of-two regions, Bloom summaries) as placement.

## Trust boundary

Evidence will not reach the allocator, executor/`std::thread::scope`, rustc/LLVM codegen, the kernel
ABI, GPU/SPIR-V, or the `unsafe` container. The spinor/double-cover reading of the unit-group action on
`N` is an `anchor` until a named corpus theorem states it; the algorithmic realization and its resource
bounds are not proved by the algebra alone.

## Open questions

- **Decided (lead):** per-face address rings. `Z/2⁶⁴` for the elliptic and parabolic faces; an odd
  composite `Z/m` with CRT idempotent channels for the hyperbolic face only; the `PHI` summary deleted;
  `Cell<i64>` kept as the value/key word kernel. The wholesale scalar pivot is rejected: a modulus-based
  address cannot be the elliptic bijection, a finite ring has no sign trichotomy, and modular reduction
  costs the hot path.
- Sol: ratify the CRT idempotent split at its exact hypotheses, the hyperbolic sound-exclusion
  obligation (shared representative, live links, complete coverage, preservation), and the adversarial
  worst-case cost statement, against the pinned corpus; name the declarations.
- Astra: revise `docs/spec/algebra-native-addressing.md` and `docs/plan/algebra-native-addressing.md`
  to the per-face boundary, the `Coord`/`Home` contract change, the serialized tasks with one gate and
  one rejecting control each, and the deletion of the summary read path.
- Lead: sequencing against the frozen `W7` and the pending summary bridge repair.
