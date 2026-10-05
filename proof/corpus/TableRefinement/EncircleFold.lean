-- STATUS: OFF the build path (no `roots` entry; nothing imports it) and NOT COMPLETE.
-- `lake env lean TableRefinement/EncircleFold.lean` exits 1 with 32 errors (measured): 27
-- `synthInstanceFailed` (failed to synthesize `CommRing R`), 3 `unsolved goals`, 2
-- `unknownIdentifier`. Committed in 439fcf0 as preserved untracked work, NOT as a finished proof. It evaluates no `#audit` row until registered, and it must not
-- be registered until these close.
-- Row: `apeiron-untracked-corpus-D95BAcvaduxXKnnKfkKPyO` -- the row this
-- file belongs to; it is off-path and incomplete, and must not be registered until the above close.
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing

/-!
# The whole-table reduce's fold law

`fold hashtrinity/src/lib.rs:891` (`Table::encircle`) reduces a whole table: it applies a `residual`
to every live value and `combine`s the results from an identity, optionally over per-task chunks
merged at a barrier. Its doc comment asserts "any task->executor assignment and any merge order give
the same answer". That assertion is a claim about the monoid, and this file discharges it.

The port is `apeiron`'s `Table::reduce_all` (`src/table/reduce.rs`) at the `AffineFoldStep` witness:
`residual(v) = <1, v>` and `combine(acc, next) = ringAffineCompose sigma next acc`, the later step
composed on the outside. Three corpus theorems carry the law:

| ported operation | corpus definition | corpus law used here |
|---|---|---|
| `AffineStep::id` | `AnyRing.ringAffineId` | `AnyRing.ringAffineCompose_id` |
| `AffineStep::compose` | `AnyRing.ringAffineCompose` | `AnyRing.ringAffineCompose_assoc` |
| `AffineStep::apply` | `AnyRing.ringAffineApply` | `AnyRing.ringAffineFold_apply` |
| `Table::reduce_all` | a left fold over live slots | `ringAffineFold_chunks` |

## No sub-step forces a face

`RingAffineStep`, `ringAffineId`, `ringAffineApply`, `ringAffineCompose` and `ringAffineFold` are
stated in the canonical corpus over `[CommRing R]` with a bare `sigma : R` and no sign hypothesis.
Every theorem below quantifies over every `sigma : R`, so the law is all-sigma and cannot be read as
a statement about one face. `foldLaw_face_witness_elliptic`, `foldLaw_face_witness_parabolic` and
`foldLaw_face_witness_hyperbolic` instantiate it at `sigma = -1, 0, 1` over `Int`, so the claim is
inhabited on each of the three faces the trichotomy names rather than merely quantified.

The corpus theorems that *do* carry a face are excluded here, not weakened:
`ringUpper_ringAffineFold` and `ringLower_ringAffineFold` each take `hr : r ^ 2 = sigma` and project
the point onto a channel through that `r`. The reduce never projects, so it needs neither, and
neither appears in this file.

## What this file does not establish

This file is algebra. It does not link `table.Table.reduce_all` in the generated extraction to these
theorems. That link needs the table representation relation of `Growable.Representation` together
with a `word_mul_denote`-to-`ringMul` instance, and it is open; it is not implied here.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Apeiron.Trichotomy

/-- **The step list the ported residual produces.**

    `AffineFoldStep::residual(v) = <1, v>`, so each stored cell becomes the injection of the step
    that translates by it and the coefficient is the unit. Over a commutative ring that is the only
    shape the port's residual has, so it is named once here and reused by every theorem below. -/
def encircleSteps (vals : List (AnyRing.RingCell R)) : List (AnyRing.RingAffineStep R) :=
  vals.map fun v => ⟨AnyRing.ringOne, v⟩

/-- **The reduce equals the corpus fold of the steps it composed.**

    `ringAffineFold_apply` in the canonical corpus, at the instance the port realises. Not `rfl`:
    the left side builds one composed step and the right side applies the steps one at a time, and
    showing those agree is exactly `ringAffineFold_apply`'s content. -/
theorem foldLaw_reduce_eq_ringAffineFold (sigma : R) (vals : List (AnyRing.RingCell R))
    (z : AnyRing.RingCell R) :
    AnyRing.ringAffineApply sigma
        (AnyRing.ringAffineFold sigma (encircleSteps vals)) z =
      (encircleSteps vals).foldl (fun acc step => AnyRing.ringAffineApply sigma step acc) z :=
  AnyRing.ringAffineFold_apply sigma _ z

/-- **Composing a step on the outside of the identity leaves it alone.**

    The port's `combine` starts from `AffineStep::id`, so this is the first step of the reduce, and
    it is what lets `chunks_from` absorb a chunk's identity before merging it into the prefix. -/
theorem foldLaw_identity_neutral (sigma : R) (step : AnyRing.RingAffineStep R) :
    AnyRing.ringAffineCompose sigma step AnyRing.ringAffineId R = step :=
  AnyRing.ringAffineCompose_id sigma step

/-- **Chunked merging, from an arbitrary accumulator.**

    The ported merge with a free initial accumulator. The accumulator is free because a merge over
    chunks starts its second chunk from the first chunk's result, not from the identity. -/
theorem chunks_from (sigma : R) (parts : List (List (AnyRing.RingAffineStep R)))
    (acc : AnyRing.RingAffineStep R) :
    (parts.map (AnyRing.ringAffineFold sigma)).foldl
        (fun acc part => AnyRing.ringAffineCompose sigma part acc) acc
      = AnyRing.ringAffineCompose sigma (AnyRing.ringAffineFold sigma parts.flatten) acc := by
  induction parts with
  | nil =>
      simp [List.map_nil, List.foldl_nil, AnyRing.ringAffineFold]
  | cons chunk rest ih =>
      rw [List.map_cons, List.foldl_cons, ih, List.flatten_cons,
        AnyRing.ringAffineFold_append]
      exact (AnyRing.ringAffineCompose_assoc sigma _ _ _).symm

/-- **The incumbent's parallel branch is licensed: any chunking gives the flat fold.**

    `fold hashtrinity/src/lib.rs:891` splits the live slots into per-task chunks, folds each chunk
    from the identity, and merges the partials in task order. This says the merged partials are the
    flat fold of the concatenated chunks, so the answer does not depend on how the work was split.

    The statement constrains only the order of the chunks, never their lengths. That is what the
    incumbent's "more tasks than executors so uneven live-density self-balances" relies on. -/
theorem ringAffineFold_chunks (sigma : R) (parts : List (List (AnyRing.RingAffineStep R))) :
    (parts.map (AnyRing.ringAffineFold sigma)).foldl
        (fun acc part => AnyRing.ringAffineCompose sigma part acc) AnyRing.ringAffineId R
      = AnyRing.ringAffineFold sigma parts.flatten := by
  rw [chunks_from, AnyRing.ringAffineCompose_id]

/-- **The chunk merge is schedule-free at the point, not only at the step.**

    Applying the merged partial to `z` equals applying the flat fold to `z`, which is what the port
    actually returns. -/
theorem foldLaw_chunk_points (sigma : R) (parts : List (List (AnyRing.RingAffineStep R)))
    (z : AnyRing.RingCell R) :
    AnyRing.ringAffineApply sigma
        ((parts.map (AnyRing.ringAffineFold sigma)).foldl
          (fun acc part => AnyRing.ringAffineCompose sigma part acc) AnyRing.ringAffineId R) z
      = AnyRing.ringAffineApply sigma (AnyRing.ringAffineFold sigma parts.flatten) z := by
  rw [ringAffineFold_chunks]

/-- **The law holds at the elliptic face.** `sigma = -1`, the complex numbers. -/
theorem foldLaw_face_witness_elliptic (steps : List (AnyRing.RingAffineStep Int))
    (z : AnyRing.RingCell Int) :
    AnyRing.ringAffineApply (-1 : Int) (AnyRing.ringAffineFold (-1 : Int) steps) z =
      steps.foldl (fun acc step => AnyRing.ringAffineApply (-1 : Int) step acc) z :=
  AnyRing.ringAffineFold_apply _ _ _

/-- **The law holds at the parabolic face.** `sigma = 0`, the dual numbers. -/
theorem foldLaw_face_witness_parabolic (steps : List (AnyRing.RingAffineStep Int))
    (z : AnyRing.RingCell Int) :
    AnyRing.ringAffineApply (0 : Int) (AnyRing.ringAffineFold (0 : Int) steps) z =
      steps.foldl (fun acc step => AnyRing.ringAffineApply (0 : Int) step acc) z :=
  AnyRing.ringAffineFold_apply _ _ _

/-- **The law holds at the hyperbolic face.** `sigma = 1`, the split-complex numbers. -/
theorem foldLaw_face_witness_hyperbolic (steps : List (AnyRing.RingAffineStep Int))
    (z : AnyRing.RingCell Int) :
    AnyRing.ringAffineApply (1 : Int) (AnyRing.ringAffineFold (1 : Int) steps) z =
      steps.foldl (fun acc step => AnyRing.ringAffineApply (1 : Int) step acc) z :=
  AnyRing.ringAffineFold_apply _ _ _

end Apeiron.TableRefinement