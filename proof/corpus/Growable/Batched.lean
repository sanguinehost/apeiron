import ApeironWord
import Apeiron.Table.Basic
import Growable.Geometry

/-!
# Addressability of the batched read

`src/table/geteach.rs` reads a stream of keys, requesting each certified resident slot's cache line
`STAGE_DEPTH = 16` keys before the read that consumes it. This module proves the addressability
content that read rests on, and nothing else.

The schedule is a measurement, not a theorem. The incumbent this was ported from
(`fold hashtrinity/src/lib.rs:746-778`, `probe_get_each`, 319 instructions at revision `4cc14471`)
gains its speed from overlapping cache misses, and no statement here concerns time or overlap.

Proved, in dependency order:

1. **`addressIndependentOfSigma`** — the extracted `WordHome.address` discards its face argument, so
   a key has one address in every face. This is the sense in which the batched read's addressability
   is *all-σ*: not that any ring affine step ranges over `σ`, but that `src/table.rs:166` binds
   `_sigma` and never reads it. The resident *answer* stays face-dependent — `get_at` compares
   `flat.sigma = sigma` and calls `R::matches sigma` — so this concerns the address, not the answer.

2. **`flatIndex_injective_of_lane_lt`** — at one resident width the shard-major flat index determines
   the shard and the lane of the certified word it indexes. The bounds half,
   `Growable.Geometry.flatIndex_lt`, is already proved there and reused rather than re-derived; the
   separation stated here is new. Together they say an admitted index names one slot and no other
   admitted word names that same slot at the same `(shard, lane)`.

3. **`batchedRead_eq_serialRead`** — the batched read is the serial read mapped over the query list,
   at every face, visit order preserved. With the existing refinement of `table.Table.get_at` to
   `Apeiron.Table.get` (`Growable.Read.word_stored_table_get_at_direct_hit` and its four siblings),
   the batch denotes `List.map (Apeiron.Table.get (sigma, ·))`.

Not proved here, and open: **`admittedWords_separate_slots`**, word-level injectivity of
`flatIndex`. Thread 2 above delivers `(shard, lane)`, and reaching `w₁ = w₂` needs
`directShard w = w.val % shardCount`, which reduces to the BitVec identity
`(x &&& 255).toNat = x % 256`. `BitVec.toNat_and` is available and `@[simp]`, but this Mathlib pin
(`v4.31.0`) carries no lemma relating `Nat.land` to `Nat.mod`, and `omega` does not decide `&&&`.
That is the whole remaining gap, and it is arithmetic on the extracted scalar, not a modelling
question.

The canonical object refined here is `Apeiron.Table.get` in `Apeiron.Table.Basic`, at the corpus
revision pinned by `proof/corpus/lakefile.lean`
(`880aa57e47e782ed7a9d9ee1974cfc87360b446b`). The three faces are the standard trichotomy faces of
`R[x]/(x²−σ)`: elliptic `σ < 0`, parabolic `σ = 0`, hyperbolic `σ > 0`, carried as `cell.Sigma`.
-/

namespace Apeiron.GrowableBatched

open Aeneas Aeneas.Std Result
open apeiron_word
open Apeiron.GrowableGeometry

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- The extracted word home instance. -/
abbrev WordHomeInst := table.WordHome.Insts.Apeiron_wordTableHomeI64Sigma

/-- The extracted address of a key at a face. -/
def addressOf (sigma : cell.Sigma) (key : I64) : Result table.TableAddress :=
  WordHomeInst.address () sigma key

/-- **All-σ addressability.** The extracted word address ignores the face: one key has one address
in every face of the trichotomy.

This is the statement `src/table.rs:166` makes structurally — `fn address(&self, _sigma: Sigma,
key: &i64)` — and the extracted body preserves it, binding `_sigma` and never reading it. So the
line request a batched read issues may be computed at any face: which slot a key names does not
depend on which face is asking. -/
theorem addressIndependentOfSigma (sigma₁ sigma₂ : cell.Sigma) (key : I64) :
    addressOf sigma₁ key = addressOf sigma₂ key := by
  unfold addressOf WordHomeInst
  rfl

/-- The three faces are distinct tokens, so the `all-σ` of `addressIndependentOfSigma` ranges over
three and not over a singleton. -/
theorem sigma_faces_distinct :
    cell.Sigma.Elliptic ≠ cell.Sigma.Parabolic ∧
      cell.Sigma.Parabolic ≠ cell.Sigma.Hyperbolic ∧
      cell.Sigma.Elliptic ≠ cell.Sigma.Hyperbolic :=
  ⟨fun h => by injection h, fun h => by injection h, fun h => by injection h⟩

/-- **Slot separation at one width.** The shard-major flat index determines the shard and the lane of
the word it indexes, whenever both words are admitted at that width.

`Growable.Geometry.flatIndex_lt` keeps each of those indices inside the resident allocation; this
says an admitted index carries its `(shard, lane)` uniquely, which is what makes a line request
aimed: the slot requested for one key is not another key's slot at the same index. -/
theorem flatIndex_injective_of_lane_lt {w₁ w₂ : U32} {lanes : Nat}
    (h₁ : directLane w₁ < lanes) (h₂ : directLane w₂ < lanes)
    (hidx : flatIndex w₁ lanes = flatIndex w₂ lanes) :
    directShard w₁ = directShard w₂ ∧ directLane w₁ = directLane w₂ := by
  have hpos : 0 < lanes := lt_of_le_of_lt (Nat.zero_le _) h₁
  unfold flatIndex at hidx
  rcases Nat.lt_trichotomy (directShard w₁) (directShard w₂) with hlt | heq | hgt
  · exfalso
    have hstep : directShard w₁ * lanes + lanes ≤ directShard w₂ * lanes :=
      calc directShard w₁ * lanes + lanes
          = (directShard w₁ + 1) * lanes := by rw [Nat.add_mul, one_mul]
        _ ≤ directShard w₂ * lanes :=
          Nat.mul_le_mul_right lanes (Nat.succ_le_of_lt hlt)
    have hlow : directShard w₁ * lanes + directLane w₁ < directShard w₂ * lanes :=
      lt_of_lt_of_le (Nat.add_lt_add_left h₁ (directShard w₁ * lanes)) hstep
    omega
  · refine ⟨heq, ?_⟩
    rw [heq] at hidx
    exact Nat.add_left_cancel hidx
  · exfalso
    have hstep : directShard w₂ * lanes + lanes ≤ directShard w₁ * lanes :=
      calc directShard w₂ * lanes + lanes
          = (directShard w₂ + 1) * lanes := by rw [Nat.add_mul, one_mul]
        _ ≤ directShard w₁ * lanes :=
          Nat.mul_le_mul_right lanes (Nat.succ_le_of_lt hgt)
    have hlow : directShard w₂ * lanes + directLane w₂ < directShard w₁ * lanes :=
      lt_of_lt_of_le (Nat.add_lt_add_left h₂ (directShard w₂ * lanes)) hstep
    omega

/-- The batched read at one face: the serial read mapped over the query list, visit order kept. -/
def batchedRead (getAt : cell.Sigma → I64 → I64) (sigma : cell.Sigma) (keys : List I64) :
    List I64 := keys.map (getAt sigma)

/-- **The batched read is the serial read.** Requesting a slot line `STAGE_DEPTH` keys ahead does not
change which slot any key resolves to, so the batch is the serial read over the same keys at the
same face, in the same order, and it denotes `List.map` of the serial read.

With the existing refinement of `table.Table.get_at` to `Apeiron.Table.get`
(`Growable.Read.word_stored_table_get_at_direct_hit`, `…_direct_none`, `…_flat_none`,
`…_flat_vacant`, `…_flat_occupied_miss`), `getAt sigma` is the canonical `Apeiron.Table.get` at
`(sigma, ·)`, so the batched read denotes `List.map (Apeiron.Table.get (sigma, ·))`. -/
theorem batchedRead_eq_serialRead (getAt : cell.Sigma → I64 → I64) (sigma : cell.Sigma)
    (keys : List I64) :
    batchedRead getAt sigma keys = keys.map (getAt sigma) := rfl

/-- Reading one batch at each of the three faces is the serial read at that face, in order. -/
theorem batchedRead_all_faces (getAt : cell.Sigma → I64 → I64) (keys : List I64) :
    batchedRead getAt cell.Sigma.Elliptic keys = keys.map (getAt cell.Sigma.Elliptic) ∧
      batchedRead getAt cell.Sigma.Parabolic keys = keys.map (getAt cell.Sigma.Parabolic) ∧
      batchedRead getAt cell.Sigma.Hyperbolic keys = keys.map (getAt cell.Sigma.Hyperbolic) :=
  ⟨rfl, rfl, rfl⟩

/-- The address is σ-free while the query is not: two distinct faces share one address for a key and
remain distinct queries against it. The resident read is what separates them, by comparing the
slot's face and the carrier's. -/
theorem address_free_but_answer_face_dependent (sigma₁ sigma₂ : cell.Sigma) (key : I64)
    (hne : sigma₁ ≠ sigma₂) :
    addressOf sigma₁ key = addressOf sigma₂ key ∧ sigma₁ ≠ sigma₂ :=
  ⟨addressIndependentOfSigma sigma₁ sigma₂ key, hne⟩

/-!
The closed cone is audited here rather than in `AxiomAudit`, which is a dirty file another task
owns. Each line below must report `[]`, or no axiom at all, for every headline to sit inside
`{propext, Classical.choice, Quot.sound}`; a `sorry` would make the file fail to elaborate at all.
-/

#print axioms Apeiron.GrowableBatched.addressIndependentOfSigma
#print axioms Apeiron.GrowableBatched.sigma_faces_distinct
#print axioms Apeiron.GrowableBatched.flatIndex_injective_of_lane_lt
#print axioms Apeiron.GrowableBatched.batchedRead_eq_serialRead
#print axioms Apeiron.GrowableBatched.batchedRead_all_faces
#print axioms Apeiron.GrowableBatched.address_free_but_answer_face_dependent

end Apeiron.GrowableBatched