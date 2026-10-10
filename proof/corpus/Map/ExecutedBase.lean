-- Shared lemmas for the executed map face.
import ApeironWord
import Apeiron.Table.Basic
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases
open Aeneas Aeneas.Std Result

/-!
# Shared lemmas for the executed map face

The executed map face is `hashtrinity::Table<u64, i64, KS>` (`src/hashtrinity/mod.rs`), extracted
as `hashtrinity.Table` in `proof/ApeironWord.lean` and rooted at `word_hashtrinity_*` (`Stored<u64>`)
and `word_coordmap_*` (`Regen<u64>`).  This module holds the generic reasoning tools the face's
proofs use:

* `loop_pc`: partial correctness of `Aeneas.Std.loop` by fixpoint induction (admissibility
  `pspec_admissible`), and `bind_inv`, the inversion of a successful bind.  Both are the lemmas of
  `proof/corpus/Count/Aggregate.lean` (count lane, `wt-count-2d`), restated here because that module
  is not on this tree's build path.
* inversion lemmas for checked scalar operations (`uadd_ok`, `usub_ok`, `umul_ok`, `udiv_ok`,
  `ushr_ok`) and for slice/`MaybeUninit` accesses, taken from the same lane's `Count/FreqAdd.lean`.
* the values of the reserved coordinates `EMPTY = 2^64 - 1` and `TOMB = 2^64 - 2`, and the
  executed `is_sentinel` / `is_live` tests as decisions on those values.

Every statement about an extracted body has the form `body = ok r → …` (partial correctness): a
failing or diverging run establishes nothing.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open Lean.Order Aeneas.Data.Coinductive
open apeiron_word

unseal Result in
theorem pspec_admissible {α : Type} (P : α → Prop) :
    admissible (fun (x : Result α) => ∀ a, x = .ok a → P a) := by
  intro c hchain h a ha
  by_cases hex : ∃ t, c t ∧ t ≠ (Result.div : Result α)
  · obtain ⟨t, ct, tne⟩ := hex
    have hle : t ⊑ CCPO.csup hchain := le_csup hchain ct
    rw [ha] at hle
    have hu : t ⊑ (Result.ok a : Result α) := hle
    have hu' := ITree.le_unfold t (Result.ok a) ▸ hu
    rcases hu' with h1 | ⟨r, h1, h2⟩ | ⟨i, t1', t2', h1, h2, _⟩
    · exact absurd h1 tne
    · have : r = a := by
        exact (Aeneas.Data.Coinductive.ret_inj.mp h2).symm
      subst this
      exact h t ct r h1
    · exact absurd h2 (by intro hh; exact ok_not_vis hh)
  · have hbot : CCPO.csup hchain = (Result.div : Result α) := by
      have hle : CCPO.csup hchain ⊑ (Result.div : Result α) :=
        csup_le hchain (fun y cy => by
          have : y = (Result.div : Result α) := by
            by_contra hne
            exact hex ⟨y, cy, hne⟩
          rw [this]
          exact PartialOrder.rel_refl)
      exact ITree.le_div_is_div _ hle
    rw [hbot] at ha
    exact absurd ha div_not_ok

theorem bind_inv {α β : Type} {e : Result α} {f : α → Result β} {r : β}
    (h : (do let a ← e; f a) = ok r) : ∃ a, e = ok a ∧ f a = ok r := by
  induction e using Result.cases with
  | ret a => exact ⟨a, rfl, by simpa using h⟩
  | vis i k =>
    exfalso
    simp [Bind.bind, Aeneas.Std.bind_vis] at h
  | div => simp at h

/-- Partial correctness of `loop`: if the loop returns `ok r`, every state it visits satisfies `inv`
and `r` satisfies `post`.  No termination or non-failure is assumed. -/
theorem loop_pc {α β : Type} (inv : α → Prop) (post : β → Prop)
    (body : α → Result (ControlFlow α β))
    (hB : ∀ x, inv x → ∀ cf, body x = ok cf →
      match cf with | .done y => post y | .cont x' => inv x')
    (x : α) (hx : inv x) (r : β) (h : loop body x = ok r) : post r := by
  have key : ∀ x, ∀ r, loop body x = ok r → (inv x → post r) := by
    refine loop.fixpoint_induct body (fun f => ∀ x, ∀ r, f x = ok r → (inv x → post r)) ?_ ?_
    · apply admissible_pi_apply (fun x (y : Result β) => ∀ r, y = ok r → (inv x → post r))
      intro x
      exact pspec_admissible (fun r => inv x → post r)
    · intro g ih x r h hx
      obtain ⟨cf, hb, hk⟩ := bind_inv h
      have := hB x hx cf hb
      cases cf with
      | cont x' => exact ih x' r hk this
      | done y =>
        have : y = r := Result.ok_injective hk
        subst this
        exact ‹post y›
  exact key x r h hx

/-! ## Checked scalar operations -/

theorem uadd_ok {ty} {x y z : UScalar ty} (h : x + y = ok z) :
    z.val = x.val + y.val ∧ x.val + y.val < 2^ty.numBits := by
  have := UScalar.add_equiv x y
  simp only [h, Result.match.ok] at this
  exact ⟨this.2.1, this.1⟩

theorem usub_ok {ty} {x y z : UScalar ty} (h : x - y = ok z) :
    x.val = z.val + y.val := by
  have := UScalar.sub_equiv x y
  simp only [h, Result.match.ok] at this
  exact this.2.1

theorem umul_ok {ty} {x y z : UScalar ty} (h : x * y = ok z) :
    z.val = x.val * y.val ∧ x.val * y.val ≤ UScalar.max ty := by
  have := UScalar.mul_equiv x y
  have h' : UScalar.mul x y = ok z := h
  simp only [h', Result.match.ok] at this
  exact ⟨this.2.1, this.1⟩

theorem udiv_ok {ty} {x y z : UScalar ty} (h : x / y = ok z) :
    y.val ≠ 0 ∧ z.val = x.val / y.val := by
  have h' : UScalar.div x y = ok z := h
  unfold UScalar.div at h'
  by_cases hy0 : y.bv = 0#ty.numBits
  · simp [hy0] at h'
  · have hy : (y.bv != 0) = true := by simpa using hy0
    simp only [hy, if_true] at h'
    have hz : z = ⟨BitVec.udiv x.bv y.bv⟩ := (Result.ok_injective h').symm
    refine ⟨?_, ?_⟩
    · intro h0
      apply hy0
      apply BitVec.eq_of_toNat_eq
      simpa using h0
    · rw [hz]
      show (BitVec.udiv x.bv y.bv).toNat = x.bv.toNat / y.bv.toNat
      exact BitVec.toNat_udiv

theorem ushr_ok {ty tys} {x : UScalar ty} {s : UScalar tys} {z : UScalar ty}
    (h : x >>> s = ok z) : s.val < ty.numBits ∧ z.val = x.val >>> s.val := by
  have h' : UScalar.shiftRight_UScalar x s = ok z := h
  unfold UScalar.shiftRight_UScalar UScalar.shiftRight at h'
  by_cases hs : s.val < ty.numBits
  · simp only [hs, if_true] at h'
    have hz : z = ⟨x.bv.ushiftRight s.val⟩ := (Result.ok_injective h').symm
    refine ⟨hs, ?_⟩
    rw [hz]
    show (x.bv.ushiftRight s.val).toNat = x.bv.toNat >>> s.val
    exact BitVec.toNat_ushiftRight _ _
  · simp [hs] at h'

theorem ushl_ok {ty tys} {x : UScalar ty} {s : UScalar tys} {z : UScalar ty}
    (h : x <<< s = ok z) : s.val < ty.numBits ∧ z.val = (x.val * 2^s.val) % 2^ty.numBits := by
  have h' : UScalar.shiftLeft_UScalar x s = ok z := h
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft at h'
  by_cases hs : s.val < ty.numBits
  · simp only [hs, if_true] at h'
    have hz : z = ⟨x.bv.shiftLeft s.val⟩ := (Result.ok_injective h').symm
    refine ⟨hs, ?_⟩
    rw [hz]
    show (x.bv.shiftLeft s.val).toNat = (x.bv.toNat * 2^s.val) % 2^ty.numBits
    exact BitVec.toNat_shiftLeft.trans (by rw [Nat.shiftLeft_eq])
  · simp [hs] at h'

/-! ## Slices, vectors and `MaybeUninit` -/

theorem slice_index_ok {α : Type} (s : Slice α) (i : Usize) (x : α) (h : s.index_usize i = ok x) :
    s.val[i.val]? = some x := by
  unfold Slice.index_usize at h
  cases hh : s[i]? with
  | none => simp [hh] at h
  | some y =>
    simp [hh] at h
    subst h
    rfl

theorem get_unchecked_ok {α : Type} (s : Slice α) (i : Usize) (x : α)
    (h : core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice α) s i = ok x) :
    s.val[i.val]? = some x := by
  exact slice_index_ok s i x h

theorem get_unchecked_mut_ok {α : Type} (s : Slice α) (i : Usize) (x : α) (back : α → Slice α)
    (h : core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice α) s i
      = ok (x, back)) :
    s.val[i.val]? = some x ∧ back = s.set i := by
  unfold core.slice.Slice.get_unchecked_mut at h
  simp only [core.slice.index.Usize.index_mut, Slice.index_mut_usize] at h
  obtain ⟨y, hy, h⟩ := bind_inv h
  have := Result.ok_injective h
  simp only [Prod.mk.injEq] at this
  obtain ⟨h1, h2⟩ := this
  subst h1
  exact ⟨slice_index_ok s i y hy, h2.symm⟩

theorem slice_set_getElem {α : Type} (s : Slice α) (i : Usize) (x : α) (k : Nat) :
    (s.set i x).val[k]? = if k = i.val ∧ i.val < s.length then some x else s.val[k]? := by
  rw [Slice.set_val_eq, List.getElem?_set]
  by_cases hk : i.val = k
  · subst hk
    by_cases hi : i.val < s.length
    · simp [hi]
    · have : s.val.length ≤ i.val := by simp [Slice.length] at hi; omega
      simp [hi]
  · simp [hk, Ne.symm hk]

theorem slice_set_length {α : Type} (s : Slice α) (i : Usize) (x : α) :
    (s.set i x).length = s.length := by
  simp [Slice.length, Slice.set_val_eq]

theorem assume_init_read_ok {T : Type} (m : core.mem.maybe_uninit.MaybeUninit T) (v : T)
    (h : core.mem.maybe_uninit.MaybeUninit.assume_init_read m = ok v) : m = some v := by
  unfold core.mem.maybe_uninit.MaybeUninit.assume_init_read at h
  cases m with
  | none => exact absurd h (by intro hh; exact ok_not_fail hh.symm)
  | some x => simp at h; rw [h]

theorem assume_init_ref_ok {T : Type} (m : core.mem.maybe_uninit.MaybeUninit T) (v : T)
    (h : core.mem.maybe_uninit.MaybeUninit.assume_init_ref m = ok v) : m = some v := by
  unfold core.mem.maybe_uninit.MaybeUninit.assume_init_ref at h
  cases m with
  | none => exact absurd h (by intro hh; exact ok_not_fail hh.symm)
  | some x => simp at h; rw [h]

theorem vec_push_ok {α : Type} (v v1 : alloc.vec.Vec α) (x : α)
    (h : alloc.vec.Vec.push v x = ok v1) : v1.val = v.val ++ [x] := by
  unfold alloc.vec.Vec.push at h
  dsimp only at h
  split at h
  · have := (Result.ok_injective h).symm
    subst this
    simp
  · simp at h

theorem vec_index_ok {α : Type} (v : alloc.vec.Vec α) (i : Usize) (x : α)
    (h : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v i = ok x) :
    v.val[i.val]? = some x := by
  unfold alloc.vec.Vec.index at h
  exact slice_index_ok v.slice i x h

/-! ## The reserved coordinates -/

/-- `EMPTY = u64::MAX`, the empty-slot marker. -/
def EMPTYN : Nat := 2^64 - 1
/-- `TOMB = EMPTY - 1`, the tombstone marker. -/
def TOMBN : Nat := 2^64 - 2

/-- A coordinate is reserved (held in the side store, never in `coords`) iff it is `EMPTY` or
`TOMB`. -/
def IsSent (c : U64) : Prop := c.val = EMPTYN ∨ c.val = TOMBN

instance (c : U64) : Decidable (IsSent c) := by unfold IsSent; infer_instance

theorem EMPTY_val : hashtrinity.EMPTY.val = EMPTYN := by
  unfold hashtrinity.EMPTY EMPTYN
  rfl

theorem EMPTY_eq_iff (c : U64) : c = hashtrinity.EMPTY ↔ c.val = EMPTYN := by
  constructor
  · intro h; rw [h, EMPTY_val]
  · intro h; exact UScalar.eq_of_val_eq (by rw [h, EMPTY_val])

theorem TOMB_ok : ∃ t : U64, hashtrinity.TOMB = ok t ∧ t.val = TOMBN := by
  unfold hashtrinity.TOMB
  have h1 : hashtrinity.EMPTY.val = 2^64 - 1 := EMPTY_val
  have h3 : (1#u64).val = 1 := rfl
  obtain ⟨t, ht, hv, _⟩ := WP.spec_imp_exists
    (UScalar.sub_spec (x := hashtrinity.EMPTY) (y := 1#u64) (by rw [h1, h3]; norm_num))
  refine ⟨t, ht, ?_⟩
  rw [hv, h1, h3]
  rfl

theorem is_sentinel_ok (c : U64) : hashtrinity.is_sentinel c = ok (decide (IsSent c)) := by
  obtain ⟨t, ht, htv⟩ := TOMB_ok
  unfold hashtrinity.is_sentinel
  by_cases he : c = hashtrinity.EMPTY
  · rw [if_pos he]
    have : IsSent c := Or.inl ((EMPTY_eq_iff c).mp he)
    simp [this]
  · rw [if_neg he, ht]
    have hne : c.val ≠ EMPTYN := fun h => he ((EMPTY_eq_iff c).mpr h)
    simp only [bind_tc_ok]
    congr 1
    have : (c = t) ↔ IsSent c := by
      constructor
      · intro h; exact Or.inr (by rw [h, htv])
      · rintro (h | h)
        · exact absurd h hne
        · exact UScalar.eq_of_val_eq (by rw [h, htv])
    exact decide_eq_decide.mpr this

theorem is_live_ok (c : U64) : hashtrinity.is_live c = ok (decide (¬ IsSent c)) := by
  obtain ⟨t, ht, htv⟩ := TOMB_ok
  unfold hashtrinity.is_live
  by_cases he : c = hashtrinity.EMPTY
  · have : (c != hashtrinity.EMPTY) = false := by simp [he]
    rw [this]
    have hs : IsSent c := Or.inl ((EMPTY_eq_iff c).mp he)
    simp [hs]
  · have : (c != hashtrinity.EMPTY) = true := by simp [he]
    rw [this, if_pos rfl, ht]
    have hne : c.val ≠ EMPTYN := fun h => he ((EMPTY_eq_iff c).mpr h)
    simp only [bind_tc_ok]
    congr 1
    have : (c != t) = true ↔ ¬ IsSent c := by
      constructor
      · intro h hs
        rcases hs with hs | hs
        · exact hne hs
        · have : c = t := UScalar.eq_of_val_eq (by rw [hs, htv])
          simp [this] at h
      · intro h
        have : c ≠ t := fun hh => h (Or.inr (by rw [hh, htv]))
        simpa using this
    by_cases hs : IsSent c
    · have : ¬ (c != t) = true := fun h => (this.mp h) hs
      simp only [Bool.not_eq_true] at this
      rw [this]; simp [hs]
    · have : (c != t) = true := this.mpr hs
      rw [this]; simp [hs]

theorem sent_ne_empty_tomb (c : U64) (h : ¬ IsSent c) : c.val ≠ EMPTYN ∧ c.val ≠ TOMBN := by
  unfold IsSent at h
  omega

/-- `Stored<u64>::matches` is `true` without reading the key column: `u64`'s coordinate is
injective (`INJECTIVE = true`, `src/hashtrinity/algebra.rs`). -/
theorem stored_matches_ok (keys : hashtrinity.algebra.Stored U64) (i : Usize) (c k : U64) :
    hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.matches
      U64.Insts.Apeiron_wordHashtrinityAlgebraCoord core.cmp.EqU64 keys i c k = ok true := by
  have hinj : U64.Insts.Apeiron_wordHashtrinityAlgebraCoord.INJECTIVE = true := by
    unfold U64.Insts.Apeiron_wordHashtrinityAlgebraCoord.INJECTIVE; rfl
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.matches
  simp [hinj]

/-- `Regen<u64>::matches` is `true`: the coordinate round-trips. -/
theorem regen_matches_ok (keys : hashtrinity.algebra.Regen U64) (i : Usize) (c k : U64) :
    hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.matches
      U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64 keys i c k = ok true := by
  rfl

end Apeiron.MapExecuted
