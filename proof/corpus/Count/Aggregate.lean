-- Row: `p0-count-body-get-the-a-EF9n3OLyqnX2ouQQWL8w9C` (COUNT-BODY step 2d): the count table's representation and `count_aggregate`'s refinement.
import ApeironWord
import Count.Helpers
import Count.FreqAdd
import Apeiron.Table.Basic
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
open Aeneas Aeneas.Std Result apeiron_word

/-!
# Refinement of `count_aggregate`

This module proves that `apeiron_word.hashtrinity.freq.FreqMapKI64.count_aggregate`:
* Refines window fold counts: `count_aggregate_refines_win`.
* Refines the count function on folds: `count_aggregate_refines`.
* Preserves exact key counts for keys below `2^64 - 3`: `cntI_key`.
* Refines the canonical `Apeiron.Table.build` and `Apeiron.Table.get`: `build_count` and `count_aggregate_canonical`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive

/-! ## `count_aggregate` -/

/-- The fold counts of a window of keys: `#{x ∈ w | fold x = f}`. -/
def cntW (w : List U64) (f : Nat) : Nat := (w.filter (fun x => foldN x.val = f)).length

/-- The window `routed[lo..min(hi, len))`. -/
def win (routed : Slice U64) (lo hi : Nat) : List U64 :=
  (routed.val.take (min hi routed.length)).drop lo

theorem cntW_snoc (w : List U64) (x : U64) (f : Nat) :
    cntW (w ++ [x]) f = if foldN x.val = f then cntW w f + 1 else cntW w f := by
  unfold cntW
  rw [List.filter_append]
  by_cases h : foldN x.val = f
  · simp [h]
  · simp [h]

theorem count_body_spec (routed : Slice U64) (lo hi : Usize) (M : Nat)
    (hM : M = max lo.val (min hi.val routed.length))
    (st : hashtrinity.freq.FreqMap U64 Std.I64 × Usize)
    (hI : LineRep st.1.inner (cntW ((routed.val.take st.2.val).drop lo.val)) ∧
      lo.val ≤ st.2.val ∧ st.2.val ≤ M)
    (cf : ControlFlow (hashtrinity.freq.FreqMap U64 Std.I64 × Usize) (hashtrinity.freq.FreqMap U64 Std.I64))
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate_loop.body
      U64.Insts.Apeiron_wordHashtrinityAlgebraCoord routed hi st.1 st.2 = ok cf) :
    match cf with
    | .done m' => LineRep m'.inner (cntW ((routed.val.take M).drop lo.val))
    | .cont st' => LineRep st'.1.inner (cntW ((routed.val.take st'.2.val).drop lo.val)) ∧
        lo.val ≤ st'.2.val ∧ st'.2.val ≤ M := by
  obtain ⟨m, i⟩ := st
  obtain ⟨hR, hlo, hiM⟩ := hI
  dsimp only at hR hlo hiM h
  have e1 : (↑routed : List U64).length = routed.length := rfl
  unfold hashtrinity.freq.FreqMapKI64.count_aggregate_loop.body at h
  have hdone : ¬ (i.val < hi.val ∧ i.val < routed.length) → i.val = M := by
    intro hh
    rw [hM] at hiM ⊢
    omega
  by_cases hlt : i < hi
  · rw [if_pos hlt] at h
    have hlt' : i.val < hi.val := (UScalar.lt_equiv i hi).mp hlt
    change (if i < Slice.len routed then _ else _) = _ at h
    by_cases hlt2 : i < Slice.len routed
    · rw [if_pos hlt2] at h
      have hlt2' : i.val < routed.length := by
        have := (UScalar.lt_equiv i (Slice.len routed)).mp hlt2
        rwa [Slice.len_val] at this
      obtain ⟨t, ht, h⟩ := bind_inv h
      obtain ⟨m1, hm1, h⟩ := bind_inv h
      obtain ⟨i2, hi2, h⟩ := bind_inv h
      have hcf : cf = ControlFlow.cont (m1, i2) := (Result.ok_injective h).symm
      subst hcf
      have hi2v : i2.val = i.val + 1 := by
        have := (uadd_ok hi2).1
        have e : (1#usize).val = 1 := rfl
        rw [e] at this; exact this
      have hts := slice_index_ok _ _ _ ht
      have hadd := add_spec m m1 _ hR t hm1
      have hwin : (routed.val.take (i.val + 1)).drop lo.val
          = (routed.val.take i.val).drop lo.val ++ [t] := by
        rw [List.take_add_one, hts]
        simp only [Option.toList_some]
        rw [List.drop_append_of_le_length]
        rw [List.length_take]; omega
      have hfun : (fun x => if x = foldN t.val then cntW ((routed.val.take i.val).drop lo.val) x + 1
          else cntW ((routed.val.take i.val).drop lo.val) x)
          = cntW ((routed.val.take (i.val + 1)).drop lo.val) := by
        funext x
        rw [hwin, cntW_snoc]
        by_cases hx : x = foldN t.val
        · simp [hx]
        · have : ¬ foldN t.val = x := fun hh => hx hh.symm
          simp [hx, this]
      rw [hfun] at hadd
      refine ⟨?_, show lo.val ≤ i2.val by omega, show i2.val ≤ M by omega⟩
      show LineRep m1.inner (cntW ((routed.val.take i2.val).drop lo.val))
      rw [hi2v]; exact hadd
    · rw [if_neg hlt2] at h
      have hcf : cf = ControlFlow.done m := (Result.ok_injective h).symm
      subst hcf
      have hlt2' : ¬ i.val < routed.length := by
        intro hh; apply hlt2
        apply (UScalar.lt_equiv i (Slice.len routed)).mpr
        rwa [Slice.len_val]
      have := hdone (fun hh => hlt2' hh.2)
      show LineRep m.inner (cntW ((routed.val.take M).drop lo.val))
      rw [← this]; exact hR
  · rw [if_neg hlt] at h
    have hcf : cf = ControlFlow.done m := (Result.ok_injective h).symm
    subst hcf
    have hlt' : ¬ i.val < hi.val := fun hh => hlt ((UScalar.lt_equiv i hi).mpr hh)
    have := hdone (fun hh => hlt' hh.1)
    show LineRep m.inner (cntW ((routed.val.take M).drop lo.val))
    rw [← this]; exact hR

theorem win_eq (routed : Slice U64) (lo hi : Nat) :
    (routed.val.take (max lo (min hi routed.length))).drop lo = win routed lo hi := by
  unfold win
  by_cases h : lo ≤ min hi routed.length
  · rw [max_eq_right h]
  · have h' : min hi routed.length < lo := by omega
    rw [max_eq_left (by omega)]
    rw [List.drop_eq_nil_of_le, List.drop_eq_nil_of_le]
    · simp only [List.length_take]; omega
    · simp only [List.length_take]; omega

theorem count_aggregate_refines_win (routed : Slice U64) (lo hi : Usize)
    (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      routed lo hi = ok m) :
    LineRep m.inner (cntW (win routed lo.val hi.val)) := by
  unfold hashtrinity.freq.FreqMapKI64.count_aggregate at h
  obtain ⟨i, hi', h⟩ := bind_inv h
  obtain ⟨m0, hm0, h⟩ := bind_inv h
  have hrep0 := with_capacity_rep _ _ _ hm0
  have key := loop_pc (fun (st : hashtrinity.freq.FreqMap U64 Std.I64 × Usize) =>
      LineRep st.1.inner (cntW ((routed.val.take st.2.val).drop lo.val)) ∧
      lo.val ≤ st.2.val ∧ st.2.val ≤ max lo.val (min hi.val routed.length))
    (fun (y : hashtrinity.freq.FreqMap U64 Std.I64) =>
      LineRep y.inner (cntW ((routed.val.take (max lo.val (min hi.val routed.length))).drop lo.val)))
    _ ?_ (m0, lo) ?_ m (by
      unfold hashtrinity.freq.FreqMapKI64.count_aggregate_loop at h
      exact h)
  · rw [win_eq] at key; exact key
  · intro st hst cf hcf
    obtain ⟨m', k⟩ := st
    dsimp only at hcf
    cases cf with
    | done y => exact count_body_spec routed lo hi _ rfl (m', k) hst (.done y) hcf
    | cont x => exact count_body_spec routed lo hi _ rfl (m', k) hst (.cont x) hcf
  · refine ⟨?_, le_refl _, show lo.val ≤ max lo.val (min hi.val routed.length) by omega⟩
    have : cntW ((routed.val.take lo.val).drop lo.val) = fun _ => 0 := by
      funext f
      unfold cntW
      rw [List.drop_eq_nil_of_le (by simp)]
      simp
    rw [this]; exact hrep0




open Classical in
/-- The count the aggregate returns: the number of positions `j ∈ [lo, min hi len)` whose key has
fold `f`. -/
noncomputable def cntI (routed : Slice U64) (lo hi : Nat) (f : Nat) : Nat :=
  ((Finset.Ico lo (min hi routed.length)).filter
    (fun j => ∃ x, routed.val[j]? = some x ∧ foldN x.val = f)).card

theorem cntW_eq_card (l : List U64) (lo : Nat) (f : Nat) :
    ∀ n, lo ≤ n → n ≤ l.length →
      cntW ((l.take n).drop lo) f
        = ((Finset.Ico lo n).filter (fun j => ∃ x, l[j]? = some x ∧ foldN x.val = f)).card := by
  classical
  intro n hlo
  induction n, hlo using Nat.le_induction with
  | base =>
    intro _
    unfold cntW
    rw [List.drop_eq_nil_of_le (by simp)]
    simp
  | succ n hlo ih =>
    intro hn
    have hn' : n < l.length := by omega
    have hts : l[n]? = some l[n] := List.getElem?_eq_getElem hn'
    have hwin : (l.take (n + 1)).drop lo = (l.take n).drop lo ++ [l[n]] := by
      rw [List.take_add_one, hts]
      simp only [Option.toList_some]
      rw [List.drop_append_of_le_length]
      rw [List.length_take]; omega
    rw [hwin, cntW_snoc, ih (by omega), Nat.Ico_succ_right_eq_insert_Ico hlo, Finset.filter_insert]
    by_cases hx : foldN l[n].val = f
    · have hp : ∃ x, l[n]? = some x ∧ foldN x.val = f := ⟨l[n], hts, hx⟩
      rw [if_pos hx, if_pos hp, Finset.card_insert_of_notMem]
      simp
    · have hp : ¬ ∃ x, l[n]? = some x ∧ foldN x.val = f := by
        rintro ⟨x, hx1, hx2⟩
        rw [hts] at hx1
        have := Option.some_injective _ hx1
        subst this
        exact hx hx2
      rw [if_neg hx, if_neg hp]

theorem cntW_win (routed : Slice U64) (lo hi : Nat) :
    cntW (win routed lo hi) = cntI routed lo hi := by
  classical
  funext f
  have e1 : (↑routed : List U64).length = routed.length := rfl
  unfold win cntI
  by_cases h : lo ≤ min hi routed.length
  · rw [cntW_eq_card routed.val lo f _ h (by omega)]
  · have h' : min hi routed.length < lo := by omega
    unfold cntW
    rw [List.drop_eq_nil_of_le (by simp only [List.length_take]; omega)]
    simp [Finset.Ico_eq_empty_of_le (show min hi routed.length ≤ lo by omega)]

/-- **`count_aggregate` refines the count function.**  If `count_aggregate routed lo hi` returns
`ok m`, then `m` represents the function sending a fold `f` to the number of positions
`j ∈ [lo, min hi len)` whose key has fold `f`. -/
theorem count_aggregate_refines (routed : Slice U64) (lo hi : Usize)
    (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      routed lo hi = ok m) :
    FreqRep m (cntI routed lo.val hi.val) := by
  have := count_aggregate_refines_win routed lo hi m h
  rw [cntW_win] at this
  exact this

/-- For a key below `2^64 - 3` the fold is the identity (`freq(k) = min(k, EMPTY - 2)`), so the count
at the fold is the exact count of that key. -/
theorem foldN_eq_iff (x k : Nat) (hk : k < 2^64 - 3) : foldN x = k ↔ x = k := by
  unfold foldN
  omega

theorem cntI_key (routed : Slice U64) (lo hi : Nat) (k : U64) (hk : k.val < 2^64 - 3) :
    cntI routed lo hi k.val
      = ((Finset.Ico lo (min hi routed.length)).filter (fun j => routed.val[j]? = some k)).card := by
  classical
  unfold cntI
  congr 1
  apply Finset.filter_congr
  intro j _
  constructor
  · rintro ⟨x, hx1, hx2⟩
    rw [foldN_eq_iff _ _ hk] at hx2
    rw [hx1]
    have : x = k := UScalar.eq_of_val_eq hx2
    rw [this]
  · intro hj
    exact ⟨k, hj, by rw [foldN_eq_iff _ _ hk]⟩

/-- The canonical object: `Apeiron.Table.build (· + ·) 0` over the stream of `1`s at each key's fold,
read with `Apeiron.Table.get`, is the count `cntW` of the window. -/
theorem build_count (w : List U64) (f : Nat) :
    Apeiron.Table.get (Apeiron.Table.build (fun (a b : Int) => a + b) (0 : Int)
      (w.map (fun x => (foldN x.val, (1 : Int))))) f = (cntW w f : Int) := by
  classical
  have key : ∀ (w : List U64) (m0 : Apeiron.Table.HMap Nat Int),
      Apeiron.Table.get ((w.map (fun x => (foldN x.val, (1 : Int)))).foldl
        (fun m kv => Apeiron.Table.insertWith (fun (a b : Int) => a + b) kv.1 kv.2 m) m0) f
        = Apeiron.Table.get m0 f + (cntW w f : Int) := by
    intro w
    induction w with
    | nil => intro m0; simp [cntW]
    | cons x rest ih =>
      intro m0
      rw [List.map_cons, List.foldl_cons, ih]
      have hc : cntW (x :: rest) f = cntW rest f + (if foldN x.val = f then 1 else 0) := by
        unfold cntW
        by_cases hx : foldN x.val = f
        · simp [hx]
        · simp [hx]
      rw [hc]
      by_cases hx : foldN x.val = f
      · subst hx
        dsimp only
        rw [Apeiron.Table.get_insertWith_self]
        simp only [Apeiron.Table.get, if_true]
        push_cast
        omega
      · dsimp only
        rw [Apeiron.Table.get_insertWith_ne _ _ _ _ (fun hh => hx hh.symm)]
        simp [hx]
  have := key w (fun _ => (0 : Int))
  unfold Apeiron.Table.build
  simpa [Apeiron.Table.get] using this

/-- **`count_aggregate` against the canonical `build`/`get`.**  Reading the canonical table built from
the stream of `1`s over the window `routed[lo..min(hi, len))` (keyed by fold) at a fold `f` is the
count `c f` the returned map represents. -/
theorem count_aggregate_canonical (routed : Slice U64) (lo hi : Usize)
    (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      routed lo hi = ok m) :
    ∃ c, FreqRep m c ∧ ∀ f,
      Apeiron.Table.get (Apeiron.Table.build (fun (a b : Int) => a + b) (0 : Int)
        ((win routed lo.val hi.val).map (fun x => (foldN x.val, (1 : Int))))) f = (c f : Int) :=
  ⟨cntW (win routed lo.val hi.val), count_aggregate_refines_win routed lo hi m h,
    fun f => build_count _ f⟩

end Apeiron.CountAggregate
