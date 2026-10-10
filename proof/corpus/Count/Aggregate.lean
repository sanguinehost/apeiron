-- The count table's representation and `count_aggregate`'s refinement.
import ApeironWord
import Count.Helpers
import Count.FreqAdd
import Count.FreqGrow
import Count.FreqInsert
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
* Preserves exact key counts for every `u64` key: `cntI_key`.
* Refines the canonical `Apeiron.Table.build` and `Apeiron.Table.get`: `build_count` and `count_aggregate_canonical`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive
open Apeiron.MapExecuted (nilN)

/-! ## `count_aggregate` -/

/-- The counts of a window of keys: `#{x ∈ w | x = f}`.  `FreqMap::freq` is the key itself (no
clamp), so every `u64` key, `EMPTY` and `TOMB` included, counts exactly. -/
def cntW (w : List U64) (f : Nat) : Nat := (w.filter (fun x => x.val = f)).length

/-- The window `routed[lo..min(hi, len))`. -/
def win (routed : Slice U64) (lo hi : Nat) : List U64 :=
  (routed.val.take (min hi routed.length)).drop lo

theorem cntW_snoc (w : List U64) (x : U64) (f : Nat) :
    cntW (w ++ [x]) f = if x.val = f then cntW w f + 1 else cntW w f := by
  unfold cntW
  rw [List.filter_append]
  by_cases h : x.val = f
  · simp [h]
  · simp [h]

theorem count_body_spec (routed : Slice U64) (lo hi : Usize) (M : Nat)
    (hM : M = max lo.val (min hi.val routed.length)) (hN : M - lo.val ≤ nilN)
    (st : hashtrinity.freq.FreqMap U64 Std.I64 × Usize)
    (hI : LineRep st.1.inner (cntW ((routed.val.take st.2.val).drop lo.val)) ∧
      lo.val ≤ st.2.val ∧ st.2.val ≤ M ∧ Pot st.1.inner ≤ st.2.val - lo.val)
    (cf : ControlFlow (hashtrinity.freq.FreqMap U64 Std.I64 × Usize) (hashtrinity.freq.FreqMap U64 Std.I64))
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate_loop.body
      U64.Insts.Apeiron_wordHashtrinityAlgebraCoord routed hi st.1 st.2 = ok cf) :
    match cf with
    | .done m' => LineRep m'.inner (cntW ((routed.val.take M).drop lo.val))
    | .cont st' => LineRep st'.1.inner (cntW ((routed.val.take st'.2.val).drop lo.val)) ∧
        lo.val ≤ st'.2.val ∧ st'.2.val ≤ M ∧ Pot st'.1.inner ≤ st'.2.val - lo.val := by
  obtain ⟨m, i⟩ := st
  obtain ⟨hR, hlo, hiM, hpot⟩ := hI
  dsimp only at hR hlo hiM hpot h
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
      have hPm : Pot m.inner < nilN := by omega
      obtain ⟨hadd, hpot1⟩ := add_spec m m1 _ hR hPm t hm1
      have hwin : (routed.val.take (i.val + 1)).drop lo.val
          = (routed.val.take i.val).drop lo.val ++ [t] := by
        rw [List.take_add_one, hts]
        simp only [Option.toList_some]
        rw [List.drop_append_of_le_length]
        rw [List.length_take]; omega
      have hfun : (fun x => if x = t.val then cntW ((routed.val.take i.val).drop lo.val) x + 1
          else cntW ((routed.val.take i.val).drop lo.val) x)
          = cntW ((routed.val.take (i.val + 1)).drop lo.val) := by
        funext x
        rw [hwin, cntW_snoc]
        by_cases hx : x = t.val
        · simp [hx]
        · have : ¬ t.val = x := fun hh => hx hh.symm
          simp [hx, this]
      rw [hfun] at hadd
      refine ⟨?_, show lo.val ≤ i2.val by omega, show i2.val ≤ M by omega,
        show Pot m1.inner ≤ i2.val - lo.val by omega⟩
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
    (hN : min hi.val routed.length - lo.val ≤ nilN)
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      routed lo hi = ok m) :
    LineRep m.inner (cntW (win routed lo.val hi.val)) := by
  unfold hashtrinity.freq.FreqMapKI64.count_aggregate at h
  obtain ⟨i, hi', h⟩ := bind_inv h
  obtain ⟨m0, hm0, h⟩ := bind_inv h
  have hrep0 := with_capacity_rep _ _ _ hm0
  have hpot0 := pot_with_capacity _ _ hm0
  have hN' : max lo.val (min hi.val routed.length) - lo.val ≤ nilN := by omega
  have key := loop_pc (fun (st : hashtrinity.freq.FreqMap U64 Std.I64 × Usize) =>
      LineRep st.1.inner (cntW ((routed.val.take st.2.val).drop lo.val)) ∧
      lo.val ≤ st.2.val ∧ st.2.val ≤ max lo.val (min hi.val routed.length) ∧
      Pot st.1.inner ≤ st.2.val - lo.val)
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
    | done y => exact count_body_spec routed lo hi _ rfl hN' (m', k) hst (.done y) hcf
    | cont x => exact count_body_spec routed lo hi _ rfl hN' (m', k) hst (.cont x) hcf
  · refine ⟨?_, le_refl _, show lo.val ≤ max lo.val (min hi.val routed.length) by omega,
      show Pot m0.inner ≤ lo.val - lo.val by rw [hpot0]; omega⟩
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
    (fun j => ∃ x, routed.val[j]? = some x ∧ x.val = f)).card

theorem cntW_eq_card (l : List U64) (lo : Nat) (f : Nat) :
    ∀ n, lo ≤ n → n ≤ l.length →
      cntW ((l.take n).drop lo) f
        = ((Finset.Ico lo n).filter (fun j => ∃ x, l[j]? = some x ∧ x.val = f)).card := by
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
    by_cases hx : l[n].val = f
    · have hp : ∃ x, l[n]? = some x ∧ x.val = f := ⟨l[n], hts, hx⟩
      rw [if_pos hx, if_pos hp, Finset.card_insert_of_notMem]
      simp
    · have hp : ¬ ∃ x, l[n]? = some x ∧ x.val = f := by
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
`ok m`, then `m` represents the function sending a frequency `f` (a key's own value, for every
`u64` key) to the number of positions `j ∈ [lo, min hi len)` whose key is `f`.  The premise `hN`
(the window holds at most `2^32 - 1` keys) keeps the overflow arena's `u32` index from wrapping
(`Overflow::alloc` casts the arena length to `u32`). -/
theorem count_aggregate_refines (routed : Slice U64) (lo hi : Usize)
    (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (hN : min hi.val routed.length - lo.val ≤ nilN)
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      routed lo hi = ok m) :
    FreqRep m (cntI routed lo.val hi.val) := by
  have := count_aggregate_refines_win routed lo hi m hN h
  rw [cntW_win] at this
  exact this

/-- The count at a key's frequency is the exact count of that key, for every `u64` key. -/
theorem cntI_key (routed : Slice U64) (lo hi : Nat) (k : U64) :
    cntI routed lo hi k.val
      = ((Finset.Ico lo (min hi routed.length)).filter (fun j => routed.val[j]? = some k)).card := by
  classical
  unfold cntI
  congr 1
  apply Finset.filter_congr
  intro j _
  constructor
  · rintro ⟨x, hx1, hx2⟩
    rw [hx1]
    have : x = k := UScalar.eq_of_val_eq hx2
    rw [this]
  · intro hj
    exact ⟨k, hj, rfl⟩

/-- The canonical object: `Apeiron.Table.build (· + ·) 0` over the stream of `1`s at each key,
read with `Apeiron.Table.get`, is the count `cntW` of the window. -/
theorem build_count (w : List U64) (f : Nat) :
    Apeiron.Table.get (Apeiron.Table.build (fun (a b : Int) => a + b) (0 : Int)
      (w.map (fun x => (x.val, (1 : Int))))) f = (cntW w f : Int) := by
  classical
  have key : ∀ (w : List U64) (m0 : Apeiron.Table.HMap Nat Int),
      Apeiron.Table.get ((w.map (fun x => (x.val, (1 : Int)))).foldl
        (fun m kv => Apeiron.Table.insertWith (fun (a b : Int) => a + b) kv.1 kv.2 m) m0) f
        = Apeiron.Table.get m0 f + (cntW w f : Int) := by
    intro w
    induction w with
    | nil => intro m0; simp [cntW]
    | cons x rest ih =>
      intro m0
      rw [List.map_cons, List.foldl_cons, ih]
      have hc : cntW (x :: rest) f = cntW rest f + (if x.val = f then 1 else 0) := by
        unfold cntW
        by_cases hx : x.val = f
        · simp [hx]
        · simp [hx]
      rw [hc]
      by_cases hx : x.val = f
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
the stream of `1`s over the window `routed[lo..min(hi, len))` (keyed by the key's own value) at a
frequency `f` is the count `c f` the returned map represents. -/
theorem count_aggregate_canonical (routed : Slice U64) (lo hi : Usize)
    (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (hN : min hi.val routed.length - lo.val ≤ nilN)
    (h : hashtrinity.freq.FreqMapKI64.count_aggregate U64.Insts.Apeiron_wordHashtrinityAlgebraCoord
      routed lo hi = ok m) :
    ∃ c, FreqRep m c ∧ ∀ f,
      Apeiron.Table.get (Apeiron.Table.build (fun (a b : Int) => a + b) (0 : Int)
        ((win routed lo.val hi.val).map (fun x => (x.val, (1 : Int))))) f = (c f : Int) :=
  ⟨cntW (win routed lo.val hi.val), count_aggregate_refines_win routed lo hi m hN h,
    fun f => build_count _ f⟩

end Apeiron.CountAggregate
