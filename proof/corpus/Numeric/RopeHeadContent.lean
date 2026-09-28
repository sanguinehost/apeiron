/-
Content of the extracted `rope.RopeTable.rotate_head_loop` (`APEIRON-ALGEBRA-G3-ROPE-W1-T1`).

`Numeric.EngineRopeRefinement` proves the per-pair bound `rope_pair_spec_proved` and loop totality
`rotate_head_loop_spec` on a bounds-only invariant. This module carries the written values through
the loop on the success domain, where every rotary index `j < d2` has its trig pair `scAt j` and
both rotated components fit `i64`, so the loop flag stays `true`.

* `ropeHeadPair`, `rotPrefix` — the closed-form reference: after `j` iterations entry `i` is the
  first `ropePairInt` component of pair `i` if `i < j`, the second component of pair `i - d2` if
  `d2 ≤ i < d2 + j`, and the original entry otherwise;
* `rotPrefix_zero`, `rotPrefix_succ` — the pure list recurrence: one iteration is two `List.set`s;
* `rotate_head_loop_content` — the extracted loop from `(x0, 0, true)` returns `(x', true)` with
  `x'` denoting `rotPrefix x0 d2 scAt d2`;
* `rotate_head_content` — the extracted `rotate_head` returns `Ok ()` with that head;
* `rotate_head_bound` — per rotary index, the two written entries satisfy `rope_pair_spec_proved`'s
  bound against `ropeRot` of the original pair.

Scope: integer content of one head on the success domain. The refusal arm, the outer `rotate` loop
and a closed-form witness of the success domain at a concrete `fixed_sin_cos` value are not covered.
-/
import Numeric.EngineRopeRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Closed-form reference -/

/-- The integer rotation of pair `k` of the original head `x0` by the trig pair `scAt k`. -/
def ropeHeadPair (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (k : ℕ) : ℤ × ℤ :=
  ropePairInt (x0.getD k 0#i64).val (x0.getD (k + d2) 0#i64).val (scAt k).cos.val (scAt k).sin.val

/-- The head after `j` loop iterations, as integers. -/
def rotPrefix (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j : ℕ) : List ℤ :=
  x0.mapIdx fun i a =>
    if i < j then (ropeHeadPair x0 d2 scAt i).1
    else if d2 ≤ i ∧ i < d2 + j then (ropeHeadPair x0 d2 scAt (i - d2)).2
    else a.val

theorem rotPrefix_length (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j : ℕ) :
    (rotPrefix x0 d2 scAt j).length = x0.length := by
  simp [rotPrefix]

theorem rotPrefix_zero (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) :
    rotPrefix x0 d2 scAt 0 = x0.map (fun a => a.val) := by
  apply List.ext_getElem
  · simp [rotPrefix]
  · intro i h1 _
    simp only [rotPrefix, List.getElem_mapIdx, List.getElem_map]
    rw [if_neg (by omega), if_neg (by omega)]

/-- Entries at the pair about to be processed are still the original ones. -/
theorem rotPrefix_getElem_orig (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j i : ℕ)
    (hj : j < d2) (hi : i = j ∨ i = j + d2) (h : i < (rotPrefix x0 d2 scAt j).length) :
    (rotPrefix x0 d2 scAt j)[i] = (x0.getD i 0#i64).val := by
  have hlt : i < x0.length := by rw [rotPrefix_length] at h; exact h
  simp only [rotPrefix, List.getElem_mapIdx]
  rw [if_neg (by omega), if_neg (by omega)]
  simp [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hlt]

/-- A slice denoting `rotPrefix x0 d2 scAt j` still holds the original pair `j` at `j`, `j + d2`. -/
theorem rotPrefix_read (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j i : ℕ)
    (xs : List I64) (hmap : xs.map (fun a => a.val) = rotPrefix x0 d2 scAt j)
    (hj : j < d2) (hi : i = j ∨ i = j + d2) (h : i < xs.length) :
    (xs[i]).val = (x0.getD i 0#i64).val := by
  have e := List.getElem_of_eq hmap (i := i) (by simpa using h)
  rw [List.getElem_map] at e
  rw [e]
  exact rotPrefix_getElem_orig x0 d2 scAt j i hj hi _

/-- **One iteration.** For `j < d2`, iteration `j` writes the two components of
pair `j` at `j` and `j + d2`. -/
theorem rotPrefix_succ (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j : ℕ)
    (hj : j < d2) :
    rotPrefix x0 d2 scAt (j + 1) =
      ((rotPrefix x0 d2 scAt j).set j (ropeHeadPair x0 d2 scAt j).1).set (j + d2)
        (ropeHeadPair x0 d2 scAt j).2 := by
  apply List.ext_getElem
  · simp [rotPrefix]
  · intro i h1 _
    simp only [rotPrefix, List.getElem_mapIdx, List.getElem_set]
    by_cases hA : j + d2 = i
    · subst hA
      rw [if_pos rfl, if_neg (by omega), if_pos (by omega)]
      congr 2
      omega
    · rw [if_neg hA]
      by_cases hB : j = i
      · subst hB
        rw [if_pos rfl, if_pos (by omega)]
      · rw [if_neg hB]
        by_cases hC : i < j
        · rw [if_pos (by omega), if_pos hC]
        · rw [if_neg (by omega), if_neg hC]
          by_cases hD : d2 ≤ i ∧ i < d2 + j
          · rw [if_pos (by omega), if_pos hD]
          · rw [if_neg (by omega), if_neg hD]

/-! ## Loop invariant and body step -/

/-- Success-domain invariant at loop state `(x, j, ok)`: the half-window fits, `j ≤ d2`, the flag is
`true`, and `x` denotes `rotPrefix x0 d2 scAt j`. -/
def RotHeadContentInv (x0 : Slice I64) (d2 : Usize) (scAt : ℕ → fixed_trig.SinCos)
    (st : Slice I64 × Usize × Bool) : Prop :=
  2 * d2.val ≤ st.1.length ∧ st.2.1.val ≤ d2.val ∧ st.2.2 = true ∧
    st.1.val.map (fun a => a.val) = rotPrefix x0.val d2.val scAt st.2.1.val

/-- Both components of every rotated pair fit `i64`. -/
def RotHeadFits (x0 : Slice I64) (d2 : Usize) (scAt : ℕ → fixed_trig.SinCos) : Prop :=
  ∀ j, j < d2.val →
    I64.min ≤ (ropeHeadPair x0.val d2.val scAt j).1 ∧ (ropeHeadPair x0.val d2.val scAt j).1 ≤ I64.max ∧
    I64.min ≤ (ropeHeadPair x0.val d2.val scAt j).2 ∧ (ropeHeadPair x0.val d2.val scAt j).2 ≤ I64.max

/-- A widened `I64` entry times a `2^33`-bounded `I64` trig component has magnitude at most `2^96`,
inside `I128`. (`EngineRopeRefinement`'s `i64_prod_max`/`i64_prod_min`/`i64_prod_abs` are private
there, so the combined bound is proved here.) -/
private theorem rh_prod (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    |((core.convert.num.FromI128I64.from a).val : ℤ) *
      ((core.convert.num.FromI128I64.from b).val : ℤ)| ≤ 2 ^ 96 := by
  rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq, abs_mul]
  have ha : |(a.val : ℤ)| ≤ 2 ^ 63 := by
    have h1 : -(2 : ℤ) ^ 63 ≤ a.val := by simpa [I64.numBits] using a.hmin
    have h2 : a.val < (2 : ℤ) ^ 63 := by simpa [I64.numBits] using a.hmax
    rw [abs_le]
    exact ⟨h1, by omega⟩
  calc |(a.val : ℤ)| * |(b.val : ℤ)| ≤ (2 : ℤ) ^ 63 * 8589934592 :=
        mul_le_mul ha hb (by positivity) (by positivity)
    _ = 2 ^ 96 := by norm_num

private theorem rh_prod_max (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    ((core.convert.num.FromI128I64.from a).val : ℤ) *
      ((core.convert.num.FromI128I64.from b).val : ℤ) ≤ I128.max := by
  have h := (abs_le.mp (rh_prod a b hb)).2
  have hmax : (2 : ℤ) ^ 96 ≤ I128.max := by norm_num [I128.max, I128.numBits]
  omega

private theorem rh_prod_min (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    I128.min ≤ ((core.convert.num.FromI128I64.from a).val : ℤ) *
      ((core.convert.num.FromI128I64.from b).val : ℤ) := by
  have h := (abs_le.mp (rh_prod a b hb)).1
  have hmin : I128.min ≤ -((2 : ℤ) ^ 96) := by norm_num [I128.min, I128.numBits]
  omega

/-- An in-range `IScalar.cast` preserves the value (from Aeneas's `IScalar.cast_inBounds_spec`). -/
private theorem rh_cast_val {src : IScalarTy} (tgt : IScalarTy) (x : IScalar src)
    (h : IScalar.min tgt ≤ x.val ∧ x.val ≤ IScalar.max tgt) :
    (IScalar.cast tgt x).val = x.val := by
  have hs := IScalar.cast_inBounds_spec tgt x h
  simpa [lift] using hs

private theorem rh_cast128 (y : I64) : (IScalar.cast IScalarTy.I128 y).val = y.val := by
  apply rh_cast_val
  have h1 := y.hmin
  have h2 := y.hmax
  simp only [IScalar.min, IScalar.max, IScalarTy.numBits] at *
  constructor <;> omega

private theorem rh_max : core.num.I64.MAX.val = I64.max := by
  simp [core.num.I64.MAX, IScalar.ofInt, I64.rMax, I64.max, I64.numBits]

private theorem rh_min : core.num.I64.MIN.val = I64.min := by
  simp [core.num.I64.MIN, IScalar.ofInt, I64.rMin, I64.min, I64.numBits]

private theorem rh_i64_range (z : ℤ) (h1 : I64.min ≤ z) (h2 : z ≤ I64.max) :
    IScalar.min IScalarTy.I64 ≤ z ∧ z ≤ IScalar.max IScalarTy.I64 := by
  simp only [I64.min, I64.max, IScalar.min, IScalar.max, I64.numBits, IScalarTy.numBits] at *
  exact ⟨h1, h2⟩

private theorem rot_head_content_body (v : alloc.vec.Vec U64) (pos : U32) (d2 : Usize)
    (x_orig : Slice I64) (scAt : ℕ → fixed_trig.SinCos)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt v pos i t (scAt i))
    (hfit : RotHeadFits x_orig d2 scAt)
    (st : Slice I64 × Usize × Bool) (hst : RotHeadContentInv x_orig d2 scAt st) :
    (rope.RopeTable.rotate_head_loop.body v pos d2 st.1 st.2.1 st.2.2) ⦃ r =>
      match r with
      | .done y => y.2 = true ∧ y.1.val.map (fun a => a.val) = rotPrefix x_orig.val d2.val scAt d2.val
      | .cont st' => RotHeadContentInv x_orig d2 scAt st' ∧
          d2.val - st'.2.1.val < d2.val - st.2.1.val ⦄ := by
  rcases st with ⟨xq, jq, oq⟩
  obtain ⟨h1, h3, hok, hmap⟩ := hst
  dsimp only at h1 h3 hok hmap
  subst hok
  unfold rope.RopeTable.rotate_head_loop.body
  dsimp only
  by_cases hlt : jq < d2
  · rw [if_pos hlt, if_pos rfl]
    have hjlt : (jq : ℕ) < (d2 : ℕ) := Iff.mp (UScalar.lt_equiv jq d2) hlt
    obtain ⟨tq, htq, aq, haq, hscq, hcos, hsin⟩ := htrig jq.val hjlt
    have hvlen : jq.val < v.length := by
      obtain ⟨hb, _⟩ := List.getElem?_eq_some_iff.mp htq
      exact hb
    simp only [lift, bind_tc_ok]
    simp only [alloc.vec.Vec.index_slice_index]
    refine Aeneas.Std.WP.spec_bind (alloc.vec.Vec.index_usize_spec v jq hvlen) ?_
    intro i1 hi1
    have hi1t : i1 = tq := by
      obtain ⟨_, hget⟩ := List.getElem?_eq_some_iff.mp htq
      rw [hi1]
      exact hget
    subst i1
    step*
    have hi3_val : i3.val = tq.val * pos.val := by
      rw [i3_post, u128_cast_u64_val, u128_cast_u32_val, hi1t]
    have hphase_val : (UScalar.cast UScalarTy.U64 (i3 &&& UScalar.cast UScalarTy.U128 core.num.U64.MAX)).val =
        Numeric.RopeTurns.phase tq.val pos.val := turn_phase_val_eq tq pos i3 hi3_val
    have hi6_val : i6.val = aq.val := by
      rw [i6_post, hphase_val, haq]
    have hi6_eq : i6 = aq := IScalar.eq_of_val_eq hi6_val
    rw [hi6_eq, hscq]
    simp only [bind_tc_ok]
    step*
    all_goals (first
      | exact rh_prod_max _ _ hcos
      | exact rh_prod_max _ _ hsin
      | exact rh_prod_min _ _ hcos
      | exact rh_prod_min _ _ hsin
      | skip)
    · rename_i hsub
      exfalso
      have hp00abs : |p00.val| ≤ (2 : ℤ) ^ 96 := by rw [p00_post]; exact rh_prod _ _ hcos
      have hp01abs : |p01.val| ≤ (2 : ℤ) ^ 96 := by rw [p01_post]; exact rh_prod _ _ hsin
      have h := I128.checked_sub_bv_spec p00 p01
      have hc' : core.num.checked_sub_IScalar p00 p01 = none := hsub
      rw [hc'] at h
      apply h
      have ha00 := abs_le.mp hp00abs
      have ha01 := abs_le.mp hp01abs
      have hI128 : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max ∧ I128.min ≤ -((2 : ℤ) ^ 97) := by
        norm_num [I128.max, I128.min, I128.numBits]
      constructor <;> omega
    · rename_i hadd
      exfalso
      have hp10abs : |p10.val| ≤ (2 : ℤ) ^ 96 := by rw [p10_post]; exact rh_prod _ _ hcos
      have hp11abs : |p11.val| ≤ (2 : ℤ) ^ 96 := by rw [p11_post]; exact rh_prod _ _ hsin
      have h := I128.checked_add_bv_spec p10 p11
      have hc' : core.num.checked_add_IScalar p10 p11 = none := hadd
      rw [hc'] at h
      apply h
      have ha10 := abs_le.mp hp10abs
      have ha11 := abs_le.mp hp11abs
      have hI128 : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max ∧ I128.min ≤ -((2 : ℤ) ^ 97) := by
        norm_num [I128.max, I128.min, I128.numBits]
      constructor <;> omega
    · rename_i hsub hadd
      have hlenxq : jq.val < xq.val.length := by simp only [Slice.length] at h1; omega
      have hlenma : i7.val < xq.val.length := by
        simp only [Slice.length] at h1; rw [i7_post]; omega
      have hxv0 : x0.val = (x_orig.val.getD jq.val 0#i64).val := by
        rw [x0_post]
        exact rotPrefix_read x_orig.val d2.val scAt jq.val jq.val xq.val hmap hjlt (Or.inl rfl) hlenxq
      have hyv0 : x1.val = (x_orig.val.getD (jq.val + d2.val) 0#i64).val := by
        rw [x1_post]
        have h := rotPrefix_read x_orig.val d2.val scAt jq.val i7.val xq.val hmap hjlt (Or.inr i7_post) hlenma
        rw [h, i7_post]
      have hp00abs : |p00.val| ≤ (2 : ℤ) ^ 96 := by rw [p00_post]; exact rh_prod _ _ hcos
      have hp01abs : |p01.val| ≤ (2 : ℤ) ^ 96 := by rw [p01_post]; exact rh_prod _ _ hsin
      have hp10abs : |p10.val| ≤ (2 : ℤ) ^ 96 := by rw [p10_post]; exact rh_prod _ _ hcos
      have hp11abs : |p11.val| ≤ (2 : ℤ) ^ 96 := by rw [p11_post]; exact rh_prod _ _ hsin
      have hp00v : p00.val = x0.val * (scAt jq.val).cos.val := by
        rw [p00_post, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
      have hp01v : p01.val = x1.val * (scAt jq.val).sin.val := by
        rw [p01_post, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
      have hp10v : p10.val = x1.val * (scAt jq.val).cos.val := by
        rw [p10_post, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
      have hp11v : p11.val = x0.val * (scAt jq.val).sin.val := by
        rw [p11_post, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
      have hre : re.val = p00.val - p01.val := by
        have h := I128.checked_sub_bv_spec p00 p01
        have hc' : core.num.checked_sub_IScalar p00 p01 = some re := hsub
        rw [hc'] at h
        exact h.2.2.1
      have him : im.val = p10.val + p11.val := by
        have h := I128.checked_add_bv_spec p10 p11
        have hc' : core.num.checked_add_IScalar p10 p11 = some im := hadd
        rw [hc'] at h
        exact h.2.2.1
      have hreabs : |re.val| ≤ (2 : ℤ) ^ 97 := by
        rw [hre]
        calc |p00.val - p01.val| ≤ |p00.val| + |p01.val| := by
              simpa [sub_eq_add_neg] using abs_add_le p00.val (-p01.val)
          _ ≤ 2 ^ 96 + 2 ^ 96 := add_le_add hp00abs hp01abs
          _ = 2 ^ 97 := by norm_num
      have himabs : |im.val| ≤ (2 : ℤ) ^ 97 := by
        rw [him]
        calc |p10.val + p11.val| ≤ |p10.val| + |p11.val| := abs_add_le _ _
          _ ≤ 2 ^ 96 + 2 ^ 96 := add_le_add hp10abs hp11abs
          _ = 2 ^ 97 := by norm_num
      have hmin_re : I128.min ≤ re.val + 2147483648 := by
        have hmin128 : I128.min ≤ -((2 : ℤ) ^ 97) + 2147483648 := by norm_num [I128.min, I128.numBits]
        have := (abs_le.mp hreabs).1
        linarith
      have hmax_re : re.val + 2147483648 ≤ I128.max := by
        have hmax128 : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max := by norm_num [I128.max, I128.numBits]
        have := (abs_le.mp hreabs).2
        linarith
      have hmin_im : I128.min ≤ im.val + 2147483648 := by
        have hmin128 : I128.min ≤ -((2 : ℤ) ^ 97) + 2147483648 := by norm_num [I128.min, I128.numBits]
        have := (abs_le.mp himabs).1
        linarith
      have hmax_im : im.val + 2147483648 ≤ I128.max := by
        have hmax128 : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max := by norm_num [I128.max, I128.numBits]
        have := (abs_le.mp himabs).2
        linarith
      refine Aeneas.Std.WP.spec_bind (value_round_shift_32_spec re hmin_re hmax_re) ?_
      intro out0 hout0r
      refine Aeneas.Std.WP.spec_bind (value_round_shift_32_spec im hmin_im hmax_im) ?_
      intro out1 hout1r
      have hout0 : out0.val = (ropeHeadPair x_orig.val d2.val scAt jq.val).1 := by
        rw [hout0r, hre, hp00v, hp01v, hxv0, hyv0]
        simp only [ropeHeadPair, ropePairInt]
      have hout1 : out1.val = (ropeHeadPair x_orig.val d2.val scAt jq.val).2 := by
        rw [hout1r, him, hp10v, hp11v, hxv0, hyv0]
        simp only [ropeHeadPair, ropePairInt]
      obtain ⟨f1, f2, f3, f4⟩ := hfit jq.val hjlt
      have hc1 : ¬ (out0 > IScalar.cast IScalarTy.I128 core.num.I64.MAX) := by
        rw [gt_iff_lt, IScalar.lt_equiv, rh_cast128, rh_max]; omega
      have hc2 : ¬ (out0 < IScalar.cast IScalarTy.I128 core.num.I64.MIN) := by
        rw [IScalar.lt_equiv, rh_cast128, rh_min]; omega
      have hc3 : ¬ (out1 > IScalar.cast IScalarTy.I128 core.num.I64.MAX) := by
        rw [gt_iff_lt, IScalar.lt_equiv, rh_cast128, rh_max]; omega
      have hc4 : ¬ (out1 < IScalar.cast IScalarTy.I128 core.num.I64.MIN) := by
        rw [IScalar.lt_equiv, rh_cast128, rh_min]; omega
      have hw0 : (IScalar.cast IScalarTy.I64 out0).val =
          (ropeHeadPair x_orig.val d2.val scAt jq.val).1 := by
        rw [rh_cast_val _ _ (by rw [hout0]; exact rh_i64_range _ f1 f2)]
        exact hout0
      have hw1 : (IScalar.cast IScalarTy.I64 out1).val =
          (ropeHeadPair x_orig.val d2.val scAt jq.val).2 := by
        rw [rh_cast_val _ _ (by rw [hout1]; exact rh_i64_range _ f3 f4)]
        exact hout1
      rw [if_neg hc1, if_neg hc2, if_neg hc3, if_neg hc4]
      step*
      subst_vars
      refine ⟨⟨?_, ?_, rfl, ?_⟩, ?_⟩ <;> try dsimp only
      · simp only [Slice.length, Slice.set_val_eq, List.length_set]
        simpa [Slice.length] using h1
      · omega
      · simp only [Slice.set_val_eq, List.map_set]
        rw [hmap, hw0, hw1, j1_post, rotPrefix_succ _ _ _ _ hjlt, i7_post]
      · omega
  · rw [if_neg hlt]
    simp only [Aeneas.Std.WP.spec_ok]
    have hjd : jq.val = d2.val := by
      have : ¬ (jq.val < d2.val) := fun h => hlt ((UScalar.lt_equiv jq d2).mpr h)
      omega
    rw [hjd] at hmap
    exact ⟨by trivial, hmap⟩

/-! ## Loop and head -/

/-- **Loop content.** On the success domain, the extracted `rotate_head_loop` from `(x0, 0, true)`
returns the flag `true` and a slice denoting `rotPrefix x0 d2 scAt d2`. -/
theorem rotate_head_loop_content (v : alloc.vec.Vec U64) (pos : U32) (d2 : Usize)
    (x0 : Slice I64) (scAt : ℕ → fixed_trig.SinCos)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt v pos i t (scAt i))
    (hfit : RotHeadFits x0 d2 scAt) (hx : 2 * d2.val ≤ x0.length) :
    ∃ x' : Slice I64, rope.RopeTable.rotate_head_loop v x0 pos d2 0#usize true = ok (x', true) ∧
      x'.val.map (fun a => a.val) = rotPrefix x0.val d2.val scAt d2.val := by
  have hspec : loop
      (fun (st : Slice I64 × Usize × Bool) =>
        rope.RopeTable.rotate_head_loop.body v pos d2 st.1 st.2.1 st.2.2)
      (x0, 0#usize, true) ⦃ r => r.2 = true ∧
        r.1.val.map (fun a => a.val) = rotPrefix x0.val d2.val scAt d2.val ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Bool => d2.val - st.2.1.val)
      (inv := RotHeadContentInv x0 d2 scAt)
      (post := fun r : Slice I64 × Bool => r.2 = true ∧
        r.1.val.map (fun a => a.val) = rotPrefix x0.val d2.val scAt d2.val)
    · intro st hst
      convert rot_head_content_body v pos d2 x0 scAt htrig hfit st hst using 1
      funext r
      cases r with
      | done y => rfl
      | cont x' => rfl
    · refine ⟨hx, by simp, rfl, ?_⟩
      show x0.val.map (fun a => a.val) = rotPrefix x0.val d2.val scAt (0#usize).val
      convert (rotPrefix_zero x0.val d2.val scAt).symm using 2
      simp
  obtain ⟨r, hr, hr2, hrm⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  refine ⟨r.1, ?_, hrm⟩
  unfold rope.RopeTable.rotate_head_loop
  rw [hr]
  rw [← hr2]

/-- **Head content.** On the success domain, the extracted `rotate_head` returns `Ok ()` and a head
denoting `rotPrefix x d2 scAt d2`: entry `i < d2` is the first `ropePairInt` component of pair `i`,
entry `d2 + i` the second. -/
theorem rotate_head_content (self : rope.RopeTable) (x : Slice I64) (pos : U32) (d2 : Usize)
    (hd2 : self.head_dim / 2#usize = ok d2) (scAt : ℕ → fixed_trig.SinCos)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt self.turns pos i t (scAt i))
    (hfit : RotHeadFits x d2 scAt) (hx : 2 * d2.val ≤ x.length) :
    ∃ x' : Slice I64, rope.RopeTable.rotate_head self x pos = ok (core.result.Result.Ok (), x') ∧
      x'.val.map (fun a => a.val) = rotPrefix x.val d2.val scAt d2.val := by
  obtain ⟨x', hl, hm⟩ := rotate_head_loop_content self.turns pos d2 x scAt htrig hfit hx
  exact ⟨x', rotate_head_success_eq self x pos d2 x' hd2 hl, hm⟩

/-! ## Component-wise real bound -/

theorem rotPrefix_full_fst (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j : ℕ)
    (hj : j < d2) (hlen : 2 * d2 ≤ x0.length) :
    (rotPrefix x0 d2 scAt d2)[j]? = some (ropeHeadPair x0 d2 scAt j).1 := by
  rw [List.getElem?_eq_getElem (by rw [rotPrefix_length]; omega)]
  simp only [rotPrefix, List.getElem_mapIdx]
  rw [if_pos hj]

theorem rotPrefix_full_snd (x0 : List I64) (d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (j : ℕ)
    (hj : j < d2) (hlen : 2 * d2 ≤ x0.length) :
    (rotPrefix x0 d2 scAt d2)[j + d2]? = some (ropeHeadPair x0 d2 scAt j).2 := by
  rw [List.getElem?_eq_getElem (by rw [rotPrefix_length]; omega)]
  simp only [rotPrefix, List.getElem_mapIdx]
  rw [if_neg (by omega), if_pos (by omega), Nat.add_sub_cancel]

private theorem rh_entry (xs : List I64) (r : List ℤ) (hm : xs.map (fun a => a.val) = r) (i : ℕ)
    (z : ℤ) (hz : r[i]? = some z) : ∃ y : I64, xs[i]? = some y ∧ y.val = z := by
  rw [← hm, List.getElem?_map] at hz
  obtain ⟨y, hy, hyz⟩ := Option.map_eq_some_iff.mp hz
  exact ⟨y, hy, hyz⟩

/-- **Head bound.** On the success domain the extracted `rotate_head` returns `Ok ()`, and for every
rotary index `j < d2` the written entries at `j` and `j + d2` are within the
`rope_pair_spec_proved` bound of `ropeRot θ` applied to the original pair, `θ = toReal angle_j`. -/
theorem rotate_head_bound (self : rope.RopeTable) (x : Slice I64) (pos : U32) (d2 : Usize)
    (hd2 : self.head_dim / 2#usize = ok d2) (scAt : ℕ → fixed_trig.SinCos)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt self.turns pos i t (scAt i))
    (hfit : RotHeadFits x d2 scAt) (hx : 2 * d2.val ≤ x.length) :
    ∃ x' : Slice I64, rope.RopeTable.rotate_head self x pos = ok (core.result.Result.Ok (), x') ∧
      ∀ j, j < d2.val → ∃ (y0 y1 angle : I64),
        x'.val[j]? = some y0 ∧ x'.val[j + d2.val]? = some y1 ∧
        word_fixed_sin_cos angle = ok (core.result.Result.Ok (scAt j)) ∧
        |(y0.val : ℝ) / q32 -
            (((x.val.getD j 0#i64).val : ℝ) / q32 * Real.cos (toReal angle) -
              ((x.val.getD (j + d2.val) 0#i64).val : ℝ) / q32 * Real.sin (toReal angle))| ≤
          (((|(x.val.getD j 0#i64).val| : ℤ) : ℝ) +
              ((|(x.val.getD (j + d2.val) 0#i64).val| : ℤ) : ℝ)) / q32 *
            (epsExp + trig_floor) + 1 / (2 * q32) ∧
        |(y1.val : ℝ) / q32 -
            (((x.val.getD (j + d2.val) 0#i64).val : ℝ) / q32 * Real.cos (toReal angle) +
              ((x.val.getD j 0#i64).val : ℝ) / q32 * Real.sin (toReal angle))| ≤
          (((|(x.val.getD j 0#i64).val| : ℤ) : ℝ) +
              ((|(x.val.getD (j + d2.val) 0#i64).val| : ℤ) : ℝ)) / q32 *
            (epsExp + trig_floor) + 1 / (2 * q32) := by
  obtain ⟨x', hrun, hm⟩ := rotate_head_content self x pos d2 hd2 scAt htrig hfit hx
  refine ⟨x', hrun, ?_⟩
  intro j hj
  have hxl : 2 * d2.val ≤ x.val.length := hx
  obtain ⟨y0, hy0, hy0v⟩ :=
    rh_entry x'.val _ hm j _ (rotPrefix_full_fst x.val d2.val scAt j hj hxl)
  obtain ⟨y1, hy1, hy1v⟩ :=
    rh_entry x'.val _ hm (j + d2.val) _ (rotPrefix_full_snd x.val d2.val scAt j hj hxl)
  obtain ⟨t, ht, angle, hangle, hsc, hcos, hsin⟩ := htrig j hj
  have hsc' : word_fixed_sin_cos angle = ok (core.result.Result.Ok (scAt j)) := by
    unfold word_fixed_sin_cos
    exact hsc
  obtain ⟨b0, b1⟩ := rope_pair_spec_proved (x.val.getD j 0#i64) (x.val.getD (j + d2.val) 0#i64)
    t pos angle (scAt j) (toReal angle) hangle hsc' rfl
  refine ⟨y0, y1, angle, hy0, hy1, hsc', ?_, ?_⟩
  · rw [hy0v]
    exact b0
  · rw [hy1v]
    exact b1

end

end apeiron_word
