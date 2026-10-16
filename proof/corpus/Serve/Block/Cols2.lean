/-
  `Serve/Cols2.lean` — refinement statements for the column-wise residual add of
  `src/engine/step/cols.rs:1325` (`step.cols.add_cols`, its loop `step.cols.add_cols_loop` and body
  `step.cols.add_cols_loop.body`) and, below it, the column loops of the projection drivers
  (`step.cols.Engine.ffn_cols`, `step.cols.group_cols`, `step.cols.compose_cols_in`).
  Extraction: `proof/ApeironWord.lean`, module `ApeironWord`, namespace `apeiron_word`; the integer
  leaf `step.add_assign_fixed` and its three headlines come from the sibling `Serve/Cols.lean`.

  ## What `step.cols.add_cols` computes

  `hidden` and `delta` are column slices (`Slice (Vec i64)`). Writing `H` for the column value lists
  of `hidden` and `D` for those of `delta`, the Rust body is

      while i < hidden.len && i < delta.len && err.is_none {
          if let Err(e) = add_assign_fixed(&mut hidden[i], &delta[i]) { err = Some(e); }
          i += 1;
      }

  so the loop writes the checked elementwise lane sum `hidden[i] += delta[i]` into the first
  `min H.length D.length` columns and stops at the first column whose checked add refuses. The
  references below state that, not the loop:

  * `cols2_colAdd a b` — the checked elementwise sum of two equal-width columns (`addPrefix` at full
    width, the `Serve/Cols.lean` reference for `step.add_assign_fixed`);
  * `cols2_colPrefix H D k` — `H` with its first `k` columns each replaced by the checked sum with
    `D`'s;
  * `cols2_colOk a b` — the joint domain of one column pair: equal widths and every lane sum
    representable.

  ## Boundary

  `step.add_assign_fixed` (`Serve/Cols.lean`) carries the opaque `From<&str> for String` call on its
  unequal-width branch; the extraction defines that call as the axiom
  `alloc.string.String.Insts.CoreConvertFromShared0Str.from` (`ApeironWord.lean:1033`). Every
  headline below mentions `step.cols.add_cols_loop.body`, whose body calls that leaf, so its
  `#print axioms` carries that axiom in addition to `{propext, Classical.choice, Quot.sound}`; the
  unequal-width branch is never reached on any path proved here. The same dependency is carried by
  the accepted headlines of `Serve/Cols.lean`.

  ## Files not reached

  `step.cols.Engine.ffn_cols`, `step.cols.group_cols` and `step.cols.compose_cols_in` are NOT proved
  here: their loops call the opaque boundaries `step.cols.Engine.ffn_single_sched`,
  `step.Engine.ffn_q8act_at` (state relative to a hypothesis on the call result) and `fp.prim` /
  `Arc` / pool-`Excl` methods. Recorded as open, not established.
-/

import Serve.Block.Cols

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-! ## §0 · Reference definitions -/

/-- The canonical empty column; the `List.getD` default for an out-of-range column. -/
def cols2_emptyVec : alloc.vec.Vec Std.I64 :=
  { slice := Slice.from ([] : List Std.I64) (by simp), cap := 0#usize }

/-- Column `j` of a column slice, as the bare value slice. -/
def cols2_col (x : Slice (alloc.vec.Vec Std.I64)) (j : ℕ) : Slice Std.I64 :=
  (x.val.getD j cols2_emptyVec).slice

/-- One `List ℤ` per column: column `v`'s lane values. -/
def cols2_cvals (x : Slice (alloc.vec.Vec Std.I64)) : List (List ℤ) :=
  x.val.map (fun v => cols_ivals v.slice)

/-- The joint domain of one column pair: equal widths, and every lane sum representable. -/
def cols2_colOk (a b : Slice Std.I64) : Prop :=
  a.val.length = b.val.length ∧
    ∀ l, l < a.val.length → fits (a.val.getD l 0#i64) (b.val.getD l 0#i64)

/-- The checked elementwise sum of one column pair: the lane sums at every lane. -/
def cols2_colAdd (a b : List ℤ) : List ℤ := addPrefix a b a.length

/-- `H` with its first `k` columns each replaced by the checked elementwise sum with `D`'s. -/
def cols2_colPrefix (H D : List (List ℤ)) : ℕ → List (List ℤ)
  | 0 => H
  | (j + 1) => (cols2_colPrefix H D j).set j (cols2_colAdd (H.getD j []) (D.getD j []))

theorem cols2_colPrefix_zero (H D : List (List ℤ)) : cols2_colPrefix H D 0 = H := rfl

theorem cols2_colPrefix_succ (H D : List (List ℤ)) (j : ℕ) :
    cols2_colPrefix H D (j + 1) =
      (cols2_colPrefix H D j).set j (cols2_colAdd (H.getD j []) (D.getD j [])) := rfl

/-! ## §0a · Slice/Vec lemmas used by the loop proof -/

/-- The value slice behind `Vec.deref` is the vector's own slice. -/
theorem cols2_ivals_deref (v : alloc.vec.Vec Std.I64) :
    cols_ivals (alloc.vec.Vec.deref v) = cols_ivals v.slice := by
  simp [cols_ivals, alloc.vec.Vec.deref, alloc.vec.Vec.val]

/-- `Vec.deref` of a vector carries the same value list as the vector's own slice. -/
theorem cols2_deref_val (v : alloc.vec.Vec Std.I64) :
    (alloc.vec.Vec.deref v).val = v.slice.val := by
  simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]

/-- `Vec.ofSlice` keeps the value slice. -/
theorem cols2_ofSlice_slice (s : Slice Std.I64) :
    (alloc.vec.Vec.ofSlice s).slice = s := rfl

/-- `cvals` of `Vec.ofSlice s` at an existing index is `cols_ivals s`. -/
theorem cols2_ivals_ofSlice (s : Slice Std.I64) :
    cols_ivals (alloc.vec.Vec.ofSlice s).slice = cols_ivals s := rfl

theorem cols2_emptyVec_slice : cols2_emptyVec.slice = Slice.from ([] : List Std.I64) (by simp) :=
  rfl

theorem cols2_ivals_emptyVec : cols_ivals cols2_emptyVec.slice = [] := by
  simp [cols2_emptyVec_slice, cols_ivals]

/-- Column `j` read through `List.getD` inside `cvals`. -/
theorem cols2_cvals_getD (x : Slice (alloc.vec.Vec Std.I64)) (j : ℕ) :
    (cols2_cvals x).getD j [] = cols_ivals (cols2_col x j) := by
  have hd : ([] : List ℤ) = cols_ivals cols2_emptyVec.slice := by
    rw [cols2_ivals_emptyVec]
  rw [hd, cols2_col, ← List.getD_map (l := x.val) (f := fun v => cols_ivals v.slice)
    (d := cols2_emptyVec) (n := j)]
  rfl

/-- `cvals` after writing one column. -/
theorem cols2_cvals_set (x : Slice (alloc.vec.Vec Std.I64)) (i : Std.Usize)
    (w : alloc.vec.Vec Std.I64) :
    cols2_cvals (x.set i w) = (cols2_cvals x).set i.val (cols_ivals w.slice) := by
  simp only [cols2_cvals, Slice.set_val_eq, List.map_set]

/-- Writing column `i` leaves every other column unchanged. -/
theorem cols2_col_set_ne (x : Slice (alloc.vec.Vec Std.I64)) (i : Std.Usize)
    (w : alloc.vec.Vec Std.I64) (j : ℕ) (h : j ≠ i.val) :
    cols2_col (x.set i w) j = cols2_col x j := by
  simp only [cols2_col, Slice.set_val_eq, List.getD_eq_getElem?_getD,
    List.getElem?_set_ne h.symm]

/-- Writing column `i` preserves the column count. -/
theorem cols2_len_set (x : Slice (alloc.vec.Vec Std.I64)) (i : Std.Usize)
    (w : alloc.vec.Vec Std.I64) : (x.set i w).val.length = x.val.length := by
  simp only [Slice.set_val_eq, List.length_set]

/-- `cvals` length is the column count. -/
theorem cols2_cvals_length (x : Slice (alloc.vec.Vec Std.I64)) :
    (cols2_cvals x).length = x.val.length := by
  simp [cols2_cvals]

/-- Two `Usize`s are equal when their values are. -/
theorem cols2_usize_eq_of_val_eq {a b : Std.Usize} (h : a.val = b.val) : a = b :=
  UScalar.val_eq_imp a b h

/-- `Slice.len` of `Vec.deref` is `Slice.len` of the vector's slice. -/
theorem cols2_len_deref (v : alloc.vec.Vec Std.I64) :
    Slice.len (alloc.vec.Vec.deref v) = Slice.len v.slice := by
  apply cols2_usize_eq_of_val_eq
  rw [Slice.len_val, Slice.len_val]
  simp only [Slice.length, cols2_deref_val]

/-! ## §1 · The `step.cols.add_cols` body turns -/

/-- **One `add_cols` body turn, continue.** Column `i` is in range in both slices and its lane sums
are representable: the turn writes the checked sum into `hidden`'s column `i`, leaves `err`, returns
control with `i + 1`, keeps the column count, and leaves every other column untouched. -/
theorem cols2_add_cols_body_cont (delta hidden : Slice (alloc.vec.Vec Std.I64))
    (err : Option step.StepError) (i : Std.Usize)
    (hn : i.val < hidden.val.length) (hd : i.val < delta.val.length)
    (her : err = none)
    (hok : cols2_colOk (cols2_col hidden i.val) (cols2_col delta i.val)) :
    ∃ s : Slice (alloc.vec.Vec Std.I64), ∃ i' : Std.Usize,
      step.cols.add_cols_loop.body delta hidden err i = ok (ControlFlow.cont (s, none, i')) ∧
      i'.val = i.val + 1 ∧
      cols2_cvals s = (cols2_cvals hidden).set i.val
        (addPrefix (cols_ivals (cols2_col hidden i.val)) (cols_ivals (cols2_col delta i.val))
          (cols2_col hidden i.val).val.length) ∧
      s.val.length = hidden.val.length ∧
      ∀ j, j ≠ i.val → cols2_col s j = cols2_col hidden j := by
  have h1 : (1#usize : Std.Usize).val = 1 := rfl
  have hmax : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    have := Slice.length_ineq hidden
    omega
  unfold step.cols.add_cols_loop.body
  dsimp only
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hn)]
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hd)]
  rw [her]
  rw [if_pos (by rfl : core.option.Option.is_none none = true)]
  obtain ⟨p, hp, hpp⟩ := WP.spec_imp_exists (Slice.index_mut_usize_spec hidden i hn)
  obtain ⟨v0, imb⟩ := p
  simp only [WP.uncurry'_pair] at hpp
  obtain ⟨hvx, hvb⟩ := hpp
  rw [hp]
  simp only [bind_tc_ok, uncurry_apply_pair, Aeneas.Std.lift, alloc.vec.Vec.deref_mut, hvb]
  obtain ⟨v1, hv1, hv1x⟩ := WP.spec_imp_exists (Slice.index_usize_spec delta i hd)
  rw [hv1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  -- column translation
  have hcolH : hidden.val.getD i.val cols2_emptyVec = v0 := by
    rw [List.getD_eq_getElem hidden.val cols2_emptyVec hn]
    exact hvx.symm
  have hcolD : delta.val.getD i.val cols2_emptyVec = v1 := by
    rw [List.getD_eq_getElem delta.val cols2_emptyVec hd]
    exact hv1x.symm
  have hlenH : (cols2_col hidden i.val).val.length = v0.slice.val.length := by
    rw [cols2_col, hcolH]
  have hlenD : (cols2_col delta i.val).val.length = (alloc.vec.Vec.deref v1).val.length := by
    rw [cols2_col, hcolD, cols2_deref_val]
  have hxy : Slice.len v0.slice = Slice.len (alloc.vec.Vec.deref v1) := by
    apply cols2_usize_eq_of_val_eq
    rw [Slice.len_val, Slice.len_val]
    have h := hok.1
    rw [hlenH, hlenD] at h
    exact h
  have hfit : ∀ l, l < v0.slice.val.length →
      fits (v0.slice.val.getD l 0#i64) ((alloc.vec.Vec.deref v1).val.getD l 0#i64) := by
    intro l hl
    have h := hok.2 l (by rw [hlenH]; exact hl)
    simpa only [cols2_col, hcolH, hcolD, cols2_deref_val] using h
  obtain ⟨s2, hs2, hs2v⟩ := word_add_assign_fixed_ok v0.slice (alloc.vec.Vec.deref v1) hxy hfit
  rw [hs2]
  simp only [bind_tc_ok, uncurry_apply_pair]
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hmax)
  rw [hi3]
  simp only [bind_tc_ok, uncurry_apply_pair]
  refine ⟨hidden.set i (alloc.vec.Vec.ofSlice s2), i3, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rw [hi3v, h1]
  · have hch : cols2_col hidden i.val = v0.slice := by rw [cols2_col, hcolH]
    have hcd : cols_ivals (cols2_col delta i.val) = cols_ivals (alloc.vec.Vec.deref v1) := by
      rw [cols2_col, hcolD, cols2_ivals_deref]
    rw [cols2_cvals_set, cols2_ivals_ofSlice, hs2v, ← hch, ← hcd]
  · exact cols2_len_set hidden i (alloc.vec.Vec.ofSlice s2)
  · intro j hj
    exact cols2_col_set_ne hidden i (alloc.vec.Vec.ofSlice s2) j hj

/-! ## §2 · The `step.cols.add_cols` body: refusal and termination turns -/

/-- **One `add_cols` body turn, refusal.** Column `i` is in range in both slices and its checked
elementwise add returns `Err e` with output `out`: the turn writes `cols_ivals out` into `hidden`'s column
`i`, sets `err = some e`, returns control with `i + 1`, keeps the column count, and leaves every
other column untouched. -/
theorem cols2_add_cols_body_refuses (delta hidden : Slice (alloc.vec.Vec Std.I64))
    (err : Option step.StepError) (i : Std.Usize) (e : step.StepError) (out : Slice Std.I64)
    (hn : i.val < hidden.val.length) (hd : i.val < delta.val.length) (her : err = none)
    (haf : step.add_assign_fixed (cols2_col hidden i.val)
             (alloc.vec.Vec.deref (delta.val.getD i.val cols2_emptyVec))
           = ok (core.result.Result.Err e, out)) :
    ∃ s : Slice (alloc.vec.Vec Std.I64), ∃ i' : Std.Usize,
      step.cols.add_cols_loop.body delta hidden err i
        = ok (ControlFlow.cont (s, some e, i')) ∧
      i'.val = i.val + 1 ∧
      cols2_cvals s = (cols2_cvals hidden).set i.val (cols_ivals out) ∧
      s.val.length = hidden.val.length ∧
      ∀ j, j ≠ i.val → cols2_col s j = cols2_col hidden j := by
  have h1 : (1#usize : Std.Usize).val = 1 := rfl
  have hmax : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    have := Slice.length_ineq hidden
    omega
  unfold step.cols.add_cols_loop.body
  dsimp only
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hn)]
  rw [if_pos (by rw [UScalar.lt_equiv, Slice.len_val]; exact hd)]
  rw [her]
  rw [if_pos (by rfl : core.option.Option.is_none none = true)]
  obtain ⟨p, hp, hpp⟩ := WP.spec_imp_exists (Slice.index_mut_usize_spec hidden i hn)
  obtain ⟨v0, imb⟩ := p
  simp only [WP.uncurry'_pair] at hpp
  obtain ⟨hvx, hvb⟩ := hpp
  rw [hp]
  simp only [bind_tc_ok, uncurry_apply_pair, Aeneas.Std.lift, alloc.vec.Vec.deref_mut, hvb]
  obtain ⟨v1, hv1, hv1x⟩ := WP.spec_imp_exists (Slice.index_usize_spec delta i hd)
  rw [hv1]
  simp only [bind_tc_ok, uncurry_apply_pair]
  have hcolH : hidden.val.getD i.val cols2_emptyVec = v0 := by
    rw [List.getD_eq_getElem hidden.val cols2_emptyVec hn]
    exact hvx.symm
  have hcolD : delta.val.getD i.val cols2_emptyVec = v1 := by
    rw [List.getD_eq_getElem delta.val cols2_emptyVec hd]
    exact hv1x.symm
  have hL : cols2_col hidden i.val = v0.slice := by rw [cols2_col, hcolH]
  rw [hL, hcolD] at haf
  rw [haf]
  simp only [bind_tc_ok, uncurry_apply_pair]
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hmax)
  rw [hi3]
  simp only [bind_tc_ok, uncurry_apply_pair]
  refine ⟨hidden.set i (alloc.vec.Vec.ofSlice out), i3, ?_, ?_, ?_, ?_, ?_⟩
  · rfl
  · rw [hi3v, h1]
  · rw [cols2_cvals_set, cols2_ivals_ofSlice]
  · exact cols2_len_set hidden i (alloc.vec.Vec.ofSlice out)
  · intro j hj
    exact cols2_col_set_ne hidden i (alloc.vec.Vec.ofSlice out) j hj

/-- **One `add_cols` body turn, termination.** One of the loop guards fails (the error is set, or a
column index is past a slice end): the turn returns `done` with the accumulator and error unchanged. -/
theorem cols2_add_cols_body_done (delta hidden : Slice (alloc.vec.Vec Std.I64))
    (err : Option step.StepError) (i : Std.Usize)
    (h : err ≠ none ∨ ¬ i.val < hidden.val.length ∨ ¬ i.val < delta.val.length) :
    step.cols.add_cols_loop.body delta hidden err i
      = ok (ControlFlow.done (hidden, err)) := by
  unfold step.cols.add_cols_loop.body
  dsimp only
  by_cases hhn : i < Slice.len hidden
  · rw [if_pos hhn]
    by_cases hhd : i < Slice.len delta
    · rw [if_pos hhd]
      have her : err ≠ none := by
        rcases h with her | hn | hd
        · exact her
        · exact absurd (by rwa [UScalar.lt_equiv, Slice.len_val] at hhn) hn
        · exact absurd (by rwa [UScalar.lt_equiv, Slice.len_val] at hhd) hd
      have hb : core.option.Option.is_none err = false := by
        cases err with
        | none => exact absurd rfl her
        | some e => rfl
      rw [if_neg (by rw [hb]; exact Bool.false_ne_true)]
    · rw [if_neg hhd]
  · rw [if_neg hhn]

/-! ## §3 · The `step.cols.add_cols` loop and its two headlines -/

/-- **The `add_cols` loop, from cursor zero, all columns representable.** If every column pair below
the joint column count `min hidden.len delta.len` has equal widths and representable lane sums, the
loop writes the checked elementwise sum into every such column and returns `err = none`. -/
theorem cols2_add_cols_loop_spec_ok (hidden delta : Slice (alloc.vec.Vec Std.I64))
    (hpre : ∀ j, j < min hidden.val.length delta.val.length →
      cols2_colOk (cols2_col hidden j) (cols2_col delta j)) :
    step.cols.add_cols_loop hidden delta none 0#usize ⦃ o =>
      o.2 = none ∧
      cols2_cvals o.1 = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta)
        (min hidden.val.length delta.val.length) ⦄ := by
  unfold step.cols.add_cols_loop
  apply loop.spec_decr_nat
    (measure := fun st : Slice (alloc.vec.Vec Std.I64) × Option step.StepError × Std.Usize =>
      min hidden.val.length delta.val.length - st.2.2.val)
    (inv := fun st : Slice (alloc.vec.Vec Std.I64) × Option step.StepError × Std.Usize =>
      st.1.val.length = hidden.val.length ∧
      (∀ j, st.2.2.val ≤ j → cols2_col st.1 j = cols2_col hidden j) ∧
      st.2.2.val ≤ min hidden.val.length delta.val.length ∧ st.2.1 = none ∧
      cols2_cvals st.1 = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) st.2.2.val)
    (post := fun o : Slice (alloc.vec.Vec Std.I64) × Option step.StepError =>
      o.2 = none ∧
      cols2_cvals o.1 = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta)
        (min hidden.val.length delta.val.length))
  · rintro ⟨h, e, i⟩ ⟨hln, hcol, hile, he, hc⟩
    dsimp only at hln hcol hile he hc
    by_cases hik : i.val < min hidden.val.length delta.val.length
    · have hnf : i.val < hidden.val.length := lt_of_lt_of_le hik (Nat.min_le_left _ _)
      have hd : i.val < delta.val.length := lt_of_lt_of_le hik (Nat.min_le_right _ _)
      have hn : i.val < h.val.length := by rw [hln]; exact hnf
      have hok : cols2_colOk (cols2_col h i.val) (cols2_col delta i.val) := by
        rw [hcol i.val (le_refl _)]; exact hpre i.val hik
      obtain ⟨s, i', hb, hi'v, hsv, hsln, hscol⟩ :=
        cols2_add_cols_body_cont delta h e i hn hd he hok
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      dsimp only
      refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
      · rw [hsln, hln]
      · intro j hj
        rw [hi'v] at hj
        rw [hscol j (by omega), hcol j (by omega)]
      · rw [hi'v]; exact Nat.succ_le_of_lt hik
      · rfl
      · rw [hsv, hc, hi'v, cols2_colPrefix_succ, cols2_cvals_getD, cols2_cvals_getD,
          cols2_colAdd, cols_ivals_length, hcol i.val (le_refl _)]
      · rw [hi'v]; omega
    · have hie : i.val = min hidden.val.length delta.val.length :=
        le_antisymm hile (Nat.le_of_not_lt hik)
      have hor : ¬ i.val < hidden.val.length ∨ ¬ i.val < delta.val.length := by
        by_cases h1 : i.val < hidden.val.length
        · by_cases h2 : i.val < delta.val.length
          · exact absurd (lt_min h1 h2) hik
          · exact Or.inr h2
        · exact Or.inl h1
      have hpost : e = none ∧
          cols2_cvals h = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta)
            (min hidden.val.length delta.val.length) :=
        ⟨he, by rw [hc, hie]⟩
      rcases hor with hnh | hnd
      · have hb := cols2_add_cols_body_done delta h e i
          (Or.inr (Or.inl (by rw [hln]; exact hnh)))
        rw [WP.spec_equiv_exists]
        exact ⟨_, hb, hpost⟩
      · have hb := cols2_add_cols_body_done delta h e i (Or.inr (Or.inr hnd))
        rw [WP.spec_equiv_exists]
        exact ⟨_, hb, hpost⟩
  · refine ⟨rfl, ?_, by simp, rfl, ?_⟩
    · intro j _
      rfl
    · show cols2_cvals hidden = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta)
        (0#usize).val
      rw [show (0#usize : Std.Usize).val = 0 from rfl, cols2_colPrefix_zero]

/-- **Success law.** Every column below the joint count representable: the extracted
`step.cols.add_cols` writes the checked elementwise sum into every such column and returns `Ok ()`. -/
theorem word_add_cols_ok (hidden delta : Slice (alloc.vec.Vec Std.I64))
    (hpre : ∀ j, j < min hidden.val.length delta.val.length →
      cols2_colOk (cols2_col hidden j) (cols2_col delta j)) :
    ∃ out : Slice (alloc.vec.Vec Std.I64),
      step.cols.add_cols hidden delta = ok (core.result.Result.Ok (), out) ∧
      cols2_cvals out = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta)
        (min hidden.val.length delta.val.length) := by
  obtain ⟨v, ho, hpost⟩ := WP.spec_imp_exists (cols2_add_cols_loop_spec_ok hidden delta hpre)
  obtain ⟨r, e⟩ := v
  dsimp only at hpost
  obtain ⟨he, hcv⟩ := hpost
  refine ⟨r, ?_, hcv⟩
  unfold step.cols.add_cols
  rw [ho]
  simp only [he, bind_tc_ok, uncurry_apply_pair]

/-- **The `add_cols` loop, from cursor zero, first refusal.** If columns `0 … i₁-1` are
representable and column `i₁`'s checked elementwise add returns `Err e₁` with output `out₁`, the loop
writes those columns, writes `cols_ivals out₁` into column `i₁`, and returns `err = some e₁`. -/
theorem cols2_add_cols_loop_spec_refuse (hidden delta : Slice (alloc.vec.Vec Std.I64))
    (i₁ : ℕ) (e₁ : step.StepError) (out₁ : Slice Std.I64)
    (hi₁ : i₁ < min hidden.val.length delta.val.length)
    (hok₁ : ∀ j, j < i₁ → cols2_colOk (cols2_col hidden j) (cols2_col delta j))
    (href : step.add_assign_fixed (cols2_col hidden i₁)
              (alloc.vec.Vec.deref (delta.val.getD i₁ cols2_emptyVec))
            = ok (core.result.Result.Err e₁, out₁)) :
    step.cols.add_cols_loop hidden delta none 0#usize ⦃ o =>
      o.2 = some e₁ ∧
      cols2_cvals o.1 =
        (cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) i₁).set i₁ (cols_ivals out₁) ⦄ := by
  unfold step.cols.add_cols_loop
  apply loop.spec_decr_nat
    (measure := fun st : Slice (alloc.vec.Vec Std.I64) × Option step.StepError × Std.Usize =>
      i₁ + 1 - st.2.2.val)
    (inv := fun st : Slice (alloc.vec.Vec Std.I64) × Option step.StepError × Std.Usize =>
      st.1.val.length = hidden.val.length ∧
      (∀ j, st.2.2.val ≤ j → cols2_col st.1 j = cols2_col hidden j) ∧
      st.2.2.val ≤ i₁ + 1 ∧
      (st.2.2.val ≤ i₁ → st.2.1 = none ∧
        cols2_cvals st.1 = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) st.2.2.val) ∧
      (i₁ < st.2.2.val → st.2.1 = some e₁ ∧
        cols2_cvals st.1 =
          (cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) i₁).set i₁ (cols_ivals out₁)))
    (post := fun o : Slice (alloc.vec.Vec Std.I64) × Option step.StepError =>
      o.2 = some e₁ ∧
      cols2_cvals o.1 =
        (cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) i₁).set i₁ (cols_ivals out₁))
  · rintro ⟨h, e, i⟩ ⟨hln, hcol, hile, h1, h2⟩
    dsimp only at hln hcol hile h1 h2
    by_cases hik : i.val < min hidden.val.length delta.val.length
    · by_cases hlt : i.val < i₁
      · have hnf : i.val < hidden.val.length := lt_of_lt_of_le hik (Nat.min_le_left _ _)
        have hde : i.val < delta.val.length := lt_of_lt_of_le hik (Nat.min_le_right _ _)
        have hn : i.val < h.val.length := by rw [hln]; exact hnf
        have he : e = none := (h1 (le_of_lt hlt)).1
        have hc : cols2_cvals h
            = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) i.val :=
          (h1 (le_of_lt hlt)).2
        have hok : cols2_colOk (cols2_col h i.val) (cols2_col delta i.val) := by
          rw [hcol i.val (le_refl _)]; exact hok₁ i.val hlt
        obtain ⟨s, i', hb, hi'v, hsv, hsln, hscol⟩ :=
          cols2_add_cols_body_cont delta h e i hn hde he hok
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        dsimp only
        refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · rw [hsln, hln]
        · intro j hj
          rw [hi'v] at hj
          rw [hscol j (by omega), hcol j (by omega)]
        · rw [hi'v]; omega
        · intro _
          refine ⟨rfl, ?_⟩
          rw [hsv, hc, hi'v, cols2_colPrefix_succ, cols2_cvals_getD, cols2_cvals_getD,
            cols2_colAdd, cols_ivals_length, hcol i.val (le_refl _)]
        · intro hgt
          rw [hi'v] at hgt
          omega
        · rw [hi'v]; omega
      · have hge : i₁ ≤ i.val := le_of_not_gt hlt
        rcases lt_or_eq_of_le hge with hlt' | heq
        · obtain ⟨he₁, hcv⟩ := h2 hlt'
          have hb := cols2_add_cols_body_done delta h e i (Or.inl (by rw [he₁]; simp))
          rw [WP.spec_equiv_exists]
          exact ⟨_, hb, he₁, hcv⟩
        · have he : e = none := (h1 (le_of_eq heq.symm)).1
          have hc : cols2_cvals h
              = cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) i₁ := by
            rw [(h1 (le_of_eq heq.symm)).2, heq]
          have hn : i.val < h.val.length := by
            rw [hln, ← heq]
            exact lt_of_lt_of_le hi₁ (Nat.min_le_left _ _)
          have hd : i.val < delta.val.length := by
            rw [← heq]
            exact lt_of_lt_of_le hi₁ (Nat.min_le_right _ _)
          obtain ⟨s, i', hb, hi'v, hsv, hsln, hscol⟩ :=
            cols2_add_cols_body_refuses delta h e i e₁ out₁ hn hd he
              (by rw [← heq, hcol i₁ (le_of_eq heq.symm)]; exact href)
          rw [WP.spec_equiv_exists]
          refine ⟨_, hb, ?_⟩
          dsimp only
          refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
          · rw [hsln, hln]
          · intro j hj
            rw [hi'v] at hj
            rw [hscol j (by omega), hcol j (by omega)]
          · rw [hi'v]; omega
          · intro hle
            rw [hi'v] at hle
            omega
          · intro _
            refine ⟨rfl, ?_⟩
            rw [hsv, hc, heq]
          · rw [hi'v]; omega
    · have hlt' : i₁ < i.val := by omega
      obtain ⟨he₁, hcv⟩ := h2 hlt'
      have hb := cols2_add_cols_body_done delta h e i (Or.inl (by rw [he₁]; simp))
      rw [WP.spec_equiv_exists]
      exact ⟨_, hb, he₁, hcv⟩
  · refine ⟨rfl, ?_, by simp, ?_, ?_⟩
    · intro j _
      rfl
    · intro _
      exact ⟨rfl, by rw [show (0#usize : Std.Usize).val = 0 from rfl, cols2_colPrefix_zero]⟩
    · intro hh
      dsimp only at hh
      rw [show (0#usize : Std.Usize).val = 0 from rfl] at hh
      omega

/-- **Overflow refusal.** At the first column `i₁` below the joint count whose lane `l₁` is the first
non-representable sum, the extracted `step.cols.add_cols` writes the prefix `i₁` of the column check,
writes the partial column `i₁` through lane `l₁`, and returns `Err Overflow`. -/
theorem word_add_cols_refuses (hidden delta : Slice (alloc.vec.Vec Std.I64)) (i₁ l₁ : ℕ)
    (hi₁ : i₁ < min hidden.val.length delta.val.length)
    (hok₁ : ∀ j, j < i₁ → cols2_colOk (cols2_col hidden j) (cols2_col delta j))
    (hw : (cols2_col hidden i₁).val.length = (cols2_col delta i₁).val.length)
    (hl₁ : l₁ < (cols2_col hidden i₁).val.length)
    (hfit₁ : ∀ l, l < l₁ → fits ((cols2_col hidden i₁).val.getD l 0#i64)
      ((cols2_col delta i₁).val.getD l 0#i64))
    (hnf₁ : ¬ fits ((cols2_col hidden i₁).val.getD l₁ 0#i64)
      ((cols2_col delta i₁).val.getD l₁ 0#i64)) :
    ∃ out : Slice (alloc.vec.Vec Std.I64),
      step.cols.add_cols hidden delta
        = ok (core.result.Result.Err step.StepError.Overflow, out) ∧
      cols2_cvals out =
        (cols2_colPrefix (cols2_cvals hidden) (cols2_cvals delta) i₁).set i₁
          (addPrefix (cols_ivals (cols2_col hidden i₁)) (cols_ivals (cols2_col delta i₁)) l₁) := by
  have hxy : Slice.len (cols2_col hidden i₁)
      = Slice.len (alloc.vec.Vec.deref (delta.val.getD i₁ cols2_emptyVec)) := by
    apply cols2_usize_eq_of_val_eq
    rw [Slice.len_val, Slice.len_val]
    simp only [Slice.length, cols2_deref_val]
    simpa only [cols2_col] using hw
  have hfit' : ∀ l, l < l₁ →
      fits ((cols2_col hidden i₁).val.getD l 0#i64)
        ((alloc.vec.Vec.deref (delta.val.getD i₁ cols2_emptyVec)).val.getD l 0#i64) := by
    intro l hl
    simpa only [cols2_deref_val, cols2_col] using hfit₁ l hl
  have hnf' : ¬ fits ((cols2_col hidden i₁).val.getD l₁ 0#i64)
      ((alloc.vec.Vec.deref (delta.val.getD i₁ cols2_emptyVec)).val.getD l₁ 0#i64) := by
    simpa only [cols2_deref_val, cols2_col] using hnf₁
  obtain ⟨out₁, href, hivals₁⟩ := word_add_assign_fixed_refuses
    (cols2_col hidden i₁) (alloc.vec.Vec.deref (delta.val.getD i₁ cols2_emptyVec))
    hxy l₁ hl₁ hfit' hnf'
  obtain ⟨v, ho, hpost⟩ := WP.spec_imp_exists
    (cols2_add_cols_loop_spec_refuse hidden delta i₁ step.StepError.Overflow out₁ hi₁ hok₁ href)
  obtain ⟨r, e⟩ := v
  dsimp only at hpost
  obtain ⟨he, hcv⟩ := hpost
  refine ⟨r, ?_, ?_⟩
  · unfold step.cols.add_cols
    rw [ho]
    simp only [he, bind_tc_ok, uncurry_apply_pair]
  · rw [hcv, hivals₁, cols2_ivals_deref]
    rfl

/-! ### Non-vacuity of §0 -/

/-- The joint domain `cols2_colOk` is inhabited: a concrete two-lane pair whose lane sums fit. -/
example : cols2_colOk (Slice.from ([1#i64, 2#i64] : List Std.I64) (by scalar_tac))
    (Slice.from ([3#i64, 4#i64] : List Std.I64) (by scalar_tac)) := by
  refine ⟨by simp, ?_⟩
  intro l hl
  have hl' : l < 2 := by simpa using hl
  have h02 : l = 0 ∨ l = 1 := by omega
  rcases h02 with h | h
  · subst h
    simp only [Slice.from_val, List.getD_cons_zero, fits]
    constructor <;> norm_num [Std.I64.min_eq, Std.I64.max_eq]
  · subst h
    simp only [Slice.from_val, List.getD_cons_zero, List.getD_cons_succ, fits]
    constructor <;> norm_num [Std.I64.min_eq, Std.I64.max_eq]

/-- The column reference at a concrete two-lane pair. -/
example : cols2_colAdd [1, 2] [3, 4] = [4, 6] := by decide

/-- The column prefix at a concrete two-column input. -/
example :
    cols2_colPrefix [[1, 2], [3, 4]] [[1, 1], [1, 1]] 2 = [[2, 3], [4, 5]] := by decide

/-! ## Axiom cone -/

#print axioms cols2_add_cols_body_cont
#print axioms cols2_add_cols_body_refuses
#print axioms cols2_add_cols_body_done
#print axioms cols2_add_cols_loop_spec_ok
#print axioms word_add_cols_ok
#print axioms cols2_add_cols_loop_spec_refuse
#print axioms word_add_cols_refuses

end apeiron_word
