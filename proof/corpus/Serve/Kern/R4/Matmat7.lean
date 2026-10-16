import Serve.Kern.R4.Matmat6

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-!
# Serving kernel group `kern.matmat` (part 7): the `row_accs` accumulation loop

`src/engine/kern/matmat.rs:494-523` is `row_accs`' inner loop: for each output block `blk` it walks the
activation columns `c ∈ [0, |state|)`, and for a `Live` column writes
`totals[c] += d · (scales_blk[c] · dots[c])` into the `i128` accumulator, bailing with
`RowOutcome.Bail` on the first checked `i64`/`i128` overflow.

This file states WHAT the extracted body and loop compute, not how they loop:

* §0 value lemmas for `QuantAct::scales`, the checked `i64`/`i128` multiply and `i128` add, and
  `Array::index_usize` / `Array::update`, read off the Aeneas model under
  `target/aeneas-model/lean/Aeneas/Std` (never axiomatized).
* §1 the body step: a `Live` column with succeeding checked ops advances to `(totals.set c total, none, c+1)`
  with `total = totals[c] + d·(scale·dot)`; a column whose first checked `i64` multiply overflows
  advances to `(totals, some Bail, c+1)`.

Trust boundaries: rustc/LLVM codegen; the `F32`/AVX2 layers; `dequant.f16_to_fixed` and the
`q8_0_block_kdots` drivers of the outer `row_accs_loop0`.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`; `decide` is only used on literal `isize`/`usize`
values, never on an extracted body.
-/

/-! ## §0 · Value lemmas for the model operations -/

/-- `QuantAct::scales` returns the stored scale vector as a slice. -/
theorem matmat7_scales_eq (qa : chart.QuantAct) :
    chart.QuantAct.impl.scales qa = ok (alloc.vec.Vec.deref qa.scales) := rfl

/-- The checked `i64` multiply, success arm: the payload is the product. -/
theorem matmat7_i64_checked_mul_val {x y z : I64} (h : I64.checked_mul x y = some z) :
    z.val = x.val * y.val := by
  have hs := I64.checked_mul_bv_spec (x := x) (y := y)
  rw [show core.num.checked_mul_IScalar x y = some z from h] at hs
  exact hs.2.2.1

/-- The checked `i64` multiply refuses exactly off the `i64` range. -/
theorem matmat7_i64_checked_mul_none {x y : I64}
    (h : ¬ (I64.min ≤ x.val * y.val ∧ x.val * y.val ≤ I64.max)) :
    I64.checked_mul x y = none := by
  rcases hc : core.num.checked_mul_IScalar x y with _ | z
  · exact hc
  · have hs := I64.checked_mul_bv_spec (x := x) (y := y)
    rw [hc] at hs
    exact absurd ⟨hs.1, hs.2.1⟩ h

/-- The checked `i64` multiply, range witness: in range it is `some` of the product. -/
theorem matmat7_i64_checked_mul_some (x y : I64)
    (hlo : I64.min ≤ x.val * y.val) (hhi : x.val * y.val ≤ I64.max) :
    ∃ z : I64, I64.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  rcases hc : core.num.checked_mul_IScalar x y with _ | z
  · exfalso
    have hs := I64.checked_mul_bv_spec (x := x) (y := y)
    rw [hc] at hs
    exact hs ⟨hlo, hhi⟩
  · exact ⟨z, hc, matmat7_i64_checked_mul_val hc⟩

/-- The checked `i128` multiply, success arm: the payload is the product. -/
theorem matmat7_i128_checked_mul_val {x y z : I128} (h : I128.checked_mul x y = some z) :
    z.val = x.val * y.val := by
  have hs := I128.checked_mul_bv_spec (x := x) (y := y)
  rw [show core.num.checked_mul_IScalar x y = some z from h] at hs
  exact hs.2.2.1

/-- The checked `i128` multiply refuses exactly off the `i128` range. -/
theorem matmat7_i128_checked_mul_none {x y : I128}
    (h : ¬ (I128.min ≤ x.val * y.val ∧ x.val * y.val ≤ I128.max)) :
    I128.checked_mul x y = none := by
  rcases hc : core.num.checked_mul_IScalar x y with _ | z
  · exact hc
  · have hs := I128.checked_mul_bv_spec (x := x) (y := y)
    rw [hc] at hs
    exact absurd ⟨hs.1, hs.2.1⟩ h

/-- The checked `i128` multiply, range witness. -/
theorem matmat7_i128_checked_mul_some (x y : I128)
    (hlo : I128.min ≤ x.val * y.val) (hhi : x.val * y.val ≤ I128.max) :
    ∃ z : I128, I128.checked_mul x y = some z ∧ z.val = x.val * y.val := by
  rcases hc : core.num.checked_mul_IScalar x y with _ | z
  · exfalso
    have hs := I128.checked_mul_bv_spec (x := x) (y := y)
    rw [hc] at hs
    exact hs ⟨hlo, hhi⟩
  · exact ⟨z, hc, matmat7_i128_checked_mul_val hc⟩

/-- The checked `i128` add, success arm: the payload is the sum. -/
theorem matmat7_i128_checked_add_val {x y z : I128} (h : I128.checked_add x y = some z) :
    z.val = x.val + y.val := by
  have hs := I128.checked_add_bv_spec (x := x) (y := y)
  rw [show core.num.checked_add_IScalar x y = some z from h] at hs
  exact hs.2.2.1

/-- The checked `i128` add refuses exactly off the `i128` range. -/
theorem matmat7_i128_checked_add_none {x y : I128}
    (h : ¬ (I128.min ≤ x.val + y.val ∧ x.val + y.val ≤ I128.max)) :
    I128.checked_add x y = none := by
  rcases hc : core.num.checked_add_IScalar x y with _ | z
  · exact hc
  · have hs := I128.checked_add_bv_spec (x := x) (y := y)
    rw [hc] at hs
    exact absurd ⟨hs.1, hs.2.1⟩ h

/-- The checked `i128` add, range witness. -/
theorem matmat7_i128_checked_add_some (x y : I128)
    (hlo : I128.min ≤ x.val + y.val) (hhi : x.val + y.val ≤ I128.max) :
    ∃ z : I128, I128.checked_add x y = some z ∧ z.val = x.val + y.val := by
  rcases hc : core.num.checked_add_IScalar x y with _ | z
  · exfalso
    have hs := I128.checked_add_bv_spec (x := x) (y := y)
    rw [hc] at hs
    exact hs ⟨hlo, hhi⟩
  · exact ⟨z, hc, matmat7_i128_checked_add_val hc⟩

/-- `Array::index_usize` in bounds returns the indexed element. -/
theorem matmat7_array_index_usize_ok {α : Type} {n : Std.Usize} (v : Array α n) (i : Std.Usize)
    (h : i.val < v.val.length) : Array.index_usize v i = ok (v.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Array.index_usize_spec v i h)
  rw [hx, hxv]

/-- `Array::update` in bounds writes the element and returns the updated array. -/
theorem matmat7_array_update_ok {α : Type} {n : Std.Usize} (v : Array α n) (i : Std.Usize) (x : α)
    (h : i.val < v.val.length) : Array.update v i x = ok (v.set i x) := by
  obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists (Array.update_spec v i x h)
  rw [hy, hyv]

/-! ## §1 · The `row_accs` inner-loop body -/

/-- The loop state of `row_accs_loop0_loop0`: `(totals, early, c)`, matching the extraction. -/
abbrev Matmat7RowAccsSt := (Array I128 16#usize) × (Option kern.matmat.RowOutcome) × Std.Usize

/-- `row_accs_loop0_loop0.body`, `early = none`, on a `Live` in-range column whose checked
operations succeed: the accumulator is updated at `c` by `d·(scale·dot)` and the loop advances to
`c+1` carrying `early = none`. The composed value is `totals[c] + d·(scales_blk[c]·dots[c])`. -/
theorem matmat7_rowaccs_body_live (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (totals : Array I128 16#usize) (c : Std.Usize)
    (hlt : c.val < k.val) (hlen : c.val < state.val.length)
    (hlive : state.val[c.val] = kern.matmat.ColumnState.Live)
    (hacts : c.val < acts.val.length)
    (hscales : blk.val < (acts.val[c.val]).scales.val.length)
    (hdot : c.val < dots.val.length)
    (htot : c.val < totals.val.length)
    (scaled : I64) (term total : I128)
    (hmul : I64.checked_mul ((acts.val[c.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c.val])) = some scaled)
    (hterm : I128.checked_mul (core.convert.num.FromI128I64.from d)
              (core.convert.num.FromI128I64.from scaled) = some term)
    (hadd : I128.checked_add (totals.val[c.val]) term = some total) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals none c ⦃ r =>
      ∃ c1 : Std.Usize,
        r = ControlFlow.cont (totals.set c total, none, c1) ∧ c1.val = c.val + 1 ∧
        total.val = (totals.val[c.val]).val + d.val *
          (((acts.val[c.val]).scales.val[blk.val]).val * (dots.val[c.val]).val) ⦄ := by
  have hstate : Slice.index_usize state c = ok (state.val[c.val]) :=
    slice_index_usize_ok state c hlen
  have heq : kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq (state.val[c.val])
      kern.matmat.ColumnState.Live = ok true := matmat6_colstate_eq_live _ hlive
  have hscales' : blk.val < (alloc.vec.Vec.deref (acts.val[c.val]).scales).val.length := by
    simpa only [alloc.vec.Vec.deref, Slice.from_val] using hscales
  have hscaleeq : Slice.index_usize (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk
      = ok ((acts.val[c.val]).scales.val[blk.val]) := by
    rw [slice_index_usize_ok (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk hscales']
    simp only [alloc.vec.Vec.deref, Slice.from_val]
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  unfold kern.matmat.row_accs_loop0_loop0.body
  rw [if_pos ((UScalar.lt_equiv c k).2 hlt)]
  simp only [core.option.Option.is_none, Option.isNone_none, if_true, hstate, heq, bind_tc_ok,
    matmat7_scales_eq, hscaleeq, slice_index_usize_ok acts c hacts,
    matmat7_array_index_usize_ok dots c hdot, lift, hmul, hterm,
    matmat7_array_index_usize_ok totals c htot, hadd, matmat7_array_update_ok totals c total htot,
    hc1]
  step*
  refine ⟨c1, rfl, ?_, ?_⟩
  · rw [hc1v, show (1#usize).val = 1 from by decide]
  · simp only [matmat7_i128_checked_add_val hadd, matmat7_i128_checked_mul_val hterm,
      matmat7_i64_checked_mul_val hmul, core.convert.num.FromI64I32.from_val_eq,
      core.convert.num.FromI128I64.from_val_eq]

/-- `row_accs_loop0_loop0.body`, `early = none`, on a `Live` in-range column whose first checked
`i64` multiply overflows: the accumulator is left unchanged, `early` is set to `some Bail`, and the
loop advances to `c+1` (the next body call then stops). -/
theorem matmat7_rowaccs_body_bail_mul (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (totals : Array I128 16#usize) (c : Std.Usize)
    (hlt : c.val < k.val) (hlen : c.val < state.val.length)
    (hlive : state.val[c.val] = kern.matmat.ColumnState.Live)
    (hacts : c.val < acts.val.length)
    (hscales : blk.val < (acts.val[c.val]).scales.val.length)
    (hdot : c.val < dots.val.length)
    (hmul : I64.checked_mul ((acts.val[c.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c.val])) = none) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals none c ⦃ r =>
      ∃ c1 : Std.Usize,
        r = ControlFlow.cont (totals, some kern.matmat.RowOutcome.Bail, c1) ∧
        c1.val = c.val + 1 ⦄ := by
  have hstate : Slice.index_usize state c = ok (state.val[c.val]) :=
    slice_index_usize_ok state c hlen
  have heq : kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq (state.val[c.val])
      kern.matmat.ColumnState.Live = ok true := matmat6_colstate_eq_live _ hlive
  have hscales' : blk.val < (alloc.vec.Vec.deref (acts.val[c.val]).scales).val.length := by
    simpa only [alloc.vec.Vec.deref, Slice.from_val] using hscales
  have hscaleeq : Slice.index_usize (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk
      = ok ((acts.val[c.val]).scales.val[blk.val]) := by
    rw [slice_index_usize_ok (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk hscales']
    simp only [alloc.vec.Vec.deref, Slice.from_val]
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  unfold kern.matmat.row_accs_loop0_loop0.body
  rw [if_pos ((UScalar.lt_equiv c k).2 hlt)]
  simp only [core.option.Option.is_none, Option.isNone_none, if_true, hstate, heq, bind_tc_ok,
    matmat7_scales_eq, hscaleeq, slice_index_usize_ok acts c hacts,
    matmat7_array_index_usize_ok dots c hdot, lift, hmul, hc1]
  step*

/-! ## §2 · The `row_accs` inner-loop law -/

/-- Scale `blk` of activation column `j`, totalised with `qaDefault`/`0` outside the domain. -/
def matmat7_scale (acts : Slice chart.QuantAct) (blk : Std.Usize) (j : Nat) : I64 :=
  (acts.val.getD j qaDefault).scales.val.getD blk.val (0#i64)

/-- Activation dot of column `j`, totalised with `0` outside `dots`' domain. -/
def matmat7_dotv (dots : Array I32 16#usize) (j : Nat) : I32 :=
  dots.val.getD j (0#i32)

/-- The `i128` contribution `d·(scale·dot)` of a `Live` column `j`. -/
def matmat7_term (acts : Slice chart.QuantAct) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (j : Nat) : ℤ :=
  d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val)

/-- The per-column contribution the loop adds: `term` on a `Live` column, `0` otherwise. -/
def matmat7_delta (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (j : Nat) : ℤ :=
  match state.val.getD j kern.matmat.ColumnState.Live with
  | .Live => matmat7_term acts blk d dots j
  | _ => 0

/-- On `acts`' domain, `matmat7_scale` is the stored scale. -/
theorem matmat7_scale_eq (acts : Slice chart.QuantAct) (blk : Std.Usize) (j : Nat)
    (h : j < acts.val.length) (hb : blk.val < (acts.val[j]).scales.val.length) :
    matmat7_scale acts blk j = (acts.val[j]).scales.val[blk.val] := by
  unfold matmat7_scale
  rw [List.getD_eq_getElem acts.val qaDefault h,
    List.getD_eq_getElem (acts.val[j]).scales.val (0#i64) hb]

/-- On `dots`' domain, `matmat7_dotv` is the stored activation. -/
theorem matmat7_dotv_eq (dots : Array I32 16#usize) (j : Nat) (h : j < dots.val.length) :
    matmat7_dotv dots j = dots.val[j] := by
  unfold matmat7_dotv
  rw [List.getD_eq_getElem dots.val (0#i32) h]

/-- On the domain, `matmat7_term` is `d·(scale·dot)` in the extracted index forms. -/
theorem matmat7_term_eq (acts : Slice chart.QuantAct) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (j : Nat) (h : j < acts.val.length)
    (hb : blk.val < (acts.val[j]).scales.val.length) (hd : j < dots.val.length) :
    matmat7_term acts blk d dots j =
      d.val * (((acts.val[j]).scales.val[blk.val]).val * (dots.val[j]).val) := by
  unfold matmat7_term
  rw [matmat7_scale_eq acts blk j h hb, matmat7_dotv_eq dots j hd]

/-- On a `Live` column, `matmat7_delta` is the contribution term. -/
theorem matmat7_delta_live (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (j : Nat)
    (h : state.val.getD j kern.matmat.ColumnState.Live = kern.matmat.ColumnState.Live) :
    matmat7_delta acts state blk d dots j = matmat7_term acts blk d dots j := by
  simp only [matmat7_delta, h]

/-- `getD` after a list `set`: the written slot holds the new value, every other slot its old one. -/
theorem matmat7_list_getD_set (l : List I128) (i j : ℕ) (a : I128) (hj : j < l.length) :
    (l.set i a).getD j (0#i128) = if j = i then a else l.getD j (0#i128) := by
  have hj' : j < (l.set i a).length := by rw [List.length_set]; exact hj
  rw [List.getD_eq_getElem (l.set i a) (0#i128) hj', List.getElem_set hj',
    List.getD_eq_getElem l (0#i128) hj]
  by_cases h : j = i
  · rw [if_pos h.symm, if_pos h]
  · rw [if_neg (fun hc => h hc.symm), if_neg h]

/-- Loop invariant of `row_accs_loop0_loop0`: `early = none`, the cursor monotone in `[c, k]`, and
each accumulator slot equals its initial value plus the contributions of the `Live` columns already
passed. -/
def matmat7_loopInv (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (t0 : Array I128 16#usize)
    (c k : Nat) (st : Matmat7RowAccsSt) : Prop :=
  st.2.1 = none ∧ c ≤ st.2.2.val ∧ st.2.2.val ≤ k ∧ st.1.val.length = t0.val.length ∧
  (∀ j, j < t0.val.length →
    (st.1.val.getD j (0#i128)).val =
      (t0.val.getD j (0#i128)).val +
        (if c ≤ j ∧ j < st.2.2.val then matmat7_delta acts state blk d dots j else 0))

/-- **`row_accs` accumulation loop law, all-`Live` domain.** When every column `c ≤ j < k` is `Live`
and every checked `i64`/`i128` operation on it stays in range, the loop runs with `early = none` to
cursor `k` and each accumulator slot gains `d·(scale·dot)` on exactly the `Live` columns of `[c, k)`. -/
theorem word_matmat7_rowaccs_live_ok
    (acts : Slice chart.QuantAct) (k : Std.Usize) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize)
    (totals : Array I128 16#usize) (c : Std.Usize)
    (hc : c.val ≤ k.val) (hk : k.val ≤ state.val.length)
    (hacts : k.val ≤ acts.val.length) (hdots : k.val ≤ dots.val.length)
    (hlent : k.val ≤ totals.val.length)
    (hblk : ∀ j (hj : j < acts.val.length), c.val ≤ j → j < k.val →
      blk.val < (acts.val[j]).scales.val.length)
    (hlive : ∀ j (hj : j < state.val.length), c.val ≤ j → j < k.val →
      state.val[j] = kern.matmat.ColumnState.Live)
    (hdom : ∀ j, c.val ≤ j → j < k.val →
      I64.min ≤ (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ∧
      (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ≤ I64.max ∧
      I128.min ≤ d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
      d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max ∧
      I128.min ≤ (totals.val.getD j (0#i128)).val +
        d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
      (totals.val.getD j (0#i128)).val +
        d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max) :
    kern.matmat.row_accs_loop0_loop0 acts k state totals none blk d dots c ⦃ r =>
      ∃ totals' : Array Std.I128 16#usize, r = (totals', none) ∧
        (∀ j, j < totals.val.length →
          (totals'.val.getD j (0#i128)).val =
            (totals.val.getD j (0#i128)).val +
              (if c.val ≤ j ∧ j < k.val then matmat7_delta acts state blk d dots j else 0)) ⦄ := by
  unfold kern.matmat.row_accs_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Matmat7RowAccsSt => k.val - st.2.2.val)
    (inv := matmat7_loopInv acts state blk d dots totals c.val k.val)
    (post := fun y : (Array Std.I128 16#usize) × (Option kern.matmat.RowOutcome) =>
      ∃ totals' : Array Std.I128 16#usize, y = (totals', none) ∧
        (∀ j, j < totals.val.length →
          (totals'.val.getD j (0#i128)).val =
            (totals.val.getD j (0#i128)).val +
              (if c.val ≤ j ∧ j < k.val then matmat7_delta acts state blk d dots j else 0)))
  · rintro ⟨totals', early', c'⟩ hinv
    obtain ⟨he, hge, hle, hlen, hpoint⟩ := hinv
    dsimp only at he hge hle hlen hpoint
    subst he
    simp only []
    have h1v : (1#usize).val = 1 := by decide
    by_cases hlt : c'.val < k.val
    · have hstate_len : c'.val < state.val.length := lt_of_lt_of_le hlt hk
      have hlive' : state.val[c'.val] = kern.matmat.ColumnState.Live :=
        hlive c'.val hstate_len hge hlt
      have hliveD : state.val.getD c'.val kern.matmat.ColumnState.Live =
          kern.matmat.ColumnState.Live := by
        rw [List.getD_eq_getElem state.val kern.matmat.ColumnState.Live hstate_len, hlive']
      have hacts' : c'.val < acts.val.length := lt_of_lt_of_le hlt hacts
      have hdots' : c'.val < dots.val.length := lt_of_lt_of_le hlt hdots
      have hblk' : blk.val < (acts.val[c'.val]).scales.val.length := hblk c'.val hacts' hge hlt
      have hlent' : c'.val < totals.val.length := lt_of_lt_of_le hlt hlent
      have htot'' : c'.val < totals'.val.length := by rw [hlen]; exact hlent'
      have hsc' : matmat7_scale acts blk c'.val = (acts.val[c'.val]).scales.val[blk.val] :=
        matmat7_scale_eq acts blk c'.val hacts' hblk'
      have hdv' : matmat7_dotv dots c'.val = dots.val[c'.val] :=
        matmat7_dotv_eq dots c'.val hdots'
      have hval : ((acts.val[c'.val]).scales.val[blk.val]).val * (dots.val[c'.val]).val =
          (matmat7_scale acts blk c'.val).val * (matmat7_dotv dots c'.val).val := by
        rw [hsc', hdv']
      have hpc : (totals'.val.getD c'.val (0#i128)).val =
          (totals.val.getD c'.val (0#i128)).val := by
        have h := hpoint c'.val hlent'
        rw [if_neg (by rintro ⟨_, h2⟩; omega), add_zero] at h
        exact h
      obtain ⟨scaled, hmul, hscaledv⟩ := matmat7_i64_checked_mul_some
        ((acts.val[c'.val]).scales.val[blk.val]) (core.convert.num.FromI64I32.from (dots.val[c'.val]))
        (by rw [core.convert.num.FromI64I32.from_val_eq, hval]
            exact (hdom c'.val hge hlt).1)
        (by rw [core.convert.num.FromI64I32.from_val_eq, hval]
            exact (hdom c'.val hge hlt).2.1)
      obtain ⟨term, hterm, htermv⟩ := matmat7_i128_checked_mul_some
        (core.convert.num.FromI128I64.from d) (core.convert.num.FromI128I64.from scaled)
        (by rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
              hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
            exact (hdom c'.val hge hlt).2.2.1)
        (by rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
              hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
            exact (hdom c'.val hge hlt).2.2.2.1)
      obtain ⟨total, hadd, htotalv⟩ := matmat7_i128_checked_add_some
        (totals'.val[c'.val]'htot'') term
        (by rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, htermv,
              core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
              hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
            exact (hdom c'.val hge hlt).2.2.2.2.1)
        (by rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, htermv,
              core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
              hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
            exact (hdom c'.val hge hlt).2.2.2.2.2)
      obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
        (matmat7_rowaccs_body_live acts k state blk d dots totals' c' hlt hstate_len hlive'
          hacts' hblk' hdots' htot'' scaled term total hmul hterm hadd)
      obtain ⟨c1, hr, hc1v, htotv⟩ := hpost
      rw [hbody, hr]
      simp only [WP.spec_ok]
      show matmat7_loopInv acts state blk d dots totals c.val k.val
              (totals'.set c' total, none, c1) ∧ k.val - c1.val < k.val - c'.val
      simp only [matmat7_loopInv]
      refine ⟨⟨trivial, by omega, by omega, ?_, ?_⟩, by omega⟩
      · rw [Array.set_val_eq, List.length_set]; exact hlen
      · intro j hj
        have hjT : j < totals'.val.length := by rw [hlen]; exact hj
        have hset := matmat7_list_getD_set totals'.val c'.val j total hjT
        rw [Array.set_val_eq]
        by_cases hjc : j = c'.val
        · rw [if_pos hjc] at hset
          rw [hset, if_pos (by rw [hjc]; exact ⟨hge, by omega⟩), hjc]
          rw [htotv, ← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc,
            matmat7_delta_live acts state blk d dots c'.val hliveD,
            matmat7_term_eq acts blk d dots c'.val hacts' hblk' hdots']
        · rw [if_neg hjc] at hset
          rw [hset, hpoint j hj]
          congr 1
          by_cases hcond : c.val ≤ j ∧ j < c'.val
          · rw [if_pos hcond, if_pos ⟨hcond.1, by omega⟩]
          · rw [if_neg hcond,
              if_neg (fun hc' => hcond ⟨hc'.1, by omega⟩)]
    · have hck : c'.val = k.val := le_antisymm hle (by omega)
      unfold kern.matmat.row_accs_loop0_loop0.body
      rw [if_neg (fun hc' => hlt ((UScalar.lt_equiv c' k).1 hc'))]
      simp only [WP.spec_ok]
      refine ⟨totals', rfl, ?_⟩
      intro j hj
      rw [hpoint j hj, hck]
  · refine ⟨rfl, le_rfl, hc, rfl, ?_⟩
    intro j hj
    dsimp only
    rw [if_neg (by rintro ⟨h1, h2⟩; omega), add_zero]

end apeiron_word
