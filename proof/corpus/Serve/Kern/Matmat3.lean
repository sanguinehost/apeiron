import Serve.Kern.Matmat2

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-!
# Serving kernel group `kern.matmat` (part 3): `fast_bound_ok_loop`

`src/engine/kern/matmat.rs:451-470` (`fast_bound_ok`) walks the `I64` activation scale slice
`q.scales` and folds an `I128` running maximum of the lane magnitudes `|scales[b]|`; the driver then
compares two saturating products against `I64::MAX` and `I128::MAX`.

This file refines the first walk, `kern.matmat.fast_bound_ok_loop`, against the plain-`Int` reference
`matmat3_smaxFrom`: the running maximum of `|l[j]|` over `j < n`. The extracted body is not restated;
the loop theorem is the `loop.spec_decr_nat` induction over the body. The headline
`word_matmat3_fast_bound_ok_loop_ok` states that the walk from `(0, 0)` returns
`matmat3_smaxRef q = max_{l < |scales|} |scales[l]|`.

Admissible domain: the fold reads `scales[b]` and increments `b` with a checked `b + 1#usize`, so the
stored slice length must satisfy `|scales| < Usize.max` (`hsz`); a slice of maximal length would make
the extracted increment refuse. The counter bound `b ≤ |scales|` is carried by the invariant
`matmat3_smaxInv`.

Trust boundary: rustc/LLVM codegen, the remaining `fast_bound_ok` driver (the `saturating_mul` /
`F16_MAX_FIXED` comparison arithmetic), the `matmat_q8act` / `per_column` / `pooled_rows` loops and
the `R[x]/(x²−σ)` word kernels are not covered here.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`. `decide` is not used on an extracted body.
-/

/-! ## Reference: the running maximum of `|scales|` -/

/-- `matmat3_smaxFrom l n` is `max_{j<n} |l[j]|` as an `Int`, with the empty fold at `0`. -/
def matmat3_smaxFrom (l : List I64) : Nat → Int
  | 0 => 0
  | n + 1 => max (matmat3_smaxFrom l n) ((l.getD n 0#i64).val.natAbs : Int)

theorem matmat3_smaxFrom_succ (l : List I64) (n : Nat) :
    matmat3_smaxFrom l (n + 1) =
      max (matmat3_smaxFrom l n) ((l.getD n 0#i64).val.natAbs : Int) := rfl

theorem matmat3_smaxFrom_nonneg (l : List I64) (n : Nat) : 0 ≤ matmat3_smaxFrom l n := by
  induction n with
  | zero => simp [matmat3_smaxFrom]
  | succ k ih => rw [matmat3_smaxFrom_succ]; exact le_trans ih (le_max_left _ _)

/-- The value of the fold over the whole stored slice: `q.scales`. -/
def matmat3_smaxRef (q : chart.QuantAct) : Int :=
  matmat3_smaxFrom q.scales.val q.scales.val.length

/-- Loop invariant: the counter stays inside the slice and the accumulator is the running maximum. -/
def matmat3_smaxInv (q : chart.QuantAct) (st : I128 × Usize) : Prop :=
  st.2.val ≤ q.scales.val.length ∧ st.1.val = matmat3_smaxFrom q.scales.val st.2.val

/-! ## Reads and the `I128` magnitude -/

/-- In-bounds `List.getElem` of an `I64` list is its `getD` (the `U8` version is
`Numeric.Packing.Format.q4_getD_getElem`). -/
theorem matmat3_i64_getD_getElem (l : List I64) (i : Usize) (h : i.val < l.length) :
    (l[i.val]'h).val = (l.getD i.val 0#i64).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

/-- The `I64 → I128` widening followed by `I128.abs` returns the `Nat`-absolute value of the lane. -/
@[step]
theorem matmat3_I128_abs_from_spec (x : I64) :
    core.num.I128.abs (core.convert.num.FromI128I64.from x)
      ⦃ y => y.val = ((x.val.natAbs : Nat) : Int) ⦄ := by
  have hx := x.hBounds
  rw [IScalarTy.I64_numBits_eq] at hx
  norm_num at hx
  unfold core.num.I128.abs
  rw [if_neg (by
    rw [core.convert.num.FromI128I64.from_val_eq, I128.min_eq]
    omega)]
  simp only [WP.spec_ok]
  rw [core.convert.num.FromI128I64.from_val_eq]
  apply I128.ofInt_val_eq
  have hY : ((x.val.natAbs : Nat) : Int) = |(x.val : Int)| :=
    (Int.abs_eq_natAbs (a := x.val)).symm
  rw [hY, IScalarTy.I128_numBits_eq]
  have habs : |(x.val : Int)| ≤ 2 ^ 63 := by
    rw [abs_le]
    exact ⟨hx.1, le_of_lt hx.2⟩
  constructor
  · exact le_trans (by norm_num : (-(2 : Int) ^ 127) ≤ 0) (abs_nonneg (x.val : Int))
  · exact lt_of_le_of_lt habs (by norm_num : (2 : Int) ^ 63 < 2 ^ 127)

/-- The guard `b < Slice.len (deref q.scales)` is the counter bound against the stored length. -/
theorem matmat3_len_lt_deref (q : chart.QuantAct) (b : Usize) :
    (b < Slice.len (alloc.vec.Vec.deref q.scales)) ↔ b.val < q.scales.val.length := by
  rw [UScalar.lt_equiv]
  simp only [Slice.len_val, Slice.length, alloc.vec.Vec.deref, Slice.from_val]

/-! ## The `fast_bound_ok_loop` body and loop -/

/-- One iteration: on `b < |scales|` it reads `scales[b]`, folds `|scales[b]|` into the maximum and
advances the counter; otherwise it returns the accumulator. -/
theorem matmat3_fast_bound_ok_loop_body_spec (q : chart.QuantAct)
    (hsz : q.scales.val.length < Usize.max) (s_max : I128) (b : Usize)
    (hinv : matmat3_smaxInv q (s_max, b)) :
    kern.matmat.fast_bound_ok_loop.body q s_max b ⦃ r =>
      match r with
      | ControlFlow.done y => y.val = matmat3_smaxRef q
      | ControlFlow.cont st' => matmat3_smaxInv q st' ∧
          q.scales.val.length - st'.2.val < q.scales.val.length - b.val ⦄ := by
  obtain ⟨hb, hsmax⟩ := hinv
  have hb' : b.val ≤ q.scales.val.length := hb
  have hsmax' : s_max.val = matmat3_smaxFrom q.scales.val b.val := hsmax
  by_cases hlt : b < Slice.len (alloc.vec.Vec.deref q.scales)
  · have hbnd : b.val < q.scales.val.length := (matmat3_len_lt_deref q b).mp hlt
    unfold kern.matmat.fast_bound_ok_loop.body
    simp only [chart.QuantAct.impl.scales, bind_tc_ok]
    rw [if_pos hlt]
    simp only [lift, bind_tc_ok]
    step*
    by_cases hs : a > s_max
    · -- the lane magnitude exceeds `s_max`: the accumulator becomes `a`
      rw [if_pos hs]
      simp only [bind_tc_ok]
      step*
      have hi1v : i1.val = (q.scales.val.getD b.val 0#i64).val := by
        rw [i1_post]
        simp only [alloc.vec.Vec.deref, Slice.from_val]
        exact matmat3_i64_getD_getElem q.scales.val b hbnd
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [b1_post]; omega
      · rw [b1_post, matmat3_smaxFrom_succ, ← hsmax', ← hi1v, ← a_post]
        exact (max_eq_right (le_of_lt ((IScalar.lt_equiv s_max a).mp hs))).symm
      · rw [b1_post]; omega
    · -- the lane magnitude does not exceed `s_max`: the accumulator is kept
      rw [if_neg hs]
      simp only [bind_tc_ok]
      step*
      have hi1v : i1.val = (q.scales.val.getD b.val 0#i64).val := by
        rw [i1_post]
        simp only [alloc.vec.Vec.deref, Slice.from_val]
        exact matmat3_i64_getD_getElem q.scales.val b hbnd
      have hle : a.val ≤ s_max.val := by
        have := (IScalar.lt_equiv s_max a).not.mp hs
        omega
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [b1_post]; omega
      · rw [b1_post, matmat3_smaxFrom_succ, ← hsmax', ← hi1v, ← a_post]
        exact (max_eq_left hle).symm
      · rw [b1_post]; omega
  · -- `b ≥ |scales|`: the fold is complete
    unfold kern.matmat.fast_bound_ok_loop.body
    simp only [chart.QuantAct.impl.scales, bind_tc_ok]
    rw [if_neg hlt]
    simp only [WP.spec_ok]
    have hge : q.scales.val.length ≤ b.val :=
      Nat.le_of_not_lt ((matmat3_len_lt_deref q b).not.mp hlt)
    have hbl : b.val = q.scales.val.length := le_antisymm hb' hge
    rw [matmat3_smaxRef, ← hbl, ← hsmax']

/-- From `(0, 0)` the walk returns the running maximum over the whole scale slice. -/
theorem word_matmat3_fast_bound_ok_loop_ok (q : chart.QuantAct)
    (hsz : q.scales.val.length < Usize.max) :
    kern.matmat.fast_bound_ok_loop q 0#i128 0#usize ⦃ r =>
      r.val = matmat3_smaxRef q ⦄ := by
  unfold kern.matmat.fast_bound_ok_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Usize => q.scales.val.length - s.2.val)
    (inv := fun s => matmat3_smaxInv q s)
    (post := fun y : I128 => y.val = matmat3_smaxRef q)
  · rintro ⟨s_max, b⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
      (matmat3_fast_bound_ok_loop_body_spec q hsz s_max b hinv)
    rw [hr]
    rcases r with y | s'
    · simpa only [WP.spec_ok] using hpost
    · simpa only [WP.spec_ok] using hpost
  · refine ⟨by simp, by simp [matmat3_smaxFrom]⟩

/-! ## Non-vacuity: the invariant and the domain are inhabited -/

example (q : chart.QuantAct) : matmat3_smaxInv q (0#i128, 0#usize) :=
  ⟨by simp, by simp [matmat3_smaxFrom]⟩

example (l : List I64) : 0 ≤ matmat3_smaxFrom l 3 := matmat3_smaxFrom_nonneg l 3

/-! ## Axiom cone -/

#print axioms apeiron_word.word_matmat3_fast_bound_ok_loop_ok
#print axioms apeiron_word.matmat3_fast_bound_ok_loop_body_spec

end apeiron_word
