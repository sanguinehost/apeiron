import Learn.Solve.Wsyrk

/-!
# The extracted `src/learn/solve/syrk.rs`: branch witnesses

The refinements in `Learn.Solve.Wsyrk` are conditional on an explicit admissible domain
(`wsyrkDomain`) or an explicit boundary law (`MapSome`). This module shows each **unconditional**
headline is inhabited on a concrete input, and that each refusal arm `wsyrk` can construct is
reachable, so neither `word_wsyrk_refuses` nor `word_hadamard_refuses` is vacuous:

* the accepted domain `wsyrkDomain wOne none 1#usize 1#usize` holds, so the refusal hypothesis
  `¬ wsyrkDomain …` is not true of every input;
* one reachable input per refusal arm, in the extraction's own check order:
  `ZeroN` (`n = 0`), `ZeroO` (`o = 0`), `Overflow` from `n·n` (`n = Usize::MAX`), `ShapeMismatch`
  from `o·n` (`o = Usize::MAX`, `n = 2`), `ShapeMismatch` from `Slice.len x ≠ o·n`,
  `ShapeMismatch` from a present `w` with `Slice.len w ≠ o`, `Overflow` from `o ≥ MAX_O = 2¹⁸`
  (a length-`2¹⁸` input), and `InputBound` from an entry outside the strict `±2³⁶` boundary;
* `hadamard_into` refuses a length mismatch and returns its first argument unwritten;
* `within_bound` returns `true` on an accepted input and its elementwise fit predicate `hadFit`
  holds at a concrete index.

**What this module does not witness, stated plainly.** `word_hadamard_ok`, the two rounder
specifications `syrk_round32_spec` / `syrk_round64_spec`, and (when written) `word_wsyrk_ok` are
stated **conditional on `MapSome`**. `MapSome` is the `some`/`none` branch law of the opaque
boundary axiom `core.option.Option.map` (`proof/ApeironWord.lean`; admitted in
`proof/boundary-axioms.txt` as a std operation absent from the pinned Aeneas model). That axiom has
no body in the model, so no term of type `MapSome` is constructible here; those headlines carry no
concrete instance and are labelled conditional rather than exhibited. `word_wsyrk_ok` itself is
open: the `wsyrk_loop0`/`wsyrk_loop1`/`transpose_panel` chain is not refined in this development.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the vector container layout are not covered;
only the extracted integer body.
-/

set_option linter.apeiron.notation false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## 1. Concrete slices -/

/-- The empty `i64` slice. -/
def wNil : Slice I64 := Slice.new I64

/-- The one-element all-zero `i64` slice. -/
def wOne : Slice I64 :=
  Slice.from [0#i64] (by have h := dense_usize_le_max (1#usize); simpa using h)

/-- The one-element `i64::MAX` slice: outside the strict `±2³⁶` boundary. -/
def wWide : Slice I64 :=
  Slice.from [core.num.I64.MAX] (by have h := dense_usize_le_max (1#usize); simpa using h)

/-- A `2¹⁸`-element all-zero list: the smallest `x` an `o ≥ MAX_O = 2¹⁸`, `n = 1` call admits. -/
def wBigList : List I64 := List.replicate 262144 (0#i64)

theorem wBigList_len : wBigList.length = 262144 := by
  rw [wBigList, List.length_replicate]

theorem wBigList_le : wBigList.length ≤ Usize.max := by
  rw [wBigList_len]
  have h := Usize.cMax_bound_concrete.1
  omega

/-- The `2¹⁸`-element all-zero `i64` slice. -/
def wBig : Slice I64 := Slice.from wBigList wBigList_le

/-! ## 2. Lengths and small value facts -/

theorem w_nil_len : Slice.len wNil = 0#usize := by
  apply UScalar.eq_of_val_eq
  rw [Slice.len_val]
  rfl

theorem w_one_len : Slice.len wOne = 1#usize := by
  apply UScalar.eq_of_val_eq
  rw [Slice.len_val]
  rfl

theorem w_big_len_val : (Slice.len wBig).val = 262144 := by
  have h : wBig.val.length = 262144 := by
    show (Slice.from wBigList wBigList_le).val.length = 262144
    rw [Slice.from_val]
    exact wBigList_len
  rw [Slice.len_val]
  exact h

theorem w_big_ge18 : 2 ^ 18 ≤ (Slice.len wBig).val := by
  rw [w_big_len_val]
  norm_num

theorem w_one_ne_zero : (1#usize : Usize) ≠ 0#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  simp at h1

theorem w_two_ne_zero : (2#usize : Usize) ≠ 0#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  simp at h1

theorem w_zero_ne_one : (0#usize : Usize) ≠ 1#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  rw [show (0#usize).val = 0 from rfl, show (1#usize).val = 1 from rfl] at h1
  omega

theorem w_max_ne_zero : core.num.Usize.MAX ≠ 0#usize := by
  intro h
  have h1 : core.num.Usize.MAX.val = (0#usize).val := congrArg UScalar.val h
  have h2 : core.num.Usize.MAX.val = Usize.max := rfl
  rw [h2, show (0#usize).val = 0 from rfl] at h1
  have h3 : 1 ≤ Usize.max := by
    have h4 : 4294967295 ≤ Usize.max := Usize.cMax_bound_concrete.1
    omega
  omega

theorem w_big_ne_zero : Slice.len wBig ≠ 0#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  rw [w_big_len_val, show (0#usize).val = 0 from rfl] at h1
  omega

/-! ## 3. Checked-arithmetic pins -/

/-- A `usize` product that does not overflow is the pinned `checked_mul`. Proved through the
bit-vector spec, not by reduction: `Usize.numBits = System.Platform.numBits` is opaque. -/
theorem w_pin_mul {a b c : Usize} (hv : a.val * b.val = c.val)
    (h : a.val * b.val ≤ Usize.max) : Usize.checked_mul a b = some c := by
  have hspec := Usize.checked_mul_bv_spec (x := a) (y := b)
  cases hm : Usize.checked_mul a b with
  | none => rw [hm] at hspec; omega
  | some z =>
    rw [hm] at hspec
    rw [Option.some.injEq]
    exact UScalar.eq_of_val_eq (by rw [← hv]; exact hspec.2.1)

theorem w_mul_one (a : Usize) : Usize.checked_mul a 1#usize = some a := by
  have hv : a.val * (1#usize).val = a.val := by
    rw [show (1#usize).val = 1 from rfl, Nat.mul_one]
  exact w_pin_mul hv (by rw [hv]; exact dense_usize_le_max a)

theorem w_mul11 : Usize.checked_mul 1#usize 1#usize = some 1#usize := w_mul_one 1#usize

theorem w_mul22 : Usize.checked_mul 2#usize 2#usize = some 4#usize := by
  have hv : (2#usize).val * (2#usize).val = (4#usize).val := by
    rw [show (2#usize).val = 2 from rfl, show (4#usize).val = 4 from rfl]
  exact w_pin_mul hv (by rw [hv]; exact dense_usize_le_max 4#usize)

theorem w_max_max_none : Usize.checked_mul core.num.Usize.MAX core.num.Usize.MAX = none := by
  have hspec := Usize.checked_mul_bv_spec (x := core.num.Usize.MAX) (y := core.num.Usize.MAX)
  cases hm : Usize.checked_mul core.num.Usize.MAX core.num.Usize.MAX with
  | none => rfl
  | some z =>
    rw [hm] at hspec
    have hx : core.num.Usize.MAX.val = Usize.max := rfl
    rw [hx] at hspec
    have h2 : 2 ≤ Usize.max := by have h := dense_usize_le_max (2#usize); simpa using h
    nlinarith [hspec.1, h2]

theorem w_max_two_none : Usize.checked_mul core.num.Usize.MAX 2#usize = none := by
  have hspec := Usize.checked_mul_bv_spec (x := core.num.Usize.MAX) (y := 2#usize)
  cases hm : Usize.checked_mul core.num.Usize.MAX 2#usize with
  | none => rfl
  | some z =>
    rw [hm] at hspec
    have hx : core.num.Usize.MAX.val = Usize.max := rfl
    have h2 : (2#usize).val = 2 := rfl
    rw [hx, h2] at hspec
    have h1 : 1 ≤ Usize.max := by
      have h4 : 4294967295 ≤ Usize.max := Usize.cMax_bound_concrete.1
      omega
    nlinarith [hspec.1, h1]

/-! ## 4. Domain checks on concrete input -/

/-- `wNil` has no length-`4` shape: `Slice.len wNil = 0`. -/
theorem w_nil_len_ne_four : Slice.len wNil ≠ 4#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  simp [w_nil_len] at h1

/-- `wNil` has no length-`1` shape: `Slice.len wNil = 0`. -/
theorem w_nil_len_ne_one : Slice.len wNil ≠ 1#usize := by
  intro h
  have h1 := congrArg UScalar.val h
  simp [w_nil_len] at h1

theorem w_getd_one : wOne.val.getD 0 default = 0#i64 := by
  show (Slice.from [0#i64] _).val.getD 0 default = 0#i64
  rw [Slice.from_val]
  rfl

theorem w_getd_wide : wWide.val.getD 0 default = core.num.I64.MAX := by
  show (Slice.from [core.num.I64.MAX] _).val.getD 0 default = core.num.I64.MAX
  rw [Slice.from_val]
  rfl

theorem w_i64max_val : core.num.I64.MAX.val = (2 : ℤ) ^ 63 - 1 := by decide

theorem w_i64zero_val : (0#i64).val = 0 := by decide

theorem w_abs_zero : (uabs (0#i64)).val = 0 := by
  rw [uabs_val, w_i64zero_val]
  rfl

theorem w_abs_wide : (uabs core.num.I64.MAX).val = 2 ^ 63 - 1 := by
  rw [uabs_val, w_i64max_val]
  norm_num

theorem w_one_val_len : wOne.val.length = 1 := by
  have h := congrArg UScalar.val w_one_len
  rw [Slice.len_val] at h
  simpa using h

theorem w_wide_len : Slice.len wWide = 1#usize := by
  apply UScalar.eq_of_val_eq
  rw [Slice.len_val]
  rfl

theorem w_wide_val_len : wWide.val.length = 1 := by
  have h := congrArg UScalar.val w_wide_len
  rw [Slice.len_val] at h
  simpa using h

/-- `wOne` is inside the strict `±2³⁶` boundary. -/
theorem w_one_bound : withinBoundRef wOne none := by
  refine ⟨?_, ?_⟩
  · intro t ht
    rw [w_one_val_len] at ht
    have ht0 : t = 0 := by omega
    subst ht0
    rw [w_getd_one, w_abs_zero, boundLimit_val]
    norm_num
  · intro ws hws
    exact absurd hws (by simp)

/-- `wWide` is outside the strict `±2³⁶` boundary: `|i64::MAX| = 2⁶³ - 1 ≥ 2³⁶`. -/
theorem w_wide_not_below : ¬ wsyrkAllBelowRef wWide boundLimit := by
  intro h
  have h0 := h 0 (by rw [w_wide_val_len]; omega)
  rw [w_getd_wide, w_abs_wide, boundLimit_val] at h0
  norm_num at h0

theorem w_wide_not_bound : ¬ withinBoundRef wWide none :=
  fun h => w_wide_not_below h.1

/-! ## 5. The accepted domain is inhabited -/

/-- `wOne` with `o = n = 1` and no weights is accepted: the refusal surface is not all inputs. -/
theorem wsyrkDomain_witness : wsyrkDomain wOne none 1#usize 1#usize := by
  refine ⟨w_one_ne_zero, w_one_ne_zero, ?_, ?_, ?_, ?_, ?_⟩
  · rw [w_mul11]; exact Option.some_ne_none 1#usize
  · intro on hon
    have hon' : on = 1#usize := Option.some.inj (hon.symm.trans w_mul11)
    rw [hon', w_one_len]
  · intro ws hws; exact absurd hws (by simp)
  · rw [show (1#usize).val = 1 from rfl]; norm_num
  · exact w_one_bound

/-- `within_bound` returns `true` on the accepted input `wOne`. -/
theorem within_bound_witness :
    ∃ b : Bool, solve_port.syrk.within_bound wOne none = ok b ∧ b = true := by
  obtain ⟨b, hb, hiff⟩ := syrk_within_bound_spec wOne none
  exact ⟨b, hb, hiff.mpr w_one_bound⟩

/-- The elementwise fit predicate holds at index `0` for the all-zero one-element pair. -/
theorem hadFit_witness : hadFit (qvals wOne) (qvals wOne) 0 := by
  have hq : qvals wOne = [0] := by simp [qvals, wOne, Slice.from_val]
  have hg : ([0] : List ℤ).getD 0 0 = 0 := by decide
  unfold hadFit
  rw [hq, hg]
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    (norm_num [I128.min_eq, I128.max_eq, rnd32_eq_div])

/-! ## 6. One reachable input per refusal arm -/

/-- `n = 0` refuses with `ZeroN`. -/
theorem wsyrk_refuse_zeroN_witness :
    ¬ wsyrkDomain wNil none 1#usize 0#usize ∧
      solve_port.syrk.wsyrk wNil none 1#usize 0#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.ZeroN) := by
  refine ⟨?_, syrk_wsyrk_refuse_zeroN wNil none 1#usize 0#usize 1#usize rfl⟩
  intro h
  exact h.1 rfl

/-- `o = 0` refuses with `ZeroO`. -/
theorem wsyrk_refuse_zeroO_witness :
    ¬ wsyrkDomain wNil none 0#usize 1#usize ∧
      solve_port.syrk.wsyrk wNil none 0#usize 1#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.ZeroO) := by
  refine ⟨?_, syrk_wsyrk_refuse_zeroO wNil none 0#usize 1#usize 1#usize w_one_ne_zero rfl⟩
  intro h
  exact h.2.1 rfl

/-- `n = Usize::MAX` overflows `n·n`, refusing with `Overflow`. -/
theorem wsyrk_refuse_mulNN_witness :
    ¬ wsyrkDomain wNil none 1#usize core.num.Usize.MAX ∧
      solve_port.syrk.wsyrk wNil none 1#usize core.num.Usize.MAX 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.Overflow) := by
  refine ⟨?_, syrk_wsyrk_refuse_mulNN wNil none 1#usize core.num.Usize.MAX 1#usize
    w_max_ne_zero w_one_ne_zero w_max_max_none⟩
  intro h
  exact h.2.2.1 w_max_max_none

/-- `o = Usize::MAX`, `n = 2` overflows `o·n`, refusing with `ShapeMismatch` (check order). -/
theorem wsyrk_refuse_mulON_witness :
    ¬ wsyrkDomain wNil none core.num.Usize.MAX 2#usize ∧
      solve_port.syrk.wsyrk wNil none core.num.Usize.MAX 2#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch) := by
  refine ⟨?_, syrk_wsyrk_refuse_mulON wNil none core.num.Usize.MAX 2#usize 1#usize
    w_two_ne_zero w_max_ne_zero w_mul22 w_max_two_none⟩
  intro h
  exact absurd h.2.2.2.2.2.1 (by
    have h1 : 4294967295 ≤ Usize.max := Usize.cMax_bound_concrete.1
    have h2 : core.num.Usize.MAX.val = Usize.max := rfl
    have h3 : 2 ^ 18 = 262144 := by norm_num
    omega)

/-- `Slice.len x ≠ o·n` (`x` empty, `o·n = 4`) refuses with `ShapeMismatch`. -/
theorem wsyrk_refuse_len_x_witness :
    ¬ wsyrkDomain wNil none 2#usize 2#usize ∧
      solve_port.syrk.wsyrk wNil none 2#usize 2#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch) := by
  refine ⟨?_, syrk_wsyrk_refuse_len_x wNil none 2#usize 2#usize 1#usize
    w_two_ne_zero w_two_ne_zero w_mul22 w_mul22 w_nil_len_ne_four⟩
  intro h
  exact w_nil_len_ne_four (h.2.2.2.1 4#usize w_mul22)

/-- A present `w` with `Slice.len w ≠ o` refuses with `ShapeMismatch`. -/
theorem wsyrk_refuse_len_w_witness :
    ¬ wsyrkDomain wOne (some wNil) 1#usize 1#usize ∧
      solve_port.syrk.wsyrk wOne (some wNil) 1#usize 1#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch) := by
  refine ⟨?_, syrk_wsyrk_refuse_len_w wOne wNil 1#usize 1#usize 1#usize
    w_one_ne_zero w_one_ne_zero w_mul11 w_mul11 w_one_len w_nil_len_ne_one⟩
  intro h
  exact w_nil_len_ne_one (h.2.2.2.2.1 wNil rfl)

/-- `o = 2¹⁸ ≥ MAX_O` refuses with `Overflow` once every shape check passes. -/
theorem wsyrk_refuse_maxO_witness :
    ¬ wsyrkDomain wBig none (Slice.len wBig) 1#usize ∧
      solve_port.syrk.wsyrk wBig none (Slice.len wBig) 1#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.Overflow) := by
  refine ⟨?_, syrk_wsyrk_refuse_maxO wBig none (Slice.len wBig) 1#usize 1#usize
    w_one_ne_zero w_big_ne_zero w_mul11 (w_mul_one (Slice.len wBig)) rfl
    (fun ws hws => absurd hws (by simp)) w_big_ge18⟩
  intro h
  exact absurd h.2.2.2.2.2.1 (by rw [w_big_len_val]; norm_num)

/-- An entry outside the strict `±2³⁶` boundary (`i64::MAX`) refuses with `InputBound`. -/
theorem wsyrk_refuse_bound_witness :
    ¬ wsyrkDomain wWide none 1#usize 1#usize ∧
      solve_port.syrk.wsyrk wWide none 1#usize 1#usize 1#usize =
        ok (core.result.Result.Err solve_port.syrk.SyrkError.InputBound) := by
  refine ⟨?_, syrk_wsyrk_refuse_bound wWide none 1#usize 1#usize 1#usize
    w_one_ne_zero w_one_ne_zero w_mul11 w_mul11 w_wide_len
    (fun ws hws => absurd hws (by simp))
    (by rw [show (1#usize).val = 1 from rfl]; norm_num) w_wide_not_bound⟩
  intro h
  exact w_wide_not_bound h.2.2.2.2.2.2

/-! ## 7. The headline refunds fire -/

/-- `word_wsyrk_refuses` fires on the `n = 0` input. -/
theorem wsyrk_refuses_witness :
    ∃ e : solve_port.syrk.SyrkError,
      solve_port.syrk.wsyrk wNil none 1#usize 0#usize 1#usize =
        ok (core.result.Result.Err e) :=
  word_wsyrk_refuses wNil none 1#usize 0#usize 1#usize wsyrk_refuse_zeroN_witness.1

/-- `word_hadamard_refuses` fires on a length mismatch, returning its first argument unwritten. -/
theorem hadamard_refuses_witness :
    solve_port.syrk.hadamard_into wNil wOne =
      ok (core.result.Result.Err solve_port.syrk.SyrkError.ShapeMismatch, wNil) :=
  word_hadamard_refuses wNil wOne (by rw [w_nil_len, w_one_len]; exact w_zero_ne_one)

end apeiron_word
