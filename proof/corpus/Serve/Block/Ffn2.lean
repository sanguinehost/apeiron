/-
  `Serve/Ffn2.lean` — reference semantics for the integer arithmetic kernels of the Q8-activation
  FFN path (`chart`, `src/engine/chart/`). Companion of `Serve/Ffn.lean` (the layout kernels
  `copy_i64_at`/`copy_i8_at`), which is not imported here (it is on no build path of this package).

  Group coverage here:
  * `chart.act_quot_rem` (`chart/mod.rs:219`) — the power-of-two quotient/remainder of the code
    rounding. `pow2` branch = shift and mask; else the checked `u64` division whose `s == 0`
    failure is the previous body's `0`.
  * `chart.ffn.proj_split` (`chart/ffn.rs:883`) — the projection `(rows_per_chunk, task_count)`
    split. Chart branch: `t = min (max actual (2·width)) (max n_out 1)`, `per = ⌈n_out/t⌉`,
    `tasks = ⌈n_out/per⌉` (or `1` when `per = 0`). Rows branch: `align = max row_align 1`,
    `tiles = ⌈n_out/align⌉`, `t = max (min (min chunks (2·width)) tiles) 1`,
    `per = ⌈tiles/t⌉·align`, `tasks = ⌈n_out/per⌉` (or `(align, 0)` when `n_out = 0`). The `Delegate`
    arm panics.
  * `chart.ffn.swiglu_one` (`chart/ffn.rs:246`) — `round(silu(gate)·up / 2^32)` in checked integer
    arithmetic, relative to the extracted silu body `ffn.silu_fixed` (whose real-valued refinement
    is `word_silu_fixed_spec`, `Numeric/Engine/Silu.lean`).
  * `chart.quant_act` (`chart/mod.rs:250`) — the Q8 activation quantiser. SKIPPED (see §4).

  Boundaries skipped here, with reason:
  * `swiglu_one_f32_scalar`, `swiglu_one_carrier`, `swiglu_f32_avx2_8_into`, `f32_to_q32_checked`
    and every other `F32`/`AVX2` body: float carrier and lane intrinsics; the float rounding model
    is a separate refinement.
  * `swiglu_hidden`/`swiglu_cols`/`swiglu_cols_carrier`: the pool dispatch
    (`chart.pool.chunks.for_vec_chunks`) and `mixer.carrier_from_env`.
  * `ActPtr::write_scales`/`write_codes`: `unsafe` raw-pointer writes, no memory model.
  * `I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from`: an unmodeled external (`axiom` in
    `ApeironWord`, the `proof/regen.sh` opaque list). `word_swiglu_one_ok` is stated *relative to
    its result*, the boundary this proof does not reach.
  * `chart.quant_act` (`chart.quant_act_loop0`, its inner absmax loop, `QuantAct.dequantizes`): a
    `Vec::push` loop whose invariant is over the growable model `Vec` and the scale power-of-two
    search. §4 states what is missing.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch, the `Vec::push` allocation loop, or the unmodeled `try_from`
  external. The statements are about the Aeneas-extracted bodies over the model `Slice`, `Vec` and
  word scalars.
-/

import ApeironWord
import Numeric.Engine.Rope

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §1 · `chart.act_quot_rem`: power-of-two quotient and remainder

`src/engine/chart/mod.rs:219`. The Rust body:

```
if pow2 { (m >> shift, m & mask) }
else { (m.checked_div(s).unwrap_or(0), m.checked_rem(s).unwrap_or(0)) }
```

The reference is the quotient/remainder as natural numbers; the `s == 0` failure of `checked_div`
and `checked_rem` is `0`, which is the previous body's value (a zero scale emits code `0` before
`act_quot_rem` is reached). -/

/-- `m / s`, or `0` when `s = 0` (the model's `checked_div … .unwrap_or 0`). -/
def actQuoRef (m s : U64) : Nat := if s.val = 0 then 0 else m.val / s.val

/-- `m % s`, or `0` when `s = 0` (the model's `checked_rem … .unwrap_or 0`). -/
def actRemRef (m s : U64) : Nat := if s.val = 0 then 0 else m.val % s.val

/-- The power-of-two quotient `m >>> shift` as a natural. -/
def actShrRef (m : U64) (shift : U32) : Nat := m.val >>> shift.val

/-- The power-of-two remainder `m &&& mask` as a natural. -/
def actAndRef (m mask : U64) : Nat := m.val &&& mask.val

/-- **`chart.act_quot_rem`, power-of-two branch.** For a shift in the word width, the body returns
the shift and the mask of `m`. The `pow2` flag and the mask are the caller's (`pow2 =
s.is_power_of_two()`, `shift = s.trailing_zeros()`, `mask = s - 1`), so `shift < 64` here.
`proved`. -/
theorem word_act_quot_rem_pow2 (m s : U64) (shift : U32) (mask : U64)
    (h : shift.val < 64) :
    ∃ q r : U64,
      chart.act_quot_rem m s true shift mask = ok (q, r) ∧
      q.val = actShrRef m shift ∧ r.val = actAndRef m mask := by
  obtain ⟨q, hq, hqv⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_spec (x := m) (y := shift) (by simpa using h))
  have hb : chart.act_quot_rem m s true shift mask
      = (do let i ← m >>> shift
            let i1 ← lift (m &&& mask)
            ok (i, i1)) := by
    unfold chart.act_quot_rem
    rfl
  rw [hb, hq]
  simp only [bind_tc_ok, lift]
  refine ⟨q, m &&& mask, rfl, ?_, ?_⟩
  · simp only [actShrRef]; exact hqv.1
  · simp only [actAndRef, UScalar.val_and]

/-- **`chart.act_quot_rem`, division branch.** With `pow2 = false` the body returns the checked
`u64` quotient and remainder, both `0` on the `s = 0` refusal. `proved`. -/
theorem word_act_quot_rem_div (m s : U64) (shift : U32) (mask : U64) :
    ∃ q r : U64,
      chart.act_quot_rem m s false shift mask = ok (q, r) ∧
      q.val = actQuoRef m s ∧ r.val = actRemRef m s := by
  have hb : chart.act_quot_rem m s false shift mask
      = (do let o : Option U64 ← lift (U64.checked_div m s)
            let i ← lift (core.option.Option.unwrap_or o 0#u64)
            let o1 : Option U64 ← lift (U64.checked_rem m s)
            let i1 ← lift (core.option.Option.unwrap_or o1 0#u64)
            ok (i, i1)) := by
    unfold chart.act_quot_rem
    rfl
  rcases hdc : U64.checked_div m s with _ | qd
  · -- division refused: `s = 0`, so the remainder is also `0`.
    have hsv : s.val = 0 := by
      have hs := U64.checked_div_bv_spec m s
      rw [hdc] at hs; exact hs
    rcases hrc : U64.checked_rem m s with _ | rd
    · rw [hb, hdc, hrc]
      simp only [lift, bind_tc_ok, core.option.Option.unwrap_or_none]
      exact ⟨0#u64, 0#u64, rfl, by simp [actQuoRef, hsv], by simp [actRemRef, hsv]⟩
    · have hsne : s.val ≠ 0 := by
        have hr := U64.checked_rem_bv_spec m s
        rw [hrc] at hr; exact hr.1
      exact absurd hsv hsne
  · have hdiv := U64.checked_div_bv_spec m s
    rw [hdc] at hdiv
    obtain ⟨hsne, hqv, _⟩ := hdiv
    rcases hrc : U64.checked_rem m s with _ | rd
    · have hsv : s.val = 0 := by
        have hr := U64.checked_rem_bv_spec m s
        rw [hrc] at hr; exact hr
      exact absurd hsv hsne
    · have hrem := U64.checked_rem_bv_spec m s
      rw [hrc] at hrem
      obtain ⟨_, hrv, _⟩ := hrem
      rw [hb, hdc, hrc]
      simp only [lift, bind_tc_ok, core.option.Option.unwrap_or_some]
      exact ⟨qd, rd, rfl, by simp [actQuoRef, hsne, hqv], by simp [actRemRef, hsne, hrv]⟩

/-- **`chart.act_quot_rem` value law.** The two branches agree with the reference pair. `proved`. -/
theorem word_act_quot_rem_ok (m s : U64) (pow2 : Bool) (shift : U32) (mask : U64)
    (h : pow2 = true → shift.val < 64) :
    ∃ q r : U64,
      chart.act_quot_rem m s pow2 shift mask = ok (q, r) ∧
      (if pow2 then q.val = actShrRef m shift ∧ r.val = actAndRef m mask
        else q.val = actQuoRef m s ∧ r.val = actRemRef m s) := by
  cases pow2 with
  | true =>
    obtain ⟨q, r, hcl, hq, hr⟩ := word_act_quot_rem_pow2 m s shift mask (h rfl)
    exact ⟨q, r, hcl, by simp [hq, hr]⟩
  | false =>
    obtain ⟨q, r, hcl, hq, hr⟩ := word_act_quot_rem_div m s shift mask
    exact ⟨q, r, hcl, by simp [hq, hr]⟩

/-! ### Non-vacuity of §1 -/

/-- Power-of-two branch at `m = 0b1011`, `shift = 2`, `mask = 3`: quotient `2`, remainder `3`. -/
example : actShrRef (11#u64) (2#u32) = 2 ∧ actAndRef (11#u64) (3#u64) = 3 := by
  constructor <;> decide

/-- Division branch: `7 / 2 = 3`, `7 % 2 = 1`; the `s = 0` refusal is `0`. -/
example : actQuoRef (7#u64) (2#u64) = 3 ∧ actRemRef (7#u64) (2#u64) = 1 ∧
    actQuoRef (7#u64) (0#u64) = 0 := by
  refine ⟨by decide, by decide, by decide⟩

/-- The §1 reference pair is inhabited at a concrete power-of-two input. -/
example : ∃ q r : U64,
    chart.act_quot_rem (11#u64) (4#u64) true (2#u32) (3#u64) = ok (q, r) ∧
    q.val = 2 ∧ r.val = 3 := by
  obtain ⟨q, r, hcl, hq, hr⟩ :=
    word_act_quot_rem_pow2 (11#u64) (4#u64) (2#u32) (3#u64) (by decide)
  exact ⟨q, r, hcl, by rw [hq]; decide, by rw [hr]; decide⟩

/-! ## §2 · `chart.ffn.proj_split`: the projection split

`src/engine/chart/ffn.rs:883`. On the chart prep the split is
`t = min (max actual (2·width)) (max n_out 1)`, `per = ⌈n_out / t⌉`, and `tasks = ⌈n_out / per⌉`
(or `1` when `per = 0`); the `Delegate` arm panics. The reference is the same integer expression
over `Nat`. -/

/-- Every model `Usize` value is at most `Usize.max`, the platform word maximum. -/
theorem usz_val_le_max (x : Usize) : x.val ≤ Usize.max := by
  have h := x.hrBounds
  simpa [UScalar.rMax, Usize.rMax, Usize.max, Usize.numBits] using h

/-- `⌈a / b⌉` over `Nat`: `a / b` rounded up, and `0` when `b = 0`. -/
def ceilDivRef (a b : Nat) : Nat := if b = 0 then 0 else a / b + if a % b = 0 then 0 else 1

/-- A ceiling quotient of two nonzero naturals is at least `1`. -/
theorem one_le_ceilDivRef (a b : Nat) (ha : a ≠ 0) (hb : b ≠ 0) : 1 ≤ ceilDivRef a b := by
  rw [ceilDivRef, if_neg hb]
  by_cases hmod : a % b = 0
  · rw [if_pos hmod]
    have hle : b ≤ a := by
      by_contra hlt
      rw [Nat.mod_eq_of_lt (Nat.lt_of_not_le hlt)] at hmod
      exact ha hmod
    have hpos : 0 < a / b := Nat.div_pos hle (Nat.pos_of_ne_zero hb)
    omega
  · rw [if_neg hmod]
    exact Nat.le_add_left 1 _

/-- The extracted `core.num.Usize.div_ceil x y` returns the ceiling quotient as a `Nat` when
`y ≠ 0`. `proved`. -/
theorem usize_div_ceil_ok (x y : Usize) (hy : y.val ≠ 0) :
    ∃ q : Usize, core.num.Usize.div_ceil x y = ok q ∧ q.val = ceilDivRef x.val y.val := by
  obtain ⟨u, hu, huv⟩ := UScalar.div_spec x (y := y) hy
  obtain ⟨v, hv, hvv⟩ := WP.spec_imp_exists (UScalar.rem_spec x (y := y) hy)
  have hu' : UScalar.div x y = ok u := hu
  have hv' : UScalar.rem x y = ok v := hv
  unfold core.num.Usize.div_ceil
  rw [hu', hv']
  simp only [bind_tc_ok]
  by_cases hrv : v.val = 0
  · -- exact division: `div_ceil` returns the quotient, and the reference adds `0`.
    rw [if_pos hrv]
    refine ⟨u, rfl, ?_⟩
    have hmod0 : x.val % y.val = 0 := by rw [← hvv]; exact hrv
    rw [ceilDivRef, if_neg hy, if_pos hmod0, huv]
    simp
  · rw [if_neg hrv]
    have hmodne : x.val % y.val ≠ 0 := by
      intro h; exact hrv (by rw [h] at hvv; exact hvv)
    have hxpos : 0 < x.val := by
      rcases Nat.eq_zero_or_pos x.val with h | h
      · rw [h] at hmodne; simp at hmodne
      · exact h
    have hy1 : 1 < y.val := by
      rcases Nat.lt_or_ge 1 y.val with h | h
      · exact h
      · have hyv : y.val = 1 := by omega
        exact (hmodne (by rw [hyv, Nat.mod_one])).elim
    have hdivlt : x.val / y.val < x.val := Nat.div_lt_self hxpos hy1
    have hmax : u.val + (UScalar.one : Usize).val ≤ Usize.max := by
      have hone : (UScalar.one : Usize).val = 1 := rfl
      have hub : u.val = x.val / y.val := huv
      have hle := usz_val_le_max x
      omega
    obtain ⟨w, hw, hwv⟩ :=
      WP.spec_imp_exists (Usize.add_spec (x := u) (y := (UScalar.one : Usize)) hmax)
    refine ⟨w, hw, ?_⟩
    rw [ceilDivRef, if_neg hy, if_neg hmodne]
    have hone : (UScalar.one : Usize).val = 1 := rfl
    omega

/-- The extracted `core.cmp.Ord.max.default core.cmp.OrdUsize x y` returns the larger word. -/
theorem ordUsize_max_eq (x y : Usize) :
    core.cmp.Ord.max.default core.cmp.OrdUsize x y
      = ok (if y.val < x.val then x else y) := by
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body
  simp [core.cmp.impls.PartialOrdUsize.lt, liftFun2]
  split <;> rfl

/-- The extracted `core.cmp.Ord.min.trait_default core.cmp.OrdUsize x y` returns the smaller word. -/
theorem ordUsize_min_eq (x y : Usize) :
    core.cmp.Ord.min.trait_default core.cmp.OrdUsize x y
      = ok (if y.val < x.val then y else x) := by
  unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body
  simp [core.cmp.impls.PartialOrdUsize.lt, liftFun2]
  split <;> rfl

/-- `max.default OrdUsize` returns `max x.val y.val` as a natural. -/
theorem ordUsize_max_spec (x y : Usize) :
    ∃ z : Usize, core.cmp.Ord.max.default core.cmp.OrdUsize x y = ok z ∧
      z.val = max x.val y.val := by
  refine ⟨if y.val < x.val then x else y, ordUsize_max_eq x y, ?_⟩
  by_cases h : y.val < x.val
  · rw [if_pos h]; exact (Nat.max_eq_left (le_of_lt h)).symm
  · rw [if_neg h]; exact (Nat.max_eq_right (le_of_not_gt h)).symm

/-- `min.trait_default OrdUsize` returns `min x.val y.val` as a natural. -/
theorem ordUsize_min_spec (x y : Usize) :
    ∃ z : Usize, core.cmp.Ord.min.trait_default core.cmp.OrdUsize x y = ok z ∧
      z.val = min x.val y.val := by
  refine ⟨if y.val < x.val then y else x, ordUsize_min_eq x y, ?_⟩
  by_cases h : y.val < x.val
  · rw [if_pos h]; exact (Nat.min_eq_right (le_of_lt h)).symm
  · rw [if_neg h]; exact (Nat.min_eq_left (le_of_not_gt h)).symm

/-- The chart-branch block count `t = min (max actual (2·width)) (max n_out 1)`. -/
def chartTRef (actual width nOut : Nat) : Nat :=
  min (max actual (2 * width)) (max nOut 1)

/-- The chart-branch split `(per, tasks)` over the natural reference. -/
def projSplitChartRef (nOut actual width : Nat) : Nat × Nat :=
  if ceilDivRef nOut (chartTRef actual width nOut) = 0
  then (ceilDivRef nOut (chartTRef actual width nOut), 1)
  else (ceilDivRef nOut (chartTRef actual width nOut),
        ceilDivRef nOut (ceilDivRef nOut (chartTRef actual width nOut)))

/-- The first component of the chart-branch reference is the ceiling quotient. -/
theorem projSplitChartRef_fst (nOut actual width : Nat) :
    (projSplitChartRef nOut actual width).1 = ceilDivRef nOut (chartTRef actual width nOut) := by
  unfold projSplitChartRef
  split <;> rfl

/-- The second component of the chart-branch reference is `1` on a zero quotient, the inner ceiling
quotient otherwise. -/
theorem projSplitChartRef_snd (nOut actual width : Nat) :
    (projSplitChartRef nOut actual width).2
      = if ceilDivRef nOut (chartTRef actual width nOut) = 0 then 1
        else ceilDivRef nOut (ceilDivRef nOut (chartTRef actual width nOut)) := by
  unfold projSplitChartRef
  split <;> rfl

/-- The Rows-branch alignment `align = max row_align 1`. -/
def projRowsAlign (rowAlign : Nat) : Nat := max rowAlign 1

/-- The Rows-branch tile count `tiles = ⌈n_out / align⌉`. -/
def projRowsTiles (nOut rowAlign : Nat) : Nat := ceilDivRef nOut (projRowsAlign rowAlign)

/-- The Rows-branch block count `t = max (min (min chunks (2·width)) tiles) 1`. -/
def projRowsT (nOut rowAlign chunks width : Nat) : Nat :=
  max (min (min chunks (2 * width)) (projRowsTiles nOut rowAlign)) 1

/-- The Rows-branch per-chunk row count `per = ⌈tiles / t⌉ · align`. -/
def projRowsPer (nOut rowAlign chunks width : Nat) : Nat :=
  ceilDivRef (projRowsTiles nOut rowAlign) (projRowsT nOut rowAlign chunks width) *
    projRowsAlign rowAlign

/-- The Rows-branch split `(per, tasks)`; on `n_out = 0` the body returns `(align, 0)`. -/
def projSplitRowsRef (nOut rowAlign chunks width : Nat) : Nat × Nat :=
  if nOut = 0 then (projRowsAlign rowAlign, 0)
  else (projRowsPer nOut rowAlign chunks width,
        ceilDivRef nOut (projRowsPer nOut rowAlign chunks width))

/-- The first component of the Rows-branch reference. -/
theorem projSplitRowsRef_fst (nOut rowAlign chunks width : Nat) :
    (projSplitRowsRef nOut rowAlign chunks width).1
      = if nOut = 0 then projRowsAlign rowAlign
        else projRowsPer nOut rowAlign chunks width := by
  by_cases h : nOut = 0 <;> simp [projSplitRowsRef, h]

/-- The second component of the Rows-branch reference. -/
theorem projSplitRowsRef_snd (nOut rowAlign chunks width : Nat) :
    (projSplitRowsRef nOut rowAlign chunks width).2
      = if nOut = 0 then 0
        else ceilDivRef nOut (projRowsPer nOut rowAlign chunks width) := by
  by_cases h : nOut = 0 <;> simp [projSplitRowsRef, h]

/-- **`chart.ffn.proj_split`, Chart branch.** Under a `width` whose `2·width` fits the word, and a
positive block count `t`, the body returns the integer split. `proved`. -/
theorem word_proj_split_chart_ok (pp : chart.ffn.ProjPrep) (p : chart.ffn.ProjSpec)
    (chunks width : Usize)
    (hW : 2 * width.val ≤ Usize.max)
    (hT : 0 < chartTRef pp.actual.val width.val p.n_out.val) :
    ∃ per tasks : Usize,
      chart.ffn.proj_split (chart.ffn.GroupPrep.Chart pp) p chunks width = ok (per, tasks) ∧
      per.val = (projSplitChartRef p.n_out.val pp.actual.val width.val).1 ∧
      tasks.val = (projSplitChartRef p.n_out.val pp.actual.val width.val).2 := by
  unfold chart.ffn.proj_split
  dsimp only
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := width) hW)
  rw [hi]; simp only [bind_tc_ok]
  obtain ⟨i1, hi1, hi1v⟩ := ordUsize_max_spec pp.actual i
  rw [hi1]; simp only [bind_tc_ok]
  obtain ⟨i2, hi2, hi2v⟩ := ordUsize_max_spec p.n_out 1#usize
  rw [hi2]; simp only [bind_tc_ok]
  obtain ⟨t, ht, htv⟩ := ordUsize_min_spec i1 i2
  rw [ht]; simp only [bind_tc_ok]
  have h1v : ((1#usize : Usize) : Nat) = 1 := rfl
  have h2v : ((2#usize : Usize) : Nat) = 2 := rfl
  have htval : t.val = chartTRef pp.actual.val width.val p.n_out.val := by
    rw [htv, hi1v, hi2v, hiv, h1v, h2v]
    simp only [chartTRef]
  have htne : t.val ≠ 0 := by rw [htval]; exact Nat.ne_of_gt hT
  obtain ⟨per, hper, hperv⟩ := usize_div_ceil_ok p.n_out t htne
  rw [hper]; simp only [bind_tc_ok]
  have hperv' : per.val = ceilDivRef p.n_out.val (chartTRef pp.actual.val width.val p.n_out.val) := by
    rw [hperv, htval]
  by_cases hper0 : per = 0#usize
  · rw [if_pos hper0]
    refine ⟨per, 1#usize, rfl, ?_, ?_⟩
    · rw [projSplitChartRef_fst]; exact hperv'
    · rw [projSplitChartRef_snd]
      have hzero : ceilDivRef p.n_out.val (chartTRef pp.actual.val width.val p.n_out.val) = 0 := by
        rw [← hperv']
        exact congrArg UScalar.val hper0
      rw [if_pos hzero]
      rfl
  · rw [if_neg hper0]
    have hperne : per.val ≠ 0 := by
      intro hz; exact hper0 (by apply UScalar.eq_of_val_eq; rw [hz]; rfl)
    obtain ⟨tasks, htasks, htasksv⟩ := usize_div_ceil_ok p.n_out per hperne
    rw [htasks]; simp only [bind_tc_ok]
    refine ⟨per, tasks, rfl, ?_, ?_⟩
    · rw [projSplitChartRef_fst]; exact hperv'
    · rw [projSplitChartRef_snd]
      have hne : ceilDivRef p.n_out.val (chartTRef pp.actual.val width.val p.n_out.val) ≠ 0 := by
        rw [← hperv']; exact hperne
      rw [if_neg hne, htasksv, hperv']

/-- **`chart.ffn.proj_split`, Delegate arm.** The delegate prep is a panic: no split is defined for
it. `proved`. -/
theorem word_proj_split_delegate (p : chart.ffn.ProjSpec) (chunks width : Usize) :
    chart.ffn.proj_split chart.ffn.GroupPrep.Delegate p chunks width = fail Error.panic := by
  unfold chart.ffn.proj_split
  rfl

/-! ### Non-vacuity of §2 -/

/-- The chart split at `actual = 4`, `width = 2`, `n_out = 10`: `t = 4`, `per = ⌈10/4⌉ = 3`,
`tasks = ⌈10/3⌉ = 4`. -/
example :
    chart.ffn.proj_split
      (chart.ffn.GroupPrep.Chart
        { actual := 4#usize, row_bytes := 0#usize, plan := chart.Q8ActPlan.Reference })
      { n_out := 10#usize, row_align := 2#usize,
        source := chart.ffn.ProjSource.Rows 0#usize }
      1#usize 2#usize = ok (3#usize, 4#usize) := by
  obtain ⟨per, tasks, h, hp, ht⟩ :=
    word_proj_split_chart_ok
      ({ actual := 4#usize, row_bytes := 0#usize, plan := chart.Q8ActPlan.Reference } :
        chart.ffn.ProjPrep)
      ({ n_out := 10#usize, row_align := 2#usize,
         source := chart.ffn.ProjSource.Rows 0#usize } : chart.ffn.ProjSpec)
      1#usize 2#usize (by exact usz_val_le_max (4#usize)) (by decide)
  rw [h]
  have hp3 : per = 3#usize := by
    apply UScalar.eq_of_val_eq
    rw [hp]
    decide
  have ht4 : tasks = 4#usize := by
    apply UScalar.eq_of_val_eq
    rw [ht]
    decide
  rw [hp3, ht4]

/-- **`chart.ffn.proj_split`, Rows branch.** `align = max row_align 1`; when `n_out = 0` the split is
`(align, 0)`, otherwise the tile-aligned split of the Rows arm. The product `per = ⌈tiles/t⌉·align`
must fit the word (`hPer`), otherwise the modeled `usize` multiply panics. `proved`. -/
theorem word_proj_split_rows_ok (p : chart.ffn.ProjSpec) (chunks width : Usize)
    (hW : 2 * width.val ≤ Usize.max)
    (hPer : projRowsPer p.n_out.val p.row_align.val chunks.val width.val ≤ Usize.max) :
    ∃ per tasks : Usize,
      chart.ffn.proj_split chart.ffn.GroupPrep.Rows p chunks width = ok (per, tasks) ∧
      per.val = (projSplitRowsRef p.n_out.val p.row_align.val chunks.val width.val).1 ∧
      tasks.val = (projSplitRowsRef p.n_out.val p.row_align.val chunks.val width.val).2 := by
  unfold chart.ffn.proj_split
  dsimp only
  obtain ⟨align, halign, halignv⟩ := ordUsize_max_spec p.row_align 1#usize
  rw [halign]; simp only [bind_tc_ok]
  have h1v : (1#usize : Usize).val = 1 := rfl
  have h2v : (2#usize : Usize).val = 2 := rfl
  by_cases hn0 : p.n_out = 0#usize
  · rw [if_pos hn0]
    refine ⟨align, 0#usize, rfl, ?_, ?_⟩
    · have hn0v : p.n_out.val = 0 := by simpa using congrArg UScalar.val hn0
      rw [projSplitRowsRef_fst, if_pos hn0v, projRowsAlign, halignv, h1v]
    · have hn0v : p.n_out.val = 0 := by simpa using congrArg UScalar.val hn0
      rw [projSplitRowsRef_snd, if_pos hn0v]
      rfl
  · rw [if_neg hn0]
    have hn0v : p.n_out.val ≠ 0 := by
      intro hz; exact hn0 (by apply UScalar.eq_of_val_eq; rw [hz]; rfl)
    have h1le : (1 : Nat) ≤ align.val := by
      rw [halignv, h1v]; exact Nat.le_max_right _ _
    have halign_ne : align.val ≠ 0 := Nat.ne_of_gt (lt_of_lt_of_le Nat.zero_lt_one h1le)
    obtain ⟨tiles, htiles, htilesv⟩ := usize_div_ceil_ok p.n_out align halign_ne
    rw [htiles]; simp only [bind_tc_ok]
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := width) hW)
    rw [hi]; simp only [bind_tc_ok]
    obtain ⟨i1, hi1, hi1v⟩ := ordUsize_min_spec chunks i
    rw [hi1]; simp only [bind_tc_ok]
    obtain ⟨i2, hi2, hi2v⟩ := ordUsize_min_spec i1 tiles
    rw [hi2]; simp only [bind_tc_ok]
    obtain ⟨t, ht, htv⟩ := ordUsize_max_spec i2 1#usize
    rw [ht]; simp only [bind_tc_ok]
    have hone_le_t : (1 : Nat) ≤ t.val := by
      rw [htv, h1v]; exact Nat.le_max_right _ _
    have ht_ne : t.val ≠ 0 := Nat.ne_of_gt (lt_of_lt_of_le Nat.zero_lt_one hone_le_t)
    have htiles_ne : tiles.val ≠ 0 := by
      rw [htilesv]
      exact Nat.ne_of_gt (lt_of_lt_of_le Nat.zero_lt_one
        (one_le_ceilDivRef p.n_out.val align.val hn0v halign_ne))
    obtain ⟨i3, hi3, hi3v⟩ := usize_div_ceil_ok tiles t ht_ne
    rw [hi3]; simp only [bind_tc_ok]
    have hi3_ne : i3.val ≠ 0 := by
      rw [hi3v]
      exact Nat.ne_of_gt (lt_of_lt_of_le Nat.zero_lt_one
        (one_le_ceilDivRef tiles.val t.val htiles_ne ht_ne))
    have hbound : i3.val * align.val ≤ Usize.max := by
      rw [hi3v, htv, hi2v, hi1v, hiv, htilesv, halignv, h1v, h2v]
      exact hPer
    obtain ⟨per, hper, hperv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := i3) (y := align) hbound)
    rw [hper]; simp only [bind_tc_ok]
    have hper_ne : per.val ≠ 0 := by
      rw [hperv]
      exact Nat.ne_of_gt (Nat.mul_pos (Nat.pos_of_ne_zero hi3_ne) (Nat.pos_of_ne_zero halign_ne))
    obtain ⟨tasks, htasks, htasksv⟩ := usize_div_ceil_ok p.n_out per hper_ne
    rw [htasks]; simp only [bind_tc_ok]
    have hperv' : per.val = projRowsPer p.n_out.val p.row_align.val chunks.val width.val := by
      rw [hperv, hi3v, htv, hi2v, hi1v, hiv, htilesv, halignv, h1v, h2v]
      rfl
    refine ⟨per, tasks, rfl, ?_, ?_⟩
    · rw [projSplitRowsRef_fst, if_neg hn0v]; exact hperv'
    · rw [projSplitRowsRef_snd, if_neg hn0v, htasksv, hperv']

/-- The Rows split at `n_out = 10`, `row_align = 4`, `chunks = 8`, `width = 2`: `align = 4`,
`tiles = ⌈10/4⌉ = 3`, `t = min (min 8 4) 3 = 3`, `per = ⌈3/3⌉·4 = 4`, `tasks = ⌈10/4⌉ = 3`. -/
example :
    chart.ffn.proj_split chart.ffn.GroupPrep.Rows
      { n_out := 10#usize, row_align := 4#usize,
        source := chart.ffn.ProjSource.Rows 0#usize }
      8#usize 2#usize = ok (4#usize, 3#usize) := by
  have hPer : projRowsPer 10 4 8 2 ≤ Usize.max := by
    have hv : projRowsPer 10 4 8 2 = 4 := by decide
    rw [hv]; exact usz_val_le_max (4#usize)
  obtain ⟨per, tasks, h, hp, ht⟩ :=
    word_proj_split_rows_ok
      { n_out := 10#usize, row_align := 4#usize,
        source := chart.ffn.ProjSource.Rows 0#usize }
      8#usize 2#usize (by exact usz_val_le_max (4#usize)) hPer
  rw [h]
  have hp4 : per = 4#usize := by
    apply UScalar.eq_of_val_eq
    rw [hp]
    decide
  have ht3 : tasks = 3#usize := by
    apply UScalar.eq_of_val_eq
    rw [ht]
    decide
  rw [hp4, ht3]

/-! ## §3 · `chart.ffn.swiglu_one`: `round(silu(gate)·up / 2^32)`

`src/engine/chart/ffn.rs:246`. The body is `silu_fixed(gate)?`, then the checked `i128` product
`i128::from(s) * i128::from(up)`, then `value::round_shift_32`, then `i64::try_from` mapped to
`FfnError::Overflow`. The reference value is `roundShift32 (s · up)` (the `round_shift_32` of
`Numeric/Engine/Rope.lean`). The `i64::try_from` step is the unmodeled external: the headline is
stated relative to its result. -/

/-- The reference value of the element: half-biased rounding of the product `s · up` (with
`roundShift32 v = (v + 2^31) >>> 32`). -/
def swigluRef (s up : ℤ) : ℤ := roundShift32 (s * up)

/-- **The value that reaches `i64::try_from`.** Under the silu outcome `s`, with the `i128` product
in range and the round-shift argument in range, the checked product is `some p` with
`p.val = s·up` and `round_shift_32 p` is the reference `swigluRef s.val up.val`. `proved`. -/
theorem word_swiglu_one_round (gate up : I64) (s : I64)
    (hs : ffn.silu_fixed gate = ok (core.result.Result.Ok s))
    (hprodlo : I128.min ≤ s.val * up.val)
    (hprodhi : s.val * up.val ≤ I128.max)
    (hrlo : I128.min ≤ s.val * up.val + 2147483648)
    (hrhi : s.val * up.val + 2147483648 ≤ I128.max) :
    ∃ p i2 : I128,
      I128.checked_mul (core.convert.num.FromI128I64.from s)
          (core.convert.num.FromI128I64.from up) = some p ∧
      value.round_shift_32 p = ok i2 ∧
      i2.val = swigluRef s.val up.val := by
  rcases hcm : core.num.checked_mul_IScalar (core.convert.num.FromI128I64.from s)
      (core.convert.num.FromI128I64.from up) with _ | p
  · have hspec := I128.checked_mul_bv_spec (x := core.convert.num.FromI128I64.from s)
      (y := core.convert.num.FromI128I64.from up)
    rw [hcm] at hspec
    apply absurd _ hspec
    rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
    exact ⟨hprodlo, hprodhi⟩
  · have hspec := I128.checked_mul_bv_spec (x := core.convert.num.FromI128I64.from s)
      (y := core.convert.num.FromI128I64.from up)
    rw [hcm] at hspec
    obtain ⟨_, _, hv, _⟩ := hspec
    rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq] at hv
    obtain ⟨i2, hi2, hi2v⟩ :=
      WP.spec_imp_exists (value_round_shift_32_spec p (by rw [hv]; exact hrlo)
        (by rw [hv]; exact hrhi))
    refine ⟨p, i2, hcm, hi2, ?_⟩
    rw [hi2v, hv, swigluRef]

/-- **`chart.ffn.swiglu_one` success.** Given the silu outcome `s`, the in-range `i128` product and
round-shift argument, and the outcome of the unmodeled `i64::try_from` (an `Ok`), the body returns
`ok (Ok out)`. The `try_from` hypothesis is the trust boundary this proof does not cross.
`proved` relative to it. -/
theorem word_swiglu_one_ok (gate up : I64) (s : I64)
    (hs : ffn.silu_fixed gate = ok (core.result.Result.Ok s))
    (hprodlo : I128.min ≤ s.val * up.val)
    (hprodhi : s.val * up.val ≤ I128.max)
    (hrlo : I128.min ≤ s.val * up.val + 2147483648)
    (hrhi : s.val * up.val + 2147483648 ≤ I128.max)
    (htry : ∀ i2 : I128, i2.val = swigluRef s.val up.val →
      ∃ out : I64,
        I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from i2 = ok (core.result.Result.Ok out)) :
    ∃ out : I64, chart.ffn.swiglu_one gate up = ok (core.result.Result.Ok out) := by
  unfold chart.ffn.swiglu_one
  rw [hs]
  simp only [bind_tc_ok]
  have hb1 : core.result.Result.Insts.CoreOpsTry.branch (T := I64) (E := ffn.FfnError)
        (core.result.Result.Ok s)
      = ok (core.ops.control_flow.ControlFlow.Continue
              (B := core.result.Result core.convert.Infallible ffn.FfnError) s) := rfl
  rw [hb1]
  simp only [bind_tc_ok, lift]
  obtain ⟨p, i2, hp, hi2, hi2v⟩ :=
    word_swiglu_one_round gate up s hs hprodlo hprodhi hrlo hrhi
  rw [hp]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  have hb2 : core.result.Result.Insts.CoreOpsTry.branch (T := I128) (E := ffn.FfnError)
        (core.result.Result.Ok p)
      = ok (core.ops.control_flow.ControlFlow.Continue
              (B := core.result.Result core.convert.Infallible ffn.FfnError) p) := rfl
  rw [hb2]
  simp only [bind_tc_ok]
  rw [hi2]
  simp only [bind_tc_ok]
  obtain ⟨out, hout⟩ := htry i2 hi2v
  rw [hout]
  simp only [bind_tc_ok, core.result.Result.map_err_ok]
  exact ⟨out, rfl⟩

/-- **`chart.ffn.swiglu_one` refusal: silu error.** When `ffn.silu_fixed` returns `Err e`, the body
returns `ok (Err e)`; the error propagates through the `?`/`from_residual` path. `proved`. -/
theorem word_swiglu_one_silu_err (gate up : I64) (e : ffn.FfnError)
    (h : ffn.silu_fixed gate = ok (core.result.Result.Err e)) :
    chart.ffn.swiglu_one gate up = ok (core.result.Result.Err e) := by
  unfold chart.ffn.swiglu_one
  rw [h]
  simp only [bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from]

/-- **`chart.ffn.swiglu_one` refusal: product overflow.** When the checked `i128` product refused,
the body returns `ok (Err FfnError.Overflow)` from the `ok_or`/`from_residual` path. `proved`. -/
theorem word_swiglu_one_mul_err (gate up : I64) (s : I64)
    (hs : ffn.silu_fixed gate = ok (core.result.Result.Ok s))
    (hov : I128.checked_mul (core.convert.num.FromI128I64.from s)
        (core.convert.num.FromI128I64.from up) = none) :
    chart.ffn.swiglu_one gate up = ok (core.result.Result.Err ffn.FfnError.Overflow) := by
  unfold chart.ffn.swiglu_one
  rw [hs]
  simp only [bind_tc_ok]
  have hb1 : core.result.Result.Insts.CoreOpsTry.branch (T := I64) (E := ffn.FfnError)
        (core.result.Result.Ok s)
      = ok (core.ops.control_flow.ControlFlow.Continue
              (B := core.result.Result core.convert.Infallible ffn.FfnError) s) := rfl
  rw [hb1]
  simp only [bind_tc_ok, lift, hov, core.option.Option.ok_or_none,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame, core.convert.FromSame.from]

/-! ### Non-vacuity of §3 -/

/-- The reference rounds `s·up = 2^32 + 2^31` to `1` from `swigluRef` and the shift identity. -/
example : swigluRef (2147483649 : ℤ) 1 = 1 := by
  unfold swigluRef roundShift32; decide

/-- The product bounds of §3 are jointly satisfiable at `s = up = 0`. -/
example : I128.min ≤ (0 : ℤ) * 0 ∧ (0 : ℤ) * 0 ≤ I128.max ∧
    I128.min ≤ (0 : ℤ) * 0 + 2147483648 ∧ (0 : ℤ) * 0 + 2147483648 ≤ I128.max := by
  rw [I128.min_eq, I128.max_eq]; norm_num

/-! ## §4 · `chart.quant_act`: NOT PROVED

`chart.quant_act` (`chart/mod.rs:250`) is a `Vec`-pushing block loop
(`chart.quant_act_loop0`, `chart.quant_act_loop0_loop0` the absmax inner loop, and
`chart.QuantAct.dequantizes`) whose positive refinement needs an invariant over the growable model
`Vec Std.I64` / `Vec Std.I8` and the power-of-two scale search `2^⌈log₂ ⌈absmax/127⌉⌉`. The
integer sub-step it consumes, `chart.act_quot_rem`, is proved above (§1). The residual obligation
is the loop invariant and the `QuantAct.dequantizes` refusal correspondence; it is not proved here.
-/

/-! ## §5 · Axiom cones -/

#print axioms word_act_quot_rem_ok
#print axioms word_proj_split_chart_ok
#print axioms word_proj_split_rows_ok
#print axioms word_proj_split_delegate
#print axioms word_swiglu_one_ok
#print axioms word_swiglu_one_round
#print axioms word_swiglu_one_silu_err
#print axioms word_swiglu_one_mul_err

end apeiron_word
