import Serve.Kern.Matmat3

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-!
# Serving kernel group `kern.matmat` (part 4): the `fast_bound_ok` driver

`src/engine/kern/matmat.rs:451-470` (`fast_bound_ok`) folds `|q.scales|` to the running maximum
(refined in `Serve/Matmat3.lean`) and then runs two saturating bound tests on the `i128` scale
`top` and the column count `n_in`.

Reference: `matmat4_fastBoundRef` is the two-step decision — refuse (`false`) when
`smax · 32 · top · 127` exceeds `i64::MAX`, else test whether the `f16::MAX`-scaled bound
`smax · 127 · top · f16Max · n_in` fits `i128`. `matmat4_satMul` is `i128::saturating_mul`.

Trust boundary: rustc/LLVM codegen; the `R[x]/(x²−σ)` word kernels, the `F32`/AVX2 layers and the
`row_accs`/`matvec_q8act` accumulation are not covered here.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`. `decide` is not used on an extracted body.
-/

/-! ## Reference: the `fast_bound_ok` bound tests -/

/-- The fixed-point `f16::MAX` the kernel uses: `65504 · 2^32`. -/
def matmat4_f16MaxFixed : Int := 65504 * 2 ^ 32

/-- The `i128` saturating product on `Int`: the product clamped to `[i128::MIN, i128::MAX]`. -/
def matmat4_satMul (x y : Int) : Int := max I128.min (min I128.max (x * y))

/-- The pure reference of `kern.matmat.fast_bound_ok`: over the activation scale `top`, the running
magnitude maximum `smax` and the column count `n`. Refuse (`false`) when `smax · 32 · top · 127`
exceeds `i64::MAX`; otherwise report whether the `f16::MAX`-scaled bound fits `i128`. -/
def matmat4_fastBoundRef (n : Nat) (top smax : Int) : Bool :=
  if matmat4_satMul smax (32 * top * 127) > 9223372036854775807 then
    false
  else
    decide
      (matmat4_satMul
          (matmat4_satMul (matmat4_satMul (matmat4_satMul (n : Int) matmat4_f16MaxFixed) top) 127)
          smax <
        170141183460469231731687303715884105727)

/-! ## Value lemmas for the constants -/

/-- The extracted `core.num.I64.MAX` is `i64::MAX`. -/
theorem matmat4_I64_MAX_val : core.num.I64.MAX.val = I64.max := by
  simp [core.num.I64.MAX, IScalar.ofInt, I64.rMax, I64.max, I64.numBits]

/-- The extracted `core.num.I128.MAX` is `i128::MAX`. -/
theorem matmat4_I128_MAX_val : core.num.I128.MAX.val = I128.max := by
  simp [core.num.I128.MAX, IScalar.ofInt, I128.rMax, I128.max, I128.numBits]

/-- `i64::MAX` as a literal. -/
theorem matmat4_I64_max_lit : I64.max = 9223372036854775807 := I64.max_eq

/-- `i128::MAX` as a literal. -/
theorem matmat4_I128_max_lit : I128.max = 170141183460469231731687303715884105727 := I128.max_eq

/-- `65504 · 2^32`, the value of `F16_MAX_FIXED`. -/
theorem matmat4_f16Max_lit : (65504 : Int) * 2 ^ 32 = 281337537757184 := by norm_num

/-- The `i128` value of `FromI128I64.from I64.MAX` is `i64::MAX`. -/
theorem matmat4_from_I64MAX_val :
    (core.convert.num.FromI128I64.from core.num.I64.MAX).val = 9223372036854775807 := by
  rw [core.convert.num.FromI128I64.from_val_eq, matmat4_I64_MAX_val, matmat4_I64_max_lit]

/-! ## The extracted `i128::saturating_mul` -/

/-- The extracted `i128::saturating_mul` returns the `Int`-clamped product. -/
@[step]
theorem matmat4_saturating_mul_spec (x y : I128) :
    core.num.I128.saturating_mul x y ⦃ r => r.val = matmat4_satMul x.val y.val ⦄ := by
  unfold core.num.I128.saturating_mul matmat4_satMul
  simp only [WP.spec_ok]
  have hb : -2 ^ (IScalarTy.I128.numBits - 1) ≤ max I128.min (min I128.max (x.val * y.val)) ∧
      max I128.min (min I128.max (x.val * y.val)) < 2 ^ (IScalarTy.I128.numBits - 1) := by
    rw [IScalarTy.I128_numBits_eq, I128.max_eq, I128.min_eq]
    omega
  rw [show (⟨BitVec.ofInt 128 (max I128.min (min I128.max (x.val * y.val)))⟩ : I128) =
      I128.ofIntCore (max I128.min (min I128.max (x.val * y.val))) hb from rfl]
  exact I128.ofInt_val_eq hb

/-! ## The `F16_MAX_FIXED` value -/

/-- The extracted `kern.matmat.F16_MAX_FIXED` returns `65504 · 2^32`. -/
@[step]
theorem matmat4_F16_MAX_FIXED_spec :
    kern.matmat.F16_MAX_FIXED ⦃ r => r.val = matmat4_f16MaxFixed ⦄ := by
  have h : kern.matmat.F16_MAX_FIXED = ok (I128.ofInt (65504 * 2 ^ 32)) := by
    unfold kern.matmat.F16_MAX_FIXED
    rfl
  rw [h]
  simp only [WP.spec_ok, matmat4_f16MaxFixed]
  apply I128.ofInt_val_eq

/-- The `UScalar.hcast .I128` of a `Usize` below `2^127` preserves the value. -/
theorem matmat4_hcast_val (n_in : Usize) (h : n_in.val < 2 ^ 127) :
    (UScalar.hcast IScalarTy.I128 n_in).val = (n_in.val : Int) := by
  have hs := UScalar.hcast_inBounds_spec (src_ty := UScalarTy.Usize) IScalarTy.I128 n_in ?_
  · obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists hs
    have hy' : UScalar.hcast IScalarTy.I128 n_in = y := by simpa [lift] using hy
    rw [hy']
    exact hyv
  · rw [IScalar.max_IScalarTy_I128_eq, I128.max_eq]
    omega

/-! ## The `fast_bound_ok` value spec -/

/-- The driver returns `matmat4_fastBoundRef` of the running magnitude maximum when both checked
`i128` products (`32 · top` and `(32 · top) · 127`) are in range and `n_in` fits `i128`. -/
theorem word_matmat4_fast_bound_ok_ok (n_in : Usize) (top : I128) (q : chart.QuantAct)
    (hsz : q.scales.val.length < Usize.max)
    (hn : n_in.val < 2 ^ 127)
    (hlo : -2 ^ 127 ≤ 32 * top.val) (hhi : 32 * top.val < 2 ^ 127)
    (hlo1 : -2 ^ 127 ≤ 32 * top.val * 127) (hhi1 : 32 * top.val * 127 < 2 ^ 127) :
    kern.matmat.fast_bound_ok n_in top q ⦃ r =>
      r = matmat4_fastBoundRef n_in.val top.val (matmat3_smaxRef q) ⦄ := by
  unfold kern.matmat.fast_bound_ok
  obtain ⟨s0, hs0, hs0v⟩ := WP.spec_imp_exists (word_matmat3_fast_bound_ok_loop_ok q hsz)
  rw [hs0]
  simp only [bind_tc_ok]
  step*
  simp only [lift, bind_tc_ok]
  have hi2v : i2.val = matmat4_satMul (matmat3_smaxRef q) (32 * top.val * 127) := by
    rw [i2_post, i1_post, i_post, hs0v]
  by_cases hc : i2 > core.convert.num.FromI128I64.from core.num.I64.MAX
  · rw [if_pos hc]
    rw [WP.spec_ok]
    unfold matmat4_fastBoundRef
    have hcv : (9223372036854775807 : Int) < matmat4_satMul (matmat3_smaxRef q) (32 * top.val * 127) := by
      have h := (IScalar.lt_equiv _ _).mp hc
      rw [matmat4_from_I64MAX_val, hi2v] at h
      exact h
    rw [if_pos hcv]
  · rw [if_neg hc]
    unfold matmat4_fastBoundRef
    have hcv : ¬ ((9223372036854775807 : Int) < matmat4_satMul (matmat3_smaxRef q) (32 * top.val * 127)) := by
      intro h
      apply hc
      exact (IScalar.lt_equiv _ _).mpr (by simpa only [matmat4_from_I64MAX_val, hi2v] using h)
    rw [if_neg hcv]
    step*
    have hi5v : i5.val = matmat4_f16MaxFixed := by assumption
    simp only [IScalar.lt_equiv]
    rw [bound_post, i8_post, i7_post, i6_post, hi5v, matmat4_hcast_val n_in hn, hs0v,
        matmat4_I128_MAX_val, matmat4_I128_max_lit]

/-! ## Refusals: the checked products overflow -/

/-- A checked `i128` product outside `[i128::MIN, i128::MAX]` refuses with `integerOverflow`. -/
theorem matmat4_I128_mul_overflow (x y : I128)
    (h : ¬ (I128.min ≤ x.val * y.val ∧ x.val * y.val ≤ I128.max)) :
    x * y = fail .integerOverflow := by
  change IScalar.mul x y = fail .integerOverflow
  unfold IScalar.mul IScalar.tryMk IScalar.tryMkOpt
  rw [dif_neg (by
    rw [IScalar.check_bounds_eq_inBounds]; intro hb
    rw [IScalar.inBounds] at hb
    rw [show (2 : Int) ^ (IScalarTy.I128.numBits - 1) = 170141183460469231731687303715884105728 from by
      rw [IScalarTy.I128_numBits_eq]; norm_num] at hb
    apply h
    rw [I128.min_eq, I128.max_eq]
    refine ⟨hb.1, ?_⟩
    omega)]
  rfl

/-- The driver refuses when the first checked product `32 · top` overflows `i128`. -/
theorem word_matmat4_fast_bound_ok_refuses (n_in : Usize) (top : I128) (q : chart.QuantAct)
    (hsz : q.scales.val.length < Usize.max)
    (h : ¬ (I128.min ≤ 32 * top.val ∧ 32 * top.val ≤ I128.max)) :
    kern.matmat.fast_bound_ok n_in top q = fail .integerOverflow := by
  unfold kern.matmat.fast_bound_ok
  obtain ⟨s0, hs0, _⟩ := WP.spec_imp_exists (word_matmat3_fast_bound_ok_loop_ok q hsz)
  rw [hs0]
  simp only [bind_tc_ok]
  rw [matmat4_I128_mul_overflow 32#i128 top (by
    rw [show (32#i128).val = 32 from by norm_num]
    exact h)]
  simp only [bind_tc_fail]

/-- The driver refuses when the second checked product `(32 · top) · 127` overflows `i128`. -/
theorem word_matmat4_fast_bound_ok_refuses_scale (n_in : Usize) (top : I128) (q : chart.QuantAct)
    (hsz : q.scales.val.length < Usize.max)
    (h1 : I128.min ≤ 32 * top.val ∧ 32 * top.val ≤ I128.max)
    (h2 : ¬ (I128.min ≤ 32 * top.val * 127 ∧ 32 * top.val * 127 ≤ I128.max)) :
    kern.matmat.fast_bound_ok n_in top q = fail .integerOverflow := by
  unfold kern.matmat.fast_bound_ok
  obtain ⟨s0, hs0, _⟩ := WP.spec_imp_exists (word_matmat3_fast_bound_ok_loop_ok q hsz)
  rw [hs0]
  simp only [bind_tc_ok]
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (I128.mul_spec (x := 32#i128) (y := top) (by
    rw [show (32#i128).val = 32 from by norm_num]; exact h1.1)
    (by rw [show (32#i128).val = 32 from by norm_num]; exact h1.2))
  rw [hi]
  simp only [bind_tc_ok]
  rw [matmat4_I128_mul_overflow i 127#i128 (by
    rw [hiv, show (32#i128).val = 32 from by norm_num, show (127#i128).val = 127 from by norm_num]
    exact h2)]
  simp only [bind_tc_fail]

/-! ## The `fast_family` admission loops -/

/-- One iteration of `fast_family_loop0` (`Q8_0`, scale `128`) in the admitting state: with
`admitted = true` and a column `c < |qs|` whose code-point count is `n_in`, whose scale-block count is
`blocks`, whose `128`-scaled bound test passes and whose magnitude status is `Exact`, the body
advances to `(c + 1, true)`. The two `Result`-valued callees (`kern.matmat.fast_bound_ok` and
`chart.QuantAct.magnitude`) enter by their values; the arithmetic between them is the extracted body.
Domain: `c` is a stored index (`hc`) and the counter increment is in range (`hinc`). -/
theorem matmat4_fast_family_loop0_body_admits (n_in : Usize) (qs : Slice chart.QuantAct)
    (blocks : Usize) (c : Usize) (hc : c.val < qs.val.length)
    (hlen : (qs.val[c.val]'hc).codes.val.length = n_in.val)
    (hblk : (qs.val[c.val]'hc).scales.val.length = blocks.val)
    (hfb : kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[c.val]'hc) = ok true)
    (hmag : chart.QuantAct.magnitude (qs.val[c.val]'hc) = ok (chart.ActMagnitude.Exact 0#i128))
    (hinc : c.val + 1 ≤ Usize.max) :
    kern.matmat.fast_family_loop0.body n_in qs blocks c true ⦃ r =>
      ∃ c1 : Usize, r = ControlFlow.cont (c1, true) ∧ c1.val = c.val + 1 ⦄ := by
  unfold kern.matmat.fast_family_loop0.body
  have hlc : c < Slice.len qs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hc
  rw [if_pos hlc, if_pos rfl, slice_index_usize_ok qs c hc]
  simp only [chart.QuantAct.len, bind_tc_ok]
  rw [if_pos (UScalar.eq_of_val_eq (by rw [alloc.vec.Vec.len_val]; exact hlen))]
  simp only [chart.QuantAct.impl.scales, bind_tc_ok]
  rw [if_pos (UScalar.eq_of_val_eq (by
    rw [Slice.len_val]; simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.length]
    exact hblk))]
  rw [hfb]
  simp only [bind_tc_ok]
  rw [if_pos (by decide)]
  simp only [hmag, bind_tc_ok]
  rw [massert_True]
  simp only [bind_tc_ok]
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  rw [hc1]
  simp only [bind_tc_ok]
  refine ⟨c1, rfl, ?_⟩
  rw [hc1v, show (1#usize).val = 1 from by decide]

/-- From `(0, true)`, if every column of `qs` has `n_in` code points, `blocks` scale blocks, a passing
`128`-scaled bound test and an `Exact` magnitude status, the `Q8_0` family loop returns `true`. This
isolates the `kern.matmat` layer: the `chart.QuantAct.magnitude` walk is supplied by its value per
column, not refined here. -/
theorem word_matmat4_fast_family_loop0_ok (n_in : Usize) (qs : Slice chart.QuantAct)
    (blocks : Usize) (hsz : qs.val.length < Usize.max)
    (hlen : ∀ (j : Nat) (hj : j < qs.val.length), (qs.val[j]'hj).codes.val.length = n_in.val)
    (hblk : ∀ (j : Nat) (hj : j < qs.val.length), (qs.val[j]'hj).scales.val.length = blocks.val)
    (hfb : ∀ (j : Nat) (hj : j < qs.val.length),
      kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[j]'hj) = ok true)
    (hmag : ∀ (j : Nat) (hj : j < qs.val.length),
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop0 n_in qs blocks 0#usize true ⦃ r => r = true ⦄ := by
  unfold kern.matmat.fast_family_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Usize × Bool => qs.val.length - s.1.val)
    (inv := fun s => s.2 = true ∧ s.1.val ≤ qs.val.length)
    (post := fun y : Bool => y = true)
  · rintro ⟨c, admitted⟩ ⟨hadm, hc⟩
    dsimp only at hadm hc ⊢
    subst hadm
    by_cases hlt : c < Slice.len qs
    · have hcv : c.val < qs.val.length := by
        rw [UScalar.lt_equiv, Slice.len_val] at hlt; exact hlt
      obtain ⟨y, hc1, c1, hy, hc1v⟩ := WP.spec_imp_exists
        (matmat4_fast_family_loop0_body_admits n_in qs blocks c hcv
          (hlen c.val hcv) (hblk c.val hcv) (hfb c.val hcv) (hmag c.val hcv)
          (by omega))
      rw [hc1, hy]
      simp only [WP.spec_ok]
      exact ⟨⟨trivial, by omega⟩, by omega⟩
    · unfold kern.matmat.fast_family_loop0.body
      rw [if_neg hlt]
      simp only [WP.spec_ok]
  · exact ⟨rfl, by simp⟩

/-- One iteration of `fast_family_loop1` (`Q5_0`, scale `16`): the `16`-scaled bound test in place of
`128`. Domain, hypotheses and conclusion as for `matmat4_fast_family_loop0_body_admits`. -/
theorem matmat4_fast_family_loop1_body_admits (n_in : Usize) (qs : Slice chart.QuantAct)
    (blocks : Usize) (c : Usize) (hc : c.val < qs.val.length)
    (hlen : (qs.val[c.val]'hc).codes.val.length = n_in.val)
    (hblk : (qs.val[c.val]'hc).scales.val.length = blocks.val)
    (hfb : kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[c.val]'hc) = ok true)
    (hmag : chart.QuantAct.magnitude (qs.val[c.val]'hc) = ok (chart.ActMagnitude.Exact 0#i128))
    (hinc : c.val + 1 ≤ Usize.max) :
    kern.matmat.fast_family_loop1.body n_in qs blocks c true ⦃ r =>
      ∃ c1 : Usize, r = ControlFlow.cont (c1, true) ∧ c1.val = c.val + 1 ⦄ := by
  unfold kern.matmat.fast_family_loop1.body
  have hlc : c < Slice.len qs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hc
  rw [if_pos hlc, if_pos rfl, slice_index_usize_ok qs c hc]
  simp only [chart.QuantAct.len, bind_tc_ok]
  rw [if_pos (UScalar.eq_of_val_eq (by rw [alloc.vec.Vec.len_val]; exact hlen))]
  simp only [chart.QuantAct.impl.scales, bind_tc_ok]
  rw [if_pos (UScalar.eq_of_val_eq (by
    rw [Slice.len_val]; simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.length]
    exact hblk))]
  rw [hfb]
  simp only [bind_tc_ok]
  rw [if_pos (by decide)]
  simp only [hmag, bind_tc_ok]
  rw [massert_True]
  simp only [bind_tc_ok]
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  rw [hc1]
  simp only [bind_tc_ok]
  refine ⟨c1, rfl, ?_⟩
  rw [hc1v, show (1#usize).val = 1 from by decide]

/-- From `(0, true)`, if every column of `qs` has `n_in` code points, `blocks` scale blocks, a passing
`16`-scaled bound test and an `Exact` magnitude status, the `Q5_0` family loop returns `true`. -/
theorem word_matmat4_fast_family_loop1_ok (n_in : Usize) (qs : Slice chart.QuantAct)
    (blocks : Usize) (hsz : qs.val.length < Usize.max)
    (hlen : ∀ (j : Nat) (hj : j < qs.val.length), (qs.val[j]'hj).codes.val.length = n_in.val)
    (hblk : ∀ (j : Nat) (hj : j < qs.val.length), (qs.val[j]'hj).scales.val.length = blocks.val)
    (hfb : ∀ (j : Nat) (hj : j < qs.val.length),
      kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[j]'hj) = ok true)
    (hmag : ∀ (j : Nat) (hj : j < qs.val.length),
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop1 n_in qs blocks 0#usize true ⦃ r => r = true ⦄ := by
  unfold kern.matmat.fast_family_loop1
  apply loop.spec_decr_nat
    (measure := fun s : Usize × Bool => qs.val.length - s.1.val)
    (inv := fun s => s.2 = true ∧ s.1.val ≤ qs.val.length)
    (post := fun y : Bool => y = true)
  · rintro ⟨c, admitted⟩ ⟨hadm, hc⟩
    dsimp only at hadm hc ⊢
    subst hadm
    by_cases hlt : c < Slice.len qs
    · have hcv : c.val < qs.val.length := by
        rw [UScalar.lt_equiv, Slice.len_val] at hlt; exact hlt
      obtain ⟨y, hc1, c1, hy, hc1v⟩ := WP.spec_imp_exists
        (matmat4_fast_family_loop1_body_admits n_in qs blocks c hcv
          (hlen c.val hcv) (hblk c.val hcv) (hfb c.val hcv) (hmag c.val hcv)
          (by omega))
      rw [hc1, hy]
      simp only [WP.spec_ok]
      exact ⟨⟨trivial, by omega⟩, by omega⟩
    · unfold kern.matmat.fast_family_loop1.body
      rw [if_neg hlt]
      simp only [WP.spec_ok]
  · exact ⟨rfl, by simp⟩

/-! ## Non-vacuity -/

/-- A zero scale maximum makes the second test hold: `matmat4_fastBoundRef 1 128 0 = true`. -/
example : matmat4_fastBoundRef 1 128 0 = true := by
  unfold matmat4_fastBoundRef matmat4_satMul matmat4_f16MaxFixed
  norm_num [I128.min_eq, I128.max_eq]

/-- The domain hypotheses of `word_matmat4_fast_bound_ok_ok` hold jointly at `n_in = 32`,
`top = 128`. -/
example : (32#usize).val < 2 ^ 127 ∧ -2 ^ 127 ≤ 32 * (128#i128).val ∧
    32 * (128#i128).val < 2 ^ 127 := by
  norm_num

/-- ... and the second-product side condition holds there too. -/
example : -2 ^ 127 ≤ 32 * (128#i128).val * 127 ∧ 32 * (128#i128).val * 127 < 2 ^ 127 := by
  norm_num

/-- With `qs` empty the family-loop hypotheses hold jointly and the `Q8_0` loop returns `true`: the
domain (`hsz`) and the four per-column hypotheses are all inhabited, and the loop terminates
immediately. -/
example (n_in : Usize) (blocks : Usize) :
    kern.matmat.fast_family_loop0 n_in (Slice.new chart.QuantAct) blocks 0#usize true ⦃
      r => r = true ⦄ :=
  word_matmat4_fast_family_loop0_ok n_in (Slice.new chart.QuantAct) blocks
    (by rw [Slice.new_val]; exact Nat.lt_of_lt_of_le Nat.zero_lt_one usize_max_ge_one)
    (fun _ hj => by simp [Slice.new_val] at hj)
    (fun _ hj => by simp [Slice.new_val] at hj)
    (fun _ hj => by simp [Slice.new_val] at hj)
    (fun _ hj => by simp [Slice.new_val] at hj)

/-! ## Axiom cone -/


end apeiron_word
