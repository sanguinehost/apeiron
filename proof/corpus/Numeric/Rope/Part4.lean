import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale
import Numeric.Rope.Part3

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Monadic kit

Three conversion lemmas for the do-notation bind shapes the extraction emits: the `bind`
congruence kit used by the value proofs below. -/

theorem c10_let_push {α β γ : Type} (e : α) (rest : Result β) (g : β → Result γ) :
    ((let _y := e; rest) >>= g) = (let _y := e; rest >>= g) := rfl

theorem c10_let_prod_push {α₁ α₂ β γ : Type} (e : α₁ × α₂) (rest : Result β)
    (g : β → Result γ) :
    ((let (_a, _b) := e; rest) >>= g) = (let (_a, _b) := e; rest >>= g) := by
  cases e
  rfl

theorem c10_pair_bind {α β γ : Type} (m : Aeneas.Std.Result (α × β))
    (K : α → β → Aeneas.Std.Result γ) :
    (m >>= fun p => K p.1 p.2) = (do let (a, b) ← m; K a b) := by
  apply congrArg (fun k => m >>= k)
  funext p
  cases p
  rfl

theorem word_fixed_sin_cos_past_guards_value (x : I64) (hx : trig_window x) :
          (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ←
          lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let k_wide ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if k_wide > i6
        then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if k_wide < i7
          then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 k_wide)
            let i8 ← lift (core.convert.num.FromI128I64.from x)
            let i9 ← lift (core.convert.num.FromI128I64.from k)
            let i10 ←
              lift (core.convert.num.FromI128I64.from fixed_trig.TWO_PI_Q32)
            let i11 ← i9 * i10
            let r_wide ← i8 - i11
            let i12 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if r_wide > i12
            then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
            else
              let i13 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if r_wide < i13
              then ok (core.result.Result.Err fixed_trig.TrigError.OutOfRange)
              else
                let r ← lift (IScalar.cast .I64 r_wide)
                let (s, negate) ←
                  if r > fixed_trig.HALF_PI_Q32
                  then do
                       let i14 ← r - fixed_trig.PI_Q32
                       ok (i14, true)
                  else
                    do
                    let i14 ← -. fixed_trig.HALF_PI_Q32
                    if r < i14
                    then let i15 ← r + fixed_trig.PI_Q32
                         ok (i15, true)
                    else ok (r, false)
                let (arg, swap) ←
                  if s > fixed_trig.QUARTER_PI_Q32
                  then do
                       let i14 ← fixed_trig.HALF_PI_Q32 - s
                       ok (i14, 1#i32)
                  else
                    do
                    let i14 ← -. fixed_trig.QUARTER_PI_Q32
                    if s < i14
                    then let i15 ← s + fixed_trig.HALF_PI_Q32
                         ok (i15, 2#i32)
                    else ok (s, 0#i32)
                let (sin_t, cos_t) ← fixed_trig.taylor_sin_cos arg
                let (sin_out, cos_out) ←
                  if swap = 1#i32
                  then ok (cos_t, sin_t)
                  else
                    if swap = 2#i32
                    then do
                         let i14 ← -. cos_t
                         ok (i14, sin_t)
                    else ok (sin_t, cos_t)
                if negate
                then
                  let sin_out1 ← -. sin_out
                  let cos_out1 ← -. cos_out
                  ok (core.result.Result.Ok ({ sin := sin_out1, cos := cos_out1 } : fixed_trig.SinCos))
                else
                  ok (core.result.Result.Ok ({ sin := sin_out, cos := cos_out } : fixed_trig.SinCos))) ⦃ z =>
      ∃ k r s arg sin_v cos_v : I64, ∃ negate : Bool, ∃ swap : I32,
        z = core.result.Result.Ok ({ sin := sin_v, cos := cos_v } : fixed_trig.SinCos) ∧
        k.val = (x.val * 683565276 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 26986075409 ∧
        s.val = fixed_trig.fold1 r.val ∧
        negate = fixed_trig.fold1_negate r.val ∧
        arg.val = fixed_trig.fold2 (fixed_trig.fold1 r.val) ∧
        swap.val = fixed_trig.fold2_swap (fixed_trig.fold1 r.val) ∧
        sin_v.val = (if negate then -1 else 1) *
          (if swap.val = 1 then fixed_trig.cos_poly arg.val
           else if swap.val = 2 then -fixed_trig.cos_poly arg.val
           else fixed_trig.sin_poly arg.val) ∧
        cos_v.val = (if negate then -1 else 1) *
          (if swap.val = 1 then fixed_trig.sin_poly arg.val
           else if swap.val = 2 then fixed_trig.sin_poly arg.val
           else fixed_trig.cos_poly arg.val) ⦄ := by
  rw [trig_window] at hx
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try norm_num at *)
  · norm_num [I128.min, I128.numBits, fixed_trig.TWO_PI_RECIP_Q32] at *
    nlinarith [hx.1, hx.2]
  · norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_RECIP_Q32] at *
    nlinarith [hx.1, hx.2]
  · have hi4 : (↑i4 : ℤ) = 2 ^ 63 := by
      rw [i4_post]; norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
    rw [product_post, hi4]
    norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_RECIP_Q32]
    nlinarith [hx.1, hx.2]
  · have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hkmax : (↑k_wide : ℤ) ≤ 652 := hsb.2
    norm_num [I64.rMax, I64.numBits] at *
    exfalso; omega
  · have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hkmin : -652 ≤ (↑k_wide : ℤ) := hsb.1
    norm_num [I64.rMin, I64.numBits] at *
    exfalso; omega
  · norm_num [I128.min, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2]
  · norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2]
  · rw [i11_post]
    norm_num [I128.min, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2, hx.1, hx.2]
  · rw [i11_post]
    norm_num [I128.max, I128.numBits, fixed_trig.TWO_PI_Q32] at *
    have h1 := int_bmod_64_lower (k_wide : ℤ)
    have h2 := int_bmod_64_upper (k_wide : ℤ)
    norm_num at h1 h2
    nlinarith [h1, h2, hx.1, hx.2]
  · have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hsr := fixed_trig_shift_bound_window x.val k_wide.val hx.1 hx.2 hsv
    have hkmod : (↑k_wide : ℤ).bmod 18446744073709551616 = k_wide.val := by
      have h := int_bmod_self_64 k_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hrv : r_wide.val = x.val - k_wide.val * 26986075409 := by
      rw [r_wide_post, i11_post]
      norm_num [fixed_trig.TWO_PI_Q32]
      rw [hkmod]
    have hrr := fixed_trig_reduction_remainder_range x.val k_wide.val hsv hsr
    norm_num [I64.rMax, I64.numBits] at *
    exfalso; omega
  · have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hsr := fixed_trig_shift_bound_window x.val k_wide.val hx.1 hx.2 hsv
    have hkmod : (↑k_wide : ℤ).bmod 18446744073709551616 = k_wide.val := by
      have h := int_bmod_self_64 k_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hrv : r_wide.val = x.val - k_wide.val * 26986075409 := by
      rw [r_wide_post, i11_post]
      norm_num [fixed_trig.TWO_PI_Q32]
      rw [hkmod]
    have hrr := fixed_trig_reduction_remainder_range x.val k_wide.val hsv hsr
    norm_num [I64.rMin, I64.numBits] at *
    exfalso; omega
  · -- success: value characterization, then fold/taylor/quadrant
    have hsv : k_wide.val = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      have h1 : (↑k_wide : ℤ) = (↑x * 683565276 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [k_wide_post, i5_post, i4_post, product_post, Int.shiftRight_eq_div_pow]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod, fixed_trig.TWO_PI_RECIP_Q32]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_trig_shift_bound x.val k_wide.val hx.1 hx.2 hsv
    have hsr := fixed_trig_shift_bound_window x.val k_wide.val hx.1 hx.2 hsv
    have hkmod : (↑k_wide : ℤ).bmod 18446744073709551616 = k_wide.val := by
      have h := int_bmod_self_64 k_wide.val (by omega) (by omega)
      simpa [show (18446744073709551616 : ℤ) = 2 ^ 64 by norm_num] using h
    have hrval : r_wide.val = x.val - k_wide.val * 26986075409 := by
      rw [r_wide_post, i11_post]
      norm_num [fixed_trig.TWO_PI_Q32]
      rw [hkmod]
    have hrt := fixed_trig_reduction_remainder_range_tight x.val k_wide.val hx.1 hx.2 hsv
    have hcast : (IScalar.cast IScalarTy.I64 r_wide).val = r_wide.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact int_bmod_self_64 r_wide.val (by omega) (by omega)
    have hr36 : |(IScalar.cast IScalarTy.I64 r_wide).val| ≤ 2 ^ 36 := by
      rw [hcast, hrval, abs_le]
      exact ⟨by linarith [hrt.1], by linarith [hrt.2]⟩
    have hcastmod : (IScalar.cast IScalarTy.I64 r_wide).val = (↑r_wide : ℤ).bmod 18446744073709551616 := by
      rw [IScalar.cast_val_eq]; norm_num
    rw [← hcastmod]
    have hr34 : |(IScalar.cast IScalarTy.I64 r_wide).val| ≤ 2 ^ 34 := by
      rw [hcast, hrval, abs_le]
      exact ⟨hrt.1, hrt.2⟩
    have hkcast : (IScalar.cast IScalarTy.I64 k_wide).val = k_wide.val := by
      rw [IScalar.cast_val_eq]; norm_num
      exact int_bmod_self_64 k_wide.val (by omega) (by omega)
    have hkval : (↑(IScalar.cast IScalarTy.I64 k_wide) : ℤ) = (x.val * 683565276 + 2 ^ 63) >>> 64 := by
      rw [hkcast]; exact hsv
    have hrval2 : (↑(IScalar.cast IScalarTy.I64 r_wide) : ℤ) =
        x.val - (↑(IScalar.cast IScalarTy.I64 k_wide)) * 26986075409 := by
      rw [hcast, hkcast]; exact hrval
    apply Aeneas.Std.WP.spec_bind (fixed_trig_fold1_spec (IScalar.cast IScalarTy.I64 r_wide) hr36)
    intro pp hp
    obtain ⟨s, negate⟩ := pp
    have hs : s.val = fixed_trig.fold1 (↑(IScalar.cast IScalarTy.I64 r_wide)) := hp.1
    have hneg : negate = fixed_trig.fold1_negate (↑(IScalar.cast IScalarTy.I64 r_wide)) := hp.2
    have hs36 : |s.val| ≤ 2 ^ 36 := by rw [hs]; exact fixed_trig.fold1_abs_le _ hr36
    apply Aeneas.Std.WP.spec_bind (fixed_trig_fold2_spec s hs36)
    intro qq hq
    obtain ⟨arg, swap⟩ := qq
    have harg : arg.val = fixed_trig.fold2 (fixed_trig.fold1 (↑(IScalar.cast IScalarTy.I64 r_wide))) := by
      rw [hq.1, hs]
    have hswap : swap.val = fixed_trig.fold2_swap (fixed_trig.fold1 (↑(IScalar.cast IScalarTy.I64 r_wide))) := by
      rw [hq.2, hs]
    have harg32 : |arg.val| ≤ 2 ^ 32 := by
      rw [harg]; exact fixed_trig.fold_arg_abs_le _ hr34
    apply Aeneas.Std.WP.spec_bind (word_taylor_spec arg harg32)
    intro sc hsc
    obtain ⟨sin_t, cos_t, rfl, hsin, hcos, hsinB, hcosB⟩ := hsc
    have hsinNeg : (↑sin_t : ℤ) ≠ IScalar.min IScalarTy.I64 := by
      intro hh
      have := abs_le.mp hsinB
      rw [hh] at this
      norm_num [IScalar.min, I64.min, I64.numBits] at this
    have hcosNeg : (↑cos_t : ℤ) ≠ IScalar.min IScalarTy.I64 := by
      intro hh
      have := abs_le.mp hcosB
      rw [hh] at this
      norm_num [IScalar.min, I64.min, I64.numBits] at this
    by_cases h1 : swap = 1#i32
    · -- swap = 1: output is (cos_t, sin_t)
      have hsw1 : (swap : ℤ) = 1 := by rw [h1]; rfl
      simp only [if_pos h1]
      by_cases hnt : negate = true
      · have hfn : fixed_trig.fold1_negate ↑(IScalar.cast IScalarTy.I64 r_wide) = true := by
          rw [← hneg]; exact hnt
        simp only [if_pos hnt]
        step*
        refine ⟨IScalar.cast IScalarTy.I64 k_wide, IScalar.cast IScalarTy.I64 r_wide, s, arg,
          sin_out1, cos_out1, rfl, hkval, hrval2, hs, harg, swap, hswap, ?_, ?_⟩
        · rw [if_pos hsw1, if_pos hfn]; simp [sin_out1_post, ← hcos]
        · rw [if_pos hsw1, if_pos hfn]; simp [cos_out1_post, ← hsin]
      · have hfn : ¬(fixed_trig.fold1_negate ↑(IScalar.cast IScalarTy.I64 r_wide) = true) := by
          rw [← hneg]; exact hnt
        simp only [if_neg hnt]
        step*
        refine ⟨IScalar.cast IScalarTy.I64 k_wide, IScalar.cast IScalarTy.I64 r_wide, s, arg,
          cos_t, sin_t, rfl, hkval, hrval2, hs, harg, swap, hswap, ?_, ?_⟩
        · rw [if_pos hsw1, if_neg hfn]; exact hcos
        · rw [if_pos hsw1, if_neg hfn]; exact hsin
    · by_cases h2 : swap = 2#i32
      · -- swap = 2: output is (-cos_t, sin_t) after negation bookkeeping
        have hsw2 : (swap : ℤ) = 2 := by rw [h2]; rfl
        have hsw1n : ¬((swap : ℤ) = 1) := by
          intro hh
          apply h1
          rw [IScalar.eq_equiv]
          rw [hh]
          norm_num
        simp only [if_neg h1, if_pos h2]
        by_cases hnt : negate = true
        · have hfn : fixed_trig.fold1_negate ↑(IScalar.cast IScalarTy.I64 r_wide) = true := by
            rw [← hneg]; exact hnt
          simp only [if_pos hnt]
          step*
          refine ⟨IScalar.cast IScalarTy.I64 k_wide, IScalar.cast IScalarTy.I64 r_wide, s, arg,
            _, _, rfl, hkval, hrval2, hs, harg, swap, hswap, ?_, ?_⟩
          · rw [if_neg hsw1n, if_pos hsw2, if_pos hfn]; simp [sin_out1_post, x_post, ← hcos]
          · rw [if_neg hsw1n, if_pos hsw2, if_pos hfn]; simp [cos_out1_post, ← hsin]
        · have hfn : ¬(fixed_trig.fold1_negate ↑(IScalar.cast IScalarTy.I64 r_wide) = true) := by
            rw [← hneg]; exact hnt
          simp only [if_neg hnt]
          step*
          refine ⟨IScalar.cast IScalarTy.I64 k_wide, IScalar.cast IScalarTy.I64 r_wide, s, arg,
            _, _, rfl, hkval, hrval2, hs, harg, swap, hswap, ?_, ?_⟩
          · rw [if_neg hsw1n, if_pos hsw2, if_neg hfn]; simp [x_post, ← hcos]
          · rw [if_neg hsw1n, if_pos hsw2, if_neg hfn]; exact hsin
      · -- swap = 0: output is (sin_t, cos_t)
        have hsw1n : ¬((swap : ℤ) = 1) := by
          intro hh
          apply h1
          rw [IScalar.eq_equiv]
          rw [hh]
          norm_num
        have hsw2n : ¬((swap : ℤ) = 2) := by
          intro hh
          apply h2
          rw [IScalar.eq_equiv]
          rw [hh]
          norm_num
        simp only [if_neg h1, if_neg h2]
        by_cases hnt : negate = true
        · have hfn : fixed_trig.fold1_negate ↑(IScalar.cast IScalarTy.I64 r_wide) = true := by
            rw [← hneg]; exact hnt
          simp only [if_pos hnt]
          step*
          refine ⟨IScalar.cast IScalarTy.I64 k_wide, IScalar.cast IScalarTy.I64 r_wide, s, arg,
            sin_out1, cos_out1, rfl, hkval, hrval2, hs, harg, swap, hswap, ?_, ?_⟩
          · rw [if_neg hsw1n, if_neg hsw2n, if_pos hfn]; simp [sin_out1_post, ← hsin]
          · rw [if_neg hsw1n, if_neg hsw2n, if_pos hfn]; simp [cos_out1_post, ← hcos]
        · have hfn : ¬(fixed_trig.fold1_negate ↑(IScalar.cast IScalarTy.I64 r_wide) = true) := by
            rw [← hneg]; exact hnt
          simp only [if_neg hnt]
          step*
          refine ⟨IScalar.cast IScalarTy.I64 k_wide, IScalar.cast IScalarTy.I64 r_wide, s, arg,
            sin_t, cos_t, rfl, hkval, hrval2, hs, harg, swap, hswap, ?_, ?_⟩
          · rw [if_neg hsw1n, if_neg hsw2n, if_neg hfn]; exact hsin
          · rw [if_neg hsw1n, if_neg hsw2n, if_neg hfn]; exact hcos

/-! ## Analytic composition: the value bounds against `Real.sin`/`Real.cos` -/

theorem sin_poly_close_le (t : I64) (ht : |t.val| ≤ 2 ^ 32) (h1 : |(t.val : ℝ) / q32| ≤ 1) :
    |((fixed_trig.sin_poly t.val : ℤ) : ℝ) / q32 - Real.sin ((t.val : ℝ) / q32)| ≤
      15 / (2 * q32) + 1 / q32 + |(t.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
  have hled := fixed_trig.sin_poly_horner t ht
  have hdef := fixed_trig.sinHorner_taylor ((t.val : ℝ) / q32) h1
  have hrem := fixed_trig.sinTaylor_remainder ((t.val : ℝ) / q32) h1
  calc |((fixed_trig.sin_poly t.val : ℤ) : ℝ) / q32 - Real.sin ((t.val : ℝ) / q32)|
      ≤ |((fixed_trig.sin_poly t.val : ℤ) : ℝ) / q32 - fixed_trig.sinHorner ((t.val : ℝ) / q32)| +
        |fixed_trig.sinHorner ((t.val : ℝ) / q32) - Real.sin ((t.val : ℝ) / q32)| := abs_sub_le _ _ _
    _ ≤ 15 / (2 * q32) + (1 / q32 + |(t.val : ℝ) / q32| ^ 9 * (10 / 3265920)) := by
        apply add_le_add hled
        calc |fixed_trig.sinHorner ((t.val : ℝ) / q32) - Real.sin ((t.val : ℝ) / q32)|
            ≤ |fixed_trig.sinHorner ((t.val : ℝ) / q32) - fixed_trig.sinTaylor ((t.val : ℝ) / q32)| +
              |fixed_trig.sinTaylor ((t.val : ℝ) / q32) - Real.sin ((t.val : ℝ) / q32)| :=
              abs_sub_le _ _ _
          _ ≤ 1 / q32 + |(t.val : ℝ) / q32| ^ 9 * (10 / 3265920) := add_le_add hdef hrem
    _ = 15 / (2 * q32) + 1 / q32 + |(t.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by ring

theorem cos_poly_close_le (t : I64) (ht : |t.val| ≤ 2 ^ 32) (h1 : |(t.val : ℝ) / q32| ≤ 1) :
    |((fixed_trig.cos_poly t.val : ℤ) : ℝ) / q32 - Real.cos ((t.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(t.val : ℝ) / q32| ^ 10 * (11 / 36288000) := by
  have hled := fixed_trig.cos_poly_horner t ht
  have hdef := fixed_trig.cosHorner_taylor ((t.val : ℝ) / q32) h1
  have hrem := fixed_trig.cosTaylor_remainder ((t.val : ℝ) / q32) h1
  calc |((fixed_trig.cos_poly t.val : ℤ) : ℝ) / q32 - Real.cos ((t.val : ℝ) / q32)|
      ≤ |((fixed_trig.cos_poly t.val : ℤ) : ℝ) / q32 - fixed_trig.cosHorner ((t.val : ℝ) / q32)| +
        |fixed_trig.cosHorner ((t.val : ℝ) / q32) - Real.cos ((t.val : ℝ) / q32)| := abs_sub_le _ _ _
    _ ≤ 40 / (2 * q32) + (1 / q32 + |(t.val : ℝ) / q32| ^ 10 * (11 / 36288000)) := by
        apply add_le_add hled
        calc |fixed_trig.cosHorner ((t.val : ℝ) / q32) - Real.cos ((t.val : ℝ) / q32)|
            ≤ |fixed_trig.cosHorner ((t.val : ℝ) / q32) - fixed_trig.cosTaylor ((t.val : ℝ) / q32)| +
              |fixed_trig.cosTaylor ((t.val : ℝ) / q32) - Real.cos ((t.val : ℝ) / q32)| :=
              abs_sub_le _ _ _
          _ ≤ 1 / q32 + |(t.val : ℝ) / q32| ^ 10 * (11 / 36288000) := add_le_add hdef hrem
    _ = 40 / (2 * q32) + 1 / q32 + |(t.val : ℝ) / q32| ^ 10 * (11 / 36288000) := by ring

theorem cos_poly_rem_le (s : ℝ) (hs : |s| ≤ 1) :
    |s| ^ 10 * (11 / 36288000) ≤ |s| ^ 9 * (10 / 3265920) := by
  have h1 : |s| * (11 / 36288000) ≤ 10 / 3265920 := by
    calc |s| * (11 / 36288000) ≤ 1 * (11 / 36288000) :=
          mul_le_mul_of_nonneg_right hs (by norm_num)
      _ ≤ 10 / 3265920 := by norm_num
  calc |s| ^ 10 * (11 / 36288000) = |s| ^ 9 * (|s| * (11 / 36288000)) := by ring
    _ ≤ |s| ^ 9 * (10 / 3265920) := mul_le_mul_of_nonneg_left h1 (by positivity)

theorem trig_floor_close (s : ℝ) (hs : |s| ≤ Real.pi / 4) :
    (trig_delta + trig_fold_delta) + (40 / (2 * q32) + 1 / q32 + |s| ^ 9 * (10 / 3265920)) ≤
      trig_floor := by
  have hrem : |s| ^ 9 * (10 / 3265920) ≤ 1 / 2000000 := by
    have hpi : Real.pi / 4 ≤ 4 / 5 := by linarith [Real.pi_lt_d20]
    have h9 : |s| ^ 9 ≤ (4 / 5 : ℝ) ^ 9 :=
      le_trans (pow_le_pow_left₀ (abs_nonneg s) hs 9)
        (pow_le_pow_left₀ (by positivity : (0 : ℝ) ≤ Real.pi / 4) hpi 9)
    exact le_trans (mul_le_mul_of_nonneg_right h9 (by norm_num)) (by norm_num)
  have hq : (trig_delta + trig_fold_delta) + (40 / (2 * q32) + 1 / q32 + 1 / 2000000) ≤
      trig_floor := by
    rw [trig_delta, trig_fold_delta, trig_floor]
    norm_num [q32]
  linarith [hrem, hq]

theorem abs_smul_cancel (b : Bool) (A B : ℝ) :
    |(if b then (-1 : ℝ) else 1) * A - (if b then (-1 : ℝ) else 1) * B| = |A - B| := by
  rw [← mul_sub, abs_mul]
  cases b <;> simp

theorem sin_out_flagSin_le (negate : Bool) (swapv : ℤ) (arg sin_v : I64) (ht : |arg.val| ≤ 2 ^ 32)
    (h1 : |(arg.val : ℝ) / q32| ≤ 1)
    (hrel : sin_v.val = (if negate then -1 else 1) *
      (if swapv = 1 then fixed_trig.cos_poly arg.val
       else if swapv = 2 then -fixed_trig.cos_poly arg.val else fixed_trig.sin_poly arg.val)) :
    |(sin_v.val : ℝ) / q32 - flagSin negate swapv ((arg.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
  have hpS := sin_poly_close_le arg ht h1
  have hpC := cos_poly_close_le arg ht h1
  have hrC := cos_poly_rem_le ((arg.val : ℝ) / q32) h1
  have hC : |((fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32 - Real.cos ((arg.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
    refine le_trans hpC ?_; linarith [hrC]
  have h1540 : (15 : ℝ) / (2 * q32) ≤ 40 / (2 * q32) :=
    div_le_div_of_nonneg_right (by norm_num) (by nlinarith [q32_pos])
  have hS : |((fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32 - Real.sin ((arg.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
    refine le_trans hpS ?_; linarith [h1540]
  rw [hrel, flagSin]
  by_cases hb1 : swapv = 1
  · rw [if_pos hb1]
    rw [show (((if negate then (-1 : ℤ) else 1) * fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32 =
        (if negate then (-1 : ℝ) else 1) * (((fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32) by
      push_cast; ring]
    rw [abs_smul_cancel, if_pos hb1]; exact hC
  · by_cases hb2 : swapv = 2
    · rw [if_neg hb1, if_pos hb2]
      rw [show (((if negate then (-1 : ℤ) else 1) * (-fixed_trig.cos_poly arg.val) : ℤ) : ℝ) / q32 =
          (if negate then (-1 : ℝ) else 1) * (((-fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32) by
        push_cast; ring]
      rw [abs_smul_cancel, if_neg hb1, if_pos hb2]
      rw [show |(((-fixed_trig.cos_poly arg.val : ℤ) : ℝ)) / q32 - (-Real.cos ((arg.val : ℝ) / q32))| =
          |((fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32 - Real.cos ((arg.val : ℝ) / q32)| by
        rw [show (((-fixed_trig.cos_poly arg.val : ℤ) : ℝ)) / q32 - (-Real.cos ((arg.val : ℝ) / q32)) =
            -((((fixed_trig.cos_poly arg.val : ℤ) : ℝ)) / q32 - Real.cos ((arg.val : ℝ) / q32)) by
          push_cast; ring, abs_neg]]
      exact hC
    · rw [if_neg hb1, if_neg hb2]
      rw [show (((if negate then (-1 : ℤ) else 1) * fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32 =
          (if negate then (-1 : ℝ) else 1) * (((fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32) by
        push_cast; ring]
      rw [abs_smul_cancel, if_neg hb1, if_neg hb2]; exact hS

theorem cos_out_flagCos_le (negate : Bool) (swapv : ℤ) (arg cos_v : I64) (ht : |arg.val| ≤ 2 ^ 32)
    (h1 : |(arg.val : ℝ) / q32| ≤ 1)
    (hrel : cos_v.val = (if negate then -1 else 1) *
      (if swapv = 1 then fixed_trig.sin_poly arg.val
       else if swapv = 2 then fixed_trig.sin_poly arg.val else fixed_trig.cos_poly arg.val)) :
    |(cos_v.val : ℝ) / q32 - flagCos negate swapv ((arg.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
  have hpS := sin_poly_close_le arg ht h1
  have hpC := cos_poly_close_le arg ht h1
  have hrC := cos_poly_rem_le ((arg.val : ℝ) / q32) h1
  have hC : |((fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32 - Real.cos ((arg.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
    refine le_trans hpC ?_; linarith [hrC]
  have h1540 : (15 : ℝ) / (2 * q32) ≤ 40 / (2 * q32) :=
    div_le_div_of_nonneg_right (by norm_num) (by nlinarith [q32_pos])
  have hS : |((fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32 - Real.sin ((arg.val : ℝ) / q32)| ≤
      40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920) := by
    refine le_trans hpS ?_; linarith [h1540]
  rw [hrel, flagCos]
  by_cases hb1 : swapv = 1
  · rw [if_pos hb1]
    rw [show (((if negate then (-1 : ℤ) else 1) * fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32 =
        (if negate then (-1 : ℝ) else 1) * (((fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32) by
      push_cast; ring]
    rw [abs_smul_cancel, if_pos hb1]; exact hS
  · by_cases hb2 : swapv = 2
    · rw [if_neg hb1, if_pos hb2]
      rw [show (((if negate then (-1 : ℤ) else 1) * fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32 =
          (if negate then (-1 : ℝ) else 1) * (((fixed_trig.sin_poly arg.val : ℤ) : ℝ) / q32) by
        push_cast; ring]
      rw [abs_smul_cancel, if_neg hb1, if_pos hb2]; exact hS
    · rw [if_neg hb1, if_neg hb2]
      rw [show (((if negate then (-1 : ℤ) else 1) * fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32 =
          (if negate then (-1 : ℝ) else 1) * (((fixed_trig.cos_poly arg.val : ℤ) : ℝ) / q32) by
        push_cast; ring]
      rw [abs_smul_cancel, if_neg hb1, if_neg hb2]; exact hC

/-- Refinement on the window: success with the combined absolute/relative bound
`2^-20 · |f (toReal x)| + 2^-20` per output. The extraction identity comes from
`word_fixed_sin_cos_window_past_guards`; the value equations from
`word_fixed_sin_cos_past_guards_value`; the analytic budget is
`trig_delta + trig_fold_delta + 15/(2q32) + 1/q32 + |arg/q32|^9 · 10/3265920 ≤ trig_floor`. -/
theorem fixed_sin_cos_refines_proved : fixed_sin_cos_refines := by
  intro x hx
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 (word_fixed_sin_cos_past_guards_value x hx) with
    ⟨y, hy, hyv⟩
  rcases hyv with ⟨k, r, s, arg, sin_v, cos_v, negate, swap, h1, hk, hr, hs, hneg, harg, hswap,
    hsinv, hcosv⟩
  refine ⟨⟨sin_v, cos_v⟩, ?_, ?_, ?_⟩
  · rw [word_fixed_sin_cos_window_past_guards x hx]
    exact hy.trans (by rw [h1])
  · have hrt := fixed_trig_reduction_remainder_range_tight x.val k.val hx.1 hx.2 hk
    have hr34 : |r.val| ≤ 2 ^ 34 := by
      rw [abs_le]; exact ⟨by rw [hr]; exact hrt.1, by rw [hr]; exact hrt.2⟩
    have harg32 : |arg.val| ≤ 2 ^ 32 := by rw [harg]; exact fixed_trig.fold_arg_abs_le r.val hr34
    have hargq : |arg.val| ≤ 3373259426 := by
      rw [harg]; exact fixed_trig.fold_arg_abs_le_quarter r.val hr34
    have hΦ : |(arg.val : ℝ) / q32| ≤ Real.pi / 4 := by
      have h : |(arg.val : ℝ) / q32| ≤ (3373259426 : ℝ) / q32 := by
        rw [abs_div, abs_of_pos q32_pos]
        exact div_le_div_of_nonneg_right (by exact_mod_cast hargq) (le_of_lt q32_pos)
      exact le_trans h (le_of_lt quarter_pi_q32_floor.1)
    have h1' : |(arg.val : ℝ) / q32| ≤ 1 := le_trans hΦ (by linarith [Real.pi_lt_d20])
    have hrel : sin_v.val = (if negate then -1 else 1) *
        (if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then fixed_trig.cos_poly arg.val
         else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then -fixed_trig.cos_poly arg.val
         else fixed_trig.sin_poly arg.val) := by rw [hsinv, hswap]
    have hout := sin_out_flagSin_le negate (fixed_trig.fold2_swap (fixed_trig.fold1 r.val)) arg sin_v
      harg32 h1' hrel
    rw [hneg] at hout
    have hflag := flag_sin_reduction_bound x r k hx hk hr
    rw [← harg] at hflag
    have hfloor := trig_floor_close ((arg.val : ℝ) / q32) hΦ
    rw [show toReal sin_v = (sin_v.val : ℝ) / q32 from rfl]
    calc |(sin_v.val : ℝ) / q32 - Real.sin (toReal x)|
        ≤ |(sin_v.val : ℝ) / q32 - flagSin (fixed_trig.fold1_negate r.val)
              (fixed_trig.fold2_swap (fixed_trig.fold1 r.val)) ((arg.val : ℝ) / q32)| +
          |flagSin (fixed_trig.fold1_negate r.val)
              (fixed_trig.fold2_swap (fixed_trig.fold1 r.val)) ((arg.val : ℝ) / q32) -
            Real.sin (toReal x)| := abs_sub_le _ _ _
      _ ≤ (40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920)) +
          (trig_delta + trig_fold_delta) := add_le_add hout hflag
      _ ≤ trig_floor := by linarith [hfloor]
      _ ≤ epsExp * |Real.sin (toReal x)| + trig_floor := by
          have : (0 : ℝ) ≤ epsExp * |Real.sin (toReal x)| :=
            mul_nonneg (by norm_num [epsExp]) (abs_nonneg _)
          linarith
  · have hrt := fixed_trig_reduction_remainder_range_tight x.val k.val hx.1 hx.2 hk
    have hr34 : |r.val| ≤ 2 ^ 34 := by
      rw [abs_le]; exact ⟨by rw [hr]; exact hrt.1, by rw [hr]; exact hrt.2⟩
    have harg32 : |arg.val| ≤ 2 ^ 32 := by rw [harg]; exact fixed_trig.fold_arg_abs_le r.val hr34
    have hargq : |arg.val| ≤ 3373259426 := by
      rw [harg]; exact fixed_trig.fold_arg_abs_le_quarter r.val hr34
    have hΦ : |(arg.val : ℝ) / q32| ≤ Real.pi / 4 := by
      have h : |(arg.val : ℝ) / q32| ≤ (3373259426 : ℝ) / q32 := by
        rw [abs_div, abs_of_pos q32_pos]
        exact div_le_div_of_nonneg_right (by exact_mod_cast hargq) (le_of_lt q32_pos)
      exact le_trans h (le_of_lt quarter_pi_q32_floor.1)
    have h1' : |(arg.val : ℝ) / q32| ≤ 1 := le_trans hΦ (by linarith [Real.pi_lt_d20])
    have hrel : cos_v.val = (if negate then -1 else 1) *
        (if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 1 then fixed_trig.sin_poly arg.val
         else if fixed_trig.fold2_swap (fixed_trig.fold1 r.val) = 2 then fixed_trig.sin_poly arg.val
         else fixed_trig.cos_poly arg.val) := by rw [hcosv, hswap]
    have hout := cos_out_flagCos_le negate (fixed_trig.fold2_swap (fixed_trig.fold1 r.val)) arg cos_v
      harg32 h1' hrel
    rw [hneg] at hout
    have hflag := flag_cos_reduction_bound x r k hx hk hr
    rw [← harg] at hflag
    have hfloor := trig_floor_close ((arg.val : ℝ) / q32) hΦ
    rw [show toReal cos_v = (cos_v.val : ℝ) / q32 from rfl]
    calc |(cos_v.val : ℝ) / q32 - Real.cos (toReal x)|
        ≤ |(cos_v.val : ℝ) / q32 - flagCos (fixed_trig.fold1_negate r.val)
              (fixed_trig.fold2_swap (fixed_trig.fold1 r.val)) ((arg.val : ℝ) / q32)| +
          |flagCos (fixed_trig.fold1_negate r.val)
              (fixed_trig.fold2_swap (fixed_trig.fold1 r.val)) ((arg.val : ℝ) / q32) -
            Real.cos (toReal x)| := abs_sub_le _ _ _
      _ ≤ (40 / (2 * q32) + 1 / q32 + |(arg.val : ℝ) / q32| ^ 9 * (10 / 3265920)) +
          (trig_delta + trig_fold_delta) := add_le_add hout hflag
      _ ≤ trig_floor := by linarith [hfloor]
      _ ≤ epsExp * |Real.cos (toReal x)| + trig_floor := by
          have : (0 : ℝ) ≤ epsExp * |Real.cos (toReal x)| :=
            mul_nonneg (by norm_num [epsExp]) (abs_nonneg _)
          linarith



end

end apeiron_word
