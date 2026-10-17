/-
Refinement of the fixed-point `sin`/`cos` body `src/numeric/fixed_trig.rs` against the real
trigonometric functions (statement in `Numeric.Rope.Part1`).

The body is `t64 = x <<< 32` (`i128`), a nearest-even quotient `k = q_rne(t64, HALF_PI_Q64)`, the
narrowed remainder `r` (`|r| <= 1`), the Taylor series of `Part3` (`|err| <= 2^-23`), and the
quadrant swap/negate on `((k % 4) + 4) % 4`. `Part5` proves the reduction head: every checked
`i128` op succeeds on the window, `|k| <= 2608`, `|r - (x - k·pi/2)| <= 2^-30` and `|r| <= 1`.

The assembly composes the two stages. The reduction error `2^-30` propagates through the
1-Lipschitz `sin`/`cos`; the series error is `2^-23`; on `k = 4m + q` the quadrant is an exact
`sin`/`cos` addition identity. Total `2^-23 + 2^-30 < 2^-20 = trig_floor <= epsExp·|f| + trig_floor`.
The measured relative error of the body is `1.1e-9`; this proof is deliberately coarse.
-/
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale
import Numeric.Rope.Part3
import Numeric.Rope.Part5

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Elementary analytic kit -/

/-- Triangle inequality in the form the two error stages need. -/
theorem abs_sub_le_add (a b c : ℝ) : |a - c| ≤ |a - b| + |b - c| := by
  rw [show a - c = (a - b) + (b - c) by ring]
  exact abs_add_le _ _

/-- `sin` is 1-Lipschitz. -/
theorem sin_shift_le (y d : ℝ) : |Real.sin (y + d) - Real.sin y| ≤ |d| := by
  have h := Real.abs_sin_sub_sin_le (y + d) y
  rwa [show y + d - y = d by ring] at h

/-- `cos` is 1-Lipschitz. -/
theorem cos_shift_le (y d : ℝ) : |Real.cos (y + d) - Real.cos y| ≤ |d| := by
  have h := Real.abs_cos_sub_cos_le (y + d) y
  rwa [show y + d - y = d by ring] at h

theorem epsExp_nonneg : (0 : ℝ) ≤ epsExp := by unfold epsExp; norm_num

/-- The two-stage budget `2^-23 + 2^-30` is inside the target `epsExp·|s| + trig_floor`. -/
theorem trig_budget (s : ℝ) :
    (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) ≤ epsExp * |s| + trig_floor := by
  have h1 : (0 : ℝ) ≤ epsExp * |s| := mul_nonneg epsExp_nonneg (abs_nonneg s)
  have h2 : (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) ≤ trig_floor := by
    unfold trig_floor
    norm_num
  linarith

/-- A Q32.32 value whose integer value is the negation of another denotes the negated real. -/
theorem toReal_neg_of_val (v w : I64) (h : w.val = -v.val) : toReal w = -toReal v := by
  rw [toReal, toReal, h]
  push_cast
  ring

/-! ## The quadrant identities

`k·(pi/2) = 4m·(pi/2) + q·(pi/2) = m·(2·pi) + q·(pi/2)`; each `q` selects the `sin`/`cos` value at
`y + d` up to sign. -/

theorem sin_quad_0 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ)) :
    Real.sin (y + (k : ℝ) * (Real.pi / 2) + d) = Real.sin (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2) = (m : ℝ) * (2 * Real.pi) := by
    rw [hk]; ring
  rw [h, show y + (m : ℝ) * (2 * Real.pi) + d = (y + d) + m * (2 * Real.pi) by ring]
  rw [Real.sin_add_int_mul_two_pi]

theorem sin_quad_1 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ) + 1) :
    Real.sin (y + (k : ℝ) * (Real.pi / 2) + d) = Real.cos (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2) = (m : ℝ) * (2 * Real.pi) + Real.pi / 2 := by
    rw [hk]; ring
  rw [h, show y + ((m : ℝ) * (2 * Real.pi) + Real.pi / 2) + d
      = (y + d + Real.pi / 2) + m * (2 * Real.pi) by ring]
  rw [Real.sin_add_int_mul_two_pi, Real.sin_add_pi_div_two]

theorem sin_quad_2 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ) + 2) :
    Real.sin (y + (k : ℝ) * (Real.pi / 2) + d) = -Real.sin (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2) = (m : ℝ) * (2 * Real.pi) + Real.pi := by
    rw [hk]; ring
  rw [h, show y + ((m : ℝ) * (2 * Real.pi) + Real.pi) + d
      = (y + d + Real.pi) + m * (2 * Real.pi) by ring]
  rw [Real.sin_add_int_mul_two_pi, Real.sin_add_pi]

theorem sin_quad_3 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ) + 3) :
    Real.sin (y + (k : ℝ) * (Real.pi / 2) + d) = -Real.cos (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2)
      = (m : ℝ) * (2 * Real.pi) + Real.pi + Real.pi / 2 := by
    rw [hk]; ring
  rw [h, show y + ((m : ℝ) * (2 * Real.pi) + Real.pi + Real.pi / 2) + d
      = (y + d + Real.pi + Real.pi / 2) + m * (2 * Real.pi) by ring]
  rw [Real.sin_add_int_mul_two_pi, Real.sin_add_pi_div_two, Real.cos_add_pi]

theorem cos_quad_0 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ)) :
    Real.cos (y + (k : ℝ) * (Real.pi / 2) + d) = Real.cos (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2) = (m : ℝ) * (2 * Real.pi) := by
    rw [hk]; ring
  rw [h, show y + (m : ℝ) * (2 * Real.pi) + d = (y + d) + m * (2 * Real.pi) by ring]
  rw [Real.cos_add_int_mul_two_pi]

theorem cos_quad_1 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ) + 1) :
    Real.cos (y + (k : ℝ) * (Real.pi / 2) + d) = -Real.sin (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2) = (m : ℝ) * (2 * Real.pi) + Real.pi / 2 := by
    rw [hk]; ring
  rw [h, show y + ((m : ℝ) * (2 * Real.pi) + Real.pi / 2) + d
      = (y + d + Real.pi / 2) + m * (2 * Real.pi) by ring]
  rw [Real.cos_add_int_mul_two_pi, Real.cos_add_pi_div_two]

theorem cos_quad_2 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ) + 2) :
    Real.cos (y + (k : ℝ) * (Real.pi / 2) + d) = -Real.cos (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2) = (m : ℝ) * (2 * Real.pi) + Real.pi := by
    rw [hk]; ring
  rw [h, show y + ((m : ℝ) * (2 * Real.pi) + Real.pi) + d
      = (y + d + Real.pi) + m * (2 * Real.pi) by ring]
  rw [Real.cos_add_int_mul_two_pi, Real.cos_add_pi]

theorem cos_quad_3 (y d : ℝ) (k m : ℤ) (hk : (k : ℝ) = 4 * (m : ℝ) + 3) :
    Real.cos (y + (k : ℝ) * (Real.pi / 2) + d) = Real.sin (y + d) := by
  have h : (k : ℝ) * (Real.pi / 2)
      = (m : ℝ) * (2 * Real.pi) + Real.pi + Real.pi / 2 := by
    rw [hk]; ring
  rw [h, show y + ((m : ℝ) * (2 * Real.pi) + Real.pi + Real.pi / 2) + d
      = (y + d + Real.pi + Real.pi / 2) + m * (2 * Real.pi) by ring]
  rw [Real.cos_add_int_mul_two_pi, Real.cos_add_pi_div_two, Real.sin_add_pi]
  ring

/-! ## The quadrant selector -/

/-- `((k % 4) + 4) % 4` is in `[0, 4)`. -/
theorem quarter_cases (q : ℤ) (h0 : 0 ≤ q) (h4 : q < 4) :
    q = 0 ∨ q = 1 ∨ q = 2 ∨ q = 3 := by omega

/-- The Rust Euclidean-modulo idiom equals `k % 4`: the inner remainder is in `(-4, 4)`, so
adding `4` lands in `(0, 8)` and the outer remainder is the nonnegative representative. -/
theorem emod_quarter (k : ℤ) (i5 qv : ℤ) (hi5 : 0 < i5) (hqv : qv = Int.tmod i5 4)
    (hbridge : i5 = k.tmod 4 + 4) : k % 4 = qv := by
  have h1 : qv = i5 % 4 := by
    rw [hqv, Int.tmod_eq_emod_of_nonneg (le_of_lt hi5)]
  have h2 : i5 % 4 = (k.tmod 4) % 4 := by
    rw [hbridge, show k.tmod 4 + 4 = k.tmod 4 + 1 * 4 by norm_num,
      Int.add_mul_emod_self_right]
  have h3 : (k.tmod 4) % 4 = k % 4 := by
    have e : k = k.tmod 4 + k.tdiv 4 * 4 := by
      have h := Int.mul_tdiv_add_tmod k 4
      omega
    conv_rhs => rw [e]
    rw [Int.add_mul_emod_self_right]
  rw [← h3, ← h2, ← h1]

/-- The three checked `i128` operations of the quadrant selector succeed; the quarter is in
`[0, 4)`; and `k = 4m + q` for the narrowed quarter `q`. -/
theorem quarter_range (k : I128) :
    ∃ i4 i5 qv : I128,
      k % (4#i128) = ok i4 ∧ i4 + (4#i128) = ok i5 ∧
      (i5 % (4#i128)) = ok qv ∧ 0 ≤ qv.val ∧ qv.val < 4 ∧
      ∃ m : ℤ, k.val = 4 * m + qv.val := by
  obtain ⟨i4, hi4, hi4v⟩ :=
    WP.spec_imp_exists (IScalar.rem_spec (x := k) (y := (4#i128)) (by norm_num))
  have hi4v' : i4.val = Int.tmod k.val 4 := by simpa using hi4v
  have hi4lo : -4 < i4.val := by
    rw [hi4v']; exact Int.lt_tmod_of_pos _ (by norm_num)
  have hi4hi : i4.val < 4 := by
    rw [hi4v']; exact Int.tmod_lt_of_pos _ (by norm_num)
  have h4 : ((4#i128).val : ℤ) = 4 := by norm_num
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists
    (IScalar.add_spec (x := i4) (y := (4#i128))
      (by
        rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq, h4]; omega)
      (by
        rw [IScalar.max_IScalarTy_I128_eq, I128.max_eq, h4]; omega))
  have hi5v' : i5.val = i4.val + 4 := by
    have := hi5v
    rw [h4] at this
    exact this
  obtain ⟨qv, hqv, hqvv⟩ :=
    WP.spec_imp_exists (IScalar.rem_spec (x := i5) (y := (4#i128)) (by norm_num))
  have hqvv' : qv.val = Int.tmod i5.val 4 := by
    have := hqvv
    rw [h4] at this
    exact this
  have hi5pos : 0 < i5.val := by rw [hi5v']; omega
  have hqv0 : 0 ≤ qv.val := by
    rw [hqvv']
    exact Int.tmod_nonneg 4 (le_of_lt hi5pos)
  have hqv4 : qv.val < 4 := by
    rw [hqvv']
    exact Int.tmod_lt_of_pos _ (by norm_num)
  have hbridge : i5.val = k.val.tmod 4 + 4 := by rw [hi5v', hi4v']
  have hqmod : k.val % 4 = qv.val :=
    emod_quarter k.val i5.val qv.val hi5pos hqvv' hbridge
  refine ⟨i4, i5, qv, hi4, hi5, hqv, hqv0, hqv4, k.val / 4, ?_⟩
  rw [← hqmod, Int.mul_ediv_add_emod]

/-! ## The assembly -/

/-- The guard-free tail of the extracted body on the window: success and the target bound. The head
is `Part5.reduction_spec`, the series is `Part3.sin_cos_series_spec`, and the quadrant is the
`k = 4m + q` decomposition of `quarter_range`. -/
theorem trigPastGuards_refines (x : I64) (hx : trig_window x) :
    ∃ sc : fixed_trig.SinCos,
      trigPastGuards x = ok (core.result.Result.Ok sc) ∧
        |toReal sc.sin - Real.sin (toReal x)| ≤
          epsExp * |Real.sin (toReal x)| + trig_floor ∧
        |toReal sc.cos - Real.cos (toReal x)| ≤
          epsExp * |Real.cos (toReal x)| + trig_floor := by
  obtain ⟨t, k, p, num2, dd, q2, r, ht, htv, hk, hkabs, hp, hsub, hd, hdval, hq2, hrn, herr,
    hr1⟩ := reduction_spec x hx
  obtain ⟨sc, hser, hserrS, hserrC⟩ := sin_cos_series_spec r hr1
  obtain ⟨i4, i5, qv, hi4, hi5, hqv, hqv0, hqv4, m, hm⟩ := quarter_range k
  -- the series outputs are bounded, so both negations are in range
  have hsin2 : |toReal sc.sin| ≤ 2 := by
    have h1 := Real.abs_sin_le_one (toReal r)
    have h3 : |toReal sc.sin| ≤ |toReal sc.sin - Real.sin (toReal r)| + |Real.sin (toReal r)| := by
      simpa using abs_sub_le_add (toReal sc.sin) (Real.sin (toReal r)) 0
    have h23 : (2 : ℝ) ^ (-23 : ℤ) ≤ 1 := by norm_num
    linarith [hserrS, h1, h3, h23]
  have hcos2 : |toReal sc.cos| ≤ 2 := by
    have h1 := Real.abs_cos_le_one (toReal r)
    have h3 : |toReal sc.cos| ≤ |toReal sc.cos - Real.cos (toReal r)| + |Real.cos (toReal r)| := by
      simpa using abs_sub_le_add (toReal sc.cos) (Real.cos (toReal r)) 0
    have h23 : (2 : ℝ) ^ (-23 : ℤ) ≤ 1 := by norm_num
    linarith [hserrC, h1, h3, h23]
  have hsin40 : |sc.sin.val| ≤ 2 ^ 40 := val_abs_le_of_abs_toReal_le sc.sin hsin2
  have hcos40 : |sc.cos.val| ≤ 2 ^ 40 := val_abs_le_of_abs_toReal_le sc.cos hcos2
  have hminval : IScalar.min .I64 = -(2 : ℤ) ^ 63 := by
    rw [IScalar.min_IScalarTy_I64_eq, I64.min_eq]
    norm_num
  have hsin_ne : sc.sin ≠ IScalar.min .I64 := by
    intro h
    rw [h, hminval] at hsin40
    norm_num at hsin40
  have hcos_ne : sc.cos ≠ IScalar.min .I64 := by
    intro h
    rw [h, hminval] at hcos40
    norm_num at hcos40
  obtain ⟨n1, hn1, hn1v⟩ := WP.spec_imp_exists (HNeg.hNeg.step sc.sin hsin_ne)
  obtain ⟨n2, hn2, hn2v⟩ := WP.spec_imp_exists (HNeg.hNeg.step sc.cos hcos_ne)
  have hn1v' : n1.val = -sc.sin.val := by simpa using hn1v
  have hn2v' : n2.val = -sc.cos.val := by simpa using hn2v
  have hn1t : toReal n1 = -toReal sc.sin := toReal_neg_of_val sc.sin n1 hn1v'
  have hn2t : toReal n2 = -toReal sc.cos := toReal_neg_of_val sc.cos n2 hn2v'
  -- the real reduction error
  set X : ℝ := toReal x with hXdef
  set Y : ℝ := toReal r with hYdef
  have hd30 : |X - (Y + (k.val : ℝ) * (Real.pi / 2))| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
    have h := herr
    rw [show X - (Y + (k.val : ℝ) * (Real.pi / 2))
        = -(Y - (X - (k.val : ℝ) * (Real.pi / 2))) by ring, abs_neg]
    exact h
  set del : ℝ := X - (Y + (k.val : ℝ) * (Real.pi / 2)) with hdeldef
  have hdel : |del| ≤ (2 : ℝ) ^ (-30 : ℤ) := hd30
  have hX : X = Y + (k.val : ℝ) * (Real.pi / 2) + del := by rw [hdeldef]; ring
  -- the head, then the quadrant tail step by step
  simp only [trigPastGuards, lift, bind_tc_ok, ht, hk, hp, hsub, hd, hq2, hrn, hser]
  rw [hi4]; simp only [bind_tc_ok]
  rw [hi5]; simp only [bind_tc_ok]
  rw [hqv]; simp only [bind_tc_ok]
  rcases quarter_cases qv.val hqv0 hqv4 with hq0 | hq1 | hq2 | hq3
  · have hqv0' : qv = 0#i128 := by
      apply IScalar.eq_of_val_eq; rw [hq0]; norm_num
    have hkm : (k.val : ℝ) = 4 * (m : ℝ) := by rw [hm, hq0]; push_cast; ring
    refine ⟨sc, ?_, ?_, ?_⟩
    · rw [hqv0']
      split
      · rfl
      · exfalso
        have hv := congrArg IScalar.val ‹(0#i128 : I128) = 1#i128›
        norm_num at hv
      · exfalso
        have hv := congrArg IScalar.val ‹(0#i128 : I128) = 2#i128›
        norm_num at hv
      · exact absurd rfl ‹(0#i128 : I128) = 0#i128 → False›
    · rw [hX, sin_quad_0 Y del k.val m hkm]
      calc |toReal sc.sin - Real.sin (Y + del)|
          ≤ |toReal sc.sin - Real.sin Y| + |Real.sin Y - Real.sin (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.sin Y - Real.sin (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := sin_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrS]
        _ ≤ epsExp * |Real.sin (Y + del)| + trig_floor := by simpa using trig_budget _
    · rw [hX, cos_quad_0 Y del k.val m hkm]
      calc |toReal sc.cos - Real.cos (Y + del)|
          ≤ |toReal sc.cos - Real.cos Y| + |Real.cos Y - Real.cos (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.cos Y - Real.cos (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := cos_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrC]
        _ ≤ epsExp * |Real.cos (Y + del)| + trig_floor := by simpa using trig_budget _
  · have hqv1' : qv = 1#i128 := by
      apply IScalar.eq_of_val_eq; rw [hq1]; norm_num
    have hkm : (k.val : ℝ) = 4 * (m : ℝ) + 1 := by rw [hm, hq1]; push_cast; ring
    refine ⟨{ sin := sc.cos, cos := n1 }, ?_, ?_, ?_⟩
    · rw [hqv1']
      split
      · exfalso
        have hv := congrArg IScalar.val ‹(1#i128 : I128) = 0#i128›
        norm_num at hv
      · rw [hn1]; simp only [bind_tc_ok]
      · exfalso
        have hv := congrArg IScalar.val ‹(1#i128 : I128) = 2#i128›
        norm_num at hv
      · exact absurd rfl ‹(1#i128 : I128) = 1#i128 → False›
    · rw [hX, sin_quad_1 Y del k.val m hkm]
      calc |toReal sc.cos - Real.cos (Y + del)|
          ≤ |toReal sc.cos - Real.cos Y| + |Real.cos Y - Real.cos (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.cos Y - Real.cos (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := cos_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrC]
        _ ≤ epsExp * |Real.cos (Y + del)| + trig_floor := by simpa using trig_budget _
    · rw [hX, cos_quad_1 Y del k.val m hkm]
      calc |toReal n1 - -Real.sin (Y + del)|
          = |toReal sc.sin - Real.sin (Y + del)| := by
            rw [hn1t]
            rw [show -toReal sc.sin - -Real.sin (Y + del)
                = -(toReal sc.sin - Real.sin (Y + del)) by ring, abs_neg]
        _ ≤ |toReal sc.sin - Real.sin Y| + |Real.sin Y - Real.sin (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.sin Y - Real.sin (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := sin_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrS]
        _ ≤ epsExp * |(-Real.sin (Y + del))| + trig_floor := by simpa using trig_budget _
  · have hqv2' : qv = 2#i128 := by
      apply IScalar.eq_of_val_eq; rw [hq2]; norm_num
    have hkm : (k.val : ℝ) = 4 * (m : ℝ) + 2 := by rw [hm, hq2]; push_cast; ring
    refine ⟨{ sin := n1, cos := n2 }, ?_, ?_, ?_⟩
    · rw [hqv2']
      split
      · exfalso
        have hv := congrArg IScalar.val ‹(2#i128 : I128) = 0#i128›
        norm_num at hv
      · exfalso
        have hv := congrArg IScalar.val ‹(2#i128 : I128) = 1#i128›
        norm_num at hv
      · rw [hn1]; simp only [bind_tc_ok]; rw [hn2]; simp only [bind_tc_ok]
      · exact absurd rfl ‹(2#i128 : I128) = 2#i128 → False›
    · rw [hX, sin_quad_2 Y del k.val m hkm]
      calc |toReal n1 - -Real.sin (Y + del)|
          = |toReal sc.sin - Real.sin (Y + del)| := by
            rw [hn1t]
            rw [show -toReal sc.sin - -Real.sin (Y + del)
                = -(toReal sc.sin - Real.sin (Y + del)) by ring, abs_neg]
        _ ≤ |toReal sc.sin - Real.sin Y| + |Real.sin Y - Real.sin (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.sin Y - Real.sin (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := sin_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrS]
        _ ≤ epsExp * |(-Real.sin (Y + del))| + trig_floor := by simpa using trig_budget _
    · rw [hX, cos_quad_2 Y del k.val m hkm]
      calc |toReal n2 - -Real.cos (Y + del)|
          = |toReal sc.cos - Real.cos (Y + del)| := by
            rw [hn2t]
            rw [show -toReal sc.cos - -Real.cos (Y + del)
                = -(toReal sc.cos - Real.cos (Y + del)) by ring, abs_neg]
        _ ≤ |toReal sc.cos - Real.cos Y| + |Real.cos Y - Real.cos (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.cos Y - Real.cos (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := cos_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrC]
        _ ≤ epsExp * |(-Real.cos (Y + del))| + trig_floor := by simpa using trig_budget _
  · have hqv3' : qv = 3#i128 := by
      apply IScalar.eq_of_val_eq; rw [hq3]; norm_num
    have hkm : (k.val : ℝ) = 4 * (m : ℝ) + 3 := by rw [hm, hq3]; push_cast; ring
    refine ⟨{ sin := n2, cos := sc.sin }, ?_, ?_, ?_⟩
    · rw [hqv3']
      split
      · exfalso
        have hv := congrArg IScalar.val ‹(3#i128 : I128) = 0#i128›
        norm_num at hv
      · exfalso
        have hv := congrArg IScalar.val ‹(3#i128 : I128) = 1#i128›
        norm_num at hv
      · exfalso
        have hv := congrArg IScalar.val ‹(3#i128 : I128) = 2#i128›
        norm_num at hv
      · rw [hn2]; simp only [bind_tc_ok]
    · rw [hX, sin_quad_3 Y del k.val m hkm]
      calc |toReal n2 - -Real.cos (Y + del)|
          = |toReal sc.cos - Real.cos (Y + del)| := by
            rw [hn2t]
            rw [show -toReal sc.cos - -Real.cos (Y + del)
                = -(toReal sc.cos - Real.cos (Y + del)) by ring, abs_neg]
        _ ≤ |toReal sc.cos - Real.cos Y| + |Real.cos Y - Real.cos (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.cos Y - Real.cos (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := cos_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrC]
        _ ≤ epsExp * |(-Real.cos (Y + del))| + trig_floor := by simpa using trig_budget _
    · rw [hX, cos_quad_3 Y del k.val m hkm]
      calc |toReal sc.sin - Real.sin (Y + del)|
          ≤ |toReal sc.sin - Real.sin Y| + |Real.sin Y - Real.sin (Y + del)| :=
            abs_sub_le_add _ _ _
        _ ≤ (2 : ℝ) ^ (-23 : ℤ) + (2 : ℝ) ^ (-30 : ℤ) := by
            have h2 : |Real.sin Y - Real.sin (Y + del)| ≤ (2 : ℝ) ^ (-30 : ℤ) := by
              have h := sin_shift_le Y del
              rw [abs_sub_comm] at h
              exact le_trans h hdel
            linarith [hserrS]
        _ ≤ epsExp * |Real.sin (Y + del)| + trig_floor := by simpa using trig_budget _

/-- Refinement on the window: success with the combined absolute/relative bound
`2^-20·|f (toReal x)| + 2^-20` per output. -/
theorem fixed_sin_cos_refines_proved : fixed_sin_cos_refines := by
  intro x hx
  obtain ⟨sc, hsc, h1, h2⟩ := trigPastGuards_refines x hx
  exact ⟨sc, by rw [word_fixed_sin_cos_window_past_guards x hx]; exact hsc, h1, h2⟩

end

end apeiron_word
