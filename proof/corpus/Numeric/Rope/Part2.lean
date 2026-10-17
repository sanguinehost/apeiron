/-
Integer specifications of the extracted Q32.32 helpers of `src/numeric/fixed_trig.rs`: `narrow`,
`q_rne`, `q_mul` and `div_fac`. Each theorem states the *bound* of the extracted body, not a
restatement of its code:

* `narrow_spec`: in the `i64` range the narrowing is the identity;
* `q_rne_spec`: the rounded division returns an integer within one unit of the true quotient;
* `q_mul_spec`: a rounded Q32.32 product is within `trig_mu` of the real product;
* `div_fac_spec`: a term divided by its exact factorial is within `trig_mu` of the real quotient.

The per-step rounding budget is `trig_mu = 2^-30`, twice the coarse `q_rne` error `2^-32` and far
above the half-ulp the nearest-even body actually attains; the accumulation through the twelve
products and thirteen divisions of the Taylor body is `Numeric.Rope.Part3`.
-/
import Numeric.Rope.Part1

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- Per-operation rounding budget of `q_mul`/`div_fac`, `2^-30`. -/
def trig_mu : ℝ := (2 : ℝ) ^ (-30 : ℤ)

theorem trig_mu_pos : (0 : ℝ) < trig_mu := by unfold trig_mu; positivity

/-- `bmod 2^64` is the identity on the `i64` range. -/
theorem int_bmod_self_64 (a : ℤ) (hlo : -2 ^ 63 ≤ a) (hhi : a ≤ 2 ^ 63 - 1) :
    a.bmod (2 ^ 64) = a := by
  by_cases h : 0 ≤ a
  · rw [Int.bmod_eq_emod_of_lt]
    · exact Int.emod_eq_of_lt h (by norm_num; omega)
    · rw [Int.emod_eq_of_lt h (by norm_num; omega)]; norm_num; omega
  · rw [Int.bmod_neg]
    · have hem : a % ((2 ^ 64 : ℕ) : ℤ) = a + ((2 ^ 64 : ℕ) : ℤ) := by
        have h1 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a + ((2 ^ 64 : ℕ) : ℤ) :=
          Int.emod_eq_of_lt (by omega) (by omega)
        have h2 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a % ((2 ^ 64 : ℕ) : ℤ) := Int.add_emod_right a ((2 ^ 64 : ℕ) : ℤ)
        omega
      rw [hem]; omega
    · have hem : a % ((2 ^ 64 : ℕ) : ℤ) = a + ((2 ^ 64 : ℕ) : ℤ) := by
        have h1 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a + ((2 ^ 64 : ℕ) : ℤ) :=
          Int.emod_eq_of_lt (by omega) (by omega)
        have h2 : (a + ((2 ^ 64 : ℕ) : ℤ)) % ((2 ^ 64 : ℕ) : ℤ)
            = a % ((2 ^ 64 : ℕ) : ℤ) := Int.add_emod_right a ((2 ^ 64 : ℕ) : ℤ)
        omega
      rw [hem]; norm_num; omega

/-! ## `narrow` -/

/-- In the `i64` range the extracted narrowing returns the value unchanged. -/
theorem narrow_spec (v : I128) (hlo : I64.min ≤ v.val) (hhi : v.val ≤ I64.max) :
    ∃ w : I64, fixed_trig.narrow v = ok (some w) ∧ w.val = v.val := by
  have hmaxv : (core.num.I64.MAX).val = I64.max := by
    rw [core.num.I64.MAX, IScalar.ofInt_val_eq, I64.rMax, I64.max_eq]
  have hminv : (core.num.I64.MIN).val = I64.min := by
    rw [core.num.I64.MIN, IScalar.ofInt_val_eq, I64.rMin, I64.min_eq]
  have h1 : ¬ v > (core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    rw [gt_iff_lt, IScalar.lt_equiv, core.convert.num.FromI128I64.from_val_eq, hmaxv]
    exact not_lt.mpr hhi
  have h2 : ¬ v < (core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    rw [IScalar.lt_equiv, core.convert.num.FromI128I64.from_val_eq, hminv]
    exact not_lt.mpr hlo
  refine ⟨IScalar.cast .I64 v, ?_, ?_⟩
  · unfold fixed_trig.narrow
    simp only [lift, bind_tc_ok, if_neg h1, if_neg h2]
  · rw [IScalar.cast_val_eq]
    exact int_bmod_self_64 v.val (by rw [I64.min_eq] at hlo; omega)
      (by rw [I64.max_eq] at hhi; omega)

lemma q_rne_step_up (num den q r : ℤ) (_hden : 0 < den)
    (hn : num = q * den + r) (hr : 0 < r) (hr2 : r < den) :
    |(q + 1) * den - num| ≤ den := by
  have h : (q + 1) * den - num = den - r := by rw [hn]; ring
  rw [h, abs_of_nonneg (by omega)]
  omega

lemma q_rne_step_down (num den q r : ℤ) (_hden : 0 < den)
    (hn : num = q * den + r) (hr : r < 0) (hr1 : -den < r) :
    |(q - 1) * den - num| ≤ den := by
  have h : (q - 1) * den - num = -(den + r) := by rw [hn]; ring
  rw [h, abs_neg, abs_of_pos (by omega)]
  omega

lemma q_rne_step_zero (num den q r : ℤ) (_hden : 0 < den)
    (hn : num = q * den + r) (hr1 : -den < r) (hr2 : r < den) :
    |q * den - num| ≤ den := by
  have h : q * den - num = -r := by rw [hn]; ring
  rw [h, abs_neg, abs_le]
  omega

theorem q_rne_spec (num den : I128) (hden : 0 < den.val)
    (hnum : |num.val| ≤ 2 ^ 120) (hden2 : den.val ≤ 2 ^ 120) :
    ∃ q : I128, fixed_trig.q_rne num den = ok (some q) ∧
      |q.val * den.val - num.val| ≤ den.val := by
  have hle : ¬ (den <= 0#i128) := by
    rw [IScalar.le_equiv]
    simp
    omega
  have h : fixed_trig.q_rne num den ⦃ z => ∃ q : I128, z = some q ∧
      |q.val * den.val - num.val| ≤ den.val ⦄ := by
    have hden0 : den.val ≠ 0 := by omega
    obtain ⟨qq, hqq, hqqv⟩ := IScalar.div_spec (ty := .I128) (x := num) (y := den) hden0
      (by rintro ⟨h1, h2⟩; rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq] at h1; omega)
    obtain ⟨rr, hrr, hrrv⟩ := WP.spec_imp_exists
      (IScalar.rem_spec (ty := .I128) (x := num) (y := den) hden0)
    have hrbound : -den.val < rr.val ∧ rr.val < den.val := by
      constructor
      · rw [hrrv]
        have h1 := Int.tmod_lt_of_pos (-num.val) hden
        rw [Int.neg_tmod] at h1
        omega
      · rw [hrrv]
        exact Int.tmod_lt_of_pos _ hden
    have hn : num.val = qq.val * den.val + rr.val := by
      have h1 := Int.tdiv_mul_add_tmod num.val den.val
      rw [← hqqv, ← hrrv] at h1
      omega
    have hqabs : |qq.val| ≤ 2 ^ 120 := by
      have h1 : (qq.val.natAbs : ℤ) ≤ (num.val.natAbs : ℤ) := by
        rw [hqqv]
        exact_mod_cast Int.natAbs_tdiv_le_natAbs num.val den.val
      have h2 : (num.val.natAbs : ℤ) = |num.val| := by
        rcases le_or_gt 0 num.val with h | h
        · rw [Int.ofNat_natAbs_of_nonneg h, abs_of_nonneg h]
        · rw [Int.ofNat_natAbs_of_nonpos (le_of_lt h), abs_of_neg h]
      have h3 : (qq.val.natAbs : ℤ) = |qq.val| := by
        rcases le_or_gt 0 qq.val with h | h
        · rw [Int.ofNat_natAbs_of_nonneg h, abs_of_nonneg h]
        · rw [Int.ofNat_natAbs_of_nonpos (le_of_lt h), abs_of_neg h]
      omega
    obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
      (IScalar.mul_spec (ty := .I128) (x := rr) (y := 2#i128)
        (by rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq]; simp; omega)
        (by rw [IScalar.max_IScalarTy_I128_eq, I128.max_eq]; simp; omega))
    have h2v : (2#i128).val = 2 := by norm_num
    have hw2 : w.val = 2 * rr.val := by rw [hwv, h2v]; ring
    have hcm : I128.checked_mul rr 2#i128 = some w := by
      show Option.ofResult (rr.mul 2#i128) = some w
      rw [show rr.mul 2#i128 = ok w from hw]
      rfl
    unfold fixed_trig.q_rne
    rw [if_neg hle]
    rw [hqq]
    simp only [bind_tc_ok]
    rw [hrr]
    simp only [bind_tc_ok]
    by_cases hrz : rr = 0#i128
    · rw [if_pos hrz]
      have hr0 : rr.val = 0 := by rw [hrz]; rfl
      refine ⟨qq, rfl, ?_⟩
      rw [hn, hr0]
      simp
      omega
    · rw [if_neg hrz]
      rw [hcm]
      simp only [lift, bind_tc_ok]
      by_cases h1 : w > den
      · rw [if_pos h1]
        simp only [bind_tc_ok]
        by_cases h2 : rr > 0#i128
        · rw [if_pos h2]
          step as ⟨i, hi⟩
          refine ⟨i, rfl, ?_⟩
          rw [hi]
          exact q_rne_step_up num.val den.val qq.val rr.val hden hn
            (by have h := h2; rw [gt_iff_lt, IScalar.lt_equiv] at h; exact h) hrbound.2
        · rw [if_neg h2]
          step as ⟨i, hi⟩
          exfalso
          have h1v : den.val < w.val := by have h := h1; rw [gt_iff_lt, IScalar.lt_equiv] at h; exact h
          have h2v' : rr.val ≤ 0 := by
            have h := h2; rw [gt_iff_lt, IScalar.lt_equiv] at h
            have h0 : (0#i128).val = 0 := by norm_num
            rw [h0] at h
            omega
          omega
      · rw [if_neg h1]
        step as ⟨d, hd⟩
        by_cases h3 : w < d
        · rw [if_pos h3]
          simp only [bind_tc_ok]
          by_cases h2 : rr > 0#i128
          · rw [if_pos h2]
            step as ⟨i, hi⟩
            refine ⟨i, rfl, ?_⟩
            rw [hi]
            exact q_rne_step_up num.val den.val qq.val rr.val hden hn
              (by have h := h2; rw [gt_iff_lt, IScalar.lt_equiv] at h; exact h) hrbound.2
          · rw [if_neg h2]
            step as ⟨i, hi⟩
            refine ⟨i, rfl, ?_⟩
            rw [hi]
            have h3v : w.val < d.val := by rwa [IScalar.lt_equiv] at h3
            simpa using q_rne_step_down num.val den.val qq.val rr.val hden hn
              (by omega) hrbound.1
        · rw [if_neg h3]
          by_cases h4 : w = den
          · rw [if_pos h4]
            step as ⟨i1, hi1⟩
            have h4v : w.val = den.val := by rw [h4]
            have h2v : 0 < rr.val := by omega
            have h2 : rr > 0#i128 := by
              rw [gt_iff_lt, IScalar.lt_equiv]
              have h0 : (0#i128).val = 0 := by norm_num
              rw [h0]
              exact h2v
            by_cases h6 : (i1 != 0#i128) = true
            · rw [if_pos h6]
              simp only [bind_tc_ok]
              rw [if_pos h2]
              step as ⟨i, hi⟩
              refine ⟨i, rfl, ?_⟩
              rw [hi]
              exact q_rne_step_up num.val den.val qq.val rr.val hden hn h2v hrbound.2
            · rw [if_neg h6]
              simp only [bind_tc_ok]
              rw [if_pos h2]
              step as ⟨i, hi⟩
              refine ⟨i, rfl, ?_⟩
              rw [hi]
              first
              | exact q_rne_step_up num.val den.val qq.val rr.val hden hn h2v hrbound.2
              | simpa using q_rne_step_zero num.val den.val qq.val rr.val hden hn hrbound.1 hrbound.2
          · rw [if_neg h4]
            by_cases h5 : w = d
            · rw [if_pos h5]
              step as ⟨i1, hi1⟩
              have h5v : w.val = d.val := by rw [h5]
              have h2v : rr.val < 0 := by omega
              have h2 : ¬ rr > 0#i128 := by
                rw [gt_iff_lt, IScalar.lt_equiv]
                have h0 : (0#i128).val = 0 := by norm_num
                rw [h0]
                omega
              by_cases h6 : (i1 != 0#i128) = true
              · rw [if_pos h6]
                simp only [bind_tc_ok]
                rw [if_neg h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                simpa using q_rne_step_down num.val den.val qq.val rr.val hden hn h2v hrbound.1
              · rw [if_neg h6]
                simp only [bind_tc_ok]
                rw [if_neg h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                first
                | simpa using q_rne_step_down num.val den.val qq.val rr.val hden hn h2v hrbound.1
                | simpa using q_rne_step_zero num.val den.val qq.val rr.val hden hn hrbound.1 hrbound.2
            · rw [if_neg h5]
              simp only [bind_tc_ok]
              by_cases h2 : rr > 0#i128
              · rw [if_pos h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                simpa using q_rne_step_zero num.val den.val qq.val rr.val hden hn hrbound.1 hrbound.2
              · rw [if_neg h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                simpa using q_rne_step_zero num.val den.val qq.val rr.val hden hn hrbound.1 hrbound.2
  obtain ⟨y, hy, q, hyq, hq⟩ := (WP.spec_equiv_exists _ _).1 h
  exact ⟨q, by rw [hy, hyq], hq⟩

/-! ## `q_mul` and `div_fac` -/

/-- A rounded Q32.32 product of two grid values of magnitude at most `2^40` is within `trig_mu`
of the real product: the `i128` product is exact, `q_rne` is within one unit of the quotient and
`narrow` is the identity on the result. -/
theorem q_mul_spec (a b : I64) (ha : |a.val| ≤ 2 ^ 40) (hb : |b.val| ≤ 2 ^ 40) :
    ∃ w : I64, fixed_trig.q_mul a b = ok (some w) ∧
      |toReal w - toReal a * toReal b| ≤ trig_mu := by
  have hprod : |a.val * b.val| ≤ 2 ^ 80 := by
    rw [abs_mul]
    nlinarith [abs_nonneg a.val, abs_nonneg b.val]
  have hmain : fixed_trig.q_mul a b ⦃ z => ∃ w : I64, z = some w ∧
      |toReal w - toReal a * toReal b| ≤ trig_mu ⦄ := by
    unfold fixed_trig.q_mul
    simp only [lift, bind_tc_ok]
    step as ⟨i2, hi2⟩
    case a =>
      rw [fixed_trig_ONE_eq]
      simp only [bind_tc_ok]
      have hi2v : i2.val = a.val * b.val := by
        rw [hi2, core.convert.num.FromI128I64.from_val_eq,
          core.convert.num.FromI128I64.from_val_eq]
      have h32v : (4294967296#i64).val = 4294967296 := by norm_num
      obtain ⟨v, hv, hve⟩ := q_rne_spec i2
        (core.convert.num.FromI128I64.from (4294967296#i64))
        (by rw [core.convert.num.FromI128I64.from_val_eq, h32v]; norm_num)
        (by
          rw [hi2, core.convert.num.FromI128I64.from_val_eq,
            core.convert.num.FromI128I64.from_val_eq]
          omega)
        (by rw [core.convert.num.FromI128I64.from_val_eq, h32v]; norm_num)
      rw [hv]
      simp only [bind_tc_ok]
      have hF : (core.convert.num.FromI128I64.from (4294967296#i64)).val
          = 4294967296 := by
        rw [core.convert.num.FromI128I64.from_val_eq, h32v]
      have hi2abs : |i2.val| ≤ 1208925819614629174706176 := by
        rw [hi2v]
        have hp := hprod
        rw [show ((2 : ℤ) ^ 80) = 1208925819614629174706176 by norm_num] at hp
        exact hp
      have hstep : |(v.val * 4294967296 - i2.val : ℤ)| ≤ 4294967296 := by
        have h := hve
        rw [core.convert.num.FromI128I64.from_val_eq, h32v] at h
        exact h
      have hvabs : |v.val| ≤ 281474976710657 := by
        have h3 : |v.val * 4294967296| ≤ 4294967296 + 1208925819614629174706176 := by
          have hsplit : |(v.val * 4294967296 : ℤ)|
              = |((v.val * 4294967296 - i2.val) + i2.val)| := by ring_nf
          rw [hsplit]
          calc |(v.val * 4294967296 - i2.val) + i2.val|
              ≤ |v.val * 4294967296 - i2.val| + |i2.val| := abs_add_le _ _
            _ ≤ 4294967296 + 1208925819614629174706176 := by linarith
        rw [abs_mul] at h3
        have h4 : |(4294967296 : ℤ)| = 4294967296 := by norm_num
        rw [h4] at h3
        omega
      have hvbound : I64.min ≤ v.val ∧ v.val ≤ I64.max := by
        obtain ⟨hlo, hhi⟩ := abs_le.mp hvabs
        rw [I64.min_eq, I64.max_eq]
        omega
      obtain ⟨w, hw, hwv⟩ := narrow_spec v hvbound.1 hvbound.2
      rw [hw]
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨w, rfl, ?_⟩
      have hcast : |(v.val : ℝ) * (2 : ℝ) ^ 32 - (i2.val : ℝ)| ≤ (2 : ℝ) ^ 32 := by
        have h := hve
        rw [core.convert.num.FromI128I64.from_val_eq, h32v] at h
        have h' : |((v.val * 4294967296 - i2.val : ℤ) : ℝ)| ≤ (4294967296 : ℝ) := by
          exact_mod_cast h
        have h4 : ((v.val * 4294967296 - i2.val : ℤ) : ℝ)
            = (v.val : ℝ) * (2 : ℝ) ^ 32 - (i2.val : ℝ) := by
          push_cast
          norm_num
        rw [h4] at h'
        norm_num at h' ⊢
        exact h'
      have hwR : (w.val : ℝ) = v.val := by rw [hwv]
      rw [toReal, toReal, toReal, hwR]
      have hq : q32 = (2 : ℝ) ^ 32 := by unfold q32; norm_num
      rw [hq]
      have hi2R : (i2.val : ℝ) = (a.val : ℝ) * b.val := by
        rw [hi2v]; push_cast; ring
      have hkey : (v.val : ℝ) / (2 : ℝ) ^ 32 -
            ((a.val : ℝ) / (2 : ℝ) ^ 32) * ((b.val : ℝ) / (2 : ℝ) ^ 32)
          = ((v.val : ℝ) * (2 : ℝ) ^ 32 - (a.val : ℝ) * b.val) / ((2 : ℝ) ^ 32) ^ 2 := by
        field_simp
      rw [hkey, ← hi2R, abs_div, abs_of_pos (by positivity : (0 : ℝ) < ((2 : ℝ) ^ 32) ^ 2)]
      have hpow : ((2 : ℝ) ^ 32) ^ 2 = (2 : ℝ) ^ 64 := by norm_num
      rw [hpow]
      have hdiv : |(v.val : ℝ) * (2 : ℝ) ^ 32 - (i2.val : ℝ)| / (2 : ℝ) ^ 64
          ≤ (2 : ℝ) ^ 32 / (2 : ℝ) ^ 64 := by
        gcongr
      have h32 : (2 : ℝ) ^ 32 / (2 : ℝ) ^ 64 = (2 : ℝ) ^ (-32 : ℤ) := by norm_num
      rw [h32] at hdiv
      have hmono : (2 : ℝ) ^ (-32 : ℤ) ≤ (2 : ℝ) ^ (-30 : ℤ) := by norm_num
      have hmu : trig_mu = (2 : ℝ) ^ (-30 : ℤ) := rfl
      rw [hmu]
      linarith [hdiv, hmono]
  obtain ⟨z, hz, w, hw, hP⟩ := (WP.spec_equiv_exists _ _).1 hmain
  exact ⟨w, by rw [hz, hw], hP⟩

theorem div_fac_spec (p f : I64) (hf : 0 < f.val) (hp : |p.val| ≤ 2 ^ 62)
    (hf2 : f.val ≤ 2 ^ 62) :
    ∃ w : I64, fixed_trig.div_fac p f = ok (some w) ∧
      |toReal w - toReal p / (f.val : ℝ)| ≤ trig_mu := by
  have hnum : |(p.val : ℤ)| ≤ 2 ^ 120 := by
    have h120 : ((2 : ℤ) ^ 62) ≤ ((2 : ℤ) ^ 120) := by norm_num
    linarith [hp, abs_nonneg p.val, h120]
  have hden2 : (f.val : ℤ) ≤ 2 ^ 120 := by
    have h120 : ((2 : ℤ) ^ 62) ≤ ((2 : ℤ) ^ 120) := by norm_num
    linarith [hf2, h120]
  have hmain : fixed_trig.div_fac p f ⦃ z => ∃ w : I64, z = some w ∧
      |toReal w - toReal p / (f.val : ℝ)| ≤ trig_mu ⦄ := by
    unfold fixed_trig.div_fac
    simp only [lift, bind_tc_ok]
    obtain ⟨v, hv, hve⟩ := q_rne_spec
      (core.convert.num.FromI128I64.from p) (core.convert.num.FromI128I64.from f)
      (by rw [core.convert.num.FromI128I64.from_val_eq]; exact hf)
      (by rw [core.convert.num.FromI128I64.from_val_eq]; exact hnum)
      (by rw [core.convert.num.FromI128I64.from_val_eq]; exact hden2)
    rw [hv]
    simp only [bind_tc_ok]
    have hstep : |(v.val * f.val - p.val : ℤ)| ≤ f.val := by
      have h := hve
      rwa [core.convert.num.FromI128I64.from_val_eq,
        core.convert.num.FromI128I64.from_val_eq] at h
    have hfR : (0 : ℝ) < (f.val : ℝ) := by exact_mod_cast hf
    have hqR : |(v.val : ℝ)| ≤ 4611686018427387905 := by
      have hpR : |(p.val : ℝ)| ≤ 4611686018427387904 := by
        have h : ((2 : ℤ) ^ 62) = 4611686018427387904 := by norm_num
        rw [h] at hp
        exact_mod_cast hp
      have h3 : |(v.val : ℝ) * (f.val : ℝ)| ≤ (f.val : ℝ) + 4611686018427387904 := by
        have hsplit : |(v.val : ℝ) * (f.val : ℝ)|
            = |((v.val : ℝ) * f.val - p.val) + p.val| := by ring_nf
        rw [hsplit]
        calc |((v.val : ℝ) * f.val - p.val) + (p.val : ℝ)|
            ≤ |(v.val : ℝ) * f.val - (p.val : ℝ)| + |(p.val : ℝ)| := abs_add_le _ _
          _ ≤ (f.val : ℝ) + 4611686018427387904 := by
              have hcast : |(v.val : ℝ) * f.val - (p.val : ℝ)| ≤ (f.val : ℝ) := by
                exact_mod_cast hstep
              linarith
      rw [abs_mul] at h3
      have hfabs : |(f.val : ℝ)| = (f.val : ℝ) := abs_of_pos hfR
      rw [hfabs] at h3
      have hf1 : (1 : ℝ) ≤ (f.val : ℝ) := by exact_mod_cast hf
      have hqnn : (0 : ℝ) ≤ |(v.val : ℝ)| := abs_nonneg _
      have hdiv : |(v.val : ℝ)| = (|(v.val : ℝ)| * (f.val : ℝ)) / (f.val : ℝ) := by
        field_simp
      rw [hdiv]
      have hmono : (|(v.val : ℝ)| * (f.val : ℝ)) / (f.val : ℝ)
          ≤ ((f.val : ℝ) + 4611686018427387904) / (f.val : ℝ) :=
        div_le_div_of_nonneg_right h3 (le_of_lt hfR)
      have hsplit2 : ((f.val : ℝ) + 4611686018427387904) / (f.val : ℝ)
          = 1 + 4611686018427387904 / (f.val : ℝ) := by
        field_simp
      rw [hsplit2] at hmono
      have hfrac : 4611686018427387904 / (f.val : ℝ) ≤ 4611686018427387904 := by
        rw [div_le_iff₀ hfR]
        nlinarith [hf1]
      linarith
    have hqZi : |v.val| ≤ 4611686018427387905 := by exact_mod_cast hqR
    have hvbound : I64.min ≤ v.val ∧ v.val ≤ I64.max := by
      obtain ⟨hlo, hhi⟩ := abs_le.mp hqZi
      rw [I64.min_eq, I64.max_eq]
      omega
    obtain ⟨w, hw, hwv⟩ := narrow_spec v hvbound.1 hvbound.2
    rw [hw]
    simp only [Aeneas.Std.WP.spec_ok]
    refine ⟨w, rfl, ?_⟩
    rw [toReal, toReal, hwv]
    have hq32 : q32 = (2 : ℝ) ^ 32 := by unfold q32; norm_num
    rw [hq32]
    have hkey : (v.val : ℝ) / (2 : ℝ) ^ 32 - ((p.val : ℝ) / (2 : ℝ) ^ 32) / (f.val : ℝ)
        = ((v.val : ℝ) * f.val - (p.val : ℝ)) / ((2 : ℝ) ^ 32 * (f.val : ℝ)) := by
      field_simp
    rw [hkey]
    have hdenpos : (0 : ℝ) < (2 : ℝ) ^ 32 * (f.val : ℝ) := by positivity
    rw [abs_div, abs_of_pos hdenpos]
    have hnum2 : |(v.val : ℝ) * f.val - (p.val : ℝ)| ≤ (f.val : ℝ) := by
      exact_mod_cast hstep
    have hmono := div_le_div_of_nonneg_right hnum2 (le_of_lt hdenpos)
    have heq : (f.val : ℝ) / ((2 : ℝ) ^ 32 * (f.val : ℝ)) = 1 / (2 : ℝ) ^ 32 := by
      field_simp
    rw [heq] at hmono
    have h32 : (1 : ℝ) / (2 : ℝ) ^ 32 = (2 : ℝ) ^ (-32 : ℤ) := by norm_num
    rw [h32] at hmono
    have hmono2 : (2 : ℝ) ^ (-32 : ℤ) ≤ (2 : ℝ) ^ (-30 : ℤ) := by norm_num
    have hmu : trig_mu = (2 : ℝ) ^ (-30 : ℤ) := rfl
    rw [hmu]
    linarith [hmono, hmono2]
  obtain ⟨z, hz, w, hw, hP⟩ := (WP.spec_equiv_exists _ _).1 hmain
  exact ⟨w, by rw [hz, hw], hP⟩

