/-
Argument-reduction constants of the extracted `fixed_trig.fixed_sin_cos`
(`src/numeric/fixed_trig.rs` lines 193–228; `proof/ApeironWord.lean`,
`fixed_trig.fixed_sin_cos`):

* `pi_red_bound`: the Q64 half-turn `HALF_PI_Q64 = 28_976_077_832_308_490_240` is within
  `1200` ulps (`1200·2^-64 ≈ 6.5e-17`) of `Real.pi / 2`; the true offset is `1129.5` ulps low,
  which the control (`1200 → 1000`) shows is load-bearing;
* `k_bound`: on `trig_window x` the extracted quotient `k = q_rne(x <<< 32, HALF_PI_Q64)` obeys
  `|k| ≤ 2608`;
* `reduction_spec`: every checked `i128` operation of the reduction head succeeds, and the
  narrowed remainder `r` satisfies `|r - (x - k·pi/2)| ≤ 2^-30` and `|r| ≤ 1`.

`q_rne_half_spec` is the nearest-even half-divisor bound. `Numeric.Rope.Part2`'s `q_rne_spec` is
deliberately coarse (`|q·den - num| ≤ den`); the two-sided remainder bound `|2·(q·den - num)| ≤ den`
that `|r| ≤ 1` needs is proved here from the same extracted body.

Grade: `proved` for each headline (kernel check, closed axiom cone). Machine, metric and interval:
none — these are exact integer bounds, not measurements.
-/
import Numeric.Rope.Part2

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The `i128` module and the Q64 denotation -/

/-- `bmod 2^128` is the identity on the `i128` value range. -/
theorem int_bmod_self_128 (a : ℤ) (hlo : -2 ^ 127 ≤ a) (hhi : a ≤ 2 ^ 127 - 1) :
    a.bmod (2 ^ 128) = a := by
  by_cases h : 0 ≤ a
  · rw [Int.bmod_eq_emod_of_lt]
    · exact Int.emod_eq_of_lt h (by norm_num; omega)
    · rw [Int.emod_eq_of_lt h (by norm_num; omega)]; norm_num; omega
  · rw [Int.bmod_neg]
    · have hem : a % ((2 ^ 128 : ℕ) : ℤ) = a + ((2 ^ 128 : ℕ) : ℤ) := by
        have h1 : (a + ((2 ^ 128 : ℕ) : ℤ)) % ((2 ^ 128 : ℕ) : ℤ)
            = a + ((2 ^ 128 : ℕ) : ℤ) :=
          Int.emod_eq_of_lt (by omega) (by omega)
        have h2 : (a + ((2 ^ 128 : ℕ) : ℤ)) % ((2 ^ 128 : ℕ) : ℤ)
            = a % ((2 ^ 128 : ℕ) : ℤ) := Int.add_emod_right a ((2 ^ 128 : ℕ) : ℤ)
        omega
      rw [hem]; omega
    · have hem : a % ((2 ^ 128 : ℕ) : ℤ) = a + ((2 ^ 128 : ℕ) : ℤ) := by
        have h1 : (a + ((2 ^ 128 : ℕ) : ℤ)) % ((2 ^ 128 : ℕ) : ℤ)
            = a + ((2 ^ 128 : ℕ) : ℤ) :=
          Int.emod_eq_of_lt (by omega) (by omega)
        have h2 : (a + ((2 ^ 128 : ℕ) : ℤ)) % ((2 ^ 128 : ℕ) : ℤ)
            = a % ((2 ^ 128 : ℕ) : ℤ) := Int.add_emod_right a ((2 ^ 128 : ℕ) : ℤ)
        omega
      rw [hem]; norm_num; omega

/-- The extracted `HALF_PI_Q64` value, `round(pi/2 · 2^64)` as a Q64 integer. -/
theorem halfPiQ64_val : fixed_trig.HALF_PI_Q64.val = 28976077832308490240 := by
  rw [fixed_trig.HALF_PI_Q64]
  norm_num

/-- The Q64 denotation of an `i128` word: `v / 2^64`. -/
def toRealQ64 (v : I128) : ℝ := (v.val : ℝ) / (2 : ℝ) ^ 64

/-! ## `pi_red_bound` -/

/-- **`pi_red_bound`.** The extracted Q64 half-turn is within `1200` ulps of `pi/2`.
`Mathlib`'s 21-digit bounds (`Real.pi_gt_d20`, `Real.pi_lt_d20`) resolve the 1129.5-ulp offset:
`1200·2^-64 = 6.51e-17`, well inside the 21-digit interval. -/
theorem pi_red_bound :
    |toRealQ64 fixed_trig.HALF_PI_Q64 - Real.pi / 2| ≤ (1200 : ℝ) * (2 : ℝ) ^ (-64 : ℤ) := by
  have hP : fixed_trig.HALF_PI_Q64.val = 28976077832308490240 := halfPiQ64_val
  have hp1 : 3.14159265358979323846 < Real.pi := Real.pi_gt_d20
  have hp2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  have hc1 : 2 * ((28976077832308490240 : ℝ) - 1200) / (2 : ℝ) ^ 64 <
      3.14159265358979323846 := by norm_num
  have hc2 : 3.14159265358979323847 <
      2 * ((28976077832308490240 : ℝ) + 1200) / (2 : ℝ) ^ 64 := by norm_num
  have he1 : 2 * ((28976077832308490240 : ℝ) - 1200) ≤ (2 : ℝ) ^ 64 * Real.pi := by
    linarith
  have he2 : (2 : ℝ) ^ 64 * Real.pi ≤ 2 * ((28976077832308490240 : ℝ) + 1200) := by
    linarith
  have hlow : (-1200 : ℝ) ≤ (28976077832308490240 : ℝ) - (2 : ℝ) ^ 63 * Real.pi := by
    nlinarith [he2]
  have hupp : (28976077832308490240 : ℝ) - (2 : ℝ) ^ 63 * Real.pi ≤ 1200 := by
    nlinarith [he1]
  have hkey : toRealQ64 fixed_trig.HALF_PI_Q64 - Real.pi / 2
      = ((28976077832308490240 : ℝ) - (2 : ℝ) ^ 63 * Real.pi) / (2 : ℝ) ^ 64 := by
    rw [toRealQ64, hP]
    field_simp
    ring
  rw [hkey, abs_le]
  constructor
  · have h : (-1200 : ℝ) / (2 : ℝ) ^ 64 ≤
        ((28976077832308490240 : ℝ) - (2 : ℝ) ^ 63 * Real.pi) / (2 : ℝ) ^ 64 :=
      div_le_div_of_nonneg_right hlow (by positivity)
    have he : -(1200 * (2 : ℝ) ^ (-64 : ℤ)) = (-1200 : ℝ) / (2 : ℝ) ^ 64 := by norm_num
    rw [he]
    exact h
  · have h : ((28976077832308490240 : ℝ) - (2 : ℝ) ^ 63 * Real.pi) / (2 : ℝ) ^ 64 ≤
        (1200 : ℝ) / (2 : ℝ) ^ 64 :=
      div_le_div_of_nonneg_right hupp (by positivity)
    have he : (1200 : ℝ) * (2 : ℝ) ^ (-64 : ℤ) = (1200 : ℝ) / (2 : ℝ) ^ 64 := by norm_num
    rw [he]
    exact h

/-! ## The shift `x <<< 32` -/

/-- The extracted `x <<< 32` on the widened `i128` is `x·2^32` on the window: the `i128` result
is the `bmod 2^128` of `x·2^32`, and `|x·2^32| ≤ 2^76` is far inside that range. -/
theorem shift32_val (x : I64) (hx : trig_window x) (t : I128)
    (ht : (core.convert.num.FromI128I64.from x) <<< (32#i32) = ok t) :
    t.val = x.val * 2 ^ 32 := by
  rw [trig_window] at hx
  obtain ⟨t', ht', htval, _⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := core.convert.num.FromI128I64.from x) (y := 32#i32)
      (by norm_num) (by norm_num [I128.numBits]))
  have htt : t = t' := Result.ok_injective (ht.symm.trans ht')
  rw [htt, htval]
  rw [core.convert.num.FromI128I64.from_val_eq]
  have h32 : (32#i32).toNat = 32 := by decide
  rw [h32, Int.shiftLeft_eq]
  rw [show I128.size = 2 ^ 128 by norm_num [I128.size, I128.numBits]]
  exact int_bmod_self_128 _ (by omega) (by omega)

/-! ## Nearest-even division: the half-divisor bound

`Numeric.Rope.Part2`'s `q_rne_spec` bounds the remainder by `den`; the `|r| ≤ 1` half of
`reduction_spec` needs the two-sided bound `|q·den - num| ≤ den/2` that nearest-even actually
attains. The step lemmas below are the sharp form of Part2's `q_rne_step_*`.
-/

/-- Rounding up from a positive remainder beyond `den/2` lands within `den/2`. -/
lemma q_rne_half_up (num den q r : ℤ) (hn : num = q * den + r)
    (h2 : den < 2 * r) (hr : r < den) :
    |2 * ((q + 1) * den - num)| ≤ den := by
  have e : (q + 1) * den - num = den - r := by rw [hn]; ring
  rw [e, abs_le]
  constructor <;> omega

/-- Rounding down from a negative remainder beyond `-den/2` lands within `den/2`. -/
lemma q_rne_half_down (num den q r : ℤ) (hn : num = q * den + r)
    (h2 : 2 * r < -den) (hr : -den < r) :
    |2 * ((q - 1) * den - num)| ≤ den := by
  have e : (q - 1) * den - num = -(den + r) := by rw [hn]; ring
  rw [e, abs_le]
  constructor <;> omega

/-- Truncation at a remainder inside `[-den/2, den/2]` stays inside. -/
lemma q_rne_half_zero (num den q r : ℤ) (hn : num = q * den + r)
    (h1 : -den ≤ 2 * r) (h2 : 2 * r ≤ den) :
    |2 * (q * den - num)| ≤ den := by
  have e : q * den - num = -r := by rw [hn]; ring
  rw [e, abs_le]
  constructor <;> omega

/-- At the upper tie `2r = den` the truncation `q` and the round-up `q + 1` are each exactly
`den/2` away; `q - 1` is `3·den/2` away and is deliberately absent. -/
lemma q_rne_half_tie_pos (num den q r : ℤ) (hden : 0 < den) (hn : num = q * den + r)
    (h : 2 * r = den) :
    |2 * (q * den - num)| ≤ den ∧ |2 * ((q + 1) * den - num)| ≤ den := by
  have e1 : q * den - num = -r := by rw [hn]; ring
  have e2 : (q + 1) * den - num = den - r := by rw [hn]; ring
  rw [e1, e2]
  constructor <;> rw [abs_le] <;> omega

/-- At the lower tie `2r = -den` the truncation `q` and the round-down `q - 1` are each exactly
`den/2` away; `q + 1` is `3·den/2` away and is deliberately absent. -/
lemma q_rne_half_tie_neg (num den q r : ℤ) (hden : 0 < den) (hn : num = q * den + r)
    (h : 2 * r = -den) :
    |2 * (q * den - num)| ≤ den ∧ |2 * ((q - 1) * den - num)| ≤ den := by
  have e1 : q * den - num = -r := by rw [hn]; ring
  have e3 : (q - 1) * den - num = -(den + r) := by rw [hn]; ring
  rw [e1, e3]
  constructor <;> rw [abs_le] <;> omega

/-- **Nearest-even half-divisor bound.** The extracted `q_rne` returns `q` with
`|2·(q·den − num)| ≤ den`, i.e. the remainder is at most half a divisor. -/
theorem q_rne_half_spec (num den : I128) (hden : 0 < den.val)
    (hnum : |num.val| ≤ 2 ^ 120) (hden2 : den.val ≤ 2 ^ 120) :
    ∃ q : I128, fixed_trig.q_rne num den = ok (some q) ∧
      |2 * (q.val * den.val - num.val)| ≤ den.val := by
  have hle : ¬ (den <= 0#i128) := by
    rw [IScalar.le_equiv]
    simp
    omega
  have h : fixed_trig.q_rne num den ⦃ z => ∃ q : I128, z = some q ∧
      |2 * (q.val * den.val - num.val)| ≤ den.val ⦄ := by
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
      · -- `2r > den`: round up when `r > 0`
        rw [if_pos h1]
        simp only [bind_tc_ok]
        by_cases h2 : rr > 0#i128
        · rw [if_pos h2]
          step as ⟨i, hi⟩
          refine ⟨i, rfl, ?_⟩
          rw [hi]
          have h1v : den.val < 2 * rr.val := by
            have h := h1; rw [gt_iff_lt, IScalar.lt_equiv] at h; rw [hw2] at h; exact h
          exact q_rne_half_up num.val den.val qq.val rr.val hn h1v hrbound.2
        · rw [if_neg h2]
          step as ⟨i, hi⟩
          exfalso
          have h1v : den.val < w.val := by
            have h := h1; rw [gt_iff_lt, IScalar.lt_equiv] at h; exact h
          have h2v' : rr.val ≤ 0 := by
            have h := h2; rw [gt_iff_lt, IScalar.lt_equiv] at h
            have h0 : (0#i128).val = 0 := by norm_num
            rw [h0] at h
            omega
          omega
      · rw [if_neg h1]
        step as ⟨d, hd⟩
        by_cases h3 : w < d
        · -- `2r < -den`: round down when `r < 0`
          rw [if_pos h3]
          simp only [bind_tc_ok]
          by_cases h2 : rr > 0#i128
          · rw [if_pos h2]
            step as ⟨i, hi⟩
            exfalso
            have h2v : 0 < rr.val := by
              have h := h2; rw [gt_iff_lt, IScalar.lt_equiv] at h
              have h0 : (0#i128).val = 0 := by norm_num
              rw [h0] at h
              exact h
            have h3v : w.val < d.val := by rwa [IScalar.lt_equiv] at h3
            omega
          · rw [if_neg h2]
            step as ⟨i, hi⟩
            refine ⟨i, rfl, ?_⟩
            rw [hi]
            have h3v : w.val < d.val := by rwa [IScalar.lt_equiv] at h3
            exact q_rne_half_down num.val den.val qq.val rr.val hn (by omega) hrbound.1
        · rw [if_neg h3]
          by_cases h4 : w = den
          · -- tie at `2r = den`
            rw [if_pos h4]
            step as ⟨i1, hi1⟩
            have h4v : 2 * rr.val = den.val := by rw [← hw2, h4]
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
              exact (q_rne_half_tie_pos num.val den.val qq.val rr.val hden hn h4v).2
            · rw [if_neg h6]
              simp only [bind_tc_ok]
              rw [if_pos h2]
              step as ⟨i, hi⟩
              refine ⟨i, rfl, ?_⟩
              rw [hi]
              simpa using (q_rne_half_tie_pos num.val den.val qq.val rr.val hden hn h4v).1
          · rw [if_neg h4]
            by_cases h5 : w = d
            · -- tie at `2r = -den`
              rw [if_pos h5]
              step as ⟨i1, hi1⟩
              have h5v : 2 * rr.val = -den.val := by
                rw [← hw2, h5]
                exact hd
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
                exact (q_rne_half_tie_neg num.val den.val qq.val rr.val hden hn h5v).2
              · rw [if_neg h6]
                simp only [bind_tc_ok]
                rw [if_neg h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                simpa using (q_rne_half_tie_neg num.val den.val qq.val rr.val hden hn h5v).1
            · -- `-den < 2r < den`: truncation
              rw [if_neg h5]
              simp only [bind_tc_ok]
              have h1le : w.val ≤ den.val := by
                have := le_of_not_gt h1
                rwa [IScalar.le_equiv] at this
              have h3le : d.val ≤ w.val := by
                have := le_of_not_gt h3
                rwa [IScalar.le_equiv] at this
              have hwlo : -den.val ≤ w.val := by omega
              have hwhi : w.val ≤ den.val := h1le
              by_cases h2 : rr > 0#i128
              · rw [if_pos h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                simpa using q_rne_half_zero num.val den.val qq.val rr.val hn
                  (by omega) (by omega)
              · rw [if_neg h2]
                step as ⟨i, hi⟩
                refine ⟨i, rfl, ?_⟩
                rw [hi]
                simpa using q_rne_half_zero num.val den.val qq.val rr.val hn
                  (by omega) (by omega)
  obtain ⟨y, hy, q, hyq, hq⟩ := (WP.spec_equiv_exists _ _).1 h
  exact ⟨q, by rw [hy, hyq], hq⟩

/-! ## `k_bound` -/

/-- **`k_bound` (integer core).** A nearest-even quotient of `t` by the Q64 half-turn with
`|t| ≤ 2^76` obeys `|k| ≤ 2608 = ⌊2^76 / HALF_PI_Q64⌋ + 1`. -/
lemma k_abs_of_half (t k : I128) (ht : |t.val| ≤ 2 ^ 76)
    (hb : |2 * (k.val * fixed_trig.HALF_PI_Q64.val - t.val)| ≤ fixed_trig.HALF_PI_Q64.val) :
    |k.val| ≤ 2608 := by
  rw [halfPiQ64_val] at hb
  have h1 : 2 * |k.val * 28976077832308490240 - t.val| ≤ 28976077832308490240 := by
    have h := hb
    rw [abs_mul, show |(2 : ℤ)| = 2 by norm_num] at h
    exact h
  have h2 : |k.val * 28976077832308490240| ≤
      |k.val * 28976077832308490240 - t.val| + |t.val| := by
    have e : k.val * 28976077832308490240
        = (k.val * 28976077832308490240 - t.val) + t.val := by ring
    calc |k.val * 28976077832308490240|
        = |(k.val * 28976077832308490240 - t.val) + t.val| := by
          nth_rewrite 1 [e]
          rfl
      _ ≤ _ := abs_add_le _ _
  have h3 : 2 * |k.val * 28976077832308490240| ≤ 28976077832308490240 + 2 * 2 ^ 76 := by
    nlinarith [h1, h2, ht]
  have h4 : |k.val * 28976077832308490240| = |k.val| * 28976077832308490240 := by
    rw [abs_mul, abs_of_pos (by norm_num : (0 : ℤ) < 28976077832308490240)]
  rw [h4] at h3
  omega

/-- **`k_bound`.** On the window the extracted quotient `k = q_rne(x <<< 32, HALF_PI_Q64)` is at
most `2608` in magnitude; the `i128` product `k · HALF_PI_Q64` and the difference with `x <<< 32`
are therefore far inside `i128`. -/
theorem k_bound (x : I64) (hx : trig_window x) (t k : I128)
    (ht : (core.convert.num.FromI128I64.from x) <<< (32#i32) = ok t)
    (hk : fixed_trig.q_rne t fixed_trig.HALF_PI_Q64 = ok (some k)) : |k.val| ≤ 2608 := by
  have htval : t.val = x.val * 2 ^ 32 := shift32_val x hx t ht
  have htw : |t.val| ≤ 2 ^ 76 := by
    rw [htval]
    rw [trig_window] at hx
    rw [abs_mul]
    have hxabs : |x.val| ≤ 17592186044416 := abs_le.mpr hx
    have h44 : (17592186044416 : ℤ) = 2 ^ 44 := by norm_num
    have hpow : |(2 : ℤ) ^ 32| = 2 ^ 32 := by norm_num
    rw [h44] at hxabs
    rw [hpow]
    nlinarith [hxabs]
  obtain ⟨k', hk', hb⟩ := q_rne_half_spec t fixed_trig.HALF_PI_Q64
    (by rw [halfPiQ64_val]; norm_num)
    (by
      have e : (2 : ℤ) ^ 76 ≤ 2 ^ 120 := by norm_num
      exact le_trans htw e)
    (by rw [halfPiQ64_val]; norm_num)
  have hkk : k = k' := by
    have hs : (some k : Option I128) = some k' := Result.ok_injective (hk.symm.trans hk')
    exact Option.some.inj hs
  rw [hkk]
  exact k_abs_of_half t k' htw hb

/-! ## `reduction_spec` -/

/-- Half-divisor bound in `ℝ`, used for both remainders. -/
lemma abs_le_half_of_two_mul {a P : ℝ} (h : |2 * a| ≤ P) : |a| ≤ P / 2 := by
  have h' : |(2 : ℝ)| * |a| ≤ P := by rwa [abs_mul] at h
  rw [show |(2 : ℝ)| = 2 by norm_num] at h'
  linarith

/-- **`reduction_spec` (error core).** Given the check-free integer facts of the reduction head —
the quotient bound, the exact `i128` value of the remainder numerator, and the two nearest-even
half-divisor bounds — the narrowed Q32.32 remainder is within `2^-30` of `x - k·pi/2` and inside
`[-1, 1]`. -/
lemma reduction_error (x : I64) (k num2 q2 : I128) (r : I64)
    (hk : |k.val| ≤ 2608)
    (hn2 : num2.val = x.val * 2 ^ 32 - k.val * fixed_trig.HALF_PI_Q64.val)
    (hhalfk : |2 * (k.val * fixed_trig.HALF_PI_Q64.val - x.val * 2 ^ 32)| ≤
      fixed_trig.HALF_PI_Q64.val)
    (hhalfq : |2 * (q2.val * 2 ^ 32 - num2.val)| ≤ (2 : ℤ) ^ 32)
    (hr : r.val = q2.val) :
    |toReal r - (toReal x - (k.val : ℝ) * (Real.pi / 2))| ≤ (2 : ℝ) ^ (-30 : ℤ) ∧
      |toReal r| ≤ 1 := by
  have hP : fixed_trig.HALF_PI_Q64.val = 28976077832308490240 := halfPiQ64_val
  rw [hP] at hn2 hhalfk
  have hkr : |(k.val : ℝ)| ≤ 2608 := by exact_mod_cast hk
  -- integer facts cast to `ℝ`
  have hAi : |2 * (k.val * 28976077832308490240 - x.val * 2 ^ 32)| ≤ 28976077832308490240 :=
    hhalfk
  have hA : |2 * ((k.val : ℝ) * 28976077832308490240 - (x.val : ℝ) * 2 ^ 32)| ≤
      (28976077832308490240 : ℝ) := by
    have h : |((2 * (k.val * 28976077832308490240 - x.val * 2 ^ 32) : ℤ) : ℝ)| ≤
        ((28976077832308490240 : ℤ) : ℝ) := by exact_mod_cast hAi
    push_cast at h
    rw [abs_mul, show |(2 : ℝ)| = 2 by norm_num]
    norm_num at h ⊢
    exact h
  have hBi : |2 * (q2.val * 2 ^ 32 - num2.val)| ≤ (2 : ℤ) ^ 32 := hhalfq
  have hB : |2 * ((q2.val : ℝ) * 2 ^ 32 - (num2.val : ℝ))| ≤ (2 : ℝ) ^ 32 := by
    have h : |((2 * (q2.val * 2 ^ 32 - num2.val) : ℤ) : ℝ)| ≤ (((2 : ℤ) ^ 32 : ℤ) : ℝ) := by
      exact_mod_cast hBi
    push_cast at h
    rw [abs_mul, show |(2 : ℝ)| = 2 by norm_num]
    norm_num at h ⊢
    exact h
  have hn2R : (num2.val : ℝ) = (x.val : ℝ) * 2 ^ 32 - (k.val : ℝ) * 28976077832308490240 := by
    rw [hn2]; push_cast; ring
  have hC : |(28976077832308490240 : ℝ) / (2 : ℝ) ^ 64 - Real.pi / 2| ≤
      (1200 : ℝ) * (2 : ℝ) ^ (-64 : ℤ) := by
    have h := pi_red_bound
    rw [toRealQ64, hP] at h
    exact h
  -- (1) the numerator is within half a divisor of the exact Q64 remainder
  have hnum2 : |(num2.val : ℝ)| ≤ (28976077832308490240 : ℝ) / 2 := by
    rw [hn2R]
    have h := abs_le_half_of_two_mul hA
    rwa [show (x.val : ℝ) * 2 ^ 32 - (k.val : ℝ) * 28976077832308490240
        = -((k.val : ℝ) * 28976077832308490240 - (x.val : ℝ) * 2 ^ 32) by ring, abs_neg]
  -- (2) the second rounding is within half a Q32.32 unit
  have hq2 : |(q2.val : ℝ) * 2 ^ 32 - (num2.val : ℝ)| ≤ (2 : ℝ) ^ 31 := by
    have h := abs_le_half_of_two_mul hB
    have h2 : (2 : ℝ) ^ 32 / 2 = (2 : ℝ) ^ 31 := by norm_num
    rwa [h2] at h
  have hstep1 : |toReal r - (num2.val : ℝ) / (2 : ℝ) ^ 64| ≤ (2 : ℝ) ^ (-33 : ℤ) := by
    have e : toReal r - (num2.val : ℝ) / (2 : ℝ) ^ 64
        = ((q2.val : ℝ) * 2 ^ 32 - (num2.val : ℝ)) / (2 : ℝ) ^ 64 := by
      rw [toReal, hr]
      unfold q32
      field_simp
    rw [e, abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 64)]
    have h2 : (2 : ℝ) ^ 31 / (2 : ℝ) ^ 64 = (2 : ℝ) ^ (-33 : ℤ) := by norm_num
    rw [← h2]
    exact div_le_div_of_nonneg_right hq2 (by positivity)
  -- (3) the Q64 constant contributes `|k| · 1200 · 2^-64`
  have hstep2 : |(num2.val : ℝ) / (2 : ℝ) ^ 64 - (toReal x - (k.val : ℝ) * (Real.pi / 2))| ≤
      2608 * (1200 * (2 : ℝ) ^ (-64 : ℤ)) := by
    have e : (num2.val : ℝ) / (2 : ℝ) ^ 64 - (toReal x - (k.val : ℝ) * (Real.pi / 2))
        = -((k.val : ℝ) * ((28976077832308490240 : ℝ) / (2 : ℝ) ^ 64 - Real.pi / 2)) := by
      rw [hn2R, toReal]
      unfold q32
      field_simp
      ring
    rw [e, abs_neg, abs_mul]
    calc |(k.val : ℝ)| *
          |(28976077832308490240 : ℝ) / (2 : ℝ) ^ 64 - Real.pi / 2|
        ≤ 2608 * |(28976077832308490240 : ℝ) / (2 : ℝ) ^ 64 - Real.pi / 2| :=
          mul_le_mul_of_nonneg_right hkr (abs_nonneg _)
      _ ≤ 2608 * (1200 * (2 : ℝ) ^ (-64 : ℤ)) :=
          mul_le_mul_of_nonneg_left hC (by norm_num)
  constructor
  · have e : toReal r - (toReal x - (k.val : ℝ) * (Real.pi / 2))
        = (toReal r - (num2.val : ℝ) / (2 : ℝ) ^ 64) +
          ((num2.val : ℝ) / (2 : ℝ) ^ 64 - (toReal x - (k.val : ℝ) * (Real.pi / 2))) := by
      ring
    rw [e]
    refine le_trans (abs_add_le _ _) ?_
    have hfin : (2 : ℝ) ^ (-33 : ℤ) + 2608 * (1200 * (2 : ℝ) ^ (-64 : ℤ)) ≤
        (2 : ℝ) ^ (-30 : ℤ) := by norm_num
    linarith [hstep1, hstep2, hfin]
  · have hsub : toReal r - (num2.val : ℝ) / (2 : ℝ) ^ 64
        = r.val / (2 : ℝ) ^ 32 - (num2.val : ℝ) / (2 : ℝ) ^ 64 := by
      rw [toReal]; unfold q32; ring
    have hsplit : |toReal r| ≤ |toReal r - (num2.val : ℝ) / (2 : ℝ) ^ 64| +
        |(num2.val : ℝ) / (2 : ℝ) ^ 64| :=
      calc |toReal r| = |(toReal r - (num2.val : ℝ) / (2 : ℝ) ^ 64) +
            (num2.val : ℝ) / (2 : ℝ) ^ 64| := by rw [sub_add_cancel]
        _ ≤ _ := abs_add_le _ _
    have hdiv : |(num2.val : ℝ) / (2 : ℝ) ^ 64| = |(num2.val : ℝ)| / (2 : ℝ) ^ 64 := by
      rw [abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ 64)]
    have hnumdiv : |(num2.val : ℝ)| / (2 : ℝ) ^ 64 ≤
        (28976077832308490240 : ℝ) / (2 * (2 : ℝ) ^ 64) := by
      have h := div_le_div_of_nonneg_right hnum2 (by positivity : (0 : ℝ) ≤ (2 : ℝ) ^ 64)
      have e : ((28976077832308490240 : ℝ) / 2) / (2 : ℝ) ^ 64
          = (28976077832308490240 : ℝ) / (2 * (2 : ℝ) ^ 64) := by
        field_simp
      rwa [e] at h
    have hC2 : (28976077832308490240 : ℝ) / (2 * (2 : ℝ) ^ 64) ≤
        Real.pi / 4 + 600 * (2 : ℝ) ^ (-64 : ℤ) := by
      have h := abs_le.mp hC
      have e : (28976077832308490240 : ℝ) / (2 * (2 : ℝ) ^ 64)
          = ((28976077832308490240 : ℝ) / (2 : ℝ) ^ 64) / 2 := by
        field_simp
      rw [e]
      linarith [h.2]
    have hpi : Real.pi / 4 < 0.78539816339744830962 := by linarith [Real.pi_lt_d20]
    have hfin : (2 : ℝ) ^ (-33 : ℤ) + ((28976077832308490240 : ℝ) / (2 * (2 : ℝ) ^ 64)) ≤ 1 := by
      have h1 : (28976077832308490240 : ℝ) / (2 * (2 : ℝ) ^ 64) ≤
          Real.pi / 4 + 600 * (2 : ℝ) ^ (-64 : ℤ) := hC2
      have h2 : (600 : ℝ) * (2 : ℝ) ^ (-64 : ℤ) ≤ 0.0000000000000001 := by norm_num
      nlinarith [h1, h2, hpi]
    linarith [hsplit, hstep1, hdiv, hnumdiv, hfin]

/-- **`reduction_spec`.** Every checked `i128` operation of the extracted reduction head
(`x <<< 32`, `q_rne`, `k · HALF_PI_Q64`, the subtraction, `1 <<< 32`, the second `q_rne`, `narrow`)
succeeds on the window, `|k| ≤ 2608`, and the narrowed remainder satisfies
`|r - (x - k·pi/2)| ≤ 2^-30` and `|r| ≤ 1`. The `pi_red_bound` and `q_rne_half_spec` terms are
`1200·2^-64` per `|k|` and `2^-33` for the final rounding. -/
theorem reduction_spec (x : I64) (hx : trig_window x) :
    ∃ (t k p num2 d q2 : I128) (r : I64),
      (core.convert.num.FromI128I64.from x) <<< (32#i32) = ok t ∧
      t.val = x.val * 2 ^ 32 ∧
      fixed_trig.q_rne t fixed_trig.HALF_PI_Q64 = ok (some k) ∧
      |k.val| ≤ 2608 ∧
      k * fixed_trig.HALF_PI_Q64 = ok p ∧
      t - p = ok num2 ∧
      (1#i128 <<< (32#i32)) = ok d ∧
      d.val = 2 ^ 32 ∧
      fixed_trig.q_rne num2 d = ok (some q2) ∧
      fixed_trig.narrow q2 = ok (some r) ∧
      |toReal r - (toReal x - (k.val : ℝ) * (Real.pi / 2))| ≤ (2 : ℝ) ^ (-30 : ℤ) ∧
      |toReal r| ≤ 1 := by
  have hx' : trig_window x := hx
  rw [trig_window] at hx
  -- `x <<< 32`
  obtain ⟨t, ht, htval, _⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := core.convert.num.FromI128I64.from x) (y := 32#i32)
      (by norm_num) (by norm_num [I128.numBits]))
  have htval' : t.val = x.val * 2 ^ 32 := shift32_val x hx' t ht
  -- `k = q_rne t HALF_PI_Q64`
  have htw : |t.val| ≤ 2 ^ 120 := by
    rw [htval']
    rw [abs_mul]
    have hxabs : |x.val| ≤ 17592186044416 := abs_le.mpr hx
    have h44 : (17592186044416 : ℤ) = 2 ^ 44 := by norm_num
    have hpow : |(2 : ℤ) ^ 32| = 2 ^ 32 := by norm_num
    rw [h44] at hxabs
    rw [hpow]
    nlinarith [hxabs]
  have hPpos : 0 < fixed_trig.HALF_PI_Q64.val := by rw [halfPiQ64_val]; norm_num
  have hPle : fixed_trig.HALF_PI_Q64.val ≤ 2 ^ 120 := by rw [halfPiQ64_val]; norm_num
  obtain ⟨k, hk, hkhalf⟩ := q_rne_half_spec t fixed_trig.HALF_PI_Q64 hPpos htw hPle
  have hkabs : |k.val| ≤ 2608 := k_abs_of_half t k (by
    rw [htval']
    rw [abs_mul]
    have hxabs : |x.val| ≤ 17592186044416 := abs_le.mpr hx
    have h44 : (17592186044416 : ℤ) = 2 ^ 44 := by norm_num
    have hpow : |(2 : ℤ) ^ 32| = 2 ^ 32 := by norm_num
    rw [h44] at hxabs
    rw [hpow]
    nlinarith [hxabs]) hkhalf
  -- `p = k · HALF_PI_Q64`, no overflow
  have hmul : |k.val * fixed_trig.HALF_PI_Q64.val| ≤ 2608 * 28976077832308490240 := by
    rw [halfPiQ64_val, abs_mul,
      abs_of_pos (by norm_num : (0 : ℤ) < 28976077832308490240)]
    nlinarith [hkabs]
  obtain ⟨p, hp, hpv⟩ := WP.spec_imp_exists
    (IScalar.mul_spec (ty := .I128) (x := k) (y := fixed_trig.HALF_PI_Q64)
      (by
        rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq]
        have := abs_le.mp hmul
        omega)
      (by
        rw [IScalar.max_IScalarTy_I128_eq, I128.max_eq]
        have := abs_le.mp hmul
        omega))
  have hpval : p.val = k.val * fixed_trig.HALF_PI_Q64.val := hpv
  -- `num2 = t - p`, no overflow
  have hsub : |t.val - p.val| ≤ 2 ^ 76 + 2608 * 28976077832308490240 := by
    have h1 : |p.val| ≤ 2608 * 28976077832308490240 := by rw [hpval]; exact hmul
    have h2 : 2 ^ 76 ≤ 2 ^ 76 := le_refl _
    have h3 := abs_le.mp h1
    have h4 := abs_le.mp htw
    rw [abs_le]
    constructor <;> omega
  obtain ⟨num2, hsub2, hn2v⟩ := WP.spec_imp_exists
    (IScalar.sub_spec (ty := .I128) (x := t) (y := p)
      (by
        rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq]
        have := abs_le.mp hsub
        omega)
      (by
        rw [IScalar.max_IScalarTy_I128_eq, I128.max_eq]
        have := abs_le.mp hsub
        omega))
  -- `d = 1 <<< 32 = 2^32`
  obtain ⟨d, hd, hdval, _⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := 1#i128) (y := 32#i32)
      (by norm_num) (by norm_num [I128.numBits]))
  have hdval' : d.val = 2 ^ 32 := by
    rw [hdval]
    have h1 : (1#i128).val = 1 := rfl
    have h32 : (32#i32).toNat = 32 := by decide
    rw [h1, h32, Int.shiftLeft_eq]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.bmod]
  -- `q2 = q_rne num2 d`
  have hn2abs : |num2.val| ≤ 2 ^ 120 := by
    have h1 : |num2.val| ≤ 2 ^ 76 + 2608 * 28976077832308490240 := by
      rw [hn2v]; exact hsub
    have h2 : (2 : ℤ) ^ 76 + 2608 * 28976077832308490240 ≤ 2 ^ 120 := by norm_num
    have h4 := abs_le.mp h1
    rw [abs_le]
    constructor <;> omega
  obtain ⟨q2, hq2, hq2half⟩ := q_rne_half_spec num2 d
    (by rw [hdval']; norm_num) hn2abs (by rw [hdval']; norm_num)
  -- `narrow q2` succeeds
  have hq2abs : |q2.val| ≤ 4611686018427387904 := by
    have h1 : |2 * (q2.val * 2 ^ 32 - num2.val)| ≤ 2 ^ 32 := by
      rw [hdval'] at hq2half
      exact hq2half
    have h2 : |q2.val * 2 ^ 32 - num2.val| ≤ 2 ^ 31 := by
      have := h1
      rw [abs_mul, show |(2 : ℤ)| = 2 by norm_num] at this
      omega
    have h3 : |num2.val| ≤ 2 ^ 76 + 2608 * 28976077832308490240 := by
      rw [hn2v]; exact hsub
    have h4 := abs_le.mp h3
    have h5 : |q2.val * 2 ^ 32| ≤ |q2.val * 2 ^ 32 - num2.val| + |num2.val| := by
      have e : (q2.val * 2 ^ 32 - num2.val) + num2.val = q2.val * 2 ^ 32 := by ring
      have h := abs_add_le (q2.val * 2 ^ 32 - num2.val) num2.val
      rwa [e] at h
    rw [abs_mul, show |(2 : ℤ) ^ 32| = 2 ^ 32 by norm_num] at h5
    have h6 : |q2.val| * 2 ^ 32 ≤ 2 ^ 31 + (2 ^ 76 + 2608 * 28976077832308490240) := by
      nlinarith [h2, h5]
    have h7 : |q2.val| * 2 ^ 32 ≤ 4611686018427387904 * 2 ^ 32 := by
      have : (2 : ℤ) ^ 31 + (2 ^ 76 + 2608 * 28976077832308490240) ≤
          4611686018427387904 * 2 ^ 32 := by norm_num
      linarith
    omega
  obtain ⟨r, hr, hrv⟩ := narrow_spec q2
    (by rw [I64.min_eq]; have := abs_le.mp hq2abs; omega)
    (by rw [I64.max_eq]; have := abs_le.mp hq2abs; omega)
  refine ⟨t, k, p, num2, d, q2, r, ht, htval', hk, hkabs, hp, hsub2, hd, hdval', hq2, hr, ?_, ?_⟩
  · exact (reduction_error x k num2 q2 r hkabs (by rw [hn2v, hpval, htval'])
      (by rw [← htval']; exact hkhalf) (by rw [← hdval']; exact hq2half) hrv).1
  · exact (reduction_error x k num2 q2 r hkabs (by rw [hn2v, hpval, htval'])
      (by rw [← htval']; exact hkhalf) (by rw [← hdval']; exact hq2half) hrv).2

end

end apeiron_word
