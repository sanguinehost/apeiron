import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.Real.Pi.Bounds
import Numeric.Scale
import Numeric.Rope.Part1

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Taylor block

The remaining success path evaluates degree-8 Taylor truncations at `arg` with `i128` Horner in
`u = arg^2`, each partial product rounded to Q32.32 by `fixed_trig.round_div_2p32`. The rounding
primitive's value spec, the pure Horner targets `fixed_trig.sin_poly`/`cos_poly`, and the target
statement `fixed_trig_taylor_value` are below; `word_fixed_taylor_first` lands the first stage
(`u = round_q32 (t·t)`), `word_taylor_spec` the full pair. The error decomposition into
`fixed_sin_cos_refines` is `fixed_sin_cos_refines_proved`, at the end of this file.
-/

/-- The general fact the negative branch of `round_div_2p32` uses; Lean core does not expose it. -/
theorem fixed_trig_int_shiftRight_nonneg (a : ℤ) (s : ℕ) (h : 0 ≤ a) : 0 ≤ a >>> s := by
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h
  show (0 : ℤ) ≤ ↑(n >>> s)
  exact Int.natCast_nonneg _

/-- The extracted rounding half-constant `2^31` evaluates. -/
theorem fixed_trig_round_div_2p32_HALF_eq :
    fixed_trig.round_div_2p32.HALF = ok (2147483648#i128) := by
  have h : fixed_trig.round_div_2p32.HALF ⦃ z => z = (2147483648#i128) ⦄ := by
    unfold fixed_trig.round_div_2p32.HALF
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- Success spec for the extracted rounding step: on `value.val ≥ 0` the result is the arithmetic
right shift of `value.val + 2^31`; on `value.val < 0` it is the negation of that shift of
`-value.val + 2^31`. -/
theorem fixed_trig_round_div_2p32_spec (value : I128)
    (hpos : 0 ≤ value.val → value.val + 2147483648 ≤ I128.max)
    (hneg : value.val < 0 → -value.val + 2147483648 ≤ I128.max) :
    fixed_trig.round_div_2p32 value ⦃ z =>
      (0 ≤ value.val → z.val = (value.val + 2147483648) >>> 32) ∧
      (value.val < 0 → z.val = - ((-value.val + 2147483648) >>> 32)) ⦄ := by
  unfold fixed_trig.round_div_2p32
  rw [fixed_trig_round_div_2p32_HALF_eq]
  by_cases hv : 0 ≤ value.val
  · have hge : (0#i128) ≤ value := by rwa [IScalar.le_equiv]
    have hmax : value.val + 2147483648 ≤ I128.max := hpos hv
    simp only [hge, if_pos]
    step*
  · have hlt : value.val < 0 := not_le.mp hv
    have hmin : value.val ≠ I128.min := by
      intro h
      have hc := hneg (by rw [h]; norm_num [IScalar.min, I128.min, I128.numBits])
      rw [h] at hc
      norm_num [IScalar.min, I128.min, I128.numBits, I128.max] at hc
    have hge : ¬ ((0#i128) ≤ value) := by rwa [IScalar.le_equiv]
    have hmax : -value.val + 2147483648 ≤ I128.max := hneg hlt
    simp only [hge]
    step*
    all_goals
      (rw [i3_post]
       have h2 : 0 ≤ i2.val := by rw [i2_post, i_post]; omega
       have h3 : 0 ≤ i2.val >>> 32 := fixed_trig_int_shiftRight_nonneg _ _ h2
       have hminlt : IScalar.min IScalarTy.I128 < 0 := by
         norm_num [IScalar.min, I128.min, I128.numBits]
       intro heq
       omega)

/-- The mathematical value of the extracted `fixed_trig.round_div_2p32`: add `2^31` and shift right
by 32 on the nonnegative branch; on the negative branch negate, add `2^31`, shift right by 32 and
negate again. -/
def fixed_trig.round_q32 (v : ℤ) : ℤ :=
  if 0 ≤ v then (v + 2147483648) >>> 32 else - ((-v + 2147483648) >>> 32)

theorem fixed_trig_round_div_2p32_val (value : I128)
    (hpos : 0 ≤ value.val → value.val + 2147483648 ≤ I128.max)
    (hneg : value.val < 0 → -value.val + 2147483648 ≤ I128.max) :
    fixed_trig.round_div_2p32 value ⦃ z => z.val = fixed_trig.round_q32 value.val ⦄ := by
  apply Aeneas.Std.WP.spec_mono (fixed_trig_round_div_2p32_spec value hpos hneg)
  intro z hz
  unfold fixed_trig.round_q32
  by_cases hv : 0 ≤ value.val
  · rw [if_pos hv]; exact hz.1 hv
  · rw [if_neg hv]; exact hz.2 (lt_of_not_ge hv)

/-- The `sin` Horner target of the extracted `taylor_sin_cos`, in `u = round_q32 (t^2)`. -/
def fixed_trig.sin_poly (t : ℤ) : ℤ :=
  let u := fixed_trig.round_q32 (t * t)
  let a0 := fixed_trig.round_q32 ((-852176) * u) + 35791394
  let a1 := fixed_trig.round_q32 (a0 * u) - 715827883
  let a2 := fixed_trig.round_q32 (a1 * u) + 4294967296
  fixed_trig.round_q32 (t * a2)

/-- The `cos` Horner target of the extracted `taylor_sin_cos`, in `u = round_q32 (t^2)`. -/
def fixed_trig.cos_poly (t : ℤ) : ℤ :=
  let u := fixed_trig.round_q32 (t * t)
  let b0 := fixed_trig.round_q32 (106522 * u) - 5965232
  let b1 := fixed_trig.round_q32 (b0 * u) + 178956971
  let b2 := fixed_trig.round_q32 (b1 * u) - 2147483648
  fixed_trig.round_q32 (b2 * u) + 4294967296

/-- Target value statement for the extracted `taylor_sin_cos`: on `|t.val| <= 2^32` it returns the
pair `(sin_poly t.val, cos_poly t.val)`. The bound is `2^32`, not the fold's loose `2^36`: at
`|t| = 2^36` the degree-8 polynomial reaches `~2^105`, which does not fit the final `i64` cast, so the
statement is false at `2^36` and true at `2^32` (all intermediates `<= 2^67`, final outputs `<= 2^35`).
Total (no refusal); the WP chain over the Horner stages and the error decomposition are the next
slice. -/
def fixed_trig_taylor_value : Prop :=
  ∀ t : I64, |t.val| ≤ 2 ^ 32 →
    ∃ sin_t cos_t : I64,
      fixed_trig.taylor_sin_cos t = ok (sin_t, cos_t) ∧
        sin_t.val = fixed_trig.sin_poly t.val ∧
        cos_t.val = fixed_trig.cos_poly t.val

/-- First Taylor stage: `u = round_q32 (t·t)` with no refusal. -/
theorem word_fixed_taylor_first (t : I64) (ht : |t.val| ≤ 2 ^ 36) :
    (do
        let t1 ← lift (core.convert.num.FromI128I64.from t)
        let isq ← t1 * t1
        let u ← fixed_trig.round_div_2p32 isq
        ok u) ⦃ z => z.val = fixed_trig.round_q32 (t.val * t.val) ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
  all_goals (try norm_num [I128.min, I128.max, I128.numBits] at *)
  all_goals (try nlinarith [(abs_le.mp ht).1, (abs_le.mp ht).2])
  · -- rounding stage
    refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val isq ?_ ?_) ?_
    · intro h0
      norm_num [I128.max, I128.numBits] at *
      nlinarith [(abs_le.mp ht).1, (abs_le.mp ht).2]
    · intro hlt
      have hnn : 0 ≤ (↑t : ℤ) * ↑t := mul_self_nonneg _
      norm_num [I128.max, I128.numBits] at *
      omega
    · intro z hz
      rw [Aeneas.Std.WP.spec_ok]
      rw [hz, isq_post]

/-- Coarse bound on the extracted rounding: `|round_q32 v| ≤ |v| / 2^32 + 1`. -/
theorem fixed_trig.round_q32_abs_le (v : ℤ) :
    |fixed_trig.round_q32 v| ≤ |v| / 2^32 + 1 := by
  rw [fixed_trig.round_q32]
  by_cases hv : 0 ≤ v
  · rw [if_pos hv]
    have hpos : 0 ≤ (v + 2147483648) >>> 32 := by
      rw [Int.shiftRight_eq_div_pow]
      exact Int.ediv_nonneg (by omega) (by norm_num)
    rw [abs_of_nonneg hpos, Int.shiftRight_eq_div_pow, abs_of_nonneg hv]
    calc (v + 2147483648) / 2^32 ≤ (v + 1 * 2^32) / 2^32 := by
          apply Int.ediv_le_ediv (by norm_num); omega
      _ = v / 2^32 + 1 := by rw [Int.add_mul_ediv_right _ _ (by norm_num)]
  · have hv' : v < 0 := lt_of_not_ge hv
    rw [if_neg hv]
    have hneg : 0 ≤ ((-v + 2147483648) >>> 32) := by
      rw [Int.shiftRight_eq_div_pow]
      exact Int.ediv_nonneg (by omega) (by norm_num)
    rw [abs_neg, abs_of_nonneg hneg, Int.shiftRight_eq_div_pow, abs_of_neg hv']
    calc (-v + 2147483648) / 2^32 ≤ (-v + 1 * 2^32) / 2^32 := by
          apply Int.ediv_le_ediv (by norm_num); omega
      _ = (-v) / 2^32 + 1 := by rw [Int.add_mul_ediv_right _ _ (by norm_num)]

/-- Parameterised form: if `|v| ≤ C · 2^32` then the rounded value is at most `C + 1` in absolute
value. -/
theorem fixed_trig.round_q32_abs_le' (v : ℤ) (C : ℤ) (hv : |v| ≤ C * 2^32) :
    |fixed_trig.round_q32 v| ≤ C + 1 := by
  have h1 := fixed_trig.round_q32_abs_le v
  have h2 : |v| / 2^32 ≤ C := by
    rw [Int.ediv_le_iff_le_mul (by norm_num : (0:ℤ) < 2^32)]
    nlinarith [hv]
  omega

theorem fixed_trig.round_q32_half (v : ℤ) :
    |((fixed_trig.round_q32 v : ℤ) : ℝ) - (v : ℝ) / q32| ≤ 1 / 2 := by
  have hd : (0 : ℤ) < 4294967296 := by norm_num
  have hdne : (4294967296 : ℤ) ≠ 0 := by norm_num
  have hD : (4294967296 : ℝ) = q32 := by norm_num [q32]
  have hH : (2147483648 : ℝ) = (4294967296 : ℝ) / 2 := by norm_num
  have hQ : (0 : ℝ) < (4294967296 : ℝ) := by norm_num
  rw [fixed_trig.round_q32]
  by_cases hv : 0 ≤ v
  · rw [if_pos hv, Int.shiftRight_eq_div_pow]
    rw [show ((((2 : ℕ) ^ 32 : ℕ)) : ℤ) = 4294967296 by norm_num]
    set z : ℤ := (v + 2147483648) / 4294967296 with hz
    have hlo : z * 4294967296 ≤ v + 2147483648 := by
      have h := Int.ediv_mul_le (v + 2147483648) hdne
      rwa [← hz] at h
    have hhi : v + 2147483648 < (z + 1) * 4294967296 := by
      have hlti : (v + 2147483648) / 4294967296 < z + 1 := by rw [hz]; omega
      exact (Int.ediv_lt_iff_lt_mul hd).mp hlti
    have hloR : (z : ℝ) * (4294967296 : ℝ) ≤ (v : ℝ) + 2147483648 := by
      exact_mod_cast hlo
    have hhiR : (v : ℝ) + 2147483648 < ((z : ℝ) + 1) * (4294967296 : ℝ) := by
      exact_mod_cast hhi
    have hloR' : (z : ℝ) * (4294967296 : ℝ) ≤ (v : ℝ) + (4294967296 : ℝ) / 2 := by
      linarith [hloR, hH]
    have hhiR' : (v : ℝ) + (4294967296 : ℝ) / 2 < z * (4294967296 : ℝ) + (4294967296 : ℝ) := by
      linarith [hhiR, hH]
    rw [show (v : ℝ) / q32 = (v : ℝ) / (4294967296 : ℝ) by rw [hD]]
    rw [abs_le]
    constructor
    · have h : ((z : ℝ) - (v : ℝ) / (4294967296 : ℝ)) * (4294967296 : ℝ) =
          (z : ℝ) * (4294967296 : ℝ) - (v : ℝ) := by field_simp
      refine (mul_le_mul_iff_of_pos_right hQ).mp ?_
      rw [h]
      nlinarith [hhiR']
    · have h : ((z : ℝ) - (v : ℝ) / (4294967296 : ℝ)) * (4294967296 : ℝ) =
          (z : ℝ) * (4294967296 : ℝ) - (v : ℝ) := by field_simp
      refine (mul_le_mul_iff_of_pos_right hQ).mp ?_
      rw [h]
      nlinarith [hloR']
  · rw [if_neg hv, Int.shiftRight_eq_div_pow]
    rw [show ((((2 : ℕ) ^ 32 : ℕ)) : ℤ) = 4294967296 by norm_num]
    set z : ℤ := (-v + 2147483648) / 4294967296 with hz
    have hlo : z * 4294967296 ≤ -v + 2147483648 := by
      have h := Int.ediv_mul_le (-v + 2147483648) hdne
      rwa [← hz] at h
    have hhi : -v + 2147483648 < (z + 1) * 4294967296 := by
      have hlti : (-v + 2147483648) / 4294967296 < z + 1 := by rw [hz]; omega
      exact (Int.ediv_lt_iff_lt_mul hd).mp hlti
    have hloR : (z : ℝ) * (4294967296 : ℝ) ≤ (-v : ℝ) + 2147483648 := by
      exact_mod_cast hlo
    have hhiR : (-v : ℝ) + 2147483648 < ((z : ℝ) + 1) * (4294967296 : ℝ) := by
      exact_mod_cast hhi
    have hloR' : (z : ℝ) * (4294967296 : ℝ) ≤ (-v : ℝ) + (4294967296 : ℝ) / 2 := by
      linarith [hloR, hH]
    have hhiR' : (-v : ℝ) + (4294967296 : ℝ) / 2 < z * (4294967296 : ℝ) + (4294967296 : ℝ) := by
      linarith [hhiR, hH]
    rw [show (v : ℝ) / q32 = (v : ℝ) / (4294967296 : ℝ) by rw [hD]]
    simp only [Int.cast_neg]
    rw [abs_le]
    constructor
    · have h : ((-(z : ℝ)) - (v : ℝ) / (4294967296 : ℝ)) * (4294967296 : ℝ) =
          -(z : ℝ) * (4294967296 : ℝ) - (v : ℝ) := by field_simp
      refine (mul_le_mul_iff_of_pos_right hQ).mp ?_
      rw [h]
      nlinarith [hhiR']
    · have h : ((-(z : ℝ)) - (v : ℝ) / (4294967296 : ℝ)) * (4294967296 : ℝ) =
          -(z : ℝ) * (4294967296 : ℝ) - (v : ℝ) := by field_simp
      refine (mul_le_mul_iff_of_pos_right hQ).mp ?_
      rw [h]
      nlinarith [hloR']

theorem fixed_trig.round_q32_ledger (v : ℤ) :
    |((fixed_trig.round_q32 v : ℤ) : ℝ) / q32 - (v : ℝ) / q32 / q32| ≤ 1 / (2 * q32) := by
  have h := fixed_trig.round_q32_half v
  have hq := q32_pos
  have hrw : ((fixed_trig.round_q32 v : ℤ) : ℝ) / q32 - (v : ℝ) / q32 / q32 =
      (((fixed_trig.round_q32 v : ℤ) : ℝ) - (v : ℝ) / q32) / q32 := by ring
  rw [hrw, abs_div, abs_of_pos hq]
  rw [show (1 : ℝ) / (2 * q32) = (1 / 2) / q32 by ring]
  rw [div_le_div_iff_of_pos_right hq]
  exact h

/-- If `|x| ≤ 2^100` then both `x + 2^31` and `-x + 2^31` stay in the `i128` range, so the
`fixed_trig.round_div_2p32` overflow premises hold. -/
theorem fixed_trig.round_safe (x : ℤ) (hx : |x| ≤ 2^100) :
    x + 2147483648 ≤ I128.max ∧ -x + 2147483648 ≤ I128.max := by
  have hx' := abs_le.mp hx
  constructor <;> norm_num [I128.max, I128.numBits] <;> omega

/-- `|a·b| ≤ B` from `|a| ≤ Ba`, `|b| ≤ Bb`, `Ba·Bb ≤ B`. -/
theorem fixed_trig.abs_mul_le (a b : I128) (Ba Bb B : ℤ)
    (ha : |a.val| ≤ Ba) (hb : |b.val| ≤ Bb) (hBa : 0 ≤ Ba)
    (h : Ba * Bb ≤ B) : |a.val * b.val| ≤ B := by
  rw [abs_mul]
  exact le_trans (mul_le_mul ha hb (abs_nonneg b.val) hBa) h

/-- `I128.min ≤ a·b ≤ I128.max` from `|a| ≤ Ba`, `|b| ≤ Bb`, `Ba·Bb ≤ 2^126`. -/
theorem fixed_trig.mul_range (a b : I128) (Ba Bb : ℤ)
    (ha : |a.val| ≤ Ba) (hb : |b.val| ≤ Bb) (hBa : 0 ≤ Ba)
    (h : Ba * Bb ≤ 2 ^ 126) :
    -(2 ^ 127) ≤ a.val * b.val ∧ a.val * b.val ≤ 2 ^ 127 - 1 := by
  have hab : |a.val * b.val| ≤ 2 ^ 126 :=
    fixed_trig.abs_mul_le a b Ba Bb (2 ^ 126) ha hb hBa h
  have := abs_le.mp hab
  constructor <;> norm_num [I128.min, I128.max, I128.numBits] <;> omega

/-- `|a·b| ≤ B` for `ℤ` factors. -/
theorem fixed_trig.abs_mul_le_z (a b : ℤ) (Ba Bb B : ℤ)
    (ha : |a| ≤ Ba) (hb : |b| ≤ Bb) (hBa : 0 ≤ Ba)
    (h : Ba * Bb ≤ B) : |a * b| ≤ B := by
  rw [abs_mul]
  exact le_trans (mul_le_mul ha hb (abs_nonneg b) hBa) h

/-- `|a + c| ≤ Ba + c` for `c ≥ 0`. -/
theorem fixed_trig.abs_add_const_le (a : I128) (Ba c : ℤ) (hc : 0 ≤ c) (ha : |a.val| ≤ Ba) :
    |a.val + c| ≤ Ba + c := by
  exact le_trans (abs_add_le _ _) (by rw [abs_of_nonneg hc]; exact add_le_add_left ha c)

/-- `|a - c| ≤ Ba + c` for `c ≥ 0`. -/
theorem fixed_trig.abs_sub_const_le (a : I128) (Ba c : ℤ) (hc : 0 ≤ c) (ha : |a.val| ≤ Ba) :
    |a.val - c| ≤ Ba + c := by
  have h : a.val - c = a.val + (-c) := by ring
  rw [h]
  exact le_trans (abs_add_le _ _) (by rw [abs_neg, abs_of_nonneg hc]; exact add_le_add_left ha c)

/-- `round_div_2p32` overflow premises from `|x| ≤ 2^66`. -/
theorem fixed_trig.round_safe66 (x : ℤ) (hx : |x| ≤ 2 ^ 66) :
    x + 2147483648 ≤ I128.max ∧ -x + 2147483648 ≤ I128.max := by
  have hx' := abs_le.mp hx
  constructor <;> norm_num [I128.max, I128.numBits] <;> omega

/-- probe: calc-based flat chain, all nine stages, trivial post. -/
theorem word_taylor_spec (t : I64) (ht : |t.val| ≤ 2 ^ 32) :
    fixed_trig.taylor_sin_cos t ⦃ z =>
      ∃ sin_t cos_t : I64,
        z = (sin_t, cos_t) ∧
        sin_t.val = fixed_trig.sin_poly t.val ∧
        cos_t.val = fixed_trig.cos_poly t.val ∧
        |sin_t.val| ≤ 2 ^ 34 ∧
        |cos_t.val| ≤ 2 ^ 34 ⦄ := by
  unfold fixed_trig.taylor_sin_cos
  simp only [lift, bind_tc_ok]
  have ht2 := abs_le.mp ht
  step*
  · simp only [core.convert.num.FromI128I64.from_val_eq] at *
    norm_num [I128.min, I128.numBits] at *
    nlinarith [ht2.1, ht2.2]
  · simp only [core.convert.num.FromI128I64.from_val_eq] at *
    norm_num [I128.max, I128.numBits] at *
    nlinarith [ht2.1, ht2.2]
  · simp only [core.convert.num.FromI128I64.from_val_eq] at *
    have hi66 : |i.val| ≤ 2 ^ 66 := by
      rw [i_post, abs_mul]
      calc |(↑t : ℤ)| * |(↑t : ℤ)| ≤ (2 ^ 32) * (2 ^ 32) :=
            mul_le_mul ht ht (abs_nonneg _) (by norm_num)
        _ ≤ 2 ^ 66 := by norm_num
    refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i ?_ ?_) ?_
    · intro h0; exact (fixed_trig.round_safe66 i.val hi66).1
    · intro hlt; exact (fixed_trig.round_safe66 i.val hi66).2
    · intro u hu
      have huB : |u.val| ≤ 2 ^ 32 + 1 := by
        rw [hu]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 32)
        rw [i_post, abs_mul]
        calc |(↑t : ℤ)| * |(↑t : ℤ)| ≤ (2 ^ 32) * (2 ^ 32) :=
              mul_le_mul ht ht (abs_nonneg _) (by norm_num)
          _ ≤ (2 ^ 32) * 2 ^ 32 := by norm_num
      have hu2 := abs_le.mp huB
      step*
      all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
      all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.SIN_C3] at *)
      all_goals (try (rw [q_post] at *; omega))
      · have hqB : |q.val| ≤ 852176 := by rw [q_post, abs_neg]; norm_num [fixed_trig.SIN_C3]
        refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i2 ?_ ?_) ?_
        · intro h0
          exact (fixed_trig.round_safe66 i2.val (by rw [i2_post]; exact fixed_trig.abs_mul_le q u 852176 (2 ^ 32 + 1) (2 ^ 66) hqB huB (by norm_num) (by norm_num))).1
        · intro hlt
          exact (fixed_trig.round_safe66 i2.val (by rw [i2_post]; exact fixed_trig.abs_mul_le q u 852176 (2 ^ 32 + 1) (2 ^ 66) hqB huB (by norm_num) (by norm_num))).2
        · intro i3 hi3
          have hi3B : |i3.val| ≤ 852178 := by
            rw [hi3]; apply fixed_trig.round_q32_abs_le' (C := 852177)
            rw [i2_post]
            exact fixed_trig.abs_mul_le q u 852176 (2 ^ 32 + 1) (852177 * 2 ^ 32) hqB huB (by norm_num) (by norm_num)
          step*
          all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
          all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.SIN_C2] at *)
          all_goals (try (have h2 := abs_le.mp hi3B; omega))
          · have hq1B : |q1.val| ≤ 2 ^ 26 := by
              rw [q1_post]
              exact le_trans (fixed_trig.abs_add_const_le i3 852178 35791394 (by norm_num) hi3B) (by norm_num)
            exact (fixed_trig.mul_range q1 u (2 ^ 26) (2 ^ 32 + 1) hq1B huB (by norm_num) (by norm_num)).1
          · have hq1B : |q1.val| ≤ 2 ^ 26 := by
              rw [q1_post]
              exact le_trans (fixed_trig.abs_add_const_le i3 852178 35791394 (by norm_num) hi3B) (by norm_num)
            exact (fixed_trig.mul_range q1 u (2 ^ 26) (2 ^ 32 + 1) hq1B huB (by norm_num) (by norm_num)).2
          · have hq1B : |q1.val| ≤ 2 ^ 26 := by
              rw [q1_post]
              exact le_trans (fixed_trig.abs_add_const_le i3 852178 35791394 (by norm_num) hi3B) (by norm_num)
            refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i5 ?_ ?_) ?_
            · intro h0
              exact (fixed_trig.round_safe66 i5.val (by rw [i5_post]; exact fixed_trig.abs_mul_le q1 u (2 ^ 26) (2 ^ 32 + 1) (2 ^ 66) hq1B huB (by norm_num) (by norm_num))).1
            · intro hlt
              exact (fixed_trig.round_safe66 i5.val (by rw [i5_post]; exact fixed_trig.abs_mul_le q1 u (2 ^ 26) (2 ^ 32 + 1) (2 ^ 66) hq1B huB (by norm_num) (by norm_num))).2
            · intro i6 hi6
              have hi6B : |i6.val| ≤ 2 ^ 27 + 1 := by
                rw [hi6]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 27)
                rw [i5_post]
                exact fixed_trig.abs_mul_le q1 u (2 ^ 26) (2 ^ 32 + 1) ((2 ^ 27) * 2 ^ 32) hq1B huB (by norm_num) (by norm_num)
              step*
              all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
              all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.SIN_C1] at *)
              all_goals (try (have h2 := abs_le.mp hi6B; omega))
              · have hq2B : |q2.val| ≤ 2 ^ 30 := by
                  rw [q2_post]
                  exact le_trans (fixed_trig.abs_sub_const_le i6 (2 ^ 27 + 1) 715827883 (by norm_num) hi6B) (by norm_num)
                exact (fixed_trig.mul_range q2 u (2 ^ 30) (2 ^ 32 + 1) hq2B huB (by norm_num) (by norm_num)).1
              · have hq2B : |q2.val| ≤ 2 ^ 30 := by
                  rw [q2_post]
                  exact le_trans (fixed_trig.abs_sub_const_le i6 (2 ^ 27 + 1) 715827883 (by norm_num) hi6B) (by norm_num)
                exact (fixed_trig.mul_range q2 u (2 ^ 30) (2 ^ 32 + 1) hq2B huB (by norm_num) (by norm_num)).2
              · have hq2B : |q2.val| ≤ 2 ^ 30 := by
                  rw [q2_post]
                  exact le_trans (fixed_trig.abs_sub_const_le i6 (2 ^ 27 + 1) 715827883 (by norm_num) hi6B) (by norm_num)
                refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i8 ?_ ?_) ?_
                · intro h0
                  exact (fixed_trig.round_safe66 i8.val (by rw [i8_post]; exact fixed_trig.abs_mul_le q2 u (2 ^ 30) (2 ^ 32 + 1) (2 ^ 66) hq2B huB (by norm_num) (by norm_num))).1
                · intro hlt
                  exact (fixed_trig.round_safe66 i8.val (by rw [i8_post]; exact fixed_trig.abs_mul_le q2 u (2 ^ 30) (2 ^ 32 + 1) (2 ^ 66) hq2B huB (by norm_num) (by norm_num))).2
                · intro i9 hi9
                  have hi9B : |i9.val| ≤ 2 ^ 31 + 1 := by
                    rw [hi9]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 31)
                    rw [i8_post]
                    exact fixed_trig.abs_mul_le q2 u (2 ^ 30) (2 ^ 32 + 1) ((2 ^ 31) * 2 ^ 32) hq2B huB (by norm_num) (by norm_num)
                  simp only [fixed_trig_ONE_eq]
                  step*
                  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
                  all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.ONE] at *)
                  · have hq3B : |q3.val| ≤ 2 ^ 33 := by
                      rw [q3_post]
                      exact le_trans (fixed_trig.abs_add_const_le i9 (2 ^ 31 + 1) 4294967296 (by norm_num) hi9B) (by norm_num)
                    have htq : |(↑t : ℤ) * ↑q3| ≤ 2 ^ 66 :=
                      fixed_trig.abs_mul_le_z (↑t) ↑q3 (2 ^ 32) (2 ^ 33) (2 ^ 66) ht hq3B (by norm_num) (by norm_num)
                    have h2 := abs_le.mp htq
                    norm_num [I128.min, I128.numBits] at *
                    omega
                  · have hq3B : |q3.val| ≤ 2 ^ 33 := by
                      rw [q3_post]
                      exact le_trans (fixed_trig.abs_add_const_le i9 (2 ^ 31 + 1) 4294967296 (by norm_num) hi9B) (by norm_num)
                    have htq : |(↑t : ℤ) * ↑q3| ≤ 2 ^ 66 :=
                      fixed_trig.abs_mul_le_z (↑t) ↑q3 (2 ^ 32) (2 ^ 33) (2 ^ 66) ht hq3B (by norm_num) (by norm_num)
                    have h2 := abs_le.mp htq
                    norm_num [I128.max, I128.numBits] at *
                    omega
                  · have hq3B : |q3.val| ≤ 2 ^ 33 := by
                      rw [q3_post]
                      exact le_trans (fixed_trig.abs_add_const_le i9 (2 ^ 31 + 1) 4294967296 (by norm_num) hi9B) (by norm_num)
                    refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i12 ?_ ?_) ?_
                    · intro h0
                      exact (fixed_trig.round_safe66 i12.val (by rw [i12_post]; exact fixed_trig.abs_mul_le_z (↑t) ↑q3 (2 ^ 32) (2 ^ 33) (2 ^ 66) ht hq3B (by norm_num) (by norm_num))).1
                    · intro hlt
                      exact (fixed_trig.round_safe66 i12.val (by rw [i12_post]; exact fixed_trig.abs_mul_le_z (↑t) ↑q3 (2 ^ 32) (2 ^ 33) (2 ^ 66) ht hq3B (by norm_num) (by norm_num))).2
                    · intro i13 hi13
                      have hi13B : |i13.val| ≤ 2 ^ 33 + 1 := by
                        rw [hi13]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 33)
                        rw [i12_post]
                        exact fixed_trig.abs_mul_le_z (↑t) ↑q3 (2 ^ 32) (2 ^ 33) ((2 ^ 33) * 2 ^ 32) ht hq3B (by norm_num) (by norm_num)
                      have hi13B2 := abs_le.mp hi13B
                      step*
                      all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
                      all_goals (try norm_num [I128.min, I128.max, I128.numBits, I64.min, I64.max,
                        I64.numBits, fixed_trig.COS_C4] at *)
                      · have h : |(106522 : ℤ) * ↑u| ≤ 2 ^ 66 :=
                          fixed_trig.abs_mul_le_z 106522 (↑u) 106522 (2 ^ 32 + 1) (2 ^ 66) (by norm_num) huB (by norm_num) (by norm_num)
                        have h2 := abs_le.mp h
                        norm_num [I128.min, I128.numBits] at *
                        omega
                      · have h : |(106522 : ℤ) * ↑u| ≤ 2 ^ 66 :=
                          fixed_trig.abs_mul_le_z 106522 (↑u) 106522 (2 ^ 32 + 1) (2 ^ 66) (by norm_num) huB (by norm_num) (by norm_num)
                        have h2 := abs_le.mp h
                        norm_num [I128.max, I128.numBits] at *
                        omega
                      · have hpB : |(106522 : ℤ)| ≤ 106522 := by norm_num
                        refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i14 ?_ ?_) ?_
                        · intro h0
                          exact (fixed_trig.round_safe66 i14.val (by rw [i14_post]; exact fixed_trig.abs_mul_le_z 106522 (↑u) 106522 (2 ^ 32 + 1) (2 ^ 66) (by norm_num) huB (by norm_num) (by norm_num))).1
                        · intro hlt
                          exact (fixed_trig.round_safe66 i14.val (by rw [i14_post]; exact fixed_trig.abs_mul_le_z 106522 (↑u) 106522 (2 ^ 32 + 1) (2 ^ 66) (by norm_num) huB (by norm_num) (by norm_num))).2
                        · intro i15 hi15
                          have hi15B : |i15.val| ≤ 2 ^ 18 + 1 := by
                            rw [hi15]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 18)
                            rw [i14_post]
                            exact fixed_trig.abs_mul_le_z 106522 (↑u) 106522 (2 ^ 32 + 1) ((2 ^ 18) * 2 ^ 32) (by norm_num) huB (by norm_num) (by norm_num)
                          step*
                          all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
                          all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.COS_C3] at *)
                          all_goals (try (have h2 := abs_le.mp hi15B; omega))
                          · have hp1B : |p1.val| ≤ 2 ^ 23 := by
                              rw [p1_post]
                              exact le_trans (fixed_trig.abs_sub_const_le i15 (2 ^ 18 + 1) 5965232 (by norm_num) hi15B) (by norm_num)
                            exact (fixed_trig.mul_range p1 u (2 ^ 23) (2 ^ 32 + 1) hp1B huB (by norm_num) (by norm_num)).1
                          · have hp1B : |p1.val| ≤ 2 ^ 23 := by
                              rw [p1_post]
                              exact le_trans (fixed_trig.abs_sub_const_le i15 (2 ^ 18 + 1) 5965232 (by norm_num) hi15B) (by norm_num)
                            exact (fixed_trig.mul_range p1 u (2 ^ 23) (2 ^ 32 + 1) hp1B huB (by norm_num) (by norm_num)).2
                          · have hp1B : |p1.val| ≤ 2 ^ 23 := by
                              rw [p1_post]
                              exact le_trans (fixed_trig.abs_sub_const_le i15 (2 ^ 18 + 1) 5965232 (by norm_num) hi15B) (by norm_num)
                            refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i17 ?_ ?_) ?_
                            · intro h0
                              exact (fixed_trig.round_safe66 i17.val (by rw [i17_post]; exact fixed_trig.abs_mul_le p1 u (2 ^ 23) (2 ^ 32 + 1) (2 ^ 66) hp1B huB (by norm_num) (by norm_num))).1
                            · intro hlt
                              exact (fixed_trig.round_safe66 i17.val (by rw [i17_post]; exact fixed_trig.abs_mul_le p1 u (2 ^ 23) (2 ^ 32 + 1) (2 ^ 66) hp1B huB (by norm_num) (by norm_num))).2
                            · intro i18 hi18
                              have hi18B : |i18.val| ≤ 2 ^ 24 + 1 := by
                                rw [hi18]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 24)
                                rw [i17_post]
                                exact fixed_trig.abs_mul_le p1 u (2 ^ 23) (2 ^ 32 + 1) ((2 ^ 24) * 2 ^ 32) hp1B huB (by norm_num) (by norm_num)
                              step*
                              all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
                              all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.COS_C2] at *)
                              all_goals (try (have h2 := abs_le.mp hi18B; omega))
                              · have hp2B : |p2.val| ≤ 2 ^ 28 := by
                                  rw [p2_post]
                                  exact le_trans (fixed_trig.abs_add_const_le i18 (2 ^ 24 + 1) 178956971 (by norm_num) hi18B) (by norm_num)
                                exact (fixed_trig.mul_range p2 u (2 ^ 28) (2 ^ 32 + 1) hp2B huB (by norm_num) (by norm_num)).1
                              · have hp2B : |p2.val| ≤ 2 ^ 28 := by
                                  rw [p2_post]
                                  exact le_trans (fixed_trig.abs_add_const_le i18 (2 ^ 24 + 1) 178956971 (by norm_num) hi18B) (by norm_num)
                                exact (fixed_trig.mul_range p2 u (2 ^ 28) (2 ^ 32 + 1) hp2B huB (by norm_num) (by norm_num)).2
                              · have hp2B : |p2.val| ≤ 2 ^ 28 := by
                                  rw [p2_post]
                                  exact le_trans (fixed_trig.abs_add_const_le i18 (2 ^ 24 + 1) 178956971 (by norm_num) hi18B) (by norm_num)
                                refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i20 ?_ ?_) ?_
                                · intro h0
                                  exact (fixed_trig.round_safe66 i20.val (by rw [i20_post]; exact fixed_trig.abs_mul_le p2 u (2 ^ 28) (2 ^ 32 + 1) (2 ^ 66) hp2B huB (by norm_num) (by norm_num))).1
                                · intro hlt
                                  exact (fixed_trig.round_safe66 i20.val (by rw [i20_post]; exact fixed_trig.abs_mul_le p2 u (2 ^ 28) (2 ^ 32 + 1) (2 ^ 66) hp2B huB (by norm_num) (by norm_num))).2
                                · intro i21 hi21
                                  have hi21B : |i21.val| ≤ 2 ^ 29 + 1 := by
                                    rw [hi21]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 29)
                                    rw [i20_post]
                                    exact fixed_trig.abs_mul_le p2 u (2 ^ 28) (2 ^ 32 + 1) ((2 ^ 29) * 2 ^ 32) hp2B huB (by norm_num) (by norm_num)
                                  step*
                                  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq] at *))
                                  all_goals (try norm_num [I128.min, I128.max, I128.numBits, fixed_trig.COS_C1] at *)
                                  all_goals (try (have h2 := abs_le.mp hi21B; omega))
                                  · have hp3B : |p3.val| ≤ 2 ^ 32 := by
                                      rw [p3_post]
                                      exact le_trans (fixed_trig.abs_sub_const_le i21 (2 ^ 29 + 1) 2147483648 (by norm_num) hi21B) (by norm_num)
                                    exact (fixed_trig.mul_range p3 u (2 ^ 32) (2 ^ 32 + 1) hp3B huB (by norm_num) (by norm_num)).1
                                  · have hp3B : |p3.val| ≤ 2 ^ 32 := by
                                      rw [p3_post]
                                      exact le_trans (fixed_trig.abs_sub_const_le i21 (2 ^ 29 + 1) 2147483648 (by norm_num) hi21B) (by norm_num)
                                    exact (fixed_trig.mul_range p3 u (2 ^ 32) (2 ^ 32 + 1) hp3B huB (by norm_num) (by norm_num)).2
                                  · have hp3B : |p3.val| ≤ 2 ^ 32 := by
                                      rw [p3_post]
                                      exact le_trans (fixed_trig.abs_sub_const_le i21 (2 ^ 29 + 1) 2147483648 (by norm_num) hi21B) (by norm_num)
                                    refine Aeneas.Std.WP.spec_bind (fixed_trig_round_div_2p32_val i23 ?_ ?_) ?_
                                    · intro h0
                                      exact (fixed_trig.round_safe66 i23.val (by rw [i23_post]; exact fixed_trig.abs_mul_le p3 u (2 ^ 32) (2 ^ 32 + 1) (2 ^ 66) hp3B huB (by norm_num) (by norm_num))).1
                                    · intro hlt
                                      exact (fixed_trig.round_safe66 i23.val (by rw [i23_post]; exact fixed_trig.abs_mul_le p3 u (2 ^ 32) (2 ^ 32 + 1) (2 ^ 66) hp3B huB (by norm_num) (by norm_num))).2
                                    · intro i24 hi24
                                      have hi24B : |i24.val| ≤ 2 ^ 33 + 1 := by
                                        rw [hi24]; apply fixed_trig.round_q32_abs_le' (C := 2 ^ 33)
                                        rw [i23_post]
                                        exact fixed_trig.abs_mul_le p3 u (2 ^ 32) (2 ^ 32 + 1) ((2 ^ 33) * 2 ^ 32) hp3B huB (by norm_num) (by norm_num)
                                      have hi24B2 := abs_le.mp hi24B
                                      step*
                                      refine ⟨IScalar.cast IScalarTy.I64 i13, IScalar.cast IScalarTy.I64 p4,
                                        rfl, rfl, ?_, ?_, ?_, ?_⟩
                                      · have hcast : (↑(IScalar.cast IScalarTy.I64 i13) : ℤ) = i13.val := by
                                          rw [IScalar.cast_val_eq]; norm_num
                                          exact int_bmod_self_64 i13.val (by omega) (by omega)
                                        rw [hcast]
                                        simp only [fixed_trig.sin_poly, hu, i_post, hi3, i2_post, q_post,
                                          q1_post, i5_post, hi6, q2_post, i8_post, hi9, q3_post, i12_post, hi13]
                                      · have hp4B2 : -(2 ^ 34) ≤ p4.val ∧ p4.val ≤ 2 ^ 34 := by
                                          rw [p4_post]
                                          norm_num [core.convert.num.FromI128I64.from_val_eq]
                                          constructor <;> omega
                                        have hcast4 : (↑(IScalar.cast IScalarTy.I64 p4) : ℤ) = p4.val := by
                                          rw [IScalar.cast_val_eq]; norm_num
                                          exact int_bmod_self_64 p4.val (by omega) (by omega)
                                        rw [hcast4]
                                        simp only [fixed_trig.cos_poly, hu, i_post, hi15, i14_post, p1_post,
                                          i17_post, hi18, p2_post, i20_post, hi21, p3_post, i23_post, hi24, p4_post]
                                        rfl
                                      · have hcast : (↑(IScalar.cast IScalarTy.I64 i13) : ℤ) = i13.val := by
                                          rw [IScalar.cast_val_eq]; norm_num
                                          exact int_bmod_self_64 i13.val (by omega) (by omega)
                                        rw [hcast]
                                        exact le_trans hi13B (by norm_num)
                                      · have hp4B2' : -(2 ^ 34) ≤ p4.val ∧ p4.val ≤ 2 ^ 34 := by
                                          rw [p4_post]
                                          norm_num [core.convert.num.FromI128I64.from_val_eq]
                                          constructor <;> omega
                                        have hcast4 : (↑(IScalar.cast IScalarTy.I64 p4) : ℤ) = p4.val := by
                                          rw [IScalar.cast_val_eq]; norm_num
                                          exact int_bmod_self_64 p4.val (by omega) (by omega)
                                        rw [hcast4]
                                        exact abs_le.mpr hp4B2'

/-- The extracted `taylor_sin_cos` returns `(sin_poly t.val, cos_poly t.val)` on `|t.val| ≤ 2^32`. -/
theorem fixed_trig_taylor_value_proved : fixed_trig_taylor_value := by
  intro t ht
  have h := word_taylor_spec t ht
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rcases hyv with ⟨sin_t, cos_t, rfl, hs, hc, _, _⟩
  exact ⟨sin_t, cos_t, hy, hs, hc⟩

/-- End-to-end value spec of the extracted post-guard body (reduction → fold → Taylor → quadrant
swap/negate), in `Result` bind/match form; the post names the exact final integer `(sin, cos)`. -/
theorem word_fixed_sin_cos_value (x : I64) (hx : trig_window x) :
    (do
        let res ← (do
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
                  ok (core.result.Result.Ok r))
        match res with
        | core.result.Result.Err e => ok (core.result.Result.Err e)
        | core.result.Result.Ok r => do
            let u ← fixed_trig.fold_step r
            match u with
            | (_, _, negate, arg, swap) => do
                let sc ← fixed_trig.taylor_sin_cos arg
                let sin_t := sc.1
                let cos_t := sc.2
                let p ←
                  if swap.val = 1
                  then ok (cos_t, sin_t)
                  else
                    if swap.val = 2
                    then do
                         let i14 ← -. cos_t
                         ok (i14, sin_t)
                    else ok (sin_t, cos_t)
                let sin_out := p.1
                let cos_out := p.2
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
  apply Aeneas.Std.WP.spec_bind (word_fixed_sin_cos_reduction_value x hx)
  intro v hv
  obtain ⟨k, r, rfl, hk, hr⟩ := hv
  dsimp only
  have hrt := fixed_trig_reduction_remainder_range_tight x.val k.val hx.1 hx.2 hk
  have hr34 : |r.val| ≤ 2 ^ 34 := by
    rw [abs_le]
    exact ⟨by rw [hr]; exact hrt.1, by rw [hr]; exact hrt.2⟩
  have hr36 : |r.val| ≤ 2 ^ 36 := le_trans hr34 (by norm_num)
  apply Aeneas.Std.WP.spec_bind (fixed_trig_fold_step_spec r hr36)
  intro u hu
  obtain ⟨s, arg, negate, swap, rfl, hs, hneg, harg, hswap⟩ := hu
  dsimp only
  have harg32 : |arg.val| ≤ 2 ^ 32 := by
    rw [harg]; exact fixed_trig.fold_arg_abs_le r.val hr34
  apply Aeneas.Std.WP.spec_bind (word_taylor_spec arg harg32)
  intro sc hsc
  obtain ⟨sin_t, cos_t, rfl, hsin, hcos, hsinB, hcosB⟩ := hsc
  dsimp only
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
  by_cases h1 : swap.val = 1
  · simp only [if_pos h1, bind_tc_ok]
    by_cases hnt : negate = true
    · simp only [if_pos hnt]
      step*
      refine ⟨k, r, s, arg, sin_out1, cos_out1, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap, ?_, ?_⟩
      · simp [sin_out1_post, hnt, h1, ← hcos]
      · simp [cos_out1_post, hnt, h1, ← hsin]
    · simp only [if_neg hnt]
      rw [Aeneas.Std.WP.spec_ok]
      refine ⟨k, r, s, arg, cos_t, sin_t, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap, ?_, ?_⟩
      · simp [hnt, h1, ← hcos]
      · simp [hnt, h1, ← hsin]
  · simp only [if_neg h1]
    by_cases h2 : swap.val = 2
    · simp only [if_pos h2]
      step*
      · refine ⟨k, r, s, arg, _, _, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap, ?_, ?_⟩
        · simp only [← hsin, ← hcos]; simp_all
        · simp only [← hsin, ← hcos]; simp_all
      · refine ⟨k, r, s, arg, _, _, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap, ?_, ?_⟩
        · simp only [← hsin, ← hcos]; simp_all
        · simp only [← hsin, ← hcos]; simp_all
    · simp only [if_neg h2, bind_tc_ok]
      by_cases hnt : negate = true
      · simp only [if_pos hnt]
        step*
        refine ⟨k, r, s, arg, sin_out1, cos_out1, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap, ?_, ?_⟩
        · simp [sin_out1_post, hnt, h1, h2, ← hsin]
        · simp [cos_out1_post, hnt, h1, h2, ← hcos]
      · simp only [if_neg hnt]
        rw [Aeneas.Std.WP.spec_ok]
        refine ⟨k, r, s, arg, sin_t, cos_t, negate, swap, rfl, hk, hr, hs, hneg, harg, hswap, ?_, ?_⟩
        · simp [hnt, h1, h2, ← hsin]
        · simp [hnt, h1, h2, ← hcos]

/-! ## Argument displacement and the Lipschitz step

The extracted `r` is `x - k·TWO_PI_Q32`; the true reduced argument is `toReal x - k·2π`. Their
difference is exactly the `TWO_PI_Q32` defect scaled by `|k| <= 652`, so it is at most
`trig_delta = 652·(1/2)/2^32 = 326/2^32 = 652/2^33`. The sine/cosine of `r` then differ from those
of the true argument by at most `trig_delta` (Lipschitz). The fold's `arg` and the Taylor remainder
are the next slices. -/

/-- Displacement budget `652/2^33` (the `TWO_PI_Q32` half-unit defect times `|k| <= 652`). -/
def trig_delta : ℝ := 326 / q32

/-- `TWO_PI_Q32 = 26986075409` is within half a Q32.32 unit of `2π`. -/
theorem two_pi_q32_defect : |(26986075409 : ℝ) - 2 * Real.pi * q32| ≤ 1 / 2 := by
  have h1 : (3.14159265358979323846 : ℝ) < Real.pi := Real.pi_gt_d20
  have h2 : Real.pi < 3.14159265358979323847 := Real.pi_lt_d20
  unfold q32
  rw [abs_le]
  constructor <;> nlinarith

/-- Argument displacement: `toReal r` differs from the true reduction `toReal x − k·2π` by at most
`trig_delta`. -/
theorem trig_reduction_displacement (x r k : I64) (hx : trig_window x)
    (hk : k.val = (x.val * 683565276 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 26986075409) :
    |toReal r - (toReal x - (k.val : ℝ) * (2 * Real.pi))| ≤ trig_delta := by
  have hk652 := fixed_trig_shift_bound x.val k.val hx.1 hx.2 hk
  have hkabs : |(k.val : ℝ)| ≤ 652 := by
    rw [abs_le]
    exact_mod_cast hk652
  have hkey : toReal r - (toReal x - (k.val : ℝ) * (2 * Real.pi))
      = -((k.val : ℝ) * ((26986075409 : ℝ) - 2 * Real.pi * q32) / q32) := by
    unfold toReal
    rw [hr]
    push_cast
    field_simp [q32_ne_zero]
    ring_nf
  rw [hkey, abs_neg, abs_div, abs_of_pos q32_pos, abs_mul]
  calc |(k.val : ℝ)| * |(26986075409 : ℝ) - 2 * Real.pi * q32| / q32
      ≤ 652 * (1 / 2) / q32 := by
        apply div_le_div_of_nonneg_right _ (le_of_lt q32_pos)
        exact mul_le_mul hkabs two_pi_q32_defect (abs_nonneg _) (by norm_num)
    _ = trig_delta := by unfold trig_delta; ring

/-- Lipschitz transfer for `sin`: `sin (toReal r)` is within `trig_delta` of `sin (toReal x)`. -/
theorem trig_sin_reduction_transfer (x r k : I64) (hx : trig_window x)
    (hk : k.val = (x.val * 683565276 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 26986075409) :
    |Real.sin (toReal r) - Real.sin (toReal x)| ≤ trig_delta := by
  have hper : Real.sin (toReal x - (k.val : ℝ) * (2 * Real.pi)) = Real.sin (toReal x) :=
    Real.sin_sub_int_mul_two_pi (toReal x) k.val
  calc |Real.sin (toReal r) - Real.sin (toReal x)|
      = |Real.sin (toReal r) - Real.sin (toReal x - (k.val : ℝ) * (2 * Real.pi))| := by
          rw [← hper]
    _ ≤ |toReal r - (toReal x - (k.val : ℝ) * (2 * Real.pi))| :=
          Real.abs_sin_sub_sin_le _ _
    _ ≤ trig_delta := trig_reduction_displacement x r k hx hk hr

/-- Lipschitz transfer for `cos`: `cos (toReal r)` is within `trig_delta` of `cos (toReal x)`. -/
theorem trig_cos_reduction_transfer (x r k : I64) (hx : trig_window x)
    (hk : k.val = (x.val * 683565276 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 26986075409) :
    |Real.cos (toReal r) - Real.cos (toReal x)| ≤ trig_delta := by
  have hper : Real.cos (toReal x - (k.val : ℝ) * (2 * Real.pi)) = Real.cos (toReal x) :=
    Real.cos_sub_int_mul_two_pi (toReal x) k.val
  calc |Real.cos (toReal r) - Real.cos (toReal x)|
      = |Real.cos (toReal r) - Real.cos (toReal x - (k.val : ℝ) * (2 * Real.pi))| := by
          rw [← hper]
    _ ≤ |toReal r - (toReal x - (k.val : ℝ) * (2 * Real.pi))| :=
          Real.abs_cos_sub_cos_le _ _
    _ ≤ trig_delta := trig_reduction_displacement x r k hx hk hr


end

end apeiron_word
