import Numeric.Engine.Exp.Part3

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- The reduction displacement budget: `31 · 10^-9`. -/
def deltaRed : ℝ := 31 / 10 ^ 9

theorem fixed_exp_ln2_q32_defect : |(2977044472 : ℝ) / q32 - Real.log 2| ≤ 10 ^ (-9 : ℤ) := by
  have h1 := Real.log_two_gt_d9
  have h2 := Real.log_two_lt_d9
  have hq : (2977044472 : ℝ) / q32 = 2977044472 / 4294967296 := by unfold q32; norm_num
  rw [hq]
  rw [abs_sub_le_iff]
  constructor <;> linarith

theorem fixed_exp_reduction_displacement (x k r : I64) (hx : window x)
    (hk : k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 2977044472) :
    |toReal r - (toReal x - (k.val : ℝ) * Real.log 2)| ≤ deltaRed := by
  have hsb := fixed_exp_shift_recip_shift_bound x.val k.val hx.1 hx.2 hk
  have hkabs : |(k.val : ℝ)| ≤ 31 := by
    rw [abs_le]
    constructor
    · have : (-(31:ℤ)) ≤ k.val := by omega
      exact_mod_cast this
    · have : k.val ≤ 31 := by omega
      exact_mod_cast this
  have hdef := fixed_exp_ln2_q32_defect
  have htr : toReal r = toReal x - (k.val : ℝ) * ((2977044472 : ℝ) / q32) := by
    unfold toReal
    rw [hr]
    push_cast
    ring
  rw [htr]
  have hrw : toReal x - (k.val : ℝ) * ((2977044472 : ℝ) / q32)
        - (toReal x - (k.val : ℝ) * Real.log 2)
      = -(k.val : ℝ) * ((2977044472 : ℝ) / q32 - Real.log 2) := by ring
  rw [hrw, abs_mul, abs_neg]
  unfold deltaRed
  calc |(k.val : ℝ)| * |(2977044472 : ℝ) / q32 - Real.log 2| ≤ 31 * (10 ^ (-9 : ℤ)) :=
        mul_le_mul hkabs hdef (abs_nonneg _) (by norm_num)
    _ = 31 / 10 ^ 9 := by norm_num [zpow_neg]

theorem fixed_exp_exponential_transfer (x k r : I64)
    (hδ : |toReal r - (toReal x - (k.val : ℝ) * Real.log 2)| ≤ deltaRed) :
    |Real.exp (toReal r + (k.val : ℝ) * Real.log 2) - Real.exp (toReal x)| ≤
      Real.exp (toReal x) * (2 * deltaRed) := by
  have hδ0 : 0 ≤ deltaRed := by unfold deltaRed; norm_num
  have he : |toReal r + (k.val : ℝ) * Real.log 2 - toReal x| ≤ deltaRed := by
    rw [show toReal r + (k.val : ℝ) * Real.log 2 - toReal x
          = toReal r - (toReal x - (k.val : ℝ) * Real.log 2) by ring]
    exact hδ
  have hxpos : 0 < Real.exp (toReal x) := Real.exp_pos _
  rw [show toReal r + (k.val : ℝ) * Real.log 2
        = toReal x + (toReal r + (k.val : ℝ) * Real.log 2 - toReal x) by ring]
  rw [Real.exp_add]
  rw [show Real.exp (toReal x) * Real.exp (toReal r + (k.val : ℝ) * Real.log 2 - toReal x)
        - Real.exp (toReal x)
        = Real.exp (toReal x) * (Real.exp (toReal r + (k.val : ℝ) * Real.log 2 - toReal x) - 1)
        by ring]
  rw [abs_mul, abs_of_pos hxpos]
  have he1 : |toReal r + (k.val : ℝ) * Real.log 2 - toReal x| ≤ 1 := by
    have hδ1 : deltaRed ≤ 1 := by unfold deltaRed; norm_num
    linarith
  have hb := Real.abs_exp_sub_one_le he1
  have hb2 : |Real.exp (toReal r + (k.val : ℝ) * Real.log 2 - toReal x) - 1| ≤ 2 * deltaRed := by
    nlinarith [hb, he, abs_nonneg (toReal r + (k.val : ℝ) * Real.log 2 - toReal x)]
  exact mul_le_mul_of_nonneg_left hb2 (le_of_lt hxpos)

/-! ## Horner polynomial error (E1b, part 1): truncation remainder and coefficient defects -/

/-- The budget for the Horner polynomial error: `2·10^-7`, dominating the degree-7 remainder at
`|t| ≤ 1/2` (`9/82575360 ≈ 1.09e-7`) plus the coefficient defects and the per-stage rounding. -/
def epsHorner : ℝ := 2 * 10 ^ (-7 : ℤ)

/-- Degree-7 truncation remainder: `|∑_{m<8} t^m/m! − exp t| ≤ |t|^8 · 9/322560` for `|t| ≤ 1`
(`Real.exp_bound` at `n = 8`). -/
theorem fixed_exp_taylor7_remainder (t : ℝ) (ht : |t| ≤ 1) :
    |(∑ m ∈ Finset.range 8, t ^ m / (m.factorial : ℝ)) - Real.exp t| ≤
      |t| ^ 8 * (9 / 322560) := by
  have h := Real.exp_bound ht (n := 8) (by norm_num)
  calc |(∑ m ∈ Finset.range 8, t ^ m / (m.factorial : ℝ)) - Real.exp t|
      = |Real.exp t - ∑ m ∈ Finset.range 8, t ^ m / (m.factorial : ℝ)| := abs_sub_comm _ _
    _ ≤ |t| ^ 8 * ((8:ℕ).succ / ((8:ℕ).factorial * 8)) := h
    _ = |t| ^ 8 * (9 / 322560) := by norm_num

/-- Coefficient defect `1/7!`: `|852176/q32 − 1/7!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_7 :
    |(852176 : ℝ) / q32 - 1 / (Nat.factorial 7 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/6!`: `|5965232/q32 − 1/6!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_6 :
    |(5965232 : ℝ) / q32 - 1 / (Nat.factorial 6 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/5!`: `|35791394/q32 − 1/5!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_5 :
    |(35791394 : ℝ) / q32 - 1 / (Nat.factorial 5 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/4!`: `|178956971/q32 − 1/4!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_4 :
    |(178956971 : ℝ) / q32 - 1 / (Nat.factorial 4 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/3!`: `|715827883/q32 − 1/3!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_3 :
    |(715827883 : ℝ) / q32 - 1 / (Nat.factorial 3 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/2!`: `|2147483648/q32 − 1/2!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_2 :
    |(2147483648 : ℝ) / q32 - 1 / (Nat.factorial 2 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/1!`: `|4294967296/q32 − 1/1!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_1 :
    |(4294967296 : ℝ) / q32 - 1 / (Nat.factorial 1 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-- Coefficient defect `1/0!`: `|4294967296/q32 − 1/0!| ≤ 1/q32`. -/
theorem fixed_exp_coeff_defect_0 :
    |(4294967296 : ℝ) / q32 - 1 / (Nat.factorial 0 : ℝ)| ≤ 1 / q32 := by
  unfold q32; norm_num [Nat.factorial]

/-! ## E1b (part 2): per-stage `round_div_2p32` ledger -/

/-- Integer form of the per-stage rounding error: `round_q32 v` differs from `v/2^32` by at most
`2^31` in the integer scale, i.e. `|round_q32 v · 2^32 − v| ≤ 2^31`. -/
theorem fixed_exp_round_q32_int_error (v : ℤ) :
    |fixed_exp.round_q32 v * (2:ℤ)^32 - v| ≤ 2^31 := by
  unfold fixed_exp.round_q32
  rw [abs_le]
  by_cases hv : 0 ≤ v
  · rw [if_pos hv, Int.shiftRight_eq_div_pow]
    have hd := Int.mul_ediv_add_emod (v + 2^31) ((2:ℤ)^32)
    have h0 : 0 ≤ (v + 2^31) % (2:ℤ)^32 := Int.emod_nonneg _ (by norm_num)
    have hlt : (v + 2^31) % (2:ℤ)^32 < (2:ℤ)^32 := Int.emod_lt_of_pos _ (by norm_num)
    norm_num at hd h0 hlt ⊢
    constructor <;> omega
  · rw [if_neg hv, Int.shiftRight_eq_div_pow]
    have hd := Int.mul_ediv_add_emod (-v + 2^31) ((2:ℤ)^32)
    have h0 : 0 ≤ (-v + 2^31) % (2:ℤ)^32 := Int.emod_nonneg _ (by norm_num)
    have hlt : (-v + 2^31) % (2:ℤ)^32 < (2:ℤ)^32 := Int.emod_lt_of_pos _ (by norm_num)
    norm_num at hd h0 hlt ⊢
    constructor <;> omega

/-- Real form of the per-stage rounding error: the extracted `round_q32` maps an `i128` product to a
Q32.32 value within `1/(2·q32)` of the exact product value `v/q32²`. -/
theorem fixed_exp_round_q32_real_error (v : ℤ) :
    |((fixed_exp.round_q32 v : ℤ) : ℝ)/q32 - (v:ℝ)/(q32*q32)| ≤ 1/(2*q32) := by
  have h := fixed_exp_round_q32_int_error v
  have h' : |(((fixed_exp.round_q32 v * (2:ℤ)^32 - v : ℤ)) : ℝ)| ≤ (2:ℝ)^31 := by
    rw [abs_le] at h ⊢
    constructor
    · exact_mod_cast h.1
    · exact_mod_cast h.2
  rw [show ((fixed_exp.round_q32 v : ℤ) : ℝ)/q32 - (v:ℝ)/(q32*q32)
        = (((fixed_exp.round_q32 v * (2:ℤ)^32 - v : ℤ)) : ℝ) / (q32*q32) by
    push_cast; unfold q32; ring]
  rw [abs_div, abs_of_pos (mul_pos q32_pos q32_pos)]
  calc |(((fixed_exp.round_q32 v * (2:ℤ)^32 - v : ℤ)) : ℝ)| / (q32*q32)
      ≤ (2:ℝ)^31 / (q32*q32) := div_le_div_of_nonneg_right h' (le_of_lt (mul_pos q32_pos q32_pos))
    _ = 1/(2*q32) := by unfold q32; norm_num
/-- One integer Horner stage: round the product and add the coefficient. -/
def fixed_exp.hand (a r c : ℤ) : ℤ := fixed_exp.round_q32 (a * r) + c

/-- One exact real Horner stage with coefficient `c/q32`. -/
def fixed_exp.handReal (P t : ℝ) (c : ℤ) : ℝ := P * t + (c : ℝ) / q32

def fixed_exp.handRealF (P t d : ℝ) : ℝ := P * t + d

theorem fixed_exp.horner_acc_eq (r : ℤ) :
    fixed_exp.horner_acc r =
      fixed_exp.hand (fixed_exp.hand (fixed_exp.hand (fixed_exp.hand
        (fixed_exp.hand (fixed_exp.hand (fixed_exp.hand 852176 r 5965232) r 35791394)
          r 178956971) r 715827883) r 2147483648) r 4294967296) r 4294967296 := by
  unfold fixed_exp.horner_acc fixed_exp.hand
  rfl

def fixed_exp.hornerReal (t : ℝ) : ℝ :=
  fixed_exp.handReal (fixed_exp.handReal (fixed_exp.handReal (fixed_exp.handReal
    (fixed_exp.handReal (fixed_exp.handReal (fixed_exp.handReal (852176/q32) t 5965232)
      t 35791394) t 178956971) t 715827883) t 2147483648) t 4294967296) t 4294967296

theorem fixed_exp.hand_bound (a r c : ℤ) (P B t : ℝ) (ht : t = (r : ℝ)/q32)
    (hB : |(a : ℝ)/q32 - P| ≤ B) :
    |((fixed_exp.hand a r c : ℤ) : ℝ)/q32 - fixed_exp.handReal P t c| ≤
      |t| * B + 1/(2*q32) := by
  unfold fixed_exp.hand fixed_exp.handReal
  have herr := fixed_exp_round_q32_real_error (a * r)
  rw [show (((fixed_exp.round_q32 (a * r) + c : ℤ)) : ℝ)/q32 - (P * t + (c : ℝ)/q32)
        = (((fixed_exp.round_q32 (a * r) : ℤ) : ℝ)/q32 - ((a * r : ℤ) : ℝ)/(q32*q32))
          + t * ((a : ℝ)/q32 - P) by
    rw [ht]; push_cast; ring]
  calc |(((fixed_exp.round_q32 (a * r) : ℤ) : ℝ)/q32 - ((a * r : ℤ) : ℝ)/(q32*q32))
          + t * ((a : ℝ)/q32 - P)|
      ≤ |(((fixed_exp.round_q32 (a * r) : ℤ) : ℝ)/q32 - ((a * r : ℤ) : ℝ)/(q32*q32))|
          + |t * ((a : ℝ)/q32 - P)| := abs_add_le _ _
    _ ≤ |t| * B + 1/(2*q32) := by
        have h2 : |t * ((a : ℝ)/q32 - P)| ≤ |t| * B := by
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_left hB (abs_nonneg t)
        linarith [herr, h2]

theorem fixed_exp.horner_round_error (r : ℤ) (t : ℝ) (ht : t = (r : ℝ)/q32)
    (hs : |t| ≤ 1) :
    |((fixed_exp.horner_acc r : ℤ) : ℝ)/q32 - fixed_exp.hornerReal t| ≤ 7 * (1/(2*q32)) := by
  rw [fixed_exp.horner_acc_eq]
  unfold fixed_exp.hornerReal
  set A0 := fixed_exp.hand 852176 r 5965232 with hA0
  set A1 := fixed_exp.hand A0 r 35791394 with hA1
  set A2 := fixed_exp.hand A1 r 178956971 with hA2
  set A3 := fixed_exp.hand A2 r 715827883 with hA3
  set A4 := fixed_exp.hand A3 r 2147483648 with hA4
  set A5 := fixed_exp.hand A4 r 4294967296 with hA5
  set A6 := fixed_exp.hand A5 r 4294967296 with hA6
  set P0 := fixed_exp.handReal (852176/q32) t 5965232 with hP0
  set P1 := fixed_exp.handReal P0 t 35791394 with hP1
  set P2 := fixed_exp.handReal P1 t 178956971 with hP2
  set P3 := fixed_exp.handReal P2 t 715827883 with hP3
  set P4 := fixed_exp.handReal P3 t 2147483648 with hP4
  set P5 := fixed_exp.handReal P4 t 4294967296 with hP5
  set P6 := fixed_exp.handReal P5 t 4294967296 with hP6
  have hD : (0:ℝ) < 1/(2*q32) := by
    apply div_pos; norm_num; exact mul_pos (by norm_num) q32_pos
  have htD : |t| * (1/(2*q32)) ≤ 1/(2*q32) := mul_le_of_le_one_left (le_of_lt hD) hs
  have e0 : |(A0 : ℝ)/q32 - P0| ≤ (1:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound 852176 r 5965232 (852176/q32) 0 t ht
      (by norm_num : |((852176:ℤ) : ℝ)/q32 - (852176:ℝ)/q32| ≤ 0)
    rw [hA0, hP0]; simpa using h
  have e1 : |(A1 : ℝ)/q32 - P1| ≤ (2:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound A0 r 35791394 P0 (1*(1/(2*q32))) t ht (by rw [hA0, hP0]; simpa using e0)
    have ht1 : |t| * (1*(1/(2*q32))) ≤ 1*(1/(2*q32)) := by simpa using htD
    rw [hA1, hP1]; linarith [h, ht1]
  have e2 : |(A2 : ℝ)/q32 - P2| ≤ (3:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound A1 r 178956971 P1 (2*(1/(2*q32))) t ht (by rw [hA1, hP1]; simpa using e1)
    have ht1 : |t| * (2*(1/(2*q32))) ≤ 2*(1/(2*q32)) := by nlinarith [htD, abs_nonneg t, le_of_lt hD]
    rw [hA2, hP2]; linarith [h, ht1]
  have e3 : |(A3 : ℝ)/q32 - P3| ≤ (4:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound A2 r 715827883 P2 (3*(1/(2*q32))) t ht (by rw [hA2, hP2]; simpa using e2)
    have ht1 : |t| * (3*(1/(2*q32))) ≤ 3*(1/(2*q32)) := by nlinarith [htD, abs_nonneg t, le_of_lt hD]
    rw [hA3, hP3]; linarith [h, ht1]
  have e4 : |(A4 : ℝ)/q32 - P4| ≤ (5:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound A3 r 2147483648 P3 (4*(1/(2*q32))) t ht (by rw [hA3, hP3]; simpa using e3)
    have ht1 : |t| * (4*(1/(2*q32))) ≤ 4*(1/(2*q32)) := by nlinarith [htD, abs_nonneg t, le_of_lt hD]
    rw [hA4, hP4]; linarith [h, ht1]
  have e5 : |(A5 : ℝ)/q32 - P5| ≤ (6:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound A4 r 4294967296 P4 (5*(1/(2*q32))) t ht (by rw [hA4, hP4]; simpa using e4)
    have ht1 : |t| * (5*(1/(2*q32))) ≤ 5*(1/(2*q32)) := by nlinarith [htD, abs_nonneg t, le_of_lt hD]
    rw [hA5, hP5]; linarith [h, ht1]
  have e6 : |(A6 : ℝ)/q32 - P6| ≤ (7:ℝ) * (1/(2*q32)) := by
    have h := fixed_exp.hand_bound A5 r 4294967296 P5 (6*(1/(2*q32))) t ht (by rw [hA5, hP5]; simpa using e5)
    have ht1 : |t| * (6*(1/(2*q32))) ≤ 6*(1/(2*q32)) := by nlinarith [htD, abs_nonneg t, le_of_lt hD]
    rw [hA6, hP6]; linarith [h, ht1]
  exact e6

/-- The exact real Horner with coefficients `c/q32`, in `handRealF` form (same nesting as
`fixed_exp.hornerReal`). -/
def fixed_exp.hornerRealF (t : ℝ) : ℝ :=
  handRealF
    (handRealF
      (handRealF
        (handRealF
          (handRealF
            (handRealF
              (handRealF (852176/q32) t (5965232/q32))
              t (35791394/q32))
            t (178956971/q32))
          t (715827883/q32))
        t (2147483648/q32))
      t (4294967296/q32))
    t (4294967296/q32)

def fixed_exp.taylorReal (t : ℝ) : ℝ :=
  handRealF
    (handRealF
      (handRealF
        (handRealF
          (handRealF
            (handRealF
              (handRealF (1/(Nat.factorial 7 : ℝ)) t (1/(Nat.factorial 6 : ℝ)))
              t (1/(Nat.factorial 5 : ℝ)))
            t (1/(Nat.factorial 4 : ℝ)))
          t (1/(Nat.factorial 3 : ℝ)))
        t (1/(Nat.factorial 2 : ℝ)))
      t (1/(Nat.factorial 1 : ℝ)))
    t (1/(Nat.factorial 0 : ℝ))

theorem fixed_exp.hornerReal_eq_F (t : ℝ) : fixed_exp.hornerReal t = fixed_exp.hornerRealF t := by
  unfold fixed_exp.hornerReal fixed_exp.hornerRealF fixed_exp.handReal fixed_exp.handRealF
  rfl

theorem fixed_exp.taylorReal_eq_sum (t : ℝ) :
    fixed_exp.taylorReal t = ∑ m ∈ Finset.range 8, t ^ m / (m.factorial : ℝ) := by
  unfold fixed_exp.taylorReal fixed_exp.handRealF
  rw [Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_succ, Finset.sum_range_succ, Finset.sum_range_succ,
      Finset.sum_range_succ, Finset.sum_range_succ]
  norm_num
  ring

theorem fixed_exp.handRealF_defect (P Q t d e B D : ℝ) (hB : |P - Q| ≤ B) (hD : |d - e| ≤ D) :
    |fixed_exp.handRealF P t d - fixed_exp.handRealF Q t e| ≤ |t| * B + D := by
  unfold fixed_exp.handRealF
  have h1 : |P * t - Q * t| ≤ |t| * B := by
    rw [← sub_mul, abs_mul, mul_comm (|P - Q|) (|t|)]
    exact mul_le_mul_of_nonneg_left hB (abs_nonneg t)
  have htri := abs_add_le (P * t - Q * t) (d - e)
  rw [show P * t + d - (Q * t + e) = (P * t - Q * t) + (d - e) by ring]
  linarith [htri, h1, hD]

/-- Horner evaluation over an explicit coefficient list: `foldl (fun acc c => acc*t + c) init cs`,
so the defect chain is proved by list induction with small per-step terms. -/
def fixed_exp.realHorner (t init : ℝ) (cs : List ℝ) : ℝ :=
  cs.foldl (fun acc c => acc * t + c) init

def fixed_exp.Cs : List ℝ :=
  [5965232/q32, 35791394/q32, 178956971/q32, 715827883/q32, 2147483648/q32, 4294967296/q32, 4294967296/q32]

def fixed_exp.Ds : List ℝ :=
  [1/(Nat.factorial 6 : ℝ), 1/(Nat.factorial 5 : ℝ), 1/(Nat.factorial 4 : ℝ),
   1/(Nat.factorial 3 : ℝ), 1/(Nat.factorial 2 : ℝ), 1/(Nat.factorial 1 : ℝ),
   1/(Nat.factorial 0 : ℝ)]

def fixed_exp.Ps : List (ℝ × ℝ) :=
  [(5965232/q32, 1/(Nat.factorial 6 : ℝ)), (35791394/q32, 1/(Nat.factorial 5 : ℝ)),
   (178956971/q32, 1/(Nat.factorial 4 : ℝ)), (715827883/q32, 1/(Nat.factorial 3 : ℝ)),
   (2147483648/q32, 1/(Nat.factorial 2 : ℝ)), (4294967296/q32, 1/(Nat.factorial 1 : ℝ)),
   (4294967296/q32, 1/(Nat.factorial 0 : ℝ))]

theorem fixed_exp.hornerReal_eq_list (t : ℝ) :
    fixed_exp.hornerReal t = fixed_exp.realHorner t (852176/q32) fixed_exp.Cs := by
  unfold fixed_exp.hornerReal fixed_exp.handReal fixed_exp.realHorner fixed_exp.Cs
  rfl

theorem fixed_exp.taylorReal_eq_list (t : ℝ) :
    fixed_exp.taylorReal t = fixed_exp.realHorner t (1/(Nat.factorial 7 : ℝ)) fixed_exp.Ds := by
  unfold fixed_exp.taylorReal fixed_exp.handRealF fixed_exp.realHorner fixed_exp.Ds
  rfl

theorem fixed_exp.realHorner_cons (t init c : ℝ) (cs : List ℝ) :
    fixed_exp.realHorner t init (c :: cs) = fixed_exp.realHorner t (init * t + c) cs := rfl

theorem fixed_exp.realHorner_step (t a b c d B D : ℝ) (hs : |t| ≤ 1)
    (hB : |a - b| ≤ B) (hD : |c - d| ≤ D) :
    |(a * t + c) - (b * t + d)| ≤ B + D := by
  have h1 : |a * t - b * t| ≤ B := by
    rw [← sub_mul, abs_mul]
    have hb0 : 0 ≤ B := le_trans (abs_nonneg _) hB
    have := mul_le_mul hB hs (abs_nonneg t) hb0
    simpa using this
  have htri := abs_add_le (a * t - b * t) (c - d)
  rw [show a * t + c - (b * t + d) = (a * t - b * t) + (c - d) by ring]
  linarith [htri, h1, hD]

theorem fixed_exp.realHorner_pair_le (t : ℝ) (hs : |t| ≤ 1) :
    ∀ (ps : List (ℝ × ℝ)) (init₁ init₂ B : ℝ), |init₁ - init₂| ≤ B →
      (∀ p ∈ ps, |p.1 - p.2| ≤ 1/q32) →
      |fixed_exp.realHorner t init₁ (ps.map Prod.fst) - fixed_exp.realHorner t init₂ (ps.map Prod.snd)|
        ≤ B + ps.length * (1/q32) := by
  intro ps
  induction ps with
  | nil =>
      intro init₁ init₂ B hB _
      simpa [fixed_exp.realHorner] using hB
  | cons p ps ih =>
      intro init₁ init₂ B hB hpts
      have hp : |p.1 - p.2| ≤ 1/q32 := hpts p (by simp)
      have hpts' : ∀ q ∈ ps, |q.1 - q.2| ≤ 1/q32 := by
        intro q hq; exact hpts q (by simp [hq])
      have hstep := fixed_exp.realHorner_step t init₁ init₂ p.1 p.2 B (1/q32) hs hB hp
      have ih' := ih (init₁ * t + p.1) (init₂ * t + p.2) (B + 1/q32) hstep hpts'
      simp only [List.map_cons, List.length_cons, fixed_exp.realHorner_cons]
      push_cast
      linarith [ih']


theorem fixed_exp.hornerReal_taylor_error (t : ℝ) (hs : |t| ≤ 1) :
    |fixed_exp.hornerReal t - fixed_exp.taylorReal t| ≤ 8 * (1/q32) := by
  rw [fixed_exp.hornerReal_eq_list, fixed_exp.taylorReal_eq_list,
      show fixed_exp.Cs = fixed_exp.Ps.map Prod.fst from rfl,
      show fixed_exp.Ds = fixed_exp.Ps.map Prod.snd from rfl]
  have hinit : |(852176:ℝ)/q32 - 1/(Nat.factorial 7 : ℝ)| ≤ 1/q32 := fixed_exp_coeff_defect_7
  have hpts : ∀ p ∈ fixed_exp.Ps, |p.1 - p.2| ≤ 1/q32 := by
    intro p hp
    simp only [fixed_exp.Ps, List.mem_cons, List.not_mem_nil, or_false] at hp
    rcases hp with h|h|h|h|h|h|h
    · subst h; exact fixed_exp_coeff_defect_6
    · subst h; exact fixed_exp_coeff_defect_5
    · subst h; exact fixed_exp_coeff_defect_4
    · subst h; exact fixed_exp_coeff_defect_3
    · subst h; exact fixed_exp_coeff_defect_2
    · subst h; exact fixed_exp_coeff_defect_1
    · subst h; exact fixed_exp_coeff_defect_0
  have h := fixed_exp.realHorner_pair_le t hs fixed_exp.Ps (852176/q32)
    (1/(Nat.factorial 7 : ℝ)) (1/q32) hinit hpts
  have hlen : (fixed_exp.Ps.length : ℝ) * (1/q32) = 7 * (1/q32) := by norm_num [fixed_exp.Ps]
  rw [hlen] at h
  linarith [h]


theorem fixed_exp.horner_acc_error (r : ℤ) (t : ℝ) (ht : t = (r : ℝ)/q32)
    (hs : |t| ≤ 1/2) :
    |((fixed_exp.horner_acc r : ℤ) : ℝ)/q32 - Real.exp t| ≤ epsHorner := by
  have hs1 : |t| ≤ 1 := le_trans hs (by norm_num)
  have hround := fixed_exp.horner_round_error r t ht hs1
  have hdef := fixed_exp.hornerReal_taylor_error t hs1
  have hsum := fixed_exp.taylorReal_eq_sum t
  have htrunc := fixed_exp_taylor7_remainder t hs1
  set A := ((fixed_exp.horner_acc r : ℤ) : ℝ)/q32 with hA
  set HR := fixed_exp.hornerReal t with hHR
  set TR := fixed_exp.taylorReal t with hTR
  have h1 : |A - Real.exp t| ≤ |A - HR| + |HR - Real.exp t| := by
    rw [show A - Real.exp t = (A - HR) + (HR - Real.exp t) by ring]
    exact abs_add_le _ _
  have h2 : |HR - Real.exp t| ≤ |HR - TR| + |TR - Real.exp t| := by
    rw [show HR - Real.exp t = (HR - TR) + (TR - Real.exp t) by ring]
    exact abs_add_le _ _
  have h4 : |TR - Real.exp t| ≤ |t| ^ 8 * (9 / 322560) := by
    rw [hsum]; exact htrunc
  have h5 : |HR - Real.exp t| ≤ 8 * (1/q32) + |t| ^ 8 * (9 / 322560) := by
    linarith [h2, hdef, h4]
  have h6 : |A - Real.exp t| ≤ 7 * (1/(2*q32)) + 8 * (1/q32) + |t| ^ 8 * (9 / 322560) := by
    linarith [h1, hround, h5]
  have ht8 : |t| ^ 8 ≤ (1/2:ℝ) ^ 8 := pow_le_pow_left₀ (abs_nonneg t) hs 8
  have hmul : |t| ^ 8 * (9 / 322560) ≤ (1/2:ℝ) ^ 8 * (9 / 322560) :=
    mul_le_mul_of_nonneg_right ht8 (by norm_num)
  have hnum : 7 * (1/(2*q32)) + 8 * (1/q32) + (1/2:ℝ) ^ 8 * (9 / 322560) ≤ epsHorner := by
    unfold epsHorner q32; norm_num
  linarith [h6, hmul, hnum]
/-! ### Gap A discharge: body-level `exp_scale` refinement with the exact integer value

`I64.val` has no range lemma in this Aeneas model, so the landed `fixed_exp_scale_*` posts drop the
guard ranges and `v.val = (acc <<< k).bmod 2^128` does not give `v.val = acc.val·2^k` (the `bmod` can
wrap). The theorems below re-run the bodies and carry the range/value relation explicitly; each takes
the range facts it needs as hypotheses. -/

/-- `bmod 2^128` is the identity on `[0, 2^63 - 1]`. -/
theorem fixed_exp_bmod128_low (a : ℤ) (ha0 : 0 ≤ a) (hlt : a ≤ 2^63 - 1) :
    a.bmod (2^128) = a := by
  have hlt128 : a < 2^128 := by have := hlt; norm_num at this ⊢; linarith
  have hcond : a % (2^128) < (2^128 + 1) / 2 := by
    rw [Int.emod_eq_of_lt ha0 hlt128]
    have := hlt; norm_num at this ⊢; linarith
  rw [Int.bmod_eq_emod_of_lt hcond]
  exact Int.emod_eq_of_lt ha0 hlt128

theorem fixed_exp_scale_nonneg_exact (k : I64) (acc6 : I128)
    (h0 : 0 ≤ k.val) (h64 : k.val < 64) (hnn : 0 ≤ acc6.val)
    (hub : acc6.val * 2 ^ k.val.toNat ≤ 2^63 - 1) :
    fixed_exp.exp_scale k acc6 ⦃ z =>
      ∃ v : I64, z = core.result.Result.Ok v ∧ v.val = acc6.val * 2 ^ k.val.toNat ⦄ := by
  unfold fixed_exp.exp_scale
  have hge : (0#i64 : I64) ≤ k := (IScalar.le_equiv _ _).2 (by simpa using h0)
  rw [if_pos hge]
  have hge64 : ¬ ((64#i64 : I64) ≤ k) := by
    intro h
    have h' : (64:ℤ) ≤ k.val := by simpa using (IScalar.le_equiv _ _).1 h
    omega
  rw [if_neg hge64]
  have hMAX : (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = 2^63 - 1 := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    apply fixed_exp_bmod_self_64 <;> norm_num [I64.max, I64.rMax, I64.numBits]
  have hMIN : (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -2^63 := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    apply fixed_exp_bmod_self_64 <;> norm_num [I64.min, I64.rMin, I64.numBits]
  simp only [lift, bind_tc_ok]
  step*
  · exact fixed_exp_scale_shift_count_lt k h0 h64
  · rename_i hgt
    exfalso
    have hn : (IScalar.hcast UScalarTy.U32 k).val = k.val.toNat := by
      rw [IScalar.hcast_val_eq]
      norm_num [UScalarTy.numBits]
      have h : k.val % 4294967296 = k.val := Int.emod_eq_of_lt h0 (by omega)
      rw [h]
    have hscaled : scaled.val = acc6.val * 2 ^ k.val.toNat := by
      rw [scaled_post, hn, Int.shiftLeft_eq]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_bmod128_low (acc6.val * 2 ^ k.val.toNat) (by positivity) hub
    rw [gt_iff_lt, IScalar.lt_equiv, hMAX] at hgt
    omega
  · rename_i hgt hlt
    exfalso
    have hn : (IScalar.hcast UScalarTy.U32 k).val = k.val.toNat := by
      rw [IScalar.hcast_val_eq]
      norm_num [UScalarTy.numBits]
      have h : k.val % 4294967296 = k.val := Int.emod_eq_of_lt h0 (by omega)
      rw [h]
    have hscaled : scaled.val = acc6.val * 2 ^ k.val.toNat := by
      rw [scaled_post, hn, Int.shiftLeft_eq]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_bmod128_low (acc6.val * 2 ^ k.val.toNat) (by positivity) hub
    rw [IScalar.lt_equiv, hMIN] at hlt
    have h0s : 0 ≤ scaled.val := by rw [hscaled]; positivity
    omega
  · rename_i hgt hlt
    refine ⟨IScalar.cast IScalarTy.I64 scaled, rfl, ?_⟩
    have hn : (IScalar.hcast UScalarTy.U32 k).val = k.val.toNat := by
      rw [IScalar.hcast_val_eq]
      norm_num [UScalarTy.numBits]
      have h : k.val % 4294967296 = k.val := Int.emod_eq_of_lt h0 (by omega)
      rw [h]
    have hscaled : scaled.val = acc6.val * 2 ^ k.val.toNat := by
      rw [scaled_post, hn, Int.shiftLeft_eq]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_bmod128_low (acc6.val * 2 ^ k.val.toNat) (by positivity) hub
    have hle : scaled.val ≤ 2^63 - 1 := by
      rw [gt_iff_lt, IScalar.lt_equiv, hMAX] at hgt; omega
    have hge' : -2^63 ≤ scaled.val := by
      rw [IScalar.lt_equiv, hMIN] at hlt; omega
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    rw [show (18446744073709551616 : ℕ) = 2^64 by norm_num]
    rw [fixed_exp_bmod_self_64 scaled.val hge' hle]
    exact hscaled


/-- `(2:ℤ)^m` for `m ≤ 62` is `bmod 2^128`-stable. -/
theorem fixed_exp_two_pow_bmod (m : ℕ) (hm : m ≤ 62) :
    ((2:ℤ)^m).bmod (2^128) = (2:ℤ)^m := by
  apply fixed_exp_bmod128_low
  · positivity
  · have h : (2:ℤ)^m ≤ 2^62 := pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) hm
    calc (2:ℤ)^m ≤ 2^62 := h
      _ ≤ 2^63 - 1 := by norm_num

theorem fixed_exp_scale_neg_core_exact (n : U64) (acc6 : I128) (hn : n.val < 64) (hn1 : 1 ≤ n.val)
    (ha : |acc6.val| ≤ 2^40)
    (hlo : -2^63 ≤ (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val)
    (hhi : (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val ≤ 2^63 - 1) :
    (do
          let i26 ← n - 1#u64
          let half ← 1#i128 <<< i26
          let i27 ← acc6 + half
          let rounded ← i27 >>> n
          let i28 ← lift (IScalar.cast .I128 core.num.I64.MAX)
          if rounded > i28
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let i29 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if rounded < i29
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i30 ← lift (IScalar.cast .I64 rounded)
              ok (core.result.Result.Ok i30)) ⦃ z =>
      ∃ v : I64, z = core.result.Result.Ok v ∧
        (v.val : ℤ) = (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val ⦄ := by
  have hMAX : (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = 2^63 - 1 := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    apply fixed_exp_bmod_self_64 <;> norm_num [I64.max, I64.rMax, I64.numBits]
  have hMIN : (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -2^63 := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    apply fixed_exp_bmod_self_64 <;> norm_num [I64.min, I64.rMin, I64.numBits]
  simp only [lift, bind_tc_ok]
  step*
  · -- add lower bound
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_two_pow_bmod i26.val (by rw [i26_post]; omega)
    rw [hhalf]
    norm_num [I128.min, I128.numBits]
    have ha' := abs_le.mp ha
    have hp : (0:ℤ) ≤ (2:ℤ)^i26.val := by positivity
    nlinarith [ha'.1, ha'.2, hp]
  · -- add upper bound
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_two_pow_bmod i26.val (by rw [i26_post]; omega)
    rw [hhalf]
    norm_num [I128.max, I128.numBits]
    have ha' := abs_le.mp ha
    have hp : (2:ℤ)^i26.val ≤ 2^62 := by
      have hb : i26.val ≤ 62 := by rw [i26_post]; omega
      exact pow_le_pow_right₀ (by norm_num : (1:ℤ) ≤ 2) hb
    nlinarith [ha'.2, hp]
  · -- guard scaled > MAX
    rename_i hgt
    exfalso
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_two_pow_bmod i26.val (by rw [i26_post]; omega)
    have hrv : rounded.val = (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val := by
      rw [rounded_post, i27_post, hhalf, i26_post]
    rw [gt_iff_lt, IScalar.lt_equiv, hMAX] at hgt
    omega
  · -- guard scaled < MIN
    rename_i hgt hlt
    exfalso
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_two_pow_bmod i26.val (by rw [i26_post]; omega)
    have hrv : rounded.val = (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val := by
      rw [rounded_post, i27_post, hhalf, i26_post]
    rw [IScalar.lt_equiv, hMIN] at hlt
    omega
  · -- value
    rename_i hgt hlt
    refine ⟨IScalar.cast IScalarTy.I64 rounded, rfl, ?_⟩
    have hle : rounded.val ≤ 2^63 - 1 := by
      rw [gt_iff_lt, IScalar.lt_equiv, hMAX] at hgt; omega
    have hge : -2^63 ≤ rounded.val := by
      rw [IScalar.lt_equiv, hMIN] at hlt; omega
    have hhalf : (↑half : ℤ) = (2:ℤ)^(i26.val) := by
      rw [half_post, Int.shiftLeft_eq, one_mul]
      rw [show I128.size = 2^128 by norm_num [I128.size, I128.numBits]]
      exact fixed_exp_two_pow_bmod i26.val (by rw [i26_post]; omega)
    have hrv : rounded.val = (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val := by
      rw [rounded_post, i27_post, hhalf, i26_post]
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    rw [show (18446744073709551616 : ℕ) = 2^64 by norm_num]
    rw [fixed_exp_bmod_self_64 rounded.val hge hle, hrv]


theorem fixed_exp_scale_neg_exact (k : I64) (acc6 : I128) (hk : k.val < 0) (hlo : -2^62 ≤ k.val)
    (ha : |acc6.val| ≤ 2^40)
    (hrlo : -2^63 ≤ (acc6.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs)
    (hrhi : (acc6.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs ≤ 2^63 - 1) :
    fixed_exp.exp_scale k acc6 ⦃ z =>
      (64 ≤ k.val.natAbs → z = core.result.Result.Ok 0#i64) ∧
      (k.val.natAbs < 64 → ∃ v : I64, z = core.result.Result.Ok v ∧
        (v.val : ℤ) = (acc6.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs) ⦄ := by
  unfold fixed_exp.exp_scale
  have hnf : ¬ ((0#i64 : I64) ≤ k) := by
    intro h
    have h' : (0:ℤ) ≤ k.val := by simpa using (IScalar.le_equiv _ _).1 h
    omega
  rw [if_neg hnf]
  refine Aeneas.Std.WP.spec_bind (fixed_exp_scale_abs_raw k hk hlo) ?_
  intro n hn
  by_cases h64 : 64 ≤ n.val
  · have hkk : 64 ≤ k.val.natAbs := by rw [← hn]; exact h64
    rw [fixed_exp_scale_flush n acc6 h64]
    refine ⟨?_, ?_⟩
    · intro _; rfl
    · intro hlt; exfalso; omega
  · have hnlt : n.val < 64 := by omega
    have hnge : ¬ ((64#u64 : U64) ≤ n) := by
      intro h
      have h' : 64 ≤ n.val := by simpa using (UScalar.le_equiv _ _).1 h
      omega
    rw [if_neg hnge]
    have hlo' : -2^63 ≤ (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val := by rw [← hn] at hrlo; exact hrlo
    have hhi' : (acc6.val + (2:ℤ)^(n.val - 1)) >>> n.val ≤ 2^63 - 1 := by rw [← hn] at hrhi; exact hrhi
    refine Aeneas.Std.WP.spec_mono
      (fixed_exp_scale_neg_core_exact n acc6 hnlt ?hn1 ha hlo' hhi') ?_
    · rw [hn]; omega
    · intro z hz
      refine ⟨?_, ?_⟩
      · intro hc; exfalso; rw [← hn] at hc; omega
      · intro _
        rcases hz with ⟨v, hv, hval⟩
        exact ⟨v, hv, by rw [← hn]; exact hval⟩
/-! ### E1C3 step 1: on-window scale range budget

Supplies the range hypotheses of `fixed_exp_scale_nonneg_exact` / `fixed_exp_scale_neg_exact` from
the window: `k ∈ [-24,30]` gives `k < 64` (the `k ≥ 64` refusal never arises) and `n = |k| ≤ 24`;
`horner_acc_error` plus `exp (-1/2) ≥ 1/2` gives `acc.val > 0`; `exp (1/2) < 7/4` bounds
`acc.val·2^k`; and `|acc.val| ≤ 2^40` bounds the round-to-nearest shift. -/

/-- `exp (1/2) < 7/4` (from `Real.exp_bound` at `n = 5`), the only `exp`-value bound the scale
range budget needs. -/
theorem fixed_exp_exp_half_lt : Real.exp (1/2) < 7/4 := by
  have hb := Real.exp_bound (x := (1/2:ℝ)) (by norm_num) (n := 5) (by norm_num)
  have hS : (∑ m ∈ Finset.range 5, (1/2:ℝ)^m / (m.factorial : ℝ)) = 1.6484375 := by
    norm_num [Finset.sum_range_succ, Nat.factorial]
  have hR : |(1/2:ℝ)|^5 * ((5:ℕ).succ / ((5:ℕ).factorial * (5:ℕ))) = 1/3200 := by norm_num
  have hb' := abs_le.mp hb
  rw [hS, hR] at hb'
  linarith [hb'.2]

/-- On-window range budget supplying the exact `exp_scale` theorems' hypotheses: `k < 64` (`k ≤ 30`
from the shift bound, so the `k ≥ 64` refusal never arises), `0 < acc`, and the two interval facts. -/
theorem fixed_exp_scale_range_budget (x k r : I64) (acc : I128) (hx : window x)
    (hk : k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64)
    (hr : r.val = x.val - k.val * 2977044472)
    (hacc : acc.val = fixed_exp.horner_acc r.val)
    (haccb : |acc.val| ≤ 2^40) :
    k.val < 64 ∧ 0 < acc.val ∧
    (0 ≤ k.val → acc.val * 2 ^ k.val.toNat ≤ 2^63 - 1) ∧
    (k.val < 0 → -2^63 ≤ (acc.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs ∧
                  (acc.val + (2:ℤ)^(k.val.natAbs - 1)) >>> k.val.natAbs ≤ 2^63 - 1) := by
  have hsb := fixed_exp_shift_recip_shift_bound x.val k.val hx.1 hx.2 hk
  have hrhalf := fixed_exp_reduction_remainder_toReal_half x k r hx hk hr
  have hherr : |toReal128 acc - Real.exp (toReal r)| ≤ epsHorner := by
    rw [toReal128, hacc]
    simpa using fixed_exp.horner_acc_error r.val (toReal r) rfl hrhalf
  have heps : epsHorner ≤ 1/100 := by unfold epsHorner; norm_num
  have heps0 : (0:ℝ) < epsHorner := by unfold epsHorner; norm_num
  have hacc_eq : (acc.val : ℝ) = q32 * toReal128 acc := by
    rw [toReal128, ← mul_div_assoc]
    exact (mul_div_cancel_left₀ (acc.val : ℝ) (ne_of_gt q32_pos)).symm
  have hacc_pos : 0 < acc.val := by
    have h1 : (1/2:ℝ) ≤ Real.exp (toReal r) := by
      have hδ : (-1/2:ℝ) ≤ toReal r := by linarith [abs_le.mp hrhalf |>.1]
      calc (1/2:ℝ) = -1/2 + 1 := by norm_num
        _ ≤ Real.exp (-1/2) := Real.add_one_le_exp (-1/2)
        _ ≤ Real.exp (toReal r) := Real.exp_le_exp.mpr hδ
    have h2 : (1/2:ℝ) - epsHorner ≤ toReal128 acc := by
      have := abs_le.mp hherr; linarith [this.1, h1]
    have h3 : (0:ℝ) < toReal128 acc := by linarith [heps, heps0]
    have : (0:ℝ) < (acc.val : ℝ) := by rw [hacc_eq]; exact mul_pos q32_pos h3
    exact_mod_cast this
  refine ⟨by omega, hacc_pos, ?_, ?_⟩
  · intro hk0
    have h1 : toReal128 acc ≤ 7/4 + epsHorner := by
      have h2 : Real.exp (toReal r) ≤ Real.exp (1/2) :=
        Real.exp_le_exp.mpr (by linarith [abs_le.mp hrhalf |>.2])
      have h3 : Real.exp (1/2) < 7/4 := fixed_exp_exp_half_lt
      have := abs_le.mp hherr
      linarith [this.2]
    have h4 : (acc.val : ℝ) ≤ (2:ℝ)^32 * (7/4 + epsHorner) := by
      rw [hacc_eq, show q32 = (2:ℝ)^32 from rfl]
      exact mul_le_mul_of_nonneg_left h1 (by positivity)
    have h6 : (2:ℝ)^k.val.toNat ≤ 2^30 :=
      pow_le_pow_right₀ (by norm_num) (by omega)
    have h7 : ((acc.val * 2 ^ k.val.toNat : ℤ) : ℝ) ≤ (((2:ℤ)^63 - 1 : ℤ) : ℝ) := by
      push_cast
      have hle : (acc.val : ℝ) * (2:ℝ)^k.val.toNat ≤ (2:ℝ)^32 * (7/4 + epsHorner) * 2^30 :=
        mul_le_mul h4 h6 (by positivity) (by positivity)
      have hnum : (2:ℝ)^32 * (7/4 + epsHorner) * 2^30 ≤ (2:ℝ)^63 - 1 := by
        norm_num [epsHorner]
      linarith [hle, hnum]
    exact_mod_cast h7
  · intro hkneg
    have hn1 : 1 ≤ k.val.natAbs := by omega
    have hn24 : k.val.natAbs ≤ 24 := by omega
    have hpow : ((2 ^ k.val.natAbs : ℕ) : ℤ) = (2:ℤ)^k.val.natAbs := by push_cast; ring
    have hpowpos : (0:ℤ) < (2:ℤ)^k.val.natAbs := by positivity
    have h1le : (1:ℤ) ≤ (2:ℤ)^k.val.natAbs := one_le_pow₀ (by norm_num)
    have h2n1nn : (0:ℤ) ≤ (2:ℤ)^(k.val.natAbs - 1) := by positivity
    have hmono : (2:ℤ)^(k.val.natAbs - 1) ≤ 2^24 := by
      calc (2:ℤ)^(k.val.natAbs - 1) ≤ (2:ℤ)^23 :=
            pow_le_pow_right₀ (by norm_num) (by omega)
        _ ≤ 2^24 := by norm_num
    have ha' := abs_le.mp haccb
    constructor
    · rw [Int.shiftRight_eq_div_pow, hpow]
      rw [Int.le_ediv_iff_mul_le hpowpos]
      nlinarith [ha'.1, h2n1nn, h1le]
    · rw [Int.shiftRight_eq_div_pow, hpow]
      rw [Int.ediv_le_iff_le_mul hpowpos]
      nlinarith [ha'.2, hmono, h1le]
/-! ### E1C4: `fixed_exp_refines` composition

The strengthened reduction spec (Ok only), the `exp`-Lipschitz and scaling budgets, the
round-to-nearest budget, and the whole-body relative bound `epsExp·exp (toReal x) + 4/q32`. -/

/-- Strengthened `exp_reduce` spec: on the window the reduction always succeeds (Ok only). -/
theorem fixed_exp_exp_reduce_spec_ok (x : I64) (hx : window x) :
    fixed_exp.exp_reduce x ⦃ z =>
      ∃ k r : I64, z = core.result.Result.Ok (k, r) ∧
        k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 2977044472 ⦄ := by
  unfold fixed_exp.exp_reduce
  rw [fixed_exp_LOWER_eq, fixed_exp_UPPER_eq]
  simp only [bind_tc_ok]
  rw [if_neg (fixed_exp_lower_test_false x hx), if_neg (fixed_exp_upper_test_false x hx)]
  apply Aeneas.Std.WP.spec_mono (word_fixed_exp_reduction_value_pair x hx)
  intro z hz
  exact hz

/-- `exp` is Lipschitz-`(1+2δ)` at the reduction displacement: for `|u - x| ≤ deltaRed`,
`exp u ≤ exp x·(1 + 2·deltaRed)` and `|exp u - exp x| ≤ exp x·(2·deltaRed)`. -/
theorem fixed_exp_exp_u_bounds (u x : ℝ) (hδ : |u - x| ≤ deltaRed) :
    Real.exp u ≤ Real.exp x * (1 + 2 * deltaRed) ∧
      |Real.exp u - Real.exp x| ≤ Real.exp x * (2 * deltaRed) := by
  have hδ0 : 0 ≤ deltaRed := by unfold deltaRed; norm_num
  have hδ1 : deltaRed ≤ 1 := by unfold deltaRed; norm_num
  have hδabs := abs_le.mp hδ
  have hu1 : |u - x| ≤ 1 := by rw [abs_le]; exact ⟨by linarith [hδabs.1], by linarith [hδabs.2]⟩
  have hexpδ : Real.exp (u - x) ≤ 1 + 2 * deltaRed := by
    have h1 : Real.exp (u - x) ≤ Real.exp deltaRed := Real.exp_le_exp.mpr hδabs.2
    have h2 : Real.exp deltaRed ≤ 1 + 2 * deltaRed := by
      have hb := Real.exp_bound (x := deltaRed) (by rw [abs_of_nonneg hδ0]; exact hδ1) (n := 1)
        (by norm_num)
      have hb' := abs_le.mp hb
      rw [abs_of_nonneg hδ0] at hb'
      norm_num at hb'
      linarith [hb'.2]
    linarith
  constructor
  · rw [show u = x + (u - x) by ring, Real.exp_add]
    exact mul_le_mul_of_nonneg_left hexpδ (le_of_lt (Real.exp_pos x))
  · have hb := Real.exp_bound (x := u - x) hu1 (n := 1) (by norm_num)
    have hb' := abs_le.mp hb
    norm_num at hb'
    rw [show Real.exp u - Real.exp x = Real.exp x * (Real.exp (u - x) - 1) by
          rw [show u = x + (u - x) by ring, Real.exp_add]; ring,
        abs_mul, abs_of_pos (Real.exp_pos x)]
    apply mul_le_mul_of_nonneg_left _ (le_of_lt (Real.exp_pos x))
    have h1 : Real.exp (u - x) ≤ 1 + 2 * deltaRed := hexpδ
    have h2 : 1 - 2 * deltaRed ≤ Real.exp (u - x) := by linarith [hb'.1, hδabs.1, hδabs.2]
    rw [abs_le]
    exact ⟨by linarith [h2], by linarith [h1]⟩

/-- Relative scaling budget, `k = n ≥ 0`: `2^n·epsHorner ≤ exp(r + n log 2)·(2·epsHorner)` for
`|r| ≤ 1/2`. -/
theorem fixed_exp_scale_abs_rel_pos (n : ℕ) (r : ℝ) (hr : |r| ≤ 1/2) :
    ((2:ℝ)^n) * epsHorner ≤ Real.exp (r + (n:ℝ) * Real.log 2) * (2 * epsHorner) := by
  have hrge : (1/2:ℝ) ≤ Real.exp r := by
    have h : (-1/2:ℝ) ≤ r := by linarith [abs_le.mp hr |>.1]
    calc (1/2:ℝ) = -1/2 + 1 := by norm_num
      _ ≤ Real.exp (-1/2) := Real.add_one_le_exp (-1/2)
      _ ≤ Real.exp r := Real.exp_le_exp.mpr h
  have heps0 : (0:ℝ) < epsHorner := by unfold epsHorner; norm_num
  have h1 : epsHorner ≤ Real.exp r * (2 * epsHorner) := by nlinarith [hrge, heps0]
  have h2 : ((2:ℝ)^n) * epsHorner ≤ ((2:ℝ)^n) * (Real.exp r * (2 * epsHorner)) :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  have h3 : ((2:ℝ)^n) * Real.exp r = Real.exp (r + (n:ℝ) * Real.log 2) := by
    rw [show ((2:ℝ)^n) = (Real.exp (Real.log 2))^n by
          rw [Real.exp_log (by norm_num : (0:ℝ) < 2)]]
    rw [← Real.exp_nat_mul, ← Real.exp_add]
    ring_nf
  have h4 : ((2:ℝ)^n) * (Real.exp r * (2 * epsHorner)) =
      Real.exp (r + (n:ℝ) * Real.log 2) * (2 * epsHorner) := by rw [← mul_assoc, h3]
  rw [h4] at h2
  exact h2

/-- Relative scaling budget, `k = -n < 0`: `2^{-n}·epsHorner ≤ exp(r - n log 2)·(2·epsHorner)`. -/
theorem fixed_exp_scale_abs_rel_neg (n : ℕ) (r : ℝ) (hr : |r| ≤ 1/2) :
    ((2:ℝ)^(-(n:ℤ))) * epsHorner ≤ Real.exp (r - (n:ℝ) * Real.log 2) * (2 * epsHorner) := by
  have hrge : (1/2:ℝ) ≤ Real.exp r := by
    have h : (-1/2:ℝ) ≤ r := by linarith [abs_le.mp hr |>.1]
    calc (1/2:ℝ) = -1/2 + 1 := by norm_num
      _ ≤ Real.exp (-1/2) := Real.add_one_le_exp (-1/2)
      _ ≤ Real.exp r := Real.exp_le_exp.mpr h
  have heps0 : (0:ℝ) < epsHorner := by unfold epsHorner; norm_num
  have h1 : epsHorner ≤ Real.exp r * (2 * epsHorner) := by nlinarith [hrge, heps0]
  have h2 : ((2:ℝ)^(-(n:ℤ))) * epsHorner ≤ ((2:ℝ)^(-(n:ℤ))) * (Real.exp r * (2 * epsHorner)) :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  have h3 : ((2:ℝ)^(-(n:ℤ))) * Real.exp r = Real.exp (r - (n:ℝ) * Real.log 2) := by
    rw [zpow_neg, zpow_natCast, inv_mul_eq_div]
    rw [show ((2:ℝ)^n) = (Real.exp (Real.log 2))^n by
          rw [Real.exp_log (by norm_num : (0:ℝ) < 2)]]
    rw [← Real.exp_nat_mul, ← Real.exp_sub]
  have h4 : ((2:ℝ)^(-(n:ℤ))) * (Real.exp r * (2 * epsHorner)) =
      Real.exp (r - (n:ℝ) * Real.log 2) * (2 * epsHorner) := by rw [← mul_assoc, h3]
  rw [h4] at h2
  exact h2

/-- Euclidean-division floor bound: `|(y / m) - y/m| ≤ 1` for `0 < m`. -/
theorem fixed_exp_ediv_self_abs (y m : ℤ) (hm : 0 < m) :
    |(((y / m : ℤ)) : ℝ) - (y:ℝ)/(m:ℝ)| ≤ 1 := by
  have hem : m * (y / m) + y % m = y := Int.mul_ediv_add_emod y m
  have h0 : 0 ≤ y % m := Int.emod_nonneg _ (ne_of_gt hm)
  have hlt : y % m < m := Int.emod_lt_of_pos _ hm
  have hcast : ((y / m : ℤ) : ℝ) = ((y:ℝ) - ((y % m : ℤ) : ℝ))/(m:ℝ) := by
    have := congrArg (fun t : ℤ => (t:ℝ)) hem
    push_cast at this
    field_simp
    linarith [this]
  have hltR : ((y % m : ℤ) : ℝ) < (m:ℝ) := by exact_mod_cast hlt
  have hmR : (0:ℝ) ≤ ((y % m : ℤ) : ℝ) := by exact_mod_cast h0
  have hmpos : (0:ℝ) < (m:ℝ) := by exact_mod_cast hm
  rw [hcast]
  rw [show ((y:ℝ) - ((y % m : ℤ) : ℝ))/(m:ℝ) - (y:ℝ)/(m:ℝ)
        = -(((y % m : ℤ) : ℝ)/(m:ℝ)) by ring, abs_neg, abs_div, abs_of_pos hmpos,
      div_le_one hmpos, abs_of_nonneg hmR]
  exact le_of_lt hltR

/-- Round-to-nearest right shift budget: `|(y >>> n) - a/2^n| ≤ 2` for `n ≥ 1`, `y = a + 2^{n-1}`. -/
theorem fixed_exp_round_neg_abs (n : ℕ) (hn1 : 1 ≤ n) (a : ℤ) :
    |(((a + (2:ℤ)^(n-1)) >>> n : ℤ) : ℝ) - (a:ℝ)/(2:ℝ)^n| ≤ 2 := by
  rw [Int.shiftRight_eq_div_pow]
  have hpow : (((2 ^ n : ℕ)) : ℤ) = (2:ℤ)^n := by push_cast; ring
  rw [hpow]
  have hb : (0:ℤ) < (2:ℤ)^n := by positivity
  have h1 : |((( (a + (2:ℤ)^(n-1)) / (2:ℤ)^n : ℤ)) : ℝ) - ((a:ℝ) + (2:ℝ)^(n-1))/(2:ℝ)^n| ≤ 1 := by
    have := fixed_exp_ediv_self_abs (a + (2:ℤ)^(n-1)) ((2:ℤ)^n) hb
    simpa [Int.cast_pow] using this
  have h2 : ((a:ℝ) + (2:ℝ)^(n-1))/(2:ℝ)^n - (a:ℝ)/(2:ℝ)^n = (1/2:ℝ) := by
    rw [show (2:ℝ)^n = 2 * (2:ℝ)^(n-1) by rw [← pow_succ']; congr 1; omega]
    field_simp
    ring
  calc |((( (a + (2:ℤ)^(n-1)) / (2:ℤ)^n : ℤ)) : ℝ) - (a:ℝ)/(2:ℝ)^n|
      = |(((( (a + (2:ℤ)^(n-1)) / (2:ℤ)^n : ℤ)) : ℝ) - ((a:ℝ) + (2:ℝ)^(n-1))/(2:ℝ)^n)
          + (1/2:ℝ)| := by
        rw [← h2]; ring
    _ ≤ |((( (a + (2:ℤ)^(n-1)) / (2:ℤ)^n : ℤ)) : ℝ) - ((a:ℝ) + (2:ℝ)^(n-1))/(2:ℝ)^n| + |(1/2:ℝ)| :=
        abs_add_le _ _
    _ ≤ 2 := by rw [abs_of_pos (by norm_num : (0:ℝ) < 1/2)]; linarith [h1]


theorem word_fixed_exp_body_refines (x : I64) (hx : window x) :
    word_fixed_exp x ⦃ z =>
      ∃ v : I64, z = core.result.Result.Ok v ∧
        |toReal v - Real.exp (toReal x)| ≤ epsExp * Real.exp (toReal x) + 4 / q32 ⦄ := by
  rw [word_fixed_exp]
  unfold fixed_exp.fixed_exp
  apply Aeneas.Std.WP.spec_bind (fixed_exp_exp_reduce_spec_ok x hx)
  intro kr hkr
  obtain ⟨k, r, rfl, hk, hr⟩ := hkr
  have hsr := fixed_exp_shift_recip_bound_window x.val k.val hx.1 hx.2 hk
  have hr32 : |r.val| ≤ 2^32 := by
    rw [hr]; exact fixed_exp_reduction_remainder_abs_le x.val k.val hk hsr
  apply Aeneas.Std.WP.spec_bind (word_fixed_exp_horner_acc r hr32)
  intro acc hacc
  obtain ⟨haccv, haccb⟩ := hacc
  have hsb := fixed_exp_shift_recip_shift_bound x.val k.val hx.1 hx.2 hk
  have hrhalf := fixed_exp_reduction_remainder_toReal_half x k r hx hk hr
  have hherr : |toReal128 acc - Real.exp (toReal r)| ≤ epsHorner := by
    rw [toReal128, haccv]
    simpa using fixed_exp.horner_acc_error r.val (toReal r) rfl hrhalf
  have hdisp : |toReal r + (k.val:ℝ) * Real.log 2 - toReal x| ≤ deltaRed := by
    have h := fixed_exp_reduction_displacement x k r hx hk hr
    rw [show toReal r + (k.val:ℝ) * Real.log 2 - toReal x
          = toReal r - (toReal x - (k.val:ℝ) * Real.log 2) by ring]
    exact h
  have hnum : (1 + 2 * deltaRed) * (2 * epsHorner) + 2 * deltaRed ≤ epsExp := by
    unfold deltaRed epsHorner epsExp; norm_num
  have hq32pos : (0:ℝ) < q32 := q32_pos
  have heps0 : (0:ℝ) < epsHorner := by unfold epsHorner; norm_num
  by_cases hk0 : 0 ≤ k.val
  · obtain ⟨h64, hpos, hub, _⟩ := fixed_exp_scale_range_budget x k r acc hx hk hr haccv haccb
    refine Aeneas.Std.WP.spec_mono
      (fixed_exp_scale_nonneg_exact k acc hk0 h64 (le_of_lt hpos) (hub hk0)) ?_
    intro z hz
    obtain ⟨v, rfl, hval⟩ := hz
    refine ⟨v, rfl, ?_⟩
    have hn : ((k.val.toNat:ℕ):ℝ) = (k.val:ℝ) := by
      exact_mod_cast (Int.toNat_of_nonneg hk0 : (k.val.toNat : ℤ) = k.val)
    set n : ℕ := k.val.toNat with hndef
    have hnR : (n:ℝ) = (k.val:ℝ) := by rw [hndef]; exact hn
    have hvalR : toReal v = ((2:ℝ)^n) * toReal128 acc := by
      rw [toReal, toReal128, hval]; push_cast; ring
    have hE : ((2:ℝ)^n) * Real.exp (toReal r) = Real.exp (toReal r + (n:ℝ) * Real.log 2) := by
      rw [show ((2:ℝ)^n) = (Real.exp (Real.log 2))^n by
            rw [Real.exp_log (by norm_num : (0:ℝ) < 2)]]
      rw [← Real.exp_nat_mul, ← Real.exp_add]
      ring_nf
    have hδn : |toReal r + (n:ℝ) * Real.log 2 - toReal x| ≤ deltaRed := by
      rw [hnR]; exact hdisp
    have huB := fixed_exp_exp_u_bounds (toReal r + (n:ℝ) * Real.log 2) (toReal x) hδn
    have hA : |toReal v - ((2:ℝ)^n) * Real.exp (toReal r)| ≤ ((2:ℝ)^n) * epsHorner := by
      rw [hvalR, ← mul_sub, abs_mul, abs_of_nonneg (by positivity)]
      exact mul_le_mul_of_nonneg_left hherr (by positivity)
    have hB : |((2:ℝ)^n) * Real.exp (toReal r) - Real.exp (toReal x)| ≤
        Real.exp (toReal x) * (2 * deltaRed) := by
      rw [hE]; exact huB.2
    have hrel : ((2:ℝ)^n) * epsHorner ≤ Real.exp (toReal x) * (1 + 2 * deltaRed) * (2 * epsHorner) := by
      have h1 := fixed_exp_scale_abs_rel_pos n (toReal r) hrhalf
      calc ((2:ℝ)^n) * epsHorner
          ≤ Real.exp (toReal r + (n:ℝ) * Real.log 2) * (2 * epsHorner) := h1
        _ ≤ (Real.exp (toReal x) * (1 + 2 * deltaRed)) * (2 * epsHorner) :=
            mul_le_mul_of_nonneg_right huB.1 (by linarith [heps0])
        _ = Real.exp (toReal x) * (1 + 2 * deltaRed) * (2 * epsHorner) := by ring
    calc |toReal v - Real.exp (toReal x)|
        ≤ |toReal v - ((2:ℝ)^n) * Real.exp (toReal r)|
          + |((2:ℝ)^n) * Real.exp (toReal r) - Real.exp (toReal x)| := by
          rw [show toReal v - Real.exp (toReal x)
                = (toReal v - ((2:ℝ)^n) * Real.exp (toReal r))
                  + (((2:ℝ)^n) * Real.exp (toReal r) - Real.exp (toReal x)) by ring]
          exact abs_add_le _ _
      _ ≤ ((2:ℝ)^n) * epsHorner + Real.exp (toReal x) * (2 * deltaRed) := add_le_add hA hB
      _ ≤ Real.exp (toReal x) * (1 + 2 * deltaRed) * (2 * epsHorner)
            + Real.exp (toReal x) * (2 * deltaRed) := add_le_add hrel le_rfl
      _ = Real.exp (toReal x) * ((1 + 2 * deltaRed) * (2 * epsHorner) + 2 * deltaRed) := by ring
      _ ≤ Real.exp (toReal x) * epsExp :=
          mul_le_mul_of_nonneg_left hnum (le_of_lt (Real.exp_pos (toReal x)))
      _ ≤ epsExp * Real.exp (toReal x) + 4 / q32 := by
          have : (0:ℝ) ≤ 4/q32 := by exact div_nonneg (by norm_num) (le_of_lt q32_pos)
          rw [mul_comm]; linarith
  · have hkneg : k.val < 0 := lt_of_not_ge hk0
    obtain ⟨_, _, _, hrange⟩ := fixed_exp_scale_range_budget x k r acc hx hk hr haccv haccb
    obtain ⟨hlo, hhi⟩ := hrange hkneg
    have hlo62 : -2^62 ≤ k.val := by omega
    refine Aeneas.Std.WP.spec_mono
      (fixed_exp_scale_neg_exact k acc hkneg hlo62 haccb hlo hhi) ?_
    intro z hz
    obtain ⟨_, hz2⟩ := hz
    have hn1 : 1 ≤ k.val.natAbs := by omega
    have hn24 : k.val.natAbs ≤ 24 := by omega
    have hn64 : k.val.natAbs < 64 := by omega
    obtain ⟨v, rfl, hval⟩ := hz2 hn64
    refine ⟨v, rfl, ?_⟩
    have h2 : k.val.natAbs = (-k.val).natAbs := by rw [Int.natAbs_neg]
    have hnat : ((k.val.natAbs:ℤ)) = -k.val := by
      rw [h2]
      exact Int.natAbs_of_nonneg (by linarith [hkneg])
    set n : ℕ := k.val.natAbs with hndef
    have h2knn : (0:ℝ) ≤ (2:ℝ)^(-(n:ℤ)) := le_of_lt (zpow_pos (by norm_num : (0:ℝ) < 2) _)
    have hnR : (n:ℝ) = -(k.val:ℝ) := by exact_mod_cast hnat
    have hkn : (k.val:ℝ) = -(n:ℝ) := by
      have := hnR; linarith
    have hvalR : toReal v = (((acc.val + (2:ℤ)^(n-1)) >>> n : ℤ) : ℝ) / q32 := by
      rw [toReal, hval]
    have hround : |((((acc.val + (2:ℤ)^(n-1)) >>> n : ℤ)) : ℝ) - (acc.val:ℝ)/(2:ℝ)^n| ≤ 2 := by
      have := fixed_exp_round_neg_abs n (by omega) acc.val
      exact this
    have hroundR : |toReal v - ((2:ℝ)^(-(n:ℤ))) * toReal128 acc| ≤ 2 / q32 := by
      rw [hvalR, toReal128]
      rw [show ((2:ℝ)^(-(n:ℤ))) * ((acc.val:ℝ)/q32) = ((acc.val:ℝ)/(2:ℝ)^n)/q32 by
            rw [zpow_neg, zpow_natCast, inv_mul_eq_div]; ring]
      rw [← sub_div, abs_div, abs_of_pos hq32pos]
      rw [div_le_div_iff_of_pos_right hq32pos]
      exact hround
    have hE : ((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r) = Real.exp (toReal r - (n:ℝ) * Real.log 2) := by
      rw [zpow_neg, zpow_natCast, inv_mul_eq_div]
      rw [show ((2:ℝ)^n) = (Real.exp (Real.log 2))^n by
            rw [Real.exp_log (by norm_num : (0:ℝ) < 2)]]
      rw [← Real.exp_nat_mul, ← Real.exp_sub]
    have hδn : |toReal r - (n:ℝ) * Real.log 2 - toReal x| ≤ deltaRed := by
      rw [show toReal r - (n:ℝ) * Real.log 2 = toReal r + (k.val:ℝ) * Real.log 2 by rw [hkn]; ring]
      exact hdisp
    have huB := fixed_exp_exp_u_bounds (toReal r - (n:ℝ) * Real.log 2) (toReal x) hδn
    have hA : |toReal v - ((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r)| ≤
        2 / q32 + ((2:ℝ)^(-(n:ℤ))) * epsHorner := by
      have h2 : ((2:ℝ)^(-(n:ℤ))) * |toReal128 acc - Real.exp (toReal r)| ≤
          ((2:ℝ)^(-(n:ℤ))) * epsHorner :=
        mul_le_mul_of_nonneg_left hherr h2knn
      rw [show toReal v - ((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r)
            = (toReal v - ((2:ℝ)^(-(n:ℤ))) * toReal128 acc)
              + (((2:ℝ)^(-(n:ℤ))) * toReal128 acc - ((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r)) by ring]
      refine le_trans (abs_add_le _ _) (add_le_add hroundR ?_)
      rw [← mul_sub, abs_mul, abs_of_nonneg h2knn]
      exact h2
    have hrel : ((2:ℝ)^(-(n:ℤ))) * epsHorner ≤
        Real.exp (toReal x) * (1 + 2 * deltaRed) * (2 * epsHorner) := by
      have h1 := fixed_exp_scale_abs_rel_neg n (toReal r) hrhalf
      calc ((2:ℝ)^(-(n:ℤ))) * epsHorner
          ≤ Real.exp (toReal r - (n:ℝ) * Real.log 2) * (2 * epsHorner) := h1
        _ ≤ (Real.exp (toReal x) * (1 + 2 * deltaRed)) * (2 * epsHorner) :=
            mul_le_mul_of_nonneg_right huB.1 (by linarith [heps0])
        _ = Real.exp (toReal x) * (1 + 2 * deltaRed) * (2 * epsHorner) := by ring
    calc |toReal v - Real.exp (toReal x)|
        ≤ |toReal v - ((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r)|
          + |((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r) - Real.exp (toReal x)| := by
          rw [show toReal v - Real.exp (toReal x)
                = (toReal v - ((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r))
                  + (((2:ℝ)^(-(n:ℤ))) * Real.exp (toReal r) - Real.exp (toReal x)) by ring]
          exact abs_add_le _ _
      _ ≤ (2 / q32 + ((2:ℝ)^(-(n:ℤ))) * epsHorner)
            + Real.exp (toReal x) * (2 * deltaRed) := by
          refine add_le_add hA ?_
          rw [hE]; exact huB.2
      _ ≤ (2 / q32 + Real.exp (toReal x) * (1 + 2 * deltaRed) * (2 * epsHorner))
            + Real.exp (toReal x) * (2 * deltaRed) := by linarith [hrel]
      _ = 2 / q32 + Real.exp (toReal x) * ((1 + 2 * deltaRed) * (2 * epsHorner) + 2 * deltaRed) := by
          ring
      _ ≤ 2 / q32 + Real.exp (toReal x) * epsExp := by
          have hm := mul_le_mul_of_nonneg_left hnum (le_of_lt (Real.exp_pos (toReal x)))
          linarith [hm]
      _ ≤ epsExp * Real.exp (toReal x) + 4 / q32 := by
          have : (2:ℝ) / q32 ≤ 4 / q32 := by
            apply div_le_div_of_nonneg_right _ (le_of_lt hq32pos); norm_num
          rw [mul_comm (Real.exp (toReal x)) epsExp]; linarith

theorem fixed_exp_refines_proved : fixed_exp_refines := by
  intro x hx
  obtain ⟨w, hw, hmem⟩ := Aeneas.Std.WP.spec_imp_exists (word_fixed_exp_body_refines x hx)
  obtain ⟨v, hv, hb⟩ := hmem
  refine ⟨v, ?_, hb⟩
  rw [word_fixed_exp, hv] at hw
  exact hw

end

end apeiron_word
