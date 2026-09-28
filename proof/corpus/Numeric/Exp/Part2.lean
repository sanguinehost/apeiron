import Numeric.Exp.Part1

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- The mathematical result of the extracted `fixed_exp.round_div_2p32`: add `2^31` and
shift right by 32 on the nonnegative branch; on the negative branch negate, add `2^31`,
shift right by 32 and negate again. Stated with `>>>` so it matches the extracted
equations. -/
def fixed_exp.round_q32 (v : ℤ) : ℤ :=
  if 0 ≤ v then (v + 2147483648) >>> 32 else - ((-v + 2147483648) >>> 32)

theorem fixed_exp_round_div_2p32_val (value : I128)
    (hpos : 0 ≤ value.val → value.val + 2147483648 ≤ I128.max)
    (hneg : value.val < 0 → -value.val + 2147483648 ≤ I128.max) :
    fixed_exp.round_div_2p32 value ⦃ z => z.val = fixed_exp.round_q32 value.val ⦄ := by
  apply Aeneas.Std.WP.spec_mono (fixed_exp_round_div_2p32_spec value hpos hneg)
  intro z hz
  unfold fixed_exp.round_q32
  by_cases hv : 0 ≤ value.val
  · rw [if_pos hv]; exact hz.1 hv
  · rw [if_neg hv]; exact hz.2 (lt_of_not_ge hv)

/-- The degree-7 Horner numerator consumed by the scaling branch: the nested integer expression in
`r` with `fixed_exp.round_q32` at each step and the Q32.32 Taylor coefficients. This is the target
expression of the chain; only the first step (`fixed_exp.round_q32 (852176 * r)`) is characterized
below. -/
def fixed_exp.horner_acc (r : ℤ) : ℤ :=
  let a0 := fixed_exp.round_q32 (852176 * r) + 5965232
  let a1 := fixed_exp.round_q32 (a0 * r) + 35791394
  let a2 := fixed_exp.round_q32 (a1 * r) + 178956971
  let a3 := fixed_exp.round_q32 (a2 * r) + 715827883
  let a4 := fixed_exp.round_q32 (a3 * r) + 2147483648
  let a5 := fixed_exp.round_q32 (a4 * r) + 4294967296
  fixed_exp.round_q32 (a5 * r) + 4294967296

/-- First Horner step, driving the extracted rounding with `fixed_exp_round_div_2p32_spec`:
`step*` handles the total `i128` arithmetic and stops at `fixed_exp.round_div_2p32`, so
`fixed_exp_round_div_2p32_val` is applied explicitly. This is the method for the rest of
the chain; only the first step is landed here. -/
theorem word_fixed_exp_horner_first (r : I64) (hr : |r.val| ≤ 2 ^ 32) :
    (do let r1 ← lift (core.convert.num.FromI128I64.from r)
        let i12 ← 852176#i128 * r1
        let i13 ← fixed_exp.round_div_2p32 i12
        ok i13) ⦃ z => z.val = fixed_exp.round_q32 (852176 * r.val) ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i12 ?_ ?_) ?_
  · intro h0
    have hr' : (-(2:ℤ)^32) ≤ r.val ∧ r.val ≤ 2^32 := (abs_le.mp hr)
    rw [i12_post, core.convert.num.FromI128I64.from_val_eq] at h0 ⊢
    norm_num [I128.max, I128.numBits] at h0 ⊢
    nlinarith [hr'.1, hr'.2, h0]
  · intro hlt
    have hr' : (-(2:ℤ)^32) ≤ r.val ∧ r.val ≤ 2^32 := (abs_le.mp hr)
    rw [i12_post, core.convert.num.FromI128I64.from_val_eq] at hlt ⊢
    norm_num [I128.max, I128.numBits] at hlt ⊢
    nlinarith [hr'.1, hr'.2, hlt]
  · intro z hz
    simp only [Aeneas.Std.WP.spec_ok]
    rw [hz, i12_post, core.convert.num.FromI128I64.from_val_eq]
/-- Coarse bound on the extracted rounding: `|round_q32 v| ≤ |v| / 2^32 + 1`. -/
theorem fixed_exp.round_q32_abs_le (v : ℤ) :
    |fixed_exp.round_q32 v| ≤ |v| / 2^32 + 1 := by
  rw [fixed_exp.round_q32]
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

/-- If `|x| ≤ 2^100` then both `x + 2^31` and `-x + 2^31` stay in the `i128` range, so the
`fixed_exp.round_div_2p32` overflow premises hold for any Horner intermediate in that band. -/
theorem fixed_exp.round_safe (x : ℤ) (hx : |x| ≤ 2^100) :
    x + 2147483648 ≤ I128.max ∧ -x + 2147483648 ≤ I128.max := by
  have hx' := abs_le.mp hx
  constructor <;> norm_num [I128.max, I128.numBits] <;> omega
/-- Parameterised form of `fixed_exp.round_q32_abs_le`: if `|v| ≤ C · 2^32` then the rounded value
is at most `C + 1` in absolute value. This is the bound the Horner chain consumes stage by stage. -/
theorem fixed_exp.round_q32_abs_le' (v : ℤ) (C : ℤ) (hv : |v| ≤ C * 2^32) :
    |fixed_exp.round_q32 v| ≤ C + 1 := by
  have h1 := fixed_exp.round_q32_abs_le v
  have h2 : |v| / 2^32 ≤ C := by
    rw [Int.ediv_le_iff_le_mul (by norm_num : (0:ℤ) < 2^32)]
    nlinarith [hv]
  omega

/-- One Horner stage: multiply the accumulator by the Q32.32 remainder, round to Q32.32 with
`fixed_exp.round_div_2p32`, and add the next Taylor coefficient. Given `B` with `|acc.val| ≤ B`,
`B ≤ 2^68`, `|r.val| ≤ 2^32` and `|c.val| ≤ 2^32`, the result is `round_q32 (acc.val * r.val) + c.val`
and no `i128` refusal fires. -/
theorem fixed_exp.horner_step (acc r c : I128) (B : ℤ)
    (hA : |acc.val| ≤ B) (hB : B ≤ 2^68) (hr : |r.val| ≤ 2^32) (hc : |c.val| ≤ 2^32) :
    (do let i ← acc * r
        let i1 ← fixed_exp.round_div_2p32 i
        let a ← i1 + c
        ok a) ⦃ z => z.val = fixed_exp.round_q32 (acc.val * r.val) + c.val ⦄ := by
  have hm : |acc.val * r.val| ≤ 2^100 := by
    rw [abs_mul]; nlinarith [hA, hB, hr, abs_nonneg acc.val, abs_nonneg r.val]
  refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc) (y := r) ?_ ?_) ?_
  · have := abs_le.mp hm; norm_num [I128.min, I128.numBits]; omega
  · have := abs_le.mp hm; norm_num [I128.max, I128.numBits]; omega
  · intro i hi
    have hi_absB : |i.val| ≤ B * 2^32 := by
      rw [hi, abs_mul]
      nlinarith [hA, hr, abs_nonneg acc.val, abs_nonneg r.val]
    have hi_abs : |i.val| ≤ 2^100 := by nlinarith [hi_absB, hB]
    refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i ?_ ?_) ?_
    · intro _; exact (fixed_exp.round_safe i.val hi_abs).1
    · intro _; exact (fixed_exp.round_safe i.val hi_abs).2
    · intro i1 hi1
      have hi1_abs : |i1.val| ≤ B + 1 := by
        rw [hi1]
        exact fixed_exp.round_q32_abs_le' i.val B hi_absB
      refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i1) (y := c) ?_ ?_) ?_
      · norm_num [I128.min, I128.numBits]
        have := abs_le.mp hi1_abs; have := abs_le.mp hc; omega
      · norm_num [I128.max, I128.numBits]
        have := abs_le.mp hi1_abs; have := abs_le.mp hc; omega
      · intro a ha
        rw [Aeneas.Std.WP.spec_ok]
        rw [ha, hi1, hi]

/-- Two Horner stages of the extracted chain, proved by the explicit per-operation route
(`I128.mul_spec` / `I128.add_spec` / `fixed_exp_round_div_2p32_val`) that avoids the `#i128`
transparency failure: `acc1.val = round_q32 ((round_q32 (852176 · r.val) + 5965232) · r.val) + 35791394`
on `|r.val| ≤ 2^32`, with no `i128` refusal. The six further stages to `acc6` need the same route but
exceed the default heartbeat cap in `spec_bind` unification. -/
theorem word_fixed_exp_horner_acc1 (r : I64) (hr : |r.val| ≤ 2 ^ 32) :
    (do let r1 ← lift (core.convert.num.FromI128I64.from r)
        let i ← 852176#i128 * r1
        let i1 ← fixed_exp.round_div_2p32 i
        let acc ← i1 + 5965232#i128
        let i2 ← acc * r1
        let i3 ← fixed_exp.round_div_2p32 i2
        let acc1 ← i3 + 35791394#i128
        ok acc1) ⦃ z => z.val = fixed_exp.round_q32 ((fixed_exp.round_q32 (852176 * r.val) + 5965232) * r.val) + 35791394 ⦄ := by
  have hr' := abs_le.mp hr
  simp only [lift, bind_tc_ok]
  step*
  have hi12_abs : |i.val| ≤ 2^52 := by
    rw [i_post, core.convert.num.FromI128I64.from_val_eq, abs_mul,
      abs_of_nonneg (by norm_num : (0:ℤ) ≤ 852176)]
    nlinarith [hr'.1, hr'.2, abs_nonneg r.val]
  have hrA : |(core.convert.num.FromI128I64.from r).val| ≤ 2^32 := by
    rw [core.convert.num.FromI128I64.from_val_eq]; exact hr
  have hi12_abs100 : |i.val| ≤ 2^100 := by nlinarith [hi12_abs]
  refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i ?_ ?_) ?_
  · intro _; exact (fixed_exp.round_safe i.val hi12_abs100).1
  · intro _; exact (fixed_exp.round_safe i.val hi12_abs100).2
  · intro i1 hi1
    have hi1_abs : |i1.val| ≤ 2^21 := by
      rw [hi1]
      have h := fixed_exp.round_q32_abs_le' i.val (2^20) (by nlinarith [hi12_abs])
      omega
    refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i1) (y := 5965232#i128) ?_ ?_) ?_
    · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi1_abs; omega
    · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi1_abs; omega
    · intro acc hacc
      have hacc_lo : -(2^23) ≤ acc.val := by
        rw [hacc]; have := abs_le.mp hi1_abs; norm_num; omega
      have hacc_hi : acc.val ≤ 2^23 := by
        rw [hacc]; have := abs_le.mp hi1_abs; norm_num; omega
      have hacc_abs : |acc.val| ≤ 2^23 := abs_le.mpr ⟨hacc_lo, hacc_hi⟩
      have hi2_prod : |acc.val * r.val| ≤ 2^100 := by
        rw [abs_mul]; nlinarith [hacc_abs, hrA, abs_nonneg acc.val, abs_nonneg r.val]
      refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
      · rw [core.convert.num.FromI128I64.from_val_eq]
        have := abs_le.mp hi2_prod
        norm_num [I128.min, I128.numBits]; omega
      · rw [core.convert.num.FromI128I64.from_val_eq]
        have := abs_le.mp hi2_prod
        norm_num [I128.max, I128.numBits]; omega
      · intro i2 hi2
        have hi2_lo : -(2^55) ≤ i2.val := by
          rw [hi2, core.convert.num.FromI128I64.from_val_eq]
          have ha := abs_le.mp hacc_abs; have hr2 := hr'
          norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
        have hi2_hi : i2.val ≤ 2^55 := by
          rw [hi2, core.convert.num.FromI128I64.from_val_eq]
          have ha := abs_le.mp hacc_abs; have hr2 := hr'
          norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
        have hi2_abs : |i2.val| ≤ 2^55 := abs_le.mpr ⟨hi2_lo, hi2_hi⟩
        have hi2_abs100 : |i2.val| ≤ 2^100 := by nlinarith [hi2_abs]
        refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i2 ?_ ?_) ?_
        · intro _; exact (fixed_exp.round_safe i2.val hi2_abs100).1
        · intro _; exact (fixed_exp.round_safe i2.val hi2_abs100).2
        · intro i3 hi3
          have hi3_abs : |i3.val| ≤ 2^24 := by
            rw [hi3]
            have h := fixed_exp.round_q32_abs_le' i2.val (2^23) (by nlinarith [hi2_abs])
            omega
          refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i3) (y := 35791394#i128) ?_ ?_) ?_
          · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi3_abs; omega
          · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi3_abs; omega
          · intro acc1 hacc1
            have hacc1_lo : -(2^26) ≤ acc1.val := by
              rw [hacc1]; have := abs_le.mp hi3_abs; norm_num; omega
            have hacc1_hi : acc1.val ≤ 2^26 := by
              rw [hacc1]; have := abs_le.mp hi3_abs; norm_num; omega
            have hacc1_abs : |acc1.val| ≤ 2^26 := abs_le.mpr ⟨hacc1_lo, hacc1_hi⟩
            rw [Aeneas.Std.WP.spec_ok]
            rw [hacc1, hi3, hi2, hacc, hi1, i_post, core.convert.num.FromI128I64.from_val_eq]
            norm_num

set_option maxHeartbeats 8000000 in
/-- `exp_horner` spec: on `|r.val| ≤ 2^32` the extracted `exp_horner` returns the accumulator value
`fixed_exp.horner_acc r.val`, with no `i128` refusal. Fresh proof against `exp_horner`'s body (the
per-operation `I128.mul_spec`/`I128.add_spec`/`fixed_exp_round_div_2p32_val` route); the
`maxHeartbeats 8000000` guard is the lead-authorized finite cap for this single theorem. -/
theorem word_fixed_exp_horner_acc (r : I64) (hr : |r.val| ≤ 2 ^ 32) :
    fixed_exp.exp_horner r ⦃ z => z.val = fixed_exp.horner_acc r.val ∧ |z.val| ≤ 2 ^ 40 ⦄ := by
  unfold fixed_exp.exp_horner

  have hr' := abs_le.mp hr
  simp only [lift, bind_tc_ok]
  step*
  have hi12_abs : |i.val| ≤ 2^52 := by
    rw [i_post, core.convert.num.FromI128I64.from_val_eq, abs_mul,
      abs_of_nonneg (by norm_num : (0:ℤ) ≤ 852176)]
    nlinarith [hr'.1, hr'.2, abs_nonneg r.val]
  have hrA : |(core.convert.num.FromI128I64.from r).val| ≤ 2^32 := by
    rw [core.convert.num.FromI128I64.from_val_eq]; exact hr
  have hi12_abs100 : |i.val| ≤ 2^100 := by nlinarith [hi12_abs]
  refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i ?_ ?_) ?_
  · intro _; exact (fixed_exp.round_safe i.val hi12_abs100).1
  · intro _; exact (fixed_exp.round_safe i.val hi12_abs100).2
  · intro i1 hi1
    have hi1_abs : |i1.val| ≤ 2^21 := by
      rw [hi1]
      have h := fixed_exp.round_q32_abs_le' i.val (2^20) (by nlinarith [hi12_abs])
      omega
    refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i1) (y := 5965232#i128) ?_ ?_) ?_
    · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi1_abs; omega
    · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi1_abs; omega
    · intro acc hacc
      have hacc_lo : -(2^23) ≤ acc.val := by
        rw [hacc]; have := abs_le.mp hi1_abs; norm_num; omega
      have hacc_hi : acc.val ≤ 2^23 := by
        rw [hacc]; have := abs_le.mp hi1_abs; norm_num; omega
      have hacc_abs : |acc.val| ≤ 2^23 := abs_le.mpr ⟨hacc_lo, hacc_hi⟩
      have hi2_prod : |acc.val * r.val| ≤ 2^100 := by
        rw [abs_mul]; nlinarith [hacc_abs, hrA, abs_nonneg acc.val, abs_nonneg r.val]
      refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
      · rw [core.convert.num.FromI128I64.from_val_eq]
        have := abs_le.mp hi2_prod
        norm_num [I128.min, I128.numBits]; omega
      · rw [core.convert.num.FromI128I64.from_val_eq]
        have := abs_le.mp hi2_prod
        norm_num [I128.max, I128.numBits]; omega
      · intro i2 hi2
        have hi2_lo : -(2^55) ≤ i2.val := by
          rw [hi2, core.convert.num.FromI128I64.from_val_eq]
          have ha := abs_le.mp hacc_abs; have hr2 := hr'
          norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
        have hi2_hi : i2.val ≤ 2^55 := by
          rw [hi2, core.convert.num.FromI128I64.from_val_eq]
          have ha := abs_le.mp hacc_abs; have hr2 := hr'
          norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
        have hi2_abs : |i2.val| ≤ 2^55 := abs_le.mpr ⟨hi2_lo, hi2_hi⟩
        have hi2_abs100 : |i2.val| ≤ 2^100 := by nlinarith [hi2_abs]
        refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i2 ?_ ?_) ?_
        · intro _; exact (fixed_exp.round_safe i2.val hi2_abs100).1
        · intro _; exact (fixed_exp.round_safe i2.val hi2_abs100).2
        · intro i3 hi3
          have hi3_abs : |i3.val| ≤ 2^24 := by
            rw [hi3]
            have h := fixed_exp.round_q32_abs_le' i2.val (2^23) (by nlinarith [hi2_abs])
            omega
          refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i3) (y := 35791394#i128) ?_ ?_) ?_
          · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi3_abs; omega
          · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi3_abs; omega
          · intro acc1 hacc1
            have hacc1_lo : -(2^26) ≤ acc1.val := by
              rw [hacc1]; have := abs_le.mp hi3_abs; norm_num; omega
            have hacc1_hi : acc1.val ≤ 2^26 := by
              rw [hacc1]; have := abs_le.mp hi3_abs; norm_num; omega
            have hacc1_abs : |acc1.val| ≤ 2^26 := abs_le.mpr ⟨hacc1_lo, hacc1_hi⟩
            have hi4_prod : |acc1.val * r.val| ≤ 2^100 := by
              rw [abs_mul]; nlinarith [hacc1_abs, hrA, abs_nonneg acc1.val, abs_nonneg r.val]
            refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc1) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
            · rw [core.convert.num.FromI128I64.from_val_eq]
              have := abs_le.mp hi4_prod
              norm_num [I128.min, I128.numBits]; omega
            · rw [core.convert.num.FromI128I64.from_val_eq]
              have := abs_le.mp hi4_prod
              norm_num [I128.max, I128.numBits]; omega
            · intro i4 hi4
              have hi4_lo : -(2^58) ≤ i4.val := by
                rw [hi4, core.convert.num.FromI128I64.from_val_eq]
                have ha := abs_le.mp hacc1_abs; have hr2 := hr'
                norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
              have hi4_hi : i4.val ≤ 2^58 := by
                rw [hi4, core.convert.num.FromI128I64.from_val_eq]
                have ha := abs_le.mp hacc1_abs; have hr2 := hr'
                norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
              have hi4_abs : |i4.val| ≤ 2^58 := abs_le.mpr ⟨hi4_lo, hi4_hi⟩
              have hi4_abs100 : |i4.val| ≤ 2^100 := by nlinarith [hi4_abs]
              refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i4 ?_ ?_) ?_
              · intro _; exact (fixed_exp.round_safe i4.val hi4_abs100).1
              · intro _; exact (fixed_exp.round_safe i4.val hi4_abs100).2
              · intro i5 hi5
                have hi5_abs : |i5.val| ≤ 2^27 := by
                  rw [hi5]
                  have h := fixed_exp.round_q32_abs_le' i4.val (2^26) (by nlinarith [hi4_abs])
                  omega
                refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i5) (y := 178956971#i128) ?_ ?_) ?_
                · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi5_abs; omega
                · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi5_abs; omega
                · intro acc2 hacc2
                  have hacc2_lo : -(2^29) ≤ acc2.val := by
                    rw [hacc2]; have := abs_le.mp hi5_abs; norm_num; omega
                  have hacc2_hi : acc2.val ≤ 2^29 := by
                    rw [hacc2]; have := abs_le.mp hi5_abs; norm_num; omega
                  have hacc2_abs : |acc2.val| ≤ 2^29 := abs_le.mpr ⟨hacc2_lo, hacc2_hi⟩
                  have hi6_prod : |acc2.val * r.val| ≤ 2^100 := by
                    rw [abs_mul]; nlinarith [hacc2_abs, hrA, abs_nonneg acc2.val, abs_nonneg r.val]
                  refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc2) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
                  · rw [core.convert.num.FromI128I64.from_val_eq]
                    have := abs_le.mp hi6_prod
                    norm_num [I128.min, I128.numBits]; omega
                  · rw [core.convert.num.FromI128I64.from_val_eq]
                    have := abs_le.mp hi6_prod
                    norm_num [I128.max, I128.numBits]; omega
                  · intro i6 hi6
                    have hi6_lo : -(2^61) ≤ i6.val := by
                      rw [hi6, core.convert.num.FromI128I64.from_val_eq]
                      have ha := abs_le.mp hacc2_abs; have hr2 := hr'
                      norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                    have hi6_hi : i6.val ≤ 2^61 := by
                      rw [hi6, core.convert.num.FromI128I64.from_val_eq]
                      have ha := abs_le.mp hacc2_abs; have hr2 := hr'
                      norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                    have hi6_abs : |i6.val| ≤ 2^61 := abs_le.mpr ⟨hi6_lo, hi6_hi⟩
                    have hi6_abs100 : |i6.val| ≤ 2^100 := by nlinarith [hi6_abs]
                    refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i6 ?_ ?_) ?_
                    · intro _; exact (fixed_exp.round_safe i6.val hi6_abs100).1
                    · intro _; exact (fixed_exp.round_safe i6.val hi6_abs100).2
                    · intro i7 hi7
                      have hi7_abs : |i7.val| ≤ 2^30 := by
                        rw [hi7]
                        have h := fixed_exp.round_q32_abs_le' i6.val (2^29) (by nlinarith [hi6_abs])
                        omega
                      refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i7) (y := 715827883#i128) ?_ ?_) ?_
                      · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi7_abs; omega
                      · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi7_abs; omega
                      · intro acc3 hacc3
                        have hacc3_lo : -(2^31) ≤ acc3.val := by
                          rw [hacc3]; have := abs_le.mp hi7_abs; norm_num; omega
                        have hacc3_hi : acc3.val ≤ 2^31 := by
                          rw [hacc3]; have := abs_le.mp hi7_abs; norm_num; omega
                        have hacc3_abs : |acc3.val| ≤ 2^31 := abs_le.mpr ⟨hacc3_lo, hacc3_hi⟩
                        have hi8_prod : |acc3.val * r.val| ≤ 2^100 := by
                          rw [abs_mul]; nlinarith [hacc3_abs, hrA, abs_nonneg acc3.val, abs_nonneg r.val]
                        refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc3) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
                        · rw [core.convert.num.FromI128I64.from_val_eq]
                          have := abs_le.mp hi8_prod
                          norm_num [I128.min, I128.numBits]; omega
                        · rw [core.convert.num.FromI128I64.from_val_eq]
                          have := abs_le.mp hi8_prod
                          norm_num [I128.max, I128.numBits]; omega
                        · intro i8 hi8
                          have hi8_lo : -(2^63) ≤ i8.val := by
                            rw [hi8, core.convert.num.FromI128I64.from_val_eq]
                            have ha := abs_le.mp hacc3_abs; have hr2 := hr'
                            norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                          have hi8_hi : i8.val ≤ 2^63 := by
                            rw [hi8, core.convert.num.FromI128I64.from_val_eq]
                            have ha := abs_le.mp hacc3_abs; have hr2 := hr'
                            norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                          have hi8_abs : |i8.val| ≤ 2^63 := abs_le.mpr ⟨hi8_lo, hi8_hi⟩
                          have hi8_abs100 : |i8.val| ≤ 2^100 := by nlinarith [hi8_abs]
                          refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i8 ?_ ?_) ?_
                          · intro _; exact (fixed_exp.round_safe i8.val hi8_abs100).1
                          · intro _; exact (fixed_exp.round_safe i8.val hi8_abs100).2
                          · intro i9 hi9
                            have hi9_abs : |i9.val| ≤ 2^32 := by
                              rw [hi9]
                              have h := fixed_exp.round_q32_abs_le' i8.val (2^31) (by nlinarith [hi8_abs])
                              omega
                            refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i9) (y := 2147483648#i128) ?_ ?_) ?_
                            · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi9_abs; omega
                            · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi9_abs; omega
                            · intro acc4 hacc4
                              have hacc4_lo : -(2^33) ≤ acc4.val := by
                                rw [hacc4]; have := abs_le.mp hi9_abs; norm_num; omega
                              have hacc4_hi : acc4.val ≤ 2^33 := by
                                rw [hacc4]; have := abs_le.mp hi9_abs; norm_num; omega
                              have hacc4_abs : |acc4.val| ≤ 2^33 := abs_le.mpr ⟨hacc4_lo, hacc4_hi⟩
                              have hi10_prod : |acc4.val * r.val| ≤ 2^100 := by
                                rw [abs_mul]; nlinarith [hacc4_abs, hrA, abs_nonneg acc4.val, abs_nonneg r.val]
                              refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc4) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
                              · rw [core.convert.num.FromI128I64.from_val_eq]
                                have := abs_le.mp hi10_prod
                                norm_num [I128.min, I128.numBits]; omega
                              · rw [core.convert.num.FromI128I64.from_val_eq]
                                have := abs_le.mp hi10_prod
                                norm_num [I128.max, I128.numBits]; omega
                              · intro i10 hi10
                                have hi10_lo : -(2^65) ≤ i10.val := by
                                  rw [hi10, core.convert.num.FromI128I64.from_val_eq]
                                  have ha := abs_le.mp hacc4_abs; have hr2 := hr'
                                  norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                                have hi10_hi : i10.val ≤ 2^65 := by
                                  rw [hi10, core.convert.num.FromI128I64.from_val_eq]
                                  have ha := abs_le.mp hacc4_abs; have hr2 := hr'
                                  norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                                have hi10_abs : |i10.val| ≤ 2^65 := abs_le.mpr ⟨hi10_lo, hi10_hi⟩
                                have hi10_abs100 : |i10.val| ≤ 2^100 := by nlinarith [hi10_abs]
                                refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i10 ?_ ?_) ?_
                                · intro _; exact (fixed_exp.round_safe i10.val hi10_abs100).1
                                · intro _; exact (fixed_exp.round_safe i10.val hi10_abs100).2
                                · intro i11 hi11
                                  have hi11_abs : |i11.val| ≤ 2^34 := by
                                    rw [hi11]
                                    have h := fixed_exp.round_q32_abs_le' i10.val (2^33) (by nlinarith [hi10_abs])
                                    omega
                                  refine Aeneas.Std.WP.spec_bind (I128.add_spec (x := i11) (y := 4294967296#i128) ?_ ?_) ?_
                                  · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi11_abs; omega
                                  · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi11_abs; omega
                                  · intro acc5 hacc5
                                    have hacc5_lo : -(2^35) ≤ acc5.val := by
                                      rw [hacc5]; have := abs_le.mp hi11_abs; norm_num; omega
                                    have hacc5_hi : acc5.val ≤ 2^35 := by
                                      rw [hacc5]; have := abs_le.mp hi11_abs; norm_num; omega
                                    have hacc5_abs : |acc5.val| ≤ 2^35 := abs_le.mpr ⟨hacc5_lo, hacc5_hi⟩
                                    have hi12_prod : |acc5.val * r.val| ≤ 2^100 := by
                                      rw [abs_mul]; nlinarith [hacc5_abs, hrA, abs_nonneg acc5.val, abs_nonneg r.val]
                                    refine Aeneas.Std.WP.spec_bind (I128.mul_spec (x := acc5) (y := core.convert.num.FromI128I64.from r) ?_ ?_) ?_
                                    · rw [core.convert.num.FromI128I64.from_val_eq]
                                      have := abs_le.mp hi12_prod
                                      norm_num [I128.min, I128.numBits]; omega
                                    · rw [core.convert.num.FromI128I64.from_val_eq]
                                      have := abs_le.mp hi12_prod
                                      norm_num [I128.max, I128.numBits]; omega
                                    · intro i12 hi12
                                      have hi12_lo : -(2^67) ≤ i12.val := by
                                        rw [hi12, core.convert.num.FromI128I64.from_val_eq]
                                        have ha := abs_le.mp hacc5_abs; have hr2 := hr'
                                        norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                                      have hi12_hi : i12.val ≤ 2^67 := by
                                        rw [hi12, core.convert.num.FromI128I64.from_val_eq]
                                        have ha := abs_le.mp hacc5_abs; have hr2 := hr'
                                        norm_num; nlinarith [ha.1, ha.2, hr2.1, hr2.2]
                                      have hi12_abs : |i12.val| ≤ 2^67 := abs_le.mpr ⟨hi12_lo, hi12_hi⟩
                                      have hi12_abs100 : |i12.val| ≤ 2^100 := by nlinarith [hi12_abs]
                                      refine Aeneas.Std.WP.spec_bind (fixed_exp_round_div_2p32_val i12 ?_ ?_) ?_
                                      · intro _; exact (fixed_exp.round_safe i12.val hi12_abs100).1
                                      · intro _; exact (fixed_exp.round_safe i12.val hi12_abs100).2
                                      · intro i13 hi13
                                        have hi13_abs : |i13.val| ≤ 2^36 := by
                                          rw [hi13]
                                          have h := fixed_exp.round_q32_abs_le' i12.val (2^35) (by nlinarith [hi12_abs])
                                          omega
                                        refine Aeneas.Std.WP.spec_mono (I128.add_spec (x := i13) (y := 4294967296#i128) ?_ ?_) ?_
                                        · norm_num [I128.min, I128.numBits]; have := abs_le.mp hi13_abs; omega
                                        · norm_num [I128.max, I128.numBits]; have := abs_le.mp hi13_abs; omega
                                        · intro z hz
                                          refine ⟨?_, ?_⟩
                                          · rw [hz]
                                            rw [hi13, hi12, hacc5, hi11, hi10, hacc4, hi9, hi8, hacc3, hi7, hi6, hacc2, hi5, hi4, hacc1, hi3, hi2, hacc, hi1, i_post]
                                            simp only [core.convert.num.FromI128I64.from_val_eq]
                                            rw [fixed_exp.horner_acc]
                                            norm_num
                                          · rw [abs_le]
                                            rw [hz]
                                            have hb := abs_le.mp hi13_abs
                                            norm_num at hb ⊢
                                            constructor <;> omega


end

end apeiron_word
