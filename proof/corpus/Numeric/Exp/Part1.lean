/-
Refinement of the fixed-point `exp` body against the real exponential
(`APEIRON-INT-TRANSCENDENTAL-W1-T1`; spec `docs/epics/inference-parity.md`).

The Rust body is `src/numeric/fixed_exp.rs`: `i64` Q32.32, window `[-16, 21]`, degree-7 Horner
after reduction by `ln 2`, `Result` refusal outside the window. It is extracted into `ApeironWord`
by `proof/regen.sh`; the refinement statement, the representation relation and the error bound are
built here under the closed axiom cone.

## Representation

An `i64` (or `i128`) `v` denotes the real number `toReal v = v.val / 2^32`, the signed Q32.32 value.
`window x` is the documented admissible domain `[-16·2^32, 21·2^32]`. The refusal correspondence is
`fixed_exp_refuses`: outside `window` the extracted body returns `Err OutOfRange`.

## Statement choice

The full window carries the combined bound

  `|toReal v - exp (toReal x)| <= 2^-20 · exp (toReal x) + 4 / 2^32`,

matching the Rust contract (`src/numeric/fixed_exp.rs` asserts `1e-8·exp + 4/2^32` on the whole
window, with `1e-8 < 2^-20`). The pure relative target `2^-20` holds only for `x >= -8`, where the
fixed-point grid does not dominate: at `x = -16` the output `483` units makes half a unit `6.9e-4`
relative, so a statement with no absolute floor would be false at the window's lower edge. The
statement below is therefore the combined form on the whole window; `fixed_exp_refines` states it.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import ApeironWord
import Numeric.Scale

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Representation relation -/

/-- The documented admissible window `[-16·2^32, 21·2^32]`, as a predicate on the signed value. -/
def window (x : I64) : Prop := (-68719476736 : ℤ) ≤ x.val ∧ x.val ≤ 90194313216

/-! ## Refusal and bound statements -/

/-- Refusal correspondence: outside the window the extracted body returns `Err OutOfRange`. -/
def fixed_exp_refuses : Prop :=
  ∀ x : I64,
    ¬ window x →
      word_fixed_exp x = ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)

/-- The target refinement bound on the window: success, with the combined absolute/relative error
bound `2^-20 · exp x + 4 / 2^32`. -/
def fixed_exp_refines : Prop :=
  ∀ x : I64,
    window x →
      ∃ v : I64,
        word_fixed_exp x = ok (core.result.Result.Ok v) ∧
          |toReal v - Real.exp (toReal x)| ≤ epsExp * Real.exp (toReal x) + 4 / q32

/-! ## Refusal: structural lemmas and endpoint evaluation

The extracted body opens with `let lower ← LOWER; if x < lower then Err else …`, with the mirrored
upper test in the second branch. `word_fixed_exp_refuses_of_lt`/`_of_gt` are proved from that
structure alone, parameterised by the values of the extracted endpoint `const`s: when the local
comparison holds, the body returns `Err OutOfRange`.

The specialised Aeneas shift and multiply specifications evaluate the extracted `const`s:
`ONE = 1 <<< 32 = 2^32`, `LOWER = -16 · ONE = -68719476736`, `UPPER = 21 · ONE = 90194313216`.
`fixed_exp_refuses_proved` instantiates the parameterised lemmas at those values and discharges
`fixed_exp_refuses` over the complement of `window`.
-/

/-- Below the evaluated lower endpoint the extracted body refuses. -/
theorem word_fixed_exp_refuses_of_lt (x c : I64)
    (hL : fixed_exp.exp_reduce.LOWER = ok c) (h : x < c) :
    word_fixed_exp x = ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange) := by
  rw [word_fixed_exp, fixed_exp.fixed_exp]
  unfold fixed_exp.exp_reduce
  rw [hL]
  simp only [bind_tc_ok, if_pos h]

/-- Above the evaluated upper endpoint the extracted body refuses. -/
theorem word_fixed_exp_refuses_of_gt (x cL cU : I64)
    (hL : fixed_exp.exp_reduce.LOWER = ok cL) (hU : fixed_exp.exp_reduce.UPPER = ok cU)
    (hn : ¬ x < cL) (h : x > cU) :
    word_fixed_exp x = ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange) := by
  rw [word_fixed_exp, fixed_exp.fixed_exp]
  unfold fixed_exp.exp_reduce
  rw [hL, hU]
  simp only [bind_tc_ok, if_neg hn, if_pos h]

/-! ## Endpoint evaluation and instantiation -/

/-- The extracted scale constant `ONE` evaluates to `2^32`. -/
theorem fixed_exp_ONE_eq : fixed_exp.ONE = ok (4294967296#i64) := by
  have h : fixed_exp.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold fixed_exp.ONE
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted lower endpoint evaluates to `-16 · 2^32`. -/
theorem fixed_exp_LOWER_eq : fixed_exp.exp_reduce.LOWER = ok ((-68719476736)#i64) := by
  have h : fixed_exp.exp_reduce.LOWER ⦃ z => z = ((-68719476736)#i64) ⦄ := by
    unfold fixed_exp.exp_reduce.LOWER
    rw [fixed_exp_ONE_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := (-16)#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted upper endpoint evaluates to `21 · 2^32`. -/
theorem fixed_exp_UPPER_eq : fixed_exp.exp_reduce.UPPER = ok (90194313216#i64) := by
  have h : fixed_exp.exp_reduce.UPPER ⦃ z => z = (90194313216#i64) ⦄ := by
    unfold fixed_exp.exp_reduce.UPPER
    rw [fixed_exp_ONE_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := 21#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- Refusal correspondence: outside `window` the extracted body returns `Err OutOfRange`. -/
theorem fixed_exp_refuses_proved : fixed_exp_refuses := by
  intro x hx
  rw [window] at hx
  by_cases hlow : x.val < (-68719476736 : ℤ)
  · exact word_fixed_exp_refuses_of_lt x _ fixed_exp_LOWER_eq (by simpa using hlow)
  · have hlowle : (-68719476736 : ℤ) ≤ x.val := le_of_not_gt hlow
    have hhigh : (90194313216 : ℤ) < x.val := by
      by_contra hc
      exact hx ⟨hlowle, le_of_not_gt hc⟩
    refine word_fixed_exp_refuses_of_gt x _ _ fixed_exp_LOWER_eq fixed_exp_UPPER_eq ?_ ?_
    · simp only [IScalar.lt_equiv]
      exact not_lt.mpr hlowle
    · simpa using hhigh

/-- The extracted rounding half-constant `2^31` evaluates, by the same `step`/`eq_imp` pattern as
the endpoint constants. -/
theorem fixed_exp_round_div_2p32_HALF_eq :
    fixed_exp.round_div_2p32.HALF = ok (2147483648#i128) := by
  have h : fixed_exp.round_div_2p32.HALF ⦃ z => z = (2147483648#i128) ⦄ := by
    unfold fixed_exp.round_div_2p32.HALF
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## Rounding primitive

`fixed_exp.round_div_2p32` is the degree-7 Horner rounding step: add `2^31` and shift right by 32,
rounding to nearest. The success spec below names the two overflow preconditions the body needs
(`+2^31` cannot exceed `I128.max` on the nonnegative branch, and on the negative branch
`-value.val ≠ I128.min` — derived from the same bound — so the negation and the shifted result do
not overflow). This is the step spec `step` needs to continue through the Horner chain.

`int_shiftRight_nonneg` is the general fact the negative branch uses; Lean's core does not expose
it as a named lemma. -/

theorem int_shiftRight_nonneg (a : ℤ) (s : ℕ) (h : 0 ≤ a) : 0 ≤ a >>> s := by
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le h
  show (0 : ℤ) ≤ ↑(n >>> s)
  exact Int.natCast_nonneg _

/-- Success spec for the extracted rounding step. On `value.val ≥ 0` the result is the arithmetic
right shift of `value.val + 2^31`; on `value.val < 0` it is the negation of that shift of
`-value.val + 2^31`. -/
theorem fixed_exp_round_div_2p32_spec (value : I128)
    (hpos : 0 ≤ value.val → value.val + 2147483648 ≤ I128.max)
    (hneg : value.val < 0 → -value.val + 2147483648 ≤ I128.max) :
    fixed_exp.round_div_2p32 value ⦃ z =>
      (0 ≤ value.val → z.val = (value.val + 2147483648) >>> 32) ∧
      (value.val < 0 → z.val = - ((-value.val + 2147483648) >>> 32)) ⦄ := by
  unfold fixed_exp.round_div_2p32
  rw [fixed_exp_round_div_2p32_HALF_eq]
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
       have h3 : 0 ≤ i2.val >>> 32 := int_shiftRight_nonneg _ _ h2
       have hminlt : IScalar.min IScalarTy.I128 < 0 := by
         norm_num [IScalar.min, I128.min, I128.numBits]
       intro heq
       omega)

/-!
## Retired: flat-body window rewrite

The old `word_fixed_exp_window_past_guards` rewrite is retired: the extracted body is now the helper
call chain `exp_reduce`/`exp_horner`/`exp_scale`, so the reduction is characterized by the
`exp_reduce` spec rather than by unfolding the flat body. -/

/-! ## Real-error decomposition skeleton

The exact `toReal` images of the two extracted reduction constants, and the integer defect of the
reciprocal approximation. The reduction computes `k` by rounding `x · LN2_RECIP_Q32 / 2^64` and then
`r = x - k · LN2_Q32`; the deviation of `LN2_RECIP_Q32` from `2^64 / LN2_Q32` is
`ln2_recip_defect`, the multiplicative error budget of the reduction. The remainder bound itself
needs the value-level unfold and is not stated here.
-/

theorem toReal_ln2 : toReal fixed_exp.LN2_Q32 = 2977044472 / q32 := by
  unfold toReal fixed_exp.LN2_Q32
  norm_num

theorem toReal_ln2_recip : toReal fixed_exp.LN2_RECIP_Q32 = 6196328019 / q32 := by
  unfold toReal fixed_exp.LN2_RECIP_Q32
  norm_num

theorem ln2_recip_defect :
    (6196328019 : ℤ) * 2977044472 - (2 : ℤ) ^ 64 = 1953109352 := by
  norm_num

/-! ## Integer range facts for `x - k · LN2_Q32`

The reduction rounds `k = (x · LN2_RECIP_Q32 + 2^63) >>> 64` (arithmetic right shift of `i64`,
`ln 2` reciprocal scaled by `2^32`) and forms `r = x - k · LN2_Q32`, both in `i64`. The extracted
body tests `k_ln2_wide` and `x - k_ln2` against the `i128`/`i64` endpoints; these lemmas supply the
arithmetic those refusal tests need, independently of the `step` context:

* `fixed_exp_shift_recip_shift_bound` bounds the shift `k` on the window (`-24 ≤ k ≤ 30`);
* `fixed_exp_shift_recip_bound_window` lifts that to `k · LN2_Q32 ∈ [i64.min, i64.max]`, so the
  `i128→i64` cast of `k_ln2_wide` is the identity (`fixed_exp_bmod_self_64`);
* `fixed_exp_reduction_remainder_range` bounds `r = x - k · LN2_Q32` in `[i64.min, i64.max]`, so
  the extracted `x - k_ln2` cannot overflow.

The proof splits `x · 6196328019 + 2^63 = k · 2^64 + t` with `t = · % 2^64 ∈ [0, 2^64)` and uses
`6196328019 · 2977044472 = 2^64 + 1953109352` (`ln2_recip_defect`) to write
`r · 6196328019 = t - 2^63 - k · 1953109352`. -/

/-- The reduction shift `k = (x · LN2_RECIP_Q32 + 2^63) >>> 64` lies in `[-24, 30]` on the window,
by `-68719476736 · 6196328019 + 2^63` and `90194313216 · 6196328019 + 2^63` after the floor
division by `2^64`. -/
theorem fixed_exp_shift_recip_shift_bound (x s : ℤ)
    (hlo : -68719476736 ≤ x) (hhi : x ≤ 90194313216)
    (hs : s = (x * 6196328019 + 2 ^ 63) >>> 64) :
    -24 ≤ s ∧ s ≤ 30 := by
  have hsdiv : s = (x * 6196328019 + 2 ^ 63) / 2 ^ 64 := by
    rw [hs, Int.shiftRight_eq_div_pow]
    norm_num
  have hA : 2 ^ 64 * s + (x * 6196328019 + 2 ^ 63) % 2 ^ 64
      = x * 6196328019 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 6196328019 + 2 ^ 63) (2 ^ 64)
    rw [← hsdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 6196328019 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 6196328019 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmin : -68719476736 * 6196328019 + 2 ^ 63 ≤ x * 6196328019 + 2 ^ 63 := by nlinarith [hlo]
  have hmax : x * 6196328019 + 2 ^ 63 ≤ 90194313216 * 6196328019 + 2 ^ 63 := by nlinarith [hhi]
  have hslo : -24 ≤ s := by
    by_contra hc
    have hsle : s ≤ -25 := by omega
    have hlt : x * 6196328019 + 2 ^ 63 < (s + 1) * 2 ^ 64 := by nlinarith [hA, htlt]
    nlinarith [hlt, hmin, hsle]
  have hshi : s ≤ 30 := by
    by_contra hc
    have hsge : 31 ≤ s := by omega
    have hle : 31 * 2 ^ 64 ≤ x * 6196328019 + 2 ^ 63 := by nlinarith [hA, ht0, hsge]
    nlinarith [hle, hmax]
  exact ⟨hslo, hshi⟩

/-- `k · LN2_Q32` lies in the `i64` range on the window, so the `i128→i64` cast and the following
subtraction see the true integer product. -/
theorem fixed_exp_shift_recip_bound_window (x s : ℤ)
    (hlo : -68719476736 ≤ x) (hhi : x ≤ 90194313216)
    (hs : s = (x * 6196328019 + 2 ^ 63) >>> 64) :
    -2 ^ 63 ≤ s * 2977044472 ∧ s * 2977044472 ≤ 2 ^ 63 - 1 := by
  obtain ⟨hslo, hshi⟩ := fixed_exp_shift_recip_shift_bound x s hlo hhi hs
  constructor <;> nlinarith

/-- Lower bound of the balanced `bmod 2^64`, independent of the value. -/
theorem fixed_exp_bmod_64_lower (a : ℤ) : -2 ^ 63 ≤ a.bmod (2 ^ 64) := by
  have h := Int.le_bmod (x := a) (m := 2 ^ 64) (by norm_num)
  norm_num at h
  exact h

/-- Upper bound of the balanced `bmod 2^64`, independent of the value. -/
theorem fixed_exp_bmod_64_upper (a : ℤ) : a.bmod (2 ^ 64) < 2 ^ 63 := by
  have h := Int.bmod_lt (x := a) (m := 2 ^ 64) (by norm_num)
  norm_num at h
  exact h

/-- `bmod 2^64` is the identity on the `i64` range. -/
theorem fixed_exp_bmod_self_64 (a : ℤ) (hlo : -2 ^ 63 ≤ a) (hhi : a ≤ 2 ^ 63 - 1) :
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

/-- The reduction remainder `r = x - k · LN2_Q32` lies in the `i64` range whenever `k` is computed
by the shift and `k · LN2_Q32` is already in range. -/
theorem fixed_exp_reduction_remainder_range (x s : ℤ)
    (hshift : s = (x * 6196328019 + 2 ^ 63) >>> 64)
    (hk : -2 ^ 63 ≤ s * 2977044472 ∧ s * 2977044472 ≤ 2 ^ 63 - 1) :
    -2 ^ 63 ≤ x - s * 2977044472 ∧ x - s * 2977044472 ≤ 2 ^ 63 - 1 := by
  have hshift' : (x * 6196328019 + 2 ^ 63) >>> 64 = s := hshift.symm
  have hdiv : (x * 6196328019 + 2 ^ 63) / 2 ^ 64 = s := by
    rw [Int.shiftRight_eq_div_pow] at hshift'
    norm_num at hshift'
    exact hshift'
  have hdecomp : 2 ^ 64 * s + (x * 6196328019 + 2 ^ 63) % 2 ^ 64
      = x * 6196328019 + 2 ^ 63 := by
    have := Int.mul_ediv_add_emod (x * 6196328019 + 2 ^ 63) (2 ^ 64)
    rw [hdiv] at this
    exact this
  have ht0 : 0 ≤ (x * 6196328019 + 2 ^ 63) % 2 ^ 64 := Int.emod_nonneg _ (by norm_num)
  have htlt : (x * 6196328019 + 2 ^ 63) % 2 ^ 64 < 2 ^ 64 := Int.emod_lt_of_pos _ (by norm_num)
  have hmR : (2977044472 : ℤ) * 6196328019 = 2 ^ 64 + 1953109352 := by norm_num
  have hDle : (1953109352 : ℤ) ≤ 2977044472 := by norm_num
  have hDnn : (0 : ℤ) ≤ 1953109352 := by norm_num
  have hsD_ub : s * 1953109352 ≤ 2 ^ 63 := by nlinarith [hk.1, hk.2, hDle, hDnn]
  have hsD_lb : -2 ^ 63 ≤ s * 1953109352 := by nlinarith [hk.1, hk.2, hDle, hDnn]
  have hrR : (x - s * 2977044472) * 6196328019
      = (x * 6196328019 + 2 ^ 63) % 2 ^ 64 - 2 ^ 63 - s * 1953109352 := by
    nlinarith [hdecomp, hmR]
  have hlow : -2 ^ 64 ≤ (x - s * 2977044472) * 6196328019 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hhigh : (x - s * 2977044472) * 6196328019 < 2 ^ 64 := by
    rw [hrR]; nlinarith [ht0, htlt, hsD_ub, hsD_lb]
  have hub : x - s * 2977044472 ≤ 2 ^ 32 - 1 := by
    by_contra hc
    have hrge : (2 : ℤ) ^ 32 ≤ x - s * 2977044472 := by omega
    have hmul : (2 : ℤ) ^ 32 * 6196328019 ≤ (x - s * 2977044472) * 6196328019 :=
      mul_le_mul_of_nonneg_right hrge (by norm_num)
    nlinarith [hhigh, hmul]
  have hlb : -(2 : ℤ) ^ 32 ≤ x - s * 2977044472 := by
    by_contra hc
    have hrle : x - s * 2977044472 ≤ -(2 : ℤ) ^ 32 := by omega
    have hmul : (x - s * 2977044472) * 6196328019 ≤ (-(2 : ℤ) ^ 32) * 6196328019 :=
      mul_le_mul_of_nonneg_right hrle (by norm_num)
    nlinarith [hlow, hmul]
  constructor <;> omega

/-- On the window the post-guard reduction prefix is arithmetically safe: every `i128`/`i64`
refusal test in `product`, `i5`, `k_ln2_wide`, the `i128→i64` narrowing and `x - k_ln2` passes, so
the extracted body does not take an `OutOfRange` branch in this prefix. The five WP side conditions
are discharged by the range lemmas above; the returned value and the `k`/`r` characterization are
`word_fixed_exp_reduction_value`. -/
theorem word_fixed_exp_reduction_arith_safe (x : I64) (hx : window x) :
    (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ←
          lift (core.convert.num.FromI128I64.from fixed_exp.LN2_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let shifted ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if shifted > i6
        then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if shifted < i7
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 shifted)
            let i8 ← lift (core.convert.num.FromI128I64.from k)
            let i9 ← lift (core.convert.num.FromI128I64.from fixed_exp.LN2_Q32)
            let k_ln2_wide ← i8 * i9
            let i10 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if k_ln2_wide > i10
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i11 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if k_ln2_wide < i11
              then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
              else
                let k_ln2 ← lift (IScalar.cast .I64 k_ln2_wide)
                let r ← x - k_ln2
                ok (core.result.Result.Ok r)) ⦃ _ => True ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq, fixed_exp.LN2_RECIP_Q32,
    fixed_exp.LN2_Q32, I128.min, I128.max, I64.min, I64.max] at *))
  all_goals (try norm_num at *)
  all_goals (try scalar_tac)
  all_goals (try (simp only [Int.shiftRight_eq_div_pow] at *))
  · -- goal: product + i4 <= I128.max
    have hi4 : (↑i4 : ℤ) = 2 ^ 63 := by
      rw [i4_post]; norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
    rw [product_post, hi4]
    norm_num [I128.numBits]
    nlinarith [hx.1, hx.2]
  · -- goal: lower bound of (shifted.bmod 2^64) * m
    have h := fixed_exp_bmod_64_lower (shifted : ℤ)
    norm_num [I128.numBits] at h ⊢
    nlinarith [h]
  · -- goal: upper bound of (shifted.bmod 2^64) * m
    have h := fixed_exp_bmod_64_upper (shifted : ℤ)
    norm_num [I128.numBits] at h ⊢
    nlinarith [h]
  · -- goal: lower range condition for r = x - k_ln2
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_exp_shift_recip_shift_bound x.val shifted.val hx.1 hx.2 hsv
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    rw [show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num]
    rw [hb, hkval]
    norm_num [I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).1
    omega
  · -- goal: upper range condition for r = x - k_ln2
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_exp_shift_recip_shift_bound x.val shifted.val hx.1 hx.2 hsv
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    rw [show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num]
    rw [hb, hkval]
    norm_num [I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).2
    omega

/-- On the window the post-guard reduction prefix returns `Ok r` (no `OutOfRange` refusal), with the
value characterization of the shift `k` and the remainder `r`: `k.val` is the arithmetic shift
`(x.val * LN2_RECIP_Q32 + 2^63) >>> 64` and `r.val = x.val - k.val * LN2_Q32`. Witness `k` is the
extracted `IScalar.cast .I64 shifted`. The four guard branches (`shifted`/`k_ln2_wide` against the
`i64` endpoints) are refuted from `fixed_exp_shift_recip_shift_bound` /
`fixed_exp_shift_recip_bound_window`. -/
theorem word_fixed_exp_reduction_value (x : I64) (hx : window x) :
    (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let i3 ←
          lift (core.convert.num.FromI128I64.from fixed_exp.LN2_RECIP_Q32)
        let product ← i2 * i3
        let i4 ← 1#i128 <<< 63#i32
        let i5 ← product + i4
        let shifted ← i5 >>> 64#i32
        let i6 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if shifted > i6
        then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
        else
          let i7 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if shifted < i7
          then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
          else
            let k ← lift (IScalar.cast .I64 shifted)
            let i8 ← lift (core.convert.num.FromI128I64.from k)
            let i9 ← lift (core.convert.num.FromI128I64.from fixed_exp.LN2_Q32)
            let k_ln2_wide ← i8 * i9
            let i10 ← lift (IScalar.cast .I128 core.num.I64.MAX)
            if k_ln2_wide > i10
            then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
            else
              let i11 ← lift (IScalar.cast .I128 core.num.I64.MIN)
              if k_ln2_wide < i11
              then ok (core.result.Result.Err fixed_exp.ExpError.OutOfRange)
              else
                let k_ln2 ← lift (IScalar.cast .I64 k_ln2_wide)
                let r ← x - k_ln2
                ok (core.result.Result.Ok r)) ⦃ z =>
      ∃ k r : I64, z = core.result.Result.Ok r ∧
        k.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 ∧
        r.val = x.val - k.val * 2977044472 ⦄ := by
  simp only [lift, bind_tc_ok]
  step*
  all_goals (try (simp only [core.convert.num.FromI128I64.from_val_eq, fixed_exp.LN2_RECIP_Q32,
    fixed_exp.LN2_Q32, I128.min, I128.max, I64.min, I64.max] at *))
  all_goals (try norm_num at *)
  all_goals (try scalar_tac)
  all_goals (try (simp only [Int.shiftRight_eq_div_pow] at *))
  · -- product + i4
    have hi4 : (↑i4 : ℤ) = 2 ^ 63 := by
      rw [i4_post]; norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
    rw [product_post, hi4]
    norm_num [I128.numBits]
    nlinarith [hx.1, hx.2]
  · -- guard: shifted <= I64.max
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_exp_shift_recip_shift_bound x.val shifted.val hx.1 hx.2 hsv
    have hle : (↑shifted : ℤ) ≤ 30 := hsb.2
    norm_num [I64.rMax] at *
    exfalso; omega
  · -- guard: shifted >= I64.min
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsb := fixed_exp_shift_recip_shift_bound x.val shifted.val hx.1 hx.2 hsv
    have hge : -24 ≤ (↑shifted : ℤ) := hsb.1
    norm_num [I64.rMin] at *
    exfalso; omega
  · -- shifted.bmod lower
    have h := fixed_exp_bmod_64_lower (shifted : ℤ)
    norm_num [I128.numBits] at h ⊢
    nlinarith [h]
  · -- shifted.bmod upper
    have h := fixed_exp_bmod_64_upper (shifted : ℤ)
    norm_num [I128.numBits] at h ⊢
    nlinarith [h]
  · -- guard: k_ln2_wide <= I64.max
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hlink : (↑k_ln2_wide : ℤ) = shifted.val * 2977044472 := hkval
    have hle : (↑k_ln2_wide : ℤ) ≤ 2 ^ 63 - 1 := by rw [hlink]; exact hsr.2
    norm_num [I64.rMax] at *
    exfalso; omega
  · -- guard: k_ln2_wide >= I64.min
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hlink : (↑k_ln2_wide : ℤ) = shifted.val * 2977044472 := hkval
    have hge : -2 ^ 63 ≤ (↑k_ln2_wide : ℤ) := by rw [hlink]; exact hsr.1
    norm_num [I64.rMin] at *
    exfalso; omega
  · -- k_ln2_wide.bmod lower
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    rw [show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num]
    rw [hb, hkval]
    norm_num [I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).1
    omega
  · -- k_ln2_wide.bmod upper
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    rw [show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num]
    rw [hb, hkval]
    norm_num [I64.numBits]
    have hr := (fixed_exp_reduction_remainder_range x.val shifted.val hsv hsr).2
    omega
  · -- post: value characterization with k = cast .I64 shifted
    have hsv : shifted.val = (x.val * 6196328019 + 2 ^ 63) >>> 64 := by
      have h1 : (↑shifted : ℤ) = (↑x * 6196328019 + 2 ^ 63) / ↑(2 ^ 64) := by
        rw [shifted_post, i5_post, i4_post, product_post]
        norm_num [I128.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
      rw [Int.shiftRight_eq_div_pow]
      exact h1
    have hsr := fixed_exp_shift_recip_bound_window x.val shifted.val hx.1 hx.2 hsv
    have hkval : k_ln2_wide.val = shifted.val * 2977044472 := by
      have h2 : (↑k_ln2_wide : ℤ) = ((shifted : ℤ)).bmod (2 ^ 64) * 2977044472 := by
        rw [k_ln2_wide_post]; norm_num
      rw [h2]
      change shifted.val.bmod (2 ^ 64) * 2977044472 = shifted.val * 2977044472
      rw [fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)]
    have hb : (k_ln2_wide : ℤ).bmod (2 ^ 64) = k_ln2_wide.val := by
      rw [hkval]
      exact fixed_exp_bmod_self_64 (shifted.val * 2977044472) hsr.1 hsr.2
    have hcast : (↑(IScalar.cast IScalarTy.I64 shifted) : ℤ) = shifted.val := by
      rw [IScalar.cast_val_eq]
      norm_num
      exact fixed_exp_bmod_self_64 shifted.val (by omega) (by omega)
    refine ⟨IScalar.cast IScalarTy.I64 shifted, ?_, ?_⟩
    · rw [hcast]
      rw [Int.shiftRight_eq_div_pow] at hsv
      norm_num at hsv ⊢
      exact hsv
    · rw [r_post, show (18446744073709551616 : ℕ) = 2 ^ 64 by norm_num, hb, hkval, hcast]

end

end apeiron_word
