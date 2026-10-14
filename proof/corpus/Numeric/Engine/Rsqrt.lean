/-
Refinement of the fixed-point `rsqrt` body against the real reciprocal square root
.

The Rust body is `src/numeric/fixed_rsqrt.rs`: `i64` Q32.32, window `[2^-20, 2^20]`, a scaled integer
square root `s = isqrt(x << 64)` plus one round-to-nearest division `out = ((1 << 80) + s/2) / s`,
`Result` refusal outside the window. It is extracted into `ApeironWord` by `proof/regen.sh`; the
representation relation and the error bound are built here under the closed axiom cone.

## Representation

An `i64` `v` denotes the real number `toReal v = v.val / 2^32`, the signed Q32.32 value (defined in
`Numeric.Engine.Exp`). `rsqrt_window x` is the documented admissible domain `[2^-20, 2^20]`, i.e.
`ONE >> 20 <= x <= ONE << 20`. The refusal correspondence is `fixed_rsqrt_refuses`: outside
`rsqrt_window` the extracted body returns `Err OutOfRange`.

## Statement choice

The window carries the combined bound

  `|toReal v - 1 / sqrt (toReal x)| <= 2^-20 · (1 / sqrt (toReal x)) + 4 / 2^32`,

mirroring the exp module's combined form (`Numeric.ExpRefinement`). The pure relative target alone is
not the right shape here: `toReal x` is the radicand of the real reciprocal square root and the output
`1 / sqrt (toReal x)` ranges over `[2^-10, 2^10]` on the window, so at the high end (`x` near `2^20`)
the output is smallest and the Q32.32 grid is largest relative to it. The `4 / 2^32` absolute floor
covers the two integer rounding steps at that edge, and the relative term carries the `2^-20` target
the Rust unit test asserts. `fixed_rsqrt_refines` states it.

## Bound structure

`word_fixed_rsqrt_unfold` reduces the success branch to the `do` block through `s = isqrt (x << 64)`
and `out = ((1 << 80) + s/2) / s`. The root and the quotient are then bounded by the pure lemmas
`isqrt_real_bounds`, `isqrt_recip_error_bound` (multiplied to `isqrt_scaled_error`) and
`round_nearest_div_error`; their shared premise is the integer-square-root characterization
`s*s <= m < (s+1)*(s+1)`. `sqrt_shift` gives the Q32.32 scaling
`sqrt (x * 2^64) = sqrt x * 2^32`, so `2^80 / sqrt (x << 64)` is `2^48 / sqrt x`. The remaining
obligation is the characterization of the extracted `fixed_rsqrt.isqrt` loop by that premise, then
composition of the two error terms into `fixed_rsqrt_refines`.
-/
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Numeric.Scale

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Representation relation -/

/-- The documented admissible window `[2^-20·2^32, 2^20·2^32] = [ONE >> 20, ONE << 20]`, as a
predicate on the signed Q32.32 value. -/
def rsqrt_window (x : I64) : Prop := (4096 : ℤ) ≤ x.val ∧ x.val ≤ 4503599627370496

/-! ## Refusal and refinement statements -/

/-- Refusal correspondence: outside the window the extracted body returns `Err OutOfRange`. -/
def fixed_rsqrt_refuses : Prop :=
  ∀ x : I64,
    ¬ rsqrt_window x →
      word_fixed_rsqrt x = ok (core.result.Result.Err fixed_rsqrt.RsqrtError.OutOfRange)

/-- The target refinement bound on the window: success, with the combined absolute/relative error
bound `2^-20 · (1 / sqrt (toReal x)) + 4 / 2^32`. -/
def fixed_rsqrt_refines : Prop :=
  ∀ x : I64,
    rsqrt_window x →
      ∃ v : I64,
        word_fixed_rsqrt x = ok (core.result.Result.Ok v) ∧
          |toReal v - 1 / Real.sqrt (toReal x)| ≤
            epsExp * (1 / Real.sqrt (toReal x)) + 4 / q32

/-! ## Refusal: structural lemmas and endpoint evaluation

The extracted body opens with `let lower ← LOWER; if x < lower then Err else …`, with the mirrored
upper test in the second branch. `word_fixed_rsqrt_refuses_of_lt`/`_of_gt` are proved from that
structure alone, parameterised by the values of the extracted endpoint `const`s: when the local
comparison holds, the body returns `Err OutOfRange`.

The specialised Aeneas shift specifications evaluate the extracted `const`s: `ONE = 1 <<< 32 = 2^32`,
`LOWER = ONE >>> 20 = 4096 = 2^12`, `UPPER = ONE <<< 20 = 2^52 = 4503599627370496`.
`fixed_rsqrt_refuses_proved` instantiates the parameterised lemmas at those values and discharges
`fixed_rsqrt_refuses` over the complement of `rsqrt_window`.
-/

/-- Below the evaluated lower endpoint the extracted body refuses. -/
theorem word_fixed_rsqrt_refuses_of_lt (x c : I64)
    (hL : fixed_rsqrt.fixed_rsqrt.LOWER = ok c) (h : x < c) :
    word_fixed_rsqrt x = ok (core.result.Result.Err fixed_rsqrt.RsqrtError.OutOfRange) := by
  rw [word_fixed_rsqrt, fixed_rsqrt.fixed_rsqrt, hL]
  simp only [bind_tc_ok, if_pos h]

/-- Above the evaluated upper endpoint the extracted body refuses. -/
theorem word_fixed_rsqrt_refuses_of_gt (x cL cU : I64)
    (hL : fixed_rsqrt.fixed_rsqrt.LOWER = ok cL) (hU : fixed_rsqrt.fixed_rsqrt.UPPER = ok cU)
    (hn : ¬ x < cL) (h : x > cU) :
    word_fixed_rsqrt x = ok (core.result.Result.Err fixed_rsqrt.RsqrtError.OutOfRange) := by
  rw [word_fixed_rsqrt, fixed_rsqrt.fixed_rsqrt, hL, hU]
  simp only [bind_tc_ok, if_neg hn, if_pos h]

/-! ## Endpoint evaluation and instantiation -/

/-- The extracted scale constant `ONE` evaluates to `2^32`. -/
theorem fixed_rsqrt_ONE_eq : fixed_rsqrt.ONE = ok (4294967296#i64) := by
  have h : fixed_rsqrt.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold fixed_rsqrt.ONE
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted lower endpoint evaluates to `ONE >>> 20 = 2^12`. -/
theorem fixed_rsqrt_LOWER_eq : fixed_rsqrt.fixed_rsqrt.LOWER = ok (4096#i64) := by
  have h : fixed_rsqrt.fixed_rsqrt.LOWER ⦃ z => z = (4096#i64) ⦄ := by
    unfold fixed_rsqrt.fixed_rsqrt.LOWER
    rw [fixed_rsqrt_ONE_eq]
    simp only [bind_tc_ok]
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftRight_eq_div_pow]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted upper endpoint evaluates to `ONE <<< 20 = 2^52`. -/
theorem fixed_rsqrt_UPPER_eq : fixed_rsqrt.fixed_rsqrt.UPPER = ok (4503599627370496#i64) := by
  have h : fixed_rsqrt.fixed_rsqrt.UPPER ⦃ z => z = (4503599627370496#i64) ⦄ := by
    unfold fixed_rsqrt.fixed_rsqrt.UPPER
    rw [fixed_rsqrt_ONE_eq]
    simp only [bind_tc_ok]
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- Refusal correspondence: outside `rsqrt_window` the extracted body returns `Err OutOfRange`. -/
theorem fixed_rsqrt_refuses_proved : fixed_rsqrt_refuses := by
  intro x hx
  rw [rsqrt_window] at hx
  by_cases hlow : x.val < (4096 : ℤ)
  · exact word_fixed_rsqrt_refuses_of_lt x _ fixed_rsqrt_LOWER_eq (by simpa using hlow)
  · have hlowle : (4096 : ℤ) ≤ x.val := le_of_not_gt hlow
    have hhigh : (4503599627370496 : ℤ) < x.val := by
      by_contra hc
      exact hx ⟨hlowle, le_of_not_gt hc⟩
    refine word_fixed_rsqrt_refuses_of_gt x _ _ fixed_rsqrt_LOWER_eq fixed_rsqrt_UPPER_eq ?_ ?_
    · simp only [IScalar.lt_equiv]
      exact not_lt.mpr hlowle
    · simpa using hhigh

/-! ## Success-branch unfold -/

/-- On the window the extracted body passes both refusal tests and reduces structurally to the
arithmetic block: the scaled conversion, the shift by 64, `isqrt`, the biased quotient, the
`i64::MAX` guard and the final cast. -/
theorem word_fixed_rsqrt_unfold (x : I64) (hx : rsqrt_window x) :
    word_fixed_rsqrt x =
      (do
        let i2 ← lift (core.convert.num.FromI128I64.from x)
        let scaled ← i2 <<< 64#i32
        let s ← fixed_rsqrt.isqrt scaled
        let i3 ← 1#i128 <<< 80#i32
        let i4 ← s / 2#i128
        let num ← i3 + i4
        let out ← num / s
        let i5 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if out > i5 then ok (core.result.Result.Err fixed_rsqrt.RsqrtError.OutOfRange)
        else
          let i6 ← lift (IScalar.cast .I64 out)
          ok (core.result.Result.Ok i6)) := by
  rw [rsqrt_window] at hx
  rw [word_fixed_rsqrt, fixed_rsqrt.fixed_rsqrt, fixed_rsqrt_LOWER_eq, fixed_rsqrt_UPPER_eq]
  have hlt : ¬ x < (4096#i64) := by
    rw [IScalar.lt_equiv]
    norm_num
    omega
  have hgt : ¬ x > (4503599627370496#i64) := by
    show ¬ ((4503599627370496#i64 : I64) < x)
    rw [IScalar.lt_equiv]
    norm_num
    omega
  simp only [bind_tc_ok, if_neg hlt, if_neg hgt]

/-! ## Root and quotient bounds

These are the pure arithmetic consequences the refinement needs, stated with their exact premise:
the integer-square-root characterization `s*s <= m < (s+1)*(s+1)` for the root, and floor division
with the half-unit bias for the quotient. The remaining obligation is to establish that premise for
the extracted `fixed_rsqrt.isqrt` loop. -/

/-- Real consequence of the integer floor square root: `s <= sqrt m < s+1`. -/
theorem isqrt_real_bounds {m s : ℕ} (hs : 0 < s)
    (hlo : s * s ≤ m) (hhi : m < (s + 1) * (s + 1)) :
    (s : ℝ) ≤ Real.sqrt m ∧ Real.sqrt m < (s : ℝ) + 1 := by
  constructor
  · have hsq : ((s : ℝ)) ^ 2 ≤ (m : ℝ) := by
      have h := (Nat.cast_le (α := ℝ)).mpr hlo
      push_cast at h
      nlinarith
    calc (s : ℝ) = Real.sqrt ((s : ℝ) ^ 2) := (Real.sqrt_sq (by positivity)).symm
      _ ≤ Real.sqrt m := Real.sqrt_le_sqrt hsq
  · have h : (m : ℝ) < ((s : ℝ) + 1) ^ 2 := by
      have h' := (Nat.cast_lt (α := ℝ)).mpr hhi
      push_cast at h'
      nlinarith
    rw [Real.sqrt_lt' (by positivity)]
    exact h

/-- Relative root-rounding error: `1/s` approximates `1/sqrt m` to within `1/(s·sqrt m)`, i.e. below
`1/s` relative; at the window's smallest root (`s >= 2^38`) this is below `2^-38`. -/
theorem isqrt_recip_error_bound {m s : ℕ} (hm : 0 < m) (hs : 0 < s)
    (hlo : s * s ≤ m) (hhi : m < (s + 1) * (s + 1)) :
    |1 / (s : ℝ) - 1 / Real.sqrt m| ≤ 1 / ((s : ℝ) * Real.sqrt m) := by
  obtain ⟨hloR, hhiR⟩ := isqrt_real_bounds hs hlo hhi
  have hsR : (0 : ℝ) < s := by exact_mod_cast hs
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hsqrt_pos : 0 < Real.sqrt m := Real.sqrt_pos.mpr hmR
  have hle : 1 / Real.sqrt m ≤ 1 / (s : ℝ) := one_div_le_one_div_of_le hsR hloR
  rw [abs_of_nonneg (sub_nonneg.mpr hle)]
  rw [div_sub_div (1 : ℝ) (1 : ℝ) (ne_of_gt hsR) (ne_of_gt hsqrt_pos)]
  apply div_le_div_of_nonneg_right _ (by positivity)
  linarith

/-- The scaled form of `isqrt_recip_error_bound`: `2^80/s` approximates `2^80/sqrt m` to within
`2^80/(s·sqrt m)`. With `m = x.val · 2^64` and `sqrt_shift`, the target is `2^48 / sqrt x.val`. -/
theorem isqrt_scaled_error {m s : ℕ} (hm : 0 < m) (hs : 0 < s)
    (hlo : s * s ≤ m) (hhi : m < (s + 1) * (s + 1)) :
    |(2 : ℝ) ^ (80 : ℕ) / s - (2 : ℝ) ^ (80 : ℕ) / Real.sqrt m| ≤
      (2 : ℝ) ^ (80 : ℕ) / ((s : ℝ) * Real.sqrt m) := by
  have h := isqrt_recip_error_bound hm hs hlo hhi
  have h2 : (0 : ℝ) ≤ (2 : ℝ) ^ (80 : ℕ) := by positivity
  have hsR : (0 : ℝ) < s := by exact_mod_cast hs
  have hmR : (0 : ℝ) < m := by exact_mod_cast hm
  have hsqrt_pos : 0 < Real.sqrt m := Real.sqrt_pos.mpr hmR
  have hkey : (2 : ℝ) ^ (80 : ℕ) / s - (2 : ℝ) ^ (80 : ℕ) / Real.sqrt m
      = (2 : ℝ) ^ (80 : ℕ) * (1 / (s : ℝ) - 1 / Real.sqrt m) := by
    field_simp
  rw [hkey, abs_mul, abs_of_nonneg h2]
  calc (2 : ℝ) ^ (80 : ℕ) * |1 / (s : ℝ) - 1 / Real.sqrt m|
      ≤ (2 : ℝ) ^ (80 : ℕ) * (1 / ((s : ℝ) * Real.sqrt m)) :=
        mul_le_mul_of_nonneg_left h h2
    _ = (2 : ℝ) ^ (80 : ℕ) / ((s : ℝ) * Real.sqrt m) := by rw [mul_one_div]

/-- Round-to-nearest division error: the half-unit-biased quotient `out = (n + d/2) / d` differs
from `n / d` by at most `1/2` in absolute value. This is the quotient step of the body, with
`n = 2^80 + s/2` and `d = s`. -/
theorem round_nearest_div_error {n d out : ℤ} (hd : 0 < d)
    (hout : out = (n + d / 2) / d) :
    |(out : ℝ) - (n : ℝ) / d| ≤ 1 / 2 := by
  have hdne : d ≠ 0 := ne_of_gt hd
  set q := n / d with hq
  set r := n % d with hr
  have hr0 : 0 ≤ r := by rw [hr]; exact Int.emod_nonneg n hdne
  have hrlt : r < d := by rw [hr]; exact Int.emod_lt_of_pos n hd
  have hn : n = q * d + r := by
    rw [hq, hr]
    have := Int.mul_ediv_add_emod n d
    linarith
  set t := (r + d / 2) / d with ht
  have hdpar : 2 * (d / 2) + d % 2 = d := Int.mul_ediv_add_emod d 2
  have hmod2lt : d % 2 < 2 := Int.emod_lt_of_pos d (by norm_num)
  have hmod2ge : 0 ≤ d % 2 := Int.emod_nonneg d (by norm_num)
  have hhalf_lt : d / 2 < d := by omega
  have ht0 : 0 ≤ t := by
    rw [ht]; exact Int.ediv_nonneg (by omega) (le_of_lt hd)
  have ht2 : t < 2 := by
    rw [ht, Int.ediv_lt_iff_lt_mul hd]
    omega
  have ht01 : t = 0 ∨ t = 1 := by omega
  have houtq : out = q + t := by
    rw [hout, ht, hn]
    have hcomm : q * d + r + d / 2 = (r + d / 2) + d * q := by ring
    rw [hcomm, Int.add_mul_ediv_left (r + d / 2) (b := d) q hdne]
    ring
  have hnR : (n : ℝ) / d = (q : ℝ) + (r : ℝ) / d := by
    rw [hn]; push_cast; field_simp
  have hdR : (0 : ℝ) < (d : ℝ) := by exact_mod_cast hd
  rw [houtq, hnR]
  push_cast
  have hsimp : (q : ℝ) + (t : ℝ) - ((q : ℝ) + (r : ℝ) / d) = (t : ℝ) - (r : ℝ) / d := by ring
  rw [hsimp]
  rcases ht01 with ht0' | ht1'
  · rw [ht0']; push_cast
    have h2r_d : 2 * r ≤ d := by
      have hlt : r + d / 2 < d := by
        by_contra hc
        have hle : d ≤ r + d / 2 := le_of_not_gt hc
        have h1 : (1 : ℤ) ≤ t := by
          rw [ht, ← Int.ediv_self hdne]
          exact Int.ediv_le_ediv hd hle
        omega
      omega
    have hge : 0 ≤ (r : ℝ) / d := div_nonneg (by exact_mod_cast hr0) (le_of_lt hdR)
    rw [zero_sub, abs_neg, abs_of_nonneg hge]
    rw [div_le_iff₀ hdR]
    have hcast : (2 : ℝ) * (r : ℝ) ≤ (d : ℝ) := by exact_mod_cast h2r_d
    nlinarith
  · rw [ht1']; push_cast
    have hd_2r : d ≤ 2 * r := by
      have hge : d ≤ r + d / 2 := by
        have hnotlt : ¬ t < 1 := by omega
        rw [ht, Int.ediv_lt_iff_lt_mul hd] at hnotlt
        omega
      omega
    have hrge : (1 : ℝ) / 2 ≤ (r : ℝ) / d := by
      rw [le_div_iff₀ hdR]
      have hcast : (d : ℝ) ≤ 2 * (r : ℝ) := by exact_mod_cast hd_2r
      nlinarith
    have hrle : (r : ℝ) / d ≤ 1 := by
      rw [div_le_one hdR]
      exact_mod_cast le_of_lt hrlt
    rw [abs_of_nonneg (by linarith)]
    linarith

/-- The Q32.32 scaling of the radicand: `sqrt (m · 2^64) = sqrt m · 2^32`, so the extracted
`2^80 / sqrt (x.val << 64)` denotes `2^48 / sqrt x.val`. -/
theorem sqrt_shift (m : ℕ) :
    Real.sqrt ((m : ℝ) * 2 ^ (64 : ℕ)) = Real.sqrt m * 2 ^ (32 : ℕ) := by
  rw [Real.sqrt_mul (by positivity)]
  congr 1
  rw [show (2 : ℝ) ^ (64 : ℕ) = ((2 : ℝ) ^ (32 : ℕ)) ^ 2 by ring]
  exact Real.sqrt_sq (by positivity)

/-! ## The extracted restoring square root: integer invariant

The extracted `fixed_rsqrt.isqrt` is the bit-by-bit restoring square root over `I128`
(Warren, *Hacker's Delight*, 2nd ed., §11-1). `isqrt_loop0` reduces the power-of-four seed
`1 <<< 126 = 4^63` while it exceeds the radicand; `isqrt_loop1` then extracts one root bit per
iteration. At the state `(rem, res, bit)` with `bit = 4^k`, `res` is the partial root scaled by
`4·bit` and `rem` the remainder, coupled by `res^2 = 4·bit·(n - rem)`; the one-sided remainder
bound `rem < 2·res + 4·bit` is what makes the final step round to the integer square root.

The lemmas below are the arithmetic core of that loop, stated over `ℤ` on the `.val` projections of
the extracted state, so the pending monadic bridge can consume them directly:

* `isqrt_inv_init` builds the invariant at the seed handed to `isqrt_loop1` by `isqrt_loop0`;
* `isqrt_step_hit`/`isqrt_step_miss` preserve it across one iteration for `bit = 4c`, `c > 0`
  (`res = 2a`); the extracted branch condition is `res + bit ≤ rem` for the hit branch;
* `isqrt_final_hit`/`isqrt_final_miss` discharge the `bit = 1` iteration, where the two branch
  results `a+1` and `a` are the floor `√n` with its two-sided characterization;
* `isqrt_inv_final` projects the terminal `bit = 0` state to that characterization.

**Landed.** The monadic lift `isqrt_loop1_spec` (below) is the `loop.spec_decr_nat` proof on
`fixed_rsqrt.isqrt_loop1` with measure `bit.val.toNat`, invariant `IsqrtInv` extended with
`IsPow4 bit ∧ bit ≤ 2^116`, and the four step/final lemmas above; `2 ∣ res` comes from the scale
relation and `4 ∣ bit` from `IsPow4` at `1 < bit`.

**Separately landed below.** The `isqrt_loop0` seed reduction (`isqrt_loop0_spec`) and
`isqrt_spec` composing it with `isqrt_loop1_spec`; the integer structural evaluation of the success
block (`word_fixed_rsqrt_success`); the window positivity / `1 / sqrt` scale helpers; and the
real-error composition closing `fixed_rsqrt_refines` (`fixed_rsqrt_refines_proved`), with
`fixed_rsqrt_refuses_proved` on the complement.
-/

/-- `bit` is a power of four, or zero (the loop exit value). -/
def IsPow4 (b : ℤ) : Prop := b = 0 ∨ ∃ k : ℕ, b = 4 ^ k

/-- State invariant of the extracted restoring square root. `rem` is the remainder, `res` the
partial root scaled by `4·bit`, coupled by `res^2 = 4·bit·(n - rem)`; `rem < 2·res + 4·bit` is the
one-sided remainder bound. At `bit = 0` the state is final and `res` is the integer square root. -/
structure IsqrtInv (n rem res bit : ℤ) : Prop where
  rem_nonneg : 0 ≤ rem
  res_nonneg : 0 ≤ res
  bit_nonneg : 0 ≤ bit
  rem_le : rem ≤ n
  scale : 0 < bit → res * res = 4 * bit * (n - rem)
  rem_bound : 0 < bit → rem < 2 * res + 4 * bit
  final : bit = 0 → res * res ≤ n ∧ n < (res + 1) * (res + 1)

/-- Initial state of `isqrt_loop1` after `isqrt_loop0` has reduced the power-of-four seed: `res = 0`,
`rem = n`, `bit > 0` with `n < 4·bit`. -/
theorem isqrt_inv_init {n bit : ℤ} (hn : 0 ≤ n) (hbit : 0 < bit) (hlt : n < 4 * bit) :
    IsqrtInv n n 0 bit :=
  ⟨hn, le_rfl, le_of_lt hbit, le_rfl, fun _ => by ring, fun _ => by linarith,
    fun h => absurd h (ne_of_gt hbit)⟩

/-- Extraction from the terminal state: the loop's `done res` at `bit = 0` is the integer square
root of `n`. -/
theorem isqrt_inv_final {n rem res : ℤ} (h : IsqrtInv n rem res 0) :
    res * res ≤ n ∧ n < (res + 1) * (res + 1) := h.final rfl

/-- Non-final step, hit branch: `res = 2a ≥ 0`, `bit = 4c` with `c > 0`. The new state
`(rem - (2a+4c), a+4c, c)` preserves the scaling relation and the remainder bound. -/
theorem isqrt_step_hit {n rem a c : ℤ} (ha : 0 ≤ a) (hc : 0 < c) (hremle : rem ≤ n)
    (hrel : a * a = 4 * c * (n - rem))
    (hbound : rem < 4 * a + 16 * c)
    (hle : 2 * a + 4 * c ≤ rem) :
    IsqrtInv n (rem - (2 * a + 4 * c)) (a + 4 * c) c :=
  ⟨by linarith, by linarith, by linarith, by linarith,
    fun _ => by nlinarith [hrel],
    fun _ => by nlinarith [hbound],
    fun hc0 => absurd hc0 (ne_of_gt hc)⟩

/-- Non-final step, miss branch: `res = 2a ≥ 0`, `bit = 4c` with `c > 0`. The new state
`(rem, a, c)` preserves the scaling relation and the remainder bound. -/
theorem isqrt_step_miss {n rem a c : ℤ} (ha : 0 ≤ a) (hc : 0 < c) (hremle : rem ≤ n)
    (hrem0 : 0 ≤ rem)
    (hrel : a * a = 4 * c * (n - rem))
    (hbound : rem < 2 * a + 4 * c) :
    IsqrtInv n rem a c :=
  ⟨hrem0, ha, by linarith, hremle, fun _ => hrel,
    fun _ => by linarith, fun hc0 => absurd hc0 (ne_of_gt hc)⟩

/-- Final step, hit branch: `bit = 1`, `res = 2a`, `a^2 = n - rem`. The result `a+1` is the integer
square root. -/
theorem isqrt_final_hit {n rem a : ℤ}
    (hrel : a * a = n - rem) (hbound : rem < 4 * a + 4) (hle : 2 * a + 1 ≤ rem) :
    (a + 1) * (a + 1) ≤ n ∧ n < (a + 1 + 1) * (a + 1 + 1) :=
  ⟨by nlinarith, by nlinarith⟩

/-- Final step, miss branch: `bit = 1`, `res = 2a`, `a^2 = n - rem`, `rem < 2a+1`. The result `a`
is the integer square root. -/
theorem isqrt_final_miss {n rem a : ℤ}
    (hrel : a * a = n - rem) (hmiss : rem < 2 * a + 1) (hrem0 : 0 ≤ rem) :
    a * a ≤ n ∧ n < (a + 1) * (a + 1) :=
  ⟨by nlinarith, by nlinarith⟩

/-! The terminal `bit = 1` states, wrapped as `IsqrtInv` at `bit = 0`, so the monadic body case can
land the loop's continuation post directly. -/

/-- `IsPow4` is preserved by the closing `bit >>> 2` (`bit / 4`). -/
theorem IsPow4.div_four {b : ℤ} (h : IsPow4 b) : IsPow4 (b / 4) := by
  rcases h with hzero | ⟨k, hk⟩
  · exact Or.inl (by rw [hzero]; norm_num)
  · rw [hk]
    cases k with
    | zero => exact Or.inl (by norm_num)
    | succ j =>
        right
        refine ⟨j, ?_⟩
        rw [pow_succ]
        exact Int.mul_ediv_cancel _ (by norm_num)

/-- A power of four greater than one is divisible by four. -/
theorem pow4_dvd_four {b : ℤ} (h : IsPow4 b) (hb : 1 < b) : (4 : ℤ) ∣ b := by
  rcases h with hzero | ⟨k, hk⟩
  · omega
  · rw [hk]
    cases k with
    | zero => rw [hk] at hb; norm_num at hb
    | succ j => exact ⟨4 ^ j, by rw [pow_succ]; ring⟩

/-- Final hit branch packaged as the terminal loop invariant: `res = 2a`, `bit = 1`, hit condition
`2a+1 ≤ rem`; the new state `(rem - (2a+1), a+1, 0)` satisfies `IsqrtInv`. -/
theorem isqrt_final_hit_inv {n rem a : ℤ} (ha : 0 ≤ a) (hremle : rem ≤ n)
    (hrel : a * a = n - rem) (hbound : rem < 4 * a + 4) (hle : 2 * a + 1 ≤ rem) :
    IsqrtInv n (rem - (2 * a + 1)) (a + 1) 0 :=
  ⟨by linarith, by linarith, le_rfl, by linarith,
    fun h => absurd h (by norm_num), fun h => absurd h (by norm_num),
    fun _ => isqrt_final_hit hrel hbound hle⟩

/-- Final miss branch packaged as the terminal loop invariant: `res = 2a`, `bit = 1`, miss condition
`rem < 2a+1`; the new state `(rem, a, 0)` satisfies `IsqrtInv`. -/
theorem isqrt_final_miss_inv {n rem a : ℤ} (hrem0 : 0 ≤ rem) (ha : 0 ≤ a) (hremle : rem ≤ n)
    (hrel : a * a = n - rem) (hmiss : rem < 2 * a + 1) :
    IsqrtInv n rem a 0 :=
  ⟨hrem0, ha, le_rfl, hremle, fun h => absurd h (by norm_num),
    fun h => absurd h (by norm_num), fun _ => isqrt_final_miss hrel hmiss hrem0⟩

/-! ## Monadic lift: the extracted `isqrt_loop1`

`loop.spec_decr_nat` on the extracted loop with measure `bit.val.toNat` and the invariant above,
extended with `IsPow4 bit` and `bit ≤ 2^116`. The body case steps through `res + bit`, the
`rem ≥ step` split, `rem - step` / `res >>> 1` / `i + bit`, `bit >>> 2`, then splits on
`b.val = 1` (terminal, discharging `isqrt_final_hit_inv`/`isqrt_final_miss_inv`) versus
`1 < b.val` (step, discharging `isqrt_step_hit`/`isqrt_step_miss`). `2 ∣ res` comes from the scale
relation, `4 ∣ b.val` from `IsPow4` at `1 < b.val`. -/
theorem isqrt_loop1_spec (nz : ℤ) (rem res bit : I128)
    (hnzle : nz ≤ (2 : ℤ) ^ (116 : ℕ))
    (hinv : IsqrtInv nz rem.val res.val bit.val)
    (hpow : IsPow4 bit.val) (hble : bit.val ≤ (2 : ℤ) ^ (116 : ℕ)) :
    fixed_rsqrt.isqrt_loop1 rem res bit ⦃ out =>
      out.val * out.val ≤ nz ∧ nz < (out.val + 1) * (out.val + 1) ⦄ := by
  unfold fixed_rsqrt.isqrt_loop1
  apply loop.spec_decr_nat
    (measure := fun st : I128 × I128 × I128 => st.2.2.val.toNat)
    (inv := fun st : I128 × I128 × I128 =>
      IsqrtInv nz st.1.val st.2.1.val st.2.2.val ∧
      IsPow4 st.2.2.val ∧ st.2.2.val ≤ (2 : ℤ) ^ (116 : ℕ))
    (post := fun out : I128 =>
      out.val * out.val ≤ nz ∧ nz < (out.val + 1) * (out.val + 1))
  · rintro ⟨r, s, b⟩ ⟨hinv', hpow', hble'⟩
    dsimp only
    unfold fixed_rsqrt.isqrt_loop1.body
    by_cases hb : b = 0#i128
    · subst hb
      simp only [bne_self_eq_false, Bool.false_eq_true, if_false, WP.spec_ok]
      exact hinv'.final (by norm_num)
    · have hbb : (b != 0#i128) = true := by simpa using hb
      rw [if_pos hbb]
      have hbpos : 0 < b.val := by
        rcases lt_or_eq_of_le hinv'.bit_nonneg with h | h
        · exact h
        · exfalso
          apply hb
          apply IScalar.eq_of_val_eq
          rw [← h]
          norm_num
      have hs0 : 0 ≤ s.val := hinv'.res_nonneg
      have hr0 : 0 ≤ r.val := hinv'.rem_nonneg
      have hrle : r.val ≤ nz := hinv'.rem_le
      have hscale := hinv'.scale hbpos
      have hrb := hinv'.rem_bound hbpos
      have hres_even : (2 : ℤ) ∣ s.val := by
        have h2 : (2 : ℤ) ∣ s.val * s.val := by
          rw [hscale]; exact ⟨2 * b.val * (nz - r.val), by ring⟩
        exact Int.prime_two.dvd_of_dvd_pow (n := 2) (by simpa [sq] using h2)
      obtain ⟨a, ha⟩ : ∃ a : ℤ, s.val = 2 * a := hres_even
      have ha0 : 0 ≤ a := by nlinarith only [hs0, ha]
      have hs2 : s.val / 2 = a := by rw [ha, Int.mul_ediv_cancel_left]; norm_num
      have hs_le : s.val ≤ (2 : ℤ) ^ (117 : ℕ) := by
        have hnr : 0 ≤ nz - r.val := by linarith
        have hprod : b.val * (nz - r.val) ≤ (2 : ℤ) ^ (116 : ℕ) * (2 : ℤ) ^ (116 : ℕ) :=
          mul_le_mul hble' (by linarith only [hnzle, hr0]) hnr (by positivity)
        have hsq : s.val * s.val ≤ 4 * ((2 : ℤ) ^ (116 : ℕ) * (2 : ℤ) ^ (116 : ℕ)) := by
          rw [hscale]; nlinarith only [hprod]
        have hK : 4 * ((2 : ℤ) ^ (116 : ℕ) * (2 : ℤ) ^ (116 : ℕ)) = ((2 : ℤ) ^ (117 : ℕ)) ^ 2 := by
          norm_num
        rw [hK] at hsq
        nlinarith only [hs0, hsq, sq_nonneg (s.val - (2 : ℤ) ^ (117 : ℕ))]
      have hmin : I128.min ≤ s.val + b.val := by
        norm_num [I128.min, I128.numBits, IScalar.min]
        linarith only [hs0, hinv'.bit_nonneg]
      have hmax : s.val + b.val ≤ I128.max := by
        norm_num [I128.max, I128.numBits, IScalar.max]
        nlinarith only [hs_le, hble']
      obtain ⟨step, hstep, hstepval⟩ :=
        WP.spec_imp_exists (I128.add_spec (x := s) (y := b) hmin hmax)
      rw [hstep]; simp only [bind_tc_ok]
      by_cases hge : step ≤ r
      · rw [if_pos hge]
        have hgeval : step.val ≤ r.val := (IScalar.le_equiv step r).mp hge
        obtain ⟨rem2, hrem2, hrem2val⟩ := WP.spec_imp_exists
          (I128.sub_spec (x := r) (y := step)
            (by norm_num [I128.min, I128.numBits, IScalar.min]; linarith only [hgeval])
            (by norm_num [I128.max, I128.numBits, IScalar.max]; linarith only [hrle, hnzle, hgeval, hstepval, hs0, hinv'.bit_nonneg]))
        rw [hrem2]; simp only [bind_tc_ok]
        obtain ⟨i, hi, hival, _⟩ := WP.spec_imp_exists
          (I128.ShiftRight_IScalar_spec (x := s) (y := 1#i32)
            (by norm_num) (by norm_num [I128.numBits]))
        rw [hi]; simp only [bind_tc_ok]
        have hival' : i.val = s.val / 2 := by
          have h1 : (1#i32).toNat = 1 := by decide
          rw [hival, h1, Int.shiftRight_eq_div_pow]; norm_num
        obtain ⟨res2, hres2, hres2val⟩ := WP.spec_imp_exists
          (I128.add_spec (x := i) (y := b)
            (by norm_num [I128.min, I128.numBits, IScalar.min]; linarith [hival', hs0, hinv'.bit_nonneg])
            (by
              norm_num [I128.max, I128.numBits, IScalar.max]
              have h2 : i.val ≤ (2 : ℤ) ^ (117 : ℕ) := by
                rw [hival']; exact le_trans (Int.ediv_le_self 2 hs0) hs_le
              nlinarith only [h2, hble']))
        rw [hres2]; simp only [bind_tc_ok]
        obtain ⟨bit1, hbit1, hbit1val, _⟩ := WP.spec_imp_exists
          (I128.ShiftRight_IScalar_spec (x := b) (y := 2#i32)
            (by norm_num) (by norm_num [I128.numBits]))
        rw [hbit1]; simp only [bind_tc_ok]
        refine (WP.spec_ok (ControlFlow.cont (rem2, res2, bit1))).mpr ?_
        dsimp only
        have hbit1val' : bit1.val = b.val / 4 := by
          have h2 : (2#i32).toNat = 2 := by decide
          rw [hbit1val, h2, Int.shiftRight_eq_div_pow]; norm_num
        have hstepval' : step.val = 2 * a + b.val := by rw [hstepval, ha]
        have hrem2val' : rem2.val = r.val - (2 * a + b.val) := by rw [hrem2val, hstepval']
        have hres2val' : res2.val = a + b.val := by rw [hres2val, hival', hs2]
        constructor
        · rcases eq_or_lt_of_le (show (1 : ℤ) ≤ b.val from hbpos) with hb1 | hbgt
          · have hrela : a * a = nz - r.val := by
              have := hscale; rw [ha, ← hb1] at this; nlinarith only [this]
            have hrbda : r.val < 4 * a + 4 := by
              have := hrb; rw [ha, ← hb1] at this; nlinarith only [this]
            have hlea : 2 * a + 1 ≤ r.val := by
              have := hgeval; rw [hstepval', ← hb1] at this; nlinarith only [this]
            have hnew : IsqrtInv nz (r.val - (2 * a + 1)) (a + 1) 0 :=
              isqrt_final_hit_inv ha0 hrle hrela hrbda hlea
            have hz : bit1.val = 0 := by rw [hbit1val', ← hb1]; norm_num
            have hresz : res2.val = a + 1 := by rw [hres2val', ← hb1]
            have hremz : rem2.val = r.val - (2 * a + 1) := by rw [hrem2val', ← hb1]
            rw [hz, hresz, hremz]; exact ⟨hnew, Or.inl rfl, by norm_num⟩
          · obtain ⟨c, hc_eq⟩ : ∃ c : ℤ, b.val = 4 * c := pow4_dvd_four hpow' hbgt
            have hcpos : 0 < c := by nlinarith only [hbgt, hc_eq]
            have hcdiv : b.val / 4 = c := by rw [hc_eq, Int.mul_ediv_cancel_left]; norm_num
            have hrelc : a * a = 4 * c * (nz - r.val) := by
              have := hscale; rw [ha, hc_eq] at this; nlinarith only [this]
            have hboundc : r.val < 4 * a + 16 * c := by
              have := hrb; rw [ha, hc_eq] at this; nlinarith only [this]
            have hlec : 2 * a + 4 * c ≤ r.val := by
              have := hgeval; rw [hstepval', hc_eq] at this; nlinarith only [this]
            have hnew : IsqrtInv nz (r.val - (2 * a + 4 * c)) (a + 4 * c) c :=
              isqrt_step_hit ha0 hcpos hrle hrelc hboundc hlec
            have hbc : bit1.val = c := by rw [hbit1val', hcdiv]
            have hresc : res2.val = a + 4 * c := by rw [hres2val', hc_eq]
            have hremc : rem2.val = r.val - (2 * a + 4 * c) := by rw [hrem2val', hc_eq]
            rw [hbc, hresc, hremc]; exact ⟨hnew, by rw [← hcdiv]; exact IsPow4.div_four hpow', by rw [← hcdiv]; exact le_trans (Int.ediv_le_self 4 (le_of_lt hbpos)) hble'⟩
        · rw [Int.toNat_lt_toNat hbpos]
          rcases eq_or_lt_of_le (show (1 : ℤ) ≤ b.val from hbpos) with hb1 | hbgt
          · rw [hbit1val', ← hb1]; norm_num
          · obtain ⟨c, hc_eq⟩ : ∃ c : ℤ, b.val = 4 * c := pow4_dvd_four hpow' hbgt
            have hcdiv : b.val / 4 = c := by rw [hc_eq, Int.mul_ediv_cancel_left]; norm_num
            rw [hbit1val', hcdiv]
            have : c < b.val := by rw [hc_eq]; nlinarith only [hbgt, hc_eq]
            exact this
      · rw [if_neg hge]
        have hltval : r.val < step.val := by
          by_contra hc
          exact hge ((IScalar.le_equiv step r).mpr (le_of_not_gt hc))
        obtain ⟨i, hi, hival, _⟩ := WP.spec_imp_exists
          (I128.ShiftRight_IScalar_spec (x := s) (y := 1#i32)
            (by norm_num) (by norm_num [I128.numBits]))
        rw [hi]; simp only [bind_tc_ok]
        have hival' : i.val = s.val / 2 := by
          have h1 : (1#i32).toNat = 1 := by decide
          rw [hival, h1, Int.shiftRight_eq_div_pow]; norm_num
        obtain ⟨bit1, hbit1, hbit1val, _⟩ := WP.spec_imp_exists
          (I128.ShiftRight_IScalar_spec (x := b) (y := 2#i32)
            (by norm_num) (by norm_num [I128.numBits]))
        rw [hbit1]; simp only [bind_tc_ok]
        refine (WP.spec_ok (ControlFlow.cont (r, i, bit1))).mpr ?_
        dsimp only
        have hbit1val' : bit1.val = b.val / 4 := by
          have h2 : (2#i32).toNat = 2 := by decide
          rw [hbit1val, h2, Int.shiftRight_eq_div_pow]; norm_num
        have hstepval' : step.val = 2 * a + b.val := by rw [hstepval, ha]
        have hres2val' : i.val = a := by rw [hival', hs2]
        constructor
        · rcases eq_or_lt_of_le (show (1 : ℤ) ≤ b.val from hbpos) with hb1 | hbgt
          · have hrela : a * a = nz - r.val := by
              have := hscale; rw [ha, ← hb1] at this; nlinarith only [this]
            have hmissa : r.val < 2 * a + 1 := by
              have := hltval; rw [hstepval', ← hb1] at this; nlinarith only [this]
            have hnew : IsqrtInv nz r.val a 0 :=
              isqrt_final_miss_inv hr0 ha0 hrle hrela hmissa
            have hz : bit1.val = 0 := by rw [hbit1val', ← hb1]; norm_num
            rw [hz, hres2val']; exact ⟨hnew, Or.inl rfl, by norm_num⟩
          · obtain ⟨c, hc_eq⟩ : ∃ c : ℤ, b.val = 4 * c := pow4_dvd_four hpow' hbgt
            have hcpos : 0 < c := by nlinarith only [hbgt, hc_eq]
            have hcdiv : b.val / 4 = c := by rw [hc_eq, Int.mul_ediv_cancel_left]; norm_num
            have hrelc : a * a = 4 * c * (nz - r.val) := by
              have := hscale; rw [ha, hc_eq] at this; nlinarith only [this]
            have hboundc : r.val < 2 * a + 4 * c := by
              have := hltval; rw [hstepval', hc_eq] at this; nlinarith only [this]
            have hnew : IsqrtInv nz r.val a c :=
              isqrt_step_miss ha0 hcpos hrle hr0 hrelc hboundc
            have hbc : bit1.val = c := by rw [hbit1val', hcdiv]
            rw [hbc, hres2val']; exact ⟨hnew, by rw [← hcdiv]; exact IsPow4.div_four hpow', by rw [← hcdiv]; exact le_trans (Int.ediv_le_self 4 (le_of_lt hbpos)) hble'⟩
        · rw [Int.toNat_lt_toNat hbpos]
          rcases eq_or_lt_of_le (show (1 : ℤ) ≤ b.val from hbpos) with hb1 | hbgt
          · rw [hbit1val', ← hb1]; norm_num
          · obtain ⟨c, hc_eq⟩ : ∃ c : ℤ, b.val = 4 * c := pow4_dvd_four hpow' hbgt
            have hcdiv : b.val / 4 = c := by rw [hc_eq, Int.mul_ediv_cancel_left]; norm_num
            rw [hbit1val', hcdiv]
            have : c < b.val := by rw [hc_eq]; nlinarith only [hbgt, hc_eq]
            exact this
  · exact ⟨hinv, hpow, hble⟩

/-! ## Seed reduction and `isqrt` spec

`isqrt_loop0_spec` is the `loop.spec_decr_nat` seed reduction (measure `bit.val.toNat`, invariant
`IsPow4 bit ∧ 0 < bit ∧ n < 4·bit`); `1 ≤ n` keeps `bit = 1` as a stop rather than a shift to zero,
the step uses `n < bit = 4·(bit/4)` and `bit/4 < bit`. `isqrt_spec` evaluates the `1 <<< 126 = 4^63`
seed, feeds `isqrt_loop0_spec`, and hands the reduced `bit` plus `isqrt_inv_init` to
`isqrt_loop1_spec`. -/

theorem isqrt_loop0_spec (n bit : I128)
    (hn : 1 ≤ n.val)
    (hpow : IsPow4 bit.val) (hpos : 0 < bit.val) (hlt : n.val < 4 * bit.val) :
    fixed_rsqrt.isqrt_loop0 n bit ⦃ out =>
      IsPow4 out.val ∧ 0 < out.val ∧ out.val ≤ n.val ∧ n.val < 4 * out.val ⦄ := by
  unfold fixed_rsqrt.isqrt_loop0
  apply loop.spec_decr_nat
    (measure := fun b : I128 => b.val.toNat)
    (inv := fun b : I128 => IsPow4 b.val ∧ 0 < b.val ∧ n.val < 4 * b.val)
    (post := fun out : I128 =>
      IsPow4 out.val ∧ 0 < out.val ∧ out.val ≤ n.val ∧ n.val < 4 * out.val)
  · rintro b ⟨hpow', hpos', hlt'⟩
    unfold fixed_rsqrt.isqrt_loop0.body
    by_cases hgt : b > n
    · rw [if_pos hgt]
      have hgtval : n.val < b.val := (IScalar.lt_equiv n b).mp hgt
      have hb1 : (1 : ℤ) < b.val := by omega
      obtain ⟨c, hc_eq⟩ : ∃ c : ℤ, b.val = 4 * c := pow4_dvd_four hpow' hb1
      have hcpos : 0 < c := by nlinarith [hb1, hc_eq]
      have hcdiv : b.val / 4 = c := by rw [hc_eq, Int.mul_ediv_cancel_left]; norm_num
      obtain ⟨bit1, hbit1, hbit1val, _⟩ := WP.spec_imp_exists
        (I128.ShiftRight_IScalar_spec (x := b) (y := 2#i32)
          (by norm_num) (by norm_num [I128.numBits]))
      rw [hbit1]; simp only [bind_tc_ok]
      refine (WP.spec_ok (ControlFlow.cont bit1)).mpr ?_
      dsimp only
      have hbit1val' : bit1.val = b.val / 4 := by
        have h2 : (2#i32).toNat = 2 := by decide
        rw [hbit1val, h2, Int.shiftRight_eq_div_pow]; norm_num
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · rw [hbit1val']; exact IsPow4.div_four hpow'
      · rw [hbit1val', hcdiv]; exact hcpos
      · rw [hbit1val', hcdiv]; rw [← hc_eq]; exact hgtval
      · rw [hbit1val', hcdiv, Int.toNat_lt_toNat hpos']; rw [hc_eq]; nlinarith [hcpos]
    · rw [if_neg hgt]
      have hle : b.val ≤ n.val := (IScalar.le_equiv b n).mp (le_of_not_gt hgt)
      exact ⟨hpow', hpos', hle, hlt'⟩
  · exact ⟨hpow, hpos, hlt⟩

theorem isqrt_spec (n : I128) (hn : 1 ≤ n.val) (hnle : n.val ≤ (2 : ℤ) ^ (116 : ℕ)) :
    fixed_rsqrt.isqrt n ⦃ s =>
      s.val * s.val ≤ n.val ∧ n.val < (s.val + 1) * (s.val + 1) ⦄ := by
  unfold fixed_rsqrt.isqrt
  obtain ⟨bit0, hbit0, hbit0val, _⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := 1#i128) (y := 126#i32)
      (by norm_num) (by norm_num [I128.numBits]))
  rw [hbit0]; simp only [bind_tc_ok]
  have hbit0val' : bit0.val = (2 : ℤ) ^ (126 : ℕ) := by
    rw [hbit0val]
    have h1 : (1#i128).val = 1 := rfl
    have h126 : (126#i32).toNat = 126 := by decide
    rw [h1, h126, Int.shiftLeft_eq]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.bmod]
  have hseed_pow : IsPow4 bit0.val := by
    rw [hbit0val']; exact Or.inr ⟨63, by norm_num⟩
  have hseed_pos : 0 < bit0.val := by rw [hbit0val']; norm_num
  have hseed_lt : n.val < 4 * bit0.val := by
    rw [hbit0val']; nlinarith [hnle]
  obtain ⟨bit1, hbit1, hpow1, hpos1, hle1, hlt1⟩ :=
    WP.spec_imp_exists (isqrt_loop0_spec n bit0 hn hseed_pow hseed_pos hseed_lt)
  rw [hbit1]; simp only [bind_tc_ok]
  exact isqrt_loop1_spec n.val n 0#i128 bit1 (by linarith [hle1, hnle])
    (isqrt_inv_init (by linarith [hn]) hpos1 hlt1) hpow1 (by linarith [hle1, hnle])

/-! ## Window positivity and the `1 / sqrt` scale identity

Helpers for the pending `fixed_rsqrt_refines` composition: `toReal` is positive on the window, and
`1 / sqrt (toReal x)` is `(2^48 / sqrt x.val) / 2^32`, the Q32.32 denominator the root and quotient
bounds are stated against. -/

/-- On the window the Q32.32 value is positive. -/
theorem toReal_pos {x : I64} (hx : rsqrt_window x) : 0 < toReal x := by
  rw [rsqrt_window] at hx
  unfold toReal
  exact div_pos (by exact_mod_cast (by linarith [hx.1] : (0 : ℤ) < x.val)) q32_pos

/-- The Q32.32 half-scale identity: `sqrt (v / 2^32) = sqrt v / 2^16` for `v ≥ 0`. -/
theorem sqrt_toReal (x : I64) (hx : 0 ≤ x.val) :
    Real.sqrt (toReal x) = Real.sqrt (x.val : ℝ) / 2 ^ (16 : ℕ) := by
  unfold toReal q32
  rw [Real.sqrt_div (by positivity)]
  congr 1
  rw [show (2 : ℝ) ^ (32 : ℕ) = ((2 : ℝ) ^ (16 : ℕ)) ^ 2 by norm_num]
  exact Real.sqrt_sq (by positivity)

/-- The reciprocal root as the Q32.32-scaled target: `1 / sqrt (toReal x)` is
`(2^48 / sqrt x.val) / 2^32`, the value `isqrt_scaled_error` and `sqrt_shift` are stated against. -/
theorem one_div_sqrt_toReal (x : I64) (hx : 0 < x.val) :
    1 / Real.sqrt (toReal x) =
      ((2 : ℝ) ^ (48 : ℕ) / Real.sqrt (x.val : ℝ)) / (2 : ℝ) ^ (32 : ℕ) := by
  rw [sqrt_toReal x (le_of_lt hx)]
  have hs : Real.sqrt (x.val : ℝ) ≠ 0 := by positivity
  field_simp

/-! ## Structural evaluation of the `rsqrt` success block

`word_fixed_rsqrt_success` walks `word_fixed_rsqrt_unfold`'s residual block step by step and
characterises the returned value: the scaled radicand `scaled.val = x.val · 2^64`; the root from
`isqrt_spec` with `2^38 ≤ s.val ≤ 2^58`; `i3.val = 2^80`; `i4.val = s.val / 2`;
`num.val = 2^80 + s.val / 2`; `out.val = (2^80 + s.val / 2) / s.val`; the guard false since
`out.val ≤ 2^61 < I64.MAX`; and the final cast preserving the value. The remaining obligation is the
real-error composition against these values. -/

theorem word_fixed_rsqrt_success (x : I64) (hx : rsqrt_window x) :
    ∃ v : I64,
      word_fixed_rsqrt x = ok (core.result.Result.Ok v) ∧
      ∃ s : I128,
        s.val * s.val ≤ x.val * (2 : ℤ) ^ (64 : ℕ) ∧
        x.val * (2 : ℤ) ^ (64 : ℕ) < (s.val + 1) * (s.val + 1) ∧
        (2 : ℤ) ^ (38 : ℕ) ≤ s.val ∧ s.val ≤ (2 : ℤ) ^ (58 : ℕ) ∧
        v.val = ((2 : ℤ) ^ (80 : ℕ) + s.val / 2) / s.val ∧
        0 ≤ v.val ∧ v.val ≤ (2 : ℤ) ^ (61 : ℕ) := by
  rw [word_fixed_rsqrt_unfold x hx]
  rw [rsqrt_window] at hx
  obtain ⟨hxlo, hxhi⟩ := hx
  have hm1 : (1 : ℤ) ≤ x.val := by linarith
  have hxs : x.val ≤ (2 : ℤ) ^ (52 : ℕ) := by
    have : (4503599627370496 : ℤ) = (2 : ℤ) ^ (52 : ℕ) := by norm_num
    linarith [hxhi, this]
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  have hI64min : IScalar.min IScalarTy.I64 = -(2 : ℤ) ^ (63 : ℕ) := by
    norm_num [IScalar.min, I64.min, I64.numBits]
  have hI64max : IScalar.max IScalarTy.I64 = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    norm_num [IScalar.max, I64.max, I64.numBits]
  have hMAX : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
  -- i2 = FromI128I64.from x
  rw [show Aeneas.Std.lift (core.convert.num.FromI128I64.from x)
      = ok (core.convert.num.FromI128I64.from x) from rfl]
  rw [bind_tc_ok]
  -- scaled
  obtain ⟨scaled, hscaled, hscaledval, _⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := core.convert.num.FromI128I64.from x) (y := 64#i32)
      (by norm_num) (by norm_num [I128.numBits]))
  rw [hscaled]; simp only [bind_tc_ok]
  have hscaledval' : scaled.val = x.val * (2 : ℤ) ^ (64 : ℕ) := by
    rw [hscaledval]
    rw [show (64#i32).toNat = 64 from by decide]
    rw [core.convert.num.FromI128I64.from_val_eq x]
    rw [Int.shiftLeft_eq]
    rw [show I128.size = 2 ^ (128 : ℕ) from by norm_num [I128.size, IScalar.size, I128.numBits]]
    refine Int.bmod_eq_of_le ?_ ?_
    · norm_num
      nlinarith [hm1]
    · norm_num
      nlinarith [hxs]
  have hscaled_lo : (2 : ℤ) ^ (76 : ℕ) ≤ scaled.val := by
    rw [hscaledval']
    have : (4096 : ℤ) * (2 : ℤ) ^ (64 : ℕ) = (2 : ℤ) ^ (76 : ℕ) := by norm_num
    nlinarith [hxlo, this]
  have hscaled_hi : scaled.val ≤ (2 : ℤ) ^ (116 : ℕ) := by
    rw [hscaledval']
    have : (2 : ℤ) ^ (52 : ℕ) * (2 : ℤ) ^ (64 : ℕ) = (2 : ℤ) ^ (116 : ℕ) := by norm_num
    nlinarith [hxs, this]
  -- s = isqrt scaled
  obtain ⟨s, hs, hspost⟩ := WP.spec_imp_exists
    (isqrt_spec scaled (by linarith [hscaled_lo]) hscaled_hi)
  rw [hs]; simp only [bind_tc_ok]
  have hsq_lt : s.val * s.val < (s.val + 1) * (s.val + 1) := lt_of_le_of_lt hspost.1 hspost.2
  have hs0 : 0 ≤ s.val := by
    have h2 : -(1 : ℤ) < 2 * s.val := by nlinarith [hsq_lt]
    omega
  have hs1 : 1 ≤ s.val := by
    by_contra hc
    have hz : s.val ≤ 0 := by omega
    have : s.val = 0 := by omega
    rw [this] at hspost
    norm_num at hspost
    linarith [hscaled_lo, hspost.2]
  have hsne : s.val ≠ 0 := by omega
  have hs_lo : (2 : ℤ) ^ (38 : ℕ) ≤ s.val := by
    have hbig : ((2 : ℤ) ^ (38 : ℕ)) * ((2 : ℤ) ^ (38 : ℕ)) < (s.val + 1) * (s.val + 1) := by
      have : (2 : ℤ) ^ (76 : ℕ) = ((2 : ℤ) ^ (38 : ℕ)) * ((2 : ℤ) ^ (38 : ℕ)) := by norm_num
      nlinarith [hscaled_lo, hspost.2, this]
    have hpos1 : 0 ≤ s.val + 1 := by linarith
    have h38 : (2 : ℤ) ^ (38 : ℕ) < s.val + 1 := by
      nlinarith [hbig, hpos1, sq_nonneg (s.val + 1 - (2 : ℤ) ^ (38 : ℕ))]
    omega
  have hs20 : (2 : ℤ) ^ (20 : ℕ) ≤ s.val := by
    have : (2 : ℤ) ^ (20 : ℕ) ≤ (2 : ℤ) ^ (38 : ℕ) := by norm_num
    linarith
  have hs_hi : s.val ≤ (2 : ℤ) ^ (58 : ℕ) := by
    have hprod : s.val * s.val ≤ (2 : ℤ) ^ (58 : ℕ) * (2 : ℤ) ^ (58 : ℕ) := by
      have h116 : (2 : ℤ) ^ (116 : ℕ) = (2 : ℤ) ^ (58 : ℕ) * (2 : ℤ) ^ (58 : ℕ) := by norm_num
      nlinarith [hspost.1, hscaled_hi, h116]
    nlinarith [hs0, hprod, sq_nonneg (s.val - (2 : ℤ) ^ (58 : ℕ))]
  -- i3 = 1 <<< 80
  obtain ⟨i3, hi3, hi3val, _⟩ := WP.spec_imp_exists
    (I128.ShiftLeft_IScalar_spec (x := 1#i128) (y := 80#i32)
      (by norm_num) (by norm_num [I128.numBits]))
  rw [hi3]; simp only [bind_tc_ok]
  have hi3val' : i3.val = (2 : ℤ) ^ (80 : ℕ) := by
    rw [hi3val]
    rw [show (80#i32).toNat = 80 from by decide]
    rw [show (1#i128).val = 1 from rfl]
    rw [Int.shiftLeft_eq]
    rw [show I128.size = 2 ^ (128 : ℕ) from by norm_num [I128.size, IScalar.size, I128.numBits]]
    refine Int.bmod_eq_of_le ?_ ?_
    · norm_num
    · norm_num
  -- i4 = s / 2
  obtain ⟨i4, hi4, hi4val⟩ := WP.spec_imp_exists
    (I128.div_spec (x := s) (y := 2#i128) (by norm_num) (by
      rintro ⟨_, h2⟩; norm_num at h2))
  rw [hi4]; simp only [bind_tc_ok]
  have hi4val' : i4.val = s.val / 2 := by
    rw [hi4val, Int.tdiv_eq_ediv_of_nonneg hs0]
    norm_num
  -- num = i3 + i4
  obtain ⟨num, hnum, hnumval⟩ := WP.spec_imp_exists
    (I128.add_spec (x := i3) (y := i4)
      (by rw [hI128min]
          have h4nn : 0 ≤ s.val / 2 := Int.ediv_nonneg hs0 (by norm_num)
          linarith [hi3val', hi4val', h4nn])
      (by rw [hI128max]
          have h4 := Int.ediv_le_self 2 hs0
          linarith [hi3val', hi4val', hs_hi, h4]))
  rw [hnum]; simp only [bind_tc_ok]
  have hnumval' : num.val = (2 : ℤ) ^ (80 : ℕ) + s.val / 2 := by
    rw [hnumval, hi3val', hi4val']
  have hnum_nonneg : 0 ≤ num.val := by
    rw [hnumval']
    have h4nn : 0 ≤ s.val / 2 := Int.ediv_nonneg hs0 (by norm_num)
    nlinarith
  -- out = num / s
  obtain ⟨out, hout, houtval⟩ := WP.spec_imp_exists
    (I128.div_spec (x := num) (y := s) hsne (by
      intro hcon
      exact absurd hcon.2 (by omega)))
  rw [hout]; simp only [bind_tc_ok]
  have houtval' : out.val = num.val.tdiv s.val := houtval
  have houtval'' : out.val = ((2 : ℤ) ^ (80 : ℕ) + s.val / 2) / s.val := by
    have hnum_nn : 0 ≤ (2 : ℤ) ^ (80 : ℕ) + s.val / 2 := by
      have h4nn : 0 ≤ s.val / 2 := Int.ediv_nonneg hs0 (by norm_num)
      positivity
    rw [houtval', hnumval']
    exact Int.tdiv_eq_ediv_of_nonneg hnum_nn
  have hout_lo : 0 ≤ out.val := by
    rw [houtval'']
    exact Int.ediv_nonneg (by positivity) (by omega)
  have hout_hi : out.val ≤ (2 : ℤ) ^ (61 : ℕ) := by
    have hmul : out.val * s.val ≤ (2 : ℤ) ^ (80 : ℕ) + s.val / 2 := by
      rw [houtval'']
      exact Int.ediv_mul_le _ hsne
    have hstep : out.val * (2 : ℤ) ^ (20 : ℕ) ≤ (2 : ℤ) ^ (81 : ℕ) := by
      have h1 : out.val * (2 : ℤ) ^ (20 : ℕ) ≤ out.val * s.val :=
        Int.mul_le_mul_of_nonneg_left hs20 hout_lo
      have h2 : s.val / 2 ≤ (2 : ℤ) ^ (58 : ℕ) :=
        le_trans (Int.ediv_le_self 2 hs0) hs_hi
      have h3 : (2 : ℤ) ^ (80 : ℕ) + (2 : ℤ) ^ (58 : ℕ) ≤ (2 : ℤ) ^ (81 : ℕ) := by norm_num
      omega
    have hle : (2 : ℤ) ^ (20 : ℕ) * out.val ≤ (2 : ℤ) ^ (20 : ℕ) * (2 : ℤ) ^ (61 : ℕ) := by
      calc (2 : ℤ) ^ (20 : ℕ) * out.val = out.val * (2 : ℤ) ^ (20 : ℕ) := by rw [Int.mul_comm]
        _ ≤ (2 : ℤ) ^ (81 : ℕ) := hstep
        _ = (2 : ℤ) ^ (20 : ℕ) * (2 : ℤ) ^ (61 : ℕ) := by norm_num
    exact Int.le_of_mul_le_mul_left hle (by positivity)
  -- i5 = cast .I128 I64.MAX
  obtain ⟨i5, hi5, hi5val⟩ := WP.spec_imp_exists
    (IScalar.cast_inBounds_spec .I128 core.num.I64.MAX (by
      rw [hMAX]
      constructor
      · norm_num [I128.min, IScalar.min, I128.numBits]
      · norm_num [I128.max, IScalar.max, I128.numBits]))
  rw [hi5]; simp only [bind_tc_ok]
  have hi5val' : i5.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [hi5val, hMAX]
  -- guard false
  have hguard : ¬ out > i5 := by
    intro hgt
    have hg : i5.val < out.val := (IScalar.lt_equiv i5 out).mp hgt
    rw [hi5val'] at hg
    linarith [hout_hi]
  rw [if_neg hguard]
  -- i6 = cast .I64 out
  obtain ⟨i6, hi6, hi6val⟩ := WP.spec_imp_exists
    (IScalar.cast_inBounds_spec .I64 out (by
      constructor
      · rw [hI64min]; linarith [hout_lo]
      · rw [hI64max]; linarith [hout_hi]))
  rw [hi6]; simp only [bind_tc_ok]
  refine ⟨i6, rfl, s, ?_, ?_, hs_lo, hs_hi, ?_, ?_, ?_⟩
  · rw [← hscaledval']; exact hspost.1
  · rw [← hscaledval']; exact hspost.2
  · rw [hi6val, houtval'']
  · rw [hi6val]; exact hout_lo
  · rw [hi6val]; exact hout_hi

/-! ## Real-error composition: `fixed_rsqrt_refines`

`isqrt_scaled_real` transports `isqrt_scaled_error` to integer radicand and root. The closure
`fixed_rsqrt_refines_proved` takes `word_fixed_rsqrt_success`'s `v`, `s` and composes
`round_nearest_div_error` (`n = 2^80`, `d = s.val`), `isqrt_scaled_error` + `sqrt_shift` and
`one_div_sqrt_toReal`: the root term is `T / s.val ≤ epsExp · T` with `T = 2^48 / √ x.val` and
`s.val ≥ 2^20`, the rounding term is `≤ 1/2`, so `|v.val − T| ≤ epsExp · T + 4`, which divides by
`2^32` to the stated Q32.32 bound. `fixed_rsqrt_refuses_proved` discharges the complement. -/

theorem isqrt_scaled_real {m s : ℤ} (hm : 0 < m) (hs : 0 < s)
    (hlo : s * s ≤ m) (hhi : m < (s + 1) * (s + 1)) :
    |(2 : ℝ) ^ (80 : ℕ) / (s : ℝ) - (2 : ℝ) ^ (80 : ℕ) / Real.sqrt (m : ℝ)| ≤
      (2 : ℝ) ^ (80 : ℕ) / ((s : ℝ) * Real.sqrt (m : ℝ)) := by
  have hm0 : 0 ≤ m := le_of_lt hm
  have hs0 : 0 ≤ s := le_of_lt hs
  have hmn : ((m.toNat : ℕ) : ℝ) = (m : ℝ) := by
    have h := Int.toNat_of_nonneg hm0
    exact_mod_cast h
  have hsn : ((s.toNat : ℕ) : ℝ) = (s : ℝ) := by
    have h := Int.toNat_of_nonneg hs0
    exact_mod_cast h
  have hlo' : s.toNat * s.toNat ≤ m.toNat := by
    have h : ((s.toNat : ℤ)) * (s.toNat : ℤ) ≤ (m.toNat : ℤ) := by
      rw [Int.toNat_of_nonneg hs0, Int.toNat_of_nonneg hm0]; exact hlo
    exact_mod_cast h
  have hhi' : m.toNat < (s.toNat + 1) * (s.toNat + 1) := by
    have h : (m.toNat : ℤ) < ((s.toNat : ℤ) + 1) * ((s.toNat : ℤ) + 1) := by
      rw [Int.toNat_of_nonneg hm0, Int.toNat_of_nonneg hs0]
      exact hhi
    exact_mod_cast h
  have h := isqrt_scaled_error (m := m.toNat) (s := s.toNat)
    (by omega) (by omega) hlo' hhi'
  simpa [hmn, hsn] using h

theorem fixed_rsqrt_refines_proved : fixed_rsqrt_refines := by
  intro x hxw
  obtain ⟨v, hv, s, hslo, hshi, hs_lo, hs_hi, hvval, hv0, hvle⟩ :=
    word_fixed_rsqrt_success x hxw
  rw [rsqrt_window] at hxw
  obtain ⟨hxlo, hxhi⟩ := hxw
  have hmpos : (0 : ℤ) < x.val := by linarith
  refine ⟨v, hv, ?_⟩
  have hspos : (0 : ℤ) < s.val := by linarith [hs_lo]
  set T : ℝ := (2 : ℝ) ^ (48 : ℕ) / Real.sqrt (x.val : ℝ) with hTdef
  have hTnn : 0 ≤ T := by rw [hTdef]; positivity
  -- the Q32.32 scaling identity for the radicand
  have hsroot : Real.sqrt ((x.val : ℝ) * 2 ^ (64 : ℕ)) = Real.sqrt (x.val : ℝ) * 2 ^ (32 : ℕ) := by
    have h := sqrt_shift x.val.toNat
    have hxcast : ((x.val.toNat : ℕ) : ℝ) = (x.val : ℝ) := by
      have := Int.toNat_of_nonneg (le_of_lt hmpos)
      exact_mod_cast this
    rwa [hxcast] at h
  have hTid : (2 : ℝ) ^ (80 : ℕ) / Real.sqrt (((x.val * 2 ^ (64 : ℕ) : ℤ)) : ℝ) = T := by
    have hcast : (((x.val * 2 ^ (64 : ℕ) : ℤ)) : ℝ) = (x.val : ℝ) * 2 ^ (64 : ℕ) := by
      push_cast; ring
    rw [hcast, hsroot, hTdef]
    have hx : (0 : ℝ) < Real.sqrt (x.val : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast hmpos)
    field_simp
  -- rounding error
  have hround : |(v.val : ℝ) - (2 : ℝ) ^ (80 : ℕ) / (s.val : ℝ)| ≤ 1 / 2 := by
    have h := round_nearest_div_error (n := (2 : ℤ) ^ (80 : ℕ)) (d := s.val) (out := v.val)
      (by linarith [hs_lo]) hvval
    simpa only [Int.cast_pow, Int.cast_ofNat] using h
  -- root error
  have hroot : |(2 : ℝ) ^ (80 : ℕ) / (s.val : ℝ) - T| ≤
      (2 : ℝ) ^ (80 : ℕ) / ((s.val : ℝ) * Real.sqrt (((x.val * 2 ^ (64 : ℕ) : ℤ)) : ℝ)) := by
    have h := isqrt_scaled_real (m := x.val * 2 ^ (64 : ℕ)) (s := s.val)
      (by positivity) hspos hslo hshi
    rwa [hTid] at h
  have hrootT : |(2 : ℝ) ^ (80 : ℕ) / (s.val : ℝ) - T| ≤ T / (s.val : ℝ) := by
    refine le_trans hroot ?_
    have hcast : (((x.val * 2 ^ (64 : ℕ) : ℤ)) : ℝ) = (x.val : ℝ) * 2 ^ (64 : ℕ) := by
      push_cast; ring
    rw [hcast, hsroot, hTdef]
    have hx : (0 : ℝ) < Real.sqrt (x.val : ℝ) := Real.sqrt_pos.mpr (by exact_mod_cast hmpos)
    have hsR : (0 : ℝ) < (s.val : ℝ) := by exact_mod_cast hspos
    field_simp
    norm_num
  -- relative term small
  have hTs : T / (s.val : ℝ) ≤ epsExp * T := by
    have hsinv : 1 / (s.val : ℝ) ≤ epsExp := by
      have hs20 : ((2 : ℝ) ^ (20 : ℕ)) ≤ (s.val : ℝ) := by
        have h20 : (2 : ℤ) ^ (20 : ℕ) ≤ s.val := by
          have : (2 : ℤ) ^ (20 : ℕ) ≤ (2 : ℤ) ^ (38 : ℕ) := by norm_num
          linarith
        exact_mod_cast h20
      rw [epsExp, show ((2 : ℝ)) ^ (-20 : ℤ) = 1 / (2 : ℝ) ^ (20 : ℕ) by
        rw [zpow_neg, one_div]; rfl]
      exact one_div_le_one_div_of_le (by positivity) hs20
    have := mul_le_mul_of_nonneg_right hsinv hTnn
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using this
  -- combine
  have hval : |(v.val : ℝ) - T| ≤ epsExp * T + 4 := by
    have htr := abs_sub_le (v.val : ℝ) ((2 : ℝ) ^ (80 : ℕ) / (s.val : ℝ)) T
    have h1 : |(v.val : ℝ) - T| ≤ 1 / 2 + T / (s.val : ℝ) := by linarith [htr, hround, hrootT]
    have h2 : (1 : ℝ) / 2 + T / (s.val : ℝ) ≤ epsExp * T + 4 := by linarith [hTs]
    linarith [h1, h2]
  -- convert to the Q32.32 statement
  have hmain : |toReal v - 1 / Real.sqrt (toReal x)| = |(v.val : ℝ) - T| / 2 ^ (32 : ℕ) := by
    rw [one_div_sqrt_toReal x hmpos]
    unfold toReal q32
    rw [hTdef]
    rw [← sub_div, abs_div, abs_of_pos (by positivity : (0 : ℝ) < 2 ^ (32 : ℕ))]
  calc |toReal v - 1 / Real.sqrt (toReal x)| = |(v.val : ℝ) - T| / 2 ^ (32 : ℕ) := hmain
    _ ≤ (epsExp * T + 4) / 2 ^ (32 : ℕ) := div_le_div_of_nonneg_right hval (by positivity)
    _ = epsExp * (T / 2 ^ (32 : ℕ)) + 4 / q32 := by unfold q32; ring
    _ = epsExp * (1 / Real.sqrt (toReal x)) + 4 / q32 := by
          rw [hTdef, ← one_div_sqrt_toReal x hmpos]

end

end apeiron_word
