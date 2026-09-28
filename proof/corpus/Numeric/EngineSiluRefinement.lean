/-
Refinement of the extracted fixed-point `SiLU` kernel (`APEIRON-ALGEBRA-G3-SILU-REFINE`;
plan `docs/epics/model-algebra.md` W1-T3, gap G3).

The Rust body is `src/engine/silu.rs`: `i64` Q32.32. The public entry point is
`apeiron_word.word_silu_fixed`, the extracted wrapper over `silu.silu_fixed`, which returns the
three saturation/refusal arms

* `v >= 16 · ONE`  → `Ok v` (overflow end, `Ok v` exactly);
* `v <= -21 · ONE` → `Ok 0` (underflow end);
* `-21 · ONE < v < 16 · ONE` → the rounded quotient
  `round(|v| · 2^32 / (2^32 + fixed_exp (-v)))` with the sign of `v` reapplied.

with `ONE = 1 <<< 32 = 2^32`, `fixed_exp.fixed_exp` the extracted `exp` body (`src/numeric/fixed_exp.rs`)
and `value.div_round_nearest` the half-biased quotient of `src/engine/value.rs`.

This module lands the two saturation facts exactly (`word_silu_fixed_high`,
`word_silu_fixed_low`), evaluates the extracted `UPPER`/`LOWER`/`engine.ONE` constants, states the
in-window refinement (`silu_window_refines`, proved through `word_silu_fixed_unfold` and
`silu_tail_refines`) with its reference `siluReal`, and proves the pure rounding lemma
(`div_round_nearest_abs_bound`) its composition needs. The two saturation arms are bounded against
`siluReal` in `Numeric/SiluSaturation.lean` (`silu_refines_total`).
-/
import Numeric.ExpRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Evaluation of the extracted constants

`engine.ONE = 1 <<< FIXED_FRAC_BITS = 2^32`; the `SiLU` arms are `16 · ONE` and `-21 · ONE`. The
`silu.*` constants are separate from `fixed_exp.ONE` but evaluate to the same scale. -/

/-- The extracted `engine.ONE` evaluates to `2^32`. -/
theorem engine_ONE_eq : engine.ONE = ok (4294967296#i64) := by
  have h : engine.ONE ⦃ z => z = (4294967296#i64) ⦄ := by
    unfold engine.ONE
    rw [engine.FIXED_FRAC_BITS]
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I64.size, IScalar.size, I64.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted upper saturation endpoint `silu.silu_fixed.UPPER = 16 · 2^32`. -/
theorem silu_UPPER_eq : silu.silu_fixed.UPPER = ok (68719476736#i64) := by
  have h : silu.silu_fixed.UPPER ⦃ z => z = (68719476736#i64) ⦄ := by
    unfold silu.silu_fixed.UPPER
    rw [engine_ONE_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := 16#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted lower saturation endpoint `silu.silu_fixed.LOWER = -21 · 2^32`. -/
theorem silu_LOWER_eq : silu.silu_fixed.LOWER = ok ((-90194313216)#i64) := by
  have h : silu.silu_fixed.LOWER ⦃ z => z = ((-90194313216)#i64) ⦄ := by
    unfold silu.silu_fixed.LOWER
    rw [engine_ONE_eq]
    simp only [bind_tc_ok]
    apply Aeneas.Std.WP.spec_mono
      (I64.mul_spec (x := (-21)#i64) (y := 4294967296#i64)
        (by norm_num [I64.min, I64.max, I64.numBits])
        (by norm_num [I64.min, I64.max, I64.numBits]))
    intro z hz
    apply IScalar.eq_imp
    rw [hz]
    norm_num
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## Saturation branches (facts 1 and 2)

The extracted body tests `v >= UPPER` and `v <= LOWER` before any `exp` call; on either side it
returns the documented saturating constant. The parameterised lemmas are proved from the extracted
structure alone; the instantiated headlines are the literal `16 · 2^32` / `-21 · 2^32` statements. -/

/-- High saturation, from the extracted `UPPER` value: `v >= UPPER` returns `Ok v`. -/
theorem word_silu_fixed_high_of (v c : I64) (hU : silu.silu_fixed.UPPER = ok c)
    (h : c ≤ v) :
    word_silu_fixed v = ok (core.result.Result.Ok v) := by
  rw [word_silu_fixed, silu.silu_fixed, hU]
  simp only [bind_tc_ok, if_pos h]

/-- Low saturation, from the extracted `LOWER`/`UPPER` values: `v <= LOWER < UPPER` returns `Ok 0`. -/
theorem word_silu_fixed_low_of (v cL cU : I64) (hU : silu.silu_fixed.UPPER = ok cU)
    (hL : silu.silu_fixed.LOWER = ok cL) (hc : cL < cU) (h : v ≤ cL) :
    word_silu_fixed v = ok (core.result.Result.Ok 0#i64) := by
  rw [word_silu_fixed, silu.silu_fixed, hU, hL]
  have hn : ¬ (cU ≤ v) := not_le_of_gt (lt_of_le_of_lt h hc)
  simp only [bind_tc_ok, if_neg hn, if_pos h]

/-- **Fact 1.** `v >= 16 · ONE` returns `Ok v` exactly. -/
theorem word_silu_fixed_high (v : I64) (h : (68719476736 : ℤ) ≤ v.val) :
    word_silu_fixed v = ok (core.result.Result.Ok v) := by
  refine word_silu_fixed_high_of v _ silu_UPPER_eq ?_
  exact (IScalar.le_equiv _ _).2 h

/-- **Fact 2.** `v <= -21 · ONE` returns `Ok 0`. -/
theorem word_silu_fixed_low (v : I64) (h : v.val ≤ (-90194313216 : ℤ)) :
    word_silu_fixed v = ok (core.result.Result.Ok 0#i64) := by
  refine word_silu_fixed_low_of v _ _ silu_UPPER_eq silu_LOWER_eq ?_ ?_
  · exact (IScalar.lt_equiv _ _).2 (by norm_num)
  · exact (IScalar.le_equiv _ _).2 h

/-! ## Rounding lemma for the quotient step (fact 3 component)

`value.div_round_nearest num den = (num + den / 2) / den` on `i128`. For `den > 0` the half-biased
quotient is a round-to-nearest: after multiplication by `den` it is within `den / 2` of `num`. This
is the integer form consumed by the in-window composition. -/

/-- **Rounding bound.** For `d > 0`, `((n + d/2)/d)·d` is within `d/2` of `n`. -/
theorem div_round_nearest_abs_bound (n d : ℤ) (hd : 0 < d) :
    |((n + d / 2) / d) * d - n| ≤ d / 2 := by
  have hdne : d ≠ 0 := ne_of_gt hd
  set q : ℤ := (n + d / 2) / d with hq
  set s : ℤ := (n + d / 2) % d with hs
  set f : ℤ := d / 2 with hf
  set e : ℤ := d % 2 with he
  have hs0 : 0 ≤ s := by rw [hs]; exact Int.emod_nonneg _ hdne
  have hslt : s < d := by rw [hs]; exact Int.emod_lt_of_pos _ hd
  have he0 : 0 ≤ e := by rw [he]; exact Int.emod_nonneg _ (by norm_num)
  have he1 : e < 2 := by rw [he]; exact Int.emod_lt_of_pos _ (by norm_num)
  have h2f : 2 * f = d - e := by
    rw [hf, he]
    have := Int.mul_ediv_add_emod d 2
    linarith
  have hdecomp : q * d + s = n + f := by
    rw [hq, hs, hf]
    have := Int.mul_ediv_add_emod (n + d / 2) d
    linarith
  have hqdn : q * d - n = f - s := by linarith
  rw [hqdn, abs_le]
  constructor <;> omega

/-! ## Real-error composition for the in-window value

Given the extracted value relations — `den = 2^32 + e`, `q = ⌊(|v|·2^32 + den/2)/den⌋`, the
sign-reapplied `i7` — and the `exp` combined bound on `e`, the quotient is within
`epsExp·|toReal v| + (4/q32)·|toReal v| + 1/(2·q32)` of `siluReal (toReal v)`. -/

/-- The real sigmoid-linear unit `x ↦ x / (1 + exp (-x))`. -/
def siluReal (x : ℝ) : ℝ := x / (1 + Real.exp (-x))

theorem silu_bound_general (v e den : I64) (q : I128) (i7 : I64)
    (hden : den.val = 4294967296 + e.val) (hdenpos : 0 < den.val)
    (hepos : 0 ≤ toReal e)
    (hq : q.val = (|v.val| * 4294967296 + den.val / 2) / den.val)
    (hi7 : i7.val = (if v.val < 0 then (-1:ℤ) else 1) * q.val)
    (hb : |toReal e - Real.exp (-toReal v)| ≤ epsExp * Real.exp (-toReal v) + 4/q32) :
    |toReal i7 - siluReal (toReal v)| ≤
      epsExp * |toReal v| + (4 / q32) * |toReal v| + 1 / (2 * q32) := by
  set S : ℝ := (if v.val < 0 then (-1:ℤ) else 1) with hS
  have hSabs : |S| = 1 := by rw [hS]; split <;> norm_num
  have hX_sign : toReal v = S * |toReal v| := by
    rw [hS]
    split_ifs with hv
    · have hneg : toReal v < 0 := by
        unfold toReal
        exact div_neg_of_neg_of_pos (by exact_mod_cast hv) q32_pos
      rw [abs_of_neg hneg]; ring
    · have hnn : 0 ≤ toReal v := by
        unfold toReal
        exact div_nonneg (by exact_mod_cast (le_of_not_gt hv)) (le_of_lt q32_pos)
      rw [abs_of_nonneg hnn]; ring
  have hi7R : toReal i7 = S * ((q.val:ℝ)/q32) := by
    unfold toReal; rw [hi7, hS]; push_cast; ring
  have hsilu : siluReal (toReal v) = S * (|toReal v| / (1 + Real.exp (-toReal v))) := by
    unfold siluReal
    nth_rewrite 1 [hX_sign]
    rw [mul_div_assoc]
  have hmain : |toReal i7 - siluReal (toReal v)| =
      |(q.val:ℝ)/q32 - |toReal v| / (1 + Real.exp (-toReal v))| := by
    rw [hi7R, hsilu, ← mul_sub, abs_mul, hSabs, one_mul]
  rw [hmain]
  -- rounding contribution
  have hq_bound : |q.val * den.val - |v.val| * 4294967296| ≤ den.val / 2 := by
    rw [hq]; exact div_round_nearest_abs_bound _ _ hdenpos
  have hround_real : |(q.val:ℝ)/q32 - (|v.val|:ℝ)/den.val| ≤ 1/(2*q32) := by
    have hdenR : (0:ℝ) < (den.val:ℝ) := by exact_mod_cast hdenpos
    have hq32R : (4294967296:ℝ) = q32 := by norm_num [q32]
    have hkey : (q.val:ℝ)/q32 - (|v.val|:ℝ)/den.val
        = (((q.val * den.val - |v.val| * 4294967296 : ℤ)) : ℝ) / ((den.val:ℝ) * q32) := by
      push_cast
      rw [hq32R]
      field_simp
    rw [hkey, abs_div, abs_of_pos (mul_pos hdenR q32_pos), ← Int.cast_abs]
    have hcast : (((|q.val * den.val - |v.val| * 4294967296| : ℤ)) : ℝ) ≤ ((den.val / 2 : ℤ) : ℝ) := by
      exact_mod_cast hq_bound
    have hdenhalf : ((den.val / 2 : ℤ) : ℝ) ≤ (den.val:ℝ)/2 := by
      have hdecomp := Int.mul_ediv_add_emod den.val 2
      have h0 : 0 ≤ den.val % 2 := Int.emod_nonneg _ (by norm_num)
      have hlt : den.val % 2 < 2 := Int.emod_lt_of_pos _ (by norm_num)
      have hle : den.val / 2 * 2 ≤ den.val := by omega
      have h' : (((den.val / 2 * 2 : ℤ)) : ℝ) ≤ (den.val:ℝ) := by exact_mod_cast hle
      push_cast at h'
      linarith
    calc (((|q.val * den.val - |v.val| * 4294967296| : ℤ)) : ℝ) / ((den.val:ℝ) * q32)
        ≤ ((den.val / 2 : ℤ) : ℝ) / ((den.val:ℝ) * q32) :=
          div_le_div_of_nonneg_right hcast (le_of_lt (mul_pos hdenR q32_pos))
      _ ≤ ((den.val:ℝ)/2) / ((den.val:ℝ) * q32) :=
          div_le_div_of_nonneg_right hdenhalf (le_of_lt (mul_pos hdenR q32_pos))
      _ = 1/(2*q32) := by field_simp
  -- exp contribution
  have hE : (den.val:ℝ)/q32 = 1 + toReal e := by
    rw [hden]; push_cast; unfold toReal q32; field_simp; norm_num
  have hXabs : |toReal v| = (|v.val|:ℝ)/q32 := by
    unfold toReal; rw [abs_div, abs_of_pos q32_pos]
  have hfirst : (|v.val|:ℝ)/den.val = |toReal v| / (den.val/q32) := by
    rw [hXabs]
    field_simp [q32_ne_zero, ne_of_gt (by exact_mod_cast hdenpos : (0:ℝ) < (den.val:ℝ))]
  have hfirst' : (|v.val|:ℝ)/den.val = |toReal v| / (1 + toReal e) := by
    rw [hfirst, hE]
  have hexp_part : |(|v.val|:ℝ)/den.val - |toReal v| / (1 + Real.exp (-toReal v))| ≤
      epsExp * |toReal v| + (4/q32) * |toReal v| := by
    rw [hfirst']
    set s : ℝ := Real.exp (-toReal v) with hs
    set E : ℝ := toReal e with hEdef
    have hsp : (0:ℝ) < 1 + s := by rw [hs]; positivity
    have hEp : (0:ℝ) < 1 + E := by rw [hEdef]; linarith
    have hE1 : (1:ℝ) ≤ 1 + E := by rw [hEdef]; linarith
    have hs1 : (1:ℝ) ≤ 1 + s := by rw [hs]; have := Real.exp_pos (-toReal v); linarith
    have hXnn : (0:ℝ) ≤ |toReal v| := abs_nonneg _
    have hnum : |s - E| ≤ epsExp * s + 4/q32 := by
      rw [abs_sub_comm, hs, hEdef]; exact hb
    have heps0 : (0:ℝ) < epsExp := by unfold epsExp; exact zpow_pos (by norm_num) _
    have hdenom : (0:ℝ) < (1 + E) * (1 + s) := mul_pos hEp hsp
    have habs : |(|toReal v| * (s - E) / ((1 + E) * (1 + s)))| =
        |toReal v| * |s - E| / ((1 + E) * (1 + s)) := by
      rw [abs_div, abs_of_pos hdenom, abs_mul, abs_of_nonneg hXnn]
    have hrw : |toReal v| / (1 + E) - |toReal v| / (1 + s)
        = |toReal v| * (s - E) / ((1 + E) * (1 + s)) := by
      field_simp
      ring
    rw [hrw, habs]
    have h1 : |toReal v| * |s - E| ≤ |toReal v| * (epsExp * s + 4/q32) :=
      mul_le_mul_of_nonneg_left hnum hXnn
    have h2 : |toReal v| * |s - E| / ((1 + E) * (1 + s))
        ≤ |toReal v| * (epsExp * s + 4/q32) / ((1 + E) * (1 + s)) :=
      div_le_div_of_nonneg_right h1 (le_of_lt hdenom)
    have hinner : (0:ℝ) ≤ epsExp * s + 4/q32 := by
      apply add_nonneg
      · exact mul_nonneg (le_of_lt heps0) (le_of_lt (by rw [hs]; positivity))
      · exact div_nonneg (by norm_num) (le_of_lt q32_pos)
    have h3 : |toReal v| * (epsExp * s + 4/q32) / ((1 + E) * (1 + s))
        ≤ |toReal v| * (epsExp * s + 4/q32) / (1 + s) :=
      div_le_div_of_nonneg_left (mul_nonneg hXnn hinner) hsp (by nlinarith [hE1, hs1])
    have hs_le : s/(1+s) ≤ 1 := by rw [div_le_one hsp]; linarith
    have h1_le : (1:ℝ)/(1+s) ≤ 1 := by rw [div_le_one hsp]; linarith
    have h4 : |toReal v| * (epsExp * s + 4/q32) / (1 + s)
        = |toReal v| * (epsExp * (s/(1+s)) + (4/q32) * ((1:ℝ)/(1+s))) := by
      field_simp
    have h5 : epsExp * (s/(1+s)) + (4/q32) * ((1:ℝ)/(1+s)) ≤ epsExp + 4/q32 := by
      have h5a : epsExp * (s/(1+s)) ≤ epsExp * 1 :=
        mul_le_mul_of_nonneg_left hs_le (le_of_lt heps0)
      have h5b : (4/q32) * ((1:ℝ)/(1+s)) ≤ (4/q32) * 1 :=
        mul_le_mul_of_nonneg_left h1_le (div_nonneg (by norm_num) (le_of_lt q32_pos))
      linarith
    calc |toReal v| * |s - E| / ((1 + E) * (1 + s))
        ≤ |toReal v| * (epsExp * s + 4/q32) / ((1 + E) * (1 + s)) := h2
      _ ≤ |toReal v| * (epsExp * s + 4/q32) / (1 + s) := h3
      _ = |toReal v| * (epsExp * (s/(1+s)) + (4/q32) * ((1:ℝ)/(1+s))) := h4
      _ ≤ |toReal v| * (epsExp + 4/q32) := mul_le_mul_of_nonneg_left h5 hXnn
      _ = epsExp * |toReal v| + (4/q32) * |toReal v| := by ring
  calc |(q.val:ℝ)/q32 - |toReal v| / (1 + Real.exp (-toReal v))|
      ≤ |(q.val:ℝ)/q32 - (|v.val|:ℝ)/den.val|
        + |(|v.val|:ℝ)/den.val - |toReal v| / (1 + Real.exp (-toReal v))| :=
        abs_sub_le _ _ _
    _ ≤ 1/(2*q32) + (epsExp * |toReal v| + (4/q32) * |toReal v|) :=
        add_le_add hround_real hexp_part
    _ = epsExp * |toReal v| + (4 / q32) * |toReal v| + 1 / (2 * q32) := by ring

/-! ## In-window refinement (fact 3)

The reference is the real sigmoid-linear unit `siluReal x = x / (1 + exp (-x))`. On the open window
`-21 · ONE < v < 16 · ONE` the extracted body is on its exact path: `w = -v` lies strictly inside the
`fixed_exp` window `[-16 · ONE, 21 · ONE]`, so `fixed_exp_refines_proved` gives `e` with its combined
bound; the denominator `2^32 + e` is positive; `num = |v| · 2^32`; the half-biased quotient is bounded
by `div_round_nearest_abs_bound`; the sign of `v` is reapplied and the `I64` narrowing succeeds.

The bound below is the composition: the quotient rounding contributes `1/(2·2^32)`, and the `exp`
relative error `epsExp` scaled by the denominator contributes `epsExp · |toReal v|` plus the `4/2^32`
absolute floor of the `exp` module scaled by the window magnitude. -/

/-- The open exact-path window `(-21·2^32, 16·2^32)` of `silu_fixed`, as a predicate on the signed
value. -/
def silu_window (v : I64) : Prop :=
  (-90194313216 : ℤ) < v.val ∧ v.val < 68719476736

/-- **In-window structural reduction.** On `silu_window` the extracted body passes both saturation
tests and reduces to the concrete monadic arithmetic block: `w = -v`, the `exp` step, the checked
`2^32 + e` denominator, the sign-magnitude `i128` product `|v| · 2^32`, the half-biased quotient
`value.div_round_nearest`, the sign reapplication and the two narrowing guards. The remaining
obligation of `silu_window_refines` is the real-error composition against this block. -/
theorem word_silu_fixed_unfold (v : I64) (h : silu_window v) :
    word_silu_fixed v =
      (do
        let w ← -. v
        let r ← fixed_exp.fixed_exp w
        match r with
        | core.result.Result.Ok e => do
          let i2 ← engine.ONE
          let o ← lift (I64.checked_add i2 e)
          match o with
          | none => ok (core.result.Result.Err silu.SiluError.Overflow)
          | some den =>
            if den <= 0#i64 then ok (core.result.Result.Err silu.SiluError.ZeroDenominator)
            else do
              let mag ←
                if v < 0#i64 then do
                  let i3 ← lift (core.convert.num.FromI128I64.from v)
                  -. i3
                else ok (core.convert.num.FromI128I64.from v)
              let i3 ← lift (core.convert.num.FromI128I64.from i2)
              let o1 ← lift (I128.checked_mul mag i3)
              match o1 with
              | none => ok (core.result.Result.Err silu.SiluError.Overflow)
              | some num => do
                let i4 ← lift (core.convert.num.FromI128I64.from den)
                let q ← value.div_round_nearest num i4
                let signed ← if v < 0#i64 then -. q else ok q
                let i5 ← lift (IScalar.cast .I128 core.num.I64.MAX)
                if signed > i5 then ok (core.result.Result.Err silu.SiluError.Overflow)
                else do
                  let i6 ← lift (IScalar.cast .I128 core.num.I64.MIN)
                  if signed < i6 then ok (core.result.Result.Err silu.SiluError.Overflow)
                  else do
                    let i7 ← lift (IScalar.cast .I64 signed)
                    ok (core.result.Result.Ok i7)
        | core.result.Result.Err _ => ok (core.result.Result.Err silu.SiluError.ExpOutOfRange)) := by
  rw [silu_window] at h
  obtain ⟨hlo, hhi⟩ := h
  rw [word_silu_fixed, silu.silu_fixed, silu_UPPER_eq, silu_LOWER_eq]
  have h1 : ¬ (68719476736#i64 ≤ v) := by
    intro hc
    have : (68719476736 : ℤ) ≤ v.val := (IScalar.le_equiv _ _).1 hc
    omega
  have h2 : ¬ (v ≤ (-90194313216)#i64) := by
    intro hc
    have : v.val ≤ (-90194313216 : ℤ) := (IScalar.le_equiv _ _).1 hc
    omega
  simp only [bind_tc_ok, if_neg h1, if_neg h2]
  rfl

theorem exp_result_val_bounds (w e : I64) (hw : window w)
    (hb : |toReal e - Real.exp (toReal w)| ≤ epsExp * Real.exp (toReal w) + 4 / q32) :
    0 < e.val ∧ e.val ≤ (2:ℤ)^63 - 1 - (2:ℤ)^32 := by
  have hw' := hw
  rw [window] at hw'
  obtain ⟨hwl, hwh⟩ := hw'
  have htw_lo : (-16:ℝ) ≤ toReal w := by
    unfold toReal
    rw [le_div_iff₀ q32_pos]
    rw [show q32 = (2:ℝ)^32 from rfl]
    have h16 : (-16:ℝ) * (2:ℝ)^32 = (-68719476736 : ℤ) := by norm_num
    rw [h16]
    exact_mod_cast hwl
  have htw_hi : toReal w ≤ 21 := by
    unfold toReal
    rw [div_le_iff₀ q32_pos]
    rw [show q32 = (2:ℝ)^32 from rfl]
    have h21 : (21:ℝ) * (2:ℝ)^32 = (90194313216 : ℤ) := by norm_num
    rw [h21]
    exact_mod_cast hwh
  have hs_lt : Real.exp (toReal w) < (3/4:ℝ) * (2:ℝ)^31 := by
    have h1 : Real.exp (toReal w) ≤ Real.exp 21 := Real.exp_le_exp.mpr htw_hi
    have h2 : Real.exp (21:ℝ) = (Real.exp 1)^21 := by
      rw [← Real.exp_nat_mul]; norm_num
    have h3 : (Real.exp 1)^21 < (2.7182818286:ℝ)^21 :=
      pow_lt_pow_left₀ Real.exp_one_lt_d9 (le_of_lt (Real.exp_pos 1)) (by norm_num)
    have h4 : (2.7182818286:ℝ)^21 < (3/4:ℝ) * (2:ℝ)^31 := by norm_num
    linarith
  have hs_gt : (2:ℝ)^(-26:ℤ) < Real.exp (toReal w) := by
    have h1 : Real.exp (-16:ℝ) ≤ Real.exp (toReal w) := Real.exp_le_exp.mpr htw_lo
    have hexp16 : Real.exp (16:ℝ) < (2:ℝ)^26 := by
      have h2 : Real.exp (16:ℝ) = (Real.exp 1)^16 := by
        rw [← Real.exp_nat_mul]; norm_num
      have h3 : (Real.exp 1)^16 < (3:ℝ)^16 :=
        pow_lt_pow_left₀ Real.exp_one_lt_three (le_of_lt (Real.exp_pos 1)) (by norm_num)
      have h4 : (3:ℝ)^16 < (2:ℝ)^26 := by norm_num
      linarith
    have h5 : (2:ℝ)^(-26:ℤ) = ((2:ℝ)^26)⁻¹ := by
      rw [show (-26:ℤ) = -((26:ℕ):ℤ) by norm_num, zpow_neg, zpow_natCast]
    rw [h5]
    have h6 : ((2:ℝ)^26)⁻¹ < (Real.exp (16:ℝ))⁻¹ :=
      (inv_lt_inv₀ (by positivity) (Real.exp_pos 16)).2 hexp16
    have h7 : (Real.exp (16:ℝ))⁻¹ = Real.exp (-16:ℝ) := by
      rw [Real.exp_neg]
    linarith [h1, h6, h7.le]
  have hE_lo : 0 < toReal e := by
    have h := (abs_le.mp hb).1
    have hkey : (0:ℝ) < (2:ℝ)^(-26:ℤ) * (1 - epsExp) - 4/q32 := by
      unfold epsExp q32; norm_num [zpow_neg]
    have hmul : (2:ℝ)^(-26:ℤ) * (1 - epsExp) < Real.exp (toReal w) * (1 - epsExp) :=
      mul_lt_mul_of_pos_right hs_gt (by unfold epsExp; norm_num)
    have hge : Real.exp (toReal w) * (1 - epsExp) - 4/q32 ≤ toReal e := by
      have hh := h
      ring_nf at hh ⊢
      linarith
    have hpos : (0:ℝ) < Real.exp (toReal w) * (1 - epsExp) - 4/q32 := by
      have h4 : (4:ℝ)/q32 < Real.exp (toReal w) * (1 - epsExp) :=
        lt_trans (by linarith [hkey]) hmul
      linarith
    linarith [hge, hpos]
  have hE_hi : toReal e < (3/4:ℝ) * (2:ℝ)^31 * (1 + epsExp) + 4/q32 := by
    have h := (abs_le.mp hb).2
    have hmul : epsExp * Real.exp (toReal w) < epsExp * ((3/4:ℝ) * (2:ℝ)^31) :=
      mul_lt_mul_of_pos_left hs_lt (by unfold epsExp; positivity)
    nlinarith [h, hmul]
  have he_val : (e.val : ℝ) = toReal e * q32 := by
    unfold toReal
    rw [div_mul_cancel₀ _ q32_ne_zero]
  constructor
  · rw [show (0:ℤ) < e.val ↔ (0:ℝ) < (e.val:ℝ) by exact_mod_cast Iff.rfl]
    rw [he_val]; exact mul_pos hE_lo q32_pos
  · have hbound : (e.val : ℝ) < (((2:ℤ)^63 - 1 - (2:ℤ)^32 : ℤ) : ℝ) := by
      rw [he_val]
      calc toReal e * q32 < ((3/4:ℝ) * (2:ℝ)^31 * (1 + epsExp) + 4/q32) * q32 :=
            mul_lt_mul_of_pos_right hE_hi q32_pos
        _ < _ := by
            unfold epsExp q32
            norm_num [zpow_neg]
    have hlt : e.val < (2:ℤ)^63 - 1 - (2:ℤ)^32 := by exact_mod_cast hbound
    exact le_of_lt hlt

/-! ## Checked-operation witnesses -/

theorem I64_checked_add_some (x y : I64)
    (hlo : I64.min ≤ x.val + y.val) (hhi : x.val + y.val ≤ I64.max) :
    ∃ z, x.checked_add y = some z ∧ z.val = x.val + y.val := by
  have hspec := I64.checked_add_bv_spec x y
  cases hc : core.num.checked_add_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_add] using hc, hspec.2.2.1⟩

theorem I128_checked_mul_some (x y : I128)
    (hlo : I128.min ≤ x.val * y.val) (hhi : x.val * y.val ≤ I128.max) :
    ∃ z, x.checked_mul y = some z ∧ z.val = x.val * y.val := by
  have hspec := I128.checked_mul_bv_spec x y
  cases hc : core.num.checked_mul_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I128.checked_mul] using hc, hspec.2.2.1⟩

theorem I128_mul_q32_some (m : I128) (hm : |m.val| ≤ (2:ℤ)^63) :
    ∃ num, m.checked_mul (core.convert.num.FromI128I64.from 4294967296#i64) = some num ∧
      num.val = m.val * 4294967296 := by
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  have hq32 : (core.convert.num.FromI128I64.from 4294967296#i64).val = 4294967296 := by
    rw [core.convert.num.FromI128I64.from_val_eq]; norm_num
  have hm' := abs_le.mp hm
  obtain ⟨num, hnum, hv⟩ := I128_checked_mul_some m (core.convert.num.FromI128I64.from 4294967296#i64)
    (by rw [hI128min, hq32]; nlinarith [hm'.1])
    (by rw [hI128max, hq32]; nlinarith [hm'.2])
  exact ⟨num, hnum, by rw [hv, hq32]⟩

/-! ## Divisor and cast blocks -/

private theorem silu_div_spec (v e den : I64) (num : I128)
    (hden : den.val = 4294967296 + e.val) (heval_nn : 0 ≤ e.val) (hdenpos : 0 < den.val)
    (hden_ub : den.val ≤ (2:ℤ)^63 - 1) (hvle : |v.val| < (2:ℤ)^63)
    (hnum : num.val = |v.val| * 4294967296) :
    value.div_round_nearest num (core.convert.num.FromI128I64.from den) ⦃ q =>
      0 ≤ q.val ∧ q.val ≤ |v.val| ∧
      q.val = (|v.val| * 4294967296 + den.val / 2) / den.val ⦄ := by
  have hI128min : I128.min = -(2:ℤ)^127 := by norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2:ℤ)^127 - 1 := by norm_num [I128.max, IScalar.max, I128.numBits]
  have hdenR : (core.convert.num.FromI128I64.from den).val = den.val :=
    core.convert.num.FromI128I64.from_val_eq den
  have hnum_nn : 0 ≤ num.val := by rw [hnum]; positivity
  have hnumub : num.val ≤ (2:ℤ)^95 := by
    rw [hnum]
    have h2 : (4294967296:ℤ) = (2:ℤ)^32 := by norm_num
    nlinarith [hvle, h2]
  unfold value.div_round_nearest
  refine WP.spec_bind (I128.div_spec (x := core.convert.num.FromI128I64.from den) (y := 2#i128)
    (by norm_num) (by intro h; exact absurd h.2 (by norm_num))) ?_
  intro i hi_val
  have hi_val' : i.val = den.val / 2 := by
    rw [hi_val, hdenR]
    exact Int.tdiv_eq_ediv_of_nonneg (le_of_lt hdenpos)
  have hinn : 0 ≤ i.val := by rw [hi_val']; exact Int.ediv_nonneg (le_of_lt hdenpos) (by norm_num)
  have hiub : i.val ≤ (2:ℤ)^63 := by
    rw [hi_val']
    have := Int.ediv_le_self 2 (le_of_lt hdenpos)
    omega
  refine WP.spec_bind (I128.add_spec (x := num) (y := i)
      (by rw [hI128min]; linarith [hnum_nn, hinn])
      (by rw [hI128max]; linarith [hnumub, hiub])) ?_
  intro i1 hi1_val
  have hi1_val' : i1.val = num.val + i.val := hi1_val
  have hdenne : (core.convert.num.FromI128I64.from den).val ≠ 0 := by rw [hdenR]; omega
  refine WP.spec_mono (I128.div_spec (x := i1) (y := core.convert.num.FromI128I64.from den) hdenne
      (by intro h; exact absurd h.2 (by rw [hdenR] at h; omega))) ?_
  intro q hq_val
  have hqval : q.val = (|v.val| * 4294967296 + den.val / 2) / den.val := by
    rw [hq_val, hi1_val', hi_val', hnum, hdenR]
    exact Int.tdiv_eq_ediv_of_nonneg (by positivity)
  have hqnn : 0 ≤ q.val := by
    rw [hqval]; exact Int.ediv_nonneg (by positivity) (le_of_lt hdenpos)
  have hqub : q.val ≤ |v.val| := by
    have hvnn : (0:ℤ) ≤ |v.val| := abs_nonneg _
    have hdenlo : (4294967296:ℤ) ≤ den.val := by rw [hden]; omega
    have h1 : |v.val| * 4294967296 ≤ |v.val| * den.val :=
      Int.mul_le_mul_of_nonneg_left hdenlo hvnn
    have h2 : den.val / 2 < den.val := by
      have hd := Int.mul_ediv_add_emod den.val 2
      have h0 : 0 ≤ den.val % 2 := Int.emod_nonneg _ (by norm_num)
      omega
    have h4 : (|v.val| * 4294967296 + den.val / 2) / den.val < |v.val| + 1 := by
      rw [Int.ediv_lt_iff_lt_mul hdenpos]; nlinarith
    rw [hqval]; omega
  exact ⟨hqnn, hqub, hqval⟩

private theorem silu_cast_block (v e den : I64) (q : I128)
    (hden : den.val = 4294967296 + e.val) (hdenpos : 0 < den.val)
    (hepos : 0 ≤ toReal e)
    (hq : q.val = (|v.val| * 4294967296 + den.val / 2) / den.val)
    (hqnn : 0 ≤ q.val) (hqub : q.val ≤ |v.val|) (hvle : |v.val| < (2:ℤ)^63)
    (hbexp : |toReal e - Real.exp (-toReal v)| ≤ epsExp * Real.exp (-toReal v) + 4/q32) :
    (do
      let signed ← if v < 0#i64 then -.q else ok q
      if signed > IScalar.cast .I128 core.num.I64.MAX then
        ok (core.result.Result.Err silu.SiluError.Overflow)
      else
        if signed < IScalar.cast .I128 core.num.I64.MIN then
          ok (core.result.Result.Err silu.SiluError.Overflow)
        else ok (core.result.Result.Ok (IScalar.cast .I64 signed))) ⦃ z => ∃ i7 : I64,
      z = core.result.Result.Ok i7 ∧
      |toReal i7 - siluReal (toReal v)| ≤
        epsExp * |toReal v| + (4 / q32) * |toReal v| + 1 / (2 * q32) ⦄ := by
  have hMAX : (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = (2:ℤ)^63 - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2:ℤ)^63 - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2:ℤ)^63 - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hMIN : (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -(2:ℤ)^63 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2:ℤ)^63 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2:ℤ)^63)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hqub' : q.val ≤ (2:ℤ)^63 - 1 := by omega
  by_cases hv : v < 0#i64
  · rw [if_pos hv]
    have hvneg : v.val < 0 := (IScalar.lt_equiv _ _).1 hv
    have hqmin : q ≠ IScalar.min IScalarTy.I128 := by
      rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq]
      intro hc; omega
    obtain ⟨signed, hsigned_eq, hsigned_val⟩ := WP.spec_imp_exists
      (IScalar.neg_step (ty := IScalarTy.I128) q hqmin)
    rw [show (HNeg.hNeg q : Result I128) = ok signed from hsigned_eq]
    rw [bind_tc_ok]
    have hsval : signed.val = -q.val := hsigned_val
    have hgt : ¬ signed > IScalar.cast IScalarTy.I128 core.num.I64.MAX := by
      intro h
      rw [gt_iff_lt, IScalar.lt_equiv, hMAX] at h
      omega
    rw [if_neg hgt]
    have hlt : ¬ signed < IScalar.cast IScalarTy.I128 core.num.I64.MIN := by
      intro h
      rw [IScalar.lt_equiv, hMIN] at h
      omega
    rw [if_neg hlt]
    rw [WP.spec_ok]
    refine ⟨IScalar.cast IScalarTy.I64 signed, rfl, ?_⟩
    have hcasteq : (IScalar.cast IScalarTy.I64 signed).val = signed.val := by
      rw [IScalar.cast_val_eq]
      norm_num [IScalarTy.numBits]
      exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 signed.val
        (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
    refine silu_bound_general v e den q (IScalar.cast IScalarTy.I64 signed)
      hden hdenpos hepos hq ?_ hbexp
    rw [hcasteq, hsval, if_pos hvneg]
    ring
  · rw [if_neg hv]
    have hvnn : ¬ v.val < 0 := by
      intro h; exact hv ((IScalar.lt_equiv _ _).2 h)
    rw [bind_tc_ok]
    have hgt : ¬ q > IScalar.cast IScalarTy.I128 core.num.I64.MAX := by
      intro h; rw [gt_iff_lt, IScalar.lt_equiv, hMAX] at h; omega
    rw [if_neg hgt]
    have hlt : ¬ q < IScalar.cast IScalarTy.I128 core.num.I64.MIN := by
      intro h; rw [IScalar.lt_equiv, hMIN] at h; omega
    rw [if_neg hlt]
    rw [WP.spec_ok]
    refine ⟨IScalar.cast IScalarTy.I64 q, rfl, ?_⟩
    have hcasteq : (IScalar.cast IScalarTy.I64 q).val = q.val := by
      rw [IScalar.cast_val_eq]
      norm_num [IScalarTy.numBits]
      exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 q.val
        (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
    refine silu_bound_general v e den q (IScalar.cast IScalarTy.I64 q)
      hden hdenpos hepos hq ?_ hbexp
    rw [hcasteq, if_neg hvnn]
    ring

/-! ## Structural evaluation of the in-window success block -/

theorem silu_tail_refines (v e den : I64) (mag : I128)
    (hdenpos : 0 < den.val) (hden_val : den.val = 4294967296 + e.val)
    (hmagv : mag.val = |v.val|) (hmagb : |mag.val| ≤ (2:ℤ)^63)
    (hvle : |v.val| < (2:ℤ)^63)
    (hepos : 0 ≤ toReal e) (hehi : e.val ≤ (2:ℤ)^63 - 1 - (2:ℤ)^32)
    (hbexp : |toReal e - Real.exp (-toReal v)| ≤ epsExp * Real.exp (-toReal v) + 4/q32) :
    (do
      let i3 ← lift (core.convert.num.FromI128I64.from 4294967296#i64)
      let o1 ← lift (I128.checked_mul mag i3)
      match o1 with
      | none => ok (core.result.Result.Err silu.SiluError.Overflow)
      | some num => do
        let i4 ← lift (core.convert.num.FromI128I64.from den)
        let q ← value.div_round_nearest num i4
        let signed ← if v < 0#i64 then -.q else ok q
        let i5 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if signed > i5 then ok (core.result.Result.Err silu.SiluError.Overflow)
        else do
          let i6 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if signed < i6 then ok (core.result.Result.Err silu.SiluError.Overflow)
          else do
            let i7 ← lift (IScalar.cast .I64 signed)
            ok (core.result.Result.Ok i7)) ⦃ z => ∃ i7 : I64,
      z = core.result.Result.Ok i7 ∧
      |toReal i7 - siluReal (toReal v)| ≤
        epsExp * |toReal v| + (4 / q32) * |toReal v| + 1 / (2 * q32) ⦄ := by
  have heval_nn : 0 ≤ e.val := by
    have h : 0 ≤ toReal e * q32 := mul_nonneg hepos (le_of_lt q32_pos)
    unfold toReal at h
    rw [div_mul_cancel₀ _ q32_ne_zero] at h
    exact_mod_cast h
  have hden_ub : den.val ≤ (2:ℤ)^63 - 1 := by rw [hden_val]; omega
  simp only [lift, bind_tc_ok]
  obtain ⟨num, hnum_eq, hnum_val⟩ := I128_mul_q32_some mag hmagb
  rw [hnum_eq]
  dsimp only
  refine WP.spec_bind (silu_div_spec v e den num hden_val heval_nn hdenpos hden_ub hvle
      (by rw [hnum_val, hmagv])) ?_
  intro q hq
  exact silu_cast_block v e den q hden_val hdenpos hepos hq.2.2 hq.1 hq.2.1 hvle hbexp

theorem word_silu_fixed_spec (v : I64) (h : silu_window v) :
    word_silu_fixed v ⦃ z => ∃ i7 : I64,
      z = core.result.Result.Ok i7 ∧
      |toReal i7 - siluReal (toReal v)| ≤
        epsExp * |toReal v| + (4 / q32) * |toReal v| + 1 / (2 * q32) ⦄ := by
  have hwin := h
  rw [silu_window] at h
  obtain ⟨hlo, hhi⟩ := h
  rw [word_silu_fixed_unfold v hwin]
  have hvmin : v ≠ IScalar.min IScalarTy.I64 := by
    rw [IScalar.min_IScalarTy_I64_eq, I64.min_eq]; intro hc; omega
  refine WP.spec_bind (IScalar.neg_step (ty := IScalarTy.I64) v hvmin) ?_
  intro w hwval
  have hw_win : window w := by rw [window]; constructor <;> omega
  refine WP.spec_bind (word_fixed_exp_body_refines w hw_win) ?_
  intro r hr
  obtain ⟨e, her, hbexp0⟩ := hr
  cases her
  dsimp only
  have hbexp : |toReal e - Real.exp (toReal w)| ≤ epsExp * Real.exp (toReal w) + 4/q32 := hbexp0
  have hw_toReal : toReal w = -toReal v := by unfold toReal; rw [hwval]; push_cast; ring
  have hbexp' : |toReal e - Real.exp (-toReal v)| ≤ epsExp * Real.exp (-toReal v) + 4/q32 := by
    rwa [hw_toReal] at hbexp
  obtain ⟨hepos_val, hehi_val⟩ := exp_result_val_bounds w e hw_win hbexp
  have hepos : 0 ≤ toReal e := by
    rw [show toReal e = (e.val:ℝ)/q32 from rfl]
    exact div_nonneg (by exact_mod_cast (le_of_lt hepos_val)) (le_of_lt q32_pos)
  have hvle : |v.val| < (2:ℤ)^63 := by rw [abs_lt]; constructor <;> omega
  rw [engine_ONE_eq]
  rw [bind_tc_ok]
  have hq32val : (4294967296#i64).val = 4294967296 := by norm_num
  have hcadd : I64.min ≤ (4294967296#i64).val + e.val ∧ (4294967296#i64).val + e.val ≤ I64.max := by
    rw [hq32val, I64.min_eq, I64.max_eq]
    constructor <;> omega
  have hca_spec : Aeneas.Std.lift (I64.checked_add 4294967296#i64 e) ⦃ o =>
      ∃ den, o = some den ∧ den.val = (4294967296#i64).val + e.val ⦄ := by
    apply WP.exists_imp_spec
    obtain ⟨den, hden_eq, hden_val⟩ := I64_checked_add_some 4294967296#i64 e hcadd.1 hcadd.2
    exact ⟨I64.checked_add 4294967296#i64 e, rfl, den, hden_eq, hden_val⟩
  refine WP.spec_bind hca_spec ?_
  intro o ho
  obtain ⟨den, hden_eq, hden_val⟩ := ho
  subst hden_eq
  dsimp only
  have hdenpos : 0 < den.val := by rw [hden_val, hq32val]; omega
  have h0val : (0#i64).val = 0 := by norm_num
  have hdenle : ¬ den ≤ 0#i64 := by
    intro hc
    have := (IScalar.le_equiv _ _).1 hc
    rw [h0val] at this
    omega
  rw [if_neg hdenle]
  by_cases hv : v < 0#i64
  · rw [if_pos hv]
    have hvneg : v.val < 0 := (IScalar.lt_equiv _ _).1 hv
    have hne : core.convert.num.FromI128I64.from v ≠ IScalar.min IScalarTy.I128 := by
      rw [IScalar.min_IScalarTy_I128_eq, I128.min_eq]
      intro hc
      have hv' : (core.convert.num.FromI128I64.from v).val = v.val :=
        core.convert.num.FromI128I64.from_val_eq v
      omega
    obtain ⟨mag, hmag_eq, hmag_val⟩ := WP.spec_imp_exists
      (IScalar.neg_step (ty := IScalarTy.I128) (core.convert.num.FromI128I64.from v) hne)
    have hmag_do : (do let i3 ← lift (core.convert.num.FromI128I64.from v); -.i3) = ok mag := by
      simp only [lift, bind_tc_ok]
      exact hmag_eq
    rw [hmag_do]
    rw [bind_tc_ok]
    have hmagv : mag.val = |v.val| := by
      rw [hmag_val, core.convert.num.FromI128I64.from_val_eq, abs_of_neg hvneg]
    have hmagb : |mag.val| ≤ (2:ℤ)^63 := by
      rw [hmagv, abs_abs]
      exact le_of_lt hvle
    exact silu_tail_refines v e den mag hdenpos hden_val hmagv hmagb hvle hepos hehi_val hbexp'
  · rw [if_neg hv]
    have hvnn : 0 ≤ v.val := by
      have h : ¬ v.val < 0 := by
        intro hlt
        exact hv ((IScalar.lt_equiv _ _).2 (by rw [h0val]; exact hlt))
      omega
    have hmagv : (core.convert.num.FromI128I64.from v).val = |v.val| := by
      rw [core.convert.num.FromI128I64.from_val_eq, abs_of_nonneg hvnn]
    have hmagb : |(core.convert.num.FromI128I64.from v).val| ≤ (2:ℤ)^63 := by
      rw [hmagv, abs_abs]
      exact le_of_lt hvle
    rw [bind_tc_ok]
    exact silu_tail_refines v e den (core.convert.num.FromI128I64.from v)
      hdenpos hden_val hmagv hmagb hvle hepos hehi_val hbexp'

theorem silu_window_refines : ∀ v : I64,
    silu_window v →
      ∃ i7 : I64,
        word_silu_fixed v = ok (core.result.Result.Ok i7) ∧
          |toReal i7 - siluReal (toReal v)| ≤
            epsExp * |toReal v| + (4 / q32) * |toReal v| + 1 / (2 * q32) := by
  intro v h
  obtain ⟨y, hy, i7, hyi7, hQ⟩ := WP.spec_imp_exists (word_silu_fixed_spec v h)
  exact ⟨i7, by rw [hy, hyi7], hQ⟩

end

end apeiron_word
