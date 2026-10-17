import ApeironWord
import Numeric.Engine.Norm
import Numeric.Engine.Rsqrt
import Numeric.SiluSaturation
import Learn.GramExact
import Numeric.MoeRouter
import Learn.Grad.DenseNormSpec

/-!
# Refinement of the extracted dense backward passes (`src/learn/grad/dense.rs`)

The Rust bodies are extracted into `ApeironWord.lean` as `learn_port.dense.*` (namespace
`apeiron_word`), rooted at `crate::learn_port::word_<module>_<fn>` (`proof/extract/learn_port.rs`).
This module proves, per public entry point, that the extracted body returns the `ℤ` specification of
`Learn.Grad.DenseNormSpec` under an explicit admissible domain: agreeing lengths, `i64` range
agreement on the narrowed results (the `narrow` window), and the callee contracts of the engine
primitives whose refinements already exist (`Numeric.Engine.Norm`,
`Numeric.Engine.Rsqrt`, `Numeric.Engine.Silu`, `Numeric.SiluSaturation`).

Loop invariants are carried over the extracted `_loop`/`_loop.body` definitions
(`loop.spec_decr_nat`), never by kernel evaluation of a whole source.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_grad

open Aeneas Aeneas.Std Result
open apeiron_norm
open apeiron_word
open learn_port.dense

/-! ## 1. The extracted rounding primitives

`dense::round32`, `dense::narrow`, `dense::mul_round`, `dense::div_round`, `dense::ov`. -/

/-- Every `i64` magnitude is at most `2⁶³`. -/
theorem i64_abs_le (x : I64) : |x.val| ≤ (2 : ℤ) ^ (63 : ℕ) := by
  have h := x.hBounds
  rw [IScalarTy.I64_numBits_eq] at h
  obtain ⟨h1, h2⟩ := h
  norm_num at h1 h2
  rw [abs_le]
  constructor <;> omega

/-- The product of two `i64` magnitudes is at most `2¹²⁶`, inside `i128`. -/
theorem mul_abs_le_126 {a b : ℤ} (ha : |a| ≤ (2 : ℤ) ^ (63 : ℕ))
    (hb : |b| ≤ (2 : ℤ) ^ (63 : ℕ)) : |a * b| ≤ (2 : ℤ) ^ (126 : ℕ) := by
  rw [abs_mul]
  nlinarith [abs_nonneg a, abs_nonneg b]

theorem I128_min_eq : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
  norm_num [I128.min, IScalar.min, I128.numBits]

theorem I128_max_eq : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
  norm_num [I128.max, IScalar.max, I128.numBits]

/-- The `IScalar.min`/`IScalar.max` form of the previous two, as it appears in the preconditions of
`IScalar.add_spec`/`IScalar.sub_spec`. -/
theorem I128_min_spec : IScalar.min IScalarTy.I128 = -(2 : ℤ) ^ (127 : ℕ) := by
  norm_num [I128.min, IScalar.min, I128.numBits]

theorem I128_max_spec : IScalar.max IScalarTy.I128 = (2 : ℤ) ^ (127 : ℕ) - 1 := by
  norm_num [I128.max, IScalar.max, I128.numBits]

/-- `x ≥ 0` as an integer statement, the form the extracted `num >= 0#i128` guard has. -/
theorem I128_ge_zero (x : I128) : (x ≥ (0#i128)) ↔ (0 : ℤ) ≤ x.val := by
  rw [ge_iff_le, IScalar.le_equiv]
  simp

/-- The `i64` upper bound as an `i128`, from `IScalar.cast_val_eq` and `Int.bmod_eq_of_le`. -/
theorem cast_I128_I64MAX_val :
    (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = 9223372036854775807 := by
  rw [IScalar.cast_val_eq]
  show (core.num.I64.MAX : I64).val.bmod (2 ^ min IScalarTy.I128.numBits IScalarTy.I64.numBits)
    = 9223372036854775807
  rw [show IScalarTy.I128.numBits = 128 from rfl, show IScalarTy.I64.numBits = 64 from rfl]
  rw [show min 128 64 = 64 from rfl]
  rw [show (core.num.I64.MAX : I64).val = 9223372036854775807 from rfl]
  rw [Int.bmod_eq_of_le (n := 9223372036854775807) (m := 2 ^ 64) (by norm_num) (by norm_num)]

/-- The `i64` lower bound as an `i128`. -/
theorem cast_I128_I64MIN_val :
    (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -9223372036854775808 := by
  rw [IScalar.cast_val_eq]
  show (core.num.I64.MIN : I64).val.bmod (2 ^ min IScalarTy.I128.numBits IScalarTy.I64.numBits)
    = -9223372036854775808
  rw [show IScalarTy.I128.numBits = 128 from rfl, show IScalarTy.I64.numBits = 64 from rfl]
  rw [show min 128 64 = 64 from rfl]
  rw [show (core.num.I64.MIN : I64).val = -9223372036854775808 from rfl]
  rw [Int.bmod_eq_of_le (n := -9223372036854775808) (m := 2 ^ 64) (by norm_num) (by norm_num)]

/-- The narrowing cast `IScalar.cast .I64` preserves an in-range `i128` value. -/
theorem cast_I64_val (v : I128)
    (hlo : -9223372036854775808 ≤ v.val) (hhi : v.val ≤ 9223372036854775807) :
    (IScalar.cast IScalarTy.I64 v).val = v.val := by
  rw [IScalar.cast_val_eq]
  show v.val.bmod (2 ^ min IScalarTy.I64.numBits IScalarTy.I128.numBits) = v.val
  rw [show IScalarTy.I64.numBits = 64 from rfl, show IScalarTy.I128.numBits = 128 from rfl]
  rw [show min 64 128 = 64 from rfl]
  rw [Int.bmod_eq_of_le (n := v.val) (m := 2 ^ 64) (by omega) (by omega)]

/-- The extracted `i64 → i128` widening and the `.I64 → .I128` signed cast are the same
    sign-extension of the source bit-vector. -/
theorem fromI64_eq_cast (z : I64) :
    core.convert.num.FromI128I64.from z = IScalar.cast IScalarTy.I128 z := rfl

/-- The `dense::narrow` upper guard, as a value comparison. -/
theorem narrow_guard_high (v : I128) :
    (v > IScalar.cast IScalarTy.I128 core.num.I64.MAX) ↔ 9223372036854775807 < v.val := by
  constructor
  · intro h
    rw [gt_iff_lt, IScalar.lt_equiv] at h
    rw [show (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = 9223372036854775807 from
      cast_I128_I64MAX_val] at h
    exact h
  · intro h
    rw [gt_iff_lt, IScalar.lt_equiv]
    rw [show (IScalar.cast IScalarTy.I128 core.num.I64.MAX).val = 9223372036854775807 from
      cast_I128_I64MAX_val]
    exact h

/-- The `dense::narrow` lower guard, as a value comparison. -/
theorem narrow_guard_low (v : I128) :
    (v < IScalar.cast IScalarTy.I128 core.num.I64.MIN) ↔ v.val < -9223372036854775808 := by
  constructor
  · intro h
    rw [IScalar.lt_equiv] at h
    rw [show (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -9223372036854775808 from
      cast_I128_I64MIN_val] at h
    exact h
  · intro h
    rw [IScalar.lt_equiv]
    rw [show (IScalar.cast IScalarTy.I128 core.num.I64.MIN).val = -9223372036854775808 from
      cast_I128_I64MIN_val]
    exact h

/-- **`dense::narrow`, value arm.** Inside the two `i64` guards the narrowing succeeds and its value
is the mathematical `i128` argument. -/
theorem dense_narrow_some (v : I128)
    (hmax : ¬ (9223372036854775807 < v.val))
    (hmin : ¬ (v.val < -9223372036854775808)) :
    solve_port.q32.narrow v = ok (some (IScalar.cast IScalarTy.I64 v)) := by
  rw [solve_port.q32.narrow]
  simp only [lift, bind_tc_ok]
  rw [if_neg (by rw [fromI64_eq_cast, narrow_guard_high]; exact hmax),
    if_neg (by rw [fromI64_eq_cast, narrow_guard_low]; exact hmin)]

/-- **`dense::narrow`, refusal arm.** Outside either guard the body returns `none` exactly. -/
theorem dense_narrow_none (v : I128)
    (h : 9223372036854775807 < v.val ∨ v.val < -9223372036854775808) :
    solve_port.q32.narrow v = ok none := by
  rw [solve_port.q32.narrow]
  simp only [lift, bind_tc_ok]
  rcases h with h | h
  · rw [if_pos (by rw [fromI64_eq_cast, narrow_guard_high]; exact h)]
  · rw [if_neg (by rw [fromI64_eq_cast, narrow_guard_high]; omega),
      if_pos (by rw [fromI64_eq_cast, narrow_guard_low]; exact h)]

/-- **`dense::round32`.** With the half-biased add inside `i128` the body returns `some` of the
specification value `round32 v.val`. -/
theorem dense_round32_val (v : I128)
    (hlo : I128.min ≤ v.val + 2 ^ (31 : ℕ)) (hhi : v.val + 2 ^ (31 : ℕ) ≤ I128.max) :
    ∃ y : I128, learn_port.dense.round32 v = ok (some y) ∧ y.val = apeiron_norm.round32 v.val := by
  have h31' : ((2147483648#i128 : I128)).val = (2 : ℤ) ^ (31 : ℕ) := by norm_num
  obtain ⟨i1, hi1, hi1v⟩ := I128_checked_add_some v (2147483648#i128) hlo hhi
  obtain ⟨y, hy, hyv⟩ := I128_shiftRight_32_val i1
  refine ⟨y, ?_, ?_⟩
  · simp only [learn_port.dense.round32, I128_shiftLeft_31, lift, bind_tc_ok, hi1, hy]
  · rw [hyv, hi1v, h31', apeiron_norm.round32, apeiron_norm.rshr]

/-- **`dense::mul_round`, value arm.** The `i128` product, its `round32` and the `i64` narrowing all
fit: the result is `mulRound a.val b.val`. -/
theorem dense_mul_round_some (a b : I64)
    (h : -9223372036854775808 ≤ apeiron_norm.mulRound a.val b.val ∧
         apeiron_norm.mulRound a.val b.val ≤ 9223372036854775807) :
    ∃ z : I64, learn_port.dense.mul_round a b = ok (some z) ∧
      z.val = apeiron_norm.mulRound a.val b.val := by
  have habs : |a.val * b.val| ≤ (2 : ℤ) ^ (126 : ℕ) := mul_abs_le_126 (i64_abs_le a) (i64_abs_le b)
  have hle := abs_le.mp habs
  have hmin : -(2 : ℤ) ^ (127 : ℕ) ≤ a.val * b.val := by
    have : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (126 : ℕ) := by norm_num
    linarith [hle.1, this]
  have hmax : a.val * b.val ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
    have : (2 : ℤ) ^ (126 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
    linarith [hle.2, this]
  obtain ⟨p, hp, hpv⟩ := I128_checked_mul_some (core.convert.num.FromI128I64.from a)
    (core.convert.num.FromI128I64.from b)
    (by rw [I128_min_eq]; rw [core.convert.num.FromI128I64.from_val_eq,
      core.convert.num.FromI128I64.from_val_eq]; exact hmin)
    (by rw [I128_max_eq]; rw [core.convert.num.FromI128I64.from_val_eq,
      core.convert.num.FromI128I64.from_val_eq]; exact hmax)
  have hpv' : p.val = a.val * b.val := by
    rw [hpv, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  have hlo : I128.min ≤ p.val + 2 ^ (31 : ℕ) := by
    rw [I128_min_eq, hpv']
    have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (62 : ℕ) := by norm_num
    linarith [hle.1, h2]
  have hhi : p.val + 2 ^ (31 : ℕ) ≤ I128.max := by
    rw [I128_max_eq, hpv']
    have h3 : (2 : ℤ) ^ (126 : ℕ) + 2 ^ (31 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
    linarith [hle.2, h3]
  obtain ⟨q, hq, hqv⟩ := dense_round32_val p hlo hhi
  have hqv' : q.val = apeiron_norm.mulRound a.val b.val := by
    rw [hqv, hpv', apeiron_norm.mulRound]
  refine ⟨IScalar.cast IScalarTy.I64 q, ?_, ?_⟩
  · have hn := dense_narrow_some q (by rw [hqv']; omega) (by rw [hqv']; omega)
    simp only [learn_port.dense.mul_round, lift, bind_tc_ok, hp, hq, hn]
  · rw [cast_I64_val q (by rw [hqv']; omega) (by rw [hqv']; omega), hqv']

/-- **`dense::mul_round`, refusal arm.** Outside the `i64` window the narrowing fails. -/
theorem dense_mul_round_none (a b : I64)
    (h : apeiron_norm.mulRound a.val b.val < -9223372036854775808 ∨
         9223372036854775807 < apeiron_norm.mulRound a.val b.val) :
    learn_port.dense.mul_round a b = ok none := by
  have habs : |a.val * b.val| ≤ (2 : ℤ) ^ (126 : ℕ) := mul_abs_le_126 (i64_abs_le a) (i64_abs_le b)
  have hle := abs_le.mp habs
  have hmin : -(2 : ℤ) ^ (127 : ℕ) ≤ a.val * b.val := by
    have : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (126 : ℕ) := by norm_num
    linarith [hle.1, this]
  have hmax : a.val * b.val ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
    have : (2 : ℤ) ^ (126 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
    linarith [hle.2, this]
  obtain ⟨p, hp, hpv⟩ := I128_checked_mul_some (core.convert.num.FromI128I64.from a)
    (core.convert.num.FromI128I64.from b)
    (by rw [I128_min_eq]; rw [core.convert.num.FromI128I64.from_val_eq,
      core.convert.num.FromI128I64.from_val_eq]; exact hmin)
    (by rw [I128_max_eq]; rw [core.convert.num.FromI128I64.from_val_eq,
      core.convert.num.FromI128I64.from_val_eq]; exact hmax)
  have hpv' : p.val = a.val * b.val := by
    rw [hpv, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  have hlo : I128.min ≤ p.val + 2 ^ (31 : ℕ) := by
    rw [I128_min_eq, hpv']
    have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (62 : ℕ) := by norm_num
    linarith [hle.1, h2]
  have hhi : p.val + 2 ^ (31 : ℕ) ≤ I128.max := by
    rw [I128_max_eq, hpv']
    have h3 : (2 : ℤ) ^ (126 : ℕ) + 2 ^ (31 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by norm_num
    linarith [hle.2, h3]
  obtain ⟨q, hq, hqv⟩ := dense_round32_val p hlo hhi
  have hqv' : q.val = apeiron_norm.mulRound a.val b.val := by
    rw [hqv, hpv', apeiron_norm.mulRound]
  have hn : solve_port.q32.narrow q = ok none := by
    apply dense_narrow_none
    rw [hqv']
    rcases h with h | h
    · exact Or.inr h
    · exact Or.inl h
  have hg : learn_port.dense.mul_round a b = ok none := by
    simp only [learn_port.dense.mul_round, lift, bind_tc_ok, hp, hq, hn]
  exact hg

/-- **`dense::div_round`.** With `d ≥ 1` and the quotient inside `i64` the body returns the signed
round-to-nearest-ties-away-from-zero `divRound v.val d.val`. -/
theorem dense_div_round_some (v d : I64) (hd : 1 ≤ d.val)
    (h : -9223372036854775808 ≤ apeiron_norm.divRound v.val d.val ∧
         apeiron_norm.divRound v.val d.val ≤ 9223372036854775807) :
    ∃ z : I64, learn_port.dense.div_round v d = ok (some z) ∧
      z.val = apeiron_norm.divRound v.val d.val := by
  have hvabs := abs_le.mp (i64_abs_le v)
  have hdabs := abs_le.mp (i64_abs_le d)
  have hd0 : (0 : ℤ) ≤ d.val / 2 := by omega
  have hd2 : -(2 : ℤ) ^ (63 : ℕ) ≤ d.val / 2 := by omega
  have hd2' : d.val / 2 ≤ (2 : ℤ) ^ (63 : ℕ) := by omega
  rw [learn_port.dense.div_round]
  simp only [lift, bind_tc_ok]
  set num : I128 := core.convert.num.FromI128I64.from v with hnum
  set den : I128 := core.convert.num.FromI128I64.from d with hden
  have hnumv : num.val = v.val := core.convert.num.FromI128I64.from_val_eq v
  have hdenv : den.val = d.val := core.convert.num.FromI128I64.from_val_eq d
  have h2v : ((2#i128 : I128)).val = (2 : ℤ) := by norm_num
  have hd_half : (d.val).tdiv 2 = d.val / 2 := Int.tdiv_eq_ediv_of_nonneg (by omega)
  obtain ⟨i1, hi1, hi1v⟩ := IScalar.div_spec (x := den) (y := (2#i128))
    (by rw [h2v]; norm_num) (by intro hc; rw [h2v] at hc; omega)
  have hi1v' : i1.val = d.val / 2 := by
    rw [hi1v, hdenv, h2v]
    exact Int.tdiv_eq_ediv_of_nonneg (by omega)
  by_cases hge : num ≥ (0#i128)
  · have hv : (0 : ℤ) ≤ v.val := by
      have h' := (I128_ge_zero num).1 hge
      rwa [hnumv] at h'
    rw [if_pos hge, hi1]
    simp only [bind_tc_ok]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (IScalar.add_spec (x := num) (y := i1)
      (by rw [I128_min_spec]; rw [hnumv, hi1v']
          have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (63 : ℕ) := by norm_num
          linarith [hvabs.1, hd0, h2])
      (by rw [I128_max_spec]; rw [hnumv, hi1v']
          have h3 : (2 : ℤ) ^ (63 : ℕ) + (2 : ℤ) ^ (63 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
            norm_num
          linarith [hvabs.2, hd2', h3]))
    rw [hi2]
    simp only [bind_tc_ok]
    obtain ⟨q, hq, hqv⟩ := IScalar.div_spec (x := i2) (y := den)
      (by rw [hdenv]; omega)
      (by intro hc; rw [hdenv] at hc; omega)
    rw [hq]
    simp only [bind_tc_ok]
    have hqval : q.val = apeiron_norm.divRound v.val d.val := by
      rw [hqv, hi2v, hnumv, hi1v', hdenv, apeiron_norm.divRound, if_pos hv,
        apeiron_norm.rdiv, apeiron_norm.rdiv, ← hd_half]
    have hn := dense_narrow_some q (by rw [hqval]; omega) (by rw [hqval]; omega)
    refine ⟨IScalar.cast IScalarTy.I64 q, ?_, ?_⟩
    · simp only [hn]
    · rw [cast_I64_val q (by rw [hqval]; omega) (by rw [hqval]; omega), hqval]
  · have hv : ¬ ((0 : ℤ) ≤ v.val) := by
      intro h'
      exact hge ((I128_ge_zero num).2 (by rwa [hnumv]))
    rw [if_neg hge, hi1]
    simp only [bind_tc_ok]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (IScalar.sub_spec (x := num) (y := i1)
      (by rw [I128_min_spec]; rw [hnumv, hi1v']
          have h2 : -(2 : ℤ) ^ (127 : ℕ) ≤ -(2 : ℤ) ^ (63 : ℕ) - (2 : ℤ) ^ (63 : ℕ) := by norm_num
          linarith [hvabs.1, hd2', h2])
      (by rw [I128_max_spec]; rw [hnumv, hi1v']
          have h3 : (2 : ℤ) ^ (63 : ℕ) + (2 : ℤ) ^ (63 : ℕ) ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
            norm_num
          linarith [hvabs.2, hd0, h3]))
    rw [hi2]
    simp only [bind_tc_ok]
    obtain ⟨q, hq, hqv⟩ := IScalar.div_spec (x := i2) (y := den)
      (by rw [hdenv]; omega)
      (by intro hc; rw [hdenv] at hc; omega)
    rw [hq]
    simp only [bind_tc_ok]
    have hqval : q.val = apeiron_norm.divRound v.val d.val := by
      rw [hqv, hi2v, hnumv, hi1v', hdenv, apeiron_norm.divRound, if_neg hv,
        apeiron_norm.rdiv, apeiron_norm.rdiv, ← hd_half]
    have hn := dense_narrow_some q (by rw [hqval]; omega) (by rw [hqval]; omega)
    refine ⟨IScalar.cast IScalarTy.I64 q, ?_, ?_⟩
    · simp only [hn]
    · rw [cast_I64_val q (by rw [hqval]; omega) (by rw [hqval]; omega), hqval]

/-- **`dense::ov`.** The `Option` lifting is `Overflow` on `none`. -/
theorem dense_ov_some (v : I64) :
    learn_port.dense.ov (some v) = ok (core.result.Result.Ok v) := by
  rw [learn_port.dense.ov]

theorem dense_ov_none :
    learn_port.dense.ov none = ok (core.result.Result.Err learn_port.dense.DenseError.Overflow) := by
  rw [learn_port.dense.ov]

/-! ## 2. `swiglu_fwd`

The loop `swiglu_fwd_loop` maps each pair `(silu(gate_k), up_k)` through `mul_round` and pushes it.
The callee `silu_one` is taken as an explicit contract (`Numeric.Engine.Silu` refines the
engine's `silu.silu_fixed`, the `ffn` mirror of the same body is not yet bridged). -/

/-- The pure value model of the `swiglu_fwd` loop: after `j` steps the vector holds
`mulRound` of the first `j` (silu value, up value) pairs. -/
def swigluVals (sv up : List I64) : ℕ → List ℤ
  | 0 => []
  | (j + 1) =>
      swigluVals sv up j ++
        [apeiron_norm.mulRound (sv.getD j default).val (up.getD j default).val]

theorem swigluVals_step (sv up : List I64) (j : ℕ) :
    swigluVals sv up (j + 1) = swigluVals sv up j ++
      [apeiron_norm.mulRound (sv.getD j default).val (up.getD j default).val] := rfl

theorem swigluVals_length (sv up : List I64) (j : ℕ) : (swigluVals sv up j).length = j := by
  induction j with
  | zero => rfl
  | succ k ih => rw [swigluVals_step, List.length_append, List.length_singleton, ih]

/-- The recursive model is the specification's `zipWith` prefix. -/
theorem swigluVals_eq_zipWith (sv up : List I64) (j : ℕ) (h1 : j ≤ sv.length)
    (h2 : j ≤ up.length) :
    swigluVals sv up j =
      (List.zipWith (fun s u => apeiron_norm.mulRound s.val u.val) sv up).take j := by
  induction j with
  | zero => rfl
  | succ k ih =>
      rw [swigluVals_step, ih (by omega) (by omega), List.take_add_one]
      congr 1
      have hk1 : k < sv.length := by omega
      have hk2 : k < up.length := by omega
      have hz : (List.zipWith (fun s u => apeiron_norm.mulRound s.val u.val) sv up)[k]?
          = some (apeiron_norm.mulRound (sv.getD k default).val (up.getD k default).val) := by
        rw [List.getElem?_zipWith, List.getElem?_eq_getElem hk1, List.getElem?_eq_getElem hk2,
          List.getD_eq_getElem _ _ hk1, List.getD_eq_getElem _ _ hk2]
      rw [hz]
      rfl

/-- The full-length form of `swigluVals_eq_zipWith`: when both argument lists have length `j`, the
model is the untaken zip. -/
theorem swigluVals_eq_zipWith_len (sv up : List I64) (j : ℕ) (h1 : sv.length = j)
    (h2 : up.length = j) :
    swigluVals sv up j =
      List.zipWith (fun s u => apeiron_norm.mulRound s.val u.val) sv up := by
  rw [swigluVals_eq_zipWith sv up j (by omega) (by omega)]
  rw [List.take_of_length_le (by rw [List.length_zipWith]; omega)]

/-- One `swiglu_fwd_loop` iteration with `ok1 = true`, a successful `silu_one` and a successful
`mul_round` advances the model by one and returns `cont`. -/
theorem swiglu_fwd_loop_body_step
    (gate up : Slice I64) (sv : List I64) (n : Usize)
    (out : alloc.vec.Vec I64) (k : Usize) (err : DenseError)
    (hn : n.val = gate.val.length) (hsv : sv.length = n.val) (hup : up.val.length = n.val)
    (hk : k.val < n.val) (hnumax : k.val + 1 ≤ Usize.max)
    (hinv : out.val.map (fun x => x.val) = swigluVals sv up.val k.val)
    (hbound : out.val.length + 1 ≤ U32.max)
    (v : I64)
    (hsilu : learn_port.dense.silu_one (gate.val.getD k.val default) =
      ok (core.result.Result.Ok (sv.getD k.val default)))
    (hmul : learn_port.dense.mul_round (sv.getD k.val default) (up.val.getD k.val default) = ok (some v))
    (hmulv : v.val = apeiron_norm.mulRound (sv.getD k.val default).val
      (up.val.getD k.val default).val) :
    ∃ out' k', learn_port.dense.swiglu_fwd_loop.body gate up n out k err true
        = ok (ControlFlow.cont (out', k', err, true)) ∧
      k'.val = k.val + 1 ∧ out'.val.map (fun x => x.val) = swigluVals sv up.val k'.val := by
  have hklt_gate : k.val < gate.val.length := by omega
  have hklt_up : k.val < up.val.length := by omega
  have hgate : Slice.index_usize gate k = ok (gate.val.getD k.val default) :=
    apeiron_gram.slice_index_getD gate k hklt_gate
  have hup' : Slice.index_usize up k = ok (up.val.getD k.val default) :=
    apeiron_gram.slice_index_getD up k hklt_up
  have hkn : k < n := (UScalar.lt_equiv k n).2 hk
  obtain ⟨out', hpush, hout'⟩ := alloc.vec.Vec.push_val out v hbound
  obtain ⟨k', hk', hk'v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by
      show k.val + (1#usize).val ≤ Usize.max
      rw [show (1#usize).val = 1 from rfl]
      exact hnumax))
  refine ⟨out', k', ?_, hk'v, ?_⟩
  · rw [learn_port.dense.swiglu_fwd_loop.body, if_pos hkn, if_pos rfl]
    simp only [hgate, hsilu, hup', hmul, hpush, hk', bind_tc_ok, uncurry_apply_pair]
  · rw [hk'v, show (↑k + ↑1#usize) = ↑k + 1 from rfl, swigluVals_step, ← hinv, hout',
      List.map_append, List.map_cons, List.map_nil, hmulv]

/-- Specification of `swiglu_fwd_loop` on the all-succeed domain. -/
theorem swiglu_fwd_loop_spec
    (gate up : Slice I64) (sv : List I64) (n : Usize)
    (out0 : alloc.vec.Vec I64)
    (hn : n.val = gate.val.length) (hsv : sv.length = n.val) (hup : up.val.length = n.val)
    (hmax : n.val ≤ Usize.max)
    (hnmax : n.val + 1 ≤ U32.max)
    (hsilu : ∀ k, k < n.val → learn_port.dense.silu_one (gate.val.getD k default) =
      ok (core.result.Result.Ok (sv.getD k default)))
    (hfit : ∀ k, k < n.val → -9223372036854775808 ≤
        apeiron_norm.mulRound (sv.getD k default).val (up.val.getD k default).val ∧
        apeiron_norm.mulRound (sv.getD k default).val (up.val.getD k default).val
          ≤ 9223372036854775807)
    (hinv0 : out0.val.map (fun x => x.val) = swigluVals sv up.val 0) :
    WP.spec (learn_port.dense.swiglu_fwd_loop gate up n out0 0#usize DenseError.Overflow true)
      (fun b => b.2.2 = true ∧
        b.1.val.map (fun x => x.val) = swigluVals sv up.val n.val) := by
  unfold learn_port.dense.swiglu_fwd_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × DenseError × Bool => n.val - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ n.val ∧ s.2.2.2 = true ∧
      s.1.val.map (fun x => x.val) = swigluVals sv up.val s.2.1.val)
    (post := fun b : alloc.vec.Vec I64 × DenseError × Bool =>
      b.2.2 = true ∧ b.1.val.map (fun x => x.val) = swigluVals sv up.val n.val)
  · rintro ⟨out', k', err', ok''⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hkle, hok, hmodel⟩ := hinv
    rw [hok]
    by_cases hklt : k'.val < n.val
    · have hnumax : k'.val + 1 ≤ Usize.max := by omega
      have hlen' : out'.val.length = k'.val := by
        have hh := congrArg List.length hmodel
        rw [List.length_map, swigluVals_length] at hh
        exact hh
      have hbound : out'.val.length + 1 ≤ U32.max := by omega
      obtain ⟨v, hmul, hmulv⟩ := dense_mul_round_some (sv.getD k'.val default)
        (up.val.getD k'.val default) (hfit k'.val hklt)
      obtain ⟨out'', k'', hexec, hk''v, hmodel''⟩ := swiglu_fwd_loop_body_step gate up sv n out'
        k' err' hn hsv hup hklt hnumax hmodel hbound v (hsilu k'.val hklt) hmul hmulv
      rw [hexec]
      refine ⟨⟨?_, rfl, hmodel''⟩, ?_⟩
      · rw [hk''v]; omega
      · dsimp only; rw [hk''v]; omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.swiglu_fwd_loop.body gate up n out' k' err' true
          = ok (ControlFlow.done (out', err', true)) := by
        rw [learn_port.dense.swiglu_fwd_loop.body, if_neg hnot]
      simp only [hbody]
      have hk'n : k'.val = n.val := by omega
      exact ⟨rfl, by rw [← hk'n]; exact hmodel⟩
  · exact ⟨by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega,
      rfl, hinv0⟩

/-- **`swiglu_fwd`.** On the all-succeed domain the extracted body returns the `Ok` vector and its
value list is `swigluFwd sv up.val`. -/
theorem swiglu_fwd_spec (gate up : Slice I64) (sv : List I64)
    (hne : gate.val.length ≠ 0)
    (hlen : up.val.length = gate.val.length)
    (hsv : sv.length = gate.val.length)
    (hnmax : gate.val.length + 1 ≤ U32.max)
    (hsilu : ∀ k, k < gate.val.length → learn_port.dense.silu_one (gate.val.getD k default) =
      ok (core.result.Result.Ok (sv.getD k default)))
    (hfit : ∀ k, k < gate.val.length → -9223372036854775808 ≤
        apeiron_norm.mulRound (sv.getD k default).val (up.val.getD k default).val ∧
        apeiron_norm.mulRound (sv.getD k default).val (up.val.getD k default).val
          ≤ 9223372036854775807) :
    ∃ out : alloc.vec.Vec I64, learn_port.dense.swiglu_fwd gate up
        = ok (core.result.Result.Ok out) ∧
      out.val.map (fun x => x.val) =
        apeiron_norm.swigluFwd (sv.map (fun x => x.val)) (up.val.map (fun x => x.val)) := by
  have hn : (Slice.len gate).val = gate.val.length := Slice.len_val gate
  have hnz : ¬ (Slice.len gate = 0#usize) := by
    intro h
    apply hne
    have hv := congrArg (fun u : Usize => u.val) h
    rw [hn, show (0#usize).val = 0 from rfl] at hv
    exact hv
  have hupv : (Slice.len up).val = up.val.length := Slice.len_val up
  have hshape : ¬ (Slice.len up != Slice.len gate) := by
    rw [bne_iff_ne]
    intro h
    apply h
    exact UScalar.eq_of_val_eq (by rw [hupv, hn, hlen])
  have hcap : (alloc.vec.Vec.with_capacity I64 (Slice.len gate)).val = [] := rfl
  have hloop := swiglu_fwd_loop_spec gate up sv (Slice.len gate)
    (alloc.vec.Vec.with_capacity I64 (Slice.len gate))
    hn hsv (hlen.trans hn.symm) (by rw [hn]; exact Slice.property gate)
    (by rw [hn]; exact hnmax) hsilu hfit (by rw [hcap]; rfl)
  rw [learn_port.dense.swiglu_fwd, if_neg hnz, if_neg hshape]
  dsimp (zeta := true) only
  obtain ⟨b, hbeq, hbpost⟩ := WP.spec_imp_exists hloop
  rcases b with ⟨wo, we, wok⟩
  obtain ⟨hwok, hwmodel⟩ := hbpost
  have hwok' : wok = true := hwok
  have hwmodel' : wo.val.map (fun x => x.val) = swigluVals sv up.val gate.val.length := by
    simpa using hwmodel
  rw [hbeq]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [if_pos hwok']
  refine ⟨wo, rfl, ?_⟩
  rw [hwmodel', swigluVals_eq_zipWith_len sv up.val gate.val.length hsv hlen]
  unfold apeiron_norm.swigluFwd
  rw [List.zipWith_map_left, List.zipWith_map_right]

/-! ## 3. `swiglu_one_bwd`

The per-element SwiGLU backward: `dgate = round(round(dy·u)·silu'(g))`, `dup = round(dy·silu(g))`.
The three engine callees (`sigmoid_one`, `silu_deriv`, `silu_one`) are explicit contracts; the three
`mul_round`s and their `ov` lifts are proved. -/

/-- **`dense::swiglu_one_bwd`.** On the all-succeed domain the returned pair is
`(swigluDgate dy u ds, swigluDup dy sv)` for the callee values `s = sigmoid_one g`, `ds = silu_deriv
g s`, `sv = silu_one g`. -/
theorem swiglu_one_bwd_spec (g u dy : I64) (s ds sv : I64)
    (hsig : learn_port.dense.sigmoid_one g = ok (core.result.Result.Ok s))
    (hder : learn_port.dense.silu_deriv g s = ok (core.result.Result.Ok ds))
    (hsilu : learn_port.dense.silu_one g = ok (core.result.Result.Ok sv))
    (hmu : -9223372036854775808 ≤ apeiron_norm.mulRound dy.val u.val ∧
      apeiron_norm.mulRound dy.val u.val ≤ 9223372036854775807)
    (hmd : -9223372036854775808 ≤
        apeiron_norm.mulRound (apeiron_norm.mulRound dy.val u.val) ds.val ∧
      apeiron_norm.mulRound (apeiron_norm.mulRound dy.val u.val) ds.val ≤ 9223372036854775807)
    (hms : -9223372036854775808 ≤ apeiron_norm.mulRound dy.val sv.val ∧
      apeiron_norm.mulRound dy.val sv.val ≤ 9223372036854775807) :
    ∃ dgate dup : I64, learn_port.dense.swiglu_one_bwd g u dy
        = ok (core.result.Result.Ok (dgate, dup)) ∧
      dgate.val = apeiron_norm.swigluDgate dy.val u.val ds.val ∧
      dup.val = apeiron_norm.swigluDup dy.val sv.val := by
  obtain ⟨v1, hm1, hm1v⟩ := dense_mul_round_some dy u hmu
  obtain ⟨v2, hm2, hm2v⟩ := dense_mul_round_some v1 ds (by rw [hm1v]; exact hmd)
  obtain ⟨v3, hm3, hm3v⟩ := dense_mul_round_some dy sv hms
  refine ⟨v2, v3, ?_, ?_, ?_⟩
  · simp only [learn_port.dense.swiglu_one_bwd, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch, hsig, hder, hsilu, hm1, hm2, hm3, dense_ov_some]
  · rw [hm2v, hm1v]; rfl
  · rw [hm3v]; rfl

/-! ## 3b. Refusal-arm witnesses

`swiglu_fwd_spec` is the success branch. The two `Err` arms of the entry are reachable and exact. -/

/-- **Empty gate.** `gate.val = []` returns `Err Empty`. -/
theorem swiglu_fwd_empty (gate up : Slice I64) (h : gate.val = []) :
    learn_port.dense.swiglu_fwd gate up =
      ok (core.result.Result.Err learn_port.dense.DenseError.Empty) := by
  have hn : Slice.len gate = 0#usize := by
    apply UScalar.eq_of_val_eq
    simp [Slice.len, h, UScalar.ofNatCore_val_eq]
  rw [learn_port.dense.swiglu_fwd, hn]
  simp

/-- **Shape mismatch.** An `up` whose length differs returns `Err Shape`. -/
theorem swiglu_fwd_shape (gate up : Slice I64) (hnz : Slice.len gate ≠ 0#usize)
    (h : (Slice.len up != Slice.len gate) = true) :
    learn_port.dense.swiglu_fwd gate up =
      ok (core.result.Result.Err learn_port.dense.DenseError.Shape) := by
  rw [learn_port.dense.swiglu_fwd, if_neg hnz]
  dsimp (zeta := true) only
  rw [if_pos h]

/-! ## 4. `silu_deriv`

`silu'(g) = round(s·(2³² + round(g·(2³² − s)/2³²))/2³²)` with `s = σ(g)`. The body is a pure chain of
the primitives already refined in section 1; the admissible domain is the four windows a checked
operation needs plus the final `i64` narrowing window. -/

/-- `I64::checked_sub` succeeds with the mathematical difference inside the `i64` window. -/
theorem norm_I64_checked_sub_some (x y : I64)
    (hlo : I64.min ≤ x.val - y.val) (hhi : x.val - y.val ≤ I64.max) :
    ∃ z, x.checked_sub y = some z ∧ z.val = x.val - y.val := by
  have hspec := I64.checked_sub_bv_spec x y
  cases hc : core.num.checked_sub_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_sub] using hc, hspec.2.2.1⟩

/-- **`dense::silu_deriv`.** On the all-succeed domain the extracted body returns `some` of
`siluDeriv g.val s.val`. -/
theorem silu_deriv_spec (g s : I64)
    (hsub : I64.min ≤ (2 : ℤ) ^ (32 : ℕ) - s.val ∧ (2 : ℤ) ^ (32 : ℕ) - s.val ≤ I64.max)
    (hmul : -9223372036854775808 ≤ apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val) ∧
        apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val) ≤ 9223372036854775807)
    (hadd : I128.min ≤ (2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val) ∧
        (2 : ℤ) ^ (32 : ℕ) + apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val) ≤ I128.max)
    (hpmul : I128.min ≤ s.val * ((2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val)) ∧
        s.val * ((2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val)) ≤ I128.max)
    (hround : I128.min ≤ s.val * ((2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val)) + 2 ^ (31 : ℕ) ∧
        s.val * ((2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val)) + 2 ^ (31 : ℕ) ≤ I128.max)
    (hfit : -9223372036854775808 ≤
        apeiron_norm.round32 (s.val * ((2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val))) ∧
        apeiron_norm.round32 (s.val * ((2 : ℤ) ^ (32 : ℕ) +
          apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val))) ≤ 9223372036854775807) :
    ∃ z : I64, learn_port.dense.silu_deriv g s = ok (core.result.Result.Ok z) ∧
      z.val = apeiron_norm.siluDeriv g.val s.val := by
  have hsub' : I64.min ≤ ((4294967296#i64) : I64).val - s.val ∧
      ((4294967296#i64) : I64).val - s.val ≤ I64.max := by
    rw [show ((4294967296#i64) : I64).val = (2 : ℤ) ^ (32 : ℕ) from by norm_num]
    exact hsub
  obtain ⟨om, hom, homv⟩ := norm_I64_checked_sub_some (4294967296#i64) s hsub'.1 hsub'.2
  have h32v : ((4294967296#i64) : I64).val = (2 : ℤ) ^ (32 : ℕ) := by norm_num
  have homv' : om.val = (2 : ℤ) ^ (32 : ℕ) - s.val := by rw [homv, h32v]
  obtain ⟨gm, hgm, hgmv⟩ := dense_mul_round_some g om (by rw [homv']; exact hmul)
  have hgmv' : gm.val = apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val) := by
    rw [hgmv, homv']
  have hfrom32 : (core.convert.num.FromI128I64.from (4294967296#i64)).val = (2 : ℤ) ^ (32 : ℕ) := by
    rw [core.convert.num.FromI128I64.from_val_eq]; norm_num
  have hfromg : (core.convert.num.FromI128I64.from gm).val = gm.val :=
    core.convert.num.FromI128I64.from_val_eq gm
  have hfroms : (core.convert.num.FromI128I64.from s).val = s.val :=
    core.convert.num.FromI128I64.from_val_eq s
  obtain ⟨t, ht, htv⟩ := WP.spec_imp_exists
    (I128.add_spec (x := core.convert.num.FromI128I64.from (4294967296#i64))
      (y := core.convert.num.FromI128I64.from gm)
      (by rw [hfrom32, hfromg, hgmv']; simpa only [I128_min_eq] using hadd.1)
      (by rw [hfrom32, hfromg, hgmv']; simpa only [I128_max_eq] using hadd.2))
  have htv' : t.val = (2 : ℤ) ^ (32 : ℕ) + apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val) := by
    rw [htv, hfrom32, hfromg, hgmv']
  obtain ⟨p, hp, hpv⟩ := I128_checked_mul_some (core.convert.num.FromI128I64.from s) t
    (by rw [hfroms, htv']; simpa only [I128_min_eq] using hpmul.1)
    (by rw [hfroms, htv']; simpa only [I128_max_eq] using hpmul.2)
  have hpv' : p.val = s.val * ((2 : ℤ) ^ (32 : ℕ) +
      apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val)) := by rw [hpv, hfroms, htv']
  obtain ⟨y, hy, hyv⟩ := dense_round32_val p
    (by rw [hpv']; simpa only [I128_min_eq] using hround.1)
    (by rw [hpv']; simpa only [I128_max_eq] using hround.2)
  have hyv' : y.val = apeiron_norm.round32 (s.val * ((2 : ℤ) ^ (32 : ℕ) +
      apeiron_norm.mulRound g.val ((2 : ℤ) ^ (32 : ℕ) - s.val))) := by rw [hyv, hpv']
  have hv1 : solve_port.q32.narrow y = ok (some (IScalar.cast IScalarTy.I64 y)) :=
    dense_narrow_some y (by rw [hyv']; exact not_lt.2 hfit.2) (by rw [hyv']; exact not_lt.2 hfit.1)
  refine ⟨IScalar.cast IScalarTy.I64 y, ?_, ?_⟩
  · simp only [learn_port.dense.silu_deriv, engine_ONE_eq, lift, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch, dense_ov_some, hom, hgm, ht, hp, hy, hv1]
  · rw [cast_I64_val y (by rw [hyv']; exact hfit.1) (by rw [hyv']; exact hfit.2), hyv',
      apeiron_norm.siluDeriv, apeiron_norm.q32]

/-! ## 5. `swiglu_bwd`

The paired loop of `swiglu_bwd`: one `swiglu_one_bwd` per element, pushing into `dgate` and `dup`
together. The callee `swiglu_one_bwd` is taken as an explicit contract (its own refinement
`swiglu_one_bwd_spec` is section 3). -/

/-- The pure paired value model of the `swiglu_bwd` loop: after `j` steps the two vectors hold the
first `j` `(dgate, dup)` callee pairs. -/
def swigluBwdVals (dg du : List I64) : ℕ → List (I64 × I64)
  | 0 => []
  | (j + 1) => swigluBwdVals dg du j ++ [(dg.getD j default, du.getD j default)]

theorem swigluBwdVals_step (dg du : List I64) (j : ℕ) :
    swigluBwdVals dg du (j + 1) = swigluBwdVals dg du j ++
      [(dg.getD j default, du.getD j default)] := rfl

theorem swigluBwdVals_length (dg du : List I64) (j : ℕ) : (swigluBwdVals dg du j).length = j := by
  induction j with
  | zero => rfl
  | succ k ih => rw [swigluBwdVals_step, List.length_append, List.length_singleton, ih]

/-- The recursive paired model is the `zip` prefix. -/
theorem swigluBwdVals_eq_zip (dg du : List I64) (j : ℕ) (h1 : j ≤ dg.length)
    (h2 : j ≤ du.length) :
    swigluBwdVals dg du j = (dg.zip du).take j := by
  induction j with
  | zero => rfl
  | succ k ih =>
      rw [swigluBwdVals_step, ih (by omega) (by omega), List.take_add_one]
      congr 1
      have hk1 : k < dg.length := by omega
      have hk2 : k < du.length := by omega
      have hz : (dg.zip du)[k]? = some (dg.getD k default, du.getD k default) := by
        rw [List.getElem?_zip_eq_some]
        exact ⟨by rw [List.getElem?_eq_getElem hk1, List.getD_eq_getElem dg default hk1],
          by rw [List.getElem?_eq_getElem hk2, List.getD_eq_getElem du default hk2]⟩
      rw [hz]
      rfl

/-- The full-length form of `swigluBwdVals_eq_zip`. -/
theorem swigluBwdVals_eq_zip_len (dg du : List I64) (j : ℕ) (h1 : dg.length = j)
    (h2 : du.length = j) : swigluBwdVals dg du j = dg.zip du := by
  rw [swigluBwdVals_eq_zip dg du j (by omega) (by omega),
    List.take_of_length_le (by rw [List.length_zip]; omega)]

/-- One `swiglu_bwd_loop` iteration with `ok1 = true` and a successful `swiglu_one_bwd` advances
both models by one pair and returns `cont`. -/
theorem swiglu_bwd_loop_body_step
    (gate up dy : Slice I64) (dg du : List I64) (n : Usize)
    (dgate dup : alloc.vec.Vec I64) (k : Usize) (err : DenseError)
    (hn : n.val = gate.val.length) (hup : up.val.length = n.val) (hdy : dy.val.length = n.val)
    (hk : k.val < n.val) (hnumax : k.val + 1 ≤ Usize.max)
    (hinv1 : dgate.val.map (fun x => x.val) = (swigluBwdVals dg du k.val).map (fun p => p.1.val))
    (hinv2 : dup.val.map (fun x => x.val) = (swigluBwdVals dg du k.val).map (fun p => p.2.val))
    (hkbound : k.val + 1 ≤ U32.max)
    (hone : learn_port.dense.swiglu_one_bwd (gate.val.getD k.val default) (up.val.getD k.val default)
        (dy.val.getD k.val default) = ok (core.result.Result.Ok
          (dg.getD k.val default, du.getD k.val default))) :
    ∃ dgate' dup' k', learn_port.dense.swiglu_bwd_loop.body gate up dy n dgate dup k err true
        = ok (ControlFlow.cont (dgate', dup', k', err, true)) ∧
      k'.val = k.val + 1 ∧
      dgate'.val.map (fun x => x.val) = (swigluBwdVals dg du k'.val).map (fun p => p.1.val) ∧
      dup'.val.map (fun x => x.val) = (swigluBwdVals dg du k'.val).map (fun p => p.2.val) := by
  have hklt_gate : k.val < gate.val.length := by omega
  have hklt_up : k.val < up.val.length := by omega
  have hklt_dy : k.val < dy.val.length := by omega
  have hgate : Slice.index_usize gate k = ok (gate.val.getD k.val default) :=
    apeiron_gram.slice_index_getD gate k hklt_gate
  have hup' : Slice.index_usize up k = ok (up.val.getD k.val default) :=
    apeiron_gram.slice_index_getD up k hklt_up
  have hdy' : Slice.index_usize dy k = ok (dy.val.getD k.val default) :=
    apeiron_gram.slice_index_getD dy k hklt_dy
  have hkn : k < n := (UScalar.lt_equiv k n).2 hk
  have hlen1 : dgate.val.length = k.val := by
    have hh := congrArg List.length hinv1
    simp only [List.length_map, swigluBwdVals_length] at hh
    exact hh
  have hlen2 : dup.val.length = k.val := by
    have hh := congrArg List.length hinv2
    simp only [List.length_map, swigluBwdVals_length] at hh
    exact hh
  have hbound : dgate.val.length + 1 ≤ U32.max := by rw [hlen1]; exact hkbound
  obtain ⟨dgate', hpush1, hdg'⟩ := alloc.vec.Vec.push_val dgate (dg.getD k.val default) hbound
  have hbound2 : dup.val.length + 1 ≤ U32.max := by rw [hlen2]; exact hkbound
  obtain ⟨dup', hpush2, hdu'⟩ := alloc.vec.Vec.push_val dup (du.getD k.val default) hbound2
  obtain ⟨k', hk', hk'v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by
      show k.val + (1#usize).val ≤ Usize.max
      rw [show (1#usize).val = 1 from rfl]
      exact hnumax))
  refine ⟨dgate', dup', k', ?_, hk'v, ?_, ?_⟩
  · rw [learn_port.dense.swiglu_bwd_loop.body, if_pos hkn, if_pos rfl]
    simp only [hgate, hup', hdy', hone, hpush1, hpush2, hk', bind_tc_ok, uncurry_apply_pair]
  · rw [hk'v, show (↑k + ↑1#usize) = ↑k + 1 from rfl, swigluBwdVals_step, List.map_append, ← hinv1,
      hdg']
    simp only [List.map_append, List.map_cons, List.map_nil]
  · rw [hk'v, show (↑k + ↑1#usize) = ↑k + 1 from rfl, swigluBwdVals_step, List.map_append, ← hinv2,
      hdu']
    simp only [List.map_append, List.map_cons, List.map_nil]

/-- Specification of `swiglu_bwd_loop` on the all-succeed domain. -/
theorem swiglu_bwd_loop_spec
    (gate up dy : Slice I64) (dg du : List I64) (n : Usize)
    (dgate0 dup0 : alloc.vec.Vec I64)
    (hn : n.val = gate.val.length) (hup : up.val.length = n.val) (hdy : dy.val.length = n.val)
    (hmax : n.val ≤ Usize.max)
    (hnmax : n.val + 1 ≤ U32.max)
    (hone : ∀ k, k < n.val →
      learn_port.dense.swiglu_one_bwd (gate.val.getD k default) (up.val.getD k default)
        (dy.val.getD k default) = ok (core.result.Result.Ok
          (dg.getD k default, du.getD k default)))
    (hinv01 : dgate0.val.map (fun x => x.val) = (swigluBwdVals dg du 0).map (fun p => p.1.val))
    (hinv02 : dup0.val.map (fun x => x.val) = (swigluBwdVals dg du 0).map (fun p => p.2.val)) :
    WP.spec (learn_port.dense.swiglu_bwd_loop gate up dy n dgate0 dup0 0#usize
        DenseError.Overflow true)
      (fun b => b.2.2.2 = true ∧
        b.1.val.map (fun x => x.val) = (swigluBwdVals dg du n.val).map (fun p => p.1.val) ∧
        b.2.1.val.map (fun x => x.val) = (swigluBwdVals dg du n.val).map (fun p => p.2.val)) := by
  unfold learn_port.dense.swiglu_bwd_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × alloc.vec.Vec I64 × Usize × DenseError × Bool =>
      n.val - s.2.2.1.val)
    (inv := fun s => s.2.2.1.val ≤ n.val ∧ s.2.2.2.2 = true ∧
      s.1.val.map (fun x => x.val) = (swigluBwdVals dg du s.2.2.1.val).map (fun p => p.1.val) ∧
      s.2.1.val.map (fun x => x.val) = (swigluBwdVals dg du s.2.2.1.val).map (fun p => p.2.val))
    (post := fun b : alloc.vec.Vec I64 × alloc.vec.Vec I64 × DenseError × Bool =>
      b.2.2.2 = true ∧
      b.1.val.map (fun x => x.val) = (swigluBwdVals dg du n.val).map (fun p => p.1.val) ∧
      b.2.1.val.map (fun x => x.val) = (swigluBwdVals dg du n.val).map (fun p => p.2.val))
  · rintro ⟨dgate', dup', k', err', ok''⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hkle, hok, hmodel1, hmodel2⟩ := hinv
    rw [hok]
    by_cases hklt : k'.val < n.val
    · have hnumax : k'.val + 1 ≤ Usize.max := by omega
      have hkbound : k'.val + 1 ≤ U32.max := by omega
      obtain ⟨dgate'', dup'', k'', hexec, hk''v, hmodel1'', hmodel2''⟩ :=
        swiglu_bwd_loop_body_step gate up dy dg du n dgate' dup' k' err' hn hup hdy hklt hnumax
          hmodel1 hmodel2 hkbound (hone k'.val hklt)
      rw [hexec]
      refine ⟨⟨?_, rfl, hmodel1'', hmodel2''⟩, ?_⟩
      · rw [hk''v]; omega
      · dsimp only; rw [hk''v]; omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.swiglu_bwd_loop.body gate up dy n dgate' dup' k' err' true
          = ok (ControlFlow.done (dgate', dup', err', true)) := by
        rw [learn_port.dense.swiglu_bwd_loop.body, if_neg hnot]
      simp only [hbody]
      have hk'n : k'.val = n.val := by omega
      exact ⟨rfl, by rw [← hk'n]; exact hmodel1, by rw [← hk'n]; exact hmodel2⟩
  · refine ⟨by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega,
      rfl, hinv01, hinv02⟩

/-- **`swiglu_bwd`.** On the all-succeed domain the extracted body returns the `Ok` pair and the two
value lists are the paired model's projections. -/
theorem swiglu_bwd_spec (gate up dy : Slice I64) (dg du : List I64)
    (hne : gate.val.length ≠ 0)
    (hup : up.val.length = gate.val.length)
    (hdy : dy.val.length = gate.val.length)
    (hdg : dg.length = gate.val.length)
    (hdu : du.length = gate.val.length)
    (hnmax : gate.val.length + 1 ≤ U32.max)
    (hone : ∀ k, k < gate.val.length →
      learn_port.dense.swiglu_one_bwd (gate.val.getD k default) (up.val.getD k default)
        (dy.val.getD k default) = ok (core.result.Result.Ok
          (dg.getD k default, du.getD k default))) :
    ∃ dgate dup : alloc.vec.Vec I64,
      learn_port.dense.swiglu_bwd gate up dy = ok (core.result.Result.Ok (dgate, dup)) ∧
      dgate.val.map (fun x => x.val) = (dg.zip du).map (fun p => p.1.val) ∧
      dup.val.map (fun x => x.val) = (dg.zip du).map (fun p => p.2.val) := by
  have hn : (Slice.len gate).val = gate.val.length := Slice.len_val gate
  have hnz : ¬ (Slice.len gate = 0#usize) := by
    intro h
    apply hne
    have hv := congrArg (fun u : Usize => u.val) h
    rw [hn, show (0#usize).val = 0 from rfl] at hv
    exact hv
  have hupv : (Slice.len up).val = up.val.length := Slice.len_val up
  have hdyv : (Slice.len dy).val = dy.val.length := Slice.len_val dy
  have hshape_up : ¬ (Slice.len up != Slice.len gate) := by
    rw [bne_iff_ne]
    intro h
    exact h (UScalar.eq_of_val_eq (by rw [hupv, hn, hup]))
  have hshape_dy : ¬ (Slice.len dy != Slice.len gate) := by
    rw [bne_iff_ne]
    intro h
    exact h (UScalar.eq_of_val_eq (by rw [hdyv, hn, hdy]))
  have hcap1 : (alloc.vec.Vec.with_capacity I64 (Slice.len gate)).val = [] := rfl
  have hlenc0 : (alloc.vec.Vec.with_capacity I64 (Slice.len gate)).val.length =
      (alloc.vec.Vec.with_capacity I64 (Slice.len gate)).val.length := rfl
  have hloop := swiglu_bwd_loop_spec gate up dy dg du (Slice.len gate)
    (alloc.vec.Vec.with_capacity I64 (Slice.len gate))
    (alloc.vec.Vec.with_capacity I64 (Slice.len gate))
    hn (hup.trans hn.symm) (hdy.trans hn.symm) (by rw [hn]; exact Slice.property gate)
    (by rw [hn]; exact hnmax)
    (by intro k hk; exact hone k (by rw [hn] at hk; exact hk)) (by rw [hcap1]; rfl) (by rw [hcap1]; rfl)
  rw [learn_port.dense.swiglu_bwd, if_neg hnz, if_neg hshape_up, if_neg hshape_dy]
  dsimp (zeta := true) only
  obtain ⟨b, hbeq, hbpost⟩ := WP.spec_imp_exists hloop
  rcases b with ⟨wdg, wdu, we, wok⟩
  obtain ⟨hwok, hwmodel1, hwmodel2⟩ := hbpost
  have hwok' : wok = true := hwok
  rw [hbeq]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [if_pos hwok']
  refine ⟨wdg, wdu, rfl, ?_, ?_⟩
  · rw [hwmodel1, swigluBwdVals_eq_zip_len dg du (Slice.len gate).val (by rw [hn]; exact hdg)
      (by rw [hn]; exact hdu)]
  · rw [hwmodel2, swigluBwdVals_eq_zip_len dg du (Slice.len gate).val (by rw [hn]; exact hdg)
      (by rw [hn]; exact hdu)]

/-! ## 6. `rmsnorm_fwd`

The gain loop `y_i = round(g_i · round(x_i · r))` over the engine's normalised row. The engine callee
`norm.rms_norm_fixed` is an explicit contract: the loop reuses the `swigluVals` value model (the same
`zipWith mulRound` prefix). -/

/-- One `rmsnorm_fwd_loop` iteration with `ok1 = true` and a successful `mul_round` advances the
model by one. -/
theorem rmsnorm_fwd_loop_body_step
    (gamma : Slice I64) (v : alloc.vec.Vec I64) (n : Usize)
    (out : alloc.vec.Vec I64) (k : Usize)
    (hn : n.val = gamma.val.length) (hvn : v.val.length = n.val)
    (hk : k.val < n.val) (hnumax : k.val + 1 ≤ Usize.max)
    (hinv : out.val.map (fun x => x.val) = swigluVals gamma.val v.val k.val)
    (hbound : out.val.length + 1 ≤ U32.max)
    (w : I64)
    (hmul : learn_port.dense.mul_round (gamma.val.getD k.val default) (v.val.getD k.val default)
      = ok (some w))
    (hmulv : w.val = apeiron_norm.mulRound (gamma.val.getD k.val default).val
      (v.val.getD k.val default).val) :
    ∃ out' k', learn_port.dense.rmsnorm_fwd_loop.body gamma n v out k true
        = ok (ControlFlow.cont (out', k', true)) ∧
      k'.val = k.val + 1 ∧ out'.val.map (fun x => x.val) = swigluVals gamma.val v.val k'.val := by
  have hklt_gamma : k.val < gamma.val.length := by omega
  have hklt_v : k.val < v.val.length := by omega
  have hgamma : Slice.index_usize gamma k = ok (gamma.val.getD k.val default) :=
    apeiron_gram.slice_index_getD gamma k hklt_gamma
  have hnormed : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice I64) v k
      = ok (v.val.getD k.val default) :=
    apeiron_gram.vec_index_getD v k hklt_v
  have hkn : k < n := (UScalar.lt_equiv k n).2 hk
  obtain ⟨out', hpush, hout'⟩ := alloc.vec.Vec.push_val out w hbound
  obtain ⟨k', hk', hk'v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by
      show k.val + (1#usize).val ≤ Usize.max
      rw [show (1#usize).val = 1 from rfl]
      exact hnumax))
  refine ⟨out', k', ?_, hk'v, ?_⟩
  · rw [learn_port.dense.rmsnorm_fwd_loop.body, if_pos hkn, if_pos rfl]
    simp only [hgamma, hnormed, hmul, hpush, hk', bind_tc_ok, uncurry_apply_pair]
  · rw [hk'v, show (↑k + ↑1#usize) = ↑k + 1 from rfl, swigluVals_step, ← hinv, hout',
      List.map_append, List.map_cons, List.map_nil, hmulv]

/-- Specification of `rmsnorm_fwd_loop` on the all-succeed domain. -/
theorem rmsnorm_fwd_loop_spec
    (gamma : Slice I64) (n : Usize) (v : alloc.vec.Vec I64) (out0 : alloc.vec.Vec I64)
    (hn : n.val = gamma.val.length) (hvn : v.val.length = n.val)
    (hmax : n.val ≤ Usize.max) (hnmax : n.val + 1 ≤ U32.max)
    (hfit : ∀ k, k < n.val → -9223372036854775808 ≤
        apeiron_norm.mulRound (gamma.val.getD k default).val (v.val.getD k default).val ∧
        apeiron_norm.mulRound (gamma.val.getD k default).val (v.val.getD k default).val
          ≤ 9223372036854775807)
    (hinv0 : out0.val.map (fun x => x.val) = swigluVals gamma.val v.val 0) :
    WP.spec (learn_port.dense.rmsnorm_fwd_loop gamma n v out0 0#usize true)
      (fun b => b.2 = true ∧ b.1.val.map (fun x => x.val) = swigluVals gamma.val v.val n.val) := by
  unfold learn_port.dense.rmsnorm_fwd_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool => n.val - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ n.val ∧ s.2.2 = true ∧
      s.1.val.map (fun x => x.val) = swigluVals gamma.val v.val s.2.1.val)
    (post := fun b : alloc.vec.Vec I64 × Bool =>
      b.2 = true ∧ b.1.val.map (fun x => x.val) = swigluVals gamma.val v.val n.val)
  · rintro ⟨out', k', ok''⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hkle, hok, hmodel⟩ := hinv
    rw [hok]
    by_cases hklt : k'.val < n.val
    · have hnumax : k'.val + 1 ≤ Usize.max := by omega
      have hlen' : out'.val.length = k'.val := by
        have hh := congrArg List.length hmodel
        rw [List.length_map, swigluVals_length] at hh
        exact hh
      have hbound : out'.val.length + 1 ≤ U32.max := by omega
      obtain ⟨w, hmul, hmulv⟩ := dense_mul_round_some (gamma.val.getD k'.val default)
        (v.val.getD k'.val default) (hfit k'.val hklt)
      obtain ⟨out'', k'', hexec, hk''v, hmodel''⟩ := rmsnorm_fwd_loop_body_step gamma v n out'
        k' hn hvn hklt hnumax hmodel hbound w hmul hmulv
      rw [hexec]
      refine ⟨⟨?_, rfl, hmodel''⟩, ?_⟩
      · rw [hk''v]; omega
      · dsimp only; rw [hk''v]; omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.rmsnorm_fwd_loop.body gamma n v out' k' true
          = ok (ControlFlow.done (out', true)) := by
        rw [learn_port.dense.rmsnorm_fwd_loop.body, if_neg hnot]
      simp only [hbody]
      have hk'n : k'.val = n.val := by omega
      exact ⟨rfl, by rw [← hk'n]; exact hmodel⟩
  · exact ⟨by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega,
      rfl, hinv0⟩

/-- **`rmsnorm_fwd`.** On the all-succeed domain the extracted body returns the `Ok` vector and its
value list is `rmsFwd gamma.val x.val r`, for the engine row `v` and root `r`. -/
theorem rmsnorm_fwd_spec (x gamma : Slice I64) (eps : I64) (v : alloc.vec.Vec I64) (r : I64)
    (hne : x.val.length ≠ 0)
    (hlen : gamma.val.length = x.val.length)
    (hnmax : x.val.length + 1 ≤ U32.max)
    (hrm : norm.rms_norm_fixed x eps = ok (core.result.Result.Ok v))
    (hrmv : v.val.map (fun y => y.val) = apeiron_norm.rmsNorm (x.val.map (fun y => y.val)) r)
    (hfit : ∀ k, k < x.val.length → -9223372036854775808 ≤
        apeiron_norm.mulRound (gamma.val.getD k default).val (v.val.getD k default).val ∧
        apeiron_norm.mulRound (gamma.val.getD k default).val (v.val.getD k default).val
          ≤ 9223372036854775807) :
    ∃ out : alloc.vec.Vec I64, learn_port.dense.rmsnorm_fwd x gamma eps
        = ok (core.result.Result.Ok out) ∧
      out.val.map (fun y => y.val) =
        apeiron_norm.rmsFwd (gamma.val.map (fun y => y.val)) (x.val.map (fun y => y.val)) r := by
  have hn : (Slice.len x).val = x.val.length := Slice.len_val x
  have hnz : ¬ (Slice.len x = 0#usize) := by
    intro h
    apply hne
    have hv := congrArg (fun u : Usize => u.val) h
    rw [hn, show (0#usize).val = 0 from rfl] at hv
    exact hv
  have hgv : (Slice.len gamma).val = gamma.val.length := Slice.len_val gamma
  have hshape : ¬ (Slice.len gamma != Slice.len x) := by
    rw [bne_iff_ne]
    intro h
    exact h (UScalar.eq_of_val_eq (by rw [hgv, hn, hlen]))
  have hcap : (alloc.vec.Vec.with_capacity I64 (Slice.len x)).val = [] := rfl
  have hvlen : v.val.length = x.val.length := by
    have hh := congrArg List.length hrmv
    simp only [List.length_map, apeiron_norm.rmsNorm] at hh
    exact hh
  have hloop := rmsnorm_fwd_loop_spec gamma (Slice.len x) v
    (alloc.vec.Vec.with_capacity I64 (Slice.len x))
    (hn.trans hlen.symm) (by rw [hn]; exact hvlen)
    (by rw [hn]; exact Slice.property x) (by rw [hn]; exact hnmax)
    (by intro k hk; exact hfit k hk) (by rw [hcap]; rfl)
  rw [learn_port.dense.rmsnorm_fwd, if_neg hnz, if_neg hshape]
  dsimp (zeta := true) only
  rw [hrm]
  simp only [bind_tc_ok]
  obtain ⟨b, hbeq, hbpost⟩ := WP.spec_imp_exists hloop
  rcases b with ⟨wo, wok⟩
  obtain ⟨hwok, hwmodel⟩ := hbpost
  have hwok' : wok = true := hwok
  rw [hbeq]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [if_pos hwok']
  refine ⟨wo, rfl, ?_⟩
  rw [hwmodel, swigluVals_eq_zipWith_len gamma.val v.val (Slice.len x).val
      (by rw [hn]; exact hlen) (by rw [hn]; exact hvlen)]
  unfold apeiron_norm.rmsFwd
  rw [← hrmv, List.zipWith_map_left, List.zipWith_map_right]

/-! ## 7. `inv_rms`

The RMS inverse-scale pass. The loop `inv_rms_loop` accumulates the checked `i128` sum of squares;
the epilogue forms `(Σxᵢ² + n·2³¹)/(n·2³²) + eps` in Q32.32 and calls the engine's `fixed_rsqrt` at
that argument. The `usize → u64 / i64` casts of the length and the `fixed_rsqrt` callee contract are
the explicit domain. -/

/-- The `u64` reinterpretation of a `usize` length is its value. -/
theorem norm_cast_U64_val (n : Usize) : (UScalar.cast UScalarTy.U64 n).val = n.val := by
  rw [UScalar.cast_val_eq, UScalarTy.U64_numBits_eq]
  apply Nat.mod_eq_of_lt
  have h : n.val < 2 ^ UScalarTy.Usize.numBits := UScalar.hBounds n
  have hu : UScalarTy.Usize.numBits ≤ 64 := by
    rw [UScalarTy.Usize_numBits_eq]
    cases System.Platform.numBits_eq <;> simp_all
  calc n.val < 2 ^ UScalarTy.Usize.numBits := h
    _ ≤ 2 ^ (64 : Nat) := Nat.pow_le_pow_right (by norm_num) hu

/-- The `u64` widening of `i64::MAX` is `2⁶³ − 1`. -/
theorem norm_hcast_I64MAX_val :
    (IScalar.hcast UScalarTy.U64 core.num.I64.MAX).val = 9223372036854775807 := by
  rw [IScalar.hcast_val_eq, UScalarTy.U64_numBits_eq]
  simp only [core.num.I64.MAX, I64.rMax, IScalar.ofInt_val_eq]
  norm_num
  exact_mod_cast Int.toNat_of_nonneg (by norm_num : (0 : ℤ) ≤ 9223372036854775807)

/-- The `i64` narrowing of a `usize` in `i64` range is its value. -/
theorem norm_hcast_I64_val (n : Usize) (hn : n.val ≤ 9223372036854775807) :
    (UScalar.hcast IScalarTy.I64 n).val = n.val := by
  rw [UScalar.hcast_val_eq, IScalarTy.I64_numBits_eq]
  exact Aeneas.Arith.Int.bmod_pow2_eq_of_inBounds 63 n.val (by omega) (by omega)

/-- The `u64` range guard of `inv_rms`, as a value comparison. -/
theorem norm_guard_iff (n : Usize) :
    (UScalar.cast UScalarTy.U64 n > IScalar.hcast UScalarTy.U64 core.num.I64.MAX) ↔
      9223372036854775807 < n.val := by
  rw [gt_iff_lt, UScalar.lt_equiv, norm_cast_U64_val, norm_hcast_I64MAX_val]

/-- One `inv_rms_loop` iteration with `ok1 = true` and the checked square/add inside `i128`
advances `sqPrefix` by one and returns `cont`. -/
theorem inv_rms_loop_body_step (x : Slice I64) (n : Usize) (sum_sq : I128) (k : Usize)
    (hk : k.val < n.val) (hn : n.val = x.val.length)
    (hsum : sum_sq.val = sqPrefix x k.val)
    (hfit : sqPrefix x k.val + (x.val.getD k.val default).val * (x.val.getD k.val default).val
      ≤ I128.max) :
    ∃ total k2, learn_port.dense.inv_rms_loop.body x n sum_sq k true
        = ok (ControlFlow.cont (total, k2, true)) ∧
      total.val = sqPrefix x k2.val ∧ k2.val = k.val + 1 := by
  have hklt_x : k.val < x.val.length := by omega
  have hkn : k < n := (UScalar.lt_equiv k n).2 hk
  have hget : x.val.getD k.val default = x.val[k.val] := List.getD_eq_getElem x.val default hklt_x
  have hxv : (x.val.getD k.val default).val = x.val[k.val].val := by rw [hget]
  have hi : Slice.index_usize x k = ok (x.val.getD k.val default) :=
    apeiron_gram.slice_index_getD x k hklt_x
  have hvv : (core.convert.num.FromI128I64.from (x.val.getD k.val default)).val =
      (x.val.getD k.val default).val := core.convert.num.FromI128I64.from_val_eq _
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from (x.val.getD k.val default)).val *
      (core.convert.num.FromI128I64.from (x.val.getD k.val default)).val := by
    rw [hvv, I128_min_eq]; nlinarith [mul_self_nonneg (x.val.getD k.val default).val]
  have hmulhi : (core.convert.num.FromI128I64.from (x.val.getD k.val default)).val *
      (core.convert.num.FromI128I64.from (x.val.getD k.val default)).val ≤ I128.max := by
    have habs := abs_le.mp (mul_abs_le_126 (i64_abs_le (x.val.getD k.val default))
      (i64_abs_le (x.val.getD k.val default)))
    rw [hvv, I128_max_eq]; nlinarith [habs.2]
  obtain ⟨sq, hsq, hsqv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hsumnn : 0 ≤ sum_sq.val := by rw [hsum]; exact sqPrefix_nonneg x k.val
  have hsuc : sum_sq.val + sq.val = sqPrefix x (k.val + 1) := by
    rw [hsqv, hvv, hget, hsum, ← sqPrefix_succ x hklt_x]
  have hfit' : sqPrefix x k.val + x.val[k.val].val * x.val[k.val].val ≤ I128.max := by
    rw [hget] at hfit; exact hfit
  have hsumhi : sum_sq.val + sq.val ≤ I128.max := by
    rw [hsuc, sqPrefix_succ x hklt_x]; exact hfit'
  obtain ⟨total, htot, htotv⟩ := I128_checked_add_some sum_sq sq
    (by rw [I128_min_eq]; nlinarith [hsumnn, sq_nonneg sq.val]) hsumhi
  have hiaddmax : k.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := x.property
    rw [h1]; omega
  obtain ⟨k2, hk2, hk2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := k) (y := 1#usize) hiaddmax)
  refine ⟨total, k2, ?_, ?_, hk2v⟩
  · rw [learn_port.dense.inv_rms_loop.body, if_pos hkn, if_pos rfl, hi]
    simp only [lift, bind_tc_ok]
    rw [hsq]
    dsimp only
    rw [htot]
    dsimp only
    rw [hk2]
    simp only [bind_tc_ok]
    rfl
  · rw [htotv, hsuc, hk2v]
    norm_num

/-- Specification of `inv_rms_loop` on the all-succeed domain. -/
theorem inv_rms_loop_spec (x : Slice I64) (n : Usize) (sum0 : I128)
    (hn : n.val = x.val.length)
    (hfit : ∀ k, k < n.val →
      sqPrefix x k + (x.val.getD k default).val * (x.val.getD k default).val ≤ I128.max)
    (hinv0 : sum0.val = sqPrefix x 0) :
    WP.spec (learn_port.dense.inv_rms_loop x n sum0 0#usize true)
      (fun b => b.2 = true ∧ b.1.val = sqPrefix x n.val) := by
  unfold learn_port.dense.inv_rms_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Usize × Bool => n.val - s.2.1.val)
    (inv := fun s => s.2.1.val ≤ n.val ∧ s.2.2 = true ∧ s.1.val = sqPrefix x s.2.1.val)
    (post := fun b : I128 × Bool => b.2 = true ∧ b.1.val = sqPrefix x n.val)
  · rintro ⟨sum_sq', k', ok''⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hkle, hok, hmodel⟩ := hinv
    rw [hok]
    by_cases hklt : k'.val < n.val
    · obtain ⟨total, k2, hexec, hmodel', hk2v⟩ :=
        inv_rms_loop_body_step x n sum_sq' k' hklt hn hmodel (hfit k'.val hklt)
      rw [hexec]
      refine ⟨⟨?_, rfl, hmodel'⟩, ?_⟩
      · rw [hk2v]; omega
      · dsimp only; rw [hk2v]; omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.inv_rms_loop.body x n sum_sq' k' true
          = ok (ControlFlow.done (sum_sq', true)) := by
        rw [learn_port.dense.inv_rms_loop.body, if_neg hnot]
      simp only [hbody]
      have hk'n : k'.val = n.val := by omega
      exact ⟨rfl, by rw [← hk'n]; exact hmodel⟩
  · refine ⟨by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega,
      rfl, ?_⟩
    show sum0.val = sqPrefix x (0#usize).val
    rw [show (0#usize).val = 0 from rfl]
    exact hinv0

/-- **`inv_rms`.** On the all-succeed domain the extracted body forms the Q32.32 mean square
`(Σxᵢ² + n·2³¹)/(n·2³²) + eps`, narrows it and returns the engine `fixed_rsqrt` of that argument. -/
theorem inv_rms_spec (x : Slice I64) (eps : I64)
    (hne : x.val.length ≠ 0)
    (hguard : x.val.length ≤ 9223372036854775807)
    (hfit : ∀ k, k < x.val.length →
      sqPrefix x k + (x.val.getD k default).val * (x.val.getD k default).val ≤ I128.max)
    (hmul : I128.min ≤ (x.val.length : ℤ) * 2 ^ (32 : ℕ) ∧
        (x.val.length : ℤ) * 2 ^ (32 : ℕ) ≤ I128.max)
    (hadd1 : I128.min ≤ sqPrefix x x.val.length +
          ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2 ∧
        sqPrefix x x.val.length + ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2 ≤ I128.max)
    (hadd2 : I128.min ≤ apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val ∧
        apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val ≤ I128.max)
    (hnarrow : -9223372036854775808 ≤ apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val ∧
        apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val ≤ 9223372036854775807)
    (hrsqrt : ∀ (a1 : I64),
        a1.val = apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val →
        ∃ r, fixed_rsqrt.fixed_rsqrt a1 = ok (core.result.Result.Ok r)) :
    ∃ a1 r, learn_port.dense.inv_rms x eps = ok (core.result.Result.Ok r) ∧
      a1.val = apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val ∧
      fixed_rsqrt.fixed_rsqrt a1 = ok (core.result.Result.Ok r) := by
  have hn : (Slice.len x).val = x.val.length := Slice.len_val x
  have hnz : ¬ (Slice.len x = 0#usize) := by
    intro h
    apply hne
    have hv := congrArg (fun u : Usize => u.val) h
    rw [hn, show (0#usize).val = 0 from rfl] at hv
    exact hv
  have hguardC : ¬ (UScalar.cast UScalarTy.U64 (Slice.len x) >
      IScalar.hcast UScalarTy.U64 core.num.I64.MAX) := by
    rw [norm_guard_iff, hn]; exact not_lt.2 hguard
  have hloop := inv_rms_loop_spec x (Slice.len x) 0#i128 hn
      (by intro k hk; rw [hn] at hk; exact hfit k hk)
      (by rw [sqPrefix]; simp)
  obtain ⟨b, hbeq, hbpost⟩ := WP.spec_imp_exists hloop
  rcases b with ⟨sum_sq, ok1⟩
  obtain ⟨hok1, hsum_sq⟩ := hbpost
  rw [hn] at hsum_sq
  -- value facts
  have hndv : (UScalar.hcast IScalarTy.I64 (Slice.len x)).val = (Slice.len x).val :=
    norm_hcast_I64_val (Slice.len x) (by rw [hn]; exact hguard)
  have hfromnd : (core.convert.num.FromI128I64.from
      (UScalar.hcast IScalarTy.I64 (Slice.len x))).val = (x.val.length : ℤ) := by
    rw [core.convert.num.FromI128I64.from_val_eq, hndv, hn]
  have honev : ((4294967296#i64 : I64)).val = (2 : ℤ) ^ (32 : ℕ) := by norm_num
  have hfromone : (core.convert.num.FromI128I64.from (4294967296#i64)).val
      = (2 : ℤ) ^ (32 : ℕ) := by
    rw [core.convert.num.FromI128I64.from_val_eq, honev]
  have hfromeps : (core.convert.num.FromI128I64.from eps).val = eps.val :=
    core.convert.num.FromI128I64.from_val_eq eps
  obtain ⟨d, hd, hdv⟩ := I128_checked_mul_some
    (core.convert.num.FromI128I64.from (UScalar.hcast IScalarTy.I64 (Slice.len x)))
    (core.convert.num.FromI128I64.from (4294967296#i64))
    (by rw [hfromnd, hfromone]; exact hmul.1)
    (by rw [hfromnd, hfromone]; exact hmul.2)
  have hdv' : d.val = (x.val.length : ℤ) * 2 ^ (32 : ℕ) := by rw [hdv, hfromnd, hfromone]
  have hlz : (0 : ℤ) < (x.val.length : ℤ) := by
    exact_mod_cast Nat.pos_of_ne_zero hne
  have hdpos : 0 < d.val := by rw [hdv']; positivity
  obtain ⟨i5, hi5, hi5v⟩ := IScalar.div_spec (x := d) (y := (2#i128))
    (by norm_num) (by rintro ⟨h1, h2⟩; norm_num at h2)
  have hi5v' : i5.val = ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2 := by
    rw [hi5v, show ((2#i128 : I128)).val = 2 from rfl, hdv',
      Int.tdiv_eq_ediv_of_nonneg (a := (x.val.length : ℤ) * 2 ^ (32 : ℕ)) (by positivity)]
  have hnumpos : 0 ≤ sqPrefix x x.val.length + ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2 :=
    add_nonneg (sqPrefix_nonneg x x.val.length) (by positivity)
  obtain ⟨h, hh, hhv⟩ := I128_checked_add_some sum_sq i5
    (by rw [I128_min_eq, hsum_sq, hi5v']; nlinarith [hnumpos])
    (by rw [hsum_sq, hi5v']; exact hadd1.2)
  have hdpos' : d.val ≠ 0 := ne_of_gt hdpos
  obtain ⟨mean, hmeandiv, hmeanvv⟩ := IScalar.div_spec (x := h) (y := d)
    hdpos'
    (by rintro ⟨h1, h2⟩; rw [hdv'] at h2; omega)
  have hmeanv' : mean.val = (sqPrefix x x.val.length + ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2) /
      ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) := by
    rw [hmeanvv, hhv, hsum_sq, hi5v', hdv',
      Int.tdiv_eq_ediv_of_nonneg (a := sqPrefix x x.val.length +
        ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2) hnumpos]
  -- the mean-square bridge to `rmsMean`
  have hsum_map : ((x.val.map (fun y => y.val)).map (fun xi => xi * xi)).sum =
      sqPrefix x x.val.length := by
    rw [sqPrefix_total, List.map_map]
    rfl
  have hmeanRound : apeiron_norm.meanRound
        (((x.val.map (fun y => y.val)).map (fun xi => xi * xi)).sum)
        ((x.val.length : ℤ) * apeiron_norm.q32) = mean.val := by
    rw [hsum_map]
    unfold apeiron_norm.meanRound apeiron_norm.rdiv apeiron_norm.q32
    rw [Int.tdiv_eq_ediv_of_nonneg (a := (x.val.length : ℤ) * 2 ^ (32 : ℕ)) (by positivity)]
    rw [hmeanv']
    exact Int.tdiv_eq_ediv_of_nonneg (a := sqPrefix x x.val.length +
      ((x.val.length : ℤ) * 2 ^ (32 : ℕ)) / 2) hnumpos
  have hme : mean.val + eps.val = apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val := by
    rw [apeiron_norm.rmsMean, List.length_map, ← hmeanRound]
  obtain ⟨a, ha, hav⟩ := I128_checked_add_some mean (core.convert.num.FromI128I64.from eps)
    (by rw [hfromeps, hme]; exact hadd2.1)
    (by rw [hfromeps, hme]; exact hadd2.2)
  have ha_v : a.val = apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val := by
    rw [hav, hfromeps, hme]
  have hnarrow_some : solve_port.q32.narrow a = ok (some (IScalar.cast IScalarTy.I64 a)) :=
    dense_narrow_some a (by rw [ha_v]; exact not_lt.2 hnarrow.2)
      (by rw [ha_v]; exact not_lt.2 hnarrow.1)
  have ha1v : (IScalar.cast IScalarTy.I64 a).val =
      apeiron_norm.rmsMean (x.val.map (fun y => y.val)) eps.val := by
    rw [cast_I64_val a (by rw [ha_v]; exact hnarrow.1) (by rw [ha_v]; exact hnarrow.2)]
    exact ha_v
  obtain ⟨r, hrsqrt'⟩ := hrsqrt (IScalar.cast IScalarTy.I64 a) ha1v
  refine ⟨IScalar.cast IScalarTy.I64 a, r, ?_, ha1v, hrsqrt'⟩
  rw [learn_port.dense.inv_rms, if_neg hnz]
  simp only [lift, bind_tc_ok]
  rw [if_neg hguardC]
  rw [hbeq]
  simp only [bind_tc_ok, uncurry_apply_pair]
  rw [if_pos hok1]
  simp only [engine_ONE_eq, bind_tc_ok, hd, hi5, hh, hmeandiv,
    ha, hnarrow_some, hrsqrt']

/-! ## 8. `rmsnorm_bwd`

The backward pass. Loop 1 (`rmsnorm_bwd_loop0`) forms `a = mulRound gamma dy` and the checked `i128`
sum `acc = Σ a_j·x_j`; the epilogue rounds `acc` to the scalar `S`; loop 2 (`rmsnorm_bwd_loop1`)
forms the per-element `dx = p − q` and `dgamma`. The `inv_rms` root is a callee contract. -/

/-- The `a = mulRound gamma dy` prefix of `rmsnorm_bwd` loop 0 (Q32.32 values). -/
def bwdAVals (gamma dy : List I64) : ℕ → List ℤ
  | 0 => []
  | (j + 1) => bwdAVals gamma dy j ++ [bwdA (gamma.getD j default).val (dy.getD j default).val]

theorem bwdAVals_step (gamma dy : List I64) (j : ℕ) :
    bwdAVals gamma dy (j + 1) =
      bwdAVals gamma dy j ++ [bwdA (gamma.getD j default).val (dy.getD j default).val] := rfl

theorem bwdAVals_length (gamma dy : List I64) (j : ℕ) : (bwdAVals gamma dy j).length = j := by
  induction j with
  | zero => rfl
  | succ k ih => rw [bwdAVals_step, List.length_append, List.length_singleton, ih]

/-- The `Σ a_j·x_j` accumulator prefix of `rmsnorm_bwd` loop 0 (single `i128`, Q64.64). -/
def bwdASum (x gamma dy : List I64) : ℕ → ℤ
  | 0 => 0
  | (j + 1) => bwdASum x gamma dy j +
      bwdA (gamma.getD j default).val (dy.getD j default).val * (x.getD j default).val

theorem bwdASum_step (x gamma dy : List I64) (j : ℕ) :
    bwdASum x gamma dy (j + 1) = bwdASum x gamma dy j +
      bwdA (gamma.getD j default).val (dy.getD j default).val * (x.getD j default).val := rfl

/-- One `rmsnorm_bwd_loop0` iteration with `ok1 = true` and the checked `mulRound`, `i128` multiply
and `i128` add inside their windows advances both models by one. -/
theorem rmsnorm_bwd_loop0_body_step
    (x gamma dy : Slice I64) (n : Usize) (a : alloc.vec.Vec I64) (acc : I128) (k : Usize)
    (hg : gamma.val.length = n.val) (hdy : dy.val.length = n.val)
    (hx : x.val.length = n.val)
    (hk : k.val < n.val) (hnumax : k.val + 1 ≤ Usize.max) (hkbound : k.val + 1 ≤ U32.max)
    (hinv1 : a.val.map (fun y => y.val) = bwdAVals gamma.val dy.val k.val)
    (hinv2 : acc.val = bwdASum x.val gamma.val dy.val k.val)
    (av : I64)
    (hav : learn_port.dense.mul_round (gamma.val.getD k.val default) (dy.val.getD k.val default)
      = ok (some av))
    (havv : av.val = bwdA (gamma.val.getD k.val default).val (dy.val.getD k.val default).val)
    (hfitmul : I128.min ≤ av.val * (x.val.getD k.val default).val ∧
        av.val * (x.val.getD k.val default).val ≤ I128.max)
    (hfitadd : I128.min ≤ acc.val + av.val * (x.val.getD k.val default).val ∧
        acc.val + av.val * (x.val.getD k.val default).val ≤ I128.max) :
    ∃ a' acc' k', learn_port.dense.rmsnorm_bwd_loop0.body x gamma dy n a acc k true
        = ok (ControlFlow.cont (a', acc', k', true)) ∧
      k'.val = k.val + 1 ∧
      a'.val.map (fun y => y.val) = bwdAVals gamma.val dy.val k'.val ∧
      acc'.val = bwdASum x.val gamma.val dy.val k'.val := by
  have hklt_g : k.val < gamma.val.length := by omega
  have hklt_d : k.val < dy.val.length := by omega
  have hklt_x : k.val < x.val.length := by omega
  have hkn : k < n := (UScalar.lt_equiv k n).2 hk
  have hget_g : gamma.val.getD k.val default = gamma.val[k.val] :=
    List.getD_eq_getElem gamma.val default hklt_g
  have hget_d : dy.val.getD k.val default = dy.val[k.val] :=
    List.getD_eq_getElem dy.val default hklt_d
  have hget_x : x.val.getD k.val default = x.val[k.val] :=
    List.getD_eq_getElem x.val default hklt_x
  have hgv : av.val = bwdA gamma.val[k.val].val dy.val[k.val].val := by rw [← hget_g, ← hget_d]; exact havv
  have hxv : (x.val.getD k.val default).val = x.val[k.val].val := by rw [hget_x]
  have hlen1 : a.val.length = k.val := by
    have hh := congrArg List.length hinv1
    simp only [List.length_map, bwdAVals_length] at hh
    exact hh
  have hbound : a.val.length + 1 ≤ U32.max := by rw [hlen1]; exact hkbound
  obtain ⟨a', hpush, ha'⟩ := alloc.vec.Vec.push_val a av hbound
  have hfrom_av : (core.convert.num.FromI128I64.from av).val = av.val :=
    core.convert.num.FromI128I64.from_val_eq av
  have hfrom_x : (core.convert.num.FromI128I64.from (x.val.getD k.val default)).val =
      (x.val.getD k.val default).val := core.convert.num.FromI128I64.from_val_eq _
  obtain ⟨p, hp, hpv⟩ := I128_checked_mul_some
    (core.convert.num.FromI128I64.from av) (core.convert.num.FromI128I64.from (x.val.getD k.val default))
    (by rw [hfrom_av, hfrom_x]; exact hfitmul.1)
    (by rw [hfrom_av, hfrom_x]; exact hfitmul.2)
  have hpv' : p.val = av.val * (x.val.getD k.val default).val := by rw [hpv, hfrom_av, hfrom_x]
  obtain ⟨acc', hadd, hacc'v⟩ := I128_checked_add_some acc p
    (by rw [hpv']; exact hfitadd.1)
    (by rw [hpv']; exact hfitadd.2)
  have hacc'' : acc'.val = acc.val + av.val * (x.val.getD k.val default).val := by rw [hacc'v, hpv']
  have hiaddmax : k.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := x.property
    rw [h1]; omega
  obtain ⟨k', hk', hk'v⟩ := WP.spec_imp_exists (Usize.add_spec (x := k) (y := 1#usize) hiaddmax)
  have hig : Slice.index_usize gamma k = ok (gamma.val.getD k.val default) :=
    apeiron_gram.slice_index_getD gamma k hklt_g
  have hidy : Slice.index_usize dy k = ok (dy.val.getD k.val default) :=
    apeiron_gram.slice_index_getD dy k hklt_d
  have hix : Slice.index_usize x k = ok (x.val.getD k.val default) :=
    apeiron_gram.slice_index_getD x k hklt_x
  refine ⟨a', acc', k', ?_, hk'v, ?_, ?_⟩
  · rw [learn_port.dense.rmsnorm_bwd_loop0.body, if_pos hkn, if_pos rfl]
    rw [hig, hidy, hix]
    simp only [hav, hp, hadd, hpush, hk', bind_tc_ok, lift]
    have hinner : (do let (i5, b) := (acc', true); ok (a', i5, b))
        = ok (a', acc', true) := rfl
    rw [hinner]
    simp only [bind_tc_ok]
    rfl
  · rw [ha']
    simp only [List.map_append, List.map_cons, List.map_nil]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hk'v, h1]
    rw [bwdAVals_step, ← hinv1, havv]
  · have h1 : (1#usize).val = 1 := by norm_num
    rw [hk'v, h1, bwdASum_step, ← hinv2, ← havv, ← hacc'']

/-- Specification of `rmsnorm_bwd_loop0` on the all-succeed domain. -/
theorem rmsnorm_bwd_loop0_spec
    (x gamma dy : Slice I64) (n : Usize) (a0 : alloc.vec.Vec I64)
    (hg : gamma.val.length = n.val) (hdy : dy.val.length = n.val)
    (hx : x.val.length = n.val)
    (hmax : n.val ≤ Usize.max)
    (hnmax : n.val + 1 ≤ U32.max)
    (hfitA : ∀ k, k < n.val →
      -9223372036854775808 ≤ bwdA (gamma.val.getD k default).val (dy.val.getD k default).val ∧
        bwdA (gamma.val.getD k default).val (dy.val.getD k default).val ≤ 9223372036854775807)
    (hfitmul : ∀ k, k < n.val →
      I128.min ≤ bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
          (x.val.getD k default).val ∧
        bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
          (x.val.getD k default).val ≤ I128.max)
    (hfitadd : ∀ k, k < n.val →
      I128.min ≤ bwdASum x.val gamma.val dy.val k +
          bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
            (x.val.getD k default).val ∧
        bwdASum x.val gamma.val dy.val k +
          bwdA (gamma.val.getD k default).val (dy.val.getD k default).val *
            (x.val.getD k default).val ≤ I128.max)
    (hinv01 : a0.val.map (fun y => y.val) = bwdAVals gamma.val dy.val 0)
    (hinv02 : (0#i128).val = bwdASum x.val gamma.val dy.val 0) :
    WP.spec (learn_port.dense.rmsnorm_bwd_loop0 x gamma dy n a0 0#i128 0#usize true)
      (fun b => b.2.2 = true ∧ b.1.val.map (fun y => y.val) = bwdAVals gamma.val dy.val n.val ∧
        b.2.1.val = bwdASum x.val gamma.val dy.val n.val) := by
  unfold learn_port.dense.rmsnorm_bwd_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × I128 × Usize × Bool => n.val - s.2.2.1.val)
    (inv := fun s => s.2.2.1.val ≤ n.val ∧ s.2.2.2 = true ∧
      s.1.val.map (fun y => y.val) = bwdAVals gamma.val dy.val s.2.2.1.val ∧
      s.2.1.val = bwdASum x.val gamma.val dy.val s.2.2.1.val)
    (post := fun b : alloc.vec.Vec I64 × I128 × Bool =>
      b.2.2 = true ∧ b.1.val.map (fun y => y.val) = bwdAVals gamma.val dy.val n.val ∧
      b.2.1.val = bwdASum x.val gamma.val dy.val n.val)
  · rintro ⟨a', acc', k', ok''⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hkle, hok, hmodel1, hmodel2⟩ := hinv
    rw [hok]
    by_cases hklt : k'.val < n.val
    · have hnumax : k'.val + 1 ≤ Usize.max := by omega
      have hkbound : k'.val + 1 ≤ U32.max := by omega
      obtain ⟨av, hav, havv⟩ := dense_mul_round_some (gamma.val.getD k'.val default)
        (dy.val.getD k'.val default) (hfitA k'.val hklt)
      obtain ⟨a'', acc'', k'', hexec, hk''v, hmodel1'', hmodel2''⟩ :=
        rmsnorm_bwd_loop0_body_step x gamma dy n a' acc' k' hg hdy hx hklt hnumax hkbound
          hmodel1 hmodel2 av hav havv
          (by rw [havv]; exact hfitmul k'.val hklt)
          (by rw [hmodel2, havv]; exact hfitadd k'.val hklt)
      rw [hexec]
      refine ⟨⟨?_, rfl, hmodel1'', hmodel2''⟩, ?_⟩
      · rw [hk''v]; omega
      · dsimp only; rw [hk''v]; omega
    · have hnot : ¬ (k' < n) := fun hc => hklt ((UScalar.lt_equiv k' n).1 hc)
      have hbody : learn_port.dense.rmsnorm_bwd_loop0.body x gamma dy n a' acc' k' true
          = ok (ControlFlow.done (a', acc', true)) := by
        rw [learn_port.dense.rmsnorm_bwd_loop0.body, if_neg hnot]
      simp only [hbody]
      have hk'n : k'.val = n.val := by omega
      exact ⟨rfl, by rw [← hk'n]; exact hmodel1, by rw [← hk'n]; exact hmodel2⟩
  · exact ⟨by show (0#usize).val ≤ n.val; rw [show (0#usize).val = 0 from rfl]; omega,
      rfl, hinv01, hinv02⟩
end apeiron_grad
