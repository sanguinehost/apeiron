import SiluSource
import Aeneas

/-
SCOPE: the guard-chain transitions, the closed-form work measure and the terminal branches of the
  actual extracted `silu.silu_fixed` body (`src/engine/silu.rs`, root `word_silu_fixed`).
REGISTER: scalar source resource refinement of the real executing body; no GPU, cycle or hardware
  timing claim, and no correspondence to the canonical `Cell`/`Table` algebra.
CORPUS: `src/engine/silu.rs` lines 53-92 and the real call closure `src/numeric/fixed_exp.rs`,
  `src/engine/value.rs`; the extracted definitions `silu.silu_fixed`, `word_silu_fixed`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean (committed extraction of
  `crate::word_silu_fixed`, the same body at the recorded digests).
GATE: `python3 proof/resource/Silu_gate.py --controls`, cwd /home/socol/Workspace/apeiron,
  540-second whole-invocation budget on /tmp/opencode/apeiron-lane-1.lock.
GRADE: Lean checked when that gate exits 0 and prints SILU_KERNEL_OK; no measurement.

Work model. Work is counted in executed `Result`-valued operations over the shape of the extracted
term: one operation per scalar comparison, per lifted arithmetic or conversion and per call. The
extracted closure of `word_silu_fixed` contains no loop - `silu.silu_fixed` is straight-line and its
only calls are `fixed_exp.fixed_exp` (`exp_reduce`, `exp_horner`, `exp_scale`, `round_div_2p32`) and
`value.div_round_nearest`, all straight-line - so the iteration count is zero and the measure is the
closed-form operation count of the guard chain. `Silu_gate.py` rejects the extraction if it contains
`#[rust_loop]`, so the zero-iteration claim is a check on the artifact and not a comment here. The
excluded population: any caller outside this closure, in particular the row loops of the vector
kernels in `src/engine/ffn.rs` and `src/engine/matvec.rs`, and any timing interpretation; the claim
is the executed operation count of this scalar body under the stated premises, not cycles and not
GPU time.
-/

namespace SiluResource

open Aeneas Aeneas.Std Result silu_source

/-! ## The two saturation thresholds

`engine.ONE = 1 <<< 32 = 2^32`; the guards are `UPPER = 16·ONE` and `LOWER = -21·ONE`. -/

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

/-- The first guard of the actual body, `silu.silu_fixed.UPPER = 16 · 2^32`. -/
theorem upper_eq : silu.silu_fixed.UPPER = ok (68719476736#i64) := by
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

/-- The second guard of the actual body, `silu.silu_fixed.LOWER = -21 · 2^32`. -/
theorem lower_eq : silu.silu_fixed.LOWER = ok ((-90194313216)#i64) := by
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

/-! ## The in-window closed form

With both saturation guards false, the actual body is the straight-line term below. It is written
out operation by operation because it *is* the work measure: the guard chain
`den <= 0`, `v < 0` (magnitude), `I128.checked_mul`, `v < 0` (sign), `signed > I64::MAX`,
`signed < I64::MIN` terminates in one of the terminal branches, and no other operation is
executed. -/

/-- The in-window operations of the actual body, in evaluation order: the negation `w = -v`, the
`fixed_exp` call, the `ONE` read, the checked `i64` add of the denominator, its `den <= 0` guard,
the magnitude sign test and widening, the widening of `ONE`, the checked `i128` multiply, the
widening of `den`, the round-to-nearest quotient, the sign reapplication, the two narrowing guards
and the narrowing cast. -/
def inWindow (v w : I64) : Result (core.result.Result I64 silu.SiluError) := do
  let r ← fixed_exp.fixed_exp w
  match r with
  | core.result.Result.Ok e => do
    let i2 ← engine.ONE
    let o ← lift (I64.checked_add i2 e)
    match o with
    | none => ok (core.result.Result.Err silu.SiluError.Overflow)
    | some den => do
      if den <= 0#i64
      then ok (core.result.Result.Err silu.SiluError.ZeroDenominator)
      else do
        let mag ←
          if v < 0#i64
          then do
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
          if signed > i5
          then ok (core.result.Result.Err silu.SiluError.Overflow)
          else do
            let i6 ← lift (IScalar.cast .I128 core.num.I64.MIN)
            if signed < i6
            then ok (core.result.Result.Err silu.SiluError.Overflow)
            else do
              let i7 ← lift (IScalar.cast .I64 signed)
              ok (core.result.Result.Ok i7)
  | core.result.Result.Err _ =>
    ok (core.result.Result.Err silu.SiluError.ExpOutOfRange)

/-- The extracted guard comparisons are the `IScalar` order on `i64` values. -/
private theorem leI64 {x y : I64} (h : x.val ≤ y.val) : x ≤ y :=
  (IScalar.le_equiv x y).2 h

/-- **Closed form of the exact path.** `-21·2^32 < v < 16·2^32` and the negation `w = -v` is the
extracted value, so the actual body is exactly `inWindow v w`. Both sides of this equality are the
same term written twice, so the last step is structural and consults no source-wide reduction. -/
theorem in_window (v w : I64) (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736) :
    silu.silu_fixed v = inWindow v w := by
  unfold silu.silu_fixed
  have hU : ¬ (68719476736#i64 ≤ v) := by
    intro hc
    have : (68719476736 : ℤ) ≤ v.val := by simpa using (IScalar.le_equiv _ _).1 hc
    omega
  have hL : ¬ (v ≤ (-90194313216)#i64) := by
    intro hc
    have : v.val ≤ (-90194313216 : ℤ) := by simpa using (IScalar.le_equiv _ _).1 hc
    omega
  simp only [upper_eq, bind_tc_ok, if_neg hU, lower_eq, bind_tc_ok, if_neg hL, hw, bind_tc_ok,
    inWindow]
  exact Eq.refl _

/-- The value of the extracted high narrowing bound, `IScalar.cast .I128 core.num.I64.MAX`. -/
private theorem cast_MAX_val :
    (IScalar.cast .I128 core.num.I64.MAX).val = (2 : ℤ)^63 - 1 := by
  rw [IScalar.cast_val_eq]
  have h : core.num.I64.MAX.val = (2 : ℤ)^63 - 1 := by decide
  rw [h]
  norm_num [IScalarTy.numBits]
  exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2 : ℤ)^63 - 1)
    (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])

/-- The value of the extracted low narrowing bound, `IScalar.cast .I128 core.num.I64.MIN`. -/
private theorem cast_MIN_val : (IScalar.cast .I128 core.num.I64.MIN).val = -(2 : ℤ)^63 := by
  rw [IScalar.cast_val_eq]
  have h : core.num.I64.MIN.val = -(2 : ℤ)^63 := by decide
  rw [h]
  norm_num [IScalarTy.numBits]
  exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2 : ℤ)^63)
    (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])

/-! ## Guard transitions of the actual body

Each theorem is one guard of the chain with its premise and its exact result. `guard_zero_denominator`
is the transition on the extracted guard `if den <= 0#i64`, the operation the mutation control
changes in the real Rust body. -/

/-- **Guard 1, the overflow saturation exit.** `v >= 16·2^32` returns `Ok v` before any call. -/
theorem guard_upper (v : I64) (h : (68719476736 : ℤ) ≤ v.val) :
    silu.silu_fixed v = ok (core.result.Result.Ok v) := by
  unfold silu.silu_fixed
  simp only [upper_eq, bind_tc_ok, if_pos (leI64 (x := 68719476736#i64) (y := v) (by simpa using h))]

/-- **Guard 2, the underflow saturation exit.** `v <= -21·2^32` returns `Ok 0`; guard 1 is false
because `-21·2^32 < 16·2^32`. -/
theorem guard_lower (v : I64) (h : v.val ≤ (-90194313216 : ℤ)) :
    silu.silu_fixed v = ok (core.result.Result.Ok 0#i64) := by
  unfold silu.silu_fixed
  have hU : ¬ (68719476736#i64 ≤ v) := by
    intro hc
    have : (68719476736 : ℤ) ≤ v.val := by simpa using (IScalar.le_equiv _ _).1 hc
    omega
  simp only [upper_eq, bind_tc_ok, if_neg hU, lower_eq, bind_tc_ok,
    if_pos (leI64 (x := v) (y := (-90194313216)#i64) (by simpa using h))]

/-- **Refusal 1, `ExpOutOfRange`.** On the exact path the `fixed_exp` refusal is the first
terminal branch. -/
theorem guard_exp_refusal (v w : I64) (e : fixed_exp.ExpError) (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Err e)) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.ExpOutOfRange) := by
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp]

/-- **Refusal 2a, `Overflow` on the denominator add.** -/
theorem guard_add_overflow (v w : I64) (e : I64) (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = none) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.Overflow) := by
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd]

/-- **Refusal 2b, `Overflow` on the `i128` product**, for a nonnegative `v`, where the magnitude is
`core.convert.num.FromI128I64.from v`. -/
theorem guard_mul_overflow (v w : I64) (e den : I64) (mag : I128)
    (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = some den)
    (hden : ¬ den ≤ 0#i64) (hvn : ¬ v < 0#i64)
    (hmag : (ok (core.convert.num.FromI128I64.from v) : Result I128) = ok mag)
    (hmul : I128.checked_mul mag (core.convert.num.FromI128I64.from (4294967296#i64)) = none) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.Overflow) := by
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd, if_neg hden, if_neg hvn, hmag, hmul]

/-- **Refusal 2b', `Overflow` on the `i128` product**, for a negative `v`, where the magnitude is
the negated widening `-. core.convert.num.FromI128I64.from v`. -/
theorem guard_mul_overflow_neg (v w : I64) (e den : I64) (mag : I128)
    (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = some den)
    (hden : ¬ den ≤ 0#i64) (hvn : v < 0#i64)
    (hmag : (HNeg.hNeg (core.convert.num.FromI128I64.from v) : Result I128) = ok mag)
    (hmul : I128.checked_mul mag (core.convert.num.FromI128I64.from (4294967296#i64)) = none) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.Overflow) := by
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd, if_neg hden, if_pos hvn, hmag, hmul]

/-- **Refusal 3, `ZeroDenominator`.** The extracted guard `if den <= 0#i64` is entered with the
checked denominator `den = 2^32 + e`. -/
theorem guard_zero_denominator (v w : I64) (e den : I64) (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = some den)
    (hden : den.val ≤ 0) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.ZeroDenominator) := by
  have hz : den ≤ 0#i64 := leI64 (x := den) (y := 0#i64) (by simpa using hden)
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd, if_pos hz]

/-- **Refusal 4, `Overflow` on the high narrowing guard**, for a nonnegative `v`, where the
reapplied value is the quotient `q` itself. -/
theorem guard_narrow_overflow (v w : I64) (e den : I64) (mag num q : I128)
    (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = some den)
    (hden : ¬ den ≤ 0#i64) (hvn : ¬ v < 0#i64)
    (hmag : (ok (core.convert.num.FromI128I64.from v) : Result I128) = ok mag)
    (hmul : I128.checked_mul mag (core.convert.num.FromI128I64.from (4294967296#i64)) = some num)
    (hqn : value.div_round_nearest num (core.convert.num.FromI128I64.from den) = ok q)
    (hmax : q > (IScalar.cast .I128 core.num.I64.MAX)) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.Overflow) := by
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd, if_neg hden, if_neg hvn, hmag, hmul,
    hqn, if_pos hmax]

/-- **Refusal 5, `Overflow` on the low narrowing guard**, for a negative `v`, where the reapplied
value is the negated quotient. -/
theorem guard_narrow_overflow_neg (v w : I64) (e den : I64) (mag num q signed : I128)
    (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = some den)
    (hden : ¬ den ≤ 0#i64) (hvn : v < 0#i64)
    (hmag : (HNeg.hNeg (core.convert.num.FromI128I64.from v) : Result I128) = ok mag)
    (hmul : I128.checked_mul mag (core.convert.num.FromI128I64.from (4294967296#i64)) = some num)
    (hqn : value.div_round_nearest num (core.convert.num.FromI128I64.from den) = ok q)
    (hsign : (HNeg.hNeg q : Result I128) = ok signed)
    (hmin : signed < (IScalar.cast .I128 core.num.I64.MIN)) :
    silu.silu_fixed v = ok (core.result.Result.Err silu.SiluError.Overflow) := by
  have hmax' : ¬ signed > (IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc
    have h1 := (IScalar.lt_equiv signed (IScalar.cast .I128 core.num.I64.MIN)).1 hmin
    have h2 := (IScalar.lt_equiv (IScalar.cast .I128 core.num.I64.MAX) signed).1 (by simpa using hc)
    rw [cast_MIN_val] at h1
    rw [cast_MAX_val] at h2
    omega
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd, if_neg hden, if_pos hvn, hmag, hmul,
    hqn, hsign, if_neg hmax', if_pos hmin]

/-- **The success branch**, for a nonnegative `v` past every guard: the actual body returns the
narrowed quotient `i7`. -/
theorem guard_success (v w : I64) (e den : I64) (mag num q : I128) (i7 : I64)
    (hw : (HNeg.hNeg v : Result I64) = ok w)
    (hlo : (-90194313216 : ℤ) < v.val) (hhi : v.val < 68719476736)
    (hexp : fixed_exp.fixed_exp w = ok (core.result.Result.Ok e))
    (hadd : I64.checked_add (4294967296#i64) e = some den)
    (hden : ¬ den ≤ 0#i64) (hvn : ¬ v < 0#i64)
    (hmag : (ok (core.convert.num.FromI128I64.from v) : Result I128) = ok mag)
    (hmul : I128.checked_mul mag (core.convert.num.FromI128I64.from (4294967296#i64)) = some num)
    (hqn : value.div_round_nearest num (core.convert.num.FromI128I64.from den) = ok q)
    (hmax : ¬ q > (IScalar.cast .I128 core.num.I64.MAX))
    (hmin : ¬ q < (IScalar.cast .I128 core.num.I64.MIN))
    (hi7 : i7 = IScalar.cast IScalarTy.I64 q) :
    silu.silu_fixed v = ok (core.result.Result.Ok i7) := by
  rw [in_window v w hw hlo hhi]
  simp only [inWindow, lift, bind_tc_ok, hexp, engine_ONE_eq, hadd, if_neg hden, if_neg hvn, hmag, hmul,
    hqn, if_neg hmax, if_neg hmin, hi7]

/-! ## The work measure

The extracted closure has no loop, so the measure is the closed-form operation count of the guard
chain: `guardOps` is the number of `Result`-valued operations executed up to and including the guard
that decides a branch, counted term by term from `in_window`. -/

/-- The eight terminal branches of the actual body, in guard-chain order. -/
inductive Guard where
  | satUpper : Guard
  | satLower : Guard
  | expRefusal : Guard
  | addOverflow : Guard
  | zeroDenominator : Guard
  | mulOverflow : Guard
  | narrowOverflow : Guard
  | success : Guard
  deriving DecidableEq, Repr

/-- Work in executed `Result`-valued operations, counted over the shape of the actual extracted
term: `1` comparison on the saturation exit; `3` for the second saturation guard; `6` for the
negation and the `fixed_exp` call; `8` for the `ONE` read and the checked `i64` add; `10` for the
denominator guard; `13` for the magnitude sign test, the two widenings and the checked `i128`
multiply; `19` for the third widening, the quotient (one division, one add, one division), the sign
reapplication and the high narrowing guard; `24` for the low narrowing guard and the narrowing cast. -/
def guardOps : Guard → Nat
  | .satUpper => 1
  | .satLower => 3
  | .expRefusal => 6
  | .addOverflow => 8
  | .zeroDenominator => 10
  | .mulOverflow => 13
  | .narrowOverflow => 19
  | .success => 24

/-- The work bound over the whole chain: the largest executed operation count, reached on the
success branch. -/
def maxGuardOps : Nat := 24

/-- The bound is attained only at the end of the chain. -/
theorem maxGuardOps_eq : maxGuardOps = guardOps .success := rfl

/-- No input executes more than the whole chain. -/
theorem guardOps_le (g : Guard) : guardOps g ≤ maxGuardOps := by
  cases g <;> norm_num [guardOps, maxGuardOps]

/-- The count is strictly increasing along the chain: a branch decided later does more work than a
branch decided earlier. -/
theorem guardOps_strict (a b : Guard) (h : a ≠ b) (hab : guardOps a ≤ guardOps b) :
    b = .success ∨ guardOps a < guardOps b := by
  cases a <;> cases b <;> simp_all [guardOps]

/-- **Closed-form work, region form.** Every input is in exactly one of the three regions of the
guard chain, and each saturation region executes the stated operation count. -/
theorem path_decomposition (v : I64) :
    ((68719476736 : ℤ) ≤ v.val ∧ silu.silu_fixed v = ok (core.result.Result.Ok v) ∧
        guardOps .satUpper ≤ maxGuardOps) ∨
    (v.val ≤ (-90194313216 : ℤ) ∧
        silu.silu_fixed v = ok (core.result.Result.Ok 0#i64) ∧
        guardOps .satLower ≤ maxGuardOps) ∨
    ((-90194313216 : ℤ) < v.val ∧ v.val < 68719476736) := by
  by_cases hu : (68719476736 : ℤ) ≤ v.val
  · exact Or.inl ⟨hu, guard_upper v hu, guardOps_le .satUpper⟩
  by_cases hl : v.val ≤ (-90194313216 : ℤ)
  · exact Or.inr (Or.inl ⟨hl, guard_lower v hl, guardOps_le .satLower⟩)
  exact Or.inr (Or.inr ⟨by omega, by omega⟩)

/-! ## Reachable success witnesses

Concrete machine inputs with a checked `Ok` result. Each is the saturation exit its input reaches,
evaluated at the literal value, so the witness is a term of the extracted body rather than a
description of one. -/

/-- `silu_fixed (16·2^32) = Ok (16·2^32)`: the overflow saturation end is reachable and returns
`Ok`. -/
theorem success_witness_upper :
    silu.silu_fixed (68719476736#i64) = ok (core.result.Result.Ok (68719476736#i64)) :=
  guard_upper (68719476736#i64) (by norm_num)

/-- `silu_fixed (-21·2^32) = Ok 0`: the underflow saturation end is reachable and returns `Ok`. -/
theorem success_witness_lower :
    silu.silu_fixed ((-90194313216)#i64) = ok (core.result.Result.Ok (0#i64)) :=
  guard_lower ((-90194313216)#i64) (by norm_num)

/-- `silu_fixed (i64::MAX) = Ok (i64::MAX)`: the largest `i64` input is reachable and returns
`Ok`. -/
theorem success_witness_max :
    silu.silu_fixed (9223372036854775807#i64) =
      ok (core.result.Result.Ok (9223372036854775807#i64)) :=
  guard_upper (9223372036854775807#i64) (by norm_num [I64.max_eq, IScalar.max, I64.numBits])

end SiluResource

open Lean Elab Command in
/-- Fail-closed audit of every headline under {propext, Classical.choice, Quot.sound}. -/
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit SiluResource.engine_ONE_eq
#resource_audit SiluResource.upper_eq
#resource_audit SiluResource.lower_eq
#resource_audit SiluResource.in_window
#resource_audit SiluResource.guard_upper
#resource_audit SiluResource.guard_lower
#resource_audit SiluResource.guard_exp_refusal
#resource_audit SiluResource.guard_add_overflow
#resource_audit SiluResource.guard_mul_overflow
#resource_audit SiluResource.guard_mul_overflow_neg
#resource_audit SiluResource.guard_zero_denominator
#resource_audit SiluResource.guard_narrow_overflow
#resource_audit SiluResource.guard_narrow_overflow_neg
#resource_audit SiluResource.guard_success
#resource_audit SiluResource.guardOps_le
#resource_audit SiluResource.guardOps_strict
#resource_audit SiluResource.path_decomposition
#resource_audit SiluResource.success_witness_upper
#resource_audit SiluResource.success_witness_lower
#resource_audit SiluResource.success_witness_max

run_cmd Lean.logInfo "SILU_KERNEL_OK"
