/-
Refinement of the extracted bit-plane serving kernels (`src/engine/kern/bitplane.rs`) to stated Lean
references. Round 2 of the `Serve/Bitplane` lane; the extracted bodies are
`apeiron_word.kern.bitplane.*` in `proof/ApeironWord.lean`. `Serve/Bitplane.lean` (round 1) holds
the layout constants, `decode_f16`, the nibble-dot references and the code window `block`.

Added here:

* `round_logit` (source lines 614-616): the half-biased Q32 rounding of a checked `i128`
  accumulator. Reference: `Int.bmod (roundShift32 acc.val) (2^64)`, the truncating `i128 -> i64`
  cast of `(acc + 2^31) >>> 32`. In-domain `ok` theorem and the exact `integerOverflow` refusal of
  the checked `i128` half-add.
* The portable (`avx = false`) nibble dots RESTATED at the extracted loop bodies
  `high_dot_loop` / `low_dot_loop`, so their `#print axioms` does not reach the extraction axiom
  `kern.bitplane.x86.high_dot_avx2`.
* `ChunkReduce::default`.

## Why the dots are stated at the loop, not at `high_dot ... false`

`def kern.bitplane.high_dot` has body
`do if avx then (let p <- as_ptr packed; let p1 <- as_ptr a; kern.bitplane.x86.high_dot_avx2 p p1)
      else kern.bitplane.high_dot_loop packed a 0#i32 0#usize`,
and `kern.bitplane.x86.high_dot_avx2` is an extraction `axiom` (an `unsafe` AVX2 body, source lines
714-721). `#print axioms` traverses the *body* of every definition named in a declaration's type or
value, so ANY theorem that names `kern.bitplane.high_dot` reports the axiom -- including the
reflexive `kern.bitplane.high_dot p a false = kern.bitplane.high_dot p a false` proved by `rfl`
(measured: `#print axioms` prints `[propext, Classical.choice, Quot.sound,
kern.bitplane.x86.high_dot_avx2]`). Consequently `word_high_dot_ok` / `word_low_dot_ok` in
`Serve/Bitplane.lean`, stated at `high_dot p a false`, cannot be made axiom-free without a change
to the extraction. The same holds for `row_interval`, `row_exact`, `stage1_chunk` and
`stage2_chunk`, whose extracted bodies call `kern.bitplane.high_dot s s1 false` directly, so no
closed-cone theorem about them is reachable either.

The executed `avx = false` arm IS `high_dot_loop packed a 0#i32 0#usize` (kernel iota on
`if false`), and that body is axiom-free. The theorems below prove exactly that body.

## Boundaries (not covered, stated)

`kern.bitplane.x86.*` are extraction axioms (the `unsafe` AVX2 bodies); `avx2()` is opaque runtime
feature detection; `argmax` is extracted only as closures plus an `Excl`/`Vec`/`Arc` driver, not as
one plain function; the `from_q8_0` builder returns `Option (Arc BitplaneHead)`; the `i128`
accumulations of `row_interval` / `row_exact` are checked and their overflow refusal is not stated.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`; `decide` only on closed literal facts.
-/
import Serve.Kern.Bitplane
import Numeric.Engine.Rope

set_option autoImplicit false
set_option linter.unusedSimpArgs false

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## `round_logit`: the checked half-add refusal

`kern.bitplane.round_logit acc = do let i <- value.round_shift_32 acc; ok (IScalar.cast .I64 i)`.
The refusal is the checked `i128` half-add `acc + 2^31` inside `value.round_shift_32`
(`Numeric.Engine.Rope`). -/

/-- Checked `i128` add overflows exactly when the mathematical sum exceeds `i128::MAX`. -/
theorem bp2_I128_add_fail (x y : I128) (h : I128.max < x.val + y.val) :
    (x + y : Result I128) = Result.fail Error.integerOverflow := by
  have hnb : ¬ IScalar.inBounds IScalarTy.I128 (x.val + y.val) := by
    rw [I128.max_eq] at h
    simp only [IScalar.inBounds, IScalarTy.I128_numBits_eq] at h ⊢
    omega
  have hopt := IScalar.tryMkOpt_eq IScalarTy.I128 (x.val + y.val)
  have hnone : IScalar.tryMkOpt IScalarTy.I128 (x.val + y.val) = none := by
    cases hp : IScalar.tryMkOpt IScalarTy.I128 (x.val + y.val) with
    | none => rfl
    | some y => exact absurd (by rw [hp] at hopt; exact hopt.2) hnb
  show IScalar.add x y = Result.fail Error.integerOverflow
  unfold IScalar.add IScalar.tryMk
  rw [hnone]
  rfl

/-- `value.round_shift_32` refuses with `integerOverflow` exactly off the half-add domain. -/
theorem bp2_value_round_shift_32_fail (v : I128) (hov : I128.max < v.val + 2147483648) :
    value.round_shift_32 v = Result.fail Error.integerOverflow := by
  have hadd : (v + 2147483648#i128 : Result I128) = Result.fail Error.integerOverflow :=
    bp2_I128_add_fail v 2147483648#i128 hov
  rw [value_round_shift_32_eq]
  rw [hadd]
  exact bind_fail Error.integerOverflow _

/-! ## The `round_logit` reference

`roundShift32 v = (v + 2^31) >>> 32` (`Numeric.Engine.Rope`) is the half-biased Q32 rounding of
`v`. `round_logit` truncates that integer to `i64` (a wrapping cast), so its denotation is the
balanced residue of `roundShift32` modulo `2^64`. -/

/-- Reference: the `i64` image of the half-biased Q32 rounding of `acc`. -/
def bp2_roundLogitRef (acc : I128) : ℤ := Int.bmod (roundShift32 acc.val) (2 ^ 64)

/-- **`round_logit` refinement.** Inside the checked half-add domain the extracted body returns
`ok` and its value is the truncating `i128 -> i64` cast of `roundShift32 acc.val`. -/
theorem bp2_round_logit_ok (acc : I128)
    (hlo : I128.min ≤ acc.val + 2147483648) (hhi : acc.val + 2147483648 ≤ I128.max) :
    ∃ v : I64, kern.bitplane.round_logit acc = ok v ∧ v.val = bp2_roundLogitRef acc := by
  obtain ⟨i, hi_eq, hi_val⟩ := WP.spec_imp_exists (value_round_shift_32_spec acc hlo hhi)
  refine ⟨IScalar.cast .I64 i, ?_, ?_⟩
  · have h1 : kern.bitplane.round_logit acc =
        (do let j ← value.round_shift_32 acc; ok (IScalar.cast .I64 j)) := rfl
    rw [h1, hi_eq]
    exact bind_ok i _
  · rw [IScalar.cast_val_eq]
    show Int.bmod i.val (2 ^ min IScalarTy.I64.numBits IScalarTy.I128.numBits) = _
    rw [hi_val]
    have hmin : min IScalarTy.I64.numBits IScalarTy.I128.numBits = 64 := by decide
    rw [hmin]
    rfl

/-- **`round_logit` refusal.** Outside the checked half-add domain the extracted body refuses with
`integerOverflow`. -/
theorem bp2_round_logit_refuses (acc : I128) (hov : I128.max < acc.val + 2147483648) :
    kern.bitplane.round_logit acc = Result.fail Error.integerOverflow := by
  have h1 : kern.bitplane.round_logit acc =
      (do let j ← value.round_shift_32 acc; ok (IScalar.cast .I64 j)) := rfl
  rw [h1, bp2_value_round_shift_32_fail acc hov]
  exact bind_fail Error.integerOverflow _

/-- `roundShift32` at zero (`(0 + 2^31) >>> 32 = 0`), stated at the `I128`-valued zero the example
uses so the rewrite matches the coercion form. -/
theorem bp2_roundShift32_zero : roundShift32 (0#i128).val = 0 := by
  have hv0 : (0#i128).val = (0 : ℤ) := by simp
  rw [hv0]
  unfold roundShift32
  rw [Int.shiftRight_eq_div_pow]
  norm_num

/-- The reference at zero is `0`. -/
theorem bp2_roundLogitRef_zero : bp2_roundLogitRef 0#i128 = 0 := by
  unfold bp2_roundLogitRef
  rw [bp2_roundShift32_zero]
  decide

/-- Non-vacuity: `acc = 0` is in the half-add domain, `roundShift32 0 = 0`, so the logit is `0`. -/
example : ∃ v : I64, kern.bitplane.round_logit 0#i128 = ok v ∧ v.val = 0 := by
  obtain ⟨v, hv, hvval⟩ :=
    bp2_round_logit_ok 0#i128 (by rw [I128.min_eq]; scalar_tac)
      (by rw [I128.max_eq]; scalar_tac)
  exact ⟨v, hv, by rw [hvval, bp2_roundLogitRef_zero]⟩

/-- Non-vacuity: at `i128::MAX` the checked half-add overflows, so `round_logit` refuses. -/
example : kern.bitplane.round_logit 170141183460469231731687303715884105727#i128
    = Result.fail Error.integerOverflow := by
  apply bp2_round_logit_refuses
  rw [I128.max_eq]
  scalar_tac

/-! ## The portable (`avx = false`) nibble dots, at the extracted loop bodies

`word_high_dot_ok` / `word_low_dot_ok` in `Serve/Bitplane.lean` state the same denotation at
`high_dot p a false`, whose definition body names the extraction axiom
`kern.bitplane.x86.high_dot_avx2`; naming that definition in a statement drags the axiom into
`#print axioms` regardless of the proof. The theorems below restate the portable arm at the body
the `avx = false` branch actually executes, so the dependency set stays inside the closed cone. -/

/-- **`high_dot` at `avx = false`.** The extracted loop `high_dot_loop packed a 0 0` -- which is
the body of the portable branch of `high_dot` -- returns `ok` equal to the signed 4-bit dot of the
16 packed bytes with the 32 activations. -/
theorem bp2_high_dot_portable_ok (packed : Slice Std.U8) (a : Slice Std.I8)
    (hp : 16 ≤ packed.val.length) (ha : 32 ≤ a.val.length) :
    ∃ v : I32, kern.bitplane.high_dot_loop packed a 0#i32 0#usize = ok v ∧
      v.val = highDotAux packed.val a.val 16 := by
  have h : kern.bitplane.high_dot_loop packed a 0#i32 0#usize
      ⦃ r => r.val = highDotAux packed.val a.val 16 ⦄ := by
    apply high_dot_loop_spec packed a 0#i32 0#usize
    · simp [highDotAux]
    · simp
    · exact hp
    · exact ha
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  exact ⟨y, hy, hyv⟩

/-- **`low_dot` at `avx = false`.** The extracted loop `low_dot_loop packed a 0 0` -- the body of
the portable branch of `low_dot` -- returns `ok` equal to the unsigned 4-bit dot of the 16 packed
bytes with the 32 activations. -/
theorem bp2_low_dot_portable_ok (packed : Slice Std.U8) (a : Slice Std.I8)
    (hp : 16 ≤ packed.val.length) (ha : 32 ≤ a.val.length) :
    ∃ v : I32, kern.bitplane.low_dot_loop packed a 0#i32 0#usize = ok v ∧
      v.val = lowDotAux packed.val a.val 16 := by
  have h : kern.bitplane.low_dot_loop packed a 0#i32 0#usize
      ⦃ r => r.val = lowDotAux packed.val a.val 16 ⦄ := by
    apply low_dot_loop_spec packed a 0#i32 0#usize
    · simp [lowDotAux]
    · simp
    · exact hp
    · exact ha
  rcases (WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  exact ⟨y, hy, hyv⟩

/-! ## `ChunkReduce::default`

The zero seed of the two-stage reduction: an all-zero structure with `has = false`. -/

/-- The extracted `ChunkReduce::default` is the all-zero seed. -/
theorem bp2_chunkreduce_default_ok :
    kern.bitplane.ChunkReduce.Insts.CoreDefaultDefault.default = ok
      ({ hi := 0#i128, row := 0#usize, lo := 0#i128, lo_row := 0#usize,
         has := false, best := 0#i64, best_row := 0#usize, count := 0#usize } :
        kern.bitplane.ChunkReduce) := by
  unfold kern.bitplane.ChunkReduce.Insts.CoreDefaultDefault.default
  simp only [core.default.DefaultBool.default, core.default.DefaultI128.default,
    core.default.DefaultUsize.default, core.default.DefaultI64.default, bind_tc_ok]
  rfl

/-! ## Open in this group (stated, with the obstruction)

* `row_interval`, `row_exact`, `stage1_chunk`, `stage2_chunk`: the extracted bodies call
  `kern.bitplane.high_dot s s1 false` (e.g. `row_interval_loop.body`), whose definition body names
  the extraction axiom `kern.bitplane.x86.high_dot_avx2` -- so any theorem naming them reports that
  axiom under `#print axioms` and cannot audit inside the closed cone. Additionally the `i128`
  accumulations (`hi + sc * i128::from(q.scales()[blk] * x_hi)`) are checked, and the Rust states
  their no-overflow condition only at the caller `argmax`, not in these bodies.
* `argmax`: not extracted as one plain function -- only `argmax.closure` plus a `Excl`-lock / `Vec`
  driver.
* `from_q8_0`: returns `Option (Arc BitplaneHead)`; its outer loop crosses `Arc::new` and
  `Excl::new`. Its inner packing loop `from_q8_0_loop0_loop0` is pure but has no theorem here.
-/

#print axioms bp2_I128_add_fail
#print axioms bp2_value_round_shift_32_fail
#print axioms bp2_round_logit_ok
#print axioms bp2_round_logit_refuses
#print axioms bp2_roundShift32_zero
#print axioms bp2_roundLogitRef_zero
#print axioms bp2_high_dot_portable_ok
#print axioms bp2_low_dot_portable_ok
#print axioms bp2_chunkreduce_default_ok

end

end apeiron_word
