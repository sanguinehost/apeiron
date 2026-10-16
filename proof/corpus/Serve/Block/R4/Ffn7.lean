/-
  `Serve/Block/R4/Ffn7.lean` — discharges the per-lane code hypothesis `hinner` that
  `Serve/Block/R4/Ffn6.lean` (and `Serve/Block/Ffn4.lean` before it) takes as a parameter, for the
  lane `chart.quant_act_loop0_loop1` of `src/engine/chart/mod.rs:250`.

  What is established:
  * `ffn7Code` is the `Ffn6` `hinner` left-hand side as a plain definition, at every
    `(s, pow2, shift, mask, value)`; `ffn7Code_is_hinner_lhs` proves the identification by `rfl`.
  * `ffn7_lane_code_eq` computes it: given `pow2 = true → shift.val < 64`,
    `pow2 = true → mask.val ≤ 2^63 - 1` and `s.val ≤ 2^63 - 1`, the lane code is the reference
    `ffn3LaneCodeRef` of the `chart.act_quot_rem` pair. This is the value-indexed discharge of
    `hinner`'s content: the code is a function of the lane's value `v`, not a lane-only constant.
  * `ffn7_hinner_false` refutes `Ffn6`'s `hinner` as stated: `codeOf : Nat → Bool × Std.I8` cannot
    depend on `v`, and the code does (`(true, 1)` at `v = 1`, `(true, 2)` at `v = 2`, `s = 1#u64`,
    `pow2 = false`). `Ffn6`'s `word_quant_act_loop0_ok`, `word_quant_act_ok` and
    `word_quant_act_refuses` are therefore vacuous, not merely unproved: no `codeOf` satisfies their
    `hinner` premise.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains` are definitions in the
    pinned model (`target/aeneas-model/lean/Aeneas/Std/Num2.lean`), not axioms; no proof step here
    invokes an axiom.
  * `hscale` (`∀ block, … → ∃ …, kern.pow2.act_scale absmax = ok (some s)`) is untouched; it
    belongs to the block-level scale search, not the lane code.
  * The `F32`/`AVX2` bodies, the pool dispatch, the `unsafe` pointer writes, the growable `Vec`
    allocation: named boundaries.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch or the two model externals. The statements are about the
  Aeneas-extracted bodies over the model `Slice`, `Vec` and word scalars.
-/

import ApeironWord
import Serve.Block.R4.Ffn6

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §0 · Model word lemmas

The `FromU64Bool` value, the `RangeInclusive` construction and containment, the `i8`/`i128` casts,
and the `i128` negation. Each is a definitional law of the pinned model. -/

/-- A `Bool`-`if` and a `Prop`-`if` agree on the same predicate. -/
theorem ffn7_condB (P : Prop) [Decidable P] :
    (if decide P then (1 : Nat) else 0) = (if P then 1 else 0) := by
  by_cases h : P <;> simp [h]

/-- The `FromU64Bool` result of the round-half-up test as a natural. -/
theorem ffn7_fromU64Bool_ge_val (i1 s : U64) :
    (core.convert.num.FromU64Bool.from (i1 >= s)).val
      = (if s.val ≤ i1.val then 1 else 0) := by
  rw [ffn3_fromU64Bool_val]
  have hb : decide (i1 ≥ s) = decide (s.val ≤ i1.val) := rfl
  rw [hb]
  exact ffn7_condB (s.val ≤ i1.val)

/-- The `i8` range `[-127, 127]` as the model constructs it. -/
theorem ffn7_range_new :
    core.ops.range.RangeInclusive.new (-127)#i128 127#i128
      = ok (⟨(-127)#i128, 127#i128, false⟩ : core.ops.range.RangeInclusive Std.I128) := rfl

/-- Containment in `[-127, 127]` as the in-range predicate. -/
theorem ffn7_contains_val (signed : Std.I128) :
    core.ops.range.RangeInclusive.contains core.cmp.PartialOrdI128 core.cmp.PartialOrdI128
        core.cmp.PartialOrdI128
        (⟨(-127)#i128, 127#i128, false⟩ : core.ops.range.RangeInclusive Std.I128) signed
      = ok (decide (-127 ≤ signed.val ∧ signed.val ≤ 127)) := by
  rw [core.ops.range.RangeInclusive.contains_val]
  show (do let b1 ← core.cmp.PartialOrdI128.le ((-127)#i128) signed
           let b2 ← core.cmp.PartialOrdI128.le signed (127#i128)
           ok (b1 && b2)) = ok (decide (-127 ≤ signed.val ∧ signed.val ≤ 127))
  rw [show core.cmp.PartialOrdI128.le ((-127)#i128) signed = ok (decide (-127 ≤ signed.val)) from rfl]
  rw [show core.cmp.PartialOrdI128.le signed (127#i128) = ok (decide (signed.val ≤ 127)) from rfl]
  simp only [bind_tc_ok]
  exact congrArg ok (by rw [Bool.decide_and])

/-- The `i128 → i8` cast is exact on the closed `i8` interval. -/
theorem ffn7_cast_i8_val (z : Std.I128) (h : -128 ≤ z.val ∧ z.val ≤ 127) :
    (IScalar.cast IScalarTy.I8 z).val = z.val := by
  have hs := IScalar.cast_inBounds_spec IScalarTy.I8 z (by
    refine ⟨?_, ?_⟩ <;> scalar_tac)
  simpa [lift] using hs

/-- The `u64 → i128` cast is exact. -/
theorem ffn7_hcast_i128_val (x : Std.U64) : (UScalar.hcast IScalarTy.I128 x).val = x.val := by
  have hs := UScalar.hcast_inBounds_spec IScalarTy.I128 x (by have := x.hBounds; scalar_tac)
  simpa [lift] using hs

/-- The checked `i128` negation succeeds on a nonnegative operand and returns its negation. -/
theorem ffn7_neg_val (x : Std.I128) (h : 0 ≤ x.val) :
    ∃ z : Std.I128, (-. x) = ok z ∧ z.val = -x.val := by
  have hmin : (IScalar.min IScalarTy.I128 : ℤ) < 0 := by
    rw [IScalar.min_IScalarTy_I128_eq]; rw [I128.min]; norm_num
  have hne : (↑x : ℤ) ≠ IScalar.min IScalarTy.I128 := by
    intro hc; rw [hc] at h; omega
  obtain ⟨z, hz, hzv⟩ := WP.spec_imp_exists (HNeg.hNeg.step (x := x) hne)
  exact ⟨z, hz, hzv⟩

/-! ## §1 · The lane code computation `chart.quant_act_loop0_loop1`

`ffn7Code` is the code `Ffn6`'s `hinner` asserts, as a definition at every scale and value;
`ffn7_lane_code_eq` discharges it against the `Ffn3` per-lane reference. -/

/-- The lane-code tail: everything after the `chart.act_quot_rem` pair. Factored from `ffn7Code`
so the pair-bind substitution in the proof is one `change`. -/
def ffn7LaneTail (s : Std.U64) (value : Std.I64) (remainder quotient : Std.U64) :
    Result (Bool × Std.I8) :=
  do
  let i1 ← 2#u64 * remainder
  let i2 ← lift (core.convert.num.FromU64Bool.from (i1 >= s))
  let magnitude ← quotient + i2
  let signed ←
    if value < 0#i64
    then
      do
      let i3 ← I128.Insts.CoreConvertFromU64.from magnitude
      -. i3
    else I128.Insts.CoreConvertFromU64.from magnitude
  let ri ← core.ops.range.RangeInclusive.new (-127)#i128 127#i128
  let b ←
    core.ops.range.RangeInclusive.contains core.cmp.PartialOrdI128
      core.cmp.PartialOrdI128 core.cmp.PartialOrdI128 ri signed
  let i3 ← if b
             then ok (IScalar.cast .I8 signed)
             else ok 0#i8
  ok (b, i3)

/-- The per-lane code computation `chart.quant_act_loop0_loop1` runs at one lane with value
`value`: the `Ffn6` `hinner` left-hand side, as a definition. Temporal and spatial cost `O(1)` in
words: one `unsigned_abs`, one `act_quot_rem`, and a fixed sequence of scalar operations. -/
def ffn7Code (s : Std.U64) (pow2 : Bool) (shift : Std.U32) (mask : Std.U64)
    (value : Std.I64) : Result (Bool × Std.I8) :=
  if s = 0#u64 then ok (true, 0#i8)
  else
    do
    let m ← core.num.I64.unsigned_abs value
    let (quotient, remainder) ← chart.act_quot_rem m s pow2 shift mask
    ffn7LaneTail s value remainder quotient

/-- **`ffn7Code` is the `Ffn6` `hinner` left-hand side.** The factored `ffn7LaneTail` is
definitionally the inline body, so this `rfl` fixes the object the refutation and the value law
below are about. `proved`. -/
theorem ffn7Code_is_hinner_lhs (s : Std.U64) (pow2 : Bool) (shift : Std.U32) (mask : Std.U64)
    (value : Std.I64) :
    ffn7Code s pow2 shift mask value
      = (if s = 0#u64 then ok (true, 0#i8)
         else
          do
          let m ← core.num.I64.unsigned_abs value
          let (quotient, remainder) ← chart.act_quot_rem m s pow2 shift mask
          let i1 ← 2#u64 * remainder
          let i2 ← lift (core.convert.num.FromU64Bool.from (i1 >= s))
          let magnitude ← quotient + i2
          let signed ←
            if value < 0#i64
            then
              do
              let i3 ← I128.Insts.CoreConvertFromU64.from magnitude
              -. i3
            else I128.Insts.CoreConvertFromU64.from magnitude
          let ri ← core.ops.range.RangeInclusive.new (-127)#i128 127#i128
          let b ←
            core.ops.range.RangeInclusive.contains core.cmp.PartialOrdI128
              core.cmp.PartialOrdI128 core.cmp.PartialOrdI128 ri signed
          let i3 ← if b
                     then ok (IScalar.cast .I8 signed)
                     else ok 0#i8
          ok (b, i3)) := rfl

/-- The range test and the `i8` cast over one computed `signed` magnitude. -/
theorem ffn7_tail_ok (signed : Std.I128) (value : Std.I64) (s q r : Nat) (hs : s ≠ 0)
    (hval : signed.val = ffn3LaneSigned (value.val : ℤ) s q r) :
    ∃ (b : Bool) (code : Std.I8),
      (do
        let ri ← core.ops.range.RangeInclusive.new (-127)#i128 127#i128
        let b ←
          core.ops.range.RangeInclusive.contains core.cmp.PartialOrdI128
            core.cmp.PartialOrdI128 core.cmp.PartialOrdI128 ri signed
        let i3 ← if b then ok (IScalar.cast .I8 signed) else ok 0#i8
        ok (b, i3)) = ok (b, code) ∧
      (b, code.val) = ffn3LaneCodeRef (value.val : ℤ) s q r := by
  by_cases hin : -127 ≤ ffn3LaneSigned (value.val : ℤ) s q r ∧ ffn3LaneSigned (value.val : ℤ) s q r ≤ 127
  · refine ⟨true, IScalar.cast IScalarTy.I8 signed, ?_, ?_⟩
    · rw [ffn7_range_new]; simp only [bind_tc_ok]
      rw [ffn7_contains_val]; simp only [bind_tc_ok]
      rw [show decide (-127 ≤ signed.val ∧ signed.val ≤ 127) = true from by
        rw [hval]; exact decide_eq_true hin]
      rw [if_pos rfl]; simp only [bind_tc_ok]
    · unfold ffn3LaneCodeRef
      rw [if_neg hs]
      rw [show decide (-127 ≤ ffn3LaneSigned (value.val : ℤ) s q r ∧
            ffn3LaneSigned (value.val : ℤ) s q r ≤ 127) = true from decide_eq_true hin]
      have hcast : (IScalar.cast IScalarTy.I8 signed).val = ffn3LaneSigned (value.val : ℤ) s q r := by
        have h1 : -128 ≤ signed.val ∧ signed.val ≤ 127 := by rw [hval]; omega
        rw [ffn7_cast_i8_val signed h1, hval]
      rw [if_pos hin, hcast]
  · refine ⟨false, 0#i8, ?_, ?_⟩
    · rw [ffn7_range_new]; simp only [bind_tc_ok]
      rw [ffn7_contains_val]; simp only [bind_tc_ok]
      rw [show decide (-127 ≤ signed.val ∧ signed.val ≤ 127) = false from by
        rw [hval]; exact decide_eq_false hin]
      rw [if_neg (by decide : ¬ (((false : Bool)) = true))]; simp only [bind_tc_ok]
    · unfold ffn3LaneCodeRef
      rw [if_neg hs]
      rw [show decide (-127 ≤ ffn3LaneSigned (value.val : ℤ) s q r ∧
            ffn3LaneSigned (value.val : ℤ) s q r ≤ 127) = false from decide_eq_false hin]
      rw [if_neg hin]
      decide

/-- **The lane code is the `Ffn3` reference, as a function of the value.** For every scale with
`pow2 = true → shift.val < 64`, `pow2 = true → mask.val ≤ 2^63 - 1` and `s.val ≤ 2^63 - 1`, the
`chart.act_quot_rem` pair `(q, r)` of `|value|` exists and `ffn7Code` returns `(b, code)` with
`(b, code.val) = ffn3LaneCodeRef value.val s.val q.val r.val`.

This is the value-indexed discharge of `Ffn6`'s `hinner`: the code depends on the lane's `value`,
so no lane-only `codeOf` can stand in for it (`ffn7_hinner_false`). The bound hypotheses are
required: `pow2` masks `r = m &&& mask ≤ mask` and the division branch gives `r < s`, and neither
`2·r` may exceed `U64.max`; `s.val ≤ 2^63 - 1` and `mask.val ≤ 2^63 - 1` (with `m.val ≤ 2^63`)
bound `2·r`. `proved`. -/
theorem ffn7_lane_code_eq (s : Std.U64) (pow2 : Bool) (shift : Std.U32) (mask : Std.U64)
    (value : Std.I64)
    (hshift : pow2 = true → shift.val < 64)
    (hmask : pow2 = true → mask.val ≤ 2 ^ 63 - 1)
    (hsle : s.val ≤ 2 ^ 63 - 1) :
    ∃ (b : Bool) (code : Std.I8) (q r : Std.U64),
      chart.act_quot_rem (⟨BitVec.ofNat 64 value.val.natAbs⟩ : Std.U64) s pow2 shift mask
        = ok (q, r) ∧
      ffn7Code s pow2 shift mask value = ok (b, code) ∧
      (b, code.val) = ffn3LaneCodeRef (value.val : ℤ) s.val q.val r.val := by
  by_cases hs0 : s = 0#u64
  · obtain ⟨q, r, hqr, _⟩ :=
      word_act_quot_rem_ok (⟨BitVec.ofNat 64 value.val.natAbs⟩ : Std.U64) s pow2 shift mask hshift
    refine ⟨true, 0#i8, q, r, hqr, ?_, ?_⟩
    · unfold ffn7Code; rw [if_pos hs0]
    · unfold ffn3LaneCodeRef
      rw [if_pos (show s.val = 0 by rw [hs0]; rfl)]
      rfl
  · have hs0v : s.val ≠ 0 := by
      intro h; exact hs0 (UScalar.eq_of_val_eq (by rw [h]; rfl))
    set m : Std.U64 := (⟨BitVec.ofNat 64 value.val.natAbs⟩ : Std.U64) with hmdef
    obtain ⟨q, r, hqr, hqrv⟩ := word_act_quot_rem_ok m s pow2 shift mask hshift
    have hmv : m.val = value.val.natAbs := by
      rw [hmdef]; simp only [UScalar.val]
      have hb := value.hBounds
      rw [IScalarTy.I64_numBits_eq] at hb; norm_num at hb
      apply Nat.mod_eq_of_lt; omega
    have hmle : m.val ≤ 2 ^ 63 := by
      rw [hmv]
      have hb := value.hBounds
      rw [IScalarTy.I64_numBits_eq] at hb; norm_num at hb; omega
    have hqle : q.val ≤ 2 ^ 63 := by
      have hv := hqrv
      by_cases hp : pow2 = true
      · rw [if_pos hp] at hv
        rw [hv.1, actShrRef]
        exact Nat.le_trans (Nat.shiftRight_le m.val shift.val) hmle
      · rw [if_neg hp] at hv
        rw [hv.1, actQuoRef, if_neg hs0v]
        exact Nat.le_trans (Nat.div_le_self m.val s.val) hmle
    have hrle : 2 * r.val ≤ U64.max := by
      have hv := hqrv
      by_cases hp : pow2 = true
      · rw [if_pos hp] at hv
        rw [hv.2, actAndRef]
        have h1 := Nat.and_le_right (n := m.val) (m := mask.val)
        have h2 := hmask hp
        rw [U64.max_eq]; omega
      · rw [if_neg hp] at hv
        rw [hv.2, actRemRef, if_neg hs0v]
        have h1 := Nat.mod_lt m.val (by omega : 0 < s.val)
        rw [U64.max_eq]; omega
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
      (U64.mul_spec (x := (2#u64 : Std.U64)) (y := r) (by simpa using hrle))
    have hi1v' : i1.val = 2 * r.val := by simpa using hi1v
    have hmulmax : q.val + (core.convert.num.FromU64Bool.from (i1 >= s)).val ≤ U64.max := by
      rw [ffn7_fromU64Bool_ge_val, hi1v']
      by_cases h : s.val ≤ 2 * r.val
      · rw [if_pos h, U64.max_eq]; omega
      · rw [if_neg h, U64.max_eq]; omega
    obtain ⟨magnitude, hmagnitude, hmagnitudev⟩ := WP.spec_imp_exists
      (U64.add_spec (x := q) (y := core.convert.num.FromU64Bool.from (i1 >= s)) hmulmax)
    have hmagnitudev' : magnitude.val = ffn3LaneMag s.val q.val r.val := by
      rw [hmagnitudev, ffn7_fromU64Bool_ge_val, hi1v', ffn3LaneMag]
    by_cases hneg : value < 0#i64
    · obtain ⟨si, hsi, hsiv⟩ := ffn7_neg_val (UScalar.hcast IScalarTy.I128 magnitude)
        (by rw [ffn7_hcast_i128_val]; exact Int.natCast_nonneg magnitude.val)
      have hsival : si.val = ffn3LaneSigned (value.val : ℤ) s.val q.val r.val := by
        unfold ffn3LaneSigned
        rw [if_pos (show (value.val : ℤ) < 0 from hneg)]
        rw [hsiv, ffn7_hcast_i128_val, hmagnitudev']
      obtain ⟨b, code, htail, hbc⟩ := ffn7_tail_ok si value s.val q.val r.val hs0v hsival
      refine ⟨b, code, q, r, hqr, ?_, hbc⟩
      unfold ffn7Code
      rw [if_neg hs0]
      rw [core.num.I64.unsigned_abs_val, ← hmdef]
      simp only [bind_tc_ok]
      rw [hqr]; simp only [bind_tc_ok]
      change ffn7LaneTail s value r q = ok (b, code)
      unfold ffn7LaneTail
      rw [hi1]; simp only [bind_tc_ok]
      simp only [lift, bind_tc_ok]
      rw [hmagnitude]; simp only [bind_tc_ok]
      rw [if_pos hneg]
      rw [show I128.Insts.CoreConvertFromU64.from magnitude
            = ok (UScalar.hcast IScalarTy.I128 magnitude) from rfl]
      simp only [bind_tc_ok]
      rw [hsi]; simp only [bind_tc_ok]
      rw [htail]
    · have hsival : (UScalar.hcast IScalarTy.I128 magnitude).val
          = ffn3LaneSigned (value.val : ℤ) s.val q.val r.val := by
        unfold ffn3LaneSigned
        rw [if_neg (show ¬ ((value.val : ℤ) < 0) from hneg)]
        rw [ffn7_hcast_i128_val, hmagnitudev']
      obtain ⟨b, code, htail, hbc⟩ :=
        ffn7_tail_ok (UScalar.hcast IScalarTy.I128 magnitude) value s.val q.val r.val hs0v hsival
      refine ⟨b, code, q, r, hqr, ?_, hbc⟩
      unfold ffn7Code
      rw [if_neg hs0]
      rw [core.num.I64.unsigned_abs_val, ← hmdef]
      simp only [bind_tc_ok]
      rw [hqr]; simp only [bind_tc_ok]
      change ffn7LaneTail s value r q = ok (b, code)
      unfold ffn7LaneTail
      rw [hi1]; simp only [bind_tc_ok]
      simp only [lift, bind_tc_ok]
      rw [hmagnitude]; simp only [bind_tc_ok]
      rw [if_neg hneg]
      rw [show I128.Insts.CoreConvertFromU64.from magnitude
            = ok (UScalar.hcast IScalarTy.I128 magnitude) from rfl]
      simp only [bind_tc_ok]
      rw [htail]

/-! ## §2 · `hinner` is value-indexed, so lane-only `codeOf` is unsatisfiable

At the unit scale the lane's code varies with the lane's value, which a `codeOf : Nat → Bool ×
Std.I8` cannot express. `Ffn6`'s `hinner` premise is therefore false, and the headline
restatements over it (`word_quant_act_loop0_ok`, `word_quant_act_ok`, `word_quant_act_refuses`) are
vacuous. -/

/-- At the unit scale (`s = 1#u64`, `pow2 = false`) the lane with value `1` codes as `(true, 1)`.
`proved`. -/
theorem ffn7Code_one :
    ∃ (code : Std.I8),
      ffn7Code 1#u64 false 0#u32 0#u64 1#i64 = ok (true, code) ∧ code.val = 1 := by
  obtain ⟨b, code, q, r, hqr, hcode, hbc⟩ :=
    ffn7_lane_code_eq 1#u64 false 0#u32 0#u64 1#i64
      (by intro h; exact absurd h (by decide))
      (by intro h; exact absurd h (by decide))
      (by decide)
  obtain ⟨q', r', hqr', hqrv'⟩ :=
    word_act_quot_rem_ok (⟨BitVec.ofNat 64 (1#i64 : Std.I64).val.natAbs⟩ : Std.U64)
      1#u64 false 0#u32 0#u64 (by intro h; exact absurd h (by decide))
  rw [if_neg (by decide : ¬ ((false : Bool) = true))] at hqrv'
  have hpair : (q, r) = (q', r') := by
    simpa only [Result.ok.injEq] using hqr.symm.trans hqr'
  have hqeq : q = q' := congrArg Prod.fst hpair
  have hreq : r = r' := congrArg Prod.snd hpair
  have hqv : q.val = 1 := by rw [hqeq, hqrv'.1]; decide
  have hrv : r.val = 0 := by rw [hreq, hqrv'.2]; decide
  have hbc' : (b, code.val) = (true, 1) := by
    rw [hbc, hqv, hrv]; decide
  have hb : b = true := congrArg Prod.fst hbc'
  have hcv : code.val = 1 := congrArg Prod.snd hbc'
  exact ⟨code, by rw [hcode, hb], hcv⟩

/-- At the unit scale the lane with value `2` codes as `(true, 2)`. `proved`. -/
theorem ffn7Code_two :
    ∃ (code : Std.I8),
      ffn7Code 1#u64 false 0#u32 0#u64 2#i64 = ok (true, code) ∧ code.val = 2 := by
  obtain ⟨b, code, q, r, hqr, hcode, hbc⟩ :=
    ffn7_lane_code_eq 1#u64 false 0#u32 0#u64 2#i64
      (by intro h; exact absurd h (by decide))
      (by intro h; exact absurd h (by decide))
      (by decide)
  obtain ⟨q', r', hqr', hqrv'⟩ :=
    word_act_quot_rem_ok (⟨BitVec.ofNat 64 (2#i64 : Std.I64).val.natAbs⟩ : Std.U64)
      1#u64 false 0#u32 0#u64 (by intro h; exact absurd h (by decide))
  rw [if_neg (by decide : ¬ ((false : Bool) = true))] at hqrv'
  have hpair : (q, r) = (q', r') := by
    simpa only [Result.ok.injEq] using hqr.symm.trans hqr'
  have hqeq : q = q' := congrArg Prod.fst hpair
  have hreq : r = r' := congrArg Prod.snd hpair
  have hqv : q.val = 2 := by rw [hqeq, hqrv'.1]; decide
  have hrv : r.val = 0 := by rw [hreq, hqrv'.2]; decide
  have hbc' : (b, code.val) = (true, 2) := by
    rw [hbc, hqv, hrv]; decide
  have hb : b = true := congrArg Prod.fst hbc'
  have hcv : code.val = 2 := congrArg Prod.snd hbc'
  exact ⟨code, by rw [hcode, hb], hcv⟩

/-- **`Ffn6`'s `hinner` is unsatisfiable.** Its left-hand side at `s = 1#u64`, `pow2 = false`,
`shift = 0`, `mask = 0` takes the value `(true, 1)` at `v = 1#i64` and `(true, 2)` at `v = 2#i64`,
while the premise asserts both equal `ok (codeOf lane.val)` for one `codeOf` that cannot see `v`.
There is no such `codeOf`. `proved`.

Consequence: the headline restatements stated over this premise — `word_quant_act_loop0_ok`,
`word_quant_act_ok`, `word_quant_act_refuses` — hold only vacuously, and the premise must be
replaced by the value-indexed `ffn7_lane_code_eq` before any of them carries content. -/
theorem ffn7_hinner_false :
    ¬ (∀ (codeOf : Nat → Bool × Std.I8) (lane : Std.Usize) (v : Std.I64),
        ffn7Code 1#u64 false 0#u32 0#u64 v = ok (codeOf lane.val)) := by
  intro h
  obtain ⟨c1, _, hcv1⟩ := ffn7Code_one
  obtain ⟨c2, _, hcv2⟩ := ffn7Code_two
  have h1 : ffn7Code 1#u64 false 0#u32 0#u64 1#i64 = ok (true, c1) := by
    simpa using h (fun _ : Nat => (true, c1)) 0#usize 1#i64
  have h2 : ffn7Code 1#u64 false 0#u32 0#u64 1#i64 = ok (true, c2) := by
    simpa using h (fun _ : Nat => (true, c2)) 0#usize 1#i64
  have hc : (true, c1) = (true, c2) := Result.ok.injEq.mp (h1.symm.trans h2)
  have hvv : c1.val = c2.val := congrArg (fun x : Std.I8 => x.val) (congrArg Prod.snd hc)
  rw [hcv1, hcv2] at hvv
  omega

/-! ## §3 · Non-vacuity

The §1 hypotheses are satisfiable and its conclusion inhabited at concrete inputs. -/

/-- The §1 headline is inhabited: at `s = 1#u64` the conclusion holds for the concrete input `0`. -/
example : ∃ (b : Bool) (code : Std.I8),
    ffn7Code 1#u64 false 0#u32 0#u64 0#i64 = ok (b, code) := by
  obtain ⟨b, code, q, r, _, hcode, _⟩ :=
    ffn7_lane_code_eq 1#u64 false 0#u32 0#u64 0#i64
      (by intro h; exact absurd h (by decide))
      (by intro h; exact absurd h (by decide))
      (by decide)
  exact ⟨b, code, hcode⟩

end apeiron_word
