/-
  `Serve/Block/R4/Ffn5.lean` — the block loop `chart.quant_act_loop0` of the Q8 activation quantiser
  `chart.quant_act` (`src/engine/chart/mod.rs:250`), composed from the one-lane loop of
  `Serve/Block/Ffn4.lean` and the absmax/scale references of `Serve/Block/Ffn3.lean`.

  `Serve/Block/Ffn4.lean` §2 refined the lane loop `chart.quant_act_loop0_loop1` to
  `ffn4LaneStep`, leaving `chart.quant_act_loop0`, `chart.quant_act` and
  `chart.QuantAct.dequantizes` open. This file closes the block body.

  Group coverage:
  * `chart.quant_act_loop0.body` — §2 the turn arm `word_quant_act_loop0_body_ok`: it determines
    `base`, `take` (`min ACT_QK (len x - base)`), the absmax (`ffn3AbsMaxRef`), the power-of-two
    scale (`ffn3ScaleLeast`), takes the non-refusing branch (unreachable refusal, §2), pushes the
    scale and returns the lane loop's codes as the reference `ffn5BlockStep`; §2 the refusing arm
    `word_quant_act_loop0_body_refuses` and the two stopping arms.

  Reference: `ffn5Take` (the lanes a block offers), `ffn5ScaleLeast` (per-block absmax scale, from
  `Serve/Block/Ffn3.lean`), `ffn5BlockStep` (the scale pushed and the round-to-nearest lane codes
  `ffn4LaneStep` of `Serve/Block/Ffn4.lean`).

  Open in this file, not yet proved (the residual of this task):
  * `chart.quant_act_loop0` (`ApeironWord:10858`), the block loop: it needs a `loop.spec_decr_nat`
    over the block counter with an invariant relating the accumulated `scales`/`codes` to a
    recursive block reference, including the early stop when the refusing arm or a lane refusal
    clears the flag (`word_quant_act_loop0_body_refuses`, `ffn5_block_body_stop` supply the arms);
  * `chart.quant_act` (`ApeironWord:10878`): `Vec::with_capacity`, `Usize.div_ceil`, the block loop
    and the `QuantAct.dequantizes` guard;
  * `chart.QuantAct.dequantizes` (`ApeironWord:10722`) and its nested loops
    (`dequantizes_loop0`, `dequantizes_loop0_loop0`): the `scale·127` overflow guard and the
    per-lane `scale·code` overflow check, not attempted here.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains` — the two externals
    `Serve/Block/Ffn3.lean` §7 names. The lane code family is the hypothesis `hinner` (as in
    `Serve/Block/Ffn4.lean` §1/§2); no proof step invokes an axiom.
  * the `F32`/`AVX2` bodies, `swiglu_one_f32_scalar`, the pool dispatch
    (`chart.pool.chunks.for_vec_chunks`) and the `unsafe` pointer writes: named boundaries, skipped.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch, the growable `Vec` allocation, or the two externals above. The
  statements are about the Aeneas-extracted bodies over the model `Slice`, `Vec` and word scalars.
-/

import ApeironWord
import Serve.Block.Ffn4

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §0 · Word helpers -/

/-- `chart.ACT_QK` is `32`. -/
theorem ffn5_act_qk : chart.ACT_QK = (32#usize : Std.Usize) := by
  unfold chart.ACT_QK; rfl

/-- `chart.ACT_QK.val = 32`. -/
theorem ffn5_act_qk_val : chart.ACT_QK.val = 32 := by
  rw [ffn5_act_qk]; rfl

/-- `i64::MAX` as `|i64::MAX|` in `u64` is `2^63 - 1`. -/
theorem ffn5_i64max_abs_val :
    (⟨BitVec.ofNat 64 core.num.I64.MAX.val.natAbs⟩ : Std.U64).val = 2 ^ 63 - 1 := by
  decide

/-! ## §1 · The block reference

One block `b` offers the lanes `base, …, base + take - 1` with `base = b·ACT_QK` and
`take = min ACT_QK (len x - base)`; the scale is the least power of two at least `⌈absmax/127⌉`
(`ffn3ScaleLeast`, `Serve/Block/Ffn3.lean`), and each lane contributes the round-to-nearest code
`ffn4LaneStep` (`Serve/Block/Ffn4.lean`). -/

/-- The lanes a block offers: `min ACT_QK (len - base)`, `0` when `base` is past the end. -/
def ffn5Take (len base : Nat) : Nat := min 32 (len - base)

/-- The scale value one block selects, as a natural: `2^k` where `ffn3ScaleLeast (ffn3Need absmax) k`
holds, and `0` at `absmax = 0`. Stated as the least power of two, not the search loop. -/
def ffn5ScaleLeast (absmax : Nat) (s : Nat) : Prop :=
  (absmax = 0 ∧ s = 0) ∨ ∃ k : Nat, ffn3ScaleLeast (ffn3Need absmax) k ∧ s = 2 ^ k

/-- One block's contribution: the scale is appended to the scale list and the reference per-lane
codes to the code list, with the running flag. Temporal cost `O(take)` appends; spatial `O(take)`. -/
def ffn5BlockStep (scales : List Std.I64) (codes : List Std.I8) (scale : Std.I64)
    (codeOf : Nat → Bool × Std.I8) (l0 n : Nat) : List Std.I64 × List Std.I8 × Bool :=
  (scales ++ [scale], codes ++ (ffn4LaneStep codeOf l0 n).1, (ffn4LaneStep codeOf l0 n).2)

/-- Unfolding of `ffn5BlockStep`. -/
theorem ffn5BlockStep_eq (scales : List Std.I64) (codes : List Std.I8) (scale : Std.I64)
    (codeOf : Nat → Bool × Std.I8) (l0 n : Nat) :
    ffn5BlockStep scales codes scale codeOf l0 n
      = (scales ++ [scale], codes ++ (ffn4LaneStep codeOf l0 n).1,
         (ffn4LaneStep codeOf l0 n).2) := rfl

/-! ## §2 · The block body `chart.quant_act_loop0.body`

`src/engine/chart/mod.rs:256-317`. On `block < blocks` with the running flag set the body computes
`base = block · ACT_QK`, `take = min ACT_QK (len x - base)`, the absmax of the segment, the scale,
and — when the scale does not exceed `|i64::MIN|` — pushes it and runs the lane loop. -/

/-- **The external behaviour of `i64::unsigned_abs`.** `core.num.I64.unsigned_abs` (defined in the
model `Aeneas/Std/Num2.lean`) returns the magnitude as a `u64`; the statements below carry this as
the named hypothesis `Serve/Block/Ffn3.lean` uses, so the block proofs invoke no axiom. -/
theorem ffn5_unsigned_abs_ok (z : Std.I64) :
    ∃ u : Std.U64, core.num.I64.unsigned_abs z = ok u ∧ u.val = z.val.natAbs := by
  refine ⟨⟨BitVec.ofNat 64 z.val.natAbs⟩, core.num.I64.unsigned_abs_val z, ?_⟩
  have hb := z.hBounds
  rw [IScalarTy.I64_numBits_eq] at hb
  norm_num at hb
  have h1 : z.val.natAbs < 2 ^ 64 := by
    have habs : |z.val| ≤ 2 ^ 63 := by
      rw [abs_le]; exact ⟨hb.1, le_of_lt hb.2⟩
    rw [Int.abs_eq_natAbs] at habs
    have : (z.val.natAbs : ℤ) < 2 ^ 64 := by omega
    omega
  simp only [UScalar.val, BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt h1

/-- **One `chart.quant_act_loop0.body` turn.** With `block < blocks`, the segment inside `x`, enough
capacity, and the lane code family `hinner`, the body determines `base = block · ACT_QK`,
`take = min ACT_QK (len x - base)`, the absmax `ffn3AbsMaxRef`, and a scale `s` (a power of two at
least `⌈absmax/127⌉`), takes the non-refusing branch (unreachable refusal because `s ≤ 2^62`), and
pushes the scale `cast_signed s`, returning the lane loop's codes. The result refines
`ffn5BlockStep`: `codes1.val` is `codes.val` followed by the reference codes and `ok2` is the
reference flag. `proved` relative to `hinner` (the opaque externals). -/
theorem word_quant_act_loop0_body_ok (x : Slice Std.I64) (blocks : Std.Usize)
    (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8) (block : Std.Usize)
    (codeOf : Nat → Bool × Std.I8)
    (hblock : block.val < blocks.val)
    (hmul : block.val * 32 ≤ Usize.max)
    (hadd : block.val + 1 ≤ Usize.max)
    (hlen : block.val * 32 ≤ (Slice.len x).val)
    (hscap : scales.val.length < Usize.max)
    (hcap : codes.val.length + ffn5Take (Slice.len x).val (block.val * 32) < Usize.max)
    (hinner : ∀ (s : Std.U64) (pow2 : Bool) (shift : Std.U32) (mask : Std.U64)
        (lane : Std.Usize) (v : Std.I64),
      (if s = 0#u64 then ok (true, 0#i8)
       else
        do
        let m ← core.num.I64.unsigned_abs v
        let (quotient, remainder) ← chart.act_quot_rem m s pow2 shift mask
        let i1 ← 2#u64 * remainder
        let i2 ← lift (core.convert.num.FromU64Bool.from (i1 >= s))
        let magnitude ← quotient + i2
        let signed ←
          if v < 0#i64
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
        ok (b, i3)) = ok (codeOf lane.val)) :
    ∃ scales1 : alloc.vec.Vec Std.I64, ∃ codes1 : alloc.vec.Vec Std.I8,
      ∃ ok2 : Bool, ∃ block1 : Std.Usize,
      chart.quant_act_loop0.body x blocks scales codes block true
        = ok (ControlFlow.cont (scales1, codes1, block1, ok2)) ∧
      block1.val = block.val + 1 ∧ scales1.val.length = scales.val.length + 1 ∧
      codes1.val = codes.val ++ (ffn4LaneStep codeOf 0
        (ffn5Take (Slice.len x).val (block.val * 32))).1 ∧
      ok2 = (ffn4LaneStep codeOf 0 (ffn5Take (Slice.len x).val (block.val * 32))).2 := by
  -- sub-call values, obtained through their specs
  obtain ⟨base, hbase, hbasev⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := chart.ACT_QK) (by rw [ffn5_act_qk_val]; exact hmul))
  have hbasev' : base.val = block.val * 32 := by rw [hbasev, ffn5_act_qk_val]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := Slice.len x) (y := base) (by rw [hbasev']; exact hlen))
  obtain ⟨take, htake, htakev⟩ := ordUsize_min_spec chart.ACT_QK i1
  have htakev' : take.val = ffn5Take (Slice.len x).val (block.val * 32) := by
    rw [htakev, ffn5_act_qk_val, ffn5Take, hi1v.1, hbasev']
  have hbound : base.val + take.val ≤ x.val.length := by
    have h1 : take.val ≤ (Slice.len x).val - base.val := by
      rw [htakev', ffn5Take, hbasev']; exact Nat.min_le_right 32 _
    have h2 : base.val ≤ (Slice.len x).val := by rw [hbasev']; exact hlen
    have h3 : (Slice.len x).val = x.val.length := Slice.len_val x
    omega
  obtain ⟨absmax, habsmax, _⟩ := WP.spec_imp_exists
    (ffn3_absmax_loop_spec x base take ffn5_unsigned_abs_ok hbound)
  obtain ⟨o, ho, hok⟩ := word_act_scale_ok absmax
  have ho_some : ∃ s0 : Std.U64, o = some s0 ∧ s0.val ≤ 2 ^ 62 := by
    rcases hok with ⟨_, hs0⟩ | ⟨_, s0, k, hsome, hk, hsv, _⟩
    · exact ⟨0#u64, hs0, by norm_num⟩
    · exact ⟨s0, hsome, by rw [hsv]; exact Nat.pow_le_pow_right (by norm_num) hk⟩
  obtain ⟨s, hos, hsle⟩ := ho_some
  rw [hos] at ho
  -- push of the scale
  obtain ⟨scales1, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec scales ⟨s.bv⟩ hscap)
  -- lane loop values
  obtain ⟨s2, hs2, _⟩ := ffn5_unsigned_abs_ok ⟨s.bv⟩
  obtain ⟨pow2, hpow2, _⟩ := WP.spec_imp_exists
    (show core.num.U64.is_power_of_two s2 ⦃ p => p = decide s2.val.isPowerOfTwo ⦄ from by
      rw [core.num.U64.is_power_of_two_val]; exact (WP.spec_ok _).mpr rfl)
  obtain ⟨shift, hshift, _⟩ := WP.spec_imp_exists
    (show core.num.U64.trailing_zeros s2 ⦃ u => u = ⟨BitVec.ofNat 32 (numGap_trailingZeros s2.bv)⟩ ⦄
      from by rw [core.num.U64.trailing_zeros_val]; exact (WP.spec_ok _).mpr rfl)
  have hcap' : codes.val.length + take.val < Usize.max := by rw [htakev']; exact hcap
  have hl0 : (0#usize : Std.Usize).val ≤ take.val := by
    rw [show (0#usize : Std.Usize).val = 0 from rfl]; exact Nat.zero_le _
  obtain ⟨⟨codes1, ok2⟩, hlane, hlane1, hlane2⟩ := WP.spec_imp_exists
    (ffn4_lane_loop_spec x base take s2 pow2 shift (core.num.U64.wrapping_sub s2 1#u64) codes
      codeOf 0#usize hbound hcap' hl0
      (hinner s2 pow2 shift (core.num.U64.wrapping_sub s2 1#u64)))
  obtain ⟨block1, hblock1, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) hadd)
  -- the body equation
  have hbody : chart.quant_act_loop0.body x blocks scales codes block true
      = ok (ControlFlow.cont (scales1, codes1, block1, ok2)) := by
    unfold chart.quant_act_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock)]
    rw [if_pos rfl]
    rw [hbase]; simp only [bind_tc_ok]
    rw [hi1]; simp only [bind_tc_ok]
    rw [htake]; simp only [bind_tc_ok]
    rw [habsmax]; simp only [bind_tc_ok]
    rw [ho]; simp only [bind_tc_ok]
    simp only [core.option.Option.unwrap_or_some, lift, bind_tc_ok]
    rw [core.num.I64.unsigned_abs_val]; simp only [bind_tc_ok]
    rw [if_neg (by rw [gt_iff_lt, UScalar.lt_equiv, ffn5_i64max_abs_val]; omega)]
    have hcast : core.num.U64.cast_signed s = ok ⟨s.bv⟩ := rfl
    rw [hcast]; simp only [bind_tc_ok]
    rw [hpush]; simp only [bind_tc_ok]
    rw [hs2]; simp only [bind_tc_ok]
    rw [hpow2]; simp only [bind_tc_ok]
    rw [hshift]; simp only [lift, bind_tc_ok]
    rw [hlane]; simp only [bind_tc_ok]
    rw [hblock1]; simp only [bind_tc_ok]
    rfl
  refine ⟨scales1, codes1, ok2, block1, hbody, ?_, ?_, ?_, ?_⟩
  · rw [hblock1v]; rfl
  · simp [hpushv]
  · have h0 : (0#usize : Std.Usize).val = 0 := rfl
    simpa only [h0, Nat.sub_zero, htakev'] using hlane1
  · have h0 : (0#usize : Std.Usize).val = 0 := rfl
    simpa only [h0, Nat.sub_zero, htakev'] using hlane2

/-- **The refusing arm of `chart.quant_act_loop0.body`.** When the block's scale `s` exceeds
`|i64::MIN|` the body takes the refusing branch: it drops the block's scale and codes, clears the
flag and advances the block counter, so the result refines the refusing `ffn5BlockStep` (no scale
appended, no codes). Unreachable from `chart.quant_act`: `word_quant_act_loop0_body_ok` shows the
scale is at most `2^62`. -/
theorem word_quant_act_loop0_body_refuses (x : Slice Std.I64) (blocks : Std.Usize)
    (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8) (block : Std.Usize)
    (base i1 take : Std.Usize) (absmax : Std.U64) (s : Std.U64)
    (hblock : block.val < blocks.val)
    (hbase : block * chart.ACT_QK = ok base)
    (hsub : Slice.len x - base = ok i1)
    (hmin : core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take)
    (habsmax : chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax)
    (hscale : kern.pow2.act_scale absmax = ok (some s))
    (hgt : 2 ^ 63 - 1 < s.val)
    (hadd : block.val + 1 ≤ Usize.max) :
    ∃ block1 : Std.Usize,
      chart.quant_act_loop0.body x blocks scales codes block true
        = ok (ControlFlow.cont (scales, codes, block1, false)) ∧
      block1.val = block.val + 1 := by
  obtain ⟨block1, hblock1, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) hadd)
  refine ⟨block1, ?_, ?_⟩
  · unfold chart.quant_act_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock)]
    rw [if_pos rfl]
    rw [hbase]; simp only [bind_tc_ok]
    rw [hsub]; simp only [bind_tc_ok]
    rw [hmin]; simp only [bind_tc_ok]
    rw [habsmax]; simp only [bind_tc_ok]
    rw [hscale]; simp only [bind_tc_ok]
    simp only [core.option.Option.unwrap_or_some, lift, bind_tc_ok]
    rw [core.num.I64.unsigned_abs_val]; simp only [bind_tc_ok]
    rw [if_pos (by rw [gt_iff_lt, UScalar.lt_equiv, ffn5_i64max_abs_val]; exact hgt)]
    rw [hblock1]; simp only [bind_tc_ok]
  · rw [hblock1v]; rfl

/-- **One `chart.quant_act_loop0.body` turn at or past the stopping bound.** -/
theorem ffn5_block_body_done (x : Slice Std.I64) (blocks : Std.Usize)
    (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8) (block : Std.Usize)
    (ok1 : Bool) (hge : ¬ block.val < blocks.val) :
    chart.quant_act_loop0.body x blocks scales codes block ok1
      = ok (ControlFlow.done (scales, codes, ok1)) := by
  unfold chart.quant_act_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hge)]

/-- **One `chart.quant_act_loop0.body` turn after a cleared flag.** -/
theorem ffn5_block_body_stop (x : Slice Std.I64) (blocks : Std.Usize)
    (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8) (block : Std.Usize)
    (hblock : block.val < blocks.val) :
    chart.quant_act_loop0.body x blocks scales codes block false
      = ok (ControlFlow.done (scales, codes, false)) := by
  unfold chart.quant_act_loop0.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock)]
  rw [if_neg (by decide)]

/-- Non-vacuity of §1/§2: the reference reaches a concrete input — a two-word block at `2^2` and
the round-to-nearest codes of the constant flag `true`/code `3`. -/
example : ffn5Take 2 0 = 2 ∧
    ffn5BlockStep [1#i64] [] 4#i64 (fun _ => (true, 3#i8)) 0 2
      = ([1#i64, 4#i64], [3#i8, 3#i8], true) := by
  refine ⟨by decide, ?_⟩
  unfold ffn5BlockStep
  rfl

end apeiron_word

