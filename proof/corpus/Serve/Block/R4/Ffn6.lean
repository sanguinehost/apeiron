/-
  `Serve/Block/R4/Ffn6.lean` — closes the residual of `Serve/Block/R4/Ffn5.lean`: the block loop
  `chart.quant_act_loop0`, the driver `chart.quant_act`, and `chart.QuantAct.dequantizes` with its
  two nested loops (`src/engine/chart/mod.rs:250`).

  Group coverage:
  * §1 `chart.quant_act_loop0` — the block loop: a `loop.spec_decr_nat` over the block counter with a
    recursive reference `ffn5Blocks` and the early-stop-on-flag-clear invariant; §1 the strengthened
    turn `ffn6_block_turn` (adds the pushed scale's value to `word_quant_act_loop0_body_ok`) and the
    value-level headline `word_quant_act_loop0_ok` over `ffn5BlockStep`.
  * §2 `chart.QuantAct.dequantizes` and its two nested loops.
  * §3 `chart.quant_act`.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains` — the opaque externals
    `Serve/Block/R4/Ffn5.lean` names. The lane code computation is the hypothesis `hinner`, exactly as
    `Serve/Block/Ffn4.lean` takes it; no proof step invokes an axiom.
  * the `F32`/`AVX2` bodies, the pool dispatch and the `unsafe` pointer writes: named boundaries.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch, the growable `Vec` allocation, or the two externals above. The
  statements are about the Aeneas-extracted bodies over the model `Slice`, `Vec` and word scalars.
-/

import ApeironWord
import Serve.Block.R4.Ffn5

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §1 · The block loop `chart.quant_act_loop0`

`src/engine/chart/mod.rs:256`. The extracted loop carries `(scales, codes, block, ok)` and stops when
`block` reaches `blocks` or when the running flag clears. The reference `ffn5Blocks` iterates a
per-block turn `turnOf` over the counter with the same early stop. -/

/-- The block loop reference: the state after `k` turns from `b`, with early stop when a turn's flag
clears. `turnOf b scales codes` is block `b`'s `cont` payload — from `ffn5BlockStep` it appends the
block's scale and codes and carries the lane-loop flag. Temporal cost `O(k)` turns; spatial `O(k)`
list appends; structural recursion on `k`. -/
def ffn5Blocks (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (scales : List Std.I64) (codes : List Std.I8) (b : Nat) :
    Nat → List Std.I64 × List Std.I8 × Bool
  | 0 => (scales, codes, true)
  | k + 1 =>
    let st := ffn5Blocks turnOf scales codes b k
    if st.2.2 then turnOf (b + k) st.1 st.2.1 else st

/-- `ffn5Blocks` at zero turns is the initial state with the flag set. -/
lemma ffn5Blocks_zero (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (scales : List Std.I64) (codes : List Std.I8) (b : Nat) :
    ffn5Blocks turnOf scales codes b 0 = (scales, codes, true) := rfl

/-- One-step unfolding of `ffn5Blocks`. -/
lemma ffn5Blocks_succ (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (scales : List Std.I64) (codes : List Std.I8) (b k : Nat) :
    ffn5Blocks turnOf scales codes b (k + 1)
      = (if (ffn5Blocks turnOf scales codes b k).2.2 then
           turnOf (b + k) (ffn5Blocks turnOf scales codes b k).1
             (ffn5Blocks turnOf scales codes b k).2.1
         else ffn5Blocks turnOf scales codes b k) := rfl

/-- One continuation step: with the running flag set, the next block's turn is taken. -/
lemma ffn5Blocks_cont (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (scales : List Std.I64) (codes : List Std.I8) (b k : Nat)
    (h : (ffn5Blocks turnOf scales codes b k).2.2 = true) :
    ffn5Blocks turnOf scales codes b (k + 1)
      = turnOf (b + k) (ffn5Blocks turnOf scales codes b k).1
          (ffn5Blocks turnOf scales codes b k).2.1 := by
  rw [ffn5Blocks_succ]
  simp only [h, if_true]

/-- Once the running flag clears, further turns leave the state unchanged (the early stop). -/
lemma ffn5Blocks_false (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (scales : List Std.I64) (codes : List Std.I8) (b k : Nat)
    (h : (ffn5Blocks turnOf scales codes b k).2.2 = false) (j : Nat) :
    ffn5Blocks turnOf scales codes b (k + j) = ffn5Blocks turnOf scales codes b k := by
  induction j with
  | zero => rw [Nat.add_zero]
  | succ j ih =>
    have hk : (ffn5Blocks turnOf scales codes b (k + j)).2.2 = false := by rw [ih]; exact h
    rw [show k + (j + 1) = (k + j) + 1 by omega, ffn5Blocks_succ,
      if_neg (by rw [hk]; decide)]
    exact ih

/-- **The `chart.quant_act_loop0` loop spec.** With the per-block turn `turnOf` (the block body's
`cont` payload, which `word_quant_act_loop0_body_ok` supplies) and the counter inside `[block0,
blocks]`, the loop from the flag `true` returns the recursive reference. The stopping arms are the
extracted body's own (`ffn5_block_body_done`, `ffn5_block_body_stop`); no turn is a case here. -/
theorem ffn6_blocks_loop_spec (x : Slice Std.I64) (blocks : Std.Usize)
    (scales0 : alloc.vec.Vec Std.I64) (codes0 : alloc.vec.Vec Std.I8) (block0 : Std.Usize)
    (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (hle : block0.val ≤ blocks.val)
    (hturn : ∀ (block : Std.Usize) (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8),
      block.val < blocks.val →
      ∃ scales1 : alloc.vec.Vec Std.I64, ∃ codes1 : alloc.vec.Vec Std.I8,
      ∃ block₁ : Std.Usize, ∃ ok2 : Bool,
        chart.quant_act_loop0.body x blocks scales codes block true
            = ok (ControlFlow.cont (scales1, codes1, block₁, ok2)) ∧
        block₁.val = block.val + 1 ∧
        (scales1.val, codes1.val, ok2) = turnOf block.val scales.val codes.val) :
    chart.quant_act_loop0 x blocks scales0 codes0 block0 true ⦃ out =>
      out.1.val = (ffn5Blocks turnOf scales0.val codes0.val block0.val
        (blocks.val - block0.val)).1 ∧
      out.2.1.val = (ffn5Blocks turnOf scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.1 ∧
      out.2.2 = (ffn5Blocks turnOf scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.2 ⦄ := by
  unfold chart.quant_act_loop0
  apply loop.spec_decr_nat
    (measure := fun st : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Std.Usize × Bool =>
      blocks.val - st.2.2.1.val)
    (inv := fun st : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Std.Usize × Bool =>
      block0.val ≤ st.2.2.1.val ∧ st.2.2.1.val ≤ blocks.val ∧
      (st.1.val, st.2.1.val, st.2.2.2)
        = ffn5Blocks turnOf scales0.val codes0.val block0.val (st.2.2.1.val - block0.val))
    (post := fun out : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Bool =>
      out.1.val = (ffn5Blocks turnOf scales0.val codes0.val block0.val
        (blocks.val - block0.val)).1 ∧
      out.2.1.val = (ffn5Blocks turnOf scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.1 ∧
      out.2.2 = (ffn5Blocks turnOf scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.2)
  · rintro ⟨scales, codes, block, ok1⟩ ⟨hge0, hleB, hinv⟩
    dsimp only at hge0 hleB hinv ⊢
    by_cases hlt : block.val < blocks.val
    · cases ok1
      · -- flag cleared: the body stops
        rw [ffn5_block_body_stop x blocks scales codes block hlt, WP.spec_ok]
        dsimp only
        have hparts : ffn5Blocks turnOf scales0.val codes0.val block0.val
            (block.val - block0.val) = (scales.val, codes.val, false) := hinv.symm
        have hstab := ffn5Blocks_false turnOf scales0.val codes0.val block0.val
          (block.val - block0.val) (by rw [hparts]) (blocks.val - block.val)
        have hidx : blocks.val - block0.val
            = (block.val - block0.val) + (blocks.val - block.val) := by omega
        refine ⟨?_, ?_, ?_⟩
        · rw [hidx, hstab, hparts]
        · rw [hidx, hstab, hparts]
        · rw [hidx, hstab, hparts]
      · -- flag set: the body takes the block turn
        obtain ⟨scales1, codes1, block₁, ok2, hb, hb1, hto⟩ := hturn block scales codes hlt
        rw [hb, WP.spec_ok]
        dsimp only
        have hflag : (ffn5Blocks turnOf scales0.val codes0.val block0.val
            (block.val - block0.val)).2.2 = true := by rw [← hinv]
        have hparts : ffn5Blocks turnOf scales0.val codes0.val block0.val
            (block.val - block0.val) = (scales.val, codes.val, true) := hinv.symm
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hb1]; omega
        · rw [hb1]; omega
        · rw [hto, hb1, show block.val + 1 - block0.val
              = (block.val - block0.val) + 1 by omega,
            ffn5Blocks_cont turnOf scales0.val codes0.val block0.val
              (block.val - block0.val) hflag]
          rw [hparts]
          rw [show block0.val + (block.val - block0.val) = block.val by omega]
        · rw [hb1]; omega
    · -- counter at or past the stopping bound
      have hbeq : block.val = blocks.val := le_antisymm hleB (le_of_not_gt hlt)
      rw [ffn5_block_body_done x blocks scales codes block ok1 hlt, WP.spec_ok]
      dsimp only
      have hidx : blocks.val - block0.val = block.val - block0.val := by omega
      refine ⟨?_, ?_, ?_⟩ <;>
        rw [hidx, ← hinv]
  · refine ⟨le_refl _, hle, ?_⟩
    rw [Nat.sub_self, ffn5Blocks_zero]

/-! ## §1b · The block turn `ffn6_block_turn`

`word_quant_act_loop0_body_ok` carries the pushed scale only through `scales1.val.length =
scales.val.length + 1`. The block loop needs the scale's value, so the turn is restated with the
sub-values (`base`, `i1`, `take`, `absmax`, the scale `s`) as hypotheses, exactly as
`word_quant_act_loop0_body_refuses` does, and the conclusion adds `scales1.val = scales.val ++
[⟨s.bv⟩]`. The proof follows `word_quant_act_loop0_body_ok`'s rewrite sequence; the extra conjunct is
the `Vec.push` value that proof already computes. -/

/-- **One `chart.quant_act_loop0.body` turn, with the pushed scale's value.** Given the block's
`base`, `i1`, `take`, absmax and scale `s` (the sub-values the body computes), the body takes the
non-refusing branch, pushes `⟨s.bv⟩`, runs the lane loop and advances. `proved` relative to `hinner`
(the opaque externals). -/
theorem ffn6_block_turn (x : Slice Std.I64) (blocks : Std.Usize)
    (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8) (block : Std.Usize)
    (base i1 take : Std.Usize) (absmax : Std.U64) (s : Std.U64)
    (codeOf : Nat → Bool × Std.I8)
    (hblock : block.val < blocks.val)
    (hbase : block * chart.ACT_QK = ok base)
    (hsub : Slice.len x - base = ok i1)
    (hmin : core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take)
    (habsmax : chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax)
    (hscale : kern.pow2.act_scale absmax = ok (some s))
    (hsle : s.val ≤ 2 ^ 63 - 1)
    (hpush : scales.val.length < Usize.max)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes.val.length + take.val < Usize.max)
    (hadd : block.val + 1 ≤ Usize.max)
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
      block1.val = block.val + 1 ∧
      scales1.val = scales.val ++ [⟨s.bv⟩] ∧
      codes1.val = codes.val ++ (ffn4LaneStep codeOf 0 take.val).1 ∧
      ok2 = (ffn4LaneStep codeOf 0 take.val).2 := by
  obtain ⟨scales1, hpush1, hpush1v⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec scales ⟨s.bv⟩ hpush)
  obtain ⟨s2, hs2, _⟩ := ffn5_unsigned_abs_ok ⟨s.bv⟩
  obtain ⟨pow2, hpow2, _⟩ := WP.spec_imp_exists
    (show core.num.U64.is_power_of_two s2 ⦃ p => p = decide s2.val.isPowerOfTwo ⦄ from by
      rw [core.num.U64.is_power_of_two_val]; exact (WP.spec_ok _).mpr rfl)
  obtain ⟨shift, hshift, _⟩ := WP.spec_imp_exists
    (show core.num.U64.trailing_zeros s2 ⦃ u => u = ⟨BitVec.ofNat 32 (numGap_trailingZeros s2.bv)⟩ ⦄
      from by rw [core.num.U64.trailing_zeros_val]; exact (WP.spec_ok _).mpr rfl)
  have hl0 : (0#usize : Std.Usize).val ≤ take.val := by
    rw [show (0#usize : Std.Usize).val = 0 from rfl]; exact Nat.zero_le _
  obtain ⟨⟨codes1, ok2⟩, hlane, hlane1, hlane2⟩ := WP.spec_imp_exists
    (ffn4_lane_loop_spec x base take s2 pow2 shift (core.num.U64.wrapping_sub s2 1#u64) codes
      codeOf 0#usize hbound hcap hl0
      (hinner s2 pow2 shift (core.num.U64.wrapping_sub s2 1#u64)))
  obtain ⟨block1, hblock1, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) hadd)
  have hbody : chart.quant_act_loop0.body x blocks scales codes block true
      = ok (ControlFlow.cont (scales1, codes1, block1, ok2)) := by
    unfold chart.quant_act_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock)]
    rw [if_pos rfl]
    rw [hbase]; simp only [bind_tc_ok]
    rw [hsub]; simp only [bind_tc_ok]
    rw [hmin]; simp only [bind_tc_ok]
    rw [habsmax]; simp only [bind_tc_ok]
    rw [hscale]; simp only [bind_tc_ok]
    simp only [core.option.Option.unwrap_or_some, lift, bind_tc_ok]
    rw [core.num.I64.unsigned_abs_val]; simp only [bind_tc_ok]
    rw [if_neg (by rw [gt_iff_lt, UScalar.lt_equiv, ffn5_i64max_abs_val]; omega)]
    have hcast : core.num.U64.cast_signed s = ok ⟨s.bv⟩ := rfl
    rw [hcast]; simp only [bind_tc_ok]
    rw [hpush1]; simp only [bind_tc_ok]
    rw [hs2]; simp only [bind_tc_ok]
    rw [hpow2]; simp only [bind_tc_ok]
    rw [hshift]; simp only [lift, bind_tc_ok]
    rw [hlane]; simp only [bind_tc_ok]
    rw [hblock1]; simp only [bind_tc_ok]
    rfl
  refine ⟨scales1, codes1, ok2, block1, hbody, ?_, hpush1v, ?_, ?_⟩
  · rw [hblock1v]; rfl
  · have h0 : (0#usize : Std.Usize).val = 0 := rfl
    simpa only [h0, Nat.sub_zero] using hlane1
  · have h0 : (0#usize : Std.Usize).val = 0 := rfl
    simpa only [h0, Nat.sub_zero] using hlane2

/-! ## §1c · The value-level headline `word_quant_act_loop0_ok`

The per-block turn is `ffn6BlockTurnOf` — `ffn5BlockStep` at lane offset `0` and lane count the
block's `take` — so the recursive reference `ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf)` names the
pushed scales and lane codes by value. -/

/-- The concrete block turn: `ffn5BlockStep` with the block's scale `scaleOf b`, the lane code
`codeOf`, lane offset `0` and lane count `ffn5Take (len x) (b·32)`. Temporal cost `O(take)`; spatial
`O(take)` appends. -/
def ffn6BlockTurnOf (x : Slice Std.I64) (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64) :
    Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool :=
  fun b sc c => ffn5BlockStep sc c (scaleOf b) codeOf 0 (ffn5Take (Slice.len x).val (b * 32))

/-- Value of `ffn6BlockTurnOf`: it appends the block scale and the reference lane codes. -/
lemma ffn6BlockTurnOf_val (x : Slice Std.I64) (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64)
    (b : Nat) (sc : List Std.I64) (c : List Std.I8) :
    ffn6BlockTurnOf x codeOf scaleOf b sc c =
      (sc ++ [scaleOf b], c ++ (ffn4LaneStep codeOf 0 (ffn5Take (Slice.len x).val (b * 32))).1,
       (ffn4LaneStep codeOf 0 (ffn5Take (Slice.len x).val (b * 32))).2) := rfl

/-- `ffn6BlockTurnOf` appends exactly one scale. -/
lemma ffn6BlockTurnOf_len1 (x : Slice Std.I64) (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64)
    (b : Nat) (sc : List Std.I64) (c : List Std.I8) :
    (ffn6BlockTurnOf x codeOf scaleOf b sc c).1.length ≤ sc.length + 1 := by
  rw [ffn6BlockTurnOf_val]
  simp

/-- `ffn6BlockTurnOf` appends at most `32` lane codes (`ffn5Take` is at most `32`). -/
lemma ffn6BlockTurnOf_len2 (x : Slice Std.I64) (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64)
    (b : Nat) (sc : List Std.I64) (c : List Std.I8) :
    (ffn6BlockTurnOf x codeOf scaleOf b sc c).2.1.length ≤ c.length + 32 := by
  rw [ffn6BlockTurnOf_val]
  simp only [List.length_append]
  have h := ffn4LaneStep_fst_len codeOf 0 (ffn5Take (Slice.len x).val (b * 32))
  have htake : ffn5Take (Slice.len x).val (b * 32) ≤ 32 := Nat.min_le_left _ _
  omega

/-- `ffn5Blocks` growth bounds: with a turn that appends at most one scale and at most `32` codes,
the accumulated lengths are bounded by the start plus the turn count. -/
lemma ffn5Blocks_len (turnOf : Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool)
    (h1 : ∀ b sc c, (turnOf b sc c).1.length ≤ sc.length + 1)
    (h2 : ∀ b sc c, (turnOf b sc c).2.1.length ≤ c.length + 32)
    (scales : List Std.I64) (codes : List Std.I8) (b k : Nat) :
    (ffn5Blocks turnOf scales codes b k).1.length ≤ scales.length + k ∧
    (ffn5Blocks turnOf scales codes b k).2.1.length ≤ codes.length + 32 * k := by
  induction k with
  | zero => rw [ffn5Blocks_zero]; simp
  | succ k ih =>
    rw [ffn5Blocks_succ]
    by_cases hf : (ffn5Blocks turnOf scales codes b k).2.2 = true
    · rw [if_pos hf]
      refine ⟨?_, ?_⟩
      · calc (turnOf (b + k) (ffn5Blocks turnOf scales codes b k).1
              (ffn5Blocks turnOf scales codes b k).2.1).1.length
            ≤ (ffn5Blocks turnOf scales codes b k).1.length + 1 := h1 _ _ _
          _ ≤ (scales.length + k) + 1 := by omega
          _ = scales.length + (k + 1) := by omega
      · calc (turnOf (b + k) (ffn5Blocks turnOf scales codes b k).1
              (ffn5Blocks turnOf scales codes b k).2.1).2.1.length
            ≤ (ffn5Blocks turnOf scales codes b k).2.1.length + 32 := h2 _ _ _
          _ ≤ (codes.length + 32 * k) + 32 := by omega
          _ = codes.length + 32 * (k + 1) := by omega
    · rw [if_neg hf]
      exact ⟨by omega, by omega⟩

/-- **`chart.quant_act_loop0` denotes the block reference.** The extracted block loop from `block0`
with the flag `true` returns the reference `ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf)` when the
per-block sub-values are what `act_scale`, the absmax loop and the counter give, the two `Vec`s have
room over the whole run, and the lane code is `codeOf`. Composed from `ffn6_block_turn` (the
strengthened `word_quant_act_loop0_body_ok`) and the extracted stopping arms. `proved` relative to
`hinner` (the opaque externals). -/
theorem word_quant_act_loop0_ok (x : Slice Std.I64) (blocks : Std.Usize)
    (scales0 : alloc.vec.Vec Std.I64) (codes0 : alloc.vec.Vec Std.I8) (block0 : Std.Usize)
    (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64)
    (hle : block0.val ≤ blocks.val)
    (hscap0 : scales0.val.length + blocks.val < Usize.max)
    (hccap0 : codes0.val.length + 32 * blocks.val < Usize.max)
    (hscale : ∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len x - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧ s.val ≤ 2 ^ 63 - 1 ∧
          scaleOf block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          take.val = ffn5Take (Slice.len x).val (block.val * 32) ∧
          base.val + take.val ≤ x.val.length)
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
    chart.quant_act_loop0 x blocks scales0 codes0 block0 true ⦃ out =>
      out.1.val = (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).1 ∧
      out.2.1.val = (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.1 ∧
      out.2.2 = (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.2 ⦄ := by
  unfold chart.quant_act_loop0
  apply loop.spec_decr_nat
    (measure := fun st : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Std.Usize × Bool =>
      blocks.val - st.2.2.1.val)
    (inv := fun st : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Std.Usize × Bool =>
      block0.val ≤ st.2.2.1.val ∧ st.2.2.1.val ≤ blocks.val ∧
      (st.1.val, st.2.1.val, st.2.2.2)
        = ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
            (st.2.2.1.val - block0.val) ∧
      st.1.val.length ≤ scales0.val.length + (st.2.2.1.val - block0.val) ∧
      st.2.1.val.length ≤ codes0.val.length + 32 * (st.2.2.1.val - block0.val))
    (post := fun out : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Bool =>
      out.1.val = (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).1 ∧
      out.2.1.val = (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.1 ∧
      out.2.2 = (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.2)
  · rintro ⟨scales, codes, block, ok1⟩ ⟨hge0, hleB, hval, hls, hlc⟩
    dsimp only at hge0 hleB hval hls hlc ⊢
    by_cases hlt : block.val < blocks.val
    · cases ok1
      · -- flag cleared: the body stops
        rw [ffn5_block_body_stop x blocks scales codes block hlt, WP.spec_ok]
        dsimp only
        have hparts : ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
            (block.val - block0.val) = (scales.val, codes.val, false) := hval.symm
        have hstab := ffn5Blocks_false (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val
          block0.val (block.val - block0.val) (by rw [hparts]) (blocks.val - block.val)
        have hidx : blocks.val - block0.val
            = (block.val - block0.val) + (blocks.val - block.val) := by omega
        refine ⟨?_, ?_, ?_⟩
        · rw [hidx, hstab, hparts]
        · rw [hidx, hstab, hparts]
        · rw [hidx, hstab, hparts]
      · -- flag set: the block turn
        obtain ⟨base, i1, take, absmax, s, hbase, hsub, hmin, habsmax, hscalev, hsle, hsof, hadd,
          htakev, hbound⟩ := hscale block hlt
        have hsc : scales.val.length < Usize.max := by omega
        have htle : take.val ≤ 32 := by rw [htakev]; exact Nat.min_le_left _ _
        have hcc : codes.val.length + take.val < Usize.max := by omega
        obtain ⟨scales1, codes1, ok2, block1, hb, hb1, hsv, hcv, hokv⟩ :=
          ffn6_block_turn x blocks scales codes block base i1 take absmax s codeOf hlt hbase hsub hmin
            habsmax hscalev hsle hsc hbound hcc hadd hinner
        rw [hb, WP.spec_ok]
        dsimp only
        have hflag : (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
            (block.val - block0.val)).2.2 = true := by rw [← hval]
        have hparts : ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
            (block.val - block0.val) = (scales.val, codes.val, true) := hval.symm
        have htov : (scales1.val, codes1.val, ok2)
            = ffn6BlockTurnOf x codeOf scaleOf block.val scales.val codes.val := by
          rw [ffn6BlockTurnOf_val, hsv, hcv, hokv, hsof, htakev]
        refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · rw [hb1]; omega
        · rw [hb1]; omega
        · rw [htov, hb1, show block.val + 1 - block0.val
              = (block.val - block0.val) + 1 by omega,
            ffn5Blocks_cont (ffn6BlockTurnOf x codeOf scaleOf) scales0.val codes0.val block0.val
              (block.val - block0.val) hflag]
          rw [hparts, show block0.val + (block.val - block0.val) = block.val by omega]
        · rw [hsv, hb1, show block.val + 1 - block0.val
              = (block.val - block0.val) + 1 by omega, List.length_append]
          simp only [List.length_cons, List.length_nil]
          omega
        · have hlane := ffn4LaneStep_fst_len codeOf 0 take.val
          rw [hcv, hb1, show block.val + 1 - block0.val
              = (block.val - block0.val) + 1 by omega, List.length_append]
          omega
        · rw [hb1]; omega
    · -- counter at or past the stopping bound
      have hbeq : block.val = blocks.val := le_antisymm hleB (le_of_not_gt hlt)
      rw [ffn5_block_body_done x blocks scales codes block ok1 hlt, WP.spec_ok]
      dsimp only
      have hidx : blocks.val - block0.val = block.val - block0.val := by omega
      refine ⟨?_, ?_, ?_⟩ <;>
        rw [hidx, ← hval]
  · refine ⟨le_refl _, hle, ?_, ?_, ?_⟩
    · rw [Nat.sub_self, ffn5Blocks_zero]
    · rw [Nat.sub_self]; simp
    · rw [Nat.sub_self]; simp

/-- Non-vacuity of §1: one block turn appends its scale to the reference's scale list. -/
example : (ffn5Blocks (ffn6BlockTurnOf (Slice.new Std.I64) (fun _ => (true, 0#i8))
    (fun _ => 1#i64)) [] [] 0 1).1 = [1#i64] := rfl

/-! ## §2 · `chart.QuantAct.dequantizes` and its two nested loops

`src/engine/chart/mod.rs:120`. The driver rounds the code count up to blocks and walks the block
counter while each block's `scale·127` bound (or, when that bound leaves `i64`, each lane's
`scale·code`) keeps the running flag. The reference `ffn6DequantBlocks` states the given block
count's flag; `ffn6DequantBlockFlag` states one block's. -/

/-- The flag one lane contributes to the inner lane loop: the `i8` code is promoted to `i64`,
multiplied by the block's scale, and the lane's flag is whether that checked product is `some`; the
emitted code is unused. -/
def ffn6LaneCodeOf (codes : List Std.I8) (scale : Std.I64) (lane : Nat) : Bool × Std.I8 :=
  (core.option.Option.is_some
    (I64.checked_mul scale (core.convert.num.FromI64I8.from (codes.getD lane 0#i8))), 0#i8)

/-- One block's flag contribution: `false` when the block index is past the scale list (a missing
scale); the lane-loop flag when the block's `scale·127` bound leaves `i64` (only then can a lane
product fail); `true` otherwise. -/
def ffn6DequantBlockFlag (scales : List Std.I64) (codes : List Std.I8) (b : Nat) : Bool :=
  if scales.length ≤ b then false
  else
    let scale := scales.getD b 0#i64
    if core.option.Option.is_none (I64.checked_mul scale 127#i64) then
      (ffn4LaneStep (ffn6LaneCodeOf codes scale) (b * 32)
        (Nat.min (b * 32 + 32) codes.length - b * 32)).2
    else true

/-- The block loop's flag after `k` turns from block `b`, stopping once a block's flag clears (a
cleared flag stays cleared). Temporal cost `O(k)`; structural recursion on `k`. -/
def ffn6DequantBlocks (scales : List Std.I64) (codes : List Std.I8) (nb b : Nat) : Nat → Bool
  | 0 => true
  | k + 1 =>
    let st := ffn6DequantBlocks scales codes nb b k
    if st then (if b + k < nb then ffn6DequantBlockFlag scales codes (b + k) else true) else false

/-- `ffn6DequantBlocks` at zero turns is the initial flag. -/
lemma ffn6DequantBlocks_zero (scales : List Std.I64) (codes : List Std.I8) (nb b : Nat) :
    ffn6DequantBlocks scales codes nb b 0 = true := rfl

/-- One-step unfolding of `ffn6DequantBlocks`. -/
lemma ffn6DequantBlocks_succ (scales : List Std.I64) (codes : List Std.I8) (nb b k : Nat) :
    ffn6DequantBlocks scales codes nb b (k + 1) =
      (if ffn6DequantBlocks scales codes nb b k then
        (if b + k < nb then ffn6DequantBlockFlag scales codes (b + k) else true)
       else false) := rfl

/-- One continuation step: with the flag set and the counter inside the block count, the next
block's flag is taken. -/
lemma ffn6DequantBlocks_cont (scales : List Std.I64) (codes : List Std.I8) (nb b k : Nat)
    (h : ffn6DequantBlocks scales codes nb b k = true) (hk : b + k < nb) :
    ffn6DequantBlocks scales codes nb b (k + 1) = ffn6DequantBlockFlag scales codes (b + k) := by
  rw [ffn6DequantBlocks_succ, if_pos h, if_pos hk]

/-- A cleared flag stays cleared. -/
lemma ffn6DequantBlocks_false (scales : List Std.I64) (codes : List Std.I8) (nb b k : Nat)
    (h : ffn6DequantBlocks scales codes nb b k = false) (j : Nat) :
    ffn6DequantBlocks scales codes nb b (k + j) = false := by
  induction j with
  | zero => rw [Nat.add_zero]; exact h
  | succ j ih =>
    rw [show k + (j + 1) = (k + j) + 1 by omega, ffn6DequantBlocks_succ,
      if_neg (by rw [ih]; decide)]

/-- `32·b < n` when `b < ⌈n/32⌉`: a block inside the rounded-up count starts inside the codes. -/
lemma ffn6_lt_mul_of_lt_div_ceil_32 {b n : Nat} (hb : b < (n + 31) / 32) : 32 * b < n := by
  by_contra h
  have hle : n ≤ 32 * b := not_lt.mp h
  have h1 : (n + 31) / 32 ≤ (32 * b + 31) / 32 := Nat.div_le_div_right (by omega)
  have h2 : (32 * b + 31) / 32 = b := by omega
  omega

/-! ### §2a · The lane loop `chart.QuantAct.dequantizes_loop0_loop0` -/

/-- **One `dequantizes` lane body turn.** With `lane < e`, the flag set and the lane's code `i`, the
body continues with that product's flag and advances. `proved`. -/
theorem ffn6_dequant_lane_body_turn (v : alloc.vec.Vec Std.I8) (scale : Std.I64) (e : Std.Usize)
    (lane : Std.Usize) (i : Std.I8) (hlane : lane.val < e.val) (hend : e.val ≤ v.val.length)
    (hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I8) v lane = ok i) :
    ∃ lane1 : Std.Usize,
      chart.QuantAct.dequantizes_loop0_loop0.body v scale e true lane
        = ok (ControlFlow.cont
            (core.option.Option.is_some
              (I64.checked_mul scale (core.convert.num.FromI64I8.from i)), lane1)) ∧
      lane1.val = lane.val + 1 := by
  have hvl := alloc.vec.Vec.property v
  obtain ⟨lane1, hlane1, hlane1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize)
      (by rw [show (1#usize : Std.Usize).val = 1 from rfl]; omega))
  refine ⟨lane1, ?_, ?_⟩
  · unfold chart.QuantAct.dequantizes_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), if_pos rfl, hidx]
    simp only [lift, bind_tc_ok]
    rw [hlane1]; simp only [bind_tc_ok]
  · rw [hlane1v]; rfl

/-- The lane body at or past the end: it stops with the running flag. -/
theorem ffn6_dequant_lane_body_done (v : alloc.vec.Vec Std.I8) (scale : Std.I64) (e : Std.Usize)
    (ok1 : Bool) (lane : Std.Usize) (hge : ¬ lane.val < e.val) :
    chart.QuantAct.dequantizes_loop0_loop0.body v scale e ok1 lane
      = ok (ControlFlow.done ok1) := by
  unfold chart.QuantAct.dequantizes_loop0_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hge)]

/-- The lane body after a cleared flag: it stops with `false`. -/
theorem ffn6_dequant_lane_body_stop (v : alloc.vec.Vec Std.I8) (scale : Std.I64) (e : Std.Usize)
    (lane : Std.Usize) (hlane : lane.val < e.val) :
    chart.QuantAct.dequantizes_loop0_loop0.body v scale e false lane
      = ok (ControlFlow.done false) := by
  unfold chart.QuantAct.dequantizes_loop0_loop0.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), if_neg (by decide)]

/-- **The `dequantizes` lane loop denotes the reference lane flag.** With the segment `[l0, e)`
inside `v`, the loop from `l0` with the flag `true` returns the reference flag
`(ffn4LaneStep (ffn6LaneCodeOf v.val scale) l0 (e - l0)).2`. `proved`. -/
theorem word_dequantizes_loop0_loop0_ok (v : alloc.vec.Vec Std.I8) (scale : Std.I64)
    (e : Std.Usize) (l0 : Std.Usize) (hend : e.val ≤ v.val.length) (hl0 : l0.val ≤ e.val) :
    chart.QuantAct.dequantizes_loop0_loop0 v true scale e l0 ⦃ out =>
      out = (ffn4LaneStep (ffn6LaneCodeOf v.val scale) l0.val (e.val - l0.val)).2 ⦄ := by
  unfold chart.QuantAct.dequantizes_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Bool × Std.Usize => e.val - st.2.val)
    (inv := fun st : Bool × Std.Usize =>
      l0.val ≤ st.2.val ∧ st.2.val ≤ e.val ∧
      st.1 = (ffn4LaneStep (ffn6LaneCodeOf v.val scale) l0.val (st.2.val - l0.val)).2)
    (post := fun out : Bool =>
      out = (ffn4LaneStep (ffn6LaneCodeOf v.val scale) l0.val (e.val - l0.val)).2)
  · rintro ⟨b, lane⟩ ⟨h1, h2, h3⟩
    dsimp only at h1 h2 h3 ⊢
    by_cases hlt : lane.val < e.val
    · by_cases htrue : b = true
      · obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
          (alloc.vec.Vec.index_usize_spec v lane
            (by rw [show v.length = v.val.length from rfl]; omega))
        have hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I8) v lane
            = ok i := by
          rw [alloc.vec.Vec.index_slice_index]; exact hi
        obtain ⟨lane1, hbody, hlane1v⟩ :=
          ffn6_dequant_lane_body_turn v scale e lane i hlt hend hidx
        have hflag : (ffn4LaneStep (ffn6LaneCodeOf v.val scale) l0.val
            (lane.val - l0.val)).2 = true := by rw [← h3]; exact htrue
        rw [htrue, hbody, WP.spec_ok]
        dsimp only
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hlane1v]; omega
        · rw [hlane1v]; omega
        · rw [hlane1v, show lane.val + 1 - l0.val = (lane.val - l0.val) + 1 by omega,
            ffn4LaneStep_succ, if_pos hflag,
            show l0.val + (lane.val - l0.val) = lane.val by omega]
          simp only [ffn6LaneCodeOf]
          rw [List.getD_eq_getElem v.val 0#i8 (by omega), ← hiv]
        · rw [hlane1v]; omega
      · have hfalse : b = false := by cases b <;> simp_all
        have hflag : (ffn4LaneStep (ffn6LaneCodeOf v.val scale) l0.val
            (lane.val - l0.val)).2 = false := by rw [← h3]; exact hfalse
        have hstop := ffn4LaneStep_false (ffn6LaneCodeOf v.val scale) l0.val
          (lane.val - l0.val) hflag (e.val - lane.val)
        have hidx : e.val - l0.val = (lane.val - l0.val) + (e.val - lane.val) := by omega
        rw [hfalse, ffn6_dequant_lane_body_stop v scale e lane hlt, WP.spec_ok]
        dsimp only
        rw [hidx, hstop]
    · have heq : lane.val = e.val := le_antisymm h2 (le_of_not_gt hlt)
      rw [ffn6_dequant_lane_body_done v scale e b lane hlt, WP.spec_ok]
      dsimp only
      rw [heq] at h3
      exact h3
  · refine ⟨le_refl _, hl0, ?_⟩
    rw [Nat.sub_self, ffn4LaneStep_zero]

/-! ### §2b · The block loop `chart.QuantAct.dequantizes_loop0` -/

/-- **One `dequantizes` block body turn: the missing-scale arm.** With the block index inside the
block count but at or past the scale list, the body clears the flag and advances. `proved`. -/
theorem ffn6_dequant_body_miss (v : alloc.vec.Vec Std.I64) (v1 : alloc.vec.Vec Std.I8)
    (nb block : Std.Usize) (hblock : block.val < nb.val) (hge : v.val.length ≤ block.val)
    (hadd : block.val + 1 ≤ Usize.max) :
    ∃ block1 : Std.Usize,
      chart.QuantAct.dequantizes_loop0.body v v1 nb true block
        = ok (ControlFlow.cont (false, block1)) ∧ block1.val = block.val + 1 := by
  obtain ⟨block1, hblock1, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) hadd)
  refine ⟨block1, ?_, ?_⟩
  · unfold chart.QuantAct.dequantizes_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock), if_pos rfl]
    simp only [alloc.vec.Vec.len_val]
    rw [if_pos (by rw [ge_iff_le, UScalar.le_equiv]; exact hge)]
    rw [hblock1]; simp only [bind_tc_ok]
  · rw [hblock1v]; rfl

/-- **One `dequantizes` block body turn: the skip arm.** With the block inside the scale list and
its `scale·127` bound inside `i64`, the body leaves the flag set and advances. `proved`. -/
theorem ffn6_dequant_body_skip (v : alloc.vec.Vec Std.I64) (v1 : alloc.vec.Vec Std.I8)
    (nb block : Std.Usize) (sc : Std.I64) (hblock : block.val < nb.val)
    (hlt : block.val < v.val.length)
    (hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I64) v block = ok sc)
    (hsome : core.option.Option.is_none (I64.checked_mul sc 127#i64) = false)
    (hadd : block.val + 1 ≤ Usize.max) :
    ∃ block1 : Std.Usize,
      chart.QuantAct.dequantizes_loop0.body v v1 nb true block
        = ok (ControlFlow.cont (true, block1)) ∧ block1.val = block.val + 1 := by
  obtain ⟨block1, hblock1, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) hadd)
  refine ⟨block1, ?_, ?_⟩
  · unfold chart.QuantAct.dequantizes_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock), if_pos rfl]
    simp only [alloc.vec.Vec.len_val]
    have hnot : ¬ block ≥ alloc.vec.Vec.len v := by
      rw [ge_iff_le, UScalar.le_equiv, alloc.vec.Vec.len_val]
      exact not_le.mpr hlt
    rw [if_neg hnot]
    rw [hidx]; simp only [bind_tc_ok, lift]
    rw [if_neg (by rw [hsome]; decide)]
    rw [hblock1]; simp only [bind_tc_ok]
  · rw [hblock1v]; rfl

/-- **One `dequantizes` block body turn: the lane arm.** With the block inside the scale list and
its `scale·127` bound leaving `i64`, the body runs the lane loop over `[base, e)` and advances with
that loop's flag. `proved`. -/
theorem ffn6_dequant_body_lane (v : alloc.vec.Vec Std.I64) (v1 : alloc.vec.Vec Std.I8)
    (nb block : Std.Usize) (sc : Std.I64) (base i1 e : Std.Usize)
    (hblock : block.val < nb.val) (hlt : block.val < v.val.length)
    (hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I64) v block = ok sc)
    (hnone : core.option.Option.is_none (I64.checked_mul sc 127#i64) = true)
    (hbase : block * chart.ACT_QK = ok base) (hbasev : base.val = block.val * 32)
    (hi1 : base + chart.ACT_QK = ok i1) (hi1v : i1.val = base.val + 32)
    (he : core.cmp.Ord.min.trait_default core.cmp.OrdUsize i1 (alloc.vec.Vec.len v1) = ok e)
    (hev : e.val = Nat.min i1.val v1.val.length)
    (hadd : block.val + 1 ≤ Usize.max) (hbl : base.val ≤ e.val) (hend : e.val ≤ v1.val.length) :
    ∃ block1 : Std.Usize, ∃ flag : Bool,
      chart.QuantAct.dequantizes_loop0.body v v1 nb true block
        = ok (ControlFlow.cont (flag, block1)) ∧ block1.val = block.val + 1 ∧
      flag = (ffn4LaneStep (ffn6LaneCodeOf v1.val sc) base.val (e.val - base.val)).2 := by
  obtain ⟨flag, hflag, hflagv⟩ := WP.spec_imp_exists
    (word_dequantizes_loop0_loop0_ok v1 sc e base hend hbl)
  obtain ⟨block1, hblock1, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) hadd)
  refine ⟨block1, flag, ?_, ?_, hflagv⟩
  · unfold chart.QuantAct.dequantizes_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock), if_pos rfl]
    simp only [alloc.vec.Vec.len_val]
    have hnot : ¬ block ≥ alloc.vec.Vec.len v := by
      rw [ge_iff_le, UScalar.le_equiv, alloc.vec.Vec.len_val]
      exact not_le.mpr hlt
    rw [if_neg hnot]
    rw [hidx]; simp only [bind_tc_ok, lift]
    rw [if_pos hnone]
    rw [hbase]; simp only [bind_tc_ok]
    rw [hi1]; simp only [bind_tc_ok]
    rw [he]; simp only [bind_tc_ok]
    rw [hflag]; simp only [bind_tc_ok]
    rw [hblock1]; simp only [bind_tc_ok]
  · rw [hblock1v]; rfl

/-- **One `dequantizes` block body turn at or past the block count.** -/
theorem ffn6_dequant_body_end (v : alloc.vec.Vec Std.I64) (v1 : alloc.vec.Vec Std.I8)
    (nb block : Std.Usize) (ok1 : Bool) (hge : ¬ block.val < nb.val) :
    chart.QuantAct.dequantizes_loop0.body v v1 nb ok1 block
      = ok (ControlFlow.done ok1) := by
  unfold chart.QuantAct.dequantizes_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hge)]

/-- **One `dequantizes` block body turn after a cleared flag.** -/
theorem ffn6_dequant_body_stop (v : alloc.vec.Vec Std.I64) (v1 : alloc.vec.Vec Std.I8)
    (nb block : Std.Usize) (hblock : block.val < nb.val) :
    chart.QuantAct.dequantizes_loop0.body v v1 nb false block
      = ok (ControlFlow.done false) := by
  unfold chart.QuantAct.dequantizes_loop0.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hblock), if_neg (by decide)]

/-- **The `dequantizes` block loop denotes `ffn6DequantBlocks`.** With the block count no larger
than the rounded-up code count and the code capacity carrying `31` words of headroom (the lane arm's
`block·32 + 32` stays inside `usize`), the loop from `b0` with the flag `true` returns the reference
flag `ffn6DequantBlocks` over the given block count. `proved`. -/
theorem word_dequantizes_loop0_ok (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8)
    (nb b0 : Std.Usize) (hle : b0.val ≤ nb.val)
    (hnb : nb.val ≤ (codes.val.length + 31) / 32)
    (hcap : codes.val.length + 31 ≤ Usize.max) :
    chart.QuantAct.dequantizes_loop0 scales codes nb true b0 ⦃ out =>
      out = ffn6DequantBlocks scales.val codes.val nb.val b0.val (nb.val - b0.val) ⦄ := by
  unfold chart.QuantAct.dequantizes_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Bool × Std.Usize => nb.val - st.2.val)
    (inv := fun st : Bool × Std.Usize =>
      b0.val ≤ st.2.val ∧ st.2.val ≤ nb.val ∧
      st.1 = ffn6DequantBlocks scales.val codes.val nb.val b0.val (st.2.val - b0.val))
    (post := fun out : Bool =>
      out = ffn6DequantBlocks scales.val codes.val nb.val b0.val (nb.val - b0.val))
  · rintro ⟨b, block⟩ ⟨h1, h2, h3⟩
    dsimp only at h1 h2 h3 ⊢
    by_cases hlt : block.val < nb.val
    · by_cases htrue : b = true
      · have hk : b0.val + (block.val - b0.val) = block.val := by omega
        have hkb : b0.val + (block.val - b0.val) < nb.val := by omega
        have hcref : ffn6DequantBlocks scales.val codes.val nb.val b0.val
            (block.val - b0.val) = true := by rw [← h3]; exact htrue
        have hstep := ffn6DequantBlocks_cont scales.val codes.val nb.val b0.val
          (block.val - b0.val) hcref hkb
        have hblk : block.val * 32 < codes.val.length := by
          have hb32 : block.val < (codes.val.length + 31) / 32 := by omega
          have := ffn6_lt_mul_of_lt_div_ceil_32 hb32
          omega
        have hcmax : codes.val.length + 31 ≤ Usize.max := hcap
        by_cases hge : scales.val.length ≤ block.val
        · -- the missing-scale arm
          have hb1 : block.val + 1 ≤ Usize.max := by omega
          obtain ⟨block1, hbody, hblock1v⟩ :=
            ffn6_dequant_body_miss scales codes nb block hlt hge hb1
          have href : ffn6DequantBlocks scales.val codes.val nb.val b0.val (block.val - b0.val + 1)
              = false := by
            rw [hstep, hk]
            unfold ffn6DequantBlockFlag
            rw [if_pos hge]
          rw [htrue, hbody, WP.spec_ok]
          dsimp only
          refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
          · rw [hblock1v]; omega
          · rw [hblock1v]; omega
          · rw [hblock1v, show block.val + 1 - b0.val = (block.val - b0.val) + 1 by omega, href]
          · rw [hblock1v]; omega
        · -- the block is inside the scale list
          have hblk2 : block.val < scales.val.length := lt_of_not_ge hge
          obtain ⟨sc, hsc1, hscv⟩ := WP.spec_imp_exists
            (alloc.vec.Vec.index_usize_spec scales block (by omega))
          have hidx : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice Std.I64) scales
              block = ok sc := by rw [alloc.vec.Vec.index_slice_index]; exact hsc1
          have hget : scales.val.getD block.val 0#i64 = sc := by
            rw [List.getD_eq_getElem scales.val 0#i64 (by omega), ← hscv]
          by_cases hnone : core.option.Option.is_none (I64.checked_mul sc 127#i64) = true
          · -- the lane arm
            have hb1 : block.val + 1 ≤ Usize.max := by
              have := alloc.vec.Vec.property scales; omega
            obtain ⟨base, hbase, hbasev⟩ := WP.spec_imp_exists
              (Usize.mul_spec (x := block) (y := chart.ACT_QK)
                (by rw [ffn5_act_qk_val]; omega))
            have hbasev' : base.val = block.val * 32 := by rw [hbasev, ffn5_act_qk_val]
            obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
              (Usize.add_spec (x := base) (y := chart.ACT_QK)
                (by rw [ffn5_act_qk_val, hbasev']; omega))
            have hi1v' : i1.val = base.val + 32 := by rw [hi1v, ffn5_act_qk_val]
            obtain ⟨e, he, hev⟩ := ordUsize_min_spec i1 (alloc.vec.Vec.len codes)
            have hev' : e.val = Nat.min i1.val codes.val.length := by
              rw [hev, alloc.vec.Vec.len_val]
            have hend : e.val ≤ codes.val.length := by rw [hev']; exact Nat.min_le_right _ _
            have hbl : base.val ≤ e.val := by
              rw [hev', hbasev']
              exact le_min (by omega) (by omega)
            obtain ⟨block1, flag, hbody, hblock1v, hflagv⟩ :=
              ffn6_dequant_body_lane scales codes nb block sc base i1 e hlt hblk2 hidx hnone
                hbase hbasev' hi1 hi1v' he hev' hb1 hbl hend
            have href : ffn6DequantBlocks scales.val codes.val nb.val b0.val (block.val - b0.val + 1)
                = (ffn4LaneStep (ffn6LaneCodeOf codes.val sc) (block.val * 32)
                    (Nat.min (block.val * 32 + 32) codes.val.length
                      - block.val * 32)).2 := by
              rw [hstep, hk]
              unfold ffn6DequantBlockFlag
              rw [if_neg (by omega), if_pos (by rw [hget]; exact hnone), hget]
            rw [htrue, hbody, WP.spec_ok]
            dsimp only
            refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
            · rw [hblock1v]; omega
            · rw [hblock1v]; omega
            · rw [hblock1v, show block.val + 1 - b0.val = (block.val - b0.val) + 1 by omega,
                href, hflagv, hbasev', hev', hi1v', hbasev']
            · rw [hblock1v]; omega
          · -- the skip arm
            have hfalse : core.option.Option.is_none (I64.checked_mul sc 127#i64) = false := by
              cases h : core.option.Option.is_none (I64.checked_mul sc 127#i64) <;> simp_all
            have hb1 : block.val + 1 ≤ Usize.max := by
              have := alloc.vec.Vec.property scales; omega
            obtain ⟨block1, hbody, hblock1v⟩ :=
              ffn6_dequant_body_skip scales codes nb block sc hlt (by omega) hidx hfalse hb1
            have href : ffn6DequantBlocks scales.val codes.val nb.val b0.val (block.val - b0.val + 1)
                = true := by
              rw [hstep, hk]
              unfold ffn6DequantBlockFlag
              rw [if_neg (by omega), if_neg (by rw [hget, hfalse]; decide)]
            rw [htrue, hbody, WP.spec_ok]
            dsimp only
            refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
            · rw [hblock1v]; omega
            · rw [hblock1v]; omega
            · rw [hblock1v, show block.val + 1 - b0.val = (block.val - b0.val) + 1 by omega, href]
            · rw [hblock1v]; omega
      · -- the flag is cleared: the body stops
        have hfalse : b = false := by cases b <;> simp_all
        have hcref : ffn6DequantBlocks scales.val codes.val nb.val b0.val (block.val - b0.val)
            = false := by rw [← h3]; exact hfalse
        have hstop := ffn6DequantBlocks_false scales.val codes.val nb.val b0.val
          (block.val - b0.val) hcref (nb.val - block.val)
        have hidx : nb.val - b0.val = (block.val - b0.val) + (nb.val - block.val) := by omega
        rw [hfalse, ffn6_dequant_body_stop scales codes nb block hlt, WP.spec_ok]
        dsimp only
        rw [hidx, hstop]
    · -- the counter reached the block count
      have heq : block.val = nb.val := le_antisymm h2 (le_of_not_gt hlt)
      rw [ffn6_dequant_body_end scales codes nb block b hlt, WP.spec_ok]
      dsimp only
      rw [heq] at h3
      exact h3
  · refine ⟨le_refl _, hle, ?_⟩
    rw [Nat.sub_self, ffn6DequantBlocks_zero]

/-! ### §2c · The driver `chart.QuantAct.dequantizes` -/

/-- The extracted `div_ceil n ACT_QK` succeeds with value `⌈n/32⌉ = (n + 31)/32`. -/
theorem ffn6_div_ceil_32_ok (n : Std.Usize) :
    ∃ blocks : Std.Usize, core.num.Usize.div_ceil n chart.ACT_QK = ok blocks ∧
      blocks.val = (n.val + 31) / 32 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := chart.ACT_QK) (by rw [ffn5_act_qk]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := chart.ACT_QK) (by rw [ffn5_act_qk]; decide))
  have hqv' : q.val = n.val / 32 := by rw [hqv, ffn5_act_qk_val]
  have hrv' : r.val = n.val % 32 := by rw [hrv, ffn5_act_qk_val]
  have hq' : UScalar.div n chart.ACT_QK = ok q := hq
  have hr' : UScalar.rem n chart.ACT_QK = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 32 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Std.Usize))
        (by rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]; exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 32 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Std.Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- **`chart.QuantAct.dequantizes` denotes the block reference.** With the code capacity carrying
`31` words of headroom, the extracted driver returns the `ffn6DequantBlocks` flag over the
rounded-up block count. `proved`. -/
theorem word_dequantizes_ok (self : chart.QuantAct)
    (hcap : self.codes.val.length + 31 ≤ Usize.max) :
    chart.QuantAct.dequantizes self = ok (ffn6DequantBlocks self.scales.val self.codes.val
      ((self.codes.val.length + 31) / 32) 0 ((self.codes.val.length + 31) / 32)) := by
  obtain ⟨nb, hdiv, hnbv⟩ := ffn6_div_ceil_32_ok (alloc.vec.Vec.len self.codes)
  have hlen : (alloc.vec.Vec.len self.codes).val = self.codes.val.length :=
    alloc.vec.Vec.len_val self.codes
  obtain ⟨out, hout, hval⟩ := WP.spec_imp_exists
    (word_dequantizes_loop0_ok self.scales self.codes nb 0#usize (Nat.zero_le _)
      (by rw [hnbv, hlen]) hcap)
  unfold chart.QuantAct.dequantizes
  dsimp only
  rw [hdiv]; simp only [bind_tc_ok]
  rw [hout, hval, hnbv, ← hlen]
  have hz : (↑(0#usize : Std.Usize) : Nat) = 0 := by decide
  simp only [hz, Nat.sub_zero]

/-- **The refusal.** When the reference flag is cleared, `chart.QuantAct.dequantizes` returns
`false`: the `Ok` case is exactly the reference's. -/
theorem word_dequantizes_refuses (self : chart.QuantAct)
    (hcap : self.codes.val.length + 31 ≤ Usize.max)
    (h : ffn6DequantBlocks self.scales.val self.codes.val
        ((self.codes.val.length + 31) / 32) 0 ((self.codes.val.length + 31) / 32) = false) :
    chart.QuantAct.dequantizes self = ok false := by
  rw [word_dequantizes_ok self hcap, h]

/-- Non-vacuity of §2: the block flag clears on a missing scale, stays set on a fitting bound, and
the block reference stops there. -/
example : ffn6DequantBlockFlag [1#i64] [3#i8, 4#i8] 0 = true ∧
    ffn6DequantBlockFlag [1#i64] [3#i8] 1 = false ∧
    ffn6DequantBlocks [1#i64] [3#i8, 4#i8] 1 0 1 = true ∧
    ffn6DequantBlocks [1#i64] [3#i8] 2 0 2 = false := by
  refine ⟨by decide, by decide, by decide, by decide⟩

/-! ## §3 · The driver `chart.quant_act`

`src/engine/chart/mod.rs:250`. The driver rounds the code count up to blocks, runs the block loop from
a fresh `with_capacity` state, and returns the `QuantAct` of the pushed scales and codes only when the
block flag and the dequant block flag both stay set. The reference `ffn6QuantActRef` is the block
reference gated by the dequant block reference over the pushed code count. -/

/-- The `chart.quant_act` reference over the block count `nb`: the block reference's final state, gated
by the dequant block flag over the pushed code count, `none` when either flag clears. Temporal
`O(nb)` turns plus the turns' lane costs; spatial `O(codes)` on the pushed lists; the sub-calls are
`ffn5Blocks` and `ffn6DequantBlocks`. -/
def ffn6QuantActRef (x : Slice Std.I64) (nb : Nat) (codeOf : Nat → Bool × Std.I8)
    (scaleOf : Nat → Std.I64) : Option (List Std.I64 × List Std.I8) :=
  let st := ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 nb
  if st.2.2 then
    if ffn6DequantBlocks st.1 st.2.1 ((st.2.1.length + 31) / 32) 0 ((st.2.1.length + 31) / 32)
    then some (st.1, st.2.1) else none
  else none

/-- **`chart.quant_act` denotes the reference.** With the block count from the extracted `div_ceil`,
the per-block sub-values from `hscale`, the lane code from `hinner`, and the code capacity carrying
`32` words of headroom over the block count, the extracted driver returns the `QuantAct` of the loop's
pushed scales and codes exactly when `ffn6QuantActRef` is `some` of them. Composed from
`word_quant_act_loop0_ok` and `word_dequantizes_ok`. `proved` relative to `hinner` (the opaque
externals). -/
theorem word_quant_act_ok (x : Slice Std.I64) (blocks : Std.Usize)
    (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64)
    (hdiv : core.num.Usize.div_ceil (Slice.len x) chart.ACT_QK = ok blocks)
    (hcap : 32 * blocks.val + 32 ≤ Usize.max)
    (hscale : ∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len x - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧ s.val ≤ 2 ^ 63 - 1 ∧
          scaleOf block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          take.val = ffn5Take (Slice.len x).val (block.val * 32) ∧
          base.val + take.val ≤ x.val.length)
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
      (∀ p : List Std.I64 × List Std.I8,
        ffn6QuantActRef x blocks.val codeOf scaleOf = some p → (scales1.val, codes1.val) = p) ∧
      chart.quant_act x =
        ok (Option.map (fun _ => ({ scales := scales1, codes := codes1 } : chart.QuantAct))
          (ffn6QuantActRef x blocks.val codeOf scaleOf)) := by
  have hscz : (alloc.vec.Vec.with_capacity Std.I64 blocks).val = ([] : List Std.I64) := rfl
  have hccz : (alloc.vec.Vec.with_capacity Std.I8 (Slice.len x)).val = ([] : List Std.I8) := rfl
  have hscap0 : (alloc.vec.Vec.with_capacity Std.I64 blocks).val.length + blocks.val < Usize.max := by
    rw [hscz]; simp only [List.length_nil, Nat.zero_add]; omega
  have hccap0 : (alloc.vec.Vec.with_capacity Std.I8 (Slice.len x)).val.length
      + 32 * blocks.val < Usize.max := by
    rw [hccz]; simp only [List.length_nil, Nat.zero_add]; omega
  obtain ⟨out, hout, h1, h2, h3⟩ := WP.spec_imp_exists
    (word_quant_act_loop0_ok x blocks (alloc.vec.Vec.with_capacity Std.I64 blocks)
      (alloc.vec.Vec.with_capacity Std.I8 (Slice.len x)) 0#usize codeOf scaleOf
      (by simp) hscap0 hccap0 hscale hinner)
  rw [hscz, hccz, show (0#usize : Std.Usize).val = 0 from rfl, Nat.sub_zero] at h1 h2 h3
  obtain ⟨scales1, codes1, ok1⟩ := out
  dsimp only at h1 h2 h3
  have hstlen : (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val).2.1.length
      ≤ 32 * blocks.val := by
    have h := (ffn5Blocks_len (ffn6BlockTurnOf x codeOf scaleOf)
      (ffn6BlockTurnOf_len1 x codeOf scaleOf) (ffn6BlockTurnOf_len2 x codeOf scaleOf)
      ([] : List Std.I64) ([] : List Std.I8) 0 blocks.val).2
    simpa using h
  have hcapdf : codes1.val.length + 31 ≤ Usize.max := by rw [h2]; omega
  have hdf := word_dequantizes_ok ({ scales := scales1, codes := codes1 } : chart.QuantAct) hcapdf
  refine ⟨scales1, codes1, ?_, ?_⟩
  · intro p hp
    rw [h1, h2]
    unfold ffn6QuantActRef at hp
    dsimp only at hp
    by_cases h22 : (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val).2.2 = true
    · rw [if_pos h22] at hp
      by_cases hF : (ffn6DequantBlocks (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0
            blocks.val).1 (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val).2.1
          (((ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val).2.1.length + 31) / 32)
          0 (((ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val).2.1.length + 31)
            / 32)) = true
      · rw [if_pos hF] at hp
        exact Option.some.inj hp
      · rw [if_neg hF] at hp
        exact absurd hp (by simp)
    · rw [if_neg h22] at hp
      exact absurd hp (by simp)
  · unfold chart.quant_act
    dsimp only
    rw [hdiv]
    simp only [bind_tc_ok]
    rw [hout]
    simp only [bind_tc_ok]
    rw [h3]
    show (if (ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val).2.2 = true then
        (do let b ← ({ scales := scales1, codes := codes1 } : chart.QuantAct).dequantizes
            if b = true then ok (some ({ scales := scales1, codes := codes1 } : chart.QuantAct))
            else ok none)
        else ok none)
      = ok (Option.map (fun _ => ({ scales := scales1, codes := codes1 } : chart.QuantAct))
          (ffn6QuantActRef x blocks.val codeOf scaleOf))
    unfold ffn6QuantActRef
    set st : List Std.I64 × List Std.I8 × Bool :=
      ffn5Blocks (ffn6BlockTurnOf x codeOf scaleOf) [] [] 0 blocks.val with hstdef
    rw [hdf]
    simp only [bind_tc_ok]
    rw [h1, h2]
    by_cases h22 : st.2.2 = true
    · rw [if_pos h22, if_pos h22]
      by_cases hF : (ffn6DequantBlocks st.1 st.2.1 ((st.2.1.length + 31) / 32) 0
          ((st.2.1.length + 31) / 32)) = true
      · rw [if_pos hF, if_pos hF]; rfl
      · rw [if_neg hF, if_neg hF]; rfl
    · rw [if_neg h22, if_neg h22]; rfl

/-- **The refusal.** When the reference is `none`, `chart.quant_act` returns `none`: the `Ok` case is
exactly the reference's. -/
theorem word_quant_act_refuses (x : Slice Std.I64) (blocks : Std.Usize)
    (codeOf : Nat → Bool × Std.I8) (scaleOf : Nat → Std.I64)
    (hdiv : core.num.Usize.div_ceil (Slice.len x) chart.ACT_QK = ok blocks)
    (hcap : 32 * blocks.val + 32 ≤ Usize.max)
    (hscale : ∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len x - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧ s.val ≤ 2 ^ 63 - 1 ∧
          scaleOf block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          take.val = ffn5Take (Slice.len x).val (block.val * 32) ∧
          base.val + take.val ≤ x.val.length)
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
        ok (b, i3)) = ok (codeOf lane.val))
    (h : ffn6QuantActRef x blocks.val codeOf scaleOf = none) :
    chart.quant_act x = ok none := by
  obtain ⟨_, _, _, hm⟩ := word_quant_act_ok x blocks codeOf scaleOf hdiv hcap hscale hinner
  rw [h] at hm
  simpa using hm

/-- Non-vacuity of §3: with no blocks the driver reference is `some` of the empty pushed lists. -/
example : ffn6QuantActRef (Slice.new Std.I64) 0 (fun _ => (true, 0#i8)) (fun _ => 0#i64) =
    some ([], []) := by decide

end apeiron_word
