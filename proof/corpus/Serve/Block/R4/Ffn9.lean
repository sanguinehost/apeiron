/-
  `Serve/Block/R4/Ffn9.lean` — re-derives the three `Serve/Block/R4/Ffn6.lean` block-level
  headlines over the value-indexed lane loop of `Serve/Block/R4/Ffn8.lean`, with no unsatisfiable
  premise.

  `Ffn6`'s `word_quant_act_loop0_ok`, `word_quant_act_ok` and `word_quant_act_refuses` carry
  `hinner : ∀ s pow2 shift mask lane v, <inner code at v> = ok (codeOf lane.val)` with a lane-only
  `codeOf : Nat → Bool × Std.I8`. `Ffn7`'s `ffn7_hinner_false` proves no such `codeOf` exists — the
  code is a function of the lane's value — so those three headlines hold only vacuously.

  This file replaces the lane-only code by a **closed function of the block's scale**
  (`ffn9CodeOf`), which the extracted body reads as the lane loop's scale. The block turn
  (`ffn9_block_turn`) resolves the lane loop through `Ffn8`'s `ffn8_lane_loop_ok_ffn7`, so the code
  is discharged from `ffn7_lane_code_eq`'s bounds rather than assumed.

  Group coverage:
  * §1 `ffn9CodeOf` (the per-block code as a function of the block's `i64` scale) and the block turn
    reference `ffn9BlockTurnOf`.
  * §2 `ffn9_block_turn` — one `chart.quant_act_loop0.body` turn with the pushed scale and the
    value-indexed lane codes, no `hinner`.
  * §3 `word_quant_act_loop0_ok9` — the block loop denotes `ffn5Blocks (ffn9BlockTurnOf x scaleOf)`.
  * §4 `ffn9QuantActRef`, `word_quant_act_ok9`, `word_quant_act_refuses9` — the driver.

  `hscale` survives as an explicit hypothesis, restated with the three lane-loop bounds of
  `ffn7_lane_code_eq`. Those bounds are properties of the body's computed scale; §5 exhibits a
  concrete input where the whole bundle holds jointly, so the hypothesis is satisfiable and the
  headlines are non-vacuous.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains` are definitions in the
    pinned model; no proof step invokes an axiom.
  * the `F32`/`AVX2` bodies, the pool dispatch, the `unsafe` pointer writes and the growable `Vec`
    allocation: named boundaries.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch or the two model externals. The statements are about the
  Aeneas-extracted bodies over the model `Slice`, `Vec` and word scalars.
-/

import ApeironWord
import Serve.Block.R4.Ffn8

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §1 · The per-block value-indexed code and the block turn

`ffn9Uabs`, `ffn9Pow2Of`, `ffn9ShiftOf`, `ffn9MaskOf` are the four word values the body derives
from the chosen `i64` scale `z`: `z`'s magnitude as a `u64` (`unsigned_abs`), its power-of-two test,
its trailing zeros and its `z - 1` mask (`wrapping_sub`). They are the same terms the extracted body
computes from `scale = cast_signed s`, so `ffn9CodeOf`'s lane code is exactly `ffn7Code` at those
four values, one code per block. -/

/-- The magnitude of an `i64` scale word as a `u64`: the result `core.num.I64.unsigned_abs` returns.
O(1): one `natAbs`. -/
def ffn9Uabs (z : Std.I64) : Std.U64 := ⟨BitVec.ofNat 64 z.val.natAbs⟩

/-- The power-of-two test of the scale's magnitude, as the body's `is_power_of_two` computes it. -/
def ffn9Pow2Of (z : Std.I64) : Bool := decide (ffn9Uabs z).val.isPowerOfTwo

/-- The scale's magnitude's trailing zeros, as the body's `trailing_zeros` computes them. -/
def ffn9ShiftOf (z : Std.I64) : Std.U32 :=
  ⟨BitVec.ofNat 32 (numGap_trailingZeros (ffn9Uabs z).bv)⟩

/-- The scale's magnitude minus one: the body's `wrapping_sub magnitude 1`. -/
def ffn9MaskOf (z : Std.I64) : Std.U64 := core.num.U64.wrapping_sub (ffn9Uabs z) 1#u64

/-- The per-lane code as a closed function of the block's scale: `Ffn8`'s concrete `ffn8CodeOf` at
the scale's magnitude, power-of-two flag, trailing zeros and mask. Lane- and value-indexed. -/
def ffn9CodeOf (z : Std.I64) : Nat → Std.I64 → Bool × Std.I8 :=
  ffn8CodeOf (ffn9Uabs z) (ffn9Pow2Of z) (ffn9ShiftOf z) (ffn9MaskOf z)

/-- The block turn: block `b` pushes its scale `scaleOf b` and the value-indexed lane codes of the
segment `[b·32, b·32 + take)` with `take = ffn5Take (len x) (b·32)`, carrying the lane loop's final
flag. Temporal cost `O(take)` appends; spatial `O(take)`; `ffn8LaneStep` is one step per lane. -/
def ffn9BlockTurnOf (x : Slice Std.I64) (scaleOf : Nat → Std.I64) :
    Nat → List Std.I64 → List Std.I8 → List Std.I64 × List Std.I8 × Bool :=
  fun b sc c =>
    (sc ++ [scaleOf b],
     c ++ (ffn8LaneStep (ffn9CodeOf (scaleOf b)) x (b * 32) 0
       (ffn5Take (Slice.len x).val (b * 32))).1,
     (ffn8LaneStep (ffn9CodeOf (scaleOf b)) x (b * 32) 0
       (ffn5Take (Slice.len x).val (b * 32))).2)

/-- Value of `ffn9BlockTurnOf`: the scale appended to the scale list and the value-indexed lane codes
to the code list, with the lane loop's flag. -/
lemma ffn9BlockTurnOf_val (x : Slice Std.I64) (scaleOf : Nat → Std.I64) (b : Nat)
    (sc : List Std.I64) (c : List Std.I8) :
    ffn9BlockTurnOf x scaleOf b sc c =
      (sc ++ [scaleOf b],
       c ++ (ffn8LaneStep (ffn9CodeOf (scaleOf b)) x (b * 32) 0
         (ffn5Take (Slice.len x).val (b * 32))).1,
       (ffn8LaneStep (ffn9CodeOf (scaleOf b)) x (b * 32) 0
         (ffn5Take (Slice.len x).val (b * 32))).2) := rfl

/-- `ffn9BlockTurnOf` appends exactly one scale. -/
lemma ffn9BlockTurnOf_len1 (x : Slice Std.I64) (scaleOf : Nat → Std.I64) (b : Nat)
    (sc : List Std.I64) (c : List Std.I8) :
    (ffn9BlockTurnOf x scaleOf b sc c).1.length ≤ sc.length + 1 := by
  rw [ffn9BlockTurnOf_val]
  simp

/-- `ffn9BlockTurnOf` appends at most `32` lane codes (`ffn5Take` is at most `32`). -/
lemma ffn9BlockTurnOf_len2 (x : Slice Std.I64) (scaleOf : Nat → Std.I64) (b : Nat)
    (sc : List Std.I64) (c : List Std.I8) :
    (ffn9BlockTurnOf x scaleOf b sc c).2.1.length ≤ c.length + 32 := by
  rw [ffn9BlockTurnOf_val]
  simp only [List.length_append]
  have h := ffn8LaneStep_fst_len (ffn9CodeOf (scaleOf b)) x (b * 32) 0
    (ffn5Take (Slice.len x).val (b * 32))
  have htake : ffn5Take (Slice.len x).val (b * 32) ≤ 32 := Nat.min_le_left _ _
  omega

/-! ## §2 · One `chart.quant_act_loop0.body` turn, value-indexed

The body computes, from the block's scale `s` (`scale = cast_signed s`), the magnitude
`unsigned_abs scale`, its power-of-two flag and trailing zeros, and the mask `magnitude - 1`, then
runs the lane loop. `ffn8_lane_loop_ok_ffn7` resolves the loop to `ffn8LaneStep` of the concrete
`ffn8CodeOf`, whose four arguments are exactly `ffn9Uabs ⟨s.bv⟩`, `ffn9Pow2Of ⟨s.bv⟩`,
`ffn9ShiftOf ⟨s.bv⟩`, `ffn9MaskOf ⟨s.bv⟩`. No `hinner` premise: the code is discharged from
`ffn7_lane_code_eq`'s bounds, carried as hypotheses about the body's computed scale. -/

/-- **One `chart.quant_act_loop0.body` turn, value-indexed.** Given the block's `base`, `i1`,
`take`, absmax and scale `s` (the sub-values the body computes), `scaleOf block = ⟨s.bv⟩`, the three
lane-loop bounds of `ffn7_lane_code_eq` about that scale, and room in both `Vec`s, the body takes the
non-refusing branch, pushes the scale, runs the lane loop and advances. The result is exactly
`ffn9BlockTurnOf x scaleOf block.val`'s contribution. No `hinner`. -/
theorem ffn9_block_turn (x : Slice Std.I64) (blocks : Std.Usize)
    (scales : alloc.vec.Vec Std.I64) (codes : alloc.vec.Vec Std.I8) (block : Std.Usize)
    (base i1 take : Std.Usize) (absmax : Std.U64) (s : Std.U64)
    (scaleOf : Nat → Std.I64)
    (hblock : block.val < blocks.val)
    (hbase : block * chart.ACT_QK = ok base)
    (hsub : Slice.len x - base = ok i1)
    (hmin : core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take)
    (habsmax : chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax)
    (hscale : kern.pow2.act_scale absmax = ok (some s))
    (hsof : scaleOf block.val = ⟨s.bv⟩)
    (hsle : s.val ≤ 2 ^ 63 - 1)
    (hbasev : base.val = block.val * 32)
    (htakev : take.val = ffn5Take (Slice.len x).val (block.val * 32))
    (hsle2 : (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)
    (hshiftB : ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64)
    (hmaskB : ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)
    (hpush : scales.val.length < Usize.max)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes.val.length + take.val < Usize.max)
    (hadd : block.val + 1 ≤ Usize.max) :
    ∃ scales1 : alloc.vec.Vec Std.I64, ∃ codes1 : alloc.vec.Vec Std.I8,
      ∃ ok2 : Bool, ∃ block1 : Std.Usize,
      chart.quant_act_loop0.body x blocks scales codes block true
        = ok (ControlFlow.cont (scales1, codes1, block1, ok2)) ∧
      block1.val = block.val + 1 ∧
      (scales1.val, codes1.val, ok2)
        = ffn9BlockTurnOf x scaleOf block.val scales.val codes.val := by
  obtain ⟨scales1, hpush1, hpush1v⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec scales ⟨s.bv⟩ hpush)
  have hs2 : core.num.I64.unsigned_abs ⟨s.bv⟩ = ok (ffn9Uabs ⟨s.bv⟩) := rfl
  have hpow2 : core.num.U64.is_power_of_two (ffn9Uabs ⟨s.bv⟩) = ok (ffn9Pow2Of ⟨s.bv⟩) := rfl
  have hshift : core.num.U64.trailing_zeros (ffn9Uabs ⟨s.bv⟩) = ok (ffn9ShiftOf ⟨s.bv⟩) := rfl
  have hmask_eq : core.num.U64.wrapping_sub (ffn9Uabs ⟨s.bv⟩) 1#u64 = ffn9MaskOf ⟨s.bv⟩ := rfl
  have hl0 : (0#usize : Std.Usize).val ≤ take.val := by
    rw [show (0#usize : Std.Usize).val = 0 from rfl]; exact Nat.zero_le _
  obtain ⟨codes1, ok2, hlane, hlane1, hlane2⟩ :=
    ffn8_lane_loop_ok_ffn7 x base take (ffn9Uabs ⟨s.bv⟩) (ffn9Pow2Of ⟨s.bv⟩)
      (ffn9ShiftOf ⟨s.bv⟩) (ffn9MaskOf ⟨s.bv⟩) codes 0#usize hbound hcap hl0 hshiftB hmaskB hsle2
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
    rw [hmask_eq]
    rw [hlane]; simp only [bind_tc_ok]
    rw [hblock1]; simp only [bind_tc_ok]
    rfl
  refine ⟨scales1, codes1, ok2, block1, hbody, ?_, ?_⟩
  · rw [hblock1v]; rfl
  · rw [ffn9BlockTurnOf_val]
    simp only [ffn9CodeOf]
    rw [hsof, hpush1v, hlane1, hlane2, hbasev, htakev]
    have h0 : (0#usize : Std.Usize).val = 0 := rfl
    simp only [h0, Nat.sub_zero]

/-! ## §3 · The block loop `chart.quant_act_loop0`

The per-block turn is `ffn9BlockTurnOf x scaleOf`, so the recursive reference
`ffn5Blocks (ffn9BlockTurnOf x scaleOf)` names the pushed scales and the value-indexed lane codes by
value. -/

/-- **`chart.quant_act_loop0` denotes the value-indexed block reference.** The extracted block loop
from `block0` with the flag `true` returns `ffn5Blocks (ffn9BlockTurnOf x scaleOf)` when the
per-block sub-values are what `act_scale`, the absmax loop and the counter give, the two `Vec`s have
room over the whole run, and the scale's magnitude satisfies the lane-loop bounds. Composed from
`ffn9_block_turn` and the extracted stopping arms. No `hinner`. -/
theorem word_quant_act_loop0_ok9 (x : Slice Std.I64) (blocks : Std.Usize)
    (scales0 : alloc.vec.Vec Std.I64) (codes0 : alloc.vec.Vec Std.I8) (block0 : Std.Usize)
    (scaleOf : Nat → Std.I64)
    (hle : block0.val ≤ blocks.val)
    (hscap0 : scales0.val.length + blocks.val < Usize.max)
    (hccap0 : codes0.val.length + 32 * blocks.val < Usize.max)
    (hscale9 : ∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len x - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧
          s.val ≤ 2 ^ 63 - 1 ∧
          scaleOf block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          base.val = block.val * 32 ∧
          take.val = ffn5Take (Slice.len x).val (block.val * 32) ∧
          base.val + take.val ≤ x.val.length ∧
          (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)) :
    chart.quant_act_loop0 x blocks scales0 codes0 block0 true ⦃ out =>
      out.1.val = (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).1 ∧
      out.2.1.val = (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.1 ∧
      out.2.2 = (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.2 ⦄ := by
  unfold chart.quant_act_loop0
  apply loop.spec_decr_nat
    (measure := fun st : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Std.Usize × Bool =>
      blocks.val - st.2.2.1.val)
    (inv := fun st : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Std.Usize × Bool =>
      block0.val ≤ st.2.2.1.val ∧ st.2.2.1.val ≤ blocks.val ∧
      (st.1.val, st.2.1.val, st.2.2.2)
        = ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
            (st.2.2.1.val - block0.val) ∧
      st.1.val.length ≤ scales0.val.length + (st.2.2.1.val - block0.val) ∧
      st.2.1.val.length ≤ codes0.val.length + 32 * (st.2.2.1.val - block0.val))
    (post := fun out : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I8) × Bool =>
      out.1.val = (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).1 ∧
      out.2.1.val = (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.1 ∧
      out.2.2 = (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
        (blocks.val - block0.val)).2.2)
  · rintro ⟨scales, codes, block, ok1⟩ ⟨hge0, hleB, hval, hls, hlc⟩
    dsimp only at hge0 hleB hval hls hlc ⊢
    by_cases hlt : block.val < blocks.val
    · cases ok1
      · -- flag cleared: the body stops
        rw [ffn5_block_body_stop x blocks scales codes block hlt, WP.spec_ok]
        dsimp only
        have hparts : ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
            (block.val - block0.val) = (scales.val, codes.val, false) := hval.symm
        have hstab := ffn5Blocks_false (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val
          block0.val (block.val - block0.val) (by rw [hparts]) (blocks.val - block.val)
        have hidx : blocks.val - block0.val
            = (block.val - block0.val) + (blocks.val - block.val) := by omega
        refine ⟨?_, ?_, ?_⟩
        · rw [hidx, hstab, hparts]
        · rw [hidx, hstab, hparts]
        · rw [hidx, hstab, hparts]
      · -- flag set: the block turn
        obtain ⟨base, i1, take, absmax, s, hbase, hsub, hmin, habsmax, hscalev, hsle, hsof, hadd,
          hbasev, htakev, hbound, hsle2, hshiftB, hmaskB⟩ := hscale9 block hlt
        have hsc : scales.val.length < Usize.max := by omega
        have htle : take.val ≤ 32 := by rw [htakev]; exact Nat.min_le_left _ _
        have hcc : codes.val.length + take.val < Usize.max := by omega
        obtain ⟨scales1, codes1, ok2, block1, hb, hb1, htov⟩ :=
          ffn9_block_turn x blocks scales codes block base i1 take absmax s scaleOf hlt hbase hsub
            hmin habsmax hscalev hsof hsle hbasev htakev hsle2 hshiftB hmaskB hsc hbound hcc hadd
        rw [hb, WP.spec_ok]
        dsimp only
        have hflag : (ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
            (block.val - block0.val)).2.2 = true := by rw [← hval]
        have hparts : ffn5Blocks (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
            (block.val - block0.val) = (scales.val, codes.val, true) := hval.symm
        have hsv : scales1.val
            = (ffn9BlockTurnOf x scaleOf block.val scales.val codes.val).1 :=
          congrArg Prod.fst htov
        have hcv : codes1.val
            = (ffn9BlockTurnOf x scaleOf block.val scales.val codes.val).2.1 :=
          congrArg (fun q => q.2.1) htov
        refine ⟨⟨?_, ?_, ?_, ?_, ?_⟩, ?_⟩
        · rw [hb1]; omega
        · rw [hb1]; omega
        · rw [htov, hb1, show block.val + 1 - block0.val
              = (block.val - block0.val) + 1 by omega,
            ffn5Blocks_cont (ffn9BlockTurnOf x scaleOf) scales0.val codes0.val block0.val
              (block.val - block0.val) hflag]
          rw [hparts, show block0.val + (block.val - block0.val) = block.val by omega]
        · rw [hsv, ffn9BlockTurnOf_val, hb1, show block.val + 1 - block0.val
              = (block.val - block0.val) + 1 by omega, List.length_append]
          simp only [List.length_cons, List.length_nil]
          omega
        · have h1 : codes1.val.length = codes.val.length
              + (ffn8LaneStep (ffn9CodeOf (scaleOf block.val)) x (block.val * 32) 0
                  take.val).1.length := by
            rw [hcv, ffn9BlockTurnOf_val, List.length_append, ← htakev]
          have h2 : (ffn8LaneStep (ffn9CodeOf (scaleOf block.val)) x (block.val * 32) 0
              take.val).1.length ≤ take.val := by
            have hlane := ffn8LaneStep_fst_len (ffn9CodeOf (scaleOf block.val)) x (block.val * 32) 0
              (ffn5Take (Slice.len x).val (block.val * 32))
            rw [← htakev] at hlane; exact hlane
          have h3 : block1.val - block0.val = (block.val - block0.val) + 1 := by rw [hb1]; omega
          rw [h1, h3]
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

/-! ## §4 · The driver `chart.quant_act`

The driver rounds the code count up to blocks, runs the block loop from a fresh `with_capacity`
state, and returns the `QuantAct` only when the block flag and the dequant block flag both stay set.
`ffn9QuantActRef` is the value-indexed block reference gated by `ffn6DequantBlocks`. -/

/-- The `chart.quant_act` reference over the block count `nb`: the value-indexed block reference's
final state, gated by the dequant block flag over the pushed code count, `none` when either flag
clears. Temporal `O(nb)` turns plus the turns' lane costs; spatial `O(codes)`; the sub-calls are
`ffn5Blocks` and `ffn6DequantBlocks`. -/
def ffn9QuantActRef (x : Slice Std.I64) (nb : Nat) (scaleOf : Nat → Std.I64) :
    Option (List Std.I64 × List Std.I8) :=
  let st := ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 nb
  if st.2.2 then
    if ffn6DequantBlocks st.1 st.2.1 ((st.2.1.length + 31) / 32) 0 ((st.2.1.length + 31) / 32)
    then some (st.1, st.2.1) else none
  else none

/-- **`chart.quant_act` denotes the value-indexed reference.** With the block count from the
extracted `div_ceil`, the per-block sub-values from `hscale9`, and the code capacity carrying `32`
words of headroom over the block count, the extracted driver returns the `QuantAct` of the loop's
pushed scales and codes exactly when `ffn9QuantActRef` is `some` of them. Composed from
`word_quant_act_loop0_ok9` and `word_dequantizes_ok`. No `hinner`. -/
theorem word_quant_act_ok9 (x : Slice Std.I64) (blocks : Std.Usize) (scaleOf : Nat → Std.I64)
    (hdiv : core.num.Usize.div_ceil (Slice.len x) chart.ACT_QK = ok blocks)
    (hcap : 32 * blocks.val + 32 ≤ Usize.max)
    (hscale9 : ∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len x - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧
          s.val ≤ 2 ^ 63 - 1 ∧
          scaleOf block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          base.val = block.val * 32 ∧
          take.val = ffn5Take (Slice.len x).val (block.val * 32) ∧
          base.val + take.val ≤ x.val.length ∧
          (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)) :
    ∃ scales1 : alloc.vec.Vec Std.I64, ∃ codes1 : alloc.vec.Vec Std.I8,
      (∀ p : List Std.I64 × List Std.I8,
        ffn9QuantActRef x blocks.val scaleOf = some p → (scales1.val, codes1.val) = p) ∧
      chart.quant_act x =
        ok (Option.map (fun _ => ({ scales := scales1, codes := codes1 } : chart.QuantAct))
          (ffn9QuantActRef x blocks.val scaleOf)) := by
  have hscz : (alloc.vec.Vec.with_capacity Std.I64 blocks).val = ([] : List Std.I64) := rfl
  have hccz : (alloc.vec.Vec.with_capacity Std.I8 (Slice.len x)).val = ([] : List Std.I8) := rfl
  have hscap0 : (alloc.vec.Vec.with_capacity Std.I64 blocks).val.length + blocks.val
      < Usize.max := by
    rw [hscz]; simp only [List.length_nil, Nat.zero_add]; omega
  have hccap0 : (alloc.vec.Vec.with_capacity Std.I8 (Slice.len x)).val.length
      + 32 * blocks.val < Usize.max := by
    rw [hccz]; simp only [List.length_nil, Nat.zero_add]; omega
  obtain ⟨out, hout, h1, h2, h3⟩ := WP.spec_imp_exists
    (word_quant_act_loop0_ok9 x blocks (alloc.vec.Vec.with_capacity Std.I64 blocks)
      (alloc.vec.Vec.with_capacity Std.I8 (Slice.len x)) 0#usize scaleOf
      (by simp) hscap0 hccap0 hscale9)
  rw [hscz, hccz, show (0#usize : Std.Usize).val = 0 from rfl, Nat.sub_zero] at h1 h2 h3
  obtain ⟨scales1, codes1, ok1⟩ := out
  dsimp only at h1 h2 h3
  have hstlen : (ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val).2.1.length
      ≤ 32 * blocks.val := by
    have h := (ffn5Blocks_len (ffn9BlockTurnOf x scaleOf)
      (ffn9BlockTurnOf_len1 x scaleOf) (ffn9BlockTurnOf_len2 x scaleOf)
      ([] : List Std.I64) ([] : List Std.I8) 0 blocks.val).2
    simpa using h
  have hcapdf : codes1.val.length + 31 ≤ Usize.max := by rw [h2]; omega
  have hdf := word_dequantizes_ok ({ scales := scales1, codes := codes1 } : chart.QuantAct) hcapdf
  refine ⟨scales1, codes1, ?_, ?_⟩
  · intro p hp
    rw [h1, h2]
    unfold ffn9QuantActRef at hp
    dsimp only at hp
    by_cases h22 : (ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val).2.2 = true
    · rw [if_pos h22] at hp
      by_cases hF : (ffn6DequantBlocks (ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0
            blocks.val).1 (ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val).2.1
          (((ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val).2.1.length + 31) / 32)
          0 (((ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val).2.1.length + 31)
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
    show (if (ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val).2.2 = true then
        (do let b ← ({ scales := scales1, codes := codes1 } : chart.QuantAct).dequantizes
            if b = true then ok (some ({ scales := scales1, codes := codes1 } : chart.QuantAct))
            else ok none)
        else ok none)
      = ok (Option.map (fun _ => ({ scales := scales1, codes := codes1 } : chart.QuantAct))
          (ffn9QuantActRef x blocks.val scaleOf))
    unfold ffn9QuantActRef
    set st : List Std.I64 × List Std.I8 × Bool :=
      ffn5Blocks (ffn9BlockTurnOf x scaleOf) [] [] 0 blocks.val with hstdef
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

/-- **The refusal.** When the reference is `none`, `chart.quant_act` returns `none`: the `Ok` case
is exactly the reference's. No `hinner`. -/
theorem word_quant_act_refuses9 (x : Slice Std.I64) (blocks : Std.Usize) (scaleOf : Nat → Std.I64)
    (hdiv : core.num.Usize.div_ceil (Slice.len x) chart.ACT_QK = ok blocks)
    (hcap : 32 * blocks.val + 32 ≤ Usize.max)
    (hscale9 : ∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len x - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧
          s.val ≤ 2 ^ 63 - 1 ∧
          scaleOf block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          base.val = block.val * 32 ∧
          take.val = ffn5Take (Slice.len x).val (block.val * 32) ∧
          base.val + take.val ≤ x.val.length ∧
          (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1))
    (h : ffn9QuantActRef x blocks.val scaleOf = none) :
    chart.quant_act x = ok none := by
  obtain ⟨_, _, hm⟩ := word_quant_act_ok9 x blocks scaleOf hdiv hcap hscale9
  rw [h] at hm
  simpa using hm

/-! ## §5 · Non-vacuity

`hscale9` is satisfiable at concrete inputs, so the three headlines carry content — the direct
contrast with `Ffn6`, whose `hinner` no input satisfies.

* On `x = Slice.from [0#i64]` with one block, block `0` has `base = 0`, `take = 1`, the absmax `0`
  and the scale `0` (the zero-scale short circuit). `hle`, `hscap0`, `hccap0`, `hdiv`, `hcap` and
  the whole `hscale9` bundle hold, and the scale's lane-loop bounds are vacuous
  (`ffn9Pow2Of ⟨(0#u64).bv⟩ = false`).
* On `x = Slice.from [9223372036854775806#i64]` (`2^63 - 2`) with the `2^57` scale the single lane
  code is `64`, whose dequant product `2^57·64 = 2^63` leaves `i64::MAX`, so `ffn9QuantActRef` is
  `none` and the refusal headline is non-vacuous. -/

/-- `Usize.max` is at least the 32-bit maximum, so the concrete indices and capacities below are
valid on either platform. -/
private lemma ffn9_usize_max_ge : 4294967295 ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h <;> rw [h]
  · norm_num [U32.max_eq]
  · norm_num [U64.max_eq]

/-- The one-word slice `[0]`. -/
private def ffn9ZeroSlice : Slice Std.I64 :=
  Slice.from [0#i64] (by simpa using ffn3_one_le_usize_max)

/-- `ffn9ZeroSlice`'s value is the list `[0]`. -/
private lemma ffn9ZeroSlice_val : ffn9ZeroSlice.val = [0#i64] := by
  unfold ffn9ZeroSlice; rw [Slice.from_val]

/-- `ffn9ZeroSlice` has length `1`. -/
private lemma ffn9ZeroSlice_len : (Slice.len ffn9ZeroSlice).val = 1 := by
  simp [ffn9ZeroSlice]

/-- **The `hscale9` bundle at the zero block.** On `x = ffn9ZeroSlice`, one block, block `0`
has `base = 0`, `i1 = 1`, `take = 1`, absmax `0` and scale `0`; every conjunct of the bundle holds. -/
theorem ffn9_scale9_zero :
    ∀ block : Std.Usize, block.val < (1#usize).val →
      ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
        block * chart.ACT_QK = ok base ∧
        Slice.len ffn9ZeroSlice - base = ok i1 ∧
        core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
        chart.quant_act_loop0_loop0 ffn9ZeroSlice base take 0#u64 0#usize = ok absmax ∧
        kern.pow2.act_scale absmax = ok (some s) ∧
        s.val ≤ 2 ^ 63 - 1 ∧
        (fun _ => 0#i64) block.val = ⟨s.bv⟩ ∧
        block.val + 1 ≤ Usize.max ∧
        base.val = block.val * 32 ∧
        take.val = ffn5Take (Slice.len ffn9ZeroSlice).val (block.val * 32) ∧
        base.val + take.val ≤ ffn9ZeroSlice.val.length ∧
        (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
        (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
        (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1) := by
  intro block hb
  have hb0 : block.val = 0 := by
    have h1 : (1#usize).val = 1 := rfl
    omega
  have hlen : (Slice.len ffn9ZeroSlice).val = 1 := ffn9ZeroSlice_len
  have hmul : block.val * chart.ACT_QK.val ≤ Usize.max := by
    rw [hb0, Nat.zero_mul]; exact Nat.zero_le _
  obtain ⟨base, hbase, hbasev⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := chart.ACT_QK) hmul)
  have hbase32 : base.val = block.val * 32 := by rw [hbasev, ffn5_act_qk_val]
  have hbase0 : base.val = 0 := by rw [hbase32, hb0]
  have hle1 : base.val ≤ (Slice.len ffn9ZeroSlice).val := by
    rw [hbase0]; exact Nat.zero_le _
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := Slice.len ffn9ZeroSlice) (y := base) hle1)
  obtain ⟨take, htake, htakev⟩ := ordUsize_min_spec chart.ACT_QK i1
  have hi1val : i1.val = 1 := by rw [hi1v.1, hbase0, hlen]
  have htakeval : take.val = 1 := by rw [htakev, ffn5_act_qk_val, hi1val]; norm_num
  have htakev' : take.val = ffn5Take (Slice.len ffn9ZeroSlice).val (block.val * 32) := by
    rw [htakeval, hb0, hlen, ffn5Take]; norm_num
  have hbound : base.val + take.val ≤ ffn9ZeroSlice.val.length := by
    have hmin : take.val ≤ (Slice.len ffn9ZeroSlice).val - base.val := by
      rw [htakev', ffn5Take, hbase32]; exact Nat.min_le_right 32 _
    have h3 : (Slice.len ffn9ZeroSlice).val = ffn9ZeroSlice.val.length :=
      Slice.len_val ffn9ZeroSlice
    omega
  obtain ⟨absmax, habsmax, habsmaxv⟩ :=
    word_quant_absmax_ok ffn9ZeroSlice base take ffn5_unsigned_abs_ok hbound
  have habsmax0 : absmax.val = 0 := by
    rw [habsmaxv, ffn9ZeroSlice_val, hbase0, htakeval]; decide
  have habsmax00 : absmax = 0#u64 := UScalar.eq_of_val_eq (by rw [habsmax0]; rfl)
  have hbig1 : block.val + 1 ≤ Usize.max := by
    rw [hb0]; have := ffn9_usize_max_ge; omega
  refine ⟨base, i1, take, absmax, 0#u64, hbase, hi1, htake, habsmax, ?_,
    by norm_num, ?_, hbig1, hbase32, htakev', ?_, ?_, ?_, ?_⟩
  · rw [habsmax00]; exact word_act_scale_zero
  · rw [hb0]; rfl
  · rw [hbase0, htakeval, ffn9ZeroSlice_val]; decide
  · decide
  · decide
  · decide

/-- **The loop headline's hypotheses hold jointly** at `x = ffn9ZeroSlice`, one block: `hle`,
`hscap0`, `hccap0` and the whole `hscale9` bundle. -/
example :
    (0#usize).val ≤ (1#usize).val ∧
    ([] : List Std.I64).length + (1#usize).val < Usize.max ∧
    ([] : List Std.I8).length + 32 * (1#usize).val < Usize.max ∧
    (∀ block : Std.Usize, block.val < (1#usize).val →
      ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
        block * chart.ACT_QK = ok base ∧
        Slice.len ffn9ZeroSlice - base = ok i1 ∧
        core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
        chart.quant_act_loop0_loop0 ffn9ZeroSlice base take 0#u64 0#usize = ok absmax ∧
        kern.pow2.act_scale absmax = ok (some s) ∧
        s.val ≤ 2 ^ 63 - 1 ∧
        (fun _ => 0#i64) block.val = ⟨s.bv⟩ ∧
        block.val + 1 ≤ Usize.max ∧
        base.val = block.val * 32 ∧
        take.val = ffn5Take (Slice.len ffn9ZeroSlice).val (block.val * 32) ∧
        base.val + take.val ≤ ffn9ZeroSlice.val.length ∧
        (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
        (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
        (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)) :=
  ⟨by decide,
   by rw [List.length_nil, show (1#usize).val = 1 from rfl, Nat.zero_add];
      have := ffn9_usize_max_ge; omega,
   by rw [List.length_nil, show (1#usize).val = 1 from rfl];
      have := ffn9_usize_max_ge; omega,
   ffn9_scale9_zero⟩

/-- **The driver headline's hypotheses hold jointly** at `x = ffn9ZeroSlice`, one block:
`hdiv`, `hcap` and the whole `hscale9` bundle. -/
example :
    ∃ blocks : Std.Usize,
      core.num.Usize.div_ceil (Slice.len ffn9ZeroSlice) chart.ACT_QK = ok blocks ∧
      32 * blocks.val + 32 ≤ Usize.max ∧
      (∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len ffn9ZeroSlice - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 ffn9ZeroSlice base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧
          s.val ≤ 2 ^ 63 - 1 ∧
          (fun _ => 0#i64) block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          base.val = block.val * 32 ∧
          take.val = ffn5Take (Slice.len ffn9ZeroSlice).val (block.val * 32) ∧
          base.val + take.val ≤ ffn9ZeroSlice.val.length ∧
          (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)) := by
  obtain ⟨blocks, hdiv, hblockv⟩ := ffn6_div_ceil_32_ok (Slice.len ffn9ZeroSlice)
  have hlen : (Slice.len ffn9ZeroSlice).val = 1 := ffn9ZeroSlice_len
  have hb1 : blocks.val = 1 := by rw [hblockv, hlen]
  have hbeq : blocks = 1#usize := UScalar.eq_of_val_eq (by rw [hb1]; rfl)
  refine ⟨blocks, hdiv, ?_, ?_⟩
  · rw [hb1]; have := ffn9_usize_max_ge; omega
  · intro block hb
    exact ffn9_scale9_zero block (by rw [← hbeq]; exact hb)

/-! ### The refusal witness: `x = [2^63 - 2]` at the `2^57` scale

The single lane code is `64` (the scale is `⌈(2^63-2)/127⌉` rounded up to a power of two), and the
dequant product `2^57·64 = 2^63` leaves `i64::MAX`, so the dequant block flag clears and
`ffn9QuantActRef` is `none` while every hypothesis of `word_quant_act_refuses9` holds. -/

/-- The one-word slice `[2^63 - 2]`, the largest runtime magnitude. -/
private def ffn9BigSlice : Slice Std.I64 :=
  Slice.from [9223372036854775806#i64] (by simpa using ffn3_one_le_usize_max)

/-- `ffn9BigSlice`'s value is the list `[2^63 - 2]`. -/
private lemma ffn9BigSlice_val : ffn9BigSlice.val = [9223372036854775806#i64] := by
  unfold ffn9BigSlice; rw [Slice.from_val]

/-- `ffn9BigSlice` has length `1`. -/
private lemma ffn9BigSlice_len : (Slice.len ffn9BigSlice).val = 1 := by
  simp [ffn9BigSlice]

/-- The scale at `absmax = 2^63 - 2` is `2^57`. -/
private lemma ffn9_act_scale_big :
    kern.pow2.act_scale 9223372036854775806#u64 = ok (some 144115188075855872#u64) := by
  obtain ⟨out, hout, hcases⟩ := word_act_scale_ok 9223372036854775806#u64
  rcases hcases with ⟨hz, _⟩ | ⟨_, s, k, houts, hk62, hsv, hleast⟩
  · exact absurd hz (by decide)
  · have hneed : ffn3Need (9223372036854775806#u64).val = 72624976668147841 := by decide
    rw [hneed] at hleast
    have hk57 : k = 57 := by
      rcases hleast with ⟨hle, hz | hlt⟩
      · subst hz; norm_num at hle
      · have hkge : 57 ≤ k := by
          by_contra h
          have h1 : (2 : Nat) ^ k ≤ 2 ^ 56 := Nat.pow_le_pow_right (by norm_num) (by omega)
          have h2 : (2 : Nat) ^ 56 = 72057594037927936 := by norm_num
          omega
        have hkle : k ≤ 57 := by
          by_contra h
          have h1 : (2 : Nat) ^ 57 ≤ 2 ^ (k - 1) :=
            Nat.pow_le_pow_right (by norm_num) (by omega)
          have h2 : (2 : Nat) ^ 57 = 144115188075855872 := by norm_num
          omega
        omega
    have hsv57 : s.val = 144115188075855872 := by rw [hsv, hk57]; norm_num
    have hs_eq : s = 144115188075855872#u64 := UScalar.eq_of_val_eq (by rw [hsv57]; rfl)
    rw [hout, houts, hs_eq]

/-- The lane code at the `2^57` scale and value `2^63 - 2` is `(true, 64)`. -/
private lemma ffn9_lane_code_big :
    ffn7Code 144115188075855872#u64 true 57#u32 144115188075855871#u64
      9223372036854775806#i64 = ok (true, 64#i8) := by
  obtain ⟨b, code, q, r, hqr, hcode, hbc⟩ :=
    ffn7_lane_code_eq 144115188075855872#u64 true 57#u32 144115188075855871#u64
      9223372036854775806#i64 (by decide) (by decide) (by decide)
  obtain ⟨q', r', hqr', hq'v, hr'v⟩ :=
    word_act_quot_rem_pow2 9223372036854775806#u64 144115188075855872#u64 57#u32
      144115188075855871#u64 (by decide)
  have hcast : (⟨BitVec.ofNat 64 (9223372036854775806#i64).val.natAbs⟩ : Std.U64)
      = 9223372036854775806#u64 := rfl
  rw [hcast] at hqr
  have hpair : (q, r) = (q', r') := by
    simpa only [Result.ok.injEq] using hqr.symm.trans hqr'
  have hqeq : q = q' := congrArg Prod.fst hpair
  have hreq : r = r' := congrArg Prod.snd hpair
  subst hqeq
  subst hreq
  have hq63 : q.val = 63 := by rw [hq'v]; decide
  have hr2 : r.val = 144115188075855870 := by rw [hr'v]; decide
  have hcodeval : (b, code.val) = (true, (64 : ℤ)) := by
    rw [hbc, hq63, hr2]
    decide
  have hb_true : b = true := congrArg Prod.fst hcodeval
  have hcode64 : code.val = 64 := by
    have h := congrArg Prod.snd hcodeval
    omega
  have hcode_eq : code = 64#i8 := IScalar.eq_of_val_eq (by rw [hcode64]; rfl)
  rw [hcode, hb_true, hcode_eq]

/-- The closed-form block code at the `2^57` scale and value `2^63 - 2` is `(true, 64)`. -/
private lemma ffn9_codeOf_big :
    ffn9CodeOf 144115188075855872#i64 0 9223372036854775806#i64 = (true, 64#i8) := by
  show ffn8CodeOf (ffn9Uabs 144115188075855872#i64) (ffn9Pow2Of 144115188075855872#i64)
      (ffn9ShiftOf 144115188075855872#i64) (ffn9MaskOf 144115188075855872#i64) 0
      9223372036854775806#i64 = (true, 64#i8)
  rw [show ffn9Uabs 144115188075855872#i64 = 144115188075855872#u64 from rfl,
      show ffn9Pow2Of 144115188075855872#i64 = true from rfl,
      show ffn9ShiftOf 144115188075855872#i64 = 57#u32 from rfl,
      show ffn9MaskOf 144115188075855872#i64 = 144115188075855871#u64 from rfl]
  unfold ffn8CodeOf
  rw [ffn9_lane_code_big]
  simp only [Result.match.ok]

/-- The lane step at the big input pushes the single code `64`. -/
private lemma ffn9_lane_step_big :
    ffn8LaneStep (ffn9CodeOf 144115188075855872#i64) ffn9BigSlice 0 0 1 = ([64#i8], true) := by
  have hshow : (0 + (0 + 0) : Nat) = 0 := rfl
  have hcode : ffn9CodeOf 144115188075855872#i64 0 (ffn9BigSlice.val.getD 0 0#i64)
      = (true, 64#i8) := by
    rw [show ffn9BigSlice.val.getD 0 0#i64 = 9223372036854775806#i64 from by
      rw [ffn9BigSlice_val]; rfl]
    exact ffn9_codeOf_big
  rw [show (1 : Nat) = 0 + 1 from rfl, ffn8LaneStep_succ, ffn8LaneStep_zero]
  rw [hshow, hcode]
  rfl

/-- The block turn at the big input pushes the scale `2^57` and the code `64`. -/
private lemma ffn9_turn_big :
    ffn9BlockTurnOf ffn9BigSlice (fun _ => 144115188075855872#i64) 0 [] [] =
      ([144115188075855872#i64], [64#i8], true) := by
  rw [ffn9BlockTurnOf_val]
  rw [show ffn5Take (Slice.len ffn9BigSlice).val (0 * 32) = 1 from by
    rw [ffn5Take, ffn9BigSlice_len]; norm_num]
  rw [show (0 : Nat) * 32 = 0 from rfl]
  rw [ffn9_lane_step_big]
  rfl

/-- The single-block run at the big input. -/
private lemma ffn9_blocks_eq :
    ffn5Blocks (ffn9BlockTurnOf ffn9BigSlice (fun _ => 144115188075855872#i64)) [] [] 0 1
      = ([144115188075855872#i64], [64#i8], true) := by
  rw [show (1 : Nat) = 0 + 1 from rfl, ffn5Blocks_succ, ffn5Blocks_zero]
  dsimp only
  rw [show (0 + 0 : Nat) = 0 from rfl, if_pos rfl]
  exact ffn9_turn_big

/-- **The reference is `none` at the big input**: the code `64` at the `2^57` scale overflows the
dequant `checked_mul`. -/
private lemma ffn9_ref_big :
    ffn9QuantActRef ffn9BigSlice 1 (fun _ => 144115188075855872#i64) = none := by
  have hdeq : ffn6DequantBlocks [144115188075855872#i64] [64#i8]
      (([64#i8].length + 31) / 32) 0 (([64#i8].length + 31) / 32) = false := by decide
  unfold ffn9QuantActRef
  rw [ffn9_blocks_eq]
  dsimp only
  rw [if_pos rfl]
  rw [hdeq]
  rfl

/-- **The `hscale9` bundle at the big block.** On `x = ffn9BigSlice`, one block, block `0` has
`base = 0`, `i1 = 1`, `take = 1`, absmax `2^63 - 2` and scale `2^57`; every conjunct holds. -/
theorem ffn9_scale9_big :
    ∀ block : Std.Usize, block.val < (1#usize).val →
      ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
        block * chart.ACT_QK = ok base ∧
        Slice.len ffn9BigSlice - base = ok i1 ∧
        core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
        chart.quant_act_loop0_loop0 ffn9BigSlice base take 0#u64 0#usize = ok absmax ∧
        kern.pow2.act_scale absmax = ok (some s) ∧
        s.val ≤ 2 ^ 63 - 1 ∧
        (fun _ => 144115188075855872#i64) block.val = ⟨s.bv⟩ ∧
        block.val + 1 ≤ Usize.max ∧
        base.val = block.val * 32 ∧
        take.val = ffn5Take (Slice.len ffn9BigSlice).val (block.val * 32) ∧
        base.val + take.val ≤ ffn9BigSlice.val.length ∧
        (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
        (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
        (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1) := by
  intro block hb
  have hb0 : block.val = 0 := by
    have h1 : (1#usize).val = 1 := rfl
    omega
  have hlen : (Slice.len ffn9BigSlice).val = 1 := ffn9BigSlice_len
  have hmul : block.val * chart.ACT_QK.val ≤ Usize.max := by
    rw [hb0, Nat.zero_mul]; exact Nat.zero_le _
  obtain ⟨base, hbase, hbasev⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := chart.ACT_QK) hmul)
  have hbase32 : base.val = block.val * 32 := by rw [hbasev, ffn5_act_qk_val]
  have hbase0 : base.val = 0 := by rw [hbase32, hb0]
  have hle1 : base.val ≤ (Slice.len ffn9BigSlice).val := by rw [hbase0]; exact Nat.zero_le _
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := Slice.len ffn9BigSlice) (y := base) hle1)
  obtain ⟨take, htake, htakev⟩ := ordUsize_min_spec chart.ACT_QK i1
  have hi1val : i1.val = 1 := by rw [hi1v.1, hbase0, hlen]
  have htakeval : take.val = 1 := by rw [htakev, ffn5_act_qk_val, hi1val]; norm_num
  have htakev' : take.val = ffn5Take (Slice.len ffn9BigSlice).val (block.val * 32) := by
    rw [htakeval, hb0, hlen, ffn5Take]; norm_num
  have hbound : base.val + take.val ≤ ffn9BigSlice.val.length := by
    have hmin : take.val ≤ (Slice.len ffn9BigSlice).val - base.val := by
      rw [htakev', ffn5Take, hbase32]; exact Nat.min_le_right 32 _
    have h3 : (Slice.len ffn9BigSlice).val = ffn9BigSlice.val.length :=
      Slice.len_val ffn9BigSlice
    omega
  obtain ⟨absmax, habsmax, habsmaxv⟩ :=
    word_quant_absmax_ok ffn9BigSlice base take ffn5_unsigned_abs_ok hbound
  have habsmax0 : absmax.val = 9223372036854775806 := by
    rw [habsmaxv, ffn9BigSlice_val, hbase0, htakeval]; decide
  have habsmax_eq : absmax = 9223372036854775806#u64 :=
    UScalar.eq_of_val_eq (by rw [habsmax0]; rfl)
  have hbig1 : block.val + 1 ≤ Usize.max := by
    rw [hb0]; have := ffn9_usize_max_ge; omega
  refine ⟨base, i1, take, absmax, 144115188075855872#u64, hbase, hi1, htake, habsmax, ?_,
    ?_, ?_, hbig1, hbase32, htakev', ?_, ?_, ?_, ?_⟩
  · rw [habsmax_eq]; exact ffn9_act_scale_big
  · norm_num
  · rw [hb0]; rfl
  · rw [hbase0, htakeval, ffn9BigSlice_val]; decide
  · decide
  · intro _; decide
  · intro _; decide

/-- **The refusal headline's hypotheses hold jointly, and the reference is `none`** at
`x = ffn9BigSlice`, one block, the `2^57` scale: `hdiv`, `hcap`, the whole `hscale9` bundle and
`ffn9QuantActRef … = none`. -/
example :
    ∃ blocks : Std.Usize,
      core.num.Usize.div_ceil (Slice.len ffn9BigSlice) chart.ACT_QK = ok blocks ∧
      32 * blocks.val + 32 ≤ Usize.max ∧
      (∀ block : Std.Usize, block.val < blocks.val →
        ∃ base i1 take : Std.Usize, ∃ absmax s : Std.U64,
          block * chart.ACT_QK = ok base ∧
          Slice.len ffn9BigSlice - base = ok i1 ∧
          core.cmp.Ord.min.trait_default core.cmp.OrdUsize chart.ACT_QK i1 = ok take ∧
          chart.quant_act_loop0_loop0 ffn9BigSlice base take 0#u64 0#usize = ok absmax ∧
          kern.pow2.act_scale absmax = ok (some s) ∧
          s.val ≤ 2 ^ 63 - 1 ∧
          (fun _ => 144115188075855872#i64) block.val = ⟨s.bv⟩ ∧
          block.val + 1 ≤ Usize.max ∧
          base.val = block.val * 32 ∧
          take.val = ffn5Take (Slice.len ffn9BigSlice).val (block.val * 32) ∧
          base.val + take.val ≤ ffn9BigSlice.val.length ∧
          (ffn9Uabs ⟨s.bv⟩).val ≤ 2 ^ 63 - 1 ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9ShiftOf ⟨s.bv⟩).val < 64) ∧
          (ffn9Pow2Of ⟨s.bv⟩ = true → (ffn9MaskOf ⟨s.bv⟩).val ≤ 2 ^ 63 - 1)) ∧
      ffn9QuantActRef ffn9BigSlice blocks.val (fun _ => 144115188075855872#i64) = none := by
  obtain ⟨blocks, hdiv, hblockv⟩ := ffn6_div_ceil_32_ok (Slice.len ffn9BigSlice)
  have hlen : (Slice.len ffn9BigSlice).val = 1 := ffn9BigSlice_len
  have hb1 : blocks.val = 1 := by rw [hblockv, hlen]
  have hbeq : blocks = 1#usize := UScalar.eq_of_val_eq (by rw [hb1]; rfl)
  refine ⟨blocks, hdiv, ?_, ?_, ?_⟩
  · rw [hb1]; have := ffn9_usize_max_ge; omega
  · intro block hb
    exact ffn9_scale9_big block (by rw [← hbeq]; exact hb)
  · rw [hb1]; exact ffn9_ref_big

end apeiron_word
