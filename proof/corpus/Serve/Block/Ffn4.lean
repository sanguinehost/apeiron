/-
  `Serve/Ffn4.lean` — `chart.quant_act_loop0_loop1` (the one-lane q8 code loop of `chart.quant_act`,
  `src/engine/chart/mod.rs:287`): its turn arm, closed by a reduction that does not fight the
  pair-bind of the extracted body, and its lane-loop spec over the reference `ffn4LaneStep`.

  `Serve/Ffn3.lean` §3 proved the two *stopping* arms (`ffn3_lane_body_done_lane`,
  `ffn3_lane_body_done_ok`) and left the *turn* arm open: after
  `unfold chart.quant_act_loop0_loop1.body` the inner code computation is bound with a tuple
  pattern (`let (ok2, code) ← ⋯`), and `rw`/`simp only [bind_tc_ok]` on that tuple bind report
  "not type-correct under the `instances` transparency level" because the pair destructuring is
  a `Prod` recursor whose motive is not reducible at that transparency. The fix is not to fight
  it: rewrite the two `if`s first (`if_pos` for `lane < take`, `if_true` for `s = 0`), use
  `bind_tc_ok` to bind, and then state the *reduced* continuation with `change` (which checks at
  the default transparency and does reduce the `Prod` let), after which the `Vec::push` and the
  counter increment rewrite with their specs.

  Group coverage:
  * `chart.quant_act_loop0_loop1` — §1 the turn arm at the zero scale (`ffn4_lane_body_turn_zero`)
    and the general turn arm (`ffn4_lane_body_turn`), both giving the pushed code, its value and the
    advanced counter; §2 the loop spec (`ffn4_lane_loop_spec`, `ffn4_lane_loop_ok`) refining the
    whole lane loop to the reference `ffn4LaneStep`, with `ffn4_lane_loop_len` giving the code count
    `take - l0` when every lane's flag holds.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains` are `axiom` in
    `ApeironWord` (`proof/regen.sh`'s opaque list). The general turn arm and the loop spec take the
    inner code computation's *outcome* as the hypothesis `hinner`, exactly as `Serve/Ffn3.lean`
    §1/§3 take `unsigned_abs`'s behaviour; no proof step invokes an axiom.
  * the growable `Vec` growth (`Vec.push`) is handled at the value level through `Vec.push_spec`
    with its `length < Usize.max` precondition; the loop spec carries the matching capacity
    hypothesis `codes0.val.length + take.val < Usize.max`. `chart.quant_act_loop0`,
    `chart.quant_act` and `chart.QuantAct.dequantizes` remain open (the residual of §2).

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch, the growable `Vec` allocation, or the two opaque externals
  above. The statements are about the Aeneas-extracted body over the model `Slice`, `Vec` and
  word scalars.
-/

import ApeironWord
import Serve.Block.Ffn3

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §1 · The turn arm of `chart.quant_act_loop0_loop1`

`src/engine/chart/mod.rs:287`. One lane `lane < take` with the flag set reads `x[base + lane]`,
computes the lane's code from the `chart.act_quot_rem` pair, pushes it and advances the counter.
The reference code a lane contributes is `ffn3LaneCodeRef` (`Serve/Ffn3.lean` §3). -/

/-- **One `chart.quant_act_loop0_loop1` body turn at the zero scale.** With `lane < take` and the
flag set, the body reads the lane, short-circuits the code computation at `s = 0` to the code `0`
with the flag set, pushes it and advances the counter. `proved`. -/
theorem ffn4_lane_body_turn_zero (x : Slice Std.I64) (base take : Std.Usize)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes : alloc.vec.Vec Std.I8)
    (lane : Std.Usize) (hpush : codes.val.length < Usize.max)
    (hbound : base.val + take.val ≤ x.val.length) (hlane : lane.val < take.val) :
    ∃ c1 : alloc.vec.Vec Std.I8, ∃ l' : Std.Usize,
      codes.push 0#i8 = ok c1 ∧ l'.val = lane.val + 1 ∧
      chart.quant_act_loop0_loop1.body x base take 0#u64 pow2 shift mask codes true lane
        = ok (ControlFlow.cont (c1, true, l')) := by
  have hlen := Slice.length_ineq x
  have hsum : base.val + lane.val ≤ Usize.max := by omega
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hsum)
  have hidx : i.val < x.val.length := by rw [hiv]; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hidx)
  have h1v : (1#usize : Std.Usize).val = 1 := rfl
  have hmax1 : lane.val + (1#usize : Std.Usize).val ≤ Usize.max := by omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := lane) (y := 1#usize) hmax1)
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec codes 0#i8 hpush)
  refine ⟨c1, i2, hc1, ?_, ?_⟩
  · rw [hi2v, h1v]
  · unfold chart.quant_act_loop0_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane)]
    rw [hi]; simp only [bind_tc_ok]
    rw [hi1]; simp only [bind_tc_ok]
    rw [if_pos (by trivial)]
    rw [if_true]
    rw [bind_tc_ok]
    change (do let codes1 ← codes.push 0#i8; let lane1 ← lane + 1#usize; ok (cont (codes1, true, lane1))) = ok (cont (c1, true, i2))
    rw [hc1]; simp only [bind_tc_ok]
    rw [hi2]; simp only [bind_tc_ok]

/-- **One `chart.quant_act_loop0_loop1` body turn, general.** With `lane < take` and the flag set,
given that the lane's code computation returns `ok (ok2, code)` — the extracted `if s = 0 then
(ok (true, 0#i8)) else ⋯` with the lane's value `v` — the body pushes `code` and advances to
`lane + 1`. The code computation is the `hinner` hypothesis (its `else` branch consumes the two
opaque externals `unsigned_abs` and `RangeInclusive.contains`); `proved` relative to it. -/
theorem ffn4_lane_body_turn (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes : alloc.vec.Vec Std.I8)
    (lane : Std.Usize) (ok2 : Bool) (code : Std.I8)
    (hpush : codes.val.length < Usize.max)
    (hbound : base.val + take.val ≤ x.val.length) (hlane : lane.val < take.val)
    (hinner : ∀ v : Std.I64,
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
        ok (b, i3)) = ok (ok2, code)) :
    ∃ c1 : alloc.vec.Vec Std.I8, ∃ l' : Std.Usize,
      codes.push code = ok c1 ∧ c1.val = codes.val ++ [code] ∧ l'.val = lane.val + 1 ∧
      chart.quant_act_loop0_loop1.body x base take s pow2 shift mask codes true lane
        = ok (ControlFlow.cont (c1, ok2, l')) := by
  have hlen := Slice.length_ineq x
  have hsum : base.val + lane.val ≤ Usize.max := by omega
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hsum)
  have hidx : i.val < x.val.length := by rw [hiv]; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hidx)
  have h1v : (1#usize : Std.Usize).val = 1 := rfl
  have hmax1 : lane.val + (1#usize : Std.Usize).val ≤ Usize.max := by omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := lane) (y := 1#usize) hmax1)
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec codes code hpush)
  refine ⟨c1, i2, hc1, hc1v, ?_, ?_⟩
  · rw [hi2v, h1v]
  · unfold chart.quant_act_loop0_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane)]
    rw [hi]; simp only [bind_tc_ok]
    rw [hi1]; simp only [bind_tc_ok]
    rw [hinner i1]
    rw [bind_tc_ok]
    change (do let codes1 ← codes.push code; let lane1 ← lane + 1#usize; ok (cont (codes1, ok2, lane1))) = ok (cont (c1, ok2, i2))
    rw [hc1]; simp only [bind_tc_ok]
    rw [hi2]; simp only [bind_tc_ok]

/-- **The zero-scale turn arm carries the Ffn3 per-lane reference code.** At `s = 0` the reference
`ffn3LaneCodeRef v 0 q r` is `(true, 0)` (`Serve/Ffn3.lean`), so the flag `true` and the pushed code
`0#i8` of `ffn4_lane_body_turn_zero` are exactly the reference lane's flag and code. -/
theorem ffn4_lane_body_turn_zero_ref (x : Slice Std.I64) (base take : Std.Usize)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes : alloc.vec.Vec Std.I8)
    (lane : Std.Usize) (v : Std.I64) (q r : Std.U64)
    (hpush : codes.val.length < Usize.max)
    (hbound : base.val + take.val ≤ x.val.length) (hlane : lane.val < take.val) :
    ∃ c1 : alloc.vec.Vec Std.I8, ∃ l' : Std.Usize,
      codes.push 0#i8 = ok c1 ∧ l'.val = lane.val + 1 ∧
      chart.quant_act_loop0_loop1.body x base take 0#u64 pow2 shift mask codes true lane
        = ok (ControlFlow.cont (c1, (ffn3LaneCodeRef (v.val : ℤ) 0 q r).1, l')) ∧
      (0#i8).val = (ffn3LaneCodeRef (v.val : ℤ) 0 q r).2 := by
  obtain ⟨c1, l', hc1, hl', hb⟩ :=
    ffn4_lane_body_turn_zero x base take pow2 shift mask codes lane hpush hbound hlane
  refine ⟨c1, l', hc1, hl', ?_, ?_⟩
  · rw [hb, ffn3LaneCodeRef_zero]
  · rw [ffn3LaneCodeRef_zero]; rfl

/-! ## §2 · The lane loop `chart.quant_act_loop0_loop1`

The extracted loop pushes one code per lane from the starting lane up to `take`, and stops early at
the first lane whose code clears the running flag. The reference `ffn4LaneStep` states the codes
contributed and the final flag as a function of a per-lane code `codeOf : Nat → Bool × Std.I8`, and
`ffn4_lane_loop_spec` refines the extracted loop to it. -/

/-- The codes contributed by the lanes `l0, …, l0 + n - 1` in order, with the running flag: while
the flag holds, lane `l0 + k` contributes `(codeOf (l0 + k)).2`; the first lane whose flag is `false`
stops the run and contributes last. Temporal and spatial cost: one step per lane, `O(n)` list
appends in the number of lanes offered; structural recursion on the lane count `n`. -/
def ffn4LaneStep (codeOf : Nat → Bool × Std.I8) (l0 : Nat) : Nat → List Std.I8 × Bool
  | 0 => ([], true)
  | n + 1 =>
    if (ffn4LaneStep codeOf l0 n).2 = true then
      ((ffn4LaneStep codeOf l0 n).1 ++ [(codeOf (l0 + n)).2], (codeOf (l0 + n)).1)
    else ((ffn4LaneStep codeOf l0 n).1, false)

/-- The zero-lane run contributes nothing and keeps the flag. -/
lemma ffn4LaneStep_zero (codeOf : Nat → Bool × Std.I8) (l0 : Nat) :
    ffn4LaneStep codeOf l0 0 = ([], true) := rfl

/-- One-step unfolding of `ffn4LaneStep`: on a set flag the next lane's code is appended and the
flag is replaced by that lane's own flag; on a cleared flag the run stops. -/
lemma ffn4LaneStep_succ (codeOf : Nat → Bool × Std.I8) (l0 n : Nat) :
    ffn4LaneStep codeOf l0 (n + 1) =
      (if (ffn4LaneStep codeOf l0 n).2 = true then
        ((ffn4LaneStep codeOf l0 n).1 ++ [(codeOf (l0 + n)).2], (codeOf (l0 + n)).1)
       else ((ffn4LaneStep codeOf l0 n).1, false)) := rfl

/-- The contributed list never exceeds the number of lanes offered. -/
lemma ffn4LaneStep_fst_len (codeOf : Nat → Bool × Std.I8) (l0 n : Nat) :
    (ffn4LaneStep codeOf l0 n).1.length ≤ n := by
  induction n with
  | zero => rw [ffn4LaneStep_zero]; simp
  | succ n ih =>
    rw [ffn4LaneStep_succ]
    by_cases h : (ffn4LaneStep codeOf l0 n).2 = true
    · rw [if_pos h]; dsimp only
      rw [List.length_append]
      simp only [List.length_cons, List.length_nil]
      omega
    · rw [if_neg h]; dsimp only; omega

/-- Once the running flag is cleared, all further lanes contribute nothing and the flag stays
cleared. -/
lemma ffn4LaneStep_false (codeOf : Nat → Bool × Std.I8) (l0 n : Nat)
    (h : (ffn4LaneStep codeOf l0 n).2 = false) (k : Nat) :
    ffn4LaneStep codeOf l0 (n + k) = ((ffn4LaneStep codeOf l0 n).1, false) := by
  induction k with
  | zero => rw [Nat.add_zero]; exact Prod.ext rfl h
  | succ k ih =>
    rw [show n + (k + 1) = (n + k) + 1 by omega, ffn4LaneStep_succ,
      if_neg (by rw [ih]; simp), ih]

/-- With every offered lane's flag set, the run contributes exactly one code per lane and ends with
the flag set: the contributed list has length `n`. -/
lemma ffn4LaneStep_all_true (codeOf : Nat → Bool × Std.I8) (l0 n : Nat)
    (h : ∀ m, m < n → (codeOf (l0 + m)).1 = true) :
    (ffn4LaneStep codeOf l0 n).1.length = n ∧ (ffn4LaneStep codeOf l0 n).2 = true := by
  induction n with
  | zero => rw [ffn4LaneStep_zero]; simp
  | succ n ih =>
    have ih' := ih (fun m hm => h m (by omega))
    rw [ffn4LaneStep_succ, if_pos ih'.2]
    refine ⟨?_, ?_⟩
    · dsimp only
      rw [List.length_append]
      simp only [List.length_cons, List.length_nil]
      omega
    · dsimp only; exact h n (by omega)

/-- **The `chart.quant_act_loop0_loop1` loop spec.** With the segment `[base, base + take)` inside
`x`, enough capacity in `codes0` for every lane, and the per-lane code `codeOf` (the extracted
inner computation's outcome at that lane), the loop from lane `l0` returns `ok (codesN, fl)` where
`codesN.val` is `codes0.val` followed by the reference codes of the lanes `l0, …, take - 1` up to the
first lane whose flag clears, and `fl` is the reference's final flag. `proved` relative to the
`hinner` hypothesis (the two opaque externals). -/
theorem ffn4_lane_loop_spec (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (codeOf : Nat → Bool × Std.I8) (l0 : Std.Usize)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes0.val.length + take.val < Usize.max) (hl0 : l0.val ≤ take.val)
    (hinner : ∀ lane : Std.Usize, ∀ v : Std.I64,
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
    chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 ⦃ out =>
      out.1.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (take.val - l0.val)).1 ∧
      out.2 = (ffn4LaneStep codeOf l0.val (take.val - l0.val)).2 ⦄ := by
  unfold chart.quant_act_loop0_loop1
  apply loop.spec_decr_nat
    (measure := fun st : (alloc.vec.Vec Std.I8) × (Bool × Std.Usize) =>
      take.val - st.2.2.val)
    (inv := fun st : (alloc.vec.Vec Std.I8) × (Bool × Std.Usize) =>
      l0.val ≤ st.2.2.val ∧ st.2.2.val ≤ take.val ∧
      st.1.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (st.2.2.val - l0.val)).1 ∧
      st.2.1 = (ffn4LaneStep codeOf l0.val (st.2.2.val - l0.val)).2)
    (post := fun out : (alloc.vec.Vec Std.I8) × Bool =>
      out.1.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (take.val - l0.val)).1 ∧
      out.2 = (ffn4LaneStep codeOf l0.val (take.val - l0.val)).2)
  · rintro ⟨codes, ok1, lane⟩ ⟨h0le, hletake, hval, hok⟩
    dsimp only at h0le hletake hval hok ⊢
    by_cases hlt : lane.val < take.val
    · by_cases htrue : ok1 = true
      · have hlen : codes.val.length < Usize.max := by
          rw [hval, List.length_append]
          have hlen' := ffn4LaneStep_fst_len codeOf l0.val (lane.val - l0.val)
          omega
        obtain ⟨c1, l', _, hc1v, hl'v, hbody⟩ :=
          ffn4_lane_body_turn x base take s pow2 shift mask codes lane
            (codeOf lane.val).1 (codeOf lane.val).2 hlen hbound hlt (fun v => hinner lane v)
        have hflag : (ffn4LaneStep codeOf l0.val (lane.val - l0.val)).2 = true := by
          rw [← hok]; exact htrue
        rw [htrue, hbody, WP.spec_ok]
        dsimp only
        show (l0.val ≤ l'.val ∧ l'.val ≤ take.val ∧
            c1.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (l'.val - l0.val)).1 ∧
            (codeOf lane.val).1 = (ffn4LaneStep codeOf l0.val (l'.val - l0.val)).2) ∧
          take.val - l'.val < take.val - lane.val
        refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
        · rw [hl'v]; omega
        · rw [hl'v]; omega
        · rw [hc1v, hval, List.append_assoc]
          congr 1
          rw [hl'v, show lane.val + 1 - l0.val = (lane.val - l0.val) + 1 by omega,
            ffn4LaneStep_succ, if_pos hflag,
            show l0.val + (lane.val - l0.val) = lane.val by omega]
        · rw [hl'v, show lane.val + 1 - l0.val = (lane.val - l0.val) + 1 by omega,
            ffn4LaneStep_succ, if_pos hflag,
            show l0.val + (lane.val - l0.val) = lane.val by omega]
        · rw [hl'v]; omega
      · have hfalse : ok1 = false := by cases ok1 <;> simp_all
        have hflag : (ffn4LaneStep codeOf l0.val (lane.val - l0.val)).2 = false := by
          rw [← hok]; exact hfalse
        have hstop := ffn4LaneStep_false codeOf l0.val (lane.val - l0.val) hflag
          (take.val - lane.val)
        have hidx : take.val - l0.val = (lane.val - l0.val) + (take.val - lane.val) := by omega
        rw [hfalse, ffn3_lane_body_done_ok x base take s pow2 shift mask codes lane hlt,
          WP.spec_ok]
        dsimp only
        show codes.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (take.val - l0.val)).1 ∧
          false = (ffn4LaneStep codeOf l0.val (take.val - l0.val)).2
        refine ⟨?_, ?_⟩
        · rw [hval, hidx, hstop]
        · rw [hidx, hstop]
    · have heq : lane.val = take.val := le_antisymm hletake (le_of_not_gt hlt)
      rw [ffn3_lane_body_done_lane x base take s pow2 shift mask codes ok1 lane hlt, WP.spec_ok]
      dsimp only
      show codes.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (take.val - l0.val)).1 ∧
        ok1 = (ffn4LaneStep codeOf l0.val (take.val - l0.val)).2
      refine ⟨?_, ?_⟩
      · rw [hval, heq]
      · rw [hok, heq]
  · refine ⟨le_refl _, hl0, ?_, ?_⟩
    · rw [Nat.sub_self, ffn4LaneStep_zero]
      simp
    · rw [Nat.sub_self, ffn4LaneStep_zero]

/-- **`chart.quant_act_loop0_loop1` is total and denotes the reference codes.** With the segment
inside `x`, enough capacity and the per-lane code `codeOf`, the loop returns `ok (codesN, fl)` with
`codesN.val` the starting codes followed by the reference codes and `fl` the reference's final flag.
`proved` relative to the `hinner` hypothesis (the two opaque externals). -/
theorem ffn4_lane_loop_ok (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (codeOf : Nat → Bool × Std.I8) (l0 : Std.Usize)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes0.val.length + take.val < Usize.max) (hl0 : l0.val ≤ take.val)
    (hinner : ∀ lane : Std.Usize, ∀ v : Std.I64,
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
    ∃ codesN : alloc.vec.Vec Std.I8, ∃ fl : Bool,
      chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 = ok (codesN, fl) ∧
      fl = (ffn4LaneStep codeOf l0.val (take.val - l0.val)).2 ∧
      codesN.val = codes0.val ++ (ffn4LaneStep codeOf l0.val (take.val - l0.val)).1 := by
  obtain ⟨out, hout, h1, h2⟩ :=
    WP.spec_imp_exists (ffn4_lane_loop_spec x base take s pow2 shift mask codes0 codeOf l0
      hbound hcap hl0 hinner)
  exact ⟨out.1, out.2, hout, h2, h1⟩

/-- **Codes of length lane-count.** With every offered lane's flag set, the loop returns the flag
set and appends exactly one code per lane, so the result's code count is the lane count
`take - l0`. -/
theorem ffn4_lane_loop_len (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (codeOf : Nat → Bool × Std.I8) (l0 : Std.Usize)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes0.val.length + take.val < Usize.max) (hl0 : l0.val ≤ take.val)
    (hinner : ∀ lane : Std.Usize, ∀ v : Std.I64,
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
    (hall : ∀ m : Nat, m < take.val → (codeOf (l0.val + m)).1 = true) :
    ∃ codesN : alloc.vec.Vec Std.I8,
      chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 = ok (codesN, true) ∧
      codesN.val.length = codes0.val.length + (take.val - l0.val) := by
  obtain ⟨codesN, fl, ho, hfl, hval⟩ :=
    ffn4_lane_loop_ok x base take s pow2 shift mask codes0 codeOf l0 hbound hcap hl0 hinner
  have hcell := ffn4LaneStep_all_true codeOf l0.val (take.val - l0.val)
    (fun m hm => hall m (by omega))
  refine ⟨codesN, ?_, ?_⟩
  · rw [ho, hfl, hcell.2]
  · rw [hval, List.length_append, hcell.1]

/-! ## §3 · Axiom cones

The headlines name the extracted bodies `chart.quant_act_loop0_loop1` and
`chart.quant_act_loop0_loop1.body` (directly and through `ffn4_lane_body_turn_zero_ref`), so their
cone is the model's closed set `{propext, Classical.choice, Quot.sound}` plus the bodies' two opaque
externals (`core.num.I64.unsigned_abs`, `core.ops.range.RangeInclusive.contains`, the
`proof/regen.sh` opaque list), exactly as `Serve/Ffn3.lean` §7 records for its body-relative
headlines. The general turn arm and the loop spec *consume* those externals through their `hinner`
hypothesis and their own proofs add no axiom. -/


end apeiron_word
