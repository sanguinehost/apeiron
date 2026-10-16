/-
  `Serve/Block/R4/Ffn8.lean` — restates the lane loop `chart.quant_act_loop0_loop1`
  (`src/engine/chart/mod.rs:250`) with a **value-indexed** per-lane code, discharging the hypothesis
  that `Serve/Block/R4/Ffn7.lean` refuted.

  `Serve/Block/R4/Ffn6.lean` (and `Serve/Block/Ffn4.lean` before it) state the loop spec with a
  lane-only `codeOf : Nat → Bool × Std.I8` and the hypothesis
  `hinner : ∀ lane v, <inner code at v> = ok (codeOf lane.val)`. `ffn7_hinner_false` proves no such
  `codeOf` exists: the code is a function of the lane's value `v`, which a lane-only index cannot
  see. So `Ffn6`'s `word_quant_act_loop0_ok`, `word_quant_act_ok` and `word_quant_act_refuses` are
  vacuous.

  This file replaces that premise:
  * §1 `ffn8LaneStep` is the lane reference with a code `codeOf : Nat → Std.I64 → Bool × Std.I8`
    indexed by the lane and the value read from the input slice — the same shape the extracted body
    reads (`x[base + lane]`), with the lane index absolute so a per-block code is expressible.
  * §2 the turn arm `ffn8_lane_body_turn` carries the value-indexed code through the extracted body.
  * §3 `ffn8_lane_loop_spec` refines the whole lane loop to `ffn8LaneStep`, with the value-indexed
    `hinner`.
  * §4 `ffn8_codeOf_spec` discharges `hinner` from `ffn7_lane_code_eq`'s concrete hypotheses
    (`pow2/shift/mask/s` bounds) with the concrete code `ffn8CodeOf`; `ffn8_lane_loop_spec_ffn7` is
    the loop spec with no `hinner` premise at all. This is the non-vacuous form.

  Worked (the value-indexed loop spec and its concrete discharge, and non-vacuity). Not worked here:
  the block-level headlines `word_quant_act_loop0_ok`/`word_quant_act_ok`/`word_quant_act_refuses`,
  which need the block turn and the two drivers restated over `ffn8LaneStep`; the file records the
  exact edit instead of making it.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains` are definitions in the
    pinned model; no proof step invokes an axiom.
  * `hscale` (the block-level `kern.pow2.act_scale` search) is untouched: it belongs to the block
    loop, not the lane code.
  * The `F32`/`AVX2` bodies, the pool dispatch, the `unsafe` pointer writes and the growable `Vec`
    allocation: named boundaries.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch or the two model externals. The statements are about the
  Aeneas-extracted bodies over the model `Slice`, `Vec` and word scalars.
-/

import ApeironWord
import Serve.Block.R4.Ffn7

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §1 · The value-indexed lane reference

`ffn8LaneStep` is `ffn4LaneStep` with the per-lane code read from the input: lane `l0 + n` carries
`codeOf (base + (l0 + n)) (x[base + (l0 + n)])`, the absolute lane index and the value the extracted
body reads. A per-block code is expressible through the absolute lane. -/

/-- The codes contributed by the lanes `l0, …, l0 + n - 1` in order, with the running flag: while
the flag holds, lane `l0 + k` contributes `(codeOf (base + l0 + k) (x[base + l0 + k])).2`; the first
lane whose flag is `false` stops the run and contributes last. `l0` is the lane the loop starts at,
`base` the segment base, so the code index is the absolute lane. Temporal and spatial cost: one step
per lane, `O(n)` list appends; structural recursion on the lane count `n`. -/
def ffn8LaneStep (codeOf : Nat → Std.I64 → Bool × Std.I8) (x : Slice Std.I64) (base l0 : Nat) :
    Nat → List Std.I8 × Bool
  | 0 => ([], true)
  | n + 1 =>
    if (ffn8LaneStep codeOf x base l0 n).2 = true then
      ((ffn8LaneStep codeOf x base l0 n).1 ++
        [(codeOf (base + (l0 + n)) (x.val.getD (base + (l0 + n)) 0#i64)).2],
       (codeOf (base + (l0 + n)) (x.val.getD (base + (l0 + n)) 0#i64)).1)
    else ((ffn8LaneStep codeOf x base l0 n).1, false)

/-- The zero-lane run contributes nothing and keeps the flag. -/
lemma ffn8LaneStep_zero (codeOf : Nat → Std.I64 → Bool × Std.I8) (x : Slice Std.I64) (base l0 : Nat) :
    ffn8LaneStep codeOf x base l0 0 = ([], true) := rfl

/-- One-step unfolding of `ffn8LaneStep`: on a set flag the next lane's code is appended and the
flag is replaced by that lane's own flag; on a cleared flag the run stops. -/
lemma ffn8LaneStep_succ (codeOf : Nat → Std.I64 → Bool × Std.I8) (x : Slice Std.I64) (base l0 n : Nat) :
    ffn8LaneStep codeOf x base l0 (n + 1) =
      (if (ffn8LaneStep codeOf x base l0 n).2 = true then
        ((ffn8LaneStep codeOf x base l0 n).1 ++
          [(codeOf (base + (l0 + n)) (x.val.getD (base + (l0 + n)) 0#i64)).2],
         (codeOf (base + (l0 + n)) (x.val.getD (base + (l0 + n)) 0#i64)).1)
       else ((ffn8LaneStep codeOf x base l0 n).1, false)) := rfl

/-- The contributed list never exceeds the number of lanes offered. -/
lemma ffn8LaneStep_fst_len (codeOf : Nat → Std.I64 → Bool × Std.I8) (x : Slice Std.I64)
    (base l0 n : Nat) : (ffn8LaneStep codeOf x base l0 n).1.length ≤ n := by
  induction n with
  | zero => rw [ffn8LaneStep_zero]; simp
  | succ n ih =>
    rw [ffn8LaneStep_succ]
    by_cases h : (ffn8LaneStep codeOf x base l0 n).2 = true
    · rw [if_pos h]; dsimp only
      rw [List.length_append]
      simp only [List.length_cons, List.length_nil]
      omega
    · rw [if_neg h]; dsimp only; omega

/-- Once the running flag is cleared, all further lanes contribute nothing and the flag stays
cleared. -/
lemma ffn8LaneStep_false (codeOf : Nat → Std.I64 → Bool × Std.I8) (x : Slice Std.I64)
    (base l0 n : Nat) (h : (ffn8LaneStep codeOf x base l0 n).2 = false) (k : Nat) :
    ffn8LaneStep codeOf x base l0 (n + k) = ((ffn8LaneStep codeOf x base l0 n).1, false) := by
  induction k with
  | zero => rw [Nat.add_zero]; exact Prod.ext rfl h
  | succ k ih =>
    rw [show n + (k + 1) = (n + k) + 1 by omega, ffn8LaneStep_succ,
      if_neg (by rw [ih]; simp), ih]

/-- With every offered lane's flag set, the run contributes exactly one code per lane and ends with
the flag set: the contributed list has length `n`. -/
lemma ffn8LaneStep_all_true (codeOf : Nat → Std.I64 → Bool × Std.I8) (x : Slice Std.I64)
    (base l0 n : Nat) (h : ∀ m, m < n → (codeOf (base + (l0 + m))
      (x.val.getD (base + (l0 + m)) 0#i64)).1 = true) :
    (ffn8LaneStep codeOf x base l0 n).1.length = n ∧ (ffn8LaneStep codeOf x base l0 n).2 = true := by
  induction n with
  | zero => rw [ffn8LaneStep_zero]; simp
  | succ n ih =>
    have ih' := ih (fun m hm => h m (by omega))
    rw [ffn8LaneStep_succ, if_pos ih'.2]
    refine ⟨?_, ?_⟩
    · dsimp only
      rw [List.length_append]
      simp only [List.length_cons, List.length_nil]
      omega
    · dsimp only; exact h n (by omega)

/-! ## §2 · The turn arm of `chart.quant_act_loop0_loop1.body`

The turn at `s` with value read `x[base + lane]`: the inner code computation is bound to the
value-indexed `hinner` at that value, then the pushed code and the running flag are `codeOf`'s
components at the absolute lane `base + lane`. -/

/-- **One `chart.quant_act_loop0_loop1` body turn, value-indexed.** With `lane < take` and the flag
set, given that the lane's code computation at a value `v` returns `ok (codeOf (base + lane) v)` (the
extracted `if s = 0 then (ok (true, 0#i8)) else ⋯` with the lane's value), the body pushes
`(codeOf (base + lane) (x[base+lane])).2` and continues with `(codeOf (base + lane) (x[base+lane])).1`,
advancing to `lane + 1`. `proved` relative to `hinner`. -/
theorem ffn8_lane_body_turn (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes : alloc.vec.Vec Std.I8)
    (lane : Std.Usize) (codeOf : Nat → Std.I64 → Bool × Std.I8)
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
        ok (b, i3)) = ok (codeOf (base.val + lane.val) v)) :
    ∃ c1 : alloc.vec.Vec Std.I8, ∃ l' : Std.Usize,
      codes.push (codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).2 = ok c1 ∧
      c1.val = codes.val ++ [(codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).2] ∧
      l'.val = lane.val + 1 ∧
      chart.quant_act_loop0_loop1.body x base take s pow2 shift mask codes true lane
        = ok (ControlFlow.cont (c1,
            (codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).1, l')) := by
  have hlen := Slice.length_ineq x
  have hsum : base.val + lane.val ≤ Usize.max := by omega
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hsum)
  have hidx : i.val < x.val.length := by rw [hiv]; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hidx)
  have hgetD : i1 = x.val.getD (base.val + lane.val) 0#i64 := by
    rw [hi1v, ← hiv]
    exact (List.getD_eq_getElem x.val 0#i64 hidx).symm
  have h1v : (1#usize : Std.Usize).val = 1 := rfl
  have hmax1 : lane.val + (1#usize : Std.Usize).val ≤ Usize.max := by omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := lane) (y := 1#usize) hmax1)
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec codes
      (codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).2 hpush)
  refine ⟨c1, i2, hc1, hc1v, ?_, ?_⟩
  · rw [hi2v, h1v]
  · unfold chart.quant_act_loop0_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane)]
    rw [hi]; simp only [bind_tc_ok]
    rw [hi1]; simp only [bind_tc_ok]
    rw [hinner i1]
    rw [bind_tc_ok]
    rw [hgetD]
    change (do let codes1 ← codes.push (codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).2;
               let lane1 ← lane + 1#usize;
               ok (cont (codes1,
                 (codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).1, lane1)))
      = ok (cont (c1,
          (codeOf (base.val + lane.val) (x.val.getD (base.val + lane.val) 0#i64)).1, i2))
    rw [hc1]; simp only [bind_tc_ok]
    rw [hi2]; simp only [bind_tc_ok]

/-! ## §3 · The lane loop `chart.quant_act_loop0_loop1`

The extracted loop pushes one code per lane from the starting lane up to `take`, stopping early at
the first lane whose code clears the running flag; `ffn8_lane_loop_spec` refines it to the
value-indexed `ffn8LaneStep`. -/

/-- **The `chart.quant_act_loop0_loop1` loop spec, value-indexed.** With the segment
`[base, base + take)` inside `x`, enough capacity in `codes0`, and the per-lane code
`codeOf : Nat → Std.I64 → Bool × Std.I8` (the extracted inner computation's outcome at the lane's
value, indexed by the absolute lane), the loop from lane `l0` returns `ok (codesN, fl)` where
`codesN.val` is `codes0.val` followed by the reference codes and `fl` the reference's final flag.
`proved` relative to `hinner`. -/
theorem ffn8_lane_loop_spec (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (codeOf : Nat → Std.I64 → Bool × Std.I8) (l0 : Std.Usize)
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
        ok (b, i3)) = ok (codeOf (base.val + lane.val) v)) :
    chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 ⦃ out =>
      out.1.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).1 ∧
      out.2 = (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).2 ⦄ := by
  unfold chart.quant_act_loop0_loop1
  apply loop.spec_decr_nat
    (measure := fun st : (alloc.vec.Vec Std.I8) × (Bool × Std.Usize) =>
      take.val - st.2.2.val)
    (inv := fun st : (alloc.vec.Vec Std.I8) × (Bool × Std.Usize) =>
      l0.val ≤ st.2.2.val ∧ st.2.2.val ≤ take.val ∧
      st.1.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val (st.2.2.val - l0.val)).1 ∧
      st.2.1 = (ffn8LaneStep codeOf x base.val l0.val (st.2.2.val - l0.val)).2)
    (post := fun out : (alloc.vec.Vec Std.I8) × Bool =>
      out.1.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).1 ∧
      out.2 = (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).2)
  · rintro ⟨codes, ok1, lane⟩ ⟨h0le, hletake, hval, hok⟩
    dsimp only at h0le hletake hval hok ⊢
    by_cases hlt : lane.val < take.val
    · by_cases htrue : ok1 = true
      · have hlen : codes.val.length < Usize.max := by
          rw [hval, List.length_append]
          have hlen' := ffn8LaneStep_fst_len codeOf x base.val l0.val (lane.val - l0.val)
          omega
        obtain ⟨c1, l', _, hc1v, hl'v, hbody⟩ :=
          ffn8_lane_body_turn x base take s pow2 shift mask codes lane codeOf hlen hbound hlt
            (fun v => hinner lane v)
        have hflag : (ffn8LaneStep codeOf x base.val l0.val (lane.val - l0.val)).2 = true := by
          rw [← hok]; exact htrue
        rw [htrue, hbody, WP.spec_ok]
        dsimp only
        show (l0.val ≤ l'.val ∧ l'.val ≤ take.val ∧
            c1.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val (l'.val - l0.val)).1 ∧
            (codeOf (base.val + lane.val)
              (x.val.getD (base.val + lane.val) 0#i64)).1
              = (ffn8LaneStep codeOf x base.val l0.val (l'.val - l0.val)).2) ∧
          take.val - l'.val < take.val - lane.val
        refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
        · rw [hl'v]; omega
        · rw [hl'v]; omega
        · rw [hc1v, hval, List.append_assoc]
          congr 1
          rw [hl'v, show lane.val + 1 - l0.val = (lane.val - l0.val) + 1 by omega,
            ffn8LaneStep_succ, if_pos hflag,
            show base.val + (l0.val + (lane.val - l0.val)) = base.val + lane.val by omega]
        · rw [hl'v, show lane.val + 1 - l0.val = (lane.val - l0.val) + 1 by omega,
            ffn8LaneStep_succ, if_pos hflag,
            show base.val + (l0.val + (lane.val - l0.val)) = base.val + lane.val by omega]
        · rw [hl'v]; omega
      · have hfalse : ok1 = false := by cases ok1 <;> simp_all
        have hflag : (ffn8LaneStep codeOf x base.val l0.val (lane.val - l0.val)).2 = false := by
          rw [← hok]; exact hfalse
        have hstop := ffn8LaneStep_false codeOf x base.val l0.val (lane.val - l0.val) hflag
          (take.val - lane.val)
        have hidx : take.val - l0.val = (lane.val - l0.val) + (take.val - lane.val) := by omega
        rw [hfalse, ffn3_lane_body_done_ok x base take s pow2 shift mask codes lane hlt,
          WP.spec_ok]
        dsimp only
        show codes.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val
            (take.val - l0.val)).1 ∧
          false = (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).2
        refine ⟨?_, ?_⟩
        · rw [hval, hidx, hstop]
        · rw [hidx, hstop]
    · have heq : lane.val = take.val := le_antisymm hletake (le_of_not_gt hlt)
      rw [ffn3_lane_body_done_lane x base take s pow2 shift mask codes ok1 lane hlt, WP.spec_ok]
      dsimp only
      show codes.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val
          (take.val - l0.val)).1 ∧
        ok1 = (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).2
      refine ⟨?_, ?_⟩
      · rw [hval, heq]
      · rw [hok, heq]
  · refine ⟨le_refl _, hl0, ?_, ?_⟩
    · rw [Nat.sub_self, ffn8LaneStep_zero]
      simp
    · rw [Nat.sub_self, ffn8LaneStep_zero]

/-- **`chart.quant_act_loop0_loop1` is total and denotes the value-indexed reference codes.** With
the segment inside `x`, enough capacity and the per-lane code `codeOf`, the loop returns `ok` with
the reference codes and flag. `proved` relative to `hinner`. -/
theorem ffn8_lane_loop_ok (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (codeOf : Nat → Std.I64 → Bool × Std.I8) (l0 : Std.Usize)
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
        ok (b, i3)) = ok (codeOf (base.val + lane.val) v)) :
    ∃ codesN : alloc.vec.Vec Std.I8, ∃ fl : Bool,
      chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 = ok (codesN, fl) ∧
      fl = (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).2 ∧
      codesN.val = codes0.val ++ (ffn8LaneStep codeOf x base.val l0.val (take.val - l0.val)).1 := by
  obtain ⟨out, hout, h1, h2⟩ :=
    WP.spec_imp_exists (ffn8_lane_loop_spec x base take s pow2 shift mask codes0 codeOf l0
      hbound hcap hl0 hinner)
  exact ⟨out.1, out.2, hout, h2, h1⟩

/-! ## §4 · Discharging the value-indexed premise from `ffn7_lane_code_eq`

`ffn8CodeOf` is the per-lane code as a plain function of the value: `ffn7Code`'s outcome. Under
`ffn7_lane_code_eq`'s bounds it satisfies the value-indexed `hinner`, so the loop spec above holds
with **no** `hinner` premise — which is exactly what `ffn7_hinner_false` forbade for the lane-only
form. -/

/-- The per-lane code as a function of the value: `ffn7Code`'s outcome, with an unreachable fallback
on the (impossible) error branch. Lane-independent; the index `lane` is accepted to match
`ffn8LaneStep`'s `codeOf`. -/
def ffn8CodeOf (s : Std.U64) (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) :
    Nat → Std.I64 → Bool × Std.I8 :=
  fun _ v => match (ffn7Code s pow2 shift mask v).match with
    | .ok bc => bc
    | _ => (false, 0#i8)

/-- At the zero scale the code is `(true, 0)`, the reference's zero-scale short circuit. -/
theorem ffn8CodeOf_zero (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (lane : Nat)
    (v : Std.I64) :
    ffn8CodeOf 0#u64 pow2 shift mask lane v = (true, 0#i8) := by
  unfold ffn8CodeOf ffn7Code
  rw [if_pos (rfl : (0#u64 : Std.U64) = 0#u64)]
  simp only [Result.match.ok]

/-- **`ffn8CodeOf` discharges the value-indexed `hinner`.** Under `ffn7_lane_code_eq`'s concrete
hypotheses — `pow2 = true → shift.val < 64`, `pow2 = true → mask.val ≤ 2^63 - 1`,
`s.val ≤ 2^63 - 1` — the extracted lane code equals `ok (ffn8CodeOf s pow2 shift mask lane v)` at
every lane and value. `proved`. -/
theorem ffn8_codeOf_spec (s : Std.U64) (pow2 : Bool) (shift : Std.U32) (mask : Std.U64)
    (hshift : pow2 = true → shift.val < 64)
    (hmask : pow2 = true → mask.val ≤ 2 ^ 63 - 1)
    (hsle : s.val ≤ 2 ^ 63 - 1) (lane : Nat) (v : Std.I64) :
    ffn7Code s pow2 shift mask v = ok (ffn8CodeOf s pow2 shift mask lane v) := by
  obtain ⟨b, code, _, _, _, hcode, _⟩ := ffn7_lane_code_eq s pow2 shift mask v hshift hmask hsle
  unfold ffn8CodeOf
  rw [hcode]
  simp only [Result.match.ok]

/-- **The lane loop spec with no `hinner` premise.** With the segment inside `x`, enough capacity,
and `ffn7_lane_code_eq`'s bounds on the scale, the loop denotes `ffn8LaneStep` of the concrete code
`ffn8CodeOf s pow2 shift mask`. This is the non-vacuous form of `ffn8_lane_loop_spec`: the premise
is discharged rather than assumed. `proved`. -/
theorem ffn8_lane_loop_spec_ffn7 (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (l0 : Std.Usize)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes0.val.length + take.val < Usize.max) (hl0 : l0.val ≤ take.val)
    (hshift : pow2 = true → shift.val < 64)
    (hmask : pow2 = true → mask.val ≤ 2 ^ 63 - 1)
    (hsle : s.val ≤ 2 ^ 63 - 1) :
    chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 ⦃ out =>
      out.1.val = codes0.val ++
        (ffn8LaneStep (ffn8CodeOf s pow2 shift mask) x base.val l0.val (take.val - l0.val)).1 ∧
      out.2 =
        (ffn8LaneStep (ffn8CodeOf s pow2 shift mask) x base.val l0.val (take.val - l0.val)).2 ⦄ := by
  apply ffn8_lane_loop_spec x base take s pow2 shift mask codes0
    (ffn8CodeOf s pow2 shift mask) l0 hbound hcap hl0
  intro lane v
  rw [← ffn7Code_is_hinner_lhs]
  exact ffn8_codeOf_spec s pow2 shift mask hshift hmask hsle (base.val + lane.val) v

/-- **The lane loop is total and denotes `ffn8CodeOf`, with no `hinner` premise.** `proved`. -/
theorem ffn8_lane_loop_ok_ffn7 (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes0 : alloc.vec.Vec Std.I8)
    (l0 : Std.Usize)
    (hbound : base.val + take.val ≤ x.val.length)
    (hcap : codes0.val.length + take.val < Usize.max) (hl0 : l0.val ≤ take.val)
    (hshift : pow2 = true → shift.val < 64)
    (hmask : pow2 = true → mask.val ≤ 2 ^ 63 - 1)
    (hsle : s.val ≤ 2 ^ 63 - 1) :
    ∃ codesN : alloc.vec.Vec Std.I8, ∃ fl : Bool,
      chart.quant_act_loop0_loop1 x codes0 true base take s pow2 shift mask l0 = ok (codesN, fl) ∧
      fl = (ffn8LaneStep (ffn8CodeOf s pow2 shift mask) x base.val l0.val
        (take.val - l0.val)).2 ∧
      codesN.val = codes0.val ++
        (ffn8LaneStep (ffn8CodeOf s pow2 shift mask) x base.val l0.val (take.val - l0.val)).1 := by
  obtain ⟨out, hout, h1, h2⟩ :=
    WP.spec_imp_exists (ffn8_lane_loop_spec_ffn7 x base take s pow2 shift mask codes0 l0
      hbound hcap hl0 hshift hmask hsle)
  exact ⟨out.1, out.2, hout, h2, h1⟩

/-! ## §5 · Non-vacuity

The value-indexed premise is satisfiable at concrete inputs — the direct contrast to
`ffn7_hinner_false`, which refuted the lane-only form — and the loop conclusion is inhabited. -/

/-- `usize` is positive: the model's `Usize.max` is `2^64 - 1`. Needed to build a concrete capacity
witness for the loop spec. -/
private lemma ffn8_usize_max_eq : Usize.max = 2 ^ System.Platform.numBits - 1 := by
  rw [Usize.max_def, Usize.numBits_def, UScalarTy.Usize_numBits_eq]

private lemma ffn8_usize_max_pos : (0 : Nat) < Usize.max := by
  have h : (1#usize).val ≤ Usize.max := by
    have hbv : (1#usize).bv.toNat < 2 ^ System.Platform.numBits := by
      simpa only [UScalarTy.Usize_numBits_eq] using (1#usize).bv.isLt
    show (1#usize).bv.toNat ≤ Usize.max
    rw [ffn8_usize_max_eq]
    omega
  have h1 : (1#usize).val = 1 := by scalar_tac
  omega

/-- **The value-indexed `hinner` is satisfiable.** At the unit scale (`s = 1#u64`, `pow2 = false`,
bounds vacuous) there is a `codeOf : Nat → Std.I64 → Bool × Std.I8` with
`ffn7Code 1#u64 false 0 0 v = ok (codeOf lane v)` at every lane and value — the premise `Ffn7`
proved unreachable for a lane-only code. -/
example : ∃ codeOf : Nat → Std.I64 → Bool × Std.I8,
    ∀ lane : Nat, ∀ v : Std.I64,
      ffn7Code 1#u64 false 0#u32 0#u64 v = ok (codeOf lane v) :=
  ⟨ffn8CodeOf 1#u64 false 0#u32 0#u64,
   fun lane v =>
     ffn8_codeOf_spec 1#u64 false 0#u32 0#u64
       (by intro h; exact absurd h (by decide))
       (by intro h; exact absurd h (by decide)) (by decide) lane v⟩

/-- **The loop conclusion is inhabited at a concrete input.** At the zero scale on an empty segment
the loop returns the starting codes with the flag set: `codesN.val = []`, `fl = true`. -/
example : ∃ codesN : alloc.vec.Vec Std.I8, ∃ fl : Bool,
    chart.quant_act_loop0_loop1 (Slice.new Std.I64)
        (alloc.vec.Vec.from ([] : List Std.I8) (by simp)) true
        0#usize 0#usize 0#u64 false 0#u32 0#u64 0#usize = ok (codesN, fl) ∧
      fl = true ∧ codesN.val = [] := by
  obtain ⟨codesN, fl, ho, hfl, hval⟩ :=
    ffn8_lane_loop_ok_ffn7 (Slice.new Std.I64) 0#usize 0#usize 0#u64 false 0#u32 0#u64
      (alloc.vec.Vec.from ([] : List Std.I8) (by simp)) 0#usize
      (by simp) (by simp; exact ffn8_usize_max_pos) (by simp)
      (by intro h; exact absurd h (by decide))
      (by intro h; exact absurd h (by decide)) (by decide)
  refine ⟨codesN, fl, ho, ?_, ?_⟩
  · simp [hfl, Nat.sub_self, ffn8LaneStep_zero]
  · simp [hval, Nat.sub_self, ffn8LaneStep_zero]

/-- **The reference reads the slice, value-indexed.** At the zero scale the per-lane code is `(true,
0)` at every value, so a two-lane reference run contributes `[0#i8, 0#i8]` with the flag set — the
value read from the slice is what the lane-only form could not carry. -/
example : (ffn8LaneStep (ffn8CodeOf 0#u64 false 0#u32 0#u64) (Slice.new Std.I64) 0 0 2).1
    = [0#i8, 0#i8] := by
  rw [show ffn8CodeOf 0#u64 false 0#u32 0#u64 = fun _ _ => (true, 0#i8) from by
    funext lane v; exact ffn8CodeOf_zero false 0#u32 0#u64 lane v]
  decide

end apeiron_word
