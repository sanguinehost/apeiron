/-
  `Serve/Ffn3.lean` — reference semantics for the Q8-activation quantiser of
  `src/engine/chart/mod.rs` (`chart.quant_act`) and the wrappers that consume it
  (`chart.ffn.quantize`, `chart.ffn.swiglu_cols`), plus the closed-cone restatement of
  `chart.ffn.swiglu_one`'s value claim.

  Companion of `Serve/Ffn2.lean` (the integer sub-steps `chart.act_quot_rem`,
  `chart.ffn.proj_split`, `chart.ffn.swiglu_one`), which is imported and not edited here.
  Group coverage:

  * `chart.quant_act` (`chart/mod.rs:250`) — §1 the absmax loop `chart.quant_act_loop0_loop0`
    equals `ffn3AbsMaxRef` (the maximum of `|x[base+l]|` over `l < take`), with its upper-bound and
    attainment characterisation; §2 `kern.pow2.act_scale` returns the least power of two at least
    `⌈absmax/127⌉` (`ffn3ScaleLeast`) and never `none`; §3 the reference code one lane of the code
    loop `chart.quant_act_loop0_loop1` contributes (`ffn3LaneCodeRef`), its range and flag
    properties, and the loop body's two stopping arms. The turn arm is open (§3's residual).
  * `chart.ffn.quantize` (`ffn.rs:236`) — §4 the `ok_or` wrapper: the `Ok` and the `Dequant`
    refusal, relative to the pooled dispatch `chart.ffn.quant_act_pooled` it delegates to.
  * `chart.ffn.swiglu_cols` (`ffn.rs:493`) — §5 the geometry refusals of `swiglu_cols_carrier`,
    relative to `mixer.carrier_from_env`.
  * `chart.ffn.swiglu_one` — §6 the value claim with the `i64::try_from` step discharged by the
    model's `I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from` definition, so the statement
    carries no hypothesis about an unmodeled external and no axiom.

  Boundaries, with reason:
  * `core.num.I64.unsigned_abs`, `core.num.U64.is_power_of_two`, `core.num.U64.trailing_zeros`,
    `core.ops.range.RangeInclusive.contains` are `axiom` in `ApeironWord` (`proof/regen.sh`'s
    opaque list). §1 and §3 take their behaviour as named hypotheses; no proof step invokes an
    axiom, and the `#print axioms` lines in §7 show the three-axiom cone for the headlines over plain
    words and reference definitions, plus exactly these externals where a statement names a body that
    calls them.
  * `alloc.vec.Vec` growth (`chart.quant_act_loop0_loop1`'s `Vec::push`, the block loop
    `chart.quant_act_loop0` and `chart.QuantAct.dequantizes`): §3 states the lane's reference and
    proves the body's stopping arms; the turn arm, the loop-level invariant over the growable `Vec`
    and the `dequantizes` correspondence are the residual obligation (§3).
  * the pool dispatch (`chart.pool.chunks.for_vec_chunks`), `mixer.carrier_from_env`, `F32`/AVX2
    bodies, `unsafe` raw-pointer writes: boundaries named in the §4/§5 statements.

  Trust boundary: nothing here covers rustc/LLVM codegen, the `unsafe` pointer writes, the `F32`
  arithmetic, the pool dispatch, the growable `Vec` allocation, or the four opaque externals above.
  The statements are about the Aeneas-extracted bodies over the model `Slice`, `Vec` and word
  scalars.
-/

import ApeironWord
import Numeric.Engine.Rope
import Serve.Block.Ffn2

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow Error

/-! ## §0 · Shared word bounds -/

/-- Every `u64` value is below `2^64`. -/
lemma ffn3_u64_lt (x : Std.U64) : x.val < 2 ^ 64 := by
  have h := x.hBounds
  simpa using h

/-- Every `u64` value is at most `U64.max = 2^64 - 1`. -/
lemma ffn3_u64_le_max (x : Std.U64) : x.val ≤ U64.max := by
  have h := ffn3_u64_lt x
  rw [U64.max_eq]; omega

/-- `List.getD` agrees with the element at an in-range index (the read `Slice.index_usize`
performs). -/
lemma ffn3_getD_getElem (xs : List Std.I64) (i : Nat) (h : i < xs.length) :
    (xs[i]'h).val = (xs.getD i 0#i64).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

/-! ## §1 · `chart.quant_act_loop0_loop0`: the absmax loop

`src/engine/chart/mod.rs:261-267`. The Rust body keeps the running maximum of `|x[base + lane]|`
over `lane < take`:

```
let mut absmax: u64 = 0;
while lane < take { let mag = x[base + lane].unsigned_abs(); if mag > absmax { absmax = mag }; lane += 1 }
```

The reference `ffn3AbsMaxRef x base take` is the maximum of `|x[base+l]|` over `l < take` — the
recursion `max (ref take) |x[base+take]|`, the mathematical maximum of the segment, not the loop's
shape. The hypotheses are the ones the body needs: `base + take` inside `x` (so every
`x[base+lane]` is in range and the `Usize` sums do not overflow) and the behaviour of the opaque
`i64::unsigned_abs`. -/

/-- `|z|` as a natural for an `i64` word: the value `core.num.I64.unsigned_abs` returns. -/
def ffn3Abs (z : Std.I64) : Nat := z.val.natAbs

/-- The maximum of `|x[base+l]|` over `l < take`; `0` at `take = 0`. -/
def ffn3AbsMaxRef (x : List Std.I64) (base : Nat) : Nat → Nat
  | 0 => 0
  | (take + 1) => max (ffn3AbsMaxRef x base take) ((x.getD (base + take) 0#i64).val.natAbs)

theorem ffn3AbsMaxRef_zero (x : List Std.I64) (base : Nat) : ffn3AbsMaxRef x base 0 = 0 := rfl

theorem ffn3AbsMaxRef_succ (x : List Std.I64) (base take : Nat) :
    ffn3AbsMaxRef x base (take + 1)
      = max (ffn3AbsMaxRef x base take) ((x.getD (base + take) 0#i64).val.natAbs) := rfl

/-- Characterisation, upper bound: every entry of the segment is at most the reference. -/
theorem ffn3AbsMaxRef_ub (x : List Std.I64) (base : Nat) :
    ∀ take l, l < take → (x.getD (base + l) 0#i64).val.natAbs ≤ ffn3AbsMaxRef x base take := by
  intro take
  induction take with
  | zero => intro l hl; exact absurd hl (Nat.not_lt_zero l)
  | succ take ih =>
    intro l hl
    rw [ffn3AbsMaxRef_succ]
    rcases lt_or_eq_of_le (Nat.le_of_lt_succ hl) with h | h
    · exact le_trans (ih l h) (Nat.le_max_left _ _)
    · subst h; exact Nat.le_max_right _ _

/-- Characterisation, attainment: at a nonempty segment the reference is one of its entries. -/
theorem ffn3AbsMaxRef_attained (x : List Std.I64) (base : Nat) :
    ∀ take, 0 < take →
      ∃ l, l < take ∧ ffn3AbsMaxRef x base take = (x.getD (base + l) 0#i64).val.natAbs := by
  intro take
  induction take with
  | zero => intro h; exact absurd h (Nat.lt_irrefl 0)
  | succ take ih =>
    intro _
    rw [ffn3AbsMaxRef_succ]
    by_cases hle : ffn3AbsMaxRef x base take ≤ (x.getD (base + take) 0#i64).val.natAbs
    · exact ⟨take, Nat.lt_succ_self take, Nat.max_eq_right hle⟩
    · have hlt : (x.getD (base + take) 0#i64).val.natAbs < ffn3AbsMaxRef x base take :=
        Nat.lt_of_not_le hle
      have hmax : max (ffn3AbsMaxRef x base take) ((x.getD (base + take) 0#i64).val.natAbs)
          = ffn3AbsMaxRef x base take := Nat.max_eq_left (le_of_lt hlt)
      rcases Nat.eq_zero_or_pos take with h | h
      · subst h
        rw [ffn3AbsMaxRef_zero] at hlt
        exact absurd hlt (Nat.not_lt_zero _)
      · obtain ⟨l, hl, heq⟩ := ih h
        exact ⟨l, Nat.lt_succ_of_lt hl, by rw [hmax, heq]⟩

/-- **One `chart.quant_act_loop0_loop0` body turn below the stopping bound.** The body reads
`x[base + lane]`, takes the `unsigned_abs` magnitude, keeps the larger of it and the accumulator,
and returns the counter `lane + 1`; the new accumulator is the reference maximum at `lane + 1`. -/
theorem ffn3_absmax_body_step (x : Slice Std.I64) (base take : Std.Usize) (absmax : Std.U64)
    (lane : Std.Usize)
    (huabs : ∀ z : Std.I64, ∃ u : Std.U64,
      core.num.I64.unsigned_abs z = ok u ∧ u.val = z.val.natAbs)
    (hbound : base.val + take.val ≤ x.val.length)
    (hlane : lane.val < take.val) :
    ∃ a' : Std.U64, ∃ l' : Std.Usize,
      chart.quant_act_loop0_loop0.body x base take absmax lane
        = ok (ControlFlow.cont (a', l')) ∧
      l'.val = lane.val + 1 ∧
      a'.val = max absmax.val ((x.val.getD (base.val + lane.val) 0#i64).val.natAbs) := by
  unfold chart.quant_act_loop0_loop0.body
  have hsum : base.val + lane.val ≤ Usize.max := by
    have hlen := Slice.length_ineq x
    omega
  have h1v : (1#usize : Std.Usize).val = 1 := rfl
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane)]
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := base) (y := lane) hsum)
  rw [hi]; simp only [bind_tc_ok]
  have hidx : i.val < x.val.length := by rw [hiv]; omega
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hidx)
  rw [hi1]; simp only [bind_tc_ok]
  obtain ⟨u, hu, huv⟩ := huabs i1
  rw [hu]; simp only [bind_tc_ok]
  have hmag : u.val = (x.val.getD (base.val + lane.val) 0#i64).val.natAbs := by
    rw [huv, hi1v, ffn3_getD_getElem x.val i.val hidx, hiv]
  have hmax1 : lane.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    have hlen := Slice.length_ineq x
    omega
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := lane) (y := 1#usize) hmax1)
  by_cases hgt : u > absmax
  · have hgtv : absmax.val < u.val := (UScalar.lt_equiv absmax u).1 hgt
    rw [if_pos hgt, hi2]; simp only [bind_tc_ok]
    refine ⟨u, i2, rfl, by rw [hi2v, h1v], ?_⟩
    rw [show ((x.val.getD (base.val + lane.val) 0#i64)).val.natAbs = u.val from hmag.symm]
    exact (Nat.max_eq_right (le_of_lt hgtv)).symm
  · have hgtv : u.val ≤ absmax.val := by
      by_contra h
      exact hgt ((UScalar.lt_equiv absmax u).2 (Nat.lt_of_not_le h))
    rw [if_neg hgt, hi2]; simp only [bind_tc_ok]
    refine ⟨absmax, i2, rfl, by rw [hi2v, h1v], ?_⟩
    rw [show ((x.val.getD (base.val + lane.val) 0#i64)).val.natAbs = u.val from hmag.symm]
    exact (Nat.max_eq_left hgtv).symm

/-- **One `chart.quant_act_loop0_loop0` body turn at or past the stopping bound.** With the counter
at `take` the body returns the accumulator unchanged. -/
theorem ffn3_absmax_body_done (x : Slice Std.I64) (base take : Std.Usize) (absmax : Std.U64)
    (lane : Std.Usize) (hge : ¬ lane.val < take.val) :
    chart.quant_act_loop0_loop0.body x base take absmax lane
      = ok (ControlFlow.done absmax) := by
  unfold chart.quant_act_loop0_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hge)]

/-- **The `chart.quant_act_loop0_loop0` loop from the zero accumulator.** Its result is the
reference maximum `ffn3AbsMaxRef x.val base.val take.val` of the segment. The invariant is the
write prefix (measured by `take - lane`), so the loop is entered with the accumulator `0` the
caller passes. `proved` relative to the `unsigned_abs` hypothesis. -/
theorem ffn3_absmax_loop_spec (x : Slice Std.I64) (base take : Std.Usize)
    (huabs : ∀ z : Std.I64, ∃ u : Std.U64,
      core.num.I64.unsigned_abs z = ok u ∧ u.val = z.val.natAbs)
    (hbound : base.val + take.val ≤ x.val.length) :
    chart.quant_act_loop0_loop0 x base take 0#u64 0#usize ⦃ out =>
      out.val = ffn3AbsMaxRef x.val base.val take.val ⦄ := by
  unfold chart.quant_act_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Std.U64 × Std.Usize => take.val - st.2.val)
    (inv := fun st => st.2.val ≤ take.val ∧
      st.1.val = ffn3AbsMaxRef x.val base.val st.2.val)
    (post := fun out : Std.U64 => out.val = ffn3AbsMaxRef x.val base.val take.val)
  · rintro ⟨a, j⟩ ⟨hj, ha⟩
    by_cases hlt : j.val < take.val
    · obtain ⟨a', j', hb, hj'v, ha'v⟩ := ffn3_absmax_body_step x base take a j huabs hbound hlt
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      show (j'.val ≤ take.val ∧ a'.val = ffn3AbsMaxRef x.val base.val j'.val) ∧
        take.val - j'.val < take.val - j.val
      refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [hj'v]; omega
      · rw [hj'v, ha'v, ha, ffn3AbsMaxRef_succ]
      · rw [hj'v]; omega
    · have hjeq : j.val = take.val := le_antisymm hj (le_of_not_gt hlt)
      have hb := ffn3_absmax_body_done x base take a j hlt
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      show a.val = ffn3AbsMaxRef x.val base.val take.val
      rw [ha, hjeq]
  · refine ⟨by simp, ?_⟩
    show (0#u64 : Std.U64).val = ffn3AbsMaxRef x.val base.val (0#usize : Std.Usize).val
    rw [show (0#usize : Std.Usize).val = 0 from rfl,
      show (0#u64 : Std.U64).val = 0 from rfl, ffn3AbsMaxRef_zero]

/-- **`chart.quant_act_loop0_loop0` is total and denotes the reference maximum.** With the segment
inside `x`, the loop returns `ok out` with `out` the maximum of `|x[base+l]|` over `l < take`.
Its characterising laws are `ffn3AbsMaxRef_ub` (every entry is at most it) and
`ffn3AbsMaxRef_attained` (a nonempty segment attains it). `proved` relative to the `unsigned_abs`
hypothesis. -/
theorem word_quant_absmax_ok (x : Slice Std.I64) (base take : Std.Usize)
    (huabs : ∀ z : Std.I64, ∃ u : Std.U64,
      core.num.I64.unsigned_abs z = ok u ∧ u.val = z.val.natAbs)
    (hbound : base.val + take.val ≤ x.val.length) :
    ∃ out : Std.U64,
      chart.quant_act_loop0_loop0 x base take 0#u64 0#usize = ok out ∧
      out.val = ffn3AbsMaxRef x.val base.val take.val :=
  WP.spec_imp_exists (ffn3_absmax_loop_spec x base take huabs hbound)

/-! ### Non-vacuity of §1 -/

/-- The reference maximum of `[3, -5, 2]` from base `0` over `take = 3` is `5`. -/
example : ffn3AbsMaxRef [(3#i64), (-5)#i64, (2#i64)] 0 3 = 5 := by decide

/-- A segment read at a base offset, `take = 2` from base `1`: `max |−5| |2| = 5`. -/
example : ffn3AbsMaxRef [(3#i64), (-5)#i64, (2#i64)] 1 2 = 5 := by decide

/-- Upper bound and attainment are jointly satisfiable at that segment. -/
example : (∀ l, l < 3 →
      (([(3#i64), (-5)#i64, (2#i64)].getD (0 + l) 0#i64).val.natAbs) ≤ 5) ∧
    (∃ l, l < 3 ∧ ffn3AbsMaxRef [(3#i64), (-5)#i64, (2#i64)] 0 3 =
      (([(3#i64), (-5)#i64, (2#i64)].getD (0 + l) 0#i64).val.natAbs)) :=
  ⟨fun l hl => ffn3AbsMaxRef_ub _ 0 3 l hl, by
    obtain ⟨l, hl, heq⟩ := ffn3AbsMaxRef_attained _ 0 3 (by norm_num)
    exact ⟨l, hl, heq⟩⟩

/-- The empty segment: the reference maximum is `0`. -/
example : ffn3AbsMaxRef [(3#i64), (-5)#i64] 0 0 = 0 := ffn3AbsMaxRef_zero _ _

/-! ## §2 · `kern.pow2.act_scale`: the least covering power of two

`src/engine/kern/pow2.rs:22`. The Rust body:

```
if absmax == 0 { Some 0 }
else {
  let need = absmax / 127 + u64::from(absmax % 127 != 0);   // ⌈absmax/127⌉
  let (mut s, mut steps) = (1u64, 0u32);
  while s < need && steps < 62 { s *= 2; steps += 1; }
  if s < need { None } else { Some s }
}
```

`ffn3Need absmax = ⌈absmax/127⌉` is the code unit the scale must cover. The loop doubles `s` from
`1 = 2^0` while it does not yet cover `need`, so the result is the least power of two at least
`need` — `ffn3ScaleLeast need k` is `need ≤ 2^k` and `k = 0` or `2^(k-1) < need`. A `u64` magnitude
gives `need ≤ 2^62`, so the `steps < 62` cap is never the exit and the body never returns `none`. -/

def ffn3Need (absmax : Nat) : Nat := ceilDivRef absmax 127

def ffn3ScaleLeast (need k : Nat) : Prop := need ≤ 2 ^ k ∧ (k = 0 ∨ 2 ^ (k - 1) < need)

theorem ffn3Need_eq (absmax : Nat) : ffn3Need absmax = ceilDivRef absmax 127 := rfl

lemma ffn3_div127_le (a : Nat) (h : a < 2 ^ 64) : a / 127 ≤ 2 ^ 58 := by
  have h1 : a / 127 ≤ a / 64 :=
    Nat.div_le_div_left (by norm_num : 64 ≤ 127) (by norm_num : 0 < 64)
  have h2 : a / 64 < 2 ^ 58 := by
    rw [Nat.div_lt_iff_lt_mul (by norm_num : 0 < 64)]
    calc a < 2 ^ 64 := h
      _ = 2 ^ 58 * 64 := by norm_num
  omega

theorem ffn3Need_le (absmax : Nat) (h : absmax < 2 ^ 64) : ffn3Need absmax ≤ 2 ^ 62 := by
  rw [ffn3Need, ceilDivRef, if_neg (by norm_num : (127 : Nat) ≠ 0)]
  have := ffn3_div127_le absmax h
  split <;> omega

lemma ffn3_pow2_le_u64max (n : Nat) (h : n ≤ 63) : (2 : Nat) ^ n ≤ U64.max := by
  rw [U64.max_eq]
  calc (2 : Nat) ^ n ≤ 2 ^ 63 := Nat.pow_le_pow_right (n := 2) (by norm_num) h
    _ ≤ 18446744073709551615 := by norm_num

lemma ffn3_fromU64Bool_val (b : Bool) :
    (core.convert.num.FromU64Bool.from b).val = (if b then 1 else 0) := by
  unfold core.convert.num.FromU64Bool.from
  cases b <;> rfl

lemma ffn3_u64_ne_zero_iff (x : U64) : (x != 0#u64) = true ↔ x.val ≠ 0 := by
  rw [bne_iff_ne]
  constructor
  · intro h hv; exact h (UScalar.eq_of_val_eq (by rw [hv]; rfl))
  · intro h he; exact h (by rw [UScalar.val_eq_of_eq he]; rfl)

theorem ffn3_act_scale_body_cont (need s : U64) (steps : U32)
    (hs : s.val < need.val) (hst : steps.val < 62) (hmul : s.val * (2#u64).val ≤ U64.max) :
    ∃ s' : U64, ∃ st' : U32,
      kern.pow2.act_scale_loop.body need s steps = ok (ControlFlow.cont (s', st')) ∧
      st'.val = steps.val + 1 ∧ s'.val = s.val * 2 := by
  unfold kern.pow2.act_scale_loop.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hs)]
  rw [if_pos (by rw [UScalar.lt_equiv]; show steps.val < 62; exact hst)]
  obtain ⟨s', hs', hs'v⟩ := WP.spec_imp_exists (U64.mul_spec (x := s) (y := 2#u64) hmul)
  rw [hs']; simp only [bind_tc_ok]
  have hadd : steps.val + (1#u32).val ≤ U32.max := by
    rw [show (1#u32).val = 1 from rfl, U32.max_eq]; omega
  obtain ⟨st', hst', hst'v⟩ := WP.spec_imp_exists (U32.add_spec (x := steps) (y := 1#u32) hadd)
  rw [hst']; simp only [bind_tc_ok]
  refine ⟨s', st', rfl, hst'v, ?_⟩
  rw [hs'v]; rfl

theorem ffn3_act_scale_body_done (need s : U64) (steps : U32) (hge : need.val ≤ s.val) :
    kern.pow2.act_scale_loop.body need s steps = ok (ControlFlow.done s) := by
  unfold kern.pow2.act_scale_loop.body
  rw [if_neg (by rw [UScalar.lt_equiv]; omega)]

theorem ffn3_act_scale_body_cap (need s : U64) (steps : U32)
    (hs : s.val < need.val) (hcap : 62 ≤ steps.val) :
    kern.pow2.act_scale_loop.body need s steps = ok (ControlFlow.done s) := by
  unfold kern.pow2.act_scale_loop.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hs)]
  rw [if_neg (by rw [UScalar.lt_equiv]; show ¬ (steps.val < 62); omega)]

theorem ffn3_act_scale_loop_spec (need : U64) (hneedle : need.val ≤ 2 ^ 62) :
    kern.pow2.act_scale_loop need 1#u64 0#u32 ⦃ out =>
      ∃ k : Nat, k ≤ 62 ∧ out.val = 2 ^ k ∧ ffn3ScaleLeast need.val k ⦄ := by
  unfold kern.pow2.act_scale_loop
  apply loop.spec_decr_nat
    (measure := fun st : U64 × U32 => 62 - st.2.val)
    (inv := fun st : U64 × U32 =>
      st.1.val = 2 ^ st.2.val ∧ st.2.val ≤ 62 ∧
      (st.2.val = 0 ∨ 2 ^ (st.2.val - 1) < need.val))
    (post := fun out : U64 => ∃ k : Nat, k ≤ 62 ∧ out.val = 2 ^ k ∧ ffn3ScaleLeast need.val k)
  · rintro ⟨s, steps⟩ ⟨hsv, hstle, hleast⟩
    by_cases hs : s.val < need.val
    · by_cases hst : steps.val < 62
      · have hmul : s.val * (2#u64).val ≤ U64.max := by
          rw [show (2#u64).val = 2 from rfl, hsv,
            show 2 ^ steps.val * 2 = 2 ^ (steps.val + 1) from by rw [pow_succ]]
          exact ffn3_pow2_le_u64max (steps.val + 1) (by omega)
        obtain ⟨s', st', hb, hst'v, hs'v⟩ := ffn3_act_scale_body_cont need s steps hs hst hmul
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        show (s'.val = 2 ^ st'.val ∧ st'.val ≤ 62 ∧
            (st'.val = 0 ∨ 2 ^ (st'.val - 1) < need.val)) ∧ 62 - st'.val < 62 - steps.val
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hs'v, hst'v, hsv, pow_succ]
        · rw [hst'v]; omega
        · right
          rw [hst'v, show steps.val + 1 - 1 = steps.val from by omega, ← hsv]
          exact hs
        · rw [hst'v]; omega
      · have hsteq : steps.val = 62 := le_antisymm hstle (Nat.le_of_not_lt hst)
        have hb := ffn3_act_scale_body_cap need s steps hs (by omega)
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        have hlt : 2 ^ 62 < need.val := by
          have hsq : s.val = 2 ^ 62 := by rw [hsv, hsteq]
          rw [← hsq]; exact hs
        exact absurd hlt (not_lt.mpr hneedle)
    · have hge : need.val ≤ s.val := by omega
      have hb := ffn3_act_scale_body_done need s steps hge
      rw [WP.spec_equiv_exists]
      refine ⟨_, hb, ?_⟩
      refine ⟨steps.val, hstle, hsv, ?_, hleast⟩
      rw [← hsv]; exact hge
  · refine ⟨?_, ?_, ?_⟩
    · show (1#u64).val = 2 ^ (0#u32).val
      rw [show (1#u64).val = 1 from rfl, show (0#u32).val = 0 from rfl]
      norm_num
    · show (0#u32).val ≤ 62
      rw [show (0#u32).val = 0 from rfl]; norm_num
    · exact Or.inl (show (0#u32).val = 0 from rfl)

theorem ffn3_act_scale_tail (need s : U64) (hge : need.val ≤ s.val) :
    (if s < need then (ok none : Result (Option U64)) else ok (some s)) = ok (some s) := by
  rw [if_neg (by rw [UScalar.lt_equiv]; omega)]

theorem word_act_scale_zero : kern.pow2.act_scale 0#u64 = ok (some 0#u64) := by
  unfold kern.pow2.act_scale
  rw [if_pos rfl]

theorem word_act_scale_ok (absmax : U64) :
    ∃ out : Option U64,
      kern.pow2.act_scale absmax = ok out ∧
      ((absmax.val = 0 ∧ out = some 0#u64) ∨
       (absmax.val ≠ 0 ∧ ∃ s : U64, ∃ k : Nat,
          out = some s ∧ k ≤ 62 ∧ s.val = 2 ^ k ∧
          ffn3ScaleLeast (ffn3Need absmax.val) k)) := by
  by_cases hz : absmax.val = 0
  · have hz' : absmax = 0#u64 := UScalar.eq_of_val_eq (by rw [hz]; rfl)
    refine ⟨some 0#u64, ?_, Or.inl ⟨hz, rfl⟩⟩
    unfold kern.pow2.act_scale
    rw [if_pos hz']
  · have hne : ¬ (absmax = 0#u64) := fun h => hz (by rw [UScalar.val_eq_of_eq h]; rfl)
    have h127 : (127#u64).val ≠ 0 := by decide
    obtain ⟨i, hi, hiv⟩ := UScalar.div_spec absmax (y := 127#u64) h127
    obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (UScalar.rem_spec absmax (y := 127#u64) h127)
    have h127v : (127#u64).val = 127 := rfl
    have hiv' : i.val = absmax.val / 127 := by rw [hiv, h127v]
    have hi1v' : i1.val = absmax.val % 127 := by rw [hi1v, h127v]
    have hi2v : (core.convert.num.FromU64Bool.from (i1 != 0#u64)).val
        = (if absmax.val % 127 = 0 then 0 else 1) := by
      rw [ffn3_fromU64Bool_val]
      have hbiff := ffn3_u64_ne_zero_iff i1
      by_cases hb : (i1 != 0#u64) = true
      · have hmod : absmax.val % 127 ≠ 0 := by rw [← hi1v']; exact hbiff.mp hb
        rw [if_pos hb, if_neg hmod]
      · have hz1 : i1.val = 0 := by
          by_contra hc; exact hb (hbiff.mpr hc)
        have hmod : absmax.val % 127 = 0 := by rw [← hi1v']; exact hz1
        rw [if_neg hb, if_pos hmod]
    have hile : i.val ≤ 2 ^ 58 := by
      rw [hiv']; exact ffn3_div127_le absmax.val (ffn3_u64_lt absmax)
    have hi2le : (core.convert.num.FromU64Bool.from (i1 != 0#u64)).val ≤ 1 := by
      rw [hi2v]; split <;> omega
    have hadd : i.val + (core.convert.num.FromU64Bool.from (i1 != 0#u64)).val ≤ U64.max := by
      rw [U64.max_eq]; omega
    obtain ⟨need, hneed, hneedv⟩ := WP.spec_imp_exists
      (U64.add_spec (x := i) (y := core.convert.num.FromU64Bool.from (i1 != 0#u64)) hadd)
    have hneedRef : need.val = ffn3Need absmax.val := by
      rw [hneedv, hiv', hi2v, ffn3Need, ceilDivRef, if_neg (by norm_num : (127 : Nat) ≠ 0)]
    have hneedle : need.val ≤ 2 ^ 62 := by
      rw [hneedRef]; exact ffn3Need_le absmax.val (ffn3_u64_lt absmax)
    obtain ⟨s, hs, hpost⟩ := WP.spec_imp_exists (ffn3_act_scale_loop_spec need hneedle)
    obtain ⟨k, hk, hsv, hleast⟩ := hpost
    have hsge : need.val ≤ s.val := by rw [hsv]; exact hleast.1
    have htail : (if s < need then (ok none : Result (Option U64)) else ok (some s))
        = ok (some s) := ffn3_act_scale_tail need s hsge
    have hleast' : ffn3ScaleLeast (ffn3Need absmax.val) k := by rw [← hneedRef]; exact hleast
    refine ⟨some s, ?_, Or.inr ⟨hz, s, k, rfl, hk, hsv, hleast'⟩⟩
    unfold kern.pow2.act_scale
    rw [if_neg hne, hi, hi1]
    simp only [lift, bind_tc_ok]
    rw [hneed]
    simp only [bind_tc_ok]
    rw [hs]
    simp only [bind_tc_ok]
    rw [htail]

theorem word_act_scale_total (absmax : U64) :
    ∃ out : Option U64, kern.pow2.act_scale absmax = ok out := by
  obtain ⟨out, h, _⟩ := word_act_scale_ok absmax
  exact ⟨out, h⟩

example : ffn3Need 127 = 1 ∧ ffn3ScaleLeast 1 0 := by
  refine ⟨?_, ?_⟩
  · decide
  · exact ⟨by norm_num, Or.inl rfl⟩

example : ffn3Need 128 = 2 ∧ ffn3ScaleLeast 2 1 := by
  refine ⟨?_, ?_⟩
  · decide
  · exact ⟨by norm_num, Or.inr (by norm_num)⟩

example : ¬ ffn3ScaleLeast 3 1 := by
  intro h; exact absurd h.1 (by norm_num)

example : ffn3ScaleLeast (ffn3Need 127) 0 := by
  rw [show ffn3Need 127 = 1 from by decide]
  exact ⟨by norm_num, Or.inl rfl⟩

example : kern.pow2.act_scale (0#u64) = ok (some 0#u64) := word_act_scale_zero

/-! ## §3 · `chart.quant_act`: one lane of the code loop

`src/engine/chart/mod.rs:287` (`chart.quant_act_loop0_loop1.body`) reads `x[base + lane]`, computes the
lane's code from the `chart.act_quot_rem` pair and the power-of-two scale `s`, pushes it and advances
the counter, or stops. The reference below states what one lane contributes, and the two stopping
arms of the extracted body are proved. **Residual — the turn arm and the composition.** The turn (the
`lane < take ∧ ok1` arm) is not proved: `unfold chart.quant_act_loop0_loop1.body` leaves the
pair-destructured inner result as a `match` on `(true, 0#i8)` that `rw`, `simp only` and `dsimp only`
do not reduce under `instances` transparency (`dsimp only` reports `made no progress` on a target it
also reports as not type-correct at that transparency), so the `Vec::push` and the counter increment
are not reachable by rewriting; and the nonzero block additionally consumes the opaque
`core.num.I64.unsigned_abs` and `core.ops.range.RangeInclusive.contains`. The verbatim unreduced
left-hand side is
`let (ok2, code) := (true, 0#i8); let codes1 ← codes.push code; let lane1 ← lane + 1#usize; ok (cont (codes1, ok2, lane1))`.
So the body-to-reference equality for the turn, the loop-level invariant over the growable `Vec`, and
`chart.QuantAct.dequantizes` stay open. -/

/-- The rounded magnitude one lane contributes: the quotient plus one when the doubled remainder
reaches the scale (round-half-up), as `Nat`. -/
def ffn3LaneMag (s q r : Nat) : Nat := q + if 2 * r ≥ s then 1 else 0

/-- The signed magnitude the lane carries: negated for a negative `i64` source. -/
def ffn3LaneSigned (value : ℤ) (s q r : Nat) : ℤ :=
  if value < 0 then -(ffn3LaneMag s q r : ℤ) else (ffn3LaneMag s q r : ℤ)

/-- The reference code one lane of `chart.quant_act_loop0_loop1` contributes, from the source value
and the `chart.act_quot_rem` pair: the zero scale short-circuits to `(true, 0)`; otherwise the signed
rounded magnitude, `(true, ·)` inside `[-127, 127]` and `(false, 0)` outside it. The extracted body
writes `IScalar.cast .I8 signed` on the in-range branch, whose `.val` is the `ℤ` component below
(the cast is exact on `[-127, 127]`); the body's in-range flag is the `Bool` component. -/
def ffn3LaneCodeRef (value : ℤ) (s q r : Nat) : Bool × ℤ :=
  if s = 0 then (true, 0)
  else
    (decide (-127 ≤ ffn3LaneSigned value s q r ∧ ffn3LaneSigned value s q r ≤ 127),
      if -127 ≤ ffn3LaneSigned value s q r ∧ ffn3LaneSigned value s q r ≤ 127
      then ffn3LaneSigned value s q r else 0)

/-- The reference at the zero scale is the code `0` with the flag set. -/
theorem ffn3LaneCodeRef_zero (value : ℤ) (q r : Nat) :
    ffn3LaneCodeRef value 0 q r = (true, 0) := rfl

/-- Every reference code lies in the `i8` magnitude range: the out-of-range branch writes `0` and
the in-range branch is bounded by `127`. -/
theorem ffn3LaneCodeRef_snd_natAbs_le (value : ℤ) (s q r : Nat) :
    (ffn3LaneCodeRef value s q r).2.natAbs ≤ 127 := by
  unfold ffn3LaneCodeRef
  by_cases hs : s = 0
  · rw [if_pos hs]; decide
  · rw [if_neg hs]
    by_cases hb : -127 ≤ ffn3LaneSigned value s q r ∧ ffn3LaneSigned value s q r ≤ 127
    · rw [if_pos hb]
      have h2 := hb.2
      omega
    · rw [if_neg hb]; simp

/-- The reference's in-range flag is set exactly on the in-range branch (nonzero scale). -/
theorem ffn3LaneCodeRef_fst_iff (value : ℤ) (s q r : Nat) (hs : s ≠ 0) :
    (ffn3LaneCodeRef value s q r).1 = true ↔
      -127 ≤ ffn3LaneSigned value s q r ∧ ffn3LaneSigned value s q r ≤ 127 := by
  unfold ffn3LaneCodeRef
  rw [if_neg hs]
  by_cases hb : -127 ≤ ffn3LaneSigned value s q r ∧ ffn3LaneSigned value s q r ≤ 127
  · rw [if_pos hb]; simp [hb]
  · rw [if_neg hb]; simp [hb]

/-- The extracted body of one lane at or past the stopping bound returns the accumulator unchanged. -/
theorem ffn3_lane_body_done_lane (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes : alloc.vec.Vec Std.I8)
    (ok1 : Bool) (lane : Std.Usize) (hge : ¬ lane.val < take.val) :
    chart.quant_act_loop0_loop1.body x base take s pow2 shift mask codes ok1 lane
      = ok (ControlFlow.done (codes, ok1)) := by
  unfold chart.quant_act_loop0_loop1.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hge)]

/-- The extracted body after a refusal stops and clears the flag. -/
theorem ffn3_lane_body_done_ok (x : Slice Std.I64) (base take : Std.Usize) (s : Std.U64)
    (pow2 : Bool) (shift : Std.U32) (mask : Std.U64) (codes : alloc.vec.Vec Std.I8)
    (lane : Std.Usize) (hlane : lane.val < take.val) :
    chart.quant_act_loop0_loop1.body x base take s pow2 shift mask codes false lane
      = ok (ControlFlow.done (codes, false)) := by
  unfold chart.quant_act_loop0_loop1.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane)]
  rfl

/-- Non-vacuity of §3: the reference rounds the doubled remainder up at the tie and negates for a
negative source, and the out-of-range arm writes `0` — both branches fire at concrete inputs. -/
example : ffn3LaneCodeRef (-300) 4 75 0 = (true, -75) ∧
    ffn3LaneCodeRef 300 1 300 0 = (false, 0) ∧
    ffn3LaneCodeRef 5 3 1 1 = (true, 1) := by
  refine ⟨?_, ?_, ?_⟩ <;> (unfold ffn3LaneCodeRef ffn3LaneSigned ffn3LaneMag; decide)

/-- Non-vacuity of §3: the reference's range bound and its flag characterisation hold at concrete
values, so neither statement is vacuous. -/
example : (ffn3LaneCodeRef (-300) 4 75 0).2.natAbs ≤ 127 ∧
    ((ffn3LaneCodeRef 300 1 300 0).1 = true ↔
      -127 ≤ ffn3LaneSigned 300 1 300 0 ∧ ffn3LaneSigned 300 1 300 0 ≤ 127) :=
  ⟨ffn3LaneCodeRef_snd_natAbs_le _ _ _ _, ffn3LaneCodeRef_fst_iff _ _ _ _ (by decide)⟩

/-! ## §4 · `chart.ffn.quantize`: the `ok_or` wrapper

`src/engine/chart/ffn.rs:236`. `quantize` delegates to the pooled dispatch
`chart.ffn.quant_act_pooled` (the block loop `chart.quant_act` of §1–§3 and its pool fan-out) and
lifts its `Option` into the `Result`: `Ok q` on `some q`, and the `Dequant FixedOverflow` refusal on
`none`. The pooled dispatch is the boundary here; its body is not restated, and the theorem consumes
its outcome as the hypothesis `h`. -/

theorem word_quantize_ok (x : Slice Std.I64) (o : Option chart.QuantAct)
    (h : chart.ffn.quant_act_pooled x = ok o) :
    (∃ q : chart.QuantAct,
        chart.ffn.quantize x = ok (core.result.Result.Ok q) ∧ o = some q) ∨
      (chart.ffn.quantize x = ok (core.result.Result.Err
          (matvec.MatvecError.Dequant error.EngineError.FixedOverflow)) ∧ o = none) := by
  unfold chart.ffn.quantize
  rw [h]
  simp only [bind_tc_ok]
  cases o with
  | none => right; exact ⟨by simp [core.option.Option.ok_or_none], rfl⟩
  | some q => left; exact ⟨q, by simp [core.option.Option.ok_or_some], rfl⟩

/-- Non-vacuity of §4: the wrapper's refusal arm fires at a concrete `o = none`, and the positive arm
is the `Ok q` of the same statement, so the two disjuncts are not vacuous. -/
example (x : Slice Std.I64) (h : chart.ffn.quant_act_pooled x = ok none) :
    chart.ffn.quantize x
        = ok (core.result.Result.Err
            (matvec.MatvecError.Dequant error.EngineError.FixedOverflow)) ∧
      chart.ffn.quant_act_pooled x = ok none := by
  refine ⟨?_, h⟩
  rcases word_quantize_ok x none h with ⟨q, _, hsome⟩ | ⟨hq, _⟩
  · cases hsome
  · exact hq

/-! ## §5 · `chart.ffn.swiglu_cols_carrier`: the geometry refusals

`src/engine/chart/ffn.rs:493`. `swiglu_cols_carrier` first requires the `up` and `out` column
buffers to have one entry per `gate` column; each mismatch returns
`Err FfnError.BadGeometry` with `out` unchanged, before any column is written. The two theorems
state the two refusal conditions; the positive path is the per-column loop over `mixer.Carrier`,
whose body is the boundary (`chart.ffn.swiglu_cols_carrier` pushes through
`mixer.carrier_from_env`, an environment read outside this refinement). -/

/-- `1` is a valid `usize` length bound, so a one-element list is a slice. -/
lemma ffn3_one_le_usize_max : (1 : Nat) ≤ Usize.max := by
  rcases Usize.bounds_eq with h | h <;> rw [h]
  · rw [U32.max_eq]; norm_num
  · rw [U64.max_eq]; norm_num

/-- `0` is a valid `usize` length bound, so the empty list is a slice. -/
lemma ffn3_zero_le_usize_max : (0 : Nat) ≤ Usize.max :=
  Nat.zero_le _

theorem word_swiglu_cols_refuses_up (gate up out : Slice (alloc.vec.Vec Std.I64))
    (carrier : mixer.Carrier) (h : up.val.length ≠ gate.val.length) :
    chart.ffn.swiglu_cols_carrier gate up out carrier
      = ok (core.result.Result.Err ffn.FfnError.BadGeometry, out) := by
  unfold chart.ffn.swiglu_cols_carrier
  rw [if_pos (by
    rw [bne_iff_ne]
    intro heq
    exact h (by
      have hv : (Slice.len up).val = (Slice.len gate).val := congrArg UScalar.val heq
      rw [Slice.len_val up, Slice.len_val gate] at hv
      exact hv))]

theorem word_swiglu_cols_refuses_out (gate up out : Slice (alloc.vec.Vec Std.I64))
    (carrier : mixer.Carrier) (hk : up.val.length = gate.val.length)
    (h : out.val.length ≠ gate.val.length) :
    chart.ffn.swiglu_cols_carrier gate up out carrier
      = ok (core.result.Result.Err ffn.FfnError.BadGeometry, out) := by
  unfold chart.ffn.swiglu_cols_carrier
  have hkn : (Slice.len up).val = (Slice.len gate).val := by
    rw [Slice.len_val up, Slice.len_val gate]; exact hk
  have hkeq : Slice.len up = Slice.len gate := UScalar.eq_of_val_eq hkn
  rw [if_neg (by rw [bne_iff_ne]; exact fun hne => hne hkeq)]
  rw [if_pos (by
    rw [bne_iff_ne]
    intro heq
    exact h (by
      have hv : (Slice.len out).val = (Slice.len gate).val := congrArg UScalar.val heq
      rw [Slice.len_val out, Slice.len_val gate] at hv
      exact hv))]

/-- Non-vacuity of §5: one `up` entry against an empty `gate` satisfies the refusal hypothesis
jointly with any `out` and `carrier`, and the refusal is the one the headline names. -/
example (v : alloc.vec.Vec Std.I64) (out : Slice (alloc.vec.Vec Std.I64))
    (carrier : mixer.Carrier) :
    chart.ffn.swiglu_cols_carrier
        (Slice.from ([] : List (alloc.vec.Vec Std.I64)) ffn3_zero_le_usize_max)
        (Slice.from ([v] : List (alloc.vec.Vec Std.I64)) ffn3_one_le_usize_max)
        out carrier
      = ok (core.result.Result.Err ffn.FfnError.BadGeometry, out) :=
  word_swiglu_cols_refuses_up
    (Slice.from ([] : List (alloc.vec.Vec Std.I64)) ffn3_zero_le_usize_max)
    (Slice.from ([v] : List (alloc.vec.Vec Std.I64)) ffn3_one_le_usize_max)
    out carrier (by simp)

/-! ## §6 · `chart.ffn.swiglu_one`: the value claim with the conversion discharged

`src/engine/chart/ffn.rs:246`. `Serve/Ffn2.lean`'s `word_swiglu_one_ok` proves the same value claim
but takes the `i128 → i64` step as the hypothesis `htry`. The Aeneas model defines
`I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from` (`target/aeneas-model/lean/Aeneas/Std/
Scalar/CoreConvertNum.lean:882`) as `if I64.min ≤ x.val ∧ x.val ≤ I64.max then ok (Ok ⟨ofInt 64
x.val⟩) else ok (Err ())`, so here that step is discharged from the model and the headline carries
no hypothesis about an unmodeled external. The remaining hypotheses are the `Int` bounds the body
would trap on: the `i128` product, its `+ 2^31` bias, and the `i64` range of `swigluRef`'s result. -/

/-- An `Int` in the `i64` range has `BitVec.ofInt 64` value equal to itself. -/
lemma ffn3_i64_ofInt_val (x : Int) (h : I64.min ≤ x ∧ x ≤ I64.max) :
    (⟨BitVec.ofInt 64 x⟩ : I64).val = x := by
  change (BitVec.ofInt 64 x).toInt = x
  apply BitVec.toInt_ofInt_eq_self (by norm_num : 0 < 64)
  · have h1 := h.1
    rw [I64.min_eq] at h1
    have : (-2 ^ (64 - 1) : Int) = -9223372036854775808 := by norm_num
    rw [this]; exact h1
  · have h2 := h.2
    rw [I64.max_eq] at h2
    have : (2 ^ (64 - 1) : Int) = 9223372036854775808 := by norm_num
    rw [this]; omega

theorem word_swiglu_one_ok_closed (gate up : I64) (s : I64)
    (hs : ffn.silu_fixed gate = ok (core.result.Result.Ok s))
    (hprodlo : I128.min ≤ s.val * up.val)
    (hprodhi : s.val * up.val ≤ I128.max)
    (hrlo : I128.min ≤ s.val * up.val + 2147483648)
    (hrhi : s.val * up.val + 2147483648 ≤ I128.max)
    (hlo : I64.min ≤ swigluRef s.val up.val) (hhi : swigluRef s.val up.val ≤ I64.max) :
    ∃ out : I64, chart.ffn.swiglu_one gate up = ok (core.result.Result.Ok out) ∧
      out.val = swigluRef s.val up.val := by
  unfold chart.ffn.swiglu_one
  rw [hs]
  simp only [bind_tc_ok]
  have hb1 : core.result.Result.Insts.CoreOpsTry.branch (T := I64) (E := ffn.FfnError)
        (core.result.Result.Ok s)
      = ok (core.ops.control_flow.ControlFlow.Continue
              (B := core.result.Result core.convert.Infallible ffn.FfnError) s) := rfl
  rw [hb1]
  simp only [bind_tc_ok, lift]
  obtain ⟨p, i2, hp, hi2, hi2v⟩ :=
    word_swiglu_one_round gate up s hs hprodlo hprodhi hrlo hrhi
  rw [hp]
  simp only [core.option.Option.ok_or_some, bind_tc_ok]
  have hb2 : core.result.Result.Insts.CoreOpsTry.branch (T := I128) (E := ffn.FfnError)
        (core.result.Result.Ok p)
      = ok (core.ops.control_flow.ControlFlow.Continue
              (B := core.result.Result core.convert.Infallible ffn.FfnError) p) := rfl
  rw [hb2]
  simp only [bind_tc_ok]
  rw [hi2]
  simp only [bind_tc_ok]
  have hbnd : I64.min ≤ i2.val ∧ i2.val ≤ I64.max := by rw [hi2v]; exact ⟨hlo, hhi⟩
  have htf : I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from i2
      = ok (core.result.Result.Ok (⟨BitVec.ofInt 64 i2.val⟩ : I64)) := by
    unfold I64.Insts.CoreConvertTryFromI128TryFromIntError.try_from
    rw [if_pos hbnd]
  rw [htf]
  simp only [bind_tc_ok, core.result.Result.map_err_ok]
  refine ⟨⟨BitVec.ofInt 64 i2.val⟩, rfl, ?_⟩
  rw [ffn3_i64_ofInt_val i2.val hbnd, hi2v]

/-- Non-vacuity of §6: the `i64`-range hypothesis is satisfiable at a concrete `i128` value, and the
converted `i64` then carries that value — the equality the headline's conclusion names. -/
example : (⟨BitVec.ofInt 64 4611686018427387904⟩ : I64).val = 4611686018427387904 ∧
    I64.min ≤ (4611686018427387904 : ℤ) ∧ (4611686018427387904 : ℤ) ≤ I64.max := by
  have hb : I64.min ≤ (4611686018427387904 : ℤ) ∧ (4611686018427387904 : ℤ) ≤ I64.max := by
    rw [I64.min_eq, I64.max_eq]; norm_num
  exact ⟨ffn3_i64_ofInt_val 4611686018427387904 hb, hb.1, hb.2⟩

/-- Non-vacuity of §6: the `swigluRef` range hypothesis is satisfiable — the reference value at a
concrete input lies in the `i64` range — so the headline's bound hypotheses are not vacuous. -/
example : I64.min ≤ swigluRef (2147483649 : ℤ) 1 ∧ swigluRef (2147483649 : ℤ) 1 ≤ I64.max := by
  rw [I64.min_eq, I64.max_eq]
  rw [show swigluRef (2147483649 : ℤ) 1 = 1 from by unfold swigluRef roundShift32; decide]
  norm_num

/-! ## §7 · Axiom cones

The printed set of a headline is the model's closed cone `{propext, Classical.choice, Quot.sound}`
plus the *extracted bodies' opaque externals reachable from the statement's own terms* — the four
`proof/regen.sh` externals and the pool boundary constants. No proof adds an axiom of its own: the
headlines whose statements are over plain words and reference definitions
(`word_act_scale_ok/_zero/_total`, `ffn3LaneCodeRef_snd_natAbs_le`, `ffn3LaneCodeRef_fst_iff`,
`ffn3_i64_ofInt_val`, `word_swiglu_one_ok_closed`) print exactly the three, and the boundary-relative
headlines print the three plus the externals their statements name. The opaque externals
(`core.num.I64.unsigned_abs`, `core.num.U64.is_power_of_two`, `core.num.U64.trailing_zeros`,
`core.ops.range.RangeInclusive.contains`) are consumed as hypotheses. -/

#print axioms word_quant_absmax_ok
#print axioms ffn3_absmax_loop_spec
#print axioms word_act_scale_ok
#print axioms word_act_scale_zero
#print axioms word_act_scale_total
#print axioms ffn3LaneCodeRef_snd_natAbs_le
#print axioms ffn3LaneCodeRef_fst_iff
#print axioms ffn3_lane_body_done_lane
#print axioms ffn3_lane_body_done_ok
#print axioms word_quantize_ok
#print axioms word_swiglu_cols_refuses_up
#print axioms word_swiglu_cols_refuses_out
#print axioms ffn3_i64_ofInt_val
#print axioms word_swiglu_one_ok_closed

end apeiron_word
