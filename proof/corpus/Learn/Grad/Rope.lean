import ApeironWord
import Numeric.Rope.Rotate

/-!
# The extracted RoPE forward and backward passes

`learn_port.attn.rope_fwd` (`src/learn/grad/attn.rs`, lines 166-175) rotates one token's vector at
position `pos` through the engine's `rope.RopeTable.rotate_one`; `learn_port.attn.rope_bwd` (lines
190-221) is its transpose, the elementwise loop `rope_elem_bwd` (lines 222-260) over the flat
output. This module refines both against a closed form over `ℤ`.

The reference is a function of the entry values alone, on the success domain only:

* `ropeBwdPair` — the transpose kernel of one rotation pair, with the sign selected by the pair half
  and the engine's round-half-up epilogue `roundShift32 v = (v + 2^31) >>> 32`;
* `ropeBwdElem` — flat component `i`: the turn `turns[(i % hd) % (hd/2)]`, partner `i + hd/2` below
  the half and `i - hd/2` at or above it;
* `ropeBwdPrefix` — the partial output buffer after `k` components, against which `rope_bwd_loop` is
  proved by `loop.spec_decr_nat` with an explicit invariant;
* `RopeOneDomain` — the static domain of both entry points: nonempty input, `head_dim ≥ 2`,
  `head_dim` even, `head_dim ∣ x.len` and `head_dim / 2 ≤ turns.len`;
* `RopeFwdSpec` — `rope_fwd` returns the copy whose content is `rotHeadsState` at
  `heads = x.len / head_dim`; `RopeBwdSpec` — `rope_bwd` returns the buffer denoting
  `ropeBwdPrefix dy hd d2 scAt dy.len`.

Sections: §1 the specification; §2 the accumulator epilogue and one component; §3 the backward loop
and its wrapper; §4 the `rotate_one` loop and `rope_fwd`; §5 witnesses.

Reused, not restated: `rotHeadsState`, `headWin`, `RotHeadFitsL`, `rotate_head_content`,
`rotate_window_arith`, `RopeTrigAt`, `ropePairInt`, `roundShift32`, `value_round_shift_32_spec`,
`turn_to_radians_refines`, `turn_phase_val_eq`, `rotPrefix_*`, `drop_setSlice!_eq`, `slice_eq_headWin`.

Scope: integer content and return shape of the two passes on the success domain, at the trig-data
hypotheses the callers must supply. rustc/LLVM codegen, the kernel ABI, the `unsafe` container and
the refusal arms of the two public wrappers outside `RopeOneDomain` are outside this module.
-/


namespace apeiron_rope

open apeiron_word
open Aeneas Aeneas.Std Result

noncomputable section

/-! ## 1. Specification over `ℤ`

The two kernels the extraction computes: the per-component transpose of one rotation pair, and the
flat component selecting the pair and its turn. `roundShift32` is the engine's
`(v + 2^31) >>> 32` (round half towards `+∞`), reused from `Numeric.EngineRopeRefinement`. -/

/-- The transpose kernel of one rotation pair with `y0 = dy[0]`, `y1 = dy[1]` and Q32.32 trig
`(c, s)`: `roundShift32 (y0·c + y1·s)` on the low half of the rotation, `roundShift32 (y0·c − y1·s)`
on the high half. -/
def ropeBwdPair (y0 y1 c s : ℤ) (lo : Bool) : ℤ :=
  if lo then roundShift32 (y0 * c + y1 * s) else roundShift32 (y0 * c - y1 * s)

/-- The flat component `i` of the `RoPE` transpose: `r = i % hd` selects the rotation half and the
turn `turns[r % (hd/2)]`, the partner is `i + hd/2` on the low half and `i − hd/2` on the high half.
-/
def ropeBwdElem (dy : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (i : ℕ) : ℤ :=
  ropeBwdPair (dy.getD i default).val
    (dy.getD (if i % hd < d2 then i + d2 else i - d2) default).val
    (scAt (i % hd % d2)).cos.val (scAt (i % hd % d2)).sin.val (i % hd < d2)

/-- The flat component written out through its definitional unfolding. -/
theorem ropeBwdElem_eq (dy : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (i : ℕ) :
    ropeBwdElem dy hd d2 scAt i =
      ropeBwdPair (dy.getD i default).val
        (dy.getD (if i % hd < d2 then i + d2 else i - d2) default).val
        (scAt (i % hd % d2)).cos.val (scAt (i % hd % d2)).sin.val (i % hd < d2) := rfl

/-- The partial output buffer of the backward loop after `k` iterations: indices below `k` hold the
transposed values, every other index stays `0` (the value the allocation is filled with). -/
def ropeBwdPrefix (dy : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (k : ℕ) : List ℤ :=
  (List.range dy.length).map fun i => if i < k then ropeBwdElem dy hd d2 scAt i else 0

theorem ropeBwdPrefix_length (dy : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (k : ℕ) :
    (ropeBwdPrefix dy hd d2 scAt k).length = dy.length := by
  simp [ropeBwdPrefix]

/-- The initial buffer is the zero fill. -/
theorem ropeBwdPrefix_zero (dy : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) :
    ropeBwdPrefix dy hd d2 scAt 0 = List.replicate dy.length 0 := by
  simp [ropeBwdPrefix]

/-- One iteration: writing component `i` over the partial buffer advances the prefix. -/
theorem ropeBwdPrefix_succ (dy : List I64) (hd d2 : ℕ) (scAt : ℕ → fixed_trig.SinCos) (i : ℕ) :
    (ropeBwdPrefix dy hd d2 scAt i).set i (ropeBwdElem dy hd d2 scAt i) =
      ropeBwdPrefix dy hd d2 scAt (i + 1) := by
  apply List.ext_getElem
  · simp [ropeBwdPrefix, List.length_set]
  · intro k h1 h2
    have hk : k < dy.length := by simpa [ropeBwdPrefix] using h2
    simp only [ropeBwdPrefix, List.getElem_map, List.getElem_range, List.getElem_set]
    by_cases hki : i = k
    · rw [if_pos hki, if_pos (by omega), hki]
    · rw [if_neg hki]
      by_cases hk1 : k < i
      · rw [if_pos hk1, if_pos (by omega)]
      · rw [if_neg hk1, if_neg (by omega)]

/-- At turn `t` and position `pos`: the checked angle `toRad (lift (phase t pos))`, its successful
`fixed_sin_cos` pair `sc`, and the `2^33` magnitude bound the `i128` products rely on. -/
def RopeTrigPair (t : U64) (pos : U32) (sc : fixed_trig.SinCos) : Prop :=
  ∃ angle : I64,
    angle.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift (Numeric.RopeTurns.phase t.val pos.val)) ∧
    fixed_trig.fixed_sin_cos angle = ok (core.result.Result.Ok sc) ∧
    |sc.cos.val| ≤ 8589934592 ∧ |sc.sin.val| ≤ 8589934592

/-- `RopeTrigPair` read at a slice: the turn stored at index `j` together with its trig pair. -/
def RopeTrigAtS (v : Slice U64) (pos : U32) (j : ℕ) (t : U64) (sc : fixed_trig.SinCos) : Prop :=
  v.val[j]? = some t ∧ RopeTrigPair t pos sc

/-- The static table domain of the single-vector rotation `RopeTable::rotate_one`, read off
`src/engine/rope.rs`: a nonempty vector, an even `head_dim ≥ 2` and a whole number of heads, with
the `head_dim/2` turn entries the table constructor demands. -/
def RopeOneDomain (table : rope.RopeTable) (x : Slice I64) : Prop :=
  0 < x.val.length ∧ 2 ≤ table.head_dim.val ∧ table.head_dim.val % 2 = 0 ∧
  x.val.length % table.head_dim.val = 0 ∧
  table.head_dim.val / 2 ≤ table.turns.val.length

/-- **Forward specification.** On the success domain, `rope_fwd` returns the flat `rotHeadsState`
content: the entry `j < hd/2` of head window `h` is `roundShift32 (x0·cos − x1·sin)` and the entry
`j + hd/2` is `roundShift32 (x1·cos + x0·sin)`, `(x0, x1)` the original pair of that window, the trig
pair of the turn at rotary index `j`. -/
def RopeFwdSpec : Prop :=
  ∀ (table : rope.RopeTable) (x : Slice I64) (pos : U32) (d2 : Usize)
    (scAt : ℕ → fixed_trig.SinCos),
    RopeOneDomain table x →
    table.head_dim / 2#usize = ok d2 →
    (∀ i, i < d2.val → ∃ t, RopeTrigAt table.turns pos i t (scAt i)) →
    (∀ h, h < x.val.length / table.head_dim.val →
      RotHeadFitsL (headWin x.val table.head_dim.val h) d2.val scAt) →
    ∃ y' : alloc.vec.Vec I64,
      learn_port.attn.rope_fwd x pos table = ok (core.result.Result.Ok y') ∧
      y'.val.map (fun a => a.val) =
        rotHeadsState x.val table.head_dim.val d2.val scAt
          (x.val.length / table.head_dim.val)

/-- **Backward specification.** On the success domain, `rope_bwd` returns the elementwise transpose
`ropeBwdPrefix` at the full length; each written component fits `i64` (the explicit no-overflow
bound), and the refusals below the static guards are excluded. -/
def RopeBwdSpec : Prop :=
  ∀ (table : rope.RopeTable) (dy : Slice I64) (pos : U32) (d2 : Usize)
    (scAt : ℕ → fixed_trig.SinCos),
    RopeOneDomain table dy →
    table.head_dim / 2#usize = ok d2 →
    (∀ j, j < d2.val → ∃ t, RopeTrigAtS table.turns.deref pos j t (scAt j)) →
    (∀ k, k < dy.val.length →
      I64.min ≤ ropeBwdElem dy.val table.head_dim.val d2.val scAt k ∧
      ropeBwdElem dy.val table.head_dim.val d2.val scAt k ≤ I64.max) →
    ∃ dx : alloc.vec.Vec I64,
      learn_port.attn.rope_bwd dy pos table = ok (core.result.Result.Ok dx) ∧
      dx.val.map (fun a => a.val) =
        ropeBwdPrefix dy.val table.head_dim.val d2.val scAt dy.val.length

/-! ## 2. The accumulator epilogue and one transpose component -/

/-- The extracted `value.round_shift_32` and `learn_port.attn.round32` are the same body:
`(v + 2^31) >>> 32` on `i128`. -/
theorem round32_eq_round_shift_32 (v : I128) :
    learn_port.attn.round32 v = value.round_shift_32 v := rfl

private theorem rp_cast_val {src : IScalarTy} (tgt : IScalarTy) (x : IScalar src)
    (h : IScalar.min tgt ≤ x.val ∧ x.val ≤ IScalar.max tgt) :
    (IScalar.cast tgt x).val = x.val := by
  have hs := IScalar.cast_inBounds_spec tgt x h
  simpa [lift] using hs

private theorem rp_max : core.num.I64.MAX.val = I64.max := by
  simp [core.num.I64.MAX, IScalar.ofInt, I64.rMax, I64.max, I64.numBits]

private theorem rp_min : core.num.I64.MIN.val = I64.min := by
  simp [core.num.I64.MIN, IScalar.ofInt, I64.rMin, I64.min, I64.numBits]

private theorem rp_from_max :
    (core.convert.num.FromI128I64.from core.num.I64.MAX).val = I64.max := by
  rw [core.convert.num.FromI128I64.from_val_eq]; exact rp_max

private theorem rp_from_min :
    (core.convert.num.FromI128I64.from core.num.I64.MIN).val = I64.min := by
  rw [core.convert.num.FromI128I64.from_val_eq]; exact rp_min

private theorem rp_i64_range (z : ℤ) (h1 : I64.min ≤ z) (h2 : z ≤ I64.max) :
    IScalar.min IScalarTy.I64 ≤ z ∧ z ≤ IScalar.max IScalarTy.I64 := by
  simp only [I64.min, I64.max, IScalar.min, IScalar.max, I64.numBits, IScalarTy.numBits] at *
  exact ⟨h1, h2⟩

/-- **Accumulator epilogue, success arm.** When the half-add fits `i128` and the rounded accumulator
fits `i64`, `acc_round` returns that value. -/
theorem acc_round_ok (acc rounded : I128)
    (hround : value.round_shift_32 acc = ok rounded)
    (hfit : I64.min ≤ rounded.val ∧ rounded.val ≤ I64.max) :
    ∃ v : I64, learn_port.attn.acc_round acc = ok (core.result.Result.Ok v) ∧
      v.val = rounded.val := by
  have hup : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
    rw [gt_iff_lt, IScalar.lt_equiv, rp_from_max]
    exact not_lt.mpr hfit.2
  have hlo : ¬ (rounded < core.convert.num.FromI128I64.from core.num.I64.MIN) := by
    rw [IScalar.lt_equiv, rp_from_min]
    exact not_lt.mpr hfit.1
  refine ⟨IScalar.cast .I64 rounded, ?_, ?_⟩
  · unfold learn_port.attn.acc_round
    rw [round32_eq_round_shift_32, hround]
    simp only [lift, bind_tc_ok]
    rw [if_neg hup, if_neg hlo]
  · exact rp_cast_val .I64 rounded (rp_i64_range rounded.val hfit.1 hfit.2)

/-- **Accumulator epilogue, refusal arm.** A rounded accumulator outside `i64` returns
`Err Overflow` exactly. -/
theorem acc_round_refuses (acc rounded : I128)
    (hround : value.round_shift_32 acc = ok rounded)
    (hbad : rounded.val < I64.min ∨ I64.max < rounded.val) :
    learn_port.attn.acc_round acc =
      ok (core.result.Result.Err learn_port.attn.AttnError.Overflow) := by
  have hminmax : I64.min ≤ I64.max := by simp [I64.min, I64.max, I64.numBits]
  unfold learn_port.attn.acc_round
  rw [round32_eq_round_shift_32, hround]
  simp only [lift, bind_tc_ok]
  rcases hbad with h | h
  · have hno : ¬ (rounded > core.convert.num.FromI128I64.from core.num.I64.MAX) := by
      rw [gt_iff_lt, IScalar.lt_equiv, rp_from_max]
      exact not_lt.mpr (le_trans (le_of_lt h) hminmax)
    rw [if_neg hno]
    rw [if_pos (by rw [IScalar.lt_equiv, rp_from_min]; exact h)]
  · rw [if_pos (by rw [gt_iff_lt, IScalar.lt_equiv, rp_from_max]; exact h)]

/-! ## 3. One transpose component -/

private theorem slice_index_eq_u64 (x : Slice U64) (k : Usize) (h : k.val < x.val.length) :
    Slice.index_usize x k = ok (x.val.getD k.val default) := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

private theorem slice_index_eq (x : Slice I64) (k : Usize) (h : k.val < x.val.length) :
    Slice.index_usize x k = ok (x.val.getD k.val default) := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

private theorem rp_prod (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    |(a.val : ℤ) * (b.val : ℤ)| ≤ 2 ^ 96 := by
  have ha : |(a.val : ℤ)| ≤ 2 ^ 63 := by
    have h1 : -(2 : ℤ) ^ 63 ≤ a.val := by simpa [I64.numBits] using a.hmin
    have h2 : a.val < (2 : ℤ) ^ 63 := by simpa [I64.numBits] using a.hmax
    rw [abs_le]
    exact ⟨h1, by omega⟩
  rw [abs_mul]
  calc |(a.val : ℤ)| * |(b.val : ℤ)| ≤ (2 : ℤ) ^ 63 * 8589934592 :=
        mul_le_mul ha hb (by positivity) (by positivity)
    _ = 2 ^ 96 := by norm_num

private theorem rp_prod_max (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    IScalar.min IScalarTy.I128 ≤
        ((core.convert.num.FromI128I64.from a).val : ℤ) *
          ((core.convert.num.FromI128I64.from b).val : ℤ) ∧
      ((core.convert.num.FromI128I64.from a).val : ℤ) *
          ((core.convert.num.FromI128I64.from b).val : ℤ) ≤ IScalar.max IScalarTy.I128 := by
  rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  rw [IScalar.min_IScalarTy_I128_eq, IScalar.max_IScalarTy_I128_eq]
  have h := abs_le.mp (rp_prod a b hb)
  have hmax128 : (2 : ℤ) ^ 96 ≤ I128.max := by norm_num [I128.max, I128.numBits]
  have hmin128 : I128.min ≤ -((2 : ℤ) ^ 96) := by norm_num [I128.min, I128.numBits]
  exact ⟨le_trans hmin128 h.1, le_trans h.2 hmax128⟩

private theorem rp_sum_bounds (p q : ℤ) (hp : |p| ≤ 2 ^ 96) (hq : |q| ≤ 2 ^ 96) :
    IScalar.min IScalarTy.I128 ≤ p + q ∧ p + q ≤ IScalar.max IScalarTy.I128 ∧
      IScalar.min IScalarTy.I128 ≤ p - q ∧ p - q ≤ IScalar.max IScalarTy.I128 := by
  rw [IScalar.min_IScalarTy_I128_eq, IScalar.max_IScalarTy_I128_eq]
  rw [abs_le] at hp hq
  have hmax : (2 : ℤ) ^ 97 ≤ I128.max := by norm_num [I128.max, I128.numBits]
  have hmin : I128.min ≤ -((2 : ℤ) ^ 97) := by norm_num [I128.min, I128.numBits]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> linarith [hp.1, hp.2, hq.1, hq.2, hmin, hmax]

private theorem rp_plus_half_bounds (z : ℤ) (hz : |z| ≤ 2 ^ 97) :
    I128.min ≤ z + 2147483648 ∧ z + 2147483648 ≤ I128.max := by
  rw [abs_le] at hz
  have hmax : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max := by norm_num [I128.max, I128.numBits]
  have hmin : I128.min ≤ -((2 : ℤ) ^ 97) + 2147483648 := by norm_num [I128.min, I128.numBits]
  exact ⟨by linarith [hz.1, hmin], by linarith [hz.2, hmax]⟩

private theorem hcast64 : core.convert.num.FromU128U64.from core.num.U64.MAX =
    UScalar.cast .U128 core.num.U64.MAX := rfl

private theorem hcastt (x : U64) : core.convert.num.FromU128U64.from x = UScalar.cast .U128 x := rfl

private theorem hcastp (x : U32) : core.convert.num.FromU128U32.from x = UScalar.cast .U128 x := rfl

/-- The extracted `fixed_sin_cos` call of `rope_elem_bwd`: the stored angle `angle` of the turn's
pair is the value the body computes, so the body's `match` takes the `Ok` arm with `sc`. -/
private theorem elem_trig_arm (turn : U64) (pos : U32) (angle : I64) (sc : fixed_trig.SinCos)
    (hangle : angle.val =
      Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift (Numeric.RopeTurns.phase turn.val pos.val)))
    (hsc : fixed_trig.fixed_sin_cos angle = ok (core.result.Result.Ok sc)) :
    (do let i1 ← lift (core.convert.num.FromU128U64.from turn)
        let i2 ← lift (core.convert.num.FromU128U32.from pos)
        let i3 ← i1 * i2
        let i4 ← lift (core.convert.num.FromU128U64.from core.num.U64.MAX)
        let i5 ← lift (i3 &&& i4)
        let phase ← lift (UScalar.cast .U64 i5)
        let i6 ← fixed_trig.turn_to_radians phase
        fixed_trig.fixed_sin_cos i6 : Result (core.result.Result fixed_trig.SinCos
          fixed_trig.TrigError)) = ok (core.result.Result.Ok sc) := by
  rw [hcastt turn, hcastp pos, hcast64]
  have h : (do
        let i1 ← lift (UScalar.cast .U128 turn)
        let i2 ← lift (UScalar.cast .U128 pos)
        let i3 ← i1 * i2
        let i4 ← lift (UScalar.cast .U128 core.num.U64.MAX)
        let i5 ← lift (i3 &&& i4)
        let phase ← lift (UScalar.cast .U64 i5)
        let i6 ← fixed_trig.turn_to_radians phase
        fixed_trig.fixed_sin_cos i6 : Result (core.result.Result fixed_trig.SinCos
          fixed_trig.TrigError)) ⦃ out => out = core.result.Result.Ok sc ⦄ := by
    step*
    have hi3v : i3.val = turn.val * pos.val := by
      simp only [i3_post, i1_post, i2_post, u128_cast_u64_val, u128_cast_u32_val]
    have hphasev : phase.val = Numeric.RopeTurns.phase turn.val pos.val := by
      have h5 : i5 = i3 &&& UScalar.cast .U128 core.num.U64.MAX := by
        apply UScalar.eq_of_val_eq
        simp only [i5_post, i4_post]
      rw [phase_post, h5]
      exact turn_phase_val_eq turn pos i3 hi3v
    have hi6v : i6 = angle := by
      apply IScalar.eq_of_val_eq
      rw [i6_post, hphasev, hangle]
    rw [hi6v, hsc]
    simp only [WP.spec_ok]
  obtain ⟨v, hv, hpv⟩ := WP.spec_imp_exists h
  rw [hv, hpv]

/-- **One transpose component, low half.** When the pair index `r = i % hd` is below `hd/2`, the
extracted `rope_elem_bwd` returns `Ok` of `roundShift32 (dy[i]·cos + dy[i + hd/2]·sin)`. -/
theorem rope_elem_bwd_spec_lo (dy : Slice I64) (i : Usize) (hd : Usize) (turn : U64) (pos : U32)
    (d2 r partner : Usize) (sc : fixed_trig.SinCos)
    (hd2 : hd / 2#usize = ok d2) (hrem : i % hd = ok r) (hrlt : r.val < d2.val)
    (hadd : i + d2 = ok partner) (_haddv : partner.val = i.val + d2.val)
    (htrig : RopeTrigPair turn pos sc)
    (hi : i.val < dy.val.length) (hp : partner.val < dy.val.length)
    (hfit : I64.min ≤ roundShift32 ((dy.val.getD i.val default).val * sc.cos.val +
              (dy.val.getD partner.val default).val * sc.sin.val) ∧
            roundShift32 ((dy.val.getD i.val default).val * sc.cos.val +
              (dy.val.getD partner.val default).val * sc.sin.val) ≤ I64.max) :
    ∃ v : I64, learn_port.attn.rope_elem_bwd dy i hd turn pos = ok (core.result.Result.Ok v) ∧
      v.val = roundShift32 ((dy.val.getD i.val default).val * sc.cos.val +
              (dy.val.getD partner.val default).val * sc.sin.val) := by
  obtain ⟨angle, hangle, hsc, hcosb, hsinb⟩ := htrig
  have hlo : r < d2 := (UScalar.lt_equiv r d2).mpr hrlt
  have hspec : learn_port.attn.rope_elem_bwd dy i hd turn pos ⦃ out =>
      ∃ v : I64, out = core.result.Result.Ok v ∧
        v.val = roundShift32 ((dy.val.getD i.val default).val * sc.cos.val +
              (dy.val.getD partner.val default).val * sc.sin.val) ⦄ := by
    unfold learn_port.attn.rope_elem_bwd
    rw [hcastt turn, hcastp pos, hcast64]
    rw [hd2]
    rw [hrem]
    conv => arg 1; simp only [bind_tc_ok]
    rw [if_pos hlo]
    rw [hadd]
    conv => arg 1; simp only [bind_tc_ok]
    step*
    have hi3v : i3.val = turn.val * pos.val := by
      simp only [i3_post, i1_post, i2_post, u128_cast_u64_val, u128_cast_u32_val]
    have hphasev : phase.val = Numeric.RopeTurns.phase turn.val pos.val := by
      have h5 : i5 = i3 &&& UScalar.cast .U128 core.num.U64.MAX := by
        apply UScalar.eq_of_val_eq
        simp only [i5_post, i4_post]
      rw [phase_post, h5]
      exact turn_phase_val_eq turn pos i3 hi3v
    have hi6v : i6 = angle := by
      apply IScalar.eq_of_val_eq
      rw [i6_post, hphasev, hangle]
    rw [hi6v, hsc]
    conv => arg 1; simp only [bind_tc_ok]
    simp only [if_pos hlo]
    rw [slice_index_eq dy i hi, slice_index_eq dy partner hp]
    conv => arg 1; simp only [lift, bind_tc_ok]
    obtain ⟨p, hp_eq, hpv⟩ := WP.spec_imp_exists
      (IScalar.mul_spec (x := core.convert.num.FromI128I64.from (dy.val.getD i.val default))
        (y := core.convert.num.FromI128I64.from sc.cos) (rp_prod_max _ _ hcosb).1
          (rp_prod_max _ _ hcosb).2)
    rw [hp_eq]
    conv => arg 1; simp only [bind_tc_ok]
    obtain ⟨q, hq_eq, hqv⟩ := WP.spec_imp_exists
      (IScalar.mul_spec (x := core.convert.num.FromI128I64.from (dy.val.getD partner.val default))
        (y := core.convert.num.FromI128I64.from sc.sin) (rp_prod_max _ _ hsinb).1
          (rp_prod_max _ _ hsinb).2)
    rw [hq_eq]
    conv => arg 1; simp only [bind_tc_ok]
    have hpv' : p.val = (dy.val.getD i.val default).val * sc.cos.val := by
      rw [hpv, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
    have hqv' : q.val = (dy.val.getD partner.val default).val * sc.sin.val := by
      rw [hqv, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
    have hpb : |p.val| ≤ 2 ^ 96 := by
      rw [hpv']
      exact rp_prod (dy.val.getD i.val default) sc.cos hcosb
    have hqb : |q.val| ≤ 2 ^ 96 := by
      rw [hqv']
      exact rp_prod (dy.val.getD partner.val default) sc.sin hsinb
    obtain ⟨sm, hsm_eq, hsmv⟩ := WP.spec_imp_exists
      (IScalar.add_spec (x := p) (y := q) (rp_sum_bounds p.val q.val hpb hqb).1
        (rp_sum_bounds p.val q.val hpb hqb).2.1)
    rw [hsm_eq]
    conv => arg 1; simp only [bind_tc_ok]
    have hsmv' : sm.val = (dy.val.getD i.val default).val * sc.cos.val +
        (dy.val.getD partner.val default).val * sc.sin.val := by
      rw [hsmv, hpv', hqv']
    have hsmb : |sm.val| ≤ 2 ^ 97 := by
      rw [hsmv', abs_le]
      obtain ⟨hp1, hp2⟩ := abs_le.mp hpb
      obtain ⟨hq1, hq2⟩ := abs_le.mp hqb
      constructor <;> linarith
    obtain ⟨rounded, hround, hroundv⟩ := WP.spec_imp_exists
      (value_round_shift_32_spec sm (rp_plus_half_bounds sm.val hsmb).1
        (rp_plus_half_bounds sm.val hsmb).2)
    have hfit' : I64.min ≤ rounded.val ∧ rounded.val ≤ I64.max := by
      rw [hroundv, hsmv']; exact hfit
    obtain ⟨v, hv, hvv⟩ := acc_round_ok sm rounded hround hfit'
    rw [hv]
    conv => arg 1; simp only [bind_tc_ok]
    simp only [WP.spec_ok]
    exact ⟨v, rfl, by rw [hvv, hroundv, hsmv']⟩
  obtain ⟨out, hout, v, hv, hvval⟩ := WP.spec_imp_exists hspec
  exact ⟨v, by rw [hout, hv], hvval⟩

/-- **One transpose component, high half.** When the pair index `r = i % hd` is at least `hd/2`,
the extracted `rope_elem_bwd` returns `Ok` of `roundShift32 (dy[i]·cos − dy[i − hd/2]·sin)`. -/
theorem rope_elem_bwd_spec_hi (dy : Slice I64) (i : Usize) (hd : Usize) (turn : U64) (pos : U32)
    (d2 r partner : Usize) (sc : fixed_trig.SinCos)
    (hd2 : hd / 2#usize = ok d2) (hrem : i % hd = ok r) (hrge : d2.val ≤ r.val)
    (hsub : i - d2 = ok partner) (_hsubv : partner.val = i.val - d2.val)
    (htrig : RopeTrigPair turn pos sc)
    (hi : i.val < dy.val.length) (hp : partner.val < dy.val.length)
    (hfit : I64.min ≤ roundShift32 ((dy.val.getD i.val default).val * sc.cos.val -
              (dy.val.getD partner.val default).val * sc.sin.val) ∧
            roundShift32 ((dy.val.getD i.val default).val * sc.cos.val -
              (dy.val.getD partner.val default).val * sc.sin.val) ≤ I64.max) :
    ∃ v : I64, learn_port.attn.rope_elem_bwd dy i hd turn pos = ok (core.result.Result.Ok v) ∧
      v.val = roundShift32 ((dy.val.getD i.val default).val * sc.cos.val -
              (dy.val.getD partner.val default).val * sc.sin.val) := by
  obtain ⟨angle, hangle, hsc, hcosb, hsinb⟩ := htrig
  have hhi : ¬ (r < d2) := by
    intro hc
    have := (UScalar.lt_equiv r d2).mp hc
    omega
  have hspec : learn_port.attn.rope_elem_bwd dy i hd turn pos ⦃ out =>
      ∃ v : I64, out = core.result.Result.Ok v ∧
        v.val = roundShift32 ((dy.val.getD i.val default).val * sc.cos.val -
              (dy.val.getD partner.val default).val * sc.sin.val) ⦄ := by
    unfold learn_port.attn.rope_elem_bwd
    rw [hcastt turn, hcastp pos, hcast64]
    rw [hd2]
    rw [hrem]
    conv => arg 1; simp only [bind_tc_ok]
    rw [if_neg hhi]
    rw [hsub]
    conv => arg 1; simp only [bind_tc_ok]
    step*
    have hi3v : i3.val = turn.val * pos.val := by
      simp only [i3_post, i1_post, i2_post, u128_cast_u64_val, u128_cast_u32_val]
    have hphasev : phase.val = Numeric.RopeTurns.phase turn.val pos.val := by
      have h5 : i5 = i3 &&& UScalar.cast .U128 core.num.U64.MAX := by
        apply UScalar.eq_of_val_eq
        simp only [i5_post, i4_post]
      rw [phase_post, h5]
      exact turn_phase_val_eq turn pos i3 hi3v
    have hi6v : i6 = angle := by
      apply IScalar.eq_of_val_eq
      rw [i6_post, hphasev, hangle]
    rw [hi6v, hsc]
    conv => arg 1; simp only [bind_tc_ok]
    simp only [if_neg hhi]
    rw [slice_index_eq dy i hi, slice_index_eq dy partner hp]
    conv => arg 1; simp only [lift, bind_tc_ok]
    obtain ⟨p, hp_eq, hpv⟩ := WP.spec_imp_exists
      (IScalar.mul_spec (x := core.convert.num.FromI128I64.from (dy.val.getD i.val default))
        (y := core.convert.num.FromI128I64.from sc.cos) (rp_prod_max _ _ hcosb).1
          (rp_prod_max _ _ hcosb).2)
    rw [hp_eq]
    conv => arg 1; simp only [bind_tc_ok]
    obtain ⟨q, hq_eq, hqv⟩ := WP.spec_imp_exists
      (IScalar.mul_spec (x := core.convert.num.FromI128I64.from (dy.val.getD partner.val default))
        (y := core.convert.num.FromI128I64.from sc.sin) (rp_prod_max _ _ hsinb).1
          (rp_prod_max _ _ hsinb).2)
    rw [hq_eq]
    conv => arg 1; simp only [bind_tc_ok]
    have hpv' : p.val = (dy.val.getD i.val default).val * sc.cos.val := by
      rw [hpv, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
    have hqv' : q.val = (dy.val.getD partner.val default).val * sc.sin.val := by
      rw [hqv, core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
    have hpb : |p.val| ≤ 2 ^ 96 := by
      rw [hpv']
      exact rp_prod (dy.val.getD i.val default) sc.cos hcosb
    have hqb : |q.val| ≤ 2 ^ 96 := by
      rw [hqv']
      exact rp_prod (dy.val.getD partner.val default) sc.sin hsinb
    have hsb : |p.val - q.val| ≤ 2 ^ 97 := by
      obtain ⟨hp1, hp2⟩ := abs_le.mp hpb
      obtain ⟨hq1, hq2⟩ := abs_le.mp hqb
      rw [abs_le]
      constructor <;> linarith
    obtain ⟨sm, hsm_eq, hsmv⟩ := WP.spec_imp_exists
      (IScalar.sub_spec (x := p) (y := q) (rp_sum_bounds p.val q.val hpb hqb).2.2.1
        (rp_sum_bounds p.val q.val hpb hqb).2.2.2)
    rw [hsm_eq]
    conv => arg 1; simp only [bind_tc_ok]
    have hsmv' : sm.val = (dy.val.getD i.val default).val * sc.cos.val -
        (dy.val.getD partner.val default).val * sc.sin.val := by
      rw [hsmv, hpv', hqv']
    have hsmb : |sm.val| ≤ 2 ^ 97 := by rw [hsmv]; exact hsb
    obtain ⟨rounded, hround, hroundv⟩ := WP.spec_imp_exists
      (value_round_shift_32_spec sm (rp_plus_half_bounds sm.val hsmb).1
        (rp_plus_half_bounds sm.val hsmb).2)
    have hfit' : I64.min ≤ rounded.val ∧ rounded.val ≤ I64.max := by
      rw [hroundv, hsmv']; exact hfit
    obtain ⟨v, hv, hvv⟩ := acc_round_ok sm rounded hround hfit'
    rw [hv]
    conv => arg 1; simp only [bind_tc_ok]
    simp only [WP.spec_ok]
    exact ⟨v, rfl, by rw [hvv, hroundv, hsmv']⟩
  obtain ⟨out, hout, v, hv, hvval⟩ := WP.spec_imp_exists hspec
  exact ⟨v, by rw [hout, hv], hvval⟩
/-! ## 3b. The backward loop -/

/-- Loop invariant at state `(dx, i, ok)`: the flag is still `true`, the index is at most `n`, the
buffer is the full length, and its content is the partial prefix at `i`. -/
def RopeBwdLoopInv (dy : Slice I64) (hd d2 n : Usize) (scAt : ℕ → fixed_trig.SinCos)
    (st : alloc.vec.Vec I64 × Usize × Bool) : Prop :=
  st.2.2 = true ∧ st.2.1.val ≤ n.val ∧ st.1.val.length = n.val ∧
  st.1.val.map (fun a => a.val) = ropeBwdPrefix dy.val hd.val d2.val scAt st.2.1.val

private theorem rb_div_two (hd d2 : Usize) (h : hd / 2#usize = ok d2) : d2.val = hd.val / 2 := by
  obtain ⟨z, hz, hzv⟩ := UScalar.div_spec hd (y := 2#usize) (by decide)
  obtain rfl : z = d2 := Result.ok_injective (hz.symm.trans h)
  exact hzv

private theorem rb_usize_max : Usize.max = 2 ^ System.Platform.numBits - 1 := by
  simp [Usize.max, Usize.numBits]

private theorem rb_bound64 (x : Usize) : x.val < 2 ^ System.Platform.numBits := by
  simpa using x.hmax

private theorem rb_heads_mul (hd n : Usize) (h : n.val % hd.val = 0) :
    n.val = n.val / hd.val * hd.val :=
  (Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero h)).symm

/-- **Body step.** On the success domain every arithmetic branch of one iteration succeeds, so the
step writes component `i` over the buffer and advances the index. -/
private theorem rope_bwd_loop_body_spec (dy : Slice I64) (pos : U32) (hd d2 n : Usize)
    (turns : Slice U64) (scAt : ℕ → fixed_trig.SinCos)
    (hh2 : 2 ≤ hd.val) (heven : hd.val % 2 = 0) (hnmod : n.val % hd.val = 0)
    (hhd2 : hd / 2#usize = ok d2)
    (htrig : ∀ j, j < d2.val → ∃ t, RopeTrigAtS turns pos j t (scAt j))
    (hfit : ∀ k, k < dy.val.length →
      I64.min ≤ ropeBwdElem dy.val hd.val d2.val scAt k ∧
      ropeBwdElem dy.val hd.val d2.val scAt k ≤ I64.max)
    (hlen : n.val = dy.val.length)
    (st : alloc.vec.Vec I64 × Usize × Bool)
    (hst : RopeBwdLoopInv dy hd d2 n scAt st) :
    learn_port.attn.rope_bwd_loop.body dy pos hd turns n st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | .done y => y.2 = true ∧
          y.1.val.map (fun a => a.val) = ropeBwdPrefix dy.val hd.val d2.val scAt n.val
      | .cont st' => RopeBwdLoopInv dy hd d2 n scAt st' ∧
          n.val - st'.2.1.val < n.val - st.2.1.val ⦄ := by
  rcases st with ⟨dx, hi, oki⟩
  obtain ⟨hok, hle, hlenx, hmap⟩ := hst
  dsimp only at hok hle hlenx hmap
  by_cases hlt : hi < n
  · have hltv : hi.val < n.val := (UScalar.lt_equiv hi n).mp hlt
    have hidy : hi.val < dy.val.length := by rw [← hlen]; exact hltv
    have hd2v : d2.val = hd.val / 2 := rb_div_two hd d2 hhd2
    have hhd : 0 < hd.val := by omega
    have hevenv : hd.val = 2 * d2.val := by omega
    have hd2pos : 0 < d2.val := by omega
    have hnval : n.val = n.val / hd.val * hd.val := rb_heads_mul hd n hnmod
    have hql : hi.val / hd.val < n.val / hd.val :=
      Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm, ← hnval]; exact hltv)
    have hn64 : n.val < 2 ^ System.Platform.numBits := rb_bound64 n
    obtain ⟨r, hr_eq, hrv⟩ := WP.spec_imp_exists (Usize.rem_spec hi (y := hd) (by omega))
    have hrv' : r.val = hi.val % hd.val := by simpa using hrv
    obtain ⟨i3, hi3_eq, hi3v⟩ := WP.spec_imp_exists (Usize.rem_spec r (y := d2) (by omega))
    have hi3v' : i3.val = r.val % d2.val := by simpa using hi3v
    have hi3lt : i3.val < d2.val := by rw [hi3v']; exact Nat.mod_lt _ hd2pos
    obtain ⟨t4, ht4, hpair4⟩ := htrig i3.val hi3lt
    obtain ⟨hturn_lt, -⟩ := List.getElem?_eq_some_iff.mp ht4
    have h1 := slice_index_eq_u64 turns i3 hturn_lt
    have hget : turns.val.getD i3.val default = t4 := by
      rw [List.getD_eq_getElem _ _ hturn_lt]
      exact (List.getElem?_eq_some_iff.mp ht4).2
    rw [← hget] at hpair4
    obtain ⟨hi1, hi5_eq, hi5v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := hi) (y := 1#usize)
        (by show hi.val + 1 ≤ Usize.max; rw [rb_usize_max]; omega))
    have hi5v' : hi1.val = hi.val + 1 := by simpa using hi5v
    have hidx : hi.val % hd.val % d2.val = i3.val := by rw [← hrv', ← hi3v']
    by_cases hrlt : r < d2
    · -- low half: the partner is `i + hd/2`
      have hrltv : r.val < d2.val := (UScalar.lt_equiv r d2).mp hrlt
      have hmodle : r.val ≤ hi.val := by rw [hrv']; exact Nat.mod_le hi.val hd.val
      have hsumlt : hi.val + d2.val < n.val := by
        have hq : hi.val / hd.val * hd.val + hi.val % hd.val = hi.val := by
          rw [Nat.mul_comm]; exact Nat.div_add_mod hi.val hd.val
        have h3 : r.val + d2.val < hd.val := by omega
        have hstep : hi.val + d2.val < (hi.val / hd.val + 1) * hd.val := by
          have hdist : (hi.val / hd.val + 1) * hd.val =
              hi.val / hd.val * hd.val + hd.val := by ring
          omega
        have hmul : (hi.val / hd.val + 1) * hd.val ≤ n.val := by
          rw [hnval]
          exact Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hql)
        omega
      obtain ⟨partner, hpa_eq, hpav⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := hi) (y := d2)
          (by show hi.val + d2.val ≤ Usize.max; rw [rb_usize_max]; omega))
      have hpav' : partner.val = hi.val + d2.val := by simpa using hpav
      have hplt : partner.val < dy.val.length := by rw [hpav', ← hlen]; exact hsumlt
      have hlt' : hi.val % hd.val < d2.val := by rw [← hrv']; exact hrltv
      have hpeq : hi.val + d2.val = partner.val := hpav'.symm
      have heq : ropeBwdElem dy.val hd.val d2.val scAt hi.val =
          roundShift32 ((dy.val.getD hi.val default).val * (scAt i3.val).cos.val +
            (dy.val.getD partner.val default).val * (scAt i3.val).sin.val) := by
        rw [ropeBwdElem_eq, ropeBwdPair]
        simp only [hidx, hlt', hpeq, decide_true, ↓reduceIte]
      have hfit' : I64.min ≤ roundShift32
            ((dy.val.getD hi.val default).val * (scAt i3.val).cos.val +
              (dy.val.getD partner.val default).val * (scAt i3.val).sin.val) ∧
          roundShift32 ((dy.val.getD hi.val default).val * (scAt i3.val).cos.val +
              (dy.val.getD partner.val default).val * (scAt i3.val).sin.val) ≤ I64.max := by
        rw [← heq]; exact hfit hi.val hidy
      obtain ⟨v, hv_eq, hvv⟩ := rope_elem_bwd_spec_lo dy hi hd
        (turns.val.getD i3.val default) pos d2 r partner (scAt i3.val) hhd2 hr_eq hrltv
        hpa_eq hpav' hpair4 hidy hplt hfit'
      obtain ⟨pq, hq_eq0, hq_spec⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.index_mut_usize_spec dx hi (by have h : dx.length = n.val := hlenx; omega))
      rcases pq with ⟨idx0, back⟩
      have hq_eq : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice I64) dx hi =
          ok (idx0, back) := by
        rw [alloc.vec.Vec.index_mut_slice_index]; exact hq_eq0
      have hback : back = dx.set hi := hq_spec.2
      have hbody : learn_port.attn.rope_bwd_loop.body dy pos hd turns n dx hi oki =
          ok (ControlFlow.cont (dx.set hi v, hi1, true)) := by
        unfold learn_port.attn.rope_bwd_loop.body
        rw [if_pos hlt, hok, if_pos rfl]
        rw [hr_eq]
        simp only [bind_tc_ok]
        rw [hhd2]
        simp only [bind_tc_ok]
        rw [hi3_eq]
        simp only [bind_tc_ok]
        rw [h1]
        simp only [bind_tc_ok]
        rw [hv_eq]
        simp only [bind_tc_ok]
        simp only [hq_eq, bind_tc_ok, uncurry_apply_pair]
        rw [hback, hi5_eq]
        simp only [bind_tc_ok]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨rfl, ?_, ?_, ?_⟩, ?_⟩
      · rw [hi5v']; omega
      · simp only [alloc.vec.Vec.set_val_eq, List.length_set, hlenx]
      · simp only [alloc.vec.Vec.set_val_eq, List.map_set, hmap, hvv, hi5v']
        rw [← heq]
        exact ropeBwdPrefix_succ dy.val hd.val d2.val scAt hi.val
      · rw [hi5v']; omega
    · -- high half: the partner is `i - hd/2`
      have hge : d2.val ≤ r.val := by
        have h := (UScalar.lt_equiv r d2).not.mp hrlt
        omega
      have hmodle : r.val ≤ hi.val := by rw [hrv']; exact Nat.mod_le hi.val hd.val
      have hle2 : d2.val ≤ hi.val := by omega
      obtain ⟨partner, hpa_eq, hpav⟩ := WP.spec_imp_exists
        (Usize.sub_spec (x := hi) (y := d2) (by show d2.val ≤ hi.val; exact hle2))
      have hpav' : partner.val = hi.val - d2.val := by simpa using hpav.1
      have hplt2 : partner.val ≤ hi.val := by rw [hpav']; omega
      have hplt : partner.val < dy.val.length := by
        have h2 : partner.val < n.val := by omega
        rw [← hlen]; exact h2
      have hlt' : ¬ (hi.val % hd.val < d2.val) := by rw [← hrv']; exact hrlt
      have hpeq : hi.val - d2.val = partner.val := hpav'.symm
      have heq : ropeBwdElem dy.val hd.val d2.val scAt hi.val =
          roundShift32 ((dy.val.getD hi.val default).val * (scAt i3.val).cos.val -
            (dy.val.getD partner.val default).val * (scAt i3.val).sin.val) := by
        rw [ropeBwdElem_eq, ropeBwdPair]
        simp only [hidx, hlt', hpeq, decide_false, Bool.false_eq_true, ↓reduceIte]
      have hfit' : I64.min ≤ roundShift32
            ((dy.val.getD hi.val default).val * (scAt i3.val).cos.val -
              (dy.val.getD partner.val default).val * (scAt i3.val).sin.val) ∧
          roundShift32 ((dy.val.getD hi.val default).val * (scAt i3.val).cos.val -
              (dy.val.getD partner.val default).val * (scAt i3.val).sin.val) ≤ I64.max := by
        rw [← heq]; exact hfit hi.val hidy
      obtain ⟨v, hv_eq, hvv⟩ := rope_elem_bwd_spec_hi dy hi hd
        (turns.val.getD i3.val default) pos d2 r partner (scAt i3.val) hhd2 hr_eq hge
        hpa_eq hpav' hpair4 hidy hplt hfit'
      obtain ⟨pq, hq_eq0, hq_spec⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.index_mut_usize_spec dx hi (by have h : dx.length = n.val := hlenx; omega))
      rcases pq with ⟨idx0, back⟩
      have hq_eq : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice I64) dx hi =
          ok (idx0, back) := by
        rw [alloc.vec.Vec.index_mut_slice_index]; exact hq_eq0
      have hback : back = dx.set hi := hq_spec.2
      have hbody : learn_port.attn.rope_bwd_loop.body dy pos hd turns n dx hi oki =
          ok (ControlFlow.cont (dx.set hi v, hi1, true)) := by
        unfold learn_port.attn.rope_bwd_loop.body
        rw [if_pos hlt, hok, if_pos rfl]
        rw [hr_eq]
        simp only [bind_tc_ok]
        rw [hhd2]
        simp only [bind_tc_ok]
        rw [hi3_eq]
        simp only [bind_tc_ok]
        rw [h1]
        simp only [bind_tc_ok]
        rw [hv_eq]
        simp only [bind_tc_ok]
        simp only [hq_eq, bind_tc_ok, uncurry_apply_pair]
        rw [hback, hi5_eq]
        simp only [bind_tc_ok]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨⟨rfl, ?_, ?_, ?_⟩, ?_⟩
      · rw [hi5v']; omega
      · simp only [alloc.vec.Vec.set_val_eq, List.length_set, hlenx]
      · simp only [alloc.vec.Vec.set_val_eq, List.map_set, hmap, hvv, hi5v']
        rw [← heq]
        exact ropeBwdPrefix_succ dy.val hd.val d2.val scAt hi.val
      · rw [hi5v']; omega
  · have hhev : hi.val = n.val := by
      have hnlt : ¬ hi.val < n.val := fun h => hlt ((UScalar.lt_equiv hi n).mpr h)
      omega
    have hbody : learn_port.attn.rope_bwd_loop.body dy pos hd turns n dx hi oki =
        ok (ControlFlow.done (dx, oki)) := by
      unfold learn_port.attn.rope_bwd_loop.body
      rw [if_neg hlt]
    rw [hbody]
    simp only [WP.spec_ok]
    exact ⟨hok, by rw [hmap, hhev]⟩
/-! ## 3c. The backward loop and the public wrapper -/

/-- **Loop spec.** On the success domain the loop returns the flag `true` and the full transposed
prefix over the zero-filled buffer. -/
theorem rope_bwd_loop_spec (dy : Slice I64) (pos : U32) (hd d2 n : Usize) (turns : Slice U64)
    (dx : alloc.vec.Vec I64) (scAt : ℕ → fixed_trig.SinCos)
    (hh2 : 2 ≤ hd.val) (heven : hd.val % 2 = 0) (hnmod : n.val % hd.val = 0)
    (hlen : n.val = dy.val.length) (hdxlen : dx.val.length = n.val)
    (hdx0 : dx.val.map (fun a => a.val) = List.replicate n.val 0)
    (hhd2 : hd / 2#usize = ok d2)
    (htrig : ∀ j, j < d2.val → ∃ t, RopeTrigAtS turns pos j t (scAt j))
    (hfit : ∀ k, k < dy.val.length →
      I64.min ≤ ropeBwdElem dy.val hd.val d2.val scAt k ∧
      ropeBwdElem dy.val hd.val d2.val scAt k ≤ I64.max) :
    ∃ dx' : alloc.vec.Vec I64,
      learn_port.attn.rope_bwd_loop dy pos hd turns n dx 0#usize true = ok (dx', true) ∧
      dx'.val.map (fun a => a.val) = ropeBwdPrefix dy.val hd.val d2.val scAt n.val := by
  have hspec : loop
      (fun (st : alloc.vec.Vec I64 × Usize × Bool) =>
        learn_port.attn.rope_bwd_loop.body dy pos hd turns n st.1 st.2.1 st.2.2)
      (dx, 0#usize, true) ⦃ r => r.2 = true ∧
        r.1.val.map (fun a => a.val) = ropeBwdPrefix dy.val hd.val d2.val scAt n.val ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun st : alloc.vec.Vec I64 × Usize × Bool => n.val - st.2.1.val)
      (inv := RopeBwdLoopInv dy hd d2 n scAt)
      (post := fun r : alloc.vec.Vec I64 × Bool => r.2 = true ∧
        r.1.val.map (fun a => a.val) = ropeBwdPrefix dy.val hd.val d2.val scAt n.val)
    · intro st hst
      convert rope_bwd_loop_body_spec dy pos hd d2 n turns scAt hh2 heven hnmod hhd2 htrig hfit
        hlen st hst using 1
      funext r
      cases r with
      | done y => rfl
      | cont st' => rfl
    · refine ⟨rfl, by simp, hdxlen, ?_⟩
      show dx.val.map (fun a => a.val) = ropeBwdPrefix dy.val hd.val d2.val scAt 0
      rw [hdx0, hlen, ropeBwdPrefix_zero]
  obtain ⟨r, hr, hr2, hrm⟩ := WP.spec_imp_exists hspec
  obtain ⟨dx', ok'⟩ := r
  have hok' : ok' = true := hr2
  refine ⟨dx', ?_, hrm⟩
  rw [learn_port.attn.rope_bwd_loop, hr, hok']

/-- **Public wrapper.** The two static guards hold on the domain, so `rope_bwd` returns the transpose
over the zero-filled buffer. -/
theorem rope_bwd_refines : RopeBwdSpec := by
  intro table dy pos d2 scAt hdom hhd2 htrig hfit
  obtain ⟨hpos, hh2, heven, hmod, hturns⟩ := hdom
  have hb : core.slice.Slice.is_empty dy = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, Nat.ne_of_gt hpos]
  obtain ⟨i1, hi1_eq, hi1v⟩ := WP.spec_imp_exists
    (Usize.rem_spec (Slice.len dy) (y := table.head_dim) (by omega))
  have hi10 : i1 = 0#usize := by
    apply UScalar.eq_of_val_eq
    rw [hi1v, Slice.len_val, hmod]
    rfl
  have hcond : ¬ ((i1 != 0#usize) = true) := by
    rw [bne_iff_ne]
    exact fun h => h hi10
  obtain ⟨dx0, hdx0_eq, hdx0v, hdx0len⟩ := WP.spec_imp_exists
    (alloc.vec.from_elem_spec core.clone.CloneI64 0#i64 (Slice.len dy) (by rfl))
  have hdx0map : dx0.val.map (fun a => a.val) = List.replicate (Slice.len dy).val 0 := by
    rw [hdx0v]
    simp
  have hdxlen : dx0.val.length = (Slice.len dy).val := hdx0len
  obtain ⟨dx', hloop_eq, hloop_map⟩ := rope_bwd_loop_spec dy pos table.head_dim d2 (Slice.len dy)
    (alloc.vec.Vec.deref table.turns) dx0 scAt hh2 heven (by rw [Slice.len_val]; exact hmod)
    (Slice.len_val dy) hdxlen hdx0map hhd2 htrig hfit
  refine ⟨dx', ?_, ?_⟩
  · unfold learn_port.attn.rope_bwd
    rw [hb]
    simp only [bind_tc_ok]
    rw [if_neg (by decide : ¬ ((false : Bool) = true))]
    rw [show rope.RopeTable.impl.head_dim table = ok table.head_dim from rfl]
    simp only [bind_tc_ok]
    rw [hi1_eq]
    simp only [bind_tc_ok]
    rw [if_neg hcond]
    rw [show rope.RopeTable.impl.turns table = ok (alloc.vec.Vec.deref table.turns) from rfl]
    simp only [bind_tc_ok]
    rw [hdx0_eq]
    simp only [bind_tc_ok]
    rw [hloop_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
    simp
  · have h2 : List.length (↑dy : List I64) = (Slice.len dy).val := (Slice.len_val dy).symm
    rw [h2]
    exact hloop_map
/-! ## 4. The forward single-vector rotation -/

/-- Loop invariant at state `(x1, h, ok)` of `rotate_one_loop`. -/
def RotOneInv (hd d2 : Usize) (scAt : ℕ → fixed_trig.SinCos)
    (x : Slice I64) (heads : Usize) (st : Slice I64 × Usize × Bool) : Prop :=
  st.2.2 = true ∧
  st.2.1.val ≤ heads.val ∧
  st.1.val.map (fun a => a.val) = rotHeadsState x.val hd.val d2.val scAt st.2.1.val ∧
  st.1.val.drop (st.2.1.val * hd.val) = x.val.drop (st.2.1.val * hd.val)

/-- **Body step.** Under the success-domain hypotheses every arithmetic branch succeeds, so one
iteration replaces the head window of `x1` by its `rotPrefix` content and advances the index. -/
private theorem rot_one_loop_body_spec (self : rope.RopeTable) (v : alloc.vec.Vec U64) (pos : U32)
    (heads : Usize) (x : Slice I64) (scAt : ℕ → fixed_trig.SinCos) (d2 : Usize)
    (hhd : 0 < self.head_dim.val)
    (hlenx : x.val.length = self.head_dim.val * heads.val)
    (hd2 : self.head_dim / 2#usize = ok d2)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt v pos i t (scAt i))
    (hfit : ∀ hh, hh < heads.val → RotHeadFitsL (headWin x.val self.head_dim.val hh) d2.val scAt)
    (st : Slice I64 × Usize × Bool)
    (hst : RotOneInv self.head_dim d2 scAt x heads st) :
    rope.RopeTable.rotate_one_loop.body self self.head_dim v pos heads
        st.1 st.2.1 st.2.2 ⦃ r =>
      match r with
      | .done y => y.2 = true ∧
          y.1.val.map (fun a => a.val) =
            rotHeadsState x.val self.head_dim.val d2.val scAt heads.val
      | .cont st' => RotOneInv self.head_dim d2 scAt x heads st' ∧
          heads.val - st'.2.1.val < heads.val - st.2.1.val ⦄ := by
  rcases st with ⟨x1, hi, oki⟩
  obtain ⟨hok, hle, hmap, hdrop⟩ := hst
  dsimp only at hok hle hmap hdrop
  have hd2v : d2.val = self.head_dim.val / 2 := by
    obtain ⟨z, hz, hzv⟩ := UScalar.div_spec self.head_dim (y := 2#usize) (by decide)
    have hz_eq : z = d2 := Result.ok_injective (hz.symm.trans hd2)
    rw [← hz_eq]; exact hzv
  have hx1len : x1.val.length = x.val.length := by
    have := congrArg List.length hmap
    rw [List.length_map, rotHeadsState_length] at this
    exact this
  by_cases hlt : hi < heads
  · have hltv : hi.val < heads.val := (UScalar.lt_equiv hi heads).mp hlt
    -- window arithmetic
    have hlenx' : (Slice.len x).val = self.head_dim.val * heads.val := by
      rw [Slice.len_val]; exact hlenx
    have hdiv : self.head_dim.val ∣ (Slice.len x).val := ⟨heads.val, hlenx'⟩
    have hheads : heads.val = (Slice.len x).val / self.head_dim.val := by
      rw [hlenx', Nat.mul_div_cancel_left _ hhd]
    obtain ⟨hwb1, hwb2, hwb3⟩ :=
      rotate_window_arith x self.head_dim hi heads hhd hdiv hheads hlt
    -- base, i
    obtain ⟨base, hbase, hbasev⟩ := rr_usize_mul hi self.head_dim hwb1
    have hbadd : base.val + self.head_dim.val ≤ Usize.max := by rw [hbasev]; exact hwb2
    obtain ⟨i, hiadd, hiv⟩ := rr_usize_add base self.head_dim hbadd
    have hivle : i.val ≤ (Slice.len x).val := by rw [hiv, hbasev]; exact hwb3
    have hivle1 : i.val ≤ x1.val.length := by rw [hx1len]; exact hivle
    have hile : base.val ≤ i.val := by rw [hiv]; omega
    -- index_mut x1
    obtain ⟨p, hp_eq0, hp_spec⟩ := WP.spec_imp_exists
      (core.slice.index.SliceIndexRangeUsizeSlice.index_mut.step_spec
        ({ start := base, «end» := i } : core.ops.range.Range Usize) x1
        ((UScalar.le_equiv base i).2 hile)
        (by simpa using hivle1))
    rcases p with ⟨s, back⟩
    have hp_eq : core.slice.index.Slice.index_mut
        (core.slice.index.SliceIndexRangeUsizeSlice I64) x1
        { start := base, «end» := i } = ok (s, back) := by
      rw [Slice.index_mut_SliceIndexRangeUsizeSliceInst]; exact hp_eq0
    obtain ⟨hs_val, _, hs_back⟩ := hp_spec
    have hs_hw1 : s.val = headWin x1.val self.head_dim.val hi.val := by
      rw [hs_val, hiv, hbasev, slice_eq_headWin]
    have hwin : headWin x1.val self.head_dim.val hi.val =
        headWin x.val self.head_dim.val hi.val := by
      unfold headWin; rw [hdrop]
    have hs_hw : s.val = headWin x.val self.head_dim.val hi.val := hs_hw1.trans hwin
    have hfit_s : RotHeadFits s d2 scAt :=
      (RotHeadFitsL_slice s d2 scAt).mp (by rw [hs_hw]; exact hfit hi.val hltv)
    have hside : (hi.val + 1) * self.head_dim.val ≤ x.val.length := by
      rw [hlenx, Nat.mul_comm self.head_dim.val heads.val]
      exact Nat.mul_le_mul_right self.head_dim.val (Nat.succ_le_of_lt hltv)
    have hslen_hd : s.val.length = self.head_dim.val := by
      rw [hs_hw, headWin_length x.val self.head_dim.val hi.val hside]
    have hx_s : 2 * d2.val ≤ s.length := by
      rw [Slice.length, hslen_hd, hd2v]
      rw [Nat.mul_comm]
      exact Nat.div_mul_le_self _ _
    obtain ⟨s1, hrot0, hcont⟩ :=
      rotate_head_content { head_dim := self.head_dim, turns := v } s pos d2 hd2 scAt
        htrig hfit_s hx_s
    have hrot : (rope.RopeTable.mk self.head_dim v).rotate_head s pos =
        ok (core.result.Result.Ok (), s1) := hrot0
    have hs1len : s1.val.length = self.head_dim.val := by
      have := congrArg List.length hcont
      rw [List.length_map, rotPrefix_length, hslen_hd] at this
      exact this
    -- the advance
    have hinc : hi.val + 1 ≤ Usize.max := by
      have h1 : hi.val + 1 ≤ heads.val := Nat.succ_le_of_lt hltv
      have h2 : heads.val ≤ Usize.max := by
        calc heads.val ≤ self.head_dim.val * heads.val := Nat.le_mul_of_pos_left _ hhd
          _ = (Slice.len x).val := hlenx'.symm
          _ ≤ Usize.max := Slice.length_ineq x
      omega
    obtain ⟨h1u, hinc_eq, h1uv⟩ := rr_usize_add hi 1#usize (by simpa using hinc)
    have h1v : h1u.val = hi.val + 1 := by simpa using h1uv
    have hbody : rope.RopeTable.rotate_one_loop.body self self.head_dim v pos heads
        x1 hi true = ok (ControlFlow.cont (x1.setSlice! base.val s1, h1u, true)) := by
      unfold rope.RopeTable.rotate_one_loop.body
      dsimp only
      rw [if_pos hlt, if_pos rfl]
      rw [hbase]
      simp only [bind_tc_ok]
      rw [hiadd]
      simp only [bind_tc_ok]
      simp only [hp_eq, bind_tc_ok, uncurry_apply_pair]
      rw [hrot]
      simp only [bind_tc_ok, uncurry_apply_pair]
      simp only [hinc_eq, bind_tc_ok]
      rw [hs_back s1]
    rw [hok, hbody]
    simp only [WP.spec_ok]
    have hx1c : (x1.setSlice! base.val s1).val.map (fun a => a.val) =
        (rotHeadsState x.val self.head_dim.val d2.val scAt hi.val).setSlice!
          (hi.val * self.head_dim.val)
          (rotPrefix (headWin x.val self.head_dim.val hi.val) d2.val scAt d2.val) := by
      rw [Slice.setSlice!_val, List.map_setSlice!, hmap, hcont, hs_hw, hbasev]
    have hx1d : (x1.setSlice! base.val s1).val.drop (h1u.val * self.head_dim.val) =
        x.val.drop (h1u.val * self.head_dim.val) := by
      rw [Slice.setSlice!_val]
      have hidx : h1u.val * self.head_dim.val = base.val + s1.val.length := by
        rw [h1v, hbasev, hs1len]; ring
      rw [hidx, drop_setSlice!_eq]
      rw [show base.val + s1.val.length = (hi.val + 1) * self.head_dim.val by
        rw [hbasev, hs1len]; ring]
      rw [show (hi.val + 1) * self.head_dim.val =
        hi.val * self.head_dim.val + self.head_dim.val by ring]
      rw [← List.drop_drop, hdrop, List.drop_drop]
    refine ⟨⟨rfl, by rw [h1v]; omega, ?_, ?_⟩, by rw [h1v]; omega⟩
    · rw [hx1c, h1v, rotHeadsState]
    · rw [hx1d, h1v]
  · have hhev : hi.val = heads.val := by
      have hnlt : ¬ hi.val < heads.val := fun h => hlt ((UScalar.lt_equiv hi heads).mpr h)
      omega
    have hbody : rope.RopeTable.rotate_one_loop.body self self.head_dim v pos heads
        x1 hi true = ok (ControlFlow.done (x1, true)) := by
      unfold rope.RopeTable.rotate_one_loop.body
      dsimp only
      rw [if_neg hlt]
    rw [hok, hbody]
    simp only [WP.spec_ok]
    exact ⟨trivial, by rw [← hhev]; exact hmap⟩

/-- **Outer loop.** On the success domain — `head_dim ∣ x.len`, `d2 = head_dim/2`, every rotary index
carries its `RopeTrigAt` trig pair and every head window satisfies `RotHeadFitsL` — `rotate_one_loop`
returns the flag `true` and the slice denotes the `rotHeadsState` content. -/
theorem rot_one_loop_spec (self : rope.RopeTable) (x : Slice I64) (pos : U32) (heads : Usize)
    (d2 : Usize) (scAt : ℕ → fixed_trig.SinCos)
    (hhd : 0 < self.head_dim.val)
    (hlenx : x.val.length = self.head_dim.val * heads.val)
    (hd2 : self.head_dim / 2#usize = ok d2)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt self.turns pos i t (scAt i))
    (hfit : ∀ hh, hh < heads.val → RotHeadFitsL (headWin x.val self.head_dim.val hh) d2.val scAt) :
    ∃ x' : Slice I64,
      rope.RopeTable.rotate_one_loop self self.head_dim self.turns x pos heads 0#usize true =
        ok (x', true) ∧
      x'.val.map (fun a => a.val) =
        rotHeadsState x.val self.head_dim.val d2.val scAt heads.val := by
  have hspec : loop
      (fun (s : Slice I64 × Usize × Bool) =>
        rope.RopeTable.rotate_one_loop.body self self.head_dim self.turns pos heads
          s.1 s.2.1 s.2.2)
      (x, 0#usize, true) ⦃ r =>
        r.2 = true ∧
        r.1.val.map (fun a => a.val) =
          rotHeadsState x.val self.head_dim.val d2.val scAt heads.val ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Bool => heads.val - st.2.1.val)
      (inv := RotOneInv self.head_dim d2 scAt x heads)
      (post := fun r : Slice I64 × Bool => r.2 = true ∧
        r.1.val.map (fun a => a.val) =
          rotHeadsState x.val self.head_dim.val d2.val scAt heads.val)
    · intro st hst
      convert rot_one_loop_body_spec self self.turns pos heads x scAt d2 hhd hlenx hd2 htrig
        hfit st hst using 1
      funext r
      cases r with
      | done y => rfl
      | cont st' => rfl
    · refine ⟨rfl, by simp, ?_, ?_⟩
      · show x.val.map (fun a => a.val) = rotHeadsState x.val self.head_dim.val d2.val scAt
          (0#usize).val
        rfl
      · show x.val.drop ((0#usize).val * self.head_dim.val) =
          x.val.drop ((0#usize).val * self.head_dim.val)
        rfl
  obtain ⟨r, hr, hr2, hrm⟩ := WP.spec_imp_exists hspec
  obtain ⟨a, c⟩ := r
  have hc : c = true := hr2
  refine ⟨a, ?_, hrm⟩
  rw [rope.RopeTable.rotate_one_loop, hr, hc]

/-- **Wrapper.** The static guard `head_dim ∣ x.len` holds, so the extracted `rotate_one` returns
`Ok ()` and the rotated slice. -/
theorem rotate_one_spec (table : rope.RopeTable) (z : Slice I64) (pos : U32) (d2 : Usize)
    (scAt : ℕ → fixed_trig.SinCos)
    (hhd : 0 < table.head_dim.val)
    (hlen : table.head_dim.val ∣ z.val.length)
    (hd2 : table.head_dim / 2#usize = ok d2)
    (htrig : ∀ i, i < d2.val → ∃ t, RopeTrigAt table.turns pos i t (scAt i))
    (hfit : ∀ h, h < z.val.length / table.head_dim.val →
      RotHeadFitsL (headWin z.val table.head_dim.val h) d2.val scAt) :
    ∃ z' : Slice I64,
      rope.RopeTable.rotate_one table z pos = ok (core.result.Result.Ok (), z') ∧
      z'.val.map (fun a => a.val) =
        rotHeadsState z.val table.head_dim.val d2.val scAt
          (z.val.length / table.head_dim.val) := by
  have hmod : z.val.length % table.head_dim.val = 0 := (Nat.dvd_iff_mod_eq_zero).mp hlen
  obtain ⟨b, hb_eq0, hb_spec⟩ := WP.spec_imp_exists
    (UScalar.is_multiple_of.spec (Slice.len z) table.head_dim)
  have hb_eq : core.num.Usize.is_multiple_of (Slice.len z) table.head_dim = ok b := hb_eq0
  have hb_true : b = true := hb_spec.mpr (by rw [Slice.len_val]; exact hmod)
  obtain ⟨heads, hheads_eq, hheadsv⟩ :=
    UScalar.div_spec (Slice.len z) (y := table.head_dim) (by omega)
  have hheadsv' : heads.val = z.val.length / table.head_dim.val := by
    rw [hheadsv, Slice.len_val]
  have hlenz : z.val.length = table.head_dim.val * heads.val := by
    rw [hheadsv', Nat.mul_div_cancel' hlen]
  have hfit'' : ∀ hh, hh < heads.val →
      RotHeadFitsL (headWin z.val table.head_dim.val hh) d2.val scAt := by
    intro hh hhlt
    rw [hheadsv'] at hhlt
    exact hfit hh hhlt
  obtain ⟨z1, hloop_eq, hloop_map⟩ :=
    rot_one_loop_spec table z pos heads d2 scAt hhd hlenz hd2 htrig hfit''
  rw [hheadsv'] at hloop_map
  refine ⟨z1, ?_, hloop_map⟩
  unfold rope.RopeTable.rotate_one
  simp only []
  rw [hb_eq]
  simp only [bind_tc_ok]
  rw [if_pos hb_true]
  rw [hheads_eq]
  simp only [bind_tc_ok]
  rw [hloop_eq]
  simp only [bind_tc_ok, uncurry_apply_pair]
  simp

/-- **Public entry.** On the success domain the extracted `rope_fwd` copies the vector, runs
`rotate_one` and returns the copy: its content is `rotHeadsState` at the head count. -/
theorem rope_fwd_refines : RopeFwdSpec := by
  intro table x pos d2 scAt hdom hhd2 htrig hfit
  obtain ⟨hpos, hh2, heven, hmod, hturns⟩ := hdom
  have hb : core.slice.Slice.is_empty x = ok false := by
    simp [core.slice.Slice.is_empty, Slice.length, Nat.ne_of_gt hpos]
  obtain ⟨y, htovec, hy_slice⟩ := WP.spec_imp_exists
    (alloc.slice.Slice.to_vec_spec core.clone.CloneI64 x (by intro a ha; rfl))
  have hval : (y.slice).val = x.val :=
    (congrArg (fun s : Slice I64 => s.val) hy_slice).symm
  have hdvd : table.head_dim.val ∣ (y.slice).val.length := by
    rw [hval]
    exact (Nat.dvd_iff_mod_eq_zero).mpr hmod
  have hfit' : ∀ h, h < (y.slice).val.length / table.head_dim.val →
      RotHeadFitsL (headWin (y.slice).val table.head_dim.val h) d2.val scAt := by
    intro h hh
    rw [hval] at hh ⊢
    exact hfit h hh
  obtain ⟨x1, hrot_eq, hrot_map⟩ :=
    rotate_one_spec table (y.slice) pos d2 scAt (by omega) hdvd hhd2 htrig hfit'
  rw [hval] at hrot_map
  have hdm : alloc.vec.Vec.deref_mut y = (y.slice, fun s => alloc.vec.Vec.ofSlice s) := by
    apply Prod.ext
    · rfl
    · funext s; rfl
  refine ⟨alloc.vec.Vec.ofSlice x1, ?_, ?_⟩
  · unfold learn_port.attn.rope_fwd
    rw [hb]
    simp only [bind_tc_ok]
    rw [if_neg (by decide : ¬ ((false : Bool) = true))]
    rw [htovec]
    simp only [bind_tc_ok]
    simp only [lift, bind_tc_ok]
    rw [hdm]
    simp only [uncurry_apply_pair]
    rw [hrot_eq]
    simp only [bind_tc_ok, uncurry_apply_pair]
  · have hof : (↑(alloc.vec.Vec.ofSlice x1) : List I64) = (↑x1 : List I64) := rfl
    rw [hof]
    exact hrot_map
/-! ## 5. Witnesses -/

/-- The backward loop's slice view of the turn table shares its contents. -/
theorem vecDeref_val (v : alloc.vec.Vec U64) : (alloc.vec.Vec.deref v).val = v.val := by
  simp [alloc.vec.Vec.deref, Slice.from_val]

/-- The witness turn slot read through the slice view the backward loop uses. -/
theorem ropeWitTable_htrigS :
    ∀ j, j < ropeWitD2.val →
      ∃ t, RopeTrigAtS ropeWitTable.turns.deref ropeWitPos j t (ropeWitScAt j) := by
  intro j hj
  obtain ⟨t, ht⟩ := ropeWitTable_htrig j hj
  refine ⟨t, ?_⟩
  rw [RopeTrigAtS, vecDeref_val]
  exact ht

/-- The witness head, table and `d2` satisfy the static domain `RopeOneDomain`. -/
theorem ropeWitTable_domain : RopeOneDomain ropeWitTable ropeWitHead := by
  have hhead : ropeWitHead.val = [4294967296#i64, 0#i64] := Slice.from_val _ _
  have hturns : ropeWitTable.turns.val = [0#u64] := alloc.vec.Vec.from_val _ _
  have hhdv : ropeWitTable.head_dim.val = 2 := by norm_num [ropeWitTable]
  have hlen : ropeWitHead.val.length = 2 := by rw [hhead]; rfl
  unfold RopeOneDomain
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · norm_num [hhead]
  · norm_num [ropeWitTable]
  · norm_num [ropeWitTable]
  · norm_num [hhead, ropeWitTable]
  · norm_num [ropeWitTable, alloc.vec.Vec.from_val]

/-- **Forward witness.** Every hypothesis of `rope_fwd_refines` holds at once at the witness table
and head, so `rope_fwd` returns the `rotHeadsState` content there. -/
theorem rope_fwd_witness :
    ∃ y' : alloc.vec.Vec I64,
      learn_port.attn.rope_fwd ropeWitHead ropeWitPos ropeWitTable =
        ok (core.result.Result.Ok y') ∧
      y'.val.map (fun a => a.val) =
        rotHeadsState ropeWitHead.val 2 ropeWitD2.val ropeWitScAt 1 := by
  have hhead : ropeWitHead.val = [4294967296#i64, 0#i64] := Slice.from_val _ _
  have hhdv : ropeWitTable.head_dim.val = 2 := by norm_num [ropeWitTable]
  have hlen : ropeWitHead.val.length = 2 := by rw [hhead]; rfl
  have hfit : ∀ h, h < ropeWitHead.val.length / ropeWitTable.head_dim.val →
      RotHeadFitsL (headWin ropeWitHead.val ropeWitTable.head_dim.val h) ropeWitD2.val
        ropeWitScAt := by
    intro h hh
    have hh0 : h = 0 := by rw [hlen, hhdv] at hh; omega
    subst hh0
    have hwin : headWin ropeWitHead.val 2 0 = ropeWitHead.val := by
      rw [headWin, List.drop_zero]
      exact List.take_of_length_le (by rw [hlen])
    rw [hhdv, hwin]
    exact (RotHeadFitsL_slice ropeWitHead ropeWitD2 ropeWitScAt).mpr ropeWitTable_hfit
  obtain ⟨y', hrun, hmap⟩ := rope_fwd_refines ropeWitTable ropeWitHead ropeWitPos ropeWitD2
    ropeWitScAt ropeWitTable_domain rope_head_domain_witness.1 ropeWitTable_htrig hfit
  refine ⟨y', hrun, ?_⟩
  rw [hmap, hhdv, hlen]

/-- `roundShift32` of a `2^65`-bounded integer stays in `i64`. -/
private theorem ropeBwd_roundShift_bounds (v : ℤ) (hv : |v| ≤ (2:ℤ)^65) :
    I64.min ≤ roundShift32 v ∧ roundShift32 v ≤ I64.max := by
  have h := roundShift32_half v
  have hvR : |(v : ℝ)| ≤ (2:ℝ)^65 := by exact_mod_cast hv
  have hq : (0:ℝ) < q32 := q32_pos
  have hpq := abs_le.mp h
  have hupper : (roundShift32 v : ℝ) ≤ (2:ℝ)^33 + 1/2 := by
    have hvle : (v:ℝ) ≤ (2:ℝ)^65 := (abs_le.mp hvR).2
    have h1 : (v:ℝ)/q32 ≤ (2:ℝ)^65/q32 := div_le_div_of_nonneg_right hvle (le_of_lt hq)
    have hqeq : (2:ℝ)^65/q32 = (2:ℝ)^33 := by norm_num [q32]
    linarith [hpq.2, h1, hqeq.le, hqeq.ge]
  have hlower : -((2:ℝ)^33) - 1/2 ≤ (roundShift32 v : ℝ) := by
    have hvge : -((2:ℝ)^65) ≤ (v:ℝ) := (abs_le.mp hvR).1
    have h1 : -((2:ℝ)^65)/q32 ≤ (v:ℝ)/q32 := div_le_div_of_nonneg_right hvge (le_of_lt hq)
    have hqeq : -((2:ℝ)^65)/q32 = -((2:ℝ)^33) := by norm_num [q32]
    linarith [hpq.1, h1, hqeq.le, hqeq.ge]
  have hup : roundShift32 v ≤ (2:ℤ)^33 := by
    have : (roundShift32 v : ℤ) < (2:ℤ)^33 + 1 := by
      exact_mod_cast (by linarith : (roundShift32 v : ℝ) < (2:ℝ)^33 + 1)
    omega
  have hlo : -((2:ℤ)^33) ≤ roundShift32 v := by
    have : -((2:ℤ)^33) - 1 < roundShift32 v := by
      exact_mod_cast (by linarith : -((2:ℝ)^33) - 1 < (roundShift32 v : ℝ))
    omega
  constructor
  · have : I64.min = -((2:ℤ)^63) := by norm_num [I64.min, I64.numBits]
    omega
  · have : I64.max = (2:ℤ)^63 - 1 := by norm_num [I64.max, I64.numBits]
    omega

/-- **Backward witness.** Every hypothesis of `rope_bwd_refines` holds at once at the witness table
and head, so `rope_bwd` returns the full transposed prefix there. -/
theorem rope_bwd_witness :
    ∃ dx : alloc.vec.Vec I64,
      learn_port.attn.rope_bwd ropeWitHead ropeWitPos ropeWitTable =
        ok (core.result.Result.Ok dx) ∧
      dx.val.map (fun a => a.val) =
        ropeBwdPrefix ropeWitHead.val 2 ropeWitD2.val ropeWitScAt 2 := by
  have hhead : ropeWitHead.val = [4294967296#i64, 0#i64] := Slice.from_val _ _
  have hhdv : ropeWitTable.head_dim.val = 2 := by norm_num [ropeWitTable]
  have hlen : ropeWitHead.val.length = 2 := by rw [hhead]; rfl
  have hcos : |4294967296 * ropeWitSc.cos.val| ≤ (2:ℤ)^65 := by
    rw [abs_mul]
    have h1 : |(4294967296 : ℤ)| = (2:ℤ)^32 := by norm_num
    rw [h1]
    calc (2:ℤ)^32 * |ropeWitSc.cos.val| ≤ (2:ℤ)^32 * (2:ℤ)^33 :=
          mul_le_mul_of_nonneg_left ropeWitSc_bounds.1 (by norm_num)
      _ = (2:ℤ)^65 := by norm_num
  have hsin : |4294967296 * ropeWitSc.sin.val| ≤ (2:ℤ)^65 := by
    rw [abs_mul]
    have h1 : |(4294967296 : ℤ)| = (2:ℤ)^32 := by norm_num
    rw [h1]
    calc (2:ℤ)^32 * |ropeWitSc.sin.val| ≤ (2:ℤ)^32 * (2:ℤ)^33 :=
          mul_le_mul_of_nonneg_left ropeWitSc_bounds.2 (by norm_num)
      _ = (2:ℤ)^65 := by norm_num
  have hfit : ∀ k, k < ropeWitHead.val.length →
      I64.min ≤ ropeBwdElem ropeWitHead.val 2 ropeWitD2.val ropeWitScAt k ∧
      ropeBwdElem ropeWitHead.val 2 ropeWitD2.val ropeWitScAt k ≤ I64.max := by
    intro k hk
    have hk2 : k < 2 := by rw [hlen] at hk; exact hk
    interval_cases k
    · have hval : ropeBwdElem ropeWitHead.val 2 ropeWitD2.val ropeWitScAt 0 =
          roundShift32 (4294967296 * ropeWitSc.cos.val) := by
        unfold ropeBwdElem ropeBwdPair
        rw [hhead]
        norm_num [ropeWitD2, ropeWitScAt]
      rw [hval]
      exact ropeBwd_roundShift_bounds _ hcos
    · have hval : ropeBwdElem ropeWitHead.val 2 ropeWitD2.val ropeWitScAt 1 =
          roundShift32 (-(4294967296 * ropeWitSc.sin.val)) := by
        unfold ropeBwdElem ropeBwdPair
        rw [hhead]
        norm_num [ropeWitD2, ropeWitScAt]
      rw [hval]
      have hneg : |-(4294967296 * ropeWitSc.sin.val)| ≤ (2:ℤ)^65 := by
        rw [abs_neg]; exact hsin
      exact ropeBwd_roundShift_bounds _ hneg
  obtain ⟨dx, hrun, hmap⟩ := rope_bwd_refines ropeWitTable ropeWitHead ropeWitPos ropeWitD2
    ropeWitScAt ropeWitTable_domain rope_head_domain_witness.1 ropeWitTable_htrigS hfit
  refine ⟨dx, hrun, ?_⟩
  rw [hmap, hhdv, hlen]
