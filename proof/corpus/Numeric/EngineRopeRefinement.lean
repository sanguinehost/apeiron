/-
Refinement of the extracted fixed-point `RoPE` head rotation.

The Rust body is `src/engine/rope.rs`. `RopeTable::rotate_head` rotates one `head_dim`-wide head in
place at token position `pos`: for each rotary index `j < head_dim/2` it forms the checked angle
`pos * freqs[j]`, asks `fixed_trig.fixed_sin_cos` for the Q32.32 pair `(cos, sin)`, and writes

  `x[j]      = round_shift_32 (x0 * cos - x1 * sin)`
  `x[j + d2] = round_shift_32 (x1 * cos + x0 * sin)`

with `x0 = x[j]`, `x1 = x[j + d2]`, `d2 = head_dim/2`, and `round_shift_32 v = (v + 2^31) >>> 32`
(`src/engine/value.rs`). `RopeTable::rotate` refuses with `VectorLength` when `q` and `k` differ in
length or the length is not a multiple of `head_dim`, and propagates `OutOfRange` when an angle
product overflows, an angle leaves the `fixed_sin_cos` window `[-4096, 4096]` radians, or a rotated
component leaves `i64`.

## Representation

An `i64` `v` denotes `toReal v = v.val / 2^32` (`Numeric.Scale`). The real target of the extracted
per-pair arithmetic is `ropeRot θ (toReal x0) (toReal x1)` at `θ = toReal angle`; the extracted
integer target is `ropePairInt x0 x1 cos sin` on top of `roundShift32`. The trig pair carries the
bound of `fixed_sin_cos_refines_proved` (`Numeric.Rope.Part4`):
`|toReal sc.cos - cos θ| <= eps*|cos θ| + trig_floor` and the `sin` analogue, `eps = 2^-20`,
`trig_floor = 2^-20`.

## Statements

* `value_round_shift_32_spec` — the extracted `value.round_shift_32` denotes `roundShift32`;
* `rope_div_round_nearest_abs_bound`, `roundShift32_half` — the shared rounding lemma, the
  `round_shift_32` instance of the `div_round_nearest_abs_bound` shape from
  `Numeric.EngineSiluRefinement`: `|roundShift32 v - v/2^32| <= 1/2`. That module is mid-edit in
  another lane and does not build, so the instance is stated and proved here rather than imported;
* `round_shift_pair_component` — the in-domain per-component bound against `ropeRot`, in the general
  form `(|a|*Ec + |b|*Es)/2^32 + 1/(2*2^32)` for arbitrary trig errors `Ec`, `Es`;
* `rope_pair_abs_bound`, `rope_pair_refines_trig`, `rope_pair_spec_proved` — the pair bound and its
  composition with `fixed_sin_cos_refines_proved`'s `eps`/floor bound;
* `rope_rotate_fixed_unfold`, `rope_rotate_fixed_vector_length`, `rotate_head_success_eq`,
  `rotate_head_refusal_eq` — the extracted body's shape and its refusal branches as equalities.

* `rotate_window_arith` — the `rotate_loop` window arithmetic: from `h < heads`,
  `heads = q.len / head_dim` and `head_dim ∣ q.len`, the three obligations the body's WP generates
  (`h * head_dim` and `base + head_dim` fit `Usize`, the `index_mut` range ends inside `q`);
* `RopeTrigAt`, `RopePairAt`, `RotateHeadInv` — the per-rotary-index trig data, the per-pair
  `ropePairInt` content and the `rotate_head_loop` state invariant carrying both;
* `rotate_loop_zero` — the zero-iteration `rotate_loop` instance, proved via `loop.spec_decr_nat`.

* `rotate_head_loop_spec` — totality of the inner loop: under the window and trig bounds it returns
  `ok r`, never `fail`, via `loop.spec_decr_nat` on the bounds-only `RotateHeadRunInv`. The
  per-step obligations are discharged: the `Usize` sums, the `Slice.index_usize`/`Slice.update`
  bounds at `j` and `j + d2`, the `I128` products (`i64_prod_max`/`i64_prod_min`), the
  `round_shift_32` half-adds, and the range-gated writes.

The residuals are the content characterisation (`RotateHeadInv`/`RopePairAt`: the written entries
are the `ropePairInt` rotation of the original pair, at integer level) and the loop-level
specification of a whole `head_dim`-wide head in `RopeHeadRefines`, which needs the outer
`rotate_loop` WP over the mutated slices; `rotate_window_arith` discharges its window obligations
and `rotate_loop_zero` its zero-iteration instance.
-/
import Numeric.Rope.Part4
import Numeric.RopeTurns

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Specification

The real and extracted targets of the per-pair arithmetic, from `rope.rs`'s doc comment. -/

/-- The real fixed-point `RoPE` rotation of the pair `(x0, x1)` by angle `θ`:
`(x0·cos θ − x1·sin θ, x1·cos θ + x0·sin θ)`. -/
def ropeRot (θ x0 x1 : ℝ) : ℝ × ℝ :=
  (x0 * Real.cos θ - x1 * Real.sin θ, x1 * Real.cos θ + x0 * Real.sin θ)

/-- The extracted `value.round_shift_32` value on an `i128` integer: `(v + 2^31) >>> 32`. -/
def roundShift32 (v : ℤ) : ℤ := (v + 2147483648) >>> 32

/-- The extracted per-pair integer rotation with `c = sc.cos.val`, `s = sc.sin.val`:
`(round_shift_32 (x0*c - x1*s), round_shift_32 (x1*c + x0*s))`. -/
def ropePairInt (x0 x1 c s : ℤ) : ℤ × ℤ :=
  (roundShift32 (x0 * c - x1 * s), roundShift32 (x1 * c + x0 * s))

/-- The extracted per-pair arithmetic, stated as the pair of integer outputs `rope.rs` writes, with
`turn` the frequency turn fraction, `pos` the position, and `angle = toRad (lift (phase turn pos))`.
This is the in-domain specification the refinement below proves component-wise. -/
def RopePairSpec : Prop :=
  ∀ (x0 x1 : I64) (turn : U64) (pos : U32) (angle : I64) (sc : fixed_trig.SinCos) (θ : ℝ),
    angle.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift (Numeric.RopeTurns.phase turn.val pos.val)) →
    word_fixed_sin_cos angle = ok (core.result.Result.Ok sc) →
    θ = toReal angle →
    |((ropePairInt x0.val x1.val sc.cos.val sc.sin.val).1 : ℝ) / q32 -
        ((x0.val : ℝ) / q32 * Real.cos θ - (x1.val : ℝ) / q32 * Real.sin θ)| ≤
      (((|x0.val| : ℤ) : ℝ) + ((|x1.val| : ℤ) : ℝ)) / q32 * (epsExp + trig_floor) +
        1 / (2 * q32) ∧
    |((ropePairInt x0.val x1.val sc.cos.val sc.sin.val).2 : ℝ) / q32 -
        ((x1.val : ℝ) / q32 * Real.cos θ + (x0.val : ℝ) / q32 * Real.sin θ)| ≤
      (((|x0.val| : ℤ) : ℝ) + ((|x1.val| : ℤ) : ℝ)) / q32 * (epsExp + trig_floor) +
        1 / (2 * q32)

/-- The admissible domain of the head rotation, read off `rope.rs`: the table's `head_dim` is at
least two and even (`RopeTable::new` and `RopeTable::from_freqs` refuse zero, odd or unit
`head_dim`), the turns table has exactly the `head_dim/2` rotary entries, the
two static guards pass (`q.len = k.len`, `head_dim ∣ q.len`), and every rotary index
`j < head_dim/2` carries a turn `turns[j]`. -/
def RopeHeadDomain (table : rope.RopeTable) (q k : Slice I64) (_pos : U32) : Prop :=
  Slice.len q = Slice.len k ∧
  2 ≤ table.head_dim.val ∧
  table.head_dim.val % 2 = 0 ∧
  (Slice.len q).val % table.head_dim.val = 0 ∧
  table.head_dim.val / 2 ≤ table.turns.val.length ∧
  ∀ j : ℕ, j < table.head_dim.val / 2 → j < table.turns.val.length →
    ∃ t : U64, table.turns.val[j]? = some t

/-- The extracted body's domain/refusal specification. Below the static guards the result is either
the `Ok ()` success arm or the `OutOfRange` refusal — never `VectorLength` (both static guards pass),
never `BadHeadDim`/`TableLength`/`BadTheta` (`RopeTable::rotate` does not construct them), and not a
`fail` (under `RopeHeadDomain` every `head_dim`-wide window is in bounds). The per-head `ropeRot`
bound on the success arm is the residual named in the file header: it needs the
`rotate`/`rotate_head` loop WP specs, so this specification is stated (not proved) and the
pair-level bound above is what is landed. -/
def RopeHeadRefines : Prop :=
  ∀ (table : rope.RopeTable) (q k : Slice I64) (pos : U32),
    RopeHeadDomain table q k pos →
    (∃ q' k' : Slice I64,
        word_rope_rotate_fixed q k pos table =
          ok (core.result.Result.Ok (), q', k')) ∨
    (∃ q' k' : Slice I64,
        word_rope_rotate_fixed q k pos table =
          ok (core.result.Result.Err rope.RopeError.OutOfRange, q', k'))

/-! ## Rounding

`value.round_shift_32` is the extracted `(v + 2^31) >>> 32` on `i128`. Its real error is the
`round_shift_32` instance of the imported `div_round_nearest_abs_bound`
(`Numeric.EngineSiluRefinement`); the two are the same half-biased integer division, instanced below
rather than restated. -/

/-- The extracted rounding half-constant `2^31` evaluates. -/
theorem i128_shift_31 : (1#i128 <<< 31#i32) = ok (2147483648#i128) := by
  have h : (1#i128 <<< 31#i32) ⦃ z => z = (2147483648#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- The extracted `value.round_shift_32` body with the half-constant evaluated:
`(v + 2^31) >>> 32`. -/
theorem value_round_shift_32_eq (v : I128) :
    value.round_shift_32 v = (do let i1 ← v + (2147483648#i128); i1 >>> 32#i32) := by
  unfold value.round_shift_32
  rw [i128_shift_31]
  simp only [bind_tc_ok]

/-- The extracted `value.round_shift_32` denotes `roundShift32`. -/
theorem value_round_shift_32_raw (v : I128)
    (hlo : I128.min ≤ v.val + 2147483648) (hhi : v.val + 2147483648 ≤ I128.max) :
    value.round_shift_32 v ⦃ z => z.val = (v.val + 2147483648) >>> 32 ⦄ := by
  rw [value_round_shift_32_eq]
  step*

/-- The extracted `value.round_shift_32` denotes `roundShift32`. -/
theorem value_round_shift_32_spec (v : I128)
    (hlo : I128.min ≤ v.val + 2147483648) (hhi : v.val + 2147483648 ≤ I128.max) :
    value.round_shift_32 v ⦃ z => z.val = roundShift32 v.val ⦄ := by
  apply Aeneas.Std.WP.spec_mono (value_round_shift_32_raw v hlo hhi)
  intro z hz
  unfold roundShift32
  exact hz

/-- **Half-biased division.** For `d > 0`, `((n + d/2)/d)·d` is within `d/2` of `n`. This is the
`div_round_nearest_abs_bound` shape from `Numeric.EngineSiluRefinement`, stated and proved here as
the instance this lane needs (that module is mid-edit in another lane and does not build, so it is
not imported); `roundShift32_half` below is its `2^32` real counterpart. -/
theorem rope_div_round_nearest_abs_bound (n d : ℤ) (hd : 0 < d) :
    |((n + d / 2) / d) * d - n| ≤ d / 2 := by
  have hdne : d ≠ 0 := ne_of_gt hd
  set q : ℤ := (n + d / 2) / d with hq
  set s : ℤ := (n + d / 2) % d with hs
  set f : ℤ := d / 2 with hf
  set e : ℤ := d % 2 with he
  have hs0 : 0 ≤ s := by rw [hs]; exact Int.emod_nonneg _ hdne
  have hslt : s < d := by rw [hs]; exact Int.emod_lt_of_pos _ hd
  have he0 : 0 ≤ e := by rw [he]; exact Int.emod_nonneg _ (by norm_num)
  have he1 : e < 2 := by rw [he]; exact Int.emod_lt_of_pos _ (by norm_num)
  have h2f : 2 * f = d - e := by
    rw [hf, he]
    have := Int.mul_ediv_add_emod d 2
    linarith
  have hdecomp : q * d + s = n + f := by
    rw [hq, hs, hf]
    have := Int.mul_ediv_add_emod (n + d / 2) d
    linarith
  have hqdn : q * d - n = f - s := by linarith
  rw [hqdn, abs_le]
  constructor <;> omega

/-- **Shared rounding lemma.** `roundShift32` is the half-biased division by `2^32`: it is within
`1/2` of `v / 2^32`. This is the real `2^32` instance of `rope_div_round_nearest_abs_bound`
(`n = v`, `d = 2^32`): the integer quotient `z = (v + 2^31) / 2^32` satisfies `0 <= v + 2^31 -
z*2^32 < 2^32`, hence `-1/2 < z - v/2^32 <= 1/2`. -/
theorem roundShift32_half (v : ℤ) :
    |((roundShift32 v : ℤ) : ℝ) - (v : ℝ) / q32| ≤ 1 / 2 := by
  have hD : (0 : ℤ) < 4294967296 := by norm_num
  have hDne : (4294967296 : ℤ) ≠ 0 := by norm_num
  have hDℝ : (4294967296 : ℝ) = q32 := by norm_num [q32]
  have hQ : (0 : ℝ) < (4294967296 : ℝ) := by norm_num
  have hdiv : roundShift32 v = (v + 2147483648) / 4294967296 := by
    unfold roundShift32
    rw [Int.shiftRight_eq_div_pow]
    norm_num
  rw [hdiv]
  set z : ℤ := (v + 2147483648) / 4294967296 with hz
  have hlo : z * 4294967296 ≤ v + 2147483648 := by
    have h := Int.ediv_mul_le (v + 2147483648) hDne
    rwa [← hz] at h
  have hhi : v + 2147483648 < (z + 1) * 4294967296 := by
    have hlti : (v + 2147483648) / 4294967296 < z + 1 := by rw [hz]; omega
    exact (Int.ediv_lt_iff_lt_mul hD).mp hlti
  have hloR : (z : ℝ) * (4294967296 : ℝ) ≤ (v : ℝ) + 2147483648 := by
    exact_mod_cast hlo
  have hhiR : (v : ℝ) + 2147483648 < ((z : ℝ) + 1) * (4294967296 : ℝ) := by
    exact_mod_cast hhi
  have hH : (2147483648 : ℝ) = (4294967296 : ℝ) / 2 := by norm_num
  have hle : (z : ℝ) - (v : ℝ) / (4294967296 : ℝ) ≤ 1 / 2 := by
    have hkey : ((z : ℝ) - (v : ℝ) / (4294967296 : ℝ)) * (4294967296 : ℝ) =
        (z : ℝ) * (4294967296 : ℝ) - (v : ℝ) := by
      field_simp
    rw [show (1 : ℝ) / 2 = (4294967296 : ℝ) / 2 / (4294967296 : ℝ) by norm_num]
    rw [le_div_iff₀ hQ]
    rw [hkey]
    linarith [hloR]
  have hge : -(1 / 2) ≤ (z : ℝ) - (v : ℝ) / (4294967296 : ℝ) := by
    rw [show -(1 / 2 : ℝ) = (-((4294967296 : ℝ) / 2)) / (4294967296 : ℝ) by norm_num]
    rw [div_le_iff₀ hQ]
    have hkey : ((z : ℝ) - (v : ℝ) / (4294967296 : ℝ)) * (4294967296 : ℝ) =
        (z : ℝ) * (4294967296 : ℝ) - (v : ℝ) := by
      field_simp
    rw [hkey]
    linarith [hhiR, hH]
  rw [hDℝ] at hle hge
  exact abs_le.mpr ⟨hge, hle⟩

/-! ## Turn ring refinement

Lemmas linking the extracted `fixed_trig.turn_to_radians` and the turn-phase
multiplication to `Numeric.RopeTurns`. -/

theorem lift_eq_bmod (x : U64) :
    Numeric.RopeTurns.lift x.val = (x.val : ℤ).bmod (2 ^ 64) := by
  unfold Numeric.RopeTurns.lift
  rw [Int.bmod_def]
  have hxval : (x.val : ℤ) % (2 ^ 64 : ℕ) = (x.val : ℤ) := by
    apply Int.emod_eq_of_lt (by omega)
    have := x.hmax
    dsimp [U64.max, UScalar.max, U64.numBits] at this
    omega
  change (if x.val < 2 ^ 63 then (x.val : ℤ) else (x.val : ℤ) - 2 ^ 64) =
    if (x.val : ℤ) % (2 ^ 64 : ℕ) < ((2 ^ 64 : ℕ) + 1 : ℤ) / 2 then (x.val : ℤ) % (2 ^ 64 : ℕ)
    else (x.val : ℤ) % (2 ^ 64 : ℕ) - (2 ^ 64 : ℕ)
  rw [hxval]
  have hmid : (((2 ^ 64 : ℕ) + 1 : ℤ) / 2) = (2 ^ 63 : ℤ) := by norm_num
  rw [hmid]
  split_ifs
  all_goals omega

theorem u64_to_i64_hcast_val (x : U64) : (UScalar.hcast .I64 x).val = Numeric.RopeTurns.lift x.val := by
  rw [UScalar.hcast_val_eq]
  exact (lift_eq_bmod x).symm

theorem i64_to_i128_cast_val (x : I64) : (IScalar.cast .I128 x).val = x.val := by
  rw [IScalar.cast_val_eq]
  have hmin : min (IScalarTy.I128.numBits) (IScalarTy.I64.numBits) = 64 := by decide
  rw [hmin]
  exact bmod_pow_numBits_eq_of_lt .I64 x.val x.hmin x.hmax

theorem i128_shift_63 : (1#i128 <<< 63#i32) = ok (9223372036854775808#i128) := by
  have h : (1#i128 <<< 63#i32) ⦃ z => z = (9223372036854775808#i128) ⦄ := by
    step
    apply IScalar.eq_imp
    rw [z_post]
    norm_num [I128.size, IScalar.size, I128.numBits, Int.shiftLeft_eq, Int.bmod]
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- **Extracted `turn_to_radians` WP specification.** The extracted `fixed_trig.turn_to_radians`
denotes `Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val)`. -/
@[step]
theorem turn_to_radians_spec (phase : U64) :
    fixed_trig.turn_to_radians phase ⦃ rad =>
      rad.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val) ⦄ := by
  unfold fixed_trig.turn_to_radians
  simp only [lift, bind_tc_ok]
  have hphase_lt : phase.val < 2 ^ 64 := by
    have := phase.hmax
    dsimp [U64.max, UScalar.max, U64.numBits] at this
    omega
  have hlift_range := Numeric.RopeTurns.lift_range phase.val hphase_lt
  have hTWO_PI_val : fixed_trig.TWO_PI_Q32.val = 26986075409 := by
    unfold fixed_trig.TWO_PI_Q32; rfl
  have hs_val : (UScalar.hcast IScalarTy.I64 phase).val = Numeric.RopeTurns.lift phase.val :=
    u64_to_i64_hcast_val phase
  have hi_val : (IScalar.cast IScalarTy.I128 (UScalar.hcast IScalarTy.I64 phase)).val =
      Numeric.RopeTurns.lift phase.val := by
    rw [i64_to_i128_cast_val, hs_val]
  have hi1_val : (IScalar.cast IScalarTy.I128 fixed_trig.TWO_PI_Q32).val = 26986075409 := by
    rw [i64_to_i128_cast_val, hTWO_PI_val]
  have hmin128 : IScalar.min .I128 ≤ (Numeric.RopeTurns.lift phase.val) * 26986075409 := by
    have : -(2 ^ 63 : ℤ) * 26986075409 ≤ (Numeric.RopeTurns.lift phase.val) * 26986075409 :=
      Int.mul_le_mul_of_nonneg_right hlift_range.1 (by norm_num)
    rw [show IScalar.min .I128 = -(2 ^ 127 : ℤ) by norm_num [IScalar.min, I128.min, I128.numBits]]
    linarith
  have hmax128 : (Numeric.RopeTurns.lift phase.val) * 26986075409 ≤ IScalar.max .I128 := by
    have : (Numeric.RopeTurns.lift phase.val) * 26986075409 ≤ (2 ^ 63 - 1) * 26986075409 :=
      Int.mul_le_mul_of_nonneg_right (by omega) (by norm_num)
    rw [show IScalar.max .I128 = (2 ^ 127 - 1 : ℤ) by norm_num [IScalar.max, I128.max, I128.numBits]]
    linarith
  refine Aeneas.Std.WP.spec_bind (IScalar.mul_spec (by rw [hi_val, hi1_val]; exact hmin128)
    (by rw [hi_val, hi1_val]; exact hmax128)) ?_
  intro i2 hi2
  rw [i128_shift_63]
  simp only [bind_tc_ok]
  have hi2_val : i2.val = (Numeric.RopeTurns.lift phase.val) * 26986075409 := by
    rw [hi2, hi_val, hi1_val]
  have h63_val : (9223372036854775808#i128).val = 9223372036854775808 := rfl
  have hadd_min : IScalar.min .I128 ≤ i2.val + (9223372036854775808#i128).val := by
    rw [hi2_val, h63_val]
    have : -(2 ^ 63 : ℤ) * 26986075409 ≤ (Numeric.RopeTurns.lift phase.val) * 26986075409 :=
      Int.mul_le_mul_of_nonneg_right hlift_range.1 (by norm_num)
    rw [show IScalar.min .I128 = -(2 ^ 127 : ℤ) by norm_num [IScalar.min, I128.min, I128.numBits]]
    linarith
  have hadd_max : i2.val + (9223372036854775808#i128).val ≤ IScalar.max .I128 := by
    rw [hi2_val, h63_val]
    have : (Numeric.RopeTurns.lift phase.val) * 26986075409 ≤ (2 ^ 63 - 1) * 26986075409 :=
      Int.mul_le_mul_of_nonneg_right (by omega) (by norm_num)
    rw [show IScalar.max .I128 = (2 ^ 127 - 1 : ℤ) by norm_num [IScalar.max, I128.max, I128.numBits]]
    linarith
  refine Aeneas.Std.WP.spec_bind (IScalar.add_spec hadd_min hadd_max) ?_
  intro i4 hi4
  refine Aeneas.Std.WP.spec_bind (IScalar.ShiftRight_IScalar_spec i4 64#i32 (by norm_num) (by norm_num [IScalarTy.numBits])) ?_
  intro r hr
  simp only [WP.spec_ok]
  rw [IScalar.cast_val_eq]
  have hmin : min (IScalarTy.I64.numBits) (IScalarTy.I128.numBits) = 64 := by decide
  rw [hmin]
  have hrval : r.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val) := by
    rw [hr.1, hi4, hi2_val]
    unfold Numeric.RopeTurns.toRad
    rw [Int.shiftRight_eq_div_pow]
    have hT : Numeric.RopeTurns.TWO_PI_Q32 = 26986075409 := rfl
    rw [hT]
    have h64 : (64#i32).toNat = 64 := rfl
    rw [h64, h63_val]
    norm_num
  have htoRad_bd := Numeric.RopeTurns.turn_to_radians_window_sharp (Numeric.RopeTurns.lift phase.val)
    hlift_range.1 hlift_range.2
  have hT : Numeric.RopeTurns.TWO_PI_Q32 = 26986075409 := rfl
  rw [hT] at htoRad_bd
  have htoRad_lo : -(2 ^ 63 : ℤ) ≤ Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val) := by
    have := (abs_le.mp htoRad_bd).1
    omega
  have htoRad_hi : Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val) < (2 ^ 63 : ℤ) := by
    have := (abs_le.mp htoRad_bd).2
    omega
  rw [hrval]
  exact bmod_pow_numBits_eq_of_lt .I64 _ htoRad_lo htoRad_hi

/-- **`turn_to_radians` refinement.** For any phase `u64`, `fixed_trig.turn_to_radians` succeeds
and produces an `i64` whose integer value is `Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val)`. -/
theorem turn_to_radians_refines (phase : U64) :
    ∃ rad : I64, fixed_trig.turn_to_radians phase = ok rad ∧
      rad.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift phase.val) := by
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 (turn_to_radians_spec phase) with ⟨rad, hrad, hv⟩
  exact ⟨rad, hrad, hv⟩

theorem u64_max_val : (core.num.U64.MAX : U64).val = 2 ^ 64 - 1 := by
  decide

theorem u128_cast_u64_val (x : U64) : (UScalar.cast .U128 x).val = x.val := by
  rw [UScalar.cast_val_eq]
  have h128 : (2 : ℕ) ^ (UScalarTy.U128.numBits) = 2 ^ 128 := by rfl
  rw [h128]
  have htq' := x.hmax; dsimp [U64.max, UScalar.max, U64.numBits] at htq'
  apply Nat.mod_eq_of_lt; omega

theorem u128_cast_u32_val (x : U32) : (UScalar.cast .U128 x).val = x.val := by
  rw [UScalar.cast_val_eq]
  have h128 : (2 : ℕ) ^ (UScalarTy.U128.numBits) = 2 ^ 128 := by rfl
  rw [h128]
  have hpos := x.hmax; dsimp [U32.max, UScalar.max, U32.numBits] at hpos
  apply Nat.mod_eq_of_lt; omega

theorem turn_phase_val_eq (turn : U64) (pos : U32) (i3 : U128) (hi3 : i3.val = turn.val * pos.val) :
    (UScalar.cast .U64 (i3 &&& UScalar.cast .U128 core.num.U64.MAX)).val =
      Numeric.RopeTurns.phase turn.val pos.val := by
  rw [UScalar.cast_val_eq, UScalar.val_and, UScalar.cast_val_eq, hi3]
  have h64 : (core.num.U64.MAX : U64).val % 2 ^ (UScalarTy.U128.numBits) = 2 ^ 64 - 1 := by
    rw [u64_max_val]
    decide
  rw [h64]
  have hand := Nat.and_two_pow_sub_one_eq_mod (turn.val * pos.val) 64
  rw [hand]
  have hnum : UScalarTy.U64.numBits = 64 := rfl
  rw [hnum]
  have hmod : (turn.val * pos.val) % 2 ^ 64 % 2 ^ 64 = (turn.val * pos.val) % 2 ^ 64 :=
    Nat.mod_mod (turn.val * pos.val) (2 ^ 64)
  rw [hmod]
  rfl

/-- **Phase computation refines `Numeric.RopeTurns.phase`.** The extracted
`(turn as u128 * pos as u128) & u64::MAX as u128 as u64` denotes `Numeric.RopeTurns.phase turn pos`. -/
theorem turn_phase_refines (turn : U64) (pos : U32) :
    (do
      let i1 ← lift (UScalar.cast .U128 turn)
      let i2 ← lift (UScalar.cast .U128 pos)
      let i3 ← i1 * i2
      let i4 ← lift (UScalar.cast .U128 core.num.U64.MAX)
      let i5 ← lift (i3 &&& i4)
      lift (UScalar.cast .U64 i5) : Result U64) ⦃ phase =>
        phase.val = Numeric.RopeTurns.phase turn.val pos.val ⦄ := by
  simp only [lift, bind_tc_ok]
  have hturn_lt : turn.val < 2 ^ 64 := by
    have := turn.hmax; dsimp [U64.max, UScalar.max, U64.numBits] at this; omega
  have hpos_lt : pos.val < 2 ^ 32 := by
    have := pos.hmax; dsimp [U32.max, UScalar.max, U32.numBits] at this; omega
  have hmul_lt : turn.val * pos.val ≤ UScalar.max .U128 := by
    rw [UScalar.max_U128_eq]
    have : turn.val ≤ 2 ^ 64 - 1 := by omega
    have : pos.val ≤ 2 ^ 32 - 1 := by omega
    nlinarith
  have hcast1 : (UScalar.cast .U128 turn).val = turn.val := u128_cast_u64_val turn
  have hcast2 : (UScalar.cast .U128 pos).val = pos.val := u128_cast_u32_val pos
  refine Aeneas.Std.WP.spec_bind (UScalar.mul_spec (by rw [hcast1, hcast2]; exact hmul_lt)) ?_
  intro i3 hi3
  simp only [WP.spec_ok]
  exact turn_phase_val_eq turn pos i3 (by rw [hi3, hcast1, hcast2])

/-- The angle derived from `turn` and `pos` is unconditionally in `trig_window`. -/
theorem turn_angle_trig_window (turn : U64) (pos : U32) (angle : I64)
    (hangle : angle.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift (Numeric.RopeTurns.phase turn.val pos.val))) :
    trig_window angle := by
  have hphase_lt := Numeric.RopeTurns.phase_lt turn.val pos.val
  have hlift_range := Numeric.RopeTurns.lift_range (Numeric.RopeTurns.phase turn.val pos.val) hphase_lt
  have hwin := Numeric.RopeTurns.turn_to_radians_window
    (Numeric.RopeTurns.lift (Numeric.RopeTurns.phase turn.val pos.val)) hlift_range.1 hlift_range.2
  unfold trig_window
  rw [hangle]
  have h4096 : (4096 : ℤ) * (2 : ℤ) ^ 32 = 17592186044416 := by norm_num
  rw [h4096] at hwin
  exact abs_le.mp hwin

/-! ## Per-pair arithmetic

The extracted pair `A = x0*cos - x1*sin` is a product of Q32.32 factors: its exact real value is
`A / 2^64`, and `round_shift_32 A ≈ A / 2^32` in Q32.32 units. The composition below splits the
error into that rounding half-unit and the trig pair's error propagated by `x0`, `x1`. -/

/-- **Per-component bound.** For arbitrary trig errors `Ec` (on `cos θ`) and `Es` (on `sin θ`), the
extracted `round_shift_32 (a*c - b*s)` is within `(|a|*Ec + |b|*Es)/2^32 + 1/(2*2^32)` of
`a/2^32 * cos θ - b/2^32 * sin θ`. -/
theorem round_shift_pair_component (a b c s : ℤ) (θ Ec Es : ℝ)
    (hc : |(c : ℝ) / q32 - Real.cos θ| ≤ Ec)
    (hs : |(s : ℝ) / q32 - Real.sin θ| ≤ Es) :
    |((roundShift32 (a * c - b * s) : ℤ) : ℝ) / q32 -
        ((a : ℝ) / q32 * Real.cos θ - (b : ℝ) / q32 * Real.sin θ)| ≤
      (((|a| : ℤ) : ℝ) * Ec + ((|b| : ℤ) : ℝ) * Es) / q32 + 1 / (2 * q32) := by
  have hq := q32_pos
  have h1 : |((roundShift32 (a * c - b * s) : ℤ) : ℝ) / q32 -
      (((a * c - b * s : ℤ) : ℝ)) / q32 / q32| ≤ 1 / (2 * q32) := by
    have h := roundShift32_half (a * c - b * s)
    have hrw : ((roundShift32 (a * c - b * s) : ℤ) : ℝ) / q32 -
        (((a * c - b * s : ℤ) : ℝ)) / q32 / q32 =
        (((roundShift32 (a * c - b * s) : ℤ) : ℝ) - (((a * c - b * s : ℤ) : ℝ)) / q32) / q32 := by
      ring
    rw [hrw, abs_div, abs_of_pos hq]
    rw [show (1 : ℝ) / (2 * q32) = (1 / 2) / q32 by ring]
    rw [div_le_div_iff_of_pos_right hq]
    exact h
  have h2 : |(((a * c - b * s : ℤ) : ℝ)) / q32 / q32 -
      ((a : ℝ) / q32 * Real.cos θ - (b : ℝ) / q32 * Real.sin θ)| ≤
      (((|a| : ℤ) : ℝ) * Ec + ((|b| : ℤ) : ℝ) * Es) / q32 := by
    have he : (((a * c - b * s : ℤ) : ℝ)) / q32 / q32 -
        ((a : ℝ) / q32 * Real.cos θ - (b : ℝ) / q32 * Real.sin θ) =
        ((a : ℝ) / q32) * ((c : ℝ) / q32 - Real.cos θ) -
          ((b : ℝ) / q32) * ((s : ℝ) / q32 - Real.sin θ) := by
      push_cast
      ring
    rw [he]
    have habs_a : |((a : ℝ) / q32)| = ((|a| : ℤ) : ℝ) / q32 := by
      rw [abs_div, abs_of_pos hq, ← Int.cast_abs]
    have habs_b : |((b : ℝ) / q32)| = ((|b| : ℤ) : ℝ) / q32 := by
      rw [abs_div, abs_of_pos hq, ← Int.cast_abs]
    have hTa : |((a : ℝ) / q32) * ((c : ℝ) / q32 - Real.cos θ)| ≤
        ((|a| : ℤ) : ℝ) / q32 * Ec := by
      rw [abs_mul, habs_a]
      exact mul_le_mul_of_nonneg_left hc (by positivity)
    have hTb : |((b : ℝ) / q32) * ((s : ℝ) / q32 - Real.sin θ)| ≤
        ((|b| : ℤ) : ℝ) / q32 * Es := by
      rw [abs_mul, habs_b]
      exact mul_le_mul_of_nonneg_left hs (by positivity)
    calc |((a : ℝ) / q32) * ((c : ℝ) / q32 - Real.cos θ) -
          ((b : ℝ) / q32) * ((s : ℝ) / q32 - Real.sin θ)|
        ≤ |((a : ℝ) / q32) * ((c : ℝ) / q32 - Real.cos θ)| +
          |((b : ℝ) / q32) * ((s : ℝ) / q32 - Real.sin θ)| := by
            simpa using abs_sub_le ((a : ℝ) / q32 * ((c : ℝ) / q32 - Real.cos θ)) (0 : ℝ)
              ((b : ℝ) / q32 * ((s : ℝ) / q32 - Real.sin θ))
      _ ≤ ((|a| : ℤ) : ℝ) / q32 * Ec + ((|b| : ℤ) : ℝ) / q32 * Es := add_le_add hTa hTb
      _ = (((|a| : ℤ) : ℝ) * Ec + ((|b| : ℤ) : ℝ) * Es) / q32 := by ring
  calc |((roundShift32 (a * c - b * s) : ℤ) : ℝ) / q32 -
        ((a : ℝ) / q32 * Real.cos θ - (b : ℝ) / q32 * Real.sin θ)|
      ≤ |((roundShift32 (a * c - b * s) : ℤ) : ℝ) / q32 -
          (((a * c - b * s : ℤ) : ℝ)) / q32 / q32| +
        |(((a * c - b * s : ℤ) : ℝ)) / q32 / q32 -
          ((a : ℝ) / q32 * Real.cos θ - (b : ℝ) / q32 * Real.sin θ)| := abs_sub_le _ _ _
    _ ≤ 1 / (2 * q32) + (((|a| : ℤ) : ℝ) * Ec + ((|b| : ℤ) : ℝ) * Es) / q32 :=
        add_le_add h1 h2
    _ = (((|a| : ℤ) : ℝ) * Ec + ((|b| : ℤ) : ℝ) * Es) / q32 + 1 / (2 * q32) := by ring

/-- **Pair bound.** The two extracted components are within the per-component bound of
`round_shift_pair_component`, at `(a, b) = (x0, x1)` and `(a, b) = (x1, -x0)`. -/
theorem rope_pair_abs_bound (x0 x1 c s : ℤ) (θ Ec Es : ℝ)
    (hc : |(c : ℝ) / q32 - Real.cos θ| ≤ Ec)
    (hs : |(s : ℝ) / q32 - Real.sin θ| ≤ Es) :
    |((ropePairInt x0 x1 c s).1 : ℝ) / q32 -
        ((x0 : ℝ) / q32 * Real.cos θ - (x1 : ℝ) / q32 * Real.sin θ)| ≤
      (((|x0| : ℤ) : ℝ) * Ec + ((|x1| : ℤ) : ℝ) * Es) / q32 + 1 / (2 * q32) ∧
    |((ropePairInt x0 x1 c s).2 : ℝ) / q32 -
        ((x1 : ℝ) / q32 * Real.cos θ + (x0 : ℝ) / q32 * Real.sin θ)| ≤
      (((|x1| : ℤ) : ℝ) * Ec + ((|x0| : ℤ) : ℝ) * Es) / q32 + 1 / (2 * q32) := by
  unfold ropePairInt
  constructor
  · exact round_shift_pair_component x0 x1 c s θ Ec Es hc hs
  · have h := round_shift_pair_component x1 (-x0) c s θ Ec Es hc hs
    have hL : x1 * c - (-x0) * s = x1 * c + x0 * s := by ring
    have hR : (x1 : ℝ) / q32 * Real.cos θ - (((-x0 : ℤ) : ℝ)) / q32 * Real.sin θ =
        (x1 : ℝ) / q32 * Real.cos θ + (x0 : ℝ) / q32 * Real.sin θ := by
      push_cast
      ring
    rw [hL, hR] at h
    rw [abs_neg] at h
    exact h

/-- **Composition with the trig bound.** From `fixed_sin_cos_refines_proved`'s per-output bound the
pair error collapses to `(|x0.val| + |x1.val|)/2^32 * (eps + trig_floor) + 1/(2*2^32)`: the relative
term uses `|cos θ|, |sin θ| <= 1`, the absolute term is the trig floor. -/
theorem rope_pair_refines_trig (x0 x1 : I64) (sc : fixed_trig.SinCos) (θ : ℝ)
    (hc : |toReal sc.cos - Real.cos θ| ≤ epsExp * |Real.cos θ| + trig_floor)
    (hs : |toReal sc.sin - Real.sin θ| ≤ epsExp * |Real.sin θ| + trig_floor) :
    |((ropePairInt x0.val x1.val sc.cos.val sc.sin.val).1 : ℝ) / q32 -
        ((x0.val : ℝ) / q32 * Real.cos θ - (x1.val : ℝ) / q32 * Real.sin θ)| ≤
      (((|x0.val| : ℤ) : ℝ) + ((|x1.val| : ℤ) : ℝ)) / q32 * (epsExp + trig_floor) +
        1 / (2 * q32) ∧
    |((ropePairInt x0.val x1.val sc.cos.val sc.sin.val).2 : ℝ) / q32 -
        ((x1.val : ℝ) / q32 * Real.cos θ + (x0.val : ℝ) / q32 * Real.sin θ)| ≤
      (((|x0.val| : ℤ) : ℝ) + ((|x1.val| : ℤ) : ℝ)) / q32 * (epsExp + trig_floor) +
        1 / (2 * q32) := by
  have hEqc : epsExp * |Real.cos θ| + trig_floor ≤ epsExp + trig_floor := by
    have heps : (0 : ℝ) ≤ epsExp := by unfold epsExp; positivity
    have := Real.abs_cos_le_one θ
    nlinarith [mul_le_mul_of_nonneg_left this heps]
  have hEqs : epsExp * |Real.sin θ| + trig_floor ≤ epsExp + trig_floor := by
    have heps : (0 : ℝ) ≤ epsExp := by unfold epsExp; positivity
    have := Real.abs_sin_le_one θ
    nlinarith [mul_le_mul_of_nonneg_left this heps]
  obtain ⟨h1, h2⟩ := rope_pair_abs_bound x0.val x1.val sc.cos.val sc.sin.val θ
    (epsExp * |Real.cos θ| + trig_floor) (epsExp * |Real.sin θ| + trig_floor)
    (by simpa [toReal] using hc) (by simpa [toReal] using hs)
  have hA : (0 : ℝ) ≤ ((|x0.val| : ℤ) : ℝ) := by positivity
  have hB : (0 : ℝ) ≤ ((|x1.val| : ℤ) : ℝ) := by positivity
  have key : ∀ (A B Ec Es : ℝ), 0 ≤ A → 0 ≤ B → Ec ≤ epsExp + trig_floor →
      Es ≤ epsExp + trig_floor →
      (A * Ec + B * Es) / q32 + 1 / (2 * q32) ≤
        (A + B) / q32 * (epsExp + trig_floor) + 1 / (2 * q32) := by
    intro A B Ec Es hA' hB' hEc hEs
    have hsum : A * Ec + B * Es ≤ (A + B) * (epsExp + trig_floor) := by
      nlinarith [mul_le_mul_of_nonneg_left hEc hA', mul_le_mul_of_nonneg_left hEs hB']
    have hq := q32_pos
    have hdiv : (A * Ec + B * Es) / q32 ≤ (A + B) * (epsExp + trig_floor) / q32 :=
      div_le_div_of_nonneg_right hsum (le_of_lt hq)
    have hrw : (A + B) * (epsExp + trig_floor) / q32 =
        (A + B) / q32 * (epsExp + trig_floor) := by ring
    linarith [hdiv, hrw.le, hrw.ge]
  constructor
  · refine le_trans h1 ?_
    exact key ((|x0.val| : ℤ) : ℝ) ((|x1.val| : ℤ) : ℝ)
      (epsExp * |Real.cos θ| + trig_floor) (epsExp * |Real.sin θ| + trig_floor)
      hA hB hEqc hEqs
  · refine le_trans h2 ?_
    rw [show ((|x0.val| : ℤ) : ℝ) + ((|x1.val| : ℤ) : ℝ) =
        ((|x1.val| : ℤ) : ℝ) + ((|x0.val| : ℤ) : ℝ) by ring]
    exact key ((|x1.val| : ℤ) : ℝ) ((|x0.val| : ℤ) : ℝ)
      (epsExp * |Real.cos θ| + trig_floor) (epsExp * |Real.sin θ| + trig_floor)
      hB hA hEqc hEqs

/-- `RopePairSpec` holds: the extracted per-pair arithmetic refines the real rotation `ropeRot` at
`θ = toReal angle` within the `fixed_sin_cos_refines_proved` `eps`/floor bound. -/
theorem rope_pair_spec_proved : RopePairSpec := by
  unfold RopePairSpec
  intro x0 x1 turn pos angle sc θ hangle hsc hθ
  subst hθ
  have hwin : trig_window angle := turn_angle_trig_window turn pos angle hangle
  obtain ⟨sc', hsc', hs, hc⟩ := fixed_sin_cos_refines_proved angle hwin
  have hsc'' : sc = sc' := by
    have h := hsc'
    rw [hsc] at h
    simpa using h
  subst hsc''
  exact rope_pair_refines_trig x0 x1 sc (toReal angle) hc hs

/-! ## Extracted body: shape and refusal branches

`value.round_shift_32`'s spec above is on the value the pair arithmetic calls; the equalities below
name the extracted body's static guards and the `rotate_head` success/refusal split, so the refusal
branches are equalities rather than implications. -/

/-- The public entry point is its inner body: `word_rope_rotate_fixed` is definitionally
`rope.rope_rotate_fixed`. -/
theorem rope_rotate_fixed_unfold (q k : Slice I64) (pos : U32) (table : rope.RopeTable) :
    word_rope_rotate_fixed q k pos table = rope.rope_rotate_fixed q k pos table := rfl

/-- The first static guard of `RopeTable::rotate` as an equality: when `q` and `k` differ in length
the extracted body returns `Err VectorLength` with both slices unchanged. -/
theorem rope_rotate_fixed_vector_length (q k : Slice I64) (pos : U32) (table : rope.RopeTable)
    (h : Slice.len q ≠ Slice.len k) :
    rope.rope_rotate_fixed q k pos table =
      ok (core.result.Result.Err rope.RopeError.VectorLength, q, k) := by
  unfold rope.rope_rotate_fixed rope.RopeTable.rotate
  dsimp only
  rw [if_pos (by rw [bne_iff_ne]; exact h)]

/-- The public entry point's first static guard as an equality. -/
theorem word_rope_rotate_fixed_vector_length (q k : Slice I64) (pos : U32)
    (table : rope.RopeTable) (h : Slice.len q ≠ Slice.len k) :
    word_rope_rotate_fixed q k pos table =
      ok (core.result.Result.Err rope.RopeError.VectorLength, q, k) := by
  rw [rope_rotate_fixed_unfold]
  exact rope_rotate_fixed_vector_length q k pos table h

/-- The second static guard of `RopeTable::rotate` as an equality: when the lengths agree but the
length is not a multiple of `head_dim`, the extracted body returns `Err VectorLength` with both
slices unchanged. `UScalar.is_multiple_of n m = ok (n.val % m.val == 0)`. -/
theorem rope_rotate_fixed_head_dim (q k : Slice I64) (pos : U32) (table : rope.RopeTable)
    (hlen : Slice.len q = Slice.len k)
    (h : ¬ (Slice.len q).val % table.head_dim.val = 0) :
    rope.rope_rotate_fixed q k pos table =
      ok (core.result.Result.Err rope.RopeError.VectorLength, q, k) := by
  unfold rope.rope_rotate_fixed rope.RopeTable.rotate
  dsimp only
  rw [if_neg (by rw [bne_iff_ne]; exact fun hc => hc hlen)]
  simp only [core.num.Usize.is_multiple_of, UScalar.is_multiple_of, bind_tc_ok]
  rw [if_neg (by rw [beq_iff_eq]; exact h)]

/-- The public entry point's second static guard as an equality. -/
theorem word_rope_rotate_fixed_head_dim (q k : Slice I64) (pos : U32)
    (table : rope.RopeTable) (hlen : Slice.len q = Slice.len k)
    (h : ¬ (Slice.len q).val % table.head_dim.val = 0) :
    word_rope_rotate_fixed q k pos table =
      ok (core.result.Result.Err rope.RopeError.VectorLength, q, k) := by
  rw [rope_rotate_fixed_unfold]
  exact rope_rotate_fixed_head_dim q k pos table hlen h

/-- The refusal arm of `rotate_head` as an equality: a loop that returns `ok (x', false)` makes the
extracted body return `Err OutOfRange` with the mutated slice. -/
theorem rotate_head_refusal_eq (self : rope.RopeTable) (x : Slice I64) (pos : U32)
    (d2 : Usize) (x' : Slice I64) (hd2 : self.head_dim / 2#usize = ok d2)
    (h : rope.RopeTable.rotate_head_loop self.turns x pos d2 0#usize true = ok (x', false)) :
    rope.RopeTable.rotate_head self x pos =
      ok (core.result.Result.Err rope.RopeError.OutOfRange, x') := by
  unfold rope.RopeTable.rotate_head
  rw [hd2]
  simp only [bind_tc_ok]
  rw [h]
  simp only [bind_tc_ok]
  rfl

/-- The success arm of `rotate_head` as an equality: a loop that returns `ok (x', true)` makes the
extracted body return `Ok ()` with the mutated slice. -/
theorem rotate_head_success_eq (self : rope.RopeTable) (x : Slice I64) (pos : U32)
    (d2 : Usize) (x' : Slice I64) (hd2 : self.head_dim / 2#usize = ok d2)
    (h : rope.RopeTable.rotate_head_loop self.turns x pos d2 0#usize true = ok (x', true)) :
    rope.RopeTable.rotate_head self x pos =
      ok (core.result.Result.Ok (), x') := by
  unfold rope.RopeTable.rotate_head
  rw [hd2]
  simp only [bind_tc_ok]
  rw [h]
  simp only [bind_tc_ok]
  rfl

/-! ## Loop skeleton: the zero-iteration instance

The extracted loops are `partial_fixpoint`, so they do not reduce by `rfl`; their values come from
`loop.spec_decr_nat`. The instance below is the zero-iteration case of `rotate_loop` (no heads),
proved with measure `0` and the invariant pinning the state. The general case carries the same
skeleton but its body's WP needs the `index_mut` range bounds and a per-head `rotate_head` spec, which
is the residual: see the file header and the exact goal state in the slice report. -/

/-- The extracted `rotate_loop` with `heads = 0` exits immediately and returns the input slices with
the loop flag unchanged. -/
theorem rotate_loop_zero (self : rope.RopeTable) (hd : Usize) (v : alloc.vec.Vec U64)
    (q k : Slice I64) (pos : U32) :
    rope.RopeTable.rotate_loop self hd v q k pos 0#usize 0#usize true = ok (q, k, true) := by
  unfold rope.RopeTable.rotate_loop
  have hspec : loop
      (fun (s : Slice I64 × Slice I64 × Usize × Bool) =>
        rope.RopeTable.rotate_loop.body self hd v pos 0#usize s.1 s.2.1 s.2.2.1 s.2.2.2)
      (q, k, 0#usize, true) ⦃ r => r = (q, k, true) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (_ : Slice I64 × Slice I64 × Usize × Bool) => 0)
      (inv := fun s => s.1 = q ∧ s.2.1 = k ∧ s.2.2.1 = 0#usize ∧ s.2.2.2 = true)
      (post := fun r => r = (q, k, true))
    · intro s hs
      obtain ⟨h1, h2, h3, h4⟩ := hs
      rw [h1, h2, h3, h4]
      unfold rope.RopeTable.rotate_loop.body
      dsimp only
      rw [if_neg (by scalar_tac : ¬ (0#usize < 0#usize))]
      simp only [Aeneas.Std.WP.spec_ok]
    · exact ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hp] at hr
  exact hr

/-! ## Window arithmetic for the `rotate` loop

The `rotate_loop` body's WP generates three obligations at head `h`: the checked products/sums
`h * self.head_dim`, `base + self.head_dim` (`base = h * self.head_dim`) fit `Usize`, and the
`Slice.index_mut` range `{ start := base, «end» := i }` ends inside `q`. They follow from
`h < heads`, `heads = q.len / head_dim` and `head_dim ∣ q.len` alone. -/

/-- **Window arithmetic.** With `hd > 0`, `hd ∣ q.len` and `heads = q.len / hd`, every head index
`h < heads` starts a window `base = h * hd` with `base + hd ≤ q.len`, and both `h * hd` and
`base + hd` fit `Usize`. This discharges the `rotate_loop` body's obligations for
`h * self.head_dim`, `base + self.head_dim` and `↑{ start := base, «end» := i }.end ≤ q.length`. -/
theorem rotate_window_arith (q : Slice I64) (hd h heads : Usize)
    (hhd : 0 < hd.val) (hdiv : hd.val ∣ (Slice.len q).val)
    (hheads : heads.val = (Slice.len q).val / hd.val)
    (hlt : h.val < heads.val) :
    h.val * hd.val ≤ Usize.max ∧
    h.val * hd.val + hd.val ≤ Usize.max ∧
    h.val * hd.val + hd.val ≤ (Slice.len q).val := by
  obtain ⟨m, hm⟩ := hdiv
  have hlen : (Slice.len q).val ≤ Usize.max := by
    rw [Slice.len_val]
    exact Slice.length_ineq q
  have hqdiv : (Slice.len q).val / hd.val = m := by
    rw [hm]
    exact Nat.mul_div_cancel_left m hhd
  have hstep : h.val + 1 ≤ m := by
    have := hlt
    rw [hheads, hqdiv] at this
    omega
  have hkey : h.val * hd.val + hd.val ≤ (Slice.len q).val := by
    have hmul : (h.val + 1) * hd.val ≤ m * hd.val := Nat.mul_le_mul_right hd.val hstep
    rw [Nat.add_mul] at hmul
    have : (1 : ℕ) * hd.val = hd.val := by omega
    rw [this] at hmul
    rw [hm, Nat.mul_comm hd.val m]
    exact hmul
  exact ⟨by nlinarith [hlen], by nlinarith [hlen], hkey⟩

/-! ## Inner loop: `rotate_head_loop`

The per-pair arithmetic above composes to the `head_dim/2`-wide inner loop. The extracted loop is a
`partial_fixpoint`, so its value comes from `loop.spec_decr_nat`; the invariant pins the index
bounds (`Slice.index_usize`/`Slice.update` at `j` and `j + d2`, the `Usize` sums `j + d2` and
`j + 1`), the original entries of the not-yet-processed half-window, and the per-rotary-index
`ropePairInt` content of the entries already written. Every arithmetic refusal (`checked_mul`,
`fixed_sin_cos` error, `checked_sub`/`checked_add`, the `i64` range checks) returns `ok (_, false)`
with the slice unchanged; only the index operations and the `Usize` sums can `fail`, and the
bounds below exclude those. -/

/-- Rotary-index trig data: at index `i < d2` the table carries the turn `t`, its checked
angle `toRad (lift (phase t pos))`, the successful
`fixed_sin_cos` pair `sc`, and the pair magnitude bounds `|sc.cos|, |sc.sin| ≤ 2^33` that keep the
`I128` products (`|x| ≤ 2^63` times `|sc| ≤ 2^33`) and the `round_shift_32` half-adds inside
`I128`. The window and success facts come from `fixed_sin_cos_refines_proved`. -/
def RopeTrigAt (v : alloc.vec.Vec U64) (pos : U32) (i : ℕ) (t : U64) (sc : fixed_trig.SinCos) :
    Prop :=
  v.val[i]? = some t ∧
  ∃ angle : I64,
    angle.val = Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift (Numeric.RopeTurns.phase t.val pos.val)) ∧
    fixed_trig.fixed_sin_cos angle = ok (core.result.Result.Ok sc) ∧
    |sc.cos.val| ≤ 8589934592 ∧ |sc.sin.val| ≤ 8589934592

/-- Per-pair content at rotary index `i`: the two entries of the current slice at `i` and
`i + d2` are the `ropePairInt` rotation of the original entries of `x0` at those positions by the
trig pair `sc` — the identity `rope_pair_spec_proved` bounds, at the integer level the loop
writes. -/
def RopePairAt (x0 x : Slice I64) (d2 : Usize) (i : ℕ) (sc : fixed_trig.SinCos) : Prop :=
  ∃ xi0 xi1 yi0 yi1 : I64,
    x0.val[i]? = some xi0 ∧
    x0.val[i + d2.val]? = some xi1 ∧
    x.val[i]? = some yi0 ∧
    x.val[i + d2.val]? = some yi1 ∧
    yi0.val = (ropePairInt xi0.val xi1.val sc.cos.val sc.sin.val).1 ∧
    yi1.val = (ropePairInt xi0.val xi1.val sc.cos.val sc.sin.val).2

/-- Inner-loop invariant at state `(x, j, ok)`: the window bounds, the not-yet-processed entries
equal to `x0`'s, and (when the flag is still `true`) the `ropePairInt` content of the processed
indices `i < j`. -/
def RotateHeadInv (v : alloc.vec.Vec U64) (x0 : Slice I64) (pos : U32) (d2 : Usize)
    (st : Slice I64 × Usize × Bool) : Prop :=
  2 * d2.val ≤ x0.length ∧
  d2.val ≤ v.length ∧
  st.1.length = x0.length ∧
  st.2.1.val ≤ d2.val ∧
  (∀ i, st.2.1.val ≤ i → i < d2.val →
    st.1.val[i]? = x0.val[i]? ∧ st.1.val[i + d2.val]? = x0.val[i + d2.val]?) ∧
  (st.2.2 = true → ∀ i, i < st.2.1.val →
    ∃ t sc, RopeTrigAt v pos i t sc ∧ RopePairAt x0 st.1 d2 i sc)

/-- `I128` product bound: a `2^63`-wide and a `2^33`-wide factor keep the product inside `I128`. -/
private theorem i128_prod_bound (a b : ℤ) (ha : |a| ≤ 2 ^ 63) (hb : |b| ≤ 2 ^ 33) :
    I128.min ≤ a * b ∧ a * b ≤ I128.max := by
  have habs : |a * b| ≤ (2 : ℤ) ^ 63 * 2 ^ 33 := by
    rw [abs_mul]
    exact mul_le_mul ha hb (by positivity) (by positivity)
  rw [show (2 : ℤ) ^ 63 * 2 ^ 33 = 2 ^ 96 by norm_num] at habs
  have hb' := abs_le.mp habs
  have hmax : (2 : ℤ) ^ 96 ≤ I128.max := by norm_num [I128.max, I128.numBits]
  have hmin : I128.min ≤ -((2 : ℤ) ^ 96) := by norm_num [I128.min, I128.numBits]
  exact ⟨le_trans hmin hb'.1, le_trans hb'.2 hmax⟩

/-- Every `I64` value is within `2^63`. -/
private theorem i64_abs_le (a : I64) : |(a.val : ℤ)| ≤ 2 ^ 63 := by
  have h1 : -(2 : ℤ) ^ 63 ≤ a.val := by simpa [I64.numBits] using a.hmin
  have h2 : a.val < (2 : ℤ) ^ 63 := by simpa [I64.numBits] using a.hmax
  rw [abs_le]
  exact ⟨h1, by omega⟩

/-- A `2^63`-wide times a `2^33`-wide `I64` factor stays under `I128.max`, as the loop's widened
products require. -/
private theorem i64_prod_max (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    ((core.convert.num.FromI128I64.from a).val : ℤ) *
      ((core.convert.num.FromI128I64.from b).val : ℤ) ≤ I128.max := by
  rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  refine (i128_prod_bound a.val b.val (i64_abs_le a) ?_).2
  have h33 : (2 : ℤ) ^ 33 = 8589934592 := by norm_num
  rwa [h33]

/-- The lower mirror of `i64_prod_max`. -/
private theorem i64_prod_min (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    I128.min ≤ ((core.convert.num.FromI128I64.from a).val : ℤ) *
      ((core.convert.num.FromI128I64.from b).val : ℤ) := by
  rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq]
  refine (i128_prod_bound a.val b.val (i64_abs_le a) ?_).1
  have h33 : (2 : ℤ) ^ 33 = 8589934592 := by norm_num
  rwa [h33]

/-- Bounds-only run invariant for `rotate_head_loop`: the half-window fits the state slice, the
frequency table covers the rotary half, and the index is in range. This is what totality of the loop
needs; the content-carrying `RotateHeadInv` above is the residual strengthening. -/
def RotateHeadRunInv (v : alloc.vec.Vec U64) (d2 : Usize)
    (st : Slice I64 × Usize × Bool) : Prop :=
  2 * d2.val ≤ st.1.length ∧
  d2.val ≤ v.length ∧
  st.2.1.val ≤ d2.val

/-- A widened `I64` product of a slice entry with a `fixed_sin_cos` component has magnitude at most
`2^96` (`2^63 · 2^33`), which is what the `round_shift_32` half-adds must stay under. -/
private theorem i64_prod_abs (a b : I64) (hb : |(b.val : ℤ)| ≤ 8589934592) :
    |((core.convert.num.FromI128I64.from a).val : ℤ) *
      ((core.convert.num.FromI128I64.from b).val : ℤ)| ≤ 2 ^ 96 := by
  rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq, abs_mul]
  have ha := i64_abs_le a
  have h33 : |(b.val : ℤ)| ≤ 2 ^ 33 := by
    have h : (2 : ℤ) ^ 33 = 8589934592 := by norm_num
    rwa [h]
  calc |a.val| * |b.val| ≤ (2 : ℤ) ^ 63 * 2 ^ 33 :=
        mul_le_mul ha h33 (by positivity) (by positivity)
    _ = 2 ^ 96 := by norm_num

/-- **Write-stage invariant maintenance.** If `RotateHeadInv` holds at the pre-write state
`(xq, jq, true)` with `jq < d2`, the loop writes `w0` at `jq` and `w1` at `ma2 = jq + d2`
(`x'' = (xq.set jq w0).set ma2 w1`), `j1 = jq + 1`, the pair read for the write is `xv = xq[jq]`,
`yv = xq[ma2]` (still the original pair by the invariant), and `w0`, `w1` are the `ropePairInt`
rotation of that pair by `scq`, then `RotateHeadInv` holds at `(x'', j1, true)`: the processed
prefix extends by one and the not-yet-processed entries keep their original values. -/
private theorem rotate_head_write_step
    (v : alloc.vec.Vec U64) (xg : Slice I64) (pos : U32) (d2 : Usize)
    (xq x2 x'' : Slice I64) (jq ma2 j1 : Usize) (tq : U64) (xv yv w0 w1 : I64)
    (scq : fixed_trig.SinCos)
    (hinv : RotateHeadInv v xg pos d2 (xq, jq, true))
    (hjq : jq.val < d2.val)
    (hj1 : j1.val = jq.val + 1)
    (hma : ma2.val = jq.val + d2.val)
    (hx2 : x2 = xq.set jq w0)
    (hx'' : x'' = x2.set ma2 w1)
    (hxv : xq.val[jq.val]? = some xv)
    (hyv : xq.val[ma2.val]? = some yv)
    (hw0 : w0.val = (ropePairInt xv.val yv.val scq.cos.val scq.sin.val).1)
    (hw1 : w1.val = (ropePairInt xv.val yv.val scq.cos.val scq.sin.val).2)
    (htrigjq : RopeTrigAt v pos jq.val tq scq) :
    RotateHeadInv v xg pos d2 (x'', j1, true) := by
  obtain ⟨ha, hb, hc, hd, he, hf⟩ := hinv
  dsimp only at ha hb hc hd he hf
  simp only [Slice.length] at ha hb hc
  have hcl : x''.val.length = xg.val.length := by
    rw [hx'', hx2, Slice.set_val_eq, Slice.set_val_eq, List.length_set, List.length_set]
    exact hc
  have hlenxq : jq.val < xq.val.length := by rw [hc]; omega
  have hma2 : ma2.val < xq.val.length := by rw [hma]; omega
  -- untouched positions (in bounds) keep the original value
  have hpres : ∀ i, i < xq.val.length → i ≠ jq.val → i ≠ ma2.val →
      x''.val[i]? = xq.val[i]? := by
    intro i hb h1 h2
    rw [hx'', hx2, Slice.set_val_eq, Slice.set_val_eq]
    rw [List.getElem?_set_of_lt w1 (xq.val.set jq.val w0) (by rw [List.length_set]; exact hb)]
    rw [if_neg (by omega)]
    rw [List.getElem?_set_of_lt w0 xq.val hb]
    rw [if_neg (by omega)]
  refine ⟨ha, hb, hcl, ?_, ?_, ?_⟩
  · show j1.val ≤ d2.val
    omega
  · show ∀ i, j1.val ≤ i → i < d2.val →
      x''.val[i]? = xg.val[i]? ∧ x''.val[i + d2.val]? = xg.val[i + d2.val]?
    intro i hi1 hi2
    have h5i := he i (by omega) hi2
    constructor
    · rw [hpres i (by omega) (by omega) (by omega)]
      exact h5i.1
    · rw [hpres (i + d2.val) (by omega) (by omega) (by omega)]
      exact h5i.2
  · show true = true → ∀ i, i < j1.val →
      ∃ tq scq, RopeTrigAt v pos i tq scq ∧ RopePairAt xg x'' d2 i scq
    intro _ i hi
    by_cases hij : i = jq.val
    · subst hij
      refine ⟨tq, scq, htrigjq, xv, yv, w0, w1, ?_, ?_, ?_, ?_, hw0, hw1⟩
      · have h := (he jq.val le_rfl hjq).1
        rw [hxv] at h
        exact h.symm
      · have h := (he jq.val le_rfl hjq).2
        rw [← hma] at h
        rw [hyv] at h
        rw [← hma]
        exact h.symm
      · rw [hx'', hx2, Slice.set_val_eq, Slice.set_val_eq]
        rw [List.getElem?_set_of_lt w1 (xq.val.set jq.val w0) (by rw [List.length_set]; exact hlenxq)]
        rw [if_neg (by omega)]
        rw [List.getElem?_set_of_lt w0 xq.val hlenxq]
        rw [if_pos rfl]
      · rw [← hma]
        rw [hx'', hx2, Slice.set_val_eq, Slice.set_val_eq]
        rw [List.getElem?_set_of_lt w1 (xq.val.set jq.val w0) (by rw [List.length_set]; exact hma2)]
        rw [if_pos rfl]
    · obtain ⟨tq', scq', htrig', hpair'⟩ := hf rfl i (by omega)
      obtain ⟨xi0, xi1, yi0, yi1, hxi0, hxi1, hyi0, hyi1, hv0, hv1⟩ := hpair'
      refine ⟨tq', scq', htrig', xi0, xi1, yi0, yi1, hxi0, hxi1, ?_, ?_, hv0, hv1⟩
      · rw [hpres i (by omega) (by omega) (by omega)]
        exact hyi0
      · rw [hpres (i + d2.val) (by omega) (by omega) (by omega)]
        exact hyi1

/-- The `rotate_head_loop` body step as a standalone WP: the explicit `WP.spec_bind` walk for the
prefix (both `if`s, the widened `pos` cast, the turn read, and turn-to-radians) followed by `step*` on the arithmetic
branches. Proved in isolation so the loop theorem never normalises the whole body at once. -/
private theorem rotate_head_body_spec (v : alloc.vec.Vec U64) (pos : U32) (d2 : Usize)
    (htrig : ∀ i, i < d2.val → ∃ tq scq, RopeTrigAt v pos i tq scq)
    (st : Slice I64 × Usize × Bool) (hst : RotateHeadRunInv v d2 st) :
    (rope.RopeTable.rotate_head_loop.body v pos d2 st.1 st.2.1 st.2.2) ⦃ r =>
      match r with
      | .done _ => True
      | .cont st' => RotateHeadRunInv v d2 st' ∧
          d2.val - st'.2.1.val < d2.val - st.2.1.val ⦄ := by
  rcases st with ⟨xq, jq, oq⟩
  obtain ⟨h1, h2, h3⟩ := hst
  dsimp only at h1 h2 h3
  unfold rope.RopeTable.rotate_head_loop.body
  dsimp only
  by_cases hlt : jq < d2
  · rw [if_pos hlt]
    have hjlt' : (jq : ℕ) < (d2 : ℕ) := Iff.mp (UScalar.lt_equiv jq d2) hlt
    by_cases hok : oq = true
    · rw [if_pos hok]
      simp only [lift, bind_tc_ok]
      simp only [alloc.vec.Vec.index_slice_index]
      refine Aeneas.Std.WP.spec_bind (alloc.vec.Vec.index_usize_spec v jq ?_) ?_
      · omega
      · intro tq htq
        obtain ⟨t_trig, scq, ht_trig, aq, haq, hscq, hcos, hsin⟩ := htrig jq.val (by omega)
        have htq_eq : tq = t_trig := by
          obtain ⟨_, hget⟩ := List.getElem?_eq_some_iff.mp ht_trig
          rw [htq]
          exact hget
        subst htq_eq
        step*
        have hi3_val : i3.val = tq.val * pos.val := by
          rw [i3_post, u128_cast_u64_val, u128_cast_u32_val]
        have hphase_val : (UScalar.cast UScalarTy.U64 (i3 &&& UScalar.cast UScalarTy.U128 core.num.U64.MAX)).val =
            Numeric.RopeTurns.phase tq.val pos.val := turn_phase_val_eq tq pos i3 hi3_val
        have hi6_val : i6.val = aq.val := by
          rw [i6_post, hphase_val, haq]
        have hi6_eq : i6 = aq := IScalar.eq_of_val_eq hi6_val
        rw [hi6_eq, hscq]
        simp only [bind_tc_ok]
        step*
        all_goals (first
          | exact i64_prod_max _ _ hcos
          | exact i64_prod_max _ _ hsin
          | exact i64_prod_min _ _ hcos
          | exact i64_prod_min _ _ hsin
          | skip)
        · dsimp only [RotateHeadRunInv]
          refine ⟨⟨h1, h2, by omega⟩, by omega⟩
        · dsimp only [RotateHeadRunInv]
          refine ⟨⟨h1, h2, by omega⟩, by omega⟩
        · rename_i hsub hadd
          have hp00abs : |p00.val| ≤ (2 : ℤ) ^ 96 := by rw [p00_post]; exact i64_prod_abs _ _ hcos
          have hp01abs : |p01.val| ≤ (2 : ℤ) ^ 96 := by rw [p01_post]; exact i64_prod_abs _ _ hsin
          have hp10abs : |p10.val| ≤ (2 : ℤ) ^ 96 := by rw [p10_post]; exact i64_prod_abs _ _ hcos
          have hp11abs : |p11.val| ≤ (2 : ℤ) ^ 96 := by rw [p11_post]; exact i64_prod_abs _ _ hsin
          have hre : re.val = p00.val - p01.val := by
            have h := I128.checked_sub_bv_spec p00 p01
            have hc' : core.num.checked_sub_IScalar p00 p01 = some re := hsub
            rw [hc'] at h
            rcases h with ⟨-, -, heq, -⟩
            exact heq
          have hreabs : |re.val| ≤ (2 : ℤ) ^ 97 := by
            rw [hre]
            calc |p00.val - p01.val| ≤ |p00.val| + |p01.val| := by
                  simpa [sub_eq_add_neg] using abs_add_le p00.val (-p01.val)
              _ ≤ 2 ^ 96 + 2 ^ 96 := add_le_add hp00abs hp01abs
              _ = 2 ^ 97 := by norm_num
          have him : im.val = p10.val + p11.val := by
            have h := I128.checked_add_bv_spec p10 p11
            have hc' : core.num.checked_add_IScalar p10 p11 = some im := hadd
            rw [hc'] at h
            rcases h with ⟨-, -, heq, -⟩
            exact heq
          have himabs : |im.val| ≤ (2 : ℤ) ^ 97 := by
            rw [him]
            calc |p10.val + p11.val| ≤ |p10.val| + |p11.val| := abs_add_le _ _
              _ ≤ 2 ^ 96 + 2 ^ 96 := add_le_add hp10abs hp11abs
              _ = 2 ^ 97 := by norm_num
          have hmin_re : I128.min ≤ re.val + 2147483648 := by
            have hmin128 : I128.min ≤ -((2 : ℤ) ^ 97) + 2147483648 := by norm_num [I128.min, I128.numBits]
            have := (abs_le.mp hreabs).1
            linarith
          have hmax_re : re.val + 2147483648 ≤ I128.max := by
            have hmax128 : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max := by norm_num [I128.max, I128.numBits]
            have := (abs_le.mp hreabs).2
            linarith
          have hmin_im : I128.min ≤ im.val + 2147483648 := by
            have hmin128 : I128.min ≤ -((2 : ℤ) ^ 97) + 2147483648 := by norm_num [I128.min, I128.numBits]
            have := (abs_le.mp himabs).1
            linarith
          have hmax_im : im.val + 2147483648 ≤ I128.max := by
            have hmax128 : (2 : ℤ) ^ 97 + 2147483648 ≤ I128.max := by norm_num [I128.max, I128.numBits]
            have := (abs_le.mp himabs).2
            linarith
          refine Aeneas.Std.WP.spec_bind (value_round_shift_32_spec re hmin_re hmax_re) ?_
          intro out0 hout0
          refine Aeneas.Std.WP.spec_bind (value_round_shift_32_spec im hmin_im hmax_im) ?_
          intro out1 hout1
          step*
          all_goals (first
            | dsimp only [RotateHeadRunInv]
              refine ⟨⟨h1, h2, by omega⟩, by omega⟩
            | dsimp only [RotateHeadRunInv]
              have hs_len : s.length = xq.length := by rw [s_post, x2_post]; simp
              refine ⟨⟨by omega, h2, by omega⟩, by omega⟩)
    · rw [if_neg hok]
      simp only [Aeneas.Std.WP.spec_ok]
  · rw [if_neg hlt]
    simp only [Aeneas.Std.WP.spec_ok]

/-- **Inner loop totality.** Under the window bounds and the per-rotary-index trig data,
`rotate_head_loop` never `fail`s: every arithmetic refusal leaves the flag `false`, and the index
operations are in range. -/
theorem rotate_head_loop_spec (v : alloc.vec.Vec U64) (pos : U32) (d2 : Usize)
    (htrig : ∀ i, i < d2.val → ∃ tq scq, RopeTrigAt v pos i tq scq)
    (hlenv : d2.val ≤ v.length)
    (xst : Slice I64) (jst : Usize) (kok : Bool)
    (hx : 2 * d2.val ≤ xst.length) (hj : jst.val ≤ d2.val) :
    ∃ r : Slice I64 × Bool, rope.RopeTable.rotate_head_loop v xst pos d2 jst kok = ok r := by
  have hspec : loop
      (fun (st : Slice I64 × Usize × Bool) =>
        rope.RopeTable.rotate_head_loop.body v pos d2 st.1 st.2.1 st.2.2)
      (xst, jst, kok) ⦃ _ => True ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun st : Slice I64 × Usize × Bool => d2.val - st.2.1.val)
      (inv := RotateHeadRunInv v d2)
      (post := fun _ : Slice I64 × Bool => True)
    · intro st hst
      convert rotate_head_body_spec v pos d2 htrig st hst using 1
      funext r
      cases r with
      | done y => rfl
      | cont x' => rfl
    · exact ⟨hx, hlenv, hj⟩
  obtain ⟨r, hr, _⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  exact ⟨r, hr⟩

/-! ## Witness -/

namespace Witness

/-- Concrete evaluation witness: `toRad (lift (2^62))` matches the Q32.32 radian value of a quarter turn
`HALF_PI_Q32 = 6746518852`. -/
theorem quarter_turn_witness :
    Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift (2 ^ 62)) = 6746518852 := by
  decide

/-- Concrete evaluation witness: `toRad (lift 0) = 0`. -/
theorem zero_turn_witness :
    Numeric.RopeTurns.toRad (Numeric.RopeTurns.lift 0) = 0 := by
  decide

end Witness

end

end apeiron_word
