/-
Joint non-vacuity witness of the `rotate_head` success domain (`APEIRON-ALGEBRA-G3-ROPE-W3-B6`).

`Numeric.Rope.HeadContent` proves `rotate_head_content` on the success domain: a table whose
`head_dim / 2 = d2`, a head `x` of at least `2·d2` entries, per-rotary-index trig data
`RopeTrigAt`, and pairwise `i64`-fit `RotHeadFits`. This module exhibits one concrete table, head,
position and `d2` at which every hypothesis holds at once, so the theorem is not vacuous:

* `ropeWitTable` — `head_dim = 2` (`d2 = 1`) with the single frequency `0`;
* `ropeWitHead` — `[2^32, 0]`, a nonzero entry in a length-2 head;
* `ropeWitPos = 0` — the checked angle is `0 · 0 = 0`, in the `trig_window`;
* `ropeWitSc` — the trig pair at angle `0`, taken from `fixed_sin_cos_refines_proved` through
  `Classical.choose` (the extracted body is never evaluated);
* `rope_head_domain_witness` — `hd2`, `htrig`, `hfit`, `hx` hold jointly;
* `rope_head_witness_ok` — `rotate_head` returns `Ok ()` with the `rotPrefix` content.

Scope: existence of an inhabited success domain. It is not a statement about arbitrary tables, and
the outer `rotate` loop and the refusal arm stay outside `RopeHeadContent`'s covered domain.
-/
import Numeric.Rope.HeadContent

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The witness table, head and trig pair -/

/-- The witness rotary half-width: `head_dim / 2 = 1`. -/
def ropeWitD2 : Usize := 1#usize

/-- The witness position `0`, so every checked angle `pos · freqs[j]` is `0`. -/
def ropeWitPos : U32 := 0#u32

/-- The witness frequency table: `head_dim = 2`, one rotary turn `0`. -/
def ropeWitTable : rope.RopeTable :=
  { head_dim := 2#usize, turns := alloc.vec.Vec.from [0#u64] (by simp; scalar_tac) }

/-- The witness head `[2^32, 0]`: `1.0` in Q32.32 at index `0`, length two. -/
def ropeWitHead : Slice I64 := Slice.from [4294967296#i64, 0#i64] (by simp; scalar_tac)

/-- The witness angle `pos · freqs[0] = 0 · 0 = 0`. -/
def ropeWitAngle : I64 := 0#i64

/-- `0` lies in the `fixed_sin_cos` window `[-4096·2^32, 4096·2^32]`. -/
theorem ropeWitAngle_window : trig_window ropeWitAngle := by
  norm_num [trig_window, ropeWitAngle]

/-- The trig pair at angle `0`, taken from the proved `fixed_sin_cos_refines_proved` (the extracted
body is not evaluated). -/
def ropeWitSc : fixed_trig.SinCos :=
  Classical.choose (fixed_sin_cos_refines_proved ropeWitAngle ropeWitAngle_window)

/-- The constant trig function the success-domain hypotheses use. -/
def ropeWitScAt : ℕ → fixed_trig.SinCos := fun _ => ropeWitSc

/-- The chosen pair is the one `word_fixed_sin_cos` returns at angle `0`, with the per-output
`eps`/floor bounds at `sin 0 = 0`, `cos 0 = 1`. -/
theorem ropeWitSc_spec :
    word_fixed_sin_cos ropeWitAngle = ok (core.result.Result.Ok ropeWitSc) ∧
      |toReal ropeWitSc.sin| ≤ trig_floor ∧
      |toReal ropeWitSc.cos - 1| ≤ epsExp + trig_floor := by
  have h := Classical.choose_spec (fixed_sin_cos_refines_proved ropeWitAngle ropeWitAngle_window)
  have h0 : toReal ropeWitAngle = 0 := by simp [toReal, ropeWitAngle]
  refine ⟨h.1, ?_, ?_⟩
  · simpa [h0, ropeWitSc] using h.2.1
  · simpa [h0, ropeWitSc] using h.2.2

/-! ## Supporting bounds -/

/-- A `fixed_sin_cos` pair at angle `0`: `cos` is within the `2^-20` window of `1` and `sin` within
the `2^-20` floor of `0`, so both Q32.32 components are `2^33`-bounded. -/
private theorem ropeWit_sc_bounds (sc : fixed_trig.SinCos)
    (hc : |toReal sc.cos - 1| ≤ epsExp + trig_floor)
    (hs : |toReal sc.sin| ≤ trig_floor) :
    |sc.cos.val| ≤ 8589934592 ∧ |sc.sin.val| ≤ 8589934592 := by
  have hsinR : |(sc.sin.val : ℝ)| ≤ q32 := by
    rw [toReal, abs_div, abs_of_pos q32_pos] at hs
    have h1 := (div_le_iff₀ q32_pos).mp hs
    have h2 : trig_floor * q32 ≤ q32 := by norm_num [trig_floor, q32]
    linarith
  have hcosR : |toReal sc.cos| ≤ 2 := by
    have hb := abs_le.mp hc
    have hsmall : epsExp + trig_floor ≤ 1 := by norm_num [epsExp, trig_floor]
    rw [abs_le]
    constructor <;> linarith [hb.1, hb.2, hsmall]
  have hcosR2 : |(sc.cos.val : ℝ)| ≤ 2 * q32 := by
    rw [toReal, abs_div, abs_of_pos q32_pos] at hcosR
    have h1 := (div_le_iff₀ q32_pos).mp hcosR
    linarith
  constructor
  · have h2 : (2:ℝ) * q32 = 8589934592 := by norm_num [q32]
    have : |(sc.cos.val : ℝ)| ≤ (8589934592 : ℝ) := by linarith
    exact_mod_cast this
  · have h2 : q32 ≤ (8589934592 : ℝ) := by norm_num [q32]
    have : |(sc.sin.val : ℝ)| ≤ (8589934592 : ℝ) := by linarith
    exact_mod_cast this

/-- The chosen pair's components are `2^33`-bounded (`RopeTrigAt`'s magnitude conjunct). -/
theorem ropeWitSc_bounds :
    |ropeWitSc.cos.val| ≤ 8589934592 ∧ |ropeWitSc.sin.val| ≤ 8589934592 :=
  ropeWit_sc_bounds ropeWitSc ropeWitSc_spec.2.2 ropeWitSc_spec.2.1

/-- `roundShift32` of a `2^65`-bounded integer stays in `i64`. -/
private theorem ropeWit_roundShift_bounds (v : ℤ) (hv : |v| ≤ (2:ℤ)^65) :
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

/-- At `j = 0` the witness pair is `(roundShift32 (2^32·cos), roundShift32 (2^32·sin))`, both in
`i64`. -/
theorem ropeWit_pair_bounds :
    I64.min ≤ (ropeHeadPair ropeWitHead.val ropeWitD2.val ropeWitScAt 0).1 ∧
    (ropeHeadPair ropeWitHead.val ropeWitD2.val ropeWitScAt 0).1 ≤ I64.max ∧
    I64.min ≤ (ropeHeadPair ropeWitHead.val ropeWitD2.val ropeWitScAt 0).2 ∧
    (ropeHeadPair ropeWitHead.val ropeWitD2.val ropeWitScAt 0).2 ≤ I64.max := by
  have hhead : ropeWitHead.val = [4294967296#i64, 0#i64] := Slice.from_val _ _
  have hd2v : ropeWitD2.val = (1 : ℕ) := by simp [ropeWitD2]
  have hscv : ropeWitScAt 0 = ropeWitSc := rfl
  have hpair : ropeHeadPair ropeWitHead.val ropeWitD2.val ropeWitScAt 0 =
      (roundShift32 (4294967296 * ropeWitSc.cos.val),
       roundShift32 (4294967296 * ropeWitSc.sin.val)) := by
    unfold ropeHeadPair ropePairInt
    rw [hhead, hd2v, hscv]
    norm_num
  have hc33 : |ropeWitSc.cos.val| ≤ (2:ℤ)^33 := by
    have h := ropeWitSc_bounds.1
    have h2 : (2:ℤ)^33 = 8589934592 := by norm_num
    rwa [h2]
  have hs33 : |ropeWitSc.sin.val| ≤ (2:ℤ)^33 := by
    have h := ropeWitSc_bounds.2
    have h2 : (2:ℤ)^33 = 8589934592 := by norm_num
    rwa [h2]
  have hv0 : |4294967296 * ropeWitSc.cos.val| ≤ (2:ℤ)^65 := by
    rw [abs_mul]
    have h1 : |(4294967296 : ℤ)| = (2:ℤ)^32 := by norm_num
    rw [h1]
    calc (2:ℤ)^32 * |ropeWitSc.cos.val| ≤ (2:ℤ)^32 * (2:ℤ)^33 :=
          mul_le_mul_of_nonneg_left hc33 (by norm_num)
      _ = (2:ℤ)^65 := by norm_num
  have hv1 : |4294967296 * ropeWitSc.sin.val| ≤ (2:ℤ)^65 := by
    rw [abs_mul]
    have h1 : |(4294967296 : ℤ)| = (2:ℤ)^32 := by norm_num
    rw [h1]
    calc (2:ℤ)^32 * |ropeWitSc.sin.val| ≤ (2:ℤ)^32 * (2:ℤ)^33 :=
          mul_le_mul_of_nonneg_left hs33 (by norm_num)
      _ = (2:ℤ)^65 := by norm_num
  have b0 := ropeWit_roundShift_bounds (4294967296 * ropeWitSc.cos.val) hv0
  have b1 := ropeWit_roundShift_bounds (4294967296 * ropeWitSc.sin.val) hv1
  rw [hpair]
  exact ⟨b0.1, b0.2, b1.1, b1.2⟩

/-! ## The joint hypotheses -/

/-- At `i = 0 < d2 = 1` the turn is `0`, the checked angle is `0`, and the
chosen pair satisfies the magnitude conjunct of `RopeTrigAt`. -/
theorem ropeWitTable_htrig :
    ∀ i, i < ropeWitD2.val → ∃ t, RopeTrigAt ropeWitTable.turns ropeWitPos i t (ropeWitScAt i) := by
  intro i hi
  have hi0 : i = 0 := by
    have : i < 1 := by simpa [ropeWitD2] using hi
    omega
  subst hi0
  refine ⟨0#u64, ?_⟩
  refine ⟨?_, 0#i64, ?_, ?_, ropeWitSc_bounds.1, ropeWitSc_bounds.2⟩
  · simp [ropeWitTable, alloc.vec.Vec.from_val]
  · decide
  · simpa [ropeWitAngle, ropeWitScAt, word_fixed_sin_cos] using ropeWitSc_spec.1

/-- The single rotary pair of the witness head fits `i64`. -/
theorem ropeWitTable_hfit : RotHeadFits ropeWitHead ropeWitD2 ropeWitScAt := by
  intro j hj
  have hj0 : j = 0 := by
    have : j < 1 := by simpa [ropeWitD2] using hj
    omega
  subst hj0
  exact ropeWit_pair_bounds

/-- The witness head's nonzero entry `2^32` at index `0`. -/
theorem ropeWitHead_entry : ropeWitHead.val[0]? = some (4294967296#i64) := by
  rw [show ropeWitHead.val = [4294967296#i64, 0#i64] from Slice.from_val _ _]
  rfl

/-- **Joint witness.** Every hypothesis of `rotate_head_content` holds at once for the witness
table, head, position and `d2`. -/
theorem rope_head_domain_witness :
    ropeWitTable.head_dim / 2#usize = ok ropeWitD2 ∧
      (∀ i, i < ropeWitD2.val →
        ∃ t, RopeTrigAt ropeWitTable.turns ropeWitPos i t (ropeWitScAt i)) ∧
      RotHeadFits ropeWitHead ropeWitD2 ropeWitScAt ∧
      2 * ropeWitD2.val ≤ ropeWitHead.length := by
  refine ⟨?_, ropeWitTable_htrig, ropeWitTable_hfit, ?_⟩
  · obtain ⟨z, hz, hzv⟩ := UScalar.div_spec (2#usize) (y := 2#usize) (by decide)
    have hz1 : z = ropeWitD2 := by
      apply UScalar.eq_of_val_eq
      rw [hzv]
      norm_num [ropeWitD2]
    rwa [hz1] at hz
  · simp [ropeWitD2, ropeWitHead, Slice.from_val]

/-- **Success at the witness.** `rotate_head` returns `Ok ()` and a head denoting the `rotPrefix`
content, the conclusion of `rotate_head_content` at the witness. -/
theorem rope_head_witness_ok :
    ∃ x' : Slice I64,
      rope.RopeTable.rotate_head ropeWitTable ropeWitHead ropeWitPos =
        ok (core.result.Result.Ok (), x') ∧
      x'.val.map (fun a => a.val) =
        rotPrefix ropeWitHead.val ropeWitD2.val ropeWitScAt ropeWitD2.val :=
  rotate_head_content ropeWitTable ropeWitHead ropeWitPos ropeWitD2
    rope_head_domain_witness.1 ropeWitScAt rope_head_domain_witness.2.1
    rope_head_domain_witness.2.2.1 rope_head_domain_witness.2.2.2

end

end apeiron_word
