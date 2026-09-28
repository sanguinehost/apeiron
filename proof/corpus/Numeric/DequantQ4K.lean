/-
The row-level half of the extracted `Q4_K` dequantizer refinement
, continued from `Numeric.DequantQ4KBlock` (imported), which
holds the 6-bit scale/min reference, the 144-byte block reader, the raw-level row references and the
innermost 32-lane loop; the moved theorems remain available through that import. This module proves
the group loop, `push_prefix`, the raw-window bridge and the row law `dequant_q4_k_row_spec`.

Refinement of the extracted `Q4_K` dequantizer.

The Rust body is `dequant::q4_k_block` and `dequant::dequant_q4_k_row` in
`src/engine/dequant.rs`: the `f16` scale `d` and minimum `dmin` decode exactly to Q32.32
(`f16_to_fixed_exact`), the 12-byte packed 6-bit scale/min table is unpacked by
`quant::q4_k_scale_min` (ggml `get_scale_min_k4`), and each weight is `d·sc_j·q_i − dmin·m_j`
with checked `i64` multiply and subtract, so the law is an `ℤ` equality, not an error bound.

## Proved

* `q4_k_scale_min_ref`: for `j < 8`, the extracted `quant.q4_k_scale_min` returns the two 6-bit
  fields of `q4kScaleMinRef` on the 12-byte table (same `&&&`/`|||`/`>>>`/`<<<` operations as the
  Rust source).
* `read_scale_ok`: a finite little-endian `f16` scale decodes to `f16Value`.
* `q4_k_block_loop_spec`: the 12-byte scale-table copy loop fills the table from block bytes
  `4..16`.
* `q4_k_block_spec`: on a whole 144-byte block with finite `d`/`dmin`, `q4_k_block` returns
  `(d, dmin, sc, codes)` with `d.val = f16Value`, `dmin.val = f16Value`, `sc` the 12 table bytes,
  and `codes` pointwise `q4CodesRef`.
* `dequant_row_fixed_q4_k`: `dequant_row_fixed` at `ggml_type = 12` (`GGML_TYPE_Q4_K`) is
  `dequant_q4_k_row`.
* `q4k_lane_body_step`, `q4k_lane_loop_spec`: the innermost 32-lane loop assembles, for each group
  `g`, the 64 entries `q4kLaneVal` (closed form `slope·code − intercept`) at lanes `0..31` and
  `32..63`, under the checked-`i64` fits.
* `q4k_group_body_step`, `q4k_group_loop_spec`: the 4-group loop computes the four
  `(scale·6-bit, min·6-bit)` affine pairs and fills all 256 entries of `q4kBlockVal`
  (`d·sc·q − dmin·m`).
* `push_prefix_spec`: `push_prefix` appends exactly `vals.take take`.
* `q4kBlockVals_getD`, `q4kWeight_eq`: the block-list reference reads back as `q4kBlockVal`, and
  row index `256·b + i` reads as block `b` entry `i`.
* `getD_slice`, `f16LeHalf_block`, `q4kScList_getD`, `q4kCodesList_getD`, `q4kBlockVal_bridge`: the
  raw-level reference on the 144-byte block window equals the decoded-array reference.
* `q4kBlock_body_step`, `q4kRow_loop0_spec`, `dequant_q4_k_row_spec`: on the per-block success
  domain `q4kBlockDomain` and with output capacity, the extracted `dequant_q4_k_row raw n out`
  succeeds with `Ok ()` and `dequantVals out' = dequantVals out ++ q4kRowValues raw n`.
* `q4kScaleMinRef_fst_lt`/`_snd_lt`, `q4CodesRef_lt_16`, `f16Value_abs_lt`: the factors are bounded
  (`< 64`, `< 16`, `< 2^48`), so every checked product/subtract fits `i64` without refusal.

## Residual

The Q4_K row-level refusals (short payload, non-finite scale, overflowing product) are not proved:
`q4kBlockDomain` is a sufficient success hypothesis, not a characterization of success, and the
refusing rows are not formalized here. The `dequant_row_fixed` dispatch and the other type arms are
out of this module.

## Trust boundary

Kernel-checked against the Aeneas model of `i64::checked_mul`/`checked_sub`, `u16::from_le_bytes`,
slice range `get`, `Array` index/update and `Vec::push` (`target/aeneas-model/lean`);
rustc/LLVM codegen and the GGUF file reader upstream of the row bytes are outside this module.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is not used on extracted
bodies.
-/
import Numeric.DequantQ4KBlock

namespace apeiron_word
open Aeneas Aeneas.Std Result

/-! ## Group-loop refinement (step b)

`dequant_q4_k_row_loop0_loop0` computes the four `(slope, intercept)` pairs of a block and calls the
lane loop once per group. `q4kGroupInv` states the assembled block values; the step composes
`q4_k_scale_min_ref`, the four checked products and `q4k_lane_loop_spec`. -/

/-- Invariant of the 4-group loop: `g` complete groups are the block reference, the rest the entry
array. -/
def q4kGroupInv (scale min_scale : Std.I64) (sc : Array Std.U8 12#usize)
    (codes : Array Std.U8 256#usize) (init vals : Array Std.I64 256#usize) (g : Nat)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ g ≤ 4 ∧
  ∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j < 64 * g then q4kBlockVal scale.val min_scale.val sc.val codes.val j
    else (init.val.getD j 0#i64).val

lemma abs_u8_le {c : Std.U8} {n : Nat} (h : c.val ≤ n) : |(c.val : ℤ)| ≤ (n : ℤ) := by
  rw [abs_of_nonneg (by positivity)]
  exact_mod_cast h

lemma hcast_u8_val (c : Std.U8) : (UScalar.hcast .I64 c).val = (c.val : ℤ) := by
  rw [UScalar.hcast_val_eq, IScalarTy.I64_numBits_eq]
  apply Int.bmod_eq_of_le
  · norm_num
  · have h := U8.lt_succ_max c
    norm_num
    omega

theorem i64_mul_fits_zb {x y A B : ℤ} (hx : |x| ≤ A) (hy : |y| ≤ B)
    (h : A * B < 2 ^ 63) : I64.min ≤ x * y ∧ x * y ≤ I64.max := by
  have hxy : |x * y| < 2 ^ 63 := by
    rw [abs_mul]
    have hA : 0 ≤ A := le_trans (abs_nonneg x) hx
    have := mul_le_mul hx hy (abs_nonneg y) hA
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  omega

lemma q4kLaneAt_zero (codes : Array Std.U8 256#usize) (g : Nat) (sL iL sH iH : ℤ)
    (vals0 : Array Std.I64 256#usize) (j : Nat) :
    q4kLaneAt codes g 0 sL iL sH iH vals0 j = (vals0.val.getD j 0#i64).val := by
  unfold q4kLaneAt
  by_cases h : j < 64 * g
  · rw [if_pos h]
  · rw [if_neg h]
    rw [if_neg]
    rintro ⟨-, h | ⟨h1, h2⟩⟩ <;> omega

lemma q4kLaneAt_full (codes : Array Std.U8 256#usize) (g : Nat) (sL iL sH iH : ℤ)
    (vals0 : Array Std.I64 256#usize) (j : Nat)
    (h1 : ¬ j < 64 * g) (h2 : j / 64 = g) :
    q4kLaneAt codes g 32 sL iL sH iH vals0 j = q4kLaneVal codes g (j % 64) sL iL sH iH := by
  unfold q4kLaneAt
  rw [if_neg h1]
  have hcond : j % 64 < 32 ∨ 32 ≤ j % 64 ∧ j % 64 < 32 + 32 := by
    rcases Nat.lt_or_ge (j % 64) 32 with h | h
    · exact Or.inl h
    · exact Or.inr ⟨h, by omega⟩
  rw [if_pos ⟨h2, hcond⟩]

lemma q4kLaneVal_blockVal (d dmin : ℤ) (sc : List Std.U8) (codes : Array Std.U8 256#usize)
    (g lane : Nat) (hlane : lane < 64) (sL iL sH iH : ℤ)
    (hsL : sL = d * ((q4kScaleMinRef sc (2 * g)).1 : ℤ))
    (hiL : iL = dmin * ((q4kScaleMinRef sc (2 * g)).2 : ℤ))
    (hsH : sH = d * ((q4kScaleMinRef sc (2 * g + 1)).1 : ℤ))
    (hiH : iH = dmin * ((q4kScaleMinRef sc (2 * g + 1)).2 : ℤ)) :
    q4kLaneVal codes g lane sL iL sH iH = q4kBlockVal d dmin sc codes.val (64 * g + lane) := by
  have hg : (64 * g + lane) / 64 = g := by omega
  have hl : (64 * g + lane) % 64 = lane := by omega
  unfold q4kBlockVal
  rw [hg, hl]
  by_cases h32 : lane < 32
  · rw [if_pos h32, q4kLaneVal_low codes g lane sL iL sH iH h32, hsL, hiL]
  · rw [if_neg h32, q4kLaneVal_high codes g lane sL iL sH iH (by omega), hsH, hiH]

lemma q4kGroupInv_succ (scale min_scale : Std.I64) (sc : Array Std.U8 12#usize)
    (codes : Array Std.U8 256#usize) (init vals vals2 : Array Std.I64 256#usize) (g : Nat)
    (slope_lo intercept_lo slope_hi intercept_hi : Std.I64)
    (hsL : slope_lo.val = scale.val * ((q4kScaleMinRef sc.val (2 * g)).1 : ℤ))
    (hiL : intercept_lo.val = min_scale.val * ((q4kScaleMinRef sc.val (2 * g)).2 : ℤ))
    (hsH : slope_hi.val = scale.val * ((q4kScaleMinRef sc.val (2 * g + 1)).1 : ℤ))
    (hiH : intercept_hi.val = min_scale.val * ((q4kScaleMinRef sc.val (2 * g + 1)).2 : ℤ))
    (hg4 : g < 4)
    (hginv : q4kGroupInv scale min_scale sc codes init vals g true none)
    (hlane : q4kLaneInv codes g slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
      vals vals2 32 true none) :
    q4kGroupInv scale min_scale sc codes init vals2 (g + 1) true none := by
  refine ⟨rfl, rfl, by omega, ?_⟩
  intro j hj
  have hlv := hlane.2.2.2 j hj
  have hgv := hginv.2.2.2 j hj
  rw [hlv]
  by_cases hlt : j < 64 * g
  · have hz : q4kLaneAt codes g 32 slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
        vals j = (vals.val.getD j 0#i64).val := by
      unfold q4kLaneAt; rw [if_pos hlt]
    rw [hz, hgv, if_pos hlt, if_pos (by omega)]
  · by_cases hgj : j / 64 = g
    · have hfull := q4kLaneAt_full codes g slope_lo.val intercept_lo.val slope_hi.val
        intercept_hi.val vals j hlt hgj
      rw [hfull]
      have hbv := q4kLaneVal_blockVal scale.val min_scale.val sc.val codes g (j % 64)
        (Nat.mod_lt _ (by norm_num)) slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
        hsL hiL hsH hiH
      have hj : 64 * g + j % 64 = j := by
        have h := Nat.div_add_mod j 64
        rw [hgj] at h
        exact h
      rw [hbv, if_pos (by omega), hj]
    · have hz : q4kLaneAt codes g 32 slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
          vals j = (vals.val.getD j 0#i64).val := by
        unfold q4kLaneAt
        rw [if_neg hlt, if_neg (fun hc => hgj hc.1)]
      rw [hz, hgv, if_neg hlt, if_neg (by intro h; exact hgj (by omega))]

/-- Destructuring a successful pair bind reduces to the direct application (the do-notation
binding is not reduced by `simp`/`dsimp`, but is definitionally transparent). -/
theorem q4k_let_pair {α β γ : Type} (x : α) (y : β) (f : α → β → Result γ) :
    (do let (a, b) ← (ok (x, y) : Result (α × β)); f a b) = f x y := by
  simp only [bind_tc_ok]
  rfl

/-- Destructuring a successful triple bind reduces to the direct application. -/
theorem q4k_let_triple {α β γ δ : Type} (x : α) (y : β) (z : γ)
    (f : α → β → γ → Result δ) :
    (do let (a, b, c) ← (ok (x, y, z) : Result (α × β × γ)); f a b c) = f x y z := by
  simp only [bind_tc_ok]
  rfl

/-- One group-loop body step: the four `(slope, intercept)` products succeed and the lane loop
writes the group's 64 weights, advancing `q4kGroupInv` by one group. -/
theorem q4k_group_body_step
    (scale min_scale : Std.I64) (sc : Array Std.U8 12#usize) (codes : Array Std.U8 256#usize)
    (init vals : Array Std.I64 256#usize) (group : Std.Usize)
    (ok1 : Bool) (err : Option error.EngineError)
    (hgroup : group.val < 4)
    (hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 15)
    (hscale : |scale.val| ≤ 2 ^ 48) (hmin : |min_scale.val| ≤ 2 ^ 48)
    (hok : ok1 = true) (herr : err = none)
    (hinv : q4kGroupInv scale min_scale sc codes init vals group.val ok1 err) :
    ∃ vals2 group2,
      dequant.dequant_q4_k_row_loop0_loop0.body scale min_scale sc codes ok1 err vals group =
        ok (ControlFlow.cont (true, none, vals2, group2)) ∧
      q4kGroupInv scale min_scale sc codes init vals2 group2.val true none ∧
      group2.val = group.val + 1 := by
  obtain ⟨i, hi, hiv⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 2#usize) (y := group) (by scalar_tac))
  have hiv' : i.val = 2 * group.val := by rw [hiv]; simp
  have hi8 : i.val < 8 := by rw [hiv']; omega
  obtain ⟨scl, mcl, hscm1, hsclv, hmclv⟩ := q4_k_scale_min_ref sc i hi8
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) (by scalar_tac))
  have hi1v' : i1.val = i.val + 1 := hi1v
  have hi1abs : i1.val = 2 * group.val + 1 := by rw [hi1v', hiv']
  have hi18 : i1.val < 8 := by rw [hi1abs]; omega
  obtain ⟨sch, mch, hscm2, hschv, hmchv⟩ := q4_k_scale_min_ref sc i1 hi18
  let i2 : Std.I64 := UScalar.hcast .I64 scl
  have hi2def : I64.Insts.CoreConvertFromU8.from scl = ok i2 := rfl
  have hi2v : i2.val = (scl.val : ℤ) := hcast_u8_val scl
  have hscl_le : scl.val ≤ 63 := by
    rw [hsclv]; have := q4kScaleMinRef_fst_lt sc.val i.val; omega
  have hi2abs : |i2.val| ≤ 63 := by rw [hi2v]; exact abs_u8_le hscl_le
  let i3 : Std.I64 := UScalar.hcast .I64 mcl
  have hi3def : I64.Insts.CoreConvertFromU8.from mcl = ok i3 := rfl
  have hi3v : i3.val = (mcl.val : ℤ) := hcast_u8_val mcl
  have hmcl_le : mcl.val ≤ 63 := by
    rw [hmclv]; have := q4kScaleMinRef_snd_lt sc.val i.val; omega
  have hi3abs : |i3.val| ≤ 63 := by rw [hi3v]; exact abs_u8_le hmcl_le
  let i4 : Std.I64 := UScalar.hcast .I64 sch
  have hi4def : I64.Insts.CoreConvertFromU8.from sch = ok i4 := rfl
  have hi4v : i4.val = (sch.val : ℤ) := hcast_u8_val sch
  have hsch_le : sch.val ≤ 63 := by
    rw [hschv]; have := q4kScaleMinRef_fst_lt sc.val i1.val; omega
  have hi4abs : |i4.val| ≤ 63 := by rw [hi4v]; exact abs_u8_le hsch_le
  let i5 : Std.I64 := UScalar.hcast .I64 mch
  have hi5def : I64.Insts.CoreConvertFromU8.from mch = ok i5 := rfl
  have hi5v : i5.val = (mch.val : ℤ) := hcast_u8_val mch
  have hmch_le : mch.val ≤ 63 := by
    rw [hmchv]; have := q4kScaleMinRef_snd_lt sc.val i1.val; omega
  have hi5abs : |i5.val| ≤ 63 := by rw [hi5v]; exact abs_u8_le hmch_le
  have hp1 := i64_mul_fits_zb (x := scale.val) (y := i2.val) hscale hi2abs (by norm_num)
  obtain ⟨p1, hprod1, hprod1v⟩ := dequant_mul_ok scale i2 hp1.1 hp1.2
  have hp1abs : |p1.val| ≤ (2 : ℤ) ^ 48 * 63 := by
    rw [hprod1v, abs_mul]
    exact mul_le_mul hscale hi2abs (abs_nonneg _) (by positivity)
  have hp2 := i64_mul_fits_zb (x := min_scale.val) (y := i3.val) hmin hi3abs (by norm_num)
  obtain ⟨p2, hprod2, hprod2v⟩ := dequant_mul_ok min_scale i3 hp2.1 hp2.2
  have hp2abs : |p2.val| ≤ (2 : ℤ) ^ 48 * 63 := by
    rw [hprod2v, abs_mul]
    exact mul_le_mul hmin hi3abs (abs_nonneg _) (by positivity)
  have hp3 := i64_mul_fits_zb (x := scale.val) (y := i4.val) hscale hi4abs (by norm_num)
  obtain ⟨p3, hprod3, hprod3v⟩ := dequant_mul_ok scale i4 hp3.1 hp3.2
  have hp3abs : |p3.val| ≤ (2 : ℤ) ^ 48 * 63 := by
    rw [hprod3v, abs_mul]
    exact mul_le_mul hscale hi4abs (abs_nonneg _) (by positivity)
  have hp4 := i64_mul_fits_zb (x := min_scale.val) (y := i5.val) hmin hi5abs (by norm_num)
  obtain ⟨p4, hprod4, hprod4v⟩ := dequant_mul_ok min_scale i5 hp4.1 hp4.2
  have hp4abs : |p4.val| ≤ (2 : ℤ) ^ 48 * 63 := by
    rw [hprod4v, abs_mul]
    exact mul_le_mul hmin hi5abs (abs_nonneg _) (by positivity)
  have hsL : p1.val = scale.val * ((q4kScaleMinRef sc.val (2 * group.val)).1 : ℤ) := by
    rw [hprod1v, hi2v, hsclv, hiv']
  have hiL : p2.val = min_scale.val * ((q4kScaleMinRef sc.val (2 * group.val)).2 : ℤ) := by
    rw [hprod2v, hi3v, hmclv, hiv']
  have hsH : p3.val = scale.val * ((q4kScaleMinRef sc.val (2 * group.val + 1)).1 : ℤ) := by
    rw [hprod3v, hi4v, hschv, hi1abs]
  have hiH : p4.val = min_scale.val * ((q4kScaleMinRef sc.val (2 * group.val + 1)).2 : ℤ) := by
    rw [hprod4v, hi5v, hmchv, hi1abs]
  have hp1_54 : |p1.val| ≤ 2 ^ 54 := by
    have h : (2 : ℤ) ^ 48 * 63 ≤ 2 ^ 54 := by norm_num
    linarith [hp1abs]
  have hp2_54 : |p2.val| ≤ 2 ^ 54 := by
    have h : (2 : ℤ) ^ 48 * 63 ≤ 2 ^ 54 := by norm_num
    linarith [hp2abs]
  have hp3_54 : |p3.val| ≤ 2 ^ 54 := by
    have h : (2 : ℤ) ^ 48 * 63 ≤ 2 ^ 54 := by norm_num
    linarith [hp3abs]
  have hp4_54 : |p4.val| ≤ 2 ^ 54 := by
    have h : (2 : ℤ) ^ 48 * 63 ≤ 2 ^ 54 := by norm_num
    linarith [hp4abs]
  have hinv0 : q4kLaneInv codes group.val p1.val p2.val p3.val p4.val vals vals 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [q4kLaneAt_zero]
  obtain ⟨vals2, hloop, hlaneinv⟩ :=
    q4k_lane_loop_spec codes group p1 p2 p3 p4 vals vals 0#usize hgroup hcode
      hp1_54 hp2_54 hp3_54 hp4_54 hinv0
  obtain ⟨group2, hgroup2, hgroup2v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := group) (y := 1#usize) (by scalar_tac))
  have hgroup2v' : group2.val = group.val + 1 := hgroup2v
  refine ⟨vals2, group2, ?_, ?_, hgroup2v'⟩
  · unfold dequant.dequant_q4_k_row_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hgroup)]
    rw [hok, herr]
    rw [hi, bind_tc_ok, hscm1, q4k_let_pair, hi1, bind_tc_ok, hscm2, q4k_let_pair]
    rw [if_pos rfl]
    simp only [hi2def, bind_tc_ok]
    simp only [hprod1, bind_tc_ok]
    simp only [hi3def, bind_tc_ok]
    simp only [hprod2, bind_tc_ok]
    simp only [hi4def, bind_tc_ok]
    simp only [hprod3, bind_tc_ok]
    simp only [hi5def, bind_tc_ok]
    simp only [hprod4, bind_tc_ok]
    change (do
        let (ok6, err5, vals1) ←
          dequant.dequant_q4_k_row_loop0_loop0_loop0 true none codes vals group p1 p2 p3 p4 0#usize
        let group1 ← group + 1#usize
        ok (ControlFlow.cont (ok6, err5, vals1, group1)))
      = ok (ControlFlow.cont (true, none, vals2, group2))
    rw [hloop, hgroup2]
    simp only [bind_tc_ok]
    rfl
  · rw [hgroup2v']
    exact q4kGroupInv_succ scale min_scale sc codes init vals vals2 group.val p1 p2 p3 p4
      hsL hiL hsH hiH hgroup (by simpa [hok, herr] using hinv) hlaneinv

/-- The 4-group loop: `loop.spec_decr_nat` with `q4kGroupInv` and measure `4 − group`.

Complexity: 4 iterations, each O(1) products plus one 32-lane pass; O(1) space. -/
theorem q4k_group_loop_spec
    (scale min_scale : Std.I64) (sc : Array Std.U8 12#usize) (codes : Array Std.U8 256#usize)
    (vals : Array Std.I64 256#usize) (group : Std.Usize)
    (hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 15)
    (hscale : |scale.val| ≤ 2 ^ 48) (hmin : |min_scale.val| ≤ 2 ^ 48)
    (hinv : q4kGroupInv scale min_scale sc codes vals vals group.val true none) :
    ∃ vals2,
      dequant.dequant_q4_k_row_loop0_loop0 true none scale min_scale sc codes vals group
        = ok (true, none, vals2) ∧
      q4kGroupInv scale min_scale sc codes vals vals2 4 true none := by
  have hspec : dequant.dequant_q4_k_row_loop0_loop0 true none scale min_scale sc codes vals group
      ⦃ r => q4kGroupInv scale min_scale sc codes vals r.2.2 4 r.1 r.2.1 ⦄ := by
    unfold dequant.dequant_q4_k_row_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Bool × Option error.EngineError × Array Std.I64 256#usize × Std.Usize =>
        4 - s.2.2.2.val)
      (inv := fun s => q4kGroupInv scale min_scale sc codes vals s.2.2.1 s.2.2.2.val s.1 s.2.1)
      (post := fun (r : Bool × Option error.EngineError × Array Std.I64 256#usize) =>
        q4kGroupInv scale min_scale sc codes vals r.2.2 4 r.1 r.2.1)
    · rintro ⟨ok1, err, valsX, gX⟩ hinv'
      dsimp only
      by_cases hlt : gX.val < 4
      · have hstep := q4k_group_body_step scale min_scale sc codes vals valsX gX ok1 err
          hlt hcode hscale hmin hinv'.1 hinv'.2.1 hinv'
        obtain ⟨vals2, g2, hbody, hinv2, hg2⟩ := hstep
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        exact ⟨hinv2, by rw [hg2]; omega⟩
      · have hbody : dequant.dequant_q4_k_row_loop0_loop0.body scale min_scale sc codes ok1 err
            valsX gX = ok (ControlFlow.done (ok1, err, valsX)) := by
          unfold dequant.dequant_q4_k_row_loop0_loop0.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hg4 : gX.val = 4 := by
          have hle : gX.val ≤ 4 := hinv'.2.2.1
          omega
        simpa [hg4] using hinv'
    · exact hinv
  obtain ⟨r, hr, hinv2⟩ := WP.spec_imp_exists hspec
  refine ⟨r.2.2, ?_, ?_⟩
  · rw [hr]
    congr 1
    exact Prod.ext hinv2.1 (Prod.ext hinv2.2.1 rfl)
  · rw [hinv2.1, hinv2.2.1] at hinv2
    exact hinv2

/-! ## `push_prefix` (step c) -/

/-- Invariant of `push_prefix_loop`: `idx` weights appended. -/
def pushInv (vals : Array Std.I64 256#usize) (take : Nat)
    (out0 out : alloc.vec.Vec Std.I64) (idx : Nat) : Prop :=
  idx ≤ take ∧ out.val = out0.val ++ vals.val.take idx

/-- The block-slice window: `q4kBlockVals` entry `i` is `q4kBlockVal` at `i`. -/
lemma q4kBlockVals_getD (raw : Slice Std.U8) (b i : Nat) (hi : i < 256) :
    (q4kBlockVals raw b).getD i 0 =
      q4kBlockVal (f16Value (f16LeHalf raw (144 * b)))
        (f16Value (f16LeHalf raw (144 * b + 2))) (q4kScList raw b) (q4kCodesList raw b) i := by
  unfold q4kBlockVals
  rw [List.getD_eq_getElem?_getD]
  have h : (List.range 256)[i]? = some i := by
    rw [List.getElem?_eq_getElem (by simpa using hi)]
    simp
  rw [List.getElem?_map, h]
  rfl

/-- `q4kWeight` at the global index of block `b`, entry `i`. -/
lemma q4kWeight_eq (raw : Slice Std.U8) (b i : Nat) (hi : i < 256) :
    q4kWeight raw (256 * b + i) = (q4kBlockVals raw b).getD i 0 := by
  unfold q4kWeight
  rw [show (256 * b + i) / 256 = b by omega, show (256 * b + i) % 256 = i by omega]

lemma push_body_step (vals : Array Std.I64 256#usize) (take : Std.Usize)
    (out0 out : alloc.vec.Vec Std.I64) (idx : Std.Usize)
    (htake : take.val ≤ 256) (hcap : out0.val.length + take.val ≤ Usize.max)
    (hlt : idx.val < take.val) (hinv : pushInv vals take.val out0 out idx.val) :
    ∃ out2 idx2,
      dequant.push_prefix_loop.body vals take out idx = ok (ControlFlow.cont (out2, idx2)) ∧
      pushInv vals take.val out0 out2 idx2.val ∧ idx2.val = idx.val + 1 := by
  obtain ⟨hidx, hout⟩ := hinv
  have hvlen : vals.val.length = 256 := Array.property vals
  have houtlen : out.val.length = out0.val.length + idx.val := by
    rw [hout, List.length_append, List.length_take]
    omega
  have hi : idx.val < vals.val.length := by rw [hvlen]; omega
  obtain ⟨v, hv, hvv⟩ := WP.spec_imp_exists (Array.index_usize_spec vals idx hi)
  have hcap' : out.val.length + 1 ≤ Usize.max := by rw [houtlen]; omega
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap')
  obtain ⟨idx2, hidx2eq, hidx2v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := idx) (y := 1#usize) (by scalar_tac))
  have hidx2v' : idx2.val = idx.val + 1 := by rw [hidx2v]; simp
  refine ⟨out2, idx2, ?_, ?_, hidx2v'⟩
  · unfold dequant.push_prefix_loop.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
    simp only [hv, hpush, hidx2eq, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    rw [hpushv, hout, hidx2v', List.take_succ_eq_append_getElem hi, hvv, List.append_assoc]

/-- The `push_prefix` loop: `loop.spec_decr_nat` with `pushInv` and measure `take − idx`. -/
theorem push_prefix_loop_spec (vals : Array Std.I64 256#usize) (take : Std.Usize)
    (out0 out : alloc.vec.Vec Std.I64) (idx : Std.Usize)
    (htake : take.val ≤ 256) (hcap : out0.val.length + take.val ≤ Usize.max)
    (hinv : pushInv vals take.val out0 out idx.val) :
    ∃ out2,
      dequant.push_prefix_loop out vals take idx = ok out2 ∧
      pushInv vals take.val out0 out2 take.val := by
  have hspec : dequant.push_prefix_loop out vals take idx
      ⦃ out2 => pushInv vals take.val out0 out2 take.val ⦄ := by
    unfold dequant.push_prefix_loop
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec Std.I64 × Std.Usize => take.val - s.2.val)
      (inv := fun s => pushInv vals take.val out0 s.1 s.2.val)
      (post := fun out2 => pushInv vals take.val out0 out2 take.val)
    · rintro ⟨outX, idxX⟩ hinv'
      dsimp only
      by_cases hlt : idxX.val < take.val
      · have hstep := push_body_step vals take out0 outX idxX htake hcap hlt hinv'
        obtain ⟨out2, idx2, hbody, hinv2, hidx2⟩ := hstep
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hle : idxX.val ≤ take.val := hinv'.1
        have hmeasure : take.val - idx2.val < take.val - idxX.val := by
          rw [hidx2]; omega
        exact ⟨hinv2, hmeasure⟩
      · have hbody : dequant.push_prefix_loop.body vals take outX idxX
            = ok (ControlFlow.done outX) := by
          unfold dequant.push_prefix_loop.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hle : idxX.val ≤ take.val := hinv'.1
        have hidx : idxX.val = take.val := by omega
        simpa [hidx] using hinv'
    · exact hinv
  obtain ⟨out2, hout2, hinv2⟩ := WP.spec_imp_exists hspec
  exact ⟨out2, hout2, hinv2⟩

/-- `push_prefix` appends exactly the first `take` weights of `vals`.

Complexity: `take` iterations, each O(1) (one array read, one vector push); O(take) words. -/
theorem push_prefix_spec (vals : Array Std.I64 256#usize) (take : Std.Usize)
    (out : alloc.vec.Vec Std.I64) (htake : take.val ≤ 256)
    (hcap : out.val.length + take.val ≤ Usize.max) :
    ∃ out2,
      dequant.push_prefix out vals take = ok out2 ∧
      out2.val = out.val ++ vals.val.take take.val := by
  have hinv0 : pushInv vals take.val out out 0 := ⟨by omega, by simp⟩
  obtain ⟨out2, hloop, hinv2⟩ :=
    push_prefix_loop_spec vals take out out 0#usize htake hcap hinv0
  exact ⟨out2, hloop, hinv2.2⟩

/-! ## Block-window byte reading

`block_slice_ok` returns `blk` with `blk.val = raw.val.slice (144·b) (144·b+144)`. The `getD` of a
`List.slice` is the `getD` of the payload at the shifted index, so the raw-level reference reads
`q4kScList`/`q4kCodesList`/`f16LeHalf` off the same bytes. -/

/-- `getD` of a `List.slice` is the payload `getD` at the shifted index. -/
lemma getD_slice (l : List U8) (s e i : Nat) (he : e ≤ l.length) (hi : i < e - s) :
    (l.slice s e).getD i 0#u8 = l.getD (s + i) 0#u8 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_slice s e i l ⟨he, by omega⟩,
    List.getD_eq_getElem?_getD]

/-- `f16LeHalf` of a 144-byte window at offset `j` is the raw half-word at `s + j`. -/
lemma f16LeHalf_block (raw : Slice U8) (s j : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 144)) (he : s + 144 ≤ raw.val.length)
    (hj : j + 1 < 144) :
    f16LeHalf blk j = f16LeHalf raw (s + j) := by
  unfold f16LeHalf
  congr 2
  rw [h, getD_slice raw.val s (s + 144) j he (by omega),
    getD_slice raw.val s (s + 144) (j + 1) he (by omega)]
  simp only [Nat.add_assoc]

/-- The 144-byte window is `(drop (144·b)).take 144`. -/
lemma q4kBlockBytes_slice (raw : Slice U8) (b : Nat) :
    q4kBlockBytes raw b = raw.val.slice (144 * b) (144 * b + 144) := by
  unfold q4kBlockBytes List.slice
  congr 1
  omega

/-- The scale-table window's `getD` is the raw `getD` at the shifted index. -/
lemma q4kScList_getD (raw : Slice U8) (b m : Nat) (hm : m < 12) :
    ((q4kScList raw b).getD m 0#u8).val = (raw.val.getD (144 * b + 4 + m) 0#u8).val := by
  unfold q4kScList
  rw [getD_drop_take raw.val (144 * b + 4) 12 m 0#u8 hm]

/-- The decoded codes list's entry `i` is `q4CodesRef` of the block bytes. -/
lemma q4kCodesList_getD (raw : Slice U8) (b i : Nat) (hi : i < 256) :
    ((q4kCodesList raw b).getD i 0#u8).val = q4CodesRef (q4kBlockBytes raw b) i := by
  unfold q4kCodesList
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[i]? = some i := by
    rw [List.getElem?_eq_getElem (by simpa using hi)]
    simp
  rw [h]
  simp only [Option.map_some, Option.getD_some, U8.ofNatCore_val_eq]

/-- `q4kScaleMinRef` depends only on the byte values at indices below twelve. -/
lemma q4kScaleMinRef_congr (sc sc' : List U8) (j : Nat) (hj : j < 8)
    (h : ∀ m, m < 12 → (sc.getD m 0#u8).val = (sc'.getD m 0#u8).val) :
    q4kScaleMinRef sc j = q4kScaleMinRef sc' j := by
  unfold q4kScaleMinRef
  by_cases h4 : j < 4
  · rw [if_pos h4, if_pos h4, h j (by omega), h (j + 4) (by omega)]
  · rw [if_neg h4, if_neg h4, h (j + 4) (by omega), h (j - 4) (by omega), h j (by omega)]

/-- The raw-level `q4kBlockVal` on the block window equals the decoded-array reference. -/
lemma q4kBlockVal_bridge (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (hslice : blk.val = raw.val.slice (144 * b) (144 * b + 144))
    (he : 144 * b + 144 ≤ raw.val.length)
    (scale min_scale : I64) (sc : Array U8 12#usize) (codes : Array U8 256#usize)
    (hd : scale.val = f16Value (f16LeHalf raw (144 * b)))
    (hdmin : min_scale.val = f16Value (f16LeHalf raw (144 * b + 2)))
    (hsc : ∀ k, k < 12 → (sc.val.getD k 0#u8).val = (blk.val.getD (4 + k) 0#u8).val)
    (hcv : ∀ i, i < 256 → (codes.val.getD i 0#u8).val = q4CodesRef blk.val i)
    (j : Nat) (hj : j < 256) :
    q4kBlockVal scale.val min_scale.val sc.val codes.val j =
      q4kBlockVal (f16Value (f16LeHalf raw (144 * b)))
        (f16Value (f16LeHalf raw (144 * b + 2))) (q4kScList raw b) (q4kCodesList raw b) j := by
  have hsc' : ∀ m, m < 12 →
      (sc.val.getD m 0#u8).val = ((q4kScList raw b).getD m 0#u8).val := by
    intro m hm
    rw [hsc m hm, hslice, getD_slice raw.val (144 * b) (144 * b + 144) (4 + m) he (by omega),
      q4kScList_getD raw b m hm]
    simp only [Nat.add_assoc]
  have hcodes' : ∀ i, i < 256 →
      (codes.val.getD i 0#u8).val = ((q4kCodesList raw b).getD i 0#u8).val := by
    intro i hi
    rw [hcv i hi, q4kCodesList_getD raw b i hi, q4kBlockBytes_slice raw b, ← hslice]
  have hsref : ∀ k, k < 8 → q4kScaleMinRef sc.val k = q4kScaleMinRef (q4kScList raw b) k :=
    fun k hk => q4kScaleMinRef_congr sc.val (q4kScList raw b) k hk hsc'
  have hif : (if j % 64 < 32 then q4kScaleMinRef sc.val (2 * (j / 64))
        else q4kScaleMinRef sc.val (2 * (j / 64) + 1))
      = (if j % 64 < 32 then q4kScaleMinRef (q4kScList raw b) (2 * (j / 64))
        else q4kScaleMinRef (q4kScList raw b) (2 * (j / 64) + 1)) := by
    by_cases h : j % 64 < 32
    · rw [if_pos h, if_pos h]
      exact hsref _ (by omega)
    · rw [if_neg h, if_neg h]
      exact hsref _ (by omega)
  unfold q4kBlockVal
  rw [hd, hdmin, hif, hcodes' j hj]

/-! ## The Q4_K row success domain and loop invariant -/

/-- The success domain of one 144-byte block `b`: the window is the raw slice
`[144·b, 144·b+144)` and `d`/`dmin` are finite. The remaining checked-`i64` fits follow from
`f16Value_abs_lt` and `q4CodesRef_lt_16` in the body step. -/
def q4kBlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  ∃ blk : Slice U8,
    dequant.block_slice raw b 144#usize = ok (.Ok blk) ∧
    blk.val = raw.val.slice (144 * b.val) (144 * b.val + 144) ∧
    144 * b.val + 144 ≤ raw.val.length ∧
    144 ≤ blk.val.length ∧
    f16Exp (f16LeHalf blk 0) ≠ 31 ∧
    f16Exp (f16LeHalf blk 2) ≠ 31

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (256·block) n` raw weights, with the success flag set. -/
def q4kRowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q4kRowValues raw (Nat.min (256 * block.val) n.val)

/-- Appending the next `take` raw weights advances the row prefix by `take`. -/
lemma q4kRowValues_append (raw : Slice U8) (m take : Nat) :
    q4kRowValues raw m ++ (List.range take).map (fun j => q4kWeight raw (m + j)) =
      q4kRowValues raw (m + take) := by
  unfold q4kRowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- The signed-`i64` default `getD` lemma. -/
lemma getD_of_lt_i64 {l : List I64} {i : Nat} (h : i < l.length) : l.getD i 0#i64 = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- The value-prefix of a long-enough list is the range map of its `getD` values. -/
lemma map_take_val_eq (l : List I64) (n : Nat) (hn : n ≤ l.length) :
    (l.take n).map (fun v : I64 => v.val) =
      (List.range n).map (fun j => (l.getD j 0#i64).val) := by
  rw [List.map_take]
  apply List.ext_getElem
  · simp [List.length_take, Nat.min_eq_left hn]
  · intro k h1 h2
    have hk : k < n := by
      simpa [List.length_take, Nat.min_eq_left hn, List.length_map] using h1
    have hkl : k < l.length := by omega
    simp only [List.length_map] at h2
    rw [List.getElem_take, List.getElem_map, List.getElem_map, List.getElem_range]
    rw [getD_of_lt_i64 hkl]

/-- `block < (n + 255)/256` gives `256·block < n`: the block loop's divisibility bound. -/
lemma lt_mul_of_lt_div_ceil_256 {b n : Nat} (hb : b < (n + 255) / 256) : 256 * b < n := by
  by_contra h
  have hle : n ≤ 256 * b := not_lt.mp h
  have h1 : (n + 255) / 256 ≤ (256 * b + 255) / 256 := Nat.div_le_div_right (by omega)
  have h2 : (256 * b + 255) / 256 = b := by omega
  omega

/-- `256·b + min 256 (n - 256·b) = min (256·(b+1)) n` when `256·b ≤ n`. -/
lemma mul_block_add_min (b n : Nat) (h : 256 * b ≤ n) :
    256 * b + Nat.min 256 (n - 256 * b) = Nat.min (256 * (b + 1)) n := by
  change 256 * b + min 256 (n - 256 * b) = min (256 * (b + 1)) n
  by_cases h1 : n - 256 * b ≤ 256
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 256 < n - 256 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

/-- `n ≤ 256 · ceil(n/256)`. -/
lemma le_mul_div_ceil_256 (n : Nat) : n ≤ 256 * ((n + 255) / 256) := by omega

/-- `core.cmp.Ord.min` on `usize` returns the smaller operand. -/
theorem q4k_min_ok (x y : Usize) :
    ∃ z : Usize, core.cmp.Ord.min.trait_default core.cmp.OrdUsize x y = ok z ∧
      z.val = Nat.min x.val y.val := by
  unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body
  simp only [core.cmp.impls.PartialOrdUsize.lt, liftFun2, bind_tc_ok]
  by_cases h : y.val < x.val
  · refine ⟨y, by simp [h], ?_⟩
    exact (Nat.min_eq_right (Nat.le_of_lt h)).symm
  · refine ⟨x, by simp [h], ?_⟩
    exact (Nat.min_eq_left (Nat.le_of_not_lt h)).symm

/-- The extracted `div_ceil n QK_K` succeeds and its value is `(n + 255)/256`. -/
theorem dequant_div_ceil_256_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.QK_K = ok blocks ∧
      blocks.val = (n.val + 255) / 256 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.QK_K) (by simp only [quant.QK_K]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.QK_K) (by simp only [quant.QK_K]; decide))
  have hqv' : q.val = n.val / 256 := by rw [hqv]; simp [quant.QK_K]
  have hrv' : r.val = n.val % 256 := by rw [hrv]; simp [quant.QK_K]
  have hq' : UScalar.div n quant.QK_K = ok q := hq
  have hr' : UScalar.rem n quant.QK_K = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 256 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by
        rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 256 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-- One block-loop body step on the success domain: decode block `block`, run the group loop,
append `take = min 256 (n - 256·block)` values and advance. -/
theorem q4kBlock_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : q4kBlockDomain raw block)
    (hcap : out.val.length + (n.val - 256 * block.val) ≤ Usize.max)
    (hinv : q4kRowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.dequant_q4_k_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q4kRowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨blk, hblk, hslice, hbnd, h144, hd, hdmin⟩ := hdom
  obtain ⟨d, dmin, sc, codes, hq4, hdv, hdminv, hscv, hcv⟩ :=
    q4_k_block_spec raw block blk hblk h144 hd hdmin
  have hdv' : d.val = f16Value (f16LeHalf raw (144 * block.val)) := by
    rw [hdv, f16LeHalf_block raw (144 * block.val) 0 blk hslice hbnd (by norm_num)]
    simp only [Nat.add_zero]
  have hdminv' : dmin.val = f16Value (f16LeHalf raw (144 * block.val + 2)) := by
    rw [hdminv, f16LeHalf_block raw (144 * block.val) 2 blk hslice hbnd (by norm_num)]
  have hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 15 := by
    intro i hi
    have h := q4CodesRef_lt_16 blk.val i
    rw [hcv i hi]
    omega
  have hfin0 : f16Exp (f16LeHalf raw (144 * block.val)) ≠ 31 := by
    have h : f16LeHalf raw (144 * block.val) = f16LeHalf blk 0 := by
      simpa only [Nat.add_zero] using
        (f16LeHalf_block raw (144 * block.val) 0 blk hslice hbnd (by norm_num)).symm
    rw [h]; exact hd
  have hfin2 : f16Exp (f16LeHalf raw (144 * block.val + 2)) ≠ 31 := by
    have h := f16LeHalf_block raw (144 * block.val) 2 blk hslice hbnd (by norm_num)
    rw [← h]; exact hdmin
  have hscale : |d.val| ≤ 2 ^ 48 := by
    rw [hdv']
    exact le_of_lt (f16Value_abs_lt (f16LeHalf raw (144 * block.val)) hfin0)
  have hmin : |dmin.val| ≤ 2 ^ 48 := by
    rw [hdminv']
    exact le_of_lt (f16Value_abs_lt (f16LeHalf raw (144 * block.val + 2)) hfin2)
  have hltmul : 256 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hblock)
  have hinv0 : q4kGroupInv d dmin sc codes (Array.repeat 256#usize 0#i64)
      (Array.repeat 256#usize 0#i64) 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ j < 64 * 0)]
  obtain ⟨vals1, hloop, hgrp⟩ :=
    q4k_group_loop_spec d dmin sc codes (Array.repeat 256#usize 0#i64) 0#usize
      hcode hscale hmin hinv0
  have hloop' : dequant.dequant_q4_k_row_loop0_loop0 true none d dmin sc codes
      (Array.repeat 256#usize 0#i64) 0#usize = ok (true, none, vals1) := hloop
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.QK_K) (by simp only [quant.QK_K]; scalar_tac))
  have hivv' : i.val = 256 * block.val := by rw [hivv]; simp [quant.QK_K]; ring
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 256 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q4k_min_ok quant.QK_K i1
  have htakev' : take.val = Nat.min 256 (n.val - 256 * block.val) := by
    rw [htakev, hi1v']; simp [quant.QK_K]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 256 := by rw [htakev']; exact min_le_left _ _
  have htake_X : take.val ≤ n.val - 256 * block.val := by rw [htakev']; exact min_le_right _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by omega
  obtain ⟨out1, hpush, hpushv⟩ := push_prefix_spec vals1 take out htake_le hcap'
  have hpushv' : dequantVals out1.val = dequantVals out.val ++
      (List.range take.val).map (fun j => (vals1.val.getD j 0#i64).val) := by
    rw [hpushv, dequantVals, List.map_append, ← dequantVals]
    congr 1
    exact map_take_val_eq vals1.val take.val (by rw [Array.property vals1]; exact htake_le)
  have hgrp_j : ∀ j, j < 256 → (vals1.val.getD j 0#i64).val =
      (q4kBlockVals raw block.val).getD j 0 := by
    intro j hj
    have h := hgrp.2.2.2 j hj
    rw [if_pos (by omega : j < 64 * 4)] at h
    rw [h, q4kBlockVals_getD raw block.val j hj]
    exact q4kBlockVal_bridge raw block.val blk hslice hbnd d dmin sc codes
      hdv' hdminv' hscv hcv j hj
  have hmap : (List.range take.val).map (fun j => (vals1.val.getD j 0#i64).val) =
      (List.range take.val).map (fun j => q4kWeight raw (256 * block.val + j)) := by
    apply List.map_congr_left
    intro j hj
    have hj' : j < take.val := List.mem_range.mp hj
    have hj256 : j < 256 := by omega
    rw [hgrp_j j hj256, q4kWeight_eq raw block.val j hj256]
  have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 256 * block.val + take.val = Nat.min (256 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals out1.val = dequantVals out0.val ++
      q4kRowValues raw (Nat.min (256 * block1.val) n.val) := by
    rw [hblock1v']
    rw [hpushv', hmap, hinv.2.2.2, hmin, List.append_assoc, q4kRowValues_append, hsum]
  refine ⟨out1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.dequant_q4_k_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq4]
  simp only [bind_tc_ok]
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  change (do
    let (ok2, err1, vals1) ←
      dequant.dequant_q4_k_row_loop0_loop0 true none d dmin sc codes
        (Array.repeat 256#usize 0#i64) 0#usize
    let out1 ← if ok2 = true then dequant.push_prefix out vals1 take else ok out
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (out1, block1, true, none))
  rw [hloop']
  simp only [bind_tc_ok]
  change (do
    let out1 ← if (true : Bool) = true then dequant.push_prefix out vals1 take else ok out
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, true, none)))
    = ok (ControlFlow.cont (out1, block1, true, none))
  rw [hpush]
  rw [hblock1a]
  rw [if_pos rfl]
  simp only [bind_tc_ok]

/-- The block loop: `loop.spec_decr_nat` with `q4kRowInv` and measure `blocks - block`, on the
success domain `q4kBlockDomain` for every block below `blocks`.

Complexity: `blocks` iterations, each a 144-byte read plus 4 groups of 32 lanes; the measure is
`blocks - block`. -/
theorem q4kRow_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : ∀ b : Usize, b.val < blocks.val → q4kBlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_q4_k_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q4kRowValues raw n.val) := by
  unfold dequant.dequant_q4_k_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q4kRowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 256 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 256 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, q4kRowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 256 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q4kBlock_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.dequant_q4_k_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_q4_k_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (256 * ((n.val + 255) / 256)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_256 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (256 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [q4kRowValues]

/-- **`Q4_K` row law.** On the block success domain, the extracted `dequant_q4_k_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `q4kRowValues raw n` to the output. -/
theorem dequant_q4_k_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 255) / 256 → q4kBlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q4_k_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q4kRowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_256_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q4kRow_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.dequant_q4_k_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl


end apeiron_word
