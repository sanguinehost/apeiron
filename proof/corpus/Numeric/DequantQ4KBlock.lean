/-
The reference and block-decoding half of the extracted `Q4_K` dequantizer refinement
(`APEIRON-ALGEBRA-G3-DEQUANT-W2-T1`), split out of `Numeric.DequantQ4K` under the 1500-line source
cap (AGENTS §4). It holds `q4kScaleMinRef`/`q4_k_scale_min_ref`, the block reader
`read_scale_ok`/`q4_k_block_loop_spec`/`q4_k_block_spec`/`dequant_row_fixed_q4_k`, the raw-level row
references `q4CodesRef_lt_16`/`q4kBlockVal`/`q4kScList`/`q4kBlockBytes`/`q4kCodesList`/
`q4kBlockVals`/`q4kWeight`/`q4kRowValues`, and the innermost 32-lane loop
`q4k_lane_body_step`/`q4k_lane_loop_spec` with the checked-`i64` bounds. The group loop,
`push_prefix`, the raw-window bridge and the row law `dequant_q4_k_row_spec` continue in
`Numeric.DequantQ4K`.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is not used on extracted
bodies.
-/
import Numeric.DequantRefinement
import Numeric.FormatRefinement
import Numeric.PackingRefinement

namespace apeiron_word
open Aeneas Aeneas.Std Result

def q4kScaleMinRef (sc : List Std.U8) (j : Nat) : Nat × Nat :=
  if j < 4 then
    ((sc.getD j 0#u8).val &&& 63, (sc.getD (j + 4) 0#u8).val &&& 63)
  else
    (((sc.getD (j + 4) 0#u8).val &&& 15) ||| ((sc.getD (j - 4) 0#u8).val >>> 6 <<< 4 % 256),
     ((sc.getD (j + 4) 0#u8).val >>> 4) ||| ((sc.getD j 0#u8).val >>> 6 <<< 4 % 256))

lemma list_getD_getElem {α : Type} (l : List α) (i : Nat) (h : i < l.length) (d : α) :
    (l[i]'h) = l.getD i d := by
  simp [List.getD, List.getElem?_eq_getElem h]

lemma arr_getD_val (a : Array Std.U8 12#usize) (i : Std.Usize) (h : i.val < a.val.length) :
    (a.val[i.val]'h).val = (a.val.getD i.val 0#u8).val := by
  rw [list_getD_getElem a.val i.val h 0#u8]

lemma slice_index_getD (blk : Slice Std.U8) (i : Std.Usize) (h : i.val < blk.val.length) :
    ∃ v : Std.U8, Slice.index_usize blk i = ok v ∧ v.val = (blk.val.getD i.val 0#u8).val := by
  obtain ⟨v, hv, hvv⟩ := WP.spec_imp_exists (Slice.index_usize_spec blk i h)
  refine ⟨v, hv, ?_⟩
  rw [hvv]
  exact congrArg (fun u : Std.U8 => u.val) (list_getD_getElem blk.val i.val h 0#u8)

theorem q4_k_scale_min_ref (sc : Array Std.U8 12#usize) (j : Std.Usize) (hj : j.val < 8) :
    ∃ a b : Std.U8, quant.q4_k_scale_min sc j = ok (a, b) ∧
      a.val = (q4kScaleMinRef sc.val j.val).1 ∧
      b.val = (q4kScaleMinRef sc.val j.val).2 := by
  have h12 : sc.val.length = 12 := Array.property sc
  have hget : ∀ (i : Std.Usize), i.val < 12 →
      ∃ v : Std.U8, sc.index_usize i = ok v ∧ v.val = (sc.val.getD i.val 0#u8).val := by
    intro i hi
    obtain ⟨v, hv, hvv⟩ :=
      WP.spec_imp_exists (Array.index_usize_spec sc i (by show i.val < sc.val.length; omega))
    exact ⟨v, hv, by rw [hvv]; exact arr_getD_val sc i (by omega)⟩
  rw [quant.q4_k_scale_min]
  by_cases hj4 : j < 4#usize
  · rw [if_pos hj4]
    have hj4' : j.val < 4 := by scalar_tac
    rw [q4kScaleMinRef, if_pos hj4']
    simp only [lift, bind_tc_ok]
    obtain ⟨v, hv, hvv⟩ := hget j (by omega)
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 4#usize) (by scalar_tac))
    have hi2v' : i2.val = j.val + 4 := by rw [hi2v]; simp
    obtain ⟨v2, hv2, hv2v⟩ := hget i2 (by rw [hi2v']; omega)
    refine ⟨v &&& 63#u8, v2 &&& 63#u8, ?_, ?_, ?_⟩
    · simp only [hv, hi2, hv2, bind_tc_ok]
    · rw [UScalar.val_and, hvv]
      simp
    · rw [UScalar.val_and, hv2v, hi2v']
      simp
  · rw [if_neg hj4]
    have hj4' : ¬ j.val < 4 := by scalar_tac
    rw [q4kScaleMinRef, if_neg hj4']
    simp only [lift, bind_tc_ok]
    obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := j) (y := 4#usize) (by scalar_tac))
    have hiv' : i.val = j.val + 4 := by rw [hiv]; simp
    obtain ⟨i1, hi1, hi1v⟩ := hget i (by rw [hiv']; omega)
    obtain ⟨i2, hi2, hi2v, -⟩ := WP.spec_imp_exists (UScalar.and_spec i1 15#u8)
    obtain ⟨i3, hi3, hi3v, -⟩ := WP.spec_imp_exists (Usize.sub_spec (x := j) (y := 4#usize) (by scalar_tac))
    have hi3' : i3.val = j.val - 4 := by rw [hi3v]; simp
    obtain ⟨i4, hi4, hi4v⟩ := hget i3 (by rw [hi3']; omega)
    obtain ⟨i5, hi5, hi5v, -⟩ := WP.spec_imp_exists
      (U8.ShiftRight_IScalar_spec i4 6#i32 (by simp) (by simp))
    obtain ⟨i6, hi6, hi6v, -⟩ := WP.spec_imp_exists
      (U8.ShiftLeft_IScalar_spec i5 4#i32 (by simp) (by simp))
    obtain ⟨i7, hi7, hi7v, -⟩ := WP.spec_imp_exists (UScalar.or_spec i2 i6)
    obtain ⟨i8, hi8, hi8v⟩ := hget i (by rw [hiv']; omega)
    have hi8i1 : i8 = i1 := by rw [hi1] at hi8; simpa using hi8.symm
    subst i8
    obtain ⟨i9, hi9, hi9v, -⟩ := WP.spec_imp_exists
      (U8.ShiftRight_IScalar_spec i1 4#i32 (by simp) (by simp))
    obtain ⟨i10, hi10, hi10v⟩ := hget j (by omega)
    obtain ⟨i11, hi11, hi11v, -⟩ := WP.spec_imp_exists
      (U8.ShiftRight_IScalar_spec i10 6#i32 (by simp) (by simp))
    obtain ⟨i12, hi12, hi12v, -⟩ := WP.spec_imp_exists
      (U8.ShiftLeft_IScalar_spec i11 4#i32 (by simp) (by simp))
    obtain ⟨i13, hi13, hi13v, -⟩ := WP.spec_imp_exists (UScalar.or_spec i9 i12)
    refine ⟨i2 ||| i6, i9 ||| i12, ?_, ?_, ?_⟩
    · have h2 : i2 = i1 &&& 15#u8 := by
        have h := hi2
        simp only [lift, Result.ok.injEq] at h
        exact h.symm
      simp only [hi, hi1, hi3, hi4, hi5, hi6, hi9, hi10, hi11, hi12, bind_tc_ok]
      rw [h2]
    · rw [UScalar.val_or, hi2v, UScalar.val_and, hi6v, hi5v, hi4v, hi1v, hiv', hi3']
      simp [U8.size, U8.numBits]
    · rw [UScalar.val_or, hi9v, hi12v, hi11v, hi10v, hi1v, hiv']
      simp [U8.size, U8.numBits]

theorem read_scale_ok (blk : Slice Std.U8) (i : Std.Usize) (hi : i.val + 2 ≤ blk.val.length)
    (hf : f16Exp (f16LeHalf blk i.val) ≠ 31) :
    ∃ v : Std.I64, dequant.read_scale blk i = ok (.Ok v) ∧
      v.val = f16Value (f16LeHalf blk i.val) := by
  obtain ⟨v, hv, hvv⟩ := f16_to_fixed_exact (f16LeHalf blk i.val) hf
  have hu := read_u16_ok blk i hi
  refine ⟨v, ?_, hvv⟩
  rw [dequant.read_scale, hu]
  simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
  exact hv

def scLoopInv (blk : Slice Std.U8) (sc0 : Array Std.U8 12#usize)
    (sc : Array Std.U8 12#usize) (kv : Nat) : Prop :=
  kv ≤ 12 ∧ ∀ j, j < 12 → (sc.val.getD j 0#u8).val =
    (if j < kv then (blk.val.getD (4 + j) 0#u8).val else (sc0.val.getD j 0#u8).val)

theorem q4_k_block_loop_spec (blk : Slice Std.U8) (sc0 : Array Std.U8 12#usize) (k : Std.Usize)
    (hblk : 16 ≤ blk.val.length) (hinv : scLoopInv blk sc0 sc0 k.val) :
    ∃ sc, dequant.q4_k_block_loop blk sc0 k = ok sc ∧ scLoopInv blk sc0 sc 12 := by
  have hspec : dequant.q4_k_block_loop blk sc0 k ⦃ sc => scLoopInv blk sc0 sc 12 ⦄ := by
    unfold dequant.q4_k_block_loop
    apply loop.spec_decr_nat
      (measure := fun s : Array Std.U8 12#usize × Std.Usize => 12 - s.2.val)
      (inv := fun s => scLoopInv blk sc0 s.1 s.2.val)
      (post := fun sc => scLoopInv blk sc0 sc 12)
    · rintro ⟨sc, kk⟩ hinv'
      change scLoopInv blk sc0 sc kk.val at hinv'
      have hinv0 := hinv'
      obtain ⟨hkv, hrec⟩ := hinv'
      dsimp only
      by_cases hlt : kk < 12#usize
      · have hklt : kk.val < 12 := by rw [UScalar.lt_equiv] at hlt; exact hlt
        obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := 4#usize) (y := kk) (by scalar_tac))
        have hiv' : i.val = 4 + kk.val := by rw [hiv]; simp
        obtain ⟨i1, hi1, hi1v⟩ := slice_index_getD blk i (by rw [hiv']; omega)
        have hub : kk.val < sc.length := by scalar_tac
        obtain ⟨a, ha, hav⟩ := WP.spec_imp_exists (Array.update_spec sc kk i1 hub)
        obtain ⟨k1, hk1, hk1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := kk) (y := 1#usize) (by scalar_tac))
        have hk1v' : k1.val = kk.val + 1 := by rw [hk1v]; simp
        have hbody : dequant.q4_k_block_loop.body blk sc kk = ok (ControlFlow.cont (a, k1)) := by
          unfold dequant.q4_k_block_loop.body
          rw [quant.Q4_K_SCALE_SIZE, if_pos hlt]
          simp only [hi, hi1, ha, hk1, bind_tc_ok]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        show scLoopInv blk sc0 a k1.val ∧ 12 - k1.val < 12 - kk.val
        refine ⟨⟨by rw [hk1v']; omega, ?_⟩, by rw [hk1v']; omega⟩
        intro j hj
        have h12 : sc.val.length = 12 := by simp [Array.property]
        have hav' : a.val = sc.val.set kk.val i1 := by rw [hav, Array.set_val_eq]
        rw [hav', getD_set]
        by_cases hjk : j = kk.val
        · subst hjk
          rw [if_pos ⟨rfl, by rw [h12]; omega⟩, hk1v', hi1v, hiv',
            if_pos (by omega : kk.val < kk.val + 1)]
        · rw [if_neg (fun h => hjk h.1), hrec j hj]
          by_cases hjlt : j < kk.val
          · rw [if_pos hjlt, if_pos (by omega)]
          · rw [if_neg hjlt, if_neg (by omega : ¬ j < k1.val)]
      · have hk12 : kk.val = 12 := by
          have h2 : ¬ kk.val < 12 := by intro h; exact hlt ((UScalar.lt_equiv kk 12#usize).2 h)
          omega
        have hbody : dequant.q4_k_block_loop.body blk sc kk = ok (ControlFlow.done sc) := by
          unfold dequant.q4_k_block_loop.body
          rw [quant.Q4_K_SCALE_SIZE, if_neg hlt]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        show scLoopInv blk sc0 sc 12
        simpa [hk12] using hinv0
    · exact hinv
  obtain ⟨sc, hsc, hinvsc⟩ := WP.spec_imp_exists hspec
  exact ⟨sc, hsc, hinvsc⟩


theorem q4_k_block_spec (raw : Slice Std.U8) (block : Std.Usize) (blk : Slice Std.U8)
    (hblk : dequant.block_slice raw block 144#usize = ok (.Ok blk))
    (h144 : 144 ≤ blk.val.length)
    (hd : f16Exp (f16LeHalf blk 0) ≠ 31)
    (hdmin : f16Exp (f16LeHalf blk 2) ≠ 31) :
    ∃ d dmin sc codes,
      dequant.q4_k_block raw block = ok (.Ok (d, dmin, sc, codes)) ∧
      d.val = f16Value (f16LeHalf blk 0) ∧
      dmin.val = f16Value (f16LeHalf blk 2) ∧
      (∀ k, k < 12 → (sc.val.getD k 0#u8).val = (blk.val.getD (4 + k) 0#u8).val) ∧
      (∀ i, i < 256 → (codes.val.getD i 0#u8).val = q4CodesRef blk.val i) := by
  have h16 : 16 ≤ blk.val.length := by omega
  obtain ⟨d, hd_eq, hdv⟩ := read_scale_ok blk 0#usize (by simp; omega) hd
  obtain ⟨dmin, hdmin_eq, hdminv⟩ := read_scale_ok blk 2#usize (by simp; omega) hdmin
  have hinv0 : scLoopInv blk (Array.repeat 12#usize 0#u8) (Array.repeat 12#usize 0#u8) 0 := by
    refine ⟨by norm_num, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ j < 0)]
  obtain ⟨sc, hsc_eq, hsc_inv⟩ :=
    q4_k_block_loop_spec blk (Array.repeat 12#usize 0#u8) 0#usize h16 hinv0
  obtain ⟨codes, hcodes_eq, hcodes⟩ := word_q4_k_codes_ok blk (by
    rw [UScalar.le_equiv]; simpa [Slice.len] using h144)
  have hcodes_eq' : quant.q4_k_codes blk = ok (.Ok codes) := hcodes_eq
  refine ⟨d, dmin, sc, codes, ?_, hdv, hdminv, ?_, hcodes⟩
  · simp only [dequant.q4_k_block, quant_Q4_K_BLOCK_BYTES_eq, hblk, hd_eq, hdmin_eq, hsc_eq,
      hcodes_eq', core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  · intro k hk
    have := hsc_inv.2 k hk
    rwa [if_pos hk] at this


/-- `dequant_row_fixed` at `ggml_type = 12` (`GGML_TYPE_Q4_K`) is the `Q4_K` row. -/
theorem dequant_row_fixed_q4_k (raw : Slice Std.U8) (n : Std.Usize) (out : alloc.vec.Vec Std.I64) :
    dequant.dequant_row_fixed 12#u32 raw n out = dequant.dequant_q4_k_row raw n out := rfl

/-! ## Lane-loop refinement (step a)

The innermost 32-lane loop `dequant_q4_k_row_loop0_loop0_loop0` writes
`vals[64·g + lane] = slope_lo·code_lo − intercept_lo` and
`vals[64·g + 32 + lane] = slope_hi·code_hi − intercept_hi` for `lane < 32`. `q4kLaneAt` is the
closed-form reference after `l` lanes; `q4kLaneVal` is the value written at one lane. -/

/-- The exact `ℤ` value written at lane `lane` of group `g`. -/
def q4kLaneVal (codes : Array Std.U8 256#usize) (g lane : Nat) (sL iL sH iH : ℤ) : ℤ :=
  (if lane < 32 then sL else sH) * ((codes.val.getD (64 * g + lane) 0#u8).val : ℤ)
    - (if lane < 32 then iL else iH)

/-- The closed-form `vals` entry at `j` after `l` lanes of group `g` are written. -/
def q4kLaneAt (codes : Array Std.U8 256#usize) (g l : Nat) (sL iL sH iH : ℤ)
    (vals0 : Array Std.I64 256#usize) (j : Nat) : ℤ :=
  if j < 64 * g then (vals0.val.getD j 0#i64).val
  else if j / 64 = g ∧ (j % 64 < l ∨ (32 ≤ j % 64 ∧ j % 64 < 32 + l))
    then q4kLaneVal codes g (j % 64) sL iL sH iH
    else (vals0.val.getD j 0#i64).val

/-- Invariant of the lane loop: `l` lanes of group `g` written, `vals` pointwise `q4kLaneAt`. -/
def q4kLaneInv (codes : Array Std.U8 256#usize) (g : Nat) (sL iL sH iH : ℤ)
    (vals0 vals : Array Std.I64 256#usize) (l : Nat) (ok1 : Bool)
    (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ l ≤ 32 ∧
  ∀ j, j < 256 → (vals.val.getD j 0#i64).val = q4kLaneAt codes g l sL iL sH iH vals0 j

lemma q4kLaneVal_low (codes : Array Std.U8 256#usize) (g l : Nat) (sL iL sH iH : ℤ)
    (hl : l < 32) :
    q4kLaneVal codes g l sL iL sH iH
      = sL * ((codes.val.getD (64 * g + l) 0#u8).val : ℤ) - iL := by
  unfold q4kLaneVal
  rw [if_pos hl, if_pos hl]

lemma q4kLaneVal_high (codes : Array Std.U8 256#usize) (g l : Nat) (sL iL sH iH : ℤ)
    (hl : 32 ≤ l) :
    q4kLaneVal codes g l sL iL sH iH
      = sH * ((codes.val.getD (64 * g + l) 0#u8).val : ℤ) - iH := by
  unfold q4kLaneVal
  rw [if_neg (by omega), if_neg (by omega)]

lemma q4kLaneAt_step (codes : Array Std.U8 256#usize) (g l : Nat) (sL iL sH iH : ℤ)
    (vals0 : Array Std.I64 256#usize) (j : Nat)
    (h1 : j ≠ 64 * g + l) (h2 : j ≠ 64 * g + 32 + l) :
    q4kLaneAt codes g (l + 1) sL iL sH iH vals0 j
      = q4kLaneAt codes g l sL iL sH iH vals0 j := by
  unfold q4kLaneAt
  by_cases hlt : j < 64 * g
  · rw [if_pos hlt, if_pos hlt]
  · rw [if_neg hlt, if_neg hlt]
    by_cases hb : j / 64 = g
    · have hlo : j % 64 ≠ l := by intro h; exact h1 (by omega)
      have hhi : j % 64 ≠ 32 + l := by intro h; exact h2 (by omega)
      by_cases hA : j % 64 < l
      · rw [if_pos ⟨hb, Or.inl (by omega : j % 64 < l + 1)⟩, if_pos ⟨hb, Or.inl hA⟩]
      · by_cases hB : 32 ≤ j % 64 ∧ j % 64 < 32 + l
        · rw [if_pos ⟨hb, Or.inr ⟨hB.1, by omega⟩⟩, if_pos ⟨hb, Or.inr hB⟩]
        · have hneg1 : ¬ (j % 64 < l + 1 ∨ 32 ≤ j % 64 ∧ j % 64 < 32 + (l + 1)) := by
            rintro (h | h) <;> omega
          have hneg2 : ¬ (j % 64 < l ∨ 32 ≤ j % 64 ∧ j % 64 < 32 + l) := by
            rintro (h | h) <;> [exact hA h; exact hB h]
          rw [if_neg (fun h => hneg1 h.2), if_neg (fun h => hneg2 h.2)]
    · rw [if_neg (fun h => hb h.1), if_neg (fun h => hb h.1)]

lemma q4kLaneAt_low (codes : Array Std.U8 256#usize) (g l : Nat) (sL iL sH iH : ℤ)
    (vals0 : Array Std.I64 256#usize) (hl : l < 32) :
    q4kLaneAt codes g (l + 1) sL iL sH iH vals0 (64 * g + l)
      = q4kLaneVal codes g l sL iL sH iH := by
  unfold q4kLaneAt
  rw [if_neg (by omega : ¬ (64 * g + l < 64 * g)),
    if_pos ⟨by omega, Or.inl (by omega)⟩]
  rw [show (64 * g + l) % 64 = l by omega]

lemma q4kLaneAt_high (codes : Array Std.U8 256#usize) (g l : Nat) (sL iL sH iH : ℤ)
    (vals0 : Array Std.I64 256#usize) (hl : l < 32) :
    q4kLaneAt codes g (l + 1) sL iL sH iH vals0 (64 * g + 32 + l)
      = q4kLaneVal codes g (32 + l) sL iL sH iH := by
  unfold q4kLaneAt
  rw [if_neg (by omega : ¬ (64 * g + 32 + l < 64 * g)),
    if_pos ⟨by omega, Or.inr ⟨by omega, by omega⟩⟩]
  rw [show (64 * g + 32 + l) % 64 = 32 + l by omega]

/-! ### Checked `i64` bounds

The Q4_K factors are bounded: a finite `f16` scale is `< 2^48` in Q32.32, a 6-bit scale/min code
is `≤ 63`, and a 4-bit code is `≤ 15`; every product and difference stays inside `i64`. -/

lemma abs_u8_code_le {c : Std.U8} (h : c.val ≤ 15) : |(c.val : ℤ)| ≤ 15 := by
  rw [abs_of_nonneg (by positivity)]
  exact_mod_cast h

theorem i64_mul_fits {x y : ℤ} (hx : |x| ≤ 2 ^ 54) (hy : |y| ≤ 15) :
    I64.min ≤ x * y ∧ x * y ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 54 * 15 < 2 ^ 63 := by norm_num
  have hxy : |x * y| < 2 ^ 63 := by
    rw [abs_mul]
    have h : |x| * |y| ≤ (2 : ℤ) ^ 54 * 15 :=
      mul_le_mul hx hy (abs_nonneg y) (by positivity)
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  omega

theorem i64_sub_fits {x y : ℤ} (hx : |x| ≤ 2 ^ 54 * 15) (hy : |y| ≤ 2 ^ 54) :
    I64.min ≤ x - y ∧ x - y ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 54 * 15 + 2 ^ 54 < 2 ^ 63 := by norm_num
  have hxy : |x - y| < 2 ^ 63 := by
    have h1 : |x - y| ≤ |x| + |y| := by simpa [sub_eq_add_neg] using abs_add_le x (-y)
    have h : |x| + |y| ≤ (2 : ℤ) ^ 54 * 15 + 2 ^ 54 := add_le_add hx hy
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  omega

theorem dequant_mul_ok (a b : Std.I64)
    (hlo : I64.min ≤ a.val * b.val) (hhi : a.val * b.val ≤ I64.max) :
    ∃ z, dequant.mul a b = ok (.Ok z) ∧ z.val = a.val * b.val := by
  obtain ⟨z, hz, hzv⟩ := I64_checked_mul_some a b hlo hhi
  refine ⟨z, ?_, hzv⟩
  unfold dequant.mul
  rw [hz]
  simp only [lift, bind_tc_ok]
  rw [core.option.Option.ok_or_some]

theorem dequant_sub_ok (a b : Std.I64)
    (hlo : I64.min ≤ a.val - b.val) (hhi : a.val - b.val ≤ I64.max) :
    ∃ z, dequant.sub a b = ok (.Ok z) ∧ z.val = a.val - b.val := by
  obtain ⟨z, hz, hzv⟩ := I64_checked_sub_some a b hlo hhi
  refine ⟨z, ?_, hzv⟩
  unfold dequant.sub
  rw [hz]
  simp only [lift, bind_tc_ok]
  rw [core.option.Option.ok_or_some]

/-- One lane-loop body step on the success path: the two checked products and differences succeed
and the two writes advance the invariant by one lane. -/
theorem q4k_lane_body_step
    (codes : Array Std.U8 256#usize) (group : Std.Usize)
    (slope_lo intercept_lo slope_hi intercept_hi : Std.I64)
    (vals0 vals : Array Std.I64 256#usize) (lane : Std.Usize)
    (ok1 : Bool) (err : Option error.EngineError)
    (hgroup : group.val < 4)
    (hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 15)
    (hsL : |slope_lo.val| ≤ 2 ^ 54) (hiL : |intercept_lo.val| ≤ 2 ^ 54)
    (hsH : |slope_hi.val| ≤ 2 ^ 54) (hiH : |intercept_hi.val| ≤ 2 ^ 54)
    (hlt : lane.val < 32) (hok : ok1 = true) (herr : err = none)
    (hinv : q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val
      intercept_hi.val vals0 vals lane.val ok1 err) :
    ∃ vals2 lane2,
      dequant.dequant_q4_k_row_loop0_loop0_loop0.body codes group slope_lo intercept_lo
        slope_hi intercept_hi ok1 err vals lane =
        ok (ControlFlow.cont (true, none, vals2, lane2)) ∧
      q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
        vals0 vals2 lane2.val true none ∧
      lane2.val = lane.val + 1 := by
  obtain ⟨-, -, hlane, hvals⟩ := hinv
  have hclen : codes.val.length = 256 := Array.property codes
  have hvlen : vals.val.length = 256 := Array.property vals
  obtain ⟨i, hi, hiv⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 64#usize) (y := group) (by scalar_tac))
  have hiv' : i.val = 64 * group.val := by rw [hiv]; simp
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := lane) (by scalar_tac))
  have hi1v' : i1.val = i.val + lane.val := hi1v
  have hi1abs : i1.val = 64 * group.val + lane.val := by rw [hi1v', hiv']
  have hi1lt : i1.val < 256 := by rw [hi1abs]; omega
  have hc1lt : i1.val < codes.val.length := by rw [hclen]; exact hi1lt
  obtain ⟨c1, hc1, hc1v⟩ :=
    WP.spec_imp_exists (Array.index_usize_spec codes i1 (by show i1.val < codes.val.length; exact hc1lt))
  have hc1v' : c1.val = (codes.val.getD i1.val 0#u8).val := by
    rw [hc1v]
    exact congrArg (fun u : Std.U8 => u.val) (list_getD_getElem codes.val i1.val hc1lt 0#u8)
  have hc1le : c1.val ≤ 15 := by rw [hc1v']; exact hcode i1.val hi1lt
  let i3 : Std.I64 := UScalar.hcast .I64 c1
  have hi3 : I64.Insts.CoreConvertFromU8.from c1 = ok i3 := rfl
  have hi3v : i3.val = (c1.val : ℤ) := by
    show (UScalar.hcast .I64 c1).val = (c1.val : ℤ)
    rw [UScalar.hcast_val_eq, IScalarTy.I64_numBits_eq]
    apply Int.bmod_eq_of_le
    · norm_num
    · have h256 : c1.val < 256 := U8.lt_succ_max c1
      norm_num
      omega
  have hi3abs : |i3.val| ≤ 15 := by rw [hi3v]; exact abs_u8_code_le hc1le
  have hpb := i64_mul_fits (x := slope_lo.val) (y := i3.val) hsL hi3abs
  obtain ⟨prod, hprod, hprodv⟩ := dequant_mul_ok slope_lo i3 hpb.1 hpb.2
  have hprodabs : |prod.val| ≤ (2 : ℤ) ^ 54 * 15 := by
    rw [hprodv, abs_mul]
    exact mul_le_mul hsL hi3abs (abs_nonneg _) (by positivity)
  have hsb := i64_sub_fits (x := prod.val) (y := intercept_lo.val) hprodabs hiL
  obtain ⟨v, hsub, hsubv⟩ := dequant_sub_ok prod intercept_lo hsb.1 hsb.2
  have hi1ltv : i1.val < vals.val.length := by rw [hvlen]; exact hi1lt
  obtain ⟨a, ha, hav⟩ := WP.spec_imp_exists (Array.update_spec vals i1 v hi1ltv)
  have hav' : a.val = vals.val.set i1.val v := by rw [hav, Array.set_val_eq]
  obtain ⟨ib, hib, hibv⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 32#usize) (by scalar_tac))
  have hibv' : ib.val = i.val + 32 := by rw [hibv]; simp
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists (Usize.add_spec (x := ib) (y := lane) (by scalar_tac))
  have hi5v' : i5.val = ib.val + lane.val := hi5v
  have hi5abs : i5.val = 64 * group.val + 32 + lane.val := by rw [hi5v', hibv', hiv']
  have hi5lt : i5.val < 256 := by rw [hi5abs]; omega
  have hc2lt : i5.val < codes.val.length := by rw [hclen]; exact hi5lt
  obtain ⟨c2, hc2, hc2v⟩ :=
    WP.spec_imp_exists (Array.index_usize_spec codes i5 (by show i5.val < codes.val.length; exact hc2lt))
  have hc2v' : c2.val = (codes.val.getD i5.val 0#u8).val := by
    rw [hc2v]
    exact congrArg (fun u : Std.U8 => u.val) (list_getD_getElem codes.val i5.val hc2lt 0#u8)
  have hc2le : c2.val ≤ 15 := by rw [hc2v']; exact hcode i5.val hi5lt
  let i7 : Std.I64 := UScalar.hcast .I64 c2
  have hi7 : I64.Insts.CoreConvertFromU8.from c2 = ok i7 := rfl
  have hi7v : i7.val = (c2.val : ℤ) := by
    show (UScalar.hcast .I64 c2).val = (c2.val : ℤ)
    rw [UScalar.hcast_val_eq, IScalarTy.I64_numBits_eq]
    apply Int.bmod_eq_of_le
    · norm_num
    · have h256 : c2.val < 256 := U8.lt_succ_max c2
      norm_num
      omega
  have hi7abs : |i7.val| ≤ 15 := by rw [hi7v]; exact abs_u8_code_le hc2le
  have hpb2 := i64_mul_fits (x := slope_hi.val) (y := i7.val) hsH hi7abs
  obtain ⟨prod2, hprod2, hprod2v⟩ := dequant_mul_ok slope_hi i7 hpb2.1 hpb2.2
  have hprod2abs : |prod2.val| ≤ (2 : ℤ) ^ 54 * 15 := by
    rw [hprod2v, abs_mul]
    exact mul_le_mul hsH hi7abs (abs_nonneg _) (by positivity)
  have hsb2 := i64_sub_fits (x := prod2.val) (y := intercept_hi.val) hprod2abs hiH
  obtain ⟨v2, hsub2, hsub2v⟩ := dequant_sub_ok prod2 intercept_hi hsb2.1 hsb2.2
  have hia : a.val.length = 256 := by rw [hav', List.length_set]; exact hvlen
  have hi5lta : i5.val < a.val.length := by rw [hia]; exact hi5lt
  obtain ⟨a2, ha2, ha2v⟩ := WP.spec_imp_exists (Array.update_spec a i5 v2 hi5lta)
  have ha2v' : a2.val = a.val.set i5.val v2 := by rw [ha2v, Array.set_val_eq]
  obtain ⟨lane2, hlane2, hlane2v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2v' : lane2.val = lane.val + 1 := hlane2v
  refine ⟨a2, lane2, ?_, ?_, hlane2v'⟩
  · unfold dequant.dequant_q4_k_row_loop0_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
    simp only [hok, herr, hi, hi1, hc1, hi3, hprod, hsub, ha, hib, hi5, hc2, hi7, hprod2,
      hsub2, hlane2, bind_tc_ok]
    rw [if_pos trivial]
    change (do
        let (ok3, err2, vals2) ← (do let b ← Array.update a i5 v2; ok (true, none, b))
        ok (ControlFlow.cont (ok3, err2, vals2, lane2)))
      = ok (ControlFlow.cont (true, none, a2, lane2))
    rw [ha2]
    simp only [bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [hlane2v']
    have hlen_a : a.val.length = 256 := hia
    have hlen_a2 : a2.val.length = 256 := by rw [ha2v', List.length_set]; exact hlen_a
    rw [ha2v', getD_set]
    by_cases h5 : j = i5.val
    · subst h5
      rw [if_pos ⟨rfl, by rw [hlen_a]; exact hi5lt⟩]
      rw [hsub2v, hprod2v, hi7v, hi5abs]
      rw [q4kLaneAt_high codes group.val lane.val slope_lo.val intercept_lo.val slope_hi.val
        intercept_hi.val vals0 hlt]
      rw [q4kLaneVal_high codes group.val (32 + lane.val) slope_lo.val intercept_lo.val
        slope_hi.val intercept_hi.val (by omega)]
      have hc2val : c2.val = (codes.val.getD (64 * group.val + (32 + lane.val)) 0#u8).val := by
        rw [hc2v', hi5abs, Nat.add_assoc]
      rw [hc2val]
    · rw [if_neg (fun h => h5 h.1)]
      rw [hav', getD_set]
      by_cases h4 : j = i1.val
      · subst h4
        rw [if_pos ⟨rfl, by rw [hvlen]; exact hi1lt⟩]
        rw [hsubv, hprodv, hi3v, hi1abs]
        rw [q4kLaneAt_low codes group.val lane.val slope_lo.val intercept_lo.val slope_hi.val
          intercept_hi.val vals0 hlt]
        rw [q4kLaneVal_low codes group.val lane.val slope_lo.val intercept_lo.val slope_hi.val
          intercept_hi.val hlt]
        have hc1val : c1.val = (codes.val.getD (64 * group.val + lane.val) 0#u8).val := by
          rw [hc1v', hi1abs]
        rw [hc1val]
      · rw [if_neg (fun h => h4 h.1)]
        rw [hvals j hj]
        refine (q4kLaneAt_step codes group.val lane.val slope_lo.val intercept_lo.val
          slope_hi.val intercept_hi.val vals0 j ?_ ?_).symm
        · intro h; exact h4 (by rw [hi1abs]; exact h)
        · intro h; exact h5 (by rw [hi5abs]; exact h)

/-- The lane loop: `loop.spec_decr_nat` with `q4kLaneInv` and measure `32 − lane`.

Complexity: 32 iterations, each O(1) (two `u8` reads, two checked products, two checked
differences, two array writes); O(1) space. -/
theorem q4k_lane_loop_spec
    (codes : Array Std.U8 256#usize) (group : Std.Usize)
    (slope_lo intercept_lo slope_hi intercept_hi : Std.I64)
    (vals0 vals : Array Std.I64 256#usize) (lane : Std.Usize)
    (hgroup : group.val < 4)
    (hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 15)
    (hsL : |slope_lo.val| ≤ 2 ^ 54) (hiL : |intercept_lo.val| ≤ 2 ^ 54)
    (hsH : |slope_hi.val| ≤ 2 ^ 54) (hiH : |intercept_hi.val| ≤ 2 ^ 54)
    (hinv : q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val
      intercept_hi.val vals0 vals lane.val true none) :
    ∃ vals2,
      dequant.dequant_q4_k_row_loop0_loop0_loop0 true none codes vals group slope_lo
        intercept_lo slope_hi intercept_hi lane = ok (true, none, vals2) ∧
      q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
        vals0 vals2 32 true none := by
  have hspec : dequant.dequant_q4_k_row_loop0_loop0_loop0 true none codes vals group slope_lo
      intercept_lo slope_hi intercept_hi lane ⦃ r =>
        q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val intercept_hi.val
          vals0 r.2.2 32 r.1 r.2.1 ⦄ := by
    unfold dequant.dequant_q4_k_row_loop0_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Bool × Option error.EngineError × Array Std.I64 256#usize × Std.Usize =>
        32 - s.2.2.2.val)
      (inv := fun s => q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val
        intercept_hi.val vals0 s.2.2.1 s.2.2.2.val s.1 s.2.1)
      (post := fun (r : Bool × Option error.EngineError × Array Std.I64 256#usize) =>
        q4kLaneInv codes group.val slope_lo.val intercept_lo.val slope_hi.val
          intercept_hi.val vals0 r.2.2 32 r.1 r.2.1)
    · rintro ⟨ok1, err, valsX, laneX⟩ hinv'
      dsimp only
      by_cases hlt : laneX.val < 32
      · have hstep := q4k_lane_body_step codes group slope_lo intercept_lo slope_hi intercept_hi
          vals0 valsX laneX ok1 err hgroup hcode hsL hiL hsH hiH hlt hinv'.1 hinv'.2.1 hinv'
        obtain ⟨vals2, lane2, hbody, hinv2, hlane2⟩ := hstep
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        exact ⟨hinv2, by rw [hlane2]; omega⟩
      · have hbody : dequant.dequant_q4_k_row_loop0_loop0_loop0.body codes group slope_lo
            intercept_lo slope_hi intercept_hi ok1 err valsX laneX
            = ok (ControlFlow.done (ok1, err, valsX)) := by
          unfold dequant.dequant_q4_k_row_loop0_loop0_loop0.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hl32 : laneX.val = 32 := by
          have hle : laneX.val ≤ 32 := hinv'.2.2.1
          omega
        simpa [hl32] using hinv'
    · exact hinv
  obtain ⟨r, hr, hinv2⟩ := WP.spec_imp_exists hspec
  refine ⟨r.2.2, ?_, ?_⟩
  · rw [hr]
    congr 1
    exact Prod.ext hinv2.1 (Prod.ext hinv2.2.1 rfl)
  · rw [hinv2.1, hinv2.2.1] at hinv2
    exact hinv2

/-! ## Q4_K row references (step b/c)

`q4kBlockVal` is the exact value of within-block index `i`; `q4kBlockVals`/`q4kRowValues` lift it to
a block and a whole row of `n` weights. `q4kWeight` is the row reference by global index. -/

/-- A Q4_K 4-bit code is below sixteen. -/
lemma q4CodesRef_lt_16 (blk : List Std.U8) (i : Nat) : q4CodesRef blk i < 16 := by
  unfold q4CodesRef
  dsimp only
  split_ifs with h
  · exact Nat.mod_lt _ (by norm_num)
  · rw [Nat.div_lt_iff_lt_mul (by norm_num)]
    have hb : (blk.getD (16 + 32 * (i / 64) + i % 64 % 32) 0#u8).val < 256 :=
      U8.lt_succ_max _
    omega

/-- The exact `ℤ` value of within-block index `i`, from the decoded scales `d`, `dmin` and the
block's `sc` (12-byte table) and `codes` (256 nibbles) byte lists. -/
def q4kBlockVal (d dmin : ℤ) (sc codes : List Std.U8) (i : Nat) : ℤ :=
  (d * ((if i % 64 < 32 then q4kScaleMinRef sc (2 * (i / 64))
          else q4kScaleMinRef sc (2 * (i / 64) + 1)).1 : ℤ))
      * ((codes.getD i 0#u8).val : ℤ)
    - (dmin * ((if i % 64 < 32 then q4kScaleMinRef sc (2 * (i / 64))
          else q4kScaleMinRef sc (2 * (i / 64) + 1)).2 : ℤ))

/-- The block's 12 scale/min bytes, read from the row at byte `144·b + 4`. -/
def q4kScList (raw : Slice Std.U8) (b : Nat) : List Std.U8 :=
  (raw.val.drop (144 * b + 4)).take 12

/-- The block's 144 raw bytes, read from the row at byte `144·b` (`d`, `dmin`, the 12-byte scale
table and the 128 `qs` bytes). -/
def q4kBlockBytes (raw : Slice Std.U8) (b : Nat) : List Std.U8 :=
  (raw.val.drop (144 * b)).take 144

/-- The 256 decoded 4-bit codes of block `b`, entry `i` being `q4CodesRef` of the block bytes. -/
def q4kCodesList (raw : Slice Std.U8) (b : Nat) : List Std.U8 :=
  (List.range 256).map (fun i =>
    U8.ofNatCore (q4CodesRef (q4kBlockBytes raw b) i) (by
      have h := q4CodesRef_lt_16 (q4kBlockBytes raw b) i
      exact Nat.lt_of_lt_of_le h (by decide)))

/-- The 256 exact weights of block `b`. -/
def q4kBlockVals (raw : Slice Std.U8) (b : Nat) : List ℤ :=
  (List.range 256).map (fun i =>
    q4kBlockVal (f16Value (f16LeHalf raw (144 * b)))
      (f16Value (f16LeHalf raw (144 * b + 2))) (q4kScList raw b) (q4kCodesList raw b) i)

/-- The exact weight of row index `j`, block `j / 256`, within-block index `j % 256`. -/
def q4kWeight (raw : Slice Std.U8) (j : Nat) : ℤ :=
  (q4kBlockVals raw (j / 256)).getD (j % 256) 0

/-- The first `n` exact Q32.32 integers of a Q4_K row. -/
def q4kRowValues (raw : Slice Std.U8) (n : Nat) : List ℤ :=
  (List.range n).map (q4kWeight raw)

lemma q4kScaleMinRef_fst_lt (sc : List Std.U8) (j : Nat) : (q4kScaleMinRef sc j).1 < 64 := by
  unfold q4kScaleMinRef
  split
  · exact Nat.lt_of_le_of_lt Nat.and_le_right (by norm_num)
  · rename_i h
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num]
    apply Nat.or_lt_two_pow
    · exact Nat.lt_of_le_of_lt Nat.and_le_right (by norm_num)
    · have hb : (sc.getD (j - 4) 0#u8).val < 256 := U8.lt_succ_max _
      have h2 : ((sc.getD (j - 4) 0#u8).val >>> 6 <<< 4) < 64 := by
        rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
        omega
      omega

lemma q4kScaleMinRef_snd_lt (sc : List Std.U8) (j : Nat) : (q4kScaleMinRef sc j).2 < 64 := by
  unfold q4kScaleMinRef
  split
  · exact Nat.lt_of_le_of_lt Nat.and_le_right (by norm_num)
  · rename_i h
    rw [show (64 : ℕ) = 2 ^ 6 by norm_num]
    apply Nat.or_lt_two_pow
    · have hb : (sc.getD (j + 4) 0#u8).val < 256 := U8.lt_succ_max _
      rw [Nat.shiftRight_eq_div_pow]
      omega
    · have hb : (sc.getD j 0#u8).val < 256 := U8.lt_succ_max _
      have h2 : ((sc.getD j 0#u8).val >>> 6 <<< 4) < 64 := by
        rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow]
        omega
      omega

lemma f16Mag_le (h : U16) (he : f16Exp h ≠ 31) : f16Mag h ≤ 2047 * 2 ^ 37 := by
  unfold f16Mag
  have hm : f16Man h ≤ 1023 := by have := f16_man_lt h; omega
  split
  · have h1 : (f16Man h : ℤ) * 2 ^ 8 ≤ 1023 * 2 ^ 8 :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hm) (by positivity)
    have h2 : (1023 : ℤ) * 2 ^ 8 ≤ 2047 * 2 ^ 37 := by norm_num
    linarith
  · have he' : f16Exp h + 7 ≤ 37 := by have := f16_exp_lt h; omega
    have h1 : ((1024 + f16Man h : ℕ) : ℤ) ≤ 2047 := by push_cast; omega
    have h2 : (2 : ℤ) ^ (f16Exp h + 7) ≤ 2 ^ 37 := pow_le_pow_right₀ (by norm_num) he'
    have h3 : ((1024 + f16Man h : ℕ) : ℤ) * 2 ^ (f16Exp h + 7) ≤ 2047 * 2 ^ 37 :=
      mul_le_mul h1 h2 (by positivity) (by norm_num)
    linarith

lemma f16Mag_nonneg (h : U16) : 0 ≤ f16Mag h := by
  unfold f16Mag
  split <;> positivity

lemma f16Value_abs_lt (h : U16) (he : f16Exp h ≠ 31) : |f16Value h| < 2 ^ 48 := by
  have hm := f16Mag_le h he
  have h48 : (2047 : ℤ) * 2 ^ 37 < 2 ^ 48 := by norm_num
  have hnn := f16Mag_nonneg h
  unfold f16Value
  split
  · rw [abs_neg, abs_of_nonneg hnn]; linarith
  · rw [abs_of_nonneg hnn]; linarith


end apeiron_word
