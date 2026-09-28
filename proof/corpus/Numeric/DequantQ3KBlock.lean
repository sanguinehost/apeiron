/-
Split of `Numeric.DequantQ3K`.

This module holds the reference definitions and the extracted block readers for the `Q3_K`
dequantizer: the block length and reader scaffolding (`quant_Q3_K_BLOCK_BYTES_eq`,
`quant_q3_k_d_ok`), the signed sub-block scale reader `quant::q3_k_scales` with its
`q3kAuxVal`/`q3kWord`/`q3kScaleRef` reference and the aux-loop and scales-loop specifications, the signed code
reader `quant::q3_k_codes` with its `q3kCodeRef` reference and four nested loop specifications, and
the block law `q3_k_block_ok`.

The Rust bodies are `dequant::q3_k_block` and the block accessors `quant::q3_k_d`,
`quant::q3_k_scales` and `quant::q3_k_codes` in `src/format/quant.rs`; the decode is exact integer
arithmetic over Q32.32. The row law over the 256-entry value array, the raw-level weight references
and the one-block joint domain witness remain in `Numeric.DequantQ3K`, which imports this module.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.DequantQ4K
import Numeric.DequantQ6K
import Numeric.DequantKWitness

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Block length and reader scaffolding -/

/-- The Q3_K block length evaluates to 110 bytes. -/
theorem quant_Q3_K_BLOCK_BYTES_eq : quant.Q3_K_BLOCK_BYTES = ok (110#usize) := by
  have h : quant.Q3_K_BLOCK_BYTES ⦃ z => z = (110#usize) ⦄ := by
    unfold quant.Q3_K_BLOCK_BYTES
    simp only [quant.QK_K]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- `quant::q3_k_d` succeeds on a whole block and reads the `f16` bits at byte 108. -/
theorem quant_q3_k_d_ok (blk : Slice U8) (hlen : 110 ≤ blk.val.length) :
    quant.q3_k_d blk = ok (.Ok (f16LeHalf blk 108)) := by
  have hneed := quant_need_ok blk 110#usize (by simpa using hlen)
  have hu := quant_u16le_ok blk 108#usize (by norm_num; omega)
  rw [quant.q3_k_d, quant_Q3_K_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, hneed]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, quant.QK_K, bind_tc_ok]
  obtain ⟨i1, h1, h1v⟩ := UScalar.div_spec (256#usize) (y := 8#usize) (by decide)
  obtain ⟨i2, h2, h2v⟩ := UScalar.div_spec (256#usize) (y := 4#usize) (by decide)
  obtain ⟨i3, h3, h3v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i1) (y := i2) (by rw [h1v, h2v]; scalar_tac))
  obtain ⟨i4, h4, h4v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i3) (y := 12#usize) (by rw [h3v]; scalar_tac))
  have h1v' : i1.val = 32 := by rw [h1v]; norm_num
  have h2v' : i2.val = 64 := by rw [h2v]; norm_num
  have h3v' : i3.val = 96 := by rw [h3v, h1v', h2v']
  have h4v' : i4.val = 108 := by rw [h4v, h3v']; norm_num
  have hi4 : i4 = 108#usize := UScalar.eq_of_val_eq (by rw [h4v']; norm_num)
  simp only [h1, h2, h3, h4, hi4, hu, bind_tc_ok]
  rfl

/-- `dequant_row_fixed` at `ggml_type = 11` (`GGML_TYPE_Q3_K`) is `dequant_q3_k_row`. -/
theorem dequant_row_fixed_q3_k (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 11#u32 raw n out = dequant.dequant_q3_k_row raw n out := rfl

/-! ## `quant::q3_k_scales`

The Rust reads the 12 packed bytes at block offset `96..108` as three little-endian `u32`, combines
them through `KMASK1 = 0x03030303` / `KMASK2 = 0x0f0f0f0f` into four words, and takes signed bytes
of each word. The reference below mirrors the same `&&&`/`|||`/`>>>`/`<<<` operations, so the loop
write values and the reference meet syntactically. -/

/-- The signed-`u32` default `getD` lemma. -/
lemma getD_of_lt_u32 {l : List U32} {i : Nat} (h : i < l.length) : l.getD i 0#u32 = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- The `i`-th little-endian `u32` of a 12-byte scale window (bytes `4·i .. 4·i+3`). -/
def q3kAuxVal (l : List U8) (i : Nat) : Nat :=
  (core.num.U32.from_le_bytes (Array.make 4#usize
    [l.getD (4 * i) 0#u8, l.getD (4 * i + 1) 0#u8,
     l.getD (4 * i + 2) 0#u8, l.getD (4 * i + 3) 0#u8])).val

/-- The four shuffled scale words as a function of the three auxiliary word values. -/
def q3kWordAux (a0 a1 a2 : Nat) : Nat → Nat
  | 0 => ((a0 &&& 0x0F0F0F0F) ||| (((a2 &&& 0x03030303) <<< 4) % 2 ^ 32))
  | 1 => ((a1 &&& 0x0F0F0F0F) ||| ((((a2 >>> 2) &&& 0x03030303) <<< 4) % 2 ^ 32))
  | 2 => (((a0 >>> 4) &&& 0x0F0F0F0F) ||| ((((a2 >>> 4) &&& 0x03030303) <<< 4) % 2 ^ 32))
  | 3 => (((a1 >>> 4) &&& 0x0F0F0F0F) ||| ((((a2 >>> 6) &&& 0x03030303) <<< 4) % 2 ^ 32))
  | _ => 0

/-- The four shuffled scale words of the 12-byte window. -/
def q3kWord (l : List U8) (i : Nat) : Nat :=
  q3kWordAux (q3kAuxVal l 0) (q3kAuxVal l 1) (q3kAuxVal l 2) i

/-- The signed 6-bit sub-block scale `k` of a 12-byte scale window. -/
def q3kScaleRef (l : List U8) (k : Nat) : ℤ :=
  Int.bmod ((q3kWord l (k / 4)) >>> (8 * (k % 4))) 256

/-- The signed byte taken from the shuffled word array at entry `j`. -/
def q3kScaleWordRef (words : Array U32 4#usize) (j : Nat) : ℤ :=
  Int.bmod ((words.val.getD (j / 4) 0#u32).val >>> (8 * (j % 4))) 256

/-- With the four word values equal to `q3kWord`, the array byte reference is `q3kScaleRef`. -/
lemma q3kScaleWordRef_eq (l : List U8) (words : Array U32 4#usize)
    (h : ∀ i, i < 4 → (words.val.getD i 0#u32).val = q3kWord l i) (j : Nat) (hj : j < 16) :
    q3kScaleWordRef words j = q3kScaleRef l j := by
  unfold q3kScaleWordRef q3kScaleRef
  rw [h (j / 4) (by omega)]

lemma KMASK2_val : (quant.q3_k_scales.KMASK2).val = 0x0F0F0F0F := by
  rw [quant.q3_k_scales.KMASK2]; decide
lemma KMASK1_val : (quant.q3_k_scales.KMASK1).val = 0x03030303 := by
  rw [quant.q3_k_scales.KMASK1]; decide

/-- The four-byte window ``Array.make`` is congruent on equal byte values. -/
lemma array_make4_eq {a b c d a' b' c' d' : U8}
    (ha : a.val = a'.val) (hb : b.val = b'.val) (hc : c.val = c'.val) (hd : d.val = d'.val) :
    Array.make 4#usize [a, b, c, d] = Array.make 4#usize [a', b', c', d'] := by
  apply Aeneas.Std.Array.ext
  simp only [Array.make_val]
  rw [UScalar.eq_of_val_eq ha, UScalar.eq_of_val_eq hb, UScalar.eq_of_val_eq hc,
    UScalar.eq_of_val_eq hd]

lemma arr_getD_val_u32 (a : Array U32 4#usize) (i : Usize) (h : i.val < a.val.length) :
    (a.val[i.val]'h).val = (a.val.getD i.val 0#u32).val := by
  rw [list_getD_getElem a.val i.val h 0#u32]

lemma arr_getD_val_i8 (a : Array I8 16#usize) (i : Usize) (h : i.val < a.val.length) :
    (a.val[i.val]'h).val = (a.val.getD i.val 0#i8).val := by
  rw [list_getD_getElem a.val i.val h 0#i8]

lemma arr_getD_val_u32_nat (a : Array U32 4#usize) (i : Nat) (h : i < a.val.length) :
    (a.val[i]'h).val = (a.val.getD i 0#u32).val := by
  rw [list_getD_getElem a.val i h 0#u32]

/-- Aux-loop invariant: after `k` word reads the first `k` entries hold `q3kAuxVal`, the rest `0`. -/
def q3kAuxInv (l : List U8) (aux : Array U32 4#usize) (k : Nat) : Prop :=
  k ≤ 3 ∧ ∀ i, i < 4 → (aux.val.getD i 0#u32).val = if i < k then q3kAuxVal l i else 0

/-- One aux-loop body step: read four bytes at `4·k`, pack little-endian, write entry `k`. -/
theorem q3kAux_body_step (sc : Slice U8) (aux : Array U32 4#usize) (k : Usize)
    (hlen : 12 ≤ sc.val.length) (hk : k.val < 3) (hinv : q3kAuxInv sc.val aux k.val) :
    ∃ (aux2 : Array U32 4#usize) (k2 : Usize),
      quant.q3_k_scales_loop0.body sc aux k = ok (ControlFlow.cont (aux2, k2)) ∧
      q3kAuxInv sc.val aux2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨o, ho, hov⟩ := WP.spec_imp_exists (Usize.mul_spec (x := 4#usize) (y := k) (by scalar_tac))
  have hov' : o.val = 4 * k.val := by rw [hov]; simp
  obtain ⟨i, hi, hiv⟩ := slice_index_getD sc o (by rw [hov']; omega)
  obtain ⟨o1, ho1, ho1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (by rw [hov']; scalar_tac))
  have ho1' : o1.val = 4 * k.val + 1 := by rw [ho1v, hov']; simp
  obtain ⟨i2, hi2, hi2v⟩ := slice_index_getD sc o1 (by rw [ho1']; omega)
  obtain ⟨o2, ho2, ho2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 2#usize) (by rw [hov']; scalar_tac))
  have ho2' : o2.val = 4 * k.val + 2 := by rw [ho2v, hov']; simp
  obtain ⟨i4, hi4, hi4v⟩ := slice_index_getD sc o2 (by rw [ho2']; omega)
  obtain ⟨o3, ho3, ho3v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 3#usize) (by rw [hov']; scalar_tac))
  have ho3' : o3.val = 4 * k.val + 3 := by rw [ho3v, hov']; simp
  obtain ⟨i6, hi6, hi6v⟩ := slice_index_getD sc o3 (by rw [ho3']; omega)
  have hmake : Array.make 4#usize [i, i2, i4, i6]
      = Array.make 4#usize [sc.val.getD (4 * k.val) 0#u8, sc.val.getD (4 * k.val + 1) 0#u8,
          sc.val.getD (4 * k.val + 2) 0#u8, sc.val.getD (4 * k.val + 3) 0#u8] :=
    array_make4_eq (by rw [hiv, hov']) (by rw [hi2v, ho1']) (by rw [hi4v, ho2'])
      (by rw [hi6v, ho3'])
  have hi7val : (core.num.U32.from_le_bytes (Array.make 4#usize [i, i2, i4, i6])).val
      = q3kAuxVal sc.val k.val := by
    unfold q3kAuxVal
    rw [← hmake]
  obtain ⟨a, ha, hav⟩ := WP.spec_imp_exists
    (Array.update_spec aux k (core.num.U32.from_le_bytes (Array.make 4#usize [i, i2, i4, i6]))
      (by show k.val < aux.val.length; rw [Array.property aux]; scalar_tac))
  obtain ⟨k2, hk2, hk2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk2' : k2.val = k.val + 1 := by rw [hk2v]; simp
  have hlen4 : aux.val.length = 4 := by simp [Array.property]
  refine ⟨a, k2, ?_, ?_, hk2'⟩
  · rw [quant.q3_k_scales_loop0.body, if_pos (by scalar_tac : k < 3#usize)]
    simp only [ho, hi, ho1, hi2, ho2, hi4, ho3, hi6, lift, ha, hk2, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro j hj
    have hset : a.val = aux.val.set k.val
        (core.num.U32.from_le_bytes (Array.make 4#usize [i, i2, i4, i6])) := by
      rw [hav, Array.set_val_eq]
    rw [hset, getD_set]
    by_cases hjk : j = k.val
    · rw [if_pos ⟨hjk, by rw [hlen4]; omega⟩, hjk, if_pos (by omega), hi7val]
    · rw [if_neg (fun h => hjk h.1), hinv.2 j hj]
      by_cases hlt : j < k.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega : ¬ j < k.val + 1)]

/-- The aux loop: `loop.spec_decr_nat` with `q3kAuxInv` and measure `3 - k`. -/
theorem q3kAux_loop_spec (sc : Slice U8) (aux0 : Array U32 4#usize)
    (hlen : 12 ≤ sc.val.length) (hinv0 : q3kAuxInv sc.val aux0 0) :
    WP.spec (quant.q3_k_scales_loop0 sc aux0 0#usize) (fun aux => q3kAuxInv sc.val aux 3) := by
  unfold quant.q3_k_scales_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array U32 4#usize × Usize => 3 - s.2.val)
    (inv := fun s => q3kAuxInv sc.val s.1 s.2.val)
  · rintro ⟨aux, k⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : k.val < 3
    · obtain ⟨aux2, k2, hbody, hinv2, hk2⟩ := q3kAux_body_step sc aux k hlen hlt hinv'
      simp only [hbody]
      refine ⟨by simpa [hk2] using hinv2, ?_⟩
      show 3 - k2.val < 3 - k.val
      omega
    · have hnot : ¬ (k < 3#usize) := fun hc => hlt ((UScalar.lt_equiv k 3#usize).1 hc)
      have hbody : quant.q3_k_scales_loop0.body sc aux k = ok (ControlFlow.done aux) := by
        rw [quant.q3_k_scales_loop0.body, if_neg hnot]
      simp only [hbody]
      have hk3 : k.val = 3 := by have := hinv'.1; omega
      simpa [hk3] using hinv'
  · exact hinv0

/-- Loop-1 invariant: after `j` writes the first `j` entries hold `q3kScaleWordRef`, the rest `0`. -/
def q3kScalesInv (words : Array U32 4#usize) (scales : Array I8 16#usize) (j : Nat) : Prop :=
  j ≤ 16 ∧ ∀ i, i < 16 → (scales.val.getD i 0#i8).val = if i < j then q3kScaleWordRef words i else 0

/-- One scales-loop body step: read word `j/4`, shift by `8·(j%4)`, signed-cast, write entry `j`. -/
theorem q3kScales_body_step (words : Array U32 4#usize) (scales : Array I8 16#usize) (j : Usize)
    (hj : j.val < 16) (hinv : q3kScalesInv words scales j.val) :
    ∃ (scales2 : Array I8 16#usize) (j2 : Usize),
      quant.q3_k_scales_loop1.body words scales j = ok (ControlFlow.cont (scales2, j2)) ∧
      q3kScalesInv words scales2 (j.val + 1) ∧ j2.val = j.val + 1 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec j (y := 4#usize) (by decide)
  have hqv' : q.val = j.val / 4 := by rw [hqv]; simp
  obtain ⟨w, hw, hwv⟩ := WP.spec_imp_exists
    (Array.index_usize_spec words q (by show q.val < words.val.length; rw [Array.property words, hqv']; scalar_tac))
  have hwval : w.val = (words.val.getD (j.val / 4) 0#u32).val := by
    rw [hwv]
    exact (arr_getD_val_u32 words q (by rw [Array.property words, hqv']; scalar_tac)).trans
      (congrArg (fun t => (words.val.getD t 0#u32).val) hqv')
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists (UScalar.rem_spec j (y := 4#usize) (by decide))
  have hrv' : r.val = j.val % 4 := by rw [hrv]; simp
  obtain ⟨s, hs, hsv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 8#usize) (y := r) (by rw [hrv']; scalar_tac))
  have hsv' : s.val = 8 * (j.val % 4) := by
    have h8 : (8#usize).val = 8 := by decide
    rw [hsv, hrv', h8]
  obtain ⟨i4, hshift, hi4v, -⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_spec w s (by
      rw [hsv', UScalarTy.U32_numBits_eq]
      have := Nat.mod_lt j.val (by norm_num : 0 < 4)
      omega))
  have hi4v' : i4.val = w.val >>> s.val := hi4v
  have hi6val : (UScalar.hcast .I8 (UScalar.cast .U8 i4)).val = q3kScaleWordRef words j.val := by
    unfold q3kScaleWordRef
    rw [UScalar.hcast_val_eq, UScalar.cast_val_eq, hi4v', hsv', hwval]
    simp only [UScalarTy.U8_numBits_eq, IScalarTy.I8_numBits_eq]
    rw [show (2:ℕ)^8 = 256 from by norm_num, Int.natCast_emod, Int.emod_bmod]
  obtain ⟨a, ha, hav⟩ := WP.spec_imp_exists
    (Array.update_spec scales j (UScalar.hcast .I8 (UScalar.cast .U8 i4))
      (by show j.val < scales.val.length; rw [Array.property scales]; scalar_tac))
  obtain ⟨j2, hj2, hj2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
  have hj2' : j2.val = j.val + 1 := by rw [hj2v]; simp
  have hlen16 : scales.val.length = 16 := by simp [Array.property]
  refine ⟨a, j2, ?_, ?_, hj2'⟩
  · rw [quant.q3_k_scales_loop1.body, if_pos (by scalar_tac : j < 16#usize)]
    simp only [hq, hw, hr, hs, hshift, lift, ha, hj2, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro i hi
    have hset : a.val = scales.val.set j.val (UScalar.hcast .I8 (UScalar.cast .U8 i4)) := by
      rw [hav, Array.set_val_eq]
    rw [hset, getD_set]
    by_cases hij : i = j.val
    · rw [if_pos ⟨hij, by rw [hlen16]; omega⟩, hij, if_pos (by omega), hi6val]
    · rw [if_neg (fun h => hij h.1), hinv.2 i hi]
      by_cases hlt : i < j.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega : ¬ i < j.val + 1)]

/-- The scales loop: `loop.spec_decr_nat` with `q3kScalesInv` and measure `16 - j`. -/
theorem q3kScales_loop_spec (words : Array U32 4#usize) (scales0 : Array I8 16#usize)
    (hinv0 : q3kScalesInv words scales0 0) :
    WP.spec (quant.q3_k_scales_loop1 words scales0 0#usize)
      (fun scales => q3kScalesInv words scales 16) := by
  unfold quant.q3_k_scales_loop1
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 16#usize × Usize => 16 - s.2.val)
    (inv := fun s => q3kScalesInv words s.1 s.2.val)
  · rintro ⟨scales, j⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : j.val < 16
    · obtain ⟨scales2, j2, hbody, hinv2, hj2⟩ := q3kScales_body_step words scales j hlt hinv'
      simp only [hbody]
      refine ⟨by simpa [hj2] using hinv2, ?_⟩
      show 16 - j2.val < 16 - j.val
      omega
    · have hnot : ¬ (j < 16#usize) := fun hc => hlt ((UScalar.lt_equiv j 16#usize).1 hc)
      have hbody : quant.q3_k_scales_loop1.body words scales j = ok (ControlFlow.done scales) := by
        rw [quant.q3_k_scales_loop1.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hj16 : j.val = 16 := by have := hinv'.1; omega
      simpa [hj16] using hinv'
  · exact hinv0

/-- The Q3_K scale reader: on a whole 110-byte block, `quant::q3_k_scales` succeeds and returns the
sixteen signed sub-block scales decoded from the packed bytes at offset 96. -/
theorem q3_k_scales_ok (blk : Slice U8) (hlen : 110 ≤ blk.val.length) :
    ∃ v : Array I8 16#usize,
      quant.q3_k_scales blk = ok (.Ok v) ∧
      ∀ k, k < 16 → (v.val.getD k 0#i8).val = q3kScaleRef ((blk.val.drop 96).take 12) k := by
  have hspec : quant.q3_k_scales blk ⦃ r => ∃ v : Array I8 16#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array I8 16#usize) quant.QuantError) ∧
      ∀ k, k < 16 → (v.val.getD k 0#i8).val
        = q3kScaleRef ((blk.val.drop 96).take 12) k ⦄ := by
    have hneed := quant_need_ok blk 110#usize (by simpa using hlen)
    unfold quant.q3_k_scales
    rw [quant_Q3_K_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, hneed]
    simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
    step*
    · rw [i5_post, i4_post, i1_post, i2_post]
      simp only [quant.QK_K]
      scalar_tac
    · have hsc12 : 12 ≤ sc_raw.val.length := by
        rw [show sc_raw.val.length = sc_raw.length from rfl, sc_raw_post1, i5_post, i4_post,
          i3_post, i1_post, i2_post]
        simp only [quant.QK_K]
        norm_num
      have hinv0 : q3kAuxInv sc_raw.val (Array.repeat 4#usize 0#u32) 0 := by
        refine ⟨by norm_num, ?_⟩
        intro i hi
        rw [Array.repeat_val]
        simp only [List.getD, List.getElem?_replicate]
        rw [if_pos (by simpa using hi), if_neg (by omega : ¬ i < 0)]
        rfl
      refine WP.spec_bind (q3kAux_loop_spec sc_raw (Array.repeat 4#usize 0#u32) hsc12 hinv0) ?_
      intro aux1 haux1
      step*
      have hscval : sc_raw.val = (blk.val.drop 96).take 12 := by
        rw [sc_raw_post, i3_post, i5_post, i4_post, i1_post, i2_post]
        simp only [quant.QK_K, List.slice]
        norm_num
      have hb0 : 0 < aux1.val.length := by rw [Array.property aux1]; norm_num
      have hb1 : 1 < aux1.val.length := by rw [Array.property aux1]; norm_num
      have hb2 : 2 < aux1.val.length := by rw [Array.property aux1]; norm_num
      have hav0 : i6.val = q3kAuxVal sc_raw.val 0 := by
        have h := haux1.2 0 (by norm_num)
        rw [if_pos (by norm_num : (0:Nat) < 3)] at h
        have hp : i6 = aux1.val[0]'hb0 := i6_post
        rw [hp, arr_getD_val_u32_nat aux1 0 hb0]
        exact h
      have hav1 : i11.val = q3kAuxVal sc_raw.val 1 := by
        have h := haux1.2 1 (by norm_num)
        rw [if_pos (by norm_num : (1:Nat) < 3)] at h
        have hp : i11 = aux1.val[1]'hb1 := i11_post
        rw [hp, arr_getD_val_u32_nat aux1 1 hb1]
        exact h
      have hav2 : tmp.val = q3kAuxVal sc_raw.val 2 := by
        have h := haux1.2 2 (by norm_num)
        rw [if_pos (by norm_num : (2:Nat) < 3)] at h
        have hp : tmp = aux1.val[2]'hb2 := tmp_post
        rw [hp, arr_getD_val_u32_nat aux1 2 hb2]
        exact h
      have hw0 : i10.val
          = q3kWordAux (q3kAuxVal sc_raw.val 0) (q3kAuxVal sc_raw.val 1) (q3kAuxVal sc_raw.val 2) 0 := by
        rw [i10_post, UScalar.val_or, i7_post, UScalar.val_and, i9_post, i8_post, UScalar.val_and,
          KMASK2_val, KMASK1_val, hav0, hav2]
        simp only [q3kWordAux, U32.size, U32.numBits, UScalarTy.U32_numBits_eq]
      have hw1 : i16.val
          = q3kWordAux (q3kAuxVal sc_raw.val 0) (q3kAuxVal sc_raw.val 1) (q3kAuxVal sc_raw.val 2) 1 := by
        rw [i16_post, UScalar.val_or, i12_post, UScalar.val_and, i15_post, i14_post, UScalar.val_and,
          i13_post, KMASK2_val, KMASK1_val, hav1, hav2]
        simp only [q3kWordAux, U32.size, U32.numBits, UScalarTy.U32_numBits_eq]
      have hw2 : i22.val
          = q3kWordAux (q3kAuxVal sc_raw.val 0) (q3kAuxVal sc_raw.val 1) (q3kAuxVal sc_raw.val 2) 2 := by
        rw [i22_post, UScalar.val_or, i18_post, UScalar.val_and, i21_post, i20_post, UScalar.val_and,
          i19_post, i17_post, KMASK2_val, KMASK1_val, hav0, hav2]
        simp only [q3kWordAux, U32.size, U32.numBits, UScalarTy.U32_numBits_eq]
      have hw3 : i28.val
          = q3kWordAux (q3kAuxVal sc_raw.val 0) (q3kAuxVal sc_raw.val 1) (q3kAuxVal sc_raw.val 2) 3 := by
        rw [i28_post, UScalar.val_or, i24_post, UScalar.val_and, i27_post, i26_post, UScalar.val_and,
          i25_post, i23_post, KMASK2_val, KMASK1_val, hav1, hav2]
        simp only [q3kWordAux, U32.size, U32.numBits, UScalarTy.U32_numBits_eq]
      have hwords : ∀ i, i < 4 →
          ((Array.make 4#usize [i10, i16, i22, i28]).val.getD i 0#u32).val = q3kWord sc_raw.val i := by
        intro i hi
        have h0 : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
        rcases h0 with rfl | rfl | rfl | rfl
        · simpa only [Array.make_val, List.getD_cons_zero, List.getD_cons_succ, q3kWord] using hw0
        · simpa only [Array.make_val, List.getD_cons_zero, List.getD_cons_succ, q3kWord] using hw1
        · simpa only [Array.make_val, List.getD_cons_zero, List.getD_cons_succ, q3kWord] using hw2
        · simpa only [Array.make_val, List.getD_cons_zero, List.getD_cons_succ, q3kWord] using hw3
      have hinvRep : q3kScalesInv (Array.make 4#usize [i10, i16, i22, i28])
          (Array.repeat 16#usize 0#i8) 0 := by
        refine ⟨by norm_num, ?_⟩
        intro i hi
        rw [Array.repeat_val]
        simp only [List.getD, List.getElem?_replicate]
        rw [if_pos (by simpa using hi), if_neg (by omega : ¬ i < 0)]
        rfl
      refine WP.spec_bind (q3kScales_loop_spec (Array.make 4#usize [i10, i16, i22, i28])
        (Array.repeat 16#usize 0#i8) hinvRep) ?_
      intro scales1 hsc1
      refine ⟨scales1, rfl, ?_⟩
      intro k hk
      rw [hsc1.2 k hk, if_pos hk,
        q3kScaleWordRef_eq sc_raw.val (Array.make 4#usize [i10, i16, i22, i28]) hwords k hk, hscval]
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  exact ⟨v, by rw [hr, hrv], hpt⟩

/-- The signed 6-bit reference code `i` of a Q3_K block: `(q - 4·[hmask bit clear])` with the
high plane `hm` at bytes `0..32` and the low plane `qs` at bytes `32..96`. -/
def q3kCodeRef (hm qs : List U8) (i : Nat) : ℤ :=
  let r := i % 128 / 32
  let q := ((qs.getD (i / 128 * 32 + i % 32) 0#u8).val >>> (2 * r)) &&& 3
  let hi := if (hm.getD (i % 32) 0#u8).val &&& 2 ^ (4 * (i / 128) + r) ≠ 0 then (0 : ℤ) else 4
  (q : ℤ) - hi

/-- The output prefix after `w` code writes: entries below `w` hold `q3kCodeRef`, the rest `0`. -/
def q3Prefix (hm qs : List U8) (out : Array I8 256#usize) (w : Nat) : Prop :=
  ∀ j, j < 256 → (out.val.getD j 0#i8).val = if j < w then q3kCodeRef hm qs j else 0

/-- Writing `q3kCodeRef` at position `y` extends the prefix by one. -/
lemma q3Prefix_push (hm qs : List U8) (out : Array I8 256#usize) (y : Usize) (v : I8)
    (hinv : q3Prefix hm qs out y.val)
    (hv : v.val = q3kCodeRef hm qs y.val) :
    q3Prefix hm qs (out.set y v) (y.val + 1) := by
  intro j hj
  have houtlen : out.val.length = 256 := by simp [Array.property]
  rw [Array.set_val_eq, getD_set]
  by_cases hjy : j = y.val
  · subst hjy
    rw [if_pos ⟨rfl, by rw [houtlen]; omega⟩, if_pos (by omega), hv]
  · rw [if_neg (fun h => hjy h.1), hinv j hj]
    by_cases hlt : j < y.val
    · rw [if_pos hlt, if_pos (by omega)]
    · rw [if_neg hlt, if_neg (by omega : ¬ j < y.val + 1)]

/-- `u8` to `i32` zero extension is the identity on the value. -/
lemma hcast_u8_i32_val (x : U8) : (UScalar.hcast .I32 x).val = (x.val : ℤ) := by
  rw [UScalar.hcast_val_eq]
  simp only [IScalarTy.I32_numBits_eq]
  rw [Int.bmod_eq_of_le] <;> scalar_tac

/-- `i32` to `i8` narrowing is the identity on values in the `i8` range. -/
lemma cast_i32_i8_val (x : I32) (h1 : -128 ≤ x.val) (h2 : x.val < 128) :
    (IScalar.cast .I8 x).val = x.val := by
  have hm : (2 : Nat) ^ (Min.min IScalarTy.I8.numBits IScalarTy.I32.numBits) = 256 := by
    rw [IScalarTy.I8_numBits_eq, IScalarTy.I32_numBits_eq]; norm_num
  rw [IScalar.cast_val_eq, hm, Int.bmod_eq_of_le] <;> omega

/-- One innermost code-loop body step: read the hmask bit and the two-bit code, write `out[y]`. -/
theorem q3L3_body_step (hm qs : Slice U8) (out : Array I8 256#usize) (m : U32)
    (y l : Usize) (half round plane : Nat) (base lo : Usize) (shift : U32)
    (hhm : 32 ≤ hm.val.length) (hqs : 64 ≤ qs.val.length) (hl : l.val < 16)
    (hbase : base.val = 32 * half) (hlo : lo.val = 16 * plane)
    (hy : y.val = 128 * half + 32 * round + 16 * plane + l.val)
    (hshift : shift.val = 2 * round) (hmv : m.val = 2 ^ (4 * half + round))
    (hplane : plane < 2) (hround : round < 4) (hhalf : half < 2)
    (hinv : q3Prefix hm.val qs.val out y.val) :
    ∃ (out2 : Array I8 256#usize) (y2 l2 : Usize),
      quant.q3_k_codes_loop0_loop0_loop0_loop0.body hm qs m base shift lo out y l
        = ok (ControlFlow.cont (out2, y2, l2)) ∧
      y2.val = y.val + 1 ∧ l2.val = l.val + 1 ∧
      q3Prefix hm.val qs.val out2 y2.val := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := l) (y := lo) (by rw [hlo]; scalar_tac))
  have hiv' : i.val = l.val + 16 * plane := by rw [hiv, hlo]
  have hmod : i.val = y.val % 32 := by rw [hiv', hy]; omega
  obtain ⟨i1, hi1, hi1v⟩ := slice_index_getD hm i (by rw [hiv']; omega)
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := base) (y := l) (by rw [hbase]; scalar_tac))
  have hi4' : i4.val = 32 * half + l.val := by rw [hi4v, hbase]
  obtain ⟨i5, hi5, hi5v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i4) (y := lo) (by rw [hi4', hlo]; scalar_tac))
  have hi5' : i5.val = 32 * half + l.val + 16 * plane := by rw [hi5v, hi4', hlo]
  have hidx : i5.val = y.val / 128 * 32 + y.val % 32 := by rw [hi5', hy]; omega
  obtain ⟨i6, hi6, hi6v⟩ := slice_index_getD qs i5 (by rw [hi5']; omega)
  obtain ⟨i7, hi7, hi7v, -⟩ := WP.spec_imp_exists
    (UScalar.ShiftRight_spec i6 shift (by rw [hshift, UScalarTy.U8_numBits_eq]; omega))
  have hi7' : i7.val = i6.val >>> shift.val := hi7v
  obtain ⟨y1, hy1, hy1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := y) (y := 1#usize) (by scalar_tac))
  have hy1' : y1.val = y.val + 1 := by rw [hy1v]; simp
  obtain ⟨l1, hl1, hl1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := l) (y := 1#usize) (by scalar_tac))
  have hl1' : l1.val = l.val + 1 := by rw [hl1v]; simp
  have hi3 : (core.convert.num.FromU32U8.from i1 &&& m).val
      = (hm.val.getD (y.val % 32) 0#u8).val &&& 2 ^ (4 * half + round) := by
    rw [UScalar.val_and, core.convert.num.FromU32U8.from_val_eq, hi1v, hmod, hmv]
  have hqv : i6.val = (qs.val.getD (y.val / 128 * 32 + y.val % 32) 0#u8).val := by
    rw [hi6v, hidx]
  have hqval : i7.val = (qs.val.getD (y.val / 128 * 32 + y.val % 32) 0#u8).val >>> shift.val := by
    rw [hi7', hqv]
  have hshift' : shift.val = 2 * (y.val % 128 / 32) := by rw [hshift, hy]; omega
  have hexp : 4 * (y.val / 128) + y.val % 128 / 32 = 4 * half + round := by rw [hy]; omega
  have hg1 : 4 * half + round < 8 := by omega
  have h9v : (UScalar.hcast .I32 (i7 &&& 3#u8)).val = ((i7.val &&& 3 : Nat) : ℤ) := by
    have h3 : (3#u8).val = 3 := by decide
    rw [hcast_u8_i32_val, UScalar.val_and, h3]
  have houtlen : out.val.length = 256 := by simp [Array.property]
  have hand : (i7.val &&& 3) ≤ 3 := Nat.and_le_right
  have h4v : (4#i32).val = (4 : ℤ) := by decide
  have h0v : (0#i32).val = (0 : ℤ) := by decide
  have hmin4 : IScalar.min .I32 ≤ (UScalar.hcast .I32 (i7 &&& 3#u8)).val - (4#i32).val := by
    rw [h9v, h4v, IScalar.min_IScalarTy_I32_eq, I32.min_eq]; omega
  have hmax4 : (UScalar.hcast .I32 (i7 &&& 3#u8)).val - (4#i32).val ≤ IScalar.max .I32 := by
    rw [h9v, h4v, IScalar.max_IScalarTy_I32_eq, I32.max_eq]; omega
  have hmin0 : IScalar.min .I32 ≤ (UScalar.hcast .I32 (i7 &&& 3#u8)).val - (0#i32).val := by
    rw [h9v, h0v, IScalar.min_IScalarTy_I32_eq, I32.min_eq]; omega
  have hmax0 : (UScalar.hcast .I32 (i7 &&& 3#u8)).val - (0#i32).val ≤ IScalar.max .I32 := by
    rw [h9v, h0v, IScalar.max_IScalarTy_I32_eq, I32.max_eq]; omega
  by_cases hbit : (core.convert.num.FromU32U8.from i1 &&& m) = 0#u32
  · have hbitv : (hm.val.getD (y.val % 32) 0#u8).val &&& 2 ^ (4 * half + round) = 0 := by
      have h := congrArg UScalar.val hbit
      rw [hi3] at h
      simpa using h
    have hbitv' : (hm.val.getD (y.val % 32) 0#u8).val
        &&& 2 ^ (4 * (y.val / 128) + y.val % 128 / 32) = 0 := by rw [hexp]; exact hbitv
    obtain ⟨i10, hi10, hi10v⟩ := WP.spec_imp_exists
      (IScalar.sub_spec (x := UScalar.hcast .I32 (i7 &&& 3#u8)) (y := 4#i32) hmin4 hmax4)
    have hi10val : i10.val = ((i7.val &&& 3 : Nat) : ℤ) - 4 := by
      rw [hi10v, h9v, h4v]
    have hi11val : (IScalar.cast .I8 i10).val = ((i7.val &&& 3 : Nat) : ℤ) - 4 := by
      rw [cast_i32_i8_val i10 (by rw [hi10val]; omega) (by rw [hi10val]; omega)]
      exact hi10val
    obtain ⟨a, ha, hav⟩ := WP.spec_imp_exists
      (Array.update_spec out y (IScalar.cast .I8 i10)
        (by show y.val < out.val.length; rw [houtlen, hy]; omega))
    refine ⟨a, y1, l1, ?_, hy1', hl1', ?_⟩
    · rw [quant.q3_k_codes_loop0_loop0_loop0_loop0.body, if_pos (by scalar_tac : l < 16#usize)]
      rw [hi]; simp only [bind_tc_ok]
      rw [hi1]; simp only [lift, bind_tc_ok]
      rw [if_neg (by rw [bne_iff_ne]; intro h; exact h hbit)]
      simp only [bind_tc_ok]
      rw [hi4]; simp only [bind_tc_ok]
      rw [hi5]; simp only [bind_tc_ok]
      rw [hi6]; simp only [bind_tc_ok]
      rw [hi7]; simp only [bind_tc_ok]
      rw [show I32.Insts.CoreConvertFromU8.from (i7 &&& 3#u8)
        = ok (UScalar.hcast .I32 (i7 &&& 3#u8)) from rfl]
      simp only [bind_tc_ok]
      rw [hi10]; simp only [bind_tc_ok]
      rw [ha]; simp only [bind_tc_ok]
      rw [hy1]; simp only [bind_tc_ok]
      rw [hl1]; simp only [bind_tc_ok]
    · rw [show a = out.set y (IScalar.cast .I8 i10) from hav, hy1']
      refine q3Prefix_push hm.val qs.val out y (IScalar.cast .I8 i10) hinv ?_
      rw [hi11val]
      unfold q3kCodeRef
      dsimp only
      rw [if_neg (fun h => h hbitv')]
      congr 1
      rw [hqval, hshift']
  · have hbitne : (core.convert.num.FromU32U8.from i1 &&& m) ≠ 0#u32 := hbit
    have hbitv : (hm.val.getD (y.val % 32) 0#u8).val &&& 2 ^ (4 * half + round) ≠ 0 := by
      intro h
      apply hbitne
      apply UScalar.eq_of_val_eq
      rw [hi3, h]
      rfl
    have hbitv' : ((hm.val.getD (y.val % 32) 0#u8).val
        &&& 2 ^ (4 * (y.val / 128) + y.val % 128 / 32)) ≠ 0 := by rw [hexp]; exact hbitv
    obtain ⟨i10, hi10, hi10v⟩ := WP.spec_imp_exists
      (IScalar.sub_spec (x := UScalar.hcast .I32 (i7 &&& 3#u8)) (y := 0#i32) hmin0 hmax0)
    have hi10val : i10.val = ((i7.val &&& 3 : Nat) : ℤ) := by
      rw [hi10v, h9v, h0v]; simp
    have hi11val : (IScalar.cast .I8 i10).val = ((i7.val &&& 3 : Nat) : ℤ) := by
      rw [cast_i32_i8_val i10 (by rw [hi10val]; omega) (by rw [hi10val]; omega)]
      exact hi10val
    obtain ⟨a, ha, hav⟩ := WP.spec_imp_exists
      (Array.update_spec out y (IScalar.cast .I8 i10)
        (by show y.val < out.val.length; rw [houtlen, hy]; omega))
    refine ⟨a, y1, l1, ?_, hy1', hl1', ?_⟩
    · rw [quant.q3_k_codes_loop0_loop0_loop0_loop0.body, if_pos (by scalar_tac : l < 16#usize)]
      rw [hi]; simp only [bind_tc_ok]
      rw [hi1]; simp only [lift, bind_tc_ok]
      rw [if_pos (by simpa only [bne_iff_ne] using hbitne)]
      simp only [bind_tc_ok]
      rw [hi4]; simp only [bind_tc_ok]
      rw [hi5]; simp only [bind_tc_ok]
      rw [hi6]; simp only [bind_tc_ok]
      rw [hi7]; simp only [bind_tc_ok]
      rw [show I32.Insts.CoreConvertFromU8.from (i7 &&& 3#u8)
        = ok (UScalar.hcast .I32 (i7 &&& 3#u8)) from rfl]
      simp only [bind_tc_ok]
      rw [hi10]; simp only [bind_tc_ok]
      rw [ha]; simp only [bind_tc_ok]
      rw [hy1]; simp only [bind_tc_ok]
      rw [hl1]; simp only [bind_tc_ok]
    · rw [show a = out.set y (IScalar.cast .I8 i10) from hav, hy1']
      refine q3Prefix_push hm.val qs.val out y (IScalar.cast .I8 i10) hinv ?_
      rw [hi11val]
      unfold q3kCodeRef
      dsimp only
      rw [if_pos hbitv']
      rw [hqval, hshift']
      simp

/-- The innermost code loop: sixteen writes extending the prefix by 16. -/
theorem q3L3_loop_spec (hm qs : Slice U8) (m : U32) (y0 : Usize) (half round plane : Nat)
    (base lo : Usize) (shift : U32) (out0 : Array I8 256#usize)
    (hhm : 32 ≤ hm.val.length) (hqs : 64 ≤ qs.val.length)
    (hbase : base.val = 32 * half) (hlo : lo.val = 16 * plane)
    (hy0 : y0.val = 128 * half + 32 * round + 16 * plane)
    (hshift : shift.val = 2 * round) (hmv : m.val = 2 ^ (4 * half + round))
    (hplane : plane < 2) (hround : round < 4) (hhalf : half < 2)
    (hinv : q3Prefix hm.val qs.val out0 y0.val) :
    WP.spec (quant.q3_k_codes_loop0_loop0_loop0_loop0 hm qs out0 m y0 base shift lo 0#usize)
      (fun p => p.2.val = y0.val + 16 ∧ q3Prefix hm.val qs.val p.1 p.2.val) := by
  unfold quant.q3_k_codes_loop0_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 256#usize × Usize × Usize => 16 - s.2.2.val)
    (inv := fun s => s.2.2.val ≤ 16 ∧
      s.2.1.val = 128 * half + 32 * round + 16 * plane + s.2.2.val ∧
      q3Prefix hm.val qs.val s.1 s.2.1.val)
  · rintro ⟨out, yv, lv⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hle, hyv, hpref⟩ := hinv'
    by_cases hlt : lv.val < 16
    · obtain ⟨out2, y2, l2, hbody, hy2, hl2, hpref2⟩ :=
        q3L3_body_step hm qs out m yv lv half round plane base lo shift
          hhm hqs hlt hbase hlo hyv hshift hmv hplane hround hhalf hpref
      rw [hbody]
      refine ⟨⟨?_, ?_, hpref2⟩, ?_⟩
      · show l2.val ≤ 16; omega
      · show y2.val = 128 * half + 32 * round + 16 * plane + l2.val; omega
      · show 16 - l2.val < 16 - lv.val; omega
    · have hnot : ¬ (lv < 16#usize) := fun hc => hlt ((UScalar.lt_equiv lv 16#usize).1 hc)
      have hbody : quant.q3_k_codes_loop0_loop0_loop0_loop0.body hm qs m base shift lo out yv lv
          = ok (ControlFlow.done (out, yv)) := by
        rw [quant.q3_k_codes_loop0_loop0_loop0_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hl16 : lv.val = 16 := by omega
      exact ⟨by omega, hpref⟩
  · dsimp only
    refine ⟨by decide, ?_, hinv⟩
    rw [hy0]; simp

/-- The plane loop: two innermost loops extend the prefix by 32. -/
theorem q3L2_loop_spec (hm qs : Slice U8) (m : U32) (y0 : Usize) (half round : Nat)
    (base : Usize) (shift : U32) (out0 : Array I8 256#usize)
    (hhm : 32 ≤ hm.val.length) (hqs : 64 ≤ qs.val.length)
    (hbase : base.val = 32 * half) (hy0 : y0.val = 128 * half + 32 * round)
    (hshift : shift.val = 2 * round) (hmv : m.val = 2 ^ (4 * half + round))
    (hround : round < 4) (hhalf : half < 2)
    (hinv : q3Prefix hm.val qs.val out0 y0.val) :
    WP.spec (quant.q3_k_codes_loop0_loop0_loop0 hm qs out0 m y0 base shift 0#usize)
      (fun p => p.2.val = y0.val + 32 ∧ q3Prefix hm.val qs.val p.1 p.2.val) := by
  unfold quant.q3_k_codes_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 256#usize × Usize × Usize => 2 - s.2.2.val)
    (inv := fun s => s.2.2.val ≤ 2 ∧
      s.2.1.val = 128 * half + 32 * round + 16 * s.2.2.val ∧
      q3Prefix hm.val qs.val s.1 s.2.1.val)
  · rintro ⟨out, yv, pv⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hle, hyv, hpref⟩ := hinv'
    by_cases hlt : pv.val < 2
    · obtain ⟨lo, hlo, hlov⟩ := WP.spec_imp_exists
        (Usize.mul_spec (x := pv) (y := 16#usize) (by scalar_tac))
      have hlo' : lo.val = 16 * pv.val := by
        rw [hlov, show (16#usize).val = 16 from by decide]; ring
      obtain ⟨p1, hbody2, hp1c⟩ := WP.spec_imp_exists
        (q3L3_loop_spec hm qs m yv half round pv.val base lo shift out
          hhm hqs hbase hlo' hyv hshift hmv hlt hround hhalf hpref)
      obtain ⟨hp1y, hp1pref⟩ := hp1c
      obtain ⟨o1, y1'⟩ := p1
      dsimp only at hbody2 hp1y hp1pref
      obtain ⟨pv1, hpv1, hpv1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := pv) (y := 1#usize) (by scalar_tac))
      have hpv1' : pv1.val = pv.val + 1 := by rw [hpv1v]; simp
      have hbody : quant.q3_k_codes_loop0_loop0_loop0.body hm qs m base shift out yv pv
          = ok (ControlFlow.cont (o1, y1', pv1)) := by
        rw [quant.q3_k_codes_loop0_loop0_loop0.body, if_pos ((UScalar.lt_equiv pv 2#usize).2 hlt),
          hlo]
        simp only [bind_tc_ok]
        rw [show quant.q3_k_codes_loop0_loop0_loop0_loop0 hm qs out m yv base shift lo 0#usize
          = ok (o1, y1') from hbody2]
        simp only [bind_tc_ok]
        rw [hpv1]
        simp only [bind_tc_ok]
        rfl
      rw [hbody]
      refine ⟨⟨?_, ?_, hp1pref⟩, ?_⟩
      · show pv1.val ≤ 2; omega
      · show y1'.val = 128 * half + 32 * round + 16 * pv1.val; omega
      · show 2 - pv1.val < 2 - pv.val; omega
    · have hnot : ¬ (pv < 2#usize) := fun hc => hlt ((UScalar.lt_equiv pv 2#usize).1 hc)
      have hbody : quant.q3_k_codes_loop0_loop0_loop0.body hm qs m base shift out yv pv
          = ok (ControlFlow.done (out, yv)) := by
        rw [quant.q3_k_codes_loop0_loop0_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hp2 : pv.val = 2 := by omega
      refine ⟨?_, hpref⟩
      show yv.val = y0.val + 32
      rw [hyv, hy0, hp2]
  · dsimp only
    refine ⟨by decide, ?_, hinv⟩
    rw [hy0]; simp

/-- The round loop: four plane loops extend the prefix by 128 and advance `shift` and `m`. -/
theorem q3L1_loop_spec (hm qs : Slice U8) (m0 : U32) (y0 : Usize) (half : Nat)
    (base : Usize) (out0 : Array I8 256#usize)
    (hhm : 32 ≤ hm.val.length) (hqs : 64 ≤ qs.val.length)
    (hbase : base.val = 32 * half) (hy0 : y0.val = 128 * half)
    (hm0v : m0.val = 2 ^ (4 * half)) (hhalf : half < 2)
    (hinv : q3Prefix hm.val qs.val out0 y0.val) :
    WP.spec (quant.q3_k_codes_loop0_loop0 hm qs out0 m0 y0 base 0#u32 0#usize)
      (fun p => p.2.2.val = y0.val + 128 ∧ p.2.1.val = 2 ^ (4 * half + 4) ∧
        q3Prefix hm.val qs.val p.1 p.2.2.val) := by
  unfold quant.q3_k_codes_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 256#usize × U32 × Usize × U32 × Usize => 4 - s.2.2.2.2.val)
    (inv := fun s => s.2.2.2.2.val ≤ 4 ∧
      s.2.2.1.val = 128 * half + 32 * s.2.2.2.2.val ∧
      s.2.2.2.1.val = 2 * s.2.2.2.2.val ∧ s.2.1.val = 2 ^ (4 * half + s.2.2.2.2.val) ∧
      q3Prefix hm.val qs.val s.1 s.2.2.1.val)
  · rintro ⟨out, m, yv, shift, round⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hle, hyv, hshift, hmv, hpref⟩ := hinv'
    by_cases hlt : round.val < 4
    · obtain ⟨p1, hbody2, hp1c⟩ := WP.spec_imp_exists
        (q3L2_loop_spec hm qs m yv half round.val base shift out
          hhm hqs hbase hyv hshift hmv hlt hhalf hpref)
      obtain ⟨hp1y, hp1pref⟩ := hp1c
      obtain ⟨o1, y1'⟩ := p1
      dsimp only at hbody2 hp1y hp1pref
      obtain ⟨shift1, hsh1, hsh1v⟩ := WP.spec_imp_exists
        (U32.add_spec (x := shift) (y := 2#u32) (by scalar_tac))
      have hsh1' : shift1.val = 2 * (round.val + 1) := by
        rw [hsh1v, hshift, show (2#u32).val = 2 from by decide]; ring
      obtain ⟨m1, hm1eq, hm1c⟩ := WP.spec_imp_exists
        (U32.ShiftLeft_IScalar_spec m 1#i32 (by simp) (by simp))
      obtain ⟨hm1v, -⟩ := hm1c
      have hm1' : m1.val = 2 ^ (4 * half + (round.val + 1)) := by
        rw [hm1v, show (1#i32).toNat = 1 from rfl, hmv]
        rw [U32.size, U32.numBits, UScalarTy.U32_numBits_eq]
        rw [Nat.shiftLeft_eq, show (2:Nat) ^ 1 = 2 from by norm_num]
        have hexpa : 4 * half + (round.val + 1) = (4 * half + round.val) + 1 := by omega
        rw [hexpa, ← pow_succ]
        have hlt : 2 ^ (4 * half + round.val) * 2 < 2 ^ 32 := by
          rw [← pow_succ]
          exact Nat.pow_lt_pow_right (by norm_num : 1 < 2) (by omega)
        exact Nat.mod_eq_of_lt hlt
      obtain ⟨round1, hr1, hr1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := round) (y := 1#usize) (by scalar_tac))
      have hr1' : round1.val = round.val + 1 := by rw [hr1v]; simp
      have hbody : quant.q3_k_codes_loop0_loop0.body hm qs base out m yv shift round
          = ok (ControlFlow.cont (o1, m1, y1', shift1, round1)) := by
        rw [quant.q3_k_codes_loop0_loop0.body, if_pos ((UScalar.lt_equiv round 4#usize).2 hlt)]
        rw [show quant.q3_k_codes_loop0_loop0_loop0 hm qs out m yv base shift 0#usize
          = ok (o1, y1') from hbody2]
        simp only [bind_tc_ok]
        rw [hsh1, hm1eq, hr1]
        simp only [bind_tc_ok]
        rfl
      rw [hbody]
      refine ⟨⟨?_, ?_, ?_, ?_, hp1pref⟩, ?_⟩
      · show round1.val ≤ 4; omega
      · show y1'.val = 128 * half + 32 * round1.val; omega
      · show shift1.val = 2 * round1.val; rw [hsh1', hr1']
      · show m1.val = 2 ^ (4 * half + round1.val); rw [hm1']; congr 1; rw [hr1']
      · show 4 - round1.val < 4 - round.val; omega
    · have hnot : ¬ (round < 4#usize) := fun hc => hlt ((UScalar.lt_equiv round 4#usize).1 hc)
      have hbody : quant.q3_k_codes_loop0_loop0.body hm qs base out m yv shift round
          = ok (ControlFlow.done (out, m, yv)) := by
        rw [quant.q3_k_codes_loop0_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hr4 : round.val = 4 := by omega
      refine ⟨?_, ?_, hpref⟩
      · show yv.val = y0.val + 128; rw [hyv, hy0, hr4]
      · show m.val = 2 ^ (4 * half + 4); rw [hmv, hr4]
  · dsimp only
    refine ⟨by decide, ?_, by decide, ?_, hinv⟩
    · rw [hy0, show (0#usize).val = 0 from by decide]; try omega
    · rw [hm0v, show (0#usize).val = 0 from by decide]; norm_num

/-- The half loop: two round loops extend the prefix by 256 and advance `m`. -/
theorem q3L0_loop_spec (hm qs : Slice U8) (m0 : U32) (y0 : Usize)
    (out0 : Array I8 256#usize) (hhm : 32 ≤ hm.val.length) (hqs : 64 ≤ qs.val.length)
    (hm0v : m0.val = 1) (hy0 : y0.val = 0)
    (hinv : q3Prefix hm.val qs.val out0 y0.val) :
    WP.spec (quant.q3_k_codes_loop0 hm qs out0 m0 y0 0#usize)
      (fun out => q3Prefix hm.val qs.val out 256) := by
  unfold quant.q3_k_codes_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 256#usize × U32 × Usize × Usize => 2 - s.2.2.2.val)
    (inv := fun s => s.2.2.2.val ≤ 2 ∧ s.2.2.1.val = 128 * s.2.2.2.val ∧
      s.2.1.val = 2 ^ (4 * s.2.2.2.val) ∧ q3Prefix hm.val qs.val s.1 s.2.2.1.val)
  · rintro ⟨out, m, yv, half⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hle, hyv, hmv, hpref⟩ := hinv'
    by_cases hlt : half.val < 2
    · obtain ⟨base, hbase, hbasev⟩ := WP.spec_imp_exists
        (Usize.mul_spec (x := half) (y := 32#usize) (by scalar_tac))
      have hbase' : base.val = 32 * half.val := by
        rw [hbasev, show (32#usize).val = 32 from by decide]; ring
      obtain ⟨p1, hbody2, hp1c⟩ := WP.spec_imp_exists
        (q3L1_loop_spec hm qs m yv half.val base out hhm hqs hbase' hyv hmv hlt hpref)
      obtain ⟨hp1y, hp1m, hp1pref⟩ := hp1c
      obtain ⟨o1, m1', y1'⟩ := p1
      dsimp only at hbody2 hp1y hp1m hp1pref
      obtain ⟨half1, hh1, hh1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := half) (y := 1#usize) (by scalar_tac))
      have hh1' : half1.val = half.val + 1 := by rw [hh1v]; simp
      have hbody : quant.q3_k_codes_loop0.body hm qs out m yv half
          = ok (ControlFlow.cont (o1, m1', y1', half1)) := by
        rw [quant.q3_k_codes_loop0.body, if_pos ((UScalar.lt_equiv half 2#usize).2 hlt), hbase]
        simp only [bind_tc_ok]
        rw [show quant.q3_k_codes_loop0_loop0 hm qs out m yv base 0#u32 0#usize
          = ok (o1, m1', y1') from hbody2]
        simp only [bind_tc_ok]
        rw [hh1]
        simp only [bind_tc_ok]
        rfl
      rw [hbody]
      refine ⟨⟨?_, ?_, ?_, hp1pref⟩, ?_⟩
      · show half1.val ≤ 2; omega
      · show y1'.val = 128 * half1.val; omega
      · show m1'.val = 2 ^ (4 * half1.val); rw [hp1m]; congr 1; rw [hh1']; ring
      · show 2 - half1.val < 2 - half.val; omega
    · have hnot : ¬ (half < 2#usize) := fun hc => hlt ((UScalar.lt_equiv half 2#usize).1 hc)
      have hbody : quant.q3_k_codes_loop0.body hm qs out m yv half = ok (ControlFlow.done out) := by
        rw [quant.q3_k_codes_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hh2 : half.val = 2 := by omega
      have hyv256 : yv.val = 256 := by rw [hyv, hh2]
      exact hyv256 ▸ hpref
  · dsimp only
    refine ⟨by decide, ?_, ?_, hinv⟩
    · rw [hy0, show (0#usize).val = 0 from by decide]; try omega
    · rw [hm0v, show (0#usize).val = 0 from by decide]; norm_num

/-- The Q3_K code reader: on a whole 110-byte block, `quant::q3_k_codes` succeeds and returns the
256 signed codes (`q - 4·[hmask bit clear]`). -/
theorem q3_k_codes_ok (blk : Slice U8) (hlen : 110 ≤ blk.val.length) :
    ∃ v : Array I8 256#usize,
      quant.q3_k_codes blk = ok (.Ok v) ∧
      ∀ i, i < 256 →
        (v.val.getD i 0#i8).val
          = q3kCodeRef (blk.val.take 32) ((blk.val.drop 32).take 64) i := by
  have hspec : quant.q3_k_codes blk ⦃ r => ∃ v : Array I8 256#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array I8 256#usize) quant.QuantError) ∧
      ∀ i, i < 256 →
        (v.val.getD i 0#i8).val
          = q3kCodeRef (blk.val.take 32) ((blk.val.drop 32).take 64) i ⦄ := by
    have hneed := quant_need_ok blk 110#usize (by simpa using hlen)
    unfold quant.q3_k_codes
    rw [quant_Q3_K_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, hneed]
    simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
    step*
    · rw [i1_post, quant.QK_K]; norm_num; omega
    · rw [i3_post, i1_post, i2_post, quant.QK_K]; norm_num; omega
    · have h256 : (256#usize).val = 256 := by decide
      have hi1 : i1.val = 32 := by rw [i1_post, quant.QK_K, h256]
      have hi3 : i3.val = 96 := by rw [i3_post, hi1, i2_post, quant.QK_K, h256]
      have hhm : hm.val = blk.val.take 32 := by
        rw [hm_post]; simp only [List.slice]; rw [hi1]; simp
      have hqs : qs.val = (blk.val.drop 32).take 64 := by
        rw [qs_post]; simp only [List.slice]; rw [hi1, hi3]
      have hhmlen : 32 ≤ hm.val.length := by rw [hhm]; simp [List.length_take]; omega
      have hqslen : 64 ≤ qs.val.length := by
        rw [hqs]; simp [List.length_take, List.length_drop]; omega
      have hinv0 : q3Prefix hm.val qs.val (Array.repeat 256#usize 0#i8) 0 := by
        intro j hj
        rw [Array.repeat_val]
        simp only [List.getD, List.getElem?_replicate]
        rw [if_pos (by simpa using hj), if_neg (by omega : ¬ j < 0)]
        rfl
      refine WP.spec_bind (q3L0_loop_spec hm qs 1#u32 0#usize (Array.repeat 256#usize 0#i8)
        hhmlen hqslen (by decide) (by decide) hinv0) ?_
      intro out1 hout1
      refine ⟨out1, rfl, ?_⟩
      intro i hi
      rw [hout1 i hi, if_pos hi, ← hhm, ← hqs]
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  exact ⟨v, by rw [hr, hrv], hpt⟩

/-- **`Q3_K` block law.** On the block window returned by `block_slice` with finite `d`,
`q3_k_block` returns `(d, scales, codes)` with `d` the exact Q32.32 `f16` value, the sixteen
signed scales, and the 256 codes. -/
theorem q3_k_block_ok (raw : Slice U8) (block : Usize) (blk : Slice U8)
    (hblk : dequant.block_slice raw block 110#usize = ok (.Ok blk))
    (h110 : 110 ≤ blk.val.length)
    (hd : f16Exp (f16LeHalf blk 108) ≠ 31) :
    ∃ (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize),
      dequant.q3_k_block raw block = ok (.Ok (d, scales, codes)) ∧
      d.val = f16Value (f16LeHalf blk 108) ∧
      (∀ k, k < 16 → (scales.val.getD k 0#i8).val = q3kScaleRef ((blk.val.drop 96).take 12) k) ∧
      (∀ i, i < 256 →
        (codes.val.getD i 0#i8).val
          = q3kCodeRef (blk.val.take 32) ((blk.val.drop 32).take 64) i) := by
  have hdok := quant_q3_k_d_ok blk h110
  obtain ⟨d, hd_eq, hdv⟩ := f16_to_fixed_exact (f16LeHalf blk 108) hd
  obtain ⟨scales, hsc_eq, hsc_inv⟩ := q3_k_scales_ok blk h110
  obtain ⟨codes, hcodes_eq, hcodes⟩ := q3_k_codes_ok blk h110
  have hcodes_eq' : quant.q3_k_codes blk = ok (.Ok codes) := hcodes_eq
  refine ⟨d, scales, codes, ?_, hdv, ?_, hcodes⟩
  · simp only [dequant.q3_k_block, quant_Q3_K_BLOCK_BYTES_eq, hblk, hdok, hd_eq, hsc_eq,
      hcodes_eq', core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  · intro k hk
    exact hsc_inv k hk

end
