/-
Refinement of the extracted ternary trit faces `TQ1_0` and `PTQ1_0`
.

The Rust bodies are `dequant::lowbit::trit`, `dequant::lowbit::tq1_0_block`,
`dequant::lowbit::ptq1_0_block`, `dequant::lowbit::dequant_tq1_0_row` and
`dequant::lowbit::dequant_ptq1_0_row` in `src/engine/dequant_lowbit.rs`; the block lengths live in
`src/format/quant.rs`. A weight is the checked `i64` product `d·(trit − 1)` of the block's exact
Q32.32 scale and a base-3 trit recovered from a packed byte.

`trit(b, n) = (((b·3^n) mod 256)·3) >> 8` (the fork's `u8` wrap written in `u16` with an explicit
`& 0xFF`), an element of `{0, 1, 2}`. The staged walk emits, for `TQ1_0`, stage 32 then 16 over
`qs` and the `qh` tail; for `PTQ1_0` the 32-byte stage does not fit 24 bytes, leaving 16 then 8.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed literal
facts, never on an extracted body.
-/
import Mathlib.Tactic.IntervalCases
import Numeric.DequantQ8_0
import Numeric.DequantQ4KBlock

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The trit decoder

| symbol | meaning |
|---|---|
| `tritRef b n` | the `n`-th base-3 trit of the packed byte `b`, most significant first |
-/

/-- The trit recovered by the fork's `u8`-wrapping rule, as an integer in `{0, 1, 2}`. -/
def tritRef (b : U8) (n : Nat) : ℤ := (((b.val * 3 ^ n) % 256) * 3 / 256 : Nat)

/-- The `n`-th base-3 digit of `q`, least significant first. -/
def tritDigit (q n : Nat) : ℤ := ((q / 3 ^ n) % 3 : Nat)

/-- `POW3` has six entries. -/
lemma POW3_len : dequant.lowbit.POW3.length = 6 :=
  Array.length_eq dequant.lowbit.POW3

/-- `POW3` read at `n` is `3^n`. -/
lemma POW3_val (n : Nat) (h : n < dequant.lowbit.POW3.length) :
    (dequant.lowbit.POW3.val[n]'h).val = 3 ^ n := by
  have hd : (dequant.lowbit.POW3.val.getD n 0#u16).val = 3 ^ n := by
    have h6 : n < 6 := by have := POW3_len; omega
    have hsplit : n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 := by omega
    rcases hsplit with h | h | h | h | h | h <;> subst h <;>
      (unfold dequant.lowbit.POW3; decide)
  rwa [List.getD_eq_getElem (l := dequant.lowbit.POW3.val) (d := 0#u16) h] at hd

/-- **Trit decode law.** For a valid plane index (`n < 6`, i.e. `3^n` in the `POW3` table) the
extracted `trit` returns the exact reference value. -/
theorem trit_spec (b : U8) (n : Usize) (hn : n.val < 6) :
    ∃ v : I64, dequant.lowbit.trit b n = ok v ∧ v.val = tritRef b n := by
  have hb : b.val ≤ 255 := by
    have := U8.lt_succ_max b
    omega
  have hpow : 3 ^ n.val ≤ 243 := by
    have := Nat.pow_le_pow_right (by norm_num : 1 ≤ 3) (by omega : n.val ≤ 5)
    norm_num at this ⊢
    omega
  have hcast : (core.convert.num.FromU16U8.from b).val = b.val :=
    core.convert.num.FromU16U8.from_val_eq b
  have hidx : n.val < dequant.lowbit.POW3.length := by rw [POW3_len]; exact hn
  obtain ⟨i1, hpow3, hi1eq⟩ := WP.spec_imp_exists
    (Array.index_usize_spec dequant.lowbit.POW3 n hidx)
  have hi1 : i1.val = 3 ^ n.val := by
    rw [hi1eq]
    exact POW3_val n.val hidx
  have hmul1_bound : b.val * 3 ^ n.val ≤ U16.max := by
    have h := Nat.mul_le_mul hb hpow
    rw [U16.max_eq]
    norm_num at h ⊢
    omega
  obtain ⟨i2, hmul1, hi2v⟩ := WP.spec_imp_exists
    (U16.mul_spec (x := core.convert.num.FromU16U8.from b) (y := i1) (by
      rw [hcast, hi1]; exact hmul1_bound))
  have hi2 : i2.val = b.val * 3 ^ n.val := by rw [hi2v, hcast, hi1]
  obtain ⟨i3, hi3eq, hi3val, -⟩ := WP.spec_imp_exists (UScalar.and_spec i2 255#u16)
  have hi3 : i3.val = (b.val * 3 ^ n.val) % 256 := by
    rw [hi3val, UScalar.val_and, hi2]
    have h255 : (255#u16).val = 2 ^ 8 - 1 := by simp
    rw [h255, Nat.and_two_pow_sub_one_eq_mod]
  have hi3lt : i3.val ≤ 255 := by
    rw [hi3]
    have := Nat.mod_lt (b.val * 3 ^ n.val) (by norm_num : 0 < 256)
    omega
  have hmul2_bound : i3.val * (3#u16).val ≤ U16.max := by
    have h3 : (3#u16).val = 3 := rfl
    rw [h3, U16.max_eq]
    omega
  obtain ⟨i4, hmul2, hi4v⟩ := WP.spec_imp_exists
    (U16.mul_spec (x := i3) (y := 3#u16) hmul2_bound)
  have hi4 : i4.val = (b.val * 3 ^ n.val) % 256 * 3 := by rw [hi4v, hi3]; norm_num
  obtain ⟨i5, hsh, hi5v, -⟩ := WP.spec_imp_exists
    (U16.ShiftRight_IScalar_spec i4 8#i32 (by simp) (by simp))
  have hi5 : i5.val = ((b.val * 3 ^ n.val) % 256 * 3 / 256) := by
    have h8 : (8#i32).toNat = 8 := by decide
    rw [hi5v, h8, Nat.shiftRight_eq_div_pow, hi4]
  have hlift : lift (core.convert.num.FromU16U8.from b)
      = ok (core.convert.num.FromU16U8.from b) := rfl
  have hcv : I64.Insts.CoreConvertFromU16.from i5 = ok (UScalar.hcast .I64 i5) := rfl
  refine ⟨UScalar.hcast .I64 i5, ?_, ?_⟩
  · rw [dequant.lowbit.trit]
    simp only [bind_tc_ok, hlift, hpow3, hmul1, hi3eq, hmul2, hsh, hcv]
  · have h2 : i5.val ≤ 2 := by
      rw [hi5]
      have := Nat.mod_lt (b.val * 3 ^ n.val) (by norm_num : 0 < 256)
      omega
    have hs : lift (UScalar.hcast .I64 i5) ⦃ y => y.val = (i5.val : ℤ) ⦄ :=
      UScalar.hcast_inBounds_spec .I64 i5 (by
        show i5.val ≤ IScalar.max IScalarTy.I64
        scalar_tac)
    obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists hs
    have hyeq : y = UScalar.hcast .I64 i5 := by
      have hh := hy
      simp only [lift, ok.injEq] at hh
      exact hh.symm
    rw [hyeq] at hyv
    rw [hyv]
    unfold tritRef
    rw [← hi5]

/-- **Fork quantizer law.** If `b` is the byte the fork's quantizer emits for `q < 243`
(`ggml-quants.c:2508-2515`, `b = (256q + 242) / 243`) then `tritRef b i` is base-3 digit `4 − i`
of `q`: the five trits `tritRef b 0 … 4` are the digits of `q` most significant first. -/
theorem trit_digits (b : U8) (q : Nat) (hq : q < 243) (hb : b.val = (q * 256 + 242) / 243) :
    ∀ i, i < 5 → tritRef b i = tritDigit q (4 - i) := by
  intro i hi
  have aux : (((q * 256 + 242) / 243) * 3 ^ i % 256) * 3 / 256
      = (q / 3 ^ (4 - i)) % 3 := by
    interval_cases q <;> interval_cases i <;> decide
  unfold tritRef tritDigit
  rw [hb]
  exact_mod_cast aux

/-! ## Block-length constants -/

/-- `quant::TQ1_0_QK` is the `usize` literal `256`. -/
theorem quant_TQ1_0_QK_eq : quant.TQ1_0_QK = 256#usize := by with_unfolding_all rfl

/-- `quant::PTQ1_0_QK` is the `usize` literal `128`. -/
theorem quant_PTQ1_0_QK_eq : quant.PTQ1_0_QK = 128#usize := by with_unfolding_all rfl

/-- `quant::TQ1_0_BLOCK_BYTES` evaluates to 54 bytes: `48 + 4 + 2`. -/
theorem quant_TQ1_0_BLOCK_BYTES_eq : quant.TQ1_0_BLOCK_BYTES = ok (54#usize) := by
  have h : quant.TQ1_0_BLOCK_BYTES ⦃ z => z = (54#usize) ⦄ := by
    unfold quant.TQ1_0_BLOCK_BYTES
    rw [quant_TQ1_0_QK_eq]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-- `quant::PTQ1_0_BLOCK_BYTES` evaluates to 28 bytes: `24 + 2 + 2`. -/
theorem quant_PTQ1_0_BLOCK_BYTES_eq : quant.PTQ1_0_BLOCK_BYTES = ok (28#usize) := by
  have h : quant.PTQ1_0_BLOCK_BYTES ⦃ z => z = (28#usize) ⦄ := by
    unfold quant.PTQ1_0_BLOCK_BYTES
    rw [quant_PTQ1_0_QK_eq]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## Slice helpers -/

/-- The exact Q32.32 scale `d` of TQ1_0 block `b`, read at byte `54·b + 52`. -/
def tq1_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (b * 54 + 52))

/-- The exact Q32.32 scale `d` of PTQ1_0 block `b`, read at byte `28·b + 26`. -/
def ptq1_0ScaleRef (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (b * 28 + 26))

/-- `f16LeHalf` of a 54-byte window at offset `j` is the raw half-word at `s + j`. -/
lemma f16LeHalf_block54 (raw : Slice U8) (s j : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 54)) (he : s + 54 ≤ raw.val.length)
    (hj : j + 1 < 54) :
    f16LeHalf blk j = f16LeHalf raw (s + j) := by
  unfold f16LeHalf
  congr 2
  rw [h, getD_slice raw.val s (s + 54) j he (by omega),
    getD_slice raw.val s (s + 54) (j + 1) he (by omega)]
  simp only [Nat.add_assoc]

/-- `f16LeHalf` of a 28-byte window at offset `j` is the raw half-word at `s + j`. -/
lemma f16LeHalf_block28 (raw : Slice U8) (s j : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 28)) (he : s + 28 ≤ raw.val.length)
    (hj : j + 1 < 28) :
    f16LeHalf blk j = f16LeHalf raw (s + j) := by
  unfold f16LeHalf
  congr 2
  rw [h, getD_slice raw.val s (s + 28) j he (by omega),
    getD_slice raw.val s (s + 28) (j + 1) he (by omega)]
  simp only [Nat.add_assoc]

/-! ## Block fill invariants -/

/-- The array fill invariant: entries below `k` hold the source bytes `o + j`. -/
def fillInv {n : Usize} (blk : Slice U8) (o : Nat) (dst : Array U8 n) (k : Nat) : Prop :=
  k ≤ n.val ∧ ∀ j, j < k → (dst.val.getD j 0#u8).val = (blk.val.getD (o + j) 0#u8).val

@[simp] lemma usize48_val : (48#usize).val = 48 := rfl
@[simp] lemma usize4_val : (4#usize).val = 4 := rfl
@[simp] lemma usize52_val : (52#usize).val = 52 := rfl
@[simp] lemma usize54_val : (54#usize).val = 54 := rfl
@[simp] lemma usize24_val : (24#usize).val = 24 := rfl
@[simp] lemma usize2_val : (2#usize).val = 2 := rfl
@[simp] lemma usize28_val : (28#usize).val = 28 := rfl
@[simp] lemma usize26_val : (26#usize).val = 26 := rfl

/-! ## `tq1_0_block_loop0` (48 `qs` bytes) -/

theorem tq1_0Qs_body_step (blk : Slice U8) (qs : Array U8 48#usize) (k : Usize)
    (hblk : 48 ≤ blk.val.length) (hk : k.val < 48) (hinv : fillInv blk 0 qs k.val) :
    ∃ (qs2 : Array U8 48#usize) (k2 : Usize),
      dequant.lowbit.tq1_0_block_loop0.body blk qs k = ok (ControlFlow.cont (qs2, k2)) ∧
      fillInv blk 0 qs2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨-, hrec⟩ := hinv
  obtain ⟨i, hidx, hiv⟩ := slice_index_getD blk k (lt_of_lt_of_le hk hblk)
  obtain ⟨a, hupd, hav⟩ := WP.spec_imp_exists
    (Array.update_spec qs k i (by show k.val < qs.val.length; rw [Array.property qs]; exact hk))
  obtain ⟨k1, hadd, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk1val : k1.val = k.val + 1 := by rw [hk1v]; simp
  refine ⟨a, k1, ?_, ?_, hk1val⟩
  · unfold dequant.lowbit.tq1_0_block_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hk)]
    simp only [hidx, hupd, hadd, bind_tc_ok]
  · refine ⟨by show k.val + 1 ≤ 48; omega, ?_⟩
    intro j hj
    rw [hav, Array.set_val_eq, getD_set]
    by_cases hjk : j = k.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [Array.property qs]; exact hk⟩]
      simpa using hiv
    · rw [if_neg (fun hh => hjk hh.1)]
      exact hrec j (by omega)

theorem tq1_0Qs_loop_spec (blk : Slice U8) (qs0 : Array U8 48#usize) (k : Usize)
    (hblk : 48 ≤ blk.val.length) (hinv : fillInv blk 0 qs0 k.val) :
    WP.spec (dequant.lowbit.tq1_0_block_loop0 blk qs0 k) (fun qs => fillInv blk 0 qs 48) := by
  unfold dequant.lowbit.tq1_0_block_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 48#usize × Usize => 48 - s.2.val)
    (inv := fun s => fillInv blk 0 s.1 s.2.val)
    (post := fun qs => fillInv blk 0 qs 48)
  · rintro ⟨qs, kk⟩ hinv'
    dsimp only
    by_cases hlt : kk.val < 48
    · obtain ⟨qs2, k2, hbody, hinv2, hk2⟩ := tq1_0Qs_body_step blk qs kk hblk hlt hinv'
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨by rw [hk2]; exact hinv2, by rw [hk2]; omega⟩
    · have hnot : ¬ (kk < 48#usize) := fun hc => hlt ((UScalar.lt_equiv kk 48#usize).1 hc)
      have hbody : dequant.lowbit.tq1_0_block_loop0.body blk qs kk = ok (ControlFlow.done qs) := by
        unfold dequant.lowbit.tq1_0_block_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hk48 : kk.val = 48 := by have := hinv'.1; simp at this; omega
      simpa [hk48] using hinv'
  · exact hinv

/-! ## `tq1_0_block_loop1` (4 `qh` bytes at 48) -/

theorem tq1_0Qh_body_step (blk : Slice U8) (qh : Array U8 4#usize) (h : Usize)
    (hblk : 52 ≤ blk.val.length) (hh : h.val < 4) (hinv : fillInv blk 48 qh h.val) :
    ∃ (qh2 : Array U8 4#usize) (h2 : Usize),
      dequant.lowbit.tq1_0_block_loop1.body blk qh h = ok (ControlFlow.cont (qh2, h2)) ∧
      fillInv blk 48 qh2 (h.val + 1) ∧ h2.val = h.val + 1 := by
  obtain ⟨-, hrec⟩ := hinv
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 48#usize) (y := h) (by scalar_tac))
  have hiv' : i.val = 48 + h.val := by rw [hiv]; simp
  obtain ⟨i1, hidx, hi1v⟩ := slice_index_getD blk i (by
    rw [hiv']; exact lt_of_lt_of_le (by omega : 48 + h.val < 52) hblk)
  obtain ⟨a, hupd, hav⟩ := WP.spec_imp_exists
    (Array.update_spec qh h i1 (by show h.val < qh.val.length; rw [Array.property qh]; exact hh))
  obtain ⟨h1, hadd, hh1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := h) (y := 1#usize) (by scalar_tac))
  have hh1val : h1.val = h.val + 1 := by rw [hh1v]; simp
  refine ⟨a, h1, ?_, ?_, hh1val⟩
  · unfold dequant.lowbit.tq1_0_block_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hh)]
    simp only [hi, hidx, hupd, hadd, bind_tc_ok]
  · refine ⟨by show h.val + 1 ≤ 4; omega, ?_⟩
    intro j hj
    rw [hav, Array.set_val_eq, getD_set]
    by_cases hjk : j = h.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [Array.property qh]; exact hh⟩]
      rw [hiv'] at hi1v
      exact hi1v
    · rw [if_neg (fun hh => hjk hh.1)]
      exact hrec j (by omega)

theorem tq1_0Qh_loop_spec (blk : Slice U8) (qh0 : Array U8 4#usize) (h : Usize)
    (hblk : 52 ≤ blk.val.length) (hinv : fillInv blk 48 qh0 h.val) :
    WP.spec (dequant.lowbit.tq1_0_block_loop1 blk qh0 h) (fun qh => fillInv blk 48 qh 4) := by
  unfold dequant.lowbit.tq1_0_block_loop1
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 4#usize × Usize => 4 - s.2.val)
    (inv := fun s => fillInv blk 48 s.1 s.2.val)
    (post := fun qh => fillInv blk 48 qh 4)
  · rintro ⟨qh, hh⟩ hinv'
    dsimp only
    by_cases hlt : hh.val < 4
    · obtain ⟨qh2, h2, hbody, hinv2, hh2⟩ := tq1_0Qh_body_step blk qh hh hblk hlt hinv'
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨by rw [hh2]; exact hinv2, by rw [hh2]; omega⟩
    · have hnot : ¬ (hh < 4#usize) := fun hc => hlt ((UScalar.lt_equiv hh 4#usize).1 hc)
      have hbody : dequant.lowbit.tq1_0_block_loop1.body blk qh hh = ok (ControlFlow.done qh) := by
        unfold dequant.lowbit.tq1_0_block_loop1.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hh4 : hh.val = 4 := by have := hinv'.1; simp at this; omega
      simpa [hh4] using hinv'
  · exact hinv

/-! ## `tq1_0_block` -/

/-- **`TQ1_0` block law.** On a whole 54-byte block with finite `d`, `tq1_0_block` returns the exact
Q32.32 scale, the 48 `qs` bytes at `54·block + j` and the 4 `qh` bytes at `54·block + 48 + h`. -/
theorem tq1_0_block_ok (raw : Slice U8) (block : Usize)
    (hlen : block.val * 54 + 54 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw (block.val * 54 + 52)) ≠ 31) :
    ∃ (d : I64) (qs : Array U8 48#usize) (qh : Array U8 4#usize),
      dequant.lowbit.tq1_0_block raw block = ok (.Ok (d, qs, qh)) ∧
      d.val = tq1_0ScaleRef raw block.val ∧
      (∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (block.val * 54 + j) 0#u8).val) ∧
      (∀ h, h < 4 →
        (qh.val.getD h 0#u8).val = (raw.val.getD (block.val * 54 + 48 + h) 0#u8).val) := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok raw block 54#usize (by simpa using hlen)
  have hslice' : blk.val = raw.val.slice (block.val * 54) (block.val * 54 + 54) := by
    rw [hslice, usize54_val]
  have hlenblk : blk.val.length = 54 := by
    rw [hslice', List.slice, List.length_take, List.length_drop]
    omega
  have hf16 : f16LeHalf blk 52 = f16LeHalf raw (block.val * 54 + 52) :=
    f16LeHalf_block54 raw (block.val * 54) 52 blk hslice' (by omega) (by norm_num)
  obtain ⟨d, hd_eq, hdv⟩ :=
    read_scale_ok blk 52#usize (by simp [hlenblk])
      (by rw [usize52_val, hf16]; exact hd)
  have hdv' : d.val = tq1_0ScaleRef raw block.val := by
    rw [hdv, usize52_val, hf16]; rfl
  have hinv048 : fillInv blk 0 (Array.repeat 48#usize 0#u8) 0 :=
    ⟨by norm_num, fun j hj => absurd hj (by omega)⟩
  have hinv04 : fillInv blk 48 (Array.repeat 4#usize 0#u8) 0 :=
    ⟨by norm_num, fun j hj => absurd hj (by omega)⟩
  obtain ⟨p, hqs_eq, hqs_inv⟩ := WP.spec_imp_exists
    (tq1_0Qs_loop_spec blk (Array.repeat 48#usize 0#u8) 0#usize (by rw [hlenblk]; omega) hinv048)
  obtain ⟨p2, hqh_eq, hqh_inv⟩ := WP.spec_imp_exists
    (tq1_0Qh_loop_spec blk (Array.repeat 4#usize 0#u8) 0#usize (by rw [hlenblk]; omega) hinv04)
  refine ⟨d, p, p2, ?_, hdv', ?_, ?_⟩
  · rw [dequant.lowbit.tq1_0_block, quant_TQ1_0_BLOCK_BYTES_eq]
    simp only [hblk, hd_eq, hqs_eq, hqh_eq, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch]
  · intro j hj
    have h := hqs_inv.2 j hj
    rw [h, hslice', show 0 + j = j by omega]
    have hj54 : j < (block.val * 54 + 54) - block.val * 54 := by
      rw [Nat.add_comm (block.val * 54) 54, Nat.add_sub_cancel]; omega
    rw [getD_slice raw.val (block.val * 54) (block.val * 54 + 54) j hlen hj54]
  · intro h hh
    have hh' := hqh_inv.2 h hh
    rw [hh', hslice']
    have hh54 : 48 + h < (block.val * 54 + 54) - block.val * 54 := by
      rw [Nat.add_comm (block.val * 54) 54, Nat.add_sub_cancel]; omega
    rw [getD_slice raw.val (block.val * 54) (block.val * 54 + 54) (48 + h) hlen hh54]
    rw [show block.val * 54 + (48 + h) = block.val * 54 + 48 + h by omega]

/-! ## `ptq1_0_block_loop0` (24 `qs` bytes) -/

theorem ptq1_0Qs_body_step (blk : Slice U8) (qs : Array U8 24#usize) (k : Usize)
    (hblk : 24 ≤ blk.val.length) (hk : k.val < 24) (hinv : fillInv blk 0 qs k.val) :
    ∃ (qs2 : Array U8 24#usize) (k2 : Usize),
      dequant.lowbit.ptq1_0_block_loop0.body blk qs k = ok (ControlFlow.cont (qs2, k2)) ∧
      fillInv blk 0 qs2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨-, hrec⟩ := hinv
  obtain ⟨i, hidx, hiv⟩ := slice_index_getD blk k (lt_of_lt_of_le hk hblk)
  obtain ⟨a, hupd, hav⟩ := WP.spec_imp_exists
    (Array.update_spec qs k i (by show k.val < qs.val.length; rw [Array.property qs]; exact hk))
  obtain ⟨k1, hadd, hk1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk1val : k1.val = k.val + 1 := by rw [hk1v]; simp
  refine ⟨a, k1, ?_, ?_, hk1val⟩
  · unfold dequant.lowbit.ptq1_0_block_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hk)]
    simp only [hidx, hupd, hadd, bind_tc_ok]
  · refine ⟨by show k.val + 1 ≤ 24; omega, ?_⟩
    intro j hj
    rw [hav, Array.set_val_eq, getD_set]
    by_cases hjk : j = k.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [Array.property qs]; exact hk⟩]
      simpa using hiv
    · rw [if_neg (fun hh => hjk hh.1)]
      exact hrec j (by omega)

theorem ptq1_0Qs_loop_spec (blk : Slice U8) (qs0 : Array U8 24#usize) (k : Usize)
    (hblk : 24 ≤ blk.val.length) (hinv : fillInv blk 0 qs0 k.val) :
    WP.spec (dequant.lowbit.ptq1_0_block_loop0 blk qs0 k) (fun qs => fillInv blk 0 qs 24) := by
  unfold dequant.lowbit.ptq1_0_block_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 24#usize × Usize => 24 - s.2.val)
    (inv := fun s => fillInv blk 0 s.1 s.2.val)
    (post := fun qs => fillInv blk 0 qs 24)
  · rintro ⟨qs, kk⟩ hinv'
    dsimp only
    by_cases hlt : kk.val < 24
    · obtain ⟨qs2, k2, hbody, hinv2, hk2⟩ := ptq1_0Qs_body_step blk qs kk hblk hlt hinv'
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨by rw [hk2]; exact hinv2, by rw [hk2]; omega⟩
    · have hnot : ¬ (kk < 24#usize) := fun hc => hlt ((UScalar.lt_equiv kk 24#usize).1 hc)
      have hbody : dequant.lowbit.ptq1_0_block_loop0.body blk qs kk = ok (ControlFlow.done qs) := by
        unfold dequant.lowbit.ptq1_0_block_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hk24 : kk.val = 24 := by have := hinv'.1; simp at this; omega
      simpa [hk24] using hinv'
  · exact hinv

/-! ## `ptq1_0_block_loop1` (2 `qh` bytes at 24) -/

theorem ptq1_0Qh_body_step (blk : Slice U8) (qh : Array U8 2#usize) (h : Usize)
    (hblk : 26 ≤ blk.val.length) (hh : h.val < 2) (hinv : fillInv blk 24 qh h.val) :
    ∃ (qh2 : Array U8 2#usize) (h2 : Usize),
      dequant.lowbit.ptq1_0_block_loop1.body blk qh h = ok (ControlFlow.cont (qh2, h2)) ∧
      fillInv blk 24 qh2 (h.val + 1) ∧ h2.val = h.val + 1 := by
  obtain ⟨-, hrec⟩ := hinv
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 24#usize) (y := h) (by scalar_tac))
  have hiv' : i.val = 24 + h.val := by rw [hiv]; simp
  obtain ⟨i1, hidx, hi1v⟩ := slice_index_getD blk i (by
    rw [hiv']; exact lt_of_lt_of_le (by omega : 24 + h.val < 26) hblk)
  obtain ⟨a, hupd, hav⟩ := WP.spec_imp_exists
    (Array.update_spec qh h i1 (by show h.val < qh.val.length; rw [Array.property qh]; exact hh))
  obtain ⟨h1, hadd, hh1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := h) (y := 1#usize) (by scalar_tac))
  have hh1val : h1.val = h.val + 1 := by rw [hh1v]; simp
  refine ⟨a, h1, ?_, ?_, hh1val⟩
  · unfold dequant.lowbit.ptq1_0_block_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hh)]
    simp only [hi, hidx, hupd, hadd, bind_tc_ok]
  · refine ⟨by show h.val + 1 ≤ 2; omega, ?_⟩
    intro j hj
    rw [hav, Array.set_val_eq, getD_set]
    by_cases hjk : j = h.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [Array.property qh]; exact hh⟩]
      rw [hiv'] at hi1v
      exact hi1v
    · rw [if_neg (fun hh => hjk hh.1)]
      exact hrec j (by omega)

theorem ptq1_0Qh_loop_spec (blk : Slice U8) (qh0 : Array U8 2#usize) (h : Usize)
    (hblk : 26 ≤ blk.val.length) (hinv : fillInv blk 24 qh0 h.val) :
    WP.spec (dequant.lowbit.ptq1_0_block_loop1 blk qh0 h) (fun qh => fillInv blk 24 qh 2) := by
  unfold dequant.lowbit.ptq1_0_block_loop1
  apply loop.spec_decr_nat
    (measure := fun s : Array U8 2#usize × Usize => 2 - s.2.val)
    (inv := fun s => fillInv blk 24 s.1 s.2.val)
    (post := fun qh => fillInv blk 24 qh 2)
  · rintro ⟨qh, hh⟩ hinv'
    dsimp only
    by_cases hlt : hh.val < 2
    · obtain ⟨qh2, h2, hbody, hinv2, hh2⟩ := ptq1_0Qh_body_step blk qh hh hblk hlt hinv'
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨by rw [hh2]; exact hinv2, by rw [hh2]; omega⟩
    · have hnot : ¬ (hh < 2#usize) := fun hc => hlt ((UScalar.lt_equiv hh 2#usize).1 hc)
      have hbody : dequant.lowbit.ptq1_0_block_loop1.body blk qh hh = ok (ControlFlow.done qh) := by
        unfold dequant.lowbit.ptq1_0_block_loop1.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hh2 : hh.val = 2 := by have := hinv'.1; simp at this; omega
      simpa [hh2] using hinv'
  · exact hinv

/-! ## `ptq1_0_block` -/

/-- **`PTQ1_0` block law.** On a whole 28-byte block with finite `d`, `ptq1_0_block` returns the
exact Q32.32 scale, the 24 `qs` bytes at `28·block + j` and the 2 `qh` bytes at `28·block + 24 + h`. -/
theorem ptq1_0_block_ok (raw : Slice U8) (block : Usize)
    (hlen : block.val * 28 + 28 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw (block.val * 28 + 26)) ≠ 31) :
    ∃ (d : I64) (qs : Array U8 24#usize) (qh : Array U8 2#usize),
      dequant.lowbit.ptq1_0_block raw block = ok (.Ok (d, qs, qh)) ∧
      d.val = ptq1_0ScaleRef raw block.val ∧
      (∀ j, j < 24 → (qs.val.getD j 0#u8).val = (raw.val.getD (block.val * 28 + j) 0#u8).val) ∧
      (∀ h, h < 2 →
        (qh.val.getD h 0#u8).val = (raw.val.getD (block.val * 28 + 24 + h) 0#u8).val) := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok raw block 28#usize (by simpa using hlen)
  have hslice' : blk.val = raw.val.slice (block.val * 28) (block.val * 28 + 28) := by
    rw [hslice, usize28_val]
  have hlenblk : blk.val.length = 28 := by
    rw [hslice', List.slice, List.length_take, List.length_drop]
    omega
  have hf16 : f16LeHalf blk 26 = f16LeHalf raw (block.val * 28 + 26) :=
    f16LeHalf_block28 raw (block.val * 28) 26 blk hslice' (by omega) (by norm_num)
  obtain ⟨d, hd_eq, hdv⟩ :=
    read_scale_ok blk 26#usize (by simp [hlenblk])
      (by rw [usize26_val, hf16]; exact hd)
  have hdv' : d.val = ptq1_0ScaleRef raw block.val := by
    rw [hdv, usize26_val, hf16]; rfl
  have hinv024 : fillInv blk 0 (Array.repeat 24#usize 0#u8) 0 :=
    ⟨by norm_num, fun j hj => absurd hj (by omega)⟩
  have hinv02 : fillInv blk 24 (Array.repeat 2#usize 0#u8) 0 :=
    ⟨by norm_num, fun j hj => absurd hj (by omega)⟩
  obtain ⟨p, hqs_eq, hqs_inv⟩ := WP.spec_imp_exists
    (ptq1_0Qs_loop_spec blk (Array.repeat 24#usize 0#u8) 0#usize (by rw [hlenblk]; omega) hinv024)
  obtain ⟨p2, hqh_eq, hqh_inv⟩ := WP.spec_imp_exists
    (ptq1_0Qh_loop_spec blk (Array.repeat 2#usize 0#u8) 0#usize (by rw [hlenblk]; omega) hinv02)
  refine ⟨d, p, p2, ?_, hdv', ?_, ?_⟩
  · rw [dequant.lowbit.ptq1_0_block, quant_PTQ1_0_BLOCK_BYTES_eq]
    simp only [hblk, hd_eq, hqs_eq, hqh_eq, bind_tc_ok,
      core.result.Result.Insts.CoreOpsTry.branch]
  · intro j hj
    have h := hqs_inv.2 j hj
    rw [h, hslice', show 0 + j = j by omega]
    have hj28 : j < (block.val * 28 + 28) - block.val * 28 := by
      rw [Nat.add_comm (block.val * 28) 28, Nat.add_sub_cancel]; omega
    rw [getD_slice raw.val (block.val * 28) (block.val * 28 + 28) j hlen hj28]
  · intro h hh
    have hh' := hqh_inv.2 h hh
    rw [hh', hslice']
    have hh28 : 24 + h < (block.val * 28 + 28) - block.val * 28 := by
      rw [Nat.add_comm (block.val * 28) 28, Nat.add_sub_cancel]; omega
    rw [getD_slice raw.val (block.val * 28) (block.val * 28 + 28) (24 + h) hlen hh28]
    rw [show block.val * 28 + (24 + h) = block.val * 28 + 24 + h by omega]

/-! ## Staged-trit order maps, weights and row values

The staged walk of the spec emits, for `TQ1_0`, planes `0…4` of `qs` bytes `0…31` then `32…47`, then
planes `0…3` of the `qh` tail; for `PTQ1_0`, planes of `qs` bytes `0…15` then `16…23`, then the tail.
`tq1_0Byte`/`tq1_0Plane` (resp. `ptq1_0Byte`/`ptq1_0Plane`) are the out-index → (block byte, plane)
map of that walk. -/

/-- Byte index within the 52-byte `qs`/`qh` region of TQ1_0 block, by output index. -/
def tq1_0Byte (r : Nat) : Nat :=
  if r < 160 then r % 32 else if r < 240 then 32 + (r - 160) % 16 else 48 + (r - 240) % 4

/-- Trit plane of TQ1_0 output index `r`. -/
def tq1_0Plane (r : Nat) : Nat :=
  if r < 160 then r / 32 else if r < 240 then (r - 160) / 16 else (r - 240) / 4

/-- Byte index within the 26-byte `qs`/`qh` region of PTQ1_0 block, by output index. -/
def ptq1_0Byte (r : Nat) : Nat :=
  if r < 80 then r % 16 else if r < 120 then 16 + (r - 80) % 8 else 24 + (r - 120) % 2

/-- Trit plane of PTQ1_0 output index `r`. -/
def ptq1_0Plane (r : Nat) : Nat :=
  if r < 80 then r / 16 else if r < 120 then (r - 80) / 8 else (r - 120) / 2

lemma tq1_0Byte_lane32 (plane lane : Nat) (hp : plane < 5) (hl : lane < 32) :
    tq1_0Byte (plane * 32 + lane) = lane := by
  simp only [tq1_0Byte]
  rw [if_pos (by omega)]
  omega

lemma tq1_0Plane_lane32 (plane lane : Nat) (hp : plane < 5) (hl : lane < 32) :
    tq1_0Plane (plane * 32 + lane) = plane := by
  simp only [tq1_0Plane]
  rw [if_pos (by omega)]
  omega

lemma tq1_0Byte_lane16 (plane lane : Nat) (hp : plane < 5) (hl : lane < 16) :
    tq1_0Byte (160 + plane * 16 + lane) = 32 + lane := by
  simp only [tq1_0Byte]
  rw [if_neg (by omega), if_pos (by omega)]
  omega

lemma tq1_0Plane_lane16 (plane lane : Nat) (hp : plane < 5) (hl : lane < 16) :
    tq1_0Plane (160 + plane * 16 + lane) = plane := by
  simp only [tq1_0Plane]
  rw [if_neg (by omega), if_pos (by omega)]
  omega

lemma tq1_0Byte_tail (plane lane : Nat) (hp : plane < 4) (hl : lane < 4) :
    tq1_0Byte (240 + plane * 4 + lane) = 48 + lane := by
  simp only [tq1_0Byte]
  rw [if_neg (by omega), if_neg (by omega)]
  omega

lemma tq1_0Plane_tail (plane lane : Nat) (hp : plane < 4) (hl : lane < 4) :
    tq1_0Plane (240 + plane * 4 + lane) = plane := by
  simp only [tq1_0Plane]
  rw [if_neg (by omega), if_neg (by omega)]
  omega

lemma ptq1_0Byte_lane16 (plane lane : Nat) (hp : plane < 5) (hl : lane < 16) :
    ptq1_0Byte (plane * 16 + lane) = lane := by
  simp only [ptq1_0Byte]
  rw [if_pos (by omega)]
  omega

lemma ptq1_0Plane_lane16 (plane lane : Nat) (hp : plane < 5) (hl : lane < 16) :
    ptq1_0Plane (plane * 16 + lane) = plane := by
  simp only [ptq1_0Plane]
  rw [if_pos (by omega)]
  omega

lemma ptq1_0Byte_lane8 (plane lane : Nat) (hp : plane < 5) (hl : lane < 8) :
    ptq1_0Byte (80 + plane * 8 + lane) = 16 + lane := by
  simp only [ptq1_0Byte]
  rw [if_neg (by omega), if_pos (by omega)]
  omega

lemma ptq1_0Plane_lane8 (plane lane : Nat) (hp : plane < 5) (hl : lane < 8) :
    ptq1_0Plane (80 + plane * 8 + lane) = plane := by
  simp only [ptq1_0Plane]
  rw [if_neg (by omega), if_pos (by omega)]
  omega

lemma ptq1_0Byte_tail (plane lane : Nat) (hp : plane < 4) (hl : lane < 2) :
    ptq1_0Byte (120 + plane * 2 + lane) = 24 + lane := by
  simp only [ptq1_0Byte]
  rw [if_neg (by omega), if_neg (by omega)]
  omega

lemma ptq1_0Plane_tail (plane lane : Nat) (hp : plane < 4) (hl : lane < 2) :
    ptq1_0Plane (120 + plane * 2 + lane) = plane := by
  simp only [ptq1_0Plane]
  rw [if_neg (by omega), if_neg (by omega)]
  omega

/-- The exact weight at within-block output index `r` of TQ1_0 block `b`. -/
def tq1_0Weight (raw : Slice U8) (b r : Nat) : ℤ :=
  tq1_0ScaleRef raw b *
    (tritRef (raw.val.getD (b * 54 + tq1_0Byte r) 0#u8) (tq1_0Plane r) - 1)

/-- The 256 exact weights of TQ1_0 block `b`, in staged order. -/
def tq1_0BlockVals (raw : Slice U8) (b : Nat) : List ℤ :=
  (List.range 256).map (tq1_0Weight raw b)

/-- The first `n` exact TQ1_0 row weights. -/
def tq1_0RowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (fun i => tq1_0Weight raw (i / 256) (i % 256))

/-- The exact weight at within-block output index `r` of PTQ1_0 block `b`. -/
def ptq1_0Weight (raw : Slice U8) (b r : Nat) : ℤ :=
  ptq1_0ScaleRef raw b *
    (tritRef (raw.val.getD (b * 28 + ptq1_0Byte r) 0#u8) (ptq1_0Plane r) - 1)

/-- The 128 exact weights of PTQ1_0 block `b`, in staged order. -/
def ptq1_0BlockVals (raw : Slice U8) (b : Nat) : List ℤ :=
  (List.range 128).map (ptq1_0Weight raw b)

/-- The first `n` exact PTQ1_0 row weights. -/
def ptq1_0RowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (fun i => ptq1_0Weight raw (i / 128) (i % 128))

/-- `dequant_row_fixed` at `ggml_type = 34` (`GGML_TYPE_TQ1_0`) is `dequant_tq1_0_row`. -/
theorem dequant_row_fixed_tq1_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 34#u32 raw n out = dequant.lowbit.dequant_tq1_0_row raw n out := rfl

/-- `dequant_row_fixed` at `ggml_type = 143` (`GGML_TYPE_PTQ1_0`) is `dequant_ptq1_0_row`. -/
theorem dequant_row_fixed_ptq1_0 (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 143#u32 raw n out = dequant.lowbit.dequant_ptq1_0_row raw n out :=
  rfl

/-! ## Innermost lane loop of `dequant_tq1_0_row` (48-byte `qs` walk)

The block body's innermost loop emits `scale·(trit(qs[base+lane], plane) − 1)` for `lane < width`
and appends while `pos < take`. `tq1_0EmitInv raw b take start out pos` records that the output is
the block's initial output `start` followed by the first `min pos take` entries of the staged stream
`tq1_0BlockVals`. -/

/-- `tritRef` depends only on the byte value. -/
lemma tritRef_congr {a b : U8} (h : a.val = b.val) (n : Nat) : tritRef a n = tritRef b n := by
  unfold tritRef; rw [h]

/-- A recovered trit is at most `2`. -/
lemma tritRef_le_two (b : U8) (n : Nat) : tritRef b n ≤ 2 := by
  unfold tritRef
  have h : (b.val * 3 ^ n) % 256 ≤ 255 := Nat.le_of_lt_succ (Nat.mod_lt _ (by norm_num))
  have : ((b.val * 3 ^ n) % 256 * 3) / 256 ≤ 2 := by omega
  omega

/-- A recovered trit is nonnegative. -/
lemma tritRef_nonneg (b : U8) (n : Nat) : 0 ≤ tritRef b n := by
  unfold tritRef; positivity

/-- The staged block stream has length 256. -/
lemma tq1_0BlockVals_length (raw : Slice U8) (b : Nat) :
    (tq1_0BlockVals raw b).length = 256 := by
  simp [tq1_0BlockVals]

/-- Entry `r < 256` of the staged block stream. -/
lemma tq1_0BlockVals_getD (raw : Slice U8) (b r : Nat) (hr : r < 256) :
    (tq1_0BlockVals raw b).getD r 0 = tq1_0Weight raw b r := by
  unfold tq1_0BlockVals
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[r]? = some r := by
    rw [List.getElem?_eq_getElem (by simpa using hr)]; simp
  rw [h]; rfl

/-- `dequantVals` of an append. -/
lemma dequantVals_append (l m : List I64) :
    dequantVals (l ++ m) = dequantVals l ++ dequantVals m := by
  unfold dequantVals; rw [List.map_append]

/-- Emission invariant: the output is the block's initial output `start` followed by the first
`min pos take` entries of the staged stream, and the success flag is set. -/
def tq1_0EmitInv (raw : Slice U8) (b : Nat) (take : Usize) (start out : alloc.vec.Vec I64)
    (pos : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧
  dequantVals out.val = dequantVals start.val ++
    (tq1_0BlockVals raw b).take (Nat.min pos.val take.val)

/-- Lane-loop invariant: `tq1_0EmitInv` and `pos = pos0 + lane`. -/
def tq1_0LaneInv (raw : Slice U8) (b : Nat) (take : Usize) (pos0 : Nat) (width : Usize)
    (start out : alloc.vec.Vec I64) (pos : Usize) (lane : Usize)
    (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  tq1_0EmitInv raw b take start out pos ok1 err ∧ pos.val = pos0 + lane.val ∧
    lane.val ≤ width.val

/-- A generic `Array` index is its `getD`. -/
lemma arr_getD_val_eq {n : Usize} (a : Array U8 n) (i : Usize) (h : i.val < a.val.length) :
    (a.val[i.val]'h).val = (a.val.getD i.val 0#u8).val :=
  congrArg (fun u : U8 => u.val) (list_getD_getElem a.val i.val h 0#u8)

/-- **Innermost lane-loop body step.** On the success path, one lane emits the `pos`-th staged value
(through the `trit`/`mul` chain), pushes it while `pos < take`, and advances both counters. -/
theorem tq1_0Lane_body_step (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 48#usize) (take : Usize) (base width plane pos0 : Usize)
    (start out : alloc.vec.Vec I64) (pos lane : Usize)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hplane5 : plane.val < 5)
    (hplane : tq1_0Plane pos.val = plane.val)
    (hbyte : tq1_0Byte pos.val = base.val + lane.val)
    (hlane : lane.val < width.val)
    (hbl : base.val + lane.val < 48)
    (hpos256 : pos.val < 256)
    (hmul : I64.min ≤
        scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out.val.length < Usize.max)
    (hinv : tq1_0LaneInv raw b take pos0.val width start out pos lane true none) :
    ∃ (out2 : alloc.vec.Vec I64) (pos2 lane2 : Usize),
      dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0.body scale qs take base
          width plane out true none pos lane = ok (ControlFlow.cont (out2, true, none, pos2, lane2)) ∧
      tq1_0LaneInv raw b take pos0.val width start out2 pos2 lane2 true none ∧
      pos2.val = pos.val + 1 ∧ lane2.val = lane.val + 1 := by
  obtain ⟨hemit, hposrel, hlanele⟩ := hinv
  obtain ⟨hok, herr, hrec⟩ := hemit
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := base) (y := lane) (by scalar_tac))
  have hbnd : i.val < qs.val.length := by rw [hiv, Array.property qs]; exact hbl
  obtain ⟨i1, hi1, hi1eq⟩ := WP.spec_imp_exists (Array.index_usize_spec qs i hbnd)
  have hi1val_i : i1.val = (qs.val.getD i.val 0#u8).val := by
    rw [hi1eq]; exact arr_getD_val_eq qs i hbnd
  have hi1val : i1.val = (qs.val.getD (base.val + lane.val) 0#u8).val := by
    rw [hi1val_i, hiv]
  obtain ⟨i2, htrit, hi2v⟩ := trit_spec i1 plane (by omega)
  have hi2val : i2.val = tritRef i1 plane.val := hi2v
  obtain ⟨i3, hsub, hi3v⟩ := WP.spec_imp_exists
    (I64.sub_spec (x := i2) (y := 1#i64)
      (by rw [hi2val, I64.min_eq, show (1#i64).val = 1 from rfl]
          have := tritRef_nonneg i1 plane.val; omega)
      (by rw [hi2val, I64.max_eq, show (1#i64).val = 1 from rfl]
          have := tritRef_le_two i1 plane.val; omega))
  have hi3val : i3.val = tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1 := by
    rw [hi3v, hi2val, tritRef_congr hi1val plane.val]; norm_num
  obtain ⟨v, hmulv, hvv⟩ := q6k_mul_ok scale i3 (by rw [hi3val]; exact hmul.1)
    (by rw [hi3val]; exact hmul.2)
  have hvval : v.val = (tq1_0BlockVals raw b)[pos.val]'(by
      rw [tq1_0BlockVals_length]; exact hpos256) := by
    have hcong := tritRef_congr (hqs (base.val + lane.val) hbl) plane.val
    rw [hvv, hi3val, hsc, hcong, ← hbyte]
    rw [list_getD_getElem (tq1_0BlockVals raw b) pos.val
      (by rw [tq1_0BlockVals_length]; exact hpos256) (0 : ℤ), tq1_0BlockVals_getD raw b pos.val hpos256]
    unfold tq1_0Weight
    rw [hplane]
  by_cases hlt : pos.val < take.val
  · obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists
      (alloc.vec.Vec.push_spec out v hcap)
    obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
    refine ⟨out2, pos2, lane2, ?_, ?_, hpos2, hlane2⟩
    · unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), hok, if_pos rfl]
      simp only [hi, hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_pos ((UScalar.lt_equiv pos take).2 hlt)]
      simp only [hpush, haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel]; omega, by rw [hlane2]; omega⟩
      have hmin1 : (pos.val + 1).min take.val = pos.val + 1 :=
        Nat.min_eq_left (by omega : pos.val + 1 ≤ take.val)
      have hmin2 : pos.val.min take.val = pos.val := Nat.min_eq_left (le_of_lt hlt)
      rw [hpushv, dequantVals_append, hrec]
      simp only [dequantVals, List.map_cons, List.map_nil, hvval]
      rw [hpos2, hmin1, hmin2,
        List.take_succ_eq_append_getElem (by rw [tq1_0BlockVals_length]; exact hpos256),
        List.append_assoc]
  · obtain ⟨pos2, haddp, hpos2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
    obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
    have hpos2 : pos2.val = pos.val + 1 := by rw [hpos2v]; simp
    have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
    refine ⟨out, pos2, lane2, ?_, ?_, hpos2, hlane2⟩
    · unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0.body
      rw [if_pos (by rw [UScalar.lt_equiv]; exact hlane), hok, if_pos rfl]
      simp only [hi, hi1, htrit, hsub, hmulv, bind_tc_ok]
      rw [if_neg (fun hc => hlt ((UScalar.lt_equiv pos take).1 hc))]
      simp only [haddp, haddl, bind_tc_ok]
      rfl
    · refine ⟨⟨rfl, rfl, ?_⟩, by rw [hpos2, hposrel]; omega, by rw [hlane2]; omega⟩
      have h1 : (pos.val + 1).min take.val = take.val :=
        Nat.min_eq_right (by omega : take.val ≤ pos.val + 1)
      have h2 : pos.val.min take.val = take.val :=
        Nat.min_eq_right (by omega : take.val ≤ pos.val)
      rw [hrec, hpos2, h1, h2]

/-- **Innermost lane loop.** From `lane = 0` at word offset `pos0` the loop emits `width` staged
values (truncated by `take`), leaves `ok = true`, and advances `pos` by `width`. -/
theorem tq1_0Lane_loop_spec (raw : Slice U8) (b : Nat) (scale : I64)
    (qs : Array U8 48#usize) (take : Usize) (base width plane pos0 : Usize)
    (start out0 : alloc.vec.Vec I64)
    (hsc : scale.val = tq1_0ScaleRef raw b)
    (hqs : ∀ j, j < 48 → (qs.val.getD j 0#u8).val = (raw.val.getD (b * 54 + j) 0#u8).val)
    (hplane5 : plane.val < 5)
    (hrun : ∀ lane : Usize, lane.val < width.val →
      tq1_0Byte (pos0.val + lane.val) = base.val + lane.val ∧
      tq1_0Plane (pos0.val + lane.val) = plane.val)
    (hpos256 : pos0.val + width.val ≤ 256)
    (hbase : base.val + width.val ≤ 48)
    (hmul : ∀ lane : Usize, lane.val < width.val →
      I64.min ≤ scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ∧
      scale.val * (tritRef (qs.val.getD (base.val + lane.val) 0#u8) plane.val - 1) ≤ I64.max)
    (hcap : out0.val.length + 256 < Usize.max)
    (hinv0 : tq1_0EmitInv raw b take start out0 pos0 true none) :
    WP.spec (dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0 out0 true none scale
        qs take pos0 base width plane 0#usize)
      (fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
        p = (out, true, none, pos') ∧ pos'.val = pos0.val + width.val ∧
        tq1_0EmitInv raw b take start out pos' true none) := by
  have hinit : tq1_0LaneInv raw b take pos0.val width start out0 pos0 0#usize true none :=
    ⟨hinv0, by simp, by simp⟩
  unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Bool × Option error.EngineError × Usize × Usize =>
      width.val - s.2.2.2.2.val)
    (inv := fun s => tq1_0LaneInv raw b take pos0.val width start s.1 s.2.2.2.1 s.2.2.2.2 s.2.1 s.2.2.1)
    (post := fun p => ∃ (out : alloc.vec.Vec I64) (pos' : Usize),
      p = (out, true, none, pos') ∧ pos'.val = pos0.val + width.val ∧
      tq1_0EmitInv raw b take start out pos' true none)
  · rintro ⟨out, ok1, err, pos, lane⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨hemit, hposrel, hlanele⟩ := hinv'
    obtain ⟨hok, herr, hrec⟩ := hemit
    rw [hok, herr]
    by_cases hlt : lane.val < width.val
    · have hpos256' : pos.val < 256 := by
        have hpx : pos.val = pos0.val + lane.val := hposrel
        omega
      have houtlen : out.val.length ≤ start.val.length + 256 := by
        have h1 := congrArg List.length hrec
        simp only [dequantVals, List.length_append, List.length_map] at h1
        rw [h1]
        have h2 := (List.take_sublist (l := tq1_0BlockVals raw b)
          (i := Nat.min pos.val take.val)).length_le
        simp only [tq1_0BlockVals_length] at h2
        omega
      have hstart : start.val.length ≤ out0.val.length := by
        have h1 := congrArg List.length hinv0.2.2
        simp only [dequantVals, List.length_append, List.length_map] at h1
        rw [h1]
        have h2 := (List.take_sublist (l := tq1_0BlockVals raw b)
          (i := Nat.min pos0.val take.val)).length_le
        simp only [tq1_0BlockVals_length] at h2
        omega
      have hcap' : out.val.length < Usize.max := by
        have := houtlen; omega
      obtain ⟨out2, pos2, lane2, hbody, hinv2, hpos2, hlane2⟩ :=
        tq1_0Lane_body_step raw b scale qs take base width plane pos0 start out pos lane
          hsc hqs hplane5 (by rw [hposrel]; exact (hrun lane hlt).2)
          (by rw [hposrel]; exact (hrun lane hlt).1) hlt (by omega) hpos256'
          (hmul lane hlt) hcap' ⟨⟨rfl, rfl, hrec⟩, hposrel, hlanele⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show width.val - lane2.val < width.val - lane.val
      omega
    · have hnot : ¬ (lane < width) := fun hc => hlt ((UScalar.lt_equiv lane width).1 hc)
      have hbody : dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0.body scale qs
          take base width plane out true none pos lane =
          ok (ControlFlow.done (out, true, none, pos)) := by
        unfold dequant.lowbit.dequant_tq1_0_row_loop0_loop0_loop0_loop0_loop0.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      have hlw : lane.val = width.val := by omega
      refine ⟨out, pos, rfl, by rw [hposrel, hlw], rfl, rfl, hrec⟩
  · exact hinit

end

end apeiron_word
