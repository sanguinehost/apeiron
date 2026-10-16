/-
Refinement of the extracted two-stage bit-plane argmax kernels (`src/engine/kern/bitplane.rs`,
`BitplaneHead`) to stated Lean references. The extracted bodies are `apeiron_word.kern.bitplane.*`
in `proof/ApeironWord.lean`.

Covered here (integer / index / layout leaves): the `HALF` / `STAGE1_BLOCK_BYTES` /
`STAGE2_BLOCK_BYTES` layout constants, the `f16` scale decode `decode_f16`, the portable nibble
dots `high_dot` / `low_dot`, and the code window `block`.

Boundaries (not covered, stated): the AVX2 bodies `kern.bitplane.x86.*` are extracted axioms; the
`avx = true` arm of `high_dot` / `low_dot` / `row_interval` / `row_exact` routes through them, so
the theorems here are stated at `avx = false` (the portable path). `avx2()` is opaque feature
detection. `round_logit` is open: its `value.round_shift_32` carries a checked `i128` add, so it is
a partial function with a refusal branch not yet stated here. The pooled `argmax` /
`stage1_chunk` / `stage2_chunk` and the `from_q8_0` builder are open.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.Dequant.Refinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Layout constants

`HALF = Q8_0_QK / 2 = 16`; `STAGE1_BLOCK_BYTES = 2 + HALF = 18` (2-byte f16 scale + 16-byte high
plane); `STAGE2_BLOCK_BYTES = HALF = 16` (low plane). -/

theorem word_HALF_eq : kern.bitplane.HALF = ok (16#usize) := by
  have h : kern.bitplane.HALF ⦃ z => z = (16#usize) ⦄ := by
    unfold kern.bitplane.HALF quant.Q8_0_QK
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

theorem word_STAGE1_BLOCK_BYTES_eq : kern.bitplane.STAGE1_BLOCK_BYTES = ok (18#usize) := by
  have h : kern.bitplane.STAGE1_BLOCK_BYTES ⦃ z => z = (18#usize) ⦄ := by
    unfold kern.bitplane.STAGE1_BLOCK_BYTES
    rw [word_HALF_eq]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

theorem word_STAGE2_BLOCK_BYTES_eq : kern.bitplane.STAGE2_BLOCK_BYTES = ok (16#usize) := by
  unfold kern.bitplane.STAGE2_BLOCK_BYTES
  exact word_HALF_eq

/-! ## `decode_f16`

Exact Q32.32 decode of a finite (and, as it happens, every) binary16 bit pattern; the reference is
`f16Value` (`Numeric.Dequant.Refinement`), the exact signed Q32.32 integer. -/

theorem word_decode_f16_ok (h : U16) :
    ∃ v : I64, kern.bitplane.decode_f16 h = ok v ∧ v.val = f16Value h := by
  obtain ⟨sign, hs, hsv⟩ := f16_shr15 h
  obtain ⟨i, hi, hiv, -⟩ :=
    WP.spec_imp_exists (U16.ShiftRight_IScalar_spec h 10#i32 (by simp) (by simp))
  have hiv' : i.val = h.val >>> 10 := by rw [hiv]; rfl
  have hE := f16_exp_eq h i hiv'
  have hM := f16_man_eq h
  have hEl := f16_exp_lt h
  have hMl := f16_man_lt h
  have hmcast : (UScalar.hcast .I64 (h &&& 1023#u16)).val = (f16Man h : ℤ) := by
    rw [UScalar.hcast_val_eq, hM]
    apply Int.bmod_eq_of_le
    · simp only [IScalarTy.I64_numBits_eq]; norm_num
    · simp only [IScalarTy.I64_numBits_eq]; norm_num; omega
  rw [kern.bitplane.decode_f16]
  simp only [hs, hi, lift, bind_tc_ok, I64.Insts.CoreConvertFromU16.from]
  by_cases he0 : (i &&& 31#u16) = 0#u16
  · have hE0 : f16Exp h = 0 := by rw [← hE, he0]; simp
    obtain ⟨mag, hmag, hmagv, -⟩ := WP.spec_imp_exists
      (I64.ShiftLeft_IScalar_spec (UScalar.hcast .I64 (h &&& 1023#u16)) 8#i32
        (by simp) (by simp))
    have hmagv' : mag.val = (f16Man h : ℤ) * 2 ^ 8 := by
      have hsz : I64.size = 2 ^ 64 := by simp [I64.size, I64.numBits]
      have h8 : (8#i32).toNat = 8 := by simp
      simp only [hmagv, hmcast, Int.shiftLeft_eq, hsz, h8]
      apply Int.bmod_eq_of_le <;> norm_num <;> omega
    simp only [if_pos he0, hmag, bind_tc_ok]
    by_cases hs1 : sign = 1#u16
    · have hS : f16Sign h = 1 := by rw [← hsv, hs1]; simp
      obtain ⟨mag1, hneg, hnegv⟩ := WP.spec_imp_exists
        (HNeg.hNeg.step mag (by
          intro hc
          rw [hmagv', IScalar.min_IScalarTy_I64_eq, I64.min_eq] at hc
          omega))
      simp only [if_pos hs1, hneg]
      refine ⟨mag1, rfl, ?_⟩
      rw [hnegv, hmagv', f16Value, if_pos hS, f16Mag, if_pos hE0]
    · have hS : f16Sign h ≠ 1 := by
        intro hc; apply hs1; apply UScalar.eq_of_val_eq; rw [hsv, hc]; simp
      simp only [if_neg hs1]
      refine ⟨mag, rfl, ?_⟩
      rw [hmagv', f16Value, if_neg hS, f16Mag, if_pos hE0]
  · have hE0 : f16Exp h ≠ 0 := by
      intro hc; apply he0; apply UScalar.eq_of_val_eq; rw [hE, hc]; simp
    simp only [if_neg he0]
    obtain ⟨sig, hsig, hsigv⟩ := WP.spec_imp_exists
      (I64.add_spec (x := UScalar.hcast .I64 (h &&& 1023#u16)) (y := 1024#i64)
        (by rw [hmcast, I64.min_eq]; simp; omega) (by rw [hmcast, I64.max_eq]; simp; omega))
    have hsigv' : sig.val = ((1024 + f16Man h : ℕ) : ℤ) := by
      rw [hsigv, hmcast]; push_cast; simp; ring
    have he2 : (core.convert.num.FromU32U16.from (i &&& 31#u16)).val = f16Exp h := by
      rw [core.convert.num.FromU32U16.from_val_eq, hE]
    obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists
      (U32.add_spec (x := core.convert.num.FromU32U16.from (i &&& 31#u16)) (y := 7#u32)
        (by rw [he2, U32.max_eq]; simp; omega))
    have hi3v' : i3.val = f16Exp h + 7 := by rw [hi3v, he2]; simp
    obtain ⟨sh, hsh, hshv, -⟩ := WP.spec_imp_exists
      (I64.ShiftLeft_spec sig i3 (by rw [hi3v']; omega))
    have hx0 : (0 : ℤ) ≤ (1024 + (f16Man h : ℤ)) * 2 ^ (f16Exp h + 7) := by positivity
    have hxlt : (1024 + (f16Man h : ℤ)) * 2 ^ (f16Exp h + 7) < 2 ^ 63 := by
      have h1 : (1024 + (f16Man h : ℤ)) ≤ 2047 := by omega
      have h2 : (2 : ℤ) ^ (f16Exp h + 7) ≤ 2 ^ 38 := pow_le_pow_right₀ (by norm_num) (by omega)
      have h3 : (1024 + (f16Man h : ℤ)) * 2 ^ (f16Exp h + 7) ≤ 2047 * 2 ^ 38 :=
        mul_le_mul h1 h2 (by positivity) (by norm_num)
      norm_num at h3 ⊢
      omega
    have hshv' : sh.val = f16Mag h := by
      have hsz : I64.size = 2 ^ 64 := by simp [I64.size, I64.numBits]
      rw [hshv, hsz, hsigv', hi3v', f16Mag, if_neg hE0, Int.shiftLeft_eq]
      apply Int.bmod_eq_of_le
      · norm_num; omega
      · norm_num; omega
    simp only [hsig, hi3, hsh, bind_tc_ok]
    by_cases hs1 : sign = 1#u16
    · have hS : f16Sign h = 1 := by rw [← hsv, hs1]; simp
      obtain ⟨mag1, hneg, hnegv⟩ := WP.spec_imp_exists
        (HNeg.hNeg.step sh (by
          intro hc
          rw [hshv', IScalar.min_IScalarTy_I64_eq, I64.min_eq] at hc
          have : 0 ≤ f16Mag h := by rw [f16Mag, if_neg hE0]; positivity
          omega))
      simp only [if_pos hs1, hneg]
      refine ⟨mag1, rfl, ?_⟩
      rw [hnegv, hshv', f16Value, if_pos hS]
    · have hS : f16Sign h ≠ 1 := by
        intro hc; apply hs1; apply UScalar.eq_of_val_eq; rw [hsv, hc]; simp
      simp only [if_neg hs1]
      refine ⟨sh, rfl, ?_⟩
      rw [hshv', f16Value, if_neg hS]

/-- Non-vacuity: `0x3C00` (binary16 `1.0`) decodes to the Q32.32 unit `2^32`. -/
example : ∃ v : I64, kern.bitplane.decode_f16 0x3C00#u16 = ok v ∧ v.val = (2 ^ 32 : ℤ) := by
  obtain ⟨v, hv, hvval⟩ := word_decode_f16_ok 0x3C00#u16
  refine ⟨v, hv, ?_⟩
  rw [hvval]
  norm_num [f16Value, f16Mag, f16Exp, f16Man, f16Sign]

/-! ## Nibble dots: `high_dot` and `low_dot` (portable path)

Each extracted dot carries an `avx : Bool`: `avx = true` routes to the extracted AVX2 axioms
(`kern.bitplane.x86.high_dot_avx2` / `low_dot_avx2`), so the theorems are stated at `avx = false`,
the portable 16-iteration loop. `high_dot` treats both nibbles as signed 4-bit values, `low_dot` as
unsigned. Every product and partial sum stays inside `i32`, so the checked arithmetic never
refuses. -/

/-! ### Scalar value lemmas -/

/-- `U8 -> I8` heterogeneous cast of a nibble is the nibble. -/
lemma hcast_I8_val (n : U8) (hn : n.val < 16) :
    (UScalar.hcast .I8 n).val = (n.val : ℤ) := by
  rw [UScalar.hcast_val_eq, IScalarTy.I8_numBits_eq]
  apply Int.bmod_eq_of_le
  · norm_num
  · norm_num; omega

/-- `U8 -> I32` heterogeneous cast is the value. -/
lemma hcast_I32_val (n : U8) : (UScalar.hcast .I32 n).val = (n.val : ℤ) := by
  rw [UScalar.hcast_val_eq, IScalarTy.I32_numBits_eq]
  apply Int.bmod_eq_of_le
  · norm_num
  · have h := n.hBounds
    rw [UScalarTy.U8_numBits_eq] at h
    norm_num at h
    norm_num; omega

/-- `i8 -> i32` (`FromI32I8`) widening preserves the value. -/
lemma fromI32I8_val (x : I8) : (core.convert.num.FromI32I8.from x).val = x.val :=
  core.convert.num.FromI32I8.from_val_eq x

/-- `n &&& 15 < 16` as a `Nat` on the value. -/
lemma val_and15_lt (n : U8) : (n.val &&& 15) < 16 :=
  Nat.and_lt_two_pow n.val (y := 15) (n := 4) (by norm_num)

/-- `n >>> 4 < 16` (the high nibble of a byte). -/
lemma val_shr4_lt (n : U8) : (n.val >>> 4) < 16 := by
  rw [Nat.shiftRight_eq_div_pow]
  have h := n.hBounds
  rw [UScalarTy.U8_numBits_eq] at h
  norm_num at h
  omega

/-- `n ^^^ 8` of a nibble stays a nibble. -/
lemma val_xor8_lt (n : U8) (hn : n.val < 16) : (n.val ^^^ 8) < 16 :=
  Nat.xor_lt_two_pow (n := 4) hn (by norm_num)

/-- `(n &&& 15#u8).val` as a `Nat` bound. -/
lemma uand15_lt (n : U8) : (n &&& 15#u8).val < 16 := by
  rw [UScalar.val_and]
  have h15 : (15#u8 : U8).val = 15 := by simp
  rw [h15]
  exact val_and15_lt n

/-- `(n ^^^ 8#u8).val` of a nibble. -/
lemma uxor8_lt (n : U8) (hn : n.val < 16) : (n ^^^ 8#u8).val < 16 := by
  rw [UScalar.val_xor]
  have h8 : (8#u8 : U8).val = 8 := by simp
  rw [h8]
  exact val_xor8_lt n hn

/-- `(n &&& 15#u8).val` unfolded to the `Nat` expression. -/
lemma uand_val (n : U8) : (n &&& 15#u8).val = n.val &&& 15 := by
  rw [UScalar.val_and]
  have h15 : (15#u8 : U8).val = 15 := by simp
  rw [h15]

/-- `(n ^^^ 8#u8).val` unfolded to the `Nat` expression. -/
lemma uxor_val (n : U8) : (n ^^^ 8#u8).val = n.val ^^^ 8 := by
  rw [UScalar.val_xor]
  have h8 : (8#u8 : U8).val = 8 := by simp
  rw [h8]

/-- Every `i8` lies in `[-128, 127]`. -/
lemma I8_val_bounds (x : I8) : -128 ≤ (x.val : ℤ) ∧ (x.val : ℤ) ≤ 127 := by
  have h := x.hBounds
  rw [IScalarTy.I8_numBits_eq] at h
  norm_num at h
  omega

/-- Signed 4-bit nibble value: the Rust `(n ^^ 8) as i8 - 8` on a nibble `n < 16`. -/
def signedNib (n : Nat) : ℤ := (((n ^^^ 8) : Nat) : ℤ) - 8

lemma signedNib_bounds (n : Nat) (hn : n < 16) : -8 ≤ signedNib n ∧ signedNib n ≤ 7 := by
  have hx : n < 2 ^ 4 := by norm_num at hn ⊢; exact hn
  have h8 : (8 : Nat) < 2 ^ 4 := by norm_num
  have h := Nat.xor_lt_two_pow hx h8
  simp only [signedNib]
  omega

/-- Bounds on `(hcast .I8 (n ^^^ 8)) - 8`, the signed nibble of a nibble `n`. -/
lemma hcastI8_xor_bounds (n : U8) (hn : n.val < 16) :
    -8 ≤ (UScalar.hcast .I8 (n ^^^ 8#u8)).val - 8 ∧
    (UScalar.hcast .I8 (n ^^^ 8#u8)).val - 8 ≤ 7 := by
  rw [hcast_I8_val _ (uxor8_lt n hn), uxor_val]
  have h := signedNib_bounds n.val hn
  simp only [signedNib] at h
  exact h

/-- Product of a signed nibble (`[-8,7]`) and an `i8` (`[-128,127]`) stays in `[-1024,1024]`. -/
lemma mul_signed_bounds (u v : ℤ) (hu : -8 ≤ u ∧ u ≤ 7) (hv : -128 ≤ v ∧ v ≤ 127) :
    -1024 ≤ u * v ∧ u * v ≤ 1024 := by
  constructor <;> nlinarith [hu.1, hu.2, hv.1, hv.2]

/-- Product of an unsigned nibble (`[0,15]`) and an `i8` stays in `[-2048,2048]`. -/
lemma mul_unsigned_bounds (u v : ℤ) (hu : 0 ≤ u ∧ u ≤ 15) (hv : -128 ≤ v ∧ v ≤ 127) :
    -2048 ≤ u * v ∧ u * v ≤ 2048 := by
  constructor <;> nlinarith [hu.1, hu.2, hv.1, hv.2]

/-- `high_dot` reference: the signed 4-bit dot over `j < n`. -/
def highDotAux (packed : List Std.U8) (a : List Std.I8) : Nat → ℤ
  | 0 => 0
  | k + 1 =>
      highDotAux packed a k
        + signedNib ((packed.getD k 0#u8).val &&& 15) * (a.getD k 0#i8).val
        + signedNib ((packed.getD k 0#u8).val >>> 4) * (a.getD (k + 16) 0#i8).val

/-- `low_dot` reference: the unsigned 4-bit dot over `j < n`. -/
def lowDotAux (packed : List Std.U8) (a : List Std.I8) : Nat → ℤ
  | 0 => 0
  | k + 1 =>
      lowDotAux packed a k
        + (((packed.getD k 0#u8).val &&& 15 : Nat) : ℤ) * (a.getD k 0#i8).val
        + (((packed.getD k 0#u8).val >>> 4 : Nat) : ℤ) * (a.getD (k + 16) 0#i8).val

lemma highDotAux_bound (packed : List U8) (a : List I8) :
    ∀ (k : Nat), k ≤ 16 → -((2048 : ℤ) * k) ≤ highDotAux packed a k ∧
      highDotAux packed a k ≤ (2048 : ℤ) * k := by
  intro k
  induction k with
  | zero => intro _; simp [highDotAux]
  | succ k ih =>
    intro hk
    have ih' := ih (by omega)
    have hb1 := signedNib_bounds _ (val_and15_lt (packed.getD k 0#u8))
    have hb2 := signedNib_bounds _ (val_shr4_lt (packed.getD k 0#u8))
    have ha1 := I8_val_bounds (a.getD k 0#i8)
    have ha2 := I8_val_bounds (a.getD (k + 16) 0#i8)
    have ht1 := mul_signed_bounds _ _ hb1 ha1
    have ht2 := mul_signed_bounds _ _ hb2 ha2
    have hstep : highDotAux packed a (k + 1) =
        highDotAux packed a k
          + signedNib ((packed.getD k 0#u8).val &&& 15) * (a.getD k 0#i8).val
          + signedNib ((packed.getD k 0#u8).val >>> 4) * (a.getD (k + 16) 0#i8).val := rfl
    rw [hstep]
    constructor
    · push_cast; linarith [ih'.1, ht1.1, ht2.1]
    · push_cast; linarith [ih'.2, ht1.2, ht2.2]

lemma lowDotAux_bound (packed : List U8) (a : List I8) :
    ∀ (k : Nat), k ≤ 16 → -((4096 : ℤ) * k) ≤ lowDotAux packed a k ∧
      lowDotAux packed a k ≤ (4096 : ℤ) * k := by
  intro k
  induction k with
  | zero => intro _; simp [lowDotAux]
  | succ k ih =>
    intro hk
    have ih' := ih (by omega)
    have hu1 : (0 : ℤ) ≤ (((packed.getD k 0#u8).val &&& 15 : Nat) : ℤ) ∧
        (((packed.getD k 0#u8).val &&& 15 : Nat) : ℤ) ≤ 15 := by
      have := val_and15_lt (packed.getD k 0#u8); omega
    have hu2 : (0 : ℤ) ≤ (((packed.getD k 0#u8).val >>> 4 : Nat) : ℤ) ∧
        (((packed.getD k 0#u8).val >>> 4 : Nat) : ℤ) ≤ 15 := by
      have := val_shr4_lt (packed.getD k 0#u8); omega
    have ha1 := I8_val_bounds (a.getD k 0#i8)
    have ha2 := I8_val_bounds (a.getD (k + 16) 0#i8)
    have ht1 := mul_unsigned_bounds _ _ hu1 ha1
    have ht2 := mul_unsigned_bounds _ _ hu2 ha2
    have hstep : lowDotAux packed a (k + 1) =
        lowDotAux packed a k
          + (((packed.getD k 0#u8).val &&& 15 : Nat) : ℤ) * (a.getD k 0#i8).val
          + (((packed.getD k 0#u8).val >>> 4 : Nat) : ℤ) * (a.getD (k + 16) 0#i8).val := rfl
    rw [hstep]
    constructor
    · simp only [Nat.cast_add, Nat.cast_one]; linarith [ih'.1, ht1.1, ht2.1]
    · simp only [Nat.cast_add, Nat.cast_one]; linarith [ih'.2, ht1.2, ht2.2]

/-- The portable `high_dot` loop: 16 iterations, both nibbles signed. -/
theorem high_dot_loop_spec (packed : Slice Std.U8) (a : Slice Std.I8)
    (s0 : Std.I32) (j0 : Std.Usize)
    (hs : s0.val = highDotAux packed.val a.val j0.val) (hj : j0.val ≤ 16)
    (hp : 16 ≤ packed.val.length) (ha : 32 ≤ a.val.length) :
    kern.bitplane.high_dot_loop packed a s0 j0
      ⦃ r => r.val = highDotAux packed.val a.val 16 ⦄ := by
  unfold kern.bitplane.high_dot_loop
  apply loop.spec_decr_nat
    (measure := fun st : Std.I32 × Std.Usize => 16 - st.2.val)
    (inv := fun st => st.1.val = highDotAux packed.val a.val st.2.val ∧ st.2.val ≤ 16)
    (post := fun r : Std.I32 => r.val = highDotAux packed.val a.val 16)
  · rintro ⟨s, j⟩ ⟨hs', hj'⟩
    dsimp only
    unfold kern.bitplane.high_dot_loop.body
    rw [word_HALF_eq]
    simp only [bind_tc_ok]
    by_cases hjlt : j < (16#usize)
    · rw [if_pos hjlt]
      have hj16 : (j.val : ℕ) < 16 := by scalar_tac
      simp only [lift, bind_tc_ok]
      step*
      · rw [I8.min_eq]
        have hb := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
        have h8 : (8#i8 : I8).val = 8 := by simp
        rw [h8]; omega
      · rw [I8.max_eq]
        have hb := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
        have h8 : (8#i8 : I8).val = 8 := by simp
        rw [h8]; omega
      · rw [I8.min_eq]
        have hb := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
        have h8 : (8#i8 : I8).val = 8 := by simp
        rw [h8]; omega
      · rw [I8.max_eq]
        have hb := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
        have h8 : (8#i8 : I8).val = 8 := by simp
        rw [h8]; omega
      · rw [I32.min_eq, fromI32I8_val, fromI32I8_val, i5_post]
        have h5 := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
        have h10 := I8_val_bounds i10
        have h := mul_signed_bounds _ _ h5 ⟨h10.1, h10.2⟩
        omega
      · rw [I32.max_eq, fromI32I8_val, fromI32I8_val, i5_post]
        have h5 := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
        have h10 := I8_val_bounds i10
        have h := mul_signed_bounds _ _ h5 ⟨h10.1, h10.2⟩
        omega
      · rw [I32.min_eq, fromI32I8_val, fromI32I8_val, i9_post]
        have h9 := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
        have h14 := I8_val_bounds i14
        have h := mul_signed_bounds _ _ h9 ⟨h14.1, h14.2⟩
        omega
      · rw [I32.max_eq, fromI32I8_val, fromI32I8_val, i9_post]
        have h9 := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
        have h14 := I8_val_bounds i14
        have h := mul_signed_bounds _ _ h9 ⟨h14.1, h14.2⟩
        omega
      · rw [I32.min_eq]
        have h12 : -1024 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 1024 := by
          rw [i12_post, fromI32I8_val, fromI32I8_val, i5_post]
          have h5 := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
          have h10 := I8_val_bounds i10
          exact mul_signed_bounds _ _ h5 ⟨h10.1, h10.2⟩
        have h16 : -1024 ≤ (i16.val : ℤ) ∧ (i16.val : ℤ) ≤ 1024 := by
          rw [i16_post, fromI32I8_val, fromI32I8_val, i9_post]
          have h9 := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
          have h14 := I8_val_bounds i14
          exact mul_signed_bounds _ _ h9 ⟨h14.1, h14.2⟩
        omega
      · rw [I32.max_eq]
        have h12 : -1024 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 1024 := by
          rw [i12_post, fromI32I8_val, fromI32I8_val, i5_post]
          have h5 := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
          have h10 := I8_val_bounds i10
          exact mul_signed_bounds _ _ h5 ⟨h10.1, h10.2⟩
        have h16 : -1024 ≤ (i16.val : ℤ) ∧ (i16.val : ℤ) ≤ 1024 := by
          rw [i16_post, fromI32I8_val, fromI32I8_val, i9_post]
          have h9 := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
          have h14 := I8_val_bounds i14
          exact mul_signed_bounds _ _ h9 ⟨h14.1, h14.2⟩
        omega
      · rw [I32.min_eq]
        have h12 : -1024 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 1024 := by
          rw [i12_post, fromI32I8_val, fromI32I8_val, i5_post]
          have h5 := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
          have h10 := I8_val_bounds i10
          exact mul_signed_bounds _ _ h5 ⟨h10.1, h10.2⟩
        have h16 : -1024 ≤ (i16.val : ℤ) ∧ (i16.val : ℤ) ≤ 1024 := by
          rw [i16_post, fromI32I8_val, fromI32I8_val, i9_post]
          have h9 := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
          have h14 := I8_val_bounds i14
          exact mul_signed_bounds _ _ h9 ⟨h14.1, h14.2⟩
        have hs : -32768 ≤ (s.val : ℤ) ∧ (s.val : ℤ) ≤ 32768 := by
          have hj16 : (j.val : ℤ) ≤ 16 := by omega
          rw [hs']
          have hb := highDotAux_bound packed.val a.val j.val (by omega)
          constructor <;> linarith [hb.1, hb.2, hj16]
        rw [i17_post]
        omega
      · rw [I32.max_eq]
        have h12 : -1024 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 1024 := by
          rw [i12_post, fromI32I8_val, fromI32I8_val, i5_post]
          have h5 := hcastI8_xor_bounds (i1 &&& 15#u8) (uand15_lt i1)
          have h10 := I8_val_bounds i10
          exact mul_signed_bounds _ _ h5 ⟨h10.1, h10.2⟩
        have h16 : -1024 ≤ (i16.val : ℤ) ∧ (i16.val : ℤ) ≤ 1024 := by
          rw [i16_post, fromI32I8_val, fromI32I8_val, i9_post]
          have h9 := hcastI8_xor_bounds i6 (by rw [i6_post]; exact val_shr4_lt i1)
          have h14 := I8_val_bounds i14
          exact mul_signed_bounds _ _ h9 ⟨h14.1, h14.2⟩
        have hs : -32768 ≤ (s.val : ℤ) ∧ (s.val : ℤ) ≤ 32768 := by
          have hj16 : (j.val : ℤ) ≤ 16 := by omega
          rw [hs']
          have hb := highDotAux_bound packed.val a.val j.val (by omega)
          constructor <;> linarith [hb.1, hb.2, hj16]
        rw [i17_post]
        omega
      · refine ⟨?_, ?_, ?_⟩
        · have hjlen : j.val < packed.val.length := by have := hp; omega
          have hgl : packed.val.getD j.val 0#u8 = i1 := by
            rw [i1_post, List.getD_eq_getElem packed.val 0#u8 hjlen]
          have hga : a.val.getD j.val 0#i8 = i10 := by
            rw [i10_post, List.getD_eq_getElem a.val 0#i8 (by have := ha; omega)]
          have hga2 : a.val.getD (j.val + 16) 0#i8 = i14 := by
            rw [i14_post, ← i13_post]
            exact List.getD_eq_getElem a.val 0#i8 (by rw [i13_post]; have := ha; omega)
          have hi5 : (↑i5 : ℤ) = (((i1.val &&& 15) ^^^ 8 : ℕ) : ℤ) - 8 := by
            rw [i5_post, hcast_I8_val _ (uxor8_lt _ (uand15_lt i1)), uxor_val, uand_val]
            try rfl
          have hi9 : (↑i9 : ℤ) = (((i1.val >>> 4) ^^^ 8 : ℕ) : ℤ) - 8 := by
            rw [i9_post, hcast_I8_val _ (uxor8_lt _ (by rw [i6_post]; exact val_shr4_lt i1)),
              uxor_val, i6_post]
            try rfl
          rw [s1_post, i17_post, i12_post, i16_post, fromI32I8_val, fromI32I8_val,
            fromI32I8_val, fromI32I8_val, j1_post]
          simp only [highDotAux, signedNib]
          rw [← hs', hi5, hi9, hgl, hga, hga2]
          ring
        · rw [j1_post]; scalar_tac
        · rw [j1_post]
          have hj16 : (j.val : ℕ) < 16 := by scalar_tac
          omega
    · rw [if_neg hjlt]
      have hj16 : j.val = 16 := by scalar_tac
      simpa [hj16] using hs'
  · exact ⟨hs, hj⟩

/-- The portable `low_dot` loop: 16 iterations, both nibbles unsigned. -/
theorem low_dot_loop_spec (packed : Slice Std.U8) (a : Slice Std.I8)
    (s0 : Std.I32) (j0 : Std.Usize)
    (hs : s0.val = lowDotAux packed.val a.val j0.val) (hj : j0.val ≤ 16)
    (hp : 16 ≤ packed.val.length) (ha : 32 ≤ a.val.length) :
    kern.bitplane.low_dot_loop packed a s0 j0
      ⦃ r => r.val = lowDotAux packed.val a.val 16 ⦄ := by
  unfold kern.bitplane.low_dot_loop
  apply loop.spec_decr_nat
    (measure := fun st : Std.I32 × Std.Usize => 16 - st.2.val)
    (inv := fun st => st.1.val = lowDotAux packed.val a.val st.2.val ∧ st.2.val ≤ 16)
    (post := fun r : Std.I32 => r.val = lowDotAux packed.val a.val 16)
  · rintro ⟨s, j⟩ ⟨hs', hj'⟩
    dsimp only
    unfold kern.bitplane.low_dot_loop.body
    rw [word_HALF_eq]
    simp only [bind_tc_ok, I32.Insts.CoreConvertFromU8.from]
    by_cases hjlt : j < (16#usize)
    · rw [if_pos hjlt]
      have hj16 : (j.val : ℕ) < 16 := by scalar_tac
      simp only [lift, bind_tc_ok]
      step*
      · rw [I32.min_eq, hcast_I32_val, fromI32I8_val]
        have hu : (0 : ℤ) ≤ ((i1 &&& 15#u8).val : ℤ) ∧ ((i1 &&& 15#u8).val : ℤ) ≤ 15 := by
          have := uand15_lt i1; omega
        have h4 := I8_val_bounds i4
        have h := mul_unsigned_bounds _ _ hu ⟨h4.1, h4.2⟩
        omega
      · rw [I32.max_eq, hcast_I32_val, fromI32I8_val]
        have hu : (0 : ℤ) ≤ ((i1 &&& 15#u8).val : ℤ) ∧ ((i1 &&& 15#u8).val : ℤ) ≤ 15 := by
          have := uand15_lt i1; omega
        have h4 := I8_val_bounds i4
        have h := mul_unsigned_bounds _ _ hu ⟨h4.1, h4.2⟩
        omega
      · rw [I32.min_eq, hcast_I32_val, fromI32I8_val]
        have hu : (0 : ℤ) ≤ (i7.val : ℤ) ∧ (i7.val : ℤ) ≤ 15 := by
          have : i7.val < 16 := by rw [i7_post]; exact val_shr4_lt i1
          omega
        have h10 := I8_val_bounds i10
        have h := mul_unsigned_bounds _ _ hu ⟨h10.1, h10.2⟩
        omega
      · rw [I32.max_eq, hcast_I32_val, fromI32I8_val]
        have hu : (0 : ℤ) ≤ (i7.val : ℤ) ∧ (i7.val : ℤ) ≤ 15 := by
          have : i7.val < 16 := by rw [i7_post]; exact val_shr4_lt i1
          omega
        have h10 := I8_val_bounds i10
        have h := mul_unsigned_bounds _ _ hu ⟨h10.1, h10.2⟩
        omega
      · rw [I32.min_eq]
        have h6 : -2048 ≤ (i6.val : ℤ) ∧ (i6.val : ℤ) ≤ 2048 := by
          rw [i6_post, hcast_I32_val, fromI32I8_val, uand_val]
          have hu : (0 : ℤ) ≤ ((i1.val &&& 15 : ℕ) : ℤ) ∧ ((i1.val &&& 15 : ℕ) : ℤ) ≤ 15 := by
            have := val_and15_lt i1; omega
          have h4 := I8_val_bounds i4
          exact mul_unsigned_bounds _ _ hu ⟨h4.1, h4.2⟩
        have h12 : -2048 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 2048 := by
          rw [i12_post, hcast_I32_val, fromI32I8_val, i7_post]
          have hu : (0 : ℤ) ≤ ((i1.val >>> 4 : ℕ) : ℤ) ∧ ((i1.val >>> 4 : ℕ) : ℤ) ≤ 15 := by
            have : i7.val < 16 := by rw [i7_post]; exact val_shr4_lt i1
            omega
          have h10 := I8_val_bounds i10
          exact mul_unsigned_bounds _ _ hu ⟨h10.1, h10.2⟩
        omega
      · rw [I32.max_eq]
        have h6 : -2048 ≤ (i6.val : ℤ) ∧ (i6.val : ℤ) ≤ 2048 := by
          rw [i6_post, hcast_I32_val, fromI32I8_val, uand_val]
          have hu : (0 : ℤ) ≤ ((i1.val &&& 15 : ℕ) : ℤ) ∧ ((i1.val &&& 15 : ℕ) : ℤ) ≤ 15 := by
            have := val_and15_lt i1; omega
          have h4 := I8_val_bounds i4
          exact mul_unsigned_bounds _ _ hu ⟨h4.1, h4.2⟩
        have h12 : -2048 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 2048 := by
          rw [i12_post, hcast_I32_val, fromI32I8_val, i7_post]
          have hu : (0 : ℤ) ≤ ((i1.val >>> 4 : ℕ) : ℤ) ∧ ((i1.val >>> 4 : ℕ) : ℤ) ≤ 15 := by
            have : i7.val < 16 := by rw [i7_post]; exact val_shr4_lt i1
            omega
          have h10 := I8_val_bounds i10
          exact mul_unsigned_bounds _ _ hu ⟨h10.1, h10.2⟩
        omega
      · rw [I32.min_eq]
        have h6 : -2048 ≤ (i6.val : ℤ) ∧ (i6.val : ℤ) ≤ 2048 := by
          rw [i6_post, hcast_I32_val, fromI32I8_val, uand_val]
          have hu : (0 : ℤ) ≤ ((i1.val &&& 15 : ℕ) : ℤ) ∧ ((i1.val &&& 15 : ℕ) : ℤ) ≤ 15 := by
            have := val_and15_lt i1; omega
          have h4 := I8_val_bounds i4
          exact mul_unsigned_bounds _ _ hu ⟨h4.1, h4.2⟩
        have h12 : -2048 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 2048 := by
          rw [i12_post, hcast_I32_val, fromI32I8_val, i7_post]
          have hu : (0 : ℤ) ≤ ((i1.val >>> 4 : ℕ) : ℤ) ∧ ((i1.val >>> 4 : ℕ) : ℤ) ≤ 15 := by
            have : i7.val < 16 := by rw [i7_post]; exact val_shr4_lt i1
            omega
          have h10 := I8_val_bounds i10
          exact mul_unsigned_bounds _ _ hu ⟨h10.1, h10.2⟩
        have hs : -65536 ≤ (s.val : ℤ) ∧ (s.val : ℤ) ≤ 65536 := by
          have hj16 : (j.val : ℤ) ≤ 16 := by omega
          rw [hs']
          have hb := lowDotAux_bound packed.val a.val j.val (by omega)
          constructor <;> linarith [hb.1, hb.2, hj16]
        rw [i13_post]
        omega
      · rw [I32.max_eq]
        have h6 : -2048 ≤ (i6.val : ℤ) ∧ (i6.val : ℤ) ≤ 2048 := by
          rw [i6_post, hcast_I32_val, fromI32I8_val, uand_val]
          have hu : (0 : ℤ) ≤ ((i1.val &&& 15 : ℕ) : ℤ) ∧ ((i1.val &&& 15 : ℕ) : ℤ) ≤ 15 := by
            have := val_and15_lt i1; omega
          have h4 := I8_val_bounds i4
          exact mul_unsigned_bounds _ _ hu ⟨h4.1, h4.2⟩
        have h12 : -2048 ≤ (i12.val : ℤ) ∧ (i12.val : ℤ) ≤ 2048 := by
          rw [i12_post, hcast_I32_val, fromI32I8_val, i7_post]
          have hu : (0 : ℤ) ≤ ((i1.val >>> 4 : ℕ) : ℤ) ∧ ((i1.val >>> 4 : ℕ) : ℤ) ≤ 15 := by
            have : i7.val < 16 := by rw [i7_post]; exact val_shr4_lt i1
            omega
          have h10 := I8_val_bounds i10
          exact mul_unsigned_bounds _ _ hu ⟨h10.1, h10.2⟩
        have hs : -65536 ≤ (s.val : ℤ) ∧ (s.val : ℤ) ≤ 65536 := by
          have hj16 : (j.val : ℤ) ≤ 16 := by omega
          rw [hs']
          have hb := lowDotAux_bound packed.val a.val j.val (by omega)
          constructor <;> linarith [hb.1, hb.2, hj16]
        rw [i13_post]
        omega
      · refine ⟨?_, ?_, ?_⟩
        · have hjlen : j.val < packed.val.length := by have := hp; omega
          have hgl : packed.val.getD j.val 0#u8 = i1 := by
            rw [i1_post, List.getD_eq_getElem packed.val 0#u8 hjlen]
          have hga : a.val.getD j.val 0#i8 = i4 := by
            rw [i4_post, List.getD_eq_getElem a.val 0#i8 (by have := ha; omega)]
          have hga2 : a.val.getD (j.val + 16) 0#i8 = i10 := by
            rw [i10_post, ← i9_post]
            exact List.getD_eq_getElem a.val 0#i8 (by rw [i9_post]; have := ha; omega)
          have hi6 : (i6.val : ℤ) = ((i1.val &&& 15 : ℕ) : ℤ) * (i4.val : ℤ) := by
            rw [i6_post, hcast_I32_val, fromI32I8_val, uand_val]
          have hi12 : (i12.val : ℤ) = ((i1.val >>> 4 : ℕ) : ℤ) * (i10.val : ℤ) := by
            rw [i12_post, hcast_I32_val, fromI32I8_val, i7_post]
          rw [s1_post, i13_post, hi6, hi12, j1_post]
          simp only [lowDotAux]
          rw [← hs', hgl, hga, hga2]
          ring
        · rw [j1_post]
          have hj16 : (j.val : ℕ) < 16 := by scalar_tac
          omega
        · rw [j1_post]
          have hj16 : (j.val : ℕ) < 16 := by scalar_tac
          omega
    · rw [if_neg hjlt]
      have hj16 : j.val = 16 := by scalar_tac
      simpa [hj16] using hs'
  · exact ⟨hs, hj⟩

/-- `high_dot` at `avx = false` computes the signed 4-bit dot of the block with the 32 activations. -/
theorem word_high_dot_ok (packed : Slice Std.U8) (a : Slice Std.I8)
    (hp : 16 ≤ packed.val.length) (ha : 32 ≤ a.val.length) :
    ∃ v : I32, kern.bitplane.high_dot packed a false = ok v ∧
      v.val = highDotAux packed.val a.val 16 := by
  have h : kern.bitplane.high_dot packed a false ⦃ r => r.val = highDotAux packed.val a.val 16 ⦄ := by
    unfold kern.bitplane.high_dot
    rw [if_neg (by decide : ¬ (false = true))]
    apply high_dot_loop_spec packed a 0#i32 0#usize
    · simp [highDotAux]
    · simp
    · exact hp
    · exact ha
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  exact ⟨y, hy, hyv⟩

/-- `low_dot` at `avx = false` computes the unsigned 4-bit dot of the block with the 32 activations. -/
theorem word_low_dot_ok (packed : Slice Std.U8) (a : Slice Std.I8)
    (hp : 16 ≤ packed.val.length) (ha : 32 ≤ a.val.length) :
    ∃ v : I32, kern.bitplane.low_dot packed a false = ok v ∧
      v.val = lowDotAux packed.val a.val 16 := by
  have h : kern.bitplane.low_dot packed a false ⦃ r => r.val = lowDotAux packed.val a.val 16 ⦄ := by
    unfold kern.bitplane.low_dot
    rw [if_neg (by decide : ¬ (false = true))]
    apply low_dot_loop_spec packed a 0#i32 0#usize
    · simp [lowDotAux]
    · simp
    · exact hp
    · exact ha
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  exact ⟨y, hy, hyv⟩

/-- Non-vacuity: one byte `0xF0` (low nibble `0`, high nibble `15`) against activations
`a[0] = 3`, `a[16] = 5` gives `highDotAux = 0*3 + (-1)*5 = -5`. -/
example : highDotAux [0xF0#u8] ([3#i8] ++ List.replicate 15 0#i8 ++ [5#i8]) 1 = -5 := by
  decide

/-- Non-vacuity: the same byte against the same activations gives
`lowDotAux = 0*3 + 15*5 = 75`. -/
example : lowDotAux [0xF0#u8] ([3#i8] ++ List.replicate 15 0#i8 ++ [5#i8]) 1 = 75 := by
  decide

/-! ## `block`: the 32-code activation window

`block q blk` is the 32 codes `q.codes[blk*32 .. (blk+1)*32]`. The extracted body performs two
checked `usize` products and one checked add before the range index, so the theorem carries two
no-overflow premises and the in-range premise; outside them the body refuses. The reference is the
`List.slice` of the code vector. -/

theorem word_block_ok (q : chart.QuantAct) (blk : Std.Usize)
    (hm : (blk.val + 1) * 32 ≤ Std.Usize.max)
    (hb : (blk.val + 1) * 32 ≤ (alloc.vec.Vec.deref q.codes).val.length) :
    ∃ r : Slice Std.I8, kern.bitplane.block q blk = ok r ∧
      r.val = (alloc.vec.Vec.deref q.codes).val.slice (blk.val * 32) ((blk.val + 1) * 32) := by
  have h : kern.bitplane.block q blk
      ⦃ r => r.val =
        (alloc.vec.Vec.deref q.codes).val.slice (blk.val * 32) ((blk.val + 1) * 32) ⦄ := by
    unfold kern.bitplane.block
    simp only [chart.QuantAct.impl.codes, bind_tc_ok, quant.Q8_0_QK]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  exact ⟨y, hy, hyv⟩

/-- Non-vacuity: a 64-code table at `blk = 1` yields the second 32 codes. -/
example : ∃ r : Slice Std.I8,
    kern.bitplane.block
        { scales := ⟨Slice.from [] (by scalar_tac), 0#usize⟩,
          codes := ⟨Slice.from (List.replicate 64 7#i8) (by scalar_tac), 0#usize⟩ }
        1#usize = ok r ∧
      r.val = List.replicate 32 7#i8 := by
  obtain ⟨r, hr, hrv⟩ := word_block_ok
    { scales := ⟨Slice.from [] (by scalar_tac), 0#usize⟩,
      codes := ⟨Slice.from (List.replicate 64 7#i8) (by scalar_tac), 0#usize⟩ }
    1#usize (by scalar_tac)
    (by simp only [alloc.vec.Vec.deref, alloc.vec.Vec.val, Slice.from_val]; scalar_tac)
  refine ⟨r, hr, ?_⟩
  rw [hrv]
  simp only [alloc.vec.Vec.deref, alloc.vec.Vec.val, Slice.from_val]
  decide

end

end apeiron_word
