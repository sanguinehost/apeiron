import ApeironWord
import Numeric.Dequant.Q5_0

set_option autoImplicit false
set_option warningAsError true

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-!
# Serving kernel group `kern.matmat`: the portable Q5_0 int8 dot

`src/engine/kern/matmat.rs` holds the k-column int8-activation projection. Its portable integer
core is `portable_q5_0_dot`, the `Q5_0` block dot used by `q5_0_block_kdots_portable` (and through it
by `q5_0_block_kdots`, `row_accs` and `matmat_q8act`). This module refines the extracted bodies of
that core against a plain-`Nat`/`Int` reference over the packed block bytes.

The reference reads the `qh`/`qs` bytes the way the `Q5_0` layout packs them — `q5Bit` is the fifth
bit plane, `q5Low5`/`q5High5` assemble a five-bit code from a nibble and its plane bit, and `q5Term`
is the signed code times the activation lane — and `q5Sum` folds the sixteen lanes into the exact
`Int` accumulator the Rust accumulates. `q5Word4` is the four-byte little-endian `qh` word.

Trust boundary: rustc/LLVM codegen, the kernel ABI and the `unsafe` AVX2 container are not covered;
only the extracted portable integer body. The `F32`/`F64` scale carriers and the `_avx2` cores sit
in a different layer (see the group report).

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`. `decide` is used only on closed literal facts
about the reference, never on an extracted body. The one `set_option` is the finite
`maxHeartbeats 400000` on `q5_dot_loop_body_spec`: its twelve checked-arithmetic goals exhaust the
default 200000.
-/

/-! ## Reference: the Q5_0 packed five-bit code -/

/-- Bit `t` of the little-endian `qh` word, as a `Nat` (`0`/`1`): the Rust `(qh_u32 >>> t) & 1`. -/
def q5Bit (w t : Nat) : Nat := (w >>> t) &&& 1

/-- The five-bit code assembled from the low nibble of a byte and its plane bit: the Rust
`(nib & 0x0F) | (bit << 4)` on an eight-bit lane. -/
def q5Low5 (nib bit : Nat) : Nat := (nib &&& 15) ||| ((bit <<< 4) % 256)

/-- The five-bit code assembled from the high nibble of a byte and its plane bit: the Rust
`(nib >> 4) | (bit << 4)`. -/
def q5High5 (nib bit : Nat) : Nat := (nib >>> 4) ||| ((bit <<< 4) % 256)

/-- The signed `Q5_0` code of a five-bit value: subtract the `16` bias. -/
def q5Code (five : Nat) : Int := (five : Int) - 16

/-- One lane's reference contribution: the low-nibble code times `xq[j]` plus the high-nibble code
times `xq[16 + j]`. `w` is the `qh` word, `qs` the packed nibble bytes, `xq` the 32 activation
lanes. -/
def q5Term (w : Nat) (qs : List U8) (xq : List I8) (j : Nat) : Int :=
  q5Code (q5Low5 (qs.getD j 0#u8).val (q5Bit w j)) * (xq.getD j 0#i8).val +
    q5Code (q5High5 (qs.getD j 0#u8).val (q5Bit w (16 + j))) * (xq.getD (16 + j) 0#i8).val

/-- The reference prefix accumulator after `n` lanes. -/
def q5Sum (w : Nat) (qs : List U8) (xq : List I8) : Nat → Int
  | 0 => 0
  | n + 1 => q5Sum w qs xq n + q5Term w qs xq n

/-- The `Q5_0` reference dot over the sixteen lanes. -/
def q5DotRef (w : Nat) (qs : List U8) (xq : List I8) : Int := q5Sum w qs xq 16

/-- The four-byte little-endian `qh` word of a block, as the extractor's `U32::from_le_bytes`
produces it: byte 0 is the least significant. -/
def q5Word4 (b0 b1 b2 b3 : U8) : Nat :=
  (BitVec.fromLEBytes [b0.bv, b1.bv, b2.bv, b3.bv]).toNat

theorem q5Sum_succ (w : Nat) (qs : List U8) (xq : List I8) (n : Nat) :
    q5Sum w qs xq (n + 1) = q5Sum w qs xq n + q5Term w qs xq n := rfl

/-! ## Value bounds -/

theorem i8_val_bounds (x : I8) : -128 ≤ x.val ∧ x.val ≤ 127 := by
  have hb := x.hBounds
  have hbits : IScalarTy.I8.numBits = 8 := by decide
  rw [hbits] at hb
  constructor
  · norm_num [IScalar.min] at hb ⊢
    omega
  · norm_num [IScalar.max] at hb ⊢
    omega

theorem q5Bit_le_one (w t : Nat) : q5Bit w t ≤ 1 := Nat.and_le_right

theorem q5Low5_lt (nib bit : Nat) (hb : bit ≤ 1) : q5Low5 nib bit < 32 := by
  have hand : nib &&& 15 < 32 := by have := Nat.and_le_right (n := nib) (m := 15); omega
  have hsh : bit <<< 4 ≤ 16 := by rw [Nat.shiftLeft_eq]; nlinarith [hb]
  have hmod : (bit <<< 4) % 256 < 32 := by have := Nat.mod_le (bit <<< 4) 256; omega
  have h := Nat.or_lt_two_pow (n := 5) hand hmod
  have h5 : (2 : Nat) ^ 5 = 32 := by norm_num
  simp only [q5Low5]
  rwa [h5] at h

theorem q5High5_lt (nib bit : Nat) (hn : nib < 256) (hb : bit ≤ 1) : q5High5 nib bit < 32 := by
  have hshr : nib >>> 4 < 32 := by rw [Nat.shiftRight_eq_div_pow]; omega
  have hsh : bit <<< 4 ≤ 16 := by rw [Nat.shiftLeft_eq]; nlinarith [hb]
  have hmod : (bit <<< 4) % 256 < 32 := by have := Nat.mod_le (bit <<< 4) 256; omega
  have h := Nat.or_lt_two_pow (n := 5) hshr hmod
  have h5 : (2 : Nat) ^ 5 = 32 := by norm_num
  simp only [q5High5]
  rwa [h5] at h

theorem q5Code_bounds (five : Nat) (h : five < 32) : -16 ≤ q5Code five ∧ q5Code five ≤ 15 := by
  unfold q5Code
  omega

theorem q5Term_bounds (w : Nat) (qs : List U8) (xq : List I8) (j : Nat) :
    -4096 ≤ q5Term w qs xq j ∧ q5Term w qs xq j ≤ 4096 := by
  have hlo : -16 ≤ q5Code (q5Low5 (qs.getD j 0#u8).val (q5Bit w j)) ∧
      q5Code (q5Low5 (qs.getD j 0#u8).val (q5Bit w j)) ≤ 15 :=
    q5Code_bounds _ (q5Low5_lt _ _ (q5Bit_le_one w j))
  have hhi : -16 ≤ q5Code (q5High5 (qs.getD j 0#u8).val (q5Bit w (16 + j))) ∧
      q5Code (q5High5 (qs.getD j 0#u8).val (q5Bit w (16 + j))) ≤ 15 :=
    q5Code_bounds _ (q5High5_lt _ _ (U8.lt_succ_max _) (q5Bit_le_one w (16 + j)))
  have hx1 := i8_val_bounds (xq.getD j 0#i8)
  have hx2 := i8_val_bounds (xq.getD (16 + j) 0#i8)
  unfold q5Term
  constructor <;> nlinarith [hlo.1, hlo.2, hhi.1, hhi.2, hx1.1, hx1.2, hx2.1, hx2.2]

theorem q5Sum_bounds (w : Nat) (qs : List U8) (xq : List I8) (n : Nat) :
    -4096 * (n : Int) ≤ q5Sum w qs xq n ∧ q5Sum w qs xq n ≤ 4096 * (n : Int) := by
  induction n with
  | zero => simp only [q5Sum]; norm_num
  | succ k ih =>
    rw [q5Sum_succ]
    have ht := q5Term_bounds w qs xq k
    constructor <;> push_cast <;> nlinarith [ih.1, ih.2, ht.1, ht.2]

/-! ## Checked-arithmetic range helpers

The extracted `i32` `-`, `*` and `+` are checked: `step*` leaves the `I32.min`/`I32.max` range goals
for the `I32.Insts.CoreOpsArith*` operations. The bounds below show every accumulator stays inside
`2^31`, so no overflow branch is reachable on the stated domain. -/

/-- The subtraction `x - 16` on the `i32` cast of a byte does not overflow: the upper bound. -/
theorem hcast_u8_I32_sub16_upper (x : U8) :
    (UScalar.hcast IScalarTy.I32 x).val - (16#i32).val ≤ I32.max := by
  rw [hcast_u8_I32_val, show (16#i32 : I32).val = 16 from by norm_num, I32.max_eq]
  have := U8.lt_succ_max x
  omega

/-- `↑x - 16` for a byte `x` lies in `[-16, 239]`. -/
theorem hcast_sub16_bounds (x : U8) :
    -16 ≤ (UScalar.hcast IScalarTy.I32 x).val - 16 ∧
      (UScalar.hcast IScalarTy.I32 x).val - 16 ≤ 239 := by
  rw [hcast_u8_I32_val]
  have hx := U8.lt_succ_max x
  constructor <;> omega

/-- `|a · b| ≤ A · B` from `|a| ≤ A`, `|b| ≤ B` and `0 ≤ A`. -/
theorem abs_mul_le_abs {a b A B : ℤ} (hA : 0 ≤ A) (ha : |a| ≤ A) (hb : |b| ≤ B) :
    |a * b| ≤ A * B := by
  rw [abs_mul]
  exact mul_le_mul ha hb (abs_nonneg b) hA

/-- The product of a `[-16, 239]` lane code and a `[-128, 127]` activation is within `±30592`,
the worst-case `i16`-scale product the `Q5_0` lane multiplies. -/
theorem abs_mul_lo_le (lo : I32) (x : I8) (h1 : -16 ≤ (↑lo : ℤ)) (h2 : (↑lo : ℤ) ≤ 239) :
    |(↑lo : ℤ) * (↑x : ℤ)| ≤ 30592 := by
  rw [abs_mul]
  have ha : |(↑lo : ℤ)| ≤ 239 := abs_le.mpr ⟨by omega, h2⟩
  have hb : |(↑x : ℤ)| ≤ 128 := by
    have h := i8_val_bounds x
    exact abs_le.mpr ⟨h.1, by omega⟩
  calc |(↑lo : ℤ)| * |(↑x : ℤ)| ≤ 239 * 128 := mul_le_mul ha hb (abs_nonneg _) (by norm_num)
    _ = 30592 := by norm_num

/-- The `U8` target of `UScalar.cast` is the low byte: for a `1`-bit source the cast is the bit. -/
theorem cast_bit_val (x : U32) :
    (↑(UScalar.cast UScalarTy.U8 (x &&& 1#u32)) : Nat) = (x.val &&& 1) := by
  rw [UScalar.cast_val_mod_pow_of_inBounds_eq]
  · rw [UScalar.val_and, show (1#u32 : U32).val = 1 from by norm_num]
  · rw [UScalar.val_and, show (1#u32 : U32).val = 1 from by norm_num]
    simp only [UScalarTy.U8_numBits_eq]
    have := Nat.and_le_right (n := x.val) (m := 1)
    omega

/-- The `U8` assembled by `nib & 0x0F | bit <<< 4` is the reference low-plane five-bit code. -/
theorem lo5_val (nib i6 : U8) (v t : Nat) (hn : (↑nib : Nat) = v)
    (h6 : (↑i6 : Nat) = (t <<< 4) % U8.size) :
    ((nib &&& 15#u8 ||| i6).val : Nat) = q5Low5 v t := by
  unfold q5Low5
  rw [UScalar.val_or, UScalar.val_and, hn,
    show (15#u8 : U8).val = 15 from by norm_num, h6, U8.size_eq]

/-- The `U8` assembled by `nib >>> 4 | bit <<< 4` is the reference high-plane five-bit code. -/
theorem hi5_val (i9 i10 : U8) (v t : Nat)
    (h9 : (↑i9 : Nat) = v >>> 4) (h10 : (↑i10 : Nat) = (t <<< 4) % U8.size) :
    ((i9 ||| i10).val : Nat) = q5High5 v t := by
  unfold q5High5
  rw [UScalar.val_or, h9, h10, U8.size_eq]

/-- In-bounds `Slice.getElem` of an `I8` slice is its `getD` (the U8 version is
`Numeric.Packing.Format.q4_getD_getElem`). -/
theorem i8_getD_getElem (xs : Slice I8) (i : Usize) (h : i.val < xs.val.length) :
    (xs.val[i.val]'h).val = (xs.val.getD i.val 0#i8).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

/-! ## The loop invariant -/

/-- The `portable_q5_0_dot` loop state: lane counter `j` (at most `16`) and an accumulator that is
the reference prefix sum. -/
def q5LoopInv (w : Nat) (qs : Slice U8) (xq : Slice I8) (st : I32 × Usize) : Prop :=
  st.2.val ≤ 16 ∧ st.1.val = q5Sum w qs.val xq.val st.2.val

/-- The `u8 → i32` cast the body uses for the assembled five-bit codes is the value cast. -/
theorem i32_from_u8_eq (x : U8) :
    I32.Insts.CoreConvertFromU8.from x = ok (UScalar.hcast IScalarTy.I32 x) := rfl

/-! ## One `portable_q5_0_dot` step

The extracted body reads `qs[j]` and the two `qh` plane bits, assembles the low and high five-bit
codes, multiplies each by the matching activation lane `xq[j]`/`xq[16 + j]`, adds both into the
accumulator and advances `j`. On the stated in-bounds domain every checked `i32` operation stays in
range, so the only outcomes are the finished dot (`j = 16`) and the next prefix accumulator. -/

set_option maxHeartbeats 400000 in
theorem q5_dot_loop_body_spec (qs : Slice U8) (xq : Slice I8) (qh : U32)
    (hqs : 16 ≤ qs.val.length) (hxq : 32 ≤ xq.val.length)
    (acc : I32) (j : Usize)
    (hinv : q5LoopInv qh.val qs xq (acc, j)) :
    kern.matmat.portable_q5_0_dot_loop.body qs xq qh acc j ⦃ r =>
      match r with
      | ControlFlow.done y => y.val = q5DotRef qh.val qs.val xq.val
      | ControlFlow.cont st' => q5LoopInv qh.val qs xq st' ∧
          16 - st'.2.val < 16 - j.val ⦄ := by
  simp only [q5LoopInv] at hinv
  obtain ⟨hle, hacc⟩ := hinv
  by_cases hlt : j < 16#usize
  · have hj : (↑j : Nat) < 16 := by
      have h := (UScalar.lt_equiv j 16#usize).1 hlt
      have h16 : ((16#usize : Usize).val) = 16 := by norm_num
      omega
    unfold kern.matmat.portable_q5_0_dot_loop.body
    rw [if_pos hlt]
    simp only [lift, bind_tc_ok, i32_from_u8_eq]
    step*
    · simpa using hcast_u8_I32_sub16_lower (nib &&& 15#u8 ||| i6)
    · simpa using hcast_u8_I32_sub16_upper (nib &&& 15#u8 ||| i6)
    · simpa using hcast_u8_I32_sub16_lower (i9 ||| i10)
    · simpa using hcast_u8_I32_sub16_upper (i9 ||| i10)
    · rw [core.convert.num.FromI32I8.from_val_eq]
      have h := abs_mul_lo_le lo i13 (by rw [lo_post]; exact (hcast_sub16_bounds _).1)
        (by rw [lo_post]; exact (hcast_sub16_bounds _).2)
      have h3 := neg_abs_le ((↑lo : ℤ) * (↑i13 : ℤ))
      rw [I32.min_eq]
      omega
    · rw [core.convert.num.FromI32I8.from_val_eq]
      have h := abs_mul_lo_le lo i13 (by rw [lo_post]; exact (hcast_sub16_bounds _).1)
        (by rw [lo_post]; exact (hcast_sub16_bounds _).2)
      have h3 := le_abs_self ((↑lo : ℤ) * (↑i13 : ℤ))
      rw [I32.max_eq]
      omega
    · have h15 : |(↑i15 : ℤ)| ≤ 30592 := by
        rw [i15_post, core.convert.num.FromI32I8.from_val_eq]
        exact abs_mul_lo_le lo i13 (by rw [lo_post]; exact (hcast_sub16_bounds _).1)
          (by rw [lo_post]; exact (hcast_sub16_bounds _).2)
      rw [hacc, I32.min_eq]
      have hq := (q5Sum_bounds (↑qh) (↑qs) (↑xq) (↑j)).1
      have h3 := neg_abs_le (↑i15 : ℤ)
      omega
    · have h15 : |(↑i15 : ℤ)| ≤ 30592 := by
        rw [i15_post, core.convert.num.FromI32I8.from_val_eq]
        exact abs_mul_lo_le lo i13 (by rw [lo_post]; exact (hcast_sub16_bounds _).1)
          (by rw [lo_post]; exact (hcast_sub16_bounds _).2)
      rw [hacc, I32.max_eq]
      have hq := (q5Sum_bounds (↑qh) (↑qs) (↑xq) (↑j)).2
      have h3 := le_abs_self (↑i15 : ℤ)
      omega
    · rw [core.convert.num.FromI32I8.from_val_eq]
      have h := abs_mul_lo_le hi i16 (by rw [hi_post]; exact (hcast_sub16_bounds _).1)
        (by rw [hi_post]; exact (hcast_sub16_bounds _).2)
      have h3 := neg_abs_le ((↑hi : ℤ) * (↑i16 : ℤ))
      rw [I32.min_eq]
      omega
    · rw [core.convert.num.FromI32I8.from_val_eq]
      have h := abs_mul_lo_le hi i16 (by rw [hi_post]; exact (hcast_sub16_bounds _).1)
        (by rw [hi_post]; exact (hcast_sub16_bounds _).2)
      have h3 := le_abs_self ((↑hi : ℤ) * (↑i16 : ℤ))
      rw [I32.max_eq]
      omega
    · have h15 : |(↑i15 : ℤ)| ≤ 30592 := by
        rw [i15_post, core.convert.num.FromI32I8.from_val_eq]
        exact abs_mul_lo_le lo i13 (by rw [lo_post]; exact (hcast_sub16_bounds _).1)
          (by rw [lo_post]; exact (hcast_sub16_bounds _).2)
      have h18 : |(↑i18 : ℤ)| ≤ 30592 := by
        rw [i18_post, core.convert.num.FromI32I8.from_val_eq]
        exact abs_mul_lo_le hi i16 (by rw [hi_post]; exact (hcast_sub16_bounds _).1)
          (by rw [hi_post]; exact (hcast_sub16_bounds _).2)
      rw [acc1_post, hacc, I32.min_eq]
      have hq := (q5Sum_bounds (↑qh) (↑qs) (↑xq) (↑j)).1
      have h3 := neg_abs_le (↑i15 : ℤ)
      have h4 := neg_abs_le (↑i18 : ℤ)
      omega
    · have h15 : |(↑i15 : ℤ)| ≤ 30592 := by
        rw [i15_post, core.convert.num.FromI32I8.from_val_eq]
        exact abs_mul_lo_le lo i13 (by rw [lo_post]; exact (hcast_sub16_bounds _).1)
          (by rw [lo_post]; exact (hcast_sub16_bounds _).2)
      have h18 : |(↑i18 : ℤ)| ≤ 30592 := by
        rw [i18_post, core.convert.num.FromI32I8.from_val_eq]
        exact abs_mul_lo_le hi i16 (by rw [hi_post]; exact (hcast_sub16_bounds _).1)
          (by rw [hi_post]; exact (hcast_sub16_bounds _).2)
      rw [acc1_post, hacc, I32.max_eq]
      have hq := (q5Sum_bounds (↑qh) (↑qs) (↑xq) (↑j)).2
      have h3 := le_abs_self (↑i15 : ℤ)
      have h4 := le_abs_self (↑i18 : ℤ)
      omega
    · refine ⟨⟨?_, ?_⟩, ?_⟩
      · rw [j1_post]; omega
      · have hnib : (↑nib : Nat) = (qs.val.getD (↑j) 0#u8).val := by
          have hltj : (↑j : Nat) < qs.val.length := by omega
          rw [nib_post]
          exact q4_getD_getElem qs j hltj
        have hbit : (↑(UScalar.cast UScalarTy.U8 (i &&& 1#u32)) : Nat) = q5Bit (↑qh) (↑j) := by
          rw [cast_bit_val, i_post]
          rfl
        have hi6 : (↑i6 : Nat) = (q5Bit (↑qh) (↑j) <<< 4) % U8.size := by
          rw [i6_post, hbit]
        have hlo : (↑lo : ℤ) =
            (q5Low5 (qs.val.getD (↑j) 0#u8).val (q5Bit (↑qh) (↑j)) : ℤ) - 16 := by
          rw [lo_post, hcast_u8_I32_val]
          rw [lo5_val nib i6 (qs.val.getD (↑j) 0#u8).val (q5Bit (↑qh) (↑j)) hnib hi6]
        have h9 : (↑i9 : Nat) = (qs.val.getD (↑j) 0#u8).val >>> 4 := by rw [i9_post, hnib]
        have hbit3 : (↑(UScalar.cast UScalarTy.U8 (i3 &&& 1#u32)) : Nat) =
            q5Bit (↑qh) (16 + (↑j)) := by
          rw [cast_bit_val, i3_post, i2_post]
          rfl
        have hi10 : (↑i10 : Nat) = (q5Bit (↑qh) (16 + (↑j)) <<< 4) % U8.size := by
          rw [i10_post, hbit3]
        have hhi : (↑hi : ℤ) =
            (q5High5 (qs.val.getD (↑j) 0#u8).val (q5Bit (↑qh) (16 + (↑j))) : ℤ) - 16 := by
          rw [hi_post, hcast_u8_I32_val]
          rw [hi5_val i9 i10 (qs.val.getD (↑j) 0#u8).val (q5Bit (↑qh) (16 + (↑j))) h9 hi10]
        have hx1 : (↑(core.convert.num.FromI32I8.from i13) : ℤ) =
            (xq.val.getD (↑j) 0#i8).val := by
          rw [core.convert.num.FromI32I8.from_val_eq]
          have hltj : (↑j : Nat) < xq.val.length := by omega
          rw [i13_post]
          exact i8_getD_getElem xq j hltj
        have hx2 : (↑(core.convert.num.FromI32I8.from i16) : ℤ) =
            (xq.val.getD (16 + (↑j)) 0#i8).val := by
          rw [core.convert.num.FromI32I8.from_val_eq]
          have hlt2 : (↑i2 : Nat) < xq.val.length := by rw [i2_post]; omega
          have h1 : (↑i16 : ℤ) = (xq.val.getD (↑i2) 0#i8).val := by
            rw [i16_post]
            exact i8_getD_getElem xq i2 hlt2
          rw [h1, i2_post]
        rw [j1_post, q5Sum_succ, acc2_post, acc1_post, hacc]
        rw [add_assoc]
        congr 1
        rw [i15_post, i18_post, hlo, hhi, hx1, hx2]
        unfold q5Term q5Code
        ring
      · rw [j1_post]; omega
  · unfold kern.matmat.portable_q5_0_dot_loop.body
    rw [if_neg hlt]
    simp only [WP.spec_ok]
    have hge : 16 ≤ (↑j : Nat) := by
      have h := (UScalar.lt_equiv j 16#usize).not.mp hlt
      have h16 : ((16#usize : Usize).val) = 16 := by norm_num
      omega
    have hj16 : (↑j : Nat) = 16 := by omega
    rw [hacc, hj16]
    rfl

/-! ## The lane loop

`portable_q5_0_dot_loop` runs the step above to `j = 16`; the measure `16 - j` decreases by one
each step.

Complexity: 16 iterations, each O(1) (two byte reads, four bit operations, two casts, two
multiplies, two adds). -/

theorem word_portable_q5_0_dot_loop_ok (qs : Slice U8) (xq : Slice I8) (qh : U32)
    (hqs : 16 ≤ qs.val.length) (hxq : 32 ≤ xq.val.length) (acc : I32) (j : Usize)
    (hinv : q5LoopInv qh.val qs xq (acc, j)) :
    kern.matmat.portable_q5_0_dot_loop qs xq qh acc j
      ⦃ r => r.val = q5DotRef qh.val qs.val xq.val ⦄ := by
  unfold kern.matmat.portable_q5_0_dot_loop
  apply loop.spec_decr_nat
    (measure := fun s : I32 × Usize => 16 - s.2.val)
    (inv := fun s => q5LoopInv qh.val qs xq s)
    (post := fun y : I32 => y.val = q5DotRef qh.val qs.val xq.val)
  · rintro ⟨acc', j'⟩ hinv'
    dsimp only at hinv' ⊢
    obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists
      (q5_dot_loop_body_spec qs xq qh hqs hxq acc' j' hinv')
    rw [hr]
    rcases r with y | s'
    · simpa only [WP.spec_ok] using hpost
    · simpa only [WP.spec_ok] using hpost
  · exact hinv

/-! ## The `qh` word -/

/-- The extracted `U32::from_le_bytes` of four bytes is the reference `q5Word4`. -/
theorem u32_from_le_bytes_val (b0 b1 b2 b3 : U8) :
    (core.num.U32.from_le_bytes (Array.make 4#usize [b0, b1, b2, b3])).val
      = q5Word4 b0 b1 b2 b3 := by
  rw [← UScalar.bv_toNat]
  unfold core.num.U32.from_le_bytes
  rw [BitVec.toNat_cast]
  rw [Array.make_val]
  simp only [List.map_cons, List.map_nil, q5Word4]

/-! ## The whole `portable_q5_0_dot`

It reads the four `qh` bytes, assembles the little-endian word and runs the lane loop from `0`. -/

theorem word_portable_q5_0_dot_ok (qh qs : Slice U8) (xq : Slice I8)
    (hqh : 4 ≤ qh.val.length) (hqs : 16 ≤ qs.val.length) (hxq : 32 ≤ xq.val.length) :
    kern.matmat.portable_q5_0_dot qh qs xq ⦃ r =>
      r.val = q5DotRef (q5Word4 (qh.val.getD 0 0#u8) (qh.val.getD 1 0#u8)
        (qh.val.getD 2 0#u8) (qh.val.getD 3 0#u8)) qs.val xq.val ⦄ := by
  unfold kern.matmat.portable_q5_0_dot
  refine WP.spec_bind (Slice.index_usize_spec qh 0#usize (by norm_num; omega)) ?_
  intro i hi
  refine WP.spec_bind (Slice.index_usize_spec qh 1#usize (by norm_num; omega)) ?_
  intro i1 hi1
  refine WP.spec_bind (Slice.index_usize_spec qh 2#usize (by norm_num; omega)) ?_
  intro i2 hi2
  refine WP.spec_bind (Slice.index_usize_spec qh 3#usize (by norm_num; omega)) ?_
  intro i3 hi3
  refine WP.spec_bind
    (core.num.U32.from_le_bytes.step_spec (Array.make 4#usize [i, i1, i2, i3])) ?_
  intro qh_u32 hq
  refine WP.spec_mono (word_portable_q5_0_dot_loop_ok qs xq qh_u32 hqs hxq 0#i32 0#usize ?_) ?_
  · refine ⟨by norm_num, ?_⟩
    norm_num [q5Sum]
  · intro r hr
    rw [hr]
    congr 1
    have hb0 : i = qh.val.getD 0 0#u8 := by
      apply UScalar.eq_of_val_eq
      rw [hi]
      exact q4_getD_getElem qh 0#usize (by norm_num; omega)
    have hb1 : i1 = qh.val.getD 1 0#u8 := by
      apply UScalar.eq_of_val_eq
      rw [hi1]
      exact q4_getD_getElem qh 1#usize (by norm_num; omega)
    have hb2 : i2 = qh.val.getD 2 0#u8 := by
      apply UScalar.eq_of_val_eq
      rw [hi2]
      exact q4_getD_getElem qh 2#usize (by norm_num; omega)
    have hb3 : i3 = qh.val.getD 3 0#u8 := by
      apply UScalar.eq_of_val_eq
      rw [hi3]
      exact q4_getD_getElem qh 3#usize (by norm_num; omega)
    have hqval : qh_u32.val = q5Word4 i i1 i2 i3 := by
      rw [← UScalar.bv_toNat]
      simp only [hq, BitVec.toNat_cast]
      rw [Array.make_val]
      simp only [List.map_cons, List.map_nil, q5Word4]
    rw [hqval, hb0, hb1, hb2, hb3]

/-! ## Refusal

On a `qh` slice shorter than four bytes the extracted `portable_q5_0_dot` refuses with
`arrayOutOfBounds`: one of its four `qh` reads is out of bounds. -/

/-- `Slice.index_usize` refuses with `arrayOutOfBounds` past the end. -/
theorem slice_index_usize_fail {α : Type} (v : Slice α) (i : Usize)
    (h : v.val.length ≤ i.val) : v.index_usize i = fail .arrayOutOfBounds := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_none (by simpa using h)]

/-- `Slice.index_usize` returns the indexed element in bounds. -/
theorem slice_index_usize_ok {α : Type} (v : Slice α) (i : Usize)
    (h : i.val < v.val.length) : v.index_usize i = ok (v.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec v i h)
  rw [hx, hxv]

theorem word_portable_q5_0_dot_refuses (qh qs : Slice U8) (xq : Slice I8)
    (h : qh.val.length < 4) :
    kern.matmat.portable_q5_0_dot qh qs xq = fail .arrayOutOfBounds := by
  unfold kern.matmat.portable_q5_0_dot
  have hlen : qh.val.length = 0 ∨ qh.val.length = 1 ∨ qh.val.length = 2 ∨
      qh.val.length = 3 := by omega
  rcases hlen with h0 | h1 | h2 | h3
  · rw [slice_index_usize_fail qh 0#usize (by rw [h0]; norm_num)]
    simp only [bind_tc_fail]
  · rw [slice_index_usize_ok qh 0#usize (by rw [h1]; norm_num)]
    simp only [bind_tc_ok]
    rw [slice_index_usize_fail qh 1#usize (by rw [h1]; norm_num)]
    simp only [bind_tc_fail]
  · rw [slice_index_usize_ok qh 0#usize (by rw [h2]; norm_num)]
    simp only [bind_tc_ok]
    rw [slice_index_usize_ok qh 1#usize (by rw [h2]; norm_num)]
    simp only [bind_tc_ok]
    rw [slice_index_usize_fail qh 2#usize (by rw [h2]; norm_num)]
    simp only [bind_tc_fail]
  · rw [slice_index_usize_ok qh 0#usize (by rw [h3]; norm_num)]
    simp only [bind_tc_ok]
    rw [slice_index_usize_ok qh 1#usize (by rw [h3]; norm_num)]
    simp only [bind_tc_ok]
    rw [slice_index_usize_ok qh 2#usize (by rw [h3]; norm_num)]
    simp only [bind_tc_ok]
    rw [slice_index_usize_fail qh 3#usize (by rw [h3]; norm_num)]
    simp only [bind_tc_fail]

/-! ## Non-vacuity of the reference -/

example : q5Low5 3 1 = 19 := by decide

example : q5High5 255 1 = 31 := by decide

example : q5Bit 1 0 = 1 := by decide

example : q5Bit 1 1 = 0 := by decide

example : q5Code 31 = 15 := by decide

example : q5Code 0 = -16 := by decide

/-- The sixteen-lane reference dot on the constant word `1` and constant lanes `1`: lane `0` takes
the low plane bit, so the sum is `-15 + 15·(-31)`. -/
example : q5DotRef 1 (List.replicate 16 1#u8) (List.replicate 32 1#i8) = -480 := by decide

/-- The refusal theorem has a witness: the empty `qh` slice refuses. -/
example : kern.matmat.portable_q5_0_dot (Slice.from [] (by simp)) (Slice.from [] (by simp))
    (Slice.from [] (by simp)) = fail .arrayOutOfBounds :=
  word_portable_q5_0_dot_refuses _ _ _ (by simp)

/-! ## Axiom cone -/

#print axioms word_portable_q5_0_dot_loop_ok
#print axioms word_portable_q5_0_dot_ok
#print axioms word_portable_q5_0_dot_refuses
#print axioms q5_dot_loop_body_spec

end apeiron_word
