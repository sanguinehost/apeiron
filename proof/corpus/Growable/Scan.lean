import ApeironWord
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases

/-!
# Exact scalar tag scan

The scalar SWAR kernel marks a lane after setting every byte's high bit.  Consequently subtraction
of the repeated one byte cannot borrow across a byte boundary.  The proofs below isolate the only
finite calculation at width eight; word composition is structural.
-/

namespace Apeiron.GrowableScan

open Aeneas Aeneas.Std Result
open apeiron_word

/-- Fixed-width trailing-zero semantics: zero returns the word width; a nonzero input returns the
least exponent whose corresponding bit is set. -/
def FixedWidthTrailingZeros (value : Usize) (zeros : U32) : Prop :=
  (value.val = 0 ∧ zeros.val = System.Platform.numBits) ∨
    (value.val ≠ 0 ∧ zeros.val < System.Platform.numBits ∧
      2 ^ zeros.val ∣ value.val ∧ ¬2 ^ (zeros.val + 1) ∣ value.val)

def TrailingZerosInv (original : Usize) (state : Usize × U32) : Prop :=
  state.2.val ≤ System.Platform.numBits ∧ original.val = state.1.val * 2 ^ state.2.val

theorem usize_and_one_val (value : Usize) :
    (value &&& 1#usize).val = value.val % 2 := by
  change (value.bv &&& (1#usize).bv).toNat = value.bv.toNat % 2
  rw [BitVec.toNat_and]
  have hone : (1#usize).bv.toNat = 1 := by rfl
  rw [hone, Nat.and_one_is_mod]

/-- The extracted bounded loop returns the word width at zero and the least set-bit index otherwise. -/
theorem summary_trailing_zeros_eq (value : Usize) :
    table.summary_trailing_zeros value ⦃ zeros => FixedWidthTrailingZeros value zeros ⦄ := by
  unfold table.summary_trailing_zeros table.summary_trailing_zeros_loop
  apply loop.spec_decr_nat
    (measure := fun state : Usize × U32 => System.Platform.numBits - state.2.val)
    (inv := TrailingZerosInv value)
    (post := FixedWidthTrailingZeros value)
  · intro state hstate
    rcases state with ⟨current, zeros⟩
    rcases hstate with ⟨hzeros, hvalue⟩
    dsimp only at hzeros hvalue ⊢
    unfold table.summary_trailing_zeros_loop.body
    by_cases hwidth : zeros < core.num.Usize.BITS
    · rw [if_pos hwidth]
      have hzerosLt : zeros.val < System.Platform.numBits := by simpa using hwidth
      simp only [lift, bind_tc_ok]
      by_cases heven : current &&& 1#usize = 0#usize
      · rw [if_pos heven]
        have hzeroAdd : zeros.val + 1 ≤ U32.max := by
          have := System.Platform.numBits_eq
          rw [U32.max_eq]
          omega
        step
        have honeWidth : 1 < System.Platform.numBits := by
          cases System.Platform.numBits_eq <;> omega
        step
        have hmod : current.val % 2 = 0 := by
          rw [← usize_and_one_val]
          exact congrArg UScalar.val heven
        have hcurrent : current.val = current.val / 2 * 2 := by omega
        refine ⟨⟨?_, ?_⟩, ?_⟩
        · simpa [zeros1_post] using Nat.succ_le_of_lt hzerosLt
        · rw [value1_post, Nat.shiftRight_eq_div_pow]
          simp only [pow_one]
          rw [zeros1_post, pow_succ]
          calc
            value.val = current.val * 2 ^ zeros.val := hvalue
            _ = (current.val / 2 * 2) * 2 ^ zeros.val := by rw [← hcurrent]
            _ = current.val / 2 * 2 ^ zeros.val * 2 := by ac_rfl
            _ = current.val / 2 * (2 ^ zeros.val * 2) :=
              Nat.mul_assoc (current.val / 2) (2 ^ zeros.val) 2
        · simp [zeros1_post]
          omega
      · rw [if_neg heven, Aeneas.Std.WP.spec_ok]
        right
        have hmod : current.val % 2 ≠ 0 := by
          intro hzero
          apply heven
          apply UScalar.eq_of_val_eq
          rw [usize_and_one_val]
          exact hzero
        refine ⟨?_, hzerosLt, ⟨current.val, by simpa [Nat.mul_comm] using hvalue⟩, ?_⟩
        · intro hzero
          rw [hzero] at hvalue
          simp at hvalue
          omega
        · rw [pow_succ]
          intro hdvd
          rw [hvalue] at hdvd
          have hevenCurrent : 2 ∣ current.val := by
            apply (Nat.mul_dvd_mul_iff_right (by positivity : 0 < 2 ^ zeros.val)).mp
            simpa only [Nat.mul_comm (2 ^ zeros.val) 2] using hdvd
          exact hmod (Nat.dvd_iff_mod_eq_zero.mp hevenCurrent)
    · rw [if_neg hwidth, Aeneas.Std.WP.spec_ok]
      left
      have hzerosEq : zeros.val = System.Platform.numBits := by
        have : ¬zeros.val < System.Platform.numBits := by simpa using hwidth
        omega
      refine ⟨?_, hzerosEq⟩
      rw [hzerosEq] at hvalue
      by_cases hcurrentZero : current.val = 0
      · rw [hcurrentZero] at hvalue
        simpa using hvalue
      · have hbound : value.val < 2 ^ System.Platform.numBits := by
          simpa [UScalar.size] using value.hBounds
        have hcurrent : 1 ≤ current.val := Nat.one_le_iff_ne_zero.mpr hcurrentZero
        have hpow : 2 ^ System.Platform.numBits ≤ current.val * 2 ^ System.Platform.numBits := by
          simpa using Nat.mul_le_mul_right (2 ^ System.Platform.numBits) hcurrent
        rw [hvalue] at hbound
        omega
  · exact ⟨Nat.zero_le _, by simp⟩

private theorem append_or {n m : Nat} (a c : BitVec n) (b d : BitVec m) :
    (a ++ b) ||| (c ++ d) = (a ||| c) ++ (b ||| d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_or, BitVec.getLsbD_append]
  split <;> rfl

private theorem append_and {n m : Nat} (a c : BitVec n) (b d : BitVec m) :
    (a ++ b) &&& (c ++ d) = (a &&& c) ++ (b &&& d) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_and, BitVec.getLsbD_append]
  split <;> rfl

private theorem append_not {n m : Nat} (a : BitVec n) (b : BitVec m) :
    ~~~(a ++ b) = (~~~a) ++ (~~~b) := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_not, BitVec.getLsbD_append]
  split
  · simp_all
  · have : i - m < n := by omega
    simp_all

private theorem toNat_append_byte {n : Nat} (a : BitVec n) (b : BitVec 8) :
    (a ++ b).toNat = a.toNat * 256 + b.toNat := by
  rw [BitVec.toNat_append, ← Nat.shiftLeft_add_eq_or_of_lt b.isLt]
  simp [Nat.shiftLeft_eq]

/-- Subtraction splits at an eight-bit boundary when the low byte cannot borrow. -/
theorem append_sub_byte_noBorrow {n : Nat} (a c : BitVec n) (b d : BitVec 8)
    (h : d.toNat ≤ b.toNat) :
    (a ++ b) - (c ++ d) = (a - c) ++ (b - d) := by
  apply BitVec.toNat_inj.mp
  simp only [BitVec.toNat_sub, toNat_append_byte]
  rw [pow_add]
  norm_num
  have ha := a.isLt
  have hc := c.isLt
  have hb := b.isLt
  have hd := d.isLt
  have hlow : (256 - d.toNat + b.toNat) % 256 = b.toNat - d.toNat := by
    have heq : 256 - d.toNat + b.toNat = 256 + (b.toNat - d.toNat) := by omega
    rw [heq, Nat.add_mod]
    simp
    omega
  rw [hlow]
  have hnum :
      2 ^ n * 256 - (c.toNat * 256 + d.toNat) + (a.toNat * 256 + b.toNat) =
        (2 ^ n - c.toNat + a.toNat) * 256 + (b.toNat - d.toNat) := by omega
  rw [hnum]
  rw [Nat.add_mod, Nat.mul_mod_mul_right]
  have hr : b.toNat - d.toNat < 2 ^ n * 256 := by
    have hp : 0 < 2 ^ n := pow_pos (by omega) _
    omega
  rw [Nat.mod_eq_of_lt hr]
  apply Nat.mod_eq_of_lt
  have hmod : (2 ^ n - c.toNat + a.toNat) % 2 ^ n < 2 ^ n :=
    Nat.mod_lt _ (pow_pos (by omega) _)
  omega

/-- The complete eight-bit zero-marker identity; this is the only finite bit-vector base case. -/
theorem byte_zero_marker (x : BitVec 8) :
    (~~~(x ||| ((x ||| 128#8) - 1#8))) &&& 128#8 =
      if x = 0#8 then 128#8 else 0#8 := by
  rcases x with ⟨⟨x, hx⟩⟩
  interval_cases x <;> decide +revert

private abbrev byteWidth (n : Nat) : Nat := Nat.rec 0 (fun _ width => width + 8) n

private def repeatByte (byte : BitVec 8) : (n : Nat) → BitVec (byteWidth n)
  | 0 => 0#0
  | n + 1 => repeatByte byte n ++ byte

private def lowByte {n : Nat} (x : BitVec (byteWidth n + 8)) : BitVec 8 :=
  BitVec.extractLsb' 0 8 x

private def upperBytes {n : Nat} (x : BitVec (byteWidth n + 8)) : BitVec (byteWidth n) :=
  BitVec.extractLsb' 8 (byteWidth n) x

private theorem upper_low_decompose {n : Nat} (x : BitVec (byteWidth n + 8)) :
    upperBytes x ++ lowByte x = x := by
  apply BitVec.eq_of_getLsbD_eq
  intro i hi
  simp only [BitVec.getLsbD_append]
  split
  · simp_all [lowByte, byteWidth]
  · have : i - 8 < byteWidth n := by omega
    simp_all [upperBytes, byteWidth]

private def mapBytes (f : BitVec 8 → BitVec 8) :
    (n : Nat) → BitVec (byteWidth n) → BitVec (byteWidth n)
  | 0, _ => 0#0
  | Nat.succ n, x => mapBytes f n (upperBytes x) ++ f (lowByte x)

private def zeroMarkerByte (x : BitVec 8) : BitVec 8 :=
  if x = 0#8 then 128#8 else 0#8

/-- Repeated-byte zero detection is the independent composition of the eight-bit identity. -/
theorem zero_marker_decomposition (n : Nat) (x : BitVec (byteWidth n)) :
    (~~~(x ||| ((x ||| repeatByte 128#8 n) - repeatByte 1#8 n))) &&&
        repeatByte 128#8 n = mapBytes zeroMarkerByte n x := by
  induction n with
  | zero =>
      apply BitVec.eq_of_getLsbD_eq
      intro i hi
      simp [byteWidth] at hi
  | succ n ih =>
      change BitVec (byteWidth n + 8) at x
      rw [← upper_low_decompose x]
      simp only [repeatByte, append_or]
      rw [append_sub_byte_noBorrow]
      · rw [append_or, append_not, append_and, ih, byte_zero_marker]
        simp [mapBytes, zeroMarkerByte, upperBytes, lowByte]
        rfl
      · apply Nat.one_le_iff_ne_zero.mpr
        intro hzero
        have hor : (lowByte x ||| 128#8) = 0#8 :=
          BitVec.toNat_inj.mp (by simpa using hzero)
        have := (BitVec.or_eq_zero_iff.mp hor).2
        exact (by norm_num : (128#8).toNat ≠ 0) (congrArg BitVec.toNat this)

/-- The lane-wise marker word denoted by `zero_swar`. -/
def zeroMarkers (word : U64) : U64 := ⟨mapBytes zeroMarkerByte 8 word.bv⟩

/-- The extracted 64-bit zero marker is eight applications of `byte_zero_marker`. -/
theorem extracted_zero_swar_exact (word : U64) :
    table.scan.zero_swar word = ok (zeroMarkers word) := by
  cases word with
  | mk bv =>
      unfold table.scan.zero_swar zeroMarkers
      simp [lift, table.scan.HI, table.scan.LO, core.num.U64.wrapping_sub,
        UScalar.wrapping_sub]
      apply congrArg (fun bits : BitVec 64 => (⟨bits⟩ : U64))
      have hhi : repeatByte 128#8 8 = 9259542123273814144#64 := by rfl
      have hlo : repeatByte 1#8 8 = 72340172838076673#64 := by rfl
      simpa [hhi, hlo] using zero_marker_decomposition 8 bv

private def byteSeries : Nat → Nat
  | 0 => 0
  | n + 1 => byteSeries n * 256 + 1

private theorem repeatByte_toNat (n : Nat) (byte : BitVec 8) :
    (repeatByte byte n).toNat = byte.toNat * byteSeries n := by
  induction n with
  | zero => simp [repeatByte, byteSeries]
  | succ n ih =>
      rw [repeatByte, toNat_append_byte, ih]
      simp only [byteSeries]
      ring

/-- Multiplication by the extracted `LO` constant broadcasts one byte without overflow. -/
theorem lo_mul_broadcast (byte : U8) :
    core.num.U64.wrapping_mul table.scan.LO (UScalar.cast .U64 byte) =
      (⟨repeatByte byte.bv 8⟩ : U64) := by
  apply UScalar.eq_of_val_eq
  simp only [core.num.U64.wrapping_mul_val_eq, UScalar.cast_val_eq, table.scan.LO]
  simp only [UScalar.val, UScalar.size, UScalarTy.numBits]
  rw [repeatByte_toNat]
  norm_num [byteSeries]
  have hlo : (72340172838076673#(UScalarTy.U64.numBits)).toNat =
      72340172838076673 := by norm_num [UScalarTy.numBits]
  rw [hlo]
  have hbyte : byte.val < 256 := by
    simpa [UScalar.size, UScalarTy.numBits] using byte.hBounds
  omega

/-- The lane-wise equality markers denoted by `eq_swar`. -/
def tagMarkers (word : U64) (byte : U8) : U64 :=
  ⟨mapBytes zeroMarkerByte 8 (word.bv ^^^ repeatByte byte.bv 8)⟩

/-- The extracted equality marker is byte broadcast, xor, then independent zero detection. -/
theorem extracted_eq_swar_exact (word : U64) (byte : U8) :
    table.scan.eq_swar word byte = ok (tagMarkers word byte) := by
  unfold table.scan.eq_swar tagMarkers
  have hbroadcast := congrArg UScalar.bv (lo_mul_broadcast byte)
  simp [lift, table.scan.HI, table.scan.LO,
    core.num.U64.wrapping_sub, UScalar.wrapping_sub]
  apply congrArg (fun bits : BitVec 64 => (⟨bits⟩ : U64))
  simp only [core.num.U64.wrapping_mul, UScalar.wrapping_mul, UScalar.cast] at hbroadcast
  have hbroadcast' :
      72340172838076673#64 * BitVec.setWidth 64 byte.bv = repeatByte byte.bv 8 := by
    simpa [table.scan.LO] using hbroadcast
  rw [hbroadcast']
  have hhi : repeatByte 128#8 8 = 9259542123273814144#64 := by rfl
  have hlo : repeatByte 1#8 8 = 72340172838076673#64 := by rfl
  simpa [hhi, hlo] using
    zero_marker_decomposition 8 (word.bv ^^^ repeatByte byte.bv 8)

private def markerNat (mask : BitVec 8) : Nat :=
  (mask.toNat % 2) * 128 +
  (mask.toNat / 2 % 2) * 32768 +
  (mask.toNat / 4 % 2) * 8388608 +
  (mask.toNat / 8 % 2) * 2147483648 +
  (mask.toNat / 16 % 2) * 549755813888 +
  (mask.toNat / 32 % 2) * 140737488355328 +
  (mask.toNat / 64 % 2) * 36028797018963968 +
  (mask.toNat / 128 % 2) * 9223372036854775808

/-- Spread the eight mask bits to bits `7 + 8i`, the marker position of lane `i`. -/
def markerWord (mask : BitVec 8) : U64 := ⟨BitVec.ofNat 64 (markerNat mask)⟩

private theorem eight_bits (x : Nat) (h : x < 256) :
    x % 2 + 2 * (x / 2 % 2) + 4 * (x / 4 % 2) + 8 * (x / 8 % 2) +
      16 * (x / 16 % 2) + 32 * (x / 32 % 2) + 64 * (x / 64 % 2) +
        128 * (x / 128 % 2) = x := by
  omega

private theorem gather_low_four (d₀ d₁ d₂ d₃ : Nat)
    (h₀ : d₀ < 2) (h₁ : d₁ < 2) (h₂ : d₂ < 2) (h₃ : d₃ < 2) :
    ((d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648) *
        567382630219905) % 18446744073709551616 =
      72057594037927936 * (d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃) +
        (d₀ * 567382630219904 + d₁ * 1134765260439552 +
          d₂ * 2269530520813568 + d₃ * 4539061024849920) := by
  omega

private theorem gather_high_four (d₄ d₅ d₆ d₇ : Nat)
    (h₄ : d₄ < 2) (h₅ : d₅ < 2) (h₆ : d₆ < 2) (h₇ : d₇ < 2) :
    ((d₄ * 549755813888 + d₅ * 140737488355328 + d₆ * 36028797018963968 +
        d₇ * 9223372036854775808) * 567382630219905) % 18446744073709551616 =
      72057594037927936 * (16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇) +
        (d₄ * 9078117754732544 + d₅ * 18155135997837312 +
          d₆ * 36028797018963968) := by
  omega

private theorem gather_eight_digits (d₀ d₁ d₂ d₃ d₄ d₅ d₆ d₇ : Nat)
    (h₀ : d₀ < 2) (h₁ : d₁ < 2) (h₂ : d₂ < 2) (h₃ : d₃ < 2)
    (h₄ : d₄ < 2) (h₅ : d₅ < 2) (h₆ : d₆ < 2) (h₇ : d₇ < 2) :
    ((d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648 +
        d₄ * 549755813888 + d₅ * 140737488355328 + d₆ * 36028797018963968 +
          d₇ * 9223372036854775808) * 567382630219905 % 18446744073709551616) /
      72057594037927936 =
        d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃ + 16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇ := by
  let lowMarker := d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648
  let highMarker := d₄ * 549755813888 + d₅ * 140737488355328 +
    d₆ * 36028797018963968 + d₇ * 9223372036854775808
  let lowBits := d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃
  let highBits := 16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇
  let lowRemainder := d₀ * 567382630219904 + d₁ * 1134765260439552 +
    d₂ * 2269530520813568 + d₃ * 4539061024849920
  let highRemainder := d₄ * 9078117754732544 + d₅ * 18155135997837312 +
    d₆ * 36028797018963968
  have hlo := gather_low_four d₀ d₁ d₂ d₃ h₀ h₁ h₂ h₃
  have hhi := gather_high_four d₄ d₅ d₆ d₇ h₄ h₅ h₆ h₇
  have hmarker :
      d₀ * 128 + d₁ * 32768 + d₂ * 8388608 + d₃ * 2147483648 +
          d₄ * 549755813888 + d₅ * 140737488355328 + d₆ * 36028797018963968 +
            d₇ * 9223372036854775808 = lowMarker + highMarker := by
    dsimp only [lowMarker, highMarker]
    omega
  have hbits :
      d₀ + 2 * d₁ + 4 * d₂ + 8 * d₃ + 16 * d₄ + 32 * d₅ + 64 * d₆ + 128 * d₇ =
        lowBits + highBits := by
    dsimp only [lowBits, highBits]
    omega
  rw [hmarker, hbits]
  have hremainder : lowRemainder + highRemainder < 72057594037927936 := by
    dsimp only [lowRemainder, highRemainder]
    omega
  have htotal :
      72057594037927936 * lowBits + lowRemainder +
          (72057594037927936 * highBits + highRemainder) <
        18446744073709551616 := by
    dsimp only [lowBits, highBits, lowRemainder, highRemainder]
    omega
  rw [Nat.add_mul, Nat.add_mod, hlo, hhi, Nat.mod_eq_of_lt htotal]
  have hsplit :
      72057594037927936 * lowBits + lowRemainder +
          (72057594037927936 * highBits + highRemainder) =
        72057594037927936 * (lowBits + highBits) +
          (lowRemainder + highRemainder) := by omega
  rw [hsplit, Nat.mul_add_div (by norm_num), Nat.div_eq_of_lt hremainder, Nat.add_zero]

private theorem gather_markerNat (mask : BitVec 8) :
    markerNat mask * 567382630219905 % 18446744073709551616 /
        72057594037927936 = mask.toNat := by
  have hmask : mask.toNat < 256 := by simpa using mask.isLt
  rw [← eight_bits mask.toNat hmask]
  unfold markerNat
  apply gather_eight_digits <;> omega

private theorem bind_two_ok {α β γ : Type} (x : α) (f : α → β) (g : β → γ) :
    (do let y ← ok x; let z ← ok (f y); ok (g z)) = ok (g (f x)) := by
  simp

private theorem lowEightOfShift (word : BitVec 64) (mask : BitVec 8)
    (h : (word.toNat >>> 56) % 256 = mask.toNat) :
    BitVec.setWidth 8 (word.ushiftRight 56) = mask := by
  apply BitVec.toNat_inj.mp
  rw [BitVec.toNat_setWidth, BitVec.ushiftRight_eq, BitVec.toNat_ushiftRight]
  norm_num only [pow_succ, pow_zero, Nat.mul_one]
  exact h

private theorem gather_markerBits (mask : BitVec 8) :
    BitVec.setWidth 8
        ((BitVec.ofNat 64 (markerNat mask) * 567382630219905#64).ushiftRight 56) = mask := by
  apply lowEightOfShift
  have hmask : mask.toNat < 256 := by simpa using mask.isLt
  have hgather : 567382630219905 % 18446744073709551616 = 567382630219905 := by
    norm_num
  calc
    ((BitVec.ofNat 64 (markerNat mask) * 567382630219905#64).toNat >>> 56) % 256 =
        (markerNat mask * 567382630219905 % 18446744073709551616 /
          72057594037927936) % 256 := by
            rw [BitVec.toNat_mul, BitVec.toNat_ofNat, BitVec.toNat_ofNat]
            norm_num only [pow_succ, pow_zero, Nat.mul_one]
            rw [← hgather, ← Nat.mul_mod, Nat.shiftRight_eq_div_pow]
    _ = mask.toNat % 256 := by rw [gather_markerNat]
    _ = mask.toNat := Nat.mod_eq_of_lt hmask

/-- The extracted multiplier gather is inverse to spreading on the lane-marker subspace. -/
theorem gather_spreadMarkerBits (mask : BitVec 8) :
    table.scan.gather (markerWord mask) = ok (⟨mask⟩ : U8) := by
  unfold table.scan.gather markerWord
  have hval : (56#i32).val = 56 := by rfl
  have hnat : (56#i32).toNat = 56 := by rfl
  simp only [lift, core.num.U64.wrapping_mul, UScalar.wrapping_mul, table.scan.GATHER,
    HShiftRight.hShiftRight, UScalar.shiftRight_IScalar, hval, hnat, UScalar.shiftRight]
  norm_num only [UScalarTy.numBits, ↓reduceIte]
  simp only [if_true]
  rw [bind_two_ok]
  exact congrArg ok (congrArg (fun bits : BitVec 8 => (⟨bits⟩ : U8)) (gather_markerBits mask))

/-- Exactness facts for the three extracted scalar scan primitives. -/
structure ScanExactness : Prop where
  zero : ∀ word, table.scan.zero_swar word = ok (zeroMarkers word)
  eq : ∀ word byte, table.scan.eq_swar word byte = ok (tagMarkers word byte)
  gather : ∀ mask, table.scan.gather (markerWord mask) = ok (⟨mask⟩ : U8)

/-- The extracted scalar scan marks bytes independently and gathers every marker exactly. -/
theorem scan_exact : ScanExactness :=
  ⟨extracted_zero_swar_exact, extracted_eq_swar_exact, gather_spreadMarkerBits⟩

end Apeiron.GrowableScan
