/-
  `ReadTrichotomy32.lean` — AR-12's read-only trichotomy instances at width 32, with the
  residuals in place.

  AR-12 (`work/proofs/AlgebraSha256Sigma.lean`, digest `cb8944ae…`) proved the two-slot reading of
  one SHA-256 rotation term at a width parameter `k`, and left the width-32 instances of its four
  composite cell forms as four `omega`s in a read-only file (`AlgebraSha256Sigma.md` OPEN 2). AR-68
  (`work/proofs/AlgebraSigma32.md` R2) restated that step and did not discharge it, because it was
  reading the halves pairing and not AR-12's carrier.

  This file is that step, and nothing else. The carrier is AR-12's: the one-slot line `(x, 0)` in
  `RingCell (ZMod (2 ^ 32))` with the rotation constants `(2^(32−n), 2^n)` in the multiplier's
  slots, and the reads are AR-12's `ringUpper` reads at the three named faces. The residuals stay
  in the statements: `rotResidual32` per rotation term at the hyperbolic face, `shrResidual32`
  (exactly one `SHR`) per rotation term at the parabolic face, and `zxor3_32` for the `⊕` layer.
  Nothing is removed, no real-field classification is attempted, and no rotation-action claim is
  made.

  The general theorems are cited, not re-proved: AR-12's `rotCell_slots`, `ringUpper_zero_read`,
  `rotCell_parabolic_read`, `rotCell_hyperbolic_read`, `mul_pow_cast`, `rotCell_parabolic_offset`,
  `rotCell_hyperbolic_offset`, `den_xor` and the four `*_cell_form` rows; AR-26's proved
  `rotrVal`, `rotrVal_eq_rotrBv`, `shrVal`, `shrVal_eq_shrxBv` (`Icarus.AlgebraSigmaHalves`,
  imported and opened here), which supply the width-generic value statements at `k = 32`; and
  `Apeiron.Trichotomy.{AnyRing,ChannelsAnyRing}` for `RingCell`, `ringMul` and `ringUpper`. The
  arithmetic proved here is only what those citations do not already give at width 32: the `2`-power
  modulus lemmas, the extraction of the corpus's `rotr`/`shrx` bodies to `rotateRight`, the elliptic
  face's missing channel root at `ZMod (2 ^ 32)`, and the eight cell-form instances.

  Trust boundary. Statements about `ZMod (2 ^ 32)` and about `Apeiron.Crypto.Sha2`'s own `Sigma0`,
  `Sigma1`, `sigma0`, `sigma1` declarations, so no re-declaration gap remains for the four
  composites. Nothing here reaches rustc/LLVM, the kernel ABI, GPU/SPIR-V or the `unsafe` container,
  and no security, collision-resistance or constant-time property is claimed.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2
import Icarus.AlgebraSigmaHalves

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace IcarusSha256ReadTrichotomy32

open Apeiron.Trichotomy.AnyRing
open AlgebraSigmaHalves (rotrVal rotrVal_eq_rotrBv shrVal shrVal_eq_shrxBv)

/-! ## §0 The carrier at width 32 -/

/-- The word ring of one SHA-256 word, `ZMod (2 ^ 32)`: AR-12's carrier `ZMod (2 ^ k)` at `k = 32`,
and the ring `Sha2.Word` is read into. -/
abbrev Ring32 : Type := ZMod (2 ^ 32)

/-- A natural number read into the word ring. AR-12 writes the cast inline; it is named here
because the two residuals and both reads are stated over it. -/
abbrev denNat32 (a : ℕ) : Ring32 := (a : ℕ)

/-- A 32-bit word read into the word ring: AR-12's `den`. -/
def den32 (x : BitVec 32) : Ring32 := denNat32 x.toNat

/-- The elliptic face's σ at width 32, `−1`. -/
def sigmaElliptic32 : Ring32 := -1

/-- The parabolic face's σ at width 32, `0`. -/
def sigmaParabolic32 : Ring32 := 0

/-- The hyperbolic face's σ at width 32, `+1`. -/
def sigmaHyperbolic32 : Ring32 := 1

/-! ## §1 The dial: the three faces as numbers, and their channel roots -/

/-- The three faces are the three numbers `−1, 0, 1` at width 32. -/
theorem sigma_faces_32 :
    sigmaElliptic32 = (-1 : Ring32) ∧ sigmaParabolic32 = (0 : Ring32)
      ∧ sigmaHyperbolic32 = (1 : Ring32) := ⟨rfl, rfl, rfl⟩

/-- The parabolic face has the channel root `r = 0`: `r ^ 2 = σ`, so `ringUpper r` is available. -/
theorem parabolic_root_32 : (0 : Ring32) ^ 2 = sigmaParabolic32 := rfl

/-- The hyperbolic face has the channel root `r = 1`. -/
theorem hyperbolic_root_32 : (1 : Ring32) ^ 2 = sigmaHyperbolic32 := rfl

/-- **The elliptic face has no channel root at width 32.** `ringUpper r` is a read only where
`r ^ 2 = σ`, and no element of `ZMod (2 ^ 32)` squares to `−1`: if `2 ^ 32 ∣ r ^ 2 + 1` then
`4 ∣ r ^ 2 + 1`, while a square is `0` or `1` modulo `4`, so `r ^ 2 + 1` is `1` or `2` modulo `4`.
AR-12 proved the same for `ZMod 4` and `ZMod 16` by `decide`; at `2 ^ 32` the enumeration is out of
reach and this is the argument. Consequence for the reading: at this face there are the two slots and
**no channel read**. -/
theorem no_channel_root_elliptic_32 : ¬ ∃ r : Ring32, r ^ 2 = sigmaElliptic32 := by
  rintro ⟨r, hr⟩
  obtain ⟨v, hv⟩ := ZMod.natCast_zmod_surjective r
  have hz : ((v ^ 2 + 1 : ℕ) : Ring32) = 0 := by
    have h : ((v ^ 2 + 1 : ℕ) : Ring32) = (v : Ring32) ^ 2 + 1 := by push_cast; rfl
    rw [h, hv, hr, sigmaElliptic32]
    ring_nf
  have hdvd : 2 ^ 32 ∣ v ^ 2 + 1 := (ZMod.natCast_eq_zero_iff (v ^ 2 + 1) (2 ^ 32)).mp hz
  obtain ⟨t, ht⟩ := hdvd
  have hpow : 2 ^ 32 = 4 * 2 ^ 30 := by norm_num
  have hsplit : v ^ 2 + 1 = 4 * (2 ^ 30 * t) := by
    rw [ht, hpow]
    ring
  have hzero : (v ^ 2 + 1) % 4 = 0 := by
    rw [hsplit]
    exact Nat.mul_mod_right 4 _
  -- A square is `0` or `1` modulo `4`, so `v ^ 2 + 1` is `1` or `2` modulo `4`.
  have hne : (v ^ 2 + 1) % 4 ≠ 0 := by
    rcases Nat.mod_two_eq_zero_or_one v with hb | hb
    · obtain ⟨a, ha⟩ : ∃ a, v = 2 * a := ⟨v / 2, by omega⟩
      rw [ha]
      have hsq : (2 * a) ^ 2 = 4 * (a * a) + 0 := by ring
      rw [hsq]
      omega
    · obtain ⟨a, ha⟩ : ∃ a, v = 2 * a + 1 := ⟨v / 2, by omega⟩
      rw [ha]
      have hsq : (2 * a + 1) ^ 2 = 4 * (a * a + a) + 1 := by ring
      rw [hsq]
      omega
  omega

/-! ## §2 The rotation cell on the one-slot line -/

/-- AR-12's `rotCell` at width 32: the word on the first axis, the rotation constants
`(2^(32−n), 2^n)` in the multiplier's slots, one `ringMul` at face `σ`. -/
def rotCell32 (σ : Ring32) (x : BitVec 32) (n : ℕ) : RingCell Ring32 :=
  ringMul σ (den32 x, 0) (denNat32 (2 ^ (32 - n)), denNat32 (2 ^ n))

/-- **The two slots are σ-free at width 32** (AR-12's `rotCell_slots`). The dial does not enter the
cell of a rotation term; it enters only the read. -/
theorem rotCell32_slots (σ : Ring32) (x : BitVec 32) (n : ℕ) :
    rotCell32 σ x n = (den32 x * denNat32 (2 ^ (32 - n)), den32 x * denNat32 (2 ^ n)) := by
  simp only [rotCell32, ringMul]
  exact Prod.ext (by ring) (by ring)

/-- **All three faces carry the same cell at width 32**: the trichotomy enters at the read, not at
the point. -/
theorem faces_share_the_cell_32 (x : BitVec 32) (n : ℕ) :
    rotCell32 sigmaElliptic32 x n = rotCell32 sigmaParabolic32 x n
      ∧ rotCell32 sigmaParabolic32 x n = rotCell32 sigmaHyperbolic32 x n :=
  ⟨by rw [rotCell32_slots, rotCell32_slots], by rw [rotCell32_slots, rotCell32_slots]⟩

/-! ## §3 The two available reads, and the two named residuals -/

/-- **The parabolic read at width 32** (AR-12's `rotCell_parabolic_read`): `ringUpper 0` off the cell
at `σ = 0`, which is the first slot. -/
def readPar32 (x : BitVec 32) (n : ℕ) : Ring32 :=
  ringUpper sigmaParabolic32 (rotCell32 sigmaParabolic32 x n)

/-- **The hyperbolic read at width 32** (AR-12's `rotCell_hyperbolic_read`): `ringUpper 1` off the
cell at `σ = 1`, which is the first slot plus the second. -/
def readHyp32 (x : BitVec 32) (n : ℕ) : Ring32 :=
  ringUpper sigmaHyperbolic32 (rotCell32 sigmaHyperbolic32 x n)

/-- The parabolic read is the wrapped half of the rotation, exactly (AR-12's
`rotCell_parabolic_read`). -/
theorem readPar32_eq (x : BitVec 32) (n : ℕ) :
    readPar32 x n = den32 x * denNat32 (2 ^ (32 - n)) := by
  rw [readPar32, rotCell32_slots]
  simp only [ringUpper, sigmaParabolic32, zero_mul, add_zero]

/-- The hyperbolic read is the wrapped half of the rotation plus the wrapped half of the reverse
rotation (AR-12's `rotCell_hyperbolic_read`). -/
theorem readHyp32_eq (x : BitVec 32) (n : ℕ) :
    readHyp32 x n = den32 x * denNat32 (2 ^ (32 - n)) + den32 x * denNat32 (2 ^ n) := by
  rw [readHyp32, rotCell32_slots]
  simp only [ringUpper, sigmaHyperbolic32, one_mul]

/-- **The hyperbolic residual, named at width 32** (AR-12's `rotResidual`): the wrapped half of the
reverse rotation minus the logical right shift, `(x·2^n mod 2^32) − ⌊x/2^n⌋`. It stands in every
hyperbolic cell-form statement below. -/
def rotResidual32 (x : BitVec 32) (n : ℕ) : Ring32 :=
  denNat32 ((x.toNat * 2 ^ n) % 2 ^ 32) - denNat32 (x.toNat / 2 ^ n)

/-- **The parabolic residual, named at width 32**: exactly one `SHR` term, as AR-12's
`rotCell_parabolic_offset` states it. -/
def shrResidual32 (x : BitVec 32) (n : ℕ) : Ring32 := den32 (Apeiron.Crypto.Sha2.shrx x n)

/-- The residual is **not removable** at width 32: at `x = 3`, `n = 2` it is `12`, not `0`. So the
hyperbolic read is not the first-slot projection, and the two offset rows below are not the same
statement. -/
theorem rotResidual32_ne_zero : rotResidual32 (BitVec.ofNat 32 3) 2 ≠ 0 := by decide

/-! ## §4 The two modulus facts the readings need -/

/-- The modulus vanishes in its own ring at width 32 (AR-12's `two_pow_cast_zero`). -/
theorem two_pow_cast_zero_32 : ((2 ^ 32 : ℕ) : Ring32) = 0 := by
  have h := ZMod.natCast_mod (2 ^ 32) (2 ^ 32)
  rw [Nat.mod_self] at h
  exact h.symm.trans (by simp)

/-- A multiple of the modulus vanishes at width 32 (AR-12's `mul_two_pow_cast_zero`). -/
theorem mul_two_pow_cast_zero_32 (q : ℕ) : ((q * 2 ^ 32 : ℕ) : Ring32) = 0 := by
  rw [show (q * 2 ^ 32 : ℕ) = 2 ^ 32 * q by rw [Nat.mul_comm], Nat.cast_mul,
    two_pow_cast_zero_32, zero_mul]

/-- **Multiplying the word by `2^(32−n)` and reading into the ring sees only the low `n` bits**
(AR-12's `mul_pow_cast` at `k = 32`). The first slot of the rotation cell is the wrapped half of
the rotation and nothing else. -/
theorem mul_pow_cast_32 (a : ℕ) (n : ℕ) (hn : n ≤ 32) :
    denNat32 (a * 2 ^ (32 - n)) = denNat32 (a % 2 ^ n * 2 ^ (32 - n)) := by
  simp only [denNat32]
  have hxpow : 2 ^ n * 2 ^ (32 - n) = 2 ^ 32 := by
    rw [← Nat.pow_add]
    congr 1
    omega
  have hnat : a * 2 ^ (32 - n) = a % 2 ^ n * 2 ^ (32 - n) + (a / 2 ^ n) * 2 ^ 32 := by
    have hdiv := Nat.mod_add_div a (2 ^ n)
    calc a * 2 ^ (32 - n) = (a % 2 ^ n + 2 ^ n * (a / 2 ^ n)) * 2 ^ (32 - n) := by rw [hdiv]
      _ = a % 2 ^ n * 2 ^ (32 - n) + (a / 2 ^ n) * (2 ^ n * 2 ^ (32 - n)) := by ring
      _ = a % 2 ^ n * 2 ^ (32 - n) + (a / 2 ^ n) * 2 ^ 32 := by rw [hxpow]
  rw [hnat, Nat.cast_add, Nat.cast_mul, mul_two_pow_cast_zero_32, add_zero]

/-! ## §5 The corpus's own bodies, extracted to `rotateRight` -/

/-- **The corpus's `Sha2.rotr x m` is `x.rotateRight m`** for `0 < m < 32` (AR-12's
`trotr_eq_rotateRight`, regime I, at the corpus's own width). -/
theorem sha2_rotr_eq_rotateRight (x : BitVec 32) (m : ℕ) (hm : 0 < m) (hmk : m < 32) :
    Apeiron.Crypto.Sha2.rotr x m = x.rotateRight m := by
  have h1 : m % 32 = m := Nat.mod_eq_of_lt hmk
  have h2 : (32 - m) % 32 = 32 - m := Nat.mod_eq_of_lt (by omega)
  simp only [Apeiron.Crypto.Sha2.rotr, h1, h2, BitVec.rotateRight_def]

/-- A rotation at width 32, read into the ring, in AR-26's value form. `rotrVal_eq_rotrBv` is
AR-26's statement that this is the extracted rotation's value. -/
theorem den32_rotrVal (x : BitVec 32) (m : ℕ) (hmk : m < 32) :
    den32 (x.rotateRight m) = denNat32 (rotrVal (k := 32) m x.toNat) := by
  have hof : (BitVec.ofNat 32 x.toNat : BitVec 32) = x :=
    BitVec.eq_of_toNat_eq (by rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt x.isLt])
  have h := rotrVal_eq_rotrBv x.toNat m x.isLt hmk
  rw [AlgebraSigmaHalves.rotrBv, hof] at h
  rw [← h]
  simp only [den32, denNat32]

/-- A shift at width 32, read into the ring, in AR-26's value form. -/
theorem den32_shrVal (x : BitVec 32) (r : ℕ) (hrk : r < 32) :
    den32 (Apeiron.Crypto.Sha2.shrx x r) = denNat32 (shrVal r x.toNat) := by
  have hof : (BitVec.ofNat 32 x.toNat : BitVec 32) = x :=
    BitVec.eq_of_toNat_eq (by rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt x.isLt])
  have h := shrVal_eq_shrxBv x.toNat r x.isLt hrk
  rw [AlgebraSigmaHalves.shrxBv, hof] at h
  simp only [Apeiron.Crypto.Sha2.shrx]
  rw [← h]
  simp only [den32, denNat32]

/-! ## §6 The exact offsets of the two reads at width 32 -/

/-- **The exact offset of the hyperbolic read: the named residual.** At width 32, for every
`0 < n < 32`, the two-slot sum read exceeds the corpus's `rotr x n` by exactly `rotResidual32 x n`
(AR-12's `rotCell_hyperbolic_offset` and `rotCell_corrected`). -/
theorem rotCell32_hyperbolic_offset (x : BitVec 32) (n : ℕ) (hn : 0 < n) (hnk : n < 32) :
    readHyp32 x n - rotResidual32 x n = den32 (Apeiron.Crypto.Sha2.rotr x n) := by
  have hc1 : den32 x * denNat32 (2 ^ (32 - n)) = denNat32 (x.toNat % 2 ^ n * 2 ^ (32 - n)) := by
    simp only [den32, denNat32]
    rw [← Nat.cast_mul]
    exact mul_pow_cast_32 x.toNat n (Nat.le_of_lt hnk)
  have hc2 : den32 x * denNat32 (2 ^ n) = denNat32 ((x.toNat * 2 ^ n) % 2 ^ 32) := by
    simp only [den32, denNat32]
    rw [← Nat.cast_mul, (ZMod.natCast_mod _ _).symm]
  rw [readHyp32_eq, rotResidual32, hc1, hc2,
    sha2_rotr_eq_rotateRight x n hn hnk, den32_rotrVal x n hnk]
  simp only [denNat32]
  rw [rotrVal, Nat.cast_add]
  ring

/-- **The exact offset of the parabolic read: the residual is one `SHR`.** At width 32, for every
`0 < n < 32`, the first-slot read misses exactly `x >>> n` (AR-12's `rotCell_parabolic_offset`). -/
theorem rotCell32_parabolic_offset (x : BitVec 32) (n : ℕ) (hn : 0 < n) (hnk : n < 32) :
    readPar32 x n + den32 (Apeiron.Crypto.Sha2.shrx x n)
      = den32 (Apeiron.Crypto.Sha2.rotr x n) := by
  have hc1 : den32 x * denNat32 (2 ^ (32 - n)) = denNat32 (x.toNat % 2 ^ n * 2 ^ (32 - n)) := by
    simp only [den32, denNat32]
    rw [← Nat.cast_mul]
    exact mul_pow_cast_32 x.toNat n (Nat.le_of_lt hnk)
  rw [readPar32_eq, hc1, den32_shrVal x n hnk, sha2_rotr_eq_rotateRight x n hn hnk,
    den32_rotrVal x n hnk]
  simp only [denNat32]
  rw [shrVal, rotrVal, Nat.cast_add]
  ring

/-- **Both available faces read the same rotation at width 32.** The hyperbolic sum read minus
`rotResidual32` and the parabolic first-slot read plus one `SHR` are one and the same value, so the
two named residuals are the two prices of the same reading. -/
theorem faces_read_agree_32 (x : BitVec 32) (n : ℕ) (hn : 0 < n) (hnk : n < 32) :
    readHyp32 x n - rotResidual32 x n = readPar32 x n + den32 (Apeiron.Crypto.Sha2.shrx x n) :=
  (rotCell32_hyperbolic_offset x n hn hnk).trans
    (rotCell32_parabolic_offset x n hn hnk).symm

/-! ## §7 The `⊕` layer, kept -/

/-- The bitwise `⊕` of two ring values through their canonical representatives (AR-12's `zxor` at
width 32): a function of the ring, not one of its operations. -/
def zxor32 (u v : Ring32) : Ring32 := ((u.val ^^^ v.val : ℕ) : Ring32)

/-- `zxor32` at width 32 is the bit-vector `⊕` of the two words (AR-12's `den_xor`). -/
theorem den32_xor (a b : BitVec 32) : den32 (a ^^^ b) = zxor32 (den32 a) (den32 b) := by
  simp only [den32, zxor32, denNat32, BitVec.toNat_xor, ZMod.val_natCast_of_lt a.isLt,
    ZMod.val_natCast_of_lt b.isLt]

/-- The three-term `⊕` layer of a composite in ring coordinates (AR-12's `zxor3`). **This is the
named `⊕` residual and it is kept in all eight cell-form statements below.** -/
def zxor3_32 (a b c : Ring32) : Ring32 := zxor32 (zxor32 a b) c

/-! ## §8 Every FIPS amount is below the width -/

/-- Every FIPS 180-4 amount used by the four composites is strictly below 32, so every width
hypothesis of AR-12's `*_cell_form` rows (`19 ≤ k`, `20 ≤ k`, `26 ≤ k`) is discharged at `k = 32`.
This is the whole content of the four `omega`s AR-12 left open. -/
theorem fips_amounts_below_32 :
    ∀ m ∈ ({2, 13, 22, 6, 11, 25, 7, 18, 17, 19, 3, 10} : Finset ℕ), 0 < m ∧ m < 32 := by
  decide

/-! ## §9 (a) The four composites at width 32, in Cell coordinates, per face -/

/-- **`Σ₀` at width 32, hyperbolic face.** AR-12's `tSigma0_cell_form` at `hk : 26 ≤ k` and
`k = 32`, with `rotResidual32` in the statement: three corrected two-slot reads and the `⊕`
residual. -/
theorem bigSigma0_cell_form_hyp_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.Sigma0 x)
      = zxor3_32 (readHyp32 x 2 - rotResidual32 x 2)
          (readHyp32 x 13 - rotResidual32 x 13) (readHyp32 x 22 - rotResidual32 x 22) := by
  simp only [Apeiron.Crypto.Sha2.Sigma0, zxor3_32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_hyperbolic_offset x 2 (by decide) (by decide),
    ← rotCell32_hyperbolic_offset x 13 (by decide) (by decide),
    ← rotCell32_hyperbolic_offset x 22 (by decide) (by decide)]

/-- **`Σ₁` at width 32, hyperbolic face.** AR-12's `tSigma1_cell_form` at `hk : 26 ≤ k`. -/
theorem bigSigma1_cell_form_hyp_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.Sigma1 x)
      = zxor3_32 (readHyp32 x 6 - rotResidual32 x 6)
          (readHyp32 x 11 - rotResidual32 x 11) (readHyp32 x 25 - rotResidual32 x 25) := by
  simp only [Apeiron.Crypto.Sha2.Sigma1, zxor3_32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_hyperbolic_offset x 6 (by decide) (by decide),
    ← rotCell32_hyperbolic_offset x 11 (by decide) (by decide),
    ← rotCell32_hyperbolic_offset x 25 (by decide) (by decide)]

/-- **`σ₀` at width 32, hyperbolic face.** The two rotation terms are corrected two-slot reads; the
`SHR^3` term is the corpus's own `shrx` read into the ring and has no cell reading at all. That
term is AR-12's OPEN 1 (a shift is not congruence-preserving), restated here and not closed. -/
theorem smallSigma0_cell_form_hyp_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.sigma0 x)
      = zxor3_32 (readHyp32 x 7 - rotResidual32 x 7)
          (readHyp32 x 18 - rotResidual32 x 18) (den32 (Apeiron.Crypto.Sha2.shrx x 3)) := by
  simp only [Apeiron.Crypto.Sha2.sigma0, zxor3_32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_hyperbolic_offset x 7 (by decide) (by decide),
    ← rotCell32_hyperbolic_offset x 18 (by decide) (by decide)]

/-- **`σ₁` at width 32, hyperbolic face.** -/
theorem smallSigma1_cell_form_hyp_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.sigma1 x)
      = zxor3_32 (readHyp32 x 17 - rotResidual32 x 17)
          (readHyp32 x 19 - rotResidual32 x 19) (den32 (Apeiron.Crypto.Sha2.shrx x 10)) := by
  simp only [Apeiron.Crypto.Sha2.sigma1, zxor3_32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_hyperbolic_offset x 17 (by decide) (by decide),
    ← rotCell32_hyperbolic_offset x 19 (by decide) (by decide)]

/-- **`Σ₀` at width 32, parabolic face.** The other available face: first-slot reads, one `SHR`
residual per term, and the `⊕` residual. This row differs from the hyperbolic one — the residual
terms differ — and neither drops the residual. -/
theorem bigSigma0_cell_form_par_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.Sigma0 x)
      = zxor3_32 (readPar32 x 2 + shrResidual32 x 2)
          (readPar32 x 13 + shrResidual32 x 13) (readPar32 x 22 + shrResidual32 x 22) := by
  simp only [Apeiron.Crypto.Sha2.Sigma0, zxor3_32, shrResidual32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_parabolic_offset x 2 (by decide) (by decide),
    ← rotCell32_parabolic_offset x 13 (by decide) (by decide),
    ← rotCell32_parabolic_offset x 22 (by decide) (by decide)]

/-- **`Σ₁` at width 32, parabolic face.** -/
theorem bigSigma1_cell_form_par_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.Sigma1 x)
      = zxor3_32 (readPar32 x 6 + shrResidual32 x 6)
          (readPar32 x 11 + shrResidual32 x 11) (readPar32 x 25 + shrResidual32 x 25) := by
  simp only [Apeiron.Crypto.Sha2.Sigma1, zxor3_32, shrResidual32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_parabolic_offset x 6 (by decide) (by decide),
    ← rotCell32_parabolic_offset x 11 (by decide) (by decide),
    ← rotCell32_parabolic_offset x 25 (by decide) (by decide)]

/-- **`σ₀` at width 32, parabolic face.** The third argument is the corpus's own `SHR^3` read, which
is `shrResidual32 x 3`: at this face the `SHR` term is a residual term of the same shape as the two
rotation terms'. -/
theorem smallSigma0_cell_form_par_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.sigma0 x)
      = zxor3_32 (readPar32 x 7 + shrResidual32 x 7)
          (readPar32 x 18 + shrResidual32 x 18) (shrResidual32 x 3) := by
  simp only [Apeiron.Crypto.Sha2.sigma0, zxor3_32, shrResidual32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_parabolic_offset x 7 (by decide) (by decide),
    ← rotCell32_parabolic_offset x 18 (by decide) (by decide)]

/-- **`σ₁` at width 32, parabolic face.** -/
theorem smallSigma1_cell_form_par_32 (x : BitVec 32) :
    den32 (Apeiron.Crypto.Sha2.sigma1 x)
      = zxor3_32 (readPar32 x 17 + shrResidual32 x 17)
          (readPar32 x 19 + shrResidual32 x 19) (shrResidual32 x 10) := by
  simp only [Apeiron.Crypto.Sha2.sigma1, zxor3_32, shrResidual32]
  rw [den32_xor, den32_xor]
  rw [← rotCell32_parabolic_offset x 17 (by decide) (by decide),
    ← rotCell32_parabolic_offset x 19 (by decide) (by decide)]

/-! ## §10 The trichotomy at width 32, one row -/

/-- **The AR-12 trichotomy at width 32, on one rotation term.** At `σ = 1` there is the sum read and
its residual is exactly `rotResidual32`; at `σ = 0` there is the first-slot read and its residual is
exactly one `SHR`; at `σ = −1` there is the cell and **no channel read**. The two available reads
agree on the corpus's `Sha2.rotr`, and the three cells coincide. -/
theorem read_trichotomy_32 (x : BitVec 32) (n : ℕ) (hn : 0 < n) (hnk : n < 32) :
    (readHyp32 x n - rotResidual32 x n = den32 (Apeiron.Crypto.Sha2.rotr x n))
      ∧ (readPar32 x n + den32 (Apeiron.Crypto.Sha2.shrx x n)
          = den32 (Apeiron.Crypto.Sha2.rotr x n))
      ∧ (¬ ∃ r : Ring32, r ^ 2 = sigmaElliptic32)
      ∧ (rotCell32 sigmaElliptic32 x n = rotCell32 sigmaHyperbolic32 x n) :=
  ⟨rotCell32_hyperbolic_offset x n hn hnk, rotCell32_parabolic_offset x n hn hnk,
    no_channel_root_elliptic_32, (faces_share_the_cell_32 x n).2.symm⟩

/-! ## Audit -/

#print axioms sigma_faces_32
#print axioms parabolic_root_32
#print axioms hyperbolic_root_32
#print axioms no_channel_root_elliptic_32
#print axioms rotCell32_slots
#print axioms faces_share_the_cell_32
#print axioms readPar32_eq
#print axioms readHyp32_eq
#print axioms rotResidual32_ne_zero
#print axioms two_pow_cast_zero_32
#print axioms mul_two_pow_cast_zero_32
#print axioms mul_pow_cast_32
#print axioms sha2_rotr_eq_rotateRight
#print axioms den32_rotrVal
#print axioms den32_shrVal
#print axioms rotCell32_hyperbolic_offset
#print axioms rotCell32_parabolic_offset
#print axioms faces_read_agree_32
#print axioms den32_xor
#print axioms fips_amounts_below_32
#print axioms bigSigma0_cell_form_hyp_32
#print axioms bigSigma1_cell_form_hyp_32
#print axioms smallSigma0_cell_form_hyp_32
#print axioms smallSigma1_cell_form_hyp_32
#print axioms bigSigma0_cell_form_par_32
#print axioms bigSigma1_cell_form_par_32
#print axioms smallSigma0_cell_form_par_32
#print axioms smallSigma1_cell_form_par_32
#print axioms read_trichotomy_32

end IcarusSha256ReadTrichotomy32