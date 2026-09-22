import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.Data.ZMod.QuotientRing

namespace Apeiron.NativeAddress

open Trichotomy.AnyRing
open Function

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

noncomputable section

/-- Channel coordinates give an equivalence when the root and two are units. -/
def quadraticChannelEquiv {R : Type*} [CommRing R] (r s h : R)
    (hrs : r * s = 1) (hh : 2 * h = 1) : RingCell R ≃ R × R where
  toFun z := (ringUpper r z, ringLower r z)
  invFun uv := ringOfHalves uv.1 uv.2 s h
  left_inv := ringOfHalves_upper_lower hrs hh
  right_inv uv := by
    apply Prod.ext
    · exact ringUpper_ringOfHalves hrs hh
    · exact ringLower_ringOfHalves hrs hh

/-- The canonical quadratic split, including its two complementary idempotents. -/
theorem quadratic_split {R : Type*} [CommRing R] {σ r s h : R}
    (hrr : r * r = σ) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingChannel σ (ringUpper r) ∧
      RingChannel σ (ringLower r) ∧
      RingComplementary σ (ringIdemPlus σ r s h) (ringIdemMinus σ r s h) ∧
      ∀ z w : RingCell R,
        quadraticChannelEquiv r s h hrs hh (ringMul σ z w) =
          ((quadraticChannelEquiv r s h hrs hh z).1 *
            (quadraticChannelEquiv r s h hrs hh w).1,
          (quadraticChannelEquiv r s h hrs hh z).2 *
            (quadraticChannelEquiv r s h hrs hh w).2) := by
  refine ⟨ringUpper_channel (by simpa [pow_two] using hrr),
    ringLower_channel (by simpa [pow_two] using hrr), ringComplementary_idem hrr hrs hh, ?_⟩
  intro z w
  exact Prod.ext (ringUpper_mul (by simpa [pow_two] using hrr) z w)
    (ringLower_mul (by simpa [pow_two] using hrr) z w)

/-- The fixed odd-prime-power address ring. -/
abbrev H3_40 := ZMod (3 ^ 40)

theorem h3_40_modulus : 3 ^ 40 = 12157665459056928801 := by norm_num

/-- The inverse of two in `H3_40`. -/
def h3_40Half : H3_40 := 6078832729528464401

theorem h3_40_two_mul_half : (2 : H3_40) * h3_40Half = 1 := by
  change ((2 : ℕ) : H3_40) * ((6078832729528464401 : ℕ) : H3_40) = 1
  rw [← Nat.cast_mul, show 2 * 6078832729528464401 = 3 ^ 40 + 1 by norm_num,
    Nat.cast_add, ZMod.natCast_self]
  simp

/-- The reusable channel-coordinate equivalence for the selected hyperbolic base. -/
def h3_40ChannelEquiv : RingCell H3_40 ≃ H3_40 × H3_40 :=
  quadraticChannelEquiv 1 1 h3_40Half (by norm_num) h3_40_two_mul_half

/-- The selected hyperbolic base is the canonical two-channel quadratic split. -/
theorem h3_40_quadratic_split :
    RingChannel (1 : H3_40) (ringUpper 1) ∧
      RingChannel (1 : H3_40) (ringLower 1) ∧
      RingComplementary (1 : H3_40)
        (ringIdemPlus 1 1 1 h3_40Half) (ringIdemMinus 1 1 1 h3_40Half) ∧
      ∀ z w : RingCell H3_40,
        h3_40ChannelEquiv (ringMul 1 z w) =
          ((h3_40ChannelEquiv z).1 * (h3_40ChannelEquiv w).1,
            (h3_40ChannelEquiv z).2 * (h3_40ChannelEquiv w).2) := by
  exact quadratic_split (by norm_num) (by norm_num) h3_40_two_mul_half

/-- Pairwise-coprime scalar factors, followed by the quadratic split in every factor. -/
def finiteChannelEquiv {ι : Type*} [Fintype ι] (a : ι → ℕ)
    (coprime : Pairwise (Nat.Coprime on a))
    (r s h : (i : ι) → ZMod (a i))
    (hrs : ∀ i, r i * s i = 1) (hh : ∀ i, 2 * h i = 1) :
    RingCell (ZMod (∏ i, a i)) ≃ ((i : ι) → ZMod (a i) × ZMod (a i)) where
  toFun z i :=
    let crt := ZMod.prodEquivPi a coprime
    (ringUpper (r i) (crt z.1 i, crt z.2 i), ringLower (r i) (crt z.1 i, crt z.2 i))
  invFun channels :=
    let crt := ZMod.prodEquivPi a coprime
    let components := fun i => ringOfHalves (channels i).1 (channels i).2 (s i) (h i)
    (crt.symm (fun i => (components i).1), crt.symm (fun i => (components i).2))
  left_inv z := by
    let crt := ZMod.prodEquivPi a coprime
    apply Prod.ext <;> apply crt.injective <;> funext i
    · dsimp [crt]
      rw [RingEquiv.apply_symm_apply]
      exact congrArg Prod.fst (ringOfHalves_upper_lower (hrs i) (hh i) (crt z.1 i, crt z.2 i))
    · dsimp [crt]
      rw [RingEquiv.apply_symm_apply]
      exact congrArg Prod.snd (ringOfHalves_upper_lower (hrs i) (hh i) (crt z.1 i, crt z.2 i))
  right_inv channels := by
    funext i
    apply Prod.ext
    · dsimp
      simp only [RingEquiv.apply_symm_apply]
      exact ringUpper_ringOfHalves (hrs i) (hh i)
    · dsimp
      simp only [RingEquiv.apply_symm_apply]
      exact ringLower_ringOfHalves (hrs i) (hh i)

/-- Scalar CRT and the quadratic channels make `2d` independent multiplication channels. -/
theorem finite_channel_mul {ι : Type*} [Fintype ι] (a : ι → ℕ)
    (coprime : Pairwise (Nat.Coprime on a))
    (r s h : (i : ι) → ZMod (a i))
    (hrr : ∀ i, r i * r i = 1) (hrs : ∀ i, r i * s i = 1) (hh : ∀ i, 2 * h i = 1)
    (z w : RingCell (ZMod (∏ i, a i))) :
    finiteChannelEquiv a coprime r s h hrs hh (ringMul 1 z w) =
      fun i =>
        ((finiteChannelEquiv a coprime r s h hrs hh z i).1 *
            (finiteChannelEquiv a coprime r s h hrs hh w i).1,
          (finiteChannelEquiv a coprime r s h hrs hh z i).2 *
            (finiteChannelEquiv a coprime r s h hrs hh w i).2) := by
  funext i
  apply Prod.ext
  · change ringUpper (r i)
        ((ZMod.prodEquivPi a coprime (ringMul 1 z w).1 i),
          (ZMod.prodEquivPi a coprime (ringMul 1 z w).2 i)) =
        ringUpper (r i)
            ((ZMod.prodEquivPi a coprime z.1 i), (ZMod.prodEquivPi a coprime z.2 i)) *
          ringUpper (r i)
            ((ZMod.prodEquivPi a coprime w.1 i), (ZMod.prodEquivPi a coprime w.2 i))
    simp only [map_add, map_mul, Pi.add_apply, Pi.mul_apply, one_mul, ringUpper]
    linear_combination (-(ZMod.prodEquivPi a coprime z.2 i *
      ZMod.prodEquivPi a coprime w.2 i)) * hrr i
  · change ringLower (r i)
        ((ZMod.prodEquivPi a coprime (ringMul 1 z w).1 i),
          (ZMod.prodEquivPi a coprime (ringMul 1 z w).2 i)) =
        ringLower (r i)
            ((ZMod.prodEquivPi a coprime z.1 i), (ZMod.prodEquivPi a coprime z.2 i)) *
          ringLower (r i)
            ((ZMod.prodEquivPi a coprime w.1 i), (ZMod.prodEquivPi a coprime w.2 i))
    simp only [map_add, map_mul, Pi.add_apply, Pi.mul_apply, one_mul, ringLower]
    linear_combination (-(ZMod.prodEquivPi a coprime z.2 i *
      ZMod.prodEquivPi a coprime w.2 i)) * hrr i

end

end Apeiron.NativeAddress
