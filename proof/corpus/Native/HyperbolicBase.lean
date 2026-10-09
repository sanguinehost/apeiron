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


/-- In `ZMod (p ^ k)` with `p` prime the only idempotents are `0` and `1`: `p ^ k ∣ n (n - 1)` and
`p` divides at most one of the coprime integers `n`, `n - 1`. -/
theorem zmod_prime_pow_idempotent {p : ℕ} (hp : p.Prime) (k : ℕ) (e : ZMod (p ^ k))
    (h : e * e = e) : e = 0 ∨ e = 1 := by
  obtain ⟨n, rfl⟩ := ZMod.intCast_surjective e
  have hdvd : (p : ℤ) ^ k ∣ n * (n - 1) := by
    have h0 : ((n * (n - 1) : ℤ) : ZMod (p ^ k)) = 0 := by
      push_cast
      rw [mul_sub, mul_one, h, sub_self]
    have := (ZMod.intCast_zmod_eq_zero_iff_dvd _ _).mp h0
    exact_mod_cast this
  have hirr : Irreducible (p : ℤ) := (Nat.prime_iff_prime_int.mp hp).irreducible
  by_cases hn : (p : ℤ) ∣ n
  · left
    have hc : IsCoprime ((p : ℤ) ^ k) (n - 1) := by
      refine IsCoprime.pow_left (hirr.coprime_iff_not_dvd.mpr ?_)
      intro h1
      have h2 : (p : ℤ) ∣ 1 := by simpa using dvd_sub hn h1
      exact hirr.not_isUnit (isUnit_of_dvd_one h2)
    have h2 : (p : ℤ) ^ k ∣ n := hc.dvd_of_dvd_mul_left (by rwa [mul_comm] at hdvd)
    rw [ZMod.intCast_zmod_eq_zero_iff_dvd]
    exact_mod_cast h2
  · right
    have hc : IsCoprime ((p : ℤ) ^ k) n :=
      IsCoprime.pow_left (hirr.coprime_iff_not_dvd.mpr hn)
    have h2 : (p : ℤ) ^ k ∣ n - 1 := hc.dvd_of_dvd_mul_left hdvd
    have h3 : ((n - 1 : ℤ) : ZMod (p ^ k)) = 0 := by
      rw [ZMod.intCast_zmod_eq_zero_iff_dvd]
      exact_mod_cast h2
    push_cast at h3
    exact sub_eq_zero.mp h3

/-- A cell of the selected hyperbolic base is idempotent exactly when each channel coordinate is
`0` or `1`. -/
theorem h3_40_idempotent_iff (z : RingCell H3_40) :
    ringMul 1 z z = z ↔
      ((h3_40ChannelEquiv z).1 = 0 ∨ (h3_40ChannelEquiv z).1 = 1) ∧
        ((h3_40ChannelEquiv z).2 = 0 ∨ (h3_40ChannelEquiv z).2 = 1) := by
  have hmul := h3_40_quadratic_split.2.2.2 z z
  have hp : Nat.Prime 3 := by norm_num
  constructor
  · intro h
    rw [h] at hmul
    exact ⟨zmod_prime_pow_idempotent hp 40 _ (congrArg Prod.fst hmul).symm,
      zmod_prime_pow_idempotent hp 40 _ (congrArg Prod.snd hmul).symm⟩
  · rintro ⟨h1, h2⟩
    apply h3_40ChannelEquiv.injective
    rw [hmul]
    refine Prod.ext ?_ ?_
    · show (h3_40ChannelEquiv z).1 * (h3_40ChannelEquiv z).1 = (h3_40ChannelEquiv z).1
      rcases h1 with h1 | h1 <;> rw [h1] <;> simp
    · show (h3_40ChannelEquiv z).2 * (h3_40ChannelEquiv z).2 = (h3_40ChannelEquiv z).2
      rcases h2 with h2 | h2 <;> rw [h2] <;> simp

/-- The selected hyperbolic base has exactly four idempotents: `0`, `1` and the two channel
projectors. -/
theorem h3_40_idempotents_ncard :
    {z : RingCell H3_40 | ringMul 1 z z = z}.ncard = 4 := by
  have hset : {z : RingCell H3_40 | ringMul 1 z z = z} =
      h3_40ChannelEquiv.symm '' (({0, 1} : Set H3_40) ×ˢ ({0, 1} : Set H3_40)) := by
    ext z
    rw [Set.mem_setOf_eq, h3_40_idempotent_iff, Equiv.image_symm_eq_preimage
      h3_40ChannelEquiv]
    simp
  have h01 : (0 : H3_40) ≠ 1 := by
    haveI : Fact (1 < 3 ^ 40) := ⟨by norm_num⟩
    exact zero_ne_one
  rw [hset, Set.ncard_image_of_injective _ h3_40ChannelEquiv.symm.injective, Set.ncard_prod,
    Set.ncard_pair h01]

-- The prime-power premise binds: `ZMod 6` carries the idempotent `3`, which is neither `0` nor `1`.
#guard ((3 : ZMod 6) * 3 = 3 && (3 : ZMod 6) != 0 && (3 : ZMod 6) != 1)

end

end Apeiron.NativeAddress
