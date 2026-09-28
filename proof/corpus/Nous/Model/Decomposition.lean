import Apeiron.Table.Basic
import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Apeiron.Trichotomy.IdempotentsAnyRing
import Nous.Model.Weighting
import Nous.Model.State
import Nous.Model.Memory
import Mathlib.Algebra.Field.Basic
import Mathlib.Logic.Equiv.Basic
import Mathlib.Tactic

/-!
# Hyperbolic Channel Decomposition and Diagonal Factoring

This module establishes the hyperbolic channel decomposition of the two-dimensional
algebra `R[x]/(x^2 - σ)` over a commutative ring `R`, and proves that sequential state
folds (`tokenFold`) and table memory updates (`memoryStep`) factor diagonally into
independent scalar operations per channel.

## Mathematical Setting

For a quadratic parameter `σ : R` admitting a root witness `r : R` (`r ^ 2 = σ`), the two
coordinate projections:
`ringUpper r z = z.1 + r * z.2`
`ringLower r z = z.1 - r * z.2`
define unital ring homomorphisms from `RingCell R` to `R`.

The combined coordinate map:
`channelPair r z = (ringUpper r z, ringLower r z)`
embeds `RingCell R` into `R × R`.

When `2` and `r` are invertible in `R` (witnessed by explicit inverses `s` with `r * s = 1`
and `h` with `2 * h = 1`, or by `IsUnit (2 : R)` and `IsUnit r`), `channelPair r` is a bijection
and an equivalence of unital commutative algebras (`channelEquiv`), with explicit two-sided
inverse:
`(u, l) ↦ ((u + l) * h, (u - l) * h * s) = ((u + l) / 2, (u - l) / (2 * r))`.

Furthermore, `channelPair r`:
1. Intertwines multiplication:
   `channelPair r (ringMul σ z w) = ((channelPair r z).1 * (channelPair r w).1, (channelPair r z).2 * (channelPair r w).2)`
2. Intertwines addition:
   `channelPair r (z + w) = ((channelPair r z).1 + (channelPair r w).1, (channelPair r z).2 + (channelPair r w).2)`
3. Preserves the multiplicative unit:
   `channelPair r ringOne = (1, 1)`
4. Factorizes the quadratic norm form:
   `ringNormForm σ z = (channelPair r z).1 * (channelPair r z).2`.

## Geometric Regime Restriction

This decomposition holds strictly on the **hyperbolic face** (`σ > 0` over ordered fields,
or more generally where `σ` has an invertible square root `r` alongside invertible `2`).
- On the **elliptic face** (`σ < 0` over `ℝ`), the polynomial `x^2 - σ` is irreducible; the
  algebra is a field (isomorphic to `ℂ`), having no nontrivial idempotents and no zero divisors,
  so no such product splitting exists.
- On the **parabolic face** (`σ = 0`), the algebra contains nonzero nilpotent elements
  (such as `(0, 1)` where `(0, 1)^2 = (0, 0)`), yielding a dual-number algebra rather than
  a product of fields.

## Diagonal Factoring of Model Operations

Under the hyperbolic decomposition:
1. Sequential token folds factor into independent scalar list products:
   `channelPair r (tokenFold σ xs) = ((xs.map (ringUpper r)).prod, (xs.map (ringLower r)).prod)`.
2. Table memory updates factor into independent scalar channel updates:
   `channelPair r ((memoryStep σ k v M) x) = if x = k then (ringUpper r v * (channelPair r (M x)).1, ringLower r v * (channelPair r (M x)).2) else channelPair r (M x)`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Table
open Apeiron.Trichotomy.AnyRing

section Decomposition

variable {R : Type*} [CommRing R]

/-- Project a ring point into its pair of channel coordinates. -/
def channelPair (r : R) (z : RingCell R) : R × R :=
  (ringUpper r z, ringLower r z)

/-- Reconstruct a ring point from channel coordinates using explicit inverse witnesses
`s` for `r` (`r * s = 1`) and `h` for `2` (`2 * h = 1`).
In standard fractional notation, this computes `((u + l) / 2, (u - l) / (2 * r))`. -/
def channelInv (s h : R) (p : R × R) : RingCell R :=
  ringOfHalves p.1 p.2 s h

/-- Left inverse property: channel reconstruction cancels projection. -/
theorem channelInv_channelPair {r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1)
    (z : RingCell R) : channelInv s h (channelPair r z) = z :=
  ringOfHalves_upper_lower hrs hh z

/-- Right inverse property: channel projection cancels reconstruction. -/
theorem channelPair_channelInv {r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1)
    (p : R × R) : channelPair r (channelInv s h p) = p := by
  apply Prod.ext
  · exact ringUpper_ringOfHalves hrs hh
  · exact ringLower_ringOfHalves hrs hh

/-- Equivalence between `RingCell R` and `R × R` induced by channel coordinates. -/
def channelEquiv (r s h : R) (hrs : r * s = 1) (hh : 2 * h = 1) :
    RingCell R ≃ R × R where
  toFun := channelPair r
  invFun := channelInv s h
  left_inv := channelInv_channelPair hrs hh
  right_inv := channelPair_channelInv hrs hh

/-- `channelPair r` is bijective when `2` and `r` have inverses. -/
theorem channelPair_bijective {r s h : R} (hrs : r * s = 1) (hh : 2 * h = 1) :
    Function.Bijective (channelPair r) :=
  (channelEquiv r s h hrs hh).bijective

/-- `channelPair r` intertwines multiplication in `R[x]/(x^2 - σ)` with componentwise multiplication. -/
theorem channelPair_mul (σ r : R) (hr : r ^ 2 = σ) (z w : RingCell R) :
    channelPair r (ringMul σ z w) =
      ((channelPair r z).1 * (channelPair r w).1,
       (channelPair r z).2 * (channelPair r w).2) := by
  apply Prod.ext
  · exact ringUpper_mul hr z w
  · exact ringLower_mul hr z w

/-- `channelPair r` preserves addition. -/
theorem channelPair_add (r : R) (z w : RingCell R) :
    channelPair r (z + w) =
      ((channelPair r z).1 + (channelPair r w).1,
       (channelPair r z).2 + (channelPair r w).2) := by
  apply Prod.ext
  · exact ringUpper_add r z w
  · exact ringLower_add r z w

/-- `channelPair r` maps the algebra unit to `(1, 1)`. -/
theorem channelPair_ringOne (r : R) :
    channelPair r ringOne = (1, 1) := by
  apply Prod.ext
  · exact ringUpper_ringOne r
  · exact ringLower_ringOne r

/-- Factorization of the quadratic norm form through channel coordinates. -/
theorem channelPair_normForm (σ r : R) (hr : r ^ 2 = σ) (z : RingCell R) :
    ringNormForm σ z = (channelPair r z).1 * (channelPair r z).2 :=
  ringNormForm_eq_upper_mul_lower hr z

/-- Hyperbolic decomposition: channel coordinate projection is bijective and intertwines
the algebra structure (multiplication, addition, unit) with componentwise operations,
factoring the norm form into the product of channels. -/
theorem hyperbolic_decomposition {σ r s h : R} (hr : r ^ 2 = σ)
    (hrs : r * s = 1) (hh : 2 * h = 1) :
    Function.Bijective (channelPair r) ∧
    (∀ z w : RingCell R,
      channelPair r (ringMul σ z w) =
        ((channelPair r z).1 * (channelPair r w).1,
         (channelPair r z).2 * (channelPair r w).2)) ∧
    (∀ z w : RingCell R,
      channelPair r (z + w) =
        ((channelPair r z).1 + (channelPair r w).1,
         (channelPair r z).2 + (channelPair r w).2)) ∧
    channelPair r ringOne = (1, 1) ∧
    (∀ z : RingCell R, ringNormForm σ z = (channelPair r z).1 * (channelPair r z).2) :=
  ⟨channelPair_bijective hrs hh,
   channelPair_mul σ r hr,
   channelPair_add r,
   channelPair_ringOne r,
   channelPair_normForm σ r hr⟩

/-- Hyperbolic decomposition under `IsUnit` hypotheses on `2` and `r`. -/
theorem hyperbolic_decomposition_of_isUnit {σ r : R} (hr : r ^ 2 = σ)
    (hu2 : IsUnit (2 : R)) (hur : IsUnit r) :
    Function.Bijective (channelPair r) ∧
    (∀ z w : RingCell R,
      channelPair r (ringMul σ z w) =
        ((channelPair r z).1 * (channelPair r w).1,
         (channelPair r z).2 * (channelPair r w).2)) ∧
    (∀ z w : RingCell R,
      channelPair r (z + w) =
        ((channelPair r z).1 + (channelPair r w).1,
         (channelPair r z).2 + (channelPair r w).2)) ∧
    channelPair r ringOne = (1, 1) ∧
    (∀ z : RingCell R, ringNormForm σ z = (channelPair r z).1 * (channelPair r z).2) := by
  rcases hu2 with ⟨u2, hu2_eq⟩
  rcases hur with ⟨ur, hur_eq⟩
  have hh : 2 * (↑(u2⁻¹) : R) = 1 := by
    rw [← hu2_eq, ← Units.val_mul, mul_inv_cancel, Units.val_one]
  have hrs : r * (↑(ur⁻¹) : R) = 1 := by
    rw [← hur_eq, ← Units.val_mul, mul_inv_cancel, Units.val_one]
  exact hyperbolic_decomposition hr hrs hh

end Decomposition

section DiagonalFold

variable {R : Type*} [CommRing R]

/-- A single sequential token transition step factors diagonally through channel coordinates. -/
theorem channelPair_tokenStep {σ r : R} (hr : r ^ 2 = σ) (z w : RingCell R) :
    channelPair r (tokenStep σ z w) =
      ((channelPair r z).1 * (channelPair r w).1,
       (channelPair r z).2 * (channelPair r w).2) :=
  channelPair_mul σ r hr z w

/-- Sequential fold of token points initialized at `ringOne` factors diagonally
into independent scalar list products per channel. -/
theorem channelPair_tokenFold {σ r : R} (hr : r ^ 2 = σ) (xs : List (RingCell R)) :
    channelPair r (tokenFold σ xs) =
      ((xs.map (ringUpper r)).prod, (xs.map (ringLower r)).prod) := by
  dsimp [channelPair, tokenFold]
  rw [ringUpper_foldl_ringOne hr xs, ringLower_foldl_ringOne hr xs]

variable {R₀ : Type} [CommRing R₀] {K : Type} [DecidableEq K]

/-- Table memory update at a key factors diagonally: reading at `x = k` multiplies
the channel coordinates by `v`'s channels, while all other keys `x ≠ k` remain unchanged. -/
theorem channelPair_memoryStep {σ r : R₀} (hr : r ^ 2 = σ)
    (k : K) (v : RingCell R₀) (M : HMap K (RingCell R₀)) (x : K) :
    channelPair r ((memoryStep σ k v M) x) =
      if x = k then
        (ringUpper r v * (channelPair r (M x)).1,
         ringLower r v * (channelPair r (M x)).2)
      else
        channelPair r (M x) := by
  simp only [memoryStep, insertWith]
  split_ifs with h
  · simp only [channelPair]
    rw [ringUpper_mul hr, ringLower_mul hr]
  · rfl

/-- Contract pairing diagonal factoring of sequential token folds and table memory updates. -/
theorem channel_diagonal_fold {σ r : R₀} (hr : r ^ 2 = σ)
    (xs : List (RingCell R₀)) (k : K) (v : RingCell R₀) (M : HMap K (RingCell R₀)) (x : K) :
    channelPair r (tokenFold σ xs) =
      ((xs.map (ringUpper r)).prod, (xs.map (ringLower r)).prod) ∧
    channelPair r ((memoryStep σ k v M) x) =
      if x = k then
        (ringUpper r v * (channelPair r (M x)).1,
         ringLower r v * (channelPair r (M x)).2)
      else
        channelPair r (M x) :=
  ⟨channelPair_tokenFold hr xs, channelPair_memoryStep hr k v M x⟩

end DiagonalFold

section Field

variable {K : Type*} [Field K]

/-- Explicit inverse reconstruction map over a field with direct division notation. -/
def channelInvField (r : K) (p : K × K) : RingCell K :=
  ((p.1 + p.2) / 2, (p.1 - p.2) / (2 * r))

/-- Equivalence of the field division reconstruction formula with `channelInv`. -/
theorem channelInvField_eq (r : K) (p : K × K) :
    channelInvField r p = channelInv r⁻¹ (2 : K)⁻¹ p := by
  simp only [channelInvField, channelInv, ringOfHalves]
  congr 1
  · ring
  · ring

/-- Hyperbolic decomposition specialized to fields of characteristic different from 2. -/
theorem hyperbolic_decomposition_field {σ r : K} (hr : r ^ 2 = σ)
    (h2 : (2 : K) ≠ 0) (hr0 : r ≠ 0) :
    Function.Bijective (channelPair r) ∧
    (∀ z w : RingCell K,
      channelPair r (ringMul σ z w) =
        ((channelPair r z).1 * (channelPair r w).1,
         (channelPair r z).2 * (channelPair r w).2)) ∧
    (∀ z w : RingCell K,
      channelPair r (z + w) =
        ((channelPair r z).1 + (channelPair r w).1,
         (channelPair r z).2 + (channelPair r w).2)) ∧
    channelPair r ringOne = (1, 1) ∧
    (∀ z : RingCell K, ringNormForm σ z = (channelPair r z).1 * (channelPair r z).2) :=
  hyperbolic_decomposition hr (mul_inv_cancel₀ hr0) (mul_inv_cancel₀ h2)

end Field

end Nous.Model
