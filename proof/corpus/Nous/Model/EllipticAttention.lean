import Apeiron.Trichotomy.AnyRing
import Nous.Model.RopeOrbit
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Tactic

/-!
# Native Elliptic Phase Attention in `R[x]/(x² - σ)` with `σ < 0`

In `Nous.Model.Attention`, the elliptic face was identified as an excluded population because
`r² = σ < 0` admits no real root witness, precluding hyperbolic idempotent channel splitting.
This module establishes the native formulation of attention on the elliptic face: **unitary phase
interference**.

## Mathematical Setting and Attribution

For a quadratic parameter `σ` in a commutative ring `R`, the two-slot algebra `R[x]/(x² - σ)`
represents points `z = (a, b)` written `a + b·x` with multiplication `ringMul σ` and quadratic
norm form `ringNormForm σ z = a² - σ b²`.

* **Complex arithmetic**: Rafael Bombelli, *L'Algebra* (1572), introduced imaginary numbers
  satisfying `x² = -1` to resolve the irreducible case (*casus irreducibilis*) of cubic equations.
* **Norm multiplicativity**: Carl Friedrich Gauss, *Disquisitiones Arithmeticae* (1801),
  established binary quadratic forms and the composition law `N(z · w) = N(z) N(w)`, showing that
  the quadratic norm form is a homomorphism into the base ring.
* **Rotary Position Embedding (RoPE)**: Jianlin Su et al. (2021/2024), representing sequence
  positions as elements of the one-parameter unitary circle group acting by isometric rotation.

## Algebraic Properties Established

1. **Norm Multiplicativity and Isometry**:
   `ringNormForm σ (ringMul σ q k) = ringNormForm σ q * ringNormForm σ k` (Gauss 1801).
   For any unitary rotor `u` (`ringNormForm σ u = 1`), the left action `z ↦ ringMul σ u z` is an
   exact isometry: `ringNormForm σ (ringMul σ u z) = ringNormForm σ z`.
2. **Positive Definiteness**:
   Over a linearly ordered field with `σ < 0`, `ringNormForm σ z ≥ 0`, with equality if and only if
   `z = (0, 0)`.
3. **Absence of Zero Divisors**:
   When `σ < 0`, `ringMul σ z w = (0, 0) ↔ z = (0, 0) ∨ w = (0, 0)`.
   Unlike the hyperbolic face (`σ > 0`), which splits into lightcone null rays `(1, ±r)` where
   `N(z) = 0` despite `z ≠ 0`, the elliptic face contains no nonzero nilpotent elements or zero
   divisors. Unitary phase attention rotates representations without dissipating or leaking energy.
4. **Phase Alignment and Rotary Orbit**:
   The relative phase alignment `ellipticPhaseAlign σ q k = ringMul σ q (ringConj k)` extracts the
   interference pattern between query `q` and key `k`. At `σ = -1` over `ℝ`:
   * `(ellipticPhaseAlign (-1) q k).1 = rhoDot q k` matches the Euclidean inner product.
   * `ellipticPhaseAlign (-1) (rho θ) (rho φ) = rho (θ - φ)` extracts the relative position rotor.
   * Modulating queries and keys with RoPE position rotors preserves total representation norm and
     makes attention scores depend strictly on position differences `n - m`, directly recovering the
     rotary orbit properties of `Nous.Model.RopeOrbit`.

## Trust Boundary

All algebraic theorems are checked by the Lean kernel over an arbitrary commutative ring `R`,
ordered field `F`, or the real continuum `ℝ`. No claims reach `rustc`/LLVM codegen, the machine-word
kernel ABI at `ZMod (2^64)`, or hardware floating-point rounding.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

section PhaseAlgebra

variable {R : Type*} [CommRing R]

/-- The phase multiplication of two elements in `R[x]/(x² - σ)`.
    Computes `(a + bx)(c + dx) = (ac + σbd) + (ad + bc)x`. -/
def ellipticPhaseMul (σ : R) (q k : RingCell R) : RingCell R :=
  ringMul σ q k

/-- The relative phase alignment of query `q` and key `k`: `q · conj(k)`.
    In complex numbers `(a + bi)(c - di) = (ac + bd) + (bc - ad)i`. -/
def ellipticPhaseAlign (σ : R) (q k : RingCell R) : RingCell R :=
  ringMul σ q (ringConj k)

/-- The elliptic attention score: the in-phase (real) component of `ellipticPhaseAlign σ q k`. -/
def ellipticScore (σ : R) (q k : RingCell R) : R :=
  (ellipticPhaseAlign σ q k).1

/-- Coordinate expansion of `ellipticPhaseMul`. -/
theorem elliptic_phase_mul_formula (σ : R) (q k : RingCell R) :
    ellipticPhaseMul σ q k = (q.1 * k.1 + σ * q.2 * k.2, q.1 * k.2 + q.2 * k.1) :=
  rfl

/-- Coordinate expansion of `ellipticPhaseAlign`. -/
theorem elliptic_phase_align_formula (σ : R) (q k : RingCell R) :
    ellipticPhaseAlign σ q k = (q.1 * k.1 - σ * q.2 * k.2, -q.1 * k.2 + q.2 * k.1) := by
  simp only [ellipticPhaseAlign]
  ext <;> ring

/-- Coordinate expansion of `ellipticScore`. -/
theorem elliptic_score_formula (σ : R) (q k : RingCell R) :
    ellipticScore σ q k = q.1 * k.1 - σ * q.2 * k.2 := by
  simp only [ellipticScore, ellipticPhaseAlign]
  ring

/-- **Gauss's multiplicative norm identity**: the norm form of the phase product is the
    product of the norm forms. -/
theorem elliptic_normForm_mul (σ : R) (q k : RingCell R) :
    ringNormForm σ (ellipticPhaseMul σ q k) = ringNormForm σ q * ringNormForm σ k :=
  ringNormForm_mul σ q k

/-- The norm form of the relative phase alignment equals the product of the norm forms,
    because `ringNormForm σ (ringConj k) = ringNormForm σ k`. -/
theorem elliptic_normForm_align (σ : R) (q k : RingCell R) :
    ringNormForm σ (ellipticPhaseAlign σ q k) = ringNormForm σ q * ringNormForm σ k := by
  unfold ellipticPhaseAlign
  rw [ringNormForm_mul, ringNormForm_conj]

/-- Elliptic phase multiplication is commutative. -/
theorem elliptic_phase_mul_comm (σ : R) (q k : RingCell R) :
    ellipticPhaseMul σ q k = ellipticPhaseMul σ k q :=
  ringMul_comm σ q k

/-- Elliptic phase multiplication is associative. -/
theorem elliptic_phase_mul_assoc (σ : R) (q k v : RingCell R) :
    ellipticPhaseMul σ (ellipticPhaseMul σ q k) v =
      ellipticPhaseMul σ q (ellipticPhaseMul σ k v) :=
  ringMul_assoc σ q k v

/-- Identity of elliptic phase multiplication. -/
theorem elliptic_phase_mul_one (σ : R) (q : RingCell R) :
    ellipticPhaseMul σ ringOne q = q :=
  ringOne_mul σ q

end PhaseAlgebra

section UnitaryIsometry

variable {R : Type*} [CommRing R]

/-- A unitary rotor is an element of `R[x]/(x² - σ)` with norm form equal to 1. -/
def IsUnitary (σ : R) (u : RingCell R) : Prop :=
  ringNormForm σ u = 1

/-- The identity element `ringOne = (1, 0)` is unitary. -/
theorem isUnitary_ringOne (σ : R) : IsUnitary σ ringOne := by
  simp [IsUnitary, ringNormForm]

/-- The conjugate of a unitary element is unitary. -/
theorem isUnitary_conj {σ : R} {u : RingCell R} (hu : IsUnitary σ u) :
    IsUnitary σ (ringConj u) := by
  simpa [IsUnitary, ringNormForm_conj] using hu

/-- The product of two unitary elements is unitary. -/
theorem isUnitary_mul {σ : R} {u v : RingCell R} (hu : IsUnitary σ u) (hv : IsUnitary σ v) :
    IsUnitary σ (ringMul σ u v) := by
  unfold IsUnitary at *
  rw [ringNormForm_mul, hu, hv, mul_one]

/-- **Unitary Isometry**: Left multiplication by a unitary rotor preserves the quadratic norm form
    identically. There is no attenuation, dissipation, or amplification of signal energy. -/
theorem elliptic_isometry {σ : R} {u : RingCell R} (hu : IsUnitary σ u) (z : RingCell R) :
    ringNormForm σ (ringMul σ u z) = ringNormForm σ z := by
  rw [ringNormForm_mul, hu, one_mul]

/-- Pre-multiplying query and key by unitary position rotors preserves their alignment norm. -/
theorem elliptic_isometry_align {σ : R} {u v : RingCell R}
    (hu : IsUnitary σ u) (hv : IsUnitary σ v) (q k : RingCell R) :
    ringNormForm σ (ellipticPhaseAlign σ (ringMul σ u q) (ringMul σ v k)) =
      ringNormForm σ (ellipticPhaseAlign σ q k) := by
  rw [elliptic_normForm_align, elliptic_normForm_align,
    elliptic_isometry hu q, elliptic_isometry hv k]

end UnitaryIsometry

section PositiveDefiniteness

variable {F : Type*} [Field F] [LinearOrder F] [IsStrictOrderedRing F]

/-- On the elliptic face `σ < 0`, the quadratic norm form is non-negative for all elements. -/
theorem elliptic_normForm_nonneg {σ : F} (hσ : σ < 0) (z : RingCell F) :
    0 ≤ ringNormForm σ z := by
  have h1 : 0 ≤ z.1 ^ 2 := sq_nonneg z.1
  have h2 : 0 ≤ -σ := by linarith
  have h3 : 0 ≤ z.2 ^ 2 := sq_nonneg z.2
  have h4 : 0 ≤ -σ * z.2 ^ 2 := mul_nonneg h2 h3
  have h5 : ringNormForm σ z = z.1 ^ 2 + (-σ * z.2 ^ 2) := by
    simp only [ringNormForm]
    ring
  rw [h5]
  exact add_nonneg h1 h4

/-- On the elliptic face `σ < 0`, the norm form vanishes if and only if the element is zero. -/
theorem elliptic_normForm_eq_zero_iff {σ : F} (hσ : σ < 0) (z : RingCell F) :
    ringNormForm σ z = 0 ↔ z = (0, 0) := by
  constructor
  · intro h
    have h1 : 0 ≤ z.1 ^ 2 := sq_nonneg z.1
    have hneg : 0 < -σ := by linarith
    have h2 : 0 ≤ -σ * z.2 ^ 2 := mul_nonneg (le_of_lt hneg) (sq_nonneg z.2)
    have hsum : z.1 ^ 2 + (-σ * z.2 ^ 2) = 0 := by
      calc z.1 ^ 2 + (-σ * z.2 ^ 2) = ringNormForm σ z := by
            simp only [ringNormForm]; ring
        _ = 0 := h
    have hz1_sq : z.1 ^ 2 = 0 := by
      have : z.1 ^ 2 ≤ 0 := by linarith [hsum, h2]
      exact le_antisymm this h1
    have hz1 : z.1 = 0 := sq_eq_zero_iff.mp hz1_sq
    have hz2_term : -σ * z.2 ^ 2 = 0 := by
      linarith [hsum, hz1_sq]
    have hz2_sq : z.2 ^ 2 = 0 := by
      cases mul_eq_zero.mp hz2_term with
      | inl h_neg => linarith
      | inr h_sq => exact h_sq
    have hz2 : z.2 = 0 := sq_eq_zero_iff.mp hz2_sq
    exact Prod.ext hz1 hz2
  · rintro rfl
    simp only [ringNormForm]
    ring

/-- For any nonzero element on the elliptic face, its quadratic norm form is strictly positive. -/
theorem elliptic_normForm_pos_of_ne_zero {σ : F} (hσ : σ < 0) {z : RingCell F}
    (hz : z ≠ (0, 0)) : 0 < ringNormForm σ z := by
  have hnonneg := elliptic_normForm_nonneg hσ z
  rcases lt_or_eq_of_le hnonneg with hpos | heq
  · exact hpos
  · exfalso
    have hz0 := (elliptic_normForm_eq_zero_iff hσ z).mp heq.symm
    exact hz hz0

/-- **Absence of Zero Divisors**: On the elliptic face `σ < 0`, a product of elements is zero
    if and only if at least one factor is zero. No nonzero elements multiply to zero, preventing
    energy dissipation or collapse into singular subspaces. -/
theorem elliptic_no_zero_divisors {σ : F} (hσ : σ < 0) (z w : RingCell F) :
    ringMul σ z w = (0, 0) ↔ z = (0, 0) ∨ w = (0, 0) := by
  constructor
  · intro h
    have hN : ringNormForm σ (ringMul σ z w) = 0 := by
      rw [h]
      simp only [ringNormForm]
      ring
    rw [ringNormForm_mul] at hN
    cases mul_eq_zero.mp hN with
    | inl hz =>
      left
      exact (elliptic_normForm_eq_zero_iff hσ z).mp hz
    | inr hw =>
      right
      exact (elliptic_normForm_eq_zero_iff hσ w).mp hw
  · rintro (rfl | rfl)
    · apply Prod.ext
      · simp
      · simp
    · apply Prod.ext
      · simp
      · simp

end PositiveDefiniteness

section HyperbolicContrast

variable {F : Type*} [Field F]

/-- Contrast: on the hyperbolic face `σ = 1`, the split-complex elements `(1, 1)` and `(1, -1)`
    are nonzero yet multiply to zero, forming zero divisors. On the elliptic face `σ < 0`,
    this is impossible by `elliptic_no_zero_divisors`. -/
theorem hyperbolic_has_zero_divisors :
    ringMul (1 : F) (1, 1) (1, -1) = (0, 0) ∧ (((1 : F), (1 : F)) ≠ (0, 0)) := by
  refine ⟨?_, ?_⟩
  · simp only [ringMul]
    ext <;> ring
  · intro h
    have : (1 : F) = 0 := congrArg Prod.fst h
    exact one_ne_zero this

end HyperbolicContrast

section RopeConnection

open Nous.Model

/-- Over `ℝ` at `σ = -1`, `ellipticScore` is exactly the Euclidean inner product `rhoDot`. -/
theorem ellipticScore_eq_rhoDot (q k : RingCell ℝ) :
    ellipticScore (-1 : ℝ) q k = rhoDot q k := by
  simp only [ellipticScore, ellipticPhaseAlign, rhoDot]
  ring

/-- The RoPE rotor element `rho θ = (cos θ, sin θ)` is unitary at `σ = -1`. -/
theorem rho_isUnitary (θ : ℝ) : IsUnitary (-1 : ℝ) (rho θ) := by
  unfold IsUnitary ringNormForm rho
  have h : Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := Real.cos_sq_add_sin_sq θ
  linear_combination h

/-- Over `ℝ` at `σ = -1`, `ringMul (-1)` coincides with `rhoMul`. -/
theorem ringMul_neg_one_eq_rhoMul (z w : RingCell ℝ) :
    ringMul (-1 : ℝ) z w = rhoMul z w := by
  apply Prod.ext
  · simp only [rhoMul]; ring
  · simp only [rhoMul]

/-- The conjugate of `rho θ` is `rho (-θ)`. -/
theorem ringConj_rho (θ : ℝ) : ringConj (rho θ) = rho (-θ) := by
  apply Prod.ext
  · simp [rho, Real.cos_neg]
  · simp [rho, Real.sin_neg]

/-- Angle addition under elliptic phase multiplication: multiplying `rho θ` and `rho φ` yields
    `rho (θ + φ)`, matching `rho_add`. -/
theorem rho_phaseMul_add (θ φ : ℝ) :
    ellipticPhaseMul (-1 : ℝ) (rho θ) (rho φ) = rho (θ + φ) := by
  rw [ellipticPhaseMul, ringMul_neg_one_eq_rhoMul, ← rho_add]

/-- Relative phase extraction under elliptic phase alignment: aligning `rho θ` with `rho φ`
    yields `rho (θ - φ)`. -/
theorem rho_phaseAlign_sub (θ φ : ℝ) :
    ellipticPhaseAlign (-1 : ℝ) (rho θ) (rho φ) = rho (θ - φ) := by
  unfold ellipticPhaseAlign
  rw [ringConj_rho, ringMul_neg_one_eq_rhoMul, ← rho_sub]

/-- The elliptic attention score between two pure rotors `rho θ` and `rho φ` is `cos(θ - φ)`. -/
theorem ellipticScore_rho (θ φ : ℝ) :
    ellipticScore (-1 : ℝ) (rho θ) (rho φ) = Real.cos (θ - φ) := by
  unfold ellipticScore
  rw [rho_phaseAlign_sub]
  rfl

/-- **RoPE Relative Position Invariance**: When queries and keys are rotated by positions `m`
    and `n` using RoPE rotors `rho (m·θ)` and `rho (n·θ)`, the elliptic attention score
    depends solely on the position difference `n - m`, recovering `rho_relative_score`. -/
theorem elliptic_rope_relative_score (m n : ℕ) (θ : ℝ) (u v : RingCell ℝ) :
    ellipticScore (-1 : ℝ)
        (ringMul (-1 : ℝ) (rho ((m : ℝ) * θ)) u)
        (ringMul (-1 : ℝ) (rho ((n : ℝ) * θ)) v)
      = rhoDot u (rhoAct (((n : ℝ) - (m : ℝ)) * θ) v) := by
  rw [ellipticScore_eq_rhoDot]
  rw [ringMul_neg_one_eq_rhoMul, ringMul_neg_one_eq_rhoMul]
  change rhoDot (rhoAct ((m : ℝ) * θ) u) (rhoAct ((n : ℝ) * θ) v) = _
  exact rho_relative_score m n θ u v

/-- Any natural power `rhoPow k (rho θ)` is unitary at `σ = -1`.
    The entire per-position RoPE table remains on the unit circle without drift. -/
theorem rhoPow_isUnitary (θ : ℝ) (k : ℕ) :
    IsUnitary (-1 : ℝ) (rhoPow k (rho θ)) := by
  rw [← rho_nat]
  exact rho_isUnitary ((k : ℝ) * θ)

end RopeConnection

section StreamInvariance

variable {R : Type*} [CommRing R]

/-- In elliptic phase attention, every token in a key-value stream maintains its norm
    under position rotation by unitary rotors. -/
theorem elliptic_stream_norm_preservation (σ : R) (rotors : List (RingCell R))
    (keys : List (RingCell R)) (hrot : ∀ r ∈ rotors, IsUnitary σ r)
    (hlen : rotors.length = keys.length) :
    (rotors.zip keys).map (fun p => ringNormForm σ (ringMul σ p.1 p.2)) =
      keys.map (ringNormForm σ) := by
  induction rotors generalizing keys with
  | nil =>
    cases keys with
    | nil => rfl
    | cons k ks => contradiction
  | cons r rs ih =>
    cases keys with
    | nil => contradiction
    | cons k ks =>
      simp only [List.zip_cons_cons, List.map_cons, List.length_cons] at hlen ⊢
      have hr : IsUnitary σ r := hrot r (List.mem_cons.2 (Or.inl rfl))
      have hrs : ∀ r' ∈ rs, IsUnitary σ r' := fun r' hr' =>
        hrot r' (List.mem_cons.2 (Or.inr hr'))
      have ih_res := ih ks hrs (Nat.succ.inj hlen)
      rw [elliptic_isometry hr k, ih_res]

end StreamInvariance

end Nous.Model
