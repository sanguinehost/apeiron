import Apeiron.Trichotomy.AnyRing
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Tactic

/-!
# Pairwise 2x2 Coordinate Transforms in R[x]/(x² - σ)

This module formalizes volume-preserving 2x2 coordinate transforms across the
trichotomy `R[x]/(x² - σ)` for pre-quantization weight conditioning:
- **Hyperbolic (`σ = 1`)**: Lorentz boost `B(c, s)` with `c² - s² = 1`.
- **Elliptic (`σ = -1`)**: Givens rotation `R(c, s)` with `c² + s² = 1`.
- **Parabolic (`σ = 0`)**: Nilpotent shear `S(s)` with determinant 1.

All three families preserve 2D phase volume (`det = 1`), ensuring that pre-quantization
rotations or boosts do not compress or destroy weight representation capacity.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unnecessarySeqFocus false
set_option linter.unusedSimpArgs false

open Matrix
open Apeiron.Trichotomy.AnyRing

namespace Nous.Combinator.CellTransform

variable {R : Type*} [CommRing R]

/-! ### 1. Hyperbolic Lorentz Boost (σ = 1) -/

/-- Hyperbolic boost matrix parameterised by `(c, s)` with `c² - s² = 1`. -/
def lorentzBoost (c s : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![c, s], ![s, c]]

/-- Determinant of Lorentz boost is `c² - s²`. -/
theorem det_lorentzBoost (c s : R) :
    det (lorentzBoost c s) = c ^ 2 - s ^ 2 := by
  rw [det_fin_two]
  dsimp [lorentzBoost]
  ring

/-- A boost with `c² - s² = 1` is strictly volume-preserving (`det = 1`). -/
theorem lorentzBoost_det_one (c s : R) (h : c ^ 2 - s ^ 2 = 1) :
    det (lorentzBoost c s) = 1 := by
  rw [det_lorentzBoost, h]

theorem lorentzBoost_mulVec (c s x y : R) :
    lorentzBoost c s *ᵥ ![x, y] = ![c * x + s * y, s * x + c * y] := by
  ext i
  fin_cases i
  · simp [lorentzBoost, mulVec, dotProduct]
  · simp [lorentzBoost, mulVec, dotProduct]

/-- Lorentz boost preserves the hyperbolic norm form `x² - y²`. -/
theorem lorentzBoost_preserves_norm (c s x y : R) (h : c ^ 2 - s ^ 2 = 1) :
    (lorentzBoost c s *ᵥ ![x, y]) 0 ^ 2 - (lorentzBoost c s *ᵥ ![x, y]) 1 ^ 2 = x ^ 2 - y ^ 2 := by
  have hv : lorentzBoost c s *ᵥ ![x, y] = ![c * x + s * y, s * x + c * y] :=
    lorentzBoost_mulVec c s x y
  rw [hv]
  dsimp
  have h1 : (c * x + s * y) ^ 2 - (s * x + c * y) ^ 2 = (c ^ 2 - s ^ 2) * (x ^ 2 - y ^ 2) := by ring
  rw [h1, h, one_mul]

/-- Inverse of Lorentz boost `B(c, s)` is `B(c, -s)`. -/
theorem lorentzBoost_inv (c s : R) (h : c ^ 2 - s ^ 2 = 1) :
    lorentzBoost c s * lorentzBoost c (-s) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j
  · simp [lorentzBoost, mul_apply, dotProduct, Fin.sum_univ_two]
    linear_combination h
  · simp [lorentzBoost, mul_apply, dotProduct, Fin.sum_univ_two]
    ring
  · simp [lorentzBoost, mul_apply, dotProduct, Fin.sum_univ_two]
    ring
  · simp [lorentzBoost, mul_apply, dotProduct, Fin.sum_univ_two]
    linear_combination h

/-! ### 2. Elliptic Givens Rotation (σ = -1) -/

/-- Elliptic Givens rotation matrix parameterised by `(c, s)` with `c² + s² = 1`. -/
def givensRot (c s : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![c, -s], ![s, c]]

/-- Determinant of Givens rotation is `c² + s²`. -/
theorem det_givensRot (c s : R) :
    det (givensRot c s) = c ^ 2 + s ^ 2 := by
  rw [det_fin_two]
  dsimp [givensRot]
  ring

/-- A Givens rotation with `c² + s² = 1` is strictly volume-preserving (`det = 1`). -/
theorem givensRot_det_one (c s : R) (h : c ^ 2 + s ^ 2 = 1) :
    det (givensRot c s) = 1 := by
  rw [det_givensRot, h]

theorem givensRot_mulVec (c s x y : R) :
    givensRot c s *ᵥ ![x, y] = ![c * x - s * y, s * x + c * y] := by
  ext i
  fin_cases i
  · simp [givensRot, mulVec, dotProduct, sub_eq_add_neg]
  · simp [givensRot, mulVec, dotProduct]

/-- Givens rotation preserves the Euclidean norm form `x² + y²`. -/
theorem givensRot_preserves_norm (c s x y : R) (h : c ^ 2 + s ^ 2 = 1) :
    (givensRot c s *ᵥ ![x, y]) 0 ^ 2 + (givensRot c s *ᵥ ![x, y]) 1 ^ 2 = x ^ 2 + y ^ 2 := by
  have hv : givensRot c s *ᵥ ![x, y] = ![c * x - s * y, s * x + c * y] :=
    givensRot_mulVec c s x y
  rw [hv]
  dsimp
  have h1 : (c * x - s * y) ^ 2 + (s * x + c * y) ^ 2 = (c ^ 2 + s ^ 2) * (x ^ 2 + y ^ 2) := by ring
  rw [h1, h, one_mul]

/-- Inverse of Givens rotation `R(c, s)` is `R(c, -s)`. -/
theorem givensRot_inv (c s : R) (h : c ^ 2 + s ^ 2 = 1) :
    givensRot c s * givensRot c (-s) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j
  · simp [givensRot, mul_apply, dotProduct, Fin.sum_univ_two]
    linear_combination h
  · simp [givensRot, mul_apply, dotProduct, Fin.sum_univ_two]
    ring
  · simp [givensRot, mul_apply, dotProduct, Fin.sum_univ_two]
    ring
  · simp [givensRot, mul_apply, dotProduct, Fin.sum_univ_two]
    linear_combination h

/-! ### 3. Parabolic Nilpotent Shear (σ = 0) -/

/-- Parabolic shear matrix parameterised by shear parameter `s`. -/
def parabolicShear (s : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![1, s], ![0, 1]]

/-- Determinant of parabolic shear is unconditionally 1. -/
theorem det_parabolicShear (s : R) :
    det (parabolicShear s) = 1 := by
  rw [det_fin_two]
  dsimp [parabolicShear]
  ring

/-- Inverse of shear `S(s)` is `S(-s)`. -/
theorem parabolicShear_inv (s : R) :
    parabolicShear s * parabolicShear (-s) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j
  · simp [parabolicShear, mul_apply, dotProduct, Fin.sum_univ_two]
  · simp [parabolicShear, mul_apply, dotProduct, Fin.sum_univ_two]
  · simp [parabolicShear, mul_apply, dotProduct, Fin.sum_univ_two]
  · simp [parabolicShear, mul_apply, dotProduct, Fin.sum_univ_two]

end Nous.Combinator.CellTransform
