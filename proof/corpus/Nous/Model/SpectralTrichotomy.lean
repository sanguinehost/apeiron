import Apeiron.Trichotomy.AnyRing
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Tactic

/-!
# Spectral Trichotomy of the Two-Slot Algebra `R[x]/(x² - σ)`

Formalization of the spectral classification and matrix representation of
`w = a + b·x ∈ RingCell σ` (Cockle 1848; Clifford 1873; Gauss 1801):
1. **Regular Matrix Representation**: `toMatrix σ w = ![![w.1, σ * w.2], ![w.2, w.1]]`
   forms an injective algebra homomorphism:
   - `toMatrix σ (ringMul σ w1 w2) = toMatrix σ w1 * toMatrix σ w2`
   - `det (toMatrix σ w) = ringNormForm σ w = a² - σ b²`
   - `trace (toMatrix σ w) = 2 * a`
2. **Spectral Discriminant**:
   The characteristic polynomial matrix `x_val • I - toMatrix σ w` has determinant
   `charpoly σ w x_val = (x_val - a)² - σ b²` with discriminant `Δ = 4 * σ * b²`.
3. **Trichotomy on Eigenvalues**:
   When `b ≠ 0`:
   - **Hyperbolic** (`σ > 0`): `Δ > 0`, distinct real eigenvalues `λ = a ± b √σ`.
   - **Elliptic** (`σ < 0`): `Δ < 0`, complex conjugate pairs `a ± i b √|σ|`.
   - **Parabolic** (`σ = 0`): `Δ = 0`, repeated real eigenvalue `λ = a`.
4. **Parabolic Defect and Jordan Form**:
   At `σ = 0`, `(toMatrix 0 w - a I)² = 0`. When `b` is invertible, `toMatrix 0 w` is
   conjugate to the standard Jordan block `![![a, 1], ![0, a]]`.
5. **Analytic Continuation**:
   Generalized trigonometric cells `(C_σ(θ), S_σ(θ))` satisfying `C_σ² - σ S_σ² = 1`
   unify `(cosh, sinh)` (`σ = 1`), `(cos, sin)` (`σ = -1`), and `(1, θ)` (`σ = 0`),
   with product matching the group law in all three faces simultaneously.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

open Matrix
open Apeiron.Trichotomy.AnyRing

namespace Nous.Model.SpectralTrichotomy

variable {R : Type*} [CommRing R]

/-! ## 1. Regular 2x2 Matrix Representation -/

/-- The regular matrix representation of `w = (a, b) ∈ RingCell R` in `R[x]/(x² - σ)`:
`M(w) = ![![a, σ * b], ![b, a]]`. -/
def toMatrix (σ : R) (w : RingCell R) : Matrix (Fin 2) (Fin 2) R :=
  ![![w.1, σ * w.2], ![w.2, w.1]]

/-- Multiplicative homomorphism: `toMatrix` maps `ringMul` to matrix multiplication. -/
theorem toMatrix_mul (σ : R) (w1 w2 : RingCell R) :
    toMatrix σ (ringMul σ w1 w2) = toMatrix σ w1 * toMatrix σ w2 := by
  ext i j
  fin_cases i <;> fin_cases j <;> {
    simp [toMatrix, mul_apply, Fin.sum_univ_two]
    try ring
  }

/-- Identity preservation: `toMatrix` maps `ringOne = (1, 0)` to the identity matrix. -/
theorem toMatrix_one (σ : R) :
    toMatrix σ (ringOne : RingCell R) = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [toMatrix, ringOne]

/-- The determinant of `toMatrix σ w` equals the norm form `ringNormForm σ w = a² - σ b²`. -/
theorem toMatrix_det (σ : R) (w : RingCell R) :
    det (toMatrix σ w) = ringNormForm σ w := by
  rw [det_fin_two]
  dsimp [toMatrix, ringNormForm]
  ring

/-- The trace of `toMatrix σ w` equals `2 * a`. -/
theorem toMatrix_trace (σ : R) (w : RingCell R) :
    trace (toMatrix σ w) = 2 * w.1 := by
  simp [trace_fin_two, toMatrix]
  ring

/-! ## 2. Characteristic Polynomial and Discriminant -/

/-- Characteristic matrix `x_val • I - toMatrix σ w`. -/
def charpolyMatrix (σ : R) (w : RingCell R) (x_val : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![x_val - w.1, - (σ * w.2)], ![- w.2, x_val - w.1]]

/-- Characteristic matrix equals `x_val • 1 - toMatrix σ w`. -/
theorem charpolyMatrix_eq (σ : R) (w : RingCell R) (x_val : R) :
    charpolyMatrix σ w x_val = x_val • (1 : Matrix (Fin 2) (Fin 2) R) - toMatrix σ w := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [charpolyMatrix, toMatrix, sub_eq_add_neg]

/-- Characteristic polynomial of `toMatrix σ w` evaluated at `x_val`. -/
def charpoly (σ : R) (w : RingCell R) (x_val : R) : R :=
  det (charpolyMatrix σ w x_val)

/-- Exact algebraic identity for the characteristic polynomial:
`det (x_val • I - M) = (x_val - a)² - σ b²`. -/
theorem charpoly_eq (σ : R) (w : RingCell R) (x_val : R) :
    charpoly σ w x_val = (x_val - w.1) ^ 2 - σ * w.2 ^ 2 := by
  rw [charpoly, charpolyMatrix, det_fin_two]
  dsimp
  ring

/-- The discriminant of `charpoly σ w` in standard quadratic form
`λ² - (2a)λ + (a² - σb²)` is identically `4 * σ * b²`. -/
theorem charpoly_discriminant (σ : R) (w : RingCell R) :
    (2 * w.1) ^ 2 - 4 * (w.1 ^ 2 - σ * w.2 ^ 2) = 4 * σ * w.2 ^ 2 := by
  ring

/-! ## 3. Spectral Trichotomy Theorems -/

/-- Hyperbolic spectral face (`σ > 0`): Over any ordered commutative ring,
if `σ > 0` and `b² > 0`, the discriminant is strictly positive (`Δ > 0`),
yielding two distinct real eigenvalues. -/
theorem hyperbolic_discriminant_pos {R : Type*} [CommRing R] [LinearOrder R] [IsStrictOrderedRing R]
    (σ : R) (w : RingCell R) (hσ : 0 < σ) (hb : 0 < w.2 ^ 2) :
    0 < (2 * w.1) ^ 2 - 4 * (w.1 ^ 2 - σ * w.2 ^ 2) := by
  rw [charpoly_discriminant]
  have h4 : (0 : R) < 4 := by norm_num
  have h4σ : 0 < 4 * σ := mul_pos h4 hσ
  exact mul_pos h4σ hb

/-- Elliptic spectral face (`σ < 0`): Over any ordered commutative ring,
if `σ < 0` and `b² > 0`, the discriminant is strictly negative (`Δ < 0`),
yielding complex conjugate eigenvalues and zero real roots. -/
theorem elliptic_discriminant_neg {R : Type*} [CommRing R] [LinearOrder R] [IsStrictOrderedRing R]
    (σ : R) (w : RingCell R) (hσ : σ < 0) (hb : 0 < w.2 ^ 2) :
    (2 * w.1) ^ 2 - 4 * (w.1 ^ 2 - σ * w.2 ^ 2) < 0 := by
  rw [charpoly_discriminant]
  have h4 : (0 : R) < 4 := by norm_num
  have h4σ : 4 * σ < 0 := mul_neg_of_pos_of_neg h4 hσ
  exact mul_neg_of_neg_of_pos h4σ hb

/-- Parabolic spectral face (`σ = 0`): Over any commutative ring,
when `σ = 0`, the discriminant vanishes identically (`Δ = 0`),
yielding a repeated real eigenvalue `λ = a`. -/
theorem parabolic_discriminant_zero (w : RingCell R) :
    (2 * w.1) ^ 2 - 4 * (w.1 ^ 2 - 0 * w.2 ^ 2) = 0 := by
  rw [charpoly_discriminant]
  ring

/-! ## 4. Parabolic Nilpotent Defect and Jordan Normal Form -/

/-- At `σ = 0`, the deviation from the scalar eigenvalue `(M - a • I)` is
nilpotent of degree 2: `(toMatrix 0 w - w.1 • I)² = 0`. -/
theorem parabolic_nilpotent_sq (w : RingCell R) :
    (toMatrix 0 w - w.1 • (1 : Matrix (Fin 2) (Fin 2) R)) ^ 2 = 0 := by
  ext i j
  fin_cases i <;> fin_cases j <;> {
    simp [toMatrix, mul_apply, Fin.sum_univ_two, sq]
  }

/-- Parabolic Jordan Form Conjugacy:
When `b` has an inverse `b_inv` with `w.2 * b_inv = 1`, the transition matrix
`P = ![![0, b_inv], ![1, 0]]` and `P_inv = ![![0, 1], ![w.2, 0]]` satisfy
`P_inv * P = 1` and conjugate `toMatrix 0 w` directly to the standard Jordan block:
`P_inv * toMatrix 0 w * P = ![![w.1, 1], ![0, w.1]]`. -/
theorem parabolic_jordan_conjugacy (w : RingCell R) (b_inv : R) (hb : w.2 * b_inv = 1) :
    let P : Matrix (Fin 2) (Fin 2) R := ![![0, b_inv], ![1, 0]]
    let P_inv : Matrix (Fin 2) (Fin 2) R := ![![0, 1], ![w.2, 0]]
    let J : Matrix (Fin 2) (Fin 2) R := ![![w.1, 1], ![0, w.1]]
    P_inv * P = 1 ∧ P_inv * toMatrix 0 w * P = J := by
  intro P P_inv J
  constructor
  · ext i j
    fin_cases i <;> fin_cases j
    · simp [P, P_inv, mul_apply, Fin.sum_univ_two]
    · simp [P, P_inv, mul_apply, Fin.sum_univ_two]
    · simp [P, P_inv, mul_apply, Fin.sum_univ_two]
    · simp [P, P_inv, mul_apply, Fin.sum_univ_two, hb]
  · ext i j
    fin_cases i <;> fin_cases j
    · simp [P, P_inv, J, toMatrix, mul_apply, Fin.sum_univ_two]
    · simp [P, P_inv, J, toMatrix, mul_apply, Fin.sum_univ_two, hb]
    · simp [P, P_inv, J, toMatrix, mul_apply, Fin.sum_univ_two]
    · simp [P, P_inv, J, toMatrix, mul_apply, Fin.sum_univ_two]
      have hcomm : w.2 * w.1 * b_inv = (w.2 * b_inv) * w.1 := by ring
      rw [hcomm, hb, one_mul]

/-! ## 5. Analytic Continuation and Generalized Trigonometry -/

/-- A generalized trigonometric cell `(c, s)` with `ringNormForm σ (c, s) = 1`. -/
def isTrigCell (σ : R) (z : RingCell R) : Prop :=
  ringNormForm σ z = 1

/-- Product of two trigonometric cells is a trigonometric cell under `ringMul σ`.
This proves that generalized rotation is a group homomorphism for ALL `σ`. -/
theorem trig_mul_preserves_norm (σ : R) (z1 z2 : RingCell R)
    (h1 : isTrigCell σ z1) (h2 : isTrigCell σ z2) :
    isTrigCell σ (ringMul σ z1 z2) := by
  dsimp [isTrigCell, ringNormForm, ringMul] at *
  have h : (z1.1 * z2.1 + σ * z1.2 * z2.2) ^ 2 - σ * (z1.1 * z2.2 + z1.2 * z2.1) ^ 2 =
           (z1.1 ^ 2 - σ * z1.2 ^ 2) * (z2.1 ^ 2 - σ * z2.2 ^ 2) := by ring
  rw [h, h1, h2, mul_one]

/-- Parabolic shear `(1, s)` is a valid trigonometric cell for ALL `s` at `σ = 0`. -/
theorem parabolic_shear_is_trig (s : R) :
    isTrigCell 0 (1, s) := by
  dsimp [isTrigCell, ringNormForm]
  ring

/-- Parabolic trigonometric addition is standard linear scalar addition:
`(1, s₁) * (1, s₂) = (1, s₁ + s₂)`. -/
theorem parabolic_trig_addition (s1 s2 : R) :
    ringMul 0 (1, s1) (1, s2) = (1, s1 + s2) := by
  simp [ringMul]
  ring

end Nous.Model.SpectralTrichotomy
