import Mathlib
import Apeiron.Trichotomy.Basic

/-!
# The elliptic cell as an extra rebit (APEIRON-QARCH-L10)

The left-multiplication matrix `rep σ z` of a cell `z` in the basis `1, x` of `R[x]/(x² − σ)`
(`Apeiron.Trichotomy.mul`). For `σ = −1` the cell `a + b·i` is the real `2 × 2` matrix
`a·1 + b·J`. `realify` maps a complex `n × n` matrix `A + iB` to the real matrix `A ⊗ I + B ⊗ J`,
so the complex unit is carried by one extra real factor.

Anchor (not proved here): McKague, Mosca and Gisin, *Simulating quantum systems using real Hilbert
spaces*, PRL 102, 020505 (2009): one extra rebit carries the complex unit, so a register needs no
complex dtype; the elliptic cell `[[a, −b], [b, a]]` is that rebit.

Grade of the statements below: `proved`, relative to the canonical `Apeiron.Trichotomy` definitions.
-/

namespace Quantum
namespace EllipticRebit

open scoped Matrix
open Apeiron.Trichotomy (Cell mul normForm)

/--
 Left multiplication by `z` in the basis `1, x`: `rep σ z` acts on the coordinate column `(u, v)`
as `z · (u + v x)`. 

`Cell` [a point: two real numbers written side by side, a + b·x]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`ℝ` [the real numbers: every number on the number line]
-/
def rep (σ : ℝ) (z : Cell) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![z.1, σ * z.2; z.2, z.1]

/--
The rotation generator `J`, with `J² = −1`.

`ℝ` [the real numbers: every number on the number line]
-/
def J : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

/--
McKague–Mosca–Gisin realification `A + iB ↦ A ⊗ I + B ⊗ J`.

`×` [paired with, so ℝ × ℝ means two real numbers side by side]
`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
`ℝ` [the real numbers: every number on the number line]
-/
def realify {n : Type} [Fintype n] [DecidableEq n] (M : Matrix n n ℂ) :
    Matrix (n × Fin 2) (n × Fin 2) ℝ :=
  Matrix.kroneckerMap (· * ·) (M.map Complex.re) 1 + Matrix.kroneckerMap (· * ·) (M.map Complex.im) J

/--
Two `2 × 2` real matrices agree when their four entries agree.

`ℝ` [the real numbers: every number on the number line]
-/
theorem ext_fin2 {A B : Matrix (Fin 2) (Fin 2) ℝ} (h00 : A 0 0 = B 0 0) (h01 : A 0 1 = B 0 1)
    (h10 : A 1 0 = B 1 0) (h11 : A 1 1 = B 1 1) : A = B := by
  ext i j
  fin_cases i <;> fin_cases j
  · exact h00
  · exact h01
  · exact h10
  · exact h11

/-- `J² = −1`: the rotation generator squares to minus the identity. -/
theorem J_sq : J * J = -1 := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [J, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [J, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [J, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [J, Matrix.mul_apply, Fin.sum_univ_two]

/-- `Jᵀ = −J`: the rotation generator is skew. -/
theorem J_transpose : Jᵀ = -J := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [J]
  · simp [J]
  · simp [J]
  · simp [J]

/-! ### Step 1: every σ -/

/--
 Left multiplication is a homomorphism: `rep σ (z·w) = rep σ z * rep σ w`, for every `σ`. 

`Cell` [a point: two real numbers written side by side, a + b·x]
`mul` [multiply two points together, the σ way]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`ℝ` [the real numbers: every number on the number line]
-/
theorem rep_mul (σ : ℝ) (z w : Cell) : rep σ (mul σ z w) = rep σ z * rep σ w := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
    ring
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
    ring
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
    ring

/--
 `rep σ 1 = 1` for every `σ`. 

`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`ℝ` [the real numbers: every number on the number line]
-/
theorem rep_one (σ : ℝ) : rep σ Apeiron.Trichotomy.one = 1 := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep]
  · simp [rep]
  · simp [rep]
  · simp [rep]

/--
 `rep σ` is additive: it sends `z + w` to `rep σ z + rep σ w`. 

`Cell` [a point: two real numbers written side by side, a + b·x]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`ℝ` [the real numbers: every number on the number line]
-/
theorem rep_add (σ : ℝ) (z w : Cell) : rep σ (z + w) = rep σ z + rep σ w := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep]
  · simp [rep]; ring
  · simp [rep]
  · simp [rep]

/--
 `rep σ` is injective for every `σ`: its entries `(0,0)` and `(1,0)` recover `z`. 

`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`ℝ` [the real numbers: every number on the number line]
-/
theorem rep_injective (σ : ℝ) : Function.Injective (rep σ) := by
  intro z w h
  have h1 : z.1 = w.1 := congrFun (congrFun h 0) 0
  have h2 : z.2 = w.2 := congrFun (congrFun h 1) 0
  exact Prod.ext h1 h2

/--
 The determinant of `rep σ z` is the norm form: `det (rep σ z) = z₁² − σ z₂²`. 

`normForm` [the norm form: a point times its conjugate, which can come out negative]
`Cell` [a point: two real numbers written side by side, a + b·x]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`ℝ` [the real numbers: every number on the number line]
-/
theorem det_rep (σ : ℝ) (z : Cell) : (rep σ z).det = normForm σ z := by
  rw [Matrix.det_fin_two]
  simp [rep, normForm, Apeiron.Trichotomy.AnyRing.ringNormForm]
  ring

/-! ### Step 2: the elliptic face (σ = −1) -/

/--
 The elliptic face: `rep (−1) z = z₁ • 1 + z₂ • J`, so `a + bi` is `a·1 + b·J`. 

`Cell` [a point: two real numbers written side by side, a + b·x]
`•` [stretch by, a number times a point]
`ℝ` [the real numbers: every number on the number line]
-/
theorem rep_neg_one (z : Cell) : rep (-1) z = z.1 • (1 : Matrix (Fin 2) (Fin 2) ℝ) + z.2 • J := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep, J]
  · simp [rep, J]
  · simp [rep, J]
  · simp [rep, J]

/--
 On the elliptic face the unit cells are rotations: `(rep (−1) z)ᵀ * rep (−1) z = normForm (−1) z • 1`. 

`normForm` [the norm form: a point times its conjugate, which can come out negative]
`Cell` [a point: two real numbers written side by side, a + b·x]
`•` [stretch by, a number times a point]
`ℝ` [the real numbers: every number on the number line]
-/
theorem rep_orthogonal (z : Cell) :
    (rep (-1) z)ᵀ * rep (-1) z = normForm (-1) z • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep, normForm, Apeiron.Trichotomy.AnyRing.ringNormForm, Matrix.mul_apply, Fin.sum_univ_two]
    ring
  · simp [rep, normForm, Apeiron.Trichotomy.AnyRing.ringNormForm, Matrix.mul_apply, Fin.sum_univ_two]
    ring
  · simp [rep, normForm, Apeiron.Trichotomy.AnyRing.ringNormForm, Matrix.mul_apply, Fin.sum_univ_two]
    ring
  · simp [rep, normForm, Apeiron.Trichotomy.AnyRing.ringNormForm, Matrix.mul_apply, Fin.sum_univ_two]
    ring

/-! ### Step 3: the hyperbolic and parabolic generators -/

/-- The hyperbolic generator `x = (0,1)` at `σ = 1` squares to `1`. -/
theorem rep_one_x : rep 1 (0, 1) * rep 1 (0, 1) = 1 := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]

/-- The parabolic generator `x = (0,1)` at `σ = 0` is nilpotent: its square is `0`. -/
theorem rep_zero_x : rep 0 (0, 1) * rep 0 (0, 1) = 0 := by
  refine ext_fin2 ?_ ?_ ?_ ?_
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]
  · simp [rep, Matrix.mul_apply, Fin.sum_univ_two]

/-! ### Step 4: realification is multiplicative -/

/--
Realification as a Kronecker sum, `A + iB ↦ A ⊗ I + B ⊗ J`.

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
`ℝ` [the real numbers: every number on the number line]
-/
theorem realify_eq {n : Type} [Fintype n] [DecidableEq n] (M : Matrix n n ℂ) :
    realify M = Matrix.kroneckerMap (· * ·) (M.map Complex.re) (1 : Matrix (Fin 2) (Fin 2) ℝ) +
      Matrix.kroneckerMap (· * ·) (M.map Complex.im) J := rfl

/--
 Realified sums: `(X + Y) ⊗ Z = X ⊗ Z + Y ⊗ Z` for the Kronecker product `⊗` (entrywise product of blocks). 

`ℝ` [the real numbers: every number on the number line]
-/
theorem add_kron_left {m n : Type} [Fintype m] [Fintype n] (X Y : Matrix m m ℝ) (Z : Matrix n n ℝ) :
    Matrix.kroneckerMap (· * ·) (X + Y) Z = Matrix.kroneckerMap (· * ·) X Z + Matrix.kroneckerMap (· * ·) Y Z := by
  ext i j
  simp [Matrix.kroneckerMap_apply, add_mul]

/--
 Negation commutes with the Kronecker product in the right factor. 

`ℝ` [the real numbers: every number on the number line]
-/
theorem kron_neg_right {m n : Type} [Fintype m] [Fintype n] (X : Matrix m m ℝ) (Y : Matrix n n ℝ) :
    Matrix.kroneckerMap (· * ·) X (-Y) = -Matrix.kroneckerMap (· * ·) X Y := by
  ext i j
  simp [Matrix.kroneckerMap_apply]

/--
 Negation commutes with the Kronecker product in the left factor. 

`ℝ` [the real numbers: every number on the number line]
-/
theorem neg_kron_left {m n : Type} (X : Matrix m m ℝ) (Y : Matrix n n ℝ) :
    Matrix.kroneckerMap (· * ·) (-X) Y = -Matrix.kroneckerMap (· * ·) X Y := by
  ext i j
  simp [Matrix.kroneckerMap_apply]

/--
 The Kronecker product is additive in its left factor for subtraction. 

`ℝ` [the real numbers: every number on the number line]
-/
theorem sub_kron_left {m n : Type} (X Y : Matrix m m ℝ) (Z : Matrix n n ℝ) :
    Matrix.kroneckerMap (· * ·) (X - Y) Z = Matrix.kroneckerMap (· * ·) X Z - Matrix.kroneckerMap (· * ·) Y Z := by
  ext i j
  simp [Matrix.kroneckerMap_apply, sub_mul]

/--
 Transposition distributes over the Kronecker product: `(X ⊗ Y)ᵀ = Xᵀ ⊗ Yᵀ`. 

`ℝ` [the real numbers: every number on the number line]
-/
theorem kron_transpose {m n : Type} (X : Matrix m m ℝ) (Y : Matrix n n ℝ) :
    (Matrix.kroneckerMap (· * ·) X Y)ᵀ = Matrix.kroneckerMap (· * ·) Xᵀ Yᵀ := by
  ext i j
  simp [Matrix.kroneckerMap_apply]

/--
 The real part of a complex matrix product, entrywise. 

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
-/
theorem re_mul {n : Type} [Fintype n] [DecidableEq n] (M N : Matrix n n ℂ) :
    (M * N).map Complex.re = M.map Complex.re * N.map Complex.re - M.map Complex.im * N.map Complex.im := by
  ext i j
  simp [Matrix.mul_apply, Complex.re_sum, Complex.mul_re, Finset.sum_sub_distrib]

/--
 The imaginary part of a complex matrix product, entrywise. 

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
-/
theorem im_mul {n : Type} [Fintype n] [DecidableEq n] (M N : Matrix n n ℂ) :
    (M * N).map Complex.im = M.map Complex.re * N.map Complex.im + M.map Complex.im * N.map Complex.re := by
  ext i j
  simp [Matrix.mul_apply, Complex.im_sum, Complex.mul_im, Finset.sum_add_distrib]

/--
Realification is multiplicative: `(A + iB)(C + iD) ↦ (AC − BD) ⊗ I + (AD + BC) ⊗ J`.

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
-/
theorem realify_mul {n : Type} [Fintype n] [DecidableEq n] (M N : Matrix n n ℂ) :
    realify (M * N) = realify M * realify N := by
  rw [realify_eq, realify_eq, realify_eq, re_mul, im_mul, add_mul, mul_add, mul_add,
    ← Matrix.mul_kronecker_mul, ← Matrix.mul_kronecker_mul, ← Matrix.mul_kronecker_mul,
    ← Matrix.mul_kronecker_mul]
  simp only [mul_one, one_mul, J_sq]
  rw [sub_kron_left, add_kron_left, kron_neg_right]
  abel

/--
Realification of the identity is the identity.

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
-/
theorem realify_one {n : Type} [Fintype n] [DecidableEq n] : realify (1 : Matrix n n ℂ) = 1 := by
  have him : (1 : Matrix n n ℂ).map Complex.im = 0 := by
    ext i j
    rw [Matrix.map_apply, Matrix.one_apply]
    split_ifs <;> simp
  rw [realify_eq, him, Matrix.zero_kronecker, add_zero, Matrix.map_one _ (by simp) (by simp),
    Matrix.one_kronecker_one]

/--
Unitary maps to orthogonal: `realify Mᴴ = (realify M)ᵀ`.

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
-/
theorem realify_conjTranspose {n : Type} [Fintype n] [DecidableEq n] (M : Matrix n n ℂ) :
    realify Mᴴ = (realify M)ᵀ := by
  have hre : (Mᴴ).map Complex.re = (M.map Complex.re)ᵀ := by
    ext i j
    simp [Matrix.conjTranspose_apply]
  have him : (Mᴴ).map Complex.im = -(M.map Complex.im)ᵀ := by
    ext i j
    simp [Matrix.conjTranspose_apply]
  rw [realify_eq, realify_eq, hre, him, Matrix.transpose_add, kron_transpose, kron_transpose,
    Matrix.transpose_one, J_transpose, kron_neg_right, neg_kron_left]

/--
The one-point realification is the elliptic cell: `(a + ib) ↦ a·1 + b·J`.

`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
`j` [the special direction that squares to give 1, not −1 the way i does]
`∀` [for every single one]
`ℝ` [the real numbers: every number on the number line]
-/
theorem realify_scalar (a b : ℝ) :
    ∀ i j : Fin 2,
      realify (Matrix.diagonal fun _ : Fin 1 => (⟨a, b⟩ : ℂ)) (0, i) (0, j) = rep (-1) (a, b) i j := by
  intro i j
  fin_cases i <;> fin_cases j <;> simp [realify_eq, rep, J, Matrix.kroneckerMap_apply]

/--
The conjunction of the L10 clauses.

`normForm` [the norm form: a point times its conjugate, which can come out negative]
`Cell` [a point: two real numbers written side by side, a + b·x]
`mul` [multiply two points together, the σ way]
`∧` [both are true]
`•` [stretch by, a number times a point]
`ℂ` [the complex numbers: pairs of reals where the second squares to −1]
`σ` [the parameter: what the adjoined direction squares to, below zero, zero, or above]
`j` [the special direction that squares to give 1, not −1 the way i does]
`∀` [for every single one]
`ℝ` [the real numbers: every number on the number line]
-/
theorem verdict_l10 :
    (∀ (σ : ℝ) (z w : Cell), rep σ (mul σ z w) = rep σ z * rep σ w) ∧
    (∀ σ : ℝ, Function.Injective (rep σ)) ∧
    (∀ σ : ℝ, rep σ Apeiron.Trichotomy.one = 1) ∧
    (∀ (σ : ℝ) (z : Cell), (rep σ z).det = normForm σ z) ∧
    (∀ z : Cell, rep (-1) z = z.1 • (1 : Matrix (Fin 2) (Fin 2) ℝ) + z.2 • J) ∧
    J * J = -1 ∧
    Jᵀ = -J ∧
    (∀ z : Cell, (rep (-1) z)ᵀ * rep (-1) z = normForm (-1) z • (1 : Matrix (Fin 2) (Fin 2) ℝ)) ∧
    rep 1 (0, 1) * rep 1 (0, 1) = 1 ∧
    rep 0 (0, 1) * rep 0 (0, 1) = 0 ∧
    (∀ (n : Type) [Fintype n] [DecidableEq n] (M N : Matrix n n ℂ),
      realify (M * N) = realify M * realify N) ∧
    (∀ (n : Type) [Fintype n] [DecidableEq n], realify (1 : Matrix n n ℂ) = 1) ∧
    (∀ (n : Type) [Fintype n] [DecidableEq n] (M : Matrix n n ℂ), realify Mᴴ = (realify M)ᵀ) ∧
    (∀ a b : ℝ, ∀ i j : Fin 2,
      realify (Matrix.diagonal fun _ : Fin 1 => (⟨a, b⟩ : ℂ)) (0, i) (0, j) = rep (-1) (a, b) i j) :=
  ⟨rep_mul, rep_injective, rep_one, det_rep, rep_neg_one, J_sq, J_transpose, rep_orthogonal,
    rep_one_x, rep_zero_x, fun _ _ _ M N => realify_mul M N, fun _ _ _ => realify_one,
    fun _ _ _ M => realify_conjTranspose M, fun a b => realify_scalar a b⟩

end EllipticRebit
end Quantum
