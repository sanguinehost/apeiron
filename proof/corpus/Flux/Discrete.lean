/-
Discrete flux pairing and the lattice divergence theorem (`docs/epics/magnetic-flux.md`).

This is the finite-lattice content of `∇·B = 0`: the divergence of a face-valued field summed over
the cells of a block equals the flux through the block's boundary, so a divergence-free field has
zero flux through any closed boundary.

The value type is an additive commutative group. No carrier algebra is assumed; the
`Apeiron.Trichotomy.AnyRing.RingCell` instance is an instance, not a requirement. The local
`Flux` library stages this material inside the bridge package until the corpus migration, following
the `Policy` precedent (`proof/corpus/lakefile.lean`).
-/
import Mathlib

namespace Flux.Discrete

variable {M : Type*} [AddCommGroup M]

/-- The one-direction difference of consecutive face values, `B (i + 1) - B i`. In the continuum
this is the differential that `Derivation` abstracts; here it is the finite difference. -/
def diff (B : ℕ → M) (i : ℕ) : M := B (i + 1) - B i

/-- One-dimensional discrete divergence theorem: the sum of the differences over `n` cells equals
the flux through the two boundary faces. -/
theorem sum_range_diff (B : ℕ → M) (n : ℕ) :
    (∑ i ∈ Finset.range n, diff B i) = B n - B 0 := by
  induction n with
  | zero => simp [diff]
  | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      simp only [diff]
      abel

/-- Outward divergence of cell `(i, j)`: the x-difference plus the y-difference. This is the cell's
net out-flux. -/
def divergence (Bx By : ℕ → ℕ → M) (i j : ℕ) : M :=
  (Bx (i + 1) j - Bx i j) + (By i (j + 1) - By i j)

/-- Flux through the four sides of the `m × n` block, outward oriented: the right and left faces
summed over `j`, then the top and bottom faces summed over `i`. -/
def boundaryFlux (Bx By : ℕ → ℕ → M) (m n : ℕ) : M :=
  (∑ j ∈ Finset.range n, (Bx m j - Bx 0 j)) + (∑ i ∈ Finset.range m, (By i n - By i 0))

/-- The x-part of a divergence sum telescopes in the first index. -/
theorem sum_divergence_x (Bx : ℕ → ℕ → M) (m n : ℕ) :
    (∑ i ∈ Finset.range m, ∑ j ∈ Finset.range n, (Bx (i + 1) j - Bx i j))
      = ∑ j ∈ Finset.range n, (Bx m j - Bx 0 j) := by
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl (fun j _ => sum_range_diff (fun i => Bx i j) m)

/-- The y-part of a divergence sum telescopes in the second index. -/
theorem sum_divergence_y (By : ℕ → ℕ → M) (m n : ℕ) :
    (∑ i ∈ Finset.range m, ∑ j ∈ Finset.range n, (By i (j + 1) - By i j))
      = ∑ i ∈ Finset.range m, (By i n - By i 0) :=
  Finset.sum_congr rfl (fun i _ => sum_range_diff (fun j => By i j) n)

/-- Two-dimensional discrete divergence theorem: the divergence summed over the cells of an `m × n`
block equals the flux through the block's boundary. -/
theorem grid_divergence_theorem (Bx By : ℕ → ℕ → M) (m n : ℕ) :
    (∑ i ∈ Finset.range m, ∑ j ∈ Finset.range n, divergence Bx By i j)
      = boundaryFlux Bx By m n := by
  simp only [divergence, boundaryFlux]
  rw [Finset.sum_congr rfl (fun i _ => Finset.sum_add_distrib)]
  rw [Finset.sum_add_distrib, sum_divergence_x Bx m n, sum_divergence_y By m n]

/-- A divergence-free field has zero flux through the boundary of any block. This is the discrete
content of `∇·B = 0`, and the reason no isolated pole exists. -/
theorem flux_zero_of_divergence_free (Bx By : ℕ → ℕ → M) (m n : ℕ)
    (h : ∀ i j, divergence Bx By i j = 0) : boundaryFlux Bx By m n = 0 := by
  rw [← grid_divergence_theorem]
  exact Finset.sum_eq_zero (fun i _ => Finset.sum_eq_zero (fun j _ => h i j))

/-! ## Non-vacuity

A divergence-free field whose boundary terms are individually nonzero and cancel pairwise. This is
the joint witness the Specification requires: the hypothesis `∀ i j, divergence .. = 0` is inhabited
at the same types as the conclusion, so the corollary is not vacuous. -/

private def rotX (i _ : ℕ) : ℤ := (i : ℤ)

private def rotY (_ j : ℕ) : ℤ := -(j : ℤ)

example : ∀ i j, divergence rotX rotY i j = 0 := by
  intro i j
  simp only [divergence, rotX, rotY]
  push_cast
  ring

/-- The two open sides carry nonzero flux, `4` and `-4`, and cancel. -/
example : (∑ j ∈ Finset.range 2, (rotX 2 j - rotX 0 j)) = 4 := by
  norm_num [rotX, Finset.sum_range_succ]

example : (∑ i ∈ Finset.range 2, (rotY i 2 - rotY i 0)) = -4 := by
  norm_num [rotY, Finset.sum_range_succ]

example : boundaryFlux rotX rotY 2 2 = 0 := by
  simp only [boundaryFlux, rotX, rotY]
  norm_num [Finset.sum_range_succ]

end Flux.Discrete
