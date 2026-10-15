/-
# The tetrahedral stencil is the qubit SIC

`APEIRON-QARCH-L3`. The four Bloch vectors `nVec k = (1/√3) • tetraV k` of the regular
tetrahedron `Numeric.BranchLaplacian.tetraV` form a symmetric informationally complete POVM
(SIC-POVM) for one qubit. Each vertex has unit norm (`nVec_unit`) and pairwise inner product
`-1/3` (`nVec_dot_ne`), so the rank-one projectors `Πₖ = (1 + Σ·nₖ)/2` (`Pi`) satisfy the SIC
overlap `tr(Πⱼ Πₖ) = 1/3` for `j ≠ k` (`sic_overlap`) and `tr(Πₖ Πₖ) = 1` (`sic_diag`); at
dimension `d = 2` the SIC condition is `1/(d + 1) = 1/3`. The four `Πₖ/2` resolve the identity
(`sic_povm`), the tetrahedral family reproduces `(4/3)·u` as its second moment
(`sic_second_moment`), and `k = 4 = d²` vectors are the minimal complete set (`sic_count`). The
verdict `verdict_l3` conjoins the headlines.

Reference: J. M. Renes, R. Blume-Kohout, A. J. Scott, C. M. Caves, "Symmetric informationally
complete quantum measurements", *J. Math. Phys.* **45** (2004) 2171.
-/

import Mathlib
import Numeric.Quantum.Qubit
import Numeric.Quantum.BranchLaplacian

namespace Quantum
namespace SicStencil

/-- `(1/√3) · (1/√3) = 1/3`. -/
private theorem inv_sqrt_three_mul : (1 / Real.sqrt 3) * (1 / Real.sqrt 3) = (1 / 3 : ℝ) := by
  rw [← sq, div_pow, one_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 3)]

/-- The four unit Bloch vectors of the regular tetrahedron inscribed in the Bloch sphere:
`nVec k = (1/√3) • tetraV k`. -/
noncomputable def nVec (k : Fin 4) : Fin 3 → ℝ :=
  (1 / Real.sqrt 3) • Numeric.BranchLaplacian.tetraV k

/-- The rank-one projector `Πₖ = (1 + Σ·nₖ)/2` on the tetrahedral Bloch vector `nVec k`. -/
noncomputable def Pi (k : Fin 4) : Matrix (Fin 2) (Fin 2) ℂ :=
  Numeric.Qubit.projP (nVec k 0) (nVec k 1) (nVec k 2)

/-- Every tetrahedral Bloch vector has unit norm (`Real.sq_sqrt` on `3`). -/
theorem nVec_unit (k : Fin 4) : (nVec k 0) ^ 2 + (nVec k 1) ^ 2 + (nVec k 2) ^ 2 = 1 := by
  fin_cases k <;>
    simp [nVec, Numeric.BranchLaplacian.tetraV, Pi.smul_apply, smul_eq_mul] <;> norm_num

/-- Each `Πₖ` is a projector, from `Numeric.Qubit.projP_idem` at unit `nVec k`. -/
theorem Pi_idem (k : Fin 4) : Pi k * Pi k = Pi k := Numeric.Qubit.projP_idem (nVec_unit k)

/-- `Σ·n` is Hermitian for real `n`. -/
theorem sigmaN_conjTranspose (n1 n2 n3 : ℝ) :
    (Numeric.Qubit.sigmaN n1 n2 n3).conjTranspose = Numeric.Qubit.sigmaN n1 n2 n3 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Numeric.Qubit.sigmaN, Matrix.conjTranspose_apply, sub_eq_add_neg]

/-- `(1 + Σ·n)/2` is Hermitian for real `n`: the conjugate transpose is linear and `Σ·n` is
Hermitian. -/
theorem projP_conjTranspose (n1 n2 n3 : ℝ) :
    (Numeric.Qubit.projP n1 n2 n3).conjTranspose = Numeric.Qubit.projP n1 n2 n3 := by
  unfold Numeric.Qubit.projP
  rw [Matrix.conjTranspose_smul, Matrix.conjTranspose_add, Matrix.conjTranspose_one,
    sigmaN_conjTranspose]
  norm_num

/-- Each `Πₖ` is Hermitian. -/
theorem Pi_herm (k : Fin 4) : (Pi k).conjTranspose = Pi k := projP_conjTranspose _ _ _

/-- `tr(Π) = 1`: each projector is rank one. -/
theorem trace_projP (n1 n2 n3 : ℝ) : Matrix.trace (Numeric.Qubit.projP n1 n2 n3) = 1 := by
  simp [Numeric.Qubit.projP, Numeric.Qubit.sigmaN, Matrix.trace_fin_two]

/-- `tr(Π_a Π_b) = (1 + a·b)/2`; the identity underlying the SIC overlaps. -/
theorem trace_projP_mul (a b : Fin 3 → ℝ) :
    Matrix.trace (Numeric.Qubit.projP (a 0) (a 1) (a 2) *
        Numeric.Qubit.projP (b 0) (b 1) (b 2)) = ((1 + a ⬝ᵥ b) / 2 : ℝ) := by
  simp only [Numeric.Qubit.projP, Numeric.Qubit.sigmaN]
  simp only [Matrix.trace_fin_two, Matrix.mul_apply, Fin.sum_univ_two, Matrix.add_apply,
    Matrix.smul_apply, Matrix.one_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.cons_val_fin_one, Matrix.of_apply]
  simp
  ring_nf
  rw [Complex.I_sq, dotProduct, Fin.sum_univ_three]
  push_cast
  ring

/-- Every tetrahedron vertex has squared length `3`. -/
private theorem tetraV_dot_self (k : Fin 4) :
    Numeric.BranchLaplacian.tetraV k ⬝ᵥ Numeric.BranchLaplacian.tetraV k = 3 := by
  fin_cases k <;>
    simp [Numeric.BranchLaplacian.tetraV, dotProduct, Fin.sum_univ_three] <;> norm_num

/-- Distinct tetrahedron vertices have inner product `-1`. -/
private theorem tetraV_dot_ne {j k : Fin 4} (h : j ≠ k) :
    Numeric.BranchLaplacian.tetraV j ⬝ᵥ Numeric.BranchLaplacian.tetraV k = -1 := by
  fin_cases j <;> fin_cases k <;>
    simp [Numeric.BranchLaplacian.tetraV, dotProduct, Fin.sum_univ_three] at h ⊢

/-- The scaled inner product factors through the tetrahedron. -/
private theorem nVec_dot_eq (j k : Fin 4) :
    nVec j ⬝ᵥ nVec k = (1 / Real.sqrt 3) * (1 / Real.sqrt 3) *
      (Numeric.BranchLaplacian.tetraV j ⬝ᵥ Numeric.BranchLaplacian.tetraV k) := by
  simp only [nVec, Pi.smul_apply, smul_eq_mul, dotProduct, Fin.sum_univ_three]
  ring

/-- `nVec k` is a unit vector. -/
theorem nVec_dot_self (k : Fin 4) : nVec k ⬝ᵥ nVec k = 1 := by
  rw [nVec_dot_eq, tetraV_dot_self, inv_sqrt_three_mul]
  norm_num

/-- Distinct tetrahedral Bloch vectors have inner product `-1/3`. -/
theorem nVec_dot_ne {j k : Fin 4} (h : j ≠ k) : nVec j ⬝ᵥ nVec k = -(1 / 3) := by
  rw [nVec_dot_eq, tetraV_dot_ne h, inv_sqrt_three_mul]
  norm_num

/-- The SIC overlap: `tr(Πⱼ Πₖ) = 1/3` for distinct tetrahedral projectors. -/
theorem sic_overlap {j k : Fin 4} (h : j ≠ k) :
    Matrix.trace (Pi j * Pi k) = (1 / 3 : ℂ) := by
  change Matrix.trace (Numeric.Qubit.projP (nVec j 0) (nVec j 1) (nVec j 2) *
      Numeric.Qubit.projP (nVec k 0) (nVec k 1) (nVec k 2)) = (1 / 3 : ℂ)
  rw [trace_projP_mul, nVec_dot_ne h]
  norm_num

/-- The SIC normalisation: `tr(Πₖ Πₖ) = 1`. -/
theorem sic_diag (k : Fin 4) : Matrix.trace (Pi k * Pi k) = 1 := by
  change Matrix.trace (Numeric.Qubit.projP (nVec k 0) (nVec k 1) (nVec k 2) *
      Numeric.Qubit.projP (nVec k 0) (nVec k 1) (nVec k 2)) = 1
  rw [trace_projP_mul, nVec_dot_self]
  norm_num

/-- The tetrahedral Bloch vectors sum to zero (from `Σ v = 0`). -/
theorem nVec_sum_zero : (∑ k : Fin 4, nVec k) = 0 := by
  have h : (∑ k : Fin 4, Numeric.BranchLaplacian.tetraV k) = 0 := by
    simpa using Numeric.BranchLaplacian.tetra_family_attains_four.1
  unfold nVec
  rw [← Finset.smul_sum, h, smul_zero]

/-- `Σ·n` is linear in `n`, so it commutes with a finite sum. -/
theorem sigmaN_sum (f : Fin 4 → Fin 3 → ℝ) :
    (∑ k : Fin 4, Numeric.Qubit.sigmaN (f k 0) (f k 1) (f k 2))
      = Numeric.Qubit.sigmaN (∑ k, f k 0) (∑ k, f k 1) (∑ k, f k 2) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Numeric.Qubit.sigmaN, Matrix.sum_apply, Finset.sum_mul, Finset.sum_add_distrib,
      Finset.sum_sub_distrib]

/-- The SIC POVM: the four `Πₖ/2` resolve the identity. -/
theorem sic_povm : (∑ k : Fin 4, (1 / 2 : ℂ) • Pi k) = 1 := by
  have hsum_sigma :
      (∑ k : Fin 4, Numeric.Qubit.sigmaN (nVec k 0) (nVec k 1) (nVec k 2)) = 0 := by
    rw [sigmaN_sum]
    have h0 : (∑ k : Fin 4, nVec k 0) = 0 := by simpa using congrFun nVec_sum_zero 0
    have h1 : (∑ k : Fin 4, nVec k 1) = 0 := by simpa using congrFun nVec_sum_zero 1
    have h2 : (∑ k : Fin 4, nVec k 2) = 0 := by simpa using congrFun nVec_sum_zero 2
    rw [h0, h1, h2]
    ext i j
    fin_cases i <;> fin_cases j <;> simp [Numeric.Qubit.sigmaN]
  have hconst : (∑ _ : Fin 4, (1 / 4 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ)) = 1 := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp
  calc (∑ k : Fin 4, (1 / 2 : ℂ) • Pi k)
      = ∑ k : Fin 4, ((1 / 4 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ)
          + (1 / 4 : ℂ) • Numeric.Qubit.sigmaN (nVec k 0) (nVec k 1) (nVec k 2)) :=
        Finset.sum_congr rfl (fun _ _ => by
          simp only [Pi, Numeric.Qubit.projP, smul_add, smul_smul]
          norm_num)
    _ = (∑ k : Fin 4, (1 / 4 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ))
          + ∑ k : Fin 4, (1 / 4 : ℂ) • Numeric.Qubit.sigmaN (nVec k 0) (nVec k 1) (nVec k 2) :=
        Finset.sum_add_distrib
    _ = (∑ k : Fin 4, (1 / 4 : ℂ) • (1 : Matrix (Fin 2) (Fin 2) ℂ))
          + (1 / 4 : ℂ) •
              (∑ k : Fin 4, Numeric.Qubit.sigmaN (nVec k 0) (nVec k 1) (nVec k 2)) := by
        rw [Finset.smul_sum]
    _ = 1 := by rw [hconst, hsum_sigma, smul_zero, add_zero]

/-- The tetrahedral second moment: `Σₖ (nₖ·u) nₖ = (4/3)·u`. -/
theorem sic_second_moment (u : Fin 3 → ℝ) :
    (∑ k : Fin 4, (nVec k ⬝ᵥ u) • nVec k) = (4 / 3 : ℝ) • u := by
  have htf : (∑ k : Fin 4, (Numeric.BranchLaplacian.tetraV k ⬝ᵥ u) •
      Numeric.BranchLaplacian.tetraV k) = (4 : ℝ) • u := by
    simpa only [Pi.one_apply, one_mul] using Numeric.BranchLaplacian.tetra_family_attains_four.2 u
  have hterm : ∀ k : Fin 4, (nVec k ⬝ᵥ u) • nVec k
      = (1 / 3 : ℝ) • ((Numeric.BranchLaplacian.tetraV k ⬝ᵥ u) •
          Numeric.BranchLaplacian.tetraV k) := by
    intro k
    have hdot : nVec k ⬝ᵥ u = (1 / Real.sqrt 3) *
        (Numeric.BranchLaplacian.tetraV k ⬝ᵥ u) := by
      simp only [nVec, smul_dotProduct, smul_eq_mul]
    have hsc : ((1 / Real.sqrt 3) * (Numeric.BranchLaplacian.tetraV k ⬝ᵥ u)) *
        (1 / Real.sqrt 3) = (1 / 3 : ℝ) *
        (Numeric.BranchLaplacian.tetraV k ⬝ᵥ u) := by
      rw [mul_comm ((1 / Real.sqrt 3) * (Numeric.BranchLaplacian.tetraV k ⬝ᵥ u))
        (1 / Real.sqrt 3), ← mul_assoc, inv_sqrt_three_mul]
    rw [hdot]
    simp only [nVec, smul_smul, hsc]
  calc (∑ k : Fin 4, (nVec k ⬝ᵥ u) • nVec k)
      = ∑ k : Fin 4, (1 / 3 : ℝ) • ((Numeric.BranchLaplacian.tetraV k ⬝ᵥ u) •
          Numeric.BranchLaplacian.tetraV k) := Finset.sum_congr rfl (fun k _ => hterm k)
    _ = (1 / 3 : ℝ) • ∑ k : Fin 4, (Numeric.BranchLaplacian.tetraV k ⬝ᵥ u) •
          Numeric.BranchLaplacian.tetraV k := by rw [← Finset.smul_sum]
    _ = (1 / 3 : ℝ) • ((4 : ℝ) • u) := by rw [htf]
    _ = (4 / 3 : ℝ) • u := by rw [smul_smul]; norm_num

/-- The tetrahedral stencil is minimal: `k = 4 = d²` vectors at `d = 2`. -/
theorem sic_count : Fintype.card (Fin 4) = 2 ^ 2 := by
  rw [Fintype.card_fin]
  norm_num

/-- The unit verdict: the tetrahedral stencil is the qubit SIC — unit Bloch vectors with overlap
`-1/3`, projectors with pairwise overlap `1/3`, an identity resolution, the second moment `4/3`,
and `d²` vectors. -/
theorem verdict_l3 :
    (∀ k : Fin 4, (nVec k 0) ^ 2 + (nVec k 1) ^ 2 + (nVec k 2) ^ 2 = 1) ∧
      (∀ j k : Fin 4, j ≠ k → Matrix.trace (Pi j * Pi k) = (1 / 3 : ℂ)) ∧
      (∑ k : Fin 4, (1 / 2 : ℂ) • Pi k) = 1 ∧
      (∀ u : Fin 3 → ℝ, (∑ k : Fin 4, (nVec k ⬝ᵥ u) • nVec k) = (4 / 3 : ℝ) • u) ∧
      Fintype.card (Fin 4) = 2 ^ 2 :=
  ⟨nVec_unit, fun _ _ h => sic_overlap h, sic_povm, sic_second_moment, sic_count⟩

end SicStencil
end Quantum
