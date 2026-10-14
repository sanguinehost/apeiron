/-
# The normalized Hadamard rotation as a block `Stage` (`APEIRON-TERNARY-W4-T3`)

`Nous.Model.BlockComposition` lands the dimension-changing `Stage E F` contract with its explicit
admissible domain; `Numeric.Sylvester` lands the integer Sylvester Hadamard matrix `sylvesterH k`
with `sylvester_transpose` and `sylvester_mul_self` (`H·H = 2^k·1`). This module packages the
*normalized* real rotation `x ↦ (1/√(2^k)) • H_k (s ⊙ x)`, with a sign vector `s_i ∈ {−1, +1}`, as
such a `Stage`.

## What is proved

* `hadamardRef`: the rotation, a real matrix-vector product by `sylvesterH k` cast to `ℝ` and the
  scalar `1/√(2^k)`.
* `hadamardRef_lipschitz_sup`: in `Stage`'s `ℓ∞` sup norm the rotation has expansion **`√(2^k)`**,
  not `1` — each row of `H_k` has `2^k` entries of absolute value `1`, so `|row·v| ≤ 2^k·‖v‖∞`, and
  the factor `1/√(2^k)` leaves `√(2^k)`. The `ℓ2` isometry is a *separate* statement and does not
  make the sup-norm expansion `1`.
* `hadamardRef_isometry_l2`: `∑ (hadamardRef x)_i² = ∑ x_i²`, from `H_k·H_k = 2^k·1` (`H_k`
  symmetric and the sign vector unimodular).
* `hadamardRef_involutive`: for the trivial sign vector `s = 1`, the rotation is an involution.
* `rotationStage`: the `Stage` on the capped domain `{x | ∀ i, |x_i| ≤ B}`, execution supplied by a
  caller hypothesis bounded by `delta`, and `L = √(2^k)`.

## Trust boundary

The algebra, the norm bounds and the involution are kernel-checked over `ℝ`. `rotationStage` carries
the executed-kernel link as an **explicit hypothesis** (`hdefect`); the Rust/Aeneas refinement
`fwht_fixed_refines` lands separately and is *not* imported here, so no claim is made that the
executed map *is* a particular extracted body. No claim is made about rustc/LLVM codegen, the kernel
ABI, GPU/SPIR-V or the `unsafe` container.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom`, `native_decide` or
`decide` over an extracted body appears.
-/
import Nous.Model.BlockComposition
import Numeric.Hadamard.Sylvester

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open apeiron_word
open scoped Matrix

noncomputable section

/-! ## Entries of the Sylvester matrix are `±1` -/

/-- Every entry of `sylvesterH k` has absolute value `1`, by the block recurrence
`H_{k+1} = [[H_k, H_k], [H_k, −H_k]]`. This is the fact that makes the `ℓ∞` row bound `2^k`. -/
theorem sylvesterH_abs_entry (k : ℕ) (i j : Fin (2 ^ k)) : |sylvesterH k i j| = 1 := by
  induction k with
  | zero =>
      have hij : i = j := by
        apply Fin.ext
        have h1 := i.isLt
        have h2 := j.isLt
        simp only [pow_zero] at h1 h2
        omega
      subst hij
      rw [sylvester_zero]
      simp
  | succ k ih =>
      rw [sylvester_succ]
      have hi : i = sylvesterBlockEquiv k ((sylvesterBlockEquiv k).symm i) :=
        (Equiv.apply_symm_apply _ i).symm
      have hj : j = sylvesterBlockEquiv k ((sylvesterBlockEquiv k).symm j) :=
        (Equiv.apply_symm_apply _ j).symm
      rw [hi, hj, reindex_apply_equiv]
      rcases hsi : (sylvesterBlockEquiv k).symm i with a | a <;>
        rcases hsj : (sylvesterBlockEquiv k).symm j with b | b <;>
        simp only [Matrix.fromBlocks_apply₁₁, Matrix.fromBlocks_apply₁₂,
          Matrix.fromBlocks_apply₂₁, Matrix.fromBlocks_apply₂₂, Matrix.neg_apply]
      · exact ih a b
      · exact ih a b
      · exact ih a b
      · rw [abs_neg]; exact ih a b

/-! ## The real Hadamard matrix and the rotation -/

/-- `sylvesterH k` with its integer entries read into `ℝ`. -/
def hadamardMat (k : ℕ) : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℝ :=
  (sylvesterH k).map (Int.castRingHom ℝ)

theorem hadamardMat_apply (k : ℕ) (i j : Fin (2 ^ k)) :
    hadamardMat k i j = ((sylvesterH k i j : ℤ) : ℝ) := rfl

/-- The real Hadamard matrix is symmetric. -/
theorem hadamardMat_transpose (k : ℕ) : (hadamardMat k)ᵀ = hadamardMat k := by
  unfold hadamardMat
  rw [← Matrix.transpose_map, sylvester_transpose]

/-- `H_k · H_k = 2^k · 1` over `ℝ`, transported from `sylvester_mul_self`. -/
theorem hadamardMat_mul_self (k : ℕ) :
    hadamardMat k * hadamardMat k = ((2 : ℝ) ^ k) • (1 : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℝ) := by
  have hcast : (Int.castRingHom ℝ) (2 ^ k) = (2 : ℝ) ^ k := by
    rw [map_pow]
    simp
  unfold hadamardMat
  rw [← Matrix.map_mul (f := (Int.castRingHom ℝ)), sylvester_mul_self k,
    Matrix.map_smulₛₗ (Int.castRingHom ℝ) (Int.castRingHom ℝ) (2 ^ k)
      (by intro a; rw [Int.zsmul_eq_mul, map_mul, smul_eq_mul]),
    Matrix.map_one (Int.castRingHom ℝ) (by simp) (by simp), hcast]

/-- Every entry of the real Hadamard matrix has absolute value `1`. -/
theorem hadamardMat_entry_abs (k : ℕ) (i j : Fin (2 ^ k)) : |hadamardMat k i j| = 1 := by
  rw [hadamardMat_apply, ← Int.cast_abs, sylvesterH_abs_entry k i j, Int.cast_one]

/-- The sign-vector hypothesis `s_i ∈ {−1, +1}`. -/
def IsSign {n : ℕ} (s : Fin n → ℝ) : Prop := ∀ i, s i = 1 ∨ s i = -1

/-- The normalized Hadamard rotation `x ↦ (1/√(2^k)) • H_k (s ⊙ x)`, with sign vector `s`. -/
def hadamardRef (k : ℕ) (s : Fin (2 ^ k) → ℝ) (x : Fin (2 ^ k) → ℝ) : Fin (2 ^ k) → ℝ :=
  (1 / Real.sqrt ((2 : ℝ) ^ k)) • ((hadamardMat k) *ᵥ (s * x))

/-! ## Arithmetic of the rotation coefficient -/

/-- `(1/√n)² · n = 1` for `n = 2^k`. -/
theorem rotationCoef_sq_mul (k : ℕ) :
    (1 / Real.sqrt ((2 : ℝ) ^ k)) ^ 2 * (2 : ℝ) ^ k = 1 := by
  have hn : 0 < (2 : ℝ) ^ k := by positivity
  rw [div_pow, one_pow, Real.sq_sqrt (le_of_lt hn)]
  exact div_mul_cancel₀ 1 (ne_of_gt hn)

/-- `(1/√n) · n = √n` for `n = 2^k`. -/
theorem rotationCoef_mul (k : ℕ) :
    (1 / Real.sqrt ((2 : ℝ) ^ k)) * (2 : ℝ) ^ k = Real.sqrt ((2 : ℝ) ^ k) := by
  have hn : 0 < (2 : ℝ) ^ k := by positivity
  have hne : Real.sqrt ((2 : ℝ) ^ k) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hn)
  have hsq : Real.sqrt ((2 : ℝ) ^ k) * Real.sqrt ((2 : ℝ) ^ k) = (2 : ℝ) ^ k :=
    Real.mul_self_sqrt (le_of_lt hn)
  rw [one_div, inv_mul_eq_div, div_eq_iff hne]
  exact hsq.symm

/-! ## Linearity and the `ℓ∞` Lipschitz bound -/

/-- The rotation difference is the rotation of the difference (linearity of the rotation). -/
theorem hadamardRef_sub (k : ℕ) (s : Fin (2 ^ k) → ℝ) (x y : Fin (2 ^ k) → ℝ) :
    hadamardRef k s x - hadamardRef k s y = hadamardRef k s (x - y) := by
  have harg : (s * x) - (s * y) = s * (x - y) := by
    funext i
    simp only [Pi.sub_apply, Pi.mul_apply]
    ring
  unfold hadamardRef
  rw [← smul_sub, ← Matrix.mulVec_sub, harg]

/-- **The `ℓ∞` expansion of the normalized rotation is `√(2^k)`.** Every row of `H_k` has `2^k`
entries of absolute value `1`, so `|row·v| ≤ 2^k·‖v‖∞`, and the factor `1/√(2^k)` leaves `√(2^k)`.
The `ℓ2` isometry below does not make this constant `1`. -/
theorem hadamardRef_lipschitz_sup (k : ℕ) (s : Fin (2 ^ k) → ℝ) (hs : IsSign s)
    (x y : Fin (2 ^ k) → ℝ) :
    ‖hadamardRef k s x - hadamardRef k s y‖ ≤ Real.sqrt ((2 : ℝ) ^ k) * ‖x - y‖ := by
  rw [hadamardRef_sub]
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg (Real.sqrt_nonneg _) (norm_nonneg _))).2
  intro i
  have hcoord : hadamardRef k s (x - y) i
      = (1 / Real.sqrt ((2 : ℝ) ^ k))
          * ∑ j, hadamardMat k i j * (s j * (x j - y j)) := by
    unfold hadamardRef Matrix.mulVec dotProduct
    simp only [Pi.smul_apply, smul_eq_mul, Pi.mul_apply, Pi.sub_apply]
  rw [Real.norm_eq_abs, hcoord]
  have hc : 0 ≤ 1 / Real.sqrt ((2 : ℝ) ^ k) :=
    div_nonneg zero_le_one (Real.sqrt_nonneg _)
  have hv : ∀ j : Fin (2 ^ k), |x j - y j| ≤ ‖x - y‖ := by
    intro j
    have h := norm_le_pi_norm (x - y) j
    rwa [Real.norm_eq_abs, Pi.sub_apply] at h
  have hsum : |∑ j, hadamardMat k i j * (s j * (x j - y j))|
      ≤ (2 : ℝ) ^ k * ‖x - y‖ := by
    calc |∑ j, hadamardMat k i j * (s j * (x j - y j))|
        ≤ ∑ j, |hadamardMat k i j * (s j * (x j - y j))| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, |x j - y j| := by
          apply Finset.sum_congr rfl
          intro j _
          rw [abs_mul, abs_mul, hadamardMat_entry_abs k i j, one_mul]
          rcases hs j with h | h <;> rw [h] <;> simp
      _ ≤ ∑ _j : Fin (2 ^ k), ‖x - y‖ := Finset.sum_le_sum (fun j _ => hv j)
      _ = (2 : ℝ) ^ k * ‖x - y‖ := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          norm_num
  calc |(1 / Real.sqrt ((2 : ℝ) ^ k)) * ∑ j, hadamardMat k i j * (s j * (x j - y j))|
      = (1 / Real.sqrt ((2 : ℝ) ^ k))
          * |∑ j, hadamardMat k i j * (s j * (x j - y j))| := by
        rw [abs_mul, abs_of_nonneg hc]
    _ ≤ (1 / Real.sqrt ((2 : ℝ) ^ k)) * ((2 : ℝ) ^ k * ‖x - y‖) :=
        mul_le_mul_of_nonneg_left hsum hc
    _ = Real.sqrt ((2 : ℝ) ^ k) * ‖x - y‖ := by
        rw [← mul_assoc, rotationCoef_mul]

/-! ## The `ℓ2` isometry -/

/-- The `ℓ2` self-pairing of a vector is its sum of squares. -/
theorem dotProduct_self_eq_sum_sq {n : ℕ} (v : Fin n → ℝ) :
    v ⬝ᵥ v = ∑ i, v i ^ 2 := by
  rw [dotProduct]
  apply Finset.sum_congr rfl
  intro i _
  ring
/-- Core isometry step: for a symmetric real matrix with `A·A = m·1`, the `ℓ2` self-pairing of
`A u` is `m` times that of `u`. Proved by moving `Aᵀ` across the bilinear form and using `Aᵀ = A`. -/
theorem mulVec_dotProduct_self {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) (m : ℝ)
    (hA : Aᵀ = A) (hmul : A * A = m • (1 : Matrix (Fin n) (Fin n) ℝ)) (u : Fin n → ℝ) :
    (A *ᵥ u) ⬝ᵥ (A *ᵥ u) = m * (u ⬝ᵥ u) := by
  calc (A *ᵥ u) ⬝ᵥ (A *ᵥ u)
      = (A *ᵥ u) ᵥ* A ⬝ᵥ u := by rw [Matrix.dotProduct_mulVec]
    _ = (Aᵀ *ᵥ (A *ᵥ u)) ⬝ᵥ u := by rw [← Matrix.mulVec_transpose]
    _ = ((Aᵀ * A) *ᵥ u) ⬝ᵥ u := by rw [← Matrix.mulVec_mulVec]
    _ = u ⬝ᵥ ((Aᵀ * A) *ᵥ u) := by rw [dotProduct_comm]
    _ = u ⬝ᵥ ((A * A) *ᵥ u) := by rw [hA]
    _ = u ⬝ᵥ ((m • (1 : Matrix (Fin n) (Fin n) ℝ)) *ᵥ u) := by rw [hmul]
    _ = u ⬝ᵥ (m • u) := by rw [Matrix.smul_mulVec, Matrix.one_mulVec]
    _ = m * (u ⬝ᵥ u) := by rw [dotProduct_smul, smul_eq_mul]

/-- A sign-vector change of variable is `ℓ2`-neutral: `s_i ∈ {−1, +1}` gives `(s_i x_i)² = x_i²`. -/
theorem sum_sq_mul_sign {n : ℕ} (s x : Fin n → ℝ) (hs : IsSign s) :
    ∑ i, ((s * x) i) ^ 2 = ∑ i, (x i) ^ 2 := by
  apply Finset.sum_congr rfl
  intro i _
  rw [Pi.mul_apply]
  rcases hs i with h | h <;> rw [h] <;> ring

/-- **The normalized rotation is an `ℓ2` isometry**: `∑ (hadamardRef x)_i² = ∑ x_i²`. -/
theorem hadamardRef_isometry_l2 (k : ℕ) (s : Fin (2 ^ k) → ℝ) (hs : IsSign s)
    (x : Fin (2 ^ k) → ℝ) :
    ∑ i, (hadamardRef k s x i) ^ 2 = ∑ i, (x i) ^ 2 := by
  have hsquare : ∑ i, (hadamardRef k s x i) ^ 2
      = (1 / Real.sqrt ((2 : ℝ) ^ k)) ^ 2
          * ∑ i, (((hadamardMat k) *ᵥ (s * x)) i) ^ 2 := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    simp only [hadamardRef, Pi.smul_apply, smul_eq_mul]
    ring
  have hcore : ∑ i, (((hadamardMat k) *ᵥ (s * x)) i) ^ 2
      = ((2 : ℝ) ^ k) * ∑ i, ((s * x) i) ^ 2 := by
    rw [← dotProduct_self_eq_sum_sq,
      mulVec_dotProduct_self (hadamardMat k) ((2 : ℝ) ^ k) (hadamardMat_transpose k)
        (hadamardMat_mul_self k) (s * x),
      dotProduct_self_eq_sum_sq]
  calc ∑ i, (hadamardRef k s x i) ^ 2
      = (1 / Real.sqrt ((2 : ℝ) ^ k)) ^ 2
          * ∑ i, (((hadamardMat k) *ᵥ (s * x)) i) ^ 2 := hsquare
    _ = (1 / Real.sqrt ((2 : ℝ) ^ k)) ^ 2
          * (((2 : ℝ) ^ k) * ∑ i, ((s * x) i) ^ 2) := by rw [hcore]
    _ = ((1 / Real.sqrt ((2 : ℝ) ^ k)) ^ 2 * (2 : ℝ) ^ k)
          * ∑ i, ((s * x) i) ^ 2 := by ring
    _ = ∑ i, ((s * x) i) ^ 2 := by rw [rotationCoef_sq_mul, one_mul]
    _ = ∑ i, (x i) ^ 2 := sum_sq_mul_sign s x hs

/-! ## Involution at the trivial sign vector -/

/-- The rotation at the trivial sign vector `s = 1` is the plain normalized transform. -/
theorem hadamardRef_one (k : ℕ) (x : Fin (2 ^ k) → ℝ) :
    hadamardRef k (fun _ => (1 : ℝ)) x
      = (1 / Real.sqrt ((2 : ℝ) ^ k)) • ((hadamardMat k) *ᵥ x) := by
  unfold hadamardRef
  rw [show (fun _ : Fin (2 ^ k) => (1 : ℝ)) = (1 : Fin (2 ^ k) → ℝ) from rfl, one_mul]

/-- **Involution.** At the trivial sign vector `s = 1` the normalized rotation composed with itself
is the identity, from `H_k·H_k = 2^k·1` and `(1/√(2^k))²·2^k = 1`. -/
theorem hadamardRef_involutive (k : ℕ) (x : Fin (2 ^ k) → ℝ) :
    hadamardRef k (fun _ => (1 : ℝ)) (hadamardRef k (fun _ => (1 : ℝ)) x) = x := by
  have hself : hadamardRef k (fun _ => (1 : ℝ)) x
      = (1 / Real.sqrt ((2 : ℝ) ^ k)) • ((hadamardMat k) *ᵥ x) := hadamardRef_one k x
  calc hadamardRef k (fun _ => (1 : ℝ)) (hadamardRef k (fun _ => (1 : ℝ)) x)
      = hadamardRef k (fun _ => (1 : ℝ))
          ((1 / Real.sqrt ((2 : ℝ) ^ k)) • ((hadamardMat k) *ᵥ x)) := by rw [hself]
    _ = (1 / Real.sqrt ((2 : ℝ) ^ k))
          • ((hadamardMat k)
            *ᵥ ((1 / Real.sqrt ((2 : ℝ) ^ k)) • ((hadamardMat k) *ᵥ x))) :=
        hadamardRef_one k _
    _ = (1 / Real.sqrt ((2 : ℝ) ^ k))
          • ((1 / Real.sqrt ((2 : ℝ) ^ k))
            • ((hadamardMat k) *ᵥ ((hadamardMat k) *ᵥ x))) := by
        rw [Matrix.mulVec_smul]
    _ = ((1 / Real.sqrt ((2 : ℝ) ^ k)) * (1 / Real.sqrt ((2 : ℝ) ^ k)))
          • ((hadamardMat k * hadamardMat k) *ᵥ x) := by
        rw [← Matrix.mulVec_mulVec, smul_smul]
    _ = ((1 / Real.sqrt ((2 : ℝ) ^ k)) * (1 / Real.sqrt ((2 : ℝ) ^ k)))
          • (((2 : ℝ) ^ k) • x) := by
        rw [hadamardMat_mul_self, Matrix.smul_mulVec, Matrix.one_mulVec]
    _ = (((1 / Real.sqrt ((2 : ℝ) ^ k)) * (1 / Real.sqrt ((2 : ℝ) ^ k))) * (2 : ℝ) ^ k) • x := by
        rw [smul_smul]
    _ = x := by
        rw [← pow_two, rotationCoef_sq_mul, one_smul]

/-! ## The rotation as a block `Stage` -/

/-- The normalized Hadamard rotation packaged as a `Stage` on the magnitude-capped domain
`{x | ∀ i, |x_i| ≤ B}`. `exec` is the caller-supplied executed map with its defect bounded by
`delta` on the domain; `L = √(2^k)` is the `ℓ∞` expansion (`hadamardRef_lipschitz_sup`). The
executed-kernel link (`fwht_fixed_refines`) is a separate landing and is not imported here. -/
def rotationStage (k : ℕ) (s : Fin (2 ^ k) → ℝ) (hs : IsSign s) (B delta : ℝ)
    (exec : (Fin (2 ^ k) → ℝ) → (Fin (2 ^ k) → ℝ)) (hdelta : 0 ≤ delta)
    (hdefect : ∀ x, (∀ i, |x i| ≤ B) → ‖exec x - hadamardRef k s x‖ ≤ delta) :
    Stage (Fin (2 ^ k) → ℝ) (Fin (2 ^ k) → ℝ) where
  ref := hadamardRef k s
  exec := exec
  dom := {x | ∀ i, |x i| ≤ B}
  delta := delta
  L := Real.sqrt ((2 : ℝ) ^ k)
  delta_nonneg := hdelta
  L_nonneg := Real.sqrt_nonneg _
  defect := fun x hx => hdefect x hx
  expansion := hadamardRef_lipschitz_sup k s hs

end

end Nous.Model
