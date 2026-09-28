/-
# The Sphere Object: RMSNorm and Rotations through Combinator and Physics Lenses

Combinator and physics / geometry foundations of the sphere object, covering:
1. Combinator lens:
   - `sylvester_church`: Sylvester/Hadamard matrix order 2^k as the k-fold iterate of the doubling step
     under Church numerals and first-order reading `readFO`.
   - `fwht_add_count`: Butterfly recurrence C(k+1) = 2·C(k) + 2^(k+1), C(0) = 0 has closed form k·2^k.
   - `sumsq_tree`: RMSNorm reduction Σ x_i² is an additive list homomorphism; any partition and merge tree
     give the same sum (instance of `add_hom_tree`).
     Doc comment: this is why the norm's single global reduction can be fused with the partial-sum reduction
     of the preceding column-split projection.
2. Physics / geometry lens:
   - `rms_projects_to_sphere`: RMSNorm projects ℝ^n \ {0} onto the sphere of radius √n.
   - `rms_equivariant`: RMSNorm commutes with orthogonal transformations Q (for any ε ≥ 0).
   - `gain_fold`: Gain and orthogonal rotation fold into the subsequent weight matrix:
     `W (g ⊙ norm(x)) = (W · diag g · Qᵀ)(norm(Q x))`.
   - `hadamard_is_rotation`: The normalized Hadamard matrix (1/√(2^k))·H_k is orthogonal.
   - `cayley_orthogonal`: Real / rational Cayley transform Q = (1 − A)⁻¹(1 + A) of a skew matrix is orthogonal.
     Doc comment: Links to `SpinorLoop.cayley_conserves_born_density` (complex/unitary version):
     Born density Σ|ψ|² is what RMSNorm's Σx² is.
3. Witness namespace:
   - Order-4 Sylvester equals 2-fold doubling and matches explicit ±1 matrix (decide on ℤ).
   - 2×2 Cayley rotation from A = [[0, -1/2], [1/2, 0]] equals the 3-4-5 rational rotation [[3/5, -4/5], [4/5, 3/5]].
   - Numeric instance of `rms_projects_to_sphere` on Fin 4.

Trust boundary: kernel-checked algebra and real analysis under the closed axiom cone
`{propext, Classical.choice, Quot.sound}`.
-/

import Apeiron.Combinator.Basic
import Apeiron.Combinator.Confluence
import Nous.Combinator.Church
import Nous.Combinator.Orbit
import Nous.Combinator.ListHom
import Nous.Model.RmsVectorStage
import Nous.Model.RotationStage
import Numeric.RmsNormIdeal
import Numeric.Sylvester
import Substrate.StationaryPhase.SpinorLoop

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Combinator.Sphere

open Apeiron.Combinator
open Apeiron.Combinator.Comb
open Nous.Combinator
open Nous.Combinator.Orbit
open Nous.Combinator.ListHom
open Nous.Model
open apeiron_word
open scoped Matrix

noncomputable section

/-! ## A. Combinator Lens -/

/-- The Sylvester doubling step on the dependent pair `(k, H_k)`. -/
def sylvesterDoublingStep :
    (Σ k : ℕ, Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) → (Σ k : ℕ, Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) :=
  fun ⟨k, H⟩ => ⟨k + 1, Matrix.reindex (sylvesterBlockEquiv k) (sylvesterBlockEquiv k)
    (Matrix.fromBlocks H H H (-H))⟩

/-- The base Sylvester matrix `H_0 = 1`. -/
def sylvesterBase : Σ k : ℕ, Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ :=
  ⟨0, sylvesterH 0⟩

/-- The Sylvester matrix of order `2^k` is the `k`-fold iterate of `sylvesterDoublingStep`. -/
theorem sylvester_iterate (k : ℕ) :
    (sylvesterDoublingStep^[k] sylvesterBase) = ⟨k, sylvesterH k⟩ := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [Function.iterate_succ_apply', ih]
    rfl

/-- **Headline 1: `sylvester_church`.** The Sylvester/Hadamard matrix of order `2^k` is the `k`-fold
iterate of the doubling step `D(H) = fromBlocks H H H (−H)` from `H_0 = 1` (`Function.iterate`), i.e.
`church k` applied to `D`, read under the `Orbit`-style first-order reading `readFO`. -/
theorem sylvester_church (k : ℕ) :
    readFO sylvesterDoublingStep sylvesterBase (iterApp (var 0) k (var 1)) =
      some ⟨k, sylvesterH k⟩ ∧
    Reduces (app (app (church k) (var 0)) (var 1)) (iterApp (var 0) k (var 1)) := by
  constructor
  · rw [readFO_iterApp, sylvester_iterate]
  · exact church_spec k (var 0) (var 1)

/-- The butterfly addition count recurrence: `C(0) = 0`, `C(k+1) = 2·C(k) + 2^(k+1)`. -/
def fwhtCost : ℕ → ℕ
  | 0 => 0
  | k + 1 => 2 * fwhtCost k + 2 ^ (k + 1)

@[simp] theorem fwhtCost_zero : fwhtCost 0 = 0 := rfl
@[simp] theorem fwhtCost_succ (k : ℕ) : fwhtCost (k + 1) = 2 * fwhtCost k + 2 ^ (k + 1) := rfl

/-- **Headline 2: `fwht_add_count`.** The butterfly recurrence `C(k+1) = 2·C(k) + 2^(k+1)`, `C(0) = 0`
has closed form `C(k) = k·2^k` (the addition count of the fast Walsh–Hadamard transform). -/
theorem fwht_add_count (k : ℕ) : fwhtCost k = k * 2 ^ k := by
  induction k with
  | zero => rfl
  | succ k ih =>
    rw [fwhtCost_succ, ih, pow_succ]
    ring

/-- Sum of squares of elements in a list of reals. -/
def sumSq (l : List ℝ) : ℝ :=
  (l.map (fun x => x ^ 2)).sum

/-- `sumSq` is an additive list homomorphism. -/
theorem sumSq_isAddListHom : IsAddListHom sumSq where
  nil := by simp [sumSq]
  append := by
    intro l m
    simp [sumSq, List.map_append, List.sum_append]

/-- **Headline 3: `sumsq_tree`.** The RMSNorm reduction `Σ x_i²` is an additive list homomorphism;
any partition of the vector and any merge tree give the same sum (instance of `add_hom_tree`).

*Design connection:* This is why the norm's single global reduction can be fused with the
partial-sum reduction of the preceding column-split projection.
Because `sumSq` is an additive list homomorphism, intermediate partial sums computed across column
shards assemble associatively into the global radicand without tree-structure bias. -/
theorem sumsq_tree (t : MTree ℝ) :
    evalAdd sumSq t = sumSq (flat t) :=
  add_hom_tree sumSq sumSq_isAddListHom t


/-! ## B. Physics / Geometry Lens -/

/-- **Headline 4: `rms_projects_to_sphere`.** For `x ≠ 0` in `ℝ^n`, the ideal RMSNorm output
`y = x / √(mean(x²))` satisfies `Σ y_i² = n` — RMSNorm is the radial projection onto the sphere of
radius `√n`. -/
theorem rms_projects_to_sphere {n : ℕ} (x : Fin n → ℝ) (hne : ∃ i, x i ≠ 0) :
    let s := ∑ i, x i ^ 2
    let meanSq := s / (n : ℝ)
    let y := fun i => x i / Real.sqrt meanSq
    ∑ i, y i ^ 2 = (n : ℝ) := by
  intro s meanSq y
  have hs_pos : 0 < s := by
    obtain ⟨i, hi⟩ := hne
    have hsq : 0 < x i ^ 2 := sq_pos_of_ne_zero hi
    have hsum : x i ^ 2 ≤ s := Finset.single_le_sum (fun j _ => sq_nonneg (x j)) (Finset.mem_univ i)
    exact lt_of_lt_of_le hsq hsum
  have hn_pos : 0 < (n : ℝ) := by
    obtain ⟨i, _⟩ := hne
    have : 0 < n := Fin.pos i
    positivity
  have hmean_pos : 0 < meanSq := div_pos hs_pos hn_pos
  have hsq_sqrt : (Real.sqrt meanSq) ^ 2 = meanSq := Real.sq_sqrt (le_of_lt hmean_pos)
  have hy_sq : ∀ i, y i ^ 2 = x i ^ 2 / meanSq := by
    intro i
    dsimp [y]
    rw [div_pow, hsq_sqrt]
  calc ∑ i, y i ^ 2
      = ∑ i, (x i ^ 2 / meanSq) := by
        apply Finset.sum_congr rfl
        intro i _
        exact hy_sq i
    _ = (∑ i, x i ^ 2) / meanSq := by rw [← Finset.sum_div]
    _ = s / (s / (n : ℝ)) := rfl
    _ = (n : ℝ) := by
        rw [div_div_eq_mul_div, mul_comm, mul_div_cancel_right₀ (n : ℝ) (ne_of_gt hs_pos)]

/-- Orthogonal matrices preserve the sum of squares (`ℓ²` norm). -/
theorem orthogonal_sum_sq {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : Qᵀ * Q = 1) (x : Fin n → ℝ) :
    ∑ i, (Q *ᵥ x) i ^ 2 = ∑ i, x i ^ 2 := by
  have h1 : ∑ i, (Q *ᵥ x) i ^ 2 = (Q *ᵥ x) ⬝ᵥ (Q *ᵥ x) := by
    dsimp [dotProduct]; apply Finset.sum_congr rfl; intro i _; ring
  have h2 : ∑ i, x i ^ 2 = x ⬝ᵥ x := by
    dsimp [dotProduct]; apply Finset.sum_congr rfl; intro i _; ring
  rw [h1, h2]
  calc (Q *ᵥ x) ⬝ᵥ (Q *ᵥ x)
      = (Q *ᵥ x) ᵥ* Q ⬝ᵥ x := by rw [Matrix.dotProduct_mulVec]
    _ = (Qᵀ *ᵥ (Q *ᵥ x)) ⬝ᵥ x := by rw [← Matrix.mulVec_transpose]
    _ = ((Qᵀ * Q) *ᵥ x) ⬝ᵥ x := by rw [← Matrix.mulVec_mulVec]
    _ = (1 *ᵥ x) ⬝ᵥ x := by rw [hQ]
    _ = x ⬝ᵥ x := by rw [Matrix.one_mulVec]

/-- **Headline 5: `rms_equivariant`.** For `Q` orthogonal (`Qᵀ Q = 1`), `rmsIdealVec`-style
normalisation commutes with `Q`: `norm(Q x) = Q (norm x)`, holding for any `ε ≥ 0`.
This is the computational invariance QuaRot relies on. -/
theorem rms_equivariant {n : ℕ} (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : Qᵀ * Q = 1)
    (eps : ℝ) (x : Fin n → ℝ) :
    rmsIdealVec eps (Q *ᵥ x) = Q *ᵥ (rmsIdealVec eps x) := by
  have hinv : ∑ j, (Q *ᵥ x) j ^ 2 = ∑ j, x j ^ 2 := orthogonal_sum_sq Q hQ x
  have hscale : Real.sqrt ((∑ j, (Q *ᵥ x) j ^ 2) / (n : ℝ) + eps) =
                Real.sqrt ((∑ j, x j ^ 2) / (n : ℝ) + eps) := by rw [hinv]
  set denom := Real.sqrt ((∑ j, x j ^ 2) / (n : ℝ) + eps)
  have h_lhs : rmsIdealVec eps (Q *ᵥ x) = (1 / denom) • (Q *ᵥ x) := by
    ext i
    dsimp [rmsIdealVec, denom]
    rw [hscale, div_eq_inv_mul, one_div]
  have h_rhs : Q *ᵥ (rmsIdealVec eps x) = (1 / denom) • (Q *ᵥ x) := by
    have h_vec : rmsIdealVec eps x = (1 / denom) • x := by
      ext i
      dsimp [rmsIdealVec, denom]
      rw [div_eq_inv_mul, one_div]
    rw [h_vec, Matrix.mulVec_smul]
  rw [h_lhs, h_rhs]

/-- **Headline 6: `gain_fold`.** With gain `g`, `W (g ⊙ norm(x)) = (W · diag g · Qᵀ)(norm(Q x))` for
orthogonal `Q` — a rotation of the residual stream is absorbed exactly into the next weight matrix. -/
theorem gain_fold {n : ℕ} (W : Matrix (Fin n) (Fin n) ℝ) (g : Fin n → ℝ)
    (Q : Matrix (Fin n) (Fin n) ℝ) (hQ : Qᵀ * Q = 1) (eps : ℝ) (x : Fin n → ℝ)
    (h_equiv : rmsIdealVec eps (Q *ᵥ x) = Q *ᵥ (rmsIdealVec eps x)) :
    W *ᵥ (g * rmsIdealVec eps x) =
      (W * Matrix.diagonal g * Qᵀ) *ᵥ (rmsIdealVec eps (Q *ᵥ x)) := by
  have h_diag : g * rmsIdealVec eps x = Matrix.diagonal g *ᵥ (rmsIdealVec eps x) := by
    ext i
    simp [Matrix.mulVec_diagonal]
  rw [h_diag, h_equiv]
  have h1 : (W * Matrix.diagonal g * Qᵀ) *ᵥ (Q *ᵥ rmsIdealVec eps x)
      = (W * Matrix.diagonal g) *ᵥ (Qᵀ *ᵥ (Q *ᵥ rmsIdealVec eps x)) := by
    exact (Matrix.mulVec_mulVec (Q *ᵥ rmsIdealVec eps x) (W * Matrix.diagonal g) Qᵀ).symm
  have h2 : Qᵀ *ᵥ (Q *ᵥ rmsIdealVec eps x) = (Qᵀ * Q) *ᵥ rmsIdealVec eps x := by
    exact Matrix.mulVec_mulVec (rmsIdealVec eps x) Qᵀ Q
  have h3 : (W * Matrix.diagonal g) *ᵥ (rmsIdealVec eps x) = W *ᵥ (Matrix.diagonal g *ᵥ rmsIdealVec eps x) := by
    exact (Matrix.mulVec_mulVec (rmsIdealVec eps x) W (Matrix.diagonal g)).symm
  rw [h1, h2, hQ, Matrix.one_mulVec, h3]

/-- The normalized real Hadamard matrix of order `2^k`. -/
def hadamardRotMat (k : ℕ) : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℝ :=
  (1 / Real.sqrt ((2 : ℝ) ^ k)) • hadamardMat k

/-- **Headline 7: `hadamard_is_rotation`.** `(1/√(2^k))·sylvesterH k` is orthogonal (derived from
`sylvester_mul_self`), hence `rms_equivariant` applies directly to it. -/
theorem hadamard_is_rotation (k : ℕ) :
    (hadamardRotMat k)ᵀ * hadamardRotMat k = 1 := by
  dsimp [hadamardRotMat]
  rw [hadamardMat_transpose k, Matrix.smul_mul, Matrix.mul_smul, smul_smul]
  rw [hadamardMat_mul_self k, smul_smul]
  have hcoef : (1 / Real.sqrt ((2 : ℝ) ^ k)) * (1 / Real.sqrt ((2 : ℝ) ^ k)) * (2 : ℝ) ^ k = 1 := by
    rw [← pow_two, rotationCoef_sq_mul]
  rw [hcoef, one_smul]

/-- **Headline 8: `cayley_orthogonal`.** For skew `A` (`Aᵀ = -A`), the Cayley transform
`Q = (1 - A)⁻¹ (1 + A)` is orthogonal: `Qᵀ Q = 1`.
Formulated over any commutative ring `R` (such as `ℝ` or `ℚ`) with an inverse witness `B` for `1 - A`.
When `A` has rational entries, `Q` has exact rational entries with no trigonometry needed.

*Physics connection:* Links to `SpinorLoop.cayley_conserves_born_density` in `Substrate/StationaryPhase/SpinorLoop.lean`:
in quantum mechanics the Cayley time-step preserves the Born density `Σ |ψ_i|²`, exactly as the real Cayley
rotation preserves the sum of squares `Σ x_i²` of the residual stream under RMSNorm. -/
theorem cayley_orthogonal {n : ℕ} {R : Type*} [CommRing R] (A : Matrix (Fin n) (Fin n) R)
    (hskew : Aᵀ = -A) (B : Matrix (Fin n) (Fin n) R)
    (hinv : B * (1 - A) = 1) (hinv2 : (1 - A) * B = 1) :
    let Q := B * (1 + A)
    Qᵀ * Q = 1 := by
  intro Q
  have h1 : (1 - A)ᵀ = 1 + A := by
    rw [Matrix.transpose_sub, Matrix.transpose_one, hskew, sub_neg_eq_add]
  have h2 : (1 + A)ᵀ = 1 - A := by
    rw [Matrix.transpose_add, Matrix.transpose_one, hskew, ← sub_eq_add_neg]
  have hB_inv : Bᵀ * (1 + A) = 1 := by
    rw [← h1, ← Matrix.transpose_mul, hinv2, Matrix.transpose_one]
  have hcomm : (1 - A) * (1 + A) = (1 + A) * (1 - A) := by
    noncomm_ring
  have hBcomm : B * (1 + A) = (1 + A) * B := by
    calc B * (1 + A) = B * (1 + A) * 1 := by rw [Matrix.mul_one]
      _ = B * (1 + A) * ((1 - A) * B) := by rw [hinv2]
      _ = B * ((1 + A) * (1 - A)) * B := by simp only [Matrix.mul_assoc]
      _ = B * ((1 - A) * (1 + A)) * B := by rw [hcomm]
      _ = (B * (1 - A)) * ((1 + A) * B) := by simp only [Matrix.mul_assoc]
      _ = 1 * ((1 + A) * B) := by rw [hinv]
      _ = (1 + A) * B := by rw [Matrix.one_mul]
  dsimp [Q]
  have hQ_comm : B * (1 + A) = (1 + A) * B := hBcomm
  rw [hQ_comm, Matrix.transpose_mul, h2]
  calc (Bᵀ * (1 - A)) * ((1 + A) * B)
      = Bᵀ * ((1 - A) * (1 + A)) * B := by simp only [Matrix.mul_assoc]
    _ = Bᵀ * ((1 + A) * (1 - A)) * B := by rw [hcomm]
    _ = (Bᵀ * (1 + A)) * ((1 - A) * B) := by simp only [Matrix.mul_assoc]
    _ = 1 * 1 := by rw [hB_inv, hinv2]
    _ = 1 := Matrix.mul_one 1


/-! ## Witness Namespace -/

namespace Witnesses

/-- The explicit 4×4 Sylvester Hadamard matrix. -/
def H4_expected : Matrix (Fin 4) (Fin 4) ℤ :=
  !![ 1,  1,  1,  1;
      1, -1,  1, -1;
      1,  1, -1, -1;
      1, -1, -1,  1]

/-- Witness 1: Order-4 Sylvester matrix matches the explicit 4×4 table. -/
theorem witness_sylvester_order4 : sylvesterH 2 = H4_expected := by
  decide

/-- Witness 1': Order-4 Sylvester matrix equals the 2-fold doubling from `sylvesterBase`. -/
theorem witness_sylvester_doubling_order4 :
    (sylvesterDoublingStep (sylvesterDoublingStep sylvesterBase)).2 = sylvesterH 2 := by
  rfl

/-- Skew-symmetric 2×2 generator over `ℚ`. -/
def A_skew2 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![0, -1/2; 1/2, 0]

/-- Identity 2×2 matrix over `ℚ`. -/
def I2_rat : Matrix (Fin 2) (Fin 2) ℚ := 1

/-- Expected 3-4-5 rational rotation matrix. -/
def expected_Q345 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![3/5, -4/5; 4/5, 3/5]

/-- Witness 2: The 2×2 Cayley rotation from `A = [[0, -1/2], [1/2, 0]]` produces the 3-4-5 triple
rotation `[[3/5, -4/5], [4/5, 3/5]]` over `ℚ`. -/
theorem witness_cayley_rational_rotation :
    (I2_rat - A_skew2)⁻¹ * (I2_rat + A_skew2) = expected_Q345 := by
  have hmul : (I2_rat - A_skew2) * expected_Q345 = I2_rat + A_skew2 := by
    ext i j
    fin_cases i <;> fin_cases j <;> simp [Matrix.mul_apply, Fin.sum_univ_two, A_skew2, I2_rat, expected_Q345] <;> norm_num
  have hdet : (I2_rat - A_skew2).det ≠ 0 := by
    rw [Matrix.det_fin_two]
    simp [I2_rat, A_skew2]
    norm_num
  have hunit : IsUnit (I2_rat - A_skew2).det := isUnit_iff_ne_zero.mpr hdet
  have hinv : (I2_rat - A_skew2)⁻¹ * (I2_rat - A_skew2) = 1 := Matrix.nonsing_inv_mul (I2_rat - A_skew2) hunit
  calc (I2_rat - A_skew2)⁻¹ * (I2_rat + A_skew2)
      = (I2_rat - A_skew2)⁻¹ * ((I2_rat - A_skew2) * expected_Q345) := by rw [hmul]
    _ = ((I2_rat - A_skew2)⁻¹ * (I2_rat - A_skew2)) * expected_Q345 := by rw [Matrix.mul_assoc]
    _ = 1 * expected_Q345 := by rw [hinv]
    _ = expected_Q345 := by rw [Matrix.one_mul]

/-- A test vector in `ℝ^4`. -/
def test_vec4 : Fin 4 → ℝ := ![1, 1, 1, 1]

/-- Witness 3: Numeric instance of `rms_projects_to_sphere` for `x = [1, 1, 1, 1]ᵀ`. -/
theorem witness_rms_projects_to_sphere :
    ∑ i : Fin 4, (rmsIdealVec 0 test_vec4 i) ^ 2 = (4 : ℝ) := by
  have hsum : ∑ j : Fin 4, test_vec4 j ^ 2 = 4 := by
    rw [Fin.sum_univ_four]
    dsimp [test_vec4]
    norm_num
  have hcoord : ∀ i : Fin 4, rmsIdealVec 0 test_vec4 i = 1 := by
    intro i
    have hi : test_vec4 i = 1 := by fin_cases i <;> rfl
    have hroot : Real.sqrt ((∑ j : Fin 4, test_vec4 j ^ 2) / (4 : ℝ) + 0) = 1 := by
      rw [hsum]
      norm_num
    dsimp [rmsIdealVec]
    rw [hi, hroot, div_one]
  calc ∑ i : Fin 4, (rmsIdealVec 0 test_vec4 i) ^ 2
      = ∑ i : Fin 4, (1 : ℝ) ^ 2 := by
        apply Finset.sum_congr rfl
        intro i _
        rw [hcoord i]
    _ = 4 := by
      rw [Fin.sum_univ_four]
      norm_num

end Witnesses

end

end Nous.Combinator.Sphere
