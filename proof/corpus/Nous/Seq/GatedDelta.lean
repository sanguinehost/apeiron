import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic.FinCases

/-!
# Gated Delta Rule and Chunkwise Affine Composition

This module mechanizes the gated delta rule and its parallel / chunkwise
formulation from:
- **Gated DeltaNet**: Songlin Yang, Jan Kautz, Arash Hatamizadeh,
  *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464 (2024).
  - Recurrent step equation: Eq. (line 453, 515):
    `S_t = S_{t-1}(α_t (I - β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`.
- **DeltaNet Chunking**: Songlin Yang et al.,
  *Parallelizing Linear Transformers with the Delta Rule over Sequence Length*,
  arXiv:2406.06484 (2024).
  - Compact WY transform derivation: Appendix B.1 (lines 1985-2084).
- **Reference Implementation**:
  `fla/ops/gated_delta_rule/naive.py:50-59` in Flash-Linear-Attention.

## Mathematical Structure

Over a commutative ring `R` and dimensions `dv, dk : ℕ`, the hidden state is
`S_t : Matrix (Fin dv) (Fin dk) R`. Given inputs:
- key `k_t : Fin dk → R`
- value `v_t : Fin dv → R`
- gate `α_t : R`
- learning rate / write strength `β_t : R`

The update is:
`S_t = α_t • (S_{t-1} * (1 - β_t • (k_t ⊗ k_t))) + β_t • (v_t ⊗ k_t)`
which is an affine transformation:
`S ↦ S * M_t + N_t`
with transition matrix `M_t = α_t • (1 - β_t • (k_t ⊗ k_t))`
and injection matrix `N_t = β_t • (v_t ⊗ k_t)`.

Affine maps `S ↦ S * M + N` form a monoid under composition. The associative scan
allows any sequence of steps to be chunked into consecutive blocks or evaluated
along an arbitrary merge tree (`recurrent_eq_chunked`). The unrolled recurrence
from `S_0 = 0` yields the closed form `closed_form`. For intra-chunk matmul-shaped
computation, the cumulative product of Householder-like factors
`Π_{r=1}^C (1 - β_r k_r k_rᵀ)` evaluates via the compact WY representation to
`1 - Σ_{r=1}^C w_r k_rᵀ` (`wy_form`), extended to the gated case by scaling
(`gated_wy_form`).
-/

open Matrix

namespace Nous.Seq.GatedDelta

variable {dv dk : ℕ} {R : Type*} [CommRing R]

/-! ### Affine Step Monoid -/

/-- An affine step sending `S : Matrix (Fin dv) (Fin dk) R` to `S * M + N`. -/
@[ext]
structure AffineStep (dv dk : ℕ) (R : Type*) where
  M : Matrix (Fin dk) (Fin dk) R
  N : Matrix (Fin dv) (Fin dk) R

/-- Applying an affine step to a matrix state `S`. -/
def affApply (p : AffineStep dv dk R) (S : Matrix (Fin dv) (Fin dk) R) :
    Matrix (Fin dv) (Fin dk) R :=
  S * p.M + p.N

/-- The identity affine step `S ↦ S * 1 + 0 = S`. -/
def affId : AffineStep dv dk R :=
  ⟨1, 0⟩

/-- Composing two affine steps: applying `q` first, then `p`.
`(S * q.M + q.N) * p.M + p.N = S * (q.M * p.M) + (q.N * p.M + p.N)`. -/
def affComp (p q : AffineStep dv dk R) : AffineStep dv dk R :=
  ⟨q.M * p.M, q.N * p.M + p.N⟩

/-- Composition of affine steps is associative. -/
theorem affComp_assoc (a b c : AffineStep dv dk R) :
    affComp (affComp a b) c = affComp a (affComp b c) := by
  ext:1
  · dsimp [affComp]
    rw [Matrix.mul_assoc]
  · dsimp [affComp]
    rw [Matrix.add_mul, Matrix.mul_assoc, add_assoc]

/-- Left identity for affine composition. -/
theorem affComp_one_left (p : AffineStep dv dk R) :
    affComp affId p = p := by
  ext:1
  · dsimp [affComp, affId]
    rw [Matrix.mul_one]
  · dsimp [affComp, affId]
    rw [Matrix.mul_one, add_zero]

/-- Right identity for affine composition. -/
theorem affComp_one_right (p : AffineStep dv dk R) :
    affComp p affId = p := by
  ext:1
  · dsimp [affComp, affId]
    rw [Matrix.one_mul]
  · dsimp [affComp, affId]
    rw [Matrix.zero_mul, zero_add]

/-- Applying a composed step equals applying the inner step then the outer step. -/
theorem affApply_comp (p q : AffineStep dv dk R) (S : Matrix (Fin dv) (Fin dk) R) :
    affApply (affComp p q) S = affApply p (affApply q S) := by
  dsimp [affApply, affComp]
  rw [Matrix.add_mul, Matrix.mul_assoc, add_assoc]

@[simp] theorem affId_apply (S : Matrix (Fin dv) (Fin dk) R) :
    affApply affId S = S := by
  dsimp [affApply, affId]
  rw [Matrix.mul_one, add_zero]

instance : Mul (AffineStep dv dk R) := ⟨affComp⟩
instance : One (AffineStep dv dk R) := ⟨affId⟩

instance : Monoid (AffineStep dv dk R) where
  mul := affComp
  one := affId
  mul_assoc := affComp_assoc
  one_mul := affComp_one_left
  mul_one := affComp_one_right

/-! ### Gated Delta Step Construction -/

/-- Construct an affine step from the gated delta parameters.
Reference: arXiv:2412.06464 Eq. (line 453, 515); fla/ops/gated_delta_rule/naive.py:50-59. -/
def stepAffine (α β : R) (k : Fin dk → R) (v : Fin dv → R) : AffineStep dv dk R :=
  ⟨α • (1 - β • vecMulVec k k), β • vecMulVec v k⟩

/-- Gated delta query read out `o_t = S_t *ᵥ q_t`. -/
def output (S : Matrix (Fin dv) (Fin dk) R) (q : Fin dk → R) : Fin dv → R :=
  S *ᵥ q

/-- The recurrent gated delta step equals applying the affine step. -/
theorem step_eq_affApply (α β : R) (k : Fin dk → R) (v : Fin dv → R)
    (S : Matrix (Fin dv) (Fin dk) R) :
    α • (S * (1 - β • vecMulVec k k)) + β • vecMulVec v k =
      affApply (stepAffine α β k v) S := by
  dsimp [affApply, stepAffine]
  rw [Matrix.mul_smul]

/-! ### Recurrence and Associative Scan -/

/-- Composed affine step of a list of steps. -/
def affFold (steps : List (AffineStep dv dk R)) : AffineStep dv dk R :=
  steps.foldl (fun acc s => affComp s acc) affId

/-- Sequential recurrence folding affine applications from an initial state `S0`. -/
def recur (steps : List (AffineStep dv dk R)) (S0 : Matrix (Fin dv) (Fin dk) R) :
    Matrix (Fin dv) (Fin dk) R :=
  steps.foldl (fun acc s => affApply s acc) S0

private theorem affFold_accum (steps : List (AffineStep dv dk R)) (acc : AffineStep dv dk R) :
    steps.foldl (fun acc s => affComp s acc) acc =
      affComp (steps.foldl (fun acc s => affComp s acc) affId) acc := by
  induction steps generalizing acc with
  | nil =>
    simp only [List.foldl_nil]
    exact (affComp_one_left acc).symm
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (affComp s acc), ih (affComp s affId)]
    rw [affComp_one_right, affComp_assoc]

/-- Scan split: composing concatenated step lists. -/
theorem scan_split (xs ys : List (AffineStep dv dk R)) :
    affFold (xs ++ ys) = affComp (affFold ys) (affFold xs) := by
  simp only [affFold]
  rw [List.foldl_append, affFold_accum]

theorem affFold_cons (s : AffineStep dv dk R) (rest : List (AffineStep dv dk R)) :
    affFold (s :: rest) = affComp (affFold rest) s := by
  have h : s :: rest = [s] ++ rest := rfl
  rw [h, scan_split]
  have hs : affFold [s] = s := by
    dsimp [affFold]
    rw [affComp_one_right]
  rw [hs]

private theorem affApply_fold (steps : List (AffineStep dv dk R)) (acc : AffineStep dv dk R)
    (S0 : Matrix (Fin dv) (Fin dk) R) :
    affApply (steps.foldl (fun acc s => affComp s acc) acc) S0 =
      steps.foldl (fun z s => affApply s z) (affApply acc S0) := by
  induction steps generalizing acc S0 with
  | nil => rfl
  | cons s rest ih =>
    simp only [List.foldl_cons]
    rw [ih (affComp s acc) S0, affApply_comp]

/-- The parallel scan law: recurrence equals single composed application. -/
theorem recur_eq_scan (steps : List (AffineStep dv dk R)) (S0 : Matrix (Fin dv) (Fin dk) R) :
    recur steps S0 = affApply (affFold steps) S0 := by
  rw [recur, affFold, affApply_fold, affId_apply]

theorem recur_append (xs ys : List (AffineStep dv dk R)) (S0 : Matrix (Fin dv) (Fin dk) R) :
    recur (xs ++ ys) S0 = recur ys (recur xs S0) := by
  simp only [recur, List.foldl_append]

/-- Sequential recurrence over chunked inputs equals folding chunk recurrences. -/
theorem recurrent_eq_chunked_recur (chunks : List (List (AffineStep dv dk R)))
    (S0 : Matrix (Fin dv) (Fin dk) R) :
    recur chunks.flatten S0 = chunks.foldl (fun acc c => recur c acc) S0 := by
  induction chunks generalizing S0 with
  | nil => rfl
  | cons c rest ih =>
    simp only [List.flatten_cons, List.foldl_cons]
    rw [recur_append, ih]

/-- Main chunking equivalence: sequential recurrence equals chunked affine composition. -/
theorem recurrent_eq_chunked (chunks : List (List (AffineStep dv dk R)))
    (S0 : Matrix (Fin dv) (Fin dk) R) :
    recur chunks.flatten S0 =
      chunks.foldl (fun acc c => affApply (affFold c) acc) S0 := by
  rw [recurrent_eq_chunked_recur]
  congr 1
  funext acc c
  rw [recur_eq_scan]

/-! ### Closed Form -/

/-- Cumulative transition matrix of a list of steps. -/
def matProd (steps : List (AffineStep dv dk R)) : Matrix (Fin dk) (Fin dk) R :=
  (affFold steps).M

/-- Unrolled closed form sum: `S_T = Σ_s N_s * Π_{r=s+1}^T M_r`. -/
def closedFormSum (steps : List (AffineStep dv dk R)) : Matrix (Fin dv) (Fin dk) R :=
  match steps with
  | [] => 0
  | s :: rest => s.N * matProd rest + closedFormSum rest

theorem affFold_N_eq_closedFormSum (steps : List (AffineStep dv dk R)) :
    (affFold steps).N = closedFormSum steps := by
  induction steps with
  | nil =>
    dsimp [affFold, affId, closedFormSum]
  | cons s rest ih =>
    rw [affFold_cons]
    dsimp [affComp, matProd, closedFormSum]
    rw [ih, add_comm]

/-- Unrolled closed form from zero initial state. -/
theorem closed_form (steps : List (AffineStep dv dk R)) :
    recur steps 0 = closedFormSum steps := by
  rw [recur_eq_scan]
  dsimp [affApply]
  rw [Matrix.zero_mul, zero_add]
  exact affFold_N_eq_closedFormSum steps

/-! ### Compact WY Representation (DeltaNet Chunking) -/

/-- Sum of outer products `Σ_i w_i k_iᵀ`. -/
def sumVecMulVec (pairs : List ((Fin dk → R) × (Fin dk → R))) : Matrix (Fin dk) (Fin dk) R :=
  (pairs.map (fun (w, k) => vecMulVec w k)).sum

/-- Vector sum `Σ_{i < r} (k_iᵀ k_r) • w_i`.
Reference: arXiv:2406.06484 Appendix B.1, line 2079. -/
def wyVectorSum (pairs : List ((Fin dk → R) × (Fin dk → R))) (k : Fin dk → R) : Fin dk → R :=
  (pairs.map (fun (w, k') => (k' ⬝ᵥ k) • w)).sum

theorem sumVecMulVec_mul_vecMulVec (pairs : List ((Fin dk → R) × (Fin dk → R))) (k : Fin dk → R) :
    sumVecMulVec pairs * vecMulVec k k = vecMulVec (wyVectorSum pairs k) k := by
  induction pairs with
  | nil =>
    simp [sumVecMulVec, wyVectorSum]
  | cons p rest ih =>
    obtain ⟨w, k'⟩ := p
    change (vecMulVec w k' + sumVecMulVec rest) * vecMulVec k k =
      vecMulVec ((k' ⬝ᵥ k) • w + wyVectorSum rest k) k
    rw [Matrix.add_mul, ih]
    rw [Matrix.vecMulVec_mul_vecMulVec]
    rw [Matrix.vecMulVec_smul, ← Matrix.smul_vecMulVec]
    rw [Matrix.add_vecMulVec]

theorem sumVecMulVec_append (p1 p2 : List ((Fin dk → R) × (Fin dk → R))) :
    sumVecMulVec (p1 ++ p2) = sumVecMulVec p1 + sumVecMulVec p2 := by
  simp [sumVecMulVec, List.map_append, List.sum_append]

private theorem test_matrix_expand (A B : Matrix (Fin dk) (Fin dk) R) :
    (1 - A) * (1 - B) = 1 - A - B + A * B := by
  rw [Matrix.sub_mul, Matrix.one_mul, Matrix.mul_sub, Matrix.mul_one]
  rw [sub_sub_eq_add_sub]
  rw [add_sub_right_comm]
  rw [sub_right_comm]

/-- Single step inductive equation for the WY transform. -/
theorem wy_step_prod (pairs : List ((Fin dk → R) × (Fin dk → R))) (β : R) (k : Fin dk → R) :
    let w : Fin dk → R := β • (k - wyVectorSum pairs k)
    (1 - sumVecMulVec pairs) * (1 - β • vecMulVec k k) =
      1 - sumVecMulVec (pairs ++ [(w, k)]) := by
  intro w
  have hsplit : sumVecMulVec (pairs ++ [(w, k)]) = sumVecMulVec pairs + vecMulVec w k := by
    rw [sumVecMulVec_append]
    simp [sumVecMulVec]
  rw [hsplit]
  have hmul := test_matrix_expand (sumVecMulVec pairs) (β • vecMulVec k k)
  rw [hmul]
  rw [Matrix.mul_smul, sumVecMulVec_mul_vecMulVec]
  dsimp [w]
  rw [smul_sub, Matrix.sub_vecMulVec, smul_vecMulVec, smul_vecMulVec]
  rw [sub_add]
  rw [sub_sub]

/-- Step accumulator updating the list of `(w_r, k_r)` pairs. -/
def wyAccum (pairs : List ((Fin dk → R) × (Fin dk → R))) (step : R × (Fin dk → R)) :
    List ((Fin dk → R) × (Fin dk → R)) :=
  pairs ++ [(step.1 • (step.2 - wyVectorSum pairs step.2), step.2)]

/-- Compute the sequence of `(w_r, k_r)` pairs from `(β_r, k_r)` inputs. -/
def wyTransform (steps : List (R × (Fin dk → R))) : List ((Fin dk → R) × (Fin dk → R)) :=
  steps.foldl wyAccum []

/-- Cumulative transition product with `α = 1`: `Π_{r=1}^C (1 - β_r k_r k_rᵀ)`. -/
def deltaProduct (steps : List (R × (Fin dk → R))) : Matrix (Fin dk) (Fin dk) R :=
  steps.foldl (fun acc (β, k) => acc * (1 - β • vecMulVec k k)) 1

theorem deltaProduct_accum (steps : List (R × (Fin dk → R)))
    (pairs : List ((Fin dk → R) × (Fin dk → R))) :
    steps.foldl (fun acc (β, k) => acc * (1 - β • vecMulVec k k)) (1 - sumVecMulVec pairs) =
      1 - sumVecMulVec (steps.foldl wyAccum pairs) := by
  induction steps generalizing pairs with
  | nil => rfl
  | cons s rest ih =>
    obtain ⟨β, k⟩ := s
    simp only [List.foldl_cons]
    dsimp [wyAccum]
    rw [wy_step_prod pairs β k]
    exact ih _

/-- Compact WY representation theorem for the plain delta rule:
`Π_{r=1}^C (1 - β_r k_r k_rᵀ) = 1 - Σ_{r=1}^C w_r k_rᵀ`.
Reference: arXiv:2406.06484 Appendix B.1, lines 1985-2084. -/
theorem wy_form (steps : List (R × (Fin dk → R))) :
    deltaProduct steps = 1 - sumVecMulVec (wyTransform steps) := by
  have h := deltaProduct_accum steps []
  simp only [sumVecMulVec, List.map_nil, List.sum_nil, sub_zero] at h
  exact h

/-! ### Gated Scaling Lemma -/

/-- Product of matrices scaled by scalar factors. -/
def scaledProduct {n : Type*} [Fintype n] [DecidableEq n]
    (steps : List (R × Matrix n n R)) : Matrix n n R :=
  steps.foldl (fun acc (α, A) => acc * (α • A)) 1

/-- Product of matrices without scalar scaling. -/
def unscaledProduct {n : Type*} [Fintype n] [DecidableEq n]
    (steps : List (R × Matrix n n R)) : Matrix n n R :=
  steps.foldl (fun acc (_, A) => acc * A) 1

private theorem scaledProduct_accum {n : Type*} [Fintype n] [DecidableEq n]
    (steps : List (R × Matrix n n R)) (c : R) (acc : Matrix n n R) :
    steps.foldl (fun acc (α, A) => acc * (α • A)) (c • acc) =
      (c * (steps.map Prod.fst).prod) • steps.foldl (fun acc (_, A) => acc * A) acc := by
  induction steps generalizing c acc with
  | nil => simp
  | cons step rest ih =>
    obtain ⟨α, A⟩ := step
    simp only [List.foldl_cons, List.map_cons, List.prod_cons]
    have hstep : (c • acc) * (α • A) = (c * α) • (acc * A) := by
      rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul]
    rw [hstep]
    rw [ih (c * α) (acc * A)]
    rw [mul_assoc]

/-- Gated scaling lemma: factoring scalar gates out of an ordered matrix product. -/
theorem gated_scaling_lemma {n : Type*} [Fintype n] [DecidableEq n]
    (steps : List (R × Matrix n n R)) :
    scaledProduct steps = (steps.map Prod.fst).prod • unscaledProduct steps := by
  have h := scaledProduct_accum steps 1 1
  simp only [one_smul, one_mul] at h
  exact h

/-- Gated transition product: `Π_{r=1}^C α_r • (1 - β_r k_r k_rᵀ)`. -/
def gatedDeltaProduct (steps : List (R × R × (Fin dk → R))) : Matrix (Fin dk) (Fin dk) R :=
  steps.foldl (fun acc (α, β, k) => acc * (α • (1 - β • vecMulVec k k))) 1

private lemma foldl_gated_eq (steps : List (R × R × (Fin dk → R))) (acc : Matrix (Fin dk) (Fin dk) R) :
    List.foldl (fun acc (α, β, k) => acc * (α • (1 - β • vecMulVec k k))) acc steps =
      List.foldl (fun acc (α, A) => acc * (α • A)) acc
        (steps.map (fun (α, β, k) => (α, 1 - β • vecMulVec k k))) := by
  induction steps generalizing acc with
  | nil => rfl
  | cons s rest ih =>
    obtain ⟨α, β, k⟩ := s
    simp only [List.map_cons, List.foldl_cons]
    exact ih _

private lemma foldl_unscaled_eq (steps : List (R × R × (Fin dk → R))) (acc : Matrix (Fin dk) (Fin dk) R) :
    List.foldl (fun acc (_, A) => acc * A) acc
      (steps.map (fun (α, β, k) => (α, 1 - β • vecMulVec k k))) =
    List.foldl (fun acc (β, k) => acc * (1 - β • vecMulVec k k)) acc
      (steps.map (fun (_, β, k) => (β, k))) := by
  induction steps generalizing acc with
  | nil => rfl
  | cons s rest ih =>
    obtain ⟨α, β, k⟩ := s
    simp only [List.map_cons, List.foldl_cons]
    exact ih _

/-- Gated WY representation: combining scalar gate extraction and the compact WY transform. -/
theorem gated_wy_form (steps : List (R × R × (Fin dk → R))) :
    gatedDeltaProduct steps =
      (steps.map (·.1)).prod • (1 - sumVecMulVec (wyTransform (steps.map (fun x => (x.2.1, x.2.2))))) := by
  have hscale : gatedDeltaProduct steps =
      scaledProduct (steps.map (fun (α, β, k) => (α, 1 - β • vecMulVec k k))) := by
    dsimp [gatedDeltaProduct, scaledProduct]
    exact foldl_gated_eq steps 1
  rw [hscale, gated_scaling_lemma]
  congr 1
  · simp only [List.map_map, Function.comp_def]
  · dsimp [unscaledProduct]
    have hunscale :
        List.foldl (fun acc (_, A) => acc * A) 1
          (steps.map (fun (α, β, k) => (α, 1 - β • vecMulVec k k))) =
        deltaProduct (steps.map (fun (_, β, k) => (β, k))) := by
      dsimp [deltaProduct]
      exact foldl_unscaled_eq steps 1
    rw [hunscale, wy_form]

/-! ### Witnesses and Non-Vacuity -/

namespace Witness

/-- Non-vacuity witness for `AffineStep`: identity step in dimension 2 over `ℤ`. -/
def witnessAffineStep : AffineStep 2 2 ℤ := affId

/-- Non-vacuity witness for hypothesis sets: domain of affine steps is inhabited. -/
instance : Inhabited (AffineStep 2 2 ℤ) := ⟨witnessAffineStep⟩

/-- Witness step 1 with dv = 2, dk = 2 over ℤ. -/
def s1 : AffineStep 2 2 ℤ := stepAffine 1 1 ![(1:ℤ), 0] ![(2:ℤ), 1]

/-- Witness step 2 with dv = 2, dk = 2 over ℤ. -/
def s2 : AffineStep 2 2 ℤ := stepAffine 1 1 ![(0:ℤ), 1] ![(1:ℤ), -1]

/-- Witness step 3 with dv = 2, dk = 2 over ℤ. -/
def s3 : AffineStep 2 2 ℤ := stepAffine 2 1 ![(1:ℤ), 1] ![(0:ℤ), 3]

/-- Initial state S0 = 0. -/
def S0 : Matrix (Fin 2) (Fin 2) ℤ := 0

/-- Recurrent output over 3 steps. -/
def seq_out : Matrix (Fin 2) (Fin 2) ℤ := recur [s1, s2, s3] S0

/-- Chunked output over 2 chunks: `[[s1, s2], [s3]]`. -/
def chunked_out : Matrix (Fin 2) (Fin 2) ℤ :=
  [[s1, s2], [s3]].foldl (fun acc c => affApply (affFold c) acc) S0

/-- Ground equality of recurrent vs chunked on the 3-step witness instance. -/
theorem seq_eq_chunked : seq_out = chunked_out := by
  ext i j
  fin_cases i <;> fin_cases j <;> rfl

#guard seq_out 0 0 == chunked_out 0 0
#guard seq_out 0 1 == chunked_out 0 1
#guard seq_out 1 0 == chunked_out 1 0
#guard seq_out 1 1 == chunked_out 1 1

/-- Three steps of (β, k) inputs for WY transform witness. -/
def wy_steps : List (ℤ × (Fin 2 → ℤ)) :=
  [(1, ![(1:ℤ), 0]), (1, ![(0:ℤ), 1]), (1, ![(1:ℤ), 1])]

def wy_prod : Matrix (Fin 2) (Fin 2) ℤ := deltaProduct wy_steps
def wy_ut : Matrix (Fin 2) (Fin 2) ℤ := 1 - sumVecMulVec (wyTransform wy_steps)

theorem wy_witness_eq : wy_prod = wy_ut := by
  ext i j
  fin_cases i <;> fin_cases j <;> rfl

#guard wy_prod 0 0 == wy_ut 0 0
#guard wy_prod 0 1 == wy_ut 0 1
#guard wy_prod 1 0 == wy_ut 1 0
#guard wy_prod 1 1 == wy_ut 1 1

end Witness

end Nous.Seq.GatedDelta
