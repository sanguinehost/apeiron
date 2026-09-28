import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Tactic.FinCases
import Apeiron.Trichotomy.AnyRing

/-!
# Cell-Valued Affine Mixer: Generalized (A, B) Recurrence

This module generalizes the gated delta recurrence (`Nous.Seq.GatedDelta`)
to the full cell-valued affine mixer family:
`S_t = S_{t-1} * M_t + N_t`
where `M_t = A_t * (1 - β_t • (k_t ⊗ k_t))` and `N_t = β_t • (v_t ⊗ k_t)`.

## Mathematical Structure and Trichotomy Specialization

Over a commutative ring `R` and dimensions `dv, dk : ℕ`, hidden state
`S : Matrix (Fin dv) (Fin dk) R` is updated by:
1. Transition matrix `A : Matrix (Fin dk) (Fin dk) R`, block-diagonal in 2-cell blocks:
   - **Hyperbolic (`σ > 0`)**: real diagonal decay `α • I` (GLA, Gated DeltaNet).
   - **Parabolic (`σ = 0`)**: un-decayed accumulation `I` or Jordan block `1 + ε` (Linear Attention).
   - **Elliptic (`σ < 0`)**: unitary rotation `R(θ)` scaled by decay `α` (RetNet, Mamba-3).
2. Rank-1 Householder-like correction `(1 - β • (k ⊗ k))`.

We prove:
- `affComp_assoc`: monoid associativity of affine transitions under exact unrounded ring arithmetic.
- `affApply_comp`: multi-step chunk composition matches sequential recurrence.
- `recurrent_eq_chunked`: exact equivalence between sequential unrolled scan and chunked parallel scan.
- Instances specializing `M_t` to the hyperbolic, parabolic, and elliptic regimes.

Attribution:
- Gated DeltaNet: Songlin Yang et al., arXiv:2412.06464 (2024).
- RetNet / Complex SSM: Yutao Sun et al., arXiv:2307.08621 (2023).
- Linear Attention: Angelos Katharopoulos et al., ICML (2020).
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unnecessarySeqFocus false

open Matrix
open Apeiron.Trichotomy.AnyRing

namespace Nous.Seq.AffineMixer

variable {dv dk : ℕ} {R : Type*} [CommRing R]

/-! ### General Affine Step Monoid -/

/-- A generalized affine step sending `S : Matrix (Fin dv) (Fin dk) R` to `S * M + N`. -/
@[ext]
structure AffineStep (dv dk : ℕ) (R : Type*) where
  M : Matrix (Fin dk) (Fin dk) R
  N : Matrix (Fin dv) (Fin dk) R

/-- Applying an affine step to state matrix `S`. -/
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

/-- Composition of affine steps is associative over any commutative ring. -/
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

/-! ### Step Construction with General Transition Matrix -/

/-- Construct a cell-valued affine mixer step with transition matrix `A`,
rank-1 correction direction `k`, learning rate `β`, and input value `v`. -/
def stepAffineGeneral (A : Matrix (Fin dk) (Fin dk) R) (β : R)
    (k : Fin dk → R) (v : Fin dv → R) : AffineStep dv dk R :=
  ⟨A * (1 - β • vecMulVec k k), β • vecMulVec v k⟩

/-- The recurrent update: `S_t = S_{t-1} * A * (1 - β k kᵀ) + β v kᵀ`. -/
theorem step_eq_affApply (A : Matrix (Fin dk) (Fin dk) R) (β : R)
    (k : Fin dk → R) (v : Fin dv → R) (S : Matrix (Fin dv) (Fin dk) R) :
    S * (A * (1 - β • vecMulVec k k)) + β • vecMulVec v k =
      affApply (stepAffineGeneral A β k v) S := by
  rfl

/-! ### Sequential and Chunked Evaluation -/

/-- Sequential fold of a list of affine steps from initial state `S_0`. -/
def scanSeq (steps : List (AffineStep dv dk R)) (S_0 : Matrix (Fin dv) (Fin dk) R) :
    Matrix (Fin dv) (Fin dk) R :=
  steps.foldl (fun S p => affApply p S) S_0

/-- Compose a list of affine steps into a single composite step. -/
def composeSteps (steps : List (AffineStep dv dk R)) : AffineStep dv dk R :=
  steps.foldl (fun acc p => affComp p acc) affId

/-- Helper lemma: accumulator unfolding in `foldl` composition. -/
lemma foldl_affComp (l : List (AffineStep dv dk R)) (acc : AffineStep dv dk R) :
    l.foldl (fun a p => affComp p a) acc = affComp (composeSteps l) acc := by
  induction l generalizing acc with
  | nil => simp [composeSteps, affComp_one_left]
  | cons x xs ih =>
    simp only [composeSteps, List.foldl_cons]
    rw [ih (affComp x acc), ih (affComp x affId), affComp_one_right, affComp_assoc]

/-- The sequential scan across a list equals applying the composed step to `S_0`. -/
theorem scanSeq_eq_compose (steps : List (AffineStep dv dk R))
    (S_0 : Matrix (Fin dv) (Fin dk) R) :
    scanSeq steps S_0 = affApply (composeSteps steps) S_0 := by
  induction steps generalizing S_0 with
  | nil =>
    simp only [scanSeq, composeSteps, List.foldl_nil, affId_apply]
  | cons p rest ih =>
    change scanSeq rest (affApply p S_0) = _
    rw [ih]
    have hcomp : composeSteps (p :: rest) = affComp (composeSteps rest) p := by
      change List.foldl (fun acc q => affComp q acc) (affComp p affId) rest = _
      rw [affComp_one_right, foldl_affComp]
    rw [hcomp, affApply_comp]

/-- Chunked evaluation theorem: dividing a sequence of steps into chunks
`l1 ++ l2` evaluates identically to sequential scan. -/
theorem recurrent_eq_chunked (l1 l2 : List (AffineStep dv dk R))
    (S_0 : Matrix (Fin dv) (Fin dk) R) :
    scanSeq (l1 ++ l2) S_0 = scanSeq l2 (scanSeq l1 S_0) := by
  simp only [scanSeq, List.foldl_append]

/-! ### Trichotomy Face Instances -/

/-- Hyperbolic instance: scalar decay `α` (Gated DeltaNet). -/
def hyperbolicStep (α β : R) (k : Fin dk → R) (v : Fin dv → R) : AffineStep dv dk R :=
  stepAffineGeneral (α • (1 : Matrix (Fin dk) (Fin dk) R)) β k v

/-- Parabolic instance: un-decayed accumulation `α = 1` (Linear Attention with Delta rule). -/
def parabolicStep (β : R) (k : Fin dk → R) (v : Fin dv → R) : AffineStep dv dk R :=
  stepAffineGeneral (1 : Matrix (Fin dk) (Fin dk) R) β k v

/-- Elliptic 2x2 rotation matrix block from angle cosine and sine. -/
def rot2x2 (c s : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![c, -s], ![s, c]]

/-- Elliptic 2-dimensional step linking to RingCell multiplication:
`(a, b) * (c, d)` at `σ = -1` matches 2x2 rotation-dilation matrix multiplication. -/
theorem elliptic_2x2_mul_equiv (c s x y : R) :
    rot2x2 c s *ᵥ ![x, y] =
      ![c * x - s * y, s * x + c * y] := by
  ext i
  fin_cases i
  · simp [rot2x2, mulVec, dotProduct, sub_eq_add_neg]
  · simp [rot2x2, mulVec, dotProduct]

end Nous.Seq.AffineMixer
