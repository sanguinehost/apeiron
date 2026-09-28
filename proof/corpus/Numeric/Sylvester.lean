/-
Sylvester Hadamard algebra (`APEIRON-TERNARY-W3-T5`).

The Sylvester Hadamard matrix `H_{2^k}` is the `k`-fold Kronecker power of
`H_2 = !![1, 1; 1, -1]`, the signed Walsh–Hadamard matrix the PrismML fork rotates its weights
by (`docs/epics/ternary-hadamard.md`, `docs/epics/ternary-hadamard.md`). It is defined here by its
block recurrence `H_{k+1} = [[H_k, H_k], [H_k, -H_k]]` (Sylvester, 1867) through `Matrix.fromBlocks`
on the equivalence `Fin (2^k) ⊕ Fin (2^k) ≃ Fin (2^{k+1})`.

## Landed here

* `sylvesterH`, `sylvester_succ`: the matrix and its block recurrence.
* `sylvester_transpose`, `sylvester_mul_self`: it is symmetric and `H · H = 2^k · 1`, so the
  unnormalized integer butterflies are exact.
* `sylvester_one`, `sylvester_one_channelPair`: `H_2` is `channelPair 1` at `σ = 1`, the butterfly
  `(u, v) ↦ (u + v, u − v)` of `Nous.Model.Decomposition` (`ringUpper`/`ringLower`) in the
  idempotent basis of the split-complex face `ℤ[x]/(x² − 1)`.
* `two_not_isUnit_int`: `2` is not a unit of `ℤ`, so `hyperbolic_decomposition`'s `2h = 1` has no
  integer witness: the factor `1/√2` per FWHT stage is exactly this obstruction.
* `sylvester_mulVec_butterfly`: the single-stage identity `H_{k+1} x = (H_k x₀ + H_k x₁,
  H_k x₀ − H_k x₁)` on the two halves of `x`, the shape the FWHT loop proof consumes.

The axiom cone is `{propext, Classical.choice, Quot.sound}`; no `sorry`, `axiom` or `decide` over an
extracted body appears.
-/
import Nous.Model.Decomposition
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Group.Int.Units
import Mathlib.Data.Fin.SuccPred
import Mathlib.Data.Matrix.Block
import Mathlib.Data.Matrix.Mul
import Mathlib.LinearAlgebra.Matrix.Reindex
import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_word

open Apeiron.Trichotomy.AnyRing
open Nous.Model
open scoped Matrix

/-! ## The Sylvester matrix and its block recurrence -/

/-- The canonical block reindex `Fin (2^k) ⊕ Fin (2^k) ≃ Fin (2^{k+1})`: the first summand is the
top half, the second the bottom half. -/
def sylvesterBlockEquiv (k : ℕ) : Fin (2 ^ k) ⊕ Fin (2 ^ k) ≃ Fin (2 ^ (k + 1)) :=
  finSumFinEquiv.trans (finCongr (by rw [pow_succ]; ring))

/-- The Sylvester Hadamard matrix `H_{2^k} : Matrix (Fin (2^k)) (Fin (2^k)) ℤ`, by the block
recurrence `H_0 = 1`, `H_{k+1} = [[H_k, H_k], [H_k, -H_k]]`. -/
def sylvesterH : (k : ℕ) → Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ
  | 0 => 1
  | k + 1 =>
      Matrix.reindex (sylvesterBlockEquiv k) (sylvesterBlockEquiv k)
        (Matrix.fromBlocks (sylvesterH k) (sylvesterH k) (sylvesterH k) (-(sylvesterH k)))

/-- The block recurrence as a standalone equation. -/
theorem sylvester_succ (k : ℕ) :
    sylvesterH (k + 1) =
      Matrix.reindex (sylvesterBlockEquiv k) (sylvesterBlockEquiv k)
        (Matrix.fromBlocks (sylvesterH k) (sylvesterH k) (sylvesterH k) (-(sylvesterH k))) :=
  rfl

/-- The empty face: `H_0 = 1`. -/
theorem sylvester_zero : sylvesterH 0 = 1 :=
  rfl

/-! ## Reindexing lemmas -/

/-- A reindexed matrix agrees with the original at an index and its image. -/
theorem reindex_apply_equiv {m n α : Type*} (e : m ≃ n) (M : Matrix m m α) (i j : m) :
    (Matrix.reindex e e M) (e i) (e j) = M i j := by
  simp [Matrix.reindex_apply, Matrix.submatrix_apply]

/-- Reindexing carries the matrix-vector product: reading the reindexed product at `e i` is
reading the original product at `i`. -/
theorem reindex_mulVec_apply {m n : Type*} [Fintype m] [Fintype n]
    (e : m ≃ n) (M : Matrix m m ℤ) (x : n → ℤ) (i : m) :
    (Matrix.reindex e e M).mulVec x (e i) = M.mulVec (x ∘ e) i := by
  simp only [Matrix.mulVec, Matrix.reindex_apply, Matrix.submatrix_apply, Equiv.symm_apply_apply]
  exact comp_equiv_symm_dotProduct (M i) x e

/-! ## Symmetry and the self-product -/

/-- `sylvesterH k` is symmetric. -/
theorem sylvester_transpose (k : ℕ) : (sylvesterH k)ᵀ = sylvesterH k := by
  induction k with
  | zero => simp [sylvester_zero]
  | succ k ih =>
      rw [sylvester_succ, Matrix.transpose_reindex, Matrix.fromBlocks_transpose, Matrix.transpose_neg,
        ih]

/-- `c • 1` on an index type is the constant diagonal matrix. -/
theorem smul_one_eq_diagonal {n : Type*} [Fintype n] [DecidableEq n] (c : ℤ) :
    c • (1 : Matrix n n ℤ) = Matrix.diagonal (fun _ => c) := by
  ext i j
  by_cases h : i = j
  · subst h; simp
  · simp [Matrix.smul_apply, h]

/-- Reindexing a constant scalar matrix fixes it. -/
theorem reindex_smul_one {m n : Type*} [Fintype m] [DecidableEq m] [Fintype n] [DecidableEq n]
    (e : m ≃ n) (c : ℤ) :
    Matrix.reindex e e (c • (1 : Matrix m m ℤ)) = c • (1 : Matrix n n ℤ) := by
  rw [smul_one_eq_diagonal, smul_one_eq_diagonal]
  ext i j
  simp only [Matrix.reindex_apply, Matrix.submatrix_apply, Matrix.diagonal_apply]
  by_cases h : i = j
  · subst h; simp
  · have h' : e.symm i ≠ e.symm j := fun hh => h (e.symm.injective hh)
    simp [h, h']

/-- Two constant scalar blocks are the scalar on the sum index. -/
theorem fromBlocks_smul_one {m n : Type*} [Fintype m] [DecidableEq m] [Fintype n]
    [DecidableEq n] (c : ℤ) :
    Matrix.fromBlocks (c • (1 : Matrix m m ℤ)) 0 0 (c • (1 : Matrix n n ℤ)) =
      c • (1 : Matrix (m ⊕ n) (m ⊕ n) ℤ) := by
  rw [smul_one_eq_diagonal, smul_one_eq_diagonal, smul_one_eq_diagonal,
    Matrix.fromBlocks_diagonal]
  ext i j
  rcases i with i | i <;> rcases j with j | j <;> simp [Matrix.diagonal_apply]

/-- The `RingEquiv` form of `reindex_smul_one`. -/
theorem reindexRingEquiv_smul_one {m n : Type*} [Fintype m] [DecidableEq m] [Fintype n]
    [DecidableEq n] (e : m ≃ n) (c : ℤ) :
    (Matrix.reindexRingEquiv ℤ e) (c • (1 : Matrix m m ℤ)) = c • (1 : Matrix n n ℤ) :=
  reindex_smul_one e c

/-- `H · H = 2^k · 1`: the unnormalized integer butterflies are exact. -/
theorem sylvester_mul_self (k : ℕ) :
    sylvesterH k * sylvesterH k = (2 ^ k : ℤ) • (1 : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) := by
  induction k with
  | zero => norm_num [sylvester_zero]
  | succ k ih =>
      have hstep :
          sylvesterH (k + 1) =
            (Matrix.reindexRingEquiv ℤ (sylvesterBlockEquiv k))
              (Matrix.fromBlocks (sylvesterH k) (sylvesterH k) (sylvesterH k)
                (-(sylvesterH k))) :=
        rfl
      rw [hstep, ← map_mul, Matrix.fromBlocks_multiply]
      have hdiag :
          (sylvesterH k * sylvesterH k + sylvesterH k * sylvesterH k :
              Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) =
            ((2 ^ k : ℤ) + 2 ^ k) • (1 : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) := by
        rw [ih, add_smul]
      have hleft :
          (sylvesterH k * sylvesterH k + sylvesterH k * -(sylvesterH k) :
              Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) = 0 := by
        rw [mul_neg, ih, add_neg_cancel]
      have hright :
          (sylvesterH k * sylvesterH k + -(sylvesterH k) * sylvesterH k :
              Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) = 0 := by
        rw [neg_mul, ih, add_neg_cancel]
      have hbr :
          (sylvesterH k * sylvesterH k + -(sylvesterH k) * -(sylvesterH k) :
              Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) =
            ((2 ^ k : ℤ) + 2 ^ k) • (1 : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) ℤ) := by
        rw [neg_mul_neg, ih, add_smul]
      rw [hdiag, hleft, hright, hbr]
      have hpow : ((2 ^ k : ℤ) + 2 ^ k) = (2 : ℤ) ^ (k + 1) := by
        rw [pow_succ]; ring
      rw [hpow, fromBlocks_smul_one, reindexRingEquiv_smul_one]

/-! ## The two-point face and the channel projection -/

/-- The first half of the two-point block reindex is the top index. -/
theorem sylvesterBlockEquiv_zero_inl :
    sylvesterBlockEquiv 0 (Sum.inl (0 : Fin (2 ^ 0))) = (0 : Fin 2) := by
  decide

/-- The second half of the two-point block reindex is the bottom index. -/
theorem sylvesterBlockEquiv_zero_inr :
    sylvesterBlockEquiv 0 (Sum.inr (0 : Fin (2 ^ 0))) = (1 : Fin 2) := by
  decide

/-- `H_2 = !![1, 1; 1, -1]`. -/
theorem sylvester_one : sylvesterH 1 = !![(1 : ℤ), 1; 1, -1] := by
  decide

/-- `H_2` is the `σ = 1` channel projection: the butterfly `(u, v) ↦ (u + v, u − v)` in the
idempotent basis of `ℤ[x]/(x² − 1)`. -/
theorem sylvester_one_channelPair (z : RingCell ℤ) :
    (sylvesterH 1).mulVec ![z.1, z.2] = ![z.1 + z.2, z.1 - z.2] ∧
      channelPair (1 : ℤ) z = (z.1 + z.2, z.1 - z.2) := by
  constructor
  · rw [sylvester_one]
    ext i
    fin_cases i
    all_goals simp [Matrix.mulVec, dotProduct, Fin.sum_univ_two, Matrix.of_apply,
      Matrix.cons_val_zero, Matrix.cons_val_one]
    all_goals ring
  · apply Prod.ext <;> simp only [channelPair, ringUpper, ringLower] <;> ring

/-! ## The integer normalization obstruction -/

/-- `2` is not a unit of `ℤ`, so `hyperbolic_decomposition`'s `2h = 1` has no integer witness: the
`1/√2` normalization per FWHT stage is not representable over `ℤ`, which is why the integer
butterflies stay unnormalized and normalize once at the end. -/
theorem two_not_isUnit_int : ¬ IsUnit (2 : ℤ) := by
  rw [Int.isUnit_iff]
  norm_num

/-! ## The butterfly stage identity -/

/-- One Sylvester stage: for `x : Fin (2^{k+1}) → ℤ` split into halves `x₀ = x ∘ e ∘ inl` and
`x₁ = x ∘ e ∘ inr`, the product `H_{k+1} x` is `(H_k x₀ + H_k x₁, H_k x₀ − H_k x₁)`. -/
theorem sylvester_mulVec_butterfly (k : ℕ) (x : Fin (2 ^ (k + 1)) → ℤ) :
    (fun s : Fin (2 ^ k) ⊕ Fin (2 ^ k) =>
        (sylvesterH (k + 1)).mulVec x (sylvesterBlockEquiv k s)) =
      Sum.elim
        (sylvesterH k *ᵥ (x ∘ sylvesterBlockEquiv k ∘ Sum.inl)
          + sylvesterH k *ᵥ (x ∘ sylvesterBlockEquiv k ∘ Sum.inr))
        (sylvesterH k *ᵥ (x ∘ sylvesterBlockEquiv k ∘ Sum.inl)
          - sylvesterH k *ᵥ (x ∘ sylvesterBlockEquiv k ∘ Sum.inr)) := by
  funext s
  rw [sylvester_succ]
  rcases s with i | i
  · rw [reindex_mulVec_apply, Matrix.fromBlocks_mulVec]
    simp only [Sum.elim_inl, Matrix.neg_mulVec, sub_eq_add_neg]
    rfl
  · rw [reindex_mulVec_apply, Matrix.fromBlocks_mulVec]
    simp only [Sum.elim_inr, Matrix.neg_mulVec, sub_eq_add_neg]
    rfl

end apeiron_word
