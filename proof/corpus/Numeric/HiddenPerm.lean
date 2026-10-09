/-
Hidden-axis permutation symmetry of a transformer block (
the ternary-hadamard note Part II). Permuting the hidden axis, with the weights that read it
permuted by columns and the weights that write it permuted by rows, leaves the function unchanged up
to the same permutation of the output. The order along which a one-bit error-shaping quantiser runs
over the hidden axis is therefore free.

Landed here, over any commutative ring `α`, with `P = σ.permMatrix α` (`P *ᵥ v = v ∘ σ`, Mathlib
`permMatrix_mulVec`) and `Pᵀ * P = 1` (`perm_transpose_mul`):

* `perm_coordwise`: coordinatewise `f` commutes with `P`;
* `hidden_perm_mlp`: `(W_next * Pᵀ) (f ((P * W_prev) x)) = W_next (f (W_prev x))`;
* `hidden_perm_gated`: the same for `W_down (f (W_gate x) ⊙ (W_up x))`;
* `rms_perm`: `rmsNorm r (P g) (P v) = P (rmsNorm r g v)`;
* `hidden_perm_residual_block`: one pre-norm gated residual block;
* `mlp_nonvacuous`: over `ℚ`, `n = 2`, `σ = swap`, `f = max · 0`: the compensated output equals the
  original and the uncompensated one differs.

Grade: `proved`, a pure algebra statement over an arbitrary commutative ring; no Rust or extracted
body is read. Trust boundary: no claim about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V, the
`unsafe` container, floating-point summation order (the statements are exact over the ring), or model
quality.
-/
import Mathlib

namespace Numeric.HiddenPerm

open Finset
open scoped Matrix

section Core

variable {α : Type*} [CommRing α] {n : ℕ} (σ : Equiv.Perm (Fin n))

/-- `Pᵀ * P = 1` for a permutation matrix. -/
theorem perm_transpose_mul :
    (σ.permMatrix α)ᵀ * σ.permMatrix α = 1 := by
  rw [Matrix.transpose_permMatrix, ← Matrix.permMatrix_mul]
  simp

/-- `Pᵀ (P v) = v`. -/
theorem perm_transpose_mulVec (v : Fin n → α) :
    (σ.permMatrix α)ᵀ *ᵥ (σ.permMatrix α *ᵥ v) = v := by
  rw [Matrix.mulVec_mulVec, perm_transpose_mul, Matrix.one_mulVec]

/-- Input-side compensation: `(W * Pᵀ) (P x) = W x`. -/
theorem mulVec_perm_cancel {m : ℕ} (W : Matrix (Fin m) (Fin n) α) (x : Fin n → α) :
    (W * (σ.permMatrix α)ᵀ) *ᵥ (σ.permMatrix α *ᵥ x) = W *ᵥ x := by
  rw [← Matrix.mulVec_mulVec, perm_transpose_mulVec]

/-- Coordinatewise `f` commutes with a permutation matrix. -/
theorem perm_coordwise (f : α → α) (v : Fin n → α) :
    (fun i => f ((σ.permMatrix α *ᵥ v) i)) = σ.permMatrix α *ᵥ (fun i => f (v i)) := by
  rw [Matrix.permMatrix_mulVec, Matrix.permMatrix_mulVec]
  rfl

/-- A permuted hidden layer with compensated next-layer weights is unchanged:
`W_prev` is permuted on rows (`P * W_prev`), `W_next` on columns (`W_next * Pᵀ`). -/
theorem hidden_perm_mlp {k m : ℕ} (f : α → α) (W_prev : Matrix (Fin n) (Fin k) α)
    (W_next : Matrix (Fin m) (Fin n) α) (x : Fin k → α) :
    (W_next * (σ.permMatrix α)ᵀ) *ᵥ (fun i => f (((σ.permMatrix α * W_prev) *ᵥ x) i))
      = W_next *ᵥ (fun i => f ((W_prev *ᵥ x) i)) := by
  have h : (σ.permMatrix α * W_prev) *ᵥ x = σ.permMatrix α *ᵥ (W_prev *ᵥ x) :=
    (Matrix.mulVec_mulVec _ _ _).symm
  rw [h, perm_coordwise, mulVec_perm_cancel]

/-- The same in `submatrix` form: rows of `W_prev` reindexed by `σ`, columns of `W_next` by `σ`
(`(P * W) i j = W (σ i) j`, `(W * Pᵀ) i j = W i (σ j)`). -/
theorem hidden_perm_mlp_submatrix {k m : ℕ} (f : α → α) (W_prev : Matrix (Fin n) (Fin k) α)
    (W_next : Matrix (Fin m) (Fin n) α) (x : Fin k → α) :
    (W_next.submatrix id σ) *ᵥ (fun i => f ((W_prev.submatrix σ id *ᵥ x) i))
      = W_next *ᵥ (fun i => f ((W_prev *ᵥ x) i)) := by
  have h1 : W_prev.submatrix σ id = σ.permMatrix α * W_prev := by
    ext i j
    simp [Matrix.mul_apply, Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply]
  have h2 : W_next.submatrix id σ = W_next * (σ.permMatrix α)ᵀ := by
    ext i j
    simp [Matrix.mul_apply, Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply, Equiv.symm_apply_eq]
  rw [h1, h2, hidden_perm_mlp]

/-- Gated MLP: `W_gate`, `W_up` permuted on rows (`P * W`), `W_down` on columns (`W * Pᵀ`). -/
theorem hidden_perm_gated {k m : ℕ} (f : α → α) (W_gate W_up : Matrix (Fin n) (Fin k) α)
    (W_down : Matrix (Fin m) (Fin n) α) (x : Fin k → α) :
    (W_down * (σ.permMatrix α)ᵀ) *ᵥ
        (fun i => f (((σ.permMatrix α * W_gate) *ᵥ x) i) * ((σ.permMatrix α * W_up) *ᵥ x) i)
      = W_down *ᵥ (fun i => f ((W_gate *ᵥ x) i) * (W_up *ᵥ x) i) := by
  have : (fun i => f (((σ.permMatrix α * W_gate) *ᵥ x) i) * ((σ.permMatrix α * W_up) *ᵥ x) i)
      = σ.permMatrix α *ᵥ (fun i => f ((W_gate *ᵥ x) i) * (W_up *ᵥ x) i) := by
    rw [← Matrix.mulVec_mulVec, ← Matrix.mulVec_mulVec]
    simp only [Matrix.permMatrix_mulVec]
    rfl
  rw [this, mulVec_perm_cancel]

/-- RMSNorm with the normaliser `r` of the sum of squares left abstract (no square root). -/
def rmsNorm (r : α → α) (g v : Fin n → α) : Fin n → α :=
  fun i => g i * v i * r (∑ j, v j ^ 2)

/-- The sum of squares is invariant under a permutation of the coordinates. -/
theorem sumSq_perm (v : Fin n → α) :
    ∑ j, (σ.permMatrix α *ᵥ v) j ^ 2 = ∑ j, v j ^ 2 := by
  simp only [Matrix.permMatrix_mulVec, Function.comp_apply]
  exact Equiv.sum_comp σ (fun j => v j ^ 2)

/-- RMSNorm commutes with the hidden permutation when the gain is permuted too. -/
theorem rms_perm (r : α → α) (g v : Fin n → α) :
    rmsNorm r (σ.permMatrix α *ᵥ g) (σ.permMatrix α *ᵥ v)
      = σ.permMatrix α *ᵥ rmsNorm r g v := by
  funext i
  simp only [rmsNorm]
  rw [sumSq_perm]
  simp only [Matrix.permMatrix_mulVec, Function.comp_apply]
  rfl

/-- One pre-norm gated residual block on the residual stream `h`:
`h + W_down (f (W_gate (rmsNorm r g h)) ⊙ W_up (rmsNorm r g h))`. -/
def block {m : ℕ} (r : α → α) (f : α → α) (g : Fin n → α)
    (W_gate W_up : Matrix (Fin m) (Fin n) α) (W_down : Matrix (Fin n) (Fin m) α)
    (h : Fin n → α) : Fin n → α :=
  h + W_down *ᵥ (fun i => f ((W_gate *ᵥ rmsNorm r g h) i) * (W_up *ᵥ rmsNorm r g h) i)

/-- Permuting the residual stream is an exact symmetry of one pre-norm gated residual block.

Permuted: the stream `h ↦ P h`, the gain `g ↦ P g`, the input-side weights `W_gate`, `W_up`
(`m × n`, they read the hidden axis) by columns `W ↦ W * Pᵀ`, and the output-side weight `W_down`
(`n × m`, it writes the hidden axis) by rows `W_down ↦ P * W_down`. The intermediate (`m`) axis is
not permuted. The result is `P` applied to the original output. -/
theorem hidden_perm_residual_block {m : ℕ} (r f : α → α) (g h : Fin n → α)
    (W_gate W_up : Matrix (Fin m) (Fin n) α) (W_down : Matrix (Fin n) (Fin m) α) :
    block r f (σ.permMatrix α *ᵥ g) (W_gate * (σ.permMatrix α)ᵀ) (W_up * (σ.permMatrix α)ᵀ)
        (σ.permMatrix α * W_down) (σ.permMatrix α *ᵥ h)
      = σ.permMatrix α *ᵥ block r f g W_gate W_up W_down h := by
  unfold block
  rw [rms_perm, mulVec_perm_cancel, mulVec_perm_cancel, ← Matrix.mulVec_mulVec, Matrix.mulVec_add]

end Core

section NonVacuous

/-- `W_prev x = (-1, 4)`, so the hidden layer is `relu = (0, 4)` and the output is `8`. -/
def Wprev : Matrix (Fin 2) (Fin 2) ℚ := !![1, -2; 3, 1]
def Wnext : Matrix (Fin 1) (Fin 2) ℚ := !![1, 2]
def xin : Fin 2 → ℚ := ![1, 1]
def relu : ℚ → ℚ := fun t => max t 0
def swapP : Matrix (Fin 2) (Fin 2) ℚ := (Equiv.swap (0 : Fin 2) 1).permMatrix ℚ

/-- Over `ℚ`, `n = 2`, `σ = swap 0 1`, `f = max · 0`: the original output is `8`; permuting
`W_prev` with `W_next` compensated gives `8`; permuting `W_prev` alone gives `4 ≠ 8`. -/
theorem mlp_nonvacuous :
    (Wnext *ᵥ (fun i => relu ((Wprev *ᵥ xin) i))) = ![8]
    ∧ ((Wnext * swapPᵀ) *ᵥ (fun i => relu (((swapP * Wprev) *ᵥ xin) i))) = ![8]
    ∧ (Wnext *ᵥ (fun i => relu (((swapP * Wprev) *ᵥ xin) i))) = ![4]
    ∧ (Wnext *ᵥ (fun i => relu (((swapP * Wprev) *ᵥ xin) i))) ≠ ![8] := by
  have hP : swapP = !![0, 1; 1, 0] := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [swapP, Equiv.Perm.permMatrix, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply]
  have e1 : (Wnext *ᵥ (fun i => relu ((Wprev *ᵥ xin) i))) = ![8] := by
    ext i
    fin_cases i
    simp [Wnext, Wprev, xin, relu, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    norm_num
  have e2 : ((Wnext * swapPᵀ) *ᵥ (fun i => relu (((swapP * Wprev) *ᵥ xin) i))) = ![8] := by
    ext i
    fin_cases i
    simp [hP, Wnext, Wprev, xin, relu, Matrix.mulVec, dotProduct, Fin.sum_univ_two,
      Matrix.mul_apply]
    norm_num
  have e3 : (Wnext *ᵥ (fun i => relu (((swapP * Wprev) *ᵥ xin) i))) = ![4] := by
    ext i
    fin_cases i
    simp [hP, Wnext, Wprev, xin, relu, Matrix.mulVec, dotProduct, Fin.sum_univ_two,
      Matrix.mul_apply]
    norm_num
  refine ⟨e1, e2, e3, ?_⟩
  rw [e3]
  intro h
  have := congrFun h 0
  simp at this

end NonVacuous

end Numeric.HiddenPerm
