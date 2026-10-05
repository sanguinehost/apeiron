import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Nous.Seq.StatePrecision

/-!
# Carrier error of the gated delta state

`Nous.Seq.StatePrecision.errorSeq_le_steady_state` bounds the scalar sequence
`e_{t+1} = α·e_t + ε`. This module proves that the state error of the gated delta rule under any
rounding carrier obeys that sequence. That is the argument that licenses evaluating the recurrence
in a carrier cheaper than exact arithmetic: binary32 in llama.cpp
(`ggml/src/ggml-cpu/ops.cpp`, `ggml_compute_forward_gated_delta_net_one_chunk`, rev `0324696b8`),
Q32.32 in `src/engine/mixer.rs`.

Per value column `j` the gated delta step (Yang, Kautz and Hatamizadeh, *Gated Delta Networks*,
arXiv:2412.06464, 2024; llama.cpp `delta-net-base.cpp`) acts on the key-dimension vector `s` as

`s ↦ α·(s − β⟪s, k⟫ k) + (β·v_j)·k`.

With `‖k‖ = 1` and `0 ≤ β ≤ 2`, the map `s ↦ s − β⟪s, k⟫ k` scales the `k` direction by `1 − β`
and fixes its orthogonal complement; at `β = 2` it is the reflection of Householder (1958). It is
nonexpansive, so the step is `α`-Lipschitz, and a carrier whose per-step deviation from the exact
step is at most `ε` keeps the state within `errorSeq α ε t ≤ ε / (1 − α)` of the exact state for
every context length `t`. The perturbed-iteration argument is the standard forward error analysis
of a contraction (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002).

The carrier enters only through `ε`, its per-step deviation. Deriving `ε` for a named carrier
(Q32.32 `round_shift_32`, binary32, binary64) from its rounding unit and the step's operation
count is a separate obligation; this module does not claim it.

## Main results

- `householder_nonexpansive`: `‖s − β⟪s, k⟫ k‖ ≤ ‖s‖` for `‖k‖ = 1`, `0 ≤ β ≤ 2`.
- `deltaCol_lipschitz`: the gated delta column step is `α`-Lipschitz.
- `carrier_error_le_errorSeq`: a perturbed iteration of `α`-Lipschitz maps stays within `errorSeq`.
- `delta_carrier_error_uniform`: the gated delta state error is at most `ε / (1 − α)`, uniformly
  in the context length, for time-varying gates `0 ≤ α_t ≤ α < 1`, `0 ≤ β_t ≤ 2`.
- `readout_error_le`: the readout `⟪S, q⟫` inherits the state error scaled by `‖q‖`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Seq.DeltaCarrier

open Nous.Seq.StatePrecision
open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- One value column of the gated delta step on the key-dimension vector `s`:
`α·(s − β⟪s, k⟫ k) + (β·v)·k`, with decay `α`, write strength `β`, value `v` and key `k`. -/
noncomputable def deltaCol (α β v : ℝ) (k s : E) : E :=
  α • (s - (β * ⟪s, k⟫_ℝ) • k) + (β * v) • k

/-- The rank-one correction `s ↦ s − β⟪s, k⟫ k` is nonexpansive for a unit key and
`0 ≤ β ≤ 2`: `‖s − β⟪s, k⟫ k‖² = ‖s‖² − β(2 − β)⟪s, k⟫²`. -/
theorem householder_nonexpansive (β : ℝ) (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) (k s : E)
    (hk : ‖k‖ = 1) : ‖s - (β * ⟪s, k⟫_ℝ) • k‖ ≤ ‖s‖ := by
  have hsq : ‖s - (β * ⟪s, k⟫_ℝ) • k‖ ^ 2 = ‖s‖ ^ 2 - β * (2 - β) * ⟪s, k⟫_ℝ ^ 2 := by
    rw [norm_sub_sq_real, real_inner_smul_right, norm_smul, hk, mul_one, Real.norm_eq_abs,
      sq_abs]
    ring
  have hnn : 0 ≤ β * (2 - β) * ⟪s, k⟫_ℝ ^ 2 :=
    mul_nonneg (mul_nonneg hβ0 (by linarith)) (sq_nonneg _)
  have hle : ‖s - (β * ⟪s, k⟫_ℝ) • k‖ ^ 2 ≤ ‖s‖ ^ 2 := by rw [hsq]; linarith
  exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp hle

/-- The difference of two gated delta steps is the linear part applied to the difference: the
injection `(β·v)·k` cancels. -/
theorem deltaCol_sub (α β v : ℝ) (k s s' : E) :
    deltaCol α β v k s - deltaCol α β v k s' = α • ((s - s') - (β * ⟪s - s', k⟫_ℝ) • k) := by
  simp only [deltaCol, inner_sub_left, mul_sub, sub_smul, smul_sub]
  abel

/-- The gated delta column step is `α`-Lipschitz for `α ≥ 0`, `0 ≤ β ≤ 2` and a unit key. -/
theorem deltaCol_lipschitz (α β v : ℝ) (hα : 0 ≤ α) (hβ0 : 0 ≤ β) (hβ2 : β ≤ 2) (k : E)
    (hk : ‖k‖ = 1) (s s' : E) :
    ‖deltaCol α β v k s - deltaCol α β v k s'‖ ≤ α * ‖s - s'‖ := by
  rw [deltaCol_sub, norm_smul, Real.norm_eq_abs, abs_of_nonneg hα]
  exact mul_le_mul_of_nonneg_left (householder_nonexpansive β hβ0 hβ2 k (s - s') hk) hα

omit [InnerProductSpace ℝ E] in
/-- Perturbed iteration: the exact sequence is `x_{t+1} = f_t(x_t)`, the carried sequence
`X_{t+1}` deviates from `f_t(X_t)` by at most `ε`, both start equal, and every `f_t` is
`α`-Lipschitz with `α ≥ 0`. Then `‖X_t − x_t‖ ≤ errorSeq α ε t` for every `t`. -/
theorem carrier_error_le_errorSeq (f : ℕ → E → E) (α ε : ℝ) (hα : 0 ≤ α)
    (hf : ∀ t a b, ‖f t a - f t b‖ ≤ α * ‖a - b‖)
    (x X : ℕ → E) (hx : ∀ t, x (t + 1) = f t (x t)) (h0 : X 0 = x 0)
    (hδ : ∀ t, ‖X (t + 1) - f t (X t)‖ ≤ ε) :
    ∀ t, ‖X t - x t‖ ≤ errorSeq α ε t := by
  intro t
  induction t with
  | zero => simp [h0]
  | succ t ih =>
    rw [errorSeq_succ, hx t]
    calc ‖X (t + 1) - f t (x t)‖
        = ‖(X (t + 1) - f t (X t)) + (f t (X t) - f t (x t))‖ := by
          congr 1; abel
      _ ≤ ‖X (t + 1) - f t (X t)‖ + ‖f t (X t) - f t (x t)‖ := norm_add_le _ _
      _ ≤ ε + α * ‖X t - x t‖ := add_le_add (hδ t) (hf t _ _)
      _ ≤ ε + α * errorSeq α ε t := by gcongr
      _ = α * errorSeq α ε t + ε := by ring

omit [InnerProductSpace ℝ E] in
/-- `carrier_error_le_errorSeq` with a strict contraction: the carried state stays within
`ε / (1 − α)` of the exact state for every `t`. -/
theorem carrier_error_uniform (f : ℕ → E → E) (α ε : ℝ) (hα0 : 0 ≤ α) (hα1 : α < 1)
    (hε : 0 ≤ ε) (hf : ∀ t a b, ‖f t a - f t b‖ ≤ α * ‖a - b‖)
    (x X : ℕ → E) (hx : ∀ t, x (t + 1) = f t (x t)) (h0 : X 0 = x 0)
    (hδ : ∀ t, ‖X (t + 1) - f t (X t)‖ ≤ ε) :
    ∀ t, ‖X t - x t‖ ≤ ε / (1 - α) := fun t =>
  (carrier_error_le_errorSeq f α ε hα0 hf x X hx h0 hδ t).trans
    (errorSeq_le_steady_state α ε hα0 hα1 hε t)

/-- The gated delta state under any carrier with per-step deviation `ε`: with time-varying gates
`0 ≤ a_t ≤ α < 1`, `0 ≤ b_t ≤ 2`, unit keys `k_t` and values `v_t`, the carried column state
stays within `ε / (1 − α)` of the exact column state for every context length `t`. -/
theorem delta_carrier_error_uniform (α ε : ℝ) (hα0 : 0 ≤ α) (hα1 : α < 1) (hε : 0 ≤ ε)
    (a b v : ℕ → ℝ) (k : ℕ → E)
    (ha : ∀ t, 0 ≤ a t ∧ a t ≤ α) (hb : ∀ t, 0 ≤ b t ∧ b t ≤ 2) (hk : ∀ t, ‖k t‖ = 1)
    (x X : ℕ → E) (hx : ∀ t, x (t + 1) = deltaCol (a t) (b t) (v t) (k t) (x t))
    (h0 : X 0 = x 0)
    (hδ : ∀ t, ‖X (t + 1) - deltaCol (a t) (b t) (v t) (k t) (X t)‖ ≤ ε) :
    ∀ t, ‖X t - x t‖ ≤ ε / (1 - α) := by
  refine carrier_error_uniform (fun t => deltaCol (a t) (b t) (v t) (k t)) α ε hα0 hα1 hε
    ?_ x X hx h0 hδ
  intro t s s'
  calc ‖deltaCol (a t) (b t) (v t) (k t) s - deltaCol (a t) (b t) (v t) (k t) s'‖
      ≤ a t * ‖s - s'‖ :=
        deltaCol_lipschitz _ _ _ (ha t).1 (hb t).1 (hb t).2 _ (hk t) s s'
    _ ≤ α * ‖s - s'‖ := mul_le_mul_of_nonneg_right (ha t).2 (norm_nonneg _)

/-- The readout `o_j = ⟪S_j, q⟫` of a carried state differs from the exact readout by at most
`‖q‖` times the state error (Cauchy–Schwarz). -/
theorem readout_error_le (X x q : E) : |⟪X, q⟫_ℝ - ⟪x, q⟫_ℝ| ≤ ‖q‖ * ‖X - x‖ := by
  rw [← inner_sub_left, mul_comm]
  exact abs_real_inner_le_norm _ _

/-! ### Joint witness

The hypotheses of `delta_carrier_error_uniform` hold together at a nonzero deviation: in the
plane, key `e₀`, gates `α = 1/2`, `β = 1`, value `1`, and a carrier that adds `(1/10)·e₁` at
every step. The theorem then bounds the carried error by `1/5`. -/

/-- The unit key `e₀` of the plane. -/
noncomputable def e0 : EuclideanSpace ℝ (Fin 2) := EuclideanSpace.single 0 1

/-- The unit vector `e₁` of the plane, the witness carrier's deviation direction. -/
noncomputable def e1 : EuclideanSpace ℝ (Fin 2) := EuclideanSpace.single 1 1

/-- `‖e₀‖ = 1`. -/
theorem e0_norm : ‖e0‖ = 1 := by simp [e0]

/-- The exact column trajectory from `e₀` at `α = 1/2`, `β = 1`, `v = 1`. -/
noncomputable def exactTraj : ℕ → EuclideanSpace ℝ (Fin 2)
  | 0 => e0
  | t + 1 => deltaCol (1 / 2) 1 1 e0 (exactTraj t)

/-- The carried trajectory: the same step plus a deviation `(1/10)·e₁` at every step. -/
noncomputable def carriedTraj : ℕ → EuclideanSpace ℝ (Fin 2)
  | 0 => e0
  | t + 1 => deltaCol (1 / 2) 1 1 e0 (carriedTraj t) + (1 / 10 : ℝ) • e1

/-- The witness carrier's per-step deviation is `1/10`. -/
theorem carriedTraj_deviation (t : ℕ) :
    ‖carriedTraj (t + 1) - deltaCol (1 / 2) 1 1 e0 (carriedTraj t)‖ ≤ 1 / 10 := by
  simp [carriedTraj, e1, norm_smul]

/-- The joint witness: every hypothesis of `delta_carrier_error_uniform` holds at once, and the
carried trajectory stays within `1/5` of the exact one. -/
theorem witness_delta_carrier (t : ℕ) : ‖carriedTraj t - exactTraj t‖ ≤ 1 / 5 := by
  have h := delta_carrier_error_uniform (E := EuclideanSpace ℝ (Fin 2)) (1 / 2) (1 / 10)
    (by norm_num) (by norm_num) (by norm_num) (fun _ => 1 / 2) (fun _ => 1) (fun _ => 1)
    (fun _ => e0) (fun _ => ⟨by norm_num, le_rfl⟩) (fun _ => ⟨by norm_num, by norm_num⟩)
    (fun _ => e0_norm) exactTraj carriedTraj (fun _ => rfl) rfl carriedTraj_deviation t
  norm_num at h
  linarith

end Nous.Seq.DeltaCarrier
