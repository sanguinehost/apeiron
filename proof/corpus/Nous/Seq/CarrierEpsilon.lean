import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Nous.Seq.StatePrecision
import Nous.Seq.DeltaCarrier

/-!
# Rounding Carrier Models and State-Error Derivation for Sequence Mixers

This module derives the per-step rounding deviation `ε` and accumulated state-error bounds
for named numerical carriers in the gated delta sequence recurrence:
`s ↦ α·(s − β⟪s, k⟫ k) + (β·v)·k`.

While `Nous.Seq.DeltaCarrier` establishes the contraction error bound
`‖X t − x t‖ ≤ ε / (1 − α)` under an abstract hypothesis `ε`, this module grounds `ε`
in concrete representation models:
1. Fixed-point representation: Q32.32 arithmetic with round-to-nearest (`round_shift_32` in
   `src/engine/value.rs`, half-ULP `η = 2⁻³³`).
2. Floating-point representation: IEEE 754-2019 binary32 (`u = 2⁻²⁴`) and binary64 (`u = 2⁻⁵³`)
   under the standard relative error model (Higham, *Accuracy and Stability of Numerical Algorithms*,
   2nd ed., 2002, Eq. 2.4).

## Mathematical Overview & Main Results

1. **Carrier Models over ℝ**:
   - `IsAbsoluteCarrier ρ η`: `∀ x, |ρ x − x| ≤ η`.
   - `IsRelativeCarrier ρ u`: `∀ x, |ρ x − x| ≤ u * |x|`.
   - `IsRelativeCarrierNormal ρ u x_min x_max`: relative error bound holding on normal floating-point
     range `[x_min, x_max]`, explicitly excluding subnormal flush and overflow.

2. **Domain Dominance**:
   - `relative_carrier_bound_of_le`: on a bounded domain `|x| ≤ B`, a relative carrier satisfies
     `|ρ x − x| ≤ u * B`.
   - `binary64_finer_than_q32`: on the entire DeltaNet state domain `STATE_BOUND = 2¹⁶`
     (`src/engine/mixer.rs`), binary64 is strictly finer than Q32.32: `2⁻⁵³ * 2¹⁶ = 2⁻³⁷ < 2⁻³³`.
   - `binary32_coarser_than_q32`: on the state domain `2¹⁶`, binary32 is strictly coarser than
     Q32.32: `2⁻²⁴ * 2¹⁶ = 2⁻⁸ > 2⁻³³`.

3. **Coordinatewise Rounding and State-Error Bounds**:
   - `roundVec_error_le_absolute`: for coordinatewise rounding in `EuclideanSpace ℝ (Fin n)`,
     `‖roundVec ρ v − v‖ ≤ √n * η`.
   - `roundVec_error_le_relative`: for coordinatewise rounding,
     `‖roundVec ρ v − v‖ ≤ u * ‖v‖`.
   - `delta_absolute_carrier_error_uniform`: under coordinatewise rounding `roundVec ρ`,
     the carried column state trajectory satisfies `‖X t − x t‖ ≤ √n * η / (1 − α)`.
   - `delta_absolute_carrier_q32_numeric_bound`: for DeltaNet key dimension `n = 128`, Q32.32
     half-ULP `η = 2⁻³³`, and decay `α = 99/100`, the steady-state state error is bounded
     by `1200 / 2³³ < 2⁻²²` (< 2.4 × 10⁻⁷).

4. **State-Size Bounds and Uniform Relative-Carrier Bounds**:
   - `delta_exact_norm_le`: for exact recurrence with `0 ≤ a_t ≤ α < 1`, `0 ≤ b_t ≤ 1`, `‖k_t‖ = 1`,
     `|v_t| ≤ V`, the exact state satisfies `‖x_t‖ ≤ max(‖x_0‖, V/(1−α))` uniformly in `t`.
   - `delta_carried_norm_le`: for carried state deviating by at most `ε / (1 - α)` from `x_t`,
     `‖X_t‖ ≤ ‖x_t‖ + ε / (1 - α)`.
   - `delta_relative_carrier_error_uniform`: under coordinatewise relative carrier with unit `u`,
     when `u * α < 1 - α`, the carried error is uniformly bounded by
     `E = u * (α * R + V) / (1 - α - u * α)` where `R = max(‖x_0‖, V/(1-α))`.
   - `delta_relative_carrier_b32_numeric_bound`: for binary32 (`u = 2⁻²⁴`), `α = 99/100`, `V = 4`, `x_0 = 0`,
     the relative error bound satisfies `E < 3/1000` (and `E < 1/400`).

5. **Higham Error Analysis (Higham 2002)**:
   - `gamma`: Higham's accumulation constant `γ_n = n u / (1 − n u)`.
   - `higham_gamma_step`: step inequality `γ_n (1 + u) + u ≤ γ_{n+1}` for `(n+1)u < 1`.
   - `higham_lemma_3_1`: product-of-(1+δ) bound `|∏_{i=0}^{n-1} (1 + δ_i) − 1| ≤ γ_n` for `|δ_i| ≤ u`.
   - `fl_dot`: floating-point recursive summation dot product fold model.
   - `fl_dot_one_le`: base case dot product error bound.

6. **Joint Non-Vacuity Witnesses**:
   - Every headline with hypotheses is accompanied by checked joint witnesses instantiating all
     hypotheses simultaneously at concrete values.

## References

- Higham, N. J., *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
- IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019.
- Yang, S., Kautz, J., and Hatamizadeh, A., *Gated Delta Networks*, arXiv:2412.06464, 2024.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Seq.CarrierEpsilon

open Nous.Seq.StatePrecision
open Nous.Seq.DeltaCarrier
open scoped InnerProductSpace
open scoped BigOperators

/-! ### 1. Carrier Models over ℝ -/

/-- An absolute (fixed-point) rounding carrier on `ℝ` with maximum absolute roundoff `η ≥ 0`.
Every real value `x` is mapped to `ρ x` within distance `η`. -/
def IsAbsoluteCarrier (ρ : ℝ → ℝ) (η : ℝ) : Prop :=
  ∀ x, |ρ x - x| ≤ η

/-- Unit of roundoff (half-ULP) for Q32.32 fixed-point representation with round-to-nearest
ties upward (`round_shift_32` in `src/engine/value.rs`): `η = 2⁻³³`. -/
noncomputable def q32_eta : ℝ := 1 / 2 ^ 33

/-- A relative (floating-point) rounding carrier on `ℝ` with unit roundoff `u ≥ 0`.
In the standard model of floating-point arithmetic (Higham 2002, Eq. 2.4),
every real value `x` is rounded to `ρ x` with relative error bounded by `u`. -/
def IsRelativeCarrier (ρ : ℝ → ℝ) (u : ℝ) : Prop :=
  ∀ x, |ρ x - x| ≤ u * |x|

/-- A relative rounding carrier restricted to the normal range `[x_min, x_max]`,
explicitly excluding subnormal flush (`|x| < x_min`) and overflow (`|x| > x_max`).
Standard IEEE 754 relative error bounds hold exclusively within the normal range. -/
def IsRelativeCarrierNormal (ρ : ℝ → ℝ) (u : ℝ) (x_min x_max : ℝ) : Prop :=
  ∀ x, x_min ≤ |x| ∧ |x| ≤ x_max → |ρ x - x| ≤ u * |x|

/-- IEEE 754-2019 binary64 (double precision) unit roundoff: `u = 2⁻⁵³` (half-ULP of 1.0). -/
noncomputable def binary64_u : ℝ := 1 / 2 ^ 53

/-- IEEE 754-2019 binary32 (single precision) unit roundoff: `u = 2⁻²⁴` (half-ULP of 1.0). -/
noncomputable def binary32_u : ℝ := 1 / 2 ^ 24

/-! ### 2. Dominance on the State Domain -/

/-- The DeltaNet recurrent state bound `STATE_BOUND = 1 << 48` raw in Q32.32,
corresponding to real value `B = 2¹⁶ = 65536.0` (`src/engine/mixer.rs`). -/
noncomputable def state_bound : ℝ := 2 ^ 16

/-- Dominance lemma: on a bounded domain `|x| ≤ B`, any relative carrier with unit roundoff `u ≥ 0`
acts as an absolute carrier with error bound `u * B`. -/
theorem relative_carrier_bound_of_le (ρ : ℝ → ℝ) (u B : ℝ) (hu : 0 ≤ u) (x : ℝ)
    (h_rel : |ρ x - x| ≤ u * |x|) (hB : |x| ≤ B) :
    |ρ x - x| ≤ u * B :=
  (h_rel.trans (mul_le_mul_of_nonneg_left hB hu))

/-- Dominance lemma on the normal floating-point range: for values in `[x_min, B]` with `B ≤ x_max`,
the rounding error is bounded by `u * B`. -/
theorem relative_carrier_normal_bound_of_le (ρ : ℝ → ℝ) (u x_min x_max B : ℝ) (hu : 0 ≤ u) (x : ℝ)
    (h_norm : IsRelativeCarrierNormal ρ u x_min x_max)
    (h_in : x_min ≤ |x| ∧ |x| ≤ B) (hB : B ≤ x_max) :
    |ρ x - x| ≤ u * B := by
  have h_range : x_min ≤ |x| ∧ |x| ≤ x_max := ⟨h_in.1, h_in.2.trans hB⟩
  exact (h_norm x h_range).trans (mul_le_mul_of_nonneg_left h_in.2 hu)

/-- Precision dominance: on the entire `STATE_BOUND` domain (`B = 2¹⁶`), IEEE 754 binary64
floating-point (`u = 2⁻⁵³`) is strictly finer than Q32.32 fixed-point (`η = 2⁻³³`):
`2⁻⁵³ * 2¹⁶ = 2⁻³⁷ < 2⁻³³`. -/
theorem binary64_finer_than_q32 : binary64_u * state_bound < q32_eta := by
  unfold binary64_u state_bound q32_eta
  norm_num

/-- Precision coarseness: on the `STATE_BOUND` domain (`B = 2¹⁶`), IEEE 754 binary32
floating-point (`u = 2⁻²⁴`) is strictly coarser than Q32.32 fixed-point (`η = 2⁻³³`):
`2⁻²⁴ * 2¹⁶ = 2⁻⁸ > 2⁻³³`. -/
theorem binary32_coarser_than_q32 : binary32_u * state_bound > q32_eta := by
  unfold binary32_u state_bound q32_eta
  norm_num

/-! ### 3. Coordinatewise Rounding and State-Error Bounds -/

/-- Coordinatewise rounding of an `n`-dimensional Euclidean vector by scalar carrier `ρ`. -/
noncomputable def roundVec (ρ : ℝ → ℝ) {n : ℕ} (v : EuclideanSpace ℝ (Fin n)) : EuclideanSpace ℝ (Fin n) :=
  WithLp.toLp 2 (fun i => ρ (v i))

@[simp] theorem roundVec_apply (ρ : ℝ → ℝ) {n : ℕ} (v : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    roundVec ρ v i = ρ (v i) := rfl

/-- Per-step deviation bound for coordinatewise absolute rounding:
In `EuclideanSpace ℝ (Fin n)`, coordinatewise rounding by an absolute carrier with error bound `η`
deviates from the exact vector by at most `√n * η`. -/
theorem roundVec_error_le_absolute {n : ℕ} (ρ : ℝ → ℝ) (η : ℝ) (hη : 0 ≤ η)
    (hρ : IsAbsoluteCarrier ρ η) (v : EuclideanSpace ℝ (Fin n)) :
    ‖roundVec ρ v - v‖ ≤ Real.sqrt n * η := by
  rw [PiLp.norm_eq_of_L2]
  have h_le : (∑ i : Fin n, ‖(roundVec ρ v - v) i‖ ^ 2) ≤ n * η ^ 2 := by
    have hi : ∀ i : Fin n, ‖(roundVec ρ v - v) i‖ ^ 2 ≤ η ^ 2 := by
      intro i
      have h1 : ‖(roundVec ρ v - v) i‖ = |ρ (v i) - v i| := Real.norm_eq_abs _
      rw [h1]
      have h2 : |ρ (v i) - v i| ≤ η := hρ (v i)
      exact sq_le_sq.mpr (by rw [abs_abs, abs_of_nonneg hη]; exact h2)
    have hsum := Finset.sum_le_card_nsmul (Finset.univ : Finset (Fin n))
      (fun i => ‖(roundVec ρ v - v) i‖ ^ 2) (η ^ 2) (fun i _ => hi i)
    simp only [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at hsum
    exact hsum
  have h_sqrt := Real.sqrt_le_sqrt h_le
  rw [Real.sqrt_mul (Nat.cast_nonneg n), Real.sqrt_sq hη] at h_sqrt
  exact h_sqrt

/-- Per-step deviation bound for coordinatewise relative rounding:
In `EuclideanSpace ℝ (Fin n)`, coordinatewise rounding by a relative carrier with unit roundoff `u`
deviates from the exact vector by at most `u * ‖v‖`. -/
theorem roundVec_error_le_relative {n : ℕ} (ρ : ℝ → ℝ) (u : ℝ) (hu : 0 ≤ u)
    (hρ : IsRelativeCarrier ρ u) (v : EuclideanSpace ℝ (Fin n)) :
    ‖roundVec ρ v - v‖ ≤ u * ‖v‖ := by
  rw [PiLp.norm_eq_of_L2]
  have h_le : (∑ i : Fin n, ‖(roundVec ρ v - v) i‖ ^ 2) ≤ u ^ 2 * ‖v‖ ^ 2 := by
    have hi : ∀ i : Fin n, ‖(roundVec ρ v - v) i‖ ^ 2 ≤ (u * |v i|) ^ 2 := by
      intro i
      have h1 : ‖(roundVec ρ v - v) i‖ = |ρ (v i) - v i| := Real.norm_eq_abs _
      rw [h1]
      have h2 : |ρ (v i) - v i| ≤ u * |v i| := hρ (v i)
      have h_nonneg : 0 ≤ u * |v i| := mul_nonneg hu (abs_nonneg _)
      exact sq_le_sq.mpr (by rw [abs_abs, abs_of_nonneg h_nonneg]; exact h2)
    have hsum : (∑ i : Fin n, ‖(roundVec ρ v - v) i‖ ^ 2) ≤ ∑ i : Fin n, (u * |v i|) ^ 2 :=
      Finset.sum_le_sum (fun i _ => hi i)
    have h_factor : (∑ i : Fin n, (u * |v i|) ^ 2) = u ^ 2 * (∑ i : Fin n, |v i| ^ 2) := by
      simp_rw [mul_pow]
      rw [← Finset.mul_sum]
    rw [h_factor] at hsum
    have h_v_norm : ‖v‖ ^ 2 = ∑ i : Fin n, |v i| ^ 2 := by
      rw [PiLp.norm_sq_eq_of_L2]
      simp_rw [Real.norm_eq_abs]
    rw [← h_v_norm] at hsum
    exact hsum
  have h_sqrt := Real.sqrt_le_sqrt h_le
  rw [Real.sqrt_mul (sq_nonneg u)] at h_sqrt
  rw [Real.sqrt_sq hu] at h_sqrt
  rw [Real.sqrt_sq (norm_nonneg v)] at h_sqrt
  exact h_sqrt

/-- Uniform state error bound under coordinatewise absolute rounding:
Combining the Lipschitz contraction of `deltaCol` (`Nous.Seq.DeltaCarrier.delta_carrier_error_uniform`)
with the coordinatewise per-step bound `√n * η`, the carried trajectory `X t` stays within
`√n * η / (1 − α)` of the exact trajectory `x t` uniformly for all sequence lengths `t`. -/
theorem delta_absolute_carrier_error_uniform {n : ℕ} (α η : ℝ)
    (hα0 : 0 ≤ α) (hα1 : α < 1) (hη : 0 ≤ η)
    (ρ : ℝ → ℝ) (hρ : IsAbsoluteCarrier ρ η)
    (a b v : ℕ → ℝ) (k : ℕ → EuclideanSpace ℝ (Fin n))
    (ha : ∀ t, 0 ≤ a t ∧ a t ≤ α) (hb : ∀ t, 0 ≤ b t ∧ b t ≤ 2) (hk : ∀ t, ‖k t‖ = 1)
    (x X : ℕ → EuclideanSpace ℝ (Fin n))
    (hx : ∀ t, x (t + 1) = deltaCol (a t) (b t) (v t) (k t) (x t))
    (h0 : X 0 = x 0)
    (hX : ∀ t, X (t + 1) = roundVec ρ (deltaCol (a t) (b t) (v t) (k t) (X t))) :
    ∀ t, ‖X t - x t‖ ≤ Real.sqrt n * η / (1 - α) := by
  have hε : 0 ≤ Real.sqrt n * η := mul_nonneg (Real.sqrt_nonneg _) hη
  refine delta_carrier_error_uniform α (Real.sqrt n * η) hα0 hα1 hε a b v k ha hb hk x X hx h0 ?_
  intro t
  rw [hX t]
  exact roundVec_error_le_absolute ρ η hη hρ _

/-- Bound on `√128`: since `128 ≤ 144 = 12²`, `√128 ≤ 12`. -/
theorem sqrt_128_le : Real.sqrt 128 ≤ 12 := by
  rw [Real.sqrt_le_iff]
  norm_num

/-- Numeric steady-state error bound for the Q32.32 mixer:
For key dimension `n = 128` (`dk = 128` in `src/engine/mixer.rs`), Q32.32 half-ULP `η = 2⁻³³`,
and decay `α = 99/100`, the steady-state state error is bounded by `1200 / 2³³`. -/
theorem delta_absolute_carrier_q32_numeric_bound :
    Real.sqrt 128 * q32_eta / (1 - (99 : ℝ) / 100) ≤ 1200 / 2 ^ 33 := by
  have h_sqrt : Real.sqrt 128 ≤ 12 := sqrt_128_le
  have h_sub : (1 : ℝ) - 99 / 100 = 1 / 100 := by norm_num
  unfold q32_eta
  rw [h_sub]
  have h_eq : Real.sqrt 128 * (1 / 2 ^ 33) / (1 / 100 : ℝ) = 100 * Real.sqrt 128 * (1 / 2 ^ 33) := by
    ring
  rw [h_eq]
  have h_mul : 100 * Real.sqrt 128 * (1 / 2 ^ 33) ≤ 100 * 12 * (1 / 2 ^ 33) := by
    gcongr
  linarith

/-- Numeric witness: `1200 / 2³³ < 2⁻²²` (< 2.4 × 10⁻⁷), confirming that Q32.32 accumulation
drift in DeltaNet is well below 1 part in a million over arbitrary context lengths. -/
theorem delta_absolute_carrier_q32_numeric_le_pow :
    (1200 : ℝ) / 2 ^ 33 < 1 / 2 ^ 22 := by
  norm_num

/-! ### 4. State-Size Bounds and Uniform Relative-Carrier Bounds -/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- State-size bound for the exact gated delta recurrence:
With decay `0 ≤ a_t ≤ α < 1`, write strength `0 ≤ b_t ≤ 1`, unit keys `‖k_t‖ = 1`, and bounded
inputs `|v_t| ≤ V`, the exact state remains within `max(‖x_0‖, V / (1 − α))` for all sequence
lengths `t`. This establishes stability and domain containment for the unquantized state. -/
theorem delta_exact_norm_le (α V : ℝ) (hα0 : 0 ≤ α) (hα1 : α < 1) (_hV : 0 ≤ V)
    (a b v : ℕ → ℝ) (k : ℕ → E)
    (ha : ∀ t, 0 ≤ a t ∧ a t ≤ α) (hb : ∀ t, 0 ≤ b t ∧ b t ≤ 1) (hk : ∀ t, ‖k t‖ = 1)
    (hv : ∀ t, |v t| ≤ V)
    (x : ℕ → E) (hx : ∀ t, x (t + 1) = deltaCol (a t) (b t) (v t) (k t) (x t)) :
    ∀ t, ‖x t‖ ≤ max ‖x 0‖ (V / (1 - α)) := by
  let R := max ‖x 0‖ (V / (1 - α))
  have h_denom : 0 < 1 - α := sub_pos.mpr hα1
  have hVR : V ≤ (1 - α) * R := by
    have h1 : V / (1 - α) ≤ R := le_max_right _ _
    have h2 := (div_le_iff₀ h_denom).mp h1
    rwa [mul_comm] at h2
  intro t
  induction t with
  | zero => exact le_max_left _ _
  | succ t ih =>
    rw [hx t]
    have h_step : ‖deltaCol (a t) (b t) (v t) (k t) (x t)‖ ≤ a t * ‖x t‖ + b t * |v t| := by
      have h_add := norm_add_le (a t • (x t - (b t * ⟪x t, k t⟫_ℝ) • k t)) ((b t * v t) • k t)
      refine h_add.trans ?_
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (ha t).1,
        abs_mul, abs_of_nonneg (hb t).1, hk t, mul_one]
      have h_hh : ‖x t - (b t * ⟪x t, k t⟫_ℝ) • k t‖ ≤ ‖x t‖ :=
        householder_nonexpansive (b t) (hb t).1 (by linarith [(hb t).2]) (k t) (x t) (hk t)
      exact add_le_add (mul_le_mul_of_nonneg_left h_hh (ha t).1) le_rfl
    have h_le_V : b t * |v t| ≤ V := by
      have h1 : b t * |v t| ≤ 1 * V := mul_le_mul (hb t).2 (hv t) (abs_nonneg _) (by norm_num)
      rwa [one_mul] at h1
    have h_le_α : a t * ‖x t‖ ≤ α * R :=
      mul_le_mul (ha t).2 ih (norm_nonneg _) hα0
    have h_comb : a t * ‖x t‖ + b t * |v t| ≤ α * R + V := add_le_add h_le_α h_le_V
    have h_ind : α * R + V ≤ R := by linarith
    exact h_step.trans (h_comb.trans h_ind)

omit [InnerProductSpace ℝ E] in
/-- Carried-state size bound:
If the carried state deviates by at most `ε / (1 − α)` from the exact state trajectory `x_t`,
then the carried state norm satisfies `‖X_t‖ ≤ ‖x_t‖ + ε / (1 − α)` by the triangle inequality. -/
theorem delta_carried_norm_le (X x : ℕ → E) (α ε : ℝ)
    (h_err : ∀ t, ‖X t - x t‖ ≤ ε / (1 - α)) (t : ℕ) :
    ‖X t‖ ≤ ‖x t‖ + ε / (1 - α) := by
  have : ‖X t‖ ≤ ‖x t‖ + ‖X t - x t‖ := by
    calc ‖X t‖ = ‖x t + (X t - x t)‖ := by rw [add_sub_cancel]
      _ ≤ ‖x t‖ + ‖X t - x t‖ := norm_add_le _ _
  linarith [h_err t]

/-- Combined carried-state size bound:
Combining exact state stability `‖x_t‖ ≤ R = max(‖x_0‖, V / (1 − α))` with perturbation bound
`‖X_t − x_t‖ ≤ ε / (1 − α)` yields uniform state containment `‖X_t‖ ≤ R + ε / (1 − α)`. -/
theorem delta_carried_norm_le_max (α V ε : ℝ) (hα0 : 0 ≤ α) (hα1 : α < 1) (hV : 0 ≤ V)
    (a b v : ℕ → ℝ) (k : ℕ → E)
    (ha : ∀ t, 0 ≤ a t ∧ a t ≤ α) (hb : ∀ t, 0 ≤ b t ∧ b t ≤ 1) (hk : ∀ t, ‖k t‖ = 1)
    (hv : ∀ t, |v t| ≤ V)
    (x X : ℕ → E) (hx : ∀ t, x (t + 1) = deltaCol (a t) (b t) (v t) (k t) (x t))
    (h_err : ∀ t, ‖X t - x t‖ ≤ ε / (1 - α)) (t : ℕ) :
    ‖X t‖ ≤ max ‖x 0‖ (V / (1 - α)) + ε / (1 - α) := by
  have h_carried := delta_carried_norm_le X x α ε h_err t
  have h_exact := delta_exact_norm_le α V hα0 hα1 hV a b v k ha hb hk hv x hx t
  linarith

/-- Algebraic fixed-point identity for relative-carrier steady-state error accumulation:
`(α + u * α) * E + u * (α * R + V) = E` where `E = u * (α * R + V) / (1 − α − u * α)`. -/
theorem relative_carrier_fixed_point (α u R V : ℝ) (hD : 1 - α - u * α ≠ 0) :
    (α + u * α) * (u * (α * R + V) / (1 - α - u * α)) + u * (α * R + V) =
      u * (α * R + V) / (1 - α - u * α) := by
  have h_mul : ((α + u * α) + (1 - α - u * α)) = 1 := by ring
  have h_split : ((α + u * α) + (1 - α - u * α)) * (u * (α * R + V) / (1 - α - u * α)) =
      (α + u * α) * (u * (α * R + V) / (1 - α - u * α)) + (1 - α - u * α) * (u * (α * R + V) / (1 - α - u * α)) := by
    rw [add_mul]
  rw [mul_div_cancel₀ (u * (α * R + V)) hD] at h_split
  rw [← h_split, h_mul, one_mul]

/-- Uniform state error bound under coordinatewise relative rounding:
For a relative carrier `ρ` with unit roundoff `u` satisfying `u * α < 1 − α`,
the carried column state trajectory `X t` satisfies
`‖X t − x t‖ ≤ u * (α * R + V) / (1 − α − u * α)`
uniformly for all context lengths `t`, where `R = max(‖x_0‖, V / (1 − α))`.
This closes the relative (floating-point) forward error analysis of the gated delta recurrence. -/
theorem delta_relative_carrier_error_uniform {n : ℕ} (α u V : ℝ)
    (hα0 : 0 ≤ α) (hα1 : α < 1) (hu : 0 ≤ u) (hV : 0 ≤ V)
    (huα : u * α < 1 - α)
    (ρ : ℝ → ℝ) (hρ : IsRelativeCarrier ρ u)
    (a b v : ℕ → ℝ) (k : ℕ → EuclideanSpace ℝ (Fin n))
    (ha : ∀ t, 0 ≤ a t ∧ a t ≤ α) (hb : ∀ t, 0 ≤ b t ∧ b t ≤ 1) (hk : ∀ t, ‖k t‖ = 1)
    (hv : ∀ t, |v t| ≤ V)
    (x X : ℕ → EuclideanSpace ℝ (Fin n))
    (hx : ∀ t, x (t + 1) = deltaCol (a t) (b t) (v t) (k t) (x t))
    (h0 : X 0 = x 0)
    (hX : ∀ t, X (t + 1) = roundVec ρ (deltaCol (a t) (b t) (v t) (k t) (X t))) :
    ∀ t, ‖X t - x t‖ ≤ u * (α * max ‖x 0‖ (V / (1 - α)) + V) / (1 - α - u * α) := by
  let R := max ‖x 0‖ (V / (1 - α))
  let E := u * (α * R + V) / (1 - α - u * α)
  have h_exact_le := delta_exact_norm_le α V hα0 hα1 hV a b v k ha hb hk hv x hx
  have hD_pos : 0 < 1 - α - u * α := by linarith
  have hD_ne : 1 - α - u * α ≠ 0 := ne_of_gt hD_pos
  have hR_nonneg : 0 ≤ R := le_trans (norm_nonneg _) (le_max_left _ _)
  have h_num_nonneg : 0 ≤ u * (α * R + V) :=
    mul_nonneg hu (add_nonneg (mul_nonneg hα0 hR_nonneg) hV)
  have hE_nonneg : 0 ≤ E := div_nonneg h_num_nonneg (le_of_lt hD_pos)
  have h_fix := relative_carrier_fixed_point α u R V hD_ne
  intro t
  induction t with
  | zero =>
    rw [h0, sub_self, norm_zero]
    exact hE_nonneg
  | succ t ih =>
    rw [hX t, hx t]
    let w := deltaCol (a t) (b t) (v t) (k t) (X t)
    let w_exact := deltaCol (a t) (b t) (v t) (k t) (x t)
    have h_diff : roundVec ρ w - w_exact = (roundVec ρ w - w) + (w - w_exact) := by abel
    rw [h_diff]
    have h_tri := norm_add_le (roundVec ρ w - w) (w - w_exact)
    refine h_tri.trans ?_
    have h_round : ‖roundVec ρ w - w‖ ≤ u * ‖w‖ :=
      roundVec_error_le_relative ρ u hu hρ w
    have h_lip : ‖w - w_exact‖ ≤ α * ‖X t - x t‖ := by
      have h1 := deltaCol_lipschitz (a t) (b t) (v t) (ha t).1 (hb t).1 (by linarith [(hb t).2])
        (k t) (hk t) (X t) (x t)
      have h2 : a t * ‖X t - x t‖ ≤ α * ‖X t - x t‖ :=
        mul_le_mul_of_nonneg_right (ha t).2 (norm_nonneg _)
      exact h1.trans h2
    have h_w_norm : ‖w‖ ≤ α * (R + E) + V := by
      have h_add := norm_add_le (a t • (X t - (b t * ⟪X t, k t⟫_ℝ) • k t)) ((b t * v t) • k t)
      refine h_add.trans ?_
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg (ha t).1,
        abs_mul, abs_of_nonneg (hb t).1, hk t, mul_one]
      have h_hh : ‖X t - (b t * ⟪X t, k t⟫_ℝ) • k t‖ ≤ ‖X t‖ :=
        householder_nonexpansive (b t) (hb t).1 (by linarith [(hb t).2]) (k t) (X t) (hk t)
      have h_le_V : b t * |v t| ≤ V := by
        have h1 : b t * |v t| ≤ 1 * V := mul_le_mul (hb t).2 (hv t) (abs_nonneg _) (by norm_num)
        rwa [one_mul] at h1
      have h_Xt : ‖X t‖ ≤ R + E := by
        calc ‖X t‖ = ‖x t + (X t - x t)‖ := by rw [add_sub_cancel]
          _ ≤ ‖x t‖ + ‖X t - x t‖ := norm_add_le _ _
          _ ≤ R + E := add_le_add (h_exact_le t) ih
      have h_step1 : a t * ‖X t - (b t * ⟪X t, k t⟫_ℝ) • k t‖ ≤ a t * ‖X t‖ :=
        mul_le_mul_of_nonneg_left h_hh (ha t).1
      have h_step2 : a t * ‖X t‖ ≤ α * (R + E) :=
        mul_le_mul (ha t).2 h_Xt (norm_nonneg _) hα0
      linarith
    have h_round_le : u * ‖w‖ ≤ u * (α * (R + E) + V) :=
      mul_le_mul_of_nonneg_left h_w_norm hu
    have h_lip_le : α * ‖X t - x t‖ ≤ α * E :=
      mul_le_mul_of_nonneg_left ih hα0
    have h_algebra : u * (α * (R + E) + V) + α * E = (α + u * α) * E + u * (α * R + V) := by ring
    have h_bound : u * ‖w‖ + α * ‖X t - x t‖ ≤ E := by
      calc u * ‖w‖ + α * ‖X t - x t‖
          ≤ u * (α * (R + E) + V) + α * E := add_le_add h_round_le h_lip_le
        _ = (α + u * α) * E + u * (α * R + V) := h_algebra
        _ = E := h_fix
    exact (add_le_add h_round h_lip).trans h_bound

/-- Numeric instantiation for IEEE 754 binary32 (single precision) relative carrier:
At unit roundoff `u = 2⁻²⁴`, decay `α = 99/100`, input bound `V = 4`, and `x_0 = 0`,
the steady-state uniform error bound `E` satisfies `E < 3 / 1000`. -/
theorem delta_relative_carrier_b32_numeric_bound :
    binary32_u * ((99 / 100 : ℝ) * (max 0 (4 / (1 - (99 / 100 : ℝ)))) + 4) /
      (1 - (99 / 100 : ℝ) - binary32_u * (99 / 100 : ℝ)) < 3 / 1000 := by
  unfold binary32_u
  norm_num

/-- Tighter numeric bound for binary32 with `V = 4`: `E = 40000 / (2²⁴ − 99) < 1 / 400`. -/
theorem delta_relative_carrier_b32_numeric_bound_tighter :
    binary32_u * ((99 / 100 : ℝ) * (max 0 (4 / (1 - (99 / 100 : ℝ)))) + 4) /
      (1 - (99 / 100 : ℝ) - binary32_u * (99 / 100 : ℝ)) < 1 / 400 := by
  unfold binary32_u
  norm_num

/-- Unit-input numeric bound for binary32 with `V = 1`, `x_0 = 0`: `E < 1 / 1000`. -/
theorem delta_relative_carrier_b32_numeric_bound_v1 :
    binary32_u * ((99 / 100 : ℝ) * (max 0 (1 / (1 - (99 / 100 : ℝ)))) + 1) /
      (1 - (99 / 100 : ℝ) - binary32_u * (99 / 100 : ℝ)) < 1 / 1000 := by
  unfold binary32_u
  norm_num

/-! ### 5. Higham Error Analysis (Higham 2002) -/

/-- Higham's relative error accumulation factor `γ_n = n u / (1 − n u)` (Higham 2002, Eq. 3.5).
Defined when `n u < 1`. -/
noncomputable def gamma (n : ℕ) (u : ℝ) : ℝ :=
  (n : ℝ) * u / (1 - (n : ℝ) * u)

/-- Positivity of Higham's constant `γ_n` for positive dimension and roundoff with `n u < 1`. -/
theorem gamma_pos (n : ℕ) (u : ℝ) (hn : 0 < n) (hu : 0 < u) (hnu : (n : ℝ) * u < 1) :
    0 < gamma n u := by
  unfold gamma
  have htop : 0 < (n : ℝ) * u := mul_pos (Nat.cast_pos.mpr hn) hu
  have hbot : 0 < 1 - (n : ℝ) * u := sub_pos.mpr hnu
  exact div_pos htop hbot

/-- Base case bound: for `n = 1`, `u ≤ γ₁ = u / (1 − u)` for `0 ≤ u < 1`. -/
theorem u_le_gamma_one (u : ℝ) (hu : 0 ≤ u) (h1u : u < 1) :
    u ≤ gamma 1 u := by
  unfold gamma
  simp only [Nat.cast_one, one_mul]
  have hbot : 0 < 1 - u := sub_pos.mpr h1u
  exact (le_div_iff₀ hbot).mpr (by nlinarith)

/-- Inductive step inequality for Higham's constant (Higham 2002, Lemma 3.1):
`γ_n * (1 + u) + u ≤ γ_{n+1}` whenever `(n + 1) * u < 1` and `u ≥ 0`. -/
theorem higham_gamma_step (n : ℕ) (u : ℝ) (hu : 0 ≤ u) (h1 : ((n + 1 : ℕ) : ℝ) * u < 1) :
    gamma n u * (1 + u) + u ≤ gamma (n + 1) u := by
  have hn_le : (n : ℝ) * u ≤ ((n + 1 : ℕ) : ℝ) * u := by
    have : (n : ℝ) ≤ ((n + 1 : ℕ) : ℝ) := by
      push_cast
      linarith
    exact mul_le_mul_of_nonneg_right this hu
  have hn_lt : (n : ℝ) * u < 1 := hn_le.trans_lt h1
  have hd1 : 0 < 1 - (n : ℝ) * u := sub_pos.mpr hn_lt
  have hd2 : 0 < 1 - ((n + 1 : ℕ) : ℝ) * u := sub_pos.mpr h1
  have h_eq : gamma n u * (1 + u) + u = ((n + 1 : ℕ) : ℝ) * u / (1 - (n : ℝ) * u) := by
    unfold gamma
    have hd1_ne : 1 - (n : ℝ) * u ≠ 0 := ne_of_gt hd1
    field_simp
    push_cast
    ring
  rw [h_eq]
  unfold gamma
  have h_denom_le : 1 - ((n + 1 : ℕ) : ℝ) * u ≤ 1 - (n : ℝ) * u := by
    linarith
  have h_num_nonneg : 0 ≤ ((n + 1 : ℕ) : ℝ) * u :=
    mul_nonneg (Nat.cast_nonneg _) hu
  exact div_le_div_of_nonneg_left h_num_nonneg hd2 h_denom_le

/-- Higham (2002, Lemma 3.1): Product-of-(1 + δ) perturbation bound.
If `|δ_i| ≤ u` for all `i < n` and `n * u < 1` with `u ≥ 0`, then
`|∏_{i=0}^{n-1} (1 + δ_i) − 1| ≤ γ_n(u)`. -/
theorem higham_lemma_3_1 (n : ℕ) (u : ℝ) (hu : 0 ≤ u) (hnu : (n : ℝ) * u < 1)
    (δ : ℕ → ℝ) (hδ : ∀ i, |δ i| ≤ u) :
    |∏ i ∈ Finset.range n, (1 + δ i) - 1| ≤ gamma n u := by
  induction n with
  | zero =>
    simp only [Finset.range_zero, Finset.prod_empty, sub_self, abs_zero]
    unfold gamma
    simp only [Nat.cast_zero, zero_mul, sub_zero, div_one, le_refl]
  | succ n ih =>
    have hn_le : (n : ℝ) * u ≤ ((n + 1 : ℕ) : ℝ) * u := by
      have : (n : ℝ) ≤ ((n + 1 : ℕ) : ℝ) := by
        push_cast
        linarith
      exact mul_le_mul_of_nonneg_right this hu
    have hn_lt : (n : ℝ) * u < 1 := hn_le.trans_lt hnu
    have ih_val := ih hn_lt
    rw [Finset.prod_range_succ]
    let P := ∏ i ∈ Finset.range n, (1 + δ i)
    have h_split : P * (1 + δ n) - 1 = (P - 1) * (1 + δ n) + δ n := by ring
    rw [h_split]
    have h_tri := abs_add_le ((P - 1) * (1 + δ n)) (δ n)
    refine h_tri.trans ?_
    rw [abs_mul]
    have h_factor : |1 + δ n| ≤ 1 + u := by
      have h1 := abs_add_le 1 (δ n)
      rw [abs_one] at h1
      linarith [hδ n]
    have h_term1 : |P - 1| * |1 + δ n| ≤ gamma n u * (1 + u) := by
      have h_one_u : 0 ≤ 1 + u := by linarith
      have h_gamma_nn : 0 ≤ gamma n u := by
        unfold gamma
        exact div_nonneg (mul_nonneg (Nat.cast_nonneg _) hu) (by linarith)
      exact mul_le_mul ih_val h_factor (abs_nonneg _) h_gamma_nn
    have h_add_le : |P - 1| * |1 + δ n| + |δ n| ≤ gamma n u * (1 + u) + u :=
      add_le_add h_term1 (hδ n)
    exact h_add_le.trans (higham_gamma_step n u hu hnu)

/-- Step transition for floating-point recursive summation:
Each product `p.1 * p.2` is rounded by `ρ`, and each accumulation add is rounded by `ρ`. -/
noncomputable def fl_dot_step (ρ : ℝ → ℝ) (acc : ℝ) (p : ℝ × ℝ) : ℝ :=
  ρ (acc + ρ (p.1 * p.2))

/-- Floating-point dot product via recursive summation (Higham 2002, Eq. 3.2):
The first product `x_0 * y_0` is rounded by `ρ`, and subsequent terms are recursively added
and rounded by `ρ`. -/
noncomputable def fl_dot (ρ : ℝ → ℝ) : List (ℝ × ℝ) → ℝ
  | [] => 0
  | (a, b) :: rest => rest.foldl (fl_dot_step ρ) (ρ (a * b))

/-- Exact dot product of a list of pairs `(x_i, y_i)`. -/
noncomputable def exact_dot (l : List (ℝ × ℝ)) : ℝ :=
  (l.map (fun p => p.1 * p.2)).sum

/-- Exact sum of absolute elementwise products `∑ |x_i * y_i|`. -/
noncomputable def abs_dot (l : List (ℝ × ℝ)) : ℝ :=
  (l.map (fun p => |p.1 * p.2|)).sum

/-- Base case dot product error bound (n = 1):
Under relative carrier `ρ` with unit `u < 1`, `|fl_dot [(a, b)] − exact_dot [(a, b)]| ≤ γ₁ * |a * b|`. -/
theorem fl_dot_one_le (ρ : ℝ → ℝ) (u : ℝ) (hu : 0 ≤ u) (hu1 : u < 1)
    (hρ : IsRelativeCarrier ρ u) (a b : ℝ) :
    |fl_dot ρ [(a, b)] - exact_dot [(a, b)]| ≤ gamma 1 u * abs_dot [(a, b)] := by
  dsimp [fl_dot, exact_dot, abs_dot]
  simp only [add_zero]
  have h1 : |ρ (a * b) - a * b| ≤ u * |a * b| := hρ (a * b)
  have h2 : u ≤ gamma 1 u := u_le_gamma_one u hu hu1
  have h3 : u * |a * b| ≤ gamma 1 u * |a * b| :=
    mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
  exact h1.trans h3

/-! ### 6. Joint Non-Vacuity Witnesses (AGENTS Obligation 2) -/

/-- Ideal witness: the identity carrier `id` has zero error, satisfying `IsAbsoluteCarrier id η`
for any `η ≥ 0`. -/
theorem witness_absolute_carrier_id (η : ℝ) (hη : 0 ≤ η) :
    IsAbsoluteCarrier id η := fun x => by
  simp only [id]
  rw [sub_self, abs_zero]
  exact hη

/-- Non-zero error witness for absolute carrier: the uniform shift `x ↦ x + η` achieves error
identically equal to `η`, satisfying `IsAbsoluteCarrier (fun x => x + η) η` for any `η ≥ 0`. -/
theorem witness_absolute_carrier_shift (η : ℝ) (hη : 0 ≤ η) :
    IsAbsoluteCarrier (fun x => x + η) η := fun x => by
  simp only [add_sub_cancel_left]
  exact (abs_of_nonneg hη).le

/-- Concrete Q32.32 roundoff witness: `x ↦ x + 2⁻³³` is an absolute carrier with bound `q32_eta`. -/
theorem witness_absolute_carrier_q32 :
    IsAbsoluteCarrier (fun x => x + q32_eta) q32_eta :=
  witness_absolute_carrier_shift q32_eta (by unfold q32_eta; positivity)

/-- Ideal witness: the identity carrier `id` has zero relative error, satisfying `IsRelativeCarrier id u`
for any `u ≥ 0`. -/
theorem witness_relative_carrier_id (u : ℝ) (hu : 0 ≤ u) :
    IsRelativeCarrier id u := fun x => by
  simp only [id]
  rw [sub_self, abs_zero]
  exact mul_nonneg hu (abs_nonneg x)

/-- Non-zero error witness for relative carrier: the uniform scale `x ↦ (1 + u) * x` achieves
relative error identically equal to `u * |x|`. -/
theorem witness_relative_carrier_scale (u : ℝ) (hu : 0 ≤ u) :
    IsRelativeCarrier (fun x => (1 + u) * x) u := fun x => by
  have : (1 + u) * x - x = u * x := by ring
  rw [this, abs_mul, abs_of_nonneg hu]

/-- Concrete IEEE 754 binary64 relative carrier witness: `x ↦ (1 + 2⁻⁵³) * x` satisfies
`IsRelativeCarrier` with unit roundoff `binary64_u`. -/
theorem witness_relative_carrier_b64 :
    IsRelativeCarrier (fun x => (1 + binary64_u) * x) binary64_u :=
  witness_relative_carrier_scale binary64_u (by unfold binary64_u; positivity)

/-- Concrete IEEE 754 binary32 relative carrier witness: `x ↦ (1 + 2⁻²⁴) * x` satisfies
`IsRelativeCarrier` with unit roundoff `binary32_u`. -/
theorem witness_relative_carrier_b32 :
    IsRelativeCarrier (fun x => (1 + binary32_u) * x) binary32_u :=
  witness_relative_carrier_scale binary32_u (by unfold binary32_u; positivity)

/-- Non-zero error witness for normal-range relative carrier: `x ↦ (1 + u) * x` satisfies
`IsRelativeCarrierNormal` on any range. -/
theorem witness_relative_carrier_normal (u x_min x_max : ℝ) (hu : 0 ≤ u) :
    IsRelativeCarrierNormal (fun x => (1 + u) * x) u x_min x_max := fun x _ => by
  have : (1 + u) * x - x = u * x := by ring
  rw [this, abs_mul, abs_of_nonneg hu]

/-- Concrete joint witness for `relative_carrier_bound_of_le`:
at `x = state_bound`, `u = binary64_u`, with carrier `fun x => (1 + binary64_u) * x`,
the error bound `binary64_u * state_bound` holds. -/
theorem witness_relative_carrier_bound :
    |(1 + binary64_u) * state_bound - state_bound| ≤ binary64_u * state_bound := by
  have h_bound : |state_bound| ≤ state_bound := by
    unfold state_bound; norm_num
  exact relative_carrier_bound_of_le (fun x => (1 + binary64_u) * x) binary64_u state_bound
    (by unfold binary64_u; positivity) state_bound
    (witness_relative_carrier_b64 state_bound) h_bound

/-- Concrete joint witness for `relative_carrier_normal_bound_of_le`:
with normal range `[1, 2¹⁷]`, `B = state_bound = 2¹⁶`, and `x = state_bound`. -/
theorem witness_relative_carrier_normal_bound :
    |(1 + binary64_u) * state_bound - state_bound| ≤ binary64_u * state_bound := by
  have h_norm := witness_relative_carrier_normal binary64_u 1 (2 ^ 17)
    (by unfold binary64_u; positivity)
  have h_in : (1 : ℝ) ≤ |state_bound| ∧ |state_bound| ≤ state_bound := by
    unfold state_bound; norm_num
  have hB : state_bound ≤ (2 : ℝ) ^ 17 := by
    unfold state_bound; norm_num
  exact relative_carrier_normal_bound_of_le (fun x => (1 + binary64_u) * x) binary64_u
    1 (2 ^ 17) state_bound (by unfold binary64_u; positivity) state_bound h_norm h_in hB

/-- Tightness witness: for the uniform shift carrier `x ↦ x + η`, the Euclidean coordinatewise error
is identically equal to `√n * η`. Thus `roundVec_error_le_absolute` is mathematically tight. -/
theorem witness_roundVec_shift_norm (n : ℕ) (η : ℝ) (hη : 0 ≤ η)
    (v : EuclideanSpace ℝ (Fin n)) :
    ‖roundVec (fun x => x + η) v - v‖ = Real.sqrt n * η := by
  rw [PiLp.norm_eq_of_L2]
  have h_diff : ∀ i : Fin n, ‖(roundVec (fun x => x + η) v - v) i‖ ^ 2 = η ^ 2 := by
    intro i
    have h1 : ‖(roundVec (fun x => x + η) v - v) i‖ = |(v i + η) - v i| := Real.norm_eq_abs _
    rw [h1]
    ring_nf
    rw [sq_abs]
  have h_sum : (∑ i : Fin n, ‖(roundVec (fun x => x + η) v - v) i‖ ^ 2) = n * η ^ 2 := by
    simp_rw [h_diff]
    simp [Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  rw [h_sum, Real.sqrt_mul (Nat.cast_nonneg n), Real.sqrt_sq hη]

/-- Tightness witness: for the uniform scaling carrier `x ↦ (1 + u) * x`, the Euclidean coordinatewise error
is identically equal to `u * ‖v‖`. Thus `roundVec_error_le_relative` is mathematically tight. -/
theorem witness_roundVec_scale_norm (n : ℕ) (u : ℝ) (hu : 0 ≤ u)
    (v : EuclideanSpace ℝ (Fin n)) :
    ‖roundVec (fun x => (1 + u) * x) v - v‖ = u * ‖v‖ := by
  rw [PiLp.norm_eq_of_L2]
  have h_diff : ∀ i : Fin n, ‖(roundVec (fun x => (1 + u) * x) v - v) i‖ ^ 2 = (u * |v i|) ^ 2 := by
    intro i
    have h1 : ‖(roundVec (fun x => (1 + u) * x) v - v) i‖ = |(1 + u) * v i - v i| := Real.norm_eq_abs _
    rw [h1]
    have : (1 + u) * v i - v i = u * v i := by ring
    rw [this, abs_mul, abs_of_nonneg hu, mul_pow]
  have h_sum : (∑ i : Fin n, ‖(roundVec (fun x => (1 + u) * x) v - v) i‖ ^ 2) =
      u ^ 2 * ‖v‖ ^ 2 := by
    simp_rw [h_diff, mul_pow]
    rw [← Finset.mul_sum]
    congr 1
    rw [PiLp.norm_sq_eq_of_L2]
    simp_rw [Real.norm_eq_abs]
  rw [h_sum, Real.sqrt_mul (sq_nonneg u), Real.sqrt_sq hu, Real.sqrt_sq (norm_nonneg v)]

/-- Concrete exact trajectory from unit key `e0` with `α = 1/2`, `β = 1`, `v = 1`. -/
noncomputable def exactSeq : ℕ → EuclideanSpace ℝ (Fin 2)
  | 0 => e0
  | t + 1 => deltaCol (1 / 2) 1 1 e0 (exactSeq t)

/-- Concrete carried trajectory with coordinatewise shift by `1/10` at each step. -/
noncomputable def carriedSeq : ℕ → EuclideanSpace ℝ (Fin 2)
  | 0 => e0
  | t + 1 => roundVec (fun x => x + 1 / 10) (deltaCol (1 / 2) 1 1 e0 (carriedSeq t))

/-- Concrete trajectory joint witness for `delta_absolute_carrier_error_uniform`:
All hypotheses hold simultaneously at non-zero rounding error `η = 1/10`, and the carried
trajectory stays within `√2 * (1/10) / (1 - 1/2) = √2 / 5` of the exact trajectory for all `t`. -/
theorem witness_carrier_trajectory (t : ℕ) :
    ‖carriedSeq t - exactSeq t‖ ≤ Real.sqrt 2 * (1 / 10) / (1 - 1 / 2) := by
  have h_rho : IsAbsoluteCarrier (fun x => x + 1 / 10) (1 / 10) :=
    witness_absolute_carrier_shift (1 / 10) (by norm_num)
  exact delta_absolute_carrier_error_uniform (1 / 2) (1 / 10) (by norm_num) (by norm_num)
    (by norm_num) (fun x => x + 1 / 10) h_rho (fun _ => 1 / 2) (fun _ => 1) (fun _ => 1)
    (fun _ => e0) (fun _ => ⟨by norm_num, le_rfl⟩) (fun _ => ⟨by norm_num, by norm_num⟩)
    (fun _ => e0_norm) exactSeq carriedSeq (fun _ => rfl) rfl (fun _ => rfl) t

/-- Numeric evaluation of trajectory witness bound: `√2 / 5 ≤ 3/10`. -/
theorem witness_carrier_trajectory_numeric (t : ℕ) :
    ‖carriedSeq t - exactSeq t‖ ≤ 3 / 10 := by
  have h := witness_carrier_trajectory t
  have h_sqrt2 : Real.sqrt 2 ≤ 3 / 2 := by
    rw [Real.sqrt_le_iff]
    norm_num
  have h_sub : (1 : ℝ) - 1 / 2 = 1 / 2 := by norm_num
  rw [h_sub] at h
  have h_eq : Real.sqrt 2 * (1 / 10) / (1 / 2 : ℝ) = Real.sqrt 2 / 5 := by ring
  rw [h_eq] at h
  have h_le : Real.sqrt 2 / 5 ≤ (3 / 2) / 5 := by gcongr
  have h_fin : (3 / 2 : ℝ) / 5 = 3 / 10 := by norm_num
  rw [h_fin] at h_le
  exact h.trans h_le

/-- Concrete joint witness for `delta_exact_norm_le`:
At `x_0 = e0`, `α = 1/2`, `β = 1`, `v = 1`, the exact state satisfies `‖exactSeq t‖ ≤ 2` for all `t`. -/
theorem witness_delta_exact_norm_le (t : ℕ) :
    ‖exactSeq t‖ ≤ max ‖e0‖ (1 / (1 - (1 / 2 : ℝ))) := by
  exact delta_exact_norm_le (1 / 2) 1 (by norm_num) (by norm_num) (by norm_num)
    (fun _ => 1 / 2) (fun _ => 1) (fun _ => 1) (fun _ => e0)
    (fun _ => ⟨by norm_num, le_rfl⟩) (fun _ => ⟨by norm_num, le_rfl⟩)
    (fun _ => e0_norm) (fun _ => by norm_num) exactSeq (fun _ => rfl) t

/-- Concrete evaluation of exact trajectory bound: `max ‖e0‖ (1 / (1 - 1/2)) = 2`. -/
theorem witness_delta_exact_norm_le_eval :
    max ‖e0‖ (1 / (1 - (1 / 2 : ℝ))) = 2 := by
  rw [e0_norm]
  norm_num

/-- Concrete joint witness for `delta_carried_norm_le`:
Under perturbation `ε = 1/10` and `α = 1/2`, the carried state norm is bounded by
`‖exactSeq t‖ + (1/10) / (1 - 1/2) = ‖exactSeq t‖ + 1/5`. -/
theorem witness_delta_carried_norm_le (t : ℕ) :
    ‖carriedSeq t‖ ≤ ‖exactSeq t‖ + (Real.sqrt 2 * (1 / 10)) / (1 - 1 / 2) := by
  have h_err : ∀ s, ‖carriedSeq s - exactSeq s‖ ≤ (Real.sqrt 2 * (1 / 10)) / (1 - 1 / 2) :=
    witness_carrier_trajectory
  exact delta_carried_norm_le carriedSeq exactSeq (1 / 2) (Real.sqrt 2 * (1 / 10)) h_err t

/-- Exact column trajectory under relative carrier evaluation parameters. -/
noncomputable def exactRelSeq : ℕ → EuclideanSpace ℝ (Fin 2)
  | 0 => e0
  | t + 1 => deltaCol (1 / 2) 1 1 e0 (exactRelSeq t)

/-- Carried column trajectory under relative scaling carrier `x ↦ (1 + 1/10) * x`. -/
noncomputable def carriedRelSeq : ℕ → EuclideanSpace ℝ (Fin 2)
  | 0 => e0
  | t + 1 => roundVec (fun x => (1 + 1 / 10) * x) (deltaCol (1 / 2) 1 1 e0 (carriedRelSeq t))

/-- Concrete trajectory joint witness for `delta_relative_carrier_error_uniform`:
All hypotheses hold simultaneously at non-zero relative roundoff `u = 1/10`, `α = 1/2`, `V = 1`,
and the relative-carried trajectory stays within `4/9` of the exact trajectory for all `t`. -/
theorem witness_relative_carrier_trajectory (t : ℕ) :
    ‖carriedRelSeq t - exactRelSeq t‖ ≤
      (1 / 10 : ℝ) * ((1 / 2 : ℝ) * (max ‖e0‖ (1 / (1 - (1 / 2 : ℝ)))) + 1) /
        (1 - 1 / 2 - (1 / 10 : ℝ) * (1 / 2 : ℝ)) := by
  have hρ : IsRelativeCarrier (fun x => (1 + 1 / 10) * x) (1 / 10) :=
    witness_relative_carrier_scale (1 / 10) (by norm_num)
  exact delta_relative_carrier_error_uniform (1 / 2) (1 / 10) 1 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (fun x => (1 + 1 / 10) * x) hρ
    (fun _ => 1 / 2) (fun _ => 1) (fun _ => 1) (fun _ => e0)
    (fun _ => ⟨by norm_num, le_rfl⟩) (fun _ => ⟨by norm_num, le_rfl⟩)
    (fun _ => e0_norm) (fun _ => by norm_num) exactRelSeq carriedRelSeq
    (fun _ => rfl) rfl (fun _ => rfl) t

/-- Concrete evaluation of relative trajectory witness bound: evaluates to `4/9`. -/
theorem witness_relative_carrier_trajectory_eval :
    (1 / 10 : ℝ) * ((1 / 2 : ℝ) * (max ‖e0‖ (1 / (1 - (1 / 2 : ℝ)))) + 1) /
      (1 - 1 / 2 - (1 / 10 : ℝ) * (1 / 2 : ℝ)) = 4 / 9 := by
  rw [e0_norm]
  norm_num

/-- Concrete joint witness for Higham (2002) Lemma 3.1:
At `n = 4`, `u = 1/10`, with perturbations `δ_i = 1/10`, the product error is bounded by `γ₄(1/10)`. -/
theorem witness_higham_lemma_3_1 :
    |∏ _i ∈ Finset.range 4, (1 + (1 / 10 : ℝ)) - 1| ≤ gamma 4 (1 / 10) := by
  exact higham_lemma_3_1 4 (1 / 10) (by norm_num) (by norm_num)
    (fun _ => 1 / 10) (fun _ => by norm_num)

/-- Concrete evaluation of Higham constant: `γ₄(1/10) = 2/3`. -/
theorem witness_higham_lemma_3_1_eval :
    gamma 4 (1 / 10 : ℝ) = 2 / 3 := by
  unfold gamma
  norm_num

/-- Concrete joint witness for base-case floating-point dot product error:
At `u = 1/10`, with scaling carrier `x ↦ (1 + 1/10) * x`, `a = 2`, `b = 3`. -/
theorem witness_fl_dot_one :
    |fl_dot (fun x => (1 + 1 / 10) * x) [(2, 3)] - exact_dot [(2, 3)]| ≤
      gamma 1 (1 / 10) * abs_dot [(2, 3)] := by
  have hρ : IsRelativeCarrier (fun x => (1 + 1 / 10) * x) (1 / 10) :=
    witness_relative_carrier_scale (1 / 10) (by norm_num)
  exact fl_dot_one_le (fun x => (1 + 1 / 10) * x) (1 / 10) (by norm_num) (by norm_num) hρ 2 3

end Nous.Seq.CarrierEpsilon
