import Apeiron.Trichotomy.Basic
import Apeiron.Trichotomy.NullBasis
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.LinearAlgebra.Matrix.Determinant.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.GroupWithZero.Units.Fintype
import Mathlib.Tactic

set_option linter.apeiron.notation false

/-!
# Face-typed quantizer: the two-plane code is the split-complex null basis

Face table rows served:

| code | face | algebra | statement |
|---|---|---|---|
| 2-plane sign code `r₁s₁ + r₂s₂`, `sᵢ ∈ {±1}` | hyperbolic, `σ = 1` | `R[x]/(x² − 1)`, null basis `upper 1`, `lower 1` | `two_plane_split_complex`, `two_plane_null_channel`, `null_basis_rate_eq`, `null_basis_diag_cross_zero` |
| product junction `x·y` | hyperbolic, `disc > 0` | 2×2 symmetric `!![0, 1/2; 1/2, 0]` | `bilinear_junction_hyperbolic` |

Canonical declarations used: `Apeiron.Trichotomy.upper`, `lower` (`Trichotomy/NullBasis.lean`),
`Apeiron.Trichotomy.rel` (`Trichotomy/Basic.lean`). Grade: `proved` (Lean kernel, closed axiom cone).
Scope: statements about real numbers and sign types; no statement about a trained weight matrix.
-/

noncomputable section

namespace Perf.FaceTyped

open Apeiron.Trichotomy

/-- **Null channels of the 2-plane cell.** With `z = (r₁, r₂)` at `σ = 1`: `upper 1 z = r₁ + r₂`,
`lower 1 z = r₁ − r₂`. Uses `Apeiron.Trichotomy.upper`, `lower` and `Real.sqrt_one`. -/
theorem null_channels_eq (r₁ r₂ : ℝ) :
    upper 1 (r₁, r₂) = r₁ + r₂ ∧ lower 1 (r₁, r₂) = r₁ - r₂ := by
  simp [upper, lower]

/-- **Statement 1: the 2-plane weight is a split-complex scalar.** For signs `s₁, s₂`,
`r₁s₁ + r₂s₂ = s₁ · (r₁ + r₂·j)` with `j = s₁s₂`, and `j` is a root of `Apeiron.Trichotomy.rel 1 = x² − 1`
(so `j² = 1`, `j = ±1`; `j ≠ ±1` as a generator is the algebra, `j = ±1` is its evaluation in `ℝ`). -/
theorem two_plane_split_complex {s₁ s₂ : ℝ} (h₁ : s₁ = 1 ∨ s₁ = -1) (h₂ : s₂ = 1 ∨ s₂ = -1)
    (r₁ r₂ : ℝ) :
    r₁ * s₁ + r₂ * s₂ = s₁ * (r₁ + r₂ * (s₁ * s₂)) ∧ (s₁ * s₂) ^ 2 = 1 ∧
      (rel 1).eval (s₁ * s₂) = 0 := by
  rcases h₁ with rfl | rfl <;> rcases h₂ with rfl | rfl <;> simp [rel] <;> ring

/-- **Statement 2: the weight is a sign times one null channel.** The channel is `upper 1 (r₁, r₂)` when
`s₁s₂ = 1` and `lower 1 (r₁, r₂)` otherwise; the sign is `s₁`. Uses `upper`, `lower`. -/
theorem two_plane_null_channel {s₁ s₂ : ℝ} (h₁ : s₁ = 1 ∨ s₁ = -1) (h₂ : s₂ = 1 ∨ s₂ = -1)
    (r₁ r₂ : ℝ) :
    r₁ * s₁ + r₂ * s₂ =
      s₁ * (if s₁ * s₂ = 1 then upper 1 (r₁, r₂) else lower 1 (r₁, r₂)) := by
  rcases h₁ with rfl | rfl <;> rcases h₂ with rfl | rfl <;> simp [upper, lower] <;> (try norm_num) <;> (try ring)

/-- **Corollary of statement 2: the value lies in `{±a, ±b}`** with `a = upper 1 (r₁, r₂)` and
`b = lower 1 (r₁, r₂)`. -/
theorem two_plane_value_mem {s₁ s₂ : ℝ} (h₁ : s₁ = 1 ∨ s₁ = -1) (h₂ : s₂ = 1 ∨ s₂ = -1)
    (r₁ r₂ : ℝ) :
    r₁ * s₁ + r₂ * s₂ = upper 1 (r₁, r₂) ∨ r₁ * s₁ + r₂ * s₂ = -upper 1 (r₁, r₂) ∨
      r₁ * s₁ + r₂ * s₂ = lower 1 (r₁, r₂) ∨ r₁ * s₁ + r₂ * s₂ = -lower 1 (r₁, r₂) := by
  rw [two_plane_null_channel h₁ h₂]
  rcases h₁ with rfl | rfl <;> split_ifs <;> simp

/-- The null-basis re-coding `(s₁, s₂) ↦ (s₁, s₁s₂)` on the sign type squared (`ℤˣ = {±1}`). -/
def nullBasisMap : ℤˣ × ℤˣ ≃ ℤˣ × ℤˣ where
  toFun x := (x.1, x.1 * x.2)
  invFun x := (x.1, x.1 * x.2)
  left_inv x := by
    obtain ⟨s, t⟩ := x
    simp [← _root_.mul_assoc, Int.units_mul_self]
  right_inv x := by
    obtain ⟨s, t⟩ := x
    simp [← _root_.mul_assoc, Int.units_mul_self]

/-- **Statement 3: the null-basis code has exactly the 2-plane code's bits.** `nullBasisMap` is an
equivalence of `{±1}²`, its own inverse, and the code has `2² = 4` words, i.e. two bits. -/
theorem null_basis_rate_eq :
    nullBasisMap.symm = nullBasisMap ∧ Function.Involutive nullBasisMap ∧
      Fintype.card (ℤˣ × ℤˣ) = 2 ^ 2 := by
  refine ⟨?_, ?_, ?_⟩
  · ext x <;> simp [nullBasisMap]
  · intro x
    obtain ⟨s, t⟩ := x
    simp [nullBasisMap, ← _root_.mul_assoc, Int.units_mul_self]
  · simp [Fintype.card_prod, Fintype.card_units_int]

section Quadratic

variable {ι : Type*} [Fintype ι]

/-- Cross coefficient term of the quadratic form: nonzero only where the channels of `p` and `q` differ. -/
def crossTerm (H : ι → ι → ℝ) (t : ι → ℝ) (m : ι → Bool) (p q : ι) : ℝ :=
  if m p ≠ m q then H p q * t p * t q else 0

/-- Coefficient of `a²`: pairs both on channel `a`. -/
def coefA (H : ι → ι → ℝ) (t : ι → ℝ) (m : ι → Bool) : ℝ :=
  ∑ p, ∑ q, if m p = true ∧ m q = true then H p q * t p * t q else 0

/-- Coefficient of `b²`: pairs both on channel `b`. -/
def coefB (H : ι → ι → ℝ) (t : ι → ℝ) (m : ι → Bool) : ℝ :=
  ∑ p, ∑ q, if m p = false ∧ m q = false then H p q * t p * t q else 0

/-- Coefficient of `a·b`: ordered pairs on different channels. -/
def coefC (H : ι → ι → ℝ) (t : ι → ℝ) (m : ι → Bool) : ℝ :=
  ∑ p, ∑ q, crossTerm H t m p q

/-- **Statement 4: the diagonal of `H` couples no channels.** Each coordinate `p` carries `w p = t p · X p`
with `X p = a` if `m p` and `b` otherwise. Then `Σ_{p,q} H p q · w p · w q = A·a² + B·b² + C·a·b`, where
`C` collects `crossTerm`, which vanishes at `p = q` (a coordinate has one channel), so the `a·b` coefficient
uses only off-diagonal entries of `H`; with `H` symmetric, `C = 2 · Σ_{m p, ¬m q} H p q t p t q`.
Holds for every real `t`, in particular `t p ∈ {±1}`. -/
theorem null_basis_diag_cross_zero (H : ι → ι → ℝ) (t : ι → ℝ) (m : ι → Bool) (a b : ℝ) :
    (∀ p, crossTerm H t m p p = 0) ∧
    ∑ p, ∑ q, H p q * (t p * (if m p then a else b)) * (t q * (if m q then a else b)) =
      coefA H t m * a ^ 2 + coefB H t m * b ^ 2 + coefC H t m * (a * b) := by
  refine ⟨fun p => by simp [crossTerm], ?_⟩
  have key : ∀ p q, H p q * (t p * (if m p then a else b)) * (t q * (if m q then a else b)) =
      (if m p = true ∧ m q = true then H p q * t p * t q else 0) * a ^ 2 +
      (if m p = false ∧ m q = false then H p q * t p * t q else 0) * b ^ 2 +
      crossTerm H t m p q * (a * b) := by
    intro p q
    cases hp : m p <;> cases hq : m q <;> simp [crossTerm, hp, hq] <;> ring
  simp only [key, Finset.sum_add_distrib, ← Finset.sum_mul, coefA, coefB, coefC]

/-- **Symmetric form of `coefC`.** For `H` symmetric, the `a·b` coefficient is twice the sum over pairs with
`p` on `a` and `q` on `b`. -/
theorem coefC_symm (H : ι → ι → ℝ) (hH : ∀ p q, H p q = H q p) (t : ι → ℝ) (m : ι → Bool) :
    coefC H t m =
      2 * ∑ p, ∑ q, if m p = true ∧ m q = false then H p q * t p * t q else 0 := by
  have split : ∀ p q, crossTerm H t m p q =
      (if m p = true ∧ m q = false then H p q * t p * t q else 0) +
      (if m q = true ∧ m p = false then H q p * t q * t p else 0) := by
    intro p q
    rw [hH q p]
    cases hp : m p <;> cases hq : m q <;> simp [crossTerm, hp, hq]
    all_goals ring
  have swap : ∑ p, ∑ q, (if m q = true ∧ m p = false then H q p * t q * t p else 0) =
      ∑ p, ∑ q, if m p = true ∧ m q = false then H p q * t p * t q else 0 :=
    Finset.sum_comm
  simp only [coefC, split, Finset.sum_add_distrib]
  rw [swap]
  ring

end Quadratic

/-- Local discriminant of a real 2×2 matrix, `tr²/4 − det`. Mirrors `Perf.CostCorrespondence.disc2`
(`Perf/CostCorrespondence.lean:536`); unified with it when the two files meet. -/
def disc2' (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ := M.trace ^ 2 / 4 - M.det

/-- The product form `x·y` as a symmetric 2×2 matrix. -/
def prodForm : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1 / 2; 1 / 2, 0]

/-- **Statement 5: the product junction is hyperbolic and gauge invariant.** `prodForm` is symmetric, its
quadratic form is `x·y`, `disc2' prodForm = 1/4 > 0` (hyperbolic face), and the gauge
`(x, y) ↦ (s·x, y/s)`, `s ≠ 0`, preserves `x·y`. -/
theorem bilinear_junction_hyperbolic :
    prodForm.IsSymm ∧ (∀ x y : ℝ, (![x, y] ⬝ᵥ prodForm.mulVec ![x, y]) = x * y) ∧
      disc2' prodForm = 1 / 4 ∧ 0 < disc2' prodForm ∧
      ∀ (s x y : ℝ), s ≠ 0 → (s * x) * (y / s) = x * y := by
  have hd : disc2' prodForm = 1 / 4 := by
    simp [disc2', prodForm, Matrix.trace_fin_two, Matrix.det_fin_two_of]
    norm_num
  refine ⟨?_, ?_, hd, by rw [hd]; norm_num, ?_⟩
  · ext i j
    fin_cases i <;> fin_cases j <;> simp [prodForm]
  · intro x y
    simp [prodForm, Matrix.mulVec, dotProduct, Fin.sum_univ_two]
    ring
  · intro s x y hs
    field_simp

/-! ## Joint witnesses -/

/-- `s₁ = 1, s₂ = −1, r₁ = 3, r₂ = 1`: `w = 2 = lower 1 (3, 1)`, `s₁s₂ = −1`. -/
example : (3 : ℝ) * 1 + 1 * (-1) = 1 * lower 1 (3, 1) := by
  have h := two_plane_null_channel (s₁ := 1) (s₂ := -1) (Or.inl rfl) (Or.inr rfl) 3 1
  rw [h]; norm_num

/-- Same bundle for statement 1. -/
example : (3 : ℝ) * 1 + 1 * (-1) = 1 * (3 + 1 * (1 * (-1))) ∧ ((1 : ℝ) * -1) ^ 2 = 1 :=
  (fun h => ⟨h.1, h.2.1⟩) (two_plane_split_complex (s₁ := 1) (s₂ := -1) (Or.inl rfl) (Or.inr rfl) 3 1)

/-- Statement 4 on `Fin 2` with `H = 1`, `t = 1`, `m = ![true, false]`: hypotheses are satisfiable and the
cross coefficient is `2`. -/
example : coefC (fun _ _ : Fin 2 => (1 : ℝ)) (fun _ => 1) ![true, false] = 2 := by
  simp [coefC, crossTerm, Fin.sum_univ_two]
  norm_num

/-- Statement 3 witness: `(1, −1) ↦ (1, −1)` and `(−1, −1) ↦ (−1, 1)`. -/
example : nullBasisMap (-1, -1) = (-1, 1) := by decide

end Perf.FaceTyped
