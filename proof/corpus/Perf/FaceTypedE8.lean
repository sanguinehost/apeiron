import Perf.FaceTyped

/-!
# Face-typed E8 code: the half-cube spinor pair and its null basis

Face table row served: 2-plane E8 `U O8` code, `σ = 1` (hyperbolic). The null basis is exact on
spinor × spinor only, where the spinor set is the half-cube `S8+ = {s ∈ {±1}⁸ : ∏ s_p = 1}`.
Canonical declarations used: `Perf.FaceTyped.nullBasisMap`, `Perf.FaceTyped.two_plane_null_channel`,
`Perf.FaceTyped.null_channels_eq` (`Apeiron.Trichotomy.upper`, `lower`). Grade: `proved` (Lean kernel,
closed axiom cone). Scope: statements about sign vectors and real coefficients; no statement about a
trained weight matrix.
-/

set_option linter.apeiron.notation false

noncomputable section

namespace Perf.FaceTypedE8

open Perf.FaceTyped Apeiron.Trichotomy

/-- The half-cube spinor set `S8+`: sign vectors in `{±1}⁸` of even parity (`∏ s_p = 1`). -/
abbrev Spin8 := {s : Fin 8 → ℤˣ // ∏ p, s p = 1}

/-- **Closure of the half-cube under pointwise product.** The product of two even-parity sign vectors
has parity `(+1)(+1) = +1`. Uses `Finset.prod_mul_distrib`. -/
theorem spinor_mul_closed (s t : Fin 8 → ℤˣ) (hs : ∏ p, s p = 1) (ht : ∏ p, t p = 1) :
    ∏ p, (s p * t p) = 1 := by
  simp [Finset.prod_mul_distrib, hs, ht]

/-- Each sign vector squares to `1` pointwise (`(±1)² = 1` in `ℤˣ`). -/
theorem spinor_mul_self (s : Fin 8 → ℤˣ) : s * s = 1 :=
  funext fun p => Int.units_mul_self (s p)

/-- The null-basis re-coding on spinor pairs, `(s₁, s₂) ↦ (s₁, s₁s₂)`; the second entry stays in
`S8+` by `spinor_mul_closed`. Mirrors `Perf.FaceTyped.nullBasisMap` on `{±1}²`. -/
def spinorNullBasis : Spin8 × Spin8 ≃ Spin8 × Spin8 where
  toFun x := (x.1, ⟨x.1.1 * x.2.1, spinor_mul_closed _ _ x.1.2 x.2.2⟩)
  invFun x := (x.1, ⟨x.1.1 * x.2.1, spinor_mul_closed _ _ x.1.2 x.2.2⟩)
  left_inv x := by
    obtain ⟨⟨s, hs⟩, ⟨t, ht⟩⟩ := x
    simp [← _root_.mul_assoc, spinor_mul_self]
  right_inv x := by
    obtain ⟨⟨s, hs⟩, ⟨t, ht⟩⟩ := x
    simp [← _root_.mul_assoc, spinor_mul_self]

/-- **`spinorNullBasis` is an involution** on `S8+ × S8+`, so it is its own inverse and the pair code
carries no extra index. -/
theorem spinorNullBasis_involutive : Function.Involutive spinorNullBasis := by
  intro x
  obtain ⟨⟨s, hs⟩, ⟨t, ht⟩⟩ := x
  simp [spinorNullBasis, ← _root_.mul_assoc, spinor_mul_self]

/-- **Pointwise null channel of the shared-field spinor pair.** With both planes on one channel field
`ch` (the `ch₁ = ch₂` specialisation), the coordinate weight `r₁·ch·s₁ + r₂·ch·s₂` equals
`s₁·a` when `s₁s₂ = 1` and `s₁·b` otherwise, where `a = ch(r₁ + r₂)` and `b = ch(r₁ − r₂)`.
Reduces to `two_plane_null_channel` with `(r₁ ch, r₂ ch)` and `null_channels_eq`. -/
theorem spinor_pair_null_channel {s₁ s₂ : ℝ} (h₁ : s₁ = 1 ∨ s₁ = -1) (h₂ : s₂ = 1 ∨ s₂ = -1)
    (r₁ r₂ ch : ℝ) :
    r₁ * ch * s₁ + r₂ * ch * s₂ =
      s₁ * (if s₁ * s₂ = 1 then ch * (r₁ + r₂) else ch * (r₁ - r₂)) := by
  obtain ⟨hu, hl⟩ := null_channels_eq (r₁ * ch) (r₂ * ch)
  rw [two_plane_null_channel h₁ h₂, hu, hl]
  split_ifs <;> ring

/-- **Cross-diagonal identity.** For a diagonal entry `H p` and signs `s₁, s₂ ∈ {±1}`, the diagonal
quadratic term `H_pp · s₁p · s₂p` equals `H_pp · j_p` with `j_p = s₁p · s₂p`. -/
theorem spinor_cross_diag {ι : Type*} [Fintype ι] (H : ι → ℝ) (s₁ s₂ : ι → ℤˣ) :
    ∑ p, H p * ((s₁ p : ℤ) : ℝ) * ((s₂ p : ℤ) : ℝ) =
      ∑ p, H p * (((s₁ p * s₂ p : ℤˣ) : ℤ) : ℝ) := by
  refine Finset.sum_congr rfl fun p _ => ?_
  rw [Units.val_mul, Int.cast_mul]
  ring

section Expansion

variable {ι : Type*} [Fintype ι]

/-- **Two-plane block cost expansion.** For symmetric `H`, the block quadratic
`Σ H_pq (a₁g₁p + a₂g₂p − t_p)(a₁g₁q + a₂g₂q − t_q)` equals
`a₁²Q₁ + a₂²Q₂ + 2a₁a₂X − 2a₁u − 2a₂v + T`, with `Q₁, Q₂, X, u, v, T` the coefficient sums
defined inline. Proof: termwise `ring`, then `Finset.sum_comm` and symmetry of `H` for the cross terms. -/
theorem joint_block_cost_expand (H : ι → ι → ℝ) (hH : ∀ p q, H p q = H q p) (g₁ g₂ t : ι → ℝ)
    (a₁ a₂ : ℝ) :
    ∑ p, ∑ q, H p q * (a₁ * g₁ p + a₂ * g₂ p - t p) * (a₁ * g₁ q + a₂ * g₂ q - t q) =
      a₁ ^ 2 * (∑ p, ∑ q, H p q * g₁ p * g₁ q) + a₂ ^ 2 * (∑ p, ∑ q, H p q * g₂ p * g₂ q)
      + 2 * a₁ * a₂ * (∑ p, ∑ q, H p q * g₁ p * g₂ q)
      - 2 * a₁ * (∑ p, ∑ q, H p q * g₁ p * t q) - 2 * a₂ * (∑ p, ∑ q, H p q * g₂ p * t q)
      + ∑ p, ∑ q, H p q * t p * t q := by
  have s12 : ∑ p, ∑ q, H p q * g₂ p * g₁ q = ∑ p, ∑ q, H p q * g₁ p * g₂ q := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    rw [hH]; ring
  have st1 : ∑ p, ∑ q, H p q * t p * g₁ q = ∑ p, ∑ q, H p q * g₁ p * t q := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    rw [hH]; ring
  have st2 : ∑ p, ∑ q, H p q * t p * g₂ q = ∑ p, ∑ q, H p q * g₂ p * t q := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun q _ => ?_
    rw [hH]; ring
  have key : ∀ p q, H p q * (a₁ * g₁ p + a₂ * g₂ p - t p) * (a₁ * g₁ q + a₂ * g₂ q - t q) =
      a₁ ^ 2 * (H p q * g₁ p * g₁ q) + a₂ ^ 2 * (H p q * g₂ p * g₂ q)
      + a₁ * a₂ * (H p q * g₁ p * g₂ q + H p q * g₂ p * g₁ q)
      - a₁ * (H p q * g₁ p * t q + H p q * t p * g₁ q)
      - a₂ * (H p q * g₂ p * t q + H p q * t p * g₂ q)
      + H p q * t p * t q := by
    intro p q; ring
  simp only [key, Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, s12, st1, st2]
  ring

end Expansion

section Move

/-- **Joint move bounded by each slice.** The minimum over the product index set is at most the
minimum over the slice `{a} × t`, for any `a ∈ s`. Uses `Finset.inf'_le`, `Finset.le_inf'_iff`. -/
theorem joint_move_le_planewise {α β : Type*} (s : Finset α) (t : Finset β) (hs : s.Nonempty)
    (ht : t.Nonempty) (f : α → β → ℝ) (a : α) (ha : a ∈ s) :
    (s ×ˢ t).inf' (hs.product ht) (fun x => f x.1 x.2) ≤ t.inf' ht (fun b => f a b) := by
  rw [Finset.le_inf'_iff]
  intro b hb
  exact Finset.inf'_le _ (show (a, b) ∈ s ×ˢ t from Finset.mem_product.2 ⟨ha, hb⟩)

/-- **Joint move bounded by the other slice.** Symmetric form: for `b ∈ t`, the product minimum is at
most the minimum over `s × {b}`. -/
theorem joint_move_le_planewise' {α β : Type*} (s : Finset α) (t : Finset β) (hs : s.Nonempty)
    (ht : t.Nonempty) (f : α → β → ℝ) (b : β) (hb : b ∈ t) :
    (s ×ˢ t).inf' (hs.product ht) (fun x => f x.1 x.2) ≤ s.inf' hs (fun a => f a b) := by
  rw [Finset.le_inf'_iff]
  intro a ha
  exact Finset.inf'_le _ (show (a, b) ∈ s ×ˢ t from Finset.mem_product.2 ⟨ha, hb⟩)

end Move

/-- **Cardinality of the half-cube.** `|S8+| = 2⁷ = 128`. Kernel `decide` over the 256 sign vectors. -/
theorem spinor_card : Fintype.card Spin8 = 128 := by
  decide +kernel

/-- **Pair rate.** `|(Fin 256 × Fin 256)| = 2¹⁶`, the bit count of a pair of 8-bit words. -/
theorem pair_rate_eq : Fintype.card (Fin 256 × Fin 256) = 2 ^ 16 := by
  rw [Fintype.card_prod, Fintype.card_fin]
  norm_num

/-- **Sparse sums are not injective.** On the words `{0, ±2}⁸`, the sum map `(u, v) ↦ u + v` takes
the same value on two distinct pairs, and that value lies in `D8` (even coordinate sum). Witness:
`(u₁, v₁) = (2e₀, 0)` and `(u₂, v₂) = (0, 2e₀)`. No square root is taken. -/
theorem sparse_not_injective :
    ∃ u₁ v₁ u₂ v₂ : Fin 8 → ℤ,
      (∀ p, u₁ p ∈ ({0, 2, -2} : Finset ℤ)) ∧ (∀ p, v₁ p ∈ ({0, 2, -2} : Finset ℤ)) ∧
      (∀ p, u₂ p ∈ ({0, 2, -2} : Finset ℤ)) ∧ (∀ p, v₂ p ∈ ({0, 2, -2} : Finset ℤ)) ∧
      (u₁ ≠ u₂ ∨ v₁ ≠ v₂) ∧ u₁ + v₁ = u₂ + v₂ ∧ (∑ p, (u₁ + v₁) p) % 2 = 0 := by
  refine ⟨![2, 0, 0, 0, 0, 0, 0, 0], 0, 0, ![2, 0, 0, 0, 0, 0, 0, 0], ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro p; fin_cases p <;> simp
  · intro p; simp
  · intro p; simp
  · intro p; fin_cases p <;> simp
  · left; intro h; have := congrFun h 0; simp at this
  · ext p; fin_cases p <;> simp
  · simp [Fin.sum_univ_succ]

/-! ## Witnesses for each hypothesis bundle -/

/-- `spinor_mul_closed` on the all-`+1` vector. -/
example : ∏ _p : Fin 8, (1 : ℤˣ) = 1 := by simp

/-- `spinorNullBasis` fixes the all-`+1` pair. -/
example : spinorNullBasis (⟨fun _ => 1, by simp⟩, ⟨fun _ => 1, by simp⟩) =
    (⟨fun _ => 1, by simp⟩, ⟨fun _ => 1, by simp⟩) := rfl

/-- `spinor_pair_null_channel` at `s₁ = 1, s₂ = −1, r₁ = 3, r₂ = 1, ch = 2`: weight `4 = 1·ch(r₁ − r₂)`. -/
example : (3 : ℝ) * 2 * 1 + 1 * 2 * (-1) = 1 * (2 * (3 - 1)) := by norm_num

/-- `joint_block_cost_expand` hypotheses are met by `H = 1` on `Fin 2`, `g₁ = g₂ = t = 1`. -/
example : ∑ _p : Fin 2, ∑ _q : Fin 2, (1 : ℝ) = 4 := by norm_num [Fin.sum_univ_two]

end Perf.FaceTypedE8
