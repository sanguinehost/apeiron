import Apeiron.Trichotomy.Basic
import Apeiron.Trichotomy.NullBasis
import Perf.FaceTyped
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Tactic

set_option linter.apeiron.notation false

/-!
# Side information on the hyperbolic face

Face served: hyperbolic `σ = 1`, `R[x]/(x² − 1)`, characteristic coordinates `u = a − b`,
`v = a + b` (`upper 1`, `lower 1`). Canonical declarations used: `Apeiron.Trichotomy.normForm`,
`upper`, `lower` (`Trichotomy/Basic.lean`, `Trichotomy/NullBasis.lean`),
`Apeiron.Trichotomy.normForm_eq_upper_mul_lower` (`Trichotomy/NullBasis.lean:122`),
`Perf.FaceTyped.nullBasisMap` (`Perf/FaceTyped.lean:66`).

Lineage: the null-basis recoding, its involution and the lost bit are the `σ = 1` instance of the
two-plane code of `Perf.FaceTyped`. The norm form sees the product `uv`, not the factors: the boost
`(u, v) ↦ (λu, v/λ)` is invisible to it (`norm_boost_blind`), and the sign fibre of the product
has two points (`lone_channel_fiber_card`). A side-information bit recovers the lost coordinate
(`side_info_recovers`).

Operational reading of `gauss_side_info_saving` as the Wyner–Ziv rate saving (Wyner and Ziv,
IEEE Trans. Inf. Theory 22(1), 1976) is an anchor. No coding theorem is claimed here.
Grades: `proved` for the Lean statements; the rate reading is an `anchor`.
Scope: real numbers and signs; no statement about a trained weight matrix.
-/

noncomputable section

namespace Perf.SideInfo

open Apeiron.Trichotomy

/-- **Characteristic coordinates on the hyperbolic face.** At `σ = 1` the norm form is the product of
the two null coordinates `upper 1 z = a + b` and `lower 1 z = a − b` (Gauss, *Disquisitiones
Arithmeticae* (1801) for the norm form; Cockle (1848) for split-complex numbers). -/
theorem char_coords_norm (a b : ℝ) :
    normForm 1 (a, b) = (a - b) * (a + b) := by
  rw [normForm_eq_upper_mul_lower zero_le_one, upper, lower]
  rw [Real.sqrt_one]
  ring

/-- **Elliptic control (`σ = −1`).** The factorisation of `char_coords_norm` fails when `σ < 0`: at
`z = (0, 1)` the norm form is `1` while the two null coordinates are both `0`. Red if the constant
`σ` in this statement is changed to a hyperbolic value. -/
example : normForm (-1) (0, 1) ≠ upper (-1) (0, 1) * lower (-1) (0, 1) := by
  have h0 : Real.sqrt (-1 : ℝ) = 0 := Real.sqrt_eq_zero'.2 (by norm_num)
  simp only [upper, lower, h0, zero_mul, add_zero, sub_zero, mul_zero]
  simp only [normForm, AnyRing.ringNormForm]
  norm_num

/-- **The boost is invisible to the norm.** For `λ ≠ 0`, `(λu)(v/λ) = uv`: the rescaling
`(u, v) ↦ (λu, v/λ)` preserves the norm form, so the norm form does not see the boost. -/
theorem norm_boost_blind {l : ℝ} (hl : l ≠ 0) (u v : ℝ) :
    (l * u) * (v / l) = u * v := by
  field_simp

/-- **One bit is lost.** Each fibre of the product map `(s₁, s₂) ↦ s₁s₂` on `ℤˣ × ℤˣ` has two
points, so the product `s₁s₂` carries one bit less than the pair. -/
theorem lone_channel_fiber_card (p : ℤˣ) :
    Fintype.card {x : ℤˣ × ℤˣ // x.1 * x.2 = p} = 2 := by
  rcases Int.units_eq_one_or p with rfl | rfl <;> decide

/-- **Side information recovers the lost bit.** Applying the null-basis recoding
`(s₁, s₂) ↦ (s₁, s₁s₂)` twice returns `(s₁, s₂)`; the first coordinate `s₁` is the side
information that fixes the fibre. Uses `Perf.FaceTyped.nullBasisMap`. -/
theorem side_info_recovers (s₁ s₂ : ℤˣ) :
    Perf.FaceTyped.nullBasisMap (Perf.FaceTyped.nullBasisMap (s₁, s₂)) = (s₁, s₂) :=
  Perf.FaceTyped.null_basis_rate_eq.2.1 (s₁, s₂)

/-- **Real-coordinate form.** For `u ≠ 0` the equation `uv = p` has exactly one solution `v = p/u`. -/
theorem side_info_char_coord {u p : ℝ} (hu : u ≠ 0) : ∃! v : ℝ, u * v = p :=
  ⟨p / u, by field_simp, fun y hy => by
    rw [eq_div_iff hu, _root_.mul_comm]
    exact hy⟩

/-- **Gaussian side-information saving.** For `0 < D ≤ s(1 − ρ)` and `0 ≤ ρ < 1`,
`½ log₂(s/D) − ½ log₂(s(1−ρ)/D) = ½ log₂(1/(1−ρ))`. Joins `Perf.CostCorrespondence.waterfill_opt`
and `waterfill_highres` as the rate saving of conditioning on side information of correlation `ρ`
(Wyner and Ziv (1976), anchor). -/
theorem gauss_side_info_saving {s D ρ : ℝ} (hD : 0 < D) (hDs : D ≤ s * (1 - ρ))
    (_hρ0 : 0 ≤ ρ) (hρ1 : ρ < 1) :
    (1 / 2) * Real.logb 2 (s / D) - (1 / 2) * Real.logb 2 (s * (1 - ρ) / D) =
      (1 / 2) * Real.logb 2 (1 / (1 - ρ)) := by
  have h1 : 0 < 1 - ρ := by linarith
  have hs : 0 < s := by
    by_contra h
    rw [not_lt] at h
    nlinarith
  have e1 : Real.log (s / D) = Real.log s - Real.log D :=
    Real.log_div hs.ne' hD.ne'
  have e2 : Real.log (s * (1 - ρ) / D) = Real.log s + Real.log (1 - ρ) - Real.log D := by
    rw [Real.log_div (mul_pos hs h1).ne' hD.ne', Real.log_mul hs.ne' h1.ne']
  have e3 : Real.log (1 / (1 - ρ)) = -Real.log (1 - ρ) := by
    rw [one_div, Real.log_inv]
  simp only [Real.logb, e1, e2, e3]
  ring

/-- Binary search on `[lo, hi)`: the index of the first key `≥ x` in a sorted array, and the number of
key comparisons. Recursion on `hi - lo`. -/
def bsearch (a : ℕ → ℚ) (x : ℚ) (lo hi : ℕ) : ℕ × ℕ :=
  if h : lo < hi then
    if a ((lo + hi) / 2) < x then
      ((bsearch a x ((lo + hi) / 2 + 1) hi).1, (bsearch a x ((lo + hi) / 2 + 1) hi).2 + 1)
    else
      ((bsearch a x lo ((lo + hi) / 2)).1, (bsearch a x lo ((lo + hi) / 2)).2 + 1)
  else (lo, 0)
termination_by hi - lo
decreasing_by all_goals omega

/-- **Lower-bound search on a sorted key array.** Binary search over `a : Fin n → ℚ` returns the
number of keys below `x` as an index in `Fin (n+1)`, together with the number of key comparisons.
Sorted input is assumed by `lowerBound_spec`; `lowerBound_count_le` needs no order.

Temporal: `⌈log2(m+1)⌉` key comparisons per lookup, for `m = n` keys. Recurrence
`c(m) = 1 + c(⌊m/2⌋)`, `c(0) = 0`; the child interval has at most `⌊m/2⌋` keys
(`mid - lo = ⌊m/2⌋`, `hi - mid - 1 ≤ ⌊m/2⌋`), and induction on `k` with `m + 1 ≤ 2^k` gives
`c(m) ≤ k`, closing at `⌈log2(m+1)⌉ = Nat.clog 2 (m+1)`.
Spatial: `m` keys plus `m` indices (words); no allocation beyond the recursion stack.
Sort: `O(m log m)` comparisons once per `(layer, kind, β)`, outside this function. -/
def lowerBound {n : ℕ} (a : Fin n → ℚ) (x : ℚ) : Fin (n + 1) × ℕ :=
  let f : ℕ → ℚ := fun i => if h : i < n then a ⟨i, h⟩ else 0
  let r := bsearch f x 0 n
  (⟨min r.1 n, Nat.lt_succ_of_le (min_le_right _ _)⟩, r.2)

/-- The comparison count is at most `⌈log2(n+1)⌉`. Induction on `k` with `hi - lo + 1 ≤ 2^k`. -/
theorem bsearch_steps (a : ℕ → ℚ) (x : ℚ) :
    ∀ k lo hi, hi - lo + 1 ≤ 2 ^ k → (bsearch a x lo hi).2 ≤ k := by
  intro k
  induction k with
  | zero =>
    intro lo hi h
    simp only [pow_zero] at h
    have hn : ¬ lo < hi := by omega
    simp [bsearch, hn]
  | succ k ih =>
    intro lo hi h
    rw [pow_succ] at h
    by_cases hlt : lo < hi
    · unfold bsearch
      rw [dif_pos hlt]
      split_ifs
      · have := ih ((lo + hi) / 2 + 1) hi (by omega)
        dsimp only
        omega
      · have := ih lo ((lo + hi) / 2) (by omega)
        dsimp only
        omega
    · simp [bsearch, hlt]

/-- Count bound: at most `⌈log2(n+1)⌉` key comparisons per lookup. -/
theorem lowerBound_count_le {n : ℕ} (a : Fin n → ℚ) (x : ℚ) :
    (lowerBound a x).2 ≤ Nat.clog 2 (n + 1) := by
  unfold lowerBound
  apply bsearch_steps
  have := Nat.le_pow_clog (by norm_num : 1 < 2) (n + 1)
  simp only [Nat.sub_zero]
  omega

/-- Invariant for `bsearch`: with `hi ≤ n` and the bracket facts below `lo` and at or above `hi`,
the returned index lies in `[lo, hi]`, keys below it are `< x` and keys from it on are `≥ x`. -/
theorem bsearch_inv (a : ℕ → ℚ) (x : ℚ) (n : ℕ)
    (hm : ∀ i j, i ≤ j → j < n → a i ≤ a j) :
    ∀ k lo hi, lo ≤ hi → hi ≤ n → hi - lo + 1 ≤ 2 ^ k →
      (∀ i, i < lo → i < n → a i < x) → (∀ i, hi ≤ i → i < n → ¬ a i < x) →
      lo ≤ (bsearch a x lo hi).1 ∧ (bsearch a x lo hi).1 ≤ hi ∧
      (∀ i, i < (bsearch a x lo hi).1 → i < n → a i < x) ∧
      (∀ i, (bsearch a x lo hi).1 ≤ i → i < n → ¬ a i < x) := by
  intro k
  induction k with
  | zero =>
    intro lo hi hle hhn h hlo hhi
    simp only [pow_zero] at h
    have heq : lo = hi := by omega
    have hn : ¬ lo < hi := by omega
    rw [bsearch, dif_neg hn]
    refine ⟨le_refl _, heq ▸ le_refl _, ?_, ?_⟩
    · intro i hi' hin; exact hlo i hi' hin
    · intro i hi' hin; exact hhi i (heq ▸ hi') hin
  | succ k ih =>
    intro lo hi hle hhn h hlo hhi
    rw [pow_succ] at h
    by_cases hlt : lo < hi
    · unfold bsearch
      rw [dif_pos hlt]
      split_ifs with hmid
      · dsimp only
        -- the midpoint is below x: search the upper half
        have hmidlt : (lo + hi) / 2 < hi := by omega
        have hmidn : (lo + hi) / 2 < n := by omega
        obtain ⟨h1, h2, h3, h4⟩ := ih ((lo + hi) / 2 + 1) hi (by omega) (by omega) (by omega)
          (by
            intro i hi' hin
            rcases Nat.lt_or_ge i lo with hl | hl
            · exact hlo i hl hin
            · exact lt_of_le_of_lt (hm i ((lo + hi) / 2) (by omega) hmidn) hmid)
          hhi
        exact ⟨by omega, h2, h3, h4⟩
      · dsimp only
        -- the midpoint is at least x: search the lower half
        have hmidlt : (lo + hi) / 2 < hi := by omega
        have hmidn : (lo + hi) / 2 < n := by omega
        obtain ⟨h1, h2, h3, h4⟩ := ih lo ((lo + hi) / 2) (by omega) (by omega) (by omega)
          hlo
          (by
            intro i hi' hin
            rcases Nat.lt_or_ge i hi with hl | hl
            · exact not_lt.2 (le_trans (not_lt.1 hmid) (hm _ _ hi' hin))
            · exact hhi i hl hin)
        exact ⟨h1, by omega, h3, h4⟩
    · rw [bsearch, dif_neg hlt]
      have heq : lo = hi := by omega
      refine ⟨le_refl _, heq ▸ le_refl _, ?_, ?_⟩
      · intro i hi' hin; exact hlo i hi' hin
      · intro i hi' hin; exact hhi i (heq ▸ hi') hin

/-- Spec: for monotone keys, the index is the number of keys below `x`. -/
theorem lowerBound_spec {n : ℕ} {a : Fin n → ℚ} (hmono : Monotone a) (x : ℚ) :
    ((lowerBound a x).1 : ℕ) = (Finset.univ.filter (fun i : Fin n => a i < x)).card := by
  have hm : ∀ i j, i ≤ j → j < n →
      (fun i : ℕ => if h : i < n then a ⟨i, h⟩ else 0) i ≤
        (fun i : ℕ => if h : i < n then a ⟨i, h⟩ else 0) j := by
    intro i j hij hj
    have hi : i < n := lt_of_le_of_lt hij hj
    simp only [dif_pos hi, dif_pos hj]
    exact hmono (Fin.mk_le_mk.2 hij)
  obtain ⟨h1, h2, h3, h4⟩ :=
    bsearch_inv (fun i : ℕ => if h : i < n then a ⟨i, h⟩ else 0) x n hm (Nat.clog 2 (n + 1)) 0 n
      (Nat.zero_le _) le_rfl
      (by have := Nat.le_pow_clog (by norm_num : 1 < 2) (n + 1); omega)
      (fun i hi _ => absurd hi (Nat.not_lt_zero i))
      (fun i hi hin => absurd hin (not_lt.2 hi))
  have hfi : ∀ i : Fin n, (if h : (i : ℕ) < n then a ⟨i, h⟩ else 0) = a i := fun i => by
    simp
  have key : ∀ i : Fin n, a i < x ↔ (i : ℕ) < (bsearch (fun i : ℕ => if h : i < n then a ⟨i, h⟩ else 0) x 0 n).1 := by
    intro i
    constructor
    · intro hai
      by_contra hc
      rw [not_lt] at hc
      exact h4 i hc i.isLt (by simp only [hfi]; exact hai)
    · intro hi
      by_contra hc
      exact hc (by have := h3 i hi i.isLt; simp only [hfi] at this; exact this)
  have hr : (lowerBound a x).1.val = (bsearch (fun i : ℕ => if h : i < n then a ⟨i, h⟩ else 0) x 0 n).1 :=
    min_eq_left h2
  rw [hr]
  have hset : Finset.univ.filter (fun i : Fin n => a i < x) =
      Finset.univ.filter (fun i : Fin n => (i : ℕ) < (bsearch (fun i : ℕ => if h : i < n then a ⟨i, h⟩ else 0) x 0 n).1) := by
    ext i
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact key i
  rw [hset, Fin.card_filter_val_lt, min_eq_right h2]

/-! ### Verdicts over the measured constants (spec section 5)

The thresholds are the decision rule; the probe receipt supplies the measured value. The lead keeps
the verdict that matches the run. Route (b) is open at `R²_H ≥ 0.13`; closed at linear/product
junction order at `R²_H ≤ 0.02`. Route (a) is closed at `R_a(f*) ≥ 2.0` b/w. -/

/-- Route (b) open: the J-weighted incremental decoded `R²_H` is at least `0.13` (`ΔR ≥ 0.10` b/w). -/
def RouteBOpen (r : ℝ) : Prop := (0.13 : ℝ) ≤ r

/-- Route (b) closed at linear/product-junction order: the incremental decoded `R²_H` is at most `0.02`. -/
def RouteBClosedLinear (r : ℝ) : Prop := r ≤ (0.02 : ℝ)

/-- Route (a) closed: the reverse-water-filling rate at `f*` is at least `2.0` b/w. -/
def RouteAClosed (R : ℝ) : Prop := (2.0 : ℝ) ≤ R

theorem w16_t6_route_b_open : RouteBOpen 0.13 ∧ ¬ RouteBClosedLinear 0.13 := by
  norm_num [RouteBOpen, RouteBClosedLinear]

theorem w16_t6_route_b_closed_linear : RouteBClosedLinear 0.02 ∧ ¬ RouteBOpen 0.02 := by
  norm_num [RouteBOpen, RouteBClosedLinear]

theorem w16_t6_route_a_closed : RouteAClosed 2.0 ∧ ¬ (RouteAClosed 1.9) := by
  norm_num [RouteAClosed]

/-- Constant control: changing the route (b) open threshold to `0.12` makes this red. -/
example : ¬ RouteBOpen 0.12 := by norm_num [RouteBOpen]

/-- Constant control: changing the linear closure threshold to `0.03` makes this red. -/
example : ¬ RouteBClosedLinear 0.03 := by norm_num [RouteBClosedLinear]

/-- Constant control: changing the route (a) threshold to `1.9` makes this red. -/
example : ¬ RouteAClosed 1.9 := by norm_num [RouteAClosed]

end Perf.SideInfo
