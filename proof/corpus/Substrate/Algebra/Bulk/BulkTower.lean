import Substrate.Algebra.Bulk.UniversalEquation
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity

/-!
# BulkTower: Multi-Level Recursive Composition of the Universal Equation
@name The Rule Inside the Rule
@child The one rule (boundary over drag) repeats inside itself, level after level, like nested boxes. At each level the answer from below flows into either the next "reaching" or the next "holding back" — and *which* one is the only real choice there is.
@scientist x = B/U recurses at every scale as an inductive tower: each transition routes the previous state x into the next boundary B (`stepB`) or drag U (`stepU`). Organisation = the fraction of transitions routed to U (delegating to substrate) versus B (re-managing as goal) — a counting measure that falls out of the structure, not a separate axiom.
@engineer `BulkTower` inductive (`base`, `stepB`, `stepU`); `state` the cumulative top-level value; `depth` / `stepU_count` / `stepB_count` the structural counters. One equation, everything downstream.

The bulk basis says `x = B / U` recurses at every scale. This file
formalises that recursion as an inductive type. At each transition, the
previous level's state `x` flows into either the next level's boundary
`B` (`stepB`) or its drag `U` (`stepU`). The choice of where `x` flows is
the structural degree of freedom that gives rise to *Organisation*: a
configuration's Organisation is the fraction of transitions that route
`x` to `U` (delegating to substrate) versus `B` (re-managing as goal).

Organisation isn't a separate axiom — it falls out of the inductive
structure as a counting measure on the tower. This is the bulk basis's
monistic discipline: one equation, everything downstream.

## Contents

* `BulkTower`: inductive structure (`base`, `stepB`, `stepU`).
* `state`: the cumulative state at the tower's top level.
* `depth`, `stepU_count`, `stepB_count`: structural counters.
* `total_steps_eq_depth`: the two counters sum to depth (consistency).
* `organization`: derived measure (`stepU_count / depth`).
* `organization_in_unit_interval`: organisation lies in `[0, 1]`.
* `flatten`: project a tower to an equivalent `UniversalEquation`.
* `state_eq_flatten_state`: tower state and flattened state agree.
* `tower_bulk_differential`: scale invariance — the quotient-rule
  decomposition holds at every level of the tower.
* `AllDifferentiable`: per-level side conditions, one clause per constructor.
* `allDifferentiable_state`: structural induction — `AllDifferentiable t p`
  implies `DifferentiableAt ℝ t.state p`.
* `AllNonzero`, `allNonzero_state_ne_zero`: leaf nonvanishing propagates to
  every level's state.
* `LeafDifferentiable`, `allDifferentiable_of_leaf_nonzero`: leaf data
  discharges `AllDifferentiable` — the `stepU` clause's `prev.state p ≠ 0`
  derived, not assumed.
* `tower_bulk_differential_of_allDifferentiable`,
  `tower_stationary_phase_of_allDifferentiable`: the decompositions under the
  inductive hypothesis.

## What is proved here, and what "at every level" means

* **Proved (native):** the structural facts — `total_steps_eq_depth` (induction + `omega`),
  `organization_nonneg`, `organization_le_one`, `organization_in_unit_interval`, and
  `state_eq_flatten_state` (`rfl` at each constructor). `organization` is `stepU_count / depth` with
  `0` at depth `0`, and the unit-interval bound is exactly `stepU_count ≤ depth`.
* **Proved (riding):** `tower_bulk_differential` and `tower_stationary_phase` are
  `UniversalEquation.bulk_differential` and `UniversalEquation.stationary_phase` after
  `rw [state_eq_flatten_state]`. The calculus is `UniversalEquation`'s; this file supplies the
  rewrite.
* **Proved (native):** by induction on the tower — `allDifferentiable_state` —
  `AllDifferentiable t p → DifferentiableAt ℝ t.state p`, structural induction,
  `DifferentiableAt.div` at each constructor. `AllDifferentiable` is checked at every level down
  to the base: leaf differentiability plus nonvanishing of each level's denominator (`e.U p` at
  `base`, `U_outer p` at `stepB`, `prev.state p` at `stepU`). `allDifferentiable_flatten` then
  discharges the flat `hB`/`hU`/`hU_nz` of `tower_bulk_differential` — the `stepB` case's
  `DifferentiableAt ℝ t.flatten.B p` (the sub-tower's state) is established, no longer assumed.
  `tower_bulk_differential_of_allDifferentiable` / `tower_stationary_phase_of_allDifferentiable`
  are the resulting corollaries; the `stepB`/`stepU` clauses carry `AllDifferentiable prev p` as
  first conjunct, so the same corollaries apply to every sub-tower via `.1`.
* **Proved (native):** `allNonzero_state_ne_zero` — `AllNonzero t p → t.state p ≠ 0` by
  induction with `div_ne_zero`, and `allDifferentiable_of_leaf_nonzero` — `LeafDifferentiable` +
  `AllNonzero` imply `AllDifferentiable`, so the `stepU` clause's `prev.state p ≠ 0` is derivable
  from leaf data. `AllNonzero` is sufficient, not necessary: it demands numerators nonzero at
  `stepB` levels where differentiability alone does not need them.
* **Hypothesis floor:** the leaf conditions themselves (differentiability and nonvanishing of the
  arbitrary `ℝ → ℝ` functions at each node) remain hypotheses; the leaves are unconstrained
  functions, so nothing below them exists to derive from.
* **Anchor:** that `organization` measures "delegating to substrate versus re-managing
  as goal", and that "configurations with higher Organisation extract more of their structural
  integration from substrate-given recursion". `organization`'s own docstring already carries the
  caveat and points downstream (`DefectsAsMaxima`); this bullet is the label for it. As proved here
  it is a syntactic count of `stepU` constructors over depth.

-/

noncomputable section

namespace Sanguine

open UniversalEquation

/-- The recursive tower of universal equations.

    * `base e`: a leaf — a single universal equation at the bottom.
    * `stepB prev U_outer`: the previous level's state becomes the new
      boundary; the new drag is `U_outer`. (Drags accumulate.)
    * `stepU B_outer prev`: the previous level's state becomes the new
      drag; the new boundary is `B_outer`. (Boundary inverts.) -/
inductive BulkTower : Type where
  | base (e : UniversalEquation) : BulkTower
  | stepB (prev : BulkTower) (U_outer : ℝ → ℝ) : BulkTower
  | stepU (B_outer : ℝ → ℝ) (prev : BulkTower) : BulkTower

namespace BulkTower

/-- The cumulative state function of a tower. -/
def state : BulkTower → ℝ → ℝ
  | base e => e.state
  | stepB prev U_outer => fun p => prev.state p / U_outer p
  | stepU B_outer prev => fun p => B_outer p / prev.state p

/-- The depth of the tower: number of compositions above the base. -/
def depth : BulkTower → ℕ
  | base _ => 0
  | stepB prev _ => prev.depth + 1
  | stepU _ prev => prev.depth + 1

/-- Count of `stepU` transitions (x flows into U — delegation to substrate). -/
def stepU_count : BulkTower → ℕ
  | base _ => 0
  | stepB prev _ => prev.stepU_count
  | stepU _ prev => prev.stepU_count + 1

/-- Count of `stepB` transitions (x flows into B — re-managed as goal). -/
def stepB_count : BulkTower → ℕ
  | base _ => 0
  | stepB prev _ => prev.stepB_count + 1
  | stepU _ prev => prev.stepB_count

/-- **Consistency: the two transition counters sum to depth.** Every
    transition is either a `stepB` or a `stepU`, so their counts
    partition the depth. -/
theorem total_steps_eq_depth (t : BulkTower) :
    t.stepU_count + t.stepB_count = t.depth := by
  induction t with
  | base e => simp [stepU_count, stepB_count, depth]
  | stepB prev U_outer ih =>
      simp [stepU_count, stepB_count, depth]; omega
  | stepU B_outer prev ih =>
      simp [stepU_count, stepB_count, depth]; omega

/-- **Organisation as a derived measure.** The fraction of transitions
    in the tower that route the previous level's state into the next
    level's drag (delegating to substrate) rather than the next level's
    boundary (re-managing as goal).

    Bulk-basis significance: this measure emerges from the tower's
    structure without needing a separate axiom. Configurations with
    higher Organisation extract more of their structural integration
    from substrate-given recursion rather than expending capacity on
    re-managing it as goals.

    Caveat: as defined here, `organization` is a syntactic counting
    measure on the tower's constructor pattern. Its bulk-basis
    significance — correlation with defect stability, Φ-maximum
    coincidence, scale-invariance survival — is established in
    downstream files (`DefectsAsMaxima`), not here.

    Convention: at depth 0 (no transitions), Organisation is 0. -/
def organization (t : BulkTower) : ℝ :=
  if t.depth = 0 then 0 else (t.stepU_count : ℝ) / (t.depth : ℝ)

/-- Organisation is non-negative. -/
theorem organization_nonneg (t : BulkTower) : 0 ≤ t.organization := by
  unfold organization
  split_ifs with h
  · rfl
  · positivity

/-- Organisation is at most 1. -/
theorem organization_le_one (t : BulkTower) : t.organization ≤ 1 := by
  unfold organization
  split_ifs with h
  · norm_num
  · have hd_pos : (t.depth : ℝ) > 0 := by
      have : t.depth > 0 := Nat.pos_of_ne_zero h
      exact_mod_cast this
    rw [div_le_one hd_pos]
    have h_le : t.stepU_count ≤ t.depth := by
      have := total_steps_eq_depth t; omega
    exact_mod_cast h_le

/-- Organisation lies in the unit interval. -/
theorem organization_in_unit_interval (t : BulkTower) :
    0 ≤ t.organization ∧ t.organization ≤ 1 :=
  ⟨organization_nonneg t, organization_le_one t⟩

/-- **One-level flattening to an equivalent universal equation.** Each
    BulkTower projects to a UniversalEquation whose state agrees with
    the tower's state pointwise. This is one-level (the current
    transition's B/U), not fully recursive collapse to base primitives;
    that distinction is what lets `state_eq_flatten_state` hold as
    `rfl`. -/
def flatten : BulkTower → UniversalEquation
  | base e => e
  | stepB prev U_outer => { B := prev.state, U := U_outer }
  | stepU B_outer prev => { B := B_outer, U := prev.state }

/-- **State consistency.** The tower's state agrees with its flattened
    equation's state by definition (each constructor is set up to make
    this hold as `rfl`). -/
theorem state_eq_flatten_state (t : BulkTower) :
    t.state = t.flatten.state := by
  cases t with
  | base e => rfl
  | stepB prev U_outer => rfl
  | stepU B_outer prev => rfl

/-- **Top-level quotient-rule decomposition for any tower.**

    Given a tower `t` whose flattened-equation `B` and `U` are
    differentiable at `p` with non-zero drag, the tower's state at `p`
    satisfies the quotient-rule decomposition:
    $$\frac{\partial t.\text{state}}{\partial p}(p) = \frac{\bar U \cdot d\bar B - \bar B \cdot d\bar U}{\bar U^2}$$
    where `B̄ = t.flatten.B` and `Ū = t.flatten.U`.

    The hypotheses here are flat: for a `stepB` node, `hB` is
    differentiability of the sub-tower's state, assumed. The depth-inductive
    version, with hypotheses checked at every level down to the base, is
    `tower_bulk_differential_of_allDifferentiable`. -/
theorem tower_bulk_differential
    (t : BulkTower) (p : ℝ)
    (hB : DifferentiableAt ℝ t.flatten.B p)
    (hU : DifferentiableAt ℝ t.flatten.U p)
    (hU_nz : t.flatten.U p ≠ 0) :
    deriv t.state p =
      (t.flatten.U p * deriv t.flatten.B p -
       t.flatten.B p * deriv t.flatten.U p) / (t.flatten.U p)^2 := by
  rw [state_eq_flatten_state]
  exact bulk_differential hB hU hU_nz

/-- **Stationary phase at every level.** The critical points of any
    tower's state are characterised by the same equation
    `U · dB = B · dU` as the base. Scale invariance for the variational
    principle. -/
theorem tower_stationary_phase
    (t : BulkTower) (p : ℝ)
    (hB : DifferentiableAt ℝ t.flatten.B p)
    (hU : DifferentiableAt ℝ t.flatten.U p)
    (hU_nz : t.flatten.U p ≠ 0) :
    deriv t.state p = 0 ↔
      t.flatten.U p * deriv t.flatten.B p =
      t.flatten.B p * deriv t.flatten.U p := by
  rw [state_eq_flatten_state]
  exact stationary_phase hB hU hU_nz

/-- Per-level side conditions for differentiability of the state, one clause
    per constructor, checked down to the base:

    * `base e`: `e.B`, `e.U` differentiable at `p` and `e.U p ≠ 0`.
    * `stepB prev U_outer`: the conditions for `prev`, plus `U_outer`
      differentiable at `p` and `U_outer p ≠ 0`.
    * `stepU B_outer prev`: the conditions for `prev`, plus `B_outer`
      differentiable at `p` and `prev.state p ≠ 0`.

    The `stepU` clause's `prev.state p ≠ 0` is a hypothesis here, not derived:
    differentiability of the leaves does not force a numerator nonzero. It is
    derivable from leaf-level nonvanishing — `allDifferentiable_of_leaf_nonzero`. -/
def AllDifferentiable : BulkTower → ℝ → Prop
  | base e, p =>
      DifferentiableAt ℝ e.B p ∧ DifferentiableAt ℝ e.U p ∧ e.U p ≠ 0
  | stepB prev U_outer, p =>
      AllDifferentiable prev p ∧ DifferentiableAt ℝ U_outer p ∧ U_outer p ≠ 0
  | stepU B_outer prev, p =>
      AllDifferentiable prev p ∧ DifferentiableAt ℝ B_outer p ∧ prev.state p ≠ 0

/-- **Differentiability of the state, by structural induction on the tower.**
    Each constructor case is `DifferentiableAt.div`; the inductive hypothesis
    supplies differentiability of the sub-tower's state, so no level's state
    differentiability is assumed. -/
theorem allDifferentiable_state (t : BulkTower) (p : ℝ)
    (h : AllDifferentiable t p) : DifferentiableAt ℝ t.state p := by
  induction t with
  | base e =>
      obtain ⟨hB, hU, hnz⟩ := h
      exact hB.div hU hnz
  | stepB prev U_outer ih =>
      obtain ⟨hprev, hU, hnz⟩ := h
      exact (ih hprev).div hU hnz
  | stepU B_outer prev ih =>
      obtain ⟨hprev, hB, hnz⟩ := h
      exact hB.div (ih hprev) hnz

/-- `AllDifferentiable` discharges the three flat hypotheses of
    `tower_bulk_differential` / `tower_stationary_phase`. The `stepB` case is
    where the induction earns its keep: `t.flatten.B = prev.state`, and its
    differentiability comes from `allDifferentiable_state`, not an assumption. -/
theorem allDifferentiable_flatten (t : BulkTower) (p : ℝ)
    (h : AllDifferentiable t p) :
    DifferentiableAt ℝ t.flatten.B p ∧ DifferentiableAt ℝ t.flatten.U p ∧
      t.flatten.U p ≠ 0 := by
  cases t with
  | base e => exact h
  | stepB prev U_outer =>
      obtain ⟨hprev, hU, hnz⟩ := h
      exact ⟨allDifferentiable_state prev p hprev, hU, hnz⟩
  | stepU B_outer prev =>
      obtain ⟨hprev, hB, hnz⟩ := h
      exact ⟨hB, allDifferentiable_state prev p hprev, hnz⟩

/-- Leaf-level nonvanishing: every `B` and `U` fed into the tower is nonzero
    at `p`. Strictly stronger than what differentiability alone needs (a
    `stepB` numerator only matters if an enclosing `stepU` makes it a
    denominator); sufficient, not necessary. -/
def AllNonzero : BulkTower → ℝ → Prop
  | base e, p => e.B p ≠ 0 ∧ e.U p ≠ 0
  | stepB prev U_outer, p => AllNonzero prev p ∧ U_outer p ≠ 0
  | stepU B_outer prev, p => B_outer p ≠ 0 ∧ AllNonzero prev p

/-- **Nonvanishing propagates to every level's state.** Induction on the
    tower; each case is `div_ne_zero`. -/
theorem allNonzero_state_ne_zero (t : BulkTower) (p : ℝ)
    (h : AllNonzero t p) : t.state p ≠ 0 := by
  induction t with
  | base e => exact div_ne_zero h.1 h.2
  | stepB prev U_outer ih => exact div_ne_zero (ih h.1) h.2
  | stepU B_outer prev ih => exact div_ne_zero h.1 (ih h.2)

/-- Leaf-level differentiability alone: every `B` and `U` fed into the tower
    is differentiable at `p`. No nonvanishing content. -/
def LeafDifferentiable : BulkTower → ℝ → Prop
  | base e, p =>
      DifferentiableAt ℝ e.B p ∧ DifferentiableAt ℝ e.U p
  | stepB prev U_outer, p =>
      LeafDifferentiable prev p ∧ DifferentiableAt ℝ U_outer p
  | stepU B_outer prev, p =>
      LeafDifferentiable prev p ∧ DifferentiableAt ℝ B_outer p

/-- **Leaf data suffices.** Leaf differentiability plus leaf nonvanishing
    imply `AllDifferentiable`; in particular the `stepU` clause's
    `prev.state p ≠ 0` is derived here (`allNonzero_state_ne_zero`), not
    assumed. -/
theorem allDifferentiable_of_leaf_nonzero (t : BulkTower) (p : ℝ)
    (hd : LeafDifferentiable t p) (hz : AllNonzero t p) :
    AllDifferentiable t p := by
  induction t with
  | base e => exact ⟨hd.1, hd.2, hz.2⟩
  | stepB prev U_outer ih => exact ⟨ih hd.1 hz.1, hd.2, hz.2⟩
  | stepU B_outer prev ih =>
      exact ⟨ih hd.1 hz.2, hd.2, allNonzero_state_ne_zero prev p hz.2⟩

/-- **Quotient-rule decomposition under the inductive hypothesis.** Same
    conclusion as `tower_bulk_differential`, but the hypothesis walks the
    tower: `AllDifferentiable t p` is checked at every level down to the
    base, and its `stepB`/`stepU` clauses carry `AllDifferentiable prev p`
    as first conjunct — so the decomposition holds of every sub-tower of a
    hypothesised tower via `.1`, by this same theorem. -/
theorem tower_bulk_differential_of_allDifferentiable
    (t : BulkTower) (p : ℝ) (h : AllDifferentiable t p) :
    deriv t.state p =
      (t.flatten.U p * deriv t.flatten.B p -
       t.flatten.B p * deriv t.flatten.U p) / (t.flatten.U p)^2 := by
  obtain ⟨hB, hU, hnz⟩ := allDifferentiable_flatten t p h
  exact tower_bulk_differential t p hB hU hnz

/-- **Stationary phase under the inductive hypothesis.** Same conclusion as
    `tower_stationary_phase`, hypothesis checked at every level down to the
    base. -/
theorem tower_stationary_phase_of_allDifferentiable
    (t : BulkTower) (p : ℝ) (h : AllDifferentiable t p) :
    deriv t.state p = 0 ↔
      t.flatten.U p * deriv t.flatten.B p =
      t.flatten.B p * deriv t.flatten.U p := by
  obtain ⟨hB, hU, hnz⟩ := allDifferentiable_flatten t p h
  exact tower_stationary_phase t p hB hU hnz

end BulkTower
end Sanguine
