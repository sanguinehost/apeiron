import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Substrate.Collapse

/-!
# Bias-Injected Recovery: The Shared Structure Behind SAT and Folding
@name The Trick Behind Solving Anything
@child Lots of hard problems share one trick: shape a landscape that's flat everywhere except a single lowest point — the answer — then just roll downhill to it. SAT and protein folding are the same trick wearing different clothes.
@scientist The abstract pattern the bulk realises for any inverse problem with a unique answer hidden in a degenerate landscape: a non-negative residual on Fin n → ℝ, a target where it vanishes, and strict positivity everywhere else — the topological-flaw injection (Axiom V) that breaks the symmetry. The target is then the unique global minimum.
@engineer From those three properties: the target is the unique global minimiser of the residual, and residual = 0 pins it exactly. SAT recovery and protein folding are two instances of this single shape.

The abstract pattern that the framework's bulk mechanism realises for any
inverse problem with a unique answer hidden inside a degenerate landscape:

  1. A non-negative residual function on `Fin n → ℝ`.
  2. A target point where the residual vanishes.
  3. The residual is strictly positive at every other point — the
     topological-flaw injection that breaks the symmetry.

From those three properties, the target is the **unique global minimum**
of the residual, and the residual-equals-zero predicate pins the target
exactly.

This is the common shape behind:
* `MultiParamCanonical.biasedResidual` — 3-SAT with the canonical-SAT
  quadratic regularizer.
* `ProteinFoldingGeneralized.biasedFoldingResidual` — protein folding
  with the native-fold quadratic regularizer.
* `SanguineEncryption.injectedTower` — encryption with the parameter-
  perturbation bias.

The instances live in their respective files (`MultiParamCanonical.lean`,
`ProteinFoldingGeneralized.lean`); this file states the abstract structure
and the headline `unique_global_min` / `zero_iff_target` theorems that
follow from the three structural properties alone.

## What is proved here vs what is read onto the instances

* **Proved (native):** from the three structural fields alone, the target is a global minimum,
  is the *only* point attaining the floor, is the only zero, and the residual is positive
  exactly off it — `global_min`, `unique_global_min`, `zero_iff_target`,
  `residual_pos_iff_ne_target`. Over an arbitrary `D`, with no order, topology, or continuity
  used. `quadraticAtZero` proves the structure non-empty without any SAT or folding machinery.
* **Proved (riding):** the residual face imports the key face — `toSeed_decides` is
  `Collapse.Seed.seedEq_correct` applied with `toSeed_canonical`, which is itself
  `zero_iff_target` on both sides.
* **Anchor:** that SAT recovery, protein folding, and a Bitcoin nonce are
  *instances* of this shape. The named instances — `MultiParamCanonical.biasedResidual`,
  `ProteinFoldingGeneralized.biasedFoldingResidual`, `SanguineEncryption.injectedTower` — exist,
  but nothing in this file constructs a `BiasInjectedRecovery` from any of them, so "the same
  collapse, not three" is stated here and discharged elsewhere or not at all.
* **Anchor:** that "just roll downhill to it" recovers the target. The theorems
  characterise the unique zero of a landscape; no descent, dynamics, search, or cost appears in
  any statement.
* **Open:** the header's "a non-negative residual function on `Fin n → ℝ`" —
  the structure is over an arbitrary `D`, so the prose understates its own generality. That
  gap runs the opposite direction to the usual one and is still a gap; restating the header
  over the arbitrary domain `D` would close it.

-/

noncomputable section

namespace Sanguine

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE ABSTRACT STRUCTURE
-- ═══════════════════════════════════════════════════════════════════

/-- A bias-injected recovery setup over an arbitrary domain `D` (the degrees of
    freedom — `Fin n → ℝ` for a continuous landscape, `Fin k → Bool` for a Boolean
    predicate, a `CExpr`, a weight tensor — *any* space with a unique answer).

    The three structural fields encode the framework's Axiom-V
    pattern (active topological-flaw injection):
    * `nonneg`: the landscape is bounded below by zero.
    * `zero_at_target`: the chosen target achieves the floor.
    * `strict_positive_off_target`: every other point is strictly
      above the floor — the symmetry-breaking that singles out the
      target as the unique zero.

    The domain is `D`, not `Fin n → ℝ`: the unique-floor theorems below never use the
    continuous structure — they need only `residual : D → ℝ` with a single zero — so
    the collapse is one mechanism over *every* domain (this is why Bitcoin's Boolean
    nonce, a SAT assignment, and a protein conformation are the same collapse, not
    three). -/
structure BiasInjectedRecovery (D : Type) where
  /-- The residual landscape. -/
  residual : D → ℝ
  /-- The recovery target — the point we want stationary phase to find. -/
  target : D
  /-- The residual is non-negative everywhere. -/
  nonneg : ∀ x : D, 0 ≤ residual x
  /-- The residual vanishes at the target. -/
  zero_at_target : residual target = 0
  /-- The residual is strictly positive at every other point. -/
  strict_positive_off_target : ∀ x : D, x ≠ target → 0 < residual x

namespace BiasInjectedRecovery

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE HEADLINE: UNIQUE GLOBAL MINIMUM AT THE TARGET
-- ═══════════════════════════════════════════════════════════════════

/-- **The target is a global minimum.** Combining `zero_at_target` and
    `nonneg`, every point's residual is ≥ the target's. -/
theorem global_min {D : Type} (S : BiasInjectedRecovery D) (x : D) :
    S.residual S.target ≤ S.residual x := by
  rw [S.zero_at_target]
  exact S.nonneg x

/-- **The unique-global-min headline.** A point achieves the target's
    residual value iff it equals the target.

    For any candidate `x`, either `x = target` (and `residual x = 0`
    by `zero_at_target`), or `x ≠ target` (and `residual x > 0` by
    `strict_positive_off_target`, hence cannot equal the target's `0`).

    This is the bulk-Axiom-V "unique stationary-phase collapse" property
    in its most distilled form: the target is the ONLY point achieving
    the residual floor. -/
theorem unique_global_min {D : Type} (S : BiasInjectedRecovery D) (x : D) :
    S.residual x = S.residual S.target ↔ x = S.target := by
  rw [S.zero_at_target]
  constructor
  · intro h_eq
    by_contra h_ne
    have h_pos := S.strict_positive_off_target x h_ne
    linarith
  · intro h_x
    rw [h_x]
    exact S.zero_at_target

/-- **Equivalent restatement.** Residual vanishes at exactly one point —
    the target. -/
theorem zero_iff_target {D : Type} (S : BiasInjectedRecovery D) (x : D) :
    S.residual x = 0 ↔ x = S.target := by
  have h := unique_global_min S x
  rw [S.zero_at_target] at h
  exact h

/-- **Contrapositive form.** If `x ≠ target`, the residual is strictly
    positive (re-exporting the structural field as an `iff`-friendly
    theorem). -/
theorem residual_pos_iff_ne_target {D : Type} (S : BiasInjectedRecovery D)
    (x : D) :
    0 < S.residual x ↔ x ≠ S.target := by
  constructor
  · intro h_pos h_eq
    rw [h_eq, S.zero_at_target] at h_pos
    exact lt_irrefl 0 h_pos
  · exact S.strict_positive_off_target x

-- ═══════════════════════════════════════════════════════════════════
-- §3. SANITY INSTANCE: THE QUADRATIC AT THE ORIGIN
-- ═══════════════════════════════════════════════════════════════════

/-- The minimal sanity instance: residual `∑ k x_k²`, recovers `0`.
    Shows the abstract structure is non-empty without depending on
    SAT or folding machinery. -/
def quadraticAtZero (n : ℕ) : BiasInjectedRecovery (Fin n → ℝ) where
  residual := fun x => ∑ k : Fin n, (x k)^2
  target := fun _ => 0
  nonneg := fun x => Finset.sum_nonneg (fun i _ => sq_nonneg (x i))
  zero_at_target := by
    show ∑ k : Fin n, ((fun (_ : Fin n) => (0 : ℝ)) k)^2 = 0
    simp
  strict_positive_off_target := by
    intro x h_ne
    have h_exists : ∃ k : Fin n, x k ≠ 0 := by
      by_contra h_all_zero
      apply h_ne
      funext k
      by_contra h_k_ne
      apply h_all_zero
      exact ⟨k, h_k_ne⟩
    obtain ⟨k, hk⟩ := h_exists
    have h_k_pos : 0 < (x k)^2 := by positivity
    have h_rest_nonneg : 0 ≤ ∑ j ∈ Finset.univ.erase k, (x j)^2 :=
      Finset.sum_nonneg (fun j _ => sq_nonneg _)
    have h_split : ∑ k : Fin n, (x k)^2 =
        (x k)^2 + ∑ j ∈ Finset.univ.erase k, (x j)^2 := by
      rw [← Finset.sum_erase_add _ _ (Finset.mem_univ k)]
      linarith
    linarith

/-- The quadratic sanity instance recovers `0`. -/
theorem quadraticAtZero_target (n : ℕ) :
    (quadraticAtZero n).target = (fun _ => (0 : ℝ)) := rfl

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE TWO FACES ARE ONE — a recovery induces a `Collapse.Seed`
-- ═══════════════════════════════════════════════════════════════════
-- DONE (third face proven, not a hole). The collapse trinity: key (`Seed`,
-- CollapseCore) · fold (`foldKey`/`streamFold`, Collapse) · residual (this
-- file = the B/U recovery face). §4 is not the missing third face — it is the
-- arrow that completes the fuse of the residual face onto the key face.
-- `Substrate/Collapse`'s docstring claims `Seed` is the most general of the
-- collapse's faces and lists the residual face beside it as "the same idea".
-- This section makes that a theorem: every `BiasInjectedRecovery` induces a
-- `Seed` whose key is the floor-verdict, and `zero_iff_target` discharges the
-- root's `hcanon` — so the residual face *imports* the key face, and the
-- corpus's "one collapse" is an arrow, not a slogan.

open scoped Classical in
/-- **The Seed a recovery induces.** The canonical key of a point is its
    floor-verdict — "does the residual rest at zero here?". One Boolean per
    point, read off the landscape `S` alone. -/
noncomputable def toSeed {D : Type} (S : BiasInjectedRecovery D) :
    Collapse.Seed D Bool :=
  ⟨fun x => decide (S.residual x = 0)⟩

open scoped Classical in
/-- **The key is canonical — `hcanon`, discharged by `zero_iff_target`.** Two
    points share the floor-verdict iff they sit on the same side of the floor,
    and the floor side is exactly "= target" (the unique zero). So the root's
    `seedEq_correct` applies: the induced `seedEq` decides target-equivalence,
    and the residual face is a `Seed` instance, not a sibling primitive. -/
theorem toSeed_canonical {D : Type} (S : BiasInjectedRecovery D) (a b : D) :
    (S.toSeed.key a = S.toSeed.key b) ↔ ((a = S.target) ↔ (b = S.target)) := by
  simp only [toSeed, decide_eq_decide]
  rw [S.zero_iff_target a, S.zero_iff_target b]

open scoped Classical in
/-- **The induced `seedEq` decides target-equivalence** — the root's generic
    `Seed.seedEq_correct` applied with `hcanon := toSeed_canonical`. The one
    collapse: recovery (residual face) = key comparison (Seed face). -/
theorem toSeed_decides {D : Type} (S : BiasInjectedRecovery D) (a b : D) :
    S.toSeed.seedEq a b = true ↔ ((a = S.target) ↔ (b = S.target)) :=
  Collapse.Seed.seedEq_correct S.toSeed
    (fun a b => (a = S.target) ↔ (b = S.target)) (toSeed_canonical S) a b

end BiasInjectedRecovery
end Sanguine
