import Apeiron.Trichotomy.AnyRing
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Algebra.Order.Ring.Defs
import Mathlib.Tactic

/-!
# Progressive Certified Decision Bounds

Formalization of the progressive certified decision framework for discrete
argmax acceleration (bit-plane, spectral series, and thresholded decoders):
1. **Partial Scores and Tail Bounds**:
   Given partial score `P i` and uniform residual tail bound `T ≥ 0` such that
   `|S i - P i| ≤ T` for all candidates `i`.
2. **Interval Enclosure**:
   True scores `S i` are guaranteed to lie in `[P i - T, P i + T]`.
3. **Argmax Certification (Stopping Rule)**:
   If the score gap between leader `star` and runner-up exceeds `2 * T`
   (`upperBound j < lowerBound star`), then `star` is certified to be the
   unique global argmax of `S`, enabling exact early termination.
4. **Candidate Pruning and Refinement Cost**:
   Candidates satisfying `upperBound j < lowerBound star` can be strictly eliminated.
   The active set cardinality `#Active ≤ #Candidates` bounds the refinement cost.
   When certified, `#Active = 1`.
5. **Separation and Non-Instance**:
   When slack exceeds gap (`2 * T ≥ gap`), the certificate refuses, proving why
   unbounded continuous interval relaxations (e.g. `DualCertificate`) fail.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSectionVars false
set_option linter.unusedSimpArgs false

namespace Nous.Model.CertifiedDecision

variable {R : Type*} [CommRing R] [LinearOrder R] [IsStrictOrderedRing R]

/-- Lower bound on candidate score: `P i - T`. -/
def lowerBound (P : R) (T : R) : R := P - T

/-- Upper bound on candidate score: `P i + T`. -/
def upperBound (P : R) (T : R) : R := P + T

/-- True score is trapped within interval `[lowerBound, upperBound]` when `|S - P| ≤ T`. -/
theorem true_score_in_bounds (S P T : R) (h : |S - P| ≤ T) :
    lowerBound P T ≤ S ∧ S ≤ upperBound P T := by
  dsimp [lowerBound, upperBound]
  rw [abs_le] at h
  constructor <;> linarith

/-- Certified Argmax Preservation:
If candidate `j` has `upperBound P_j T < lowerBound P_star T`, then `S j < S star`
holds unconditionally for the true unobserved scores. -/
theorem certified_argmax_preservation (S_star P_star S_j P_j T : R)
    (h_star : |S_star - P_star| ≤ T)
    (h_j : |S_j - P_j| ≤ T)
    (h_sep : upperBound P_j T < lowerBound P_star T) :
    S_j < S_star := by
  have h1 : lowerBound P_star T ≤ S_star := (true_score_in_bounds S_star P_star T h_star).1
  have h2 : S_j ≤ upperBound P_j T := (true_score_in_bounds S_j P_j T h_j).2
  exact lt_of_le_of_lt h2 (lt_of_lt_of_le h_sep h1)

/-- Strict argmax preservation via explicit interval bounds. -/
theorem certified_strict_argmax (S_star P_star S_j P_j T : R)
    (h_star : lowerBound P_star T ≤ S_star)
    (h_j : S_j ≤ upperBound P_j T)
    (h_sep : upperBound P_j T < lowerBound P_star T) :
    S_j < S_star :=
  lt_of_le_of_lt h_j (lt_of_lt_of_le h_sep h_star)

/-- Gap Condition Equivalence:
The separation condition `upperBound P_j T < lowerBound P_star T` is identically
equivalent to `2 * T < P_star - P_j`. -/
theorem certified_gap_condition (P_star P_j T : R) :
    upperBound P_j T < lowerBound P_star T ↔ 2 * T < P_star - P_j := by
  dsimp [lowerBound, upperBound]
  constructor <;> intro h <;> linarith

/-- Pruning Elimination:
Any candidate `j` with `P_star - P_j > 2 * T` cannot be the argmax. -/
theorem candidate_prunable (P_star P_j T : R) (hgap : 2 * T < P_star - P_j) :
    upperBound P_j T < lowerBound P_star T := by
  dsimp [lowerBound, upperBound] at *
  linarith

/-! ## 2. Active Set Cardinality and Refinement Cost -/

variable {α : Type*} [DecidableEq α]

/-- The active candidate set at stage `m`: candidates whose upper bound meets or
exceeds the leader's lower bound. -/
def activeCandidates (P : α → R) (T : R) (star : α) (candidates : Finset α) : Finset α :=
  candidates.filter (fun j => lowerBound (P star) T ≤ upperBound (P j) T)

/-- Refinement cost bound: the active set size is bounded by the total candidate count. -/
theorem activeCandidates_card_le (P : α → R) (T : R) (star : α) (candidates : Finset α) :
    (activeCandidates P T star candidates).card ≤ candidates.card :=
  Finset.card_filter_le candidates _

/-- Leader membership: if `T ≥ 0`, the leader is always in its own active set. -/
theorem leader_mem_activeCandidates (P : α → R) (T : R) (star : α) (candidates : Finset α)
    (hmem : star ∈ candidates) (hT : 0 ≤ T) :
    star ∈ activeCandidates P T star candidates := by
  rw [activeCandidates, Finset.mem_filter]
  refine ⟨hmem, ?_⟩
  dsimp [lowerBound, upperBound]
  linarith

/-- Optimal early exit: when all rivals are separated (`gap > 2 * T`),
the active set shrinks to a singleton `{star}`, giving `O(1)` verification cost. -/
theorem activeCandidates_singleton_of_certified (P : α → R) (T : R) (star : α)
    (candidates : Finset α) (hmem : star ∈ candidates) (hT : 0 ≤ T)
    (h_sep : ∀ j ∈ candidates, j ≠ star → upperBound (P j) T < lowerBound (P star) T) :
    activeCandidates P T star candidates = {star} := by
  ext x
  rw [activeCandidates, Finset.mem_filter, Finset.mem_singleton]
  constructor
  · rintro ⟨hx, hle⟩
    by_contra hne
    have hlt := h_sep x hx hne
    linarith
  · rintro rfl
    refine ⟨hmem, ?_⟩
    dsimp [lowerBound, upperBound]
    linarith

/-! ## 3. Non-Instance and Refusal Condition -/

/-- Refusal Condition:
When the score gap does not strictly exceed `2 * T`, the candidate cannot be pruned.
This formalizes the operational failure of continuous dual-number certificates (E4). -/
theorem unprunable_when_slack_le (P_star P_j T : R) (h : P_star - P_j ≤ 2 * T) :
    lowerBound P_star T ≤ upperBound P_j T := by
  dsimp [lowerBound, upperBound]
  linarith

end Nous.Model.CertifiedDecision
