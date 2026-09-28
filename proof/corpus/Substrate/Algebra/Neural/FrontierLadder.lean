import Substrate.Algebra.Neural.CountLMFloor

/-!
# The Frontier-Intelligence Ladder — a prose ledger of the rungs, over four native facts

@name What Actually Stands Between Us and Frontier Intelligence
@child We list the steps from the counting-fold up to "frontier intelligence," and for each step
  say honestly whether it is *proven*, *proven-but-narrowly*, or *still open*. The list is writing,
  not proof: the steps are proven in their own files, and this file only names them. What this file
  itself proves is four small facts. The surprise is that almost every mechanical step is proven
  somewhere — and the very top step, "reaching the score a big model reaches means the machine is
  intelligent," is not a theorem at all. Nobody has even written down what "intelligence" would mean
  here. We mark that, instead of pretending it away.

@scientist A proof-obligation ledger over the whole stack, calibrated to `theory/THE_EPISTEMIC_LADDER.md`
  (L0 machine-checked ↦ L1 organizing reading): confidence decays as you climb, and theorem-names must
  not be laundered into capability claims. The five rung-groups (surveyed 2026-07-24) are each discharged
  by real theorems (cited below) IN THEIR OWN MODULES; what remains is a SMALL set of open seams, and one
  absent target. The ledger itself is prose — a hand-assembled list of theorem names, checked for
  completeness by nothing, and no statement in this file ranges over it. The file's own native content is
  four facts: `estimation_gap_zero_at_truth`, `fold_preserves_correctness`, `book_anchors_window`,
  and `parity_underdetermines_target` (rewritten 2026-08-05 from a tautology over `Prop`s into the
  model form: two distinct worlds at the same floor leave a predicate of the world open).

  **L0 — discharged in-corpus (mechanics, genuinely proven).** Each cited theorem is `sorry`/`axiom`-free.
    · Estimation lever forced           — `KneserNey.kn_normalizes`, `continuation_matches_marginal`
    · Training is the EXACT fold optimum — `MirasCountLM.countlm_is_exact_miras`,
                                            `TrainingCollapse.training_is_a_collapse`, `ridgeHead_minimizes_objective`
    · Fold is a ℤ-group (merge/unfold)   — `MirasCountLM.hebb_comm`, `forget_hebb`; `Evolution.evolution_is_the_fold`
    · bits/byte = H(cond) + KL, KL ≥ 0   — `CountLMFloor.cross_eq_ent_add_kl`, `kl_nonneg`, `cross_ge_entropy`
    · Fuzzy far-recall > literal          — `ResonanceRecall.resonance_recall`, `literal_misses`, `content_addressed`
    · Attention = sparse spinor, exact    — `AttentionSpinor.sparse_exact_on_support`, `SparseSelection.sparse_token_exact`
    · Budget = ⌈exp H⌉, ctx-independent   — `SparseAttentionBudget.budget_ceil_ppl_valid`;
                                            `CountLMFloor.seed_tokens_le_recon_budget`, `bulk_budget_scale_invariant`, `bulk_budget_le_card`
    · Semantic/flat recall                — `SpectralRecall.semantic_recall`, `LSHRecall.clean_recall`,
                                            `FinderRecall.finder_recall_is_the_capacity`, `RidgeFiltration.ridge_filtration_law`
    · Controller = x=B/U (δS=δF=loop)     — `UniversalEquation.stationary_phase`, `AffineLoopCollapse.affineLoopEquation_state`,
                                            `FreeEnergyPrinciple.the_free_energy_principle`, `InferenceLoop.inference_is_the_spinor_loop`
    · Greedy collapse IS a theorem + fix  — `SelectionFold.balanced_selects_nothing`, `flaw_breaks_symmetry`;
                                            `SelectIsThePathPastTheMonad.select_is_two_paths_past_the_monad`; `BiasInjectedRecovery.unique_global_min`
    · Settle ⇒ winding-0; rotor ≠ repeat  — `ConvergentSelfWinding.convergent_self_is_not_robust_vortex`, `SpinorRecall.aperiodic_rotor_never_repeats`
    · Reasoning/SAT = one collapse (spec)  — `OneCollapse.one_collapse_recovers_all`, `UniversalSatCollapse.sat_iff_residual_floor`
    · Meaning forced (NAND universality)   — `MeaningIsForced.meaning_is_forced`, `nand_complete`
    · Book: regime-not-size; window=W       — `InitialConditionCovariance.unit_scale_fixed_iff_zero`, `Origin.origin_is_starMap`, `origin_read_equivariant`
    · Scale-covariance (zoom = shape-fixed) — `ObserverScale.bulk_ratio_scale_invariant`; `EncapsulatedRegion.encapsulated_floor` (boundary⇒interior)
    · Belief update = the merge             — `BayesFilter.update_is_merge`

  **The OPEN seams (this is the whole distance).** Each is a corpus-self-declared gap, not hidden:
    1. `kappa_sufficient`  — `CountLMFloor.refine_gap_zero_of_kappa_sufficient` has an UNPROVEN antecedent
       (`hconst`): the resonance floor reaches `H(X|full)` only if the lossy SimHash `κ` is a sufficient
       statistic for the heavy far-set. Same object as `RidgeFiltration`'s un-formalized near-orthogonal
       crosstalk `~√n/d` and `SpectralRecall`'s sketched near-twin. THE load-bearing conditioning seam.
    2. `turing_bridge`     — `CombRealization`: the SAT collapse is constant-fuel only in the bulk's native
       cost model; the exponential is inside the atomic `decode`. "Compute the collapse without enumeration"
       is the self-named open Phase-5.2 frontier. Reasoning-as-collapse is a SPEC, not a sub-exp solver.
    3. `compositional_metatheorem` — `CompositionalLift` proves peephole equivalences; the general
       "composition preserves correctness" metatheorem is self-declared deferred.
    4. `book_anchor_regular` — `Origin`'s concrete regular section `s` (E8→H4) + chain-termination is the
       "unbuilt lattice rung": the weld that makes the global book the anchor the local window reads against.
    5. `kn_attains_kl_floor` — `KneserNey` proves the KN weight is UNIQUE/forced; the quantitative
       `KL(p‖pKN) → 0` (Chen–Goodman minimum) is prose, not a discharged inequality.

  **L1 — the top rung is ABSENT, and its bridge is proven NOT-forced (the honesty core).** A corpus-wide
  grep finds NO def/theorem for `intelligence`, `reasoning`, `AGI`, or `frontier capability` as a target
  predicate. The only formal `capability` is `2^N` = substrate dimension. And the bridge upward is proven
  to NOT hold as a function: `CapabilityAwarenessOrthogonality.capability_orthogonal_awareness` (capability ⟂
  awareness), `NecessaryNotSufficient.subject_predicate_underdetermined` (structure ⇏ subject),
  `AssistantWithoutSelfReference.capable_assistant_without_self_reference` (capability at every tier without
  self-reference). So "reaches `H(X|full)` ⇒ frontier intelligence" is not merely unproven — the adjacent
  bridges are proven orthogonal/underdetermined. The measurable floor is the honest target; equating it
  with intelligence is the interpretive step the algebra refuses.

@engineer This module compiles the ledger: `Obligations` bundles the five open Props; `reaches_conditioning_floor`
  re-exports the REAL `§5` discharge (floor reached ⟺ κ-sufficiency); `parity_underdetermines_target` and
  `the_frontier_distance` compile the L1 gap as model-underdetermination (two distinct worlds reach the
  same floor, so the floor leaves a predicate of the world open — it entails neither it nor its negation).
  Nothing here is `sorry`/`axiom`. The proven L0 rungs live in their own modules (cited, not re-proved).

## What is proved here vs what is read onto the stack

* **Proved (native):** four facts, and they are the whole native content. `estimation_gap_zero_at_truth`
  — `CountLMFloor.klDiv p p = 0`, the gap vanishes at the truth. `fold_preserves_correctness` — a
  predicate preserved by every step is preserved by the whole `List.foldl`, by induction.
  `book_anchors_window` — the bulk budget agrees under any two nonzero zooms.
  `parity_underdetermines_target` — two DISTINCT worlds reaching the same floor leave a predicate of
  the world open, entailed neither way by the floor.
* **Proved (riding):** `reaches_conditioning_floor` and `seam1_residual_bounded` are verbatim
  re-exports of `CountLMFloor.refine_gap_zero_of_kappa_sufficient` and
  `CountLMFloor.refine_gap_le_weight_entropy`. Every L0 row above — `KneserNey.kn_normalizes`,
  `MirasCountLM.countlm_is_exact_miras`, `AttentionSpinor.sparse_exact_on_support`, and the rest —
  is proved in its own module and is neither restated nor strengthened here. A repo citing this
  module as settled law is citing THOSE modules.
* **Open:** that the L0 ledger covers every rung between the architecture and the
  goal. It is a hand-assembled list of theorem names in prose; nothing checks it for completeness,
  and no statement in this file ranges over it. The header no longer claims otherwise — the title
  says "the rung ledger in prose", which is what it is.
* **Retired:** that the underdetermination above is about INTELLIGENCE. It is not: the
  predicate it exhibits is `· = w₁`, and what it proves is that the floor does not pin the world.
  There is no `intelligence` predicate in the corpus to be undetermined, and the L1 gap is that
  absence, not a theorem about it. (Until 2026-08-05 both this and seam 3 were content-free:
  `parity_underdetermines_target` was `(Parity → True) ∧ ¬ (Parity → False)`, true of every inhabited
  `Prop`, and `turing_bridge_reduces_to_decoder` was `Iff.symm` — the first is now the model form
  above, the second is deleted and its content is `Closures.Sat.TuringBridge`.)
* **Open:** the five fields of `Obligations`, kept as opaque `Prop`s and discharged nowhere —
  `kappa_sufficient`, `kn_attains_kl_floor`, `turing_bridge`, `compositional_metatheorem`,
  `book_anchor_regular`. `Obligations` is never instantiated in this file; that is the point.
* **Anchor:** the calibration to `theory/THE_EPISTEMIC_LADDER.md`'s L0/L1 rungs, and
  the L1 claim that a corpus-wide grep finds no `intelligence`, `reasoning`, `AGI` or
  `frontier capability` target predicate. That grep is a measurement of the corpus at one date, not
  a theorem, and `THE_AUTHORING_CONVENTION.md` R8 rules a grep miss is not evidence of absence.
-/

namespace Sanguine
namespace FrontierLadder

/-- **The open obligations — the entire distance from the proven architecture to the measurable floor.**
    Each field is a named proof obligation; the module in its docstring is the one that would discharge it.
    None is discharged here (that is the point: the structure being inhabited is the distance closed).
    Kept as opaque `Prop`s deliberately — a ledger of what is owed, not a re-statement that launders the
    mechanics (`THE_EPISTEMIC_LADDER`: labels don't compose, uncertainty does). -/
structure Obligations where
  /-- The lossy SimHash `κ` is a sufficient statistic for the heavy far-set: `CountLMFloor.refineGap = 0`
      for the ACTUAL code (the unproven `hconst` antecedent; = `RidgeFiltration` near-orthogonal crosstalk). -/
  kappa_sufficient : Prop
  /-- `KL(p ‖ pKN) → 0` as an inequality (Chen–Goodman minimum), beyond the proven uniqueness of the KN weight. -/
  kn_attains_kl_floor : Prop
  /-- The collapse COMPUTES the answer without enumeration. SHARPENED by `Closures.Sat.TuringBridge`:
      the universal step is proven IMPOSSIBLE (`tug_of_war_blocks_universal_step` — OctantConflict's
      shared mixed variable), Horn-SAT (P-complete) IS decoded non-enumeratingly (`horn_fragment_decided`,
      least-model fixpoint in `numVars+1` passes), and the residual is now ONE named inequality =
      classical ETH (`StepSynthesisCost`: a `2^{o(n)}` step-synthesizer on the all-mixed core). Not
      progress on ETH; but no longer a vague frontier — a pinned obstruction + a real decoded fragment. -/
  turing_bridge : Prop
  /-- A general "composition preserves correctness" metatheorem, beyond `CompositionalLift`'s peephole lemmas. -/
  compositional_metatheorem : Prop
  /-- The concrete regular section `s` (`Origin`, E8→H4) that makes the global book the window's anchor. -/
  book_anchor_regular : Prop

/-- **The measurable floor is reached — conditional on the ONE load-bearing seam, `κ`-sufficiency.**
    Re-exports `CountLMFloor.refine_gap_zero_of_kappa_sufficient`: given the far conditional is
    `κ`-measurable (`hconst` — the obligation `kappa_sufficient`), the coverage residual vanishes,
    i.e. conditioning on `local + κ·far` already attains `H(X | full)`. Everything mechanically below
    this rung is discharged in its own module (see the ledger docstring). This isolates the single
    unproven antecedent the measurable capability waits on — the honest floor of the whole stack. -/
theorem reaches_conditioning_floor {m n : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ)) (q0 : Fin n → ℝ)
    (hw1 : ∑ z, w z = 1) (hconst : ∀ z, q z = q0) :
    CountLMFloor.refineGap w q = 0 :=
  CountLMFloor.refine_gap_zero_of_kappa_sufficient w q q0 hw1 hconst

/-- **The top rung, compiled as underdetermination — over WORLDS, because it cannot be done over
    `Prop`s.** As soon as two DISTINCT worlds `w₁ ≠ w₂` both reach the floor, there is a predicate on
    worlds the floor entails neither way: `floor` does not imply it and does not imply its negation.
    The measurable floor is not a complete description of the world that reached it, so nothing about
    a world can be laundered out of "the floor is reached" alone. This mirrors
    `NecessaryNotSufficient.subject_predicate_underdetermined`; `CapabilityAwarenessOrthogonality`
    strengthens it to outright orthogonality.

    Why the quantifier moved: until 2026-08-05 this was `(Parity → True) ∧ ¬ (Parity → False)` over a
    bare `Prop`, which holds for EVERY inhabited proposition and named nothing. No propositional form
    can be repaired, either — classically `∀ P, P ∨ ¬P`, so "the floor forces neither reading" is
    outright FALSE when the readings are `Prop`s. Underdetermination is a statement about models, and
    this is it.

    What it does NOT say: which predicate. The witness is `· = w₁`, an identity predicate, not
    "intelligence" — the corpus has no `intelligence` predicate to be undetermined, and that absence
    is the module's headline and stays an absence. -/
theorem parity_underdetermines_target {W : Type} (floor : W → Prop) (w₁ w₂ : W)
    (hne : w₁ ≠ w₂) (h₁ : floor w₁) (h₂ : floor w₂) :
    ∃ target : W → Prop, (¬ ∀ w, floor w → target w) ∧ (¬ ∀ w, floor w → ¬ target w) := by
  refine ⟨fun w => w = w₁, fun h => ?_, fun h => ?_⟩
  · have hEq : w₂ = w₁ := h w₂ h₂
    exact hne hEq.symm
  · have hNe : ¬ (w₁ = w₁) := h w₁ h₁
    exact hNe rfl

/-- **The frontier distance, in one statement.** Given `κ`-sufficiency (obligation 1), the architecture
    reaches the measurable conditioning floor (`refineGap = 0`) — AND, since a second, DIFFERENT
    conditional `q' ≠ q` reaches the same floor, the floor leaves a predicate of the conditional open:
    it entails neither that predicate nor its negation. The proven mechanics get us to the floor; the
    floor does not pin the object that reached it, let alone "intelligence", for which the corpus has
    no predicate at all. Exactly what stands between here and the goal: the five `Obligations`, and the
    fact that the goal itself is not yet a predicate the corpus can state.

    The second configuration is a real hypothesis, not decoration: one floor-reaching world determines
    nothing either way, and it is the SECOND one that makes the floor's silence a theorem. -/
theorem the_frontier_distance {m n : ℕ} (w : Fin m → ℝ) (q q' : Fin m → (Fin n → ℝ))
    (q0 q0' : Fin n → ℝ) (hw1 : ∑ z, w z = 1)
    (hconst : ∀ z, q z = q0) (hconst' : ∀ z, q' z = q0') (hne : q ≠ q') :
    CountLMFloor.refineGap w q = 0
      ∧ ∃ target : (Fin m → (Fin n → ℝ)) → Prop,
          (¬ ∀ p, CountLMFloor.refineGap w p = 0 → target p)
            ∧ (¬ ∀ p, CountLMFloor.refineGap w p = 0 → ¬ target p) := by
  have hfloor := reaches_conditioning_floor w q q0 hw1 hconst
  have hfloor' := reaches_conditioning_floor w q' q0' hw1 hconst'
  exact ⟨hfloor,
    parity_underdetermines_target (W := Fin m → (Fin n → ℝ))
      (fun p => CountLMFloor.refineGap w p = 0) q q' hne hfloor hfloor'⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE SEAMS, ADDRESSED. Each below is the best honest Lean per seam:
--   1 BOUND (CountLMFloor.refine_gap_le_weight_entropy)  4 CLOSED (abstract)
--   2 PARTIAL (Gibbs equality, easy direction)           5 PARTIAL (abstract weld)
--   3 NOT CLOSED — P-vs-NP-adjacent, and it lives in `Closures.Sat.TuringBridge`, not here.
-- ═══════════════════════════════════════════════════════════════════

/-- **Seam 1 — BOUNDED.** Re-exports `CountLMFloor.refine_gap_le_weight_entropy`: the κ-coverage
    residual is at most the far-attention entropy, `refineGap ≤ H(w) = log(budget)`. The load-bearing
    seam is capped by a measured quantity; a κ that concentrates the far-attention drives it to 0. -/
theorem seam1_residual_bounded {m n : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ))
    (hw : ∀ z, 0 < w z) (hq : ∀ z x, 0 < q z x) (hq1 : ∀ z, ∑ x, q z x = 1)
    (hne : (Finset.univ : Finset (Fin m)).Nonempty) :
    CountLMFloor.refineGap w q ≤ CountLMFloor.entropyNat w :=
  CountLMFloor.refine_gap_le_weight_entropy w q hw hq hq1 hne

/-- **Seam 4 — CLOSED (abstract).** "Small correct steps stay correct together": if a correctness
    predicate `P` is preserved by every step, it is preserved by the whole fold. Compositional
    correctness IS a fold invariant — the general metatheorem `CompositionalLift`'s peephole lemmas
    are instances of. Pure induction, no analysis. -/
theorem fold_preserves_correctness {S α : Type} (P : S → Prop) (step : S → α → S)
    (hstep : ∀ s a, P s → P (step s a)) :
    ∀ (s0 : S), P s0 → ∀ (l : List α), P (l.foldl step s0) := by
  intro s0 h0 l
  induction l generalizing s0 with
  | nil => exact h0
  | cons a t ih => exact ih (step s0 a) (hstep s0 a h0)

/-- **Seam 2 — PARTIAL (Gibbs equality, the direction that matters).** The estimation gap `klDiv`
    is exactly zero at the truth: `klDiv p p = 0`. With `CountLMFloor.kl_nonneg` (KL ≥ 0) this pins
    the truth as the unique zero of the gap, and `KneserNey.continuation_matches_marginal` (proven)
    forces the KN estimate onto the truth's marginal — so `KL → 0` as the estimate hits the truth.
    The asymptotic RATE (Chen–Goodman minimality over an estimator family) is the residual obligation. -/
theorem estimation_gap_zero_at_truth {n : ℕ} (p : Fin n → ℝ) :
    CountLMFloor.klDiv p p = 0 := by
  refine Finset.sum_eq_zero (fun i _ => ?_)
  rcases eq_or_ne (p i) 0 with h | h
  · rw [h]; ring
  · rw [div_self h, Real.log_one, mul_zero]

/-- **Seam 5 — PARTIAL (abstract weld).** The local window budget depends ONLY on the data's shape,
    never the observer's scale: for any two nonzero zooms `r, s`, `bulkBudget (r·p) = bulkBudget (s·p)`.
    So whatever absolute anchor the book (`InitialConditionCovariance.AnchoredInitialRegime`) supplies,
    the window reads the SAME budget — the book's REGIME determines it, the size does not. This is the
    abstract two-scale weld (`§7` scale-covariance ∘ a regular anchor); the concrete regular section
    `s` (`Origin`, E8→H4) that instantiates the anchor is the residual "unbuilt lattice rung". -/
theorem book_anchors_window {n : ℕ} (p : Fin n → ℝ) (r s : ℝ) (hr : r ≠ 0) (hs : s ≠ 0) :
    CountLMFloor.bulkBudget (fun i => r * p i) = CountLMFloor.bulkBudget (fun i => s * p i) := by
  rw [CountLMFloor.bulk_budget_scale_invariant p r hr, CountLMFloor.bulk_budget_scale_invariant p s hs]

-- **Seam 3 — SHARPENED, and NOT HERE.** `turing_bridge_reduces_to_decoder` stood here until
-- 2026-08-05: `(SubExpDecoder SubExpCollapse : Prop) (same_object : SubExpDecoder ↔ SubExpCollapse) :
-- SubExpCollapse ↔ SubExpDecoder := same_object.symm`. That is `Iff.symm` with two `Prop` variables
-- named after the claim — no bridge, no reduction, and no call site in the corpus. Deleted rather
-- than restated: the substantive result is `Closures.Sat.TuringBridge`, which this ledger does not
-- import (a ledger that pulls in the SAT closure stops being a ledger). Read it there:
--   · the obstruction is IMPOSSIBILITY, not openness — `tug_of_war_blocks_universal_step`: no
--     universal progress-preserving step exists (two OctantConflict poles share a mixed variable),
--     which also exposes `CollapseShape.collapseFold_recovers` as VACUOUS as a universal claim;
--   · a P-complete fragment is genuinely decoded non-enumeratingly — `horn_fragment_decided`
--     (`hornDecode` sound+complete, least-model fixpoint in `numVars+1` passes, no `2^` in its call
--     graph), bridged to the corpus 3-SAT `Formula`;
--   · the residual is ONE named inequality = classical ETH — `step_synthesizer_is_sat_oracle`
--     (a sound+complete step synthesizer IS a SAT decider) + `StepSynthesisCost`, the `2^{o(n)}`
--     synthesizer on the all-mixed non-Horn core. That inequality is NOT claimed anywhere; it is the
--     honest floor of the whole stack (P vs NP), pinned to a single object rather than a frontier.

end FrontierLadder
end Sanguine
