import Substrate.Algebra.Neural.TrainingCollapse
import Substrate.Algebra.Security.HashTrinityCore
import Substrate.Algebra.NullCell.Opportunity
import Substrate.Collapse
import Mathlib.Data.List.Sort
import Mathlib.Tactic

/-!
# The ML Primitives Bridge — libtorch's load-bearing ops are `x = B/U` operations
@name What PyTorch Is Really Doing
@child The thousands of operations in a popular AI toolkit turn out to be a handful of moves wearing different hats. This shows the big ones are really the same one rule.
@scientist libtorch's load-bearing ops (matmul, softmax, flash-attention, sort, Bellman, the quant floor) collapse onto x = B/U (state = Boundary over Undifferentiated-drag) and the accumulator fold; some rows are algebra, some are labelled readings.
@engineer `contraction_is_fold` (matmul), `softmax_is_bulk`/`softmax_shift`, `osFuse_is_add` (flash-attention tiling), `sorted_is_unique_floor` (sort), `bellman_fixed_point_is_monad` (RL), `output_error_is_function_distortion`/`aware_scale_is_the_chart`/`weight_only_is_dual_blind` (quantization — the floor's lever is the dual's precision, α=½ forced).

The PyTorch/libtorch op surface (2,691 registry ops; 1,024 pure compositions with no kernel;
546 pointwise + 103 reduction, all through ONE iterator + a lambda; every convolution lowered to
im2col + gemm; 686 backward entries, each a formula of forward ops) collapses onto the
framework's standing primitives. This file proves the identifications that are algebra and
labels the ones that are readings.

## The rows

1. **Contraction is the B/U accumulator fold.** A matmul entry is a commutative-monoid sum of
   products (`contraction_is_fold`); the blocked form is `TrainingCollapse.gram_stack`.
2. **Softmax is `x = B/U` at the drag `U = Σexp`** (`softmax_is_bulk`), it normalizes
   (`softmax_sum_one`), and the max-shift is the monad cancelling, exact (`softmax_shift`).
3. **Flash-attention tiling is the accumulator merge in a chart.** The online-softmax pair
   `(m, s)` is a chart on `s·e^m`; the tile merge read through it is plain `+` (`osFuse_is_add`),
   so regrouping is exact by ℝ's `add_assoc` (`osFuse_grouping_free`) — the
   `HashTrinity.fusion_is_commutative_monoid` law at the log-sum-exp chart.
4. **Sort is collapse to the canonical floor.** The sorted form is the canonical key of the
   permutation orbit (`sort_key_classifies`), its unique floor (`sorted_is_unique_floor`), and a
   `Collapse.Seed` instance (`sortSeed_decides_perm`). The Coord face's native gradient; topk =
   the B/U accumulator at the max-monoid.
5. **The Bellman fixed point is the monad.** TD ≡ 0 iff the return field is conservative
   (`bellman_fixed_point_is_monad`), a converged value extracts nothing
   (`converged_extracts_nothing`), the advantage captured per loop equals the injected flaw
   (`advantage_equals_flaw`), and over a cost-bearing field the gain is `δ − drag`
   (`advantage_minus_drag`, the KL-penalty / λ-floor shape) — `Opportunity`'s theorems in RL
   names.
6. **The quant floor reads the dual.** The output error a quantized layer feels is the
   precision-weighted residual `Σ τ_j·r_j²`, τ the calib stream's channel energy
   (`output_error_is_function_distortion`); `s = √τ` is the chart that flattens it — AWQ's
   α = ½ is forced, not tuned (`aware_scale_is_the_chart`); and no function of W alone can
   read τ (`weight_only_is_dual_blind`) — the measured "no weight-only floor crosses NF4"
   (`theory/SEED_SPEC.md`) is this algebra, not an execution gap.

## Reused, not re-proved

* the forward-only gradient/curvature orbits (`QuantumClassicalIsomorphism`) and the one-step
  quadratic landing (`TrainingCollapse.newton_step_lands`);
* the solve (`TrainingCollapse.ridgeHead_unique`), the streaming fold (`gram_stack`),
  weight-as-repetition (`gram_replicate`);
* embedding/gather is the dictionary GET (`HashTrinity.map_lookup`), scatter-add the B/U merge
  (`count_two`); dispatch (`HardwareProjectionDispatch.dispatchHP_eq_K_state`) and write
  reordering (`StateVectorMemoryModel.store_store_swap`).

Proved (Mathlib): the sort row's `mergeSort` wrappers and the contraction row's matrix
`mul_apply` are Mathlib's statements, restated under framework names and not re-derived here.
Interpreted (bridge): ATen's dispatch-key table read as an HP
marker table, the value function as the potential `ψ`; float's
broken `add_assoc` is the genuine numerics residue — the chart law is exact over ℝ, fp rounding a
lossy applicative whose residual Kahan-style carrying stores (the ③ Gen face).

Operational forms in `Runtime/Model.lean §5d`; catalogue in `theory/THE_HASHMAP_PATH.md`.

## Which rows are algebra and which are the reading

* **Proved (native):** the algebra of every row — `softmax_is_bulk`, `softmax_sum_one`,
  `softmax_shift`; `osFuse_comm`, `osFuse_is_add`, `osFuse_grouping_free`; `sortSeed_decides_perm`;
  `bellman_fixed_point_is_monad`, `converged_extracts_nothing`; `output_error_is_function_distortion`,
  `aware_scale_is_the_chart`, `weight_only_is_dual_blind`; `kn_floor_refinement`; and the bundle
  `ml_primitives_bridge`.
* **Proved (Mathlib):** `contraction_is_fold` IS `Matrix.mul_apply` — the statement is Mathlib's,
  restated under a framework name; `sort_key_classifies` and `sorted_is_unique_floor` assemble
  `List.mergeSort_perm` and `List.Perm.eq_of_pairwise'`.
* **Proved (riding):** the RL row's content is `Opportunity`'s, instantiated —
  `Opportunity.monad_is_sterile`, `Opportunity.captured_equals_flaw`,
  `Opportunity.captured_minus_drag`; the `Seed` correctness is `Collapse.Seed.seedEq_correct`; the
  reused solve/fold/dispatch are `TrainingCollapse.ridgeHead_unique`, `TrainingCollapse.gram_stack`,
  `TrainingCollapse.gram_replicate`, `TrainingCollapse.newton_step_lands`,
  `HashTrinity.map_lookup`, `HashTrinity.count_two`,
  `HardwareProjectionDispatch.dispatchHP_eq_K_state`, `StateVectorMemoryModel.store_store_swap`.
* **Anchor:** that these theorems ARE libtorch. Nothing here reads the PyTorch
  registry, so "matmul", "flash-attention tiling", "sort", "Bellman", "AWQ" naming `contraction_is_fold`,
  `osFuse_is_add`, `sort_key_classifies`, `bellman_fixed_point_is_monad`, `aware_scale_is_the_chart`
  are identifications, not equalities the kernel checked. `advantage_equals_flaw`'s SGD/GAN remark
  labels itself the same way in place.
* **Anchor:** the op-surface census — 2,691 registry ops, 1,024 pure compositions,
  546 pointwise + 103 reduction, 686 backward entries, "64% composite". Counts of another
  codebase, quoted here; no theorem establishes them.
* **Anchor:** the measured floor and finetune numbers — `theory/SEED_SPEC.md`'s
  "no weight-only floor crosses NF4" and GATE B 3.914 < 3.9605; and §6b's spinor-ttl race (Qwen2.5
  -0.5B, 5.9 MB Lean domain, 8192 B held-out: base 0.910 b/B, LoRA r=16 0.903, fold⊕base 0.724).
  `weight_only_is_dual_blind` proves a weight-only seal cannot READ `τ`; it does not measure NF4,
  and `kn_floor_refinement` proves mass conservation, not a bits/byte.
* **Open:** float. The chart laws (`osFuse_is_add`, `osFuse_grouping_free`) are exact over ℝ, and
  fp `add_assoc` is broken — the genuine numerics residue, named in the prose above and not closed
  by any theorem in this file.

-/

noncomputable section

namespace Sanguine
namespace MLPrimitivesBridge

open Matrix

-- ═══════════════════════════════════════════════════════════════════
-- §1. CONTRACTION IS THE B/U ACCUMULATOR FOLD
-- ═══════════════════════════════════════════════════════════════════

/-- **A matmul entry is the accumulator fold** — a commutative-monoid sum of products. The
    order-freedom that licenses blocking is `Finset.sum`'s commutativity; the blocked form is
    `TrainingCollapse.gram_stack`. Every dense layer, attention score, and post-im2col
    convolution is this one merge. -/
theorem contraction_is_fold {n m p : ℕ} (A : Matrix (Fin n) (Fin m) ℝ)
    (B : Matrix (Fin m) (Fin p) ℝ) (i : Fin n) (j : Fin p) :
    (A * B) i j = ∑ k, A i k * B k j :=
  Matrix.mul_apply

-- ═══════════════════════════════════════════════════════════════════
-- §2. SOFTMAX IS `x = B/U` AT THE NAMED DRAG `U = Σexp`
-- ═══════════════════════════════════════════════════════════════════

/-- Softmax — the boundary `e^{xᵢ}` over the drag `U = Σⱼ e^{xⱼ}`. -/
def softmax {n : ℕ} (x : Fin n → ℝ) (i : Fin n) : ℝ :=
  Real.exp (x i) / ∑ j, Real.exp (x j)

/-- **Softmax IS the bulk ratio** — definitionally `B/U` with the drag the reduction `Σexp`. -/
theorem softmax_is_bulk {n : ℕ} (x : Fin n → ℝ) (i : Fin n) :
    softmax x i = Real.exp (x i) / ∑ j, Real.exp (x j) := rfl

/-- **Softmax normalizes** — summing the boundary against its own drag is `U/U = 1`, the monad
    `b/b = 1`. -/
theorem softmax_sum_one {n : ℕ} (hn : 0 < n) (x : Fin n → ℝ) :
    ∑ i, softmax x i = 1 := by
  haveI : Nonempty (Fin n) := ⟨⟨0, hn⟩⟩
  have hpos : 0 < ∑ j, Real.exp (x j) :=
    Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty
  simp only [softmax]
  rw [← Finset.sum_div, div_self hpos.ne']

/-- **The max-shift trick is exact** — shifting every logit by `c` changes nothing: the shift
    enters boundary and drag as the same factor `e^c` and cancels, `B·g / U·g = B/U`. The bulk's
    scale invariance, not an approximation. -/
theorem softmax_shift {n : ℕ} (x : Fin n → ℝ) (c : ℝ) :
    softmax (fun i => x i + c) = softmax x := by
  funext i
  simp only [softmax, Real.exp_add, ← Finset.sum_mul]
  rw [mul_div_mul_right _ _ (Real.exp_ne_zero c)]

-- ═══════════════════════════════════════════════════════════════════
-- §3. FLASH-ATTENTION TILING IS THE ACCUMULATOR MERGE IN A CHART
-- ═══════════════════════════════════════════════════════════════════

/-- The online-softmax state: a pair `(m, s)` — running max and rescaled exp-sum. A chart on the
    single number `s·e^m` (the un-normalized drag), chosen so `exp` never overflows. -/
def osFuse (a b : ℝ × ℝ) : ℝ × ℝ :=
  (max a.1 b.1,
   a.2 * Real.exp (a.1 - max a.1 b.1) + b.2 * Real.exp (b.1 - max a.1 b.1))

/-- What the chart denotes: `(m, s) ↦ s·e^m`. -/
def osVal (a : ℝ × ℝ) : ℝ := a.2 * Real.exp a.1

/-- The merge is symmetric. -/
theorem osFuse_comm (a b : ℝ × ℝ) : osFuse a b = osFuse b a := by
  simp [osFuse, max_comm, add_comm]

/-- **The tile merge IS addition, read through the chart.** The online-softmax recombination
    `(m₁,s₁) ⊕ (m₂,s₂) = (m, s₁e^{m₁−m}+s₂e^{m₂−m})` denotes exactly `s₁e^{m₁} + s₂e^{m₂}` — the
    plain B/U accumulator `+`. The chart is data; the merge never left the monoid. -/
theorem osFuse_is_add (a b : ℝ × ℝ) : osVal (osFuse a b) = osVal a + osVal b := by
  simp only [osVal, osFuse, add_mul, mul_assoc, ← Real.exp_add]
  ring_nf

/-- **Tiling is grouping-free** — any bracketing of tile merges denotes the same sum, by ℝ's
    associativity through the chart. The `HashTrinity.fusion_is_commutative_monoid`
    "batch = sequential" law at the log-sum-exp chart: flash-attention's tiling exactness, as a
    theorem. -/
theorem osFuse_grouping_free (a b c : ℝ × ℝ) :
    osVal (osFuse (osFuse a b) c) = osVal (osFuse a (osFuse b c)) := by
  simp [osFuse_is_add, add_assoc]

-- ═══════════════════════════════════════════════════════════════════
-- §4. SORT IS COLLAPSE TO THE CANONICAL FLOOR (a `Seed` instance)
-- ═══════════════════════════════════════════════════════════════════

section Sorting

variable {A : Type} [LinearOrder A]

/-- **The sorted form is the canonical key of the permutation orbit** — two lists have the same
    sorted form iff they are permutations. Sorting is the collapse of the orbit to its unique
    floor (`canonical_floor_unique`'s shape, on lists). -/
theorem sort_key_classifies (l m : List A) :
    l.mergeSort (· ≤ ·) = m.mergeSort (· ≤ ·) ↔ l.Perm m := by
  constructor
  · intro h
    exact (List.mergeSort_perm l _).symm.trans (h ▸ List.mergeSort_perm m _)
  · intro hp
    exact List.Perm.eq_of_pairwise'
      (List.pairwise_mergeSort' (· ≤ ·) l)
      (List.pairwise_mergeSort' (· ≤ ·) m)
      ((List.mergeSort_perm l _).trans (hp.trans (List.mergeSort_perm m _).symm))

/-- **The floor is unique in the orbit** — any sorted permutation of `l` IS `l`'s sorted form.
    Exactly one sorted element per orbit; sorting lands on it from anywhere. (Inversions ≥ 0,
    zero exactly here — the `BiasInjectedRecovery` shape.) -/
theorem sorted_is_unique_floor {l m : List A} (hp : l.Perm m)
    (hm : m.Pairwise (· ≤ ·)) : m = l.mergeSort (· ≤ ·) :=
  List.Perm.eq_of_pairwise' hm (List.pairwise_mergeSort' (· ≤ ·) l)
    (hp.symm.trans (List.mergeSort_perm l _).symm)

/-- **Sort as a `Seed`** — the canonical key of the list space, `key = mergeSort`; its
    canonicity is `sort_key_classifies`. -/
def sortSeed (A : Type) [LinearOrder A] : Collapse.Seed (List A) (List A) :=
  ⟨fun l => l.mergeSort (· ≤ ·)⟩

/-- **The sort Seed decides permutation-equivalence** — `seedEq` on sorted floors is exactly
    `l ~ m`, by `Collapse.Seed.seedEq_correct`. Identity on floors, zero per-pair work: the
    hashmap path's ordering clause, closed. -/
theorem sortSeed_decides_perm [BEq A] [LawfulBEq A] (l m : List A) :
    (sortSeed A).seedEq l m = true ↔ l.Perm m :=
  Collapse.Seed.seedEq_correct (sortSeed A) List.Perm
    (fun a b => sort_key_classifies a b) l m

end Sorting

-- ═══════════════════════════════════════════════════════════════════
-- §5. TRAINING DYNAMICS — THE BELLMAN FIXED POINT IS THE MONAD
-- ═══════════════════════════════════════════════════════════════════

section Dynamics

variable {V : Type*}

/-- **The TD error** — the local failure of the one-step return field `w` to be the coboundary
    of the value potential `ψ`. RL's central residual, the flaw coordinate of `Opportunity`. -/
def tdError (w : V → V → ℝ) (ψ : V → ℝ) (u v : V) : ℝ :=
  w u v - (ψ u - ψ v)

/-- **The Bellman fixed point IS the monad.** A value function with TD error zero everywhere
    exists iff the return field is conservative — the round trip telescopes, `B/(B/U) = U`,
    nothing left to extract. TD convergence is arrival at `Opportunity.Conservative`. -/
theorem bellman_fixed_point_is_monad (w : V → V → ℝ) :
    (∃ ψ : V → ℝ, ∀ u v, tdError w ψ u v = 0) ↔ Opportunity.Conservative w := by
  constructor
  · rintro ⟨ψ, h⟩
    exact ⟨ψ, fun u v => by have := h u v; simp only [tdError] at this; linarith⟩
  · rintro ⟨ψ, h⟩
    exact ⟨ψ, fun u v => by simp [tdError, h u v]⟩

/-- **A converged value extracts nothing** — at the Bellman fixed point no closed walk gains: the
    policy-improvement well is dry exactly when TD ≡ 0. `monad_is_sterile` in the RL name. -/
theorem converged_extracts_nothing {w : V → V → ℝ}
    (h : ∃ ψ : V → ℝ, ∀ u v, tdError w ψ u v = 0) :
    ¬ Opportunity.HasOpportunity w :=
  Opportunity.monad_is_sterile ((bellman_fixed_point_is_monad w).mp h)

/-- **The advantage captured per loop equals the flaw, exactly** — dislocate one edge of a
    converged field by `δ` and the recoverable advantage around the 2-cycle is precisely `δ`:
    linear, zero at the fixed point. `Opportunity.captured_equals_flaw`, instantiated. *(SGD
    corollary, read not proven: autograd's field is a coboundary by construction — 686 derivative
    entries of ONE potential — so `monad_no_circulation` guarantees a floor; two-player gradient
    play breaks it, and the GAN cycling is `HasOpportunity`.)* -/
theorem advantage_equals_flaw [DecidableEq V] {w : V → V → ℝ}
    (hc : Opportunity.Conservative w) (a b : V) (δ : ℝ) (hab : a ≠ b) :
    Opportunity.circ (Opportunity.inject w a b δ) [a, b, a] = δ :=
  Opportunity.captured_equals_flaw hc a b δ hab

/-- **Regularized improvement must clear the drag** — over a cost-bearing field the captured
    advantage is `δ − drag` (`Opportunity.captured_minus_drag`), so an update gains iff the signal
    clears the round-trip cost. The KL-penalized objective's shape (`advantage − β·KL > 0`) and
    `TrainingCollapse.ridge_bias`'s λ-floor, the field's third face: `+` opportunity · `0` monad
    · `−` drag. -/
theorem advantage_minus_drag [DecidableEq V] (w : V → V → ℝ) (a b : V) (δ : ℝ)
    (hab : a ≠ b) :
    Opportunity.circ (Opportunity.inject w a b δ) [a, b, a]
      = δ - Opportunity.drag w a b :=
  Opportunity.captured_minus_drag w a b δ hab

end Dynamics

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE QUANT FLOOR READS THE DUAL — distortion is precision-weighted
-- ═══════════════════════════════════════════════════════════════════
-- A quantized layer's model feels ‖X(W−Ŵ)‖², never ‖W−Ŵ‖². Three facts pin the
-- AWQ phenomenon: the output error IS the precision-weighted residual (τ = the dual
-- stream's channel energy — the Kalman τ of `HashTrinity.mean_is_bulk_ratio`), √τ is
-- the chart that flattens it (α = ½ forced), and no function of W alone can read τ.
-- The measured floor history (`theory/SEED_SPEC.md`: every weight-only floor loses to
-- NF4; the activation-aware seal beats it, GATE B 3.914 < 3.9605) is this algebra.

/-- Function distortion of a per-channel residual `r` under dual channel energy `τ`:
    `D = Σ τ_j·r_j²` — precision-weighted squared prediction error, the Gaussian-limit
    free energy with the measure the function forces. -/
def functionDistortion {n : ℕ} (τ r : Fin n → ℝ) : ℝ := ∑ j, τ j * r j ^ 2

/-- **The output error IS the function distortion.** For a calib stream `X` whose channels
    decouple (Gram diagonal), `‖X·r‖² = Σ τ_j·r_j²` with `τ_j = Σ_t X t j²` the channel
    energy. The forward feels the residual through the dual's measure, not Lebesgue. -/
theorem output_error_is_function_distortion {T n : ℕ} (X : Fin T → Fin n → ℝ)
    (r : Fin n → ℝ) (horth : ∀ j k, j ≠ k → ∑ t, X t j * X t k = 0) :
    ∑ t, (∑ j, X t j * r j) ^ 2
      = functionDistortion (fun j => ∑ t, X t j ^ 2) r := by
  unfold functionDistortion
  calc ∑ t, (∑ j, X t j * r j) ^ 2
      = ∑ t, ∑ j, ∑ k, (X t j * r j) * (X t k * r k) := by
        refine Finset.sum_congr rfl fun t _ => ?_
        rw [sq, Finset.sum_mul_sum]
    _ = ∑ j, ∑ k, (r j * r k) * ∑ t, X t j * X t k := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun k _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun t _ => by ring
    _ = ∑ j, (∑ t, X t j ^ 2) * r j ^ 2 := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [Finset.sum_eq_single j
          (fun k _ hk => by rw [horth j k (Ne.symm hk), mul_zero])
          (fun h => absurd (Finset.mem_univ j) h)]
        have : ∑ t, X t j * X t j = ∑ t, X t j ^ 2 :=
          Finset.sum_congr rfl fun t _ => (sq (X t j)).symm
        rw [this]; ring

/-- **√τ is the chart — AWQ's α = ½ is forced, not tuned.** The precision-weighted
    distortion equals the PLAIN distortion of the `√τ`-scaled residual: quantize `W⊙s`,
    fold `1/s` back, and minimizing ordinary MSE in the scaled chart minimizes the
    function distortion in the true one. -/
theorem aware_scale_is_the_chart {n : ℕ} (τ r : Fin n → ℝ) (hτ : ∀ j, 0 ≤ τ j) :
    functionDistortion τ r = ∑ j, (Real.sqrt (τ j) * r j) ^ 2 := by
  unfold functionDistortion
  refine Finset.sum_congr rfl fun j _ => ?_
  rw [mul_pow, Real.sq_sqrt (hτ j)]

/-- **The weight cannot read the dual (the blind spot).** A seal computed from W alone fixes
    ONE residual `r` for every dual; two duals differ in distortion by exactly
    `Σ (τ−τ')·r_j²`, and any nonzero residual channel admits a dual pushing the distortion
    past any bound. "No weight-only floor crosses NF4" is not an execution gap: the lever
    is τ, and τ is not in W. -/
theorem weight_only_is_dual_blind {n : ℕ} (r : Fin n → ℝ) :
    (∀ τ τ' : Fin n → ℝ, functionDistortion τ r - functionDistortion τ' r
        = ∑ j, (τ j - τ' j) * r j ^ 2)
    ∧ (∀ j₀ : Fin n, r j₀ ≠ 0 → ∀ C : ℝ, ∃ τ : Fin n → ℝ,
        (∀ j, 0 ≤ τ j) ∧ C < functionDistortion τ r) := by
  constructor
  · intro τ τ'
    unfold functionDistortion
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j _ => by ring
  · intro j₀ hr C
    have hr2 : (0 : ℝ) < r j₀ ^ 2 := by positivity
    refine ⟨fun j => if j = j₀ then (|C| + 1) / r j₀ ^ 2 else 0, fun j => ?_, ?_⟩
    · by_cases h : j = j₀ <;> simp only [h, if_true, if_false] <;> positivity
    · unfold functionDistortion
      rw [Finset.sum_eq_single j₀
        (fun k _ hk => by simp [hk])
        (fun h => absurd (Finset.mem_univ j₀) h)]
      simp only [if_true]
      rw [div_mul_cancel₀ _ (ne_of_gt hr2)]
      exact lt_of_le_of_lt (le_abs_self C) (lt_add_one _)

-- ═══════════════════════════════════════════════════════════════════
-- §6b. THE FINETUNE IS A MASS-CONSERVING TRANSFER ONTO THE EVIDENCE
-- ═══════════════════════════════════════════════════════════════════
-- Domain adaptation without gradients: fold the domain into a count table (the
-- B/U accumulator), then refine the BASE model's distribution by one KN
-- absolute-discount lane — the base is the zero-evidence FLOOR, the counts move
-- exactly the discounted mass onto the seen continuations. The two-table merge
-- (global ⊕ ledger) as one theorem. Measured (spinor-ttl finetune race, Qwen2.5
-- -0.5B, 5.9MB Lean-domain, 8192B held-out): base 0.910 b/B · LoRA r=16 0.903 ·
-- fold⊕base 0.724, trained in 2s, no gradient, exact unfold by subtraction.

/-- **One KN lane over an arbitrary floor conserves mass, stays nonneg, and lands
    the evidence.** Floor `p` (a distribution), counts `c`, discount `D` with
    `D 0 = 0` and `0 ≤ D n ≤ n` (the corpus clamps d₁<1, d₂<2, d₃<3 satisfy it);
    `γ = ΣD(c_j)/tot` the discounted mass. The refined
    `q_j = γ·p_j + (c_j − D c_j)/tot`:
    (1) sums to 1 — finetuning transfers mass, never creates it;
    (2) is nonnegative;
    (3) keeps the table's evidence regardless of the floor: `q_j ≥ (c_j − D c_j)/tot`. -/
theorem kn_floor_refinement {V : ℕ} (p : Fin V → ℝ) (c : Fin V → ℕ) (D : ℕ → ℝ)
    (hp1 : ∑ j, p j = 1) (hpnn : ∀ j, 0 ≤ p j)
    (_hD0 : D 0 = 0) (hDnn : ∀ n, 0 ≤ D n) (hDle : ∀ n, D n ≤ n)
    (htot : 0 < ∑ j, (c j : ℝ)) :
    (∑ j, ((∑ i, D (c i)) / (∑ i, (c i : ℝ)) * p j
        + ((c j : ℝ) - D (c j)) / (∑ i, (c i : ℝ))) = 1)
    ∧ (∀ j, 0 ≤ (∑ i, D (c i)) / (∑ i, (c i : ℝ)) * p j
        + ((c j : ℝ) - D (c j)) / (∑ i, (c i : ℝ)))
    ∧ (∀ j, ((c j : ℝ) - D (c j)) / (∑ i, (c i : ℝ))
        ≤ (∑ i, D (c i)) / (∑ i, (c i : ℝ)) * p j
        + ((c j : ℝ) - D (c j)) / (∑ i, (c i : ℝ))) := by
  have hγnn : 0 ≤ (∑ i, D (c i)) / (∑ i, (c i : ℝ)) :=
    div_nonneg (Finset.sum_nonneg fun i _ => hDnn (c i)) (le_of_lt htot)
  refine ⟨?_, fun j => ?_, fun j => ?_⟩
  · have htne : (∑ i, (c i : ℝ)) ≠ 0 := ne_of_gt htot
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, hp1, mul_one, ← Finset.sum_div,
        Finset.sum_sub_distrib, ← add_div]
    have hnum : (∑ i, D (c i)) + ((∑ j, (c j : ℝ)) - ∑ j, D (c j)) = ∑ j, (c j : ℝ) := by
      ring
    rw [hnum, div_self htne]
  · have h1 : 0 ≤ (∑ i, D (c i)) / (∑ i, (c i : ℝ)) * p j := mul_nonneg hγnn (hpnn j)
    have h2 : 0 ≤ ((c j : ℝ) - D (c j)) / (∑ i, (c i : ℝ)) :=
      div_nonneg (by linarith [hDle (c j)]) (le_of_lt htot)
    linarith
  · have : 0 ≤ (∑ i, D (c i)) / (∑ i, (c i : ℝ)) * p j := mul_nonneg hγnn (hpnn j)
    linarith

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE HEADLINE — THE BRIDGE
-- ═══════════════════════════════════════════════════════════════════

/-- **The ML primitives bridge (headline).** The load-bearing primitives of the ML stack are the
    framework's own operations:

    1. **contraction** is the B/U accumulator fold (`contraction_is_fold`);
    2. **softmax** is `x = B/U` at the drag `Σexp`, normalizing to `1`, shift exact
       (`softmax_sum_one`, `softmax_shift`);
    3. **flash-attention tiling** is the accumulator `+` through the `(m, s)` chart, grouping-free
       (`osFuse_is_add`);
    4. **sort** is collapse of the permutation orbit to its canonical floor (`sort_key_classifies`,
       a `Seed` instance);
    5. **the Bellman fixed point** is the monad — TD ≡ 0 iff the return field is conservative, the
       extractable advantage exactly the injected flaw (`bellman_fixed_point_is_monad`).

    With the standing theorems (`ridgeHead_unique`, `gram_stack`, the forward-only orbits,
    `map_lookup`, `dispatchHP_eq_K_state`), the 2,691-op registry — 64% composite, the rest one
    iterator + a lambda — stands on `{0, 1, /}` and the one merge. The lambda was always the
    applicative. -/
theorem ml_primitives_bridge {n m p : ℕ} (hn : 0 < n)
    (A : Matrix (Fin n) (Fin m) ℝ) (B : Matrix (Fin m) (Fin p) ℝ)
    (i : Fin n) (j : Fin p) (x : Fin n → ℝ) (c : ℝ) (a b : ℝ × ℝ)
    {A' : Type} [LinearOrder A'] (l q : List A')
    {V : Type*} (w : V → V → ℝ) :
    ((A * B) i j = ∑ k, A i k * B k j)
      ∧ (∑ i, softmax x i = 1 ∧ softmax (fun i => x i + c) = softmax x)
      ∧ (osVal (osFuse a b) = osVal a + osVal b)
      ∧ (l.mergeSort (· ≤ ·) = q.mergeSort (· ≤ ·) ↔ l.Perm q)
      ∧ ((∃ ψ : V → ℝ, ∀ u v, tdError w ψ u v = 0) ↔ Opportunity.Conservative w) :=
  ⟨contraction_is_fold A B i j,
   ⟨softmax_sum_one hn x, softmax_shift x c⟩,
   osFuse_is_add a b,
   sort_key_classifies l q,
   bellman_fixed_point_is_monad w⟩

end MLPrimitivesBridge
end Sanguine

end
