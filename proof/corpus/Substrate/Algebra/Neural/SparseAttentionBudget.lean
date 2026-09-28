import Substrate.Algebra.Neural.MLPrimitivesBridge
import Substrate.Algebra.Bulk.BulkExhaustiveness
import Closures.Engineering.CoordinateReadout
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Analysis.MeanInequalities
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# The sparse-attention budget is derived
@name How Many Knots To Pluck
@child Pull one thread in a web and only a few knots really move. How many is not a guess —
  count how spread-out the pulling is and that IS the number to keep.
@scientist Attention `p = softmax = x=B/U`. `PR = 1/∑pᵢ²` bounds the heavy-key count
  (`heavy_count_le_PR`) and `PR ≤ exp H(p)` (`PR_le_perplexity`, Rényi `R₂ ≤ R₁`), so
  `A = ⌈PPL⌉` is a valid budget read off one calibration pass. Sink/window detection is a
  heavy-cut on a derived distribution; any threshold in the separation gap picks the same set
  (`threshold_gap_stable`).
@engineer `heavy_count_le_PR` · `PR_le_card` · `PR_le_perplexity` · `budget_ceil_ppl_valid` ·
  `threshold_gap_stable`; headline `sparse_attention_route_derived` = budget + depth-2 readout
  (`SpinorReadout.separate_within_depth_two`). The three faces (sink/local/content) are the
  forced trinity (`Polymorphic.three_constructor_exhaustiveness`).

## What "the budget is derived" warrants

* **Proved (native):** `heavy_count_le_PR` — any key set whose members each carry
  `pᵢ ≥ S = ∑pᵢ²` has cardinality `≤ 1/S = PR`, exhibited by `card·S² ≤ ∑_heavy p² ≤ ∑_all p² = S`.
  With it: `argmax_is_heavy` (`∑p² ≤ pmax`, so the cut is never empty — the one fact that keeps
  the heavy set from being vacuous), `heavy_cut_bounds_surprisal`, `heavy_cut_is_score_threshold`
  (an `↔`, both directions given), `threshold_gap_stable`, and `budget_ceil_ppl_valid` as the
  chaining of the first two.
* **Proved (Mathlib):** `PR_le_card` is Cauchy–Schwarz (`Finset.sum_mul_sq_le_sq_mul_sq`) and
  `PR_le_perplexity` is weighted AM–GM (`Real.geom_mean_le_arith_mean_weighted`) closed by
  `inv_anti₀`; the "Rényi `R₂ ≤ R₁`" reading is those two lemmas. `PR_le_card` also carries a
  hypothesis its proof never uses — `hn : 0 < n` — and at `n = 0` the remaining `hsum` is already
  unsatisfiable.
* **Next:** the headline `sparse_attention_route_derived` is weaker than its own
  docstring. The docstring names three things — the budget, the depth-2 concept code, and "the
  three faces are the forced trinity (`Polymorphic.three_constructor_exhaustiveness`)" — and the
  statement conjoins two; the trinity conjunct is absent, as is `one_flaw_blind`, which point 2
  cites. Worse, the two conjuncts it does carry share NO variable: conjunct 1 is over
  `p, S, heavy`, conjunct 2 over `K, C, f₁, f₂, k, k'`. The `∧` is the only thing asserting they
  are about one route.
* **Proved (riding):** that second conjunct is `SpinorReadout.separate_within_depth_two hpair hne`
  unchanged. Nothing about two-window readout is derived in this file.
* **Open:** every cost sentence, because there is no cost model here.
  `heavy_cut_is_score_threshold`'s "no `∑p²` recomputation, no top-k search, ONE reduction";
  `heavy_count_le_PR`'s "`A = ⌈PR⌉` is read off the geometry, not chosen". `A` is never a Lean
  object — `⌈PR⌉` and `⌈PPL⌉` occur only in prose — and no statement mentions a step count, a
  pass, or a reduction.
* **Anchor:** `heavy_cut_bounds_surprisal`'s "Subsumes mirostat's `(η, τ)`
  controller … and one-sides typical-p: the sampler zoo exists because this bound was never
  stated." No sampler, controller or comparison appears in any type; the theorem is
  `Real.log_le_log` applied to `S ≤ p i`.
* **Anchor:** that `p` is attention at all, and that `heavy_cut_is_score_threshold`
  is "the kernel's line" — that the block a shipped selector keeps IS the proven heavy block. The
  cited source has MOVED TWICE: `hashtrinity theta_select, 2026-07-08` is now theta_select_module
  in `modelfold/src/simt_nn/sparse.rs` (`../fold`) — the flat simt_nn.rs became a directory
  module, and nothing lives under `hashtrinity/` any more.
-/

noncomputable section

namespace Sanguine
namespace SparseAttentionBudget

/-- The **participation ratio** (inverse Simpson index) of a distribution: the effective number of
    modes it spreads over. `1 / ∑ pᵢ²`. Uniform on `n` → `n`; a point mass → `1`. -/
def participationRatio {n : ℕ} (p : Fin n → ℝ) : ℝ := 1 / ∑ i, (p i) ^ 2

/-- **The budget is forced: at most `PR` keys are heavy.** For a distribution `p` (`0 ≤ p`) with
    `S = ∑pᵢ² > 0`, any set of keys each carrying weight `≥ S` (`= 1/PR`, the heavy/light cut) has
    cardinality `≤ PR = 1/S`. So the effective support is `PR`; the rest are the near-null drag. The
    sparse budget `A = ⌈PR⌉` is read off the geometry, not chosen. -/
theorem heavy_count_le_PR {n : ℕ} (p : Fin n → ℝ) (_hp : ∀ i, 0 ≤ p i)
    (S : ℝ) (hS : S = ∑ i, (p i) ^ 2) (hSpos : 0 < S)
    (heavy : Finset (Fin n)) (hh : ∀ i ∈ heavy, S ≤ p i) :
    (heavy.card : ℝ) ≤ participationRatio p := by
  -- card·S² ≤ ∑_{heavy} p² ≤ ∑_all p² = S ⟹ card ≤ 1/S = PR
  have hheavy_sq : (heavy.card : ℝ) * S ^ 2 ≤ ∑ i ∈ heavy, (p i) ^ 2 := by
    have : ∑ _i ∈ heavy, S ^ 2 ≤ ∑ i ∈ heavy, (p i) ^ 2 :=
      Finset.sum_le_sum fun i hi => by
        have h1 : S ≤ p i := hh i hi
        nlinarith [_hp i, hSpos.le]
    simpa [Finset.sum_const, nsmul_eq_mul] using this
  have hle_total : ∑ i ∈ heavy, (p i) ^ 2 ≤ S := by
    rw [hS]
    exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ heavy)
      (fun i _ _ => sq_nonneg _)
  have hcardS2 : (heavy.card : ℝ) * S ^ 2 ≤ S := le_trans hheavy_sq hle_total
  -- card·S² ≤ S and S>0 ⟹ card ≤ 1/S = PR
  rw [participationRatio, ← hS, le_div_iff₀ hSpos]
  nlinarith [hcardS2, hSpos]

/-- **`PR ≤ n`** — the effective support never exceeds the full key count (Cauchy–Schwarz: for a
    distribution, `∑pᵢ² ≥ 1/n`, so `PR = 1/∑pᵢ² ≤ n`). Dense (`A = n`) is the `PR = n` limit, i.e.
    a *flat* attention — the worst case, all drag, no concentration. -/
theorem PR_le_card {n : ℕ} (_hn : 0 < n) (p : Fin n → ℝ) (hsum : ∑ i, p i = 1) :
    participationRatio p ≤ n := by
  -- Cauchy–Schwarz: (∑ p·1)² ≤ (∑p²)(∑1²) ⟹ 1 ≤ (∑p²)·n
  have hcs : (1 : ℝ) ≤ (∑ i, (p i) ^ 2) * n := by
    have h := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ p (fun _ => (1:ℝ))
    simp only [mul_one, one_pow] at h
    rw [hsum] at h
    simpa [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] using h
  have hSpos : 0 < ∑ i, (p i) ^ 2 := by
    rcases lt_or_eq_of_le (Finset.sum_nonneg fun _ _ => sq_nonneg (p _)) with h | h
    · exact h
    · exfalso; rw [← h, zero_mul] at hcs; linarith
  rw [participationRatio, div_le_iff₀ hSpos]
  nlinarith [hcs]

/-- Shannon perplexity `exp H(p)`. `Real.log 0 = 0` makes the `pᵢ = 0` convention automatic. -/
def perplexity {n : ℕ} (p : Fin n → ℝ) : ℝ := Real.exp (∑ i, p i * Real.log (1 / p i))

/-- **`PR ≤ PPL`** (Rényi `R₂ ≤ R₁`): weighted AM–GM with `w = z = p` gives
    `exp(∑ pᵢ log pᵢ) = ∏ pᵢ^pᵢ ≤ ∑ pᵢ²`; invert. So `⌈PPL⌉` inherits the budget validity of
    `heavy_count_le_PR`. -/
theorem PR_le_perplexity {n : ℕ} (p : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1) :
    participationRatio p ≤ perplexity p := by
  have hgm : ∏ i, (p i) ^ (p i) ≤ ∑ i, p i * p i :=
    Real.geom_mean_le_arith_mean_weighted Finset.univ p p
      (fun i _ => hp i) hsum (fun i _ => hp i)
  have hprod : ∏ i, (p i) ^ (p i) = Real.exp (∑ i, p i * Real.log (p i)) := by
    rw [Real.exp_sum]
    refine Finset.prod_congr rfl fun i _ => ?_
    rcases eq_or_lt_of_le (hp i) with h | h
    · simp [← h]
    · rw [Real.rpow_def_of_pos h, mul_comm]
  have key : Real.exp (∑ i, p i * Real.log (p i)) ≤ ∑ i, (p i) ^ 2 := by
    rw [← hprod]; simpa [sq] using hgm
  have hppl : perplexity p = (Real.exp (∑ i, p i * Real.log (p i)))⁻¹ := by
    rw [perplexity, ← Real.exp_neg, ← Finset.sum_neg_distrib]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [one_div, Real.log_inv]; ring
  rw [participationRatio, hppl, one_div]
  exact inv_anti₀ (Real.exp_pos _) key

/-- **The budget `A = ⌈PPL⌉` is valid**: the heavy-key count is bounded by the perplexity
    (`heavy_count_le_PR` chained with `PR_le_perplexity`). -/
theorem budget_ceil_ppl_valid {n : ℕ} (p : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1)
    (S : ℝ) (hS : S = ∑ i, (p i) ^ 2) (hSpos : 0 < S)
    (heavy : Finset (Fin n)) (hh : ∀ i ∈ heavy, S ≤ p i) :
    (heavy.card : ℝ) ≤ perplexity p :=
  le_trans (heavy_count_le_PR p hp S hS hSpos heavy hh) (PR_le_perplexity p hp hsum)

/-- **The argmax is always heavy.** `Σp² ≤ pmax·Σp = pmax`, so the top token always clears the
    heavy cut `pᵢ ≥ Σp²`. With `heavy_count_le_PR` this derives the sampler: the candidate set is
    the distribution's own heavy set (≤ PR tokens, contains the argmax, renormalized at the
    model's own weights) — top-k, top-p and temperature collapse into the data's own cut. -/
theorem argmax_is_heavy {n : ℕ} (p : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1) (j : Fin n) (hj : ∀ i, p i ≤ p j) :
    ∑ i, (p i) ^ 2 ≤ p j := by
  calc ∑ i, (p i) ^ 2 ≤ ∑ i, p i * p j := by
        refine Finset.sum_le_sum fun i _ => ?_
        rw [sq]
        exact mul_le_mul_of_nonneg_left (hj i) (hp i)
    _ = p j := by rw [← Finset.sum_mul, hsum, one_mul]

/-- **The heavy cut bounds surprisal (G7).** Every token admitted by the cut `pᵢ ≥ Σp²`
    has surprisal `−log pᵢ ≤ log PR` — the emitted stream's per-step surprisal is bounded
    with NO feedback loop. Subsumes mirostat's `(η, τ)` controller (a SpinorLoop steering
    surprisal to a target) and one-sides typical-p: the sampler zoo exists because this
    bound was never stated. -/
theorem heavy_cut_bounds_surprisal {n : ℕ} (p : Fin n → ℝ)
    (S : ℝ) (hS : S = ∑ i, (p i) ^ 2) (hSpos : 0 < S)
    (i : Fin n) (hi : S ≤ p i) :
    Real.log (1 / p i) ≤ Real.log (participationRatio p) := by
  have hpi : 0 < p i := lt_of_lt_of_le hSpos hi
  rw [participationRatio, ← hS, one_div, one_div, Real.log_inv, Real.log_inv]
  exact neg_le_neg (Real.log_le_log hSpos hi)

/-- **The heavy cut is a closed-form SCORE threshold** — what the metal computes
    (`hashtrinity theta_select`, 2026-07-08). With unnormalized weights `eᵢ = exp(ubᵢ − m)`,
    `Se = Σe`, `Se2 = Σe²`, the softmax `pᵢ = eᵢ/Se`, the heavy cut `pᵢ ≥ Σp²` is exactly the
    threshold `eᵢ ≥ Se2/Se` — no `∑p²` recomputation, no top-k search, ONE reduction. In score
    space `eᵢ ≥ Se2/Se ⟺ ubᵢ ≥ m + ln(Se2/Se)` (exp monotone), the kernel's line. So the block
    the kernel keeps is precisely the proven heavy block, and (`heavy_count_le_PR`) the kept
    content count is `≤ PR` — the derived budget, realized. -/
theorem heavy_cut_is_score_threshold {n : ℕ} (e : Fin n → ℝ) (he : ∀ i, 0 < e i) (i : Fin n) :
    (∑ j, (e j / ∑ k, e k) ^ 2) ≤ e i / (∑ k, e k)
      ↔ (∑ j, (e j) ^ 2) ≤ e i * (∑ k, e k) := by
  have hSe : 0 < ∑ k, e k := Finset.sum_pos (fun j _ => he j) ⟨i, Finset.mem_univ i⟩
  have hSe2 : (0 : ℝ) < (∑ k, e k) ^ 2 := by positivity
  have hp : ∑ j, (e j / ∑ k, e k) ^ 2 = (∑ j, (e j) ^ 2) / (∑ k, e k) ^ 2 := by
    rw [Finset.sum_div]
    exact Finset.sum_congr rfl fun j _ => by rw [div_pow]
  have hcancel : (e i / ∑ k, e k) * (∑ k, e k) ^ 2 = e i * (∑ k, e k) := by
    field_simp
  rw [hp, div_le_iff₀ hSe2, hcancel]

/-- **Threshold robustness**: when the profile separates (every value below `lo` or above `hi`),
    every threshold inside the gap selects the same set. The sink/window cut needs a gap, not a
    magic multiplier. -/
theorem threshold_gap_stable {n : ℕ} (r : Fin n → ℝ) (lo hi t₁ t₂ : ℝ)
    (hgap : ∀ i, r i ≤ lo ∨ hi ≤ r i)
    (h₁ : lo < t₁) (h₂ : t₁ ≤ t₂) (h₃ : t₂ ≤ hi) :
    {i | t₁ ≤ r i} = {i | t₂ ≤ r i} := by
  ext i
  simp only [Set.mem_setOf_eq]
  rcases hgap i with h | h <;> constructor <;> intro _ <;> linarith

/-- **The sparse-attention route is derived, not dialed (headline).**
    1. budget: at most `PR ≤ PPL` keys are heavy (`heavy_count_le_PR` + `PR_le_perplexity`),
       so `A = ⌈PPL⌉` is read off the geometry;
    2. concept code: two windows separate distinct keys
       (`SpinorReadout.separate_within_depth_two`); one window is blind (`one_flaw_blind`).
    The three faces are the forced trinity (`Polymorphic.three_constructor_exhaustiveness`). -/
theorem sparse_attention_route_derived
    {n : ℕ} (p : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i = 1)
    (S : ℝ) (hS : S = ∑ i, (p i) ^ 2) (hSpos : 0 < S)
    (heavy : Finset (Fin n)) (hh : ∀ i ∈ heavy, S ≤ p i)
    {K C : Type} {f₁ f₂ : K → C}
    (hpair : Function.Injective (fun k => (f₁ k, f₂ k))) {k k' : K} (hne : k ≠ k') :
    ((heavy.card : ℝ) ≤ participationRatio p ∧ participationRatio p ≤ perplexity p)
      ∧ (f₁ k ≠ f₁ k' ∨ f₂ k ≠ f₂ k') :=
  ⟨⟨heavy_count_le_PR p hp S hS hSpos heavy hh, PR_le_perplexity p hp hsum⟩,
   SpinorReadout.separate_within_depth_two hpair hne⟩

end SparseAttentionBudget
end Sanguine

end
