import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Real.Basic

/-!
# attention-cut T4: keys read by page-bound selection at 32k, the registered predictions against the measured constants

the hattn-page-select note: Qwen2.5-0.5B Q4_K_M, the attention-cut T3 32,768-token dump, 336 heads, the last
8 queries before `t ∈ {2048, 8192, 32768}` (`tools/hattn/page_select.py (unpublished) --full`, output
`~/hattn-long/page_select.out`). Every figure is a `model` input copied from that table; no
theorem here produces a measurement. Reads per key in ten-thousandths of `t`, shares per mille, exponents in
thousandths, page counts in tenths. Adaptive arms stop on the oracle kept mass; rustc, numpy and the instrument
are outside every statement.

`quest_bound_sound` is the selector's soundness (Tang et al., arXiv 2406.10774v2, sec. 3): for a key inside the
page's element-wise box `[m, M]`, `q·k ≤ Σ max(q_i m_i, q_i M_i)`. It is why a page ranked low by its bound
cannot hide a key scoring above that bound; it says nothing about how tight the bound is, which is what PQ3 and
PQ6 measure.
-/

namespace Perf

namespace HattnT4

/-! ## Measured constants at `t = 32768` (medians over layer, head, query) -/

/-- Arm A (Quest, 16-key pages): far tokens selected to reach KL 0.05. -/
def aTok : ℕ := 400

/-- Oracle page order (true page mass), 16-key pages: far tokens selected. -/
def opTok : ℕ := 152

/-- Total key-vector reads per `t`, ten-thousandths: A, A with 64-key pages, oracle pages, best-first tree. -/
def aReads : ℕ := 1444

def a64Reads : ℕ := 582

def opReads : ℕ := 128

def hReads : ℕ := 1458

/-- Pages selected, tenths: A, and B (bound over RoPE pairs `j ≥ 16` only). -/
def aPages : ℕ := 250

def bPages : ℕ := 825

/-- Share of (layer, head, query) at KL ≤ 0.05 with a fixed far budget of 2048 tokens, per mille. -/
def fixed2048 : ℕ := 654

/-- Median growth exponent of A's selected far tokens, `t = 2048..32768`, thousandths. -/
def aAlpha : ℕ := 750

/-! ## Verdicts -/

/-- **PQ1 held.** A selects at most 1024 far tokens at the median. -/
theorem pq1_held : aTok ≤ 1024 := by decide

/-- **PQ2 held.** A reads at most `0.20 t` at the median. -/
theorem pq2_held : aReads ≤ 2000 := by decide

/-- **PQ3 refuted.** The slow-pair bound needs 3.3× the pages of the full bound, not at most as many. -/
theorem pq3_refuted : ¬ bPages ≤ aPages := by decide

theorem pq3_slow_bound_worse : 3 * aPages < bPages := by decide

/-- **PQ4 refuted.** A fixed 2048-token budget meets KL 0.05 for 65.4 % of cases, not 90 %. -/
theorem pq4_refuted : ¬ 900 ≤ fixed2048 := by decide

/-- **PQ5 refuted.** A's selected tokens grow as `t^0.75`, not at most `t^0.5`. -/
theorem pq5_refuted : ¬ aAlpha ≤ 500 := by decide

theorem pq5_sublinear : aAlpha < 1000 := by decide

/-- **PQ6 refuted.** The best-first tree reads `0.146 t`, not at most `0.10 t`. -/
theorem pq6_refuted : ¬ hReads ≤ 1000 := by decide

/-! ## Where the reads go -/

/-- **The bound costs 2.6× the clumping floor.** A selects more than twice the far tokens the true page order needs. -/
theorem bound_looseness : 2 * opTok < aTok := by decide

/-- **The metadata scan dominates A.** Two bound vectors per 16-key page is `0.125 t` of A's `0.144 t`. -/
theorem metadata_dominates : 1250 * 10 > aReads * 8 := by decide

/-- **64-key pages read 2.5× less than 16-key pages**, and still 4.5× above the oracle-page floor. -/
theorem pages64_best_real : 2 * a64Reads < aReads ∧ 4 * opReads < a64Reads := by decide

/-! ## Soundness of the page bound -/

/-- **Quest's bound is sound.** For every key in the element-wise box `[m, M]`,
`Σ q_i k_i ≤ Σ max(q_i m_i, q_i M_i)`. -/
theorem quest_bound_sound {n : ℕ} (q m M k : Fin n → ℝ) (hm : ∀ i, m i ≤ k i) (hM : ∀ i, k i ≤ M i) :
    ∑ i, q i * k i ≤ ∑ i, max (q i * m i) (q i * M i) := by
  refine Finset.sum_le_sum fun i _ => ?_
  rcases le_total 0 (q i) with h | h
  · exact (mul_le_mul_of_nonneg_left (hM i) h).trans (le_max_right _ _)
  · exact (mul_le_mul_of_nonpos_left (hm i) h).trans (le_max_left _ _)

end HattnT4

end Perf
