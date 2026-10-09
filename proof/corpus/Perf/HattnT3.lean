import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen

/-!
# HATTN-T3: attention as a screened N-body sum at 32k, the registered predictions against the measured constants

`docs/evidence/hattn-far-field.md`: Qwen2.5-0.5B Q4_K_M, one 32,768-token prompt, 24 layers × 14 heads,
queries the last 8 positions before `t ∈ {512, 2048, 8192, 32768}` (`tools/hattn/far_field.py --full`,
output `/home/socol/oc-tmp/hattn-long/far_field.out`). Every figure below is a `model` input copied from that
table; no theorem here produces a measurement. Exponents and coherences in thousandths, errors and shares in
ten-thousandths, key counts in tenths. rustc, numpy and the instrument's own correctness are outside every
statement here.

The two analytic theorems state why the moment far field behaves as measured:
`first_cumulant_le` (Jensen) — the first-cumulant field `log N + mean score` never exceeds the far set's
log partition, so it can only under-weight the far set; `second_cumulant_overshoots` — the second-cumulant
field `mean + var/2` is not a bound: on a two-point cloud it exceeds the true log partition.
-/

namespace Perf

namespace HattnT3

/-! ## P-PH1: concentration -/

/-- Median over the 336 heads of the growth exponent of `k_ε(t)`, `t = 2048..32768`, thousandths. -/
def alphaMedian : ℕ := 394

/-- Share of heads with exponent above `0.8`, per mille. -/
def shareDiffuse : ℕ := 196

/-- **P-PH1 refuted.** It needed the median at most `0.5` and at least 20 % diffuse heads; the median
holds, the diffuse share is 19.6 %. -/
theorem ph1_refuted : ¬ (alphaMedian ≤ 500 ∧ 200 ≤ shareDiffuse) := by decide

/-- **P-PH1, the half that holds.** The median head's key count grows sublinearly. -/
theorem ph1_median_sublinear : alphaMedian < 500 := by decide

/-- Median `k_ε` at `t = 32768`, tenths of a key (161.2). -/
def kEps32k : ℕ := 1612

/-- **Concentration at 32k.** The median head holds `1 - ε` of its mass (`KL = 0.05`) in under 0.5 % of
the 32,768 keys. -/
theorem keps_32k_lt_half_percent : kEps32k * 200 < 32768 * 10 := by decide

/-! ## P-PH2: Gaussian far field -/

/-- Median attention-output rel-L2 at `t = 32768`, ten-thousandths: drop the far set, first-cumulant
field, second-cumulant field. -/
def errTrunc : ℕ := 1056

def errG1 : ℕ := 1039

def errG2 : ℕ := 3446

/-- **P-PH2 refuted.** It needed the second-cumulant field at most `0.10` and at least 2× below the
first-cumulant field; it is `0.34` and 3.3× above. -/
theorem ph2_refuted : ¬ (errG2 ≤ 1000 ∧ 2 * errG2 ≤ errG1) := by decide

/-- **The second cumulant hurts.** -/
theorem ph2_second_cumulant_worse : errG1 < errG2 ∧ 3 * errG1 < errG2 := by decide

/-- **The first-cumulant field barely moves off truncation** (consistent with `first_cumulant_le`:
it can only under-weight the far set). -/
theorem ph2_g1_near_trunc : errG1 ≤ errTrunc ∧ errTrunc - errG1 < 20 := by decide

/-! ## P-PH3: mass against screening -/

/-- Median share of score variance from the mass term `s|k|²/2`, ten-thousandths. -/
def massShare : ℕ := 3377

/-- **P-PH3 refuted.** It needed at most `0.30`; the mass term carries `0.34`. -/
theorem ph3_refuted : ¬ massShare ≤ 3000 := by decide

/-- **Screening still dominates the mass term.** -/
theorem ph3_screening_dominates : 2 * massShare < 10000 := by decide

/-! ## P-PH4: RoPE phase coherence of the far keys -/

/-- Median coherence `|E z_j| / √(E|z_j|²)` of far keys per pair, thousandths, `j = 0..31`. -/
def coh : List ℕ :=
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 20, 10, 30, 10, 60, 40, 170, 80, 400, 520, 450,
   370, 480, 400, 420, 370, 390, 510, 740]

/-- **P-PH4 refuted.** It needed every high-frequency pair at most `0.1` and every low-frequency pair
`j ≥ 24` at least `0.5`. -/
theorem ph4_refuted : ¬ ((∀ c ∈ coh.take 16, c ≤ 100) ∧ (∀ c ∈ coh.drop 24, 500 ≤ c)) := by decide

/-- **P-PH4, the half that holds.** Every high-frequency pair is phase-random to within `0.03`. -/
theorem ph4_high_freq_random : ∀ c ∈ coh.take 16, c ≤ 30 := by decide

/-! ## Why the moment far field fails -/

/-- **Jensen.** The first-cumulant field never exceeds the far set's partition function:
`exp (mean x) ≤ mean (exp x)`. -/
theorem first_cumulant_le (n : ℕ) (hn : 0 < n) (x : Fin n → ℝ) :
    Real.exp ((∑ i, x i) / n) ≤ (∑ i, Real.exp (x i)) / n := by
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  have h := ConvexOn.map_sum_le convexOn_exp (t := Finset.univ) (w := fun _ => (1 : ℝ) / n) (p := x)
    (fun _ _ => by positivity) (by simp [Finset.card_univ]; field_simp) (fun _ _ => Set.mem_univ _)
  simp only [smul_eq_mul] at h
  rw [← Finset.mul_sum, ← Finset.mul_sum] at h
  rwa [one_div, inv_mul_eq_div, inv_mul_eq_div] at h

/-- **The second cumulant is not a bound.** Two keys scoring `0` and `10`: mean `5`, variance `25`, the
second-cumulant field `exp (5 + 25/2)` exceeds the true mean `(1 + e¹⁰)/2`. -/
theorem second_cumulant_overshoots :
    (Real.exp 0 + Real.exp 10) / 2 < Real.exp (5 + 25 / 2) := by
  have h1 : Real.exp 0 ≤ Real.exp 10 := Real.exp_le_exp.mpr (by norm_num)
  have h2 : Real.exp 10 < Real.exp (5 + 25 / 2) := Real.exp_lt_exp.mpr (by norm_num)
  linarith

end HattnT3

end Perf
