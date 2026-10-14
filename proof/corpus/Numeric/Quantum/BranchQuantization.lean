/-
Branch quantization condition read from `K` classical branches (`QREG-QUANTIZATION-SUM`, epic
`docs/epics/quantum-register.md`).

Lohmiller and Slotine (arXiv 2405.06328, Lemma 3) read the quantization condition from the phases
of `K` classical branches. For a phase `e^{iθ}` the branch sum is `Σ_{κ<K} e^{iκθ}`. This file
proves the two limits and the finite-`K` zero set of that sum, all over `ℂ` with `e^{iθ}` on the
unit circle.

## Landed here

* `branchSum_eq_of_exp_eq_one`: `e^{iθ} = 1` gives branch sum `K`, so the average is `1`.
* `branchSum_geom`: `e^{iθ} ≠ 1` gives the geometric closed form.
* `norm_branchSum_le`: `‖Σ‖ ≤ 2 / ‖e^{iθ} − 1‖` for every `K`.
* `branchAvg_tendsto_zero`, `branchAvg_tendsto_one`: the average tends to `0` off the trivial
  phase and to `1` on it, as `K → ∞`.
* `branchSum_eq_zero_iff`: for `K > 0`, the branch sum vanishes exactly at the nontrivial `K`-th
  roots of unity.
* `branchSum_four_quarter`: the four rotor branches at `θ = π/2` sum to `0`.
* `exp_eq_one_iff_mem`: `e^{iθ} = 1` iff `θ ∈ 2πℤ`.

## Not claimed

* No rate for the `1/K` convergence beyond the bound in `norm_branchSum_le`.
* No statement about a quantum register or amplitude; the file is the branch sum only.
* No floating-point claim.

## Complexity

`branchSum θ K` is a sum of `K` exponentials: O(K) in time, O(1) in space, counted in
exponential evaluations. The closed form `branchSum_geom` is O(1) in exponentials.

The axiom cone is checked by the build's `AxiomAudit` route, not by this file.
-/
import Mathlib.Algebra.Field.GeomSum
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Complex.Log
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Tactic

namespace Numeric.BranchQuantization

open Complex Filter Topology

/-- The branch sum `Σ_{κ<K} e^{iκθ}`. Complexity: `K` exponentials, O(K) time, O(1) space. -/
noncomputable def branchSum (θ : ℝ) (K : ℕ) : ℂ :=
  ∑ κ ∈ Finset.range K, exp ((κ : ℂ) * (θ : ℂ) * I)

/-- Each branch term is the `κ`-th power of the unit phase. -/
theorem term_eq_pow (θ : ℝ) (κ : ℕ) :
    exp ((κ : ℂ) * (θ : ℂ) * I) = exp ((θ : ℂ) * I) ^ κ := by
  rw [← Complex.exp_nat_mul]
  congr 1
  ring

/-- If `e^{iθ} = 1`, every branch equals `1`, so the branch sum is `K`. -/
theorem branchSum_eq_of_exp_eq_one (θ : ℝ) (h : exp ((θ : ℂ) * I) = 1) (K : ℕ) :
    branchSum θ K = K := by
  simp [branchSum, term_eq_pow, h]

/-- If `e^{iθ} ≠ 1`, the branch sum is the geometric closed form. -/
theorem branchSum_geom (θ : ℝ) (h : exp ((θ : ℂ) * I) ≠ 1) (K : ℕ) :
    branchSum θ K = (exp ((θ : ℂ) * I) ^ K - 1) / (exp ((θ : ℂ) * I) - 1) := by
  unfold branchSum
  simp only [term_eq_pow]
  exact geom_sum_eq h K

/-- Off the trivial phase the branch sum is bounded independently of `K`. -/
theorem norm_branchSum_le (θ : ℝ) (h : exp ((θ : ℂ) * I) ≠ 1) (K : ℕ) :
    ‖branchSum θ K‖ ≤ 2 / ‖exp ((θ : ℂ) * I) - 1‖ := by
  have hn : ‖exp ((θ : ℂ) * I)‖ = 1 := Complex.norm_exp_ofReal_mul_I θ
  have hpow : ‖exp ((θ : ℂ) * I) ^ K‖ = 1 := by
    rw [norm_pow, hn, one_pow]
  have hnum : ‖exp ((θ : ℂ) * I) ^ K - 1‖ ≤ 2 := by
    calc ‖exp ((θ : ℂ) * I) ^ K - 1‖ ≤ ‖exp ((θ : ℂ) * I) ^ K‖ + ‖(1 : ℂ)‖ := norm_sub_le _ _
      _ = 2 := by rw [hpow, norm_one]; norm_num
  rw [branchSum_geom θ h K, norm_div]
  gcongr

/-- Off the trivial phase the branch average tends to `0` as `K → ∞`. -/
theorem branchAvg_tendsto_zero (θ : ℝ) (h : exp ((θ : ℂ) * I) ≠ 1) :
    Tendsto (fun K : ℕ => branchSum θ K / K) atTop (𝓝 0) := by
  have hC : Tendsto (fun K : ℕ => (2 / ‖exp ((θ : ℂ) * I) - 1‖) / (K : ℝ)) atTop (𝓝 0) :=
    tendsto_const_div_atTop_nhds_zero_nat _
  refine squeeze_zero_norm' ?_ hC
  filter_upwards [eventually_ge_atTop 1] with K hK
  have hK' : ‖(K : ℂ)‖ = (K : ℝ) := by simp
  rw [norm_div, hK']
  have hb := norm_branchSum_le θ h K
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  gcongr

/-- On the trivial phase the branch average is `1` for every `K ≥ 1`, so it tends to `1`. -/
theorem branchAvg_tendsto_one (θ : ℝ) (h : exp ((θ : ℂ) * I) = 1) :
    Tendsto (fun K : ℕ => branchSum θ K / K) atTop (𝓝 1) := by
  refine tendsto_const_nhds.congr' ?_
  filter_upwards [eventually_ge_atTop 1] with K hK
  have hK0 : (K : ℂ) ≠ 0 := by exact_mod_cast (show K ≠ 0 by omega)
  rw [branchSum_eq_of_exp_eq_one θ h K, div_self hK0]

/-- For `K > 0`, the branch sum vanishes exactly at the nontrivial `K`-th roots of unity. -/
theorem branchSum_eq_zero_iff (θ : ℝ) (K : ℕ) (hK : 0 < K) :
    branchSum θ K = 0 ↔ exp ((θ : ℂ) * I) ^ K = 1 ∧ exp ((θ : ℂ) * I) ≠ 1 := by
  constructor
  · intro hz
    have hne : exp ((θ : ℂ) * I) ≠ 1 := by
      intro h1
      rw [branchSum_eq_of_exp_eq_one θ h1 K] at hz
      exact absurd hz (Nat.cast_ne_zero.mpr hK.ne')
    refine ⟨?_, hne⟩
    rw [branchSum_geom θ hne K] at hz
    rcases div_eq_zero_iff.mp hz with h2 | h2
    · exact sub_eq_zero.mp h2
    · exact absurd h2 (sub_ne_zero.mpr hne)
  · rintro ⟨hK1, hne⟩
    rw [branchSum_geom θ hne K, hK1, sub_self, zero_div]

/-- `e^{i Real.pi/2} = i`. -/
theorem exp_pi_div_two_mul_I : exp (((Real.pi / 2 : ℝ) : ℂ) * I) = I := by
  rw [Complex.exp_mul_I, ← Complex.ofReal_cos, ← Complex.ofReal_sin, Real.cos_pi_div_two,
    Real.sin_pi_div_two]
  simp

/-- The four rotor branches at `θ = Real.pi/2` sum to `0`: `i⁴ = 1` and `i ≠ 1`. -/
theorem branchSum_four_quarter : branchSum (Real.pi / 2) 4 = 0 := by
  refine (branchSum_eq_zero_iff (Real.pi / 2) 4 (by norm_num)).mpr ⟨?_, ?_⟩
  · rw [exp_pi_div_two_mul_I, Complex.I_pow_four]
  · rw [exp_pi_div_two_mul_I]
    intro h
    have := congrArg Complex.re h
    simp at this

/-- `e^{iθ} = 1` iff `θ` is an integer multiple of `2Real.pi`. -/
theorem exp_eq_one_iff_mem (θ : ℝ) :
    exp ((θ : ℂ) * I) = 1 ↔ ∃ n : ℤ, θ = n * (2 * Real.pi) := by
  rw [Complex.exp_eq_one_iff]
  constructor
  · rintro ⟨n, hn⟩
    refine ⟨n, ?_⟩
    have h2 : ((θ : ℝ) : ℂ) = ((n * (2 * Real.pi) : ℝ) : ℂ) := by
      apply mul_right_cancel₀ Complex.I_ne_zero
      rw [hn]
      push_cast
      ring
    exact_mod_cast h2
  · rintro ⟨n, hn⟩
    exact ⟨n, by rw [hn]; push_cast; ring⟩

end Numeric.BranchQuantization
