import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.MeasureTheory.Measure.Count
import Mathlib.MeasureTheory.Group.Action
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Matrix.Order
import Mathlib.Algebra.Order.Chebyshev
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Perf.Roofline
import Perf.Chain
import Perf.Overlap
import Apeiron.Trichotomy.Basic

set_option linter.apeiron.notation false

/-!
# The correspondence between the performance cost model and the quantisation cost model

The performance model (`Perf.Roofline`, `Perf.Chain`, `Perf.Overlap`) predicts because it has a floor on the
binding resource. This module proves that the quantisation side has the same floor, from the same lemma, and states
where the two models differ.

| role | performance | quantisation | theorem |
|---|---|---|---|
| floor | bytes ≤ steps · capacity (counting measure) | `r^d ≤ |C| · ε^d` (Haar measure) | `cover_bound`, `bytes_floor`, `covering_floor` |
| what is counted | the binding field | the stiff directions of the loss | `stiff_floor` |
| free rewrites | fusion preserves denotation | a gauge preserves the function | `gauge_lift_bound`, `perm_gauge_le`, `gauge_null` |
| composition, upper | overlap ≤ sequence (ℓ∞ triangle) | `√K` subadditive (ℓ² triangle, Minkowski) | `roofline_subadd`, `kl_sqrt_subadd`, `kl_joint_le` |
| composition, lower | overlap ≥ max (no cancellation) | none: scores cancel | `roofline_nocancel`, `kl_cancels` |
| serial composition | chain cost adds (ℓ¹) | | `chain_append` |
| pair form | | discriminant `⟪v,w⟫² − ‖v‖²‖w‖² ≤ 0`: elliptic or parabolic, never hyperbolic | `score_form_disc_nonpos` |
| partition | rounds `⌈k/p⌉`; greedy ≤ `W/S + w/s` | fluid floor `W/S` ↔ reverse water-filling level `θ` (`fluid_floor`, `waterfill_floor`) | `greedy_makespan`, `refine_gain_le` |
| carrier ↔ coder | bytes per token = `N · R / 8` | rate `R` | `tg_floor_ratio` |

The asymmetry in composition is the sign of the summands, not the norm: performance demands are non-negative
counts, so a sum never falls below its largest part; layer scores are signed vectors, so their sum can.

## What is proved here and what is not

* **Proved** under the closed axiom cone: every theorem below.
* **`model` readings.** `covering_floor` is the worst-case covering converse for a ball. The quantisation error is
  an average over a source, not a worst case, and the loss is a quadratic form only in the small-perturbation
  regime; the bits-per-dof figure in `q4_bits_per_stiff_dof` is the reading of this bound at the Q4_K-type layer
  error `ρ = 0.0045`, not a rate–distortion theorem for the weight source. The partition verdicts read measured
  GMAC/s (`0bf8b4d`) against the homogeneous rounds model; P/E-core speeds are not measured here.
* **Not proved.** No byte, cycle or nat follows from a theorem here; the numbers in the verdicts are measured
  inputs.
-/

namespace Perf

namespace CostCorrespondence

open MeasureTheory Metric Matrix Pointwise

/-! ## 1 · One floor -/

/-- **The shared floor.** If `A` is covered by the cells `s i`, `i ∈ I`, and each cell has measure at most `c`, then
`μ A ≤ |I| · c`. With the counting measure this is the roofline (bytes ≤ steps · capacity); with Haar measure it is
the covering converse. -/
theorem cover_bound {α ι : Type*} [MeasurableSpace α] (μ : Measure α) (A : Set α) (I : Finset ι)
    (s : ι → Set α) (c : ENNReal) (hA : A ⊆ ⋃ i ∈ I, s i) (hc : ∀ i ∈ I, μ (s i) ≤ c) :
    μ A ≤ I.card * c := by
  calc μ A ≤ μ (⋃ i ∈ I, s i) := measure_mono hA
    _ ≤ ∑ i ∈ I, μ (s i) := measure_biUnion_finset_le I s
    _ ≤ ∑ _i ∈ I, c := Finset.sum_le_sum hc
    _ = I.card * c := by simp [Finset.sum_const, nsmul_eq_mul]

/-- **Performance instance.** Every byte of `B` is moved in some step `i < n`, and a step moves at most `cap` bytes:
then `|B| ≤ n · cap`, so the step count is at least `⌈|B| / cap⌉`, which is `Perf.fieldLB` on that field. -/
theorem bytes_floor {α : Type*} [MeasurableSpace α] [MeasurableSingletonClass α] (B : Finset α) (n cap : ℕ)
    (hcap : 0 < cap) (s : ℕ → Finset α) (hmove : ∀ b ∈ B, ∃ i < n, b ∈ s i) (hstep : ∀ i < n, (s i).card ≤ cap) :
    B.card ≤ n * cap ∧ ceilDiv B.card cap ≤ n := by
  have hA : (B : Set α) ⊆ ⋃ i ∈ Finset.range n, ((s i : Finset α) : Set α) := by
    intro b hb
    obtain ⟨i, hi, hbi⟩ := hmove b hb
    exact Set.mem_iUnion₂.2 ⟨i, Finset.mem_range.2 hi, hbi⟩
  have h := cover_bound Measure.count (B : Set α) (Finset.range n) (fun i => ((s i : Finset α) : Set α))
    (cap : ENNReal) hA (fun i hi => by
      rw [Measure.count_apply_finset]
      exact_mod_cast hstep i (Finset.mem_range.1 hi))
  rw [Measure.count_apply_finset, Finset.card_range] at h
  have hB : B.card ≤ n * cap := by exact_mod_cast h
  exact ⟨hB, Chain.ceilDiv_le_of_le_mul hcap hB⟩

/-- **Quantisation instance (covering converse).** A codebook of `|I|` words whose `ε`-balls cover the `r`-ball of a
`d`-dimensional real space has `r^d ≤ |I| · ε^d`: at least `d · log₂ (r/ε)` bits. -/
theorem covering_floor {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    [MeasurableSpace E] [BorelSpace E] (μ : Measure E) [μ.IsAddHaarMeasure] {ι : Type*} (I : Finset ι)
    (z : ι → E) {r ε : ℝ} (hr : 0 < r) (hε : 0 < ε) (hcov : ball (0 : E) r ⊆ ⋃ i ∈ I, ball (z i) ε) :
    r ^ Module.finrank ℝ E ≤ I.card * ε ^ Module.finrank ℝ E := by
  have h := cover_bound μ _ I (fun i => ball (z i) ε) _ hcov
    (fun i _ => le_of_eq (μ.addHaar_ball_of_pos (z i) hε))
  rw [μ.addHaar_ball_of_pos 0 hr, ← mul_assoc] at h
  have h0 : μ (ball (0 : E) 1) ≠ 0 := (isOpen_ball.measure_pos μ (nonempty_ball.2 one_pos)).ne'
  have hT : μ (ball (0 : E) 1) ≠ ⊤ := measure_ball_lt_top.ne
  rw [ENNReal.mul_le_mul_iff_left h0 hT, ← ENNReal.ofReal_natCast,
    ← ENNReal.ofReal_mul (Nat.cast_nonneg _)] at h
  exact (ENNReal.ofReal_le_ofReal_iff (by positivity)).1 h

/-- **Stiff floor.** `f` is a 1-Lipschitz map from the weight space onto a `k`-dimensional space of stiff
coordinates (the loss metric dominates the distance there). If an `ε`-codebook covers the admissible weights `A` and
`f '' A` contains the `r`-ball, the codebook has at least `(r/ε)^k` words, however large the weight space is: the
count is set by the stiff dimension `k`, not the parameter count. -/
theorem stiff_floor {X F : Type*} [PseudoMetricSpace X] [NormedAddCommGroup F] [NormedSpace ℝ F]
    [FiniteDimensional ℝ F] [MeasurableSpace F] [BorelSpace F] (μ : Measure F) [μ.IsAddHaarMeasure]
    (f : X → F) (hf : LipschitzWith 1 f) (A : Set X) {ι : Type*} (I : Finset ι) (z : ι → X) {r ε : ℝ}
    (hr : 0 < r) (hε : 0 < ε) (hcov : A ⊆ ⋃ i ∈ I, ball (z i) ε) (hball : ball (0 : F) r ⊆ f '' A) :
    r ^ Module.finrank ℝ F ≤ I.card * ε ^ Module.finrank ℝ F := by
  refine covering_floor μ I (fun i => f (z i)) hr hε ?_
  intro y hy
  obtain ⟨a, ha, rfl⟩ := hball hy
  obtain ⟨i, hi, hai⟩ := Set.mem_iUnion₂.1 (hcov ha)
  refine Set.mem_iUnion₂.2 ⟨i, hi, ?_⟩
  rw [mem_ball] at hai ⊢
  have := hf.dist_le_mul a (z i)
  simp only [NNReal.coe_one, one_mul] at this
  exact lt_of_le_of_lt this hai

/-- Reading at the Q4_K-type layer error `ρ = (ε/r)² = 45/10000`: `(r/ε)² = 10000/45 > 2^(31/4)`, so each stiff
direction costs more than `31/8 = 3.875` bits, and a budget of `1.18` bits per weight covers a stiff fraction of at
most `0.3046` of the weights (`(118/100) / (31/8) = 944/3100 ≤ 0.3046`). -/
theorem q4_bits_per_stiff_dof :
    (2 : ℕ) ^ 31 * 45 ^ 4 < 10000 ^ 4 ∧ 944 * 10000 ≤ 3046 * 3100 := by
  constructor <;> norm_num

/-- W10 stiff fractions of the function (`r99(W H Wᵀ)` of 896 on `ffn_down` blocks 3 / 11 / 23: 99, 770, 441) against
the `0.3046` threshold at Q4_K-type error and 1.18 bits per weight: block 3 (0.110) is below, blocks 11 (0.859) and
23 (0.492) are above. -/
theorem stiff_fraction_vs_w10 :
    99 * 10000 ≤ 3046 * 896 ∧ ¬ (770 * 10000 ≤ 3046 * 896) ∧ ¬ (441 * 10000 ≤ 3046 * 896) := by
  decide

/-! ## 2 · Gauge: rewrites that preserve the function -/

/-- **A finite gauge saves at most `log₂ |G|` bits.** If every admissible weight is carried by some `g ∈ G` into a
cell of a quotient codebook of `|I|` cells of measure at most `c`, and `G` preserves `μ`, then `μ A ≤ |G| · |I| · c`.
With the floor `μ A`, the quotient codebook is smaller than a weight codebook by at most the factor `|G|`. -/
theorem gauge_lift_bound {α G ι : Type*} [MeasurableSpace α] [Group G] [MulAction G α] [Fintype G]
    (μ : Measure α) [SMulInvariantMeasure G α μ] (A : Set α) (I : Finset ι) (s : ι → Set α) (c : ENNReal)
    (hA : ∀ a ∈ A, ∃ g : G, ∃ i ∈ I, g • a ∈ s i) (hc : ∀ i ∈ I, μ (s i) ≤ c) :
    μ A ≤ (Fintype.card G * I.card : ℕ) * c := by
  have hcov : A ⊆ ⋃ p ∈ (Finset.univ ×ˢ I : Finset (G × ι)), p.1⁻¹ • s p.2 := by
    intro a ha
    obtain ⟨g, i, hi, hgi⟩ := hA a ha
    refine Set.mem_iUnion₂.2 ⟨(g, i), Finset.mem_product.2 ⟨Finset.mem_univ _, hi⟩, ?_⟩
    exact Set.mem_inv_smul_set_iff.2 hgi
  have h := cover_bound μ A _ (fun p : G × ι => p.1⁻¹ • s p.2) c hcov (fun p hp => by
    rw [measure_smul]
    exact hc p.2 (Finset.mem_product.1 hp).2)
  rwa [Finset.card_product, Finset.card_univ] at h

/-- Permutations of `n` hidden channels: `n! ≤ 2^(b·n)` when `n ≤ 2^b`. -/
theorem perm_gauge_le (n b : ℕ) (hb : n ≤ 2 ^ b) : n.factorial ≤ 2 ^ (b * n) := by
  calc n.factorial ≤ n ^ n := Nat.factorial_le_pow n
    _ ≤ (2 ^ b) ^ n := Nat.pow_le_pow_left hb n
    _ = 2 ^ (b * n) := by rw [← pow_mul]

/-- Qwen2.5-0.5B: 24 layers of `4864!` FFN permutations, `4864 ≤ 2^13`, so at most `24 · 13 · 4864 = 1,517,568`
bits over `494,032,768` weights, below `0.004` bits per weight. -/
theorem qwen05b_perm_gauge :
    (4864 : ℕ) ≤ 2 ^ 13 ∧ 24 * (13 * 4864) * 1000 ≤ 4 * 494032768 := by
  constructor <;> norm_num

/-- **A continuous exact gauge is Fisher-null.** A direction `x` along which the loss is flat to second order
(`xᵀ F x = 0`, `F` positive semidefinite) leaves every quadratic error unchanged: `(D + t x)ᵀ F (D + t x) = Dᵀ F D`.
Water-filling on `F` gives such a direction zero rate, so the gauge choice moves only the coder's gap to the
floor, never the floor. -/
theorem gauge_null {n : Type*} [Fintype n] [DecidableEq n] {F : Matrix n n ℝ} (hF : F.PosSemidef)
    (x D : n → ℝ) (hx : x ⬝ᵥ F *ᵥ x = 0) (t : ℝ) :
    (D + t • x) ⬝ᵥ F *ᵥ (D + t • x) = D ⬝ᵥ F *ᵥ D := by
  have hFx : F *ᵥ x = 0 := (hF.dotProduct_mulVec_zero_iff x).1 (by simpa using hx)
  have hs : Fᵀ = F := by
    simpa [Matrix.conjTranspose_eq_transpose_of_trivial] using hF.isHermitian.eq
  have hxF : x ᵥ* F = 0 := by rw [← Matrix.mulVec_transpose, hs, hFx]
  rw [Matrix.mulVec_add, Matrix.mulVec_smul, hFx, smul_zero, add_zero, add_dotProduct, smul_dotProduct,
    Matrix.dotProduct_mulVec x F D, hxF, zero_dotProduct, smul_zero, add_zero]

/-! ## 3 · Composition -/

/-- A cost `N` is subadditive under the composition `op`. -/
def Subadd {M R : Type*} [Add R] [LE R] (op : M → M → M) (N : M → R) : Prop :=
  ∀ a b, N (op a b) ≤ N a + N b

/-- A cost `N` never falls below its larger part under `op`: no cancellation. -/
def NoCancel {M R : Type*} [Max R] [LE R] (op : M → M → M) (N : M → R) : Prop :=
  ∀ a b, max (N a) (N b) ≤ N (op a b)

/-- Overlapping two phases costs at most running them in sequence (the ℓ∞ triangle over fields). -/
theorem roofline_subadd (m : Machine) : Subadd Res.add (timeLB m) :=
  fun A B => overlap_le_seq m A B

/-- Overlapping two phases costs at least the larger (demands are non-negative counts). -/
theorem roofline_nocancel (m : Machine) : NoCancel Res.add (timeLB m) :=
  fun A B => overlap_ge_max m A B

/-- A serial dispatch chain adds exactly (ℓ¹). -/
theorem chain_append (c L : ℕ) (ws vs : List ℕ) :
    Chain.chainLB c L (ws ++ vs) = Chain.chainLB c L ws + Chain.chainLB c L vs := by
  simp [Chain.chainLB, List.map_append, List.sum_append]

/-- The square root of a second-order KL is the norm of the score functional in `L²`: subadditive (Minkowski). -/
theorem kl_sqrt_subadd {E : Type*} [NormedAddCommGroup E] : Subadd (· + ·) (fun v : E => ‖v‖) :=
  fun a b => norm_add_le a b

/-- Second-order KL cancels: a perturbation and its negative compose to zero. -/
theorem kl_cancels {E : Type*} [NormedAddCommGroup E] [Nontrivial E] :
    ¬ NoCancel (· + ·) (fun v : E => ‖v‖ ^ 2) := by
  intro h
  obtain ⟨a, ha⟩ := exists_ne (0 : E)
  have := h a (-a)
  simp only [add_neg_cancel, norm_zero, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, norm_neg,
    max_self] at this
  exact ha (norm_eq_zero.1 ((pow_eq_zero_iff two_ne_zero).1 (le_antisymm this (sq_nonneg _))))

/-- **Joint KL over `L` blocks.** `K = ‖Σ v_l‖² ≤ (Σ ‖v_l‖)² ≤ L · Σ ‖v_l‖²`. The first bound is reached when the
block scores are aligned; the per-block sum is reached when they are orthogonal. -/
theorem kl_joint_le {E ι : Type*} [NormedAddCommGroup E] (s : Finset ι) (v : ι → E) :
    ‖∑ i ∈ s, v i‖ ^ 2 ≤ (∑ i ∈ s, ‖v i‖) ^ 2 ∧
      (∑ i ∈ s, ‖v i‖) ^ 2 ≤ s.card * ∑ i ∈ s, ‖v i‖ ^ 2 :=
  ⟨pow_le_pow_left₀ (norm_nonneg _) (norm_sum_le s v) 2, sq_sum_le_card_mul_sum_sq⟩

/-- Two blocks compose sub-additively exactly when their scores are negatively correlated. -/
theorem kl_pair_sub_iff {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (a b : E) :
    ‖a + b‖ ^ 2 < ‖a‖ ^ 2 + ‖b‖ ^ 2 ↔ inner ℝ a b < 0 := by
  rw [norm_add_sq_real]
  constructor <;> intro h <;> linarith

/-- **The two-block form is never hyperbolic.** `q(α, β) = ‖α v + β w‖² = α² A + 2αβ B + β² C` with
`A = ‖v‖²`, `B = ⟪v, w⟫`, `C = ‖w‖²`; its discriminant `σ = B² − A C` is `≤ 0` (Cauchy–Schwarz). The form is
elliptic (`σ < 0`) or parabolic (`σ = 0`), so no measured additivity ratio places a pair of layers on the hyperbolic
face. -/
theorem score_form_disc_nonpos {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (v w : E) :
    (∀ α β : ℝ, ‖α • v + β • w‖ ^ 2 =
        α ^ 2 * ‖v‖ ^ 2 + 2 * α * β * inner ℝ v w + β ^ 2 * ‖w‖ ^ 2) ∧
      inner ℝ v w ^ 2 - ‖v‖ ^ 2 * ‖w‖ ^ 2 ≤ 0 := by
  refine ⟨fun α β => ?_, ?_⟩
  · rw [norm_add_sq_real, norm_smul, norm_smul, mul_pow, mul_pow, real_inner_smul_left, real_inner_smul_right,
      Real.norm_eq_abs, Real.norm_eq_abs, sq_abs, sq_abs]
    ring
  · have h := real_inner_mul_inner_self_le v w
    rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq] at h
    nlinarith [h]

/-- W14 instrument check (per-token KL × 512, hundredths): every arm's joint KL lies under the proved bound
`(Σ_l √k_l)²`, at `0.04–0.062` of it, while joint / `Σ_l k_l` is `0.53–1.03`: the 21 block scores sit near the
orthogonal (ℓ²) composition, far from the aligned (ℓ¹) one. -/
theorem w14_within_minkowski :
    23411 ≤ 424600 ∧ 16892 ≤ 326210 ∧ 203 ≤ 3372 ∧ 181911 ≤ 2949100 ∧ 14947 ≤ 358820 ∧
      100 * 23411 < 6 * 424600 ∧ 100 * 14947 < 5 * 358820 := by
  norm_num

/-! ## 4 · Partition -/

/-- **List scheduling.** The last task (work `w`, speed `s`) starts at `t`, and before `t` every machine was busy, so
`S · t ≤ W − w` with `S` the total speed: the makespan `t + w/s` is at most `W/S + w/s`. -/
theorem greedy_makespan {W S w s t : ℝ} (hS : 0 < S) (hbusy : S * t ≤ W - w) (hw : 0 ≤ w) :
    t + w / s ≤ W / S + w / s := by
  have : t ≤ W / S := by
    rw [le_div_iff₀ hS]
    nlinarith
  linarith

/-- Any schedule takes at least `W/S` (the floor), and greedy at most `W/S + w/s`: refining the partition gains at
most the factor `1 + (w/s) / (W/S)`. -/
theorem refine_gain_le {W S w s T₀ T₁ : ℝ} (hS : 0 < S) (hW : 0 < W) (hws : 0 ≤ w / s)
    (h₀ : T₀ ≤ W / S + w / s) (h₁ : W / S ≤ T₁) : T₀ / T₁ ≤ 1 + (w / s) / (W / S) := by
  have hWS : 0 < W / S := div_pos hW hS
  have hT₁ : 0 < T₁ := lt_of_lt_of_le hWS h₁
  rw [div_le_iff₀ hT₁]
  have hk : 0 ≤ 1 + (w / s) / (W / S) := by positivity
  calc T₀ ≤ W / S + w / s := h₀
    _ = (1 + (w / s) / (W / S)) * (W / S) := by field_simp
    _ ≤ (1 + (w / s) / (W / S)) * T₁ := mul_le_mul_of_nonneg_left h₁ hk

/-- `0bf8b4d` (ffn_down, attn_output, ssm_out: 64 → 256 equal tasks on 24 workers). The homogeneous rounds model
predicts `⌈64/24⌉ = 3` against `⌈256/24⌉ = 11` rounds of a quarter-size task, a gain of `768/704 = 1.0909`. The
measured gains `1.105–1.139` exceed it (refuted), and stay under the greedy bound `1 + 24/64 = 1.375`. -/
theorem partition_0bf8b4d :
    ceilDiv 64 24 = 3 ∧ ceilDiv 256 24 = 11 ∧ (768 : ℚ) / 704 < 1105 / 1000 ∧
      (1139 : ℚ) / 1000 < 1 + 24 / 64 := by
  refine ⟨by decide, by decide, by norm_num, by norm_num⟩

/-- **The fluid floor (performance water level).** Divisible work on workers of speed `s_i > 0`, every worker done by
`T`: the total work is at most `T · Σ s_i`, so the makespan is at least `W / S`. -/
theorem fluid_floor {ι : Type*} (I : Finset ι) (x s : ι → ℝ) (T : ℝ) (hs : ∀ i ∈ I, 0 < s i)
    (hT : ∀ i ∈ I, x i / s i ≤ T) : ∑ i ∈ I, x i ≤ T * ∑ i ∈ I, s i := by
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum fun i hi => ?_
  have := hT i hi
  rw [div_le_iff₀ (hs i hi)] at this
  linarith

/-- Proportional loads `x_i = s_i · W / S` carry all the work and finish together at the level `W / S`: the fluid
floor is attained, every loaded worker at the same level. -/
theorem fluid_attained {ι : Type*} (I : Finset ι) (s : ι → ℝ) (W : ℝ) (hS : 0 < ∑ i ∈ I, s i)
    (hs : ∀ i ∈ I, 0 < s i) :
    ∑ i ∈ I, s i * (W / ∑ j ∈ I, s j) = W ∧
      ∀ i ∈ I, s i * (W / ∑ j ∈ I, s j) / s i = W / ∑ j ∈ I, s j := by
  refine ⟨?_, fun i hi => ?_⟩
  · rw [← Finset.sum_mul]
    field_simp
  · have := (hs i hi).ne'
    field_simp

/-- **Reverse water-filling is optimal (quantisation water level).** Component `i` coded at rate `r_i` leaves
distortion `λ_i e^{-c r_i}` (Gaussian components: `c = 2 ln 2` per bit). An allocation `r*` whose active components
(`r*_i > 0`) sit at the level `θ` and whose inactive ones sit below it leaves the least total distortion among all
allocations with no more total rate. Proof: the tangent line of `exp` at `r*_i`, summed against the budget (the KKT
conditions). -/
theorem waterfill_opt {ι : Type*} (I : Finset ι) (lam r rs : ι → ℝ) {c θ : ℝ} (hc : 0 ≤ c) (hθ : 0 ≤ θ)
    (hlam : ∀ i ∈ I, 0 ≤ lam i) (hr : ∀ i ∈ I, 0 ≤ r i) (hrs : ∀ i ∈ I, 0 ≤ rs i)
    (hle : ∀ i ∈ I, lam i * Real.exp (-(c * rs i)) ≤ θ)
    (hact : ∀ i ∈ I, 0 < rs i → lam i * Real.exp (-(c * rs i)) = θ)
    (hbud : ∑ i ∈ I, r i ≤ ∑ i ∈ I, rs i) :
    ∑ i ∈ I, lam i * Real.exp (-(c * rs i)) ≤ ∑ i ∈ I, lam i * Real.exp (-(c * r i)) := by
  have key : ∀ i ∈ I, lam i * Real.exp (-(c * rs i)) - c * θ * (r i - rs i) ≤
      lam i * Real.exp (-(c * r i)) := by
    intro i hi
    have ha0 : 0 ≤ lam i * Real.exp (-(c * rs i)) := mul_nonneg (hlam i hi) (Real.exp_pos _).le
    have hsplit : lam i * Real.exp (-(c * r i)) =
        lam i * Real.exp (-(c * rs i)) * Real.exp (-(c * (r i - rs i))) := by
      rw [mul_assoc, ← Real.exp_add]
      congr 2
      ring
    have htan : 1 + -(c * (r i - rs i)) ≤ Real.exp (-(c * (r i - rs i))) := by
      have := Real.add_one_le_exp (-(c * (r i - rs i)))
      linarith
    have hprod : 0 ≤ c * (r i - rs i) * (θ - lam i * Real.exp (-(c * rs i))) := by
      rcases (hrs i hi).lt_or_eq with hpos | hzero
      · rw [hact i hi hpos, sub_self, mul_zero]
      · have hl := hle i hi
        rw [← hzero] at hl ⊢
        exact mul_nonneg (mul_nonneg hc (by linarith [hr i hi])) (by linarith)
    rw [hsplit]
    nlinarith [mul_le_mul_of_nonneg_left htan ha0]
  calc ∑ i ∈ I, lam i * Real.exp (-(c * rs i))
      ≤ ∑ i ∈ I, lam i * Real.exp (-(c * rs i)) - c * θ * (∑ i ∈ I, r i - ∑ i ∈ I, rs i) := by
        have : 0 ≤ c * θ * (∑ i ∈ I, rs i - ∑ i ∈ I, r i) :=
          mul_nonneg (mul_nonneg hc hθ) (by linarith)
        linarith
    _ = ∑ i ∈ I, (lam i * Real.exp (-(c * rs i)) - c * θ * (r i - rs i)) := by
        rw [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_sub_distrib]
    _ ≤ _ := Finset.sum_le_sum key

/-- **The water level.** With `r*_i = max 0 (log (λ_i / θ) / c)` the residual distortion of component `i` is
`min λ_i θ`: a component below the level gets no rate, a component above it is coded down to it. -/
theorem waterfill_level {lam θ c : ℝ} (hlam : 0 < lam) (hθ : 0 < θ) (hc : 0 < c) :
    lam * Real.exp (-(c * max 0 (Real.log (lam / θ) / c))) = min lam θ := by
  rcases le_or_gt lam θ with h | h
  · have h1 : Real.log (lam / θ) ≤ 0 :=
      Real.log_nonpos (div_pos hlam hθ).le ((div_le_one hθ).2 h)
    have h2 : Real.log (lam / θ) / c ≤ 0 := div_nonpos_of_nonpos_of_nonneg h1 hc.le
    rw [max_eq_left h2, mul_zero, neg_zero, Real.exp_zero, mul_one, min_eq_left h]
  · have hpos : 0 < Real.log (lam / θ) / c := div_pos (Real.log_pos ((one_lt_div hθ).2 h)) hc
    have hcc : c * (Real.log (lam / θ) / c) = Real.log (lam / θ) := by field_simp
    rw [max_eq_right hpos.le, hcc, Real.exp_neg, Real.exp_log (div_pos hlam hθ), min_eq_right h.le]
    field_simp

/-- **The water level is the floor.** At level `θ`, no allocation of at most `Σ_i max 0 (log (λ_i/θ) / c)` total
rate leaves less distortion than `Σ_i min λ_i θ`. Over the components of every tensor at once this is the global
rate allocation across tensors: the quantisation analogue of `fluid_floor`. -/
theorem waterfill_floor {ι : Type*} (I : Finset ι) (lam r : ι → ℝ) {c θ : ℝ} (hc : 0 < c) (hθ : 0 < θ)
    (hlam : ∀ i ∈ I, 0 < lam i) (hr : ∀ i ∈ I, 0 ≤ r i)
    (hbud : ∑ i ∈ I, r i ≤ ∑ i ∈ I, max 0 (Real.log (lam i / θ) / c)) :
    ∑ i ∈ I, min (lam i) θ ≤ ∑ i ∈ I, lam i * Real.exp (-(c * r i)) := by
  have hopt := waterfill_opt I lam r (fun i => max 0 (Real.log (lam i / θ) / c)) hc.le hθ.le
    (fun i hi => (hlam i hi).le) hr (fun i _ => le_max_left _ _)
    (fun i hi => by
      rw [waterfill_level (hlam i hi) hθ hc]
      exact min_le_right _ _)
    (fun i hi hpos => by
      rw [waterfill_level (hlam i hi) hθ hc]
      apply min_eq_right
      by_contra hlt
      have h1 : Real.log (lam i / θ) ≤ 0 :=
        Real.log_nonpos (div_pos (hlam i hi) hθ).le ((div_le_one hθ).2 (not_le.mp hlt).le)
      rw [max_eq_left (div_nonpos_of_nonpos_of_nonneg h1 hc.le)] at hpos
      exact lt_irrefl _ hpos)
    hbud
  calc ∑ i ∈ I, min (lam i) θ
      = ∑ i ∈ I, lam i * Real.exp (-(c * max 0 (Real.log (lam i / θ) / c))) :=
        Finset.sum_congr rfl fun i hi => (waterfill_level (hlam i hi) hθ hc).symm
    _ ≤ _ := hopt

/-- **The Kronecker metric is diagonal in the factor eigenbases.** With `G = U diag(g) Uᵀ`, `H = V diag(h) Vᵀ`
(`UᵀU = 1`, `VᵀV = 1`) and the weight error `D = U E Vᵀ`, the K-FAC form `tr(Dᵀ G D H) = Σ_{ij} g_i h_j E_{ij}²`:
each coefficient of the rotated error is one component of weight `g_i h_j`, the input of `waterfill_floor`. -/
theorem kron_diag {m n : Type*} [Fintype m] [Fintype n] [DecidableEq m] [DecidableEq n]
    (U : Matrix m m ℝ) (V : Matrix n n ℝ) (hU : Uᵀ * U = 1) (hV : Vᵀ * V = 1) (g : m → ℝ) (h : n → ℝ)
    (E : Matrix m n ℝ) :
    trace ((U * E * Vᵀ)ᵀ * (U * diagonal g * Uᵀ) * (U * E * Vᵀ) * (V * diagonal h * Vᵀ)) =
      ∑ i, ∑ j, g i * h j * E i j ^ 2 := by
  have hU' : ∀ X : Matrix m n ℝ, Uᵀ * (U * X) = X := fun X => by
    rw [← Matrix.mul_assoc, hU, Matrix.one_mul]
  have hV' : ∀ Y : Matrix n n ℝ, Vᵀ * (V * Y) = Y := fun Y => by
    rw [← Matrix.mul_assoc, hV, Matrix.one_mul]
  simp only [transpose_mul, transpose_transpose, Matrix.mul_assoc, hU', hV']
  rw [Matrix.trace_mul_comm]
  simp only [Matrix.mul_assoc, hV, Matrix.mul_one]
  simp only [Matrix.trace, Matrix.diag_apply, Matrix.mul_apply, Matrix.transpose_apply]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  simp only [Matrix.diagonal_apply, ite_mul, mul_ite, zero_mul, mul_zero, Finset.sum_ite_eq, Finset.sum_ite_eq',
    Finset.mem_univ, if_true]
  ring

/-- W15a ray scan (`tools/onebit/ray_scan.py`, receipt `2026-10-11-ray.json`, 100 C4
chunks of 512 tokens, all 21 `ffn_down` tensors of Qwen2.5-0.5B moved jointly along `W + tD`). Per-token KL at
`t = 1/8` and `t = 1` (six significant figures). Verdicts:
1. the quadratic form carries the full step: `KL(1) / (64 · KL(1/8)) < 1.16` for `IQ1_S`, `IQ1_M`, Q4_K_M and
   `had-gptq1`; RTN exceeds `2.7` (outside the quadratic regime);
2. the exact quadratic `64 · KL(1/8)` orders `had-gptq1 > IQ1_S > IQ1_M`, the P1 KLD order, so the W14 sampled-Fisher
   inversion (`had-gptq1` below `IQ1_S`) is estimator error;
3. the odd remainder of `had-gptq1` at `t = 1` is negative: `KL(D) < KL(-D)`. -/
theorem ray_w15a :
    (449916 : ℚ) / 10 ^ 6 / (64 * (61071 / 10 ^ 7)) < 116 / 100 ∧
    (344035 : ℚ) / 10 ^ 6 / (64 * (477229 / 10 ^ 8)) < 116 / 100 ∧
    (450143 : ℚ) / 10 ^ 8 / (64 * (70449 / 10 ^ 9)) < 116 / 100 ∧
    (460010 : ℚ) / 10 ^ 6 / (64 * (654562 / 10 ^ 8)) < 116 / 100 ∧
    (27 : ℚ) / 10 < 982565 / 10 ^ 5 / (64 * (555673 / 10 ^ 7)) ∧
    (61071 : ℚ) / 10 ^ 7 < 654562 / 10 ^ 8 ∧ (477229 : ℚ) / 10 ^ 8 < 61071 / 10 ^ 7 ∧
    (460010 : ℚ) / 10 ^ 6 < 106903 / 10 ^ 5 := by
  norm_num

/-- W15b K-FAC water level (`tools/onebit/kfac_floor.py`, receipt `2026-10-11-kfac.json`,
100 C4 chunks, 4 sampled targets, all 168 linear tensors of Qwen2.5-0.5B). Per-token KL, five significant figures.
1. K-FAC `½ tr(DᵀGDH)` summed over the 21 `ffn_down` tensors is within `[1/2, 2]` of the exact quadratic
   `64 · KL(1/8)` for `IQ1_S` 0.42329/0.39085, `IQ1_M` 0.33660/0.30543, Q4_K_M 0.0036205/0.0045087 and `had-gptq1`
   0.27639/0.41892, but it orders `had-gptq1` lowest where the exact quadratic orders it highest (prediction
   `kfac-1-1` order clause refuted);
2. separable-variance water-filling at 1.18 b/w leaves 0.26454, more than 16 times Q4_K_M's K-FAC KL 0.015675 on the
   same 168 tensors; the oracle reference leaves 0.10378, more than 6 times;
3. the separable rate that matches Q4_K_M is above 3.17 b/w, the oracle rate above 2.35 b/w. -/
theorem kfac_w15b :
    ((1 : ℚ) / 2 < 42329 / 39085 ∧ (42329 : ℚ) / 39085 < 2) ∧
    ((1 : ℚ) / 2 < 33660 / 30543 ∧ (33660 : ℚ) / 30543 < 2) ∧
    ((1 : ℚ) / 2 < 36205 / 45087 ∧ (36205 : ℚ) / 45087 < 2) ∧
    ((1 : ℚ) / 2 < 27639 / 41892 ∧ (27639 : ℚ) / 41892 < 2) ∧
    ((27639 : ℚ) < 33660 ∧ (33660 : ℚ) < 42329 ∧ (39085 : ℚ) < 41892) ∧
    (16 * (15675 : ℚ) / 10 ^ 6 < 26454 / 10 ^ 5 ∧ 6 * (15675 : ℚ) / 10 ^ 6 < 10378 / 10 ^ 5) ∧
    ((317 : ℚ) / 100 < 31760 / 10 ^ 4 ∧ (235 : ℚ) / 100 < 23526 / 10 ^ 4) := by
  norm_num

/-- **The high-resolution regime.** When the level `θ` lies below every `λ_i` (no component is dropped), the
water-filling distortion is `Σ min λ_i θ = |I| θ` and the rate constraint `Σ log (λ_i / θ) / c = R` fixes
`log θ = (Σ log λ_i − c R) / |I|`: the log-determinant form, so `θ` falls by the factor `exp (−c / |I|)` per unit
of rate. This is the Gaussian achievability side; it is not a converse for a non-Gaussian source. -/
theorem waterfill_highres {ι : Type*} (I : Finset ι) (lam : ι → ℝ) {c θ R : ℝ} (hc : 0 < c) (hθ : 0 < θ)
    (hI : I.Nonempty) (hlam : ∀ i ∈ I, 0 < lam i) (hge : ∀ i ∈ I, θ ≤ lam i)
    (hR : ∑ i ∈ I, Real.log (lam i / θ) / c = R) :
    ∑ i ∈ I, min (lam i) θ = I.card * θ ∧
      Real.log θ = (∑ i ∈ I, Real.log (lam i) - c * R) / I.card := by
  have hcard : (0 : ℝ) < I.card := by exact_mod_cast hI.card_pos
  constructor
  · rw [Finset.sum_congr rfl fun i hi => min_eq_right (hge i hi)]
    simp
  · have hsum : ∑ i ∈ I, Real.log (lam i / θ) = c * R := by
      rw [← hR, ← Finset.sum_div]
      field_simp
    have hdiv : ∑ i ∈ I, Real.log (lam i / θ) = ∑ i ∈ I, Real.log (lam i) - I.card * Real.log θ := by
      rw [Finset.sum_congr rfl fun i hi => Real.log_div (hlam i hi).ne' hθ.ne']
      simp [Finset.sum_sub_distrib]
    rw [eq_div_iff hcard.ne']
    linarith

/-- **Distortion ratio per unit rate.** In the high-resolution regime at two rates `R`, `R'` (levels `θ`, `θ'`
both below every `λ_i`), `KL(R) / KL(R') = exp (−c (R − R') / |I|)`; with `c = 2 ln 2` this is `4^{−(R−R')/|I|}`
for `R` counted in total bits, i.e. `4^{−ΔR}` per bit per weight. -/
theorem waterfill_exclusion {ι : Type*} (I : Finset ι) (lam : ι → ℝ) {c θ θ' R R' : ℝ} (hc : 0 < c)
    (hθ : 0 < θ) (hθ' : 0 < θ') (hI : I.Nonempty) (hlam : ∀ i ∈ I, 0 < lam i)
    (hge : ∀ i ∈ I, θ ≤ lam i) (hge' : ∀ i ∈ I, θ' ≤ lam i)
    (hR : ∑ i ∈ I, Real.log (lam i / θ) / c = R) (hR' : ∑ i ∈ I, Real.log (lam i / θ') / c = R') :
    (∑ i ∈ I, min (lam i) θ) / (∑ i ∈ I, min (lam i) θ') = Real.exp (-(c * (R - R')) / I.card) := by
  have hcard : (0 : ℝ) < I.card := by exact_mod_cast hI.card_pos
  obtain ⟨h1, h2⟩ := waterfill_highres I lam hc hθ hI hlam hge hR
  obtain ⟨h1', h2'⟩ := waterfill_highres I lam hc hθ' hI hlam hge' hR'
  rw [h1, h1', mul_div_mul_left _ _ hcard.ne']
  have hlog : Real.log (θ / θ') = -(c * (R - R')) / I.card := by
    rw [Real.log_div hθ.ne' hθ'.ne', h2, h2']
    field_simp
    ring
  rw [← hlog, Real.exp_log (div_pos hθ hθ')]

/-- **The Shannon-lower-bound gap, as algebra.** Hypothesis `hslb` (an `anchor`, Shannon 1959, not proved here):
every code that reaches distortion `D` has rate at least `Rg D − J`, where `Rg` is the Gaussian rate and `J ≥ 0` the
negentropy (marginal plus total correlation). Then a code at rate `R_T` that reaches `D_Q` requires
`J ≥ Rg D_Q − R_T`: the negentropy a 1.18 b/w code must have in the source. -/
theorem slb_gap {reach : ℝ → ℝ → Prop} {Rg : ℝ → ℝ} {J RT DQ : ℝ}
    (hslb : ∀ R D, reach R D → Rg D - J ≤ R) (h : reach RT DQ) : Rg DQ - RT ≤ J := by
  have := hslb RT DQ h
  linarith

/-- **Verdict for the converse measurements** (Qwen2.5-0.5B, 168 tensors, 100 C4 chunks;
`tools/onebit/converse.py`, receipts `2026-10-11-converse-{controls,main,ekfac}`). Grade `measured` for the numbers,
`anchor` for the Shannon lower bound. In order:

1. `K0` from log-determinants 1.1918 is within 15% of the bisection `K0` at 4.5 b/w (1.2271) and at 1.18 b/w (1.3123);
2. the separable-normalised K-FAC coefficients at the Q4-matching level have marginal negentropy 6.9e-5 b per
   component (below 1e-3), and 0.00815 b in the raw weight basis (below 1e-2);
3. the causal-neighbour context gain is -0.031 b, the same sign and size as the iid control (-0.032 b): no context gain;
4. the Gaussian total-correlation excess over the shuffled null is 0.14143 b per weight, so marginal plus structure
   (0.00815 + 0.14143) is below 3/20 b/w;
5. `slb_gap` demands `J ≥ Rg(D_Q4) − 1.18`: 3.174 − 1.18 = 1.994 b/w for the separable rate and 2.353 − 1.18 = 1.173
   b/w for the oracle rate; the measured `J` is below 3/20, a factor above 12 and above 7 short;
6. eigenvalue-corrected K-FAC leaves `had-gptq1` at 0.70085 of the exact quadratic (K-FAC 0.65897), below 0.85, and
   the order flip persists (ekfac `had-gptq1` 0.29360 < `IQ1_M` 0.39060, exact 0.41892 > `IQ1_S` 0.39085). -/
theorem converse_w15c :
    ((85 : ℚ) / 100 < 11918 / 12271 ∧ (11918 : ℚ) / 12271 < 115 / 100) ∧
    ((85 : ℚ) / 100 < 11918 / 13123 ∧ (11918 : ℚ) / 13123 < 115 / 100) ∧
    ((69 : ℚ) / 10 ^ 6 < 1 / 10 ^ 3 ∧ (815 : ℚ) / 10 ^ 5 < 1 / 10 ^ 2) ∧
    ((-31 : ℚ) / 10 ^ 3 < 0 ∧ (-32 : ℚ) / 10 ^ 3 < 0) ∧
    ((815 : ℚ) / 10 ^ 5 + 14143 / 10 ^ 5 < 3 / 20) ∧
    ((12 : ℚ) * (3 / 20) < 3174 / 1000 - 118 / 100 ∧ (7 : ℚ) * (3 / 20) < 2353 / 1000 - 118 / 100) ∧
    (((70085 : ℚ) / 10 ^ 5 < 85 / 100) ∧ (29360 : ℚ) < 39060 ∧ (39085 : ℚ) < 41892) := by
  norm_num

/-! ## 5 · Carrier ↔ coder -/

/-- **The rate is the decode numerator.** A decode step reads `N · R / 8` bytes at bandwidth `BW`; the floor ratio of
two rates on the same model is the rate ratio. -/
theorem tg_floor_ratio {N BW R₁ R₂ : ℝ} (hN : 0 < N) (hBW : 0 < BW) (hR₂ : 0 < R₂) :
    (N * R₁ / 8 / BW) / (N * R₂ / 8 / BW) = R₁ / R₂ := by
  field_simp

/-- Qwen3.8-27B at `1.18` bits per weight: `3.9825` GB, decode floor below `4.15` ms per token at `960` GB/s. -/
theorem qwen27b_target_floor :
    (27 * 10 ^ 9 : ℚ) * 118 / 100 / 8 = 3982500000 ∧ (3982500000 : ℚ) / (960 * 10 ^ 9) < 415 / 100000 := by
  constructor <;> norm_num

/-! ## 8 · Faces of the pipeline -/

/-- **Discriminant of a real 2×2 matrix.** `disc2 M = (tr M)²/4 − det M`. Its sign gives the face of the algebra
generated by `M`: negative (elliptic), zero (parabolic), positive (hyperbolic). -/
noncomputable def disc2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ := M.trace ^ 2 / 4 - M.det

/-- **Discriminant in entries.** `disc2 [[a, b], [c, d]] = ((a − d)/2)² + b c`. -/
theorem disc2_eq (a b c d : ℝ) : disc2 !![a, b; c, d] = ((a - d) / 2) ^ 2 + b * c := by
  simp [disc2, Matrix.trace_fin_two, Matrix.det_fin_two_of]
  ring

/-- **Cayley–Hamilton for 2×2.** With `N = M − (tr M / 2) · 1` (trace zero), `N² = disc2 M · 1`. So the algebra
generated by `M` is `R[x]/(x² − disc2 M)`. -/
theorem disc2_sq (M : Matrix (Fin 2) (Fin 2) ℝ) :
    (M - (M.trace / 2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) * (M - (M.trace / 2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) =
      disc2 M • (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [disc2, Matrix.mul_apply, Fin.sum_univ_two, Matrix.trace_fin_two, Matrix.det_fin_two, Matrix.one_apply] <;>
    ring

/-- **The face of a 2×2 step is the canonical σ.** With `N = M − (tr M / 2)·1`, the point `a·1 + b·N` multiplies as
`Apeiron.Trichotomy.mul (disc2 M)` on its coordinates `(a, b)`: the algebra generated by `M` is `R[x]/(x² − disc2 M)`. -/
theorem face_mul_canonical (M : Matrix (Fin 2) (Fin 2) ℝ) (a b c d : ℝ) :
    let N := M - (M.trace / 2) • (1 : Matrix (Fin 2) (Fin 2) ℝ)
    (a • 1 + b • N) * (c • 1 + d • N) =
      (Apeiron.Trichotomy.mul (disc2 M) (a, b) (c, d)).1 • 1 + (Apeiron.Trichotomy.mul (disc2 M) (a, b) (c, d)).2 • N := by
  intro N
  have hN : N * N = disc2 M • (1 : Matrix (Fin 2) (Fin 2) ℝ) := disc2_sq M
  simp only [add_mul, mul_add, Matrix.smul_mul, Matrix.mul_smul, smul_smul, one_mul, mul_one, hN]
  module

/-- **Elliptic face (rotation).** A plane rotation by `θ` with `sin θ ≠ 0` has `disc2 = −sin² θ < 0`. This is the
Hadamard/incoherence rotation. -/
theorem rotation_elliptic (θ : ℝ) (h : Real.sin θ ≠ 0) :
    disc2 !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ] < 0 := by
  have h2 : 0 < Real.sin θ ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 h))
  rw [disc2_eq]
  nlinarith [h2]

/-- **Parabolic face (shear).** A unit upper-triangular `[[1, c], [0, 1]]` has `disc2 = 0`. This is the unit-triangular
factor of the LDL/Cholesky error feedback. -/
theorem shear_parabolic (c : ℝ) : disc2 !![1, c; 0, 1] = 0 := by
  rw [disc2_eq]
  ring

/-- **Nilpotent shear.** `([[1, c], [0, 1]] − 1)² = 0`: the shear minus identity is nilpotent of index 2. -/
theorem shear_nilpotent (c : ℝ) :
    (!![1, c; 0, 1] - (1 : Matrix (Fin 2) (Fin 2) ℝ)) * (!![1, c; 0, 1] - (1 : Matrix (Fin 2) (Fin 2) ℝ)) = 0 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.one_apply]

/-- **Hyperbolic face (rebalance).** `disc2 diag(s, s⁻¹) = ((s − s⁻¹)/2)² > 0` for `s > 0`, `s ≠ 1`: the AWQ/SmoothQuant
scale pair lies on the hyperbolic face. -/
theorem rebalance_hyperbolic (s : ℝ) (hs : 0 < s) (h1 : s ≠ 1) : 0 < disc2 !![s, 0; 0, s⁻¹] := by
  have hne : s - s⁻¹ ≠ 0 := by
    intro h
    have e : s * (s - s⁻¹) = s * s - 1 := by rw [mul_sub, mul_inv_cancel₀ hs.ne']
    rw [h, mul_zero] at e
    have h3 : (s - 1) * (s + 1) = 0 := by linear_combination -e
    rcases mul_eq_zero.mp h3 with h4 | h4
    · exact h1 (by linarith)
    · linarith
  rw [disc2_eq, mul_zero, add_zero]
  exact lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 (div_ne_zero hne two_ne_zero)))

/-- **Rebalance preserves the product.** Scaling a weight column by `s` and the matching activation row by `s⁻¹`
leaves `(s w)(s⁻¹ a) = w a` unchanged. -/
theorem rebalance_preserves_product (s w a : ℝ) (hs : s ≠ 0) : (s * w) * (s⁻¹ * a) = w * a := by
  have e : (s * w) * (s⁻¹ * a) = (s * s⁻¹) * (w * a) := by ring
  rw [e, mul_inv_cancel₀ hs, one_mul]

/-- **Corrector identity.** For `A = ‖p‖²`, `B = ⟪δ, p⟫`, `C = ‖δ‖²`:
`‖δ + c p‖² A = C A − B² + (B + c A)²`. Every `c` leaves a non-negative remainder `(B + cA)²`. -/
theorem corrector_identity {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (δ p : E) (c : ℝ) :
    ‖δ + c • p‖ ^ 2 * ‖p‖ ^ 2 =
      ‖δ‖ ^ 2 * ‖p‖ ^ 2 - inner ℝ δ p ^ 2 + (inner ℝ δ p + c * ‖p‖ ^ 2) ^ 2 := by
  rw [norm_add_sq_real, norm_smul, real_inner_smul_right, Real.norm_eq_abs, mul_pow, sq_abs]
  ring

/-- **Corrector residual (lower bound).** The residual of a corrector `δ + c p` in the Fisher metric is at least the
negative pair discriminant over `‖p‖²`: `‖δ‖²‖p‖² − ⟪δ,p⟫² ≤ ‖δ + c p‖²‖p‖²`. The residual is the score-form
discriminant of `score_form_disc_nonpos` with its sign reversed. -/
theorem corrector_residual {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (δ p : E) (c : ℝ) :
    ‖δ‖ ^ 2 * ‖p‖ ^ 2 - inner ℝ δ p ^ 2 ≤ ‖δ + c • p‖ ^ 2 * ‖p‖ ^ 2 := by
  have h := corrector_identity δ p c
  have h2 := sq_nonneg (inner ℝ δ p + c * ‖p‖ ^ 2)
  linarith

/-- **Corrector optimum.** For `p ≠ 0` the choice `c = −⟪δ,p⟫/‖p‖²` attains the bound with equality:
`‖δ + c p‖² ‖p‖² = ‖δ‖²‖p‖² − ⟪δ,p⟫²`. -/
theorem corrector_attained {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (δ p : E) (hp : p ≠ 0) :
    ‖δ + (-(inner ℝ δ p) / ‖p‖ ^ 2) • p‖ ^ 2 * ‖p‖ ^ 2 = ‖δ‖ ^ 2 * ‖p‖ ^ 2 - inner ℝ δ p ^ 2 := by
  have hA : ‖p‖ ^ 2 ≠ 0 := pow_ne_zero 2 (norm_ne_zero_iff.mpr hp)
  have h0 : inner ℝ δ p + (-(inner ℝ δ p) / ‖p‖ ^ 2) * ‖p‖ ^ 2 = 0 := by
    field_simp
    ring
  rw [corrector_identity, h0]
  ring

/-- **Parabolic boundary of the corrector.** The optimal residual vanishes exactly when the pair discriminant
`⟪δ,p⟫² − ‖δ‖²‖p‖²` vanishes, that is when `δ` and `p` are parallel (Cauchy–Schwarz equality). -/
theorem corrector_zero_iff {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (δ p : E) (hp : p ≠ 0) :
    ‖δ + (-(inner ℝ δ p) / ‖p‖ ^ 2) • p‖ ^ 2 * ‖p‖ ^ 2 = 0 ↔
      inner ℝ δ p ^ 2 - ‖δ‖ ^ 2 * ‖p‖ ^ 2 = 0 := by
  rw [corrector_attained δ p hp]
  constructor <;> intro h <;> linarith

/-- **Floor invariance under change of basis.** For invertible `T`, the product `(T S Tᵀ)((Tᵀ)⁻¹ F T⁻¹)` is the
conjugate `T (S F) T⁻¹`, so its characteristic polynomial equals that of `S F`. The reverse water-filling floor is a
function of the eigenvalues of `S F`; no invertible reparametrisation of the weights (rotation, rebalance, triangular
feedback) moves it. -/
theorem spectrum_conj {n : Type*} [Fintype n] [DecidableEq n] (S F T : Matrix n n ℝ) (hT : IsUnit T.det) :
    ((T * S * Tᵀ) * ((Tᵀ)⁻¹ * F * T⁻¹)).charpoly = (S * F).charpoly := by
  have hTT : IsUnit Tᵀ.det := by rw [Matrix.det_transpose]; exact hT
  have h1 : Tᵀ * (Tᵀ)⁻¹ = 1 := Matrix.mul_nonsing_inv Tᵀ hTT
  have hinv : T⁻¹ * T = 1 := Matrix.nonsing_inv_mul T hT
  have key : (T * S * Tᵀ) * ((Tᵀ)⁻¹ * F * T⁻¹) = T * (S * F) * T⁻¹ := by
    calc (T * S * Tᵀ) * ((Tᵀ)⁻¹ * F * T⁻¹) = T * S * (Tᵀ * (Tᵀ)⁻¹) * F * T⁻¹ := by
          simp only [Matrix.mul_assoc]
      _ = T * (S * F) * T⁻¹ := by rw [h1]; simp only [Matrix.mul_one, Matrix.mul_assoc]
  rw [key, Matrix.charpoly_mul_comm (T * (S * F)) T⁻¹, ← Matrix.mul_assoc T⁻¹ T (S * F), hinv, Matrix.one_mul]

/-- **Verdict, method gap at 1.133 b/w.** The fold-block KL `3.0897` over the uniform-rate separable floor `0.3887`
exceeds `7`. Measured inputs from the W15 receipts. -/
theorem method_gap_1133 : (3.0897 : ℝ) / 0.3887 > 7 := by norm_num

/-- **Verdict, reverse water-filling gain at 1.133 b/w.** The uniform-rate separable floor `0.3887` over the
water-filled floor `0.2837` lies in `(1.3, 1.4)`. -/
theorem waterfill_gain_1133 : (0.3887 : ℝ) / 0.2837 > 1.3 ∧ (0.3887 : ℝ) / 0.2837 < 1.4 := by
  constructor <;> norm_num

/-- **Verdict, floor gaps against Q4_K_M.** The oracle floor `0.1128` and the separable floor `0.2837`, over the
Q4_K_M K-FAC KL `0.015665`, exceed `7` and `18`. -/
theorem floor_gap_1133 : (0.1128 : ℝ) / 0.015665 > 7 ∧ (0.2837 : ℝ) / 0.015665 > 18 := by
  constructor <;> norm_num

/-- **Verdict, variance ladder refuted.** The best structured variance model's coefficient-only rate `R* = 3.129`
bits per weight lies within `0.1` of the separable `3.176`: it recovers less than `0.1` b/w. -/
theorem variance_ladder_refuted : (3.129 : ℝ) > 3.176 - 0.1 := by norm_num

end CostCorrespondence

end Perf
