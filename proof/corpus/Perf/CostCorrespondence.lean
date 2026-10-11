import Mathlib.MeasureTheory.Measure.Lebesgue.EqHaar
import Mathlib.MeasureTheory.Measure.Count
import Mathlib.MeasureTheory.Group.Action
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.Matrix.Order
import Mathlib.Algebra.Order.Chebyshev
import Perf.Roofline
import Perf.Chain
import Perf.Overlap

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
| partition | rounds `⌈k/p⌉`; greedy ≤ `W/S + w/s` | water level across tensors (not built here) | `greedy_makespan`, `refine_gain_le` |
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

end CostCorrespondence

end Perf
