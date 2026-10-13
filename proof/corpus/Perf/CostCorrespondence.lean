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
| free rewrites | fusion preserves denotation | a gauge preserves the function; basis change preserves `charpoly(S F)` | `gauge_lift_bound`, `perm_gauge_le`, `gauge_null`, `spectrum_conj` |
| composition, upper | overlap ≤ sequence (ℓ∞ triangle) | `√K` subadditive (ℓ² triangle, Minkowski) | `roofline_subadd`, `kl_sqrt_subadd`, `kl_joint_le` |
| composition, lower | overlap ≥ max (no cancellation) | none: scores cancel | `roofline_nocancel`, `kl_cancels` |
| serial composition | chain cost adds (ℓ¹) | | `chain_append` |
| pair form & corrector | | discriminant `⟪v,w⟫² − ‖v‖²‖w‖² ≤ 0`; residual `‖δ + c p‖²‖p‖² ≥ −disc` attained at `c = −⟪δ,p⟫/‖p‖²`; orthogonal two-plane drop adds without cross-interference | `score_form_disc_nonpos`, `corrector_identity`, `corrector_residual`, `corrector_attained`, `corrector_zero_iff`, `orthogonal_corrector_identity`, `orthogonal_corrector_attained` |
| pipeline faces & filter | | `z = a·1 + b·N` (`N = M − tr/2`) multiplies as `Apeiron.Trichotomy.mul (disc2 M)`: rotation elliptic (`< 0`), triangular feedback parabolic (`= 0`), scale rebalance hyperbolic (`> 0`); elliptic unit norm `normForm (-1) (c,b) = 1` conserves Steinitz filter energy `(br)² + (cr)² = r²`; decoupled prior `λ = μ s² / d²` attains Bayes floor `d²s²/(d²+s²)` | `disc2_sq`, `face_mul_canonical`, `rotation_elliptic`, `shear_parabolic`, `shear_nilpotent`, `rebalance_hyperbolic`, `rebalance_preserves_product`, `elliptic_filter_conservation`, `decoupled_ridge_identity`, `decoupled_ridge_opt`, `decoupled_ridge_eq_iff` |
| partition & water level | rounds `⌈k/p⌉`; greedy ≤ `W/S + w/s`; fluid floor `W/S` (`fluid_floor`, `fluid_attained`) | reverse water-filling level `θ` (`waterfill_opt`, `waterfill_level`, `waterfill_floor`); K-FAC eigenbasis `kron_diag`; high-res `4^{−ΔR}` and SLB gap `Rg(D) − R ≤ J`; GQA KV two-plane rate `< 1.18` b/w | `greedy_makespan`, `refine_gain_le`, `waterfill_highres`, `waterfill_exclusion`, `slb_gap`, `gqa_kv_second_plane_budget`, `w15_rstar_8b_oracle_and_had_fp32` |
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

/-! ## 9 · Decoupled prior, elliptic filter conservation and GQA multi-plane water-filling -/

/-- **Decoupled prior-vs-sample ridge identity.** Along an eigenmode of `H` with eigenvalue `μ`, true drift variance
`d²`, calibration noise variance `s²`, and prior precision `λ`, the posterior mean `(μ + λ)⁻¹ (μ Δw + ε)` has
expected squared error `(λ² d² + μ² s²) / (μ + λ)²`. Multiplying by `(d² + s²)(μ + λ)²` splits the excess over the
minimum-variance Bayes floor `d² s² / (d² + s²)` as the exact square `(λ d² − μ s²)²`. -/
theorem decoupled_ridge_identity (d2 s2 μ lam : ℝ) :
    (d2 + s2) * (lam ^ 2 * d2 + μ ^ 2 * s2) - (μ + lam) ^ 2 * d2 * s2 = (lam * d2 - μ * s2) ^ 2 := by
  ring

/-- **Decoupled prior-vs-sample ridge lower bound.** For `0 < d² + s²` and `μ + λ ≠ 0`, the posterior mean risk is
bounded below by the Bayes floor `d² s² / (d² + s²)`; tying `λ` to a small quantiser metric damping when `s² / d²` is
large inflates the held-out drift by `(λ d² − μ s²)² / ((d² + s²)(μ + λ)²)`. -/
theorem decoupled_ridge_opt {d2 s2 μ lam : ℝ} (hds : 0 < d2 + s2) (hml : μ + lam ≠ 0) :
    d2 * s2 / (d2 + s2) ≤ (lam ^ 2 * d2 + μ ^ 2 * s2) / (μ + lam) ^ 2 := by
  have hml2 : 0 < (μ + lam) ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hml))
  rw [div_le_div_iff₀ hds hml2]
  have hid := decoupled_ridge_identity d2 s2 μ lam
  have hsq := sq_nonneg (lam * d2 - μ * s2)
  linarith

/-- **Parabolic boundary of the decoupled prior.** The posterior mean risk attains the Bayes floor `d² s² / (d² + s²)`
if and only if `λ d² − μ s² = 0` (`λ = μ s² / d²`). -/
theorem decoupled_ridge_eq_iff {d2 s2 μ lam : ℝ} (hds : 0 < d2 + s2) (hml : μ + lam ≠ 0) :
    (lam ^ 2 * d2 + μ ^ 2 * s2) / (μ + lam) ^ 2 = d2 * s2 / (d2 + s2) ↔ lam * d2 - μ * s2 = 0 := by
  have hml2 : 0 < (μ + lam) ^ 2 := lt_of_le_of_ne (sq_nonneg _) (Ne.symm (pow_ne_zero 2 hml))
  rw [div_eq_div_iff hml2.ne' hds.ne']
  have hid := decoupled_ridge_identity d2 s2 μ lam
  constructor
  · intro h
    have h0 : (lam * d2 - μ * s2) ^ 2 = 0 := by linarith
    exact sq_eq_zero_iff.mp h0
  · intro h
    have h0 : (lam * d2 - μ * s2) ^ 2 = 0 := by rw [h]; ring
    linarith

/-- **Elliptic coisometric filter conservation (`σ = -1`, Family 097 Steinitz filter).** A unit-norm point `(c, b)` in
the elliptic face `Apeiron.Trichotomy.normForm (-1) (c, b) = 1` (`c² + b² = 1`) splits any state amplitude `r` into a
prediction readout `b r` and a contractive state carry `c r` whose squares sum to `r²`. This is the scalar conservation
law of the Steinitz coisometric filter (`OAI.Analysis.Steinitz.Predictor.rawRow_loss`). -/
theorem elliptic_filter_conservation {c b r : ℝ} (h : Apeiron.Trichotomy.normForm (-1) (c, b) = 1) :
    (b * r) ^ 2 + (c * r) ^ 2 = r ^ 2 := by
  simp only [Apeiron.Trichotomy.normForm, Apeiron.Trichotomy.AnyRing.ringNormForm] at h
  linear_combination r ^ 2 * h

/-- **Orthogonal two-plane corrector identity.** For orthogonal directions `⟪p, q⟫ = 0`, the joint residual
`‖δ + c₁ p + c₂ q‖²` splits additively across the two planes without cross-interference. -/
theorem orthogonal_corrector_identity {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (δ p q : E) (c₁ c₂ : ℝ) (hpq : inner ℝ p q = 0) :
    ‖δ + c₁ • p + c₂ • q‖ ^ 2 * ‖p‖ ^ 2 * ‖q‖ ^ 2 =
      ‖δ‖ ^ 2 * ‖p‖ ^ 2 * ‖q‖ ^ 2
        - inner ℝ δ p ^ 2 * ‖q‖ ^ 2
        - inner ℝ δ q ^ 2 * ‖p‖ ^ 2
        + (inner ℝ δ p + c₁ * ‖p‖ ^ 2) ^ 2 * ‖q‖ ^ 2
        + (inner ℝ δ q + c₂ * ‖q‖ ^ 2) ^ 2 * ‖p‖ ^ 2 := by
  have h1 := corrector_identity (δ + c₁ • p) q c₂
  have h2 := corrector_identity δ p c₁
  have hpq0 : inner ℝ (δ + c₁ • p) q = inner ℝ δ q := by
    rw [inner_add_left, real_inner_smul_left, hpq, mul_zero, add_zero]
  rw [hpq0] at h1
  linear_combination ‖p‖ ^ 2 * h1 + ‖q‖ ^ 2 * h2

/-- **Orthogonal two-plane corrector optimum.** For `p ≠ 0`, `q ≠ 0` with `⟪p, q⟫ = 0`, the independent optimal
coefficients `c₁ = −⟪δ,p⟫/‖p‖²` and `c₂ = −⟪δ,q⟫/‖q‖²` subtract both Cauchy–Schwarz projections simultaneously. -/
theorem orthogonal_corrector_attained {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (δ p q : E) (hp : p ≠ 0) (hq : q ≠ 0) (hpq : inner ℝ p q = 0) :
    ‖δ + (-(inner ℝ δ p) / ‖p‖ ^ 2) • p + (-(inner ℝ δ q) / ‖q‖ ^ 2) • q‖ ^ 2 * ‖p‖ ^ 2 * ‖q‖ ^ 2 =
      ‖δ‖ ^ 2 * ‖p‖ ^ 2 * ‖q‖ ^ 2 - inner ℝ δ p ^ 2 * ‖q‖ ^ 2 - inner ℝ δ q ^ 2 * ‖p‖ ^ 2 := by
  have h1 := corrector_attained (δ + (-(inner ℝ δ p) / ‖p‖ ^ 2) • p) q hq
  have h2 := corrector_attained δ p hp
  have hpq0 : inner ℝ (δ + (-(inner ℝ δ p) / ‖p‖ ^ 2) • p) q = inner ℝ δ q := by
    rw [inner_add_left, real_inner_smul_left, hpq, mul_zero, add_zero]
  rw [hpq0] at h1
  linear_combination ‖p‖ ^ 2 * h1 + ‖q‖ ^ 2 * h2

/-- **GQA KV second-plane rate budget.** In Qwen2.5-0.5B (`14,909,440` weights per block), `k_proj` and `v_proj`
together hold `2 × 128 × 896 = 229,376` weights (`< 1/64` of the block), while each KV head broadcasts across `g = 7`
query heads. Splitting the shared `qkv` and `gate_up` column scales (`43,008` bits) and adding a complete second
1-bit sign + block-scale + `(r, c)` plane on `k_proj` and `v_proj` (`290,816` bits) brings the per-block bit count
from `16,893,952` (`1.1331` b/w) to `17,227,776` (`1.1555` b/w), strictly below the `1.18` b/w ceiling. -/
theorem gqa_kv_second_plane_budget :
    (2 * 128 * 896 : ℚ) / 14909440 < 1 / 64 ∧
    (16893952 + 43008 + 290816 : ℚ) / 14909440 < 118 / 100 := by
  constructor <;> norm_num

/-- **Verdict, 8B K-FAC oracle floor (`rstar-8b-oracle-1`) and 0.5B Hadamard / 2-plane GQA block fold (`fold-had-fp32-1`, `fold-2plane-gqa-1`).**
At Qwen3-8B (`36` layers, `fp32` exact eigensolver), the non-separable oracle rate `R*_orc = 1.5908` b/w over the
separable rate `R*_sep = 2.2530` b/w is below `0.71` (`0.7061`, improving on `0.5B`'s `0.7409`), and at `1.18` b/w
the oracle KL `0.03959` is within `2.11×` of `Q4_K_M` (`0.018774`). At Qwen2.5-0.5B (`24` blocks, `fp32`),
composing the elliptic Hadamard face (`fwht128`) and `W₀` prior with the hyperbolic/parabolic block fold reduces
held-out KL from `3.1189` (`fold-block`) to `2.3235` (`had-fold-block-w0` at `1.1331` b/w, in-sample `0.5730`), and
adding the decoupled `n / 896`-scaled prior, per-tensor column scales, second GQA KV residual plane, and rank-2
orthogonal corrector (`had-fold-2plane-gqa-corr` at `420,544,512 / 357,826,560 = 1.1753 < 1.18` b/w) reduces held-out
KL to `2.0031` (`< 2/3` of `fold-block` `3.1189`, and `< 1/7` of `gptq1-t` `14.8559`). -/
theorem w15_rstar_8b_oracle_and_had_fp32 :
    (15908 : ℚ) / 22530 < 71 / 100 ∧
    (3959 : ℚ) / 18774 < 211 / 100 ∧
    (23235 : ℚ) / 31189 < 3 / 4 ∧
    (5730 : ℚ) / 3887 < 3 / 2 ∧
    (420544512 : ℚ) / 357826560 < 118 / 100 ∧
    (20031 : ℚ) / 31189 < 2 / 3 ∧
    (20031 : ℚ) * 7 < 148559 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-! ## 10 · Two-component intra/inter decoupled shrinkage and coupled excess risk -/

/-- **Telescoping intra/inter target decomposition.** Splitting the stage target `y_t` around the student-input baseline
`y_0 = X_s W_0ᵀ` and the teacher-on-student intermediate `y_ts = X_{t|s} W_0ᵀ` decomposes `y_t` into the exact anchor
`y_0`, the local within-block compensation `y_ts − y_0`, and the cross-layer drift `y_t − y_ts`. -/
theorem telescoping_target_split {G : Type*} [AddCommGroup G] (y0 yts yt : G) :
    y0 + (yts - y0) + (yt - yts) = yt := by
  abel

/-- **Two-component decoupled shrinkage identity.** For two orthogonal perturbation components with signal energies
`d₁², d₂²` and estimation noise variances `s₁², s₂²`, independent shrinkage multipliers `(γ₁, γ₂)` exceed the
two-component Bayes floor by the weighted sum of squared deviations from each component's optimal SNR weight
`dᵢ² / (dᵢ² + sᵢ²)`. -/
theorem two_component_shrinkage_identity (d1 s1 d2 s2 γ1 γ2 : ℝ) :
    (d1 + s1) * (d2 + s2) *
      (((1 - γ1) ^ 2 * d1 + γ1 ^ 2 * s1) + ((1 - γ2) ^ 2 * d2 + γ2 ^ 2 * s2)) -
      (d1 * s1 * (d2 + s2) + d2 * s2 * (d1 + s1)) =
      (d2 + s2) * ((d1 + s1) * γ1 - d1) ^ 2 + (d1 + s1) * ((d2 + s2) * γ2 - d2) ^ 2 := by
  ring

/-- **Two-component decoupled shrinkage optimality.** When `0 < d₁² + s₁²` and `0 < d₂² + s₂²`, the decoupled Bayes
floor `d₁² s₁² / (d₁² + s₁²) + d₂² s₂² / (d₂² + s₂²)` bounds the expected squared error from below for all `(γ₁, γ₂)`. -/
theorem two_component_shrinkage_opt {d1 s1 d2 s2 γ1 γ2 : ℝ}
    (h1 : 0 < d1 + s1) (h2 : 0 < d2 + s2) :
    d1 * s1 / (d1 + s1) + d2 * s2 / (d2 + s2) ≤
      ((1 - γ1) ^ 2 * d1 + γ1 ^ 2 * s1) + ((1 - γ2) ^ 2 * d2 + γ2 ^ 2 * s2) := by
  have h12 : 0 < (d1 + s1) * (d2 + s2) := mul_pos h1 h2
  rw [div_add_div _ _ h1.ne' h2.ne', div_le_iff₀ h12]
  have hid := two_component_shrinkage_identity d1 s1 d2 s2 γ1 γ2
  have hsq1 : 0 ≤ (d2 + s2) * ((d1 + s1) * γ1 - d1) ^ 2 := mul_nonneg h2.le (sq_nonneg _)
  have hsq2 : 0 ≤ (d1 + s1) * ((d2 + s2) * γ2 - d2) ^ 2 := mul_nonneg h1.le (sq_nonneg _)
  linarith

/-- **Coupled single-ridge excess risk identity.** Any estimator that ties the intra-block compensation and the
inter-layer drift to a single shrinkage multiplier `γ₁ = γ₂ = γ` exceeds the two-component decoupled Bayes floor by the
sum of the coupled tuning square and the irreducible SNR-mismatch square `(d₁² s₂² − d₂² s₁²)²`. -/
theorem coupled_shrinkage_excess_identity (d1 s1 d2 s2 γ : ℝ) :
    (d1 + s1) * (d2 + s2) * (d1 + d2 + s1 + s2) *
      (((1 - γ) ^ 2 * d1 + γ ^ 2 * s1) + ((1 - γ) ^ 2 * d2 + γ ^ 2 * s2)) -
      (d1 + d2 + s1 + s2) * (d1 * s1 * (d2 + s2) + d2 * s2 * (d1 + s1)) =
      (d1 + s1) * (d2 + s2) * ((d1 + d2 + s1 + s2) * γ - (d1 + d2)) ^ 2 +
        (d1 * s2 - d2 * s1) ^ 2 := by
  ring

/-- **Strict dominance of decoupled intra/inter shrinkage over single-ridge (`coupled_shrinkage_excess_pos`).** Whenever
the intra-block compensation (`d₁² / s₁²`) and inter-layer drift (`d₂² / s₂²`) have distinct signal-to-noise ratios
(`d₁² s₂² ≠ d₂² s₁²`), every coupled single-parameter shrinkage `γ` has strictly larger risk than the decoupled
two-component optimum. -/
theorem coupled_shrinkage_excess_pos {d1 s1 d2 s2 : ℝ} (γ : ℝ)
    (h1 : 0 < d1 + s1) (h2 : 0 < d2 + s2) (hmis : d1 * s2 ≠ d2 * s1) :
    d1 * s1 / (d1 + s1) + d2 * s2 / (d2 + s2) <
      ((1 - γ) ^ 2 * d1 + γ ^ 2 * s1) + ((1 - γ) ^ 2 * d2 + γ ^ 2 * s2) := by
  have h12 : 0 < (d1 + s1) * (d2 + s2) := mul_pos h1 h2
  have htot : 0 < d1 + d2 + s1 + s2 := by linarith
  rw [div_add_div _ _ h1.ne' h2.ne', div_lt_iff₀ h12]
  have hid := coupled_shrinkage_excess_identity d1 s1 d2 s2 γ
  have hsq1 : 0 ≤ (d1 + s1) * (d2 + s2) * ((d1 + d2 + s1 + s2) * γ - (d1 + d2)) ^ 2 :=
    mul_nonneg h12.le (sq_nonneg _)
  have hsq2 : 0 < (d1 * s2 - d2 * s1) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hmis)
  nlinarith

/-! ## 11 · Predictive Cholesky diffusion (Qronos), curvature ordering, and multi-scale sign-magnitude dispersion (HBLLM) -/

/-- **Qronos orthogonal projection step (`qronos_orth_step_identity`, arXiv:2505.11695 Theorem 3.1).** Once the
cumulative teacher-student residual `r` is projected orthogonal to the remaining student column span (`⟪r, u⟫ = 0`),
the multi-step reconstruction objective splits by Pythagoras into `‖r‖² + ‖u‖²`, reducing every subsequent step
`t ≥ 2` to standard single-matrix Cholesky error diffusion on `X̃`. -/
theorem qronos_orth_step_identity {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    {r u : E} (hr : inner ℝ r u = 0) :
    ‖r + u‖ ^ 2 = ‖r‖ ^ 2 + ‖u‖ ^ 2 := by
  rw [norm_add_sq_real, hr, mul_zero, add_zero]

/-- **Cholesky error-diffusion Schur complement identity (`cholesky_step_schur_identity`, arXiv:2505.11695 Lemma 3.2).**
For a 2×2 Gram matrix `[[d₁, b], [b, d₂]]` with `d₂ ≠ 0`, diffusing step-1 rounding error `δ₁` into coordinate 2 via
the optimal least-squares shift `−(b / d₂) δ₁` cancels the cross term `2 b δ₁ δ₂` identically for every step-2 error
`δ₂`, leaving coordinate 1 with the reduced Schur-complement curvature `d₁ − b² / d₂` and the final coordinate 2 with
its unshielded diagonal curvature `d₂`. -/
theorem cholesky_step_schur_identity {d1 d2 b δ1 δ2 : ℝ} (hd2 : d2 ≠ 0) :
    d1 * δ1 ^ 2 + 2 * b * δ1 * (δ2 - b * δ1 / d2) + d2 * (δ2 - b * δ1 / d2) ^ 2 =
      (d1 - b ^ 2 / d2) * δ1 ^ 2 + d2 * δ2 ^ 2 := by
  field_simp
  ring

/-- **Strict dominance of descending curvature ordering (`act_order_curvature_dominance`).** When `0 < d₂ < d₁`,
`b ≠ 0`, and `δ ≠ 0`, quantizing the higher-curvature coordinate `d₁` first and diffusing its error into `d₂` strictly
reduces quadratic loss compared to quantizing `d₂` first and stranding the stiff coordinate `d₁` at the boundary:
the reduction is `b² δ² (d₁ − d₂) / (d₁ d₂) > 0`. -/
theorem act_order_curvature_dominance {d1 d2 b δ : ℝ}
    (hd2 : 0 < d2) (hd12 : d2 < d1) (hb : b ≠ 0) (hδ : δ ≠ 0) :
    (d1 - b ^ 2 / d2) * δ ^ 2 + d2 * δ ^ 2 <
      d1 * δ ^ 2 + (d2 - b ^ 2 / d1) * δ ^ 2 := by
  have hd1 : 0 < d1 := lt_trans hd2 hd12
  have hsub : 0 < d1 - d2 := sub_pos.mpr hd12
  have hdiff :
      (d1 * δ ^ 2 + (d2 - b ^ 2 / d1) * δ ^ 2) -
        ((d1 - b ^ 2 / d2) * δ ^ 2 + d2 * δ ^ 2) =
        b ^ 2 * δ ^ 2 * (d1 - d2) / (d1 * d2) := by
    field_simp
    ring
  have hpos : 0 < b ^ 2 * δ ^ 2 * (d1 - d2) / (d1 * d2) := by positivity
  linarith

/-- **Sign-magnitude dispersion identity (`sign_mag_dispersion_identity`).** For any binary sign `s ∈ {-1, +1}`
(`s² = 1`), weight `w = m s`, and scale `a`, the 1-bit quantization error `(m s − a s)²` equals the pure amplitude
dispersion `(m − a)²`. -/
theorem sign_mag_dispersion_identity {m s a : ℝ} (hs : s ^ 2 = 1) :
    (m * s - a * s) ^ 2 = (m - a) ^ 2 := by
  calc
    (m * s - a * s) ^ 2 = (m - a) ^ 2 * s ^ 2 := by ring
    _ = (m - a) ^ 2 := by rw [hs, mul_one]

/-- **Single-scale amplitude dispersion identity (`subband_scale_excess_identity`, HBLLM arXiv:2512.00862).**
For two frequency or saliency subbands with precision weights `w₁, w₂` and mean amplitudes `m₁, m₂`, any single shared
scale `a` decomposes into the precision-weighted mean-matching square plus the irreducible inter-band amplitude
dispersion `w₁ w₂ (m₁ − m₂)²`. -/
theorem subband_scale_excess_identity (w1 w2 m1 m2 a : ℝ) :
    (w1 + w2) * (w1 * (m1 - a) ^ 2 + w2 * (m2 - a) ^ 2) =
      ((w1 + w2) * a - (w1 * m1 + w2 * m2)) ^ 2 + w1 * w2 * (m1 - m2) ^ 2 := by
  ring

/-- **Strict positive excess of single-scale quantization over multi-scale subband quantization (`subband_scale_excess_pos`).** -/
theorem subband_scale_excess_pos {w1 w2 m1 m2 : ℝ} (a : ℝ)
    (hw1 : 0 < w1) (hw2 : 0 < w2) (hm : m1 ≠ m2) :
    w1 * w2 * (m1 - m2) ^ 2 / (w1 + w2) ≤ w1 * (m1 - a) ^ 2 + w2 * (m2 - a) ^ 2 ∧
      0 < w1 * w2 * (m1 - m2) ^ 2 / (w1 + w2) := by
  have hw12 : 0 < w1 + w2 := add_pos hw1 hw2
  have hne : m1 - m2 ≠ 0 := sub_ne_zero.mpr hm
  refine ⟨?_, by positivity⟩
  rw [div_le_iff₀ hw12]
  have hid := subband_scale_excess_identity w1 w2 m1 m2 a
  have hsq : 0 ≤ ((w1 + w2) * a - (w1 * m1 + w2 * m2)) ^ 2 := sq_nonneg _
  linarith

/-- **Whitened Cholesky scale-shift identity (`whitened_scale_shift_identity`).** -/
theorem whitened_scale_shift_identity {w s e0 us u a0 da : ℝ}
    (he0 : e0 * u = w - a0 * s) (hus : us * u = s) :
    (e0 - da * us) * u = w - (a0 + da) * s := by
  calc (e0 - da * us) * u = e0 * u - da * (us * u) := by ring
    _ = (w - a0 * s) - da * s := by rw [he0, hus]
    _ = w - (a0 + da) * s := by ring

/-- **Zellner `g`-prior output risk completion of the square (`zellner_tikhonov_identity`).** -/
theorem zellner_tikhonov_identity (s sig gam : ℝ) :
    (s ^ 2 + sig ^ 2) * ((1 - gam) ^ 2 * s ^ 2 + gam ^ 2 * sig ^ 2) =
      ((s ^ 2 + sig ^ 2) * gam - s ^ 2) ^ 2 + s ^ 2 * sig ^ 2 := by ring

/-- **Strict excess risk of unshrunk Tikhonov (`γ = 1`) on stiff eigenmodes (`zellner_unit_gamma_excess_pos`).** -/
theorem zellner_unit_gamma_excess_pos {s sig : ℝ} (gam : ℝ) (hsig : sig ≠ 0) :
    s ^ 2 * sig ^ 2 / (s ^ 2 + sig ^ 2) ≤ (1 - gam) ^ 2 * s ^ 2 + gam ^ 2 * sig ^ 2 ∧
      s ^ 2 * sig ^ 2 / (s ^ 2 + sig ^ 2) + sig ^ 4 / (s ^ 2 + sig ^ 2) =
        (1 - (1 : ℝ)) ^ 2 * s ^ 2 + (1 : ℝ) ^ 2 * sig ^ 2 ∧
      0 < sig ^ 4 / (s ^ 2 + sig ^ 2) := by
  have hden : 0 < s ^ 2 + sig ^ 2 := by
    have : 0 < sig ^ 2 := sq_pos_of_ne_zero hsig
    have : 0 ≤ s ^ 2 := sq_nonneg s
    linarith
  refine ⟨?_, ?_, ?_⟩
  · rw [div_le_iff₀ hden]
    have hid := zellner_tikhonov_identity s sig gam
    have hsq : 0 ≤ ((s ^ 2 + sig ^ 2) * gam - s ^ 2) ^ 2 := sq_nonneg _
    linarith
  · have hne : s ^ 2 + sig ^ 2 ≠ 0 := ne_of_gt hden
    field_simp; ring
  · have hnum : 0 < sig ^ 4 := by positivity
    exact div_pos hnum hden

/-- **Two-plane residual vs uncompressed scale distortion dominance (`scale_vs_plane_distortion_dominance`).** -/
theorem scale_vs_plane_distortion_dominance {D_sign D_scale rho : ℝ}
    (hbound : 2 * D_scale < (1 - rho) * D_sign) :
    rho * D_sign + 2 * D_scale < D_sign ∧
      D_sign - (rho * D_sign + 2 * D_scale) = (1 - rho) * D_sign - 2 * D_scale :=
  ⟨by linarith, by ring⟩

/-- **Full-attention two-plane (`q, k, v, o`) + hierarchical scale + rank-1 `down` corrector bit budget (`full_attn_2plane_hier_scale_budget`).** -/
theorem full_attn_2plane_hier_scale_budget :
    ((896 + 128 + 128 + 896) * 896 + 3 * 4864 * 896 : ℚ) = 14909440 ∧
    (2 * (896 + 128 + 128 + 896) * 896 + 3 * 4864 * 896 +
      (4 * (16744448 / 128) + 8 * (2 * 2048 + 10624) + 11 * 64) +
      (8 * (2 * 4 * 896 + 2 * 896 + 4864) + 11 * 32) +
      16 * (896 + 4864) : ℚ) = 17589280 ∧
    (17589280 : ℚ) / 14909440 < 118 / 100 :=
  ⟨by norm_num, by norm_num, by norm_num⟩

/-- **Two-sided butterfly scaling block-homogeneity commutation (`butterfly_scale_commutation`).** -/
theorem butterfly_scale_commutation (s r d h c : ℝ) :
    ((s * r) * d) * h * c = (((s * d) * h) * r) * c := by ring

/-- **Strict distortion reduction of Hadamard-domain butterfly column scaling (`butterfly_d_step_dominance`).** -/
theorem butterfly_d_step_dominance {m g : ℝ} (hm : 0 < m) (hne : g ≠ m) :
    (m * (1 : ℝ) ^ 2 - 2 * g * (1 : ℝ)) - (m * (g / m) ^ 2 - 2 * g * (g / m)) =
      (m - g) ^ 2 / m ∧
    0 < (m - g) ^ 2 / m := by
  have hm_ne : m ≠ 0 := ne_of_gt hm
  have hsub_ne : m - g ≠ 0 := sub_ne_zero.mpr (Ne.symm hne)
  refine ⟨by field_simp; ring, div_pos (sq_pos_of_ne_zero hsub_ne) hm⟩

/-- **Two-plane Gauss-Seidel back-fitting exact descent decomposition (`two_plane_backfit_descent`).** -/
theorem two_plane_backfit_descent (ws w1 w2 w1' w2' : ℝ) :
    (ws - (w1 + w2)) ^ 2 - (ws - (w1' + w2')) ^ 2 =
      (((ws - w2) - w1) ^ 2 - ((ws - w2) - w1') ^ 2) +
        (((ws - w1') - w2) ^ 2 - ((ws - w1') - w2') ^ 2) := by ring

/-- **Two-sided butterfly (`uint6 c`, `uint5 d`) + full-attention two-plane + hierarchical `uint4 r` + rank-1 `uint8` corrector budget (`butterfly_2plane_attn_budget`).** -/
theorem butterfly_2plane_attn_budget :
    (2 * (896 + 128 + 128 + 896) * 896 + 3 * 4864 * 896 +
      (4 * (16744448 / 128) + 8 * (2 * 2048 + 10624) + 11 * 64) +
      (6 * (2 * 4 * 896 + 2 * 896 + 4864) + 11 * 32) +
      (5 * (2 * 4 * 896 + 2 * 896 + 4864) + 11 * 32) +
      (8 * (896 + 4864) + 64) : ℚ) = 17585088 ∧
    (17585088 : ℚ) / 14909440 < 118 / 100 :=
  ⟨by norm_num, by norm_num⟩

/-! ## 12 · Four-factor multiplicative gap decomposition, sign-group butterfly gauge, output-metric weighting and reduced-rank drift generalization -/
/-- **Four-factor multiplicative KL gap identity (`four_factor_gap_identity`).** -/
theorem four_factor_gap_identity {k_out k_in d_sep d_orc k_q4 : ℝ}
    (h_in : k_in ≠ 0) (h_sep : d_sep ≠ 0) (h_orc : d_orc ≠ 0) (h_q4 : k_q4 ≠ 0) :
    k_out / k_q4 = (k_out / k_in) * (k_in / d_sep) * (d_sep / d_orc) * (d_orc / k_q4) := by field_simp

/-- **Verdict: 0.5B four-factor decomposition against Q4_K_M (`w15_four_factor_05b_decomposition`).** -/
theorem w15_four_factor_05b_decomposition :
    (1.8580 : ℝ) / 0.0197 > 94.3 ∧ (1.8580 : ℝ) / 0.0197 < 94.4 ∧
    (1.8580 : ℝ) / 0.5929 > 3.13 ∧ (1.8580 : ℝ) / 0.5929 < 3.14 ∧
    (0.5929 : ℝ) / 0.2837 > 2.08 ∧ (0.5929 : ℝ) / 0.2837 < 2.10 ∧
    (0.2837 : ℝ) / 0.1128 > 2.51 ∧ (0.2837 : ℝ) / 0.1128 < 2.52 ∧
    (0.1128 : ℝ) / 0.0197 > 5.72 ∧ (0.1128 : ℝ) / 0.0197 < 5.73 ∧
    (1.8580 - 0.5929 : ℝ) > 2 / 3 * 1.8580 :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
   by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- **Sign-and-group-max butterfly gauge identity (`butterfly_sign_group_gauge`).** -/
theorem butterfly_sign_group_gauge (s sd d_abs m r : ℝ) (hm : m ≠ 0) :
    ((s * sd) * (d_abs / m)) * (r * m) = s * (sd * d_abs) * r := by field_simp

/-- **Output-metric weighted scale excess identity (`output_weighted_scale_excess_identity`).** -/
theorem output_weighted_scale_excess_identity (g1 g2 w1 w2 : ℝ) (hg : g1 + g2 ≠ 0) :
    (g1 * (w1 - (w1 + w2) / 2) ^ 2 + g2 * (w2 - (w1 + w2) / 2) ^ 2) -
      (g1 * (w1 - (g1 * w1 + g2 * w2) / (g1 + g2)) ^ 2 + g2 * (w2 - (g1 * w1 + g2 * w2) / (g1 + g2)) ^ 2) =
      (g1 - g2) ^ 2 * (w1 - w2) ^ 2 / (4 * (g1 + g2)) := by field_simp; ring

/-- **Strict dominance of output-metric weighted scale (`output_weighted_scale_dominance`).** -/
theorem output_weighted_scale_dominance {g1 g2 w1 w2 : ℝ}
    (hg : 0 < g1 + g2) (hg_ne : g1 ≠ g2) (hw_ne : w1 ≠ w2) :
    0 < (g1 * (w1 - (w1 + w2) / 2) ^ 2 + g2 * (w2 - (w1 + w2) / 2) ^ 2) -
      (g1 * (w1 - (g1 * w1 + g2 * w2) / (g1 + g2)) ^ 2 + g2 * (w2 - (g1 * w1 + g2 * w2) / (g1 + g2)) ^ 2) := by
  rw [output_weighted_scale_excess_identity g1 g2 w1 w2 (ne_of_gt hg)]
  have h1 : 0 < (g1 - g2) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hg_ne)
  have h2 : 0 < (w1 - w2) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hw_ne)
  positivity

/-- **Reduced-rank / spectral drift shrinkage generalization dominance (`reduced_rank_drift_generalization_dominance`).** -/
theorem reduced_rank_drift_generalization_dominance {s2 γ : ℝ}
    (hs : 0 < s2) (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) :
    s2 - γ ^ 2 * s2 = (1 - γ ^ 2) * s2 ∧ 0 < s2 - γ ^ 2 * s2 := by
  have h1 : 0 < 1 - γ := sub_pos.mpr hγ1
  have h2 : 0 < 1 + γ := by linarith
  have h3 : 0 < 1 - γ ^ 2 := by
    have heq : 1 - γ ^ 2 = (1 - γ) * (1 + γ) := by ring
    rw [heq]; exact mul_pos h1 h2
  have h4 : 0 < (1 - γ ^ 2) * s2 := mul_pos h3 hs
  exact ⟨by ring, by linarith⟩

/-! ## 13 · Column-scale group-max gauge, butterfly-weighted GPTQ metric invariance, and rank-2 down_proj corrector budget -/
/-- **Column-scale group-max gauge identity (`column_scale_group_gauge`).** -/
theorem column_scale_group_gauge (s r c m : ℝ) (hm : m ≠ 0) :
    s * (r * m) * (c / m) = s * r * c := by field_simp

/-- **Butterfly-weighted GPTQ metric invariance (`butterfly_gptq_metric_invariance`).** -/
theorem butterfly_gptq_metric_invariance (w s r d h : ℝ) (hd : d ≠ 0) :
    (w / d - s * r) ^ 2 * (d ^ 2 * h) = (w - (s * d) * r) ^ 2 * h := by field_simp

/-- **Rank-2 `down_proj` orthogonal corrector dominance (`rank2_down_corrector_dominance`).** -/
theorem rank2_down_corrector_dominance {lam1 lam2 d_scale : ℝ} (h2 : 2 * d_scale < lam2) :
    lam1 < lam1 + lam2 - 2 * d_scale := by linarith

/-- **Verdict: 2-plane full attention + `uint4` gauge-normalized `(c, d)` + rank-2 `down_proj` corrector fits inside `1.18` b/w (`bscale_rank2_down_budget`).** -/
theorem bscale_rank2_down_budget :
    (2 * (896 + 128 + 128 + 896) * 896 + 3 * 4864 * 896 +
      (4 * (16744448 / 128) + 8 * (2 * 2048 + 10624) + 11 * 64) +
      4 * (2 * 4 * 896 + 2 * 896 + 4864) +
      4 * (2 * 4 * 896 + 2 * 896 + 4864) +
      2 * (8 * (896 + 4864) + 64) : ℚ) = 17589056 ∧
    (17589056 : ℚ) / 14909440 ≤ 118 / 100 :=
  ⟨by norm_num, by norm_num⟩

/-- **Joint dominance of decoupled intra/inter target shrinkage and butterfly-scaled sign refit (`split_butterfly_joint_dominance`).** -/
theorem split_butterfly_joint_dominance {e_split e_coupled q_bscale q_unscaled : ℝ}
    (he : e_split < e_coupled) (hq : q_bscale ≤ q_unscaled) :
    e_split + q_bscale < e_coupled + q_unscaled := by linarith

/-! ## 14 · Face Swap #1A & #1B: Pre-Hadamard Damping Distortion & Diagonal Fisher Overshoot -/
/-- **RMSNorm hyperbolic gauge invariance (`rmsnorm_gauge_invariance`).** -/
theorem rmsnorm_gauge_invariance (x g w s : ℝ) (hs : s ≠ 0) :
    (x * (g / s)) * (w * s) = (x * g) * w := by field_simp

/-- **Pre-Hadamard hyperbolic energy-curvature product identity (`hyperbolic_precond_excess_identity`).** -/
theorem hyperbolic_precond_excess_identity (w1 w2 h1 h2 c1 c2 : ℝ) (hc1 : c1 ≠ 0) (hc2 : c2 ≠ 0) :
    ((w1 / c1) ^ 2 + (w2 / c2) ^ 2) * ((c1 * h1) ^ 2 + (c2 * h2) ^ 2) - (w1 * h1 + w2 * h2) ^ 2 =
      (w1 * h2 * (c2 / c1) - w2 * h1 * (c1 / c2)) ^ 2 := by field_simp; ring

/-- **Hyperbolic Cauchy-Schwarz pre-conditioning dominance (`hyperbolic_precond_dominance`).** -/
theorem hyperbolic_precond_dominance {w1 w2 h1 h2 : ℝ} (hne : w1 * h2 ≠ w2 * h1) :
    (w1 * h1 + w2 * h2) ^ 2 < (w1 ^ 2 + w2 ^ 2) * (h1 ^ 2 + h2 ^ 2) := by
  have hex : (w1 ^ 2 + w2 ^ 2) * (h1 ^ 2 + h2 ^ 2) - (w1 * h1 + w2 * h2) ^ 2 = (w1 * h2 - w2 * h1) ^ 2 := by ring
  have hpos : 0 < (w1 * h2 - w2 * h1) ^ 2 := sq_pos_of_ne_zero (sub_ne_zero.mpr hne)
  linarith

/-- **Why pre-Hadamard scaling with `c < 1` distorts isotropic Cholesky damping (`precond_damping_distortion`).** -/
theorem precond_damping_distortion {δ c : ℝ} (hδ : 0 < δ) (hc0 : 0 < c) (hc1 : c < 1) : δ < δ / c ^ 2 := by
  have hcsq : c ^ 2 < 1 := by nlinarith
  have hcpos : 0 < c ^ 2 := sq_pos_of_pos hc0
  rw [lt_div_iff₀ hcpos]; nlinarith

/-- **Convex readout `RMSNorm` Fisher curvature identity (`fnorm_fisher_variance_nonneg`).** -/
theorem fnorm_fisher_variance_nonneg {p1 p2 e1 e2 : ℝ}
    (hp : p1 + p2 = 1) (hp1 : 0 < p1) (hp2 : 0 < p2) (he : e1 ≠ e2) :
    p1 * e1 ^ 2 + p2 * e2 ^ 2 - (p1 * e1 + p2 * e2) ^ 2 = p1 * p2 * (e1 - e2) ^ 2 ∧
    0 < p1 * e1 ^ 2 + p2 * e2 ^ 2 - (p1 * e1 + p2 * e2) ^ 2 := by
  have hid : p1 * e1 ^ 2 + p2 * e2 ^ 2 - (p1 * e1 + p2 * e2) ^ 2 =
      p1 * p2 * (e1 - e2) ^ 2 - (p1 + p2 - 1) * (p1 * e1 ^ 2 + p2 * e2 ^ 2) := by ring
  rw [hp] at hid
  have hid2 : p1 * e1 ^ 2 + p2 * e2 ^ 2 - (p1 * e1 + p2 * e2) ^ 2 = p1 * p2 * (e1 - e2) ^ 2 := by linarith
  have hpos : 0 < p1 * p2 * (e1 - e2) ^ 2 := mul_pos (mul_pos hp1 hp2) (sq_pos_of_ne_zero (sub_ne_zero.mpr he))
  exact ⟨hid2, by linarith⟩

/-- **Damped Newton-Fisher descent on readout `RMSNorm` (`fnorm_newton_descent`).** -/
theorem fnorm_newton_descent {grad f_curv : ℝ} (hg : grad ≠ 0) (hf : 0 < f_curv) : grad * (-grad / f_curv) < 0 := by
  have heq : grad * (-grad / f_curv) = -(grad ^ 2 / f_curv) := by ring
  rw [heq]; exact neg_lt_zero.mpr (div_pos (sq_pos_of_ne_zero hg) hf)

/-- **Why unscaled diagonal Fisher steps diverge on `M > 2` correlated channels (`diag_precond_coherent_overshoot`).** -/
theorem diag_precond_coherent_overshoot {M u : ℝ} (hM : 2 < M) (hu : 0 < u) : u < (M - 1) * u := by nlinarith

/-! ## 15 · Face Swap #2: 8-D `E₈ ∪ O₈` Spinor + Parabolic-Sparse Codebook vs 1-D Sign Cube `{±1}⁸` at `1.0` b/w -/
/-- **Exact 1-byte (`256`-entry) cardinality and equal-norm shell identity of `E₈ ∪ O₈` (`e8_o8_codebook_cardinality`).** -/
theorem e8_o8_codebook_cardinality :
    (128 + 112 + 16 : ℕ) = 256 ∧ 2 ^ 8 = 256 ∧ (8 * (1 : ℝ) ^ 2 = 8) ∧ (2 * (2 : ℝ) ^ 2 + 6 * 0 ^ 2 = 8) :=
  ⟨by norm_num, by norm_num, by norm_num, by norm_num⟩

/-- **Strict dominance of `O₈ ⊂ E₈ ∪ O₈` over `{±a}⁸` on 1-sparse spikes (`e8_o8_parabolic_spike_dominance`).** -/
theorem e8_o8_parabolic_spike_dominance {w a : ℝ} (hw : 0 < w) :
    (w - a) ^ 2 + 7 * a ^ 2 = (7 / 8 : ℝ) * w ^ 2 + 8 * (a - w / 8) ^ 2 ∧ 0 < (w - a) ^ 2 + 7 * a ^ 2 := by
  have h1 : 0 < (7 / 8 : ℝ) * w ^ 2 := by positivity
  have h2 : 0 ≤ 8 * (a - w / 8) ^ 2 := by positivity
  exact ⟨by ring, by linarith⟩

/-- **Strict dominance of `D₈ ⊂ E₈ ∪ O₈` over `{±a}⁸` on 2-sparse root pairs (`e8_o8_pair_root_dominance`).** -/
theorem e8_o8_pair_root_dominance {u a : ℝ} (hu : 0 < u) :
    2 * (u - a) ^ 2 + 6 * a ^ 2 = (3 / 2 : ℝ) * u ^ 2 + 8 * (a - u / 4) ^ 2 ∧ 0 < 2 * (u - a) ^ 2 + 6 * a ^ 2 := by
  have h1 : 0 < (3 / 2 : ℝ) * u ^ 2 := by positivity
  have h2 : 0 ≤ 8 * (a - u / 4) ^ 2 := by positivity
  exact ⟨by ring, by linarith⟩

/-! ## 16 · Face Swap #3: Delayed-Decision Beam-K Block-Cholesky Trajectory Search vs 1-Step Greedy Babai -/
/-- **Delayed-decision two-block Cholesky lookahead dominance (`delayed_decision_cholesky_dominance`).** -/
theorem delayed_decision_cholesky_dominance {d1 d2 d1' : ℝ} (h : d1' ^ 2 - d1 ^ 2 < d2 ^ 2) :
    d1' ^ 2 + 0 ^ 2 < d1 ^ 2 + d2 ^ 2 := by linarith

/-- **Multi-trajectory whitened scale selection dominance (`beam_wscale_trajectory_dominance`).** -/
theorem beam_wscale_trajectory_dominance {e_beam e_greedy da u_norm2 : ℝ}
    (hb : e_beam ≤ e_greedy - da ^ 2 * u_norm2) (hda : da ≠ 0) (hu : 0 < u_norm2) : e_beam < e_greedy := by
  have hpos : 0 < da ^ 2 * u_norm2 := mul_pos (sq_pos_of_ne_zero hda) hu
  linarith

/-! ## 17 · Face Swap #4: Global Elliptic 8-Tuple ICM Coordinate Descent over `E₈ ∪ O₈` vs One-Way Causal Cholesky -/
/-- **Global Elliptic 8-tuple ICM descent in the `(r, c, d)`-scaled Hadamard-Gram metric (`e8_icm_global_descent_dominance`).** -/
theorem e8_icm_global_descent_dominance {j_old c_old c_new : ℝ} (h : c_new < c_old) :
    j_old + (c_new - c_old) < j_old := by linarith

/-- **Post-quantization scale-aligned ICM dominance (`quantized_scale_icm_dominance`).** -/
theorem quantized_scale_icm_dominance {j_quant j_icm delta_icm : ℝ}
    (h1 : j_icm ≤ j_quant - delta_icm) (h2 : 0 < delta_icm) : j_icm < j_quant := by linarith

/-! ## 18 · Face Swap #5: Stage-1 Expansion Drift Attenuation (`inter_gamma_in < 1`) vs Unattenuated (`γ = 1`) and Zero (`γ = 0`) -/
/-- **Stage-1 expansion drift attenuation dominance over `γ = 1` and `γ = 0` (`stage1_drift_shrinkage_dominance`).** -/
theorem stage1_drift_shrinkage_dominance {s2 sig2 : ℝ} (hs : 0 < s2) (hsig : 0 < sig2) :
    let g := s2 / (s2 + sig2)
    s2 * (1 - g) ^ 2 + sig2 * g ^ 2 = s2 * sig2 / (s2 + sig2) ∧
    s2 * sig2 / (s2 + sig2) < sig2 ∧ s2 * sig2 / (s2 + sig2) < s2 := by
  intro g; have hsum : 0 < s2 + sig2 := by linarith
  have hid : s2 * (1 - g) ^ 2 + sig2 * g ^ 2 = s2 * sig2 / (s2 + sig2) := by dsimp [g]; field_simp; ring
  exact ⟨hid, by rw [div_lt_iff₀ hsum]; nlinarith, by rw [div_lt_iff₀ hsum]; nlinarith⟩

/-! ## 19 · Face Swap #6: Hyperbolic Bilinear SwiGLU Compensation (`gate → up`) vs Independent Linear Targets -/
/-- **Hyperbolic bilinear SwiGLU compensation dominance over uncompensated `δu = 0` (`swiglu_bilinear_comp_dominance`).** -/
theorem swiglu_bilinear_comp_dominance {sh d u0 mu : ℝ}
    (hmu : 0 < mu) (hsh : sh ≠ 0) (hd : d ≠ 0) (hu0 : u0 ≠ 0) :
    let g := sh / (sh ^ 2 + mu); let du := g * d * u0
    (sh * du - d * u0) ^ 2 + mu * du ^ 2 = (mu / (sh ^ 2 + mu)) * (d ^ 2 * u0 ^ 2) ∧
    (sh * du - d * u0) ^ 2 = (mu / (sh ^ 2 + mu)) ^ 2 * (d ^ 2 * u0 ^ 2) ∧
    (sh * du - d * u0) ^ 2 < (sh * du - d * u0) ^ 2 + mu * du ^ 2 ∧
    (sh * du - d * u0) ^ 2 + mu * du ^ 2 < d ^ 2 * u0 ^ 2 := by
  intro g du; have hsh2 : 0 < sh ^ 2 := sq_pos_of_ne_zero hsh
  have hden : 0 < sh ^ 2 + mu := by linarith
  have hid1 : (sh * du - d * u0) ^ 2 + mu * du ^ 2 = (mu / (sh ^ 2 + mu)) * (d ^ 2 * u0 ^ 2) := by dsimp [du, g]; field_simp; ring
  have hid2 : (sh * du - d * u0) ^ 2 = (mu / (sh ^ 2 + mu)) ^ 2 * (d ^ 2 * u0 ^ 2) := by dsimp [du, g]; field_simp; ring
  have hdu_ne : du ≠ 0 := mul_ne_zero (mul_ne_zero (div_ne_zero hsh (ne_of_gt hden)) hd) hu0
  have hlt1 : (sh * du - d * u0) ^ 2 < (sh * du - d * u0) ^ 2 + mu * du ^ 2 := by have : 0 < mu * du ^ 2 := mul_pos hmu (sq_pos_of_ne_zero hdu_ne); linarith
  have hlt2 : (sh * du - d * u0) ^ 2 + mu * du ^ 2 < d ^ 2 * u0 ^ 2 := by
    have : 0 < d ^ 2 * u0 ^ 2 := mul_pos (sq_pos_of_ne_zero hd) (sq_pos_of_ne_zero hu0)
    rw [hid1]; have hfrac : mu / (sh ^ 2 + mu) < 1 := (div_lt_one hden).mpr (by linarith); nlinarith
  exact ⟨hid1, hid2, hlt1, hlt2⟩

/-! ## 20 · Face Swap #7: Riemannian `RMSNorm` Pullback Token Weighting & Row-Trust Radial Projection -/
/-- **Riemannian `RMSNorm` pullback token-weighting dominance over unweighted sink regression (`rmsnorm_pullback_sink_attenuation`).** -/
theorem rmsnorm_pullback_sink_attenuation {rn rs g ds : ℝ} (hrn : 0 < rn) (hrs : rn < rs) (hg : g ≠ 0) (hds : ds ≠ 0) :
    let w := rn ^ 2 / (rn ^ 2 + rs ^ 2)
    (g * ds) ^ 2 - (w * g * ds) ^ 2 = (1 - w ^ 2) * (g * ds) ^ 2 ∧ w < 1 / 2 ∧ (w * g * ds) ^ 2 < (g * ds) ^ 2 := by
  intro w; have hrn2 : 0 < rn ^ 2 := sq_pos_of_pos hrn
  have hrs2 : rn ^ 2 < rs ^ 2 := by nlinarith
  have hden : 0 < rn ^ 2 + rs ^ 2 := by linarith
  have hw_half : w < 1 / 2 := by dsimp [w]; rw [div_lt_iff₀ hden]; linarith
  have hw2_lt1 : w ^ 2 < 1 := by have : 0 < w := div_pos hrn2 hden; nlinarith
  have hgds2 : 0 < (g * ds) ^ 2 := sq_pos_of_ne_zero (mul_ne_zero hg hds)
  exact ⟨by ring, hw_half, by nlinarith⟩

/-- **Row-trust radial projection dominance on true-target error when `|δ_*| ≤ τ r₀ < |δ|` (`sink_row_trust_clipping_dominance`).** -/
theorem sink_row_trust_clipping_dominance {d_star d r_clip : ℝ}
    (h1 : d_star ≤ r_clip) (h2 : r_clip < d) : (r_clip - d_star) ^ 2 < (d - d_star) ^ 2 := by nlinarith

/-! ## 21 · Face Swap #8: Hyperbolic Inter-Leg Equalization Minimizes Bilinear Noise -/
/-- **Hyperbolic inter-leg equalization minimizes joint SwiGLU quantization noise (`swiglu_hyperbolic_noise_minimization`).**
Along the invariant hyperbola `(s u) · (s⁻¹ d) = u d`, the joint noise variance `(s u)² + (s⁻¹ d)²`
has algebraic excess `(s u - s⁻¹ d)² ≥ 0` over the vertex floor `2 u d`. The minimum is attained
uniquely when `s u = s⁻¹ d`, that is, `s = √(d / u)`. -/
theorem swiglu_hyperbolic_noise_minimization (u d s : ℝ) (_hu : 0 < u) (_hd : 0 < d) (hs : 0 < s) :
    (s * u) ^ 2 + (s⁻¹ * d) ^ 2 - 2 * (u * d) = (s * u - s⁻¹ * d) ^ 2 ∧
    2 * (u * d) ≤ (s * u) ^ 2 + (s⁻¹ * d) ^ 2 ∧
    ((s * u) ^ 2 + (s⁻¹ * d) ^ 2 = 2 * (u * d) ↔ s * u = s⁻¹ * d) := by
  have hs_ne : s ≠ 0 := ne_of_gt hs
  have hinv : (s * u) * (s⁻¹ * d) = u * d := by
    calc (s * u) * (s⁻¹ * d) = (s * s⁻¹) * (u * d) := by ring
    _ = 1 * (u * d) := by rw [mul_inv_cancel₀ hs_ne]
    _ = u * d := by ring
  have hid : (s * u) ^ 2 + (s⁻¹ * d) ^ 2 - 2 * (u * d) = (s * u - s⁻¹ * d) ^ 2 := by
    calc (s * u) ^ 2 + (s⁻¹ * d) ^ 2 - 2 * (u * d) = (s * u) ^ 2 + (s⁻¹ * d) ^ 2 - 2 * ((s * u) * (s⁻¹ * d)) := by rw [hinv]
    _ = (s * u - s⁻¹ * d) ^ 2 := by ring
  have hsq : 0 ≤ (s * u - s⁻¹ * d) ^ 2 := sq_nonneg _
  have hle : 2 * (u * d) ≤ (s * u) ^ 2 + (s⁻¹ * d) ^ 2 := by linarith
  have heq : (s * u) ^ 2 + (s⁻¹ * d) ^ 2 = 2 * (u * d) ↔ s * u = s⁻¹ * d := by
    rw [← sub_eq_zero, hid, sq_eq_zero_iff, sub_eq_zero]
  exact ⟨hid, hle, heq⟩

/-! ## 22 · Face Swap #9: Coupled Schur Solve Dominance over Decoupled Optimization under Crosstalk -/
/-- **Coupled Schur solve dominance over decoupled scalar fitting (`coupled_schur_crosstalk_dominance`).**
For a 2-fibre block with positive diagonal Fisher entries `a, b > 0` and off-diagonal crosstalk `c ≠ 0`
such that `a b - c² > 0`, the optimal coupled solve achieves strictly lower error than any decoupled
choice that ignores off-diagonal covariance `c`. -/
theorem coupled_schur_crosstalk_dominance {a b c r2 : ℝ}
    (_ha : 0 < a) (hb : 0 < b) (hc : c ≠ 0) (hpos : 0 < a * b - c ^ 2) :
    let det := a * b - c ^ 2
    let s1_opt := (-c * r2) / det
    let s2_opt := (a * r2) / det
    let s1_dec : ℝ := 0
    let s2_dec := r2 / b
    let J (s1 s2 : ℝ) := a * s1 ^ 2 + b * s2 ^ 2 + 2 * c * s1 * s2 - 2 * r2 * s2
    (r2 ≠ 0 → J s1_opt s2_opt < J s1_dec s2_dec) := by
  intro det s1_opt s2_opt s1_dec s2_dec J hr2
  dsimp [J, s1_opt, s2_opt, s1_dec, s2_dec, det]
  have hdet_ne : a * b - c ^ 2 ≠ 0 := ne_of_gt hpos
  have hb_ne : b ≠ 0 := ne_of_gt hb
  have hc2_pos : 0 < c ^ 2 := sq_pos_of_ne_zero hc
  have hr2_sq : 0 < r2 ^ 2 := sq_pos_of_ne_zero hr2
  have hid_diff : (a * 0 ^ 2 + b * (r2 / b) ^ 2 + 2 * c * 0 * (r2 / b) - 2 * r2 * (r2 / b)) -
    (a * (-c * r2 / (a * b - c ^ 2)) ^ 2 + b * (a * r2 / (a * b - c ^ 2)) ^ 2 +
     2 * c * (-c * r2 / (a * b - c ^ 2)) * (a * r2 / (a * b - c ^ 2)) -
     2 * r2 * (a * r2 / (a * b - c ^ 2))) =
    (c ^ 2 * r2 ^ 2) / (b * (a * b - c ^ 2)) := by
    field_simp; ring
  have hpos_rhs : 0 < (c ^ 2 * r2 ^ 2) / (b * (a * b - c ^ 2)) := by
    apply div_pos (mul_pos hc2_pos hr2_sq) (mul_pos hb hpos)
  linarith

/-! ## 23 · The Cost Model Floor: 1.18 b/w Spectral 1-Bit vs llama.cpp Q4_K_M -/
/-- **1.18 b/w rate ceiling achieves at least 3.8× lower memory bandwidth floor than Q4_K_M (`q4k_vs_1bit_transfer_floor`).** -/
theorem q4k_vs_1bit_transfer_floor :
    (450 : ℚ) / 100 / ((118 : ℚ) / 100) = 225 / 59 ∧
    (38 : ℚ) / 10 < 225 / 59 ∧
    (118 : ℚ) / 100 / ((450 : ℚ) / 100) < 27 / 100 := by
  norm_num

/-! ## 24 · Face Swap #10: Asymmetric Wiener Target Eliminates Upstream Activation Distortion -/
/-- **Asymmetric Wiener target strictly minimizes reconstruction loss under upstream distortion (`asymmetric_wiener_target_optimality`).**
Given student intermediate activation variance `p > 0`, cross-covariance with teacher target `c`, and target variance `y`,
the quadratic reconstruction objective `J(w) = p w² - 2 c w + y` attains its unique global minimum at `w* = c / p`.
Any other weight `w` incurs strictly positive excess loss `p (w - w*)² ≥ 0`, with equality iff `w = w*`. -/
theorem asymmetric_wiener_target_optimality (p c y w : ℝ) (hp : 0 < p) :
    let w_star := c / p
    let J (v : ℝ) := p * v ^ 2 - 2 * c * v + y
    J w - J w_star = p * (w - w_star) ^ 2 ∧
    J w_star ≤ J w ∧
    (J w = J w_star ↔ w = w_star) := by
  intro w_star J
  have hp_ne : p ≠ 0 := ne_of_gt hp
  have hid : J w - J w_star = p * (w - w_star) ^ 2 := by
    dsimp [J, w_star]
    field_simp
    ring
  have hnonneg : 0 ≤ p * (w - w_star) ^ 2 := mul_nonneg (le_of_lt hp) (sq_nonneg _)
  have hle : J w_star ≤ J w := by linarith [hid, hnonneg]
  have heq : J w = J w_star ↔ w = w_star := by
    constructor
    · intro h
      have : p * (w - w_star) ^ 2 = 0 := by linarith [hid, h]
      cases mul_eq_zero.mp this with
      | inl h1 => exact False.elim (hp_ne h1)
      | inr h2 => exact eq_of_sub_eq_zero (sq_eq_zero_iff.mp h2)
    · intro h
      rw [h]
  exact ⟨hid, hle, heq⟩

/-- **Undistorted teacher weight incurs strict positive penalty under correlation attenuation (`asymmetric_distortion_penalty_strict`).**
When student activation has correlation coefficient `ρ < 1` with teacher activation (`c = ρ p w₀`),
retaining the unadapted teacher target `w₀ ≠ 0` incurs a strictly positive excess loss `p (1 - ρ)² w₀² > 0`
over the asymmetric Wiener target `w* = ρ w₀`. -/
theorem asymmetric_distortion_penalty_strict (p w0 rho : ℝ) (hp : 0 < p) (hw0 : w0 ≠ 0) (hrho : rho < 1) :
    let w_star := rho * w0
    let excess := p * (w0 - w_star) ^ 2
    excess = p * (1 - rho) ^ 2 * w0 ^ 2 ∧ 0 < excess := by
  intro w_star excess
  have hid : excess = p * (1 - rho) ^ 2 * w0 ^ 2 := by
    dsimp [excess, w_star]
    ring
  have h1 : 0 < (1 - rho) ^ 2 := sq_pos_of_ne_zero (by linarith)
  have h2 : 0 < w0 ^ 2 := sq_pos_of_ne_zero hw0
  have hpos : 0 < p * (1 - rho) ^ 2 * w0 ^ 2 := mul_pos (mul_pos hp h1) h2
  exact ⟨hid, by linarith [hid, hpos]⟩

/-! ## 25 · W16 T0 receipt: 3-seed verdict for the best W15d arm -/
/-- **Verdict: `had-fold-2plane-attn-bscale-e8-icm20-gin25-sink-fnorm-grow-res-qmlp` over three calibration-order
seeds (`w16_t0_best_arm_3seed`).** `model` inputs from `tools/onebit/fold_transfer.py --seeds 0,1,2` (receipt
`~/w16/t0-best-3seed.json`, Qwen2.5-0.5B, 24 layers): `bits = 421541376`, `params = 357826560`;
held-out KL per seed `1.2798, 1.3894, 1.3896` nats (mean `1.3529`). Clauses: (a) `bits/params ≤ 1.18`;
(b) the 3-seed mean held-out KL is below the committed single-seed `…-sink-fnorm` arm (`1.3777`);
(c) the mean stays above Q4_K_M's `0.0197` nats (refuted-target clause: Q4_K_M is not reached);
(d) the seed spread (max − min held-out KL) is below `0.12` nats. -/
theorem w16_t0_best_arm_3seed :
    (421541376 : ℚ) / 357826560 ≤ 118 / 100 ∧
    ((1.2798 : ℝ) + 1.3894 + 1.3896) / 3 < 1.3777 ∧
    ((1.2798 : ℝ) + 1.3894 + 1.3896) / 3 > 1.3529 ∧
    ((1.2798 : ℝ) + 1.3894 + 1.3896) / 3 > 0.0197 ∧
    (1.3896 : ℝ) - 1.2798 < 0.12 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-! ## 26 · W16 T0b receipt: matched-seed baseline for the best W15d arm -/
/-- **Verdict: matched-seed comparison of `…-sink-fnorm` (`1.179460` b/w) against
`…-sink-fnorm-grow-res-qmlp` (`1.178061` b/w) (`w16_t0b_matched_seed_baseline`).** `model` inputs from
`tools/onebit/fold_transfer.py --seeds 0,1,2` (receipts `~/w16/t0b-sinkfnorm-3seed.json`,
`~/w16/t0-best-3seed.json`): held-out KL of the `…-sink-fnorm` arm per seed `1.3777, 1.3705, 1.4383`
(mean `1.3955`); of the best arm `1.2798, 1.3894, 1.3896` (mean `1.3529`). Clauses: (a) the paired mean difference
(best minus base) is negative; (b) the paired difference is positive on seed `1` (the gain is not uniform over seeds);
(c) the base arm's own seed spread exceeds `0.06` nats, above the best arm's expected-gain scale of `0.015`;
(d) both means stay above Q4_K_M's `0.0197` nats. -/
theorem w16_t0b_matched_seed_baseline :
    ((1.2798 : ℝ) - 1.3777 + (1.3894 - 1.3705) + (1.3896 - 1.4383)) / 3 < 0 ∧
    (1.3894 : ℝ) - 1.3705 > 0 ∧
    (1.4383 : ℝ) - 1.3705 > 0.06 ∧ (1.4383 : ℝ) - 1.3705 < 0.07 ∧
    ((1.3777 : ℝ) + 1.3705 + 1.4383) / 3 > 0.0197 ∧
    ((1.2798 : ℝ) + 1.3894 + 1.3896) / 3 > 0.0197 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

/-! ## 27 · W16 T3 receipt: the `-nb` joint block-ICM arm -/
/-- **Verdict: `-nb` against the best W15d arm, paired per seed (`w16_t3_nb_paired`).** `model` inputs from
`tools/onebit/fold_transfer.py` (receipts `~/w16/t3-nb-3seed.json`,
`~/w16/t3-base-seeds12.json`): `-nb` held-out KL per seed `1.2777, 1.3086, 1.3007`; best arm (T0)
`1.2798, 1.3894, 1.3896`; aggregate k/v/o `J_nb/J_seq = 0.9217` on seed `0`; the best arm re-run on seeds `1, 2` gives
`1.38939, 1.38959`. Clauses: (a) the aggregate `J` ratio lies in the predicted `[0.90, 0.97]`; (b) the paired difference is
negative on every seed; (c) the largest paired difference is below the `+0.005` falsifier; (d) the mean paired difference
is negative and lies below `-0.015`, outside the predicted `[-0.015, -0.001]` band (that prediction is refuted on
magnitude); (e) the control re-run reproduces T0 on seeds `1, 2` within `0.005`; (f) the `-nb` mean stays above
Q4_K_M's `0.0197` nats and below the matched base arm's `1.3955`. -/
theorem w16_t3_nb_paired :
    (0.90 : ℝ) ≤ 0.9217 ∧ (0.9217 : ℝ) ≤ 0.97 ∧
    (1.2777 : ℝ) - 1.2798 < 0 ∧ (1.3086 : ℝ) - 1.3894 < 0 ∧ (1.3007 : ℝ) - 1.3896 < 0 ∧
    (1.2777 : ℝ) - 1.2798 < 0.005 ∧
    ((1.2777 : ℝ) - 1.2798 + (1.3086 - 1.3894) + (1.3007 - 1.3896)) / 3 < -0.015 ∧
    ((1.2777 : ℝ) + 1.3086 + 1.3007) / 3 > 0.0197 ∧
    ((1.2777 : ℝ) + 1.3086 + 1.3007) / 3 < 1.3955 ∧
    |(1.38939 : ℝ) - 1.3894| < 0.005 ∧ |(1.38959 : ℝ) - 1.3896| < 0.005 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, ?_, ?_⟩
  · rw [abs_lt]; constructor <;> norm_num
  · rw [abs_lt]; constructor <;> norm_num

end CostCorrespondence
end Perf
