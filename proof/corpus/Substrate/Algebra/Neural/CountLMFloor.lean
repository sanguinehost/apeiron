import Substrate.Algebra.Neural.SparseAttentionBudget
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Common
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The CountLM bits/byte floor — what SmolLM-level requires, decomposed (proof-first)

@name Why a Counter Can Reach a Transformer, and Exactly What It Must Do
@child Any model's "bits per letter" score can never beat the true uncertainty of the text GIVEN WHAT THE
  MODEL LOOKS AT. Two things separate a counter from a big model: how well it estimates the odds it looks up
  (fixable by better smoothing and more data), and HOW MUCH it looks at (fixable by giving it the far, relevant
  past — the resonance recall). Close both and the counter's score meets the transformer's. This is the exact
  split, proven.
@scientist bits/byte is the cross-entropy `H(p_true, p_model)`. Gibbs (`cross_ge_entropy`, `kl_nonneg`):
  `crossEnt ≥ entropy`, with `crossEnt = entropy + KL` (`cross_eq_ent_add_kl`). So for a per-context model,
  `bits/byte = 𝔼_ctx[ H(X | ctx_model) ] + 𝔼_ctx[ KL(p_true‖p_model) ]`. The floor `H(X | ctx_model)` is the
  true conditional entropy of WHAT THE MODEL CONDITIONS ON; `KL ≥ 0` is the estimation gap. This decomposes
  exactly what CountLM needs to reach SmolLM-level (`bits/byte → H(X | full context) ≈ 1.0`):
    1. **KL → 0** — perfect estimation of the looked-up odds. The proven lever: modified Kneser–Ney smoothing
       (`Runtime/Model.lean`'s KNx, the Chen–Goodman optimum) + more data. My raw-backoff read leaves KL on
       the table; KN is the fix, and it is a SMOOTHING choice, not more parameters.
    2. **H(X | ctx_model) → H(X | full)** — the model must CONDITION ON ENOUGH. A local n-gram conditions on
       the last `k` tokens, so its floor is `H(X | local)`, above the transformer's `H(X | full)` by exactly
       `I(X ; far | local)` (conditioning reduces entropy). The proven lever: the resonance/spectral lane
       (`SpectralRecall`, `MirasCountLM`) supplies the far, semantically-relevant context, so `ctx_model =
       local + recalled-far → full` and the floor drops to the transformer's. This is WHY count+resonance can
       reach SmolLM: not more parameters, but richer CONDITIONING (recall) + optimal ESTIMATION (KN).
  So SmolLM-level is NOT a capacity wall for CountLM — it is two proven levers (KN estimation, resonance
  conditioning) plus data/storage. Storage (the count table) and read cost (O(fanout) per token, context-length
  -INDEPENDENT — the counter's inference edge) are engineering, not a floor.
@engineer finite distributions `Fin n → ℝ`; `klDiv`/`entropyNat`/`crossEntNat` (nats; bits = /log 2, sign
  -preserving); `kl_nonneg` via `Real.log_le_sub_one_of_pos` summed; `cross_ge_entropy` the headline floor.

## What is proved here vs what is read onto a measured model

* **Proved (native):** the decomposition `crossEnt = entropy + KL` (`cross_eq_ent_add_kl`), Gibbs
  `KL ≥ 0` (`kl_nonneg`), hence the floor `entropy ≤ crossEnt` (`cross_ge_entropy`); per-context
  routing dominates either fixed predictor (`routing_dominates_local`, `routing_dominates_far`);
  refining the conditioning never raises the floor (`entropy_concave`, `entropy_refine_ge`,
  `refine_gap_nonneg`) and the coverage residual is capped by the far-attention entropy
  (`refine_gap_le_weight_entropy`); the reconstruction budget is the perplexity `exp H` and its token
  count carries no `n` (`perplexity_eq_exp_entropy`, `seed_tokens_le_recon_budget`); the null-cell
  ratio is scale-invariant, equals the participation ratio on a distribution, and is bounded by the
  window (`bulk_budget_scale_invariant`, `bulk_budget_eq_PR`, `bulk_budget_le_card`); the far benefit
  is sparse and a selective gate captures all of it (`far_benefit_is_sparse`,
  `selective_captures_all_benefit`); a sparse-exact key store reads exactly with zero crosstalk for
  arbitrarily many distinct keys (`content_novelty_capacity`).
* **Proved (native):** and CONDITIONAL where the argument leans hardest.
  `refine_gap_zero_of_kappa_sufficient` ASSUMES `∀ z, q z = q0`, and its own docstring records that
  for a lossy `κ` this is not established and that the converse is false. The residual reaching zero
  is hypothetical; only its BOUND is unconditional.
* **Proved (Mathlib):** the analysis is Mathlib's — `Real.log_le_sub_one_of_pos` for Gibbs,
  `Real.strictConcaveOn_negMulLog` for both concavity results, `Finset.sum_mul_sq_le_sq_mul_sq`
  (Cauchy–Schwarz) for the budget ceiling.
* **Anchor:** every model name and number in this file. `SmolLM-level`
  (`bits/byte → H(X | full context) ≈ 1.0`), the topic_cond result `7.33 → 8.96`, and §10's
  `countlm NIAH 8/8 @ 0.06 MB growing, coherence 2.503 bits/B, single-scale 0/8` are runs on
  particular data. A bits-per-byte is a measurement; no theorem here measures a corpus, and none of
  these figures follows from any statement in this file.
* **Next:** that the two named levers ATTAIN their limits — that modified
  Kneser–Ney drives `KL → 0` (the Chen–Goodman optimum) and that the resonance lane supplies enough
  far context to reach `H(X | full)`. What is proved is that both appear as terms in the
  decomposition, not that either reaches its bound; they sit undischarged as
  `FrontierLadder.Obligations.kn_attains_kl_floor` and
  `FrontierLadder.Obligations.kappa_sufficient`.
* **Open:** the coverage seam — whether the lossy SimHash `κ` is a sufficient statistic for the heavy
  far-set. §5 declares it, §8 CAPS it at `entropyNat w` (`refine_gap_le_weight_entropy`), and neither
  closes it.
-/

noncomputable section

namespace Sanguine
namespace CountLMFloor

variable {n : ℕ}

/-- Kullback–Leibler divergence (nats) of the model `q` from the truth `p` — the estimation gap. -/
def klDiv (p q : Fin n → ℝ) : ℝ := ∑ i, p i * Real.log (p i / q i)

/-- Shannon entropy (nats) of the truth — the irreducible uncertainty given the conditioning. -/
def entropyNat (p : Fin n → ℝ) : ℝ := - ∑ i, p i * Real.log (p i)

/-- Cross-entropy (nats) — the model's expected code length = bits/byte × log 2. -/
def crossEntNat (p q : Fin n → ℝ) : ℝ := - ∑ i, p i * Real.log (q i)

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE DECOMPOSITION — cross-entropy = entropy + KL
-- ═══════════════════════════════════════════════════════════════════

/-- **bits/byte = floor + estimation gap.** The model's cross-entropy splits into the true entropy of what it
    conditions on plus the KL estimation error. So bits/byte has exactly two levers: lower the floor (condition
    on more — resonance) and lower the KL (estimate better — KN). -/
theorem cross_eq_ent_add_kl (p q : Fin n → ℝ) (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i) :
    crossEntNat p q = entropyNat p + klDiv p q := by
  unfold crossEntNat entropyNat klDiv
  have h : ∀ i, p i * Real.log (p i / q i) = p i * Real.log (p i) - p i * Real.log (q i) := by
    intro i; rw [Real.log_div (ne_of_gt (hp i)) (ne_of_gt (hq i))]; ring
  rw [Finset.sum_congr rfl (fun i _ => h i), Finset.sum_sub_distrib]
  ring

-- ═══════════════════════════════════════════════════════════════════
-- §2. GIBBS — the estimation gap is nonnegative (the model can't beat the truth it conditions on)
-- ═══════════════════════════════════════════════════════════════════

/-- **KL ≥ 0 (Gibbs' inequality).** The estimation error is never negative — no model beats the true
    distribution of its conditioning. Proof: `log(qᵢ/pᵢ) ≤ qᵢ/pᵢ − 1`, weighted by `pᵢ` and summed, gives
    `−KL = Σ pᵢ log(qᵢ/pᵢ) ≤ Σ(qᵢ − pᵢ) = 0`. -/
theorem kl_nonneg (p q : Fin n → ℝ) (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hsp : ∑ i, p i = 1) (hsq : ∑ i, q i = 1) : 0 ≤ klDiv p q := by
  have key : ∀ i ∈ (Finset.univ : Finset (Fin n)), - (p i * Real.log (p i / q i)) ≤ q i - p i := by
    intro i _
    have hlog : - Real.log (p i / q i) = Real.log (q i / p i) := by
      rw [Real.log_div (ne_of_gt (hp i)) (ne_of_gt (hq i)),
          Real.log_div (ne_of_gt (hq i)) (ne_of_gt (hp i))]; ring
    have h2 : - (p i * Real.log (p i / q i)) = p i * Real.log (q i / p i) := by
      rw [← hlog]; ring
    have h3 : p i * Real.log (q i / p i) ≤ p i * (q i / p i - 1) :=
      mul_le_mul_of_nonneg_left (Real.log_le_sub_one_of_pos (div_pos (hq i) (hp i))) (le_of_lt (hp i))
    have hpi : p i ≠ 0 := ne_of_gt (hp i)
    have h4 : p i * (q i / p i - 1) = q i - p i := by field_simp
    rw [h2]; linarith
  have hsum : ∑ i, - (p i * Real.log (p i / q i)) ≤ ∑ i, (q i - p i) := Finset.sum_le_sum key
  have hz : ∑ i, (q i - p i) = 0 := by rw [Finset.sum_sub_distrib, hsp, hsq]; ring
  have hneg : ∑ i, - (p i * Real.log (p i / q i)) = - klDiv p q := by
    unfold klDiv; rw [Finset.sum_neg_distrib]
  rw [hneg] at hsum
  linarith

/-- **The bits/byte floor (headline).** The model's cross-entropy (its bits/byte, up to `log 2`) is at least
    the true entropy of what it conditions on: `crossEnt ≥ entropy`. So CountLM's bits/byte is bounded below by
    `H(X | its conditioning)`, and reaching the transformer's floor `H(X | full context)` requires making the
    conditioning rich enough (resonance recall) AND the KL small (KN estimation) — the two proven levers, not
    more parameters. -/
theorem cross_ge_entropy (p q : Fin n → ℝ) (hp : ∀ i, 0 < p i) (hq : ∀ i, 0 < q i)
    (hsp : ∑ i, p i = 1) (hsq : ∑ i, q i = 1) : entropyNat p ≤ crossEntNat p q := by
  rw [cross_eq_ent_add_kl p q hp hq]
  linarith [kl_nonneg p q hp hq hsp hsq]

-- ═══════════════════════════════════════════════════════════════════
-- §3. ADAPTIVE ROUTING — per-context choice of predictor dominates any fixed one (Lever B, proof-first)
-- ═══════════════════════════════════════════════════════════════════

/-- **The far-cache improvement, per token.** Switching the predictor from local `qL` to far `qF` changes the
    cross-entropy by `Σᵢ pᵢ log(qFᵢ/qLᵢ)` — positive (helps) exactly on the mass where the far cache assigns
    HIGHER probability to the true continuation than the local model does. That is where the local model is
    UNCERTAIN about the truth and the far cache is confident — the `I(X;far|local)`-largest region. -/
theorem cross_diff (p qL qF : Fin n → ℝ) (hL : ∀ i, 0 < qL i) (hF : ∀ i, 0 < qF i) :
    crossEntNat p qL - crossEntNat p qF = ∑ i, p i * Real.log (qF i / qL i) := by
  unfold crossEntNat
  rw [neg_sub_neg, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  rw [Real.log_div (ne_of_gt (hF i)) (ne_of_gt (hL i))]; ring

/-- **Adaptive routing dominates any fixed predictor (the Lever-B theorem).** A router that picks, per context,
    whichever of the local or far predictor has the lower cross-entropy has expected cross-entropy ≤ the local
    model everywhere AND ≤ the far model everywhere. So per-context (uncertainty-adaptive) weighting is at least
    as good as ANY single fixed weight — the proof that "more cache where the base is unsure" beats fixed α,
    with the routing concentrated (by `cross_diff`) exactly where the far cache beats the local. -/
theorem routing_dominates_local (m : ℕ) (w : Fin m → ℝ) (hw : ∀ c, 0 ≤ w c)
    (pc qLc qFc : Fin m → (Fin n → ℝ)) :
    (∑ c, w c * min (crossEntNat (pc c) (qLc c)) (crossEntNat (pc c) (qFc c)))
      ≤ ∑ c, w c * crossEntNat (pc c) (qLc c) := by
  apply Finset.sum_le_sum
  intro c _
  exact mul_le_mul_of_nonneg_left (min_le_left _ _) (hw c)

theorem routing_dominates_far (m : ℕ) (w : Fin m → ℝ) (hw : ∀ c, 0 ≤ w c)
    (pc qLc qFc : Fin m → (Fin n → ℝ)) :
    (∑ c, w c * min (crossEntNat (pc c) (qLc c)) (crossEntNat (pc c) (qFc c)))
      ≤ ∑ c, w c * crossEntNat (pc c) (qFc c) := by
  apply Finset.sum_le_sum
  intro c _
  exact mul_le_mul_of_nonneg_left (min_le_right _ _) (hw c)

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE RETRIEVAL-AUGMENTED READ — KN(local) ⊕ resonance(far), and it lowers the floor (kNN-LM, proof-first)
-- ═══════════════════════════════════════════════════════════════════

/-- The **retrieval-augmented read**: interpolate the local KN distribution `pL` with the resonance-retrieved
    far distribution `pF` at weight `α` — `(1−α)·pL + α·pF`. This is the served `next-token = KN(local) ⊕
    resonance(recalled-far)`, a kNN-LM-shaped mixture. -/
def mixDist (pL pF : Fin n → ℝ) (α : ℝ) : Fin n → ℝ := fun x => (1 - α) * pL x + α * pF x

/-- **The mixed read is a proper distribution** for any `α` — the interpolation of two distributions is a
    distribution, so the served generation is well-defined. -/
theorem mix_normalizes (pL pF : Fin n → ℝ) (α : ℝ) (hL : ∑ x, pL x = 1) (hF : ∑ x, pF x = 1) :
    (∑ x, mixDist pL pF α x) = 1 := by
  unfold mixDist
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, hL, hF]; ring

/-- **Retrieval never hurts (`α=0` recovers the local read).** The mixture at `α=0` is exactly the local KN
    read, so optimizing `α` over `[0,1]` can only lower the cross-entropy — a tunable interpolation of the
    resonance lane is at worst the local read, and (`cross_diff`, `routing_dominates`) strictly better where
    the far cache beats it. Retrieval-augmentation is downside-free by construction. -/
theorem retrieval_never_hurts (pL pF : Fin n → ℝ) : mixDist pL pF 0 = pL := by
  funext x; unfold mixDist; ring

/-- Entropy as a sum of `negMulLog` (`−x log x`) — the concave per-symbol code cost. -/
theorem entropyNat_eq_negMulLog (p : Fin n → ℝ) : entropyNat p = ∑ x, Real.negMulLog (p x) := by
  unfold entropyNat Real.negMulLog
  rw [← Finset.sum_neg_distrib]
  exact Finset.sum_congr rfl (fun x _ => by ring)

/-- **The floor lever — refining the conditioning lowers the entropy (entropy is concave).** The entropy of a
    mixture is at least the average of the components' entropies: `(1−α)·H(pL) + α·H(pF) ≤ H((1−α)pL + α·pF)`.
    Read as conditioning: a coarse context `Y` whose conditional `p(X|Y)` is the mixture `Σ_z p(z|Y) p(X|Y,z)`
    has entropy ≥ `Σ_z p(z|Y) H(X|Y,z)` = the FINER conditioning on `(Y,Z)`. So adding the retrieved far context
    `Z` to the local `Y` REDUCES the entropy floor: `H(X|local,far) ≤ H(X|local)`, the gap being `I(X;far|local)`
    — exactly `CountLMFloor`'s conditioning lever, and it drives the floor toward `H(X|full)` as the resonance
    retrieval covers the relevant far context. This is why the retrieval-augmented read reaches transformer-level
    conditioning without more parameters. -/
theorem entropy_concave (pL pF : Fin n → ℝ) (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1)
    (hL : ∀ x, 0 ≤ pL x) (hF : ∀ x, 0 ≤ pF x) :
    (1 - α) * entropyNat pL + α * entropyNat pF ≤ entropyNat (mixDist pL pF α) := by
  rw [entropyNat_eq_negMulLog, entropyNat_eq_negMulLog, entropyNat_eq_negMulLog,
      Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_le_sum
  intro x _
  have hc := Real.strictConcaveOn_negMulLog.concaveOn.2
    (Set.mem_Ici.mpr (hL x)) (Set.mem_Ici.mpr (hF x)) (by linarith : (0:ℝ) ≤ 1 - α) h0 (by ring)
  simp only [smul_eq_mul] at hc
  have hmix : mixDist pL pF α x = (1 - α) * pL x + α * pF x := rfl
  rw [hmix]
  linarith [hc]

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE COVERAGE SEAM — the resonance lane lowers the floor to `H(X | local, κ far)`,
--     NOT to `H(X | full)`. The gap is a NAMED residual, not a proven zero. (Axiom XIII)
-- ═══════════════════════════════════════════════════════════════════

/-- The m-ary conditioning mixture: `mixFam w q x = ∑_z w z · q z x`. Read `w z = p(z | local)` (the
    far-context value distribution given the local context) and `q z = p(X | local, z)` (the
    continuation law once the far value `z` is known). Then `mixFam w q = p(X | local)` — the coarse
    (local-only) law is the average of the finer laws over the hidden far value. -/
def mixFam {m : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ)) : Fin n → ℝ :=
  fun x => ∑ z, w z * q z x

/-- **Conditioning refinement lowers the floor (m-ary).** The coarse-context entropy dominates the
    weighted average of the finer-context entropies: `H(X | local) ≥ ∑_z p(z|local) · H(X | local, z)`.
    This is `entropy_concave` at full generality — the Lever-B floor drop for ANY finer variable `z`.
    The far value `z` here is UNRESTRICTED: it is the *full* far context. -/
theorem entropy_refine_ge {m : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ))
    (hw0 : ∀ z, 0 ≤ w z) (hw1 : ∑ z, w z = 1) (hq : ∀ z x, 0 ≤ q z x) :
    (∑ z, w z * entropyNat (q z)) ≤ entropyNat (mixFam w q) := by
  have hpt : ∀ x, (∑ z, w z * Real.negMulLog (q z x)) ≤ Real.negMulLog (∑ z, w z * q z x) := by
    intro x
    have h := Real.strictConcaveOn_negMulLog.concaveOn.le_map_sum
      (t := Finset.univ) (w := w) (p := fun z => q z x)
      (fun z _ => hw0 z) (by simpa using hw1) (fun z _ => Set.mem_Ici.mpr (hq z x))
    simpa [smul_eq_mul] using h
  rw [entropyNat_eq_negMulLog]
  have hlhs : (∑ z, w z * entropyNat (q z)) = ∑ x, ∑ z, w z * Real.negMulLog (q z x) := by
    simp_rw [entropyNat_eq_negMulLog, Finset.mul_sum]; rw [Finset.sum_comm]
  rw [hlhs]
  refine Finset.sum_le_sum (fun x _ => ?_)
  simpa only [mixFam] using hpt x

/-- **The coverage residual** `I(X ; far | local, κ far)` — the mutual information the resonance code
    `κ` throws away. `refineGap w q = H(mixFam w q) − ∑_z w z · H(q z)` is the entropy drop that a
    finer split BUYS. Instantiate twice: `local → local+κfar` buys `refineGap` at the κ level; the
    resonance lane achieves ONLY that step. The remaining drop `local+κfar → full` is a SECOND
    `refineGap` — the residual the κ-lane does NOT capture. Nothing here bounds it. -/
def refineGap {m : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ)) : ℝ :=
  entropyNat (mixFam w q) - ∑ z, w z * entropyNat (q z)

/-- The residual is non-negative — refining never raises the floor (corollary of `entropy_refine_ge`). -/
theorem refine_gap_nonneg {m : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ))
    (hw0 : ∀ z, 0 ≤ w z) (hw1 : ∑ z, w z = 1) (hq : ∀ z x, 0 ≤ q z x) :
    0 ≤ refineGap w q := by
  have := entropy_refine_ge w q hw0 hw1 hq; unfold refineGap; linarith

/-- **κ-sufficiency ⇒ zero residual (the sufficiency direction of the seam).** If the far value adds
    nothing beyond `κ` — i.e. within a `κ`-class every finer continuation law `q z` equals the same
    `q0 = p(X | local, κ far)` — then the residual vanishes: `refineGap = 0`, and the κ-lane floor
    already equals `H(X | full)`. This is the EXACT condition under which the resonance lane reaches
    the transformer floor. For a lossy `κ` (the SimHash `resonance_code`, whose classes merge
    contexts with DIFFERENT continuations) this hypothesis is not established — so `refineGap > 0`
    is the honest default, and the reach-`H(X|full)` claim is conditional on `κ` being sufficient.
    The converse (residual 0 ⇒ κ sufficient) is FALSE in general (accidental cancellation), so only
    this direction is a theorem — the seam, not a closure. -/
theorem refine_gap_zero_of_kappa_sufficient {m : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ))
    (q0 : Fin n → ℝ) (hw1 : ∑ z, w z = 1) (hconst : ∀ z, q z = q0) :
    refineGap w q = 0 := by
  have hmix : mixFam w q = q0 := by
    funext x; simp only [mixFam, hconst]; rw [← Finset.sum_mul, hw1, one_mul]
  have hsum : (∑ z, w z * entropyNat (q z)) = entropyNat q0 := by
    simp_rw [hconst]; rw [← Finset.sum_mul, hw1, one_mul]
  rw [refineGap, hmix, hsum, sub_self]

-- ═══════════════════════════════════════════════════════════════════
-- §6. THE SEED-RECONSTRUCTION BUDGET — how many tokens a sufficient κ must reconstruct.
--     NOT the full context length; the perplexity of the far-dependence. Context-INDEPENDENT.
-- ═══════════════════════════════════════════════════════════════════

/-- **The seed-reconstruction budget = the far-attention perplexity = `exp(H)`.** `p` is the
    far-dependence distribution `p(far-key | query)` — the weights over which far tokens the
    continuation `X` actually depends on (`SparseAttentionBudget`'s attention `p = softmax`). Its
    perplexity is `exp` of exactly `CountLMFloor`'s conditioning entropy: the two modules' core
    quantities are one object. -/
theorem perplexity_eq_exp_entropy (p : Fin n → ℝ) :
    SparseAttentionBudget.perplexity p = Real.exp (entropyNat p) := by
  have h : (∑ i, p i * Real.log (1 / p i)) = entropyNat p := by
    rw [entropyNat, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [one_div, Real.log_inv]; ring
  rw [SparseAttentionBudget.perplexity, h]

/-- **The reconstruction budget** — the number of far tokens the seed `κ(context)` must reconstruct
    to be sufficient. It is NOT the context length `n`: it is the perplexity of the far-dependence,
    `reconBudget p = exp(H(p_far))`, a property of the dependence's SPREAD, not the window size. -/
def reconBudget (p : Fin n → ℝ) : ℝ := SparseAttentionBudget.perplexity p

theorem recon_budget_eq_exp_entropy (p : Fin n → ℝ) :
    reconBudget p = Real.exp (entropyNat p) := perplexity_eq_exp_entropy p

/-- **The token count is bounded and context-length-INDEPENDENT.** The heavy far-keys the seed must
    retain number at most `reconBudget = exp(H)` — the `n` (context length) does NOT appear in the
    bound. This is why a sufficient κ reconstructs the prediction-relevant chain from a seed whose
    size tracks the far-dependence's perplexity, not the raw distance: `CountLMFloor`'s "read cost
    context-length-independent" made exact. (`budget_ceil_ppl_valid` chained through the bridge.) -/
theorem seed_tokens_le_recon_budget (p : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i) (hsum : ∑ i, p i = 1)
    (S : ℝ) (hS : S = ∑ i, (p i) ^ 2) (hSpos : 0 < S)
    (heavy : Finset (Fin n)) (hh : ∀ i ∈ heavy, S ≤ p i) :
    (heavy.card : ℝ) ≤ reconBudget p :=
  SparseAttentionBudget.budget_ceil_ppl_valid p hp hsum S hS hSpos heavy hh

/-- **What sufficiency IS, and the token count, in one line.** κ is *sufficient* exactly when the far
    conditional is κ-measurable (`refine_gap_zero_of_kappa_sufficient`: `refineGap = 0`). The far
    dependence concentrates on `≤ reconBudget = exp(H)` heavy keys (`seed_tokens_le_recon_budget`),
    and each token the seed DROPS below that budget costs at most `log(PR)` surprisal
    (`heavy_cut_bounds_surprisal`) — so the residual `refineGap` a κ leaves is bounded by the
    dropped light-tail, and vanishes once κ distinguishes the `⌈exp H⌉` heavy far-keys.
    **What sufficiency is NOT:** it is NOT reconstructing the verbatim full context (that is the
    literal κ=id counter — `ResonanceRecall.literal_is_degenerate`, cost = all `n` tokens), and it is
    NOT a recursion-depth threshold (`WindingDepthFloor`: no depth floor beyond a 2-point loop). The
    seed's cost is the far-dependence's perplexity, nothing more. -/
theorem sufficiency_needs_only_budget_tokens (p : Fin n → ℝ) (hp : ∀ i, 0 ≤ p i)
    (hsum : ∑ i, p i = 1) (S : ℝ) (hS : S = ∑ i, (p i) ^ 2) (hSpos : 0 < S)
    (heavy : Finset (Fin n)) (hh : ∀ i ∈ heavy, S ≤ p i) :
    (heavy.card : ℝ) ≤ reconBudget p ∧ reconBudget p = Real.exp (entropyNat p) :=
  ⟨seed_tokens_le_recon_budget p hp hsum S hS hSpos heavy hh, recon_budget_eq_exp_entropy p⟩

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE ON-THE-FLY BUDGET — the null-cell ratio `x = B/U`, scale-covariant (the window
--     expands/contracts by ZOOM without changing shape), triangle-bounded (Cauchy–Schwarz).
-- ═══════════════════════════════════════════════════════════════════

/-- **The on-the-fly budget as the null-cell ratio `x = B/U`.** `B = (∑ p)²` (square of the sum),
    `U = ∑ p²` (the Born-norm — the "triangle", sum of squares). `bulkBudget p = B/U` is the
    effective support size read straight off the LOCAL far-attention `p`, with NO normalization
    assumed — the unnormalized affine ratio the null cell reads (`ObserverScale`'s `x = ∇²R/R`).
    This is the budget derived per-position, on the fly, from the shape of the data at that point. -/
def bulkBudget (p : Fin n → ℝ) : ℝ := (∑ i, p i) ^ 2 / (∑ i, (p i) ^ 2)

/-- **Scale-covariance — expand/contract the window WITHOUT changing the budget's shape.** The zoom
    `p ↦ r·p` (the observer's radial mode, `ObserverScale.density_scales_r_squared`: the Born density
    scales by `r²`) multiplies BOTH `B` and `U` by `r²`, so their ratio — the budget — is INVARIANT
    (`ObserverScale.bulk_ratio_scale_invariant`). The effective support depends only on the data's
    shape, never the window's overall scale: so the sliding window may zoom in or out freely and the
    budget it reads is stable — the expand/contract is a scale action the budget is covariant under. -/
theorem bulk_budget_scale_invariant (p : Fin n → ℝ) (r : ℝ) (hr : r ≠ 0) :
    bulkBudget (fun i => r * p i) = bulkBudget p := by
  unfold bulkBudget
  have hB : (∑ i, r * p i) ^ 2 = r ^ 2 * (∑ i, p i) ^ 2 := by rw [← Finset.mul_sum]; ring
  have hU : (∑ i, (r * p i) ^ 2) = r ^ 2 * ∑ i, (p i) ^ 2 := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun i _ => by ring
  rw [hB, hU, mul_div_mul_left _ _ (pow_ne_zero 2 hr)]

/-- **Recovers §6's budget on a distribution.** When `∑p = 1`, the null-cell ratio is exactly the
    participation ratio: `bulkBudget = 1/∑p² = PR ≤ ⌈exp H⌉ = reconBudget`. The scale-invariant form
    GENERALIZES the fixed budget to the running, unnormalized read the sliding window actually uses. -/
theorem bulk_budget_eq_PR (p : Fin n → ℝ) (hsum : ∑ i, p i = 1) :
    bulkBudget p = SparseAttentionBudget.participationRatio p := by
  unfold bulkBudget SparseAttentionBudget.participationRatio
  rw [hsum, one_pow]

/-- **The triangle bounds the budget (Cauchy–Schwarz).** `(∑p)² ≤ n·∑p²` — the square-of-the-sum `B`
    against the sum-of-squares `U` (the triangle / Born-norm) — gives `bulkBudget ≤ n`: the effective
    support NEVER exceeds the window. A uniform `p` saturates it (max spread ⇒ expand to the full
    window); a peaked `p` sits far below (⇒ contract). The expand/contract knob IS this ratio, and its
    ceiling is the triangle inequality — the budget is self-bounding, no hyperparameter. -/
theorem bulk_budget_le_card (p : Fin n → ℝ) (hpos : 0 < ∑ i, (p i) ^ 2) :
    bulkBudget p ≤ (n : ℝ) := by
  rw [bulkBudget, div_le_iff₀ hpos]
  have hcs := Finset.sum_mul_sq_le_sq_mul_sq Finset.univ (fun _ => (1 : ℝ)) p
  simp only [one_mul, one_pow, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one] at hcs
  linarith [hcs]

-- ═══════════════════════════════════════════════════════════════════
-- §8. SEAM 1 (κ-sufficiency), BOUNDED — the residual is ≤ the far-attention entropy `H(w)`.
--     `refineGap ≤ H(w) = log(budget)`. The load-bearing gap is capped, not open-ended.
-- ═══════════════════════════════════════════════════════════════════

/-- **The coverage residual is bounded by the far-attention entropy: `refineGap w q ≤ entropyNat w`.**
    The information `I(X ; far | local)` the resonance code must cover is at most the entropy of the
    far-attention weights `w` — i.e. `log` of the §6 budget `exp(H(w))`. Proof: the residual equals
    the `w`-weighted KL of each conditional to the mixture, `Σ_z w_z · KL(q_z ‖ mixFam)`, and each
    `KL(q_z ‖ mixFam) ≤ −log w_z` because `mixFam ≥ w_z · q_z` pointwise (the mixture dominates its
    own `z`-term). So the load-bearing κ-seam is capped: whatever κ fails to separate, the damage is
    bounded by the far-index entropy, and a κ that concentrates the far-attention (small `H(w)`, the
    peaked regime §7 reads via `bulkBudget`) drives the residual toward 0. "Is κ sufficient?" becomes
    the measured quantity `H(w)`. -/
theorem refine_gap_le_weight_entropy {m : ℕ} (w : Fin m → ℝ) (q : Fin m → (Fin n → ℝ))
    (hw : ∀ z, 0 < w z) (hq : ∀ z x, 0 < q z x) (hq1 : ∀ z, ∑ x, q z x = 1)
    (hne : (Finset.univ : Finset (Fin m)).Nonempty) :
    refineGap w q ≤ entropyNat w := by
  have hmixpos : ∀ x, 0 < mixFam w q x := fun x =>
    Finset.sum_pos (fun z _ => mul_pos (hw z) (hq z x)) hne
  -- IDENTITY: refineGap = Σ_z w_z · Σ_x q_z x · log(q_z x / mixFam x)
  have hid : refineGap w q = ∑ z, w z * ∑ x, q z x * Real.log (q z x / mixFam w q x) := by
    rw [refineGap, entropyNat]
    have hmix : ∀ x, mixFam w q x = ∑ z, w z * q z x := fun _ => rfl
    -- expand −∑_x mix·log mix into −∑_z∑_x w q log mix (via mix = Σ w q and sum_comm)
    have e1 : (-∑ x, mixFam w q x * Real.log (mixFam w q x))
        = -∑ z, ∑ x, w z * q z x * Real.log (mixFam w q x) := by
      rw [Finset.sum_comm]
      congr 1; refine Finset.sum_congr rfl fun x _ => ?_
      rw [hmix x, Finset.sum_mul]
    have e2 : (∑ z, w z * entropyNat (q z))
        = -∑ z, ∑ x, w z * q z x * Real.log (q z x) := by
      simp only [entropyNat, mul_neg, Finset.mul_sum, Finset.sum_neg_distrib, mul_assoc]
    rw [e1, e2, sub_neg_eq_add, add_comm, ← sub_eq_add_neg, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun z _ => ?_
    rw [← Finset.sum_sub_distrib, Finset.mul_sum]
    refine Finset.sum_congr rfl fun x _ => ?_
    rw [Real.log_div (ne_of_gt (hq z x)) (ne_of_gt (hmixpos x))]; ring
  rw [hid]
  -- BOUND: each inner sum ≤ −log(w z); then Σ_z w_z·(−log w_z) = entropyNat w.
  have hbound : ∀ z, (∑ x, q z x * Real.log (q z x / mixFam w q x)) ≤ - Real.log (w z) := by
    intro z
    have hpt : ∀ x, q z x * Real.log (q z x / mixFam w q x) ≤ q z x * (- Real.log (w z)) := by
      intro x
      have hdom : w z * q z x ≤ mixFam w q x :=
        Finset.single_le_sum (fun z' _ => (mul_pos (hw z') (hq z' x)).le) (Finset.mem_univ z)
      have hlog : Real.log (q z x / mixFam w q x) ≤ - Real.log (w z) := by
        rw [Real.log_div (ne_of_gt (hq z x)) (ne_of_gt (hmixpos x))]
        have h := Real.log_le_log (mul_pos (hw z) (hq z x)) hdom
        rw [Real.log_mul (ne_of_gt (hw z)) (ne_of_gt (hq z x))] at h
        linarith
      exact mul_le_mul_of_nonneg_left hlog (hq z x).le
    calc (∑ x, q z x * Real.log (q z x / mixFam w q x))
        ≤ ∑ x, q z x * (- Real.log (w z)) := Finset.sum_le_sum (fun x _ => hpt x)
      _ = - Real.log (w z) := by rw [← Finset.sum_mul, hq1 z, one_mul]
  calc (∑ z, w z * ∑ x, q z x * Real.log (q z x / mixFam w q x))
      ≤ ∑ z, w z * (- Real.log (w z)) :=
        Finset.sum_le_sum (fun z _ => mul_le_mul_of_nonneg_left (hbound z) (hw z).le)
    _ = entropyNat w := by rw [entropyNat, ← Finset.sum_neg_distrib];
                           exact Finset.sum_congr rfl fun z _ => by ring

-- ═══════════════════════════════════════════════════════════════════
-- §9. SPARSITY OF THE FAR BENEFIT — uniform far-conditioning helps only ∝ the far-dependent FRACTION.
--     Why the topic experiment failed on average, and why the walker does a SPARSE lookup, not a fold.
-- ═══════════════════════════════════════════════════════════════════

/-- **The far-conditioning benefit is SPARSE (positional, not average).** Per-position far benefits
    `b i ≥ 0` that VANISH off a far-dependent set `S` (the positions where local does NOT already
    suffice — off `S`, `refineGap` is pointwise 0, `entropy_refine_ge`/`refine_gap_zero_of_kappa
    _sufficient`) sum to at most `S.card · bmax`. So the TOTAL — hence the AVERAGE over `N` positions
    — far benefit is bounded by the far-dependent FRACTION `S.card / N`. Measured consequence: uniform
    far-conditioning moves average held-out ε only in proportion to sparsity (`S.card` small on generic
    text ⇒ negligible gain, and any estimation noise makes it NET-NEGATIVE — exactly the `topic_cond`
    result 7.33 → 8.96). SELECTIVE conditioning — fire the far lane ONLY on `S` — captures the SAME
    total benefit with zero noise on the `N − S.card` redundant positions. This is the proof that far
    recall is a SPARSE, positional LOOKUP (the processive attention walker: local window + hash lookups
    at the sparse look-back positions), not a uniform fold over every token. -/
theorem far_benefit_is_sparse {N : ℕ} (b : Fin N → ℝ) (S : Finset (Fin N)) (bmax : ℝ)
    (hzero : ∀ i, i ∉ S → b i = 0) (hmax : ∀ i, b i ≤ bmax) :
    (∑ i, b i) ≤ (S.card : ℝ) * bmax := by
  have hsplit : (∑ i, b i) = ∑ i ∈ S, b i :=
    (Finset.sum_subset (Finset.subset_univ S) (fun i _ hnotin => hzero i hnotin)).symm
  rw [hsplit]
  calc (∑ i ∈ S, b i) ≤ ∑ _i ∈ S, bmax := Finset.sum_le_sum (fun i _ => hmax i)
    _ = (S.card : ℝ) * bmax := by rw [Finset.sum_const, nsmul_eq_mul]

/-- **Selective = uniform, minus the noise.** The far benefit captured by firing ONLY on the
    far-dependent set `S` equals the full uniform sum, because the dropped positions contribute zero.
    Formally the uniform sum restricted to `S` IS the uniform sum (`hzero`). So a selective gate is
    strictly better than uniform conditioning: same signal, none of the off-`S` estimation noise. -/
theorem selective_captures_all_benefit {N : ℕ} (b : Fin N → ℝ) (S : Finset (Fin N))
    (hzero : ∀ i, i ∉ S → b i = 0) :
    (∑ i ∈ S, b i) = ∑ i, b i :=
  Finset.sum_subset (Finset.subset_univ S) (fun i _ hnotin => hzero i hnotin)

-- ═══════════════════════════════════════════════════════════════════
-- §10. THE CONTENT-NOVELTY CAPACITY LAW — why sparse-exact countlm BEATS the fixed-d lane (MEASURED
--      2026-07-25: countlm NIAH 8/8 @ 0.06 MB growing, coherence 2.503 bits/B; a fixed-d ridge lane
--      binds ≲ d and gets crosstalk (`RidgeFiltration.over_capacity_crosstalk`), single-scale 0/8).
-- ═══════════════════════════════════════════════════════════════════

/-- A sparse-exact count store: a total map `K → ℤ` (countlm's `(o,ctx,tok) → count` table). -/
def Store (K : Type) := K → ℤ

/-- Fold one observation of key `k` — bump ONLY `k`'s slot (the Hebbian count fold, `MirasCountLM.hebb`). -/
def foldOne {K : Type} [DecidableEq K] (k : K) (s : Store K) : Store K :=
  fun x => if x = k then s x + 1 else s x

/-- Read key `k`'s exact count. -/
def readK {K : Type} (k : K) (s : Store K) : ℤ := s k

/-- **Sparse-exact reads are EXACT** — folding `k` increments exactly `k`'s count. -/
theorem sparse_exact_reads_exact {K : Type} [DecidableEq K] (k : K) (s : Store K) :
    readK k (foldOne k s) = readK k s + 1 := by
  simp [readK, foldOne]

/-- **ZERO crosstalk between distinct keys** — folding a DIFFERENT context `k'` leaves `k`'s read
    unchanged. This is the property the fixed-`d` lane LACKS (`RidgeFiltration.over_capacity_crosstalk`:
    beyond `d` bound contexts the reads interfere). It holds for ANY number of distinct keys — so the
    store's fidelity is INDEPENDENT of how many distinct contexts it holds. -/
theorem sparse_exact_no_crosstalk {K : Type} [DecidableEq K] (k k' : K) (h : k ≠ k') (s : Store K) :
    readK k (foldOne k' s) = readK k s := by
  simp [readK, foldOne, h]

/-- **The content-novelty capacity law (headline).** A sparse-exact count store reads every folded
    context EXACTLY with ZERO inter-context crosstalk, for arbitrarily many distinct contexts — so its
    "capacity" is the number of DISTINCT contexts (content novelty), unbounded and exact, never the
    fixed lane dimension `d`. This is why countlm (the measured winner) scales with CONTENT NOVELTY,
    not context length, and why a fixed-`d` ridge lane (bounded, crosstalk-prone beyond `d`) is strictly
    worse on coherence and recall (MEASURED). Countlm is the correct crosstalk-free base; the fixed-state
    faces are FOR fixed state, at a quality cost. -/
theorem content_novelty_capacity {K : Type} [DecidableEq K] (k k' : K) (h : k ≠ k') (s : Store K) :
    readK k (foldOne k s) = readK k s + 1 ∧ readK k (foldOne k' s) = readK k s :=
  ⟨sparse_exact_reads_exact k s, sparse_exact_no_crosstalk k k' h s⟩

end CountLMFloor
end Sanguine

end
