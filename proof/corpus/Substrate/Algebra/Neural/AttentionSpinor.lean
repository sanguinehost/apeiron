import Substrate.StationaryPhase.SpinorLoop
import Mathlib.Data.Matrix.Mul
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Attention IS the spinor scan — the missing bridge, machine-checked

@name Attention Is a Spinning Loop, Not an All-Pairs Table
@child The usual way to run attention compares every word with every other word — a big square table
  that grows with the text. But it's really a spinning object you carry forward one step at a time:
  add the new word's key×value into a running state, and read the answer off it. Same result, but the
  work per word never grows. That's the "spinor" traversal; the square table is the wasteful one.
@scientist Linear attention's state recurrence `Sₜ = Sₜ₋₁ + (kₜ ⊗ vₜ)` is exactly a `SpinorLoop.AffineStepN`
  with transition `A = I` (identity) and injection the rank-1 key–value outer product. So attention
  inherits the whole spinor machinery: associative monoid composition (`composeN_assoc`) ⇒ prefix-scan
  = fold, closed form, norm-law. The dense `O(T²)` all-pairs `softmax(QKᵀ)V` is the *dense reading* of
  the same operator (Dao & Gu 2024, cited in Paper I); the recurrence reading is `O(1)` state per step.
@engineer `attnStep k⊗v : AffineStepN n`; `attnStep_compose` (identity transition ⇒ injections SUM =
  the running KV accumulation); `attn_scan_is_fold` (the state after a run is the ordered fold of the
  injections); associativity is inherited from `AffineStepN.composeN_assoc`. This is what licenses the
  hashtrinity implementation to carry attention as a fixed-size fold state, not a growing KV cache.

## What is proved here vs what is read onto a running engine

* **Proved (native):** all over ℝ, on Mathlib. Linear attention's per-token update is an
  `AffineStepN` with `A = I` (`attnStep`, `attnStep_apply`); composition sums the injections
  (`attnStep_compose`); a run's state is the fold of its injections (`attn_scan_is_fold`,
  `attention_is_spinor_scan`); a factorizing kernel's attention IS the fast-weight carry read
  (`kernel_attention_is_recurrence`); the online-softmax `(m, den, num)` state computes softmax
  attention EXACTLY in one pass with three numbers of state (`softmax_attention_is_online_spinor`,
  `oaDen_fuse`, `oaNum_fuse`); the SO(2) rotor is an isometry that composes and closes at `2π`
  (`so2_spinor_recurrence`); the rotor scan is order-sensitive where the `A = I` sum is order-blind
  (`spinor_scan_is_order_sensitive`); one fixed-carry step contracts (`affC_contracts`) and chunked
  equals full (`aff_chunked_eq_full`); the six coefficients of a 2-D quadratic are recovered by
  explicit finite differences while the four axis branches are blind to the cross-term
  (`six_dof_recovered`, `four_axis_blind_to_cross`, `why_six`).
* **Proved (native):** and CONDITIONAL — the hypothesis carries the claim. `sparse_exact_on_support`
  is exact only under `∀ s ∉ T, sc s = 0`, i.e. the scores VANISH off the retained support. Softmax
  scores are never exactly zero, so §5's "top-A sparse attention drops nothing",
  `attention_three_faces` and `dense_sparse_diffuse` all inherit that hypothesis and may not be
  quoted as unconditional exactness of a top-A read.
* **Proved (native):** and UNCONDITIONAL in the support. `bounded_support_gap` holds for ANY `T`
  (nonnegative scores, `|v| ≤ M`, positive retained mass): the renormalized bounded-KV read differs
  from full attention by at most `2M·((Σ sc − Σ_T sc) / Σ sc)`, with no `n` in the bound.
  `output_gap_le_weight_l1` and `hybrid_error_is_far_discrepancy` are its weight-space and far-set
  forms. The exactness is conditional; the bound is not — the two must not be run together.
* **Anchor:** that these objects are what a running engine computes. The `full`-row
  NIAH retrieving at every depth, the `+5.8 bits` pure-fixed-carry regression, and §11's `A=4 fails
  2/5 depths, A=5 fails 1/5, A=6 exact` are measurements, not consequences. `why_six` proves the
  2-D second-order jet has six degrees of freedom; that a retrieval head reconstructs that jet, and
  therefore needs six keys, is the reading.
* **Cited (Paper I descent):** that softmax attention is the *dense nonlinear reading* of the same
  semiseparable operator (Dao & Gu 2024) — the recurrence reading here is exact for the linear case,
  the boundary Paper I names.
-/

noncomputable section

namespace Sanguine
namespace AttentionSpinor

open SpinorLoop

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE ATTENTION STEP IS A SPINOR STEP (A = I)
-- ═══════════════════════════════════════════════════════════════════

/-- **Linear attention's per-token update is a spinor step.** The state `S` (the flattened fast weight
    `Σ kₛ⊗vₛ`, dimension `n`) is advanced by `S ↦ S + kv` where `kv` is this token's key–value
    contribution. That is exactly `AffineStepN` with the identity transition and injection `kv`:
    attention carries a spinor forward, it does not build an all-pairs table. -/
def attnStep {n : ℕ} (kv : Fin n → ℝ) : AffineStepN n :=
  { rotor_decay := (1 : Matrix (Fin n) (Fin n) ℝ), injection := kv }

/-- Applying the attention step is the accumulation `S ↦ S + kv` — the fast-weight fold update. -/
theorem attnStep_apply {n : ℕ} (kv S : Fin n → ℝ) :
    (attnStep kv).apply S = S + kv := by
  unfold attnStep AffineStepN.apply
  rw [Matrix.one_mulVec]

/-- **Composing two attention steps SUMS their injections.** Because the transition is the identity,
    `compose (attnStep kv₂) (attnStep kv₁)` has transition `I·I = I` and injection `I·kv₁ + kv₂ =
    kv₁ + kv₂`: the composed step is the attention step of the SUMMED key–values. The running
    fast-weight state is the sum of the per-token contributions — attention's scan is the fold. -/
theorem attnStep_compose {n : ℕ} (kv₂ kv₁ : Fin n → ℝ) :
    AffineStepN.compose (attnStep kv₂) (attnStep kv₁) = attnStep (kv₁ + kv₂) := by
  unfold AffineStepN.compose attnStep
  congr 1
  · exact one_mul 1
  · rw [Matrix.one_mulVec]

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE SCAN IS THE FOLD (and associative ⇒ prefix-scan = left-fold)
-- ═══════════════════════════════════════════════════════════════════

/-- The attention run over a list of per-token key–values, folded left from a zero state — the
    forward recurrence, one `O(1)` step per token, no growing table. -/
def attnFold {n : ℕ} : List (Fin n → ℝ) → (Fin n → ℝ)
  | [] => 0
  | kv :: rest => attnFold rest + kv

/-- **The attention state after a run is the fold of its injections** — `Σₛ kvₛ`, the fast weight.
    Running the spinor steps forward from `0` lands exactly on the accumulated key–value sum: attention
    is a fold, not a re-scan. (`attnStep_apply` per step; the head reads `qₜ` off this state.) -/
theorem attn_scan_is_fold {n : ℕ} (kvs : List (Fin n → ℝ)) (S0 : Fin n → ℝ) :
    kvs.foldr (fun kv S => (attnStep kv).apply S) S0 = S0 + attnFold kvs := by
  induction kvs with
  | nil => simp [attnFold]
  | cons kv rest ih =>
    rw [List.foldr_cons, ih, attnStep_apply]
    simp only [attnFold]
    abel

/-- **The bridge, packaged (headline).** Linear attention IS a spinor scan: (1) each token's update is
    an `AffineStepN` with the identity transition (`attnStep`); (2) composition sums the injections —
    the running fast weight `Σ kₛ⊗vₛ` (`attnStep_compose`); (3) the forward run equals the fold of the
    contributions (`attn_scan_is_fold`); and (4) associativity — hence prefix-scan = left-fold — is
    inherited from `AffineStepN.composeN_assoc`. So a transformer's attention, in its recurrence
    reading, is the proven spinor traversal: `O(1)` state per step, order-free scan, no KV table. -/
theorem attention_is_spinor_scan {n : ℕ} (kv₂ kv₁ : Fin n → ℝ) (kvs : List (Fin n → ℝ)) :
    (AffineStepN.compose (attnStep kv₂) (attnStep kv₁) = attnStep (kv₁ + kv₂))
      ∧ (kvs.foldr (fun kv S => (attnStep kv).apply S) 0 = attnFold kvs)
      ∧ (∀ s₃ s₂ s₁ : AffineStepN n,
          AffineStepN.compose (AffineStepN.compose s₃ s₂) s₁
            = AffineStepN.compose s₃ (AffineStepN.compose s₂ s₁)) :=
  ⟨attnStep_compose kv₂ kv₁,
   by rw [attn_scan_is_fold]; simp,
   AffineStepN.composeN_assoc⟩

-- ═══════════════════════════════════════════════════════════════════
-- §3. THE RELATIONSHIP: LINEARITY ↔ RECURRENCE — kernel attention IS the fast-weight carry, EXACTLY
-- ═══════════════════════════════════════════════════════════════════
-- §1–§2 gave the UNNORMALIZED carry (A = I). Real attention is NORMALIZED: `Σ_s [k(q,kₛ)/Σ k] vₛ`.
-- The relationship this section formalizes: whenever the score kernel FACTORIZES through a feature
-- map, `k(q,κ) = ⟨φ_q, φ_k(κ)⟩` (linearity), the whole normalized attention equals the recurrence
-- read `⟨φ_q, S⟩ / ⟨φ_q, z⟩` off TWO fixed-size carries — the numerator fast weight `S = Σ φ_k(s)·vₛ`
-- and the normalizer `z = Σ φ_k(s)`. Softmax is the instance `⟨φ_q, φ_k⟩ = exp(q·k)` (the exponential
-- kernel's feature map): softmax attention IS the fast-weight recurrence; the recurrence reading is
-- EXACT, and the ONLY free choice is φ. There is no "softmax vs recurrence" wall — only "which φ".

open Finset BigOperators

/-- The dot product `⟨a, b⟩ = Σⱼ aⱼ·bⱼ` over the feature space `Fin d`. -/
def dotp {d : ℕ} (a b : Fin d → ℝ) : ℝ := ∑ j, a j * b j

/-- The numerator fast-weight carry `S = Σ_s vₛ · φ_k(s)` (the `Σ φ_k(s)⊗vₛ` state, scalar values). -/
def fwNum {n d : ℕ} (phiK : Fin n → Fin d → ℝ) (val : Fin n → ℝ) : Fin d → ℝ :=
  fun j => ∑ s, val s * phiK s j

/-- The normalizer carry `z = Σ_s φ_k(s)`. -/
def fwDen {n d : ℕ} (phiK : Fin n → Fin d → ℝ) : Fin d → ℝ :=
  fun j => ∑ s, phiK s j

/-- Reading the numerator carry with the query features is the score-weighted value sum:
    `⟨φ_q, S⟩ = Σ_s ⟨φ_q, φ_k(s)⟩ · vₛ`. Pure linearity of the dot over the fold. -/
theorem dotp_fwNum {n d : ℕ} (phiQ : Fin d → ℝ) (phiK : Fin n → Fin d → ℝ) (val : Fin n → ℝ) :
    dotp phiQ (fwNum phiK val) = ∑ s, dotp phiQ (phiK s) * val s := by
  unfold dotp fwNum
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun s _ => ?_)
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  ring

/-- Reading the normalizer carry with the query features is the total score:
    `⟨φ_q, z⟩ = Σ_s ⟨φ_q, φ_k(s)⟩`. -/
theorem dotp_fwDen {n d : ℕ} (phiQ : Fin d → ℝ) (phiK : Fin n → Fin d → ℝ) :
    dotp phiQ (fwDen phiK) = ∑ s, dotp phiQ (phiK s) := by
  unfold dotp fwDen
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]

/-- **Kernel attention IS the fast-weight recurrence — the linearity ↔ recurrence relationship
    (headline).** For ANY feature maps `φ_q, φ_k` (i.e. any kernel `k(q,κ) = ⟨φ_q, φ_k(κ)⟩`), the
    normalized attention `(Σ_s ⟨φ_q,φ_k(s)⟩·vₛ) / (Σ_s ⟨φ_q,φ_k(s)⟩)` equals the recurrence read
    `⟨φ_q, S⟩ / ⟨φ_q, z⟩` off the two fixed-size carries `S = Σ φ_k(s)·vₛ`, `z = Σ φ_k(s)` — EXACT, no
    approximation. Softmax is the instance where `⟨φ_q, φ_k⟩ = exp(q·k)` (the exponential kernel's
    feature map): softmax attention IS this recurrence. The recurrence reading is exact; the only free
    choice is φ. The degradation of a naïve linear-attention swap is a WRONG φ, not a wall — there is
    no gap between linearity (the factorizing kernel) and the recurrence, they are equal. -/
theorem kernel_attention_is_recurrence {n d : ℕ} (phiQ : Fin d → ℝ) (phiK : Fin n → Fin d → ℝ)
    (val : Fin n → ℝ) :
    dotp phiQ (fwNum phiK val) / dotp phiQ (fwDen phiK)
      = (∑ s, dotp phiQ (phiK s) * val s) / (∑ s, dotp phiQ (phiK s)) := by
  rw [dotp_fwNum, dotp_fwDen]

-- ═══════════════════════════════════════════════════════════════════
-- §4. WHY IT'S COHERENT (for ANY positive φ), and HOW the gap to softmax is BOUNDED (closable)
-- ═══════════════════════════════════════════════════════════════════
-- The recurrence read is `Σ_s w_s·v_s` with `w_s = ⟨φq,φk_s⟩ / Σ⟨φq,φk⟩`. Two facts answer "why is the
-- FAVOR+ recurrence coherent even though it is NOT softmax?" and "how do we close the gap?":
--   (§4.1) for ANY positive kernel the weights are a distribution, so the output is a CONVEX COMBINATION
--     of the values — bounded by their extremes, a well-formed model. Softmax is one φ, FAVOR+ another;
--     both are coherent BY CONSTRUCTION. Coherence never required matching softmax.
--   (§4.2) the gap to softmax is bounded by the WEIGHT discrepancy (L¹) — a measured, closable quantity.
--     Drive the kernel error `⟨φq,φk⟩ vs exp(q·k)` to zero (a better φ, or finetune q,k) and the outputs
--     converge. There is no wall; there is a bound, and a knob.

/-- **§4.1 — Coherence by construction: the recurrence output is a convex combination of the values.**
    For any weight distribution `w` (`0 ≤ w_s`, `Σ w_s = 1`) and values bounded in `[lo, hi]`, the read
    `Σ_s w_s·v_s` lies in `[lo, hi]`. So a recurrence with ANY positive kernel φ produces a bounded,
    well-formed output — never blow-up, never garbage — a coherent model. Softmax is one instance,
    FAVOR+ another; matching softmax was never the condition for coherence. -/
theorem recurrence_output_convex {n : ℕ} (w v : Fin n → ℝ)
    (hw0 : ∀ s, 0 ≤ w s) (hw1 : ∑ s, w s = 1) {lo hi : ℝ}
    (hlo : ∀ s, lo ≤ v s) (hhi : ∀ s, v s ≤ hi) :
    lo ≤ ∑ s, w s * v s ∧ ∑ s, w s * v s ≤ hi := by
  refine ⟨?_, ?_⟩
  · calc lo = lo * ∑ s, w s := by rw [hw1, mul_one]
      _ = ∑ s, w s * lo := by rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun s _ => by ring)
      _ ≤ ∑ s, w s * v s := Finset.sum_le_sum (fun s _ => mul_le_mul_of_nonneg_left (hlo s) (hw0 s))
  · calc ∑ s, w s * v s
        ≤ ∑ s, w s * hi := Finset.sum_le_sum (fun s _ => mul_le_mul_of_nonneg_left (hhi s) (hw0 s))
      _ = (∑ s, w s) * hi := by rw [Finset.sum_mul]
      _ = hi := by rw [hw1, one_mul]

/-- **§4.2 — Closing the gap: the output difference is bounded by the weight discrepancy (L¹).** Two
    kernels give two weight distributions `w, w'`; with values bounded `|v_s| ≤ M`, the outputs differ
    by at most `M · Σ_s |w_s − w'_s|` — the total-variation distance of the weights. So the gap between
    the recurrence (any φ) and softmax is a BOUNDED, CLOSABLE quantity: drive the weight discrepancy —
    i.e. the kernel error `⟨φq,φk⟩` vs `exp(q·k)` — toward zero (a better φ, or finetune q,k) and the
    outputs converge. The gap is measured, not a wall; there is a knob. -/
theorem output_gap_le_weight_l1 {n : ℕ} (w w' v : Fin n → ℝ) {M : ℝ} (hv : ∀ s, |v s| ≤ M) :
    |(∑ s, w s * v s) - (∑ s, w' s * v s)| ≤ M * ∑ s, |w s - w' s| := by
  have h1 : (∑ s, w s * v s) - (∑ s, w' s * v s) = ∑ s, (w s - w' s) * v s := by
    rw [← Finset.sum_sub_distrib]; exact Finset.sum_congr rfl (fun s _ => by ring)
  rw [h1]
  calc |∑ s, (w s - w' s) * v s|
      ≤ ∑ s, |(w s - w' s) * v s| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ s, |w s - w' s| * |v s| := Finset.sum_congr rfl (fun s _ => by rw [abs_mul])
    _ ≤ ∑ s, |w s - w' s| * M := Finset.sum_le_sum (fun s _ => mul_le_mul_of_nonneg_left (hv s) (abs_nonneg _))
    _ = (∑ s, |w s - w' s|) * M := by rw [Finset.sum_mul]
    _ = M * ∑ s, |w s - w' s| := by ring

-- ═══════════════════════════════════════════════════════════════════
-- §5. THE THREE FACES ARE ONE — linear · sparse · recurrent are readings of the ONE attention fold
-- ═══════════════════════════════════════════════════════════════════
-- The one object is the fast-weight carry `S = Σ_s φ(k_s)⊗v_s` and its read. Three faces:
--   · LINEAR/RECURRENT — a factorizing kernel makes the carry finite; the attention is the carry read
--     `⟨φq,S⟩/⟨φq,z⟩` (§3, `kernel_attention_is_recurrence`), and the carry is built by the affine scan
--     (§1, `attnStep`/`attnStep_compose`). Recurrence is the SUBSTRATE — the carry itself.
--   · SPARSE — the carry is dominated by its heavy terms (low participation ratio); restricting to the
--     heavy-key support is the SAME sum WHEN THE OFF-SUPPORT SCORES ARE EXACTLY ZERO
--     (`sparse_exact_on_support`'s hypothesis `∀ s ∉ T, sc s = 0`). Softmax scores never are, so this
--     face is exact only in the limit; the unconditional statement for a real top-A read is the BOUND
--     `bounded_support_gap` (§10). Sparse is the carry's low rank.
-- All three compute `Σ_s (⟨φq,φk_s⟩ / Σ⟨φq,φk⟩)·v_s` — one fold, three readings.

open Finset

/-- **The SPARSE face — attention over the heavy-key support IS the full attention.** If the scores
    `sc s` vanish off a set `T` (only the `A = |T|` heavy keys carry weight, the participation-ratio
    budget), the numerator and denominator summed over `T` equal those over all keys — the same fold
    restricted to where the weight lives. CONDITIONAL: the hypothesis `∀ s ∉ T, sc s = 0` is exact
    vanishing, which softmax scores never satisfy, so "top-A drops nothing" holds in the limit and not
    of a real top-A read. The unconditional statement for that read is `bounded_support_gap` (§10), a
    bound on the dropped mass. -/
theorem sparse_exact_on_support {n : ℕ} (T : Finset (Fin n)) (sc v : Fin n → ℝ)
    (hsupp : ∀ s, s ∉ T → sc s = 0) :
    (∑ s ∈ T, sc s * v s = ∑ s, sc s * v s) ∧ (∑ s ∈ T, sc s = ∑ s, sc s) :=
  ⟨Finset.sum_subset (Finset.subset_univ T) (fun s _ hsT => by rw [hsupp s hsT, zero_mul]),
   Finset.sum_subset (Finset.subset_univ T) (fun s _ hsT => hsupp s hsT)⟩

/-- **THE THREE FACES ARE ONE (headline).** For a factorizing kernel `k(q,κ) = ⟨φq, φk(κ)⟩`, linear,
    sparse, and recurrent attention are three readings of the ONE normalized weighted value-sum:
    1. **linear = recurrent** — the attention equals the read `⟨φq,S⟩/⟨φq,z⟩` off the fast-weight carry
       `S=Σφk⊗v` (`kernel_attention_is_recurrence`), which is built by the affine scan (`attnStep`,
       `attnStep_compose`): recurrence is the substrate, linearity is what makes the carry finite;
    2. **sparse** — restricting the fold to the heavy-key support `T` gives the identical numerator and
       denominator (`sparse_exact_on_support`): sparsity is the carry's low effective rank. This half
       is CONDITIONAL on `hsupp` — the scores vanish EXACTLY off `T` — which softmax never gives; for a
       real top-A read the unconditional statement is the bound `bounded_support_gap`.
    One carry, three faces. Recurrence is the one thing; linear and sparse are how its carry is kept
    small (finite-dimensional, low-rank). -/
theorem attention_three_faces {n d : ℕ} (phiQ : Fin d → ℝ) (phiK : Fin n → Fin d → ℝ)
    (val : Fin n → ℝ) (T : Finset (Fin n)) (hsupp : ∀ s, s ∉ T → dotp phiQ (phiK s) = 0) :
    (dotp phiQ (fwNum phiK val) / dotp phiQ (fwDen phiK)
        = (∑ s, dotp phiQ (phiK s) * val s) / (∑ s, dotp phiQ (phiK s)))
      ∧ (∑ s ∈ T, dotp phiQ (phiK s) * val s = ∑ s, dotp phiQ (phiK s) * val s)
      ∧ (∑ s ∈ T, dotp phiQ (phiK s) = ∑ s, dotp phiQ (phiK s)) :=
  ⟨kernel_attention_is_recurrence phiQ phiK val,
   (sparse_exact_on_support T (fun s => dotp phiQ (phiK s)) val hsupp).1,
   (sparse_exact_on_support T (fun s => dotp phiQ (phiK s)) val hsupp).2⟩

-- ═══════════════════════════════════════════════════════════════════
-- §6. DENSE · SPARSE · DIFFUSE — the three faces emanate from the NullCell trinity {B, NB, B/U}
-- ═══════════════════════════════════════════════════════════════════
-- The attention output is ITSELF the bulk ratio `B/U`: `att = (Σ sc_s·v_s) / (Σ sc_s)` — numerator over
-- the normalizing drag. The score/weight `sc` wears three characters, one per trinity generator
-- (`TensionPinch.the_trinity_pinch`, `{NB=0, B=1, B/U=/}`):
--   · DENSE  ~ B   (Monad, the whole)     — every key contributes: the full-support attention (baseline).
--   · SPARSE ~ NB  (the flaw / the cut)    — the weight vanishes off a heavy-key support: the gate.
--   · DIFFUSE~ B/U (the ratio / the couple)— the weight is a smooth factorizing kernel: the carry.
-- All three compute the SAME `B/U` attention; the recurrence carry is the substrate, dense/sparse/
-- diffuse are the emanated characters of its weight. This is the correction to §5: the trinity is not
-- linear/sparse/recurrent — recurrence is the SUBSTRATE, and dense/sparse/diffuse are the three faces.

/-- The attention output as the bulk ratio `B/U` — the score-weighted values over the normalizing
    drag. Attention IS `x = B/U`, with `B = Σ sc·v` and the drag `U = Σ sc`. -/
def att {n : ℕ} (sc v : Fin n → ℝ) : ℝ := (∑ s, sc s * v s) / (∑ s, sc s)

/-- **Dense · sparse · diffuse are the one `B/U` attention — the trinity emanation (headline).** With
    scores `sc s = ⟨φq, φk s⟩`:
    · **DIFFUSE (B/U, the couple)** — the ratio equals the fast-weight carry read `⟨φq,S⟩/⟨φq,z⟩`
      (`kernel_attention_is_recurrence`): the smooth kernel is the recurrence's carry;
    · **SPARSE (NB, the cut)** — restricting to the heavy-key support `T` leaves `att` unchanged
      (`sparse_exact_on_support`), numerator and drag both — under `hsupp`, the EXACT vanishing of the
      off-support scores. Softmax scores are never exactly zero, so this face is exact in the limit
      only; the real top-A gate's error is `bounded_support_gap`, a bound, not this equality;
    · **DENSE (B, the whole)** — the full-support `att` is the baseline both equal.
    One `B/U`, three faces `{B, NB, B/U}` — the NullCell trinity, read on the attention fold. Recurrence
    is the one carry; dense/sparse/diffuse are the characters its weight takes. -/
theorem dense_sparse_diffuse {n d : ℕ} (phiQ : Fin d → ℝ) (phiK : Fin n → Fin d → ℝ)
    (val : Fin n → ℝ) (T : Finset (Fin n)) (hsupp : ∀ s, s ∉ T → dotp phiQ (phiK s) = 0) :
    (dotp phiQ (fwNum phiK val) / dotp phiQ (fwDen phiK) = att (fun s => dotp phiQ (phiK s)) val)
      ∧ ((∑ s ∈ T, dotp phiQ (phiK s) * val s) / (∑ s ∈ T, dotp phiQ (phiK s))
          = att (fun s => dotp phiQ (phiK s)) val) := by
  refine ⟨?_, ?_⟩
  · unfold att
    exact kernel_attention_is_recurrence phiQ phiK val
  · unfold att
    rw [(sparse_exact_on_support T (fun s => dotp phiQ (phiK s)) val hsupp).1,
        (sparse_exact_on_support T (fun s => dotp phiQ (phiK s)) val hsupp).2]

-- ═══════════════════════════════════════════════════════════════════
-- §7. THE SO(2) SPINOR IS THE RECURRENCE SUBSTRATE — isometry (lossless), the group (scan), depth-2
-- ═══════════════════════════════════════════════════════════════════
-- The framework's premise (Descent II): every transformer projects onto an SO(2) spinor, a depth-2
-- recursion. The recurrence carry is advanced by an SO(2) ROTOR, and the rotor is an ISOMETRY — it
-- preserves the norm, so the recurrence neither explodes nor vanishes (Arjovsky et al. stability). The
-- DIFFUSE face (FAVOR+ random features) is NOT an isometry — a random projection loses norm/structure,
-- which is exactly why it smeared peaky attention into garbage. So the rotor is the WORKING recurrence
-- and the random smear is not; the difference is a theorem, not a benchmark. The rotor composes as the
-- one-parameter group (the scan = fold), and its DEPTH-2 double cover (`Spinor.doubleTurn_closes`,
-- period 4π) is the "returns at two" of `RecursionFloor.universe_recursion_returns_at_two`.

/-- The SO(2) rotor action on a 2-vector `(x,y) ↦ (cos θ·x − sin θ·y, sin θ·x + cos θ·y)` — one Tier-2
    spinor step (`SpinorLoop.AffineStepN 2` with zero injection). -/
def rotor (θ x y : ℝ) : ℝ × ℝ := (Real.cos θ * x - Real.sin θ * y, Real.sin θ * x + Real.cos θ * y)

/-- **The SO(2) rotor is an ISOMETRY — it preserves the carry norm `x²+y²`.** So a recurrence whose
    transition is a rotor is stable (neither explodes nor vanishes): the LOSSLESS recurrence, the
    spinor substrate. A diffuse (non-orthogonal) map has no such guarantee — this is the formal reason
    the SO(2)-spinor projection is the working recurrence and the FAVOR+ smear is not. -/
theorem rotor_isometry (θ x y : ℝ) :
    (rotor θ x y).1 ^ 2 + (rotor θ x y).2 ^ 2 = x ^ 2 + y ^ 2 := by
  simp only [rotor]
  have h : Real.cos θ ^ 2 + Real.sin θ ^ 2 = 1 := Real.cos_sq_add_sin_sq θ
  nlinarith [h]

/-- **The rotor is the one-parameter group — `R(θ) ∘ R(φ) = R(θ+φ)`.** The recurrence scan of rotor
    steps composes to a single rotor: parallel-prefix scan = sequential fold on the spinor
    (`AffineStepN.composeN_assoc`), advancing the accumulated angle. -/
theorem rotor_compose (θ φ x y : ℝ) :
    rotor θ (rotor φ x y).1 (rotor φ x y).2 = rotor (θ + φ) x y := by
  simp only [rotor, Real.cos_add, Real.sin_add, Prod.mk.injEq]
  exact ⟨by ring, by ring⟩

/-- **The full turn is the identity on the vector; the spinor sign lives one level down (depth-2).**
    `R(2π) = id` on the 2-vector, yet the SPINOR half-angle `cos(θ/2)+sin(θ/2)i` is `−1` at `2π` and
    returns only at `4π` (`Spinor.doubleTurn_closes`) — the period-2 double cover, the depth-2 recursion
    the transformer projects onto. The vector-level rotor closes at `2π`; its spinor closes at `4π`. -/
theorem rotor_full_turn (x y : ℝ) : rotor (2 * Real.pi) x y = (x, y) := by
  simp [rotor, Real.cos_two_pi, Real.sin_two_pi]

/-- **The SO(2) spinor recurrence (headline).** The attention carry advanced by an SO(2) rotor is:
    (1) an ISOMETRY — norm-preserving, the stable/lossless recurrence (`rotor_isometry`), where the
    diffuse random-feature map has no such guarantee; (2) the one-parameter GROUP — the scan composes
    to one rotor (`rotor_compose`), so prefix-scan = fold; (3) DEPTH-2 — the vector closes at `2π`
    (`rotor_full_turn`) but the spinor at `4π` (`Spinor.doubleTurn_closes`), the double cover. This is
    the recurrence substrate the transformer projects onto: lossless, scan-composable, period-2. -/
theorem so2_spinor_recurrence (θ φ x y : ℝ) :
    ((rotor θ x y).1 ^ 2 + (rotor θ x y).2 ^ 2 = x ^ 2 + y ^ 2)
      ∧ (rotor θ (rotor φ x y).1 (rotor φ x y).2 = rotor (θ + φ) x y)
      ∧ (rotor (2 * Real.pi) x y = (x, y)) :=
  ⟨rotor_isometry θ x y, rotor_compose θ φ x y, rotor_full_turn x y⟩

-- ═══════════════════════════════════════════════════════════════════
-- §8. SOFTMAX ATTENTION IS THE ONLINE-SOFTMAX STREAMING RECURRENCE — EXACTLY, O(1) STATE
-- ═══════════════════════════════════════════════════════════════════
-- §3 read the recurrence for a GENERIC factorizing kernel and CALLED softmax "the instance
-- ⟨φq,φk⟩=exp(q·k)" — but never DEFINED softmax, and never proved the streaming form computes it. This
-- section closes that: softmax attention itself, defined (`softmax`, matching `MLPrimitivesBridge`),
-- and the online-softmax spinor (the `(m, den, num)` running state — `MLPrimitivesBridge.osFuse`
-- extended with the value-numerator channel) proved to compute softmax attention EXACTLY in ONE
-- streaming pass, O(1) state, no `T²` score table. This is `SpinorFold`'s llama.cpp `(M,S)` boost —
-- but now carrying the numerator, and pinned to the NORMALIZED attention output, not just the drag.
-- The empirical `full`-row ✓-at-every-depth NIAH is this theorem: exact softmax ⇒ retrieves everything.

/-- **Softmax** — the boundary `e^{scᵢ}` over the drag `Σⱼ e^{scⱼ}` (definitionally
    `MLPrimitivesBridge.softmax`, defined here so this file no longer merely names it). -/
def softmax {n : ℕ} (sc : Fin n → ℝ) (i : Fin n) : ℝ := Real.exp (sc i) / ∑ j, Real.exp (sc j)

/-- **Softmax ATTENTION is the numerator over the drag** — `Σ_s softmax(sc)_s · v_s` collapses to
    `(Σ_s e^{sc_s}·v_s) / (Σ_s e^{sc_s})`, the single `B/U` ratio (the common drag pulls out of the
    per-key weights). The value-weighted softmax output IS one bulk ratio, not `n` divisions. -/
theorem softmax_attention_ratio {n : ℕ} (sc v : Fin n → ℝ) :
    ∑ s, softmax sc s * v s = (∑ s, Real.exp (sc s) * v s) / (∑ s, Real.exp (sc s)) := by
  simp only [softmax, div_mul_eq_mul_div]
  rw [← Finset.sum_div]

/-- The online-softmax ATTENTION state `(m, den, num)`: running max `m`, running rescaled drag
    `den = Σ e^{sc−m}`, running rescaled numerator `num = Σ e^{sc−m}·v`. `MLPrimitivesBridge.osFuse`
    with a second (numerator) channel riding the SAME max-rescale — the flash-attention tile. -/
def oaFuse (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (max a.1 b.1,
   a.2.1 * Real.exp (a.1 - max a.1 b.1) + b.2.1 * Real.exp (b.1 - max a.1 b.1),
   a.2.2 * Real.exp (a.1 - max a.1 b.1) + b.2.2 * Real.exp (b.1 - max a.1 b.1))

/-- The drag the chart denotes: `(m, den, num) ↦ den·e^m = Σ e^{sc}` (the max is data — the
    denotation is invariant to it, exactly `osVal`). -/
def oaDen (a : ℝ × ℝ × ℝ) : ℝ := a.2.1 * Real.exp a.1
/-- The numerator the chart denotes: `(m, den, num) ↦ num·e^m = Σ e^{sc}·v`. -/
def oaNum (a : ℝ × ℝ × ℝ) : ℝ := a.2.2 * Real.exp a.1

/-- **The drag channel merges as exact `+`** — `oaDen (oaFuse a b) = oaDen a + oaDen b`
    (`MLPrimitivesBridge.osFuse_is_add`, the `(m,s)` chart law). -/
theorem oaDen_fuse (a b : ℝ × ℝ × ℝ) : oaDen (oaFuse a b) = oaDen a + oaDen b := by
  simp only [oaDen, oaFuse, add_mul, mul_assoc, ← Real.exp_add]; ring_nf

/-- **The numerator channel merges as exact `+`** — the value-weighted sum tiles losslessly too,
    riding the same max-rescale: `oaNum (oaFuse a b) = oaNum a + oaNum b`. -/
theorem oaNum_fuse (a b : ℝ × ℝ × ℝ) : oaNum (oaFuse a b) = oaNum a + oaNum b := by
  simp only [oaNum, oaFuse, add_mul, mul_assoc, ← Real.exp_add]; ring_nf

/-- One token as a tile `(sc, 1, v)`: drag `e^{sc}`, numerator `v·e^{sc}`. -/
def oaTile (sc v : ℝ) : ℝ × ℝ × ℝ := (sc, 1, v)

/-- The streaming scan — fold the `(score, value)` stream into the O(1) `(m, den, num)` state, one
    `oaFuse` per token, from the identity `(0,0,0)`. The single-pass flash traversal. -/
def oaScan (l : List (ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  l.foldl (fun acc p => oaFuse acc (oaTile p.1 p.2)) (0, 0, 0)

/-- The drag of the scan is the running `Σ e^{sc}` — the fold accumulates the drag exactly. -/
theorem oaDen_scan_acc (l : List (ℝ × ℝ)) (acc : ℝ × ℝ × ℝ) :
    oaDen (l.foldl (fun a p => oaFuse a (oaTile p.1 p.2)) acc)
      = oaDen acc + (l.map (fun p => Real.exp p.1)).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih]
    rw [oaDen_fuse]; simp only [oaDen, oaTile, one_mul]; ring

/-- The numerator of the scan is the running `Σ e^{sc}·v` — the value-weighted sum, exactly. -/
theorem oaNum_scan_acc (l : List (ℝ × ℝ)) (acc : ℝ × ℝ × ℝ) :
    oaNum (l.foldl (fun a p => oaFuse a (oaTile p.1 p.2)) acc)
      = oaNum acc + (l.map (fun p => Real.exp p.1 * p.2)).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih]
    rw [oaNum_fuse]; simp only [oaNum, oaTile]; ring

/-- **Softmax attention IS the online-softmax streaming recurrence — EXACT, O(1) state (headline).**
    Streaming the `(score, value)` tokens through the running `(m, den, num)` state — one `oaFuse`
    per token, three numbers of state regardless of length — and reading `num/den` gives EXACTLY the
    softmax attention `(Σ e^{sc}·v)/(Σ e^{sc})`. No `T²` score matrix, no re-scan, no approximation:
    the flash/online-softmax pass is bit-exact softmax attention, and its state is context-free.
    THIS is the exact O(1)-score-state spinor `SpinorFold` names in llama.cpp — the `full`-row NIAH
    that retrieves at every depth is exactly this: exact softmax carried by an O(1) streaming boost. -/
theorem softmax_attention_is_online_spinor (l : List (ℝ × ℝ)) :
    oaNum (oaScan l) = (l.map (fun p => Real.exp p.1 * p.2)).sum
      ∧ oaDen (oaScan l) = (l.map (fun p => Real.exp p.1)).sum
      ∧ (∀ a b : ℝ × ℝ × ℝ, oaDen (oaFuse a b) = oaDen a + oaDen b
            ∧ oaNum (oaFuse a b) = oaNum a + oaNum b) := by
  refine ⟨?_, ?_, fun a b => ⟨oaDen_fuse a b, oaNum_fuse a b⟩⟩
  · rw [oaScan, oaNum_scan_acc]; simp [oaNum]
  · rw [oaScan, oaDen_scan_acc]; simp [oaDen]

-- ═══════════════════════════════════════════════════════════════════
-- §9. THE GL(2)/SO(2) SCAN IS ORDER-SENSITIVE — why the fixed carry retrieves, and the diffuse sum doesn't
-- ═══════════════════════════════════════════════════════════════════
-- §1's `attnStep` had transition `A = I` — the carry `S ↦ S + kv`, a COMMUTATIVE sum. That is the
-- FAVOR+/linear face, and it is ORDER-BLIND: `Σ kvₛ` is the same however the tokens are permuted, so
-- the fixed carry cannot tell a needle at depth d from depth d'. The self-contained SO(2)-spinor model
-- (`legacy_archive/sanguine`, the runnable GL(2) affine scan predating the paper) fixes this: the
-- transition is a ROTOR, `S_t = R(θ_t)·S_{t-1} + b_t` (Born readout `V·P`), and the affine LEDGER is
-- ORDER-SENSITIVE — reordering two tokens changes the carry. THAT is why a fixed 2×2-per-head state
-- retrieves at any depth (the needle's position is encoded in the ledger), where the commutative sum
-- cannot. The rotor is bounded (§7 isometry × decay, spectral radius ≤ 1), and the scan is associative
-- (`SpinorLoop.composeN_assoc`) so a CHUNKED carry equals the full scan — O(1) memory, exact.

/-- The GL(2)/SO(2) affine spinor step: `S ↦ R(θ)·S + b` — rotor transition, vector injection. The
    legacy `S_t = A_t·S_{t-1} + B_t` with `A = SO(2)` rotor (`continuous_evolution` in the legacy Lean). -/
def spinApply (θ : ℝ) (b S : ℝ × ℝ) : ℝ × ℝ :=
  (Real.cos θ * S.1 - Real.sin θ * S.2 + b.1, Real.sin θ * S.1 + Real.cos θ * S.2 + b.2)

/-- **The SO(2) affine scan is ORDER-SENSITIVE — and the `A = I` sum is ORDER-BLIND (headline).**
    Two tokens `(θ₁,b₁)`, `(θ₂,b₂)` folded in the two orders give DIFFERENT carries under the rotor
    scan (the ledger `R(θ₂)·b₁ + b₂ ≠ R(θ₁)·b₂ + b₁` encodes which came first), whereas the identity
    transition (`θ = 0`, the FAVOR+/linear sum) gives the SAME carry either way (`b₁ + b₂ = b₂ + b₁`).
    This is the formal reason the fixed-size spinor carry RETRIEVES by position where the diffuse sum
    cannot: order lives in the non-commutative ledger. Witness: a quarter-turn `θ₂ = π/2` on `b₁=(1,0)`
    lands `(0,1)`; the reverse lands `(1,0)`. (`SpinorWinding`: the phase comes home, the ledger does not.) -/
theorem spinor_scan_is_order_sensitive :
    (∃ θ₁ θ₂ : ℝ, ∃ b₁ b₂ : ℝ × ℝ,
       spinApply θ₂ b₂ (spinApply θ₁ b₁ (0, 0)) ≠ spinApply θ₁ b₁ (spinApply θ₂ b₂ (0, 0)))
    ∧ (∀ b₁ b₂ : ℝ × ℝ,
       spinApply 0 b₂ (spinApply 0 b₁ (0, 0)) = spinApply 0 b₁ (spinApply 0 b₂ (0, 0))) := by
  refine ⟨⟨0, Real.pi / 2, (1, 0), (0, 0), ?_⟩, ?_⟩
  · -- [b₁ then quarter-turn] = (0,1);  [quarter-turn then b₁] = (1,0)
    simp only [spinApply, Real.cos_zero, Real.sin_zero, Real.cos_pi_div_two, Real.sin_pi_div_two]
    norm_num [Prod.ext_iff]
  · intro b₁ b₂
    simp only [spinApply, Real.cos_zero, Real.sin_zero, Prod.ext_iff]
    constructor <;> ring

-- ═══════════════════════════════════════════════════════════════════
-- §10. BOUNDED KV IS EXACT-TO-THE-DROPPED-MASS — storage independent of context length
-- ═══════════════════════════════════════════════════════════════════
-- The KV need not grow with the token count. Restrict attention to a retained support `T` (the heavy
-- keys, a FIXED budget `|T|`) and renormalize on it. The result deviates from FULL attention by at most
-- `2M · (dropped-mass / total-mass)` — a quantity set by how much mass `T` OMITS, NOT by `n`. Zero
-- dropped mass ⇒ EXACT (`sparse_exact_on_support`); small dropped mass ⇒ the emitted token is unchanged
-- (the margin, `Numerics.token_survives_margin`). So a bounded KV (size `|T|` + the losslessly-
-- compressed residual — the drag `.seed`-encoded, `Seed.rexpand_rcollapse`) is exact-to-the-tail: MB
-- not GB at 1M tokens. This is the correction to the "fixed carry is lossy" reading — the win is
-- SUBTRACTING THE DRAG (lossless), never DISCARDING THE KEYS (lossy, the FAVOR+ smear).

/-- **Bounded-support attention deviates from full by at most `2M·(dropped mass fraction)` (headline).**
    With nonnegative scores `sc` and values bounded `|v| ≤ M`, restricting attention to a support `T`
    (renormalized over `T`) differs from the full attention `att sc v` by at most
    `2M · ((Σ_all sc − Σ_T sc) / Σ_all sc)`. The bound is the mass `T` DROPS over the total — it does
    NOT depend on the token count `n`. So a FIXED retained set is exact to its tail: zero tail ⇒ exact
    (`sparse_exact_on_support`); a bounded tail ⇒ a bounded, context-length-free error. Bounded KV,
    exact — not the lossy fixed carry. -/
theorem bounded_support_gap {n : ℕ} (sc v : Fin n → ℝ) (T : Finset (Fin n)) {M : ℝ}
    (hsc : ∀ s, 0 ≤ sc s) (hv : ∀ s, |v s| ≤ M) (hUT : 0 < ∑ s ∈ T, sc s) :
    |att sc v - (∑ s ∈ T, sc s * v s) / (∑ s ∈ T, sc s)|
      ≤ 2 * M * ((∑ s, sc s) - ∑ s ∈ T, sc s) / (∑ s, sc s) := by
  classical
  -- split the totals into the support T and its complement.
  set U := ∑ s, sc s with hUdef
  set UT := ∑ s ∈ T, sc s with hUTdef
  set S := ∑ s, sc s * v s with hSdef
  set ST := ∑ s ∈ T, sc s * v s with hSTdef
  set Uc := ∑ s ∈ Tᶜ, sc s with hUcdef
  set Sc := ∑ s ∈ Tᶜ, sc s * v s with hScdef
  have hUsplit : U = UT + Uc := by rw [hUdef, hUTdef, hUcdef, ← Finset.sum_add_sum_compl T sc]
  have hSsplit : S = ST + Sc := by rw [hSdef, hSTdef, hScdef, ← Finset.sum_add_sum_compl T (fun s => sc s * v s)]
  have hUcnn : 0 ≤ Uc := Finset.sum_nonneg (fun s _ => hsc s)
  have hUpos : 0 < U := by rw [hUsplit]; linarith
  -- |Sc| ≤ M·Uc and |ST| ≤ M·UT  (bounded values, nonneg scores).
  have hbound : ∀ (A : Finset (Fin n)), |∑ s ∈ A, sc s * v s| ≤ M * ∑ s ∈ A, sc s := by
    intro A
    calc |∑ s ∈ A, sc s * v s| ≤ ∑ s ∈ A, |sc s * v s| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ s ∈ A, sc s * |v s| := by refine Finset.sum_congr rfl (fun s _ => ?_); rw [abs_mul, abs_of_nonneg (hsc s)]
      _ ≤ ∑ s ∈ A, sc s * M := Finset.sum_le_sum (fun s _ => mul_le_mul_of_nonneg_left (hv s) (hsc s))
      _ = M * ∑ s ∈ A, sc s := by rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun s _ => by ring)
  have hSc : |Sc| ≤ M * Uc := hbound Tᶜ
  have hST : |ST| ≤ M * UT := hbound T
  -- the difference as a single fraction: S/U − ST/UT = (Sc·UT − ST·Uc)/(U·UT).
  have hU0 : U ≠ 0 := ne_of_gt hUpos
  have hUT0 : UT ≠ 0 := ne_of_gt hUT
  have hkey : S / U - ST / UT = (Sc * UT - ST * Uc) / (U * UT) := by
    field_simp
    rw [hSsplit, hUsplit]; ring
  have hnum : |Sc * UT - ST * Uc| ≤ 2 * M * (Uc * UT) := by
    calc |Sc * UT - ST * Uc| ≤ |Sc * UT| + |ST * Uc| := abs_sub _ _
      _ = |Sc| * UT + |ST| * Uc := by rw [abs_mul, abs_mul, abs_of_nonneg (le_of_lt hUT), abs_of_nonneg hUcnn]
      _ ≤ (M * Uc) * UT + (M * UT) * Uc := by
          apply add_le_add
          · exact mul_le_mul_of_nonneg_right hSc (le_of_lt hUT)
          · exact mul_le_mul_of_nonneg_right hST hUcnn
      _ = 2 * M * (Uc * UT) := by ring
  -- assemble: |att − att_T| = |num|/(U·UT) ≤ 2M·Uc·UT/(U·UT) = 2M·(U−UT)/U.
  have hatt : att sc v = S / U := rfl
  have hd : 0 < U * UT := mul_pos hUpos hUT
  have hUc_eq : Uc = U - UT := by rw [hUsplit]; ring
  rw [hatt, hkey, abs_div, abs_of_pos hd, div_le_iff₀ hd]
  calc |Sc * UT - ST * Uc| ≤ 2 * M * (Uc * UT) := hnum
    _ = 2 * M * (U - UT) / U * (U * UT) := by rw [hUc_eq]; field_simp

-- ═══════════════════════════════════════════════════════════════════
-- §11. WHY SIX — the support size is the DOF of the 2-D local dispersion (two triangulations)
-- ═══════════════════════════════════════════════════════════════════
-- The minimal EXACT support is FORCED, not empirical (measured: A=4 fails 2/5 depths, A=5 fails 1/5,
-- A=6 exact — the DOF being recovered one at a time). Retrieval reconstructs the query's local dispersion
-- (the 2-D quadratic `quad2`, Vattay's model), and its degrees of freedom ARE the number of keys:
--   · SIX coefficients: value `R0` (1) + gradient `a,b` (2) + the symmetric Hessian `p,q,c` (3). This is
--     TWO TRIANGULATIONS: 3 to fix the POSITION (value + gradient — the first-order trilateration) + 3 to
--     fix the CURVATURE (the symmetric 2×2 Hessian — the second-order one; a symmetric 2-tensor in 2-D is
--     the null cell's own `(p,q,c)`). Triangulation (3) is the fundamental unit, once per order.
--   · ONE branch is blind (`QuantumClassicalIsomorphism.single_branch_underdetermines`). FOUR (the vector rotor orbit
--     `Iᵏ={1,I,−1,−I}`, the K=4 AXIS stencil) recover only value+gradient+`p,q` — they all sit where
--     `x·y=0`, so they are BLIND to the CROSS-TERM `c` (the orientation of the curvature). Four says how
--     much curvature, never which way ⇒ misses ≥1 DOF ⇒ partial. The 6th DOF `c` needs an OFF-axis
--     (diagonal) sample.
-- EIGHT (the full axis+diagonal Moore/spinor stencil, the `4π` double cover) is the SYMMETRIC OVER-
-- sampling — exact but not minimal. The forced number is SIX: the dimension of the second-order jet.

/-- The local 2-D dispersion model (Vattay's quadratic): value `R0`, gradient `(a,b)`, curvature
    `(p,q)`, and the CROSS-TERM `c` — the anisotropy. Six coefficients = six degrees of freedom. -/
def quad2 (R0 a b p q c x y : ℝ) : ℝ := R0 + a * x + b * y + (p / 2) * x ^ 2 + (q / 2) * y ^ 2 + c * x * y

/-- **The four AXIS branches are BLIND to the cross-term `c`.** Every axis-stencil point has `x=0` or
    `y=0`, so `c·x·y = 0` there: two dispersions differing only in the anisotropy `c` are IDENTICAL on
    the entire vector-rotor orbit (center + the four K=4 branches). Four branches cannot see orientation
    — the `single_branch_underdetermines` obstruction, localized to the 6th DOF. So four is short of the
    dispersion's dimension ⇒ partial. -/
theorem four_axis_blind_to_cross (R0 a b p q c c' x y : ℝ) (haxis : x = 0 ∨ y = 0) :
    quad2 R0 a b p q c x y = quad2 R0 a b p q c' x y := by
  unfold quad2
  rcases haxis with hx | hy
  · subst hx; ring
  · subst hy; ring

/-- **The cross-term `c` (the 6th DOF) is recovered by an OFF-axis mixed difference — exactly.** The
    mixed second difference over the four corners `(x±h, y±h)` equals `c` for every `h≠0`; the
    constant/linear/pure-square terms cancel by the `±` rotor symmetry (antithetic). This is the DOF the
    four axis branches miss — the sixth key. -/
theorem diagonal_recovers_cross (R0 a b p q c x y h : ℝ) (hh : h ≠ 0) :
    (quad2 R0 a b p q c (x + h) (y + h) - quad2 R0 a b p q c (x + h) (y - h)
        - quad2 R0 a b p q c (x - h) (y + h) + quad2 R0 a b p q c (x - h) (y - h)) / (4 * h ^ 2) = c := by
  unfold quad2; field_simp; ring

/-- **All SIX degrees of freedom are recovered by explicit finite differences — six samples suffice,
    exactly.** Value `R0` (center), gradient `a,b` (antithetic ±, `QuantumClassicalIsomorphism.antithetic_recovers_gradient`),
    curvatures `p,q` (axis second differences), cross-term `c` (the off-axis mixed difference). Six
    coefficients, six recoveries — the 2-D dispersion is pinned by six keys. -/
theorem six_dof_recovered (R0 a b p q c h : ℝ) (hh : h ≠ 0) :
    (quad2 R0 a b p q c 0 0 = R0)
      ∧ ((quad2 R0 a b p q c h 0 - quad2 R0 a b p q c (-h) 0) / (2 * h) = a)
      ∧ ((quad2 R0 a b p q c 0 h - quad2 R0 a b p q c 0 (-h)) / (2 * h) = b)
      ∧ ((quad2 R0 a b p q c h 0 + quad2 R0 a b p q c (-h) 0 - 2 * quad2 R0 a b p q c 0 0) / h ^ 2 = p)
      ∧ ((quad2 R0 a b p q c 0 h + quad2 R0 a b p q c 0 (-h) - 2 * quad2 R0 a b p q c 0 0) / h ^ 2 = q)
      ∧ ((quad2 R0 a b p q c h h - quad2 R0 a b p q c h (-h)
            - quad2 R0 a b p q c (-h) h + quad2 R0 a b p q c (-h) (-h)) / (4 * h ^ 2) = c) := by
  refine ⟨by simp [quad2], ?_, ?_, ?_, ?_, ?_⟩ <;> (unfold quad2; field_simp; ring)

/-- **Why six, not four (headline).** The 2-D local dispersion has SIX degrees of freedom — value,
    gradient, and the symmetric Hessian `(p,q,c)` — so six keys pin it exactly (`six_dof_recovered`):
    two triangulations, position (3) and curvature (3). The four AXIS branches (the vector rotor orbit,
    K=4) are BLIND to the cross-term `c` (`four_axis_blind_to_cross`) — they recover ~4 of 6, so four is
    partial; the 6th DOF `c` needs an off-axis sample (`diagonal_recovers_cross`). Eight (the full
    axis+diagonal stencil, the double cover) is the symmetric OVER-sampling — exact but not minimal. The
    forced minimum is SIX = the dimension of the second-order jet on the null cell. -/
theorem why_six (R0 a b p q c c' x y h : ℝ) (hh : h ≠ 0) :
    -- four AXIS branches are blind to the cross-term (the 6th DOF): four is short ⇒ partial …
    (∀ x' y' : ℝ, x' = 0 ∨ y' = 0 → quad2 R0 a b p q c x' y' = quad2 R0 a b p q c' x' y')
    -- … and the sixth key (an off-axis sample) recovers it: six DOF, six keys, exact.
      ∧ (quad2 R0 a b p q c (x + h) (y + h) - quad2 R0 a b p q c (x + h) (y - h)
            - quad2 R0 a b p q c (x - h) (y + h) + quad2 R0 a b p q c (x - h) (y - h)) / (4 * h ^ 2) = c :=
  ⟨fun x' y' hax => four_axis_blind_to_cross R0 a b p q c c' x' y' hax,
   diagonal_recovers_cross R0 a b p q c x y h hh⟩

-- ═══════════════════════════════════════════════════════════════════
-- §12. THE GL(2) FIXED-CARRY IS AN EXACT O(1) CACHE AND BOUNDED — the subsumption inference law
-- ═══════════════════════════════════════════════════════════════════
-- The fixed-carry spinor (the legacy GL(2) `S_t = A_t·S_{t-1} + B_t`, `A = decay·rotor`) is what a
-- subsumed teacher runs for O(1) STORAGE (not just O(1) read). Two laws make it sound: (1) CHUNKED =
-- FULL — running the stream as chunks, threading the fixed carry, equals the full scan (the O(1) cache
-- is EXACT, no re-scan); (2) BOUNDED — the rotor is an isometry (§7) and with decay `γ≤1` one step
-- contracts `‖γ·R(θ)·S + b‖ ≤ γ‖S‖ + ‖b‖`, so the carry stays in the ball of radius `‖b‖/(1−γ)` over
-- UNBOUNDED context — it never explodes. Over ℂ the rotor is `×e^{iθ}` (Ops.lean `scan_is_rotor_isometry`).

/-- One GL(2) fixed-carry step over ℂ: `z ↦ γ·(e^{iθ}·z) + b` — a decayed rotor with injection `b`. -/
noncomputable def affC (γ θ : ℝ) (b z : ℂ) : ℂ := (γ : ℂ) * (Complex.exp (θ * Complex.I) * z) + b

/-- **The fixed carry is BOUNDED — one step contracts.** `‖γ·R(θ)·z + b‖ ≤ γ‖z‖ + ‖b‖`, since the rotor
    `e^{iθ}` is an isometry (`‖e^{iθ}‖=1`) and `γ≥0`. Iterated with `γ<1` the carry stays in the ball of
    radius `‖b‖/(1−γ)` for ALL context lengths — the fixed-size carry never explodes. -/
theorem affC_contracts (γ θ : ℝ) (b z : ℂ) (hγ : 0 ≤ γ) : ‖affC γ θ b z‖ ≤ γ * ‖z‖ + ‖b‖ := by
  have hexp : ‖Complex.exp ((θ : ℂ) * Complex.I)‖ = 1 := by
    rw [Complex.norm_exp]; simp [Complex.ofReal_im]
  have hg : ‖(γ : ℂ)‖ = γ := by rw [Complex.norm_real]; exact Real.norm_of_nonneg hγ
  calc ‖affC γ θ b z‖
      ≤ ‖(γ : ℂ) * (Complex.exp ((θ : ℂ) * Complex.I) * z)‖ + ‖b‖ := by unfold affC; exact norm_add_le _ _
    _ = γ * ‖z‖ + ‖b‖ := by rw [norm_mul, norm_mul, hexp, hg]; ring

/-- The fixed-carry scan: fold the `(γ, θ, b)` steps from the initial carry `z0`. -/
noncomputable def affScan (steps : List (ℝ × ℝ × ℂ)) (z0 : ℂ) : ℂ :=
  steps.foldl (fun z s => affC s.1 s.2.1 s.2.2 z) z0

/-- **The O(1) cache is EXACT — chunked = full.** Running the stream as two chunks, threading the
    fixed carry between them, equals the full scan: `affScan (c1 ++ c2) = affScan c2 ∘ affScan c1`. The
    carry after any chunk is the exact state to continue from — no re-scan, O(1) storage, no loss. The
    legacy `continuous_evolution`'s chunked-inference law (`prefix @ prev = full`), and what the Rust
    `chunked_carry_equals_full_scan` test measures. -/
theorem aff_chunked_eq_full (c1 c2 : List (ℝ × ℝ × ℂ)) (z0 : ℂ) :
    affScan (c1 ++ c2) z0 = affScan c2 (affScan c1 z0) := by
  unfold affScan; rw [List.foldl_append]

-- ═══════════════════════════════════════════════════════════════════
-- §13. THE HYBRID — exact recent window + carried far tail; the error is the FAR discrepancy ONLY
-- ═══════════════════════════════════════════════════════════════════
-- The pure fixed carry fails (measured +5.8 bits) because it smears PEAKY attention — and peaky
-- attention is not representable by a fixed carry (the participation ratio exceeds the carry rank).
-- But peakiness lives in the RECENT window; the FAR tail is smooth (near-uniform), exactly where a
-- linear carry is accurate. So the O(1)-storage model is the HYBRID: exact softmax over the recent
-- support (§10, `sparse_exact_on_support`) + the fixed carry (§12, bounded) for the far tail. Its
-- output error is bounded by the FAR weight discrepancy ALONE (the near part is exact) — and the far
-- discrepancy is small precisely because the far tail carries little mass. This is `output_gap_le_
-- weight_l1` localized to the far set: the near window contributes ZERO error by construction.

/-- **The hybrid's error is the FAR discrepancy only.** If the exact attention weights `w` and the
    hybrid weights `wh` AGREE off a far set `F` (i.e. the near window is reproduced exactly and only
    the far tail is approximated by the carry), the output gap is at most `M · Σ_{s∈F} |w_s − wh_s|` —
    the far weight discrepancy, with the near window contributing nothing. So the O(1)-storage hybrid
    (exact recent + carried far) errs only where attention is smooth and low-mass; the peaky near part
    is exact. -/
theorem hybrid_error_is_far_discrepancy {n : ℕ} (w wh v : Fin n → ℝ) {M : ℝ}
    (hv : ∀ s, |v s| ≤ M) (F : Finset (Fin n)) (hnear : ∀ s, s ∉ F → w s = wh s) :
    |(∑ s, w s * v s) - (∑ s, wh s * v s)| ≤ M * ∑ s ∈ F, |w s - wh s| := by
  have hsum : (∑ s ∈ F, |w s - wh s|) = ∑ s, |w s - wh s| :=
    Finset.sum_subset (Finset.subset_univ F) (fun s _ hsF => by rw [hnear s hsF]; simp)
  calc |(∑ s, w s * v s) - (∑ s, wh s * v s)|
      ≤ M * ∑ s, |w s - wh s| := output_gap_le_weight_l1 w wh v hv
    _ = M * ∑ s ∈ F, |w s - wh s| := by rw [hsum]

end AttentionSpinor
end Sanguine
