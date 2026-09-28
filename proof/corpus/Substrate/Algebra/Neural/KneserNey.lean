import Mathlib.Tactic

/-!
# Kneser–Ney — `λ = D·N₁₊/N` is a weight at which the interpolation normalizes (the KL-closing lever, proof-first)

@name Why Backoff Has Exactly One Right Weight
@child A counter that only trusts the longest exact match throws away the shorter matches. The better way
  shaves a fixed sliver off every count it DID see and hands that sliver to the shorter-context guess — and
  the size of the sliver you hand over is not a free dial: it is exactly the total shaved (the discount times
  how many distinct words you saw), because probabilities must add to one. So "how much to back off" is
  DERIVED, not tuned.
@scientist `CountLMFloor` says bits/byte = H(X|conditioning) + KL; closing KL is the estimation lever, and
  modified Kneser–Ney is the best of the smoothed n-gram estimators Chen & Goodman 1998 compared — their
  measurement, read here, not proved here. Raw longest-order backoff (my current read) is NOT KN — it
  ignores the lower orders. KN INTERPOLATES: `pKN(w) = max(c(w)−D,0)/N + λ·pLower(w)`. This file proves the
  structural heart: at `λ = D·N₁₊/N` (discount × distinct-continuations / total) `pKN` is a proper
  distribution (`kn_normalizes`: `∑ pKN = 1`), because the absolute-discount mass shaved off the seen counts
  (`discountedMass_eq`: `N − D·N₁₊`) exactly equals the mass handed to the backoff. That no OTHER `λ`
  normalizes is not stated in any theorem here — the uniqueness lives in this prose. Implementing THIS (vs raw
  backoff) is what closes the measured bits/byte flattening. (The continuation-count choice of pLower is the
  further KN refinement; this normalization holds for any lower distribution.)
@engineer finite vocab `Fin V`; `total`/`N1plus`; `discountedMass_eq` (Σ max(c−D,0) = N − D·N₁₊, per-term
  case split on `0 < c w` with `0 ≤ D ≤ 1`); `pKN`; `kn_normalizes` (Σ pKN = 1 with λ = D·N₁₊/N). Elementary.

## What is proved here vs what the prose above claims

* **Proved (native):** `discountedMass_eq` (absolute discounting removes exactly `D·N₁₊`),
  `kn_normalizes` (`∑ pKN = 1` at `λ = D·N₁₊/N`), `discMass_ctx`, and
  `continuation_matches_marginal` (`∑_v knMass = cw`, the marginal identity `pCont` satisfies).
  Four elementary finite-sum identities over ℝ.
* **Proved (riding):** that bits/byte splits as entropy + KL, so closing KL is the estimation
  lever — `CountLMFloor.cross_eq_ent_add_kl` and `CountLMFloor.cross_ge_entropy`, not restated
  here.
* **Open:** UNIQUENESS, in both places the prose asserts it. `kn_normalizes`
  proves that `λ = D·N₁₊/N` normalizes — not that no other `λ` does.
  `continuation_matches_marginal` proves that `pCont` reproduces `cw` — not that no other lower
  distribution does. Both uniqueness arguments live in the prose, not in a theorem.
* **Anchor:** that modified Kneser–Ney is the optimum among smoothed n-gram
  estimators. That is the smoothing literature's result (Chen & Goodman 1998), read here, never
  proved here.
* **Anchor:** that implementing this interpolation rather than raw longest-order
  backoff is what closes the measured bits/byte flattening. That is a prediction about a run; no
  theorem in this file measures a corpus.
-/

noncomputable section

namespace Sanguine
namespace KneserNey

variable {V : ℕ}

/-- Total count in a context — `N = Σ_w c(w)`. -/
def total (c : Fin V → ℕ) : ℝ := ∑ w, (c w : ℝ)

/-- The **continuation count** `N₁₊` — the number of DISTINCT words seen after this context. Kneser–Ney's
    key quantity: the backoff mass is proportional to how many types the context continues, not the token
    total. -/
def N1plus (c : Fin V → ℕ) : ℝ := ∑ w, (if 0 < c w then (1 : ℝ) else 0)

-- ═══════════════════════════════════════════════════════════════════
-- §1. THE DISCOUNTED MASS — absolute discounting shaves exactly D·N₁₊
-- ═══════════════════════════════════════════════════════════════════

/-- **Absolute discounting shaves exactly `D·N₁₊`.** Subtracting the discount `D` from every seen count (and
    clipping unseen to 0) removes total mass `D·N₁₊` — `D` for each of the `N₁₊` distinct continuations. This
    is the mass that MUST be handed to the backoff, and it is what forces `λ`. -/
theorem discountedMass_eq (c : Fin V → ℕ) (D : ℝ) (hD0 : 0 ≤ D) (hD1 : D ≤ 1) :
    (∑ w, max ((c w : ℝ) - D) 0) = total c - D * N1plus c := by
  unfold total N1plus
  have hterm : ∀ w, max ((c w : ℝ) - D) 0 = (c w : ℝ) - D * (if 0 < c w then 1 else 0) := by
    intro w
    by_cases h : 0 < c w
    · rw [if_pos h]
      have h1 : (1 : ℝ) ≤ (c w : ℝ) := by exact_mod_cast h
      rw [max_eq_left (by linarith)]; ring
    · rw [if_neg h]
      have h0 : c w = 0 := Nat.eq_zero_of_not_pos h
      rw [h0]; push_cast; rw [max_eq_right (by linarith)]; ring
  rw [Finset.sum_congr rfl (fun w _ => hterm w), Finset.sum_sub_distrib, ← Finset.mul_sum]

-- ═══════════════════════════════════════════════════════════════════
-- §2. THE KN READ NORMALIZES — at λ = D·N₁₊/N (the mass shaved IS the mass handed over)
-- ═══════════════════════════════════════════════════════════════════

/-- The **interpolated Kneser–Ney probability**: discounted higher-order estimate plus the backoff, weighted
    by `λ = D·N₁₊/N`. Unlike raw longest-order backoff, it USES the lower distribution `pLower` on every word,
    weighted by the reliability the discount leaves. -/
def pKN (c : Fin V → ℕ) (pLower : Fin V → ℝ) (D : ℝ) (w : Fin V) : ℝ :=
  max ((c w : ℝ) - D) 0 / total c + (D * N1plus c / total c) * pLower w

/-- **The KN read is a proper distribution at `λ = D·N₁₊/N`.** The
    higher-order discounted mass `(N − D·N₁₊)/N` plus the backoff mass `(D·N₁₊/N)·1` sum to exactly `1`. The
    weight is read off conservation of probability rather than tuned; that no other `λ` normalizes is not
    what this theorem states — it says `∑ pKN = 1` at this one `λ`. This is the
    correctness that separates the KN read (which closes KL, `CountLMFloor`) from my raw longest-order backoff
    (which leaves it open). -/
theorem kn_normalizes (c : Fin V → ℕ) (pLower : Fin V → ℝ) (D : ℝ)
    (hD0 : 0 ≤ D) (hD1 : D ≤ 1) (htot : 0 < total c) (hlower : ∑ w, pLower w = 1) :
    (∑ w, pKN c pLower D w) = 1 := by
  unfold pKN
  rw [Finset.sum_add_distrib]
  have h1 : (∑ w, max ((c w : ℝ) - D) 0 / total c) = (total c - D * N1plus c) / total c := by
    rw [← Finset.sum_div, discountedMass_eq c D hD0 hD1]
  have h2 : (∑ w, D * N1plus c / total c * pLower w) = D * N1plus c / total c := by
    rw [← Finset.mul_sum, hlower, mul_one]
  rw [h1, h2, ← add_div, show total c - D * N1plus c + D * N1plus c = total c by ring]
  exact div_self htot.ne'
-- ═══════════════════════════════════════════════════════════════════
-- §3. CONTINUATION COUNTS — the lower distribution that keeps the marginal exact (the real KN insight)
-- ═══════════════════════════════════════════════════════════════════

/-- Word (token) count `c(•w) = Σ_v c(v,w)` — how often `w` occurs. -/
def cw (c : Fin V → Fin V → ℕ) (w : Fin V) : ℝ := ∑ v, (c v w : ℝ)

/-- **Continuation count `N₁₊(•w)`** — the number of DISTINCT contexts `v` that precede `w`. Kneser–Ney's
    signature: a word's backoff weight comes from its CONTEXT DIVERSITY, not its frequency. "Francisco" is
    frequent but `N₁₊(•Francisco)=1` (only "San" precedes it), so it gets a tiny backoff probability. -/
def N1dotW (c : Fin V → Fin V → ℕ) (w : Fin V) : ℝ := ∑ v, (if 0 < c v w then (1 : ℝ) else 0)

/-- Distinct successors of context `v` — `N₁₊(v•)`. -/
def N1Vdot (c : Fin V → Fin V → ℕ) (v : Fin V) : ℝ := ∑ w, (if 0 < c v w then (1 : ℝ) else 0)

/-- Total distinct bigram types `N₁₊(••) = Σ_v N₁₊(v•)`. -/
def N1dotdot (c : Fin V → Fin V → ℕ) : ℝ := ∑ v, N1Vdot c v

/-- The **continuation distribution** `pCont(w) = N₁₊(•w)/N₁₊(••)` — the KN lower-order model. -/
def pCont (c : Fin V → Fin V → ℕ) (w : Fin V) : ℝ := N1dotW c w / N1dotdot c

/-- Absolute discounting over CONTEXTS (for a fixed word `w`): `Σ_v max(c(v,w)−D,0) = c(•w) − D·N₁₊(•w)`. -/
theorem discMass_ctx (c : Fin V → Fin V → ℕ) (w : Fin V) (D : ℝ) (hD0 : 0 ≤ D) (hD1 : D ≤ 1) :
    (∑ v, max ((c v w : ℝ) - D) 0) = cw c w - D * N1dotW c w := by
  unfold cw N1dotW
  have hterm : ∀ v, max ((c v w : ℝ) - D) 0 = (c v w : ℝ) - D * (if 0 < c v w then 1 else 0) := by
    intro v
    by_cases h : 0 < c v w
    · rw [if_pos h]
      have h1 : (1 : ℝ) ≤ (c v w : ℝ) := by exact_mod_cast h
      rw [max_eq_left (by linarith)]; ring
    · rw [if_neg h]
      have h0 : c v w = 0 := Nat.eq_zero_of_not_pos h
      rw [h0]; push_cast; rw [max_eq_right (by linarith)]; ring
  rw [Finset.sum_congr rfl (fun v _ => hterm v), Finset.sum_sub_distrib, ← Finset.mul_sum]

/-- The KN mass one context contributes to word `w`: `c(v)·p_KN(w|v) = max(c(v,w)−D,0) + D·N₁₊(v•)·pCont(w)`. -/
def knMass (c : Fin V → Fin V → ℕ) (D : ℝ) (w v : Fin V) : ℝ :=
  max ((c v w : ℝ) - D) 0 + D * N1Vdot c v * pCont c w

/-- **Continuation counts keep the marginal EXACT (the headline of §3).** With the continuation
    distribution as the lower model, the KN bigram's context-weighted marginal reproduces the empirical word
    count EXACTLY: `Σ_v c(v)·p_KN(w|v) = c(•w)`. The discount `D·N₁₊(•w)` shaved off `w` across its contexts is
    handed back to it precisely by `D·pCont(w)·N₁₊(••) = D·N₁₊(•w)` — which cancels ONLY because
    `pCont(w) = N₁₊(•w)/N₁₊(••)`. So the continuation distribution KEEPS the interpolated
    absolute-discounting model marginally consistent — that no other lower model does is not what this
    theorem states, and no theorem here states it. This is WHY KN backs off by distinct
    contexts, not frequency, and it is the estimation lever `CountLMFloor` names (closing KL). -/
theorem continuation_matches_marginal (c : Fin V → Fin V → ℕ) (D : ℝ) (w : Fin V)
    (hD0 : 0 ≤ D) (hD1 : D ≤ 1) (hNN : N1dotdot c ≠ 0) :
    (∑ v, knMass c D w v) = cw c w := by
  unfold knMass
  rw [Finset.sum_add_distrib, discMass_ctx c w D hD0 hD1]
  have h2 : (∑ v, D * N1Vdot c v * pCont c w) = D * pCont c w * N1dotdot c := by
    rw [N1dotdot, Finset.mul_sum]
    exact Finset.sum_congr rfl (fun v _ => by ring)
  rw [h2, pCont]
  field_simp
  ring

end KneserNey
end Sanguine

end
