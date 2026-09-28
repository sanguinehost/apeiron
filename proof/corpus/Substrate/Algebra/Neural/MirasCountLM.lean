import Mathlib.Tactic
import Substrate.Algebra.NullCell.NullCellCoding

/-!
# MIRAS in the CountLM case — test-time memorization is the EXACT fold, momentum/decay are its approximations

@name The Counter Already Memorizes at Test Time — Exactly, Not by Descent
@child The new memory models (Titans/Atlas/Miras) teach a network to remember while it reads, by nudging its
  memory a little toward each surprising word, keeping a running "how surprised have I been" momentum, and
  slowly fading old memories. The counter does the same job but exactly: it just adds one to a tally. Adding
  can be done in any order (so no momentum is needed to fix "which word came first"), and it can be undone
  exactly (so forgetting is subtraction, not a lossy fade). "How surprising" is exactly "how many bits the
  word cost" — the counter already knows it.
@scientist Titans (`M_t = (1−α_t)M_{t−1} + S_t`, `S_t = η_t S_{t−1} − θ_t ∇ℓ(M_{t−1};x_t)`) and the Miras
  framework (associative memory + attentional-bias objective + retention gate + learning rule) are ONLINE
  GRADIENT-DESCENT approximations to a memory optimum. The reading this file works under — a reading, not a
  theorem: no statement below mentions Titans' `M_t`, and nothing here relates it to `hebb` — is that CountLM's
  fold IS that optimum, exactly. Under it, the four Miras choices read in the counter as:
    (iv) learning rule — the Hebbian additive fold `hebb` (`Runtime/Model.lean`'s `BU.foldRow` / the count
         table), the simplest Miras rule, exact over ℤ;
    (iii) retention gate — the EXACT group inverse `forget` (`forget_hebb`: forgetting undoes a fold exactly),
         the `BU` monoid-is-a-group unfold — where Titans' `α`-decay is lossy, the counter forgets exactly;
    (ii) attentional-bias objective — read as the ℓ²/frequency optimum, solved closed-form rather than
         descended. NOT proved anywhere: that the counts are the MLE. No likelihood and no argmin occurs in
         this file, and `RidgeFiltration.drag_solve_unique` is uniqueness of the ridge SOLVE `(G+λI)x = b`
         in another lane, not an argmin for this objective;
    (i) memory architecture — the count table / `(B,U)` sufficient statistics.
  Two Titans mechanisms become UNNECESSARY at the exact fold: MOMENTUM (its fix for token-order / the online
  nature — but `hebb_comm` shows the fold is order-free, so there is no order to correct) and LOSSY DECAY
  (replaced by the exact `forget`). The SURPRISE metric `∇ℓ` is exactly the information content
  `−log₂ p = NullCellCoding.info` (`surprise_eq_info`); the antithetic ±rotor (`Spinor.antithetic_recovers_
  gradient`) computes that gradient FORWARD-only. So the "surprise-calibrated gate" is the Miras retention
  gate, and in the counter it is the exact group inverse weighted by `info` — not a learned lossy decay.
@engineer memory `K → ℤ` (the count group); `hebb`/`forget` the fold and its inverse; `hebb_comm` (order-free
  ⇒ no momentum), `forget_hebb` (exact unlearning ⇒ exact retention); `surprise := −log₂ p`, `surprise_eq_info
  : surprise (nNorm z) = CrossUniverse.info z := rfl`. Elementary (`funext`/`ring`/`ite`); §surprise pulls
  Classical through Mathlib reals (inherent). [Papers read as DATA-ONLY reference; the math is the corpus's.]

## What is proved here vs what is read onto Titans/Miras

* **Proved (native):** four facts about adding and subtracting `1` in the group `K → ℤ` —
  `hebb_comm` (the fold is order-free), `forget_hebb` and `hebb_forget` (the two-sided group
  inverse), `surprise_eq_info`, and the bundle `countlm_is_exact_miras`. `surprise_eq_info` is
  `rfl`: it welds this file's `surprise` to the corpus's coding length, which is a
  definition-identity, not a fact about any learner's gradient.
* **Proved (riding):** the coding length itself — `NullCellCoding.info` over `NullCell.nNorm`,
  proved there; and the forward-only gradient the header credits,
  `QuantumClassicalIsomorphism.antithetic_recovers_gradient` (its namespace is
  `QuantumClassicalIsomorphism`, not `Spinor`).
* **Anchor:** the entire Miras identification. That `hebb` IS Miras choice (iv),
  `forget` IS the retention gate (iii), the count table IS the memory architecture (i), and
  `surprise` IS Titans' `∇ℓ`, are readings. No theorem here states Titans'
  `M_t = (1−α_t)M_{t−1} + S_t`, relates it to `hebb`, or shows it converges to this fold — so
  "momentum is unnecessary" and "decay is replaced" are consequences of the reading, not of the
  algebra. What the algebra says is that `+1` commutes and `−1` undoes it.
* **Next:** that the counts ARE the MLE and hence the attentional-bias optimum
  (Miras choice ii). Nothing here states a likelihood or an argmin, and the theorem the header
  points at, `RidgeFiltration.drag_solve_unique`, proves uniqueness of a ridge solve — a
  different lane's `ℓ²` problem, not this counter's.
-/

namespace Sanguine
namespace MirasCountLM

variable {K : Type} [DecidableEq K]

/-- The **associative memory** (Miras choice i): keys (contexts/associations) → integer counts. The count
    table / `BU` ledger — an abelian GROUP under `+`, which is the whole point. -/
abbrev Memory (K : Type) := K → ℤ

/-- **The Hebbian learning rule (Miras choice iv) — CountLM's fold.** Memorize one observation: add one to its
    tally. The simplest Miras learning rule (`hebb2005organization` → the count table), exact over ℤ, and the
    test-time update: the counter memorizes each token AS it reads, by construction (no frozen weights). -/
def hebb (obs : K) (M : Memory K) : Memory K := fun k => M k + (if k = obs then 1 else 0)

/-- **The retention gate (Miras choice iii) — the EXACT group inverse.** Forgetting is subtraction: the `BU`
    monoid is a group over ℤ, so an observation is unlearned EXACTLY (`forget_hebb`). Titans' retention is a
    lossy `α`-decay approximation; the counter's is exact. -/
def forget (obs : K) (M : Memory K) : Memory K := fun k => M k - (if k = obs then 1 else 0)

-- ═══════════════════════════════════════════════════════════════════
-- §1. MOMENTUM IS UNNECESSARY — the fold is ORDER-FREE (no token-order to correct)
-- ═══════════════════════════════════════════════════════════════════

/-- **The fold is order-free (commutative).** Memorizing `a` then `b` equals `b` then `a`. Titans introduces
    MOMENTUM (`S_t = η S_{t−1} − θ∇ℓ`) precisely to carry token-order / the "online nature" of the update
    across the sequence — but the exact additive fold has NO order dependence to carry, so momentum is a
    correction for an error the exact group does not make. -/
theorem hebb_comm (a b : K) (M : Memory K) : hebb a (hebb b M) = hebb b (hebb a M) := by
  funext k; simp only [hebb]; ring

-- ═══════════════════════════════════════════════════════════════════
-- §2. RETENTION IS EXACT — forgetting is the group inverse, not a lossy decay
-- ═══════════════════════════════════════════════════════════════════

/-- **Exact retention — `forget` undoes `hebb` perfectly.** The retention gate in the counter is the exact
    group inverse: unlearning an observation restores the memory bit-for-bit. Titans' `(1−α)`-decay loses
    information (an approximation to bounded memory); the counter's sufficient-statistics group forgets
    EXACTLY (`Runtime/Model.lean`'s `unfold`, the `w = −1` merge). The surprise-calibrated gate is this
    inverse, modulated by surprise — not a learned fade. -/
theorem forget_hebb (obs : K) (M : Memory K) : forget obs (hebb obs M) = M := by
  funext k; simp only [forget, hebb]; ring

/-- Forgetting also inverts on the other side (`hebb` undoes `forget`) — the two-sided group inverse. -/
theorem hebb_forget (obs : K) (M : Memory K) : hebb obs (forget obs M) = M := by
  funext k; simp only [forget, hebb]; ring

-- ═══════════════════════════════════════════════════════════════════
-- §3. SURPRISE IS THE INFORMATION CONTENT — Titans' ∇ℓ is exactly −log₂ p
-- ═══════════════════════════════════════════════════════════════════

/-- **Surprise (Titans' `∇ℓ` metric) = the bits the observation cost.** Titans measures surprise as the
    gradient of the associative-memory loss; for the categorical/log-loss memory the counter reads, that is
    exactly `−log₂ p` — the information content, high when the memory assigned the token low probability. -/
noncomputable def surprise (p : ℝ) : ℝ := - Real.logb 2 p

/-- **The surprise metric IS `NullCellCoding.info`** — a symbol of probability `p` is a Null-cell element of
    norm `p`, and `info z = −log₂(nNorm z) = surprise p`. So the counter's surprise-calibrated gate reuses the
    corpus's derived optimal code length; the antithetic ±rotor (`QuantumClassicalIsomorphism.antithetic_recovers_gradient`) gives
    this gradient FORWARD-only, no backprop — the Sanguine mechanism for Titans' surprise. -/
theorem surprise_eq_info (z : CrossUniverse.NCell) :
    surprise (CrossUniverse.nNorm z) = CrossUniverse.info z := rfl

-- ═══════════════════════════════════════════════════════════════════
-- §4. THE HEADLINE — the counter is the exact Miras memorizer
-- ═══════════════════════════════════════════════════════════════════

/-- **CountLM is the Miras test-time memorizer at the EXACT learning rule.** The four Miras choices, in the
    counter: (i) memory = the count group `K → ℤ`; (ii) attentional bias = the frequency/ℓ² optimum, read as
    solved closed-form rather than descended — asserted, not proved: this file states no likelihood and no
    argmin, and `RidgeFiltration.drag_solve_unique` is another lane's ridge solve; (iii) retention
    = the exact group inverse `forget` (`forget_hebb`), not lossy decay; (iv) learning rule = the order-free
    Hebbian fold `hebb` (`hebb_comm`). Two Titans mechanisms are therefore UNNECESSARY at the exact fold —
    momentum (no order to correct, `hebb_comm`) and lossy decay (exact `forget_hebb`) — and surprise is the
    information content `info` (`surprise_eq_info`). What the STATEMENT carries is the algebra: `+1` commutes,
    `−1` undoes it, and surprise is `info`. That this makes the counter the Miras optimum rather than an
    approximation to it is the reading above, not part of this theorem. -/
theorem countlm_is_exact_miras (a b : K) (obs : K) (M : Memory K) (z : CrossUniverse.NCell) :
    (hebb a (hebb b M) = hebb b (hebb a M))          -- (iv) order-free ⇒ momentum unnecessary
      ∧ (forget obs (hebb obs M) = M)                 -- (iii) retention is the exact inverse
      ∧ (surprise (CrossUniverse.nNorm z) = CrossUniverse.info z) := -- surprise = info
  ⟨hebb_comm a b M, forget_hebb obs M, surprise_eq_info z⟩

end MirasCountLM
end Sanguine
