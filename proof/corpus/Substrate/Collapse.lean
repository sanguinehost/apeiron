import Substrate.CollapseCore
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Quaternion
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.ZMod.Basic
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Bound
import Mathlib.Tactic.Common
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.GCongr
import Mathlib.Tactic.Group
import Mathlib.Tactic.IntervalCases
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Module
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring

/-!
# Collapse — the streaming law (the fold face, Mathlib)
@name Never Count Twice
@child If you keep a running total as things arrive, you never have to go back and recount from the start. We prove the running total always equals the full recount — so the slow recount is never needed.
@scientist The streaming law: when a key is a fold of per-item contributions into a commutative monoid (`foldKey`), the incremental read equals the full re-scan (`streaming_exact`), order is irrelevant (`streaming_order_free`), and removal is exact (`streaming_exact_removal`). The proof IS the performance — the license to never re-scan.
@engineer `streaming_exact` / `streaming_order_free` / `streaming_exact_removal` over a commutative-monoid fold; the `Seed` canonical-key primitive (the discrete face) lives in `CollapseCore`. The algebraic parity behind the three lanes' agreement columns.

The `Seed` canonical-key primitive (the discrete face, pure) lives in `Substrate.CollapseCore`.
This file re-imports it and adds the **streaming law**: when the key is a fold of per-item
contributions into a commutative monoid (`foldKey`), the incremental read equals the full re-scan
(`streaming_exact`), order is irrelevant (`streaming_order_free`), and removal is exact
(`streaming_exact_removal`). The license to never re-scan -- the algebraic parity behind the three
lanes' agreement columns.

## What "the streaming law" warrants, and what the corpus reads into it

This module is a named corpus root, so read the weight before leaning on it. Every forward
statement here is one Mathlib list lemma; the value is that the three lanes were shown to be the
same lemma, not that a new fact was established.

* **Proved (Mathlib):** `streaming_exact` — `foldKey c (rest ++ flaw) = foldKey c rest +
  foldKey c flaw` is `List.map_append` and `List.sum_append` under one `simp [foldKey]`.
  Likewise `streaming_exact_removal` (the same, plus `add_sub_cancel_right` in an
  `AddCommGroup`) and `streaming_order_free`, whose proof term is exactly
  `List.Perm.sum_eq (h.map c)`.
* **Proved (Mathlib):** `stream_exact_general` — the non-commutative prefix decomposition is
  `List.foldl_append` and nothing else. Its docstring's identification of that lemma with
  Merkle–Damgård and SHA-256's block chain is a reading, listed below; the lemma itself holds
  of every `f : σ → α → σ`.
* **Proved (native):** `add_streamFold_eq` — the only real induction in the file, generalised
  over the accumulator so `cons` goes through, with `foldKey_is_streamFold` its `a = 0` corollary.
  This is the one statement here that is not a Mathlib lemma re-spelled.
* **Proved (native):** `streaming_law` — the conjunction, and the conjunction is its content.
  Read its binders: conjuncts 1 and 3 are about `rest`/`flaw`, conjunct 2 about a separate
  `{xs ys}` under `h : xs.Perm ys`. The three share no object; the `∧` is a bundle, not a
  statement about one dataset arriving in two orders.
* **Anchor:** that the three lanes named in the section comment — the Dict's
  global memo over 134,120 declarations, `Opportunity.dragfield_sterile`'s per-edge market
  field, and `TrainingCollapse.gram_stack`'s `(B,U)` batch ledger — are instances of
  `streaming_exact`. The 134,120 figure is real and reachable (`theory/THE_HASHMAP_PATH.md`,
  `theory/SUBSUMPTION_SPRINT.md`), and the two theorems exist; but neither module is imported
  here and no theorem below is applied to either, so the identification is read on, not checked.
* **Anchor:** *"The proof IS the performance"* and the `O(N·full) → O(seed +
  critical)` collapse in `streaming_exact`'s own docstring. No cost model, no complexity
  measure and no re-scan occurs in any type in this file; `foldKey` is a `List.sum`. The
  strongest honest form is that the equation licenses skipping a re-scan, not that any program
  is faster for it.
* **Anchor:** the reading downstream takes. `Runtime/SangClone.lean` cites
  `Substrate/Collapse.streaming_law` as *"the proven seal-to-disk pattern"* backing *"memory is
  bounded to one chunk + the carried model, constant in repo size"*. Boundedness, chunking and
  memory appear in no statement here. `stream_exact_general` is the right shape for that claim
  and still does not make it: it says where the fold resumes, not what it costs to hold.
* **Open:** inversion of a non-commutative `streamFold`. The section comment names it — SHA's
  step map carries Null cells and blocks inversion — and no statement here or in
  `CollapseCore` addresses it; only the forward law is proved.

-/

namespace Sanguine.Collapse

-- ═══════════════════════════════════════════════════════════════════
-- THE STREAMING LAW — the fold face: never re-touch the rest
-- ═══════════════════════════════════════════════════════════════════
-- Three lanes found this law independently in one week, each through its own
-- measured failure (the cold benchmark, three times): the kernel check (the Dict's
-- global memo — 134,120 decls, a new decl pays only its un-cached coordinates), the
-- market field (`Opportunity`: rest certified once by `dragfield_sterile`, each swap
-- read `δ − drag` at its edge), and the trainer (`TrainingCollapse.gram_stack`: the
-- (B,U) ledger folds a batch without re-touching N). They are ONE statement: when
-- the key is a FOLD of per-item contributions, the incremental read equals the full
-- re-scan exactly — so the rest state, certified once, is never re-scanned, and the
-- per-flaw cost is the flaw's alone. **The proof IS the performance**: legacy
-- re-scans precisely because it holds no such theorem for its incremental state;
-- every lane's "agreement" column is this law verified live.

/-- **The streaming key** — the canonical key of a *dataset* is the sum of its items'
    contributions `c` into a commutative monoid. The `(B,U)` normal-equations pair,
    the histogram, the field's per-edge weights, the Dict's per-coordinate verdicts
    are all this shape. -/
def foldKey {δ κ : Type} [AddCommMonoid κ] (c : δ → κ) (xs : List δ) : κ :=
  (xs.map c).sum

variable {δ : Type} {κ : Type}

/-- **Streaming is exact** — the key of `rest ++ flaw` is `key rest + key flaw`: the
    update reads ONLY the flaw; the rest, certified once, is never re-touched. The
    O(N·full) → O(seed + critical) collapse, as algebra. -/
theorem streaming_exact [AddCommMonoid κ] (c : δ → κ) (rest flaw : List δ) :
    foldKey c (rest ++ flaw) = foldKey c rest + foldKey c flaw := by
  simp [foldKey]

/-- **Streaming is order-free** — any arrival order of the same data lands the same
    key (the curriculum orders the gates, never the endpoint). -/
theorem streaming_order_free [AddCommMonoid κ] (c : δ → κ) {xs ys : List δ}
    (h : xs.Perm ys) : foldKey c xs = foldKey c ys :=
  List.Perm.sum_eq (h.map c)

/-- **Removal is exact** (the group case) — subtracting a stage's key IS never having
    folded it: provable unlearning, bit-identical to the counterfactual. -/
theorem streaming_exact_removal [AddCommGroup κ] (c : δ → κ) (rest flaw : List δ) :
    foldKey c (rest ++ flaw) - foldKey c flaw = foldKey c rest := by
  simp [foldKey]

/-- **The streaming law (headline).** Over any fold-shaped key: the incremental read
    equals the full re-scan (never re-touch the rest), arrival order is irrelevant,
    and removal is exact. The one theorem behind the three lanes' agreement columns —
    the license to never re-scan. -/
theorem streaming_law [AddCommGroup κ] (c : δ → κ) (rest flaw : List δ)
    {xs ys : List δ} (h : xs.Perm ys) :
    (foldKey c (rest ++ flaw) = foldKey c rest + foldKey c flaw)
      ∧ (foldKey c xs = foldKey c ys)
      ∧ (foldKey c (rest ++ flaw) - foldKey c flaw = foldKey c rest) :=
  ⟨streaming_exact c rest flaw, streaming_order_free c h, streaming_exact_removal c rest flaw⟩

-- ═══════════════════════════════════════════════════════════════════
-- THE STREAMING LAW, GENERALIZED — the fold is the fold (append-only)
-- ═══════════════════════════════════════════════════════════════════
-- `streaming_exact` above takes a COMMUTATIVE key. That commutativity is what
-- `streaming_order_free` buys; it is NOT what the "never re-touch the rest"
-- license needs. The prefix-decomposition holds for ANY state-dependent fold
-- `foldl f` — no commutativity, no associativity, no monoid: the rest, folded
-- once into the running state, is never re-scanned; a new chunk continues from
-- that state. This IS Merkle–Damgård / online hashing. So SHA-256's block
-- compression and the model's accumulate are the SAME object — one `foldl` —
-- and the commutative `foldKey` is the special case that ADDITIONALLY drops
-- order. What separates SHA is not the fold (shared) but (a) order-dependence
-- (the step monoid is non-commutative — a generalization, not a wall) and
-- (b) its step map carries Null cells, which blocks INVERSION (a *different*
-- operation than this forward law). The "different regime" was an artifact of
-- formalizing only the commutative case.

/-- **The general streaming key** — fold each item into the running state with a
    state-dependent step `f`. No commutativity, no monoid: the bare `foldl`. The
    histogram/`(B,U)` ledger is the special case `f s x = s + c x`; SHA's block
    chain is the case `f = compress`, the model accumulate is `f = mixAt`. -/
def streamFold {σ α : Type} (f : σ → α → σ) (init : σ) (xs : List α) : σ :=
  xs.foldl f init

/-- **The streaming law, non-commutative (the append-only optimum).** The state
    after `rest ++ flaw` is the state after `rest`, *continued* through `flaw`:
    the prefix is folded once and never re-touched. Needs neither commutativity
    nor associativity — it is the bare fold's `foldl_append`. Merkle–Damgård
    (SHA's block chain) and the model accumulate are both this; incremental
    hashing IS this theorem. -/
theorem stream_exact_general {σ α : Type} (f : σ → α → σ) (init : σ)
    (rest flaw : List α) :
    streamFold f init (rest ++ flaw)
      = streamFold f (streamFold f init rest) flaw := by
  simp [streamFold, List.foldl_append]

/-- The order-free key `foldKey` is `streamFold` over the add-step, with `0`.
    (Generalized over the accumulator so the induction goes through.) -/
theorem add_streamFold_eq {κ : Type} [AddCommMonoid κ] (c : δ → κ) :
    ∀ (xs : List δ) (a : κ), a + foldKey c xs = streamFold (fun s x => s + c x) a xs := by
  intro xs
  induction xs with
  | nil => intro a; simp [foldKey, streamFold]
  | cons x t ih =>
    intro a
    simp only [foldKey, streamFold, List.map_cons, List.sum_cons, List.foldl_cons]
    have := ih (a + c x)
    simp only [foldKey, streamFold] at this
    rw [← this, ← add_assoc]

/-- **The commutative law is the special case.** `foldKey` (the order-free key)
    is exactly `streamFold` over the add-step — the same `foldl`, plus the
    order-freedom `streaming_order_free` that SHA's non-commutative chain drops.
    The fold is shared; only commutativity is special. -/
theorem foldKey_is_streamFold {κ : Type} [AddCommMonoid κ] (c : δ → κ) (xs : List δ) :
    foldKey c xs = streamFold (fun s x => s + c x) 0 xs := by
  have := add_streamFold_eq c xs 0
  simpa using this

end Sanguine.Collapse
