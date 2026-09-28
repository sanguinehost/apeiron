/-
Lazy (conditional) rescaling of the online softmax.
wave-1 spec row for `Nous/Seq/LazySoftmax.lean`
(task ). Namespace `Nous.Seq.LazySoftmax`.
-/
import Mathlib
import Substrate.Algebra.Neural.MLPrimitivesBridge
import Nous.Combinator.ListHom

/-!
# Lazy (conditional) softmax rescaling

wave-1 spec row for `Nous/Seq/LazySoftmax.lean`
(task ). Namespace `Nous.Seq.LazySoftmax`.

FlashAttention-4 replaces the classical online-softmax running max `m_j = max(m_{j-1}, rowmax(S_j))`
by a *lazy reference* `m_ref` that is updated only when the incoming block maximum exceeds the
reference by more than a threshold `τ ≥ 0`. The paper's conditional-rescaling rule
(arXiv 2603.05451, §3.1.4, equation (6)) is

```
O_j = e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j   if m_j - m_{j-1} > τ
O_j = O_{j-1} + e^{S_j - m_{j-1}} V_j                  otherwise
```

The reference implementation gates the rescale on whether the log2 scale `acc_scale_` is inside the
threshold window (`ref/flash-attention/flash_attn/cute/softmax.py:354-358`, `rescale_threshold`
8.0 selected in `ref/flash-attention/flash_attn/cute/interface.py:1195-1197`).

This module states over `ℝ`:

* `softmax_ref_invariant` — the reference-max quotient `(Σ vᵢe^{xᵢ−m})/(Σ e^{xᵢ−m})` is the
  softmax-weighted average, independent of the reference `m`;
* `lazy_exact` — after any sequence of lazy steps from the neutral state, `n / s` is the
  softmax-weighted average, for every threshold `τ` (and in particular every `τ ≥ 0`);
* `lazy_bounded` — every accumulated term is `≤ e^τ`, hence `s ≤ count · e^τ`;
* `lazyMerge_exact` / `lazy_merge_tree_exact` — merging two lazy states (the smaller-reference one
  is rescaled) is exact, and any merge tree over the sequence agrees; the tree layer is the
  `(m_ref, s, n)`-carrier counterpart of `Nous.Combinator.ListHom.os_tree`;
* `rescale_count_le` — with slack `τ` the number of rescales is at most the number with `τ = 0`
  (the number of strict running-max rises).

The online-softmax chart and its merge algebra are imported, not restated:
`Sanguine.MLPrimitivesBridge.{osFuse, osVal, osFuse_is_add, osFuse_comm}` and
`Nous.Combinator.ListHom.{MTree, flat, osFuseNum, os_denominator, os_numerator, os_tree}`.
-/

set_option autoImplicit false
set_option warningAsError true

namespace Nous.Seq.LazySoftmax

open Nous.Combinator.ListHom (MTree flat)

noncomputable section

/-! ## Reference-max independence -/

/-- A lazy online-softmax state `(m_ref, s, n)`: the reference max, the rescaled denominator
`s = Σ e^{xᵢ − m_ref}` and the rescaled numerator `n = Σ vᵢe^{xᵢ − m_ref}` accumulated so far. The
normalized output is `n / s`, independent of `m_ref` (`softmax_ref_invariant`). -/
structure LazyState where
  /-- Reference maximum. -/
  mref : ℝ
  /-- Rescaled denominator `Σ e^{xᵢ − m_ref}`. -/
  s : ℝ
  /-- Rescaled numerator `Σ vᵢe^{xᵢ − m_ref}`. -/
  n : ℝ

/-- The neutral initial state with reference `m0` and no accumulated mass. -/
def init (m0 : ℝ) : LazyState := ⟨m0, 0, 0⟩

/-- Reference denominator `Σ e^{xᵢ − m}` for a list of score-value pairs. -/
def refDenom (m : ℝ) (l : List (ℝ × ℝ)) : ℝ :=
  (l.map fun sv => Real.exp (sv.1 - m)).sum

/-- Reference numerator `Σ vᵢe^{xᵢ − m}` for a list of score-value pairs. -/
def refNum (m : ℝ) (l : List (ℝ × ℝ)) : ℝ :=
  (l.map fun sv => sv.2 * Real.exp (sv.1 - m)).sum

/-- The softmax-weighted average `(Σ vᵢe^{xᵢ}) / (Σ e^{xᵢ})` (reference `0`). -/
def softmaxWeightedAvg (l : List (ℝ × ℝ)) : ℝ :=
  (l.map fun sv => sv.2 * Real.exp sv.1).sum / (l.map fun sv => Real.exp sv.1).sum

theorem refDenom_append (m : ℝ) (l₁ l₂ : List (ℝ × ℝ)) :
    refDenom m (l₁ ++ l₂) = refDenom m l₁ + refDenom m l₂ := by
  simp [refDenom, List.map_append, List.sum_append]

theorem refNum_append (m : ℝ) (l₁ l₂ : List (ℝ × ℝ)) :
    refNum m (l₁ ++ l₂) = refNum m l₁ + refNum m l₂ := by
  simp [refNum, List.map_append, List.sum_append]

theorem refDenom_singleton (m x v : ℝ) : refDenom m [(x, v)] = Real.exp (x - m) := by
  simp [refDenom]

theorem refNum_singleton (m x v : ℝ) : refNum m [(x, v)] = v * Real.exp (x - m) := by
  simp [refNum]

/-- Changing the reference scales the denominator by `e^{m − m'}`. -/
theorem refDenom_scale (m m' : ℝ) (l : List (ℝ × ℝ)) :
    refDenom m' l = refDenom m l * Real.exp (m - m') := by
  induction l with
  | nil => simp [refDenom]
  | cons sv rest ih =>
    change Real.exp (sv.1 - m') + refDenom m' rest =
      (Real.exp (sv.1 - m) + refDenom m rest) * Real.exp (m - m')
    rw [ih, add_mul]
    have hexp : sv.1 - m' = (sv.1 - m) + (m - m') := by ring
    rw [hexp, Real.exp_add]

/-- Changing the reference scales the numerator by `e^{m − m'}`. -/
theorem refNum_scale (m m' : ℝ) (l : List (ℝ × ℝ)) :
    refNum m' l = refNum m l * Real.exp (m - m') := by
  induction l with
  | nil => simp [refNum]
  | cons sv rest ih =>
    change sv.2 * Real.exp (sv.1 - m') + refNum m' rest =
      (sv.2 * Real.exp (sv.1 - m) + refNum m rest) * Real.exp (m - m')
    rw [ih, add_mul]
    have hexp : sv.1 - m' = (sv.1 - m) + (m - m') := by ring
    rw [hexp, Real.exp_add]
    ring

/-- The softmax-weighted average is the reference-`0` quotient. -/
theorem softmaxWeightedAvg_eq_zero (l : List (ℝ × ℝ)) :
    softmaxWeightedAvg l = refNum 0 l / refDenom 0 l := by
  simp [softmaxWeightedAvg, refNum, refDenom]

/-- **Reference-max invariance.** The quotient `(Σ vᵢe^{xᵢ−m})/(Σ e^{xᵢ−m})` is the
softmax-weighted average, for every reference `m` and every nonempty list. -/
theorem softmax_ref_invariant (m : ℝ) (l : List (ℝ × ℝ)) (_hl : l ≠ []) :
    refNum m l / refDenom m l = softmaxWeightedAvg l := by
  rw [softmaxWeightedAvg_eq_zero, refNum_scale 0 m l, refDenom_scale 0 m l]
  exact mul_div_mul_right _ _ (Real.exp_ne_zero (0 - m))

/-- The quotient of any two references agrees. -/
theorem softmax_ref_independent (m₁ m₂ : ℝ) (l : List (ℝ × ℝ)) (hl : l ≠ []) :
    refNum m₁ l / refDenom m₁ l = refNum m₂ l / refDenom m₂ l := by
  rw [softmax_ref_invariant m₁ l hl, softmax_ref_invariant m₂ l hl]

/-- Reuse of `Nous.Combinator.ListHom.os_numerator` and `os_denominator`: the softmax-weighted
average is the imported online-softmax numerator over denominator. -/
theorem softmaxWeightedAvg_eq_os (l : List (ℝ × ℝ)) :
    softmaxWeightedAvg l =
      (Nous.Combinator.ListHom.optValNum (Nous.Combinator.ListHom.osFoldNum l)).2 /
        Nous.Combinator.ListHom.optVal (Nous.Combinator.ListHom.osFold (l.map Prod.fst)) := by
  simp only [softmaxWeightedAvg, Nous.Combinator.ListHom.os_numerator,
    Nous.Combinator.ListHom.os_denominator, List.map_map, Function.comp_def]

/-! ## Lazy step -/

/-- One FA4 lazy step on `(m_ref, s, n)` with threshold `τ`. If the incoming score satisfies
`x ≤ m_ref + τ`, the reference is kept and `e^{x − m_ref}` is accumulated; otherwise the state is
rescaled to `m_ref' = x` (arXiv 2603.05451, §3.1.4, equation (6)). -/
def lazyStep (τ : ℝ) (st : LazyState) (x v : ℝ) : LazyState :=
  if st.mref + τ < x then
    ⟨x, st.s * Real.exp (st.mref - x) + 1, st.n * Real.exp (st.mref - x) + v⟩
  else
    ⟨st.mref, st.s + Real.exp (x - st.mref), st.n + v * Real.exp (x - st.mref)⟩

/-- Fold the lazy step over a list of score-value pairs. -/
def lazyFold (τ : ℝ) (st : LazyState) : List (ℝ × ℝ) → LazyState
  | [] => st
  | sv :: rest => lazyFold τ (lazyStep τ st sv.1 sv.2) rest

/-- `Consistent st l`: `st.s` and `st.n` are the reference denominator and numerator of `l` at the
state's own reference. -/
def Consistent (st : LazyState) (l : List (ℝ × ℝ)) : Prop :=
  st.s = refDenom st.mref l ∧ st.n = refNum st.mref l

theorem consistent_init (m0 : ℝ) : Consistent (init m0) [] := by
  simp [Consistent, init, refDenom, refNum]

/-- One lazy step preserves consistency, appending the processed pair. -/
theorem lazyStep_consistent (τ : ℝ) (st : LazyState) (xs : List (ℝ × ℝ)) (x v : ℝ)
    (h : Consistent st xs) :
    Consistent (lazyStep τ st x v) (xs ++ [(x, v)]) := by
  rcases h with ⟨hs, hn⟩
  by_cases htrig : st.mref + τ < x
  · simp only [Consistent, lazyStep, if_pos htrig]
    constructor
    · rw [refDenom_append, refDenom_singleton, refDenom_scale st.mref x xs, hs, sub_self,
        Real.exp_zero]
    · rw [refNum_append, refNum_singleton, refNum_scale st.mref x xs, hn, sub_self,
        Real.exp_zero]
      ring
  · simp only [Consistent, lazyStep, if_neg htrig]
    constructor
    · rw [refDenom_append, refDenom_singleton, hs]
    · rw [refNum_append, refNum_singleton, hn]

/-- Any number of lazy steps preserves consistency with the concatenation of the segments. -/
theorem lazyFold_consistent (τ : ℝ) (st : LazyState) (xs : List (ℝ × ℝ)) (l : List (ℝ × ℝ))
    (h : Consistent st xs) :
    Consistent (lazyFold τ st l) (xs ++ l) := by
  induction l generalizing st xs with
  | nil =>
    have hnil : lazyFold τ st [] = st := rfl
    rw [hnil, List.append_nil]
    exact h
  | cons sv rest ih =>
    have hstep : Consistent (lazyStep τ st sv.1 sv.2) (xs ++ [sv]) :=
      lazyStep_consistent τ st xs sv.1 sv.2 h
    have hrec := ih (lazyStep τ st sv.1 sv.2) (xs ++ [sv]) hstep
    simpa only [lazyFold, List.append_assoc, List.singleton_append] using hrec

/-- The lazy fold from the neutral state is consistent with the whole list. -/
theorem lazyFold_correct (τ m0 : ℝ) (l : List (ℝ × ℝ)) :
    Consistent (lazyFold τ (init m0) l) l := by
  have h := lazyFold_consistent τ (init m0) [] l (consistent_init m0)
  simpa using h

/-- **Lazy exactness.** After any sequence of lazy steps from the neutral state, `n / s` equals the
softmax-weighted average. The threshold `τ` is arbitrary (in particular every `τ ≥ 0`): only the
stored reference changes, not the value. -/
theorem lazy_exact (τ m0 : ℝ) (l : List (ℝ × ℝ)) (hl : l ≠ []) :
    (lazyFold τ (init m0) l).n / (lazyFold τ (init m0) l).s = softmaxWeightedAvg l := by
  rcases lazyFold_correct τ m0 l with ⟨hs, hn⟩
  rw [hn, hs]
  exact softmax_ref_invariant _ l hl

/-! ## Bounded accumulated mass -/

/-- The reference is nondecreasing for `τ ≥ 0`. -/
theorem lazyStep_mref_le (τ : ℝ) (hτ : 0 ≤ τ) (st : LazyState) (x v : ℝ) :
    st.mref ≤ (lazyStep τ st x v).mref := by
  by_cases h : st.mref + τ < x
  · simp only [lazyStep, if_pos h]
    linarith
  · simp only [lazyStep, if_neg h]
    exact le_rfl

/-- The reference after any fold is at least the initial reference. -/
theorem lazyFold_mref_le (τ : ℝ) (hτ : 0 ≤ τ) (st : LazyState) (l : List (ℝ × ℝ)) :
    st.mref ≤ (lazyFold τ st l).mref := by
  induction l generalizing st with
  | nil => simp [lazyFold]
  | cons sv rest ih =>
    exact le_trans (lazyStep_mref_le τ hτ st sv.1 sv.2) (ih (lazyStep τ st sv.1 sv.2))

/-- If every score is within `τ` of the initial reference, then every accumulated term is
`≤ e^τ`. -/
theorem lazyFold_term_le (τ : ℝ) (hτ : 0 ≤ τ) (m0 : ℝ) (l : List (ℝ × ℝ))
    (hbound : ∀ x ∈ l.map Prod.fst, x ≤ m0 + τ) :
    ∀ x ∈ l.map Prod.fst, Real.exp (x - (lazyFold τ (init m0) l).mref) ≤ Real.exp τ := by
  intro x hx
  have hx' : x ≤ m0 + τ := hbound x hx
  have hmono : m0 ≤ (lazyFold τ (init m0) l).mref := lazyFold_mref_le τ hτ (init m0) l
  exact Real.exp_le_exp.mpr (by linarith)

/-- A sum of exponentials bounded by `e^c` is bounded by the count times `e^c`. -/
theorem sum_exp_le (l : List (ℝ × ℝ)) (m c : ℝ)
    (h : ∀ x ∈ l.map Prod.fst, x - m ≤ c) :
    (l.map fun sv => Real.exp (sv.1 - m)).sum ≤ (l.length : ℝ) * Real.exp c := by
  induction l with
  | nil => simp
  | cons sv rest ih =>
    rw [List.map_cons, List.sum_cons, List.length_cons]
    have hsv : sv.1 - m ≤ c := h sv.1 (by simp)
    have hrest : ∀ x ∈ rest.map Prod.fst, x - m ≤ c := by
      intro x hx
      exact h x (by simp [hx])
    have ih' := ih hrest
    have he : Real.exp (sv.1 - m) ≤ Real.exp c := Real.exp_le_exp.mpr hsv
    push_cast
    have hsplit : ((rest.length : ℝ) + 1) * Real.exp c
        = (rest.length : ℝ) * Real.exp c + Real.exp c := by ring
    rw [hsplit]
    linarith

/-- **Lazy boundedness.** If every score is within `τ` of the initial reference, every accumulated
term is `≤ e^τ` and the accumulated denominator is at most `count · e^τ`: lazy rescaling never
overflows by more than the threshold factor `e^τ`. -/
theorem lazy_bounded (τ : ℝ) (hτ : 0 ≤ τ) (m0 : ℝ) (l : List (ℝ × ℝ))
    (hbound : ∀ x ∈ l.map Prod.fst, x ≤ m0 + τ) :
    (lazyFold τ (init m0) l).s ≤ (l.length : ℝ) * Real.exp τ := by
  rcases lazyFold_correct τ m0 l with ⟨hs, _⟩
  rw [hs]
  exact sum_exp_le l (lazyFold τ (init m0) l).mref τ (by
    intro x hx
    have hx' : x ≤ m0 + τ := hbound x hx
    have hmono : m0 ≤ (lazyFold τ (init m0) l).mref := lazyFold_mref_le τ hτ (init m0) l
    linarith)

/-! ## Merging lazy states -/

/-- Merge two lazy states by rescaling both to the larger reference. This is the
`(m_ref, s, n)`-carrier form of `Nous.Combinator.ListHom.osFuseNum`; the state with the smaller
reference is rescaled. -/
def lazyMerge (a b : LazyState) : LazyState where
  mref := max a.mref b.mref
  s := a.s * Real.exp (a.mref - max a.mref b.mref) + b.s * Real.exp (b.mref - max a.mref b.mref)
  n := a.n * Real.exp (a.mref - max a.mref b.mref) + b.n * Real.exp (b.mref - max a.mref b.mref)

/-- Merging two consistent states is consistent with the concatenation. -/
theorem lazyMerge_consistent (a b : LazyState) (l₁ l₂ : List (ℝ × ℝ))
    (h₁ : Consistent a l₁) (h₂ : Consistent b l₂) :
    Consistent (lazyMerge a b) (l₁ ++ l₂) := by
  rcases h₁ with ⟨hs₁, hn₁⟩
  rcases h₂ with ⟨hs₂, hn₂⟩
  constructor
  · dsimp only [lazyMerge, Consistent]
    rw [refDenom_append, refDenom_scale a.mref (max a.mref b.mref) l₁,
      refDenom_scale b.mref (max a.mref b.mref) l₂, ← hs₁, ← hs₂]
  · dsimp only [lazyMerge, Consistent]
    rw [refNum_append, refNum_scale a.mref (max a.mref b.mref) l₁,
      refNum_scale b.mref (max a.mref b.mref) l₂, ← hn₁, ← hn₂]

/-- **Lazy merge exactness.** Merging two consistent lazy states is exact: the merged `n / s` is the
softmax-weighted average of the concatenated sequence. -/
theorem lazyMerge_exact (a b : LazyState) (l₁ l₂ : List (ℝ × ℝ))
    (h₁ : Consistent a l₁) (h₂ : Consistent b l₂) (h : l₁ ++ l₂ ≠ []) :
    (lazyMerge a b).n / (lazyMerge a b).s = softmaxWeightedAvg (l₁ ++ l₂) := by
  rcases lazyMerge_consistent a b l₁ l₂ h₁ h₂ with ⟨hs, hn⟩
  rw [hn, hs]
  exact softmax_ref_invariant _ _ h

/-- Evaluate a merge tree of lazy states: leaves fold a segment with threshold `τ`, nodes merge. -/
def evalLazyTree (τ : ℝ) (m0 : ℝ) : MTree (ℝ × ℝ) → LazyState
  | MTree.leaf l => lazyFold τ (init m0) l
  | MTree.node t₁ t₂ => lazyMerge (evalLazyTree τ m0 t₁) (evalLazyTree τ m0 t₂)

theorem evalLazyTree_consistent (τ m0 : ℝ) (t : MTree (ℝ × ℝ)) :
    Consistent (evalLazyTree τ m0 t) (flat t) := by
  induction t with
  | leaf l => exact lazyFold_correct τ m0 l
  | node t₁ t₂ ih₁ ih₂ =>
    dsimp only [evalLazyTree, flat]
    exact lazyMerge_consistent _ _ _ _ ih₁ ih₂

/-- **Tree merge exactness.** Any merge tree over the sequence agrees with the softmax-weighted
average of the flattened leaves. This is the lazy-rescaling counterpart of
`Nous.Combinator.ListHom.os_tree`. -/
theorem lazy_merge_tree_exact (τ m0 : ℝ) (t : MTree (ℝ × ℝ)) (h : flat t ≠ []) :
    (evalLazyTree τ m0 t).n / (evalLazyTree τ m0 t).s = softmaxWeightedAvg (flat t) := by
  rcases evalLazyTree_consistent τ m0 t with ⟨hs, hn⟩
  rw [hn, hs]
  exact softmax_ref_invariant _ _ h

/-! ## Rescale count -/

/-- A step of the rescale counter: `(m_ref, count)`. The count increments exactly when the incoming
score exceeds the reference by more than `τ`. -/
def rescaleStep (τ : ℝ) (st : ℝ × ℕ) (x : ℝ) : ℝ × ℕ :=
  if st.1 + τ < x then (x, st.2 + 1) else st

/-- Number of rescales triggered by `xs` starting from reference `x0`. -/
def rescaleCount (τ : ℝ) (x0 : ℝ) (xs : List ℝ) : ℕ :=
  (xs.foldl (rescaleStep τ) (x0, 0)).2

/-- Coupling a slack-`τ` counter with the exact (`τ = 0`) counter: the lazy reference stays at or
below the running max, within `τ` of it, and has no more rescales. At `τ = 0` the counter counts the
strict running-max rises, so no rescale is skipped there that the exact counter would count. -/
theorem rescaleStep_invariant (τ : ℝ) (hτ : 0 ≤ τ) (a b : ℝ × ℕ) (x : ℝ)
    (hle : a.1 ≤ b.1) (hbound : b.1 ≤ a.1 + τ) (hcnt : a.2 ≤ b.2) :
    (rescaleStep τ a x).1 ≤ (rescaleStep 0 b x).1 ∧
      (rescaleStep 0 b x).1 ≤ (rescaleStep τ a x).1 + τ ∧
      (rescaleStep τ a x).2 ≤ (rescaleStep 0 b x).2 := by
  by_cases h1 : a.1 + τ < x
  · by_cases h2 : b.1 + 0 < x
    · simp only [rescaleStep, if_pos h1, if_pos h2]
      exact ⟨le_rfl, by linarith, by omega⟩
    · exfalso
      have h2' : x ≤ b.1 := by linarith
      linarith
  · by_cases h2 : b.1 + 0 < x
    · simp only [rescaleStep, if_neg h1, if_pos h2]
      have h2' : b.1 < x := by linarith
      exact ⟨by linarith, by linarith, by omega⟩
    · simp only [rescaleStep, if_neg h1, if_neg h2]
      exact ⟨hle, hbound, hcnt⟩

/-- The count inequality follows by folding the step invariant. -/
theorem rescaleCount_le_fold (τ : ℝ) (hτ : 0 ≤ τ) (xs : List ℝ) (a b : ℝ × ℕ)
    (hle : a.1 ≤ b.1) (hbound : b.1 ≤ a.1 + τ) (hcnt : a.2 ≤ b.2) :
    (xs.foldl (rescaleStep τ) a).2 ≤ (xs.foldl (rescaleStep 0) b).2 := by
  induction xs generalizing a b with
  | nil => exact hcnt
  | cons x rest ih =>
    have hinv := rescaleStep_invariant τ hτ a b x hle hbound hcnt
    dsimp only [List.foldl]
    exact ih _ _ hinv.1 hinv.2.1 hinv.2.2

/-- **Rescale count bound.** With slack `τ ≥ 0`, the number of rescales is at most the number at
`τ = 0`, i.e. at most the number of strict running-max rises. -/
theorem rescale_count_le (τ : ℝ) (hτ : 0 ≤ τ) (x0 : ℝ) (xs : List ℝ) :
    rescaleCount τ x0 xs ≤ rescaleCount 0 x0 xs := by
  dsimp only [rescaleCount]
  exact rescaleCount_le_fold τ hτ xs (x0, 0) (x0, 0) le_rfl (by linarith) le_rfl

/-! ## Witnesses -/

namespace Witness

/-- Two score-value pairs used by the lazy witnesses. -/
def demo : List (ℝ × ℝ) := [(1, 1), (3, 2)]

/-- Non-vacuity of `softmax_ref_invariant`'s and `lazy_exact`'s nonemptiness hypothesis. -/
theorem demo_ne_nil : demo ≠ [] := by simp [demo]

/-- Non-vacuity of `softmax_ref_invariant` at the reference `0`. -/
example : refNum 0 demo / refDenom 0 demo = softmaxWeightedAvg demo :=
  softmax_ref_invariant 0 demo demo_ne_nil

/-- Independence of the reference is non-vacuous. -/
example : refNum 0 demo / refDenom 0 demo = refNum 5 demo / refDenom 5 demo :=
  softmax_ref_independent 0 5 demo demo_ne_nil

/-- Non-vacuity of `lazy_exact` (`τ = 1`, reference `0`). -/
example :
    (lazyFold 1 (init 0) demo).n / (lazyFold 1 (init 0) demo).s = softmaxWeightedAvg demo :=
  lazy_exact 1 0 demo demo_ne_nil

/-- Non-vacuity of `lazy_bounded`: with `τ = 1`, reference `0` all scores of `demo` are `> 1`, so
the hypothesis fails there; with reference `2` and `τ = 1` every score is `≤ 3`. -/
example : (lazyFold 1 (init 2) demo).s ≤ (demo.length : ℝ) * Real.exp 1 :=
  lazy_bounded 1 (by norm_num) 2 demo (by
    intro x hx
    simp [demo] at hx
    rcases hx with rfl | rfl <;> norm_num)

/-- Term-wise form of the bound at reference `2`. -/
example (x : ℝ) (hx : x ∈ demo.map Prod.fst) :
    Real.exp (x - (lazyFold 1 (init 2) demo).mref) ≤ Real.exp 1 :=
  lazyFold_term_le 1 (by norm_num) 2 demo (by
    intro y hy
    simp [demo] at hy
    rcases hy with rfl | rfl <;> norm_num) x hx

/-- Non-vacuity of `lazyMerge_exact`: two consistent leaves merge exactly. -/
example :
    (lazyMerge (lazyFold 1 (init 0) [(1, 1)]) (lazyFold 1 (init 0) [(3, 2)])).n /
        (lazyMerge (lazyFold 1 (init 0) [(1, 1)]) (lazyFold 1 (init 0) [(3, 2)])).s
      = softmaxWeightedAvg ([(1, 1)] ++ [(3, 2)]) :=
  lazyMerge_exact _ _ _ _ (lazyFold_correct 1 0 [(1, 1)]) (lazyFold_correct 1 0 [(3, 2)])
    (by simp)

/-- The tree witness over a two-leaf node. -/
def demoT : MTree (ℝ × ℝ) := MTree.node (MTree.leaf [(1, 1)]) (MTree.leaf [(3, 2)])

/-- Non-vacuity of `lazy_merge_tree_exact`. -/
example :
    (evalLazyTree 1 0 demoT).n / (evalLazyTree 1 0 demoT).s = softmaxWeightedAvg (flat demoT) :=
  lazy_merge_tree_exact 1 0 demoT (by simp [demoT, flat])

/-- Zero rescales at `τ = 1`: the score `0.5` is within the reference plus slack. -/
example : rescaleCount 1 0 [0.5] = 0 := by
  norm_num [rescaleCount, rescaleStep]

/-- One rescale at `τ = 0`: the score `0.5` exceeds the exact running max. -/
example : rescaleCount 0 0 [0.5] = 1 := by
  norm_num [rescaleCount, rescaleStep]

/-- Strictness: the slack count is strictly below the exact count on this sequence. -/
example : rescaleCount 1 0 [0.5] < rescaleCount 0 0 [0.5] := by
  norm_num [rescaleCount, rescaleStep]

/-- Non-vacuity of `rescale_count_le`'s `0 ≤ τ` hypothesis. -/
example : rescaleCount 1 0 [0.5] ≤ rescaleCount 0 0 [0.5] :=
  rescale_count_le 1 (by norm_num) 0 [0.5]

end Witness

end

end Nous.Seq.LazySoftmax
