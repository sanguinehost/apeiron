/-
# One attention head as a `Stage`

`Nous/Model/BlockComposition.lean` carries the attention sub-block as the fully open hypothesis
`BlockResidual.attn` (`BlockComposition.lean:371`): a `Stage` from the fanned-out `(Q, K, V)`
projections to the pre-output-matvec attention result, with no reference formula pinned. This module
pins the *single-head, fixed-KV-state* case:

* `attnRef` — the reference map, from a real query vector `y : Fin head_dim → ℝ` to the ideal
  attention output component `i` = `denseAttnReal (listAbsMax sc) sc (toRealList vs_i)`
  (`Numeric.EngineAttnRefinement`), where the score list `sc` is the real `<key_j, y> / √head_dim`
  (`Numeric.EngineAttnRefinement.idealScoreList`); `attnScoreList_gridVec` proves the two agree on
  `gridVec q` (`Numeric/Matvec/Bridge.lean`) under `q.val.length = head_dim`.
* `attnRef_lipschitz` — the global expansion bound of `attnRef`, with constant `attnStageL`.
* `attnStageDom` — the admissible domain: the `gridVec` images of the `attn_domain_v2` slices with a
  component bound `B`.
* the `attnErrBoundV2`/`scoreErrV2` nonnegativity and monotonicity lemmas that a consumer of the
  stage needs.

## Dropped with the attention-downstream move

`Numeric.AttnHeadRefines`, `Numeric.AttnHeadWindowsScore`, `Numeric.AttnHeadWindowsValue`,
`Numeric.AttnLoopScore` and `Numeric.AttnLoopOutput` moved to
`work/wip/attn-downstream/` (`attn.score_at` and `attn.value_component` are no longer extraction
targets). The executed half of this module rested on `attn_head_refines_v2_proved` from
`Numeric.AttnHeadRefines`, so `attnOut`, `attnVec`, `attnExec`, `attnOut_spec`, `attnVec_bound`,
`attnHeadStage` and `attnHeadStage_dom_witness` are dropped with it. `attn_head_refines_v2` itself
remains an `anchor` in `Numeric.AttnSoftmaxPerturb` and is **not** proved in the corpus. The
reference map, its Lipschitz constant, the domain predicate and the error-bound monotonicity
lemmas stand on `Numeric.AttnSoftmaxPerturb` and `Numeric.AttnScoreStage` alone.
`Nous/Model/AttnStageWitness.lean` consumed only `attnHeadStage`; it is left dangling.

## The global expansion constant

`denseAttnReal_lip` (proved): the dense softmax attention of a fixed value list is Lipschitz in its
scores with constant `2·n·(hi−lo)` in the score sup norm. The proof shifts each score list by its own
maximum (the mean is shift-invariant, `denseAttnReal_shift_invariant`), so both shifted scores are
`≤ 0`, where `|e^a − e^b| ≤ |a − b|` (`abs_exp_sub_exp_le_of_nonpos`), and applies
`reweight_mean_perturb` (`Numeric.AttnSoftmaxPerturb.lean:112`) with a denominator `≥ 1`. The score
map `y ↦ <key_j, y>/√head_dim` is linear with constant `R/√head_dim`, so the composed expansion is
`L = 4·(pos+1)·V·R/√head_dim`, stated explicitly as `attnStageL`. `attnRef_lipschitz` proves the
global bound; no domain restriction is needed.

## Trust boundary

Kernel-checked `ℝ`/`ℤ` algebra plus the `denseAttnReal`/`idealScoreList` denotations of
`Numeric.Engine.Attn`. No claim here reaches the extracted `word_attn_head` body any more;
see "Dropped with the attention-downstream move" above. Does not cover rustc/LLVM codegen
of `src/engine/attn.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container. The `B` conjunct of
the domain is a restriction carried for the caller's uniform `δ` bound; it is not used by the
Lipschitz proof.
-/
import Nous.Model.BlockComposition
import Numeric.Attn.SoftmaxPerturb
import Numeric.Attn.ScoreStage
import Numeric.Matvec.Bridge

namespace Nous.Model

open apeiron_word Aeneas Aeneas.Std Result
open MatvecBridge
open Classical

noncomputable section

/-! ## The exponential Lipschitz bound on the nonpositive half-line -/

/-- On `a ≤ b ≤ 0` the exponential has slope at most `1`: `e^b − e^a ≤ b − a`. -/
theorem exp_sub_exp_le_of_le_nonpos {a b : ℝ} (hab : a ≤ b) (hb : b ≤ 0) :
    Real.exp b - Real.exp a ≤ b - a := by
  have h1 : Real.exp b - Real.exp a = Real.exp b * (1 - Real.exp (a - b)) := by
    rw [mul_sub, mul_one, ← Real.exp_add]
    congr 1
    ring
  rw [h1]
  have hb1 : Real.exp b ≤ 1 := Real.exp_le_one_iff.mpr hb
  have hnonneg : 0 ≤ 1 - Real.exp (a - b) := by
    have hab' : a - b ≤ 0 := by linarith
    have := Real.exp_le_one_iff.mpr hab'
    linarith
  have h3 : 1 - Real.exp (a - b) ≤ b - a := by
    have h4 : 1 - Real.exp (-(b - a)) ≤ b - a := by
      have := Real.add_one_le_exp (-(b - a))
      linarith
    have heq : a - b = -(b - a) := by ring
    rw [heq]; exact h4
  calc Real.exp b * (1 - Real.exp (a - b)) ≤ 1 * (1 - Real.exp (a - b)) :=
        mul_le_mul_of_nonneg_right hb1 hnonneg
    _ ≤ b - a := by rw [one_mul]; exact h3

/-- Absolute form: `|e^a − e^b| ≤ |a − b|` for `a, b ≤ 0`. -/
theorem abs_exp_sub_exp_le_of_nonpos {a b : ℝ} (ha : a ≤ 0) (hb : b ≤ 0) :
    |Real.exp a - Real.exp b| ≤ |a - b| := by
  rcases le_total a b with h | h
  · have h1 : Real.exp a - Real.exp b ≤ 0 := sub_nonpos.mpr (Real.exp_le_exp.mpr h)
    have h2 : a - b ≤ 0 := sub_nonpos.mpr h
    rw [abs_of_nonpos h1, abs_of_nonpos h2]
    have := exp_sub_exp_le_of_le_nonpos h hb
    linarith
  · have h1 : 0 ≤ Real.exp a - Real.exp b := sub_nonneg.mpr (Real.exp_le_exp.mpr h)
    have h2 : 0 ≤ a - b := sub_nonneg.mpr h
    rw [abs_of_nonneg h1, abs_of_nonneg h2]
    have := exp_sub_exp_le_of_le_nonpos h ha
    linarith

/-! ## The dense softmax attention is Lipschitz in the scores -/

/-- The sum of a mapped list divided by a constant is the sum of the divided list. -/
theorem list_sum_map_div (l : List ℝ) (c : ℝ) : (l.map (fun x => x / c)).sum = l.sum / c := by
  induction l with
  | nil => simp
  | cons a t ih => simp only [List.map_cons, List.sum_cons, ih, add_div]

/-- A list is the `getD` window of its own range. -/
private theorem map_getD_range_eq {α : Type*} (l : List α) (d : α) :
    (List.range l.length).map (fun j => l.getD j d) = l := by
  apply List.ext_getElem
  · simp [List.length_map]
  · intro k h1 h2
    rw [List.getElem_map, List.getElem_range (by simpa using h1), List.getD_eq_getElem l d h2]

/-- **Softmax-attention Lipschitz.** For values in `[lo, hi]`, the dense softmax mean of the same
value list is Lipschitz in the scores with constant `2·n·(hi − lo)` in the score sup norm. Both sides
are shifted by their own score maximum, so the shifted scores are `≤ 0` and the exponential is
`1`-Lipschitz there; `reweight_mean_perturb` closes the bound with denominator `≥ 1`. -/
theorem denseAttnReal_lip (n : ℕ) (x x' v : ℕ → ℝ) (lo hi d : ℝ)
    (hn : 1 ≤ n) (hv : ∀ j < n, lo ≤ v j ∧ v j ≤ hi) (hd0 : 0 ≤ d)
    (hb : ∀ j < n, |x j - x' j| ≤ d) :
    |denseAttnReal 0 ((List.range n).map x) ((List.range n).map v)
        - denseAttnReal 0 ((List.range n).map x') ((List.range n).map v)|
      ≤ 2 * (n : ℝ) * (hi - lo) * d := by
  have hn0 : 0 < n := by omega
  have hne : (Finset.range n).Nonempty := ⟨0, Finset.mem_range.mpr hn0⟩
  set M := (Finset.range n).sup' hne x
  set M' := (Finset.range n).sup' hne x'
  obtain ⟨j0, hj0, hj0eq⟩ := Finset.exists_mem_eq_sup' hne x
  obtain ⟨j1, hj1, hj1eq⟩ := Finset.exists_mem_eq_sup' hne x'
  have hxM : ∀ j ∈ Finset.range n, x j ≤ M :=
    fun j hj => (Finset.le_sup'_iff hne).mpr ⟨j, hj, le_rfl⟩
  have hxM' : ∀ j ∈ Finset.range n, x' j ≤ M' :=
    fun j hj => (Finset.le_sup'_iff hne).mpr ⟨j, hj, le_rfl⟩
  have hMM' : |M - M'| ≤ d := by
    have h1 : M - M' ≤ d := by
      have hb0 := hb j0 (Finset.mem_range.mp hj0)
      rw [abs_le] at hb0
      have h2 : x' j0 ≤ M' := hxM' j0 hj0
      have h3 : M = x j0 := hj0eq
      linarith [hb0.2]
    have h2 : M' - M ≤ d := by
      have hb1 := hb j1 (Finset.mem_range.mp hj1)
      rw [abs_le] at hb1
      have h3 : x j1 ≤ M := hxM j1 hj1
      have h4 : M' = x' j1 := hj1eq
      linarith [hb1.1]
    rw [abs_le]
    exact ⟨by linarith, h1⟩
  have hshift : ∀ j ∈ Finset.range n, |(x j - M) - (x' j - M')| ≤ 2 * d := by
    intro j hj
    have h := hb j (Finset.mem_range.mp hj)
    have heq : x j - M - (x' j - M') = (x j - x' j) - (M - M') := by ring
    have htri : |(x j - x' j) - (M - M')| ≤ |x j - x' j| + |M - M'| := by
      rw [sub_eq_add_neg, ← abs_neg (M - M')]
      exact abs_add_le _ _
    rw [heq]
    linarith
  have hnonpos : ∀ j ∈ Finset.range n, x j - M ≤ 0 := fun j hj => by linarith [hxM j hj]
  have hnonpos' : ∀ j ∈ Finset.range n, x' j - M' ≤ 0 := fun j hj => by linarith [hxM' j hj]
  have hexp_le : ∀ j ∈ Finset.range n,
      |Real.exp (x j - M) - Real.exp (x' j - M')| ≤ 2 * d := fun j hj =>
    le_trans (abs_exp_sub_exp_le_of_nonpos (hnonpos j hj) (hnonpos' j hj)) (hshift j hj)
  have hsumpos : 0 < ∑ j ∈ Finset.range n, Real.exp (x j - M) := by
    have hle := Finset.single_le_sum (f := fun j => Real.exp (x j - M))
      (fun j _ => (Real.exp_pos _).le) hj0
    have h0 : Real.exp (x j0 - M) = 1 := by rw [sub_eq_zero.mpr hj0eq.symm]; simp
    rw [h0] at hle
    linarith
  have hsumpos' : 0 < ∑ j ∈ Finset.range n, Real.exp (x' j - M') := by
    have hle := Finset.single_le_sum (f := fun j => Real.exp (x' j - M'))
      (fun j _ => (Real.exp_pos _).le) hj1
    have h0 : Real.exp (x' j1 - M') = 1 := by rw [sub_eq_zero.mpr hj1eq.symm]; simp
    rw [h0] at hle
    linarith
  have hden : 1 ≤ ∑ j ∈ Finset.range n, Real.exp (x j - M) := by
    have hle := Finset.single_le_sum (f := fun j => Real.exp (x j - M))
      (fun j _ => (Real.exp_pos _).le) hj0
    have h0 : Real.exp (x j0 - M) = 1 := by rw [sub_eq_zero.mpr hj0eq.symm]; simp
    rwa [h0] at hle
  have hnum : ∑ j ∈ Finset.range n, |Real.exp (x j - M) - Real.exp (x' j - M')|
      ≤ (n : ℝ) * (2 * d) := by
    calc ∑ j ∈ Finset.range n, |Real.exp (x j - M) - Real.exp (x' j - M')|
        ≤ ∑ _j ∈ Finset.range n, (2 * d) := Finset.sum_le_sum hexp_le
      _ = (n : ℝ) * (2 * d) := by rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  have hlohi : 0 ≤ hi - lo := by
    have h0 := hv 0 hn0
    linarith
  have hrepr : denseAttnReal 0 ((List.range n).map x) ((List.range n).map v)
      = (∑ j ∈ Finset.range n, Real.exp (x j - M) * v j)
        / (∑ j ∈ Finset.range n, Real.exp (x j - M)) := by
    rw [denseAttnReal_shift_invariant 0 M]
    exact denseAttnReal_range M n x v
  have hrepr' : denseAttnReal 0 ((List.range n).map x') ((List.range n).map v)
      = (∑ j ∈ Finset.range n, Real.exp (x' j - M') * v j)
        / (∑ j ∈ Finset.range n, Real.exp (x' j - M')) := by
    rw [denseAttnReal_shift_invariant 0 M']
    exact denseAttnReal_range M' n x' v
  have hrw := reweight_mean_perturb (Finset.range n) (fun j => Real.exp (x' j - M'))
    (fun j => Real.exp (x j - M)) v lo hi (fun j _ => (Real.exp_pos _).le) hsumpos' hsumpos
    (fun j hj => hv j (Finset.mem_range.mp hj))
  rw [hrepr, hrepr']
  refine le_trans hrw ?_
  have hstep1 : (hi - lo) * (∑ j ∈ Finset.range n, |Real.exp (x j - M) - Real.exp (x' j - M')|)
      / (∑ j ∈ Finset.range n, Real.exp (x j - M))
      ≤ (hi - lo) * ((n : ℝ) * (2 * d)) / (∑ j ∈ Finset.range n, Real.exp (x j - M)) :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hnum hlohi) (le_of_lt hsumpos)
  refine le_trans hstep1 ?_
  rw [div_le_iff₀ hsumpos]
  have hA : 0 ≤ (hi - lo) * ((n : ℝ) * (2 * d)) :=
    mul_nonneg hlohi (mul_nonneg (Nat.cast_nonneg n) (by linarith))
  have hkey : (hi - lo) * ((n : ℝ) * (2 * d))
      ≤ (hi - lo) * ((n : ℝ) * (2 * d))
        * (∑ j ∈ Finset.range n, Real.exp (x j - M)) := by
    have := mul_le_mul_of_nonneg_left hden hA
    rwa [mul_one] at this
  have heq : 2 * (n : ℝ) * (hi - lo) * d * (∑ j ∈ Finset.range n, Real.exp (x j - M))
      = (hi - lo) * ((n : ℝ) * (2 * d))
        * (∑ j ∈ Finset.range n, Real.exp (x j - M)) := by ring
  rw [heq]
  exact hkey

/-- List form of `denseAttnReal_lip`: on equal-length lists the `listAbsMax` shift cancels. -/
theorem denseAttnReal_lip_list (n : ℕ) (sc sc' vs : List ℝ) (lo hi d : ℝ)
    (hn : 1 ≤ n) (hsc : sc.length = n) (hsc' : sc'.length = n) (hvs : vs.length = n)
    (hv : ∀ x ∈ vs, lo ≤ x ∧ x ≤ hi) (hd0 : 0 ≤ d)
    (hb : ∀ j < n, |sc.getD j 0 - sc'.getD j 0| ≤ d) :
    |denseAttnReal (listAbsMax sc) sc vs - denseAttnReal (listAbsMax sc') sc' vs|
      ≤ 2 * (n : ℝ) * (hi - lo) * d := by
  rw [denseAttnReal_shift_invariant (listAbsMax sc) 0 sc vs,
    denseAttnReal_shift_invariant (listAbsMax sc') 0 sc' vs]
  have hscm : sc = (List.range n).map (fun j => sc.getD j 0) := by
    rw [← hsc]; exact (map_getD_range_eq sc 0).symm
  have hscm' : sc' = (List.range n).map (fun j => sc'.getD j 0) := by
    rw [← hsc']; exact (map_getD_range_eq sc' 0).symm
  have hvsm : vs = (List.range n).map (fun j => vs.getD j 0) := by
    rw [← hvs]; exact (map_getD_range_eq vs 0).symm
  rw [hscm, hscm', hvsm]
  exact denseAttnReal_lip n (fun j => sc.getD j 0) (fun j => sc'.getD j 0)
    (fun j => vs.getD j 0) lo hi d hn
    (fun j hj => hv _ (by
      rw [List.getD_eq_getElem vs 0 (by rw [hvs]; exact hj)]
      exact List.getElem_mem _)) hd0 hb

/-! ## The single-head score function on real query vectors -/

/-- The KB head index of the query head. -/
abbrev kvhOf (state : attn.KvState) (q_head : Usize) : ℕ :=
  q_head.val / (state.n_head.val / state.n_head_kv.val)

/-- The ideal real score `<key_s, y> / √head_dim` of one causal key against a real query vector. -/
def keyRowScore {hd : ℕ} (kp : List I64) (kv kvh s : ℕ) (hdim : ℝ) (y : Fin hd → ℝ) : ℝ :=
  (∑ i : Fin hd, toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64) * y i) / Real.sqrt hdim

/-- The ideal score list of the causal prefix against a real query vector. -/
def attnScoreList {hd : ℕ} (kp : List I64) (kv kvh pos : ℕ) (hdim : ℝ)
    (y : Fin hd → ℝ) : List ℝ :=
  (List.range (pos + 1)).map (fun s => keyRowScore kp kv kvh s hdim y)

/-- The integer dot product and the `toReal` dot product agree over `2^64`. -/
theorem dot_bridge (w x : List I64) (n : ℕ) (hw : w.length = n) (hx : x.length = n) :
    ((w.zip x).map (fun p => (p.1.val : ℝ) * (p.2.val : ℝ))).sum / q32 ^ 2
      = ∑ j : Fin n, toReal (w.getD j.val 0#i64) * toReal (x.getD j.val 0#i64) := by
  have hmap : (fun p : I64 × I64 => (p.1.val : ℝ) * (p.2.val : ℝ) / q32 ^ 2)
      = fun p : I64 × I64 => toReal p.1 * toReal p.2 := by
    funext p
    unfold toReal
    field_simp
  have h1 : ((w.zip x).map (fun p => (p.1.val : ℝ) * (p.2.val : ℝ))).sum / q32 ^ 2
      = ((w.zip x).map (fun p => toReal p.1 * toReal p.2)).sum := by
    rw [← list_sum_map_div, List.map_map]
    change ((w.zip x).map (fun p => (p.1.val : ℝ) * (p.2.val : ℝ) / q32 ^ 2)).sum
      = ((w.zip x).map (fun p => toReal p.1 * toReal p.2)).sum
    rw [hmap]
  rw [h1]
  exact MatvecBridge.list_dot_bridge w x n hw hx

/-- The `s`-th `attnScoreList` entry is the `s`-th `keyRowScore`. -/
theorem attnScoreList_getD {hd : ℕ} (kp : List I64) (kv kvh pos : ℕ) (hdim : ℝ)
    (y : Fin hd → ℝ) (j : ℕ) (hj : j < pos + 1) :
    (attnScoreList (hd := hd) kp kv kvh pos hdim y).getD j 0 = keyRowScore kp kv kvh j hdim y := by
  rw [List.getD_eq_getElem _ 0 (by simp [attnScoreList]; omega)]
  simp [attnScoreList]

/-- **Score-list bridge.** On `gridVec q` with `q` filling the window, the real score list is
`idealScoreList`. -/
theorem attnScoreList_gridVec {hd : ℕ} (kp : List I64) (q : Slice I64) (kv kvh pos : ℕ)
    (hdim : ℝ) (hq : q.val.length = hd) :
    attnScoreList (hd := hd) kp kv kvh pos hdim (MatvecBridge.gridVec (n := hd) q)
      = idealScoreList kp q.val kv hd kvh hdim pos := by
  unfold attnScoreList idealScoreList
  apply List.map_congr_left
  intro s _
  simp only [keyRowScore, MatvecBridge.gridVec]
  rw [dot_bridge (planeRow kp kv hd kvh s) q.val hd (by simp [planeRow]) hq]

/-! ## The key block bound and the expansion constant -/

/-- The max row `ℓ1` norm of the key block over the causal prefix. -/
def keyBlockBound (kp : List I64) (kv hd kvh pos : ℕ) : ℝ :=
  (Finset.range (pos + 1)).sup' ⟨0, Finset.mem_range.mpr (Nat.succ_pos pos)⟩
    (fun s => ∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64)|)

theorem keyBlockBound_nonneg (kp : List I64) (kv hd kvh pos : ℕ) :
    0 ≤ keyBlockBound kp kv hd kvh pos := by
  unfold keyBlockBound
  exact le_trans (Finset.sum_nonneg (fun i _ => abs_nonneg _))
    ((Finset.le_sup'_iff _).mpr ⟨0, Finset.mem_range.mpr (Nat.succ_pos pos), le_rfl⟩)

/-- Every causal key row `ℓ1` norm is below the block bound. -/
theorem keyRowBound_le (kp : List I64) (kv hd kvh pos s : ℕ) (hs : s ≤ pos) :
    (∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64)|)
      ≤ keyBlockBound kp kv hd kvh pos := by
  unfold keyBlockBound
  exact (Finset.le_sup'_iff _).mpr ⟨s, Finset.mem_range.mpr (by omega), le_rfl⟩

/-- **The explicit expansion constant.** `L = 4·(pos+1)·V·R/√head_dim`, where `R` is the max row
`ℓ1` norm of the key block and `V` the value bound: `2·(hi−lo)` from `denseAttnReal_lip` at `hi = V`,
`lo = −V`, times `2` from the score shift, times `R/√head_dim`. -/
def attnStageL (pos : ℕ) (V R hdim : ℝ) : ℝ :=
  4 * ((pos + 1 : ℕ) : ℝ) * V * R / Real.sqrt hdim

theorem attnStageL_nonneg (pos : ℕ) {V R hdim : ℝ} (hV : 0 ≤ V) (hR : 0 ≤ R) (hh : 0 < hdim) :
    0 ≤ attnStageL pos V R hdim := by
  unfold attnStageL
  apply div_nonneg _ (le_of_lt (Real.sqrt_pos.mpr hh))
  exact mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) (Nat.cast_nonneg _)) hV) hR

/-! ## The score map is Lipschitz -/

/-- Each key row score is Lipschitz in the query with constant the row `ℓ1` norm over `√head_dim`. -/
theorem keyRowScore_lipschitz {hd : ℕ} (kp : List I64) (kv kvh s : ℕ) {hdim : ℝ}
    (y y' : Fin hd → ℝ) :
    |keyRowScore kp kv kvh s hdim y - keyRowScore kp kv kvh s hdim y'|
      ≤ (∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64)|)
          * ‖y - y'‖ / Real.sqrt hdim := by
  unfold keyRowScore
  have hsplit : (∑ i : Fin hd, toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64) * y i)
        / Real.sqrt hdim
      - (∑ i : Fin hd, toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64) * y' i)
        / Real.sqrt hdim
      = (∑ i : Fin hd, toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64) * (y i - y' i))
        / Real.sqrt hdim := by
    rw [div_sub_div_same]
    congr 1
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl (fun i _ => by ring)
  rw [hsplit, abs_div, abs_of_nonneg (Real.sqrt_nonneg hdim)]
  apply div_le_div_of_nonneg_right _ (Real.sqrt_nonneg hdim)
  calc |∑ i : Fin hd, toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64) * (y i - y' i)|
      ≤ ∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64) * (y i - y' i)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ = ∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64)| * |y i - y' i| :=
        Finset.sum_congr rfl (fun i _ => abs_mul _ _)
    _ ≤ ∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64)| * ‖y - y'‖ := by
        apply Finset.sum_le_sum
        intro i _
        apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
        simpa only [Pi.sub_apply, ← Real.norm_eq_abs] using norm_le_pi_norm (y - y') i
    _ = (∑ i : Fin hd, |toReal ((planeRow kp kv hd kvh s).getD i.val 0#i64)|) * ‖y - y'‖ := by
        rw [Finset.sum_mul]

/-! ## The reference map and its global expansion -/

/-- The single-head reference: the ideal attention output component `i` of the real query vector
`y` — the dense softmax attention of the ideal score list against the value column. -/
def attnRef (state : attn.KvState) (q_head pos : Usize) (kplane vplane : alloc.vec.Vec I64)
    (hd : ℕ) (y : Fin hd → ℝ) : Fin hd → ℝ :=
  fun i => denseAttnReal
    (listAbsMax (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head)
      pos.val (hd : ℝ) y))
    (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head) pos.val (hd : ℝ) y)
    (toRealList (valueList vplane.val state.n_head_kv.val hd (kvhOf state q_head) pos.val i.val))

/-- `toRealList` preserves length. -/
theorem toRealList_length (l : List ℤ) : (toRealList l).length = l.length := by
  simp [toRealList]

/-- A value column of the causal prefix has `pos + 1` entries. -/
theorem valueList_length (plane : List I64) (kv hd kvh pos i : ℕ) :
    (valueList plane kv hd kvh pos i).length = pos + 1 := by
  simp [valueList]

/-- **Global expansion.** `attnRef` is Lipschitz in the query with the explicit constant
`attnStageL`: the scores move by `R/√head_dim` per unit query (sup norm), and the dense softmax
attention by `2·n·(2V)` per unit score. -/
theorem attnRef_lipschitz (state : attn.KvState) (q_head pos : Usize)
    (kplane vplane : alloc.vec.Vec I64) (hd : ℕ) (hh : 0 < hd) (V : ℝ) (hV : 0 ≤ V)
    (hVb : ∀ i : Fin hd, listAbsMax (toRealList (valueList vplane.val state.n_head_kv.val hd
      (kvhOf state q_head) pos.val i.val)) ≤ V)
    (z w : Fin hd → ℝ) :
    ‖attnRef state q_head pos kplane vplane hd z - attnRef state q_head pos kplane vplane hd w‖
      ≤ attnStageL pos.val V
          (keyBlockBound kplane.val state.n_head_kv.val hd (kvhOf state q_head) pos.val)
          (hd : ℝ) * ‖z - w‖ := by
  have hhR : 0 < (hd : ℝ) := by exact_mod_cast hh
  apply (pi_norm_le_iff_of_nonneg (mul_nonneg
    (attnStageL_nonneg pos.val hV
      (keyBlockBound_nonneg kplane.val state.n_head_kv.val hd (kvhOf state q_head) pos.val) hhR)
    (norm_nonneg _))).2
  intro i
  set R := keyBlockBound kplane.val state.n_head_kv.val hd (kvhOf state q_head) pos.val
  set d : ℝ := R * ‖z - w‖ / Real.sqrt (hd : ℝ) with hddef
  have hd0 : 0 ≤ d := by
    rw [hddef]
    exact div_nonneg (mul_nonneg (keyBlockBound_nonneg _ _ _ _ _) (norm_nonneg _))
      (Real.sqrt_nonneg _)
  have hsc : (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head)
      pos.val (hd : ℝ) z).length = pos.val + 1 := by simp [attnScoreList]
  have hsc' : (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head)
      pos.val (hd : ℝ) w).length = pos.val + 1 := by simp [attnScoreList]
  have hvs : (toRealList (valueList vplane.val state.n_head_kv.val hd (kvhOf state q_head)
      pos.val i.val)).length = pos.val + 1 := by
    rw [toRealList_length, valueList_length]
  have hv : ∀ x ∈ toRealList (valueList vplane.val state.n_head_kv.val hd (kvhOf state q_head)
      pos.val i.val), -V ≤ x ∧ x ≤ V := by
    intro x hx
    have h1 : |x| ≤ V := le_trans (listAbsMax_ge _ x hx) (hVb i)
    rw [abs_le] at h1
    exact h1
  have hb : ∀ j < pos.val + 1,
      |(attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head) pos.val
            (hd : ℝ) z).getD j 0
        - (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head) pos.val
            (hd : ℝ) w).getD j 0| ≤ d := by
    intro j hj
    rw [attnScoreList_getD _ _ _ _ _ z j hj, attnScoreList_getD _ _ _ _ _ w j hj, hddef]
    calc |keyRowScore kplane.val state.n_head_kv.val (kvhOf state q_head) j (hd : ℝ) z
          - keyRowScore kplane.val state.n_head_kv.val (kvhOf state q_head) j (hd : ℝ) w|
        ≤ (∑ i : Fin hd, |toReal ((planeRow kplane.val state.n_head_kv.val hd (kvhOf state q_head)
              j).getD i.val 0#i64)|) * ‖z - w‖ / Real.sqrt (hd : ℝ) :=
          keyRowScore_lipschitz _ _ _ _ z w
      _ ≤ R * ‖z - w‖ / Real.sqrt (hd : ℝ) := by
          apply div_le_div_of_nonneg_right _ (Real.sqrt_nonneg _)
          exact mul_le_mul_of_nonneg_right
            (keyRowBound_le kplane.val state.n_head_kv.val hd (kvhOf state q_head) pos.val j
              (by omega)) (norm_nonneg _)
  have hmain := denseAttnReal_lip_list (pos.val + 1)
    (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head) pos.val
      (hd : ℝ) z)
    (attnScoreList (hd := hd) kplane.val state.n_head_kv.val (kvhOf state q_head) pos.val
      (hd : ℝ) w)
    (toRealList (valueList vplane.val state.n_head_kv.val hd (kvhOf state q_head) pos.val i.val))
    (-V) V d (by omega) hsc hsc' hvs hv hd0 hb
  have htarget : 2 * ((pos.val + 1 : ℕ) : ℝ) * (V - -V)
        * (R * ‖z - w‖ / Real.sqrt (hd : ℝ))
      = attnStageL pos.val V R (hd : ℝ) * ‖z - w‖ := by
    have hsq : Real.sqrt (hd : ℝ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hhR)
    unfold attnStageL
    rw [show V - -V = 2 * V by ring]
    field_simp [hsq]
    ring
  rw [Real.norm_eq_abs]
  unfold attnRef
  exact le_trans hmain (le_of_eq htarget)

/-! ## The domain of the stage -/

/-- The `attn_domain_v2` slices whose `gridVec` is `y`, with the query component bound `B`. -/
def attnStageDom (state : attn.KvState) (layer q_head pos : Usize)
    (k v : alloc.vec.Vec (alloc.vec.Vec I64)) (kplane vplane : alloc.vec.Vec I64)
    (hd : ℕ) (B : ℝ) : Set (Fin hd → ℝ) :=
  {y | ∃ p : Slice I64 × Slice I64, y = MatvecBridge.gridVec (n := hd) p.1 ∧
    attn_domain_v2 state layer q_head pos p.1 p.2 k v kplane vplane ∧
    (∀ i : Fin hd, |toReal (p.1.val.getD i.val 0#i64)| ≤ B)}

/-! ## `attnErrBoundV2` monotonicity -/

theorem scoreErrV2_nonneg {hdim idealAbs scaleAbs : ℝ} (hi : 0 ≤ idealAbs) (hs : 0 ≤ scaleAbs) :
    0 ≤ scoreErrV2 hdim idealAbs scaleAbs := by
  unfold scoreErrV2
  have hsqrt : 0 ≤ Real.sqrt hdim := Real.sqrt_nonneg _
  have h1 : 0 ≤ scaleAbs * (1 / (2 * q32)) :=
    mul_nonneg hs (div_nonneg zero_le_one (mul_nonneg (by norm_num) q32_pos.le))
  have h2 : 0 ≤ Real.sqrt hdim * idealAbs * (epsExp * (1 / Real.sqrt hdim) + 4 / q32) := by
    apply mul_nonneg (mul_nonneg hsqrt hi)
    exact add_nonneg (mul_nonneg epsExp_pos.le (div_nonneg zero_le_one hsqrt))
      (div_nonneg (by norm_num) q32_pos.le)
  have h3 : 0 ≤ 1 / (2 * q32) := div_nonneg zero_le_one (mul_nonneg (by norm_num) q32_pos.le)
  linarith

theorem softmaxPerturbBound_nonneg (n : ℕ) {δ : ℝ} (hδ : 0 ≤ δ) :
    0 ≤ softmaxPerturbBound n δ := by
  unfold softmaxPerturbBound
  have h1 : 0 < 1 - epsExp := by linarith [epsExp_lt_one]
  have h2 : 0 < 1 - epsExp - 4 / q32 := attn_total_floor_pos
  have hX : 0 ≤ (n : ℝ) * (4 / q32 + Real.exp (-16)) :=
    mul_nonneg (Nat.cast_nonneg n)
      (add_nonneg (div_nonneg (by norm_num) q32_pos.le) (Real.exp_pos _).le)
  have ha : 0 ≤ epsExp + (Real.exp δ - 1) := by
    have := Real.one_le_exp hδ
    linarith [epsExp_pos]
  have hA : 0 ≤ (epsExp + (Real.exp δ - 1)) / (1 - epsExp) := div_nonneg ha h1.le
  have h2' : 0 ≤ ((epsExp + (Real.exp δ - 1)) / (1 - epsExp) + 1)
      * ((n : ℝ) * (4 / q32 + Real.exp (-16))) / (1 - epsExp - 4 / q32) :=
    div_nonneg (mul_nonneg (by linarith) hX) h2.le
  linarith

theorem softmaxPerturbBound_mono (n : ℕ) {δ δ' : ℝ} (h : δ ≤ δ') :
    softmaxPerturbBound n δ ≤ softmaxPerturbBound n δ' := by
  unfold softmaxPerturbBound
  have h1 : 0 < 1 - epsExp := by linarith [epsExp_lt_one]
  have h2 : 0 < 1 - epsExp - 4 / q32 := attn_total_floor_pos
  have hX : 0 ≤ (n : ℝ) * (4 / q32 + Real.exp (-16)) :=
    mul_nonneg (Nat.cast_nonneg n)
      (add_nonneg (div_nonneg (by norm_num) q32_pos.le) (Real.exp_pos _).le)
  have ha : epsExp + (Real.exp δ - 1) ≤ epsExp + (Real.exp δ' - 1) := by
    have := Real.exp_le_exp.mpr h
    linarith
  have hA : (epsExp + (Real.exp δ - 1)) / (1 - epsExp)
      ≤ (epsExp + (Real.exp δ' - 1)) / (1 - epsExp) := div_le_div_of_nonneg_right ha h1.le
  have hB : (epsExp + (Real.exp δ - 1)) / (1 - epsExp) + 1
      ≤ (epsExp + (Real.exp δ' - 1)) / (1 - epsExp) + 1 := by linarith
  have h2' : ((epsExp + (Real.exp δ - 1)) / (1 - epsExp) + 1)
        * ((n : ℝ) * (4 / q32 + Real.exp (-16))) / (1 - epsExp - 4 / q32)
      ≤ ((epsExp + (Real.exp δ' - 1)) / (1 - epsExp) + 1)
        * ((n : ℝ) * (4 / q32 + Real.exp (-16))) / (1 - epsExp - 4 / q32) :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hB hX) h2.le
  linarith

theorem attnErrBoundV2_nonneg (n : ℕ) {V δ : ℝ} (hV : 0 ≤ V) (hδ : 0 ≤ δ) :
    0 ≤ attnErrBoundV2 n V δ := by
  unfold attnErrBoundV2
  have hsp := softmaxPerturbBound_nonneg n hδ
  have h2 : 0 ≤ (n : ℝ) * V / (2 * q32) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg n) hV) (mul_nonneg (by norm_num) q32_pos.le)
  have h3 : 0 ≤ 2 * V * softmaxPerturbBound n δ := mul_nonneg (by linarith) hsp
  have h1 : 0 ≤ 1 / (2 * q32) := div_nonneg zero_le_one (mul_nonneg (by norm_num) q32_pos.le)
  linarith

theorem attnErrBoundV2_mono (n : ℕ) {V V' δ δ' : ℝ} (hV : 0 ≤ V) (hVV : V ≤ V')
    (hδ : 0 ≤ δ) (hδδ : δ ≤ δ') :
    attnErrBoundV2 n V δ ≤ attnErrBoundV2 n V' δ' := by
  unfold attnErrBoundV2
  have hsp := softmaxPerturbBound_mono n hδδ
  have hsp0 := softmaxPerturbBound_nonneg n hδ
  have hsp0' : 0 ≤ softmaxPerturbBound n δ' := le_trans hsp0 hsp
  have h3 : 2 * V * softmaxPerturbBound n δ ≤ 2 * V' * softmaxPerturbBound n δ' :=
    mul_le_mul (by linarith) hsp hsp0 (by linarith)
  have h4 : (n : ℝ) * V / (2 * q32) ≤ (n : ℝ) * V' / (2 * q32) :=
    div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hVV (Nat.cast_nonneg n))
      (mul_nonneg (by norm_num) q32_pos.le)
  linarith

end

end Nous.Model
