/-
# Composed per-operation error through a layer fold, with a decision margin

The verified numeric kernels of `Numeric/` each carry a per-operation relative error bound
(`apeiron_word.epsExp = 2^-20`, `apeiron_word.epsHorner = 2*10^-7`,
`apeiron_word.trig_floor = 2^-20` in `Numeric.Scale`, `Numeric.ExpRefinement`,
`Numeric.RopeRefinement`). A token passes through many such operations in sequence, so the
per-operation budgets do not by themselves bound the token-level error. This module supplies the
missing composition step: an abstract, engine-independent layer contract, its `N`-fold
accumulation and the margin predicate lemma that turns the accumulated budget into a decision
statement.

## Layer contract

A `Layer E` on a normed space `E` pairs an executing map `exec` with a reference map `ref` and
carries

* an absolute defect budget `delta` with `‖exec z - ref z‖ ≤ delta` for every `z`, and
* an expansion factor `L` with `‖ref z - ref w‖ ≤ L * ‖z - w‖`.

`Layer.ofRelative` derives `delta = eps * B` from the relative form used by the numeric refinements,
`‖exec z - ref z‖ ≤ eps * ‖ref z‖`, together with a scale bound `‖ref z‖ ≤ B` on the reference
layer's output. The relative form at a reference trajectory of norm at most `B` is the per-operation
`eps` of the inference-parity note.

## Accumulation

`Layer.budget` is the accumulated absolute error of a fold. For a list of layers whose head is
outermost (applied last),

  `budget [g_0, g_1, …, g_{N-1}] = sum_{k<N} delta_k * prod_{j<k} L_j`,
  `= delta_0 + L_0 * (delta_1 + L_1 * (… ))`.

`Layer.norm_execComp_sub_refComp_le` proves `‖execComp l x - refComp l x‖ ≤ budget l`, and
`Layer.budget_replicate_le` bounds the identical-fold case by `N * delta` when `L ≤ 1` (a
contraction), giving the geometric accumulation `delta * sum_{k<N} L^k ≤ N * delta`.

## Margin

`StrictWinner s margin best` is the the inference-parity note margin shape: `best` leads every
other index by at least `margin`. `strictWinner_perturb` shows that a perturbation bounded by `eta`
with `2*eta < margin` preserves a positive margin, and `decision_margin_preserved` wires the layer
fold into it: a reference decision whose gap exceeds twice the scored fold budget selects the same
index when the executing fold is run. `numeric_fold_margin_usable` and `numeric_decision_unchanged`
instantiate the chain with the landed numeric budgets and a four-layer fold.

## Trust boundary

The development is conditional on the per-layer hypotheses (`defect`, `expansion`, and the scale
bound of `ofRelative`). It does not establish those hypotheses for any executing kernel — that is
the per-body refinement of `Numeric/` — and it makes no claim about `rustc`/LLVM codegen, the
extracted word denotation or the accuracy of any sampled output.
-/
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Tactic
import Numeric.Scale
import Numeric.ExpRefinement
import Numeric.RopeRefinement
import Numeric.EngineNormRefinement

namespace Nous.Model

variable {E : Type*} [NormedAddCommGroup E]

/-- A layer contract on a normed space: an executing map `exec`, a reference map `ref`, an absolute
defect budget `delta` and an expansion factor `L`.

The defect is absolute, `‖exec z - ref z‖ ≤ delta`; `Layer.ofRelative` derives it from a relative
error budget and a scale bound on the reference output. The expansion factor is the reference map's
Lipschitz constant. -/
structure Layer (E : Type*) [NormedAddCommGroup E] where
  /-- The reference map of the layer. -/
  ref : E → E
  /-- The executing map of the layer. -/
  exec : E → E
  /-- The absolute defect budget `‖exec z - ref z‖ ≤ delta`. -/
  delta : ℝ
  /-- The reference expansion factor `‖ref z - ref w‖ ≤ L * ‖z - w‖`. -/
  L : ℝ
  delta_nonneg : 0 ≤ delta
  L_nonneg : 0 ≤ L
  defect : ∀ z, ‖exec z - ref z‖ ≤ delta
  expansion : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖

namespace Layer

/-- Reference composition of a layer list. The head is the outermost layer, applied last; the tail
is applied innermost. -/
def refComp : List (Layer E) → E → E
  | [], x => x
  | g :: gs, x => g.ref (refComp gs x)

/-- Executing composition of a layer list. The head is the outermost layer, applied last; the tail
is applied innermost. -/
def execComp : List (Layer E) → E → E
  | [], x => x
  | g :: gs, x => g.exec (execComp gs x)

/-- Accumulated absolute error budget of a layer fold, defined by the recurrence
`budget (g :: gs) = g.delta + g.L * budget gs`. Unfolded, for a head-outermost list
`[g_0, …, g_{N-1}]` it is `sum_{k<N} delta_k * prod_{j<k} L_j`. -/
def budget : List (Layer E) → ℝ
  | [] => 0
  | g :: gs => g.delta + g.L * budget gs

/-- The budget is nonnegative. -/
theorem budget_nonneg (l : List (Layer E)) : 0 ≤ budget l := by
  induction l with
  | nil => simp [budget]
  | cons g gs ih =>
      simp only [budget]
      exact add_nonneg g.delta_nonneg (mul_nonneg g.L_nonneg ih)

/-- The two-layer head expansion of the budget recurrence. -/
theorem budget_cons_cons (g h : Layer E) (gs : List (Layer E)) :
    budget (g :: h :: gs) = g.delta + g.L * (h.delta + h.L * budget gs) := by
  simp only [budget]

/-- The three-layer closed form `delta_0 + L_0*delta_1 + L_0*L_1*delta_2`, i.e.
`sum_{k<3} delta_k * prod_{j<k} L_j`. -/
theorem budget_three (g₀ g₁ g₂ : Layer E) :
    budget [g₀, g₁, g₂] = g₀.delta + g₀.L * g₁.delta + g₀.L * g₁.L * g₂.delta := by
  simp only [budget]
  ring

/-- The composed executing fold differs from the composed reference fold by at most the accumulated
budget. -/
theorem norm_execComp_sub_refComp_le (l : List (Layer E)) (x : E) :
    ‖execComp l x - refComp l x‖ ≤ budget l := by
  induction l with
  | nil => simp [execComp, refComp, budget]
  | cons g gs ih =>
      simp only [execComp, refComp, budget]
      have hsplit :
          g.exec (execComp gs x) - g.ref (refComp gs x)
            = (g.exec (execComp gs x) - g.ref (execComp gs x))
              + (g.ref (execComp gs x) - g.ref (refComp gs x)) := by
        abel
      have hstep :
          ‖g.exec (execComp gs x) - g.ref (refComp gs x)‖
            ≤ ‖g.exec (execComp gs x) - g.ref (execComp gs x)‖
              + ‖g.ref (execComp gs x) - g.ref (refComp gs x)‖ := by
        rw [hsplit]
        exact norm_add_le _ _
      calc
        ‖g.exec (execComp gs x) - g.ref (refComp gs x)‖
            ≤ ‖g.exec (execComp gs x) - g.ref (execComp gs x)‖
              + ‖g.ref (execComp gs x) - g.ref (refComp gs x)‖ := hstep
        _ ≤ g.delta + g.L * ‖execComp gs x - refComp gs x‖ :=
              add_le_add (g.defect (execComp gs x))
                (g.expansion (execComp gs x) (refComp gs x))
        _ ≤ g.delta + g.L * budget gs :=
              add_le_add le_rfl (mul_le_mul_of_nonneg_left ih g.L_nonneg)

/-- Identical-layer fold: with expansion factor at most one the accumulated budget is at most
`N * delta`, dominating the geometric sum `delta * sum_{k<N} L^k`. -/
theorem budget_replicate_le (g : Layer E) (hL : g.L ≤ 1) (n : ℕ) :
    budget (List.replicate n g) ≤ n * g.delta := by
  induction n with
  | zero => simp [budget]
  | succ n ih =>
      rw [List.replicate_succ, budget, Nat.cast_add, Nat.cast_one]
      have hmono : g.L * budget (List.replicate n g) ≤ 1 * budget (List.replicate n g) :=
        mul_le_mul_of_nonneg_right hL (budget_nonneg _)
      nlinarith [hmono, ih]

/-- Derive an absolute-defect `Layer` from the relative error form used by the numeric refinements,
`‖exec z - ref z‖ ≤ eps * ‖ref z‖`, and a scale bound `‖ref z‖ ≤ B` on the reference output. The
resulting defect budget is `eps * B`. -/
def ofRelative (ref exec : E → E) (eps L B : ℝ)
    (hε : 0 ≤ eps) (hL : 0 ≤ L) (hB : 0 ≤ B)
    (hrefB : ∀ z, ‖ref z‖ ≤ B)
    (herr : ∀ z, ‖exec z - ref z‖ ≤ eps * ‖ref z‖)
    (hlip : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖) : Layer E where
  ref := ref
  exec := exec
  delta := eps * B
  L := L
  delta_nonneg := mul_nonneg hε hB
  L_nonneg := hL
  defect := fun z => le_trans (herr z) (mul_le_mul_of_nonneg_left (hrefB z) hε)
  expansion := hlip

/-- A concrete affine layer on `ℝ`: reference `z ↦ z`, executing `z ↦ z + e`. Its absolute defect is
`|e|` and its expansion factor is `1`. -/
def affineLayer (e : ℝ) : Layer ℝ where
  ref := fun z => z
  exec := fun z => z + e
  delta := |e|
  L := 1
  delta_nonneg := abs_nonneg e
  L_nonneg := zero_le_one
  defect := by
    intro z
    rw [add_sub_cancel_left]
    exact (Real.norm_eq_abs e).le
  expansion := by
    intro z w
    simp

end Layer

/-- A real-valued readout with a uniform Lipschitz bound, used to turn a state-space error bound
into a score gap. -/
structure LipschitzScore (E : Type*) [NormedAddCommGroup E] where
  /-- The readout functional. -/
  toFun : E → ℝ
  /-- The Lipschitz constant of the readout. -/
  lip : ℝ
  lip_nonneg : 0 ≤ lip
  isLipschitz : ∀ u v, |toFun u - toFun v| ≤ lip * ‖u - v‖

/-- `best` leads every other index by at least `margin` under the score `s`. This is the
`StrictWinner`/margin shape of the inference-parity note. -/
def StrictWinner {ι : Type*} (s : ι → ℝ) (margin : ℝ) (best : ι) : Prop :=
  ∀ i, i ≠ best → s best ≥ s i + margin

/-- A perturbation bounded by `eta` moves a margin of `margin` down to `margin - 2*eta`. -/
theorem strictWinner_perturb {ι : Type*} (s s' : ι → ℝ) (margin eta : ℝ) (best : ι)
    (hwin : StrictWinner s margin best) (heta : ∀ i, |s' i - s i| ≤ eta) :
    StrictWinner s' (margin - 2 * eta) best := by
  intro i hi
  have hbest := (abs_le.mp (heta best)).1
  have hi' := (abs_le.mp (heta i)).2
  have hw := hwin i hi
  linarith [hbest, hi', hw]

/-- A positive margin makes `best` the strict unique argmax. -/
theorem strictWinner_strict_argmax {ι : Type*} {s : ι → ℝ} {margin : ℝ} {best : ι}
    (hwin : StrictWinner s margin best) (hpos : 0 < margin) :
    ∀ i, i ≠ best → s i < s best := by
  intro i hi
  have h := hwin i hi
  linarith

/-- A perturbation bounded by `eta`, with `2*eta < margin`, keeps the strict argmax of the score. -/
theorem strictWinner_perturb_argmax {ι : Type*} (s s' : ι → ℝ) (margin eta : ℝ) (best : ι)
    (hwin : StrictWinner s margin best) (heta : ∀ i, |s' i - s i| ≤ eta)
    (hgap : 2 * eta < margin) :
    ∀ i, i ≠ best → s' i < s' best :=
  strictWinner_strict_argmax (strictWinner_perturb s s' margin eta best hwin heta)
    (sub_pos.mpr hgap)

/-- The scored fold error: a Lipschitz readout of the executing fold differs from the readout of
the reference fold by at most `lip * budget l`. -/
theorem score_perturb_bound (score : LipschitzScore E) (l : List (Layer E)) (x : E) :
    |score.toFun (Layer.execComp l x) - score.toFun (Layer.refComp l x)|
      ≤ score.lip * Layer.budget l :=
  calc
    |score.toFun (Layer.execComp l x) - score.toFun (Layer.refComp l x)|
        ≤ score.lip * ‖Layer.execComp l x - Layer.refComp l x‖ := score.isLipschitz _ _
    _ ≤ score.lip * Layer.budget l :=
        mul_le_mul_of_nonneg_left (Layer.norm_execComp_sub_refComp_le l x) score.lip_nonneg

/-- The decision margin is preserved across an executing layer fold: if the reference decision leads
by more than twice the scored fold budget, the executing fold selects the same strict argmax. -/
theorem decision_margin_preserved {ι : Type*} (score : LipschitzScore E) (l : List (Layer E))
    (x₀ : ι → E) (best : ι) (margin : ℝ)
    (hwin : StrictWinner (fun i => score.toFun (Layer.refComp l (x₀ i))) margin best)
    (hgap : 2 * (score.lip * Layer.budget l) < margin) :
    ∀ i, i ≠ best →
      score.toFun (Layer.execComp l (x₀ i))
        < score.toFun (Layer.execComp l (x₀ best)) :=
  strictWinner_perturb_argmax (fun i => score.toFun (Layer.refComp l (x₀ i)))
    (fun i => score.toFun (Layer.execComp l (x₀ i))) margin (score.lip * Layer.budget l) best
    hwin (fun i => score_perturb_bound score l (x₀ i)) hgap

/-! ## Numeric instantiation

The landed per-operation budgets of the numeric refinements: `exp`'s relative `2^-20`, the Horner
stage's `2*10^-7` and the trigonometric absolute floor `2^-20`. Their sum is the per-layer scale
used by `Layer.ofRelative`; a four-layer contraction fold keeps the accumulated budget below the
`1/1000` decision margin. -/

/-- The landed per-operation numeric budgets combined into one scale: `exp`'s relative `2^-20`, the
Horner stage's `2*10^-7` and the trigonometric absolute floor `2^-20`. -/
noncomputable def numericScale : ℝ :=
  apeiron_word.epsExp + apeiron_word.epsHorner + apeiron_word.trig_floor

/-- The combined numeric scale is below `1/100000`. -/
theorem numericScale_lt_one_hundred_thousandth : numericScale < 1 / 100000 := by
  unfold numericScale apeiron_word.epsExp apeiron_word.epsHorner apeiron_word.trig_floor
  norm_num

/-- Sanity instantiation of the fold: a four-layer contraction fold with per-layer defect at most
`numericScale` and a readout of Lipschitz constant at most one has scored budget below `1/1000`. -/
theorem numeric_fold_margin_usable (g : Layer E) (hL : g.L ≤ 1) (hdelta : g.delta ≤ numericScale)
    (score : LipschitzScore E) (hlip : score.lip ≤ 1) :
    2 * (score.lip * Layer.budget (List.replicate 4 g)) < 1 / 1000 := by
  have hb : Layer.budget (List.replicate 4 g) ≤ 4 * numericScale :=
    calc
      Layer.budget (List.replicate 4 g) ≤ 4 * g.delta := Layer.budget_replicate_le g hL 4
      _ ≤ 4 * numericScale := by linarith
  have hprod : score.lip * Layer.budget (List.replicate 4 g) ≤ 1 * (4 * numericScale) :=
    mul_le_mul hlip hb (Layer.budget_nonneg _) (by norm_num)
  have hnum : (1 : ℝ) * (4 * numericScale) < 1 / 2000 := by
    linarith [numericScale_lt_one_hundred_thousandth]
  nlinarith [hprod, hnum]

/-- End-to-end numeric instantiation: a four-layer contraction fold whose per-layer defect is
within `numericScale` and whose readout has Lipschitz constant at most one leaves the reference
decision fixed against a margin of `1/1000`. -/
theorem numeric_decision_unchanged {ι : Type*} (g : Layer E) (hL : g.L ≤ 1)
    (hdelta : g.delta ≤ numericScale) (score : LipschitzScore E) (hlip : score.lip ≤ 1)
    (x₀ : ι → E) (best : ι)
    (hwin : StrictWinner (fun i => score.toFun (Layer.refComp (List.replicate 4 g) (x₀ i)))
      (1 / 1000) best) :
    ∀ i, i ≠ best →
      score.toFun (Layer.execComp (List.replicate 4 g) (x₀ i))
        < score.toFun (Layer.execComp (List.replicate 4 g) (x₀ best)) :=
  decision_margin_preserved score (List.replicate 4 g) x₀ best (1 / 1000) hwin
    (numeric_fold_margin_usable g hL hdelta score hlip)

/-- Concrete four-layer sanity: the affine example layer on `ℝ` obeys the accumulated
`4 * |e|` bound. -/
theorem affine_fold_four (e x : ℝ) :
    ‖Layer.execComp (List.replicate 4 (Layer.affineLayer e)) x
        - Layer.refComp (List.replicate 4 (Layer.affineLayer e)) x‖ ≤ 4 * |e| :=
  calc
    ‖Layer.execComp (List.replicate 4 (Layer.affineLayer e)) x
        - Layer.refComp (List.replicate 4 (Layer.affineLayer e)) x‖
        ≤ Layer.budget (List.replicate 4 (Layer.affineLayer e)) :=
          Layer.norm_execComp_sub_refComp_le _ _
    _ ≤ 4 * (Layer.affineLayer e).delta := Layer.budget_replicate_le _ le_rfl 4
    _ = 4 * |e| := by rfl

/-! ## Per-operation ledger discharge

The numeric kernel refinements of `Numeric/` each bound the deviation of an extracted Q32.32 body
from its real reference on the body's window. This section turns those pointwise bounds into `Layer`
objects, so the per-layer defect budget `delta` of the fold is *proved* for SiLU
(`silu_window_refines`) and RMSNorm (`rmsElement_bound`) instead of assumed. The remaining per-layer
hypotheses are the operations whose `Layer` is not built from a refinement in this file: matvec,
attention-softmax, lm_head, the sampler and RoPE. Each now has a proved kernel refinement elsewhere
(`matvec_refines_proved`, `attn_head_refines_v2_proved`, `lm_head_refines_proved`,
`word_argmax_fixed_spec`, `rope_pair_spec_proved`), not wired into this `Layer E` fold, which cannot carry
a dimension-changing map; `Nous/Model/BlockComposition.lean` composes them as `Stage E F`.
`residualOps` names the four non-RoPE entries.

A kernel acts on the Q32.32 grid, not on all of `ℝ`, so `Layer.ofKernel` realizes a grid-indexed map
as a total map that equals the reference off the admissible domain: the defect is proved on the
domain from the kernel bound and is zero elsewhere, and the absolute budget is `eps * B + floor`,
with `B` the domain magnitude bound and `(eps, floor)` the per-operation relative/floor ledger. -/

open apeiron_word Aeneas Aeneas.Std Result Classical

namespace Layer

/-- A grid-indexed per-operation refinement realized as a total `Layer`. `γ` embeds the executing
grid into the carrier `E`; `g` is the real denotation of the extracted output; `dom` is the
operation's admissible domain; `eps`/`floor` are the relative/floor ledger; `B` bounds `‖γ v‖` on
`dom`; `L` is the reference expansion factor. The defect budget is `eps * B + floor`, proved on
`dom` from `hbound` and zero off it on the grid and off the grid entirely. -/
noncomputable def ofKernel {ι : Type*} {E : Type*} [NormedAddCommGroup E] (γ : ι → E)
    (ref : E → E) (g : ι → E) (dom : ι → Prop) (eps floor B L : ℝ)
    (hε : 0 ≤ eps) (hf : 0 ≤ floor) (hB : 0 ≤ B) (hL : 0 ≤ L)
    (hbound : ∀ v, dom v → ‖g v - ref (γ v)‖ ≤ eps * ‖γ v‖ + floor)
    (hscale : ∀ v, dom v → ‖γ v‖ ≤ B)
    (hlip : ∀ z w, ‖ref z - ref w‖ ≤ L * ‖z - w‖) : Layer E :=
  let exec : E → E := fun z =>
    if h : ∃ v, γ v = z ∧ dom v then g (Classical.choose h)
    else if h : ∃ v, γ v = z then ref (γ (Classical.choose h)) else ref z
  { ref := ref
    exec := exec
    delta := eps * B + floor
    L := L
    delta_nonneg := add_nonneg (mul_nonneg hε hB) hf
    L_nonneg := hL
    defect := by
      intro z
      show ‖exec z - ref z‖ ≤ eps * B + floor
      dsimp only [exec]
      by_cases h : ∃ v, γ v = z ∧ dom v
      · rw [dif_pos h]
        obtain ⟨hγ, hd⟩ := Classical.choose_spec h
        have hgoal : ‖g (Classical.choose h) - ref z‖
            = ‖g (Classical.choose h) - ref (γ (Classical.choose h))‖ := by rw [hγ]
        rw [hgoal]
        have hb := hbound (Classical.choose h) hd
        have hs := hscale (Classical.choose h) hd
        calc ‖g (Classical.choose h) - ref (γ (Classical.choose h))‖
            ≤ eps * ‖γ (Classical.choose h)‖ + floor := hb
          _ ≤ eps * B + floor := by
              have hm : eps * ‖γ (Classical.choose h)‖ ≤ eps * B :=
                mul_le_mul_of_nonneg_left hs hε
              linarith
      · rw [dif_neg h]
        by_cases h2 : ∃ v, γ v = z
        · rw [dif_pos h2]
          have hγ : γ (Classical.choose h2) = z := Classical.choose_spec h2
          rw [hγ, sub_self, norm_zero]
          exact add_nonneg (mul_nonneg hε hB) hf
        · rw [dif_neg h2, sub_self, norm_zero]
          exact add_nonneg (mul_nonneg hε hB) hf
    expansion := hlip }

/-- The one-layer fold budget is the layer's own defect budget. -/
theorem budget_singleton (g : Layer E) : budget [g] = g.delta := by
  simp [budget]

end Layer

/-! ### SiLU ledger -/

/-- The SiLU per-operation relative coefficient `epsExp + 4/q32`. -/
noncomputable def siluEps : ℝ := epsExp + 4 / q32

/-- The SiLU per-operation absolute floor `1/(2·2^32)`. -/
noncomputable def siluFloor : ℝ := 1 / (2 * q32)

/-- The SiLU absolute defect budget on the window `|x| ≤ 21`: `(epsExp + 4/q32)·21 + 1/(2·2^32)`. -/
noncomputable def siluDelta : ℝ := siluEps * 21 + siluFloor

theorem epsExp_pos : 0 < epsExp := by
  unfold epsExp
  exact zpow_pos (by norm_num) _

theorem epsExp_nonneg : 0 ≤ epsExp := le_of_lt epsExp_pos

theorem trigFloor_pos : 0 < trig_floor := by
  unfold trig_floor
  exact zpow_pos (by norm_num) _

theorem trigFloor_nonneg : 0 ≤ trig_floor := le_of_lt trigFloor_pos

theorem q32_nonneg : 0 ≤ q32 := le_of_lt q32_pos

theorem siluEps_nonneg : 0 ≤ siluEps := by
  have h1 : (0 : ℝ) ≤ epsExp := epsExp_nonneg
  have h2 : (0 : ℝ) ≤ 4 / q32 := div_nonneg (by norm_num) q32_nonneg
  unfold siluEps
  linarith

theorem siluFloor_nonneg : 0 ≤ siluFloor := by
  unfold siluFloor
  exact div_nonneg (by norm_num) (by linarith [q32_pos])

/-- On the SiLU window the Q32.32 value has magnitude at most 21. -/
theorem silu_window_norm_le (v : I64) (hv : silu_window v) : ‖toReal v‖ ≤ 21 := by
  rw [Real.norm_eq_abs, abs_le]
  rw [silu_window] at hv
  obtain ⟨hlo, hhi⟩ := hv
  constructor
  · unfold toReal
    rw [le_div_iff₀ q32_pos]
    rw [show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl]
    have h : (-21 : ℝ) * (2 : ℝ) ^ (32 : ℕ) = (-90194313216 : ℤ) := by norm_num
    rw [h]
    exact_mod_cast (le_of_lt hlo)
  · unfold toReal
    rw [div_le_iff₀ q32_pos]
    rw [show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl]
    have h : (21 : ℝ) * (2 : ℝ) ^ (32 : ℕ) = (90194313216 : ℤ) := by norm_num
    rw [h]
    exact_mod_cast (le_trans (le_of_lt hhi) (by norm_num))

/-- The SiLU grid map: the real denotation of the extracted output on the window, the reference
elsewhere. -/
noncomputable def siluGrid (v : I64) : ℝ :=
  if hv : silu_window v then toReal (Classical.choose (silu_window_refines v hv))
  else siluReal (toReal v)

theorem siluGrid_bound (v : I64) (hv : silu_window v) :
    ‖siluGrid v - siluReal (toReal v)‖ ≤ siluEps * ‖toReal v‖ + siluFloor := by
  have hb := (Classical.choose_spec (silu_window_refines v hv)).2
  rw [siluGrid, dif_pos hv, Real.norm_eq_abs]
  simp only [Real.norm_eq_abs]
  unfold siluEps siluFloor
  linarith [hb]

/-- Derivative of the real SiLU `x ↦ x / (1 + exp (-x))`. -/
theorem siluReal_hasDerivAt (x : ℝ) :
    HasDerivAt siluReal
      ((1 * (1 + Real.exp (-x)) - x * (-Real.exp (-x))) / (1 + Real.exp (-x)) ^ 2) x := by
  have hinner : HasDerivAt (fun y : ℝ => 1 + Real.exp (-y)) (-Real.exp (-x)) x := by
    have hneg : HasDerivAt (fun y : ℝ => -y) (-1) x := hasDerivAt_neg x
    have hcomp : HasDerivAt (fun y : ℝ => Real.exp (-y)) (Real.exp (-x) * (-1)) x :=
      (Real.hasDerivAt_exp (-x)).comp x hneg
    have h := hcomp.const_add 1
    simpa using h
  have hd : (1 + Real.exp (-x)) ≠ 0 := (by positivity : (0 : ℝ) < 1 + Real.exp (-x)).ne'
  have h := (hasDerivAt_id x).div hinner hd
  unfold siluReal
  exact h

/-- The SiLU derivative is bounded by `3` on `ℝ`. -/
theorem siluReal_deriv_bound (x : ℝ) : ‖deriv siluReal x‖ ≤ 3 := by
  rw [(siluReal_hasDerivAt x).deriv]
  set e : ℝ := Real.exp (-x) with he
  have hep : 0 < e := by rw [he]; exact Real.exp_pos (-x)
  have hlog_ub : Real.log e ≤ e - 1 := Real.log_le_sub_one_of_pos hep
  have hlog_lb : 1 - e⁻¹ ≤ Real.log e := by
    have h := Real.log_le_sub_one_of_pos (inv_pos.mpr hep)
    rw [Real.log_inv] at h
    linarith
  have hmul_ub : e * Real.log e ≤ e * (e - 1) :=
    mul_le_mul_of_nonneg_left hlog_ub (le_of_lt hep)
  have hmul_lb : e - 1 ≤ e * Real.log e := by
    have h := mul_le_mul_of_nonneg_left hlog_lb (le_of_lt hep)
    have hs : e * (1 - e⁻¹) = e - 1 := by field_simp
    linarith [h, hs]
  have hxe : x * e = -(e * Real.log e) := by
    have hx : x = -Real.log e := by rw [he, Real.log_exp]; ring
    rw [hx]; ring
  have hnum : |1 * (1 + e) - x * (-e)| ≤ 3 * (1 + e) ^ 2 := by
    have hxe' : 1 * (1 + e) - x * (-e) = 1 + e - e * Real.log e := by
      rw [show x * (-e) = -(x * e) from by ring, hxe]; ring
    rw [hxe', abs_le]
    constructor <;> nlinarith [sq_nonneg (e + 1), hep]
  have hdenpos : (0 : ℝ) < (1 + e) ^ 2 := pow_pos (by linarith) 2
  rw [Real.norm_eq_abs, abs_div, abs_of_pos hdenpos]
  rw [div_le_iff₀ hdenpos]
  exact hnum

/-- The real SiLU is `3`-Lipschitz. -/
theorem siluReal_lipschitz (z w : ℝ) :
    ‖siluReal z - siluReal w‖ ≤ 3 * ‖z - w‖ :=
  Convex.norm_image_sub_le_of_norm_deriv_le (𝕜 := ℝ) (G := ℝ) (s := Set.univ) (C := 3)
    (fun x _ => (siluReal_hasDerivAt x).differentiableAt)
    (fun x _ => siluReal_deriv_bound x)
    convex_univ (Set.mem_univ w) (Set.mem_univ z)

/-- The SiLU layer, defect budget `siluDelta` from `silu_window_refines` and expansion `3`. -/
noncomputable def siluLayer : Layer ℝ :=
  Layer.ofKernel (ι := I64) (E := ℝ) toReal siluReal siluGrid silu_window
    siluEps siluFloor 21 3 siluEps_nonneg siluFloor_nonneg (by norm_num) (by norm_num)
    siluGrid_bound silu_window_norm_le siluReal_lipschitz

/-! ### RMSNorm ledger -/

/-- The extracted `fixed_rsqrt` output at `a`, chosen from the proved refinement. -/
noncomputable def rmsRoot (a : I64) (ha : rsqrt_window a) : I64 :=
  Classical.choose (fixed_rsqrt_refines_proved a ha)

theorem rmsRoot_refines (a : I64) (ha : rsqrt_window a) :
    |toReal (rmsRoot a ha) - 1 / Real.sqrt (toReal a)| ≤
      epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32 :=
  (Classical.choose_spec (fixed_rsqrt_refines_proved a ha)).2

/-- The RMSNorm per-element relative coefficient `epsExp · (1/√a) + 4/q32` at the fixed argument
`a`. -/
noncomputable def rmsEps (a : I64) : ℝ := epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32

/-- The RMSNorm per-element absolute floor `1/(2·2^32)`. -/
noncomputable def rmsFloor : ℝ := 1 / (2 * q32)

/-- The RMSNorm absolute budget at magnitude bound `B`: `rmsEps a · B + rmsFloor`. -/
noncomputable def rmsDelta (a : I64) (B : ℝ) : ℝ := rmsEps a * B + rmsFloor

theorem rmsEps_nonneg (a : I64) : 0 ≤ rmsEps a := by
  unfold rmsEps
  have h1 : (0 : ℝ) ≤ epsExp := epsExp_nonneg
  have h2 : (0 : ℝ) ≤ 1 / Real.sqrt (toReal a) := div_nonneg (by norm_num) (Real.sqrt_nonneg _)
  have h3 : (0 : ℝ) ≤ 4 / q32 := div_nonneg (by norm_num) q32_nonneg
  have hm : (0 : ℝ) ≤ epsExp * (1 / Real.sqrt (toReal a)) := mul_nonneg h1 h2
  linarith

theorem rmsFloor_nonneg : 0 ≤ rmsFloor := by
  unfold rmsFloor
  exact div_nonneg (by norm_num) (by linarith [q32_pos])

/-- The scalar RMSNorm domain at root `r` and magnitude bound `B`: the input magnitude is bounded
and the epilogue quotient is representable as an `i64`. -/
def rmsDom (r : I64) (B : ℝ) (x : I64) : Prop :=
  ‖toReal x‖ ≤ B ∧
    ∃ y : I64, y.val = divRoundNearest (x.val * r.val) ((2 : ℤ) ^ (32 : ℕ))

/-- The scalar RMSNorm grid map: the real denotation of the epilogue output `round(x·r/2^32)`. -/
noncomputable def rmsGrid (r : I64) (x : I64) : ℝ :=
  if h : ∃ y : I64, y.val = divRoundNearest (x.val * r.val) ((2 : ℤ) ^ (32 : ℕ)) then
    toReal (Classical.choose h)
  else 0

/-- The RMSNorm scalar reference at the fixed argument `a`: `z ↦ z · (1/√a)`. -/
noncomputable def rmsRef (a : I64) (z : ℝ) : ℝ := z * (1 / Real.sqrt (toReal a))

theorem rmsGrid_bound (a r : I64) (ha : rsqrt_window a) (hr : r = rmsRoot a ha)
    (B : ℝ) (x : I64) (hx : rmsDom r B x) :
    ‖rmsGrid r x - rmsRef a (toReal x)‖ ≤ rmsEps a * ‖toReal x‖ + rmsFloor := by
  obtain ⟨-, y, hy⟩ := hx
  have hdom : ∃ y : I64, y.val = divRoundNearest (x.val * r.val) ((2 : ℤ) ^ (32 : ℕ)) := ⟨y, hy⟩
  rw [rmsGrid, dif_pos hdom, Real.norm_eq_abs]
  simp only [Real.norm_eq_abs]
  have hrsq : |toReal r - 1 / Real.sqrt (toReal a)| ≤
      epsExp * (1 / Real.sqrt (toReal a)) + 4 / q32 := by
    rw [hr]; exact rmsRoot_refines a ha
  have hb := rmsElement_bound a r x (Classical.choose hdom) hrsq (Classical.choose_spec hdom)
  simp only [rmsReal] at hb
  simp only [rmsRef]
  unfold rmsEps rmsFloor
  linarith [hb]

/-- The real RMSNorm reference `z ↦ z · (1/√a)` is Lipschitz with constant `1/√a`. -/
theorem rmsRef_lipschitz (a : I64) (z w : ℝ) :
    ‖rmsRef a z - rmsRef a w‖ ≤ (1 / Real.sqrt (toReal a)) * ‖z - w‖ := by
  have hc : 0 ≤ 1 / Real.sqrt (toReal a) := div_nonneg (by norm_num) (Real.sqrt_nonneg _)
  have hid : rmsRef a z - rmsRef a w = (z - w) * (1 / Real.sqrt (toReal a)) := by
    unfold rmsRef; ring
  rw [hid, Real.norm_eq_abs, abs_mul, abs_of_nonneg hc, Real.norm_eq_abs]
  exact le_of_eq (mul_comm _ _)

/-- The scalar RMSNorm layer at `a` in the `fixed_rsqrt` window and magnitude bound `B`. -/
noncomputable def rmsLayer (a : I64) (ha : rsqrt_window a) (B : ℝ) (hB : 0 ≤ B) :
    Layer ℝ :=
  Layer.ofKernel (ι := I64) (E := ℝ) toReal (rmsRef a) (rmsGrid (rmsRoot a ha))
    (rmsDom (rmsRoot a ha) B) (rmsEps a) rmsFloor B (1 / Real.sqrt (toReal a))
    (rmsEps_nonneg a) rmsFloor_nonneg hB (div_nonneg (by norm_num) (Real.sqrt_nonneg _))
    (fun x hx => rmsGrid_bound a (rmsRoot a ha) ha rfl B x hx)
    (fun x hx => hx.1)
    (rmsRef_lipschitz a)

/-! ### Composition instantiation and residual

The refined layers enter `decision_margin_preserved` with their defect budgets proved. The residual
per-layer hypotheses are the operations whose `Layer` is not built from a refinement in this file:
matvec, attention-softmax, lm_head and the sampler (each refined elsewhere; see the section header
above). -/

/-- The operations whose `Layer` in this fold is assumed rather than built from a refinement; each
contributes an assumed `Layer` (defect `delta` and expansion `L`). -/
inductive UnrefinedOp where
  | matvec
  | attentionSoftmax
  | lmHead
  | sampler

/-- The residual per-layer hypothesis set. `decision_margin_preserved` consumes these `Layer`
entries unchanged; their `delta` and `L` fields are not derived in this file. -/
def residualOps : List UnrefinedOp :=
  [UnrefinedOp.matvec, UnrefinedOp.attentionSoftmax, UnrefinedOp.lmHead, UnrefinedOp.sampler]

theorem siluLayer_delta : siluLayer.delta = siluDelta := rfl

theorem rmsLayer_delta (a : I64) (ha : rsqrt_window a) (B : ℝ) (hB : 0 ≤ B) :
    (rmsLayer a ha B hB).delta = rmsDelta a B := rfl

/-- SiLU margin instantiation: a reference decision leading by more than twice the scored SiLU
budget is decided identically by the executing fold. -/
theorem silu_decision_margin_unchanged {ι : Type*} (score : LipschitzScore ℝ) (x₀ : ι → ℝ)
    (best : ι) (margin : ℝ)
    (hwin : StrictWinner (fun i => score.toFun (Layer.refComp [siluLayer] (x₀ i))) margin best)
    (hgap : 2 * (score.lip * siluDelta) < margin) :
    ∀ i, i ≠ best →
      score.toFun (Layer.execComp [siluLayer] (x₀ i))
        < score.toFun (Layer.execComp [siluLayer] (x₀ best)) := by
  refine decision_margin_preserved score [siluLayer] x₀ best margin hwin ?_
  rw [Layer.budget_singleton, siluLayer_delta]
  exact hgap

/-- RMSNorm margin instantiation. -/
theorem rms_decision_margin_unchanged {ι : Type*} (a : I64) (ha : rsqrt_window a) (B : ℝ)
    (hB : 0 ≤ B) (score : LipschitzScore ℝ) (x₀ : ι → ℝ) (best : ι) (margin : ℝ)
    (hwin : StrictWinner (fun i => score.toFun (Layer.refComp [rmsLayer a ha B hB] (x₀ i)))
      margin best)
    (hgap : 2 * (score.lip * rmsDelta a B) < margin) :
    ∀ i, i ≠ best →
      score.toFun (Layer.execComp [rmsLayer a ha B hB] (x₀ i))
        < score.toFun (Layer.execComp [rmsLayer a ha B hB] (x₀ best)) := by
  refine decision_margin_preserved score [rmsLayer a ha B hB] x₀ best margin hwin ?_
  rw [Layer.budget_singleton, rmsLayer_delta]
  exact hgap


end Nous.Model
