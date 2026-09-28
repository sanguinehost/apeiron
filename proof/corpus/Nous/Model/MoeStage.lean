/-
Copyright (c) 2026 Apeiron contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Apeiron contributors
-/

import Mathlib.Data.Fin.VecNotation
import Nous.Model.TopKRouter
import Nous.Model.BlockComposition

/-!
# Hard top-`k` mixture-of-experts as a block stage: discontinuity and stable-region Lipschitz

`Nous.Model.TopKRouter` proves that the fixed-point router's selection is the exact top-`k` set and
that the renormalized gate weights are the softmax over the selection. `Nous.Model.BlockComposition`
gives the dimension-changing stage contract `Stage E F` with an explicit admissible domain and a
**global** expansion factor. This module connects the two for the real-valued hard top-`k` mixture

  `moe x = Σ_{e ∈ sel x} softmax_S(z x)_e • f_e x`,  `z x = R x`,  `S = sel x`,

with each expert `f_e` `L_e`-Lipschitz and `‖f_e x‖ ≤ M_e`, and the linear router bound
`‖R‖_{∞→∞} ≤ ρ` (the max row `ℓ1` norm, `Stage.matvecRef_lipschitz`).

## Results

* `moe_not_continuous` — **no global Lipschitz constant exists** for hard top-`k` routing. A concrete
  two-expert one-dimensional instance (`R = ![1, 0]`, `f_0 = 0`, `f_1 = 1`, `k = 1`) with the exact
  top-`1` selection `cxSel` jumps at the tie boundary, so `¬ ∃ L, ∀ x y, ‖moe x − moe y‖ ≤ L‖x − y‖`.
* `fixedMoe_lipschitz` — the **fixed-selection** mixture `Σ_{e ∈ S} softmax_S(z x)_e • f_e x` is
  *globally* Lipschitz with constant `Lmax + 2·Mmax·ρ`, through `selWeight_sum_abs_le`: the
  renormalized softmax over `S` is `2`-Lipschitz from the `ℓ∞` norm on the logits to the `ℓ1` norm on
  the weights. That proof is the elementary ratio argument (`r_e = w'_e / w_e ∈ [e^{−2ε}, e^{2ε}]`,
  `Σ w_e r_e = 1`), reduced by `sum_abs_sub_le_of_bounds` to the scalar bound
  `(e^{2t} − 1)/(e^{2t} + 1) ≤ t` (`exp_two_mul_sub_div_le`).
* `moe_lipschitz_on_stable` — on a domain on which the router selects the same `S` everywhere, the hard
  mixture equals the fixed-selection mixture and inherits the bound (convexity of the domain is part
  of the stable-region data; the estimate is global for the reference mixture and does not need it).
* `margin_stable` — a logit margin bigger than `2ρr` freezes the selection on the ball of radius `r`:
  with a gap `> 2ρr` between every selected and unselected logit, every `y` with `‖y − x‖ ≤ r` has the
  same top-`k` selection `S`, so ball-shaped stable regions exist around margin points.
* `moeStage` — a `Stage (Fin d → ℝ) (Fin d → ℝ)` whose `ref` is the fixed-`S` mixture globally (hence
  the global `expansion` the contract demands), whose `dom` is the stable region `D`, and whose `exec`
  is the hard mixture. On `D` the hard mixture *is* the reference, so `defect` is `0`. **Off `dom` the
  `ref` is the fixed-`S` mixture, not the hard router** — `Stage.expansion` is a statement about
  `ref`, and the executing `exec` carries the hard selection with its own (empty on `D`) defect.

## Trust boundary

The counterexample, the fixed-selection Lipschitz bound, the stable-region bound and the `Stage` are
kernel-checked real-algebra statements (no `sorry`/`axiom`/`native_decide`). No Rust source, no
extraction and no fixed-point rounding is modelled here; the connection to `route_topk`/`gate_weights`
in `src/engine/moe.rs` and to the extracted bodies is the separate refinement obligation of the epic
(`docs/epics/moe-router.md`). Nothing is claimed about rustc/LLVM codegen, the kernel ABI, GPU/SPIR-V
or the `unsafe` container.
-/

namespace Nous.Model

open scoped BigOperators

/-! ## 1. Router logits, gate weights and the mixtures -/

/-- The router logits `z(x) = R x` of a linear router `R` with expert rows. Identical to
`Stage.matvecRef`, so the `ℓ∞ → ℓ∞` bound applies verbatim (`routerLogits_lipschitz`). -/
def routerLogits {E d : ℕ} (R : Fin E → Fin d → ℝ) (x : Fin d → ℝ) : Fin E → ℝ :=
  fun e => ∑ j, R e j * x j

theorem routerLogits_eq_matvecRef {E d : ℕ} (R : Fin E → Fin d → ℝ) (x : Fin d → ℝ) :
    routerLogits R x = Stage.matvecRef R x := rfl

/-- The router's `ℓ∞ → ℓ∞` Lipschitz bound: the max row `ℓ1` norm (hypothesis `hR`) bounds the row
sum, and `Stage.matvecRef_lipschitz` gives the bound with no loss. -/
theorem routerLogits_lipschitz {E d : ℕ} (R : Fin E → Fin d → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hR : ∀ e, ∑ j, |R e j| ≤ ρ) (x y : Fin d → ℝ) :
    ‖routerLogits R x - routerLogits R y‖ ≤ ρ * ‖x - y‖ :=
  Stage.matvecRef_lipschitz R ρ hρ hR x y

/-- The renormalized softmax gate weight of expert `e` over the selected set `S`:
`exp (z e) / Σ_{s ∈ S} exp (z s)`. This is the weight the fixed-point router computes with
renormalization (`softmax_topk_renorm`, the common partition function cancelling). -/
noncomputable def selWeight {E : ℕ} (z : Fin E → ℝ) (S : Finset (Fin E)) (e : Fin E) : ℝ :=
  Real.exp (z e) / ∑ s ∈ S, Real.exp (z s)

/-- The gate-weight vector of a router on a fixed selection. -/
noncomputable def gate {E d : ℕ} (R : Fin E → Fin d → ℝ) (S : Finset (Fin E))
    (x : Fin d → ℝ) : Fin E → ℝ :=
  selWeight (routerLogits R x) S

/-- The hard top-`k` mixture with an explicit selection function: sum the selected experts weighted by
the renormalized softmax over the selection. -/
noncomputable def moeWith {E d : ℕ} (R : Fin E → Fin d → ℝ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (sel : (Fin d → ℝ) → Finset (Fin E))
    (x : Fin d → ℝ) : Fin d → ℝ :=
  ∑ e ∈ sel x, gate R (sel x) x e • f e x

/-- The fixed-selection mixture. It is the extension of the stable-region formula used as
`moeStage.ref`. -/
noncomputable def fixedMoe {E d : ℕ} (R : Fin E → Fin d → ℝ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (S : Finset (Fin E))
    (x : Fin d → ℝ) : Fin d → ℝ :=
  ∑ e ∈ S, gate R S x e • f e x

theorem moeWith_eq_fixedMoe {E d : ℕ} (R : Fin E → Fin d → ℝ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (sel : (Fin d → ℝ) → Finset (Fin E))
    (S : Finset (Fin E)) (x : Fin d → ℝ) (h : sel x = S) :
    moeWith R f sel x = fixedMoe R f S x := by
  subst h
  rfl

/-! ## 2. The renormalized softmax is `2`-Lipschitz in `ℓ∞`/`ℓ1` -/

/-- The numeric bound `e < 3`, from the exponential series remainder (`Real.exp_bound'`, `n = 5`). -/
theorem exp_one_lt_three : Real.exp 1 < 3 := by
  have hb := Real.exp_bound' (x := 1) (by norm_num) (by norm_num) (n := 5) (by norm_num)
  norm_num [Finset.sum_range_succ, Nat.factorial] at hb
  linarith

/-- The scalar kernel of the softmax Lipschitz bound: `(e^{2t} − 1)/(e^{2t} + 1) ≤ t` for `t ≥ 0`.
This is `tanh t ≤ t`. For `t ≤ 1/2` the proof uses the second-order exponential upper bound
`Real.exp_bound'` (`n = 3`); for `1/2 ≤ t < 1` it uses the tangent-line bound
`Real.exp_bound_div_one_sub_of_interval` on the shifted exponent `2t − 1` together with `e < 3`; for
`t ≥ 1` the left side is nonpositive. -/
theorem exp_two_mul_sub_div_le (t : ℝ) (ht : 0 ≤ t) :
    (Real.exp (2 * t) - 1) / (Real.exp (2 * t) + 1) ≤ t := by
  have hpos : 0 < Real.exp (2 * t) + 1 := by positivity
  rw [div_le_iff₀ hpos]
  rcases le_or_gt t (1 / 2) with h | h
  · have hx1 : 0 ≤ 2 * t := by linarith
    have hx2 : 2 * t ≤ 1 := by linarith
    have hb := Real.exp_bound' (x := 2 * t) hx1 hx2 (n := 3) (by norm_num)
    have hs : (∑ m ∈ Finset.range 3, (2 * t) ^ m / (m.factorial : ℝ)) = 1 + 2 * t + 2 * t ^ 2 := by
      norm_num [Finset.sum_range_succ, Nat.factorial]
      ring
    have hc : (2 * t) ^ 3 * ((3 : ℕ) + 1) / ((3 : ℕ).factorial * (3 : ℕ)) = (16 / 9) * t ^ 3 := by
      norm_num [Nat.factorial]
      ring
    rw [hs, hc] at hb
    have h1t : 0 ≤ 1 - t := by linarith
    have hmult : Real.exp (2 * t) * (1 - t)
        ≤ (1 + 2 * t + 2 * t ^ 2 + (16 / 9) * t ^ 3) * (1 - t) :=
      mul_le_mul_of_nonneg_right hb h1t
    nlinarith [hmult]
  · rcases le_or_gt t 1 with h2 | h2
    · rcases lt_or_eq_of_le h2 with h2 | rfl
      · have hx1 : 0 ≤ 2 * t - 1 := by linarith
        have hx2 : 2 * t - 1 < 1 := by linarith
        have hb := Real.exp_bound_div_one_sub_of_interval hx1 hx2
        have h1t : 0 < 1 - t := by linarith
        have he : Real.exp (2 * t) = Real.exp (2 * t - 1) * Real.exp 1 := by
          rw [← Real.exp_add]
          congr 1
          ring
        have hstep : (1 - t) * Real.exp (2 * t) ≤ Real.exp 1 / 2 := by
          rw [he]
          have hb' : Real.exp (2 * t - 1) ≤ 1 / (1 - (2 * t - 1)) := hb
          have h1 : (1 - t) * Real.exp (2 * t - 1)
              ≤ (1 - t) * (1 / (1 - (2 * t - 1))) :=
            mul_le_mul_of_nonneg_left hb' (le_of_lt h1t)
          have h2' : (1 - t) * (1 / (1 - (2 * t - 1))) = 1 / 2 := by
            have hden : (1 : ℝ) - (2 * t - 1) = 2 * (1 - t) := by ring
            have hne : (2 : ℝ) * (1 - t) ≠ 0 :=
              mul_ne_zero (by norm_num) (ne_of_gt h1t)
            rw [hden]
            field_simp
          calc (1 - t) * (Real.exp (2 * t - 1) * Real.exp 1)
              = ((1 - t) * Real.exp (2 * t - 1)) * Real.exp 1 := by ring
            _ ≤ ((1 - t) * (1 / (1 - (2 * t - 1)))) * Real.exp 1 :=
                mul_le_mul_of_nonneg_right h1 (le_of_lt (Real.exp_pos _))
            _ = Real.exp 1 / 2 := by rw [h2']; ring
        have he2 : Real.exp 1 / 2 ≤ 1 + t := by
          have h3 : Real.exp 1 < 3 := exp_one_lt_three
          linarith
        nlinarith [hstep, he2]
      · linarith
    · have h1t : 1 - t < 0 := by linarith
      have h4 : Real.exp (2 * t) * (1 - t) ≤ 0 :=
        mul_nonpos_of_nonneg_of_nonpos (le_of_lt (Real.exp_pos _)) (le_of_lt h1t)
      linarith

/-- The scalar bound in the shape produced by `sum_abs_sub_le_of_bounds` for
`α = exp (−2ε)`, `β = exp (2ε)`. -/
theorem exp_ratio_bound (ε : ℝ) (hε : 0 ≤ ε) :
    (Real.exp (2 * ε) - 1) * (1 - Real.exp (-(2 * ε))) /
        ((Real.exp (2 * ε) - 1) + (1 - Real.exp (-(2 * ε)))) ≤ ε := by
  have heq : (Real.exp (2 * ε) - 1) * (1 - Real.exp (-(2 * ε))) /
        ((Real.exp (2 * ε) - 1) + (1 - Real.exp (-(2 * ε))))
      = (Real.exp (2 * ε) - 1) / (Real.exp (2 * ε) + 1) := by
    rw [Real.exp_neg]
    field_simp
  rw [heq]
  exact exp_two_mul_sub_div_le ε hε

/-- **Absolute-difference Lipschitz bound for weights with ratio bounds.** If `w, w'` are probability
weights on `S`, `w > 0` on `S`, and `α · w ≤ w' ≤ β · w` pointwise with `0 ≤ α ≤ 1 ≤ β`, then
`Σ_{e ∈ S} |w e − w' e| ≤ 2 (β−1)(1−α)/((β−1)+(1−α))`. The proof splits `S` by the sign of
`r = w'/w`, gets `R₊ ≤ (β−1) W₊`, `R₋ ≤ (1−α) W₋`, `W₊ + W₋ = 1`, and combines. -/
theorem sum_abs_sub_le_of_bounds {ι : Type*} (S : Finset ι) (w w' : ι → ℝ) (α β : ℝ)
    (hw : ∀ e ∈ S, 0 < w e) (hsum : ∑ e ∈ S, w e = 1) (hsum' : ∑ e ∈ S, w' e = 1)
    (hlower : ∀ e ∈ S, α * w e ≤ w' e) (hupper : ∀ e ∈ S, w' e ≤ β * w e)
    (hα1 : α ≤ 1) (hβ : 1 ≤ β) :
    ∑ e ∈ S, |w e - w' e| ≤ 2 * ((β - 1) * (1 - α) / ((β - 1) + (1 - α))) := by
  have hp : 0 ≤ β - 1 := by linarith
  have hq : 0 ≤ 1 - α := by linarith
  rcases S.eq_empty_or_nonempty with rfl | hSne
  · rw [Finset.sum_empty]
    have hd : 0 ≤ (β - 1) * (1 - α) / ((β - 1) + (1 - α)) :=
      div_nonneg (mul_nonneg hp hq) (by linarith)
    linarith
  set r : ι → ℝ := fun e => w' e / w e with hr
  have hr_eq : ∀ e ∈ S, w e * r e = w' e := by
    intro e he
    have hw0 : w e ≠ 0 := ne_of_gt (hw e he)
    have h : w e * (w' e / w e) = w e * w' e / w e := by ring
    simp only [hr]
    rw [h, mul_comm (w e) (w' e)]
    exact mul_div_cancel_right₀ (w' e) hw0
  have hr_lo : ∀ e ∈ S, α ≤ r e := by
    intro e he
    have h := hlower e he
    simp only [hr]
    rw [le_div_iff₀ (hw e he)]
    linarith
  have hr_hi : ∀ e ∈ S, r e ≤ β := by
    intro e he
    have h := hupper e he
    simp only [hr]
    rw [div_le_iff₀ (hw e he)]
    linarith
  have hsumr : ∑ e ∈ S, w e * r e = 1 := by
    rw [Finset.sum_congr rfl hr_eq, hsum']
  set Wp : ℝ := ∑ e ∈ S, if 1 ≤ r e then w e else 0 with hWp
  set Wn : ℝ := ∑ e ∈ S, if r e < 1 then w e else 0 with hWn
  have hWsum : Wp + Wn = 1 := by
    have h1 : Wp + Wn = ∑ e ∈ S, w e := by
      rw [hWp, hWn, ← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro e _
      by_cases h : 1 ≤ r e
      · rw [if_pos h, if_neg (not_lt.mpr h), add_zero]
      · rw [if_neg h, if_pos (not_le.mp h), zero_add]
    rw [h1, hsum]
  set Rpos : ℝ := ∑ e ∈ S, w e * max 0 (r e - 1) with hRpos
  set Rneg : ℝ := ∑ e ∈ S, w e * max 0 (1 - r e) with hRneg
  have hWpos : Rpos ≤ (β - 1) * Wp := by
    rw [hRpos, hWp, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro e he
    by_cases h : 1 ≤ r e
    · rw [if_pos h, max_eq_right (by linarith : (0 : ℝ) ≤ r e - 1)]
      have hh := hr_hi e he
      calc w e * (r e - 1) ≤ w e * (β - 1) :=
            mul_le_mul_of_nonneg_left (by linarith) (le_of_lt (hw e he))
        _ = (β - 1) * w e := by ring
    · have hlt : r e < 1 := not_le.mp h
      rw [if_neg h, mul_zero, max_eq_left (by linarith : r e - 1 ≤ 0), mul_zero]
  have hWneg : Rneg ≤ (1 - α) * Wn := by
    rw [hRneg, hWn, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro e he
    by_cases h : r e < 1
    · rw [if_pos h, max_eq_right (by linarith : (0 : ℝ) ≤ 1 - r e)]
      have hh := hr_lo e he
      calc w e * (1 - r e) ≤ w e * (1 - α) :=
            mul_le_mul_of_nonneg_left (by linarith) (le_of_lt (hw e he))
        _ = (1 - α) * w e := by ring
    · have hge : 1 ≤ r e := not_lt.mp h
      rw [if_neg h, mul_zero, max_eq_left (by linarith : 1 - r e ≤ 0), mul_zero]
  have hzero : ∑ e ∈ S, w e * (r e - 1) = 0 := by
    have h1 : ∑ e ∈ S, w e * (r e - 1) = (∑ e ∈ S, w e * r e) - ∑ e ∈ S, w e := by
      rw [← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro e _
      ring
    rw [h1, hsumr, hsum, sub_self]
  have hReq : Rpos = Rneg := by
    have h1 : Rpos - Rneg = ∑ e ∈ S, w e * (r e - 1) := by
      rw [hRpos, hRneg, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro e _
      rcases le_or_gt (r e) 1 with h | h
      · rw [max_eq_left (by linarith : r e - 1 ≤ 0),
          max_eq_right (by linarith : (0 : ℝ) ≤ 1 - r e)]
        ring
      · rw [max_eq_right (by linarith : (0 : ℝ) ≤ r e - 1),
          max_eq_left (by linarith : 1 - r e ≤ 0)]
        ring
    have h2 : Rpos - Rneg = 0 := by rw [h1, hzero]
    linarith
  have htotal : ∑ e ∈ S, |w e - w' e| = 2 * Rpos := by
    have h1 : ∑ e ∈ S, |w e - w' e|
        = ∑ e ∈ S, (w e * max 0 (1 - r e) + w e * max 0 (r e - 1)) := by
      apply Finset.sum_congr rfl
      intro e he
      have hw0 := hw e he
      have hprod : w e - w' e = w e * (1 - r e) := by
        rw [(hr_eq e he).symm]
        ring
      rw [hprod, abs_mul, abs_of_pos hw0]
      rcases le_or_gt (r e) 1 with h | h
      · rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ 1 - r e),
          max_eq_left (by linarith : r e - 1 ≤ 0),
          max_eq_right (by linarith : (0 : ℝ) ≤ 1 - r e)]
        ring
      · rw [abs_of_nonpos (by linarith : 1 - r e ≤ 0),
          max_eq_right (by linarith : (0 : ℝ) ≤ r e - 1),
          max_eq_left (by linarith : 1 - r e ≤ 0)]
        ring
    calc ∑ e ∈ S, |w e - w' e|
        = ∑ e ∈ S, (w e * max 0 (1 - r e) + w e * max 0 (r e - 1)) := h1
      _ = (∑ e ∈ S, w e * max 0 (1 - r e)) + (∑ e ∈ S, w e * max 0 (r e - 1)) :=
          Finset.sum_add_distrib
      _ = Rneg + Rpos := by rw [hRneg, hRpos]
      _ = 2 * Rpos := by rw [← hReq]; ring
  have hRle : Rpos * ((β - 1) + (1 - α)) ≤ (β - 1) * (1 - α) := by
    have e1 : Rpos * (1 - α) ≤ (β - 1) * Wp * (1 - α) := mul_le_mul_of_nonneg_right hWpos hq
    have e2 : Rneg * (β - 1) ≤ (1 - α) * Wn * (β - 1) := mul_le_mul_of_nonneg_right hWneg hp
    rw [← hReq] at e2
    have e3 : (β - 1) * Wp * (1 - α) + (1 - α) * Wn * (β - 1) = (β - 1) * (1 - α) := by
      have h2 : (β - 1) * (1 - α) = (β - 1) * (1 - α) * (Wp + Wn) := by rw [hWsum, mul_one]
      rw [h2]
      ring
    linarith [e1, e2, e3]
  have hfin : Rpos ≤ (β - 1) * (1 - α) / ((β - 1) + (1 - α)) := by
    rcases eq_or_lt_of_le (by linarith : (0 : ℝ) ≤ (β - 1) + (1 - α)) with h0 | h0
    · have hb1 : β - 1 = 0 := by linarith [h0, hq]
      have hz : (β - 1) * (1 - α) / ((β - 1) + (1 - α)) = 0 := by
        rw [hb1, zero_mul, zero_div]
      rw [hz]
      calc Rpos ≤ (β - 1) * Wp := hWpos
        _ = 0 := by rw [hb1, zero_mul]
    · rw [le_div_iff₀ h0]
      exact hRle
  rw [htotal]
  linarith

/-- Positivity of the gate weight at any expert when the selection is nonempty. -/
theorem selWeight_pos {E : ℕ} (z : Fin E → ℝ) (S : Finset (Fin E)) (hSne : S.Nonempty)
    (e : Fin E) : 0 < selWeight z S e :=
  div_pos (Real.exp_pos _) (Finset.sum_pos (fun _ _ => Real.exp_pos _) hSne)

/-- Positivity of a gate weight of the router on a fixed selection. -/
theorem gate_pos {E d : ℕ} (R : Fin E → Fin d → ℝ) (S : Finset (Fin E)) (hSne : S.Nonempty)
    (x : Fin d → ℝ) (e : Fin E) : 0 < gate R S x e :=
  selWeight_pos _ S hSne e

/-- The gate weights over a nonempty selection sum to one. -/
theorem sum_selWeight {E : ℕ} (z : Fin E → ℝ) (S : Finset (Fin E)) (hSne : S.Nonempty) :
    ∑ e ∈ S, selWeight z S e = 1 := by
  unfold selWeight
  rw [← Finset.sum_div]
  exact div_self (ne_of_gt (Finset.sum_pos (fun _ _ => Real.exp_pos _) hSne))

/-- **Gate-weight Lipschitz bound.** The renormalized softmax over a nonempty selection is
`2`-Lipschitz from the `ℓ∞` norm on the logits to the `ℓ1` norm on the weights. -/
theorem selWeight_sum_abs_le {E : ℕ} (S : Finset (Fin E)) (hSne : S.Nonempty)
    (z z' : Fin E → ℝ) :
    ∑ e ∈ S, |selWeight z S e - selWeight z' S e| ≤ 2 * ‖z - z'‖ := by
  have hA : 0 < ∑ s ∈ S, Real.exp (z s) := Finset.sum_pos (fun _ _ => Real.exp_pos _) hSne
  have hB : 0 < ∑ s ∈ S, Real.exp (z' s) := Finset.sum_pos (fun _ _ => Real.exp_pos _) hSne
  have hcoord : ∀ e, |z e - z' e| ≤ ‖z - z'‖ := by
    intro e
    have h := norm_le_pi_norm (z - z') e
    rwa [Pi.sub_apply, Real.norm_eq_abs] at h
  have hzl : ∀ e, z e ≤ z' e + ‖z - z'‖ := fun e => by
    have := (abs_le.mp (hcoord e)).2; linarith
  have hzr : ∀ e, z' e ≤ z e + ‖z - z'‖ := fun e => by
    have := (abs_le.mp (hcoord e)).1; linarith
  have hAle : ∑ s ∈ S, Real.exp (z s) ≤ Real.exp ‖z - z'‖ * ∑ s ∈ S, Real.exp (z' s) := by
    calc ∑ s ∈ S, Real.exp (z s)
        ≤ ∑ s ∈ S, Real.exp (z' s + ‖z - z'‖) := by
          apply Finset.sum_le_sum
          intro s _
          exact Real.exp_le_exp.mpr (hzl s)
      _ = ∑ s ∈ S, Real.exp (z' s) * Real.exp ‖z - z'‖ := by
          apply Finset.sum_congr rfl
          intro s _
          rw [Real.exp_add]
      _ = (∑ s ∈ S, Real.exp (z' s)) * Real.exp ‖z - z'‖ := by rw [Finset.sum_mul]
      _ = Real.exp ‖z - z'‖ * ∑ s ∈ S, Real.exp (z' s) := by ring
  have hBle : ∑ s ∈ S, Real.exp (z' s) ≤ Real.exp ‖z - z'‖ * ∑ s ∈ S, Real.exp (z s) := by
    calc ∑ s ∈ S, Real.exp (z' s)
        ≤ ∑ s ∈ S, Real.exp (z s + ‖z - z'‖) := by
          apply Finset.sum_le_sum
          intro s _
          exact Real.exp_le_exp.mpr (hzr s)
      _ = ∑ s ∈ S, Real.exp (z s) * Real.exp ‖z - z'‖ := by
          apply Finset.sum_congr rfl
          intro s _
          rw [Real.exp_add]
      _ = (∑ s ∈ S, Real.exp (z s)) * Real.exp ‖z - z'‖ := by rw [Finset.sum_mul]
      _ = Real.exp ‖z - z'‖ * ∑ s ∈ S, Real.exp (z s) := by ring
  have hupper : ∀ e ∈ S, selWeight z' S e ≤ Real.exp (2 * ‖z - z'‖) * selWeight z S e := by
    intro e _
    have hclear : Real.exp (z' e) * (∑ s ∈ S, Real.exp (z s))
        ≤ (Real.exp (2 * ‖z - z'‖) * Real.exp (z e)) * ∑ s ∈ S, Real.exp (z' s) := by
      calc Real.exp (z' e) * ∑ s ∈ S, Real.exp (z s)
          ≤ Real.exp (z' e) * (Real.exp ‖z - z'‖ * ∑ s ∈ S, Real.exp (z' s)) :=
            mul_le_mul_of_nonneg_left hAle (le_of_lt (Real.exp_pos _))
        _ = Real.exp (z' e + ‖z - z'‖) * ∑ s ∈ S, Real.exp (z' s) := by
            rw [Real.exp_add]; ring
        _ ≤ Real.exp (z e + ‖z - z'‖ + ‖z - z'‖) * ∑ s ∈ S, Real.exp (z' s) := by
            apply mul_le_mul_of_nonneg_right _ (le_of_lt hB)
            exact Real.exp_le_exp.mpr (by linarith [hzr e])
        _ = (Real.exp (2 * ‖z - z'‖) * Real.exp (z e)) * ∑ s ∈ S, Real.exp (z' s) := by
            rw [show z e + ‖z - z'‖ + ‖z - z'‖ = 2 * ‖z - z'‖ + z e by ring, Real.exp_add]
    rw [selWeight, selWeight]
    rw [show Real.exp (2 * ‖z - z'‖) * (Real.exp (z e) / ∑ s ∈ S, Real.exp (z s))
        = Real.exp (2 * ‖z - z'‖) * Real.exp (z e) / ∑ s ∈ S, Real.exp (z s) by ring]
    rw [div_le_div_iff₀ hB hA]
    exact hclear
  have hlower : ∀ e ∈ S,
      Real.exp (-(2 * ‖z - z'‖)) * selWeight z S e ≤ selWeight z' S e := by
    intro e _
    have hclear : (Real.exp (-(2 * ‖z - z'‖)) * Real.exp (z e)) * (∑ s ∈ S, Real.exp (z' s))
        ≤ Real.exp (z' e) * ∑ s ∈ S, Real.exp (z s) := by
      calc (Real.exp (-(2 * ‖z - z'‖)) * Real.exp (z e)) * ∑ s ∈ S, Real.exp (z' s)
          ≤ (Real.exp (-(2 * ‖z - z'‖)) * Real.exp (z e))
              * (Real.exp ‖z - z'‖ * ∑ s ∈ S, Real.exp (z s)) :=
            mul_le_mul_of_nonneg_left hBle (by positivity)
        _ = Real.exp (z e - ‖z - z'‖) * ∑ s ∈ S, Real.exp (z s) := by
            rw [show z e - ‖z - z'‖ = z e + -‖z - z'‖ by ring, Real.exp_add,
              show -‖z - z'‖ = -(2 * ‖z - z'‖) + ‖z - z'‖ by ring, Real.exp_add]
            ring
        _ ≤ Real.exp (z' e) * ∑ s ∈ S, Real.exp (z s) := by
            apply mul_le_mul_of_nonneg_right _ (le_of_lt hA)
            exact Real.exp_le_exp.mpr (by linarith [hzl e])
    rw [selWeight, selWeight]
    rw [show Real.exp (-(2 * ‖z - z'‖)) * (Real.exp (z e) / ∑ s ∈ S, Real.exp (z s))
        = Real.exp (-(2 * ‖z - z'‖)) * Real.exp (z e) / ∑ s ∈ S, Real.exp (z s) by ring]
    rw [div_le_div_iff₀ hA hB]
    exact hclear
  have hmain := sum_abs_sub_le_of_bounds S (selWeight z S) (selWeight z' S)
    (Real.exp (-(2 * ‖z - z'‖))) (Real.exp (2 * ‖z - z'‖))
    (fun e _ => selWeight_pos z S hSne e) (sum_selWeight z S hSne) (sum_selWeight z' S hSne)
    hlower hupper
    (by rw [← Real.exp_zero, Real.exp_le_exp]; linarith [norm_nonneg (z - z')])
    (by rw [← Real.exp_zero, Real.exp_le_exp]; linarith [norm_nonneg (z - z')])
  have hfin := exp_ratio_bound ‖z - z'‖ (norm_nonneg _)
  linarith [hmain, hfin]

/-! ## 3. The fixed-selection mixture is globally Lipschitz -/

/-- **Global Lipschitz bound for the fixed-selection mixture.** With `Lmax`, `Mmax` bounding the
expert Lipschitz constants and expert norms on `S`, the fixed-`S` mixture is Lipschitz with constant
`Lmax + 2·Mmax·ρ` — the term `2·Mmax·ρ` is the gate-weight variation (`selWeight_sum_abs_le`) times
the router expansion. -/
theorem fixedMoe_lipschitz {E d : ℕ} (R : Fin E → Fin d → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hR : ∀ e, ∑ j, |R e j| ≤ ρ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (Lf Mf : Fin E → ℝ)
    (hfM : ∀ e x, ‖f e x‖ ≤ Mf e) (hfL : ∀ e x y, ‖f e x - f e y‖ ≤ Lf e * ‖x - y‖)
    (S : Finset (Fin E)) (hSne : S.Nonempty) (Lmax Mmax : ℝ) (hMmax : 0 ≤ Mmax)
    (hLf' : ∀ e ∈ S, Lf e ≤ Lmax) (hMf' : ∀ e ∈ S, Mf e ≤ Mmax) (x y : Fin d → ℝ) :
    ‖fixedMoe R f S x - fixedMoe R f S y‖ ≤ (Lmax + 2 * Mmax * ρ) * ‖x - y‖ := by
  have hW : 0 ≤ ‖x - y‖ := norm_nonneg _
  have hsum : ∑ e ∈ S, gate R S x e = 1 := sum_selWeight (routerLogits R x) S hSne
  have hsoft : ∑ e ∈ S, |gate R S x e - gate R S y e| ≤ 2 * (ρ * ‖x - y‖) := by
    have h1 := selWeight_sum_abs_le S hSne (routerLogits R x) (routerLogits R y)
    have h2 : ‖routerLogits R x - routerLogits R y‖ ≤ ρ * ‖x - y‖ :=
      routerLogits_lipschitz R ρ hρ hR x y
    simp only [gate]
    linarith
  have hdecomp : fixedMoe R f S x - fixedMoe R f S y
      = (∑ e ∈ S, gate R S x e • (f e x - f e y))
        + (∑ e ∈ S, (gate R S x e - gate R S y e) • f e y) := by
    simp only [fixedMoe]
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro e _
    rw [smul_sub, sub_smul]
    abel
  rw [hdecomp]
  have hA : ‖∑ e ∈ S, gate R S x e • (f e x - f e y)‖ ≤ Lmax * ‖x - y‖ := by
    refine le_trans (norm_sum_le _ _) ?_
    calc ∑ e ∈ S, ‖gate R S x e • (f e x - f e y)‖
        = ∑ e ∈ S, gate R S x e * ‖f e x - f e y‖ := by
          apply Finset.sum_congr rfl
          intro e _
          rw [norm_smul, Real.norm_of_nonneg (le_of_lt (gate_pos R S hSne x e))]
      _ ≤ ∑ e ∈ S, gate R S x e * (Lmax * ‖x - y‖) := by
          apply Finset.sum_le_sum
          intro e he
          apply mul_le_mul_of_nonneg_left _ (le_of_lt (gate_pos R S hSne x e))
          calc ‖f e x - f e y‖ ≤ Lf e * ‖x - y‖ := hfL e x y
            _ ≤ Lmax * ‖x - y‖ := mul_le_mul_of_nonneg_right (hLf' e he) hW
      _ = (∑ e ∈ S, gate R S x e) * (Lmax * ‖x - y‖) := by rw [Finset.sum_mul]
      _ = Lmax * ‖x - y‖ := by rw [hsum, one_mul]
  have hB : ‖∑ e ∈ S, (gate R S x e - gate R S y e) • f e y‖
      ≤ 2 * Mmax * ρ * ‖x - y‖ := by
    refine le_trans (norm_sum_le _ _) ?_
    calc ∑ e ∈ S, ‖(gate R S x e - gate R S y e) • f e y‖
        = ∑ e ∈ S, |gate R S x e - gate R S y e| * ‖f e y‖ := by
          apply Finset.sum_congr rfl
          intro e _
          rw [norm_smul, Real.norm_eq_abs]
      _ ≤ ∑ e ∈ S, |gate R S x e - gate R S y e| * Mmax := by
          apply Finset.sum_le_sum
          intro e he
          apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
          exact le_trans (hfM e y) (hMf' e he)
      _ = (∑ e ∈ S, |gate R S x e - gate R S y e|) * Mmax := by rw [Finset.sum_mul]
      _ ≤ (2 * (ρ * ‖x - y‖)) * Mmax := mul_le_mul_of_nonneg_right hsoft hMmax
      _ = 2 * Mmax * ρ * ‖x - y‖ := by ring
  calc ‖(∑ e ∈ S, gate R S x e • (f e x - f e y))
        + (∑ e ∈ S, (gate R S x e - gate R S y e) • f e y)‖
      ≤ ‖∑ e ∈ S, gate R S x e • (f e x - f e y)‖
        + ‖∑ e ∈ S, (gate R S x e - gate R S y e) • f e y‖ := norm_add_le _ _
    _ ≤ Lmax * ‖x - y‖ + 2 * Mmax * ρ * ‖x - y‖ := add_le_add hA hB
    _ = (Lmax + 2 * Mmax * ρ) * ‖x - y‖ := by ring

theorem le_sup'_of_mem {ι : Type*} (S : Finset ι) (hS : S.Nonempty) (f : ι → ℝ) {e : ι}
    (he : e ∈ S) : f e ≤ S.sup' hS f :=
  Finset.le_sup' f he

theorem sup'_nonneg {ι : Type*} (S : Finset ι) (hS : S.Nonempty) (f : ι → ℝ)
    (hf : ∀ e ∈ S, 0 ≤ f e) : 0 ≤ S.sup' hS f := by
  have ⟨e, he⟩ := hS
  exact le_trans (hf e he) (Finset.le_sup' f he)

/-- **Stable-region Lipschitz bound.** On a domain `D` on which the router selects the same `S` for
every input, the hard mixture equals the fixed-`S` mixture and the bound holds with constant
`max_{e∈S} L_e + 2 · max_{e∈S} M_e · ρ`. -/
theorem moe_lipschitz_on_stable {E d : ℕ} (R : Fin E → Fin d → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hR : ∀ e, ∑ j, |R e j| ≤ ρ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (Lf Mf : Fin E → ℝ)
    (hMf : ∀ e, 0 ≤ Mf e) (hfM : ∀ e x, ‖f e x‖ ≤ Mf e)
    (hfL : ∀ e x y, ‖f e x - f e y‖ ≤ Lf e * ‖x - y‖)
    (k : ℕ) (S : Finset (Fin E)) (hSne : S.Nonempty) (sel : (Fin d → ℝ) → Finset (Fin E))
    (D : Set (Fin d → ℝ)) (hsel : ∀ x ∈ D, sel x = S)
    (_hS : ∀ x ∈ D, IsTopKSet (routerLogits R x) k S)
    (x y : Fin d → ℝ) (hx : x ∈ D) (hy : y ∈ D) :
    ‖moeWith R f sel x - moeWith R f sel y‖
      ≤ (S.sup' hSne Lf + 2 * S.sup' hSne Mf * ρ) * ‖x - y‖ := by
  rw [moeWith_eq_fixedMoe R f sel S x (hsel x hx),
    moeWith_eq_fixedMoe R f sel S y (hsel y hy)]
  exact fixedMoe_lipschitz R ρ hρ hR f Lf Mf hfM hfL S hSne (S.sup' hSne Lf) (S.sup' hSne Mf)
    (sup'_nonneg S hSne Mf (fun e _ => hMf e)) (fun e he => le_sup'_of_mem S hSne Lf he)
    (fun e he => le_sup'_of_mem S hSne Mf he) x y

/-! ## 4. Margin stability of the selection -/

/-- **Margin stability.** If at `x` every selected logit exceeds every unselected logit by more than
`2ρr`, then every `y` within `‖y − x‖ ≤ r` has the same top-`k` selection `S`. The margin is the
selected-minus-unselected logit gap (in particular a tie at `x` never satisfies it). -/
theorem margin_stable {E d : ℕ} (R : Fin E → Fin d → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hR : ∀ e, ∑ j, |R e j| ≤ ρ) (k : ℕ) (S : Finset (Fin E)) (x : Fin d → ℝ)
    (hx : IsTopKSet (routerLogits R x) k S) (r : ℝ)
    (hgap : ∀ i ∈ S, ∀ j ∉ S, 2 * ρ * r < routerLogits R x i - routerLogits R x j) :
    ∀ y : Fin d → ℝ, ‖y - x‖ ≤ r → IsTopKSet (routerLogits R y) k S := by
  intro y hy
  refine ⟨hx.1, ?_⟩
  intro i hi j hj
  left
  have hcoord : ∀ e, |routerLogits R y e - routerLogits R x e| ≤ ρ * r := by
    intro e
    have h1 : ‖routerLogits R y - routerLogits R x‖ ≤ ρ * ‖y - x‖ :=
      routerLogits_lipschitz R ρ hρ hR y x
    have h2 : ρ * ‖y - x‖ ≤ ρ * r := mul_le_mul_of_nonneg_left hy hρ
    have h3 := norm_le_pi_norm (routerLogits R y - routerLogits R x) e
    rw [Pi.sub_apply, Real.norm_eq_abs] at h3
    linarith
  have hj' : routerLogits R y j - routerLogits R x j ≤ ρ * r := (abs_le.mp (hcoord j)).2
  have hi' : routerLogits R x i - routerLogits R y i ≤ ρ * r := by
    have := (abs_le.mp (hcoord i)).1
    linarith
  have hg := hgap i hi j hj
  have : routerLogits R y j - routerLogits R y i < 0 := by linarith
  linarith

/-! ## 5. The mixture as a `Stage` -/

/-- The hard mixture as a block `Stage`. `ref` is the fixed-`S` mixture (globally Lipschitz with
constant `Lmax + 2·Mmax·ρ`, `fixedMoe_lipschitz`, hence a legal global `expansion`), `dom` is the
stable region `D` and `exec` is the hard mixture. On `D` the two coincide (`hsel`), so `defect` is `0`.
**Off `dom` the `ref` is the fixed-`S` mixture, not the hard router** — `Stage.expansion` speaks about
`ref` only. -/
noncomputable def moeStage {E d : ℕ} (R : Fin E → Fin d → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hR : ∀ e, ∑ j, |R e j| ≤ ρ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (Lf Mf : Fin E → ℝ)
    (hfM : ∀ e x, ‖f e x‖ ≤ Mf e) (hfL : ∀ e x y, ‖f e x - f e y‖ ≤ Lf e * ‖x - y‖)
    (S : Finset (Fin E)) (hSne : S.Nonempty) (Lmax Mmax : ℝ)
    (hLmax : 0 ≤ Lmax) (hMmax : 0 ≤ Mmax)
    (hLf' : ∀ e ∈ S, Lf e ≤ Lmax) (hMf' : ∀ e ∈ S, Mf e ≤ Mmax)
    (sel : (Fin d → ℝ) → Finset (Fin E)) (D : Set (Fin d → ℝ))
    (hsel : ∀ x ∈ D, sel x = S) :
    Stage (Fin d → ℝ) (Fin d → ℝ) where
  ref := fixedMoe R f S
  exec := moeWith R f sel
  dom := D
  delta := 0
  L := Lmax + 2 * Mmax * ρ
  delta_nonneg := le_refl 0
  L_nonneg := by nlinarith [mul_nonneg hMmax hρ]
  defect := by
    intro z hz
    rw [moeWith_eq_fixedMoe R f sel S z (hsel z hz)]
    simp
  expansion := by
    intro z w
    exact fixedMoe_lipschitz R ρ hρ hR f Lf Mf hfM hfL S hSne Lmax Mmax hMmax hLf' hMf' z w

/-- The `Stage`'s expansion factor is `Lmax + 2·Mmax·ρ`, exactly the stable-region constant. -/
@[simp] theorem moeStage_L {E d : ℕ} (R : Fin E → Fin d → ℝ) (ρ : ℝ) (hρ : 0 ≤ ρ)
    (hR : ∀ e, ∑ j, |R e j| ≤ ρ)
    (f : Fin E → (Fin d → ℝ) → (Fin d → ℝ)) (Lf Mf : Fin E → ℝ)
    (hfM : ∀ e x, ‖f e x‖ ≤ Mf e) (hfL : ∀ e x y, ‖f e x - f e y‖ ≤ Lf e * ‖x - y‖)
    (S : Finset (Fin E)) (hSne : S.Nonempty) (Lmax Mmax : ℝ)
    (hLmax : 0 ≤ Lmax) (hMmax : 0 ≤ Mmax)
    (hLf' : ∀ e ∈ S, Lf e ≤ Lmax) (hMf' : ∀ e ∈ S, Mf e ≤ Mmax)
    (sel : (Fin d → ℝ) → Finset (Fin E)) (D : Set (Fin d → ℝ))
    (hsel : ∀ x ∈ D, sel x = S) :
    (moeStage R ρ hρ hR f Lf Mf hfM hfL S hSne Lmax Mmax hLmax hMmax hLf' hMf' sel D hsel).L
      = Lmax + 2 * Mmax * ρ := rfl

/-! ## 6. The counterexample: hard top-`k` routing is discontinuous -/

/-- The linear router of the counterexample: expert `0` reads the input, expert `1` reads `0`. -/
def cxR : Fin 2 → Fin 1 → ℝ := ![fun _ => (1 : ℝ), fun _ => 0]

/-- The experts of the counterexample: expert `0` is the zero map, expert `1` is the constant `1`. -/
def cxF : Fin 2 → (Fin 1 → ℝ) → (Fin 1 → ℝ) :=
  ![fun _ _ => (0 : ℝ), fun _ _ => (1 : ℝ)]

/-- The exact top-`1` selection (`k = 1`, ties to the lower index): `{1}` for `x 0 < 0`, else `{0}`. -/
noncomputable def cxSel (x : Fin 1 → ℝ) : Finset (Fin 2) := if x 0 < 0 then {1} else {0}

/-- The counterexample mixture: `moe` with the exact top-`1` routing. -/
noncomputable def cxMoe : (Fin 1 → ℝ) → (Fin 1 → ℝ) := moeWith cxR cxF cxSel

theorem cxSel_isTopK (x : Fin 1 → ℝ) : IsTopKSet (routerLogits cxR x) 1 (cxSel x) := by
  by_cases h : x 0 < 0
  · refine ⟨by simp [cxSel, h], ?_⟩
    intro i hi j hj
    rw [cxSel, if_pos h] at hi hj
    have hi1 : i = 1 := Finset.mem_singleton.mp hi
    have hj0 : j = 0 := by
      have hj1 : j ≠ 1 := fun hj1 => hj (Finset.mem_singleton.mpr hj1)
      fin_cases j <;> simp_all
    subst hi1
    subst hj0
    left
    simpa [routerLogits, cxR] using h
  · have hx0 : 0 ≤ x 0 := le_of_not_gt h
    refine ⟨by simp [cxSel, h], ?_⟩
    intro i hi j hj
    rw [cxSel, if_neg h] at hi hj
    have hi0 : i = 0 := Finset.mem_singleton.mp hi
    have hj1 : j = 1 := by
      have hj0 : j ≠ 0 := fun hj0 => hj (Finset.mem_singleton.mpr hj0)
      fin_cases j <;> simp_all
    subst hi0
    subst hj1
    rcases lt_or_eq_of_le hx0 with hlt | heq
    · left
      simpa [routerLogits, cxR] using hlt
    · right
      constructor
      · simpa [routerLogits, cxR] using heq
      · decide

theorem cxMoe_of_neg {x : Fin 1 → ℝ} (h : x 0 < 0) : cxMoe x = fun _ => (1 : ℝ) := by
  have h1 : cxSel x = {1} := by simp [cxSel, h]
  have hw : gate cxR ({1} : Finset (Fin 2)) x 1 = 1 := by
    rw [gate, selWeight, Finset.sum_singleton]
    exact div_self (ne_of_gt (Real.exp_pos _))
  rw [cxMoe, moeWith, h1, Finset.sum_singleton]
  rw [hw, one_smul]
  simp [cxF]

theorem cxMoe_of_nonneg {x : Fin 1 → ℝ} (h : 0 ≤ x 0) : cxMoe x = fun _ => (0 : ℝ) := by
  have hn : ¬ x 0 < 0 := by linarith
  have h0 : cxSel x = {0} := by simp [cxSel, hn]
  have hw : gate cxR ({0} : Finset (Fin 2)) x 0 = 1 := by
    rw [gate, selWeight, Finset.sum_singleton]
    exact div_self (ne_of_gt (Real.exp_pos _))
  rw [cxMoe, moeWith, h0, Finset.sum_singleton]
  rw [hw, one_smul]
  simp [cxF]

/-- For a one-dimensional `Fin 1 → ℝ`, the sup norm is the coordinate absolute value. -/
theorem norm_fin_one (g : Fin 1 → ℝ) : ‖g‖ = |g 0| := by
  refine le_antisymm ?_ ?_
  · refine (pi_norm_le_iff_of_nonneg (abs_nonneg (g 0))).mpr ?_
    intro i
    fin_cases i
    simp
  · rw [← Real.norm_eq_abs]
    exact norm_le_pi_norm g 0

/-- **Hard top-`k` routing is discontinuous: no global Lipschitz constant exists.** The concrete
two-expert one-dimensional instance `cxR`/`cxF`/`cxSel` (exact top-`1` selection) jumps by `1` across
the tie boundary at `0`, so no `L` bounds `‖moe x − moe y‖` by `L‖x − y‖`. -/
theorem moe_not_continuous :
    ¬ ∃ L : ℝ, ∀ x y : Fin 1 → ℝ, ‖cxMoe x - cxMoe y‖ ≤ L * ‖x - y‖ := by
  rintro ⟨L, hL⟩
  set c : ℝ := 1 / (|L| + 1) with hc
  have hden : 0 < |L| + 1 := by positivity
  have hcpos : 0 < c := by rw [hc]; positivity
  have hneg : (fun _ : Fin 1 => -c) 0 < 0 := by
    simpa using neg_lt_zero.mpr hcpos
  have hnonneg : 0 ≤ (fun _ : Fin 1 => (0 : ℝ)) 0 := le_refl 0
  have hval := hL (fun _ : Fin 1 => -c) (fun _ : Fin 1 => (0 : ℝ))
  rw [cxMoe_of_neg hneg, cxMoe_of_nonneg hnonneg] at hval
  have hnorm : ‖(fun _ : Fin 1 => (1 : ℝ)) - (fun _ : Fin 1 => (0 : ℝ))‖ = 1 := by
    rw [norm_fin_one]
    simp
  have hdist : ‖(fun _ : Fin 1 => -c) - (fun _ : Fin 1 => (0 : ℝ))‖ = c := by
    rw [norm_fin_one]
    have h0 : ((fun _ : Fin 1 => -c) - (fun _ : Fin 1 => (0 : ℝ))) 0 = -c := by simp
    rw [h0, abs_of_neg (by linarith)]
    ring
  rw [hnorm, hdist] at hval
  have hbound : L * c < 1 := by
    have h2 : |L| * c = |L| / (|L| + 1) := by rw [hc]; ring
    have h3 : |L| / (|L| + 1) < 1 := by
      rw [div_lt_one hden]
      linarith [abs_nonneg L]
    nlinarith [le_abs_self L, h2, h3]
  linarith

end Nous.Model
