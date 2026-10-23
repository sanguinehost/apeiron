import Learn.QuantVerdicts


/-!
# NanoQuant on the crate: budget, measured quality and the tune's cost equation

The port is `src/learn/nanoquant.rs` (`SamsungLabs/NanoQuant` @ `a9e0a43`); the epic is
the nanoquant-subsume note. Every measured figure is a `model` input copied from its log under
`lead-1023/nq/`: relH `tr(EHEᵀ)/tr(WHWᵀ)` in millionths, a share in
millionths. The cost theorems count multiply-adds of the stated equations; they are `model` grade, not a
wall-clock measurement. No clause here is linked to an extracted body: `rankForBits` and `storedBits`
restate `nanoquant::rank_for_bits` and `stored_bits` over `Nat` (the Aeneas link is open, W4). rustc, the
`limb` kernel and its error bound are outside every statement.

Fixture: Qwen3.5-0.8B `blk.23.ffn_down`, `o = 1024`, `n = 3584`, the cached `c6x29` calibration Gram.
-/

namespace Learn

namespace NanoQuant

/-! ## 1 · Budget -/

/-- `nanoquant::rank_for_bits`: `⌊a·b·ppm / (10⁶·(a + b))⌋ − 16` (saturating), floored to a multiple of 32,
and `minRank` when that is smaller. -/
def rankForBits (a b ppm minRank : Nat) : Nat :=
  max ((a * b * ppm / (1000000 * (a + b)) - 16) / 32 * 32) minRank

/-- `nanoquant::stored_bits`: `r` sign rows on both sides plus two 16-bit scale vectors, `(r + 16)(a + b)`. -/
def storedBits (a b r : Nat) : Nat := (r + 16) * (a + b)

/-- **The l23 `down` arm is rank 896 at 1.1451 b/w and fits 1.18**: the 1.175 b/w budget floors to `r = 896`,
whose `912 · 4608 = 4 202 496` stored bits are between 1.1450 and 1.1451 bits per weight of the
`1024 · 3584` matrix, under 1.18. `∧` [both are true]; `≤` [is at most]. `<` [is smaller than]. -/
theorem rank_l23_down :
    rankForBits 1024 3584 1175000 32 = 896 ∧
    storedBits 1024 3584 896 = 4202496 ∧
    100 * storedBits 1024 3584 896 ≤ 118 * (1024 * 3584) ∧
    11450 * (1024 * 3584) < 10000 * storedBits 1024 3584 896 ∧
    10000 * storedBits 1024 3584 896 < 11451 * (1024 * 3584) := by
  decide

/-! ## 2 · Measured relH, l23 `down` (`nq_fit`, `w2a-micro.log`, `w2b-1024.log`) -/

/-- Q4_K_M on the same tensor, relH millionths (0.000151). -/
def relQ4KM : Nat := 151
/-- NanoQuant ADMM, linear schedule, 400 iterations (0.02446). -/
def relAdmm : Nat := 24460
/-- ADMM plus the layer-scope STE tune, 1024 steps (W2, 0.01727). -/
def relW2 : Nat := 17270
/-- The c6x29 gate code at 1.175 b/w (0.019483). -/
def relC6 : Nat := 19483
/-- The c6x29-sel gate code at 1.175 b/w (0.012146). -/
def relSel : Nat := 12146

/-- **W2 beats ADMM and c6x29, loses to c6x29-sel, and is 114× Q4_K_M**: relH 0.01727 against ADMM 0.02446,
c6x29 0.019483 and c6x29-sel 0.012146, all at ≤ 1.18 b/w; Q4_K_M's 0.000151 is 4.75 b/w. `∧` [both are
true]; `<` [is smaller than]. `∧` [both are true]. -/
theorem w2_relH :
    relW2 < relAdmm ∧ relW2 < relC6 ∧ relSel < relW2 ∧
    114 * relQ4KM < relW2 ∧ relW2 < 115 * relQ4KM := by
  decide

/-! ## 3 · Tune cost per step (multiply-adds, `model`) -/

/-- Full-`H` tune step: `E·H` (`o·n²`) plus `U·V`, `∂U`, `∂V` (`3·o·n·r`). -/
def tuneStep (o n r : Nat) : Nat := o * n * n + 3 * o * n * r

/-- Sketched tune step on `H ≈ Q·H_s·Qᵀ` (rank `s`): `P = V·diag(s_pre)·Q`, `U·P` and the two gradients
(`r·s·3(o + n)`), `E_Q·H_s` (`o·s²`). -/
def sketchStep (o n r s : Nat) : Nat := r * s * 3 * (o + n) + o * s * s

/-- **The full-`H` step is 23.0 GMAC at l23 `down`; the sketch at `s = 256` is 7–8× cheaper, at `s = 128`
14–15×.** The measured step is 0.39 s (59 GMAC/s, i9-14900KF). `∧` [both are true]. `<` [is smaller than]. -/
theorem step_cost_l23_down :
    tuneStep 1024 3584 896 = 23018340352 ∧
    7 * sketchStep 1024 3584 896 256 < tuneStep 1024 3584 896 ∧
    tuneStep 1024 3584 896 < 8 * sketchStep 1024 3584 896 256 ∧
    14 * sketchStep 1024 3584 896 128 < tuneStep 1024 3584 896 ∧
    tuneStep 1024 3584 896 < 15 * sketchStep 1024 3584 896 128 := by
  decide

/-- **Any step that forms `E` keeps the `3·o·n·r` term, so removing `E·H` alone saves at most `1 + n/(3r)`**:
2.33× at the 0.8B `down`, 2.29× at the Qwen3.8-27B `down` (`o = 5120`, `n = 17408`, `r = 4512`), 1.38× at its
`gate`/`up` (`n = 5120`). This is the ceiling of a sketch-plus-diagonal metric. `∧` [both are true]. `<` [is smaller than]. -/
theorem formE_ceiling :
    233 * (3 * 1024 * 3584 * 896) < 100 * tuneStep 1024 3584 896 ∧
    100 * tuneStep 1024 3584 896 < 234 * (3 * 1024 * 3584 * 896) ∧
    2286 * (3 * 5120 * 17408 * 4512) < 1000 * tuneStep 5120 17408 4512 ∧
    1000 * tuneStep 5120 17408 4512 < 2287 * (3 * 5120 * 17408 * 4512) ∧
    1378 * (3 * 17408 * 5120 * 4512) < 1000 * tuneStep 17408 5120 4512 ∧
    1000 * tuneStep 17408 5120 4512 < 1379 * (3 * 17408 * 5120 * 4512) := by
  decide

/-! ## 4 · The sketch premise (`nq_sketch_share`, `sketch.log`)

`share(X) = tr(X·PHP·Xᵀ) / tr(X·H·Xᵀ)`, `P = QQᵀ` the rank-`s` subspace iteration of `H` (3 passes). Rows:
`s = 64, 128, 256`. -/

/-- `tr(PHP)/tr(H)`. `List` [a finite sequence of things, in order]. -/
def shareH : List Nat := [839800, 885800, 923200]
/-- The weight `W`. `List` [a finite sequence of things, in order]. -/
def shareW : List Nat := [935000, 951400, 965900]
/-- The W2 error `Ŵ − W`. `List` [a finite sequence of things, in order]. -/
def shareW2 : List Nat := [219900, 306900, 435400]
/-- The Q4_K_M error. `List` [a finite sequence of things, in order]. -/
def shareQ4 : List Nat := [838600, 883300, 920400]

/-- **The tuned error lives in `H`'s tail: the rank-256 sketch carries 43.5% of W2's `H`-energy against 92.0% of
Q4_K_M's.** An `H`-weighted fit removes its error along `H`'s dominant directions, so the residual is
concentrated where `H` is small; an `H`-agnostic quantiser's error share tracks `tr(PHP)/tr(H)` (within 0.003
at every `s`). `∧` [both are true]; `<` [is smaller than]. `≤` [is smaller than or equal to]; `∀` [for every single one]. -/
theorem sketch_share_tail :
    (∀ i : Fin 3, 2 * shareW2[i]! < 1000000) ∧
    (∀ i : Fin 3, shareW2[i]! < shareQ4[i]!) ∧
    (∀ i : Fin 3, shareQ4[i]! ≤ shareH[i]! ∧ shareH[i]! < shareQ4[i]! + 3000) ∧
    (∀ i : Fin 3, shareH[i]! < shareW[i]!) := by
  decide

/-! ## 5 · The sketched tune (`tune-sk256.log`) -/

/-- `tune_sketch` at `s = 256` from the same start as W2, 1024 steps: full-`H` relH 0.01678. -/
def relSk256 : Nat := 16780
/-- Its wall time, tenths of a second (110.1 s), against W2's full-`H` tune (403.9 s). -/
def wallSk256 : Nat := 1101
/-- W2's full-`H` tune wall time, tenths of a second. -/
def wallW2 : Nat := 4039
/-- `tune_sketch` at `s = 128`, same start: full-`H` relH 0.01719 (`tune-sk128.log`). -/
def relSk128 : Nat := 17190
/-- Its wall time, tenths of a second (105.0 s). -/
def wallSk128 : Nat := 1050

/-- **The share is not the criterion: the rank-256 sketched tune reaches a lower full-`H` relH than the full
tune (0.01678 against 0.01727) in 0.27× the wall time (110.1 s against 403.9 s, 3.67×)**, although the sketch
sees under half of the tuned error's energy (`sketch_share_tail`). The out-of-sketch error does not grow under
the sketched steps on this tensor. One run, one tensor; the 7.1× multiply-add ratio (`step_cost_l23_down`) is
not reached in wall time. `∧` [both are true]; `<` [is smaller than]. -/
theorem sketch_tune_l23_down :
    relSk256 < relW2 ∧ 2 * shareW2[2]! < 1000000 ∧
    3 * wallSk256 < wallW2 ∧ wallW2 < 4 * wallSk256 := by
  decide

/-- **Halving the sketch rank to 128 halves the step's multiply-adds (`step_cost_l23_down`: 14–15× against 7–8×
under the full step) but saves 5% wall time (105.0 s against 110.1 s)**, and still lands under W2 (0.01719 against
0.01727). The sketched step's wall time is not its `r·s·3(o + n) + o·s²` term: a per-step cost independent of
`s` dominates at `s ≤ 256`. `∧` [both are true]; `<` [is smaller than]. -/
theorem sketch_rank_floor :
    relSk128 < relW2 ∧ relSk256 < relSk128 ∧
    wallSk128 < wallSk256 ∧ 20 * (wallSk256 - wallSk128) < wallSk256 ∧
    2 * sketchStep 1024 3584 896 128 < sketchStep 1024 3584 896 256 + sketchStep 1024 3584 896 128 / 10 := by
  decide

/-! ## 6 · Held-out KL, same scope (`kl-*-89.log`)

Only `blk.23.ffn_down` replaced, every other tensor at the teacher's precision; mean over chunks 8–9,
510 tokens, micro-nats. -/

/-- Q4_K_M (4.75 b/w), KL 0.000952 ± 0.000037. -/
def klQ4KM : Nat := 952
/-- W2 (ADMM + layer tune, 1.1451 b/w), KL 0.075035 ± 0.004615. -/
def klW2 : Nat := 75035
/-- NanoQuant ADMM only (1.1451 b/w), KL 0.077613 ± 0.003045. -/
def klAdmm : Nat := 77613
/-- c6x29-sel (1.175 b/w), KL 0.050752 ± 0.000676. -/
def klSel : Nat := 50752
/-- W2's spread (sd), micro-nats. -/
def klW2sd : Nat := 4615

/-- **At one tensor, W2 is 78.8× Q4_K_M in KL, the layer tune's 3% KL gain over ADMM is inside W2's spread, and
c6x29-sel beats both (53.3× Q4_K_M)**: relH ordering (`w2_relH`) carries to KL for c6x29-sel against W2, but
W2's 29% relH gain over ADMM does not carry. `∧` [both are true]; `<` [is smaller than]. -/
theorem kl_l23_down :
    78 * klQ4KM < klW2 ∧ klW2 < 79 * klQ4KM ∧
    klW2 < klAdmm ∧ klAdmm - klW2 < klW2sd ∧
    klSel < klW2 ∧ 53 * klQ4KM < klSel ∧ klSel < 54 * klQ4KM := by
  decide

end NanoQuant

end Learn
