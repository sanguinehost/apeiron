import Learn.QuantVerdicts


/-!
# NanoQuant on the crate: budget, measured quality and the tune's cost equation

The port is `src/learn/nanoquant.rs` (`SamsungLabs/NanoQuant` @ `a9e0a43`); the epic is
`docs/epics/nanoquant-subsume.md`. Every measured figure is a `model` input copied from its log under
`/home/socol/oc-tmp/escalation/lead-1023/nq/`: relH `tr(EHEᵀ)/tr(WHWᵀ)` in millionths, a share in
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

/-! ## 7 · The ADMM ridge solve: `H⁻¹` and the limb count (`b32c.log`, `invsmall.log`, `invbase.log`,
`seed-mixed.log`) -/

/-- Kernel multiply-adds of `nanoquant::spd_inverse_at` above the base order `b`: at each level with
`p = ⌊n/2⌋`, `q = n − p`, the products `X = A⁻¹B` (`p²q`), `BᵀX` (`q²p`), `Y = X·S⁻¹` (`pq²`) and `YXᵀ`
(`p²q`), then the two recursive halves. `fuel` bounds the depth. Below `b` the base is scalar
(`inv_factor_upper`) and counted in no term here.
`→` [takes the left and gives the right; between statements, if the left holds so does the right];
`≤` [is smaller than or equal to]. -/
def invMacs (b : Nat) : Nat → Nat → Nat
  | 0, _ => 0
  | fuel + 1, n =>
    if n ≤ b then 0
    else
      invMacs b fuel (n / 2) + invMacs b fuel (n - n / 2) +
        (2 * (n / 2) * (n / 2) * (n - n / 2) + 2 * (n / 2) * (n - n / 2) * (n - n / 2))

/-- `spd_inverse_at` wall at order 768, base 32, mean of 5 runs, hundredths of a millisecond (12.17 ms). -/
def invWall32 : Nat := 1217
/-- Same, base 192 (17.42 ms). -/
def invWall192 : Nat := 1742
/-- `max |H·X − I|` at order 300 against an `f64` oracle, base 32, units of `10⁻⁸` (7.705e-6). -/
def invRes32 : Nat := 771
/-- Same, unrecursed (base ≥ 300; 3.419e-5). -/
def invResNone : Nat := 3419

/-- **The recursion is `2n³/3` kernel multiply-adds at base 32 (301 694 976 at order 768, within 0.1% of
`2·768³/3`); base 192 runs 6% fewer kernel multiply-adds and 43% more wall, so the scalar base, not the
kernel, sets the inverse's wall; base 32 is also 4.4× more accurate.** `∧` [both are true]; `<` [is smaller
than]; `≤` [is smaller than or equal to]. -/
theorem inverse_base_l23_down :
    invMacs 32 10 768 = 301694976 ∧
    1000 * invMacs 32 10 768 ≤ 1000 * (2 * 768 ^ 3 / 3) ∧
    999 * (2 * 768 ^ 3 / 3) < 1000 * invMacs 32 10 768 ∧
    invMacs 192 10 768 < invMacs 32 10 768 ∧
    100 * invMacs 32 10 768 < 107 * invMacs 192 10 768 ∧
    invWall32 < invWall192 ∧ 14 * invWall32 < 10 * invWall192 ∧
    4 * invRes32 < invResNone ∧ invResNone < 5 * invRes32 := by
  decide

/-- 20-iteration ADMM wall, blk.23 `down`, `r = 768`, 3 limbs, base 32, seeds 0–2, tenths of a second. `List` [a finite sequence of things, in order]. -/
def fitBase32 : List Nat := [33, 33, 32]
/-- The same fit before the base change (3.5 s). -/
def fitBase192 : Nat := 35
/-- `admm_loss`, `10⁻⁵`, 3 limbs, base 32, seeds 0–2. `List` [a finite sequence of things, in order]. -/
def loss3 : List Nat := [16482, 16522, 16484]
/-- `admm_loss`, `10⁻⁵`, the mixed arm (2-limb `W·Z`, right-hand side and `rank1`; 3-limb Gram and
inverse), seeds 0–4. `List` [a finite sequence of things, in order]. Board: APEIRON-QUALITY-118-W1-T01. -/
def loss2 : List Nat := [16805, 16818, 16841, 16827, 16822]

/-- **Base 32 is faster on every seed, and 2 limbs lose on every seed**: every base-32 fit is under 3.5 s;
the mixed arm's best seed (0.16805) is above 3 limbs' worst (0.16522), and its mean is 1.98% above (`K3`: 3
limbs stay). `∀` [for every]; `∈` [is one of the things in]; `∧` [both are true]; `<` [is smaller than]; `>` [is
larger than]. -/
theorem admm_base_and_limbs :
    (∀ t ∈ fitBase32, t < fitBase192) ∧
    (∀ a ∈ loss3, ∀ c ∈ loss2, a < c) ∧
    100 * 3 * loss2.sum > 101 * 5 * loss3.sum ∧
    100 * 3 * loss2.sum < 102 * 5 * loss3.sum := by
  decide

/-- Sketched tune wall, blk.23 `down`, `r = 768`, `s = 256`, 128 steps from the base-32 factor, sketch
included, seeds 0–2, tenths of a second, before the rewrite (`tsb.log`). `List` [a finite sequence of things, in order]. -/
def tuneWallBefore : List Nat := [100, 99, 98]
/-- Same, after packed Adam moments, parallel sign and real maps and word transposes (`tsw.log`). `List` [a finite sequence of things, in order]. -/
def tuneWallAfter : List Nat := [71, 71, 71]
/-- Final `relH` (`10⁻⁵`), best `relH` (`10⁻⁵`) and sign flips of the run before the rewrite; every seed
printed the same triple. `List` [a finite sequence of things, in order]. -/
def traceBefore : List Nat := [3321, 1721, 50042]
/-- Same, after the rewrite. `List` [a finite sequence of things, in order]. Board: APEIRON-QUALITY-118-W1-T01. -/
def traceAfter : List Nat := [3321, 1721, 50042]

/-- **The rewrite is faster on every seed and leaves the trace unchanged**: every rewritten run is under
every original run, the mean wall falls 1.39× (9.90 → 7.10 s), and final `relH`, best `relH` and flips
match digit for digit. `∀` [for every]; `∈` [is one of the things in]; `∧` [both are true]; `<` [is smaller
than]; `=` [is the same as]. -/
theorem tune_rewrite_l23_down :
    (∀ a ∈ tuneWallAfter, ∀ b ∈ tuneWallBefore, a < b) ∧
    138 * tuneWallAfter.sum < 100 * tuneWallBefore.sum ∧
    100 * tuneWallBefore.sum < 140 * tuneWallAfter.sum ∧
    traceAfter = traceBefore := by
  decide

/-- Sketched tune wall, blk.23 `down`, `r = 768`, `s = 256`, 128 steps from the base-32 factor, seeds 0–2,
hundredths of a second, before the column-sum gradient (`tss.log`).
`List` [a finite sequence of things, in order]. -/
def colsumWallBefore : List Nat := [671, 660, 657]
/-- Same, with `∂s_pre[j] = Σ_i sign(V_ij)·(A·Qᵀ)_ij` as one exact signed column sum over the `A·Qᵀ` words in
place of the `sign(V)ᵀ` transpose and the `VᵀA·Q` row dots (`colsum-clean.log`).
`List` [a finite sequence of things, in order]. -/
def colsumWallAfter : List Nat := [525, 523, 527]
/-- Final `relH` (`10⁻⁵`), best `relH` (`10⁻⁵`) and sign flips before the column sum; every seed printed the
same triple (`tss.log`). `List` [a finite sequence of things, in order]. -/
def colsumTraceBefore : List Nat := [3321, 1721, 50042]
/-- Same, after. `List` [a finite sequence of things, in order]. Board: APEIRON-QUALITY-118-W1-T01. -/
def colsumTraceAfter : List Nat := [3320, 1722, 50103]

/-- **The column-sum gradient is faster on every seed at the same quality**: every run after is under every
run before, the mean wall falls 1.26× (6.63 → 5.25 s), final `relH` does not rise, best `relH` moves under
0.1% and sign flips under 1%. `∀` [for every]; `∈` [is one of the things in]; `∧` [both are true]; `<` [is
smaller than]; `≤` [is at most]. -/
theorem tune_colsum_l23_down :
    (∀ a ∈ colsumWallAfter, ∀ b ∈ colsumWallBefore, a < b) ∧
    125 * colsumWallAfter.sum < 100 * colsumWallBefore.sum ∧
    100 * colsumWallBefore.sum < 127 * colsumWallAfter.sum ∧
    colsumTraceAfter[0]! ≤ colsumTraceBefore[0]! ∧
    1000 * colsumTraceAfter[1]! ≤ 1001 * colsumTraceBefore[1]! ∧
    100 * colsumTraceAfter[2]! ≤ 101 * colsumTraceBefore[2]! := by
  decide

/-- Sketched tune wall after the column-sum gradient, then with `A·Qᵀ` taken against `Q` as a prepared kernel
operand (one plane per tune, not per step) and `∂V` and `∂s_pre` read from it in one fused pass blocked by
eight rows (`colgrads-clean.log`); hundredths of a second, seeds 0–2.
`List` [a finite sequence of things, in order]. -/
def colgradsWallAfter : List Nat := [479, 480, 477]
/-- Final `relH` (`10⁻⁵`), best `relH` (`10⁻⁵`) and sign flips after; every seed printed the same triple
(`colgrads-clean.log`). `List` [a finite sequence of things, in order]. Board: APEIRON-QUALITY-118-W1-T01. -/
def colgradsTraceAfter : List Nat := [3320, 1722, 50103]

/-- **The prepared-`Q` product and the fused gradient pass are faster on every seed with the same trace**: every
run after is under every run before, the mean wall falls 1.10× (5.25 → 4.79 s) and the final `relH`, best
`relH` and sign flips are equal. `∀` [for every]; `∈` [is one of the things in]; `∧` [both are true]; `<` [is
smaller than]. -/
theorem tune_colgrads_l23_down :
    (∀ a ∈ colgradsWallAfter, ∀ b ∈ colsumWallAfter, a < b) ∧
    109 * colgradsWallAfter.sum < 100 * colsumWallAfter.sum ∧
    100 * colsumWallAfter.sum < 110 * colgradsWallAfter.sum ∧
    colgradsTraceAfter = colsumTraceAfter := by
  decide

/-- ADMM at the reference's 400 outer iterations, blk.23 `ffn_down`, r = 768, seeds 2 and 0 (`i400.log`):
`admm_loss` (`10⁻⁵`). The 20-iteration losses are `admm_base_and_limbs`'s seeds 0–2 (`b32c.log`).
`List` [a finite sequence of things, in order]. -/
def iters400Loss : List Nat := [10344, 10341]
/-- `admm_loss` (`10⁻⁵`) at 20 iterations, seeds 0–2 (`b32c.log`).
`List` [a finite sequence of things, in order]. -/
def iters20Loss : List Nat := [16482, 16522, 16484]
/-- Final `relH` (`10⁻⁵`) after the 128-step sketched tune from the 400-iteration factor, seeds 2 and 0
(`i400.log`). `List` [a finite sequence of things, in order]. -/
def iters400Final : List Nat := [2478, 2471]
/-- ADMM fit wall per tensor, tenths of a second: 400 iterations (`i400.log`, both seeds), then 20
(`b32c.log`, seeds 0–2). `List` [a finite sequence of things, in order]. Board: APEIRON-QUALITY-118-W1-T01. -/
def iters400Wall : List Nat := [613, 613]
/-- Same, 20 iterations. `List` [a finite sequence of things, in order]. -/
def iters20Wall : List Nat := [33, 33, 32]

/-- **400 ADMM iterations beat 20 on quality and cost 18× the fit wall**: every 400-iteration `admm_loss` is
under 0.63× every 20-iteration one, every final `relH` after the same tune is under 0.75× the 20-iteration
trace (`colgradsTraceAfter`, 0.03320), and every 400-iteration fit is over 18× every 20-iteration fit, so
the ADMM is the dominant per-tensor term. `∀` [for every]; `∈` [is one of the things in]; `∧` [both are
true]; `<` [is smaller than]. -/
theorem admm_iters_l23_down :
    (∀ a ∈ iters400Loss, ∀ b ∈ iters20Loss, 100 * a < 63 * b) ∧
    (∀ a ∈ iters400Final, 100 * a < 75 * colgradsTraceAfter[0]!) ∧
    (∀ a ∈ iters400Wall, ∀ b ∈ iters20Wall, 18 * b < a) := by
  decide

/-- Final `relH` (`10⁻⁵`) after the 128-step sketched tune (s = 256), seed 0, blk.23 `ffn_down`, r = 768,
at 50, 100, 200 and 400 ADMM iterations (`icurve.log`, `i400.log`).
`List` [a finite sequence of things, in order]. -/
def iterCurveFinal : List Nat := [2787, 2622, 2521, 2471]
/-- ADMM fit wall, tenths of a second, at the same 50, 100, 200, 400 iterations.
`List` [a finite sequence of things, in order]. -/
def iterCurveWall : List Nat := [79, 155, 309, 613]
/-- Final `relH` (`10⁻⁵`) after 1024 further sketched tune steps from the 100- and the 400-iteration
factors (`budget.log`). `List` [a finite sequence of things, in order]. -/
def budgetFinal : List Nat := [2157, 2039]
/-- Tune wall for those 1024 steps, tenths of a second (`budget.log`).
`List` [a finite sequence of things, in order]. Board: APEIRON-QUALITY-118-W1-T01. -/
def budgetTuneWall : List Nat := [344, 341]

/-- **Past 100 ADMM iterations, tune steps buy `relH` over 4× faster per second than iterations**: the
final `relH` falls at every step of the 50/100/200/400 curve; 1024 tune steps from the 100-iteration
factor cut `relH` 0.02622 → 0.02157 in 34.4 s, while iterations 100 → 400 cut it 0.02622 → 0.02471 in
45.8 s, and the rate ratio `(465/344)/(151/458)` exceeds 4; after the same 1024 steps the 400-iteration
factor ends only 0.00118 lower, under the 0.00151 that the extra iterations bought before the tune.
`∀` [for every]; `∧` [both are true]; `<` [is smaller than]. -/
theorem admm_budget_l23_down :
    (iterCurveFinal.zip iterCurveFinal.tail).all (fun p => p.2 < p.1) = true ∧
    4 * (iterCurveFinal[1]! - iterCurveFinal[3]!) * budgetTuneWall[0]! <
      (iterCurveFinal[1]! - budgetFinal[0]!) * (iterCurveWall[3]! - iterCurveWall[1]!) ∧
    budgetFinal[0]! - budgetFinal[1]! < iterCurveFinal[1]! - iterCurveFinal[3]! := by
  decide

/-- Entries that differ between the device limb product (`src/gpu/limb.rs`, int8 cooperative matrices,
64-bit lo/hi accumulator, pooled host finish) and the CPU `product_rows`, at 1024×768×3584, 768×768×3584,
768×128×3584, 1024×768×128, 1024×768×768, 768×768×1024, 256×256×1024 and 100×37×70, direct and routed
through `gemm_q32` (`gfp.log`). `List` [a finite sequence of things, in order]. -/
def limbDeviceDiff : List Nat := [0, 0, 0, 0, 0, 0, 0, 0]
/-- `gemm_q32` wall at the ADMM `W·Z` shape 1024×768×3584, 34×33 limbs, hundredths of a millisecond:
three CPU-only runs, then three routed runs (`gfp.log`). `List` [a finite sequence of things, in order]. -/
def limbCpuWall : List Nat := [3671, 3021, 2827]
/-- See `limbCpuWall`. `List` [a finite sequence of things, in order]. -/
def limbRoutedWall : List Nat := [709, 669, 685]

/-- **The device limb product is the CPU product, over 4× faster at the `W·Z` shape**: no entry differs at
any of the eight shapes, every routed run is under a third of every CPU-only run, and the routed sum
is under a quarter of the CPU sum (mean 31.73 → 6.88 ms).
`∀` [for every]; `∈` [is one of the things in]; `∧` [both are true]; `<` [is smaller than]. -/
theorem limb_device_route :
    limbDeviceDiff.all (· = 0) = true ∧
    (∀ a ∈ limbRoutedWall, ∀ b ∈ limbCpuWall, 3 * a < b) ∧
    4 * limbRoutedWall.sum < limbCpuWall.sum := by
  decide

/-- `admm_loss` (`10⁻⁵`) of the 100-iteration fit, blk.23 `ffn_down`, r = 768, seed 0, in six interleaved
runs, device on then off (`dfit2.log`). `List` [a finite sequence of things, in order]. -/
def deviceFitLoss : List Nat := [11172, 11172, 11172, 11172, 11172, 11172]
/-- Final `relH` (`10⁻⁵`) after the 128-step sketched tune (s = 128) in the same six runs.
`List` [a finite sequence of things, in order]. -/
def deviceFitFinal : List Nat := [2631, 2631, 2631, 2631, 2631, 2631]
/-- ADMM wall, tenths of a second, device on (`APEIRON_LIMB_DEVICE=1`), three runs.
`List` [a finite sequence of things, in order]. -/
def deviceOnAdmm : List Nat := [122, 126, 126]
/-- ADMM wall, tenths of a second, device off, three runs interleaved with `deviceOnAdmm`.
`List` [a finite sequence of things, in order]. -/
def deviceOffAdmm : List Nat := [157, 153, 153]
/-- Tune wall for 128 steps, hundredths of a second, device on.
`List` [a finite sequence of things, in order]. -/
def deviceOnTune : List Nat := [287, 265, 270]
/-- Tune wall for 128 steps, hundredths of a second, device off.
`List` [a finite sequence of things, in order]. -/
def deviceOffTune : List Nat := [315, 318, 317]

/-- **Routing the limb products to the device leaves the fit unchanged and cuts its ADMM wall under
0.83×**: the loss and the final `relH` are one value across all six runs, every device-on ADMM run is
under 0.83× every device-off run (means 12.47 against 15.43 s, 0.808×), and every device-on tune is
faster than every device-off tune (2.74 against 3.17 s).
`∀` [for every]; `∈` [is one of the things in]; `∧` [both are true]; `<` [is smaller than]. -/
theorem limb_device_fit_l23_down :
    deviceFitLoss.all (· = deviceFitLoss[0]!) = true ∧
    deviceFitFinal.all (· = deviceFitFinal[0]!) = true ∧
    (∀ a ∈ deviceOnAdmm, ∀ b ∈ deviceOffAdmm, 100 * a < 83 * b) ∧
    (∀ a ∈ deviceOnTune, ∀ b ∈ deviceOffTune, a < b) := by
  decide

end NanoQuant

end Learn
