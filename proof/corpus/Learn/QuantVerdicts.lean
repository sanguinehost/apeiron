import Mathlib.Tactic
import Mathlib.Analysis.Real.Pi.Bounds
import Learn.FitIcm

set_option linter.apeiron.notation false

/-!
# The 1-bit quantiser spikes: every clause against its measured constant

Eight receipts. The six spikes ran on qwen2.5-0.5B (`blk.{0,11,23}.{attn_q,attn_v,ffn_gate,ffn_down}`,
imatrix-weighted
`D_H = Σh e² / Σh w²`) and on synthetic `N(0,1)`; R1 is a refinement audit (§9). Receipts, by sha256 prefix:

| spike | receipt | sha256 |
|---|---|---|
| IC (cipher diffusion, keyed streams) | `lead-1021/IC/receipt.md` | `728f6afdbfe99030` |
| RT-1 (minimal cut, greedy allocation) | `lead-1021/RT/receipt.md` | `041746b6d7a1c6a9` |
| QM (σ-face metric transforms) | `lead-1021/QM/receipt.md` | `80b97beac35c3301` |
| R1 (refinement audit of `Learn.FitIcm`) | `lead-1021/R1/receipt.md` | `b5daa7c47da05abf` |
| R2 (prior art, codebook model) | `lead-1021/R2/receipt.md` | `543662be74dbd54a` |
| RT-2 (radius as bulk coordinate) | `lead-1021/RT2/receipt.md` | `6203395b72d30be9` |
| PF (contact map as `H`, funnel, Gibbs) | `lead-1021/PF/receipt.md` | `7cd6ff26c0c8ac69` |
| KLICM (KL-driven Gauss–Seidel ICM, §12) | `lead-1021/KLICM/receipt.md` | `81d872f66b96efee` |
| KLICM (KL-driven Gauss–Seidel ICM, §12) | `lead-1021/KLICM/receipt.md` | `81d872f66b96efee` |

Every figure below is a `model` input copied from its receipt table: a distortion in millionths
(`D × 10⁶`), a rate in ten-thousandths of a bit. No theorem here produces a measurement. Every
`D_H` is a weight-space proxy: no clause before §12 moves, bounds or predicts KL; §12's KL figures are
on a 2-layer fixture, not the 0.5B model. Nothing here is on the
production fit path (`src/learn/fold/fit/{icm,gptq_e8}.rs`). rustc, the harnesses and the imatrix
itself are outside every statement.

The structural section proves the three facts the measurements kept rediscovering, so a proposal
that contradicts one is refused before it is measured:

* `signCode_nearest`: a code whose codewords are all `±1` vectors (one radius, both parity classes) has
  the coordinatewise sign as its nearest point: it is the 1-bit scalar quantiser, whatever it is
  called. A 256-entry 8-dimensional "ball" built from one radius is this code.
* `selection_ge_argmin`: a zero-bit encoder-side choice of codeword from the same codebook, decoded by
  the same map and scored by the same per-block metric, never beats the per-block metric argmin.
* `rate_of_entries`: `2ᵏ` entries over 8 weights cost `k / 8` bits per weight.
-/

namespace Learn

namespace QuantVerdicts

/-! ## 1 · Structure -/

/-- **The sign code's nearest point is the coordinatewise sign.** For every target `y` and every
codeword `c` with entries in `{1, -1}`, the vector `sgn y` (with `sgn 0 = 1`) is no farther than `c`.
A codebook containing only `±1` vectors, of either parity class or both, is therefore nearest-decoded
exactly as 1-bit scalar quantisation. -/
theorem signCode_nearest {n : ℕ} (y c : Fin n → ℝ) (hc : ∀ i, c i = 1 ∨ c i = -1) :
    ∑ i, ((if 0 ≤ y i then (1 : ℝ) else -1) - y i) ^ 2 ≤ ∑ i, (c i - y i) ^ 2 := by
  refine Finset.sum_le_sum fun i _ => ?_
  split_ifs with h <;> rcases hc i with h1 | h1 <;> simp only [h1] <;> nlinarith

/-- **The two parity classes of `{±1}⁸` together are the whole cube.** Every sign vector has an even or
an odd number of minus signs, so a code holding both chiralities at the one radius `√8` holds all
`2⁸ = 256` sign vectors: at 256 entries it has no room for a second radius. -/
theorem parity_classes_cover (v : Fin 8 → Bool) :
    Even ((Finset.univ.filter fun i => v i = true).card) ∨
      Odd ((Finset.univ.filter fun i => v i = true).card) :=
  Nat.even_or_odd _

/-- The cube has `2⁸ = 256` points: the sign code is exactly a 256-entry, 1-bit-per-weight code. -/
theorem signCode_card : Fintype.card (Fin 8 → Bool) = 256 := by
  simp

/-- **Selection never beats the argmin.** Over blocks `I`, each block choosing a codeword from the
finite codebook `C` with cost `f b c`, one choice `opt` is no worse than every selection `sel` with the
same codebook. QM's encoder-side faces A2 (elliptic) and A3 (parabolic) are selections of this kind;
A1 is the argmin. This covers the per-block argmin with the scale held fixed, not the coordinate
descent `src/learn/fold/fit/icm.rs` runs (that body is `Learn.FitIcm`), and not a face that changes the
decode map (QM A4, A5, A7), which pays its own side bits. -/
theorem selection_ge_argmin {ι α : Type*} (I : Finset ι) (C : Finset α) (hC : C.Nonempty)
    (f : ι → α → ℝ) :
    ∃ opt : ι → α, (∀ b, opt b ∈ C) ∧
      ∀ sel : ι → α, (∀ b ∈ I, sel b ∈ C) → ∑ b ∈ I, f b (opt b) ≤ ∑ b ∈ I, f b (sel b) := by
  choose opt hmem hmin using fun b => Finset.exists_min_image C (f b) hC
  exact ⟨opt, hmem, fun sel hsel => Finset.sum_le_sum fun b hb => hmin b (sel b) (hsel b hb)⟩

/-- **Rate of a codebook.** `2ᵏ` entries indexed per 8 weights cost `k / 8` bits per weight. -/
theorem rate_of_entries (k : ℕ) : (Nat.log 2 (2 ^ k) : ℚ) / 8 = k / 8 := by
  rw [Nat.log_pow (by norm_num)]

/-- QuIP# E8P (`2¹⁶` entries) is 2 b/w; a 256-entry code is 1 b/w; with one 16-bit scale per 128
weights the production pack is `1 + 16/128 = 9/8` b/w. -/
theorem rates : (16 : ℚ) / 8 = 2 ∧ (8 : ℚ) / 8 = 1 ∧ (1 : ℚ) + 16 / 128 = 9 / 8 := by norm_num

/-- `model` (analytic): the distortion per dimension of the sign code with its best scale
`r = E|y| = √(2/π)` on `N(0,1)` is `1 − 2/π`. R2 records it as 0.3634. -/
theorem signCode_gaussian_value : 0.3633 < 1 - 2 / Real.pi ∧ 1 - 2 / Real.pi < 0.3634 := by
  have h1 := Real.pi_gt_d6
  have h2 := Real.pi_lt_d6
  constructor
  · have : 2 / Real.pi < 0.6367 := by rw [div_lt_iff₀ Real.pi_pos]; nlinarith
    linarith
  · have : 0.6366 < 2 / Real.pi := by rw [lt_div_iff₀ Real.pi_pos]; nlinarith
    linarith

/-! ## 2 · Synthetic `N(0,1)`: the production code against the sign code and `D(1)` -/

/-- `model`: synthetic `D_mse/σ²` of the production code (sign + FWHT-128 + E8∪O8, one 16-bit LS scale
per 128): RT-1 (256×1024, 3 seeds), QM seed 1 (128×128), IC (256×256). -/
def synthRT : ℕ := 316530

def synthQM : ℕ := 317098

def synthIC : ℕ := 345800

/-- `model` (R2 table): shell plus a free gain per block, and shell plus one 4-bit gain per 16 blocks. -/
def r2FreeGain : ℕ := 274100

def r2GroupGain4bit : ℕ := 486970

/-- Shannon's `D(1) = 2⁻²` for `N(0,1)` at one bit per dimension, in millionths. -/
def shannonD1 : ℕ := 250000

/-- **A one-radius code would be worse than what ships.** Every synthetic control of the production
code sits below the sign code's 0.3633, so replacing the shell by a 256-entry code with both
chiralities at one radius (`signCode_nearest`) raises distortion. This refutes "the ball codebook is
the best candidate" at 1 b/w. -/
theorem production_beats_signCode : synthRT < 363300 ∧ synthQM < 363300 ∧ synthIC < 363300 := by
  decide

/-- **R2's group-gain defect is its 4-bit gain, not the production code.** R2's 0.48697 for "one gain
per 16 blocks" exceeds every measured synthetic control of the shipped 16-bit-scale code by more than
1.4×, so the "+78 % defect" does not describe the production pack. The headroom from a free per-block
gain over the shipped code is at most `1 − 0.2741/0.3165 < 14 %`, not 78 %. -/
theorem r2_group_gain_is_not_production :
    14 * synthIC < 10 * r2GroupGain4bit ∧ 100 * (synthRT - r2FreeGain) < 14 * synthRT := by
  decide

/-- **The direction code is near `D(1)`.** Shell plus free gain is within 10 % of Shannon's `D(1)`:
no 1-bit-per-dimension direction code (ball, lattice, trellis) can win more than that on `N(0,1)`
MSE. Every larger `D_H` gain comes from allocation over a non-uniform `h`, not from shape. -/
theorem shape_headroom : r2FreeGain * 10 < shannonD1 * 11 := by decide

/-! ## 3 · IC: cipher diffusion and keyed streams (KILL) -/

/-- `model` (IC §4, 12 tensors, 3 seeds): `D_H` baseline, support-8 rotation, keyed dither, per-row
reverse water-fill (spinor-only index), and `D_H` of the named Q4_K_M file. -/
def icBase : ℕ := 564611

def icW8 : ℕ := 515982

def icDither : ℕ := 581150

def icRowWaterFill : ℕ := 628069

def icRef : ℕ := 2049

/-- **IC kill: best mechanism closes 8.64 % of the gap**, below the 10 % bar and above 8 %. -/
theorem ic_kill :
    10 * (icBase - icW8) < icBase - icRef ∧ 8 * (icBase - icRef) < 100 * (icBase - icW8) := by
  decide

/-- **IC prediction 2 falsified.** The support dial was predicted to move `D_H` under 5 %; it moved
8.6 %. -/
theorem ic_p2_falsified : 5 * icBase < 100 * (icBase - icW8) := by decide

/-- **Keyed dither loses.** -/
theorem ic_dither_loses : icBase < icDither := by decide

/-- **A water-fill can lose.** IC's per-row water-fill over a spinor-only index raises `D_H` by over
11 %: the sign of a water-fill gain depends on the rate–distortion curve it allocates over. -/
theorem ic_waterfill_loses : 111 * icBase < 100 * icRowWaterFill := by decide

/-! ## 4 · RT-1: minimal cut and greedy allocation (KILL; one survivor) -/

/-- `model` (RT-1 §5, `blk.0`, 4 tensors, 1 seed): uniform (8-bit index, scale/128) `D_H`, the snapped
minimal-cut closed form, the greedy water-fill on measured curves, Q4 on the same tensors; and the
three σ faces of the cut. -/
def rtBase : ℕ := 267358

def rtMinCut : ℕ := 275254

def rtGreedy : ℕ := 229871

def rtRef : ℕ := 1450

def rtElliptic : ℕ := 267447

def rtHyperbolic : ℕ := 267937

/-- **RT-1 kill: the minimal cut loses to uniform.** -/
theorem rt_mincut_loses : rtBase < rtMinCut := by decide

/-- **The σ faces do not move the cut.** Elliptic and hyperbolic sit within 0.25 % of identity. -/
theorem rt_faces_flat :
    400 * (rtElliptic - rtBase) < rtBase ∧ 400 * (rtHyperbolic - rtBase) < rtBase := by decide

/-- **Survivor: greedy allocation closes 14.1 % of the gap** (above 10 %, below 15 %), one seed, four
tensors of `blk.0`. -/
theorem rt_greedy_survives :
    rtBase - rtRef < 10 * (rtBase - rtGreedy) ∧ 100 * (rtBase - rtGreedy) < 15 * (rtBase - rtRef) := by
  decide

/-- **RT-1 prediction falsified by more than 3×.** Greedy was predicted at most 4 %; it measured 14 %. -/
theorem rt_greedy_prediction_falsified : 3 * (4 * rtBase) < 100 * (rtBase - rtGreedy) := by decide

/-- **The survivor is still 158× the reference.** -/
theorem rt_greedy_far : 158 * rtRef < rtGreedy := by decide

/-! ## 5 · QM: σ-face metric transforms (KILL) -/

/-- `model` (QM §5, 12 tensors, 3 seeds): `D_H` per arm, and the named Q4_K_M file (6.5156 b/w). -/
def qmA0 : ℕ := 297728

def qmA1 : ℕ := 208888

def qmA2 : ℕ := 277055

def qmA3 : ℕ := 270301

def qmA4 : ℕ := 268560

def qmA5 : ℕ := 273201

def qmA6 : ℕ := 230945

def qmA7 : ℕ := 294564

def qmRef : ℕ := 888

/-- Half-spreads (n = 3) of A0 and A2. -/
def qmA0Spread : ℕ := 3931

def qmA2Spread : ℕ := 8808

/-- Rates, ten-thousandths of a bit: A0–A3 1.1250, A4–A5 1.3750, A6 1.1274, A7 1.1406. -/
def rateA0 : ℕ := 11250

def rateA4 : ℕ := 13750

def rateA6 : ℕ := 11274

def rateA7 : ℕ := 11406

/-- The target rate, 1.18 b/w. -/
def rateTarget : ℕ := 11800

/-- **QM kill, same-bits form.** A7 (hyperbolic, 2 bits per 128-group) closes under 1.1 % of the gap,
and under 1.1 % of `D_H(A0)` itself (an infinitely good reference). -/
theorem qm_kill_same_bits :
    100 * (qmA0 - qmA7) < 11 * (qmA0 - qmRef) / 10 ∧ 1000 * (qmA0 - qmA7) < 11 * qmA0 := by decide

/-- **The per-block hyperbolic oracle is under 10 % and over budget.** A4 closes under 10 % of
`D_H(A0)` and spends 1.375 b/w, above the 1.18 target. -/
theorem qm_oracle_bounded : 10 * (qmA0 - qmA4) < qmA0 ∧ rateTarget < rateA4 := by decide

/-- **QM prediction refuted.** The aligned hyperbolic face was predicted at `≤ 0.85 · D_H(A0)`. -/
theorem qm_prediction_refuted : ¬ 100 * qmA4 ≤ 85 * qmA0 := by decide

/-- **The elliptic face is not neutral.** A2 = A0 was predicted; A2 sits below A0 by more than the
summed half-spreads. -/
theorem qm_elliptic_not_neutral : qmA2 + qmA0Spread + qmA2Spread < qmA0 := by decide

/-- **The parabolic face does not raise `D_H`.** A3 > A0 was predicted; A3 is lower. -/
theorem qm_parabolic_refuted : qmA3 < qmA0 := by decide

/-- **Every encoder-side face is dominated by the metric argmin**, as `selection_ge_argmin` requires;
A1 closes 29.9 % of the gap, which is H-weighted rounding that the production fit already runs. -/
theorem qm_argmin_dominates :
    qmA1 < qmA2 ∧ qmA1 < qmA3 ∧ qmA1 < qmA4 ∧ qmA1 < qmA5 ∧ qmA1 < qmA7 ∧
      299 * (qmA0 - qmRef) < 1000 * (qmA0 - qmA1) := by decide

/-- **Survivor: water-fill over rows closes 22.5 % of the gap at 1.1274 b/w**, beating every face, at
fewer bits than the per-block face and within the 1.18 target. -/
theorem qm_waterfill_survives :
    22 * (qmA0 - qmRef) < 100 * (qmA0 - qmA6) ∧ qmA6 < qmA4 ∧ rateA6 < rateA4 ∧ rateA6 < rateTarget := by
  decide

/-- **Water-fill's sign is measured, not given.** RT-1's greedy and QM's A6 gain; IC's row water-fill
loses. A cost-model claim that water-fill always gains at 1 b/w is refuted by these receipts. -/
theorem waterfill_sign_mixed : rtGreedy < rtBase ∧ qmA6 < qmA0 ∧ icBase < icRowWaterFill := by decide

/-! ## 6 · RT-2: the radius as a bulk coordinate (KILL)

Receipt `lead-1021/RT2/receipt.md`, sha256 `6203395b72d30be9`; 12 tensors,
3 seeds. -/

/-- `model` (RT-2 table): baseline `D_H`, the free-gain ceiling (≈ 2 b/w, not a rate point), the best
radius arm at any rate (C, 4 Lloyd levels, 1.375 b/w), the Q4_K_M file's `D_H`. -/
def rt2Base : ℕ := 369550

def rt2FreeGain : ℕ := 324260

def rt2LloydL4 : ℕ := 330850

def rt2Ref : ℕ := 1961

/-- `model`: every RT-2 arm inside 1.18 b/w, as `(rate in ten-thousandths, D_H)`: the baseline, two
shells × 128 shapes (first 128, best 128), four shells × 64 (first, best), the shape-only control. -/
def rt2InEnvelope : List (ℕ × ℕ) :=
  [(11250, 369550), (11250, 422640), (11250, 419490), (11250, 582310), (11250, 484050),
    (11250, 442550)]

/-- `model`: `D_H` of C at 4 levels per σ spacing: hyperbolic, parabolic, elliptic. -/
def rt2Faces : List ℕ := [330910, 332260, 334440]

/-- **RT-2 kill.** No arm inside 1.18 b/w reaches `0.9 · D_H(A)`; no radius arm inside it beats the
baseline. -/
theorem rt2_kill :
    (∀ p ∈ rt2InEnvelope, p.1 ≤ rateTarget → 9 * rt2Base < 10 * p.2) ∧
      ∀ p ∈ rt2InEnvelope, rt2Base ≤ p.2 := by decide

/-- **The ceiling was predicted.** With its bits given away, a free gain per block closes 12.3 % of the
gap, inside the `< 14 %` that `r2_group_gain_is_not_production` derived on `N(0,1)` before RT-2 ran. -/
theorem rt2_ceiling_as_predicted :
    12 * (rt2Base - rt2Ref) < 100 * (rt2Base - rt2FreeGain) ∧
      100 * (rt2Base - rt2FreeGain) < 14 * (rt2Base - rt2Ref) := by decide

/-- **σ does not choose the radial spacing.** Each face sits within 1.1 % of the fitted Lloyd levels. -/
theorem rt2_faces_flat : ∀ x ∈ rt2Faces, rt2LloydL4 ≤ x ∧ 1000 * (x - rt2LloydL4) < 11 * rt2LloydL4 := by
  decide

/-! ## 7 · What the budget admits: predictions before a run -/

/-- **The side-information budget.** Over the 8-bit index per 8 weights and the 16-bit scale per 128,
`s` more bits per 8-block fit 1.18 b/w all-in exactly when `s ≤ 0.44`. -/
theorem side_budget (s : ℚ) : 1 + 16 / 128 + s / 8 ≤ 118 / 100 ↔ s ≤ 11 / 25 := by
  constructor <;> intro h <;> linarith

/-- **A fixed field of one bit per block does not fit.** Any per-block radius, gain or face flag coded
in a whole bit sets the rate to at least 1.25 b/w. RT-2's C arms (1.25–1.5) and QM's A4/A5 (1.375)
are instances; the budget refused them before they ran. -/
theorem block_bit_overruns (s : ℚ) (hs : 1 ≤ s) : 118 / 100 < 1 + 16 / 128 + s / 8 := by linarith

/-- **Per 128-group the budget is 7.04 bits.** QM's A7 (2 bits per group) fits; a lever that needs
more per-group side information than that does not. -/
theorem group_budget (g : ℚ) : 1 + 16 / 128 + g / 128 ≤ 118 / 100 ↔ g ≤ 176 / 25 := by
  constructor <;> intro h <;> linarith

/-- **The rate–distortion floor at ≤ 1.25 b/w is at least 89× the Q4 file.** `model` (Shannon 1959,
the Gaussian distortion–rate function `D(R) = 2^{-2R}` for an i.i.d. `N(0,1)` source; this module
proves the arithmetic, not the converse). For every rate up to 1.25 b/w, `D(R) > 0.176`, which
exceeds 89 × RT-2's measured Q4_K_M `D_H` (0.001961). This bounds MSE on the rotated weights. It
does not bound `D_H`, which allocation over a non-uniform `h` can push below `D(R)`
(`Learn.WaterFill.gaussian_floor`). The prediction it licenses is that no weight-space code at
this rate matches the Q4 file's distortion, so Q4-class KL at 1.18 b/w needs KL to be insensitive to
most of the distortion. That is the end-to-end finish, not a codebook. -/
theorem shannon_floor_vs_reference (R : ℝ) (hR : R ≤ 5 / 4) :
    (89 : ℝ) * (rt2Ref : ℝ) / 10 ^ 6 < 0.176 ∧ (0.176 : ℝ) < (2 : ℝ) ^ (-(2 * R)) := by
  refine ⟨by norm_num [rt2Ref], ?_⟩
  have hmono : (2 : ℝ) ^ (-(5 / 2 : ℝ)) ≤ 2 ^ (-(2 * R)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith)
  have hx : ((2 : ℝ) ^ (5 / 2 : ℝ)) ^ 2 = 32 := by
    rw [← Real.rpow_mul_natCast (by norm_num)]
    rw [show (5 / 2 : ℝ) * ((2 : ℕ) : ℝ) = ((5 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hpos : 0 < (2 : ℝ) ^ (5 / 2 : ℝ) := Real.rpow_pos_of_pos (by norm_num) _
  have hlt : (2 : ℝ) ^ (5 / 2 : ℝ) < 5.66 := by nlinarith
  rw [Real.rpow_neg (by norm_num)] at hmono
  have hinv : (0.176 : ℝ) < ((2 : ℝ) ^ (5 / 2 : ℝ))⁻¹ := by
    rw [lt_inv_comm₀ (by norm_num) hpos]
    linarith [show (5.66 : ℝ) < (0.176 : ℝ)⁻¹ by norm_num]
  linarith

/-! ## 8 · The 27B footprint at 27.318 G weights -/

/-- `model`: the weight count of the 27B model, in millions (`tests/learn_fold_budget.rs:22`, the
document's 27.318 G). -/
def weights27b : ℕ := 27318

/-- **1.18 b/w is 4.0294 GB: over 4 GB, under 4 GiB.** -/
theorem footprint_118 :
    (weights27b : ℚ) * 10 ^ 6 * (118 / 100) / 8 = 4029405000 ∧
      (4 * 10 ^ 9 : ℚ) < 4029405000 ∧ (4029405000 : ℚ) < 2 ^ 32 := by norm_num [weights27b]

/-- **The frozen reference rate 1.1795 is also over 4 GB.** -/
theorem footprint_frozen : (4 * 10 ^ 9 : ℚ) < (weights27b : ℚ) * 10 ^ 6 * (11795 / 10000) / 8 := by
  norm_num [weights27b]

/-- **The 4 GB rate is `16000/13659`, between 1.1713 and 1.1714.** 1.1714 itself overshoots 4 GB; the
largest four-decimal rate that fits is 1.1713. Matches `max_body_bpw` in
`tests/learn_fold_budget.rs:127`. -/
theorem rate_4gb :
    (4 * 10 ^ 9 * 8 : ℚ) / (weights27b * 10 ^ 6) = 16000 / 13659 ∧
      (11713 / 10000 : ℚ) < 16000 / 13659 ∧ (16000 / 13659 : ℚ) < 11714 / 10000 ∧
      (4 * 10 ^ 9 : ℚ) < weights27b * 10 ^ 6 * (11714 / 10000) / 8 := by
  norm_num [weights27b]

/-- **`Perf.CostCorrespondence.qwen27b_target_floor` rests on 27·10⁹ weights.** At the 27.318 G count
the 1.18 b/w footprint crosses 4 GB and the 960 GB/s decode floor crosses its stated 4.15 ms. -/
theorem target_floor_stale :
    (27 * 10 ^ 9 : ℚ) * 118 / 100 / 8 < 4 * 10 ^ 9 ∧
      (415 / 100000 : ℚ) < 4029405000 / (960 * 10 ^ 9) := by norm_num

/-! ## 9 · R1: the exact word-cost correspondence at the `O₈` rows (refuted)

R1 §2.2 states that `Learn.FitIcm.WordCostCorresponds` has no instance at the call-site scale. Two
theorems make that a kernel fact about the extracted `fit::icm::block_cost`:

* `wordCostCorresponds_uninhabited`: at `H = I`, `x = 1`, `a = 1`, for every `aa` divisible by `2⁶⁴`
  and every `two_a` divisible by `2³³` (the call-site scales `aa = (a·2³²)²`, `two_a = 2a·2³²` at any
  integer `a`) and every pair of `i64` slices, the exact correspondence fails at row 240. The
  extracted value `aa·q − two_a·dot·2³²` is then divisible by `2⁶⁴`, while `2⁹⁶·blockReduce` is
  `2³²·SQRT8² − 2⁶⁵·Σ C`, and `2³² ∤ SQRT8²` (`not_dvd_sqrt8_sq`).
* `o8_row_is_argmin`: the 16 `O₈` rows cannot be dropped from the correspondence instead. At `H = I`,
  `a = 1`, `x = (4, 0, …, 0)`, row 241 (`+√8·e₀`) costs strictly less than every one of the 240 exact
  rows, so an argmin restricted to the exact rows is not the table's argmin.

What this licenses: an `ε`-correspondence composed with `word_argmin_cost_bound` (R1 §2.5), not an
exact one. What it rules out: any statement that the full 256-row argmin is proved through
`WordCostCorresponds`. -/

section R1

open Learn.FitIcm Aeneas Aeneas.Std Result

/-- `2⁹⁶·blockReduce` at `H = I`, `x = 1`, `a = 1` in integer row units. -/
theorem diag_reduce_word (j : Fin 256) :
    2 ^ 96 * blockReduce diagBlock onesVec 1 j
      = 2 ^ 32 * (((∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val ^ 2 : ℤ)) : ℝ)
        - 2 ^ 65 * (((∑ p : Fin 8, Learn.E8Code.codewordSpec j.val p.val : ℤ)) : ℝ) := by
  rw [diagBlock_reduce]
  have hsq : (∑ p : Fin 8, codebook j p * codebook j p)
      = ∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ) ^ 2) / 2 ^ 64 :=
    Finset.sum_congr rfl fun p _ => by simp only [codebook]; ring
  have hlin : (∑ p : Fin 8, codebook j p)
      = ∑ p : Fin 8, ((Learn.E8Code.codewordSpec j.val p.val : ℝ)) / 2 ^ 32 :=
    Finset.sum_congr rfl fun p _ => by simp only [codebook]
  rw [hsq, hlin, ← Finset.sum_div, ← Finset.sum_div]
  push_cast
  ring

/-- **The exact word-cost correspondence is uninhabited at the call-site scale.** -/
theorem wordCostCorresponds_uninhabited (aa two_a : Std.I128) (q dot : Slice Std.I64)
    (haa : (2 : ℤ) ^ 64 ∣ aa.val) (htw : (2 : ℤ) ^ 33 ∣ two_a.val) :
    ¬ WordCostCorresponds diagBlock onesVec 1 aa two_a q dot := by
  intro h
  obtain ⟨c, hc, hcv⟩ := h ⟨240, by norm_num⟩
  have hw := block_cost_ok hc
  unfold wordCost at hw
  rw [diag_reduce_word] at hcv
  have hz : c.val = 2 ^ 32 * (∑ p : Fin 8, Learn.E8Code.codewordSpec 240 p.val ^ 2)
      - 2 ^ 65 * (∑ p : Fin 8, Learn.E8Code.codewordSpec 240 p.val) := by
    exact_mod_cast hcv
  rw [o8_sq_sum ⟨240, by norm_num⟩ (by norm_num)] at hz
  obtain ⟨k, hk⟩ := haa
  obtain ⟨m, hm⟩ := htw
  rw [hk, hm] at hw
  apply not_dvd_sqrt8_sq
  refine ⟨k * (q.val[240]!).val - 2 * m * (dot.val[240]!).val
      + 2 * ∑ p : Fin 8, Learn.E8Code.codewordSpec 240 p.val, ?_⟩
  have h2 : (2 : ℤ) ^ 32 * Learn.E8Code.SQRT8 ^ 2
      = 2 ^ 32 * (2 ^ 32 * (k * (q.val[240]!).val - 2 * m * (dot.val[240]!).val
          + 2 * ∑ p : Fin 8, Learn.E8Code.codewordSpec 240 p.val)) := by
    linear_combination hw - hz
  exact mul_left_cancel₀ (by norm_num) h2

/-- The residual `x = (4, 0, …, 0)`. -/
def x4 : Fin 8 → ℝ := fun p => if p = 0 then 4 else 0

/-- At `H = I`, `x = (4, 0, …, 0)`, `a = 1` the reduction is `Σ_p C(p)² − 8·C(0)`. -/
theorem diagBlock_reduce_x4 (c : CodeIndex) :
    blockReduce diagBlock x4 1 c = ∑ p, codebook c p * codebook c p - 8 * codebook c 0 := by
  unfold blockReduce
  simp [Fin.sum_univ_eight, diagBlock, x4]
  ring

/-- The 240 exact rows have `Σ_p C(p)² = 8` in real units (`codewordSpec_sq_sum`). -/
theorem exact_row_sq (c : CodeIndex) (hc : c.val < 240) :
    ∑ p, codebook c p * codebook c p = 8 := by
  have hR : (∑ p : Fin 8, (Learn.E8Code.codewordSpec c.val p.val : ℝ) ^ 2) = 2 ^ 67 := by
    exact_mod_cast codewordSpec_sq_sum c hc
  have hp : ∀ p : Fin 8, codebook c p * codebook c p
      = (Learn.E8Code.codewordSpec c.val p.val : ℝ) ^ 2 / 2 ^ 64 := by
    intro p; simp only [codebook]; ring
  rw [Finset.sum_congr rfl fun p _ => hp p, ← Finset.sum_div, hR]
  norm_num

/-- On the 240 exact rows every coordinate is at most `2` (spinor `±1`, root `0, ±2`). -/
theorem exact_row_coord_le (c : CodeIndex) (hc : c.val < 240) (k : Fin 8) : codebook c k ≤ 2 := by
  rcases codebook_row_shape c with ⟨_, s, hs⟩ | ⟨_, _, hr⟩ | ⟨h8, _⟩
  · rw [hs k]
    rcases Int.units_eq_one_or (s.1 k) with h | h <;> rw [h] <;> norm_num
  · rcases hr k with h | h | h <;> rw [h] <;> norm_num
  · omega

/-- Row 241 is `+√8·e₀`. -/
theorem codebook_241 (p : Fin 8) :
    codebook ⟨241, by norm_num⟩ p = if p = 0 then (Learn.E8Code.SQRT8 : ℝ) / 2 ^ 32 else 0 := by
  fin_cases p <;>
    simp [codebook, Learn.E8Code.codewordSpec, Learn.E8Code.o8Spec]

/-- **An `O₈` row is the strict argmin.** Row 241 costs `s² − 8s ≈ −14.63` at `s = SQRT8/2³²`;
every exact row costs `8 − 8·C(0) ≥ −8`. -/
theorem o8_row_is_argmin (c : CodeIndex) (hc : c.val < 240) :
    blockReduce diagBlock x4 1 ⟨241, by norm_num⟩ < blockReduce diagBlock x4 1 c := by
  rw [diagBlock_reduce_x4, diagBlock_reduce_x4, exact_row_sq c hc]
  have h0 := exact_row_coord_le c hc 0
  rw [Fin.sum_univ_eight]
  simp only [codebook_241]
  simp
  norm_num [Learn.E8Code.SQRT8]
  linarith

end R1

/-! ## 10 · R2: the prior art at ≤ 1.25 b/w (`anchor`)

Paper-reported Llama-2 / Qwen3 WikiText-2 perplexities from R2's table, as R2 quotes them from the
retrieved papers; not re-executed here. Grade `anchor`: the theorems prove arithmetic on the quoted
numbers, not the numbers. -/

/-- `anchor`: `(rate × 100, W2 × 100, FP16 W2 × 100)` for GLVQ-8D L2-7B and L2-70B, PV-Tuning L2-7B
(1.02) and L2-70B (1.14), OneBit L2-7B, AF1 Qwen3-32B. -/
def priorArt : List (ℕ × ℕ × ℕ) :=
  [(100, 783, 512), (100, 611, 312), (102, 828, 512), (114, 552, 312), (100, 973, 512),
    (100, 1128, 761)]

/-- **No published method at ≤ 1.25 b/w is within 48 % of its FP16 perplexity.** The 1.18 b/w
target cannot be met by subsuming an incumbent unchanged; every row is a ratio ≥ 1.48. -/
theorem prior_art_gap : ∀ r ∈ priorArt, r.1 ≤ 125 → 148 * r.2.2 < 100 * r.2.1 := by decide

/-- `anchor` (PV-Tuning Table 1, L2-7B, 1.58 b/w plain VQ): calibration-only 20.26 → PV 7.32, a
12.94 move; continuous-only reaches 8.17, carrying 12.09 of it. -/
def pvFinishTotal : ℕ := 1294

def pvFinishContinuous : ℕ := 1209

/-- **The finish is carried by the continuous parameters**: more than 93 % of PV-Tuning's move. -/
theorem prior_art_finish_share : 93 * pvFinishTotal < 100 * pvFinishContinuous := by decide

/-- **At 1.00 b/w a better shape without end-to-end finish beats finish on a plain VQ**: GLVQ-8D 7.83
against PV-Tuning 8.28 (at 1.02), both L2-7B. -/
theorem prior_art_shape_over_finish : (783 : ℕ) < 828 := by decide

/-- **R2's "+78 % group-gain defect" is `0.48697 / 0.27410 − 1`, between 77 % and 78 %**, a ratio of
two synthetic arms of R2's own model. `r2_group_gain_is_not_production` shows the 4-bit arm is not the
shipped pack, and `rt2_ceiling_as_predicted` shows the payable gain headroom on the shipped pack is
12.3 %. R2's survivor "(b) ball / learned shape" was the hypothesis RT-2 tested; `rt2_kill` and
`rt2_faces_flat` refute it on the production metric. -/
theorem r2_group_defect :
    77 * r2FreeGain < 100 * (r2GroupGain4bit - r2FreeGain) ∧
      100 * (r2GroupGain4bit - r2FreeGain) < 78 * r2FreeGain := by decide

/-! ## 11 · PF: the contact map as `H`, funnel order, Gibbs/mean-field (KILL)

Receipt `lead-1021/PF/receipt.md`, sha256 `7cd6ff26c0c8ac69`; 12 tensors,
3 seeds, 1.125 b/w; board row . The spike baseline is deliberately unweighted; production `e8_icm_step` already
minimises the `H`-weighted `J`, so the `H`-weighted arm is the production objective, not a lever the
mapping adds. -/

/-- `model` (PF table, `D_H × 10⁶`): unweighted baseline, `H`-weighted ICM, funnel + Gibbs anneal,
funnel at `T = 0`, the hyperbolic face swap `−H`, the metric-off mutation, three-tier water-fill, and
the Q4_K_M file. -/
def pfBase : ℕ := 415040

def pfHIcm : ℕ := 144582

def pfFunnelGibbs : ℕ := 163854

def pfFunnelT0 : ℕ := 145306

def pfFaceSwap : ℕ := 28012000

def pfNoMetric : ℕ := 415040

def pfWaterFill : ℕ := 273073

def pfRef : ℕ := 1961

/-- **PF kill.** The folding-specific content never beats the `H`-weighted ICM: funnel order sits within
1 % of it and the Gibbs anneal is worse. -/
theorem pf_kill :
    pfHIcm ≤ pfFunnelT0 ∧ 100 * (pfFunnelT0 - pfHIcm) < pfHIcm ∧ pfHIcm < pfFunnelGibbs := by
  decide

/-- **The metric is the whole effect.** Switching it off reproduces the baseline exactly, and the
`H`-weighted objective closes between 65 % and 66 % of the gap to Q4_K_M. -/
theorem pf_metric_is_the_lever :
    pfNoMetric = pfBase ∧ 65 * (pfBase - pfRef) < 100 * (pfBase - pfHIcm) ∧
      100 * (pfBase - pfHIcm) < 66 * (pfBase - pfRef) := by decide

/-- **The hyperbolic face is not a quantiser metric.** `−H` multiplies `D_H` by more than 193 against the
`H`-weighted arm, the measured side of `Perf.Iq1s.pair_discriminant_nonpos` (`H = XᵀX` is never
hyperbolic). -/
theorem pf_hyperbolic_face_diverges : 193 * pfHIcm < pfFaceSwap := by decide

/-- **Water-fill alone closes about a third**: between 34 % and 35 % of the gap, about half of what the
metric closes; composing the two is PF's one open measurement. -/
theorem pf_waterfill_third :
    34 * (pfBase - pfRef) < 100 * (pfBase - pfWaterFill) ∧
      100 * (pfBase - pfWaterFill) < 35 * (pfBase - pfRef) ∧ pfHIcm < pfWaterFill := by decide


/-! ## 12 · KLICM: KL-driven Gauss–Seidel ICM, `Q(d) = g·d + ½(F+λ) dᵀH d` (KILL)

Receipt `lead-1021/KLICM/receipt.md`, sha256 `81d872f66b96efee`; a 2-layer,
128-wide, 64-vocab Qwen2 fixture on the crate's Q32.32 `forward`/`backward`/`kl_mean`, head codes movable,
3 seeds, held-out batch disjoint; board row . The lead re-ran the gate
(`tests/spike_klicm.rs` sha256 `efab0a13…`, `1 passed`) and every mean reproduced. Figures are held-out
KL `× 10⁶` measured against the fit-only arm A, and ratios realised/predicted in thousandths. The
model-level descent of each accepted move is already `Learn.FitIcm.spec_quad_le`; what these clauses
record is that the quadratic's descent does not carry over to the realised KL at codeword step size. -/

/-- `model` (KLICM table, held-out): fit-only KL; the increase over A of C@4, its spread, B and D@4; C@4's
predicted drop; `−(realised/predicted)` for C@4; the increase of the `g`-free mover E′@4; the exact
re-linearisation C_X@1's calibration drop and held-out increase; the one-block realised/predicted on
the three seeds (the third negative, stored as its magnitude). -/
def klicmKlA : ℕ := 22940

def klicmC4Worse : ℕ := 165409

def klicmC4Spread : ℕ := 53320

def klicmBWorse : ℕ := 165479

def klicmD4Worse : ℕ := 165409

def klicmC4PredDrop : ℕ := 282000

def klicmC4RatioNeg : ℕ := 510

def klicmEp4Worse : ℕ := 5408

def klicmCX1CalDrop : ℕ := 21718

def klicmCX1HoldWorse : ℕ := 24734

def klicmBlockRatioS1 : ℕ := 866

def klicmBlockRatioS2 : ℕ := 110

def klicmBlockRatioS3Neg : ℕ := 1553

/-- **KLICM kill.** The quadratic predicts a drop and the held-out KL rises by more than three spreads, so
realised/predicted is negative (`−0.510`), and the `g`-free mover E′ is 30× closer to A than C. -/
theorem klicm_kill :
    0 < klicmC4PredDrop ∧ 3 * klicmC4Spread < klicmC4Worse ∧ 0 < klicmC4RatioNeg ∧
      30 * klicmEp4Worse < klicmC4Worse := by decide

/-- **The cross-block term carries nothing here.** Replacing `H` by its block diagonal (D) gives C's
held-out KL exactly, and C's edge over the all-at-once V-step B is far inside one spread. -/
theorem klicm_cross_block_null :
    klicmD4Worse = klicmC4Worse ∧ klicmC4Worse < klicmBWorse ∧
      100 * (klicmBWorse - klicmC4Worse) < klicmC4Spread := by decide

/-- **One codeword step is outside the quadratic's radius.** One seed of three realises at least half
the predicted drop of a single block's best move; one realises 11 %, one reverses sign. -/
theorem klicm_outside_radius :
    500 ≤ klicmBlockRatioS1 ∧ klicmBlockRatioS2 < 500 ∧ 0 < klicmBlockRatioS3Neg := by decide

/-- **Exact re-linearisation fits the batch, not the model.** C_X@1 lowers the calibration KL and raises
the held-out KL by more than KL_A itself (+108 %). -/
theorem klicm_relinearisation_overfits :
    0 < klicmCX1CalDrop ∧ klicmKlA < klicmCX1HoldWorse := by decide

/-! ## 13 · SCHUR: real-Schur 2×2 eigenblock faces of the quantisation error (KILL)

Receipt `lead-1021/SCHUR/receipt.md`, sha256 `93a7038608fddb23`; six square
896×896 f16 tensors of qwen2.5-0.5B (`blk.{0,11,23}.{attn_q,attn_output}`), error `E = W − Ŵ` of the
sign + FWHT-128 + E8 baseline at 1.125 b/w against an i.i.d. Gaussian `G` of equal variance, 3 sign seeds;
`tests/spike_schur.rs` sha256 `81ed8f3b…`; board row . The face of a block is the sign of its discriminant
`(a−d)² + 4bc`, the matrix image of `Nous.Model.SpectralTrichotomy.charpoly_discriminant`. Face fractions
are `× 10⁵`, `D_H` is `× 10⁶`, energy shares are in thousandths. -/

/-- `model` (SCHUR face table): per tensor, `|E_ell − mean G_ell|`, three times G's seed range, and G's seed
sd, in the order q0, o0, q11, o11, q23, o23. -/
def schurEllDelta : List ℕ := [75, 350, 170, 373, 437, 305]

def schurEll3Range : List ℕ := [2193, 1083, 4177, 1300, 1396, 1147]

def schurEllSd : List ℕ := [398, 200, 697, 232, 237, 208]

/-- `model` (SCHUR lever table, pooled `D_H`): the baseline, the face-directed lever at 8 bits per pair,
Eckart–Young rank 1 at equal bits (2-bit `u`, `v`, fp16 `σ`), rank 1 at fp16 (8× the bits), the
baseline's gap to Q4_K_M; the ρ-weighted share of block-diagonal energy, E's largest against G's. -/
def schurDhBase : ℕ := 434830

def schurDhLever : ℕ := 2082302

def schurDhRank1 : ℕ := 433345

def schurDhRank1Fp16 : ℕ := 423746

def schurGapQ4 : ℕ := 431496

def schurEShareMax : ℕ := 416

def schurGShare : ℕ := 500

def schurGShareSd : ℕ := 1

/-- **The error's face split is the i.i.d. split.** On every tensor E's elliptic fraction is within three
of G's seed ranges, so clause A of the kill criterion is false. -/
theorem schur_face_split_iid :
    (List.zip schurEllDelta schurEll3Range).all (fun p => decide (p.1 < p.2)) = true := by decide

/-- **The receipt's "below one seed sd" is not what the table says.** On four of six tensors (o0, o11,
q23, o23) the difference exceeds G's seed sd; the face-split verdict rests on the three-range criterion
alone. -/
theorem schur_delta_not_within_sd :
    (List.zip schurEllSd schurEllDelta).countP (fun p => decide (p.1 < p.2)) = 4 := by decide

/-- **The face-directed lever is dominated.** At equal bits, rank 1 lowers `D_H` and the face lever
raises it more than fourfold, so clause B of the kill criterion is false. -/
theorem schur_lever_dominated :
    schurDhRank1 < schurDhBase ∧ 4 * schurDhBase < schurDhLever := by decide

/-- **Neither arm closes a tenth of the gap.** Rank 1 at equal bits closes 0.34 % of the gap to Q4_K_M,
and at 8× the bits 2.6 %. -/
theorem schur_closes_under_tenth :
    10 * (schurDhBase - schurDhRank1) < schurGapQ4 ∧
      10 * (schurDhBase - schurDhRank1Fp16) < schurGapQ4 := by decide

/-- **The structure that does exist is coupling, not face.** E's largest block-diagonal energy share is
more than 80 of G's seed sds below G's, so E's excess energy sits in the strictly-upper coupling, which a
per-block lever cannot reach. -/
theorem schur_nonnormal_coupling :
    schurEShareMax + 80 * schurGShareSd < schurGShare := by decide

/-! ## 14 · KLICM-2: top-`k` accept and the KL line search (KILL; the line search transfers 23 %)

Receipt `lead-1021/KLICM2/receipt.md`, sha256 `71a06e9efb7c1b7f`; the §12
fixture, 3 seeds, held-out batch disjoint; `tests/spike_klicm2.rs` sha256 `7f350b8f…`, re-run by the lead
(`1 passed`, every mean reproduced). KL is `× 10⁶`, measured against fit-only arm A; ratios
realised/predicted are in thousandths. -/

/-- `model` (KLICM-2 means): `KL_A` held-out; K1's held-out drop, spread, held-out and calibration ratio;
K8's calibration increase; the held-out increase of K8, K64, K256, K1024; LS8's held-out drop, spread,
ratio; the moves and repeated moves of seed `0x33` K8 and seed `0x22` K1024; mean moves of K8 and LS8. -/
def klicm2KlAHold : ℕ := 22060

def klicm2K1Drop : ℕ := 3049

def klicm2K1Spread : ℕ := 2213

def klicm2K1HoldRatio : ℕ := 174

def klicm2K1CalRatio : ℕ := 673

def klicm2K8CalWorse : ℕ := 1210

def klicm2K8Worse : ℕ := 4647

def klicm2K64Worse : ℕ := 60441

def klicm2K256Worse : ℕ := 217295

def klicm2K1024Worse : ℕ := 610240

def klicm2LsDrop : ℕ := 5179

def klicm2LsSpread : ℕ := 3939

def klicm2LsHoldRatio : ℕ := 144

def klicm2K8Moves : ℕ := 128

def klicm2K8Repeats : ℕ := 109

def klicm2K1024Moves : ℕ := 16383

def klicm2K1024Repeats : ℕ := 15359

def klicm2LsMoves : ℕ := 45

/-- **KLICM-2 kill.** No arm realises half its predicted held-out drop; K8 raises the held-out KL. -/
theorem klicm2_kill :
    klicm2K1HoldRatio < 500 ∧ klicm2LsHoldRatio < 500 ∧ 0 < klicm2K8Worse := by decide

/-- **The quadratic's trust region is one block per re-linearisation.** K1 realises at least half its
predicted calibration drop; K8 raises the calibration KL. -/
theorem klicm2_trust_region_one_block :
    500 ≤ klicm2K1CalRatio ∧ 0 < klicm2K8CalWorse := by decide

/-- **Past one block the accept is a 2-cycle.** The held-out increase grows with `k`, and more than 80 %
of the moves at `k = 8` and `k = 1024` re-move a block already moved. -/
theorem klicm2_two_cycle :
    klicm2K8Worse < klicm2K64Worse ∧ klicm2K64Worse < klicm2K256Worse ∧
      klicm2K256Worse < klicm2K1024Worse ∧ 8 * klicm2K8Moves ≤ 10 * klicm2K8Repeats ∧
      8 * klicm2K1024Moves ≤ 10 * klicm2K1024Repeats := by decide

/-- **The KL line search transfers a fifth of the fit error, at 1.3 spreads.** LS8's held-out drop is
above one spread and below three, between 20 % and 25 % of `KL_A`; its edge over K1 is inside K1's
spread. The control K8 is LS8 with the accept test removed and takes more moves. -/
theorem klicm2_line_search_partial :
    klicm2LsSpread < klicm2LsDrop ∧ klicm2LsDrop < 3 * klicm2LsSpread ∧
      klicm2KlAHold ≤ 5 * klicm2LsDrop ∧ 4 * klicm2LsDrop < klicm2KlAHold ∧
      klicm2LsDrop - klicm2K1Drop < klicm2K1Spread ∧ klicm2LsMoves < klicm2K8Moves := by decide

/-! ## 15 · TEQ: the per-layer transfer cost equation (GO)

Receipt `lead-1021/TEQ/receipt.md`, sha256 `3377bbeea7675d05`;
`tests/spike_transfer_eq.rs` sha256 `1fa2c6c6…`, re-run by the lead (`1 passed`, units within 2.4 %). One
qwen2.5-0.5B layer; walls in tenths of a second. The equation's units are `measured`; the projections are
`model`; R4 and R5 borrow the engine's Q8_0 rate on its own operands and are `anchor`. -/

/-- `model` (TEQ §3–§5): projected and recorded fit and layer walls; the largest per-tensor residual in
thousandths; the sweep's share of the layer; the minimum-work wall at the 5-limb rate (9.7e11 MAC at
39 GMAC/s); the rung walls R1–R5; the 0.5B transfer bar (24 layers, fp32 reference) and one run's
ceiling. -/
def teqFitProj : ℕ := 4740

def teqFitRec : ℕ := 4613

def teqLayerProj : ℕ := 5710

def teqLayerRec : ℕ := 5450

def teqMaxResidual : ℕ := 112

def teqSweep : ℕ := 2450

def teqMinWork : ℕ := 249

def teqR1 : ℕ := 3750

def teqR2 : ℕ := 3210

def teqR3 : ℕ := 380

def teqR4 : ℕ := 176

def teqR5 : ℕ := 29

def teqTransferBar : ℕ := 939

def teqRunCeiling : ℕ := 6000

/-- **The equation reproduces the record.** Fit and layer projections are within 25 %, and so is every
per-tensor row. -/
theorem teq_equation_fits :
    4 * teqFitProj ≤ 5 * teqFitRec ∧ 4 * teqLayerProj ≤ 5 * teqLayerRec ∧ teqMaxResidual < 250 := by
  decide

/-- **The sweep dominates, and the layer is far from its minimum work.** The `e8_icm_step` sweep is over
40 % of the layer; the layer runs more than 21× its minimum-work wall at the shaped kernel's rate. -/
theorem teq_sweep_dominates :
    4 * teqLayerRec ≤ 10 * teqSweep ∧ 21 * teqMinWork ≤ teqLayerRec := by decide

/-- **Lazy batching alone does not reach the bar.** R1 and R2 stay above a 25 s layer; R3 (every product
on the 5-limb kernel) misses it, and R4 (residual at the Q8_0 rate) meets it. -/
theorem teq_rungs_layer :
    250 < teqR2 ∧ teqR2 < teqR1 ∧ 250 < teqR3 ∧ teqR4 ≤ 250 := by decide

/-- **The 0.5B transfer bar needs R5.** At 24 layers R4 fits one run's 600 s ceiling but is 4.5× the
93.9 s reference; only R5 (every product at the Q8_0 rate) is under it. R3 does not fit one run. -/
theorem teq_transfer_bar_needs_r5 :
    24 * teqR4 ≤ teqRunCeiling ∧ teqRunCeiling < 24 * teqR3 ∧ teqTransferBar < 24 * teqR4 ∧
      24 * teqR5 ≤ teqTransferBar := by decide

/-! ## 16 · ATTRIB: where the fit time outside the sweep goes (REFUTED)

Receipt `lead-1021/ATTRIB/receipt.md`, sha256 `98e153f40b5b859b`;
`tests/spike_fit_attrib.rs` sha256 `015336ec…`, re-run by the lead on `t_k_128_896`; board row
. One qwen2.5-0.5B layer; walls in tenths of a second, rates in tenths of
GMAC/s, control residuals in thousandths. Walls are `measured` (each phase called through its public entry
and multiplied by the calls `fold_fit` makes; `perf` is refused at `perf_event_paranoid=2`); MAC and call
counts are `model`. The hypothesis was: the non-sweep fit time is at least half `limb::*` calls below
10 GMAC/s. -/

/-- `measured` (ATTRIB §3–§4): the layer's fit and its sweep and non-sweep shares; the non-sweep split
into `nb_refine` (`e8_joint_icm_2plane`, a `limb` call at 1–2 GMAC/s), `gptq_e8` (scalar block-Cholesky
and CVP scan, no kernel call), the `c`/`r`/`d` steps, `objective` and the unphased residue; the slowest
shaped-product rate measured against TEQ's assumed 2.2 GMAC/s; the steps' wall after a swap to the
5-limb rate (39 GMAC/s). -/
def attribFit : ℕ := 5331

def attribSweep : ℕ := 2465

def attribNonSweep : ℕ := 2866

def attribNbRefine : ℕ := 1157

def attribGptq : ℕ := 855

def attribSteps : ℕ := 546

def attribObjective : ℕ := 69

def attribResidue : ℕ := 240

def attribShapedRateMin : ℕ := 71

def teqShapedRateAssumed : ℕ := 22

def attribStepsSwapped : ℕ := 90

/-- `measured` (ATTRIB §2): phase sum against measured `fit_tensor` wall, in thousandths of the wall:
`k` −2.1 %, `o` −8.2 %, `gate` +11.3 %. -/
def attribCtrlK : ℕ := 21

def attribCtrlO : ℕ := 82

def attribCtrlGate : ℕ := 113

/-- **The sweep is reproduced independently.** Four measured shapes give the sweep within 1 % of TEQ's
245 s, and the fit splits into sweep and non-sweep exactly. -/
theorem attrib_sweep_reproduces :
    attribSweep ≤ teqSweep + teqSweep / 100 ∧ teqSweep ≤ attribSweep ∧
      attribSweep + attribNonSweep = attribFit := by decide

/-- **The hypothesis is refuted.** The non-sweep time is five mechanisms summing to the measured total;
the `limb` calls below 10 GMAC/s (`nb_refine`) hold under half of it. -/
theorem attrib_slow_kernel_not_dominant :
    attribNbRefine + attribGptq + attribSteps + attribObjective + attribResidue = attribNonSweep + 1 ∧
      2 * attribNbRefine < attribNonSweep := by decide

/-- **The largest non-sweep terms are `nb_refine` then `gptq_e8`, and the second is no kernel call.**
Together they hold over two thirds of the non-sweep time. -/
theorem attrib_largest_terms :
    attribGptq < attribNbRefine ∧ attribSteps < attribGptq ∧
      3 * (attribNbRefine + attribGptq) ≥ 2 * attribNonSweep := by decide

/-- **TEQ's shaped-product premise is false.** The slowest shaped product runs over 3× TEQ's assumed
2.2 GMAC/s, and a swap of every shaped product to the 5-limb rate leaves the fit above 470 s, over 12×
R3's 38 s/layer. R3 is not reachable by the rung it names. -/
theorem attrib_teq_premise_false :
    3 * teqShapedRateAssumed < attribShapedRateMin ∧
      12 * teqR3 < attribFit - attribSteps + attribStepsSwapped := by decide

/-- **The control is two of three.** `k` and `o` close within 10 %; `gate` misses by 1.3 points. -/
theorem attrib_control_two_of_three :
    attribCtrlK < 100 ∧ attribCtrlO < 100 ∧ 100 < attribCtrlGate := by decide

/-! ## §17 Lazy-batch residual in `sweep_blocks` (SWEEP-LB, TEQ R1/R2), board 

`measured`: one `e8_icm_step` sweep, `t(sweeps = 1) − t(sweeps = 0)`, 3 paired runs per shape, mean and
`[min, max]` in milliseconds (`icm_lazy_batch_microbench`, `tests/learn_fold_icm_speed.rs`). The base arm is
the `84f2d8d3` `icm.rs` in the lane's tree; the new arm is the lead's re-run on the live tree (`icm.rs`
sha256 `1a2e89f9…`). GPTQ's lazy batch (Frantar et al., arXiv 2210.17323 §3) over `GROUP = 128` columns,
row bands folded into the kernel's `m`. `tested`: `J` and every block code equal to the base arm on three
`(64, 512)` fixtures and the `down` shape `(896, 4864)`. -/
def sweeplbDownBase : ℕ := 26527

def sweeplbDownBaseMin : ℕ := 26494

def sweeplbDownNew : ℕ := 6733

def sweeplbDownNewMax : ℕ := 6765

def sweeplb896Base : ℕ := 1921

def sweeplb896New : ℕ := 423

/-- The brief's projection for the `down` sweep, in milliseconds. -/
def sweeplbDownProj : ℕ := 6000

/-- `down`, 2 sweeps: `J` of the new and base arms, and the codes that agree out of the total. -/
def sweeplbJBase : ℕ := 6998698622758382

def sweeplbJNew : ℕ := 6998698622758382

def sweeplbAgree : ℕ := 544768

def sweeplbBlocks : ℕ := 544768

/-- `measured`: the remaining out-of-group flush at inner 128, against the 5-limb kernel at inner 4864, in
tenths of GMAC/s. -/
def sweeplbFlushRate : ℕ := 37

def sweeplbKernelRate : ℕ := 390

/-- `model`: the per-layer sweep after the change, in tenths of a second: ATTRIB's 246.5 s scaled by the
`down` ratio's conservative bound `max_new / min_base`, rounded up. -/
def sweeplbSweepProj : ℕ := 630

/-- **The lazy batch is ≥ 3.9× on `down`, conservatively.** The worst new run is under a tenth of 39 times
the best base run, and the `(896, 896)` shape is over 4×. -/
theorem sweeplb_down_ratio :
    39 * sweeplbDownNewMax ≤ 10 * sweeplbDownBaseMin ∧ sweeplbDownBaseMin ≤ sweeplbDownBase ∧
      4 * sweeplb896New < sweeplb896Base := by decide

/-- **The brief's ≤ 6 s projection is missed.** -/
theorem sweeplb_projection_missed : sweeplbDownProj < sweeplbDownNew := by decide

/-- **No re-pin.** On `down` the objective is equal and every code agrees. -/
theorem sweeplb_exact : sweeplbJNew = sweeplbJBase ∧ sweeplbAgree = sweeplbBlocks := by decide

/-- **The flush is the sweep's next term.** It runs under a tenth of the kernel's rate at full inner. -/
theorem sweeplb_flush_residual : 10 * sweeplbFlushRate < sweeplbKernelRate := by decide

/-- **The sweep no longer dominates the fit** (`model`). The projected sweep bounds ATTRIB's sweep scaled by
the conservative ratio; the non-sweep time is over 4× it; the projected fit is still over 9× R3. -/
theorem sweeplb_nonsweep_dominates :
    attribSweep * sweeplbDownNewMax ≤ sweeplbDownBaseMin * sweeplbSweepProj ∧
      4 * sweeplbSweepProj < attribNonSweep ∧
      9 * teqR3 < attribFit - attribSweep + sweeplbSweepProj := by decide

/-! ## §18 The red `fold_fit` pins: first bad commit `ee33eacc`, a route change carrying a defect

`tested`, PIN-BISECT: `cargo test --release --test learn_fold_fold_fit` over `57d21789..84f2d8d3`
(104 commits, 7 bisection probes). The first bad commit `ee33eacc` moves the eight `steps.rs` product
sites onto `limb.rs` at 3 limbs (`i128` count 33 → 20); its own gate (`learn_fold_steps`, 17 passed)
never ran the `fold_fit`, `nb_refine` and `tensor` pins whose words it moves. The `fold_fit` oracle
calls the same `src` bodies, so its `1e-9` pin moves with any `steps.rs` change. One moved test is a
refusal: `qmlp_top_rows_and_boundaries` fits at the parent `7a56eb47` and returns
`Fit(NotPositiveDefinite)` at `ee33eacc`. A revert is refused by §0 K2: the kernel-route hook refuses a
grown `i128` count. Board row . -/
def pinRange : ℕ := 104

def pinProbes : ℕ := 7

/-- `fold_fit` seed `0x3d3d0002` words: the recorded pin, and the moved value at `ee33eacc` and at
`84f2d8d3`. -/
def pinSeed2Pin : ℕ := 14662245486467828602

def pinSeed2FirstBad : ℕ := 11813287334381752278

def pinSeed2RangeEnd : ℕ := 11813287334381752278

/-- `fold_fit` oracle `trace[0]` at `ee33eacc` and its pin, in units of `10⁻¹⁴`. -/
def pinTraceFirstBad : ℕ := 9553691067174077

def pinTracePin : ℕ := 9553694787010723

/-- `nb_refine` words: the pin, the value at `ee33eacc`, the value at `84f2d8d3`. -/
def pinNbPin : ℕ := 15580025767902961563

def pinNbFirstBad : ℕ := 14276473178736957625

def pinNbRangeEnd : ℕ := 17063009677069469865

/-- `steps.rs` `i128` count before and after `ee33eacc`. -/
def pinI128Before : ℕ := 33

def pinI128After : ℕ := 20

/-- Whether `qmlp_top_rows_and_boundaries` fits at the parent and at the first bad commit. -/
def pinFitsParent : Bool := true

def pinFitsFirstBad : Bool := false

/-- **The search was a bisection.** 104 commits need 7 probes, and 7 were run. -/
theorem pin_bisect_probes : 2 ^ (pinProbes - 1) < pinRange ∧ pinRange ≤ 2 ^ pinProbes := by decide

/-- **One commit moves the `fold_fit` witness**: the seed-2 words differ from the pin and are the same at
the first bad commit and the range end. -/
theorem pin_seed2_one_commit :
    pinSeed2FirstBad ≠ pinSeed2Pin ∧ pinSeed2FirstBad = pinSeed2RangeEnd := by decide

/-- **The oracle drift is inside the limb bound and outside the pin.** The `trace[0]` move is more than
`10⁻⁹` relative (the pin's tolerance) and at most `2·10⁻⁵` (the test's stated limb bound). -/
theorem pin_oracle_drift :
    pinTracePin < (pinTracePin - pinTraceFirstBad) * 1000000000 ∧
      (pinTracePin - pinTraceFirstBad) * 50000 ≤ pinTracePin := by decide

/-- **`nb_refine` moved twice in the range**: at the first bad commit and again after it. -/
theorem pin_nb_moved_twice :
    pinNbFirstBad ≠ pinNbPin ∧ pinNbRangeEnd ≠ pinNbFirstBad ∧ pinNbRangeEnd ≠ pinNbPin := by decide

/-- **A defect, not only a moved digest; the fix is forward.** The fixture fitted at the parent and
refuses at the first bad commit, so no re-pin absorbs it; a revert grows the `i128` count the
kernel-route hook refuses. -/
theorem pin_defect_fix_forward :
    pinFitsParent = true ∧ pinFitsFirstBad = false ∧ pinI128After < pinI128Before := by decide

/-! ## §19 `gptq_e8`'s `feedback_u` on the limb kernel: KILL, and the 52.3 s is not the factor

GFACT, `tested` and `measured` on base `66ef2f9c`. The attempt moves the block-Cholesky trailing
update of `inv_factor_upper` onto `syrk_q32` (`src/learn/solve/chol_kern.rs`, sha256 `3c56cadb…`, not
installed). The speeds are means of 3 alternating pairs. The receipt is
`lead-1021/GFACT/receipt.md`. Board row .

Three grounds:
1. A factor word is `≤ 2³²`, so a 32-lane block exponent is `≤ 35`, which is `⌈35/7⌉ = 5` signed-byte
   digits. The exact product needs the pair `(4,4)`. `kept_pairs` keeps `s + t < limbs`
   (`src/engine/kern/limb.rs:413`), so the exact product needs 9 limbs, against `MAX_LIMBS = 8`
   (`limb.rs:86`).
2. The diagonal solve, the panel solve and `inv_row_entry` consume the unrounded `i128` dot. The kernel
   returns the `Q32.32` word, and 1125 of 2016 inverse entries change at `n = 129`.
3. The one reachable stage is slower than the contract. Even an infinitely fast trailing update leaves
   `inv_factor_upper(4864)` over `1/5` of its time, against the brief's `≥ 5×`.

Re-attribution: ATTRIB timed `gptq_e8` at `down` (896, 4864) at 52.324 s (1 rep). There,
`inv_factor_upper(4864)` is 6.312 s, so more than 80 % of that term lies outside the factor. The
largest remaining body is the `beam_row` scan (lane BEAM-CVP). -/

/-- Block exponent bound and digit width of a factor word. -/
def gfBlockExp : ℕ := 35

def gfDigitBits : ℕ := 7

def gfDigits : ℕ := (gfBlockExp + gfDigitBits - 1) / gfDigitBits

def gfMaxLimbs : ℕ := 8

/-- Speed ratios contract/kernel in thousandths: `(n, limbs)` = (4864, 3), (4864, 8), (896, 3),
(896, 8). -/
def gfRatio4864L3 : ℕ := 716

def gfRatio4864L8 : ℕ := 278

def gfRatio896L3 : ℕ := 925

def gfRatio896L8 : ℕ := 250

/-- `inv_factor_upper(4864)` total and its factor stage, in ms. -/
def gfInvTotalMs : ℕ := 6312

def gfFactorMs : ℕ := 1851

/-- Inverse entries that change when the dot returns as a word, of all, at `n = 129`, block `[0,64)`. -/
def gfInvChanged : ℕ := 1125

def gfInvEntries : ℕ := 2016

/-- ATTRIB's `gptq_e8` time at `down`, in ms. -/
def gfAttribDownMs : ℕ := 52324

/-- **The exact product does not fit the kernel.** Five digits per operand; the pair `(4,4)` needs
`limbs > 8`, and the kernel caps at 8. -/
theorem gf_exact_needs_nine_limbs :
    gfDigits = 5 ∧ gfMaxLimbs < (gfDigits - 1) + (gfDigits - 1) + 1 := by decide

/-- **The word-returning dot changes the inverse.** A nonzero share of entries move. -/
theorem gf_word_dot_moves_inverse : 0 < gfInvChanged ∧ gfInvChanged < gfInvEntries := by decide

/-- **The kernel route is slower at every measured shape and limb count.** -/
theorem gf_kernel_slower :
    gfRatio4864L3 < 1000 ∧ gfRatio4864L8 < 1000 ∧ gfRatio896L3 < 1000 ∧ gfRatio896L8 < 1000 := by
  decide

/-- **The 5× bar is unreachable on this primitive.** With the trailing update at zero cost, the
inverse stage alone is more than `1/5` of the total. -/
theorem gf_ceiling_below_five : gfInvTotalMs < 5 * (gfInvTotalMs - gfFactorMs) := by decide

/-- **ATTRIB's `gptq_e8` term is mostly not the factor.** More than 80 % of the 52.3 s lies outside
`inv_factor_upper`. -/
theorem gf_attrib_residual :
    gfAttribDownMs * 8 < (gfAttribDownMs - gfInvTotalMs) * 10 := by decide

/-! ## §20 The lazy-batch residual in `joint_band`: bit-identical, 1.46×, and the serial decision is the floor

NB-JOINT, `tested` and `measured` on base `7e3c4dbd`. `src/learn/fold/fit/icm.rs` (sha256 `f57a1791…`)
splits `joint_band` into `joint_bands_batched` (the full sweep: one flush per `GROUP = 128`-column group,
an in-batch prefix per 8-block, one band) and `joint_bands_per_block` (the base body, kept for a caller
`blocks` subset). The receipt is `lead-1021/NBJOINT/receipt.md`. Board row
.

- Speed: per sweep of `e8_joint_icm_2plane`, `(t(2) − t(0))/2`, 3 reps per arm. The arms ran in
  separate processes, with the base file swapped in and then restored (`cmp` identical).
- Identity: the `T = 0` arm, 2 sweeps, matches the base arm's captured `J`, both plane digests and
  `moves` on four fixtures, the (896, 896) one at 71 % movement. The scheduled arm is deterministic but
  not identical: a regrouped `i128` sum flips a Boltzmann draw that no margin protects.
- The serial `best_pair_q` scan: 100,352 calls for one sweep take 13.213 s. That is the floor of any
  residual regroup, so the brief's `≥ 3×` at (896, 896) is out of reach on this route. The next lane
  (NB-JOINT-2) moves that decision onto the pool. -/

/-- Seconds per sweep in ms, `(o, n) = (896, 896)`: base mean, base min, new mean, new max. -/
def nbBase896 : ℕ := 20292

def nbBase896Min : ℕ := 20263

def nbNew896 : ℕ := 13905

def nbNew896Max : ℕ := 13918

/-- The same at `(128, 896)`. -/
def nbBase128 : ℕ := 5919

def nbBase128Min : ℕ := 5909

def nbNew128 : ℕ := 2076

def nbNew128Max : ℕ := 2085

/-- One sweep's serial `best_pair_q` decision at (896, 896), in ms. -/
def nbPair896 : ℕ := 13213

/-- The (896, 896) identity fixture: `J` and `moves`, base arm and new arm. -/
def nbJBase : ℕ := 432176896298905

def nbJNew : ℕ := 432176896298905

def nbMovesBase : ℕ := 142478

def nbMovesNew : ℕ := 142478

/-- **Faster at both shapes, with disjoint spreads.** The slowest new rep beats the fastest base
rep. -/
theorem nb_faster_disjoint : nbNew896Max < nbBase896Min ∧ nbNew128Max < nbBase128Min := by decide

/-- **The ratios.** At least 1.45× at (896, 896) and 2.8× at (128, 896), on the means. -/
theorem nb_ratios : nbNew896 * 145 ≤ nbBase896 * 100 ∧ nbNew128 * 28 ≤ nbBase128 * 10 := by decide

/-- **The (896, 896) witness is bit-identical.** `J` and `moves` agree between arms. -/
theorem nb_identity_896 : nbJNew = nbJBase ∧ nbMovesNew = nbMovesBase := by decide

/-- **The residual work was cut by over 90 %.** What is left after the decision fell from 7.08 s
to 0.69 s. -/
theorem nb_residual_cut : (nbNew896 - nbPair896) * 10 < nbBase896 - nbPair896 := by decide

/-- **`≥ 3×` is unreachable by a residual regroup.** Even with no residual left, the base is below
three times the serial decision. -/
theorem nb_three_x_unreachable : nbBase896 < 3 * nbPair896 := by decide

/-- **The decision is the sweep now.** Over 95 % of the new sweep is the serial pair scan. -/
theorem nb_decision_dominates : nbNew896 * 95 < nbPair896 * 100 := by decide

/-! ## §21 `beam_row`'s 256-way scan made multiply-free: bit-identical, 1.53× per pass, and not the `down` residual

BEAM-CVP, `tested` and `measured` on base `3b4758f7`. `src/learn/fold/fit/gptq_e8.rs` (sha256
`760153f0…`) replaces the per-candidate multiply in the 256-row shell scan of `beam_row` with a table read.
`tests/learn_fold_gptq_e8_beam.rs` (sha256 `5688f713…`) carries `scan_arms_are_identical` and the probe. The
receipt is `lead-1021/BEAMCVP/receipt.md`. Board row
.

- R0 attribution at (896, 896), `wscale = true` (production): scan 47.6 %, `wscale` tail 24.4 %,
  rebuild + feedback 16.9 %, align 7.4 %, `select_k` 3.8 % of `beam_row` CPU.
- R1 is bit-identical on the pinned digests. The control `t[2][1] += 1` turns `scan_arms_are_identical`
  red at `k = 132`.
- R2 (a pruned scan) is killed by its ceiling: after R1 the scan is 11.4 % of row work.
- The `down` residual: GFACT's factor plus 76 beam calls at the base rate is under a quarter of the
  ATTRIB `gptq_e8` at `down`. That row is `model` (the per-call beam time is measured at n = 896); the
  next lane, GFEED, measures `inter_group_feedback` there. -/

/-- The scan stage's CPU per call at (896, 896), `wscale = true`, in µs: base, new. -/
def bcScanBase : ℕ := 561500

def bcScanNew : ℕ := 80000

/-- `beam_pass` per call at (896, 896), in µs, `wscale = true`: base mean, base min, new mean, new max. -/
def bcPassBase : ℕ := 80400

def bcPassBaseMin : ℕ := 80200

def bcPassNew : ℕ := 52500

def bcPassNewMax : ℕ := 52900

/-- The same, `wscale = false`: base mean, new mean. -/
def bcPassBaseNoW : ℕ := 60000

def bcPassNewNoW : ℕ := 31400

/-- Scan fraction of row work after R1, in per-mille. -/
def bcScanAfterPermille : ℕ := 114

/-- Whole `gptq_e8` at (896, 896), in 0.1 ms: ATTRIB (n = 1) and the new mean (n = 3). -/
def bcGptqAttrib : ℕ := 22090

def bcGptqNew : ℕ := 18262

/-- At `down` (896, 4864), in µs: ATTRIB `gptq_e8`, GFACT `inv_factor_upper(4864)`, and the beam calls
(38 groups × 2). -/
def bcDownGptq : ℕ := 52324000

def bcDownFactor : ℕ := 6312000

def bcDownBeamCalls : ℕ := 76

/-- **The scan is 7× cheaper.** -/
theorem bc_scan_seven_x : 7 * bcScanNew ≤ bcScanBase := by decide

/-- **Faster per pass, with disjoint spreads** (`wscale = true`). -/
theorem bc_pass_disjoint : bcPassNewMax < bcPassBaseMin := by decide

/-- **The ratios.** At least 1.53× with `wscale` and 1.9× without, on the means. -/
theorem bc_pass_ratios :
    bcPassNew * 153 ≤ bcPassBase * 100 ∧ bcPassNewNoW * 190 ≤ bcPassBaseNoW * 100 := by decide

/-- **R2's ceiling is below 1.15×.** Removing the whole post-R1 scan leaves `1 − 0.114` of the row
work. -/
theorem bc_r2_ceiling : 1000 * 1000 < (1000 - bcScanAfterPermille) * 1150 := by decide

/-- **Whole `gptq_e8` at (896, 896) is at least 1.20× faster** than the ATTRIB sample. -/
theorem bc_gptq_ratio : bcGptqNew * 120 ≤ bcGptqAttrib * 100 := by decide

/-- **The `down` residual is neither the factor nor the beam** (`model`): both together are under a
quarter of `gptq_e8` at `down`, even at the base beam rate. -/
theorem bc_down_residual :
    (bcDownFactor + bcDownBeamCalls * bcPassBase) * 4 < bcDownGptq := by decide

end QuantVerdicts

end Learn
