import Mathlib.Tactic
import Mathlib.Analysis.Real.Pi.Bounds
import Learn.FitIcm

set_option linter.apeiron.notation false

/-!
# The 1-bit quantiser spikes: every clause against its measured constant

Seven receipts. The six spikes ran on qwen2.5-0.5B (`blk.{0,11,23}.{attn_q,attn_v,ffn_gate,ffn_down}`,
imatrix-weighted
`D_H = Σh e² / Σh w²`) and on synthetic `N(0,1)`; R1 is a refinement audit (§9). Receipts, by sha256 prefix:

| spike | receipt | sha256 |
|---|---|---|
| IC (cipher diffusion, keyed streams) | `/home/socol/oc-tmp/escalation/lead-1021/IC/receipt.md` | `728f6afdbfe99030` |
| RT-1 (minimal cut, greedy allocation) | `/home/socol/oc-tmp/escalation/lead-1021/RT/receipt.md` | `041746b6d7a1c6a9` |
| QM (σ-face metric transforms) | `/home/socol/oc-tmp/escalation/lead-1021/QM/receipt.md` | `80b97beac35c3301` |
| R1 (refinement audit of `Learn.FitIcm`) | `/home/socol/oc-tmp/escalation/lead-1021/R1/receipt.md` | `b5daa7c47da05abf` |
| R2 (prior art, codebook model) | `/home/socol/oc-tmp/escalation/lead-1021/R2/receipt.md` | `543662be74dbd54a` |
| RT-2 (radius as bulk coordinate) | `/home/socol/oc-tmp/escalation/lead-1021/RT2/receipt.md` | `6203395b72d30be9` |
| PF (contact map as `H`, funnel, Gibbs) | `/home/socol/oc-tmp/escalation/lead-1021/PF/receipt.md` | `7cd6ff26c0c8ac69` |

Every figure below is a `model` input copied from its receipt table: a distortion in millionths
(`D × 10⁶`), a rate in ten-thousandths of a bit. No theorem here produces a measurement. Every
`D_H` is a weight-space proxy: no clause here moves, bounds or predicts KL, and nothing here is on the
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

Receipt `/home/socol/oc-tmp/escalation/lead-1021/RT2/receipt.md`, sha256 `6203395b72d30be9`; 12 tensors,
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

Receipt `/home/socol/oc-tmp/escalation/lead-1021/PF/receipt.md`, sha256 `7cd6ff26c0c8ac69`; 12 tensors,
3 seeds, 1.125 b/w; board row `APEIRON-FORM-QUANT-W1-T01`. The spike baseline is deliberately unweighted; production `e8_icm_step` already
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

end QuantVerdicts

end Learn
