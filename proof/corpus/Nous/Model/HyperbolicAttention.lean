import Apeiron.Trichotomy.ChannelsAnyRing
import Apeiron.Trichotomy.Flow
import Apeiron.Physics.Wick
import Nous.Model.RopeOrbit
import Nous.Model.EllipticAttention
import Nous.Model.Weighting
import Nous.Model.FaceRecurrence
import Substrate.Algebra.Neural.AttentionSpinor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.LinearAlgebra.Dimension.Finite
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.Tactic

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false

/-!
# Vector-level attention spec on Qwen-style heads in the two-slot algebra

A Qwen-style head is `Fin n → RingCell ℝ`, one RoPE pair `q_p ∈ ℝ[x]/(x²+1)` per frequency.
This module states the vector-level specification the engine is derived from: the score, the
weight, the streaming scan, and — decided by the kernel, not by prose — which exponent families
admit an exact constant-size state and which do not.

## 1. The RoPE-relative elliptic score

`qwenScore n θ t s q k` is the Qwen score of query position `t` against key position `s`: for each
pair `p` the query pair is rotated by `t·θ_p` and the key pair by `s·θ_p`, and the pairs are paired
through the elliptic phase alignment at `σ = -1`,

    Σ_p (ringMul (-1) (ropeRot (t θ_p) (q p)) (ringConj (ropeRot (s θ_p) (k p)))).1.

`qwenScore_eq_rel` shows it is `Σ_p rhoDot (q p) (rhoAct ((s - t) θ_p) (k p))`, the rotation by the
*relative* angle; `qwenScore_relative` shows the score depends on `t, s` only through `t - s`
(rotations compose, RoPE: Su et al. 2021/2024; `Nous.Model.RopeOrbit.rho_relative`,
`Nous.Model.EllipticAttention.elliptic_rope_relative_score`). Grade: `proved`.

## 2. The softmax weight is the upper channel of the hyperbolic flow

The hyperbolic flow at `σ = 1` is `flow 1 s = (cosh s, sinh s)`
(`Apeiron.Trichotomy.Flow.flow_one`), and the channel reads at the root `r = 1` are

    ringUpper 1 (flow 1 s) = exp s,   ringLower 1 (flow 1 s) = exp (-s).

So the unnormalized softmax weight is the upper channel and the complementary mass is the lower
channel; their ratio is the two-class softmax (`channelWeight_flow_one`, reusing
`Nous.Model.Weighting.channelWeight` and `softmax2`). The two faces are one flow at two signs:
`Apeiron.Physics.Wick.flow_wick` is the pointer that the elliptic turn is this boost at imaginary
time. Grade: `proved`.

## 3. Exact streaming (online softmax), vector-valued

`softmax_attention_is_online_spinor_vec` lifts the landed scalar online-softmax identity
(`Sanguine.AttentionSpinor.softmax_attention_is_online_spinor`, with `oaDen_fuse`, `oaNum_fuse`)
to values in `Fin d → ℝ`: one `oaFuseV` per token, the running `(m, den, num)` state, and the
readout `num / den` equals `(Σ e^{sc}·v) / (Σ e^{sc})` exactly. `oaDenV_scan_append` /
`oaNumV_scan_append` give the chunk-fusion law: the denotation of a concatenated scan is the
denotational sum of the two scans. Grade: `proved`.

## 4. The constant-state question, decided by the kernel

**Negative (`exp_linearIndependent`, `exp_kernel_not_finite_rank`).** Distinct exponentials
`q ↦ e^{k q}` are linearly independent over `ℝ`: a vanishing finite combination has all
coefficients zero. Equivalently the exponential kernel `(q, k) ↦ e^{q k}` admits no finite-rank
factorization `Σ_j φ_j(q) ψ_j(k)` at any dimension `D`, because `D + 1` distinct keys give `D + 1`
linearly independent functions inside a `D`-dimensional span. Hence full `exp(q·k)` softmax
attention is not computed by any fixed finite-dimensional carry; it requires the key–value scan.
Grade: `proved`.

**Positive (`boost_factor`, `boost_finite_rank_recurrence`).** Where the exponent factorizes —
a finite-rank content kernel `Σ_j φ_j(q) ψ_j(k)` times the hyperbolic relative-position boost
`e^{λ(t - s)} = ringUpper 1 (flow 1 (λ t)) · ringLower 1 (flow 1 (λ s))` — normalized attention is
exactly the fixed-size carry read (`Sanguine.AttentionSpinor.kernel_attention_is_recurrence`):
`N` tokens accumulate `num_s = ψ(k_s) e^{-λ s} v_s` and `den_s = ψ(k_s) e^{-λ s}`, and the query
at position `t` reads `Σ_j φ_j(q) e^{λ t} num_j / Σ_j φ_j(q) e^{λ t} den_j`. The state size is
`dim φ · dim v + dim φ`, independent of `N`. Grade: `proved` (the identity); the state-size
statement is the index type of the carries, not a measurement.

**Which families.** Admits exact constant state: any kernel `e^{λ(t-s)} · Σ_j φ_j(q) ψ_j(k)`, i.e.
a finite-rank content kernel times a hyperbolic relative-position factor (the token's position
enters the carry only through `e^{-λ s}`, the query only through `e^{λ t} φ(q)`). Does not admit:
the full exponential `e^{q·k}` (infinite rank, the negative above). A general kernel is exact only
if it factorizes this way; the graded partition is `boost_finite_rank_recurrence` versus
`exp_kernel_not_finite_rank`.

## Joint witnesses

`qwenScore_witness` (pairs, positions and angles instantiated), `flow_channel_witness` (§2 domain
inhabited), `streaming_witness` (§3 scan evaluated), `exp_distinct_witness` and
`exp_value_map_injective` (§4 negative domain inhabited), `boost_witness` /
`boost_score_witness` (§4 positive family inhabited) are the checked witness terms for the claimed
domains (`AGENTS.md` §4.2).

## What is open

* `ℝ` only. The refinement to the Q32.32 engine words (fixed-point score, weight, carry) is not
  stated here; the `ZMod (2^64)` caveat blocks the channel split at the word ring and is outside
  this module.
* No cost claim: the constant state size is combinatorics, not a measurement (`AGENTS.md` §10).
* No runtime theorem: nothing reaches `rustc`/LLVM, the extracted kernels or hardware rounding.

## Trust boundary

Kernel-checked over `ℝ` for the analytic identities and over `Fin d → ℝ` for the streaming law.
The linear-independence limit argument uses `atTop` convergence of `x^n` for `0 ≤ x < 1`, not any
unbounded reduction. The elliptic/hyperbolic identification is the real shadow of
`Apeiron.Physics.Wick.flow_wick`.
-/

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing
open scoped BigOperators
open Sanguine.AttentionSpinor

noncomputable section

/-! ## 1. The RoPE-relative elliptic score -/

/-- The elliptic phase-alignment first coordinate: the pairing `rhoDot a b` equals the in-phase
component of `ringMul (-1) a (ringConj b)`. -/
theorem ellipticAlign_fst_eq_rhoDot (a b : RingCell ℝ) :
    (ringMul (-1) a (ringConj b)).1 = rhoDot a b := by
  simpa only [ellipticScore, ellipticPhaseAlign] using ellipticScore_eq_rhoDot a b

/-- One head pair rotated by its angle: left multiplication by `rho θ` at `σ = -1`, the RoPE rotor
action on the plane of that pair. -/
def ropeRot (θ : ℝ) (z : RingCell ℝ) : RingCell ℝ := rhoAct θ z

/-- The Qwen score: pair `p` is rotated by `t·θ_p` on the query side and `s·θ_p` on the key side,
and the pairs are paired through the elliptic phase alignment. -/
def qwenScore (n : ℕ) (θ : Fin n → ℝ) (t s : ℕ) (q k : Fin n → RingCell ℝ) : ℝ :=
  ∑ p, (ringMul (-1) (ropeRot ((t : ℝ) * θ p) (q p))
    (ringConj (ropeRot ((s : ℝ) * θ p) (k p)))).1

/-- The mean of the Qwen score: the pairing of the unrotated query pair with the key pair rotated
by the relative angle `d·θ_p`. This is the quantity `qwenScore` denotes. -/
def qwenRelScore (n : ℕ) (θ : Fin n → ℝ) (d : ℝ) (q k : Fin n → RingCell ℝ) : ℝ :=
  ∑ p, rhoDot (q p) (rhoAct (d * θ p) (k p))

/-- The Qwen score as the summed product of the elliptic phase-alignments, with the alignment
first coordinate read as the pairing `rhoDot`. -/
theorem qwenScore_eq_sum_rhoDot (n : ℕ) (θ : Fin n → ℝ) (t s : ℕ)
    (q k : Fin n → RingCell ℝ) :
    qwenScore n θ t s q k
      = ∑ p, rhoDot (ropeRot ((t : ℝ) * θ p) (q p)) (ropeRot ((s : ℝ) * θ p) (k p)) := by
  rw [qwenScore]
  refine Finset.sum_congr rfl (fun p _ => ?_)
  exact ellipticAlign_fst_eq_rhoDot _ _

/-- The Qwen score is the relative-angle form: the pair `p` contributes the unrotated pairing
against the key pair rotated by `(s - t)·θ_p`. -/
theorem qwenScore_eq_rel (n : ℕ) (θ : Fin n → ℝ) (t s : ℕ) (q k : Fin n → RingCell ℝ) :
    qwenScore n θ t s q k = qwenRelScore n θ ((s : ℝ) - (t : ℝ)) q k := by
  rw [qwenScore, qwenRelScore]
  refine Finset.sum_congr rfl (fun p _ => ?_)
  rw [ellipticAlign_fst_eq_rhoDot]
  simp only [ropeRot]
  rw [rho_relative_score]

/-- **Relative position invariance (headline).** The Qwen score depends on the positions `t, s`
only through `t - s`: rotations compose, so equal position differences give equal scores. -/
theorem qwenScore_relative (n : ℕ) (θ : Fin n → ℝ) (t s t' s' : ℕ)
    (q k : Fin n → RingCell ℝ)
    (h : (s : ℝ) - (t : ℝ) = (s' : ℝ) - (t' : ℝ)) :
    qwenScore n θ t s q k = qwenScore n θ t' s' q k := by
  rw [qwenScore_eq_rel, qwenScore_eq_rel, h]

/-- **Joint witness for the Qwen domain.** One pair, zero angle, identity pairs at positions `3`
and `5`: the score is `1`. The angle, both positions and both pairs are instantiated, so the
claimed domain is inhabited. -/
theorem qwenScore_witness :
    qwenScore 1 (fun _ => (0 : ℝ)) 3 5 (fun _ => ((1 : ℝ), (0 : ℝ)))
      (fun _ => ((1 : ℝ), (0 : ℝ))) = 1 := by
  simp [qwenScore, ropeRot, rhoAct, rhoMul, rho, ringMul, ringConj]

/-! ## 2. The softmax weight is the upper channel of the hyperbolic flow -/

/-- The upper channel of the hyperbolic flow at `σ = 1` is `exp`: at the root `r = 1`,
`ringUpper 1 (cosh s, sinh s) = cosh s + sinh s`. -/
theorem ringUpper_flow_one (s : ℝ) : ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) = Real.exp s := by
  rw [Apeiron.Trichotomy.flow_one]
  simp only [ringUpper]
  rw [one_mul, Real.cosh_add_sinh]

/-- The lower channel of the hyperbolic flow at `σ = 1` is `exp (-s)`:
`ringLower 1 (cosh s, sinh s) = cosh s - sinh s`. -/
theorem ringLower_flow_one (s : ℝ) :
    ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) = Real.exp (-s) := by
  rw [Apeiron.Trichotomy.flow_one]
  simp only [ringLower]
  rw [one_mul, Real.cosh_sub_sinh]

/-- **The softmax weight is the upper channel of the hyperbolic flow (headline).** The
unnormalized weight `exp s` is `ringUpper 1` at the flow, the complementary mass `exp (-s)` is
`ringLower 1`, and their product is `1` (the flow lies on the unit hyperbola). -/
theorem softmax_weight_is_upper_channel (s : ℝ) :
    ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) = Real.exp s
      ∧ ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) = Real.exp (-s)
      ∧ ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 s)
          * ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) = 1 := by
  refine ⟨ringUpper_flow_one s, ringLower_flow_one s, ?_⟩
  rw [ringUpper_flow_one, ringLower_flow_one, ← Real.exp_add]
  norm_num

/-- The channel weight of the hyperbolic flow is the two-class softmax of the log-channels
`(s, -s)`: the ratio of the upper channel to the channel sum. -/
theorem channelWeight_flow_one (s : ℝ) :
    channelWeight (1 : ℝ) (Apeiron.Trichotomy.flow 1 s) = softmax2 s (-s) := by
  rw [channelWeight, ringUpper_flow_one, ringLower_flow_one]
  rfl

/-- **Joint witness for the flow-channel domain.** At `s = 0` both channels read `1`: the flow is
the algebra unit and the weight is `1/2`. -/
theorem flow_channel_witness :
    ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 0) = 1
      ∧ ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 0) = 1
      ∧ channelWeight (1 : ℝ) (Apeiron.Trichotomy.flow 1 0) = 1 / 2 := by
  refine ⟨?_, ?_, ?_⟩
  · rw [ringUpper_flow_one]; norm_num
  · rw [ringLower_flow_one]; norm_num
  · rw [channelWeight_flow_one]; norm_num [softmax2]

/-! ## 3. Exact streaming: online softmax over vector values -/

/-- The vector-valued online-softmax state `(m, den, num)`: running max, running rescaled drag
`den`, and running rescaled numerator `num : Fin d → ℝ`; `MLPrimitivesBridge.osFuse` with a
numerator channel per value coordinate. -/
abbrev OaStateV (d : ℕ) : Type := ℝ × ℝ × (Fin d → ℝ)

/-- One token as a tile `(sc, 1, v)`: drag `e^{sc}`, numerator `v·e^{sc}`. -/
def oaTileV {d : ℕ} (sc : ℝ) (v : Fin d → ℝ) : OaStateV d := (sc, 1, v)

/-- The vector-valued max-rescale fuse: the drag channel and each numerator coordinate ride the
same running max, exactly `oaFuse` per coordinate. -/
def oaFuseV {d : ℕ} (a b : OaStateV d) : OaStateV d :=
  (max a.1 b.1,
   a.2.1 * Real.exp (a.1 - max a.1 b.1) + b.2.1 * Real.exp (b.1 - max a.1 b.1),
   fun i => a.2.2 i * Real.exp (a.1 - max a.1 b.1) + b.2.2 i * Real.exp (b.1 - max a.1 b.1))

/-- The streaming scan: fold the `(score, value)` stream into the `O(1)` state, one `oaFuseV` per
token, from the identity `(0, 0, 0)`. -/
def oaScanV {d : ℕ} (l : List (ℝ × (Fin d → ℝ))) : OaStateV d :=
  l.foldl (fun acc p => oaFuseV acc (oaTileV p.1 p.2)) (0, 0, fun _ => 0)

/-- The drag the chart denotes: `(m, den, num) ↦ den·e^m = Σ e^{sc}`; the third component is
irrelevant, so it reads the scalar `oaDen`. -/
def oaDenV {d : ℕ} (a : OaStateV d) : ℝ := a.2.1 * Real.exp a.1

/-- The numerator coordinate the chart denotes: `(m, den, num) ↦ numᵢ·e^m = Σ e^{sc}·vᵢ`. -/
def oaNumV {d : ℕ} (a : OaStateV d) (i : Fin d) : ℝ := a.2.2 i * Real.exp a.1

/-- The vector drag is the scalar `oaDen` at the same chart. -/
theorem oaDenV_eq_oaDen {d : ℕ} (a : OaStateV d) :
    oaDenV a = oaDen (a.1, a.2.1, (0 : ℝ)) := rfl

/-- The vector numerator coordinate is the scalar `oaNum` at the same chart and coordinate. -/
theorem oaNumV_eq_oaNum {d : ℕ} (a : OaStateV d) (i : Fin d) :
    oaNumV a i = oaNum (a.1, a.2.1, a.2.2 i) := rfl

/-- **The drag channel merges as exact `+`** (vector lift of `oaDen_fuse`). -/
theorem oaDenV_fuse {d : ℕ} (a b : OaStateV d) :
    oaDenV (oaFuseV a b) = oaDenV a + oaDenV b := by
  have h := oaDen_fuse (a.1, a.2.1, (0 : ℝ)) (b.1, b.2.1, (0 : ℝ))
  simpa only [oaDenV, oaDen, oaFuseV, oaFuse] using h

/-- **The numerator coordinate merges as exact `+`** (vector lift of `oaNum_fuse`). -/
theorem oaNumV_fuse {d : ℕ} (a b : OaStateV d) (i : Fin d) :
    oaNumV (oaFuseV a b) i = oaNumV a i + oaNumV b i := by
  have h := oaNum_fuse (a.1, a.2.1, a.2.2 i) (b.1, b.2.1, b.2.2 i)
  simpa only [oaNumV, oaNum, oaFuseV, oaFuse] using h

/-- The drag of the scan step accumulates `e^{sc}` exactly, from any starting state. -/
theorem oaDenV_scan_acc {d : ℕ} (l : List (ℝ × (Fin d → ℝ))) (acc : OaStateV d) :
    oaDenV (l.foldl (fun a p => oaFuseV a (oaTileV p.1 p.2)) acc)
      = oaDenV acc + (l.map (fun p => Real.exp p.1)).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih, oaDenV_fuse]
    simp only [oaDenV, oaTileV, one_mul]
    ring

/-- The numerator coordinate of the scan step accumulates `e^{sc}·vᵢ` exactly. -/
theorem oaNumV_scan_acc {d : ℕ} (l : List (ℝ × (Fin d → ℝ))) (acc : OaStateV d) (i : Fin d) :
    oaNumV (l.foldl (fun a p => oaFuseV a (oaTileV p.1 p.2)) acc) i
      = oaNumV acc i + (l.map (fun p => Real.exp p.1 * p.2 i)).sum := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    simp only [List.foldl_cons, List.map_cons, List.sum_cons, ih, oaNumV_fuse]
    simp only [oaNumV, oaTileV, one_mul]
    ring

/-- **Softmax attention is the online-softmax streaming recurrence, vector-valued (headline).**
Streaming the `(score, value)` tokens through the running `(m, den, num)` state — one `oaFuseV`
per token, three fields of state regardless of length — and reading `num / den` gives exactly
`(Σ e^{sc}·v) / (Σ e^{sc})`. The two fuse laws are the vector lifts of `oaDen_fuse` and
`oaNum_fuse`, so a chunked pass fuses losslessly at the denotation level. -/
theorem softmax_attention_is_online_spinor_vec {d : ℕ} (l : List (ℝ × (Fin d → ℝ))) :
    (∀ i : Fin d, oaNumV (oaScanV l) i = (l.map (fun p => Real.exp p.1 * p.2 i)).sum)
      ∧ oaDenV (oaScanV l) = (l.map (fun p => Real.exp p.1)).sum
      ∧ (∀ a b : OaStateV d, oaDenV (oaFuseV a b) = oaDenV a + oaDenV b)
      ∧ (∀ (a b : OaStateV d) (i : Fin d),
          oaNumV (oaFuseV a b) i = oaNumV a i + oaNumV b i) := by
  refine ⟨?_, ?_, oaDenV_fuse, oaNumV_fuse⟩
  · intro i
    rw [oaScanV, oaNumV_scan_acc]
    simp [oaNumV]
  · rw [oaScanV, oaDenV_scan_acc]
    simp [oaDenV]

/-- The normalized online readout is the softmax attention denominator over the exp numerator,
exactly. -/
theorem softmax_attention_is_online_readout {d : ℕ} (l : List (ℝ × (Fin d → ℝ))) (i : Fin d) :
    oaNumV (oaScanV l) i / oaDenV (oaScanV l)
      = (l.map (fun p => Real.exp p.1 * p.2 i)).sum / (l.map (fun p => Real.exp p.1)).sum := by
  rw [(softmax_attention_is_online_spinor_vec l).1 i,
    (softmax_attention_is_online_spinor_vec l).2.1]

/-- **Chunk fusion, drag channel.** The denotation of a concatenated scan is the sum of the two
scans' denotations: a chunked traversal fuses losslessly (`oaDenV_fuse`). -/
theorem oaDenV_scan_append {d : ℕ} (l₁ l₂ : List (ℝ × (Fin d → ℝ))) :
    oaDenV (oaScanV (l₁ ++ l₂)) = oaDenV (oaScanV l₁) + oaDenV (oaScanV l₂) := by
  have h₁ : oaScanV (l₁ ++ l₂)
      = l₂.foldl (fun a p => oaFuseV a (oaTileV p.1 p.2)) (oaScanV l₁) := by
    rw [oaScanV, List.foldl_append]
    rfl
  have h₂ : oaScanV l₂ = l₂.foldl (fun a p => oaFuseV a (oaTileV p.1 p.2)) (0, 0, fun _ => 0) := rfl
  rw [h₁, oaDenV_scan_acc, h₂, oaDenV_scan_acc]
  simp [oaDenV]

/-- **Chunk fusion, numerator channel.** Same for every numerator coordinate. -/
theorem oaNumV_scan_append {d : ℕ} (l₁ l₂ : List (ℝ × (Fin d → ℝ))) (i : Fin d) :
    oaNumV (oaScanV (l₁ ++ l₂)) i = oaNumV (oaScanV l₁) i + oaNumV (oaScanV l₂) i := by
  have h₁ : oaScanV (l₁ ++ l₂)
      = l₂.foldl (fun a p => oaFuseV a (oaTileV p.1 p.2)) (oaScanV l₁) := by
    rw [oaScanV, List.foldl_append]
    rfl
  have h₂ : oaScanV l₂ = l₂.foldl (fun a p => oaFuseV a (oaTileV p.1 p.2)) (0, 0, fun _ => 0) := rfl
  rw [h₁, oaNumV_scan_acc, h₂, oaNumV_scan_acc]
  simp [oaNumV]

/-- **Joint witness for the streaming domain.** One token, score `0`, value `1`: the drag is `1`
and the numerator is `1`. -/
theorem streaming_witness :
    oaDenV (oaScanV ([(0, fun _ => (1 : ℝ))] : List (ℝ × (Fin 1 → ℝ)))) = 1
      ∧ oaNumV (oaScanV ([(0, fun _ => (1 : ℝ))] : List (ℝ × (Fin 1 → ℝ)))) 0 = 1 := by
  refine ⟨?_, ?_⟩
  · rw [(softmax_attention_is_online_spinor_vec _).2.1]; simp
  · rw [(softmax_attention_is_online_spinor_vec _).1 0]; simp

/-! ## 4a. Negative: distinct exponentials are independent, the exponential kernel has no finite rank -/

/-- **Distinct exponentials are linearly independent (headline).** If a finite combination of
`q ↦ e^{kᵢ q}` vanishes identically with the keys `kᵢ` pairwise distinct, every coefficient is
zero. The proof multiplies by `e^{-k_j n}` for the support-maximal key `j` and sends `n → ∞`:
every off-maximal term is a power `x^n` with `0 ≤ x < 1`. -/
theorem exp_linearIndependent {N : ℕ} (k : Fin N → ℝ) (hk : Function.Injective k)
    (c : Fin N → ℝ) (h : ∀ q : ℝ, ∑ i, c i * Real.exp (k i * q) = 0) :
    ∀ i, c i = 0 := by
  classical
  by_contra hc
  have hcne : ∃ i, c i ≠ 0 := not_forall.mp hc
  let S : Finset (Fin N) := Finset.univ.filter (fun i => c i ≠ 0)
  have hSne : S.Nonempty := by
    obtain ⟨i, hi⟩ := hcne
    exact ⟨i, by simp [S, hi]⟩
  obtain ⟨j, hjS, hjmax⟩ := Finset.exists_max_image S k hSne
  have hcj : c j ≠ 0 := by simpa [S] using hjS
  have hklt : ∀ i ∈ S, i ≠ j → k i < k j := by
    intro i hi hij
    exact lt_of_le_of_ne (hjmax i hi) (fun hkij => hij (hk hkij))
  have hzero : ∀ n : ℕ, ∑ i, c i * Real.exp (k i - k j) ^ n = 0 := by
    intro n
    have hq := h ((n : ℝ))
    calc ∑ i, c i * Real.exp (k i - k j) ^ n
        = ∑ i, c i * (Real.exp (k i * (n : ℝ)) * Real.exp (-(k j * (n : ℝ)))) := by
          refine Finset.sum_congr rfl (fun i _ => ?_)
          rw [← Real.exp_nat_mul, ← Real.exp_add]
          congr 1
          ring_nf
      _ = (∑ i, c i * Real.exp (k i * (n : ℝ))) * Real.exp (-(k j * (n : ℝ))) := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl (fun i _ => by ring)
      _ = 0 := by rw [hq, zero_mul]
  have hterm : ∀ i : Fin N,
      Filter.Tendsto (fun n : ℕ => c i * Real.exp (k i - k j) ^ n) Filter.atTop
        (nhds (if i = j then c i else 0)) := by
    intro i
    by_cases hij : i = j
    · subst hij
      have hrj : Real.exp (k i - k i) = 1 := by simp
      simp only [hrj, one_pow, mul_one]
      simp
    · by_cases hci : c i = 0
      · simp [hci, hij]
      · have hiS : i ∈ S := by simp [S, hci]
        have hlt : Real.exp (k i - k j) < 1 :=
          Real.exp_lt_one_iff.mpr (sub_neg.mpr (hklt i hiS hij))
        have h0 : 0 ≤ Real.exp (k i - k j) := (Real.exp_pos _).le
        have ht : Filter.Tendsto (fun n : ℕ => Real.exp (k i - k j) ^ n) Filter.atTop
            (nhds 0) := tendsto_pow_atTop_nhds_zero_of_lt_one h0 hlt
        have := ht.const_mul (c i)
        simpa [hij] using this
  have hsum := tendsto_finsetSum Finset.univ (fun i _ => hterm i)
  have hval : (∑ i : Fin N, if i = j then c i else 0) = c j := by
    simp
  rw [hval] at hsum
  have hconst : Filter.Tendsto (fun n : ℕ => ∑ i, c i * Real.exp (k i - k j) ^ n)
      Filter.atTop (nhds 0) := by
    rw [show (fun n : ℕ => ∑ i, c i * Real.exp (k i - k j) ^ n) = fun _ => 0 from
      funext (fun n => hzero n)]
    exact tendsto_const_nhds
  exact hcj (tendsto_nhds_unique hconst hsum).symm

/-- **The exponential kernel has no finite-rank factorization (headline).** For every dimension
`D` there is no pair of feature maps `φ, ψ : ℝ → Fin D → ℝ` with `e^{q k} = Σ_j φ_j(q) ψ_j(k)` for
all real `q, k`. Taking `D + 1` distinct keys yields `D + 1` linearly independent functions inside
a `D`-dimensional span. -/
theorem exp_kernel_not_finite_rank (D : ℕ) (φ ψ : ℝ → Fin D → ℝ) :
    ¬ (∀ q k : ℝ, Real.exp (q * k) = ∑ j, φ q j * ψ k j) := by
  intro hf
  have hkey : Function.Injective (fun i : Fin (D + 1) => (i : ℝ)) := by
    intro i i' hii
    have hii' : ((i : ℕ) : ℝ) = ((i' : ℕ) : ℝ) := by simpa using hii
    exact Fin.ext (by exact_mod_cast hii')
  have hnot : ¬ LinearIndependent ℝ (fun i : Fin (D + 1) => (fun i : Fin (D + 1) => ψ (i : ℝ)) i) := by
    intro hli
    have hcard := hli.fintype_card_le_finrank (R := ℝ) (M := Fin D → ℝ)
    rw [Fintype.card_fin, Module.finrank_fintype_fun_eq_card, Fintype.card_fin] at hcard
    omega
  obtain ⟨g, hg, i0, hgi0⟩ := Fintype.not_linearIndependent_iff.mp hnot
  have hpoint : ∀ j : Fin D, ∑ i, g i * ψ (i : ℝ) j = 0 := by
    intro j
    have := congrFun hg j
    simpa using this
  have hzero : ∀ q : ℝ, ∑ i, g i * Real.exp ((i : ℝ) * q) = 0 := by
    intro q
    calc ∑ i, g i * Real.exp ((i : ℝ) * q)
        = ∑ i, g i * (∑ j, φ q j * ψ (i : ℝ) j) := by
          refine Finset.sum_congr rfl (fun i _ => ?_)
          rw [mul_comm ((i : ℝ)) q, hf q (i : ℝ)]
      _ = ∑ j, φ q j * (∑ i, g i * ψ (i : ℝ) j) := by
          simp_rw [Finset.mul_sum]
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl (fun j _ => ?_)
          refine Finset.sum_congr rfl (fun i _ => by ring)
      _ = 0 := by simp [hpoint]
  exact hgi0 (exp_linearIndependent (fun i : Fin (D + 1) => (i : ℝ)) hkey g hzero i0)

/-- **The value map is injective (headline).** For fixed pairwise-distinct keys, the values are
determined by the query function `q ↦ Σᵢ vᵢ e^{kᵢ q}`: distinct exponentials are independent.
This is the injectivity that forbids a fixed finite-dimensional state from reproducing the exact
softmax attention for all future queries. -/
theorem exp_value_map_injective {N : ℕ} (k : Fin N → ℝ) (hk : Function.Injective k) :
    Function.Injective
      (fun v : Fin N → ℝ => fun q : ℝ => ∑ i, v i * Real.exp (k i * q)) := by
  intro u v huv
  funext i
  have hzero : ∀ q : ℝ, ∑ i, (u i - v i) * Real.exp (k i * q) = 0 := by
    intro q
    have hq := congrFun huv q
    change (∑ i, u i * Real.exp (k i * q)) = (∑ i, v i * Real.exp (k i * q)) at hq
    calc ∑ i, (u i - v i) * Real.exp (k i * q)
        = (∑ i, u i * Real.exp (k i * q)) - (∑ i, v i * Real.exp (k i * q)) := by
          rw [← Finset.sum_sub_distrib]
          refine Finset.sum_congr rfl (fun i _ => by ring)
      _ = 0 := by rw [hq, sub_self]
  exact sub_eq_zero.mp (exp_linearIndependent k hk (fun i => u i - v i) hzero i)

/-- **Joint witness for the negative domain.** Two distinct keys give distinct exponential values,
so the injectivity hypothesis is inhabited on a nontrivial instance. -/
theorem exp_distinct_witness :
    Real.exp ((0 : ℝ) * 1) ≠ Real.exp ((1 : ℝ) * 1) := by
  rw [ne_eq, Real.exp_injective.eq_iff]
  norm_num

/-! ## 4b. Positive: finite-rank content times the hyperbolic boost is exact with constant state -/

/-- The relative-position boost `e^{λ(t - s)}`, the hyperbolic analogue of RoPE's relative
rotation. -/
def boost (lam : ℝ) (t s : ℝ) : ℝ := Real.exp (lam * (t - s))

/-- **The boost factors through the hyperbolic flow (headline).** `e^{λ(t-s)}` is the upper
channel of the flow at `λt` times the lower channel at `λs`: the relative-position factor is the
product of a query factor and a key factor, exactly the factorization a constant state needs. -/
theorem boost_factor (lam t s : ℝ) :
    boost lam t s = ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 (lam * t))
      * ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 (lam * s)) := by
  rw [boost, ringUpper_flow_one, ringLower_flow_one, ← Real.exp_add]
  congr 1
  ring

/-- The relative-position finite-rank score: a finite-rank content pairing times the hyperbolic
boost. -/
def boostScore {n dφ : ℕ} (lam : ℝ) (φ ψ : (Fin n → ℝ) → Fin dφ → ℝ)
    (t s : ℝ) (q k : Fin n → ℝ) : ℝ :=
  (∑ j, φ q j * ψ k j) * boost lam t s

/-- The query feature at position `t`: the content query features times the upper-channel factor
`e^{λt}`. -/
def boostQ {n dφ : ℕ} (lam t : ℝ) (φ : (Fin n → ℝ) → Fin dφ → ℝ) (q : Fin n → ℝ) : Fin dφ → ℝ :=
  fun j => φ q j * Real.exp (lam * t)

/-- The key feature at position `s`: the content key features times the lower-channel factor
`e^{-λs}` — the only place the token's position enters the carry. -/
def boostK {N n dφ : ℕ} (lam : ℝ) (ψ : (Fin n → ℝ) → Fin dφ → ℝ)
    (pos : Fin N → ℝ) (ks : Fin N → (Fin n → ℝ)) : Fin N → Fin dφ → ℝ :=
  fun s j => ψ (ks s) j * Real.exp (-(lam * pos s))

/-- The boosted score is the kernel pairing of the boosted features. -/
theorem boostScore_eq_dotp {N n dφ : ℕ} (lam : ℝ) (φ ψ : (Fin n → ℝ) → Fin dφ → ℝ)
    (pos : Fin N → ℝ) (ks : Fin N → (Fin n → ℝ)) (t : ℝ) (q : Fin n → ℝ) (s : Fin N) :
    boostScore lam φ ψ t (pos s) q (ks s)
      = dotp (boostQ lam t φ q) (boostK lam ψ pos ks s) := by
  simp only [boostScore, boostQ, boostK, dotp, boost]
  rw [Finset.sum_mul]
  refine Finset.sum_congr rfl (fun j _ => ?_)
  have hexp : Real.exp (lam * t) * Real.exp (-(lam * pos s))
      = Real.exp (lam * (t - pos s)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc (φ q j * ψ (ks s) j) * Real.exp (lam * (t - pos s))
      = (φ q j * ψ (ks s) j) * (Real.exp (lam * t) * Real.exp (-(lam * pos s))) := by
        rw [hexp]
    _ = (φ q j * Real.exp (lam * t)) * (ψ (ks s) j * Real.exp (-(lam * pos s))) := by
        ring

/-- **Exact constant-state streaming for the boosted finite-rank kernel (headline).** Normalized
attention over the score `Σ_j φ_j(q) ψ_j(k) · e^{λ(t - s)}` equals the fixed-size carry read:
`N` tokens accumulate `num_s = ψ(k_s) e^{-λs} v_s` (`Fin dφ → ℝ`) and `den_s = ψ(k_s) e^{-λs}`
(`Fin dφ`), and the position-`t` query reads `Σ_j φ_j(q) e^{λt} num_j / Σ_j φ_j(q) e^{λt} den_j`.
The identity is `Sanguine.AttentionSpinor.kernel_attention_is_recurrence` instantiated at the
boosted features; the state size is `dφ` (scalar values) independent of `N`. -/
theorem boost_finite_rank_recurrence {N n dφ : ℕ}
    (lam : ℝ) (φ ψ : (Fin n → ℝ) → Fin dφ → ℝ) (pos : Fin N → ℝ)
    (ks : Fin N → (Fin n → ℝ)) (vals : Fin N → ℝ) (t : ℝ) (q : Fin n → ℝ) :
    dotp (boostQ lam t φ q) (fwNum (boostK lam ψ pos ks) vals)
        / dotp (boostQ lam t φ q) (fwDen (boostK lam ψ pos ks))
      = (∑ s, boostScore lam φ ψ t (pos s) q (ks s) * vals s)
          / (∑ s, boostScore lam φ ψ t (pos s) q (ks s)) := by
  rw [kernel_attention_is_recurrence (phiQ := boostQ lam t φ q)
    (phiK := boostK lam ψ pos ks) (val := vals)]
  refine congrArg₂ (· / ·) ?_ ?_
  · refine Finset.sum_congr rfl (fun s _ => ?_)
    rw [boostScore_eq_dotp]
  · refine Finset.sum_congr rfl (fun s _ => ?_)
    rw [boostScore_eq_dotp]

/-- **The vector-valued carry read.** With values `Fin dv → ℝ` the boost carry is
`Fin dφ → Fin dv → ℝ` and the normalizer carry `Fin dφ → ℝ`; each output coordinate is the scalar
recurrence at that coordinate, so the read is exact coordinatewise. State size `dφ·dv + dφ`,
independent of the token count `N`. -/
theorem boost_finite_rank_recurrence_vec {N n dφ dv : ℕ}
    (lam : ℝ) (φ ψ : (Fin n → ℝ) → Fin dφ → ℝ) (pos : Fin N → ℝ)
    (ks : Fin N → (Fin n → ℝ)) (vals : Fin N → Fin dv → ℝ) (t : ℝ) (q : Fin n → ℝ)
    (i : Fin dv) :
    (∑ j, boostQ lam t φ q j * (∑ s, vals s i * boostK lam ψ pos ks s j))
        / (∑ j, boostQ lam t φ q j * (∑ s, boostK lam ψ pos ks s j))
      = (∑ s, boostScore lam φ ψ t (pos s) q (ks s) * vals s i)
          / (∑ s, boostScore lam φ ψ t (pos s) q (ks s)) := by
  have h := boost_finite_rank_recurrence lam φ ψ pos ks (fun s => vals s i) t q
  simpa only [dotp, fwNum, fwDen] using h

/-- **Joint witness for the boost domain.** At `λ = 2`, `t = 3`, `s = 1` the boost is `e^4`, and
it equals the flow-channel product. -/
theorem boost_witness :
    boost 2 3 1 = Real.exp 4
      ∧ ringUpper (1 : ℝ) (Apeiron.Trichotomy.flow 1 (2 * 3))
          * ringLower (1 : ℝ) (Apeiron.Trichotomy.flow 1 (2 * 1)) = Real.exp 4 := by
  refine ⟨?_, ?_⟩
  · norm_num [boost]
  · rw [ringUpper_flow_one, ringLower_flow_one, ← Real.exp_add]
    norm_num

/-- **Joint witness for the boosted-score domain.** Constant content `1` at zero position and
query: the boosted score is `1`, so the kernel family is inhabited on a nonzero instance. -/
theorem boost_score_witness :
    boostScore (n := 1) (dφ := 1) 1 (fun _ _ => (1 : ℝ)) (fun _ _ => (1 : ℝ)) 0 0
      (fun (_ : Fin 1) => (0 : ℝ)) (fun (_ : Fin 1) => (0 : ℝ)) = 1 := by
  simp [boostScore, boost]

end

end Nous.Model
