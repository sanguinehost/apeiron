import Nous.Model.Decomposition
import Mathlib.Algebra.Field.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Channel-form attention: one streaming scan, normalized once

The ring-generic reading of the landed real-valued attention scans
(`Substrate/Algebra/Neural/AttentionSpinor.lean`: `att`, `softmax_attention_is_online_spinor`).
A stream of key-value tokens `(k_i, v_i)` is read against a query `q` with the channel weights
of the query-key products,

    w_i = channelWeight r (ringMul σ q k_i) = ringUpper r (ringMul σ q k_i) / (2 * (ringMul σ q k_i).1),

the factorizable kernel of the hyperbolic split: `ringUpper r (ringMul σ q k_i)
= ringUpper r q * ringUpper r k_i` (`ringUpper_mul`), so each score's numerator factors into a
query channel and a key channel. The streaming form keeps two running numbers — the normalizer
`Σ_i w_i` and the weighted sum `Σ_i w_i v_i` — and divides once at the end.

## Domain: `ChannelFormDomain`

* `r ^ 2 = σ` — a root witness. The elliptic face (`σ < 0` over the reals) has none, so the
  channel weight of a query-key product is not a channel reading there.
* `(2 : R) ≠ 0` and `r ≠ 0` — the split premises of `hyperbolic_decomposition_field`. The
  parabolic face has `r = 0`, where `ringUpper` and `ringLower` coincide and the channel weight
  degenerates to `1/2`; the elliptic and parabolic faces do not split.
* `∀ p ∈ l, 2 * (ringMul σ q p.1).1 ≠ 0` — each score's channel denominator is nonzero, exactly
  where `channelWeight` factors through `ringUpper` (`channelWeight_eq`).
* `channelDen σ r q l ≠ 0` — the readout is a genuine division.

Every premise is consumed by `attention_channel_form`: the root witness feeds `ringUpper_mul`
and `hyperbolic_decomposition_field`, the split premises feed the bijectivity clause, the
per-token denominators feed `channelWeight_eq`, and the nonzero normalizer feeds the
"normalized once" clause. `channel_form_domain_witness` exhibits a joint witness over the
rationals, so the domain is inhabited; `channelDen_nil` is the refusal side, where the
normalizer vanishes and no division is meaningful.

## The other two faces

* **Parabolic (`σ = 0`, `r = 0`): the channel form degenerates to the uniform average.** The
  root witness `0 ^ 2 = 0` holds, but the split premise `r ≠ 0` fails, so `ChannelFormDomain`
  is empty on this face (`channelFormDomain_parabolic_empty`) and the hyperbolic headline does
  not apply. The channel reading degenerates instead: `ringUpper 0 z = ringLower 0 z = z.1`
  (`ringUpper_parabolic`, `ringLower_parabolic`), so `channelWeight 0 z` is uniformly `1 / 2`
  wherever the channel denominator is nonzero (`channelWeight_parabolic`), every score is `1 / 2`
  (`channelScore_parabolic`), the normalizer is `l.length * (1 / 2)` (`channelDen_parabolic`),
  and the readout is the uniform average `(Σ_i v_i) / l.length` of the values
  (`attention_channel_form_parabolic`). The exact domain is `ParabolicChannelFormDomain`:
  `2 ≠ 0`, each score's channel denominator nonzero, and a nonzero normalizer — which on this
  face is exactly `(l.length : R) ≠ 0` (`channelDen_parabolic_ne_zero_iff`).
* **Elliptic (`σ < 0` over `ℝ`): excluded population, with its reason.** No root witness
  `r ^ 2 = σ` exists at `σ < 0` over the reals: a square is nonnegative (`sq_nonneg`), so
  `r ^ 2 = σ < 0` is impossible (`elliptic_no_root`, `elliptic_no_root_witness`). With no root
  there is no channel reading `ringUpper r`/`ringLower r` at a root, the factorizable kernel
  `channelWeight_factor` cannot be formed, and `ChannelFormDomain` has no inhabitant
  (`channelFormDomain_elliptic_empty`). The population is excluded, not approximated: no split
  is asserted at a non-root, and no channel-form claim in this module covers it.

## Trust boundary

No cost claim: the constant streaming state size is combinatorics, not a measurement
(`AGENTS.md` §10). No float runtime theorem: `R` is any field; the rationals witness the
domain, the reals are the reference face. Nothing here reaches `rustc`/LLVM, the extracted
kernels, or the word ring at `Z/2^64`, whose 2-adic caveat blocks the channel split.
-/

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

variable {R : Type*} [Field R]

/-- One streaming step of channel-form attention: add the token's weight to the running
normalizer, and the weight times the value to the running weighted sum.

    ×   [paired with, so R × R means two elements of R side by side] -/
def channelAttnStep (acc p : R × R) : R × R :=
  (acc.1 + p.1, acc.2 + p.1 * p.2)

/-- The streaming scan: fold the `(weight, value)` stream into the running
`(normalizer, weighted sum)` state from the zero state, one `channelAttnStep` per token.

    List   [a finite sequence of things, in order]
    ×      [paired with, so R × R means two elements of R side by side] -/
def channelAttnScan (l : List (R × R)) : R × R :=
  l.foldl channelAttnStep (0, 0)

/-- Normalize once at the end: the running weighted sum divided by the running normalizer.

    ×   [paired with, so R × R means two elements of R side by side] -/
def channelAttnOut (st : R × R) : R := st.2 / st.1

/-- Dividing a sum by a common denominator distributes over the summands.

    List   [a finite sequence of things, in order] -/
theorem list_sum_div_map (l : List R) (d : R) :
    l.sum / d = (l.map (fun x => x / d)).sum := by
  induction l with
  | nil => simp
  | cons x xs ih => rw [List.sum_cons, List.map_cons, List.sum_cons, add_div, ih]

/-- The scan accumulates the running normalizer and the running weighted sum from any
starting state.

    List   [a finite sequence of things, in order]
    ×      [paired with, so R × R means two elements of R side by side] -/
theorem channelAttnScan_acc (l : List (R × R)) (acc : R × R) :
    l.foldl channelAttnStep acc =
      (acc.1 + (l.map (fun p => p.1)).sum, acc.2 + (l.map (fun p => p.1 * p.2)).sum) := by
  induction l generalizing acc with
  | nil => simp
  | cons p ps ih =>
    rw [List.foldl_cons, ih, List.map_cons, List.map_cons, List.sum_cons, List.sum_cons]
    refine Prod.ext ?_ ?_
    · show (acc.1 + p.1) + (ps.map (fun q => q.1)).sum
        = acc.1 + (p.1 + (ps.map (fun q => q.1)).sum)
      ring
    · show (acc.2 + p.1 * p.2) + (ps.map (fun q => q.1 * q.2)).sum
        = acc.2 + (p.1 * p.2 + (ps.map (fun q => q.1 * q.2)).sum)
      ring

/-- From the zero state the streaming scan is exactly the pair of sums: the normalizer
`Σ_i w_i` and the weighted sum `Σ_i w_i v_i`.

    List   [a finite sequence of things, in order]
    ×      [paired with, so R × R means two elements of R side by side] -/
theorem channelAttnScan_eq (l : List (R × R)) :
    channelAttnScan l = ((l.map (fun p => p.1)).sum, (l.map (fun p => p.1 * p.2)).sum) := by
  show l.foldl channelAttnStep (0, 0) = _
  rw [channelAttnScan_acc]
  refine Prod.ext ?_ ?_
  · show (0 : R) + (l.map (fun p => p.1)).sum = (l.map (fun p => p.1)).sum
    ring
  · show (0 : R) + (l.map (fun p => p.1 * p.2)).sum = (l.map (fun p => p.1 * p.2)).sum
    ring

/-- Normalizing the streaming state once at the end equals the per-token normalized weighted
sum `Σ_i (w_i / Σ_j w_j) v_i` for generic weights `w_i`, values `v_i`. No hypothesis: in the
field operations both sides also agree when the normalizer vanishes.

    List   [a finite sequence of things, in order]
    ×      [paired with, so R × R means two elements of R side by side] -/
theorem channelAttnOut_normal (l : List (R × R)) :
    channelAttnOut (channelAttnScan l)
      = ((l.map (fun p => (p.1 / (l.map (fun q => q.1)).sum) * p.2)).sum) := by
  rw [channelAttnScan_eq]
  show (l.map (fun p => p.1 * p.2)).sum / (l.map (fun p => p.1)).sum
      = ((l.map (fun p => (p.1 / (l.map (fun q => q.1)).sum) * p.2)).sum)
  rw [list_sum_div_map, List.map_map]
  refine congrArg List.sum (List.map_congr_left fun p _ => ?_)
  show p.1 * p.2 / (l.map (fun q => q.1)).sum = (p.1 / (l.map (fun q => q.1)).sum) * p.2
  exact (div_mul_eq_mul_div p.1 _ p.2).symm

/-- The channel score of a query-key product: the channel weight of `ringMul σ q k`, the
factorizable-kernel reading of the weight.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x] -/
def channelScore (σ r : R) (q k : RingCell R) : R :=
  channelWeight r (ringMul σ q k)

/-- The normalizer of a channel-form attention stream: the sum of the channel scores.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
def channelDen (σ r : R) (q : RingCell R) (l : List (RingCell R × R)) : R :=
  (l.map (fun p => channelScore σ r q p.1)).sum

/-- The weighted sum of a channel-form attention stream: the sum of the score times the value.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
def channelNum (σ r : R) (q : RingCell R) (l : List (RingCell R × R)) : R :=
  (l.map (fun p => channelScore σ r q p.1 * p.2)).sum

/-- The channel-form stream of a query against key-value tokens: each token `(k, v)`
contributes its channel score and its value.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
def channelStream (σ r : R) (q : RingCell R) (l : List (RingCell R × R)) : List (R × R) :=
  l.map (fun p => (channelScore σ r q p.1, p.2))

/-- The factorizable domain of channel-form attention: a root witness for the split, the
split premises `2 ≠ 0` and `r ≠ 0`, every score's channel denominator nonzero, and a nonzero
normalizer.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
def ChannelFormDomain (σ r : R) (q : RingCell R) (l : List (RingCell R × R)) : Prop :=
  r ^ 2 = σ ∧ (2:R) ≠ 0 ∧ r ≠ 0 ∧ (∀ p ∈ l, 2 * (ringMul σ q p.1).1 ≠ 0) ∧
    channelDen σ r q l ≠ 0

/-- The factorizable kernel: where the score's channel denominator is nonzero, the channel
weight of the query-key product is the factorized ratio of query channel times key channel
over the denominator. Root witness `r ^ 2 = σ`; the elliptic face admits no root witness and
the parabolic face has `r = 0`.

    σ        [the parameter: what the adjoined direction squares to]
    ≠        [is not equal to]
    ringMul  [multiply two points together, the σ way]
    ringUpper [how far a point lies along the first direction of size zero]
    RingCell [a point: two ring elements written side by side, a + b·x] -/
theorem channelWeight_factor {σ r : R} (hr : r ^ 2 = σ) (q k : RingCell R)
    (hk : 2 * (ringMul σ q k).1 ≠ 0) :
    channelWeight r (ringMul σ q k)
      = (ringUpper r q * ringUpper r k) / (2 * (ringMul σ q k).1) := by
  rw [channelWeight_eq (ringMul σ q k) hk, ringUpper_mul hr q k]

/-- The streaming state of the channel stream is the pair `(channelDen σ r q l,
channelNum σ r q l)`: the running normalizer `Σ_i w_i` and the running weighted sum
`Σ_i w_i v_i` in closed form.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelAttnScan_channelStream (σ r : R) (q : RingCell R) (l : List (RingCell R × R)) :
    channelAttnScan (channelStream σ r q l) = (channelDen σ r q l, channelNum σ r q l) := by
  rw [channelAttnScan_eq]
  simp only [channelStream, List.map_map, channelDen, channelNum]
  rfl

/-- The channel stream, normalized once at the end, is the per-token normalized weighted sum
`Σ_i (w_i / Σ_j w_j) v_i` with `w_i = channelScore σ r q k_i`.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelAttnOut_channelStream (σ r : R) (q : RingCell R) (l : List (RingCell R × R)) :
    channelAttnOut (channelAttnScan (channelStream σ r q l))
      = ((l.map (fun p => (channelScore σ r q p.1 / channelDen σ r q l) * p.2)).sum) := by
  rw [channelAttnOut_normal]
  simp only [channelStream, List.map_map, channelDen]
  rfl

/-- **Channel-form attention (headline).** On the factorizable domain `ChannelFormDomain
σ r q l` the streaming scan and its single normalization give exactly the normalized weighted
sum over the channel weights. Five clauses:

1. the streaming state is the running normalizer `Σ_i w_i` and weighted sum `Σ_i w_i v_i`;
2. normalizing once at the end gives `Σ_i (w_i / Σ_j w_j) v_i`, `w_i` the channel weights
   `channelWeight r (ringMul σ q k_i)`;
3. the readout times the normalizer is the weighted sum — the normalization is exact;
4. the normal form is the factorized-score form `Σ_i ((ringUpper r q * ringUpper r k_i) /
   (2 * (ringMul σ q k_i).1) / Σ_j w_j) v_i`, the key-value scores factoring per token;
5. `channelPair r` is bijective — the split exists exactly on this domain, the elliptic face
   having no root witness and the parabolic face having `r = 0`.

    σ         [the parameter: what the adjoined direction squares to]
    ∧         [both are true]
    ringMul   [multiply two points together, the σ way]
    ringUpper [how far a point lies along the first direction of size zero]
    RingCell  [a point: two ring elements written side by side, a + b·x]
    List      [a finite sequence of things, in order]
    ×         [paired with, so R × R means two elements of R side by side] -/
theorem attention_channel_form {σ r : R} (q : RingCell R) (l : List (RingCell R × R))
    (hdom : ChannelFormDomain σ r q l) :
    channelAttnScan (channelStream σ r q l) = (channelDen σ r q l, channelNum σ r q l)
      ∧ channelAttnOut (channelAttnScan (channelStream σ r q l))
          = ((l.map (fun p => (channelScore σ r q p.1 / channelDen σ r q l) * p.2)).sum)
      ∧ channelAttnOut (channelAttnScan (channelStream σ r q l)) * channelDen σ r q l
          = channelNum σ r q l
      ∧ ((l.map (fun p => (channelScore σ r q p.1 / channelDen σ r q l) * p.2)).sum)
          = ((l.map (fun p => (((ringUpper r q * ringUpper r p.1) / (2 * (ringMul σ q p.1).1))
                / channelDen σ r q l) * p.2)).sum)
      ∧ Function.Bijective (channelPair r) := by
  unfold ChannelFormDomain at hdom
  rcases hdom with ⟨hr, h2, hr0, hden, hnorm⟩
  refine ⟨channelAttnScan_channelStream σ r q l, ?_, ?_, ?_,
    (hyperbolic_decomposition_field hr h2 hr0).1⟩
  · exact channelAttnOut_channelStream σ r q l
  · rw [channelAttnScan_channelStream σ r q l]
    show channelNum σ r q l / channelDen σ r q l * channelDen σ r q l = channelNum σ r q l
    exact div_mul_cancel₀ _ hnorm
  · refine congrArg List.sum (List.map_congr_left fun p hp => ?_)
    show (channelWeight r (ringMul σ q p.1) / channelDen σ r q l) * p.2
        = (((ringUpper r q * ringUpper r p.1) / (2 * (ringMul σ q p.1).1))
            / channelDen σ r q l) * p.2
    rw [channelWeight_factor hr q p.1 (hden p hp)]

/-- The factorizable domain is inhabited: a joint witness over the rationals at `σ = 1`,
`r = 1`, `q = ringOne`, one token `(ringOne, 1)`. The channel score computes to `1/2`, so the
normalizer is nonzero.

    ringOne  [the point (1, 0), which changes nothing when you multiply by it]
    RingCell [a point: two ring elements written side by side, a + b·x] -/
theorem channel_form_domain_witness :
    ChannelFormDomain (1:ℚ) 1 ringOne [((ringOne : RingCell ℚ), 1)] := by
  unfold ChannelFormDomain
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · norm_num
  · norm_num
  · norm_num
  · intro p hp
    simp only [List.mem_cons, List.mem_nil_iff] at hp
    rcases hp with rfl | hp
    · norm_num [ringMul, ringOne]
    · exact hp.elim
  · norm_num [channelDen, channelScore, channelWeight, ringUpper, ringLower, ringMul, ringOne]

/-- The refusal side of the normalizer: at the empty stream it is zero, so the normalization
is refused there and `channelDen σ r q l ≠ 0` is the admissibility premise.

    σ        [the parameter: what the adjoined direction squares to]
    RingCell [a point: two ring elements written side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelDen_nil {σ r : R} (q : RingCell R) :
    channelDen σ r q ([] : List (RingCell R × R)) = 0 := by
  simp [channelDen]

section ParabolicFace

/-! ### The parabolic face: `σ = 0`, `r = 0`

The root witness `0 ^ 2 = 0` holds but the split premise `r ≠ 0` fails, so `ChannelFormDomain`
is empty here (`channelFormDomain_parabolic_empty`) and `attention_channel_form` does not apply.
The channel reading degenerates instead: both channels read the first coordinate
(`ringUpper_parabolic`, `ringLower_parabolic`), so the channel weight is uniformly `1 / 2` where
the channel denominator is nonzero (`channelWeight_parabolic`), every score is `1 / 2`
(`channelScore_parabolic`), the normalizer is the token count times `1 / 2`
(`channelDen_parabolic`), and the readout is the uniform average of the values
(`attention_channel_form_parabolic`). The exact domain is `ParabolicChannelFormDomain`.
-/

/-- The parabolic face collapses the two channels: at `r = 0` the upper reading is the first
coordinate.

    ringUpper [how far a point lies along the first direction of size zero, over a ring]
    RingCell  [a point: two ring elements side by side, a + b·x] -/
theorem ringUpper_parabolic (z : RingCell R) : ringUpper (0:R) z = z.1 := by
  simp [ringUpper]

/-- The parabolic face collapses the two channels: at `r = 0` the lower reading is the first
coordinate, the same as the upper one.

    ringLower [how far a point lies along the other direction of size zero, over a ring]
    RingCell  [a point: two ring elements side by side, a + b·x] -/
theorem ringLower_parabolic (z : RingCell R) : ringLower (0:R) z = z.1 := by
  simp [ringLower]

/-- Parabolic channel weight: at `r = 0` both channels read `z.1`, so the weight is the constant
`1 / 2` wherever the channel denominator `2 * z.1` is nonzero.

    ≠        [is not equal to]
    RingCell [a point: two ring elements side by side, a + b·x] -/
theorem channelWeight_parabolic {z : RingCell R} (h : 2 * z.1 ≠ 0) :
    channelWeight (0:R) z = 1 / 2 := by
  have h2 : (2:R) ≠ 0 := by
    intro hh
    exact h (by rw [hh, zero_mul])
  have hz : z.1 ≠ 0 := by
    intro hh
    exact h (by rw [hh, mul_zero])
  rw [channelWeight_eq z h, ringUpper_parabolic]
  field_simp [h2, hz]

/-- Parabolic channel score: at `σ = 0`, `r = 0` every score is `1 / 2` where its channel
denominator is nonzero.

    σ        [the parameter: what the adjoined direction squares to]
    ≠        [is not equal to]
    ringMul  [multiply two points together, the σ way]
    RingCell [a point: two ring elements side by side, a + b·x] -/
theorem channelScore_parabolic {q k : RingCell R} (hk : 2 * (ringMul (0:R) q k).1 ≠ 0) :
    channelScore (0:R) (0:R) q k = 1 / 2 :=
  channelWeight_parabolic hk

/-- A constant map sums to the token count times the constant.

    List [a finite sequence of things, in order] -/
theorem list_map_const_sum {α : Type*} (l : List α) (c : R) :
    (l.map (fun _ => c)).sum = (l.length : R) * c := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    rw [ih]
    ring

/-- Parabolic normalizer: every score is `1 / 2`, so the normalizer is the token count times
`1 / 2`.

    σ        [the parameter: what the adjoined direction squares to]
    ≠        [is not equal to]
    ∀        [for every single one]
    ∈        [is one of the things in]
    ringMul  [multiply two points together, the σ way]
    RingCell [a point: two ring elements side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelDen_parabolic (q : RingCell R) (l : List (RingCell R × R))
    (hden : ∀ p ∈ l, 2 * (ringMul (0:R) q p.1).1 ≠ 0) :
    channelDen (0:R) (0:R) q l = (l.length : R) * (1 / 2) := by
  have hmap : (l.map fun p => channelScore (0:R) (0:R) q p.1) = l.map fun _ => (1 / 2 : R) :=
    List.map_congr_left fun p hp => channelScore_parabolic (hden p hp)
  show (l.map fun p => channelScore (0:R) (0:R) q p.1).sum = _
  rw [hmap]
  exact list_map_const_sum l (1 / 2)

/-- The parabolic normalizer is nonzero exactly when the token count is: given `2 ≠ 0`, the
normalizer `l.length * (1 / 2)` vanishes exactly when `(l.length : R)` does. This is the
`channelDen σ r q l ≠ 0` clause of `ChannelFormDomain`, read on this face.

    σ        [the parameter: what the adjoined direction squares to]
    ≠        [is not equal to]
    ↔        [exactly when, true in both directions]
    ∀        [for every single one]
    ∈        [is one of the things in]
    ringMul  [multiply two points together, the σ way]
    RingCell [a point: two ring elements side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelDen_parabolic_ne_zero_iff (h2 : (2:R) ≠ 0) (q : RingCell R)
    (l : List (RingCell R × R)) (hden : ∀ p ∈ l, 2 * (ringMul (0:R) q p.1).1 ≠ 0) :
    channelDen (0:R) (0:R) q l ≠ 0 ↔ ((l.length : R) ≠ 0) := by
  rw [channelDen_parabolic q l hden]
  exact mul_ne_zero_iff.trans (and_iff_left (one_div_ne_zero h2))

/-- The exact parabolic domain of the channel-form readout. The face fixes `σ = 0` and `r = 0`;
the premises are `2 ≠ 0` so the halving divides, each score's channel denominator
`2 * (ringMul 0 q k).1` nonzero, and a nonzero normalizer — which on this face is exactly
`(l.length : R) ≠ 0` (`channelDen_parabolic_ne_zero_iff`). This is `ChannelFormDomain` with the
root witness read at `r = 0` and the split premise `r ≠ 0` dropped: the parabolic face carries
no split (`channelFormDomain_parabolic_empty`), so the readout on this domain is the uniform
average of the values, not a channel weighting.

    σ        [the parameter: what the adjoined direction squares to]
    ≠        [is not equal to]
    ∧        [both are true]
    ∀        [for every single one]
    ∈        [is one of the things in]
    ringMul  [multiply two points together, the σ way]
    RingCell [a point: two ring elements side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
def ParabolicChannelFormDomain (q : RingCell R) (l : List (RingCell R × R)) : Prop :=
  (2:R) ≠ 0 ∧ (∀ p ∈ l, 2 * (ringMul (0:R) q p.1).1 ≠ 0) ∧ ((l.length : R) ≠ 0)

/-- The parabolic face is outside the factorizable domain: `ChannelFormDomain` demands `r ≠ 0`
and the face has `r = 0`, so no inhabitant exists there and `attention_channel_form` says
nothing about the parabolic readout.

    ≠        [is not equal to]
    ¬        [this is not true]
    RingCell [a point: two ring elements side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelFormDomain_parabolic_empty {q : RingCell R} {l : List (RingCell R × R)} :
    ¬ ChannelFormDomain (0:R) (0:R) q l := by
  intro h
  exact h.2.2.1 rfl

/-- **Parabolic channel-form attention (degenerate face).** At `σ = 0`, `r = 0` the channel
weights are uniformly `1 / 2` and the normalized readout is the uniform average of the values:
`channelAttnOut (channelAttnScan (channelStream 0 0 q l)) = (Σ_i v_i) / l.length`. The exact
domain is `ParabolicChannelFormDomain`; at the empty stream the normalizer vanishes
(`channelDen_nil`) and no division is meaningful. The streaming state is
`(channelDen 0 0 q l, channelNum 0 0 q l)` with `channelDen 0 0 q l = l.length * (1 / 2)`
(`channelDen_parabolic`). This is the degenerate face of `attention_channel_form`, which
requires the split premise `r ≠ 0` and so excludes it (`channelFormDomain_parabolic_empty`).

    σ        [the parameter: what the adjoined direction squares to]
    ≠        [is not equal to]
    ∀        [for every single one]
    ∈        [is one of the things in]
    ringMul  [multiply two points together, the σ way]
    RingCell [a point: two ring elements side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem attention_channel_form_parabolic (q : RingCell R) (l : List (RingCell R × R))
    (hdom : ParabolicChannelFormDomain q l) :
    channelAttnOut (channelAttnScan (channelStream (0:R) (0:R) q l))
      = (l.map (fun p => p.2)).sum / (l.length : R) := by
  rcases hdom with ⟨h2, hden, hlen⟩
  rw [channelAttnOut_channelStream, channelDen_parabolic q l hden, list_sum_div_map]
  simp only [List.map_map]
  refine congrArg List.sum (List.map_congr_left fun p hp => ?_)
  show (channelScore (0:R) (0:R) q p.1 / ((l.length : R) * (1 / 2))) * p.2
    = p.2 / (l.length : R)
  rw [channelScore_parabolic (hden p hp)]
  have hhalf : (1 / 2 : R) ≠ 0 := one_div_ne_zero h2
  have hprod : ((l.length : R)) * (1 / 2) ≠ 0 := mul_ne_zero hlen hhalf
  field_simp [h2, hlen, hhalf, hprod]

end ParabolicFace

section EllipticFace

/-! ### The elliptic face: `σ < 0` over `ℝ`, excluded

No root witness `r ^ 2 = σ` exists at `σ < 0` over the reals: a square is nonnegative
(`sq_nonneg`), so `r ^ 2 = σ < 0` is impossible (`elliptic_no_root`). The channel split is not
defined on this face, `ChannelFormDomain` has no inhabitant
(`channelFormDomain_elliptic_empty`), and no channel-form attention claim covers it. This is the
excluded population of this module, excluded because no root witness exists: no channel reading
at a root is available and the factorizable kernel cannot be formed. The population is excluded,
not estimated; nothing here is approximated by a split at a non-root.
-/

/-- **Elliptic obstruction.** A square is nonnegative in the reals, so a negative `σ` admits no
root witness `r ^ 2 = σ`.

    σ  [the parameter: what the adjoined direction squares to]
    <  [is smaller than]
    ¬  [this is not true]
    ℝ  [the real numbers: every number on the number line] -/
theorem elliptic_no_root {σ : ℝ} (hσ : σ < 0) (r : ℝ) : ¬ r ^ 2 = σ := by
  intro hr
  have hsq : (0:ℝ) ≤ r ^ 2 := sq_nonneg r
  linarith

/-- The elliptic face admits no root witness at all: nothing squares to a negative `σ`.

    σ  [the parameter: what the adjoined direction squares to]
    <  [is smaller than]
    ¬  [this is not true]
    ∃  [there is at least one]
    ℝ  [the real numbers: every number on the number line] -/
theorem elliptic_no_root_witness {σ : ℝ} (hσ : σ < 0) : ¬ ∃ r : ℝ, r ^ 2 = σ := by
  rintro ⟨r, hr⟩
  exact elliptic_no_root hσ r hr

/-- **Excluded population: the elliptic face.** `ChannelFormDomain` has no inhabitant at
`σ < 0` over `ℝ`: its root-witness premise `r ^ 2 = σ` is unsatisfiable (`elliptic_no_root`), so
the channel split does not exist there and no channel-form claim covers that population.

    σ        [the parameter: what the adjoined direction squares to]
    <        [is smaller than]
    ¬        [this is not true]
    ℝ        [the real numbers: every number on the number line]
    RingCell [a point: two ring elements side by side, a + b·x]
    List     [a finite sequence of things, in order]
    ×        [paired with, so R × R means two elements of R side by side] -/
theorem channelFormDomain_elliptic_empty {σ r : ℝ} (hσ : σ < 0)
    (q : RingCell ℝ) (l : List (RingCell ℝ × ℝ)) : ¬ ChannelFormDomain σ r q l :=
  fun h => elliptic_no_root hσ r h.1

end EllipticFace

end Nous.Model
