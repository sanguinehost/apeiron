import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Mathlib.Algebra.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

/-!
# Channel-Ratio Weighting Over a Field

For a quadratic parameter `σ` and root witness `r` with `r ^ 2 = σ`,
the coordinate projections `ringUpper r z = z.1 + r * z.2` and
`ringLower r z = z.1 - r * z.2` define ring channel homomorphisms.

The channel weighting function is defined over a field `R` by:
`channelWeight r z := ringUpper r z / (ringUpper r z + ringLower r z)`.

Because `ringUpper r z + ringLower r z = 2 * z.1`, whenever `2 * z.1 ≠ 0`
the denominator simplifies to `2 * z.1`:
`channelWeight r z = ringUpper r z / (2 * z.1)`.
When `2 * z.1 = 0`, the denominator is zero and the ratio is unavailable
(zero-denominator refusal).

The channel projections `ringUpper r` and `ringLower r` are multiplicative
homomorphisms (`RingChannel σ`), so they compose multiplicatively across
sequential token steps:
`ringUpper r (ringMul σ s x) = ringUpper r s * ringUpper r x`,
and across list folds:
`ringUpper r (xs.foldl (ringMul σ) s) = ringUpper r s * (xs.map (ringUpper r)).prod`.

The ratio `channelWeight r z` itself is recomputed at each step from the
projections and is not claimed to be multiplicative. No attention mechanism is
claimed. On the positive domain the ratio is the binary softmax of the
log-channel pair: `ratio_eq_exp_log`, `channelWeight_binary_softmax`. The
original un-substituted softmax-equivalence phrasing stays declined.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

section Multiplicativity

variable {R : Type*} [CommRing R]

/-- The sum of the two channel projections is twice the first coordinate. -/
theorem ringUpper_add_ringLower (r : R) (z : RingCell R) :
    ringUpper r z + ringLower r z = 2 * z.1 := by
  simp only [ringUpper, ringLower]
  ring

/-- The difference of the two channel projections is `2 * r * z.2`. -/
theorem ringUpper_sub_ringLower (r : R) (z : RingCell R) :
    ringUpper r z - ringLower r z = 2 * r * z.2 := by
  simp only [ringUpper, ringLower]
  ring

/-- Upper channel multiplicativity across a single token step. -/
theorem ringUpper_tokenStep {σ r : R} (hr : r ^ 2 = σ) (s x : RingCell R) :
    ringUpper r (ringMul σ s x) = ringUpper r s * ringUpper r x :=
  ringUpper_mul hr s x

/-- Lower channel multiplicativity across a single token step. -/
theorem ringLower_tokenStep {σ r : R} (hr : r ^ 2 = σ) (s x : RingCell R) :
    ringLower r (ringMul σ s x) = ringLower r s * ringLower r x :=
  ringLower_mul hr s x

/-- Upper channel multiplicativity across a sequence of token steps folded from a seed. -/
theorem ringUpper_foldl {σ r : R} (hr : r ^ 2 = σ)
    (xs : List (RingCell R)) (s : RingCell R) :
    ringUpper r (xs.foldl (ringMul σ) s) =
      ringUpper r s * (xs.map (ringUpper r)).prod := by
  induction xs generalizing s with
  | nil =>
    simp only [List.foldl_nil, List.map_nil, List.prod_nil, mul_one]
  | cons x rest ih =>
    simp only [List.foldl_cons, List.map_cons, List.prod_cons]
    rw [ih (ringMul σ s x), ringUpper_tokenStep hr s x]
    ring

/-- Lower channel multiplicativity across a sequence of token steps folded from a seed. -/
theorem ringLower_foldl {σ r : R} (hr : r ^ 2 = σ)
    (xs : List (RingCell R)) (s : RingCell R) :
    ringLower r (xs.foldl (ringMul σ) s) =
      ringLower r s * (xs.map (ringLower r)).prod := by
  induction xs generalizing s with
  | nil =>
    simp only [List.foldl_nil, List.map_nil, List.prod_nil, mul_one]
  | cons x rest ih =>
    simp only [List.foldl_cons, List.map_cons, List.prod_cons]
    rw [ih (ringMul σ s x), ringLower_tokenStep hr s x]
    ring

/-- Upper channel multiplicativity for a sequence folded from the unit element `ringOne`. -/
theorem ringUpper_foldl_ringOne {σ r : R} (hr : r ^ 2 = σ)
    (xs : List (RingCell R)) :
    ringUpper r (xs.foldl (ringMul σ) ringOne) =
      (xs.map (ringUpper r)).prod := by
  rw [ringUpper_foldl hr xs ringOne, ringUpper_ringOne r, one_mul]

/-- Lower channel multiplicativity for a sequence folded from the unit element `ringOne`. -/
theorem ringLower_foldl_ringOne {σ r : R} (hr : r ^ 2 = σ)
    (xs : List (RingCell R)) :
    ringLower r (xs.foldl (ringMul σ) ringOne) =
      (xs.map (ringLower r)).prod := by
  rw [ringLower_foldl hr xs ringOne, ringLower_ringOne r, one_mul]

end Multiplicativity

section Weighting

variable {R : Type*} [Field R]

/-- Channel weighting ratio: upper channel coordinate divided by the total channel sum. -/
def channelWeight (r : R) (z : RingCell R) : R :=
  ringUpper r z / (ringUpper r z + ringLower r z)

/-- Complementary lower channel weighting ratio. -/
def lowerChannelWeight (r : R) (z : RingCell R) : R :=
  ringLower r z / (ringUpper r z + ringLower r z)

/-- When `2 * z.1 ≠ 0`, the channel weight equals `ringUpper r z / (2 * z.1)`. -/
theorem channelWeight_eq {r : R} (z : RingCell R) (_h : 2 * z.1 ≠ 0) :
    channelWeight r z = ringUpper r z / (2 * z.1) := by
  simp only [channelWeight, ringUpper_add_ringLower]

/-- Conditional equality of channel weight under non-zero denominator condition. -/
theorem channelWeight_eq_of_ne {r : R} (z : RingCell R) (h : 2 * z.1 ≠ 0) :
    channelWeight r z = ringUpper r z / (2 * z.1) :=
  channelWeight_eq z h

/-- Explicit-root variant of `channelWeight_eq`. -/
theorem channelWeight_eq' (r : R) (z : RingCell R) (h : 2 * z.1 ≠ 0) :
    channelWeight r z = ringUpper r z / (2 * z.1) :=
  channelWeight_eq z h

/-- When `2 * z.1 ≠ 0`, the complementary lower weight equals `ringLower r z / (2 * z.1)`. -/
theorem lowerChannelWeight_eq {r : R} (z : RingCell R) (_h : 2 * z.1 ≠ 0) :
    lowerChannelWeight r z = ringLower r z / (2 * z.1) := by
  simp only [lowerChannelWeight, ringUpper_add_ringLower]

/-- The channel denominator vanishes if and only if `2 * z.1 = 0`. -/
theorem channelWeight_denominator_eq_zero_iff (r : R) (z : RingCell R) :
    ringUpper r z + ringLower r z = 0 ↔ 2 * z.1 = 0 := by
  rw [ringUpper_add_ringLower]

/-- The channel weights sum to unity when `2 * z.1 ≠ 0`. -/
theorem channelWeight_add_lowerChannelWeight {r : R} (z : RingCell R)
    (h : 2 * z.1 ≠ 0) :
    channelWeight r z + lowerChannelWeight r z = 1 := by
  rw [channelWeight_eq z h, lowerChannelWeight_eq z h, ← add_div,
    ringUpper_add_ringLower, div_self h]

end Weighting

section Softmax

/-- Two-class softmax over the log-channel pair: `exp x / (exp x + exp y)`.
Softmax: Boltzmann (1877), Gibbs (1902); the name is Bridle (1990). -/
noncomputable def softmax2 (x y : ℝ) : ℝ :=
  Real.exp x / (Real.exp x + Real.exp y)

/-- The channel ratio `u / (u + l)` is the binary softmax of the log-channel pair
`(Real.log u, Real.log l)` for positive `u`, `l`, in explicit exponential form. -/
theorem ratio_eq_exp_log {u l : ℝ} (hu : 0 < u) (hl : 0 < l) :
    u / (u + l) =
      Real.exp (Real.log u) / (Real.exp (Real.log u) + Real.exp (Real.log l)) := by
  rw [Real.exp_log hu, Real.exp_log hl]

/-- `ratio_eq_exp_log` under the `softmax2` abbreviation. -/
theorem ratio_eq_softmax2_log {u l : ℝ} (hu : 0 < u) (hl : 0 < l) :
    u / (u + l) = softmax2 (Real.log u) (Real.log l) :=
  ratio_eq_exp_log hu hl

/-- Channel-ratio form of the binary-softmax identification: with `u = ringUpper r z`
and `l = ringLower r z` both positive, `channelWeight r z` is the softmax of
`(Real.log u, Real.log l)`. The denominator `2 * z.1 = u + l` is nonzero because
`u + l > 0`. -/
theorem channelWeight_binary_softmax {r : ℝ} {z : RingCell ℝ} {u l : ℝ}
    (hu : ringUpper r z = u) (hl : ringLower r z = l) (hupos : 0 < u) (hlpos : 0 < l) :
    channelWeight r z = softmax2 (Real.log u) (Real.log l) := by
  have h2 : 2 * z.1 = u + l := by
    rw [← ringUpper_add_ringLower r z, hu, hl]
  have hden : 2 * z.1 ≠ 0 := by
    rw [h2]
    exact ne_of_gt (by linarith)
  rw [channelWeight_eq z hden, h2, hu]
  exact ratio_eq_softmax2_log hupos hlpos

end Softmax

end Nous.Model
