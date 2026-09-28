import Nous.Model.Weighting
import Numeric.EngineSiluRefinement

/-!
# The SiLU gate as the carrier's binary channel weight

The sigmoid-linear unit `siluReal x = x / (1 + exp (-x))` (`Numeric/EngineSiluRefinement.lean`)
carries a logistic gate `siluGate x = 1 / (1 + exp (-x))`. This module shows that gate is the
carrier's binary channel weight at the point `siluCell x = ((exp x + 1) / 2, (exp x - 1) / 2)` in
`R[x]/(x² − σ)` at `σ = 1` (`Nous.Model.Weighting`), whose norm form is the exponential.

## The reuse link

`silu_gate_eq_channelWeight` is the link: `siluGate x = channelWeight (1 : ℝ) (siluCell x)`. The
logistic gate the `SiLU` kernel multiplies by, the two-class softmax over the log-channel pair
(`siluGate_eq_softmax2` composed with `channelWeight_binary_softmax`), and the hyperbolic channel
fold (`channelWeight` over `RingCell ℝ` at `σ = 1`) are one object read three ways. The two
refinements `siluCell_normForm` (the exponential is the cell's norm form) and
`siluReal_eq_mul_channelWeight` (the kernel's real value is `x` times that channel weight)
instantiate the canonical definitions rather than restating them.

## Root

`Apeiron.Trichotomy.AnyRing.ringNormForm`, `Apeiron.Trichotomy.ChannelsAnyRing.ringUpper`, and
`Nous.Model.channelWeight` are used at their `σ = 1` real instantiation. No new algebraic identity
is introduced: each theorem rewrites an existing canonical quantity to a concrete real value.

## Evidence

Grade `proved`: each theorem below is a Lean kernel check, with no `sorry`, `admit` or `set_option`.
`#guard`s are omitted because `ℝ` is not computable, so no value decision procedure exists for these
statements.

## Trust boundary

No claim reaches `rustc`/LLVM, the kernel ABI, or the fixed-point extraction's rounding error: this
module is about the real-valued reference `siluReal`, not about the extracted `word_silu_fixed`
value bound. The `silu_window_refines` error statement in `Numeric/EngineSiluRefinement.lean` is a
separate obligation.
-/

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

noncomputable section

/-- The logistic gate of the sigmoid-linear unit: `1 / (1 + exp (-x))`.

    ℝ [the real numbers: every number on the number line] -/
def siluGate (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

/-- The carrier point at `σ = 1` whose channels are `(exp x + 1) / 2` and `(exp x - 1) / 2`; its
upper channel reading is `exp x` and its lower reading is `1`.

    ℝ        [the real numbers: every number on the number line]
    RingCell [a point: two ring elements side by side, a + b·x] -/
def siluCell (x : ℝ) : RingCell ℝ := ⟨(Real.exp x + 1) / 2, (Real.exp x - 1) / 2⟩

/-- The exponential is the norm form of `siluCell x` at `σ = 1`: the point's size is `exp x`.

    ringNormForm [the norm form: a point times its conjugate, which can come out negative]
    ℝ            [the real numbers: every number on the number line] -/
theorem siluCell_normForm (x : ℝ) :
    ringNormForm (1 : ℝ) (siluCell x) = Real.exp x := by
  simp only [ringNormForm, siluCell]
  ring

/-- The logistic gate is the two-class softmax of the log-channel pair `(x, 0)`.

    ℝ [the real numbers: every number on the number line] -/
theorem siluGate_eq_softmax2 (x : ℝ) : siluGate x = softmax2 x 0 := by
  unfold siluGate softmax2
  rw [Real.exp_neg, Real.exp_zero]
  have h1 : (1 : ℝ) + (Real.exp x)⁻¹ = (Real.exp x + 1) / Real.exp x := by
    rw [add_div, div_self (Real.exp_ne_zero x), one_div]
  rw [h1, one_div_div]

/-- The channel weight of `siluCell x` at `σ = 1` is `exp x / (exp x + 1)`. The denominator
`2 * (siluCell x).1 = exp x + 1` is nonzero because `exp x > 0`.

    ℝ [the real numbers: every number on the number line] -/
theorem channelWeight_siluCell (x : ℝ) :
    channelWeight (1 : ℝ) (siluCell x) = Real.exp x / (Real.exp x + 1) := by
  have htwice : 2 * (siluCell x).1 = Real.exp x + 1 := by
    simp only [siluCell]
    ring
  have hden : 2 * (siluCell x).1 ≠ 0 := by
    rw [htwice]
    positivity
  rw [channelWeight_eq' (1 : ℝ) (siluCell x) hden]
  have hupper : ringUpper (1 : ℝ) (siluCell x) = Real.exp x := by
    simp only [ringUpper, siluCell]
    ring
  rw [hupper, htwice]

/-- **Reuse link.** The logistic gate is the carrier's binary channel weight at `siluCell x`:
`siluGate x = channelWeight (1 : ℝ) (siluCell x)`.

    ℝ [the real numbers: every number on the number line] -/
theorem silu_gate_eq_channelWeight (x : ℝ) :
    siluGate x = channelWeight (1 : ℝ) (siluCell x) := by
  rw [siluGate_eq_softmax2, channelWeight_siluCell]
  unfold softmax2
  rw [Real.exp_zero]

/-- The sigmoid-linear unit's real value is `x` times the carrier's channel weight:
`siluReal x = x * channelWeight (1 : ℝ) (siluCell x)`.

    ℝ [the real numbers: every number on the number line] -/
theorem siluReal_eq_mul_channelWeight (x : ℝ) :
    apeiron_word.siluReal x = x * channelWeight (1 : ℝ) (siluCell x) := by
  rw [← silu_gate_eq_channelWeight]
  unfold apeiron_word.siluReal siluGate
  rw [div_eq_mul_one_div]

end

end Nous.Model
