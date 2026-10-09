import Apeiron.Trichotomy.Flow
import Apeiron.Physics.Quadratic
import Mathlib.Tactic

/-!
# Kepler's equation on the elliptic flow

The second coordinate of the flow at parameter minus one is the sine of its time.
Writing Kepler's equation with that coordinate fixes the trichotomy face used by
the nonlinear equation. The equation does not assert that flow composition or an
affine fold computes its root.

## Root

| from | here |
|---|---|
| `Trichotomy.flow_neg_one` | `flow_snd_neg_one`, `equation_iff_flow` |
| `Trichotomy.flow_neg_one`, `Physics.flowPoint_neg_one` | `flowPoint_eq_flow_neg_one` |
| `Physics.flow_is_one_param`, `flowPoint_eq_flow_neg_one` | `flow_neg_one_is_one_param` |
| `Physics.flow_is_one_param_unique`, `flowPoint_eq_flow_neg_one` | `flow_neg_one_unique` |

The derivative and uniqueness statements concern a linear generator flow, not
the nonlinear fixed point of the Kepler equation. This file uses only the
normalized parameter minus one; arbitrary negative parameters rescale time.
-/

namespace Apeiron.Physics.Kepler

/-- The second slot of the standard elliptic flow is sine.

    (x : ℝ)                                     [a real flow parameter]
    (Trichotomy.flow (-1) x).2 = Real.sin x    [the second slot is its sine]
    flow                                        [the size-one flow of the algebra]
    ℝ                      [the real numbers: every number on the number line] -/
theorem flow_snd_neg_one (x : ℝ) : (Trichotomy.flow (-1) x).2 = Real.sin x := by
  exact congrArg Prod.snd (Trichotomy.flow_neg_one x)

/-- Kepler's equation reads its sine from the elliptic flow.

    (m e x : ℝ)                                [mean value, coefficient and candidate]
    m = x - e * Real.sin x                    [the real Kepler equation]
    ↔                                           [exactly when, true in both directions]
    m = x - e * (Trichotomy.flow (-1) x).2   [the same equation using the flow]
    flow                                        [the size-one flow of the algebra] -/
theorem equation_iff_flow (m e x : ℝ) :
    m = x - e * Real.sin x ↔ m = x - e * (Trichotomy.flow (-1) x).2 := by
  rw [flow_snd_neg_one]

/-- The quadratic flow point and the trichotomy flow agree at minus one.

    (t : ℝ)                                   [a real flow parameter]
    Physics.flowPoint (-1) t                  [the quadratic-flow point]
    = Trichotomy.flow (-1) t                  [the trichotomy-flow point]
    flow                                        [the size-one flow of the algebra] -/
theorem flowPoint_eq_flow_neg_one (t : ℝ) :
    Physics.flowPoint (-1) t = Trichotomy.flow (-1) t := by
  rw [Physics.flowPoint_neg_one, Trichotomy.flow_neg_one]
  rfl

/-- The normalized elliptic flow solves the quadratic generator equation.

    (x0 : Trichotomy.Cell)                    [the initial point]
    (t : ℝ)                                   [a real flow parameter]
    HasDerivAt                                [is changing at that rate at that time]
    (fun u => Trichotomy.mul (-1) (Trichotomy.flow (-1) u) x0)
                                               [the flow applied to the initial point]
    (Trichotomy.mul (-1) ((0 : ℝ), (1 : ℝ))
      (Trichotomy.mul (-1) (Trichotomy.flow (-1) t) x0))
                                               [the generator acting on the point]
    t                                          [the time of the derivative]
    mul                                        [multiply two points together, the σ way]
    flow                                       [the size-one flow of the algebra] -/
theorem flow_neg_one_is_one_param (x0 : Trichotomy.Cell) (t : ℝ) :
    HasDerivAt (fun u => Trichotomy.mul (-1) (Trichotomy.flow (-1) u) x0)
      (Trichotomy.mul (-1) ((0 : ℝ), (1 : ℝ))
        (Trichotomy.mul (-1) (Trichotomy.flow (-1) t) x0)) t := by
  simpa only [flowPoint_eq_flow_neg_one] using Physics.flow_is_one_param (-1) x0 t

/-- A curve with this generator and initial point is the normalized flow.

    (x0 : Trichotomy.Cell)                    [the initial point]
    (g : ℝ → Trichotomy.Cell)                  [a curve of points]
    (hg : ∀ t, HasDerivAt g
      (Trichotomy.mul (-1) ((0 : ℝ), (1 : ℝ)) (g t)) t)
                                               [the generator equation at every time]
    (h0 : g 0 = x0)                           [the initial value]
    g = fun t => Trichotomy.mul (-1) (Trichotomy.flow (-1) t) x0
                                               [the curve is the elliptic flow]
    ∀                                          [for every single one]
    HasDerivAt                                [is changing at that rate at that time]
    mul                                        [multiply two points together, the σ way]
    flow                                       [the size-one flow of the algebra] -/
theorem flow_neg_one_unique (x0 : Trichotomy.Cell) (g : ℝ → Trichotomy.Cell)
    (hg : ∀ t, HasDerivAt g
      (Trichotomy.mul (-1) ((0 : ℝ), (1 : ℝ)) (g t)) t) (h0 : g 0 = x0) :
    g = fun t => Trichotomy.mul (-1) (Trichotomy.flow (-1) t) x0 := by
  simpa only [flowPoint_eq_flow_neg_one] using
    Physics.flow_is_one_param_unique (-1) x0 g hg h0

end Apeiron.Physics.Kepler
