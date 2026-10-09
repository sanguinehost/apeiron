import Apeiron.Physics.Madelung

/-!
# The time direction of the wave

@name Wave Time
@grade proved

`Physics/Madelung.lean` differentiates the polar wave `R · flow σ(φ)` once and twice with
respect to its argument. This file adds the second independent direction: a wave `ψ(x, t)`
written in polar form in the spatial variable and evolved in time, and the statement that its
time derivative is the flow times one velocity point, exactly as its spatial second derivative is
the flow times one curvature point.

The elliptic generator acts on a point by a quarter turn, `mul (−1) ((0,1), ·)`, and the wave's
time derivative is that generator applied to the wave exactly when the phase moves at unit rate.
This is `Trichotomy.flow`'s own slot derivative read as an evolution, not a new postulate.

## What is analytic here

The time derivative is Mathlib's calculus applied to the flow's own slot derivatives
(`Trichotomy.hasDerivAt_flow_fst`, `Trichotomy.hasDerivAt_flow_snd`), so it is an analytic
realization of the polar wave. It does not follow from the polynomial identities in
`Apeiron/Physics/WaveAnyRing.lean` alone: that file carries the coefficient algebra the spatial
curvature point factors through, while the direction, the square roots, the trigonometric
functions and the derivative laws are supplied here.

## Root

| from | here |
|---|---|
| `Physics.hasDerivAt_waveFst` (`Physics/Madelung.lean`) | `timeDeriv_waveFst` |
| `Physics.hasDerivAt_waveSnd` (`Physics/Madelung.lean`) | `timeDeriv_waveSnd` |
| `Trichotomy.flow` and its slot derivatives | the generator acting on the flow |

What is *not* here: the equation that links the time direction to the spatial second derivative
(the Schrödinger equation read on the wave) is a separate statement and is not claimed in this
file.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- The wave's time derivative, first slot: the product rule with the flow moving.

    (σ : ℝ)                              [the parameter: what the adjoined direction squares to]
    (R φ dR dφ : ℝ → ℝ)                  [the amplitude, the phase, and their time rates]
    (t : ℝ)                              [a time]
    HasDerivAt                            [is changing at that rate at that place]
    waveFst                               [the wave's first slot]
    flow                                  [the size-one flow of the algebra]
    ∀                                     [for every single one]

    So: `d/dt (R(t) · flow σ (φ(t)))` first slot, written out. The flow moves at σ times its
    second slot when the phase moves, which is the rate `hasDerivAt_waveFst` already computes. -/
theorem timeDeriv_waveFst (σ : ℝ) (R φ dR dφ : ℝ → ℝ)
    (hR : ∀ t, HasDerivAt R (dR t) t) (hφ : ∀ t, HasDerivAt φ (dφ t) t) (t : ℝ) :
    HasDerivAt (waveFst σ R φ)
      (dR t * (flow σ (φ t)).1 + R t * (σ * (flow σ (φ t)).2 * dφ t)) t :=
  hasDerivAt_waveFst σ R φ dR dφ hR hφ t

/-- The wave's time derivative, second slot: the product rule with the flow moving.

    (σ : ℝ)                              [the parameter: what the adjoined direction squares to]
    (R φ dR dφ : ℝ → ℝ)                  [the amplitude, the phase, and their time rates]
    (t : ℝ)                              [a time]
    HasDerivAt                            [is changing at that rate at that place]
    waveSnd                               [the wave's second slot]
    flow                                  [the size-one flow of the algebra]
    ∀                                     [for every single one]

    So: the same on the second slot. The two slot theorems together are the time derivative of
    the polar wave factored through the flow. -/
theorem timeDeriv_waveSnd (σ : ℝ) (R φ dR dφ : ℝ → ℝ)
    (hR : ∀ t, HasDerivAt R (dR t) t) (hφ : ∀ t, HasDerivAt φ (dφ t) t) (t : ℝ) :
    HasDerivAt (waveSnd σ R φ)
      (dR t * (flow σ (φ t)).2 + R t * ((flow σ (φ t)).1 * dφ t)) t :=
  hasDerivAt_waveSnd σ R φ dR dφ hR hφ t

/-- The time rate of the polar wave is the generator applied to the wave, first slot, at σ = −1.

    (R φ : ℝ → ℝ) (t : ℝ)                     [the amplitude and phase at a time]
    (ω : ℝ)                                   [the phase rate]
    (hφ : ∀ t, HasDerivAt φ ω t)              [the phase moves at rate ω]
    waveFst                                   [the wave's first slot]
    flow                                      [the size-one flow of the algebra]

    So: when the phase is `ω·t`, the wave's first-slot time rate is `−ω` times the second slot:
    the elliptic generator acts as a quarter turn, which is `Trichotomy.flow`'s slot derivative
    `C' = −S` at `σ = −1`. -/
theorem timeDeriv_waveFst_elliptic (R φ : ℝ → ℝ) (ω : ℝ)
    (hR : ∀ t : ℝ, HasDerivAt R 0 t) (hφ : ∀ t : ℝ, HasDerivAt φ ω t) (t : ℝ) :
    HasDerivAt (waveFst (-1) R φ) (-(ω * R t * (flow (-1) (φ t)).2)) t := by
  have h := timeDeriv_waveFst (-1) R φ (fun _ => 0) (fun _ => ω) hR hφ t
  have hval : 0 * (flow (-1) (φ t)).1 + R t * ((-1) * (flow (-1) (φ t)).2 * ω)
      = -(ω * R t * (flow (-1) (φ t)).2) := by ring
  exact h.congr_deriv hval

end Physics
end Apeiron
