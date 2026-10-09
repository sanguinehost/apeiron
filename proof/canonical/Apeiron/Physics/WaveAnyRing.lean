import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing

/-!
# The algebraic wave and its finite jet over any commutative ring

@name The Algebraic Wave Over Any Commutative Ring
@cite kocik-1999
@cite euclid-elements-ii-5
@grade proved

`Apeiron.Physics.Madelung` writes a wave as an amplitude times the algebra's flow and
differentiates it twice. `Apeiron.Physics.ClassicalWaves` samples the norm form at four points.
Once the flow's slot derivatives are supplied, both rest on identities between ring elements:
the trigonometry, the square roots and the derivatives are the analytic realization, and the
polynomial identities are what remains when that realization is set aside. This file installs
those identities over an arbitrary commutative ring, so the real statements are specialisations
rather than restatements.

`ringWave σ r u` is the wave point: the amplitude coefficient `r` on the unit basis direction
multiplied by the phase point `u`, written with `ringMul` as `ringMul σ (r, 0) u`. Its norm
form is the product density, `r²` times the phase's own size (`ringNormForm_ringWave`).

`ringCurv σ a dR ddR dφ ddφ` is the curvature point: the second-order data of the amplitude and
the phase assembled into one point. `ringJet` assembles the wave's second derivative from its
coefficients, and `ringMul_ringCurv` says the phase point multiplied by the curvature point is
that jet. That is the polynomial content of `Physics.secondDeriv_waveFst` and
`Physics.secondDeriv_waveSnd`; the analytic content, that the flow's second derivative is the
jet, needs `Trichotomy.hasDerivAt_flow_fst`, `Trichotomy.hasDerivAt_flow_snd` and Mathlib's
calculus.

`ringStencil4_normForm` is the exact four-sample identity of the norm form: the symmetric sum of
four samples minus four times the centre is `2·(1−σ)·h²`, at every step `h`, with no limit
taken. It is an identity of polynomials with whole-number coefficients, so it holds in every
commutative ring or in none. `ringNormForm_wick` is the sign flip: an element squaring to minus
one turns the norm form at `σ` into the norm form at `−σ`, which is the algebraic half of
`Physics.wick_flips_the_sign`.

What is not here: the flow, the trigonometric and square-root realization of it, the derivative
theorems, and the Schrödinger reading. Those are analytic and live in the files that carry them.

## Root

| from | here |
|---|---|
| `AnyRing.ringMul`, `AnyRing.ringNormForm` | `ringWave`, `ringCurv`, `ringJet` |
| `AnyRing.ringNormForm_mul` | `ringNormForm_ringWave` |
| `AnyRing.ringMul` | `ringMul_ringCurv`, `ringCurv_snd_flux` |
| `AnyRing.ringNormForm` | `ringStencil4_normForm`, `ringNormForm_wick` |

`ringWave` is `ringMul σ (r, 0) u`, the amplitude coefficient stretching the phase point, and
`ringNormForm_ringWave` reads its size as `r ^ 2` times the phase's own size. `ringCurv` and
`ringJet` assemble the second-order data, and `ringMul_ringCurv` is the polynomial half of
`Physics.secondDeriv_waveFst` and `Physics.secondDeriv_waveSnd`. `ringStencil4_normForm` and
`ringNormForm_wick` are polynomial identities with whole-number coefficients, so they hold in
every commutative ring. What does not transfer is the analytic realization: the flow, its
trigonometric and square-root realization, the derivative theorems and the Schrödinger reading
stay in `Madelung.lean`, `ClassicalWaves.lean` and `Wick.lean`. `ringNormForm_wick` is the
algebraic half of `wick_flips_the_sign`, whose `i` with `i ^ 2 = -1` is the complex instance.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- The wave point: an amplitude coefficient stretching a phase point.

    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (r : R)                    [the amplitude: the ring element on the unit direction]
    (u : RingCell R)           [the phase direction: a point the amplitude stretches]
    R                          [the commutative ring the point is built over]
    ringMul σ (r, 0) u         [the algebra's product of the amplitude and the phase point]

    So: the polar form `r · u` written with the algebra's own product, the construction the
    multiplication laws are stated over. -/
abbrev ringWave (σ r : R) (u : RingCell R) : RingCell R := ringMul σ (r, 0) u

/-- The wave point's norm form is the amplitude squared times the phase's own size.

    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (r : R)                    [the amplitude: the ring element on the unit direction]
    (u : RingCell R)           [the phase direction: a point the amplitude stretches]
    ringNormForm σ (ringWave σ r u) = r ^ 2 * ringNormForm σ u
                               [the size of the wave point is the amplitude squared times the
                                phase point's size]
    ringNormForm_mul           [the size of a product is the product of the sizes]
    `ringWave`                  [an amplitude coefficient on the unit direction times a phase point]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    So: the product density. It is `ringNormForm_mul` with the amplitude's size `r²` read off
    `(r, 0)`; no second polynomial expansion is done. -/
theorem ringNormForm_ringWave (σ r : R) (u : RingCell R) :
    ringNormForm σ (ringWave σ r u) = r ^ 2 * ringNormForm σ u := by
  simp only [ringWave]
  rw [ringNormForm_mul]
  simp only [ringNormForm]
  ring

/-- The curvature point: the second-order data of the amplitude and the phase.

    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (a dR ddR dφ ddφ : R)      [the amplitude's value, slope and curvature, and the phase's
                                slope and curvature]
    RingCell R                 [the point: two ring elements written side by side]
    ddR + σ * a * dφ ^ 2       [the first slot: amplitude curvature and the kinetic term]
    2 * dR * dφ + a * ddφ      [the second slot: the current's divergence, undivided]

    So: the one point that, multiplied into the phase point, carries the wave's second
    derivative. Its first slot carries `σ` and its second does not. -/
abbrev ringCurv (σ a dR ddR dφ ddφ : R) : RingCell R :=
  (ddR + σ * a * dφ ^ 2, 2 * dR * dφ + a * ddφ)

/-- The finite jet: the wave's second derivative assembled from its coefficient data.

    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (C S : R)                  [the phase point's two slots]
    (a dR ddR dφ ddφ : R)      [the amplitude's value, slope and curvature, and the phase's
                                slope and curvature]
    RingCell R                 [the point: two ring elements written side by side]
    ddR * C + 2 * dR * (σ * S * dφ) + a * (σ * C * dφ ^ 2 + σ * S * ddφ)
                               [the first slot: the amplitude's second difference carried by
                                the first phase slot, plus the cross and phase-curvature terms]
    ddR * S + 2 * dR * (C * dφ) + a * (σ * S * dφ ^ 2 + C * ddφ)
                               [the second slot: the same, carried by the second phase slot]

    So: the point the analytic realization identifies with the wave's second derivative, once
    the phase point moves as `(C, S)` with `C' = σ·S·dφ` and `S' = C·dφ`. It is a polynomial in
    the coefficient data, so it lives in every commutative ring. -/
abbrev ringJet (σ C S a dR ddR dφ ddφ : R) : RingCell R :=
  (ddR * C + 2 * dR * (σ * S * dφ) + a * (σ * C * dφ ^ 2 + σ * S * ddφ),
   ddR * S + 2 * dR * (C * dφ) + a * (σ * S * dφ ^ 2 + C * ddφ))

/-- The phase point multiplied by the curvature point is the finite jet.

    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (C S : R)                  [the phase point's two slots]
    (a dR ddR dφ ddφ : R)      [the amplitude's value, slope and curvature, and the phase's
                                slope and curvature]
    ringMul σ (C, S) (ringCurv σ a dR ddR dφ ddφ)
                               [the phase point times the curvature point]
    = ringJet σ C S a dR ddR dφ ddφ
                               [is the assembled second derivative]
    `ringMul`                   [multiply two points together, the σ way]
    `ringCurv`                  [the point carrying the wave's second-order data]
    `ringJet`                   [the second derivative, assembled from the coefficients]

    So: the polynomial content of the wave's second-derivative factorization. The analytic
    theorem is that the second derivative of `a · (C, S)` is this jet, which is where the slot
    derivatives `C' = σ·S·dφ` and `S' = C·dφ` enter. -/
theorem ringMul_ringCurv (σ C S a dR ddR dφ ddφ : R) :
    ringMul σ (C, S) (ringCurv σ a dR ddR dφ ddφ) = ringJet σ C S a dR ddR dφ ddφ := by
  simp only [ringJet, ringMul, Prod.mk.injEq]
  constructor <;> ring

/-- The second slot of the curvature point, carried by the amplitude.

    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (a dR ddR dφ ddφ : R)      [the amplitude's value, slope and curvature, and the phase's
                                slope and curvature]
    a * (ringCurv σ a dR ddR dφ ddφ).2
                               [the amplitude times the curvature point's second slot]
    = (dR * a + a * dR) * dφ + a * a * ddφ
                               [is the product rule's expansion of the density flux]
    `ringCurv`                  [the point carrying the wave's second-order data]

    So: the polynomial half of `Physics.continuity`. The density is `a·a` and the flux is
    `a·a·dφ`, so the right-hand side is the derivative of the flux with the two product-rule
    terms written out. The derivative itself is analytic and stays in `Physics.Madelung`. -/
theorem ringCurv_snd_flux (σ a dR ddR dφ ddφ : R) :
    a * (ringCurv σ a dR ddR dφ ddφ).2 = (dR * a + a * dR) * dφ + a * a * ddφ := by
  ring

/-- The symmetric four-sample combination of the norm form is its Hessian trace.

    (σ u v h : R)              [the parameter, a place's two slots, and a step]
    ringNormForm σ (u + h, v)  [the form sampled one step along the first slot]
    + ringNormForm σ (u - h, v)  [and one step the other way]
    + ringNormForm σ (u, v + h)  [and one step along the second slot]
    + ringNormForm σ (u, v - h)  [and one step the other way]
    - 4 * ringNormForm σ (u, v)  [minus four times the centre]
    = 2 * (1 - σ) * h ^ 2      [is twice one minus the parameter, times the step squared]

    So: the four-sample identity, exact at every step `h`, with no limit taken. The recovered
    value `2·(1−σ)` is the trace of the form's Hessian: the two second differences in the first
    slot give `2`, the two in the second give `−2σ`, and the cross and first-order terms cancel
    by the symmetry of the four samples. This is `Physics.stencil4_normForm` over any
    commutative ring, and it holds there because it is a polynomial identity with whole-number
    coefficients. -/
theorem ringStencil4_normForm (σ u v h : R) :
    ringNormForm σ (u + h, v) + ringNormForm σ (u - h, v)
      + ringNormForm σ (u, v + h) + ringNormForm σ (u, v - h)
        - 4 * ringNormForm σ (u, v) = 2 * (1 - σ) * h ^ 2 := by
  simp only [ringNormForm]
  ring

/-- An element squaring to minus one flips the sign of the parameter in the norm form.

    {i : R}                    [the element that squares to minus one]
    (hi : i ^ 2 = -1)          [the squaring hypothesis that makes it such an element]
    (σ : R)                    [the parameter: what the adjoined direction squares to]
    (z : RingCell R)           [the point whose second slot is turned by i]
    ringNormForm σ (z.1, z.2 * i)
                               [the size with the second slot turned by i]
    = ringNormForm (-σ) z      [is the size at the negated parameter]

    So: putting an `i` on the second slot and negating `σ` are the same operation, and the
    identity is polynomial in `i` and `σ`. `Physics.wick_flips_the_sign` is its specialisation
    at `ℂ` and `Complex.I`. -/
theorem ringNormForm_wick {i : R} (hi : i ^ 2 = -1) (σ : R) (z : RingCell R) :
    ringNormForm σ (z.1, z.2 * i) = ringNormForm (-σ) z := by
  simp only [ringNormForm]
  rw [mul_pow, hi]
  ring

end Physics
end Apeiron
