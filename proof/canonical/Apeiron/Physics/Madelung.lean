--//@ file: Apeiron/Physics/Madelung.lean
--//@ does: Madelung quantum hydrodynamics and conserved currents across sigma-trichotomy
--//@ ring: R0
--//@ onto: observation:physics:madelung_hydrodynamics
--//@ cite: Madelung (1927), Schrödinger (1926), Bohm (1952)
--//@ grade: proved

import Apeiron.Trichotomy.Flow
import Apeiron.Physics.WaveAnyRing

/-!
# One wave, split into two channels, at every value of σ

@name Madelung
@cite kocik-1999
@cite herranz-ortega-santander-2000
@cite schrodinger-1926
@cite madelung-1927
@cite bohm-1952
@grade proved

Write a wave as a size times a flow:

```
   ψ(y)  =  R(y) · flow σ (φ(y))
```

`R` is the amplitude and `φ` the phase, and `Trichotomy.flow` is the one-parameter subgroup of
the algebra, so this is the polar form written once for all three faces rather than once for the
complex numbers. Differentiate it twice. The generator's square is the only place `σ`
enters, and it enters exactly once, so the second derivative is the flow multiplied by one
curvature point:

```
   ψ''  =  flow σ (φ)  ·  (R'' + σ·R·(φ')² ,  2R'φ' + R·φ'')
```

`secondDeriv_waveFst` and `secondDeriv_waveSnd` are that, slot by slot, from Mathlib's calculus.
The two slots of the curvature point are the two Madelung equations.

**First slot, divided by the amplitude.** `R''/R + σ·(φ')²`. The first term is the quantum
potential, the ratio of the amplitude's curvature to the amplitude. The second is the kinetic
term, and its sign is the sign of `σ`. Below zero it enters negatively, which is the
non-relativistic quantum Hamilton-Jacobi equation; above zero it enters positively, which is the
relativistic one with the wave operator in place of the Laplacian. `hamilton_jacobi` is one
theorem covering both, and the Wick rotation between them is `Apeiron/Physics/Wick.lean`.

**Second slot, multiplied by the amplitude.** `(R²φ')'`, the divergence of the current carried by
the phase gradient. `continuity` is that identity, and it holds at every value of `σ` with
the same proof. The conserved current is the same object on all three faces; only the kinetic
sign moves.

## What is new here and what is not

Kocik, *Duplex numbers, diffusion systems, and generalized quantum mechanics*, Int. J. Theor.
Phys. 38 (1999) 2221, arXiv:1201.3069, proves this split for the complex and the split-complex
face and states the continuity equation's sameness across them, his equations (2.2), (2.4),
(2.10) and (3.2). His operator carries the generator's inverse, so the face where the generator
squares to zero is outside his statement and is new here. What is also new is that one
statement covers all three faces with one proof, because `Trichotomy.flow` is one function of the
`σ` rather than a case split.

## Root

Instantiates `Trichotomy.flow` and its two slot derivatives, `Trichotomy.hasDerivAt_flow_fst`
(`C' = σ·S`) and `Trichotomy.hasDerivAt_flow_snd` (`S' = C`), from `Apeiron/Trichotomy/Flow.lean`,
which in turn rest on `Trichotomy.mul` and `Trichotomy.normForm` in
`Apeiron/Trichotomy/Basic.lean`. The edges proved here from them:

* `secondDeriv_waveFst` and `secondDeriv_waveSnd` are the two slot derivatives applied twice. The
  polynomial half is `Physics.ringMul_ringCurv` in `Apeiron/Physics/WaveAnyRing.lean`, which
  factors the second derivative through the generic curvature point; the derivative half is
  Mathlib's calculus applied to the two flow slot derivatives above.
* `hamilton_jacobi` carries `σ`: it is a statement at each value, and the two readings are
  the elliptic face at `σ = −1` and the hyperbolic face at `σ = 1`. Its polynomial half is the
  first slot of `Physics.ringCurv`, and the division by the amplitude is why it stays over `ℝ`.
* `continuity` is uniform in `σ`: the same equation at every setting, proved once. Its polynomial
  half is `Physics.ringCurv_snd_flux`, the product rule's expansion of the density flux, and the
  derivative that closes it is analytic.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- The first slot of a wave: a size times the flow's first slot.

    σ                          [the parameter]
    ℝ                          [the real numbers: every number on the number line]
    →                          [takes the left and gives the right]
    R y * (flow σ (φ y)).1     [the size at that place, times the flow's first slot]
    flow                       [the size-one flow of the algebra]

    So: the polar form's first coordinate, written for whichever face σ picks out. -/
noncomputable def waveFst (σ : ℝ) (R φ : ℝ → ℝ) (y : ℝ) : ℝ := R y * (flow σ (φ y)).1

/-- The second slot of a wave: a size times the flow's second slot.

    σ                          [the parameter]
    ℝ                          [the real numbers: every number on the number line]
    →                          [takes the left and gives the right]
    R y * (flow σ (φ y)).2     [the size at that place, times the flow's second slot]
    flow                       [the size-one flow of the algebra]

    So: the polar form's second coordinate. -/
noncomputable def waveSnd (σ : ℝ) (R φ : ℝ → ℝ) (y : ℝ) : ℝ := R y * (flow σ (φ y)).2

/-- The curvature point of a wave, read off the generic curvature point.

    σ                                   [the parameter]
    ℝ                                   [the real numbers: every number on the number line]
    →                                   [takes the left and gives the right]
    curv                                [the wave's curvature point]
    Cell                                [a point: two real numbers written side by side]
    ringCurv                            [the generic curvature point over any commutative ring]

    So: the wave's curvature point is `Physics.ringCurv` at the amplitude's and the phase's
    second-order data. Its first slot carries σ and its second does not. The generic point and
    its jet factorization are in `Apeiron/Physics/WaveAnyRing.lean`; this is its real
    specialisation, and the trigonometry and the derivative laws stay here. -/
noncomputable def curv (σ : ℝ) (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) : Cell :=
  ringCurv σ (R y) (dR y) (ddR y) (dφ y) (ddφ y)

/-- The first slot's rate of change.

    σ                       [the parameter]
    ℝ                       [the real numbers: every number on the number line]
    →                       [takes the left and gives the right]
    ∀                       [for every single one]
    HasDerivAt              [is changing at that rate at that place]
    waveFst                 [the wave's first slot]
    flow                    [the size-one flow of the algebra]

    So: the product rule, with the flow's first slot changing at σ times its second. -/
theorem hasDerivAt_waveFst (σ : ℝ) (R φ dR dφ : ℝ → ℝ)
    (hR : ∀ u, HasDerivAt R (dR u) u) (hφ : ∀ u, HasDerivAt φ (dφ u) u) (y : ℝ) :
    HasDerivAt (waveFst σ R φ)
      (dR y * (flow σ (φ y)).1 + R y * (σ * (flow σ (φ y)).2 * dφ y)) y :=
  (hR y).mul ((hasDerivAt_flow_fst σ (φ y)).comp y (hφ y))

/-- The second slot's rate of change.

    σ                       [the parameter]
    ℝ                       [the real numbers: every number on the number line]
    →                       [takes the left and gives the right]
    ∀                       [for every single one]
    HasDerivAt              [is changing at that rate at that place]
    waveSnd                 [the wave's second slot]
    flow                    [the size-one flow of the algebra]

    So: the product rule, with the flow's second slot changing at its first. -/
theorem hasDerivAt_waveSnd (σ : ℝ) (R φ dR dφ : ℝ → ℝ)
    (hR : ∀ u, HasDerivAt R (dR u) u) (hφ : ∀ u, HasDerivAt φ (dφ u) u) (y : ℝ) :
    HasDerivAt (waveSnd σ R φ)
      (dR y * (flow σ (φ y)).2 + R y * ((flow σ (φ y)).1 * dφ y)) y :=
  (hR y).mul ((hasDerivAt_flow_snd σ (φ y)).comp y (hφ y))

/-- **The wave's second derivative is the flow times one curvature point, first slot.**

    σ                       [the parameter]
    ℝ                       [the real numbers: every number on the number line]
    →                       [takes the left and gives the right]
    ∀                       [for every single one]
    HasDerivAt              [is changing at that rate at that place]
    deriv                   [how fast the thing is changing, at that place]
    waveFst                 [the wave's first slot]
    mul                     [multiply two points together, the σ way]
    flow                    [the size-one flow of the algebra]
    curv                    [the wave's curvature point]

    So: differentiating twice returns the flow multiplied by one point, and this is that point's
    first slot. Nothing beyond the product rule and the flow's own two slot derivatives. -/
theorem secondDeriv_waveFst (σ : ℝ) (R φ dR dφ ddR ddφ : ℝ → ℝ)
    (hR : ∀ u, HasDerivAt R (dR u) u) (hdR : ∀ u, HasDerivAt dR (ddR u) u)
    (hφ : ∀ u, HasDerivAt φ (dφ u) u) (hdφ : ∀ u, HasDerivAt dφ (ddφ u) u) (y : ℝ) :
    deriv (deriv (waveFst σ R φ)) y
      = (mul σ (flow σ (φ y)) (curv σ R dR ddR dφ ddφ y)).1 := by
  have hd : deriv (waveFst σ R φ)
      = fun u => dR u * (flow σ (φ u)).1 + R u * (σ * (flow σ (φ u)).2 * dφ u) :=
    funext fun u => (hasDerivAt_waveFst σ R φ dR dφ hR hφ u).deriv
  rw [hd]
  have hC : HasDerivAt (fun u => (flow σ (φ u)).1) (σ * (flow σ (φ y)).2 * dφ y) y :=
    (hasDerivAt_flow_fst σ (φ y)).comp y (hφ y)
  have hS : HasDerivAt (fun u => (flow σ (φ u)).2) ((flow σ (φ y)).1 * dφ y) y :=
    (hasDerivAt_flow_snd σ (φ y)).comp y (hφ y)
  have hsum := ((hdR y).mul hC).add ((hR y).mul ((hS.const_mul σ).mul (hdφ y)))
  have hval : ddR y * (flow σ (φ y)).1 + dR y * (σ * (flow σ (φ y)).2 * dφ y)
      + (dR y * (σ * (flow σ (φ y)).2 * dφ y)
        + R y * (σ * ((flow σ (φ y)).1 * dφ y) * dφ y + σ * (flow σ (φ y)).2 * ddφ y))
      = (mul σ (flow σ (φ y)) (curv σ R dR ddR dφ ddφ y)).1 := by
    rw [show (mul σ (flow σ (φ y)) (curv σ R dR ddR dφ ddφ y)).1
        = (ringJet σ (flow σ (φ y)).1 (flow σ (φ y)).2
            (R y) (dR y) (ddR y) (dφ y) (ddφ y)).1 from
      congrArg Prod.fst (ringMul_ringCurv σ (flow σ (φ y)).1 (flow σ (φ y)).2
        (R y) (dR y) (ddR y) (dφ y) (ddφ y))]
    ring
  rw [← hval]
  exact hsum.deriv

/-- **The wave's second derivative is the flow times one curvature point, second slot.**

    σ                       [the parameter]
    ℝ                       [the real numbers: every number on the number line]
    →                       [takes the left and gives the right]
    ∀                       [for every single one]
    HasDerivAt              [is changing at that rate at that place]
    deriv                   [how fast the thing is changing, at that place]
    waveSnd                 [the wave's second slot]
    mul                     [multiply two points together, the σ way]
    flow                    [the size-one flow of the algebra]
    curv                    [the wave's curvature point]

    So: the same, on the second slot. One curvature point serves both. -/
theorem secondDeriv_waveSnd (σ : ℝ) (R φ dR dφ ddR ddφ : ℝ → ℝ)
    (hR : ∀ u, HasDerivAt R (dR u) u) (hdR : ∀ u, HasDerivAt dR (ddR u) u)
    (hφ : ∀ u, HasDerivAt φ (dφ u) u) (hdφ : ∀ u, HasDerivAt dφ (ddφ u) u) (y : ℝ) :
    deriv (deriv (waveSnd σ R φ)) y
      = (mul σ (flow σ (φ y)) (curv σ R dR ddR dφ ddφ y)).2 := by
  have hd : deriv (waveSnd σ R φ)
      = fun u => dR u * (flow σ (φ u)).2 + R u * ((flow σ (φ u)).1 * dφ u) :=
    funext fun u => (hasDerivAt_waveSnd σ R φ dR dφ hR hφ u).deriv
  rw [hd]
  have hC : HasDerivAt (fun u => (flow σ (φ u)).1) (σ * (flow σ (φ y)).2 * dφ y) y :=
    (hasDerivAt_flow_fst σ (φ y)).comp y (hφ y)
  have hS : HasDerivAt (fun u => (flow σ (φ u)).2) ((flow σ (φ y)).1 * dφ y) y :=
    (hasDerivAt_flow_snd σ (φ y)).comp y (hφ y)
  have hsum := ((hdR y).mul hS).add ((hR y).mul (hC.mul (hdφ y)))
  have hval : ddR y * (flow σ (φ y)).2 + dR y * ((flow σ (φ y)).1 * dφ y)
      + (dR y * ((flow σ (φ y)).1 * dφ y)
        + R y * (σ * (flow σ (φ y)).2 * dφ y * dφ y + (flow σ (φ y)).1 * ddφ y))
      = (mul σ (flow σ (φ y)) (curv σ R dR ddR dφ ddφ y)).2 := by
    rw [show (mul σ (flow σ (φ y)) (curv σ R dR ddR dφ ddφ y)).2
        = (ringJet σ (flow σ (φ y)).1 (flow σ (φ y)).2
            (R y) (dR y) (ddR y) (dφ y) (ddφ y)).2 from
      congrArg Prod.snd (ringMul_ringCurv σ (flow σ (φ y)).1 (flow σ (φ y)).2
        (R y) (dR y) (ddR y) (dφ y) (ddφ y))]
    ring
  rw [← hval]
  exact hsum.deriv

/-- **The first channel is Hamilton-Jacobi, and the sign of σ is the kinetic sign.**

@cite madelung-1927
@cite bohm-1952
@cite hamilton-jacobi-1834

    σ                                [the parameter]
    ℝ                                [the real numbers: every number on the number line]
    →                                [takes the left and gives the right]
    ≠                                [is not equal to]
    curv                             [the wave's curvature point]
    ddR y / R y + σ * dφ y ^ 2       [the amplitude's curvature ratio, plus the kinetic term]

    So: one equation at every setting, and the sign of `σ` decides the kinetic term's sign. Below
    zero the phase gradient enters with a minus, which is the non-relativistic quantum
    Hamilton-Jacobi equation; above zero with a plus, which is its relativistic form. The quantum
    potential `R''/R` is the same term on both faces, so the rotation moves only the kinetic
    sign. -/
theorem hamilton_jacobi (σ : ℝ) (R dR ddR dφ ddφ : ℝ → ℝ) (y : ℝ) (hR : R y ≠ 0) :
    (curv σ R dR ddR dφ ddφ y).1 / R y = ddR y / R y + σ * dφ y ^ 2 := by
  have h : (curv σ R dR ddR dφ ddφ y).1 = ddR y + σ * R y * dφ y ^ 2 := rfl
  rw [h]
  field_simp

/-- **The second channel is the current's divergence, at every value of σ.**

@cite madelung-1927

    σ                       [the parameter]
    ℝ                       [the real numbers: every number on the number line]
    →                       [takes the left and gives the right]
    ∀                       [for every single one]
    HasDerivAt              [is changing at that rate at that place]
    deriv                   [how fast the thing is changing, at that place]
    curv                    [the wave's curvature point]
    R u * R u * dφ u        [the size squared, carried by the phase gradient]

    So: the size squared, carried by the phase gradient, is conserved. σ appears in the
    statement because `curv` carries it, and the equation holds at every setting with one proof:
    the conserved current is the same object on all three faces. -/
theorem continuity (σ : ℝ) (R dR ddR dφ ddφ : ℝ → ℝ)
    (hR : ∀ u, HasDerivAt R (dR u) u) (hdφ : ∀ u, HasDerivAt dφ (ddφ u) u) (y : ℝ) :
    R y * (curv σ R dR ddR dφ ddφ y).2 = deriv (fun u => R u * R u * dφ u) y := by
  have hcur : HasDerivAt (fun u => R u * R u * dφ u)
      ((dR y * R y + R y * dR y) * dφ y + R y * R y * ddφ y) y :=
    ((hR y).mul (hR y)).mul (hdφ y)
  rw [hcur.deriv]
  exact ringCurv_snd_flux σ (R y) (dR y) (ddR y) (dφ y) (ddφ y)

end Physics
end Apeiron
