import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier
import Apeiron.Physics.Madelung

/-!
# The phase cell is the flow

@name Product Carrier Flow
@cite kocik-1999
@grade proved

The carrier readout's phase direction is the algebra's flow: `u = flow σ (φ x y)`, whose size is
one by `Trichotomy.normForm_flow`. Instantiating `density_eq_amplitude_sq` at that direction
gives the Born density of the actual phase cell, `R²`, and the phase flux of the same cell.

This file imports the flow and is therefore on the analysis part of the corpus; the root
`Apeiron/Physics/Carrier.lean` stays on `Trichotomy.Basic` alone.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- The phase cell's direction is the flow, so the density is the amplitude squared.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R φ : PhaseField)              [a real amplitude field and a real phase field]
    (x y : ℝ)                       [two real coordinates]
    density σ R (flow σ (φ x y)) x y = R x y * R x y   [the amount is the amplitude squared]
    ℝ                               [the real numbers: every number on the number line]
    `density`                       [the amount: the size of the phase cell at a place]
    `flow`                          [the size-one points reached by turning, shearing or boosting]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    `normForm_flow` supplies the unit-size hypothesis, so this is the phase-cell reading of
    `density_eq_amplitude_sq`. -/
theorem density_flow (σ : ℝ) (R φ : PhaseField) (x y : ℝ) :
    density σ R (flow σ (φ x y)) x y = R x y * R x y :=
  density_eq_amplitude_sq σ R (flow σ (φ x y)) x y (normForm_flow σ (φ x y))

/-- The phase flux of the flow direction is the amplitude squared times the phase difference.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R φ : PhaseField)              [a real amplitude field and a real phase field]
    (x y : ℝ)                       [two real coordinates]
    fluxFirst σ R (flow σ (φ x y)) φ x y
      = R x y * R x y * (φ (x + 1) y - φ x y)   [the flux is amount times phase difference]
    ℝ                               [the real numbers: every number on the number line]
    `fluxFirst`                     [the first-direction flux: amount times phase slope]
    `density`                       [the amount: the size of the phase cell at a place]
    `flow`                          [the size-one points reached by turning, shearing or boosting]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    The flow instantiation of `fluxFirst_eq`. -/
theorem fluxFirst_flow (σ : ℝ) (R φ : PhaseField) (x y : ℝ) :
    fluxFirst σ R (flow σ (φ x y)) φ x y = R x y * R x y * (φ (x + 1) y - φ x y) :=
  fluxFirst_eq σ R (flow σ (φ x y)) φ x y (normForm_flow σ (φ x y))

end Physics
end Apeiron
