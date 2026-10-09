import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing

/-!
# The carrier readout over any commutative ring

@name Product Carrier Readout Over Any Commutative Ring
@cite kocik-1999
@cite euclid-elements-ii-5
@grade proved

`Apeiron.Physics.Carrier` reads a Born density off `Trichotomy.normForm` of a phase cell, where
the phase cell is the algebra's product `ringMul σ (a, 0) u`, the amplitude `a` stretching the
direction `u`. On a point whose second slot is zero that product agrees with the library's
slot-by-slot scalar stretch `a • u` (`ringMul_mk_zero_eq_smul`); the agreement is propositional
rather than definitional, and the algebra's own product stays authoritative as the one
construction its multiplication laws are stated over. This file installs the readout once over an
arbitrary commutative ring, so the real and machine-word readouts are its specialisations rather
than restatements.

`pointDensity σ a u` is `ringNormForm` of `ringMul σ (a, 0) u`: the size of the phase cell the
amplitude `a` stretches the direction `u` into. `pointFlux σ a u p n` is that density times the
forward phase difference `n - p`. `pointDensity_eq` factors the size with
`AnyRing.ringNormForm_mul`, and `pointFlux_eq` reads a unit direction off that factor.

No order appears here. The nonnegativity and zero-iff theorems for the real readout are instance
results for `ℝ`, stated in `Apeiron.Physics.Carrier`, and they do not transfer to a ring such as
`ZMod (2 ^ 64)` where no order exists.

## Root

| from | here |
|---|---|
| `AnyRing.ringMul`, `AnyRing.ringNormForm` | `pointDensity`, `pointFlux` |
| `AnyRing.ringNormForm_mul` | `pointDensity_eq`, `pointFlux_eq` |
| `AnyRing.ringOne` | `pointDensity_ringOne`, `pointFlux_ringOne` |

`pointDensity σ a u` is `ringNormForm σ (ringMul σ (a, 0) u)`, the size of the phase cell the
amplitude `a` stretches the direction `u` into, and `pointFlux` multiplies it by the forward
phase difference. `pointDensity_eq` factors it through `ringNormForm_mul`, and
`ringMul_mk_zero_eq_smul` identifies the algebra's product with the slot-by-slot stretch. What
does not transfer is the order: nonnegativity, the zero-iff reading and the CFL bound are `ℝ`
results in `Apeiron.Physics.Carrier`.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- The algebra's product of a stretched unit and a point is the slot-by-slot scalar stretch.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (u : RingCell R)                [the phase direction: a point the amplitude stretches]
    R                               [the commutative ring the point is built over]
    ringMul σ (a, 0) u = a • u      [the product equals the slot-by-slot stretch]
    •                               [stretch by, a number times a point]

    The second slot of `(a, 0)` is zero, so the dial's term in the product vanishes and both
    sides agree. The equality is propositional, not definitional: `rfl` does not close it, while
    `ext` followed by `simp` does. -/
theorem ringMul_mk_zero_eq_smul (σ a : R) (u : RingCell R) :
    ringMul σ (a, 0) u = a • u := by
  ext <;> simp

/-- The density a phase cell carries, over any commutative ring.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (u : RingCell R)                [the phase direction: a point the amplitude stretches]
    R                               [the commutative ring the point is built over]
    ringNormForm σ (ringMul σ (a, 0) u)
                                    [the size of the phase cell the amplitude stretches the
                                     direction into]

    The stretch is the algebra's product `ringMul σ (a, 0) u`. It agrees with the library's
    slot-by-slot scalar product `a • u` by `ringMul_mk_zero_eq_smul`, but not definitionally, so
    the product is written out as the one construction the algebra's multiplication laws use. -/
def pointDensity (σ : R) (a : R) (u : RingCell R) : R :=
  ringNormForm σ (ringMul σ (a, 0) u)

/-- The phase flux a cell carries across one forward step, over any commutative ring.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (u : RingCell R)                [the phase direction: a point the amplitude stretches]
    (p n : R)                       [the phase at this sample and at the next one]
    R                               [the commutative ring the point is built over]
    pointDensity σ a u * (n - p)    [the density times the forward phase difference]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    The single phase fixes the flux's direction; no independent momentum is supplied. -/
def pointFlux (σ : R) (a : R) (u : RingCell R) (p n : R) : R :=
  pointDensity σ a u * (n - p)

/-- The density factors into the amplitude's size and the direction's size.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (u : RingCell R)                [the phase direction: a point the amplitude stretches]
    pointDensity σ a u = ringNormForm σ (a, 0) * ringNormForm σ u
                                    [the size of the stretched cell equals the amplitude's size
                                     times the direction's size]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `ringNormForm_mul`              [the size of a product is the product of the sizes]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    Multiplicativity of the norm form is the whole proof; no polynomial expansion is redone. -/
theorem pointDensity_eq (σ : R) (a : R) (u : RingCell R) :
    pointDensity σ a u = ringNormForm σ (a, 0) * ringNormForm σ u := by
  unfold pointDensity
  rw [ringNormForm_mul]

/-- At a unit direction, the flux is the amplitude's size times the phase difference.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (u : RingCell R)                [the phase direction: a point the amplitude stretches]
    (p n : R)                       [the phase at this sample and at the next one]
    (hu : ringNormForm σ u = 1)     [the phase direction has size one]
    pointFlux σ a u p n = ringNormForm σ (a, 0) * (n - p)
                                    [the flux equals the amplitude's size times the phase
                                     difference]
    `pointFlux`                     [the first-direction flux: amount times phase difference]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    The unit direction drops out of `pointDensity_eq`, so no second polynomial expansion is
    needed. -/
theorem pointFlux_eq (σ : R) (a : R) (u : RingCell R) (p n : R)
    (hu : ringNormForm σ u = 1) :
    pointFlux σ a u p n = ringNormForm σ (a, 0) * (n - p) := by
  unfold pointFlux
  rw [pointDensity_eq, hu, mul_one]

/-- At the unit direction the density is the amplitude's square, at any dial.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    R                               [the commutative ring the point is built over]
    pointDensity σ a ringOne = a ^ 2
                                    [the unit-direction density is the amplitude squared]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `ringOne`                   [the point (1, 0), which changes nothing]

    The unit direction has no second slot, so the dial's term vanishes and the size is the
    amplitude's own square. -/
theorem pointDensity_ringOne (σ : R) (a : R) : pointDensity σ a ringOne = a ^ 2 := by
  unfold pointDensity
  simp

/-- The unit-direction density does not consult the dial.

    (σ σ' : R)                      [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    R                               [the commutative ring the point is built over]
    pointDensity σ a ringOne = pointDensity σ' a ringOne
                                    [the unit-direction density is the same at either dial]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `ringOne`                   [the point (1, 0), which changes nothing]

    Both sides are the amplitude's square, so the dial is immaterial at the unit direction. -/
theorem pointDensity_ringOne_independent (σ σ' : R) (a : R) :
    pointDensity σ a ringOne = pointDensity σ' a ringOne := by
  rw [pointDensity_ringOne, pointDensity_ringOne]

/-- At the unit direction the flux is the amplitude's square times the phase difference.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (p n : R)                       [the phase at this sample and at the next one]
    R                               [the commutative ring the point is built over]
    pointFlux σ a ringOne p n = a ^ 2 * (n - p)
                                    [the unit-direction flux is the amplitude squared times the
                                     phase difference]
    `pointFlux`                     [the first-direction flux: amount times phase difference]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `ringOne`                   [the point (1, 0), which changes nothing]

    The unit direction drops out of `pointDensity_ringOne`, so no second expansion is needed. -/
theorem pointFlux_ringOne (σ : R) (a : R) (p n : R) :
    pointFlux σ a ringOne p n = a ^ 2 * (n - p) := by
  unfold pointFlux
  rw [pointDensity_ringOne]

/-- The unit-direction flux does not consult the dial.

    (σ σ' : R)                      [the parameter: what the adjoined direction squares to]
    (a : R)                         [the amplitude: the ring element stretching the direction]
    (p n : R)                       [the phase at this sample and at the next one]
    R                               [the commutative ring the point is built over]
    pointFlux σ a ringOne p n = pointFlux σ' a ringOne p n
                                    [the unit-direction flux is the same at either dial]
    `pointFlux`                     [the first-direction flux: amount times phase difference]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `ringOne`                   [the point (1, 0), which changes nothing]

    Both sides are the amplitude's square times the same phase difference, so the dial is
    immaterial at the unit direction. -/
theorem pointFlux_ringOne_independent (σ σ' : R) (a : R) (p n : R) :
    pointFlux σ a ringOne p n = pointFlux σ' a ringOne p n := by
  rw [pointFlux_ringOne, pointFlux_ringOne]

end Physics
end Apeiron
