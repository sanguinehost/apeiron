import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier.Derived.CirculationAnyRing

/-!
# Basis-derived carrier circulation

@name Carrier Circulation Accounting
@grade proved

A two-slot trichotomy cell can supply the two oriented forward-face values leaving one lattice
anchor. This file is the real specialisation of `Apeiron.Physics.CarrierCirculationAnyRing`: every
name below is an alias of the ring-generic declaration at `R = ℝ`, and every theorem below is the
ring-generic theorem read at the real ring. No proof is restated and no `ring` call is repeated.

The first basis coordinate is the positive x-face value and the second is the positive y-face
value. The mapping is not forced by `normForm`. It is admissible because both face values are read
from one `RingCell`, and the readback theorem reconstructs that cell without a third field.

Walking counterclockwise around one plaquette gives bottom x plus right y minus top x minus left y.
A field made from the forward differences of one scalar phase has zero circulation because its four
increments telescope. The basis field `rotationalWitness` instead gives circulation exactly two, so
two basis coordinates can carry nonzero kinematic circulation even though one scalar gradient
cannot.

Over a finite rectangle, interior edges cancel and only the oriented outer boundary remains. This
is a discrete Stokes accounting theorem. It supplies no time law, viscosity, pressure, forcing, or
vortical material evolution.

## Root

`BasisField2` is `RingBasisField ℝ`. `basisFaces_readback` is `ringBasisFaces_readback` at `ℝ`, and
`rectangle_discrete_stokes` is `ringRectangle_discrete_stokes` at `ℝ`. No circulation scalar is
accepted as state: `plaquetteCirculation` is always computed from the two basis coordinates.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

/-- A trichotomy cell at every anchor of a two-dimensional counting lattice, over the reals.

    BasisField2                    [the name for one cell at every lattice anchor]
    ℕ                              [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                              [the real numbers: every number on the number line]
    →                              [takes the left and gives the right]
    RingBasisField ℝ               [the ring-generic cell field read at the real ring]

    The two cell coordinates are the sole face-value authority. -/
abbrev BasisField2 : Type := RingBasisField ℝ

/-- The positive x-face value leaving an anchor, read from the first basis coordinate.

    (field : BasisField2)          [the trichotomy-cell field]
    (i j : ℕ)                      [the anchor coordinates]
    ℝ                              [the real numbers: every number on the number line]
    ℕ                              [the counting numbers 0, 1, 2, 3, and so on]
    →                              [takes the left and gives the right]
    `BasisField2`                  [one trichotomy cell at every lattice anchor]
    `ringBasisFaceX`               [the ring-generic first-coordinate x-face read]

    This is the real reading of the explicit face adaptor, not a consequence of `normForm`. -/
abbrev basisFaceX : BasisField2 → ℕ → ℕ → ℝ := ringBasisFaceX

/-- The positive y-face value leaving an anchor, read from the second basis coordinate.

    (field : BasisField2)          [the trichotomy-cell field]
    (i j : ℕ)                      [the anchor coordinates]
    ℝ                              [the real numbers: every number on the number line]
    ℕ                              [the counting numbers 0, 1, 2, 3, and so on]
    →                              [takes the left and gives the right]
    `BasisField2`                  [one trichotomy cell at every lattice anchor]
    `ringBasisFaceY`               [the ring-generic second-coordinate y-face read]

    This is the other half of the same explicit face adaptor. -/
abbrev basisFaceY : BasisField2 → ℕ → ℕ → ℝ := ringBasisFaceY

/-- The two oriented face reads reconstruct the source cell exactly.

    (field : BasisField2)          [the trichotomy-cell field]
    (i j : ℕ)                      [the anchor coordinates]
    (basisFaceX field i j, basisFaceY field i j) = field i j
                                     [the two reads are exactly the source basis coordinates]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    `BasisField2`                    [one trichotomy cell at every lattice anchor]
    `basisFaceX`                     [the first-coordinate x-face read]
    `basisFaceY`                     [the second-coordinate y-face read]

    This is `ringBasisFaces_readback` read at the real ring. -/
theorem basisFaces_readback (field : BasisField2) (i j : ℕ) :
    (basisFaceX field i j, basisFaceY field i j) = field i j := by
  exact ringBasisFaces_readback field i j

/-- The counterclockwise circulation computed around one plaquette.

    (field : BasisField2)          [the trichotomy-cell field]
    (i j : ℕ)                      [the bottom-left anchor coordinates]
    ℝ                              [the real numbers: every number on the number line]
    ℕ                              [the counting numbers 0, 1, 2, 3, and so on]
    →                              [takes the left and gives the right]
    `BasisField2`                  [one trichotomy cell at every lattice anchor]
    `basisFaceX`                   [the first-coordinate x-face read]
    `basisFaceY`                   [the second-coordinate y-face read]
    `ringPlaquetteCirculation`     [the ring-generic counterclockwise sum]

    Bottom and right follow the walk; top and left oppose it. -/
abbrev plaquetteCirculation : BasisField2 → ℕ → ℕ → ℝ := ringPlaquetteCirculation

/-- The basis field obtained from the two forward differences of one scalar phase.

    (phase : ℕ → ℕ → ℝ)            [one scalar value at every lattice anchor]
    BasisField2                    [one trichotomy cell at every lattice anchor]
    ℕ                              [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                              [the real numbers: every number on the number line]
    →                              [takes the left and gives the right]
    `BasisField2`                  [one trichotomy cell at every lattice anchor]
    `ringScalarGradientBasis`      [the ring-generic two forward differences]

    The first coordinate is the x increment and the second is the y increment. -/
abbrev scalarGradientBasis : (ℕ → ℕ → ℝ) → BasisField2 := ringScalarGradientBasis

/-- Every plaquette circulation of a scalar forward gradient is zero.

    (phase : ℕ → ℕ → ℝ)            [one scalar value at every lattice anchor]
    (i j : ℕ)                      [the bottom-left anchor coordinates]
    plaquetteCirculation (scalarGradientBasis phase) i j = 0
                                     [the four oriented increments cancel]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                                [the real numbers: every number on the number line]
    →                                [takes the left and gives the right]
    `plaquetteCirculation`           [the counterclockwise sum around one plaquette]
    `scalarGradientBasis`            [the two forward differences of one scalar phase]

    This is `ringScalarGradient_plaquetteCirculation_zero` read at the real ring. -/
theorem scalarGradient_plaquetteCirculation_zero (phase : ℕ → ℕ → ℝ) (i j : ℕ) :
    plaquetteCirculation (scalarGradientBasis phase) i j = 0 := by
  exact ringScalarGradient_plaquetteCirculation_zero phase i j

/-- A basis field whose first face value is minus y and whose second is x.

    BasisField2                    [one trichotomy cell at every lattice anchor]
    `BasisField2`                  [one trichotomy cell at every lattice anchor]
    `ringRotationalWitness`        [the ring-generic rotational witness read at the reals]

    Both oriented values are supplied by the two coordinates of `RingCell`. -/
abbrev rotationalWitness : BasisField2 := ringRotationalWitness (R := ℝ)

/-- The rotational basis witness has circulation exactly two on every plaquette.

    (i j : ℕ)                      [the bottom-left anchor coordinates]
    plaquetteCirculation rotationalWitness i j = 2
                                     [the computed counterclockwise sum is two]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    `plaquetteCirculation`           [the counterclockwise sum around one plaquette]
    `rotationalWitness`              [the basis field with face values minus y and x]

    This is `ringRotationalWitness_plaquetteCirculation` read at the real ring. The witness itself
    is ring-generic; only this corollary names the reals. -/
theorem rotationalWitness_plaquetteCirculation (i j : ℕ) :
    plaquetteCirculation rotationalWitness i j = 2 := by
  exact ringRotationalWitness_plaquetteCirculation (R := ℝ) i j

/-- The counterclockwise sum on the oriented outer boundary of a finite rectangle.

    (field : BasisField2)          [the trichotomy-cell field]
    (width height : ℕ)             [the rectangle side lengths]
    ℝ                              [the real numbers: every number on the number line]
    ℕ                              [the counting numbers 0, 1, 2, 3, and so on]
    →                              [takes the left and gives the right]
    `BasisField2`                  [one trichotomy cell at every lattice anchor]
    `ringRectangleBoundaryCirculation` [the ring-generic outer-boundary sum]

    Bottom and right are positive; top and left are negative. -/
abbrev rectangleBoundaryCirculation : BasisField2 → ℕ → ℕ → ℝ :=
  ringRectangleBoundaryCirculation

/-- The sum of plaquette circulations equals the oriented rectangle boundary sum.

    (field : BasisField2)          [the trichotomy-cell field]
    (width height : ℕ)             [the rectangle side lengths]
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      plaquetteCirculation field i k) = rectangleBoundaryCirculation field width height
                                     [interior edges cancel and the outer walk remains]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                                [the real numbers: every number on the number line]
    ∈                                [is one of the things in]
    `BasisField2`                    [one trichotomy cell at every lattice anchor]
    `Finset.range`                   [the counting positions below a side's endpoint]
    `plaquetteCirculation`           [the counterclockwise sum around one plaquette]
    `rectangleBoundaryCirculation`   [the counterclockwise sum on the outer boundary]

    This is `ringRectangle_discrete_stokes` read at the real ring. -/
theorem rectangle_discrete_stokes (field : BasisField2) (width height : ℕ) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      plaquetteCirculation field i k) = rectangleBoundaryCirculation field width height := by
  exact ringRectangle_discrete_stokes field width height

/-- Identifying opposite boundary face values makes total circulation zero.

    (field : BasisField2)          [the trichotomy-cell field]
    (width height : ℕ)             [the rectangle side lengths]
    (hX : ∀ i, basisFaceX field i height = basisFaceX field i 0)
                                     [top and bottom x-face values are identified]
    (hY : ∀ k, basisFaceY field width k = basisFaceY field 0 k)
                                     [right and left y-face values are identified]
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      plaquetteCirculation field i k) = 0
                                     [the total computed circulation vanishes]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                                [the real numbers: every number on the number line]
    ∀                                [for every single one]
    ∈                                [is one of the things in]
    `BasisField2`                    [one trichotomy cell at every lattice anchor]
    `Finset.range`                   [the counting positions below a side's endpoint]
    `basisFaceX`                     [the first-coordinate x-face read]
    `basisFaceY`                     [the second-coordinate y-face read]
    `plaquetteCirculation`           [the counterclockwise sum around one plaquette]

    This is `ringClosed_rectangle_circulation_cancels` read at the real ring. -/
theorem closed_rectangle_circulation_cancels (field : BasisField2) (width height : ℕ)
    (hX : ∀ i, basisFaceX field i height = basisFaceX field i 0)
    (hY : ∀ k, basisFaceY field width k = basisFaceY field 0 k) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      plaquetteCirculation field i k) = 0 := by
  exact ringClosed_rectangle_circulation_cancels field width height hX hY

end Physics
end Apeiron
