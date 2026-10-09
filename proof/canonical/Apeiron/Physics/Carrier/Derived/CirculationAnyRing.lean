import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing

/-!
# Circulation accounting over any commutative ring

@name Carrier Circulation Accounting Over Any Commutative Ring
@grade proved

A two-slot `RingCell` at every anchor of a two-dimensional counting lattice supplies the two
oriented forward-face values leaving that anchor. This file installs that mapping once over an
arbitrary commutative ring, so the real readout in `Apeiron.Physics.CarrierCirculation` is a
specialisation rather than a restatement.

The first cell coordinate is the positive x-face value and the second is the positive y-face
value. Walking counterclockwise around one plaquette gives bottom x plus right y minus top x
minus left y. A field made from the forward differences of one scalar phase has zero circulation,
because its four increments telescope. The rotational witness instead gives circulation exactly
two, so two cell coordinates can carry nonzero kinematic circulation even though one scalar
gradient cannot.

Over a finite rectangle, interior edges cancel and only the oriented outer boundary remains.
This is a discrete Stokes accounting theorem. It supplies no time law, viscosity, pressure,
forcing, or vortical material evolution.

## Root

`RingBasisField` uses `RingCell` directly. `ringBasisFaces_readback` accounts for both face
reads, and `ringRectangle_discrete_stokes` telescopes one coordinate at a time. No circulation
scalar is accepted as state: `ringPlaquetteCirculation` is always computed from the two cell
coordinates.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- A ring cell at every anchor of a two-dimensional counting lattice.

    RingBasisField R                [the name for one cell at every lattice anchor]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    RingCell R                      [a point over the ring: two ring elements side by side]
    R                               [the commutative ring the point is built over]

    The two cell coordinates are the sole face-value authority. -/
abbrev RingBasisField (R : Type*) [CommRing R] : Type _ := ℕ → ℕ → RingCell R

/-- The positive x-face value leaving an anchor, read from the first cell coordinate.

    (field : RingBasisField R)      [the ring-cell field]
    (i j : ℕ)                       [the anchor coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    `RingBasisField`                [one ring cell at every lattice anchor]

    This is the explicit face adaptor, not a consequence of any norm form. -/
def ringBasisFaceX (field : RingBasisField R) (i j : ℕ) : R := (field i j).1

/-- The positive y-face value leaving an anchor, read from the second cell coordinate.

    (field : RingBasisField R)      [the ring-cell field]
    (i j : ℕ)                       [the anchor coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    `RingBasisField`                [one ring cell at every lattice anchor]

    This is the other half of the same explicit face adaptor. -/
def ringBasisFaceY (field : RingBasisField R) (i j : ℕ) : R := (field i j).2

/-- The two oriented face reads reconstruct the source cell exactly.

    (field : RingBasisField R)      [the ring-cell field]
    (i j : ℕ)                       [the anchor coordinates]
    (ringBasisFaceX field i j, ringBasisFaceY field i j) = field i j
                                     [the two reads are exactly the source cell coordinates]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    `RingBasisField`                 [one ring cell at every lattice anchor]
    `ringBasisFaceX`                 [the first-coordinate x-face read]
    `ringBasisFaceY`                 [the second-coordinate y-face read]

    There is no independently authored vector or circulation value behind the adaptor. -/
theorem ringBasisFaces_readback (field : RingBasisField R) (i j : ℕ) :
    (ringBasisFaceX field i j, ringBasisFaceY field i j) = field i j := by
  exact Prod.eta _

/-- The counterclockwise circulation computed around one plaquette.

    (field : RingBasisField R)      [the ring-cell field]
    (i j : ℕ)                       [the bottom-left anchor coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `RingBasisField`                [one ring cell at every lattice anchor]
    `ringBasisFaceX`                [the first-coordinate x-face read]
    `ringBasisFaceY`                [the second-coordinate y-face read]

    Bottom and right follow the walk; top and left oppose it. -/
def ringPlaquetteCirculation (field : RingBasisField R) (i j : ℕ) : R :=
  ringBasisFaceX field i j + ringBasisFaceY field (i + 1) j
    - ringBasisFaceX field i (j + 1) - ringBasisFaceY field i j

/-- The ring field obtained from the two forward differences of one scalar phase.

    (phase : ℕ → ℕ → R)             [one ring value at every lattice anchor]
    RingBasisField R                [one ring cell at every lattice anchor]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    `RingBasisField`                [one ring cell at every lattice anchor]

    The first coordinate is the x increment and the second is the y increment. -/
def ringScalarGradientBasis (phase : ℕ → ℕ → R) : RingBasisField R :=
  fun i j => (phase (i + 1) j - phase i j, phase i (j + 1) - phase i j)

/-- Every plaquette circulation of a scalar forward gradient is zero.

    (phase : ℕ → ℕ → R)             [one ring value at every lattice anchor]
    (i j : ℕ)                       [the bottom-left anchor coordinates]
    ringPlaquetteCirculation (ringScalarGradientBasis phase) i j = 0
                                     [the four oriented increments cancel]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    →                                [takes the left and gives the right]
    `ringPlaquetteCirculation`       [the counterclockwise sum around one plaquette]
    `ringScalarGradientBasis`        [the two forward differences of one scalar phase]

    Each corner phase occurs once with each sign. -/
theorem ringScalarGradient_plaquetteCirculation_zero (phase : ℕ → ℕ → R) (i j : ℕ) :
    ringPlaquetteCirculation (ringScalarGradientBasis phase) i j = 0 := by
  simp only [ringPlaquetteCirculation, ringScalarGradientBasis, ringBasisFaceX, ringBasisFaceY]
  ring

/-- A ring field whose first face value is minus y and whose second is x.

    RingBasisField R                [one ring cell at every lattice anchor]
    `RingBasisField`                [one ring cell at every lattice anchor]

    Both oriented values are supplied by the two coordinates of `RingCell`. -/
def ringRotationalWitness : RingBasisField R := fun i j => (-(j : R), (i : R))

/-- The rotational ring witness has circulation exactly two on every plaquette.

    (i j : ℕ)                       [the bottom-left anchor coordinates]
    ringPlaquetteCirculation ringRotationalWitness i j = 2
                                     [the computed counterclockwise sum is two]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    `ringPlaquetteCirculation`       [the counterclockwise sum around one plaquette]
    `ringRotationalWitness`          [the ring field with face values minus y and x]

    This is nonzero kinematic authority derived from `RingCell`, not scalar-gradient authority.
    The two natural-number coordinates are read in the ring by its canonical cast. -/
theorem ringRotationalWitness_plaquetteCirculation (i j : ℕ) :
    ringPlaquetteCirculation (ringRotationalWitness (R := R)) i j = 2 := by
  simp only [ringPlaquetteCirculation, ringRotationalWitness, ringBasisFaceX, ringBasisFaceY,
    Nat.cast_add, Nat.cast_one]
  ring

/-- A finite line of forward differences leaves only its two endpoint values.

    (f : ℕ → R)                    [one single-valued ring value along a line]
    (n : ℕ)                        [the number of cells in the line]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    ∈                               [is one of the things in]
    `Finset.range`                  [the counting positions below the endpoint]

    Every intermediate value occurs once positively and once negatively, so it cancels. -/
theorem ringSum_forward_difference (f : ℕ → R) (n : ℕ) :
    (∑ i ∈ Finset.range n, (f (i + 1) - f i)) = f n - f 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      ring

/-- The counterclockwise sum on the oriented outer boundary of a finite rectangle.

    (field : RingBasisField R)      [the ring-cell field]
    (width height : ℕ)              [the rectangle side lengths]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    ∈                               [is one of the things in]
    `RingBasisField`                [one ring cell at every lattice anchor]
    `Finset.range`                  [the counting positions below a side's endpoint]
    `ringBasisFaceX`                [the first-coordinate x-face read]
    `ringBasisFaceY`                [the second-coordinate y-face read]

    Bottom and right are positive; top and left are negative. -/
def ringRectangleBoundaryCirculation (field : RingBasisField R) (width height : ℕ) : R :=
  (∑ i ∈ Finset.range width, ringBasisFaceX field i 0)
    + (∑ j ∈ Finset.range height, ringBasisFaceY field width j)
    - (∑ i ∈ Finset.range width, ringBasisFaceX field i height)
    - ∑ j ∈ Finset.range height, ringBasisFaceY field 0 j

/-- The sum of plaquette circulations equals the oriented rectangle boundary sum.

    (field : RingBasisField R)      [the ring-cell field]
    (width height : ℕ)              [the rectangle side lengths]
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      ringPlaquetteCirculation field i k) = ringRectangleBoundaryCirculation field width height
                                     [interior edges cancel and the outer walk remains]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    ∈                                [is one of the things in]
    `RingBasisField`                 [one ring cell at every lattice anchor]
    `Finset.range`                   [the counting positions below a side's endpoint]
    `ringPlaquetteCirculation`       [the counterclockwise sum around one plaquette]
    `ringRectangleBoundaryCirculation` [the counterclockwise sum on the outer boundary]

    The proof telescopes each face-coordinate field with `ringSum_forward_difference`. -/
theorem ringRectangle_discrete_stokes (field : RingBasisField R) (width height : ℕ) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      ringPlaquetteCirculation field i k)
      = ringRectangleBoundaryCirculation field width height := by
  have hx :
      (∑ i ∈ Finset.range width, ∑ j ∈ Finset.range height,
          (ringBasisFaceX field i (j + 1) - ringBasisFaceX field i j))
        = ∑ i ∈ Finset.range width,
            (ringBasisFaceX field i height - ringBasisFaceX field i 0) := by
    apply Finset.sum_congr rfl
    intro i _
    exact ringSum_forward_difference (ringBasisFaceX field i) height
  have hy :
      (∑ i ∈ Finset.range width, ∑ j ∈ Finset.range height,
          (ringBasisFaceY field (i + 1) j - ringBasisFaceY field i j))
        = ∑ j ∈ Finset.range height,
            (ringBasisFaceY field width j - ringBasisFaceY field 0 j) := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro j _
    exact ringSum_forward_difference (fun i => ringBasisFaceY field i j) width
  calc
    (∑ i ∈ Finset.range width, ∑ j ∈ Finset.range height,
        ringPlaquetteCirculation field i j)
        = ∑ i ∈ Finset.range width, ∑ j ∈ Finset.range height,
            ((ringBasisFaceY field (i + 1) j - ringBasisFaceY field i j)
              - (ringBasisFaceX field i (j + 1) - ringBasisFaceX field i j)) := by
                apply Finset.sum_congr rfl
                intro i _
                apply Finset.sum_congr rfl
                intro j _
                simp only [ringPlaquetteCirculation]
                ring
    _ = (∑ i ∈ Finset.range width, ∑ j ∈ Finset.range height,
          (ringBasisFaceY field (i + 1) j - ringBasisFaceY field i j))
        - ∑ i ∈ Finset.range width, ∑ j ∈ Finset.range height,
          (ringBasisFaceX field i (j + 1) - ringBasisFaceX field i j) := by
            simp_rw [Finset.sum_sub_distrib]
    _ = ringRectangleBoundaryCirculation field width height := by
          rw [hy, hx]
          simp_rw [Finset.sum_sub_distrib]
          simp only [ringRectangleBoundaryCirculation]
          ring

/-- Identifying opposite boundary face values makes total circulation zero.

    (field : RingBasisField R)      [the ring-cell field]
    (width height : ℕ)              [the rectangle side lengths]
    (hX : ∀ i, ringBasisFaceX field i height = ringBasisFaceX field i 0)
                                     [top and bottom x-face values are identified]
    (hY : ∀ k, ringBasisFaceY field width k = ringBasisFaceY field 0 k)
                                     [right and left y-face values are identified]
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      ringPlaquetteCirculation field i k) = 0
                                     [the total computed circulation vanishes]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    ∀                                [for every single one]
    ∈                                [is one of the things in]
    `RingBasisField`                 [one ring cell at every lattice anchor]
    `Finset.range`                   [the counting positions below a side's endpoint]
    `ringBasisFaceX`                 [the first-coordinate x-face read]
    `ringBasisFaceY`                 [the second-coordinate y-face read]
    `ringPlaquetteCirculation`       [the counterclockwise sum around one plaquette]

    The identified outer walk cancels after the interior edges telescope. -/
theorem ringClosed_rectangle_circulation_cancels (field : RingBasisField R) (width height : ℕ)
    (hX : ∀ i, ringBasisFaceX field i height = ringBasisFaceX field i 0)
    (hY : ∀ k, ringBasisFaceY field width k = ringBasisFaceY field 0 k) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      ringPlaquetteCirculation field i k) = 0 := by
  rw [ringRectangle_discrete_stokes, ringRectangleBoundaryCirculation]
  simp [hX, hY]

end Physics
end Apeiron
