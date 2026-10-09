import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.CarrierAnyRing
import Apeiron.Physics.CarrierFiniteAnyRing
import Apeiron.Physics.CarrierDissipationAnyRing

/-!
# The scalar carrier step and its exact-half flux over any commutative ring

@name Scalar Carrier Step Over Any Commutative Ring
@grade proved

A scalar production kernel updates one carrier sample by
`d - (outgoing - incoming) + source`, with the diffusive face flux the exact half
`-(right - left) / 2`. Neither the update nor the flux needs an order, a square root or a
division by two, so both are stated here over an arbitrary commutative ring. The real
declarations in `Apeiron.Physics.CarrierEvolve` and `Apeiron.Physics.CarrierDissipation` are
their ordered instances.

`scalarCarrierStep` is the update: the old value, less the outgoing-minus-incoming flux, plus
the source. `scalarCarrierStep_eq` reads the definition back, and `scalarCarrierStep_continuity`
is the local balance with the source on the right.

A general ring has no division by two. `DoublesTo q x` is the explicit doubling witness
`2 * q = x`, and `exactHalfFlux f left right` is `DoublesTo f (left - right)`, the value
`-(right - left) / 2` written with no division. `exactHalfFlux_neg` reads the same relation as
`2 * f = -(right - left)`, `doublesTo_neg` moves a witness across a sign, `doublesTo_sub`
subtracts two witnesses, `exactHalfFlux_of_scalarHalf` turns a scalar half `h` with `2 * h = 1`
into an
exact half, and `exactHalfFlux_difference` assembles the two face witnesses into the second
difference. A bridge over `ZMod (2 ^ 64)` instantiates the same declarations: two is not
invertible there, so it supplies the witness `2 * f = left - right` from the kernel's own
shift rather than a scalar half.

`diffusiveUpdate_half_three_anyRing` is the three-cell case at a doubling witness: with the
right ghost face closed, each cell averages with its neighbor, the two shared faces are exact
halves, and the three-cell total is conserved. It is the ring analogue of the real
`CarrierDissipation.diffusiveUpdate_half_three`, and it makes no energy claim: the real factor
`1/4` needs the order of `ℝ`.

## Root

This file instantiates no `Apeiron.Trichotomy` theorem: the update and the flux are arithmetic
identities over any commutative ring, closed by the ring laws alone. The values they act on are
the carrier readouts of `Apeiron.Physics.CarrierAnyRing`, `pointDensity` and `pointFlux`, so the
file is their pointwise arithmetic. The finite array telescoping and total-mass laws are
`Apeiron.Physics.CarrierFiniteAnyRing`; this file links its `ringEvolveCarrierDensity` to the
scalar step with the shared face flux as the outgoing and incoming amounts, and states the same
telescoping directly over `scalarCarrierStep`. The diffusive face and update of
`Apeiron.Physics.CarrierDissipationAnyRing` are the scalar step on that file's shared flux, and a
doubling witness turns that flux into an `exactHalfFlux` without dividing. What does not transfer
is every ordered claim: nonnegativity of the updated value, the stable range of the diffusive
coefficient and the Dirichlet energy bound are `Apeiron.Physics.CarrierEvolve` and
`Apeiron.Physics.CarrierDissipation` results.

## Reading the symbols

Statements are glossed line by line in brackets, one gloss per line.
`Apeiron.Linter.Registry` holds one canonical meaning per symbol, and
`Apeiron.Linter.Notation` checks each statement against it during elaboration.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-! ## The polynomial scalar update -/

/-- One scalar carrier update: the old value, less the outgoing-minus-incoming flux, plus the
source.

    (d outgoing incoming source : R)
                                    [the old value, the outgoing flux, the incoming flux and
                                     the source]
    R                               [the commutative ring the values live in]
    d - (outgoing - incoming) + source
                                    [the old value less the net outward flux plus the source]

    The source is one additive term, so it does not change the flux accounting. -/
def scalarCarrierStep (d outgoing incoming source : R) : R :=
  d - (outgoing - incoming) + source

/-- The scalar update written out.

    (d outgoing incoming source : R)
                                    [the old value, the outgoing flux, the incoming flux and
                                     the source]
    R                               [the commutative ring the values live in]
    `scalarCarrierStep`             [the scalar carrier update]
    d - (outgoing - incoming) + source
                                    [the old value less the net outward flux plus the source]

    This is the definition read back, so a bridge rewrites to the polynomial form with it. -/
theorem scalarCarrierStep_eq (d outgoing incoming source : R) :
    scalarCarrierStep d outgoing incoming source = d - (outgoing - incoming) + source := rfl

/-- The update satisfies the local balance with the source on the right.

    (d outgoing incoming source : R)
                                    [the old value, the outgoing flux, the incoming flux and
                                     the source]
    R                               [the commutative ring the values live in]
    `scalarCarrierStep`             [the scalar carrier update]
    scalarCarrierStep d outgoing incoming source - d + (outgoing - incoming) = source
                                    [the change plus the net outward flux equals the source]

    This is `rho' - rho + div J = source`, obtained by unfolding the definition. -/
theorem scalarCarrierStep_continuity (d outgoing incoming source : R) :
    scalarCarrierStep d outgoing incoming source - d + (outgoing - incoming) = source := by
  rw [scalarCarrierStep_eq]
  abel

/-! ## The exact-half diffusive flux witness -/

/-- Twice the first value is the second: the explicit witness that a half exists.

    (q x : R)                       [the half candidate and the value it doubles to]
    R                               [the commutative ring the values live in]
    2 * q = x                       [twice the candidate is the value]

    A general ring has no division, so a half is carried as this witness rather than computed.
    Over `ZMod (2 ^ 64)` the relation has a solution exactly when the value is even, which is
    what the kernel's shift supplies. -/
def DoublesTo (q x : R) : Prop := 2 * q = x

/-- The exact half diffusive flux across a face from `left` to `right`.

    (f left right : R)              [the flux candidate and the two face samples]
    R                               [the commutative ring the values live in]
    DoublesTo f (left - right)      [the flux doubles to the forward difference]
    `DoublesTo`                     [twice the first value is the second]

    This is `-(right - left) / 2` written with no division: `left - right` is the negative of
    the backward difference, so the same value is the exact half in either orientation. -/
def exactHalfFlux (f left right : R) : Prop := DoublesTo f (left - right)

/-- The exact-half flux relation is the doubling witness.

    (f left right : R)              [the flux candidate and the two face samples]
    R                               [the commutative ring the values live in]
    exactHalfFlux f left right      [the flux is the exact half across the face]
    ↔                               [exactly when, true in both directions]
    DoublesTo f (left - right)      [the flux doubles to the forward difference]
    `DoublesTo`                     [twice the first value is the second]

    This is the definition read back. -/
theorem exactHalfFlux_iff (f left right : R) :
    exactHalfFlux f left right ↔ DoublesTo f (left - right) := Iff.rfl

/-- The exact-half flux is also the negative half of the backward difference.

    (f left right : R)              [the flux candidate and the two face samples]
    R                               [the commutative ring the values live in]
    exactHalfFlux f left right      [the flux is the exact half across the face]
    ↔                               [exactly when, true in both directions]
    2 * f = -(right - left)         [twice the flux is the negative backward difference]

    This is the `-(right - left) / 2` reading of the flux. -/
theorem exactHalfFlux_neg (f left right : R) :
    exactHalfFlux f left right ↔ 2 * f = -(right - left) := by
  simp only [exactHalfFlux, DoublesTo, neg_sub]

/-- A doubling witness moves across a sign.

    {q x : R}                       [the half candidate and the value it doubles to]
    R                               [the commutative ring the values live in]
    DoublesTo q x                   [twice the candidate is the value]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    DoublesTo (-q) (-x)             [twice the negated candidate is the negated value]

    Negating both sides of the witness is the whole proof. -/
theorem doublesTo_neg {q x : R} (h : DoublesTo q x) : DoublesTo (-q) (-x) := by
  unfold DoublesTo at h ⊢
  rw [mul_neg, h]

/-- Two doubling witnesses subtract.

    {q r x y : R}                   [the two half candidates and the two values]
    R                               [the commutative ring the values live in]
    DoublesTo q x                   [twice the first candidate is the first value]
    DoublesTo r y                   [twice the second candidate is the second value]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    DoublesTo (q - r) (x - y)       [twice the candidate difference is the value difference]

    The net flux is the difference of two exact halves, so this is the step that assembles
    the update's flux difference from the two face witnesses. -/
theorem doublesTo_sub {q r x y : R} (hq : DoublesTo q x) (hr : DoublesTo r y) :
    DoublesTo (q - r) (x - y) := by
  unfold DoublesTo at hq hr ⊢
  rw [mul_sub, hq, hr]

/-- A scalar half turns a difference into an exact half flux.

    (h left right : R)              [the scalar half and the two face samples]
    R                               [the commutative ring the values live in]
    (hh : 2 * h = 1)                [twice the scalar half is one]
    exactHalfFlux (h * (left - right)) left right
                                    [the scaled difference is the exact half flux]
    `exactHalfFlux`                 [the exact half diffusive flux across the face]

    This is the route for a ring such as the dyadics that carries a half. `ZMod (2 ^ 64)` has
    no such `h`, so a bridge there supplies the witness directly instead. -/
theorem exactHalfFlux_of_scalarHalf (h left right : R) (hh : 2 * h = 1) :
    exactHalfFlux (h * (left - right)) left right := by
  unfold exactHalfFlux DoublesTo
  rw [← mul_assoc, hh, one_mul]

/-- The net flux difference doubles to the second difference.

    {qOut qIn d left right : R}
                                    [the outgoing and incoming half candidates, the cell value
                                     and its two neighbors]
    R                               [the commutative ring the values live in]
    exactHalfFlux qOut d right      [the outgoing flux is the exact half to the right]
    exactHalfFlux qIn left d        [the incoming flux is the exact half from the left]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    `DoublesTo`                     [twice the first value is the second]
    DoublesTo (qOut - qIn) (2 * d - right - left)
                                    [twice the net flux is the second difference]

    The two face witnesses assemble through `doublesTo_sub`, and the second difference is the
    forward difference less the backward one. -/
theorem exactHalfFlux_difference {qOut qIn d left right : R}
    (hOut : exactHalfFlux qOut d right) (hIn : exactHalfFlux qIn left d) :
    DoublesTo (qOut - qIn) (2 * d - right - left) := by
  have h := doublesTo_sub hOut hIn
  unfold DoublesTo at h ⊢
  rw [h]
  ring

/-! ## The finite shared-face total and the diffusive convention -/

/-- The finite carrier evolution is the scalar step on the shared face fluxes, with no source.

    (τ σ : R)                       [the time step and algebra parameter]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    R                               [the commutative ring the values live in]
    `ringEvolveCarrierDensity`      [the conservative material update of the density]
    `scalarCarrierStep`             [the scalar carrier update]
    `ringCarrierDensity`            [the density a phase cell carries at the amplitude]

    The outgoing face is the flux at `i + 1`, the incoming face the flux at `i`, and the finite
    update has no source, so the source argument is zero. -/
theorem ringEvolveCarrierDensity_eq_scalarCarrierStep
    (τ σ : R) (amp phase : R → R → R) (u : RingCell R) (i : ℕ) :
    ringEvolveCarrierDensity τ σ amp phase u i =
      scalarCarrierStep (ringCarrierDensity σ amp u i)
        (τ * ringCarrierFaceFlux σ amp phase u (i + 1))
        (τ * ringCarrierFaceFlux σ amp phase u i) 0 := by
  rw [ringEvolveCarrierDensity, ringCarrierDivergence, scalarCarrierStep_eq]
  ring

/-- A finite line of scalar steps over one shared face array leaves only the endpoint flux plus
the sources.

    (d face source : ℕ → R)         [the density, the shared face flux and the source arrays]
    (n : ℕ)                         [the number of cells in the line]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    R                               [the commutative ring the values live in]
    ∈                               [is one of the things in]
    `scalarCarrierStep`             [the scalar carrier update]
    `Finset.range`                  [the counting positions below the endpoint]

    Cell `i` sends `face (i + 1)` out and receives `face i`, so every interior face is read from
    the same array once with each sign and cancels. -/
theorem sum_scalarCarrierStep_eq (d face source : ℕ → R) (n : ℕ) :
    (∑ i ∈ Finset.range n, scalarCarrierStep (d i) (face (i + 1)) (face i) (source i))
      = (∑ i ∈ Finset.range n, d i) - (face n - face 0)
        + ∑ i ∈ Finset.range n, source i := by
  simp_rw [scalarCarrierStep_eq]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, sum_forward_difference_generic face n]

/-- A closed finite line with no net source conserves the scalar total.

    (d face source : ℕ → R)         [the density, the shared face flux and the source arrays]
    (n : ℕ)                         [the number of cells in the line]
    (hLeft : face 0 = 0)            [the left endpoint face carries nothing]
    (hRight : face n = 0)           [the right endpoint face carries nothing]
    (hSource : (∑ i ∈ Finset.range n, source i) = 0)
                                     [the finite source total vanishes]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    R                               [the commutative ring the values live in]
    ∈                               [is one of the things in]
    `scalarCarrierStep`             [the scalar carrier update]
    `Finset.range`                  [the counting positions below the endpoint]

    The telescoping theorem leaves the endpoint faces and the source total; the closed ends and
    the zero source total cancel them. -/
theorem scalarCarrierStep_total_mass (d face source : ℕ → R) (n : ℕ)
    (hLeft : face 0 = 0) (hRight : face n = 0)
    (hSource : (∑ i ∈ Finset.range n, source i) = 0) :
    (∑ i ∈ Finset.range n, scalarCarrierStep (d i) (face (i + 1)) (face i) (source i))
      = ∑ i ∈ Finset.range n, d i := by
  rw [sum_scalarCarrierStep_eq, hLeft, hRight, hSource]
  ring

/-- The diffusive update is the scalar step on the shared diffusive face fluxes, with no source.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (rho : ℕ → R)                   [the density array]
    (i : ℕ)                         [the counting cell position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    R                               [the commutative ring the values live in]
    `scalarCarrierStep`             [the scalar carrier update]

    Cell `i` loses the outgoing face `i + 1` and gains the incoming face `i`, which is the scalar
    step on the two face values with no source. -/
theorem diffusiveUpdate_eq_scalarCarrierStep (ν : R) (rho : ℕ → R) (i : ℕ) :
    diffusiveUpdate ν rho i =
      scalarCarrierStep (rho i) (diffusiveFlux ν rho (i + 1)) (diffusiveFlux ν rho i) 0 := by
  rw [diffusiveUpdate, scalarCarrierStep_eq]
  ring

/-- A scalar half makes the shared diffusive face an exact-half flux witness.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (hν : 2 * ν = 1)                [twice the parameter is one, the half witness]
    (rho : ℕ → R)                   [the density array]
    (i : ℕ)                         [the counting face position]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    R                               [the commutative ring the values live in]
    `exactHalfFlux`                 [the exact half diffusive flux across the face]

    No `1/2` is written: the hypothesis is the doubling witness `2 * ν = 1`, so a ring where two
    is not invertible can still supply the face value. -/
theorem exactHalfFlux_diffusiveFlux (ν : R) (hν : 2 * ν = 1) (rho : ℕ → R) (i : ℕ) :
    exactHalfFlux (diffusiveFlux ν rho (i + 1)) (rho i) (rho (i + 1)) := by
  have hface : diffusiveFlux ν rho (i + 1) = ν * (rho i - rho (i + 1)) := by
    rw [diffusiveFlux]
    simp only [smul_eq_mul]
    ring
  unfold exactHalfFlux DoublesTo
  rw [hface, ← mul_assoc, hν, one_mul]

/-- The three-cell half step over any commutative ring averages each cell with its neighbor.

    (ν : R)                         [the explicit gradient-flow time-step parameter]
    (hν : DoublesTo ν 1)            [twice the parameter is one, the half witness]
    (rho : ℕ → R)                   [the density array]
    (hRight : diffusiveFlux ν rho 3 = 0)
                                     [the right ghost face carries no flux]
    exactHalfFlux (diffusiveFlux ν rho 1) (rho 0) (rho 1)
                                     [the first shared face is an exact half]
    ∧ exactHalfFlux (diffusiveFlux ν rho 2) (rho 1) (rho 2)
                                     [the second shared face is an exact half]
    ∧ diffusiveUpdate ν rho 0 = ν • (rho 0 + rho 1)
                                     [the first cell averages with cell 1]
    ∧ diffusiveUpdate ν rho 1 = ν • (rho 0 + rho 2)
                                     [the middle cell averages its two neighbors]
    ∧ diffusiveUpdate ν rho 2 = ν • (rho 1 + rho 2)
                                     [the last cell averages with cell 1]
    ∧ diffusiveUpdate ν rho 0 + diffusiveUpdate ν rho 1 + diffusiveUpdate ν rho 2
        = rho 0 + rho 1 + rho 2
                                     [the three-cell total is unchanged]
    `DoublesTo`                     [twice the first value is the second]
    `exactHalfFlux`                 [the exact half flux: doubles to the forward difference]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `diffusiveFlux`                 [the shared face flux: the negative forward difference]
    `•`                             [stretch by, a number times a point]
    ∧                               [both are true]

    This is the ring analogue of `CarrierDissipation.diffusiveUpdate_half_three`: at the doubling
    witness the interior weight `1 - 2 * ν` vanishes, so every cell averages with its neighbor,
    the shared faces are exact halves, and the three-cell total is conserved. No energy is
    claimed: the real factor `1/4` needs the order of `ℝ`, which this ring does not carry. -/
theorem diffusiveUpdate_half_three_anyRing (ν : R) (hν : DoublesTo ν 1) (rho : ℕ → R)
    (hRight : diffusiveFlux ν rho 3 = 0) :
    exactHalfFlux (diffusiveFlux ν rho 1) (rho 0) (rho 1) ∧
    exactHalfFlux (diffusiveFlux ν rho 2) (rho 1) (rho 2) ∧
    diffusiveUpdate ν rho 0 = ν • (rho 0 + rho 1) ∧
    diffusiveUpdate ν rho 1 = ν • (rho 0 + rho 2) ∧
    diffusiveUpdate ν rho 2 = ν • (rho 1 + rho 2) ∧
    diffusiveUpdate ν rho 0 + diffusiveUpdate ν rho 1 + diffusiveUpdate ν rho 2
      = rho 0 + rho 1 + rho 2 := by
  have hν' : 2 * ν = 1 := hν
  refine ⟨exactHalfFlux_diffusiveFlux ν hν' rho 0,
    exactHalfFlux_diffusiveFlux ν hν' rho 1, ?_, ?_, ?_, ?_⟩
  · rw [diffusiveUpdate_zero]
    simp only [smul_eq_mul]
    linear_combination (-(rho 0)) * hν'
  · rw [diffusiveUpdate_succ]
    simp only [smul_eq_mul]
    linear_combination (-(rho 1)) * hν'
  · rw [diffusiveUpdate, hRight]
    simp only [diffusiveFlux, smul_eq_mul]
    linear_combination (-(rho 2)) * hν'
  · have h := diffusiveUpdate_total_mass ν rho 3 hRight
    simpa [Finset.sum_range_succ] using h

/-! ## What is not here

The ordered claims stay with the real modules. Nonnegativity of the updated value, the stable
range of the diffusive coefficient and the Dirichlet energy bound use the order of `ℝ`, which
a general commutative ring does not carry, so they are `Apeiron.Physics.CarrierEvolve` and
`Apeiron.Physics.CarrierDissipation` results and are not restated here.
-/

end Physics
end Apeiron
