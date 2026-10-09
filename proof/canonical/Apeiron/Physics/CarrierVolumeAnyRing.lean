import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.AnyRing
import Apeiron.Physics.CarrierAnyRing
import Apeiron.Physics.CarrierFiniteAnyRing

/-!
# Three-dimensional carrier volume accounting over any commutative ring

@name Three-Dimensional Carrier Volume Accounting Over Any Commutative Ring
@cite kocik-1999
@grade proved

`Apeiron.Physics.CarrierVolume` states the three-dimensional carrier accounting for `ℝ`. The
telescoping is additive, the divergence is a difference of one shared face value, and the update
is a forward-Euler subtraction, so none of it needs the order, the square root or the completeness
of the reals. This file states the laws over the weakest structure that carries them and lets the
real declarations be their specialisations.

The extended sample lattice is `ℕ × ℕ × ℕ`: index 0 on every axis is a ghost layer, and logical
cell `(x,y,z)` reads the sample at `(x+1,y+1,z+1)`. `ringSampleDensity3` is the generic
`pointDensity` at that sample, and `ringCarrierDensity3` is its logical-cell shift. Face `q` on
the x axis is the shared face between logical cells `q-1` and `q`; it carries one stored value read
from the extended amplitude at `(q,y+1,z+1)` and the forward phase difference
`(q,y+1,z+1) → (q+1,y+1,z+1)`. For `q ≥ 1` that amplitude is the lower logical cell's, so the
value is at once the outgoing x-read of cell `q-1` and the incoming x-read of cell `q`; for `q = 0`
it is the explicit boundary ghost. The y and z faces shift their transverse coordinates the same
way. No separate face-amplitude field is posited: a symmetric face amplitude would be new material
state, and the one-dimensional `FaceFlux` already fixes the honest single-valued convention.

`LatticeFaceFlux3` is one value on every forward face. `latticeCellDivergence3` is the
outgoing-minus-incoming difference across the six faces of one cell, and `sum_forward_difference3_x`
and its y and z twins telescope a line of them. Those need only an additive commutative group.
`box_divergence_balance_generic` telescopes the triple sum to the six outer faces and
`closed_box_flux_cancels_generic` identifies opposite faces. `ringCarrierDivergence3` and
`ringEvolveCarrierDensity3` are the ring readout and one forward-Euler step; continuity, the
telescoped divergence and closed-box total conservation are their ring-universal laws.

No order appears here. The CFL bound and the nonnegativity of the updated density are instance
results for `ℝ` and stay in `Apeiron.Physics.CarrierVolume`.

## Root

| from | here |
|---|---|
| `CarrierAnyRing.pointDensity` | `ringSampleDensity3`, `ringCarrierDensity3` |
| `CarrierAnyRing.pointFlux` | `ringFaceFluxX`, `ringFaceFluxY`, `ringFaceFluxZ` |
| `CarrierFiniteAnyRing.sum_forward_difference_generic` | `sum_forward_difference3_x` |
| `Trichotomy.AnyRing.ringNormForm` | `ringSampleDensity3_eq_pointDensity` |

Each face is a sample of `pointFlux` at the amplitude and the two phase samples, so no second
flux formula is introduced.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

/-! ## Shared-face accounting over an additive commutative group -/

variable {G : Type*} [AddCommGroup G]

/-- A single value on every forward face of a three-dimensional counting lattice.

    (G : Type*)                      [the additive commutative group the values live in]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    LatticeFaceFlux3 G              [one value on every forward face of the lattice]

    The first input chooses a face coordinate; the next two choose the transverse logical
    coordinates. Sharing this function between adjacent cells makes an interior face
    single-valued. -/
abbrev LatticeFaceFlux3 (G : Type*) : Type _ := ℕ → ℕ → ℕ → G

/-- The outgoing-minus-incoming flux of one cell across its six faces.

    (first second third : LatticeFaceFlux3 G)
                                     [the single-valued fluxes on the three axes]
    (x y z : ℕ)                      [the cell's three counting-lattice coordinates]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    G                                [the additive commutative group the values live in]
    LatticeFaceFlux3 G               [one value on every forward face of the lattice]

    Each incoming face is the preceding cell's outgoing face, with the opposite sign. -/
def latticeCellDivergence3 (first second third : LatticeFaceFlux3 G) (x y z : ℕ) : G :=
  (first (x + 1) y z - first x y z) + (second x (y + 1) z - second x y z)
    + (third x y (z + 1) - third x y z)

/-- The x-axis part of the box sum telescopes to its two outer x faces.

    (f : LatticeFaceFlux3 G)         [one single-valued flux on the three axes]
    (width height depth : ℕ)         [the box's three side lengths]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    G                                [the additive commutative group the values live in]
    ∈                                [is one of the things in]
    LatticeFaceFlux3 G               [one value on every forward face of the lattice]

    Every intermediate x face occurs once positively and once negatively, so it cancels. -/
theorem sum_forward_difference3_x (f : LatticeFaceFlux3 G) (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        (f (x + 1) y z - f x y z))
      = ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth, (f width y z - f 0 y z) := by
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro y _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro z _
  exact sum_forward_difference_generic (fun x => f x y z) width

/-- The y-axis part of the box sum telescopes to its two outer y faces.

    (f : LatticeFaceFlux3 G)         [one single-valued flux on the three axes]
    (width height depth : ℕ)         [the box's three side lengths]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    G                                [the additive commutative group the values live in]
    ∈                                [is one of the things in]
    LatticeFaceFlux3 G               [one value on every forward face of the lattice]

    Every intermediate y face occurs once positively and once negatively, so it cancels. -/
theorem sum_forward_difference3_y (f : LatticeFaceFlux3 G) (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        (f x (y + 1) z - f x y z))
      = ∑ x ∈ Finset.range width, ∑ z ∈ Finset.range depth, (f x height z - f x 0 z) := by
  apply Finset.sum_congr rfl
  intro x _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro z _
  exact sum_forward_difference_generic (fun y => f x y z) height

/-- The z-axis part of the box sum telescopes to its two outer z faces.

    (f : LatticeFaceFlux3 G)         [one single-valued flux on the three axes]
    (width height depth : ℕ)         [the box's three side lengths]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    G                                [the additive commutative group the values live in]
    ∈                                [is one of the things in]
    LatticeFaceFlux3 G               [one value on every forward face of the lattice]

    Every intermediate z face occurs once positively and once negatively, so it cancels. -/
theorem sum_forward_difference3_z (f : LatticeFaceFlux3 G) (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        (f x y (z + 1) - f x y z))
      = ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, (f x y depth - f x y 0) := by
  apply Finset.sum_congr rfl
  intro x _
  apply Finset.sum_congr rfl
  intro y _
  exact sum_forward_difference_generic (fun z => f x y z) depth

/-- The sum of cell divergences in a finite box is its six oriented outer faces.

    (first second third : LatticeFaceFlux3 G)
                                     [the single-valued fluxes on the three axes]
    (width height depth : ℕ)         [the box's three side lengths]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    G                                [the additive commutative group the values live in]
    ∈                                [is one of the things in]
    LatticeFaceFlux3 G               [one value on every forward face of the lattice]

    Interior faces disappear solely because every occurrence is read from the same face field. -/
theorem box_divergence_balance_generic (first second third : LatticeFaceFlux3 G)
    (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        latticeCellDivergence3 first second third x y z)
      = (∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth, (first width y z - first 0 y z))
        + (∑ x ∈ Finset.range width, ∑ z ∈ Finset.range depth,
            (second x height z - second x 0 z))
        + ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height,
            (third x y depth - third x y 0) := by
  have hx := sum_forward_difference3_x first width height depth
  have hy := sum_forward_difference3_y second width height depth
  have hz := sum_forward_difference3_z third width height depth
  simp_rw [latticeCellDivergence3, Finset.sum_add_distrib]
  rw [hx, hy, hz]

/-- A closed box has zero total divergence when each pair of opposite faces is one flux.

    (first second third : LatticeFaceFlux3 G)
                                     [the single-valued fluxes on the three axes]
    (width height depth : ℕ)         [the box's three side lengths]
    (hFirst : ∀ y z, first width y z = first 0 y z)
                                     [right and left x faces are identified]
    (hSecond : ∀ x z, second x height z = second x 0 z)
                                     [top and bottom y faces are identified]
    (hThird : ∀ x y, third x y depth = third x y 0)
                                     [front and back z faces are identified]
    ℕ                                [the counting numbers 0, 1, 2, 3, and so on]
    G                                [the additive commutative group the values live in]
    ∀                                [for every single one]
    ∈                                [is one of the things in]
    LatticeFaceFlux3 G               [one value on every forward face of the lattice]

    The box theorem first cancels all interior faces; closed-side identification then cancels the
    six remaining boundary contributions. -/
theorem closed_box_flux_cancels_generic (first second third : LatticeFaceFlux3 G)
    (width height depth : ℕ)
    (hFirst : ∀ y z, first width y z = first 0 y z)
    (hSecond : ∀ x z, second x height z = second x 0 z)
    (hThird : ∀ x y, third x y depth = third x y 0) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
      latticeCellDivergence3 first second third x y z) = 0 := by
  rw [box_divergence_balance_generic]
  simp [hFirst, hSecond, hThird]

/-! ## The carrier readout on the three-dimensional lattice over any commutative ring -/

variable {R : Type*} [CommRing R]

/-- A ring amplitude or phase field over the three-dimensional counting carrier.

    (R : Type*)                      [the commutative ring the point is built over]
    RingPhaseField3 R                [a ring value at every triple of counting coordinates]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]

    Three counting coordinates in, one ring value out. Index 0 on any axis is a ghost layer. -/
abbrev RingPhaseField3 (R : Type*) : Type _ := ℕ → ℕ → ℕ → R

/-- The sampled density at one extended sample coordinate.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (a b c : ℕ)                     [an extended sample coordinate]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    This is the generic `pointDensity` at the amplitude; no second material field appears. -/
def ringSampleDensity3 (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (a b c : ℕ) : R :=
  pointDensity σ (A a b c) u

/-- The sampled density is the CarrierAnyRing point density.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (a b c : ℕ)                     [an extended sample coordinate]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `ringSampleDensity3`            [the generic amount at one extended sample]

    By definition, so the root is the generic point density. -/
theorem ringSampleDensity3_eq_pointDensity (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (a b c : ℕ) : ringSampleDensity3 σ A u a b c = pointDensity σ (A a b c) u := rfl

/-- Density sampled from the amplitude carrier at one logical cell.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    Logical cell `(x,y,z)` reads the extended sample at `(x+1,y+1,z+1)`. -/
def ringCarrierDensity3 (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (x y z : ℕ) : R :=
  ringSampleDensity3 σ A u (x + 1) (y + 1) (z + 1)

/-- The sampled density is the amplitude times itself at a unit direction.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (a b c : ℕ)                     [an extended sample coordinate]
    (hu : ringNormForm σ u = 1)     [the phase direction has size one]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringSampleDensity3`            [the generic amount at one extended sample]

    The unit direction leaves a factor one, so the sample comes out squared. -/
theorem ringSampleDensity3_eq_amplitude_sq (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (a b c : ℕ) (hu : ringNormForm σ u = 1) :
    ringSampleDensity3 σ A u a b c = A a b c * A a b c := by
  rw [ringSampleDensity3, pointDensity_eq, hu, mul_one]
  simp only [ringNormForm]
  ring

/-- The logical cell density is the shifted sample squared at a unit direction.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    (hu : ringNormForm σ u = 1)     [the phase direction has size one]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringCarrierDensity3`           [the generic density at the logical cell's shifted sample]

    The same squared stretch as the extended sample, read at the logical cell's shift. -/
theorem ringCarrierDensity3_eq_amplitude_sq (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (x y z : ℕ) (hu : ringNormForm σ u = 1) :
    ringCarrierDensity3 σ A u x y z
      = A (x + 1) (y + 1) (z + 1) * A (x + 1) (y + 1) (z + 1) := by
  rw [ringCarrierDensity3,
    ringSampleDensity3_eq_amplitude_sq σ A u (x + 1) (y + 1) (z + 1) hu]

/-- The single-valued forward face flux in the x direction.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (q y z : ℕ)                     [the face coordinate and its two transverse logical
                                     coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    Face `q` is the shared face between logical cells `q-1` and `q`. It reads the extended sample
    at `(q,y+1,z+1)` and the forward phase difference `(q,y+1,z+1) → (q+1,y+1,z+1)`, so for `q ≥ 1`
    it is the outgoing x-read of the lower cell and the incoming x-read of the upper cell at
    once. -/
def ringFaceFluxX (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (q y z : ℕ) : R :=
  pointFlux σ (A q (y + 1) (z + 1)) u (phase q (y + 1) (z + 1))
    (phase (q + 1) (y + 1) (z + 1))

/-- The single-valued forward face flux in the y direction.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x q z : ℕ)                     [the face coordinate and its two transverse logical
                                     coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    The y-axis twin of `ringFaceFluxX`: face `q` reads the extended sample at `(x+1,q,z+1)` and
    the forward phase difference `(x+1,q,z+1) → (x+1,q+1,z+1)`. -/
def ringFaceFluxY (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x q z : ℕ) : R :=
  pointFlux σ (A (x + 1) q (z + 1)) u (phase (x + 1) q (z + 1))
    (phase (x + 1) (q + 1) (z + 1))

/-- The single-valued forward face flux in the z direction.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x y q : ℕ)                     [the face coordinate and its two transverse logical
                                     coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    The z-axis twin of `ringFaceFluxX`: face `q` reads the extended sample at `(x+1,y+1,q)` and
    the forward phase difference `(x+1,y+1,q) → (x+1,y+1,q+1)`. -/
def ringFaceFluxZ (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x y q : ℕ) : R :=
  pointFlux σ (A (x + 1) (y + 1) q) u (phase (x + 1) (y + 1) q)
    (phase (x + 1) (y + 1) (q + 1))

/-- The x face flux is the sampled density times the forward phase difference.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (q y z : ℕ)                     [the face coordinate and its two transverse logical
                                     coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringSampleDensity3`            [the generic amount at one extended sample]
    `pointFlux`                     [the first-direction flux: amount times phase difference]
    `ringFaceFluxX`                 [the single-valued forward x-face flux]

    This is `pointFlux` read through the sampled density, so no second flux formula appears. -/
theorem ringFaceFluxX_eq (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (q y z : ℕ) :
    ringFaceFluxX σ A u phase q y z
      = ringSampleDensity3 σ A u q (y + 1) (z + 1)
        * (phase (q + 1) (y + 1) (z + 1) - phase q (y + 1) (z + 1)) := by
  rw [ringFaceFluxX, pointFlux, ringSampleDensity3]

/-- The y face flux is the sampled density times the forward phase difference.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x q z : ℕ)                     [the face coordinate and its two transverse logical
                                     coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringSampleDensity3`            [the generic amount at one extended sample]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    This is the y-axis read of the same single formula. -/
theorem ringFaceFluxY_eq (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x q z : ℕ) :
    ringFaceFluxY σ A u phase x q z
      = ringSampleDensity3 σ A u (x + 1) q (z + 1)
        * (phase (x + 1) (q + 1) (z + 1) - phase (x + 1) q (z + 1)) := by
  rw [ringFaceFluxY, pointFlux, ringSampleDensity3]

/-- The z face flux is the sampled density times the forward phase difference.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x y q : ℕ)                     [the face coordinate and its two transverse logical
                                     coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringSampleDensity3`            [the generic amount at one extended sample]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    This is the z-axis read of the same single formula. -/
theorem ringFaceFluxZ_eq (σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x y q : ℕ) :
    ringFaceFluxZ σ A u phase x y q
      = ringSampleDensity3 σ A u (x + 1) (y + 1) q
        * (phase (x + 1) (y + 1) (q + 1) - phase (x + 1) (y + 1) q) := by
  rw [ringFaceFluxZ, pointFlux, ringSampleDensity3]

/-- The time-step-scaled outgoing-minus-incoming carrier flux at one cell.

    (τ σ : R)                       [the time step and the algebra parameter]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    LatticeFaceFlux3 R              [one value on every forward face of the lattice]

    Scaling the six-face difference by `τ` makes this the amount transferred during one
    forward-Euler step. -/
def ringCarrierDivergence3 (τ σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x y z : ℕ) : R :=
  τ * latticeCellDivergence3 (ringFaceFluxX σ A u phase) (ringFaceFluxY σ A u phase)
    (ringFaceFluxZ σ A u phase) x y z

/-- One explicit conservative material update of the carrier density.

    (τ σ : R)                       [the time step and the algebra parameter]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    The old norm-form density of the same logical cell loses exactly its net outward
    phase-derived six-face transfer. -/
def ringEvolveCarrierDensity3 (τ σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x y z : ℕ) : R :=
  ringCarrierDensity3 σ A u x y z - ringCarrierDivergence3 τ σ A u phase x y z

/-- The constructed update satisfies discrete continuity at every cell.

    (τ σ : R)                       [the time step and the algebra parameter]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringCarrierDensity3`           [the generic density at the logical cell's shifted sample]

    This is `rho' - rho + div J = 0`, obtained by unfolding the forward-Euler definition. -/
theorem ringEvolveCarrierDensity3_continuity (τ σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (x y z : ℕ) :
    ringEvolveCarrierDensity3 τ σ A u phase x y z - ringCarrierDensity3 σ A u x y z
      + ringCarrierDivergence3 τ σ A u phase x y z = 0 := by
  rw [ringEvolveCarrierDensity3]
  ring

/-- Pulling a constant factor out of a triple sum.

    (τ : R)                         [the time step: how far forward one update steps]
    (g : ℕ → ℕ → ℕ → R)             [a ring value at every triple of counting coordinates]
    (a b c : ℕ)                     [the three side lengths of the box]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    ∈                               [is one of the things in]

    Used to expose the time-scaled six-face boundary sum of one evolution step. -/
theorem sum_const_mul3 (τ : R) (g : ℕ → ℕ → ℕ → R) (a b c : ℕ) :
    (∑ x ∈ Finset.range a, ∑ y ∈ Finset.range b, ∑ z ∈ Finset.range c, τ * g x y z)
      = τ * (∑ x ∈ Finset.range a, ∑ y ∈ Finset.range b, ∑ z ∈ Finset.range c, g x y z) := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro x _
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro y _
  rw [Finset.mul_sum]

/-- The sum of one-step carrier divergences is the scaled six-face boundary sum.

    (τ σ : R)                       [the time step and the algebra parameter]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (width height depth : ℕ)        [the box's three side lengths]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ∈                               [is one of the things in]
    LatticeFaceFlux3 R              [one value on every forward face of the lattice]
    `ringFaceFluxX`                 [the single-valued forward x-face flux]

    Interior face terms cancel by `box_divergence_balance_generic`; only the oriented outer
    faces survive. -/
theorem ringSum_carrierDivergence3 (τ σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        ringCarrierDivergence3 τ σ A u phase x y z)
      = τ * ((∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
                (ringFaceFluxX σ A u phase width y z - ringFaceFluxX σ A u phase 0 y z))
          + (∑ x ∈ Finset.range width, ∑ z ∈ Finset.range depth,
                (ringFaceFluxY σ A u phase x height z - ringFaceFluxY σ A u phase x 0 z))
          + ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height,
                (ringFaceFluxZ σ A u phase x y depth - ringFaceFluxZ σ A u phase x y 0)) := by
  simp_rw [ringCarrierDivergence3]
  rw [sum_const_mul3]
  congr 1
  exact box_divergence_balance_generic (ringFaceFluxX σ A u phase) (ringFaceFluxY σ A u phase)
    (ringFaceFluxZ σ A u phase) width height depth

/-- A closed finite carrier box conserves total density in one nontrivial update.

    (τ σ : R)                       [the time step and the algebra parameter]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (width height depth : ℕ)        [the box's three side lengths]
    (hFirst : ∀ y z, ringFaceFluxX σ A u phase width y z = ringFaceFluxX σ A u phase 0 y z)
                                    [right and left x faces are identified]
    (hSecond : ∀ x z, ringFaceFluxY σ A u phase x height z = ringFaceFluxY σ A u phase x 0 z)
                                    [top and bottom y faces are identified]
    (hThird : ∀ x y, ringFaceFluxZ σ A u phase x y depth = ringFaceFluxZ σ A u phase x y 0)
                                    [front and back z faces are identified]
    R                               [the commutative ring the point is built over]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ∀                               [for every single one]
    ∈                               [is one of the things in]
    `ringCarrierDensity3`           [the generic density at the logical cell's shifted sample]
    `ringFaceFluxX`                 [the single-valued forward x-face flux]

    Telescoping removes every shared interior transfer; the identified outer transfers leave the
    old and new finite sums equal. -/
theorem ringEvolveCarrierDensity3_total_mass (τ σ : R) (A : RingPhaseField3 R) (u : RingCell R)
    (phase : RingPhaseField3 R) (width height depth : ℕ)
    (hFirst : ∀ y z, ringFaceFluxX σ A u phase width y z = ringFaceFluxX σ A u phase 0 y z)
    (hSecond : ∀ x z, ringFaceFluxY σ A u phase x height z = ringFaceFluxY σ A u phase x 0 z)
    (hThird : ∀ x y, ringFaceFluxZ σ A u phase x y depth = ringFaceFluxZ σ A u phase x y 0) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        ringEvolveCarrierDensity3 τ σ A u phase x y z)
      = ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
          ringCarrierDensity3 σ A u x y z := by
  have hdiv :
      (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
          ringCarrierDivergence3 τ σ A u phase x y z) = 0 := by
    rw [ringSum_carrierDivergence3]
    simp [hFirst, hSecond, hThird]
  simp_rw [ringEvolveCarrierDensity3, Finset.sum_sub_distrib]
  rw [hdiv, sub_zero]

/- The density, face, divergence and update chain reduces at the integer instance. -/
#guard ringEvolveCarrierDensity3 (R := Int) 2 2
  (fun a b c => Int.ofNat (a + b + c + 1)) (2, 1)
  (fun a b c => Int.ofNat (a + b + c)) 0 0 0 == -52

end Physics
end Apeiron
