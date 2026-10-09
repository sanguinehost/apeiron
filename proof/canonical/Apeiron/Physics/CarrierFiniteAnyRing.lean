import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.CarrierAnyRing

/-!
# Finite carrier boundary and evolution over any commutative ring

@name Finite Carrier Boundary And Evolution Over Any Commutative Ring
@grade proved

`Apeiron.Physics.CarrierBoundary` and `Apeiron.Physics.CarrierEvolve` state the finite
one-dimensional carrier accounting for `ℝ`. The telescoping is additive, the divergence is a
difference of one shared face value, and the evolution is a forward-Euler subtraction, so none of
it needs the order, the square root or the completeness of the reals. This file states the laws
over the weakest structure that carries them and lets the real declarations be their
specialisations.

`LatticeFaceFlux` is one value on every forward face of a counting lattice.
`latticeCellDivergence` is the outgoing-minus-incoming difference of two such fields, and
`sum_forward_difference_generic` telescopes a line of them. Those three need only an additive
commutative group.

`ringSampledFluxFirst` samples the generic `pointFlux` of `Apeiron.Physics.CarrierAnyRing` on the
counting lattice. `ringCarrierDensity`, `ringCarrierFaceFlux`, `ringCarrierDivergence` and
`ringEvolveCarrierDensity` are the one-dimensional readout and one forward-Euler step over an
arbitrary commutative ring, so `ZMod (2 ^ 64)` is an instance rather than an analogy. Continuity,
the telescoped divergence and closed-boundary total conservation are their ring-universal laws.

No order appears here. The CFL bound and the nonnegativity of the updated density are instance
results for `ℝ` and stay in `Apeiron.Physics.CarrierEvolve`.

## Root

| from | here |
|---|---|
| `CarrierAnyRing.pointDensity` | `ringCarrierDensity`, `ringEvolveCarrierDensity` |
| `CarrierAnyRing.pointFlux` | `ringSampledFluxFirst`, `ringCarrierFaceFlux` |

`ringSampledFluxFirst` reads `pointFlux` at the amplitude and the two phase samples; no second
flux formula is introduced.
-/

namespace Apeiron
namespace Physics

open Trichotomy.AnyRing

/-! ## Shared-face accounting over an additive commutative group -/

variable {G : Type*} [AddCommGroup G]

/-- A single flux value on every forward face of a counting lattice.

    (G : Type*)                      [the additive commutative group the values live in]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    LatticeFaceFlux G               [one value on every forward face of the lattice]

    The first input chooses a line and the second chooses a face on that line. Sharing this
    function between adjacent cells is what makes an interior face single-valued. -/
abbrev LatticeFaceFlux (G : Type*) : Type _ := ℕ → ℕ → G

/-- The outgoing-minus-incoming flux of one product cell.

    (first second : LatticeFaceFlux G)
                                    [the single-valued fluxes in the two directions]
    (i j : ℕ)                       [the cell's counting-lattice coordinates]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    G                               [the additive commutative group the values live in]
    LatticeFaceFlux G               [one value on every forward face of the lattice]

    Each incoming face is the preceding cell's outgoing face, with the opposite sign. -/
def latticeCellDivergence (first second : LatticeFaceFlux G) (i j : ℕ) : G :=
  (first (i + 1) j - first i j) + (second i (j + 1) - second i j)

/-- A finite line of forward differences leaves only its two endpoint faces.

    (f : ℕ → G)                     [one single-valued flux along a line]
    (n : ℕ)                         [the number of cells in the line]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    ∈                               [is one of the things in]
    `Finset.range`                  [the counting positions below the endpoint]

    Every intermediate value occurs once positively and once negatively, so it cancels. -/
theorem sum_forward_difference_generic (f : ℕ → G) (n : ℕ) :
    (∑ i ∈ Finset.range n, (f (i + 1) - f i)) = f n - f 0 := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Finset.sum_range_succ, ih]
      abel

/-- The sum of cell divergences in a finite rectangle is its four oriented outer sides.

    (first second : LatticeFaceFlux G)
                                    [the single-valued fluxes in the two directions]
    (width height : ℕ)             [the rectangle's two side lengths]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    G                               [the additive commutative group the values live in]
    ∈                               [is one of the things in]
    LatticeFaceFlux G               [one value on every forward face of the lattice]
    `Finset.range`                  [the counting positions below a side's endpoint]

    The first sum is right minus left; the second is top minus bottom. Interior faces disappear
    solely because every occurrence is read from the same face field. -/
theorem rectangle_boundary_balance_generic (first second : LatticeFaceFlux G)
    (width height : ℕ) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
        latticeCellDivergence first second i k)
      = (∑ k ∈ Finset.range height, (first width k - first 0 k))
        + ∑ i ∈ Finset.range width, (second i height - second i 0) := by
  have hfirst :
      (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
          (first (i + 1) k - first i k))
        = ∑ k ∈ Finset.range height, (first width k - first 0 k) := by
    rw [Finset.sum_comm]
    apply Finset.sum_congr rfl
    intro k _
    exact sum_forward_difference_generic (fun i => first i k) width
  have hsecond :
      (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
          (second i (k + 1) - second i k))
        = ∑ i ∈ Finset.range width, (second i height - second i 0) := by
    apply Finset.sum_congr rfl
    intro i _
    exact sum_forward_difference_generic (second i) height
  simp_rw [latticeCellDivergence, Finset.sum_add_distrib]
  rw [hfirst, hsecond]

/-- A closed rectangle has zero total divergence when each pair of opposite sides is one flux.

    (first second : LatticeFaceFlux G)
                                    [the single-valued fluxes in the two directions]
    (width height : ℕ)             [the rectangle's two side lengths]
    (hFirst : ∀ k, first width k = first 0 k)
                                    [right and left are the same identified faces]
    (hSecond : ∀ i, second i height = second i 0)
                                    [top and bottom are the same identified faces]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    G                               [the additive commutative group the values live in]
    ∀                               [for every single one]
    ∈                               [is one of the things in]
    LatticeFaceFlux G               [one value on every forward face of the lattice]
    `Finset.range`                  [the counting positions below a side's endpoint]

    The rectangle theorem first cancels all interior faces; closed-side identification then
    cancels the four remaining boundary contributions. -/
theorem closed_rectangle_flux_cancels_generic (first second : LatticeFaceFlux G)
    (width height : ℕ) (hFirst : ∀ k, first width k = first 0 k)
    (hSecond : ∀ i, second i height = second i 0) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      latticeCellDivergence first second i k) = 0 := by
  rw [rectangle_boundary_balance_generic]
  simp [hFirst, hSecond]

/-! ## The carrier readout on the finite lattice over any commutative ring -/

variable {R : Type*} [CommRing R]

/-- The generic first-direction flux, sampled on counting-lattice faces.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    R                               [the commutative ring the values live in]
    →                               [takes the left and gives the right]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    LatticeFaceFlux R               [one value on every forward face of the lattice]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    Each face is a sample of the generic `pointFlux` at the amplitude and the two phase samples,
    so no second flux formula is introduced. -/
def ringSampledFluxFirst (σ : R) (amp phase : R → R → R) (u : RingCell R) :
    LatticeFaceFlux R :=
  fun i j => pointFlux σ (amp (i : R) (j : R)) u (phase (i : R) (j : R))
    (phase ((i : R) + 1) (j : R))

/-- The density a phase cell carries, sampled at the lattice amplitude.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (amp : R → R → R)               [the amplitude field over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    This is the generic `pointDensity` at the amplitude; no second material field appears. -/
def ringCarrierDensity (σ : R) (amp : R → R → R) (u : RingCell R) (i : ℕ) : R :=
  pointDensity σ (amp (i : R) 0) u

/-- The carrier-derived flux on a lattice face.

    (σ : R)                         [the parameter: what the adjoined direction squares to]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (i : ℕ)                         [the forward face's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `pointFlux`                     [the first-direction flux: amount times phase difference]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    This is exactly `J = rho * Δφ`, sampled from `pointFlux`; it is not independent state. -/
def ringCarrierFaceFlux (σ : R) (amp phase : R → R → R) (u : RingCell R) (i : ℕ) : R :=
  pointFlux σ (amp (i : R) 0) u (phase (i : R) 0) (phase ((i : R) + 1) 0)

/-- The time-step-scaled outgoing-minus-incoming carrier flux at one cell.

    (τ σ : R)                       [the time step and algebra parameter]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringCarrierFaceFlux`           [the carrier-derived flux on a lattice face]

    Scaling the face-flux difference by `τ` makes this the amount transferred during one
    forward-Euler step. -/
def ringCarrierDivergence (τ σ : R) (amp phase : R → R → R) (u : RingCell R)
    (i : ℕ) : R :=
  τ * (ringCarrierFaceFlux σ amp phase u (i + 1) - ringCarrierFaceFlux σ amp phase u i)

/-- One explicit conservative material update of the carrier density.

    (τ σ : R)                       [the time step and algebra parameter]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringCarrierDensity`            [the density a phase cell carries at the amplitude]
    `ringCarrierDivergence`         [the scaled outgoing-minus-incoming carrier flux]

    The old norm-form density loses exactly its net outward phase-derived face transfer. -/
def ringEvolveCarrierDensity (τ σ : R) (amp phase : R → R → R) (u : RingCell R)
    (i : ℕ) : R :=
  ringCarrierDensity σ amp u i - ringCarrierDivergence τ σ amp phase u i

/-- The constructed update satisfies discrete continuity at every cell.

    (τ σ : R)                       [the time step and algebra parameter]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    `ringEvolveCarrierDensity`      [the conservative material update of the density]
    `ringCarrierDivergence`         [the scaled outgoing-minus-incoming carrier flux]

    This is `rho' - rho + div J = 0`, obtained by unfolding the forward-Euler definition. -/
theorem ringEvolveCarrierDensity_continuity (τ σ : R) (amp phase : R → R → R)
    (u : RingCell R) (i : ℕ) :
    ringEvolveCarrierDensity τ σ amp phase u i - ringCarrierDensity σ amp u i
      + ringCarrierDivergence τ σ amp phase u i = 0 := by
  rw [ringEvolveCarrierDensity]
  ring

/-- The sum of one-step divergences is the scaled endpoint flux difference.

    (τ σ : R)                       [the time step and algebra parameter]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (n : ℕ)                         [the number of cells]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ∈                               [is one of the things in]
    `ringCarrierDivergence`         [the scaled outgoing-minus-incoming carrier flux]
    `ringCarrierFaceFlux`           [the carrier-derived flux on a lattice face]
    `Finset.range`                  [the counting positions below the endpoint]

    Interior face terms cancel by `sum_forward_difference_generic`. -/
theorem sum_ringCarrierDivergence (τ σ : R) (amp phase : R → R → R) (u : RingCell R)
    (n : ℕ) :
    (∑ i ∈ Finset.range n, ringCarrierDivergence τ σ amp phase u i)
      = τ * (ringCarrierFaceFlux σ amp phase u n - ringCarrierFaceFlux σ amp phase u 0) := by
  simp_rw [ringCarrierDivergence]
  rw [← Finset.mul_sum]
  congr 1
  exact sum_forward_difference_generic (ringCarrierFaceFlux σ amp phase u) n

/-- A closed finite carrier lattice conserves total amount in one nontrivial update.

    (τ σ : R)                       [the time step and algebra parameter]
    (amp phase : R → R → R)         [the amplitude and phase fields over the ring]
    (u : RingCell R)                [the fixed phase direction]
    (n : ℕ)                         [the number of cells]
    (hLeft : ringCarrierFaceFlux σ amp phase u 0 = 0)
                                    [the left boundary flux is zero]
    (hRight : ringCarrierFaceFlux σ amp phase u n = 0)
                                    [the right boundary flux is zero]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    R                               [the commutative ring the values live in]
    RingCell R                      [a point: two ring elements side by side, a + b·x]
    ∈                               [is one of the things in]
    `ringEvolveCarrierDensity`      [the conservative material update of the density]
    `ringCarrierDensity`            [the density a phase cell carries at the amplitude]
    `Finset.range`                  [the counting positions below the endpoint]

    Telescoping removes every shared interior transfer; the two zero endpoint transfers leave
    the old and new finite sums equal. -/
theorem ringEvolveCarrierDensity_total_mass (τ σ : R) (amp phase : R → R → R)
    (u : RingCell R) (n : ℕ)
    (hLeft : ringCarrierFaceFlux σ amp phase u 0 = 0)
    (hRight : ringCarrierFaceFlux σ amp phase u n = 0) :
    (∑ i ∈ Finset.range n, ringEvolveCarrierDensity τ σ amp phase u i)
      = ∑ i ∈ Finset.range n, ringCarrierDensity σ amp u i := by
  simp_rw [ringEvolveCarrierDensity]
  rw [Finset.sum_sub_distrib, sum_ringCarrierDivergence]
  rw [hLeft, hRight]
  ring

end Physics
end Apeiron
