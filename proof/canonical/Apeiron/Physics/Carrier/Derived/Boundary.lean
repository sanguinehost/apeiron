import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier
import Apeiron.Physics.CarrierFiniteAnyRing

/-!
# Finite carrier boundary accounting

@name Carrier Rectangle Boundary Accounting
@grade proved

A carrier lab records one value on each oriented face. Because neighboring cells read the same
face value, adding their discrete divergences cancels every interior face. Only the four outer
sides of a finite rectangle remain. If opposite outer sides are identified, those remaining
terms cancel as well.

This is the finite algebraic closure needed by `Carrier`: `sampledFluxFirst` samples the derived
flux on the counting lattice, while the accounting theorem applies to any two such single-valued
face fields. No independent edge enumeration or analytic assumption is required.

The accounting is rooted in `Apeiron.Physics.CarrierFiniteAnyRing`. `LatticeFaceFlux`,
`latticeCellDivergence`, `sum_forward_difference_generic`, `rectangle_boundary_balance_generic`
and `closed_rectangle_flux_cancels_generic` are stated there over an additive commutative group,
and every declaration here is its specialisation to `ℝ`. `sampledFluxFirst` samples the generic
`ringSampledFluxFirst`, so the single flux formula stays in `CarrierAnyRing`.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- A single real flux value on every forward face of a counting lattice.

    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right]

    The first input chooses a line and the second chooses a face on that line. Sharing this
    function between adjacent cells is what makes an interior face single-valued. This is
    `LatticeFaceFlux` at `ℝ`. -/
abbrev FaceFlux : Type := LatticeFaceFlux ℝ

/-- The carrier's derived first-direction flux, sampled on counting-lattice faces.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    `ringSampledFluxFirst`          [the generic first-direction flux sampled on lattice faces]
    `fluxFirst`                     [the first-direction flux: amount times phase slope]
    `FaceFlux`                      [one real value on every forward lattice face]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    This introduces no new flux data: each face is the generic `pointFlux` read at the amplitude
    and the two phase samples, the same read `Carrier.fluxFirst` makes. -/
noncomputable def sampledFluxFirst (σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) : FaceFlux :=
  ringSampledFluxFirst σ R phase u

/-- The outgoing-minus-incoming flux of one product cell.

    (first second : FaceFlux)      [the single-valued fluxes in the two directions]
    (i j : ℕ)                      [the cell's counting-lattice coordinates]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    `FaceFlux`                      [one real value on every forward lattice face]
    `latticeCellDivergence`         [outgoing flux minus incoming flux in both directions]

    Each incoming face is the preceding cell's outgoing face, with the opposite sign. This is
    the generic `latticeCellDivergence` at `ℝ`. -/
abbrev cellDivergence (first second : FaceFlux) (i j : ℕ) : ℝ :=
  latticeCellDivergence first second i j

/-- A finite line of forward differences leaves only its two endpoint faces.

    (f : ℕ → ℝ)                    [one single-valued flux along a line]
    (n : ℕ)                        [the number of cells in the line]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right]
    ∈                               [is one of the things in]
    `Finset.range`                  [the counting positions below the endpoint]

    Every intermediate value occurs once positively and once negatively, so it cancels. This is
    `sum_forward_difference_generic` at `ℝ`. -/
theorem sum_forward_difference (f : ℕ → ℝ) (n : ℕ) :
    (∑ i ∈ Finset.range n, (f (i + 1) - f i)) = f n - f 0 :=
  sum_forward_difference_generic f n

/-- The sum of cell divergences in a finite rectangle is its four oriented outer sides.

    (first second : FaceFlux)      [the single-valued fluxes in the two directions]
    (width height : ℕ)             [the rectangle's two side lengths]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∈                               [is one of the things in]
    `FaceFlux`                      [one real value on every forward lattice face]
    `cellDivergence`               [outgoing flux minus incoming flux in both directions]
    `Finset.range`                  [the counting positions below a side's endpoint]

    The first sum is right minus left; the second is top minus bottom. Interior faces disappear
    solely because every occurrence is read from the same face field. This is
    `rectangle_boundary_balance_generic` at `ℝ`. -/
theorem rectangle_boundary_balance (first second : FaceFlux) (width height : ℕ) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
        cellDivergence first second i k)
      = (∑ k ∈ Finset.range height, (first width k - first 0 k))
        + ∑ i ∈ Finset.range width, (second i height - second i 0) :=
  rectangle_boundary_balance_generic first second width height

/-- A closed rectangle has zero total divergence when each pair of opposite sides is one flux.

    (first second : FaceFlux)      [the single-valued fluxes in the two directions]
    (width height : ℕ)             [the rectangle's two side lengths]
    (hFirst : ∀ k, first width k = first 0 k)
                                    [right and left are the same identified faces]
    (hSecond : ∀ i, second i height = second i 0)
                                    [top and bottom are the same identified faces]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∀                               [for every single one]
    ∈                               [is one of the things in]
    `FaceFlux`                      [one real value on every forward lattice face]
    `cellDivergence`               [outgoing flux minus incoming flux in both directions]
    `Finset.range`                  [the counting positions below a side's endpoint]

    The rectangle theorem first cancels all interior faces; closed-side identification then
    cancels the four remaining boundary contributions. This is
    `closed_rectangle_flux_cancels_generic` at `ℝ`. -/
theorem closed_rectangle_flux_cancels (first second : FaceFlux) (width height : ℕ)
    (hFirst : ∀ k, first width k = first 0 k)
    (hSecond : ∀ i, second i height = second i 0) :
    (∑ i ∈ Finset.range width, ∑ k ∈ Finset.range height,
      cellDivergence first second i k) = 0 :=
  closed_rectangle_flux_cancels_generic first second width height hFirst hSecond

end Physics
end Apeiron
