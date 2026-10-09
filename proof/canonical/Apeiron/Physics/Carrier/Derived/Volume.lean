import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Trichotomy.Basic
import Apeiron.Physics.CarrierVolumeAnyRing

/-!
# Three-dimensional carrier volume accounting

@name Carrier Volume Accounting
@cite kocik-1999
@grade proved

This module is the real specialisation of `Apeiron.Physics.CarrierVolumeAnyRing`. The generic
module states the three-dimensional carrier accounting over an arbitrary commutative ring and the
shared-face telescoping over an additive commutative group; every algebraic law here is that law
at `ℝ`. The amplitude `R` and phase `phase` are **extended** sample fields on `ℕ × ℕ × ℕ`: index 0
is a ghost layer on every axis, and logical cell `(x,y,z)` reads the sample at `(x+1,y+1,z+1)`.
`sampleDensity3` is the trichotomy norm form of that shifted sample, and
`sampleDensity3_eq_ringSampleDensity3` identifies it with the generic `ringSampleDensity3` read
from `CarrierAnyRing.pointDensity`.

Face `q` on the x axis is the shared face between logical cells `q-1` and `q` (with cell `-1` the
ghost). It carries one stored value read from the extended amplitude at `(q,y+1,z+1)` and the
forward phase difference `(q,y+1,z+1) → (q+1,y+1,z+1)`. For `q ≥ 1` that amplitude is the lower
logical cell's, so the value is at once the outgoing x-read of cell `q-1` and the incoming x-read
of cell `q`; for `q = 0` it is the explicit boundary ghost. The y and z faces shift their
transverse coordinates the same way. No separate face-amplitude field is posited: a symmetric face
amplitude would be new material state, and the one-dimensional `FaceFlux` already fixes the honest
single-valued convention. The boundary ghost amplitude is an explicit extended sample, not a silent
assumption: the closed-boundary theorems identify opposite faces rather than choose a ghost value,
and the Rust fixture documents its clamped ghost. Nothing here pretends that cell-local unequal
amplitudes give one shared face.

## Supported regime, assumptions, error and work bounds

The regime is a three-dimensional, unit-spacing, single-valued phase carrier with one shared flux
per face. It supports phase-gradient transport with prescribed closed or open face flux. In
particular it assumes no independently stored circulation: circulation is only what a
single-valued sampled phase can express. The step adds no physical dissipation and no body work or
external forcing beyond the explicit source term of the accounting theorem; it has no viscosity
model and no pressure projection. A dissipative phase law or flux limiter would have to be supplied
separately and would still have to preserve the shared-face accounting. Positivity is not
automatic: `carrierCFL3` states the exact local forward-Euler bound used below.

One evolution step over a `width × height × depth` box takes work proportional to
`width * height * depth`, with constant work per cell; the proofs add no runtime operation. The
update has first-order time truncation error and uses first forward phase differences at unit
spacing. This is readout-carrier volume accounting, not a volumetric water, cave, or general fluid
solver: it has no stored height, velocity, pressure, viscosity, surface tension, or global pressure
solve.

## Root

`sampleDensity3` is the real trichotomy norm form of an extended sample, and
`sampleDensity3_eq_ringSampleDensity3` identifies it with the generic `ringSampleDensity3` read from
`CarrierAnyRing.pointDensity`; `sampleDensity3_eq_amplitude_sq` is then the generic
`ringSampleDensity3_eq_amplitude_sq` at `ℝ`. `carrierDensity3`, `faceFluxX`, `faceFluxY`,
`faceFluxZ`, `cellDivergence3`, `carrierDivergence3` and `evolveCarrierDensity3` are the generic
declarations at `ℝ`, and `box_divergence_balance`, `closed_box_flux_cancels`,
`sum_carrierDivergence3`, `evolveCarrierDensity3_continuity` and
`evolveCarrierDensity3_total_mass` are their generic laws at `ℝ`. `carrierDensity3_nonnegative`,
`carrierCFL3` and `evolveCarrierDensity3_nonnegative_iff_cfl` are the ordered-real instance results
and have no general-ring analogue.
-/

namespace Apeiron
namespace Physics

open Trichotomy
open Trichotomy.AnyRing

/-- A real amplitude or phase field over the three-dimensional counting carrier.

    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    Three counting coordinates in, one real value out: the shape both the amplitude and the phase
    have. Index 0 is a ghost layer. This is `RingPhaseField3 ℝ`. -/
abbrev PhaseField3 : Type := RingPhaseField3 ℝ

/-- One real flux value on every forward face of a three-dimensional counting lattice.

    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    The first input is a face coordinate; the next two are the transverse logical coordinates. The
    value is the one amount carried there, so sharing this function makes a face single-valued.
    This is `LatticeFaceFlux3 ℝ`. -/
abbrev FaceFlux3 : Type := LatticeFaceFlux3 ℝ

/-- The trichotomy norm-form amount of an arbitrary extended amplitude sample.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (a b c : ℕ)                     [an extended sample coordinate]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    Index 0 on any axis is a ghost layer; this is the size of the sample there, and it is the
    amplitude authority every face flux is read from. -/
noncomputable def sampleDensity3 (σ : ℝ) (R : PhaseField3) (u : Cell) (a b c : ℕ) : ℝ :=
  normForm σ (R a b c • u)

/-- The real sampled density is the generic CarrierAnyRing point density.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (a b c : ℕ)                     [an extended sample coordinate]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `ringSampleDensity3`            [the generic amount at one extended sample]

    So the real model is the generic `ringSampleDensity3` at `ℝ`, not a second readout. -/
theorem sampleDensity3_eq_ringSampleDensity3 (σ : ℝ) (R : PhaseField3) (u : Cell)
    (a b c : ℕ) : sampleDensity3 σ R u a b c = ringSampleDensity3 σ R u a b c := by
  simp only [sampleDensity3, ringSampleDensity3, pointDensity, ringNormForm, normForm,
    Prod.smul_fst, Prod.smul_snd, smul_eq_mul]
  ring

/-- Density sampled from the amplitude carrier at one logical cell.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    Logical cell `(x,y,z)` reads the extended sample at `(x+1,y+1,z+1)`; no second material field
    is introduced. This is `ringCarrierDensity3` at `ℝ`. -/
noncomputable def carrierDensity3 (σ : ℝ) (R : PhaseField3) (u : Cell) (x y z : ℕ) : ℝ :=
  ringCarrierDensity3 σ R u x y z

/-- The logical cell density is the shifted extended sample.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    The shift is the one the generic `ringCarrierDensity3` already carries. -/
theorem carrierDensity3_eq_sampleDensity3 (σ : ℝ) (R : PhaseField3) (u : Cell) (x y z : ℕ) :
    carrierDensity3 σ R u x y z = sampleDensity3 σ R u (x + 1) (y + 1) (z + 1) := by
  simp only [carrierDensity3, ringCarrierDensity3, sampleDensity3_eq_ringSampleDensity3]

/-- The extended sample density is the amplitude squared, by the direction's unit size.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (a b c : ℕ)                     [an extended sample coordinate]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    sampleDensity3 σ R u a b c = R a b c * R a b c   [the amount is the sample times itself]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    The stretch comes out squared and the unit direction leaves a factor one. This is the generic
    `ringSampleDensity3_eq_amplitude_sq` at `ℝ`. -/
theorem sampleDensity3_eq_amplitude_sq (σ : ℝ) (R : PhaseField3) (u : Cell) (a b c : ℕ)
    (hu : normForm σ u = 1) : sampleDensity3 σ R u a b c = R a b c * R a b c := by
  rw [sampleDensity3_eq_ringSampleDensity3]
  exact ringSampleDensity3_eq_amplitude_sq σ R u a b c hu

/-- The logical cell density is the shifted amplitude squared, by the direction's unit size.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    carrierDensity3 σ R u x y z
      = R (x+1) (y+1) (z+1) * R (x+1) (y+1) (z+1)   [the amount is the shifted sample times itself]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    The same squared stretch as the extended sample, read at the logical cell's shifted coordinate.
    This is the generic `ringCarrierDensity3_eq_amplitude_sq` at `ℝ`. -/
theorem carrierDensity3_eq_amplitude_sq (σ : ℝ) (R : PhaseField3) (u : Cell) (x y z : ℕ)
    (hu : normForm σ u = 1) :
    carrierDensity3 σ R u x y z
      = R (x + 1) (y + 1) (z + 1) * R (x + 1) (y + 1) (z + 1) := by
  simpa only [carrierDensity3] using ringCarrierDensity3_eq_amplitude_sq σ R u x y z hu

/-- The single-valued forward face flux in the x direction.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (q y z : ℕ)                     [the face coordinate and its two transverse logical coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    Face `q` is the shared face between logical cells `q-1` and `q`. It reads the extended sample
    at `(q,y+1,z+1)` and the forward phase difference `(q,y+1,z+1) → (q+1,y+1,z+1)`, so for `q ≥ 1`
    it is the outgoing x-read of the lower cell and the incoming x-read of the upper cell at once.
    This is `ringFaceFluxX` at `ℝ`. -/
noncomputable def faceFluxX (σ : ℝ) (R : PhaseField3) (u : Cell) (phase : PhaseField3)
    (q y z : ℕ) : ℝ :=
  ringFaceFluxX σ R u phase q y z

/-- The single-valued forward face flux in the y direction.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x q z : ℕ)                     [the face coordinate and its two transverse logical coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    The y-axis twin of `faceFluxX`: face `q` reads the extended sample at `(x+1,q,z+1)` and the
    forward phase difference `(x+1,q,z+1) → (x+1,q+1,z+1)`. This is `ringFaceFluxY` at `ℝ`. -/
noncomputable def faceFluxY (σ : ℝ) (R : PhaseField3) (u : Cell) (phase : PhaseField3)
    (x q z : ℕ) : ℝ :=
  ringFaceFluxY σ R u phase x q z

/-- The single-valued forward face flux in the z direction.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y q : ℕ)                     [the face coordinate and its two transverse logical coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    The z-axis twin of `faceFluxX`: face `q` reads the extended sample at `(x+1,y+1,q)` and the
    forward phase difference `(x+1,y+1,q) → (x+1,y+1,q+1)`. This is `ringFaceFluxZ` at `ℝ`. -/
noncomputable def faceFluxZ (σ : ℝ) (R : PhaseField3) (u : Cell) (phase : PhaseField3)
    (x y q : ℕ) : ℝ :=
  ringFaceFluxZ σ R u phase x y q

/-- The outgoing-minus-incoming flux of one product cell across its six faces.

    (first second third : FaceFlux3) [the single-valued fluxes on the three axes]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℝ                               [the real numbers: every number on the number line]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    FaceFlux3                       [one real value on every forward lattice face]

    Each incoming face is the preceding cell's outgoing face, with the opposite sign. This is
    `latticeCellDivergence3` at `ℝ`. -/
def cellDivergence3 (first second third : FaceFlux3) (x y z : ℕ) : ℝ :=
  latticeCellDivergence3 first second third x y z

/-- The time-step-scaled outgoing-minus-incoming carrier flux at one cell.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    Scaling the six-face difference by `τ` makes this the amount transferred during one
    forward-Euler step. This is `ringCarrierDivergence3` at `ℝ`. -/
noncomputable def carrierDivergence3 (τ σ : ℝ) (R : PhaseField3) (u : Cell) (phase : PhaseField3)
    (x y z : ℕ) : ℝ :=
  ringCarrierDivergence3 τ σ R u phase x y z

/-- One explicit conservative material update of the carrier density.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    The old norm-form density of the same logical cell loses exactly its net outward
    phase-derived six-face transfer. This is `ringEvolveCarrierDensity3` at `ℝ`. -/
noncomputable def evolveCarrierDensity3 (τ σ : ℝ) (R : PhaseField3) (u : Cell)
    (phase : PhaseField3) (x y z : ℕ) : ℝ :=
  ringEvolveCarrierDensity3 τ σ R u phase x y z

/-- The constructed update satisfies discrete continuity at every cell.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]

    This is `rho' - rho + div J = 0`, and it is the generic
    `ringEvolveCarrierDensity3_continuity` at `ℝ`. -/
theorem evolveCarrierDensity3_continuity (τ σ : ℝ) (R : PhaseField3) (u : Cell)
    (phase : PhaseField3) (x y z : ℕ) :
    evolveCarrierDensity3 τ σ R u phase x y z - carrierDensity3 σ R u x y z
      + carrierDivergence3 τ σ R u phase x y z = 0 := by
  simpa only [evolveCarrierDensity3, carrierDensity3, carrierDivergence3] using
    ringEvolveCarrierDensity3_continuity τ σ R u phase x y z

/-- The sum of cell divergences in a finite box is its six oriented outer faces.

    (first second third : FaceFlux3) [the single-valued fluxes on the three axes]
    (width height depth : ℕ)        [the box's three side lengths]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∈                               [is one of the things in]
    FaceFlux3                       [one real value on every forward lattice face]

    Interior faces disappear solely because every occurrence is read from the same face field.
    This is `box_divergence_balance_generic` at `ℝ`. -/
theorem box_divergence_balance (first second third : FaceFlux3) (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        cellDivergence3 first second third x y z)
      = (∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth, (first width y z - first 0 y z))
        + (∑ x ∈ Finset.range width, ∑ z ∈ Finset.range depth,
            (second x height z - second x 0 z))
        + ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height,
            (third x y depth - third x y 0) := by
  simpa only [cellDivergence3] using
    box_divergence_balance_generic first second third width height depth

/-- A closed box has zero total divergence when each pair of opposite faces is one flux.

    (first second third : FaceFlux3) [the single-valued fluxes on the three axes]
    (width height depth : ℕ)        [the box's three side lengths]
    (hFirst : ∀ y z, first width y z = first 0 y z)   [right and left x faces are identified]
    (hSecond : ∀ x z, second x height z = second x 0 z) [top and bottom y faces are identified]
    (hThird : ∀ x y, third x y depth = third x y 0)   [front and back z faces are identified]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∈                               [is one of the things in]
    ∀                               [for every single one]
    FaceFlux3                       [one real value on every forward lattice face]

    The box theorem first cancels all interior faces; closed-side identification then cancels the
    six remaining boundary contributions. This is `closed_box_flux_cancels_generic` at `ℝ`. -/
theorem closed_box_flux_cancels (first second third : FaceFlux3) (width height depth : ℕ)
    (hFirst : ∀ y z, first width y z = first 0 y z)
    (hSecond : ∀ x z, second x height z = second x 0 z)
    (hThird : ∀ x y, third x y depth = third x y 0) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
      cellDivergence3 first second third x y z) = 0 := by
  simpa only [cellDivergence3] using
    closed_box_flux_cancels_generic first second third width height depth hFirst hSecond hThird

/-- The sum of one-step carrier divergences is the scaled six-face boundary sum.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (width height depth : ℕ)        [the box's three side lengths]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∈                               [is one of the things in]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    Interior face terms cancel by `box_divergence_balance`; only the oriented outer faces
    survive. This is `ringSum_carrierDivergence3` at `ℝ`. -/
theorem sum_carrierDivergence3 (τ σ : ℝ) (R : PhaseField3) (u : Cell) (phase : PhaseField3)
    (width height depth : ℕ) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        carrierDivergence3 τ σ R u phase x y z)
      = τ * ((∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
                (faceFluxX σ R u phase width y z - faceFluxX σ R u phase 0 y z))
          + (∑ x ∈ Finset.range width, ∑ z ∈ Finset.range depth,
                (faceFluxY σ R u phase x height z - faceFluxY σ R u phase x 0 z))
          + ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height,
                (faceFluxZ σ R u phase x y depth - faceFluxZ σ R u phase x y 0)) := by
  simpa only [carrierDivergence3, faceFluxX, faceFluxY, faceFluxZ] using
    ringSum_carrierDivergence3 τ σ R u phase width height depth

/-- A closed finite carrier box conserves total density in one nontrivial update.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (width height depth : ℕ)        [the box's three side lengths]
    (hFirst : ∀ y z, faceFluxX σ R u phase width y z = faceFluxX σ R u phase 0 y z)
                                    [right and left x faces are identified]
    (hSecond : ∀ x z, faceFluxY σ R u phase x height z = faceFluxY σ R u phase x 0 z)
                                    [top and bottom y faces are identified]
    (hThird : ∀ x y, faceFluxZ σ R u phase x y depth = faceFluxZ σ R u phase x y 0)
                                    [front and back z faces are identified]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∈                               [is one of the things in]
    ∀                               [for every single one]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    Telescoping removes every shared interior transfer; the identified outer transfers leave the
    old and new finite sums equal. This is `ringEvolveCarrierDensity3_total_mass` at `ℝ`. -/
theorem evolveCarrierDensity3_total_mass (τ σ : ℝ) (R : PhaseField3) (u : Cell)
    (phase : PhaseField3) (width height depth : ℕ)
    (hFirst : ∀ y z, faceFluxX σ R u phase width y z = faceFluxX σ R u phase 0 y z)
    (hSecond : ∀ x z, faceFluxY σ R u phase x height z = faceFluxY σ R u phase x 0 z)
    (hThird : ∀ x y, faceFluxZ σ R u phase x y depth = faceFluxZ σ R u phase x y 0) :
    (∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
        evolveCarrierDensity3 τ σ R u phase x y z)
      = ∑ x ∈ Finset.range width, ∑ y ∈ Finset.range height, ∑ z ∈ Finset.range depth,
          carrierDensity3 σ R u x y z := by
  have hF : ∀ y z, ringFaceFluxX σ R u phase width y z = ringFaceFluxX σ R u phase 0 y z := by
    intro y z
    simpa only [faceFluxX] using hFirst y z
  have hS : ∀ x z, ringFaceFluxY σ R u phase x height z = ringFaceFluxY σ R u phase x 0 z := by
    intro x z
    simpa only [faceFluxY] using hSecond x z
  have hT : ∀ x y, ringFaceFluxZ σ R u phase x y depth = ringFaceFluxZ σ R u phase x y 0 := by
    intro x y
    simpa only [faceFluxZ] using hThird x y
  simpa only [evolveCarrierDensity3, carrierDensity3] using
    ringEvolveCarrierDensity3_total_mass τ σ R u phase width height depth hF hS hT

/-- The logical cell density is never negative.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    0 ≤ carrierDensity3 σ R u x y z [the amount is at or above zero]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ≤                               [is smaller than or equal to]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    A square on every axis, so the wet/dry seam is exactly amplitude zero. This is an instance
    result for `ℝ` and has no general-ring analogue. -/
theorem carrierDensity3_nonnegative (σ : ℝ) (R : PhaseField3) (u : Cell) (x y z : ℕ)
    (hu : normForm σ u = 1) : 0 ≤ carrierDensity3 σ R u x y z := by
  rw [carrierDensity3_eq_amplitude_sq σ R u x y z hu]
  exact mul_self_nonneg _

/-- The explicit local CFL condition for retaining nonnegative carrier density.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    The net outward transfer must not exceed the old norm-form amount. Incoming net transfer
    satisfies this automatically; excessive outgoing transfer must be limited or the step made
    smaller. This is an instance result for the ordered reals. -/
def carrierCFL3 (τ σ : ℝ) (R : PhaseField3) (u : Cell) (phase : PhaseField3)
    (x y z : ℕ) : Prop :=
  carrierDivergence3 τ σ R u phase x y z ≤ carrierDensity3 σ R u x y z

/-- The CFL condition is exactly what makes the constructed next density nonnegative.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ↔                               [exactly when, true in both directions]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    Positivity is therefore exposed as a checkable hypothesis rather than silently assumed. This
    uses the order of the reals, which a general commutative ring does not carry. -/
theorem evolveCarrierDensity3_nonnegative_iff_cfl (τ σ : ℝ) (R : PhaseField3) (u : Cell)
    (phase : PhaseField3) (x y z : ℕ) :
    0 ≤ evolveCarrierDensity3 τ σ R u phase x y z ↔ carrierCFL3 τ σ R u phase x y z := by
  simp only [evolveCarrierDensity3, carrierCFL3, carrierDensity3, carrierDivergence3,
    ringEvolveCarrierDensity3]
  exact sub_nonneg

end Physics
end Apeiron
