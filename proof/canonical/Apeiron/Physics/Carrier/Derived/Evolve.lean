import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier
import Apeiron.Physics.Carrier.Derived.Boundary
import Apeiron.Physics.CarrierFiniteAnyRing

/-!
# Conservative evolution of the carrier density

@name Conservative Carrier Evolution
@grade proved

This module gives one constructive material step on a finite one-dimensional sampling of the
amplitude/phase carrier. Cell amount is read from the generic `pointDensity`, the same read
`Carrier.density` makes. A face flux is read from the same amplitude and phase by the generic
`pointFlux`, the same read `Carrier.fluxFirst` makes, and a forward-Euler step subtracts the
difference of the two neighboring face transfers. Thus continuity is a consequence of the update,
not a zero-residual assumption.

The additive shape is rooted in `Apeiron.Physics.CarrierFiniteAnyRing`. `ringCarrierDensity`,
`ringCarrierFaceFlux`, `ringCarrierDivergence` and `ringEvolveCarrierDensity` are stated there over
an arbitrary commutative ring and read the generic `pointDensity` and `pointFlux`. Continuity, the
telescoped divergence and closed-boundary total conservation are their ring-universal laws. Every
declaration here is the specialisation of one of them to `ℝ`, so no law is re-proved.

## Supported regime, assumptions, error and work bounds

The regime is a one-dimensional, unit-spacing, single-valued phase carrier with one shared flux
per face. It supports phase-gradient transport with prescribed closed or open endpoint flux. In
particular it assumes no independently stored circulation: circulation is only what a
single-valued sampled phase can express. The step adds no physical dissipation; a dissipative
phase law or flux limiter would have to be supplied separately and would still have to preserve
the shared-face accounting. Positivity is not automatic: `carrierCFL` states the exact local
forward-Euler bound used below.

The update has first-order time truncation error and uses a first forward phase difference at
unit spacing. One step over `n` cells takes linear work and constant work per cell; the proofs add
no runtime operation. This is a readout-carrier evolution, not a general water solver: it has no
stored height, velocity or pressure, no viscosity model, and no global pressure solve.

`carrierCFL` and `evolveCarrierDensity_nonnegative_iff_cfl` are instance results for `ℝ`: they use
the order of the reals, which a general commutative ring does not carry.
-/

namespace Apeiron
namespace Physics

open Trichotomy

/-- Density sampled from the amplitude carrier on a one-dimensional counting lattice.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `ringCarrierDensity`            [the generic density a phase cell carries at the amplitude]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    No second material field is introduced: this value is the norm-form readout of `R`. This is
    `ringCarrierDensity` at `ℝ`. -/
noncomputable def carrierDensity (σ : ℝ) (R : PhaseField) (u : Cell) (i : ℕ) : ℝ :=
  ringCarrierDensity σ R u i

/-- The carrier-derived flux on a lattice face.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the forward face's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `ringCarrierFaceFlux`           [the generic carrier-derived flux on a lattice face]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    This is exactly `J = rho * Δφ`, sampled from the generic `pointFlux`; it is not independent
    state. This is `ringCarrierFaceFlux` at `ℝ`. -/
noncomputable def carrierFaceFlux (σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (i : ℕ) : ℝ :=
  ringCarrierFaceFlux σ R phase u i

/-- The time-step-scaled outgoing-minus-incoming carrier flux at one cell.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `ringCarrierDivergence`         [the generic scaled outgoing-minus-incoming carrier flux]

    Scaling the face-flux difference by `τ` makes this the amount transferred during one
    forward-Euler step. This is `ringCarrierDivergence` at `ℝ`. -/
noncomputable def carrierDivergence (τ σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (i : ℕ) : ℝ :=
  ringCarrierDivergence τ σ R phase u i

/-- One explicit conservative material update of the carrier density.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `ringEvolveCarrierDensity`      [the generic conservative material update of the density]

    The old norm-form density loses exactly its net outward phase-derived face transfer. This is
    `ringEvolveCarrierDensity` at `ℝ`. -/
noncomputable def evolveCarrierDensity (τ σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (i : ℕ) : ℝ :=
  ringEvolveCarrierDensity τ σ R phase u i

/-- The constructed update satisfies discrete continuity at every cell.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `ringEvolveCarrierDensity_continuity`
                                    [the generic update satisfies discrete continuity]

    This is `rho' - rho + div J = 0`, and it is the generic continuity law at `ℝ`. -/
theorem evolveCarrierDensity_continuity (τ σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (i : ℕ) :
    evolveCarrierDensity τ σ R u phase i - carrierDensity σ R u i
      + carrierDivergence τ σ R u phase i = 0 := by
  simpa only [evolveCarrierDensity, carrierDensity, carrierDivergence] using
    ringEvolveCarrierDensity_continuity τ σ R phase u i

/-- The sum of one-step divergences is the scaled endpoint flux difference.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (n : ℕ)                         [the number of cells]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ∈                               [is one of the things in]
    `sum_ringCarrierDivergence`     [the generic sum of divergences is the endpoint difference]
    `Finset.range`                  [the counting positions below the endpoint]

    Interior face terms cancel by the generic telescoping law, so this is
    `sum_ringCarrierDivergence` at `ℝ`. -/
theorem sum_carrierDivergence (τ σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (n : ℕ) :
    (∑ i ∈ Finset.range n, carrierDivergence τ σ R u phase i)
      = τ * (carrierFaceFlux σ R u phase n - carrierFaceFlux σ R u phase 0) := by
  simpa only [carrierDivergence, carrierFaceFlux] using
    sum_ringCarrierDivergence τ σ R phase u n

/-- A closed finite carrier lattice conserves total mass in one nontrivial update.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (n : ℕ)                         [the number of cells]
    (hLeft : carrierFaceFlux σ R u phase 0 = 0)   [the left boundary flux is zero]
    (hRight : carrierFaceFlux σ R u phase n = 0)  [the right boundary flux is zero]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ∈                               [is one of the things in]
    `ringEvolveCarrierDensity_total_mass`
                                    [the generic closed lattice conserves total amount]
    `Finset.range`                  [the counting positions below the endpoint]

    Telescoping removes every shared interior transfer; the two zero endpoint transfers leave
    the old and new finite sums equal. This is `ringEvolveCarrierDensity_total_mass` at `ℝ`. -/
theorem evolveCarrierDensity_total_mass (τ σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (n : ℕ)
    (hLeft : carrierFaceFlux σ R u phase 0 = 0)
    (hRight : carrierFaceFlux σ R u phase n = 0) :
    (∑ i ∈ Finset.range n, evolveCarrierDensity τ σ R u phase i)
      = ∑ i ∈ Finset.range n, carrierDensity σ R u i := by
  simpa only [evolveCarrierDensity, carrierDensity, carrierFaceFlux] using
    ringEvolveCarrierDensity_total_mass τ σ R phase u n
      (by simpa only [carrierFaceFlux] using hLeft)
      (by simpa only [carrierFaceFlux] using hRight)

/-- The explicit local CFL condition for retaining nonnegative carrier density.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    Cell                            [a point: two real numbers written side by side, a + b·x]

    The net outward transfer must not exceed the old norm-form amount. Incoming net transfer
    satisfies this automatically; excessive outgoing transfer must be limited or the step made
    smaller. This is an instance result for the ordered reals and has no general-ring analogue. -/
def carrierCFL (τ σ : ℝ) (R : PhaseField) (u : Cell) (phase : PhaseField) (i : ℕ) : Prop :=
  carrierDivergence τ σ R u phase i ≤ carrierDensity σ R u i

/-- The CFL condition is exactly what makes the constructed next density nonnegative.

    (τ σ : ℝ)                      [the time step and algebra parameter]
    (R phase : PhaseField)          [the amplitude and phase fields]
    (u : Cell)                      [the fixed phase direction]
    (i : ℕ)                         [the cell's counting-lattice coordinate]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ↔                               [exactly when, true in both directions]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    `evolveCarrierDensity`          [the conservative material update of the density]
    `carrierCFL`                    [the explicit local bound for nonnegative updated density]

    Positivity is therefore exposed as a checkable hypothesis rather than silently assumed. This
    uses the order of the reals, which a general commutative ring does not carry. -/
theorem evolveCarrierDensity_nonnegative_iff_cfl (τ σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (i : ℕ) :
    0 ≤ evolveCarrierDensity τ σ R u phase i ↔ carrierCFL τ σ R u phase i := by
  simp only [evolveCarrierDensity, carrierCFL, carrierDensity, carrierDivergence]
  exact sub_nonneg

end Physics
end Apeiron
