import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.CarrierAnyRing
import Apeiron.Physics.Carrier.Derived.ScalarKernelAnyRing
import Apeiron.Trichotomy.Basic

/-!
# The carrier readout: density and phase flux

@name Product Carrier Readout
@cite kocik-1999
@grade proved

A phase cell is a point of the algebra: the amplitude at a place, stretching the face's
size-one direction. Its **size**, the `Trichotomy.normForm` of that point, is the Born density,
and the density carried by the phase's forward difference is the first-direction phase flux.
This file derives both from the trichotomy rather than positing `R²`.

The readout is installed once, generically, in `Apeiron.Physics.CarrierAnyRing`. `wavePoint` is
the algebra's own product `ringMul σ (R x y, 0) u`. It agrees with the library's slot-by-slot
stretch `R x y • u` by the generic `ringMul_mk_zero_eq_smul`, but the equality is propositional
rather than definitional, so the product is kept as the one construction the algebra's laws are
stated over instead of being bridged at every use.
`density` is the generic `pointDensity` at the amplitude, and `fluxFirst` is the generic
`pointFlux` at the amplitude and the two phase samples. `density_eq_amplitude_sq` is the generic
`pointDensity_eq` with the direction's unit size, so the stretch comes out squared and the unit
direction leaves a factor one. The concrete direction is the algebra's flow, `flow σ φ`; its unit
size `normForm_flow` and the instantiation are in the companion
`Apeiron/Physics/CarrierFlow.lean`, kept separate so this file stays on the trichotomy core.

`wordDensity` and `wordFluxFirst` are the integer-lattice samples the Aeneas-extracted
`physics_math.density` and `physics_math.phase_flux` compute. They specialise the generic readout
at the unit direction `ringOne`, where `pointDensity_ringOne` makes it the amplitude's square and
`pointDensity_ringOne_independent` shows the dial does not enter, so the machine-word kernel's
choice of dial is immaterial rather than assumed. The amplitude is kept as a signed integer rather
than a `u16`, which is what lets an amplitude sign survive a refinement.

The generic scalar update kernel `scalarCarrierStep` and its exact-half flux witnesses live in
`Apeiron/Physics/CarrierScalarKernelAnyRing.lean`, imported here so the carrier family reaches
them from this root.

`density_nonnegative` and `density_eq_zero_iff` are instance results for `ℝ`. They use the order
and the lack of zero divisors of the reals, and they do not transfer to a ring such as
`ZMod (2 ^ 64)`.

## Root

`wavePoint` is `Trichotomy.AnyRing.ringMul` of the amplitude point and the direction; `density`
is the generic `pointDensity` read off it, and `density_eq_amplitude_sq` rides the generic
`pointDensity_eq` and the direction's unit-size hypothesis. `fluxFirst` is the generic
`pointFlux`, which multiplies the density by the phase's forward difference.
-/

namespace Apeiron
namespace Physics

open Trichotomy
open Trichotomy.AnyRing

/-- A real amplitude or phase field over the Cartesian carrier.

    ℝ                               [the real numbers: every number on the number line]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]

    Two real coordinates in, one real value out: the shape both the amplitude and the phase
    have. -/
abbrev PhaseField : Type := ℝ → ℝ → ℝ

/-- The phase cell: the amplitude at a place stretching a fixed phase direction.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (x y : ℝ)                       [two real coordinates]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℝ                               [the real numbers: every number on the number line]

    The one point the density is read off. The stretch is the algebra's product
    `ringMul σ (R x y, 0) u`, which agrees with the library's slot-by-slot `R x y • u` by
    `ringMul_mk_zero_eq_smul` but not definitionally, so the product is written out as the one
    construction the algebra's laws are stated over. -/
noncomputable def wavePoint (σ : ℝ) (R : PhaseField) (u : Cell) (x y : ℝ) : Cell :=
  ringMul σ (R x y, 0) u

/-- The Born density: the generic point density at the amplitude.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (x y : ℝ)                       [two real coordinates]
    ℝ                               [the real numbers: every number on the number line]
    `density`                       [the amount: the size of the phase cell at a place]
    `pointDensity`                  [the amount: the size of the phase cell at a place]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]
    `wavePoint`                     [the phase cell: the amplitude stretching a phase direction]

    The real readout is the generic `pointDensity` at the amplitude `R x y`. It is the algebra's
    own size of the phase cell, not a second field. -/
noncomputable def density (σ : ℝ) (R : PhaseField) (u : Cell) (x y : ℝ) : ℝ :=
  pointDensity σ (R x y) u

/-- The density reads the authoritative wave point.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (x y : ℝ)                       [two real coordinates]
    density σ R u x y = normForm σ (wavePoint σ R u x y)
                                    [the real amount equals the size of the wave point]
    `density`                       [the amount: the size of the phase cell at a place]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `wavePoint`                     [the phase cell: the amplitude stretching a phase direction]

    By definition: `wavePoint` is `ringMul σ (R x y, 0) u`, which is what `pointDensity`
    measures, so this is a definitional identity rather than a bridge. -/
theorem density_eq_normForm_wavePoint (σ : ℝ) (R : PhaseField) (u : Cell) (x y : ℝ) :
    density σ R u x y = normForm σ (wavePoint σ R u x y) := rfl

/-- The density is the amplitude squared, by the direction's unit size.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (x y : ℝ)                       [two real coordinates]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    density σ R u x y = R x y * R x y   [the amount is the amplitude at the place, times itself]
    ℝ                               [the real numbers: every number on the number line]
    `density`                       [the amount: the size of the phase cell at a place]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    The stretch comes out squared and the unit direction leaves a factor one, so the generic
    `pointDensity_eq` collapses to `R²`. The amount is the size of the wave point, not a posited
    square. -/
theorem density_eq_amplitude_sq (σ : ℝ) (R : PhaseField) (u : Cell) (x y : ℝ)
    (hu : normForm σ u = 1) : density σ R u x y = R x y * R x y := by
  rw [density, pointDensity_eq]
  rw [show ringNormForm σ u = 1 from hu, mul_one]
  simp only [ringNormForm]
  ring

/-- The forward difference of a phase field in the first direction.

    (phase : PhaseField)            [a real phase field]
    (x y : ℝ)                       [two real coordinates]
    ℝ                               [the real numbers: every number on the number line]
    `slopeFirst`                    [the first forward difference: next place minus this place]

    The phase slope the flux is carried by. -/
noncomputable def slopeFirst (phase : PhaseField) (x y : ℝ) : ℝ := phase (x + 1) y - phase x y

/-- The first-direction phase flux: the generic point flux at the amplitude and the two samples.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (phase : PhaseField)            [a real phase field]
    (x y : ℝ)                       [two real coordinates]
    ℝ                               [the real numbers: every number on the number line]
    `fluxFirst`                     [the first-direction flux: amount times phase slope]
    `pointFlux`                     [the first-direction flux: amount times phase difference]
    `density`                       [the amount: the size of the phase cell at a place]

    The generic `pointFlux` at the amplitude `R x y` and the phase samples `phase x y` and
    `phase (x + 1) y`. The single phase fixes the flux's direction; no independent momentum is
    supplied. -/
noncomputable def fluxFirst (σ : ℝ) (R : PhaseField) (u : Cell) (phase : PhaseField)
    (x y : ℝ) : ℝ :=
  pointFlux σ (R x y) u (phase x y) (phase (x + 1) y)

/-- The first-direction flux is the amplitude squared times the first phase difference.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (phase : PhaseField)            [a real phase field]
    (x y : ℝ)                       [two real coordinates]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    fluxFirst σ R u phase x y
      = R x y * R x y * (phase (x + 1) y - phase x y)   [the flux is amount times phase
                                                          difference]
    ℝ                               [the real numbers: every number on the number line]
    `fluxFirst`                     [the first-direction flux: amount times phase slope]
    `density`                       [the amount: the size of the phase cell at a place]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    Density times the one phase's forward difference, with the density itself the norm form of
    the phase cell. -/
theorem fluxFirst_eq (σ : ℝ) (R : PhaseField) (u : Cell) (phase : PhaseField) (x y : ℝ)
    (hu : normForm σ u = 1) :
    fluxFirst σ R u phase x y = R x y * R x y * (phase (x + 1) y - phase x y) := by
  rw [fluxFirst, pointFlux_eq σ (R x y) u (phase x y) (phase (x + 1) y) hu]
  simp only [ringNormForm]
  ring

/-- The density is never negative.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (x y : ℝ)                       [two real coordinates]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    0 ≤ density σ R u x y           [the amount is at or above zero]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    `density`                       [the amount: the size of the phase cell at a place]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    A square, so the wet/dry seam is exactly amplitude zero. This is an instance result for `ℝ`:
    the proof uses the order and `mul_self_nonneg`, and no such statement holds in a ring such as
    `ZMod (2 ^ 64)`. -/
theorem density_nonnegative (σ : ℝ) (R : PhaseField) (u : Cell) (x y : ℝ)
    (hu : normForm σ u = 1) : 0 ≤ density σ R u x y := by
  rw [density_eq_amplitude_sq σ R u x y hu]
  exact mul_self_nonneg _

/-- The density vanishes exactly where the amplitude does.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (x y : ℝ)                       [two real coordinates]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    density σ R u x y = 0 ↔ R x y = 0   [the amount is zero exactly at amplitude zero]
    ℝ                               [the real numbers: every number on the number line]
    ↔                               [exactly when, true in both directions]
    `density`                       [the amount: the size of the phase cell at a place]
    `normForm`                      [the norm form: a point times its conjugate, which can come
                                     out negative]

    The null seam is not an independently painted mask. This is an instance result for `ℝ`: the
    proof uses `mul_self_eq_zero`, which needs no zero divisors, and it does not transfer to a
    ring such as `ZMod (2 ^ 64)`. -/
theorem density_eq_zero_iff (σ : ℝ) (R : PhaseField) (u : Cell) (x y : ℝ)
    (hu : normForm σ u = 1) : density σ R u x y = 0 ↔ R x y = 0 := by
  rw [density_eq_amplitude_sq σ R u x y hu]
  exact mul_self_eq_zero

/-- A density-null cell carries no outgoing first-direction flux.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [a real amplitude field]
    (u : Cell)                      [the phase direction: a point the amplitude stretches]
    (phase : PhaseField)            [a real phase field]
    (x y : ℝ)                       [two real coordinates]
    (h : density σ R u x y = 0)     [the cell's amount is zero]
    fluxFirst σ R u phase x y = 0   [so its outgoing flux is zero]
    ℝ                               [the real numbers: every number on the number line]
    `fluxFirst`                     [the first-direction flux: amount times phase slope]
    `density`                       [the amount: the size of the phase cell at a place]

    The wet/dry seam read from the phase cell, not a separate mask rule. -/
theorem fluxFirst_zero_of_density_zero (σ : ℝ) (R : PhaseField) (u : Cell)
    (phase : PhaseField) (x y : ℝ) (h : density σ R u x y = 0) :
    fluxFirst σ R u phase x y = 0 := by
  have hf : fluxFirst σ R u phase x y
      = density σ R u x y * (phase (x + 1) y - phase x y) := rfl
  rw [hf, h, zero_mul]

/-- The integer density readout the machine-word kernel computes: a signed sample squared.

    (R : ℤ)                         [a signed amplitude sample]
    ℤ                               [the whole numbers: the counting numbers and their negatives]
    `wordDensity`                   [the machine-word amount: the sample times itself]
    `pointDensity`                  [the amount: the size of the phase cell at a place]

    The Aeneas-extracted `physics_math.density` computes exactly this on `i64`. The machine-word
    kernel carries no dial; the unit direction does not consult one, so the generic readout is
    instantiated at `0`, and `pointDensity_ringOne_independent` records that any other dial gives
    the same value. The amplitude is signed so an amplitude sign survives a refinement that a
    `u16` result would erase. -/
def wordDensity (R : ℤ) : ℤ := pointDensity (0 : ℤ) R ringOne

/-- The integer first-direction flux readout the machine-word kernel computes.

    (R p n : ℤ)                     [the amplitude sample, this phase and the next]
    ℤ                               [the whole numbers: the counting numbers and their negatives]
    `wordFluxFirst`                 [the machine-word flux: amount times the phase difference]
    `wordDensity`                   [the machine-word amount: the sample times itself]
    `pointFlux`                     [the first-direction flux: amount times phase difference]

    Density times the forward phase difference, at integer samples. The Aeneas-extracted
    `physics_math.phase_flux` computes exactly this, with the sign of the difference retained. The
    dial is immaterial here too, by `pointFlux_ringOne_independent`. -/
def wordFluxFirst (R p n : ℤ) : ℤ := pointFlux (0 : ℤ) R ringOne p n

/-- The integer density unfolds to the sample times itself.

    (R : ℤ)                         [a signed amplitude sample]
    wordDensity R = R * R           [the machine-word amount is the sample times itself]
    ℤ                               [the whole numbers: the counting numbers and their negatives]
    `wordDensity`                   [the machine-word amount: the sample times itself]

    The integer image of the rooted real density, read through the generic `pointDensity_ringOne`
    rather than by a second polynomial expansion. -/
theorem wordDensity_eq (R : ℤ) : wordDensity R = R * R := by
  simpa [wordDensity, pow_two] using pointDensity_ringOne (0 : ℤ) R

/-- The integer first-direction flux unfolds to the amount times the phase difference.

    (R p n : ℤ)                     [the amplitude sample, this phase and the next]
    wordFluxFirst R p n
      = wordDensity R * (n - p)     [the flux unfolds to amount times phase difference]
    ℤ                               [the whole numbers: the counting numbers and their negatives]
    `wordFluxFirst`                 [the machine-word flux: amount times the phase difference]
    `wordDensity`                   [the machine-word amount: the sample times itself]

    The declaration the Aeneas-extracted `physics_math.phase_flux` refines, read through the
    generic `pointFlux_ringOne` and `pointDensity_ringOne`. -/
theorem wordFluxFirst_eq (R p n : ℤ) :
    wordFluxFirst R p n = wordDensity R * (n - p) := by
  rw [wordFluxFirst, pointFlux_ringOne, wordDensity, pointDensity_ringOne]

end Physics
end Apeiron
