import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Apeiron.Physics.Carrier.Derived.AuthorityConstructive
import Apeiron.Physics.Carrier.Derived.Volume
import Apeiron.Physics.Carrier.Derived.Dissipation

/-!
# Trichotomy authority closure for carrier amounts

@name Carrier Authority Closure
@cite kocik-1999
@grade proved

The carrier modules read every amount from the trichotomy norm form, but the evolution and
diffusive updates hand back a *scalar*. This module closes that gap in representation. The
executable construction lives in `Apeiron.Physics.CarrierAuthorityConstructive`: an explicit
amplitude with a proof that it squares to the amount, or a `RingCell` already on the basis
direction, reads back by ring algebra alone. That interface is computable and carries the root
as a hypothesis.

The real-valued `basisRealization d` is `Real.sqrt d • one`. It is kept as the real model
instance of the generic interface and not as the executable authority: `Real.sqrt` is
noncomputable, so it names no operation an exact integer or machine-word execution can run.
`basisRealization_eq_ringBasis_sqrt` exhibits the instance, and
`normForm_basisRealization_via_ringBasis` reads the amount back through the generic theorem.
`normForm_basisRealization` states the same real instance directly. The same sqrt construction
lifts a nonnegative scalar field to an extended three-dimensional amplitude whose
`sampleDensity3` is the field. `ringEvolveCarrierDensity3_ringBasis_of_sq` is the generic readback
of the three-dimensional evolution update and `diffusiveUpdate_ringBasis_of_sq` the generic
readback of the diffusive update, both over any commutative ring;
`evolveCarrierDensity3_basisRealization`, `diffusiveUpdate_basisRealization` and
`carrierState_diffusiveUpdate_basisRealization` are their real model specializations.

## Supported regime, assumptions and scope

The construction is **representation, not dynamics**. It shows that a nonnegative scalar amount
can be written again as the norm form of a trichotomy-basis amplitude; it does not derive a
physical damping coefficient, a phase, a circulation, or any body work. The dissipative bound
`0 ≤ ν ≤ 1/2` is the explicit stability range of the diffusive stencil, not a measured
viscosity. The reconstruction is pointwise on the extended sample lattice and adds no runtime
operation.

The control is scoped to the canonical real-amplitude, unit-direction realization: the size of
`r • one` is `r²`, which is never negative, so a negative scalar has no such preimage. This is
not a claim that arbitrary negative values are unrealizable on hyperbolic faces of the algebra.
The root hypothesis is not vacuous either: `no_sq_two_int` in the constructive module proves
that no whole number squares to two, so even a nonnegative amount can have no amplitude witness.

## Root

`basisRealization` is `Real.sqrt d • one`, the unit direction stretched by the root of the
amount, and `basisRealization_eq_ringBasis_sqrt` reads it as the generic `ringBasis` at
`Real.sqrt d`. `normForm_basisRealization` proves the readback at `0 ≤ d`, and
`normForm_basisRealization_via_ringBasis` derives it from the generic
`ringNormForm_ringBasis_of_sq`, with `Real.sq_sqrt` supplying the root hypothesis.
`basisAmplitude3` is the same construction pointwise on a scalar field, and
`sampleDensity3_basisAmplitude3` reads the field back.
`evolveCarrierDensity3_basisRealization` and `diffusiveUpdate_basisRealization` apply the
readback to the two updates once their own nonnegativity hypotheses hold.
`ringEvolveCarrierDensity3_ringBasis_of_sq` and `diffusiveUpdate_ringBasis_of_sq` are the generic
chains: they consume the basis-point readback from
`Apeiron.Physics.CarrierAuthorityConstructive` and the generic updates from
`Apeiron.Physics.CarrierVolumeAnyRing` and `Apeiron.Physics.CarrierDissipationAnyRing`, and carry
the root as an explicit hypothesis. `negative_not_realizable` fixes the scope of the control.
-/

namespace Apeiron
namespace Physics

open Trichotomy
open Trichotomy.AnyRing

/-- The real-valued model realization of a scalar amount.

    (d : ℝ)                         [the scalar amount: a real number]
    Real.sqrt d • one               [the unit direction stretched by the root of the amount]
    Cell                            [a point: two real numbers written side by side, a + b·x]
    ℝ                               [the real numbers: every number on the number line]
    •                               [stretch by, a number times a point]
    `Real.sqrt`                     [the number that squares to give this]
    `one`                           [the point (1, 0), which changes nothing when multiplied]

    The direction is the trichotomy unit `one`, not a posited second material field: the amount
    is carried by the amplitude `Real.sqrt d` along the algebra's own basis direction. This
    realization is noncomputable and is the real model instance; the executable construction is
    `ringBasis` in `Apeiron.Physics.CarrierAuthorityConstructive`. -/
noncomputable def basisRealization (d : ℝ) : Cell := Real.sqrt d • one

/-- The norm form of the canonical realization is the amount it realizes.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (d : ℝ)                         [the scalar amount: a real number]
    (hd : 0 ≤ d)                    [the amount is at or above zero]
    normForm σ (basisRealization d) = d
                                    [measuring the realization returns the amount]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `one`                           [the point (1, 0), which changes nothing when multiplied]

    The proof stretches the unit direction: `normForm_smul` brings out the root squared, the
    unit's own size is one, and `Real.sq_sqrt` returns `d`. It is a readback of the algebra's
    size, not an existence assertion. The executable route is `ringNormForm_ringBasis_of_sq`,
    and `normForm_basisRealization_via_ringBasis` derives this statement from it. -/
theorem normForm_basisRealization (σ : ℝ) {d : ℝ} (hd : 0 ≤ d) :
    normForm σ (basisRealization d) = d := by
  rw [basisRealization, normForm_smul]
  have hone : normForm σ one = 1 := by simp [normForm]
  rw [hone, mul_one, Real.sq_sqrt hd]

/-- The real square-root realization is the generic basis point at the real root.

    (d : ℝ)                         [the scalar amount: a real number]
    ℝ                               [the real numbers: every number on the number line]
    basisRealization d              [the real model realization: the unit direction stretched
                                     by the root of the amount]
    =                               [is equal to]
    ringBasis (Real.sqrt d)         [the generic basis point at the real root]
    `Real.sqrt`                     [the number that squares to give this]
    `ringBasis`                     [the generic basis point: amplitude first, zero second]

    So: the noncomputable real construction is the generic computable constructor applied to
    the real square root. The generic `ringNormForm_ringBasis_of_sq` is the authority; this
    equation records the real instance of it. -/
theorem basisRealization_eq_ringBasis_sqrt (d : ℝ) :
    basisRealization d = ringBasis (Real.sqrt d) := by
  ext <;> simp [basisRealization, ringBasis, one, smul_eq_mul]

/-- The real readback is the generic readback at the real root.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (d : ℝ)                         [the scalar amount: a real number]
    (hd : 0 ≤ d)                    [the amount is at or above zero]
    normForm σ (basisRealization d) = d
                                    [measuring the real realization returns the amount]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]

    So: `Real.sq_sqrt` supplies the squaring hypothesis to `ringNormForm_ringBasis_of_sq`, so
    the real readback is an instance of the generic one rather than a separate authority. -/
theorem normForm_basisRealization_via_ringBasis (σ : ℝ) {d : ℝ} (hd : 0 ≤ d) :
    normForm σ (basisRealization d) = d := by
  rw [basisRealization_eq_ringBasis_sqrt]
  exact ringNormForm_ringBasis_of_sq σ (Real.sq_sqrt hd)

/-- The extended three-dimensional amplitude realizing a nonnegative scalar field.

    (d : PhaseField3)               [the scalar amount at every extended sample]
    PhaseField3                     [a real value at every triple of counting coordinates]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right]
    ℝ                               [the real numbers: every number on the number line]
    `Real.sqrt`                     [the number that squares to give this]

    Pointwise, `basisAmplitude3 d a b c` is the root of `d a b c` along the canonical unit
    direction; the direction itself is supplied by the readback theorem below. -/
noncomputable def basisAmplitude3 (d : PhaseField3) : PhaseField3 :=
  fun a b c => Real.sqrt (d a b c)

/-- The extended sample density reads back the scalar field it was built from.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (d : PhaseField3)               [the scalar amount at every extended sample]
    (u : Cell)                      [the fixed phase direction]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    (hd : ∀ a b c, 0 ≤ d a b c)     [every sample amount is at or above zero]
    (a b c : ℕ)                     [an extended sample coordinate]
    sampleDensity3 σ (basisAmplitude3 d) u a b c = d a b c
                                    [the sample density returns the field exactly]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∀                               [for every single one]
    `sampleDensity3`                [the trichotomy norm-form amount of an extended sample]
    `basisAmplitude3`               [the extended amplitude realizing a scalar field]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]

    This closes representation, not dynamics: it says the scalar field can be written again as a
    norm-form amplitude field, with no claim about how that field evolves. -/
theorem sampleDensity3_basisAmplitude3 (σ : ℝ) (d : PhaseField3) (u : Cell)
    (hu : normForm σ u = 1) (hd : ∀ a b c, 0 ≤ d a b c) (a b c : ℕ) :
    sampleDensity3 σ (basisAmplitude3 d) u a b c = d a b c := by
  simp only [sampleDensity3, basisAmplitude3]
  rw [normForm_smul, hu, mul_one, Real.sq_sqrt (hd a b c)]

/-- The canonical unit-direction readback: at `one`, the sample density is the field.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (d : PhaseField3)               [the scalar amount at every extended sample]
    (hd : ∀ a b c, 0 ≤ d a b c)     [every sample amount is at or above zero]
    (a b c : ℕ)                     [an extended sample coordinate]
    sampleDensity3 σ (basisAmplitude3 d) one a b c = d a b c
                                    [at the unit direction the sample density is the field]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∀                               [for every single one]
    `sampleDensity3`                [the trichotomy norm-form amount of an extended sample]
    `basisAmplitude3`               [the extended amplitude realizing a scalar field]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `one`                           [the point (1, 0), which changes nothing when multiplied]

    The direction is the trichotomy unit `one`, whose size is one, so this is the canonical
    unit-direction instance of the general readback. -/
theorem sampleDensity3_basisAmplitude3_one (σ : ℝ) (d : PhaseField3)
    (hd : ∀ a b c, 0 ≤ d a b c) (a b c : ℕ) :
    sampleDensity3 σ (basisAmplitude3 d) one a b c = d a b c :=
  sampleDensity3_basisAmplitude3 σ d one (by simp [normForm]) hd a b c

/-- The generic three-dimensional evolution update reads back through an explicit root witness.

    (τ σ : R)                       [the time step and the algebra parameter]
    (A : RingPhaseField3 R)         [the ring amplitude field]
    (u : RingCell R)                [the fixed phase direction]
    (phase : RingPhaseField3 R)     [the ring phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    (a : R)                         [the amplitude witness]
    (ha : a ^ 2 = ringEvolveCarrierDensity3 τ σ A u phase x y z)
                                    [the witness squares to the generic update value]
    ringNormForm σ (ringBasis a) = ringEvolveCarrierDensity3 τ σ A u phase x y z
                                    [measuring the basis point returns the updated amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `ringBasis`                 [the amplitude in the first slot, zero in the second]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    This is the executable authority for the three-dimensional evolution update. It consumes the
    generic `ringEvolveCarrierDensity3` from `Apeiron.Physics.CarrierVolumeAnyRing` and the
    basis-point readback `ringNormForm_ringBasis_of_sq` from
    `Apeiron.Physics.CarrierAuthorityConstructive`, and it carries the root as a hypothesis
    rather than searching for one. The real `evolveCarrierDensity3_basisRealization` below is its
    specialization at `R = ℝ`, with `Real.sq_sqrt` supplying the witness. -/
theorem ringEvolveCarrierDensity3_ringBasis_of_sq {R : Type*} [CommRing R]
    (τ σ : R) (A : RingPhaseField3 R) (u : RingCell R) (phase : RingPhaseField3 R)
    (x y z : ℕ) {a : R}
    (ha : a ^ 2 = ringEvolveCarrierDensity3 τ σ A u phase x y z) :
    ringNormForm σ (ringBasis a) = ringEvolveCarrierDensity3 τ σ A u phase x y z :=
  ringNormForm_ringBasis_of_sq σ ha

/-- Every CFL-admissible carrier-volume update value has a basis realization.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    (hcfl : carrierCFL3 τ σ R u phase x y z)
                                    [the local forward-Euler bound on the net outward transfer]
    normForm σ (basisRealization (evolveCarrierDensity3 τ σ R u phase x y z))
      = evolveCarrierDensity3 τ σ R u phase x y z
                                    [measuring the realization returns the updated amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `evolveCarrierDensity3`         [the conservative material update of the carrier density]
    `carrierCFL3`                   [the explicit local bound for nonnegative updated density]

    The CFL hypothesis is exactly nonnegativity of the updated scalar, so the canonical
    realization applies and the updated amount is again a norm form. This is the real model
    specialization of `ringEvolveCarrierDensity3_ringBasis_of_sq`: the executable route carries
    the root as a hypothesis and never evaluates `Real.sqrt`. -/
theorem evolveCarrierDensity3_basisRealization (τ σ : ℝ) (R : PhaseField3) (u : Cell)
    (phase : PhaseField3) (x y z : ℕ) (hcfl : carrierCFL3 τ σ R u phase x y z) :
    normForm σ (basisRealization (evolveCarrierDensity3 τ σ R u phase x y z))
      = evolveCarrierDensity3 τ σ R u phase x y z := by
  rw [basisRealization_eq_ringBasis_sqrt]
  exact ringEvolveCarrierDensity3_ringBasis_of_sq τ σ R u phase x y z
    (Real.sq_sqrt ((evolveCarrierDensity3_nonnegative_iff_cfl τ σ R u phase x y z).mpr hcfl))

/-- Some trichotomy-basis point realizes every CFL-admissible carrier-volume update.

    (τ σ : ℝ)                       [the time step and the algebra parameter]
    (R : PhaseField3)               [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (phase : PhaseField3)           [the real phase field]
    (x y z : ℕ)                     [the cell's three counting-lattice coordinates]
    (hcfl : carrierCFL3 τ σ R u phase x y z)
                                    [the local forward-Euler bound on the net outward transfer]
    ∃ w : Cell, normForm σ w = evolveCarrierDensity3 τ σ R u phase x y z
                                    [there is a point whose size is the updated amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∃                               [there is at least one]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `evolveCarrierDensity3`         [the conservative material update of the carrier density]
    `carrierCFL3`                   [the explicit local bound for nonnegative updated density]

    The witness is the canonical realization itself, so this is an existence corollary of the
    constructive equality rather than a separate assertion. -/
theorem exists_basisRealization_evolveCarrierDensity3 (τ σ : ℝ) (R : PhaseField3) (u : Cell)
    (phase : PhaseField3) (x y z : ℕ) (hcfl : carrierCFL3 τ σ R u phase x y z) :
    ∃ w : Cell, normForm σ w = evolveCarrierDensity3 τ σ R u phase x y z :=
  ⟨basisRealization (evolveCarrierDensity3 τ σ R u phase x y z),
    evolveCarrierDensity3_basisRealization τ σ R u phase x y z hcfl⟩

/-- The generic diffusive update reads back through an explicit root witness.

    (σ ν : R)                       [the algebra parameter and the time-step parameter]
    (rho : ℕ → R)                   [the density array, over the ring]
    (i : ℕ)                         [the counting cell position]
    (a : R)                         [the amplitude witness]
    (ha : a ^ 2 = diffusiveUpdate ν rho i)
                                    [the witness squares to the generic update value]
    ringNormForm σ (ringBasis a) = diffusiveUpdate ν rho i
                                    [measuring the basis point returns the updated amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `ringBasis`                 [the amplitude in the first slot, zero in the second]
    `ringNormForm`              [the norm form: a point times its conjugate, negative or not]

    This is the executable authority for the diffusive update. It consumes the generic
    `diffusiveUpdate` from `Apeiron.Physics.CarrierDissipationAnyRing` and the basis-point
    readback `ringNormForm_ringBasis_of_sq` from
    `Apeiron.Physics.CarrierAuthorityConstructive`, and it carries the root as a hypothesis
    rather than searching for one. The real `diffusiveUpdate_basisRealization` below is its
    specialization at `R = ℝ`, with `Real.sq_sqrt` supplying the witness. -/
theorem diffusiveUpdate_ringBasis_of_sq {R : Type*} [CommRing R]
    (σ ν : R) (rho : ℕ → R) (i : ℕ) {a : R}
    (ha : a ^ 2 = diffusiveUpdate ν rho i) :
    ringNormForm σ (ringBasis a) = diffusiveUpdate ν rho i :=
  ringNormForm_ringBasis_of_sq σ ha

/-- Every stable diffusive update value has a basis realization.

    (ν σ : ℝ)                       [the time-step parameter and the algebra parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (rho : ℕ → ℝ)                   [the density array]
    (hrho : ∀ i, 0 ≤ rho i)         [every cell amount is at or above zero]
    (i : ℕ)                         [the counting cell position]
    normForm σ (basisRealization (diffusiveUpdate ν rho i)) = diffusiveUpdate ν rho i
                                    [measuring the realization returns the updated amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∀                               [for every single one]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]

    The stable range makes the updated scalar nonnegative, so the canonical realization applies.
    This is the real model specialization of `diffusiveUpdate_ringBasis_of_sq`: the executable
    route carries the root as a hypothesis and never evaluates `Real.sqrt`. -/
theorem diffusiveUpdate_basisRealization (ν σ : ℝ) (hν0 : 0 ≤ ν) (hν : ν ≤ 1 / 2)
    (rho : ℕ → ℝ) (hrho : ∀ i, 0 ≤ rho i) (i : ℕ) :
    normForm σ (basisRealization (diffusiveUpdate ν rho i)) = diffusiveUpdate ν rho i := by
  rw [basisRealization_eq_ringBasis_sqrt]
  exact diffusiveUpdate_ringBasis_of_sq σ ν rho i
    (Real.sq_sqrt (diffusiveUpdate_nonnegative hν0 hν hrho i))

/-- Some trichotomy-basis point realizes every stable diffusive update.

    (ν σ : ℝ)                       [the time-step parameter and the algebra parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (rho : ℕ → ℝ)                   [the density array]
    (hrho : ∀ i, 0 ≤ rho i)         [every cell amount is at or above zero]
    (i : ℕ)                         [the counting cell position]
    ∃ w : Cell, normForm σ w = diffusiveUpdate ν rho i
                                    [there is a point whose size is the updated amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    →                               [takes the left and gives the right; between statements, if
                                     the left holds so does the right]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∀                               [for every single one]
    ∃                               [there is at least one]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `Cell`                          [a point: two real numbers written side by side, a + b·x]

    The witness is the canonical realization of the update value. -/
theorem exists_basisRealization_diffusiveUpdate (ν σ : ℝ) (hν0 : 0 ≤ ν) (hν : ν ≤ 1 / 2)
    (rho : ℕ → ℝ) (hrho : ∀ i, 0 ≤ rho i) (i : ℕ) :
    ∃ w : Cell, normForm σ w = diffusiveUpdate ν rho i :=
  ⟨basisRealization (diffusiveUpdate ν rho i),
    diffusiveUpdate_basisRealization ν σ hν0 hν rho hrho i⟩

/-- The carrier state itself is a norm form and has a basis realization.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    (i : ℕ)                         [the counting cell position]
    normForm σ (basisRealization (carrierState σ R u i)) = carrierState σ R u i
                                    [measuring the realization returns the state amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `carrierState`                  [the carrier density readout on the finite line]

    The state is already nonnegative, so its canonical realization reads back the same amount. -/
theorem carrierState_basisRealization (σ : ℝ) (R : PhaseField) (u : Cell)
    (hu : normForm σ u = 1) (i : ℕ) :
    normForm σ (basisRealization (carrierState σ R u i)) = carrierState σ R u i :=
  normForm_basisRealization σ (carrierState_nonnegative σ R u hu i)

/-- Some trichotomy-basis point realizes the carrier state at every cell.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    (i : ℕ)                         [the counting cell position]
    ∃ w : Cell, normForm σ w = carrierState σ R u i
                                    [there is a point whose size is the state amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ∃                               [there is at least one]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `carrierState`                  [the carrier density readout on the finite line]

    The state stays within trichotomy-realizable authority at every cell. -/
theorem exists_basisRealization_carrierState (σ : ℝ) (R : PhaseField) (u : Cell)
    (hu : normForm σ u = 1) (i : ℕ) :
    ∃ w : Cell, normForm σ w = carrierState σ R u i :=
  ⟨basisRealization (carrierState σ R u i), carrierState_basisRealization σ R u hu i⟩

/-- The carrier state's stable diffusive update stays within trichotomy-realizable authority.

    (ν σ : ℝ)                       [the time-step parameter and the algebra parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    (i : ℕ)                         [the counting cell position]
    normForm σ (basisRealization (diffusiveUpdate ν (carrierState σ R u) i))
      = diffusiveUpdate ν (carrierState σ R u) i
                                    [measuring the realization returns the updated state amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `carrierState`                  [the carrier density readout on the finite line]

    The state is pointwise nonnegative by `carrierState_nonnegative`, so the stable diffusive
    update is nonnegative and its canonical realization reads the updated amount back. This is
    the real model specialization of `diffusiveUpdate_ringBasis_of_sq`, whose executable route
    carries the root as a hypothesis. -/
theorem carrierState_diffusiveUpdate_basisRealization (ν σ : ℝ) (hν0 : 0 ≤ ν)
    (hν : ν ≤ 1 / 2) (R : PhaseField) (u : Cell) (hu : normForm σ u = 1) (i : ℕ) :
    normForm σ (basisRealization (diffusiveUpdate ν (carrierState σ R u) i))
      = diffusiveUpdate ν (carrierState σ R u) i := by
  rw [basisRealization_eq_ringBasis_sqrt]
  exact diffusiveUpdate_ringBasis_of_sq σ ν (carrierState σ R u) i
    (Real.sq_sqrt (diffusiveUpdate_nonnegative hν0 hν
      (fun j => carrierState_nonnegative σ R u hu j) i))

/-- Some trichotomy-basis point realizes the carrier state's stable diffusive update.

    (ν σ : ℝ)                       [the time-step parameter and the algebra parameter]
    (hν0 : 0 ≤ ν)                   [the time-step parameter is at or above zero]
    (hν : ν ≤ 1/2)                  [the time-step parameter is at most one half]
    (R : PhaseField)                [the real amplitude field]
    (u : Cell)                      [the fixed phase direction]
    (hu : normForm σ u = 1)         [the phase direction has size one]
    (i : ℕ)                         [the counting cell position]
    ∃ w : Cell, normForm σ w = diffusiveUpdate ν (carrierState σ R u) i
                                    [there is a point whose size is the updated state amount]
    ℕ                               [the counting numbers 0, 1, 2, 3, and so on]
    ℝ                               [the real numbers: every number on the number line]
    ≤                               [is smaller than or equal to]
    ∃                               [there is at least one]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `basisRealization`              [the canonical realization: the unit direction stretched by
                                     the root of the amount]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `carrierState`                  [the carrier density readout on the finite line]

    The witness is the canonical realization of the updated state value. -/
theorem exists_basisRealization_carrierState_diffusiveUpdate (ν σ : ℝ) (hν0 : 0 ≤ ν)
    (hν : ν ≤ 1 / 2) (R : PhaseField) (u : Cell) (hu : normForm σ u = 1) (i : ℕ) :
    ∃ w : Cell, normForm σ w = diffusiveUpdate ν (carrierState σ R u) i :=
  ⟨basisRealization (diffusiveUpdate ν (carrierState σ R u) i),
    carrierState_diffusiveUpdate_basisRealization ν σ hν0 hν R u hu i⟩

/-- A negative scalar has no canonical real-amplitude unit-direction realization.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    (d : ℝ)                         [the scalar amount: a real number]
    (hd : d < 0)                    [the amount is below zero]
    ¬ ∃ r : ℝ, normForm σ (r • one) = d
                                    [no real amplitude along the unit realizes it]
    ℝ                               [the real numbers: every number on the number line]
    <                               [is smaller than]
    ∃                               [there is at least one]
    ¬                               [this is not true]
    •                               [stretch by, a number times a point]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `one`                           [the point (1, 0), which changes nothing when multiplied]

    The control is scoped to the canonical real-amplitude unit-direction realization: the size
    of `r • one` is `r²`, which is never negative. A negative amount may still be the norm form
    of a point on a hyperbolic face; that is not claimed here. -/
theorem negative_not_realizable (σ : ℝ) {d : ℝ} (hd : d < 0) :
    ¬ ∃ r : ℝ, normForm σ (r • one) = d := by
  rintro ⟨r, hr⟩
  have hnn : 0 ≤ normForm σ (r • one) := by
    rw [normForm_smul]
    have hone : normForm σ one = 1 := by simp [normForm]
    rw [hone, mul_one]
    exact sq_nonneg r
  rw [hr] at hnn
  linarith

/-- An overlarge diffusive step drives the control witness below zero.

    diffusiveUpdate (1 : ℝ) controlState 1 < 0
                                    [the step at `ν = 1` makes cell one negative]
    ℝ                               [the real numbers: every number on the number line]
    <                               [is smaller than]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `controlState`                  [the negative-control density witness]

    At `ν = 1`, outside the stable range, the interior weight `1 - 2ν` is `-1`, so the middle
    cell's value becomes negative. -/
theorem overlarge_diffusiveUpdate_negative :
    diffusiveUpdate (1 : ℝ) controlState 1 < 0 := by
  have h := diffusiveUpdate_succ (1 : ℝ) controlState 0
  simp only [smul_eq_mul] at h
  norm_num [controlState] at h
  rw [h]
  norm_num

/-- The overlarge step's negative value fails the canonical realization precondition.

    (σ : ℝ)                         [the parameter: what the adjoined direction squares to]
    ¬ ∃ r : ℝ, normForm σ (r • one) = diffusiveUpdate (1 : ℝ) controlState 1
                                    [no real amplitude along the unit realizes the negative step]
    ℝ                               [the real numbers: every number on the number line]
    ∃                               [there is at least one]
    ¬                               [this is not true]
    •                               [stretch by, a number times a point]
    `normForm`                      [the norm form: a point times its conjugate, negative or not]
    `one`                           [the point (1, 0), which changes nothing when multiplied]
    `diffusiveUpdate`               [the conservative update: old density less the flux difference]
    `controlState`                  [the negative-control density witness]

    This is the negative control for the stable bound: the overlarge step produces a value the
    canonical unit-direction realization cannot represent, still scoped to that realization. -/
theorem overlarge_step_not_realizable (σ : ℝ) :
    ¬ ∃ r : ℝ, normForm σ (r • one) = diffusiveUpdate (1 : ℝ) controlState 1 :=
  negative_not_realizable σ overlarge_diffusiveUpdate_negative

end Physics
end Apeiron
