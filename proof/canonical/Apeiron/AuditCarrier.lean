import Apeiron.AuditClosed
import Apeiron

/-!
# Audit: the carrier corpus

@name Carrier Audit
@grade proved

The real carrier modules, named in dependency order, with a `#audit` line for each headline
declaration. The command and the closed cone live in `Apeiron.AuditClosed`; this module only
names the declarations, so the fail-closed check is the same one `Audit.lean` runs over the rest
of the corpus. A carrier declaration is covered when it has a line here or in `Audit.lean`.
-/

-- The carrier readout: the algebra's own density and phase flux, and the machine-word samples
#audit Apeiron.Physics.PhaseField
#audit Apeiron.Physics.wavePoint
#audit Apeiron.Physics.density
#audit Apeiron.Physics.slopeFirst
#audit Apeiron.Physics.fluxFirst
#audit Apeiron.Physics.wordDensity
#audit Apeiron.Physics.wordFluxFirst
#audit Apeiron.Physics.density_eq_normForm_wavePoint
#audit Apeiron.Physics.density_eq_amplitude_sq
#audit Apeiron.Physics.fluxFirst_eq
#audit Apeiron.Physics.density_nonnegative
#audit Apeiron.Physics.density_eq_zero_iff
#audit Apeiron.Physics.fluxFirst_zero_of_density_zero
#audit Apeiron.Physics.wordDensity_eq
#audit Apeiron.Physics.wordFluxFirst_eq

-- The finite rectangle boundary accounting
#audit Apeiron.Physics.FaceFlux
#audit Apeiron.Physics.sampledFluxFirst
#audit Apeiron.Physics.cellDivergence
#audit Apeiron.Physics.sum_forward_difference
#audit Apeiron.Physics.rectangle_boundary_balance
#audit Apeiron.Physics.closed_rectangle_flux_cancels

-- The phase cell is the flow
#audit Apeiron.Physics.density_flow
#audit Apeiron.Physics.fluxFirst_flow

-- The conservative carrier evolution and its CFL bound
#audit Apeiron.Physics.carrierDensity
#audit Apeiron.Physics.carrierFaceFlux
#audit Apeiron.Physics.carrierDivergence
#audit Apeiron.Physics.evolveCarrierDensity
#audit Apeiron.Physics.carrierCFL
#audit Apeiron.Physics.evolveCarrierDensity_continuity
#audit Apeiron.Physics.sum_carrierDivergence
#audit Apeiron.Physics.evolveCarrierDensity_total_mass
#audit Apeiron.Physics.evolveCarrierDensity_nonnegative_iff_cfl

-- The diffusive relaxation and its Dirichlet energy
#audit Apeiron.Physics.carrierState
#audit Apeiron.Physics.dirichletEnergy
#audit Apeiron.Physics.carrierState_eq
#audit Apeiron.Physics.carrierState_nonnegative
#audit Apeiron.Physics.diffusiveUpdate_nonnegative
#audit Apeiron.Physics.dirichletEnergy_nonnegative
#audit Apeiron.Physics.smoothSlope_energy_le
#audit Apeiron.Physics.dirichletEnergy_nonincreasing
#audit Apeiron.Physics.carrierState_energy_nonincreasing
#audit Apeiron.Physics.overlarge_step_increases_energy
#audit Apeiron.Physics.diffusiveUpdate_half_three
#audit Apeiron.Physics.dirichletEnergy_halfStep_three
#audit Apeiron.Physics.dirichletEnergy_halfStep_three_lt
#audit Apeiron.Physics.halfStep_three_control

-- The three-dimensional volume accounting
#audit Apeiron.Physics.PhaseField3
#audit Apeiron.Physics.FaceFlux3
#audit Apeiron.Physics.sampleDensity3
#audit Apeiron.Physics.carrierDensity3
#audit Apeiron.Physics.faceFluxX
#audit Apeiron.Physics.faceFluxY
#audit Apeiron.Physics.faceFluxZ
#audit Apeiron.Physics.cellDivergence3
#audit Apeiron.Physics.carrierDivergence3
#audit Apeiron.Physics.evolveCarrierDensity3
#audit Apeiron.Physics.carrierCFL3
#audit Apeiron.Physics.sampleDensity3_eq_ringSampleDensity3
#audit Apeiron.Physics.carrierDensity3_eq_sampleDensity3
#audit Apeiron.Physics.sampleDensity3_eq_amplitude_sq
#audit Apeiron.Physics.carrierDensity3_eq_amplitude_sq
#audit Apeiron.Physics.evolveCarrierDensity3_continuity
#audit Apeiron.Physics.box_divergence_balance
#audit Apeiron.Physics.closed_box_flux_cancels
#audit Apeiron.Physics.sum_carrierDivergence3
#audit Apeiron.Physics.evolveCarrierDensity3_total_mass
#audit Apeiron.Physics.carrierDensity3_nonnegative
#audit Apeiron.Physics.evolveCarrierDensity3_nonnegative_iff_cfl

-- The trichotomy authority closure for carrier amounts
#audit Apeiron.Physics.basisRealization
#audit Apeiron.Physics.normForm_basisRealization
#audit Apeiron.Physics.basisRealization_eq_ringBasis_sqrt
#audit Apeiron.Physics.basisAmplitude3
#audit Apeiron.Physics.sampleDensity3_basisAmplitude3
#audit Apeiron.Physics.sampleDensity3_basisAmplitude3_one
#audit Apeiron.Physics.evolveCarrierDensity3_basisRealization
#audit Apeiron.Physics.exists_basisRealization_evolveCarrierDensity3
#audit Apeiron.Physics.diffusiveUpdate_basisRealization
#audit Apeiron.Physics.exists_basisRealization_diffusiveUpdate
#audit Apeiron.Physics.carrierState_basisRealization
#audit Apeiron.Physics.exists_basisRealization_carrierState
#audit Apeiron.Physics.carrierState_diffusiveUpdate_basisRealization
#audit Apeiron.Physics.exists_basisRealization_carrierState_diffusiveUpdate
#audit Apeiron.Physics.negative_not_realizable
#audit Apeiron.Physics.overlarge_diffusiveUpdate_negative
#audit Apeiron.Physics.overlarge_step_not_realizable

-- The equal-mass body exchange over any commutative ring
#audit Apeiron.Physics.ringExchangeImpulse
#audit Apeiron.Physics.ringCarrierMomentumAfter
#audit Apeiron.Physics.ringBodyMomentumAfter
#audit Apeiron.Physics.ringNormFormLedger
#audit Apeiron.Physics.ringCarrierMomentumAfter_eq
#audit Apeiron.Physics.ringBodyMomentumAfter_eq
#audit Apeiron.Physics.ringExchangeImpulse_antisymm
#audit Apeiron.Physics.ringTotalMomentum_conserved
#audit Apeiron.Physics.ringNormFormLedger_conserved

-- The real body exchange and its work ledger
#audit Apeiron.Physics.carrierWork
#audit Apeiron.Physics.bodyWork
#audit Apeiron.Physics.kineticEnergy
#audit Apeiron.Physics.acceptsImpulse
#audit Apeiron.Physics.independentImpulse
#audit Apeiron.Physics.carrierMomentumAfter_eq
#audit Apeiron.Physics.bodyMomentumAfter_eq
#audit Apeiron.Physics.exchangeImpulse_antisymm
#audit Apeiron.Physics.totalMomentum_conserved
#audit Apeiron.Physics.normFormLedger_conserved
#audit Apeiron.Physics.carrierWork_eq_neg_bodyWork
#audit Apeiron.Physics.totalWork_zero
#audit Apeiron.Physics.kineticEnergy_nonnegative_of_nonpos
#audit Apeiron.Physics.carrierWork_eq_two_mul_kineticEnergyChange
#audit Apeiron.Physics.bodyWork_eq_two_mul_kineticEnergyChange
#audit Apeiron.Physics.accepts_derivedImpulse
#audit Apeiron.Physics.independentImpulse_rejected

-- The finite basis momentum field and its transactional exchange
#audit Apeiron.Physics.sum_single_site
#audit Apeiron.Physics.sum_single_site_real
#audit Apeiron.Physics.fieldMomentumAfter_site
#audit Apeiron.Physics.fieldMomentumAfter_site_eq_body
#audit Apeiron.Physics.fieldMomentumAfter_other
#audit Apeiron.Physics.fieldMomentumAfter_eq_update
#audit Apeiron.Physics.totalFieldMomentum_conserved
#audit Apeiron.Physics.normFormFieldLedger_conserved
#audit Apeiron.Physics.exchangeTransaction_accepts
#audit Apeiron.Physics.exchangeTransaction_rejects_position
#audit Apeiron.Physics.exchangeTransaction_rejects_impulse
#audit Apeiron.Physics.exchangeOrRollback_of_rejected_position
#audit Apeiron.Physics.exchangeOrRollback_of_rejected_impulse

-- The basis-derived circulation and discrete Stokes
#audit Apeiron.Physics.basisFaces_readback
#audit Apeiron.Physics.scalarGradient_plaquetteCirculation_zero
#audit Apeiron.Physics.rotationalWitness_plaquetteCirculation
#audit Apeiron.Physics.rectangle_discrete_stokes
#audit Apeiron.Physics.closed_rectangle_circulation_cancels

-- The time direction of the wave
#audit Apeiron.Physics.timeDeriv_waveFst
#audit Apeiron.Physics.timeDeriv_waveSnd
#audit Apeiron.Physics.timeDeriv_waveFst_elliptic
