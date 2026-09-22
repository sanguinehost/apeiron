import WordRefinement
import Native.HyperbolicBase
import Native.LocatorSpec
import Growable.Identity
import Growable.Geometry
import Growable.Scan
import Growable.Search
import Growable.Relocation
import Growable.Refinement
import Growable.Resources
import Growable.Representation
import Growable.Rebuild
import Joint.WordBridge
import Lean

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open apeiron_word

/-!
# Audit: the axiom cone

`#audit` on every extracted and generic headline below, run as part of the build. A clean cone is
`{propext, Classical.choice, Quot.sound}`, the three standard Lean axioms. `#audit` collects each
theorem's transitive axiom dependencies with `Lean.collectAxioms`, the same mechanism
`#print axioms` uses, and calls `throwError` on anything outside that set, so a `sorryAx` here is a
build failure rather than a line in a log nobody reads. `#print axioms` alone cannot gate: it only
prints, so a `sorryAx` in the list would compile clean.

Add a `#audit` line for each new headline as the bridge grows.
-/

open Lean Elab Command in
/-- The three axioms every proof in this bridge may depend on. -/
def auditClosedAxioms : Lean.NameSet :=
  ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound

open Lean Elab Command in
/-- **The gate.** `#audit foo` collects `foo`'s transitive axiom dependencies with
    `Lean.collectAxioms` and calls `throwError` if any of them falls outside `auditClosedAxioms`.
    A `sorry` in `foo`'s proof, or in anything `foo` depends on, surfaces as `Lean.sorryAx`, which
    is not in the closed set, so the build fails at this line. Prints the cone on success, so the
    log carries the same information `#print axioms` did. -/
elab "#audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditClosedAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms: {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the closed cone " ++
        m!"(propext, Classical.choice, Quot.sound): {bad.toList}"

-- The extracted word kernels refine the canonical algebra
#audit sigma_value_eq
#audit denote_from_i64
#audit word_mul_denote
#audit word_norm_denote
#audit denoteCell_cellConj
#audit word_mul_total
#audit word_norm_total
#audit word_norm_mul
#audit word_mul_conj
#audit word_norm_of_mul_denote

-- Point density and flux refinement
#audit word_point_density_unfold
#audit word_point_density_denote
#audit word_point_density_total
#audit word_point_density_eq
#audit word_point_flux_unfold
#audit word_point_flux_denote
#audit word_point_density_ringOne
#audit word_point_density_ringOne_independent
#audit word_point_flux_ringOne
#audit word_point_flux_ringOne_independent

-- Exchange refinement
#audit word_exchange_impulse_denote
#audit word_elastic_exchange_denote
#audit word_elastic_exchange_carrier
#audit word_elastic_exchange_body
#audit word_elastic_exchange_totalMomentum
#audit word_elastic_exchange_normFormLedger
#audit word_exchange_impulse_antisymm

-- The checked-claim acceptance kernel
#audit word_accept_derived_impulse_spec
#audit word_accept_derived_impulse_rejects
#audit word_accept_derived_impulse_rejects_of_ne
#audit word_accept_derived_impulse_accepts
#audit word_accept_derived_impulse_accepts_derived
#audit word_accept_derived_impulse_denote
#audit word_accept_derived_impulse_denote_canonical

-- The scalar carrier step and the exact-half flux
#audit word_conservative_density_step_denote
#audit word_conservative_density_step_eq_some_iff
#audit word_conservative_density_step_success
#audit word_diffusive_flux_half_denote
#audit word_diffusive_flux_half_success
#audit word_diffusive_flux_half_odd_refusal
#audit word_diffusive_flux_half_unrepresentable_refusal
#audit word_conservative_density_step_overflow_refusal

-- The finite conservative total through the extracted step
#audit word_conservative_total_mass
#audit word_conservative_total_mass_closed

-- The extracted three-cell diffusive line
#audit sum_line3
#audit word_diffusive_flux_half_cases
#audit word_conservative_density_step_cases
#audit word_diffusive_line3_step_eq_some_iff
#audit word_diffusive_line3_step_denote_of
#audit word_diffusive_line3_step_denote
#audit word_diffusive_line3_step_total_mass
#audit word_diffusive_line3_step_total_mass_zero_source
#audit word_diffusive_line3_step_odd_refusal
#audit word_diffusive_line3_step_zero_success
#audit word_diffusive_line3_step_unrepresentable_refusal

-- The ring-generic exchange specification
#audit Apeiron.Physics.ringCarrierMomentumAfter_eq
#audit Apeiron.Physics.ringBodyMomentumAfter_eq
#audit Apeiron.Physics.ringExchangeImpulse_antisymm
#audit Apeiron.Physics.ringTotalMomentum_conserved
#audit Apeiron.Physics.ringNormFormLedger_conserved

-- The canonical point readout at the unit direction
#audit Apeiron.Physics.pointDensity_eq
#audit Apeiron.Physics.pointFlux_eq
#audit Apeiron.Physics.pointDensity_ringOne
#audit Apeiron.Physics.pointDensity_ringOne_independent
#audit Apeiron.Physics.pointFlux_ringOne
#audit Apeiron.Physics.pointFlux_ringOne_independent

-- The scalar step links the finite carrier update, the shared-face total and the diffusive faces
#audit Apeiron.Physics.ringEvolveCarrierDensity_eq_scalarCarrierStep
#audit Apeiron.Physics.sum_scalarCarrierStep_eq
#audit Apeiron.Physics.scalarCarrierStep_total_mass
#audit Apeiron.Physics.diffusiveUpdate_eq_scalarCarrierStep
#audit Apeiron.Physics.exactHalfFlux_diffusiveFlux

-- The odd-prime-power quadratic split and finite scalar-CRT channel family
#audit Apeiron.NativeAddress.quadraticChannelEquiv
#audit Apeiron.NativeAddress.quadratic_split
#audit Apeiron.NativeAddress.h3_40_modulus
#audit Apeiron.NativeAddress.h3_40_two_mul_half
#audit Apeiron.NativeAddress.h3_40ChannelEquiv
#audit Apeiron.NativeAddress.h3_40_quadratic_split
#audit Apeiron.NativeAddress.finiteChannelEquiv
#audit Apeiron.NativeAddress.finite_channel_mul

-- Algebra-native locator membership and adversarial-cost specification
#audit Apeiron.NativeLocator.NO_CHILD
#audit Apeiron.NativeLocator.scalarWordAt
#audit Apeiron.NativeLocator.CheckedRecordArena
#audit Apeiron.NativeLocator.ScalarArenaDirectory
#audit Apeiron.NativeLocator.ScalarArenaShape
#audit Apeiron.NativeLocator.ArenaPathEndpoint
#audit Apeiron.NativeLocator.ScalarTraversalContract
#audit Apeiron.NativeLocator.ScalarLinkingPreservation
#audit Apeiron.NativeLocator.LocatorInvariant
#audit Apeiron.NativeLocator.absent_of_projected_missing
#audit Apeiron.NativeLocator.componentConjunction_of_presence
#audit Apeiron.NativeLocator.SourceLinkObligation
#audit Apeiron.NativeLocator.HashDOSBound
#audit Apeiron.NativeLocator.equalDescriptors_are_covered
#audit Apeiron.NativeLocator.triePepper_cost_not_yet_implied
#audit Apeiron.NativeLocator.FiniteExactIdentityContract
#audit Apeiron.NativeLocator.byteKeyExactIdentity
#audit Apeiron.NativeLocator.FaceResidueCapability
#audit Apeiron.NativeLocator.FaceHomeLaws
#audit Apeiron.NativeLocator.FaceLocatorCapability
#audit Apeiron.NativeLocator.manyToOne_residue_not_exactIdentity
#audit Apeiron.NativeLocator.table_admission_map_exact
#audit Apeiron.NativeLocator.migration_options_exact
#audit Apeiron.NativeLocator.api_migration_is_pending
#audit Apeiron.NativeLocator.TableEntryAtCoverageAdapter
#audit Apeiron.NativeLocator.tableEntryAt_absent_of_missing
#audit Apeiron.NativeLocator.SourceCoverageObligations
#audit Apeiron.NativeLocator.SourceLinkedCostObligation
#audit Apeiron.NativeLocator.current_source_blockers_exact
#audit Apeiron.NativeLocator.extraction_targets_exact
#audit Apeiron.NativeLocator.ReserveBeforeRetainContract
#audit Apeiron.NativeLocator.LocatorReservationReceipt
#audit Apeiron.NativeLocator.ScalarArenaJointWitnesses
#audit Apeiron.NativeLocator.scalarArena_joint_success_coverage
#audit Apeiron.NativeLocator.scalarArena_joint_failure_committed_stutter

-- Extracted coordinate identity for the current prepared-core carrier bridge
#audit Apeiron.GrowableIdentity.coord_spread_id_injective

-- Extracted prepared-core carrier identity against canonical HMap (Sigma × I64) keys
#audit Apeiron.GrowableIdentity.coord_unspread_spread_id
#audit Apeiron.GrowableIdentity.decodeStored_identity
#audit Apeiron.GrowableIdentity.decodeRegen_identity
#audit Apeiron.GrowableIdentity.i64_codec_roundtrip
#audit Apeiron.GrowableIdentity.stored_clone_identity
#audit Apeiron.GrowableIdentity.stored_matches_iff
#audit Apeiron.GrowableIdentity.regen_slot_matches_iff_of_reachable
#audit Apeiron.GrowableIdentity.regen_address_held_agrees
#audit Apeiron.GrowableIdentity.keyRep_identity
#audit Apeiron.GrowableIdentity.canonical_sigma_insert_separate

-- Extracted prepared-core sizing, direct indexing, density and destination bounds
#audit Apeiron.GrowableGeometry.extracted_shard_of_direct_word
#audit Apeiron.GrowableGeometry.preparedFlatIndex_of_admission
#audit Apeiron.GrowableGeometry.flatIndex_lt
#audit Apeiron.GrowableGeometry.directLane_lt_two_pow_24
#audit Apeiron.GrowableGeometry.laneExtent_representable
#audit Apeiron.GrowableGeometry.checkedPow2Ceil_spec
#audit Apeiron.GrowableGeometry.checkedSizing_positive_spec
#audit Apeiron.GrowableGeometry.checkedSizing_matches_formula
#audit Apeiron.GrowableGeometry.checkedSizing_group_geometry
#audit Apeiron.GrowableGeometry.empty_region_allocates_nothing
#audit Apeiron.GrowableGeometry.densityAdmits_eq_true
#audit Apeiron.GrowableGeometry.densityAdmits_mono
#audit Apeiron.GrowableGeometry.prepared_destination_in_bounds
#audit Apeiron.GrowableGeometry.geometry_wellFormed

-- Extracted SWAR byte marker and structural no-cross-byte-borrow decomposition
#audit Apeiron.GrowableScan.FixedWidthTrailingZeros
#audit Apeiron.GrowableScan.TrailingZerosInv
#audit Apeiron.GrowableScan.usize_and_one_val
#audit Apeiron.GrowableScan.summary_trailing_zeros_eq
#audit Apeiron.GrowableScan.append_sub_byte_noBorrow
#audit Apeiron.GrowableScan.byte_zero_marker
#audit Apeiron.GrowableScan.zero_marker_decomposition
#audit Apeiron.GrowableScan.extracted_zero_swar_exact
#audit Apeiron.GrowableScan.lo_mul_broadcast
#audit Apeiron.GrowableScan.extracted_eq_swar_exact
#audit Apeiron.GrowableScan.gather_spreadMarkerBits
#audit Apeiron.GrowableScan.scan_exact

-- Extracted search apply under the unextracted SearchDecisionCorrect premise
#audit Apeiron.GrowableSearch.SearchDecisionCorrect
#audit Apeiron.GrowableSearch.QueryAddressCorrect
#audit Apeiron.GrowableSearch.QueryAddressCorrect.generatedAddress
#audit Apeiron.GrowableSearch.QueryAddressCorrect.vacantLocation
#audit Apeiron.GrowableSearch.FreshLengthBound
#audit Apeiron.GrowableSearch.ProductionWordCarrier
#audit Apeiron.GrowableSearch.GrowableCallbackLaws
#audit Apeiron.GrowableSearch.GrowableCallbackLaws.carrier
#audit Apeiron.GrowableSearch.GrowableCallbackLaws.generatedIdentity
#audit Apeiron.GrowableSearch.GrowableCallbackLaws.newFirst
#audit Apeiron.GrowableSearch.GrowableCallbackLaws.identity
#audit Apeiron.GrowableSearch.GrowableCallbackLaws.exactlyOnce
#audit Apeiron.GrowableSearch.queryAddressCorrect_vacant_firstFit
#audit Apeiron.GrowableSearch.queryAddressCorrect_generatedAddress
#audit Apeiron.GrowableSearch.freshLengthBound_vacant
#audit Apeiron.GrowableSearch.growableCallbackLaws_identity
#audit Apeiron.GrowableSearch.growableCallbackLaws_generated_once
#audit Apeiron.GrowableSearch.wordMerge_newFirst
#audit Apeiron.GrowableSearch.word_stored_apply_get_found
#audit Apeiron.GrowableSearch.word_stored_apply_get_absent
#audit Apeiron.GrowableSearch.word_stored_search_decision_simulates
#audit Apeiron.GrowableSearch.word_regen_search_decision_simulates
#audit Apeiron.GrowableSearch.word_stored_apply_insert_found_flat
#audit Apeiron.GrowableSearch.word_stored_apply_get_after_found_flat
#audit Apeiron.GrowableSearch.word_stored_apply_insert_found_flat_self
#audit Apeiron.GrowableSearch.word_stored_apply_insert_vacant_flat
#audit Apeiron.GrowableSearch.word_stored_apply_insert_vacant_flat_self
#audit Apeiron.GrowableSearch.word_regen_apply_insert_vacant_flat
#audit Apeiron.GrowableSearch.word_regen_apply_insert_vacant_flat_self
#audit Apeiron.GrowableSearch.word_regen_apply_insert_found_flat
#audit Apeiron.GrowableSearch.word_regen_apply_insert_found_flat_self
#audit Apeiron.GrowableSearch.word_stored_apply_insert_found_spill
#audit Apeiron.GrowableSearch.word_regen_apply_insert_found_spill
#audit Apeiron.GrowableSearch.word_stored_apply_insert_found_spill_self
#audit Apeiron.GrowableSearch.word_regen_apply_insert_found_spill_self
#audit Apeiron.GrowableSearch.word_stored_apply_insert_vacant_spill
#audit Apeiron.GrowableSearch.word_regen_apply_insert_vacant_spill
#audit Apeiron.GrowableSearch.word_stored_apply_insert_vacant_spill_self
#audit Apeiron.GrowableSearch.word_regen_apply_insert_vacant_spill_self
#audit Apeiron.GrowableSearch.word_stored_apply_insert_needsGrowth
#audit Apeiron.GrowableSearch.word_regen_apply_insert_needsGrowth
#audit Apeiron.GrowableSearch.committed_insertWith_self
#audit Apeiron.GrowableSearch.committed_insertWith_ne
#audit Apeiron.GrowableSearch.search_decision_simulates

-- Extracted production relocation preserves one canonical observation for both carriers
#audit Apeiron.GrowableRelocation.nested_region_payload_set_readback
#audit Apeiron.GrowableRelocation.nested_region_set_readback
#audit Apeiron.GrowableRelocation.writeFlat_frame_flat
#audit Apeiron.GrowableRelocation.writeFlat_frame_spill
#audit Apeiron.GrowableRelocation.writeSpill_frame_flat
#audit Apeiron.GrowableRelocation.writeSpill_frame_other_shard
#audit Apeiron.GrowableRelocation.writeSpill_frame_same_shard
#audit Apeiron.GrowableRelocation.word_stored_relocate_entry_flat_preserves
#audit Apeiron.GrowableRelocation.word_regen_relocate_entry_flat_preserves
#audit Apeiron.GrowableRelocation.word_stored_relocate_entry_spill_preserves
#audit Apeiron.GrowableRelocation.word_regen_relocate_entry_spill_preserves
#audit Apeiron.GrowableRelocation.word_stored_relocate_all_preserves
#audit Apeiron.GrowableRelocation.word_regen_relocate_all_preserves

-- The extracted pending commit is the ordinary refined insert apply on rebuilt storage
#audit Apeiron.GrowableRefinement.word_stored_apply_pending_and_commit_eq_apply_insert
#audit Apeiron.GrowableRefinement.word_regen_apply_pending_and_commit_eq_apply_insert

-- Extracted prepared-resource publication and its exact input-side certificate interface
#audit apeiron_word.word_stored_publish_prepared
#audit apeiron_word.word_regen_publish_prepared
#audit apeiron_word.word_capacity_bound_holds
#audit Apeiron.GrowableResources.PendingInput
#audit Apeiron.GrowableResources.storedRelocationKeyRep
#audit Apeiron.GrowableResources.regenRelocationKeyRep
#audit Apeiron.GrowableResources.ProductionRelocationKeyRep
#audit Apeiron.GrowableResources.ProductionRelocationKeyRep.keyRep
#audit Apeiron.GrowableResources.storedProductionRelocationKeyRep
#audit Apeiron.GrowableResources.regenProductionRelocationKeyRep
#audit Apeiron.GrowableResources.CoreCallTrace
#audit Apeiron.GrowableResources.NoCoreEffects
#audit Apeiron.GrowableResources.DestinationInBounds
#audit Apeiron.GrowableResources.MaterializedVacant
#audit Apeiron.GrowableResources.ReceiptMatchesStorage
#audit Apeiron.GrowableResources.receiptMatches_flatExtent
#audit Apeiron.GrowableResources.receiptMatches_spillDirectory
#audit Apeiron.GrowableResources.receiptMatches_regionLengths
#audit Apeiron.GrowableResources.CapacityBound_R
#audit Apeiron.GrowableResources.EveryScalarCapacityCheck_R
#audit Apeiron.GrowableResources.PreparedResourceClauses
#audit Apeiron.GrowableResources.PreparedResources_R
#audit Apeiron.GrowableResources.SourceEnumerationCertificate
#audit Apeiron.GrowableResources.EntryObservationCertificate
#audit Apeiron.GrowableResources.applyPlannedWrite
#audit Apeiron.GrowableResources.PendingSymbolicDecision
#audit Apeiron.GrowableResources.PendingDecisionValid
#audit Apeiron.GrowableResources.PlannedCoverage
#audit Apeiron.GrowableResources.PlannedDirectCover
#audit Apeiron.GrowableResources.TargetGeometryCertificate
#audit Apeiron.GrowableResources.RebuildPlanClauses
#audit Apeiron.GrowableResources.RebuildPlanClauses.keyRep
#audit Apeiron.GrowableResources.RebuildPlanClauses.cloneHeld_eq
#audit Apeiron.GrowableResources.RebuildPlanClauses.addressHeld_eq
#audit Apeiron.GrowableResources.RebuildPlanClauses.locateHeld_eq
#audit Apeiron.GrowableResources.runtimeLocation
#audit Apeiron.GrowableResources.DecisionSymbolicCorrespondence
#audit Apeiron.GrowableResources.DecisionSymbolicCorrespondence.corresponds
#audit Apeiron.GrowableResources.RebuildPlanCorrect_R
#audit Apeiron.GrowableResources.decisionSymbolic_found
#audit Apeiron.GrowableResources.decisionSymbolic_vacant
#audit Apeiron.GrowableResources.decisionSymbolic_vacant_valid
#audit Apeiron.GrowableResources.EveryObservedCapacitySufficient
#audit Apeiron.GrowableResources.rebuildPlan_destination
#audit Apeiron.GrowableResources.rebuildPlan_destinations_disjoint
#audit Apeiron.GrowableResources.rebuildPlan_prefix_vacancy
#audit Apeiron.GrowableResources.rebuildPlan_cloneHeld_binding
#audit Apeiron.GrowableResources.rebuildPlan_addressHeld_binding
#audit Apeiron.GrowableResources.rebuildPlan_locateHeld_binding
#audit Apeiron.GrowableResources.rebuildPlanCorrect_productionKeyRep
#audit Apeiron.GrowableResources.rebuildPlan_stored_keyRep
#audit Apeiron.GrowableResources.rebuildPlan_regen_keyRep
#audit Apeiron.GrowableResources.rebuildPlan_stored_locateHeld_binding
#audit Apeiron.GrowableResources.rebuildPlan_regen_locateHeld_binding
#audit Apeiron.GrowableResources.rebuildPlan_stored_operations_binding
#audit Apeiron.GrowableResources.rebuildPlan_regen_operations_binding
#audit Apeiron.GrowableResources.rebuildPlan_entry_observation
#audit Apeiron.GrowableResources.rebuildPlan_logicalKey_injective
#audit Apeiron.GrowableResources.rebuildPlan_coverage
#audit Apeiron.GrowableResources.rebuildPlan_directCover
#audit Apeiron.GrowableResources.rebuildPlan_targetLiveSet
#audit Apeiron.GrowableResources.rebuildPlan_targetLanes
#audit Apeiron.GrowableResources.rebuildPlan_targetUnique
#audit Apeiron.GrowableResources.rebuildPlan_targetCoverage
#audit Apeiron.GrowableResources.rebuildPlan_targetAddress
#audit Apeiron.GrowableResources.rebuildPlan_targetDirectIndex
#audit Apeiron.GrowableResources.rebuildPlan_targetDirectCover
#audit Apeiron.GrowableResources.rebuildPlan_targetDirectory
#audit Apeiron.GrowableResources.rebuildPlan_targetLength
#audit Apeiron.GrowableResources.preparedResources_materializedVacant
#audit Apeiron.GrowableResources.preparedResources_zero_allocation_merge
#audit Apeiron.GrowableResources.ResourceAdapterSimulates
#audit Apeiron.GrowableResources.ResourceAdapterSimulates.prepared
#audit Apeiron.GrowableResources.ResourceAdapterSimulates.refused
#audit Apeiron.GrowableResources.resourceAdapter_refusal_stutter
#audit Apeiron.GrowableResources.word_capacity_bound_holds_iff
#audit Apeiron.GrowableResources.everyScalarCapacityCheck_of_sufficient
#audit Apeiron.GrowableResources.word_stored_publish_prepared_carries
#audit Apeiron.GrowableResources.word_regen_publish_prepared_carries
#audit Apeiron.GrowableResources.preparedResources_of_publication
#audit Apeiron.GrowableResources.word_stored_reachable_publication_witness
#audit Apeiron.GrowableResources.word_regen_reachable_publication_witness

-- Whole-table flat/spill representation and reserved-to-committed boundary
#audit Apeiron.GrowableRepresentation.Occupied
#audit Apeiron.GrowableRepresentation.LiveSetEqual
#audit Apeiron.GrowableRepresentation.ObservationCorrect
#audit Apeiron.GrowableRepresentation.UniqueRepresentation
#audit Apeiron.GrowableRepresentation.PhysicalCoverage
#audit Apeiron.GrowableRepresentation.DirectIndexCorrect
#audit Apeiron.GrowableRepresentation.AddressCorrect
#audit Apeiron.GrowableRepresentation.SpillDirectoryConsistent
#audit Apeiron.GrowableRepresentation.LengthCorrect
#audit Apeiron.GrowableRepresentation.RepR
#audit Apeiron.GrowableRepresentation.ReservedRepR
#audit Apeiron.GrowableRepresentation.PendingCommittedR
#audit Apeiron.GrowableRepresentation.repR_liveSet
#audit Apeiron.GrowableRepresentation.repR_unique
#audit Apeiron.GrowableRepresentation.repR_coverage
#audit Apeiron.GrowableRepresentation.repR_directCover
#audit Apeiron.GrowableRepresentation.repR_directory
#audit Apeiron.GrowableRepresentation.reservedRepR_of_components
#audit Apeiron.GrowableRepresentation.reservedRepR_liveSet
#audit Apeiron.GrowableRepresentation.reservedRepR_unique
#audit Apeiron.GrowableRepresentation.reservedRepR_coverage
#audit Apeiron.GrowableRepresentation.reservedRepR_directCover
#audit Apeiron.GrowableRepresentation.reservedRepR_directory
#audit Apeiron.GrowableRepresentation.reservedRepR_length
#audit Apeiron.GrowableRepresentation.reservedRepR_iff_rebuiltView
#audit Apeiron.GrowableRepresentation.pendingCommitted_reserved
#audit Apeiron.GrowableRepresentation.pendingCommitted_committed
#audit Apeiron.GrowableRepresentation.sourceLocation
#audit Apeiron.GrowableRepresentation.sourceObservation_eq_entryAt
#audit Apeiron.GrowableRepresentation.repR_singleSource
#audit Apeiron.GrowableRepresentation.destinationLocation
#audit Apeiron.GrowableRepresentation.reservedObservation_eq_entryAt
#audit Apeiron.GrowableRepresentation.reservedRepR_singleDestination

-- Whole rebuild traversal composition for both production carriers
#audit Apeiron.GrowableRebuild.TraversalEffectsPrepared
#audit Apeiron.GrowableRebuild.word_stored_rebuild_preserves
#audit Apeiron.GrowableRebuild.word_regen_rebuild_preserves

#audit Apeiron.Joint.WordI64Rel
#audit Apeiron.Joint.wordI64Rel_rust_unique
#audit Apeiron.Joint.wordI64Rel_song_unique
#audit Apeiron.Joint.wordI64Rel_eq_iff
#audit Apeiron.Joint.HMapRel
#audit Apeiron.Joint.hMapRel_insertWith_replacement
