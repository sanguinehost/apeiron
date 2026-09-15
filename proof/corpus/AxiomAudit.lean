import WordRefinement
import TableRefinement
import TableRefinementGeneric
import TableRefinementBuild
import TableRefinementAdmission
import TableRefinementBuildGeneric
import TableRefinementAdmissionGeneric
import TableRefinementPhysical
import TableRefinementPhysicalBuild
import TableRefinement.PhysicalBuildGeneric
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

-- The executing fixed-shard table refines the canonical total table
#audit Apeiron.TableRefinement.word_table_new_eq_empty
#audit Apeiron.TableRefinement.emptyWordTable_read
#audit Apeiron.TableRefinement.word_table_new_rep
#audit Apeiron.TableRefinement.word_table_new_joint_witness
#audit Apeiron.TableRefinement.table_insert_empty_zero
#audit Apeiron.TableRefinement.word_table_insert_new_first
#audit Apeiron.TableRefinement.word_table_get_new_first
#audit Apeiron.TableRefinement.table_insert_present_zero
#audit Apeiron.TableRefinement.word_table_insert_replaces_present
#audit Apeiron.TableRefinement.word_table_get_replaced
#audit Apeiron.TableRefinement.table_build_loop_two
#audit Apeiron.TableRefinement.word_table_build_ordered
#audit Apeiron.TableRefinement.word_table_build_get_acc
#audit Apeiron.TableRefinement.find_slots_spec
#audit Apeiron.TableRefinement.insert_full_loop_spec
#audit Apeiron.TableRefinement.table_insert_shardFull
#audit Apeiron.TableRefinement.word_table_shardFull_refusal
#audit Apeiron.TableRefinement.word_table_joint_branch_witness

-- Generic-N present-key write preservation through the functional state denotation
#audit Apeiron.TableRefinement.insert_with_present_exact
#audit Apeiron.TableRefinement.stateRead_replacePresent
#audit Apeiron.TableRefinement.insert_with_present_preserves
#audit Apeiron.TableRefinement.insert_with_present_rep_preservation
#audit Apeiron.TableRefinement.insert_with_present_joint_witness

-- Generic-N executing reads through state-indexed generated routing
#audit Apeiron.TableRefinement.find_slots_spec_generic
#audit Apeiron.TableRefinement.findPostN_unique
#audit Apeiron.TableRefinement.find_slots_eq_of_matches
#audit Apeiron.TableRefinement.find_slots_insertVacant_ne
#audit Apeiron.TableRefinement.word_shard_index_spec
#audit Apeiron.TableRefinement.word_shard_index_state_independent
#audit Apeiron.TableRefinement.generatedRoute_exists
#audit Apeiron.TableRefinement.generatedRoute_none_of_find
#audit Apeiron.TableRefinement.generatedRoute_some_of_find
#audit Apeiron.TableRefinement.generatedLocator_routes
#audit Apeiron.TableRefinement.table_get_eq_stateRead_of_route
#audit Apeiron.TableRefinement.table_get_eq_stateRead
#audit Apeiron.TableRefinement.functionalRep_iff_readRep_of_routes
#audit Apeiron.TableRefinement.generatedFunctionalRep_iff_readRep
#audit Apeiron.TableRefinement.readRep_iff_Rep

-- Generic-N absent-key insertion through empty and vacant physical slots
#audit Apeiron.TableRefinement.stateRead_insertVacant
#audit Apeiron.TableRefinement.stateRead_insertEmptyShard
#audit Apeiron.TableRefinement.find_slots_replacePresent
#audit Apeiron.TableRefinement.table_find_insertVacant_ne
#audit Apeiron.TableRefinement.table_find_replacePresent
#audit Apeiron.TableRefinement.find_slots_insertVacant_self
#audit Apeiron.TableRefinement.routes_replacePresent
#audit Apeiron.TableRefinement.routes_insertVacant
#audit Apeiron.TableRefinement.insert_with_empty_shard_exact
#audit Apeiron.TableRefinement.insert_with_loop0_vacant_body_exact
#audit Apeiron.TableRefinement.insert_with_loop0_vacant_spec
#audit Apeiron.TableRefinement.insert_with_loop0_full_spec_generic
#audit Apeiron.TableRefinement.insert_with_vacant_exact
#audit Apeiron.TableRefinement.insert_with_vacant_rep_preservation
#audit Apeiron.TableRefinement.insert_with_vacant_readRep_preservation
#audit Apeiron.TableRefinement.insert_with_present_readRep_preservation
#audit Apeiron.TableRefinement.insert_with_vacant_readRep_preservation_routed
#audit Apeiron.TableRefinement.insert_with_absent_joint_witness

-- Ordered generated build refinement, including success and capacity-refusal loop branches
#audit Apeiron.TableRefinement.sliceEntries_length
#audit Apeiron.TableRefinement.orderedFold_eq_build
#audit Apeiron.TableRefinement.build_insert_premises_success
#audit Apeiron.TableRefinement.emptyWordTable_tableInv
#audit Apeiron.TableRefinement.ShardKeysDistinct
#audit Apeiron.TableRefinement.TableInv2
#audit Apeiron.TableRefinement.emptyWordTable_tableInv2
#audit Apeiron.TableRefinement.occupiedPrefix_distinct_key_count
#audit Apeiron.TableRefinement.insert_with_full_exact
#audit Apeiron.TableRefinement.tableInv_insert_with
#audit Apeiron.TableRefinement.tableInv2_insert_with
#audit Apeiron.TableRefinement.buildInsertResult_of_readRep
#audit Apeiron.TableRefinement.allBuildInsertResults
#audit Apeiron.TableRefinement.table_build_loop_result_readRep
#audit Apeiron.TableRefinement.word_table_build_result_readRep
#audit Apeiron.TableRefinement.table_build_loop_readRep
#audit Apeiron.TableRefinement.word_table_build_readRep
#audit Apeiron.TableRefinement.PrefixShardCapacity
#audit Apeiron.TableRefinement.routedStreamKeys
#audit Apeiron.TableRefinement.DistinctKeysPerShard
#audit Apeiron.TableRefinement.PrefixShardKeyCoverage
#audit Apeiron.TableRefinement.prefixShardCapacity_of_distinctKeysPerShard
#audit Apeiron.TableRefinement.allBuildInsertsAdmitted_of_prefixShardCapacity
#audit Apeiron.TableRefinement.word_table_build_readRep_of_prefixShardCapacity
#audit Apeiron.TableRefinement.word_table_build_readRep_of_distinctKeysPerShard

-- Reachable-prefix stream-only admission
#audit Apeiron.TableRefinement.ReachablePrefixCoverage
#audit Apeiron.TableRefinement.emptyWordTable_reachablePrefixCoverage
#audit Apeiron.TableRefinement.reachablePrefixCoverage_capacity
#audit Apeiron.TableRefinement.reachablePrefix_insert
#audit Apeiron.TableRefinement.table_build_loop_readRep_of_distinctKeysPerShard
#audit Apeiron.TableRefinement.word_table_build_readRep_of_distinctKeysPerShard_alone

-- Exact reachable-prefix key sets and the tight per-step admission condition
#audit Apeiron.TableRefinement.ReachablePrefixExact
#audit Apeiron.TableRefinement.emptyWordTable_reachablePrefixExact
#audit Apeiron.TableRefinement.reachablePrefixExact_insert
#audit Apeiron.TableRefinement.BuildAdmissible
#audit Apeiron.TableRefinement.buildAdmissible_of_distinctKeysPerShard
#audit Apeiron.TableRefinement.word_table_build_readRep_of_buildAdmissible
#audit Apeiron.TableRefinement.word_table_build_refusal_implies_not_buildAdmissible
#audit Apeiron.TableRefinement.reachablePrefixExact_admissible_of_insert_success
#audit Apeiron.TableRefinement.word_table_build_success_implies_buildAdmissible
#audit Apeiron.TableRefinement.word_table_build_refuses_of_not_buildAdmissible
#audit Apeiron.TableRefinement.word_table_build_refuses_iff_not_buildAdmissible

-- Generic-N ordered build refinement with the concrete word callbacks
#audit Apeiron.TableRefinement.emptyWordTableN
#audit Apeiron.TableRefinement.emptyWordTableN_read
#audit Apeiron.TableRefinement.emptyWordTableN_readRep
#audit Apeiron.TableRefinement.BuildInsertPremisesN
#audit Apeiron.TableRefinement.build_insert_premises_successN
#audit Apeiron.TableRefinement.insert_with_full_exactN
#audit Apeiron.TableRefinement.BuildInsertResultN
#audit Apeiron.TableRefinement.buildInsertResult_of_readRepN
#audit Apeiron.TableRefinement.AllBuildInsertResultsN
#audit Apeiron.TableRefinement.allBuildInsertResultsN
#audit Apeiron.TableRefinement.table_build_loop_result_readRepN
#audit Apeiron.TableRefinement.table_build_result_readRepN
#audit Apeiron.TableRefinement.AllBuildInsertsAdmittedN
#audit Apeiron.TableRefinement.table_build_loop_readRepN
#audit Apeiron.TableRefinement.table_build_readRepN

-- Abstract-callback ordered build refinement at general N
#audit Apeiron.TableRefinement.sliceEntriesG_length
#audit Apeiron.TableRefinement.orderedFoldG
#audit Apeiron.TableRefinement.ReadRepG
#audit Apeiron.TableRefinement.BuildInsertResultG
#audit Apeiron.TableRefinement.BuildInsertOkG
#audit Apeiron.TableRefinement.table_build_loop_result_readRepG
#audit Apeiron.TableRefinement.table_build_loop_readRepG
#audit Apeiron.TableRefinement.cloneI64_identity
#audit Apeiron.TableRefinement.buildInsertResultG_word_iff
#audit Apeiron.TableRefinement.buildInsertResultG_word
#audit Apeiron.TableRefinement.buildInsertOkG_word
#audit Apeiron.TableRefinement.table_build_loop_result_readRepG_word
#audit Apeiron.TableRefinement.table_build_loop_readRepG_word
#audit Apeiron.TableRefinement.WordBuildOneStepContractN
#audit Apeiron.TableRefinement.wordBuildOneStepContractN
#audit Apeiron.TableRefinement.wordBuildOneStepContractN_iff_concrete
#audit Apeiron.TableRefinement.wordBuildOneStepContractN_sufficient

-- Generic-N occupied-prefix capacity and stream-level distinct-key admission
#audit Apeiron.TableRefinement.OccupiedPrefixN
#audit Apeiron.TableRefinement.TableInvN
#audit Apeiron.TableRefinement.ShardKeysDistinctN
#audit Apeiron.TableRefinement.TableInv2N
#audit Apeiron.TableRefinement.emptyWordTableN_tableInv
#audit Apeiron.TableRefinement.emptyWordTableN_tableInv2
#audit Apeiron.TableRefinement.occupiedPrefix_distinct_key_countN
#audit Apeiron.TableRefinement.PrefixShardCapacityN
#audit Apeiron.TableRefinement.DistinctKeysPerShardN
#audit Apeiron.TableRefinement.PrefixShardKeyCoverageN
#audit Apeiron.TableRefinement.prefixShardCapacity_of_distinctKeysPerShardN
#audit Apeiron.TableRefinement.allBuildInsertsAdmitted_of_prefixShardCapacityN
#audit Apeiron.TableRefinement.table_build_readRepN_of_distinctKeysPerShardN

-- Exact stream-only admission at general N
#audit Apeiron.TableRefinement.tableInv2N_insert_with
#audit Apeiron.TableRefinement.ReachablePrefixCoverageN
#audit Apeiron.TableRefinement.emptyWordTableN_reachablePrefixCoverageN
#audit Apeiron.TableRefinement.ReachablePrefixExactN
#audit Apeiron.TableRefinement.emptyWordTableN_reachablePrefixExactN
#audit Apeiron.TableRefinement.reachablePrefixExact_insertN
#audit Apeiron.TableRefinement.reachablePrefixExact_admissible_of_insert_successN
#audit Apeiron.TableRefinement.BuildAdmissibleN
#audit Apeiron.TableRefinement.buildAdmissibleN_of_distinctKeysPerShardN
#audit Apeiron.TableRefinement.table_build_readRepN_of_buildAdmissibleN
#audit Apeiron.TableRefinement.table_build_success_implies_buildAdmissibleN
#audit Apeiron.TableRefinement.table_build_refusesN_of_not_buildAdmissibleN
#audit Apeiron.TableRefinement.table_build_refusesN_iff_not_buildAdmissibleN
#audit Apeiron.TableRefinement.wpSpec_bind_iff
#audit Apeiron.TableRefinement.BuildInsertSpecG
#audit Apeiron.TableRefinement.AllBuildInsertSpecsG
#audit Apeiron.TableRefinement.BuildLoopSpecG
#audit Apeiron.TableRefinement.buildLoopSpecG_iff_allBuildInsertSpecsG
#audit Apeiron.TableRefinement.buildLoopSpecG_true_iff_buildInsertResultG
#audit Apeiron.TableRefinement.buildLoopSpecG_false_iff_buildInsertOkG

-- Universal physical representation at general N
#audit Apeiron.TableRefinement.globalKeysDistinct_of_tableInv2N_homePlaced
#audit Apeiron.TableRefinement.physicalInv_iff
#audit Apeiron.TableRefinement.readRep_exists
#audit Apeiron.TableRefinement.offHomeTable_not_homePlaced
#audit Apeiron.TableRefinement.readRep_exists_without_physicalShape
#audit Apeiron.TableRefinement.emptyWordTableN_homePlaced
#audit Apeiron.TableRefinement.emptyWordTableN_physicalInv
#audit Apeiron.TableRefinement.homePlaced_insert_with
#audit Apeiron.TableRefinement.PhysicalInsertResult
#audit Apeiron.TableRefinement.physicalInv_insert_with
#audit Apeiron.TableRefinement.insert_with_zero_exact
#audit Apeiron.TableRefinement.physicalInv_insert_with_zero
#audit Apeiron.TableRefinement.physicalInv_insert_with_all

-- Physical build postcondition and zero-capacity build characterization at general N
#audit Apeiron.TableRefinement.table_build_physicalInv_of_buildAdmissibleN
#audit Apeiron.TableRefinement.wordBuildN_zero_of_empty
#audit Apeiron.TableRefinement.wordBuildN_zero_refuses_of_nonempty
#audit Apeiron.TableRefinement.wordBuildN_zero_ok_iff_empty

-- Arbitrary-callback table execution, physical preservation, admission, and real build lift
#audit Apeiron.TableRefinement.CallbackLawsG
#audit Apeiron.TableRefinement.insert_with_present_callback_exact
#audit Apeiron.TableRefinement.insert_with_empty_shard_callback_exact
#audit Apeiron.TableRefinement.insert_with_loop0_vacant_callback_exact
#audit Apeiron.TableRefinement.insert_with_populated_vacant_callback_exact
#audit Apeiron.TableRefinement.insert_with_loop0_full_callback_exact
#audit Apeiron.TableRefinement.insert_with_populated_full_callback_exact
#audit Apeiron.TableRefinement.insert_with_zero_callback_exact
#audit Apeiron.TableRefinement.find_slots_spec_callback_exact
#audit Apeiron.TableRefinement.firstVacant_or_fullShardG
#audit Apeiron.TableRefinement.find_slots_callback_result_exact
#audit Apeiron.TableRefinement.find_slots_callback_some_exact
#audit Apeiron.TableRefinement.find_slots_callback_none_exact
#audit Apeiron.TableRefinement.PhysicalInvG
#audit Apeiron.TableRefinement.table_new_eq_emptyTableG
#audit Apeiron.TableRefinement.table_branch_callback_exact
#audit Apeiron.TableRefinement.CallbackInsertOutcomeG
#audit Apeiron.TableRefinement.insert_with_callback_result_exact
#audit Apeiron.TableRefinement.homePlacedG_replacePresent
#audit Apeiron.TableRefinement.globalKeysDistinctG_replacePresent
#audit Apeiron.TableRefinement.physicalClausesG_replacePresent
#audit Apeiron.TableRefinement.get_callback_of_uniquePresent
#audit Apeiron.TableRefinement.get_callback_of_absent
#audit Apeiron.TableRefinement.readRepG_replacePresent
#audit Apeiron.TableRefinement.physicalInvG_replacePresent
#audit Apeiron.TableRefinement.emptyTableG_physicalInv
#audit Apeiron.TableRefinement.physicalInvG_populated_vacant
#audit Apeiron.TableRefinement.physicalInvG_empty_shard
#audit Apeiron.TableRefinement.CallbackInsertPostG
#audit Apeiron.TableRefinement.insert_with_callback_admitted
#audit Apeiron.TableRefinement.InsertCapacityG
#audit Apeiron.TableRefinement.BuildCapacityG
#audit Apeiron.TableRefinement.OccupiedPositionG
#audit Apeiron.TableRefinement.OccupiedCountG
#audit Apeiron.TableRefinement.PrefixPhysicalCountBoundG
#audit Apeiron.TableRefinement.routedPrefixKeysG
#audit Apeiron.TableRefinement.DistinctRoutedKeysPerShardG
#audit Apeiron.TableRefinement.PrefixPhysicalKeyCoverageG
#audit Apeiron.TableRefinement.PhysicalKeyCoverageAtG
#audit Apeiron.TableRefinement.routedPrefixKeysG_mono
#audit Apeiron.TableRefinement.routedPrefixKeysG_next_key
#audit Apeiron.TableRefinement.emptyTableG_physicalKeyCoverageAtG
#audit Apeiron.TableRefinement.physicalKeyCoverageAtG_replacePresent_present
#audit Apeiron.TableRefinement.physicalKeyCoverageAtG_replacePresent_vacant
#audit Apeiron.TableRefinement.physicalKeyCoverageAtG_insertEmptyShard
#audit Apeiron.TableRefinement.emptySliceG
#audit Apeiron.TableRefinement.BuildAdmissibleG
#audit Apeiron.TableRefinement.prefixPhysicalCountBoundG_of_distinctRoutedKeysPerShardG
#audit Apeiron.TableRefinement.buildCapacityG_of_prefixPhysicalCountBoundG
#audit Apeiron.TableRefinement.buildCapacityG_of_distinctRoutedKeysPerShardG
#audit Apeiron.TableRefinement.buildCapacityG_implies_buildAdmissibleG
#audit Apeiron.TableRefinement.emptySliceG_buildCapacityG
#audit Apeiron.TableRefinement.table_build_physicalInv_of_buildAdmissibleG
#audit Apeiron.TableRefinement.table_build_physicalInv_of_distinctRoutedKeysPerShardG
#audit Apeiron.TableRefinement.table_build_physicalInv_of_distinctRoutedKeysPerShardG_reachable
#audit Apeiron.TableRefinement.table_build_emptySliceG_physicalInv
#audit Apeiron.TableRefinement.table_build_zero_of_empty
#audit Apeiron.TableRefinement.table_build_zero_refuses_of_nonempty
#audit Apeiron.TableRefinement.table_build_zero_ok_iff_empty
#audit Apeiron.TableRefinement.table_build_shardFull_requires_not_buildAdmissibleG
#audit Apeiron.TableRefinement.table_build_physicalInv_of_distinctRoutedKeysPerShardG_all
#audit Apeiron.Joint.WordI64Rel
#audit Apeiron.Joint.wordI64Rel_rust_unique
#audit Apeiron.Joint.wordI64Rel_song_unique
#audit Apeiron.Joint.wordI64Rel_eq_iff
#audit Apeiron.Joint.HMapRel
#audit Apeiron.Joint.hMapRel_insertWith_replacement
