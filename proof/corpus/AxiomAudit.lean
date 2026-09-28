import WordRefinement
import Native.HyperbolicBase
import Native.LocatorSpec
import Security.InterpretationWitnesses
import Growable.Identity
import Growable.Geometry
import Growable.Scan
import Growable.Search
import Growable.Relocation
import Growable.RelocationWF
import Growable.Refinement
import Growable.Resources
import Growable.Representation
import Growable.Rebuild
import Growable.ReadLoop
import Growable.Read
import Growable.ReadComposition
import Growable.Classifier
import Numeric.Scale
import Numeric.ExpRefinement
import Numeric.FormatRefinement
import Numeric.RsqrtRefinement
import Numeric.RopeRefinement
import Numeric.EngineSiluRefinement
import Numeric.EngineNormRefinement
import Numeric.EngineDotRefinement
import Numeric.PackingRefinement
import Numeric.EngineMatvecRefinement
import Numeric.EngineSamplerRefinement
import Numeric.EngineLmHeadRefinement
import Numeric.EngineAttnRefinement
import Spirv.Fragment
import Spirv.Semantics
import Spirv.Decode
import Spirv.Dp4a
import Spirv.Check
import Growable.LocateLoop
import Nous.Core.Autograd
import Nous.Core.Fold
import Nous.Model.Decomposition
import Nous.Model.Model
import Nous.Model.ErrorComposition
import Nous.Model.Attention
import Nous.Model.Instances
import Nous.Model.SiluChannel
import Nous.Model.RopeOrbit
import Nous.Model.ParabolicAttention
import Nous.Model.EllipticAttention
import Nous.Model.TrichotomyAttention
import Substrate.Algebra.Neural.MirasCountLM
import Substrate.Algebra.Neural.KneserNey
import Substrate.Algebra.Neural.CountLMFloor
import Substrate.Algebra.Neural.AttentionSpinor
import Substrate.Algebra.Neural.MLPrimitivesBridge
import Substrate.Algebra.Neural.TrainingCollapse
import Substrate.Algebra.Neural.FrontierLadder
import Substrate.Algebra.NullCell.SpinorFold
import Kepler.Certificate
import Joint.WordBridge
import Numeric.AttnSoftmaxPerturb
import Numeric.AttnLoopScore
import Numeric.AttnLoopOutput
import Numeric.AttnHeadWindowsScore
import Numeric.AttnHeadWindowsValue
import Numeric.AttnScoreStage
import Numeric.AttnHeadRefines
import Numeric.DequantRefinement
import Numeric.DequantQ4K
import Numeric.DequantQ6K
import Numeric.DequantKWitness
import Numeric.DequantRefusals
import Numeric.DequantQ8_0
import Numeric.PackingUnpack
import Numeric.PackingEpilogueFold
import Numeric.RopeHeadContent
import Numeric.RopeHeadWitness
import Numeric.SiluSaturation
import Numeric.RmsNormIdeal
import Nous.Model.BlockComposition
import Nous.Model.BlockGrid
import Spirv.CheckScan
import Numeric.DequantQ5_0
import Spirv.CheckScanState
import Numeric.DequantQ2K
import Spirv.CheckApplyShape
import Numeric.Sylvester
import Numeric.DequantQ1Q2
import Numeric.DequantQ1Q2Block
import Numeric.DequantTQ2_0
import Numeric.DequantTQ1_0
import Numeric.DequantTQ1_0Row
import Numeric.DequantPTQ1_0
import Numeric.HadamardStage
import Numeric.HadamardRefinement
import Numeric.DequantLowBitRefusals
import Numeric.MatvecBridge
import Numeric.HadamardFixed
import Nous.Model.FaceRecurrence
import Nous.Model.FaceContraction
import Nous.Model.SpeculativeAccept
import Nous.Model.RmsVectorStage
import Nous.Model.RotationStage
import Numeric.MatvecBridgeGrid
import Nous.Model.BlockGrounded
import Numeric.HadamardFixedSpec
import Numeric.KvReachable
import Spirv.CheckApplyShapeAssembly
import Numeric.PackingDecodeFields
import Numeric.RopeRotate
import Nous.Model.AttnStage
import Nous.Model.BlockGroundedWitness
import Nous.Model.AttnStageWitness
import Nous.Model.TopKRouter
import Numeric.PackingDecodeBlock
import Nous.Model.BlockGroundedWitness1
import Nous.Model.BlockMulBox
import Nous.Model.MoeStage
import Numeric.DequantQ3K
import Numeric.MatvecChunk
import Policy.Laws
import Policy.Analysis
import Numeric.ChartDot
import Numeric.MoeRouter
import Numeric.LutPQ2
import Numeric.PrunedArgmax
import Numeric.ColumnCover
import Nous.Combinator.Church
import Nous.Combinator.ListHom
import Numeric.RopeTurns
import Nous.Combinator.Orbit
import Nous.Combinator.DoubleCover
import Nous.Seq.Fenwick
import Nous.Seq.LazySoftmax
import Nous.Seq.GatedDelta
import Nous.Seq.CertifiedSoftmax
import Nous.Seq.ProductKeys
import Nous.Seq.HyperbolicAddress
import Nous.Seq.ThresholdCells
import Numeric.BitplaneArgmax
import Nous.Model.SpeculativeRollback
import Nous.Combinator.Sphere
import Substrate.StationaryPhase.SpinorLoop
import Nous.Model.JacobiDecode
import Nous.Seq.AffineMixer
import Nous.Model.EllipticModes
import Nous.Combinator.CellTransform
import Nous.Model.DualCertificate
import Nous.Seq.TwoChannelRouter
import Nous.Seq.BitplaneRouter
import Nous.Seq.StatePrecision
import Nous.Seq.ChunkedRounding
import Nous.Model.SpectralTrichotomy
import Nous.Model.CertifiedDecision
import Spirv.SplitLimb
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

-- Generated elliptic address pack/unpack roundtrip: first producer of the address-face refinement
#audit Apeiron.NativeLocator.Native.Address.pack_roundtrip
#audit Apeiron.NativeLocator.Native.Address.home_face_denote
#audit Apeiron.NativeLocator.Native.Address.h3_40_new_ok
#audit Apeiron.NativeLocator.Native.Address.home_face_joint_witness
#audit Apeiron.NativeLocator.Native.Address.held_address_agrees
#audit Apeiron.NativeLocator.Native.Address.held_address_joint_witness

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
#audit Apeiron.GrowableScan.summary_set_maybe_one_word_witness
#audit Apeiron.GrowableScan.oneWordZeroRegion
#audit Apeiron.GrowableScan.oneSlotZeroRegion
#audit Apeiron.GrowableScan.summary_set_implies_maybe
#audit Apeiron.GrowableScan.publish_live_implies_summary_maybe
#audit Apeiron.GrowableScan.u64_one_shift_nonzero
#audit Apeiron.GrowableScan.u64_or_mask_and_mask
#audit Apeiron.GrowableScan.summary_remainder_lt64
#audit Apeiron.GrowableScan.summary_mask_nonzero
#audit Apeiron.GrowableScan.summary_mask_zero_word_witness
#audit Apeiron.GrowableScan.summary_index_word_lt_len
#audit Apeiron.GrowableScan.summary_max_shift_six
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

-- Summary shape, recorded-bit exclusion, tag readback and fingerprint exactness (read-classifier row 2)
#audit Apeiron.GrowableScan.SpillSummaryShape
#audit Apeiron.GrowableScan.spillSummaryShape
#audit Apeiron.GrowableScan.SummaryBitRecorded
#audit Apeiron.GrowableScan.summary_maybe_false_excludes_recorded
#audit Apeiron.GrowableScan.tag_ne_zero
#audit Apeiron.GrowableScan.publish_live_tag_readback
#audit Apeiron.GrowableScan.spillFingerprint
#audit Apeiron.GrowableScan.SpillFingerprintExact
#audit Apeiron.GrowableScan.spillFingerprintExact
#audit Apeiron.GrowableScan.spillFingerprint_eq_of_identity
#audit Apeiron.GrowableScan.spillFingerprint_collision_witness
#audit Apeiron.GrowableScan.SpillTagHit
#audit Apeiron.GrowableScan.fingerprint_hit_requires_full_matches
#audit Apeiron.GrowableScan.word_at_exact
#audit Apeiron.GrowableScan.ascendingByteLine
#audit Apeiron.GrowableScan.mapBytes_lane
#audit Apeiron.GrowableScan.from_le_bytes_lane
#audit Apeiron.GrowableScan.from_le_bytes_end_lanes
#audit Apeiron.GrowableScan.tagMarkers_from_le_bytes_lane
#audit Apeiron.GrowableScan.tagMarkers_from_le_bytes_word
#audit Apeiron.GrowableScan.tagMarkers_asymmetric_witness
#audit Apeiron.GrowableScan.generated_clear_hits_strict
#audit Apeiron.GrowableScan.generated_tail_steps_in_bound
#audit Apeiron.GrowableScan.least_set_lane_selects
#audit Apeiron.GrowableScan.least_set_lane_exact
#audit Apeiron.GrowableScan.least_set_lane_in_range
#audit Apeiron.GrowableScan.least_set_lane_high_fixture
#audit Apeiron.GrowableScan.match_tag_window_exact
#audit Apeiron.GrowableScan.match_tag_exact
#audit Apeiron.GrowableScan.generated_clear_hits_exact
#audit Apeiron.GrowableScan.generated_clear_hits_live_decrease
#audit Apeiron.GrowableScan.generated_spill_read_clears_selected
#audit Apeiron.GrowableReadLoop.generated_outer_stops_on_hit
#audit Apeiron.GrowableReadLoop.generated_outer_stops_at_limit
#audit Apeiron.GrowableReadLoop.generated_tail_indices_eligible
#audit Apeiron.GrowableReadLoop.stored_candidate_full_match
#audit Apeiron.GrowableReadLoop.InnerCompared
#audit Apeiron.GrowableReadLoop.InnerReady
#audit Apeiron.GrowableReadLoop.InnerSweepInv
#audit Apeiron.GrowableReadLoop.InnerSweepOutcome
#audit Apeiron.GrowableReadLoop.generated_inner_sweep_outcome
#audit Apeiron.GrowableReadLoop.OuterStepReady
#audit Apeiron.GrowableReadLoop.OuterStepResult
#audit Apeiron.GrowableReadLoop.OuterSweepInv
#audit Apeiron.GrowableReadLoop.OuterSweepOutcome
#audit Apeiron.GrowableReadLoop.generated_outer_sweep_outcome
#audit Apeiron.GrowableReadLoop.generated_outer_none_covers_steps
#audit Apeiron.GrowableReadLoop.generated_tail_group_surjective
#audit Apeiron.GrowableReadLoop.generated_outer_none_covers_groups
#audit Apeiron.GrowableReadLoop.generated_outer_increment_checked
#audit Apeiron.GrowableReadLoop.generated_group_bounded
#audit Apeiron.GrowableReadLoop.inner_ready_of_scan_geometry
#audit Apeiron.GrowableReadLoop.outer_step_ready_of_geometry
#audit Apeiron.GrowableReadLoop.repR_region_outer_ready
#audit Apeiron.GrowableReadLoop.repR_stored_region_outer_ready
#audit Apeiron.GrowableReadLoop.grouped_admission_of_checked_branch
#audit Apeiron.GrowableReadLoop.repR_summary_read_branch
#audit Apeiron.GrowableReadLoop.repR_live_spill_summary_positive
#audit Apeiron.GrowableRepresentation.summary_set_frames
#audit Apeiron.GrowableReadLoop.StoredLiveTagAddress
#audit Apeiron.GrowableReadLoop.StoredTaggedRepR
#audit Apeiron.GrowableReadLoop.stored_live_tag_address_of_empty_repR
#audit Apeiron.GrowableReadLoop.word_stored_table_new_tagged_repR
#audit Apeiron.GrowableReadLoop.stored_live_tag_address_of_home_shards_eq
#audit Apeiron.GrowableReadLoop.stored_live_tag_address_replace_spill_value
#audit Apeiron.GrowableReadLoop.word_stored_apply_insert_found_spill_keeps_tags
#audit Apeiron.GrowableReadLoop.generated_publish_live_frames
#audit Apeiron.GrowableReadLoop.stored_live_tag_address_publish_spill_of_frame
#audit Apeiron.GrowableReadLoop.stored_live_tag_address_publish_spill
#audit Apeiron.GrowableReadLoop.word_stored_apply_insert_found_flat_keeps_tags
#audit Apeiron.GrowableReadLoop.word_stored_apply_insert_vacant_flat_keeps_tags
#audit Apeiron.GrowableReadLoop.word_stored_apply_insert_vacant_spill_keeps_tags
#audit Apeiron.GrowableRepresentation.repR_singleSource
#audit Apeiron.GrowableRepresentation.destinationLocation
#audit Apeiron.GrowableRepresentation.reservedObservation_eq_entryAt
#audit Apeiron.GrowableRepresentation.reservedRepR_singleDestination
#audit Apeiron.GrowableRepresentation.word_stored_vacant_spill_return_shards
#audit Apeiron.GrowableRepresentation.word_stored_vacant_flat_return_frame
#audit Apeiron.GrowableReadLoop.repR_stored_published_tag_address
#audit Apeiron.GrowableReadLoop.stored_live_tags_initial_reserved
#audit Apeiron.GrowableReadLoop.prepared_stored_live_tags_initial
#audit Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_run
#audit Apeiron.GrowableRepresentation.word_stored_relocate_entry_flat_run
#audit Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_publishes_tag_address
#audit Apeiron.GrowableRepresentation.word_stored_relocate_entry_spill_entryAt
#audit Apeiron.GrowableRepresentation.reservedRepR_singleDestination_of_relocated
#audit Apeiron.GrowableRepresentation.repR_singleSource_logical
#audit Apeiron.GrowableRepresentation.word_stored_relocate_all_preserves_reserved
#audit Apeiron.GrowableReadLoop.ReservedStoredLiveTagAddress
#audit Apeiron.GrowableReadLoop.stored_relocate_entry_spill_keeps_reserved_tags
#audit Apeiron.GrowableReadLoop.stored_relocate_entry_flat_keeps_reserved_tags
#audit Apeiron.GrowableReadLoop.word_stored_relocate_all_keeps_reserved_tags
#audit Apeiron.GrowableReadLoop.word_stored_relocate_all_keeps_reserved_tags_from_prepared
#audit Apeiron.GrowableReadLoop.generated_spill_read_unfold
#audit Apeiron.GrowableReadLoop.spill_read_none_implies_no_full_match
#audit Apeiron.GrowableReadLoop.spill_read_some_implies_full_match
#audit Apeiron.GrowableReadLoop.spill_read_none_implies_no_shard_full_match
#audit Apeiron.GrowableReadLoop.spill_read_after_relocate_all_none_implies_no_full_match

-- Public read refinement: constructor reachability, the direct-hit partition, and the
-- localization/absence core feeding partitions 2-5
#audit Apeiron.GrowableRead.ConstructorReachableRepR
#audit Apeiron.GrowableRead.word_stored_table_get_at_direct_hit
#audit Apeiron.GrowableRead.word_regen_table_get_at_direct_hit
#audit Apeiron.GrowableRead.query_rep
#audit Apeiron.GrowableRead.absent_of_exclusions
#audit Apeiron.GrowableRead.default_of_absent
#audit Apeiron.GrowableRead.spill_read_none_of_summary_miss
#audit Apeiron.GrowableRead.flat_index_none_pin
#audit Apeiron.GrowableRead.flat_index_some_pin
#audit Apeiron.GrowableRead.flat_index_compute
#audit Apeiron.GrowableRead.word_stored_table_get_at_direct_none
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_none
#audit Apeiron.GrowableRead.flat_vacant_entry_none
#audit Apeiron.GrowableRead.flat_vacant_exclusion_of_prepared
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_vacant
#audit Apeiron.GrowableRead.flat_occupied_ne_exclusion
#audit Apeiron.GrowableRead.flat_occupied_exclusion_of_prepared
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_occupied_miss
#audit Apeiron.GrowableRead.spill_read_some_tagged_full_match
#audit Apeiron.GrowableRead.spill_read_some_entryAt
#audit Apeiron.GrowableRead.spill_read_none_entryAt_ne
#audit Apeiron.GrowableRead.valid_n_16
#audit Apeiron.GrowableRead.StoredReadRoute
#audit Apeiron.GrowableRead.word_stored_table_get_at_direct_none_hit
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_none_hit
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_occupied_miss_hit
#audit Apeiron.GrowableRead.word_stored_table_get_at_direct_none_sweep_none
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_none_sweep_none
#audit Apeiron.GrowableRead.word_stored_table_get_at_flat_occupied_miss_sweep_none
#audit Apeiron.GrowableRead.word_stored_table_get_at_root

#audit Apeiron.GrowableRelocation.generated_same_location_total
#audit Apeiron.GrowableRelocation.retarget_held_eq_ok_self_of_same
#audit Apeiron.GrowableRelocation.retarget_held_eq_ok_self_of_no_handle
#audit Apeiron.GrowableRelocation.retarget_held_eq_ok_self_of_link_out_of_bounds
#audit Apeiron.GrowableRelocation.retarget_held_eq_ok_self_of_current_distinct
#audit Apeiron.GrowableRelocation.retarget_held_eq_ok_set_of_current_matches
#audit Apeiron.GrowableRelocation.retarget_held_fails_of_linked_handle_fail
#audit Apeiron.GrowableRelocation.retarget_held_succeeds_iff
#audit Apeiron.GrowableRelocation.retarget_held_refuses_of_linked_handle_fail

-- Locator well-formedness: derived-read bounds, totality class lemmas and preservation
#audit Apeiron.GrowableRelocation.read_word_ok
#audit Apeiron.GrowableRelocation.DirectoryWF.of_words_eq
#audit Apeiron.GrowableRelocation.EdgeIndexWF.of_words_eq
#audit Apeiron.GrowableRelocation.IdentityNodeWF.of_words_eq
#audit Apeiron.GrowableRelocation.ResidueNodeWF.of_words_eq
#audit Apeiron.GrowableRelocation.DirectoryWF.edgeIndexWF_of_first_edge
#audit Apeiron.GrowableRelocation.DirectoryWF.edgeIndexWF_of_next
#audit Apeiron.GrowableRelocation.DirectoryWF.identityNodeWF_of_edge_child
#audit Apeiron.GrowableRelocation.DirectoryWF.identityNodeWF_of_identity_root
#audit Apeiron.GrowableRelocation.DirectoryWF.residueNodeWF_of_child
#audit Apeiron.GrowableRelocation.directory_new_wf
#audit Apeiron.GrowableRelocation.recycle_scratch_wf
#audit Apeiron.GrowableRelocation.recycle_staged_wf

-- Extracted search apply under the unextracted SearchDecisionCorrect premise
#audit Apeiron.GrowableSearch.SearchDecisionCorrect
#audit Apeiron.GrowableSearch.SummaryMissSound
#audit Apeiron.GrowableSearch.AbsenceCertificate
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

-- Classifier producer bridges: RepR to the Search absence certificate (row 7)
#audit Apeiron.GrowableClassifier.summaryMissSound_of_repR
#audit Apeiron.GrowableClassifier.flatMatchAdmitted_of_repR
#audit Apeiron.GrowableClassifier.spillMatchRouted_of_repR
#audit Apeiron.GrowableClassifier.repR_absenceCertificate_of_summaryMiss
#audit Apeiron.GrowableClassifier.repR_absent_of_summaryMiss
#audit Apeiron.GrowableClassifier.DecisionProvenance
#audit Apeiron.GrowableClassifier.search_decision_provenance
#audit Apeiron.GrowableClassifier.search_decision_wellFormed
#audit Apeiron.GrowableClassifier.search_decision_found_flat_guards
#audit Apeiron.GrowableClassifier.vec_index_readback
#audit Apeiron.GrowableClassifier.table_matches_of_guards
#audit Apeiron.GrowableClassifier.stored_entryAt_flat
#audit Apeiron.GrowableClassifier.regen_entryAt_flat
#audit Apeiron.GrowableClassifier.foundDenotes
#audit Apeiron.GrowableClassifier.foundDenotes_of_repR
#audit Apeiron.GrowableClassifier.search_decision_found_flat_denotes_stored
#audit Apeiron.GrowableClassifier.search_decision_found_flat_denotes_regen
#audit Apeiron.GrowableClassifier.stored_entryAt_spill
#audit Apeiron.GrowableClassifier.regen_entryAt_spill
-- Statement only: a `spill_locate` `.Found` carries no full-match certificate yet (row 7 residual)
#audit Apeiron.GrowableClassifier.SpillLocateFoundFullMatch

-- Locator loop totality under the 64-bit width premise (-CONT)
#audit Apeiron.GrowableRelocationWF.cast_usize_val
#audit Apeiron.GrowableRelocationWF.locator_word_const_vals
#audit Apeiron.GrowableRelocationWF.usize_max_room
#audit Apeiron.GrowableRelocationWF.find_edge_loop_succeeds
#audit Apeiron.GrowableRelocationWF.find_edge_succeeds
#audit Apeiron.GrowableRelocationWF.walk_residue_loop_succeeds
#audit Apeiron.GrowableRelocationWF.walk_residue_succeeds
#audit Apeiron.GrowableRelocationWF.identity_terminal_val
#audit Apeiron.GrowableRelocationWF.residue_identity_val
#audit Apeiron.GrowableRelocationWF.vec_deref_val
#audit Apeiron.GrowableRelocationWF.walk_identity_loop_succeeds
#audit Apeiron.GrowableRelocationWF.walk_identity_succeeds
#audit Apeiron.GrowableRelocationWF.directory_walk_succeeds
#audit Apeiron.GrowableRelocationWF.linked_handle_succeeds
#audit Apeiron.GrowableRelocationWF.retarget_held_hstep_of_wf

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
#audit Apeiron.GrowableRepresentation.LivePublicationRecorded
#audit Apeiron.GrowableRepresentation.publish_live_recorded
#audit Apeiron.GrowableRepresentation.CarrierAddressRespecting
#audit Apeiron.GrowableRepresentation.SpillSummaryShapes
#audit Apeiron.GrowableRepresentation.SpillSummaryRecorded
#audit Apeiron.GrowableRepresentation.SpillFingerprintExactRegions
#audit Apeiron.GrowableRepresentation.SpillSummaryMissSound
#audit Apeiron.GrowableRepresentation.repR_spillSummaryMissSound
#audit Apeiron.GrowableRepresentation.reservedRepR_summaryShape
#audit Apeiron.GrowableRepresentation.reservedRepR_summaryRecorded
#audit Apeiron.GrowableRepresentation.reservedRepR_fingerprintExact
#audit Apeiron.GrowableRepresentation.word_stored_table_new_repR
#audit Apeiron.GrowableRepresentation.word_regen_table_new_repR

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

-- Finite security interpretation witnesses: model grade, no deployed-cipher claim
#audit Apeiron.Security.InterpretationWitnesses.uniform_observer_equality
#audit Apeiron.Security.InterpretationWitnesses.biased_observer_distinguishes
#audit Apeiron.Security.InterpretationWitnesses.otp_perfect_secrecy
#audit Apeiron.Security.InterpretationWitnesses.biased_unique_compatible_key
#audit Apeiron.Security.InterpretationWitnesses.fixed_key_round_trip
#audit Apeiron.Security.InterpretationWitnesses.cipher_fiber_is_a_singleton
#audit Apeiron.Security.InterpretationWitnesses.public_message_key_map_not_injective
#audit Apeiron.Security.InterpretationWitnesses.short_key_passthrough

-- Conditional Kepler certificate over exact rational denotations
#audit Kepler.Certificate.FiniteBinary64
#audit Kepler.Certificate.FiniteBinary64.denoteRational
#audit Kepler.Certificate.FiniteBinary64.denoteReal
#audit Kepler.Certificate.FiniteBinary64.zeroBits
#audit Kepler.Certificate.signed_zero_denotes_zero
#audit Kepler.Certificate.CertifiedBounds
#audit Kepler.Certificate.CertifiedBounds.Sound
#audit Kepler.Certificate.CertifiedResult
#audit Kepler.Certificate.kepler_abs_defect_enclosure_denoted
#audit Kepler.Certificate.accepted_error_encloses
#audit Kepler.Certificate.refusal_has_no_value
#audit Kepler.Certificate.circular_joint_witness
#audit Kepler.Certificate.positive_joint_witness

-- Stage-2 neural rebase, wave-1 headlines (vendored subtree builds 46/46; cones pre-flighted
-- in an isolated probe, all inside {propext, Classical.choice, Quot.sound})
#audit Sanguine.MirasCountLM.hebb_comm
#audit Sanguine.MirasCountLM.forget_hebb
#audit Sanguine.MirasCountLM.hebb_forget
#audit Sanguine.MirasCountLM.surprise_eq_info
#audit Sanguine.MirasCountLM.countlm_is_exact_miras
#audit Sanguine.KneserNey.kn_normalizes
#audit Sanguine.CountLMFloor.refine_gap_zero_of_kappa_sufficient
#audit Sanguine.AttentionSpinor.attention_is_spinor_scan
#audit Sanguine.AttentionSpinor.spinor_scan_is_order_sensitive
#audit Sanguine.MLPrimitivesBridge.osFuse_is_add
#audit Sanguine.MLPrimitivesBridge.osFuse_grouping_free
#audit Sanguine.MLPrimitivesBridge.output_error_is_function_distortion
#audit Sanguine.MLPrimitivesBridge.aware_scale_is_the_chart
#audit Sanguine.MLPrimitivesBridge.weight_only_is_dual_blind
#audit Sanguine.MLPrimitivesBridge.kn_floor_refinement
#audit Sanguine.TrainingCollapse.ridgeHead_unique
#audit Sanguine.TrainingCollapse.gram_stack
#audit Sanguine.TrainingCollapse.normal_equations_stream

-- Stage-2 neural rebase, wave-2 headlines (FrontierLadder and SpinorFold; NullTransformer names
-- pending )
#audit Sanguine.FrontierLadder.reaches_conditioning_floor
#audit Sanguine.FrontierLadder.seam1_residual_bounded
#audit Sanguine.FrontierLadder.parity_underdetermines_target
#audit Sanguine.FrontierLadder.the_frontier_distance
#audit Sanguine.FrontierLadder.fold_preserves_correctness
#audit Sanguine.FrontierLadder.estimation_gap_zero_at_truth
#audit Sanguine.FrontierLadder.book_anchors_window
#audit Sanguine.Spinor.nmul_assoc
#audit Sanguine.Spinor.nmul_is_diagonal
#audit Sanguine.Spinor.spinorFold_eq
#audit Sanguine.Spinor.spinorFold_schedule_free
#audit Sanguine.Spinor.rotor_folds_into_weight
#audit Sanguine.Spinor.norm_is_product_of_halves
#audit Sanguine.Spinor.monad_is_the_third
#audit Sanguine.Spinor.nulls_are_the_two_single_halves

-- AI-track re-anchor, Nous surface and model headlines (waves 1-4; the architecture track pauses
-- after assembly by user direction — existing-model parity has priority)
#audit Nous.Core.opSpec_compose_vjp
#audit Nous.Core.OpSpec.compose_cotangentLinear
#audit Nous.Core.accumList_getLastD
#audit Nous.Core.mapAccum_back?_cons
#audit Nous.Core.mapAccumRight_getElem?_zero
#audit Nous.Model.ringUpper_tokenStep
#audit Nous.Model.ringLower_tokenStep
#audit Nous.Model.channelWeight_eq
#audit Nous.Model.channelWeight_add_lowerChannelWeight
#audit Nous.Model.token_step_fold
#audit Nous.Model.tokenFold_append
#audit Nous.Model.memory_step_model
#audit Nous.Model.memory_step_get_self
#audit Nous.Model.channelPair_mul
#audit Nous.Model.channelPair_normForm
#audit Nous.Model.hyperbolic_decomposition
#audit Nous.Model.hyperbolic_decomposition_field
#audit Nous.Model.channelPair_tokenFold
#audit Nous.Model.channelPair_memoryStep
#audit Nous.Model.channel_diagonal_fold
#audit Nous.Model.run_state_eq_tokenFold
#audit Nous.Model.run_memory_eq_memoryFold
#audit Nous.Model.run_eq
#audit Nous.Model.readout_add_lowerReadout

-- Locate-loop slice: the `spill_locate` inner completion invariant and the
-- scanner-coincidence transfer; `SpillLocateFoundFullMatch` closes after the outer sweep (L4-L6)
#audit Apeiron.GrowableLocateLoop.least_set_lane_loop_selects
#audit Apeiron.GrowableLocateLoop.least_set_lane_loop_in_range
#audit Apeiron.GrowableLocateLoop.scan_lane_of_loop_lane
#audit Apeiron.GrowableLocateLoop.generated_spill_locate_clears_selected
#audit Apeiron.GrowableLocateLoop.generated_locate_inner_outcome

-- Locate-loop slice 2: the outer sweep completion invariant
#audit Apeiron.GrowableLocateLoop.LocateInnerInv
#audit Apeiron.GrowableLocateLoop.LocateInnerOutcome
#audit Apeiron.GrowableLocateLoop.LocateMatchWitness
#audit Apeiron.GrowableLocateLoop.LocateVacantWitness
#audit Apeiron.GrowableLocateLoop.LocateOuterStepMiss
#audit Apeiron.GrowableLocateLoop.LocateOuterStepReady
#audit Apeiron.GrowableLocateLoop.LocateOuterInv
#audit Apeiron.GrowableLocateLoop.LocateOuterOutcome
#audit Apeiron.GrowableLocateLoop.generated_locate_outer_outcome

-- Locate-loop slice 3: Found unfold and geometry discharge
#audit Apeiron.GrowableLocateLoop.generated_spill_locate_unfold
#audit Apeiron.GrowableLocateLoop.locate_outer_step_ready_of_geometry

-- Locate-loop slice 4: inner-body inversion and slot certificate
#audit Apeiron.GrowableLocateLoop.LocateSlotCertified
#audit Apeiron.GrowableLocateLoop.locate_inner_body_shape

-- Locate-loop slice 5: loop-equation bridge and inner found extraction
#audit Apeiron.GrowableLocateLoop.wrapper_loop_form
#audit Apeiron.GrowableLocateLoop.generated_locate_inner_found

-- Locate-loop slice 6: outer found extraction and the final classifier theorem
#audit Apeiron.GrowableLocateLoop.outer_wrapper_loop_form
#audit Apeiron.GrowableLocateLoop.locate_outer_body_shape
#audit Apeiron.GrowableLocateLoop.generated_locate_outer_found
#audit Apeiron.GrowableLocateLoop.generated_spill_locate_unfold_run
#audit Apeiron.GrowableClassifier.spill_locate_found_full_match

-- : fixed-point exp refinement surface
#audit apeiron_word.q32
#audit apeiron_word.toReal
#audit apeiron_word.toReal128
#audit apeiron_word.window
#audit apeiron_word.epsExp
#audit apeiron_word.fixed_exp_refuses
#audit apeiron_word.fixed_exp_refines
#audit apeiron_word.word_fixed_exp_refuses_of_lt
#audit apeiron_word.word_fixed_exp_refuses_of_gt

-- : evaluated endpoints and the refusal instance
#audit apeiron_word.fixed_exp_ONE_eq
#audit apeiron_word.fixed_exp_LOWER_eq
#audit apeiron_word.fixed_exp_UPPER_eq
#audit apeiron_word.fixed_exp_refuses_proved

-- : exp post-guard unfold and reduction budget
#audit apeiron_word.fixed_exp_exp_reduce_spec
#audit apeiron_word.fixed_exp_exp_scale_spec
#audit apeiron_word.fixed_exp_lower_test_false
#audit apeiron_word.fixed_exp_upper_test_false
#audit apeiron_word.fixed_exp_round_div_2p32_HALF_eq
#audit apeiron_word.fixed_exp_round_div_2p32_spec
#audit apeiron_word.toReal_ln2
#audit apeiron_word.toReal_ln2_recip
#audit apeiron_word.ln2_recip_defect
#audit apeiron_word.int_shiftRight_nonneg

-- : rsqrt refusal surface and bound decomposition
#audit apeiron_word.rsqrt_window
#audit apeiron_word.fixed_rsqrt_refuses
#audit apeiron_word.word_fixed_rsqrt_refuses_of_lt
#audit apeiron_word.word_fixed_rsqrt_refuses_of_gt
#audit apeiron_word.fixed_rsqrt_ONE_eq
#audit apeiron_word.fixed_rsqrt_LOWER_eq
#audit apeiron_word.fixed_rsqrt_UPPER_eq
#audit apeiron_word.fixed_rsqrt_refuses_proved
#audit apeiron_word.word_fixed_rsqrt_unfold
#audit apeiron_word.isqrt_real_bounds
#audit apeiron_word.isqrt_recip_error_bound
#audit apeiron_word.isqrt_scaled_error
#audit apeiron_word.round_nearest_div_error
#audit apeiron_word.sqrt_shift
#audit apeiron_word.fixed_rsqrt_refines

-- : extracted restoring-square-root invariant
#audit apeiron_word.IsPow4
#audit apeiron_word.IsqrtInv
#audit apeiron_word.isqrt_inv_init
#audit apeiron_word.isqrt_inv_final
#audit apeiron_word.isqrt_step_hit
#audit apeiron_word.isqrt_step_miss
#audit apeiron_word.isqrt_final_hit
#audit apeiron_word.isqrt_final_miss

-- : reduction range lemmas (seven side conditions)
#audit apeiron_word.fixed_exp_shift_recip_shift_bound
#audit apeiron_word.fixed_exp_shift_recip_bound_window
#audit apeiron_word.fixed_exp_bmod_64_lower
#audit apeiron_word.fixed_exp_bmod_64_upper
#audit apeiron_word.fixed_exp_bmod_self_64
#audit apeiron_word.fixed_exp_reduction_remainder_range

-- / C: reduction safety wiring and the loop1 lift
#audit apeiron_word.word_fixed_exp_reduction_arith_safe
#audit apeiron_word.isqrt_loop1_spec
#audit apeiron_word.IsPow4.div_four
#audit apeiron_word.pow4_dvd_four
#audit apeiron_word.isqrt_final_hit_inv
#audit apeiron_word.isqrt_final_miss_inv

-- : reduction value characterization
#audit apeiron_word.word_fixed_exp_reduction_value

-- : isqrt seed spec and the extracted isqrt spec
#audit apeiron_word.isqrt_loop0_spec
#audit apeiron_word.isqrt_spec

-- : round-value spec and the first Horner step
#audit apeiron_word.fixed_exp.round_q32
#audit apeiron_word.fixed_exp.horner_acc
#audit apeiron_word.fixed_exp_round_div_2p32_val
#audit apeiron_word.word_fixed_exp_horner_first

-- : rsqrt real-analysis helpers
#audit apeiron_word.toReal_pos
#audit apeiron_word.sqrt_toReal
#audit apeiron_word.one_div_sqrt_toReal

-- : rounding bounds for the Horner chain
#audit apeiron_word.fixed_exp.round_q32_abs_le
#audit apeiron_word.fixed_exp.round_safe

-- : parameterised rounding bound for the chain stages
#audit apeiron_word.fixed_exp.round_q32_abs_le'

-- : rsqrt success-block structural evaluation
#audit apeiron_word.word_fixed_rsqrt_success

-- : factored Horner stage lemma
#audit apeiron_word.fixed_exp.horner_step

-- : rsqrt refinement closed
#audit apeiron_word.isqrt_scaled_real
#audit apeiron_word.fixed_rsqrt_refines_proved

-- : trig refusal surface, statement and reduction facts
#audit apeiron_word.trig_window
#audit apeiron_word.fixed_sin_cos_refuses
#audit apeiron_word.trig_floor
#audit apeiron_word.fixed_sin_cos_refines
#audit apeiron_word.fixed_sin_cos_refuses_proved
#audit apeiron_word.word_fixed_sin_cos_window_past_guards
#audit apeiron_word.toReal_two_pi
#audit apeiron_word.toReal_two_pi_recip
#audit apeiron_word.toReal_pi
#audit apeiron_word.toReal_half_pi
#audit apeiron_word.toReal_quarter_pi
#audit apeiron_word.two_pi_recip_defect
#audit apeiron_word.fixed_trig_shift_bound
#audit apeiron_word.fixed_trig_shift_bound_window
#audit apeiron_word.fixed_trig_reduction_remainder_range
#audit apeiron_word.fixed_trig_ONE_eq
#audit apeiron_word.fixed_trig_LOWER_eq
#audit apeiron_word.fixed_trig_UPPER_eq

-- : two-stage Horner prefix
#audit apeiron_word.word_fixed_exp_horner_acc1

-- : trig reduction value characterization
#audit apeiron_word.word_fixed_sin_cos_reduction_value

-- : quadrant fold values and composition
#audit apeiron_word.fixed_trig.fold1
#audit apeiron_word.fixed_trig.fold1_negate
#audit apeiron_word.fixed_trig.fold2
#audit apeiron_word.fixed_trig.fold2_swap
#audit apeiron_word.fixed_trig.fold1_abs_le
#audit apeiron_word.fixed_trig_fold1_spec
#audit apeiron_word.fixed_trig_fold2_spec
#audit apeiron_word.word_fixed_sin_cos_fold_value

-- : reduction-fold bridge and the Taylor head
#audit apeiron_word.fixed_trig.fold_step
#audit apeiron_word.fixed_trig_fold_step_spec
#audit apeiron_word.word_fixed_sin_cos_reduction_fold_value
#audit apeiron_word.fixed_trig_int_shiftRight_nonneg
#audit apeiron_word.fixed_trig_round_div_2p32_HALF_eq
#audit apeiron_word.fixed_trig_round_div_2p32_spec
#audit apeiron_word.fixed_trig.round_q32
#audit apeiron_word.fixed_trig_round_div_2p32_val
#audit apeiron_word.fixed_trig.sin_poly
#audit apeiron_word.fixed_trig.cos_poly
#audit apeiron_word.fixed_trig_taylor_value
#audit apeiron_word.word_fixed_taylor_first

-- : Taylor stage bounds and the corrected domain
#audit apeiron_word.fixed_trig.round_q32_abs_le
#audit apeiron_word.fixed_trig.round_q32_abs_le'
#audit apeiron_word.fixed_trig.round_safe

-- : full Horner chain (bounded cap)
#audit apeiron_word.word_fixed_exp_horner_acc

-- : scaling shift-count bound
#audit apeiron_word.fixed_exp_scale_shift_count_lt

-- : k >= 0 scaling branch
#audit apeiron_word.fixed_exp_scale_nonneg

-- : Taylor totality and the bound helpers
#audit apeiron_word.fixed_trig.abs_mul_le
#audit apeiron_word.fixed_trig.mul_range
#audit apeiron_word.fixed_trig.abs_mul_le_z
#audit apeiron_word.fixed_trig.abs_add_const_le
#audit apeiron_word.fixed_trig.abs_sub_const_le
#audit apeiron_word.fixed_trig.round_safe66
#audit apeiron_word.word_taylor_spec

-- : Taylor value closure
#audit apeiron_word.fixed_trig_taylor_value_proved

-- : k < 0 rounding core
#audit apeiron_word.fixed_exp_scale_neg_core

-- : fold argument bound
#audit apeiron_word.fixed_trig.fold_arg_abs_le

-- : flush and full k < 0 branch
#audit apeiron_word.fixed_exp_scale_flush
#audit apeiron_word.fixed_exp_scale_neg

-- : tight reduction remainder bound
#audit apeiron_word.fixed_trig_reduction_remainder_range_tight

-- : scaling helpers
#audit apeiron_word.fixed_exp_scale_abs
#audit apeiron_word.fixed_exp_scale_nonneg_ge64

-- : full scaling block
#audit apeiron_word.fixed_exp_scale_abs_raw
#audit apeiron_word.word_fixed_exp_scale

-- : end-to-end trig value
#audit apeiron_word.word_fixed_sin_cos_value

-- : pair-returning reduction chunk
#audit apeiron_word.word_fixed_exp_reduction_value_pair

-- : trig displacement and transfers
#audit apeiron_word.trig_delta
#audit apeiron_word.two_pi_q32_defect
#audit apeiron_word.trig_reduction_displacement
#audit apeiron_word.trig_sin_reduction_transfer
#audit apeiron_word.trig_cos_reduction_transfer

-- : fold constant defects
#audit apeiron_word.pi_q32_defect
#audit apeiron_word.half_pi_q32_defect
#audit apeiron_word.quarter_pi_q32_defect
#audit apeiron_word.trig_fold_delta

-- : success-path pair chunk
#audit apeiron_word.word_fixed_exp_reduction_pair_succ

-- : fold thresholds and displacements
#audit apeiron_word.half_pi_q32_floor
#audit apeiron_word.quarter_pi_q32_floor
#audit apeiron_word.fold1T
#audit apeiron_word.fold2T
#audit apeiron_word.fold1_displacement
#audit apeiron_word.fold2_displacement

-- : fold phase identities and flag selection
#audit apeiron_word.sin_fold1T
#audit apeiron_word.cos_fold1T
#audit apeiron_word.sin_fold2T
#audit apeiron_word.cos_fold2T
#audit apeiron_word.sin_phase_select
#audit apeiron_word.cos_phase_select

-- : flagged output bounds
#audit apeiron_word.flagSin
#audit apeiron_word.flagCos
#audit apeiron_word.flag_sin_fold_bound
#audit apeiron_word.flag_cos_fold_bound
#audit apeiron_word.flag_sin_reduction_bound
#audit apeiron_word.flag_cos_reduction_bound

-- : quarter-angle bound
#audit apeiron_word.fixed_trig.fold_arg_abs_le_quarter

-- : rounding ledger
#audit apeiron_word.fixed_trig.round_q32_half
#audit apeiron_word.fixed_trig.round_q32_ledger

-- : Horner chain ledgers
#audit apeiron_word.fixed_trig.sinHorner
#audit apeiron_word.fixed_trig.cosHorner
#audit apeiron_word.fixed_trig.sin_poly_horner
#audit apeiron_word.fixed_trig.cos_poly_horner

-- : Taylor defects and remainders
#audit apeiron_word.fixed_trig.sinTaylor
#audit apeiron_word.fixed_trig.cosTaylor
#audit apeiron_word.fixed_trig.sinHorner_taylor
#audit apeiron_word.fixed_trig.cosHorner_taylor
#audit apeiron_word.fixed_trig.sinTaylor_remainder
#audit apeiron_word.fixed_trig.cosTaylor_remainder

-- : whole-body exp value
#audit apeiron_word.fixed_exp_reduction_remainder_abs_le
#audit apeiron_word.word_fixed_exp_body

-- : exp displacement and transfer
#audit apeiron_word.deltaRed
#audit apeiron_word.fixed_exp_ln2_q32_defect
#audit apeiron_word.fixed_exp_reduction_displacement
#audit apeiron_word.fixed_exp_exponential_transfer

-- : Horner decomposition pieces (post-reboot reconciliation)
#audit apeiron_word.epsHorner
#audit apeiron_word.fixed_exp_taylor7_remainder
#audit apeiron_word.fixed_exp_coeff_defect_7
#audit apeiron_word.fixed_exp_coeff_defect_6
#audit apeiron_word.fixed_exp_coeff_defect_5
#audit apeiron_word.fixed_exp_coeff_defect_4
#audit apeiron_word.fixed_exp_coeff_defect_3
#audit apeiron_word.fixed_exp_coeff_defect_2
#audit apeiron_word.fixed_exp_coeff_defect_1
#audit apeiron_word.fixed_exp_coeff_defect_0

-- : Q32.32 rounding ledger
#audit apeiron_word.fixed_exp_round_q32_int_error
#audit apeiron_word.fixed_exp_round_q32_real_error

-- : Horner round propagation
#audit apeiron_word.fixed_exp.hand
#audit apeiron_word.fixed_exp.handReal
#audit apeiron_word.fixed_exp.handRealF
#audit apeiron_word.fixed_exp.horner_acc_eq
#audit apeiron_word.fixed_exp.hornerReal
#audit apeiron_word.fixed_exp.hand_bound
#audit apeiron_word.fixed_exp.horner_round_error

-- : coefficient machinery
#audit apeiron_word.fixed_exp.hornerRealF
#audit apeiron_word.fixed_exp.taylorReal
#audit apeiron_word.fixed_exp.hornerReal_eq_F
#audit apeiron_word.fixed_exp.taylorReal_eq_sum
#audit apeiron_word.fixed_exp.handRealF_defect

-- : list Horner and combined error
#audit apeiron_word.fixed_exp.realHorner
#audit apeiron_word.fixed_exp.Cs
#audit apeiron_word.fixed_exp.Ds
#audit apeiron_word.fixed_exp.Ps
#audit apeiron_word.fixed_exp.hornerReal_eq_list
#audit apeiron_word.fixed_exp.taylorReal_eq_list
#audit apeiron_word.fixed_exp.realHorner_step
#audit apeiron_word.fixed_exp.realHorner_pair_le
#audit apeiron_word.fixed_exp.hornerReal_taylor_error
#audit apeiron_word.fixed_exp.horner_acc_error

-- : sharpened exp remainder
#audit apeiron_word.fixed_exp_reduction_remainder_abs_le_2p31
#audit apeiron_word.fixed_exp_reduction_remainder_toReal_half
#audit apeiron_word.word_fixed_sin_cos_past_guards_value
#audit apeiron_word.fixed_sin_cos_refines_proved
#audit apeiron_word.sin_out_flagSin_le
#audit apeiron_word.cos_out_flagCos_le
#audit apeiron_word.trig_floor_close
#audit apeiron_word.sin_poly_close_le
#audit apeiron_word.cos_poly_close_le
#audit apeiron_word.cos_poly_rem_le
#audit apeiron_word.abs_smul_cancel
#audit apeiron_word.c10_let_push
#audit apeiron_word.c10_let_prod_push
#audit apeiron_word.c10_pair_bind

-- : exp_scale body exact values (Gap A)
#audit apeiron_word.fixed_exp_bmod128_low
#audit apeiron_word.fixed_exp_scale_nonneg_exact
#audit apeiron_word.fixed_exp_two_pow_bmod
#audit apeiron_word.fixed_exp_scale_neg_core_exact
#audit apeiron_word.fixed_exp_scale_neg_exact

-- : on-window scale range budget
#audit apeiron_word.fixed_exp_exp_half_lt
#audit apeiron_word.fixed_exp_scale_range_budget

-- : exp headline
#audit apeiron_word.fixed_exp_exp_reduce_spec_ok
#audit apeiron_word.fixed_exp_exp_u_bounds
#audit apeiron_word.fixed_exp_scale_abs_rel_pos
#audit apeiron_word.fixed_exp_scale_abs_rel_neg
#audit apeiron_word.fixed_exp_ediv_self_abs
#audit apeiron_word.fixed_exp_round_neg_abs
#audit apeiron_word.word_fixed_exp_body_refines
#audit apeiron_word.fixed_exp_refines_proved

-- : K-quant code unpack — refusal correspondence
#audit apeiron_word.q4CodesRef
#audit apeiron_word.q6CodesRef
#audit apeiron_word.quant_Q4_K_BLOCK_BYTES_eq
#audit apeiron_word.quant_Q6_K_BLOCK_BYTES_eq
#audit apeiron_word.word_q4_k_codes_refuses
#audit apeiron_word.word_q6_k_codes_refuses

-- : Q4_K loop invariants
#audit apeiron_word.q4QsRef
#audit apeiron_word.q4Val
#audit apeiron_word.q4Inv
#audit apeiron_word.getD_set
#audit apeiron_word.q4div
#audit apeiron_word.q4mod
#audit apeiron_word.q4mod32
#audit apeiron_word.q4QsRef_low
#audit apeiron_word.q4QsRef_high
#audit apeiron_word.q4Val_step
#audit apeiron_word.q4Val_low
#audit apeiron_word.q4Val_high

-- : Q4_K loop specs and bridge
#audit apeiron_word.q4_getD_getElem
#audit apeiron_word.nat_and_15
#audit apeiron_word.nat_shr_4
#audit apeiron_word.q4_inner_spec
#audit apeiron_word.q4Val_succ
#audit apeiron_word.q4Inv_succ
#audit apeiron_word.q4_outer_spec
#audit apeiron_word.q4QsRef_eq_q4CodesRef

-- : fragment semantics and layout verdict
#audit Spirv.byte
#audit Spirv.byte_order
#audit Spirv.evalSUDot
#audit Spirv.sdot_denote
#audit Spirv.eval
#audit Spirv.step
#audit Spirv.denote
#audit Spirv.eval_sudot
#audit Spirv.realizedSigma
#audit Spirv.realizedRoot
#audit Spirv.realizedCoRoot
#audit Spirv.ringMul_diag
#audit Spirv.evalIMul_face
#audit Spirv.FaceWitness
#audit Spirv.root_witness
#audit Spirv.wrong_sigma_root
#audit Spirv.ringMul_channel0
#audit Spirv.channel0_is_ringForm
#audit Spirv.ringTrace
#audit Spirv.ringTrace_two_channel0
#audit Spirv.two_not_unit
#audit Spirv.actCell
#audit Spirv.wgtCell
#audit Spirv.sdot_split_ringMul
#audit Spirv.sdot_split_realized
#audit Spirv.cross_channel_not_realized
#audit Spirv.servedOpcodes
#audit Spirv.floatEpilogueOpcodes
#audit Spirv.executingOpcodes
#audit Spirv.IsDp4aFragment
#audit Spirv.IsExecuting
#audit Spirv.executingOpcodes_eq

-- : Q4_K positive equality
#audit apeiron_word.word_q4_k_codes_ok
#audit apeiron_word.q4Val_zero
#audit apeiron_word.q4Val_four
#audit apeiron_word.q4Inv_repeat

-- : Q6_K references and bridge
#audit apeiron_word.q6QsRef
#audit apeiron_word.q6Val
#audit apeiron_word.q6Inv
#audit apeiron_word.getD_take
#audit apeiron_word.getD_drop_take
#audit apeiron_word.q6QsRef_eq_q6CodesRef

-- : Q6_K write layer
#audit apeiron_word.hcast_u8_I32_val
#audit apeiron_word.hcast_sub32_lower
#audit apeiron_word.q6Val_zero
#audit apeiron_word.q6Val_four
#audit apeiron_word.q6Val_step
#audit apeiron_word.q6Val_1
#audit apeiron_word.q6Val_2
#audit apeiron_word.q6Val_3
#audit apeiron_word.q6Val_4
#audit apeiron_word.q6Val_succ
#audit apeiron_word.q6Inv_succ
#audit apeiron_word.q6Inv_repeat

-- : Q6_K bit identities
#audit apeiron_word.q6_combine_lt
#audit apeiron_word.bmod_sub32

-- : Q6_K reference evaluations
#audit apeiron_word.shift4_mod
#audit apeiron_word.q6QsRef_idx0
#audit apeiron_word.q6QsRef_idx1
#audit apeiron_word.q6QsRef_idx2
#audit apeiron_word.q6QsRef_idx3
-- : decoder soundness
#audit Spirv.parseInsts
#audit Spirv.decode
#audit Spirv.wordsOfBytes
#audit Spirv.bytes
#audit Spirv.WFInst
#audit Spirv.DefUsesB
#audit Spirv.UniqueB
#audit Spirv.TypeOKB
#audit Spirv.StorageOKB
#audit Spirv.WFCode
#audit Spirv.WFState
#audit Spirv.WellFormed
#audit Spirv.D1_wordCount_zero
#audit Spirv.D1_overrun
#audit Spirv.D2_defUses
#audit Spirv.D2_unique
#audit Spirv.D3_typeOK
#audit Spirv.D4_storageOK
#audit Spirv.decode_sound
#audit Spirv.words_of_bytes_bytes
#audit Spirv.validTestModule
#audit Spirv.duplicateTestModule

-- : Q6_K factored identities
#audit apeiron_word.q6_combine_lt'
#audit apeiron_word.q6_high_lt
#audit apeiron_word.q6_cast_value
#audit apeiron_word.q6_write_val_gen

-- : Q6_K identities and pointwise split
#audit apeiron_word.q6_val_1_ident
#audit apeiron_word.q6_val_2_ident
#audit apeiron_word.q6_val_3_ident
#audit apeiron_word.q6_val_4_ident
#audit apeiron_word.q6_pointwise

-- : Q6_K body values chunk
#audit apeiron_word.q6_body_vals

-- : Q6_K inner body
#audit apeiron_word.q6_idx_lt_64
#audit apeiron_word.q6_idx_lt_32
#audit apeiron_word.q6_idx_lt_128
#audit apeiron_word.q6_measure
#audit apeiron_word.q6_succ_le
#audit apeiron_word.q6_succ_le'
#audit apeiron_word.q6_inner_body

-- : Q6_K loop specs
#audit apeiron_word.q6_measure2
#audit apeiron_word.q6_inner_spec
#audit apeiron_word.q6_outer_spec

-- : Q6_K positive equality
#audit apeiron_word.word_q6_k_codes_ok

-- : GGUF well-formedness predicates
#audit apeiron_word.GGUF_MAGIC_BYTES
#audit apeiron_word.magic_ok
#audit apeiron_word.version_ok
#audit apeiron_word.block_geom
#audit apeiron_word.block_count
#audit apeiron_word.dims_prod
#audit apeiron_word.required_bytes
#audit apeiron_word.TensorRangeWF
#audit apeiron_word.HeaderWF
#audit apeiron_word.block_geom_zero
#audit apeiron_word.block_geom_q4k
#audit apeiron_word.block_geom_bad
#audit apeiron_word.dims_prod_nil
#audit apeiron_word.required_bytes_none_iff
#audit apeiron_word.version_2_ok
#audit apeiron_word.version_3_ok

-- : GGUF cursor primitives
#audit apeiron_word.gguf_cur_u8_some
#audit apeiron_word.gguf_cur_u8_none

-- : register the online-spinor attention identity
#audit Sanguine.AttentionSpinor.softmax_attention_is_online_spinor

-- : composed error and margin
#audit Nous.Model.Layer
#audit Nous.Model.Layer.ofRelative
#audit Nous.Model.Layer.affineLayer
#audit Nous.Model.Layer.budget
#audit Nous.Model.Layer.norm_execComp_sub_refComp_le
#audit Nous.Model.Layer.budget_replicate_le
#audit Nous.Model.StrictWinner
#audit Nous.Model.strictWinner_perturb
#audit Nous.Model.strictWinner_perturb_argmax
#audit Nous.Model.score_perturb_bound
#audit Nous.Model.decision_margin_preserved
#audit Nous.Model.numericScale
#audit Nous.Model.numericScale_lt_one_hundred_thousandth
#audit Nous.Model.numeric_fold_margin_usable
#audit Nous.Model.numeric_decision_unchanged
#audit Nous.Model.affine_fold_four
#audit Nous.Model.ratio_eq_exp_log
#audit Nous.Model.ratio_eq_softmax2_log
#audit Nous.Model.channelWeight_binary_softmax
#audit Nous.Model.attention_channel_form
#audit Nous.Model.channel_form_domain_witness
#audit Nous.Model.channelAttnOut_normal
#audit Nous.Model.channelWeight_factor
#audit Nous.Model.hyperbolic_ringMul
#audit Nous.Model.hyperbolic_ringNormForm
#audit Nous.Model.hyperbolic_ringMul_assoc
#audit Nous.Model.hyperbolic_ringMul_comm
#audit Nous.Model.hyperbolic_ringOne_mul
#audit Nous.Model.hyperbolic_ringNormForm_mul
#audit Nous.Model.parabolic_ringMul
#audit Nous.Model.parabolic_ringNormForm
#audit Nous.Model.parabolic_ringMul_assoc
#audit Nous.Model.parabolic_ringMul_comm
#audit Nous.Model.parabolic_ringOne_mul
#audit Nous.Model.parabolic_ringNormForm_mul
#audit Nous.Model.elliptic_ringMul
#audit Nous.Model.elliptic_ringNormForm
#audit Nous.Model.elliptic_ringMul_assoc
#audit Nous.Model.elliptic_ringMul_comm
#audit Nous.Model.elliptic_ringOne_mul
#audit Nous.Model.elliptic_ringNormForm_mul
#audit Nous.Model.word_denoteSigma_hyperbolic
#audit Nous.Model.word_denoteSigma_parabolic
#audit Nous.Model.word_denote_neg_one
#audit Nous.Model.word_denoteSigma_elliptic
#audit Nous.Model.word_mul_denote_hyperbolic
#audit Nous.Model.word_mul_denote_parabolic
#audit Nous.Model.word_mul_denote_elliptic
#audit Nous.Model.word_norm_denote_hyperbolic
#audit Nous.Model.word_norm_denote_parabolic
#audit Nous.Model.word_norm_denote_elliptic
#audit Nous.Model.zmod_two_not_unit
#audit apeiron_word.engine_ONE_eq
#audit apeiron_word.silu_UPPER_eq
#audit apeiron_word.silu_LOWER_eq
#audit apeiron_word.word_silu_fixed_high
#audit apeiron_word.word_silu_fixed_low
#audit apeiron_word.div_round_nearest_abs_bound
#audit apeiron_word.silu_bound_general
#audit apeiron_word.word_silu_fixed_unfold
#audit apeiron_word.exp_result_val_bounds
#audit apeiron_word.silu_tail_refines
#audit apeiron_word.word_silu_fixed_spec
#audit apeiron_word.silu_window_refines
#audit apeiron_word.round_shift_32_spec
#audit apeiron_word.rms_norm_fixed_loop0_eq
#audit apeiron_word.rms_norm_fixed_loop1_eq
#audit apeiron_word.div_round_nearest_spec
#audit apeiron_word.rmsElement_bound
#audit apeiron_word.rms_norm_refines_proved
#audit apeiron_word.word_rms_norm_fixed_out_refuses
#audit Nous.Model.siluCell_normForm
#audit Nous.Model.siluGate_eq_softmax2
#audit Nous.Model.channelWeight_siluCell
#audit Nous.Model.silu_gate_eq_channelWeight
#audit Nous.Model.siluReal_eq_mul_channelWeight
#audit apeiron_word.gguf_cur_u8_eq_some
#audit apeiron_word.gguf_cur_u8_eq_none
#audit apeiron_word.gguf_cur_u16_eq_some
#audit apeiron_word.gguf_cur_u16_short
#audit apeiron_word.gguf_cur_u32_eq_some
#audit apeiron_word.gguf_cur_u32_short
#audit apeiron_word.gguf_cur_u64_eq_some
#audit apeiron_word.gguf_cur_u64_short
#audit apeiron_word.gguf_cur_four_eq_some
#audit apeiron_word.gguf_cur_four_short
#audit apeiron_word.gguf_counts_guard_tail
#audit apeiron_word.option_branch_some
#audit apeiron_word.option_branch_none
#audit apeiron_word.core_result_branch_ok
#audit apeiron_word.core_result_branch_err
#audit apeiron_word.core_result_from_residual_err
#audit apeiron_word.word_gguf_parse_header_truncated_magic
#audit apeiron_word.word_gguf_parse_header_bad_magic
#audit apeiron_word.word_gguf_parse_header_truncated_version
#audit apeiron_word.word_gguf_parse_header_unsupported_version
#audit apeiron_word.word_gguf_parse_header_truncated_tensor_count
#audit apeiron_word.word_gguf_parse_header_truncated_kv_count
#audit apeiron_word.word_gguf_parse_header_truncated_kv_tensor_count
#audit Nous.Model.rhoMul_assoc
#audit Nous.Model.rho_add
#audit Nous.Model.rho_nat
#audit Nous.Model.rho_sub
#audit Nous.Model.rho_relative
#audit Nous.Model.rhoAct_add
#audit Nous.Model.rhoDot_act
#audit Nous.Model.rho_relative_score
#audit Nous.Model.rotMat_mul
#audit Nous.Model.rotMat_transpose
#audit Nous.Model.rotMat_relative
#audit Nous.Model.rope_orbit
#audit Nous.Model.channelWeight_parabolic
#audit Nous.Model.channelScore_parabolic
#audit Nous.Model.channelDen_parabolic
#audit Nous.Model.channelDen_parabolic_ne_zero_iff
#audit Nous.Model.attention_channel_form_parabolic
#audit Nous.Model.elliptic_no_root
#audit Nous.Model.elliptic_no_root_witness
#audit Nous.Model.channelFormDomain_elliptic_empty
#audit Nous.Model.siluReal_lipschitz
#audit Nous.Model.rmsRoot_refines
#audit Nous.Model.rmsEps_nonneg
#audit Nous.Model.rmsFloor_nonneg
#audit Nous.Model.rmsGrid_bound
#audit Nous.Model.rmsRef_lipschitz
#audit Nous.Model.siluLayer_delta
#audit Nous.Model.rmsLayer_delta
#audit Nous.Model.silu_decision_margin_unchanged
#audit Nous.Model.rms_decision_margin_unchanged
#audit apeiron_word.row_dot_refines_proved
#audit apeiron_word.dot_rounding_bound
#audit apeiron_word.row_dot_fixed_loop_eq
#audit apeiron_word.word_row_dot_fixed_row_length
#audit apeiron_word.word_row_dot_fixed_acc_refuses
#audit apeiron_word.word_row_dot_fixed_out_high_refuses
#audit apeiron_word.word_row_dot_fixed_out_low_refuses
#audit apeiron_word.word_row_dot_fixed_unfold
#audit apeiron_word.rowDotReal_eq
-- G3 matvec (dominant kernel)
#audit apeiron_word.matvec_refines_proved
#audit apeiron_word.matvec_fixed_loop_spec
#audit apeiron_word.matvec_fixed_loop_body_step
#audit apeiron_word.word_matvec_fixed_unfold
#audit apeiron_word.word_matvec_fixed_row_length_refuses
#audit apeiron_word.word_matvec_fixed_dequant_refuses
#audit apeiron_word.word_matvec_fixed_acc_refuses
#audit apeiron_word.word_matvec_fixed_out_refuses
-- G3 attention head (stage bounds + refusals)
#audit apeiron_word.word_attn_head_unfold
#audit apeiron_word.word_attn_head_ok
#audit apeiron_word.word_attn_head_zero_sum
#audit apeiron_word.word_attn_head_exp_refuses
#audit apeiron_word.word_attn_head_out_refuses
#audit apeiron_word.score_epilogue_bound
#audit apeiron_word.weight_at_spec
#audit apeiron_word.weight_rounding_bound
#audit apeiron_word.value_component_rounding_bound
#audit apeiron_word.value_sum_toReal
#audit apeiron_word.exp_at_real_bound
#audit apeiron_word.exp_at_underflow_real
#audit apeiron_word.softmaxWeights_shift_invariant
-- canonical packing decoder (affine fold + refusals + per-element value law)
#audit apeiron_word.packing_decode_block_eq_epilogue
#audit apeiron_word.word_packing_decode_block_truncated
#audit apeiron_word.word_packing_decode_block_block_error
#audit apeiron_word.packing_mul_ok
#audit apeiron_word.packing_sub_ok
#audit apeiron_word.packing_mul2_affine
#audit apeiron_word.epilogue_entry_fixed_nomin
-- G3 sampler / KV state
#audit apeiron_word.argmax_loop_spec
#audit apeiron_word.word_argmax_fixed_spec
#audit apeiron_word.word_argmax_fixed_empty
#audit apeiron_word.argmaxAt_real
#audit apeiron_word.word_kv_state_new_geometry
#audit apeiron_word.word_kv_state_new_not_multiple
#audit apeiron_word.word_kv_append_ok
#audit apeiron_word.word_kv_append_capacity_refusal
#audit apeiron_word.word_kv_append_row_length_k
#audit apeiron_word.word_kv_append_row_length_v
#audit apeiron_word.word_kv_append_layer_refusal
-- G3 LM head (conditional on matvec)
#audit apeiron_word.word_lm_head_fixed_shape_mismatch
#audit apeiron_word.word_lm_head_fixed_unsupported_quant
#audit apeiron_word.word_lm_head_fixed_unfold
#audit apeiron_word.word_lm_head_fixed_of_matvec_ok
#audit apeiron_word.word_lm_head_fixed_of_matvec_err
#audit apeiron_word.lm_head_refines_of_matvec
-- GPU/SPIR-V DP4A refinement + in-bounds
#audit Spirv.dp4a_gemv_spec
#audit Spirv.scale_fixed_point
#audit Spirv.chunk_dot_refines
#audit Spirv.seg_dot_refines
#audit Spirv.gpu_dp4a_refines_shared
#audit Spirv.cpu_dp4a_refines_shared
#audit Spirv.gpu_dp4a_word_refines_cpu_word
#audit Spirv.splitK_fold_reassoc
#audit Spirv.splitK_row_reassoc
#audit Spirv.splitK_normForm_fold
#audit Spirv.accessChain_in_bounds
#audit Spirv.Refines.trans
-- GPU/SPIR-V checker soundness (header leg)
#audit Spirv.check_fragment_accepts_header
#audit Spirv.checkFragment_header
#audit Spirv.parseHeader_shape
#audit Spirv.magic_eq
#audit Spirv.version_eq
#audit Spirv.headerWords_val

-- Model-algebra closing wave: attention end to end, dequant faces, block stages, SPIR-V checker scans
-- Numeric.AttnSoftmaxPerturb
#audit apeiron_word.not_attn_head_refines
#audit apeiron_word.reweight_mean_perturb
#audit apeiron_word.score_shift_perturb
#audit apeiron_word.fixed_exp_numerator_perturb
#audit apeiron_word.saturation_numerator_perturb
#audit apeiron_word.exp_at_numerator_case
#audit apeiron_word.attn_output_perturb
#audit apeiron_word.attn_numerator_facts
#audit apeiron_word.attn_stage_compose
#audit apeiron_word.attn_stage_compose_closed
#audit apeiron_word.attn_domain_v2_witness
-- Numeric.AttnLoopScore
#audit apeiron_word.row_dot_fixed_exact
#audit apeiron_word.score_at_spec
#audit apeiron_word.attn_head_loop0_spec
#audit apeiron_word.attnNumeratorCase_le
#audit apeiron_word.attn_head_loop1_spec
-- Numeric.AttnLoopOutput
#audit apeiron_word.attn_head_loop2_spec
#audit apeiron_word.attn_value_component_loop_spec
#audit apeiron_word.value_component_spec
#audit apeiron_word.attn_head_loop3_spec
-- Numeric.AttnHeadWindowsScore
#audit apeiron_word.attn_v2_prefix
#audit apeiron_word.attn_v2_loop0_domain
#audit apeiron_word.attn_v2_loop1_domain
#audit apeiron_word.attn_loop0_ok
#audit apeiron_word.attn_loop1_ok
#audit apeiron_word.attn_loop_stage_facts
-- Numeric.AttnHeadWindowsValue
#audit apeiron_word.attn_weight_bounds
#audit apeiron_word.attn_weight_window_of_numerators
#audit apeiron_word.plane_at_spec
#audit apeiron_word.attn_value_fit
#audit apeiron_word.attn_value_window_of_v2
#audit apeiron_word.attn_value_components
-- Numeric.AttnScoreStage
#audit apeiron_word.attn_scale_bound
#audit apeiron_word.attnScoreInt_err
#audit apeiron_word.attn_score_stage
-- Numeric.AttnHeadRefines
#audit apeiron_word.attn_head_refines_v2_proved
-- Numeric.EngineLmHeadRefinement
#audit apeiron_word.lm_head_refines_proved
-- Numeric.DequantRefinement
#audit apeiron_word.f16_to_fixed_exact
#audit apeiron_word.f16_to_fixed_nonfinite
#audit apeiron_word.f16_to_fixed_real
#audit apeiron_word.read_u16_ok
#audit apeiron_word.dequant_f16_row_exact
#audit apeiron_word.dequant_row_fixed_f16
#audit apeiron_word.block_geom_f16
#audit apeiron_word.matvec_domain_f16_witness
#audit apeiron_word.matvec_f16_witness_refines
-- Numeric.DequantQ4K
#audit apeiron_word.q4_k_block_spec
#audit apeiron_word.q4k_lane_loop_spec
#audit apeiron_word.q4k_group_loop_spec
#audit apeiron_word.dequant_q4_k_row_spec
#audit apeiron_word.dequant_row_fixed_q4_k
-- Numeric.DequantQ6K
#audit apeiron_word.q6_k_block_ok
#audit apeiron_word.q6kLane_loop_spec
#audit apeiron_word.q6kGroup_loop_spec
#audit apeiron_word.dequant_q6_k_row_spec
#audit apeiron_word.dequant_row_fixed_q6_k
-- Numeric.DequantKWitness
#audit apeiron_word.q4kBlockDomain_witness
#audit apeiron_word.q4kWitness_row
#audit apeiron_word.q6kBlockDomain_witness
#audit apeiron_word.q6kWitness_row
-- Numeric.DequantRefusals
#audit apeiron_word.dequant_f16_row_short
#audit apeiron_word.dequant_q4_k_row_short
#audit apeiron_word.dequant_q4_k_row_nonfinite_d
#audit apeiron_word.dequant_q4_k_row_nonfinite_dmin
#audit apeiron_word.dequant_q6_k_row_short
#audit apeiron_word.dequant_q6_k_row_nonfinite_d
-- Numeric.DequantQ8_0
#audit apeiron_word.q8_0_block_ok
#audit apeiron_word.dequant_q8_0_row_spec
#audit apeiron_word.dequant_row_fixed_q8_0
#audit apeiron_word.q8_0BlockDomain_witness
#audit apeiron_word.q8_0Witness_row
-- Numeric.PackingUnpack
#audit apeiron_word.unpack_code_loop_spec
#audit apeiron_word.unpack_code_ref
-- Numeric.PackingEpilogueFold
#audit apeiron_word.epilogue_entry_fixed_min
#audit apeiron_word.epilogue_loop2_fixed_spec
#audit apeiron_word.epilogue_loop1_fixed_spec
-- Numeric.RopeHeadContent
#audit apeiron_word.rotPrefix_succ
#audit apeiron_word.rotate_head_loop_content
#audit apeiron_word.rotate_head_content
#audit apeiron_word.rotate_head_bound
-- Numeric.RopeHeadWitness
#audit apeiron_word.rope_head_domain_witness
#audit apeiron_word.rope_head_witness_ok
-- Numeric.SiluSaturation
#audit apeiron_word.silu_high_real
#audit apeiron_word.silu_low_real
#audit apeiron_word.silu_refines_total
-- Numeric.RmsNormIdeal
#audit apeiron_word.rmsMean_real_bound
#audit apeiron_word.inv_sqrt_lipschitz
#audit apeiron_word.rms_norm_refines_ideal
-- Nous.Model.BlockComposition
#audit Nous.Model.Stage.comp_bound
#audit Nous.Model.Stage.matvecRef_lipschitz
#audit Nous.Model.block_error_bound
#audit Nous.Model.toyBlock_bound
-- Nous.Model.BlockGrid
#audit Nous.Model.siluStage_dom_witness
#audit Nous.Model.rmsStage_dom_witness
#audit Nous.Model.block_error_bound'
#audit Nous.Model.fullBlock'_delta_eq
-- Spirv.CheckScan
#audit Spirv.ty_get_spec
#audit Spirv.defs_contains_spec
#audit Spirv.firstLookup_reverse_lookupTy
#audit Spirv.ty_get_eq_lookupTy_of_nodup
-- Numeric.DequantQ5_0
#audit apeiron_word.q5_0_d_ok
#audit apeiron_word.q5_0_codes_ok
#audit apeiron_word.q5_0_block_ok
#audit apeiron_word.dequant_q5_0_row_spec
#audit apeiron_word.dequant_row_fixed_q5_0
#audit apeiron_word.q5_0BlockDomain_witness
#audit apeiron_word.q5_0Witness_row
-- Spirv.CheckScanState
#audit Spirv.decoration_admitted_spec
#audit Spirv.storage_class_admitted_spec
#audit Spirv.resolve_tag_spec
#audit Spirv.ty_insert_spec
#audit Spirv.apply_shape_loop0_body_cont
#audit Spirv.apply_shape_loop0_body_undefined
-- Numeric.DequantQ2K
#audit apeiron_word.q2_k_codes_ok
#audit apeiron_word.q2_k_block_spec
#audit apeiron_word.dequant_row_fixed_q2_k
#audit apeiron_word.q2kLane_loop_spec
#audit apeiron_word.q2kGroup_loop_spec
#audit apeiron_word.dequant_q2_k_row_spec
#audit apeiron_word.q2kBlockDomain_witness
-- Spirv.CheckApplyShape
#audit Spirv.apply_shape_scan_spec
#audit Spirv.apply_shape_loop0_spec
#audit Spirv.apply_shape_loop7_spec
#audit Spirv.apply_shape_loop0_body_exit
-- Numeric.DequantQ3K
#audit apeiron_word.q3_k_block_ok
#audit apeiron_word.dequant_q3_k_row_spec
#audit apeiron_word.dequant_row_fixed_q3_k
#audit apeiron_word.q3kBlockDomain_witness
-- Ternary/Hadamard epic: Sylvester Hadamard algebra (the ternary-hadamard note )
#audit apeiron_word.sylvester_succ
#audit apeiron_word.sylvester_transpose
#audit apeiron_word.sylvester_mul_self
#audit apeiron_word.sylvester_one
#audit apeiron_word.sylvester_one_channelPair
#audit apeiron_word.two_not_isUnit_int
#audit apeiron_word.sylvester_mulVec_butterfly
-- Ternary/Hadamard epic W3: low-bit row laws and the exact integer FWHT
-- Numeric.DequantQ1Q2
#audit apeiron_word.dequant_q1_0_row_spec
#audit apeiron_word.dequant_row_fixed_q1_0
#audit apeiron_word.dequant_pq2_0_row_spec
#audit apeiron_word.dequant_row_fixed_pq2_0
#audit apeiron_word.q1_0Witness_row
#audit apeiron_word.q2_0Witness_row
#audit apeiron_word.pq2_0Witness_row
-- Numeric.DequantQ1Q2Block
#audit apeiron_word.q1_q2_block_ok
#audit apeiron_word.pq2_0_block_ok
#audit apeiron_word.dequant_q2_0_row_spec
#audit apeiron_word.dequant_row_fixed_q2_0
-- Numeric.DequantTQ2_0
#audit apeiron_word.tq2_0_block_ok
#audit apeiron_word.dequant_tq2_0_row_spec
#audit apeiron_word.dequant_row_fixed_tq2_0
#audit apeiron_word.tq2_0Witness_row
-- Numeric.DequantTQ1_0
#audit apeiron_word.trit_spec
#audit apeiron_word.trit_digits
#audit apeiron_word.tq1_0_block_ok
#audit apeiron_word.ptq1_0_block_ok
#audit apeiron_word.dequant_row_fixed_tq1_0
-- Numeric.DequantTQ1_0Row
#audit apeiron_word.dequant_tq1_0_row_spec
#audit apeiron_word.tq1_0Witness_row
-- Numeric.DequantPTQ1_0
#audit apeiron_word.dequant_ptq1_0_row_spec
#audit apeiron_word.dequant_row_fixed_ptq1_0_spec
#audit apeiron_word.ptq1_0Witness_row
-- Numeric.HadamardStage
#audit apeiron_word.fwhtRef_segment
#audit apeiron_word.fwhtRef_abs_le
#audit apeiron_word.block_exponent_spec_pow
#audit apeiron_word.block_exponent_spec_not_pow
-- Numeric.HadamardRefinement
#audit apeiron_word.fwht_int_spec
#audit apeiron_word.fwht_int_sylvester
-- Numeric.DequantLowBitRefusals (ternary/Hadamard )
#audit apeiron_word.dequant_q1_0_row_short
#audit apeiron_word.dequant_q1_0_row_nonfinite_d
#audit apeiron_word.dequant_q2_0_row_short
#audit apeiron_word.dequant_q2_0_row_nonfinite_d
#audit apeiron_word.dequant_pq2_0_row_short
#audit apeiron_word.dequant_pq2_0_row_nonfinite_d
#audit apeiron_word.dequant_tq2_0_row_short
#audit apeiron_word.dequant_tq2_0_row_nonfinite_d
#audit apeiron_word.dequant_tq1_0_row_short
#audit apeiron_word.dequant_tq1_0_row_nonfinite_d
#audit apeiron_word.dequant_ptq1_0_row_short
#audit apeiron_word.dequant_ptq1_0_row_nonfinite_d
-- AI formalization lanes (the frontier-consumer note; ternary-hadamard partial)
-- Numeric.MatvecBridge
#audit MatvecBridge.rowDotReal_eq_matvecRef
#audit MatvecBridge.matvec_stage_proved
-- Numeric.HadamardFixed
#audit apeiron_word.round_q32_signed_spec
#audit apeiron_word.C_close
#audit apeiron_word.fwht_fixed_loop1_spec
#audit apeiron_word.fwht_fixed_loop2_spec
#audit apeiron_word.fwht_fixed_not_pow
#audit apeiron_word.fwht_fixed_bad_sign
-- Nous.Model.FaceRecurrence
#audit Nous.Model.affComp_assoc
#audit Nous.Model.affApply_comp
#audit Nous.Model.recur_eq_scan
#audit Nous.Model.scan_split
#audit Nous.Model.recur_channel_split
-- Nous.Model.FaceContraction
#audit Nous.Model.euclid_contraction
#audit Nous.Model.hyper_contraction
#audit Nous.Model.para_contraction
#audit Nous.Model.paraRate_exists_lt_one
#audit Nous.Model.contraction_orbit
#audit Nous.Model.bounded_orbit
-- Nous.Model.SpeculativeAccept
#audit Nous.Model.specStep_eq_target
#audit Nous.Model.accept_rate
-- Nous.Model.RmsVectorStage
#audit Nous.Model.rmsIdealVec_lipschitz
#audit Nous.Model.rmsVectorStage_dom_witness
-- Nous.Model.RotationStage
#audit Nous.Model.hadamardRef_lipschitz_sup
#audit Nous.Model.hadamardRef_isometry_l2
#audit Nous.Model.hadamardRef_involutive
-- Numeric.MatvecBridgeGrid
#audit MatvecBridgeGrid.gridVec_injective
#audit MatvecBridgeGrid.matvec_stage_grid_proved
#audit MatvecBridgeGrid.matvecSpecGrid_dom_nonempty
-- Nous.Model.BlockGrounded
#audit Nous.Model.gridProj_stage_defect
#audit Nous.Model.block_error_bound_grounded
#audit Nous.Model.fullBlockGrounded_delta_closed
-- Numeric.HadamardFixedSpec (ternary/Hadamard headline)
#audit apeiron_word.fwht_fixed_loop3_spec
#audit apeiron_word.fwht_fixed_spec
#audit apeiron_word.fwht_fixed_refines
#audit apeiron_word.fwht_fixed_refines_sylvester
-- Numeric.KvReachable (attn_domain_v2 structural reachability)
#audit apeiron_word.KvReachable_inv
#audit apeiron_word.attn_domain_v2_structural
#audit apeiron_word.attn_head_refines_reachable
-- Spirv.CheckApplyShapeAssembly
#audit Spirv.apply_shape_sudot_checks
#audit Spirv.apply_shape_spec
#audit Spirv.ty_get_deref_spec
-- Numeric.PackingDecodeFields
#audit apeiron_word.decode_block_fields_spec
-- Numeric.RopeRotate
#audit apeiron_word.rotate_loop_spec
#audit apeiron_word.rotate_refuses
#audit apeiron_word.rope_two_head_witness
-- Nous.Model.AttnStage
#audit Nous.Model.attnRef_lipschitz
#audit Nous.Model.attnHeadStage_dom_witness
-- Nous.Model.BlockGroundedWitness
#audit Nous.Model.fullBlockGrounded_dom_nonempty
-- Nous.Model.AttnStageWitness
#audit Nous.Model.attnHeadStage_witness
#audit Nous.Model.attnNZState_witness
-- Nous.Model.TopKRouter (MoE )
#audit Nous.Model.topkSel_spec
#audit Nous.Model.topkSel_perm_invariant
#audit Nous.Model.softmax_topk_renorm
#audit Nous.Model.softmax_topk_renorm_max
#audit Nous.Model.topk_monotone_softmax
-- Numeric.PackingDecodeBlock
#audit apeiron_word.epilogue_fixed_spec
#audit apeiron_word.decode_block_spec
-- Nous.Model.BlockGroundedWitness1 (finding: a Stage pinned to the global product is impossible for h ≥ 1)
#audit Nous.Model.pinnedMul_stage_impossible
#audit Nous.Model.fullBlockGrounded_dom_nonempty_1
-- Nous.Model.BlockMulBox (gate-product stage; non-degenerate block witness)
#audit Nous.Model.clampB_lipschitz
#audit Nous.Model.mulBoxStage_ref_on
#audit Nous.Model.blockResidual_nonempty
#audit Nous.Model.fullBlockGrounded_dom_nonempty_111
-- Nous.Model.MoeStage (MoE )
#audit Nous.Model.moe_not_continuous
#audit Nous.Model.fixedMoe_lipschitz
#audit Nous.Model.moe_lipschitz_on_stable
#audit Nous.Model.margin_stable
-- Numeric.MatvecChunk (row-chunked projection = the whole projection)
#audit apeiron_word.matvec_fixed_rows
#audit apeiron_word.matvec_fixed_chunk
#audit apeiron_word.rowOut_join
#audit apeiron_word.matvec_fixed_chunks
-- Policy.Core / Policy.Laws (policy lane )
#audit Apeiron.Policy.Atom.eval
#audit Apeiron.Policy.guardEval
#audit Apeiron.Policy.mask
#audit Apeiron.Policy.decide
#audit Apeiron.Policy.granted
#audit Apeiron.Policy.Completes
#audit Apeiron.Policy.decide_default_deny
#audit Apeiron.Policy.decide_forbid_wins
#audit Apeiron.Policy.decide_granted_le_want
#audit Apeiron.Policy.decide_granted_mono_completion
#audit Apeiron.Policy.decide_perm
#audit Apeiron.Policy.decide_add_forbid_le
#audit Apeiron.Policy.decide_deny_named
#audit Apeiron.Policy.oneBit_no_attenuate
#audit Apeiron.Policy.noAsk_no_ask
#audit Apeiron.Policy.witness_allow
#audit Apeiron.Policy.witness_attenuate
#audit Apeiron.Policy.witness_ask
#audit Apeiron.Policy.witness_deny_noPermit
#audit Apeiron.Policy.witness_deny_forbid
-- Policy.Analysis (policy lane )
#audit Apeiron.Policy.Analysis.atomRows
#audit Apeiron.Policy.Analysis.negAtomRows
#audit Apeiron.Policy.Analysis.guardRows
#audit Apeiron.Policy.Analysis.canAllow
#audit Apeiron.Policy.Analysis.atomRows_iff
#audit Apeiron.Policy.Analysis.negAtomRows_iff
#audit Apeiron.Policy.Analysis.shadowed_witness
#audit Apeiron.Policy.Analysis.live_witness
#audit Apeiron.Policy.Analysis.unknown_exact_witness
#audit Apeiron.Policy.Analysis.neverAllows_of_canAllow_false
#audit Apeiron.Policy.Analysis.exists_allow_of_canAllow_true
-- Numeric.ChartDot (int8 activation quantizer spec, chart factoring)
#audit ChartDot.qcode_range
#audit ChartDot.qcode_err
#audit ChartDot.scale_eq_zero_block
#audit ChartDot.qcode_smul
#audit ChartDot.quantBlock_idem
#audit ChartDot.dequantBlock_idem
#audit ChartDot.chart_factor
#audit ChartDot.dequant_block_factor
#audit ChartDot.row_err_bound
-- Numeric.MoeRouter (extracted route_topk = topkSel)
#audit apeiron_word.best_unchosen_spec
#audit apeiron_word.route_topk_spec
#audit apeiron_word.route_topk_zero
#audit apeiron_word.route_topk_bad_k
-- Numeric.LutPQ2 (exact lookup-table decomposition of the PQ2_0 row dot)
#audit apeiron_word.LutPQ2.block_dot_lut
#audit apeiron_word.LutPQ2.row_dot_lut
#audit apeiron_word.LutPQ2.dequant_dot_lut
#audit apeiron_word.LutPQ2.lutEntry_bitplane
#audit apeiron_word.LutPQ2.lutEntry_mirror
#audit apeiron_word.LutPQ2.planeSum_bound
#audit apeiron_word.LutPQ2.lutEntry_bound
#audit apeiron_word.LutPQ2.planeSum_shift_bound

-- Numeric.PrunedArgmax (bound-pruned exact greedy argmax; Hölder block bound)
#audit PrunedArgmax.pruned_argmax_exact
#audit PrunedArgmax.holder_block_bound
#audit PrunedArgmax.argmax_fixed_returns_lowest
#audit PrunedArgmax.word_argmax_fixed_returns_lowest
#audit PrunedArgmax.Witness.witness

-- Numeric.ColumnCover (exact FFN split by hidden slice; block-local quantization)
#audit ColumnCover.column_cover_round
#audit ColumnCover.column_cover_row
#audit ColumnCover.column_cover_matrix
#audit ColumnCover.quant_locality
#audit ColumnCover.dequant_locality
#audit ColumnCover.split_ffn_exact
#audit ColumnCover.split_ffn_exact_dequant
#audit ColumnCover.Witness.split_eq

-- Nous.Combinator.Church (Church numerals, B, W in the canonical SK calculus)
#audit Nous.Combinator.B_spec
#audit Nous.Combinator.W_spec
#audit Nous.Combinator.church_spec
#audit Nous.Combinator.church_add
#audit Nous.Combinator.church_mul
#audit Nous.Combinator.church_double
#audit Nous.Combinator.church_separated

-- Nous.Combinator.ListHom (list homomorphisms: any split and merge tree agree)
#audit Nous.Combinator.ListHom.hom_flatten
#audit Nous.Combinator.ListHom.hom_tree
#audit Nous.Combinator.ListHom.add_hom_tree
#audit Nous.Combinator.ListHom.colSum_tree
#audit Nous.Combinator.ListHom.quant_flatten
#audit Nous.Combinator.ListHom.quant_tree
#audit Nous.Combinator.ListHom.os_denominator
#audit Nous.Combinator.ListHom.os_tree
#audit Nous.Combinator.ListHom.os_numerator
#audit Nous.Combinator.ListHom.os_numerator_tree

-- Numeric.RopeTurns (RoPE angle on the turn ring Z/2^64; spec, Rust link pending L9-T3)
#audit Numeric.RopeTurns.phase_add
#audit Numeric.RopeTurns.phase_mul
#audit Numeric.RopeTurns.lift_range
#audit Numeric.RopeTurns.lift_congr
#audit Numeric.RopeTurns.turn_to_radians_window
#audit Numeric.RopeTurns.turn_angle_error
#audit Numeric.RopeTurns.turn_angle_congr
#audit Numeric.RopeTurns.turn_total_error

-- Nous.Combinator.Orbit (Church numeral read on a rotation orbit and on the turn ring Z/2^64)
#audit Nous.Combinator.Orbit.readFO_iterApp
#audit Nous.Combinator.Orbit.church_reads_orbit
#audit Nous.Combinator.Orbit.church_ring_mod
#audit Nous.Combinator.Orbit.church_ring_distinct
#audit Nous.Combinator.Orbit.turn_iter
#audit Nous.Combinator.Orbit.turn_period
#audit Nous.Combinator.Orbit.church_turn_mod

-- Nous.Combinator.DoubleCover (Spin(2) -> SO(2) degree 2; exact Pythagorean rotations; Gauss product)
#audit Nous.Combinator.DoubleCover.W_squares
#audit Nous.Combinator.DoubleCover.sq_fiber
#audit Nous.Combinator.DoubleCover.sq_surjective_circle
#audit Nous.Combinator.DoubleCover.sq_two_to_one
#audit Nous.Combinator.DoubleCover.pythagorean_rotation
#audit Nous.Combinator.DoubleCover.spinor_rotation_components
#audit Nous.Combinator.DoubleCover.half_angle
#audit Nous.Combinator.DoubleCover.gauss_three_mult

-- RoPE turn-ring Rust link (Numeric.EngineRopeRefinement, L9)
#audit apeiron_word.turn_to_radians_refines
#audit apeiron_word.turn_phase_refines
#audit apeiron_word.turn_angle_trig_window

-- Nous.Seq.Fenwick
#audit Nous.Seq.Fenwick.buckets_disjoint
#audit Nous.Seq.Fenwick.buckets_cover
#audit Nous.Seq.Fenwick.bucket_count
#audit Nous.Seq.Fenwick.merges_total_le
#audit Nous.Seq.Fenwick.loglinear_eq_masked

-- Nous.Seq.LazySoftmax
#audit Nous.Seq.LazySoftmax.softmax_ref_invariant
#audit Nous.Seq.LazySoftmax.lazy_exact
#audit Nous.Seq.LazySoftmax.lazy_bounded
#audit Nous.Seq.LazySoftmax.lazy_merge_tree_exact
#audit Nous.Seq.LazySoftmax.rescale_count_le

-- Nous.Seq.GatedDelta
#audit Nous.Seq.GatedDelta.recurrent_eq_chunked
#audit Nous.Seq.GatedDelta.closed_form
#audit Nous.Seq.GatedDelta.wy_form
#audit Nous.Seq.GatedDelta.gated_wy_form

-- Nous.Seq.CertifiedSoftmax
#audit Nous.Seq.CertifiedSoftmax.prune_max_preserved
#audit Nous.Seq.CertifiedSoftmax.prune_exact
#audit Nous.Seq.CertifiedSoftmax.box_bound_sound
#audit Nous.Seq.CertifiedSoftmax.ball_bound_sound
#audit Nous.Seq.CertifiedSoftmax.block_prune_exact

-- Nous.Combinator.Sphere
#audit Nous.Combinator.Sphere.sylvester_church
#audit Nous.Combinator.Sphere.fwht_add_count
#audit Nous.Combinator.Sphere.sumsq_tree
#audit Nous.Combinator.Sphere.rms_projects_to_sphere
#audit Nous.Combinator.Sphere.rms_equivariant
#audit Nous.Combinator.Sphere.gain_fold
#audit Nous.Combinator.Sphere.hadamard_is_rotation
#audit Nous.Combinator.Sphere.cayley_orthogonal

-- Sanguine.SpinorLoop
#audit Sanguine.SpinorLoop.UnitaryLoop.cayley_conserves_born_density
-- Nous.Model.JacobiDecode (Jacobi/lookahead decoding = greedy)
#audit Nous.Model.jacobi_prefix_correct
#audit Nous.Model.jacobi_exact
#audit Nous.Model.jacobi_fixed_point_iff
#audit Nous.Model.verify_accept_prefix
#audit Nous.Model.verify_accept_append
#audit Nous.Model.passes_bound
#audit Nous.Model.Witness.jacobi_exact_witness

-- Nous.Seq.ProductKeys
#audit Nous.Seq.ProductKeys.product_topk_subset
#audit Nous.Seq.ProductKeys.product_topk_exact
#audit Nous.Seq.ProductKeys.multi_product_topk
#audit Nous.Seq.ProductKeys.multi_slot_card
#audit Nous.Seq.ProductKeys.multi_candidate_card
#audit Nous.Seq.ProductKeys.slot_address_bijective
#audit Nous.Seq.ProductKeys.slot_address_val
#audit Nous.Seq.ProductKeys.witness_top2_eq

-- Nous.Seq.HyperbolicAddress
#audit Nous.Seq.HyperbolicAddress.slot_channels
#audit Nous.Seq.HyperbolicAddress.slot_normForm
#audit Nous.Seq.HyperbolicAddress.slot_mul
#audit Nous.Seq.HyperbolicAddress.normForm_mono_channels
#audit Nous.Seq.HyperbolicAddress.hyperbolicSlotScore_eq
#audit Nous.Seq.HyperbolicAddress.two_phase_exact_hyperbolic
#audit Nous.Seq.HyperbolicAddress.log_additive
#audit Nous.Seq.HyperbolicAddress.hyperbolic_precPair_eq
#audit Nous.Seq.HyperbolicAddress.hyperbolic_product_topk_bridge
#audit Nous.Seq.HyperbolicAddress.split_iff_hyperbolic
#audit Nous.Seq.HyperbolicAddress.split_iff_nontrivial_idempotent
#audit Nous.Seq.HyperbolicAddress.parabolic_one_channel
#audit Nous.Seq.HyperbolicAddress.elliptic_normForm_pos_def
#audit Nous.Seq.HyperbolicAddress.elliptic_no_channels
#audit Nous.Seq.HyperbolicAddress.rank1_core_is_cell
#audit Nous.Seq.HyperbolicAddress.tucker_rank2_breaks
#audit Nous.Seq.HyperbolicAddress.threshold_bound
#audit Nous.Seq.HyperbolicAddress.rank_r_threshold_exact
#audit Nous.Seq.HyperbolicAddress.Witness.slot_mul_sigma1
#audit Nous.Seq.HyperbolicAddress.Witness.slot_mul_sigma4
#audit Nous.Seq.HyperbolicAddress.Witness.two_phase_3x3_instance
#audit Nous.Seq.HyperbolicAddress.Witness.parabolic_boundary_witness
#audit Nous.Seq.HyperbolicAddress.Witness.elliptic_boundary_witness

-- Nous.Seq.ThresholdCells
#audit Nous.Seq.ThresholdCells.score_as_cells
#audit Nous.Seq.ThresholdCells.unseen_le_threshold
#audit Nous.Seq.ThresholdCells.threshold_stop_exact
#audit Nous.Seq.ThresholdCells.seen_card_le
#audit Nous.Seq.ThresholdCells.seen_rank1
#audit Nous.Seq.ThresholdCells.rank1_candidate_exact
#audit Nous.Seq.ThresholdCells.Witness.tucker_threshold_d1
#audit Nous.Seq.ThresholdCells.Witness.tucker_threshold_d2
#audit Nous.Seq.ThresholdCells.Witness.tucker_stop_d1_fails
#audit Nous.Seq.ThresholdCells.Witness.tucker_exact_d2
#audit Nous.Seq.ThresholdCells.Witness.sigma4_enters_only_through_slotCell

-- Numeric.BitplaneArgmax
#audit BitplaneArgmax.nibble_split
#audit BitplaneArgmax.nibble_hi_range
#audit BitplaneArgmax.lo_interval_sound
#audit BitplaneArgmax.bitplane_prune_sound
#audit BitplaneArgmax.survivors_nonempty
#audit BitplaneArgmax.bitplane_argmax_exact
#audit BitplaneArgmax.bitplane_argmax_exact_eq
#audit BitplaneArgmax.witness_row2_pruned
#audit BitplaneArgmax.witness_row1_survives
#audit BitplaneArgmax.lo_upper_needed

-- Nous.Model.ParabolicAttention
#audit Nous.Model.parabolic_streaming_exact
#audit Nous.Model.parabolicScore_eq_ringMul
#audit Nous.Model.lin_streaming_num_eq
#audit Nous.Model.lin_streaming_den_eq

-- Nous.Model.EllipticAttention
#audit Nous.Model.elliptic_normForm_mul
#audit Nous.Model.elliptic_isometry
#audit Nous.Model.elliptic_no_zero_divisors
#audit Nous.Model.elliptic_rope_relative_score

-- Nous.Model.TrichotomyAttention
#audit Nous.Model.face_dispatch_hyperbolic
#audit Nous.Model.face_dispatch_parabolic
#audit Nous.Model.face_dispatch_elliptic
#audit Nous.Model.trichotomy_streaming_linear_complexity

-- Nous.Seq.AffineMixer
#audit Nous.Seq.AffineMixer.affComp_assoc
#audit Nous.Seq.AffineMixer.affApply_comp
#audit Nous.Seq.AffineMixer.scanSeq_eq_compose
#audit Nous.Seq.AffineMixer.recurrent_eq_chunked
#audit Nous.Seq.AffineMixer.elliptic_2x2_mul_equiv

-- Nous.Model.EllipticModes
#audit Nous.Model.EllipticModes.rotorMul_norm
#audit Nous.Model.EllipticModes.rotorPow_norm
#audit Nous.Model.EllipticModes.unitary_rotor_preserves_norm
#audit Nous.Model.EllipticModes.real_decay_strictly_positive
#audit Nous.Model.EllipticModes.quarterTurnRotor_sq
#audit Nous.Model.EllipticModes.elliptic_sign_alternation_witness
#audit Nous.Model.EllipticModes.quarterTurnRotor_unitary

-- Nous.Combinator.CellTransform
#audit Nous.Combinator.CellTransform.det_lorentzBoost
#audit Nous.Combinator.CellTransform.lorentzBoost_det_one
#audit Nous.Combinator.CellTransform.lorentzBoost_preserves_norm
#audit Nous.Combinator.CellTransform.lorentzBoost_inv
#audit Nous.Combinator.CellTransform.det_givensRot
#audit Nous.Combinator.CellTransform.givensRot_det_one
#audit Nous.Combinator.CellTransform.givensRot_preserves_norm
#audit Nous.Combinator.CellTransform.givensRot_inv
#audit Nous.Combinator.CellTransform.det_parabolicShear
#audit Nous.Combinator.CellTransform.parabolicShear_inv

-- Nous.Model.DualCertificate
#audit Nous.Model.DualCertificate.dual_slack_bounds_actual
#audit Nous.Model.DualCertificate.dual_cert_preserves_argmax
#audit Nous.Model.DualCertificate.witness_certificate_success
#audit Nous.Model.DualCertificate.witness_slack_reversal_exists

-- Nous.Seq.TwoChannelRouter
#audit Nous.Seq.TwoChannelRouter.e_pos_add_e_neg
#audit Nous.Seq.TwoChannelRouter.e_pos_sq
#audit Nous.Seq.TwoChannelRouter.e_neg_sq
#audit Nous.Seq.TwoChannelRouter.e_pos_mul_e_neg_zero
#audit Nous.Seq.TwoChannelRouter.cell_decompose
#audit Nous.Seq.TwoChannelRouter.cross_channel_zero

-- Nous.Seq.BitplaneRouter
#audit Nous.Seq.BitplaneRouter.residual_bound
#audit Nous.Seq.BitplaneRouter.bitplane_pruning_negative
#audit Nous.Seq.BitplaneRouter.witness_bitplane_pruned

-- Nous.Seq.StatePrecision
#audit Nous.Seq.StatePrecision.errorSeq_nonneg
#audit Nous.Seq.StatePrecision.errorSeq_telescope
#audit Nous.Seq.StatePrecision.errorSeq_le_steady_state
#audit Nous.Seq.StatePrecision.witness_steady_state_bound
#audit Nous.Seq.StatePrecision.witness_steady_state_pos

-- Nous.Seq.ChunkedRounding
#audit Nous.Seq.ChunkedRounding.stepDiscrepancy_nonneg
#audit Nous.Seq.ChunkedRounding.stepDiscrepancy_telescope
#audit Nous.Seq.ChunkedRounding.chunked_discrepancy_eq
#audit Nous.Seq.ChunkedRounding.chunked_discrepancy_uniform_bound
#audit Nous.Seq.ChunkedRounding.witness_delta_net_100_lsb
#audit Nous.Seq.ChunkedRounding.witness_delta_net_bound_pos

-- Nous.Model.SpectralTrichotomy
#audit Nous.Model.SpectralTrichotomy.toMatrix_mul
#audit Nous.Model.SpectralTrichotomy.toMatrix_one
#audit Nous.Model.SpectralTrichotomy.toMatrix_det
#audit Nous.Model.SpectralTrichotomy.toMatrix_trace
#audit Nous.Model.SpectralTrichotomy.charpoly_eq
#audit Nous.Model.SpectralTrichotomy.charpoly_discriminant
#audit Nous.Model.SpectralTrichotomy.hyperbolic_discriminant_pos
#audit Nous.Model.SpectralTrichotomy.elliptic_discriminant_neg
#audit Nous.Model.SpectralTrichotomy.parabolic_discriminant_zero
#audit Nous.Model.SpectralTrichotomy.parabolic_nilpotent_sq
#audit Nous.Model.SpectralTrichotomy.parabolic_jordan_conjugacy
#audit Nous.Model.SpectralTrichotomy.trig_mul_preserves_norm
#audit Nous.Model.SpectralTrichotomy.parabolic_shear_is_trig
#audit Nous.Model.SpectralTrichotomy.parabolic_trig_addition

-- Nous.Model.CertifiedDecision
#audit Nous.Model.CertifiedDecision.true_score_in_bounds
#audit Nous.Model.CertifiedDecision.certified_argmax_preservation
#audit Nous.Model.CertifiedDecision.certified_strict_argmax
#audit Nous.Model.CertifiedDecision.certified_gap_condition
#audit Nous.Model.CertifiedDecision.candidate_prunable
#audit Nous.Model.CertifiedDecision.activeCandidates_card_le
#audit Nous.Model.CertifiedDecision.leader_mem_activeCandidates
#audit Nous.Model.CertifiedDecision.activeCandidates_singleton_of_certified
#audit Nous.Model.CertifiedDecision.unprunable_when_slack_le

-- Spirv.SplitLimb
#audit Spirv.SplitLimb.splitMul_eq
#audit Spirv.SplitLimb.roundShift32_perturbation
#audit Spirv.SplitLimb.reduction_tree4_exact
#audit Spirv.SplitLimb.reduction_tree4_zero_drift
#audit Spirv.SplitLimb.reduction_tree8_exact

-- Nous.Model.SpeculativeRollback
#audit Nous.Model.spec_round_eq_greedy
#audit Nous.Model.spec_decode_eq_greedy
#audit Nous.Model.verifyAux_tokens_eq_accepted
#audit Nous.Model.spec_round_tokens_eq_greedy_via_jacobi
#audit Nous.Model.undo_write
#audit Nous.Model.undo_writePass
#audit Nous.Model.undo_write_witness
#audit Nous.Model.truncate_append_length
#audit Nous.Model.truncate_length_add_append
#audit Nous.Model.RollbackWitness.spec_round_witness
#audit Nous.Model.RollbackWitness.spec_round_witness_all
