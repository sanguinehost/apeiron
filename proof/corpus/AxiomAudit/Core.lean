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
import Numeric.Engine.Exp
import Numeric.Packing.Format
import Numeric.Engine.Rsqrt
import Numeric.Rope.Refinement
import Numeric.Engine.Silu
import Numeric.Engine.Norm
import Numeric.Engine.Dot
import Numeric.Packing.Refinement
import Numeric.Engine.Matvec
import Numeric.Engine.Sampler
import Numeric.Engine.LmHead
import Numeric.Engine.Attn
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
-- Attention-downstream wave (lead-authorized): `Numeric.AttnLoopScore`, `Numeric.AttnLoopOutput`,
-- `Numeric.AttnHeadWindowsScore`, `Numeric.AttnHeadWindowsValue` and `Numeric.AttnHeadRefines`
-- were moved out of the corpus to `work/wip/attn-downstream/`; their `#audit` rows here went with
-- them. `Numeric.AttnSoftmaxPerturb` and `Numeric.AttnScoreStage` stay: they were retargeted green
-- against the live `attn.KvState` extraction. See that directory's `NOTES.md`.
import Numeric.Attn.SoftmaxPerturb
import Numeric.Attn.ScoreStage
import Numeric.Dequant.Refinement
import Numeric.Dequant.K.Q4K
import Numeric.Dequant.K.Q6K
import Numeric.Dequant.K.Witness
import Numeric.Dequant.Refusals
import Numeric.Dequant.Q8_0
import Numeric.Packing.Unpack
import Numeric.Packing.EpilogueFold
import Numeric.Rope.HeadContent
import Numeric.Rope.HeadWitness
import Numeric.SiluSaturation
import Numeric.RmsNormIdeal
import Nous.Model.BlockComposition
import Nous.Model.BlockGrid
import Spirv.CheckScan
import Numeric.Dequant.Q5_0
import Spirv.CheckScanState
import Numeric.Dequant.K.Q2K
import Spirv.CheckApplyShape
import Numeric.Hadamard.Sylvester
import Numeric.Dequant.Q1Q2
import Numeric.Dequant.Q1Q2Block
import Numeric.Dequant.TQ2_0
import Numeric.Dequant.TQ1_0
import Numeric.Dequant.TQ1_0Row
import Numeric.Dequant.PTQ1_0
import Numeric.Hadamard.Stage
import Numeric.Hadamard.Refinement
import Numeric.Dequant.LowBitRefusals
import Numeric.Matvec.Bridge
import Numeric.Hadamard.Fixed
import Nous.Model.FaceRecurrence
import Nous.Model.FaceContraction
import Nous.Model.SpeculativeAccept
import Nous.Model.RmsVectorStage
import Nous.Model.RotationStage
import Numeric.Matvec.BridgeGrid
import Nous.Model.BlockGrounded
import Numeric.Hadamard.FixedSpec
import Spirv.CheckApplyShapeAssembly
import Numeric.Packing.DecodeFields
import Numeric.Rope.Rotate
import Nous.Model.AttnStage
import Nous.Model.BlockGroundedWitness
-- Attention-downstream wave (lead-authorized): `Numeric.KvReachable` and
-- `Nous.Model.AttnStageWitness` were moved out of the corpus to `work/wip/attn-downstream/`
-- along with the imports above; their `#audit` rows went with them.
import Nous.Model.TopKRouter
import Numeric.Packing.DecodeBlock
import Nous.Model.BlockGroundedWitness1
import Nous.Model.BlockMulBox
import Nous.Model.MoeStage
import Numeric.Dequant.K.Q3K
import Numeric.Matvec.Chunk
import Policy.Laws
import Policy.Analysis
import Numeric.Matvec.ChartDot
import Numeric.MoeRouter
import Numeric.Matvec.LutPQ2
import Numeric.OneBit.PrunedArgmax
import Numeric.Matvec.ColumnCover
import Nous.Combinator.Church
import Nous.Combinator.ListHom
import Numeric.Rope.Turns
import Nous.Combinator.Orbit
import Nous.Combinator.DoubleCover
import Nous.Seq.Fenwick
import Nous.Seq.LazySoftmax
import Nous.Seq.GatedDelta
import Nous.Seq.CertifiedSoftmax
import Nous.Seq.ProductKeys
import Nous.Seq.HyperbolicAddress
import Nous.Seq.ThresholdCells
import Numeric.OneBit.BitplaneArgmax
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
#audit Apeiron.NativeAddress.zmod_prime_pow_idempotent
#audit Apeiron.NativeAddress.h3_40_idempotent_iff
#audit Apeiron.NativeAddress.h3_40_idempotents_ncard
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

-- Tag readback and fingerprint exactness (read-classifier row 2)
#audit Apeiron.GrowableScan.tag_ne_zero
#audit Apeiron.GrowableScan.publish_live_tag_readback
#audit Apeiron.GrowableScan.spillFingerprint
#audit Apeiron.GrowableScan.SpillFingerprintExact
#audit Apeiron.GrowableScan.spillFingerprintExact
#audit Apeiron.GrowableScan.spillFingerprint_eq_of_identity
#audit Apeiron.GrowableScan.spillFingerprint_collision_witness
#audit Apeiron.GrowableScan.fingerprint_hit_requires_full_matches
#audit Apeiron.GrowableReadLoop.generated_probe_stops_on_hit
#audit Apeiron.GrowableReadLoop.generated_probe_stops_at_bound
#audit Apeiron.GrowableReadLoop.stored_candidate_full_match
#audit Apeiron.GrowableReadLoop.StoredProbeReady
#audit Apeiron.GrowableReadLoop.StoredProbeInv
#audit Apeiron.GrowableReadLoop.StoredProbeOutcome
#audit Apeiron.GrowableReadLoop.StoredProbeChainSafe
#audit Apeiron.GrowableReadLoop.generated_probe_sweep_outcome
#audit Apeiron.GrowableReadLoop.generated_probe_sweep_some
#audit Apeiron.GrowableReadLoop.generated_probe_sweep_none
#audit Apeiron.GrowableReadLoop.generated_probe_increment_checked
#audit Apeiron.GrowableReadLoop.stored_probe_visit_unique
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
#audit Apeiron.GrowableReadLoop.vec_index_readback
#audit Apeiron.GrowableReadLoop.stored_matches_slot_iff
#audit Apeiron.GrowableReadLoop.stored_matcher_slot_total
#audit Apeiron.GrowableReadLoop.generated_spill_read_some_run
#audit Apeiron.GrowableReadLoop.generated_spill_read_none_run
#audit Apeiron.GrowableReadLoop.spill_read_none_implies_no_full_match
#audit Apeiron.GrowableReadLoop.spill_read_some_implies_full_match

-- Public read refinement: constructor reachability, the direct-hit partition, and the
-- localization/absence core feeding partitions 2-5
#audit Apeiron.GrowableRead.ConstructorReachableRepR
#audit Apeiron.GrowableRead.word_stored_table_get_at_direct_hit
#audit Apeiron.GrowableRead.word_regen_table_get_at_direct_hit
#audit Apeiron.GrowableRead.query_rep
#audit Apeiron.GrowableRead.absent_of_exclusions
#audit Apeiron.GrowableRead.default_of_absent
-- pending integration: Growable.Read is being rewritten concurrently; the name does not resolve
-- in the current file, so the row is held rather than deleted.
-- #audit Apeiron.GrowableRead.spill_read_none_of_summary_miss
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
-- pending integration: Growable.Read is being rewritten concurrently; these route names do not
-- resolve in the current file and are held rather than deleted.
-- #audit Apeiron.GrowableRead.valid_n_16
-- #audit Apeiron.GrowableRead.StoredReadRoute
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_direct_none_hit
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_flat_none_hit
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_flat_occupied_miss_hit
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_direct_none_sweep_none
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_flat_none_sweep_none
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_flat_occupied_miss_sweep_none
-- #audit Apeiron.GrowableRead.word_stored_table_get_at_root

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
#audit Apeiron.GrowableRelocation.reservedObservation_writeFlat_same
#audit Apeiron.GrowableRelocation.reservedObservation_writeSpill_same
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
#audit Apeiron.GrowableRepresentation.CarrierAddressRespecting
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

-- Locate-loop slice: the `spill_locate` inner completion invariant;
-- `SpillLocateFoundFullMatch` closes after the outer sweep (L4-L6). The scanner-coincidence
-- transfer and the `table.least_set_lane` rows were retired with the linear-probe port.
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

-- : trig refusal surface (Part1) and the Part2/Part3/Part5
-- re-proof headlines that carry the retired reduction, rounding and series rows below. Part2
-- `narrow_spec`, `q_rne_spec`, `q_mul_spec`, `div_fac_spec`; Part3 `q_mul_step`, `div_fac_step`,
-- `sin_remainder`, `cos_remainder`, `sin_cos_series_spec`; Part5 `pi_red_bound`, `k_bound`,
-- `reduction_spec`, `q_rne_half_spec`.
#audit apeiron_word.trig_window
#audit apeiron_word.fixed_sin_cos_refuses
#audit apeiron_word.trig_floor
#audit apeiron_word.fixed_sin_cos_refines
#audit apeiron_word.fixed_sin_cos_refuses_proved
#audit apeiron_word.word_fixed_sin_cos_window_past_guards
#audit apeiron_word.fixed_trig_ONE_eq
#audit apeiron_word.fixed_trig_LOWER_eq
#audit apeiron_word.fixed_trig_UPPER_eq
#audit apeiron_word.narrow_spec
#audit apeiron_word.q_rne_spec
#audit apeiron_word.q_mul_spec
#audit apeiron_word.div_fac_spec
#audit apeiron_word.q_mul_step
#audit apeiron_word.div_fac_step
#audit apeiron_word.sin_remainder
#audit apeiron_word.cos_remainder
#audit apeiron_word.sin_cos_series_spec
#audit apeiron_word.pi_red_bound
#audit apeiron_word.k_bound
#audit apeiron_word.reduction_spec
#audit apeiron_word.q_rne_half_spec

-- Removed with the old `fixed_sin_cos` body (`1c00b28b`) and the Part1/2/3 re-proof; retired
-- names with no counterpart, kept here only as a receipt. Top level:
-- `toReal_{two_pi,two_pi_recip,pi,half_pi,quarter_pi}`, `two_pi_recip_defect`,
-- `{two_pi,pi,half_pi,quarter_pi}_q32_defect`, `{half_pi,quarter_pi}_q32_floor`,
-- `fixed_trig_shift_bound{,_window}`, `fixed_trig_reduction_remainder_range{,_tight}`,
-- `word_fixed_sin_cos_reduction_value`, `word_fixed_sin_cos_fold_value`,
-- `word_fixed_sin_cos_reduction_fold_value`, `word_fixed_sin_cos_value`,
-- `word_fixed_taylor_first`, `word_taylor_spec`, `fixed_trig_taylor_value{,_proved}`,
-- `trig_delta`, `trig_reduction_displacement`, `trig_sin_reduction_transfer`,
-- `trig_cos_reduction_transfer`, `trig_fold_delta`, `fold{1,2}T`, `fold{1,2}_displacement`,
-- `{sin,cos}_fold{1,2}T`, `{sin,cos}_phase_select`, `flag{Sin,Cos}`,
-- `flag_{sin,cos}_{fold,reduction}_bound`, `fixed_trig_fold{1,2}_spec`,
-- `fixed_trig_fold_step_spec`, `fixed_trig_int_shiftRight_nonneg`,
-- `fixed_trig_round_div_2p32{HALF_eq,_spec,_val}`. In `fixed_trig`: `fold1`, `fold1_negate`,
-- `fold2`, `fold2_swap`, `fold1_abs_le`, `fold_arg_abs_le`, `fold_arg_abs_le_quarter`,
-- `fold_step`, `round_q32`, `round_q32_abs_le`, `round_q32_abs_le'`, `round_q32_half`,
-- `round_q32_ledger`, `round_safe`, `round_safe66`, `abs_mul_le`, `abs_mul_le_z`,
-- `abs_add_const_le`, `abs_sub_const_le`, `mul_range`, `{sin,cos}_poly`,
-- `{sin,cos}_poly_horner`, `{sin,cos}Horner`, `{sin,cos}Horner_taylor`, `{sin,cos}Taylor`,
-- `{sin,cos}Taylor_remainder`.

-- : two-stage Horner prefix
#audit apeiron_word.word_fixed_exp_horner_acc1

-- : full Horner chain (bounded cap)
#audit apeiron_word.word_fixed_exp_horner_acc

-- : scaling shift-count bound
#audit apeiron_word.fixed_exp_scale_shift_count_lt

-- : k >= 0 scaling branch
#audit apeiron_word.fixed_exp_scale_nonneg

-- : k < 0 rounding core
#audit apeiron_word.fixed_exp_scale_neg_core

-- : flush and full k < 0 branch
#audit apeiron_word.fixed_exp_scale_flush
#audit apeiron_word.fixed_exp_scale_neg

-- : scaling helpers
#audit apeiron_word.fixed_exp_scale_abs
#audit apeiron_word.fixed_exp_scale_nonneg_ge64

-- : full scaling block
#audit apeiron_word.fixed_exp_scale_abs_raw
#audit apeiron_word.word_fixed_exp_scale

-- : pair-returning reduction chunk
#audit apeiron_word.word_fixed_exp_reduction_value_pair

-- : success-path pair chunk
#audit apeiron_word.word_fixed_exp_reduction_pair_succ

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
#audit apeiron_word.fixed_sin_cos_refines_proved
-- Removed with the old `fixed_sin_cos` body (`1c00b28b`); no counterpart:
-- `word_fixed_sin_cos_past_guards_value`, `sin_out_flagSin_le`, `cos_out_flagCos_le`,
-- `trig_floor_close`, `sin_poly_close_le`, `cos_poly_close_le`, `cos_poly_rem_le`,
-- `abs_smul_cancel`, `c10_let_push`, `c10_let_prod_push`, `c10_pair_bind`.

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
