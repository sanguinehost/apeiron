import AxiomAudit.Serve
import Numeric.Engine.Pow2
import Flux.Discrete
import Count.Helpers
import Count.FreqRep
import Count.OverflowOps
import Count.FreqAdd
import Count.FreqSideOv
import Count.FreqGrow
import Count.FreqInsert
import Count.Aggregate
import Map.PrefixMasks
import Map.ExecutedRead
import Map.ExecutedReadUncond
import Map.ExecutedOverflowInsert
import Map.ExecutedResize
import Map.ExecutedBuild
import Map.ExecutedBuildWitness
import Spirv.Carrier
import Perf.Roofline
import Perf.Concurrency
import Perf.Instances
import Perf.Chain
import Perf.Issue
import Perf.Occupancy
import Perf.Moves
import Perf.MapMoves
import Perf.CountLookup
import Perf.HattnT1
import Perf.HattnT3
import Perf.HattnT4
import Perf.HattnT5
import Perf.HattnT6
import Perf.HattnT7
import Perf.HattnT8
import Perf.HattnT8Q7
import Perf.HattnT8Q7L
import Perf.HattnT8Q8
import Perf.HattnT8Q9
import Perf.HattnT8Farb
import Perf.HattnT8Sum
import Perf.HattnT8SumU
import Perf.HattnT8Layer
import Perf.HattnT8LFarb
import Perf.HattnT8Code
import Perf.HattnT8Tail
import Perf.HattnT8TailProse
import Perf.HattnT8LGroup
import Perf.HattnT8Oracle
import Perf.HattnT8Rerank
import Perf.HattnT8LowBit
import Perf.HattnT8LbPool
import Perf.HattnT8Cells
import Perf.HattnT3Train
import Perf.HattnHotCold
import Perf.PubVerifyCpu
import Perf.OnebitW9
import Perf.OnebitW10
import Perf.Iq1s
import Perf.Elliptic
import Perf.OnebitW13
import Perf.OnebitW14
import Perf.CostCorrespondence
import Perf.FacesPolarBound
import Perf.Identity
import Perf.AttnCoop
import Perf.IdentityFloat
import Perf.Overlap
import Perf.FaceTyped
import Perf.FaceTypedE8
import Perf.SideInfo
import Perf.Generalisation
import Perf.KLTilt
import Learn.GramExact
import Learn.Grad.DenseNorm
import Learn.Grad.DenseNormBwd
import Learn.E8Code
import Learn.FitIcm
import Learn.Limb
import Learn.WaterFill
import Learn.IcmError
import Learn.IcmCount
import Learn.Log2
import Learn.Grad.DenseLinear
import Learn.Grad.DenseLinearWitness
import Learn.Grad.Rope
import Numeric.OneBit.SigmaDeltaRefinement
import Learn.Grad.Head
import Learn.Grad.HeadLoop
import Learn.Grad.HeadRows
import Learn.Grad.HeadSoftmax
import Learn.Solve.GemmSpec
import Learn.Solve.Gemm
import Learn.Solve.GemmWitness
import Learn.Solve.WsyrkSpec
import Learn.Solve.Wsyrk
import Learn.Solve.WsyrkWitness
import Learn.QuantVerdicts
import Perf.BitsRefine
import Numeric.Quantum.Register
import Numeric.Quantum.RegisterRefinement
import Numeric.Quantum.Qubit
import Numeric.Quantum.BranchLaplacian
import Numeric.Quantum.BranchMadelung
import Numeric.Quantum.TetraLimit
import Numeric.Quantum.TetraLaplacian
import Numeric.Quantum.BranchQuantization
import Numeric.Quantum.PathDensity
import Perf.QuantumRotation
import Perf.SpecDraftKL
import Perf.Handoff
import Perf.Train
import Numeric.Quantum.Ledger.FaceShift
import Numeric.Quantum.Ledger.AmplitudeKL
import Numeric.Quantum.Ledger.DesignLadder
import Numeric.Quantum.Ledger.EllipticTensor
import Numeric.Quantum.Ledger.CellClifford
import Numeric.Quantum.Ledger.SicStencil
import Numeric.Quantum.Measured.InterferenceGram
import Numeric.Quantum.Measured.JointFisher
import Numeric.Quantum.Ledger.DeltaEnds
import Numeric.Quantum.Ledger.ExactUncertainty
import Numeric.Quantum.Ledger.EllipticRebit
import Gpu.Handoff

-- Discrete flux (docs/epics/magnetic-flux.md): the lattice divergence theorem, `∇·B = 0`.
#audit Flux.Discrete.sum_range_diff
#audit Flux.Discrete.grid_divergence_theorem
#audit Flux.Discrete.flux_zero_of_divergence_free

-- Count helpers (Count/Helpers.lean): refinement lemmas for the five extracted count helpers.
#audit Apeiron.CountHelpers.home_spec
#audit Apeiron.CountHelpers.word_count_home_lt
#audit Apeiron.CountHelpers.word_count_admit_spec
#audit Apeiron.CountHelpers.word_count_slot_ok
#audit Apeiron.CountHelpers.word_count_slot_refuse
#audit Apeiron.CountHelpers.word_count_slot_spec
#audit Apeiron.CountHelpers.hist_body_step
#audit Apeiron.CountHelpers.hist_loop_spec
#audit Apeiron.CountHelpers.word_count_histogram_spec
#audit Apeiron.CountHelpers.off_inner_step
#audit Apeiron.CountHelpers.off_outer_step
#audit Apeiron.CountHelpers.word_count_offsets_spec_main
#audit Apeiron.CountHelpers.word_count_offsets_refuse_mul
#audit Apeiron.CountHelpers.word_count_offsets_refuse_len
#audit Apeiron.CountHelpers.word_count_offsets_spec

-- Count aggregate (Count/Aggregate.lean): the count table's representation and `count_aggregate`'s refinement.
#audit Apeiron.CountAggregate.pspec_admissible
#audit Apeiron.CountAggregate.loop_pc
#audit Apeiron.CountAggregate.probe_inner_spec
#audit Apeiron.CountAggregate.probe_tagged_spec
#audit Apeiron.CountAggregate.geom_insert
#audit Apeiron.CountAggregate.with_bits_spec
#audit Apeiron.CountAggregate.body_insert_fresh
#audit Apeiron.CountAggregate.body_update
#audit Apeiron.CountAggregate.grow_body_spec
#audit Apeiron.CountAggregate.grow_loop_spec
#audit Apeiron.CountAggregate.grow_spec
#audit Apeiron.CountAggregate.insert_with_spec
#audit Apeiron.CountAggregate.add_spec
#audit Apeiron.CountAggregate.freqRep_empty
#audit Apeiron.CountAggregate.add_preserves
#audit Apeiron.CountAggregate.count_body_spec
#audit Apeiron.CountAggregate.count_aggregate_refines_win
#audit Apeiron.CountAggregate.count_aggregate_refines
#audit Apeiron.CountAggregate.cntI_key
#audit Apeiron.CountAggregate.build_count
#audit Apeiron.CountAggregate.count_aggregate_canonical
#audit Apeiron.CountAggregate.home_group_spec
#audit Apeiron.CountAggregate.descend_group_spec
#audit Apeiron.CountAggregate.match_tag_spec
#audit Apeiron.CountAggregate.match_empty_spec
#audit Apeiron.CountAggregate.group_tag_mask_spec
#audit Apeiron.CountAggregate.movemask_spec
#audit Apeiron.CountAggregate.with_capacity_rep
#audit Apeiron.CountAggregate.freq_spec

-- Count face, capped and identity-exact (Count/{OverflowOps,FreqSideOv,FreqGrow,FreqInsert}.lean):
-- the overflow store under the count face, the side store, `place_capped`/`grow`, `insert_with`.
#audit Apeiron.CountAggregate.nd_nodup
#audit Apeiron.CountAggregate.in_order_descend
#audit Apeiron.CountAggregate.in_order_tail
#audit Apeiron.CountAggregate.in_order_spec
#audit Apeiron.CountAggregate.next_spec
#audit Apeiron.CountAggregate.into_entries_inner
#audit Apeiron.CountAggregate.into_entries_loop_spec
#audit Apeiron.CountAggregate.into_entries_spec
#audit Apeiron.CountAggregate.new_spec
#audit Apeiron.CountAggregate.oinv_node_update
#audit Apeiron.CountAggregate.nodeok_node_update
#audit Apeiron.CountAggregate.update_spec
#audit Apeiron.CountAggregate.keys_absent_of
#audit Apeiron.CountAggregate.keys_absent
#audit Apeiron.CountAggregate.insert_ovok
#audit Apeiron.CountAggregate.not_blocked_of_vacant
#audit Apeiron.CountAggregate.blocked_insert
#audit Apeiron.CountAggregate.empty_rep
#audit Apeiron.CountAggregate.find_spec
#audit Apeiron.CountAggregate.side_push_rep
#audit Apeiron.CountAggregate.side_update_rep
#audit Apeiron.CountAggregate.body_side_change
#audit Apeiron.CountAggregate.inOv_iff_keys
#audit Apeiron.CountAggregate.ov_start
#audit Apeiron.CountAggregate.body_ov_change
#audit Apeiron.CountAggregate.body_ov_insert
#audit Apeiron.CountAggregate.body_ov_update
#audit Apeiron.CountAggregate.place_blocked_tail
#audit Apeiron.CountAggregate.place_capped_spec
#audit Apeiron.CountAggregate.body_attach_side
#audit Apeiron.CountAggregate.entries_aux
#audit Apeiron.CountAggregate.grow_loop1_spec
#audit Apeiron.CountAggregate.side_upd_exec
#audit Apeiron.CountAggregate.side_live_exec
#audit Apeiron.CountAggregate.side_new_spec
#audit Apeiron.CountAggregate.side_upd_spec
#audit Apeiron.CountAggregate.pot_with_capacity
#audit Apeiron.CountAggregate.probe_outer_spec

-- Map face `prefix_masks` (Map/PrefixMasks.lean): the AVX2 branch equals the scalar branch.
#audit Apeiron.PrefixMasks.u64_ne_u8
#audit Apeiron.PrefixMasks.as_ptr_u64
#audit Apeiron.PrefixMasks.ofLeBytes_flat
#audit Apeiron.PrefixMasks.loop_pc
#audit Apeiron.PrefixMasks.tail_spec
#audit Apeiron.PrefixMasks.avx2_spec
#audit Apeiron.PrefixMasks.avx2_fail
#audit Apeiron.PrefixMasks.avxBranch_spec
#audit Apeiron.PrefixMasks.avxBranch_fail
#audit Apeiron.PrefixMasks.body_inv
#audit Apeiron.PrefixMasks.body_ex
#audit Apeiron.PrefixMasks.body_done
#audit Apeiron.PrefixMasks.loop_total
#audit Apeiron.PrefixMasks.scalar_pc
#audit Apeiron.PrefixMasks.scalar_ex
#audit Apeiron.PrefixMasks.spec_unique
#audit Apeiron.PrefixMasks.prefix_masks_branches_agree
#audit Apeiron.PrefixMasks.prefix_masks_probe_independent
#audit Apeiron.PrefixMasks.prefix_masks_spec

#audit apeiron_word.word_f16_split_nonfinite
#audit apeiron_word.word_f16_split_exact
#audit apeiron_word.word_f16_split_finite_witness
#audit apeiron_word.word_f16_split_nonfinite_witness

-- The executed map face (Map/Executed*.lean; `selectedReadRep_of_tableRep` is audited in Map/ExecutedSelected.lean, which imports `Apeiron.AuditClosed` through Map.Selected and cannot be imported here): `hashtrinity::Table<u64, i64, KS>`.
#audit Apeiron.MapExecuted.loop_pc
#audit Apeiron.MapExecuted.is_sentinel_ok
#audit Apeiron.MapExecuted.is_live_ok
#audit Apeiron.MapExecuted.stored_matches_ok
#audit Apeiron.MapExecuted.regen_matches_ok
#audit Apeiron.MapExecuted.probe_limit_val
#audit Apeiron.MapExecuted.desc_lin
#audit Apeiron.MapExecuted.desc_frame
#audit Apeiron.MapExecuted.probe_rest_spec
#audit Apeiron.MapExecuted.prefix_masks_loop_spec
#audit Apeiron.MapExecuted.prefix_masks_spec
#audit Apeiron.MapExecuted.u32_tz_spec
#audit Apeiron.MapExecuted.pcw_loop_spec
#audit Apeiron.MapExecuted.pcw_spec
#audit Apeiron.MapExecuted.pcc_loop_spec
#audit Apeiron.MapExecuted.pcc_spec
#audit Apeiron.MapExecuted.pck_spec
#audit Apeiron.MapExecuted.emb_unique
#audit Apeiron.MapExecuted.find_node_spec
#audit Apeiron.MapExecuted.bst_key_inj
#audit Apeiron.MapExecuted.ov_get_spec
#audit Apeiron.MapExecuted.side_find_spec
#audit Apeiron.MapExecuted.with_bits_rep
#audit Apeiron.MapExecuted.stored_with_bits_rep
#audit Apeiron.MapExecuted.new_rep
#audit Apeiron.MapExecuted.with_capacity_rep
#audit Apeiron.MapExecuted.coordinate_with_capacity_rep
#audit Apeiron.MapExecuted.vacant_not_live
#audit Apeiron.MapExecuted.blocked_not_main
#audit Apeiron.MapExecuted.blocked_some_spec
#audit Apeiron.MapExecuted.get_coord_spec
#audit Apeiron.MapExecuted.get_refines
#audit Apeiron.MapExecuted.coord_get_refines

-- The AVX2 premise of the read discharged
-- (`Map/ExecutedReadUncond.lean`) and the executed AVL write path (`Map/ExecutedOverflow{AvlBase,Avl,Chain,Descend,Insert}.lean`).
#audit Apeiron.MapExecuted.avx2_spec_holds
#audit Apeiron.MapExecuted.get_refines_uncond
#audit Apeiron.MapExecuted.coord_get_refines_uncond
#audit Apeiron.MapExecuted.h_spec
#audit Apeiron.MapExecuted.update_spec
#audit Apeiron.MapExecuted.rotate_right_spec
#audit Apeiron.MapExecuted.rotate_left_spec
#audit Apeiron.MapExecuted.rebalance_spec
#audit Apeiron.MapExecuted.rebalance_frT
#audit Apeiron.MapExecuted.chain_relink
#audit Apeiron.MapExecuted.rebalance_chain_loop_spec
#audit Apeiron.MapExecuted.ancOld_of_valid
#audit Apeiron.MapExecuted.insert_node_loop_spec
#audit Apeiron.MapExecuted.find_entry_loop_spec
#audit Apeiron.MapExecuted.find_entry_core
#audit Apeiron.MapExecuted.find_entry_none_iff
#audit Apeiron.MapExecuted.find_entry_spec
#audit Apeiron.MapExecuted.alloc_spec
#audit Apeiron.MapExecuted.insert_node_spec
#audit Apeiron.MapExecuted.insert_new_at_spec
#audit Apeiron.MapExecuted.insert_new_at_spec_fe

-- The executed insert refines `Apeiron.Table.insertWith`
-- (`Map/ExecutedInsertSide.lean`, `Map/ExecutedInsertSlot.lean`, `Map/ExecutedInsert.lean`,
-- `Map/ExecutedResizeOrder.lean`, `Map/ExecutedResize.lean`).
#audit Apeiron.MapExecuted.rep_side_update
#audit Apeiron.MapExecuted.push_spec
#audit Apeiron.MapExecuted.insert_sent_rep
#audit Apeiron.MapExecuted.insert_side_refines
#audit Apeiron.MapExecuted.coord_insert_side_refines
#audit Apeiron.MapExecuted.rep_hit
#audit Apeiron.MapExecuted.rep_fill
#audit Apeiron.MapExecuted.rep_set_ov
#audit Apeiron.MapExecuted.ov_full_of_some
#audit Apeiron.MapExecuted.ov_full_new
#audit Apeiron.MapExecuted.blocked_new_full_abs
#audit Apeiron.MapExecuted.blocked_new_full
#audit Apeiron.MapExecuted.blocked_hit_full
#audit Apeiron.MapExecuted.grow_rep
#audit Apeiron.MapExecuted.insert_ns_rep
#audit Apeiron.MapExecuted.skc_set
#audit Apeiron.MapExecuted.insert_preserves_of_resize
#audit Apeiron.MapExecuted.coord_insert_preserves_of_resize
#audit Apeiron.MapExecuted.insert_refines_of_resize
#audit Apeiron.MapExecuted.coord_insert_refines_of_resize
#audit Apeiron.MapExecuted.push_spine_spec
#audit Apeiron.MapExecuted.in_order_spec
#audit Apeiron.MapExecuted.entries_loop_single
#audit Apeiron.MapExecuted.into_entries_spec
#audit Apeiron.MapExecuted.place_rep
#audit Apeiron.MapExecuted.resize_loop0_spec
#audit Apeiron.MapExecuted.resize_loop1_spec
#audit Apeiron.MapExecuted.attach_side
#audit Apeiron.MapExecuted.resize_rep
#audit Apeiron.MapExecuted.stored_take_key
#audit Apeiron.MapExecuted.regen_take_key
#audit Apeiron.MapExecuted.resizeOK_stored
#audit Apeiron.MapExecuted.resizeOK_regen
#audit Apeiron.MapExecuted.insert_preserves
#audit Apeiron.MapExecuted.coord_insert_preserves
#audit Apeiron.MapExecuted.insert_refines
#audit Apeiron.MapExecuted.coord_insert_refines
#audit Apeiron.MapExecuted.with_bits_wf
#audit Apeiron.MapExecuted.new_wf
#audit Apeiron.MapExecuted.with_capacity_wf
#audit Apeiron.MapExecuted.coordinate_with_capacity_wf

-- The executed build refines
-- `Apeiron.Table.build` (`Map/ExecutedBuild.lean`).
#audit Apeiron.MapExecuted.countP_set_le
#audit Apeiron.MapExecuted.map_get_refines
#audit Apeiron.MapExecuted.coord_map_get_refines
#audit Apeiron.MapExecuted.map_insert_refines
#audit Apeiron.MapExecuted.coord_map_insert_refines
#audit Apeiron.MapExecuted.new_mapRep
#audit Apeiron.MapExecuted.with_capacity_mapRep
#audit Apeiron.MapExecuted.coordinate_with_capacity_mapRep
#audit Apeiron.MapExecuted.fold_refines
#audit Apeiron.MapExecuted.coord_fold_refines
#audit Apeiron.MapExecuted.build_refines
#audit Apeiron.MapExecuted.coord_build_refines
#audit Apeiron.MapExecuted.build_get_refines
#audit Apeiron.MapExecuted.coord_build_get_refines
#audit Apeiron.MapExecuted.build_witness_exec
#audit Apeiron.MapExecuted.build_refines_witness

-- One decode and one
-- split lemma indexed by the bit width b (`Spirv/Carrier.lean`).
#audit Spirv.carrier_decode
#audit Spirv.carrier_split
#audit Spirv.carrier_split_cells

-- The roofline lower
-- bound, the move speedup bound and the two fusion results over the resource carrier
-- (`Perf/Roofline.lean`).
#audit Perf.timeLB_mono
#audit Perf.move_speedup_le
#audit Perf.move_speedup_le_binding
#audit Perf.fusion_reduces_latency
#audit Perf.fanout_forces_cost
#audit Perf.scalar_not_sufficient

-- Latency hiding on the
-- carrier (`Perf/Concurrency.lean`) and the width, walk-count and cache-line moves as instances of
-- the move bound (`Perf/Instances.lean`).
#audit Perf.timeLBc_mono_conc
#audit Perf.latency_hidden
#audit Perf.latency_bound_below_hidden
#audit Perf.fusion_hides_the_coreduction
#audit Perf.fused_strictly_beats_isolated
#audit Perf.instrWidth_speedup_le
#audit Perf.width_no_DemandRatioBound
#audit Perf.walk_demandRatio
#audit Perf.walk_speedup
#audit Perf.walk_mix
#audit Perf.map01_line_control
#audit Perf.map01_move_speedup
-- Dependent dispatch chain and its knee (Perf/Chain.lean)
#audit Perf.Chain.chain_ge_bytes
#audit Perf.Chain.knee
#audit Perf.Chain.dispLB_of_le_knee
#audit Perf.Chain.chain_tax
#audit Perf.Chain.chain_tax_zero_of_knee
#audit Perf.Chain.fuse_le
#audit Perf.Chain.fuse_saves_L
#audit Perf.Chain.merge_independent
#audit Perf.Chain.merge_saves
#audit Perf.Chain.chain_fuse_ratio
#audit Perf.Chain.chain_fuse_pair_saving

-- The shared issue-port
-- cut, the slot move gain and the lever ceiling (`Perf/Issue.lean`).
#audit Perf.shared_ge_timeLB
#audit Perf.shared_strict
#audit Perf.slot_move_gain
#audit Perf.slot_move_gain_sub
#audit Perf.lever_ceiling
#audit Perf.upload_convs_token_free
#audit Perf.loopRes_sharedLB
#audit Perf.loopRes_issue_strict

-- Resident workgroups from
-- the register file and shared memory, the occupancy gain bound and ceiling (`Perf/Occupancy.lean`).
#audit Perf.residentWG_antitone
#audit Perf.residentWG_step
#audit Perf.residentWG_const
#audit Perf.residentWG_default
#audit Perf.residentWG_t128x32
#audit Perf.residentMin_le_left
#audit Perf.residentMin_le_right
#audit Perf.occupancy_gain_le
#audit Perf.occupancy_ceiling
#audit Perf.residentMin_occ48k
#audit Perf.residentMin_committed
#audit Perf.Moves.p15_saves_L
#audit Perf.Moves.p15_measured
#audit Perf.Moves.p3_unfuse_adds_L
#audit Perf.Moves.p3_ratio_below_one
#audit Perf.Moves.prologue_depth_gain
#audit Perf.Moves.prologue_gain_cap
#audit Perf.Moves.prologue_depth_gain_p4
#audit Perf.Moves.q5_row_words_two_thirds
#audit Perf.Moves.q5_chart_gain_when_bytes_bind
#audit Perf.Moves.q5_chart_under_tile_threshold
#audit Perf.Moves.q5_coopmat_dot_under_magic
#audit Perf.Moves.q6_row_words_seven_ninths
#audit Perf.Moves.q5_spread_exact
#audit Perf.Moves.q6_signed_byte_exact
#audit Perf.Moves.q6_coopmat_half_split
#audit Perf.Moves.p47_gu_coop_floor
#audit Perf.Moves.p48_flash_floor
#audit Perf.Moves.p49_resid_once_floor
#audit Perf.Moves.p50_selection_bytes
#audit Perf.Moves.p51_sel_install_band
#audit Perf.Moves.p53_ropeflash_floor
#audit Perf.Moves.p54_ropeflash_install_band
#audit Perf.Moves.p52_amax_tail
#audit Perf.Moves.p55_amaxp_install_floor
#audit Perf.Moves.p56_upload_inventory
#audit Perf.Moves.p57_dn_byte_difference
#audit Perf.Moves.p58_refutation_arithmetic
#audit Perf.Moves.p60_overhead_arithmetic
#audit Perf.Moves.p61_stamp_bias_arithmetic
#audit Perf.Moves.p62_gap_attribution_bounds
#audit Perf.Moves.p63_drain_ceiling
#audit Perf.Moves.p64_pdl_edge_install
#audit Perf.Moves.p65_edge_extension
#audit Perf.Moves.p66_pdl3_default
#audit Perf.Moves.p69_flash_merged_prologue
#audit Perf.Moves.p68_kslice_not_confirmed
#audit Perf.Moves.p67_qf_target_id_bug
#audit Perf.Moves.p70_gu_cost_terms
#audit Perf.Moves.p71_gu_workgroups
#audit Perf.Moves.p72_last_reduce_terms
#audit Perf.Moves.p73_transport_arms
#audit Perf.Moves.p10_mma_floor
#audit Perf.Moves.p10_mma_rate_ratio
#audit Perf.Moves.p10_mma_floor_p8
#audit Perf.Moves.fieldMax_singleField
#audit Perf.Moves.timeLB_singleField
#audit Perf.Moves.non_mma_move_ceiling
#audit Perf.Moves.mma_issue_ratio
#audit Perf.Moves.ceilDiv_le_mul_ceilDiv
#audit Perf.Moves.mma_issue_bound
#audit Perf.Moves.hd128_tile_exceeds
#audit Perf.MapMoves.map08_flatSpillRead
#audit Perf.MapMoves.map02_hashedGetCoord
#audit Perf.MapMoves.map07_subsumedParity
#audit Perf.MapMoves.count01_bucketTagDisjoint
#audit Perf.MapMoves.count02_homeGroupFastPath
#audit Perf.MapMoves.count03_readPathProduct
#audit Perf.CountLookup.overlap_gain
#audit Perf.CountLookup.slack_of_partial_gain
#audit Perf.CountLookup.lever_closes_only_if
#audit Perf.CountLookup.tag_collisions_sum
#audit Perf.CountLookup.p74_clause2_refuted
#audit Perf.CountLookup.p74_walk_cannot_close
#audit Perf.CountLookup.p75_refuted
#audit Perf.CountLookup.p76_clause1_refuted
#audit Perf.CountLookup.p76_clause2
#audit Perf.CountLookup.p76_clause34
#audit Perf.CountLookup.p77_clause1
#audit Perf.CountLookup.p77_clause2
#audit Perf.CountLookup.p77_clause3
#audit Perf.CountLookup.p77_clause4
#audit Perf.CountLookup.lever_gain_le
#audit Perf.CountLookup.lever_cannot_close
#audit Perf.CountLookup.p77_uop_difference_under_half_gap
#audit Perf.CountLookup.p77_retired_uop_lever_cannot_close
#audit Perf.CountLookup.p78_clause1_refuted
#audit Perf.CountLookup.p78_no_off_home
#audit Perf.CountLookup.p78_excess_less_loaded
#audit Perf.CountLookup.p79_clause1
#audit Perf.CountLookup.p79_clause2
#audit Perf.CountLookup.p79_clause3
#audit Perf.CountLookup.p79_clause4
#audit Perf.CountLookup.p79_pairs_only
#audit Perf.CountLookup.p80_clause1_refuted
#audit Perf.CountLookup.p80_clause2
#audit Perf.CountLookup.p80_clause3_refuted
#audit Perf.CountLookup.p80_split_by_pool
#audit Perf.CountLookup.p80_large_cells_match_par
#audit Perf.HattnT1.p1_holds
#audit Perf.HattnT1.p2_refuted
#audit Perf.HattnT1.p3_refuted
#audit Perf.HattnT1.p3_no_cell
#audit Perf.HattnT1.p4_refuted
#audit Perf.HattnT3.ph1_refuted
#audit Perf.HattnT3.ph1_median_sublinear
#audit Perf.HattnT3.keps_32k_lt_half_percent
#audit Perf.HattnT3.ph2_refuted
#audit Perf.HattnT3.ph2_second_cumulant_worse
#audit Perf.HattnT3.ph2_g1_near_trunc
#audit Perf.HattnT3.ph3_refuted
#audit Perf.HattnT3.ph3_screening_dominates
#audit Perf.HattnT3.ph4_refuted
#audit Perf.HattnT3.ph4_high_freq_random
#audit Perf.HattnT3.first_cumulant_le
#audit Perf.HattnT3.second_cumulant_overshoots
#audit Perf.HattnT4.pq1_held
#audit Perf.HattnT4.pq2_held
#audit Perf.HattnT4.pq3_refuted
#audit Perf.HattnT4.pq3_slow_bound_worse
#audit Perf.HattnT4.pq4_refuted
#audit Perf.HattnT4.pq5_refuted
#audit Perf.HattnT4.pq5_sublinear
#audit Perf.HattnT4.pq6_refuted
#audit Perf.HattnT4.bound_looseness
#audit Perf.HattnT4.metadata_dominates
#audit Perf.HattnT4.pages64_best_real
#audit Perf.HattnT4.quest_bound_sound
#audit Perf.HattnT5.pt1_held
#audit Perf.HattnT5.pt2_held
#audit Perf.HattnT5.pt3_held
#audit Perf.HattnT5.pt4_held
#audit Perf.HattnT5.cert_exceeds_dense
#audit Perf.HattnT5.pt5_refuted
#audit Perf.HattnT5.pt5_holds_at_8k
#audit Perf.HattnT5.pt6_held
#audit Perf.HattnT5.pt7_held
#audit Perf.HattnT5.compounding_attenuates
#audit Perf.HattnT5.cert_stop_sound
#audit Perf.HattnT6.ps1_held
#audit Perf.HattnT6.ps2_held
#audit Perf.HattnT6.ps3_refuted
#audit Perf.HattnT6.ps4_held
#audit Perf.HattnT6.ps5_refuted
#audit Perf.HattnT6.ps6_refuted
#audit Perf.HattnT6.fix1024_cheapest
#audit Perf.HattnT6.fix1024_fourteen_x
#audit Perf.HattnT6.bounds_dominate
#audit Perf.HattnT6.attenuation_weak_at_quarter
#audit Perf.HattnT7.singles_agree
#audit Perf.HattnT7.pg1_held
#audit Perf.HattnT7.pg2_held
#audit Perf.HattnT7.pg3_refuted
#audit Perf.HattnT7.pg4_held
#audit Perf.HattnT7.pg5_held
#audit Perf.HattnT7.pg6_held
#audit Perf.HattnT7.gqa_correction
#audit Perf.HattnT7.grp2048r8_thirteen_x
#audit Perf.HattnT7.oracle_gap
#audit Perf.HattnT8.singles_agree
#audit Perf.HattnT8.pt1_held
#audit Perf.HattnT8.pt2_held
#audit Perf.HattnT8.pt3_held
#audit Perf.HattnT8.pt4_held
#audit Perf.HattnT8.pt5_refuted
#audit Perf.HattnT8.clu4_cuts_scan
#audit Perf.HattnT8.budget_slack
#audit Perf.HattnT8.climb_log
#audit Perf.HattnT8.model_bounds_32k
#audit Perf.HattnT8.pt6_held
#audit Perf.HattnT8Q7.singles_agree
#audit Perf.HattnT8Q7.p71_held
#audit Perf.HattnT8Q7.p72_held
#audit Perf.HattnT8Q7.p73_held
#audit Perf.HattnT8Q7.p74_held
#audit Perf.HattnT8Q7.p75_held
#audit Perf.HattnT8Q7.p76_held
#audit Perf.HattnT8Q7.all_arms_pass
#audit Perf.HattnT8Q7.flat_below_half_b
#audit Perf.HattnT8Q7.recall_falls_kld_holds
#audit Perf.HattnT8Q7.scan_cut_7b
#audit Perf.HattnT8Q7L.q1_control_held
#audit Perf.HattnT8Q7L.q1_mut_threshold_refuted
#audit Perf.HattnT8Q7L.q2_split
#audit Perf.HattnT8Q7L.q3_refuted
#audit Perf.HattnT8Q7L.q4_held
#audit Perf.HattnT8Q7L.model_bounds_64k
#audit Perf.HattnT8Q7L.q5_held
#audit Perf.HattnT8Q7L.q6_held
#audit Perf.HattnT8Q7L.tree_fails_cluster_holds
#audit Perf.HattnT8Q8.q1_split
#audit Perf.HattnT8Q8.q2_held
#audit Perf.HattnT8Q8.q3_refuted
#audit Perf.HattnT8Q8.q4_held
#audit Perf.HattnT8Q8.q5_held
#audit Perf.HattnT8Q8.q6_held
#audit Perf.HattnT8Q8.all_pass_262144
#audit Perf.HattnT8Q8.cluster_scan_scaling
#audit Perf.HattnT8Q8.bytes_7b_256k
#audit Perf.HattnT8Q7L.cluster_cuts_scan_128k
#audit Perf.HattnT8Q7L.flat_drift
#audit Perf.HattnT8Q7L.bytes_7b_128k
#audit Perf.HattnT8Q9.q1_split
#audit Perf.HattnT8Q9.q2_split
#audit Perf.HattnT8Q9.q3_held
#audit Perf.HattnT8Q9.q4_held
#audit Perf.HattnT8Q9.model_bounds_512k
#audit Perf.HattnT8Q9.q5_held
#audit Perf.HattnT8Q9.q6_held
#audit Perf.HattnT8Q9.q7_held
#audit Perf.HattnT8Q9.budget_cliff
#audit Perf.HattnT8Q9.global_failure_1m
#audit Perf.HattnT8Q9.clu4_tracks_flat
#audit Perf.HattnT8Q9.bytes_7b_1m_measured
#audit Perf.HattnT8Farb.f1_refuted
#audit Perf.HattnT8Farb.f2_refuted
#audit Perf.HattnT8Farb.f3_refuted
#audit Perf.HattnT8Farb.f4_held
#audit Perf.HattnT8Farb.doublings_to_dense
#audit Perf.HattnT8Farb.bytes_double
#audit Perf.HattnT8Sum.s1_refuted
#audit Perf.HattnT8Sum.s2_refuted
#audit Perf.HattnT8Sum.s3_held
#audit Perf.HattnT8Sum.s4_held
#audit Perf.HattnT8Sum.mass_not_bottleneck
#audit Perf.HattnT8Sum.summary_gain_and_loss
#audit Perf.HattnT8SumU.u1_held
#audit Perf.HattnT8SumU.u2_held
#audit Perf.HattnT8SumU.u3_refuted
#audit Perf.HattnT8SumU.u4_held
#audit Perf.HattnT8SumU.no_unit_passes
#audit Perf.HattnT8Layer.l1_refuted
#audit Perf.HattnT8Layer.l2_held
#audit Perf.HattnT8Layer.l3_refuted
#audit Perf.HattnT8Layer.l4_held
#audit Perf.HattnT8Layer.early_layers_dominate
#audit Perf.HattnT8Layer.quarters_overshoot
#audit Perf.HattnT8Layer.m1_refuted
#audit Perf.HattnT8Layer.m2_refuted
#audit Perf.HattnT8Layer.m3_held
#audit Perf.HattnT8Layer.m4_held
#audit Perf.HattnT8Layer.m5_refuted
#audit Perf.HattnT8Layer.compounding
#audit Perf.HattnT8LFarb.n1_held
#audit Perf.HattnT8LFarb.n2_held
#audit Perf.HattnT8LFarb.n3_held
#audit Perf.HattnT8LFarb.n4_held
#audit Perf.HattnT8LFarb.n5_held
#audit Perf.HattnT8LFarb.c1_held
#audit Perf.HattnT8LFarb.nonmonotone
#audit Perf.HattnT8Code.k1_held
#audit Perf.HattnT8Code.k2_held
#audit Perf.HattnT8Code.k3_held
#audit Perf.HattnT8Code.k4_held
#audit Perf.HattnT8Code.between
#audit Perf.HattnT8Code.heavy_tail
#audit Perf.HattnT8Code.code_vs_prose
#audit Perf.HattnT8.bytes_7b_1m

-- The SIMT shared-load
-- dominance, the staging halving rule, the coopmat node comparison and the pp512 gain
-- (`Perf/AttnCoop.lean`).
#audit Perf.simt_issue_bound_dominates
#audit Perf.staging_halves
#audit Perf.coop_stage_halves
#audit Perf.coop_node_below_simt
#audit Perf.attn_node_gain

-- The identity floor under a relative-precision
-- rounding (`Perf/IdentityFloat.lean`): every one of the five roundings is deletion-necessary.
#audit Perf.Float.rne_mono
#audit Perf.Float.fl_mono
#audit Perf.Float.fl_idem
#audit Perf.Float.drop1_changes
#audit Perf.Float.drop2_changes
#audit Perf.Float.drop3_changes
#audit Perf.Float.drop4_changes
#audit Perf.Float.drop5_changes
#audit Perf.Float.identity_floor

-- The two-phase
-- overlap model and the P8/P9 instance (`Perf/Overlap.lean`).
#audit Perf.ceilDiv_add_of_dvd
#audit Perf.maxVec_add_le
#audit Perf.fieldMax_add
#audit Perf.overlap_le_seq
#audit Perf.overlap_ge_max
#audit Perf.overlap_no_gain_of_shared_binding
#audit Perf.overlap_full_of_disjoint
#audit Perf.p8_disjoint
#audit Perf.p8_overlap_full
#audit Perf.p8_seq
#audit Perf.p8_overlap_predicted_gain
#audit Perf.residual_not_in_fields

-- The fixed-grid
-- rounding model, the node-deletion family and its identity floor (`Perf/Identity.lean`).
#audit Perf.rnd_idem
#audit Perf.rnd_mono
#audit Perf.drop1_changes
#audit Perf.drop4_changes
#audit Perf.drop5_changes
#audit Perf.redundant_two
#audit Perf.redundant_three
#audit Perf.identity_floor

-- The extracted
-- exact-integer Gram fold `gram.gram_fold_i8` refines the integer Gram sum (`Learn/GramExact.lean`).
#audit apeiron_gram.row_pair_loop_spec
#audit apeiron_gram.row_loop_spec
#audit apeiron_gram.rows_loop_spec
#audit apeiron_gram.gram_fold_i8_ok
#audit apeiron_gram.gram_fold_i8_refusal
#audit apeiron_gram.gram_fold_i8_entry
#audit apeiron_gram.gram_fold_i8_stack

-- The extracted E8 codebook `learn_port.e8` refines integer specifications (`Learn/E8Code.lean`).
#audit Learn.E8Code.popcount_parity_spec
#audit Learn.E8Code.spinor_row_spec
#audit Learn.E8Code.spinor_key_parity
#audit Learn.E8Code.e8_codeword_spec
#audit Learn.E8Code.e8_codeword_spin8
#audit Learn.E8Code.e8_nearest_spec
#audit Learn.E8Code.witness_codeword_ok
#audit Learn.E8Code.witness_codeword_err
#audit Learn.E8Code.witness_nonpositive_scale
#audit Learn.E8Code.witness_nearest_ok
#audit Learn.E8Code.witness_overflow

-- The E8 `∪` O8 ICM block decision (`Learn/FitIcm.lean`): the argmin specification, the codebook
-- shape, and the sweep non-increase. The extracted body's own audit is commented below because its
-- externals are the pool and `Option`/`FirstErr` boundaries; `Learn.FitIcm.icm_step_spec_of_body_argmin`
-- is the conditional closure.
#audit Learn.FitIcm.icmStep_spec
#audit Learn.FitIcm.spec_cost_eq_argmin
#audit Learn.FitIcm.spec_quad_le
#audit Learn.FitIcm.spec_sweep_le
#audit Learn.FitIcm.spec_sweep_lt
#audit Learn.FitIcm.blockQuad_eq_blockReduce_add
#audit Learn.FitIcm.blockReduce_eq_word
#audit Learn.FitIcm.pairStep_spec
#audit Learn.FitIcm.joint_pair_le_slice_left
#audit Learn.FitIcm.joint_pair_le_slice_right
#audit Learn.FitIcm.pair_codebook_rate
#audit Learn.FitIcm.codebook_row_shape
#audit Learn.FitIcm.icmStep_row_shape
#audit Learn.FitIcm.codebook_card
#audit Learn.FitIcm.icm_step_spec_of_body_argmin
#audit Learn.FitIcm.icmStep_witness
#audit Learn.FitIcm.block_cost_ok
#audit Learn.FitIcm.block_cost_total
#audit Learn.FitIcm.best_code_q_loop_spec
#audit Learn.FitIcm.best_code_q_argmin
#audit Learn.FitIcm.best_code_q_argmin_real
#audit Learn.FitIcm.icm_step_spec_of_extracted_argmin
#audit apeiron_word.fit_port.word_fit_icm_best_code_q
-- The extracted Q32.32 `i64` linear forward and its two adjoints
-- (`learn_port.dense.linear_fwd` / `linear_bwd_input` / `linear_bwd_weight`) refine the independent
-- `ℤ`/`List ℤ` model (`Learn/Grad/DenseLinear.lean`); the branch witnesses are in
-- `Learn/Grad/DenseLinearWitness.lean`.
#audit apeiron_word.dense_expect_len_ok
#audit apeiron_word.dense_expect_len_empty
#audit apeiron_word.dense_expect_len_shape
#audit apeiron_word.dense_expect_len_overflow
#audit apeiron_word.linear_fwd_acc_spec
#audit apeiron_word.linear_fwd_ok
#audit apeiron_word.linear_bwd_input_acc_spec
#audit apeiron_word.linear_bwd_input_ok
#audit apeiron_word.linear_bwd_weight_acc_spec
#audit apeiron_word.linear_bwd_weight_ok
#audit apeiron_word.dense_expect_len_ne_range
#audit apeiron_word.linear_fwd_witness_ok
#audit apeiron_word.linear_fwd_witness_empty
#audit apeiron_word.linear_fwd_witness_shape
#audit apeiron_word.linear_fwd_witness_overflow
#audit apeiron_word.linear_bwd_input_witness_ok
#audit apeiron_word.linear_bwd_input_witness_empty
#audit apeiron_word.linear_bwd_input_witness_shape
#audit apeiron_word.linear_bwd_input_witness_overflow
#audit apeiron_word.linear_bwd_weight_witness_ok
#audit apeiron_word.linear_bwd_weight_witness_empty
#audit apeiron_word.linear_bwd_weight_witness_shape
#audit apeiron_word.linear_bwd_weight_witness_overflow
-- The extracted RoPE passes `learn_port.attn.rope_fwd`/`rope_bwd` (`src/learn/grad/attn.rs`) refine
-- the `rotHeadsState` head content and the `ropeBwdPrefix` transpose (`Learn/Grad/Rope.lean`).
#audit apeiron_rope.rope_elem_bwd_spec_lo
#audit apeiron_rope.rope_elem_bwd_spec_hi
#audit apeiron_rope.rope_bwd_loop_spec
#audit apeiron_rope.rope_bwd_refines
#audit apeiron_rope.rot_one_loop_spec
#audit apeiron_rope.rotate_one_spec
#audit apeiron_rope.rope_fwd_refines
#audit apeiron_rope.rope_fwd_witness
#audit apeiron_rope.rope_bwd_witness
-- The extracted vocabulary-head exact-integer bodies `learn_port.head` refine their `ℤ` specs
-- (`Learn/Grad/Head.lean`): rounding, narrowing, sign, checked arithmetic, the `log p` entry and the
-- `log_softmax_row` projection.
#audit apeiron_head.abs128_spec
#audit apeiron_head.sgn_spec
#audit apeiron_head.to64_ok
#audit apeiron_head.to64_low
#audit apeiron_head.to64_high
#audit apeiron_head.add128_ok
#audit apeiron_head.add128_refuse
#audit apeiron_head.mul128_ok
#audit apeiron_head.mul128_refuse
#audit apeiron_head.dim128_spec
#audit apeiron_head.mul_dims_ok
#audit apeiron_head.mul_dims_refuse
#audit apeiron_head.lp_entry_ok
#audit apeiron_head.round_div_spec
#audit apeiron_head.log_softmax_row_spec
#audit apeiron_head.log_softmax_row_err
#audit apeiron_head.to64_witness_ok
#audit apeiron_head.add128_witness_ok
#audit apeiron_head.mul128_witness_ok
#audit apeiron_head.lp_entry_witness_ok

-- The extracted `learn_port.head` `_loop` bodies refine their `ℤ` accumulator models
-- (`Learn/Grad/HeadLoop.lean`), each over `loop.spec_decr_nat` with an explicit non-overflow domain.
#audit apeiron_head.row_max_loop_body_spec
#audit apeiron_head.row_max_loop_spec
#audit apeiron_head.kl_acc_loop_body_spec
#audit apeiron_head.kl_acc_loop_spec
#audit apeiron_head.dz_diff_loop_body_spec
#audit apeiron_head.dz_diff_loop_spec
#audit apeiron_head.dot_round_loop_body_spec
#audit apeiron_head.dot_round_loop_spec
#audit apeiron_head.dw_fill_loop_body_spec
#audit apeiron_head.dw_fill_loop_spec
#audit apeiron_head.dw_out_fill_loop_body_spec
#audit apeiron_head.dw_out_fill_loop_spec
#audit apeiron_head.tv_scan_loop_body_spec
#audit apeiron_head.tv_scan_loop_spec

-- The remaining extracted `learn_port.head` `_loop` bodies and their public entry points
-- (`Learn/Grad/HeadRows.lean`): the `W_head` backward (`dh_entry`/`dh_fill`) and the TV gradient
-- (`tv_dz`), each over `loop.spec_decr_nat` with an explicit non-overflow domain.
#audit apeiron_head.dhAux_add
#audit apeiron_head.dhAux_succ
#audit apeiron_head.dhVals_succ
#audit apeiron_head.dhVals_length
#audit apeiron_head.dh_entry_loop_body_spec
#audit apeiron_head.dh_entry_loop_spec
#audit apeiron_head.dh_entry_spec
#audit apeiron_head.dh_fill_loop_body_spec
#audit apeiron_head.dh_fill_loop_spec
#audit apeiron_head.dh_fill_spec
#audit apeiron_head.tvDzNum_eq
#audit apeiron_head.tvDzVals_succ
#audit apeiron_head.tvDzVals_length
#audit apeiron_head.tv_dz_loop_body_spec
#audit apeiron_head.tv_dz_loop_spec
#audit apeiron_head.tv_dz_spec

-- The student logits (`logits_row`): the checked dot/round helper, the row model, the `Vec.index_mut`
-- write and the `v < k` walk, each over `loop.spec_decr_nat` with an explicit non-overflow domain.
#audit apeiron_head.dot_round_spec
#audit apeiron_head.logitsVals_succ
#audit apeiron_head.logitsVals_length
#audit apeiron_head.vvals_length
#audit apeiron_head.vec_index_mut_spec
#audit apeiron_head.logits_row_loop_body_spec
#audit apeiron_head.logits_row_loop_spec
#audit apeiron_head.logits_row_spec

-- The softmax numerator fill (`exp_fill`): the flush threshold and the four `exp_entry` arms.
#audit apeiron_head.head_FLUSH_eq
#audit apeiron_head.flush_neg_val
#audit apeiron_head.exp_entry_flush
#audit apeiron_head.exp_entry_overflow
#audit apeiron_head.exp_entry_exp_err
#audit apeiron_head.exp_entry_ok
#audit apeiron_head.expSum_succ
#audit apeiron_head.exp_fill_loop_body_spec
#audit apeiron_head.exp_fill_loop_spec
#audit apeiron_head.exp_fill_spec
#audit apeiron_head.fillInv_set
#audit apeiron_head.fill_lp_p_loop_body_spec
#audit apeiron_head.fill_lp_p_loop_spec
#audit apeiron_head.fill_lp_p_spec
#audit apeiron_head.rows_loop_loop_body_spec
#audit apeiron_head.rows_loop_loop_spec
#audit apeiron_head.rows_loop_spec
#audit apeiron_head.kl_acc_spec
#audit apeiron_head.dz_diff_spec
#audit apeiron_head.tv_scan_spec
#audit apeiron_head.kl_row_shape
#audit apeiron_head.tv_row_shape
#audit apeiron_head.kl_row_spec
#audit apeiron_head.tv_row_spec
#audit apeiron_head.dw_out_fill_spec
#audit apeiron_head.head_chunked_shape_n
#audit apeiron_head.head_chunked_shape_d
#audit apeiron_head.head_chunked_shape_v
#audit apeiron_head.head_chunked_shape_chunk
#audit apeiron_head.head_chunked_spec
#audit apeiron_head.dot_round_overflow
#audit apeiron_head.dot_round_overflow_witness
#audit apeiron_head.head_chunked_shape_witness
#audit apeiron_head.head_chunked_domain_witness

-- The extracted checked-`i64` sigma-delta body `sigma_delta.sd_binary_block`
-- refines `Numeric.SigmaDelta.sdState`/`sdSign` (`Numeric/OneBit/SigmaDeltaRefinement.lean`).
#audit apeiron_word.sd_step_spec
#audit apeiron_word.sd_loop_spec
#audit apeiron_word.sd_binary_block_shape
#audit apeiron_word.sd_binary_block_spec
#audit apeiron_word.sd_binary_block_witness_some
#audit apeiron_word.sd_binary_block_witness_none
#audit apeiron_word.sd_binary_block_witness
#audit Perf.HattnT8Tail.population
#audit Perf.HattnT8Tail.tp1_held
#audit Perf.HattnT8Tail.tp2_refuted
#audit Perf.HattnT8Tail.tp3_refuted
#audit Perf.HattnT8Tail.tp4_refuted
#audit Perf.HattnT8Tail.tp5_refuted
#audit Perf.HattnT8Tail.tp6_refuted
#audit Perf.HattnT8Tail.tp7_held
#audit Perf.HattnT8Tail.tp8_held
#audit Perf.HattnT8Tail.tp9_held
#audit Perf.HattnT8Tail.unflagged_mean
#audit Perf.HattnT8Tail.tp10_held
#audit Perf.HattnT8Tail.tp11_held
#audit Perf.HattnT8Tail.tp12_held
#audit Perf.HattnT8Tail.failures_are_uncertain
#audit Perf.HattnT8Tail.flat8192_passes_on_code
#audit Perf.HattnT8Tail.hybrid8192_cheaper
#audit Perf.HattnT8TailProse.tq1_held
#audit Perf.HattnT8TailProse.population
#audit Perf.HattnT8TailProse.tq2_held_degenerate
#audit Perf.HattnT8TailProse.tq3_held_unselective
#audit Perf.HattnT8TailProse.tq4_held
#audit Perf.HattnT8TailProse.tq5_refuted
#audit Perf.HattnT8TailProse.budget_monotone_not_enough
#audit Perf.HattnT8LGroup.lg1_held
#audit Perf.HattnT8LGroup.lg2_held
#audit Perf.HattnT8LGroup.lg3_held
#audit Perf.HattnT8LGroup.lg4_held
#audit Perf.HattnT8LGroup.group_share
#audit Perf.HattnT8Oracle.or1_refuted
#audit Perf.HattnT8Oracle.or2_held
#audit Perf.HattnT8Oracle.or3_held
#audit Perf.HattnT8Oracle.or4_held
#audit Perf.HattnT8Oracle.or5_held
#audit Perf.HattnT8Oracle.selection_beats_budget
#audit Perf.HattnT8Oracle.union_cost
#audit Perf.HattnT8Oracle.oracle_dominates_flat
#audit Perf.HattnT8Rerank.rr1_refuted
#audit Perf.HattnT8Rerank.rr2_refuted
#audit Perf.HattnT8Rerank.rr3_refuted
#audit Perf.HattnT8Rerank.rr4_held
#audit Perf.HattnT8Rerank.rr5_refuted
#audit Perf.HattnT8Rerank.pool_limit
#audit Perf.HattnT8Rerank.bytes_under_oracle
#audit Perf.HattnT8LowBit.control_held
#audit Perf.HattnT8LowBit.lb1_held
#audit Perf.HattnT8LowBit.lb2_refuted
#audit Perf.HattnT8LowBit.lb3_held
#audit Perf.HattnT8LowBit.lb4_refuted
#audit Perf.HattnT8LowBit.lb5_refuted
#audit Perf.HattnT8LowBit.lb6_held
#audit Perf.HattnT8LowBit.four_bit_matches_oracle
#audit Perf.HattnT8LowBit.one_bit_halves_flat
#audit Perf.HattnT8LbPool.lp1_refuted
#audit Perf.HattnT8LbPool.lp2_refuted
#audit Perf.HattnT8LbPool.lp3_refuted
#audit Perf.HattnT8LbPool.lp4_held
#audit Perf.HattnT8LbPool.lp5_held
#audit Perf.HattnT8LbPool.sign_pool_misses
#audit Perf.HattnT8LbPool.pool_reads_exceed_target
#audit Perf.HattnT8Cells.f1a_unmeasured
#audit Perf.HattnT8Cells.f1b_refuted
#audit Perf.HattnT8Cells.f1c_ratio_refuted_524k
#audit Perf.HattnT8Cells.f1c_ratio_refuted_1m
#audit Perf.HattnT8Cells.f1c_alpha_above_one
#audit Perf.HattnT8Cells.f1_control_held
#audit Perf.HattnT8Cells.f2a_held
#audit Perf.HattnT8Cells.f2b_lower_bound_only
#audit Perf.HattnT8Cells.f2c_held
#audit Perf.HattnT8Cells.f2d_held
#audit Perf.HattnT8Cells.f2_control_held
#audit Perf.HattnT8Cells.f1_kill_not_met
#audit Perf.HattnT8Cells.f2_kill_met
#audit Perf.HattnT8Cells.f2_pass_costs_more_than_scan
#audit Perf.HattnT8Cells.f2_flat_at_1m
#audit Perf.HattnT8Cells.f1_pass_bytes_above_oracle
#audit Perf.HattnT3Train.t3a_refuted
#audit Perf.HattnT3Train.t3b_held
#audit Perf.HattnT3Train.t3c_held
#audit Perf.HattnT3Train.t3d_point_refuted
#audit Perf.HattnT3Train.t3d_gate_not_reached
#audit Perf.HattnT3Train.t3e_held
#audit Perf.HattnT3Train.t3f_refuted
#audit Perf.HattnT3Train.port_rule_not_met
#audit Perf.HattnT3Train.ss_conversion_fails
#audit Perf.HattnT3Train.rotation_doubles_7b
#audit Perf.HattnT3Train.sti_best_onebit
#audit Perf.HattnT3Train.l4_closest
#audit Perf.HattnHotCold.gate_passes
#audit Perf.HattnHotCold.thin_margin_1m
#audit Perf.HattnHotCold.hot_beats_read_floor
#audit Perf.HattnHotCold.hot_loses_floor_131k
#audit Perf.HattnHotCold.hot_vs_dense_1m
#audit Perf.HattnHotCold.tok_rate_1m
#audit Perf.HattnHotCold.scan_under_third_1m
#audit Perf.HattnHotCold.controls_held
#audit Perf.PubVerifyCpu.within_one_percent
#audit Perf.PubVerifyCpu.under_point_two_percent
#audit Perf.PubVerifyCpu.prefill_and_main_agree
#audit Perf.OnebitW9.control_zero
#audit Perf.OnebitW9.writer_matches_llama
#audit Perf.OnebitW9.q1_fewest_bits
#audit Perf.OnebitW9.p1_had_beats_q1_twelvefold
#audit Perf.OnebitW9.p1_iq1_beats_had
#audit Perf.OnebitW9.p1_order
#audit Perf.OnebitW9.p2_order
#audit Perf.OnebitW9.p2_not_usable
#audit Perf.OnebitW10.output_error_eq_quadratic
#audit Perf.OnebitW10.invisible_iff
#audit Perf.OnebitW10.same_function_iff
#audit Perf.OnebitW10.ffn_block3_below_quarter
#audit Perf.OnebitW10.ffn_block11_not_below_quarter
#audit Perf.OnebitW10.ffn_block23_not_below_quarter
#audit Perf.OnebitW10.ffn_function_block11_not_below_half
#audit Perf.OnebitW10.attn_in_none_below_half
#audit Perf.OnebitW10.refit_pythagoras
#audit Perf.OnebitW10.refit_le
#audit Perf.OnebitW10.block3_removes_95
#audit Perf.OnebitW10.removed_share_order
#audit Perf.OnebitW10.scale_only_range_fails
#audit Perf.OnebitW10.run_in_predicted_band
#audit Perf.OnebitW10.control_in_band
#audit Perf.OnebitW10.run_below_w9
#audit Perf.OnebitW10.run_not_usable
#audit Perf.FacesPolarBound.p1_refuted
#audit Perf.FacesPolarBound.p2_held
#audit Perf.FacesPolarBound.p3_held
#audit Perf.FacesPolarBound.polar_alone_loses
#audit Perf.Iq1s.rate
#audit Perf.Iq1s.err_expand
#audit Perf.Iq1s.err_eq_best_add_sq
#audit Perf.Iq1s.refit_scalar_le
#audit Perf.Iq1s.idx_mono
#audit Perf.Iq1s.idx_nearest
#audit Perf.Iq1s.exists_monotone_le
#audit Perf.Iq1s.pair_discriminant_nonpos
#audit Perf.Iq1s.pair_form_on_direction
#audit Perf.Iq1s.blocks_compared
#audit Perf.Iq1s.weights_are_blocks
#audit Perf.Iq1s.p1_verdict
-- The elliptic form of the quantisation distortion and the transform-coding bound
-- (`Perf/Elliptic.lean`, QUALITY-118).
#audit Perf.Elliptic.pair_form_psd
#audit Perf.Elliptic.pair_form_psd_of_gram
#audit Perf.Elliptic.coding_gain_sq
#audit Perf.Elliptic.coding_gain_bound
#audit Perf.Elliptic.coding_gain_eq_iff
#audit Perf.Elliptic.coding_gain_ge_one
#audit Perf.Elliptic.register_stage_orthogonal
#audit Perf.Elliptic.register_stage_pi_div_four
#audit Perf.OnebitW13.quad_eq
#audit Perf.OnebitW13.pivot_le_diag
#audit Perf.OnebitW13.shaped_le_white
#audit Perf.OnebitW13.det_eq_prod_pivots
#audit Perf.OnebitW13.hadamard_chol
#audit Perf.OnebitW13.gm_le_am
#audit Perf.OnebitW13.hi_res_floor
#audit Perf.OnebitW13.sign_mse_min
#audit Perf.OnebitW13.sign_loss_bounds
#audit Perf.OnebitW13.sign_loss
#audit Perf.OnebitW13.arm_above_floor_1125
#audit Perf.OnebitW13.one_bit_floor_vs_q4k
#audit Perf.OnebitW13.floor_rate_at_q4k
#audit Perf.OnebitW14.pred_order
#audit Perf.OnebitW14.p1_order
#audit Perf.OnebitW14.rank_inversion
#audit Perf.OnebitW14.ratio_band
#audit Perf.OnebitW14.additivity
#audit Perf.CostCorrespondence.cover_bound
#audit Perf.CostCorrespondence.bytes_floor
#audit Perf.CostCorrespondence.covering_floor
#audit Perf.CostCorrespondence.stiff_floor
#audit Perf.CostCorrespondence.q4_bits_per_stiff_dof
#audit Perf.CostCorrespondence.stiff_fraction_vs_w10
#audit Perf.CostCorrespondence.gauge_lift_bound
#audit Perf.CostCorrespondence.perm_gauge_le
#audit Perf.CostCorrespondence.qwen05b_perm_gauge
#audit Perf.CostCorrespondence.gauge_null
#audit Perf.CostCorrespondence.roofline_subadd
#audit Perf.CostCorrespondence.roofline_nocancel
#audit Perf.CostCorrespondence.chain_append
#audit Perf.CostCorrespondence.kl_sqrt_subadd
#audit Perf.CostCorrespondence.kl_cancels
#audit Perf.CostCorrespondence.kl_joint_le
#audit Perf.CostCorrespondence.kl_pair_sub_iff
#audit Perf.CostCorrespondence.score_form_disc_nonpos
#audit Perf.CostCorrespondence.w14_within_minkowski
#audit Perf.CostCorrespondence.greedy_makespan
#audit Perf.CostCorrespondence.refine_gain_le
#audit Perf.CostCorrespondence.partition_0bf8b4d
#audit Perf.CostCorrespondence.fluid_floor
#audit Perf.CostCorrespondence.fluid_attained
#audit Perf.CostCorrespondence.waterfill_opt
#audit Perf.CostCorrespondence.waterfill_level
#audit Perf.CostCorrespondence.waterfill_floor
#audit Perf.CostCorrespondence.kron_diag
#audit Perf.CostCorrespondence.ray_w15a
#audit Perf.CostCorrespondence.kfac_w15b
#audit Perf.CostCorrespondence.waterfill_highres
#audit Perf.CostCorrespondence.waterfill_exclusion
#audit Perf.CostCorrespondence.slb_gap
#audit Perf.CostCorrespondence.converse_w15c
#audit Perf.CostCorrespondence.tg_floor_ratio
#audit Perf.CostCorrespondence.qwen27b_target_floor
#audit Perf.CostCorrespondence.disc2_eq
#audit Perf.CostCorrespondence.disc2_sq
#audit Perf.CostCorrespondence.face_mul_canonical
#audit Perf.CostCorrespondence.rotation_elliptic
#audit Perf.CostCorrespondence.shear_parabolic
#audit Perf.CostCorrespondence.shear_nilpotent
#audit Perf.CostCorrespondence.rebalance_hyperbolic
#audit Perf.CostCorrespondence.rebalance_preserves_product
#audit Perf.CostCorrespondence.corrector_identity
#audit Perf.CostCorrespondence.corrector_residual
#audit Perf.CostCorrespondence.corrector_attained
#audit Perf.CostCorrespondence.corrector_zero_iff
#audit Perf.CostCorrespondence.spectrum_conj
#audit Perf.CostCorrespondence.method_gap_1133
#audit Perf.CostCorrespondence.waterfill_gain_1133
#audit Perf.CostCorrespondence.floor_gap_1133
#audit Perf.CostCorrespondence.variance_ladder_refuted
#audit Perf.CostCorrespondence.decoupled_ridge_identity
#audit Perf.CostCorrespondence.decoupled_ridge_opt
#audit Perf.CostCorrespondence.decoupled_ridge_eq_iff
#audit Perf.CostCorrespondence.elliptic_filter_conservation
#audit Perf.CostCorrespondence.orthogonal_corrector_identity
#audit Perf.CostCorrespondence.orthogonal_corrector_attained
#audit Perf.CostCorrespondence.gqa_kv_second_plane_budget
#audit Perf.CostCorrespondence.w15_rstar_8b_oracle_and_had_fp32
#audit Perf.CostCorrespondence.telescoping_target_split
#audit Perf.CostCorrespondence.two_component_shrinkage_identity
#audit Perf.CostCorrespondence.two_component_shrinkage_opt
#audit Perf.CostCorrespondence.coupled_shrinkage_excess_identity
#audit Perf.CostCorrespondence.coupled_shrinkage_excess_pos
#audit Perf.CostCorrespondence.qronos_orth_step_identity
#audit Perf.CostCorrespondence.cholesky_step_schur_identity
#audit Perf.CostCorrespondence.act_order_curvature_dominance
#audit Perf.CostCorrespondence.sign_mag_dispersion_identity
#audit Perf.CostCorrespondence.subband_scale_excess_identity
#audit Perf.CostCorrespondence.subband_scale_excess_pos
#audit Perf.CostCorrespondence.whitened_scale_shift_identity
#audit Perf.CostCorrespondence.zellner_tikhonov_identity
#audit Perf.CostCorrespondence.zellner_unit_gamma_excess_pos
#audit Perf.CostCorrespondence.scale_vs_plane_distortion_dominance
#audit Perf.CostCorrespondence.full_attn_2plane_hier_scale_budget
#audit Perf.CostCorrespondence.butterfly_scale_commutation
#audit Perf.CostCorrespondence.butterfly_d_step_dominance
#audit Perf.CostCorrespondence.two_plane_backfit_descent
#audit Perf.CostCorrespondence.butterfly_2plane_attn_budget
#audit Perf.CostCorrespondence.four_factor_gap_identity
#audit Perf.CostCorrespondence.w15_four_factor_05b_decomposition
#audit Perf.CostCorrespondence.butterfly_sign_group_gauge
#audit Perf.CostCorrespondence.output_weighted_scale_excess_identity
#audit Perf.CostCorrespondence.output_weighted_scale_dominance
#audit Perf.CostCorrespondence.reduced_rank_drift_generalization_dominance
#audit Perf.CostCorrespondence.column_scale_group_gauge
#audit Perf.CostCorrespondence.butterfly_gptq_metric_invariance
#audit Perf.CostCorrespondence.rank2_down_corrector_dominance
#audit Perf.CostCorrespondence.bscale_rank2_down_budget
#audit Perf.CostCorrespondence.split_butterfly_joint_dominance
#audit Perf.CostCorrespondence.rmsnorm_gauge_invariance
#audit Perf.CostCorrespondence.hyperbolic_precond_excess_identity
#audit Perf.CostCorrespondence.hyperbolic_precond_dominance
#audit Perf.CostCorrespondence.precond_damping_distortion
#audit Perf.CostCorrespondence.fnorm_fisher_variance_nonneg
#audit Perf.CostCorrespondence.fnorm_newton_descent
#audit Perf.CostCorrespondence.diag_precond_coherent_overshoot
#audit Perf.CostCorrespondence.e8_o8_codebook_cardinality
#audit Perf.CostCorrespondence.e8_o8_parabolic_spike_dominance
#audit Perf.CostCorrespondence.e8_o8_pair_root_dominance
#audit Perf.CostCorrespondence.delayed_decision_cholesky_dominance
#audit Perf.CostCorrespondence.beam_wscale_trajectory_dominance
#audit Perf.CostCorrespondence.e8_icm_global_descent_dominance
#audit Perf.CostCorrespondence.quantized_scale_icm_dominance
#audit Perf.CostCorrespondence.stage1_drift_shrinkage_dominance
#audit Perf.CostCorrespondence.swiglu_bilinear_comp_dominance
#audit Perf.CostCorrespondence.rmsnorm_pullback_sink_attenuation
#audit Perf.CostCorrespondence.sink_row_trust_clipping_dominance
#audit Perf.CostCorrespondence.swiglu_hyperbolic_noise_minimization
#audit Perf.CostCorrespondence.coupled_schur_crosstalk_dominance
#audit Perf.CostCorrespondence.q4k_vs_1bit_transfer_floor
#audit Perf.CostCorrespondence.asymmetric_wiener_target_optimality
#audit Perf.CostCorrespondence.asymmetric_distortion_penalty_strict
#audit Perf.CostCorrespondence.w16_t0_best_arm_3seed
#audit Perf.CostCorrespondence.w16_t0b_matched_seed_baseline
#audit Perf.CostCorrespondence.w16_t3_nb_paired
#audit Perf.FaceTyped.null_channels_eq
#audit Perf.FaceTyped.two_plane_split_complex
#audit Perf.FaceTyped.two_plane_null_channel
#audit Perf.FaceTyped.two_plane_value_mem
#audit Perf.FaceTyped.null_basis_rate_eq
#audit Perf.FaceTyped.null_basis_diag_cross_zero
#audit Perf.FaceTyped.coefC_symm
#audit Perf.FaceTyped.bilinear_junction_hyperbolic
#audit Perf.FaceTypedE8.spinor_mul_closed
#audit Perf.FaceTypedE8.spinor_mul_self
#audit Perf.FaceTypedE8.spinorNullBasis_involutive
#audit Perf.FaceTypedE8.spinor_pair_null_channel
#audit Perf.FaceTypedE8.spinor_cross_diag
#audit Perf.FaceTypedE8.joint_block_cost_expand
#audit Perf.FaceTypedE8.joint_move_le_planewise
#audit Perf.FaceTypedE8.joint_move_le_planewise'
#audit Perf.FaceTypedE8.spinor_card
#audit Perf.FaceTypedE8.pair_rate_eq
#audit Perf.FaceTypedE8.sparse_not_injective
#audit Perf.CostCorrespondence.w16_t6_sideinfo
#audit Perf.SideInfo.char_coords_norm
#audit Perf.SideInfo.norm_boost_blind
#audit Perf.SideInfo.lone_channel_fiber_card
#audit Perf.SideInfo.side_info_recovers
#audit Perf.SideInfo.side_info_char_coord
#audit Perf.SideInfo.gauss_side_info_saving
#audit Perf.SideInfo.lowerBound_count_le
#audit Perf.SideInfo.lowerBound_spec
#audit Perf.SideInfo.w16_t6_route_b_open
#audit Perf.SideInfo.w16_t6_route_b_closed_linear
#audit Perf.SideInfo.w16_t6_route_a_closed
#audit Perf.Generalisation.J_nonneg
#audit Perf.Generalisation.J_mono
#audit Perf.Generalisation.J_smul
#audit Perf.Generalisation.loewner_sandwich
#audit Perf.Generalisation.argmin_optimism
#audit Perf.Generalisation.mp_floor_gt
#audit Perf.Generalisation.w17_c_sandwich_verdict
#audit Perf.Generalisation.w17_c_bookkeeping
#audit Perf.Generalisation.w17_c_q_dominates
#audit Perf.Generalisation.w17_c_o_decomposition
#audit Perf.Generalisation.w17_c_p1_aggregate_in_band
#audit Perf.Generalisation.w17_c_p1_per_kind_refuted
#audit Perf.Generalisation.w17_c_p2_refuted
#audit Perf.Generalisation.w17_c_p3_alpha_band
#audit Perf.Generalisation.w17_c_p3_egen_decreasing
#audit Perf.Generalisation.w17_c_p3_egen25_refuted
#audit Perf.Generalisation.w17_c_sandwich_measured
#audit Perf.Generalisation.w17_c_run100_bits
#audit Perf.Generalisation.w17_d_q_split
#audit Perf.Generalisation.w17_d_tilt_ratio
#audit Perf.Generalisation.w17_d_qlin_ratio
#audit Perf.Generalisation.w17_d_p_tilt_bands
#audit Perf.Generalisation.w17_d_falsifier_not_triggered
#audit Perf.Generalisation.w17_d_lin_dominates
#audit Perf.Generalisation.w17_d_ratio_dominance_of_log
#audit Perf.Generalisation.w17_d_ratio_dominance_of_log_fifth
#audit Perf.KLTilt.kl_sm_eq
#audit Perf.KLTilt.A_hasDerivAt
#audit Perf.KLTilt.A_second
#audit Perf.KLTilt.kl_eq_half_var
#audit Perf.KLTilt.wvar_le_range
#audit Perf.KLTilt.kl_le_range_sq_div_8
#audit Perf.KLTilt.witness_half_var
#audit Perf.KLTilt.witness_range

-- One-bit bit budget (Perf/BitsRefine.lean): executed bodies refine the closed forms.
#audit apeiron_word.BitsRefine.sign_layer_spec
#audit apeiron_word.BitsRefine.bits_sign_spec
#audit apeiron_word.BitsRefine.params_spec
#audit apeiron_word.BitsRefine.bits_r_spec
#audit apeiron_word.BitsRefine.bits_c_spec
#audit apeiron_word.BitsRefine.bits_d_spec
#audit apeiron_word.BitsRefine.bits_corr_spec
#audit apeiron_word.BitsRefine.total_bits_spec
#audit apeiron_word.BitsRefine.index_bits_spec
#audit apeiron_word.BitsRefine.qwen_fits
#audit apeiron_word.BitsRefine.qwen_bits_sign
#audit apeiron_word.BitsRefine.qwen_bits_r
#audit apeiron_word.BitsRefine.qwen_bits_c
#audit apeiron_word.BitsRefine.qwen_bits_d
#audit apeiron_word.BitsRefine.qwen_bits_corr
#audit apeiron_word.BitsRefine.qwen_total_is_sum
#audit apeiron_word.BitsRefine.qwen_total_bits
#audit apeiron_word.BitsRefine.qwen_total_bits_receipt
#audit apeiron_word.BitsRefine.qwen_params
#audit apeiron_word.BitsRefine.qwen_params_receipt
#audit apeiron_word.BitsRefine.choose_4864_416_le
#audit apeiron_word.BitsRefine.lt_choose_4864_416
#audit apeiron_word.BitsRefine.index_bits_per_set_minimal
#audit apeiron_word.BitsRefine.index_bits_qwen
#audit apeiron_word.BitsRefine.row_index_cover
#audit apeiron_word.BitsRefine.headline_bpw_bounds
#audit apeiron_word.BitsRefine.counted_bpw_bounds
#audit apeiron_word.BitsRefine.headline_bpw_receipt

-- quantum-register (docs/epics/quantum-register.md): register gates, Bloch = KS, branch Laplacian,
-- branch Madelung, tetra limit, branch quantization, path density, learned-angle verdicts.
-- Board: quantum-register-learned-LD7oW6M8EVQgAkWyMtEOs.
#audit apeiron_word.Register.pairStage_low
#audit apeiron_word.Register.pairStage_high
#audit apeiron_word.Register.mod_add_low
#audit apeiron_word.Register.mod_sub_high
#audit apeiron_word.Register.pairStage_comp
#audit apeiron_word.Register.pairStage_smul
#audit apeiron_word.Register.add_lt_pow
#audit apeiron_word.Register.sum_pairs
#audit apeiron_word.Register.register_stage_pairs
#audit apeiron_word.Register.pairStage_sum_sq
#audit apeiron_word.Register.register_stage_normForm
#audit apeiron_word.Register.register_stage_conj
#audit apeiron_word.Register.hadRef_succ
#audit apeiron_word.Register.hadRef_shift
#audit apeiron_word.Register.register_hadamard
#audit apeiron_word.Register.register_hadamard_fwht
#audit apeiron_word.Register.register_stage_diagonal
#audit apeiron_word.Register.rep_parabolic_is_shear
#audit apeiron_word.Register.diag_eq_hadamard_conj_rep_hyperbolic
#audit apeiron_word.Register.iwasawa_faces
#audit apeiron_word.Register.iwasawa_faces_hadamard
#audit apeiron_word.Register.register_cost
#audit apeiron_word.Register.reflPair_involutive
#audit apeiron_word.Register.pairStage_id
#audit apeiron_word.Register.reflStage_involutive
#audit apeiron_word.Register.reflStage_sum_sq
#audit apeiron_word.Register.reflStep_involutive
#audit apeiron_word.Register.reflStages_append
#audit apeiron_word.Register.reflStages_reverse_cancel
#audit apeiron_word.Register.refl_quarter
#audit apeiron_word.Register.quarter_sq
#audit apeiron_word.Register.cz_involutive
#audit apeiron_word.Register.cz_sum_sq
#audit apeiron_word.Register.productState_cross
#audit apeiron_word.Register.cz_entangles
#audit apeiron_word.Register.measProj_idem
#audit apeiron_word.Register.measProj_add
#audit apeiron_word.Register.measProj_orth
#audit apeiron_word.Register.born_sum
#audit Numeric.Qubit.phase_re
#audit Numeric.Qubit.phase_im
#audit Numeric.Qubit.ks_norm
#audit Numeric.Qubit.bloch_eq_ks
#audit Numeric.Qubit.bloch_norm
#audit Numeric.Qubit.bloch_rz
#audit Numeric.Qubit.bloch_ry
#audit Numeric.Qubit.suR_blochVec
#audit Numeric.Qubit.suR_orthogonal
#audit Numeric.Qubit.sigmaN_sq
#audit Numeric.Qubit.sigmaN_sq_unit
#audit Numeric.Qubit.projP_idem
#audit Numeric.Qubit.projM_idem
#audit Numeric.Qubit.projP_add_projM
#audit Numeric.Qubit.projP_sub_projM
#audit Numeric.Qubit.projP_mul_projM
#audit Numeric.Qubit.projM_mul_projP
#audit Numeric.Qubit.sigmaN_meridian
#audit Numeric.Qubit.sigmaN_meridian_sq
#audit Numeric.Qubit.sigmaN_meridian_pi_four
#audit Numeric.BranchLaplacian.tetra_stencil_reconstructs_laplacian
#audit Numeric.BranchLaplacian.branch_count_ge_four
#audit Numeric.BranchLaplacian.tetra_family_attains_four
#audit Numeric.BranchLaplacian.secondDiff_tendsto
#audit Numeric.BranchLaplacian.stencil4_tendsto_laplacian
#audit Numeric.BranchLaplacian.second_derivative_comp
#audit Numeric.BranchLaplacian.stencil4_quad_comp_affine
#audit Numeric.BranchLaplacian.stencil4_affine_comp_quad
#audit Numeric.BranchMadelung.branchQ_tendsto
#audit Numeric.BranchMadelung.schrodinger_iff_branch_madelung
#audit Numeric.TetraLimit.oneSided_tendsto
#audit Numeric.TetraLimit.tetra4_tendsto_lines
#audit Numeric.TetraLimit.tetra4_tendsto_field
#audit Numeric.TetraLaplacian.line_second_deriv
#audit Numeric.TetraLaplacian.tetra_sum_eq_lap
#audit Numeric.TetraLaplacian.tetra4_tendsto_laplacian
#audit Numeric.BranchQuantization.term_eq_pow
#audit Numeric.BranchQuantization.branchSum_eq_of_exp_eq_one
#audit Numeric.BranchQuantization.branchSum_geom
#audit Numeric.BranchQuantization.norm_branchSum_le
#audit Numeric.BranchQuantization.branchAvg_tendsto_zero
#audit Numeric.BranchQuantization.branchAvg_tendsto_one
#audit Numeric.BranchQuantization.branchSum_eq_zero_iff
#audit Numeric.BranchQuantization.exp_pi_div_two_mul_I
#audit Numeric.BranchQuantization.branchSum_four_quarter
#audit Numeric.BranchQuantization.exp_eq_one_iff_mem
#audit Numeric.PathDensity.pathDensity_hasDerivAt
#audit Numeric.PathDensity.pathDensity_zero
#audit Numeric.PathDensity.pathDensity_unique
#audit Numeric.PathDensity.pathDensity_pos
#audit Perf.QuantumRotation.kan_probe_protocol
#audit Perf.QuantumRotation.kan_probe_angle_budget
#audit Perf.QuantumRotation.kan_probe_sum_loses
#audit Perf.QuantumRotation.kan_probe_geo_wins
#audit Perf.QuantumRotation.kan_probe_attn_out_in
#audit Perf.QuantumRotation.kan_probe_attn_in
#audit Perf.QuantumRotation.kan_probe_ffn_loses
#audit Perf.QuantumRotation.kan_exact_protocol
#audit Perf.QuantumRotation.kan_exact_step_cost
#audit Perf.QuantumRotation.kan_exact_loses_wall
#audit Perf.QuantumRotation.butterfly_curve_convex
#audit Perf.QuantumRotation.kan_exact_wins_matched_step
#audit Perf.QuantumRotation.kan_mm_protocol
#audit Perf.QuantumRotation.kan_mm_step_cost
#audit Perf.QuantumRotation.kan_mm_loses_wall
#audit Perf.QuantumRotation.kan_mm_wins_matched_step
#audit Perf.QuantumRotation.kan_mmb_protocol
#audit Perf.QuantumRotation.kan_mmb_step_cost
#audit Perf.QuantumRotation.kan_mmb_beats_butterfly_wall
#audit Perf.QuantumRotation.kan_mmb_loses_hadmm
#audit Perf.QuantumRotation.kan_mmb_wins_matched_step
#audit Perf.QuantumRotation.kan_pair_step_parity
#audit Perf.QuantumRotation.kan_pair_each
#audit Perf.QuantumRotation.pair_host_slowdown
#audit Perf.QuantumRotation.t5_corr_store_refines
#audit Perf.QuantumRotation.t5_corr_loses
#audit Perf.QuantumRotation.attrib_no_layer_holds_kl
#audit Perf.QuantumRotation.attrib_kl_nonadditive
#audit Perf.QuantumRotation.attrib_kinds_coadapted
#audit Perf.QuantumRotation.iso_kl_superadditive
#audit Perf.QuantumRotation.iso_groups_saturate
#audit Perf.QuantumRotation.iso_finishing_saturates
#audit Perf.QuantumRotation.iso_every_layer_costs
#audit Perf.QuantumRotation.blockft_refuted
#audit Perf.QuantumRotation.blockft_phase_raises_kl
#audit Perf.QuantumRotation.blockft_block_step_cost
#audit Perf.QuantumRotation.YaqaBlock.dJ_exact
#audit Perf.QuantumRotation.YaqaBlock.block_complete_square
#audit Perf.QuantumRotation.YaqaBlock.block_target_attains
#audit Perf.QuantumRotation.yaqa_refuted
#audit Perf.QuantumRotation.yaqa_jo_falls
#audit Perf.QuantumRotation.yaqa_capture_noise
#audit Perf.QuantumRotation.yaqaxv_fits_noise
#audit Perf.QuantumRotation.yaqaxv_shrink_heldout
#audit Perf.QuantumRotation.yaqas_refuted
#audit Perf.QuantumRotation.yaqas90_small_move
#audit Perf.QuantumRotation.yaqab_shrink_wins
#audit Perf.QuantumRotation.yaqab_alpha_interior
#audit Perf.QuantumRotation.yaqab_before_misleads
#audit Perf.QuantumRotation.yaqalw_loses
#audit Perf.QuantumRotation.yaqalw_intensity_small
#audit Perf.SpecDraftKL.accept_ge
#audit Perf.SpecDraftKL.mean_accept_ge
#audit Perf.SpecDraftKL.q4km_accept
#audit Perf.SpecDraftKL.resq_accept
#audit Perf.SpecDraftKL.blockLenAt_const
#audit Perf.SpecDraftKL.blockLen_geom
#audit Perf.SpecDraftKL.blockLen_mono
#audit Perf.SpecDraftKL.blockLen_bounds
#audit Perf.SpecDraftKL.spec_faster
#audit Perf.SpecDraftKL.blockLen_witness
#audit Perf.SpecDraftKL.spec_accept_measured
#audit Perf.SpecDraftKL.spec_runs_cluster
#audit Perf.SpecDraftKL.softmax_adjoint
#audit Perf.SpecDraftKL.spectv_accept
#audit Perf.SpecDraftKL.spectv_top1_kl
#audit Perf.SpecDraftKL.verify_curve
#audit Perf.SpecDraftKL.verify_best_k
#audit Perf.SpecDraftKL.spec27b_cpu
#audit Perf.Train.rmsnormBwdBytes_eq
#audit Perf.Train.rmsnormDgammaReduceBytes_eq
#audit Perf.Train.linearBwdInputReads_eq
#audit Perf.Train.linearBwdWeightReads_eq
#audit Perf.Train.attnBwdQReads_eq
#audit Perf.Train.attnBwdDsReads_eq
#audit Perf.Train.denseBwdTiledReads_eq
#audit Perf.Train.adamUpdateReads_eq
#audit Perf.Train.denseFwdTiledReads_eq
#audit Perf.Train.attnFwdReads_eq
#audit Perf.Train.attnRowsumReads_eq
#audit Perf.Train.attnBwdRowsumTiledReads_eq
#audit Perf.Train.swigluFwdReads_eq
#audit Perf.Train.swigluBwdReads_eq
#audit Perf.Train.klHeadReads_eq
#audit Perf.Train.attnDkdvReads_eq
#audit Perf.Train.rmsnormFwdReads_eq
#audit Perf.Train.ropeFwdReads_eq
#audit Perf.Train.ropeBwdReads_eq
#audit Perf.Train.attnBwdDsTiledReads_eq
#audit Perf.Train.layerStepDispatches_eq
#audit Perf.Train.cacheSavesNineBuilds
#audit Perf.Train.cacheSavesNineBuildsAtMeasured
#audit Perf.Train.measuredPostFixBelowTarget
#audit Perf.Handoff.align256_eq
#audit Perf.Handoff.writeChunk_le_cap
#audit Perf.Handoff.writeChunk_le_remaining
#audit apeiron_word.RegisterRefinement.checked_mul_spec
#audit apeiron_word.RegisterRefinement.checked_mul_none
#audit apeiron_word.RegisterRefinement.checked_mul_none_of_a
#audit apeiron_word.RegisterRefinement.stage_mul_not_pow
#audit apeiron_word.RegisterRefinement.stage_mul_too_large
#audit apeiron_word.RegisterRefinement.stage_mul_length_mismatch
#audit apeiron_word.RegisterRefinement.stage_block_spec
#audit apeiron_word.RegisterRefinement.stageRef_denote
#audit apeiron_word.RegisterRefinement.stage_mul_denote
#audit apeiron_word.RegisterRefinement.stage_mul_denote_witness
#audit Quantum.FaceShift.C_add
#audit Quantum.FaceShift.S_add
#audit Quantum.FaceShift.C_neg
#audit Quantum.FaceShift.S_neg
#audit Quantum.FaceShift.C_eq_one_add
#audit Quantum.FaceShift.hasDerivAt_C
#audit Quantum.FaceShift.hasDerivAt_S
#audit Quantum.FaceShift.hasDerivAt_K
#audit Quantum.FaceShift.hasDerivAt_f
#audit Quantum.FaceShift.hasDerivAt_f1
#audit Quantum.FaceShift.S_central
#audit Quantum.FaceShift.C_central
#audit Quantum.FaceShift.K_central
#audit Quantum.FaceShift.S_second
#audit Quantum.FaceShift.C_second
#audit Quantum.FaceShift.K_second
#audit Quantum.FaceShift.shift_first
#audit Quantum.FaceShift.shift_second
#audit Quantum.FaceShift.S_neg_one_half_pi
#audit Quantum.FaceShift.K_neg_one_half_pi
#audit Quantum.FaceShift.parameter_shift_elliptic
#audit Quantum.FaceShift.central_difference_parabolic
#audit Quantum.FaceShift.S_one
#audit Quantum.FaceShift.K_one
#audit Quantum.FaceShift.shift_hyperbolic
#audit Quantum.FaceShift.cubic_violates_shift_first
#audit Quantum.FaceShift.verdict_l5

-- Amplitude-form KL divergence (`Numeric/Quantum/Ledger/AmplitudeKL.lean`): the Hellinger-type lower bound, the
-- second-order agreement of `s ↦ KL p (q_s)` with `s ↦ 2 ‖√(q_s) − √p‖²`, the logistic/softmax
-- identity and its amplitude form, and the unit verdict.
#audit Quantum.AmplitudeKL.kl_ge_amplitude_sq
#audit Quantum.AmplitudeKL.kl_second_order
#audit Quantum.AmplitudeKL.logistic_eq_softmax_two
#audit Quantum.AmplitudeKL.logistic_amplitude_add
#audit Quantum.AmplitudeKL.verdict_l6

-- Spherical t-design moment ladder in ℤ³ (`Numeric/Quantum/Ledger/DesignLadder.lean`): the
-- tetrahedron 2-design and its odd-moment refusal of order 3, the octahedron 3-design and its
-- refusal of order 4, the cube 3-design and its refusal of order 4, the antipodal closure
-- `T ∪ (−T) = C`, and the unit verdict.
#audit Quantum.DesignLadder.tetra_deg1
#audit Quantum.DesignLadder.tetra_deg2
#audit Quantum.DesignLadder.tetra_xyz
#audit Quantum.DesignLadder.tetra_design_two
#audit Quantum.DesignLadder.tetra_not_design_three
#audit Quantum.DesignLadder.octa_deg1
#audit Quantum.DesignLadder.octa_deg2
#audit Quantum.DesignLadder.octa_deg3
#audit Quantum.DesignLadder.octa_x4
#audit Quantum.DesignLadder.octa_design_three
#audit Quantum.DesignLadder.octa_not_design_four
#audit Quantum.DesignLadder.cube_deg1
#audit Quantum.DesignLadder.cube_deg2
#audit Quantum.DesignLadder.cube_deg3
#audit Quantum.DesignLadder.cube_x4
#audit Quantum.DesignLadder.cube_design_three
#audit Quantum.DesignLadder.cube_not_design_four
#audit Quantum.DesignLadder.tetra_union_is_cube
#audit Quantum.DesignLadder.verdict_l4

-- Two composed elliptic cells `ℂ ⊗_ℝ ℂ = (RingCell ℂ, ringMul (−1))`
-- (`Numeric/Quantum/Ledger/EllipticTensor.lean`): each factor elliptic, the hyperbolic face
-- `j = i ⊗ i` with `j² = +1`, `j ≠ ±1`, the complementary idempotent pair `e`, `e' = 1 − e`,
-- the tensor rule `i ⊗ 1 = 1 ⊗ i` on `e` with its conjugate on `e'`, the ring isomorphism
-- `ℂ ⊗_ℝ ℂ ≅ ℂ × ℂ`, and the unit verdict.
#audit Quantum.EllipticTensor.iL_sq
#audit Quantum.EllipticTensor.iR_sq
#audit Quantum.EllipticTensor.j_eq
#audit Quantum.EllipticTensor.j_sq
#audit Quantum.EllipticTensor.j_ne_one
#audit Quantum.EllipticTensor.j_ne_neg_one
#audit Quantum.EllipticTensor.e_idem
#audit Quantum.EllipticTensor.e'_idem
#audit Quantum.EllipticTensor.e_complementary
#audit Quantum.EllipticTensor.tensor_rule
#audit Quantum.EllipticTensor.anti_rule
#audit Quantum.EllipticTensor.split_mul
#audit Quantum.EllipticTensor.split_one
#audit Quantum.EllipticTensor.split_e
#audit Quantum.EllipticTensor.split_e'
#audit Quantum.EllipticTensor.verdict_l2

-- The cell `ℝ[x]/(x² − σ)` is the Clifford algebra of the form `Q σ t = σ·t²`
-- (`Numeric/Quantum/Ledger/CellClifford.lean`): the algebra isomorphism `cliffordEquiv` with
-- `toCell` carrying Clifford multiplication to `mul σ`, the elliptic face `σ = −1` (`ℂ`) and the
-- parabolic face `σ = 0` (dual numbers), two anticommuting generators `e1`, `e2` squaring to `c₁`,
-- `c₂` as a quaternion algebra at `(−1,−1)` and `(+1,+1)`, and the rebit `ℍ[ℝ,1,0,1] ≅ M₂(ℝ)`.
#audit Quantum.CellClifford.toCell_mul
#audit Quantum.CellClifford.toCell_one
#audit Quantum.CellClifford.toCell_ι
#audit Quantum.CellClifford.toCell_bijective
#audit Quantum.CellClifford.Q_neg_one
#audit Quantum.CellClifford.Q_zero
#audit Quantum.CellClifford.elliptic_complex
#audit Quantum.CellClifford.parabolic_dual
#audit Quantum.CellClifford.e1_sq
#audit Quantum.CellClifford.e2_sq
#audit Quantum.CellClifford.anticommute
#audit Quantum.CellClifford.k_sq_hamilton
#audit Quantum.CellClifford.k_sq_split
#audit Quantum.CellClifford.rebit_bijective
#audit Quantum.CellClifford.verdict_l1

-- The extracted Q32.32 dense-normalisation primitives and the `swiglu_fwd` loop
-- (`Learn/Grad/DenseNorm.lean`) refine the integer spec `Learn/Grad/DenseNormSpec.lean`.
#audit apeiron_grad.dense_round32_val
#audit apeiron_grad.dense_mul_round_some
#audit apeiron_grad.dense_div_round_some
#audit apeiron_grad.dense_narrow_some
#audit apeiron_grad.dense_ov_some
#audit apeiron_grad.swigluVals_eq_zipWith_len
#audit apeiron_grad.swiglu_fwd_loop_body_step
#audit apeiron_grad.swiglu_fwd_loop_spec
#audit apeiron_grad.swiglu_fwd_spec
#audit apeiron_grad.swiglu_one_bwd_spec
#audit apeiron_grad.swiglu_fwd_empty
#audit apeiron_grad.swiglu_fwd_shape

-- The `silu_deriv`, `swiglu_bwd`, `rmsnorm_fwd` and `inv_rms` entry points, the `rmsnorm_bwd`
-- accumulator and output loops, and their `ℤ`-specification links
-- (`Learn/Grad/DenseNorm.lean`, `Learn/Grad/DenseNormBwd.lean`).
#audit apeiron_grad.silu_deriv_spec
#audit apeiron_grad.swigluBwdVals_eq_zip_len
#audit apeiron_grad.swiglu_bwd_loop_body_step
#audit apeiron_grad.swiglu_bwd_loop_spec
#audit apeiron_grad.swiglu_bwd_spec
#audit apeiron_grad.rmsnorm_fwd_loop_body_step
#audit apeiron_grad.rmsnorm_fwd_loop_spec
#audit apeiron_grad.rmsnorm_fwd_spec
#audit apeiron_grad.norm_cast_U64_val
#audit apeiron_grad.norm_hcast_I64MAX_val
#audit apeiron_grad.norm_hcast_I64_val
#audit apeiron_grad.norm_guard_iff
#audit apeiron_grad.norm_I64_checked_sub_some
#audit apeiron_grad.inv_rms_loop_body_step
#audit apeiron_grad.inv_rms_loop_spec
#audit apeiron_grad.inv_rms_spec
#audit apeiron_grad.bwdAVals_step
#audit apeiron_grad.bwdAVals_length
#audit apeiron_grad.bwdASum_step
#audit apeiron_grad.rmsnorm_bwd_loop0_body_step
#audit apeiron_grad.rmsnorm_bwd_loop0_spec
#audit apeiron_grad.norm_I128_shiftLeft_63
#audit apeiron_grad.norm_I128_shiftRight_64_eq
#audit apeiron_grad.norm_I128_shiftRight_64_val
#audit apeiron_grad.bwdDxVals_step
#audit apeiron_grad.bwdDxVals_length
#audit apeiron_grad.bwdDgVals_step
#audit apeiron_grad.bwdDgVals_length
#audit apeiron_grad.rmsnorm_bwd_loop1_body_step
#audit apeiron_grad.rmsnorm_bwd_loop1_spec
#audit apeiron_grad.bwdDxValsL
#audit apeiron_grad.bwdDxValsL_step
#audit apeiron_grad.bwdDxValsL_length
#audit apeiron_grad.bwdDxVals_eq_L
#audit apeiron_grad.rmsnorm_bwd_spec

-- The tetrahedral stencil is the qubit SIC (`Numeric/Quantum/Ledger/SicStencil.lean`): the four
-- unit Bloch vectors `nVec k = (1/√3) • tetraV k` with pairwise inner product `-1/3`, the rank-one
-- projectors `Πₖ = (1 + Σ·nₖ)/2`, the SIC overlap `tr(Πⱼ Πₖ) = 1/3`, the identity resolution
-- `Σ Πₖ/2 = 1`, the second moment `(4/3)·u`, and `d² = 4` vectors.
#audit Quantum.SicStencil.nVec_unit
#audit Quantum.SicStencil.nVec_dot_self
#audit Quantum.SicStencil.nVec_dot_ne
#audit Quantum.SicStencil.Pi_idem
#audit Quantum.SicStencil.sigmaN_conjTranspose
#audit Quantum.SicStencil.projP_conjTranspose
#audit Quantum.SicStencil.Pi_herm
#audit Quantum.SicStencil.trace_projP
#audit Quantum.SicStencil.trace_projP_mul
#audit Quantum.SicStencil.sic_overlap
#audit Quantum.SicStencil.sic_diag
#audit Quantum.SicStencil.nVec_sum_zero
#audit Quantum.SicStencil.sigmaN_sum
#audit Quantum.SicStencil.sic_povm
#audit Quantum.SicStencil.sic_second_moment
#audit Quantum.SicStencil.sic_count
#audit Quantum.SicStencil.verdict_l3

-- Q1 interference Gram, measured (`Numeric/Quantum/Measured/InterferenceGram.lean`, receipt
-- `docs/evidence/receipts/onebit/qarch-q1.json`): the amplitude prediction `2 ‖Σ δψ_l‖²` over-predicts
-- whole-model KL in every cell and misses the prefix and suffix groups with opposite signs (refuted).
#audit Quantum.InterferenceGram.all_overpredicted
#audit Quantum.InterferenceGram.prefix_over_suffix_under
#audit Quantum.InterferenceGram.diagonal_share_falls
#audit Quantum.InterferenceGram.verdict_q1

-- Q2 joint vs separable Fisher-sketch allocation, measured (`Numeric/Quantum/Measured/JointFisher.lean`, receipt
-- `docs/evidence/receipts/onebit/qarch-q2.json`): the joint arm is not below the separable arm at equal rate (refuted).
#audit Quantum.JointFisher.separable_below_uniform
#audit Quantum.JointFisher.joint_not_below_separable
#audit Quantum.JointFisher.joint_below_on_w0
#audit Quantum.JointFisher.equal_rate
#audit Quantum.JointFisher.regime_far
#audit Quantum.JointFisher.verdict_q2

-- APEIRON-QARCH-L7: measurement and reflection, the ends of the β family (Numeric/Quantum/Ledger/DeltaEnds.lean).
#audit Quantum.DeltaEnds.inner_corr_k
#audit Quantum.DeltaEnds.corr_k
#audit Quantum.DeltaEnds.corr_orth
#audit Quantum.DeltaEnds.corr_comp
#audit Quantum.DeltaEnds.corr_selfadj
#audit Quantum.DeltaEnds.norm_sq_corr
#audit Quantum.DeltaEnds.measure_idem
#audit Quantum.DeltaEnds.measure_born
#audit Quantum.DeltaEnds.reflect_invol
#audit Quantum.DeltaEnds.reflect_isometry
#audit Quantum.DeltaEnds.reflect_k
#audit Quantum.DeltaEnds.nonexpansive_iff
#audit Quantum.DeltaEnds.idem_iff
#audit Quantum.DeltaEnds.invol_iff
#audit Quantum.DeltaEnds.isometry_iff
#audit Quantum.DeltaEnds.verdict_l7_general
#audit Quantum.DeltaEnds.verdict_l7

-- APEIRON-QARCH-L9: the exact uncertainty relation (Numeric/Quantum/Ledger/ExactUncertainty.lean).
#audit Quantum.ExactUncertainty.sqrt_second_deriv
#audit Quantum.ExactUncertainty.laplacian_sqrt
#audit Quantum.ExactUncertainty.rho_Q_pointwise
#audit Quantum.ExactUncertainty.rho_ncMom_sq
#audit Quantum.ExactUncertainty.integral_second_zero
#audit Quantum.ExactUncertainty.integral_rho_Q
#audit Quantum.ExactUncertainty.quantum_energy_eq_nc_kinetic
#audit Quantum.ExactUncertainty.exact_uncertainty
#audit Quantum.ExactUncertainty.cramer_rao_core
#audit Quantum.ExactUncertainty.heisenberg_from_cr
#audit Quantum.ExactUncertainty.cramer_rao_integral
#audit Quantum.ExactUncertainty.witness_pointwise
#audit Quantum.ExactUncertainty.witness_ncMom
#audit Quantum.ExactUncertainty.verdict_l9
#audit Quantum.ExactUncertainty.fisher_quad_nonneg
#audit Quantum.ExactUncertainty.cauchy_schwarz_fisher
#audit Quantum.ExactUncertainty.cramer_rao
#audit Quantum.ExactUncertainty.verdict_l9b

-- APEIRON-QARCH-L9c: the full Heisenberg relation from the Madelung wavefunction
-- (Numeric/Quantum/Ledger/ExactUncertainty.lean).
#audit Quantum.ExactUncertainty.star_exp_I
#audit Quantum.ExactUncertainty.madelung_hasDerivAt
#audit Quantum.ExactUncertainty.madelung_normSq
#audit Quantum.ExactUncertainty.madelung_current
#audit Quantum.ExactUncertainty.weighted_var_nonneg
#audit Quantum.ExactUncertainty.momentum_spread_sq
#audit Quantum.ExactUncertainty.ncSpread_eq
#audit Quantum.ExactUncertainty.nc_le_momSpread
#audit Quantum.ExactUncertainty.heisenberg
#audit Quantum.ExactUncertainty.witness_madelung
#audit Quantum.ExactUncertainty.verdict_l9c

-- APEIRON-QARCH-L10: the elliptic cell is the extra rebit (Numeric/Quantum/Ledger/EllipticRebit.lean).
#audit Quantum.EllipticRebit.rep_mul
#audit Quantum.EllipticRebit.rep_one
#audit Quantum.EllipticRebit.rep_add
#audit Quantum.EllipticRebit.rep_injective
#audit Quantum.EllipticRebit.det_rep
#audit Quantum.EllipticRebit.rep_neg_one
#audit Quantum.EllipticRebit.J_sq
#audit Quantum.EllipticRebit.J_transpose
#audit Quantum.EllipticRebit.rep_orthogonal
#audit Quantum.EllipticRebit.rep_one_x
#audit Quantum.EllipticRebit.rep_zero_x
#audit Quantum.EllipticRebit.realify_mul
#audit Quantum.EllipticRebit.realify_one
#audit Quantum.EllipticRebit.realify_conjTranspose
#audit Quantum.EllipticRebit.realify_scalar
#audit Quantum.EllipticRebit.verdict_l10

-- The extracted `solve_port.gemm` bound scans and refusal chain refine the `ℤ` model
-- (`Learn/Solve/GemmSpec.lean`, `Gemm.lean`); `GemmWitness.lean` shows the domain and each refusal arm are inhabited.
#audit apeiron_word.outNn_length
#audit apeiron_word.outNt_length
#audit apeiron_word.outNn_getD
#audit apeiron_word.outNt_getD
#audit apeiron_word.bnd_getD
#audit apeiron_word.bnd_mul
#audit apeiron_word.dsum_abs_le
#audit apeiron_word.dsum_abs_lt
#audit apeiron_word.round_half_fits
#audit apeiron_word.dsum_round_fits
#audit apeiron_word.MAXK_eq
#audit apeiron_word.BND_eq
#audit apeiron_word.gemm_BOUND_BITS_val
#audit apeiron_word.gemm_unsigned_abs_val
#audit apeiron_word.gemm_allBelowRef_eq_true
#audit apeiron_word.gemm_allBelowRef_true
#audit apeiron_word.gemm_allBelowRef_append
#audit apeiron_word.gemm_all_below_loop_body_exec
#audit apeiron_word.gemm_all_below_loop_body_step
#audit apeiron_word.gemm_all_below_loop_spec
#audit apeiron_word.gemm_all_below_spec
#audit apeiron_word.gemm_limit_val
#audit apeiron_word.gemm_within_bound_spec
#audit apeiron_word.gemm_allBelowRef_iff
#audit apeiron_word.gemm_check_zero_dim
#audit apeiron_word.gemm_usize_bne_true_iff
#audit apeiron_word.gemm_check_overflow_k
#audit apeiron_word.gemm_check_overflow_mk
#audit apeiron_word.gemm_check_overflow_kn
#audit apeiron_word.gemm_check_overflow_nm
#audit apeiron_word.gemm_check_shape_a
#audit apeiron_word.gemm_check_shape_b
#audit apeiron_word.gemm_check_input_bound
#audit apeiron_word.gemm_check_ok
#audit apeiron_word.witness_usize_max_ge
#audit apeiron_word.witness_usize_max_le
#audit apeiron_word.witness_dims_ok
#audit apeiron_word.witness_withinBound_A
#audit apeiron_word.witness_withinBound_Bnn
#audit apeiron_word.witness_withinBound_Bnt
#audit apeiron_word.witness_accepted
#audit apeiron_word.witness_outNn
#audit apeiron_word.witness_outNt
#audit apeiron_word.gemm_witness_joint
#audit apeiron_word.witness_refuse_zero_dim
#audit apeiron_word.witness_refuse_overflow_k
#audit apeiron_word.witness_refuse_overflow_dims
#audit apeiron_word.witness_refuse_shape
#audit apeiron_word.witness_refuse_input_bound

open Lean Elab Command in
/-- Axioms of the `solve_port.syrk` extraction admitted in `proof/boundary-axioms.txt`: the std operation
    `core.option.Option.map` (absent from the pinned Aeneas model) and the thread driver
    `solve_port.syrk.block_rows_pool` (opaque, `tested`). -/
def auditSyrkBoundaryAxioms : Lean.NameSet :=
  auditClosedAxioms |>.insert `apeiron_word.core.option.Option.map
    |>.insert `apeiron_word.solve_port.syrk.block_rows_pool

open Lean Elab Command in
/-- `#audit_boundary foo`: as `#audit`, but the cone may also contain exactly the two named
    `solve_port.syrk` boundary axioms. Any other axiom, `sorryAx` included, is a build failure.
    The headline's statement is free of them; the proof prints them because it unfolds the extracted
    body that calls the boundary (`Learn/Solve/Wsyrk.lean`, "The rounders and the boundary"). -/
elab "#audit_boundary " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditSyrkBoundaryAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms (boundary rows included): {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the cone plus the syrk boundary: {bad.toList}"

-- The extracted `solve_port.syrk` `wsyrk` / `hadamard_into` refusal chain and bound scans refine the `ℤ` model
-- (`Learn/Solve/WsyrkSpec.lean`, `Wsyrk.lean`); `WsyrkWitness.lean` inhabits the domain and each refusal arm.
#audit apeiron_word.rnd64_eq_div
#audit apeiron_word.gsum_succ
#audit apeiron_word.gsum_comm
#audit apeiron_word.wref_none
#audit apeiron_word.wref_some
#audit apeiron_word.gRef_symm
#audit apeiron_word.gramOut_length
#audit apeiron_word.gramOut_getD
#audit apeiron_word.listBound_le
#audit apeiron_word.wfunBound_abs
#audit apeiron_word.gsum_abs_le
#audit apeiron_word.gsum_lt_126
#audit apeiron_word.gsum_i128
#audit apeiron_word.rnd64_add_i128
#audit apeiron_word.rnd64_fits_i64
#audit apeiron_word.gsum_domain
#audit apeiron_word.hadStep_succ
#audit apeiron_word.hadStep_length
#audit apeiron_word.hadStep_getD_lt
#audit apeiron_word.hadStep_getD_self
#audit apeiron_word.syrk_bound_bits_val
#audit apeiron_word.syrk_block_val
#audit apeiron_word.syrk_shift32_toNat
#audit apeiron_word.syrk_shift64_toNat
#audit apeiron_word.u64_ofNat_val
#audit apeiron_word.syrk_i64_natAbs_lt
#audit apeiron_word.uabs_eq
#audit apeiron_word.uabs_val
#audit apeiron_word.boundLimit_val
#audit apeiron_word.syrk_bound_limit
#audit apeiron_word.syrk_shl63
#audit apeiron_word.wsyrkAllBelowRef_congr
#audit apeiron_word.all_below_loop_body_exec_cont
#audit apeiron_word.all_below_loop_body_step
#audit apeiron_word.all_below_loop_spec
#audit apeiron_word.all_below_spec
#audit apeiron_word.syrk_within_bound_true
#audit apeiron_word.syrk_within_bound_false
#audit apeiron_word.syrk_within_bound_spec
#audit apeiron_word.syrk_usize_numBits_gt18
#audit apeiron_word.syrk_max_o
#audit_boundary apeiron_word.syrk_round32_spec
#audit_boundary apeiron_word.syrk_round64_spec
#audit apeiron_word.syrk_i64max_val
#audit apeiron_word.syrk_i64min_val
#audit apeiron_word.syrk_narrow_ok
#audit apeiron_word.syrk_narrow_none_hi
#audit apeiron_word.syrk_narrow_none_lo
#audit apeiron_word.qvals_set
#audit_boundary apeiron_word.syrk_hadamard_body_step
#audit_boundary apeiron_word.syrk_hadamard_loop_spec
#audit_boundary apeiron_word.word_hadamard_ok
#audit_boundary apeiron_word.word_hadamard_refuses
#audit apeiron_word.syrk_bne_len_x
#audit_boundary apeiron_word.syrk_wsyrk_refuse_zeroN
#audit_boundary apeiron_word.syrk_wsyrk_refuse_zeroO
#audit_boundary apeiron_word.syrk_wsyrk_refuse_mulNN
#audit_boundary apeiron_word.syrk_wsyrk_refuse_mulON
#audit_boundary apeiron_word.syrk_wsyrk_refuse_len_x
#audit_boundary apeiron_word.syrk_wsyrk_refuse_len_w
#audit_boundary apeiron_word.syrk_wsyrk_refuse_maxO
#audit_boundary apeiron_word.syrk_wsyrk_refuse_bound
#audit_boundary apeiron_word.word_wsyrk_refuses
#audit apeiron_word.wBigList_len
#audit apeiron_word.wBigList_le
#audit apeiron_word.w_nil_len
#audit apeiron_word.w_one_len
#audit apeiron_word.w_big_len_val
#audit apeiron_word.w_big_ge18
#audit apeiron_word.w_one_ne_zero
#audit apeiron_word.w_two_ne_zero
#audit apeiron_word.w_zero_ne_one
#audit apeiron_word.w_max_ne_zero
#audit apeiron_word.w_big_ne_zero
#audit apeiron_word.w_pin_mul
#audit apeiron_word.w_mul_one
#audit apeiron_word.w_mul11
#audit apeiron_word.w_mul22
#audit apeiron_word.w_max_max_none
#audit apeiron_word.w_max_two_none
#audit apeiron_word.w_nil_len_ne_four
#audit apeiron_word.w_nil_len_ne_one
#audit apeiron_word.w_getd_one
#audit apeiron_word.w_getd_wide
#audit apeiron_word.w_i64max_val
#audit apeiron_word.w_i64zero_val
#audit apeiron_word.w_abs_zero
#audit apeiron_word.w_abs_wide
#audit apeiron_word.w_one_val_len
#audit apeiron_word.w_wide_len
#audit apeiron_word.w_wide_val_len
#audit apeiron_word.w_one_bound
#audit apeiron_word.w_wide_not_below
#audit apeiron_word.w_wide_not_bound
#audit apeiron_word.wsyrkDomain_witness
#audit apeiron_word.within_bound_witness
#audit apeiron_word.hadFit_witness
#audit_boundary apeiron_word.wsyrk_refuse_zeroN_witness
#audit_boundary apeiron_word.wsyrk_refuse_zeroO_witness
#audit_boundary apeiron_word.wsyrk_refuse_mulNN_witness
#audit_boundary apeiron_word.wsyrk_refuse_mulON_witness
#audit_boundary apeiron_word.wsyrk_refuse_len_x_witness
#audit_boundary apeiron_word.wsyrk_refuse_len_w_witness
#audit_boundary apeiron_word.wsyrk_refuse_maxO_witness
#audit_boundary apeiron_word.wsyrk_refuse_bound_witness
#audit_boundary apeiron_word.wsyrk_refuses_witness
#audit_boundary apeiron_word.hadamard_refuses_witness

open Lean Elab Command in
/-- Axioms of the `solve_port.chol` extraction admitted in `proof/boundary-axioms.txt`: the engine-pool
    dispatch `chart.pool.chunks_mut_global`, its width query, and the refusal cell
    (`FirstErr.new`/`record`/`take`), all opaque with the rest of `crate::chart::pool` (`tested`), plus
    the std operation `core.option.Option.is_none_or` (absent from the pinned Aeneas model).
    The per-column serial arithmetic (`solve_cols`) calls none of them and audits closed. -/
def auditCholBoundaryAxioms : Lean.NameSet :=
  auditClosedAxioms |>.insert `apeiron_word.chart.pool.chunks_mut_global
    |>.insert `apeiron_word.chart.pool.width_global
    |>.insert `apeiron_word.chart.pool.FirstErr
    |>.insert `apeiron_word.chart.pool.FirstErr.new
    |>.insert `apeiron_word.chart.pool.FirstErr.record
    |>.insert `apeiron_word.chart.pool.FirstErr.take
    |>.insert `apeiron_word.core.option.Option.is_none_or

open Lean Elab Command in
/-- `#audit_chol_boundary foo`: as `#audit`, but the cone may also contain exactly the registered
    `chart.pool` rows the pooled `chol::solve_rhs` body calls. Any other axiom, `sorryAx` included,
    is a build failure. -/
elab "#audit_chol_boundary " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !auditCholBoundaryAxioms.contains a)
    if bad.isEmpty then
      logInfo m!"'{c}' depends on axioms (chol boundary rows included): {axioms.toList}"
    else
      throwError m!"'{c}' depends on axioms outside the cone plus the chol boundary: {bad.toList}"

-- The pooled `solve_port.chol` right-hand-side solve: `solve_cols` (the per-column serial body) is
-- extracted in full; `solve_rhs` adds the dispatch, the serial-fallback threshold and the transpose,
-- and reaches only the registered `chart.pool` rows. The bit-identity of the split is the
-- `learn_fold_gptq_pool` digest (`tested`), not a theorem.
#audit apeiron_word.solve_port.chol.solve_cols
#audit_chol_boundary apeiron_word.solve_port.chol.solve_rhs
#audit_chol_boundary apeiron_word.solve_port.chol.RhsColumns.Insts.Apeiron_wordChartTaskChunkTaskI64.run

-- ── Bounded-surface epic: the table, hashtrinity and fold-fit bodies rooted by the last waves ──

-- Table locator bodies (`src/map/table/locator.rs`, `include!`d into `table`): path plan, refusal, bound, locator methods and loops.
#audit apeiron_word.table.live_locations_of
#audit apeiron_word.table.live_locations_of_loop0
#audit apeiron_word.table.live_locations_of_loop0.body
#audit apeiron_word.table.live_locations_of_loop1
#audit apeiron_word.table.live_locations_of_loop1.body
#audit apeiron_word.table.locator_bound
#audit apeiron_word.table.locator_path_plan
#audit apeiron_word.table.locator_refusal
#audit apeiron_word.table.observe_locator_receipt
#audit apeiron_word.table.payload_of
#audit apeiron_word.table.push_live_spill
#audit apeiron_word.table.push_live_spill_loop
#audit apeiron_word.table.push_live_spill_loop.body
#audit apeiron_word.table.query_identity_bytes_equal
#audit apeiron_word.table.query_identity_bytes_equal_loop
#audit apeiron_word.table.query_identity_bytes_equal_loop.body
#audit apeiron_word.table.QueryIdentity.Insts.CoreCmpEq
#audit apeiron_word.table.QueryIdentity.Insts.CoreCmpPartialEqQueryIdentity.eq
#audit apeiron_word.table.same_location
#audit apeiron_word.table.Table.assemble_directory
#audit apeiron_word.table.Table.assemble_directory_loop
#audit apeiron_word.table.Table.assemble_directory_loop.body
#audit apeiron_word.table.TableLocator.linked_handle
#audit apeiron_word.table.TableLocator.profile_or_init
#audit apeiron_word.table.TableLocator.publish_for_query
#audit apeiron_word.table.TableLocator.record_admitted_link
#audit apeiron_word.table.TableLocator.record_link
#audit apeiron_word.table.TableLocator.reserve_batch_paths
#audit apeiron_word.table.TableLocator.reserve_batch_paths.closure
#audit apeiron_word.table.TableLocator.reserve_bytes
#audit apeiron_word.table.TableLocator.reserve_path
#audit apeiron_word.table.TableLocator.reserve_path.closure
#audit apeiron_word.table.TableLocator.retarget_held
#audit apeiron_word.table.Table.reserve_locator_batch
#audit apeiron_word.table.Table.reserve_locator_path

-- Table state bodies (`src/map/table/state.rs`): Region/Flat/SearchDecision, `matches`, `slots_for`, `valid_n`, `capacity_bound`.
#audit apeiron_word.table.ASSUMED_SLOT_BYTES
#audit apeiron_word.table.capacity_bound
#audit apeiron_word.table.COORD_EMPTY
#audit apeiron_word.table.Flat.byte_len_for
#audit apeiron_word.table.Flat.empty
#audit apeiron_word.table.flat_index
#audit apeiron_word.table.Flat.index
#audit apeiron_word.table.Flat.initialize
#audit apeiron_word.table.Flat.initialize_loop
#audit apeiron_word.table.Flat.initialize_loop.body
#audit apeiron_word.table.Flat.reserve
#audit apeiron_word.table.HOME_SPREAD
#audit apeiron_word.table.matches
#audit apeiron_word.table.Region.bound
#audit apeiron_word.table.Region.coord_of
#audit apeiron_word.table.Region.empty
#audit apeiron_word.table.Region.home
#audit apeiron_word.table.Region.initialize
#audit apeiron_word.table.Region.initialize_loop0
#audit apeiron_word.table.Region.initialize_loop0.body
#audit apeiron_word.table.Region.initialize_loop1
#audit apeiron_word.table.Region.initialize_loop1.body
#audit apeiron_word.table.Region.initialize_loop2
#audit apeiron_word.table.Region.initialize_loop2.body
#audit apeiron_word.table.Region.publish_live
#audit apeiron_word.table.Region.reserve_empty
#audit apeiron_word.table.SearchClass.Insts.CoreCmpPartialEqSearchClass.eq
#audit apeiron_word.table.SearchDecision.found_flat
#audit apeiron_word.table.SearchDecision.from_spill
#audit apeiron_word.table.SearchDecision.vacant_flat
#audit apeiron_word.table.shard
#audit apeiron_word.table.slots_for
#audit apeiron_word.table.slots_for_loop
#audit apeiron_word.table.slots_for_loop.body
#audit apeiron_word.table.SpillSlot.live
#audit apeiron_word.table.TableLocator.new
#audit apeiron_word.table.tag
#audit apeiron_word.table.valid_n

-- Table prepared and exact-read bodies (`src/map/table/prepared.rs`, `getexact.rs`): reservation, placement, spill accounting.
#audit apeiron_word.table.account_spill_load
#audit apeiron_word.table.apply_prepared_shard
#audit apeiron_word.table.apply_prepared_shard_placed
#audit apeiron_word.table.get_prepared_shard
#audit apeiron_word.table.increment_spill_load
#audit apeiron_word.table.initialize_reserved_regions
#audit apeiron_word.table.initialize_reserved_regions_loop
#audit apeiron_word.table.initialize_reserved_regions_loop.body
#audit apeiron_word.table.mark_selected_occupied
#audit apeiron_word.table.reserve_plan_regions
#audit apeiron_word.table.reserve_plan_regions_loop
#audit apeiron_word.table.reserve_plan_regions_loop.body
#audit apeiron_word.table.Table.get_exact_spill
#audit apeiron_word.table.Table.resident_exact

-- Table parallel bodies (`src/map/table/parallel.rs`) and the eight `table_port` word roots that call them.
#audit apeiron_word.table.parallel.distribute_regions
#audit apeiron_word.table.parallel.distribute_regions_loop0
#audit apeiron_word.table.parallel.distribute_regions_loop0.body
#audit apeiron_word.table.parallel.distribute_regions_loop0_loop0
#audit apeiron_word.table.parallel.distribute_regions_loop0_loop0.body
#audit apeiron_word.table.parallel.PhaseNanos.absorb
#audit apeiron_word.table.parallel.PhaseNanos.total
#audit apeiron_word.table.parallel.reserve_all_shards
#audit apeiron_word.table.parallel.reserve_all_shards_loop
#audit apeiron_word.table.parallel.reserve_all_shards_loop.body
#audit apeiron_word.table.parallel.reserve_placement_records
#audit apeiron_word.table.parallel.reserve_placement_records_loop
#audit apeiron_word.table.parallel.reserve_placement_records_loop.body
#audit apeiron_word.table.parallel.reserve_worker_outcomes
#audit apeiron_word.table.parallel.reserve_worker_outcomes_loop
#audit apeiron_word.table.parallel.reserve_worker_outcomes_loop.body
#audit apeiron_word.table.parallel.schedule_bins_with_regions
#audit apeiron_word.table.parallel.schedule_bins_with_regions_loop0
#audit apeiron_word.table.parallel.schedule_bins_with_regions_loop0.body
#audit apeiron_word.table.parallel.schedule_bins_with_regions_loop0_loop0
#audit apeiron_word.table.parallel.schedule_bins_with_regions_loop0_loop0.body
#audit apeiron_word.table_port.word_table_parallel_distribute_regions
#audit apeiron_word.table_port.word_table_parallel_phase_absorb
#audit apeiron_word.table_port.word_table_parallel_phase_total
#audit apeiron_word.table_port.word_table_parallel_reserve_all_shards
#audit apeiron_word.table_port.word_table_parallel_reserve_placement_records
#audit apeiron_word.table_port.word_table_parallel_reserve_worker_outcomes
#audit apeiron_word.table_port.word_table_parallel_schedule_bins_with_regions

-- Hashtrinity algebra roots (`hashtrinity_port::word_alg_*`, nine wrappers over the word table).
#audit apeiron_word.hashtrinity_port.word_alg_fold
#audit apeiron_word.hashtrinity_port.word_alg_has
#audit apeiron_word.hashtrinity_port.word_alg_keep
#audit apeiron_word.hashtrinity_port.word_alg_merge_count
#audit apeiron_word.hashtrinity_port.word_alg_put
#audit apeiron_word.hashtrinity_port.word_alg_set
#audit apeiron_word.hashtrinity_port.word_alg_species
#audit apeiron_word.hashtrinity_port.word_alg_table_count

-- Hashtrinity freq roots (`hashtrinity_port::word_freq_*`, five wrappers and the `insert_with` closure).
#audit apeiron_word.hashtrinity_port.word_freq_insert
#audit apeiron_word.hashtrinity_port.word_freq_insert_with
#audit apeiron_word.hashtrinity_port.word_freq_insert_with.closure
#audit apeiron_word.hashtrinity_port.word_freq_longest_probe
#audit apeiron_word.hashtrinity_port.word_freq_primary_examinations
#audit apeiron_word.hashtrinity_port.word_freq_stream_with_capacity

-- Crypto wrappers (`crypto_port::word_crypto2_*`, forty roots over big-integer, RSA/PSS, DER/X.509 and base64).
#audit apeiron_word.crypto_port.word_crypto2_bitlen
#audit apeiron_word.crypto_port.word_crypto2_bn_add
#audit apeiron_word.crypto_port.word_crypto2_bn_divmod
#audit apeiron_word.crypto_port.word_crypto2_bn_from_u64
#audit apeiron_word.crypto_port.word_crypto2_bn_is_even
#audit apeiron_word.crypto_port.word_crypto2_bn_is_zero
#audit apeiron_word.crypto_port.word_crypto2_bn_modexp_big
#audit apeiron_word.crypto_port.word_crypto2_bn_shr1
#audit apeiron_word.crypto_port.word_crypto2_bn_sub1
#audit apeiron_word.crypto_port.word_crypto2_der_algid_ed25519
#audit apeiron_word.crypto_port.word_crypto2_der_bitstr
#audit apeiron_word.crypto_port.word_crypto2_der_ctx_explicit
#audit apeiron_word.crypto_port.word_crypto2_der_int
#audit apeiron_word.crypto_port.word_crypto2_der_len
#audit apeiron_word.crypto_port.word_crypto2_der_oid
#audit apeiron_word.crypto_port.word_crypto2_der_seq
#audit apeiron_word.crypto_port.word_crypto2_der_set
#audit apeiron_word.crypto_port.word_crypto2_der_spki
#audit apeiron_word.crypto_port.word_crypto2_der_tlv
#audit apeiron_word.crypto_port.word_crypto2_der_validity
#audit apeiron_word.crypto_port.word_crypto2_emsa_pss_encode
#audit apeiron_word.crypto_port.word_crypto2_mgf1
#audit apeiron_word.crypto_port.word_crypto2_mod_inverse
#audit apeiron_word.crypto_port.word_crypto2_rsa_algid
#audit apeiron_word.crypto_port.word_crypto2_rsa_from_pkcs8_der
#audit apeiron_word.crypto_port.word_crypto2_rsa_to_pkcs8_der
#audit apeiron_word.crypto_port.word_crypto2_trim_int
#audit apeiron_word.crypto_port.word_crypto2_x509_der_children
#audit apeiron_word.crypto_port.word_crypto2_x509_der_tlv

-- Crypto callees the wrappers call directly (`hashtrinity.crypto.{rsa,x509,certgen,jwt,age}.*`).
#audit apeiron_word.hashtrinity.crypto.x509.der_tlv
#audit apeiron_word.hashtrinity.crypto.x509.der_children
#audit apeiron_word.hashtrinity.crypto.rsa.trim_int
#audit apeiron_word.hashtrinity.crypto.rsa.rsa_to_pkcs8_der
#audit apeiron_word.hashtrinity.crypto.rsa.rsa_from_pkcs8_der
#audit apeiron_word.hashtrinity.crypto.rsa.rsa_algid
#audit apeiron_word.hashtrinity.crypto.rsa.mod_inverse
#audit apeiron_word.hashtrinity.crypto.rsa.mgf1
#audit apeiron_word.hashtrinity.crypto.rsa.emsa_pss_encode
#audit apeiron_word.hashtrinity.crypto.rsa.bn_sub1
#audit apeiron_word.hashtrinity.crypto.rsa.bn_shr1
#audit apeiron_word.hashtrinity.crypto.rsa.bn_modexp_big
#audit apeiron_word.hashtrinity.crypto.rsa.bn_is_zero
#audit apeiron_word.hashtrinity.crypto.rsa.bn_is_even
#audit apeiron_word.hashtrinity.crypto.rsa.bn_from_u64
#audit apeiron_word.hashtrinity.crypto.rsa.bn_divmod
#audit apeiron_word.hashtrinity.crypto.rsa.bn_add
#audit apeiron_word.hashtrinity.crypto.rsa.bitlen
#audit apeiron_word.hashtrinity.crypto.certgen.validity
#audit apeiron_word.hashtrinity.crypto.certgen.tlv
#audit apeiron_word.hashtrinity.crypto.certgen.spki
#audit apeiron_word.hashtrinity.crypto.certgen.set
#audit apeiron_word.hashtrinity.crypto.certgen.seq
#audit apeiron_word.hashtrinity.crypto.certgen.oid
#audit apeiron_word.hashtrinity.crypto.certgen.der_len
#audit apeiron_word.hashtrinity.crypto.certgen.der_int
#audit apeiron_word.hashtrinity.crypto.certgen.ctx_explicit
#audit apeiron_word.hashtrinity.crypto.certgen.bitstr
#audit apeiron_word.hashtrinity.crypto.certgen.algid_ed25519

-- Fold fit ICM roots (`fit_port::word_fit_icm_*`, eight wrappers over the `fold_port.fit.icm` bodies).
#audit apeiron_word.fit_port.word_fit_icm_codebook
#audit apeiron_word.fit_port.word_fit_icm_nearest_index
#audit apeiron_word.fit_port.word_fit_icm_plane_w

/- Open cones, 2026-10-19: these extracted defs are not audited because each depends on an
   unmodeled external. Re-enable the `#audit` line once the external is modeled or the source
   avoids it. The finding is the coverage root that pulled it in, not the name. -/
-- #audit apeiron_word.table.parallel.reserve_handle_table  -- apeiron_word.core.option.Option.Insts.CoreCloneClone.clone, apeiron_word.core.option.Option.Insts.CoreCloneClone.clone_from
-- #audit apeiron_word.table_port.word_table_parallel_reserve_handle_table  -- apeiron_word.core.option.Option.Insts.CoreCloneClone.clone, apeiron_word.core.option.Option.Insts.CoreCloneClone.clone_from
-- #audit apeiron_word.hashtrinity_port.word_alg_merge_link  -- apeiron_word.alloc.vec.Vec.Insts.CoreIterTraitsCollectExtend.extend
-- #audit apeiron_word.crypto_port.word_crypto2_age_b64d  -- apeiron_word.core.str.Str.as_bytes
-- #audit apeiron_word.crypto_port.word_crypto2_der_name_cn  -- apeiron_word.core.str.Str.as_bytes
-- #audit apeiron_word.crypto_port.word_crypto2_jwt_b64url_decode  -- apeiron_word.core.str.Str.as_bytes
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pkcs1_sha256_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pkcs1_sha384_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pkcs1_sha384_verify  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pkcs1_sha512_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pkcs1_sha512_verify  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pkcs1_sign_generic  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pss_sha256_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.crypto_port.word_crypto2_rsa_pss_sha256_verify  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pss_sha256_verify  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pss_sha256_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pkcs1_sign_generic  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pkcs1_sha512_verify  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pkcs1_sha512_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pkcs1_sha384_verify  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pkcs1_sha384_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.rsa.rsa_pkcs1_sha256_sign  -- Aeneas.Std.core.fmt.Formatter
-- #audit apeiron_word.hashtrinity.crypto.jwt.b64url_decode  -- apeiron_word.core.str.Str.as_bytes
-- #audit apeiron_word.hashtrinity.crypto.certgen.name_cn  -- apeiron_word.core.str.Str.as_bytes
-- #audit apeiron_word.hashtrinity.crypto.age.b64d  -- apeiron_word.core.str.Str.as_bytes
-- Open: the extracted `fit::icm::e8_icm_step` body leaves the closed cone on 31 `axiom`s — the pool
-- (`chart.pool.width_global`, `.run_global`, `.chunks_mut_global`, `.Flag`, `.chunks.for_chunks*`),
-- the AVX2 surface (`chart.x86.*`, `chart.q8_0_row_kcols_avx2`, `kern.matmat.q5_0_block_kdots_avx2`,
-- the four `…FastBlockDecoder.decode_step_avx2` instances), the raw-pointer/slice allocators and
-- `core.option.Option.map`. The conditional refinement is
-- `Learn.FitIcm.icm_step_spec_of_body_argmin` (audited above); enable this line when a pool-free
-- `best_code_q` is rooted.
-- Open: the extracted `fit::icm::e8_icm_step` body leaves the closed cone on 31 `axiom`s — the pool
-- (`chart.pool.width_global`, `.run_global`, `.chunks_mut_global`, `.Flag`, `.chunks.for_chunks*`),
-- the AVX2 surface (`chart.x86.*`, `chart.q8_0_row_kcols_avx2`, `kern.matmat.q5_0_block_kdots_avx2`,
-- the four `…FastBlockDecoder.decode_step_avx2` instances), the raw-pointer/slice allocators and
-- `core.option.Option.map`. The conditional refinement is
-- `Learn.FitIcm.icm_step_spec_of_body_argmin` (audited above); enable this line when a pool-free
-- `best_code_q` is rooted.
-- #audit apeiron_word.fit_port.word_fit_icm_e8_icm_step  -- apeiron_word.chart.pool.width_global, apeiron_word.solve_port.gemm.block_rows_pool, apeiron_word.chart.pool.chunks.for_chunks3, apeiron_word.core.option.Option.map
-- #audit apeiron_word.fit_port.word_fit_icm_j_rows  -- apeiron_word.solve_port.gemm.block_rows_pool
-- #audit apeiron_word.fit_port.word_fit_icm_matmul  -- apeiron_word.solve_port.gemm.block_rows_pool
-- #audit apeiron_word.fit_port.word_fit_icm_ridged_metric  -- apeiron_word.core.option.Option.map
-- #audit apeiron_word.fit_port.word_fit_icm_r_step2  -- apeiron_word.chart.pool.FirstErr, apeiron_word.chart.pool.chunks_mut_global, apeiron_word.chart.pool.width_global, apeiron_word.solve_port.chol.trailing_update_pool, apeiron_word.solve_port.gemm.block_rows_pool, apeiron_word.chart.pool.FirstErr.new, apeiron_word.chart.pool.FirstErr.record, apeiron_word.chart.pool.FirstErr.take, apeiron_word.chart.pool.chunks.for_chunks2, apeiron_word.core.option.Option.is_none_or, apeiron_word.core.option.Option.map

-- The extracted reverse water-filling solver (`Learn/WaterFill.lean`): the integer level,
-- active-set and budget invariants, the exact denotation onto the Gaussian floor, and the
-- instantiated `waterfill_floor` at `c = 2 ln 2`.
#audit Learn.WaterFill.distWords_le_sum
#audit Learn.WaterFill.distWords_le_card_mul
#audit Learn.WaterFill.distWords_mono
#audit Learn.WaterFill.distWords_active_set
#audit Learn.WaterFill.rateSum_anti
#audit Learn.WaterFill.rateWords_anti
#audit Learn.WaterFill.rateWords_floor
#audit Learn.WaterFill.distWords_denote
#audit Learn.WaterFill.distWords_denote_scale
#audit Learn.WaterFill.gaussian_floor
#audit Learn.WaterFill.gaussian_floor_words
#audit Learn.WaterFill.witness_fields_4
#audit Learn.WaterFill.witness_fields_1
#audit Learn.WaterFill.witness_active_set
#audit Learn.WaterFill.early_witness
-- The distortion loop refinement: `distortion_words_loop` to the integer fold by
-- `loop.spec_decr_nat`, and `distortion_words` through `narrow`/`ok_or` on the stated `i128` window.
#audit Learn.WaterFill.distList_nonneg
#audit Learn.WaterFill.distList_eq_distWords
#audit Learn.WaterFill.slice_next_eq
#audit Learn.WaterFill.body_eq_cont
#audit Learn.WaterFill.body_eq_done
#audit Learn.WaterFill.distortion_words_loop_spec
#audit Learn.WaterFill.distortion_words_eq
#audit Learn.WaterFill.distortion_words_spec
-- The rate loop refinement: `rate_at_logs_loop` to `rateSum` by `loop.spec_decr_nat`, on the
-- no-overflow window for the `i64` subtraction and the `i128` accumulator window.
#audit Learn.WaterFill.checked_sub_some_of_window
#audit Learn.WaterFill.i128_add_from
#audit Learn.WaterFill.rate_at_logs_loop_spec
#audit Learn.WaterFill.rate_at_logs_eq
#audit Learn.WaterFill.rate_at_logs_loop_zero
#audit Learn.WaterFill.rate_at_logs_spec
-- The bisection's sum loop: `water_fill_words_loop1` to the prefix sum by `loop.spec_decr_nat` on
-- the same `i128` window, and that sum read as `distWords` at the maximum level.
#audit Learn.WaterFill.vals_sum_nonneg
#audit Learn.WaterFill.lamSuffix_nonneg
#audit Learn.WaterFill.sum_body_eq
#audit Learn.WaterFill.sum_body_eq_done
#audit Learn.WaterFill.water_fill_words_loop1_spec
#audit Learn.WaterFill.water_fill_words_loop1_zero
#audit Learn.WaterFill.le_maxOf
#audit Learn.WaterFill.sum_min_eq
#audit Learn.WaterFill.distWords_maxOf
#audit Learn.WaterFill.water_fill_words_loop1_distWords

-- The concrete witness slices built from the codebook (`q = 2^35` on the 240 exact rows,
-- `dot = Σ_p C_j(p)`): the row identity, the correspondence on those rows, and the argmin transfer.
-- The 16 `O8` rows are excluded by `not_dvd_sqrt8_sq`; no `i64` `q` satisfies the row identity there.
#audit Learn.FitIcm.block_cost_ok_bounded
#audit Learn.FitIcm.codewordSpec_sq_sum
#audit Learn.FitIcm.wordCostCorresponds_row
#audit Learn.FitIcm.witnessWordCostCorrespondsOn
#audit Learn.FitIcm.best_code_q_argmin_real_on
#audit Learn.FitIcm.witnessArgmin
#audit Learn.FitIcm.witnessArgmin_le
#audit Learn.FitIcm.not_dvd_sqrt8_sq
#audit Learn.FitIcm.o8_q_not_integral

-- The O8 separation and the no-error post: `best_code_q_loop_noerr` removes `hbest`; the 16 O8
-- rows' real `blockReduce` is nonnegative (`diagBlock_reduce_o8_nonneg`), so the full 256-row real
-- argmin is row 0 (`witnessArgmin_all`); the step spec (`witnessStepSpec`) still carries `i < 240`.
#audit Learn.FitIcm.best_code_q_loop_noerr
#audit Learn.FitIcm.diagBlock_reduce_o8_nonneg
#audit Learn.FitIcm.o8_sq_sum
#audit Learn.FitIcm.o8_sum_bounds
#audit Learn.FitIcm.witnessArgmin_all
#audit Learn.FitIcm.witness_best_code_q_ok
#audit Learn.FitIcm.witness_block_cost_ok
#audit Learn.FitIcm.witnessDot_abs_le
#audit Learn.FitIcm.witnessQ_abs_le
#audit Learn.FitIcm.witnessQ_val_all
#audit Learn.FitIcm.witnessStepSpec

-- The step spec is unconditional on the witness: `o8_block_cost_nonneg` gives the O8 word cost ≥ 0,
-- `witnessArgmin_index_lt` derives `i < 240` from the extracted comparison against row 0's -2^96·8,
-- and `witnessStepSpec_exists` composes the existence with the step spec.
#audit Learn.FitIcm.blockReduce_zero
#audit Learn.FitIcm.o8_block_cost_nonneg
#audit Learn.FitIcm.o8_sum_bounds_Z
#audit Learn.FitIcm.witnessArgmin_index_lt
#audit Learn.FitIcm.witnessStepSpec_exists

-- The sweep: `witnessSweep_le`/`witnessSweep_lt` carry the per-block step spec over a `Finset`
-- index set (`sweepTotal`), and `witnessSweep_lt_o8` is the concrete strict case against the O8 row.
#audit Learn.FitIcm.witnessSweep_le
#audit Learn.FitIcm.witnessSweep_lt
#audit Learn.FitIcm.witnessSweep_lt_o8

-- Water-fill loop 0 (the max/min/log prep pass): `water_fill_words_loop0_spec` is the loop's
-- postcondition and `water_fill_words_loop0_zero` its concrete zero-initialized instance; the
-- supporting max/min/log lemmas are audited with it.
#audit Learn.WaterFill.logsOf_set
#audit Learn.WaterFill.maxOf_append_singleton
#audit Learn.WaterFill.maxOf_take_succ
#audit Learn.WaterFill.maxStep_val
#audit Learn.WaterFill.minPosOf_append_singleton
#audit Learn.WaterFill.minPosOf_take_succ
#audit Learn.WaterFill.minStep_val
#audit Learn.WaterFill.prep_body_eq
#audit Learn.WaterFill.prep_body_eq_done
#audit Learn.WaterFill.vals_set
#audit Learn.WaterFill.water_fill_words_loop0_spec
#audit Learn.WaterFill.water_fill_words_loop0_zero

-- The residual-update probe: `residualStep_comp` shows two steps add their codes; `twoPass_quad` and
-- `twoPass_le_iff` give the exact second-pass identity and its one condition; `twoPass_not_le`
-- exhibits the failure at the exhausted residual (one-pass 0, two-pass 32); `blockQuad_residual_energy`
-- identifies the block cost with the residual energy. This refutes the subtractive two-pass model;
-- the algorithm's state update is the reconstruction delta (`sweep_rows`), not a second subtraction.
#audit Learn.FitIcm.blockQuad_diagBlock
#audit Learn.FitIcm.blockQuad_residual_energy
#audit Learn.FitIcm.residualStep_comp
#audit Learn.FitIcm.twoPass_le_iff
#audit Learn.FitIcm.twoPass_not_le
#audit Learn.FitIcm.twoPass_quad
#audit Learn.FitIcm.witness_onePass_zero

-- The shipped delta update: `stateResid`/`blockGrad`/`stateQuad` define the state;
-- `stateResid_update`, `blockGrad_update`/`blockGrad_update_code` and `jointGrad_update_left` prove
-- the gradient invariant; `state_step_descends` and `sweep_state_descent{,_lt}` are the
-- coordinate-descent non-increase; `blockReduce_eq_word_cost`/`word_argmin_cost_bound` give the real
-- word correspondence with the Q32.32 error bound as an explicit hypothesis (limb εq/εd open).
#audit Learn.FitIcm.blockGrad_update
#audit Learn.FitIcm.blockGrad_update_code
#audit Learn.FitIcm.blockReduce_eq_word_cost
#audit Learn.FitIcm.jointGrad_update_left
#audit Learn.FitIcm.lin_update
#audit Learn.FitIcm.stateQuad_delta
#audit Learn.FitIcm.stateQuad_eq_resid_form
#audit Learn.FitIcm.stateQuad_eq_stateReduce_add
#audit Learn.FitIcm.stateResid_update
#audit Learn.FitIcm.state_step_descends
#audit Learn.FitIcm.sweep_state_descent
#audit Learn.FitIcm.sweep_state_descent_lt
#audit Learn.FitIcm.word_argmin_cost_bound

-- Water-fill loop 2 (the `t_lo` doubling walk): `water_fill_words_loop2_spec` is the exact
-- three-way post (a feasible probe, an error, or the exhausted walk with `walkProbesBelow`);
-- `water_fill_words_loop2_converged` is loop 3's conditional feasibility premise.
#audit Learn.WaterFill.checked_mul_some_val
#audit Learn.WaterFill.checked_sub_some_val
#audit Learn.WaterFill.l2_body_eq_done
#audit Learn.WaterFill.l2_body_eq_err
#audit Learn.WaterFill.l2_body_eq_probe
#audit Learn.WaterFill.l2_body_eq_walked
#audit Learn.WaterFill.walkGap_succ
#audit Learn.WaterFill.walkLevel_succ
#audit Learn.WaterFill.water_fill_words_loop2_converged
#audit Learn.WaterFill.water_fill_words_loop2_spec

-- The ordered pass: `blockOrder` is the `sweep_rows` parity order; `icmPass = List.foldl blockStep`
-- and `icmPasses` carry `(s, f)`. `rowQuad_update_delta`, `rowQuad_step_le`, `gradState_step` and
-- `codeState_step` preserve the invariant and never raise the row objective; `icmPass_inv_le` and
-- `icmPasses_quad_le` compose it over a pass and over passes; `fblk_eq_data_add_fout` is the
-- sequential `f_blk` under `GradState`.
#audit Learn.FitIcm.blockOrder_even
#audit Learn.FitIcm.blockOrder_length
#audit Learn.FitIcm.blockOrder_nodup
#audit Learn.FitIcm.blockOrder_odd
#audit Learn.FitIcm.codeState_step
#audit Learn.FitIcm.fblk_eq_data_add_fout
#audit Learn.FitIcm.gradState_step
#audit Learn.FitIcm.icmPasses_quad_le
#audit Learn.FitIcm.icmPass_inv_le
#audit Learn.FitIcm.phi_delta
#audit Learn.FitIcm.rowQuad_step_le
#audit Learn.FitIcm.rowQuad_update_delta

-- The joint pair step: `pairPhi_codebook_eq` is the exact cost (`av + bv + 2a₁a₂·cross`, the ridge
-- inside the diagonal blocks); `pairArgmin` is the flat strict-`<` first minimiser over 65536 pairs;
-- `pairFblk1`/`pairFblk2` are the effective data; `planePhi_delta`/`cross_delta`/`pairPhi_delta` and
-- `jointQuad_update_delta` the exact delta; `jointQuad_step_le` and `pairPass{es}_inv_le` the
-- non-increase over the step and the pass.
#audit Learn.FitIcm.cross_delta
#audit Learn.FitIcm.crossForm_H21
#audit Learn.FitIcm.grad_step_generic
#audit Learn.FitIcm.jointGradState_step
#audit Learn.FitIcm.jointQuad_delta_form
#audit Learn.FitIcm.jointQuad_step_le
#audit Learn.FitIcm.jointQuad_update_delta
#audit Learn.FitIcm.pairArgmin_le
#audit Learn.FitIcm.pairArgmin_spec
#audit Learn.FitIcm.pairFblk1_sub
#audit Learn.FitIcm.pairFblk2_sub
#audit Learn.FitIcm.pairPasses_inv
#audit Learn.FitIcm.pairPasses_quad_le
#audit Learn.FitIcm.pairPass_inv_le
#audit Learn.FitIcm.pairPhi_codebook_eq
#audit Learn.FitIcm.pairPhi_delta
#audit Learn.FitIcm.planePhi_delta
#audit Learn.FitIcm.twoCodeState_step
#audit Learn.FitIcm.twoInv_step

-- Water-fill loop 3 (the bisection midpoint loop): `FeasibleAt`/`ConvergedAt`/`RejectedAt`/
-- `BisectBracket`; `water_fill_words_loop3_spec` is the unconditional three-way post (converged,
-- error, or the exhausted last probe with `RejectedAt` or the zero-length `t0` exit), and
-- `water_fill_words_loop3_converged` is the no-rejection corollary. `t_hi` is not certified
-- infeasible; `hconv` is the `converged` call premise.
#audit Learn.WaterFill.bmod2_parity
#audit Learn.WaterFill.bmod_small_two_w
#audit Learn.WaterFill.div2_bounds
#audit Learn.WaterFill.div2_succ_le_of_odd
#audit Learn.WaterFill.i128_add_one_spec
#audit Learn.WaterFill.i128_and_one_val
#audit Learn.WaterFill.I128_min_val
#audit Learn.WaterFill.I64_max_val
#audit Learn.WaterFill.I64_min_val
#audit Learn.WaterFill.l3_body_eq_done
#audit Learn.WaterFill.l3_body_eq_doneFlag
#audit Learn.WaterFill.l3_body_eq_err
#audit Learn.WaterFill.l3_body_eq_probe
#audit Learn.WaterFill.l3_body_eq_probe_neg_even
#audit Learn.WaterFill.l3_body_eq_probe_odd
#audit Learn.WaterFill.l3_body_eq_probe_pos
#audit Learn.WaterFill.l3_tail_run
#audit Learn.WaterFill.le_div2_succ
#audit Learn.WaterFill.shiftRight_one_spec
#audit Learn.WaterFill.water_fill_words_loop3_converged
#audit Learn.WaterFill.water_fill_words_loop3_spec

-- The annealed pair step: `pairWeight`/`pairZ`/`pairLaw` and the Gibbs variational principle
-- (`pairFree_ge`, equality `pairFree_law_eq` through `pairFree_sub_law` = KL/β); the integer table's
-- law and its free-energy gap (`intLaw_rel`, `pairFree_gap_le`), the clamp as a clipped cost
-- (`tableLaw_eq_pairLaw_of_unclamped`), and the sampling gap with `hP : P = intLaw w` explicit.
#audit Learn.FitIcm.abs_div_sub_div_le
#audit Learn.FitIcm.abs_mul_sub_mul_le
#audit Learn.FitIcm.div_le_rel_aux
#audit Learn.FitIcm.div_sq_le_aux
#audit Learn.FitIcm.draw_gap_le
#audit Learn.FitIcm.exp_abs_sub_le
#audit Learn.FitIcm.intLaw_rel
#audit Learn.FitIcm.intLaw_sum_one
#audit Learn.FitIcm.klDiv_le_of_rel
#audit Learn.FitIcm.klDiv_nonneg_of_nonneg
#audit Learn.FitIcm.klDiv_nonneg_of_pos
#audit Learn.FitIcm.klDiv_pairLaw
#audit Learn.FitIcm.klDiv_pairLaw_self
#audit Learn.FitIcm.klTerm_ge
#audit Learn.FitIcm.log_pairLaw
#audit Learn.FitIcm.pairFree_gap_le
#audit Learn.FitIcm.pairFree_ge
#audit Learn.FitIcm.pairFree_law_eq
#audit Learn.FitIcm.pairFree_law_mul_beta
#audit Learn.FitIcm.pairFree_mul_beta
#audit Learn.FitIcm.pairFree_sub_law
#audit Learn.FitIcm.pairLaw_pos
#audit Learn.FitIcm.pairLaw_scale
#audit Learn.FitIcm.pairLaw_shift
#audit Learn.FitIcm.pairLaw_sum_one
#audit Learn.FitIcm.pairWeight_pos
#audit Learn.FitIcm.pairWeight_shift
#audit Learn.FitIcm.pairZ_pos
#audit Learn.FitIcm.pairZ_shift
#audit Learn.FitIcm.sq_le_sq_aux
#audit Learn.FitIcm.tableLaw_eq_pairLaw_of_unclamped
#audit Learn.FitIcm.tableLaw_free_ge
#audit Learn.FitIcm.tableLaw_pos
#audit Learn.FitIcm.tableLaw_sum_one
#audit Learn.FitIcm.weight_pos_of_rel
#audit Learn.FitIcm.weight_sum_ge
#audit Learn.FitIcm.weight_sum_pos

-- The measured word bound at the Down peaks: `word_argmin_cost_bound_down_const` is the exact
-- constant `102492/2^32`, and `word_argmin_cost_bound_down` carries it through the argmin
-- comparison at `aa = 1`, `two_a = 2`; the measurement (12/182 words, 1.75e-7) is a docstring.
#audit Learn.FitIcm.word_argmin_cost_bound_down
#audit Learn.FitIcm.word_argmin_cost_bound_down_const

-- The water tail: `distortion_words` exact (nonpositive, post, overflow) and `rate_words` exact
-- under the `log2` call hypotheses (`rateAt`, `rateBad`, `rate_words_loop0_spec`,
-- `rate_words_loop1_spec`, `rate_words_pos_{none,eq,spec,fail}`, `rate_words_nonpos`); the real
-- `½ log₂(λ/θ)` value stays conditional on the unrefined `log2`/`fixed_ln` bound.
#audit Learn.WaterFill.anyPos
#audit Learn.WaterFill.anyPos_append_singleton
#audit Learn.WaterFill.anyPos_nil
#audit Learn.WaterFill.checked_sub_some_window
#audit Learn.WaterFill.distortion_words_nonpos
#audit Learn.WaterFill.distortion_words_overflow
#audit Learn.WaterFill.distortion_words_post
#audit Learn.WaterFill.i128_add_from'
#audit Learn.WaterFill.l0_body_eq_cont_neg
#audit Learn.WaterFill.l0_body_eq_cont_pos
#audit Learn.WaterFill.l0_body_eq_done
#audit Learn.WaterFill.rate_and_then
#audit Learn.WaterFill.rateAt
#audit Learn.WaterFill.rateAt_append
#audit Learn.WaterFill.rateAt_cons
#audit Learn.WaterFill.rateAt_drop_le
#audit Learn.WaterFill.rateAt_nil
#audit Learn.WaterFill.rateAt_nonneg
#audit Learn.WaterFill.rateBad
#audit Learn.WaterFill.rateBad_append_iff
#audit Learn.WaterFill.rateBad_append_left
#audit Learn.WaterFill.rateBad_append_right
#audit Learn.WaterFill.rateBad_singleton
#audit Learn.WaterFill.rate_words_loop0_spec
#audit Learn.WaterFill.rate_words_loop1_spec
#audit Learn.WaterFill.rate_words_nonpos
#audit Learn.WaterFill.rate_words_pos_eq
#audit Learn.WaterFill.rate_words_pos_fail
#audit Learn.WaterFill.rate_words_pos_none
#audit Learn.WaterFill.rate_words_pos_spec
#audit Learn.WaterFill.rw_body_eq_done
#audit Learn.WaterFill.rw_body_eq_idle
#audit Learn.WaterFill.rw_body_eq_overflow
#audit Learn.WaterFill.rw_body_eq_set
#audit Learn.WaterFill.rw_body_eq_success

-- The whole `water_fill_words` body, branch by branch: `water_fill_words_spec` is the 18-way
-- conjunction over the sub/len/loop0/empty/max/t≤0/log2/loop2/mul/loop3/exp2/dist/rate branches,
-- with the loop/tail results carried as step equalities; `solve_water_fill_{ok,err}` is the
-- wrapper. Composing those equalities through the loop/tail specs is the next layer.
#audit Learn.WaterFill.branch_err
#audit Learn.WaterFill.branch_err_i64
#audit Learn.WaterFill.branch_ok
#audit Learn.WaterFill.from_residual_same
#audit Learn.WaterFill.solve_water_fill_err
#audit Learn.WaterFill.solve_water_fill_ok
#audit Learn.WaterFill.water_fill_words_dist_err
#audit Learn.WaterFill.water_fill_words_empty
#audit Learn.WaterFill.water_fill_words_exp2_none
#audit Learn.WaterFill.water_fill_words_len_overflow
#audit Learn.WaterFill.water_fill_words_logmax_add_none
#audit Learn.WaterFill.water_fill_words_logmax_none
#audit Learn.WaterFill.water_fill_words_logmin_none
#audit Learn.WaterFill.water_fill_words_logmin_sub_none
#audit Learn.WaterFill.water_fill_words_loop0_err
#audit Learn.WaterFill.water_fill_words_loop2_err
#audit Learn.WaterFill.water_fill_words_loop3_err
#audit Learn.WaterFill.water_fill_words_max_nonpos
#audit Learn.WaterFill.water_fill_words_mul_none
#audit Learn.WaterFill.water_fill_words_ok
#audit Learn.WaterFill.water_fill_words_rate_err
#audit Learn.WaterFill.water_fill_words_spec
#audit Learn.WaterFill.water_fill_words_sub_none
#audit Learn.WaterFill.water_fill_words_t_nonpos_none
#audit Learn.WaterFill.water_fill_words_t_nonpos_some

-- The block-floating limb-product bound: `limb_error_le` (the documented `G`), `rshift32_spec`,
-- `abs_limbPre_sub_exactPre_le`, and the per-call Down specializations `limb_q_error_down`
-- (≤ 1562/2^32) and `limb_d_error_down` (≤ 24842/2^32). These bound one kernel call; the shipped
-- `q_all` is a two-call composite, so they do not bound it end to end.
#audit Learn.Limb.abs_limbPre_sub_exactPre_le
#audit Learn.Limb.cL_peak_values
#audit Learn.Limb.Gwords_d_down
#audit Learn.Limb.Gwords_q_down
#audit Learn.Limb.limb_d_error_down
#audit Learn.Limb.limb_error_le
#audit Learn.Limb.limb_q_error_down
#audit Learn.Limb.rshift32_spec

-- Branch 17 composed: `water_fill_words_ok_composed` derives the loop-0, loop-1,
-- `distortion_words` and `rate_words` step equalities from their specs; the loop-2/loop-3
-- step equalities remain the `hdown` caller fact.
#audit Learn.WaterFill.water_fill_words_ok_composed

-- The limb bound at the shipped path: `limb_q_error_codebook`/`limb_t_error` compose the two
-- `nt_q` calls (`q_all_end2end`, `epsilon_q_end2end`), `fblk_error_le` derives the state term
-- (`dots_end2end`, `dots_end2end_state`, `epsilon_d_end2end_local`); the downstream
-- `word_argmin_cost_bound_end2end{,_const,_local}` carries the end-to-end margin. `εres` (the
-- `sub_q` residual-state invariant) is a hypothesis.
#audit Learn.Limb.dot8_comp_le
#audit Learn.Limb.dot8_error_le
#audit Learn.Limb.dots_end2end
#audit Learn.Limb.dots_end2end_state
#audit Learn.Limb.epsilon_d_end2end_local
#audit Learn.Limb.epsilon_q_end2end
#audit Learn.Limb.fblk_error_le
#audit Learn.Limb.limb_error_le_G
#audit Learn.Limb.limb_q_error_codebook
#audit Learn.Limb.limb_t_error
#audit Learn.Limb.q_all_end2end
#audit Learn.FitIcm.word_argmin_cost_bound_end2end
#audit Learn.FitIcm.word_argmin_cost_bound_end2end_const
#audit Learn.FitIcm.word_argmin_cost_bound_end2end_local

-- The quantiser's word-error chain: `mulQ_spec` (the `mul_q` rounding derived), `resid_step_le`/
-- `resid_iter_le` (the `sub_q` residual recurrence), `limb_error_chg`, `resid_le_derived`,
-- `dots_derived`, and `word_argmin_cost_bound_end2end_derived` at the derived margin
-- (`derived_const_value` at N = 128, εres0 = 0).
#audit Learn.IcmError.abs_tmod_lt
#audit Learn.IcmError.derived_const_value
#audit Learn.IcmError.dots_derived
#audit Learn.IcmError.epsDEnd_local_eq
#audit Learn.IcmError.epsQEnd_eq
#audit Learn.IcmError.eps_step_eq
#audit Learn.IcmError.limb_error_chg
#audit Learn.IcmError.margin_derived_eq
#audit Learn.IcmError.mulQ_spec
#audit Learn.IcmError.resid_iter_le
#audit Learn.IcmError.resid_le_derived
#audit Learn.IcmError.resid_step_le
#audit Learn.IcmError.rne32_int_spec
#audit Learn.IcmError.rneStep_abs_le
#audit Learn.IcmError.word_argmin_cost_bound_end2end_derived
#audit Learn.IcmError.word_argmin_cost_bound_end2end_derived_const

-- The `Ea ≤ 35` state chain (measured `chg` peak at the Down witness): `limb_error_chg_Ea35`
-- (3114/2^32), `eps_step_eq_Ea35` (6229/2^33), `resid_le_derived_Ea35`, `dots_derived_Ea35`,
-- `margin_derived_eq_Ea35`, and the N = 128 / measured N = 1216 margins. The `Ea ≤ 36` chain
-- stays as the input-general fallback.
#audit Learn.IcmError.derived_const_value_Ea35_N1216
#audit Learn.IcmError.derived_const_value_Ea35_N128
#audit Learn.IcmError.dots_derived_Ea35
#audit Learn.IcmError.eps_step_eq_Ea35
#audit Learn.IcmError.limb_error_chg_Ea35
#audit Learn.IcmError.margin_derived_eq_Ea35
#audit Learn.IcmError.resid_le_derived_Ea35
#audit Learn.IcmError.word_argmin_cost_bound_end2end_derived_Ea35
#audit Learn.IcmError.word_argmin_cost_bound_end2end_derived_Ea35_const
#audit Learn.IcmError.word_argmin_cost_bound_end2end_derived_Ea35_measured

-- The per-entry step count: `stepCount_le_blocks`, `stepCount_sweeps_le` (`N ≤ sweeps·blocks`),
-- `stepCount_eq_zero` (no change, no step), `stepCount_sweeps_eq` (tight), and the margin at the
-- structural maximum, including the Down instantiation at 1216.
#audit Learn.IcmCount.down_blocks
#audit Learn.IcmCount.down_step_count
#audit Learn.IcmCount.epsDEnd_le_of_le
#audit Learn.IcmCount.epsF35_le_of_le
#audit Learn.IcmCount.stepCount_eq_zero
#audit Learn.IcmCount.stepCount_le_blocks
#audit Learn.IcmCount.stepCount_sweeps_eq
#audit Learn.IcmCount.stepCount_sweeps_le
#audit Learn.IcmCount.word_argmin_cost_bound_end2end_derived_Ea35_count
#audit Learn.IcmCount.word_argmin_cost_bound_end2end_derived_Ea35_count_down

-- The chain's looseness as a kernel-checked statement: the increment is positive, the base and the
-- Down margin exceed the measured `1.75e-7` at every `N ≥ 1`, and the exact ratios (`> 900×` base,
-- `> 4·10^5×` Down). This is a statement about the bound, not a refutation of the argmin decision.
#audit Learn.IcmCount.chain_base_gt_measured
#audit Learn.IcmCount.chain_base_ratio_gt
#audit Learn.IcmCount.chain_increment_pos
#audit Learn.IcmCount.chain_margin_down_eq
#audit Learn.IcmCount.chain_margin_down_gt_measured
#audit Learn.IcmCount.chain_margin_down_ratio_gt
#audit Learn.IcmCount.chain_margin_gt_measured

-- The water composition's loop layer: the two `log2` closure steps, loop 2 and loop 3 supplied
-- from their specs at `{0, 200}`, and `water_fill_words_ok_loops` (loops derived; `log2`/`exp2`
-- accuracy and the loops' `e = none` remain caller facts).
#audit Learn.WaterFill.closure_1_step
#audit Learn.WaterFill.closure_2_step
#audit Learn.WaterFill.water_fill_words_loop2_at
#audit Learn.WaterFill.water_fill_words_loop3_at
#audit Learn.WaterFill.water_fill_words_ok_loops

-- `water_fill_words_ok_loops_derived`: the loop-2/loop-3 equations are discharged inside the
-- composition from the `_at` conclusions (plus the `e = none` facts); `log2`/`exp2` accuracy is
-- still out of scope.
#audit Learn.WaterFill.water_fill_words_ok_loops_derived

-- `Learn.Log2`: the `basis.log2`/`fixed_ln` real-value constants. The four named composition
-- theorems (`round_div2p32_half`, `round_div_half`, `poly_error`, `fixed_ln_spec`,
-- `log2_spec`) are statements in the module doc, not yet proved.
#audit Learn.Log2.ln2_q32_sharp
#audit Learn.Log2.sqrt2_q62_bound
#audit Learn.Log2.coef_error
#audit Learn.Log2.two_atanh_remainder
#audit Learn.Log2.log_normalization
#audit Learn.Log2.s_abs_le_third
#audit Learn.Log2.s_abs_le_third_of_sqrt2
#audit Learn.QuantVerdicts.signCode_nearest
#audit Learn.QuantVerdicts.selection_ge_argmin
#audit Learn.QuantVerdicts.signCode_gaussian_value
#audit Learn.QuantVerdicts.production_beats_signCode
#audit Learn.QuantVerdicts.r2_group_gain_is_not_production
#audit Learn.QuantVerdicts.ic_kill
#audit Learn.QuantVerdicts.rt_greedy_survives
#audit Learn.QuantVerdicts.qm_kill_same_bits
#audit Learn.QuantVerdicts.qm_argmin_dominates
#audit Learn.QuantVerdicts.qm_waterfill_survives
#audit Learn.QuantVerdicts.waterfill_sign_mixed
#audit Learn.QuantVerdicts.rate_4gb
#audit Learn.QuantVerdicts.target_floor_stale
#audit Learn.QuantVerdicts.rt2_kill
#audit Learn.QuantVerdicts.rt2_ceiling_as_predicted
#audit Learn.QuantVerdicts.side_budget
#audit Learn.QuantVerdicts.block_bit_overruns
#audit Learn.QuantVerdicts.shannon_floor_vs_reference
#audit Learn.QuantVerdicts.wordCostCorresponds_uninhabited
#audit Learn.QuantVerdicts.o8_row_is_argmin
#audit Learn.QuantVerdicts.prior_art_gap
#audit Learn.QuantVerdicts.prior_art_finish_share
#audit Learn.QuantVerdicts.r2_group_defect
#audit Learn.QuantVerdicts.pf_kill
#audit Learn.QuantVerdicts.pf_metric_is_the_lever
#audit Learn.QuantVerdicts.pf_hyperbolic_face_diverges
#audit Learn.QuantVerdicts.pf_waterfill_third
#audit Learn.QuantVerdicts.klicm_kill
#audit Learn.QuantVerdicts.klicm_cross_block_null
#audit Learn.QuantVerdicts.klicm_outside_radius
#audit Learn.QuantVerdicts.klicm_relinearisation_overfits
#audit Learn.QuantVerdicts.schur_face_split_iid
#audit Learn.QuantVerdicts.schur_delta_not_within_sd
#audit Learn.QuantVerdicts.schur_lever_dominated
#audit Learn.QuantVerdicts.schur_closes_under_tenth
#audit Learn.QuantVerdicts.schur_nonnormal_coupling
#audit Learn.QuantVerdicts.klicm2_kill
#audit Learn.QuantVerdicts.klicm2_trust_region_one_block
#audit Learn.QuantVerdicts.klicm2_two_cycle
#audit Learn.QuantVerdicts.klicm2_line_search_partial
#audit Learn.QuantVerdicts.teq_equation_fits
#audit Learn.QuantVerdicts.teq_sweep_dominates
#audit Learn.QuantVerdicts.teq_rungs_layer
#audit Learn.QuantVerdicts.teq_transfer_bar_needs_r5
#audit Learn.QuantVerdicts.attrib_sweep_reproduces
#audit Learn.QuantVerdicts.attrib_slow_kernel_not_dominant
#audit Learn.QuantVerdicts.attrib_largest_terms
#audit Learn.QuantVerdicts.attrib_teq_premise_false
#audit Learn.QuantVerdicts.attrib_control_two_of_three
#audit Learn.QuantVerdicts.sweeplb_down_ratio
#audit Learn.QuantVerdicts.sweeplb_projection_missed
#audit Learn.QuantVerdicts.sweeplb_exact
#audit Learn.QuantVerdicts.sweeplb_flush_residual
#audit Learn.QuantVerdicts.sweeplb_nonsweep_dominates
#audit Learn.QuantVerdicts.pin_bisect_probes
#audit Learn.QuantVerdicts.pin_seed2_one_commit
#audit Learn.QuantVerdicts.pin_oracle_drift
#audit Learn.QuantVerdicts.pin_nb_moved_twice
#audit Learn.QuantVerdicts.pin_defect_fix_forward
