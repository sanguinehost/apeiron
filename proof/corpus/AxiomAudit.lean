import AxiomAudit.Remaining
import Numeric.Pow2Refinement
import Flux.Discrete
import Count.Helpers
import Count.FreqRep
import Count.FreqAdd
import Count.Aggregate
import Map.PrefixMasks

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
