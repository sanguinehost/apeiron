import AxiomAudit.Remaining
import Numeric.Pow2Refinement
import Flux.Discrete
import Count.Helpers

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

#audit apeiron_word.word_f16_split_nonfinite
#audit apeiron_word.word_f16_split_exact
#audit apeiron_word.word_f16_split_finite_witness
#audit apeiron_word.word_f16_split_nonfinite_witness
