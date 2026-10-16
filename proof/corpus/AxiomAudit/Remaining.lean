import AxiomAudit.Core
import Numeric.Quantum.Ledger.AmplitudeKLExact
import Nous.Model.HyperbolicAttention
import Nous.Model.OneBitAttention
import Numeric.OneBit.SigmaDelta
import Numeric.Hadamard.HiddenPerm
import Numeric.OneBit.Units
import Numeric.OneBit.SignPlanes

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open apeiron_word

-- APEIRON-INT-FORMAT-W3-T3: K-quant code unpack — refusal correspondence
#audit apeiron_word.q4CodesRef
#audit apeiron_word.q6CodesRef
#audit apeiron_word.quant_Q4_K_BLOCK_BYTES_eq
#audit apeiron_word.quant_Q6_K_BLOCK_BYTES_eq
#audit apeiron_word.word_q4_k_codes_refuses
#audit apeiron_word.word_q6_k_codes_refuses

-- APEIRON-INT-FORMAT-W3-T6: Q4_K loop invariants
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

-- APEIRON-INT-FORMAT-W3-T7: Q4_K loop specs and bridge
#audit apeiron_word.q4_getD_getElem
#audit apeiron_word.nat_and_15
#audit apeiron_word.nat_shr_4
#audit apeiron_word.q4_inner_spec
#audit apeiron_word.q4Val_succ
#audit apeiron_word.q4Inv_succ
#audit apeiron_word.q4_outer_spec
#audit apeiron_word.q4QsRef_eq_q4CodesRef

-- APEIRON-GPU-SPV-W1-T1: fragment semantics and layout verdict
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

-- APEIRON-INT-FORMAT-W3-T8: Q4_K positive equality
#audit apeiron_word.word_q4_k_codes_ok
#audit apeiron_word.q4Val_zero
#audit apeiron_word.q4Val_four
#audit apeiron_word.q4Inv_repeat

-- APEIRON-INT-FORMAT-W3-T9: Q6_K references and bridge
#audit apeiron_word.q6QsRef
#audit apeiron_word.q6Val
#audit apeiron_word.q6Inv
#audit apeiron_word.getD_take
#audit apeiron_word.getD_drop_take
#audit apeiron_word.q6QsRef_eq_q6CodesRef

-- APEIRON-INT-FORMAT-W3-T10: Q6_K write layer
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

-- APEIRON-INT-FORMAT-W3-T11: Q6_K bit identities
#audit apeiron_word.q6_combine_lt
#audit apeiron_word.bmod_sub32

-- APEIRON-INT-FORMAT-W3-T12: Q6_K reference evaluations
#audit apeiron_word.shift4_mod
#audit apeiron_word.q6QsRef_idx0
#audit apeiron_word.q6QsRef_idx1
#audit apeiron_word.q6QsRef_idx2
#audit apeiron_word.q6QsRef_idx3
-- APEIRON-GPU-SPV-W2-T1: decoder soundness
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

-- APEIRON-INT-FORMAT-W3-T13: Q6_K factored identities
#audit apeiron_word.q6_combine_lt'
#audit apeiron_word.q6_high_lt
#audit apeiron_word.q6_cast_value
#audit apeiron_word.q6_write_val_gen

-- APEIRON-INT-FORMAT-W3-T15: Q6_K identities and pointwise split
#audit apeiron_word.q6_val_1_ident
#audit apeiron_word.q6_val_2_ident
#audit apeiron_word.q6_val_3_ident
#audit apeiron_word.q6_val_4_ident
#audit apeiron_word.q6_pointwise

-- APEIRON-INT-FORMAT-W3-T17: Q6_K body values chunk
#audit apeiron_word.q6_body_vals

-- APEIRON-INT-FORMAT-W3-T18: Q6_K inner body
#audit apeiron_word.q6_idx_lt_64
#audit apeiron_word.q6_idx_lt_32
#audit apeiron_word.q6_idx_lt_128
#audit apeiron_word.q6_measure
#audit apeiron_word.q6_succ_le
#audit apeiron_word.q6_succ_le'
#audit apeiron_word.q6_inner_body

-- APEIRON-INT-FORMAT-W3-T21: Q6_K loop specs
#audit apeiron_word.q6_measure2
#audit apeiron_word.q6_inner_spec
#audit apeiron_word.q6_outer_spec

-- APEIRON-INT-FORMAT-W3-T22: Q6_K positive equality
#audit apeiron_word.word_q6_k_codes_ok

-- APEIRON-INT-FORMAT-W3-T23: GGUF well-formedness predicates
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

-- APEIRON-INT-FORMAT-W3-T24: GGUF cursor primitives
#audit apeiron_word.gguf_cur_u8_some
#audit apeiron_word.gguf_cur_u8_none

-- APEIRON-ALGEBRA-D1: register the online-spinor attention identity
#audit Sanguine.AttentionSpinor.softmax_attention_is_online_spinor

-- APEIRON-ALGEBRA-D2: composed error and margin
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
-- word_attn_head_unfold dropped: it depends on attn.KvState.attn_head_f32, an f32 extraction
-- boundary declaration (proof/boundary-axioms.txt:20), outside this kernel-cone audit.
-- word_attn_head_{ok, zero_sum, exp_refuses, out_refuses} dropped: removed in the EngineAttn
-- repair (Numeric/Engine/Attn.lean:101-103), so no constant remains to audit.
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
-- KV state rows dropped: attn.alloc_f32_storage and mixer.f32_carrier.q32_to_f32 (with
-- core.f32.F32.is_finite) are f32 extraction boundary declarations (proof/boundary-axioms.txt:17-19),
-- outside this kernel cone and covered by the boundary allowlist instead.
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
-- Attention-downstream wave (lead-authorized): `not_attn_head_refines` is removed from
-- `Numeric.AttnSoftmaxPerturb` (its proof used `word_attn_head_pos_refuses`, which needs
-- `carrierQ32 state`, and `Std.Discriminant.read_discriminant` is an opaque instance projection, so
-- no concrete state discharges it). See `work/wip/attn-downstream/NOTES.md`.
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
-- Attention-downstream wave (lead-authorized): `Numeric.AttnLoopScore`, `Numeric.AttnLoopOutput`,
-- `Numeric.AttnHeadWindowsScore`, `Numeric.AttnHeadWindowsValue` and `Numeric.AttnHeadRefines`
-- moved to `work/wip/attn-downstream/`, so all of their rows below are dead and dropped:
-- `row_dot_fixed_exact`, `score_at_spec`, `attn_head_loop0_spec`, `attnNumeratorCase_le`,
-- `attn_head_loop1_spec`, `attn_head_loop2_spec`, `attn_value_component_loop_spec`,
-- `value_component_spec`, `attn_head_loop3_spec`, `attn_v2_prefix`, `attn_v2_loop0_domain`,
-- `attn_v2_loop1_domain`, `attn_loop0_ok`, `attn_loop1_ok`, `attn_loop_stage_facts`,
-- `attn_weight_bounds`, `attn_weight_window_of_numerators`, `plane_at_spec`, `attn_value_fit`,
-- `attn_value_window_of_v2`, `attn_value_components`, `attn_head_refines_v2_proved`.
-- `Numeric.AttnScoreStage` is retained: it was retargeted green against the live extraction.
-- Numeric.AttnScoreStage
#audit apeiron_word.attn_scale_bound
#audit apeiron_word.attnScoreInt_err
#audit apeiron_word.attn_score_stage
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
-- Ternary/Hadamard epic: Sylvester Hadamard algebra (docs/epics/ternary-hadamard.md W3-T5)
#audit apeiron_word.sylvester_succ
#audit apeiron_word.sylvester_transpose
#audit apeiron_word.sylvester_mul_self
#audit apeiron_word.sylvester_one
#audit apeiron_word.sylvester_one_channelPair
#audit apeiron_word.two_not_isUnit_int
#audit apeiron_word.sylvester_mulVec_butterfly
-- Ternary/Hadamard epic W3: low-bit row laws and the exact integer FWHT (docs/epics/ternary-hadamard.md)
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
-- Numeric.DequantLowBitRefusals (ternary/Hadamard W3-T7)
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
-- AI formalization lanes (docs/epics/frontier-consumer.md; ternary-hadamard W3-T6b partial)
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
-- Moved to `work/wip/attn-downstream/`: `KvReachable_inv`, `attn_domain_v2_structural` and
-- `attn_head_refines_reachable` are dropped with it (lead-authorized).
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
-- `attnHeadStage_dom_witness` is dropped from `Nous.Model.AttnStage` with `attnHeadStage`
-- (lead-authorized attention-downstream wave).
-- Nous.Model.BlockGroundedWitness
#audit Nous.Model.fullBlockGrounded_dom_nonempty
-- Nous.Model.AttnStageWitness moved to `work/wip/attn-downstream/`: `attnHeadStage_witness` and
-- `attnNZState_witness` are dropped with it (lead-authorized).
-- Nous.Model.TopKRouter (MoE W1-T2)
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
-- Nous.Model.MoeStage (MoE W3-T2)
#audit Nous.Model.moe_not_continuous
#audit Nous.Model.fixedMoe_lipschitz
#audit Nous.Model.moe_lipschitz_on_stable
#audit Nous.Model.margin_stable
-- Numeric.MatvecChunk (row-chunked projection = the whole projection)
#audit apeiron_word.matvec_fixed_rows
#audit apeiron_word.matvec_fixed_chunk
#audit apeiron_word.rowOut_join
#audit apeiron_word.matvec_fixed_chunks
-- Policy.Core / Policy.Laws (policy lane W1-T1)
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
-- Policy.Analysis (policy lane W1-T4)
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

-- APEIRON-MODEL-ALGEBRA-SPIKE-hyperbolic-attention-spec: vector-level Qwen attention spec
#audit Nous.Model.ellipticAlign_fst_eq_rhoDot
#audit Nous.Model.qwenScore_eq_sum_rhoDot
#audit Nous.Model.qwenScore_eq_rel
#audit Nous.Model.qwenScore_relative
#audit Nous.Model.qwenScore_witness
#audit Nous.Model.ringUpper_flow_one
#audit Nous.Model.ringLower_flow_one
#audit Nous.Model.softmax_weight_is_upper_channel
#audit Nous.Model.OneBit.exp_mul_sign
#audit Nous.Model.OneBit.onebit_factor_is_flow
#audit Nous.Model.OneBit.onebit_exp_product
#audit Nous.Model.OneBit.prod_flow_expansion
#audit Nous.Model.OneBit.onebit_exp_expansion
#audit Nous.Model.OneBit.carry_read_num
#audit Nous.Model.OneBit.carry_read_den
#audit Nous.Model.OneBit.onebit_attention_is_finite_carry
#audit Nous.Model.OneBit.carry_card
#audit Nous.Model.OneBit.carryNum_snoc
#audit Nous.Model.OneBit.carryDen_snoc
#audit Nous.Model.OneBit.onebit_truncation_bound
#audit Nous.Model.OneBit.carry_query_le_scan
#audit Nous.Model.OneBit.scanCausal_closed
#audit Nous.Model.OneBit.carry_causal_le_scan
#audit Nous.Model.OneBit.truncated_card
#audit Nous.Model.OneBit.onebit_witness
#audit Nous.Model.channelWeight_flow_one
#audit Nous.Model.flow_channel_witness
#audit Nous.Model.oaDenV_fuse
#audit Nous.Model.oaNumV_fuse
#audit Nous.Model.oaDenV_scan_acc
#audit Nous.Model.oaNumV_scan_acc
#audit Nous.Model.softmax_attention_is_online_spinor_vec
#audit Nous.Model.softmax_attention_is_online_readout
#audit Nous.Model.oaDenV_scan_append
#audit Nous.Model.oaNumV_scan_append
#audit Nous.Model.streaming_witness
#audit Nous.Model.exp_linearIndependent
#audit Nous.Model.exp_kernel_not_finite_rank
#audit Nous.Model.exp_value_map_injective
#audit Nous.Model.exp_distinct_witness
#audit Nous.Model.boost_factor
#audit Nous.Model.boostScore_eq_dotp
#audit Nous.Model.boost_finite_rank_recurrence
#audit Nous.Model.boost_finite_rank_recurrence_vec
#audit Nous.Model.boost_witness
#audit Nous.Model.boost_score_witness

-- one-bit W6-T3: sigma-delta bound, prefix, Abel, shaping
#audit Numeric.SigmaDelta.sd_state_bound
#audit Numeric.SigmaDelta.sd_prefix
#audit Numeric.SigmaDelta.sd_abel
#audit Numeric.SigmaDelta.sd_dot_bound
#audit Numeric.SigmaDelta.shape_dot
#audit Numeric.SigmaDelta.shape_null_exact
#audit Numeric.SigmaDelta.sd_beats_sign

-- one-bit W6-T4: hidden-axis permutation symmetry
#audit Numeric.HiddenPerm.perm_coordwise
#audit Numeric.HiddenPerm.hidden_perm_mlp
#audit Numeric.HiddenPerm.hidden_perm_mlp_submatrix
#audit Numeric.HiddenPerm.hidden_perm_gated
#audit Numeric.HiddenPerm.rms_perm
#audit Numeric.HiddenPerm.hidden_perm_residual_block
#audit Numeric.HiddenPerm.mlp_nonvacuous

-- one-bit W6-T1: one-bit units per face
#audit apeiron_word.OneBitUnits.channelPair_ringMul_left
#audit apeiron_word.OneBitUnits.channelPair_one_mul_one
#audit apeiron_word.OneBitUnits.channelPair_one_mul_negOne
#audit apeiron_word.OneBitUnits.channelPair_one_mul_j
#audit apeiron_word.OneBitUnits.channelPair_one_mul_negJ
#audit apeiron_word.OneBitUnits.hyper_mul_table
#audit apeiron_word.OneBitUnits.mul_i_components
#audit apeiron_word.OneBitUnits.i_sq
#audit apeiron_word.OneBitUnits.i_four
#audit apeiron_word.OneBitUnits.i_mul_negI
#audit apeiron_word.OneBitUnits.elliptic_order_two_unique

-- one-bit W6-T2: exact sign-plane decomposition
#audit apeiron_word.SignPlanes.signPlane_decomp
#audit apeiron_word.SignPlanes.planeNat_eq_finFunctionFinEquiv
#audit apeiron_word.SignPlanes.planeNat_injective
#audit apeiron_word.SignPlanes.planeNat_surj
#audit apeiron_word.SignPlanes.planeVal_injective
#audit apeiron_word.SignPlanes.planeVal_range
#audit apeiron_word.SignPlanes.matvec_decomp
#audit apeiron_word.SignPlanes.q4k_instance
#audit apeiron_word.SignPlanes.q8_0_instance
-- Exact amplitude form of KL (`Numeric/Quantum/Ledger/AmplitudeKLExact.lean`): KL = 2 ampDistSq + Σ p·h(u), h antitone on (-1,∞).
#audit Quantum.AmplitudeKL.h_deriv
#audit Quantum.AmplitudeKL.h_antitone
#audit Quantum.AmplitudeKL.h_nonneg_of_le_zero
#audit Quantum.AmplitudeKL.h_nonpos_of_ge_zero
#audit Quantum.AmplitudeKL.kl_eq_amp_sq_add_h
#audit Quantum.AmplitudeKL.kl_le_two_amp_sq_of_u_nonneg
#audit Quantum.AmplitudeKL.kl_ge_two_amp_sq_of_u_nonpos
