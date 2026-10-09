import Apeiron
import Apeiron.AuditClosed
import Apeiron.AuditCarrier
import Apeiron.AuditCombinator
import Apeiron.Combinator.Linear.Core.Duplication
import Apeiron.Combinator.RouteAnyRing
import Apeiron.Combinator.Linear.Core.Vanishing
import Apeiron.Trichotomy.IdempotentsAnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Apeiron.Doubling.AnyRing
import Apeiron.Projective.AnyRing
import Apeiron.Transport.Ordered
import Apeiron.Numeric.Generic
import Apeiron.Cosmology.AnyRing
import Apeiron.Decider.Extended.Completeness
import Apeiron.Trichotomy.Derived.Elliptic
import Apeiron.Physics.Kepler.Enclosure

/-!
# Audit: the axiom cone

@name Audit
@grade proved

`#audit` on every headline declaration in the corpus, run as part of the build. The command and
the closed cone `{propext, Classical.choice, Quot.sound}` live in `Apeiron.AuditClosed`; the
real carrier modules are audited in `Apeiron.AuditCarrier`, the rest here. Both run on import,
so a `sorryAx` in any named cone is a build failure rather than a printed line.

`#audit` speaks about declarations. It cannot see a runtime check such as a `#guard`, which adds
no declaration; those fail the build on their own and are not axiom cones. The parity gate in
`Apeiron/Decider/Parity.lean` is one, and it is not audited here.

Add a line for each new headline as the library grows.
-/

open Apeiron.Trichotomy

-- The grouped-literal parser model
#audit Apeiron.Parser.parse_sound
#audit Apeiron.Parser.parse_zero_fuel
#audit Apeiron.Parser.parse_grouped_literal
#audit Apeiron.Parser.parse_nested_grouped_literal

-- The bounded arithmetic parser model
#audit Apeiron.Parser.Arithmetic.parse_sound
#audit Apeiron.Parser.Arithmetic.primary_suffix_length
#audit Apeiron.Parser.Arithmetic.mulTail_suffix_length
#audit Apeiron.Parser.Arithmetic.mul_suffix_length
#audit Apeiron.Parser.Arithmetic.exprTail_suffix_length
#audit Apeiron.Parser.Arithmetic.expr_suffix_length
#audit Apeiron.Parser.Arithmetic.primary_node_bound
#audit Apeiron.Parser.Arithmetic.mulTail_node_bound
#audit Apeiron.Parser.Arithmetic.mul_node_bound
#audit Apeiron.Parser.Arithmetic.exprTail_node_bound
#audit Apeiron.Parser.Arithmetic.expr_node_bound
#audit Apeiron.Parser.Arithmetic.expr_capacity
#audit Apeiron.Parser.Arithmetic.primary_token_fuel
#audit Apeiron.Parser.Arithmetic.mulTail_token_fuel
#audit Apeiron.Parser.Arithmetic.mul_token_fuel
#audit Apeiron.Parser.Arithmetic.exprTail_token_fuel
#audit Apeiron.Parser.Arithmetic.expr_token_fuel
#audit Apeiron.Parser.Arithmetic.primary_complete
#audit Apeiron.Parser.Arithmetic.mulTail_complete
#audit Apeiron.Parser.Arithmetic.mul_complete
#audit Apeiron.Parser.Arithmetic.exprTail_complete
#audit Apeiron.Parser.Arithmetic.expr_complete
#audit Apeiron.Parser.Arithmetic.parseComplete_sound
#audit Apeiron.Parser.Arithmetic.mul_suffix_not_times
#audit Apeiron.Parser.Arithmetic.expr_suffix_not_plus
#audit Apeiron.Parser.Arithmetic.parse_precedence
#audit Apeiron.Parser.Arithmetic.parse_nested
#audit Apeiron.Parser.Arithmetic.parse_left_chains
#audit Apeiron.Parser.Arithmetic.parse_missing_close
#audit Apeiron.Parser.Arithmetic.parse_zero_fuel

-- The trichotomy
#audit Apeiron.Trichotomy.normForm_mul
#audit Apeiron.Trichotomy.hasZeroDivisor_iff_disc_nonneg
#audit Apeiron.Trichotomy.noZeroDivisors_iff_disc_neg
#audit Apeiron.Trichotomy.mul_distrib_add
#audit Apeiron.Trichotomy.mul_distrib_sub
#audit Apeiron.Trichotomy.mul_origin
#audit Apeiron.Trichotomy.recip_on_cone

-- The same construction over an arbitrary commutative ring, so a finite ring is an instance
-- rather than an analogy. `ringNormForm_mul` is what crosses; the idempotent statement needs
-- hypotheses that `ℝ` supplies for free and `Z/2^n` supplies differently.
#audit Apeiron.Trichotomy.AnyRing.ringNormForm_mul
#audit Apeiron.Trichotomy.AnyRing.exists_inverse_iff_isUnit_ringNormForm
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_snd_eq_zero
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_trivial

-- The affine-step monoid over any commutative ring: one step, its apply, composition, the
-- chronological left fold, the monoid and fold laws, and the integer order/sigma witness
#audit Apeiron.Trichotomy.AnyRing.RingAffineStep
#audit Apeiron.Trichotomy.AnyRing.ringAffineId
#audit Apeiron.Trichotomy.AnyRing.ringAffineApply
#audit Apeiron.Trichotomy.AnyRing.ringAffineCompose
#audit Apeiron.Trichotomy.AnyRing.ringAffineFold
#audit Apeiron.Trichotomy.AnyRing.ringAffineId_apply
#audit Apeiron.Trichotomy.AnyRing.ringAffineId_compose
#audit Apeiron.Trichotomy.AnyRing.ringAffineCompose_id
#audit Apeiron.Trichotomy.AnyRing.ringAffineCompose_apply
#audit Apeiron.Trichotomy.AnyRing.ringAffineCompose_assoc
#audit Apeiron.Trichotomy.AnyRing.ringAffineFold_apply
#audit Apeiron.Trichotomy.AnyRing.ringAffineFold_append
#audit Apeiron.Trichotomy.AnyRing.ringAffine_hyperbolic_witness

#audit Apeiron.Trichotomy.classification
#audit Apeiron.Trichotomy.trichotomy_complete
#audit Apeiron.Trichotomy.hasNullPair_iff_disc_pos

-- The ring quotient
#audit Apeiron.Trichotomy.mul_eq_quotient_mul
#audit Apeiron.Trichotomy.root_sq
#audit Apeiron.Trichotomy.quotient_classification
#audit Apeiron.Trichotomy.quot_hasZeroDivisor_iff_disc_nonneg
#audit Apeiron.Trichotomy.quot_noZeroDivisors_iff_disc_neg
#audit Apeiron.Trichotomy.quot_exists_nontrivial_idempotent_iff_disc_pos
#audit Apeiron.Trichotomy.toQuotEquiv_apply

-- Idempotents
#audit Apeiron.Trichotomy.exists_nontrivial_idempotent_iff_disc_pos
#audit Apeiron.Trichotomy.idempotent_iff_trivial_of_disc_nonpos
#audit Apeiron.Trichotomy.idempotent_iff

-- The null basis
#audit Apeiron.Trichotomy.upper_mul
#audit Apeiron.Trichotomy.lower_mul
#audit Apeiron.Trichotomy.normForm_eq_upper_mul_lower
#audit Apeiron.Trichotomy.idempotent_locus
#audit Apeiron.Trichotomy.le_iff_exists_sq
#audit Apeiron.Trichotomy.int_le_iff_exists_sum_four_sq
#audit Apeiron.Trichotomy.two_not_sq
#audit Apeiron.Trichotomy.seven_not_sum_three_sq
#audit Apeiron.Trichotomy.upper_le_iff_exists_sq
#audit Apeiron.Trichotomy.lower_le_iff_exists_sq
#audit Apeiron.Trichotomy.inCell_meet_iff
#audit Apeiron.Trichotomy.crossing_independent_of_cone

-- The idempotent monoid
#audit Apeiron.Trichotomy.idempotent_classification
#audit Apeiron.Trichotomy.idempotent_mul
#audit Apeiron.Trichotomy.mul_assoc

-- Projection: what discarding costs
#audit Apeiron.Trichotomy.discards_iff_disc_pos
#audit Apeiron.Trichotomy.proj_proj
#audit Apeiron.Trichotomy.proj_idemPlus_halves
#audit Apeiron.Trichotomy.proj_keeps_first_channel

-- Logic gates
#audit Apeiron.Trichotomy.fold_is_and_and_nor
#audit Apeiron.Trichotomy.normForm_fold
#audit Apeiron.Trichotomy.fold_perm

-- Channels: the reads are forced
#audit Apeiron.Trichotomy.no_channel_of_disc_neg
#audit Apeiron.Trichotomy.channel_iff_fst_of_disc_zero
#audit Apeiron.Trichotomy.channel_iff_of_disc_pos
#audit Apeiron.Trichotomy.two_channels_iff_disc_pos

-- Application: the ordered read and the tower
#audit Apeiron.Trichotomy.ratio_mul
#audit Apeiron.Trichotomy.ratio_ofHalves
#audit Apeiron.Trichotomy.Tower.state_node
#audit Apeiron.Trichotomy.Tower.state_const
#audit Apeiron.Trichotomy.Tower.node_faithful_iff_disc_pos

-- C*-algebraic and operator structures
#audit Apeiron.Trichotomy.cstar_identity_pos
#audit Apeiron.Trichotomy.one_add_star_mul_self_isUnit
#audit Apeiron.Trichotomy.conj_not_cstar_pos
#audit Apeiron.Trichotomy.fundamentalSymmetry_eq_conj
#audit Apeiron.Trichotomy.fundamentalSymmetry_involutive
#audit Apeiron.Trichotomy.krein_adjoint_operator
#audit Apeiron.Trichotomy.krein_adjoint_eq_conj
#audit Apeiron.Trichotomy.jacobson_radical_zero
#audit Apeiron.Trichotomy.no_cstar_norm_zero
#audit Apeiron.Trichotomy.ratio_spectral_representation
#audit Apeiron.Trichotomy.recovery_by_rotation_cell
#audit Apeiron.Trichotomy.cstarNormNeg_nonneg
#audit Apeiron.Trichotomy.cstarNormNeg_eq_zero_iff
#audit Apeiron.Trichotomy.cstarNormNeg_mul
#audit Apeiron.Trichotomy.cstar_identity_neg
#audit Apeiron.Trichotomy.one_add_star_mul_self_isUnit_neg
#audit Apeiron.Trichotomy.trichotomy_cstar_classification

-- Combinators

-- Confluence

-- The table
#audit Apeiron.Table.merge_is_the_only_freedom
#audit Apeiron.Table.get_build_acc
#audit Apeiron.Table.build_gauge_invariance
#audit Apeiron.Table.build_gauge_invariance_mul
#audit Apeiron.Table.build_eq_of_key_streams_eq
#audit Apeiron.Table.get_build_ringAffineCompose
#audit Apeiron.Table.OrderBlind
#audit Apeiron.Table.orderBlind_iff_leftTranslations_commute
#audit Apeiron.Table.orderBlind_iff_commutative
#audit Apeiron.Table.build_orderBlind

-- The low-word identity: odd multiplication is injective at every power-of-two width, the
-- spread constant specializes it at 64 bits, and the widened product and even control bound it
#audit Apeiron.Table.lowWord
#audit Apeiron.Table.odd_mul_zmod_injective
#audit Apeiron.Table.lowWord_mul_eq_zmod_mul
#audit Apeiron.Table.phi64
#audit Apeiron.Table.phi64_odd
#audit Apeiron.Table.phi64_mul_zmod_injective
#audit Apeiron.Table.phi64_product_fits_u128
#audit Apeiron.Table.phi64_lowWord_eq_zmod_mul
#audit Apeiron.Table.phi64_lowWord_injective
#audit Apeiron.Table.two_mul_zmod64_not_injective

-- The graph face: order is part of the meaning
#audit Apeiron.Table.get_build_append
#audit Apeiron.Table.build_append_perm_pointwise
#audit Apeiron.Table.build_append_swap_ne
#audit Apeiron.Table.append_not_gauge
#audit Apeiron.Table.get_build_replace
#audit Apeiron.Table.replace_is_head_of_append

-- The hashtrinity readout: one channel decides, the other is a genuine key
#audit Apeiron.Table.one_half_decides
#audit Apeiron.Table.opaque_needs_lower
#audit Apeiron.Table.coord_home_collisionFree
#audit Apeiron.Table.opaque_home_collides

-- Use-once resources
#audit Apeiron.Trichotomy.normForm_zero_not_invertible
#audit Apeiron.Trichotomy.exists_normForm_zero_iff_disc_nonneg
#audit Apeiron.Trichotomy.isUnit_iff_normForm_ne_zero
#audit Apeiron.Trichotomy.mul_conj

-- The projective line
#audit Apeiron.Projective.channels_ne_zero
#audit Apeiron.Projective.dehomUpper_scale
#audit Apeiron.Projective.act_mul
#audit Apeiron.Projective.act_one
#audit Apeiron.Projective.dehom_agree_of_disc_zero

-- The bulk
#audit Apeiron.Projective.mobiusNum_conj_snd
#audit Apeiron.Projective.den_normForm_pos_of_disc_neg
#audit Apeiron.Projective.bulk_preserved_of_disc_neg
#audit Apeiron.Projective.bulk_not_preserved_of_disc_nonneg
#audit Apeiron.Projective.mobius_snd
#audit Apeiron.Projective.mobiusNum_cross
#audit Apeiron.Projective.normForm_mobiusTangent

-- The metric on the half-plane
#audit Apeiron.Projective.mobius_isometry
#audit Apeiron.Projective.lineElement_elliptic
#audit Apeiron.Projective.lineElement_parabolic
#audit Apeiron.Projective.lineElement_hyperbolic
#audit Apeiron.Projective.lineElement_pos_of_disc_neg
#audit Apeiron.Projective.exists_null_direction_iff_disc_nonneg

-- The factor is forced, and the boundary
#audit Apeiron.Projective.scaleInvariant_iff_inverse
#audit Apeiron.Projective.lineElement_eq_genLineElement
#audit Apeiron.Projective.genLineElement_scale
#audit Apeiron.Projective.mobius_boundary
#audit Apeiron.Projective.mobius_transitive

-- Length, and the boundary at infinity
#audit Apeiron.Projective.arcLength_vertical
#audit Apeiron.Projective.log_height_le_arcLength
#audit Apeiron.Projective.vertical_minimises
#audit Apeiron.Projective.boundary_at_infinite_distance

-- Division in the algebra
#audit Apeiron.Trichotomy.mul_recip_cancel
#audit Apeiron.Trichotomy.normForm_recip

-- The bridge to song's cell
#audit Apeiron.Bridge.upper_ofBounds
#audit Apeiron.Bridge.conj_ofBounds
#audit Apeiron.Bridge.crossing_independent_of_cone_ofBounds
#audit Apeiron.Bridge.meet_ofBounds_iff
#audit Apeiron.Bridge.meet_ofBounds
#audit Apeiron.Bridge.upper_conj_ofBounds

-- The verdict: normalize-and-compare, exact on the fragment it decides

-- The bridge to song's prover
#audit Apeiron.Bridge.iota_generates_s
#audit Apeiron.Bridge.church_pair_fst
#audit Apeiron.Bridge.distinct_vars
#audit Apeiron.Bridge.omega_fuel

-- Executing evaluator and complete-input certificate-reader contracts
#audit Apeiron.Bridge.Execution.Completion
#audit Apeiron.Bridge.Execution.Request
#audit Apeiron.Bridge.Execution.Value
#audit Apeiron.Bridge.Execution.evaluate
#audit Apeiron.Bridge.Execution.Request.Related
#audit Apeiron.Bridge.Execution.Value.Related
#audit Apeiron.Bridge.Execution.Evaluator
#audit Apeiron.Bridge.Execution.Evaluator.Refines
#audit Apeiron.Bridge.Execution.Evaluator.Refines.verdict_decided_exact
#audit Apeiron.Bridge.Execution.Reader
#audit Apeiron.Bridge.Execution.Reader.Refines
#audit Apeiron.Bridge.Execution.Reader.Refines.accepts_iff
#audit Apeiron.Bridge.Execution.Reader.Refines.accepted_sound
#audit Apeiron.Bridge.Execution.canonicalEvaluator
#audit Apeiron.Bridge.Execution.canonicalEvaluator_refines
#audit Apeiron.Bridge.Execution.ReaderFailure
#audit Apeiron.Bridge.Execution.canonicalReader
#audit Apeiron.Bridge.Execution.canonicalReader_refines

-- Non-termination
#audit Apeiron.Resource.no_double_use
#audit Apeiron.Resource.use_once_needs_a_residue
#audit Apeiron.Resource.consumeAt_second_alias_is_nothing
#audit Apeiron.Resource.consumeLocal_double_delivery_under_aliasing

-- Reuse, and the generation tag that closes it
#audit Apeiron.Resource.reuse_readmits_delivery
#audit Apeiron.Resource.genConsume_stale
#audit Apeiron.Resource.generation_closes_reuse
#audit Apeiron.Resource.generation_closes_alias

-- The machine encoding: guards realise total ops, schedules realise the one fold
#audit Apeiron.Machine.guard_realises_divT
#audit Apeiron.Machine.guard_realises_tmodT
#audit Apeiron.Machine.foldl_chunks
#audit Apeiron.Machine.testBit_pack

-- Capability provenance
#audit Apeiron.Capability.reachable_iff_inScope
#audit Apeiron.Capability.eval_in_scope
#audit Apeiron.Capability.forge_reaches_only_scope
#audit Apeiron.Capability.cexpr_exhaustive

-- The extent table cannot authenticate a coordinate
#audit Apeiron.Capability.ungranted_reads_granted_extent
#audit Apeiron.Capability.lookup_period
#audit Apeiron.Capability.lookup_below_period

-- The finite region table receives raw keys through the naming fold
#audit Apeiron.Capability.maskedKey_complete
#audit Apeiron.Capability.maskedRead_after_write
#audit Apeiron.Capability.maskedRead_after_other_write
#audit Apeiron.Capability.maskedRead_after_alias_write
#audit Apeiron.Capability.guardedRead_ungranted
#audit Apeiron.Capability.guardedRead_granted

-- The carrier boundary is its unique total, residue-absorbing merge
#audit Apeiron.Boundary.boundaryRespecting_iff_opTwo
#audit Apeiron.Boundary.map_state_iff

-- Transport spread
#audit Apeiron.Spread.spread_ratio
#audit Apeiron.Spread.rest_le_max_source
#audit Apeiron.Spread.rest_ge_min_source
#audit Apeiron.Spread.mixing_cannot_widen
#audit Apeiron.Spread.drag_ratio_of_exponential_response
#audit Apeiron.Spread.inCell_restEnvelope
#audit Apeiron.Spread.inCell_meet_restEnvelope_iff
#audit Apeiron.Spread.restEnvelope_joint_witness
#audit Apeiron.Spread.spread_le_source_meet_exponential_range

-- Executable split-face classification
#audit Apeiron.Classification.Margin.SplitFace
#audit Apeiron.Classification.Margin.upper_channel
#audit Apeiron.Classification.Margin.lower_channel
#audit Apeiron.Classification.Margin.ringRatio
#audit Apeiron.Classification.Margin.ofChannels
#audit Apeiron.Classification.Margin.ringUpper_ofChannels
#audit Apeiron.Classification.Margin.ringLower_ofChannels
#audit Apeiron.Classification.Margin.RingInCell
#audit Apeiron.Classification.Margin.ringMeet
#audit Apeiron.Classification.Margin.ringInCell_meet_iff
#audit Apeiron.Classification.Margin.ratioBand
#audit Apeiron.Classification.Margin.ringInCell_ratioBand_iff
#audit Apeiron.Classification.Margin.ringInCell_meet_ratioBand_iff
#audit Apeiron.Classification.Margin.ringRatio_ofChannels_one
#audit Apeiron.Classification.Margin.IsWinner
#audit Apeiron.Classification.Margin.margin_preserves_label
#audit Apeiron.Classification.Margin.margin_sharp
#audit Apeiron.Classification.Margin.cut
#audit Apeiron.Classification.Margin.axisCell
#audit Apeiron.Classification.Margin.ClearsCuts
#audit Apeiron.Classification.Margin.cut_side_stable
#audit Apeiron.Classification.Margin.axisCell_stable
#audit Apeiron.Classification.Margin.axisCell_sharp
#audit Apeiron.Classification.Margin.boxLabel
#audit Apeiron.Classification.Margin.boxLabel_stable
#audit Apeiron.Classification.Margin.rationalFace

-- Scale-invariant split-face classification
#audit Apeiron.Classification.ScaleInvariant.ringRatio_scale_invariant
#audit Apeiron.Classification.ScaleInvariant.ratioBand_scale_invariant
#audit Apeiron.Classification.ScaleInvariant.cut_scale_invariant
#audit Apeiron.Classification.ScaleInvariant.axisCell_scale_invariant
#audit Apeiron.Classification.ScaleInvariant.boxLabel_scale_invariant
#audit Apeiron.Classification.ScaleInvariant.absoluteUpperCut
#audit Apeiron.Classification.ScaleInvariant.absolute_upper_threshold_moves
#audit Apeiron.Classification.ScaleInvariant.zero_scale_changes_ratio

-- Baroclinic instability
#audit Apeiron.Physics.growth_iff_noZeroDivisors
#audit Apeiron.Physics.grows_iff_disc_neg
#audit Apeiron.Physics.sign_change_of_weighted_zero
#audit Apeiron.Physics.deformation_balances

-- The algebra as grids: the norm form is the determinant
#audit Apeiron.Physics.rep_mul
#audit Apeiron.Physics.rep_one
#audit Apeiron.Physics.det_rep
#audit Apeiron.Physics.det_generator
#audit Apeiron.Physics.generator_trichotomy
#audit Apeiron.Physics.rep_det_one_iff
#audit Apeiron.Physics.unitNorm_mul
#audit Apeiron.Physics.elliptic_one_param
#audit Apeiron.Physics.elliptic_rotors_commute
#audit Apeiron.Physics.parabolic_one_param
#audit Apeiron.Physics.shear_additive_iff_parabolic
#audit Apeiron.Physics.hyperbolic_one_param
#audit Apeiron.Physics.rep_elliptic_is_rotation
#audit Apeiron.Trichotomy.form_mul
#audit Apeiron.Physics.score_sees_only_the_gap

-- The pair

-- The FUEL semantics: a false floor, and a discarding fixed point


-- The corpus on the ring itself
#audit Apeiron.Trichotomy.quot_discards_iff_disc_pos
#audit Apeiron.Trichotomy.quot_idempotent_locus
#audit Apeiron.Trichotomy.quot_upper_mul
#audit Apeiron.Trichotomy.quot_fold_is_and_and_nor

-- The hyperbolic face
#audit Apeiron.Trichotomy.SplitComplex.disc_eq_one
#audit Apeiron.Trichotomy.SplitComplex.nullPair_mul_eq_zero
#audit Apeiron.Trichotomy.SplitComplex.ePlus_idem
#audit Apeiron.Trichotomy.SplitComplex.e_orthogonal
#audit Apeiron.Trichotomy.SplitComplex.e_add_eq_one

-- The carrier boundary
#audit Apeiron.Boundary.typed_never_null
#audit Apeiron.Boundary.mismatch_is_null
#audit Apeiron.Boundary.map_state
#audit Apeiron.Boundary.reinterpret_breaks_meaning
#audit Apeiron.Boundary.type_safety_from_the_boundary

-- The graded region
#audit Apeiron.Region.chart_shift_up
#audit Apeiron.Region.chart_shift_down
#audit Apeiron.Region.chart_mono
#audit Apeiron.Region.graded_access_in_region
#audit Apeiron.Region.graded_no_overflow_into_neighbor
#audit Apeiron.Region.chart_decode
#audit Apeiron.Region.decode_chart

-- Fragment containment
#audit Apeiron.Fragment.Row.value_add_valuation
#audit Apeiron.Fragment.Row.scaled_product_not_a_row
#audit Apeiron.Fragment.Row.product_not_a_row
#audit Apeiron.Fragment.denote_eval
#audit Apeiron.Fragment.eval_is_a_row
#audit Apeiron.Fragment.expr_exhaustive
#audit Apeiron.Fragment.no_expr_denotes_a_product
#audit Apeiron.Fragment.scale_distributes
#audit Apeiron.Fragment.monomial_cancels
#audit Apeiron.Fragment.rows_iff
#audit Apeiron.Fragment.combo_holds
#audit Apeiron.Fragment.unsat_of_negative_combo
#audit Apeiron.Fragment.sat_iff_finite_search
#audit Apeiron.Fragment.unit_elimination_exact
#audit Apeiron.Fragment.two_caps_iff_disc_pos
#audit Apeiron.Fragment.holds_iff_four_squares
#audit Apeiron.Fragment.holds_iff_not_crossing
#audit Apeiron.Fragment.crossingRow_is_the_crossing
#audit Apeiron.Fragment.norm_not_a_row
#audit Apeiron.Fragment.verdict_needs_no_norm

-- The roots of one, and the one flow across the three faces
#audit Apeiron.Trichotomy.exists_sqrt_neg_one_iff_disc_neg
#audit Apeiron.Trichotomy.exists_extra_sqrt_one_iff_disc_pos
#audit Apeiron.Trichotomy.exists_cube_root_one_iff_disc_neg
#audit Apeiron.Trichotomy.normForm_flow
#audit Apeiron.Trichotomy.flow_add
#audit Apeiron.Trichotomy.hasDerivAt_flow
#audit Apeiron.Trichotomy.flow_returns_iff_disc_neg

-- Imaginary time, and the temperature closure fixes
#audit Apeiron.Physics.wick_flips_the_sign
#audit Apeiron.Physics.flow_wick
#audit Apeiron.Physics.no_period_of_disc_nonneg
#audit Apeiron.Physics.smoothness_fixes_temperature
#audit Apeiron.Physics.flow_solves_the_generator_ode

-- One wave, two channels
#audit Apeiron.Physics.secondDeriv_waveFst
#audit Apeiron.Physics.secondDeriv_waveSnd
#audit Apeiron.Physics.hamilton_jacobi
#audit Apeiron.Physics.continuity

-- The Schrodinger equation split into its two actual polar derivatives
#audit Apeiron.Physics.polarWave
#audit Apeiron.Physics.polarTimeJet
#audit Apeiron.Physics.polarSpaceJet
#audit Apeiron.Physics.quantumPotential
#audit Apeiron.Physics.hamiltonJacobiResidual
#audit Apeiron.Physics.continuityResidual
#audit Apeiron.Physics.schrodingerResidual
#audit Apeiron.Physics.normForm_polarWave
#audit Apeiron.Physics.schrodingerResidual_factorization
#audit Apeiron.Physics.schrodinger_polar_iff_madelung
#audit Apeiron.Physics.quantumPotential_eq_laplacian_sqrt_density
#audit Apeiron.Physics.quantumPotential_eq_actual_laplacian_sqrt_density
#audit Apeiron.Physics.zero_amplitude_counterexample
#audit Apeiron.Physics.polarWaveTimeDerivative
#audit Apeiron.Physics.polarWaveSpaceDerivative
#audit Apeiron.Physics.analyticSchrodingerResidual
#audit Apeiron.Physics.polarWave_time_derivative_eq_jet
#audit Apeiron.Physics.polarWave_space_derivative_eq_jet
#audit Apeiron.Physics.continuityResidual_eq_amplitudeSquared_current_derivative
#audit Apeiron.Physics.analyticSchrodingerResidual_eq_jet
#audit Apeiron.Physics.actual_schrodinger_iff_madelung
#audit Apeiron.Physics.schrodinger_joint_witness
#audit Apeiron.Physics.zero_wave_nonzero_action_time_counterexample

-- The three explicit quadratic Van Vleck formulas as pointwise Schrodinger solutions
#audit Apeiron.Physics.PointwiseSchrodingerExact
#audit Apeiron.Physics.pointwiseSchrodingerExact_iff
#audit Apeiron.Physics.parabolicVanVleckPotential
#audit Apeiron.Physics.parabolicVanVleckDensity
#audit Apeiron.Physics.parabolicVanVleckAction
#audit Apeiron.Physics.parabolic_vanVleck_density_eq_mixed_action
#audit Apeiron.Physics.parabolic_vanVleck_face
#audit Apeiron.Physics.parabolic_vanVleck_pointwise_exact
#audit Apeiron.Physics.parabolic_vanVleck_joint_witness
#audit Apeiron.Physics.ellipticVanVleckPotential
#audit Apeiron.Physics.ellipticVanVleckDensity
#audit Apeiron.Physics.ellipticVanVleckAction
#audit Apeiron.Physics.elliptic_vanVleck_density_eq_mixed_action
#audit Apeiron.Physics.elliptic_vanVleck_face
#audit Apeiron.Physics.elliptic_vanVleck_pointwise_exact
#audit Apeiron.Physics.elliptic_vanVleck_joint_witness
#audit Apeiron.Physics.hyperbolicVanVleckPotential
#audit Apeiron.Physics.hyperbolicVanVleckDensity
#audit Apeiron.Physics.hyperbolicVanVleckAction
#audit Apeiron.Physics.hyperbolic_vanVleck_density_eq_mixed_action
#audit Apeiron.Physics.hyperbolic_vanVleck_face
#audit Apeiron.Physics.hyperbolic_vanVleck_pointwise_exact
#audit Apeiron.Physics.hyperbolic_vanVleck_joint_witness

-- The classical residue, and the four branches that carry it
#audit Apeiron.Physics.residue_is_curvature
#audit Apeiron.Physics.sq_neg_one_implies_zero_first
#audit Apeiron.Physics.sq_neg_one_implies_nonzero_second
#audit Apeiron.Physics.fourth_power_of_sq_neg_one
#audit Apeiron.Physics.sq_neg_one_implies_not_unit
#audit Apeiron.Physics.sq_neg_one_implies_not_neg_unit
#audit Apeiron.Physics.kinetic_vanishes_at_degenerate
#audit Apeiron.Physics.kinetic_sign_flips
#audit Apeiron.Physics.flow_quarter_turn_points
#audit Apeiron.Physics.stencil4_normForm
#audit Apeiron.Physics.stencil4_normForm_split
#audit Apeiron.Physics.stencil4_normForm_elliptic

-- The cone, the ray, the shell
#audit Apeiron.Physics.cone_rays
#audit Apeiron.Physics.massless_needs_disc_nonneg
#audit Apeiron.Physics.mover_transport
#audit Apeiron.Physics.kg_residual
#audit Apeiron.Physics.planeWave_law_iff_on_shell
#audit Apeiron.Physics.law_holds_off_shell_at_a_node

-- The flavour cone
#audit Apeiron.Physics.third_turn_is_the_cube_root
#audit Apeiron.Physics.cone_law
#audit Apeiron.Physics.koide_of_trinity
#audit Apeiron.Physics.koide_iff_amplitude_sq_two
#audit Apeiron.Physics.amplitude_sq_of_koide
#audit Apeiron.Physics.azimuth_invariant
#audit Apeiron.Physics.trinity_root
#audit Apeiron.Physics.koide_lands
#audit Apeiron.Physics.koide_intervals_bracket_the_data
#audit Apeiron.Physics.koide_iff_null
#audit Apeiron.Physics.koide_iff_spread

-- The two mixing rules, and what follows from them
#audit Apeiron.Physics.maximal_mixing_iff_on_hyperbolic_cone
#audit Apeiron.Physics.balanced_forces_half
#audit Apeiron.Physics.quarter_turn_is_balanced
#audit Apeiron.Physics.gatto_predicts_cabibbo

-- The ladder
#audit Apeiron.Physics.residue_counts_null_pairs
#audit Apeiron.Physics.dimOf_as_cells
#audit Apeiron.Physics.one_cell_column

-- The chargeless sector
#audit Apeiron.Cosmology.charge_rotor_iff_disc_neg
#audit Apeiron.Cosmology.rephasing_fixes_nothing
#audit Apeiron.Cosmology.weightSpread_eq_variance
#audit Apeiron.Cosmology.equal_weights_silence
#audit Apeiron.Cosmology.phase_floor
#audit Apeiron.Cosmology.double_beta_lands
#audit Apeiron.Cosmology.clean_spectrum_cubic
#audit Apeiron.Cosmology.registered_spectrum_matches_the_splittings
#audit Apeiron.Cosmology.registered_sum_is_above_the_standard_bound

-- The Cayley-Dickson doubling, and the corpus's algebra as its first case
#audit Apeiron.Doubling.real_mul
#audit Apeiron.Doubling.real_star
#audit Apeiron.Doubling.real_norm
#audit Apeiron.Doubling.gen_sq
#audit Apeiron.Doubling.star_star_eq
#audit Apeiron.Doubling.exists_null_iff_sigma_nonneg
#audit Apeiron.Doubling.trichotomy_isotropy_from_doubling
#audit Apeiron.Doubling.no_break_of_composition_of_anisotropy
#audit Apeiron.Doubling.no_composition_of_anisotropy_of_break
#audit Apeiron.Doubling.isotropic_of_composition_of_break
#audit Apeiron.Doubling.split_mul_eq_zero
#audit Apeiron.Doubling.finrank_double

-- The eight-dimensional cases, definite and split
#audit Apeiron.Doubling.normO_mul
#audit Apeiron.Doubling.normO_eq_zero_iff
#audit Apeiron.Doubling.no_zero_divisors
#audit Apeiron.Doubling.not_associative
#audit Apeiron.Doubling.normS_mul
#audit Apeiron.Doubling.null_cone
#audit Apeiron.Doubling.zero_divisor
#audit Apeiron.Doubling.splitComplex_is_the_real_double

-- Dimension sixteen: anisotropic, with a zero divisor, so not composing
#audit Apeiron.Doubling.no_null_vector
#audit Apeiron.Doubling.mul_eq_zero
#audit Apeiron.Doubling.no_anisotropic_composition_norm
#audit Apeiron.Doubling.normSed_not_multiplicative

-- The eight of the ladder as a dimension
#audit Apeiron.Doubling.finrank_octonion
#audit Apeiron.Doubling.finrank_sedenion
#audit Apeiron.Doubling.ladder_period_is_octonion
#audit Apeiron.Doubling.definite_form_is_the_split_form_at_imaginary_second_slot

-- Integral symmetric forms
#audit Apeiron.Lattice.IntForm.SignatureSign
#audit Apeiron.Lattice.IntForm.SignatureSign.toReal
#audit Apeiron.Lattice.IntForm.SignatureSign.toReal_eq_one_or_neg_one
#audit Apeiron.Lattice.IntForm.HasSignature
#audit Apeiron.Lattice.IntForm.sigPos
#audit Apeiron.Lattice.IntForm.sigNeg
#audit Apeiron.Lattice.IntForm.sigPos_add_sigNeg
#audit Apeiron.Lattice.IntForm.sigNeg_eq_count_neg_one
#audit Apeiron.Lattice.IntForm.isEven_iff_even_diag
#audit Apeiron.Lattice.IntForm.dsum_hasSignature
#audit Apeiron.Lattice.IntForm.pow_hasSignature
#audit Apeiron.Lattice.IntForm.hasSignature_one_of_gram_eq_mulTranspose
#audit Apeiron.Lattice.IntForm.det_eq_one_of_gram_eq_mulTranspose
#audit Apeiron.Lattice.IntForm.bil_pos_of_gram_eq_mulTranspose

-- The two generators, and rank 26
#audit Apeiron.Lattice.hyperbolicPlane_gram_is_the_null_pair
#audit Apeiron.Lattice.e8_octonion_gram
#audit Apeiron.Lattice.E8_det
#audit Apeiron.Lattice.E8_posDef
#audit Apeiron.Lattice.evenUnimodular_hasSignature
#audit Apeiron.Lattice.signature_difference
#audit Apeiron.Lattice.eight_dvd_signature_difference
#audit Apeiron.Lattice.euRank_eq_dimOf
#audit Apeiron.Lattice.bosonic_twentysix
#audit Apeiron.Lattice.twentysix_also_carries_five_minus_signs
#audit Apeiron.Lattice.local_cannonball_uniqueness
#audit Apeiron.Lattice.cannonball_selects_twentysix

-- Multipliers of a planar fixed point, and the renormalisation operator
#audit Apeiron.Physics.multiplier_eq_zero_iff
#audit Apeiron.Physics.exists_real_multiplier_iff_disc_nonneg
#audit Apeiron.Physics.two_real_multipliers_iff_disc_pos
#audit Apeiron.Physics.multiplier_is_one_at_the_parabolic_face
#audit Apeiron.Physics.multipliers_on_the_unit_circle
#audit Apeiron.Physics.bulk_comp
#audit Apeiron.Physics.bulk_is_saddle_node
#audit Apeiron.Physics.tangency_fixed
#audit Apeiron.Physics.drift_quadruples
#audit Apeiron.Physics.four_is_a_renorm_multiplier
#audit Apeiron.Physics.polynomial_renorm_fixed_iff
#audit Apeiron.Physics.no_polynomial_fixed_point_above_degree_one
#audit Apeiron.Physics.golden_fixed
#audit Apeiron.Physics.golden_multiplier
#audit Apeiron.Physics.upper_flow_one
#audit Apeiron.Physics.lower_flow_one
#audit Apeiron.Physics.boost_exponents_multiply_to_one
#audit Apeiron.Physics.lyapunov_rate

-- The decider: the face, and which value of σ the whole numbers force
#audit Apeiron.Decider.face_is_forced
#audit Apeiron.Decider.disc_one_of_integer_basis
#audit Apeiron.Decider.mem_intervalOf
#audit Apeiron.Decider.meet_intervalOf
#audit Apeiron.Decider.mem_meet_intervalOf
#audit Apeiron.Decider.conj_intervalOf
#audit Apeiron.Decider.pick_is_conj
#audit Apeiron.Decider.crossing_intervalOf
#audit Apeiron.Decider.crossing_empty

-- The wire format, and the guard that is not decoration
#audit Apeiron.Decider.eq_line_is_two_caps
#audit Apeiron.Decider.satRows_loadRows
#audit Apeiron.Decider.satLines_iff_four_squares
#audit Apeiron.Decider.nc_mag_eq
#audit Apeiron.Decider.sum_can_leave_range

-- Phase one: propagation
#audit Apeiron.Decider.propPick_is_conj
#audit Apeiron.Decider.meet_is_the_algebra
#audit Apeiron.Decider.bnd_meet_holds
#audit Apeiron.Decider.crossed_iff_crossing
#audit Apeiron.Decider.crossed_empty
#audit Apeiron.Decider.rhsOf_le
#audit Apeiron.Decider.derived_bound_four_squares
#audit Apeiron.Decider.propOne_sound
#audit Apeiron.Decider.propFix_sound
#audit Apeiron.Decider.groundBad_sound
#audit Apeiron.Decider.propUnsat_sound

-- Phase two: Fourier-Motzkin and the cutting plane
#audit Apeiron.Decider.combineFM_holds
#audit Apeiron.Decider.combineFM_cancels
#audit Apeiron.Decider.tidyRow_holds
#audit Apeiron.Decider.tidy_four_squares
#audit Apeiron.Decider.eliminate_sound
#audit Apeiron.Decider.eliminate_four_squares

-- The certificate
#audit Apeiron.Decider.certRaw_holds
#audit Apeiron.Decider.combo_is_farkas
#audit Apeiron.Decider.crCombo_holds
#audit Apeiron.Decider.crNeg_holds
#audit Apeiron.Decider.crMeet_holds
#audit Apeiron.Decider.crTidy_holds
#audit Apeiron.Decider.impossible_is_crossing
#audit Apeiron.Decider.crImpossible_empty
#audit Apeiron.Decider.certStep_holds
#audit Apeiron.Decider.checkCert_refutes
#audit Apeiron.Decider.checkCert_sound
#audit Apeiron.Decider.CertOp
#audit Apeiron.Decider.synthesizeCert
#audit Apeiron.Decider.synthesizeCert_accepts
#audit Apeiron.Decider.certificate_complete
#audit Apeiron.Decider.cert_four_squares
#audit Apeiron.Decider.negative_combo_refutes

-- Residue normalization and exact unit projection: the semantic equivalences used by uncapped
-- decision recursion.
#audit Apeiron.Decider.rowDivisor
#audit Apeiron.Decider.rowDivisor_pos
#audit Apeiron.Decider.rowDivisor_dvd
#audit Apeiron.Decider.floor_bound_iff
#audit Apeiron.Decider.residueRow
#audit Apeiron.Decider.residueRow_unit
#audit Apeiron.Decider.residueRow_holds_iff
#audit Apeiron.Decider.residueModulus
#audit Apeiron.Decider.residueModulus_pos
#audit Apeiron.Decider.rowDivisor_dvd_residueModulus
#audit Apeiron.Decider.satRows_residue_iff
#audit Apeiron.Decider.exists_satRows_residue
#audit Apeiron.Decider.tailRow
#audit Apeiron.Decider.value_cons
#audit Apeiron.Decider.UnitHead
#audit Apeiron.Decider.UnitHeads
#audit Apeiron.Decider.unitProject
#audit Apeiron.Decider.satRows_unitProject
#audit Apeiron.Decider.satRows_extend_unitProject
#audit Apeiron.Decider.satRows_unitProject_iff

-- The uncapped residue decision and independently checked finite refutation tree.
#audit Apeiron.Decider.residueChoices
#audit Apeiron.Decider.mem_residueChoices_iff
#audit Apeiron.Decider.completeSat
#audit Apeiron.Decider.unitHeads_residueRows
#audit Apeiron.Decider.completeSat_eq_true_iff
#audit Apeiron.Decider.completeSat_eq_false_iff
#audit Apeiron.Decider.ResidueCertificate
#audit Apeiron.Decider.checkResidueBranches
#audit Apeiron.Decider.checkResidueBranches_sound
#audit Apeiron.Decider.checkResidueCertificate
#audit Apeiron.Decider.checkResidueCertificate_decision
#audit Apeiron.Decider.checkResidueCertificate_sound
#audit Apeiron.Decider.synthesizeResidueCertificate
#audit Apeiron.Decider.synthesizeResidueCertificate_accepts
#audit Apeiron.Decider.residue_certificate_complete

-- Framed certificate streams: executable replay and framewise soundness
#audit Apeiron.Decider.CertFrame
#audit Apeiron.Decider.CertFrame.Accepted
#audit Apeiron.Decider.CertFrame.accepted_sound
#audit Apeiron.Decider.CertStream.Replays
#audit Apeiron.Decider.checkCertStream
#audit Apeiron.Decider.checkCertStream_eq_true_iff
#audit Apeiron.Decider.checkCertStream_sound
#audit Apeiron.Decider.CertRecord
#audit Apeiron.Decider.decodeRecordRow
#audit Apeiron.Decider.RecordState
#audit Apeiron.Decider.recordTransition
#audit Apeiron.Decider.RecordStep
#audit Apeiron.Decider.recordTransition_sound
#audit Apeiron.Decider.RecordSteps
#audit Apeiron.Decider.parseCertRecordsAux
#audit Apeiron.Decider.parseCertRecordsAux_sound
#audit Apeiron.Decider.parseCertRecords
#audit Apeiron.Decider.CertRecord.Frames
#audit Apeiron.Decider.parseCertRecords_sound
#audit Apeiron.Decider.checkCertRecords
#audit Apeiron.Decider.checkCertRecords_sound

-- Complete exhaustive certificates for the capped fragment
#audit Apeiron.Decider.cappedAssignments
#audit Apeiron.Decider.firstFailingRow?
#audit Apeiron.Decider.synthesizeCappedCert
#audit Apeiron.Decider.mem_cappedAssignments_iff
#audit Apeiron.Decider.capped_certificate_complete
#audit Apeiron.Decider.capped_certificate_extract
#audit Apeiron.Decider.rescaled_floor_le
#audit Apeiron.Decider.rescaled_ceil_ge
#audit Apeiron.Decider.certStep_scale

-- Finite cutting-plane derivations compile to the existing certificate language
#audit Apeiron.Decider.CuttingPlaneInference
#audit Apeiron.Decider.CuttingPlaneInference.compile
#audit Apeiron.Decider.CuttingPlaneInference.compile_correct
#audit Apeiron.Decider.CuttingPlaneInference.compile_continuation
#audit Apeiron.Decider.CuttingPlaneDerivation
#audit Apeiron.Decider.CuttingPlaneDerivation.compile
#audit Apeiron.Decider.CuttingPlaneDerivation.compile_correct
#audit Apeiron.Decider.CuttingPlaneDerivation.compile_accepts
#audit Apeiron.Decider.CuttingPlaneDerivation.refutes

-- The fold, and its headline
#audit Apeiron.Decider.run_unsat_sound
#audit Apeiron.Decider.decide_unsat_sound

-- Parity against the compiled decider, and the certificates it printed. The whole-corpus
-- agreement is a build-failing `#guard` in `Decider/Parity.lean`, not a theorem: it raises an
-- error on a false boolean and adds no declaration, so no declaration depends on
-- `Lean.ofReduceBool` and there is nothing outside the closed cone for `#audit` to collect. That
-- gate is runtime evidence on Lean's evaluation, not a proof, and `#audit` does not cover it. The
-- kernel-checked core and its no-contradiction form are audited here.
#audit Apeiron.Decider.parity_kernel
#audit Apeiron.Decider.parity_no_contradiction
#audit Apeiron.Decider.sat_verdict_is_not_sound
#audit Apeiron.Decider.cert_replay_all
#audit Apeiron.Decider.cert_rejects_tampered
#audit Apeiron.Decider.cert_rejects_truncated
#audit Apeiron.Decider.cert_rejects_altered_multiplier
#audit Apeiron.Decider.cert_accepts_unaltered

-- The face, its order, and the implicative structure that order carries
#audit Apeiron.Trichotomy.positive_face_is_one_face
#audit Apeiron.Trichotomy.no_iso_across_the_sign
#audit Apeiron.Trichotomy.ordered_positive_face_is_one_face
#audit Apeiron.Trichotomy.precedes_iff_exists_sq
#audit Apeiron.Trichotomy.precedes_no_top
#audit Apeiron.Trichotomy.idem_mul_is_meet
#audit Apeiron.Trichotomy.idem_compl_partition
#audit Apeiron.Trichotomy.conj_is_swap_not_complement
#audit Apeiron.Trichotomy.two_boolean_channels_iff_disc_pos
#audit Apeiron.Trichotomy.implApply_ofAdjunction
#audit Apeiron.Trichotomy.no_K_of_comm_assoc
#audit Apeiron.Trichotomy.idem_no_K
#audit Apeiron.Trichotomy.nodeOp_faces
#audit Apeiron.Trichotomy.nodeOp_no_K

-- A model of the calculus over the points that survive squaring

-- What a point of the algebra can say about a term

-- Feedback: the loop, the execution formula, and which faces let it close
#audit Apeiron.Trichotomy.feedback_tighten
#audit Apeiron.Trichotomy.feedback_slide
#audit Apeiron.Trichotomy.feedback_vanishing_unit
#audit Apeiron.Trichotomy.feedback_yank
#audit Apeiron.Trichotomy.det_inverse
#audit Apeiron.Trichotomy.feedback_vanishing_pair
#audit Apeiron.Trichotomy.loopSum_eq_sum
#audit Apeiron.Trichotomy.loopSum_telescopes
#audit Apeiron.Trichotomy.execution_formula
#audit Apeiron.Trichotomy.loop_solvable_iff_off_cone
#audit Apeiron.Trichotomy.loop_always_solvable_iff_disc_neg
#audit Apeiron.Trichotomy.normForm_cpow
#audit Apeiron.Trichotomy.nilpotent_iff_disc_zero
#audit Apeiron.Trichotomy.elliptic_never_terminates
#audit Apeiron.Trichotomy.unit_loop_isometric
#audit Apeiron.Trichotomy.hyperbolic_loop_grows
#audit Apeiron.Trichotomy.parabolic_failing_and_dying_loops
#audit Apeiron.Trichotomy.divergent_loop_of_disc_pos

-- Depth two: three iterates in place of the loop
#audit Apeiron.Trichotomy.loopSum_step
#audit Apeiron.Trichotomy.loopSum_second_difference
#audit Apeiron.Trichotomy.resolvent_from_three_terms
#audit Apeiron.Trichotomy.restPoint_fixed
#audit Apeiron.Trichotomy.affineIter_sub_restPoint
#audit Apeiron.Trichotomy.affineIter_first_difference
#audit Apeiron.Trichotomy.affineIter_second_difference
#audit Apeiron.Trichotomy.normForm_second_difference
#audit Apeiron.Trichotomy.aitken_exact
#audit Apeiron.Trichotomy.collapse_always_exact_iff_disc_neg
#audit Apeiron.Trichotomy.depth_two_trichotomy
#audit Apeiron.Trichotomy.affineIter_telescopes
#audit Apeiron.Trichotomy.resolvent_from_three_partial_sums

-- The replicate fold: the iterate link and the collapse read on prefix states
#audit Apeiron.Trichotomy.affineIter_eq_ringAffineFold
#audit Apeiron.Trichotomy.aitken_exact_of_prefix_fold
#audit Apeiron.Trichotomy.aitken_hyperbolic_witness
#audit Apeiron.Trichotomy.parabolic_square_zero_prefix_fold
#audit Apeiron.Trichotomy.collapse_always_exact_iff_disc_neg_prefix_fold
#audit Apeiron.Trichotomy.positive_face_prefix_fold_failure

-- The degenerate face carries a derivative
#audit Apeiron.Trichotomy.parabolic_is_leibniz
#audit Apeiron.Trichotomy.value_slot_iff_disc_zero
#audit Apeiron.Trichotomy.change_slot_linear
#audit Apeiron.Trichotomy.change_slot_smul
#audit Apeiron.Trichotomy.dual_pow
#audit Apeiron.Trichotomy.dual_taylor_two

-- The carrier a feedback calculus needs

-- The concrete interaction situation: wires on the counting numbers

-- Ports: the numeric seam, the verdict, the recursion floor and the home map
#audit Apeiron.Bridge.distinct_licence
#audit Apeiron.Physics.flowPoint_neg_one
#audit Apeiron.Physics.hasDerivAt_flowPoint_fst
#audit Apeiron.Physics.hasDerivAt_flowPoint_snd
#audit Apeiron.Physics.hasDerivAt_flowPoint
#audit Apeiron.Physics.flow_is_one_param
#audit Apeiron.Physics.generator_lipschitz
#audit Apeiron.Physics.flow_is_one_param_unique

-- Kepler's real fixed point, rooted at the second slot of the elliptic flow
#audit Apeiron.Physics.Kepler.flow_snd_neg_one
#audit Apeiron.Physics.Kepler.equation_iff_flow
#audit Apeiron.Physics.Kepler.flowPoint_eq_flow_neg_one
#audit Apeiron.Physics.Kepler.flow_neg_one_is_one_param
#audit Apeiron.Physics.Kepler.flow_neg_one_unique
#audit Apeiron.Physics.Kepler.keplerMap
#audit Apeiron.Physics.Kepler.map_eq_flow
#audit Apeiron.Physics.Kepler.equation_iff_fixed
#audit Apeiron.Physics.Kepler.map_lipschitz
#audit Apeiron.Physics.Kepler.map_strict
#audit Apeiron.Physics.Kepler.map_contracting
#audit Apeiron.Physics.Kepler.eccentricAnomaly
#audit Apeiron.Physics.Kepler.anomaly_fixed
#audit Apeiron.Physics.Kepler.anomaly_solves
#audit Apeiron.Physics.Kepler.anomaly_solves_flow
#audit Apeiron.Physics.Kepler.fixed_unique
#audit Apeiron.Physics.Kepler.anomaly_exists_unique
#audit Apeiron.Physics.Kepler.circular_solution
#audit Apeiron.Physics.Kepler.squaredResidual
#audit Apeiron.Physics.Kepler.squared_zero_iff
#audit Apeiron.Physics.Kepler.circular_joint_witness
#audit Apeiron.Physics.Kepler.positive_joint_witness
#audit Apeiron.Physics.Kepler.absolute_defect_enclosure
#audit Apeiron.Physics.Kepler.circular_enclosure_joint_witness
#audit Apeiron.Physics.Kepler.positive_enclosure_joint_witness
#audit Apeiron.Numeric.dot_abs_bounded
#audit Apeiron.Numeric.accumulator_fits
#audit Apeiron.Numeric.lossy_plus_carry_is_exact
#audit Apeiron.Numeric.lossy_residue_bounded
#audit Apeiron.Numeric.lossy_error_is_additive
#audit Apeiron.Numeric.token_survives_margin
#audit Apeiron.Numeric.output_error_is_function_distortion
#audit Apeiron.Numeric.aware_scale_is_the_chart
#audit Apeiron.Numeric.weight_only_is_dual_blind
#audit Apeiron.Numeric.osFuse_is_add
#audit Apeiron.Numeric.osFuse_fold_eq_add
#audit Apeiron.Numeric.tileTotal_perm
#audit Apeiron.Numeric.tileTotal_chunks
#audit Apeiron.Numeric.flash_attention_exact
#audit Apeiron.Numeric.softmax_shift_exact

-- Using an argument twice: the exponential

-- Tracing two loops one at a time is tracing them together

-- Three tracks on one wire

-- Using an argument twice, with the copies coming from the tensor

-- A stored route holding two routes wired to each other

-- The composing combinator

-- The exchanging combinator

-- The substituting combinator

-- Dereliction and digging

-- Functoriality

-- Splitting a reusable copy in two

-- Combinators as constant routes

-- The composing combinator's routing

-- The composing combinator as an element

-- The substituting combinator from the basis

-- Terms as routes: the denotation, and preservation with no fragment

-- What an application costs, and what vanishing does not save

-- The generic carrier readout, installed once over any commutative ring
#audit Apeiron.Physics.pointDensity
#audit Apeiron.Physics.pointFlux
#audit Apeiron.Physics.pointDensity_eq
#audit Apeiron.Physics.pointFlux_eq

-- The constructive authority: the executable basis point, its readbacks, and the root obstruction
#audit Apeiron.Physics.ringBasis
#audit Apeiron.Physics.ringNormForm_ringBasis
#audit Apeiron.Physics.ringNormForm_ringBasis_of_sq
#audit Apeiron.Physics.ringNormForm_of_snd_eq_zero
#audit Apeiron.Physics.ringNormForm_of_snd_eq_zero_of_sq
#audit Apeiron.Physics.no_sq_two_int

-- The real readback is the generic readback at the real root
#audit Apeiron.Physics.normForm_basisRealization_via_ringBasis
#audit Apeiron.Physics.ringEvolveCarrierDensity3_ringBasis_of_sq
#audit Apeiron.Physics.diffusiveUpdate_ringBasis_of_sq

-- The bounded route evaluators: a finite budget for the trace and for a path

-- The finite lattice carrier over an arbitrary commutative ring: the generic divergence and
-- rectangle balance, then the sampled ring carrier with its continuity and mass laws
#audit Apeiron.Physics.latticeCellDivergence
#audit Apeiron.Physics.sum_forward_difference_generic
#audit Apeiron.Physics.rectangle_boundary_balance_generic
#audit Apeiron.Physics.closed_rectangle_flux_cancels_generic
#audit Apeiron.Physics.ringSampledFluxFirst
#audit Apeiron.Physics.ringCarrierDensity
#audit Apeiron.Physics.ringCarrierFaceFlux
#audit Apeiron.Physics.ringCarrierDivergence
#audit Apeiron.Physics.ringEvolveCarrierDensity
#audit Apeiron.Physics.ringEvolveCarrierDensity_continuity
#audit Apeiron.Physics.sum_ringCarrierDivergence
#audit Apeiron.Physics.ringEvolveCarrierDensity_total_mass

-- Circulation on the ring: the two faces, the plaquette, the gradient and rotational
-- witnesses, and discrete Stokes
#audit Apeiron.Physics.ringBasisFaceX
#audit Apeiron.Physics.ringBasisFaceY
#audit Apeiron.Physics.ringBasisFaces_readback
#audit Apeiron.Physics.ringPlaquetteCirculation
#audit Apeiron.Physics.ringScalarGradientBasis
#audit Apeiron.Physics.ringScalarGradient_plaquetteCirculation_zero
#audit Apeiron.Physics.ringRotationalWitness
#audit Apeiron.Physics.ringRotationalWitness_plaquetteCirculation
#audit Apeiron.Physics.ringSum_forward_difference
#audit Apeiron.Physics.ringRectangleBoundaryCirculation
#audit Apeiron.Physics.ringRectangle_discrete_stokes
#audit Apeiron.Physics.ringClosed_rectangle_circulation_cancels

-- Volume on the ring: the 3D generic divergence and closed-box cancellation, then the sampled
-- density, the three face fluxes, the divergence, and the evolved density
#audit Apeiron.Physics.latticeCellDivergence3
#audit Apeiron.Physics.sum_forward_difference3_x
#audit Apeiron.Physics.sum_forward_difference3_y
#audit Apeiron.Physics.sum_forward_difference3_z
#audit Apeiron.Physics.box_divergence_balance_generic
#audit Apeiron.Physics.closed_box_flux_cancels_generic
#audit Apeiron.Physics.ringSampleDensity3
#audit Apeiron.Physics.ringSampleDensity3_eq_pointDensity
#audit Apeiron.Physics.ringCarrierDensity3
#audit Apeiron.Physics.ringSampleDensity3_eq_amplitude_sq
#audit Apeiron.Physics.ringCarrierDensity3_eq_amplitude_sq
#audit Apeiron.Physics.ringFaceFluxX
#audit Apeiron.Physics.ringFaceFluxY
#audit Apeiron.Physics.ringFaceFluxZ
#audit Apeiron.Physics.ringFaceFluxX_eq
#audit Apeiron.Physics.ringFaceFluxY_eq
#audit Apeiron.Physics.ringFaceFluxZ_eq
#audit Apeiron.Physics.ringCarrierDivergence3
#audit Apeiron.Physics.ringEvolveCarrierDensity3
#audit Apeiron.Physics.ringEvolveCarrierDensity3_continuity
#audit Apeiron.Physics.sum_const_mul3
#audit Apeiron.Physics.ringSum_carrierDivergence3
#audit Apeiron.Physics.ringEvolveCarrierDensity3_total_mass

-- Dissipation on the ring: the slope, the smoothing stencil, the diffusive flux and update,
-- and their continuity, vanishing, successor and mass laws
#audit Apeiron.Physics.slope
#audit Apeiron.Physics.prevSlope
#audit Apeiron.Physics.smoothSlope
#audit Apeiron.Physics.diffusiveFlux
#audit Apeiron.Physics.diffusiveUpdate
#audit Apeiron.Physics.diffusiveFlux_eq
#audit Apeiron.Physics.slope_diffusiveUpdate
#audit Apeiron.Physics.diffusiveUpdate_continuity
#audit Apeiron.Physics.diffusiveUpdate_zero
#audit Apeiron.Physics.diffusiveUpdate_succ
#audit Apeiron.Physics.diffusiveUpdate_total_mass

-- The scalar carrier step and its exact-half flux on the ring: the pointwise update, the local
-- balance, the doubling witness and its sign, subtraction and scalar-half forms, the net flux
-- difference as the second difference, and the three-cell averaging characterization
#audit Apeiron.Physics.scalarCarrierStep
#audit Apeiron.Physics.scalarCarrierStep_eq
#audit Apeiron.Physics.scalarCarrierStep_continuity
#audit Apeiron.Physics.DoublesTo
#audit Apeiron.Physics.exactHalfFlux
#audit Apeiron.Physics.exactHalfFlux_iff
#audit Apeiron.Physics.exactHalfFlux_neg
#audit Apeiron.Physics.doublesTo_neg
#audit Apeiron.Physics.doublesTo_sub
#audit Apeiron.Physics.exactHalfFlux_of_scalarHalf
#audit Apeiron.Physics.exactHalfFlux_difference
#audit Apeiron.Physics.ringEvolveCarrierDensity_eq_scalarCarrierStep
#audit Apeiron.Physics.sum_scalarCarrierStep_eq
#audit Apeiron.Physics.scalarCarrierStep_total_mass
#audit Apeiron.Physics.diffusiveUpdate_eq_scalarCarrierStep
#audit Apeiron.Physics.exactHalfFlux_diffusiveFlux
#audit Apeiron.Physics.diffusiveUpdate_half_three_anyRing

-- The finite body-field exchange on the ring: the site update, total momentum, the norm-form
-- ledger, and the executable transaction and rollback evaluator
#audit Apeiron.Physics.ringSitePosition
#audit Apeiron.Physics.ringFieldMomentumAfter
#audit Apeiron.Physics.ringFieldMomentumAfter_site
#audit Apeiron.Physics.ringFieldMomentumAfter_site_eq_body
#audit Apeiron.Physics.ringFieldMomentumAfter_other
#audit Apeiron.Physics.ringFieldMomentumAfter_eq_update
#audit Apeiron.Physics.ringBodyMomentumFieldAfter
#audit Apeiron.Physics.ringTotalMomentum
#audit Apeiron.Physics.ringSum_single_site
#audit Apeiron.Physics.ringTotalFieldMomentum_conserved
#audit Apeiron.Physics.ringNormFormFieldLedger
#audit Apeiron.Physics.ringNormFormFieldLedger_conserved
#audit Apeiron.Physics.ringExchangeTransaction
#audit Apeiron.Physics.ringExchangeTransaction_accepts
#audit Apeiron.Physics.ringExchangeTransaction_rejects_position
#audit Apeiron.Physics.ringExchangeTransaction_rejects_impulse
#audit Apeiron.Physics.ringExchangeOrRollback
#audit Apeiron.Physics.ringExchangeOrRollback_of_rejected_position
#audit Apeiron.Physics.ringExchangeOrRollback_of_rejected_impulse

-- The generic carrier readout at the unit direction: the dial drops out of the density and the
-- flux, and the scalar action is the algebra's product against a stretched unit
#audit Apeiron.Physics.ringMul_mk_zero_eq_smul
#audit Apeiron.Physics.pointDensity_ringOne
#audit Apeiron.Physics.pointDensity_ringOne_independent
#audit Apeiron.Physics.pointFlux_ringOne
#audit Apeiron.Physics.pointFlux_ringOne_independent

-- Weighted routes over any commutative ring: the wire monoid, the two-track retraction, the
-- trace relation and the two fuel-bounded evaluators

-- Idempotents over any commutative ring: the slot expansion, the monoid, Peirce's orthogonal and
-- complementary pairs, projections, and the two extra idempotents from a root, inverse and half
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_iff
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_zero
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_one
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_mul
#audit Apeiron.Trichotomy.AnyRing.RingOrthogonal
#audit Apeiron.Trichotomy.AnyRing.RingComplementary
#audit Apeiron.Trichotomy.AnyRing.ringOrthogonal_comm
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_comm
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_fst
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_snd
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_mul
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_add
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_snd_eq
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_fst_eq
#audit Apeiron.Trichotomy.AnyRing.ringProj
#audit Apeiron.Trichotomy.AnyRing.ringProj_proj
#audit Apeiron.Trichotomy.AnyRing.ringProj_one
#audit Apeiron.Trichotomy.AnyRing.ringProj_zero
#audit Apeiron.Trichotomy.AnyRing.ringIdemPlus
#audit Apeiron.Trichotomy.AnyRing.ringIdemMinus
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_ringIdemPlus
#audit Apeiron.Trichotomy.AnyRing.ringIdempotent_ringIdemMinus
#audit Apeiron.Trichotomy.AnyRing.ringIdemPlus_mul_ringIdemMinus
#audit Apeiron.Trichotomy.AnyRing.ringIdemPlus_add_ringIdemMinus
#audit Apeiron.Trichotomy.AnyRing.ringComplementary_idem

-- Channels and gates over any commutative ring: the read, the two coordinate reads, the null
-- basis, the fold and the two gates built from the constructed idempotents
#audit Apeiron.Trichotomy.AnyRing.RingChannel
#audit Apeiron.Trichotomy.AnyRing.ringChannel_apply
#audit Apeiron.Trichotomy.AnyRing.ringChannel_gen_sq
#audit Apeiron.Trichotomy.AnyRing.ringUpper
#audit Apeiron.Trichotomy.AnyRing.ringLower
#audit Apeiron.Trichotomy.AnyRing.ringUpper_mul
#audit Apeiron.Trichotomy.AnyRing.ringLower_mul
#audit Apeiron.Trichotomy.AnyRing.ringNormForm_eq_upper_mul_lower
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringOne
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringOne
#audit Apeiron.Trichotomy.AnyRing.ringUpper_add
#audit Apeiron.Trichotomy.AnyRing.ringLower_add
#audit Apeiron.Trichotomy.AnyRing.ringUpper_smul
#audit Apeiron.Trichotomy.AnyRing.ringLower_smul
#audit Apeiron.Trichotomy.AnyRing.ringUpper_channel
#audit Apeiron.Trichotomy.AnyRing.ringLower_channel
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringAffineApply
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringAffineApply
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringAffineFold
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringAffineFold
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringIdemPlus
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringIdemPlus
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringIdemMinus
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringIdemMinus
#audit Apeiron.Trichotomy.AnyRing.ringOfHalves
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringOfHalves
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringOfHalves
#audit Apeiron.Trichotomy.AnyRing.ringOfHalves_upper_lower
#audit Apeiron.Trichotomy.AnyRing.ringFold
#audit Apeiron.Trichotomy.AnyRing.ringUpper_fold
#audit Apeiron.Trichotomy.AnyRing.ringLower_fold
#audit Apeiron.Trichotomy.AnyRing.ringNormForm_fold
#audit Apeiron.Trichotomy.AnyRing.ringFold_perm
#audit Apeiron.Trichotomy.AnyRing.ringBoolVal
#audit Apeiron.Trichotomy.AnyRing.ringBoolVal_mul
#audit Apeiron.Trichotomy.AnyRing.ringCellOfBool
#audit Apeiron.Trichotomy.AnyRing.ringUpper_ringCellOfBool
#audit Apeiron.Trichotomy.AnyRing.ringLower_ringCellOfBool
#audit Apeiron.Trichotomy.AnyRing.ringUpper_fold_bool
#audit Apeiron.Trichotomy.AnyRing.ringLower_fold_bool
#audit Apeiron.Trichotomy.AnyRing.ringFold_is_and_and_nor

-- The scalar Cayley factor over any commutative ring: the norm form is conjugation-invariant,
-- and a Cayley factor with a carried inverse has norm form one
#audit Apeiron.Trichotomy.AnyRing.ringNormForm_conj
#audit Apeiron.Trichotomy.AnyRing.ringCayley_normForm

-- The matrix affine monoid over a finite index type and a semiring: the carrier, the step,
-- the composition, its apply and associativity laws, the identity and the chronological fold
#audit Apeiron.Trichotomy.AnyRing.MatrixAffineStep
#audit Apeiron.Trichotomy.AnyRing.matrixAffine
#audit Apeiron.Trichotomy.AnyRing.matrixStepComp
#audit Apeiron.Trichotomy.AnyRing.matrixStep_apply
#audit Apeiron.Trichotomy.AnyRing.matrixStep_comp
#audit Apeiron.Trichotomy.AnyRing.matrixStepId
#audit Apeiron.Trichotomy.AnyRing.matrixStepId_apply
#audit Apeiron.Trichotomy.AnyRing.matrixStepId_comp
#audit Apeiron.Trichotomy.AnyRing.matrixStepComp_id
#audit Apeiron.Trichotomy.AnyRing.matrixStepFold
#audit Apeiron.Trichotomy.AnyRing.matrixStepFold_apply
#audit Apeiron.Trichotomy.AnyRing.matrixStepFold_append

-- The Cayley-Dickson doubling over any commutative ring: the carrier, its components, the
-- product and involution, the generator relation, and the first-rung identification
#audit Apeiron.Doubling.AnyRing.RingDouble
#audit Apeiron.Doubling.AnyRing.ofPair
#audit Apeiron.Doubling.AnyRing.fst
#audit Apeiron.Doubling.AnyRing.snd
#audit Apeiron.Doubling.AnyRing.fst_ofPair
#audit Apeiron.Doubling.AnyRing.snd_ofPair
#audit Apeiron.Doubling.AnyRing.ext
#audit Apeiron.Doubling.AnyRing.fst_zero
#audit Apeiron.Doubling.AnyRing.snd_zero
#audit Apeiron.Doubling.AnyRing.fst_one
#audit Apeiron.Doubling.AnyRing.snd_one
#audit Apeiron.Doubling.AnyRing.fst_mul
#audit Apeiron.Doubling.AnyRing.snd_mul
#audit Apeiron.Doubling.AnyRing.fst_add
#audit Apeiron.Doubling.AnyRing.snd_add
#audit Apeiron.Doubling.AnyRing.fst_neg
#audit Apeiron.Doubling.AnyRing.snd_neg
#audit Apeiron.Doubling.AnyRing.fst_star
#audit Apeiron.Doubling.AnyRing.snd_star
#audit Apeiron.Doubling.AnyRing.star_star_eq
#audit Apeiron.Doubling.AnyRing.gen_sq
#audit Apeiron.Doubling.AnyRing.ringDoubleNorm
#audit Apeiron.Doubling.AnyRing.ringDoubleNorm_eq_ringNormForm
#audit Apeiron.Doubling.AnyRing.ringDouble_mul_eq_ringMul

-- Transport spread over an ordered ring and field: the convex mixture, non-negativity, and the
-- uniform-retention bound. The headline ratio and its two source bounds are audited above.
#audit Apeiron.Spread.Transport.a_nonneg
#audit Apeiron.Spread.Transport.a_lt_one
#audit Apeiron.Spread.Transport.IsRest
#audit Apeiron.Spread.mix_le_max
#audit Apeiron.Spread.min_le_mix
#audit Apeiron.Spread.rest_nonneg
#audit Apeiron.Spread.uniform_retention_ratio

-- The exact numerics seam over its weakest structures: the additive fusion total and the
-- calibration distortion. The reduction bounds and the output-error identity are audited above.
#audit Apeiron.Numeric.foldTotal
#audit Apeiron.Numeric.foldl_add_eq
#audit Apeiron.Numeric.foldTotal_cons
#audit Apeiron.Numeric.foldTotal_append
#audit Apeiron.Numeric.foldTotal_perm
#audit Apeiron.Numeric.foldTotal_chunks
#audit Apeiron.Numeric.functionDistortion

-- The online numerator and denominator chart over the numerics seam: the per-channel fusion,
-- the running numerator and denominator, their scan additivity, and the softmax read
#audit Apeiron.Numeric.osDen_fuse
#audit Apeiron.Numeric.osNum_fuse
#audit Apeiron.Numeric.osDen_scan_acc
#audit Apeiron.Numeric.osNum_scan_acc
#audit Apeiron.Numeric.softmax_attention_is_online_chart

-- The order-three weighted sum over any commutative ring: the explicit witness, the order-three
-- relation, the variance identity, equal-weight silence and the rephasing fixed points
#audit Apeiron.Cosmology.ringTrinitySum
#audit Apeiron.Cosmology.ringWeightSpread
#audit Apeiron.Cosmology.ringRoot_sq
#audit Apeiron.Cosmology.ringRoot_snd
#audit Apeiron.Cosmology.ringRoot_two_mul_add_one
#audit Apeiron.Cosmology.ringRoot_fst
#audit Apeiron.Cosmology.ringWeightSpread_eq_spread
#audit Apeiron.Cosmology.ringEqualWeights_silence
#audit Apeiron.Cosmology.ringFixedOfRealAxis
#audit Apeiron.Cosmology.ringRephasing_factor
#audit Apeiron.Cosmology.ringRephasing_fixes_nothing

-- The projective line over any commutative ring: the polynomial determinant, the Mobius
-- numerator and denominator, the cleared cross identity, the reciprocal with its carried
-- inverse, the cleared tangent, and the projective pair with its unit size
#audit Apeiron.Projective.Generic.ringDet2
#audit Apeiron.Projective.Generic.ringMobiusNum
#audit Apeiron.Projective.Generic.ringMobiusDen
#audit Apeiron.Projective.Generic.ringMobiusNum_conj_snd
#audit Apeiron.Projective.Generic.ringMobiusNum_cross
#audit Apeiron.Projective.Generic.ringNormForm_ringConj
#audit Apeiron.Projective.Generic.ringRecip
#audit Apeiron.Projective.Generic.ringMul_ringRecip
#audit Apeiron.Projective.Generic.ringNormForm_ringRecip
#audit Apeiron.Projective.Generic.ringMobiusTangent
#audit Apeiron.Projective.Generic.ringMobiusDen_unit_sq
#audit Apeiron.Projective.Generic.ringNormForm_ringMobiusTangent
#audit Apeiron.Projective.Generic.ringNormForm_ringMobiusTangent_mul
#audit Apeiron.Projective.Generic.RingProjPair
#audit Apeiron.Projective.Generic.ringProjPair_invertible
#audit Apeiron.Projective.Generic.ringProjScale

-- The algebraic wave over any commutative ring: the wave point, its norm form, the curvature
-- point, the finite jet and its factorization, the density flux, the four-sample identity and
-- the sign flip
#audit Apeiron.Physics.ringWave
#audit Apeiron.Physics.ringNormForm_ringWave
#audit Apeiron.Physics.ringCurv
#audit Apeiron.Physics.ringJet
#audit Apeiron.Physics.ringMul_ringCurv
#audit Apeiron.Physics.ringCurv_snd_flux
#audit Apeiron.Physics.ringStencil4_normForm
#audit Apeiron.Physics.ringNormForm_wick

-- The ring cell as a two-by-two matrix and its affine step: the representation, the vector
-- embedding, its multiplicativity, the step apply and composition, and the real-register
-- agreement with the existing `Apeiron.Physics.rep`
#audit Apeiron.Physics.ringRep
#audit Apeiron.Physics.cellVec
#audit Apeiron.Physics.toMatrixStep
#audit Apeiron.Physics.ringRep_mul
#audit Apeiron.Physics.matrixStep_apply
#audit Apeiron.Physics.matrixStep_comp
#audit Apeiron.Physics.ringRep_real_eq_rep

-- The real specialisations of the generic projective and wave statements. The real wave
-- theorems `secondDeriv_waveFst`, `secondDeriv_waveSnd`, `continuity`, `wick_flips_the_sign`
-- and `stencil4_normForm` are audited above and are not repeated.
#audit Apeiron.Projective.ProjPair.toGeneric
#audit Apeiron.Projective.det2_eq_ringDet2
#audit Apeiron.Projective.mobiusNum_eq_ringMobiusNum
#audit Apeiron.Projective.mobiusDen_eq_ringMobiusDen
#audit Apeiron.Projective.recip_eq_smul_conj
#audit Apeiron.Projective.mobiusTangent_eq_ringMobiusTangent
#audit Apeiron.Physics.curv

-- The elliptic face as the complex plane: the relabelling, the operations it carries
-- (multiplication, norm form, conjugation, flow), the affine fold and the σ = -1
-- quarter-turn order witness
#audit Apeiron.Trichotomy.ellipticEquiv
#audit Apeiron.Trichotomy.ellipticEquiv_mul
#audit Apeiron.Trichotomy.ellipticEquiv_normSq
#audit Apeiron.Trichotomy.ellipticEquiv_neg_four_witness
#audit Apeiron.Trichotomy.ellipticEquiv_conj
#audit Apeiron.Trichotomy.ellipticEquiv_flow
#audit Apeiron.Trichotomy.ellipticEquiv_affineFold
#audit Apeiron.Trichotomy.elliptic_quarter_turn_order_witness

-- The damped affine fold on the elliptic face: a flow-weighted coefficient and a
-- complementary injection, the fold size bound, and the σ = -4 joint witness
#audit Apeiron.Trichotomy.dampedStep
#audit Apeiron.Trichotomy.damped_fold_bounded
#audit Apeiron.Trichotomy.damped_step_joint_witness

-- The Cayley step on a complex inner-product space: the skew real part vanishes, and the
-- conditional relation preserves the norm, both locally and from full skew-adjointness
#audit Apeiron.Trichotomy.skew_re_zero
#audit Apeiron.Trichotomy.cayley_isometry_of_re_zero
#audit Apeiron.Trichotomy.cayley_isometry
