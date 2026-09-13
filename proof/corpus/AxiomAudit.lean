import WordRefinement
import SafeTableRefinement
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

-- The extracted safe table search: the first total loop.spec_decr_nat refinement
#audit apeiron_word.safe_table.find_from_loop_spec
#audit apeiron_word.safe_table.find_from_spec
#audit apeiron_word.safe_table.findIdx_spec
#audit apeiron_word.safe_table.findIdx_none_iff
#audit apeiron_word.safe_table.get_spec
#audit apeiron_word.safe_table.get_spec_canonical
#audit apeiron_word.safe_table.empty_from_loop_spec
#audit apeiron_word.safe_table.empty_from_spec
#audit apeiron_word.safe_table.findIdx_set_some_self
#audit apeiron_word.safe_table.findIdx_set_some_ne
#audit apeiron_word.safe_table.tableHMap_set_self
#audit apeiron_word.safe_table.tableHMap_set_ne
#audit apeiron_word.safe_table.tableHMap_set_none
#audit apeiron_word.safe_table.tableHMap_set_none_self
#audit apeiron_word.safe_table.try_insert_spec
#audit apeiron_word.safe_table.remove_spec
#audit apeiron_word.safe_table.keysUnique_set_none
#audit apeiron_word.safe_table.keysUnique_set_some

-- The generated empty-state witness, its denotation, and the insertion branch witnesses
#audit apeiron_word.safe_table.safe_table_new_eq_emptyTable
#audit apeiron_word.safe_table.word_safe_table4_new_eq_emptyTable
#audit apeiron_word.safe_table.safe_table_new_wellFormed
#audit apeiron_word.safe_table.emptyTable_wellFormed
#audit apeiron_word.safe_table.emptyTable_tableHMap
#audit apeiron_word.safe_table.try_insert_emptyTable_success
#audit apeiron_word.safe_table.oneTable_wellFormed
#audit apeiron_word.safe_table.try_insert_oneTable_replace
#audit apeiron_word.safe_table.try_insert_refusal
#audit apeiron_word.safe_table.try_insert_fullTable_refusal
