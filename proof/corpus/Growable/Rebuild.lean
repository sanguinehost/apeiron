import Growable.Representation
import Growable.Relocation
import Growable.Resources

/-!
# Whole rebuild relocation boundary

Per-entry relocation is proved from the extracted body in `Growable.Relocation`.  A complete
traversal changes the table locator at every successful `retarget_held`, and spill publication also
changes the region summary.  The unextracted planner must therefore provide the successful
traversal equation together with the resulting whole-table representation; exact source-state
stutter and summary-blind symbolic-prefix equations are not valid contracts for the current body.
-/

namespace Apeiron.GrowableRebuild

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

abbrev StoredI64 := table.keyrep.Stored I64
abbrev RegenI64 := table.keyrep.Regen table.keyrep.I64WordCodec
abbrev LogicalKey := Apeiron.GrowableRepresentation.LogicalKey
abbrev WordTable := Apeiron.GrowableRepresentation.WordTable

/--
Successful whole-traversal evidence supplied by the planner/resource boundary.  `run` names the
actual extracted traversal; `payloadState` records the locator-only source-table mutation, while
`reservedRep` keeps the target rooted in canonical `Apeiron.Table.HMap` semantics.
-/
structure TraversalEffectsPrepared {R : Type}
    (run : Result (WordTable R × table.RebuildReserved R I64))
    (state : WordTable R) (decode : cell.Sigma → R → Option I64)
    (direct : LogicalKey → Option Apeiron.GrowableRepresentation.Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64) where
  nextState : WordTable R
  nextStorage : table.RebuildReserved R I64
  run_eq : run = ok (nextState, nextStorage)
  payloadState : Apeiron.GrowableRelocation.PayloadStateEq state nextState
  reservedRep : Apeiron.GrowableRepresentation.ReservedRepR nextState nextStorage decode direct live
    logical

/-- Stored relocation returns its successful locator-mutated state and canonical rebuilt map. -/
theorem word_stored_rebuild_preserves
    (state : WordTable StoredI64) (plan : table.RebuildPlan)
    (storage : table.RebuildReserved StoredI64 I64)
    (decode : cell.Sigma → StoredI64 → Option I64)
    (direct : LogicalKey → Option Apeiron.GrowableRepresentation.Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64)
    (effects : TraversalEffectsPrepared
      (word_stored_relocate_all state plan storage) state decode direct live logical) :
    word_stored_relocate_all state plan storage ⦃ result =>
      Apeiron.GrowableRelocation.PayloadStateEq state result.1 ∧
        Apeiron.GrowableRepresentation.ReservedRepR result.1 result.2 decode direct live logical ⦄ := by
  rw [effects.run_eq, WP.spec_ok]
  exact ⟨effects.payloadState, effects.reservedRep⟩

/-- Regenerating relocation has the same successful locator-mutated canonical postcondition. -/
theorem word_regen_rebuild_preserves
    (state : WordTable RegenI64) (plan : table.RebuildPlan)
    (storage : table.RebuildReserved RegenI64 I64)
    (decode : cell.Sigma → RegenI64 → Option I64)
    (direct : LogicalKey → Option Apeiron.GrowableRepresentation.Location)
    (live : LogicalKey → Prop) (logical : Apeiron.Table.HMap LogicalKey I64)
    (effects : TraversalEffectsPrepared
      (word_regen_relocate_all state plan storage) state decode direct live logical) :
    word_regen_relocate_all state plan storage ⦃ result =>
      Apeiron.GrowableRelocation.PayloadStateEq state result.1 ∧
        Apeiron.GrowableRepresentation.ReservedRepR result.1 result.2 decode direct live logical ⦄ := by
  rw [effects.run_eq, WP.spec_ok]
  exact ⟨effects.payloadState, effects.reservedRep⟩

end Apeiron.GrowableRebuild
