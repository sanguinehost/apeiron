import Growable.Relocation
import Growable.Search

/-!
# Rebuild commit boundary

The relocation traversal and the pending commit are separate extracted roots.  This module identifies
the pending commit with the already refined straight-line insert apply on the rebuilt storage.  It
does not supply the unextracted planner, initialized-resource, or search-decision certificates.
-/

namespace Apeiron.GrowableRefinement

open Aeneas Aeneas.Std Result
open apeiron_word

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-- View materialized rebuilt storage as a table without changing committed scalar fields. -/
def rebuiltView {R : Type}
    (state : table.Table I64 I64 table.WordHome R 16#usize)
    (reserved : table.RebuildReserved R I64) :
    table.Table I64 I64 table.WordHome R 16#usize :=
  { state with resident := reserved.resident, shards := reserved.regions }

/-- Project the committed branch of the ordinary extracted insert apply. -/
def committedResult {R : Type}
    (run : Result (table.Table I64 I64 table.WordHome R 16#usize × Option (I64 × I64))) :
    Result (table.Table I64 I64 table.WordHome R 16#usize) := do
  let (next, pending) ← run
  match pending with
  | none => ok next
  | some _ => fail Error.panic

private theorem committedProjection {R : Type}
    (next : table.Table I64 I64 table.WordHome R 16#usize) :
    ok next = (do
      let result ←
        ok ((table.ApplyOutcome.Committed : table.ApplyOutcome I64 I64), next)
      let (state, pending) ← match result.1 with
        | .Committed => ok (result.2, none)
        | .NeedsGrowth key value => ok (result.2, some (key, value))
      match pending with
  | none => ok state
  | some _ => fail Error.panic) := by
  rw [bind_tc_ok, bind_tc_ok]
  rfl

/--
The stored-carrier rebuild commit is the ordinary extracted insert apply on `rebuiltView`.  The
carrier equation is the input-side pending carrier certificate from `RebuildPlanCorrect`.
-/
theorem word_stored_apply_pending_and_commit_eq_apply_insert
    (state : table.Table I64 I64 table.WordHome (table.keyrep.Stored I64) 16#usize)
    (reserved : table.RebuildReserved (table.keyrep.Stored I64) I64)
    (decision : table.SearchDecision) (sigma : cell.Sigma) (key value : I64)
    (held : table.keyrep.Stored I64) (address : table.TableAddress)
    (hclass : decision.class = .Found ∨ decision.class = .Vacant)
    (hstaged : state.locator.staged = none)
    (hhold :
      Apeiron.GrowableIdentity.storedKeyRep.hold sigma key address = ok held) :
    word_stored_apply_pending_and_commit state reserved decision sigma value held address =
      committedResult
        (word_stored_apply_insert (rebuiltView state reserved) decision sigma key value address) := by
  have hheld : held = ({ key := key } : table.keyrep.Stored I64) := by
    have h := hhold
    simp [table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold] at h
    exact h.symm
  subst held
  have hlocator : { state.locator with staged := none } = state.locator := by
    cases hlocatorEq : state.locator with
    | mk profile directory links staged =>
        rw [hlocatorEq] at hstaged
        simp only at hstaged ⊢
        subst staged
        rfl
  rcases hclass with hfound | hvacant <;>
    cases hlocation : decision.location <;>
    simp_all [word_stored_apply_pending_and_commit, table.Table.apply_pending_and_commit,
      word_stored_apply_insert, table.Table.apply_insert, rebuiltView, committedResult,
      table.TableLocator.publish_for_held, table.TableLocator.publish_for_query,
      table.keyrep.Stored.Insts.Apeiron_wordTableKeyrepKeyRep.hold,
      table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
      alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
      alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
  all_goals repeat' first | split <;> simp_all
  all_goals cases_type Prod
  all_goals cases_type Prod
  all_goals exact committedProjection _

/--
The regenerating-carrier rebuild commit is the ordinary extracted insert apply on `rebuiltView`.
`hhold` names the codec/address carrier equation rather than assuming the resulting table relation.
-/
theorem word_regen_apply_pending_and_commit_eq_apply_insert
    (state : table.Table I64 I64 table.WordHome
      (table.keyrep.Regen table.keyrep.I64WordCodec) 16#usize)
    (reserved : table.RebuildReserved (table.keyrep.Regen table.keyrep.I64WordCodec) I64)
    (decision : table.SearchDecision) (sigma : cell.Sigma) (key value : I64)
    (held : table.keyrep.Regen table.keyrep.I64WordCodec) (address : table.TableAddress)
    (hclass : decision.class = .Found ∨ decision.class = .Vacant)
    (hstaged : state.locator.staged = none)
    (hhold :
      Apeiron.GrowableIdentity.regenKeyRep.hold sigma key address = ok held) :
    word_regen_apply_pending_and_commit state reserved decision sigma value held address =
      committedResult
        (word_regen_apply_insert (rebuiltView state reserved) decision sigma key value address) := by
  have hlocator : { state.locator with staged := none } = state.locator := by
    cases hlocatorEq : state.locator with
    | mk profile directory links staged =>
        rw [hlocatorEq] at hstaged
        simp only at hstaged ⊢
        subst staged
        rfl
  rcases hclass with hfound | hvacant <;>
    cases hlocation : decision.location <;>
    simp_all [word_regen_apply_pending_and_commit, table.Table.apply_pending_and_commit,
      word_regen_apply_insert, table.Table.apply_insert, rebuiltView, committedResult,
      table.TableLocator.publish_for_held, table.TableLocator.publish_for_query,
      table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge,
      alloc.vec.Vec.index_slice_index, alloc.vec.Vec.index_mut_slice_index,
      alloc.vec.Vec.index_usize, alloc.vec.Vec.index_mut_usize]
  all_goals repeat' first | split <;> simp_all
  all_goals cases_type Prod
  all_goals cases_type Prod
  all_goals exact committedProjection _

end Apeiron.GrowableRefinement
