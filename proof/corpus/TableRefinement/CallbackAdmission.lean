import TableRefinement.CallbackPhysical

/-! # Exact callback admission for arbitrary insertion

This module composes the generated branch equations with their physical-preservation theorems.
The postcondition is stated independently of execution: success refines canonical `insertWith`,
while `ShardFull` returns the input state exactly and retains its invariant.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- Exact, non-circular postcondition for one generic callback insertion. -/
def CallbackInsertPostG {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (t : table.Table K V H N) (h : HMap K V) (mergeFn : V → V → V)
    (key : K) (value : V)
    (result : core.result.Result Unit table.StorageError × table.Table K V H N) : Prop :=
  match result.1 with
  | core.result.Result.Ok _ =>
      PhysicalInvG cmp HomeInst result.2 (insertWith mergeFn key value h)
  | core.result.Result.Err table.StorageError.ShardFull =>
      result.2 = t ∧ PhysicalInvG cmp HomeInst result.2 h

/-- The real generated generic insertion satisfies the exact physical/canonical postcondition for
    every total callback implementation obeying `CallbackLawsG`. -/
theorem insert_with_callback_admitted {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value : V)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h) :
    Aeneas.Std.WP.spec
      (table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value)
      (CallbackInsertPostG cmp HomeInst t h mergeFn key value) := by
  have hinvCopy := hinv
  obtain ⟨shard, hshard, hpresent | habsent⟩ :=
    table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst t merge homeFn mergeFn
      h key laws hinv
  · obtain ⟨slots, i, old, hfind, houter, hslot, hunique⟩ := hpresent
    have hexact := insert_with_present_callback_exact cmp cloneK cloneV HomeInst MergeInst
      t merge homeFn mergeFn key key value old shard i slots laws hshard hfind houter hslot
    have hpreserve := physicalInvG_replacePresent cmp cloneK cloneV HomeInst MergeInst t merge
      homeFn mergeFn h key value old shard i slots laws hinvCopy
      ⟨houter, hslot, hunique⟩
    rw [hexact, Aeneas.Std.WP.spec_ok]
    simpa [CallbackInsertPostG] using hpreserve
  · obtain ⟨habsentKey, hfind, houterEmpty | ⟨slots, houter, hvacant | hfull⟩⟩ := habsent
    · by_cases hN : 0#usize < N
      · have hshardBound : shard < 256#usize := by
          obtain ⟨routed, hrouted, hbound⟩ :=
            shard_index_callback_spec HomeInst t homeFn key (laws.home_total key)
          have heq : routed = shard :=
            Aeneas.Std.Result.ok_injective (hrouted.symm.trans hshard)
          subst routed
          simpa using hbound
        have hexact := insert_with_empty_shard_callback_exact cmp cloneK cloneV HomeInst
          MergeInst t merge homeFn mergeFn key value shard laws hshard hfind houterEmpty hN
          hshardBound
        have hpreserve := physicalInvG_empty_shard cmp cloneK cloneV HomeInst MergeInst t merge
          homeFn mergeFn h key value shard laws hinvCopy habsentKey hshard hfind houterEmpty hN
        rw [hexact, Aeneas.Std.WP.spec_ok]
        simpa [CallbackInsertPostG] using hpreserve
      · have hzero : N = 0#usize := by scalar_tac
        subst N
        have hexact := insert_with_zero_callback_exact cmp cloneV HomeInst MergeInst t merge
          key value shard hshard hfind
        rw [hexact, Aeneas.Std.WP.spec_ok]
        exact ⟨rfl, hinvCopy⟩
    · obtain ⟨first, hfirst⟩ := hvacant
      have hexact := insert_with_populated_vacant_callback_exact cmp cloneK cloneV HomeInst
        MergeInst t merge homeFn mergeFn key value shard first slots laws hshard hfind houter
        hfirst.1 hfirst.2.1 hfirst.2.2
      have hpreserve := physicalInvG_populated_vacant cmp cloneK cloneV HomeInst MergeInst t
        merge homeFn mergeFn h key value shard first slots laws hinvCopy habsentKey hshard hfind
        houter hfirst
      rw [hexact, Aeneas.Std.WP.spec_ok]
      simpa [CallbackInsertPostG] using hpreserve
    · by_cases hN : 0#usize < N
      · have hexact := insert_with_populated_full_callback_exact cmp cloneV HomeInst MergeInst
          t merge key value shard slots hN hshard hfind houter hfull
        rw [hexact, Aeneas.Std.WP.spec_ok]
        exact ⟨rfl, hinvCopy⟩
      · have hzero : N = 0#usize := by scalar_tac
        subst N
        have hexact := insert_with_zero_callback_exact cmp cloneV HomeInst MergeInst t merge
          key value shard hshard hfind
        rw [hexact, Aeneas.Std.WP.spec_ok]
        exact ⟨rfl, hinvCopy⟩

end Apeiron.TableRefinement
