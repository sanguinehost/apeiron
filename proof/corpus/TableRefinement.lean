import ApeironWord
import Apeiron.Table.Basic
import Apeiron.Table.Adjacency
import Apeiron.Table.Home

/-!
# Executing-table specification frontier

The canonical table is a total function. These theorems establish the constant-default and
new-first insertion contracts and exhibit both resource-refusal constructors. They do not refine
`src/table.rs`: Aeneas rejects its stored function pointer before generating `table.Table`.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Apeiron.Table

variable {K V : Type}

/-- The canonical state denoted by a newly constructed table. -/
def newTable (default : V) : HMap K V := fun _ => default

/-- A new canonical table reads its default at every key. -/
theorem newTable_default_read (default : V) (key : K) :
    Apeiron.Table.get (newTable (K := K) default) key = default := rfl

/-- The first write is new-first and merges against the default. -/
theorem newTable_insertWith_self [DecidableEq K] (merge : V → V → V)
    (default : V) (key : K) (value : V) :
    Apeiron.Table.get (insertWith merge key value (newTable default)) key = merge value default := by
  rw [get_insertWith_self]
  rfl

/-- The complete first-write function: the written key is merged and every other key stays default. -/
theorem newTable_insertWith [DecidableEq K] (merge : V → V → V)
    (default : V) (key : K) (value : V) :
    insertWith merge key value (newTable default) =
      fun x => if x = key then merge value default else default := by
  rfl

/-- One term jointly witnesses initialization, default read, and new-first insertion. -/
theorem newTable_joint_witness [DecidableEq K] (merge : V → V → V)
    (default : V) (key : K) (value : V) :
    ∃ h : HMap K V,
      (∀ x, Apeiron.Table.get h x = default) ∧
      Apeiron.Table.get (insertWith merge key value h) key = merge value default := by
  refine ⟨newTable default, ?_, newTable_insertWith_self merge default key value⟩
  intro x
  exact newTable_default_read default x

/-- A resource refusal preserves the logical table. -/
def Refusal (error : apeiron_word.table.StorageError) (before after : HMap K V) : Prop :=
  after = before ∧ (error = .CapacityOverflow ∨ error = .AllocationFailed)

/-- The capacity-overflow refusal branch is inhabited with unchanged logical state. -/
theorem capacityOverflow_refusal_witness (h : HMap K V) :
    apeiron_word.word_storage_capacity_overflow =
        .ok .CapacityOverflow ∧
      ∃ after, Refusal .CapacityOverflow h after :=
  ⟨rfl, h, rfl, Or.inl rfl⟩

/-- The allocation-failure refusal branch is inhabited with unchanged logical state. -/
theorem allocationFailed_refusal_witness (h : HMap K V) :
    apeiron_word.word_storage_allocation_failed =
        .ok .AllocationFailed ∧
      ∃ after, Refusal .AllocationFailed h after :=
  ⟨rfl, h, rfl, Or.inr rfl⟩

end Apeiron.TableRefinement
