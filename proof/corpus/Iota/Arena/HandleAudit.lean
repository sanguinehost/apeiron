/-
# Empty-cone audit for the bounded state/result arenas

Imports the two arena modules and registers every public declaration, every
projection, both constructors, the append helper and the two theorems with
`#audit_empty`. The command rejects a missing name, an `axiom`, a type mismatch
and any nonempty `Lean.collectAxioms` set, so the whole population must be
axiom-free.

The concrete fixtures below exercise the append and refusal behavior: a fresh
allocation resolves to the appended value, an old handle resolves unchanged,
and a full arena's `allocate` takes the `.capacity` branch. The rejecting
control (an allocation that reuses handle 0, or one that allocates at
capacity) is a throwaway check, not a declaration here.
-/
import Iota.Audit.Empty
import Iota.Arena.StateResult

set_option autoImplicit false
set_option warningAsError true

namespace Iota.Arena.HandleAudit

/-- Positive fixture: `rfl` uses no axiom. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

/-- One-slot before-arena. -/
def beforeFixture : Arena Nat 2 := ⟨[5], Nat.le_succ 1⟩

/-- The after-arena with `9` appended. -/
def afterFixture : Arena Nat 2 := ⟨[5, 9], Nat.le_refl 2⟩

/-- A concrete successful allocation of `9` into `beforeFixture`. -/
def allocationFixture : Allocation beforeFixture 9 where
  after := afterFixture
  handle := ⟨1, by decide⟩
  «extends» := rfl
  pointsTo := rfl

/-- A fresh allocation resolves to the appended value. -/
theorem fresh_resolves : resolve allocationFixture.after allocationFixture.handle = 9 :=
  allocationFixture.pointsTo

/-- An old handle resolves unchanged across the allocation. -/
theorem old_handle_unchanged :
    resolve allocationFixture.after ⟨0, by decide⟩ = resolve beforeFixture ⟨0, by decide⟩ :=
  Allocation.old_resolves_unchanged allocationFixture ⟨0, by decide⟩

/-- A one-slot arena holding `3`, at capacity. -/
def fullFixture : Arena Nat 1 := ⟨[3], Nat.le_refl 1⟩

/-- A full arena's `allocate` returns the `.capacity` branch. -/
theorem full_refuses : allocate fullFixture 8 = .capacity (by rfl) := by
  unfold allocate
  rw [dif_pos (by rfl)]

end Iota.Arena.HandleAudit

open Iota.Arena
open Iota.Arena.HandleAudit

-- Bounded.lean, the type, its projections, the helper and the theorems.
#audit_empty Iota.Arena.Arena : Type → Nat → Type
#audit_empty Iota.Arena.Arena.entries :
  {α : Type} → {capacity : Nat} → Arena α capacity → List α
#audit_empty Iota.Arena.Arena.bounded :
  {α : Type} → {capacity : Nat} → (self : Arena α capacity) → self.entries.length ≤ capacity
#audit_empty Iota.Arena.Handle :
  {α : Type} → {capacity : Nat} → Arena α capacity → Type
#audit_empty Iota.Arena.resolve :
  {α : Type} → {capacity : Nat} → (arena : Arena α capacity) → Handle arena → α
#audit_empty Iota.Arena.length_append_singleton :
  ∀ {α : Type} (l : List α) (value : α), (l ++ [value]).length = l.length + 1
#audit_empty Iota.Arena.Allocation :
  {α : Type} → {capacity : Nat} → Arena α capacity → α → Type
#audit_empty Iota.Arena.Allocation.after :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    Allocation before value → Arena α capacity
#audit_empty Iota.Arena.Allocation.handle :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    (self : Allocation before value) → Handle self.after
#audit_empty Iota.Arena.Allocation.extends :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    (self : Allocation before value) → self.after.entries = before.entries ++ [value]
#audit_empty Iota.Arena.Allocation.pointsTo :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    (self : Allocation before value) → resolve self.after self.handle = value
#audit_empty Iota.Arena.AllocationResult :
  {α : Type} → {capacity : Nat} → Arena α capacity → α → Type
#audit_empty Iota.Arena.AllocationResult.allocated :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    Allocation before value → AllocationResult before value
#audit_empty Iota.Arena.AllocationResult.capacity :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    before.entries.length = capacity → AllocationResult before value
#audit_empty Iota.Arena.allocate :
  {α : Type} → {capacity : Nat} → (before : Arena α capacity) → (value : α) →
    AllocationResult before value
#audit_empty Iota.Arena.allocate_capacity_iff :
  ∀ {α : Type} {capacity : Nat} (arena : Arena α capacity) (value : α),
    (∃ full, allocate arena value = .capacity full) ↔ arena.entries.length = capacity
#audit_empty Iota.Arena.Allocation.old_resolves_unchanged :
  ∀ {α : Type} {capacity : Nat} {before : Arena α capacity} {value : α}
    (allocation : Allocation before value) (handle : Handle before),
    resolve allocation.after
        ⟨handle.val, allocation.extends ▸ Nat.lt_of_lt_of_le handle.isLt
          (by rw [length_append_singleton]; exact Nat.le_add_right _ 1)⟩ =
      resolve before handle

-- StateResult.lean, the record types, specializations and handles.
#audit_empty Iota.Arena.StateRecord : Type
#audit_empty Iota.Arena.StateRecord.environment :
  StateRecord → Iota.Machine.Environment
#audit_empty Iota.Arena.StateRecord.memory : StateRecord → Iota.Machine.Memory
#audit_empty Iota.Arena.StateRecord.callDepth : StateRecord → Nat
#audit_empty Iota.Arena.ResultRecord : Type
#audit_empty Iota.Arena.ResultRecord.raw : ResultRecord → Iota.Word
#audit_empty Iota.Arena.StateArena : Nat → Type
#audit_empty Iota.Arena.ResultArena : Nat → Type
#audit_empty Iota.Arena.StateHandle :
  {capacity : Nat} → StateArena capacity → Type
#audit_empty Iota.Arena.ResultHandle :
  {capacity : Nat} → ResultArena capacity → Type

-- The fixtures themselves.
#audit_empty Iota.Arena.HandleAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Arena.HandleAudit.beforeFixture : Arena Nat 2
#audit_empty Iota.Arena.HandleAudit.afterFixture : Arena Nat 2
#audit_empty Iota.Arena.HandleAudit.allocationFixture : Allocation beforeFixture 9
#audit_empty Iota.Arena.HandleAudit.fresh_resolves :
  resolve allocationFixture.after allocationFixture.handle = 9
#audit_empty Iota.Arena.HandleAudit.old_handle_unchanged :
  resolve allocationFixture.after ⟨0, by decide⟩ = resolve beforeFixture ⟨0, by decide⟩
#audit_empty Iota.Arena.HandleAudit.fullFixture : Arena Nat 1
#audit_empty Iota.Arena.HandleAudit.full_refuses :
  allocate fullFixture 8 = .capacity (by rfl)
