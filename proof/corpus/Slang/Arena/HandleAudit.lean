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
import Slang.Audit.Empty
import Slang.Arena.StateResult

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Arena.HandleAudit

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

end Slang.Arena.HandleAudit

open Slang.Arena
open Slang.Arena.HandleAudit

-- Bounded.lean, the type, its projections, the helper and the theorems.
#audit_empty Slang.Arena.Arena : Type → Nat → Type
#audit_empty Slang.Arena.Arena.entries :
  {α : Type} → {capacity : Nat} → Arena α capacity → List α
#audit_empty Slang.Arena.Arena.bounded :
  {α : Type} → {capacity : Nat} → (self : Arena α capacity) → self.entries.length ≤ capacity
#audit_empty Slang.Arena.Handle :
  {α : Type} → {capacity : Nat} → Arena α capacity → Type
#audit_empty Slang.Arena.resolve :
  {α : Type} → {capacity : Nat} → (arena : Arena α capacity) → Handle arena → α
#audit_empty Slang.Arena.length_append_singleton :
  ∀ {α : Type} (l : List α) (value : α), (l ++ [value]).length = l.length + 1
#audit_empty Slang.Arena.Allocation :
  {α : Type} → {capacity : Nat} → Arena α capacity → α → Type
#audit_empty Slang.Arena.Allocation.after :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    Allocation before value → Arena α capacity
#audit_empty Slang.Arena.Allocation.handle :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    (self : Allocation before value) → Handle self.after
#audit_empty Slang.Arena.Allocation.extends :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    (self : Allocation before value) → self.after.entries = before.entries ++ [value]
#audit_empty Slang.Arena.Allocation.pointsTo :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    (self : Allocation before value) → resolve self.after self.handle = value
#audit_empty Slang.Arena.AllocationResult :
  {α : Type} → {capacity : Nat} → Arena α capacity → α → Type
#audit_empty Slang.Arena.AllocationResult.allocated :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    Allocation before value → AllocationResult before value
#audit_empty Slang.Arena.AllocationResult.capacity :
  {α : Type} → {capacity : Nat} → {before : Arena α capacity} → {value : α} →
    before.entries.length = capacity → AllocationResult before value
#audit_empty Slang.Arena.allocate :
  {α : Type} → {capacity : Nat} → (before : Arena α capacity) → (value : α) →
    AllocationResult before value
#audit_empty Slang.Arena.allocate_capacity_iff :
  ∀ {α : Type} {capacity : Nat} (arena : Arena α capacity) (value : α),
    (∃ full, allocate arena value = .capacity full) ↔ arena.entries.length = capacity
#audit_empty Slang.Arena.Allocation.old_resolves_unchanged :
  ∀ {α : Type} {capacity : Nat} {before : Arena α capacity} {value : α}
    (allocation : Allocation before value) (handle : Handle before),
    resolve allocation.after
        ⟨handle.val, allocation.extends ▸ Nat.lt_of_lt_of_le handle.isLt
          (by rw [length_append_singleton]; exact Nat.le_add_right _ 1)⟩ =
      resolve before handle

-- StateResult.lean, the record types, specializations and handles.
#audit_empty Slang.Arena.StateRecord : Type
#audit_empty Slang.Arena.StateRecord.environment :
  StateRecord → Slang.Machine.Environment
#audit_empty Slang.Arena.StateRecord.memory : StateRecord → Slang.Machine.Memory
#audit_empty Slang.Arena.StateRecord.callDepth : StateRecord → Nat
#audit_empty Slang.Arena.ResultRecord : Type
#audit_empty Slang.Arena.ResultRecord.raw : ResultRecord → Slang.Word
#audit_empty Slang.Arena.StateArena : Nat → Type
#audit_empty Slang.Arena.ResultArena : Nat → Type
#audit_empty Slang.Arena.StateHandle :
  {capacity : Nat} → StateArena capacity → Type
#audit_empty Slang.Arena.ResultHandle :
  {capacity : Nat} → ResultArena capacity → Type

-- The fixtures themselves.
#audit_empty Slang.Arena.HandleAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Arena.HandleAudit.beforeFixture : Arena Nat 2
#audit_empty Slang.Arena.HandleAudit.afterFixture : Arena Nat 2
#audit_empty Slang.Arena.HandleAudit.allocationFixture : Allocation beforeFixture 9
#audit_empty Slang.Arena.HandleAudit.fresh_resolves :
  resolve allocationFixture.after allocationFixture.handle = 9
#audit_empty Slang.Arena.HandleAudit.old_handle_unchanged :
  resolve allocationFixture.after ⟨0, by decide⟩ = resolve beforeFixture ⟨0, by decide⟩
#audit_empty Slang.Arena.HandleAudit.fullFixture : Arena Nat 1
#audit_empty Slang.Arena.HandleAudit.full_refuses :
  allocate fullFixture 8 = .capacity (by rfl)
