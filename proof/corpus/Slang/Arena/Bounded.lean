/-
# Bounded arenas and their handles

A fixed-capacity append-only arena. `entries` carries a length proof, so a
handle is a `Fin` index into the arena it was drawn from and an out-of-range
handle is unconstructible by type. `allocate` either appends the value once or
refuses with the proof that the arena is full; there is no probing and no
reuse of an existing slot.

The `Allocation` structure is the positive branch witness: it carries the
after-arena, the new handle, the append equation and the resolution equation.
The cone here is empty. The core append lemmas `List.length_append` and
`List.getElem_concat_length` depend on `propext`, so this module proves the two
append facts it needs directly from `List.length_cons`,
`List.getElem_cons_zero` and `List.getElem_cons_succ`, all axiom-free.
-/
import Slang.Machine

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Arena

/-- An append-only arena with a fixed capacity and its length bound. -/
structure Arena (α : Type) (capacity : Nat) where
  entries : List α
  bounded : entries.length ≤ capacity

/-- A handle into a specific arena: an index below the arena's current length. -/
abbrev Handle {α : Type} {capacity : Nat} (arena : Arena α capacity) := Fin arena.entries.length

/-- Resolve a handle to the value stored at its index. -/
def resolve {α : Type} {capacity : Nat} (arena : Arena α capacity) (handle : Handle arena) : α :=
  arena.entries.get handle

/-- Appending one element raises the length by one. Stated without
`List.length_append`, whose proof uses `propext`. -/
theorem length_append_singleton {α : Type} (l : List α) (value : α) :
    (l ++ [value]).length = l.length + 1 := by
  induction l with
  | nil => rfl
  | cons head tail ih =>
      change (tail ++ [value]).length + 1 = (tail.length + 1) + 1
      rw [ih]

/-- The element appended at the end resolves to that element. Written with the
explicit `GetElem.getElem` so no `get_elem_tactic` proof enters the term. -/
private theorem getElem_append_singleton {α : Type} (l : List α) (value : α)
    (h : l.length < (l ++ [value]).length) :
    GetElem.getElem (l ++ [value]) l.length h = value := by
  induction l with
  | nil =>
      exact List.getElem_cons_zero value [] h
  | cons head tail ih =>
      change GetElem.getElem (head :: (tail ++ [value])) (tail.length + 1) _ = value
      rw [List.getElem_cons_succ]
      exact ih _

/-- A successful allocation: the after-arena, the fresh handle, the append
equation and the equation that the handle resolves to the appended value. -/
structure Allocation {α : Type} {capacity : Nat} (before : Arena α capacity) (value : α) where
  after : Arena α capacity
  handle : Handle after
  /-- The after-arena is exactly the before-arena with the value appended. -/
  «extends» : after.entries = before.entries ++ [value]
  /-- The fresh handle resolves to the appended value. -/
  pointsTo : resolve after handle = value

/-- The two branches of an allocation attempt. -/
inductive AllocationResult {α : Type} {capacity : Nat} (before : Arena α capacity) (value : α) where
  /-- The value was appended and a fresh handle returned. -/
  | allocated (allocation : Allocation before value)
  /-- The arena was full; the proof records that the length equals the capacity. -/
  | capacity (full : before.entries.length = capacity)

/-- Append `value` once when there is room, otherwise refuse with the full proof. -/
def allocate {α : Type} {capacity : Nat} (before : Arena α capacity) (value : α) :
    AllocationResult before value :=
  if hfull : before.entries.length = capacity then
    .capacity hfull
  else
    .allocated
      { after :=
          { entries := before.entries ++ [value]
            bounded := by
              rw [length_append_singleton]
              exact Nat.lt_of_le_of_ne before.bounded hfull }
        handle :=
          ⟨before.entries.length, by
            show before.entries.length < (before.entries ++ [value]).length
            rw [length_append_singleton]
            exact Nat.lt_succ_self _⟩
        «extends» := rfl
        pointsTo := by
          unfold resolve
          rw [List.get_eq_getElem]
          exact getElem_append_singleton before.entries value _ }

/-- `allocate` returns `.capacity` exactly when the arena is full. -/
theorem allocate_capacity_iff {α : Type} {capacity : Nat} (arena : Arena α capacity) (value : α) :
    (∃ full, allocate arena value = .capacity full) ↔ arena.entries.length = capacity := by
  constructor
  · rintro ⟨full, h⟩
    by_cases hfull : arena.entries.length = capacity
    · exact hfull
    · unfold allocate at h
      rw [dif_neg hfull] at h
      cases h
  · intro hfull
    exact ⟨hfull, by unfold allocate; rw [dif_pos hfull]⟩

/-- A handle drawn from the before-arena still resolves to the same value after
an allocation: the append only adds a slot at the end. -/
theorem Allocation.old_resolves_unchanged {α : Type} {capacity : Nat} {before : Arena α capacity} {value : α}
    (allocation : Allocation before value) (handle : Handle before) :
    resolve allocation.after
        ⟨handle.val, allocation.extends ▸ Nat.lt_of_lt_of_le handle.isLt
          (by rw [length_append_singleton]; exact Nat.le_add_right _ 1)⟩ =
      resolve before handle := by
  unfold resolve
  rw [List.get_of_eq allocation.extends]
  rw [List.get_eq_getElem]
  rw [List.getElem_append_left handle.isLt]
  exact List.get_eq_getElem.symm

end Slang.Arena
