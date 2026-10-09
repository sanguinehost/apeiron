import Apeiron.Capability.Naming
import Apeiron.Table.Basic

/-!
# A finite table under the naming fold

@name Masked Table
@grade proved

A region table has finitely many addressable slots. `Table.HMap (Fin (2^W)) V` holds those slots;
`maskedKey` is the boundary that accepts an arbitrary raw key and folds it into that finite domain.
Every finite slot is reached by its own raw coordinate. A write and a read through equal folds meet
at the same slot; distinct folds leave the read untouched.

The authorization fold is separate. `guardedRead` reads the table only at a positive extent; a zero
extent returns the supplied sink value. This is the value-level counterpart of `Naming.foldMap`:
the finite key names an addressable table slot, while an ungranted raw key never receives its value.
-/

namespace Apeiron
namespace Capability

/-- Fold an arbitrary raw key into the `2^W` addressable table slots.

    (W key : Nat)       [the table width and a raw key]
    Fin (2 ^ W)         [one slot in that finite table] -/
def maskedKey (W key : Nat) : Fin (2 ^ W) :=
  ⟨nameMap W key, nameMap_lt W key⟩

/-- The finite key carries exactly the naming-fold coordinate.

    (W key : Nat)                               [the table width and a raw key]
    (maskedKey W key).val = nameMap W key       [its finite value is the folded coordinate] -/
theorem maskedKey_val (W key : Nat) : (maskedKey W key).val = nameMap W key := rfl

/-- Every finite table slot is reached by its own raw coordinate.

    (W : Nat) (slot : Fin (2 ^ W))              [the table width and one finite slot]
    maskedKey W slot.val = slot                  [folding that coordinate returns the slot] -/
theorem maskedKey_complete (W : Nat) (slot : Fin (2 ^ W)) : maskedKey W slot.val = slot := by
  apply Fin.ext
  exact nameMap_surjective W slot.val slot.isLt

/-- Read a finite region table through an arbitrary raw key.

    {V : Type} (W : Nat)                         [the cell type and table width]
    Table.HMap (Fin (2 ^ W)) V → Nat → V         [the table and raw key, to its cell value] -/
def maskedRead {V : Type} (W : Nat) (table : Table.HMap (Fin (2 ^ W)) V) (key : Nat) : V :=
  Table.get table (maskedKey W key)

/-- Write a finite region table through an arbitrary raw key.

    {V : Type} (W : Nat)                         [the cell type and table width]
    (V → V → V)                                  [the table merge]
    Nat → V                                      [the raw key and new value]
    Table.HMap (Fin (2 ^ W)) V                   [the old table]
    Table.HMap (Fin (2 ^ W)) V                   [the table after the folded write] -/
def maskedInsertWith {V : Type} (W : Nat) (merge : V → V → V) (key : Nat) (value : V)
    (table : Table.HMap (Fin (2 ^ W)) V) : Table.HMap (Fin (2 ^ W)) V :=
  Table.insertWith merge (maskedKey W key) value table

/-- Reading through the raw key just written reaches that write's folded slot.

    {V : Type} (W : Nat) (merge : V → V → V)     [the cells, width, and merge]
    (key : Nat) (value : V) (table : Table.HMap (Fin (2 ^ W)) V) [the write and old table]
    maskedRead W (maskedInsertWith W merge key value table) key [read after that write]
      = merge value (maskedRead W table key)     [the merge at its folded slot] -/
theorem maskedRead_after_write {V : Type} (W : Nat) (merge : V → V → V) (key : Nat)
    (value : V) (table : Table.HMap (Fin (2 ^ W)) V) :
    maskedRead W (maskedInsertWith W merge key value table) key
      = merge value (maskedRead W table key) := by
  simpa [maskedRead, maskedInsertWith, Table.get] using
    Table.get_insertWith_self merge (maskedKey W key) value table

/-- A write cannot affect a raw read whose folded coordinates differ.

    {V : Type} (W : Nat) (merge : V → V → V)     [the cells, width, and merge]
    (readKey writeKey : Nat) (value : V)          [the read, write, and new value]
    (table : Table.HMap (Fin (2 ^ W)) V)          [the old table]
    (nameMap W readKey ≠ nameMap W writeKey)      [the two raw keys fold to different slots]
    maskedRead W (maskedInsertWith W merge writeKey value table) readKey [read after the write]
      = maskedRead W table readKey                [the unrelated cell is unchanged] -/
theorem maskedRead_after_other_write {V : Type} (W : Nat) (merge : V → V → V)
    (readKey writeKey : Nat) (value : V) (table : Table.HMap (Fin (2 ^ W)) V)
    (h : nameMap W readKey ≠ nameMap W writeKey) :
    maskedRead W (maskedInsertWith W merge writeKey value table) readKey
      = maskedRead W table readKey := by
  simp only [maskedRead, maskedInsertWith]
  apply Table.get_insertWith_ne
  intro heq
  apply h
  exact congrArg Fin.val heq

/-- Raw aliases name one table slot, so a write through either is read through either.

    {V : Type} (W : Nat) (merge : V → V → V)     [the cells, width, and merge]
    (readKey writeKey : Nat) (value : V)          [the two raw keys and new value]
    (table : Table.HMap (Fin (2 ^ W)) V)          [the old table]
    (nameMap W readKey = nameMap W writeKey)      [the raw keys fold to one slot]
    maskedRead W (maskedInsertWith W merge writeKey value table) readKey [read through the alias]
      = merge value (maskedRead W table readKey)  [the same folded merge] -/
theorem maskedRead_after_alias_write {V : Type} (W : Nat) (merge : V → V → V)
    (readKey writeKey : Nat) (value : V) (table : Table.HMap (Fin (2 ^ W)) V)
    (h : nameMap W readKey = nameMap W writeKey) :
    maskedRead W (maskedInsertWith W merge writeKey value table) readKey
      = merge value (maskedRead W table readKey) := by
  simp only [maskedRead, maskedInsertWith]
  rw [show maskedKey W readKey = maskedKey W writeKey from Fin.ext h]
  exact Table.get_insertWith_self merge (maskedKey W writeKey) value table

/-- Read a masked table only when the named coordinate has a positive extent.

    {V : Type} (W : Nat) (sink : V)               [the cells, width, and ungranted result]
    (extent : Nat → Nat)                           [the stamped extent at each folded coordinate]
    (table : Table.HMap (Fin (2 ^ W)) V)           [the finite region table]
    (key : Nat)                                    [a raw key]
    V                                              [the granted cell value, or the sink] -/
def guardedRead {V : Type} (W : Nat) (sink : V) (extent : Nat → Nat)
    (table : Table.HMap (Fin (2 ^ W)) V) (key : Nat) : V :=
  if extent (nameMap W key) = 0 then sink else maskedRead W table key

/-- An ungranted raw key returns the sink, never its finite table slot's value.

    {V : Type} (W : Nat) (sink : V)               [the cells, width, and ungranted result]
    (extent : Nat → Nat)                           [the stamped extents]
    (table : Table.HMap (Fin (2 ^ W)) V)           [the finite table]
    (key : Nat)                                    [a raw key]
    (extent (nameMap W key) = 0)                   [the folded coordinate is ungranted]
    guardedRead W sink extent table key = sink     [it returns only the sink] -/
theorem guardedRead_ungranted {V : Type} (W : Nat) (sink : V) (extent : Nat → Nat)
    (table : Table.HMap (Fin (2 ^ W)) V) (key : Nat) (h : extent (nameMap W key) = 0) :
    guardedRead W sink extent table key = sink := by
  simp [guardedRead, h]

/-- A granted raw key reads exactly the cell at its masked table slot.

    {V : Type} (W : Nat) (sink : V)               [the cells, width, and ungranted result]
    (extent : Nat → Nat)                           [the stamped extents]
    (table : Table.HMap (Fin (2 ^ W)) V)           [the finite table]
    (key : Nat)                                    [a raw key]
    (0 < extent (nameMap W key))                   [the folded coordinate is granted]
    guardedRead W sink extent table key            [the guarded read]
      = maskedRead W table key                     [is its ordinary masked-table read] -/
theorem guardedRead_granted {V : Type} (W : Nat) (sink : V) (extent : Nat → Nat)
    (table : Table.HMap (Fin (2 ^ W)) V) (key : Nat) (h : 0 < extent (nameMap W key)) :
    guardedRead W sink extent table key = maskedRead W table key := by
  simp [guardedRead, Nat.ne_of_gt h]

end Capability
end Apeiron
