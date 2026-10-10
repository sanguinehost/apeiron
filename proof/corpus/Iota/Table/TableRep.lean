import Iota.Machine
import Apeiron.Table.Basic

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Table

open Iota Iota.Machine

def tableEmpty : Word := Word.ofNat (wordModulus - 1)

def coordCeiling : Nat := 8796093022208

/-- `2 ^ exponent`, by structural recursion on the exponent.

Temporal O(exponent): one `Nat` multiplication per recursive step, `exponent` steps. Spatial
O(exponent): the result `2 ^ exponent` occupies `exponent + 1` bits, and the step
`2 * powTwo exponent` consumes the recursive result, so the recursion is not tail-recursive and
retains one frame per step. Operations counted: `Nat` multiplications, recursion depth. -/
def powTwo : Nat → Nat
  | 0 => 1
  | exponent + 1 => 2 * powTwo exponent

theorem powTwo_pos (exponent : Nat) : 0 < powTwo exponent := by
  induction exponent with
  | zero => exact Nat.zero_lt_succ 0
  | succ exponent inductionHypothesis =>
      exact Nat.mul_pos (Nat.zero_lt_succ 1) inductionHypothesis

def PositivePowerOfTwo (cap : Word) : Prop :=
  ∃ exponent : Nat, cap.val = powTwo exponent

theorem PositivePowerOfTwo.pos {cap : Word} (hcap : PositivePowerOfTwo cap) : 0 < cap.val := by
  rcases hcap with ⟨exponent, hexponent⟩
  rw [hexponent]
  exact powTwo_pos exponent

def InExtent (base cap : Word) (coordinate : Nat) : Prop :=
  base.val ≤ coordinate ∧ coordinate < base.val + 2 * cap.val

def ValidBacking (memory : Memory) (base cap : Word) : Prop :=
  base.val + 2 * cap.val ≤ coordCeiling ∧
    base.val + 2 * cap.val ≤ wordSignBit ∧
    ∀ offset < 2 * cap.val, memory.owned (base.val + offset) = true

def ValidExtent (memory : Memory) (base cap : Word) : Prop :=
  ValidBacking memory base cap ∧
    ∀ offset < 2 * cap.val, (memory.initialized (base.val + offset)).isSome

def PreservesOutside (before after : Memory) (base cap : Word) : Prop :=
  SameOwnership before after ∧
    ∀ coordinate, ¬InExtent base cap coordinate →
      after.initialized coordinate = before.initialized coordinate

def keyCoordinate (base : Word) (slot : Nat) : Nat := base.val + 2 * slot

def valueCoordinate (base : Word) (slot : Nat) : Nat := base.val + 2 * slot + 1

def keyAt (memory : Memory) (base : Word) (slot : Nat) : Option Word :=
  memory.initialized (keyCoordinate base slot)

def valueAt (memory : Memory) (base : Word) (slot : Nat) : Option Word :=
  memory.initialized (valueCoordinate base slot)

theorem slotAddress_word (base slot : Word) :
    Word.add base (Word.mul (Word.ofNat 2) slot) =
      Word.ofNat (keyCoordinate base slot.val) := by
  apply Fin.ext
  simp [Word.add, Word.mul, Word.ofNat, keyCoordinate, wordModulus, wordBits]

theorem slotValueAddress_word (base slot : Word) :
    Word.add (Word.add base (Word.mul (Word.ofNat 2) slot)) (Word.ofNat 1) =
      Word.ofNat (valueCoordinate base slot.val) := by
  rw [slotAddress_word base slot]
  apply Fin.ext
  simp [Word.add, Word.ofNat, valueCoordinate, keyCoordinate, wordModulus, wordBits]

theorem coordinate_ofNat_below_signBit (coordinate : Nat) (hcoordinate : coordinate < wordSignBit) :
    Memory.coordinate (Word.ofNat coordinate) = some coordinate := by
  have hmodulus : coordinate < wordModulus := by
    exact Nat.lt_trans hcoordinate (by decide)
  simp [Memory.coordinate, Word.signed, Word.ofNat, Nat.mod_eq_of_lt hmodulus, hcoordinate]

theorem ofNat_val_of_lt_signBit (coordinate : Nat) (hcoordinate : coordinate < wordSignBit) :
    (Word.ofNat coordinate).val = coordinate := by
  have hmodulus : coordinate < wordModulus := Nat.lt_trans hcoordinate (by decide)
  simp [Word.ofNat, Nat.mod_eq_of_lt hmodulus]

theorem signed_ofNat_of_lt_signBit (coordinate : Nat) (hcoordinate : coordinate < wordSignBit) :
    Word.signed (Word.ofNat coordinate) = Int.ofNat coordinate := by
  unfold Word.signed
  rw [ofNat_val_of_lt_signBit coordinate hcoordinate]
  exact if_pos hcoordinate

theorem keyCoordinate_below_signBit (base cap : Word) (slot : Nat)
    (hbound : base.val + 2 * cap.val ≤ wordSignBit) (hslot : slot < cap.val) :
    keyCoordinate base slot < wordSignBit := by
  unfold keyCoordinate
  have slots : 2 * slot < 2 * cap.val :=
    (Nat.mul_lt_mul_left (by decide : 0 < 2)).2 hslot
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left slots base.val) hbound

theorem valueCoordinate_below_signBit (base cap : Word) (slot : Nat)
    (hbound : base.val + 2 * cap.val ≤ wordSignBit) (hslot : slot < cap.val) :
    valueCoordinate base slot < wordSignBit := by
  unfold valueCoordinate
  have nextSlot : slot + 1 ≤ cap.val := hslot
  have doubled : 2 * (slot + 1) ≤ 2 * cap.val := Nat.mul_le_mul_left 2 nextSlot
  rw [Nat.mul_add, Nat.mul_one] at doubled
  have beforeDoubled : 2 * slot + 1 < 2 * slot + 2 := Nat.lt_succ_self (2 * slot + 1)
  have withinSlots : 2 * slot + 1 < 2 * cap.val :=
    Nat.lt_of_lt_of_le beforeDoubled doubled
  exact Nat.lt_of_lt_of_le (Nat.add_lt_add_left withinSlots base.val) hbound

def probeSlot (cap key : Word) (step : Nat) : Nat :=
  ((Word.tableSpread key).val + step) % cap.val

/-- List membership by `Word` equality, stopping at the first match.

Temporal O(n) for n = `occupied.length`: one `Word.decidableEq` comparison per element until a
match, then the scan stops. Spatial O(1) stack: the `isFalse` branch returns `member tail key`
directly, so the recursion is tail-recursive. Operations counted: `Word` equality comparisons. -/
def member : List Word → Word → Bool
  | [], _ => false
  | head :: tail, key =>
      match Word.decidableEq head key with
      | isTrue _ => true
      | isFalse _ => member tail key

/-- Add `key` to `occupied` when absent, otherwise return `occupied` unchanged.

Temporal O(n) for n = `occupied.length`: one `member` scan, stopping at the first match. Spatial
O(1) extra: the result aliases `occupied` on a match and otherwise allocates one cons cell.
Operations counted: `Word` equality comparisons, cons cells allocated. -/
def insertOccupied (occupied : List Word) (key : Word) : List Word :=
  if member occupied key then occupied else key :: occupied

def DuplicateFree : List Word → Prop
  | [] => True
  | head :: tail => member tail head = false ∧ DuplicateFree tail

def MemberEquiv (left right : List Word) : Prop :=
  ∀ key, member left key = member right key

theorem member_cons_self (key : Word) (tail : List Word) : member (key :: tail) key = true := by
  unfold member
  cases equal : Word.decidableEq key key with
  | isTrue => rfl
  | isFalse unequal => exact False.elim (unequal rfl)

theorem member_insertOccupied_self (occupied : List Word) (key : Word) :
    member (insertOccupied occupied key) key = true := by
  unfold insertOccupied
  cases present : member occupied key with
  | false =>
      rw [if_neg (fun equal => Bool.noConfusion equal)]
      exact member_cons_self key occupied
  | true =>
      rw [if_pos rfl]
      exact present

theorem duplicateFree_insertOccupied (occupied : List Word) (key : Word)
    (duplicateFree : DuplicateFree occupied) : DuplicateFree (insertOccupied occupied key) := by
  unfold insertOccupied
  cases present : member occupied key with
  | false =>
      rw [if_neg (fun equal => Bool.noConfusion equal)]
      exact ⟨present, duplicateFree⟩
  | true =>
      rw [if_pos rfl]
      exact duplicateFree

theorem duplicate_occupancy_rejected (key : Word) : ¬DuplicateFree [key, key] := by
  intro duplicateFree
  have absent : member [key] key = false := duplicateFree.1
  rw [member_cons_self] at absent
  contradiction

def PhysicalKeysListed (memory : Memory) (base cap : Word) (occupied : List Word) : Prop :=
  ∀ slot < cap.val, keyAt memory base slot = some tableEmpty ∨
    ∃ key, member occupied key = true ∧ keyAt memory base slot = some key

def PhysicalOccurrenceExact (memory : Memory) (base cap : Word)
    (occupied : List Word) : Prop :=
  (∀ i < cap.val, ∀ j < cap.val, ∀ key, key ≠ tableEmpty →
      keyAt memory base i = some key → keyAt memory base j = some key → i = j) ∧
    ∀ key, member occupied key = true ↔
      ∃ slot, slot < cap.val ∧ keyAt memory base slot = some key ∧ key ≠ tableEmpty

theorem unlisted_physical_key_rejected (memory : Memory) (base cap : Word)
    (occupied : List Word) (slot : Nat) (key : Word)
    (listed : PhysicalKeysListed memory base cap occupied) (slotBound : slot < cap.val)
    (present : keyAt memory base slot = some key) (notEmpty : key ≠ tableEmpty)
    (absent : member occupied key = false) : False := by
  rcases listed slot slotBound with empty | ⟨listedKey, isMember, physical⟩
  · have equal : key = tableEmpty := Option.some.inj (Eq.trans present.symm empty)
    exact notEmpty equal
  · have equal : key = listedKey := Option.some.inj (Eq.trans present.symm physical)
    cases equal
    rw [isMember] at absent
    contradiction

def RepSong (memory : Machine.Memory) (base cap default : Word)
    (occupied : List Word) (h : Apeiron.Table.HMap Word Word) : Prop :=
  PositivePowerOfTwo cap ∧
  ValidExtent memory base cap ∧
  (DuplicateFree occupied ∧ ∀ key, member occupied key = true → key ≠ tableEmpty) ∧
  PhysicalKeysListed memory base cap occupied ∧
  PhysicalOccurrenceExact memory base cap occupied ∧
  (∀ key, member occupied key = true → ∃ step, step < cap.val ∧
      keyAt memory base (probeSlot cap key step) = some key ∧
      ∀ earlier, earlier < step →
        keyAt memory base (probeSlot cap key earlier) ≠ some tableEmpty) ∧
  ((∀ slot, slot < cap.val → ∀ key, member occupied key = true →
      keyAt memory base slot = some key → valueAt memory base slot = some (h key)) ∧
    ∀ key, member occupied key = false → h key = default)

theorem RepSong.positivePowerOfTwo {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Apeiron.Table.HMap Word Word}
    (rep : RepSong memory base cap default occupied h) : PositivePowerOfTwo cap := rep.1

theorem RepSong.validExtent {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Apeiron.Table.HMap Word Word}
    (rep : RepSong memory base cap default occupied h) : ValidExtent memory base cap := rep.2.1

theorem RepSong.owned_extent {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Apeiron.Table.HMap Word Word}
    (rep : RepSong memory base cap default occupied h) :
    ∀ offset < 2 * cap.val, memory.owned (base.val + offset) = true :=
  rep.validExtent.1.2.2

theorem RepSong.outside_default {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Apeiron.Table.HMap Word Word}
    (rep : RepSong memory base cap default occupied h) :
    ∀ key, member occupied key = false → h key = default := rep.2.2.2.2.2.2.2

theorem repSong_empty_of_layout (memory : Memory) (base cap default : Word)
    (hcap : PositivePowerOfTwo cap) (hextent : ValidExtent memory base cap)
    (hkeys : ∀ slot < cap.val, keyAt memory base slot = some tableEmpty) :
    RepSong memory base cap default [] (fun _ => default) := by
  refine ⟨hcap, hextent, ⟨True.intro, ?_⟩, ?_, ?_, ?_, ?_⟩
  · intro key memberKey
    contradiction
  · intro slot slotBound
    exact Or.inl (hkeys slot slotBound)
  · constructor
    · intro i iBound j jBound key notEmpty keyI keyJ
      have equal : key = tableEmpty := Option.some.inj (Eq.trans keyI.symm (hkeys i iBound))
      exact False.elim (notEmpty equal)
    · intro key
      constructor
      · intro memberKey
        contradiction
      · intro occurrence
        rcases occurrence with ⟨slot, slotBound, physical, notEmpty⟩
        have equal : key = tableEmpty :=
          Option.some.inj (Eq.trans physical.symm (hkeys slot slotBound))
        exact False.elim (notEmpty equal)
  · intro key memberKey
    contradiction
  · constructor
    · intro slot slotBound key memberKey
      contradiction
    · intro key absent
      rfl

end Iota.Table
