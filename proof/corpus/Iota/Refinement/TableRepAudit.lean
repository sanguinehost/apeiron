import Iota.Table.TableRep
import Iota.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Refinement.TableRepAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Iota.Refinement.TableRepAudit

#audit_empty Iota.Refinement.TableRepAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Table.tableEmpty : Iota.Word
#audit_empty Iota.Table.coordCeiling : Nat
#audit_empty Iota.Table.powTwo : Nat → Nat
#audit_empty Iota.Table.powTwo_pos : ∀ (exponent : Nat), 0 < Iota.Table.powTwo exponent
#audit_empty Iota.Table.PositivePowerOfTwo : Iota.Word → Prop
#audit_empty Iota.Table.PositivePowerOfTwo.pos : ∀ {cap : Iota.Word}, Iota.Table.PositivePowerOfTwo cap → 0 < cap.val
#audit_empty Iota.Table.InExtent : Iota.Word → Iota.Word → Nat → Prop
#audit_empty Iota.Table.ValidBacking : Iota.Machine.Memory → Iota.Word → Iota.Word → Prop
#audit_empty Iota.Table.ValidExtent : Iota.Machine.Memory → Iota.Word → Iota.Word → Prop
#audit_empty Iota.Table.PreservesOutside : Iota.Machine.Memory → Iota.Machine.Memory → Iota.Word → Iota.Word → Prop
#audit_empty Iota.Table.keyCoordinate : Iota.Word → Nat → Nat
#audit_empty Iota.Table.valueCoordinate : Iota.Word → Nat → Nat
#audit_empty Iota.Table.keyAt : Iota.Machine.Memory → Iota.Word → Nat → Option Iota.Word
#audit_empty Iota.Table.valueAt : Iota.Machine.Memory → Iota.Word → Nat → Option Iota.Word
#audit_empty Iota.Table.probeSlot : Iota.Word → Iota.Word → Nat → Nat
#audit_empty Iota.Table.member : List Iota.Word → Iota.Word → Bool
#audit_empty Iota.Table.insertOccupied : List Iota.Word → Iota.Word → List Iota.Word
#audit_empty Iota.Table.DuplicateFree : List Iota.Word → Prop
#audit_empty Iota.Table.MemberEquiv : List Iota.Word → List Iota.Word → Prop
#audit_empty Iota.Table.member_cons_self : ∀ (key : Iota.Word) (tail : List Iota.Word), Iota.Table.member (key :: tail) key = true
#audit_empty Iota.Table.member_insertOccupied_self : ∀ (occupied : List Iota.Word) (key : Iota.Word), Iota.Table.member (Iota.Table.insertOccupied occupied key) key = true
#audit_empty Iota.Table.duplicateFree_insertOccupied : ∀ (occupied : List Iota.Word) (key : Iota.Word), Iota.Table.DuplicateFree occupied → Iota.Table.DuplicateFree (Iota.Table.insertOccupied occupied key)
#audit_empty Iota.Table.duplicate_occupancy_rejected : ∀ (key : Iota.Word), ¬Iota.Table.DuplicateFree [key, key]
#audit_empty Iota.Table.PhysicalKeysListed : Iota.Machine.Memory → Iota.Word → Iota.Word → List Iota.Word → Prop
#audit_empty Iota.Table.PhysicalOccurrenceExact : Iota.Machine.Memory → Iota.Word → Iota.Word → List Iota.Word → Prop
#audit_empty Iota.Table.unlisted_physical_key_rejected : ∀ (memory : Iota.Machine.Memory) (base cap : Iota.Word) (occupied : List Iota.Word) (slot : Nat) (key : Iota.Word), Iota.Table.PhysicalKeysListed memory base cap occupied → slot < cap.val → Iota.Table.keyAt memory base slot = some key → key ≠ Iota.Table.tableEmpty → Iota.Table.member occupied key = false → False
#audit_empty Iota.Table.RepSong : Iota.Machine.Memory → Iota.Word → Iota.Word → Iota.Word → List Iota.Word → Apeiron.Table.HMap Iota.Word Iota.Word → Prop
#audit_empty Iota.Table.RepSong.positivePowerOfTwo : ∀ {memory : Iota.Machine.Memory} {base cap default : Iota.Word} {occupied : List Iota.Word} {h : Apeiron.Table.HMap Iota.Word Iota.Word}, Iota.Table.RepSong memory base cap default occupied h → Iota.Table.PositivePowerOfTwo cap
#audit_empty Iota.Table.RepSong.validExtent : ∀ {memory : Iota.Machine.Memory} {base cap default : Iota.Word} {occupied : List Iota.Word} {h : Apeiron.Table.HMap Iota.Word Iota.Word}, Iota.Table.RepSong memory base cap default occupied h → Iota.Table.ValidExtent memory base cap
#audit_empty Iota.Table.RepSong.owned_extent : ∀ {memory : Iota.Machine.Memory} {base cap default : Iota.Word} {occupied : List Iota.Word} {h : Apeiron.Table.HMap Iota.Word Iota.Word}, Iota.Table.RepSong memory base cap default occupied h → ∀ offset < 2 * cap.val, memory.owned (base.val + offset) = true
#audit_empty Iota.Table.RepSong.outside_default : ∀ {memory : Iota.Machine.Memory} {base cap default : Iota.Word} {occupied : List Iota.Word} {h : Apeiron.Table.HMap Iota.Word Iota.Word}, Iota.Table.RepSong memory base cap default occupied h → ∀ key, Iota.Table.member occupied key = false → h key = default
#audit_empty Iota.Table.repSong_empty_of_layout : ∀ (memory : Iota.Machine.Memory) (base cap default : Iota.Word), Iota.Table.PositivePowerOfTwo cap → Iota.Table.ValidExtent memory base cap → (∀ slot < cap.val, Iota.Table.keyAt memory base slot = some Iota.Table.tableEmpty) → Iota.Table.RepSong memory base cap default [] (fun _ => default)
