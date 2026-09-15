import Slang.Table.TableRep
import Slang.Audit.Empty

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Refinement.TableRepAudit

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Refinement.TableRepAudit

#audit_empty Slang.Refinement.TableRepAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Table.tableEmpty : Slang.Word
#audit_empty Slang.Table.coordCeiling : Nat
#audit_empty Slang.Table.powTwo : Nat → Nat
#audit_empty Slang.Table.powTwo_pos : ∀ (exponent : Nat), 0 < Slang.Table.powTwo exponent
#audit_empty Slang.Table.PositivePowerOfTwo : Slang.Word → Prop
#audit_empty Slang.Table.PositivePowerOfTwo.pos : ∀ {cap : Slang.Word}, Slang.Table.PositivePowerOfTwo cap → 0 < cap.val
#audit_empty Slang.Table.InExtent : Slang.Word → Slang.Word → Nat → Prop
#audit_empty Slang.Table.ValidBacking : Slang.Machine.Memory → Slang.Word → Slang.Word → Prop
#audit_empty Slang.Table.ValidExtent : Slang.Machine.Memory → Slang.Word → Slang.Word → Prop
#audit_empty Slang.Table.PreservesOutside : Slang.Machine.Memory → Slang.Machine.Memory → Slang.Word → Slang.Word → Prop
#audit_empty Slang.Table.keyCoordinate : Slang.Word → Nat → Nat
#audit_empty Slang.Table.valueCoordinate : Slang.Word → Nat → Nat
#audit_empty Slang.Table.keyAt : Slang.Machine.Memory → Slang.Word → Nat → Option Slang.Word
#audit_empty Slang.Table.valueAt : Slang.Machine.Memory → Slang.Word → Nat → Option Slang.Word
#audit_empty Slang.Table.probeSlot : Slang.Word → Slang.Word → Nat → Nat
#audit_empty Slang.Table.member : List Slang.Word → Slang.Word → Bool
#audit_empty Slang.Table.insertOccupied : List Slang.Word → Slang.Word → List Slang.Word
#audit_empty Slang.Table.DuplicateFree : List Slang.Word → Prop
#audit_empty Slang.Table.MemberEquiv : List Slang.Word → List Slang.Word → Prop
#audit_empty Slang.Table.member_cons_self : ∀ (key : Slang.Word) (tail : List Slang.Word), Slang.Table.member (key :: tail) key = true
#audit_empty Slang.Table.member_insertOccupied_self : ∀ (occupied : List Slang.Word) (key : Slang.Word), Slang.Table.member (Slang.Table.insertOccupied occupied key) key = true
#audit_empty Slang.Table.duplicateFree_insertOccupied : ∀ (occupied : List Slang.Word) (key : Slang.Word), Slang.Table.DuplicateFree occupied → Slang.Table.DuplicateFree (Slang.Table.insertOccupied occupied key)
#audit_empty Slang.Table.duplicate_occupancy_rejected : ∀ (key : Slang.Word), ¬Slang.Table.DuplicateFree [key, key]
#audit_empty Slang.Table.PhysicalKeysListed : Slang.Machine.Memory → Slang.Word → Slang.Word → List Slang.Word → Prop
#audit_empty Slang.Table.PhysicalOccurrenceExact : Slang.Machine.Memory → Slang.Word → Slang.Word → List Slang.Word → Prop
#audit_empty Slang.Table.unlisted_physical_key_rejected : ∀ (memory : Slang.Machine.Memory) (base cap : Slang.Word) (occupied : List Slang.Word) (slot : Nat) (key : Slang.Word), Slang.Table.PhysicalKeysListed memory base cap occupied → slot < cap.val → Slang.Table.keyAt memory base slot = some key → key ≠ Slang.Table.tableEmpty → Slang.Table.member occupied key = false → False
#audit_empty Slang.Table.RepSong : Slang.Machine.Memory → Slang.Word → Slang.Word → Slang.Word → List Slang.Word → Apeiron.Table.HMap Slang.Word Slang.Word → Prop
#audit_empty Slang.Table.RepSong.positivePowerOfTwo : ∀ {memory : Slang.Machine.Memory} {base cap default : Slang.Word} {occupied : List Slang.Word} {h : Apeiron.Table.HMap Slang.Word Slang.Word}, Slang.Table.RepSong memory base cap default occupied h → Slang.Table.PositivePowerOfTwo cap
#audit_empty Slang.Table.RepSong.validExtent : ∀ {memory : Slang.Machine.Memory} {base cap default : Slang.Word} {occupied : List Slang.Word} {h : Apeiron.Table.HMap Slang.Word Slang.Word}, Slang.Table.RepSong memory base cap default occupied h → Slang.Table.ValidExtent memory base cap
#audit_empty Slang.Table.RepSong.owned_extent : ∀ {memory : Slang.Machine.Memory} {base cap default : Slang.Word} {occupied : List Slang.Word} {h : Apeiron.Table.HMap Slang.Word Slang.Word}, Slang.Table.RepSong memory base cap default occupied h → ∀ offset < 2 * cap.val, memory.owned (base.val + offset) = true
#audit_empty Slang.Table.RepSong.outside_default : ∀ {memory : Slang.Machine.Memory} {base cap default : Slang.Word} {occupied : List Slang.Word} {h : Apeiron.Table.HMap Slang.Word Slang.Word}, Slang.Table.RepSong memory base cap default occupied h → ∀ key, Slang.Table.member occupied key = false → h key = default
#audit_empty Slang.Table.repSong_empty_of_layout : ∀ (memory : Slang.Machine.Memory) (base cap default : Slang.Word), Slang.Table.PositivePowerOfTwo cap → Slang.Table.ValidExtent memory base cap → (∀ slot < cap.val, Slang.Table.keyAt memory base slot = some Slang.Table.tableEmpty) → Slang.Table.RepSong memory base cap default [] (fun _ => default)
