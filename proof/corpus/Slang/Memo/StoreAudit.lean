import Slang.Audit.Empty
import Slang.Memo.Store

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false

namespace Slang.Memo.StoreAudit

/-- A zero-axiom audit fixture. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

end Slang.Memo.StoreAudit

open Slang.Memo

#audit_empty Slang.Memo.StoreAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Slang.Memo.MemoEntry : type_of% @Slang.Memo.MemoEntry
#audit_empty Slang.Memo.MemoStore : type_of% @Slang.Memo.MemoStore
#audit_empty Slang.Memo.empty : type_of% @Slang.Memo.empty
#audit_empty Slang.Memo.findKeySlot : type_of% @Slang.Memo.findKeySlot
#audit_empty Slang.Memo.findVacantSlot : type_of% @Slang.Memo.findVacantSlot
#audit_empty Slang.Memo.writeSlot : type_of% @Slang.Memo.writeSlot
#audit_empty Slang.Memo.readEntry? : type_of% @Slang.Memo.readEntry?
#audit_empty Slang.Memo.readValue : type_of% @Slang.Memo.readValue
#audit_empty Slang.Memo.denote : type_of% @Slang.Memo.denote
#audit_empty Slang.Memo.WellFormed : type_of% @Slang.Memo.WellFormed
#audit_empty Slang.Memo.RepMemo : type_of% @Slang.Memo.RepMemo
#audit_empty Slang.Memo.InsertResult : type_of% @Slang.Memo.InsertResult
#audit_empty Slang.Memo.insert : type_of% @Slang.Memo.insert
#audit_empty Slang.Memo.HomeFull : type_of% @Slang.Memo.HomeFull
#audit_empty Slang.Memo.update_same : type_of% @Slang.Memo.update_same
#audit_empty Slang.Memo.update_other : type_of% @Slang.Memo.update_other
#audit_empty Slang.Memo.writeSlot_same : type_of% @Slang.Memo.writeSlot_same
#audit_empty Slang.Memo.writeSlot_other_slot : type_of% @Slang.Memo.writeSlot_other_slot
#audit_empty Slang.Memo.writeSlot_other_shard : type_of% @Slang.Memo.writeSlot_other_shard
#audit_empty Slang.Memo.findKeySlot_some : type_of% @Slang.Memo.findKeySlot_some
#audit_empty Slang.Memo.findKeySlot_none_iff : type_of% @Slang.Memo.findKeySlot_none_iff
#audit_empty Slang.Memo.findVacantSlot_some : type_of% @Slang.Memo.findVacantSlot_some
#audit_empty Slang.Memo.findVacantSlot_none_iff : type_of% @Slang.Memo.findVacantSlot_none_iff
#audit_empty Slang.Memo.readEntry_key : type_of% @Slang.Memo.readEntry_key
#audit_empty Slang.Memo.read_dial_consistent : type_of% @Slang.Memo.read_dial_consistent
#audit_empty Slang.Memo.read_denotes_get : type_of% @Slang.Memo.read_denotes_get
#audit_empty Slang.Memo.empty_rep : type_of% @Slang.Memo.empty_rep

#print axioms Slang.Memo.findKeySlot_some
#print axioms Slang.Memo.findKeySlot_none_iff
#print axioms Slang.Memo.findVacantSlot_some
#print axioms Slang.Memo.findVacantSlot_none_iff
#print axioms Slang.Memo.readEntry_key
#print axioms Slang.Memo.read_dial_consistent
#print axioms Slang.Memo.read_denotes_get
#print axioms Slang.Memo.empty_rep

set_option warningAsError false
