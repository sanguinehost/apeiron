import Slang.Audit.Empty
import Slang.Memo.Store

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo.StoreAudit

/-- A zero-axiom audit fixture. -/
theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl

private theorem fin_one_zero (index : Fin 1) : index = ⟨0, by decide⟩ := by
  apply Fin.ext
  cases index with
  | mk value bound =>
      cases value with
      | zero => rfl
      | succ value => exact False.elim (Nat.not_lt_zero value (Nat.lt_of_succ_lt_succ bound))

/-- A one-slot store whose single slot contains an abstract, transition-checked entry. -/
def singletonStore {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) (entry : MemoEntry sources states results) :
    MemoStore sources states results 1 :=
  writeSlot (empty defaultValue) (Slang.Address.homeShard entry.key) ⟨0, by decide⟩ entry

/-- A different key routed to the occupied shard is refused by the one-slot fixture. -/
theorem control_full_shard_refuses_second {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) (first second : MemoEntry sources states results)
    (different : second.key ≠ first.key)
    (sameHome : Slang.Address.homeShard second.key = Slang.Address.homeShard first.key) :
    insert (singletonStore defaultValue first) second =
      (.shardFull : InsertResult (singletonStore defaultValue first)) := by
  exact (insert_shardFull_iff (singletonStore defaultValue first) second).mpr ⟨
    (findKeySlot_none_iff _ _).mpr (by
      intro index stored present
      have indexEqual := fin_one_zero index
      cases indexEqual
      rw [sameHome] at present
      change (writeSlot (empty defaultValue) (Slang.Address.homeShard first.key) ⟨0, by decide⟩ first).slots
        (Slang.Address.homeShard first.key) ⟨0, by decide⟩ = some stored at present
      rw [writeSlot_same] at present
      have firstEqual : first = stored := Option.some.inj present
      cases firstEqual
      exact fun keyEqual => different keyEqual.symm),
    (by
      intro index
      have indexEqual := fin_one_zero index
      cases indexEqual
      rw [sameHome]
      intro absent
      change (writeSlot (empty defaultValue) (Slang.Address.homeShard first.key) ⟨0, by decide⟩ first).slots
        (Slang.Address.homeShard first.key) ⟨0, by decide⟩ = none at absent
      rw [writeSlot_same] at absent
      cases absent)⟩

/-- Every insertion into a zero-capacity store is refused. -/
theorem control_zero_refuses {sources : Slang.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results 0} (entry : MemoEntry sources states results) :
    insert store entry = (.shardFull : InsertResult store) :=
  insert_zero entry

/-- Successful insertion does not replace the value read at a different key. -/
theorem control_unrelated_never_replaced {sources : Slang.Source.SourceArena}
    {stateCapacity resultCapacity N : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {results : Slang.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results N} {entry : MemoEntry sources states results}
    {after : MemoStore sources states results N} (wf : WellFormed store)
    (ins : insert store entry = .inserted after) {key : MemoKey sources states}
    (different : key ≠ entry.key) : readValue after key = readValue store key :=
  inserted_reads_other wf ins different

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
#audit_empty Slang.Memo.inserted_reads_self : type_of% @Slang.Memo.inserted_reads_self
#audit_empty Slang.Memo.inserted_reads_other : type_of% @Slang.Memo.inserted_reads_other
#audit_empty Slang.Memo.insert_preserves_wellFormed : type_of% @Slang.Memo.insert_preserves_wellFormed
#audit_empty Slang.Memo.insert_ok_rep : type_of% @Slang.Memo.insert_ok_rep
#audit_empty Slang.Memo.insert_shardFull_iff : type_of% @Slang.Memo.insert_shardFull_iff
#audit_empty Slang.Memo.insert_zero : type_of% @Slang.Memo.insert_zero
#audit_empty Slang.Memo.StoreAudit.singletonStore : type_of% @Slang.Memo.StoreAudit.singletonStore
#audit_empty Slang.Memo.StoreAudit.control_full_shard_refuses_second : type_of% @Slang.Memo.StoreAudit.control_full_shard_refuses_second
#audit_empty Slang.Memo.StoreAudit.control_zero_refuses : type_of% @Slang.Memo.StoreAudit.control_zero_refuses
#audit_empty Slang.Memo.StoreAudit.control_unrelated_never_replaced : type_of% @Slang.Memo.StoreAudit.control_unrelated_never_replaced

#print axioms Slang.Memo.findKeySlot_some
#print axioms Slang.Memo.findKeySlot_none_iff
#print axioms Slang.Memo.findVacantSlot_some
#print axioms Slang.Memo.findVacantSlot_none_iff
#print axioms Slang.Memo.readEntry_key
#print axioms Slang.Memo.read_dial_consistent
#print axioms Slang.Memo.read_denotes_get
#print axioms Slang.Memo.empty_rep
#print axioms Slang.Memo.inserted_reads_self
#print axioms Slang.Memo.inserted_reads_other
#print axioms Slang.Memo.insert_preserves_wellFormed
#print axioms Slang.Memo.insert_ok_rep
#print axioms Slang.Memo.insert_shardFull_iff
#print axioms Slang.Memo.insert_zero
#print axioms Slang.Memo.StoreAudit.singletonStore
#print axioms Slang.Memo.StoreAudit.control_full_shard_refuses_second
#print axioms Slang.Memo.StoreAudit.control_zero_refuses
#print axioms Slang.Memo.StoreAudit.control_unrelated_never_replaced

set_option warningAsError false
