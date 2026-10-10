import Iota.Audit.Empty
import Iota.Memo.Store

set_option autoImplicit false
set_option warningAsError true
set_option linter.missingDocs false
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Memo.StoreAudit

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
def singletonStore {sources : Iota.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity} {results : Iota.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) (entry : MemoEntry sources states results) :
    MemoStore sources states results 1 :=
  writeSlot (empty defaultValue) (Iota.Address.homeShard entry.key) ⟨0, by decide⟩ entry

/-- A different key routed to the occupied shard is refused by the one-slot fixture. -/
theorem control_full_shard_refuses_second {sources : Iota.Source.SourceArena}
    {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity} {results : Iota.Arena.ResultArena resultCapacity}
    (defaultValue : MemoValue sources states results) (first second : MemoEntry sources states results)
    (different : second.key ≠ first.key)
    (sameHome : Iota.Address.homeShard second.key = Iota.Address.homeShard first.key) :
    insert (singletonStore defaultValue first) second =
      (.shardFull : InsertResult (singletonStore defaultValue first)) := by
  exact (insert_shardFull_iff (singletonStore defaultValue first) second).mpr ⟨
    (findKeySlot_none_iff _ _).mpr (by
      intro index stored present
      have indexEqual := fin_one_zero index
      cases indexEqual
      rw [sameHome] at present
      change (writeSlot (empty defaultValue) (Iota.Address.homeShard first.key) ⟨0, by decide⟩ first).slots
        (Iota.Address.homeShard first.key) ⟨0, by decide⟩ = some stored at present
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
      change (writeSlot (empty defaultValue) (Iota.Address.homeShard first.key) ⟨0, by decide⟩ first).slots
        (Iota.Address.homeShard first.key) ⟨0, by decide⟩ = none at absent
      rw [writeSlot_same] at absent
      cases absent)⟩

/-- Every insertion into a zero-capacity store is refused. -/
theorem control_zero_refuses {sources : Iota.Source.SourceArena} {stateCapacity resultCapacity : Nat}
    {states : Iota.Arena.StateArena stateCapacity} {results : Iota.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results 0} (entry : MemoEntry sources states results) :
    insert store entry = (.shardFull : InsertResult store) :=
  insert_zero entry

/-- Successful insertion does not replace the value read at a different key. -/
theorem control_unrelated_never_replaced {sources : Iota.Source.SourceArena}
    {stateCapacity resultCapacity N : Nat}
    {states : Iota.Arena.StateArena stateCapacity} {results : Iota.Arena.ResultArena resultCapacity}
    {store : MemoStore sources states results N} {entry : MemoEntry sources states results}
    {after : MemoStore sources states results N} (wf : WellFormed store)
    (ins : insert store entry = .inserted after) {key : MemoKey sources states}
    (different : key ≠ entry.key) : readValue after key = readValue store key :=
  inserted_reads_other wf ins different

end Iota.Memo.StoreAudit

open Iota.Memo

#audit_empty Iota.Memo.StoreAudit.nat_add_zero : (0 : Nat) + 0 = 0
#audit_empty Iota.Memo.MemoEntry : type_of% @Iota.Memo.MemoEntry
#audit_empty Iota.Memo.MemoStore : type_of% @Iota.Memo.MemoStore
#audit_empty Iota.Memo.empty : type_of% @Iota.Memo.empty
#audit_empty Iota.Memo.findKeySlot : type_of% @Iota.Memo.findKeySlot
#audit_empty Iota.Memo.findVacantSlot : type_of% @Iota.Memo.findVacantSlot
#audit_empty Iota.Memo.writeSlot : type_of% @Iota.Memo.writeSlot
#audit_empty Iota.Memo.readEntry? : type_of% @Iota.Memo.readEntry?
#audit_empty Iota.Memo.readValue : type_of% @Iota.Memo.readValue
#audit_empty Iota.Memo.denote : type_of% @Iota.Memo.denote
#audit_empty Iota.Memo.WellFormed : type_of% @Iota.Memo.WellFormed
#audit_empty Iota.Memo.RepMemo : type_of% @Iota.Memo.RepMemo
#audit_empty Iota.Memo.InsertResult : type_of% @Iota.Memo.InsertResult
#audit_empty Iota.Memo.insert : type_of% @Iota.Memo.insert
#audit_empty Iota.Memo.HomeFull : type_of% @Iota.Memo.HomeFull
#audit_empty Iota.Memo.update_same : type_of% @Iota.Memo.update_same
#audit_empty Iota.Memo.update_other : type_of% @Iota.Memo.update_other
#audit_empty Iota.Memo.writeSlot_same : type_of% @Iota.Memo.writeSlot_same
#audit_empty Iota.Memo.writeSlot_other_slot : type_of% @Iota.Memo.writeSlot_other_slot
#audit_empty Iota.Memo.writeSlot_other_shard : type_of% @Iota.Memo.writeSlot_other_shard
#audit_empty Iota.Memo.findKeySlot_some : type_of% @Iota.Memo.findKeySlot_some
#audit_empty Iota.Memo.findKeySlot_none_iff : type_of% @Iota.Memo.findKeySlot_none_iff
#audit_empty Iota.Memo.findVacantSlot_some : type_of% @Iota.Memo.findVacantSlot_some
#audit_empty Iota.Memo.findVacantSlot_none_iff : type_of% @Iota.Memo.findVacantSlot_none_iff
#audit_empty Iota.Memo.readEntry_key : type_of% @Iota.Memo.readEntry_key
#audit_empty Iota.Memo.read_dial_consistent : type_of% @Iota.Memo.read_dial_consistent
#audit_empty Iota.Memo.read_denotes_get : type_of% @Iota.Memo.read_denotes_get
#audit_empty Iota.Memo.empty_rep : type_of% @Iota.Memo.empty_rep
#audit_empty Iota.Memo.inserted_reads_self : type_of% @Iota.Memo.inserted_reads_self
#audit_empty Iota.Memo.inserted_reads_other : type_of% @Iota.Memo.inserted_reads_other
#audit_empty Iota.Memo.insert_preserves_wellFormed : type_of% @Iota.Memo.insert_preserves_wellFormed
#audit_empty Iota.Memo.insert_ok_rep : type_of% @Iota.Memo.insert_ok_rep
#audit_empty Iota.Memo.insert_shardFull_iff : type_of% @Iota.Memo.insert_shardFull_iff
#audit_empty Iota.Memo.insert_zero : type_of% @Iota.Memo.insert_zero
#audit_empty Iota.Memo.StoreAudit.singletonStore : type_of% @Iota.Memo.StoreAudit.singletonStore
#audit_empty Iota.Memo.StoreAudit.control_full_shard_refuses_second : type_of% @Iota.Memo.StoreAudit.control_full_shard_refuses_second
#audit_empty Iota.Memo.StoreAudit.control_zero_refuses : type_of% @Iota.Memo.StoreAudit.control_zero_refuses
#audit_empty Iota.Memo.StoreAudit.control_unrelated_never_replaced : type_of% @Iota.Memo.StoreAudit.control_unrelated_never_replaced

#print axioms Iota.Memo.findKeySlot_some
#print axioms Iota.Memo.findKeySlot_none_iff
#print axioms Iota.Memo.findVacantSlot_some
#print axioms Iota.Memo.findVacantSlot_none_iff
#print axioms Iota.Memo.readEntry_key
#print axioms Iota.Memo.read_dial_consistent
#print axioms Iota.Memo.read_denotes_get
#print axioms Iota.Memo.empty_rep
#print axioms Iota.Memo.inserted_reads_self
#print axioms Iota.Memo.inserted_reads_other
#print axioms Iota.Memo.insert_preserves_wellFormed
#print axioms Iota.Memo.insert_ok_rep
#print axioms Iota.Memo.insert_shardFull_iff
#print axioms Iota.Memo.insert_zero
#print axioms Iota.Memo.StoreAudit.singletonStore
#print axioms Iota.Memo.StoreAudit.control_full_shard_refuses_second
#print axioms Iota.Memo.StoreAudit.control_zero_refuses
#print axioms Iota.Memo.StoreAudit.control_unrelated_never_replaced

set_option warningAsError false
