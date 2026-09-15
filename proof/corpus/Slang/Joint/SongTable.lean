import Slang.Table.TableAdmission

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# W29: replacement execution linked to canonical `Apeiron.Table.insertWith`

The decoded `tableInsertWith` body on the `MERGE_MAP` face stores and returns `value` on both the
replacement and the fresh-key path. This module links that checked execution to the canonical
`Apeiron.Table.insertWith (fun new _old => new)`: the write preserves `RepSong` for the occupied set
extended by the written key, it keeps the exact raw return and the actual post-memory, and every
read observation at an occupied key is the canonical `Apeiron.Table.get`. Accumulator semantics are
out of scope here; the fresh-key `MERGE_ACC` path and the canonical default are a separate
specification.
-/

namespace Slang.Joint

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement Slang.Table

/-- Adding a key already present leaves the occupied list unchanged. -/
theorem insertOccupied_of_member_true (occupied : List Word) (key : Word)
    (present : member occupied key = true) : insertOccupied occupied key = occupied := by
  unfold insertOccupied
  rw [if_pos present]

/-- Adding an absent key prepends it to the occupied list. -/
theorem insertOccupied_of_member_false (occupied : List Word) (key : Word)
    (absent : member occupied key = false) : insertOccupied occupied key = key :: occupied := by
  unfold insertOccupied
  rw [if_neg (fun equal => Bool.noConfusion (absent.symm.trans equal))]

/-- On an occupied key the state-aware observation returns the map value, which is the canonical
`Apeiron.Table.get`. The absent branch is deliberately not claimed here. -/
theorem observeRead_get {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word} {key raw : Word}
    (observation : observeRead memory base cap default occupied h key raw)
    (present : member occupied key = true) :
    raw = Apeiron.Table.get h key := by
  have second := observation.2
  rw [present] at second
  change ReadObservation true default (h key) raw at second
  cases second with
  | present _ _ _ agrees => exact agrees

/-- **W29 correspondence headline.** For an admitted `MERGE_MAP` write through the decoded
`tableInsertWith`, the checked call returns `value` and the actual post-memory `memory'`; `RepSong`
is preserved for the occupied set `insertOccupied occupied key` under the canonical replacement map
`Apeiron.Table.insertWith (fun new _old => new) key value h`; and every read observation on
`memory'` at an occupied key is the canonical `Apeiron.Table.get` of that map. -/
theorem songTable_replacement_link (context : Context)
    (capacityOk spread findProbe slotKey slotVal slotSet slotValSet : FunctionDecl)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hspread : context.functions "tableSpread" = some spread)
    (hfindProbe : context.functions "tableFindSlotGo" = some findProbe)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hslotVal : context.functions "tableSlotVal" = some slotVal)
    (hslotSet : context.functions "tableSlotSet" = some slotSet)
    (hslotValSet : context.functions "tableSlotValSet" = some slotValSet)
    (hinsert : context.functions "tableInsertWith" = some insertWithDecl)
    (hcapacityShape : TableCapacityOkShape capacityOk)
    (hspreadShape : TableSpreadShape spread)
    (hfindShape : TableFindSlotGoShape findProbe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hslotValShape : TableSlotValShape slotVal)
    (hslotSetShape : TableSlotSetShape slotSet)
    (hslotValSetShape : TableSlotValSetShape slotValSet)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (hmergeSet : context.constants "MERGE_SET" = some (Word.ofNat 1))
    (hmergeAcc : context.constants "MERGE_ACC" = some (Word.ofNat 3))
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (base cap default key value : Word) (occupied : List Word)
    (h : Word → Word) (rep : RepSong memory base cap default occupied h)
    (hkey : key ≠ tableEmpty) (admitted : WriteAdmitted memory base cap key) :
    ∃ raw memory' trace,
      EvalCall context .unbounded 0 "tableInsertWith"
        [base, cap, key, value, Word.ofNat 2] memory (.value raw memory') trace ∧
      raw = value ∧
      RepSong memory' base cap default (insertOccupied occupied key)
        (Apeiron.Table.insertWith (fun new _old => new) key value h) ∧
      (∀ key' raw', observeRead memory' base cap default (insertOccupied occupied key)
          (Apeiron.Table.insertWith (fun new _old => new) key value h) key' raw' →
        member (insertOccupied occupied key) key' = true →
          raw' = Apeiron.Table.get
            (Apeiron.Table.insertWith (fun new _old => new) key value h) key') := by
  rcases tableInsertWith_map_execution context capacityOk spread findProbe slotKey slotVal slotSet
    slotValSet hcapacityOk hspread hfindProbe hslotKey hslotVal hslotSet hslotValSet hinsert
    hcapacityShape hspreadShape hfindShape hslotKeyShape hslotValShape hslotSetShape
    hslotValSetShape hstride hempty hmergeSet hmergeAcc hphi memory base cap default key value
    occupied h rep hkey with ⟨raw, memory', trace, hexec, outcome⟩
  rcases WriteMapOutcome.preserves rep hkey outcome with
    ⟨slot, slotBound, present, keyAtSlot, hraw, hmem, hrep⟩ |
    ⟨slot, step, stepBound, slotEq, absent, emptyAt, hraw, hrep⟩ |
    ⟨hraw, hmem, exhausted⟩
  · have hocc := insertOccupied_of_member_true occupied key present
    refine ⟨raw, memory', trace, hexec, hraw, ?_, ?_⟩
    · rw [hocc]
      exact hrep
    · intro key' raw' observation memberKey'
      exact observeRead_get observation memberKey'
  · have hocc := insertOccupied_of_member_false occupied key absent
    refine ⟨raw, memory', trace, hexec, hraw, ?_, ?_⟩
    · rw [hocc]
      exact hrep
    · intro key' raw' observation memberKey'
      exact observeRead_get observation memberKey'
  · exact False.elim (admitted.2 exhausted)

end Slang.Joint
