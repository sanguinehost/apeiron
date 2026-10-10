import Iota.Table.TableWrite

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Iota.Table

open Iota Iota.Decode Iota.Machine Iota.Source Iota.Refinement

/-- The `MERGE_MAP` write-admission predicate: the key is not the reserved sentinel and the probe
does not exhaust the layout. It is stated on the call, never on the raw return, so a successful
write of `tableEmpty` stays a success. Memory and stack faults are not part of this predicate and
are not recoded as refusal. -/
def WriteAdmitted (memory : Memory) (base cap key : Word) : Prop :=
  key ≠ tableEmpty ∧ ¬ ProbeExhausted memory base cap key

/-- The replacement or insertion success of a checked `tableInsertWith` call, carrying the exact
raw return and post-memory. -/
def WriteSuccess (memory : Memory) (base cap : Word) (occupied : List Word)
    (key value raw : Word) (memory' : Memory) : Prop :=
  (∃ slot, slot < cap.val ∧ member occupied key = true ∧
      keyAt memory base slot = some key ∧ raw = value ∧
      memory' = storeValueAt memory base slot value) ∨
  (∃ slot step, step < cap.val ∧ slot = probeSlot cap key step ∧
      member occupied key = false ∧
      keyAt memory base (probeSlot cap key step) = some tableEmpty ∧ raw = value)

/-- A successful write is never the full-table refusal: the replaced key sits at its probe slot
and the inserted empty slot is a probe slot, so both contradict `ProbeExhausted`. -/
theorem WriteSuccess.not_exhausted {memory : Memory} {base cap default : Word}
    {occupied : List Word} {h : Word → Word} {key value raw : Word} {memory' : Memory}
    (rep : RepSong memory base cap default occupied h)
    (success : WriteSuccess memory base cap occupied key value raw memory') :
    ¬ ProbeExhausted memory base cap key := by
  rintro exhausted
  rcases success with ⟨slot, slotBound, present, keyAtSlot, _, _⟩ |
    ⟨slot, step, stepBound, slotEq, _, emptyAt, _⟩
  · rcases rep.2.2.2.2.2.1 key present with ⟨step, stepBound, probeKey, _⟩
    have probeBound : probeSlot cap key step < cap.val :=
      Nat.mod_lt _ rep.positivePowerOfTwo.pos
    have keyNe : key ≠ tableEmpty := rep.2.2.1.2 key present
    have same : slot = probeSlot cap key step :=
      rep.2.2.2.2.1.1 slot slotBound (probeSlot cap key step) probeBound key keyNe keyAtSlot probeKey
    rw [same] at keyAtSlot
    exact (exhausted step stepBound).2 keyAtSlot
  · exact (exhausted step stepBound).1 emptyAt

/-- On an admitted write the checked call returns a value, replaces or inserts, and the exact raw
return and post-memory are one of the two `WriteSuccess` shapes. -/
theorem tableInsertWith_admitted_success (context : Context)
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
      WriteMapOutcome memory base cap default occupied h key value raw memory' ∧
      WriteSuccess memory base cap occupied key value raw memory' := by
  rcases tableInsertWith_map_execution context capacityOk spread findProbe slotKey slotVal slotSet
    slotValSet hcapacityOk hspread hfindProbe hslotKey hslotVal hslotSet hslotValSet hinsert
    hcapacityShape hspreadShape hfindShape hslotKeyShape hslotValShape hslotSetShape
    hslotValSetShape hstride hempty hmergeSet hmergeAcc hphi memory base cap default key value
    occupied h rep hkey with ⟨raw, memory', trace, hexec, houtcome⟩
  rcases WriteMapOutcome.preserves rep hkey houtcome with
    ⟨slot, slotBound, present, keyAtSlot, hraw, hmem, _⟩ |
    ⟨slot, step, stepBound, slotEq, absent, emptyAt, hraw, _⟩ |
    ⟨_, _, exhausted⟩
  · exact ⟨raw, memory', trace, hexec, houtcome,
      Or.inl ⟨slot, slotBound, present, keyAtSlot, hraw, hmem⟩⟩
  · exact ⟨raw, memory', trace, hexec, houtcome,
      Or.inr ⟨slot, step, stepBound, slotEq, absent, emptyAt, hraw⟩⟩
  · exact False.elim (admitted.2 exhausted)

/-- On an exhausted full layout the checked call returns raw `tableEmpty` (`-1`) and leaves the
memory unchanged. This is the source's explicit full-table refusal; the capacity, reserved-key and
merge-range guards are outside the admitted domain and are not recoded here. -/
theorem tableInsertWith_refusal_full (context : Context)
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
    (hkey : key ≠ tableEmpty) (exhausted : ProbeExhausted memory base cap key) :
    ∃ trace, EvalCall context .unbounded 0 "tableInsertWith"
        [base, cap, key, value, Word.ofNat 2] memory (.value tableEmpty memory) trace := by
  rcases tableInsertWith_map_execution context capacityOk spread findProbe slotKey slotVal slotSet
    slotValSet hcapacityOk hspread hfindProbe hslotKey hslotVal hslotSet hslotValSet hinsert
    hcapacityShape hspreadShape hfindShape hslotKeyShape hslotValShape hslotSetShape
    hslotValSetShape hstride hempty hmergeSet hmergeAcc hphi memory base cap default key value
    occupied h rep hkey with ⟨raw, memory', trace, hexec, houtcome⟩
  rcases WriteMapOutcome.preserves rep hkey houtcome with
    ⟨slot, slotBound, present, keyAtSlot, hraw, hmem, _⟩ |
    ⟨slot, step, stepBound, slotEq, absent, emptyAt, hraw, _⟩ |
    ⟨hraw, hmem, _⟩
  · exact False.elim (WriteSuccess.not_exhausted rep
      (Or.inl ⟨slot, slotBound, present, keyAtSlot, hraw, hmem⟩) exhausted)
  · have success : WriteSuccess memory base cap occupied key value raw memory' :=
      Or.inr ⟨slot, step, stepBound, slotEq, absent, emptyAt, hraw⟩
    exact False.elim (WriteSuccess.not_exhausted rep success exhausted)
  · subst hraw
    subst hmem
    exact ⟨trace, hexec⟩

/-- Exact write admission: under `RepSong` and a nonreserved key, the checked call has a
replacement/insertion success exactly when the write-admission predicate holds. The success is
decided by the checked outcome, never by `raw ≠ tableEmpty`. -/
theorem tableInsertWith_admission_exact (context : Context)
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
    (hkey : key ≠ tableEmpty) :
    WriteAdmitted memory base cap key ↔
      ∃ raw memory' trace,
        EvalCall context .unbounded 0 "tableInsertWith"
          [base, cap, key, value, Word.ofNat 2] memory (.value raw memory') trace ∧
        WriteMapOutcome memory base cap default occupied h key value raw memory' ∧
        WriteSuccess memory base cap occupied key value raw memory' := by
  constructor
  · intro admitted
    exact tableInsertWith_admitted_success context capacityOk spread findProbe slotKey slotVal
      slotSet slotValSet hcapacityOk hspread hfindProbe hslotKey hslotVal hslotSet hslotValSet
      hinsert hcapacityShape hspreadShape hfindShape hslotKeyShape hslotValShape hslotSetShape
      hslotValSetShape hstride hempty hmergeSet hmergeAcc hphi memory base cap default key value
      occupied h rep hkey admitted
  · rintro ⟨raw, memory', trace, _, _, success⟩
    exact ⟨hkey, WriteSuccess.not_exhausted rep success⟩

end Iota.Table
