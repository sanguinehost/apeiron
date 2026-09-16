import Slang.Table.TableProbe

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

def TableGetShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap", "key"] ∧
    declaration.body =
      [.bind "slot" (.call "tableFind" [.name "base", .name "cap", .name "key"]),
       .if_ (.binary .lt (.name "slot") (.literal 0))
         [.return_ (.name "TABLE_EMPTY")] none,
       .return_ (.call "tableSlotVal" [.name "base", .name "slot"])]

inductive ReadObservation : Bool → Word → Word → Word → Prop where
  | present (default value raw : Word) (agrees : raw = value) :
      ReadObservation true default value raw
  | absent (default value raw : Word) (sentinel : raw = tableEmpty) :
      ReadObservation false default value raw

def observeRead (memory : Memory) (base cap default : Word) (occupied : List Word)
    (h : Word → Word) (key raw : Word) : Prop :=
  RepSong memory base cap default occupied h ∧
    if member occupied key then ReadObservation true default (h key) raw
    else ReadObservation false default default raw

theorem tableSlotVal_execution (context : Context) (slotVal : FunctionDecl)
    (hfunction : context.functions "tableSlotVal" = some slotVal)
    (hshape : TableSlotValShape slotVal)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (memory : Memory) (base slot : Word) (depth : Nat) (raw : Word)
    (hcoordinate : valueCoordinate base slot.val < wordSignBit)
    (howned : memory.owned (valueCoordinate base slot.val) = true)
    (hvalue : valueAt memory base slot.val = some raw) :
    EvalCall context .unbounded depth "tableSlotVal" [base, slot] memory
      (.value raw memory)
      [.call "tableSlotVal" (depth + 1), .arithmetic .mul, .arithmetic .add,
        .arithmetic .add, .load (Word.ofNat (valueCoordinate base slot.val))] := by
  let environment : Environment := [("base", base), ("slot", slot)]
  have hframe : Environment.ofParameters slotVal.parameters [base, slot] = some environment := by
    rw [hshape.1]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    rfl
  have hslot : Environment.lookup environment "slot" = some slot := by
    rfl
  have hlocal : Environment.lookup environment "TABLE_STRIDE" = none := by
    rfl
  have haddress := eval_slot_value_address context .unbounded (depth + 1) environment memory
    base slot hbase hslot hlocal hstride
  have hload : memory.load (Word.ofNat (valueCoordinate base slot.val)) = some raw := by
    unfold Memory.load
    rw [coordinate_ofNat_emptyCone _ hcoordinate]
    change (if memory.owned (valueCoordinate base slot.val) then
      memory.initialized (valueCoordinate base slot.val) else none) = some raw
    rw [howned]
    exact hvalue
  apply EvalCall.run (policy := .unbounded) (depth := depth) hfunction hframe True.intro
  rw [hshape.2]
  exact ExecList.returned (ExecStmt.return_ (EvalExpr.load haddress hload))

theorem wordLt_ofNat_zero (value : Nat) (bound : value < wordSignBit) :
    Word.lt (Word.ofNat value) (Word.ofNat 0) = Word.ofNat 0 := by
  have nonnegative : ¬Word.signed (Word.ofNat value) < Word.signed (Word.ofNat 0) := by
    rw [signed_ofNat_emptyCone 0 wordSignBit_pos]
    exact signed_ofNat_nonnegative value bound
  unfold Word.lt Word.bool
  rw [decide_eq_false nonnegative]
  rfl

theorem tableEmpty_lt_zero : Word.lt tableEmpty (Word.ofNat 0) = Word.ofNat 1 := by
  have negative : Word.signed tableEmpty < Word.signed (Word.ofNat 0) := by
    rw [show Word.signed tableEmpty = -1 by exact Word.signed_allOnes]
    rw [signed_ofNat_emptyCone 0 wordSignBit_pos]
    decide
  unfold Word.lt Word.bool
  rw [decide_eq_true negative]
  rfl

theorem tableGet_execution (context : Context)
    (slotKey slotVal capacityOk spread lookup find get : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hslotVal : context.functions "tableSlotVal" = some slotVal)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hspread : context.functions "tableSpread" = some spread)
    (hlookup : context.functions "tableLookupSlotGo" = some lookup)
    (hfind : context.functions "tableFind" = some find)
    (hget : context.functions "tableGet" = some get)
    (hslotKeyShape : TableSlotKeyShape slotKey) (hslotValShape : TableSlotValShape slotVal)
    (hcapacityShape : TableCapacityOkShape capacityOk) (hspreadShape : TableSpreadShape spread)
    (hlookupShape : TableLookupSlotGoShape lookup) (hfindShape : TableFindShape find)
    (hgetShape : TableGetShape get)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (rep : RepSong memory base cap default occupied h)
    (hkey : key ≠ tableEmpty) :
    ∃ raw trace,
      EvalCall context .unbounded 0 "tableGet" [base, cap, key] memory
        (.value raw memory) trace ∧
      observeRead memory base cap default occupied h key raw := by
  rcases tableFind_execution context slotKey capacityOk spread lookup find hslotKey hcapacityOk
    hspread hlookup hfind hslotKeyShape hcapacityShape hspreadShape hlookupShape hfindShape
    hstride hempty hphi memory base cap default key occupied h 1 rep hkey with
    ⟨slotRaw, findTrace, hfindCall, houtcome⟩
  let environment : Environment := [("base", base), ("cap", cap), ("key", key)]
  have hframe : Environment.ofParameters get.parameters [base, cap, key] = some environment := by
    rw [hgetShape.1]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    rfl
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    rfl
  have hkeyLocal : Environment.lookup environment "key" = some key := by
    rfl
  have hfindArgs : EvalArgs context .unbounded 1 environment memory
      [.name "base", .name "cap", .name "key"] (.value [base, cap, key] memory) [] :=
    EvalArgs.cons (EvalExpr.local hbase)
      (EvalArgs.cons (EvalExpr.local hcapLocal)
        (EvalArgs.cons (EvalExpr.local hkeyLocal) EvalArgs.nil))
  have hfindExpr : EvalExpr context .unbounded 1 environment memory
      (.call "tableFind" [.name "base", .name "cap", .name "key"])
      (.value slotRaw memory) findTrace := by
    exact EvalExpr.call hfindArgs hfindCall
  let slotEnvironment := Environment.bind environment "slot" slotRaw
  have hslotBind : ExecStmt context .unbounded 1 environment memory
      (.bind "slot" (.call "tableFind" [.name "base", .name "cap", .name "key"]))
      (.next slotEnvironment memory) findTrace := ExecStmt.bind hfindExpr
  have hslotLocal : Environment.lookup slotEnvironment "slot" = some slotRaw :=
    Environment.lookup_bind_same _ _ _
  have hemptyLocal : Environment.lookup slotEnvironment "TABLE_EMPTY" = none := by
    rfl
  have hbaseSlot : Environment.lookup slotEnvironment "base" = some base := by
    rw [Environment.lookup_bind_other _ "slot" "base" slotRaw (by decide)]
    exact hbase
  cases hmember : member occupied key with
  | true =>
    rcases rep.2.2.2.2.2.1 key hmember with ⟨found, hfound, hmatch, hprefix⟩
    rcases ProbeOutcome.lookup_present memory base cap key 0 found slotRaw houtcome
      (Nat.zero_le found) hfound hmatch
      (fun earlier _ hearler => hprefix earlier hearler) (rep.2.2.1.2 key hmember) with
      ⟨slot, hslotCap, hslotRaw, hslotKey⟩
    subst slotRaw
    have hslotSign : slot < wordSignBit := lt_trans hslotCap
      (cap_below_signBit_of_backing memory base cap rep.validExtent.1)
    have hltZero : Word.lt (Word.ofNat slot) (Word.ofNat 0) = Word.ofNat 0 := by
      exact wordLt_ofNat_zero slot hslotSign
    have hltEval : EvalExpr context .unbounded 1 slotEnvironment memory
        (.binary .lt (.name "slot") (.literal 0))
        (.value (Word.ofNat 0) memory) [.comparison .lt] := by
      rw [← hltZero]
      simpa only [Machine.applyBinOp, Machine.operatorEvent, List.nil_append] using
        EvalExpr.binary (operator := .lt) (EvalExpr.local hslotLocal) EvalExpr.literal
    have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hz => hz rfl
    have hguard : ExecStmt context .unbounded 1 slotEnvironment memory
        (.if_ (.binary .lt (.name "slot") (.literal 0))
          [.return_ (.name "TABLE_EMPTY")] none)
        (.next slotEnvironment memory) [.comparison .lt] :=
      ExecStmt.ifFalse hltEval hzero ExecList.nil
    have hvalue := rep.2.2.2.2.2.2.1 slot hslotCap key hmember hslotKey
    have hslotNatVal : (Word.ofNat slot).val = slot := ofNat_val_emptyCone slot hslotSign
    have hvalueCall := tableSlotVal_execution context slotVal hslotVal hslotValShape hstride
      memory base (Word.ofNat slot) 1 (h key)
      (by
        rw [hslotNatVal]
        exact valueCoordinate_below_signBit_emptyCone base cap slot rep.validExtent.1.2.1 hslotCap)
      (by
        rw [hslotNatVal]
        exact rep.owned_extent (2 * slot + 1) (by
          have doubled : 2 * slot + 2 ≤ 2 * cap.val := by
            simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt hslotCap)
          exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * slot + 1)) doubled))
      (by rw [hslotNatVal]; exact hvalue)
    rw [hslotNatVal] at hvalueCall
    have hvalueArgs : EvalArgs context .unbounded 1 slotEnvironment memory
        [.name "base", .name "slot"] (.value [base, Word.ofNat slot] memory) [] :=
      EvalArgs.cons (EvalExpr.local hbaseSlot)
        (EvalArgs.cons (EvalExpr.local hslotLocal) EvalArgs.nil)
    have hreturn : ExecStmt context .unbounded 1 slotEnvironment memory
        (.return_ (.call "tableSlotVal" [.name "base", .name "slot"]))
        (.returned (h key) memory)
        [.call "tableSlotVal" 2, .arithmetic .mul, .arithmetic .add, .arithmetic .add,
          .load (Word.ofNat (valueCoordinate base slot))] := by
      apply ExecStmt.return_
      exact EvalExpr.call hvalueArgs hvalueCall
    have hbody : ExecList context .unbounded 1 environment memory get.body
        (.returned (h key) memory)
        (findTrace ++ [.comparison .lt] ++
          [.call "tableSlotVal" 2, .arithmetic .mul, .arithmetic .add, .arithmetic .add,
            .load (Word.ofNat (valueCoordinate base slot))]) := by
      rw [hgetShape.2]
      simpa only [listAppendAssoc] using ExecList.next hslotBind
        (ExecList.next hguard (ExecList.returned hreturn))
    let trace : Trace := .call "tableGet" 1 ::
      (findTrace ++ [.comparison .lt] ++
        [.call "tableSlotVal" 2, .arithmetic .mul, .arithmetic .add, .arithmetic .add,
          .load (Word.ofNat (valueCoordinate base slot))])
    refine ⟨h key, trace, ?_, ?_⟩
    · exact EvalCall.run hget hframe True.intro hbody
    · refine ⟨rep, ?_⟩
      rw [hmember]
      exact .present default (h key) (h key) rfl
  | false =>
    have hmiss := houtcome.lookup_miss memory base cap default key occupied h 0 slotRaw rep
      hmember hkey
    subst slotRaw
    have hltOne : Word.lt tableEmpty (Word.ofNat 0) = Word.ofNat 1 := by
      exact tableEmpty_lt_zero
    have hltEval : EvalExpr context .unbounded 1 slotEnvironment memory
        (.binary .lt (.name "slot") (.literal 0))
        (.value (Word.ofNat 1) memory) [.comparison .lt] := by
      rw [← hltOne]
      simpa only [Machine.applyBinOp, Machine.operatorEvent, List.nil_append] using
        EvalExpr.binary (operator := .lt) (EvalExpr.local hslotLocal) EvalExpr.literal
    have hone : Word.nonzero (Word.ofNat 1) := by
      exact wordOne_nonzero
    have hguard : ExecStmt context .unbounded 1 slotEnvironment memory
        (.if_ (.binary .lt (.name "slot") (.literal 0))
          [.return_ (.name "TABLE_EMPTY")] none)
        (.returned tableEmpty memory) [.comparison .lt] := by
      apply ExecStmt.ifTrue hltEval hone
      exact ExecList.returned
        (ExecStmt.return_ (EvalExpr.constant hemptyLocal hempty))
    have hbody : ExecList context .unbounded 1 environment memory get.body
        (.returned tableEmpty memory) (findTrace ++ [.comparison .lt]) := by
      rw [hgetShape.2]
      exact ExecList.next hslotBind (ExecList.returned hguard)
    let trace : Trace := .call "tableGet" 1 :: (findTrace ++ [.comparison .lt])
    refine ⟨tableEmpty, trace, ?_, ?_⟩
    · exact EvalCall.run hget hframe True.intro hbody
    · refine ⟨rep, ?_⟩
      rw [hmember]
      exact .absent default default tableEmpty rfl

theorem tableGet_read_correct (context : Context)
    (slotKey slotVal capacityOk spread lookup find get : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hslotVal : context.functions "tableSlotVal" = some slotVal)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hspread : context.functions "tableSpread" = some spread)
    (hlookup : context.functions "tableLookupSlotGo" = some lookup)
    (hfind : context.functions "tableFind" = some find)
    (hget : context.functions "tableGet" = some get)
    (hslotKeyShape : TableSlotKeyShape slotKey) (hslotValShape : TableSlotValShape slotVal)
    (hcapacityShape : TableCapacityOkShape capacityOk) (hspreadShape : TableSpreadShape spread)
    (hlookupShape : TableLookupSlotGoShape lookup) (hfindShape : TableFindShape find)
    (hgetShape : TableGetShape get)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (rep : RepSong memory base cap default occupied h)
    (hkey : key ≠ tableEmpty) :
    ∃ raw trace,
      EvalCall context .unbounded 0 "tableGet" [base, cap, key] memory
        (.value raw memory) trace ∧
      observeRead memory base cap default occupied h key raw ∧
      (member occupied key = true → raw = Apeiron.Table.get h key) := by
  rcases tableGet_execution context slotKey slotVal capacityOk spread lookup find get hslotKey
    hslotVal hcapacityOk hspread hlookup hfind hget hslotKeyShape hslotValShape hcapacityShape
    hspreadShape hlookupShape hfindShape hgetShape hstride hempty hphi memory base cap default key
    occupied h rep hkey with ⟨raw, trace, execution, observation⟩
  refine ⟨raw, trace, execution, observation, ?_⟩
  intro hmember
  have observed := observation.2
  rw [hmember] at observed
  change ReadObservation true default (h key) raw at observed
  cases observed with
  | present _ _ _ agrees => exact agrees

theorem tableGet_execution_of_certificate
    (value : CertifiedDecoded) (program : Program)
    (derivation : DecodeDerivation Slang.Generated.TableDerivation.tableSource
      Slang.Generated.TableDerivation.stdSource value)
    (programEq : value.val.program = program)
    (selectedSyntax : Slang.SourceAudit.ExpectedSelectedSyntax program = true)
    (closure : ParsesClosure Slang.Generated.TableDerivation.entryBundle program)
    (slotKey slotVal capacityOk spread lookup find get : FunctionDecl)
    (hslotKey : (checkedContext program).functions "tableSlotKey" = some slotKey)
    (hslotVal : (checkedContext program).functions "tableSlotVal" = some slotVal)
    (hcapacityOk : (checkedContext program).functions "tableCapacityOk" = some capacityOk)
    (hspread : (checkedContext program).functions "tableSpread" = some spread)
    (hlookup : (checkedContext program).functions "tableLookupSlotGo" = some lookup)
    (hfind : (checkedContext program).functions "tableFind" = some find)
    (hget : (checkedContext program).functions "tableGet" = some get)
    (hslotKeyShape : TableSlotKeyShape slotKey) (hslotValShape : TableSlotValShape slotVal)
    (hcapacityShape : TableCapacityOkShape capacityOk) (hspreadShape : TableSpreadShape spread)
    (hlookupShape : TableLookupSlotGoShape lookup) (hfindShape : TableFindShape find)
    (hgetShape : TableGetShape get)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty)
    (hphi : (checkedContext program).constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (rep : RepSong memory base cap default occupied h)
    (hkey : key ≠ tableEmpty) :
    ∃ raw trace,
      Decode.decodeClosure (encodeBytes Slang.Generated.TableDerivation.tableSource)
          (encodeBytes Slang.Generated.TableDerivation.stdSource) = .ok value ∧
      EvalCall (checkedContext program) .unbounded 0 "tableGet" [base, cap, key] memory
        (.value raw memory) trace ∧
      observeRead memory base cap default occupied h key raw ∧
      (member occupied key = true → raw = Apeiron.Table.get h key) := by
  subst program
  have source := Slang.Source.decodeDerivation_sound derivation
  rcases tableGet_read_correct (checkedContext value.val.program) slotKey slotVal capacityOk spread
    lookup find get hslotKey hslotVal hcapacityOk hspread hlookup hfind hget hslotKeyShape
    hslotValShape hcapacityShape hspreadShape hlookupShape hfindShape hgetShape hstride hempty hphi
    memory base cap default key occupied h rep hkey with ⟨raw, trace, execution, observation, link⟩
  exact ⟨raw, trace, source, execution, observation, link⟩

end Slang.Table
