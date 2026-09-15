import Slang.Table.TableProbe

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

/-! ## Termination of the decoded recursion

`tableInitGo`, `tableFindSlotGo` and `tableLookupSlotGo` recurse on a cursor that the decoded guard
compares against `cap` and that the decoded recursive edge increments through the wrapping word
`Word.add`. The measures below are the decoded `cap - i` and `cap - step`. The lemmas connect the
measure to the decoded edge: the guard is true exactly at the bound, and the recursive argument is
the wrapping successor. The termination headlines then state the finite `EvalCall` derivation that
`TableInit`/`TableProbe` construct by strong induction on exactly these measures. -/

/-- The decoded measure of `tableInitGo`: `cap - i`. -/
def initGoMeasure (cap : Word) (i : Nat) : Nat := cap.val - i

/-- The decoded measure of both probe functions: `cap - step`. -/
def probeMeasure (cap : Word) (step : Nat) : Nat := cap.val - step

theorem initGoMeasure_decrease (cap : Word) (i : Nat) (bound : i < cap.val) :
    initGoMeasure cap (i + 1) < initGoMeasure cap i := by
  unfold initGoMeasure
  rw [sub_succ_difference bound]
  exact Nat.lt_succ_self _

theorem probeMeasure_decrease (cap : Word) (step : Nat) (bound : step < cap.val) :
    probeMeasure cap (step + 1) < probeMeasure cap step := by
  unfold probeMeasure
  rw [sub_succ_difference bound]
  exact Nat.lt_succ_self _

theorem initGoMeasure_zero_at_bound (cap : Word) (i : Nat) (bound : i = cap.val) :
    initGoMeasure cap i = 0 := by
  unfold initGoMeasure
  rw [bound, Nat.sub_self]

theorem probeMeasure_zero_at_bound (cap : Word) (step : Nat) (bound : step = cap.val) :
    probeMeasure cap step = 0 := by
  unfold probeMeasure
  rw [bound, Nat.sub_self]

/-- The decoded recursive argument of `tableInitGo` is the wrapping word `i + 1`; the successor
stays below the sign bit, so the wrapping result has the readable representative `i + 1`. -/
theorem initGo_recursive_argument (i : Nat) (bound : i + 1 < wordSignBit) :
    Word.add (Word.ofNat i) (Word.ofNat 1) = Word.ofNat (i + 1) ∧
      (Word.ofNat (i + 1)).val = i + 1 := by
  constructor
  · apply Fin.ext
    change ((i % wordModulus + 1 % wordModulus) % wordModulus) = (i + 1) % wordModulus
    rw [modEqOfLt (by decide : 1 < wordModulus), modAddAbsorbLeft]
  · exact ofNat_val_emptyCone (i + 1) bound

/-- The decoded recursive argument of both probes is the wrapping word `step + 1`, again below the
sign bit. -/
theorem probe_recursive_argument (step : Nat) (bound : step + 1 < wordSignBit) :
    Word.add (Word.ofNat step) (Word.ofNat 1) = Word.ofNat (step + 1) ∧
      (Word.ofNat (step + 1)).val = step + 1 := by
  constructor
  · apply Fin.ext
    change ((step % wordModulus + 1 % wordModulus) % wordModulus) = (step + 1) % wordModulus
    rw [modEqOfLt (by decide : 1 < wordModulus), modAddAbsorbLeft]
  · exact ofNat_val_emptyCone (step + 1) bound

/-- The decoded `tableInitGo` guard is satisfied exactly at the bound. -/
theorem initGo_guard_at_bound (cap : Word) (sign : cap.val < wordSignBit) :
    Word.ge (Word.ofNat cap.val) cap = Word.ofNat 1 := by
  have capWord : Word.ofNat cap.val = cap := Fin.ext (ofNat_val_emptyCone cap.val sign)
  exact Eq.mp (congrArg (fun right => Word.ge (Word.ofNat cap.val) right = Word.ofNat 1) capWord)
    (wordGe_ofNat_self cap.val sign)

/-- The decoded probe guard is satisfied exactly at the bound. -/
theorem probe_guard_at_bound (cap : Word) (sign : cap.val < wordSignBit) :
    Word.ge (Word.ofNat cap.val) cap = Word.ofNat 1 :=
  initGo_guard_at_bound cap sign

/-- The decoded `tableInitGo` body ends in the recursive call at the wrapping successor. -/
theorem initGo_recursive_edge (declaration : FunctionDecl) (shape : TableInitGoShape declaration) :
    declaration.body =
      [.if_ (.binary .ge (.name "i") (.name "cap")) [.return_ (.literal 0)] none,
       .expression (.call "tableSlotSet"
         [.name "base", .name "i", .name "TABLE_EMPTY", .literal 0]),
       .return_ (.call "tableInitGo"
         [.name "base", .name "cap", .binary .add (.name "i") (.literal 1)])] := shape.2

/-- The decoded probe body ends in the recursive call at the wrapping successor. -/
theorem probe_recursive_edge (callee : String) (emptyResult : Expr) (declaration : FunctionDecl)
    (shape : declaration.parameters = ["base", "cap", "mask", "key", "slot", "step"] ∧
      declaration.body = probeBody callee emptyResult) :
    declaration.body =
      probeBody callee emptyResult := shape.2

/-- `tableInitGo` terminates under its joint hypotheses: the strong induction runs on
`initGoMeasure cap i`, and the derivation is the `EvalCall` returned here. -/
theorem tableInitGo_termination (context : Context) (slotSet initGo : FunctionDecl)
    (hslotSet : context.functions "tableSlotSet" = some slotSet)
    (hinitGo : context.functions "tableInitGo" = some initGo)
    (hslotSetShape : TableSlotSetShape slotSet) (hinitGoShape : TableInitGoShape initGo)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap : Word) (i depth : Nat)
    (hi : i ≤ cap.val) (hbacking : ValidBacking memory base cap) :
    ∃ after trace,
      EvalCall context .unbounded depth "tableInitGo" [base, cap, Word.ofNat i] memory
        (.value (Word.ofNat 0) after) trace := by
  rcases tableInitGo_execution context slotSet initGo hslotSet hinitGo hslotSetShape hinitGoShape
    hstride hempty memory base cap i depth hi hbacking with
    ⟨after, trace, hcall, _⟩
  exact ⟨after, trace, hcall⟩

/-- `tableFindSlotGo` terminates under its joint hypotheses: the strong induction runs on
`probeMeasure cap step`, and the derivation is the `EvalCall` returned here. -/
theorem tableFindSlotGo_termination (context : Context) (slotKey probe : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hprobe : context.functions "tableFindSlotGo" = some probe)
    (hslotKeyShape : TableSlotKeyShape slotKey) (hprobeShape : TableFindSlotGoShape probe)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (step depth : Nat)
    (rep : RepSong memory base cap default occupied h) (hstep : step ≤ cap.val) :
    ∃ raw trace,
      EvalCall context .unbounded depth "tableFindSlotGo"
        [base, cap, Word.sub cap (Word.ofNat 1), key,
          Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1)), Word.ofNat step]
        memory (.value raw memory) trace ∧
      ProbeOutcome true memory base cap key step raw :=
  tableFindSlotGo_execution context slotKey probe hslotKey hprobe hslotKeyShape hprobeShape
    hstride hempty memory base cap default key occupied h step depth rep hstep

/-- `tableLookupSlotGo` terminates under its joint hypotheses. -/
theorem tableLookupSlotGo_termination (context : Context) (slotKey probe : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hprobe : context.functions "tableLookupSlotGo" = some probe)
    (hslotKeyShape : TableSlotKeyShape slotKey) (hprobeShape : TableLookupSlotGoShape probe)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (step depth : Nat)
    (rep : RepSong memory base cap default occupied h) (hstep : step ≤ cap.val) :
    ∃ raw trace,
      EvalCall context .unbounded depth "tableLookupSlotGo"
        [base, cap, Word.sub cap (Word.ofNat 1), key,
          Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1)), Word.ofNat step]
        memory (.value raw memory) trace ∧
      ProbeOutcome false memory base cap key step raw :=
  tableLookupSlotGo_execution context slotKey probe hslotKey hprobe hslotKeyShape hprobeShape
    hstride hempty memory base cap default key occupied h step depth rep hstep

theorem arithmeticCount_append (left right : Trace) :
    arithmeticCount (left ++ right) = arithmeticCount left + arithmeticCount right := by
  induction left with
  | nil => rw [List.nil_append, arithmeticCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | arithmetic _ =>
          change arithmeticCount (rest ++ right) + 1 =
            (arithmeticCount rest + 1) + arithmeticCount right
          rw [ih, Nat.add_right_comm]
      | comparison _ | bitwise _ | load _ | store _ | call _ _ =>
          exact ih

theorem comparisonCount_append (left right : Trace) :
    comparisonCount (left ++ right) = comparisonCount left + comparisonCount right := by
  induction left with
  | nil => rw [List.nil_append, comparisonCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | comparison _ =>
          change comparisonCount (rest ++ right) + 1 =
            (comparisonCount rest + 1) + comparisonCount right
          rw [ih, Nat.add_right_comm]
      | arithmetic _ | bitwise _ | load _ | store _ | call _ _ =>
          exact ih

theorem bitwiseCount_append (left right : Trace) :
    bitwiseCount (left ++ right) = bitwiseCount left + bitwiseCount right := by
  induction left with
  | nil => rw [List.nil_append, bitwiseCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | bitwise _ =>
          change bitwiseCount (rest ++ right) + 1 =
            (bitwiseCount rest + 1) + bitwiseCount right
          rw [ih, Nat.add_right_comm]
      | arithmetic _ | comparison _ | load _ | store _ | call _ _ =>
          exact ih

theorem loadCount_append (left right : Trace) :
    loadCount (left ++ right) = loadCount left + loadCount right := by
  induction left with
  | nil => rw [List.nil_append, loadCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | load _ =>
          change loadCount (rest ++ right) + 1 =
            (loadCount rest + 1) + loadCount right
          rw [ih, Nat.add_right_comm]
      | arithmetic _ | comparison _ | bitwise _ | store _ | call _ _ =>
          exact ih

theorem storeCount_append (left right : Trace) :
    storeCount (left ++ right) = storeCount left + storeCount right := by
  induction left with
  | nil => rw [List.nil_append, storeCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | store _ =>
          change storeCount (rest ++ right) + 1 =
            (storeCount rest + 1) + storeCount right
          rw [ih, Nat.add_right_comm]
      | arithmetic _ | comparison _ | bitwise _ | load _ | call _ _ =>
          exact ih

theorem callCount_append (left right : Trace) :
    callCount (left ++ right) = callCount left + callCount right := by
  induction left with
  | nil => rw [List.nil_append, callCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | call _ _ =>
          change callCount (rest ++ right) + 1 =
            (callCount rest + 1) + callCount right
          rw [ih, Nat.add_right_comm]
      | arithmetic _ | comparison _ | bitwise _ | load _ | store _ =>
          exact ih

theorem isProbe_probeName (b : Bool) : isProbe (probeName b) = true := by
  cases b <;> rfl

theorem not_isProbe_tableSlotKey : isProbe "tableSlotKey" = false := rfl

theorem probeIterationCount_append (left right : Trace) :
    probeIterationCount (left ++ right) = probeIterationCount left + probeIterationCount right := by
  induction left with
  | nil => rw [List.nil_append, probeIterationCount, Nat.zero_add]
  | cons event rest ih =>
      cases event with
      | call callee depth =>
          change (probeIterationCount (rest ++ right) +
              if isProbe callee = true then 1 else 0) =
            (probeIterationCount rest + if isProbe callee = true then 1 else 0) +
              probeIterationCount right
          by_cases hp : isProbe callee = true
          · rw [if_pos hp, ih, Nat.add_right_comm]
          · rw [if_neg hp, ih, Nat.add_zero, Nat.add_zero]
      | arithmetic _ | comparison _ | bitwise _ | load _ | store _ =>
          exact ih

theorem natMax_le {left right bound : Nat} (hleft : left ≤ bound) (hright : right ≤ bound) :
    max left right ≤ bound := by
  rw [Nat.max_def]
  by_cases order : left ≤ right
  · rw [if_pos order]
    exact hright
  · rw [if_neg order]
    exact hleft

theorem natLe_max_left (left right : Nat) : left ≤ max left right := by
  rw [Nat.max_def]
  by_cases order : left ≤ right
  · rw [if_pos order]
    exact order
  · rw [if_neg order]

theorem natLe_max_right (left right : Nat) : right ≤ max left right := by
  rw [Nat.max_def]
  by_cases order : left ≤ right
  · rw [if_pos order]
  · rw [if_neg order]
    exact Nat.le_of_not_ge order

theorem natMax_comm (left right : Nat) : max left right = max right left := by
  apply Nat.le_antisymm
  · exact natMax_le (natLe_max_right right left) (natLe_max_left right left)
  · exact natMax_le (natLe_max_right left right) (natLe_max_left left right)

theorem natMax_assoc (left middle right : Nat) :
    max (max left middle) right = max left (max middle right) := by
  apply Nat.le_antisymm
  · exact natMax_le
      (natMax_le (natLe_max_left left (max middle right))
        (Nat.le_trans (natLe_max_left middle right) (natLe_max_right left (max middle right))))
      (Nat.le_trans (natLe_max_right middle right) (natLe_max_right left (max middle right)))
  · exact natMax_le
      (Nat.le_trans (natLe_max_left left middle) (natLe_max_left (max left middle) right))
      (natMax_le
        (Nat.le_trans (natLe_max_right left middle) (natLe_max_left (max left middle) right))
        (natLe_max_right (max left middle) right))

theorem callStackMaximum_append (left right : Trace) :
    callStackMaximum (left ++ right) = max (callStackMaximum left) (callStackMaximum right) := by
  induction left with
  | nil => rfl
  | cons event rest ih =>
      cases event with
      | call callee depth =>
          change max (callStackMaximum (rest ++ right)) depth =
            max (max (callStackMaximum rest) depth) (callStackMaximum right)
          rw [ih, natMax_assoc, natMax_assoc,
            natMax_comm (callStackMaximum right) depth]
      | arithmetic _ | comparison _ | bitwise _ | load _ | store _ =>
          exact ih

-- ---------------------------------------------------------------------------
-- Initialization trace shapes and their exact counts
-- ---------------------------------------------------------------------------

def initGoBaseTrace (depth : Nat) : Trace :=
  [.call "tableInitGo" (depth + 1), .comparison .ge]

def initGoStepTrace (depth : Nat) (base : Word) (i : Nat) (rest : Trace) : Trace :=
  .call "tableInitGo" (depth + 1) ::
    ([.comparison .ge] ++ (slotSetTrace (depth + 1) base (Word.ofNat i) ++
      ([.arithmetic .add] ++ rest)))

theorem not_isProbe_tableSlotSet : isProbe "tableSlotSet" = false := rfl

theorem slotSetTrace_arithmetic (depth : Nat) (base slot : Word) :
    arithmeticCount (slotSetTrace depth base slot) = 5 := by
  rfl

theorem slotSetTrace_comparisons (depth : Nat) (base slot : Word) :
    comparisonCount (slotSetTrace depth base slot) = 0 := by
  rfl

theorem slotSetTrace_bitwise (depth : Nat) (base slot : Word) :
    bitwiseCount (slotSetTrace depth base slot) = 0 := by
  rfl

theorem slotSetTrace_loads (depth : Nat) (base slot : Word) :
    loadCount (slotSetTrace depth base slot) = 0 := by
  rfl

theorem slotSetTrace_stores (depth : Nat) (base slot : Word) :
    storeCount (slotSetTrace depth base slot) = 2 := by
  rfl

theorem slotSetTrace_calls (depth : Nat) (base slot : Word) :
    callCount (slotSetTrace depth base slot) = 1 := by
  rfl

theorem slotSetTrace_iterations (depth : Nat) (base slot : Word) :
    probeIterationCount (slotSetTrace depth base slot) = 0 := by
  rfl

theorem slotSetTrace_stack (depth : Nat) (base slot : Word) :
    callStackMaximum (slotSetTrace depth base slot) = depth + 1 := by
  rfl

theorem initGoBaseTrace_arithmetic (depth : Nat) :
    arithmeticCount (initGoBaseTrace depth) = 0 := rfl

theorem initGoBaseTrace_comparisons (depth : Nat) :
    comparisonCount (initGoBaseTrace depth) = 1 := rfl

theorem initGoBaseTrace_bitwise (depth : Nat) :
    bitwiseCount (initGoBaseTrace depth) = 0 := rfl

theorem initGoBaseTrace_loads (depth : Nat) :
    loadCount (initGoBaseTrace depth) = 0 := rfl

theorem initGoBaseTrace_stores (depth : Nat) :
    storeCount (initGoBaseTrace depth) = 0 := rfl

theorem initGoBaseTrace_calls (depth : Nat) :
    callCount (initGoBaseTrace depth) = 1 := rfl

theorem initGoBaseTrace_iterations (depth : Nat) :
    probeIterationCount (initGoBaseTrace depth) = 0 := rfl

theorem initGoBaseTrace_stack (depth : Nat) :
    callStackMaximum (initGoBaseTrace depth) = depth + 1 := by
  rfl

theorem initGoStepTrace_arithmetic (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    arithmeticCount (initGoStepTrace depth base i rest) = arithmeticCount rest + 6 := by
  rfl

theorem initGoStepTrace_comparisons (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    comparisonCount (initGoStepTrace depth base i rest) = comparisonCount rest + 1 := by
  rfl

theorem initGoStepTrace_bitwise (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    bitwiseCount (initGoStepTrace depth base i rest) = bitwiseCount rest := by
  rfl

theorem initGoStepTrace_loads (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    loadCount (initGoStepTrace depth base i rest) = loadCount rest := by
  rfl

theorem initGoStepTrace_stores (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    storeCount (initGoStepTrace depth base i rest) = storeCount rest + 2 := by
  rfl

theorem initGoStepTrace_calls (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    callCount (initGoStepTrace depth base i rest) = callCount rest + 2 := by
  rfl

theorem initGoStepTrace_iterations (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    probeIterationCount (initGoStepTrace depth base i rest) = probeIterationCount rest := by
  rfl

theorem initGoStepTrace_stack (depth : Nat) (base : Word) (i : Nat) (rest : Trace) :
    callStackMaximum (initGoStepTrace depth base i rest)
      = max (callStackMaximum rest) (depth + 2) := by
  unfold initGoStepTrace slotSetTrace
  change max (max (callStackMaximum rest) (depth + 2)) (depth + 1) =
    max (callStackMaximum rest) (depth + 2)
  rw [Nat.max_def]
  rw [if_neg (by
    intro bound
    have impossible : depth + 2 ≤ depth + 1 :=
      Nat.le_trans (natLe_max_right (callStackMaximum rest) (depth + 2)) bound
    exact (Nat.not_succ_le_self (depth + 1)) impossible)]

-- ---------------------------------------------------------------------------
-- Initialization resource bounds
-- ---------------------------------------------------------------------------

def initGoArithmeticBound (cap : Word) (i : Nat) : Nat := 6 * (cap.val - i)
def initGoComparisonBound (cap : Word) (i : Nat) : Nat := cap.val - i + 1
def initGoBitwiseBound (_cap : Word) (_i : Nat) : Nat := 0
def initGoLoadBound (_cap : Word) (_i : Nat) : Nat := 0
def initGoStoreBound (cap : Word) (i : Nat) : Nat := 2 * (cap.val - i)
def initGoCallBound (cap : Word) (i : Nat) : Nat := 2 * (cap.val - i) + 1
def initGoIterationBound (_cap : Word) (_i : Nat) : Nat := 0
def initGoStackBound (depth : Nat) (cap : Word) (i : Nat) : Nat := depth + (cap.val - i) + 1

def InitGoResourceBounds (cap : Word) (i depth : Nat) (trace : Trace) : Prop :=
  arithmeticCount trace ≤ initGoArithmeticBound cap i ∧
  comparisonCount trace ≤ initGoComparisonBound cap i ∧
  bitwiseCount trace ≤ initGoBitwiseBound cap i ∧
  loadCount trace ≤ initGoLoadBound cap i ∧
  storeCount trace ≤ initGoStoreBound cap i ∧
  callCount trace ≤ initGoCallBound cap i ∧
  probeIterationCount trace ≤ initGoIterationBound cap i ∧
  callStackMaximum trace ≤ initGoStackBound depth cap i

theorem initGoArithmeticBound_step (cap : Word) (i : Nat) (bound : i < cap.val) :
    initGoArithmeticBound cap (i + 1) + 6 ≤ initGoArithmeticBound cap i := by
  unfold initGoArithmeticBound
  rw [sub_succ_difference bound, Nat.mul_succ]

theorem initGoComparisonBound_step (cap : Word) (i : Nat) (bound : i < cap.val) :
    initGoComparisonBound cap (i + 1) + 1 ≤ initGoComparisonBound cap i := by
  unfold initGoComparisonBound
  rw [sub_succ_difference bound]

theorem initGoBitwiseBound_step (cap : Word) (i : Nat) (_bound : i < cap.val) :
    initGoBitwiseBound cap (i + 1) + 0 ≤ initGoBitwiseBound cap i := by
  unfold initGoBitwiseBound
  exact Nat.le_refl 0

theorem initGoLoadBound_step (cap : Word) (i : Nat) (_bound : i < cap.val) :
    initGoLoadBound cap (i + 1) + 0 ≤ initGoLoadBound cap i := by
  unfold initGoLoadBound
  exact Nat.le_refl 0

theorem initGoStoreBound_step (cap : Word) (i : Nat) (bound : i < cap.val) :
    initGoStoreBound cap (i + 1) + 2 ≤ initGoStoreBound cap i := by
  unfold initGoStoreBound
  rw [sub_succ_difference bound, Nat.mul_succ]

theorem initGoCallBound_step (cap : Word) (i : Nat) (bound : i < cap.val) :
    initGoCallBound cap (i + 1) + 2 ≤ initGoCallBound cap i := by
  apply Nat.le_of_eq
  unfold initGoCallBound
  rw [sub_succ_difference bound, Nat.mul_succ]

theorem initGoIterationBound_step (cap : Word) (i : Nat) (_bound : i < cap.val) :
    initGoIterationBound cap (i + 1) + 0 ≤ initGoIterationBound cap i := by
  unfold initGoIterationBound
  exact Nat.le_refl 0

theorem initGoStackBound_step (cap : Word) (i depth : Nat) (bound : i < cap.val) :
    initGoStackBound (depth + 1) cap (i + 1) = initGoStackBound depth cap i := by
  unfold initGoStackBound
  rw [sub_succ_difference bound]
  rw [Nat.add_assoc (depth + 1) (cap.val - (i + 1)) 1,
      Nat.add_assoc depth 1 (cap.val - (i + 1) + 1),
      Nat.add_comm 1 (cap.val - (i + 1) + 1),
      ← Nat.add_assoc depth (cap.val - (i + 1) + 1) 1]

theorem initGoBaseBounds (cap : Word) (i depth : Nat) (done : i = cap.val) :
    InitGoResourceBounds cap i depth (initGoBaseTrace depth) := by
  subst i
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [initGoBaseTrace_arithmetic]
    unfold initGoArithmeticBound
    rw [Nat.sub_self]
  · rw [initGoBaseTrace_comparisons]
    unfold initGoComparisonBound
    rw [Nat.sub_self]
  · rw [initGoBaseTrace_bitwise]
    exact Nat.le_refl 0
  · rw [initGoBaseTrace_loads]
    exact Nat.le_refl 0
  · rw [initGoBaseTrace_stores]
    exact Nat.zero_le _
  · rw [initGoBaseTrace_calls]
    unfold initGoCallBound
    rw [Nat.sub_self]
  · rw [initGoBaseTrace_iterations]
    exact Nat.le_refl 0
  · rw [initGoBaseTrace_stack]
    unfold initGoStackBound
    rw [Nat.sub_self, Nat.add_zero]

theorem initGoStepBounds (cap : Word) (base : Word) (i depth : Nat) (bound : i < cap.val)
    (rest : Trace) (hrest : InitGoResourceBounds cap (i + 1) (depth + 1) rest) :
    InitGoResourceBounds cap i depth (initGoStepTrace depth base i rest) := by
  obtain ⟨harith, hcomparison, hbitwise, hload, hstore, hcall, hiteration, hstack⟩ := hrest
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [initGoStepTrace_arithmetic]
    exact Nat.le_trans (Nat.add_le_add_right harith 6) (initGoArithmeticBound_step cap i bound)
  · rw [initGoStepTrace_comparisons]
    exact Nat.le_trans (Nat.add_le_add_right hcomparison 1)
      (initGoComparisonBound_step cap i bound)
  · rw [initGoStepTrace_bitwise]
    exact hbitwise
  · rw [initGoStepTrace_loads]
    exact hload
  · rw [initGoStepTrace_stores]
    exact Nat.le_trans (Nat.add_le_add_right hstore 2) (initGoStoreBound_step cap i bound)
  · rw [initGoStepTrace_calls]
    exact Nat.le_trans (Nat.add_le_add_right hcall 2) (initGoCallBound_step cap i bound)
  · rw [initGoStepTrace_iterations]
    exact hiteration
  · rw [initGoStepTrace_stack]
    have hprefix : callStackMaximum rest ≤ initGoStackBound depth cap i :=
      Nat.le_trans hstack (Nat.le_of_eq (initGoStackBound_step cap i depth bound))
    have hself : depth + 2 ≤ initGoStackBound depth cap i := by
      unfold initGoStackBound
      have hpos : 1 ≤ cap.val - i := Nat.succ_le_of_lt (natSub_pos_of_lt bound)
      exact Nat.add_le_add_right (Nat.add_le_add_left hpos depth) 1
    exact natMax_le hprefix hself

theorem tableInitGo_execution_resources (context : Context) (slotSet initGo : FunctionDecl)
    (hslotSet : context.functions "tableSlotSet" = some slotSet)
    (hinitGo : context.functions "tableInitGo" = some initGo)
    (hslotSetShape : TableSlotSetShape slotSet)
    (hinitGoShape : TableInitGoShape initGo)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap : Word) (i depth : Nat)
    (hi : i ≤ cap.val) (hbacking : ValidBacking memory base cap) :
    ∃ after trace,
      EvalCall context .unbounded depth "tableInitGo" [base, cap, Word.ofNat i] memory
        (.value (Word.ofNat 0) after) trace ∧
      InitializedEmptySuffix after base cap i ∧
      storeAddresses trace = expectedInitStoresFrom base cap i ∧
      PreservesOutsideSuffix memory after base cap i ∧
      InitGoResourceBounds cap i depth trace := by
  generalize measureEq : cap.val - i = measure
  revert memory i depth
  refine Nat.strong_induction_on measure ?_
  intro measure inductionHypothesis memory i depth hi hbacking measureEq
  have hcapSign : cap.val < wordSignBit := cap_below_signBit_of_backing memory base cap hbacking
  have hiSign : i < wordSignBit := Nat.lt_of_le_of_lt hi hcapSign
  have hcapWord : Word.ofNat cap.val = cap := by exact Fin.ext (ofNat_val_emptyCone _ hcapSign)
  have hframe : Environment.ofParameters initGo.parameters [base, cap, Word.ofNat i] =
      some [("base", base), ("cap", cap), ("i", Word.ofNat i)] := by
    rw [hinitGoShape.1]
    rfl
  let environment : Environment := [("base", base), ("cap", cap), ("i", Word.ofNat i)]
  have hbase : Environment.lookup environment "base" = some base := by
    simp only [environment]
    exact Environment.lookup_bind_same _ _ _
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    simp only [environment]
    change Environment.lookup (Environment.bind [("cap", cap), ("i", Word.ofNat i)]
      "base" base) "cap" = some cap
    rw [Environment.lookup_bind_other _ "base" "cap" base (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hiLocal : Environment.lookup environment "i" = some (Word.ofNat i) := by
    simp only [environment]
    change Environment.lookup (Environment.bind [("cap", cap), ("i", Word.ofNat i)]
      "base" base) "i" = some (Word.ofNat i)
    rw [Environment.lookup_bind_other _ "base" "i" base (by decide)]
    change Environment.lookup (Environment.bind [("i", Word.ofNat i)] "cap" cap)
      "i" = some (Word.ofNat i)
    rw [Environment.lookup_bind_other _ "cap" "i" cap (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hcondition : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .ge (.name "i") (.name "cap"))
      (.value (Word.ge (Word.ofNat i) cap) memory) [.comparison .ge] := by
    exact EvalExpr.binary (EvalExpr.local hiLocal) (EvalExpr.local hcapLocal)
  by_cases hdone : i = cap.val
  · subst i
    have hge : Word.ge (Word.ofNat cap.val) cap = Word.ofNat 1 := by
      exact Eq.mp (congrArg (fun right =>
        Word.ge (Word.ofNat cap.val) right = Word.ofNat 1) hcapWord)
        (wordGe_ofNat_self cap.val hcapSign)
    have hnonzero : Word.nonzero (Word.ge (Word.ofNat cap.val) cap) := by
      rw [hge]
      intro equal
      exact Nat.noConfusion (congrArg Fin.val equal)
    have hif : ExecStmt context .unbounded (depth + 1) environment memory
        (.if_ (.binary .ge (.name "i") (.name "cap")) [.return_ (.literal 0)] none)
        (.returned (Word.ofNat 0) memory) [.comparison .ge] := by
      apply ExecStmt.ifTrue hcondition hnonzero
      exact ExecList.returned (ExecStmt.return_ EvalExpr.literal)
    have hbody : ExecList context .unbounded (depth + 1) environment memory initGo.body
        (.returned (Word.ofNat 0) memory) [.comparison .ge] := by
      rw [hinitGoShape.2]
      exact ExecList.returned hif
    refine ⟨memory, initGoBaseTrace depth, ?_, ?_, ?_, ?_, ?_⟩
    · exact EvalCall.run (policy := .unbounded) (depth := depth)
        hinitGo hframe True.intro hbody
    · intro slot lower upper
      exact False.elim (Nat.not_lt_of_ge lower upper)
    · unfold storeAddresses expectedInitStoresFrom
      rw [Nat.sub_self]
      rfl
    · exact ⟨rfl, fun coordinate _ => rfl⟩
    · exact initGoBaseBounds cap cap.val depth rfl
  · have hlt : i < cap.val := Nat.lt_of_le_of_ne hi hdone
    have hgeZero : Word.ge (Word.ofNat i) cap = Word.ofNat 0 := by
      rw [← hcapWord]
      exact wordGe_ofNat_lt i cap.val hiSign hcapSign hlt
    have hnotNonzero : ¬ Word.nonzero (Word.ge (Word.ofNat i) cap) := by
      rw [hgeZero]
      exact fun h => h rfl
    have hif : ExecStmt context .unbounded (depth + 1) environment memory
        (.if_ (.binary .ge (.name "i") (.name "cap")) [.return_ (.literal 0)] none)
        (.next environment memory) [.comparison .ge] := by
      apply ExecStmt.ifFalse hcondition hnotNonzero
      exact ExecList.nil
    have hkeyOwned : memory.owned (keyCoordinate base i) = true := by
      have doubled : 2 * i + 2 ≤ 2 * cap.val := by
        simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt hlt)
      exact hbacking.2.2 (2 * i)
        (Nat.lt_of_lt_of_le (Nat.lt_trans (Nat.lt_succ_self (2 * i))
          (Nat.lt_succ_self (2 * i + 1))) doubled)
    have hvalueOwned : memory.owned (valueCoordinate base i) = true := by
      exact hbacking.2.2 (2 * i + 1) (by
        have doubled : 2 * i + 2 ≤ 2 * cap.val := by
          simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt hlt)
        exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * i + 1)) doubled)
    have hkeySign := keyCoordinate_below_signBit_emptyCone base cap i hbacking.2.1 hlt
    have hvalueSign := valueCoordinate_below_signBit_emptyCone base cap i hbacking.2.1 hlt
    rcases tableSlotSet_execution context slotSet hslotSet hslotSetShape hstride memory base
      (Word.ofNat i) tableEmpty (Word.ofNat 0) (depth + 1) (by
        rw [ofNat_val_emptyCone i hiSign]
        exact hkeyOwned) (by
        rw [ofNat_val_emptyCone i hiSign]
        exact hvalueOwned) (by
        rw [ofNat_val_emptyCone i hiSign]
        exact hkeySign) (by
        rw [ofNat_val_emptyCone i hiSign]
        exact hvalueSign) with
      ⟨afterSlot, hslotCall, hslotOwned, hslotKey, hslotValue, hslotOther⟩
    rw [ofNat_val_emptyCone i hiSign] at hslotKey hslotValue hslotOther
    have hslotArgs : EvalArgs context .unbounded (depth + 1) environment memory
        [.name "base", .name "i", .name "TABLE_EMPTY", .literal 0]
        (.value [base, Word.ofNat i, tableEmpty, Word.ofNat 0] memory) [] := by
      exact EvalArgs.cons (EvalExpr.local hbase)
        (EvalArgs.cons (EvalExpr.local hiLocal)
          (EvalArgs.cons (EvalExpr.constant (by
            simp only [environment]
            change Environment.lookup (Environment.bind [("cap", cap), ("i", Word.ofNat i)]
              "base" base) "TABLE_EMPTY" = none
            rw [Environment.lookup_bind_other _ "base" "TABLE_EMPTY" base (by decide)]
            change Environment.lookup (Environment.bind [("i", Word.ofNat i)] "cap" cap)
              "TABLE_EMPTY" = none
            rw [Environment.lookup_bind_other _ "cap" "TABLE_EMPTY" cap (by decide)]
            change Environment.lookup (Environment.bind [] "i" (Word.ofNat i))
              "TABLE_EMPTY" = none
            rw [Environment.lookup_bind_other _ "i" "TABLE_EMPTY" (Word.ofNat i) (by decide)]
            rfl) hempty)
            (EvalArgs.cons EvalExpr.literal EvalArgs.nil)))
    have hslotExpr : ExecStmt context .unbounded (depth + 1) environment memory
        (.expression (.call "tableSlotSet"
          [.name "base", .name "i", .name "TABLE_EMPTY", .literal 0]))
        (.next environment afterSlot) (slotSetTrace (depth + 1) base (Word.ofNat i)) := by
      exact ExecStmt.expression (EvalExpr.call hslotArgs hslotCall)
    have hbackingNext : ValidBacking afterSlot base cap := by
      exact ⟨hbacking.1, hbacking.2.1, fun offset hoffset => by
        rw [hslotOwned]
        exact hbacking.2.2 offset hoffset⟩
    have measureDecrease : cap.val - (i + 1) < measure := by
      rw [← measureEq, sub_succ_difference hlt]
      exact Nat.lt_succ_self _
    rcases inductionHypothesis (cap.val - (i + 1)) measureDecrease afterSlot (i + 1)
      (depth + 1) (Nat.succ_le_of_lt hlt) hbackingNext rfl with
      ⟨after, recursiveTrace, hrecursive, hsuffix, hstores, hrecursiveFrame, hresources⟩
    have hsuccSign : i + 1 < wordSignBit := lt_of_le_of_lt (Nat.succ_le_of_lt hlt) hcapSign
    have hadd : Word.add (Word.ofNat i) (Word.ofNat 1) = Word.ofNat (i + 1) := by
      apply Fin.ext
      change ((i % wordModulus + 1 % wordModulus) % wordModulus) =
        (i + 1) % wordModulus
      rw [modEqOfLt (by decide : 1 < wordModulus), modAddAbsorbLeft]
    have hrecursiveArgs : EvalArgs context .unbounded (depth + 1) environment afterSlot
        [.name "base", .name "cap", .binary .add (.name "i") (.literal 1)]
        (.value [base, cap, Word.ofNat (i + 1)] afterSlot) [.arithmetic .add] := by
      apply EvalArgs.cons (EvalExpr.local hbase)
      apply EvalArgs.cons (EvalExpr.local hcapLocal)
      have hthird : EvalExpr context .unbounded (depth + 1) environment afterSlot
          (.binary .add (.name "i") (.literal 1))
          (.value (Word.ofNat (i + 1)) afterSlot) [.arithmetic .add] := by
        rw [← hadd]
        exact EvalExpr.binary (operator := .add) (EvalExpr.local hiLocal)
          (EvalExpr.literal (literalValue := 1))
      exact EvalArgs.cons hthird EvalArgs.nil
    have hreturn : ExecStmt context .unbounded (depth + 1) environment afterSlot
        (.return_ (.call "tableInitGo"
          [.name "base", .name "cap", .binary .add (.name "i") (.literal 1)]))
        (.returned (Word.ofNat 0) after) ([.arithmetic .add] ++ recursiveTrace) := by
      exact ExecStmt.return_ (EvalExpr.call hrecursiveArgs hrecursive)
    have hbody : ExecList context .unbounded (depth + 1) environment memory initGo.body
        (.returned (Word.ofNat 0) after)
        ([.comparison .ge] ++ (slotSetTrace (depth + 1) base (Word.ofNat i) ++
          ([.arithmetic .add] ++ recursiveTrace))) := by
      rw [hinitGoShape.2]
      exact ExecList.next hif (ExecList.next hslotExpr (ExecList.returned hreturn))
    refine ⟨after, initGoStepTrace depth base i recursiveTrace, ?_, ?_, ?_, ?_, ?_⟩
    · exact EvalCall.run (policy := .unbounded) (depth := depth)
        hinitGo hframe True.intro hbody
    · intro slot hislot hslot
      by_cases hcurrent : slot = i
      · subst slot
        constructor
        · unfold keyAt
          rw [hrecursiveFrame.2 (keyCoordinate base i)
            (current_key_not_in_suffix base cap i)]
          exact hslotKey
        · unfold valueAt
          rw [hrecursiveFrame.2 (valueCoordinate base i)
            (current_value_not_in_suffix base cap i)]
          exact hslotValue
      · exact hsuffix slot (Nat.succ_le_of_lt (Nat.lt_of_le_of_ne hislot (Ne.symm hcurrent))) hslot
    · simp only [initGoStepTrace, storeAddresses, slotSetTrace, storeAddresses_append,
        ofNat_val_emptyCone i hiSign, List.nil_append]
      rw [hstores]
      unfold expectedInitStoresFrom
      rw [sub_succ_difference hlt]
      rw [expectedInitStoresFuel]
      rfl
    · constructor
      · exact hrecursiveFrame.1.trans hslotOwned
      · intro coordinate houtside
        rw [hrecursiveFrame.2 coordinate (by
          intro hs
          apply houtside
          rcases hs with ⟨slot, hslotLower, hslotUpper, hcoord⟩
          exact ⟨slot, Nat.le_trans (Nat.le_succ i) hslotLower, hslotUpper, hcoord⟩)]
        apply hslotOther coordinate
        · intro heq
          apply houtside
          exact ⟨i, le_rfl, hlt, Or.inl heq⟩
        · intro heq
          apply houtside
          exact ⟨i, le_rfl, hlt, Or.inr heq⟩
    · exact initGoStepBounds cap base i depth hlt recursiveTrace hresources

end Slang.Table
