import Slang.Table.TableInit

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

theorem listAppendAssoc {α : Type} (left middle right : List α) :
    (left ++ middle) ++ right = left ++ (middle ++ right) := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      exact congrArg (List.cons head) inductionHypothesis

theorem listAppendNil {α : Type} (left : List α) : left ++ [] = left := by
  induction left with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      exact congrArg (List.cons head) inductionHypothesis

theorem boolFalseNeTrue : (false : Bool) ≠ true := by
  intro equal
  exact Bool.noConfusion equal

def TableSlotKeyShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "slot"] ∧
    declaration.body = [.return_ (.memory slotAddressExpr)]

def TableSlotValShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "slot"] ∧
    declaration.body = [.return_ (.memory slotValueAddressExpr)]

def probeBody (callee : String) (emptyResult : Expr) : List Stmt :=
  [.if_ (.binary .ge (.name "step") (.name "cap"))
    [.return_ (.binary .sub (.literal 0) (.literal 1))] none,
   .bind "currSlot" (.binary .bitAnd
     (.binary .add (.name "slot") (.name "step")) (.name "mask")),
   .bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]),
   .if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
     [.return_ emptyResult] none,
   .if_ (.binary .eq (.name "slotKey") (.name "key"))
     [.return_ (.name "currSlot")] none,
   .return_ (.call callee
     [.name "base", .name "cap", .name "mask", .name "key", .name "slot",
      .binary .add (.name "step") (.literal 1)])]

def TableFindSlotGoShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap", "mask", "key", "slot", "step"] ∧
    declaration.body = probeBody "tableFindSlotGo" (.name "currSlot")

def TableLookupSlotGoShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap", "mask", "key", "slot", "step"] ∧
    declaration.body = probeBody "tableLookupSlotGo" (.binary .sub (.literal 0) (.literal 1))

def TableFindShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap", "key"] ∧
    declaration.body =
      [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
        [.return_ (.name "TABLE_EMPTY")] none,
       .if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
        [.return_ (.name "TABLE_EMPTY")] none,
       .bind "mask" (.binary .sub (.name "cap") (.literal 1)),
       .bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"])
         (.name "mask")),
       .return_ (.call "tableLookupSlotGo"
         [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot",
          .literal 0])]

def TableSpreadShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["k"] ∧
    declaration.body =
      [.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k"))
        (.binary .lt (.name "k") (.literal 0)) (.name "k")),
       .return_ (.binary .bitAnd
         (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903))]

theorem nat_and_power_pred_eq_mod (value exponent : Nat) :
    value &&& (2 ^ exponent - 1) = value % 2 ^ exponent := by
  exact Nat.and_two_pow_sub_one_eq_mod value exponent

theorem powTwo_mono {left right : Nat} (bound : left ≤ right) :
    powTwo left ≤ powTwo right := by
  induction right generalizing left with
  | zero =>
      have leftZero : left = 0 := Nat.eq_zero_of_le_zero bound
      subst left
      exact Nat.le_refl 1
  | succ right inductionHypothesis =>
      cases left with
      | zero => exact Nat.succ_le_of_lt (powTwo_pos (right + 1))
      | succ left =>
          unfold powTwo
          exact Nat.mul_le_mul_left 2
            (inductionHypothesis (Nat.le_of_succ_le_succ bound))

theorem powTwo_dvd_of_le {left right : Nat} (bound : left ≤ right) :
    powTwo left ∣ powTwo right := by
  induction right generalizing left with
  | zero =>
      have leftZero : left = 0 := Nat.eq_zero_of_le_zero bound
      subst left
      exact ⟨1, rfl⟩
  | succ right inductionHypothesis =>
      cases left with
      | zero => exact ⟨powTwo (right + 1), Nat.one_mul _ |>.symm⟩
      | succ left =>
          rcases inductionHypothesis (Nat.le_of_succ_le_succ bound) with ⟨factor, factorEq⟩
          refine ⟨factor, ?_⟩
          unfold powTwo
          rw [factorEq]
          exact (mulAssoc 2 (powTwo left) factor).symm

theorem powTwo_dvd_wordModulus (exponent : Nat) (bound : exponent ≤ wordBits) :
    powTwo exponent ∣ wordModulus := by
  change powTwo exponent ∣ powTwo wordBits
  exact powTwo_dvd_of_le bound

theorem mask_val (cap : Word) (hcap : PositivePowerOfTwo cap)
    (_hcapSign : cap.val < wordSignBit) :
    (Word.sub cap (Word.ofNat 1)).val = cap.val - 1 := by
  exact word_sub_one_val cap hcap.pos

theorem bitwise_and_right_zero (fuel value : Nat) :
    Word.bitwise Bool.and fuel value 0 = 0 := by
  induction fuel generalizing value with
  | zero => rfl
  | succ fuel inductionHypothesis =>
      unfold Word.bitwise
      rw [inductionHypothesis]
      change (if (value % 2 == 1 && false) then 1 else 0) = 0
      cases value % 2 == 1 <;> rfl

theorem oddDoubleModTwo (value : Nat) : (value + value + 1) % 2 = 1 := by
  induction value with
  | zero => rfl
  | succ value inductionHypothesis =>
      rw [show (value + 1) + (value + 1) + 1 = (value + value + 1) + 2 by
        rw [Nat.add_succ, Nat.succ_add, Nat.add_succ]]
      rw [modAddModulus, inductionHypothesis]

theorem double_sub_of_le (left right : Nat) (bound : left ≤ right) :
    2 * right - 2 * left = 2 * (right - left) := by
  have decomposition : right = left + (right - left) := (addSubOfLe bound).symm
  calc
    2 * right - 2 * left = 2 * (left + (right - left)) - 2 * left :=
      congrArg (fun value => 2 * value - 2 * left) decomposition
    _ = 2 * (right - left) := by
      rw [Nat.mul_add, Nat.add_comm, addSubCancel]

theorem double_lt_double {left right : Nat} (bound : left < right) :
    2 * left < 2 * right := by
  rw [Nat.two_mul, Nat.two_mul]
  exact Nat.add_lt_add bound bound

theorem double_add_one_sub_of_le (left right : Nat) (bound : left ≤ right) :
    2 * right + 1 - 2 * left = 2 * (right - left) + 1 := by
  have decomposition : right = left + (right - left) := (addSubOfLe bound).symm
  calc
    2 * right + 1 - 2 * left =
        2 * (left + (right - left)) + 1 - 2 * left :=
      congrArg (fun value => 2 * value + 1 - 2 * left) decomposition
    _ = 2 * (right - left) + 1 := by
      rw [Nat.mul_add]
      rw [show 2 * left + 2 * (right - left) + 1 =
          (2 * (right - left) + 1) + 2 * left by
        exact (Nat.add_assoc _ _ _).trans (Nat.add_comm _ _)]
      rw [addSubCancel]

theorem double_mod_double (value modulus : Nat) (positive : 0 < modulus) :
    (2 * value) % (2 * modulus) = 2 * (value % modulus) := by
  induction value using Nat.strong_induction_on with
  | h value inductionHypothesis =>
      by_cases below : value < modulus
      · rw [modEqOfLt below]
        exact modEqOfLt (double_lt_double below)
      · have lower : modulus ≤ value := Nat.le_of_not_gt below
        rw [modEqSub positive lower]
        rw [modEqSub (Nat.mul_pos (by decide) positive) (Nat.mul_le_mul_left 2 lower)]
        rw [double_sub_of_le modulus value lower]
        exact inductionHypothesis (value - modulus) (Nat.sub_lt_self positive lower)

theorem oddDouble_mod_double (value modulus : Nat) (positive : 0 < modulus) :
    (2 * value + 1) % (2 * modulus) = 2 * (value % modulus) + 1 := by
  induction value using Nat.strong_induction_on with
  | h value inductionHypothesis =>
      by_cases below : value < modulus
      · rw [modEqOfLt below]
        apply modEqOfLt
        have doubled : 2 * value + 2 ≤ 2 * modulus := by
          simpa only [Nat.mul_succ] using
            Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt below)
        exact Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * value + 1)) doubled
      · have lower : modulus ≤ value := Nat.le_of_not_gt below
        rw [modEqSub positive lower]
        rw [modEqSub (Nat.mul_pos (by decide) positive) (by
          exact Nat.le_trans (Nat.mul_le_mul_left 2 lower) (Nat.le_add_right _ 1))]
        rw [double_add_one_sub_of_le modulus value lower]
        exact inductionHypothesis (value - modulus) (Nat.sub_lt_self positive lower)

theorem evenOrOdd (value : Nat) :
    ∃ half, value = 2 * half ∨ value = 2 * half + 1 := by
  induction value with
  | zero => exact ⟨0, Or.inl rfl⟩
  | succ value inductionHypothesis =>
      rcases inductionHypothesis with ⟨half, even | odd⟩
      · exact ⟨half, Or.inr (congrArg Nat.succ even)⟩
      · refine ⟨half + 1, Or.inl ?_⟩
        rw [odd, Nat.mul_succ]

theorem modDouble_reconstruct (value modulus : Nat) (positive : 0 < modulus) :
    2 * (value / 2 % modulus) + value % 2 = value % (2 * modulus) := by
  rcases evenOrOdd value with ⟨half, even | odd⟩
  · rw [even]
    have halfDiv : 2 * half / 2 = half := by
      rw [Nat.two_mul]
      exact evenDoubleDivTwo half
    have halfMod : 2 * half % 2 = 0 := by
      rw [Nat.two_mul]
      exact evenDoubleModTwo half
    rw [halfDiv, halfMod, Nat.add_zero, double_mod_double half modulus positive]
  · rw [odd]
    have halfDiv : (2 * half + 1) / 2 = half := by
      rw [Nat.two_mul]
      exact oddDoubleDivTwo half
    have halfMod : (2 * half + 1) % 2 = 1 := by
      rw [Nat.two_mul]
      exact oddDoubleModTwo half
    rw [halfDiv, halfMod, oddDouble_mod_double half modulus positive]

theorem natBeq_zero_one : Nat.beq 0 1 = false := rfl

theorem natBeq_one_one : Nat.beq 1 1 = true := rfl

theorem natBEq_zero_one : ((0 : Nat) == 1) = false := rfl

theorem natBEq_one_one : ((1 : Nat) == 1) = true := rfl

theorem bitwiseLowFalse (higher : Nat) :
    (let low := ((0 : Nat) == 1) && ((1 : Nat) == 1)
     if low then higher + higher + 1 else higher + higher) = higher + higher := rfl

theorem bitwiseLowTrue (higher : Nat) :
    (let low := ((1 : Nat) == 1) && ((1 : Nat) == 1)
     if low then higher + higher + 1 else higher + higher) = higher + higher + 1 := rfl

theorem modTwoCases (value : Nat) : value % 2 = 0 ∨ value % 2 = 1 := by
  have less : value % 2 < 2 := Nat.mod_lt value (by decide)
  cases equal : value % 2 with
  | zero => exact Or.inl rfl
  | succ remainder =>
      cases remainder with
      | zero => exact Or.inr rfl
      | succ remainder =>
          rw [equal] at less
          exact False.elim (Nat.not_lt_zero remainder
            (Nat.lt_of_succ_lt_succ (Nat.lt_of_succ_lt_succ less)))

theorem powTwoPredHalf (exponent : Nat) :
    (2 * powTwo exponent - 1) / 2 = powTwo exponent - 1 :=
  doublePredDivTwo (powTwo exponent) (powTwo_pos exponent)

theorem powTwoPredLow (exponent : Nat) : (2 * powTwo exponent - 1) % 2 = 1 := by
  cases powerEq : powTwo exponent with
  | zero =>
      have positive := powTwo_pos exponent
      rw [powerEq] at positive
      exact False.elim (Nat.not_lt_zero 0 positive)
  | succ predecessor =>
      rw [Nat.mul_succ, Nat.succ_sub_one, Nat.two_mul]
      exact oddDoubleModTwo predecessor

theorem doublePredModTwo (modulus : Nat) (positive : 0 < modulus) :
    (2 * modulus - 1) % 2 = 1 := by
  cases modulusEq : modulus with
  | zero =>
      rw [modulusEq] at positive
      exact False.elim (Nat.not_lt_zero 0 positive)
  | succ predecessor =>
      rw [Nat.mul_succ, Nat.succ_sub_one, Nat.two_mul]
      exact oddDoubleModTwo predecessor

theorem bitwiseAndStep (fuel value modulus : Nat) (positive : 0 < modulus)
    (recursive : Word.bitwise Bool.and fuel (value / 2) (modulus - 1) =
      value / 2 % modulus) :
    Word.bitwise Bool.and (fuel + 1) value (2 * modulus - 1) =
      value % (2 * modulus) := by
  unfold Word.bitwise
  rw [doublePredDivTwo modulus positive, doublePredModTwo modulus positive, recursive]
  rcases modTwoCases value with low | low
  · rw [low, bitwiseLowFalse]
    have result := modDouble_reconstruct value modulus positive
    rw [low, Nat.add_zero] at result
    rw [← Nat.two_mul]
    exact result
  · rw [low, bitwiseLowTrue]
    have result := modDouble_reconstruct value modulus positive
    rw [low] at result
    rw [← Nat.two_mul]
    exact result

theorem bitwise_and_powTwo_pred (fuel exponent value : Nat) (bound : exponent ≤ fuel) :
    Word.bitwise Bool.and fuel value (powTwo exponent - 1) = value % powTwo exponent := by
  induction fuel generalizing exponent value with
  | zero =>
      have exponentZero : exponent = 0 := Nat.eq_zero_of_le_zero bound
      subst exponent
      unfold Word.bitwise powTwo
      exact Nat.mod_one value |>.symm
  | succ fuel inductionHypothesis =>
      cases exponent with
      | zero =>
          unfold powTwo
          rw [Nat.mod_one]
          exact bitwise_and_right_zero (fuel + 1) value
      | succ exponent =>
          have exponentBound : exponent ≤ fuel := Nat.le_of_succ_le_succ bound
          have positive := powTwo_pos exponent
          exact bitwiseAndStep fuel value (powTwo exponent) positive
            (inductionHypothesis exponent (value / 2) exponentBound)

theorem modSelf (modulus : Nat) (positive : 0 < modulus) : modulus % modulus = 0 := by
  rw [modEqSub positive (Nat.le_refl modulus), Nat.sub_self]
  exact modEqOfLt positive

theorem modAddMultiple (value modulus factor : Nat) :
    (value + modulus * factor) % modulus = value % modulus := by
  induction factor with
  | zero => rw [Nat.mul_zero, Nat.add_zero]
  | succ factor inductionHypothesis =>
      rw [Nat.mul_succ, ← Nat.add_assoc, modAddModulus, inductionHypothesis]

theorem modModOfMultiple (value small large factor : Nat)
    (largePositive : 0 < large)
    (factorEq : large = small * factor) : value % large % small = value % small := by
  induction value using Nat.strong_induction_on with
  | h value inductionHypothesis =>
      by_cases below : value < large
      · rw [modEqOfLt below]
      · have lower : large ≤ value := Nat.le_of_not_gt below
        rw [modEqSub largePositive lower]
        rw [inductionHypothesis (value - large) (Nat.sub_lt_self largePositive lower)]
        have decomposition : large + (value - large) = value := addSubOfLe lower
        calc
          (value - large) % small = ((value - large) + small * factor) % small :=
            (modAddMultiple (value - large) small factor).symm
          _ = (large + (value - large)) % small := by
            rw [factorEq, Nat.add_comm]
          _ = value % small := congrArg (fun number => number % small) decomposition

theorem source_probe_slot_eq (cap key : Word) (step : Nat)
    (hcap : PositivePowerOfTwo cap) (hcapSign : cap.val < wordSignBit)
    (hstep : step < cap.val) :
    (Word.bitAnd
      (Word.add (Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1)))
        (Word.ofNat step))
      (Word.sub cap (Word.ofNat 1))).val = probeSlot cap key step := by
  rcases hcap with ⟨exponent, hexponent⟩
  have hpositive : 0 < cap.val := hexponent ▸ powTwo_pos exponent
  have hmask := mask_val cap ⟨exponent, hexponent⟩ hcapSign
  have exponentBound : exponent ≤ wordBits := by
    by_contra notBound
    have bitSucc : wordBits + 1 ≤ exponent := Nat.lt_of_not_ge notBound
    have powerBound := powTwo_mono bitSucc
    have modulusLe : wordModulus ≤ cap.val := by
      change powTwo wordBits ≤ cap.val
      rw [hexponent]
      exact Nat.le_trans (powTwo_mono (Nat.le_succ wordBits)) powerBound
    exact Nat.not_lt_of_ge modulusLe (Nat.lt_trans hcapSign wordSignBit_lt_modulus)
  have hhome :
      (Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1))).val =
        (Word.tableSpread key).val % cap.val := by
    unfold Word.bitAnd Word.ofNat
    change Word.bitwise Bool.and wordBits (Word.tableSpread key).val
        (Word.sub cap (Word.ofNat 1)).val % wordModulus = _
    rw [hmask, hexponent, bitwise_and_powTwo_pred wordBits exponent _ exponentBound]
    exact modEqOfLt
      (Nat.lt_trans (Nat.mod_lt _ (powTwo_pos exponent))
        (Nat.lt_trans (hexponent ▸ hcapSign) wordSignBit_lt_modulus))
  unfold Word.bitAnd Word.add Word.ofNat probeSlot
  change Word.bitwise Bool.and wordBits
      (((Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1))).val +
        step % wordModulus) % wordModulus)
      (Word.sub cap (Word.ofNat 1)).val % wordModulus =
    ((Word.tableSpread key).val + step) % cap.val
  rw [modEqOfLt (Nat.lt_trans hstep (Nat.lt_trans hcapSign wordSignBit_lt_modulus)),
    hmask, hexponent, bitwise_and_powTwo_pred wordBits exponent _ exponentBound]
  rw [modEqOfLt (Nat.lt_trans (Nat.mod_lt _ (powTwo_pos exponent))
    (Nat.lt_trans (hexponent ▸ hcapSign) wordSignBit_lt_modulus))]
  rcases powTwo_dvd_wordModulus exponent exponentBound with ⟨factor, factorEq⟩
  rw [modModOfMultiple _ (powTwo exponent) wordModulus factor wordModulus_pos factorEq]
  rw [hhome, ← hexponent, modAddAbsorbLeft]

theorem tableSlotKey_execution (context : Context) (slotKey : FunctionDecl)
    (hfunction : context.functions "tableSlotKey" = some slotKey)
    (hshape : TableSlotKeyShape slotKey)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (memory : Memory) (base slot : Word) (depth : Nat) (raw : Word)
    (hcoordinate : keyCoordinate base slot.val < wordSignBit)
    (howned : memory.owned (keyCoordinate base slot.val) = true)
    (hvalue : keyAt memory base slot.val = some raw) :
    EvalCall context .unbounded depth "tableSlotKey" [base, slot] memory
      (.value raw memory)
      [.call "tableSlotKey" (depth + 1), .arithmetic .mul, .arithmetic .add,
       .load (Word.ofNat (keyCoordinate base slot.val))] := by
  let environment : Environment := [("base", base), ("slot", slot)]
  have hframe : Environment.ofParameters slotKey.parameters [base, slot] = some environment := by
    rw [hshape.1]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    rfl
  have hslot : Environment.lookup environment "slot" = some slot := by
    rfl
  have hlocal : Environment.lookup environment "TABLE_STRIDE" = none := by
    rfl
  have haddress := eval_slot_address context .unbounded (depth + 1) environment memory base slot
    hbase hslot hlocal hstride
  have hload : memory.load (Word.ofNat (keyCoordinate base slot.val)) = some raw := by
    unfold Memory.load
    rw [coordinate_ofNat_emptyCone _ hcoordinate]
    change (if memory.owned (keyCoordinate base slot.val) then
      memory.initialized (keyCoordinate base slot.val) else none) = some raw
    rw [howned]
    exact hvalue
  apply EvalCall.run (policy := .unbounded) (depth := depth) hfunction hframe True.intro
  rw [hshape.2]
  exact ExecList.returned (ExecStmt.return_ (EvalExpr.load haddress hload))

inductive ProbeOutcome (emptyReturnsSlot : Bool) (memory : Memory) (base cap key : Word) :
    Nat → Word → Prop where
  | exhausted (step : Nat) (hstep : step = cap.val) :
      ProbeOutcome emptyReturnsSlot memory base cap key step tableEmpty
  | empty (step : Nat) (hstep : step < cap.val)
      (hempty : keyAt memory base (probeSlot cap key step) = some tableEmpty) :
      ProbeOutcome emptyReturnsSlot memory base cap key step
        (if emptyReturnsSlot then Word.ofNat (probeSlot cap key step) else tableEmpty)
  | match (step : Nat) (hstep : step < cap.val)
      (hmatch : keyAt memory base (probeSlot cap key step) = some key) :
      ProbeOutcome emptyReturnsSlot memory base cap key step
        (Word.ofNat (probeSlot cap key step))
  | next (step : Nat) (raw : Word) (hstep : step < cap.val)
      (hneEmpty : keyAt memory base (probeSlot cap key step) ≠ some tableEmpty)
      (hneKey : keyAt memory base (probeSlot cap key step) ≠ some key)
      (later : ProbeOutcome emptyReturnsSlot memory base cap key (step + 1) raw) :
      ProbeOutcome emptyReturnsSlot memory base cap key step raw

def probeEmptyExpr : Bool → Expr
  | true => .name "currSlot"
  | false => .binary .sub (.literal 0) (.literal 1)

def probeName : Bool → String
  | true => "tableFindSlotGo"
  | false => "tableLookupSlotGo"

theorem tableEmpty_sub : Word.sub (Word.ofNat 0) (Word.ofNat 1) = tableEmpty := by
  apply Fin.ext
  change (wordModulus - 1) % wordModulus = (wordModulus - 1) % wordModulus
  rfl

theorem wordOne_nonzero : Word.nonzero (Word.ofNat 1) := by
  intro equal
  exact Nat.noConfusion (congrArg Fin.val equal)

theorem natBeq_self (value : Nat) : Nat.beq value value = true := by
  induction value with
  | zero => rfl
  | succ value inductionHypothesis => exact inductionHypothesis

theorem natBeq_eq {left right : Nat} (equal : Nat.beq left right = true) : left = right := by
  induction left generalizing right with
  | zero =>
      cases right with
      | zero => rfl
      | succ right => contradiction
  | succ left inductionHypothesis =>
      cases right with
      | zero => contradiction
      | succ right => exact congrArg Nat.succ (inductionHypothesis equal)

theorem wordEq_self (value : Word) : Word.eq value value = Word.ofNat 1 := by
  unfold Word.eq Word.bool
  change (if Nat.beq value.val value.val then Word.ofNat 1 else Word.ofNat 0) = Word.ofNat 1
  rw [natBeq_self]
  rfl

theorem wordEq_ne (left right : Word) (unequal : left ≠ right) :
    Word.eq left right = Word.ofNat 0 := by
  unfold Word.eq Word.bool
  change (if Nat.beq left.val right.val then Word.ofNat 1 else Word.ofNat 0) = Word.ofNat 0
  cases result : Nat.beq left.val right.val with
  | false => rfl
  | true =>
      have valuesEqual : left.val = right.val := natBeq_eq result
      exact False.elim (unequal (Fin.ext valuesEqual))

theorem wordLt_true (left right : Word) (less : Word.signed left < Word.signed right) :
    Word.lt left right = Word.ofNat 1 := by
  unfold Word.lt Word.bool
  rw [decide_eq_true less]
  rfl

theorem wordLt_false (left right : Word) (notLess : ¬Word.signed left < Word.signed right) :
    Word.lt left right = Word.ofNat 0 := by
  unfold Word.lt Word.bool
  rw [decide_eq_false notLess]
  rfl

theorem probe_execution (emptyReturnsSlot : Bool) (context : Context)
    (slotKey probe : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hprobe : context.functions (probeName emptyReturnsSlot) = some probe)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hprobeParameters : probe.parameters =
      ["base", "cap", "mask", "key", "slot", "step"])
    (hprobeBody : probe.body =
      probeBody (probeName emptyReturnsSlot) (probeEmptyExpr emptyReturnsSlot))
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hemptyConstant : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (step depth : Nat)
    (rep : RepSong memory base cap default occupied h) (hstep : step ≤ cap.val) :
    ∃ raw trace,
      EvalCall context .unbounded depth (probeName emptyReturnsSlot)
        [base, cap, Word.sub cap (Word.ofNat 1), key,
          Word.bitAnd (Word.tableSpread key) (Word.sub cap (Word.ofNat 1)), Word.ofNat step]
        memory (.value raw memory) trace ∧
       ProbeOutcome emptyReturnsSlot memory base cap key step raw := by
  generalize measureEq : cap.val - step = measure
  revert step depth
  refine Nat.strong_induction_on measure ?_
  intro measure inductionHypothesis step depth hstep measureEq
  have hcap := rep.positivePowerOfTwo
  have hbacking := rep.validExtent.1
  have hcapSign : cap.val < wordSignBit := by
    exact cap_below_signBit_of_backing memory base cap hbacking
  have hstepSign : step < wordSignBit := lt_of_le_of_lt hstep hcapSign
  let mask := Word.sub cap (Word.ofNat 1)
  let home := Word.bitAnd (Word.tableSpread key) mask
  let stepWord := Word.ofNat step
  let environment : Environment :=
    [("base", base), ("cap", cap), ("mask", mask), ("key", key), ("slot", home),
      ("step", stepWord)]
  have hframe : Environment.ofParameters probe.parameters
      [base, cap, mask, key, home, stepWord] = some environment := by
    rw [hprobeParameters]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    rfl
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    rfl
  have hmaskLocal : Environment.lookup environment "mask" = some mask := by
    rfl
  have hkeyLocal : Environment.lookup environment "key" = some key := by
    rfl
  have hhomeLocal : Environment.lookup environment "slot" = some home := by
    rfl
  have hstepLocal : Environment.lookup environment "step" = some stepWord := by
    rfl
  have hgeEval : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .ge (.name "step") (.name "cap"))
      (.value (Word.ge stepWord cap) memory) [.comparison .ge] := by
    exact EvalExpr.binary (operator := .ge) (EvalExpr.local hstepLocal)
      (EvalExpr.local hcapLocal)
  by_cases hdone : step = cap.val
  · have hcapWord : Word.ofNat step = cap := by
      apply Fin.ext
      rw [hdone]
      exact ofNat_val_emptyCone cap.val hcapSign
    have hgeOne : Word.ge stepWord cap = Word.ofNat 1 := by
      exact Eq.mp (congrArg (fun right =>
        Word.ge (Word.ofNat step) right = Word.ofNat 1) hcapWord)
        (wordGe_ofNat_self step hstepSign)
    have hnonzero : Word.nonzero (Word.ge stepWord cap) := by
      rw [hgeOne]
      exact wordOne_nonzero
    have hguard : ExecStmt context .unbounded (depth + 1) environment memory
        (.if_ (.binary .ge (.name "step") (.name "cap"))
          [.return_ (.binary .sub (.literal 0) (.literal 1))] none)
        (.returned tableEmpty memory) [.comparison .ge, .arithmetic .sub] := by
      apply ExecStmt.ifTrue hgeEval hnonzero
      apply ExecList.returned
      apply ExecStmt.return_
      rw [← tableEmpty_sub]
      exact EvalExpr.binary (operator := .sub) (EvalExpr.literal (literalValue := 0))
        (EvalExpr.literal (literalValue := 1))
    have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
        (.returned tableEmpty memory) [.comparison .ge, .arithmetic .sub] := by
      rw [hprobeBody, probeBody]
      exact ExecList.returned hguard
    refine ⟨tableEmpty,
      [.call (probeName emptyReturnsSlot) (depth + 1), .comparison .ge, .arithmetic .sub],
      ?_, .exhausted step hdone⟩
    exact EvalCall.run hprobe hframe True.intro hbody
  · have hlt : step < cap.val := Nat.lt_of_le_of_ne hstep hdone
    have hgeZero : Word.ge stepWord cap = Word.ofNat 0 := by
      have capWord : Word.ofNat cap.val = cap := Fin.ext (ofNat_val_emptyCone _ hcapSign)
      rw [← capWord]
      exact wordGe_ofNat_lt step cap.val hstepSign hcapSign hlt
    have hguardFalse : ¬ Word.nonzero (Word.ge stepWord cap) := by
      rw [hgeZero]
      exact fun hzero => hzero rfl
    have hguard : ExecStmt context .unbounded (depth + 1) environment memory
        (.if_ (.binary .ge (.name "step") (.name "cap"))
          [.return_ (.binary .sub (.literal 0) (.literal 1))] none)
        (.next environment memory) [.comparison .ge] := by
      exact ExecStmt.ifFalse hgeEval hguardFalse ExecList.nil
    let currNat := probeSlot cap key step
    let curr := Word.ofNat currNat
    have hcurrLt : currNat < cap.val := Nat.mod_lt _ hcap.pos
    have hcurrSign : currNat < wordSignBit := lt_trans hcurrLt hcapSign
    have hcurrEq : Word.bitAnd (Word.add home stepWord) mask = curr := by
      apply Fin.ext
      rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
      exact source_probe_slot_eq cap key step hcap hcapSign hlt
    have hcurrEval : EvalExpr context .unbounded (depth + 1) environment memory
        (.binary .bitAnd (.binary .add (.name "slot") (.name "step")) (.name "mask"))
        (.value curr memory) [.arithmetic .add, .bitwise .bitAnd] := by
      rw [← hcurrEq]
      exact EvalExpr.binary (operator := .bitAnd)
        (EvalExpr.binary (operator := .add) (EvalExpr.local hhomeLocal)
          (EvalExpr.local hstepLocal)) (EvalExpr.local hmaskLocal)
    let currEnvironment := Environment.bind environment "currSlot" curr
    have hcurrBind : ExecStmt context .unbounded (depth + 1) environment memory
        (.bind "currSlot"
          (.binary .bitAnd (.binary .add (.name "slot") (.name "step")) (.name "mask")))
        (.next currEnvironment memory) [.arithmetic .add, .bitwise .bitAnd] := by
      exact ExecStmt.bind hcurrEval
    have hcurrLocal : Environment.lookup currEnvironment "currSlot" = some curr := by
      exact Environment.lookup_bind_same _ _ _
    have hbaseCurr : Environment.lookup currEnvironment "base" = some base := by
      rw [Environment.lookup_bind_other _ "currSlot" "base" curr (by decide)]
      exact hbase
    rcases rep.2.2.2.1 currNat hcurrLt with hslotEmpty | ⟨slotValue, hmember, hslotValue⟩
    · let slotWord := tableEmpty
      have hslotCall := tableSlotKey_execution context slotKey hslotKey hslotKeyShape hstride
        memory base curr (depth + 1) slotWord
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact keyCoordinate_below_signBit_emptyCone base cap currNat hbacking.2.1 hcurrLt)
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact rep.owned_extent (2 * currNat)
            (double_lt_double hcurrLt))
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact hslotEmpty)
      let slotEnvironment := Environment.bind currEnvironment "slotKey" slotWord
      have hargs : EvalArgs context .unbounded (depth + 1) currEnvironment memory
          [.name "base", .name "currSlot"] (.value [base, curr] memory) [] := by
        exact EvalArgs.cons (EvalExpr.local hbaseCurr)
          (EvalArgs.cons (EvalExpr.local hcurrLocal) EvalArgs.nil)
      have hslotBind : ExecStmt context .unbounded (depth + 1) currEnvironment memory
          (.bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]))
          (.next slotEnvironment memory)
          [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
            .load (Word.ofNat (keyCoordinate base curr.val))] := by
        apply ExecStmt.bind
        exact EvalExpr.call hargs hslotCall
      have hslotLocal : Environment.lookup slotEnvironment "slotKey" = some tableEmpty := by
        exact Environment.lookup_bind_same _ _ _
      have hemptyLocal : Environment.lookup slotEnvironment "TABLE_EMPTY" = none := by
        rfl
      have heqOne : Word.eq tableEmpty tableEmpty = Word.ofNat 1 := by
        exact wordEq_self tableEmpty
      have heqEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
          (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
          (.value (Word.ofNat 1) memory) [.comparison .eq] := by
        rw [← heqOne]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
          (EvalExpr.constant hemptyLocal hemptyConstant)
      have hone : Word.nonzero (Word.ofNat 1) := by
        exact wordOne_nonzero
      cases emptyReturnsSlot with
      | false =>
          have hemptyIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
              (.if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
                [.return_ (.binary .sub (.literal 0) (.literal 1))] none)
              (.returned tableEmpty memory) [.comparison .eq, .arithmetic .sub] := by
            apply ExecStmt.ifTrue heqEval hone
            apply ExecList.returned
            apply ExecStmt.return_
            rw [← tableEmpty_sub]
            exact EvalExpr.binary (operator := .sub)
              (EvalExpr.literal (literalValue := 0))
              (EvalExpr.literal (literalValue := 1))
          have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
              (.returned tableEmpty memory)
              ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
                [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                  .load (Word.ofNat (keyCoordinate base curr.val))] ++
                [.comparison .eq, .arithmetic .sub]) := by
            rw [hprobeBody]
            simp only [probeName, probeEmptyExpr, probeBody]
            simpa only [listAppendAssoc] using ExecList.next hguard
              (ExecList.next hcurrBind (ExecList.next hslotBind (ExecList.returned hemptyIf)))
          let trace : Trace := .call "tableLookupSlotGo" (depth + 1) ::
            ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
              [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                .load (Word.ofNat (keyCoordinate base curr.val))] ++
              [.comparison .eq, .arithmetic .sub])
          refine ⟨tableEmpty, trace, ?_, ?_⟩
          · simpa only [trace, mask, home, stepWord, probeName] using
              EvalCall.run (policy := .unbounded) (depth := depth)
                hprobe hframe True.intro hbody
          · exact ProbeOutcome.empty (emptyReturnsSlot := false) step hlt hslotEmpty
      | true =>
          have hcurrSlotIn : Environment.lookup slotEnvironment "currSlot" = some curr := by
            rw [Environment.lookup_bind_other _ "slotKey" "currSlot" slotWord (by decide)]
            exact hcurrLocal
          have hemptyIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
              (.if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
                [.return_ (.name "currSlot")] none)
              (.returned curr memory) [.comparison .eq] := by
            apply ExecStmt.ifTrue heqEval hone
            exact ExecList.returned (ExecStmt.return_ (EvalExpr.local hcurrSlotIn))
          have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
              (.returned curr memory)
              ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
                [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                  .load (Word.ofNat (keyCoordinate base curr.val))] ++ [.comparison .eq]) := by
            rw [hprobeBody]
            simp only [probeName, probeEmptyExpr, probeBody]
            simpa only [listAppendAssoc] using ExecList.next hguard
              (ExecList.next hcurrBind (ExecList.next hslotBind (ExecList.returned hemptyIf)))
          let trace : Trace := .call "tableFindSlotGo" (depth + 1) ::
            ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
              [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                .load (Word.ofNat (keyCoordinate base curr.val))] ++ [.comparison .eq])
          refine ⟨curr, trace, ?_, ?_⟩
          · simpa only [trace, mask, home, stepWord, probeName] using
              EvalCall.run (policy := .unbounded) (depth := depth)
                hprobe hframe True.intro hbody
          · change ProbeOutcome true memory base cap key step
              (Word.ofNat (probeSlot cap key step))
            exact ProbeOutcome.empty (emptyReturnsSlot := true) step hlt hslotEmpty
    · have hslotCall := tableSlotKey_execution context slotKey hslotKey hslotKeyShape hstride
        memory base curr (depth + 1) slotValue
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact keyCoordinate_below_signBit_emptyCone base cap currNat hbacking.2.1 hcurrLt)
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact rep.owned_extent (2 * currNat)
            (double_lt_double hcurrLt))
        (by
          rw [show curr.val = currNat by exact ofNat_val_emptyCone currNat hcurrSign]
          exact hslotValue)
      let slotEnvironment := Environment.bind currEnvironment "slotKey" slotValue
      have hargs : EvalArgs context .unbounded (depth + 1) currEnvironment memory
          [.name "base", .name "currSlot"] (.value [base, curr] memory) [] := by
        exact EvalArgs.cons (EvalExpr.local hbaseCurr)
          (EvalArgs.cons (EvalExpr.local hcurrLocal) EvalArgs.nil)
      have hslotBind : ExecStmt context .unbounded (depth + 1) currEnvironment memory
          (.bind "slotKey" (.call "tableSlotKey" [.name "base", .name "currSlot"]))
          (.next slotEnvironment memory)
          [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
            .load (Word.ofNat (keyCoordinate base curr.val))] := by
        apply ExecStmt.bind
        exact EvalExpr.call hargs hslotCall
      have hslotLocal : Environment.lookup slotEnvironment "slotKey" = some slotValue := by
        exact Environment.lookup_bind_same _ _ _
      have hkeyIn : Environment.lookup slotEnvironment "key" = some key := by
        rw [Environment.lookup_bind_other _ "slotKey" "key" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "key" curr (by decide)]
        exact hkeyLocal
      have hbaseIn : Environment.lookup slotEnvironment "base" = some base := by
        rw [Environment.lookup_bind_other _ "slotKey" "base" slotValue (by decide)]
        exact hbaseCurr
      have hcapIn : Environment.lookup slotEnvironment "cap" = some cap := by
        rw [Environment.lookup_bind_other _ "slotKey" "cap" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "cap" curr (by decide)]
        exact hcapLocal
      have hmaskIn : Environment.lookup slotEnvironment "mask" = some mask := by
        rw [Environment.lookup_bind_other _ "slotKey" "mask" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "mask" curr (by decide)]
        exact hmaskLocal
      have hslotIn : Environment.lookup slotEnvironment "slot" = some home := by
        rw [Environment.lookup_bind_other _ "slotKey" "slot" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "slot" curr (by decide)]
        exact hhomeLocal
      have hstepIn : Environment.lookup slotEnvironment "step" = some stepWord := by
        rw [Environment.lookup_bind_other _ "slotKey" "step" slotValue (by decide)]
        rw [Environment.lookup_bind_other _ "currSlot" "step" curr (by decide)]
        exact hstepLocal
      have hcurrIn : Environment.lookup slotEnvironment "currSlot" = some curr := by
        rw [Environment.lookup_bind_other _ "slotKey" "currSlot" slotValue (by decide)]
        exact hcurrLocal
      have hemptyLocal : Environment.lookup slotEnvironment "TABLE_EMPTY" = none := by
        rfl
      have hneEmpty : slotValue ≠ tableEmpty := rep.2.2.1.2 slotValue hmember
      have heqEmptyZero : Word.eq slotValue tableEmpty = Word.ofNat 0 := by
        exact wordEq_ne slotValue tableEmpty hneEmpty
      have heqEmptyEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
          (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
          (.value (Word.ofNat 0) memory) [.comparison .eq] := by
        rw [← heqEmptyZero]
        exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
          (EvalExpr.constant hemptyLocal hemptyConstant)
      have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hzero => hzero rfl
      have hemptyIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
          (.if_ (.binary .eq (.name "slotKey") (.name "TABLE_EMPTY"))
            [.return_ (probeEmptyExpr emptyReturnsSlot)] none)
          (.next slotEnvironment memory) [.comparison .eq] := by
        exact ExecStmt.ifFalse heqEmptyEval hzero ExecList.nil
      by_cases hmatch : slotValue = key
      · subst slotValue
        have heqKeyOne : Word.eq key key = Word.ofNat 1 := wordEq_self key
        have heqKeyEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
            (.binary .eq (.name "slotKey") (.name "key"))
            (.value (Word.ofNat 1) memory) [.comparison .eq] := by
          rw [← heqKeyOne]
          exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
            (EvalExpr.local hkeyIn)
        have hone : Word.nonzero (Word.ofNat 1) := by
          exact wordOne_nonzero
        have hmatchIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
            (.if_ (.binary .eq (.name "slotKey") (.name "key"))
              [.return_ (.name "currSlot")] none)
            (.returned curr memory) [.comparison .eq] := by
          apply ExecStmt.ifTrue heqKeyEval hone
          exact ExecList.returned (ExecStmt.return_ (EvalExpr.local hcurrIn))
        have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
            (.returned curr memory)
            ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
              [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                .load (Word.ofNat (keyCoordinate base curr.val))] ++
              [.comparison .eq] ++ [.comparison .eq]) := by
          rw [hprobeBody]
          unfold probeBody
          simpa only [listAppendAssoc] using ExecList.next hguard
            (ExecList.next hcurrBind (ExecList.next hslotBind
              (ExecList.next hemptyIf (ExecList.returned hmatchIf))))
        let trace : Trace := .call (probeName emptyReturnsSlot) (depth + 1) ::
          ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
            [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
              .load (Word.ofNat (keyCoordinate base curr.val))] ++
            [.comparison .eq] ++ [.comparison .eq])
        refine ⟨curr, trace, ?_, ?_⟩
        · simpa only [trace, mask, home, stepWord] using
            EvalCall.run (policy := .unbounded) (depth := depth)
              hprobe hframe True.intro hbody
        · change ProbeOutcome emptyReturnsSlot memory base cap key step
            (Word.ofNat (probeSlot cap key step))
          exact .match step hlt hslotValue
      · have heqKeyZero : Word.eq slotValue key = Word.ofNat 0 := by
          exact wordEq_ne slotValue key hmatch
        have heqKeyEval : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
            (.binary .eq (.name "slotKey") (.name "key"))
            (.value (Word.ofNat 0) memory) [.comparison .eq] := by
          rw [← heqKeyZero]
          exact EvalExpr.binary (operator := .eq) (EvalExpr.local hslotLocal)
            (EvalExpr.local hkeyIn)
        have hmatchIf : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
            (.if_ (.binary .eq (.name "slotKey") (.name "key"))
              [.return_ (.name "currSlot")] none)
            (.next slotEnvironment memory) [.comparison .eq] := by
          exact ExecStmt.ifFalse heqKeyEval hzero ExecList.nil
        have measureDecrease : cap.val - (step + 1) < measure := by
          rw [← measureEq, sub_succ_difference hlt]
          exact Nat.lt_succ_self _
        rcases inductionHypothesis (cap.val - (step + 1)) measureDecrease
          (step + 1) (depth + 1) (Nat.succ_le_of_lt hlt) rfl with
          ⟨raw, recursiveTrace, hrecursive, houtcome⟩
        have hsucc : Word.add stepWord (Word.ofNat 1) = Word.ofNat (step + 1) := by
          apply Fin.ext
          unfold stepWord Word.add Word.ofNat
          change ((step % wordModulus + 1 % wordModulus) % wordModulus) =
            (step + 1) % wordModulus
          rw [modEqOfLt (by decide : 1 < wordModulus), modAddAbsorbLeft]
        have hrecursiveArgs : EvalArgs context .unbounded (depth + 1) slotEnvironment memory
            [.name "base", .name "cap", .name "mask", .name "key", .name "slot",
              .binary .add (.name "step") (.literal 1)]
            (.value [base, cap, mask, key, home, Word.ofNat (step + 1)] memory)
            [.arithmetic .add] := by
          apply EvalArgs.cons (EvalExpr.local hbaseIn)
          apply EvalArgs.cons (EvalExpr.local hcapIn)
          apply EvalArgs.cons (EvalExpr.local hmaskIn)
          apply EvalArgs.cons (EvalExpr.local hkeyIn)
          apply EvalArgs.cons (EvalExpr.local hslotIn)
          have hthird : EvalExpr context .unbounded (depth + 1) slotEnvironment memory
              (.binary .add (.name "step") (.literal 1))
              (.value (Word.ofNat (step + 1)) memory) [.arithmetic .add] := by
            rw [← hsucc]
            exact EvalExpr.binary (operator := .add) (EvalExpr.local hstepIn)
              (EvalExpr.literal (literalValue := 1))
          exact EvalArgs.cons hthird EvalArgs.nil
        have hreturn : ExecStmt context .unbounded (depth + 1) slotEnvironment memory
            (.return_ (.call (probeName emptyReturnsSlot)
              [.name "base", .name "cap", .name "mask", .name "key", .name "slot",
                .binary .add (.name "step") (.literal 1)]))
            (.returned raw memory) ([.arithmetic .add] ++ recursiveTrace) := by
          apply ExecStmt.return_
          simpa only [mask, home] using EvalExpr.call hrecursiveArgs hrecursive
        have hbody : ExecList context .unbounded (depth + 1) environment memory probe.body
            (.returned raw memory)
            ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
              [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
                .load (Word.ofNat (keyCoordinate base curr.val))] ++
              [.comparison .eq] ++ [.comparison .eq] ++
              ([.arithmetic .add] ++ recursiveTrace)) := by
          rw [hprobeBody]
          unfold probeBody
          simpa only [listAppendAssoc] using ExecList.next hguard
            (ExecList.next hcurrBind (ExecList.next hslotBind
              (ExecList.next hemptyIf (ExecList.next hmatchIf (ExecList.returned hreturn)))))
        let trace : Trace := .call (probeName emptyReturnsSlot) (depth + 1) ::
          ([.comparison .ge] ++ [.arithmetic .add, .bitwise .bitAnd] ++
            [.call "tableSlotKey" (depth + 2), .arithmetic .mul, .arithmetic .add,
              .load (Word.ofNat (keyCoordinate base curr.val))] ++
            [.comparison .eq] ++ [.comparison .eq] ++
            ([.arithmetic .add] ++ recursiveTrace))
        refine ⟨raw, trace, ?_, ?_⟩
        · simpa only [trace, mask, home, stepWord] using
            EvalCall.run (policy := .unbounded) (depth := depth)
              hprobe hframe True.intro hbody
        · exact ProbeOutcome.next step raw hlt
            (by
              intro heq
              rw [hslotValue] at heq
              exact hneEmpty (Option.some.inj heq))
            (by
              intro heq
              rw [hslotValue] at heq
              exact hmatch (Option.some.inj heq)) houtcome
theorem tableLookupSlotGo_execution (context : Context) (slotKey probe : FunctionDecl)
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
      ProbeOutcome false memory base cap key step raw := by
  exact probe_execution false context slotKey probe hslotKey hprobe hslotKeyShape hprobeShape.1
    (by simpa only [probeName, probeEmptyExpr] using hprobeShape.2)
    hstride hempty memory base cap
    default key occupied h step depth rep hstep

theorem tableFindSlotGo_execution (context : Context) (slotKey probe : FunctionDecl)
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
      ProbeOutcome true memory base cap key step raw := by
  exact probe_execution true context slotKey probe hslotKey hprobe hslotKeyShape hprobeShape.1
    (by simpa only [probeName, probeEmptyExpr] using hprobeShape.2)
    hstride hempty memory base cap
    default key occupied h step depth rep hstep

theorem ProbeOutcome.lookup_present (memory : Memory) (base cap key : Word)
    (start found : Nat) (rawWord : Word)
    (outcome : ProbeOutcome false memory base cap key start rawWord)
    (hstart : start ≤ found) (hfound : found < cap.val)
    (hmatch : keyAt memory base (probeSlot cap key found) = some key)
    (hneEmpty : ∀ earlier, start ≤ earlier → earlier < found →
      keyAt memory base (probeSlot cap key earlier) ≠ some tableEmpty)
    (hkey : key ≠ tableEmpty) :
    ∃ slot < cap.val, rawWord = Word.ofNat slot ∧ keyAt memory base slot = some key := by
  induction outcome with
  | exhausted step stepEq =>
      exact False.elim (Nat.not_lt_of_ge (stepEq ▸ hstart) hfound)
  | empty step hstep hempty =>
      have hle : step ≤ found := hstart
      by_cases heq : step = found
      · subst found
        rw [hempty] at hmatch
        exact (hkey (Option.some.inj hmatch.symm)).elim
      · exact (hneEmpty step le_rfl (Nat.lt_of_le_of_ne hle heq) hempty).elim
  | «match» step hstep hmatchNow =>
      exact ⟨probeSlot cap key step, Nat.mod_lt _ (Nat.zero_lt_of_lt hstep), rfl, hmatchNow⟩
  | next step laterRaw hstep hneEmptyNow hneKey later ih =>
      have hnext : step + 1 ≤ found := by
        by_contra hnot
        have foundLe : found ≤ step := Nat.le_of_lt_succ (Nat.lt_of_not_ge hnot)
        have : step = found := Nat.le_antisymm hstart foundLe
        subst found
        exact hneKey hmatch
      apply ih hnext
      exact fun earlier hearler hlater =>
        hneEmpty earlier (Nat.le_trans (Nat.le_succ step) hearler) hlater

theorem ProbeOutcome.lookup_miss (memory : Memory) (base cap default key : Word)
    (occupied : List Word) (h : Word → Word) (start : Nat) (rawWord : Word)
    (rep : RepSong memory base cap default occupied h) (hnot : member occupied key = false)
    (hkey : key ≠ tableEmpty)
    (outcome : ProbeOutcome false memory base cap key start rawWord) : rawWord = tableEmpty := by
  induction outcome with
  | exhausted => rfl
  | empty => rfl
  | «match» step hstep hmatch =>
      have hmember := (rep.2.2.2.2.1.2 key).2
        ⟨probeSlot cap key step, Nat.mod_lt _ rep.positivePowerOfTwo.pos, hmatch, hkey⟩
      rw [hmember] at hnot
      contradiction
  | next _ _ _ _ _ _ ih => exact ih

theorem tableSpread_execution (context : Context) (spread : FunctionDecl)
    (hspread : context.functions "tableSpread" = some spread)
    (hshape : TableSpreadShape spread)
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (key : Word) (depth : Nat) :
    ∃ trace, EvalCall context .unbounded depth "tableSpread" [key] memory
      (.value (Word.tableSpread key) memory) trace := by
  let environment : Environment := [("k", key)]
  have hframe : Environment.ofParameters spread.parameters [key] = some environment := by
    rw [hshape.1]
    rfl
  have hkey : Environment.lookup environment "k" = some key := by
    rfl
  have hphiLocal : Environment.lookup environment "TABLE_PHI" = none := by
    rfl
  have hltEval : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .lt (.name "k") (.literal 0))
      (.value (Word.lt key (Word.ofNat 0)) memory) [.comparison .lt] := by
    exact EvalExpr.binary (operator := .lt) (EvalExpr.local hkey) EvalExpr.literal
  by_cases hnegative : Word.signed key < 0
  · have hltOne : Word.lt key (Word.ofNat 0) = Word.ofNat 1 := by
      apply wordLt_true
      rw [signed_ofNat_emptyCone 0 wordSignBit_pos]
      exact hnegative
    have hone : Word.nonzero (Word.ofNat 1) := by
      exact wordOne_nonzero
    have hposEval : EvalExpr context .unbounded (depth + 1) environment memory
        (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k"))
        (.value (Word.spreadPosition key) memory) [.comparison .lt, .arithmetic .sub] := by
      rw [Word.spreadPosition, if_pos hnegative]
      have hltEvalOne := hltEval
      rw [hltOne] at hltEvalOne
      apply EvalExpr.conditionalTrue hltEvalOne hone
      exact EvalExpr.binary (operator := .sub) (EvalExpr.literal (literalValue := 0))
        (EvalExpr.local hkey)
    let nextEnvironment := Environment.bind environment "pos" (Word.spreadPosition key)
    have hbind : ExecStmt context .unbounded (depth + 1) environment memory
        (.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k")))
        (.next nextEnvironment memory) [.comparison .lt, .arithmetic .sub] :=
      ExecStmt.bind hposEval
    have hpos : Environment.lookup nextEnvironment "pos" = some (Word.spreadPosition key) :=
      Environment.lookup_bind_same _ _ _
    have hphiNext : Environment.lookup nextEnvironment "TABLE_PHI" = none := by
      rw [Environment.lookup_bind_other _ "pos" "TABLE_PHI" _ (by decide)]
      exact hphiLocal
    have hreturn : ExecStmt context .unbounded (depth + 1) nextEnvironment memory
        (.return_ (.binary .bitAnd
          (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903)))
        (.returned (Word.tableSpread key) memory) [.arithmetic .mul, .bitwise .bitAnd] := by
      apply ExecStmt.return_
      exact EvalExpr.binary (operator := .bitAnd)
        (EvalExpr.binary (operator := .mul) (EvalExpr.local hpos)
          (EvalExpr.constant hphiNext hphi))
        (EvalExpr.literal (literalValue := 4611686018427387903))
    have hbody : ExecList context .unbounded (depth + 1) environment memory spread.body
        (.returned (Word.tableSpread key) memory)
        ([.comparison .lt, .arithmetic .sub] ++ [.arithmetic .mul, .bitwise .bitAnd]) := by
      rw [hshape.2]
      exact ExecList.next hbind (ExecList.returned hreturn)
    let trace : Trace := .call "tableSpread" (depth + 1) ::
      ([.comparison .lt, .arithmetic .sub] ++ [.arithmetic .mul, .bitwise .bitAnd])
    refine ⟨trace, ?_⟩
    exact EvalCall.run hspread hframe True.intro hbody
  · have hltZero : Word.lt key (Word.ofNat 0) = Word.ofNat 0 := by
      apply wordLt_false
      rw [signed_ofNat_emptyCone 0 wordSignBit_pos]
      exact hnegative
    have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hz => hz rfl
    have hposEval : EvalExpr context .unbounded (depth + 1) environment memory
        (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k"))
        (.value (Word.spreadPosition key) memory) [.comparison .lt] := by
      rw [Word.spreadPosition, if_neg hnegative]
      have hltEvalZero := hltEval
      rw [hltZero] at hltEvalZero
      exact EvalExpr.conditionalFalse hltEvalZero hzero (EvalExpr.local hkey)
    let nextEnvironment := Environment.bind environment "pos" (Word.spreadPosition key)
    have hbind : ExecStmt context .unbounded (depth + 1) environment memory
        (.bind "pos" (.conditional (.binary .sub (.literal 0) (.name "k"))
          (.binary .lt (.name "k") (.literal 0)) (.name "k")))
        (.next nextEnvironment memory) [.comparison .lt] := ExecStmt.bind hposEval
    have hpos : Environment.lookup nextEnvironment "pos" = some (Word.spreadPosition key) :=
      Environment.lookup_bind_same _ _ _
    have hphiNext : Environment.lookup nextEnvironment "TABLE_PHI" = none := by
      rw [Environment.lookup_bind_other _ "pos" "TABLE_PHI" _ (by decide)]
      exact hphiLocal
    have hreturn : ExecStmt context .unbounded (depth + 1) nextEnvironment memory
        (.return_ (.binary .bitAnd
          (.binary .mul (.name "pos") (.name "TABLE_PHI")) (.literal 4611686018427387903)))
        (.returned (Word.tableSpread key) memory) [.arithmetic .mul, .bitwise .bitAnd] := by
      apply ExecStmt.return_
      exact EvalExpr.binary (operator := .bitAnd)
        (EvalExpr.binary (operator := .mul) (EvalExpr.local hpos)
          (EvalExpr.constant hphiNext hphi))
        (EvalExpr.literal (literalValue := 4611686018427387903))
    have hbody : ExecList context .unbounded (depth + 1) environment memory spread.body
        (.returned (Word.tableSpread key) memory)
        ([.comparison .lt] ++ [.arithmetic .mul, .bitwise .bitAnd]) := by
      rw [hshape.2]
      exact ExecList.next hbind (ExecList.returned hreturn)
    let trace : Trace := .call "tableSpread" (depth + 1) ::
      ([.comparison .lt] ++ [.arithmetic .mul, .bitwise .bitAnd])
    refine ⟨trace, ?_⟩
    exact EvalCall.run hspread hframe True.intro hbody

theorem tableFind_execution (context : Context)
    (slotKey capacityOk spread lookup find : FunctionDecl)
    (hslotKey : context.functions "tableSlotKey" = some slotKey)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hspread : context.functions "tableSpread" = some spread)
    (hlookup : context.functions "tableLookupSlotGo" = some lookup)
    (hfind : context.functions "tableFind" = some find)
    (hslotKeyShape : TableSlotKeyShape slotKey)
    (hcapacityShape : TableCapacityOkShape capacityOk)
    (hspreadShape : TableSpreadShape spread)
    (hlookupShape : TableLookupSlotGoShape lookup) (hfindShape : TableFindShape find)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (hphi : context.constants "TABLE_PHI" = some Word.tablePhi)
    (memory : Memory) (base cap default key : Word) (occupied : List Word)
    (h : Word → Word) (depth : Nat) (rep : RepSong memory base cap default occupied h)
    (hkey : key ≠ tableEmpty) :
    ∃ raw trace,
      EvalCall context .unbounded depth "tableFind" [base, cap, key] memory
        (.value raw memory) trace ∧
      ProbeOutcome false memory base cap key 0 raw := by
  let environment : Environment := [("base", base), ("cap", cap), ("key", key)]
  have hframe : Environment.ofParameters find.parameters [base, cap, key] = some environment := by
    rw [hfindShape.1]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    rfl
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    rfl
  have hkeyLocal : Environment.lookup environment "key" = some key := by
    rfl
  have hemptyLocal : Environment.lookup environment "TABLE_EMPTY" = none := by
    rfl
  have hcapSign : cap.val < wordSignBit := by
    exact cap_below_signBit_of_backing memory base cap rep.validExtent.1
  have hcapCall := tableCapacityOk_execution context capacityOk hcapacityOk hcapacityShape
    memory cap (depth + 1) rep.positivePowerOfTwo hcapSign
  have hcapArgs : EvalArgs context .unbounded (depth + 1) environment memory [.name "cap"]
      (.value [cap] memory) [] := EvalArgs.cons (EvalExpr.local hcapLocal) EvalArgs.nil
  have hcapExpr : EvalExpr context .unbounded (depth + 1) environment memory
      (.call "tableCapacityOk" [.name "cap"]) (.value (Word.ofNat 1) memory)
      [.call "tableCapacityOk" (depth + 2), .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq] := by
    exact EvalExpr.call hcapArgs hcapCall
  have hcapEq : Word.eq (Word.ofNat 1) (Word.ofNat 0) = Word.ofNat 0 := by
    exact wordEq_ne _ _ (fun equal => Nat.noConfusion (congrArg Fin.val equal))
  have hcapGuardExpr : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
      (.value (Word.ofNat 0) memory)
      [.call "tableCapacityOk" (depth + 2), .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] := by
    rw [← hcapEq]
    exact EvalExpr.binary (operator := .eq) hcapExpr (EvalExpr.literal (literalValue := 0))
  have hzero : ¬ Word.nonzero (Word.ofNat 0) := fun hz => hz rfl
  have hcapGuard : ExecStmt context .unbounded (depth + 1) environment memory
      (.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
        [.return_ (.name "TABLE_EMPTY")] none)
      (.next environment memory)
      [.call "tableCapacityOk" (depth + 2), .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] := by
    exact ExecStmt.ifFalse hcapGuardExpr hzero ExecList.nil
  have hkeyEq : Word.eq key tableEmpty = Word.ofNat 0 := by
    exact wordEq_ne key tableEmpty hkey
  have hkeyGuardExpr : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
      (.value (Word.ofNat 0) memory) [.comparison .eq] := by
    rw [← hkeyEq]
    simpa only [Machine.applyBinOp, Machine.operatorEvent, List.nil_append] using
      EvalExpr.binary (operator := .eq) (EvalExpr.local hkeyLocal)
        (EvalExpr.constant hemptyLocal hempty)
  have hkeyGuard : ExecStmt context .unbounded (depth + 1) environment memory
      (.if_ (.binary .eq (.name "key") (.name "TABLE_EMPTY"))
        [.return_ (.name "TABLE_EMPTY")] none)
      (.next environment memory) [.comparison .eq] :=
    ExecStmt.ifFalse hkeyGuardExpr hzero ExecList.nil
  let mask := Word.sub cap (Word.ofNat 1)
  let maskEnvironment := Environment.bind environment "mask" mask
  have hmaskEval : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .sub (.name "cap") (.literal 1)) (.value mask memory) [.arithmetic .sub] := by
    simpa only [mask, Machine.applyBinOp, Machine.operatorEvent, List.nil_append] using
      EvalExpr.binary (operator := .sub) (EvalExpr.local hcapLocal) EvalExpr.literal
  have hmaskBind : ExecStmt context .unbounded (depth + 1) environment memory
      (.bind "mask" (.binary .sub (.name "cap") (.literal 1)))
      (.next maskEnvironment memory) [.arithmetic .sub] := ExecStmt.bind hmaskEval
  have hkeyMask : Environment.lookup maskEnvironment "key" = some key := by
    rw [Environment.lookup_bind_other _ "mask" "key" mask (by decide)]
    exact hkeyLocal
  have hmaskLocal : Environment.lookup maskEnvironment "mask" = some mask :=
    Environment.lookup_bind_same _ _ _
  rcases tableSpread_execution context spread hspread hspreadShape hphi memory key (depth + 1) with
    ⟨spreadTrace, hspreadCall⟩
  have hspreadArgs : EvalArgs context .unbounded (depth + 1) maskEnvironment memory [.name "key"]
      (.value [key] memory) [] := EvalArgs.cons (EvalExpr.local hkeyMask) EvalArgs.nil
  have hspreadExpr : EvalExpr context .unbounded (depth + 1) maskEnvironment memory
      (.call "tableSpread" [.name "key"]) (.value (Word.tableSpread key) memory) spreadTrace := by
    exact EvalExpr.call hspreadArgs hspreadCall
  let home := Word.bitAnd (Word.tableSpread key) mask
  have hhomeEval : EvalExpr context .unbounded (depth + 1) maskEnvironment memory
      (.binary .bitAnd (.call "tableSpread" [.name "key"]) (.name "mask"))
      (.value home memory) (spreadTrace ++ [.bitwise .bitAnd]) := by
    simpa only [home, Machine.applyBinOp, Machine.operatorEvent, List.nil_append, listAppendNil] using
      EvalExpr.binary (operator := .bitAnd) hspreadExpr (EvalExpr.local hmaskLocal)
  let homeEnvironment := Environment.bind maskEnvironment "initSlot" home
  have hhomeBind : ExecStmt context .unbounded (depth + 1) maskEnvironment memory
      (.bind "initSlot" (.binary .bitAnd (.call "tableSpread" [.name "key"])
        (.name "mask"))) (.next homeEnvironment memory)
      (spreadTrace ++ [.bitwise .bitAnd]) := ExecStmt.bind hhomeEval
  have lookupResult := tableLookupSlotGo_execution context slotKey lookup hslotKey hlookup
    hslotKeyShape hlookupShape hstride hempty memory base cap default key occupied h 0
    (depth + 1) rep (Nat.zero_le cap.val)
  rcases lookupResult with ⟨raw, lookupTrace, hlookupCall, houtcome⟩
  have hbaseHome : Environment.lookup homeEnvironment "base" = some base := by
    rfl
  have hcapHome : Environment.lookup homeEnvironment "cap" = some cap := by
    rfl
  have hmaskHome : Environment.lookup homeEnvironment "mask" = some mask := by
    rfl
  have hkeyHome : Environment.lookup homeEnvironment "key" = some key := by
    rfl
  have hhomeLocal : Environment.lookup homeEnvironment "initSlot" = some home :=
    Environment.lookup_bind_same _ _ _
  have hlookupArgs : EvalArgs context .unbounded (depth + 1) homeEnvironment memory
      [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot", .literal 0]
      (.value [base, cap, mask, key, home, Word.ofNat 0] memory) [] := by
    exact EvalArgs.cons (EvalExpr.local hbaseHome)
      (EvalArgs.cons (EvalExpr.local hcapHome)
        (EvalArgs.cons (EvalExpr.local hmaskHome)
          (EvalArgs.cons (EvalExpr.local hkeyHome)
            (EvalArgs.cons (EvalExpr.local hhomeLocal)
              (EvalArgs.cons EvalExpr.literal EvalArgs.nil)))))
  have hreturn : ExecStmt context .unbounded (depth + 1) homeEnvironment memory
      (.return_ (.call "tableLookupSlotGo"
        [.name "base", .name "cap", .name "mask", .name "key", .name "initSlot", .literal 0]))
      (.returned raw memory) lookupTrace := by
    apply ExecStmt.return_
    simpa only [mask, home, List.nil_append] using EvalExpr.call hlookupArgs hlookupCall
  have hbody : ExecList context .unbounded (depth + 1) environment memory find.body
      (.returned raw memory)
      ([.call "tableCapacityOk" (depth + 2), .comparison .le, .arithmetic .sub,
          .bitwise .bitAnd, .comparison .eq, .comparison .eq] ++ [.comparison .eq] ++
        [.arithmetic .sub] ++ (spreadTrace ++ [.bitwise .bitAnd]) ++ lookupTrace) := by
    rw [hfindShape.2]
    simpa only [listAppendAssoc] using ExecList.next hcapGuard
      (ExecList.next hkeyGuard (ExecList.next hmaskBind
        (ExecList.next hhomeBind (ExecList.returned hreturn))))
  let trace : Trace := .call "tableFind" (depth + 1) ::
    ([.call "tableCapacityOk" (depth + 2), .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] ++ [.comparison .eq] ++
      [.arithmetic .sub] ++ (spreadTrace ++ [.bitwise .bitAnd]) ++ lookupTrace)
  exact ⟨raw, trace, by
    exact EvalCall.run (policy := .unbounded) (depth := depth)
      hfind hframe True.intro hbody, houtcome⟩

end Slang.Table
