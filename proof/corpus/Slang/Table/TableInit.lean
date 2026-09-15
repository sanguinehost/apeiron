import Slang.Table.TableRep
import Slang.SourceAudit
import Slang.Refinement.Context
import Slang.Refinement.WordRing

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Table

open Slang Slang.Decode Slang.Machine Slang.Source Slang.Refinement

def slotAddressExpr : Expr :=
  .binary .add (.name "base") (.binary .mul (.name "TABLE_STRIDE") (.name "slot"))

def slotValueAddressExpr : Expr := .binary .add slotAddressExpr (.literal 1)

def TableSlotSetShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "slot", "key", "value"] ∧
  declaration.body =
    [.store slotAddressExpr (.name "key"),
     .store slotValueAddressExpr (.name "value"),
     .return_ (.name "value")]

def TableInitGoShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap", "i"] ∧
  declaration.body =
    [.if_ (.binary .ge (.name "i") (.name "cap")) [.return_ (.literal 0)] none,
     .expression (.call "tableSlotSet"
       [.name "base", .name "i", .name "TABLE_EMPTY", .literal 0]),
     .return_ (.call "tableInitGo"
       [.name "base", .name "cap", .binary .add (.name "i") (.literal 1)])]

def TableInitShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["base", "cap"] ∧
  declaration.body =
    [.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
      [.return_ (.name "TABLE_EMPTY")] none,
     .return_ (.call "tableInitGo" [.name "base", .name "cap", .literal 0])]

def TableCapacityOkShape (declaration : FunctionDecl) : Prop :=
  declaration.parameters = ["cap"] ∧
  declaration.body =
    [.if_ (.binary .le (.name "cap") (.literal 0)) [.return_ (.literal 0)] none,
     .return_ (.conditional (.literal 1)
       (.binary .eq
         (.binary .bitAnd (.name "cap")
           (.binary .sub (.name "cap") (.literal 1)))
         (.literal 0))
       (.literal 0))]

def InitializedEmptyLayout (memory : Memory) (base cap : Word) : Prop :=
  (∀ slot < cap.val, keyAt memory base slot = some tableEmpty) ∧
  ∀ slot < cap.val, valueAt memory base slot = some (Word.ofNat 0)

def storeAddresses : Trace → List Word
  | [] => []
  | event :: rest =>
      match event with
      | .store address => address :: storeAddresses rest
      | .arithmetic _ | .comparison _ | .bitwise _ | .load _ | .call _ _ =>
          storeAddresses rest

def expectedInitStoresFuel (base : Word) : Nat → Nat → List Word
  | _, 0 => []
  | slot, fuel + 1 =>
      Word.ofNat (keyCoordinate base slot) ::
        Word.ofNat (valueCoordinate base slot) :: expectedInitStoresFuel base (slot + 1) fuel

def expectedInitStoresFrom (base cap : Word) (i : Nat) : List Word :=
  expectedInitStoresFuel base i (cap.val - i)

def expectedInitStores (base cap : Word) : List Word := expectedInitStoresFrom base cap 0

def InInitSuffix (base cap : Word) (i coordinate : Nat) : Prop :=
  ∃ slot, i ≤ slot ∧ slot < cap.val ∧
    (coordinate = keyCoordinate base slot ∨ coordinate = valueCoordinate base slot)

def InitializedEmptySuffix (memory : Memory) (base cap : Word) (i : Nat) : Prop :=
  ∀ slot, i ≤ slot → slot < cap.val →
    keyAt memory base slot = some tableEmpty ∧
      valueAt memory base slot = some (Word.ofNat 0)

def PreservesOutsideSuffix (before after : Memory) (base cap : Word) (i : Nat) : Prop :=
  after.owned = before.owned ∧
    ∀ coordinate, ¬ InInitSuffix base cap i coordinate →
      after.initialized coordinate = before.initialized coordinate

def slotSetTrace (depth : Nat) (base slot : Word) : Trace :=
  .call "tableSlotSet" (depth + 1) ::
    ([.arithmetic .mul, .arithmetic .add,
      .store (Word.ofNat (keyCoordinate base slot.val))] ++
     [.arithmetic .mul, .arithmetic .add, .arithmetic .add,
      .store (Word.ofNat (valueCoordinate base slot.val))])

structure ConstructorExecution (context : Context) (before after : Memory)
    (base cap : Word) (trace : Trace) : Prop where
  call : EvalCall context .unbounded 0 "tableInit" [base, cap] before
    (.value (Word.ofNat 0) after) trace
  layout : InitializedEmptyLayout after base cap
  exactStores : storeAddresses trace = expectedInitStores base cap
  storeCount : (resources trace).stores = 2 * cap.val
  frame : PreservesOutside before after base cap

theorem wordSignBit_pos : 0 < wordSignBit := by decide

theorem wordSignBit_lt_modulus : wordSignBit < wordModulus := by decide

theorem intOfNat_not_lt_zero (value : Nat) : ¬Int.ofNat value < 0 := by
  intro negative
  change Int.NonNeg (0 - (Int.ofNat value + 1)) at negative
  cases value <;> cases negative

theorem natSub_pos_of_lt {left right : Nat} (bound : left < right) : 0 < right - left := by
  induction right generalizing left with
  | zero => exact False.elim (Nat.not_lt_zero left bound)
  | succ right inductionHypothesis =>
      cases left with
      | zero => exact Nat.zero_lt_succ right
      | succ left =>
          rw [Nat.succ_sub_succ_eq_sub]
          exact inductionHypothesis (Nat.lt_of_succ_lt_succ bound)

theorem intOfNat_le_refl (value : Nat) : Int.le (Int.ofNat value) (Int.ofNat value) := by
  cases value with
  | zero => exact Int.NonNeg.mk 0
  | succ value =>
      unfold Int.le
      change Int.NonNeg (Int.sub (Int.ofNat (value + 1)) (Int.ofNat (value + 1)))
      unfold Int.sub
      change Int.NonNeg (Int.add (Int.ofNat (value + 1))
        (Int.neg (Int.ofNat (value + 1))))
      dsimp only [Int.neg, Int.negOfNat, Int.add]
      change Int.NonNeg (Int.subNatNat (value + 1) (value + 1))
      unfold Int.subNatNat
      rw [Nat.sub_self]
      exact Int.NonNeg.mk 0

theorem intOfNat_not_le_of_lt {left right : Nat} (bound : left < right) :
    ¬Int.le (Int.ofNat right) (Int.ofNat left) := by
  cases right with
  | zero => exact False.elim (Nat.not_lt_zero left bound)
  | succ right =>
      intro lower
      unfold Int.le at lower
      change Int.NonNeg (Int.sub (Int.ofNat left) (Int.ofNat (right + 1))) at lower
      unfold Int.sub at lower
      change Int.NonNeg (Int.add (Int.ofNat left) (Int.neg (Int.ofNat (right + 1)))) at lower
      dsimp only [Int.neg, Int.negOfNat, Int.add] at lower
      change Int.NonNeg (Int.subNatNat left (right + 1)) at lower
      unfold Int.subNatNat at lower
      cases differenceEq : right + 1 - left with
      | zero =>
          exact False.elim (Nat.not_lt_zero 0 (differenceEq ▸ natSub_pos_of_lt bound))
      | succ difference =>
          rw [differenceEq] at lower
          cases lower

theorem signed_ofNat_nonnegative (coordinate : Nat) (bound : coordinate < wordSignBit) :
    ¬Word.signed (Word.ofNat coordinate) < 0 := by
  have modulusBound : coordinate < wordModulus :=
    Nat.lt_trans bound wordSignBit_lt_modulus
  have valueEq : (Word.ofNat coordinate).val = coordinate := modEqOfLt modulusBound
  have below : (Word.ofNat coordinate).val < wordSignBit :=
    Nat.lt_of_le_of_lt (Nat.le_of_eq valueEq) bound
  have signedEq : Word.signed (Word.ofNat coordinate) =
      Int.ofNat (Word.ofNat coordinate).val := by
    unfold Word.signed
    exact if_pos below
  intro negative
  have impossible : Int.ofNat (Word.ofNat coordinate).val < 0 :=
    Eq.mp (congrArg (fun value => value < 0) signedEq) negative
  exact intOfNat_not_lt_zero _ impossible

theorem coordinate_ofNat_emptyCone (coordinate : Nat) (bound : coordinate < wordSignBit) :
    Memory.coordinate (Word.ofNat coordinate) = some coordinate := by
  have modulusBound : coordinate < wordModulus :=
    Nat.lt_trans bound wordSignBit_lt_modulus
  have valueEq : (Word.ofNat coordinate).val = coordinate := modEqOfLt modulusBound
  unfold Memory.coordinate
  exact Eq.trans (if_neg (signed_ofNat_nonnegative coordinate bound)) (congrArg some valueEq)

theorem ofNat_val_emptyCone (coordinate : Nat) (bound : coordinate < wordSignBit) :
    (Word.ofNat coordinate).val = coordinate :=
  modEqOfLt (Nat.lt_trans bound wordSignBit_lt_modulus)

theorem signed_ofNat_emptyCone (coordinate : Nat) (bound : coordinate < wordSignBit) :
    Word.signed (Word.ofNat coordinate) = Int.ofNat coordinate := by
  have valueEq := ofNat_val_emptyCone coordinate bound
  have below : (Word.ofNat coordinate).val < wordSignBit :=
    Nat.lt_of_le_of_lt (Nat.le_of_eq valueEq) bound
  exact Eq.trans (if_pos below) (congrArg Int.ofNat valueEq)

theorem wordGe_ofNat_self (value : Nat) (bound : value < wordSignBit) :
    Word.ge (Word.ofNat value) (Word.ofNat value) = Word.ofNat 1 := by
  have signedEq := signed_ofNat_emptyCone value bound
  have lower : Int.le (Word.signed (Word.ofNat value))
      (Word.signed (Word.ofNat value)) :=
    Eq.mpr (congrArg₂ Int.le signedEq signedEq) (intOfNat_le_refl value)
  unfold Word.ge Word.bool
  cases decision : Int.decLe (Word.signed (Word.ofNat value))
      (Word.signed (Word.ofNat value)) with
  | isTrue _ => rfl
  | isFalse refute => exact False.elim (refute lower)

theorem wordGe_ofNat_lt (left right : Nat) (leftBound : left < wordSignBit)
    (rightBound : right < wordSignBit) (strict : left < right) :
    Word.ge (Word.ofNat left) (Word.ofNat right) = Word.ofNat 0 := by
  have notCompare : ¬Int.le (Word.signed (Word.ofNat right))
      (Word.signed (Word.ofNat left)) := by
    have leftEq := signed_ofNat_emptyCone left leftBound
    have rightEq := signed_ofNat_emptyCone right rightBound
    exact fun lower => intOfNat_not_le_of_lt strict
      (Eq.mp (congrArg₂ Int.le rightEq leftEq) lower)
  unfold Word.ge Word.bool
  cases decision : Int.decLe (Word.signed (Word.ofNat right))
      (Word.signed (Word.ofNat left)) with
  | isTrue lower => exact False.elim (notCompare lower)
  | isFalse _ => rfl

theorem wordLe_ofNat_pos_zero (value : Nat) (bound : value < wordSignBit)
    (positive : 0 < value) :
    Word.le (Word.ofNat value) (Word.ofNat 0) = Word.ofNat 0 := by
  have notCompare : ¬Int.le (Word.signed (Word.ofNat value))
      (Word.signed (Word.ofNat 0)) := by
    have valueEq := signed_ofNat_emptyCone value bound
    have zeroEq := signed_ofNat_emptyCone 0 wordSignBit_pos
    exact fun lower => intOfNat_not_le_of_lt positive
      (Eq.mp (congrArg₂ Int.le valueEq zeroEq) lower)
  unfold Word.le Word.bool
  cases decision : Int.decLe (Word.signed (Word.ofNat value))
      (Word.signed (Word.ofNat 0)) with
  | isTrue lower => exact False.elim (notCompare lower)
  | isFalse _ => rfl

theorem keyCoordinate_below_signBit_emptyCone (base cap : Word) (slot : Nat)
    (frameBound : base.val + 2 * cap.val ≤ wordSignBit) (slotBound : slot < cap.val) :
    keyCoordinate base slot < wordSignBit := by
  have doubled : 2 * slot + 2 ≤ 2 * cap.val := by
    simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt slotBound)
  have strict : 2 * slot < 2 * cap.val :=
    Nat.lt_of_lt_of_le (Nat.lt_trans (Nat.lt_succ_self (2 * slot))
      (Nat.lt_succ_self (2 * slot + 1))) doubled
  exact Nat.lt_of_lt_of_le
    (Nat.add_lt_add_left strict base.val)
    frameBound

theorem valueCoordinate_below_signBit_emptyCone (base cap : Word) (slot : Nat)
    (frameBound : base.val + 2 * cap.val ≤ wordSignBit) (slotBound : slot < cap.val) :
    valueCoordinate base slot < wordSignBit := by
  have doubled : 2 * slot + 2 ≤ 2 * cap.val := by
    simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt slotBound)
  exact Nat.lt_of_lt_of_le
    (Nat.add_lt_add_left (Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * slot + 1)) doubled)
      base.val)
    frameBound

theorem cap_below_signBit_of_backing (memory : Memory) (base cap : Word)
    (hbacking : ValidBacking memory base cap) : cap.val < wordSignBit := by
  by_cases hzero : cap.val = 0
  · rw [hzero]
    exact wordSignBit_pos
  · have capPositive : 0 < cap.val := Nat.pos_of_ne_zero hzero
    have capLtDouble : cap.val < 2 * cap.val := by
      rw [Nat.two_mul]
      exact Nat.lt_add_of_pos_left capPositive
    have doubleLeFrame : 2 * cap.val ≤ base.val + 2 * cap.val := Nat.le_add_left _ _
    exact Nat.lt_of_lt_of_le capLtDouble
      (Nat.le_trans doubleLeFrame hbacking.2.1)

theorem sub_succ_difference {i cap : Nat} (hlt : i < cap) :
    cap - i = (cap - (i + 1)) + 1 := by
  induction cap generalizing i with
  | zero => exact False.elim (Nat.not_lt_zero i hlt)
  | succ cap inductionHypothesis =>
      cases i with
      | zero => rfl
      | succ i =>
          rw [Nat.succ_sub_succ_eq_sub, Nat.succ_sub_succ_eq_sub]
          exact inductionHypothesis (Nat.lt_of_succ_lt_succ hlt)

theorem keyCoordinate_mono (base : Word) {left right : Nat} (bound : left ≤ right) :
    keyCoordinate base left ≤ keyCoordinate base right := by
  exact Nat.add_le_add_left (Nat.mul_le_mul_left 2 bound) base.val

theorem keyCoordinate_strict_mono (base : Word) {left right : Nat} (bound : left < right) :
    keyCoordinate base left < keyCoordinate base right := by
  have successorBound : left + 1 ≤ right := Nat.succ_le_of_lt bound
  have doubled : 2 * left + 2 ≤ 2 * right := by
    simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 successorBound
  exact Nat.add_lt_add_left
    (Nat.lt_of_lt_of_le (Nat.lt_trans (Nat.lt_succ_self (2 * left))
      (Nat.lt_succ_self (2 * left + 1))) doubled) base.val

theorem valueCoordinate_strict_mono (base : Word) {left right : Nat} (bound : left < right) :
    valueCoordinate base left < valueCoordinate base right := by
  exact Nat.add_lt_add_right (keyCoordinate_strict_mono base bound) 1

theorem keyCoordinate_lt_valueCoordinate (base : Word) (slot : Nat) :
    keyCoordinate base slot < valueCoordinate base slot := Nat.lt_succ_self _

theorem valueCoordinate_lt_nextKey (base : Word) (slot : Nat) :
    valueCoordinate base slot < keyCoordinate base (slot + 1) := by
  unfold valueCoordinate keyCoordinate
  rw [Nat.mul_add, Nat.mul_one, Nat.add_assoc]
  exact Nat.lt_succ_self (base.val + 2 * slot + 1)

theorem current_key_not_in_suffix (base cap : Word) (i : Nat) :
    ¬InInitSuffix base cap (i + 1) (keyCoordinate base i) := by
  intro occurrence
  rcases occurrence with ⟨slot, lower, _, coordinate⟩
  have strict : i < slot := Nat.lt_of_succ_le lower
  rcases coordinate with coordinate | coordinate
  · exact Nat.ne_of_lt (keyCoordinate_strict_mono base strict) coordinate
  · have less := Nat.lt_trans (keyCoordinate_strict_mono base strict)
      (keyCoordinate_lt_valueCoordinate base slot)
    exact Nat.ne_of_lt less coordinate

theorem current_value_not_in_suffix (base cap : Word) (i : Nat) :
    ¬InInitSuffix base cap (i + 1) (valueCoordinate base i) := by
  intro occurrence
  rcases occurrence with ⟨slot, lower, _, coordinate⟩
  have nextLe : i + 1 ≤ slot := lower
  have valueLessKey : valueCoordinate base i < keyCoordinate base slot :=
    Nat.lt_of_lt_of_le (valueCoordinate_lt_nextKey base i) (keyCoordinate_mono base nextLe)
  rcases coordinate with coordinate | coordinate
  · exact Nat.ne_of_lt valueLessKey coordinate
  · exact Nat.ne_of_lt (Nat.lt_trans valueLessKey (keyCoordinate_lt_valueCoordinate base slot))
      coordinate

theorem initSuffix_implies_extent (base cap : Word) (coordinate : Nat)
    (suffix : InInitSuffix base cap 0 coordinate) : InExtent base cap coordinate := by
  rcases suffix with ⟨slot, _, slotBound, coordinateEq⟩
  constructor
  · rcases coordinateEq with rfl | rfl
    · exact Nat.le_add_right _ _
    · exact Nat.le_trans (Nat.le_add_right _ _) (Nat.le_add_right _ _)
  · rcases coordinateEq with rfl | rfl
    · exact Nat.add_lt_add_left
        (by
          have doubled : 2 * slot + 2 ≤ 2 * cap.val := by
            simpa only [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt slotBound)
          exact Nat.lt_of_lt_of_le (Nat.lt_trans (Nat.lt_succ_self (2 * slot))
            (Nat.lt_succ_self (2 * slot + 1))) doubled) base.val
    · have doubled : 2 * slot + 2 ≤ 2 * cap.val := by
        simpa [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt slotBound)
      exact Nat.add_lt_add_left
        (Nat.lt_of_lt_of_le (Nat.lt_succ_self (2 * slot + 1)) doubled) base.val

theorem divGoFuelCongr (value divisor fuel₁ fuel₂ : Nat) (positive : 0 < divisor)
    (bound₁ : value < fuel₁) (bound₂ : value < fuel₂) :
    Nat.div.go divisor positive fuel₁ value bound₁ =
      Nat.div.go divisor positive fuel₂ value bound₂ := by
  match fuel₁, fuel₂ with
  | 0, _ => exact False.elim (Nat.not_lt_zero _ bound₁)
  | _, 0 => exact False.elim (Nat.not_lt_zero _ bound₂)
  | fuel₁ + 1, fuel₂ + 1 =>
      unfold Nat.div.go
      match Nat.decLe divisor value with
      | isTrue lower =>
          exact congrArg (fun quotient => quotient + 1)
            (divGoFuelCongr (value - divisor) divisor fuel₁ fuel₂ positive
              (Nat.div_rec_fuel_lemma positive lower bound₁)
              (Nat.div_rec_fuel_lemma positive lower bound₂))
      | isFalse _ => rfl
termination_by structural fuel₁

theorem divGoValueCongr (left right divisor fuel₁ fuel₂ : Nat) (positive : 0 < divisor)
    (leftBound : left < fuel₁) (rightBound : right < fuel₂) (equal : left = right) :
    Nat.div.go divisor positive fuel₁ left leftBound =
      Nat.div.go divisor positive fuel₂ right rightBound := by
  cases equal
  exact divGoFuelCongr left divisor fuel₁ fuel₂ positive leftBound rightBound

theorem divPositive (value divisor : Nat) (positive : 0 < divisor) :
    Nat.div value divisor =
      Nat.div.go divisor positive (value + 1) value (Nat.lt_succ_self value) := by
  unfold Nat.div
  exact dif_pos positive

theorem divGoStep (value divisor fuel : Nat) (positive : 0 < divisor)
    (bound : value < fuel + 1) :
    Nat.div.go divisor positive (fuel + 1) value bound =
      if lower : divisor ≤ value then
        Nat.div.go divisor positive fuel (value - divisor)
          (Nat.div_rec_fuel_lemma positive lower bound) + 1
      else 0 := rfl

theorem divAddDivisor (value divisor : Nat) (positive : 0 < divisor) :
    Nat.div (value + divisor) divisor = Nat.div value divisor + 1 := by
  refine Eq.trans (divPositive (value + divisor) divisor positive) ?_
  refine Eq.trans ?_ (congrArg (fun quotient => quotient + 1)
    (divPositive value divisor positive).symm)
  refine Eq.trans (divGoStep (value + divisor) divisor (value + divisor) positive
    (Nat.lt_succ_self (value + divisor))) ?_
  refine Eq.trans (dif_pos (Nat.le_add_left divisor value)) ?_
  exact congrArg (fun quotient => quotient + 1)
    (divGoValueCongr (value + divisor - divisor) value divisor
      (value + divisor) (value + 1) positive
      (Nat.div_rec_fuel_lemma positive (Nat.le_add_left divisor value)
        (Nat.lt_succ_self (value + divisor)))
      (Nat.lt_succ_self value) (addSubCancel value divisor))

theorem oddDoubleDivTwo (value : Nat) : (value + value + 1) / 2 = value := by
  induction value with
  | zero => rfl
  | succ value inductionHypothesis =>
      rw [show (value + 1) + (value + 1) + 1 = (value + value + 1) + 2 by
        rw [Nat.add_succ, Nat.succ_add, Nat.add_succ]]
      change Nat.div (value + value + 1 + 2) 2 = value + 1
      rw [divAddDivisor (value + value + 1) 2 (by decide)]
      change Nat.div (value + value + 1) 2 + 1 = value + 1
      change Nat.div (value + value + 1) 2 = value at inductionHypothesis
      rw [inductionHypothesis]

theorem evenDoubleDivTwo (value : Nat) : (value + value) / 2 = value := by
  induction value with
  | zero => rfl
  | succ value inductionHypothesis =>
      rw [show (value + 1) + (value + 1) = (value + value) + 2 by
        rw [Nat.add_succ, Nat.succ_add]]
      change Nat.div (value + value + 2) 2 = value + 1
      rw [divAddDivisor (value + value) 2 (by decide)]
      change Nat.div (value + value) 2 + 1 = value + 1
      change Nat.div (value + value) 2 = value at inductionHypothesis
      rw [inductionHypothesis]

theorem evenDoubleModTwo (value : Nat) : (value + value) % 2 = 0 := by
  induction value with
  | zero => rfl
  | succ value inductionHypothesis =>
      rw [show (value + 1) + (value + 1) = (value + value) + 2 by
        rw [Nat.add_succ, Nat.succ_add]]
      rw [modAddModulus, inductionHypothesis]

theorem bitwise_zero_zero (remaining : Nat) : Word.bitwise Bool.and remaining 0 0 = 0 := by
  induction remaining with
  | zero => rfl
  | succ remaining inductionHypothesis =>
      unfold Word.bitwise
      rw [inductionHypothesis]
      rfl

theorem doublePredDivTwo (value : Nat) (positive : 0 < value) :
    (2 * value - 1) / 2 = value - 1 := by
  cases value with
  | zero => exact False.elim (Nat.not_lt_zero _ positive)
  | succ value =>
      rw [Nat.succ_sub_one, Nat.mul_succ]
      rw [show 2 * value + 2 = (2 * value + 1) + 1 by rfl, addSubCancel]
      rw [Nat.two_mul]
      exact oddDoubleDivTwo value

theorem bitwise_powTwo_pred_zero (fuel exponent : Nat) :
    Word.bitwise Bool.and fuel (powTwo exponent) (powTwo exponent - 1) = 0 := by
  induction fuel generalizing exponent with
  | zero => rfl
  | succ fuel inductionHypothesis =>
      cases exponent with
      | zero =>
          unfold powTwo Word.bitwise
          rw [bitwise_zero_zero fuel]
          rfl
      | succ exponent =>
          unfold powTwo Word.bitwise
          rw [show 2 * powTwo exponent / 2 = powTwo exponent by
            rw [Nat.two_mul]
            exact evenDoubleDivTwo _]
          rw [show 2 * powTwo exponent % 2 = 0 by
            rw [Nat.two_mul]
            exact evenDoubleModTwo _]
          rw [doublePredDivTwo (powTwo exponent) (powTwo_pos exponent)]
          rw [inductionHypothesis exponent]
          rfl

theorem word_sub_one_val (cap : Word) (hpositive : 0 < cap.val) :
    (Word.sub cap (Word.ofNat 1)).val = cap.val - 1 := by
  change ((cap.val + wordModulus - (Word.ofNat 1).val) % wordModulus) = cap.val - 1
  rw [show (Word.ofNat 1).val = 1 by rfl]
  cases hcap : cap.val with
  | zero => exact False.elim (Nat.not_lt_zero _ (hcap ▸ hpositive))
  | succ predecessor =>
      rw [Nat.succ_sub_one]
      rw [show predecessor + 1 + wordModulus - 1 = predecessor + wordModulus by
        rw [Nat.add_assoc, Nat.add_comm 1 wordModulus, ← Nat.add_assoc,
          addSubCancel]]
      rw [modAddModulus, modEqOfLt]
      have predecessorBound : predecessor + 1 < wordModulus := hcap ▸ cap.isLt
      exact Nat.lt_trans (Nat.lt_succ_self predecessor) predecessorBound

theorem powerOfTwo_bitAnd_pred_eq_zero (cap : Word) (hcap : PositivePowerOfTwo cap)
    (_hcapSign : cap.val < wordSignBit) :
    Word.bitAnd cap (Word.sub cap (Word.ofNat 1)) = Word.ofNat 0 := by
  rcases hcap with ⟨exponent, hexponent⟩
  apply Fin.ext
  change Word.bitwise Bool.and wordBits cap.val
      (Word.sub cap (Word.ofNat 1)).val % wordModulus = 0
  rw [word_sub_one_val cap (hexponent ▸ powTwo_pos exponent), hexponent,
    bitwise_powTwo_pred_zero]
  rfl

theorem slotAddress_word_emptyCone (base slot : Word) :
    Word.add base (Word.mul (Word.ofNat 2) slot) =
      Word.ofNat (keyCoordinate base slot.val) := by
  apply Fin.ext
  change (base.val + ((2 % wordModulus) * slot.val) % wordModulus) % wordModulus =
    (base.val + 2 * slot.val) % wordModulus
  rw [modEqOfLt (by decide : 2 < wordModulus)]
  exact modAddAbsorbRight base.val (2 * slot.val) wordModulus

theorem slotValueAddress_word_emptyCone (base slot : Word) :
    Word.add (Word.add base (Word.mul (Word.ofNat 2) slot)) (Word.ofNat 1) =
      Word.ofNat (valueCoordinate base slot.val) := by
  rw [slotAddress_word_emptyCone]
  apply Fin.ext
  change (((keyCoordinate base slot.val) % wordModulus + (1 % wordModulus)) % wordModulus) =
    (valueCoordinate base slot.val) % wordModulus
  rw [modEqOfLt (by decide : 1 < wordModulus), modAddAbsorbLeft]
  rfl

theorem execList_two_stores_return (context : Context) (policy : StackPolicy) (depth : Nat)
    (environment : Environment) (memory₀ memory₁ memory₂ memory₃ : Memory)
    (first second : Stmt) (result : Expr) (raw : Word)
    (trace₁ trace₂ trace₃ : Trace)
    (hfirst : ExecStmt context policy depth environment memory₀ first
      (.next environment memory₁) trace₁)
    (hsecond : ExecStmt context policy depth environment memory₁ second
      (.next environment memory₂) trace₂)
    (hreturn : EvalExpr context policy depth environment memory₂ result
      (.value raw memory₃) trace₃) :
    ExecList context policy depth environment memory₀ [first, second, .return_ result]
      (.returned raw memory₃) (trace₁ ++ (trace₂ ++ trace₃)) :=
  ExecList.next hfirst
    (ExecList.next hsecond (ExecList.returned (ExecStmt.return_ hreturn) (rest := [])))

theorem store_preserves_outside (memory next : Memory) (address value : Word)
    (coordinate : Nat) (hcoordinate : Memory.coordinate address = some coordinate)
    (hstore : memory.store address value = some next) :
    next.owned = memory.owned ∧
      ∀ other, other ≠ coordinate → next.initialized other = memory.initialized other := by
  exact ⟨Memory.store_ownership memory next address value hstore,
    fun other hne => Memory.store_other memory next address value coordinate other
      hcoordinate hstore hne⟩

@[simp] theorem storeAddresses_append (left right : Trace) :
    storeAddresses (left ++ right) = storeAddresses left ++ storeAddresses right := by
  induction left with
  | nil => rfl
  | cons event rest inductionHypothesis =>
      cases event with
      | store address => exact congrArg (List.cons address) inductionHypothesis
      | arithmetic _ | comparison _ | bitwise _ | load _ | call _ _ =>
          exact inductionHypothesis

theorem storeCount_eq_storeAddresses_length (trace : Trace) :
    storeCount trace = (storeAddresses trace).length := by
  induction trace with
  | nil => rfl
  | cons event rest inductionHypothesis =>
      cases event with
      | store _ => exact congrArg Nat.succ inductionHypothesis
      | arithmetic _ | comparison _ | bitwise _ | load _ | call _ _ =>
          exact inductionHypothesis

theorem expectedInitStores_length (base cap : Word) :
    (expectedInitStores base cap).length = 2 * cap.val := by
  have fuelLength (slot fuel : Nat) :
      (expectedInitStoresFuel base slot fuel).length = 2 * fuel := by
    induction fuel generalizing slot with
    | zero => rfl
    | succ fuel inductionHypothesis =>
        unfold expectedInitStoresFuel
        change (expectedInitStoresFuel base (slot + 1) fuel).length + 2 = 2 * (fuel + 1)
        rw [inductionHypothesis, Nat.mul_succ]
  unfold expectedInitStores expectedInitStoresFrom
  rw [Nat.sub_zero]
  exact fuelLength 0 cap.val

theorem eval_slot_address (context : Context) (policy : StackPolicy) (depth : Nat)
    (environment : Environment) (memory : Memory) (base slot : Word)
    (hbase : Environment.lookup environment "base" = some base)
    (hslot : Environment.lookup environment "slot" = some slot)
    (hstrideLocal : Environment.lookup environment "TABLE_STRIDE" = none)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2)) :
    EvalExpr context policy depth environment memory slotAddressExpr
      (.value (Word.ofNat (keyCoordinate base slot.val)) memory)
      [.arithmetic .mul, .arithmetic .add] := by
  rw [← slotAddress_word_emptyCone base slot]
  exact EvalExpr.binary (EvalExpr.local hbase)
    (EvalExpr.binary (EvalExpr.constant hstrideLocal hstride) (EvalExpr.local hslot))

theorem eval_slot_value_address (context : Context) (policy : StackPolicy) (depth : Nat)
    (environment : Environment) (memory : Memory) (base slot : Word)
    (hbase : Environment.lookup environment "base" = some base)
    (hslot : Environment.lookup environment "slot" = some slot)
    (hstrideLocal : Environment.lookup environment "TABLE_STRIDE" = none)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2)) :
    EvalExpr context policy depth environment memory slotValueAddressExpr
      (.value (Word.ofNat (valueCoordinate base slot.val)) memory)
      [.arithmetic .mul, .arithmetic .add, .arithmetic .add] := by
  rw [← slotValueAddress_word_emptyCone base slot]
  exact EvalExpr.binary
    (EvalExpr.binary (EvalExpr.local hbase)
      (EvalExpr.binary (EvalExpr.constant hstrideLocal hstride) (EvalExpr.local hslot)))
    EvalExpr.literal

theorem tableSlotSet_execution (context : Context) (slotSet : FunctionDecl)
    (hfunction : context.functions "tableSlotSet" = some slotSet)
    (hshape : TableSlotSetShape slotSet)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (memory : Memory) (base slot key value : Word) (depth : Nat)
    (hkeyOwned : memory.owned (keyCoordinate base slot.val) = true)
    (hvalueOwned : memory.owned (valueCoordinate base slot.val) = true)
    (hkeyCoordinate : keyCoordinate base slot.val < wordSignBit)
    (hvalueCoordinate : valueCoordinate base slot.val < wordSignBit) :
    ∃ after : Memory,
      EvalCall context .unbounded depth "tableSlotSet" [base, slot, key, value] memory
        (.value value after) (slotSetTrace depth base slot) ∧
      after.owned = memory.owned ∧
      after.initialized (keyCoordinate base slot.val) = some key ∧
      after.initialized (valueCoordinate base slot.val) = some value ∧
      (∀ coordinate, coordinate ≠ keyCoordinate base slot.val →
        coordinate ≠ valueCoordinate base slot.val →
        after.initialized coordinate = memory.initialized coordinate) := by
  let keyAddress := Word.ofNat (keyCoordinate base slot.val)
  let valueAddress := Word.ofNat (valueCoordinate base slot.val)
  let afterKey : Memory :=
    { owned := memory.owned
      initialized := fun coordinate =>
        if coordinate = keyCoordinate base slot.val then some key else memory.initialized coordinate }
  let afterValue : Memory :=
    { owned := memory.owned
      initialized := fun coordinate =>
        if coordinate = valueCoordinate base slot.val then some value else afterKey.initialized coordinate }
  have hkeyCoord : Memory.coordinate keyAddress = some (keyCoordinate base slot.val) :=
    coordinate_ofNat_emptyCone _ hkeyCoordinate
  have hvalueCoord : Memory.coordinate valueAddress = some (valueCoordinate base slot.val) :=
    coordinate_ofNat_emptyCone _ hvalueCoordinate
  have hkeyStore : memory.store keyAddress key = some afterKey := by
    unfold Memory.store
    rw [hkeyCoord]
    change (if memory.owned (keyCoordinate base slot.val) = true then some afterKey else none) =
      some afterKey
    rw [hkeyOwned]
    exact if_pos rfl
  have hvalueStore : afterKey.store valueAddress value = some afterValue := by
    unfold Memory.store
    rw [hvalueCoord]
    change (if memory.owned (valueCoordinate base slot.val) = true then some afterValue else none) =
      some afterValue
    rw [hvalueOwned]
    exact if_pos rfl
  have hframe : Environment.ofParameters slotSet.parameters [base, slot, key, value] =
      some [("base", base), ("slot", slot), ("key", key), ("value", value)] := by
    rw [hshape.1]
    rfl
  have hbase : Environment.lookup [("base", base), ("slot", slot), ("key", key),
      ("value", value)] "base" = some base := Environment.lookup_bind_same _ _ _
  have hslot : Environment.lookup [("base", base), ("slot", slot), ("key", key),
      ("value", value)] "slot" = some slot := by
    change Environment.lookup (Environment.bind [("slot", slot), ("key", key),
      ("value", value)] "base" base) "slot" = some slot
    rw [Environment.lookup_bind_other _ "base" "slot" base (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hkey : Environment.lookup [("base", base), ("slot", slot), ("key", key),
      ("value", value)] "key" = some key := by
    change Environment.lookup (Environment.bind [("slot", slot), ("key", key),
      ("value", value)] "base" base) "key" = some key
    rw [Environment.lookup_bind_other _ "base" "key" base (by decide)]
    change Environment.lookup (Environment.bind [("key", key), ("value", value)]
      "slot" slot) "key" = some key
    rw [Environment.lookup_bind_other _ "slot" "key" slot (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hvalue : Environment.lookup [("base", base), ("slot", slot), ("key", key),
      ("value", value)] "value" = some value := by
    change Environment.lookup (Environment.bind [("slot", slot), ("key", key),
      ("value", value)] "base" base) "value" = some value
    rw [Environment.lookup_bind_other _ "base" "value" base (by decide)]
    change Environment.lookup (Environment.bind [("key", key), ("value", value)]
      "slot" slot) "value" = some value
    rw [Environment.lookup_bind_other _ "slot" "value" slot (by decide)]
    change Environment.lookup (Environment.bind [("value", value)] "key" key)
      "value" = some value
    rw [Environment.lookup_bind_other _ "key" "value" key (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hstrideLocal : Environment.lookup [("base", base), ("slot", slot), ("key", key),
      ("value", value)] "TABLE_STRIDE" = none := by
    change Environment.lookup (Environment.bind [("slot", slot), ("key", key),
      ("value", value)] "base" base) "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "base" "TABLE_STRIDE" base (by decide)]
    change Environment.lookup (Environment.bind [("key", key), ("value", value)]
      "slot" slot) "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "slot" "TABLE_STRIDE" slot (by decide)]
    change Environment.lookup (Environment.bind [("value", value)] "key" key)
      "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "key" "TABLE_STRIDE" key (by decide)]
    change Environment.lookup (Environment.bind [] "value" value) "TABLE_STRIDE" = none
    rw [Environment.lookup_bind_other _ "value" "TABLE_STRIDE" value (by decide)]
    rfl
  have hfirst : ExecStmt context .unbounded (depth + 1)
      [("base", base), ("slot", slot), ("key", key), ("value", value)] memory
      (.store slotAddressExpr (.name "key"))
      (.next [("base", base), ("slot", slot), ("key", key), ("value", value)] afterKey)
      [.arithmetic .mul, .arithmetic .add, .store keyAddress] := by
    exact ExecStmt.store
      (eval_slot_address context .unbounded (depth + 1) _ memory base slot hbase hslot
        hstrideLocal hstride)
      (EvalExpr.local hkey) hkeyStore
  have hsecond : ExecStmt context .unbounded (depth + 1)
      [("base", base), ("slot", slot), ("key", key), ("value", value)] afterKey
      (.store slotValueAddressExpr (.name "value"))
      (.next [("base", base), ("slot", slot), ("key", key), ("value", value)] afterValue)
      [.arithmetic .mul, .arithmetic .add, .arithmetic .add, .store valueAddress] := by
    exact ExecStmt.store
      (eval_slot_value_address context .unbounded (depth + 1) _ afterKey base slot hbase hslot
        hstrideLocal hstride)
      (EvalExpr.local hvalue) hvalueStore
  have hbody : ExecList context .unbounded (depth + 1)
      [("base", base), ("slot", slot), ("key", key), ("value", value)] memory slotSet.body
      (.returned value afterValue)
      ([.arithmetic .mul, .arithmetic .add, .store keyAddress] ++
       [.arithmetic .mul, .arithmetic .add, .arithmetic .add, .store valueAddress] ++ []) := by
    rw [hshape.2]
    exact execList_two_stores_return context .unbounded (depth + 1) _ memory afterKey afterValue
      afterValue _ _ _ value _ _ [] hfirst hsecond (EvalExpr.local hvalue)
  refine ⟨afterValue, ?_, ?_, ?_, ?_, ?_⟩
  · exact EvalCall.run (context := context) (policy := .unbounded) (depth := depth)
      hfunction hframe True.intro hbody
  · rfl
  · change (if keyCoordinate base slot.val = valueCoordinate base slot.val then some value
      else if keyCoordinate base slot.val = keyCoordinate base slot.val then some key
      else memory.initialized (keyCoordinate base slot.val)) = some key
    rw [if_neg (Nat.ne_of_lt (keyCoordinate_lt_valueCoordinate base slot.val)), if_pos rfl]
  · change (if valueCoordinate base slot.val = valueCoordinate base slot.val then some value
      else afterKey.initialized (valueCoordinate base slot.val)) = some value
    rw [if_pos rfl]
  · intro coordinate hneKey hneValue
    change (if coordinate = valueCoordinate base slot.val then some value
      else if coordinate = keyCoordinate base slot.val then some key
      else memory.initialized coordinate) = memory.initialized coordinate
    rw [if_neg hneValue, if_neg hneKey]

theorem tableInitGo_execution (context : Context) (slotSet initGo : FunctionDecl)
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
      PreservesOutsideSuffix memory after base cap i := by
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
    refine ⟨memory, [.call "tableInitGo" (depth + 1), .comparison .ge], ?_, ?_, ?_, ?_⟩
    · exact EvalCall.run hinitGo hframe True.intro hbody
    · intro slot lower upper
      exact False.elim (Nat.not_lt_of_ge lower upper)
    · unfold storeAddresses expectedInitStoresFrom
      rw [Nat.sub_self]
      rfl
    · exact ⟨rfl, fun coordinate _ => rfl⟩
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
          simpa [Nat.mul_succ] using Nat.mul_le_mul_left 2 (Nat.succ_le_of_lt hlt)
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
      ⟨after, recursiveTrace, hrecursive, hsuffix, hstores, hrecursiveFrame⟩
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
    let trace := .call "tableInitGo" (depth + 1) ::
      ([.comparison .ge] ++ (slotSetTrace (depth + 1) base (Word.ofNat i) ++
        ([.arithmetic .add] ++ recursiveTrace)))
    refine ⟨after, trace, ?_, ?_, ?_, ?_⟩
    · exact EvalCall.run hinitGo hframe True.intro hbody
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
    · simp only [trace, storeAddresses, slotSetTrace, storeAddresses_append,
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

theorem tableCapacityOk_execution (context : Context) (capacityOk : FunctionDecl)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hshape : TableCapacityOkShape capacityOk) (memory : Memory) (cap : Word) (depth : Nat)
    (hcap : PositivePowerOfTwo cap) (hcapSign : cap.val < wordSignBit) :
    EvalCall context .unbounded depth "tableCapacityOk" [cap] memory
      (.value (Word.ofNat 1) memory)
      [.call "tableCapacityOk" (depth + 1), .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq] := by
  let environment : Environment := [("cap", cap)]
  have hframe : Environment.ofParameters capacityOk.parameters [cap] = some environment := by
    rw [hshape.1]
    rfl
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    simp only [environment]
    exact Environment.lookup_bind_same _ _ _
  have hcapPositive := hcap.pos
  have hleZero : Word.le cap (Word.ofNat 0) = Word.ofNat 0 := by
    have capWord : Word.ofNat cap.val = cap := Fin.ext (ofNat_val_emptyCone _ hcapSign)
    exact Eq.mp (congrArg (fun left => Word.le left (Word.ofNat 0) = Word.ofNat 0) capWord)
      (wordLe_ofNat_pos_zero cap.val hcapSign hcapPositive)
  have hcondition : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .le (.name "cap") (.literal 0))
      (.value (Word.ofNat 0) memory) [.comparison .le] := by
    rw [← hleZero]
    exact EvalExpr.binary (operator := .le) (EvalExpr.local hcapLocal) EvalExpr.literal
  have hif : ExecStmt context .unbounded (depth + 1) environment memory
      (.if_ (.binary .le (.name "cap") (.literal 0)) [.return_ (.literal 0)] none)
      (.next environment memory) [.comparison .le] := by
    apply ExecStmt.ifFalse hcondition
    · exact fun h => h rfl
    · exact ExecList.nil
  have hmask := powerOfTwo_bitAnd_pred_eq_zero cap hcap hcapSign
  have hsub : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .sub (.name "cap") (.literal 1))
      (.value (Word.sub cap (Word.ofNat 1)) memory) [.arithmetic .sub] := by
    exact EvalExpr.binary (operator := .sub) (EvalExpr.local hcapLocal) EvalExpr.literal
  have hbit : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .bitAnd (.name "cap") (.binary .sub (.name "cap") (.literal 1)))
      (.value (Word.ofNat 0) memory) [.arithmetic .sub, .bitwise .bitAnd] := by
    rw [← hmask]
    exact EvalExpr.binary (operator := .bitAnd) (EvalExpr.local hcapLocal) hsub
  have heqOne : Word.eq (Word.ofNat 0) (Word.ofNat 0) = Word.ofNat 1 := by
    rfl
  have heq : EvalExpr context .unbounded (depth + 1) environment memory
      (.binary .eq
        (.binary .bitAnd (.name "cap") (.binary .sub (.name "cap") (.literal 1)))
        (.literal 0))
      (.value (Word.ofNat 1) memory)
      [.arithmetic .sub, .bitwise .bitAnd, .comparison .eq] := by
    rw [← heqOne]
    exact EvalExpr.binary (operator := .eq) hbit (EvalExpr.literal (literalValue := 0))
  have hnonzero : Word.nonzero (Word.ofNat 1) := by
    intro equal
    exact Nat.noConfusion (congrArg Fin.val equal)
  have hconditional : EvalExpr context .unbounded (depth + 1) environment memory
      (.conditional (.literal 1)
        (.binary .eq
          (.binary .bitAnd (.name "cap") (.binary .sub (.name "cap") (.literal 1)))
          (.literal 0))
        (.literal 0))
      (.value (Word.ofNat 1) memory)
      [.arithmetic .sub, .bitwise .bitAnd, .comparison .eq] := by
    exact EvalExpr.conditionalTrue heq hnonzero EvalExpr.literal
  have hreturn : ExecStmt context .unbounded (depth + 1) environment memory
      (.return_ (.conditional (.literal 1)
        (.binary .eq
          (.binary .bitAnd (.name "cap") (.binary .sub (.name "cap") (.literal 1)))
          (.literal 0))
        (.literal 0)))
      (.returned (Word.ofNat 1) memory)
      [.arithmetic .sub, .bitwise .bitAnd, .comparison .eq] :=
    ExecStmt.return_ hconditional
  have hbody : ExecList context .unbounded (depth + 1) environment memory capacityOk.body
      (.returned (Word.ofNat 1) memory)
      ([.comparison .le] ++ [.arithmetic .sub, .bitwise .bitAnd, .comparison .eq]) := by
    rw [hshape.2]
    exact ExecList.next hif (ExecList.returned hreturn)
  exact EvalCall.run hcapacityOk hframe True.intro hbody

theorem tableInit_execution (context : Context) (slotSet initGo init capacityOk : FunctionDecl)
    (hslotSet : context.functions "tableSlotSet" = some slotSet)
    (hinitGo : context.functions "tableInitGo" = some initGo)
    (hinit : context.functions "tableInit" = some init)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hslotSetShape : TableSlotSetShape slotSet) (hinitGoShape : TableInitGoShape initGo)
    (hinitShape : TableInitShape init) (hcapacityOkShape : TableCapacityOkShape capacityOk)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap : Word) (hcap : PositivePowerOfTwo cap)
    (hbacking : ValidBacking memory base cap) :
    ∃ after trace, ConstructorExecution context memory after base cap trace := by
  have hcapSign : cap.val < wordSignBit := cap_below_signBit_of_backing memory base cap hbacking
  let environment : Environment := [("base", base), ("cap", cap)]
  have hframe : Environment.ofParameters init.parameters [base, cap] = some environment := by
    rw [hinitShape.1]
    rfl
  have hbase : Environment.lookup environment "base" = some base := by
    simp only [environment]
    exact Environment.lookup_bind_same _ _ _
  have hcapLocal : Environment.lookup environment "cap" = some cap := by
    simp only [environment]
    change Environment.lookup (Environment.bind [("cap", cap)] "base" base)
      "cap" = some cap
    rw [Environment.lookup_bind_other _ "base" "cap" base (by decide)]
    exact Environment.lookup_bind_same _ _ _
  have hcapacityArgs : EvalArgs context .unbounded 1 environment memory [.name "cap"]
      (.value [cap] memory) [] := by
    exact EvalArgs.cons (EvalExpr.local hcapLocal) EvalArgs.nil
  have hcapacityCall := tableCapacityOk_execution context capacityOk hcapacityOk hcapacityOkShape
    memory cap 1 hcap hcapSign
  have hcapacityExpr : EvalExpr context .unbounded 1 environment memory
      (.call "tableCapacityOk" [.name "cap"]) (.value (Word.ofNat 1) memory)
      [.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq] := by
    exact EvalExpr.call hcapacityArgs hcapacityCall
  have heqZero : Word.eq (Word.ofNat 1) (Word.ofNat 0) = Word.ofNat 0 := by
    rfl
  have hguardExpr : EvalExpr context .unbounded 1 environment memory
      (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
      (.value (Word.ofNat 0) memory)
      [.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] := by
    rw [← heqZero]
    exact EvalExpr.binary (operator := .eq) hcapacityExpr
      (EvalExpr.literal (literalValue := 0))
  have hguard : ExecStmt context .unbounded 1 environment memory
      (.if_ (.binary .eq (.call "tableCapacityOk" [.name "cap"]) (.literal 0))
        [.return_ (.name "TABLE_EMPTY")] none)
      (.next environment memory)
      [.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] := by
    apply ExecStmt.ifFalse hguardExpr
    · exact fun h => h rfl
    · exact ExecList.nil
  rcases tableInitGo_execution context slotSet initGo hslotSet hinitGo hslotSetShape
    hinitGoShape hstride hempty memory base cap 0 1 (Nat.zero_le cap.val) hbacking with
    ⟨after, goTrace, hgo, hsuffix, hstores, hgoFrame⟩
  have hgoArgs : EvalArgs context .unbounded 1 environment memory
      [.name "base", .name "cap", .literal 0]
      (.value [base, cap, Word.ofNat 0] memory) [] := by
    exact EvalArgs.cons (EvalExpr.local hbase)
      (EvalArgs.cons (EvalExpr.local hcapLocal) (EvalArgs.cons EvalExpr.literal EvalArgs.nil))
  have hreturn : ExecStmt context .unbounded 1 environment memory
      (.return_ (.call "tableInitGo" [.name "base", .name "cap", .literal 0]))
      (.returned (Word.ofNat 0) after) goTrace := by
    exact ExecStmt.return_ (EvalExpr.call hgoArgs hgo)
  have hbody : ExecList context .unbounded 1 environment memory init.body
      (.returned (Word.ofNat 0) after)
      ([.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] ++ goTrace) := by
    rw [hinitShape.2]
    exact ExecList.next hguard (ExecList.returned hreturn)
  let trace := .call "tableInit" 1 ::
    ([.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub,
      .bitwise .bitAnd, .comparison .eq, .comparison .eq] ++ goTrace)
  have hexactStores : storeAddresses trace = expectedInitStores base cap := by
    unfold trace
    change storeAddresses
      ([.call "tableCapacityOk" 2, .comparison .le, .arithmetic .sub,
        .bitwise .bitAnd, .comparison .eq, .comparison .eq] ++ goTrace) =
          expectedInitStores base cap
    rw [storeAddresses_append]
    change [] ++ storeAddresses goTrace = expectedInitStores base cap
    rw [hstores]
    rfl
  have hstoreCount : (resources trace).stores = 2 * cap.val := by
    rw [resources_stores]
    rw [storeCount_eq_storeAddresses_length]
    rw [hexactStores]
    exact expectedInitStores_length base cap
  refine ⟨after, trace, ?_⟩
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact EvalCall.run hinit hframe True.intro hbody
  · exact ⟨fun slot hslot => (hsuffix slot (Nat.zero_le slot) hslot).1,
      fun slot hslot => (hsuffix slot (Nat.zero_le slot) hslot).2⟩
  · exact hexactStores
  · exact hstoreCount
  · constructor
    · intro coordinate
      exact congrFun hgoFrame.1 coordinate
    · intro coordinate houtside
      exact hgoFrame.2 coordinate (by
        intro hsuffixCoordinate
        apply houtside
        exact initSuffix_implies_extent base cap coordinate hsuffixCoordinate)

theorem offset_key_or_value (offset : Nat) :
    ∃ slot, offset = 2 * slot ∨ offset = 2 * slot + 1 := by
  induction offset with
  | zero => exact ⟨0, Or.inl rfl⟩
  | succ offset inductionHypothesis =>
      rcases inductionHypothesis with ⟨slot, even | odd⟩
      · exact ⟨slot, Or.inr (congrArg Nat.succ even)⟩
      · refine ⟨slot + 1, Or.inl ?_⟩
        rw [odd, Nat.mul_succ]

theorem validExtent_of_constructor (before after : Memory) (base cap : Word)
    (backing : ValidBacking before base cap)
    (execution : InitializedEmptyLayout after base cap)
    (ownership : SameOwnership before after) : ValidExtent after base cap := by
  constructor
  · exact ⟨backing.1, backing.2.1, fun offset offsetBound => by
      rw [ownership (base.val + offset)]
      exact backing.2.2 offset offsetBound⟩
  · intro offset offsetBound
    rcases offset_key_or_value offset with ⟨slot, even | odd⟩
    · have slotBound : slot < cap.val := by
        by_contra notLess
        have capLe : cap.val ≤ slot := Nat.le_of_not_gt notLess
        exact Nat.not_lt_of_ge (Nat.mul_le_mul_left 2 capLe) (even ▸ offsetBound)
      rw [even]
      exact congrArg Option.isSome (execution.1 slot slotBound)
    · have slotBound : slot < cap.val := by
        by_contra notLess
        have capLe : cap.val ≤ slot := Nat.le_of_not_gt notLess
        have doubled := Nat.mul_le_mul_left 2 capLe
        exact Nat.not_lt_of_ge (Nat.le_trans doubled (Nat.le_add_right (2 * slot) 1))
          (odd ▸ offsetBound)
      rw [odd]
      exact congrArg Option.isSome (execution.2 slot slotBound)

theorem constructor_execution_of_shapes
    (context : Context) (slotSet initGo init capacityOk : FunctionDecl)
    (hslotSet : context.functions "tableSlotSet" = some slotSet)
    (hinitGo : context.functions "tableInitGo" = some initGo)
    (hinit : context.functions "tableInit" = some init)
    (hcapacityOk : context.functions "tableCapacityOk" = some capacityOk)
    (hslotSetShape : TableSlotSetShape slotSet)
    (hinitGoShape : TableInitGoShape initGo)
    (hinitShape : TableInitShape init)
    (hcapacityOkShape : TableCapacityOkShape capacityOk)
    (hstride : context.constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : context.constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap : Word)
    (hcapacity : PositivePowerOfTwo cap)
    (hbacking : ValidBacking memory base cap) :
    ∃ memory' trace,
      ConstructorExecution context memory memory' base cap trace ∧
      RepSong memory' base cap (Word.ofNat 0) [] (fun _ => Word.ofNat 0) := by
  rcases tableInit_execution context slotSet initGo init capacityOk
    hslotSet hinitGo hinit hcapacityOk hslotSetShape hinitGoShape hinitShape hcapacityOkShape
    hstride hempty memory base cap hcapacity hbacking with ⟨memory', trace, execution⟩
  have hextent := validExtent_of_constructor memory memory' base cap hbacking execution.layout
    execution.frame.1
  have rep := repSong_empty_of_layout memory' base cap (Word.ofNat 0) hcapacity hextent
    execution.layout.1
  exact ⟨memory', trace, execution, rep⟩

theorem constructor_execution_of_certificate
    (value : CertifiedDecoded) (program : Program)
    (derivation : DecodeDerivation
      Slang.Generated.TableDerivation.tableSource
      Slang.Generated.TableDerivation.stdSource value)
    (programEq : value.val.program = program)
    (selectedSyntax : Slang.SourceAudit.ExpectedSelectedSyntax program = true)
    (closure : ParsesClosure Slang.Generated.TableDerivation.entryBundle program)
    (slotSet initGo init capacityOk : FunctionDecl)
    (hslotSet : (checkedContext program).functions "tableSlotSet" = some slotSet)
    (hinitGo : (checkedContext program).functions "tableInitGo" = some initGo)
    (hinit : (checkedContext program).functions "tableInit" = some init)
    (hcapacityOk : (checkedContext program).functions "tableCapacityOk" = some capacityOk)
    (hslotSetShape : TableSlotSetShape slotSet)
    (hinitGoShape : TableInitGoShape initGo)
    (hinitShape : TableInitShape init)
    (hcapacityOkShape : TableCapacityOkShape capacityOk)
    (hstride : (checkedContext program).constants "TABLE_STRIDE" = some (Word.ofNat 2))
    (hempty : (checkedContext program).constants "TABLE_EMPTY" = some tableEmpty)
    (memory : Memory) (base cap : Word)
    (hcapacity : PositivePowerOfTwo cap)
    (hbacking : ValidBacking memory base cap) :
    ∃ memory' trace,
      Decode.decodeClosure
        (encodeBytes Slang.Generated.TableDerivation.tableSource)
        (encodeBytes Slang.Generated.TableDerivation.stdSource) = .ok value ∧
      EvalCall (checkedContext program) .unbounded 0 "tableInit" [base, cap] memory
        (.value (Word.ofNat 0) memory') trace ∧
      RepSong memory' base cap (Word.ofNat 0) [] (fun _ => Word.ofNat 0) ∧
      storeAddresses trace = expectedInitStores base cap ∧
      (resources trace).stores = 2 * cap.val ∧
      InitializedEmptyLayout memory' base cap ∧
      PreservesOutside memory memory' base cap := by
  subst program
  have source := Slang.SourceAudit.source_binding derivation selectedSyntax closure
  rcases constructor_execution_of_shapes (checkedContext value.val.program) slotSet initGo init
    capacityOk hslotSet hinitGo hinit hcapacityOk hslotSetShape hinitGoShape hinitShape
    hcapacityOkShape hstride hempty memory base cap hcapacity hbacking with
    ⟨memory', trace, execution, rep⟩
  exact ⟨memory', trace, source.1, execution.call, rep, execution.exactStores,
    execution.storeCount, execution.layout, execution.frame⟩

end Slang.Table
