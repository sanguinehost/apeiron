/-
# The memo key: the computation identity of the canonical re-root

A memo key is the tuple a memo table hashes on: which source reference the
computation came from, the operation name, the bound context, the precedence,
the resource limits, the input state handles, the algebra parameter `σ`, the
face, and the address cell. The address cell is a `RingCell Slang.Word`, so the
key is rooted in the canonical re-root: `Slang.Source.SourceRef` for the source,
`Slang.Arena.StateHandle` for the input states, `Slang.Word` for the word
carrier and `Apeiron.Trichotomy.AnyRing.RingCell` for the two-slot point. `σ` is
a field of every key, so there is no path that omits it.

`MemoKey.equal` compares every data field. Its correctness, `MemoKey.equal_iff`,
and the `DecidableEq` instance are built by explicit case analysis over the
fields; the proof field `addressAdmitted` is discharged by definitional proof
irrelevance after its dependent data (`face`, `addressCell`) is substituted.
Nothing here reaches `propext`, `Quot.sound` or `Classical.choice`, so the
module audits with the empty axiom cone.
-/
import Slang.Source.Arena
import Slang.Arena.StateResult
import Slang.Refinement.WordRing
import Apeiron.Trichotomy.AnyRing

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Slang.Memo

open Slang

/-- The commutative-ring structure on the 64-bit word carrier. It exists only so that
`Apeiron.Trichotomy.AnyRing.RingCell Slang.Word` is a well-formed type; it reuses the axiom-free
laws of `Slang.Refinement.wordRing` and adds no arithmetic. The instance is private because
`RingCell` ignores its argument in the body, so the key never computes through it. -/
private instance wordCommRing : CommRing Word where
  zero := Slang.Refinement.wordRing.zero
  add := Slang.Refinement.wordRing.add
  neg := Slang.Refinement.wordRing.neg
  sub := Slang.Refinement.wordRing.sub
  mul := Slang.Refinement.wordRing.mul
  one := Slang.Refinement.wordRing.one
  add_assoc := Slang.Refinement.wordRing.add_assoc
  zero_add := Slang.Refinement.wordRing.add_zero_left
  add_zero := Slang.Refinement.wordRing.add_zero_right
  add_comm := Slang.Refinement.wordRing.add_comm
  nsmul := fun count value =>
    Nat.rec Slang.Refinement.wordRing.zero
      (fun _ total => Slang.Refinement.wordRing.add total value) count
  nsmul_zero := fun _ => rfl
  nsmul_succ := fun _ _ => rfl
  neg_add_cancel := fun value =>
    Eq.trans (Slang.Refinement.wordRing.add_comm _ _)
      (Slang.Refinement.wordRing.add_neg_right value)
  sub_eq_add_neg := Slang.Refinement.wordRing.sub_eq_add_neg
  zsmul := fun integer value =>
    match integer with
    | .ofNat count =>
        Nat.rec Slang.Refinement.wordRing.zero
          (fun _ total => Slang.Refinement.wordRing.add total value) count
    | .negSucc count =>
        Slang.Refinement.wordRing.neg
          (Nat.rec Slang.Refinement.wordRing.zero
            (fun _ total => Slang.Refinement.wordRing.add total value) (count + 1))
  zsmul_zero' := fun _ => rfl
  zsmul_succ' := fun _ _ => rfl
  zsmul_neg' := fun _ _ => rfl
  mul_assoc := Slang.Refinement.wordRing.mul_assoc
  one_mul := Slang.Refinement.wordRing.mul_one_left
  mul_one := Slang.Refinement.wordRing.mul_one_right
  zero_mul := by
    intro value
    change Word.mul (Word.ofNat 0) value = Word.ofNat 0
    apply Fin.ext
    exact congrArg (fun natural => natural % wordModulus) (Nat.zero_mul value.val)
  mul_zero := by
    intro value
    change Word.mul value (Word.ofNat 0) = Word.ofNat 0
    apply Fin.ext
    exact congrArg (fun natural => natural % wordModulus) (Nat.mul_zero value.val)
  left_distrib := Slang.Refinement.wordRing.mul_add
  right_distrib := Slang.Refinement.wordRing.add_mul
  mul_comm := Slang.Refinement.wordRing.mul_comm

/-- The three faces of the split algebra `R[x]/(x² - σ)`. -/
inductive Face where
  | elliptic
  | parabolic
  | hyperbolic
  deriving DecidableEq

/-- The resource limits carried by a memo key. -/
structure Limits where
  recursion : Nat
  tokens : Nat
  bytes : Nat
  stack : Nat
  results : Nat
  deriving DecidableEq

/-- The admissible address cells for each face. The elliptic face requires both coordinates below
`2 ^ 32`; the parabolic and hyperbolic faces admit every cell. The bound is written `Nat.pow 2 32`
rather than `2 ^ 32` because Mathlib's `instHPow` for `Nat`, in scope through the `Apeiron` import,
depends on `propext`, and the empty axiom cone forbids that. -/
def FaceDomain (face : Face) (cell : Apeiron.Trichotomy.AnyRing.RingCell Slang.Word) : Prop :=
  match face with
  | .elliptic => cell.1.val < Nat.pow 2 32 ∧ cell.2.val < Nat.pow 2 32
  | .parabolic => True
  | .hyperbolic => True

/-- The data that identifies one memoized computation before admission. -/
structure MemoInput (sources : Slang.Source.SourceArena) {stateCapacity : Nat}
    (states : Slang.Arena.StateArena stateCapacity) where
  source : Slang.Source.SourceRef sources
  operation : String
  context : List (String × Slang.Word)
  precedence : Nat
  limits : Limits
  inputStates : List (Slang.Arena.StateHandle states)
  addressCell : Apeiron.Trichotomy.AnyRing.RingCell Slang.Word

/-- An input together with the proof that its address cell is admissible on the named face. `σ` is
carried as a parameter so the key always carries the algebra parameter. -/
structure AdmittedInput (sources : Slang.Source.SourceArena) {stateCapacity : Nat}
    (states : Slang.Arena.StateArena stateCapacity) (sigma : Slang.Word) (face : Face) where
  input : MemoInput sources states
  admitted : FaceDomain face input.addressCell

/-- The computation identity: one source reference, the operation and its context, the resource
limits, the input state handles, `σ`, the face and the admitted address cell. -/
structure MemoKey (sources : Slang.Source.SourceArena) {stateCapacity : Nat}
    (states : Slang.Arena.StateArena stateCapacity) where
  source : Slang.Source.SourceRef sources
  operation : String
  context : List (String × Slang.Word)
  precedence : Nat
  limits : Limits
  inputStates : List (Slang.Arena.StateHandle states)
  sigma : Slang.Word
  face : Face
  addressCell : Apeiron.Trichotomy.AnyRing.RingCell Slang.Word
  addressAdmitted : FaceDomain face addressCell

/-- Build the key of an admitted input. -/
def mkMemoKey {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} {sigma : Slang.Word} {face : Face}
    (input : AdmittedInput sources states sigma face) : MemoKey sources states where
  source := input.input.source
  operation := input.input.operation
  context := input.input.context
  precedence := input.input.precedence
  limits := input.input.limits
  inputStates := input.input.inputStates
  sigma := sigma
  face := face
  addressCell := input.input.addressCell
  addressAdmitted := input.admitted

/-- Decidable equality on a checked interval: compare the two data fields, then the structure is
equal by definitional proof irrelevance. -/
instance instDecidableEqCheckedInterval {bound : Nat} :
    DecidableEq (Slang.Source.CheckedInterval bound) :=
  fun a b =>
    match Nat.decEq a.start b.start with
    | isTrue hs =>
        match Nat.decEq a.stop b.stop with
        | isTrue ht => isTrue (by cases a; cases b; cases hs; cases ht; rfl)
        | isFalse ht => isFalse (fun h => ht (congrArg Slang.Source.CheckedInterval.stop h))
    | isFalse hs => isFalse (fun h => hs (congrArg Slang.Source.CheckedInterval.start h))

/-- Decidable equality on a checked source reference: compare the file index, the byte interval and
the token interval; the `correspondence` proof is proof-irrelevant. -/
instance instDecidableEqSourceRef (arena : Slang.Source.SourceArena) :
    DecidableEq (Slang.Source.SourceRef arena) :=
  fun a b => by
    cases a with
    | mk af abi ati ac =>
      cases b with
      | mk bf bbi bti bc =>
        cases hf : decEq af bf with
        | isTrue he =>
            cases he
            cases hb : decEq abi bbi with
            | isTrue hb' =>
                cases hb'
                cases ht : decEq ati bti with
                | isTrue ht' => cases ht'; exact isTrue rfl
                | isFalse ht' =>
                    exact isFalse (fun h => ht' (by cases h; rfl))
            | isFalse hb' =>
                exact isFalse (fun h => hb' (by cases h; rfl))
        | isFalse hf' =>
            exact isFalse (fun h => hf' (congrArg Slang.Source.SourceRef.file h))

/-- Decidable equality on a memo key: case over every data field in turn, then the structure is
equal by definitional proof irrelevance of `addressAdmitted`. -/
instance MemoKey.instDecidableEq {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} : DecidableEq (MemoKey sources states) :=
  fun left right => by
    cases left with
    | mk lsource loperation lcontext lprecedence llimits linputStates lsigma lface laddressCell
        laddressAdmitted =>
      cases right with
      | mk rsource roperation rcontext rprecedence rlimits rinputStates rsigma rface raddressCell
          raddressAdmitted =>
        cases h1 : decEq lsource rsource with
        | isTrue e1 =>
            cases e1
            cases h2 : decEq loperation roperation with
            | isTrue e2 =>
                cases e2
                cases h3 : decEq lcontext rcontext with
                | isTrue e3 =>
                    cases e3
                    cases h4 : decEq lprecedence rprecedence with
                    | isTrue e4 =>
                        cases e4
                        cases h5 : decEq llimits rlimits with
                        | isTrue e5 =>
                            cases e5
                            cases h6 : decEq linputStates rinputStates with
                            | isTrue e6 =>
                                cases e6
                                cases h7 : decEq lsigma rsigma with
                                | isTrue e7 =>
                                    cases e7
                                    cases h8 : decEq lface rface with
                                    | isTrue e8 =>
                                        cases e8
                                        cases h9 : decEq laddressCell raddressCell with
                                        | isTrue e9 =>
                                            cases e9
                                            exact isTrue rfl
                                        | isFalse e9 =>
                                            exact isFalse (fun h =>
                                              e9 (congrArg MemoKey.addressCell h))
                                    | isFalse e8 =>
                                        exact isFalse (fun h => e8 (congrArg MemoKey.face h))
                                | isFalse e7 =>
                                    exact isFalse (fun h => e7 (congrArg MemoKey.sigma h))
                            | isFalse e6 =>
                                exact isFalse (fun h => e6 (congrArg MemoKey.inputStates h))
                        | isFalse e5 =>
                            exact isFalse (fun h => e5 (congrArg MemoKey.limits h))
                    | isFalse e4 =>
                        exact isFalse (fun h => e4 (congrArg MemoKey.precedence h))
                | isFalse e3 =>
                    exact isFalse (fun h => e3 (congrArg MemoKey.context h))
            | isFalse e2 =>
                exact isFalse (fun h => e2 (congrArg MemoKey.operation h))
        | isFalse e1 =>
            exact isFalse (fun h => e1 (congrArg MemoKey.source h))

/-- The Bool comparison: `true` exactly when the two keys are equal. -/
def MemoKey.equal {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} (left right : MemoKey sources states) : Bool :=
  match decEq left right with
  | isTrue _ => true
  | isFalse _ => false

/-- `MemoKey.equal` is `true` exactly when the keys are equal. -/
theorem MemoKey.equal_iff {sources : Slang.Source.SourceArena} {stateCapacity : Nat}
    {states : Slang.Arena.StateArena stateCapacity} (left right : MemoKey sources states) :
    MemoKey.equal left right = true ↔ left = right := by
  unfold MemoKey.equal
  cases h : decEq left right with
  | isTrue hh => exact ⟨fun _ => hh, fun _ => rfl⟩
  | isFalse hh =>
      exact ⟨fun hf => False.elim (Bool.false_ne_true hf), fun he => absurd he hh⟩

end Slang.Memo
