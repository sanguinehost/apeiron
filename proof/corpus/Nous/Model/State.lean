import Apeiron.Trichotomy.AnyRing
import Mathlib.Tactic

/-!
# Token State Fold Over a Commutative Ring

This module formalizes the sequential state transition step and list fold over
`Apeiron.Trichotomy.AnyRing.RingCell`.

Given a quadratic parameter `σ` in a commutative ring `R`, the transition step
multiplies an accumulator state `z` by an incoming token point `w`:
`tokenStep σ z w := ringMul σ z w`.

The sequential fold across a list of token points initialized at the unit element
`ringOne = (1, 0)` is:
`tokenFold σ xs := xs.foldl (ringMul σ) ringOne`.

Because `ringMul σ` is associative (`ringMul_assoc`) and has identity `ringOne`
(`ringOne_mul`, `ringMul_one`), the fold over concatenated lists splits
multiplicatively:
`List.foldl (ringMul σ) ringOne (xs ++ ys) =
  ringMul σ (List.foldl (ringMul σ) ringOne xs) (List.foldl (ringMul σ) ringOne ys)`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Trichotomy.AnyRing

variable {R : Type*} [CommRing R]

/-- A single sequential token transition step multiplying an accumulator by an input token. -/
def tokenStep (σ : R) (z w : RingCell R) : RingCell R :=
  ringMul σ z w

/-- Sequential fold of a list of token points starting from the unit element `ringOne`. -/
def tokenFold (σ : R) (xs : List (RingCell R)) : RingCell R :=
  xs.foldl (ringMul σ) ringOne

/-- The token step is commutative, inheriting commutativity from `ringMul σ`. -/
theorem tokenStep_comm (σ : R) (z w : RingCell R) :
    tokenStep σ z w = tokenStep σ w z :=
  ringMul_comm σ z w

/-- The token step is associative, inheriting associativity from `ringMul σ`. -/
theorem tokenStep_assoc (σ : R) (z w v : RingCell R) :
    tokenStep σ (tokenStep σ z w) v = tokenStep σ z (tokenStep σ w v) :=
  ringMul_assoc σ z w v

/-- The unit element `ringOne` is a left identity for `tokenStep`. -/
theorem tokenStep_one_mul (σ : R) (z : RingCell R) :
    tokenStep σ ringOne z = z :=
  ringOne_mul σ z

/-- The unit element `ringOne` is a right identity for `tokenStep`. -/
theorem tokenStep_mul_one (σ : R) (z : RingCell R) :
    tokenStep σ z ringOne = z :=
  ringMul_one σ z

/-- Folding from an arbitrary accumulator equals multiplying the accumulator by the fold
from the unit element `ringOne`. -/
theorem foldl_ringMul_accum (σ : R) (xs : List (RingCell R)) (acc : RingCell R) :
    xs.foldl (ringMul σ) acc = ringMul σ acc (xs.foldl (ringMul σ) ringOne) := by
  induction xs generalizing acc with
  | nil =>
    simp only [List.foldl_nil]
    exact (ringMul_one σ acc).symm
  | cons x rest ih =>
    simp only [List.foldl_cons]
    rw [ih (ringMul σ acc x), ih (ringMul σ ringOne x)]
    rw [ringOne_mul σ x]
    rw [ringMul_assoc σ acc x]

/-- The empty list fold produces the unit element `ringOne`. -/
@[simp] theorem token_step_fold_nil (σ : R) :
    List.foldl (ringMul σ) ringOne ([] : List (RingCell R)) = ringOne :=
  rfl

/-- The sequential fold across concatenated lists splits into the product of the individual folds. -/
theorem token_step_fold (σ : R) (xs ys : List (RingCell R)) :
    List.foldl (ringMul σ) ringOne (xs ++ ys) =
      ringMul σ (List.foldl (ringMul σ) ringOne xs) (List.foldl (ringMul σ) ringOne ys) := by
  rw [List.foldl_append]
  exact foldl_ringMul_accum σ ys (List.foldl (ringMul σ) ringOne xs)

/-- Contract pairing the append-split property and the empty-list base case. -/
theorem token_step_fold_contract (σ : R) (xs ys : List (RingCell R)) :
    List.foldl (ringMul σ) ringOne (xs ++ ys) =
      ringMul σ (List.foldl (ringMul σ) ringOne xs) (List.foldl (ringMul σ) ringOne ys) ∧
    List.foldl (ringMul σ) ringOne [] = ringOne :=
  ⟨token_step_fold σ xs ys, token_step_fold_nil σ⟩

/-- The empty list fold with `tokenFold` produces `ringOne`. -/
@[simp] theorem tokenFold_nil (σ : R) :
    tokenFold σ [] = ringOne :=
  rfl

/-- `tokenFold` over concatenated lists splits through `ringMul σ`. -/
theorem tokenFold_append (σ : R) (xs ys : List (RingCell R)) :
    tokenFold σ (xs ++ ys) = ringMul σ (tokenFold σ xs) (tokenFold σ ys) :=
  token_step_fold σ xs ys

end Nous.Model
