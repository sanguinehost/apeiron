import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
# Finite interpretation witnesses

This file checks the `ZMod 2` observer fixture from the security-algebra rebase specification.
It is a finite model only: no declaration connects it to a deployed cipher or claims security.
-/

namespace Apeiron.Security.InterpretationWitnesses

open Function

/-- The finite message and pad domain selected by the interpretation fixture. -/
abbrev Bit := ZMod 2

/-- Additive encryption in the finite fixture. -/
def enc (message key : Bit) : Bit := message + key

/-- A key law records the two nonnegative masses and their normalization. -/
structure KeyLaw where
  mass : Bit → ℚ
  nonnegative : ∀ key, 0 ≤ mass key
  normalized : mass 0 + mass 1 = 1

/-- Ciphertext observation probability for a fixed message. -/
def observationProbability (law : KeyLaw) (ciphertext message : Bit) : ℚ :=
  law.mass (ciphertext - message)

/-- Uniform independent pad law on the two-element fixture. -/
def uniformKeyLaw : KeyLaw where
  mass := fun _ => 1 / 2
  nonnegative := by intro; norm_num
  normalized := by norm_num

/-- The uniform pad makes every ciphertext observation independent of the message. -/
theorem uniform_observer_equality (message₀ message₁ ciphertext : Bit) :
    observationProbability uniformKeyLaw ciphertext message₀ =
      observationProbability uniformKeyLaw ciphertext message₁ := by
  rfl

/-- Biased pad law used by the distinguishing counterexample. -/
def biasedKeyLaw : KeyLaw where
  mass := fun key => if key = 0 then 3 / 4 else 1 / 4
  nonnegative := by
    intro key
    split <;> norm_num
  normalized := by norm_num

/-- The biased law distinguishes messages `0` and `1` at ciphertext `0`. -/
theorem biased_observer_distinguishes :
    observationProbability biasedKeyLaw 0 0 ≠
      observationProbability biasedKeyLaw 0 1 := by
  norm_num [observationProbability, biasedKeyLaw]

/-- `RecoveryTrinity.otp_perfect_secrecy` at the finite fixture's exact source statement. -/
theorem otp_perfect_secrecy (ciphertext message : Bit) :
    ∃! key : Bit, enc message key = ciphertext := by
  refine ⟨ciphertext - message, by simp [enc], ?_⟩
  intro key hkey
  apply add_left_cancel (a := message)
  simpa [enc] using hkey

/-- Unique compatible keys remain available under the biased key law. -/
theorem biased_unique_compatible_key (ciphertext message : Bit) :
    ∃! key : Bit, enc message key = ciphertext :=
  otp_perfect_secrecy ciphertext message

/-- Supplying a fixed key recovers the message exactly. -/
theorem fixed_key_round_trip (message key : Bit) : enc message key - key = message := by
  simp [enc]

/-- A fixed-key additive map has the singleton-fiber property named by the old source. -/
theorem cipher_fiber_is_a_singleton (key ciphertext : Bit) :
    ∃! message : Bit, enc message key = ciphertext := by
  refine ⟨ciphertext - key, by simp [enc], ?_⟩
  intro message hmessage
  apply add_right_cancel (b := key)
  simpa [enc] using hmessage

/-- Publishing both message and key as inputs gives a different, non-injective map. -/
theorem public_message_key_map_not_injective :
    ¬Injective (fun input : Bit × Bit => enc input.1 input.2) := by
  intro hinjective
  have hcollision : enc (0 : Bit) 0 = enc 1 1 := by
    change (0 : Bit) = 2
    exact (ZMod.natCast_self 2).symm
  have hpairs : ((0 : Bit), (0 : Bit)) = ((1 : Bit), (1 : Bit)) := hinjective hcollision
  have : (0 : Bit) = 1 := congrArg Prod.fst hpairs
  norm_num at this

/-- The old source operation: key exhaustion passes the remaining data through unchanged. -/
def xorStream : List Nat → List Nat → List Nat
  | _, [] => []
  | [], datum :: data => datum :: data
  | key :: keys, datum :: data => (datum ^^^ key) :: xorStream keys data

/-- A one-element key changes only the first element; the tail is visible verbatim. -/
theorem short_key_passthrough (key datum : Nat) (tail : List Nat) :
    xorStream [key] (datum :: tail) = (datum ^^^ key) :: tail := by
  cases tail <;> rfl

end Apeiron.Security.InterpretationWitnesses
