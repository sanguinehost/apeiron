import Apeiron.Table.Basic
import Mathlib.Data.ZMod.Basic

/-!
# Identity words

@name Table Word Identity
@grade proved

A word multiplication keeps its low `w` bits by reduction modulo `2^w`. An odd multiplier is a
unit in that ring, so multiplication by it is injective. At width 64 this applies to
`0x9E3779B97F4A7C15`, whose natural-number value is `11400714819323198485`.

The product of two 64-bit inputs is below `2^128`. Natural-number multiplication therefore models
the widened `u128` product, and reduction modulo `2^64` models its `U64` truncation. The bridge to
the extracted body still has to identify those two operations with its widening multiply and
truncating cast.

## Root

This module instantiates no trichotomy theorem. Its root is the finite-ring primitive
`ZMod.isUnit_iff_coprime`, and its edge is the identity projection needed by the table:

| from | here |
|---|---|
| `ZMod.isUnit_iff_coprime` | `odd_mul_zmod_injective` |
| `ZMod.val_mul` | `lowWord_mul_eq_zmod_mul`, `phi64_lowWord_eq_zmod_mul` |
| `odd_mul_zmod_injective` | `phi64_lowWord_injective` |
-/

namespace Apeiron
namespace Table

/-- The low `w` bits of a natural number.

    (w n : ℕ)       [a word width and a natural number]
    : ℕ             [the natural number represented by the low word] -/
def lowWord (w n : ℕ) : ℕ := n % 2 ^ w

/-- **An odd multiplier is injective modulo a power of two.**

    {w a : ℕ}                              [a word width and a multiplier]
    (ha : Odd a)                           [the multiplier is odd]
    Function.Injective                     [different inputs have different outputs]
      (fun x : ZMod (2 ^ w) => a * x)      [for multiplication in the word ring]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ZMod` [the integers modulo a natural number]
    `Odd` [one more than twice a natural number]
    `^` [raised to a natural-number power] -/
theorem odd_mul_zmod_injective {w a : ℕ} (ha : Odd a) :
    Function.Injective (fun x : ZMod (2 ^ w) => a * x) := by
  apply IsUnit.mul_right_injective
  rw [ZMod.isUnit_iff_coprime]
  exact ha.coprime_two_right.pow_right w

/-- **Low-word reduction is multiplication in the word ring.**

    (w a x : ℕ)                                      [a word width and two natural numbers]
    (lowWord w (a * x) : ZMod (2 ^ w))               [the low word cast into the word ring]
      = (a : ZMod (2 ^ w)) * (x : ZMod (2 ^ w))      [equals their product in that ring]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ZMod` [the integers modulo a natural number]
    `lowWord` [the low bits of a natural number]
    `^` [raised to a natural-number power] -/
theorem lowWord_mul_eq_zmod_mul (w a x : ℕ) :
    (lowWord w (a * x) : ZMod (2 ^ w)) =
      (a : ZMod (2 ^ w)) * (x : ZMod (2 ^ w)) := by
  simp [lowWord, ZMod.natCast_mod]

/-- The odd 64-bit multiplier used by `coord::spread`.

    phi64 : ℕ       [the multiplier as a natural number] -/
def phi64 : ℕ := 11400714819323198485

/-- The `coord::spread` multiplier is odd.

    Odd phi64       [the multiplier is one more than twice a natural number]

    `Odd` [one more than twice a natural number]
    `phi64` [the 64-bit multiplier used by coordinate spreading] -/
theorem phi64_odd : Odd phi64 := by
  exact ⟨5700357409661599242, by norm_num [phi64]⟩

/-- **The `coord::spread` multiplier is injective in the 64-bit word ring.**

    Function.Injective                               [different inputs have different outputs]
      (fun x : ZMod (2 ^ 64) => phi64 * x)           [for multiplication by the spread constant]

    `ZMod` [the integers modulo a natural number]
    `phi64` [the 64-bit multiplier used by coordinate spreading]
    `^` [raised to a natural-number power] -/
theorem phi64_mul_zmod_injective :
    Function.Injective (fun x : ZMod (2 ^ 64) => phi64 * x) :=
  odd_mul_zmod_injective phi64_odd

/-- **The widened product of a 64-bit input and `phi64` fits in 128 bits.**

    {x : ℕ}                         [a natural-number input]
    (hx : x < 2 ^ 64)               [the input fits in 64 bits]
    x * phi64 < 2 ^ 128             [the widened product fits in 128 bits]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `phi64` [the 64-bit multiplier used by coordinate spreading]
    `^` [raised to a natural-number power] -/
theorem phi64_product_fits_u128 {x : ℕ} (hx : x < 2 ^ 64) : x * phi64 < 2 ^ 128 := by
  have hphi : phi64 < 2 ^ 64 := by norm_num [phi64]
  calc
    x * phi64 < 2 ^ 64 * phi64 := (Nat.mul_lt_mul_right (by norm_num [phi64])).2 hx
    _ < 2 ^ 64 * 2 ^ 64 := (Nat.mul_lt_mul_left (by positivity)).2 hphi
    _ = 2 ^ 128 := by norm_num [← pow_add]

/-- **The truncated widened spread product is multiplication in the 64-bit word ring.**

    (x : ℕ)                                             [a natural-number input]
    (lowWord 64 (x * phi64) : ZMod (2 ^ 64))            [its low widened spread product]
      = (x : ZMod (2 ^ 64)) * (phi64 : ZMod (2 ^ 64))   [equals the word-ring product]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `ZMod` [the integers modulo a natural number]
    `lowWord` [the low bits of a natural number]
    `phi64` [the 64-bit multiplier used by coordinate spreading]
    `^` [raised to a natural-number power] -/
theorem phi64_lowWord_eq_zmod_mul (x : ℕ) :
    (lowWord 64 (x * phi64) : ZMod (2 ^ 64)) =
      (x : ZMod (2 ^ 64)) * (phi64 : ZMod (2 ^ 64)) :=
  lowWord_mul_eq_zmod_mul 64 x phi64

/-- **Equal low 64-bit spread products have equal 64-bit inputs.**

    {x y : ℕ}                                      [two natural-number inputs]
    (hx : x < 2 ^ 64) (hy : y < 2 ^ 64)            [both inputs fit in 64 bits]
    (h : lowWord 64 (x * phi64)                    [their first low spread word]
      = lowWord 64 (y * phi64))                    [equals their second low spread word]
    x = y                                          [the inputs are equal]

    `ℕ` [the counting numbers 0, 1, 2, 3, and so on]
    `lowWord` [the low bits of a natural number]
    `phi64` [the 64-bit multiplier used by coordinate spreading]
    `^` [raised to a natural-number power] -/
theorem phi64_lowWord_injective {x y : ℕ} (hx : x < 2 ^ 64) (hy : y < 2 ^ 64)
    (h : lowWord 64 (x * phi64) = lowWord 64 (y * phi64)) : x = y := by
  have hz : (x : ZMod (2 ^ 64)) = (y : ZMod (2 ^ 64)) :=
    phi64_mul_zmod_injective <| by
      change (phi64 : ZMod (2 ^ 64)) * (x : ZMod (2 ^ 64)) =
        (phi64 : ZMod (2 ^ 64)) * (y : ZMod (2 ^ 64))
      rw [← lowWord_mul_eq_zmod_mul, ← lowWord_mul_eq_zmod_mul]
      exact congrArg (fun n : ℕ => (n : ZMod (2 ^ 64))) <| by
        simpa [Nat.mul_comm] using h
  have hv := congrArg ZMod.val hz
  rwa [ZMod.val_natCast_of_lt hx, ZMod.val_natCast_of_lt hy] at hv

/-- **Multiplication by two is not injective in the 64-bit word ring.**

    ¬ Function.Injective                          [different inputs can have equal outputs]
      (fun x : ZMod (2 ^ 64) => 2 * x)            [for multiplication by the even number two]

    `ZMod` [the integers modulo a natural number]
    `¬` [this is not true]
    `^` [raised to a natural-number power] -/
theorem two_mul_zmod64_not_injective :
    ¬ Function.Injective (fun x : ZMod (2 ^ 64) => 2 * x) := by
  intro h
  have heq : (0 : ZMod (2 ^ 64)) = ((2 ^ 63 : ℕ) : ZMod (2 ^ 64)) := by
    apply h
    calc
      (2 : ZMod (2 ^ 64)) * 0 = 0 := mul_zero _
      _ = (2 ^ 64 : ℕ) := (ZMod.natCast_self (2 ^ 64)).symm
      _ = (2 : ZMod (2 ^ 64)) * ((2 ^ 63 : ℕ) : ZMod (2 ^ 64)) := by
        norm_num [← Nat.cast_mul]
  have hne : (0 : ZMod (2 ^ 64)) ≠ ((2 ^ 63 : ℕ) : ZMod (2 ^ 64)) := by
    intro hz
    have hv := congrArg ZMod.val hz
    rw [ZMod.val_zero, ZMod.val_natCast_of_lt (by norm_num)] at hv
    norm_num at hv
  exact hne heq

end Table
end Apeiron
