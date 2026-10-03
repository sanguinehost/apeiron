/-
`GF(2^255 − 19)`: the prime field of Curve25519 / Ed25519 / X25519, carried as `ZMod p`.

# Scope

The Rust source mirrored here is `work/crypto-field25519/src/lib.rs`: the modulus constant
`P: [u64; 4]` (`src/lib.rs:49`), the limb form `P_LIMBS: [u64; 5]` (`:60`), the exponent constants
`EXP_P58 = 2^252 − 3` (`:72`) and `EXP_INV = p − 2` (`:82`), `invert` / `invert_or_zero` (`:213`),
`pow_p58` (`:228`), and `from_bytes` / `to_bytes` (`:131` / `:149`). The five-51-bit-limb
representation and the `reduce` carry are **not** modelled: this file is over `ZMod p`, the
canonical field, and the limb layout is the Rust crate's carrier choice, not a mathematical
object. The linkage is at the level of the *values* — modulus, exponents, byte encoding — pinned by
the `decide` theorems below.

Generic prime-field mathematics is not re-derived. It is used from `Crypto/Field/ZModPrime.lean`
in the same corpus: `invZMod`, `invZMod_mul_of_ne_zero`, `invZMod_eq_pow_sub_two`, `powZMod`,
`powZMod_eq`, `is_square_iff_euler` (Euler's criterion, itself `ZMod.euler_criterion`).

# The admissible domain of the field laws

`ZMod p` is a commutative ring for every `p`, so every ring law below holds at the concrete
`p = 2^255 − 19` with no hypothesis. `ZMod p` is a **field only under `Nat.Prime p`**, and Mathlib
at the pinned revision (`leanprover/lean4:v4.31.0`) carries no primality certificate for
`2^255 − 19`. `Nat.Prime p` is therefore **not** established here. Every statement that needs a
division — `invert_mul_cancel`, `sqrtMinusOne_sq`, Euler's criterion, the square-root correction —
takes `Nat.Prime p` as an explicit hypothesis, and the square-root correctness additionally takes
the explicit residue hypothesis `a^((p−1)/2) = 1`. Those hypotheses are documented `Prop`s, not
assumptions disguised as proofs: no `axiom`, no `sorry`, and nothing is proved from them other
than by the ordinary Mathlib field rules that hold of every prime.

# Complexity

`powF` is `Monoid` exponentiation, so `powF a e` costs `⌊log₂ e⌋` squarings and at most `⌊log₂ e⌋`
further multiplications in `ZMod p` (the binary recursion `e = 2m ↦ (a^m)^2`, `e = 2m+1 ↦
(a^m)^2 · a` closes because the exponent halves), and `O(log e)` words. `powP58`,
`sqrtCandidate`, `invert` and `sqrtMinusOne` are therefore `Θ(255)` multiplications of 255-bit
residues — the same exponent arithmetic as the Rust crate's square-and-multiply over 64-bit words,
which reaches `2^255 − 21` in 255 steps. `sqrt` adds one squaring, one comparison and at most one
multiplication. `bytes32Value` and `encode` are `O(32)` `Nat` operations on 255-bit words.

# Trust boundary

Nothing here extracts the Rust code: the crate is `forbid(unsafe_code)` and is read, not linked.
The evaluated pins at the foot are `native_decide` and carry grade `tested`, not `proved`. The
limb representation, `reduce`/`normalize` and the RFC 7919 non-canonical reduction path
(`x ≥ p ↦ x − p`) are outside this file.
-/

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Field.ZMod
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Nat.Bitwise
import Mathlib.NumberTheory.LegendreSymbol.Basic
import Mathlib.NumberTheory.LegendreSymbol.QuadraticChar.GaussSum

import Crypto.Field.ZModPrime

set_option autoImplicit false
set_option warningAsError true
-- The constant pins reduce 255-bit `Nat` arithmetic inside the kernel (`decide` on `2^255`, on
-- the limb sums and on the exponent identities). The recursion is over 255-bit numerals, bounded
-- and small; the depth is raised to admit the stated 255-bit constants, not to defeat a guard.
-- The largest reduction is `decide` on the exponent identities of `p = 2^255 − 19`, which runs
-- the kernel's `Nat.div` and `Nat.pow` recursions over a 255-bit numeral.
set_option maxRecDepth 10000

namespace Apeiron.Crypto.Field.Fp25519

/-! ## The modulus -/

/-- The Curve25519 field modulus `p = 2^255 − 19` as a natural number. -/
def p : ℕ := 2 ^ 255 - 19

/-- `GF(2^255 − 19)`, carried as `ZMod p`. A commutative ring unconditionally; a field under the
hypothesis `Nat.Prime p`, which this file states rather than proves. -/
abbrev F := ZMod p

/-- `p = 2^255 − 19`, decimal. -/
theorem p_eq_decimal :
    p = 57896044618658097711785492504343953926634992332820282019728792003956564819949 := by
  decide

/-- `p ≠ 0`; the only modulus fact needed outside `Nat.Prime p`. -/
theorem p_ne_zero : p ≠ 0 := by decide

/-- `2 < p`, hence `1 ≤ p − 2`: the exponents below are positive. -/
theorem two_lt_p : 2 < p := by decide

/-- `p < 2^255`, the width bound that makes one conditional subtraction canonical in the Rust crate
(`src/lib.rs`, the `reduce` carry). -/
theorem p_lt_two_255 : p < 2 ^ 255 := by decide

/-- `p ≡ 5 (mod 8)`, the shape that licenses the exponent `(p+3)/8` candidate and the
`sqrt(−1)` correction below. -/
theorem p_mod_eight : p % 8 = 5 := by decide

/-- `p` is odd, so `p / 2 = (p − 1) / 2`: the exponent form of `ZMod.pow_div_two_eq_neg_one_or_one`
transported to `(p−1)/2`. -/
theorem div_half : p / 2 = (p - 1) / 2 := by decide

local instance instNeZeroP : NeZero p := ⟨p_ne_zero⟩

/-! ## The defining reduction -/

/-- `2^255 ≡ 19 (mod p)`: the reduction the whole field rests on. -/
theorem two_pow_255_eq_nineteen : (2 : F) ^ 255 = 19 := by decide

/-- The modulus itself is zero in `ZMod p`. -/
theorem self_eq_zero : (p : F) = 0 := by decide

/-! ## Rust constants, as `Nat` values -/

/-- `EXP_INV = p − 2`, the Fermat exponent of `FieldElement::invert` (`src/lib.rs:82`). -/
def expInv : ℕ := p - 2

/-- `EXP_P58 = (p − 5) / 8 = 2^252 − 3`, the exponent of `FieldElement::pow_p58`
(`src/lib.rs:72`). -/
def expP58 : ℕ := (p - 5) / 8

/-- `EXP_P58` as the crate writes it, `2^252 − 3`. -/
theorem expP58_eq_two_pow_252_sub_3 : expP58 = 2 ^ 252 - 3 := by decide

/-- `P_LIMBS`, the five saturated 51-bit limbs of `p` (`src/lib.rs:60`): limb 0 is short by 19,
limbs 1–4 are `2^51 − 1`. -/
def pLimbs : Fin 5 → ℕ :=
  ![2 ^ 51 - 19, 2 ^ 51 - 1, 2 ^ 51 - 1, 2 ^ 51 - 1, 2 ^ 51 - 1]

/-- The limb form encodes `p`. -/
theorem pLimbs_sum : (∑ i : Fin 5, pLimbs i * 2 ^ (51 * (i : ℕ))) = p := by decide

/-- `P`, the four 64-bit little-endian words of `p` (`src/lib.rs:49`), encode `p`. -/
def pWords : Fin 4 → ℕ :=
  ![0xffffffffffffffed, 0xffffffffffffffff, 0xffffffffffffffff, 0x7fffffffffffffff]

/-- The 64-bit word form encodes `p`. -/
theorem pWords_sum : (∑ i : Fin 4, pWords i * 2 ^ (64 * (i : ℕ))) = p := by decide

/-! ## Ring laws

`ZMod p` is a commutative ring for every `p`, so these hold at the concrete `p` with no hypothesis.
Divisibility is the one place `Nat.Prime p` is needed and it appears only as an explicit hypothesis
further down. -/

theorem add_comm (a b : F) : a + b = b + a := _root_.add_comm a b

theorem add_assoc (a b c : F) : (a + b) + c = a + (b + c) := _root_.add_assoc a b c

theorem add_zero (a : F) : a + 0 = a := _root_.add_zero a

theorem zero_add (a : F) : 0 + a = a := _root_.zero_add a

theorem add_neg_cancel (a : F) : a + -a = 0 := _root_.add_neg_cancel a

theorem mul_comm (a b : F) : a * b = b * a := _root_.mul_comm a b

theorem mul_assoc (a b c : F) : (a * b) * c = a * (b * c) := _root_.mul_assoc a b c

theorem mul_one (a : F) : a * 1 = a := _root_.mul_one a

theorem one_mul (a : F) : 1 * a = a := _root_.one_mul a

theorem mul_zero (a : F) : a * 0 = 0 := _root_.MulZeroClass.mul_zero a

theorem add_mul (a b c : F) : (a + b) * c = a * c + b * c := _root_.add_mul a b c

theorem mul_add (a b c : F) : a * (b + c) = a * b + a * c := _root_.mul_add a b c

theorem sub_eq_add_neg' (a b : F) : a - b = a + -b := _root_.sub_eq_add_neg a b

theorem pow_add' (a : F) (m n : ℕ) : a ^ (m + n) = a ^ m * a ^ n := _root_.pow_add a m n

theorem pow_mul' (a : F) (m n : ℕ) : a ^ (m * n) = (a ^ m) ^ n := _root_.pow_mul a m n

/-! ## Exponentiation -/

/-- Exponentiation in the field: `powF a e = a^e`. Cost: `⌊log₂ e⌋` squarings and at most
`⌊log₂ e⌋` further multiplications in `ZMod p`, from the binary recursion
`a^(2m) = (a^m)^2`, `a^(2m+1) = (a^m)^2 · a`; the recursion closes because the exponent halves, and
it occupies `O(log e)` words. -/
def powF (a : F) (e : ℕ) : F := a ^ e

theorem powF_eq (a : F) (e : ℕ) : powF a e = a ^ e := rfl

theorem powF_zero' (a : F) : powF a 0 = 1 := powF_eq a 0

theorem powF_succ (a : F) (e : ℕ) : powF a (e + 1) = a * powF a e :=
  (powF_eq a (e + 1)).trans (pow_succ' a e)

/-- `powF` is the corpus executable `Crypto.Field.ZModPrime.powZMod`, on the prime domain. -/
theorem powF_eq_powZMod [Fact (Nat.Prime p)] (a : F) (e : ℕ) : powF a e = powZMod a e := by
  rw [powF_eq, powZMod_eq]

/-- `powP58 a = a^((p−5)/8) = a^(2^252−3)`, the `FieldElement::pow_p58` of the Rust crate
(`src/lib.rs:228`). Cost: 252 squarings and at most 252 multiplications in `ZMod p`. -/
def powP58 (a : F) : F := powF a expP58

/-- `powP58 a = a^(2^252−3)` on the nose. -/
theorem powP58_eq (a : F) : powP58 a = a ^ (2 ^ 252 - 3) := by
  rw [powP58, powF_eq, expP58_eq_two_pow_252_sub_3]

/-! ## Square roots at `p ≡ 5 (mod 8)`

At `p ≡ 5 (mod 8)` the candidate `a^((p+3)/8)` squares to `±a`, and multiplying by a square root of
`−1` swaps the sign. This is the `p − 1 ≡ 4 (mod 8)` case: `p − 1` has exactly two factors of `2`,
so a single correction suffices and no search over powers of two is needed. -/

/-- `sqrtMinusOne = 2^((p−1)/4)`, the `sqrt(−1)` correction factor. Its square is `2^((p−1)/2)`,
which is `−1` at a prime `p ≡ 5 (mod 8)` because `2` is then a quadratic nonresidue; that instance
is `sqrtMinusOne_sq` and uses the supplement to quadratic reciprocity, not a computation.
Cost: 253 squarings and at most 253 multiplications in `ZMod p`. -/
def sqrtMinusOne : F := powF 2 ((p - 1) / 4)

/-- `sqrtCandidate a = a^((p+3)/8)`. Cost: 253 squarings and at most 253 multiplications. -/
def sqrtCandidate (a : F) : F := powF a ((p + 3) / 8)

/-- `sqrt a`: the candidate when its square is already `a`, otherwise the candidate multiplied by
`sqrtMinusOne`. One squaring, one comparison and at most one extra multiplication in `ZMod p`. On
a quadratic nonresidue the returned value is not a root — `sqrt_not_of_nonresidue` states what
holds there. -/
def sqrt (a : F) : F :=
  let x := sqrtCandidate a
  if x * x = a then x else x * sqrtMinusOne

theorem sqrtCandidate_eq (a : F) : sqrtCandidate a = a ^ ((p + 3) / 8) := powF_eq a ((p + 3) / 8)

theorem sqrtMinusOne_eq : sqrtMinusOne = (2 : F) ^ ((p - 1) / 4) := powF_eq 2 ((p - 1) / 4)

/-- `(p−5)/8 + 1 = (p+3)/8`: the Rust `pow_p58` exponent is one below the square-root candidate
exponent, so `powP58 a * a` is the candidate. -/
theorem powP58_mul (a : F) : powP58 a * a = sqrtCandidate a := by
  have hexp : (p + 3) / 8 = (p - 5) / 8 + 1 := by decide
  rw [powP58, powF_eq, expP58, sqrtCandidate, powF_eq, hexp, pow_add, pow_one]

/-- `sqrtCandidate` is `powP58 a * a`: the Rust exponent, lifted by one. -/
theorem sqrtCandidate_eq_powP58_mul (a : F) : sqrtCandidate a = powP58 a * a :=
  (powP58_mul a).symm

/-- Two squarings of `powP58 a` are `a^(2^253 − 6)`: the candidate exponent `2^252 − 3` doubled,
which is the exponent arithmetic of the Rust `pow_p58` lifted by one. -/
theorem powP58_sq_pow (a : F) : powP58 a ^ 2 = a ^ (2 ^ 253 - 6) := by
  rw [powP58, powF_eq, expP58_eq_two_pow_252_sub_3, ← pow_mul',
    show (2 ^ 252 - 3) * 2 = 2 ^ 253 - 6 by decide]

/-- `sqrtCandidate a ^ 2 = a^((p−1)/4) · a`, the exponent identity behind the construction. -/
theorem sqrtCandidate_sq_eq (a : F) : sqrtCandidate a ^ 2 = a ^ ((p - 1) / 4) * a := by
  have hexp : ((p + 3) / 8) * 2 = (p - 1) / 4 + 1 := by decide
  rw [sqrtCandidate, powF_eq, ← pow_mul', hexp, pow_add', pow_one]

/-- `((p−1)/4) · 2 = (p−1)/2`. -/
theorem quarter_sq_eq (a : F) : (a ^ ((p - 1) / 4)) ^ 2 = a ^ ((p - 1) / 2) := by
  rw [← pow_mul', show ((p - 1) / 4) * 2 = (p - 1) / 2 by decide]

/-- `sqrtMinusOne ^ 2 = 2^((p−1)/2)`. -/
theorem sqrtMinusOne_sq_pow : sqrtMinusOne ^ 2 = (2 : F) ^ ((p - 1) / 2) := by
  rw [sqrtMinusOne, powF_eq, ← pow_mul', show ((p - 1) / 4) * 2 = (p - 1) / 2 by decide]

/-- `2 ≠ 0` in `F`, since `2 < p`. -/
theorem two_ne_zero : (2 : F) ≠ 0 := by
  intro h
  have hz : ((2 : ℕ) : F) = (0 : F) := h
  have hv : ZMod.val ((2 : ℕ) : F) = 0 := by rw [hz, ZMod.val_zero]
  rw [ZMod.val_natCast p 2, Nat.mod_eq_of_lt two_lt_p] at hv
  omega

/-- The `sqrt(−1)` witness: at the prime `p = 2^255 − 19`, `2^((p−1)/4)` squares to `−1`. The proof
is that `2` is a quadratic nonresidue at `p ≡ 5 (mod 8)`, by `FiniteField.isSquare_two_iff`
(`Mathlib/NumberTheory/LegendreSymbol/QuadraticChar/GaussSum.lean`, the supplement to quadratic
reciprocity) together with `p_mod_eight`, then `ZMod.pow_div_two_eq_neg_one_or_one` and Euler's
criterion. Hypothesis `Nat.Prime p`; see the module header. -/
theorem sqrtMinusOne_sq (hp : Nat.Prime p) : sqrtMinusOne ^ 2 = -1 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hns : ¬ IsSquare (2 : F) := by
    rw [FiniteField.isSquare_two_iff, ZMod.card p, p_mod_eight]
    norm_num
  rw [sqrtMinusOne_sq_pow, ← div_half]
  rcases ZMod.pow_div_two_eq_neg_one_or_one p two_ne_zero with h1 | h2
  · exact absurd ((is_square_iff_euler (2 : F) two_ne_zero).mpr (by rw [powZMod_eq]; exact h1)) hns
  · exact h2

/-- Euler's criterion at this `p`, restated with the exponent `(p−1)/2`: `a` is a square exactly
when `a^((p−1)/2) = 1`. Hypothesis `Nat.Prime p`; the generic proof is
`Crypto.Field.ZModPrime.is_square_iff_euler`. -/
theorem is_square_iff_euler_25519 (hp : Nat.Prime p) (a : F) (ha : a ≠ 0) :
    IsSquare a ↔ a ^ ((p - 1) / 2) = 1 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  rw [← div_half, is_square_iff_euler a ha, powZMod_eq a (p / 2)]

/-- The witness at the level of identities: under the residue hypothesis `a^((p−1)/2) = 1` and the
`sqrt(−1)` identity, either the bare candidate or the corrected one squares to `a`. Both hypotheses
are `Prop`s, not axioms; `sqrtMinusOne_sq` discharges the second from `Nat.Prime p`. -/
theorem sqrtWitness (hp : Nat.Prime p) (a : F) (hqr : a ^ ((p - 1) / 2) = 1)
    (hs : sqrtMinusOne ^ 2 = -1) :
    sqrtCandidate a ^ 2 = a ∨ (sqrtCandidate a * sqrtMinusOne) ^ 2 = a := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  have hu := quarter_sq_eq a
  rw [hqr] at hu
  rcases (sq_eq_one_iff.mp hu) with h | h
  · left
    rw [sqrtCandidate_sq_eq a, h, one_mul]
  · right
    calc (sqrtCandidate a * sqrtMinusOne) ^ 2 = sqrtCandidate a ^ 2 * sqrtMinusOne ^ 2 := by ring
      _ = (a ^ ((p - 1) / 4) * a) * -1 := by rw [sqrtCandidate_sq_eq a, hs]
      _ = a := by rw [h]; ring

/-- `sqrt` of a quadratic residue squares to the residue. Hypotheses: `Nat.Prime p` and the
explicit residue condition `a^((p−1)/2) = 1`; `sqrtWitness` is the identity used. -/
theorem sqrt_of_residue (hp : Nat.Prime p) (a : F) (hqr : a ^ ((p - 1) / 2) = 1) :
    sqrt a ^ 2 = a := by
  have hs := sqrtMinusOne_sq hp
  simp only [sqrt]
  split
  · rename_i h
    rw [pow_two]
    exact h
  · rename_i h
    rcases sqrtWitness hp a hqr hs with h1 | h1
    · rw [pow_two] at h1
      exact absurd h1 h
    · exact h1

/-- The stated partiality of `sqrt`: on a nonresidue the returned value is not a root, because it
is then a square of a field element and `a` is not one. Hypothesis `Nat.Prime p`. -/
theorem sqrt_not_of_nonresidue (hp : Nat.Prime p) (a : F) (hn : ¬ IsSquare a) :
    sqrt a ^ 2 ≠ a := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  intro h
  exact hn ((isSquare_iff_exists_sq a).mpr ⟨sqrt a, h.symm⟩)

/-! ## Inversion

`FieldElement::invert` (`src/lib.rs:213`) is Fermat's `x^(p−2)`; `FieldElement::invert_or_zero`
(`:219`) is the total variant. Both are modelled on the canonical carrier, not on the five-limb
representation. -/

/-- `invert a = a ^ (p − 2)`, the Fermat inverse of `FieldElement::invert`. Cost: 254 squarings
and at most 254 multiplications in `ZMod p`. -/
def invert (a : F) : F := powF a expInv

/-- `invertOrZero 0 = 0` and `invertOrZero a = a^(p−2)` otherwise, the total
`FieldElement::invert_or_zero`; the value at zero is not an inverse of anything. Cost: one
equality test in `ZMod p` and 254 squarings. -/
def invertOrZero (a : F) : F := if a = 0 then 0 else invert a

theorem invert_zero : invert (0 : F) = 0 := by
  have h2 : 2 ≤ p := by have := two_lt_p; omega
  have hpos : 0 < expInv := by unfold expInv; exact Nat.sub_pos_iff_lt.mpr two_lt_p
  have hs : expInv = (expInv - 1) + 1 := by omega
  rw [invert, powF_eq, hs, pow_succ, mul_zero]

theorem invertOrZero_zero : invertOrZero (0 : F) = 0 := rfl

theorem invertOrZero_ne_zero (a : F) (ha : a ≠ 0) : invertOrZero a = invert a := if_neg ha

/-- `invZMod 0 = 0`: the corpus inverse has no value at zero. Hypothesis `Nat.Prime p`. -/
theorem invZMod_zero (hp : Nat.Prime p) : invZMod (0 : F) = 0 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  show Inv.inv (0 : F) = 0
  exact inv_zero

/-- `invert a` is the corpus executable inverse `Crypto.Field.ZModPrime.invZMod`. -/
theorem invert_eq_invZMod (hp : Nat.Prime p) (a : F) : invert a = invZMod a := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  by_cases ha : a = 0
  · subst ha
    rw [invert_zero]
    exact (invZMod_zero hp).symm
  · rw [invert, powF_eq, expInv]
    exact (invZMod_eq_pow_sub_two a ha).symm

/-- Fermat's little theorem at this `p`: `invert a * a = 1` for `a ≠ 0`. Hypothesis `Nat.Prime p`:
`ZMod p` is a field only there, and this file does not prove primality. -/
theorem invert_mul_cancel (hp : Nat.Prime p) (a : F) (ha : a ≠ 0) : invert a * a = 1 := by
  haveI : Fact (Nat.Prime p) := ⟨hp⟩
  rw [invert_eq_invZMod hp a]
  exact invZMod_mul_of_ne_zero a ha

/-- The same for the total variant, on its nonzero domain. -/
theorem invertOrZero_mul_cancel (hp : Nat.Prime p) (a : F) (ha : a ≠ 0) :
    invertOrZero a * a = 1 := by
  rw [invertOrZero_ne_zero a ha]
  exact invert_mul_cancel hp a ha

/-! ## The 32-byte little-endian encoding

`FieldElement::from_bytes` / `to_bytes` (`src/lib.rs:131` and `:149`) are modelled by the
value-level encoding: `decode` reads 32 bytes little-endian truncated to 255 bits, so the top bit
of the last byte is ignored, and `encode` writes the canonical representative back. Both are `O(32)`
`Nat` operations on 255-bit words. -/

/-- A 32-byte string, byte `i` in `0 … 255`. -/
abbrev ByteVec := Fin 32 → Fin 256

/-- The integer a byte string denotes, little-endian, reduced mod `2^255`: this reduction is the
top-bit mask of `src/lib.rs:133` (`(acc >> shift) & LIMB_MASK` for the top limb). -/
def bytes32Value (b : ByteVec) : ℕ :=
  (∑ i : Fin 32, (b i : ℕ) * 2 ^ (8 * (i : ℕ))) % 2 ^ 255

/-- `decode`: the field element a byte string denotes. Total: every byte string denotes an element,
because the encoding is a `ZMod p` cast. -/
def decode (b : ByteVec) : F := (bytes32Value b : ZMod p)

/-- `encode x`: the canonical 32-byte little-endian encoding of the representative `x < p`; byte
`i` is `(x / 2^(8i)) mod 256`. The bound is the domain of the definition, not a runtime check.
Cost: 32 divisions and 32 reductions of a 255-bit `Nat`. -/
def encode (x : ℕ) (_hx : x < p) : ByteVec :=
  fun i => ⟨(x / 2 ^ (8 * (i : ℕ))) % 256, Nat.mod_lt _ (by decide)⟩

/-- The mask itself: `bytes32Value` is the byte sum reduced mod `2^255`, so it is always a 255-bit
word; `2^255` itself is therefore the additive period of the raw sum. -/
theorem bytes32Value_lt_top_bit (b : ByteVec) : bytes32Value b < 2 ^ 255 :=
  Nat.mod_lt _ (by decide)

/-- Reassembly: for `x < 2^(8n)`, the byte-wise sum of `x`'s own digits recovers `x`. Induct on `n`,
one byte per step; the step closes because `8 * (n+1) = 8 * n + 8`, so
`x / 2^(8·(j+1)) = (x / 2^(8·j)) / 256` by `Nat.div_div_eq_div_mul` and
`x / 2^(8·n) = x / 256 / 2^(8·(n−j))` at the head, and `Nat.mod_add_div` reassembles the two
halves. -/
theorem byteRoundTrip (x : ℕ) {n : ℕ} (hx : x < 2 ^ (8 * n)) :
    (∑ i : Fin n, ((x / 2 ^ (8 * (i : ℕ))) % 256) * 2 ^ (8 * (i : ℕ))) = x := by
  induction n generalizing x with
  | zero =>
      have h0 : x = 0 := by simpa using hx
      subst h0
      exact Finset.sum_eq_zero fun i => Fin.elim0 i
  | succ n ih =>
      have hpow : 2 ^ (8 * (n + 1)) = 2 ^ (8 * n) * 256 := by
        rw [show 8 * (n + 1) = 8 * n + 8 by omega, Nat.pow_add]
      have htail : x / 256 < 2 ^ (8 * n) :=
        (Nat.div_lt_iff_lt_mul (by decide : 0 < (256 : ℕ))).mpr (hpow ▸ hx)
      have hkey : ∀ j : Fin n,
          ((x / 2 ^ (8 * ((j.succ : Fin (n + 1)) : ℕ))) % 256)
              * 2 ^ (8 * ((j.succ : Fin (n + 1)) : ℕ))
            = 256 * (((x / 256) / 2 ^ (8 * (j : ℕ))) % 256) * 2 ^ (8 * (j : ℕ)) := by
        intro j
        have hj : 8 * ((j.succ : Fin (n + 1)) : ℕ) = 8 * (j : ℕ) + 8 := by
          rw [Fin.val_succ]
          omega
        have hdv' : (x / 2 ^ (8 * (j : ℕ))) / 256 = (x / 256) / 2 ^ (8 * (j : ℕ)) := by
          rw [Nat.div_div_eq_div_mul, Nat.div_div_eq_div_mul]
          exact congrArg (fun t => x / t) (Nat.mul_comm _ _)
        have hdv : x / 2 ^ (8 * ((j.succ : Fin (n + 1)) : ℕ))
            = (x / 2 ^ (8 * (j : ℕ))) / 256 := by
          rw [hj, Nat.pow_add, ← Nat.div_div_eq_div_mul]
        rw [hdv, hj, Nat.pow_add, hdv']
        ring
      have hG : (∑ j : Fin n, (((x / 256) / 2 ^ (8 * (j : ℕ))) % 256)
            * 2 ^ (8 * (j : ℕ))) = x / 256 :=
        ih (x / 256) htail
      rw [Fin.sum_univ_succ]
      rw [show (∑ i : Fin n, ((x / 2 ^ (8 * ((i.succ : Fin (n + 1)) : ℕ))) % 256)
              * 2 ^ (8 * ((i.succ : Fin (n + 1)) : ℕ))) = 256 * (x / 256) from by
            rw [Finset.sum_congr (s₁ := Finset.univ) (s₂ := Finset.univ)
              (f := fun i : Fin n =>
                ((x / 2 ^ (8 * ((i.succ : Fin (n + 1)) : ℕ))) % 256)
                  * 2 ^ (8 * ((i.succ : Fin (n + 1)) : ℕ)))
              (g := fun i : Fin n =>
                256 * (((x / 256) / 2 ^ (8 * (i : ℕ))) % 256 * 2 ^ (8 * (i : ℕ))))
              rfl (fun j _ => by rw [hkey j, Nat.mul_assoc])]
            rw [← Finset.mul_sum, hG]]
      simp only [Fin.val_zero, Nat.mul_zero, Nat.pow_zero, Nat.div_one, Nat.mul_one]
      exact Nat.mod_add_div x 256

theorem bytes32Value_encode (x : ℕ) (hx : x < p) : bytes32Value (encode x hx) = x := by
  have h256 : 2 ^ 255 ≤ 2 ^ 256 := by decide
  have h8 : 2 ^ 256 = 2 ^ (8 * 32) := by rw [show 8 * 32 = 256 by decide]
  have hx' : x < 2 ^ (8 * 32) :=
    Nat.lt_trans hx (p_lt_two_255.trans_le (h256.trans_eq h8))
  have hsum : (∑ i : Fin 32, (encode x hx i : ℕ) * 2 ^ (8 * (i : ℕ))) = x := by
    simpa [encode] using byteRoundTrip x hx'
  rw [bytes32Value, hsum, Nat.mod_eq_of_lt (Nat.lt_trans hx p_lt_two_255)]

/-- `decode (encode x) = x` for every canonical representative. -/
theorem decode_encode (x : ℕ) (hx : x < p) : decode (encode x hx) = (x : F) := by
  rw [decode, bytes32Value_encode x hx]

/-- `encode (decode b) = b` is **false** in general: two byte strings differing in bit 7 of byte 31
differ by `2^255 ≡ 19` in the field, so they are different field elements and no function can
round-trip both. What holds is that `decode` identifies them (`bytes32Value_lt_top_bit` with the
pin below) and that `decode` inverts `encode` on canonical representatives (`decode_encode`,
`encode_decode`). -/
theorem encode_decode (a : F) : decode (encode a.val (ZMod.val_lt a)) = a := by
  rw [decode, bytes32Value_encode a.val (ZMod.val_lt a), ZMod.natCast_zmod_val]

/-! ## Known-answer pins

The pins on `2^255`, `p` and the exponent arithmetic are `decide` over 255-bit numerals: kernel
reductions, grade `proved`. The pins on `sqrt`, `sqrtMinusOne` and `invert` are 254-step
computations in `ZMod p` that the kernel will not reduce inside the build budget, so they run in
`native_decide`: computational pins at grade `tested`, each accompanied by the proved theorem
carrying the same content under its documented hypothesis (`sqrtMinusOne_sq`, `sqrt_of_residue`,
`invert_mul_cancel`). -/

/-- `2^255 − 19` is zero in `ZMod p`: the reduction the whole field rests on. -/
example : ((2 ^ 255 - 19 : ℕ) : ZMod p) = 0 := by decide

/-- `2^255 ≡ 19`, stated on the same reduction. -/
example : ((2 ^ 255 : ℕ) : ZMod p) = 19 := by decide

/-- A known square: `4 = 2^2` in this field. -/
example : ((2 : ℕ) : F) ^ 2 = 4 := by decide

/-- Small arithmetic. -/
example : ((7 : ℕ) : F) + 5 = 12 := by decide

example : ((7 : ℕ) : F) * 5 = 35 := by decide

example : (12 : F) - 19 = -7 := by decide

example : (3 : F) * 3 + 1 = 10 := by decide

/-- `sqrt 4` is a root of 4, evaluated. `tested`; the proof is `sqrt_of_residue`. -/
example : sqrt (4 : F) ^ 2 = 4 := by native_decide

/-- `sqrt 4` is one of the two roots. `tested`. -/
example : sqrt (4 : F) = 2 ∨ sqrt (4 : F) = -2 := by native_decide

/-- For 4, either the bare candidate or the corrected candidate is a root. `tested`; the proved
statement is `sqrtWitness`. -/
example : sqrtCandidate (4 : F) ^ 2 = 4
    ∨ (sqrtCandidate (4 : F) * sqrtMinusOne) ^ 2 = 4 := by native_decide

/-- The `sqrt(−1)` witness, evaluated. `tested`; the proof is `sqrtMinusOne_sq` under
`Nat.Prime p`. -/
example : sqrtMinusOne ^ 2 = -1 := by native_decide

/-- `invert 2` is the inverse of 2, evaluated. `tested`; the proof is `invert_mul_cancel` under
`Nat.Prime p`. -/
example : invert (2 : F) * 2 = 1 := by native_decide

/-- `powP58 2 · 2` is the candidate, evaluated on the pinned exponents. `tested`; the identity is
`powP58_mul`. -/
example : powP58 (2 : F) * 2 = sqrtCandidate 2 := by native_decide

/-- The canonical encoding of a small element is its own little-endian bytes. -/
example : bytes32Value (encode 1 (by decide : 1 < p)) = 1 := by decide

/-- Byte 31 with the top bit set denotes the same element as byte 31 without it, since the value is
reduced mod `2^255` and `2^255 ≡ 19`. This is the top-bit mask of RFC 7919. -/
example : decode (fun i => if i.val = 31 then ⟨255, by decide⟩ else ⟨0, by decide⟩)
    = decode (fun i => if i.val = 31 then ⟨127, by decide⟩ else ⟨0, by decide⟩) := by decide

end Apeiron.Crypto.Field.Fp25519