/-
FIPS PUB 180-4: SHA-256 over `Vector (BitVec 32) 64` message schedules and
`Vector (BitVec 32) 8` states.

The state is the eight chaining words `H[0] … H[7]` of §6.2.1 in order. `pad` is the
§5.1.1 `M` construction: append `0x80`, zero-fill, then the 64-bit big-endian message length in
bits, so the result is a whole number of 64-byte blocks. `compress` is one §6.2.2 block
transformation from a state and one 512-bit block; `sha256` folds `compress` over the padded message
from the §6.1 initial hash value. Word arithmetic is `BitVec 32`, so it wraps mod `2^32` exactly as
the specification's mod-`2^32` word arithmetic requires.
-/

import Init.Data.Vector
import Init.Data.ByteArray.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Sha2

/-- One SHA-256 word: 32 bits of the state or message schedule. -/
abbrev Word := BitVec 32

/-- The SHA-256 chaining state: `H[0] … H[7]` of FIPS 180-4 §6.2.1. -/
abbrev State := Vector Word 8

/-- A 512-bit message block as sixteen big-endian words (`W[0] … W[15]`). -/
abbrev Block := Vector Word 16

/-! ## Constants (FIPS 180-4 §4.2.2, §5.3.3) -/

/-- FIPS 180-4 §4.2.2: the first thirty-two bits of the fractional parts of the cube roots of the
first sixty-three primes. -/
def K : Vector Word 64 :=
  #v[0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
     0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
     0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
     0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
     0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
     0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
     0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
     0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

/-- FIPS 180-4 §5.3.3: the first thirty-two bits of the fractional parts of the square roots of the
first eight primes. -/
def H0 : State :=
  #v[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

/-- Rotate a word right by `n` bits (`0 < n < 32`), the `ROTR` of FIPS 180-4 §3.2. -/
def rotr (x : Word) (n : Nat) : Word := (x >>> (n % 32)) ||| (x <<< ((32 - n) % 32))

/-- Shift a word right by `n` bits (`0 ≤ n < 32`), the `SHR` of FIPS 180-4 §3.2. A logical
shift fills with zeros; `BitVec` `>>>` is exactly that. -/
def shrx (x : Word) (n : Nat) : Word := x >>> (n % 32)

/-! ## Round functions (FIPS 180-4 §4.1.2) -/

/-- `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ SHR^3 x`, FIPS 180-4 §4.1.2. The third term is a shift, not a
rotation. O(1). -/
def sigma0 (x : Word) : Word := rotr x 7 ^^^ rotr x 18 ^^^ shrx x 3

/-- `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ SHR^10 x`, FIPS 180-4 §4.1.2. The third term is a shift, not
a rotation. O(1). -/
def sigma1 (x : Word) : Word := rotr x 17 ^^^ rotr x 19 ^^^ shrx x 10

/-- `Σ₀(x) = ROTR^2 x ⊕ ROTR^13 x ⊕ ROTR^22 x`, FIPS 180-4 §4.1.3. O(1). -/
def Sigma0 (x : Word) : Word := rotr x 2 ^^^ rotr x 13 ^^^ rotr x 22

/-- `Σ₁(x) = ROTR^6 x ⊕ ROTR^11 x ⊕ ROTR^25 x`, FIPS 180-4 §4.1.3. O(1). -/
def Sigma1 (x : Word) : Word := rotr x 6 ^^^ rotr x 11 ^^^ rotr x 25

/-- `Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z)`, FIPS 180-4 §4.1.2. O(1). -/
def choose (x y z : Word) : Word := (x &&& y) ^^^ (~~~x &&& z)

/-- `Maj(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)`, FIPS 180-4 §4.1.2. O(1). -/
def majority (x y z : Word) : Word := (x &&& y) ^^^ (x &&& z) ^^^ (y &&& z)

/-! ## Message schedule (FIPS 180-4 §6.2.2 step 1) -/

/-- Extend the sixteen words of `w` to the full sixty-four word schedule by applying
`W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]` for `t = 16 … 63`. O(64) time and space. -/
def schedule (w : Block) : Vector Word 64 :=
  let start : Vector Word 64 :=
    (Vector.ofFn fun i : Fin 16 => w.getD i 0) ++ Vector.replicate 48 (0 : Word)
  (List.range 48).foldl (fun acc k =>
    let t := 16 + k
    let wt := sigma1 (acc.getD (t - 2) 0) + acc.getD (t - 7) 0
              + sigma0 (acc.getD (t - 15) 0) + acc.getD (t - 16) 0
    Vector.ofFn fun i : Fin 64 => if i.val == t then wt else acc.getD i 0) start

/-! ## Block compression (FIPS 180-4 §6.2.2) -/

/-- One §6.2.2 compression: extend the block to `W`, initialise `a … h`, run the sixty-four rounds
`T₁ = h + Σ₁(e) + Ch(e, f, g) + K[t] + W[t]` and `T₂ = Σ₀(a) + Maj(a, b, c)`, then add the
eight working words back into the incoming state. O(64) time and O(64) space. -/
def compress (h : State) (block : Block) : State :=
  let w := schedule block
  let v0 : State := h
  let v := (List.range 64).foldl (fun acc t =>
    let a := acc.getD 0 0
    let b := acc.getD 1 0
    let c := acc.getD 2 0
    let d := acc.getD 3 0
    let e := acc.getD 4 0
    let f := acc.getD 5 0
    let g := acc.getD 6 0
    let hh := acc.getD 7 0
    let t1 := hh + Sigma1 e + choose e f g + K.getD t 0 + w.getD t 0
    let t2 := Sigma0 a + majority a b c
    Vector.ofFn fun i : Fin 8 =>
      match i.val with
      | 0 => t1 + t2
      | 1 => a
      | 2 => b
      | 3 => c
      | 4 => d + t1
      | 5 => e
      | 6 => f
      | _ => g) v0
  Vector.ofFn fun i : Fin 8 => h.getD i 0 + v.getD i.val 0

/-! ## Padding and hashing -/

/-- Big-endian byte `i` (`0 ≤ i ≤ 7`) of the 64-bit bit-length `bits`. O(1). -/
def lengthByte (bits : Nat) (i : Fin 8) : BitVec 8 := (bits >>> (8 * (7 - i.val))) % 256

/-- The eight bytes of the 64-bit big-endian bit-length of a message of `n` bytes. O(1). -/
def lengthBlock (n : Nat) : List (BitVec 8) := List.ofFn fun i : Fin 8 => lengthByte (8 * n) i

/-- FIPS 180-4 §5.1.1: `m` followed by `0x80`, enough zeros, and the 64-bit big-endian bit-length,
so that the total length is the least multiple of 64 that is at least `|m| + 9`. The zeros count is
`total - |m| - 9`, which is in `0 … 63` for every `|m|`. O(|m|) time and space. -/
def pad (m : List (BitVec 8)) : List (BitVec 8) :=
  let r := (m.length + 9) % 64
  let total := if r = 0 then m.length + 9 else m.length + 9 + (64 - r)
  m ++ [0x80] ++ List.replicate (total - m.length - 9) 0 ++ lengthBlock m.length

/-- The `k`-th 64-byte block of the padded message as sixteen big-endian words. O(64) time. -/
private def paddedBlock (p : List (BitVec 8)) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 =>
    (List.range 4).foldl (fun acc j =>
      ((acc.toNat <<< 8) ||| (p.getD (64 * k + 4 * i.val + j) 0).toNat) % 4294967296) 0

/-- FIPS 180-4 §6.2.1: fold `compress` over the padded message starting from `H0`. O(|m|) time
(one compression per 64-byte block) and O(|m|) space for the padding. -/
def hash (m : List (BitVec 8)) : State :=
  let p := pad m
  (List.range (p.length / 64)).foldl (fun h k => compress h (paddedBlock p k)) H0

/-- SHA-256 of a packed byte array: the eight big-endian chaining words in order `H[0] … H[7]`.
O(|m|) time and O(|m|) space. -/
def sha256 (m : ByteArray) : State :=
  hash (m.toList.map fun b => (b.toNat : BitVec 8))

/-! ## Known-answer checks

`#guard` runs the untrusted evaluator, so these pin the word order, the schedule, the padding and
the length encoding against regressions; they are not kernel proofs. References: the FIPS 180-4
examples and the NIST "abc" vector. -/

#guard (sha256 ByteArray.empty).toList ==
  [0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855]

#guard (sha256 "abc".toUTF8).toList ==
  [0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad]

-- Two blocks: the padding is a full extra block (`0x80`, zeros, then the 64-bit length).
#guard (sha256 "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".toUTF8).toList ==
  [0x248d6a61, 0xd20638b8, 0xe5c02693, 0x0c3e6039, 0xa33ce459, 0x64ff2167, 0xf6ecedd4, 0x19db06c1]

-- Two padded blocks: `|m| + 9 = 121`, so one zero byte closes the second block and the length
-- field occupies the last eight bytes of it.
#guard (sha256 (ByteArray.mk ⟨List.replicate 112 (97 : UInt8)⟩)).toList ==
  [0xf5435300, 0x8a255326, 0x2ecdc4a3, 0x4749563b, 0xa0950e8b, 0x0fc86527, 0x80b0a614, 0xb99683c1]

#guard (pad []).length == 64
#guard (pad (List.replicate 55 0)).length == 64
#guard (pad (List.replicate 56 0)).length == 128
-- One more message byte forces a second padded block: `64 + 9 = 73` rounds up to 128.
#guard (pad (List.replicate 64 0)).length == (pad []).length + 64

end Apeiron.Crypto.Sha2
