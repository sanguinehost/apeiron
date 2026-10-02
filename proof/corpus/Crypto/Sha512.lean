/-
FIPS PUB 180-4: SHA-512 over `Vector (BitVec 64) 80` message schedules and
`Vector (BitVec 64) 8` states.

The state is the eight chaining words `H[0] … H[7]` of §6.4.1 in order. `pad` is the
§5.1.1 `M` construction: append `0x80`, zero-fill, then the 128-bit big-endian message length in
bits, so the result is a whole number of 128-byte blocks. `compress` is one §6.4.2 block
transformation from a state and one 1024-bit block; `sha512` folds `compress` over the padded
message from the §6.1 initial hash value. Word arithmetic is `BitVec 64`, so it wraps mod `2^64`
exactly as the specification's mod-`2^64` word arithmetic requires.
-/

import Init.Data.Vector
import Init.Data.ByteArray.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Sha512

/-- One SHA-512 word: 64 bits of the state or message schedule. -/
abbrev Word := BitVec 64

/-- The SHA-512 chaining state: `H[0] … H[7]` of FIPS 180-4 §6.4.1. -/
abbrev State := Vector Word 8

/-- A 1024-bit message block as sixteen big-endian words (`W[0] … W[15]`). -/
abbrev Block := Vector Word 16

/-! ## Constants (FIPS 180-4 §4.2.3, §5.3.5) -/

/-- FIPS 180-4 §4.2.3: the first sixty-four bits of the fractional parts of the cube roots of the
first eighty primes. -/
def K : Vector Word 80 :=
  #v[0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
     0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
     0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
     0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
     0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
     0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
     0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
     0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
     0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
     0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
     0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
     0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
     0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
     0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
     0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
     0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
     0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
     0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
     0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
     0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817]

/-- FIPS 180-4 §5.3.5: the first sixty-four bits of the fractional parts of the square roots of the
first eight primes. -/
def H0 : State :=
  #v[0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
     0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]

/-- Rotate a word right by `n` bits (`0 < n < 64`), the `ROTR` of FIPS 180-4 §3.2. O(1). -/
def rotr (x : Word) (n : Nat) : Word := (x >>> (n % 64)) ||| (x <<< ((64 - n) % 64))

/-- Shift a word right by `n` bits (`0 ≤ n < 64`), the `SHR` of FIPS 180-4 §3.2. A logical
shift fills with zeros; `BitVec` `>>>` is exactly that. O(1). -/
def shrx (x : Word) (n : Nat) : Word := x >>> (n % 64)

/-! ## Round functions (FIPS 180-4 §4.1.2) -/

/-- `σ₀(x) = ROTR^1 x ⊕ ROTR^8 x ⊕ SHR^7 x`, FIPS 180-4 §4.1.2. The third term is a shift, not a
rotation. O(1). -/
def sigma0 (x : Word) : Word := rotr x 1 ^^^ rotr x 8 ^^^ shrx x 7

/-- `σ₁(x) = ROTR^19 x ⊕ ROTR^61 x ⊕ SHR^6 x`, FIPS 180-4 §4.1.2. The third term is a shift, not
a rotation. O(1). -/
def sigma1 (x : Word) : Word := rotr x 19 ^^^ rotr x 61 ^^^ shrx x 6

/-- `Σ₀(x) = ROTR^28 x ⊕ ROTR^34 x ⊕ ROTR^39 x`, FIPS 180-4 §4.1.3. O(1). -/
def Sigma0 (x : Word) : Word := rotr x 28 ^^^ rotr x 34 ^^^ rotr x 39

/-- `Σ₁(x) = ROTR^14 x ⊕ ROTR^18 x ⊕ ROTR^41 x`, FIPS 180-4 §4.1.3. O(1). -/
def Sigma1 (x : Word) : Word := rotr x 14 ^^^ rotr x 18 ^^^ rotr x 41

/-- `Ch(x, y, z) = (x ∧ y) ⊕ (¬x ∧ z)`, FIPS 180-4 §4.1.2. O(1). -/
def choose (x y z : Word) : Word := (x &&& y) ^^^ (~~~x &&& z)

/-- `Maj(x, y, z) = (x ∧ y) ⊕ (x ∧ z) ⊕ (y ∧ z)`, FIPS 180-4 §4.1.2. O(1). -/
def majority (x y z : Word) : Word := (x &&& y) ^^^ (x &&& z) ^^^ (y &&& z)

/-! ## Message schedule (FIPS 180-4 §6.4.2 step 1) -/

/-- Extend the sixteen words of `w` to the full eighty word schedule by applying
`W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]` for `t = 16 … 79`. O(80) time and space. -/
def schedule (w : Block) : Vector Word 80 :=
  let start : Vector Word 80 :=
    (Vector.ofFn fun i : Fin 16 => w.getD i 0) ++ Vector.replicate 64 (0 : Word)
  (List.range 64).foldl (fun acc k =>
    let t := 16 + k
    let wt := sigma1 (acc.getD (t - 2) 0) + acc.getD (t - 7) 0
              + sigma0 (acc.getD (t - 15) 0) + acc.getD (t - 16) 0
    Vector.ofFn fun i : Fin 80 => if i.val == t then wt else acc.getD i 0) start

/-! ## Block compression (FIPS 180-4 §6.4.2) -/

/-- One §6.4.2 compression: extend the block to `W`, initialise `a … h`, run the eighty rounds
`T₁ = h + Σ₁(e) + Ch(e, f, g) + K[t] + W[t]` and `T₂ = Σ₀(a) + Maj(a, b, c)`, then add the
eight working words back into the incoming state. O(80) time and O(80) space. -/
def compress (h : State) (block : Block) : State :=
  let w := schedule block
  let v0 : State := h
  let v := (List.range 80).foldl (fun acc t =>
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

/-- Big-endian byte `i` (`0 ≤ i ≤ 15`) of the 128-bit bit-length `bits`. O(1). -/
def lengthByte (bits : Nat) (i : Fin 16) : BitVec 8 := (bits >>> (8 * (15 - i.val))) % 256

/-- The sixteen bytes of the 128-bit big-endian bit-length of a message of `n` bytes. O(1). -/
def lengthBlock (n : Nat) : List (BitVec 8) := List.ofFn fun i : Fin 16 => lengthByte (8 * n) i

/-- FIPS 180-4 §5.1.1: `m` followed by `0x80`, enough zeros, and the 128-bit big-endian bit-length,
so that the total length is the least multiple of 128 that is at least `|m| + 17`. The zeros count
is `total - |m| - 17`, which is in `0 … 111` for every `|m|`. O(|m|) time and space. -/
def pad (m : List (BitVec 8)) : List (BitVec 8) :=
  let r := (m.length + 17) % 128
  let total := if r = 0 then m.length + 17 else m.length + 17 + (128 - r)
  m ++ [0x80] ++ List.replicate (total - m.length - 17) 0 ++ lengthBlock m.length

/-- The `k`-th 128-byte block of the padded message as sixteen big-endian words. O(128) time. -/
private def paddedBlock (p : List (BitVec 8)) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 =>
    (List.range 8).foldl (fun acc j =>
      ((acc.toNat <<< 8) ||| (p.getD (128 * k + 8 * i.val + j) 0).toNat)) 0

/-- FIPS 180-4 §6.4.1: fold `compress` over the padded message starting from `H0`. O(|m|) time
(one compression per 128-byte block) and O(|m|) space for the padding. -/
def hash (m : List (BitVec 8)) : State :=
  let p := pad m
  (List.range (p.length / 128)).foldl (fun h k => compress h (paddedBlock p k)) H0

/-- SHA-512 of a packed byte array: the eight big-endian chaining words in order `H[0] … H[7]`.
O(|m|) time and O(|m|) space. -/
def sha512 (m : ByteArray) : State :=
  hash (m.toList.map fun b => (b.toNat : BitVec 8))

/-! ## Known-answer checks

`#guard` runs the untrusted evaluator, so these pin the word order, the schedule, the padding and
the length encoding against regressions; they are not kernel proofs. References: the FIPS 180-4
examples and the NIST "abc" vector, reproduced by
`python3 -c "import hashlib; print(hashlib.sha512(b'abc').hexdigest())"`. -/

#guard (sha512 ByteArray.empty).toList ==
  [0xcf83e1357eefb8bd, 0xf1542850d66d8007, 0xd620e4050b5715dc, 0x83f4a921d36ce9ce,
   0x47d0d13c5d85f2b0, 0xff8318d2877eec2f, 0x63b931bd47417a81, 0xa538327af927da3e]

#guard (sha512 "abc".toUTF8).toList ==
  [0xddaf35a193617aba, 0xcc417349ae204131, 0x12e6fa4e89a97ea2, 0x0a9eeee64b55d39a,
   0x2192992a274fc1a8, 0x36ba3c23a3feebbd, 0x454d4423643ce80e, 0x2a9ac94fa54ca49f]

-- Two padded blocks: `|m| = 56`, so `0x80`, the zeros and the 128-bit length field close the
-- second block and the length field ends the padded message.
#guard (sha512 "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".toUTF8).toList ==
  [0x204a8fc6dda82f0a, 0x0ced7beb8e08a416, 0x57c16ef468b228a8, 0x279be331a703c335,
   0x96fd15c13b1b07f9, 0xaa1d3bea57789ca0, 0x31ad85c7a71dd703, 0x54ec631238ca3445]

#guard (pad []).length == 128
#guard (pad (List.replicate 111 0)).length == 128
#guard (pad (List.replicate 112 0)).length == 256
-- One more message byte forces a second padded block: `128 + 17 = 145` rounds up to 256.
#guard (pad (List.replicate 128 0)).length == (pad []).length + 128

end Apeiron.Crypto.Sha512