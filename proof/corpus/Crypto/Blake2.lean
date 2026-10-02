/-
BLAKE2b (RFC 7693) over `Vector (BitVec 64) 16` working states and `Vector (BitVec 64) 8`
chaining values.

`IV` is the SHA-512 initialisation vector of RFC 7693 §2.6; `SIGMA` is the twelve-round message
word permutation of §2.7; `g` is the mixing function of §3.1 restricted to the diagonal/cross
slices; `round` applies the eight `g` calls of §3.1 to one message-schedule row; `compress` is the
`F` function of §3.2 for one 128-byte block, the 128-bit offset `t` and the final-block flag.
`hash` performs the §3.2 block sequencing: every non-final full block carries `t = 128 (i+1)`, and
the final block carries `t = |m|` with the `f₁` flag complementing `v[14]`. The parameter block
word 0 is `0x01010000 ⊕ outlen` (digest length in the low byte, no key, fanout 1, depth 1).

The API mirrors the Rust crate `work/crypto-blake2`: `blake2b msg outLen` returns the first
`outLen` bytes of the digest, little-endian per word. Word arithmetic is `BitVec 64`, so it wraps
mod `2^64` exactly as the specification requires.
-/

import Init.Data.Vector
import Init.Data.ByteArray.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Blake2

/-- One BLAKE2b word. -/
abbrev Word := BitVec 64

/-- The chaining value `h[0] … h[7]` (RFC 7693 §2.6). -/
abbrev Chain := Vector Word 8

/-- The working state `v[0] … v[15]` (RFC 7693 §3.2). -/
abbrev Work := Vector Word 16

/-- One 128-byte message block as sixteen little-endian words `m[0] … m[15]`. -/
abbrev Block := Vector Word 16

/-- The 128-bit offset `t`, as `(low, high)`. -/
abbrev Counter := Word × Word

/-- `2^64`, the modulus of the `BitVec 64` word representation. -/
def two64 : Nat := 18446744073709551616

/-- A `Fin 16` index from a `Nat`, reduced mod 16. O(1). -/
def i16 (k : Nat) : Fin 16 := ⟨k % 16, Nat.mod_lt k (by decide)⟩

/-- A `Fin 8` index from a `Nat`, reduced mod 8. O(1). -/
def i8 (k : Nat) : Fin 8 := ⟨k % 8, Nat.mod_lt k (by decide)⟩

/-! ## Constants (RFC 7693 §2.6, §2.7) -/

/-- RFC 7693 §2.6: the BLAKE2b initialisation vector, identical to the SHA-512 IV. -/
def IV : Chain :=
  #v[0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
     0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179]

/-- RFC 7693 §2.7: the message word permutation `σ[r][0] … σ[r][15]`, twelve rows, rows 10 and 11
repeating rows 0 and 1. -/
def SIGMA : Vector (Vector Nat 16) 12 :=
  #v[#v[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
     #v[14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
     #v[11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
     #v[7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
     #v[9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
     #v[2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
     #v[12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
     #v[13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
     #v[6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
     #v[10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
     #v[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
     #v[14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3]]

/-! ## Mixing and rounds (RFC 7693 §3.1) -/

/-- RFC 7693 §3.1: the mixing function `G` on the four words `a, b, c, d` of one slice, with the
two message words `x, y`. O(1): four `BitVec 64` additions and four rotations per call. -/
def g (v : Work) (a b c d : Fin 16) (x y : Word) : Work :=
  let v := v.set a (v.get a + v.get b + x)
  let v := v.set d ((v.get d ^^^ v.get a).rotateRight 32)
  let v := v.set c (v.get c + v.get d)
  let v := v.set b ((v.get b ^^^ v.get c).rotateRight 24)
  let v := v.set a (v.get a + v.get b + y)
  let v := v.set d ((v.get d ^^^ v.get a).rotateRight 16)
  let v := v.set c (v.get c + v.get d)
  v.set b ((v.get b ^^^ v.get c).rotateRight 63)

/-- RFC 7693 §3.1: one round, the eight `G` calls of the diagonal and cross halves against message
schedule row `SIGMA[r]`. O(1): eight `g` calls, each O(1) words. -/
def round (v : Work) (m : Block) (r : Fin 12) : Work :=
  let row := SIGMA.get r
  let s := fun i : Fin 16 => m.get ⟨row.get i % 16, Nat.mod_lt (row.get i) (by decide)⟩
  let v := g v (i16 0) (i16 4) (i16 8) (i16 12) (s ⟨0, by decide⟩) (s ⟨1, by decide⟩)
  let v := g v (i16 1) (i16 5) (i16 9) (i16 13) (s ⟨2, by decide⟩) (s ⟨3, by decide⟩)
  let v := g v (i16 2) (i16 6) (i16 10) (i16 14) (s ⟨4, by decide⟩) (s ⟨5, by decide⟩)
  let v := g v (i16 3) (i16 7) (i16 11) (i16 15) (s ⟨6, by decide⟩) (s ⟨7, by decide⟩)
  let v := g v (i16 0) (i16 5) (i16 10) (i16 15) (s ⟨8, by decide⟩) (s ⟨9, by decide⟩)
  let v := g v (i16 1) (i16 6) (i16 11) (i16 12) (s ⟨10, by decide⟩) (s ⟨11, by decide⟩)
  let v := g v (i16 2) (i16 7) (i16 8) (i16 13) (s ⟨12, by decide⟩) (s ⟨13, by decide⟩)
  g v (i16 3) (i16 4) (i16 9) (i16 14) (s ⟨14, by decide⟩) (s ⟨15, by decide⟩)

/-! ## Compression and the hash (RFC 7693 §3.2) -/

/-- RFC 7693 §3.2: the compression function `F` for one 128-byte block, the 128-bit offset `t` and
the final-block flag. The working state is `h ‖ IV`; `t` is xored into `v[12]`, `v[13]`, and `f₁`
complements `v[14]`. The chaining value is updated as `h[i] ← h[i] ⊕ v[i] ⊕ v[i+8]`. O(1): twelve
rounds, each eight `g` calls, over sixteen words. -/
def compress (h : Chain) (block : Block) (t : Counter) (last : Bool) : Chain :=
  let v0 : Work := Vector.ofFn fun i : Fin 16 =>
    if hi : i.val < 8 then h.get ⟨i.val, hi⟩ else IV.get (i8 (i.val - 8))
  let v1 := v0.set (i16 12) (v0.get (i16 12) ^^^ t.1)
  let v2 := v1.set (i16 13) (v1.get (i16 13) ^^^ t.2)
  let v3 := if last then v2.set (i16 14) (~~~v2.get (i16 14)) else v2
  let vr : Work := (List.finRange 12).foldl (fun acc r => round acc block r) v3
  Vector.ofFn fun i : Fin 8 =>
    h.get i ^^^ vr.get ⟨i.val % 16, Nat.mod_lt i.val (by decide)⟩ ^^^ vr.get (i16 (i.val + 8))

/-- The 128-bit offset `t = n` as `(low, high)`. O(1). -/
def counterOf (n : Nat) : Counter := (n % two64, n / two64)

/-- RFC 7693 §2.8: the parameter block, whose first word is
`digest_length | key_length ≪ 8 | fanout ≪ 16 | depth ≪ 24`; the unkeyed sequential mode fixes
`key_length = 0`, `fanout = depth = 1`, leaving `0x01010000 ⊕ outlen`. O(1). -/
def initialChain (outLen : Nat) : Chain :=
  IV.set (i8 0) (IV.get (i8 0) ^^^ ((0x01010000 ^^^ outLen) % two64))

/-- Word `i` of the message block whose first byte is at index `i` of `m`, read little-endian;
bytes past the end of `m` read as zero, which is the zero padding of the final block. O(1): eight
byte lookups and one fold. -/
private def wordAt (m : List (BitVec 8)) (i : Nat) : Word :=
  show Nat from
    (List.range 8).foldl (fun acc j => acc + (m.getD (8 * i + j) 0).toNat * 256 ^ j) 0 % two64

/-- The `k`-th full 128-byte block of `m` as sixteen little-endian words. O(1): sixteen `wordAt`. -/
private def fullBlock (m : List (BitVec 8)) (k : Nat) : Block :=
  -- `wordAt` takes a word index, so the block contributes sixteen words from word `16 * k`.
  Vector.ofFn fun i : Fin 16 => wordAt m (16 * k + i.val)

/-- The all-zero block, the single block compressed for the empty message. O(1). -/
def zeroBlock : Block := #v[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- RFC 7693 §3.2: the BLAKE2b chaining value of `m` at digest length `outLen`. O(|m|) time
(⌈|m| / 128⌉ + 1 compressions, each constant) and O(1) extra space: blocks are read in place
through `wordAt` and the folding state is eight words. -/
def hash (m : List (BitVec 8)) (outLen : Nat) : Chain :=
  let h0 := initialChain outLen
  if m.isEmpty then
    compress h0 zeroBlock (counterOf 0) true
  else
    let full := m.length / 128
    let trailing := m.length % 128 != 0
    -- When the length is an exact multiple of 128 the last full block is the final one.
    let nonFinal := if trailing then full else full - 1
    let h := (List.range nonFinal).foldl
      (fun acc i => compress acc (fullBlock m i) (counterOf (128 * (i + 1))) false) h0
    compress h (fullBlock m nonFinal) (counterOf m.length) true

/-- The 64 digest bytes of a chaining value, each word little-endian. O(1): 64 byte extractions. -/
def digestBytes (h : Chain) : Vector (BitVec 8) 64 :=
  Vector.ofFn fun i : Fin 64 =>
    -- The `show Nat from` pins the shift arithmetic in `Nat`; without it the expected `BitVec 8`
    -- result type would make `^` the exclusive-or and the byte extract wrong.
    show Nat from ((h.get ⟨(i.val / 8) % 8, Nat.mod_lt (i.val / 8) (by decide)⟩).toNat / 2 ^ (8 * (i.val % 8)) % 256)

/-- The BLAKE2b digest of a byte list, truncated to `outLen` bytes, matching the Rust
`blake2b(msg, out_len) -> Vec<u8>`. O(|m|) time and O(|m|) space for the list copy. -/
def blake2bList (m : List (BitVec 8)) (outLen : Nat) : List (BitVec 8) :=
  (digestBytes (hash m outLen)).toList.take outLen

/-- The BLAKE2b digest of a packed byte array, truncated to `outLen` bytes. O(|m|) time and O(|m|)
space for the list copy. -/
def blake2b (m : ByteArray) (outLen : Nat) : List (BitVec 8) :=
  blake2bList (m.toList.map fun b => (b.toNat : BitVec 8)) outLen

/-! ## Hex rendering and known-answer checks

`#guard` runs the untrusted evaluator, so these pin the IV, the schedule, the parameter block, the
block sequencing and the output order against regressions; they are not kernel proofs. The
expected values are those of `python3 -c "import hashlib; print(hashlib.blake2b(b'abc').hexdigest())"`
and of `work/crypto-blake2/tests/kat.rs`. -/

/-- The lowercase hex digit for `n < 16`. O(1): one list lookup. -/
def hexDigit (n : Nat) : Char :=
  ("0123456789abcdef".toList).getD (n % 16) '0'

/-- The lowercase hex encoding of a digest. O(|l|) time and space in characters. -/
def hex (l : List (BitVec 8)) : String :=
  String.ofList (l.flatMap fun b => [hexDigit (b.toNat / 16), hexDigit b.toNat])

#guard hex (blake2b ByteArray.empty 64) ==
  "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"

#guard hex (blake2b "abc".toUTF8 64) ==
  "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d17d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923"

-- The digest length is xored into the parameter block, so a short digest is not a prefix of the
-- long one.
#guard hex (blake2b ByteArray.empty 32) == "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
#guard hex (blake2b ByteArray.empty 1) == "2e"

-- One byte past the block boundary: one non-final full block at t = 128, then a final block of one
-- message byte and 127 zero bytes at t = 129.
#guard hex (blake2bList (List.replicate 129 (65 : BitVec 8)) 64) ==
  "3dc8c5e69fe2ad0d8ea8bc16732f00cd7c1ce619783cb91a2f684ccf2e1e95a4aba27640e9f6339df2b1d572c4fb3900deae6330b94e900a934e131b9ca5d136"

-- Exactly one full block: the single block carries the final-block flag at t = 128.
#guard hex (blake2bList (List.replicate 128 (65 : BitVec 8)) 64) ==
  "f5011c14425def0732ae5ad325ea7ceb558b908e390cb8157d15c365226d4e076789bd1e9534353bfc852bb90c0c1c85755a7cc43f9fafecd8fabade9bcc8d77"

-- The parameter block word: outlen 64 changes only the low byte of `h[0]`.
#guard (initialChain 64).get (i8 0) == (0x6a09e667f3bcc908 ^^^ 0x01010040 : Word)
#guard (initialChain 0).get (i8 0) == (0x6a09e667f3bcc908 ^^^ 0x01010000 : Word)

#guard (SIGMA.get ⟨0, by decide⟩).toList == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#guard (SIGMA.get ⟨11, by decide⟩).toList == [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3]

end Apeiron.Crypto.Blake2
