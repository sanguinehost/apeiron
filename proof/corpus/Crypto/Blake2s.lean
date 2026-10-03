/-
BLAKE2s (RFC 7693) over `Vector (BitVec 32) 16` working states and `Vector (BitVec 32) 8`
chaining values.

`IV` is the SHA-256 initialisation vector of RFC 7693 §2.6; `SIGMA` is the ten-round message
word permutation of §2.7; `g` is the mixing function of §3.1 restricted to the diagonal/cross
slices, with rotations 16/12/8/7; `round` applies the eight `g` calls of §3.1 to one message-schedule
row; `compress` is the `F` function of §3.2 for one 64-byte block, the 64-bit offset `t` and the
final-block flag. `hash` performs the §3.2 block sequencing: every non-final full block carries
`t = 64 (i+1)`, and the final block carries `t = |m|` with the `f₁` flag complementing `v[14]`. The
parameter block word 0 is `0x01010000 ⊕ outlen` (digest length in the low byte, no key, fanout 1,
depth 1).

The API mirrors the Rust crate `~/Workspace/work/crypto-blake2s`: `blake2s msg outLen` returns the
first `outLen` bytes of the digest, little-endian per word. Word arithmetic is `BitVec 32`, so it
wraps mod `2^32` exactly as the specification requires.
-/

import Init.Data.Vector
import Init.Data.ByteArray.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Blake2s

/-- One BLAKE2s word. -/
abbrev Word := BitVec 32

/-- The chaining value `h[0] … h[7]` (RFC 7693 §2.6). -/
abbrev Chain := Vector Word 8

/-- The working state `v[0] … v[15]` (RFC 7693 §3.2). -/
abbrev Work := Vector Word 16

/-- One 64-byte message block as sixteen little-endian words `m[0] … m[15]`. BLAKE2s mixes over
sixteen words of a 64-byte block, not the eight 32-bit words duplicated twice. -/
abbrev Block := Vector Word 16

/-- The 64-bit offset `t`, as `(low, high)`. -/
abbrev Counter := Word × Word

/-- `2^32`, the modulus of the `BitVec 32` word representation. -/
def two32 : Nat := 4294967296

/-- A `Fin 16` index from a `Nat`, reduced mod 16. O(1). -/
def i16 (k : Nat) : Fin 16 := ⟨k % 16, Nat.mod_lt k (by decide)⟩

/-- A `Fin 8` index from a `Nat`, reduced mod 8. O(1). -/
def i8 (k : Nat) : Fin 8 := ⟨k % 8, Nat.mod_lt k (by decide)⟩

/-! ## Constants (RFC 7693 §2.6, §2.7) -/

/-- RFC 7693 §2.6: the BLAKE2s initialisation vector, identical to the SHA-256 IV. -/
def IV : Chain :=
  #v[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

/-- RFC 7693 §2.7: the message word permutation `σ[r][0] … σ[r][15]`, ten rows. -/
def SIGMA : Vector (Vector Nat 16) 10 :=
  #v[#v[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
     #v[14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
     #v[11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
     #v[7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
     #v[9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
     #v[2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
     #v[12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
     #v[13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
     #v[6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
     #v[10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]]

/-! ## Mixing and rounds (RFC 7693 §3.1) -/

/-- RFC 7693 §3.1: the mixing function `G` on the four words `a, b, c, d` of one slice, with the
two message words `x, y`, at the BLAKE2s rotations 16, 12, 8, 7. O(1): four `BitVec 32` additions
and four rotations per call. -/
def g (v : Work) (a b c d : Fin 16) (x y : Word) : Work :=
  let v := v.set a (v.get a + v.get b + x)
  let v := v.set d ((v.get d ^^^ v.get a).rotateRight 16)
  let v := v.set c (v.get c + v.get d)
  let v := v.set b ((v.get b ^^^ v.get c).rotateRight 12)
  let v := v.set a (v.get a + v.get b + y)
  let v := v.set d ((v.get d ^^^ v.get a).rotateRight 8)
  let v := v.set c (v.get c + v.get d)
  v.set b ((v.get b ^^^ v.get c).rotateRight 7)

/-- RFC 7693 §3.1: one round, the eight `G` calls of the diagonal and cross halves against message
schedule row `SIGMA[r]`. O(1): eight `g` calls, each O(1) words. -/
def round (v : Work) (m : Block) (r : Fin 10) : Work :=
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

/-- RFC 7693 §3.2: the compression function `F` for one 64-byte block, the 64-bit offset `t` and
the final-block flag. The working state is `h ‖ IV`; `t` is xored into `v[12]` and `v[13]`, and
`f₁` complements `v[14]`. The chaining value is updated as `h[i] ← h[i] ⊕ v[i] ⊕ v[i+8]`. O(1):
ten rounds, each eight `g` calls, over sixteen words. -/
def compress (h : Chain) (block : Block) (t : Counter) (last : Bool) : Chain :=
  let v0 : Work := Vector.ofFn fun i : Fin 16 =>
    if hi : i.val < 8 then h.get ⟨i.val, hi⟩ else IV.get (i8 (i.val - 8))
  let v1 := v0.set (i16 12) (v0.get (i16 12) ^^^ t.1)
  let v2 := v1.set (i16 13) (v1.get (i16 13) ^^^ t.2)
  let v3 := if last then v2.set (i16 14) (~~~v2.get (i16 14)) else v2
  let vr : Work := (List.finRange 10).foldl (fun acc r => round acc block r) v3
  Vector.ofFn fun i : Fin 8 =>
    h.get i ^^^ vr.get ⟨i.val % 16, Nat.mod_lt i.val (by decide)⟩ ^^^ vr.get (i16 (i.val + 8))

/-- The 64-bit offset `t = n` as `(low, high)`. O(1). -/
def counterOf (n : Nat) : Counter := (n % two32, n / two32)

/-- RFC 7693 §2.8: the parameter block, whose first word is
`digest_length | key_length ≪ 8 | fanout ≪ 16 | depth ≪ 24`; the unkeyed sequential mode fixes
`key_length = 0`, `fanout = depth = 1`, leaving `0x01010000 ⊕ outlen`. O(1). -/
def initialChain (outLen : Nat) : Chain :=
  IV.set (i8 0) (IV.get (i8 0) ^^^ ((0x01010000 ^^^ outLen) % two32))

/-- Word `i` of the message block whose first byte is at index `i` of `m`, read little-endian;
bytes past the end of `m` read as zero, which is the zero padding of the final block. O(1): four
byte lookups and one fold. -/
private def wordAt (m : List (BitVec 8)) (i : Nat) : Word :=
  show Nat from
    (List.range 4).foldl (fun acc j => acc + (m.getD (4 * i + j) 0).toNat * 256 ^ j) 0 % two32

/-- The `k`-th full 64-byte block of `m` as sixteen little-endian words. O(1): sixteen `wordAt`. -/
private def fullBlock (m : List (BitVec 8)) (k : Nat) : Block :=
  -- `wordAt` takes a word index, so the block contributes sixteen words from word `16 * k`.
  Vector.ofFn fun i : Fin 16 => wordAt m (16 * k + i.val)

/-- The all-zero block, the single block compressed for the empty message. O(1). -/
def zeroBlock : Block := #v[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

/-- RFC 7693 §3.2: the BLAKE2s chaining value of `m` at digest length `outLen`. O(|m|) time
(⌈|m| / 64⌉ + 1 compressions, each constant) and O(1) extra space: blocks are read in place
through `wordAt` and the folding state is eight words. -/
def hash (m : List (BitVec 8)) (outLen : Nat) : Chain :=
  let h0 := initialChain outLen
  if m.isEmpty then
    compress h0 zeroBlock (counterOf 0) true
  else
    let full := m.length / 64
    let trailing := m.length % 64 != 0
    -- When the length is an exact multiple of 64 the last full block is the final one.
    let nonFinal := if trailing then full else full - 1
    let h := (List.range nonFinal).foldl
      (fun acc i => compress acc (fullBlock m i) (counterOf (64 * (i + 1))) false) h0
    compress h (fullBlock m nonFinal) (counterOf m.length) true

/-- The 32 digest bytes of a chaining value, each word little-endian. O(1): 32 byte extractions. -/
def digestBytes (h : Chain) : Vector (BitVec 8) 32 :=
  Vector.ofFn fun i : Fin 32 =>
    -- The `show Nat from` pins the shift arithmetic in `Nat`; without it the expected `BitVec 8`
    -- result type would make `^` the exclusive-or and the byte extract wrong.
    show Nat from ((h.get ⟨(i.val / 4) % 8, Nat.mod_lt (i.val / 4) (by decide)⟩).toNat / 2 ^ (8 * (i.val % 4)) % 256)

/-- The BLAKE2s digest of a byte list, truncated to `outLen` bytes, matching the Rust
`blake2s(msg, out_len) -> Vec<u8>`. O(|m|) time and O(|m|) space for the list copy. -/
def blake2sList (m : List (BitVec 8)) (outLen : Nat) : List (BitVec 8) :=
  (digestBytes (hash m outLen)).toList.take outLen

/-- The BLAKE2s digest of a packed byte array, truncated to `outLen` bytes. O(|m|) time and O(|m|)
space for the list copy. -/
def blake2s (m : ByteArray) (outLen : Nat) : List (BitVec 8) :=
  blake2sList (m.toList.map fun b => (b.toNat : BitVec 8)) outLen

/-! ## Hex rendering and known-answer checks

`#guard` runs the untrusted evaluator, so these pin the IV, the schedule, the parameter block, the
block sequencing and the output order against regressions; they are not kernel proofs. The
expected values are those of `python3 -c "import hashlib; print(hashlib.blake2s(b'abc').hexdigest())"`
and of `~/Workspace/work/crypto-blake2s/tests/kat.rs`. -/

/-- The lowercase hex digit for `n < 16`. O(1): one list lookup. -/
def hexDigit (n : Nat) : Char :=
  ("0123456789abcdef".toList).getD (n % 16) '0'

/-- The lowercase hex encoding of a digest. O(|l|) time and space in characters. -/
def hex (l : List (BitVec 8)) : String :=
  String.ofList (l.flatMap fun b => [hexDigit (b.toNat / 16), hexDigit b.toNat])

#guard hex (blake2s ByteArray.empty 32) ==
  "69217a3079908094e11121d042354a7c1f55b6482ca1a51e1b250dfd1ed0eef9"

#guard hex (blake2s "abc".toUTF8 32) ==
  "508c5e8c327c14e2e1a72ba34eeb452f37458b209ed63a294d999b4c86675982"

-- The digest length is xored into the parameter block, so a short digest is not a prefix of the
-- long one.
#guard hex (blake2s ByteArray.empty 16) == "64550d6ffe2c0a01a14aba1eade0200c"
#guard hex (blake2s ByteArray.empty 1) == "a1"

-- One byte past the block boundary: one non-final full block at t = 64, then a final block of one
-- message byte and 63 zero bytes at t = 65. The message is the byte ramp `0, 1, …, 64`.
#guard hex (blake2sList ((List.range 65).map fun i => (⟨i % 256, Nat.mod_lt i (by decide)⟩ : BitVec 8)) 32) ==
  "1b53ee94aaf34e4b159d48de352c7f0661d0a40edff95a0b1639b4090e974472"

-- Exactly one full block: the single block carries the final-block flag at t = 64.
#guard hex (blake2sList ((List.range 64).map fun i => (⟨i % 256, Nat.mod_lt i (by decide)⟩ : BitVec 8)) 32) ==
  "56f34e8b96557e90c1f24b52d0c89d51086acf1b00f634cf1dde9233b8eaaa3e"

-- The parameter block word: outlen 32 changes only the low byte of `h[0]`.
#guard (initialChain 32).get (i8 0) == (0x6a09e667 ^^^ 0x01010020 : Word)
#guard (initialChain 0).get (i8 0) == (0x6a09e667 ^^^ 0x01010000 : Word)

#guard (SIGMA.get ⟨0, by decide⟩).toList == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
#guard (SIGMA.get ⟨9, by decide⟩).toList == [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0]

end Apeiron.Crypto.Blake2s