/-
RFC 2104 (Krawczyk, 1997) HMAC and RFC 5869 (Krawczyk and Eron, 2010) HKDF over
`Apeiron.Crypto.Sha2.sha256`.

`hmacSha256` is the RFC 2104 §2 construction: a 64-byte block derived from the key, the key XORed
with `ipad = 0x36` and with `opad = 0x5c` over that block, and the digest of the inner keyed
message XORed into the outer digest. `hkdfExtract` is `hmacSha256` with the salt as the key (RFC
5869 §2.2), and `hkdfExpand` is the iterated `T(1) = HMAC(PRK, info ‖ 0x01)` chain of RFC 5869 §2.3
concatenated to `L` bytes. Digests are the eight SHA-256 chaining words of
`Apeiron.Crypto.Sha2.State`; `digestBytes` is their big-endian byte serialisation, the 32-byte
form used by the RFC 5869 `T` chain.
-/

import Crypto.Sha2

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Hmac

/-- A SHA-256 digest: the eight chaining words `H[0] … H[7]`. -/
abbrev Digest := Sha2.State

/-- A byte from its natural numeral value, truncating mod `2^8`. O(1). -/
def byte (n : Nat) : BitVec 8 := (n : BitVec 32).setWidth 8

/-- The HMAC-SHA-256 block size in bytes (RFC 2104 §2). -/
def blockSize : Nat := 64

/-- RFC 2104 §2: `ipad`, the constant XORed into the key block for the inner digest. -/
def ipad : BitVec 8 := 0x36

/-- RFC 2104 §2: `opad`, the constant XORed into the key block for the outer digest. -/
def opad : BitVec 8 := 0x5c

/-- The SHA-256 output size in bytes (RFC 2104 §2, RFC 5869 §2.2). -/
def digestSize : Nat := 32

/-! ## Serialisation -/

/-- Byte `i` of the big-endian serialisation of a digest: the `(i / 4)`-th word's byte
`i % 4`, most significant first. O(1). -/
def digestByte (d : Digest) (i : Fin 32) : BitVec 8 :=
  (d.getD (i.val / 4) 0).extractLsb' (8 * (3 - i.val % 4)) 8

/-- The 32-byte big-endian serialisation of a digest, the octet string of RFC 5869 §2.2. O(32)
time and space, a fixed count. -/
def digestBytes (d : Digest) : List (BitVec 8) := List.ofFn fun i : Fin 32 => digestByte d i

/-! ## HMAC (RFC 2104 §2) -/

/-- The 64-byte RFC 2104 §2 key block: `key` when it is at most 64 bytes, otherwise its
SHA-256 digest, zero-padded to 64 bytes. O(64) time and space. -/
def keyBlock (key : List (BitVec 8)) : List (BitVec 8) :=
  let k := if key.length > blockSize then digestBytes (Sha2.hash key) else key
  k ++ List.replicate (blockSize - k.length) 0

/-- XOR the 64-byte key block with the given constant. O(64) time and space. -/
def keyedBlock (c : BitVec 8) (key : List (BitVec 8)) : List (BitVec 8) :=
  (keyBlock key).zipWith (fun a b => a ^^^ b) (List.replicate blockSize c)

/-- RFC 2104 §2 HMAC-SHA-256: `H((K ⊕ opad) ‖ H((K ⊕ ipad) ‖ m))` over the 64-byte key block.
O(|m| + 64) time (two SHA-256 passes over the padded input) and O(|m| + 64) space. -/
def hmacSha256 (key msg : List (BitVec 8)) : Digest :=
  Sha2.hash (keyedBlock opad key ++ digestBytes (Sha2.hash (keyedBlock ipad key ++ msg)))

/-! ## HKDF (RFC 5869 §2) -/

/-- RFC 5869 §2.2 HKDF-Extract: `PRK = HMAC-Hash(salt, IKM)`. An empty salt is a valid input
and is used as the zero-length key. O(|IKM| + 64) time and O(|IKM| + 64) space. -/
def hkdfExtract (salt ikm : List (BitVec 8)) : Digest := hmacSha256 salt ikm

/-- The RFC 5869 §2.3 chain `T(1) ‖ … ‖ T(n)`, appended to the accumulator `acc`, where
`T(k) = HMAC(PRK, acc ‖ info ‖ k)` is 32 bytes. The counter byte is `|acc| / 32 + 1`, so it counts
the blocks already in `acc` and not the remaining recursion depth; the recursion is structural on
`n`. Each step hashes `|acc| + |info| + 1` bytes, so the cost is O(32 · n · (32 · n + |info|)) time
and O(32 · n + |info|) space. -/
def hkdfChain (prk : Digest) (info : List (BitVec 8)) :
    Nat → List (BitVec 8) → List (BitVec 8)
  | 0, acc => acc
  | n + 1, acc =>
    hkdfChain prk info n
      (acc ++ digestBytes (hmacSha256 (digestBytes prk)
        (acc ++ info ++ [byte (acc.length / digestSize + 1)])))

/-- RFC 5869 §2.3 HKDF-Expand: the first `len` bytes of `T(1) ‖ T(2) ‖ …`, each `T(i)` 32 bytes
long. `hBound` is the RFC 5869 §2.3 output bound `1 ≤ L ≤ 255 · HashLen` stated as a hypothesis
rather than checked at the call site, so the function is total on its stated domain. The chain runs
to `⌈len / 32⌉` blocks: O(32 · ⌈len / 32⌉ · (len + |info|)) time and O(len + |info|) space. -/
def hkdfExpand (prk : Digest) (info : List (BitVec 8)) (len : Nat)
    (_hBound : 1 ≤ len ∧ len ≤ 255 * digestSize) : Vector (BitVec 8) len :=
  Vector.ofFn fun i : Fin len =>
    (hkdfChain prk info (i.val / digestSize + 1) []).getD i.val 0

/-- RFC 5869 §2 HKDF: `Expand(Extract(salt, IKM), info, L)`, the one-step derivation. The output
bound is the §2.3 hypothesis `1 ≤ L ≤ 255 · HashLen`. -/
def hkdf (salt ikm info : List (BitVec 8)) (len : Nat)
    (hBound : 1 ≤ len ∧ len ≤ 255 * digestSize) : Vector (BitVec 8) len :=
  hkdfExpand (hkdfExtract salt ikm) info len hBound

/-! ## Known-answer checks

`#guard` runs the untrusted evaluator, so these pin the key block, the byte order and the `T`
chain against regressions; they are not kernel proofs. Every expected value was computed with
`python3 -c` using `hashlib.sha256` and `hmac`. -/

/-- The bytes of a string literal, for the KAT inputs. O(|s|) time and space. -/
def utf8 (s : String) : List (BitVec 8) := s.toUTF8.toList.map fun b => (b.toNat : BitVec 8)

/-- A byte sequence written as natural numerals, for the KAT expected values. O(|l|) time and
space. -/
def bytes (l : List Nat) : List (BitVec 8) := l.map byte

-- RFC 4231 test case 1: a 20-byte key, data "Hi There".
#guard digestBytes (hmacSha256 (List.replicate 20 0x0b) (utf8 "Hi There")) == bytes
  [0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf, 0xce, 0xaf, 0x0b, 0xf1, 0x2b,
   0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83, 0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32, 0xcf, 0xf7]

-- RFC 4231 test case 6: a 131-byte key, so the key is replaced by its SHA-256 digest first.
#guard digestBytes (hmacSha256 (List.replicate 131 0xaa)
  (utf8 "Test Using Larger Than Block-Size Key - Hash Key First")) == bytes
  [0x60, 0xe4, 0x31, 0x59, 0x1e, 0xe0, 0xb6, 0x7f, 0x0d, 0x8a, 0x26, 0xaa, 0xcb, 0xf5, 0xb7, 0x7f,
   0x8e, 0x0b, 0xc6, 0x21, 0x37, 0x28, 0xc5, 0x14, 0x05, 0x46, 0x04, 0x0f, 0x0e, 0xe3, 0x7f, 0x54]

-- RFC 4231 test case 5: an 80-byte key, likewise hashed first.
#guard digestBytes (hmacSha256 (List.replicate 80 0x0c) (utf8
  "Test With Larger Than Block-Size Key and Larger Than Block-Size Data")) == bytes
  [0xf3, 0x88, 0xb1, 0x5a, 0x89, 0x9c, 0xc4, 0x97, 0xd1, 0xb1, 0x6f, 0xd8, 0x2f, 0xe1, 0x85, 0xbb,
   0x9e, 0x47, 0x4e, 0xa0, 0xa4, 0xef, 0xec, 0x4b, 0x42, 0x08, 0xf7, 0x6a, 0x37, 0xfc, 0xd5, 0xe9]

/-- RFC 5869 test case 1 input: the 22-byte IKM `0x0b` repeated. -/
def rfc5869_ikm : List (BitVec 8) := List.replicate 22 0x0b

/-- RFC 5869 test case 1 input: the 13-byte salt `0x00 … 0x0c`. -/
def rfc5869_salt : List (BitVec 8) := (List.range 13).map byte

/-- RFC 5869 test case 1 input: the 10-byte info `0xf0 … 0xf9`. -/
def rfc5869_info : List (BitVec 8) := (List.range 10).map (byte ∘ (fun i => 0xf0 + i))

-- RFC 5869 test case 1: PRK.
#guard digestBytes (hkdfExtract rfc5869_salt rfc5869_ikm) == bytes
  [0x07, 0x77, 0x09, 0x36, 0x2c, 0x2e, 0x32, 0xdf, 0x0d, 0xdc, 0x3f, 0x0d, 0xc4, 0x7b, 0xba, 0x63,
   0x90, 0xb6, 0xc7, 0x3b, 0xb5, 0x0f, 0x9c, 0x31, 0x22, 0xec, 0x84, 0x4a, 0xd7, 0xc2, 0xb3, 0xe5]

-- RFC 5869 test case 1: OKM, L = 42, so T(1) and the first ten bytes of T(2).
#guard (hkdf rfc5869_salt rfc5869_ikm rfc5869_info 42 (by decide)).toList == bytes
  [0x3c, 0xb2, 0x5f, 0x25, 0xfa, 0xac, 0xd5, 0x7a, 0x90, 0x43, 0x4f, 0x64, 0xd0, 0x36, 0x2f, 0x2a,
   0x2d, 0x2d, 0x0a, 0x90, 0xcf, 0x1a, 0x5a, 0x4c, 0x5d, 0xb0, 0x2d, 0x56, 0xec, 0xc4, 0xc5, 0xbf,
   0x34, 0x00, 0x72, 0x08, 0xd5, 0xb8, 0x87, 0x18, 0x58, 0x65]

/-! ## Structural checks -/

#guard (keyBlock []).length == blockSize
#guard (keyBlock (List.replicate 64 0xaa)).length == blockSize
-- A 65-byte key is hashed first, so the result still fits the block.
#guard (keyBlock (List.replicate 65 0xaa)).length == blockSize
#guard (keyedBlock ipad []) == List.replicate blockSize ipad
#guard (keyedBlock opad []) == List.replicate blockSize opad
#guard (digestBytes (Sha2.hash (utf8 "abc"))).length == digestSize
-- The zero-length key of RFC 4231 is a valid HMAC key.
#guard digestBytes (hmacSha256 [] (utf8 "The quick brown fox jumps over the lazy dog")) == bytes
  [0xfb, 0x01, 0x1e, 0x61, 0x54, 0xa1, 0x9b, 0x9a, 0x4c, 0x76, 0x73, 0x73, 0xc3, 0x05, 0x27, 0x5a,
   0x5a, 0x69, 0xe8, 0xb6, 0x8b, 0x0b, 0x4c, 0x92, 0x00, 0xc3, 0x83, 0xdc, 0xed, 0x19, 0xa4, 0x16]

end Apeiron.Crypto.Hmac
