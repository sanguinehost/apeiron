/-
RFC 8439: the ChaCha20 stream cipher and the Poly1305 authenticator.

ChaCha20 state (RFC 8439 §2.1): 16 32-bit words `0…15` laid out as `constants ‖ key ‖ counter ‖
nonce`, the four constant words spelling `"expand 32-byte k"` as little-endian 32-bit words, the
256-bit key as 8 words, the 32-bit block counter as 1 word and the 96-bit nonce as 3 words. `qr` is
the quarter round of §2.1, `rounds` is one double round (column round then diagonal round), `block`
is §2.3 (20 rounds on a working copy, then the feed-forward addition of the input state), `keystream`
is §2.4's output byte stream for `n` blocks and `chacha20_xor` is §2.5 encryption.

Poly1305 (§2.5): arithmetic modulo `p = 2^130 − 5` on `BitVec 128` residues. `poly1305_clamp` is
the key split (`r` from `key[0..16]`, clamped; `s` from `key[16..32]`), `poly1305_block` is the
accumulator step `h ← (h + n) · r mod p` and `poly1305_mac` is the tag `((h + s) mod 2^128)` read
little-endian. Messages are byte lists; the tail block carries the `0x01` terminator of §2.5.
-/

import Init.Data.Vector

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.ChaCha

/-! ## ChaCha20 state and round function -/

/-- One 32-bit word of the ChaCha20 state (RFC 8439 §2.1). -/
abbrev Word := BitVec 32

/-- The ChaCha20 state: 16 words `constants ‖ key ‖ counter ‖ nonce` (RFC 8439 §2.1). -/
abbrev State := Vector Word 16

/-- RFC 8439 §2.1 constants `"expand 32-byte k"` as little-endian 32-bit words. -/
def constants : Vector Word 4 := #v[0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]

/-- The RFC 8439 §2.1 initial state for `key`, the 96-bit `nonce` and the 32-bit block `counter`,
read as 8, 3 and 1 little-endian 32-bit words. O(1). -/
def initialState (key : Vector Word 8) (nonce : Vector Word 3) (counter : Word) : State :=
  constants ++ key ++ #v[counter] ++ nonce

/-- The RFC 8439 §2.1 quarter round on words `a, b, c, d`:
`a += b; d ← rotl(d ⊕ a, 16); c += d; b ← rotl(b ⊕ c, 12); a += b; d ← rotl(d ⊕ a, 8);
c += d; b ← rotl(b ⊕ c, 7)`. Each step reads the state written by the previous one, so the
additions, exclusive ors and rotations compose in RFC order. O(1). -/
def qr (a b c d : Fin 16) (s : State) : State :=
  let s1 := s.set a (s.get a + s.get b)
  let s2 := s1.set d (s1.get d ^^^ s1.get a)
  let s3 := s2.set d ((s2.get d).rotateLeft 16)
  let s4 := s3.set c (s3.get c + s3.get d)
  let s5 := s4.set b (s4.get b ^^^ s4.get c)
  let s6 := s5.set b ((s5.get b).rotateLeft 12)
  let s7 := s6.set a (s6.get a + s6.get b)
  let s8 := s7.set d (s7.get d ^^^ s7.get a)
  let s9 := s8.set d ((s8.get d).rotateLeft 8)
  let s10 := s9.set c (s9.get c + s9.get d)
  let s11 := s10.set b (s10.get b ^^^ s10.get c)
  s11.set b ((s11.get b).rotateLeft 7)

/-- RFC 8439 §2.1 column round: the four quarter rounds `(0,4,8,12)`, `(1,5,9,13)`, `(2,6,10,14)`,
`(3,7,11,15)`. The four index sets are disjoint, so the order of the four applications is
immaterial. O(1). -/
def columnRound (s : State) : State :=
  qr 0 4 8 12 (qr 1 5 9 13 (qr 2 6 10 14 (qr 3 7 11 15 s)))

/-- RFC 8439 §2.1 diagonal round: the four quarter rounds `(0,5,10,15)`, `(1,6,11,12)`, `(2,7,8,13)`,
`(3,4,9,14)`. O(1). -/
def diagonalRound (s : State) : State :=
  qr 0 5 10 15 (qr 1 6 11 12 (qr 2 7 8 13 (qr 3 4 9 14 s)))

/-- One RFC 8439 §2.1 double round: the diagonal round applied to the column round. O(1). -/
def rounds (s : State) : State := diagonalRound (columnRound s)

/-- RFC 8439 §2.3: `n` double rounds, so `2 · n` quarter-round layers; the RFC's 20 rounds are
`n = 10`. The recursion has `n` steps of 4 + 4 quarter rounds. O(n) word updates, O(1) space. -/
def doubleRounds : Nat → State → State
  | 0, s => s
  | n + 1, s => rounds (doubleRounds n s)

/-- RFC 8439 §2.3 block function: 20 rounds on a working copy of `s`, then the feed-forward addition
`w + s` wordwise; the added copy is the initial state, not the running state, so `block` is pure.
O(1) time and space: 20 rounds over a 16-word state. -/
def block (s : State) : State :=
  let w := doubleRounds 10 s
  Vector.ofFn fun i => w.get i + s.get i

/-! ## Keystream and encryption (RFC 8439 §2.4, §2.5) -/

/-- `n` blocks of ChaCha20 keystream as bytes: block `k` is `block (initialState key nonce
(counter + k))` serialised little-endian (§2.3), byte `j` being bit offset `8 · (j mod 4)` of word
`(j / 4) mod 16` of block `j / 64`. `Vector.ofFn` builds all `64 · n` bytes without intermediate
lists, so the cost is `n` block functions and `64n` byte extractions: O(n) time and O(n) space. -/
def keystream (key : Vector Word 8) (nonce : Vector Word 3) (counter : Word) (n : Nat) :
    Vector (BitVec 8) (64 * n) :=
  Vector.ofFn fun j : Fin (64 * n) =>
    (block (initialState key nonce (counter + (j.val / 64 : Word)))).get
      ⟨(j.val / 4) % 16, Nat.mod_lt (j.val / 4) (by decide : 0 < 16)⟩ |>.extractLsb' ((j.val % 4) * 8) 8

/-- RFC 8439 §2.5 encryption: the message XORed with the keystream starting at `counter`, read as
little-endian bytes. The keystream runs past the message and is truncated to `|m|` bytes, so a
message that is not a multiple of 64 bytes uses its last block partially. `List.ofFn` writes each
output byte once, with one keystream block per 64 bytes plus one for the tail: O(|m|) time and
O(|m| / 64 + 1) space for the keystream. -/
def chacha20_xor (key : Vector Word 8) (nonce : Vector Word 3) (counter : Word)
    (msg : List (BitVec 8)) : List (BitVec 8) :=
  let ks := keystream key nonce counter (msg.length / 64 + 1)
  List.ofFn fun j : Fin msg.length => msg.getD j.val 0 ^^^ ks.getD j.val 0

/-! ## Poly1305 (RFC 8439 §2.5) -/

/-- The Poly1305 prime `p = 2^130 − 5`, at the 260-bit width that carries both residues and their
products. -/
def poly1305P : BitVec 260 := BitVec.ofNat 260 (2 ^ 130 - 5)

/-- The Poly1305 clamp mask of RFC 8439 §2.5: bits `0, 1, 2` and `7` of each of the four bytes
3, 7, 11, 15 are cleared. -/
def poly1305Mask : BitVec 128 := 0x0ffffffc0ffffffc0ffffffc0fffffff

/-- The one-bit-beyond-the-octets term of RFC 8439 §2.5, at the 260-bit width that holds it: for a
full 16-byte block this is `2^128`, the `0x01` byte written past the block. It is never truncated,
since bit 128 is the term itself. A short final block carries the same one bit as a `0x01` *inside*
its 16 bytes, so it takes no further term. -/
def poly1305Bit : BitVec 260 := BitVec.ofNat 260 (2 ^ 128)

/-- `b` read as a little-endian 128-bit unsigned integer: byte `i` at bit offset `8 · i`. The 16
disjoint shifted bytes are combined with `|||`. O(1): 16 shifts and 16 ors on 128-bit words. -/
def le128 (b : Vector (BitVec 8) 16) : BitVec 128 :=
  (Vector.ofFn fun i => ((b.get i).zeroExtend 128) <<< (8 * i.val)).foldl (fun acc x => acc ||| x) 0

/-- Addition of two 128-bit integers modulo `2^128`. O(1). -/
def add128 (a b : BitVec 128) : BitVec 128 := a + b

/-- Reduction of `x` to the representative in `[0, p)` congruent to `x (mod p)`: one 260-bit
remainder by `p`. O(1): a fixed-width divide of 260-bit words. -/
def poly1305Reduce (x : BitVec 260) : BitVec 260 :=
  x % poly1305P

/-- The Poly1305 key split of RFC 8439 §2.5: `r` is `key[0..16]` with bytes 3, 7, 11, 15 masked to
their low four bits and the whole word clamped, `s` is `key[16..32]` read little-endian. O(1). -/
def poly1305_clamp (key : Vector (BitVec 8) 32) : BitVec 128 × BitVec 128 :=
  let rb := Vector.ofFn fun i : Fin 16 =>
    if i.val % 4 = 3 then key.getD i.val 0 &&& 15 else key.getD i.val 0
  (le128 rb &&& poly1305Mask, le128 (Vector.ofFn fun i : Fin 16 => key.getD (i.val + 16) 0))

/-- One Poly1305 accumulator step (§2.5): `h ← (h + n) · r mod p` for the block number `n`. Both
operands are below `p < 2^130`, so the 260-bit product is exact and `% p` reduces it;
`h + n < 2^131` is below `2^260`. O(1) time and space: two 260-bit reductions, one 260-bit add and
one 130×130-bit product per block, all on fixed-width words. -/
def poly1305_block (r h n : BitVec 260) : BitVec 260 :=
  poly1305Reduce (poly1305Reduce (h + n) * r)

/-- The Poly1305 block numbers of `m` (§2.5), each a 16-byte block read little-endian with one bit
added beyond the octets it occupies. Indices `0 … |m| / 16 - 1` are the full blocks, each
`le128 + 2^128`; the final entry is the zero-padded remainder with `0x01` in byte `|m| mod 16`,
which already is that bit and so takes no `2^128`. `List.ofFn` writes `|m| / 16 + 1` entries
directly, so the tail needs no append. O(|m|) time and space: one pass to drop the remainder plus
`|m| / 16 + 1` 16-byte reads. -/
def poly1305Blocks (m : List (BitVec 8)) : List (BitVec 260) :=
  let full := m.length / 16
  let rest := m.drop (16 * full)
  List.ofFn fun k : Fin (full + 1) =>
    if k.val < full then
      BitVec.zeroExtend 260 (le128 (Vector.ofFn fun i : Fin 16 => m.getD (16 * k.val + i.val) 0))
        + poly1305Bit
    else
      BitVec.zeroExtend 260 (le128 (Vector.ofFn fun i : Fin 16 =>
        if i.val < rest.length then rest.getD i.val 0
        else if i.val = rest.length then 1 else 0))

/-- RFC 8439 §2.5 tag: the accumulator folded over the padded blocks, added to `s` and truncated to
128 bits, read little-endian as 16 bytes. One `poly1305_block` per padded block: O(|m| / 16) 260-bit
word operations, O(1) accumulator space. -/
def poly1305_mac (key : Vector (BitVec 8) 32) (m : List (BitVec 8)) : Vector (BitVec 8) 16 :=
  let (r128, s) := poly1305_clamp key
  let h := (poly1305Blocks m).foldl (poly1305_block (BitVec.zeroExtend 260 r128)) 0
  Vector.ofFn fun i : Fin 16 => (add128 (h.extractLsb' 0 128) s).extractLsb' (8 * i.val) 8

/-! ## Known-answer checks

`#guard` runs the untrusted evaluator, so these pin the round order, the rotation amounts, the
little-endian serialisation, the counter indexing and the Poly1305 clamp against regressions; they
are not kernel proofs. References: RFC 8439 §2.3.2 (block function), §2.4.2 (encryption) and §2.5.2
(Poly1305), each confirmed against OpenSSL 3.5.5 `enc -chacha20` and Python `cryptography`
`Poly1305.generate_tag`. -/

private def key : Vector Word 8 :=
  #v[0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c]

private def nonce : Vector Word 3 := #v[0x09000000, 0x4a000000, 0x00000000]

private def polyKey : Vector (BitVec 8) 32 :=
  #v[0x85, 0xd6, 0xbe, 0x78, 0x57, 0x55, 0x6d, 0x33, 0x7f, 0x44, 0x52, 0xfe, 0x42, 0xd5, 0x06, 0xa8,
     0x01, 0x03, 0x80, 0x8a, 0xfb, 0x0d, 0xb2, 0xfd, 0x4a, 0xbf, 0xf6, 0xaf, 0x41, 0x49, 0xf5, 0x1b]

-- RFC 8439 §2.3.2: the block function output at counter 1, as bytes.
#guard (keystream key nonce 1 1).toList ==
  [0x10, 0xf1, 0xe7, 0xe4, 0xd1, 0x3b, 0x59, 0x15, 0x50, 0x0f, 0xdd, 0x1f, 0xa3,
   0x20, 0x71, 0xc4, 0xc7, 0xd1, 0xf4, 0xc7, 0x33, 0xc0, 0x68, 0x03, 0x04, 0x22,
   0xaa, 0x9a, 0xc3, 0xd4, 0x6c, 0x4e, 0xd2, 0x82, 0x64, 0x46, 0x07, 0x9f, 0xaa,
   0x09, 0x14, 0xc2, 0xd7, 0x05, 0xd9, 0x8b, 0x02, 0xa2, 0xb5, 0x12, 0x9c, 0xd1,
   0xde, 0x16, 0x4e, 0xb9, 0xcb, 0xd0, 0x83, 0xe8, 0xa2, 0x50, 0x3c, 0x4e]

-- The second block of a two-block keystream is the block function at counter + 1.
#guard (keystream key nonce 0 2).toList.drop 64 == (keystream key nonce 1 1).toList

-- RFC 8439 §2.4.2 encryption of the §2.4.2 plaintext at counter 1.
#guard chacha20_xor key nonce 1
    [0x4c, 0x61, 0x64, 0x69, 0x65, 0x73, 0x20, 0x61, 0x6e, 0x64, 0x20, 0x47, 0x65,
     0x6e, 0x74, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x20, 0x6f, 0x66, 0x20, 0x74, 0x68,
     0x65, 0x20, 0x63, 0x6c, 0x61, 0x73, 0x73, 0x20, 0x6f, 0x66, 0x20, 0x27, 0x39,
     0x39, 0x3a, 0x20, 0x49, 0x66, 0x20, 0x49, 0x20, 0x63, 0x6f, 0x75, 0x6c, 0x64,
     0x20, 0x6f, 0x66, 0x66, 0x65, 0x72, 0x20, 0x79, 0x6f, 0x75, 0x20, 0x6f, 0x6e,
     0x6c, 0x79, 0x20, 0x6f, 0x6e, 0x65, 0x20, 0x74, 0x69, 0x70, 0x20, 0x66, 0x6f,
     0x72, 0x20, 0x74, 0x68, 0x65, 0x20, 0x66, 0x75, 0x74, 0x75, 0x72, 0x65, 0x2c,
     0x20, 0x73, 0x75, 0x6e, 0x73, 0x63, 0x72, 0x65, 0x65, 0x6e, 0x20, 0x77, 0x6f,
     0x75, 0x6c, 0x64, 0x20, 0x62, 0x65, 0x20, 0x69, 0x74, 0x2e] ==
  [0x5c, 0x90, 0x83, 0x8d, 0xb4, 0x48, 0x79, 0x74, 0x3e, 0x6b, 0xfd, 0x58, 0xc6,
   0x4e, 0x05, 0xa8, 0xa2, 0xbc, 0x91, 0xa9, 0x13, 0xaf, 0x0e, 0x23, 0x70, 0x4a,
   0xcf, 0xba, 0xa0, 0xb8, 0x0d, 0x3d, 0xa1, 0xa2, 0x0b, 0x20, 0x27, 0xb8, 0x93,
   0x30, 0x2e, 0xe2, 0x9e, 0x63, 0xf9, 0xc2, 0x22, 0xc1, 0xda, 0x67, 0xf0, 0xb5,
   0xfe, 0x79, 0x28, 0xdf, 0xae, 0xa2, 0xa3, 0x91, 0xcd, 0x25, 0x1c, 0x21, 0x64,
   0xe4, 0xfa, 0x57, 0x56, 0xb9, 0xda, 0x6e, 0x8c, 0xa5, 0xdc, 0x90, 0x8c, 0x44,
   0xcb, 0xf6, 0xe9, 0x3e, 0xa6, 0xb4, 0xcc, 0x40, 0x69, 0x88, 0xd7, 0xda, 0x69,
   0xbf, 0x79, 0x5b, 0xf1, 0x9b, 0x84, 0x53, 0x9d, 0xf7, 0x3b, 0xd9, 0xb3, 0xe9,
   0xca, 0x4d, 0x03, 0xbc, 0x0a, 0x58, 0x6f, 0xf5, 0x28, 0xdc]

-- The Poly1305 clamp of the §2.5.2 key gives `r = 0x0806d5400e52447c036d555408bed685` and
-- `s = 0x1bf54941aff6bf4afdb20dfb8a800301`.
#guard (poly1305_clamp polyKey).1 == 0x0806d5400e52447c036d555408bed685

#guard (poly1305_clamp polyKey).2 == 0x1bf54941aff6bf4afdb20dfb8a800301

-- RFC 8439 §2.5.2 Poly1305 tag over `"Cryptographic Forum Research Group"`.
#guard (poly1305_mac polyKey
    [0x43, 0x72, 0x79, 0x70, 0x74, 0x6f, 0x67, 0x72, 0x61, 0x70, 0x68, 0x69, 0x63, 0x20, 0x46,
     0x6f, 0x72, 0x75, 0x6d, 0x20, 0x52, 0x65, 0x73, 0x65, 0x61, 0x72, 0x63, 0x68, 0x20, 0x47,
     0x72, 0x6f, 0x75, 0x70]).toList ==
  [0xa8, 0x06, 0x1d, 0xc1, 0x30, 0x51, 0x36, 0xc6, 0xc2, 0x2b, 0x8b, 0xaf, 0x0c, 0x01, 0x27, 0xa9]

-- A message whose length is a multiple of 16 still takes a terminator block after the last full
-- block; a one-byte message takes only that block.
#guard (poly1305Blocks [0x01]).length == 1

#guard (poly1305Blocks (List.replicate 32 (0x01 : BitVec 8))).length == 3

end Apeiron.Crypto.ChaCha