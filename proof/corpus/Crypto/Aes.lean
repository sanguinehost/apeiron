/-
FIPS-197 AES-128.

The state is `Vector (BitVec 8) 16`, the 16 bytes of the block in the standard's column-major
order: byte index `4 * c + r` is row `r` of column `c`, so the four bytes of column `c` are the
contiguous indices `4c, 4c + 1, 4c + 2, 4c + 3` and the input block `in[0] … in[15]` of Appendix B
is the state read straight off. `sbox` is the FIPS-197 Figure 7 substitution table read as the
16x16 array `sbox[16 * hi + lo]`, flattened row-major into a `Vector (BitVec 8) 256`;
`invSboxTable` is the Figure 14 inverse table, built as the compositional inverse of the same
256-entry map. `xtime` is multiplication by `x` in `GF(2^8)` modulo the AES polynomial
`x^8 + x^4 + x^3 + x + 1` (FIPS-197 §5.1.1), and `gfPowX` iterates it.

`shiftRows` rotates row `r` left by `r`, `invShiftRows` rotates it right by `r`, `mixColumn`
applies the §5.1.1 matrix to each column of four bytes (index `4c + r`) and `invMixColumn` the
§5.3.1 matrix, `addRoundKey` xors a round key in. `keyExpansion` is §5.2: from the previous
round key, `RotWord` of the last word, `SubWord`, the round constant `Rcon[i/4]` in byte 0, and a
word-wise chain `w'[0] = w[0] ^ temp`, `w'[i] = w[i] ^ w[i-1]`. `encryptBlock` is §5.1
(AddRoundKey, then rounds 1…9 with MixColumns, then the final round without), `decryptBlock` is the
equivalent inverse §6.1 running the round keys in reverse.

Known-answer tests at the foot of the file are FIPS-197 Appendix C.1 and a second pair, both
confirmed against `openssl enc -aes-128-ecb -nopad`.
-/

import Init.Data.BitVec
import Init.Data.Vector

set_option autoImplicit false
set_option warningAsError true
-- The two 256-entry table literals are 256-deep nested constructor applications.
set_option maxRecDepth 4096

namespace Apeiron.Crypto.Aes

/-- One AES byte. -/
abbrev Byte := BitVec 8

/-- The AES state: 16 bytes, column-major, index `4 * c + r` is row `r` of column `c`. -/
abbrev State := Vector Byte 16

/-! ## Substitution tables (FIPS-197 Figures 7 and 14) -/

/-- The FIPS-197 Figure 7 S-box, flattened row-major: entry `16 * hi + lo` is the table cell
`(hi, lo)`. A `#v` literal of 256 ground bytes, so a table read is one `get`. -/
def sboxTable : Vector Byte 256 :=
  #v[
  0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
  0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
  0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
  0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
  0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
  0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
  0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
  0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
  0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
  0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
  0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
  0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
  0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
  0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
  0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
  0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]

/-- The FIPS-197 Figure 14 inverse S-box, flattened row-major with the same `16 * hi + lo`
convention as `sboxTable`. -/
def invSboxTable : Vector Byte 256 :=
  #v[
  0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
  0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
  0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
  0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
  0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
  0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
  0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
  0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
  0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
  0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
  0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
  0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
  0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
  0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
  0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
  0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d]

/-- FIPS-197 Figure 7 SubBytes: the table cell `(b >>> 4, b &&& 0x0f)`. O(1): two shifts, one
mask, one lookup. -/
def sbox (b : Byte) : Byte := sboxTable.getD ((b >>> 4).toNat * 16 + (b &&& 0x0f).toNat) 0

/-- FIPS-197 Figure 14 InvSubBytes: the same cell read from `invSboxTable`. O(1). -/
def invSbox (b : Byte) : Byte := invSboxTable.getD ((b >>> 4).toNat * 16 + (b &&& 0x0f).toNat) 0

/-! ## GF(2^8) arithmetic (FIPS-197 §4.2, §5.1.1) -/

/-- FIPS-197 §4.2 multiplication by `x`: a left shift reduced modulo the AES polynomial
`x^8 + x^4 + x^3 + x + 1`, i.e. `0x1b` is xored in exactly when bit 7 was set. O(1). -/
def xtime (b : Byte) : Byte :=
  (b <<< 1) ^^^ (if (b &&& 0x80) != 0 then 0x1b else 0x00)

/-- `gfPowX k b` is `x ^ k * b` in `GF(2^8)`: `k` applications of `xtime`. O(k). -/
def gfPowX : Nat → Byte → Byte
  | 0, b => b
  | k + 1, b => gfPowX k (xtime b)

/-! ## The round transformations -/

/-- FIPS-197 §5.1 SubBytes: `sbox` in every byte. O(1). -/
def subBytes (s : State) : State := Vector.ofFn fun i : Fin 16 => sbox (s.getD i.val 0)

/-- FIPS-197 §5.3 InvSubBytes: `invSbox` in every byte. O(1). -/
def invSubBytes (s : State) : State := Vector.ofFn fun i : Fin 16 => invSbox (s.getD i.val 0)

/-- FIPS-197 §5.2 ShiftRows: row `r` rotated left by `r` positions, so output byte `4c + r` is
input byte `4 * ((c + r) % 4) + r`. The four rows are disjoint, so the rotation order is
immaterial. O(1). -/
def shiftRows (s : State) : State :=
  Vector.ofFn fun i =>
    let c := i.val / 4
    let r := i.val % 4
    s.getD (4 * ((c + r) % 4) + r) 0

/-- FIPS-197 §5.3 InvShiftRows: row `r` rotated right by `r`, i.e. output byte `4c + r` is input
byte `4 * ((c + 4 - r % 4) % 4) + r`. O(1). -/
def invShiftRows (s : State) : State :=
  Vector.ofFn fun i =>
    let c := i.val / 4
    let r := i.val % 4
    s.getD (4 * ((c + 4 - r % 4) % 4) + r) 0

/-- FIPS-197 §5.1 MixColumns: the matrix `[2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]` in
`GF(2^8)` applied to each column, the four bytes at indices `4c … 4c + 3`. The four columns are
disjoint. Each output byte is four table-free products, so O(1) per state. -/
def mixColumn (s : State) : State :=
  Vector.ofFn fun i =>
    let c := i.val / 4
    let a0 := s.getD (4 * c) 0
    let a1 := s.getD (4 * c + 1) 0
    let a2 := s.getD (4 * c + 2) 0
    let a3 := s.getD (4 * c + 3) 0
    let b0 := xtime a0 ^^^ (xtime a1 ^^^ a1) ^^^ a2 ^^^ a3
    let b1 := a0 ^^^ xtime a1 ^^^ (xtime a2 ^^^ a2) ^^^ a3
    let b2 := a0 ^^^ a1 ^^^ xtime a2 ^^^ (xtime a3 ^^^ a3)
    let b3 := (xtime a0 ^^^ a0) ^^^ a1 ^^^ a2 ^^^ xtime a3
    match i.val % 4 with
    | 0 => b0
    | 1 => b1
    | 2 => b2
    | _ => b3

/-- FIPS-197 §5.3 InvMixColumns: the matrix `[14 11 13 9; 9 14 11 13; 13 9 14 11; 11 13 9 14]`
in `GF(2^8)` applied to each column. O(1) per state. -/
def invMixColumn (s : State) : State :=
  Vector.ofFn fun i =>
    let c := i.val / 4
    let a0 := s.getD (4 * c) 0
    let a1 := s.getD (4 * c + 1) 0
    let a2 := s.getD (4 * c + 2) 0
    let a3 := s.getD (4 * c + 3) 0
    let m9 b := gfPowX 3 b ^^^ b
    let m11 b := gfPowX 3 b ^^^ xtime b ^^^ b
    let m13 b := gfPowX 3 b ^^^ xtime (xtime b) ^^^ b
    let m14 b := gfPowX 3 b ^^^ xtime (xtime b) ^^^ xtime b
    let b0 := m14 a0 ^^^ m11 a1 ^^^ m13 a2 ^^^ m9 a3
    let b1 := m9 a0 ^^^ m14 a1 ^^^ m11 a2 ^^^ m13 a3
    let b2 := m13 a0 ^^^ m9 a1 ^^^ m14 a2 ^^^ m11 a3
    let b3 := m11 a0 ^^^ m13 a1 ^^^ m9 a2 ^^^ m14 a3
    match i.val % 4 with
    | 0 => b0
    | 1 => b1
    | 2 => b2
    | _ => b3

/-- FIPS-197 §5.1 AddRoundKey: the round key xored in bytewise; the key shares the state layout. -/
def addRoundKey (s k : State) : State := Vector.ofFn fun i => s.getD i.val 0 ^^^ k.getD i.val 0

/-! ## Key expansion (FIPS-197 §5.2) -/

/-- The §5.2 round constant `Rcon[i] = x ^ (i-1)` for `i = 1 … 10`, here indexed from 0:
`rcon 0 = 0x01` and each further value is `xtime` of the previous one. O(i). -/
def rcon : Nat → Byte := fun i => (List.range i).foldl (fun b _ => xtime b) 0x01

/-- One step of the §5.2 key schedule: from the round key `prev` (four words of four bytes at
indices `0…3, 4…7, 8…11, 12…15`) and the round constant, `RotWord` of the last word followed by
`SubWord` and the constant in byte 0 gives `temp`, and the new key is `w'[0] = w[0] ^ temp`,
`w'[1] = w[1] ^ w[0]`, `w'[2] = w[2] ^ w[1]`, `w'[3] = w[3] ^ w[2]`. O(1). -/
def nextRoundKey (prev : State) (rc : Byte) : State :=
  let w0 i := prev.getD i 0
  let w1 i := prev.getD (4 + i) 0
  let w2 i := prev.getD (8 + i) 0
  let w3 i := prev.getD (12 + i) 0
  let t0 := sbox (w3 1) ^^^ rc
  let t1 := sbox (w3 2)
  let t2 := sbox (w3 3)
  let t3 := sbox (w3 0)
  let temp i := match i with
    | 0 => t0
    | 1 => t1
    | 2 => t2
    | _ => t3
  let n0 i := w0 i ^^^ temp i
  let n1 i := w1 i ^^^ n0 i
  let n2 i := w2 i ^^^ n1 i
  let n3 i := w3 i ^^^ n2 i
  Vector.ofFn fun i =>
    let b := i.val / 4
    let o := i.val % 4
    match b with
    | 0 => n0 o
    | 1 => n1 o
    | 2 => n2 o
    | _ => n3 o

/-- FIPS-197 §5.2 key expansion: the 11 round keys of AES-128, `rks[0]` the cipher key itself
and `rks[10]` the last generated one. Round key `i` is `i` steps of `nextRoundKey` from the key,
each step applying `rcon m` to produce round key `m + 1` from round key `m`. The count `n`
decreases by one per step, so `go` closes after `n` steps: `Vector.ofFn` calls it eleven times,
O(11 · 16) byte operations and eleven 16-byte keys. -/
def keyExpansion (key : State) : Vector State 11 :=
  let rec go (n : Nat) (s : State) : State :=
    match n with
    | 0 => s
    | m + 1 => nextRoundKey (go m s) (rcon m)
  Vector.ofFn fun i => go i.val key

/-! ## Cipher and inverse cipher -/

/-- One §5.1 round: SubBytes, ShiftRows, MixColumns, AddRoundKey. O(1) per state. -/
def encRound (rk s : State) : State := addRoundKey (mixColumn (shiftRows (subBytes s))) rk

/-- One §5.3 inverse round, the inverse of `encRound`: InvShiftRows, InvSubBytes, AddRoundKey,
InvMixColumns. O(1) per state. -/
def decRound (rk s : State) : State := invMixColumn (addRoundKey (invSubBytes (invShiftRows s)) rk)

/-- The §5.1 final round: no MixColumns. O(1). -/
def encFinalRound (rk s : State) : State := addRoundKey (shiftRows (subBytes s)) rk

/-- The §5.3 inverse final round, the inverse of `encFinalRound`. O(1). -/
def decFinalRound (rk s : State) : State := addRoundKey (invSubBytes (invShiftRows s)) rk

/-- FIPS-197 §5.1 AES-128 encryption of one block: AddRoundKey with `rks[0]`, then rounds 1…9
`encRound` with `rks[1] … rks[9]` in that order, then the final round with `rks[10]`. The fold is
over the nine-element list `1 … 9` and closes after it, so the cost is nine rounds plus one final
round over a 16-byte state: O(1) time and O(1) space, 10 round keys and 10 state copies. -/
def encryptBlock (key plaintext : State) : State :=
  let rks := keyExpansion key
  let afterRound0 := addRoundKey plaintext (rks.getD 0 0)
  let afterRounds := (List.range 9).foldl (fun s r => encRound (rks.getD (r + 1) 0) s) afterRound0
  encFinalRound (rks.getD 10 0) afterRounds

/-- FIPS-197 §6.1 AES-128 decryption of one block: the equivalent inverse cipher — AddRoundKey with
`rks[10]`, then the inverse rounds with `rks[9] … rks[1]` in that order, then the inverse final
round with `rks[0]`. O(1) time and O(1) space, ten rounds over a 16-byte state. -/
def decryptBlock (key ciphertext : State) : State :=
  let rks := keyExpansion key
  let afterRound0 := addRoundKey ciphertext (rks.getD 10 0)
  let afterRounds := (List.range 9).foldl (fun s r => decRound (rks.getD (9 - r) 0) s) afterRound0
  decFinalRound (rks.getD 0 0) afterRounds

/-! ## Known-answer tests -/

private def hexNib (c : Char) : Nat :=
  let n := c.toNat
  if n >= '0'.toNat && n <= '9'.toNat then n - '0'.toNat
  else if n >= 'a'.toNat && n <= 'f'.toNat then n - 'a'.toNat + 10
  else if n >= 'A'.toNat && n <= 'F'.toNat then n - 'A'.toNat + 10
  else 0

private def hexByte (cs : List Char) (i : Nat) : Byte :=
  let hi := hexNib (cs.getD (2 * i) '0')
  let lo := hexNib (cs.getD (2 * i + 1) '0')
  BitVec.ofNat 8 (hi * 16 + lo)

/-- Sixteen bytes from a 32-character hex string, in order. -/
def stateOfHex (s : String) : State :=
  let cs := s.toList
  Vector.ofFn fun i => hexByte cs i.val

private def hexOfState (s : State) : String :=
  let d := "0123456789abcdef".toList
  String.ofList <| (s.toList.map fun b =>
    [d.getD (b.toNat / 16) '0', d.getD (b.toNat % 16) '0']).flatten

/-- FIPS-197 Appendix C.1 AES-128: key `000102…0f`, plaintext `001122…eeff`. -/
private def c1key : State := stateOfHex "000102030405060708090a0b0c0d0e0f"
private def c1pt : State := stateOfHex "00112233445566778899aabbccddeeff"
private def c1ct : State := stateOfHex "69c4e0d86a7b0430d8cdb78070b4c55a"

-- Appendix C.1 key schedule, FIPS-197 Appendix A.1.
#guard (keyExpansion c1key).getD 0 0 == c1key
#guard (keyExpansion c1key).getD 1 0 == stateOfHex "d6aa74fdd2af72fadaa678f1d6ab76fe"
#guard (keyExpansion c1key).getD 2 0 == stateOfHex "b692cf0b643dbdf1be9bc5006830b3fe"
#guard (keyExpansion c1key).getD 10 0 == stateOfHex "13111d7fe3944a17f307a78b4d2b30c5"

-- The round-1 state of Appendix C.1 §B: SubBytes, ShiftRows and MixColumns of the plaintext
-- after the initial AddRoundKey.
#guard addRoundKey c1pt c1key == stateOfHex "00102030405060708090a0b0c0d0e0f0"
#guard mixColumn (shiftRows (subBytes (addRoundKey c1pt c1key)))
  == stateOfHex "5f72641557f5bc92f7be3b291db9f91a"

-- Appendix C.1 ciphertext.
#guard encryptBlock c1key c1pt == c1ct

-- The same block under the second pair, checked against `openssl enc -aes-128-ecb -nopad`.
private def k2 : State := stateOfHex "2b7e151628aed2a6abf7158809cf4f3c"
private def p2 : State := stateOfHex "101112131415161718191a1b1c1d1e1f"
private def c2 : State := stateOfHex "c84af0b613435d5d9182801a9bd9320b"

#guard encryptBlock k2 p2 == c2

-- Appendix C.1 round trip: the inverse cipher recovers the plaintext.
#guard decryptBlock c1key c1ct == c1pt

-- The same round trip on the second key and plaintext.
#guard decryptBlock k2 c2 == p2


end Apeiron.Crypto.Aes