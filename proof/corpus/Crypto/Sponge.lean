/-
FIPS PUB 202: the SHA-3 and SHAKE sponge over `Vector (BitVec 64) 25`.

The state is the Keccak-f[1600] state of `Crypto.Keccak.Round`: 25 little-endian 64-bit lanes,
lane index `x + 5 * y`. State byte `k` is byte `k % 8` of lane `k / 8`, least significant byte
first (§3.1.2), so messages are byte lists. `pad` is the byte-granular `pad10*1` of §5.1 with the
SHA-3 (`01`, byte `0x06`) and SHAKE (`1111`, byte `0x1f`) domain separators; the final `1` is XORed
as `0x80` into the last rate byte. `absorb` (§4) XORs each rate-byte block into the state and
permutes; `squeeze` (§4) reads `rate` bytes per output block and permutes between blocks. The named
functions are §6.1 SHA3-256 / SHA3-512 and §6.2 SHAKE128 / SHAKE256; rates are byte rates.
-/

import Crypto.Keccak.Round

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Keccak

/-- One little-endian 64-bit Keccak lane. -/
abbrev Lane := BitVec 64

/-- The Keccak-f[1600] state: 25 lanes, lane index `x + 5 * y` (FIPS-202 §3.1). -/
abbrev State := Vector Lane 25

/-! ## Parameters (FIPS-202 §6) -/

/-- SHA3-256 byte rate: 1088 bits. -/
def rate_sha3_256 : Nat := 136

/-- SHA3-512 byte rate: 576 bits. -/
def rate_sha3_512 : Nat := 72

/-- SHAKE128 byte rate: 1344 bits. -/
def rate_shake128 : Nat := 168

/-- SHAKE256 byte rate: 1088 bits. -/
def rate_shake256 : Nat := 136

/-- SHA-3 domain separator: suffix `01` followed by the `pad10*1` leading `1` (`0b00000110`). -/
def domain_sha3 : BitVec 8 := 0x06

/-- SHAKE domain separator: suffix `1111` followed by the `pad10*1` leading `1`. -/
def domain_shake : BitVec 8 := 0x1f

/-! ## Bytes, padding and absorption -/

/-- FIPS-202 §3.1.2 byte `k` of the state: byte `k % 8` of lane `k / 8`. The lane index is reduced
mod 25, so the function is total for arbitrary `k`. O(1). -/
def byteOf (s : State) (k : Nat) : BitVec 8 :=
  (s.get ⟨(k / 8) % 25, Nat.mod_lt (k / 8) (by decide : 0 < 25)⟩).extractLsb' (8 * (k % 8)) 8

/-- XOR a `rate`-byte block into the state in FIPS-202 byte order: byte `8 * i + j` goes to byte
`j` of lane `i`. Bytes past the 25 lanes are ignored; O(rate) time, O(1) space. -/
def xorBlock (rate : Nat) (s : State) (block : Vector (BitVec 8) rate) : State :=
  Vector.ofFn fun i : Fin 25 =>
    s.get i ^^^
      (List.range 8).foldl
        (fun acc j => acc ^^^ ((block.getD (8 * i.val + j) 0).setWidth 64 <<< (8 * j)))
        (0 : BitVec 64)

/-- FIPS-202 §5.1 `pad10*1` with domain separation, at byte granularity: append `domain`, then
zero bytes, then XOR `0x80` into the last byte of the final rate block; when the message ends
`rate - 1` bytes into a block the two bytes merge into `domain ^^^ 0x80`. For `0 < rate` the result
length is a multiple of `rate`. O(|m|) time, O(|m| + rate) space. -/
def pad (rate : Nat) (domain : BitVec 8) (m : List (BitVec 8)) : List (BitVec 8) :=
  let k := m.length % rate
  if k = rate - 1 then
    m ++ [domain ^^^ 0x80]
  else
    m ++ [domain] ++ List.replicate (rate - k - 2) 0 ++ [0x80]

/-- Split `l` into `n` consecutive `rate`-byte blocks, zero-padding the tail of a short block. -/
private def chunk (rate : Nat) : Nat → List (BitVec 8) → List (Vector (BitVec 8) rate)
  | 0, _ => []
  | n + 1, l => Vector.ofFn (fun i : Fin rate => l.getD i.val 0) :: chunk rate n (l.drop rate)

/-- The rate-byte blocks of `l`: `(|l| + rate - 1) / rate` consecutive blocks, the last zero-padded
when `rate ∤ |l|`. O(|l|) time, O(|l|) space. -/
def blocks (rate : Nat) (l : List (BitVec 8)) : List (Vector (BitVec 8) rate) :=
  chunk rate ((l.length + rate - 1) / rate) l

/-- FIPS-202 §4 absorb from the zero state: `pad` the message, XOR each rate-byte block into the
state, and apply `permute` after each block. O(|m| + ⌈|m| / rate⌉ · permute) time, O(|m|) space. -/
def absorb (rate : Nat) (domain : BitVec 8) (m : List (BitVec 8)) : State :=
  (blocks rate (pad rate domain m)).foldl (fun s block => permute (xorBlock rate s block))
    (Vector.replicate 25 0)

/-! ## Squeezing and the named functions -/

/-- FIPS-202 §4 squeeze: output byte `i` is byte `i % rate` of the state after `i / rate`
permutations, i.e. output block `k` is read from `permute^[k]` of the absorbed state. Total for
every `n`; `n = 0` gives the empty output. O(n + ⌈n / rate⌉ · permute) time, O(n) space. -/
def squeeze (rate : Nat) (s : State) (n : Nat) : Vector (BitVec 8) n :=
  Vector.ofFn fun i : Fin n => byteOf (Nat.repeat permute (i.val / rate) s) (i.val % rate)

/-- The FIPS-202 §4 sponge: `squeeze (absorb m)`, with `rate` in bytes, `domain` the domain
separator, and `outLen` output bytes. -/
def sponge (rate : Nat) (domain : BitVec 8) (m : List (BitVec 8)) (outLen : Nat) :
    Vector (BitVec 8) outLen :=
  squeeze rate (absorb rate domain m) outLen

/-- FIPS-202 §6.1 SHA3-256: rate 1088 bits, domain `0x06`, 256-bit digest. -/
def sha3_256 (m : List (BitVec 8)) : Vector (BitVec 8) 32 :=
  sponge rate_sha3_256 domain_sha3 m 32

/-- FIPS-202 §6.1 SHA3-512: rate 576 bits, domain `0x06`, 512-bit digest. -/
def sha3_512 (m : List (BitVec 8)) : Vector (BitVec 8) 64 :=
  sponge rate_sha3_512 domain_sha3 m 64

/-- FIPS-202 §6.2 SHAKE128: rate 1344 bits, domain `0x1f`, `outLen` output bytes. -/
def shake128 (m : List (BitVec 8)) (outLen : Nat) : Vector (BitVec 8) outLen :=
  sponge rate_shake128 domain_shake m outLen

/-- FIPS-202 §6.2 SHAKE256: rate 1088 bits, domain `0x1f`, `outLen` output bytes. -/
def shake256 (m : List (BitVec 8)) (outLen : Nat) : Vector (BitVec 8) outLen :=
  sponge rate_shake256 domain_shake m outLen

/-! ## Known-answer checks

`#guard` runs the untrusted evaluator, so these pin byte order, padding and rate selection against
regressions but are not kernel proofs. References: FIPS-202 examples and `hashlib` for the
multi-block inputs. -/

#guard (sha3_256 []).toList ==
  [0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66, 0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61, 0xd6, 0x62,
   0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa, 0x82, 0xd8, 0x0a, 0x4b, 0x80, 0xf8, 0x43, 0x4a]

#guard (sha3_256 [0x61, 0x62, 0x63]).toList ==
  [0x3a, 0x98, 0x5d, 0xa7, 0x4f, 0xe2, 0x25, 0xb2, 0x04, 0x5c, 0x17, 0x2d, 0x6b, 0xd3, 0x90, 0xbd,
   0x85, 0x5f, 0x08, 0x6e, 0x3e, 0x9d, 0x52, 0x5b, 0x46, 0xbf, 0xe2, 0x45, 0x11, 0x43, 0x15, 0x32]

#guard (sha3_512 []).toList ==
  [0xa6, 0x9f, 0x73, 0xcc, 0xa2, 0x3a, 0x9a, 0xc5, 0xc8, 0xb5, 0x67, 0xdc, 0x18, 0x5a, 0x75, 0x6e,
   0x97, 0xc9, 0x82, 0x16, 0x4f, 0xe2, 0x58, 0x59, 0xe0, 0xd1, 0xdc, 0xc1, 0x47, 0x5c, 0x80, 0xa6,
   0x15, 0xb2, 0x12, 0x3a, 0xf1, 0xf5, 0xf9, 0x4c, 0x11, 0xe3, 0xe9, 0x40, 0x2c, 0x3a, 0xc5, 0x58,
   0xf5, 0x00, 0x19, 0x9d, 0x95, 0xb6, 0xd3, 0xe3, 0x01, 0x75, 0x85, 0x86, 0x28, 0x1d, 0xcd, 0x26]

-- Two exact blocks: the padding is a full extra block (`0x06`, zeros, `0x80`).
#guard (sha3_512 (List.replicate 144 0x61)).toList ==
  [0x44, 0x6c, 0xd4, 0xd7, 0xba, 0x19, 0x51, 0x0d, 0xcc, 0x77, 0x6b, 0x21, 0x04, 0x5b, 0xc6, 0x8d,
   0x42, 0x4b, 0x5b, 0x84, 0x0e, 0x14, 0x68, 0x5e, 0x14, 0x9b, 0xb2, 0x38, 0xb5, 0xf4, 0x73, 0xc0,
   0x35, 0x6b, 0x69, 0xe0, 0x4f, 0x0f, 0x57, 0x85, 0xee, 0xfc, 0xe2, 0x0f, 0xf0, 0x9e, 0x67, 0x8b,
   0x08, 0x0d, 0x8a, 0xac, 0x64, 0x56, 0x8c, 0x5e, 0xdf, 0x00, 0x1c, 0xd3, 0x2b, 0x2e, 0xd7, 0xa8]

-- `k = rate - 1`: the domain byte and the final `0x80` merge into `0x86`.
#guard (sha3_256 (List.replicate 135 0)).toList ==
  [0x7d, 0x08, 0x0d, 0x7b, 0xa9, 0x78, 0xa7, 0x5c, 0x8a, 0x7d, 0x1f, 0x9b, 0xe5, 0x66, 0xc8, 0x59,
   0x08, 0x45, 0x09, 0xc9, 0xc2, 0xb4, 0x92, 0x84, 0x35, 0xc2, 0x25, 0xd5, 0x77, 0x7d, 0x98, 0xe3]

#guard (shake128 (List.replicate 72 0x61) 32).toList ==
  [0x7b, 0xfe, 0x63, 0x48, 0xbd, 0x84, 0x0a, 0xe7, 0x89, 0x32, 0x55, 0x24, 0xa6, 0x86, 0x55, 0x57,
   0xbf, 0x4b, 0x3d, 0x30, 0xf1, 0xf8, 0x95, 0xf8, 0xd5, 0x21, 0x0e, 0x79, 0xed, 0x82, 0xe2, 0xc5]

-- Two absorb blocks (300 = 168 + 132, padding closes the second) and a 40-byte output.
#guard (shake128 (List.replicate 300 0x61) 40).toList ==
  [0xd5, 0xfb, 0x6e, 0x23, 0x90, 0x91, 0xe1, 0x7b, 0x86, 0x38, 0xf2, 0x16, 0x7b, 0xc6, 0xee, 0xa2,
   0xe7, 0xae, 0xad, 0x90, 0x37, 0xd7, 0x05, 0xa0, 0x67, 0xbc, 0xbc, 0x75, 0x5f, 0x66, 0xb9, 0xd4,
   0xa6, 0x49, 0x07, 0xdc, 0x46, 0x66, 0xbb, 0x02]

#guard (shake256 [] 32).toList ==
  [0x46, 0xb9, 0xdd, 0x2b, 0x0b, 0xa8, 0x8d, 0x13, 0x23, 0x3b, 0x3f, 0xeb, 0x74, 0x3e, 0xeb, 0x24,
   0x3f, 0xcd, 0x52, 0xea, 0x62, 0xb8, 0x1b, 0x82, 0xb5, 0x0c, 0x27, 0x64, 0x6e, 0xd5, 0x76, 0x2f]

-- Padding length is a multiple of the rate in all three regimes.
#guard (pad 136 0x06 []).length == 136
#guard (pad 136 0x06 (List.replicate 135 0)).length == 136
#guard (pad 136 0x06 (List.replicate 136 0)).length == 272

end Apeiron.Crypto.Keccak
