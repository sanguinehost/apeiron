/-
BLAKE3 — the tree hash of O'Connor, Gentle, Neves, Wilcox-O'Hearn and the BLAKE3 team, ported
from the Rust crate `work/crypto-blake3` (`src/lib.rs`: `compress`, `round`, `permute`,
`ChunkState`, `chunk_output`, `parent_output`, `hash`).

`IV` is the SHA-256 initialisation vector of FIPS 180-4 §5.3.3, used as the chaining-value seed;
`MSG_PERMUTATION` is the message-word permutation applied between rounds, replacing BLAKE2's
repeated `SIGMA` schedule; `g` is the mixing function with rotations 16/12/8/7; `round` is the
eight `g` calls on the four columns then the four diagonals; `compress` runs seven rounds and
returns all sixteen state words (the eight-word chaining value is their first half).

`Output` is a node ready to emit: the input chaining value, the block words, the 64-bit chunk
counter, the block length in bytes and the domain-separation flags. `chunkOutput` is the Rust
`ChunkState`: the blocks already compressed are `⌊(|m|−1)/64⌋`, so the last block is compressed by
`chunkOutput` itself with `blockLen = |m| mod 64` — which is `64`, not `0`, when `|m|` is a multiple
of the block size — and carries `CHUNK_END`, plus `CHUNK_START` when no block has been compressed
yet. `parentOutput` is the internal tree node: the two child chaining values in one block,
`counter = 0`, `blockLen = 64`, `PARENT` set. `hash` compresses the root node with `ROOT` set and
keeps the low 32 bytes of the state, which is the extended-output stream's first block.

## Scope

In scope: the compression function with the message permutation, the chunk/block state with
`CHUNK_START`, `CHUNK_END`, `PARENT`, `ROOT`, and the one-shot `hash` of any message of at most
`CHUNK_LEN = 1024` bytes, where the tree is a single chunk and the chunk output is the root node.

Out of scope, and therefore absent: `hashAll` for `|m| > 1024`, i.e. the chaining-value stack that
folds several chunks into the tree (only the two-child parent node is given, and `PARENT` is
unreachable from `hash` in the admissible domain); the keyed mode (`KEYED_HASH`, a 32-byte key
replacing the `IV` seed) and `keyed_hash`; `derive_key` (`DERIVE_KEY_CONTEXT`, `DERIVE_KEY_MATERIAL`
and the context subkey); and the extendable output beyond the 32-byte default, i.e. the
counter-`0,1,…` stream of `root_output_bytes` past one block. The admissible domain of `hash` is
`|m| ≤ 1024`; on a longer input its value is not a BLAKE3 digest.
-/

import Init.Data.Vector
import Init.Data.ByteArray.Basic

set_option autoImplicit false
set_option warningAsError true

namespace Apeiron.Crypto.Blake3

/-- One BLAKE3 word. -/
abbrev Word := BitVec 32

/-- One input or output byte. -/
abbrev Byte := BitVec 8

/-- A chaining value `cv[0] … cv[7]`. -/
abbrev Chain := Vector Word 8

/-- The working state `v[0] … v[15]`. -/
abbrev Work := Vector Word 16

/-- One 64-byte block as sixteen little-endian words `m[0] … m[15]`. -/
abbrev Block := Vector Word 16

/-- `2^32`, the modulus of the `BitVec 32` word representation. -/
def two32 : Nat := 4294967296

/-- The chunk size: a chunk is sixteen blocks of 64 bytes. -/
def CHUNK_LEN : Nat := 1024

/-- The word `n`, reduced mod `2^32`; the construction needs no `2^32` literal in the goal. O(1). -/
def wordOfNat (n : Nat) : Word := ⟨n % 4294967296, Nat.mod_lt n (by decide)⟩

/-- The byte `n`, reduced mod `2^8`. O(1). -/
def byteOfNat (n : Nat) : Byte := ⟨n % 256, Nat.mod_lt n (by decide)⟩

/-- A `Fin 16` index from a `Nat`, reduced mod 16. O(1). -/
def i16 (k : Nat) : Fin 16 := ⟨k % 16, Nat.mod_lt k (by decide)⟩

/-- A `Fin 8` index from a `Nat`, reduced mod 8. O(1). -/
def i8 (k : Nat) : Fin 8 := ⟨k % 8, Nat.mod_lt k (by decide)⟩

/-! ## Constants (FIPS 180-4 §5.3.3; BLAKE3 reference `IV`, `MSG_PERMUTATION`) -/

/-- The SHA-256 initialisation vector, used as the BLAKE3 chaining-value seed. -/
def IV : Chain :=
  #v[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

/-- The message-word permutation applied between consecutive rounds: message word `i` of the next
round is message word `MSG_PERMUTATION[i]` of this one. -/
def MSG_PERMUTATION : Vector Nat 16 :=
  #v[2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

/-- Domain-separation flag of the first block of a chunk. -/
def CHUNK_START : Word := 1

/-- Domain-separation flag of the last block of a chunk. -/
def CHUNK_END : Word := 2

/-- Domain-separation flag of an internal tree node. -/
def PARENT : Word := 4

/-- Domain-separation flag of the root node. -/
def ROOT : Word := 8

/-! ## Mixing, rounds and the message schedule -/

/-- The mixing function `g` on the four state words `a, b, c, d` with the two message words
`x, y`. O(1): four additions and four rotations on `Word`s. -/
def g (v : Work) (a b c d : Fin 16) (x y : Word) : Work :=
  let v := v.set a (v.get a + v.get b + x)
  let v := v.set d ((v.get d ^^^ v.get a).rotateRight 16)
  let v := v.set c (v.get c + v.get d)
  let v := v.set b ((v.get b ^^^ v.get c).rotateRight 12)
  let v := v.set a (v.get a + v.get b + y)
  let v := v.set d ((v.get d ^^^ v.get a).rotateRight 8)
  let v := v.set c (v.get c + v.get d)
  v.set b ((v.get b ^^^ v.get c).rotateRight 7)

/-- One round: the four `g` calls mixing the columns, then the four mixing the diagonals. O(1):
eight `g` calls over sixteen words. -/
def round (v : Work) (m : Block) : Work :=
  let v := g v (i16 0) (i16 4) (i16 8) (i16 12) (m.get (i16 0)) (m.get (i16 1))
  let v := g v (i16 1) (i16 5) (i16 9) (i16 13) (m.get (i16 2)) (m.get (i16 3))
  let v := g v (i16 2) (i16 6) (i16 10) (i16 14) (m.get (i16 4)) (m.get (i16 5))
  let v := g v (i16 3) (i16 7) (i16 11) (i16 15) (m.get (i16 6)) (m.get (i16 7))
  let v := g v (i16 0) (i16 5) (i16 10) (i16 15) (m.get (i16 8)) (m.get (i16 9))
  let v := g v (i16 1) (i16 6) (i16 11) (i16 12) (m.get (i16 10)) (m.get (i16 11))
  let v := g v (i16 2) (i16 7) (i16 8) (i16 13) (m.get (i16 12)) (m.get (i16 13))
  g v (i16 3) (i16 4) (i16 9) (i16 14) (m.get (i16 14)) (m.get (i16 15))

/-- The message-word permutation. O(1): sixteen word reads and one vector construction. -/
def permute (m : Block) : Block :=
  Vector.ofFn fun i : Fin 16 => m.get (i16 (MSG_PERMUTATION.get i))

/-- `iterate n b` is `[b, σb, …, σⁿb]`; it terminates because the recursion is on `n`. O(n) time
and space in blocks of sixteen words. -/
def iterate : Nat → Block → List Block
  | 0, b => [b]
  | n + 1, b => b :: iterate n (permute b)

/-- The seven message schedule rows of one block, `b` through `σ⁶b`: BLAKE3 permutes the message
between the seven rounds instead of reusing BLAKE2's `SIGMA` rows. O(1): seven permutations of
sixteen words. -/
def schedule (b : Block) : List Block := iterate 6 b

/-- The state before the first round: `cv ‖ IV[0..3]`, the counter as low and high words in
`v[12]`, `v[13]`, the block length in `v[14]` and the flags in `v[15]`. O(1). -/
def initState (cv : Chain) (counter : Nat) (blockLen : Nat) (flags : Word) : Work :=
  Vector.ofFn fun i : Fin 16 =>
    if hi : i.val < 8 then cv.get ⟨i.val, hi⟩
    else if i.val < 12 then IV.get (i8 i.val)
    -- Each word goes through `wordOfNat`: an ascription `(x : Word)` on an arithmetic expression
    -- would make the `Word` division the exclusive-or-free `%` and `/` of `BitVec` and wrap the
    -- operands mod `2^32` first.
    else if i.val == 12 then wordOfNat counter
    else if i.val == 13 then wordOfNat (counter / two32)
    else if i.val == 14 then wordOfNat blockLen
    else flags

/-- The compression function: 64 bytes in, 16 words out. Seven rounds over the permuted schedule,
then `v[i] ← v[i] ⊕ v[i+8]` and `v[i+8] ← v[i+8] ⊕ cv[i]` for `i < 8`, which is the chaining value
in the low half and the root-output words in the high half. O(1): seven rounds and six
permutations over sixteen words. -/
def compress (cv : Chain) (block : Block) (counter : Nat) (blockLen : Nat) (flags : Word) : Work :=
  let s := (schedule block).foldl (fun acc m => round acc m) (initState cv counter blockLen flags)
  Vector.ofFn fun i : Fin 16 =>
    if i.val < 8 then s.get (i16 i.val) ^^^ s.get (i16 (i.val + 8))
    else s.get (i16 i.val) ^^^ cv.get (i8 (i.val + 24))

/-- The chaining value carried by a compression output: its first eight words. O(1). -/
def cvFromWords (w : Work) : Chain := Vector.ofFn fun i : Fin 8 => w.get (i16 i.val)

/-! ## Nodes and output -/

/-- A node ready to emit: the input chaining value, the block words, the 64-bit chunk counter, the
number of block bytes and the domain-separation flags. Compressing it once gives the chaining value
(the low eight words) and, with `ROOT` set, the first 32 bytes of the extended output. -/
structure Output where
  /-- The chaining value the compression starts from. -/
  inputCV : Chain
  /-- The block to compress. -/
  blockWords : Block
  /-- The chunk counter, low and high halves in `v[12]` and `v[13]`. -/
  counter : Nat
  /-- The number of message bytes in the block, in `v[14]`; `64` for a full block. -/
  blockLen : Nat
  /-- The domain-separation flags, in `v[15]`. -/
  flags : Word
  deriving Repr

/-- The chaining value of a non-root node. O(1): one compression. -/
def Output.chainingValue (o : Output) : Chain :=
  cvFromWords (compress o.inputCV o.blockWords o.counter o.blockLen o.flags)

/-- The first block of the extended output of a root node: `ROOT` ored into the flags. O(1). -/
def Output.rootWords (o : Output) : Work :=
  compress o.inputCV o.blockWords o.counter o.blockLen (o.flags ||| ROOT)

/-- The first 32 bytes of the extended output of a root node, each word little-endian. O(1): 32
byte extractions. -/
def Output.rootBytes (o : Output) : Vector Byte 32 :=
  Vector.ofFn fun i : Fin 32 =>
    -- The `show Nat from` pins the shift arithmetic in `Nat`; without it the expected `Byte`
    -- result type would make `^` the exclusive-or and the byte extract wrong.
    show Nat from ((o.rootWords.get (i16 (i.val / 4))).toNat / 2 ^ (8 * (i.val % 4)) % 256)

/-! ## Chunks and the tree -/

/-- Word `i` of block `b` of `m`, read little-endian; bytes past the end of `m` read as zero, the
zero padding of the final block. O(|m|) time per word in `List.getD` (a linear list scan) and O(1)
space. -/
def blockWord (m : List Byte) (b i : Nat) : Word :=
  wordOfNat (show Nat from
    (List.range 4).foldl (fun acc j => acc + (m.getD (64 * b + 4 * i + j) 0).toNat * 256 ^ j) 0)

/-- The 64-byte block of `m` at block index `b`, zero-padded past the end. O(64 · |m|) time in
`blockWord` and O(1) space of sixteen words. -/
def chunkBlock (m : List Byte) (b : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 => blockWord m b i.val

/-- `CHUNK_START` for the first compressed block of a chunk, `0` afterwards. O(1). -/
def startFlag (blocksCompressed : Nat) : Word :=
  if blocksCompressed == 0 then CHUNK_START else 0

/-- The number of blocks a chunk of `n` bytes compresses before its output node is formed,
`⌊(n − 1) / 64⌋`, and `0` for the empty chunk: the last block is compressed by the output node
itself, carrying `blockLen = n mod 64`, which is `64` and not `0` when `64 ∣ n`. O(1). -/
def blocksCompressed (n : Nat) : Nat := if n == 0 then 0 else (n - 1) / 64

/-- The chaining value of a chunk of `|m|` bytes after `n` blocks have been compressed, starting
from `key`, with chunk counter `counter` and domain flags `flags`. O(n · |m|) time in `compress`
and `chunkBlock` and O(1) space: the recursion carries one eight-word chaining value. -/
def chunkCV (m : List Byte) (key : Chain) (counter : Nat) (flags : Word) : Nat → Chain
  | 0 => key
  | n + 1 =>
    cvFromWords (compress (chunkCV m key counter flags n) (chunkBlock m n) counter 64
      (flags ||| startFlag n))

/-- The output node of a chunk: the chunk state after its `⌊(|m|−1)/64⌋` compressed blocks, with
`CHUNK_END` set and `CHUNK_START` set when no block has been compressed yet. The domain is any
chunk of at most `CHUNK_LEN` bytes. O(|m|) time in `chunkCV` and O(1) extra space. -/
def chunkOutput (m : List Byte) (key : Chain) (counter : Nat) (flags : Word) : Output where
  inputCV := chunkCV m key counter flags (blocksCompressed m.length)
  blockWords := chunkBlock m (blocksCompressed m.length)
  counter := counter
  blockLen := m.length - 64 * blocksCompressed m.length
  flags := flags ||| startFlag (blocksCompressed m.length) ||| CHUNK_END

/-- The output node of an internal tree node whose children are `left` and `right`: both chaining
values in one block, `counter = 0`, `blockLen = 64`, `PARENT` set. O(1). -/
def parentOutput (left right : Chain) (key : Chain) (flags : Word) : Output where
  inputCV := key
  blockWords :=
    Vector.ofFn fun i : Fin 16 =>
      if hi : i.val < 8 then left.get ⟨i.val, hi⟩ else right.get (i8 (i.val + 24))
  counter := 0
  blockLen := 64
  flags := PARENT ||| flags

/-- The one-shot unkeyed BLAKE3 digest of `m`: 32 bytes, the compressed root node of the chunk.
Admissible domain: `m.length ≤ 1024 = CHUNK_LEN`, where the tree is the single chunk `m` and the
chunk output node is the root. O(|m|) time in `chunkCV` and O(1) extra space. -/
def hash (m : List Byte) : Vector Byte 32 :=
  (chunkOutput m IV 0 0).rootBytes

/-- The one-shot unkeyed BLAKE3 digest of a packed byte array. O(|m|) time and O(|m|) space for
the list copy. -/
def blake3 (m : ByteArray) : Vector Byte 32 :=
  hash (m.toList.map fun b => (b.toNat : Byte))

/-- The official BLAKE3 test-vector input of `n` bytes: byte `i` is `i % 251`. O(n) time and space
in bytes. -/
def pattern (n : Nat) : List Byte := (List.range n).map fun i => byteOfNat (i % 251)

/-! ## Hex rendering and known-answer checks

`#guard` runs the untrusted evaluator, so these pin the IV, the schedule, the flags, the chunk
sequencing and the output order against regressions; they are not kernel proofs. The expected
values are the 32-byte prefixes of the official BLAKE3 test vectors transcribed in
`work/crypto-blake3/tests/kat.rs` (case `official_vectors_unkeyed`, whose first 131 bytes are the
extended output and whose first 32 are `hash`). -/

/-- The lowercase hex digit for `n < 16`. O(1): one list lookup. -/
def hexDigit (n : Nat) : Char :=
  ("0123456789abcdef".toList).getD (n % 16) '0'

/-- The lowercase hex encoding of a 32-byte digest. O(1) time and space: 32 bytes, 64 characters. -/
def hex (d : Vector Byte 32) : String :=
  String.ofList ((d.toList : List Byte).flatMap fun b : Byte => [hexDigit (b.toNat / 16), hexDigit b.toNat])

-- The seed, the schedule and the flags.
#guard IV.get (i8 0) == 0x6a09e667
#guard IV.get (i8 7) == 0x5be0cd19
#guard MSG_PERMUTATION.toList == [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

/-- The block whose word `i` is `i`, so that a permutation guard reads the permutation directly. -/
def idBlock : Block := Vector.ofFn fun i : Fin 16 => wordOfNat i.val

#guard (permute idBlock).toList.map BitVec.toNat == [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
#guard (schedule idBlock).length == 7
#guard ((schedule idBlock).map (fun b => b.toList.map BitVec.toNat)).take 2
  == [List.range 16, [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]]
#guard (CHUNK_START, CHUNK_END, PARENT, ROOT) == (1, 2, 4, 8)
#guard (CHUNK_START ||| CHUNK_END) == 3 && (PARENT ||| ROOT) == 12

-- The state layout before the rounds: the seed, the low IV half, the counter, the block length.
#guard (initState IV 0 0 CHUNK_START).get (i16 15) == CHUNK_START
#guard (initState IV 1 64 ROOT).get (i16 12) == 1
#guard (initState IV 0x100000000 64 ROOT).get (i16 13) == 1
#guard (initState IV 0 64 (CHUNK_END ||| ROOT)).get (i16 14) == 64

-- A block of `pattern 64`: sixteen little-endian words, the first four of block 0.
#guard (chunkBlock (pattern 64) 0).get (i16 0) == (0x03020100 : Word)
#guard (chunkBlock (pattern 64) 0).get (i16 1) == (0x07060504 : Word)
#guard (chunkBlock (pattern 64) 1).get (i16 0) == 0

-- A chunk compresses `⌊(n − 1) / 64⌋` blocks before the output node; an exact multiple of 64
-- leaves a full, not an empty, final block.
#guard blocksCompressed 0 == 0
#guard blocksCompressed 1 == 0
#guard blocksCompressed 64 == 0
#guard blocksCompressed 65 == 1
#guard blocksCompressed 1024 == 15
#guard (chunkOutput (pattern 64) IV 0 0).blockLen == 64
#guard (chunkOutput (pattern 65) IV 0 0).blockLen == 1
#guard (chunkOutput [] IV 0 0).blockLen == 0
#guard (chunkOutput (pattern 1) IV 0 0).flags == (CHUNK_START ||| CHUNK_END)
#guard (chunkOutput (pattern 64) IV 0 0).flags == (CHUNK_START ||| CHUNK_END)
#guard (chunkOutput (pattern 65) IV 0 0).flags == CHUNK_END
#guard (chunkOutput (pattern 128) IV 0 0).flags == CHUNK_END
#guard (chunkOutput (pattern 1024) IV 0 0).flags == CHUNK_END

-- A parent node puts the left chaining value in the low half of its block and the right one in the
-- high half.
#guard (parentOutput IV IV IV 0).blockWords.get (i16 0) == IV.get (i8 0)
#guard (parentOutput IV IV IV 0).blockWords.get (i16 8) == IV.get (i8 0)
#guard (parentOutput IV IV IV 0).blockLen == 64
#guard (parentOutput IV IV IV 0).counter == 0

-- The official vectors: byte `i` is `i % 251`, and the digest is the first 32 bytes of the
-- 131-byte extended output.
#guard hex (hash []) ==
  "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
#guard hex (hash (pattern 1)) ==
  "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"
#guard hex (hash (pattern 2)) ==
  "7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"
#guard hex (hash (pattern 3)) ==
  "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"
#guard hex (hash (pattern 4)) ==
  "f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"
#guard hex (hash (pattern 5)) ==
  "b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"
#guard hex (hash (pattern 6)) ==
  "06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844"
#guard hex (hash (pattern 7)) ==
  "3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe"
#guard hex (hash (pattern 8)) ==
  "2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb"
#guard hex (hash (pattern 63)) ==
  "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"
#guard hex (hash (pattern 64)) ==
  "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"
#guard hex (hash (pattern 65)) ==
  "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"
#guard hex (hash (pattern 127)) ==
  "d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d"
#guard hex (hash (pattern 128)) ==
  "f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef"
#guard hex (hash (pattern 129)) ==
  "683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12"

-- The chunk boundary of the admissible domain: 1023 bytes compress 15 blocks and 1024 bytes
-- compress 16, and the last block of 1024 is full.
set_option maxHeartbeats 4000000 in
#guard hex (hash (pattern 1023)) ==
  "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"
set_option maxHeartbeats 4000000 in
#guard hex (hash (pattern 1024)) ==
  "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"

-- The parent node of the official 2048-byte vector, which is the tree root of two chunks: out of
-- the domain of `hash`, and given by `parentOutput` alone.
set_option maxHeartbeats 8000000 in
#guard hex (parentOutput (chunkOutput ((pattern 2048).take 1024) IV 0 0).chainingValue
    ((chunkOutput ((pattern 2048).drop 1024) IV 1 0).chainingValue) IV 0).rootBytes ==
  "e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"

-- The digest of a packed byte array agrees with the digest of the byte list, and the `"abc"`
-- vector of the official test-vector file.
#guard blake3 ByteArray.empty == hash []
#guard blake3 "abc".toUTF8 ==
  hash ("abc".toUTF8.toList.map fun b => (b.toNat : Byte))
#guard hex (hash ("abc".toUTF8.toList.map fun b => (b.toNat : Byte))) ==
  "6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"

end Apeiron.Crypto.Blake3