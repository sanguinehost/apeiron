/-
Specification-confidence web for the canonical FIPS specifications, adapted from CatCrypt's
specification-confidence method (`reference/catcrypt/catcrypt.txt` §2.1 "Specification confidence",
§6 "GenAI Methodology").

The canonical corpus modules here are hand transcriptions of FIPS PUB 180-4 and FIPS PUB 202:

* `repos/apeiron/proof/corpus/Crypto/Sha2.lean` (`Apeiron.Crypto.Sha2`)
* `repos/apeiron/proof/corpus/Crypto/Sponge.lean` (namespace `Apeiron.Crypto.Keccak`)
* `repos/apeiron/proof/corpus/Crypto/Keccak/Round.lean`

A transcription error in one of those definitions survives every downstream refinement: a refinement
proves `extracted ≡ canonical`, and a wrong canonical is wrong on both sides. CatCrypt's answer for
security definitions is a *web of formally proved implications between alternative formulations of
the same notion*, so that "a bug in any single definition would break at least one implication in
the web" (§2.1). This file builds the analogous web for *functional specifications*: two independent
transcriptions of the same standard text, and a proved equivalence between them.

Every definition in the `SpecWeb.*` namespaces below is transcribed from the named FIPS section and
is **not** copied from the corpus module it is compared against. Each theorem names the corpus
definition it is an implication about. `#guard` known-answer checks run the untrusted evaluator and
are labelled as evaluator evidence, not kernel proofs.

Nothing here edits the corpus, a pin, or an extracted body, and nothing is committed.
-/

import Crypto.Sha2
import Crypto.Sponge

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false

namespace SpecWeb

/-! # 0 · Attribution, and how this differs from the method it ports

`reference/catcrypt/catcrypt.txt` §2.1, read in place:

> "To further build confidence in our security definitions, we have constructed a web of formally
> proved implications between alternative formulations of the same security notion. [...] A bug in
> any single definition would break at least one implication in the web, providing an additional
> layer of assurance beyond what any individual cross-check could offer."

The same section names two companion sources of confidence: systematic cross-referencing against
published sources ("we have systematically cross-referenced all 138 security bounds against their
published sources (see Section 6)") and reuse ("over 40 examples have been ported from The Joy of
Cryptography"). §6 adds: "The size of the library and the reuse of definitions and assumptions across
many protocols provides additional confidence that the specifications are correct."

What is ported here is the *web* mechanism. What is **not** ported is the object. A security notion
(`IND-CPA`, `EUF-CMA`, ...) has several genuinely distinct formulations whose relative strength is a
mathematical question, so an implication between two of them carries information about security. A
functional specification has one denotation: the two formulations here are two encodings of the same
standard text, and the implication between them is a statement about transcription, not about
security. Consequences, stated precisely:

| | CatCrypt §2.1 | this file |
|---|---|---|
| object of the web | security notions, `IND-CPA` / `IND$` / semantic security | functional specifications, SHA-256 and SHA-3 padding |
| what an edge means | a reduction between two games | a proved equality between two transcriptions of one FIPS section |
| what a broken edge means | the notion or the reduction is mis-stated | one transcription is mis-transcribed |
| reuse argument (§6) | 172 protocols share definitions | `Crypto/Sha2` and `Crypto/Sponge` are reused by `Sha256Certificate`, `Keccak/SpongeCertificate`, `Hmac`, `Transcript`, `Merkle` |
| constants | part of the reduction proof | **outside the web**: `Sha2.K`, `Sha2.H0`, `RC`, `ROT` are literal tables, so no equivalence can check a wrong entry in them; only KATs can |

**What this web catches.** A transcription error that changes the value computed: a transposed byte
order, a wrong place value, an off-by-one in a zero count, a branch on the wrong residue, a zero
count that truncates. Each of those makes one side of an edge false.

**What it cannot catch.**
1. *Both transcriptions wrong in the same way.* Two encodings of the same paragraph read by the same
   reader agree on the same misreading. The method raises the number of independent readings that
   must coincide; it does not make any reading correct.
2. *Spec-versus-standard divergence where the standard itself is ambiguous.* An edge proves the two
   formulations agree, not that either matches the intent of FIPS 180-4 §5.1.1 or FIPS 202 §5.1. The
   KATs are the only link to the published intent, and they are evaluator evidence.
3. *Wrong constants.* `Sha2.K` and `Sha2.H0` are quoted tables, shared by both formulations. See the
   table above.
4. *Anything not reached by an edge.* §5 of this file names the open pairs; a primitive with no
   second formulation has no web at all. -/

/-! # 1 · Pair A — SHA-256 message padding

FIPS PUB 180-4 §5.1.1 (preprocessing / padding):

> "Append a single bit '1' to the message. Append `k` zero bits, where `k` is the least number
>  non-negative integer such that the resulting message has a length of ℓ = 512 multiple. Finally,
>  append the 64-bit representation of the length ℓ of the original message (before this
>  preprocessing has been added) most significant byte first."

Canonical formulation (`Crypto/Sha2.lean:133-136`, `Apeiron.Crypto.Sha2.pad`): the zero count comes
from the padded *total* length, `total - |m| - 9`, and the length field is
`lengthBlock n = List.ofFn fun i => lengthByte (8 * n) i` (`Crypto/Sha2.lean:128`), byte `i` read as
`(8 * n) >>> (8 * (7 - i)) % 256`.

Second formulation (`Sha256B` below): the zero count comes from the *position of the appended `1`
bit*, and the length field is an explicit most-significant-first list of eight base-256 digits. -/

namespace Sha256B

/-- A SHA-256 message byte. -/
abbrev Byte := BitVec 8

/-- The canonical FIPS 180-4 §5.1.1 padding, `Apeiron.Crypto.Sha2.pad`
(`repos/apeiron/proof/corpus/Crypto/Sha2.lean:133`). Named so that every theorem below states which
corpus definition it is an implication about. -/
abbrev padCanonical (m : List Byte) : List Byte := Apeiron.Crypto.Sha2.pad m

/-- The canonical FIPS 180-4 §5.1.1 length field, `Apeiron.Crypto.Sha2.lengthBlock`
(`Crypto/Sha2.lean:128`). -/
abbrev lengthBlockCanonical (n : Nat) : List Byte := Apeiron.Crypto.Sha2.lengthBlock n

/-- The canonical FIPS 180-4 §4.2.2/§5.3.3 initial hash value, `Apeiron.Crypto.Sha2.H0`
(`Crypto/Sha2.lean:46`). -/
abbrev H0Canonical : Vector (BitVec 32) 8 := Apeiron.Crypto.Sha2.H0

/-- The canonical FIPS 180-4 §6.2.2 compression, `Apeiron.Crypto.Sha2.compress`
(`Crypto/Sha2.lean:96`). -/
abbrev compressCanonical (h : Vector (BitVec 32) 8) (b : Vector (BitVec 32) 16) :
    Vector (BitVec 32) 8 := Apeiron.Crypto.Sha2.compress h b

/-- A 512-bit message block as sixteen big-endian words. -/
abbrev Block := Vector (BitVec 32) 16

/-- FIPS 180-4 §5.1.1: the `k` of "append `k` zero bits", counted from the position of the appended
`1` bit. After that bit the message has `|m| + 1` bytes, and eight length bytes must close the last
block, so the zeros fill up to the block's eighth-from-last byte. `(|m| + 1) % 64 = 56` is the
boundary at which no zero byte fits, and `|m| + 1 ≡ 0 (mod 64)` is the boundary at which a whole
block of zeros is inserted. The `+ 64` is needed because ℕ subtraction truncates. O(1). -/
def zeroFill (n : Nat) : Nat := ((56 + 64 - (n + 1) % 64) % 64)

/-- FIPS 180-4 §5.1.1: base-256 digit `j` of `x`, `j` counted from the least significant end. O(1). -/
def beByte (x : Nat) (j : Nat) : Byte := (x >>> (8 * j)) % 256

/-- FIPS 180-4 §5.1.1: "the 64-bit representation of the length of the original message, most
significant byte first", written out digit by digit. O(1). -/
def be64 (x : Nat) : List Byte :=
  [beByte x 7, beByte x 6, beByte x 5, beByte x 4, beByte x 3, beByte x 2, beByte x 1, beByte x 0]

/-- FIPS 180-4 §5.1.1 applied to a byte list `m`: `m`, the `0x80` byte, `zeroFill |m|` zero bytes and
the eight big-endian length bytes. O(|m|) time and space. -/
def padB (m : List Byte) : List Byte :=
  m ++ [0x80] ++ List.replicate (zeroFill m.length) 0 ++ be64 (8 * m.length)

/-- FIPS 180-4 §5.1.1 read as a total length: the padded message has `|m| + 1` bytes plus the zero
fill plus the eight length bytes, and that total is a multiple of 64. O(1). -/
def totalOf (n : Nat) : Nat :=
  if (n + 9) % 64 = 0 then n + 9 else n + 9 + (64 - (n + 9) % 64)

/-- FIPS 180-4 §6.2.2 step 1: word `i` of a message block as the base-256 reading of bytes
`o … o+3` most significant first, accumulated in Horner form, so the place values are produced by
the accumulation rather than named. O(1). -/
def wordB (p : List Byte) (o : Nat) : BitVec 32 :=
  BitVec.ofNat 32
    (((((p.getD o 0).toNat * 256 + (p.getD (o + 1) 0).toNat) * 256
        + (p.getD (o + 2) 0).toNat) * 256) + (p.getD (o + 3) 0).toNat)

/-- The `k`-th 64-byte block of `p` as sixteen big-endian words. O(64) time. -/
def blockB (p : List Byte) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 => wordB p (64 * k + 4 * i.val)

/-- FIPS 180-4 §6.2.1 through the second padding: `compress` folded over the blocks of `padB m`
from `H0`. The compression function is the canonical one and this file makes no claim about it
(§5, obligation E2). O(|m|) time and O(|m|) space. -/
def hashB (m : List Byte) : Vector (BitVec 32) 8 :=
  (List.range ((padB m).length / 64)).foldl
    (fun h k => compressCanonical h (blockB (padB m) k)) H0Canonical

/-- SHA-256 of a packed byte array through the second padding. O(|m|) time and space. -/
def sha256B (m : ByteArray) : Vector (BitVec 32) 8 :=
  hashB (m.toList.map fun b => (b.toNat : BitVec 8))

/-! ### List infrastructure

An eight-byte list literal (`be64`) and `List.ofFn` (`lengthBlockCanonical`) have no common list
presentation, so the two encoders are compared through `getElem?`. -/

/-- Two lists with the same `getElem?` at every index are equal. Induction on the first list,
generalising the second. -/
theorem eq_of_getElem?_eq {α : Type} (a b : List α) (h : ∀ i : Nat, a[i]? = b[i]?) : a = b := by
  induction a generalizing b with
  | nil =>
      cases b with
      | nil => rfl
      | cons y b =>
          have h0 := h 0
          simp at h0
  | cons x a ih =>
      cases b with
      | nil =>
          have h0 := h 0
          simp at h0
      | cons y b =>
          have h0 := h 0
          simp at h0
          subst h0
          congr 1
          exact ih b fun i => by simpa using h (i + 1)

theorem range8_getElem?_of_lt (i : Nat) (hi : i < 8) : (List.range 8)[i]? = some i :=
  List.getElem?_range hi

theorem range8_getElem?_ge (i : Nat) (hi : 8 ≤ i) : (List.range 8)[i]? = none :=
  List.getElem?_eq_none (by simp; omega)

theorem ofFn8_getElem?_of_lt (f : Fin 8 → Byte) (i : Nat) (hi : i < 8) :
    (List.ofFn f)[i]? = some (f ⟨i, by omega⟩) := by
  rw [List.getElem?_eq_getElem (h := by simp; omega), List.getElem_ofFn (h := by simp; omega)]

theorem ofFn8_getElem?_ge (f : Fin 8 → Byte) (i : Nat) (hi : 8 ≤ i) :
    (List.ofFn f)[i]? = none :=
  List.getElem?_eq_none (by simp; omega)

/-! ### Edge A1 — the 64-bit big-endian length field

`be64` (second formulation: an explicit list of eight base-256 digits, most significant first)
against `lengthBlockCanonical` (canonical: `List.ofFn` over `Fin 8`, byte `i` read at shift
`8 * (7 - i)`). FIPS 180-4 §5.1.1 supplies both readings of "most significant byte first"; an
implementation that transposed the digit order satisfies neither. -/

theorem be64_getElem?_of_lt (x : Nat) (i : Nat) (hi : i < 8) :
    (be64 x)[i]? = some (beByte x (7 - i)) := by
  show ((List.range 8).map (fun j : Nat => beByte x (7 - j)))[i]? = some (beByte x (7 - i))
  rw [List.getElem?_map, range8_getElem?_of_lt _ hi]
  rfl

theorem be64_getElem?_ge (x : Nat) (i : Nat) (hi : 8 ≤ i) : (be64 x)[i]? = none := by
  show ((List.range 8).map (fun j : Nat => beByte x (7 - j)))[i]? = none
  rw [List.getElem?_map, range8_getElem?_ge _ hi]
  rfl

theorem lengthBlock_getElem?_of_lt (n : Nat) (i : Nat) (hi : i < 8) :
    (lengthBlockCanonical n)[i]? = some (Apeiron.Crypto.Sha2.lengthByte (8 * n) ⟨i, by omega⟩) :=
  ofFn8_getElem?_of_lt _ _ hi

theorem lengthBlock_getElem?_ge (n : Nat) (i : Nat) (hi : 8 ≤ i) :
    (lengthBlockCanonical n)[i]? = none :=
  ofFn8_getElem?_ge _ _ hi

/-- **Web edge A1.** The two transcriptions of the FIPS 180-4 §5.1.1 length field agree, for every
message length. -/
theorem be64_eq_lengthBlock (n : Nat) : be64 (8 * n) = lengthBlockCanonical n := by
  refine eq_of_getElem?_eq _ _ fun i => ?_
  by_cases hi : i < 8
  · rw [be64_getElem?_of_lt _ _ hi, lengthBlock_getElem?_of_lt _ _ hi]
    rfl
  · rw [be64_getElem?_ge _ _ (by omega), lengthBlock_getElem?_ge _ _ (by omega)]

/-! ### Edge A2 — the zero fill

The canonical `pad` derives the zero count from the padded total length; `zeroFill` derives it from
the position of the `0x80` byte. They must agree at every length, including the two boundaries:
`|m| + 1 ≡ 56 (mod 64)`, where the count is `0`, and `|m| + 1 ≡ 0 (mod 64)`, where it jumps to `56`.
-/

theorem totalOf_ge (n : Nat) : n + 9 ≤ totalOf n := by
  unfold totalOf
  split <;> rename_i h <;> omega

theorem totalOf_mod (n : Nat) : totalOf n % 64 = 0 := by
  unfold totalOf
  split <;> rename_i h <;> omega

/-- **Web edge A2.** The two transcriptions of the `k` of "append `k` zero bits" agree, for every
message length. -/
theorem zeroFill_eq (n : Nat) : zeroFill n = totalOf n - n - 9 := by
  unfold zeroFill totalOf
  split <;> rename_i h <;> omega

/-- `n + 1 + zeroFill n + 8` is the padded total, i.e. the two transcriptions agree on the total
length as well as on the zero count. -/
theorem totalOf_eq (n : Nat) : n + 1 + zeroFill n + 8 = totalOf n := by
  rw [zeroFill_eq]
  have h := totalOf_ge n
  omega

/-! ### Edge A3 — the padding itself -/

/-- **Web edge A3.** The two transcriptions of FIPS 180-4 §5.1.1 produce the same byte list, for
every message. A defect in the zero count of either one breaks edge A2 and this edge; a defect in the
length field of either one breaks edge A1 and this edge. -/
theorem padB_eq_pad (m : List Byte) : padB m = padCanonical m := by
  show m ++ [0x80] ++ List.replicate (zeroFill m.length) 0 ++ be64 (8 * m.length)
      = m ++ [0x80] ++ List.replicate (totalOf m.length - m.length - 9) 0
        ++ lengthBlockCanonical m.length
  rw [be64_eq_lengthBlock, zeroFill_eq]

/-- The second formulation's padded message has the padded total length. -/
theorem padB_length (m : List Byte) : (padB m).length = totalOf m.length := by
  have h := totalOf_eq m.length
  simp only [padB, be64, List.length_append, List.length_replicate, List.length_cons,
    List.length_nil]
  omega

/-- **Edge A3a.** The padded message of the second formulation is a whole number of 64-byte blocks.
This is the property `Sha2.hash` relies on when it fixes its fold count at `pad m |>.length / 64`. -/
theorem padB_length_mod (m : List Byte) : (padB m).length % 64 = 0 := by
  rw [padB_length, totalOf_mod]

/-- The zero fill takes the values the §5.1.1 boundary argument requires: `0` when the `1` bit sits
in the eighth-from-last byte of a block, `56` when it sits in the last byte of a block. -/
theorem zeroFill_boundary_55 : zeroFill 55 = 0 := by decide

theorem zeroFill_boundary_56 : zeroFill 56 = 63 := by decide

theorem zeroFill_boundary_63 : zeroFill 63 = 56 := by decide

theorem zeroFill_boundary_64 : zeroFill 64 = 55 := by decide

/-- A defect in `zeroFill` is caught: the two boundary values of the §5.1.1 minimum-zero argument
are distinct and both are forced by `zeroFill_eq`. -/
theorem zeroFill_lt_56 (n : Nat) : zeroFill n < 64 := by
  unfold zeroFill
  omega

/-! ### Edge A4 — the big-endian word regrouping, at closed instances

`wordB` is the Horner-form reading of four bytes; the canonical regrouping is the shift-and-or
accumulation of `Sha2.paddedBlock` (`Crypto/Sha2.lean:139`, `private`, so not nameable from here)
and the closed-form `beWordAt` of `work/proofs/Sha256Blocks.lean:103`, another owner's file. The
general edge is blocked by that `private` (§5, obligation E1); the instances below are kernel
checked by `decide` over four bytes and four `Nat` products. -/

theorem wordB_of_ascii : wordB [0x61, 0x62, 0x63, 0x64] 0 = 0x61626364 := by decide

theorem wordB_of_ascii_offset :
    wordB [0x00, 0x61, 0x62, 0x63, 0x64, 0x00] 1 = 0x61626364 := by decide

theorem wordB_order :
    wordB [0x01, 0x02, 0x03, 0x04] 0 = 0x01020304
      ∧ wordB [0x04, 0x03, 0x02, 0x01] 0 = 0x04030201 := by decide

theorem wordB_reads_out_of_bounds_as_zero : wordB [0xaa] 0 = 0xaa000000 := by decide

/-! ### Edge A5 — the composite: measured blocker, not a theorem

`padB_eq_pad` is a component edge. The composite `sha256B m = Apeiron.Crypto.Sha2.sha256 m` is not
available here, for two independent reasons, both measured.

1. *The block reader cannot be named.* `Sha2.paddedBlock` (`Crypto/Sha2.lean:139`) is `private`, so
   the canonical 64-byte block reader cannot be mentioned outside the module and the two folds over
   blocks cannot be compared. `work/proofs/Sha256Blocks.lean:47-49` and
   `work/proofs/AlgebraSha256d.lean:224-225` record the same blocker.
2. *The closed instances do not close either.* Both sides are pure functions, so a closed-instance
   equality would be a `rfl` on two whole-SHA-256 evaluations. Measured: the goal

   ```text
   (Apeiron.Crypto.Sha2.sha256 "abc".toUTF8).toList
     = (Apeiron.Crypto.Sha2.hash [0x61, 0x62, 0x63]).toList := by rfl
   ```

   a *single-block* message, dies with `(deterministic) timeout at whnf, maximum number of heartbeats
   (200000) has been reached` after 47 s (log `/tmp/opencode/spec-web-probe11.log`). `State` is
   `Vector Word 8`, i.e. `Array`, so `isDefEq` must evaluate the entire compression to compare the
   two arrays. Raising `maxHeartbeats` would be the forbidden "reduction that needs the guard
   defeated" (`AGENTS.md` §6, §10), so the composite SHA-256 edge stays an obligation (§5, E2).

The `#guard` checks below do make the same comparison, through the evaluator, in seconds. That is why
they are labelled evaluator evidence and not proofs of the composite. -/

/-! ### KATs, both sides

`#guard` runs the untrusted evaluator: these pin values against regressions and cross-check the two
transcriptions at the digest level. They are **evaluator evidence, not kernel proofs**; the kernel
proofs of this pair are edges A1, A2, A3, A3a and the closed instances above.

Published vectors: FIPS 180-4 §B.1 (`"abc"` and the 56-byte example) and the NIST vectors already
pinned at `Crypto/Sha2.lean:161-174`. -/

#guard (sha256B ByteArray.empty).toList ==
  [0xe3b0c442, 0x98fc1c14, 0x9afbf4c8, 0x996fb924, 0x27ae41e4, 0x649b934c, 0xa495991b, 0x7852b855]

#guard (sha256B "abc".toUTF8).toList ==
  [0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61, 0xf20015ad]

#guard (sha256B "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".toUTF8).toList ==
  [0x248d6a61, 0xd20638b8, 0xe5c02693, 0x0c3e6039, 0xa33ce459, 0x64ff2167, 0xf6ecedd4, 0x19db06c1]

-- The two transcriptions agree at the digest level, for every published vector above.
#guard sha256B "abc".toUTF8 = Apeiron.Crypto.Sha2.sha256 "abc".toUTF8
#guard sha256B ByteArray.empty = Apeiron.Crypto.Sha2.sha256 ByteArray.empty
#guard sha256B "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".toUTF8 =
  Apeiron.Crypto.Sha2.sha256 "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".toUTF8

-- Padding boundary: the extra block appears at `|m| = 56`, not at `|m| = 55`.
#guard (padB (List.replicate 55 0)).length == 64
#guard (padB (List.replicate 56 0)).length == 128
#guard (padB (List.replicate 64 0)).length == 128
#guard (padB []).length == (padB (List.replicate 55 0)).length

-- The two length-field transcriptions, at the §5.1.1 boundaries: `8 · 55 = 440 = 0x01B8`.
#guard be64 (8 * 55) == [0, 0, 0, 0, 0, 0, 1, 184]
#guard be64 (8 * 56) == [0, 0, 0, 0, 0, 0, 1, 192]
#guard be64 (8 * 63) == [0, 0, 0, 0, 0, 0, 1, 248]
#guard be64 (8 * 64) == [0, 0, 0, 0, 0, 0, 2, 0]

end Sha256B

/-! # 2 · Pair B — FIPS 202 sponge padding

FIPS PUB 202 §5.1 (`pad10*1` with a domain suffix), at byte granularity:

> "Append the domain separation bits `d` to the message. [...] Append `1` and then the minimum
>  number of zeroes needed for the final block to have a length of ℓ = r bits."

Canonical formulation (`Crypto/Sponge.lean:66-71`, `Apeiron.Crypto.Keccak.pad`): branch on
`|m| % rate = rate - 1`, the `0x86` merge case.

Second formulation (`KeccakB` below): branch on the position `(n + 1) % rate` of the *domain* byte
inside the final block; position `0` means that byte is the block's last byte, so it absorbs the `1`
bit. -/

namespace KeccakB

/-- A message byte. -/
abbrev Byte := BitVec 8

/-- The canonical FIPS 202 §5.1 padding, `Apeiron.Crypto.Keccak.pad`
(`repos/apeiron/proof/corpus/Crypto/Sponge.lean:66`). -/
abbrev padCanonical (rate : Nat) (domain : Byte) (m : List Byte) : List Byte :=
  Apeiron.Crypto.Keccak.pad rate domain m

/-- FIPS 202 §5.1: the byte-granular `pad10*1` tail of a message of `n` bytes, read off the position
`(n + 1) % rate` of the domain byte inside the final block. Position `0` means the domain byte is the
block's last byte, so it absorbs the `1` bit and the tail is the single byte `domain ^^^ 0x80`;
otherwise the `1` bit needs a further byte, and the zero count is what is left of the block. The
count is a truncated `Nat` subtraction, so the function is total for every `rate`; the equivalence
with the canonical padding is claimed at the published rates below, not at `rate = 0`. O(rate) time
and space. -/
def padTail (rate : Nat) (domain : Byte) (n : Nat) : List Byte :=
  if (n + 1) % rate = 0 then [domain ^^^ 0x80]
  else [domain] ++ List.replicate (rate - 1 - ((n + 1) % rate)) 0 ++ [0x80]

/-- FIPS 202 §5.1 applied to a byte list. O(|m| + rate) time and space. -/
def padK (rate : Nat) (domain : Byte) (m : List Byte) : List Byte :=
  m ++ padTail rate domain m.length

/-- FIPS 202 §4 absorb of an already padded message, through the canonical `Apeiron.Crypto.Keccak.blocks`,
`xorBlock` and `permute`, using the second padding. O(|m| + rate · permute) time, O(|m|) space. -/
def absorbK (rate : Nat) (domain : Byte) (m : List Byte) : Vector (BitVec 64) 25 :=
  (Apeiron.Crypto.Keccak.blocks rate (padK rate domain m)).foldl
    (fun s block => Apeiron.Crypto.Keccak.permute (Apeiron.Crypto.Keccak.xorBlock rate s block))
    (Vector.replicate 25 0)

/-- SHA3-256 through the second padding: rate 1088 bits (136 bytes), domain `0x06`, §6.1. -/
def sha3_256B (m : List Byte) : Vector Byte 32 :=
  Apeiron.Crypto.Keccak.squeeze 136 (absorbK 136 0x06 m) 32

/-- SHA3-512 through the second padding: rate 576 bits (72 bytes), domain `0x06`, §6.1. -/
def sha3_512B (m : List Byte) : Vector Byte 64 :=
  Apeiron.Crypto.Keccak.squeeze 72 (absorbK 72 0x06 m) 64

/-- SHAKE128 through the second padding: rate 1344 bits (168 bytes), domain `0x1f`, §6.2. -/
def shake128B (m : List Byte) (outLen : Nat) : Vector Byte outLen :=
  Apeiron.Crypto.Keccak.squeeze 168 (absorbK 168 0x1f m) outLen

/-- SHAKE256 through the second padding: rate 1088 bits (136 bytes), domain `0x1f`, §6.2. -/
def shake256B (m : List Byte) (outLen : Nat) : Vector Byte outLen :=
  Apeiron.Crypto.Keccak.squeeze 136 (absorbK 136 0x1f m) outLen

/-! ### The parameter literals

The second formulation writes the rates as literals, so its edges attach to the corpus's named
functions only through these four `rfl` facts. -/

theorem rate_sha3_256_eq : Apeiron.Crypto.Keccak.rate_sha3_256 = 136 := rfl

theorem rate_sha3_512_eq : Apeiron.Crypto.Keccak.rate_sha3_512 = 72 := rfl

theorem rate_shake128_eq : Apeiron.Crypto.Keccak.rate_shake128 = 168 := rfl

theorem rate_shake256_eq : Apeiron.Crypto.Keccak.rate_shake256 = 136 := rfl

theorem domain_sha3_eq : Apeiron.Crypto.Keccak.domain_sha3 = 0x06 := rfl

theorem domain_shake_eq : Apeiron.Crypto.Keccak.domain_shake = 0x1f := rfl

/-! ### Edge B1 — the sponge padding, at the four published rates

Each theorem is a web edge between the second transcription (`padK`) and the canonical
`Apeiron.Crypto.Keccak.pad`, at one published rate. The two branch on different quantities — the
canonical on `|m| % rate = rate - 1` (`Crypto/Sponge.lean:67`), the second on
`(|m| + 1) % rate = 0` — so each edge carries the modular-arithmetic step relating them.
`Apeiron.Crypto.Keccak.pad` is parameterised by an arbitrary `rate`; the general statement is left
open (§5, obligation E4). -/

private theorem modAdd_136 (k : Nat) (hk : k < 136) :
    ((k + 1) % 136 = 0) <-> (k = 135) := by
  constructor
  · intro h
    rcases Nat.lt_or_ge (k + 1) 136 with h1 | h1
    · rw [Nat.mod_eq_of_lt h1] at h
      omega
    · omega
  · intro h
    subst h
    exact Nat.mod_self 136

theorem padK_eq_pad_136 (domain : Byte) (m : List Byte) :
    padK 136 domain m = padCanonical 136 domain m := by
  have hm : (m.length + 1) % 136 = (m.length % 136 + 1) % 136 :=
    (Nat.mod_add_mod m.length 136 1).symm
  have hk : m.length % 136 < 136 := Nat.mod_lt _ (by decide)
  have hab : 136 - 1 - (m.length % 136 + 1) = 136 - m.length % 136 - 2 := by omega
  unfold padK padTail padCanonical Apeiron.Crypto.Keccak.pad
  by_cases h : m.length % 136 = 135
  · have hj0 : (m.length + 1) % 136 = 0 := by omega
    simp only [hj0, h, if_pos]
  · have hj : (m.length % 136 + 1) % 136 = m.length % 136 + 1 := by
      rw [Nat.mod_eq_of_lt (show m.length % 136 + 1 < 136 by omega)]
    have hne : ¬ (m.length % 136 + 1 = 0) := by omega
    simp only [hm, hj, if_neg hne, if_neg h, hab, List.append_assoc]

private theorem modAdd_72 (k : Nat) (hk : k < 72) :
    ((k + 1) % 72 = 0) <-> (k = 71) := by
  constructor
  · intro h
    rcases Nat.lt_or_ge (k + 1) 72 with h1 | h1
    · rw [Nat.mod_eq_of_lt h1] at h
      omega
    · omega
  · intro h
    subst h
    exact Nat.mod_self 72

theorem padK_eq_pad_72 (domain : Byte) (m : List Byte) :
    padK 72 domain m = padCanonical 72 domain m := by
  have hm : (m.length + 1) % 72 = (m.length % 72 + 1) % 72 :=
    (Nat.mod_add_mod m.length 72 1).symm
  have hk : m.length % 72 < 72 := Nat.mod_lt _ (by decide)
  have hab : 72 - 1 - (m.length % 72 + 1) = 72 - m.length % 72 - 2 := by omega
  unfold padK padTail padCanonical Apeiron.Crypto.Keccak.pad
  by_cases h : m.length % 72 = 71
  · have hj0 : (m.length + 1) % 72 = 0 := by omega
    simp only [hj0, h, if_pos]
  · have hj : (m.length % 72 + 1) % 72 = m.length % 72 + 1 := by
      rw [Nat.mod_eq_of_lt (show m.length % 72 + 1 < 72 by omega)]
    have hne : ¬ (m.length % 72 + 1 = 0) := by omega
    simp only [hm, hj, if_neg hne, if_neg h, hab, List.append_assoc]

private theorem modAdd_168 (k : Nat) (hk : k < 168) :
    ((k + 1) % 168 = 0) <-> (k = 167) := by
  constructor
  · intro h
    rcases Nat.lt_or_ge (k + 1) 168 with h1 | h1
    · rw [Nat.mod_eq_of_lt h1] at h
      omega
    · omega
  · intro h
    subst h
    exact Nat.mod_self 168

theorem padK_eq_pad_168 (domain : Byte) (m : List Byte) :
    padK 168 domain m = padCanonical 168 domain m := by
  have hm : (m.length + 1) % 168 = (m.length % 168 + 1) % 168 :=
    (Nat.mod_add_mod m.length 168 1).symm
  have hk : m.length % 168 < 168 := Nat.mod_lt _ (by decide)
  have hab : 168 - 1 - (m.length % 168 + 1) = 168 - m.length % 168 - 2 := by omega
  unfold padK padTail padCanonical Apeiron.Crypto.Keccak.pad
  by_cases h : m.length % 168 = 167
  · have hj0 : (m.length + 1) % 168 = 0 := by omega
    simp only [hj0, h, if_pos]
  · have hj : (m.length % 168 + 1) % 168 = m.length % 168 + 1 := by
      rw [Nat.mod_eq_of_lt (show m.length % 168 + 1 < 168 by omega)]
    have hne : ¬ (m.length % 168 + 1 = 0) := by omega
    simp only [hm, hj, if_neg hne, if_neg h, hab, List.append_assoc]

/-! ### Edge B2 — absorb, and the named functions, in general

Edge B1 is a component edge. Because `absorbK` reuses the canonical `Apeiron.Crypto.Keccak.blocks`,
`xorBlock`, `permute` and `squeeze` and differs from `Apeiron.Crypto.Keccak.absorb` only in its
padding, edge B1 transports directly: the composite equivalences below hold for every message, not
only at closed instances. -/

theorem absorbK_eq_absorb_136 (domain : Byte) (m : List Byte) :
    absorbK 136 domain m = Apeiron.Crypto.Keccak.absorb 136 domain m := by
  simp only [absorbK, Apeiron.Crypto.Keccak.absorb]
  rw [padK_eq_pad_136 domain m]

theorem absorbK_eq_absorb_72 (domain : Byte) (m : List Byte) :
    absorbK 72 domain m = Apeiron.Crypto.Keccak.absorb 72 domain m := by
  simp only [absorbK, Apeiron.Crypto.Keccak.absorb]
  rw [padK_eq_pad_72 domain m]

theorem absorbK_eq_absorb_168 (domain : Byte) (m : List Byte) :
    absorbK 168 domain m = Apeiron.Crypto.Keccak.absorb 168 domain m := by
  simp only [absorbK, Apeiron.Crypto.Keccak.absorb]
  rw [padK_eq_pad_168 domain m]

/-- **Web edge B2, SHA3-256.** FIPS 202 §6.1 SHA3-256 through the second padding equals
`Apeiron.Crypto.Keccak.sha3_256`, for every message. -/
theorem sha3_256B_eq_sha3_256 (m : List Byte) :
    sha3_256B m = Apeiron.Crypto.Keccak.sha3_256 m := by
  simp only [sha3_256B, Apeiron.Crypto.Keccak.sha3_256, Apeiron.Crypto.Keccak.sponge,
    Apeiron.Crypto.Keccak.rate_sha3_256, Apeiron.Crypto.Keccak.domain_sha3,
    absorbK_eq_absorb_136]

/-- **Web edge B2, SHA3-512.** FIPS 202 §6.1 SHA3-512, for every message. -/
theorem sha3_512B_eq_sha3_512 (m : List Byte) :
    sha3_512B m = Apeiron.Crypto.Keccak.sha3_512 m := by
  simp only [sha3_512B, Apeiron.Crypto.Keccak.sha3_512, Apeiron.Crypto.Keccak.sponge,
    Apeiron.Crypto.Keccak.rate_sha3_512, Apeiron.Crypto.Keccak.domain_sha3,
    absorbK_eq_absorb_72]

/-- **Web edge B2, SHAKE128.** FIPS 202 §6.2 SHAKE128, for every message and output length. -/
theorem shake128B_eq_shake128 (m : List Byte) (outLen : Nat) :
    shake128B m outLen = Apeiron.Crypto.Keccak.shake128 m outLen := by
  simp only [shake128B, Apeiron.Crypto.Keccak.shake128, Apeiron.Crypto.Keccak.sponge,
    Apeiron.Crypto.Keccak.rate_shake128, Apeiron.Crypto.Keccak.domain_shake,
    absorbK_eq_absorb_168]

/-- **Web edge B2, SHAKE256.** FIPS 202 §6.2 SHAKE256, for every message and output length. -/
theorem shake256B_eq_shake256 (m : List Byte) (outLen : Nat) :
    shake256B m outLen = Apeiron.Crypto.Keccak.shake256 m outLen := by
  simp only [shake256B, Apeiron.Crypto.Keccak.shake256, Apeiron.Crypto.Keccak.sponge,
    Apeiron.Crypto.Keccak.rate_shake256, Apeiron.Crypto.Keccak.domain_shake,
    absorbK_eq_absorb_136]

/-! ### KATs, both sides

`#guard` runs the untrusted evaluator: evaluator evidence, not kernel proofs. The published vectors
are the FIPS-202 examples already pinned at `Crypto/Sponge.lean:125-163`. -/

#guard (sha3_256B ([] : List Byte)).toList == (Apeiron.Crypto.Keccak.sha3_256 ([] : List Byte)).toList
#guard (sha3_256B [0x61, 0x62, 0x63]).toList
    == (Apeiron.Crypto.Keccak.sha3_256 [0x61, 0x62, 0x63]).toList
#guard (sha3_256B (List.replicate 135 (0 : Byte))).toList
    == (Apeiron.Crypto.Keccak.sha3_256 (List.replicate 135 (0 : Byte))).toList
#guard (sha3_256B (List.replicate 136 (0 : Byte))).toList
    == (Apeiron.Crypto.Keccak.sha3_256 (List.replicate 136 (0 : Byte))).toList
#guard (sha3_512B (List.replicate 144 0x61)).toList
    == (Apeiron.Crypto.Keccak.sha3_512 (List.replicate 144 0x61)).toList
#guard (shake128B (List.replicate 300 0x61) 40).toList
    == (Apeiron.Crypto.Keccak.shake128 (List.replicate 300 0x61) 40).toList
#guard (shake256B ([] : List Byte) 32).toList
    == (Apeiron.Crypto.Keccak.shake256 ([] : List Byte) 32).toList

-- The second transcription reproduces the published digest on its own.
#guard (sha3_256B [0x61, 0x62, 0x63]).toList ==
  [0x3a, 0x98, 0x5d, 0xa7, 0x4f, 0xe2, 0x25, 0xb2, 0x04, 0x5c, 0x17, 0x2d, 0x6b, 0xd3, 0x90, 0xbd,
   0x85, 0x5f, 0x08, 0x6e, 0x3e, 0x9d, 0x52, 0x5b, 0x46, 0xbf, 0xe2, 0x45, 0x11, 0x43, 0x15, 0x32]

-- The `0x86` merge case `|m| % rate = rate - 1`, and the block-count boundaries around it.
#guard (padK 136 0x06 (List.replicate 134 0)).length == 136
#guard (padK 136 0x06 (List.replicate 135 0)).length == 136
#guard (padK 136 0x06 (List.replicate 136 0)).length == 272
#guard (padK 136 0x06 []).length == (padK 136 0x06 (List.replicate 134 0)).length
#guard (padTail 136 0x06 135) == [0x86]
#guard (padTail 136 0x06 136).length == 136
#guard (padTail 136 0x06 134) == [0x06, 0x80]

end KeccakB

/-! # 3 · Existing members of the web, cited not proved here

* `repos/apeiron/proof/corpus/Crypto/Keccak/SpongeCertificate.lean:292`,
  `apeiron_word.absorbLane_eq_foldLane` — the extracted `U64::from_le_bytes` of one absorb iteration
  is the `foldLane` of `Crypto/Keccak/SpongeCertificate.lean:167`, which is the same per-lane
  eight-byte accumulation `Sponge.xorBlock` runs. This is one instance of the method, between the
  Rust extraction and the canonical lane read, already `#audit`ed at
  `repos/apeiron/proof/corpus/Crypto/Keccak/Audit.lean:92`. Grade `proved`; cited here.
* `Crypto/Keccak/SpongeCertificate.lean:191`, `foldLane_eq_fromLEBytes`
  (`Crypto/Keccak/Audit.lean:112`).
* `work/proofs/Sha256Blocks.lean:160`, `beWordAt_eq_fold` — the big-endian word regrouping in two
  spellings, against the canonical `Sha2.paddedBlock` regrouping.
* `work/proofs/AlgebraSha256Sigma.lean`, `tsigma0`/`tsigma1`/`tSigma0`/`tSigma1`, and
  `work/proofs/AlgebraSha256Sched.lean:612,616` — width-generic second transcriptions of the
  FIPS 180-4 §4.1.2/§4.1.3 round functions, equal to `Apeiron.Crypto.Sha2.sigma0`/`sigma1`/
  `Sigma0`/`Sigma1` at width 32. Another owner's files, cited at their own grade.
-/

#print axioms SpecWeb.Sha256B.eq_of_getElem?_eq
#print axioms SpecWeb.Sha256B.range8_getElem?_of_lt
#print axioms SpecWeb.Sha256B.range8_getElem?_ge
#print axioms SpecWeb.Sha256B.ofFn8_getElem?_of_lt
#print axioms SpecWeb.Sha256B.ofFn8_getElem?_ge
#print axioms SpecWeb.Sha256B.be64_getElem?_of_lt
#print axioms SpecWeb.Sha256B.be64_getElem?_ge
#print axioms SpecWeb.Sha256B.lengthBlock_getElem?_of_lt
#print axioms SpecWeb.Sha256B.lengthBlock_getElem?_ge
#print axioms SpecWeb.Sha256B.be64_eq_lengthBlock
#print axioms SpecWeb.Sha256B.totalOf_ge
#print axioms SpecWeb.Sha256B.totalOf_mod
#print axioms SpecWeb.Sha256B.zeroFill_eq
#print axioms SpecWeb.Sha256B.totalOf_eq
#print axioms SpecWeb.Sha256B.padB_eq_pad
#print axioms SpecWeb.Sha256B.padB_length
#print axioms SpecWeb.Sha256B.padB_length_mod
#print axioms SpecWeb.Sha256B.zeroFill_boundary_55
#print axioms SpecWeb.Sha256B.zeroFill_boundary_56
#print axioms SpecWeb.Sha256B.zeroFill_boundary_63
#print axioms SpecWeb.Sha256B.zeroFill_boundary_64
#print axioms SpecWeb.Sha256B.zeroFill_lt_56
#print axioms SpecWeb.Sha256B.wordB_of_ascii
#print axioms SpecWeb.Sha256B.wordB_of_ascii_offset
#print axioms SpecWeb.Sha256B.wordB_order
#print axioms SpecWeb.Sha256B.wordB_reads_out_of_bounds_as_zero
#print axioms SpecWeb.KeccakB.rate_sha3_256_eq
#print axioms SpecWeb.KeccakB.rate_sha3_512_eq
#print axioms SpecWeb.KeccakB.rate_shake128_eq
#print axioms SpecWeb.KeccakB.rate_shake256_eq
#print axioms SpecWeb.KeccakB.domain_sha3_eq
#print axioms SpecWeb.KeccakB.domain_shake_eq
#print axioms SpecWeb.KeccakB.padK_eq_pad_136
#print axioms SpecWeb.KeccakB.padK_eq_pad_72
#print axioms SpecWeb.KeccakB.padK_eq_pad_168
#print axioms SpecWeb.KeccakB.absorbK_eq_absorb_136
#print axioms SpecWeb.KeccakB.absorbK_eq_absorb_72
#print axioms SpecWeb.KeccakB.absorbK_eq_absorb_168
#print axioms SpecWeb.KeccakB.sha3_256B_eq_sha3_256
#print axioms SpecWeb.KeccakB.sha3_512B_eq_sha3_512
#print axioms SpecWeb.KeccakB.shake128B_eq_shake128
#print axioms SpecWeb.KeccakB.shake256B_eq_shake256

end SpecWeb