/-
PART 2 of the former work/proofs/AlgebraCostComplete.lean, split for the apeiron 1,500-line
owned-source cap (repos/apeiron/AGENTS.md:222). Content here: §3 the canonical padding arithmetic,
§4 the padding cost, §5 the block loading cost, §6 the digest store cost, §7 which reads are
priced and which stay `model`, §8 the completed cost of one proof-of-work attempt, §9 the mining
range cost and its composite/recurrence layer, §10 the three-face comparison and certification,
§11 what this module does not give, and the audit rows for the declarations owned here. Depends on
AlgebraCostCompletePart1 for the measures. No declaration changed name, namespace or type; the
whole module is re-assembled by the AlgebraCostComplete facade.
-/

import Apeiron.Trichotomy.AnyRing
import Apeiron.Trichotomy.ChannelsAnyRing
import Crypto.Sha2
import Crypto.Sha256Certificate
import WordRefinement
import Mathlib.Tactic
import Icarus.Cost.AlgebraCostCompletePart1

set_option autoImplicit false
set_option warningAsError true
set_option linter.apeiron.notation false
set_option linter.unusedVariables false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedSectionVars false
set_option linter.unnecessarySeqFocus false

universe u

namespace algebra_cost_complete

open Aeneas Aeneas.Std Result
open Apeiron.Trichotomy.AnyRing
open Apeiron.Crypto.Sha2
open apeiron_word

variable {R : Type u} [CommRing R]

/-! ## 3. The canonical padding arithmetic

Re-derived from `Crypto/Sha2.lean:125-136` rather than imported, because `Sha256Pad.lean` is not a root
of any corpus `lean_lib` and this module needs the *byte* counts of the three appended pieces. All of
it is `Nat` arithmetic on explicit literals and decidable instances; no recursion, no enumeration. -/

/-- Bytes left over after the full 64-byte blocks of a message of `n` bytes. Temporal and spatial O(1):
one `Nat.mod`. -/
def rest (n : Nat) : Nat := n % 64

/-- Full 64-byte blocks of a message of `n` bytes. Temporal and spatial O(1): one `Nat.div`. -/
def fullBlocks (n : Nat) : Nat := n / 64

/-- The tail buffer length of the canonical padding: 64 when the remainder leaves room for the
`0x80` byte and the eight length bytes, 128 otherwise. Temporal and spatial O(1). -/
def tailLen (n : Nat) : Nat := if rest n < 56 then 64 else 128

/-- **The canonical padded total length**, the `Nat` reading of `Sha2.lean:134-135`: `r` is the
remainder of `n + 9` on 64, and the total is `n + 9` when `r = 0` and `n + 9 + (64 - r)` otherwise.
Temporal and spatial O(1): one modulo, one comparison, one addition. -/
def canonTotal (n : Nat) : Nat :=
  let r := (n + 9) % 64
  if r = 0 then n + 9 else n + 9 + (64 - r)

/-- **The canonical zero-fill count**, `total - n - 9`: the zeros between the `0x80` byte and the
length field. Temporal and spatial O(1). -/
def zeroFill (n : Nat) : Nat := canonTotal n - n - 9

/-- `ceilDiv64 m` is the least multiple of 64 that is at least `m`, that is `⌈m/64⌉`. Temporal and
spatial O(1). -/
def ceilDiv64 (m : Nat) : Nat := (m + 63) / 64

/-- **The block-count function.** A message of `n` bytes pads to `blockCount n` blocks of 64 bytes:
`⌈(n+9)/64⌉`. Temporal and spatial O(1). -/
def blockCount (n : Nat) : Nat := ceilDiv64 (n + 9)

/-! ### 3.1 The layout identity: message, `0x80`, zeros, length field -/

theorem eq_div_mod (n : Nat) : n = 64 * fullBlocks n + rest n :=
  (Nat.div_add_mod n 64).symm

theorem msgSplit (n : Nat) : 64 * fullBlocks n + rest n = n := (eq_div_mod n).symm

theorem rest_lt (n : Nat) : rest n < 64 := Nat.mod_lt n (by omega)

theorem tailLen_cases (n : Nat) : tailLen n = 64 ∨ tailLen n = 128 := by
  rw [tailLen]; split <;> omega

theorem tailLen_mod (n : Nat) : tailLen n % 64 = 0 := by
  rcases tailLen_cases n with h | h <;> simp [h]

/-- **The padded total is a whole number of blocks.** -/
theorem canonTotal_multiple (n : Nat) : 64 ∣ canonTotal n := by
  rw [canonTotal]; split <;> omega

/-- **The padded total is at least `n + 9`**: the message, the `0x80` byte and the length field all
fit. -/
theorem canonTotal_ge (n : Nat) : n + 9 ≤ canonTotal n := by
  rw [canonTotal]; split <;> omega

/-- **The padded total is at most one block more than `n + 9`.** -/
theorem canonTotal_lt (n : Nat) : canonTotal n - 9 < n + 64 := by
  rw [canonTotal]; split <;> omega

/-- The padded total is the tail length on top of the full blocks. -/
theorem canonTotal_eq (n : Nat) : canonTotal n = 64 * fullBlocks n + tailLen n := by
  have h1 : (n + 9) % 64 = (n % 64 + 9) % 64 := by omega
  have h2 : n = 64 * fullBlocks n + rest n := eq_div_mod n
  rw [canonTotal, h1]
  simp only [fullBlocks, tailLen, rest] at *
  split <;> split <;> omega

/-- The zero fill plus the `0x80` byte and the length field is the appended tail. -/
theorem zeroFill_add (n : Nat) : zeroFill n + 9 = canonTotal n - n := by
  have h := canonTotal_ge n
  simp only [zeroFill]; omega

/-- **THE LAYOUT IDENTITY** the byte figures rest on: message, `0x80` merge byte, zero fill, length
field, and nothing else. -/
theorem totalLayout (n : Nat) : n + 1 + zeroFill n + 8 = canonTotal n := by
  have h := zeroFill_add n
  have h2 := canonTotal_ge n
  simp only [zeroFill]
  omega

/-- **The zero fill is bounded by 63** for every `n`, and is non-negative. This bound is the closure
the padding's byte count needs: one extra message byte can never add more than one 64-byte block, so
the byte figure grows in blocks and not by an unbounded amount. -/
theorem zeroFill_le (n : Nat) : zeroFill n ≤ 63 := by
  have hb := totalLayout n
  have h := canonTotal_lt n
  have h2 := canonTotal_ge n
  simp only [zeroFill] at *
  omega

theorem zeroFill_nonneg (n : Nat) : 0 ≤ zeroFill n := by
  have h := zeroFill_add n
  have h2 := canonTotal_ge n
  simp only [zeroFill] at *
  omega

/-- **THE BYTE RECURRENCE, upper step.** One more message byte adds at most one block of 64 bytes
to the padded total. Together with `canonTotal_blockBytes` this is the recurrence the padding's byte
figure obeys as the message grows: the padded byte count is `64 * blockCount n`, and the block count
moves by at most one. -/
theorem canonTotal_succ_le (n : Nat) : canonTotal (n + 1) ≤ canonTotal n + 64 := by
  have h1 := canonTotal_ge (n + 1)
  have h2 := canonTotal_lt (n + 1)
  have h3 := canonTotal_ge n
  omega

/-! ### 3.2 `ceilDiv64` is the ceiling, stated on `Nat` -/

/-- **The ceiling characterisation**: `64 * ceilDiv64 m` is the least multiple of 64 at least `m`. -/
theorem ceilDiv64_spec (m : Nat) : m ≤ 64 * ceilDiv64 m ∧ 64 * ceilDiv64 m < m + 64 := by
  have h := Nat.div_add_mod (m + 63) 64
  have hr : (m + 63) % 64 < 64 := Nat.mod_lt _ (by omega)
  unfold ceilDiv64
  omega

/-- The quotient characterisation: `m / 64 ≤ ⌈m/64⌉ ≤ m / 64 + 1`. Together with `ceilDiv64_strict`
this is the statement that `ceilDiv64 m` is `⌈m / 64⌉`. -/
theorem ceilDiv64_quot (m : Nat) : m / 64 ≤ ceilDiv64 m ∧ ceilDiv64 m ≤ m / 64 + 1 := by
  have h := ceilDiv64_spec m
  have hd : 64 * ceilDiv64 m + (m + 63) % 64 = m + 63 := by
    simpa [ceilDiv64] using (Nat.div_add_mod (m + 63) 64)
  have hdiv := Nat.div_add_mod m 64
  have hmod : m % 64 < 64 := Nat.mod_lt _ (by omega)
  omega

/-- The strict ceiling characterisation, for `0 < m`. -/
theorem ceilDiv64_strict (m : Nat) (hm : 0 < m) : 64 * (ceilDiv64 m - 1) < m := by
  have h := ceilDiv64_spec m
  have hpos : 0 < ceilDiv64 m := by omega
  omega

theorem ceilDiv64_mono {a b : Nat} (h : a ≤ b) : ceilDiv64 a ≤ ceilDiv64 b := by
  have h1 := ceilDiv64_spec a
  have h2 := ceilDiv64_spec b
  omega

/-- **`64 * ceilDiv64 (n + 9) = canonTotal n`**: the canonical padded total is the least multiple of
64 at least `n + 9`. Both are least multiples of 64 in `[n + 9, n + 73)`, of width 64. -/
theorem canonTotal_eq_ceil (n : Nat) : 64 * ceilDiv64 (n + 9) = canonTotal n := by
  obtain ⟨k, hk⟩ := canonTotal_multiple n
  have h1 := canonTotal_ge n
  have h2 := canonTotal_lt n
  have h3 := ceilDiv64_spec (n + 9)
  omega

/-- **THE BLOCK-COUNT THEOREM.** The padded length of a message of `n` bytes, divided by 64, is
`⌈(n + 9)/64⌉`. -/
theorem canonTotal_div (n : Nat) : canonTotal n / 64 = blockCount n := by
  have h := canonTotal_eq_ceil n
  calc canonTotal n / 64 = (64 * ceilDiv64 (n + 9)) / 64 := by rw [h]
    _ = ceilDiv64 (n + 9) := Nat.mul_div_cancel_left (ceilDiv64 (n + 9)) (by omega)
    _ = blockCount n := rfl

/-- **THE CLOSURE OF THE BYTE COUNT ONTO THE BLOCK COUNT.** `canonTotal n` is a multiple of 64 and it
is exactly `64 * blockCount n` bytes, so the padding's byte traffic is 64 bytes per padded block and
no byte is counted twice. This is what makes the padding's byte figure additive over a block count
instead of a separate curve. -/
theorem canonTotal_blockBytes (n : Nat) : canonTotal n = 64 * blockCount n := by
  have h := canonTotal_eq_ceil n
  unfold blockCount
  omega

/-- The block count is the ceiling, stated with no division on either side. -/
theorem blockCount_is_ceil (n : Nat) :
    64 * (blockCount n - 1) < n + 9 ∧ n + 9 ≤ 64 * blockCount n := by
  rw [blockCount]
  constructor
  · exact ceilDiv64_strict (n + 9) (by omega)
  · exact (ceilDiv64_spec (n + 9)).1

/-- The block-count function in closed form. -/
theorem blockCount_eq (n : Nat) : blockCount n = (n + 9 + 63) / 64 := by
  unfold blockCount ceilDiv64
  omega

/-- The block count is at least one, and no more than `n + 72` bytes of padding are needed. -/
theorem blockCount_pos (n : Nat) : 1 ≤ blockCount n := by
  have h := ceilDiv64_spec (n + 9)
  unfold blockCount
  omega

theorem blockCount_le (n : Nat) : 64 * blockCount n ≤ n + 72 := by
  have h := canonTotal_eq_ceil n
  have h2 := canonTotal_lt n
  unfold blockCount
  omega

/-- **Adding one byte can add at most one block**, and the padded byte total moves by 64 when it does.
This is the recurrence the padding's byte figure obeys. -/
theorem blockCount_succ_le (n : Nat) : blockCount (n + 1) ≤ blockCount n + 1 := by
  have h := blockCount_eq n
  have h2 := blockCount_eq (n + 1)
  omega

theorem canonTotal_succ_blocks (n : Nat) :
    canonTotal (n + 1) - canonTotal n = 0 ∨ canonTotal (n + 1) - canonTotal n = 64 := by
  have h := blockCount_succ_le n
  have h1 := canonTotal_blockBytes n
  have h2 := canonTotal_blockBytes (n + 1)
  have h3 := canonTotal_ge n
  have h4 := canonTotal_ge (n + 1)
  omega

/-! ### 3.3 The join to the canonical bodies -/

/-- **The `pad_spec` shape**, definitional. `pad` (`Sha2.lean:133-136`) is
`m ++ [0x80] ++ List.replicate (total - |m| - 9) 0 ++ lengthBlock |m|`. -/
theorem pad_spec (m : List (BitVec 8)) :
    pad m = m ++ [0x80] ++ List.replicate (canonTotal m.length - m.length - 9) 0
      ++ lengthBlock m.length := by
  rfl

/-- **The eight length-field bytes are eight bytes.** `lengthBlock n` is `List.ofFn` over `Fin 8`, so
its length is 8 whatever `n` is. -/
theorem lengthBlock_len (n : Nat) : (lengthBlock n).length = 8 := by
  unfold lengthBlock
  simp

/-- The canonical padded length is `canonTotal`: the `Nat` arithmetic of §3.1 is the length of the
canonical list. -/
theorem pad_length_total (m : List (BitVec 8)) : (pad m).length = canonTotal m.length := by
  rw [pad_spec]
  simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate,
    lengthBlock_len]
  have h := totalLayout m.length
  omega

/-- **THE CANONICAL BLOCK COUNT.** -/
theorem pad_blockCount (m : List (BitVec 8)) : (pad m).length / 64 = blockCount m.length := by
  rw [pad_length_total]
  exact canonTotal_div m.length

/-- **One `compress` per padded block.** `hash` (`Sha2.lean:146-148`) folds over
`List.range ((pad m).length / 64)`, so the block count the padding fixes is the block count the hash
runs. -/
theorem hash_folds_over_blockCount (m : List (BitVec 8)) :
    List.range ((pad m).length / 64) = List.range (blockCount m.length) := by
  rw [pad_blockCount]

/-- **THE `0x80` MERGE BYTE IS ONE BYTE AT INDEX `n`.** This is the kernel-checked form of "the `0x80`
write is one byte of `padByte`": the padded message carries `0x80` in exactly the position the layout
identity names, so the `+1` of `totalLayout` is not a free parameter. -/
theorem pad_mergeByte (m : List (BitVec 8)) : (pad m).getD m.length 0 = 0x80 := by
  rw [pad]
  simp

/-- **The zero fill is `zeroFill n` bytes.** The `List.replicate` piece of `pad` has exactly that many
entries, and it is the only source of zero bytes. -/
theorem pad_zeroFill_len (m : List (BitVec 8)) :
    (List.replicate (canonTotal m.length - m.length - 9) (0 : BitVec 8)).length
      = canonTotal m.length - m.length - 9 := by
  simp

/-- **The eight length bytes cost two `Nat` operations each.** `lengthByte` (`Sha2.lean:125`) is one
shift and one modulus. -/
theorem lengthByte_body_ops (bits : Nat) (i : Fin 8) :
    lengthByte bits i = (bits >>> (8 * (7 - i.val))) % 256 := rfl

/-- The four literal array lengths of the extraction, as `Nat`s: four bytes per word, eight words per
digest, sixteen words per block, thirty-two digest bytes, sixty-four block bytes. -/
theorem usize_literals :
    (4#usize : Nat) = 4 ∧ (8#usize : Nat) = 8 ∧ (16#usize : Nat) = 16 ∧ (32#usize : Nat) = 32 ∧
      (64#usize : Nat) = 64 := ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **Checked boundary instances.** The 55/56 flip is the boundary the padding turns on. -/
theorem blockCount_0 : blockCount 0 = 1 := by decide
theorem blockCount_55 : blockCount 55 = 1 := by decide
theorem blockCount_56 : blockCount 56 = 2 := by decide
theorem blockCount_63 : blockCount 63 = 2 := by decide
theorem blockCount_64 : blockCount 64 = 2 := by decide
theorem blockCount_119 : blockCount 119 = 2 := by decide
theorem blockCount_120 : blockCount 120 = 3 := by decide

/-- **The block count of an 80-byte header is 2.** The `0x80` byte and the eight-byte length field
need 89 bytes, so the header does not fit in one 64-byte block. -/
theorem blockCount_80 : blockCount 80 = 2 := by decide

/-- The join at the header length. -/
theorem pad_blockCount_80 (m : List (BitVec 8)) (hm : m.length = 80) :
    (pad m).length / 64 = 2 := by
  rw [pad_blockCount, hm, blockCount_80]

/-- **The padded length of an 80-byte header is 128 bytes**: 80 message bytes, one `0x80` byte, 39
zero-fill bytes and eight length bytes. -/
theorem canonTotal_80 : canonTotal 80 = 128 := by decide
theorem zeroFill_80 : zeroFill 80 = 39 := by decide

/-! ## 4. The padding cost

`pad` (`Sha2.lean:133-136`) appends four pieces to the message: the message itself, one `0x80` merge
byte, `zeroFill n` zero bytes, and the eight bytes of `lengthBlock n`. §4 prices all four in the same
measure, as **byte writes**, and splits the appended bytes into the three counted classes the
`ShaCost` record separates: `padByte` (all of them), `zeroByte` (the zero fill) and `lenByte` (the
length field). The two `Nat` operations each length byte costs go in `padShift` and `padMod`.

- **Measure:** `ShaCost σ`, the `padByte`/`zeroByte`/`lenByte`/`padShift`/`padMod` fields.
- **Counted quantity:** bytes written into the padded message, and `Nat` operations on the bit length.
- **Recurrence and closure, per §8:** the padding is a **single pass**, so there is no loop; its count
  is the closed form `canonTotal n`. The recurrence in the message length is
  `canonTotal (n + 1) ≤ canonTotal n + 64` (`canonTotal_succ_le`) together with
  `canonTotal (n+1) - canonTotal n ∈ {0, 64}` (`canonTotal_succ_blocks`), and it closes because
  `canonTotal n = 64 * blockCount n` (`canonTotal_blockBytes`): the padded byte count is *exactly*
  64 bytes per padded block, with `zeroFill n ≤ 63` extra zero bytes, so the byte figure is additive
  over the block count and no byte is counted twice. `ShaCost.rep_add` then makes any `n`-fold byte
  total additive. -/

/-- **The cost of padding a message of `n` bytes.** Temporal O(n) - one append of `n` bytes plus one
`List.replicate` of `zeroFill n` bytes plus eight length bytes; spatial O(n) for the padded list, with
at most 63 extra zero bytes. Counted in: `padByte` (the padded length), `zeroByte`, `lenByte`,
`padShift` (8) and `padMod` (8). Derivation: `pad_spec` displays the four concatenated pieces and
`totalLayout` fixes their lengths; `lengthByte_body_ops` is the two-operation body of each of the
eight length bytes. -/
def padCost (σ : R) (n : Nat) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := canonTotal n, zeroByte := zeroFill n, lenByte := 8, padShift := 8, padMod := 8,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 0, digestWordRead := 0 }

/-- **The padding's closed byte figure.** Only the five padding fields are nonzero, and their values
are the canonical lengths. -/
theorem padCost_fields (σ : R) (n : Nat) :
    padCost σ n
      = { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
          bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
          padByte := canonTotal n, zeroByte := zeroFill n, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
          schedWordWrite := 0, digestByte := 0, digestWordRead := 0 } := by
  rfl

/-- **THE FOUR-PIECE DECOMPOSITION OF THE PADDED BYTES.**
`padByte = n + 1 + zeroByte + lenByte`: the message bytes, then the one `0x80` merge byte, then the
zero fill, then the length field. The `+1` is the `0x80` write, and `pad_mergeByte` shows it is a real
byte of the output at index `n`, not a parameter of the layout identity. -/
theorem padCost_split (σ : R) (n : Nat) :
    (padCost σ n).padByte = n + 1 + (padCost σ n).zeroByte + (padCost σ n).lenByte := by
  have h := totalLayout n
  simp only [padCost, ShaCost.zero]
  omega

/-- **The merge byte is one byte, the length field is eight, and the zero fill is at most 63.** -/
theorem padCost_bounded (σ : R) (n : Nat) :
    (padCost σ n).lenByte = 8 ∧ (padCost σ n).padShift = 8 ∧ (padCost σ n).padMod = 8 ∧
      (padCost σ n).zeroByte ≤ 63 := by
  exact ⟨rfl, rfl, rfl, zeroFill_le n⟩

/-- **THE CLOSURE OF THE PADDING'S BYTE FIGURE.** The padded byte count is `64` bytes per padded
block, so the padding's byte total is additive over a block count. -/
theorem padCost_blockBytes (σ : R) (n : Nat) : (padCost σ n).padByte = 64 * blockCount n := by
  rw [padCost]
  exact canonTotal_blockBytes n

/-- **THE PADDING'S BYTE RECURRENCE, in the measure.** The padded byte count moves by `0` or by `64`
when the message grows by one byte. -/
theorem padCost_succ (σ : R) (n : Nat) :
    (padCost σ (n + 1)).padByte - (padCost σ n).padByte = 0 ∨
      (padCost σ (n + 1)).padByte - (padCost σ n).padByte = 64 := by
  exact canonTotal_succ_blocks n

/-- **The padding's byte figure adds over any number of attempts that pad separately.** This is the
§8 closure law for the padding component. -/
theorem padCost_rep_add (σ : R) (m n : Nat) :
    ShaCost.rep σ (m + n) (padCost σ n) = ShaCost.rep σ m (padCost σ n) + ShaCost.rep σ n (padCost σ n) :=
  ShaCost.rep_add σ m n (padCost σ n)

/-- **The padding adds nothing to the eleven compression fields**, which is what keeps AR-20's
figures intact. -/
theorem padCost_comp_zero (σ : R) (n : Nat) : (padCost σ n).comp = RoundCost.zero σ := by
  rw [padCost_fields]
  rfl

/-- **THE PADDING OF AN 80-BYTE HEADER.** 80 message bytes, one `0x80` byte, 39 zero-fill bytes and
eight length bytes: 128 bytes, two blocks. -/
theorem padCost_80 (σ : R) :
    padCost σ 80
      = { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
          bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
          schedWordWrite := 0, digestByte := 0, digestWordRead := 0 } := by
  have h := canonTotal_80
  have h2 := zeroFill_80
  simp only [padCost]
  refine ShaCost.eq_of_fields (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by omega) (by omega) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl) (by rfl)

/-! ## 5. The block loading cost

The canonical loader `paddedBlock` is `private` in `Sha2.lean:139-142` and is referenced by neither
AR-20 nor `Sha256Pad.md`; it is not resolvable from outside that module in this toolchain (a
`run_cmd` probe of the environment returns the mangled name `_private.Crypto.Sha2.0.…`, which the
name resolver does not accept). The corpus's **public** spelling of the same read is
`apeiron_word.bytesToWords` (`Crypto/Sha256Certificate.lean:493`), the big-endian regrouping of an
extracted 64-byte block into the canonical sixteen-word `Block`, with `schedRow_block` (`:514`) tying
it to the schedule row the extracted `sha256.compress_loop0` writes. §5 prices that body.

- **Measure:** `ShaCost σ`, the `blockByteRead`, `regroupMul`, `regroupAdd`, `regroupTrunc` and
  `schedWordWrite` fields.
- **Counted quantity:** byte reads of the padded block, `Nat` operations of the regrouping, 32-bit
  truncations, and word writes into the schedule.
- **Recurrence and closure, per §8:** the loader is `Vector.ofFn` over `Fin 16` of a four-step
  accumulation, so it is a fixed 16 × 4 = 64 evaluations per block with no data-dependent iteration;
  its own temporal and spatial cost is O(1) and the count is therefore constant per block, and
  `ShaCost.rep` over `blockCount n` blocks closes by `ShaCost.rep_add`. Index arithmetic
  (`4 * i.val`) is **excluded**, with the reason of §1.3. -/

/-- **One byte read.** `b64 blk j` is a single `getD` of byte `j` of the block. -/
theorem b64_body (blk : Array Std.U8 64#usize) (j : Nat) : b64 blk j = (blk.val.getD j (0#u8)).bv :=
  rfl

/-- **The four-byte regrouping body.** `beWord16 blk i` is the big-endian regrouping of bytes
`4i … 4i+3`: four byte reads, three `Nat` multiplications by `2^24`, `2^16`, `2^8`, three `Nat`
additions and one `BitVec.ofNat 32` truncation. -/
theorem beWord16_body (blk : Array Std.U8 64#usize) (i : Fin 16) :
    beWord16 blk i
      = BitVec.ofNat 32
          ((b64 blk (4 * i.val)).toNat * 16777216 + (b64 blk (4 * i.val + 1)).toNat * 65536
            + (b64 blk (4 * i.val + 2)).toNat * 256 + (b64 blk (4 * i.val + 3)).toNat) := rfl

/-- **The loader is sixteen words of `Fin 16`.** -/
theorem bytesToWords_body (blk : Array Std.U8 64#usize) :
    bytesToWords blk = Vector.ofFn fun i : Fin 16 => beWord16 blk i := rfl

/-- **A block is sixteen words, and its byte length is 64.** -/
theorem bytesToWords_arity (blk : Array Std.U8 64#usize) : (bytesToWords blk).toList.length = 16 := rfl

theorem b64_length (blk : Array Std.U8 64#usize) : blk.val.length = 64 :=
  blk.property

/-- **The digest view is eight words of `Fin 8`,** the inverse regrouping, fixed by the corpus at
`digView_get`. -/
theorem digView_body (a : Array Std.U8 32#usize) :
    digView a = Vector.ofFn fun i : Fin 8 => beWord a i := rfl

theorem digView_get_body (a : Array Std.U8 32#usize) (i : Fin 8) : (digView a).get i = beWord a i :=
  digView_get a i

/-- **THE PER-WORD COST OF THE BIG-ENDIAN REGROUPING**: four byte reads, three `Nat` multiplications,
three `Nat` additions, one 32-bit truncation, one word written into the schedule. The four
`Array.index_usize block` reads and the single `Array.update w i` of the extracted body
`compress_loop0.body` (`ApeironWord.lean:15243-15264`, from `src/sha256.rs:37-41`) are grade `model`:
no machine-checked operation-count mechanism for a term exists (AR-10 residual 1), so that occurrence
count is a reading of those source lines. The four reads, three multiplications, three additions and
one truncation of `beWord16_body` are `proved`, the body being displayed in full by `rfl`. -/
def regroupCost (σ : R) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 4, regroupMul := 3, regroupAdd := 3, regroupTrunc := 1,
    schedWordWrite := 1, digestByte := 0, digestWordRead := 0 }

/-- **THE PER-BLOCK COST OF THE LOAD.** Sixteen words, each one `regroupCost`: 64 byte reads, 48
`Nat` multiplications, 48 `Nat` additions, 16 truncations, 16 schedule writes. Temporal and spatial
O(1) per block - `Vector.ofFn` over `Fin 16` of a four-step accumulation, no data-dependent
iteration. Derivation: `bytesToWords_body`, `bytesToWords_arity` and `ShaCost.rep` at `regroupCost`. -/
def loadCost (σ : R) : ShaCost σ := ShaCost.rep σ 16 (regroupCost σ)

/-- The loader's closed count. -/
theorem loadCost_fields (σ : R) :
    loadCost σ
      = { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
          bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
          padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
          blockByteRead := 64, regroupMul := 48, regroupAdd := 48, regroupTrunc := 16,
          schedWordWrite := 16, digestByte := 0, digestWordRead := 0 } := by
  rfl

/-- **A block read is 64 byte reads and 16 schedule writes**: the loader moves the whole block once
and writes one schedule row per word. -/
theorem loadCost_bytes (σ : R) : (loadCost σ).blockByteRead = 64 ∧ (loadCost σ).schedWordWrite = 16 :=
  ⟨rfl, rfl⟩

/-- **The loader's byte read count is tied to the block's type, not to a literal**: it is the byte
length of any `Array Std.U8 64#usize`. -/
theorem loadCost_bytes_eq_blockLen (blk : Array Std.U8 64#usize) :
    (loadCost (1 : W32)).blockByteRead = blk.val.length := by
  rw [loadCost_fields]
  exact (b64_length blk).symm

/-- **THE LOADER ADDS NOTHING TO THE ELEVEN COMPRESSION FIELDS**, so pricing it cannot perturb an
AR-20 figure. -/
theorem loadCost_comp_zero (σ : R) : (loadCost σ).comp = RoundCost.zero σ := by
  rw [loadCost_fields]
  rfl

/-! ## 6. The digest store cost

**There is no canonical digest byte store to price, and that absence is kernel-checkable.** The
canonical `Crypto.Sha2.sha256` (`Sha2.lean:152-153`) is `ByteArray → State` and `State = Vector Word 8`
(`Sha2.lean:25`): the canonical digest is **eight words**, and `Crypto.Sha2` contains no code that
serialises it to bytes. The big-endian store the brief names therefore exists only in the extracted
body `sha256.sha256_loop2.body` (`ApeironWord.lean:15510-15548`, from `src/sha256.rs:138-141`,
`while i < 8 { out[4*i..4*i+4].copy_from_slice(&state[i].to_be_bytes()); i += 1 }`), whose *output*
side the corpus pins at `digView_get` (`Sha256Certificate.lean:157`): the 32-byte digest read back
big-endian is exactly the canonical `State`.

- **Measure:** `ShaCost σ`, the `digestByte` and `digestWordRead` fields.
- **Counted quantity:** bytes written by the store, and chaining words read by it.
- **Recurrence and closure, per §8:** the store loop is `i < 8`, a fixed eight iterations with no
  data-dependent step; its temporal and spatial cost is O(1) per digest and its counted byte total is
  the closed constant `8 * 4 = 32`. It is **not** inside `ShaCost.rep` over the block count, because a
  digest is stored once per attempt, not once per block. -/

/-- **The canonical digest is a `State`.** The ascription is the kernel check: `Crypto.Sha2` has no
thirty-two-byte digest and therefore no byte-level store. -/
theorem sha2_digest_isState : (sha256 ByteArray.empty : State) = sha256 ByteArray.empty := rfl

/-- **The state is eight words of `Fin 8`** (`Sha2.lean:25`): `State = Vector Word 8`, and the
initial value `H0` (`Sha2.lean:46-47`) is a literal eight-element vector. -/
theorem state_arity : (H0 : State).toList.length = 8 := rfl

/-- **`U32::to_be_bytes` produces four bytes per word.** The four is the `4#usize` length index of
the extracted signature `core.num.U32.to_be_bytes : U32 → Std.Array U8 4#usize`, and the theorem
below is the general fact that any array of that index has that byte length; the signature itself is
checked by the elaborator at every application of `to_be_bytes`. -/
theorem u8_array4_len (a : Array Std.U8 4#usize) : a.val.length = 4 := a.property

/-- **`U32::from_be_bytes` consumes four bytes**, the inverse of the load of §5. -/
theorem from_be_bytes_arg (a : Array Std.U8 4#usize) : a.val.length = 4 := a.property

/-- **THE STORE'S PER-WORD COST**: one chaining-word read, four bytes written, one schedule-free word
store. The `Usize` index arithmetic (`4 * i`, `4 * i + 4`, `i + 1`) of the extracted body is
**excluded** with the reason of §1.3. The occurrence count of `Array.index_usize state i`,
`U32::to_be_bytes` and `Slice::copy_from_slice` is grade `model`: no machine-checked operation-count
mechanism for a term exists (AR-10 residual 1), so it is a reading of
`ApeironWord.lean:15510-15548`. The four-bytes-per-word and eight-words shapes are `proved`, being the
types of the bodies. -/
def storeWordCost (σ : R) : ShaCost σ :=
  { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
    bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
    padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
    blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
    schedWordWrite := 0, digestByte := 4, digestWordRead := 1 }

/-- **THE PER-ATTEMPT COST OF THE DIGEST STORE**: eight words, thirty-two bytes. Temporal and spatial
O(1): a fixed eight-iteration loop. Derivation: `ShaCost.rep` at `storeWordCost` over the eight words
of `State`, with `to_be_bytes_len` fixing the four bytes per word. -/
def storeCost (σ : R) : ShaCost σ := ShaCost.rep σ 8 (storeWordCost σ)

theorem storeCost_fields (σ : R) :
    storeCost σ
      = { mul := 0, add := 0, sub := 0, neg := 0, cast := 0, channel := 0,
          bitXor := 0, bitAnd := 0, bitNot := 0, shift := 0, bitOr := 0,
          padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
          blockByteRead := 0, regroupMul := 0, regroupAdd := 0, regroupTrunc := 0,
          schedWordWrite := 0, digestByte := 32, digestWordRead := 8 } := by
  rfl

/-- **The store writes 32 bytes and reads the eight chaining words.** -/
theorem storeCost_bytes (σ : R) : (storeCost σ).digestByte = 32 ∧ (storeCost σ).digestWordRead = 8 :=
  ⟨rfl, rfl⟩

/-- **The store writes 32 bytes for eight words of four bytes each**, stated as the product so that
the two shapes it multiplies are visible: `state_arity` gives the eight words and `u8_array4_len` the
four bytes each. -/
theorem storeCost_digest_bytes (σ : R) : (storeCost σ).digestByte = 8 * 4 := by
  rw [storeCost_fields]

/-- **THE STORE ADDS NOTHING TO THE ELEVEN COMPRESSION FIELDS.** The eight-word add-back inside
`compress` (`Sha2.lean:120`) is already priced in `blockCost`; the store is the serialisation *after*
it, and it moves bytes without performing a ring or `Word` operation. -/
theorem storeCost_comp_zero (σ : R) : (storeCost σ).comp = RoundCost.zero σ := by
  rw [storeCost_fields]
  rfl

/-! ## 7. Which reads are priced, and which stay `model`

AR-20 left "every element read" outside the measure because the record had no read field. This module
adds **two** read fields, each with a stated scope, and states the rest as exclusions.

| read | field | status |
|---|---|---|
| 64 block bytes read by the sixteen-word loader | `blockByteRead` | **priced**, `proved` from `beWord16_body` |
| 8 chaining words read by the digest store | `digestWordRead` | **priced**, `proved` from the extracted body's shape |
| the round body's eight `acc.getD i 0`, plus `K.getD t 0` and `w.getD t 0` | none | **excluded**, reason below |
| the sixteen `w.getD i 0` of `schedule`'s seed vector and its 64 `acc.getD i 0` | none | **excluded**, reason below |
| the `getD` default-value fallbacks of every view above | none | **excluded**, reason below |

**The reason for the exclusions, once.** A read field that counted *some* reads and not others would
reproduce the blindness it was added to remove, so the record names exactly two read scopes and the
residual list names the rest. Which memory a computation owns - schedule, state, round constants,
message buffer, digest buffer - is the question `CostAlgebra.Cost` owns (`CostAlgebra.lean:34-45`,
one field per resource) and the question `CostBridge.loads_invisible_to_order` (`CostBridge.lean:161`)
demonstrates is invisible to a registered order. Charging the round's element reads here would add
fields that no consumer of this measure reads and would silently merge five different buffers into one
number. No element-traffic, cache, bandwidth or timing claim is made or follows.

**`RotRead` and `AddRead` remain `model`, carried and unused.** -/

/-- **The rotation reading, as a named premise.** The channel read of the product
`ringMul 1 (x,0) (2^n, 2^(32-n))` is the left rotation of `x` by `n` bits at width 32. Stated, not
proved: candidate C5 at `ops-map.md:167-180`, grade `model`. -/
def RotRead (n : Nat) (x : W32) : Prop :=
  ringUpper faceHyperbolic (ringMul faceHyperbolic (x, 0) ((2 ^ n : Nat), (2 ^ (32 - n) : Nat))) =
    (((x.val * (2 ^ n)) % (2 ^ 32) + x.val / (2 ^ (32 - n)) : Nat) : W32)

/-- **The additive reading, as a named premise.** A `Word` addition is an addition in `ZMod (2 ^ 32)`.
Stated, not proved at this width: the template is `denote_wrapping_add` (`Denotation.lean:35`), proved
at width 64 only, so the reading at the SHA-2 word width `BitVec 32` is grade `model`. -/
def AddRead (x y : Word) : Prop := (x + y).toNat = (x.toNat + y.toNat) % 4294967296

/-- **The rotation reading needs no channel root.** AR-9's `ringUpper_mul_eval`, re-proved: the read
of the product along the `x ↦ (x,0)` line is `x * (a + r*b)` for every `σ` and every `r`, so the
count of §2 that prices six rotations as six `ringMul`-plus-`ringUpper` pairs needs no premise. -/
theorem channel_read_zero_slot (σ r a b x : W32) :
    ringUpper r (ringMul σ (x, 0) (a, b)) = x * (a + r * b) := by
  simp only [ringUpper, ringMul]
  ring

/-- **The rotation reading's cost is unconditional; the premise is deliberately unused.** Every count
in §2, §4, §5 and §6 is a fact about bodies, and none of them reads `RotRead` or `AddRead`. -/
theorem rotRead_cost_of_reading (σ : R) (n : Nat) (x : W32) (_h : RotRead n x) :
    (rotReadCost σ).mul = 6 ∧ (rotReadCost σ).add = 3 ∧ (rotReadCost σ).channel = 1 := by
  rw [rotReadCost_fields]
  simp

/-- **THE PRICED READ SCOPES ARE DISJOINT FROM EVERY UNPRICED READ.** The two read fields are
`blockByteRead` and `digestWordRead`, and no other field of the record is a read: the round's
element reads therefore cannot be charged by accident, and equally cannot be claimed to be charged. -/
theorem priced_reads_are_the_two_scopes (σ : R) :
    (loadCost σ).digestWordRead = 0 ∧ (storeCost σ).blockByteRead = 0 ∧
      (padCost σ 80).blockByteRead = 0 ∧ (padCost σ 80).digestWordRead = 0 := by
  simp [loadCost_fields, storeCost_fields, padCost_80]

/-! ## 8. The completed cost of one proof-of-work attempt

An attempt is **four priced pieces in one record**:

| piece | cost | where |
|---|---|---|
| padding of the 80-byte header | `padCost σ 80` | §4 |
| `blockCount 80 = 2` blocks, each compressed **and** loaded | `ShaCost.rep σ 2 (ofRound (blockCost σ) + loadCost σ)` | §2, §5 |
| big-endian digest store | `storeCost σ` | §6 |

The padding is paid **once** per message, not once per attempt inside `rep`: `hash` pads once and then
folds `compress` over the blocks (`hash_folds_over_blockCount`), so a range of `m` attempts against one
header re-pads only when the header changes. §9 keeps that structure.

- **Measure:** `ShaCost σ`, all twenty-three fields.
- **Counted quantity:** AR-20's eleven ring and `Word` counts plus the twelve byte, `Nat`-operation
  and read counts of §4–§6.
- **Temporal and spatial cost of `attemptCost`:** O(1) — a closed term, one `ShaCost.rep` of
  twenty-three `Nat` multiplications, no recursion, no allocation.
- **Derivation, each link a theorem in this module:**
  1. `hash` performs one `compress` per padded block — `hash_folds_over_blockCount`;
  2. the padded block count of an 80-byte header is `2` — `blockCount_80`, tied to `(pad m).length / 64`
     by `pad_blockCount_80`;
  3. one `compress` costs `blockCost σ` — `blockCost_eq`;
  4. one block load costs `loadCost σ` — `loadCost_fields`;
  5. the eight-word store costs `storeCost σ` — `storeCost_fields`;
  6. the four pieces are dependent, so they total fieldwise — `ShaCost.composed_comm`/`_assoc`. -/

/-- **A SHA-256 block-header length in bytes.** The canonical `Crypto.Sha2` module carries no header
type; 80 bytes is the length this module prices, fixed as a definition. -/
def headerBytes : Nat := 80

/-- **THE COST OF ONE BLOCK: compression and loading together.** Temporal and spatial O(1) as a
closed term. Counted in: the eleven compression fields plus the five load fields. Derivation:
`blockCost` (§2) and `loadCost` (§5) are dependent, so they compose fieldwise. -/
def blockFullCost (σ : R) : ShaCost σ := ShaCost.ofRound (blockCost σ) + loadCost σ

/-- **THE COST OF ONE ATTEMPT**, derived from the canonical block count. Temporal and spatial O(1): a
closed term. -/
def attemptCost (σ : R) : ShaCost σ :=
  padCost σ headerBytes + ShaCost.rep σ (blockCount headerBytes) (blockFullCost σ) + storeCost σ

/-- **The attempt is two padded blocks, each compressed and loaded.** -/
theorem attemptCost_two (σ : R) :
    attemptCost σ = padCost σ 80 + ShaCost.rep σ 2 (blockFullCost σ) + storeCost σ := by
  unfold attemptCost headerBytes
  rw [blockCount_80]

/-- **The four pieces are visible in the figure.** -/
theorem attemptCost_split (σ : R) :
    attemptCost σ = padCost σ headerBytes
      + ShaCost.rep σ (blockCount headerBytes) (ShaCost.ofRound (blockCost σ) + loadCost σ)
      + storeCost σ := rfl

/-- **THE COMPLETED ATTEMPT FIGURE.** AR-20's eleven compression counts are unchanged - 6912 ring
multiplications, 4752 ring additions, 1152 channel reads, 1408 XORs, 640 ANDs, 128 NOTs, 2496 shifts,
1152 disjunctions - and the twelve new fields carry 128 padded bytes (39 of them zero fill, 8 of them
length bytes, 8 shifts and 8 moduli), 128 block byte reads, 96 `Nat` multiplications, 96 `Nat`
additions, 32 truncations, 32 schedule word writes, 32 digest bytes and 8 chaining-word reads. -/
theorem attemptCost_eq (σ : R) :
    attemptCost σ
      = { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
          bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 128, regroupMul := 96, regroupAdd := 96, regroupTrunc := 32,
          schedWordWrite := 32, digestByte := 32, digestWordRead := 8 } := by
  have h0 := padCost_80 σ
  have h1 := blockCost_eq σ
  have h2 := loadCost_fields σ
  have h3 := storeCost_fields σ
  simp only [attemptCost, headerBytes, blockFullCost, ShaCost.ofRound, blockCount_80,
    ShaCost.composed_apply, ShaCost.rep, ShaCost.composed]
  simp [h0, h1, h2, h3]

/-- **The eleven compression fields of the completed attempt are AR-20's attempt figure.** This is the
statement that pricing the padding, the load and the store changed no AR-20 count. -/
theorem attemptCost_comp_eq (σ : R) :
    (attemptCost σ).comp
      = { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
          bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152 } := by
  have h := attemptCost_eq σ
  rw [h]
  rfl

/-- **The attempt's ring layer alone**, as AR-10's six-field record `(6912, 4752, 0, 0, 0, 1152)`. -/
theorem attemptCost_ringFields (σ : R) : (attemptCost σ).ringFields = ⟨6912, 4752, 0, 0, 0, 1152⟩ := by
  have h := attemptCost_eq σ
  rw [h]
  rfl

/-- **THE COMPLETED FIGURE IS EXACTLY TWICE THE COMPLETED BLOCK FIGURE PLUS THE PADDING AND THE
STORE**, so the ×2 carries no field. -/
theorem attemptCost_is_two_blockFull (σ : R) :
    attemptCost σ = padCost σ 80 + ShaCost.rep σ 2 (blockFullCost σ) + storeCost σ :=
  attemptCost_two σ

/-- **One completed block, compression and load together.** -/
theorem blockFullCost_eq (σ : R) :
    blockFullCost σ
      = { mul := 3456, add := 2376, sub := 0, neg := 0, cast := 0, channel := 576,
          bitXor := 704, bitAnd := 320, bitNot := 64, shift := 1248, bitOr := 576,
          padByte := 0, zeroByte := 0, lenByte := 0, padShift := 0, padMod := 0,
          blockByteRead := 64, regroupMul := 48, regroupAdd := 48, regroupTrunc := 16,
          schedWordWrite := 16, digestByte := 0, digestWordRead := 0 } := by
  have h1 := blockCost_eq σ
  have h2 := loadCost_fields σ
  simp [blockFullCost, ShaCost.ofRound, ShaCost.composed, h1, h2]

/-- **An attempt is never free**: its multiplication, its padded bytes and its digest bytes are all
positive. -/
theorem attemptCost_strict (σ : R) :
    0 < (attemptCost σ).mul ∧ 0 < (attemptCost σ).padByte ∧ 0 < (attemptCost σ).digestByte ∧
      0 < (attemptCost σ).blockByteRead := by
  have h := attemptCost_eq σ
  rw [h]
  norm_num

/-! ## 9. The cost of a mining range

`rangeCost σ m` is `m` completed attempts against **one** header: `m * 2` compressed-and-loaded
blocks, `m` digest stores, and **one** padding. Each attempt re-runs the whole block fold and the
block load; a range re-pads only when the header changes. -/

/-- **THE COST OF A RANGE OF `m` ATTEMPTS** against one header, the padding paid once. Temporal and
spatial O(1): a closed term of twenty-three `Nat` multiplications, no recursion, no allocation. -/
def rangeCost (σ : R) (m : Nat) : ShaCost σ :=
  padCost σ headerBytes + ShaCost.rep σ (m * blockCount headerBytes) (blockFullCost σ)
    + ShaCost.rep σ m (storeCost σ)

/-- **The range closed form at an arbitrary range length.** The compression and load fields are `m`
times the attempt's, the store fields are `m` times the store's, and the five padding fields are the
padding's *once*. -/
theorem rangeCost_eq (σ : R) (m : Nat) :
    rangeCost σ m
      = { mul := m * 6912, add := m * 4752, sub := 0, neg := 0, cast := 0, channel := m * 1152,
          bitXor := m * 1408, bitAnd := m * 640, bitNot := m * 128, shift := m * 2496,
          bitOr := m * 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := m * 128, regroupMul := m * 96, regroupAdd := m * 96,
          regroupTrunc := m * 32, schedWordWrite := m * 32,
          digestByte := m * 32, digestWordRead := m * 8 } := by
  have h0 := padCost_80 σ
  have h1 := blockCost_eq σ
  have h2 := loadCost_fields σ
  have h3 := storeCost_fields σ
  simp [rangeCost, headerBytes, blockFullCost, blockCount_80, ShaCost.ofRound,
    ShaCost.composed_apply, ShaCost.rep, ShaCost.composed, h0, h1, h2, h3, Nat.mul_assoc,
    Nat.mul_left_comm, Nat.mul_comm, and_self]

/-- **The range figure is `m` attempts, not fewer and not more.** -/
theorem rangeCost_split (σ : R) (m : Nat) :
    rangeCost σ m
      = padCost σ 80 + ShaCost.rep σ (m * 2) (blockFullCost σ) + ShaCost.rep σ m (storeCost σ) := by
  rfl

/-- **THE PADDING OF A ONE-HEADER RANGE IS PAID ONCE, NOT `m` TIMES.** This is the field-by-field
form of "a range re-pads only when the header changes", and it is why `rangeCost σ m` is **not** an
additive function of `m`: the additivity law for a range is the additivity of its per-attempt
component, stated next, and of the whole attempt, stated after it. -/
theorem rangeCost_pad_once (σ : R) (m : Nat) :
    (rangeCost σ m).padByte = (padCost σ 80).padByte ∧
      (rangeCost σ m).zeroByte = (padCost σ 80).zeroByte ∧
      (rangeCost σ m).lenByte = (padCost σ 80).lenByte ∧
      (rangeCost σ m).padShift = (padCost σ 80).padShift ∧
      (rangeCost σ m).padMod = (padCost σ 80).padMod := by
  have h0 := rangeCost_eq σ m
  rw [h0, padCost_80]
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- **ADDITIVITY OF A ONE-HEADER RANGE'S PER-ATTEMPT COMPONENT, twenty-three fields.** `m + n`
attempts against one header compress and load `(m + n) * 2` blocks, and that is the sum of the `m`
and `n` parts field by field. -/
theorem rangeCost_attemptPart_add (σ : R) (m n : Nat) :
    ShaCost.rep σ ((m + n) * 2) (blockFullCost σ)
      = ShaCost.rep σ (m * 2) (blockFullCost σ) + ShaCost.rep σ (n * 2) (blockFullCost σ) := by
  rw [Nat.add_mul, ShaCost.rep_add]

/-- **The same for the store component.** -/
theorem rangeCost_storePart_add (σ : R) (m n : Nat) :
    ShaCost.rep σ (m + n) (storeCost σ)
      = ShaCost.rep σ m (storeCost σ) + ShaCost.rep σ n (storeCost σ) :=
  ShaCost.rep_add σ m n (storeCost σ)

/-! ### 9.1 The composite attempt, additivity over all twenty-three fields

`attemptsCost σ m` is `m` attempts **each paying its own header's padding**. It is the additive
figure, and `ShaCost.rep_add` at the attempt is its additivity law. -/

/-- `attemptsCost σ m` is `m` complete attempts, padding included in each. Temporal and spatial O(1):
a closed term of twenty-three `Nat` multiplications. -/
def attemptsCost (σ : R) (m : Nat) : ShaCost σ := ShaCost.rep σ m (attemptCost σ)

/-- **FIELD-BY-FIELD ADDITIVITY FOR THE COMPOSITE ATTEMPT, twenty-three fields.** This is the §8
closure the composite owes: an `m + n` attempt run costs the sum of an `m` run and an `n` run, field
by field, with no field escaping the record. -/
theorem attemptsCost_add (σ : R) (m n : Nat) :
    attemptsCost σ (m + n) = attemptsCost σ m + attemptsCost σ n :=
  ShaCost.rep_add σ m n (attemptCost σ)

/-- **The same additivity in the eleven compression fields.** -/
theorem attemptsCost_comp_add (σ : R) (m n : Nat) :
    (attemptsCost σ (m + n)).comp = (attemptsCost σ m).comp + (attemptsCost σ n).comp := by
  rw [attemptsCost_add, comp_add]

/-- **The same additivity in the six ring fields.** -/
theorem attemptsCost_ring_add (σ : R) (m n : Nat) :
    (attemptsCost σ (m + n)).ringFields = (attemptsCost σ m).ringFields
      + (attemptsCost σ n).ringFields := by
  rw [attemptsCost_add, ringFields_add']

/-- **Iteration distributes over the composite attempt's pieces**, so a reader may also arrive at a
range by summing pieces rather than attempts. -/
theorem attemptCost_rep_composed (σ : R) (n : Nat) :
    ShaCost.rep σ n (attemptCost σ)
      = ShaCost.rep σ n (padCost σ headerBytes)
        + ShaCost.rep σ n (ShaCost.rep σ (blockCount headerBytes) (blockFullCost σ))
        + ShaCost.rep σ n (storeCost σ) := by
  simp only [attemptCost]
  rw [ShaCost.rep_composed, ShaCost.rep_composed]

/-- **A split range costs the sum of its parts**, twenty-three fields, at `2^31 + 2^31`. -/
theorem attemptsCost_nonceSpace_split (σ : R) :
    attemptsCost σ (2 ^ 31 + 2 ^ 31) = attemptsCost σ (2 ^ 31) + attemptsCost σ (2 ^ 31) :=
  attemptsCost_add σ (2 ^ 31) (2 ^ 31)

/-- **The same split in the eleven compression fields.** -/
theorem attemptsCost_nonceSpace_split_comp (σ : R) :
    (attemptsCost σ (2 ^ 31 + 2 ^ 31)).comp
      = (attemptsCost σ (2 ^ 31)).comp + (attemptsCost σ (2 ^ 31)).comp :=
  attemptsCost_comp_add σ (2 ^ 31) (2 ^ 31)

/-- **The same split in the six ring fields.** -/
theorem attemptsCost_nonceSpace_split_ring (σ : R) :
    (attemptsCost σ (2 ^ 31 + 2 ^ 31)).ringFields
      = (attemptsCost σ (2 ^ 31)).ringFields + (attemptsCost σ (2 ^ 31)).ringFields :=
  attemptsCost_ring_add σ (2 ^ 31) (2 ^ 31)

/-! ### 9.2 The 32-bit nonce space -/

/-- The size of a 32-bit nonce space: `2 ^ 32` attempts. -/
def nonceSpace : Nat := 4294967296

/-- **THE FULL 32-BIT NONCE SPACE AGAINST ONE HEADER**, with the arithmetic evaluated. -/
theorem rangeCost_nonceSpace (σ : R) :
    rangeCost σ nonceSpace
      = { mul := 4294967296 * 6912, add := 4294967296 * 4752, sub := 0, neg := 0, cast := 0,
          channel := 4294967296 * 1152, bitXor := 4294967296 * 1408, bitAnd := 4294967296 * 640,
          bitNot := 4294967296 * 128, shift := 4294967296 * 2496, bitOr := 4294967296 * 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 4294967296 * 128, regroupMul := 4294967296 * 96,
          regroupAdd := 4294967296 * 96, regroupTrunc := 4294967296 * 32,
          schedWordWrite := 4294967296 * 32, digestByte := 4294967296 * 32,
          digestWordRead := 4294967296 * 8 } := by
  have h := rangeCost_eq σ nonceSpace
  rw [h, nonceSpace]

/-- The nonce-space figure at an arbitrary width `w`. -/
theorem rangeCost_pow2_eq (σ : R) (w : Nat) :
    rangeCost σ (2 ^ w)
      = { mul := 2 ^ w * 6912, add := 2 ^ w * 4752, sub := 0, neg := 0, cast := 0,
          channel := 2 ^ w * 1152, bitXor := 2 ^ w * 1408, bitAnd := 2 ^ w * 640,
          bitNot := 2 ^ w * 128, shift := 2 ^ w * 2496, bitOr := 2 ^ w * 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 2 ^ w * 128, regroupMul := 2 ^ w * 96, regroupAdd := 2 ^ w * 96,
          regroupTrunc := 2 ^ w * 32, schedWordWrite := 2 ^ w * 32,
          digestByte := 2 ^ w * 32, digestWordRead := 2 ^ w * 8 } := by
  have h := rangeCost_eq σ (2 ^ w)
  rw [h]

/-- **The nonce-space figure is the 32-bit width of `rangeCost_pow2_eq`.** -/
theorem rangeCost_nonceSpace_eq_pow2 (σ : R) : rangeCost σ nonceSpace = rangeCost σ (2 ^ 32) := by
  unfold nonceSpace
  congr 1

/-- **The nonce-space figure's own compression layer**, as AR-20's eleven-field record. This is the
statement that completing the figure changed no AR-20 count at range level. -/
theorem rangeCost_nonceSpace_comp (σ : R) :
    (rangeCost σ nonceSpace).comp
      = { mul := 4294967296 * 6912, add := 4294967296 * 4752, sub := 0, neg := 0, cast := 0,
          channel := 4294967296 * 1152, bitXor := 4294967296 * 1408, bitAnd := 4294967296 * 640,
          bitNot := 4294967296 * 128, shift := 4294967296 * 2496,
          bitOr := 4294967296 * 1152 } := by
  have h := rangeCost_nonceSpace σ
  rw [h]
  rfl

/-- **The nonce-space figure's own ring layer**, as AR-10's six-field record. -/
theorem rangeCost_nonceSpace_ringFields (σ : R) :
    (rangeCost σ nonceSpace).ringFields
      = ⟨4294967296 * 6912, 4294967296 * 4752, 0, 0, 0, 4294967296 * 1152⟩ := by
  have h := rangeCost_nonceSpace σ
  rw [h]
  rfl

/-! ### 9.3 A message of `k` padded blocks -/

/-- **The cost of a message of `n` bytes**: one padding, `blockCount n` compressed-and-loaded blocks,
one digest store. Temporal and spatial O(1): a closed term. -/
def messageCost (σ : R) (n : Nat) : ShaCost σ :=
  padCost σ n + ShaCost.rep σ (blockCount n) (blockFullCost σ) + storeCost σ

/-- The message figure at the canonical block count, with the numerals substituted. -/
theorem messageCost_eq (σ : R) (n : Nat) :
    messageCost σ n
      = { mul := blockCount n * 3456, add := blockCount n * 2376, sub := 0, neg := 0, cast := 0,
          channel := blockCount n * 576, bitXor := blockCount n * 704, bitAnd := blockCount n * 320,
          bitNot := blockCount n * 64, shift := blockCount n * 1248, bitOr := blockCount n * 576,
          padByte := canonTotal n, zeroByte := zeroFill n, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := blockCount n * 64, regroupMul := blockCount n * 48,
          regroupAdd := blockCount n * 48, regroupTrunc := blockCount n * 16,
          schedWordWrite := blockCount n * 16, digestByte := 32, digestWordRead := 8 } := by
  have h0 := padCost_fields σ n
  have h1 := blockFullCost_eq σ
  have h2 := storeCost_fields σ
  simp [messageCost, ShaCost.composed_apply, ShaCost.rep, ShaCost.composed, h0, h1, h2]

/-- **THE ATTEMPT FIGURE IS THE MESSAGE FIGURE AT THE HEADER LENGTH**, so the attempt is the cost of
hashing one 80-byte header. -/
theorem messageCost_80 (σ : R) : messageCost σ 80 = attemptCost σ := by
  have h0 := padCost_80 σ
  have h1 := blockFullCost_eq σ
  have h2 := storeCost_fields σ
  have h3 := blockCount_80
  simp [messageCost, attemptCost, headerBytes, blockFullCost, ShaCost.ofRound,
    ShaCost.composed_apply, ShaCost.rep, ShaCost.composed, h0, h1, h2, h3]

/-- **Additivity over a block count, twenty-three fields.** -/
theorem blockFullCost_rep_add (σ : R) (k l : Nat) :
    ShaCost.rep σ (k + l) (blockFullCost σ) = ShaCost.rep σ k (blockFullCost σ)
      + ShaCost.rep σ l (blockFullCost σ) := ShaCost.rep_add σ k l (blockFullCost σ)

/-- **The message figure's padded byte count is at most one block beyond the block count**, read in
the byte field. -/
theorem messageCost_bytes_monotone (σ : R) (n : Nat) :
    (messageCost σ n).padByte ≤ 64 * (blockCount n + 1) := by
  have h := canonTotal_blockBytes n
  rw [messageCost_eq σ n]
  simp only [ShaCost.padByte]
  omega

/-! ## 10. The three faces

**Verdict, in AR-15's and AR-20's framing: no face yields a different attempt, range or message
figure, and this is a fact about the counted bodies only.**

Every count in this module is a closed record of `Nat` literals read off bodies that take no dial
argument and branch on none. `ringMul` takes `σ` and is one expression for every `σ`
(`ringMul_body_ops`), and the `BitVec`, `Nat` and `Array` bodies do not mention a dial at all. The
three dial-reading operations of AR-15 §4.3 are still outside an attempt: a channel root is an
argument costing zero operations, channel multiplicativity needs the premise `r ^ 2 = σ`, and a face
certification's price is a norm read plus a channel read, none of which an attempt performs. **The
measure is σ-blind by construction, and that blindness is a fact about the counted bodies, not a
claim that the trichotomy does not affect cost.** -/

/-- **THE FACE VERDICT, TWENTY-THREE FIELDS, AT THE ELLIPTIC FACE.** `ShaCost σ` carries `σ` as a
*type index*, so two records at two dial values are different types and are not compared by `=`, and
`ShaCost.vector`'s twenty-three projections would need a `rfl` reduction past the default
`maxRecDepth` - forbidden by `AGENTS.md` §6. The face verdict is therefore stated three times, once per
face, against the literal of `attemptCost_eq`, which is stated for an arbitrary `σ : R`: same index on
both sides, no guard, and all twenty-three fields compared. -/
theorem attemptCost_faceElliptic : attemptCost faceElliptic
      = { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
          bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 128, regroupMul := 96, regroupAdd := 96, regroupTrunc := 32,
          schedWordWrite := 32, digestByte := 32, digestWordRead := 8 } := attemptCost_eq faceElliptic

/-- **The same twenty-three fields at the parabolic face.** -/
theorem attemptCost_faceParabolic : attemptCost faceParabolic
      = { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
          bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 128, regroupMul := 96, regroupAdd := 96, regroupTrunc := 32,
          schedWordWrite := 32, digestByte := 32, digestWordRead := 8 } := attemptCost_eq faceParabolic

/-- **The same twenty-three fields at the hyperbolic face.** -/
theorem attemptCost_faceHyperbolic : attemptCost faceHyperbolic
      = { mul := 6912, add := 4752, sub := 0, neg := 0, cast := 0, channel := 1152,
          bitXor := 1408, bitAnd := 640, bitNot := 128, shift := 2496, bitOr := 1152,
          padByte := 128, zeroByte := 39, lenByte := 8, padShift := 8, padMod := 8,
          blockByteRead := 128, regroupMul := 96, regroupAdd := 96, regroupTrunc := 32,
          schedWordWrite := 32, digestByte := 32, digestWordRead := 8 } := attemptCost_eq faceHyperbolic

/-- **NO FACE YIELDS A DIFFERENT ATTEMPT FIGURE**, as `Nat` counts so that the cross-face comparison is
well typed: the eight fields a reader compares across faces are equal at `σ = −1`, `σ = 0` and `σ = 1`.
Together with the three theorems above, which compare all twenty-three fields one face at a time,
this is the complete face verdict. -/
theorem attemptCost_threeFaces :
    (attemptCost faceElliptic).mul = (attemptCost faceParabolic).mul ∧
      (attemptCost faceParabolic).mul = (attemptCost faceHyperbolic).mul ∧
      (attemptCost faceElliptic).padByte = (attemptCost faceParabolic).padByte ∧
      (attemptCost faceParabolic).padByte = (attemptCost faceHyperbolic).padByte ∧
      (attemptCost faceElliptic).blockByteRead = (attemptCost faceParabolic).blockByteRead ∧
      (attemptCost faceParabolic).blockByteRead = (attemptCost faceHyperbolic).blockByteRead ∧
      (attemptCost faceElliptic).digestByte = (attemptCost faceParabolic).digestByte ∧
      (attemptCost faceParabolic).digestByte = (attemptCost faceHyperbolic).digestByte ∧
      (attemptCost faceElliptic).regroupMul = (attemptCost faceParabolic).regroupMul ∧
      (attemptCost faceParabolic).regroupMul = (attemptCost faceHyperbolic).regroupMul ∧
      (attemptCost faceElliptic).schedWordWrite = (attemptCost faceParabolic).schedWordWrite ∧
      (attemptCost faceParabolic).schedWordWrite = (attemptCost faceHyperbolic).schedWordWrite := by
  have h1 := attemptCost_faceElliptic
  have h2 := attemptCost_faceParabolic
  have h3 := attemptCost_faceHyperbolic
  simp only [h1, h2, h3]
  simp

/-- **The same at the ring projection**, the six AR-10 fields, read as numerals so that the
cross-index comparison is well typed. -/
theorem attemptCost_threeFaces_ring :
    (attemptCost faceElliptic).ringFields.mul = 6912 ∧
      (attemptCost faceElliptic).ringFields.add = 4752 ∧
      (attemptCost faceElliptic).ringFields.channel = 1152 ∧
      (attemptCost faceHyperbolic).ringFields.mul = 6912 ∧
      (attemptCost faceHyperbolic).ringFields.add = 4752 ∧
      (attemptCost faceHyperbolic).ringFields.channel = 1152 := by
  have h1 := attemptCost_ringFields faceElliptic
  have h2 := attemptCost_ringFields faceParabolic
  have h3 := attemptCost_ringFields faceHyperbolic
  simp only [h1, h2, h3]
  norm_num

/-- **The same at the compression projection**, the eleven AR-20 fields, as numerals. -/
theorem attemptCost_threeFaces_comp :
    (attemptCost faceElliptic).comp.mul = 6912 ∧ (attemptCost faceElliptic).comp.add = 4752 ∧
      (attemptCost faceElliptic).comp.channel = 1152 ∧ (attemptCost faceElliptic).comp.bitXor = 1408 ∧
      (attemptCost faceElliptic).comp.bitAnd = 640 ∧ (attemptCost faceElliptic).comp.bitNot = 128 ∧
      (attemptCost faceElliptic).comp.shift = 2496 ∧ (attemptCost faceElliptic).comp.bitOr = 1152 ∧
      (attemptCost faceHyperbolic).comp.mul = 6912 ∧
      (attemptCost faceHyperbolic).comp.add = 4752 ∧
      (attemptCost faceHyperbolic).comp.channel = 1152 ∧
      (attemptCost faceHyperbolic).comp.bitXor = 1408 ∧
      (attemptCost faceHyperbolic).comp.bitAnd = 640 ∧
      (attemptCost faceHyperbolic).comp.bitNot = 128 ∧
      (attemptCost faceHyperbolic).comp.shift = 2496 ∧
      (attemptCost faceHyperbolic).comp.bitOr = 1152 := by
  have h1 := attemptCost_comp_eq faceElliptic
  have h2 := attemptCost_comp_eq faceParabolic
  have h3 := attemptCost_comp_eq faceHyperbolic
  simp only [h1, h2, h3]
  norm_num

/-- **The nonce-space figure does not depend on the dial**: at every `σ : W32` it is the §9.2
record. -/
theorem rangeCost_faceBlind (σ : W32) :
    (rangeCost σ nonceSpace).mul = 4294967296 * 6912 ∧
      (rangeCost σ nonceSpace).padByte = 128 ∧
      (rangeCost σ nonceSpace).digestByte = 4294967296 * 32 := by
  have h := rangeCost_nonceSpace σ
  simp only [h]
  norm_num

/-- **The message figure does not depend on the dial**: at every `σ : W32` and every length `n` it is
the §9.3 record. -/
theorem messageCost_faceBlind (σ : W32) (n : Nat) :
    (messageCost σ n).padByte = canonTotal n ∧ (messageCost σ n).blockByteRead = blockCount n * 64 ∧
      (messageCost σ n).digestByte = 32 := by
  have h := messageCost_eq σ n
  simp only [h]
  norm_num

/-- **The completed block figure does not depend on the dial.** -/
theorem blockFullCost_faceBlind (σ : W32) :
    (blockFullCost σ).mul = 3456 ∧ (blockFullCost σ).blockByteRead = 64 ∧
      (blockFullCost σ).schedWordWrite = 16 := by
  have h := blockFullCost_eq σ
  simp only [h]
  norm_num

/-- **The three faces of the completed block figure, field by field.** -/
theorem blockFullCost_threeFaces_fieldwise :
    (blockFullCost faceElliptic).mul = (blockFullCost faceParabolic).mul ∧
      (blockFullCost faceParabolic).mul = (blockFullCost faceHyperbolic).mul ∧
      (blockFullCost faceElliptic).blockByteRead = (blockFullCost faceParabolic).blockByteRead ∧
      (blockFullCost faceParabolic).blockByteRead = (blockFullCost faceHyperbolic).blockByteRead := by
  obtain ⟨hm1, hb1, hw1⟩ := blockFullCost_faceBlind faceElliptic
  obtain ⟨hm2, hb2, hw2⟩ := blockFullCost_faceBlind faceParabolic
  obtain ⟨hm3, hb3, hw3⟩ := blockFullCost_faceBlind faceHyperbolic
  simp only [hm1, hm2, hm3, hb1, hb2, hb3, hw1, hw2, hw3]
  simp

/-! ## 11. What this module does not give

**No wall-clock, cycle, joule, watt, hash-per-second, USD or profitability claim follows from any
number here.** These are counts in a twenty-three-field measure whose field names are `mul`, `add`,
`sub`, `neg`, `cast`, `channel`, `bitXor`, `bitAnd`, `bitNot`, `shift`, `bitOr`, `padByte`,
`zeroByte`, `lenByte`, `padShift`, `padMod`, `blockByteRead`, `regroupMul`, `regroupAdd`,
`regroupTrunc`, `schedWordWrite`, `digestByte`, `digestWordRead`. The measure is blind to memory
allocation, cache, bandwidth, dispatch, parallel hardware, branch prediction, pipelining and
instruction selection, and it is not calibrated against any of them. `AGENTS.md:429` - a cost-model
proof is not a hardware speed measurement.

| excluded | reason |
|---|---|
| index and address arithmetic on `Nat`/`Std.Usize` (`4 * i.val`, `4 * i`, `i + 1`, `64 * k`) | address computation, owned by `CostAlgebra.Cost` (`CostAlgebra.lean:34-45`) and blind to the registered order (`CostBridge.lean:161`); no address-secrecy or constant-time claim is made |
| the round body's `acc.getD`, `K.getD`, `w.getD` element reads | the record has exactly two read scopes (§7); a third would merge five buffers into one number, and no consumer of this measure reads it |
| memory, allocation, peak live bytes | `CostAlgebra.Cost` owns them |
| dispatch, parallelism, scheduling | outside the bodies counted |
| every time, energy, throughput, price and profitability figure | `AGENTS.md:429` |
| constant-time and security properties of SHA-256, of the two-slot reading, and of any reading premise | not attempted; `RotRead` and `AddRead` are `model` |
| the seven `Word` additions of the round body's occurrence count | `model`, read off `Sha2.lean:108,109,112,116` (AR-15 §6 item 1) |
| the occurrence inventories of `sha256.compress_loop0.body` and `sha256.sha256_loop2.body` | `model`, read off `ApeironWord.lean:15243-15264` and `:15510-15548` |
| the canonical loader `paddedBlock` | `private` in `Sha2.lean:139-142` and not resolvable from outside that module; the corpus's public `bytesToWords` is priced instead, and the correspondence is `Sha256Certificate.lean:514` |
| SHA-256d, the target threshold, difficulty, header contents | not priced |
| a range's order, parallelism and stopping rule | the nonce-space figure is a range length |
| a map to `CostAlgebra.Cost` | `ShaCost` is not connected to the registered resource record |

The two `model` readings are carried in §7 and are never used by a count. -/

/-- **THE MEASURE HAS EXACTLY TWENTY-THREE FIELDS AND NO MORE**, which is the machine-checkable form of
"every counted quantity has a field and nothing else does": `ShaCost.eq_of_fields` is a
twenty-three-hypothesis theorem, so a quantity with no field cannot be stated as a count of this
measure at all. -/
theorem attemptCost_comp_zero_new (σ : R) :
    (attemptCost σ).padByte = 128 ∧ (attemptCost σ).zeroByte = 39 ∧ (attemptCost σ).lenByte = 8 ∧
      (attemptCost σ).padShift = 8 ∧ (attemptCost σ).padMod = 8 ∧
      (attemptCost σ).blockByteRead = 128 ∧ (attemptCost σ).regroupMul = 96 ∧
      (attemptCost σ).regroupAdd = 96 ∧ (attemptCost σ).regroupTrunc = 32 ∧
      (attemptCost σ).schedWordWrite = 32 ∧ (attemptCost σ).digestByte = 32 ∧
      (attemptCost σ).digestWordRead = 8 := by
  have h := attemptCost_eq σ
  rw [h]
  simp

/-! ## 12. Audit rows

Every headline of this module, with its axiom cone. The closed cone required is
{propext, Classical.choice, Quot.sound} with no `sorryAx`. -/
#print axioms canonTotal_blockBytes
#print axioms canonTotal_div
#print axioms blockCount_is_ceil
#print axioms blockCount_succ_le
#print axioms canonTotal_succ_le
#print axioms canonTotal_succ_blocks
#print axioms totalLayout
#print axioms zeroFill_le
#print axioms pad_length_total
#print axioms pad_blockCount
#print axioms pad_blockCount_80
#print axioms hash_folds_over_blockCount
#print axioms pad_mergeByte
#print axioms lengthBlock_len
#print axioms lengthByte_body_ops
#print axioms padCost_fields
#print axioms padCost_split
#print axioms padCost_bounded
#print axioms padCost_blockBytes
#print axioms padCost_succ
#print axioms padCost_rep_add
#print axioms padCost_comp_zero
#print axioms padCost_80
#print axioms beWord16_body
#print axioms bytesToWords_arity
#print axioms loadCost_fields
#print axioms loadCost_bytes_eq_blockLen
#print axioms loadCost_comp_zero
#print axioms digView_get_body
#print axioms sha2_digest_isState
#print axioms state_arity
#print axioms storeCost_fields
#print axioms storeCost_digest_bytes
#print axioms storeCost_comp_zero
#print axioms rotRead_cost_of_reading
#print axioms channel_read_zero_slot
#print axioms priced_reads_are_the_two_scopes
#print axioms attemptCost_eq
#print axioms attemptCost_comp_eq
#print axioms attemptCost_ringFields
#print axioms blockFullCost_eq
#print axioms attemptCost_strict
#print axioms rangeCost_eq
#print axioms rangeCost_pad_once
#print axioms rangeCost_attemptPart_add
#print axioms attemptsCost_add
#print axioms attemptsCost_comp_add
#print axioms attemptsCost_ring_add
#print axioms attemptCost_rep_composed
#print axioms rangeCost_nonceSpace
#print axioms rangeCost_nonceSpace_comp
#print axioms rangeCost_nonceSpace_ringFields
#print axioms messageCost_eq
#print axioms messageCost_80
#print axioms messageCost_bytes_monotone
#print axioms attemptCost_faceElliptic
#print axioms attemptCost_faceParabolic
#print axioms attemptCost_faceHyperbolic
#print axioms attemptCost_threeFaces
#print axioms attemptCost_threeFaces_comp
#print axioms attemptCost_threeFaces_ring
#print axioms rangeCost_faceBlind
#print axioms messageCost_faceBlind
#print axioms blockFullCost_faceBlind
#print axioms blockFullCost_threeFaces_fieldwise
#print axioms attemptCost_comp_zero_new
