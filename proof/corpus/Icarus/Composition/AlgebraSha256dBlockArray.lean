/-
`AlgebraSha256dBlockArray.lean` — the canonical 64-byte array construction and the
unconditional byte transport for SHA-256d's outer block.

Task AR-46 continues AR-40 (`work/proofs/AlgebraSha256dCompose.lean`, read-only and
hash-bound here; that receipt names this row as its next bounded task). AR-40's
`bytesToWords_eq_blockOf` carries the byte premise

  `h : ∀ j, j < 64 → (blk.val.getD j (0#u8)).bv = p.getD j 0`

so it transports any 64-byte array whose bytes are the byte list's bytes, and the canonical
array construction is left to the certificate owner. This file builds that construction
(`blockArray`), proves its length and byte-access properties, and instantiates the transport
at it with no premise:

  `bytesToWords_blockArray (p) : apeiron_word.bytesToWords (blockArray p) = blockOf p 0`.

`blockArray` is the shape the canonical reader consumes: `apeiron_word.bytesToWords`
(`Crypto/Sha256Certificate.lean:493`) reads an `Array Std.U8 64#usize` byte `j` through
`Std.U8.bv`; the extracted `sha256` (`src/sha256.rs`) holds each compressed block as
`[u8; 64]`, which Aeneas models as `Array Std.U8 64#usize`.

AR-40 is not on the pinned import path (`work/proofs` is not a lake package root), so the
AR-40 definitions used below — `beWordAt`/`blockOf`/`padBlock32` (its §2), the private
reader's body `paddedBlockBody` and the Word-fold bridge (its §1, §4), `wordByte`/
`stateToBytes`/`pad_len` and the one-block collapse (its §6-§7) — are restated verbatim from
that file and re-elaborated here; the kernel re-checks each, and
`hash_eq_fold_paddedBlockBody` remains the definitional identification of the canonical
`Sha2.hash` with the fold over the transcription of the private `Sha2.paddedBlock`
(`Crypto/Sha2.lean:139-142`). The private name is never written.

`lengthBlock_collapse_eq` (AR-14) is a length-field collision only; nothing here restates it
as a message collision, and nothing here is a security, cost or mining claim. Evidence:
`#print axioms` rows at the end, within `{propext, Classical.choice, Quot.sound}`.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

namespace AlgebraSha256dBlockArray

open Apeiron.Crypto.Sha2 (Word State Block H0 lengthBlock pad compress)
open Aeneas Aeneas.Std Result
open apeiron_word (vec_getD_eq)

/-! ## 1 · The canonical 64-byte array construction

`blockArray p` is the `[u8; 64]` whose byte `j < 64` is `p`'s byte `j`; the type fixes the
length, reads are through `Std.U8.bv`, and the construction is total — no `Option`, no
failure branch, no premise. -/

/-- The 64-byte extracted array of a byte list, the construction the canonical reader
    `apeiron_word.bytesToWords` consumes: byte `j < 64` is `p.getD j 0`. -/
def blockArray (p : List (BitVec 8)) : Array Std.U8 64#usize :=
  Array.from (List.ofFn fun j : Fin 64 => (⟨p.getD j.val 0⟩ : Std.U8))
    (by rw [List.length_ofFn, show (64#usize : Std.Usize).val = 64 from rfl])

/-- The construction has length `64`, the length its `Array` type fixes. -/
theorem blockArray_length (p : List (BitVec 8)) : (blockArray p).val.length = 64 := by
  have h := Array.property (blockArray p)
  rw [show (64#usize : Std.Usize).val = 64 from rfl] at h
  exact h

private theorem list_ofFn_getD {α : Type} [Inhabited α] {n : Nat} (f : Fin n → α)
    (d : α) (i : Nat) (hi : i < n) : (List.ofFn f).getD i d = f ⟨i, hi⟩ := by
  simp [List.getD, List.getElem?_ofFn, hi]

/-- **Byte access.** Byte `j` of the construction is `p`'s byte `j`, for every `j < 64`;
    this is the premise AR-40's `bytesToWords_eq_blockOf` carries, here discharged. -/
theorem blockArray_getD (p : List (BitVec 8)) (j : Nat) (hj : j < 64) :
    ((blockArray p).val.getD j (0#u8)).bv = p.getD j 0 := by
  rw [blockArray, Array.from_val, list_ofFn_getD _ _ j hj]

/-- The byte property is inhabited at the actual list `[0x61, 0x62]`, byte `1`. -/
example : ((blockArray [0x61, 0x62]).val.getD 1 (0#u8)).bv = 0x62 :=
  blockArray_getD [0x61, 0x62] 1 (by omega)

/-- And at a byte above the message's own length, where the read is the zero default. -/
example : ((blockArray [0x61]).val.getD 5 (0#u8)).bv = 0 :=
  blockArray_getD [0x61] 5 (by omega)

/-! ## 2 · The closed block read, and the unconditional transport

`beWordAt`/`blockOf` are AR-40 §2's closed form of the block read, verbatim.
`apeiron_word.bytesToWords` is the corpus's public certificate read
(`Crypto/Sha256Certificate.lean:493`). -/

/-- Word at byte offset `o`, closed form: `b₀·2²⁴ + b₁·2¹⁶ + b₂·2⁸ + b₃` (AR-40 `beWordAt`). -/
def beWordAt (b : List (BitVec 8)) (o : Nat) : Word :=
  BitVec.ofNat 32
    ((b.getD o 0).toNat * 16777216 + (b.getD (o + 1) 0).toNat * 65536
      + (b.getD (o + 2) 0).toNat * 256 + (b.getD (o + 3) 0).toNat)

/-- The 64-byte block read of a byte list, closed form (AR-40 `blockOf`). -/
def blockOf (b : List (BitVec 8)) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 => beWordAt b (64 * k + 4 * i.val)

private theorem vec_ofFn_getD' {α : Type} {n : Nat} [Inhabited α] (f : Fin n → α)
    (d : α) (i : Nat) (hi : i < n) : (Vector.ofFn f).getD i d = f ⟨i, hi⟩ := by
  simp [Vector.getD, hi]

/-- **The transport at the canonical construction, unconditionally.** AR-40's
    `bytesToWords_eq_blockOf` at `blockArray p`, with its byte premise discharged by
    `blockArray_getD`. -/
theorem bytesToWords_blockArray (p : List (BitVec 8)) :
    apeiron_word.bytesToWords (blockArray p) = blockOf p 0 := by
  refine vec_getD_eq _ _ 0 fun i hi => ?_
  rw [apeiron_word.bytesToWords_getD (blockArray p) i hi, blockOf, vec_ofFn_getD' _ 0 i hi, beWordAt]
  simp only [apeiron_word.beWord16, apeiron_word.b64]
  rw [blockArray_getD p (4 * i) (by omega), blockArray_getD p (4 * i + 1) (by omega),
    blockArray_getD p (4 * i + 2) (by omega), blockArray_getD p (4 * i + 3) (by omega)]
  simp only [Nat.mul_zero, Nat.zero_add]

/-! ## 3 · The private reader's body, and the public bridge (AR-40 §1, §4 restated)

`paddedBlockBody` is the body of the private `Sha2.paddedBlock` (`Crypto/Sha2.lean:139-142`)
with the block index explicit, byte-for-byte. Because it is that body, the kernel's
definitional equality equates it with the private constant, so every statement below that
uses `paddedBlockBody` is a statement about the canonical reader; only the printable name
differs. -/

/-- The body of the private `Sha2.paddedBlock` (`Crypto/Sha2.lean:139-142`), with the block
    index explicit (AR-40 `paddedBlockBody`, verbatim). -/
def paddedBlockBody (p : List (BitVec 8)) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 =>
    (List.range 4).foldl (fun acc j =>
      ((acc.toNat <<< 8) ||| (p.getD (64 * k + 4 * i.val + j) 0).toNat) % 4294967296) 0

/-- The canonical `Sha2.hash` fold is the `paddedBlockBody` fold, definitionally: `rfl`
    checks it for every message. -/
theorem hash_eq_fold_paddedBlockBody (m : List (BitVec 8)) :
    Apeiron.Crypto.Sha2.hash m = (List.range ((pad m).length / 64)).foldl
      (fun h k => compress h (paddedBlockBody (pad m) k)) H0 := rfl

/-- The 64-byte block read of a byte list, Horner form (AR-40 `padBlock32`). -/
def padBlock32 (p : List (BitVec 8)) : Block :=
  Vector.ofFn fun i : Fin 16 =>
    ((List.range 4).foldl
      (fun (acc : Nat) j => acc * 256 + (p.getD (4 * i.val + j) 0).toNat) 0) % 4294967296

private theorem foldl_range4 (f : Nat → Nat → Nat) (a : Nat) :
    (List.range 4).foldl f a = f (f (f (f a 0) 1) 2) 3 := by
  simp [List.range_succ]

private theorem foldl_range4_word (f : Word → Nat → Word) (a : Word) :
    (List.range 4).foldl f a = f (f (f (f a 0) 1) 2) 3 := by
  simp [List.range_succ]

private theorem byte_lt_256 (c : BitVec 8) : c.toNat < 256 := by
  have h := c.isLt
  omega

/-- One Word-typed step of the transcribed private body's fold (AR-40 `wstep`). -/
private def wstep (acc : Word) (c : Nat) : Word :=
  (((acc.toNat <<< 8 : Nat) : Word) ||| ((c : Nat) : Word)) % (4294967296 : Word)

/-- The same step in `Nat`, without the width truncation (AR-40 `nstep`). -/
private def nstep (acc c : Nat) : Nat := (acc <<< 8) ||| c

private theorem nstep_eq (a c : Nat) (hc : c < 256) : nstep a c = a * 256 + c := by
  unfold nstep
  rw [← Nat.shiftLeft_add_eq_or_of_lt (b := c) (i := 8) (by omega), Nat.shiftLeft_eq]

private theorem wstep_toNat (acc : Word) (c : Nat)
    (ha : acc.toNat <<< 8 < 4294967296) (hc : c < 256) :
    (wstep acc c).toNat = nstep acc.toNat c := by
  have h0 : (4294967296 : Word) = 0#32 := by decide
  unfold wstep nstep
  rw [h0, BitVec.umod_zero]
  rw [BitVec.toNat_or]
  simp only [BitVec.natCast_eq_ofNat, BitVec.toNat_ofNat,
    show (2 : Nat) ^ 32 = 4294967296 by norm_num]
  rw [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt (by omega : c < 4294967296)]

/-- The private body's Word-typed four-step fold is the closed form. -/
private theorem word_fold_eq_closed (b : List (BitVec 8)) (o : Nat) :
    wstep (wstep (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
        (b.getD (o + 2) 0).toNat) (b.getD (o + 3) 0).toNat
      = BitVec.ofNat 32 ((b.getD o 0).toNat * 16777216
          + (b.getD (o + 1) 0).toNat * 65536 + (b.getD (o + 2) 0).toNat * 256
          + (b.getD (o + 3) 0).toNat) := by
  have hb0 := byte_lt_256 (b.getD o 0)
  have hb1 := byte_lt_256 (b.getD (o + 1) 0)
  have hb2 := byte_lt_256 (b.getD (o + 2) 0)
  have hb3 := byte_lt_256 (b.getD (o + 3) 0)
  have hn1 : nstep 0 (b.getD o 0).toNat = (b.getD o 0).toNat := by
    rw [nstep_eq 0 _ hb0]; omega
  have hn2 : nstep ((b.getD o 0).toNat) (b.getD (o + 1) 0).toNat
      = (b.getD o 0).toNat * 256 + (b.getD (o + 1) 0).toNat := nstep_eq _ _ hb1
  have hn3 : nstep ((b.getD o 0).toNat * 256 + (b.getD (o + 1) 0).toNat)
        (b.getD (o + 2) 0).toNat
      = (b.getD o 0).toNat * 65536 + (b.getD (o + 1) 0).toNat * 256
        + (b.getD (o + 2) 0).toNat := by
    rw [nstep_eq _ _ hb2]; ring
  have hn4 : nstep ((b.getD o 0).toNat * 65536 + (b.getD (o + 1) 0).toNat * 256
        + (b.getD (o + 2) 0).toNat) (b.getD (o + 3) 0).toNat
      = (b.getD o 0).toNat * 16777216 + (b.getD (o + 1) 0).toNat * 65536
        + (b.getD (o + 2) 0).toNat * 256 + (b.getD (o + 3) 0).toNat := by
    rw [nstep_eq _ _ hb3]; ring
  have h1 : (wstep 0 (b.getD o 0).toNat).toNat = nstep 0 (b.getD o 0).toNat :=
    wstep_toNat 0 _ (by simp) hb0
  have h2b : (wstep 0 (b.getD o 0).toNat).toNat <<< 8 < 4294967296 := by
    rw [h1, hn1, Nat.shiftLeft_eq]; omega
  have h2 : (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat).toNat
      = nstep (nstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat := by
    rw [wstep_toNat (wstep 0 (b.getD o 0).toNat) _ h2b hb1, h1]
  have h3b : (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat).toNat <<< 8
      < 4294967296 := by
    rw [h2, hn1, hn2, Nat.shiftLeft_eq]; omega
  have h3 : (wstep (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
        (b.getD (o + 2) 0).toNat).toNat
      = nstep (nstep (nstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
          (b.getD (o + 2) 0).toNat := by
    rw [wstep_toNat (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat) _
      h3b hb2, h2]
  have h4b : (wstep (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
        (b.getD (o + 2) 0).toNat).toNat <<< 8 < 4294967296 := by
    rw [h3, hn1, hn2, hn3, Nat.shiftLeft_eq]; omega
  have h4 : (wstep (wstep (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
        (b.getD (o + 2) 0).toNat) (b.getD (o + 3) 0).toNat).toNat
      = nstep (nstep (nstep (nstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
          (b.getD (o + 2) 0).toNat) (b.getD (o + 3) 0).toNat := by
    rw [wstep_toNat (wstep (wstep (wstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
      (b.getD (o + 2) 0).toNat) _ h4b hb3, h3]
  have hbnd : nstep (nstep (nstep (nstep 0 (b.getD o 0).toNat) (b.getD (o + 1) 0).toNat)
      (b.getD (o + 2) 0).toNat) (b.getD (o + 3) 0).toNat < 4294967296 := by
    rw [hn1, hn2, hn3, hn4]; omega
  have hbnd' : (b.getD o 0).toNat * 16777216 + (b.getD (o + 1) 0).toNat * 65536
      + (b.getD (o + 2) 0).toNat * 256 + (b.getD (o + 3) 0).toNat < 4294967296 := by
    rw [hn1, hn2, hn3, hn4] at hbnd
    exact hbnd
  apply BitVec.eq_of_toNat_eq
  rw [h4, BitVec.toNat_ofNat, show (2 : Nat) ^ 32 = 4294967296 by norm_num,
    Nat.mod_eq_of_lt hbnd']
  rw [hn1, hn2, hn3, hn4]

/-- The private body's Word-typed fold equals the closed form. -/
theorem blockOf_eq_paddedBlockBody (p : List (BitVec 8)) (k : Nat) :
    blockOf p k = paddedBlockBody p k := by
  refine vec_getD_eq _ _ 0 fun i hi => ?_
  rw [blockOf, paddedBlockBody, vec_ofFn_getD' _ 0 i hi, vec_ofFn_getD' _ 0 i hi]
  have h := word_fold_eq_closed p (64 * k + 4 * i)
  have hexp : ((List.range 4).foldl (fun acc j =>
        (((acc.toNat <<< 8 : Nat) : Word)
          ||| (((p.getD (64 * k + 4 * i + j) 0).toNat : Nat) : Word))
        % (4294967296 : Word)) 0)
      = wstep (wstep (wstep (wstep 0 (p.getD (64 * k + 4 * i) 0).toNat)
          (p.getD (64 * k + 4 * i + 1) 0).toNat)
        (p.getD (64 * k + 4 * i + 2) 0).toNat)
        (p.getD (64 * k + 4 * i + 3) 0).toNat := by
    rw [foldl_range4_word]
    simp only [Nat.add_zero]
    rfl
  rw [beWordAt, hexp, h]

/-- The re-declared Horner read equals the closed form (AR-40 `padBlock32_eq_blockOf`). -/
theorem padBlock32_eq_blockOf (p : List (BitVec 8)) : padBlock32 p = blockOf p 0 := by
  refine vec_getD_eq _ _ 0 fun i hi => ?_
  rw [padBlock32, blockOf, vec_ofFn_getD' _ 0 i hi, vec_ofFn_getD' _ 0 i hi]
  rw [beWordAt]
  simp only [Nat.mul_zero, Nat.zero_add]
  have hfold : ((List.range 4).foldl
      (fun (acc : Nat) j => acc * 256 + (p.getD (4 * i + j) 0).toNat) 0)
      = (p.getD (4 * i) 0).toNat * 16777216 + (p.getD (4 * i + 1) 0).toNat * 65536
        + (p.getD (4 * i + 2) 0).toNat * 256 + (p.getD (4 * i + 3) 0).toNat := by
    rw [foldl_range4]
    simp only [Nat.add_zero]
    ring
  rw [hfold, show (4294967296 : Word) = 0#32 by decide, BitVec.umod_zero]
  exact BitVec.natCast_eq_ofNat 32 _

/-- The re-declared block read equals the canonical reader's body (AR-40
    `padBlock32_eq_paddedBlock`). -/
theorem padBlock32_eq_paddedBlock (p : List (BitVec 8)) :
    padBlock32 p = paddedBlockBody p 0 :=
  (padBlock32_eq_blockOf p).trans (blockOf_eq_paddedBlockBody p 0)

/-! ## 4 · The digest store and the one-block collapse (AR-40 §6-§7 restated) -/

/-- Byte `j` of a `Sha2.Word`, most significant first, as a `Nat` (AR-40 `wordByte`). -/
def wordByte (w : Word) (j : Nat) : Nat :=
  (w.toNat / 2 ^ (8 * (3 - j))) % 256

/-- The 32-byte big-endian flattening of a chaining state (AR-40 `stateToBytes`). -/
def stateToBytes (s : State) : List (BitVec 8) :=
  List.ofFn fun k : Fin 32 => (wordByte (s.getD (k / 4) 0) (k % 4) : BitVec 8)

theorem lengthBlock_length (n : Nat) : (lengthBlock n).length = 8 := by
  simp [lengthBlock]

theorem stateToBytes_length (s : State) : (stateToBytes s).length = 32 := by
  simp [stateToBytes, List.length_ofFn]

/-- `Sha2.pad`'s padded length, `Nat` arithmetic only (AR-40 `pad_len`). -/
theorem pad_len (m : List (BitVec 8)) :
    (pad m).length = m.length + 9 + (64 - (m.length + 9) % 64) % 64 := by
  have hz : pad m = m ++ [0x80] ++ List.replicate
      ((if (m.length + 9) % 64 = 0 then m.length + 9
        else m.length + 9 + (64 - (m.length + 9) % 64)) - m.length - 9) 0
      ++ lengthBlock m.length := rfl
  rw [hz]
  simp only [List.length_append, List.length_cons, List.length_nil, List.length_replicate,
    lengthBlock_length]
  by_cases h : (m.length + 9) % 64 = 0
  · simp only [h, if_true]
    omega
  · simp only [h, if_false]
    omega

/-- The 32-byte digest store pads to exactly one block, for every state. -/
theorem pad_stateToBytes_length (s : State) : (pad (stateToBytes s)).length = 64 := by
  have h := pad_len (stateToBytes s)
  rw [stateToBytes_length] at h
  omega

/-- One-block `Sha2.hash` equals one compression from `H0` of the canonical reader's body
    (AR-40 `hash_one_block_eq_compress_H0`). -/
theorem hash_one_block_eq_compress_H0 (m : List (BitVec 8)) (h : (pad m).length = 64) :
    Apeiron.Crypto.Sha2.hash m = compress H0 (paddedBlockBody (pad m) 0) := by
  simp only [Apeiron.Crypto.Sha2.hash]
  rw [h]
  rfl

/-- **The outer half of SHA-256d, one-block premise discharged at the real call** (AR-40
    `hash_outer_eq_compress_H0`). -/
theorem hash_outer_eq_compress_H0 (s : State) :
    Apeiron.Crypto.Sha2.hash (stateToBytes s)
      = compress H0 (paddedBlockBody (pad (stateToBytes s)) 0) :=
  hash_one_block_eq_compress_H0 (stateToBytes s) (pad_stateToBytes_length s)

/-- The same at the re-declared Horner reader (AR-40's `padBlock32` form). -/
theorem hash_outer_eq_compress_padBlock32 (s : State) :
    Apeiron.Crypto.Sha2.hash (stateToBytes s)
      = compress H0 (padBlock32 (pad (stateToBytes s))) := by
  rw [hash_outer_eq_compress_H0 s]
  rw [← padBlock32_eq_paddedBlock]

/-! ## 5 · Wiring through AR-40's composition statements

The unconditional transport at `blockArray` composes with AR-40's bridge to the private
reader's body and with AR-40's one-block collapse. The consumer form: the second application
of SHA-256d is one `compress H0` of the `bytesToWords` read of the 64-byte array of the
padded digest store, with no hypothesis. -/

/-- The transport at the re-declared Horner reader; a consumer rewriting AR-40's
    `padBlock32` statements uses this. -/
theorem bytesToWords_blockArray_eq_padBlock32 (p : List (BitVec 8)) :
    apeiron_word.bytesToWords (blockArray p) = padBlock32 p :=
  (bytesToWords_blockArray p).trans (padBlock32_eq_blockOf p).symm

/-- The transport at the canonical reader's body. -/
theorem bytesToWords_blockArray_eq_paddedBlockBody (p : List (BitVec 8)) :
    apeiron_word.bytesToWords (blockArray p) = paddedBlockBody p 0 :=
  (bytesToWords_blockArray p).trans (blockOf_eq_paddedBlockBody p 0)

/-- **The outer half of SHA-256d at the array read, no hypothesis.** -/
theorem hash_outer_eq_compress_bytesToWords_blockArray (s : State) :
    Apeiron.Crypto.Sha2.hash (stateToBytes s)
      = compress H0 (apeiron_word.bytesToWords (blockArray (pad (stateToBytes s)))) := by
  rw [hash_outer_eq_compress_H0 s, ← bytesToWords_blockArray_eq_paddedBlockBody]

/-- **The SHA-256d composition at the array read.** SHA-256d (the canonical `Sha2.hash`
    applied to the canonical 32-byte store of its own output) is one compression from `H0`
    of the `bytesToWords` read of the 64-byte array of the padded digest store, for every
    message; no premise. -/
theorem sha256d_eq_outer_compress_bytesToWords_blockArray (m : List (BitVec 8)) :
    Apeiron.Crypto.Sha2.hash (stateToBytes (Apeiron.Crypto.Sha2.hash m))
      = compress H0 (apeiron_word.bytesToWords
          (blockArray (pad (stateToBytes (Apeiron.Crypto.Sha2.hash m))))) :=
  hash_outer_eq_compress_bytesToWords_blockArray (Apeiron.Crypto.Sha2.hash m)

/-! ## 6 · Fail-closed axiom cone -/

#print axioms blockArray
#print axioms blockArray_length
#print axioms blockArray_getD
#print axioms bytesToWords_blockArray
#print axioms beWordAt
#print axioms blockOf
#print axioms paddedBlockBody
#print axioms hash_eq_fold_paddedBlockBody
#print axioms padBlock32
#print axioms padBlock32_eq_blockOf
#print axioms blockOf_eq_paddedBlockBody
#print axioms padBlock32_eq_paddedBlock
#print axioms wordByte
#print axioms stateToBytes
#print axioms lengthBlock_length
#print axioms stateToBytes_length
#print axioms pad_len
#print axioms pad_stateToBytes_length
#print axioms hash_one_block_eq_compress_H0
#print axioms hash_outer_eq_compress_H0
#print axioms hash_outer_eq_compress_padBlock32
#print axioms bytesToWords_blockArray_eq_padBlock32
#print axioms bytesToWords_blockArray_eq_paddedBlockBody
#print axioms hash_outer_eq_compress_bytesToWords_blockArray
#print axioms sha256d_eq_outer_compress_bytesToWords_blockArray

end AlgebraSha256dBlockArray
