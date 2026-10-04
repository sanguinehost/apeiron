/-
`AlgebraSha256dCompose.lean` — the canonical outer-block equality for SHA-256d.

Task AR-40, the smallest slice of the round -> compression -> digest -> double-application
row of `receipts/sol-face-adjudication.md` §5: the second application of `Sha2.hash` in
SHA-256d is exactly one compression from `H0` of the 64-byte block whose leading eight
words are the inner digest, and every domain premise is discharged at the real call.

The one obstruction: `Apeiron.Crypto.Sha2.paddedBlock` (`Crypto/Sha2.lean:139`) is
`private`, so its name cannot appear in a statement elaborated from this module (the
environment name is `_private.Crypto.Sha2.0.Apeiron.Crypto.Sha2.paddedBlock` and the name
resolver rejects it; probes `alg-sha256d-compose-probe2.log`, plus AR-20/AR-27's recorded
probes). `paddedBlockBody` below is the private reader's body transcribed verbatim. It is
not a copy on the executing path and not an axiom: the kernel unfolds it to the same term
as the private constant, so `rfl` checks that the canonical `Sha2.hash` is the fold over
`paddedBlockBody` (probes `alg-sha256d-compose-probe1.log`, `probe3.log`). The private name
itself is never written.

Evidence: `#print axioms` rows at the end, within `{propext, Classical.choice, Quot.sound}`.
`lengthBlock_collapse_eq` (AR-14) is a length-field collision only; nothing here restates
it as a message collision, and nothing here is a security, cost or mining claim.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false

namespace AlgebraSha256dCompose

open Apeiron.Crypto.Sha2 (Word State Block H0 lengthBlock pad compress)
open Aeneas Aeneas.Std Result
open apeiron_word (vec_getD_eq)

/-! ## 1 · The canonical block reader in public form

`paddedBlockBody p k` is the body of the private `Sha2.paddedBlock` (`Crypto/Sha2.lean:139-142`)
with the block index `k` explicit, byte-for-byte. Because it is that body, the kernel's
definitional equality equates it with the private constant, so every statement below that
uses `paddedBlockBody` is a statement about the canonical reader; only the printable name
differs. -/

def paddedBlockBody (p : List (BitVec 8)) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 =>
    (List.range 4).foldl (fun acc j =>
      ((acc.toNat <<< 8) ||| (p.getD (64 * k + 4 * i.val + j) 0).toNat) % 4294967296) 0

/-- The canonical `Sha2.hash` fold is the `paddedBlockBody` fold, definitionally: `rfl`
checks it for every message. -/
theorem hash_eq_fold_paddedBlockBody (m : List (BitVec 8)) :
    Apeiron.Crypto.Sha2.hash m = (List.range ((pad m).length / 64)).foldl
      (fun h k => compress h (paddedBlockBody (pad m) k)) H0 := rfl

/-! ## 2 · The re-declared reader, and its independent closed form

`wordByte`/`stateToBytes`/`padBlock32` are re-declared verbatim from
`work/proofs/AlgebraSha256d.lean:74-155` (read-only there; the pinned lane does not put
`work/proofs` on the import path, so the bodies are restated and cited). `beWordAt`/
`blockOf` are the closed form of `paddedBlockBody`, in the spelling of
`work/proofs/Sha256Blocks.lean:103-113` (`beWordAt`/`msgBlock`) and of
`Crypto/Sha256Certificate.lean:139-154` (`byte32`/`beWord`/`digView`). -/

/-- Byte `j` of a `Sha2.Word`, most significant first, as a `Nat` (AR-14 `wordByte`). -/
def wordByte (w : Word) (j : Nat) : Nat :=
  (w.toNat / 2 ^ (8 * (3 - j))) % 256

/-- The 32-byte big-endian flattening of a chaining state (AR-14 `stateToBytes`). -/
def stateToBytes (s : State) : List (BitVec 8) :=
  List.ofFn fun k : Fin 32 => (wordByte (s.getD (k / 4) 0) (k % 4) : BitVec 8)

/-- The 64-byte block read of a byte list, Horner form (AR-14 `padBlock32`). -/
def padBlock32 (p : List (BitVec 8)) : Block :=
  Vector.ofFn fun i : Fin 16 =>
    ((List.range 4).foldl
      (fun (acc : Nat) j => acc * 256 + (p.getD (4 * i.val + j) 0).toNat) 0) % 4294967296

/-- Word at byte offset `o`, closed form: `b₀·2²⁴ + b₁·2¹⁶ + b₂·2⁸ + b₃`. -/
def beWordAt (b : List (BitVec 8)) (o : Nat) : Word :=
  BitVec.ofNat 32
    ((b.getD o 0).toNat * 16777216 + (b.getD (o + 1) 0).toNat * 65536
      + (b.getD (o + 2) 0).toNat * 256 + (b.getD (o + 3) 0).toNat)

/-- The 64-byte block read of a byte list, closed form. -/
def blockOf (b : List (BitVec 8)) (k : Nat) : Block :=
  Vector.ofFn fun i : Fin 16 => beWordAt b (64 * k + 4 * i.val)

/-! ## 3 · Vector and list infrastructure -/

private theorem vec_ofFn_getD' {α : Type} {n : Nat} [Inhabited α] (f : Fin n → α)
    (d : α) (i : Nat) (hi : i < n) : (Vector.ofFn f).getD i d = f ⟨i, hi⟩ := by
  simp [Vector.getD, hi]

private theorem list_ofFn_getD {α : Type} [Inhabited α] {n : Nat} (f : Fin n → α)
    (d : α) (i : Nat) (hi : i < n) : (List.ofFn f).getD i d = f ⟨i, hi⟩ := by
  simp [List.getD, List.getElem?_ofFn, hi]

private theorem natCast8_toNat (n : Nat) : ((n : Nat) : BitVec 8).toNat = n % 256 := rfl

private theorem getD_append_of_lt {α : Type} [Inhabited α] (l₁ l₂ : List α) (i : Nat) (d : α)
    (h : i < l₁.length) : (l₁ ++ l₂).getD i d = l₁.getD i d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_append_left h]

private theorem getD_append_snoc2 {α : Type} [Inhabited α] (a b₁ b₂ : List α) (i : Nat) (d : α)
    (h : i < a.length) : ((a ++ b₁) ++ b₂).getD i d = a.getD i d := by
  rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
  rw [List.getElem?_append]
  split
  · rename_i h₁
    rw [List.getElem?_append]
    split
    · rfl
    · exfalso
      omega
  · exfalso
    rename_i h₁
    simp only [List.length_append] at h₁
    omega

/-- `pad` does not change the bytes the message itself supplies. -/
theorem pad_getD_of_lt (m : List (BitVec 8)) (i : Nat) (h : i < m.length) :
    (pad m).getD i 0 = m.getD i 0 := by
  have hz : pad m = m ++ [0x80] ++ List.replicate
      ((if (m.length + 9) % 64 = 0 then m.length + 9
        else m.length + 9 + (64 - (m.length + 9) % 64)) - m.length - 9) 0
      ++ lengthBlock m.length := rfl
  rw [hz]
  have h1 : i < (m ++ [0x80]).length := by
    simp only [List.length_append, List.length_cons, List.length_nil]
    omega
  by_cases hr : (m.length + 9) % 64 = 0
  · simp only [hr, if_true]
    rw [getD_append_snoc2 (m ++ [0x80]) _ _ i 0 h1, getD_append_of_lt m [0x80] i 0 h]
  · simp only [hr, if_false]
    rw [getD_append_snoc2 (m ++ [0x80]) _ _ i 0 h1, getD_append_of_lt m [0x80] i 0 h]

theorem lengthBlock_length (n : Nat) : (lengthBlock n).length = 8 := by
  simp [lengthBlock]

theorem stateToBytes_length (s : State) : (stateToBytes s).length = 32 := by
  simp [stateToBytes, List.length_ofFn]

/-! ## 4 · `padBlock32` equals the canonical block reader

The closed form is reached first (`padBlock32_eq_blockOf`), then the private body's
Word-typed shift-or fold (`blockOf_eq_paddedBlockBody`), then the headline. The fold
arithmetic is the `beWordAt_eq_fold` statement of `work/proofs/Sha256Blocks.lean:160`,
proved here at the `List (BitVec 8)` types this file uses. -/

private theorem foldl_range4 (f : Nat → Nat → Nat) (a : Nat) :
    (List.range 4).foldl f a = f (f (f (f a 0) 1) 2) 3 := by
  simp [List.range_succ]

private theorem foldl_range4_word (f : Word → Nat → Word) (a : Word) :
    (List.range 4).foldl f a = f (f (f (f a 0) 1) 2) 3 := by
  simp [List.range_succ]

private theorem byte_lt_256 (c : BitVec 8) : c.toNat < 256 := by
  have h := c.isLt
  omega

private theorem shl_or_step (c a : Nat) (hc : c < 256) (ha : a < 16777216) :
    ((a <<< 8) ||| c) % 4294967296 = a * 256 + c := by
  rw [← Nat.shiftLeft_add_eq_or_of_lt (b := c) (i := 8) (by omega), Nat.shiftLeft_eq]
  omega

/-- The closed-form regrouping of four bytes is the `List.range 4` shift-and-or
accumulation in `Nat` spelling of `Sha2.paddedBlock`. -/
theorem beWordAt_eq_fold (b : List (BitVec 8)) (o : Nat) :
    beWordAt b o = BitVec.ofNat 32
      ((List.range 4).foldl
        (fun (acc : Nat) j => ((acc <<< 8) ||| (b.getD (o + j) 0).toNat) % 4294967296) 0) := by
  have e0 := byte_lt_256 (b.getD o 0)
  have e1 := byte_lt_256 (b.getD (o + 1) 0)
  have e2 := byte_lt_256 (b.getD (o + 2) 0)
  have e3 := byte_lt_256 (b.getD (o + 3) 0)
  have hfold : ((List.range 4).foldl
        (fun (acc : Nat) j => ((acc <<< 8) ||| (b.getD (o + j) 0).toNat) % 4294967296) 0)
      = (b.getD o 0).toNat * 16777216 + (b.getD (o + 1) 0).toNat * 65536
        + (b.getD (o + 2) 0).toNat * 256 + (b.getD (o + 3) 0).toNat := by
    rw [foldl_range4]
    simp only [Nat.add_zero]
    rw [shl_or_step _ 0 e0 (by omega)]
    rw [shl_or_step _ _ e1 (by omega)]
    rw [shl_or_step _ _ e2 (by omega)]
    rw [shl_or_step _ _ e3 (by omega)]
    omega
  unfold beWordAt
  exact congrArg (BitVec.ofNat 32) hfold.symm

/-- One Word-typed step of the transcribed private body's fold. -/
private def wstep (acc : Word) (c : Nat) : Word :=
  (((acc.toNat <<< 8 : Nat) : Word) ||| ((c : Nat) : Word)) % (4294967296 : Word)

/-- The same step in `Nat`, without the width truncation. -/
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

/-- The re-declared Horner read equals the closed form. -/
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

/-- **The re-declared block read equals the canonical reader.** `paddedBlockBody` is the
body of the private `Sha2.paddedBlock`; the kernel equates them by definition, so this is
`padBlock32 p = Sha2.paddedBlock p 0` in printable form. -/
theorem padBlock32_eq_paddedBlock (p : List (BitVec 8)) :
    padBlock32 p = paddedBlockBody p 0 :=
  (padBlock32_eq_blockOf p).trans (blockOf_eq_paddedBlockBody p 0)

/-! ## 5 · The `bytesToWords` transport

The corpus's public certificate read of an extracted `64`-byte block
(`Crypto/Sha256Certificate.lean:493`) agrees with the closed block read whenever the
array's bytes are the byte list's bytes. The array itself is the certificate owner's
construction; no injectivity direction is claimed. -/

theorem bytesToWords_eq_blockOf (blk : Array Std.U8 64#usize) (p : List (BitVec 8))
    (h : ∀ j, j < 64 → (blk.val.getD j (0#u8)).bv = p.getD j 0) :
    apeiron_word.bytesToWords blk = blockOf p 0 := by
  refine vec_getD_eq _ _ 0 fun i hi => ?_
  rw [apeiron_word.bytesToWords_getD blk i hi, blockOf, vec_ofFn_getD' _ 0 i hi, beWordAt]
  simp only [apeiron_word.beWord16, apeiron_word.b64]
  rw [h (4 * i) (by omega), h (4 * i + 1) (by omega), h (4 * i + 2) (by omega),
    h (4 * i + 3) (by omega)]
  simp only [Nat.mul_zero, Nat.zero_add]

/-! ## 6 · The one-block collapse

For any message whose padding is a single block, `Sha2.hash` is one compression from `H0`
of the canonical block. The right-hand side names the private reader's body
(`paddedBlockBody`) and, in the second form, the re-declared reader. -/

theorem hash_one_block_eq_compress_H0 (m : List (BitVec 8)) (h : (pad m).length = 64) :
    Apeiron.Crypto.Sha2.hash m = compress H0 (paddedBlockBody (pad m) 0) := by
  simp only [Apeiron.Crypto.Sha2.hash]
  rw [h]
  rfl

theorem hash_one_block_eq_compress_H0_padBlock32 (m : List (BitVec 8))
    (h : (pad m).length = 64) :
    Apeiron.Crypto.Sha2.hash m = compress H0 (padBlock32 (pad m)) := by
  rw [hash_one_block_eq_compress_H0 m h, ← padBlock32_eq_paddedBlock]

/-! ## 7 · Composition on reachable states

The outer application of SHA-256d receives the 32-byte digest store, which pads to exactly
one block (`pad_stateToBytes_length`, for every state, no hypothesis carried), and the
block's leading eight words are the digest's own words (`blockOf_pad_stateToBytes_getD`).
The composition theorem then has no premise: the second application is one compression
from `H0` of the digest block. -/

/-- `Sha2.pad`'s padded length, `Nat` arithmetic only (AR-14 `pad_len`). -/
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

/-- `outer_compression_count` (`work/proofs/AlgebraSha256d.lean:226`), re-derived at these
definitions: one outer block. -/
theorem outer_compression_count (s : State) : (pad (stateToBytes s)).length / 64 = 1 := by
  rw [pad_stateToBytes_length]

/-- **The outer half of SHA-256d, with the domain premise discharged at the real call.**
The second application is exactly one compression from `H0` of the canonical block read of
the padded 32-byte digest store. No hypothesis. -/
theorem hash_outer_eq_compress_H0 (s : State) :
    Apeiron.Crypto.Sha2.hash (stateToBytes s)
      = compress H0 (paddedBlockBody (pad (stateToBytes s)) 0) :=
  hash_one_block_eq_compress_H0 (stateToBytes s) (pad_stateToBytes_length s)

/-- **The composition on reachable states.** SHA-256d (the canonical `Sha2.hash` applied
to the canonical 32-byte store of its own output) is one compression from `H0` of the
digest block, for every message; the one-block premise is discharged by
`pad_stateToBytes_length`, not carried. -/
theorem sha256d_eq_outer_compress_H0 (m : List (BitVec 8)) :
    Apeiron.Crypto.Sha2.hash (stateToBytes (Apeiron.Crypto.Sha2.hash m))
      = compress H0 (paddedBlockBody (pad (stateToBytes (Apeiron.Crypto.Sha2.hash m))) 0) :=
  hash_outer_eq_compress_H0 (Apeiron.Crypto.Sha2.hash m)

/-- The same composition with the re-declared reader on the right. -/
theorem sha256d_eq_outer_compress_padBlock32 (m : List (BitVec 8)) :
    Apeiron.Crypto.Sha2.hash (stateToBytes (Apeiron.Crypto.Sha2.hash m))
      = compress H0 (padBlock32 (pad (stateToBytes (Apeiron.Crypto.Sha2.hash m)))) := by
  rw [hash_outer_eq_compress_H0 (Apeiron.Crypto.Sha2.hash m)]
  rw [← padBlock32_eq_paddedBlock]

/-- The closed-form regrouping of four bytes as a `Nat` digit identity. -/
private theorem be4_read (w : Nat) (hw : w < 4294967296) :
    ((w / 2 ^ (8 * (3 - 0))) % 256) * 16777216 + ((w / 2 ^ (8 * (3 - 1))) % 256) * 65536
      + ((w / 2 ^ (8 * (3 - 2))) % 256) * 256 + ((w / 2 ^ (8 * (3 - 3))) % 256) = w := by
  norm_num
  omega

/-- **The outer block is the digest re-read.** The first eight words of the closed block
read of `pad (stateToBytes s)` are the state's words: word `i` is `s[i]`, for every
`i < 8`. -/
theorem blockOf_pad_stateToBytes_getD (s : State) (i : Nat) (hi : i < 8) :
    (blockOf (pad (stateToBytes s)) 0).getD i 0 = s.getD i 0 := by
  have hi16 : i < 16 := by omega
  rw [blockOf, vec_ofFn_getD' _ 0 i hi16]
  rw [beWordAt]
  simp only [Nat.mul_zero, Nat.zero_add]
  have hw : (s.getD i 0).toNat < 4294967296 := by
    have h2 := BitVec.toNat_lt_twoPow_of_le (m := 32) (n := 32) (le_refl 32) (x := s.getD i 0)
    rw [show (2 : Nat) ^ 32 = 4294967296 by norm_num] at h2
    exact h2
  have hread : ∀ j : Nat, j < 4 → ((pad (stateToBytes s)).getD (4 * i + j) 0).toNat
      = ((s.getD i 0).toNat / 2 ^ (8 * (3 - j))) % 256 := by
    intro j hj
    have hlt : 4 * i + j < 32 := by omega
    have hp : (pad (stateToBytes s)).getD (4 * i + j) 0
        = (stateToBytes s).getD (4 * i + j) 0 := by
      apply pad_getD_of_lt
      rw [stateToBytes_length]
      omega
    have h1 : (stateToBytes s).getD (4 * i + j) 0 =
        (wordByte (s.getD ((4 * i + j) / 4) 0) ((4 * i + j) % 4) : BitVec 8) := by
      rw [stateToBytes]
      exact list_ofFn_getD _ 0 (4 * i + j) hlt
    have hq : (4 * i + j) / 4 = i := by omega
    have hr : (4 * i + j) % 4 = j := by omega
    rw [hp, h1, hq, hr, natCast8_toNat, wordByte]
    exact Nat.mod_mod _ _
  have hsum : ((pad (stateToBytes s)).getD (4 * i) 0).toNat * 16777216
        + ((pad (stateToBytes s)).getD (4 * i + 1) 0).toNat * 65536
        + ((pad (stateToBytes s)).getD (4 * i + 2) 0).toNat * 256
        + ((pad (stateToBytes s)).getD (4 * i + 3) 0).toNat = (s.getD i 0).toNat := by
    rw [show ((pad (stateToBytes s)).getD (4 * i) 0).toNat
        = ((pad (stateToBytes s)).getD (4 * i + 0) 0).toNat from by rw [Nat.add_zero]]
    rw [hread 0 (by omega), hread 1 (by omega), hread 2 (by omega), hread 3 (by omega)]
    exact be4_read (s.getD i 0).toNat hw
  rw [hsum]
  apply BitVec.eq_of_toNat_eq
  rw [BitVec.toNat_ofNat, show (2 : Nat) ^ 32 = 4294967296 by norm_num,
    Nat.mod_eq_of_lt hw]

/-- The same head read at `paddedBlockBody`, the canonical reader's body. -/
theorem paddedBlockBody_pad_stateToBytes_getD (s : State) (i : Nat) (hi : i < 8) :
    (paddedBlockBody (pad (stateToBytes s)) 0).getD i 0 = s.getD i 0 := by
  rw [← blockOf_eq_paddedBlockBody]
  exact blockOf_pad_stateToBytes_getD s i hi

/-! ## 8 · Fail-closed axiom cone -/

#print axioms paddedBlockBody
#print axioms hash_eq_fold_paddedBlockBody
#print axioms wordByte
#print axioms stateToBytes
#print axioms padBlock32
#print axioms beWordAt
#print axioms blockOf
#print axioms pad_getD_of_lt
#print axioms lengthBlock_length
#print axioms stateToBytes_length
#print axioms beWordAt_eq_fold
#print axioms padBlock32_eq_blockOf
#print axioms blockOf_eq_paddedBlockBody
#print axioms padBlock32_eq_paddedBlock
#print axioms bytesToWords_eq_blockOf
#print axioms hash_one_block_eq_compress_H0
#print axioms hash_one_block_eq_compress_H0_padBlock32
#print axioms pad_len
#print axioms pad_stateToBytes_length
#print axioms outer_compression_count
#print axioms hash_outer_eq_compress_H0
#print axioms sha256d_eq_outer_compress_H0
#print axioms sha256d_eq_outer_compress_padBlock32
#print axioms blockOf_pad_stateToBytes_getD
#print axioms paddedBlockBody_pad_stateToBytes_getD

end AlgebraSha256dCompose
