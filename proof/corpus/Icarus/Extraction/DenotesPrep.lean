/-
`Icarus.Extraction.DenotesPrep` — the conditional denotes link for the SHA-256d composition route
through the extracted `word_sha256`.

Port of the staging receipt AR-73 (`work/proofs/AlgebraSha256dDenotes.lean`, sha256
`852ea73ad6ae12cb6d8828368edb6430c57b5e27b8e7ee3392f3e36b3de67c65`), at its existing grade
(kernel-checked, conditional). This is the W5 extraction-obligation prep of
the icarus-sha256-rooting note.

The extracted `word_sha256d` is absent from `proof/ApeironWord.lean` (`grep -c word_sha256d`
= 0; the file hashes to `855528a8a05b76328c99ad1ba1dd2e36458e5fe8aaccdc2c2596d7a58899d27c`),
so the double application is stated at the composition the regen root would run:
`wordSha256dStrength` applies `apeiron_word.word_sha256` twice, the second time to the slice
view of the first application's 32-byte digest — the body `pub fn word_sha256d` of
`proof/extract/lib.rs:1109` calling the real `src/sha256.rs` `sha256d`. The named hypothesis
`WordSha256DenotesAll` is the certificate's `apeiron_word.WordSha256Denotes`
(`Crypto/Sha256Certificate.lean:908`) universally quantified: carried, not asserted. This file
makes no freshness claim about the extraction and no security, cost or performance claim.

The canonical-side definitions `wordByte`, `stateToBytes`, `paddedBlockBody`,
`stateToBytes_length`, `pad_len`, `pad_stateToBytes_length`, `hash_one_block_eq_compress_H0`,
`hash_outer_eq_compress_H0` and `sha256d_eq_outer_compress_H0` were re-declared verbatim from
AR-40 (`work/proofs/AlgebraSha256dCompose.lean`, sha256
`d7eef60f59722007af296e942802e243b1fd1fff7a82ae3ac0aa888789fa0dd0`, read-only) while the W3 port
of AR-40 was absent from this tree. W3 has since landed at
`Icarus/Composition/AlgebraSha256dCompose.lean`, byte-identical to the staging source (same sha256
`d7eef60f59722007af296e942802e243b1fd1fff7a82ae3ac0aa888789fa0dd0`), so those re-declarations are
deleted here and replaced by an import of that module; AR-40 is now the single authority for the
canonical chain, and this file's `#print axioms` rows for those names read its theorems. The
conclusion of `word_sha256d_strength_eq_outer_compress_H0` is exactly AR-40's right-hand side at
the message's byte view; `word_sha256d_strength_denotes` reaches AR-40's left-hand side from the
composed extracted run.

Named obligations (detail in the icarus-regen-manifest note):
  AR-73-R1 (open, regen window): an exclusive `proof/regen.sh` run must emit
    `apeiron_word.word_sha256d : Slice Std.U8 → Result (Array Std.U8 32#usize)` wrapping the
    real `sha256.sha256d`, whose Lean body is definitionally the two-call composition
    `wordSha256dStrength`; checked post-regen by `rfl` (plus `#print axioms`, `sorryAx` = 0).
    Then the conditional theorems here instantiate at the named root.
  AR-73-R2 (open, certificate): `WordSha256DenotesAll` itself is the certificate owner's
    stage 1-4 obligation and is **not** discharged by the regen.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate
import Icarus.Composition.AlgebraSha256dCompose

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace AlgebraSha256dDenotes

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Sha2 (Word State Block H0 lengthBlock compress pad)

/-! ## 1 · The canonical-side definitions of AR-40, imported

`paddedBlockBody`, `hash_eq_fold_paddedBlockBody`, `wordByte`, `stateToBytes`,
`lengthBlock_length`, `stateToBytes_length`, `pad_len`, `pad_stateToBytes_length`,
`hash_one_block_eq_compress_H0`, `hash_outer_eq_compress_H0` and `sha256d_eq_outer_compress_H0`
are not declared in this file. They are `Icarus.Composition.AlgebraSha256dCompose`'s (AR-40,
sha256 `d7eef60f59722007af296e942802e243b1fd1fff7a82ae3ac0aa888789fa0dd0`), opened below by
name so that no unqualified use of a canonical-side identifier can resolve to a local
re-declaration. -/
open AlgebraSha256dCompose (paddedBlockBody hash_eq_fold_paddedBlockBody wordByte stateToBytes
  lengthBlock_length stateToBytes_length pad_len pad_stateToBytes_length
  hash_one_block_eq_compress_H0 hash_outer_eq_compress_H0 sha256d_eq_outer_compress_H0)

/-! ## 2 · The named hypothesis, the composition, and the predicate it would feed

`wordSha256dStrength` is the composition of two `apeiron_word.word_sha256` applications at the
extracted types. Both applications are through `word_sha256`: the inner one on `msg`, the outer
one on the slice view of the inner digest. This is the body shape the regen root
`apeiron_word.word_sha256d` must have (AR-73-R1). -/

/-- The named hypothesis: the extracted `word_sha256` denotes canonical SHA-256 on every input
slice. `apeiron_word.WordSha256Denotes` is a definition, deliberately not asserted
(`Crypto/Sha256Certificate.lean:902-911`); this is its universal closure, carried. -/
def WordSha256DenotesAll : Prop :=
  ∀ msg : Slice Std.U8, apeiron_word.WordSha256Denotes msg

/-- The `word_sha256d`-strength composition: apply `word_sha256` to `msg`, then to the slice
view of the 32-byte digest it returned. Mirrors the real `sha256d` body
(`proof/extract/lib.rs:1109`), which `ApeironWord.lean` does not carry. -/
def wordSha256dStrength (msg : Slice Std.U8) : Result (Array Std.U8 32#usize) := do
  let inner ← apeiron_word.word_sha256 msg
  apeiron_word.word_sha256 (Array.to_slice inner)

/-- The predicate the composition route would carry: the composed run returns a digest whose
`digView` is the canonical double application `sha256d = sha256 ∘ sha256` of the message's byte
view. The `word_sha256d`-strength analogue of `apeiron_word.WordSha256Denotes`. -/
def WordSha256dStrengthDenotes (msg : Slice Std.U8) : Prop :=
  ∃ out : Array Std.U8 32#usize,
    wordSha256dStrength msg = Result.ok out ∧
    apeiron_word.digView out
      = Apeiron.Crypto.Sha2.hash
          (stateToBytes (Apeiron.Crypto.Sha2.hash (apeiron_word.sliceBytes msg)))

/-! ## 3 · The byte-view bridge

The composition's outer application receives the extracted digest array through
`Array.to_slice`; its `sliceBytes` must be the canonical 32-byte big-endian flattening of the
digest state. That is a closed `Nat`/`BitVec` identity between the certificate's `digView`
(`Crypto/Sha256Certificate.lean:139-160`) and AR-40's `stateToBytes`, proved here for every
array — not carried as a premise. -/

private theorem byteToNat_lt (a : Array Std.U8 32#usize) (j : Nat) :
    (apeiron_word.byte32 a j).toNat < 256 := by
  have h := (apeiron_word.byte32 a j).isLt
  omega

/-- The Horner digit identity: for four bytes each `< 256`, the reassembled 32-bit word's
`2^24` / `2^16` / `2^8` / `1` digits are the four bytes. -/
private theorem horner_digit (b0 b1 b2 b3 : Nat) (h0 : b0 < 256) (h1 : b1 < 256)
    (h2 : b2 < 256) (h3 : b3 < 256) :
    (b0 * 16777216 + b1 * 65536 + b2 * 256 + b3) / 2 ^ 24 % 256 = b0 ∧
    (b0 * 16777216 + b1 * 65536 + b2 * 256 + b3) / 2 ^ 16 % 256 = b1 ∧
    (b0 * 16777216 + b1 * 65536 + b2 * 256 + b3) / 2 ^ 8 % 256 = b2 ∧
    (b0 * 16777216 + b1 * 65536 + b2 * 256 + b3) / 1 % 256 = b3 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have hlt : b1 * 65536 + b2 * 256 + b3 < 2 ^ 24 := by omega
    rw [show b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
        = (b1 * 65536 + b2 * 256 + b3) + b0 * 2 ^ 24 by ring]
    rw [Nat.add_mul_div_right _ _ (by norm_num : (0 : Nat) < 2 ^ 24),
      Nat.div_eq_of_lt hlt, Nat.zero_add, Nat.mod_eq_of_lt h0]
  · have hlt : b2 * 256 + b3 < 2 ^ 16 := by omega
    rw [show b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
        = (b2 * 256 + b3) + (b0 * 256 + b1) * 2 ^ 16 by ring]
    rw [Nat.add_mul_div_right _ _ (by norm_num : (0 : Nat) < 2 ^ 16),
      Nat.div_eq_of_lt hlt, Nat.zero_add]
    rw [show b0 * 256 + b1 = b1 + b0 * 256 by ring, Nat.add_mul_mod_self_right,
      Nat.mod_eq_of_lt h1]
  · have hlt : b3 < 2 ^ 8 := by omega
    rw [show b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
        = b3 + (b0 * 65536 + b1 * 256 + b2) * 2 ^ 8 by ring]
    rw [Nat.add_mul_div_right _ _ (by norm_num : (0 : Nat) < 2 ^ 8),
      Nat.div_eq_of_lt hlt, Nat.zero_add]
    rw [show b0 * 65536 + b1 * 256 + b2 = b2 + (b0 * 256 + b1) * 256 by ring,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h2]
  · rw [Nat.div_one]
    rw [show b0 * 16777216 + b1 * 65536 + b2 * 256 + b3
        = b3 + (b0 * 65536 + b1 * 256 + b2) * 256 by ring,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt h3]

/-- Digit `j` of the big-endian regrouping `beWord` is byte `4q + j`, for every `q < 8` and
`j < 4`. -/
private theorem byte32_eq_wordByte_beWord (a : Array Std.U8 32#usize) (q : Nat) (j : Fin 4)
    (hq : q < 8) :
    (apeiron_word.byte32 a (4 * q + j.val)).toNat
      = wordByte (apeiron_word.beWord a ⟨q, hq⟩) j.val := by
  obtain ⟨hd0, hd1, hd2, hd3⟩ :=
    horner_digit (apeiron_word.byte32 a (4 * q)).toNat
      (apeiron_word.byte32 a (4 * q + 1)).toNat
      (apeiron_word.byte32 a (4 * q + 2)).toNat
      (apeiron_word.byte32 a (4 * q + 3)).toNat
      (byteToNat_lt a (4 * q)) (byteToNat_lt a (4 * q + 1))
      (byteToNat_lt a (4 * q + 2)) (byteToNat_lt a (4 * q + 3))
  have hS : (apeiron_word.byte32 a (4 * q)).toNat * 16777216
      + (apeiron_word.byte32 a (4 * q + 1)).toNat * 65536
      + (apeiron_word.byte32 a (4 * q + 2)).toNat * 256
      + (apeiron_word.byte32 a (4 * q + 3)).toNat < 2 ^ 32 := by
    have h0 := byteToNat_lt a (4 * q)
    have h1 := byteToNat_lt a (4 * q + 1)
    have h2 := byteToNat_lt a (4 * q + 2)
    have h3 := byteToNat_lt a (4 * q + 3)
    omega
  unfold wordByte apeiron_word.beWord
  simp only [Fin.val_mk]
  rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt hS]
  fin_cases j
  · simpa using hd0.symm
  · simpa using hd1.symm
  · simpa using hd2.symm
  · simpa using hd3.symm

/-- The state-`wordByte` read of `digView` at index `i` is byte `i` of the array. -/
private theorem byte32_getD_eq_wordByte (a : Array Std.U8 32#usize) (i : Nat)
    (hq : i / 4 < 8) :
    (a.val.getD i (0#u8)).bv.toNat
      = wordByte (apeiron_word.beWord a ⟨i / 4, hq⟩) (i % 4) := by
  have hj : i % 4 < 4 := Nat.mod_lt _ (by norm_num)
  have hdiv : 4 * (i / 4) + i % 4 = i := Nat.div_add_mod i 4
  have key := byte32_eq_wordByte_beWord a (i / 4) ⟨i % 4, hj⟩ hq
  change (apeiron_word.byte32 a i).toNat = _
  conv_lhs => rw [show i = 4 * (i / 4) + i % 4 from hdiv.symm]
  exact key

/-- Two lists of the same length that agree on every in-range `getD` read are equal. -/
private theorem list_eq_of_getD {α : Type} [Inhabited α] {l₁ l₂ : List α} (d : α)
    (h : l₁.length = l₂.length) (hd : ∀ i, i < l₁.length → l₁.getD i d = l₂.getD i d) :
    l₁ = l₂ := by
  apply List.ext_getElem h
  intro i h1 h2
  rw [List.getElem_eq_getD (h := h1) (fallback := d),
    List.getElem_eq_getD (h := h2) (fallback := d)]
  exact hd i h1

private theorem list_ofFn_getD {α : Type} [Inhabited α] {n : Nat} (f : Fin n → α)
    (d : α) (i : Nat) (hi : i < n) : (List.ofFn f).getD i d = f ⟨i, hi⟩ := by
  simp [List.getD, List.getElem?_ofFn, hi]

/-- The `stateToBytes` read at an in-range index, in the `wordByte` spelling. -/
private theorem stateToBytes_getD (s : State) (i : Nat) (hi : i < 32) (d : BitVec 8) :
    (stateToBytes s).getD i d = (wordByte (s.getD (i / 4) 0) (i % 4) : BitVec 8) := by
  unfold stateToBytes
  exact list_ofFn_getD (fun k : Fin 32 =>
    (wordByte (s.getD (k / 4) 0) (k % 4) : BitVec 8)) d i hi

/-- **The bridge.** The raw byte list of an extracted 32-byte digest array is the canonical
big-endian flattening of its `digView`. Kernel arithmetic only; no premise. -/
theorem bytesList_eq_stateToBytes_digView (a : Array Std.U8 32#usize) :
    apeiron_word.bytesList a.val = stateToBytes (apeiron_word.digView a) := by
  have ha : a.val.length = 32 := by simpa using a.property
  refine list_eq_of_getD (d := Std.U8.bv (0#u8)) ?_ ?_
  · rw [apeiron_word.bytesList_length, ha, stateToBytes_length]
  · intro i hi
    have hi32 : i < 32 := by
      rw [apeiron_word.bytesList_length, ha] at hi
      exact hi
    have hq : i / 4 < 8 := by omega
    rw [apeiron_word.bytesList_getD]
    rw [stateToBytes_getD (apeiron_word.digView a) i hi32 (Std.U8.bv (0#u8))]
    have hdig : (apeiron_word.digView a).getD (i / 4) 0
        = apeiron_word.beWord a ⟨i / 4, hq⟩ := by
      unfold apeiron_word.digView
      rw [apeiron_word.ofFn_getD' (fun j : Fin 8 => apeiron_word.beWord a j) (i / 4) hq
        (0 : Word)]
    rw [hdig]
    change (a.val.getD i (0#u8)).bv
      = BitVec.ofNat 8 (wordByte (apeiron_word.beWord a ⟨i / 4, hq⟩) (i % 4))
    apply BitVec.eq_of_toNat_eq
    rw [BitVec.toNat_ofNat, show (2 : Nat) ^ 8 = 256 by norm_num]
    unfold wordByte
    rw [Nat.mod_mod]
    exact byte32_getD_eq_wordByte a i hq

/-- The bridge at the composition's actual input: the slice view of a digest array reads as
AR-40's `stateToBytes` of its `digView`. -/
theorem sliceBytes_to_slice_eq_stateToBytes_digView (a : Array Std.U8 32#usize) :
    apeiron_word.sliceBytes (Array.to_slice a) = stateToBytes (apeiron_word.digView a) := by
  have hval : (Array.to_slice a).val = a.val := by
    simp only [Array.to_slice, Slice.from_val]
  show apeiron_word.bytesList (Array.to_slice a).val = stateToBytes (apeiron_word.digView a)
  rw [hval]
  exact bytesList_eq_stateToBytes_digView a

/-! ## 4 · The conditional link

`word_sha256d_strength_denotes` applies the named hypothesis at both inputs — the inner one at
`msg`, the outer one at `Array.to_slice inner` — and lands on `sha256d = sha256 ∘ sha256`.
`word_sha256d_strength_eq_outer_compress_H0` rewrites that with AR-40's imported
`sha256d_eq_outer_compress_H0`, so its conclusion is exactly AR-40's right-hand side. Both are
conditional: no certificate stage is asserted here. -/

/-- **The conditional link.** If `word_sha256` denotes canonical SHA-256 on every slice, then
the two-application composition denotes the canonical double application of the message's byte
view. -/
theorem word_sha256d_strength_denotes (h : WordSha256DenotesAll) (msg : Slice Std.U8) :
    WordSha256dStrengthDenotes msg := by
  obtain ⟨inner, hinner, hinnerview⟩ := h msg
  obtain ⟨outer, houter, houterview⟩ := h (Array.to_slice inner)
  refine ⟨outer, ?_, ?_⟩
  · simp only [wordSha256dStrength, hinner, bind_ok, bind_tc_ok]
    exact houter
  · rw [houterview, sliceBytes_to_slice_eq_stateToBytes_digView inner, hinnerview]

/-- **The conditional link at AR-40's conclusion.** Under the same named hypothesis, the
composed extracted run's digest view is exactly the right-hand side of AR-40's
`sha256d_eq_outer_compress_H0` at `m = sliceBytes msg` (the second application as one
compression from `H0` of the padded digest store). -/
theorem word_sha256d_strength_eq_outer_compress_H0 (h : WordSha256DenotesAll)
    (msg : Slice Std.U8) :
    ∃ out : Array Std.U8 32#usize,
      wordSha256dStrength msg = Result.ok out ∧
      apeiron_word.digView out
        = compress H0
            (paddedBlockBody
              (pad (stateToBytes (Apeiron.Crypto.Sha2.hash (apeiron_word.sliceBytes msg))))
              0) := by
  obtain ⟨out, hout, hview⟩ := word_sha256d_strength_denotes h msg
  exact ⟨out, hout,
    hview.trans (sha256d_eq_outer_compress_H0 (apeiron_word.sliceBytes msg))⟩

/-! ## 5 · Axiom cone

The eleven rows for the canonical chain are AR-40's theorems, read here through the import; the
seven rows below them are this file's own statements. -/

#print axioms paddedBlockBody
#print axioms hash_eq_fold_paddedBlockBody
#print axioms wordByte
#print axioms stateToBytes
#print axioms stateToBytes_length
#print axioms WordSha256DenotesAll
#print axioms wordSha256dStrength
#print axioms WordSha256dStrengthDenotes
#print axioms lengthBlock_length
#print axioms pad_len
#print axioms pad_stateToBytes_length
#print axioms hash_one_block_eq_compress_H0
#print axioms hash_outer_eq_compress_H0
#print axioms sha256d_eq_outer_compress_H0
#print axioms bytesList_eq_stateToBytes_digView
#print axioms sliceBytes_to_slice_eq_stateToBytes_digView
#print axioms word_sha256d_strength_denotes
#print axioms word_sha256d_strength_eq_outer_compress_H0

end AlgebraSha256dDenotes