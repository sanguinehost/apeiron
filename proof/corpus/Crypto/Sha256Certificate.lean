/-
  Crypto/Sha256Certificate.lean

  Stage 0 of the `word_sha256` refinement is the representation layer; stage 1
  adds the constant tables and the round primitives. The
  extraction (`proof/ApeironWord.lean:24896`, `word_sha256 = crypto.sha256.sha256`)
  carries messages as `Slice Std.U8`, the chaining state and the schedule as
  `Array Std.U32`, the digest as `Array Std.U8 32#usize`; `Crypto/Sha2.lean`
  carries the canonical FIPS 180-4 objects over `List (BitVec 8)` and
  `Vector (BitVec 32)`. The two sides live at different types, so this file
  carries the one-way conversions and their read/length lemmas:

  * `b_u8`, `bytesList`, `sliceBytes`: bytes read as the canonical message byte
    list (elementwise `Std.U8.bv`, the model's own byte view);
  * `w_u32`: an extracted `u32` read as the canonical `Sha2.Word`;
  * `stateView`: `Array Std.U32 8#usize` read as the canonical
    `Sha2.State = Vector (BitVec 32) 8`, with `stateView_get`;
  * `byte32`, `beWord`, `digView`: the `[u8; 32]` digest read back as the
    canonical `Sha2.State` by the big-endian regrouping
    `H[i] = out[4i]·2²⁴ + out[4i+1]·2¹⁶ + out[4i+2]·2⁸ + out[4i+3]`,
    with `digView_get` fixing that grouping;
  * `K_get`, `H0_get`: the two constant tables, read at each of their indices;
  * `rotr_step`, `shr_step3`, `shr_step10`: the extracted `U32::rotate_right` and
    `U32 >>> n` read as the canonical `rotr`/`shrx`;
  * `Sigma0_round`, `Sigma1_round`, `ch_step`, `maj_step`: the four round
    primitives at the rotate/shift/bitwise sequences of the Rust bodies;
  * `WordSha256Denotes`: the headline, carried as a `def` of a `Prop` and
    deliberately not asserted here.

  The views read through `getD` with the zero default, so no read is out of
  bounds at any index the type admits: an `Array α n` has `n.val = n` entries
  and every index below is `< n`. The consequence of that fact for the digest
  regrouping (the four bytes of word `i` occupy `4i … 4i+3`) is discharged at
  each `Fin 8` index below by `omega`.

  Two write lemmas are deliberately not carried here and belong to the loop
  stages: `stateView` read after an `Array.set` of one word (stage 3, the
  compression round's working words) and `digView` read after an `Array.set` of
  one byte (stage 4, the `sha256_loop2` big-endian store). Both go through
  `List.getD`, for which no `set` lemma exists in the pinned toolchain, so each
  is routed through `List.getD_eq_getElem?_getD` and `List.getElem?_set` at
  the stage that needs it.

  No `maxHeartbeats` override is taken: stages 0 and 1 are closed reduction
  plus literal evaluation only.

  Open obligations, next stages: the 64-word schedule (stage 2), the 64-round
  compression (stage 3), padding and the outer block fold (stage 4). The
  `digView` write lemma for the `sha256_loop2` big-endian store — where only
  the word regrouping containing the written byte changes — belongs to stage 4
  with the store loop; it is not asserted here.
-/

import Crypto.Sha2
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Sha2 (Word State)

/-! ## Byte views

`Array Std.U8 n` is a `ListN` wrapper (`Array.val`, `Array.property`) and
`Slice Std.U8` is a length-bounded `ListN` (`Slice.val`, `Slice.len`), so the
canonical byte list is read through `Std.U8.bv` elementwise. One direction
only: no injectivity of the byte view is claimed. -/

/-- The extracted byte read as the canonical byte. -/
def b_u8 (b : Std.U8) : BitVec 8 := b.bv

/-- An extracted byte list read as the canonical message byte list. -/
def bytesList (l : List Std.U8) : List (BitVec 8) := l.map Std.U8.bv

/-- An extracted byte slice read as the canonical message byte list. -/
def sliceBytes (s : Slice Std.U8) : List (BitVec 8) := bytesList s.val

theorem b_u8_eq (b : Std.U8) : b_u8 b = Std.U8.bv b := rfl

theorem bytesList_length (l : List Std.U8) : (bytesList l).length = l.length := by
  simp [bytesList]

/-- Pointwise read: the canonical byte at `i` is the view of the extracted byte
    at `i`. -/
theorem bytesList_getD (l : List Std.U8) (i : Nat) :
    (bytesList l).getD i (Std.U8.bv (0#u8)) = (l.getD i (0#u8)).bv :=
  List.getD_map _ _ _

theorem sliceBytes_length (s : Slice Std.U8) :
    (sliceBytes s).length = s.val.length := by
  simp [sliceBytes, bytesList]

/-- The extracted `Slice::len` is the canonical message length. -/
theorem sliceBytes_len (s : Slice Std.U8) :
    (Std.Slice.len s).val = (sliceBytes s).length := by
  simp [Std.Slice.len_val, sliceBytes, bytesList]

/-- A `32`-byte array read as canonical bytes has length `32`. -/
theorem bytesList_digest_length (a : Array Std.U8 32#usize) :
    (bytesList a.val).length = 32 := by
  have h : (bytesList a.val).length = a.val.length := bytesList_length _
  rw [h]
  have ha : a.val.length = 32 := by simpa using a.property
  exact ha

/-! ## Word and state views -/

/-- The extracted `u32` read as the canonical `Sha2.Word`. -/
def w_u32 (w : Std.U32) : Word := w.bv

/-- Word `i` of an extracted eight-word chaining state, read as the canonical
    word. `i < 8 = s.val.length` holds, so the default is never taken. -/
def stateGet (s : Array Std.U32 8#usize) (i : Nat) : Word :=
  (s.val.getD i (0#u32)).bv

/-- The extracted eight-word chaining state read as the canonical `Sha2.State`. -/
def stateView (s : Array Std.U32 8#usize) : State :=
  Vector.ofFn fun i : Fin 8 => stateGet s i.val

/-- Reading the state through `stateView` is reading the word's bit vector. -/
theorem stateView_get (s : Array Std.U32 8#usize) (i : Fin 8) :
    (stateView s).get i = stateGet s i.val := by
  show (stateView s)[i.val] = stateGet s i.val
  simp only [stateView, Vector.getElem_ofFn]

/-! ## Digest view

The extracted digest is an `Array Std.U8 32#usize` written by the
`sha256_loop2` store loop through `U32::to_be_bytes`; the canonical object is a
`Sha2.State`. The regrouping is therefore big-endian, and is fixed once, here. -/

/-- Byte `j` of a `32`-byte array as the canonical byte. -/
def byte32 (a : Array Std.U8 32#usize) (j : Nat) : BitVec 8 :=
  (a.val.getD j (0#u8)).bv

/-- Word `i` of the big-endian regrouping of the digest bytes: the inverse of
    `U32::to_be_bytes`. -/
def beWord (a : Array Std.U8 32#usize) (i : Fin 8) : Word :=
  BitVec.ofNat 32
    ((byte32 a (4 * i.val)).toNat * 16777216 +
      (byte32 a (4 * i.val + 1)).toNat * 65536 +
      (byte32 a (4 * i.val + 2)).toNat * 256 +
      (byte32 a (4 * i.val + 3)).toNat)

/-- The extracted `32`-byte digest read as the canonical `Sha2.State`: word `i`
    is the big-endian regrouping of bytes `4i … 4i+3`. -/
def digView (a : Array Std.U8 32#usize) : State :=
  Vector.ofFn fun i : Fin 8 => beWord a i

/-- Reading the digest through `digView` is the big-endian regrouping. -/
theorem digView_get (a : Array Std.U8 32#usize) (i : Fin 8) :
    (digView a).get i = beWord a i := by
  show (digView a)[i.val] = beWord a i
  simp only [digView, Vector.getElem_ofFn]

/-! ## Constant tables

`sha256.K` and `sha256.H0` are `Array.make` literals (`src/sha256.rs:11-26`); the
canonical tables are `Sha2.K` and `Sha2.H0` (`Sha2.lean:34-47`). Each index is a
literal equality, discharged by unfolding the array and evaluating the word. -/

/-- The extracted round constant `K[t]`, read as the canonical word, is `Sha2.K[t]`,
    for every `t < 64`. -/
theorem K_get (t : Fin 64) : (sha256.K[t.val]).bv = Apeiron.Crypto.Sha2.K.get t := by
  fin_cases t <;> (simp only [sha256.K, Array.make_val]; decide)

/-- The extracted initial hash value `H0[i]`, read as the canonical word, is
    `Sha2.H0[i]`, for every `i < 8`. -/
theorem H0_get (i : Fin 8) : (sha256.H0[i.val]).bv = Apeiron.Crypto.Sha2.H0.get i := by
  fin_cases i <;> (simp only [sha256.H0, Array.make_val]; decide)

/-! ## Rotate and shift bridges

`U32::rotate_right n` is modelled as `BitVec.rotateRight` on the word's bit vector
(`Aeneas/Std/Scalar/Rotate.lean:31`), and `U32 >>> n` as `BitVec.ushiftRight` there;
the canonical `rotr`/`shrx` are the `(x >>> n) ||| (x <<< (32 - n))` and `x >>> n`
forms. The bridge therefore holds at every shift `0 < n < 32`, which covers the ten
rotation amounts and the two shift amounts that `src/sha256.rs` uses. -/

/-- The extracted `rotate_right` at a shift strictly inside the word width is the
    canonical `rotr` at the same amount. -/
theorem rotr_step (x : Std.U32) {n : Std.U32} (h0 : 0 < n.val) (hn : n.val < 32) :
    (core.num.U32.rotate_right x n).bv = Apeiron.Crypto.Sha2.rotr x.bv n.val := by
  have h32 : UScalarTy.U32.numBits = 32 := by rfl
  simp only [core.num.U32.rotate_right, UScalar.rotate_right,
    Apeiron.Crypto.Sha2.rotr, BitVec.rotateRight, BitVec.rotateRightAux, h32]
  rw [Nat.mod_eq_of_lt (Nat.lt_of_lt_of_le hn (by decide)),
    Nat.mod_eq_of_lt (Nat.sub_lt (n := 32) (m := n.val) (by decide) h0)]

/-- The extracted logical `>>> 3` is the canonical `SHR^3`. -/
theorem shr_step3 (x : Std.U32) :
    ∃ z : Std.U32, x >>> 3#i32 = ok z ∧ z.bv = Apeiron.Crypto.Sha2.shrx x.bv 3 := by
  obtain ⟨z, hz, hok⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_IScalar_spec x 3#i32 (by decide) (by decide))
  refine ⟨z, hz, ?_⟩
  unfold Apeiron.Crypto.Sha2.shrx
  change z.bv = x.bv >>> (3 % 32)
  rw [hok.2]
  have h3 : (3#i32 : Std.I32).toNat = 3 := by decide
  rw [h3]

/-- The extracted logical `>>> 10` is the canonical `SHR^10`. -/
theorem shr_step10 (x : Std.U32) :
    ∃ z : Std.U32, x >>> 10#i32 = ok z ∧ z.bv = Apeiron.Crypto.Sha2.shrx x.bv 10 := by
  obtain ⟨z, hz, hok⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_IScalar_spec x 10#i32 (by decide) (by decide))
  refine ⟨z, hz, ?_⟩
  unfold Apeiron.Crypto.Sha2.shrx
  change z.bv = x.bv >>> (10 % 32)
  rw [hok.2]
  have h3 : (10#i32 : Std.I32).toNat = 10 := by decide
  rw [h3]

/-! ## Round functions

Each lemma is the extracted round body read at its bit vectors, at the exact
sequence of rotate and shift amounts of `src/sha256.rs:43-51` and `:63-84`. -/

/-- `Σ₁(e) = ROTR^6 e ⊕ ROTR^11 e ⊕ ROTR^25 e` at the extracted rotate sequence. -/
theorem Sigma1_round (e : Std.U32) :
    ((core.num.U32.rotate_right e 6#u32).bv ^^^
      (core.num.U32.rotate_right e 11#u32).bv ^^^
      (core.num.U32.rotate_right e 25#u32).bv) =
      Apeiron.Crypto.Sha2.Sigma1 e.bv := by
  simp only [Apeiron.Crypto.Sha2.Sigma1]
  rw [rotr_step _ (by decide) (by decide), rotr_step _ (by decide) (by decide),
    rotr_step _ (by decide) (by decide)]
  norm_num

/-- `Σ₀(a) = ROTR^2 a ⊕ ROTR^13 a ⊕ ROTR^22 a` at the extracted rotate sequence. -/
theorem Sigma0_round (a : Std.U32) :
    ((core.num.U32.rotate_right a 2#u32).bv ^^^
      (core.num.U32.rotate_right a 13#u32).bv ^^^
      (core.num.U32.rotate_right a 22#u32).bv) =
      Apeiron.Crypto.Sha2.Sigma0 a.bv := by
  simp only [Apeiron.Crypto.Sha2.Sigma0]
  rw [rotr_step _ (by decide) (by decide), rotr_step _ (by decide) (by decide),
    rotr_step _ (by decide) (by decide)]
  norm_num

/-- `Ch(e, f, g) = (e ∧ f) ⊕ (¬e ∧ g)` at the extracted bitwise sequence. -/
theorem ch_step (e f g : Std.U32) :
    ((e &&& f).bv ^^^ ((~~~e).bv &&& g.bv)) =
      Apeiron.Crypto.Sha2.choose e.bv f.bv g.bv := by
  simp only [Apeiron.Crypto.Sha2.choose, UScalar.bv_and, UScalar.bv_xor, UScalar.bv_not]

/-- `Maj(a, b, c) = (a ∧ b) ⊕ (a ∧ c) ⊕ (b ∧ c)` at the extracted bitwise sequence. -/
theorem maj_step (a b c : Std.U32) :
    ((a &&& b).bv ^^^ (a &&& c).bv ^^^ (b &&& c).bv) =
      Apeiron.Crypto.Sha2.majority a.bv b.bv c.bv := by
  simp only [Apeiron.Crypto.Sha2.majority, UScalar.bv_and, UScalar.bv_xor]

/-! ## Headline -/

/-- The refinement target: on every input, `word_sha256` returns a digest array
    `out` whose canonical `Sha2.State` view is
    `Apeiron.Crypto.Sha2.hash (sliceBytes msg)`, that is `Apeiron.Crypto.Sha2.sha256`
    of the byte view of the message (`sha256` is `hash` after the byte view;
    stage 4 carries that lemma). Carried, not asserted: stages 1-4 supply the
    constant, schedule, compression and padding lemmas it is built from. -/
def WordSha256Denotes (msg : Slice Std.U8) : Prop :=
  ∃ out : Array Std.U8 32#usize,
    word_sha256 msg = Result.ok out ∧ digView out = Apeiron.Crypto.Sha2.hash (sliceBytes msg)

end apeiron_word