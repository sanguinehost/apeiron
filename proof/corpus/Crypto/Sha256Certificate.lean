/-
  Crypto/Sha256Certificate.lean

  Stage 0 of the `word_sha256` refinement: the representation layer. The
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

  No `maxHeartbeats` override is taken: stage 0 is closed reduction only.

  Open obligations, next stages: the constant tables (`K`, `H0`) and the
  rotate/shift bodies (stage 1), the 64-word schedule (stage 2), the 64-round
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