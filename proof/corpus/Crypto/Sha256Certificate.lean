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
open Apeiron.Crypto.Sha2 (Word State Block)

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

/-! ## The schedule view

`Array Std.U32 64#usize` is a `ListN` wrapper, so the canonical
`Vector Word 64` schedule is read through `getD` with the zero default: an
`Array α n` has `n.val = n` entries, so no read at an index below `n` takes the
default. `wView` is that read as a `Vector`, and `wGet_set` is the write side,
routed through `List.getD_eq_getElem?_getD` and `List.getElem?_set` (the pinned
toolchain has no `List.getD_set`).

The canonical `Sha2.schedule` is a `List.range 48 |>.foldl` over a
`Vector.ofFn` that rewrites a single row, so it is read here through three
spellings of the same fold — `schedStart`, `schedStep`, `schedN` — with
`schedule_eq_schedN` the definitional link. `schedN_eq_rowVec` is the general
induction over the forty-eight extension steps: the step case reads only rows
`t-2`, `t-7`, `t-15` and `t-16`, which all lie strictly below the row being
written, so the loop invariant closes without enumerating the steps.
`schedule_getD` reads the canonical schedule at any row. -/

/-- The canonical schedule row `W[j]`, by strong recursion on the row index:
    the sixteen block words for `j < 16`, and `σ₁(W[j-2]) + W[j-7] + σ₀(W[j-15])
    + W[j-16]` above, FIPS 180-4 §6.2.2 step 1. -/
def schedRow (b : Block) (j : Nat) : Word :=
  if _h : j < 16 then b.getD j 0
  else Apeiron.Crypto.Sha2.sigma1 (schedRow b (j - 2)) + schedRow b (j - 7)
       + Apeiron.Crypto.Sha2.sigma0 (schedRow b (j - 15)) + schedRow b (j - 16)
termination_by j
decreasing_by
  all_goals have _hj : 16 ≤ j := by omega
  all_goals omega

/-- Above row sixteen the canonical row is the FIPS step-1 recurrence. -/
theorem schedRow_step (b : Block) (t : Nat) (ht : 16 ≤ t) :
    schedRow b t = Apeiron.Crypto.Sha2.sigma1 (schedRow b (t - 2)) + schedRow b (t - 7)
      + Apeiron.Crypto.Sha2.sigma0 (schedRow b (t - 15)) + schedRow b (t - 16) := by
  rw [schedRow, dif_neg (by omega)]

/-- The schedule before any extension step: the sixteen block words followed by
    forty-eight zeros. -/
def schedStart (b : Block) : Vector Word 64 :=
  (Vector.ofFn fun i : Fin 16 => b.getD i 0) ++ Vector.replicate 48 (0 : Word)

/-- One `Sha2.schedule` fold step: rewrite row `16 + k` with the FIPS recurrence
    and leave every other row alone. -/
def schedStep (acc : Vector Word 64) (k : Nat) : Vector Word 64 :=
  let t := 16 + k
  let wt := Apeiron.Crypto.Sha2.sigma1 (acc.getD (t - 2) 0) + acc.getD (t - 7) 0
            + Apeiron.Crypto.Sha2.sigma0 (acc.getD (t - 15) 0) + acc.getD (t - 16) 0
  Vector.ofFn fun i : Fin 64 => if i.val == t then wt else acc.getD i 0

/-- The canonical fold, spelled out of `Sha2.schedule`. -/
def schedN (b : Block) (k : Nat) : Vector Word 64 :=
  (List.range k).foldl (fun acc k => schedStep acc k) (schedStart b)

/-- The schedule after `k` extension steps, as the vector the loop reaches. -/
def rowVec (b : Block) (k : Nat) : Vector Word 64 :=
  Vector.ofFn fun j : Fin 64 => if j.val < 16 + k then schedRow b j.val else 0

/-- Two vectors of the same length that agree through `getD` are equal. -/
theorem vec_getD_eq {α : Type} {n : Nat} [Inhabited α]
    (u v : Vector α n) (d : α) (h : ∀ i, i < n → u.getD i d = v.getD i d) : u = v := by
  refine Vector.ext fun i hi => ?_
  rw [show u[i] = u.getD i d by simp [Vector.getD, hi],
      show v[i] = v.getD i d by simp [Vector.getD, hi]]
  exact h i hi

/-- Reading a `Vector.ofFn` through `getD` is reading the function. -/
theorem ofFn_getD (f : Fin 64 → Word) (i : Nat) (hi : i < 64) :
    (Vector.ofFn f).getD i 0 = f ⟨i, hi⟩ := by
  simp [Vector.getD, hi]

/-- `Sha2.schedule` is the forty-eight-step fold, by definition. -/
theorem schedule_eq_schedN (b : Block) :
    Apeiron.Crypto.Sha2.schedule b = schedN b 48 := rfl

/-- One fold step rewrites exactly row `16 + k` and leaves the rest alone. -/
theorem schedStep_getD (acc : Vector Word 64) (k : Nat) (j : Nat) (hj : j < 64) :
    (schedStep acc k).getD j 0 =
      if j = 16 + k then
        Apeiron.Crypto.Sha2.sigma1 (acc.getD (16 + k - 2) 0) + acc.getD (16 + k - 7) 0
          + Apeiron.Crypto.Sha2.sigma0 (acc.getD (16 + k - 15) 0) + acc.getD (16 + k - 16) 0
      else acc.getD j 0 := by
  simp [schedStep, Vector.getD, hj]

/-- One more fold step is one more `schedStep` at the previous count. -/
theorem schedN_succ (b : Block) (k : Nat) :
    schedN b (k + 1) = schedStep (schedN b k) k := by
  show (List.range (k + 1)).foldl (fun acc k => schedStep acc k) (schedStart b)
    = schedStep ((List.range k).foldl (fun acc k => schedStep acc k) (schedStart b)) k
  rw [List.range_succ, List.foldl_append]
  rfl

/-- Before any extension step the schedule holds the block words below row
    sixteen and zeros above. -/
theorem schedStart_getD (b : Block) (j : Nat) (hj : j < 64) :
    (schedStart b).getD j 0 = if j < 16 then b.getD j 0 else 0 := by
  have h : (schedStart b).getD j 0 = (schedStart b)[j] := by
    simp [Vector.getD, hj]
  rw [h]
  show ((Vector.ofFn fun i : Fin 16 => b.getD i 0) ++ Vector.replicate 48 (0 : Word))[j] = _
  by_cases hj16 : j < 16
  · rw [Vector.getElem_append_left (by omega)]
    show (Vector.ofFn fun i : Fin 16 => b.getD i 0)[j] = _
    simp [Vector.getElem_ofFn hj16, hj16]
  · rw [Vector.getElem_append_right (by omega) (by omega)]
    show (Vector.replicate 48 (0 : Word))[j - 16] = _
    simp [hj16]

/-- `rowVec b k` reads as the canonical row below `16 + k` and zero above. -/
theorem rowVec_getD (b : Block) (k : Nat) (i : Nat) (hi : i < 64) :
    (rowVec b k).getD i 0 = if i < 16 + k then schedRow b i else 0 :=
  ofFn_getD _ i hi

/-- `schedN b k` is exactly the vector whose rows below `16 + k` are the
    canonical FIPS 180-4 §6.2.2 step-1 values. This is the general induction
    over the forty-eight extension steps: the step case needs only rows
    `t-2`, `t-7`, `t-15` and `t-16`, all strictly below the row being written,
    so no step is ever enumerated. -/
theorem schedN_eq_rowVec (b : Block) (k : Nat) (hk : k ≤ 48) :
    schedN b k = rowVec b k := by
  induction k with
  | zero =>
      refine vec_getD_eq _ _ 0 fun i hi => ?_
      rw [show schedN b 0 = schedStart b from rfl, schedStart_getD b i hi]
      rw [rowVec_getD b 0 i hi]
      by_cases hj16 : i < 16
      · rw [if_pos hj16, if_pos (by omega), schedRow, dif_pos hj16]
      · rw [if_neg hj16, if_neg (by omega)]
  | succ k ih =>
      have hk48 : k ≤ 48 := by omega
      refine vec_getD_eq _ _ 0 fun i hi => ?_
      rw [schedN_succ, schedStep_getD _ k i hi, rowVec_getD b (k + 1) i hi]
      have h2 : 16 + k - 2 < 16 + k := by omega
      have h7 : 16 + k - 7 < 16 + k := by omega
      have h15 : 16 + k - 15 < 16 + k := by omega
      have h16 : 16 + k - 16 < 16 + k := by omega
      by_cases heq : i = 16 + k
      · subst heq
        rw [if_pos rfl, if_pos (by omega)]
        rw [ih hk48, rowVec_getD _ _ _ (by omega), if_pos h2,
            rowVec_getD _ _ _ (by omega), if_pos h7,
            rowVec_getD _ _ _ (by omega), if_pos h15,
            rowVec_getD _ _ _ (by omega), if_pos h16]
        exact (schedRow_step b (16 + k) (by omega)).symm
      · rw [if_neg heq, ih hk48, rowVec_getD b k i hi]
        by_cases hc : i < 16 + k
        · rw [if_pos hc, if_pos (by omega)]
        · rw [if_neg hc, if_neg (by omega)]

/-- The canonical schedule, read at any row. -/
theorem schedule_getD (b : Block) (i : Nat) (hi : i < 64) :
    (Apeiron.Crypto.Sha2.schedule b).getD i 0 = schedRow b i := by
  rw [schedule_eq_schedN, schedN_eq_rowVec b 48 (by omega), rowVec_getD b 48 i hi,
      if_pos (by omega)]

/-! ## Schedule reads and writes -/

/-- Row `j` of an extracted sixty-four word schedule, as the canonical word. -/
def wGet (w : Array Std.U32 64#usize) (j : Nat) : Word :=
  (w.val.getD j (0#u32)).bv

/-- The extracted sixty-four word schedule read as the canonical schedule. -/
def wView (w : Array Std.U32 64#usize) : Vector Word 64 :=
  Vector.ofFn fun i : Fin 64 => wGet w i.val

/-- Reading the schedule through `wView` is reading the word's bit vector. -/
theorem wView_get (w : Array Std.U32 64#usize) (i : Fin 64) :
    (wView w).get i = wGet w i.val := by
  show (wView w)[i.val] = wGet w i.val
  simp only [wView, Vector.getElem_ofFn]

/-- A bounded `Array.index_usize` read is the underlying list read. -/
theorem array_index_usize_eq {α : Type} {n : Std.Usize} (a : Array α n) (i : Std.Usize)
    (h : i.val < a.val.length) : Array.index_usize a i = ok (a.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Array.index_usize_spec a i h)
  rw [hxv] at hx
  exact hx

/-- An in-bounds read through the list is the read through `getD`. -/
theorem getElem_getD {α : Type} (l : List α) (j : Nat) (h : j < l.length) (d : α) :
    (l[j]'h) = l.getD j d := by
  show l[j] = (l.getD j d)
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h, Option.getD_some]

/-- Reading a list after one element is written. -/
theorem getD_set {α : Type} (l : List α) (i j : Nat) (v : α)
    (h : i < l.length) (d : α) :
    (l.set i v).getD j d = if i = j then v else l.getD j d := by
  by_cases heq : i = j
  · subst heq
    rw [List.getD_eq_getElem?_getD, List.getElem?_set_self h, Option.getD_some]
    simp
  · rw [List.getD_eq_getElem?_getD, List.getElem?_set, if_neg heq,
      ← List.getD_eq_getElem?_getD]
    simp [heq]

/-- Reading a row of the extracted schedule after one word is written. -/
theorem wGet_set (w : Array Std.U32 64#usize) (i : Std.Usize) (v : Std.U32)
    (hi : i.val < 64) (j : Nat) :
    wGet (w.set i v) j = if i.val = j then v.bv else wGet w j := by
  have hl : w.val.length = 64 := by simpa using w.property
  unfold wGet
  rw [Array.set_val_eq, getD_set _ _ _ _ (by omega) (0#u32)]
  by_cases h : i.val = j <;> simp [h]

/-! ## The block-word path

`sha256.compress` fills its schedule with `w[i] = u32::from_be_bytes([b[4i], b[4i+1],
b[4i+2], b[4i+3]])` (`src/sha256.rs:37-41`), so the extracted `Array Std.U8 64#usize`
is read as the canonical `Sha2.Block` by big-endian regrouping. `bytesToWords` is
that one read; there is no second spelling of it.

`ofFn_getD'` is `ofFn_getD` at an arbitrary length, needed here because the
block is `Vector Word 16` while the schedule is `Vector Word 64`. -/

/-- Reading a `Vector.ofFn` of any length through `getD` is reading the function. -/
theorem ofFn_getD' {α : Type} {n : Nat} (f : Fin n → α) (i : Nat) (hi : i < n) (d : α) :
    (Vector.ofFn f).getD i d = f ⟨i, hi⟩ := by
  simp [Vector.getD, hi]

/-- Byte `j` of an extracted 64-byte block, as the canonical byte. -/
def b64 (blk : Array Std.U8 64#usize) (j : Nat) : BitVec 8 :=
  (blk.val.getD j (0#u8)).bv

/-- Word `i` of the big-endian regrouping of an extracted 64-byte block: the
    inverse of `u32::from_be_bytes` at `[b[4i], b[4i+1], b[4i+2], b[4i+3]]`. -/
def beWord16 (blk : Array Std.U8 64#usize) (i : Fin 16) : Word :=
  BitVec.ofNat 32
    ((b64 blk (4 * i.val)).toNat * 16777216 +
      (b64 blk (4 * i.val + 1)).toNat * 65536 +
      (b64 blk (4 * i.val + 2)).toNat * 256 +
      (b64 blk (4 * i.val + 3)).toNat)

/-- The extracted 64-byte block read as the canonical sixteen-word block:
    word `i` is the big-endian regrouping of bytes `4i … 4i+3`. -/
def bytesToWords (blk : Array Std.U8 64#usize) : Block :=
  Vector.ofFn fun i : Fin 16 => beWord16 blk i

/-- Reading the block through `bytesToWords` is the big-endian regrouping. -/
theorem bytesToWords_get (blk : Array Std.U8 64#usize) (i : Fin 16) :
    (bytesToWords blk).get i = beWord16 blk i := by
  show (bytesToWords blk)[i.val] = beWord16 blk i
  simp only [bytesToWords, Vector.getElem_ofFn]

/-- The same read through `getD`, at any row below sixteen. -/
theorem bytesToWords_getD (blk : Array Std.U8 64#usize) (j : Nat) (hj : j < 16) :
    (bytesToWords blk).getD j 0 = beWord16 blk ⟨j, hj⟩ :=
  ofFn_getD' _ j hj 0

/-- A 64-byte block has length 64, so every `beWord16` byte index is in range. -/
theorem b64_length (blk : Array Std.U8 64#usize) : blk.val.length = 64 := by
  simpa using blk.property

/-- Below row sixteen the canonical schedule row of an extracted block is the
    big-endian regrouping of that block's bytes. -/
theorem schedRow_block (blk : Array Std.U8 64#usize) (j : Nat) (hj : j < 16) :
    schedRow (bytesToWords blk) j = beWord16 blk ⟨j, hj⟩ := by
  rw [schedRow, dif_pos hj, bytesToWords_getD blk j hj]

/-! ## The zero schedule

`sha256.compress` starts its schedule at `Array.repeat 64#usize 0#u32`, so the
state of every row before `sha256.compress_loop0` writes row `0` is zero, and
the canonical fold before its first extension step is `schedStart`. -/

/-- The canonical fold before its first extension step is the start vector: the
    sixty-four rows `sha256.compress` holds before `compress_loop1` runs are the
    sixteen block words below row sixteen and zero above. -/
theorem schedZero_spec (b : Block) : schedN b 0 = schedStart b := rfl

/-- Every row of the all-zero 64-word array `sha256.compress` starts its schedule
    from reads as zero. -/
theorem wGet_zero (j : Nat) (hj : j < 64) :
    wGet (Array.repeat 64#usize 0#u32) j = 0 := by
  show ((Array.repeat 64#usize 0#u32).val.getD j (0#u32)).bv = 0
  rw [Array.repeat_val]
  have h64 : (64#usize : Std.Usize).val = 64 := by rfl
  rw [h64]
  have hrep : (List.replicate 64 (0#u32)).getD j (0#u32) = 0#u32 := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_replicate, if_pos hj, Option.getD_some]
  rw [hrep]
  rfl

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