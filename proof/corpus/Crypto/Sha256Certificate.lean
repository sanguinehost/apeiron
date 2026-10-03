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

/-! ### Four-summand reordering in `BitVec n`

The two schedule-extension obligations below are one obligation: the extracted body sums the
row's four words in the order `W16 + σ₀ + W7 + σ₁` while FIPS 180-4 §6.2.2 step 1 reads
`σ₀ + W7 + σ₁ + W16`, so the two spellings of the same word differ by a permutation of four
`BitVec n` summands.

The permutation is proved from core's `BitVec.add_assoc` and `BitVec.add_comm`
(`Init.Data.BitVec.Lemmas`), which hold by `eq_of_toNat_eq` over `Nat.add_assoc`/`Nat.add_comm`
and are hence true at every width. Two spellings of the permutation are needed and they are
different permutations, so one statement cannot serve both goals; `bv_add4_rot` closes the
extracted order against the FIPS order and `bv_add4_swap13`, derived from it by two summand
swaps, closes the `schedRow` order against the FIPS order.

The alternatives are not usable here, and the reason is recorded so it is not retried. `ac_rfl`
does not close either goal: the pinned toolchain's `ac_rfl` does not consume the `Std.Associative`
/ `Std.Commutative` instances that `BitVec` registers for `+`. `bv_decide` is not usable at the
*call sites*: the summands there are schematic `wGet` reads, not closed bit-vectors (it would be
usable for a helper stated over four abstract summands, which is why the helpers are stated
abstractly and instantiated). A `BitVec.eq_of_toInt_eq` route leaves `bmod 2 ^ n` residues that
`omega` cannot discharge, and `simp only` over `add_assoc`/`add_comm` alone normalizes only the
associativity: there is no `BitVec.add_left_comm` in this toolchain, so the summand order is left
untouched (`d + (c + (a + b)) = d + (a + (b + c))` remains). No bound on any intermediate sum is
needed: both helpers reassociate and commute four `BitVec` summands directly. -/

/-- **Rotation of four `BitVec n` summands**, from `BitVec.add_assoc` and `BitVec.add_comm`. -/
theorem bv_add4_rot {n : Nat} (a b c d : BitVec n) : a + b + c + d = b + c + d + a := by
  calc a + b + c + d = (a + b) + (c + d) := BitVec.add_assoc (a + b) c d
    _ = a + (b + (c + d)) := BitVec.add_assoc a b (c + d)
    _ = a + ((b + c) + d) := by rw [BitVec.add_assoc b c d]
    _ = ((b + c) + a) + d := by
      rw [← BitVec.add_assoc a (b + c) d, BitVec.add_comm a (b + c)]
    _ = ((b + c) + d) + a := by
      rw [BitVec.add_assoc (b + c) a d, BitVec.add_comm a d, ← BitVec.add_assoc (b + c) d a]
    _ = b + c + d + a := rfl

/-- **Swap of the first and third of four `BitVec n` summands**, from `bv_add4_rot` followed by
    a swap of the first two and a swap of the last two summands. -/
theorem bv_add4_swap13 {n : Nat} (a b c d : BitVec n) : a + b + c + d = c + b + a + d :=
  (bv_add4_rot a b c d).trans (by
    calc b + c + d + a = c + b + d + a := by rw [BitVec.add_comm b c]
      _ = (c + b) + (d + a) := by rw [BitVec.add_assoc (c + b) d a]
      _ = (c + b) + (a + d) := by rw [BitVec.add_comm d a]
      _ = c + b + a + d := by rw [← BitVec.add_assoc (c + b) a d])

/-! ## One schedule-extension visit

`sha256.compress_loop1.body` (`src/sha256.rs:43-51`) reads rows `i-15` three times for
`σ₀`, rows `i-2` three times for `σ₁`, rows `i-16` and `i-7` once each, and writes row `i`
with their sum. Every read is below the row being written, and every read index is in
range because `16 ≤ i.val < 64` and the array holds `w.val.length = 64` entries.

`sigma0_step`/`sigma1_step` are the extracted `σ₀`/`σ₁` sequences at the extracted rotate
and shift amounts, read at the shift result `z`; the shift result is what the body binds,
so the shift amount appears once, as a hypothesis.

The successor index is not identified with the sum `i + 1#usize` as a value: `Usize.add` is
a partial operation whose bound premise is discharged separately, so the body theorem names
the successor index and states its `.val` relation. -/

/-- The extracted `σ₀(x) = ROTR^7 x ⊕ ROTR^18 x ⊕ (x >>> 3)` at the extracted rotate
    sequence, read at the shift result `z`. -/
theorem sigma0_step (x z : Std.U32) (h : x >>> 3#i32 = ok z) :
    ((core.num.U32.rotate_right x 7#u32).bv ^^^
      (core.num.U32.rotate_right x 18#u32).bv ^^^ z.bv) =
      Apeiron.Crypto.Sha2.sigma0 x.bv := by
  obtain ⟨z', hz', hz'ok⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_IScalar_spec x 3#i32 (by decide) (by decide))
  have hzz : z.bv = z'.bv :=
    congrArg (fun t : Std.U32 => t.bv) (Result.ok_injective (h.symm.trans hz'))
  have h3 : (3#i32 : Std.I32).toNat = 3 := by decide
  have h7 : (7#u32 : Std.U32).val = 7 := rfl
  have h18 : (18#u32 : Std.U32).val = 18 := rfl
  simp only [Apeiron.Crypto.Sha2.sigma0]
  rw [rotr_step _ (by rw [h7]; omega) (by rw [h7]; omega),
    rotr_step _ (by rw [h18]; omega) (by rw [h18]; omega)]
  simp only [hzz, hz'ok.2, h3, h7, h18, Apeiron.Crypto.Sha2.shrx]

/-- The extracted `σ₁(x) = ROTR^17 x ⊕ ROTR^19 x ⊕ (x >>> 10)` at the extracted rotate
    sequence, read at the shift result `z`. -/
theorem sigma1_step (x z : Std.U32) (h : x >>> 10#i32 = ok z) :
    ((core.num.U32.rotate_right x 17#u32).bv ^^^
      (core.num.U32.rotate_right x 19#u32).bv ^^^ z.bv) =
      Apeiron.Crypto.Sha2.sigma1 x.bv := by
  obtain ⟨z', hz', hz'ok⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_IScalar_spec x 10#i32 (by decide) (by decide))
  have hzz : z.bv = z'.bv :=
    congrArg (fun t : Std.U32 => t.bv) (Result.ok_injective (h.symm.trans hz'))
  have h10 : (10#i32 : Std.I32).toNat = 10 := by decide
  have h17 : (17#u32 : Std.U32).val = 17 := rfl
  have h19 : (19#u32 : Std.U32).val = 19 := rfl
  simp only [Apeiron.Crypto.Sha2.sigma1]
  rw [rotr_step _ (by rw [h17]; omega) (by rw [h17]; omega),
    rotr_step _ (by rw [h19]; omega) (by rw [h19]; omega)]
  simp only [hzz, hz'ok.2, h10, h17, h19, Apeiron.Crypto.Sha2.shrx]

/-- **One schedule-extension visit.** With `16 ≤ i.val < 64` and `w.val.length = 64`,
    `sha256.compress_loop1.body w i` is `ok (cont (w.set i v, i'))` where `v` is exactly
    the FIPS 180-4 §6.2.2 step-1 word
    `σ₀(W[i-15]) + W[i-7] + σ₁(W[i-2]) + W[i-16]` read through `wGet`, and
    `i'.val = i.val + 1`. The body never fails: each subtraction is guarded by
    `16 ≤ i.val`, each read index is `< 64`, and the write index is `< 64`. -/
theorem schedBody_step (w : Array Std.U32 64#usize) (i : Std.Usize)
    (hlo : 16 ≤ i.val) (hlt : i.val < 64) :
    ∃ v : Std.U32, ∃ i' : Std.Usize,
      sha256.compress_loop1.body w i = ok (ControlFlow.cont (w.set i v, i')) ∧
      i'.val = i.val + 1 ∧
      v.bv = Apeiron.Crypto.Sha2.sigma0 (wGet w (i.val - 15)) + wGet w (i.val - 7)
        + Apeiron.Crypto.Sha2.sigma1 (wGet w (i.val - 2)) + wGet w (i.val - 16) := by
  have hl : w.val.length = 64 := by simpa using w.property
  have h1v : (1#usize : Std.Usize).val = 1 := rfl
  have h2v : (2#usize : Std.Usize).val = 2 := rfl
  have h7v : (7#usize : Std.Usize).val = 7 := rfl
  have h15v : (15#usize : Std.Usize).val = 15 := rfl
  have h16v : (16#usize : Std.Usize).val = 16 := rfl
  -- The four subtraction indices, read out of the checked `usize` subtractions. The
  -- bounds below are stated against the index the body passes (`i1.val`, `i9.val`, …),
  -- because `Array.index_usize_spec` needs the bound on that index.
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.sub_spec (x := i) (y := 15#usize) (by rw [h15v]; omega))
  have h1 : i1.val = i.val - 15 := by rw [hi1v.1, h15v]
  obtain ⟨i9, hi9, hi9v⟩ :=
    WP.spec_imp_exists (Usize.sub_spec (x := i) (y := 2#usize) (by rw [h2v]; omega))
  have h9 : i9.val = i.val - 2 := by rw [hi9v.1, h2v]
  obtain ⟨i17, hi17, hi17v⟩ :=
    WP.spec_imp_exists (Usize.sub_spec (x := i) (y := 16#usize) (by rw [h16v]; omega))
  have h17 : i17.val = i.val - 16 := by rw [hi17v.1, h16v]
  obtain ⟨i20, hi20, hi20v⟩ :=
    WP.spec_imp_exists (Usize.sub_spec (x := i) (y := 7#usize) (by rw [h7v]; omega))
  have h20 : i20.val = i.val - 7 := by rw [hi20v.1, h7v]
  have hb15 : i1.val < w.val.length := by rw [h1, hl]; omega
  have hb2 : i9.val < w.val.length := by rw [h9, hl]; omega
  have hb16 : i17.val < w.val.length := by rw [h17, hl]; omega
  have hb7 : i20.val < w.val.length := by rw [h20, hl]; omega
  have hbn15 : i.val - 15 < w.val.length := by rw [hl]; omega
  have hbn2 : i.val - 2 < w.val.length := by rw [hl]; omega
  have hbn16 : i.val - 16 < w.val.length := by rw [hl]; omega
  have hbn7 : i.val - 7 < w.val.length := by rw [hl]; omega
  -- The four rows, named at the index the body reads and matched to the `wGet` row
  -- through `getD`, whose rewrite in the read index is type correct.
  let r15 : Std.U32 := w.val[i.val - 15]'hbn15
  let r2 : Std.U32 := w.val[i.val - 2]'hbn2
  let r16 : Std.U32 := w.val[i.val - 16]'hbn16
  let r7 : Std.U32 := w.val[i.val - 7]'hbn7
  have hd15 : w.val.getD i1.val (0#u32) = w.val.getD (i.val - 15) (0#u32) := by rw [h1]
  have hd2 : w.val.getD i9.val (0#u32) = w.val.getD (i.val - 2) (0#u32) := by rw [h9]
  have hd16 : w.val.getD i17.val (0#u32) = w.val.getD (i.val - 16) (0#u32) := by rw [h17]
  have hd7 : w.val.getD i20.val (0#u32) = w.val.getD (i.val - 7) (0#u32) := by rw [h20]
  have heq15 : w.val[i1.val]'hb15 = w.val[i.val - 15]'hbn15 := by
    calc w.val[i1.val]'hb15 = w.val.getD i1.val (0#u32) := getElem_getD _ _ _ _
      _ = w.val.getD (i.val - 15) (0#u32) := hd15
      _ = w.val[i.val - 15]'hbn15 := (getElem_getD _ _ _ _).symm
  have heq2 : w.val[i9.val]'hb2 = w.val[i.val - 2]'hbn2 := by
    calc w.val[i9.val]'hb2 = w.val.getD i9.val (0#u32) := getElem_getD _ _ _ _
      _ = w.val.getD (i.val - 2) (0#u32) := hd2
      _ = w.val[i.val - 2]'hbn2 := (getElem_getD _ _ _ _).symm
  have heq16 : w.val[i17.val]'hb16 = w.val[i.val - 16]'hbn16 := by
    calc w.val[i17.val]'hb16 = w.val.getD i17.val (0#u32) := getElem_getD _ _ _ _
      _ = w.val.getD (i.val - 16) (0#u32) := hd16
      _ = w.val[i.val - 16]'hbn16 := (getElem_getD _ _ _ _).symm
  have heq7 : w.val[i20.val]'hb7 = w.val[i.val - 7]'hbn7 := by
    calc w.val[i20.val]'hb7 = w.val.getD i20.val (0#u32) := getElem_getD _ _ _ _
      _ = w.val.getD (i.val - 7) (0#u32) := hd7
      _ = w.val[i.val - 7]'hbn7 := (getElem_getD _ _ _ _).symm
  -- The eight reads, three each at rows `i-15` and `i-2` and one each at `i-16`, `i-7`.
  have hix2 : Array.index_usize w i1 = ok r15 := by
    rw [array_index_usize_eq w i1 hb15]
    show ok (w.val[i1.val]'hb15) = ok (w.val[i.val - 15]'hbn15)
    exact congrArg (fun t : Std.U32 => ok t) heq15
  have hix4 : Array.index_usize w i1 = ok r15 := hix2
  have hix7 : Array.index_usize w i1 = ok r15 := hix2
  have hix10 : Array.index_usize w i9 = ok r2 := by
    rw [array_index_usize_eq w i9 hb2]
    show ok (w.val[i9.val]'hb2) = ok (w.val[i.val - 2]'hbn2)
    exact congrArg (fun t : Std.U32 => ok t) heq2
  have hix12 : Array.index_usize w i9 = ok r2 := hix10
  have hix15 : Array.index_usize w i9 = ok r2 := hix10
  have hix18 : Array.index_usize w i17 = ok r16 := by
    rw [array_index_usize_eq w i17 hb16]
    show ok (w.val[i17.val]'hb16) = ok (w.val[i.val - 16]'hbn16)
    exact congrArg (fun t : Std.U32 => ok t) heq16
  have hix21 : Array.index_usize w i20 = ok r7 := by
    rw [array_index_usize_eq w i20 hb7]
    show ok (w.val[i20.val]'hb7) = ok (w.val[i.val - 7]'hbn7)
    exact congrArg (fun t : Std.U32 => ok t) heq7
  -- The two logical shifts, the index increment and the written word.
  obtain ⟨i8, hi8, _⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_IScalar_spec r15 3#i32 (by decide) (by decide))
  obtain ⟨i16, hi16, _⟩ :=
    WP.spec_imp_exists (UScalar.ShiftRight_IScalar_spec r2 10#i32 (by decide) (by decide))
  have hbound : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    rw [h1v]; simp [Usize.max, Usize.numBits];
        cases System.Platform.numBits_eq <;> simp_all <;> omega
  obtain ⟨i24, hi24, hi24v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hbound)
  let v : Std.U32 :=
    core.num.U32.wrapping_add (core.num.U32.wrapping_add
      (core.num.U32.wrapping_add r16
        (core.num.U32.rotate_right r15 7#u32 ^^^ core.num.U32.rotate_right r15 18#u32 ^^^ i8))
      r7)
      (core.num.U32.rotate_right r2 17#u32 ^^^ core.num.U32.rotate_right r2 19#u32 ^^^ i16)
  have hltlen : i.val < w.length := by
    simpa only [Array.length, hl] using hlt
  have hupd : Array.update w i
      (core.num.U32.wrapping_add (core.num.U32.wrapping_add
        (core.num.U32.wrapping_add r16
          (core.num.U32.rotate_right r15 7#u32 ^^^ core.num.U32.rotate_right r15 18#u32 ^^^ i8))
        r7)
        (core.num.U32.rotate_right r2 17#u32 ^^^ core.num.U32.rotate_right r2 19#u32 ^^^ i16)) =
      ok (w.set i v) := by
    obtain ⟨nv, hnv, hnv'⟩ := WP.spec_imp_exists (Array.update_spec w i v hltlen)
    rw [hnv'] at hnv
    exact hnv
  have hr15 : r15.bv = wGet w (i.val - 15) := by
    show (w.val[i.val - 15]'hbn15).bv = _
    simp only [wGet, getElem_getD _ _ hbn15 (0#u32)]
  have hr2 : r2.bv = wGet w (i.val - 2) := by
    show (w.val[i.val - 2]'hbn2).bv = _
    simp only [wGet, getElem_getD _ _ hbn2 (0#u32)]
  have hr16 : r16.bv = wGet w (i.val - 16) := by
    show (w.val[i.val - 16]'hbn16).bv = _
    simp only [wGet, getElem_getD _ _ hbn16 (0#u32)]
  have hr7 : r7.bv = wGet w (i.val - 7) := by
    show (w.val[i.val - 7]'hbn7).bv = _
    simp only [wGet, getElem_getD _ _ hbn7 (0#u32)]
  refine ⟨v, i24, ?_, ?_, ?_⟩
  · -- The replay: every read, shift and write is its specification, in one pass.
    unfold sha256.compress_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
    simp only [hi1, hi9, hi17, hi20, hix2, hix4, hix7, hix10, hix12, hix15, hix18, hix21,
      hi8, hi16, hupd, hi24, lift, bind_tc_ok]
  · rw [hi24v]
    rfl
  · have hbvv : v.bv =
        r16.bv +
            ((core.num.U32.rotate_right r15 7#u32).bv ^^^
              (core.num.U32.rotate_right r15 18#u32).bv ^^^ i8.bv) +
          r7.bv +
            ((core.num.U32.rotate_right r2 17#u32).bv ^^^
              (core.num.U32.rotate_right r2 19#u32).bv ^^^ i16.bv) := rfl
    rw [hbvv, hr16, hr7, sigma0_step r15 i8 hi8, hr15, sigma1_step r2 i16 hi16, hr2]
    -- The extracted order `W16 + σ₀ + W7 + σ₁` against the FIPS order
    -- `σ₀ + W7 + σ₁ + W16`, by `bv_add4_rot`.
    exact bv_add4_rot _ _ _ _

/-- Reading the extracted schedule through `wView` through `getD` is reading the row. -/
theorem wView_getD (w : Array Std.U32 64#usize) (j : Nat) (hj : j < 64) :
    (wView w).getD j 0 = wGet w j :=
  ofFn_getD _ j hj

/-- The loop invariant for `sha256.compress_loop1`, at the invariant measure
    `64 - i.val`: the rows written so far are exactly the canonical rows below `i`. -/
def schedInv (b : Block) (p : Array Std.U32 64#usize × Std.Usize) : Prop :=
  16 ≤ p.2.val ∧ p.2.val ≤ 64 ∧ wView p.1 = rowVec b (p.2.val - 16)

/-- **One extension step preserves the schedule invariant.** Writing row `i` with the
    FIPS 180-4 §6.2.2 step-1 word makes the written schedule read as `rowVec b` at the
    next row index, because every row the step reads lies strictly below `i`. -/
theorem schedStep_inv (b : Block) (w : Array Std.U32 64#usize) (i : Std.Usize) (v : Std.U32)
    (hlo : 16 ≤ i.val) (hlt : i.val < 64)
    (hinv : wView w = rowVec b (i.val - 16))
    (hv : v.bv = Apeiron.Crypto.Sha2.sigma0 (wGet w (i.val - 15)) + wGet w (i.val - 7)
        + Apeiron.Crypto.Sha2.sigma1 (wGet w (i.val - 2)) + wGet w (i.val - 16)) :
    wView (w.set i v) = rowVec b (i.val - 16 + 1) := by
  refine vec_getD_eq _ _ 0 fun j hj => ?_
  have hgj : wGet w j = (rowVec b (i.val - 16)).getD j 0 := by
    rw [← wView_getD w j hj, hinv]
  rw [wView_getD _ j hj, wGet_set w i v hlt j,
    rowVec_getD b (i.val - 16 + 1) j hj, hgj, rowVec_getD b (i.val - 16) j hj]
  by_cases hji : i.val = j
  · subst hji
    have hsub : i.val - 16 + 16 = i.val := Nat.sub_add_cancel hlo
    have hcond : i.val < 16 + (i.val - 16 + 1) := by
      have h1 : 16 + (i.val - 16) = i.val := by rw [Nat.add_comm, hsub]
      have h2 : 16 + (i.val - 16 + 1) = i.val + 1 := by
        rw [← Nat.add_assoc, h1]
      rw [h2]
      omega
    rw [if_pos rfl, if_pos hcond]
    have hA : wGet w (i.val - 15) = schedRow b (i.val - 15) := by
      have hX : (i.val - 15 : Nat) < 64 := by omega
      rw [← wView_getD w _ hX, hinv, rowVec_getD b (i.val - 16) _ hX, if_pos (by omega)]
    have hB : wGet w (i.val - 7) = schedRow b (i.val - 7) := by
      have hX : (i.val - 7 : Nat) < 64 := by omega
      rw [← wView_getD w _ hX, hinv, rowVec_getD b (i.val - 16) _ hX, if_pos (by omega)]
    have hC : wGet w (i.val - 2) = schedRow b (i.val - 2) := by
      have hX : (i.val - 2 : Nat) < 64 := by omega
      rw [← wView_getD w _ hX, hinv, rowVec_getD b (i.val - 16) _ hX, if_pos (by omega)]
    have hD : wGet w (i.val - 16) = schedRow b (i.val - 16) := by
      have hX : (i.val - 16 : Nat) < 64 := by omega
      rw [← wView_getD w _ hX, hinv, rowVec_getD b (i.val - 16) _ hX, if_pos (by omega)]
    simp only [hv, hA, hB, hC, hD]
    -- `schedRow_step` spells the row `σ₁ + W7 + σ₀ + W16`; the goal carries the FIPS order
    -- `σ₀ + W7 + σ₁ + W16`, which is the same four summands by `bv_add4_swap13`.
    exact (bv_add4_swap13 _ _ _ _).trans (schedRow_step b i.val hlo).symm
  · simp only [if_neg hji]
    by_cases hjlt : j < 16 + (i.val - 16)
    · rw [if_pos hjlt, if_pos (by omega)]
    · rw [if_neg hjlt, if_neg (by omega)]

/-! ## The extracted schedule loop

`sha256.compress_loop1` is `loop (sha256.compress_loop1.body) w 16#usize`, run by
`loop.spec_decr_nat` on the measure `64 - i.val`: forty-eight extension visits from the
sixteen block words to the full sixty-four row schedule, `Sha2.schedule`. -/

/-- **`sha256.compress_loop1` extends a schedule holding the canonical block words in
    its first sixteen rows into `Sha2.schedule`.** The invariant is `schedInv` at the
    measure `64 - i.val`, discharged by `schedStep_inv`; the terminal visit is the
    `64`-th row, where `schedInv` already reads as the full forty-eight-step fold. -/
theorem sched_loop1 (w : Array Std.U32 64#usize) (b : Block) (w' : Array Std.U32 64#usize)
    (hinv : wView w = rowVec b 0)
    (h : sha256.compress_loop1 w 16#usize = ok w') :
    wView w' = Apeiron.Crypto.Sha2.schedule b := by
  have h16 : (16#usize : Std.Usize).val = 16 := rfl
  have h0 : 16 - 16 = 0 := by omega
  have hloop :
      loop (fun p : Array Std.U32 64#usize × Std.Usize => sha256.compress_loop1.body p.1 p.2)
        (w, 16#usize) ⦃ fun y => wView y = Apeiron.Crypto.Sha2.schedule b ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : Array Std.U32 64#usize × Std.Usize => 64 - p.2.val)
      (inv := schedInv b)
      (post := fun y => wView y = Apeiron.Crypto.Sha2.schedule b)
    · intro p hp
      obtain ⟨hlo, hle, hw⟩ := hp
      by_cases hlt : p.2.val < 64
      · obtain ⟨v, i24, hb, hi24v, hv⟩ := schedBody_step p.1 p.2 hlo hlt
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hi24v]; omega
        · rw [hi24v]; omega
        · show wView (p.1.set p.2 v) = rowVec b (i24.val - 16)
          exact (schedStep_inv b p.1 p.2 v hlo hlt hw hv).trans (by rw [hi24v]; congr 1; omega)
        · rw [hi24v]; omega
      · rw [WP.spec_equiv_exists]
        refine ⟨ControlFlow.done p.1, ?_, ?_⟩
        · unfold sha256.compress_loop1.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        · refine vec_getD_eq _ _ 0 fun j hj => ?_
          rw [hw, rowVec_getD b (p.2.val - 16) j hj, schedule_getD b j hj]
          have hk : p.2.val - 16 = 48 := by omega
          rw [hk, if_pos (by omega)]
    · refine ⟨?_, ?_, ?_⟩
      · rw [h16]
      · rw [h16]
        omega
      · rw [h16, h0]
        exact hinv
  have hloop' :
      loop (fun p : Array Std.U32 64#usize × Std.Usize => sha256.compress_loop1.body p.1 p.2)
        (w, 16#usize) = ok w' := by
    simpa only [sha256.compress_loop1] using h
  obtain ⟨y, hy, hPy⟩ := WP.spec_imp_exists hloop
  rw [hloop'] at hy
  have hw'y : w' = y := Result.ok_injective hy
  rw [← hw'y] at hPy
  exact hPy

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