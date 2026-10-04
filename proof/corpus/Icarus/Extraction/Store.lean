/-
  `Icarus/Extraction/Store.lean` — **O9 (c), the digest-store link.**

  Owned files: this one and the task receipt. No corpus edit, no extraction refresh, no commit.
  Read-only upstream: `Crypto/Sha2.lean`, `Crypto/Sha256Certificate.lean`, and the generated
  `ApeironWord.lean`. The campaign staging source `work/proofs/Sha256Store.lean` lives outside this
  package and is cited, not imported. Nothing from the rejected bundle
  `work/quarantine/Sha256Headline.vacuous-bundle.lean.incomplete` is imported or reused: there is no
  `Sha256Stages` here, and no hypothesis of any kind is assumed anywhere in this file.

  ## The extracted loop

  `sha256.sha256_loop2` (`ApeironWord.lean:15510-15548`, from `src/sha256.rs:138-141`):

  ```rust
  let mut i = 0;
  while i < 8 { out[4*i..4*i+4].copy_from_slice(&state[i].to_be_bytes()); i += 1; }
  ```

  so the loop writes the eight chaining words big-endian into the 32-byte digest and does nothing
  else. It is the last stage of `word_sha256` before the digest is returned.

  ## The link

  The canonical object is `Apeiron.Crypto.Sha2.State = Vector (BitVec 32) 8`
  (`Crypto/Sha2.lean:25`), read at two ends of this repository's bridge:

  * `stateView` — the extracted `Array Std.U32 8#usize` read as `Sha2.State`;
  * `digView` — the extracted `[u8; 32]` digest read back as `Sha2.State` by the big-endian
    regrouping `H[i] = out[4i]·2²⁴ + out[4i+1]·2¹⁶ + out[4i+2]·2⁸ + out[4i+3]`
    (`Crypto/Sha256Certificate.lean:144-157`).

  `store_spec_exists` is the link: the loop **succeeds**, and its result's `digView` **is** the input
  state's `stateView`. `Crypto/Sha256Certificate.lean:36-51` records that the `digView` write lemma
  is not carried at stage 0 and defers it to stage 4; it is discharged here.

  ## What is proved, in order

  * §1 pure kernels. `BitVec.toBEBytes` is `List.reverse (toLEBytes b)` and `toLEBytes` peels the low
    byte first, so the four bytes of a 32-bit word are `b>>>24, b>>>16, b>>>8, b`, each truncated to
    eight bits. `beByte0 … beByte3` fix those and `beWordStored_eq` closes the regrouping.
    `regroup_diff` / `leWord4_ne` are the **byte-reversal discriminator**: the big-endian and
    little-endian readings of four bytes `a b c d` differ whenever `a ≠ d`, so a store that wrote the
    bytes in the wrong order cannot satisfy the regrouping. `O(1)` throughout: four-byte `BitVec`
    arithmetic, no extracted body, no `Array`.
  * §2 the `Result` plumbing: an in-range `index_mut` succeeds with the `List.setSlice!` back
    function, and `copy_from_slice` of two length-four slices succeeds.
  * §3 the body step `storeBody_step`: for `i.val < 8` the body succeeds, writes the four
    big-endian bytes of `stateGet state i.val` at positions `4i … 4i+3`, agrees with the incoming
    array below `4i` and at or above `4i+4`, and returns `i.val + 1`. `storeBody_done` is the other
    branch and `storeBody_total` the no-failure statement.
  * §4 the preserved-prefix invariant `storeInv`, with `storeInv_init` and `storeBody_step_pres`.
  * §5 the loop by `loop.spec_decr_nat` at the measure `8 - i.val`.
  * §6 the two headline shapes kept apart: `store_spec_of_ok` (correctness **conditioned on** a
    successful result; no existence argument) and `store_spec_exists` (total existence of a
    successful, correctly-denoting result).
  * §7 the witnesses: an explicit eight-word state whose words are pairwise distinct and distinct
    from each other's byte reversals, on which `digView out = stateView state` holds while every
    byte-reversed reading differs.

  ## Grades and non-claims

  `proved` for §§1–7 from the build gate recorded in the receipt, `#print axioms` rows inside
  `{propext, Classical.choice, Quot.sound}`, no `sorryAx`. No `native_decide`, no `decide` over a
  source-sized proposition, no `maxHeartbeats`/`maxRecDepth` override; the literal evaluations are
  over four-byte `BitVec` words and eight-element vectors.

  **The store is a pure `Array`-to-`Array` byte kernel.** No SHA schedule, compression, padding or
  `Sha2.hash` fact is used or needed, and none is claimed: `store_spec_exists` links the loop to
  `stateView`, which is the same view the head of `sha256.sha256` produces
  (`ApeironWord.lean:15588-15590`). It is not a statement about the whole hash. Nothing here bears on
  security, collision or preimage resistance, differential cryptanalysis, side channels or
  constant-time behaviour, and no timing is measured.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false
set_option linter.unnecessarySeqFocus false

namespace Icarus
namespace Extraction
namespace Store

open Apeiron.Crypto.Sha2 (Word State Block)
open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## §1 · Pure kernels: big-endian regrouping, and byte reversal detected

Everything in this section is `BitVec 32` arithmetic over four bytes: no extracted body, no `Array`,
no SHA schedule or compression fact. -/

/-- Byte `k` of `U32::to_be_bytes b`, as a canonical byte. `O(1)`. -/
def beByte (b : BitVec 32) (k : Nat) : BitVec 8 :=
  ((b.toBEBytes.map (@UScalar.mk UScalarTy.U8)).getD k (UScalar.mk 0) : Std.U8).bv

/-- Reading the mapped `toBEBytes` list at index `k` is reading the byte list at index `k`. This is
the one place where the extracted `U32` byte and the canonical byte meet. `proved`. -/
theorem beByte_getD (b : BitVec 32) (k : Nat) :
    beByte b k = b.toBEBytes.getD k (0 : BitVec 8) := by
  unfold beByte
  rw [List.getD_map]

/-- A 32-bit word is below `2^32`. `proved`. -/
theorem bnd32 (b : BitVec 32) : b.toNat < 4294967296 :=
  @BitVec.toNat_lt_twoPow_of_le 32 32 (by omega) b

/-- A byte is below `2^8`. `proved`. -/
theorem bnd8 (b : BitVec 8) : b.toNat < 256 := by omega

/-- `toBEBytes` byte 0 is the word's most significant byte. `proved`. -/
theorem beByte0 (b : BitVec 32) : (beByte b 0).toNat = (b.toNat >>> 24) % 256 := by
  have hlt := bnd32 b
  rw [beByte_getD]
  simp [BitVec.toBEBytes, BitVec.toLEBytes, BitVec.toNat_setWidth] <;> omega

/-- `toBEBytes` byte 1. `proved`. -/
theorem beByte1 (b : BitVec 32) : (beByte b 1).toNat = (b.toNat >>> 16) % 256 := by
  have hlt := bnd32 b
  rw [beByte_getD]
  simp [BitVec.toBEBytes, BitVec.toLEBytes, BitVec.toNat_setWidth] <;> omega

/-- `toBEBytes` byte 2. `proved`. -/
theorem beByte2 (b : BitVec 32) : (beByte b 2).toNat = (b.toNat >>> 8) % 256 := by
  have hlt := bnd32 b
  rw [beByte_getD]
  simp [BitVec.toBEBytes, BitVec.toLEBytes, BitVec.toNat_setWidth] <;> omega

/-- `toBEBytes` byte 3 is the word's least significant byte. `proved`. -/
theorem beByte3 (b : BitVec 32) : (beByte b 3).toNat = b.toNat % 256 := by
  have hlt := bnd32 b
  rw [beByte_getD]
  simp [BitVec.toBEBytes, BitVec.toLEBytes, BitVec.toNat_setWidth] <;> omega

/-- The regrouping of the four big-endian bytes of `w`, the word the store writes for `w`: the
`toBEBytes` bytes of §1, regrouped as `b₀·2²⁴ + b₁·2¹⁶ + b₂·2⁸ + b₃`. Spelled through `toNat`
shifts rather than the byte list so that `beWordStored_eq`'s use sites do not force the
`toBEBytes`/`toLEBytes` list recursion at every unification; `beWordStored_eq` below is the
statement that this is the same regrouping §1 computes. `O(1)`. -/
def beWordStored (w : BitVec 32) : Word :=
  BitVec.ofNat 32 ((w.toNat >>> 24) % 256 * 16777216 + (w.toNat >>> 16) % 256 * 65536
    + (w.toNat >>> 8) % 256 * 256 + w.toNat % 256)

/-- **The big-endian regrouping of `toBEBytes` is the word.** This is the one lemma that ties the
extracted write to the canonical `Sha2.State` word, and it is the receipt's byte-order control site.
`proved`. -/
theorem beWordStored_eq (w : BitVec 32) : beWordStored w = w := by
  have hlt := bnd32 w
  have hsplit : ((w.toNat >>> 24) % 256) * 16777216 + ((w.toNat >>> 16) % 256) * 65536
      + ((w.toNat >>> 8) % 256) * 256 + (w.toNat % 256) = w.toNat := by omega
  show BitVec.ofNat 32
      ((w.toNat >>> 24) % 256 * 16777216 + (w.toNat >>> 16) % 256 * 65536
        + (w.toNat >>> 8) % 256 * 256 + w.toNat % 256) = w
  rw [hsplit, BitVec.ofNat_toNat, BitVec.setWidth_eq]

/-- **The shift regrouping is the byte regrouping.** `beWordStored` as defined above and the
`toBEBytes` bytes of §1 regrouped in the same order are the same word: this is the row that lets
§4's invariant, which is stated at the byte level, use `beWordStored_eq`. `proved`. -/
theorem beWordStored_beBytes (w : BitVec 32) :
    beWordStored w = BitVec.ofNat 32
      ((beByte w 0).toNat * 16777216 + (beByte w 1).toNat * 65536
        + (beByte w 2).toNat * 256 + (beByte w 3).toNat) := by
  have hlt := bnd32 w
  rw [beByte0, beByte1, beByte2, beByte3]
  have hsplit : ((w.toNat >>> 24) % 256) * 16777216 + ((w.toNat >>> 16) % 256) * 65536
      + ((w.toNat >>> 8) % 256) * 256 + (w.toNat % 256) = w.toNat := by omega
  show BitVec.ofNat 32
      ((w.toNat >>> 24) % 256 * 16777216 + (w.toNat >>> 16) % 256 * 65536
        + (w.toNat >>> 8) % 256 * 256 + w.toNat % 256) = _
  rw [hsplit, BitVec.ofNat_toNat, BitVec.setWidth_eq]

/-- The big-endian reading of four bytes `a b c d`, as a `Sha2.Word`. `O(1)`. -/
def beWord4 (a b c d : Nat) : Word :=
  BitVec.ofNat 32 (a * 16777216 + b * 65536 + c * 256 + d)

/-- The little-endian reading of the same four bytes: the word a store that wrote them in the
opposite order would read back as. `O(1)`. -/
def leWord4 (a b c d : Nat) : Word :=
  BitVec.ofNat 32 (d * 16777216 + c * 65536 + b * 256 + a)

/-- The big-endian reading of `a b c d` is congruent to `d` modulo `256`. `proved`. -/
theorem be_lo (a b c d : Nat) :
    (a * 16777216 + b * 65536 + c * 256 + d) % 256 = d % 256 := by
  have h : 256 ∣ a * 16777216 + b * 65536 + c * 256 := ⟨a * 65536 + b * 256 + c, by ring⟩
  omega

/-- The little-endian reading of `a b c d` is congruent to `a` modulo `256`. `proved`. -/
theorem le_lo (a b c d : Nat) :
    (d * 16777216 + c * 65536 + b * 256 + a) % 256 = a % 256 := by
  have h : 256 ∣ d * 16777216 + c * 65536 + b * 256 := ⟨d * 65536 + c * 256 + b, by ring⟩
  omega

/-- **Byte reversal is detected.** The two readings of four bytes `a b c d` differ as soon as
`a ≠ d`: reduced modulo `256` the big-endian reading is `d` and the little-endian one is `a`, and
both lie in `0 … 255`. `proved`. -/
theorem regroup_diff (a b c d : Nat) (ha : a < 256) (hb : b < 256) (hc : c < 256)
    (hd : d < 256) (hne : a ≠ d) :
    a * 16777216 + b * 65536 + c * 256 + d
      ≠ d * 16777216 + c * 65536 + b * 256 + a := by
  intro hEq
  have hm : (a * 16777216 + b * 65536 + c * 256 + d) % 256
      = (d * 16777216 + c * 65536 + b * 256 + a) % 256 := by rw [hEq]
  rw [be_lo, le_lo] at hm
  omega

/-- **The four-byte sum is below `2^32`.** The bound the `Nat.mod_eq_of_lt` reductions below need,
proved by a monotone chain over the byte bounds instead of `omega` over the large coefficients:
`omega` hits `maxRecDepth` on this goal once a sibling bound of the same shape is in context. `proved`. -/
theorem word4_bound (a b c d : Nat) (ha : a < 256) (hb : b < 256) (hc : c < 256)
    (hd : d < 256) :
    a * 16777216 + b * 65536 + c * 256 + d < 4294967296 := by
  have hsum : a * 16777216 + b * 65536 + c * 256 + d
      ≤ 255 * 16777216 + 255 * 65536 + 255 * 256 + 255 :=
    Nat.add_le_add (Nat.add_le_add (Nat.add_le_add
      (Nat.mul_le_mul_right _ (Nat.le_of_lt_succ ha))
      (Nat.mul_le_mul_right _ (Nat.le_of_lt_succ hb)))
      (Nat.mul_le_mul_right _ (Nat.le_of_lt_succ hc)))
      (Nat.le_of_lt_succ hd)
  exact Nat.lt_of_le_of_lt hsum (by norm_num)

/-- **Byte reversal is detected, as a statement about words.** Two 32-bit `ofNat`s built from the
four bytes `a b c d` are equal only if the outermost bytes agree, so a store that wrote
`out[4i..4i+4]` in the opposite order cannot produce the digest `beWordStored_eq` computes. This is
the discriminator the store byte-order control reddens. `proved`. -/
theorem leWord4_ne (a b c d : Nat) (ha : a < 256) (hb : b < 256) (hc : c < 256)
    (hd : d < 256) (hne : a ≠ d) : leWord4 a b c d ≠ beWord4 a b c d := by
  intro hEq
  have h1 : d * 16777216 + c * 65536 + b * 256 + a < 4294967296 :=
    word4_bound d c b a hd hc hb ha
  have h2 : a * 16777216 + b * 65536 + c * 256 + d < 4294967296 :=
    word4_bound a b c d ha hb hc hd
  have hle : (leWord4 a b c d).toNat = d * 16777216 + c * 65536 + b * 256 + a := by
    show (BitVec.ofNat 32 (d * 16777216 + c * 65536 + b * 256 + a)).toNat = _
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt h1]
  have hbe : (beWord4 a b c d).toNat = a * 16777216 + b * 65536 + c * 256 + d := by
    show (BitVec.ofNat 32 (a * 16777216 + b * 65536 + c * 256 + d)).toNat = _
    rw [BitVec.toNat_ofNat, Nat.mod_eq_of_lt h2]
  have h := congrArg BitVec.toNat hEq
  rw [hle, hbe] at h
  exact regroup_diff a b c d ha hb hc hd hne h.symm

/-! ## §2 · `Result` plumbing of the body

The body calls three operations that can fail. Each is discharged here under the bound the loop
guard supplies. -/

/-- An in-range `a[st … e]` never fails: the subslice is `a.val.slice st e` of length `e - st`, and
the back function writes `List.setSlice! st`. `proved`. -/
theorem index_mut_range_ok {α : Type} [Inhabited α] {n : Std.Usize}
    (a : Array α n) (st e : Std.Usize) (hst : st ≤ e) (he : e ≤ n) :
    ∃ (s : Slice α) (back : Slice α → Array α n),
      core.array.Array.index_mut (core.ops.index.IndexMutSlice
          (core.slice.index.SliceIndexRangeUsizeSlice α)) a { start := st, «end» := e }
        = Result.ok (s, back) ∧
      s.val = a.val.slice st.val e.val ∧ s.length = e.val - st.val ∧
      (∀ x, (back x).val = a.val.setSlice! st.val x.val) := by
  obtain ⟨x, hx, hspec⟩ :=
    WP.spec_imp_exists (Array.index_mut_SliceIndexRangeUsizeSlice.step a { start := st, «end» := e }
      (by rw [UScalar.le_equiv]; exact hst) (by rw [UScalar.le_equiv]; exact he))
  exact ⟨x.1, x.2, hx, hspec.1, hspec.2.1, fun y => hspec.2.2 y⟩

/-- `copy_from_slice` of two equal-length slices never fails. `proved`. -/
theorem copy_from_slice_ok (dst src : Slice Std.U8) (h : dst.length = src.length) :
    core.slice.Slice.copy_from_slice core.marker.CopyU8 dst src = Result.ok src := by
  obtain ⟨x, hx, hspec⟩ :=
    WP.spec_imp_exists (core.slice.Slice.copy_from_slice.step_spec core.marker.CopyU8 dst src h)
  rw [hspec] at hx
  exact hx

/-- A pair-destructuring monadic `let` is its body at the pair's components. `proved`. -/
theorem bind_tc_pair {α β γ : Type} (x : α × β) (f : α → β → Result γ) :
    (do let (a, b) ← Result.ok x; f a b) = f x.1 x.2 := by
  simp only [Bind.bind, Result.ok]
  exact bind_ok x (uncurry fun a b => f a b)

/-- `Array.to_slice (U32.to_be_bytes w)` carries the mapped `toBEBytes` list. `proved`. -/
theorem to_be_bytes_slice (w : Std.U32) :
    (Array.to_slice (core.num.U32.to_be_bytes w)).val
      = w.bv.toBEBytes.map (@UScalar.mk UScalarTy.U8) := by
  rw [Array.val_to_slice]
  simp [core.num.U32.to_be_bytes, Array.from_val]

/-- `Array.to_slice (U32.to_be_bytes w)` has length four, the length of the slice
`out[4i … 4i+4]` the body takes. `proved`. -/
theorem to_be_bytes_slice_len (w : Std.U32) :
    (Array.to_slice (core.num.U32.to_be_bytes w)).length = 4 := by
  simp [core.num.U32.to_be_bytes, Array.from_val, List.length_map, BitVec.toBEBytes_length]

/-- The mapped `toBEBytes` list a 32-bit word produces has length four: the width of the
`setSlice!` the body performs. `proved`. -/
theorem beBytes_len (b : BitVec 32) : (b.toBEBytes.map (@UScalar.mk UScalarTy.U8)).length = 4 := by
  rw [List.length_map, BitVec.toBEBytes_length]

/-! ## §3 · The body step -/

/-- **One body step.** For `i.val < 8` the extracted body succeeds. It returns an array `out1` and a
counter `i4` with `i4.val = i.val + 1`; `out1` has the four big-endian bytes of
`stateGet state i.val` at byte positions `4i … 4i+3`, agrees with the incoming array at every
position below `4i`, and agrees with it at every position at or above `4i+4`. `proved`. -/
theorem storeBody_step (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize)
    (i : Std.Usize) (hlt : i.val < 8) :
    ∃ out1 : Array Std.U8 32#usize, ∃ i4 : Std.Usize,
      sha256.sha256_loop2.body state out i = Result.ok (ControlFlow.cont (out1, i4)) ∧
      i4.val = i.val + 1 ∧
      (∀ j, j < 4 * i.val →
        (out1.val.getD j (0#u8) : Std.U8).bv = (out.val.getD j (0#u8) : Std.U8).bv) ∧
      (∀ j, 4 * i.val + 4 ≤ j →
        (out1.val.getD j (0#u8) : Std.U8).bv = (out.val.getD j (0#u8) : Std.U8).bv) ∧
      (out1.val.getD (4 * i.val) (0#u8) : Std.U8).bv = beByte (stateGet state i.val) 0 ∧
      (out1.val.getD (4 * i.val + 1) (0#u8) : Std.U8).bv = beByte (stateGet state i.val) 1 ∧
      (out1.val.getD (4 * i.val + 2) (0#u8) : Std.U8).bv = beByte (stateGet state i.val) 2 ∧
      (out1.val.getD (4 * i.val + 3) (0#u8) : Std.U8).bv = beByte (stateGet state i.val) 3 := by
  have h4v : ((4#usize : Std.Usize).val) = 4 := rfl
  have h1v : ((1#usize : Std.Usize).val) = 1 := rfl
  have h32v : ((32#usize : Std.Usize).val) = 32 := rfl
  have h8v : ((8#usize : Std.Usize).val) = 8 := rfl
  have hlt64 : i.val < 64 := by omega
  have h1bound : (4#usize : Std.Usize).val * i.val ≤ Usize.max := by
    rw [h4v]; simp [Usize.max, Usize.numBits];
        cases System.Platform.numBits_eq <;> simp_all <;> omega
  obtain ⟨i1, hi1, hi1v⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 4#usize) (y := i) h1bound)
  have hi1v' : i1.val = 4 * i.val := by rw [hi1v, h4v, Nat.mul_comm]
  have h2bound : i1.val + (4#usize : Std.Usize).val ≤ Usize.max := by
    rw [h4v]; simp [Usize.max, Usize.numBits];
        cases System.Platform.numBits_eq <;> simp_all <;> omega
  obtain ⟨i2, hi2, hi2v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i1) (y := 4#usize) h2bound)
  have hi2v' : i2.val = 4 * i.val + 4 := by rw [hi2v, h4v, hi1v']
  have hst_le : i1 ≤ i2 := by rw [UScalar.le_equiv, hi1v', hi2v']; omega
  have he_le : i2 ≤ (32#usize : Std.Usize) := by
    rw [UScalar.le_equiv, hi2v', h32v]
    omega
  obtain ⟨s, back, hidxr, _, hslen, hback⟩ :=
    index_mut_range_ok out i1 i2 hst_le he_le
  have hslice_len : s.length = 4 := by rw [hslen, hi2v', hi1v']; omega
  have hsl8 : state.val.length = 8 := by simpa using state.property
  have hidx_lt : i.val < state.val.length := by rw [hsl8]; exact hlt
  have hidx := array_index_usize_eq state i hidx_lt
  have hcopy : core.slice.Slice.copy_from_slice core.marker.CopyU8 s
      (Array.to_slice (core.num.U32.to_be_bytes (state.val[i.val]'hidx_lt)))
      = Result.ok (Array.to_slice (core.num.U32.to_be_bytes (state.val[i.val]'hidx_lt))) :=
    copy_from_slice_ok _ _ (by rw [hslice_len, to_be_bytes_slice_len])
  have h4bound : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    rw [h1v]; simp [Usize.max, Usize.numBits];
        cases System.Platform.numBits_eq <;> simp_all <;> omega
  obtain ⟨i4, hi4, hi4v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) h4bound)
  have hi4v' : i4.val = i.val + 1 := by rw [hi4v, h1v]
  set out1 : Array Std.U8 32#usize :=
    back (Array.to_slice (core.num.U32.to_be_bytes (state.val[i.val]'hidx_lt))) with hout1def
  have hout1 : out1.val = out.val.setSlice! i1.val
      ((state.val[i.val]'hidx_lt).bv.toBEBytes.map (@UScalar.mk UScalarTy.U8)) := by
    rw [hout1def, hback, to_be_bytes_slice]
  have houtlen : out.val.length = 32 := by simpa using out.property
  have hbyteslen : ((state.val[i.val]'hidx_lt).bv.toBEBytes.map
      (@UScalar.mk UScalarTy.U8)).length = 4 := beBytes_len _
  have hget_pre {j : Nat} (hj : j < i1.val) :
      (out1.val.getD j (0#u8) : Std.U8).bv = (out.val.getD j (0#u8) : Std.U8).bv := by
    rw [hout1, List.getD_eq_getElem?_getD, List.setSlice!_getElem?_prefix _ _ _ _ hj,
      List.getD_eq_getElem?_getD]
  have hget_suf {j : Nat} (hj : i1.val + 4 ≤ j) :
      (out1.val.getD j (0#u8) : Std.U8).bv = (out.val.getD j (0#u8) : Std.U8).bv := by
    rw [hout1]
    rw [List.getD_eq_getElem?_getD,
      List.setSlice!_getElem?_suffix _ _ _ _ (by rw [hbyteslen]; exact hj),
      List.getD_eq_getElem?_getD]
  have hget_mid {j : Nat} (hj1 : i1.val ≤ j) (hj2 : j - i1.val < 4) :
      (out1.val.getD j (0#u8) : Std.U8).bv
        = ((state.val[i.val]'hidx_lt).bv.toBEBytes.getD (j - i1.val) (0 : BitVec 8) : BitVec 8) := by
    rw [hout1]
    rw [List.getD_eq_getElem?_getD,
      List.setSlice!_getElem?_middle _ _ _ _
        ⟨hj1, by rw [hbyteslen]; exact hj2,
          by rw [houtlen]; omega⟩,
      List.getD_eq_getElem?_getD]
    show (((List.map (@UScalar.mk UScalarTy.U8) ((state.val[i.val]'hidx_lt).bv.toBEBytes)).getD
        (j - i1.val) (UScalar.mk 0) : Std.U8).bv)
      = ((state.val[i.val]'hidx_lt).bv.toBEBytes.getD (j - i1.val) 0 : BitVec 8)
    rw [List.getD_map]
  have hsb : (state.val[i.val]'hidx_lt).bv = stateGet state i.val := by
    show (state.val[i.val]'hidx_lt).bv = _
    simp only [stateGet, getElem_getD _ _ hidx_lt (0#u32)]
  have hmid (k : Nat) (hk : k < 4) :
      (out1.val.getD (4 * i.val + k) (0#u8) : Std.U8).bv = beByte (stateGet state i.val) k := by
    rw [hget_mid (by rw [hi1v']; omega) (by rw [hi1v']; omega), hsb, beByte_getD]
    show ((stateGet state i.val).toBEBytes.getD ((4 * i.val + k) - i1.val) (0 : BitVec 8)
        : BitVec 8) = (stateGet state i.val).toBEBytes.getD k (0 : BitVec 8)
    congr 1
    omega
  refine ⟨out1, i4, ?_, hi4v', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · unfold sha256.sha256_loop2.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
    simp only [hi1, hi2, hi4, hidx, hcopy, lift, out1, bind_tc_ok]
    rw [hidxr]
    simp only [bind_tc_pair, bind_tc_ok]
    show (do
      let s2 ← core.slice.Slice.copy_from_slice core.marker.CopyU8 s
        (Array.to_slice (core.num.U32.to_be_bytes (state.val[i.val]'hidx_lt)))
      ok (ControlFlow.cont (back s2, i4)))
      = ok (ControlFlow.cont (out1, i4))
    rw [hcopy]
    simp only [bind_tc_ok]
    exact congrArg (fun y => ok (ControlFlow.cont (y, i4))) hout1def.symm
  · intro j hj
    exact hget_pre (by rw [hi1v']; exact hj)
  · intro j hj
    exact hget_suf (by rw [hi1v']; exact hj)
  · exact hmid 0 (by omega)
  · exact hmid 1 (by omega)
  · exact hmid 2 (by omega)
  · exact hmid 3 (by omega)

/-- **The `done` branch.** At `8 ≤ i.val` the body returns the incoming array unchanged. Together
with `storeBody_step` this is the pair of branch witnesses obligation 4 asks of a partial operation.
`proved`. -/
theorem storeBody_done (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize)
    (i : Std.Usize) (hge : 8 ≤ i.val) :
    sha256.sha256_loop2.body state out i = Result.ok (ControlFlow.done out) := by
  unfold sha256.sha256_loop2.body
  rw [if_neg (by
    show ¬ i.val < (8#usize : Std.Usize).val
    have h8 : (8#usize : Std.Usize).val = 8 := by norm_num
    omega)]

/-- **The body has no failure path**, at any index: below the guard it succeeds with the written
array, above it with `done`. This is the admissibility statement for the `Result` the body returns.
`proved`. -/
theorem storeBody_total (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize)
    (i : Std.Usize) :
    (∃ out1 i4, sha256.sha256_loop2.body state out i = Result.ok (ControlFlow.cont (out1, i4)))
      ∨ sha256.sha256_loop2.body state out i = Result.ok (ControlFlow.done out) := by
  by_cases hlt : i.val < 8
  · exact Or.inl (Exists.elim (storeBody_step state out i hlt)
      (fun out1 h2 => Exists.elim h2 (fun i4 hP => ⟨out1, i4, hP.1⟩)))
  · exact Or.inr (storeBody_done state out i (by omega))

/-! ## §4 · The preserved-prefix invariant

After `i` visits, word `j` for every `j < i` is stored big-endian in bytes `4j … 4j+3`, and nothing
else is carried. The incoming array the extracted pipeline produces is `Array.repeat 32#usize 0#u8`
(`ApeironWord.lean:15589`), but the invariant does not use that: it holds for every incoming array,
because each visit writes only its own four bytes. -/

/-- The canonical byte `byte32` reads at `j`, unfolded to the list read the body step gives.
`proved`. -/
theorem byte32_getD (a : Array Std.U8 32#usize) (j : Nat) :
    (byte32 a j).toNat = ((a.val.getD j (0#u8) : Std.U8).bv).toNat := rfl

/-- The canonical word `beWord` reads at `k` is the big-endian regrouping of the four bytes
`4k … 4k+3` (`Sha256Certificate.lean:144-155`). `proved`. -/
theorem beWord_get (a : Array Std.U8 32#usize) (k : Fin 8) :
    beWord a k = BitVec.ofNat 32
        ((byte32 a (4 * k.val)).toNat * 16777216
          + (byte32 a (4 * k.val + 1)).toNat * 65536
          + (byte32 a (4 * k.val + 2)).toNat * 256
          + (byte32 a (4 * k.val + 3)).toNat) := rfl

/-- The loop invariant at the counter `p.2`: the counter is at most eight, and every word below it is
already stored big-endian. The right-hand side is spelled at the `toNat` level so that the body
step's byte reads compose with it by rewriting. -/
def storeInv (state : Array Std.U32 8#usize)
    (p : Array Std.U8 32#usize × Std.Usize) : Prop :=
  p.2.val ≤ 8 ∧
    ∀ j : Fin 8, j.val < p.2.val →
      BitVec.ofNat 32
        (((p.1.val.getD (4 * j.val) (0#u8) : Std.U8).bv).toNat * 16777216
          + ((p.1.val.getD (4 * j.val + 1) (0#u8) : Std.U8).bv).toNat * 65536
          + ((p.1.val.getD (4 * j.val + 2) (0#u8) : Std.U8).bv).toNat * 256
          + ((p.1.val.getD (4 * j.val + 3) (0#u8) : Std.U8).bv).toNat)
        = beWordStored (stateGet state j.val)

/-- **Initialization.** At counter zero the written prefix is empty, so the invariant holds for
every state and every incoming array — no hypothesis. `proved`. -/
theorem storeInv_init (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize) :
    storeInv state (out, 0#usize) := by
  refine ⟨?_, ?_⟩
  · show (0#usize : Std.Usize).val ≤ 8
    exact Nat.zero_le 8
  · intro j hj
    exact absurd hj (by
      show ¬ j.val < ((out, 0#usize) : Array Std.U8 32#usize × Std.Usize).2.val
      have h0 : ((out, 0#usize) : Array Std.U8 32#usize × Std.Usize).2.val = 0 := by norm_num
      omega)

/-- **The invariant at the spelled counter eight.** If the counter's value is eight, the invariant
at the pair is the invariant at `(a, 8#usize)`: only the counter's value is used. This is the
transport the loop's `done` branch needs, where the counter is known only by its value. `proved`. -/
theorem storeInv_eight {state : Array Std.U32 8#usize}
    {p : Array Std.U8 32#usize × Std.Usize} (hp : storeInv state p) (h8 : p.2.val = 8) :
    storeInv state (p.1, 8#usize) := by
  obtain ⟨_, hwords⟩ := hp
  have h8v : (8#usize : Std.Usize).val = 8 := by norm_num
  refine ⟨?_, ?_⟩
  · show (8#usize : Std.Usize).val ≤ 8
    rw [h8v]
  · intro j hj
    have hj' : j.val < (8#usize : Std.Usize).val := hj
    rw [h8v] at hj'
    exact hwords j (by omega)

/-- **One body step preserves the invariant.** A visit at `i.val < 8` writes word `i` in full and
leaves bytes below `4i` alone, so it extends the written prefix from `i.val` to `i.val + 1` without
disturbing the earlier words. `proved`. -/
theorem storeBody_step_pres (state : Array Std.U32 8#usize)
    (p : Array Std.U8 32#usize × Std.Usize) (hinv : storeInv state p)
    (hlt : p.2.val < 8) :
    sha256.sha256_loop2.body state p.1 p.2
      ⦃ r => match r with
          | ControlFlow.done _ => True
          | ControlFlow.cont x' => storeInv state x' ⦄ := by
  obtain ⟨hle, hwords⟩ := hinv
  obtain ⟨out1, i4, hbody, hi4v, hpre, hsuf, hb0, hb1, hb2, hb3⟩ :=
    storeBody_step state p.1 p.2 hlt
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out1, i4), hbody, ?_⟩
  change storeInv state (out1, i4)
  change i4.val ≤ 8 ∧ ∀ j : Fin 8, j.val < i4.val →
    BitVec.ofNat 32
      (((out1.val.getD (4 * j.val) (0#u8) : Std.U8).bv).toNat * 16777216
        + ((out1.val.getD (4 * j.val + 1) (0#u8) : Std.U8).bv).toNat * 65536
        + ((out1.val.getD (4 * j.val + 2) (0#u8) : Std.U8).bv).toNat * 256
        + ((out1.val.getD (4 * j.val + 3) (0#u8) : Std.U8).bv).toNat)
      = beWordStored (stateGet state j.val)
  refine ⟨?_, ?_⟩
  · omega
  · intro j hj
    have hi4v'' : i4.val = p.2.val + 1 := by rw [hi4v]
    rw [hi4v''] at hj
    by_cases hji : j.val < p.2.val
    · have hbytes : ∀ (t : Nat) (ht : t < 4),
          ((out1.val.getD (4 * j.val + t) (0#u8) : Std.U8).bv).toNat
            = ((p.1.val.getD (4 * j.val + t) (0#u8) : Std.U8).bv).toNat := by
        intro t ht
        exact congrArg BitVec.toNat (hpre _ (by omega))
      rw [beWordStored]
      have hb0' := hbytes 0 (by omega)
      have hb1' := hbytes 1 (by omega)
      have hb2' := hbytes 2 (by omega)
      have hb3' := hbytes 3 (by omega)
      rw [show (4 * j.val + 0 : Nat) = 4 * j.val from rfl] at hb0'
      rw [hb0', hb1', hb2', hb3']
      exact hwords j hji
    · have hj_eq : j.val = p.2.val := by omega
      rw [hj_eq]
      rw [show ((out1.val.getD (4 * p.2.val) (0#u8) : Std.U8).bv).toNat
          = (beByte (stateGet state p.2.val) 0).toNat from congrArg BitVec.toNat hb0,
        show ((out1.val.getD (4 * p.2.val + 1) (0#u8) : Std.U8).bv).toNat
          = (beByte (stateGet state p.2.val) 1).toNat from congrArg BitVec.toNat hb1,
        show ((out1.val.getD (4 * p.2.val + 2) (0#u8) : Std.U8).bv).toNat
          = (beByte (stateGet state p.2.val) 2).toNat from congrArg BitVec.toNat hb2,
        show ((out1.val.getD (4 * p.2.val + 3) (0#u8) : Std.U8).bv).toNat
          = (beByte (stateGet state p.2.val) 3).toNat from congrArg BitVec.toNat hb3]
      exact (beWordStored_beBytes (stateGet state p.2.val)).symm

/-! ## §5 · The extracted loop -/

/-- The extracted loop as a `loop` over the body, read off the generated definition
(`ApeironWord.lean:15541-15548`). `proved`, `rfl`. -/
theorem sha256_loop2_eq (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize)
    (i : Std.Usize) :
    sha256.sha256_loop2 state out i
      = loop (fun p : Array Std.U8 32#usize × Std.Usize => sha256.sha256_loop2.body state p.1 p.2)
          (out, i) := rfl

/-- The invariant `storeInv` read off at the terminal counter eight, in the `digView`/`stateView`
form the loop postcondition asks for. The hypothesis is the invariant itself: the statement is not a
kernel identity of the array (a store that wrote different bytes satisfies it only by failing the
invariant). `proved`. -/
theorem storeInv_getD (state : Array Std.U32 8#usize)
    (a : Array Std.U8 32#usize) (j : Fin 8)
    (hp : storeInv state (a, 8#usize)) :
    beWord a j = stateGet state j.val := by
  have hb := beWord_get a j
  rw [byte32_getD, byte32_getD, byte32_getD, byte32_getD] at hb
  rw [hb]
  have h8 : (8#usize : Std.Usize).val = 8 := by norm_num
  exact (hp.2 j (by rw [h8]; exact j.isLt)).trans (beWordStored_eq _)

/-- The measure strictly decreases across one body step: the counter rises from `i.val` to
`i.val + 1` while `i.val < 8`. `proved`. -/
theorem measure_dec (p : Array Std.U8 32#usize × Std.Usize) (i4 : Std.Usize)
    (hlt : p.2.val < 8) (hi4 : i4.val = p.2.val + 1) : 8 - i4.val < 8 - p.2.val := by
  rw [hi4]
  omega

/-- **The terminal visit.** At counter eight the invariant says every word is stored big-endian, and
the regrouping closes into the state word, so the digest view of the array is the state view.
`proved`. -/
theorem storeInv_post (state : Array Std.U32 8#usize) (a : Array Std.U8 32#usize)
    (hp : storeInv state (a, 8#usize)) : digView a = stateView state := by
  rw [digView, stateView]
  refine vec_getD_eq _ _ 0 fun j hj => ?_
  rw [ofFn_getD', ofFn_getD']
  exact storeInv_getD state a ⟨j, hj⟩ hp

/-- **The store loop.** For every eight-word state and every 32-byte starting array,
`sha256_loop2` run from counter zero succeeds, and the canonical digest view `digView` of its result
is the canonical state view `stateView` of the input state. `proved`. -/
theorem store_loop2 (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize)
    (out' : Array Std.U8 32#usize)
    (h : sha256.sha256_loop2 state out 0#usize = Result.ok out') :
    digView out' = stateView state := by
  have hloop :
      loop (fun p : Array Std.U8 32#usize × Std.Usize => sha256.sha256_loop2.body state p.1 p.2)
        (out, 0#usize) ⦃ y => digView y = stateView state ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : Array Std.U8 32#usize × Std.Usize => 8 - p.2.val)
      (inv := storeInv state)
      (post := fun y => digView y = stateView state)
    · intro p hp
      by_cases hlt : p.2.val < 8
      · obtain ⟨out1, i4, hb, hi4v, _, _, _, _, _, _⟩ := storeBody_step state p.1 p.2 hlt
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        show storeInv state (out1, i4) ∧ 8 - i4.val < 8 - p.2.val
        have hpres := storeBody_step_pres state p hp hlt
        have hpres' : storeInv state (out1, i4) := by
          rw [hb] at hpres
          simpa only [WP.spec_ok] using hpres
        refine ⟨hpres', ?_⟩
        rw [hi4v]
        omega
      · rw [WP.spec_equiv_exists]
        have h8 : p.2.val = 8 := by
          have hle : p.2.val ≤ 8 := hp.1
          omega
        exact ⟨ControlFlow.done p.1, storeBody_done state p.1 p.2 (by omega),
          storeInv_post state p.1 (storeInv_eight hp h8)⟩
    · exact storeInv_init state out
  have hloop' : loop (fun p : Array Std.U8 32#usize × Std.Usize =>
        sha256.sha256_loop2.body state p.1 p.2) (out, 0#usize) = Result.ok out' := by
    simpa only [sha256_loop2_eq] using h
  obtain ⟨y, hy, hPy⟩ := WP.spec_imp_exists hloop
  rw [hloop'] at hy
  have hw'y : out' = y := Result.ok_injective hy
  rw [← hw'y] at hPy
  exact hPy

/-! ## §6 · The two headline shapes, kept apart -/

/-- **Correctness conditioned on success.** If the extracted store loop succeeds, its result's
canonical digest view is the canonical state view of the input. This uses no existence argument:
it is the content of any successful result, so it is exactly the shape that must be distinguished
from total existence. `proved`. -/
theorem store_spec_of_ok (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize)
    (out' : Array Std.U8 32#usize)
    (h : sha256.sha256_loop2 state out 0#usize = Result.ok out') :
    digView out' = stateView state :=
  store_loop2 state out out' h

/-- **Total existence — the digest-store link.** For every eight-word state and every 32-byte
starting array there **is** a successful result of the store loop whose canonical digest view
`digView` **is** the canonical state view `stateView` of the input. The body never fails below the
guard and the guard stops the loop at eight, so the `loop` rule inhabits its own postcondition; no
hypothesis bundle is used. `proved`. -/
theorem store_spec_exists (state : Array Std.U32 8#usize) (out : Array Std.U8 32#usize) :
    ∃ out' : Array Std.U8 32#usize,
      sha256.sha256_loop2 state out 0#usize = Result.ok out' ∧
      digView out' = stateView state := by
  have hloop :
      loop (fun p : Array Std.U8 32#usize × Std.Usize => sha256.sha256_loop2.body state p.1 p.2)
        (out, 0#usize) ⦃ y => digView y = stateView state ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : Array Std.U8 32#usize × Std.Usize => 8 - p.2.val)
      (inv := storeInv state)
      (post := fun y => digView y = stateView state)
    · intro p hp
      by_cases hlt : p.2.val < 8
      · obtain ⟨out1, i4, hb, hi4v, _, _, _, _, _, _⟩ := storeBody_step state p.1 p.2 hlt
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        show storeInv state (out1, i4) ∧ 8 - i4.val < 8 - p.2.val
        have hpres := storeBody_step_pres state p hp hlt
        have hpres' : storeInv state (out1, i4) := by
          rw [hb] at hpres
          simpa only [WP.spec_ok] using hpres
        refine ⟨hpres', ?_⟩
        rw [hi4v]
        omega
      · rw [WP.spec_equiv_exists]
        have h8 : p.2.val = 8 := by
          have hle : p.2.val ≤ 8 := hp.1
          omega
        exact ⟨ControlFlow.done p.1, storeBody_done state p.1 p.2 (by omega),
          storeInv_post state p.1 (storeInv_eight hp h8)⟩
    · exact storeInv_init state out
  obtain ⟨y, hy, hPy⟩ := WP.spec_imp_exists hloop
  exact ⟨y, hy, hPy⟩

/-! ## §7 · The byte-order discriminator, as a closed refutation

`store_spec_exists` links two views of the same array; on its own it would not distinguish a
big-endian store from a byte-reversed one, because both are *some* equality. `leWord4_ne` (§1)
closes that gap as a **kernel-checked refutation**: for four bytes whose outermost differ, the two
readings are unequal words, so the store's byte order is pinned by the link and not merely present
in it. `stored_word_eq` is the same statement in the form the loop's invariant consumes. -/

/-- **The store's byte order is pinned by the regrouping.** The word `store_spec_exists` reads back
through `digView` is the input word itself, and `leWord4_ne` refutes the byte-reversed reading, so
no permutation of the four bytes of a word can satisfy it when the outermost bytes differ. This is
the pair the receipt's byte-order control separates. `proved`. -/
theorem stored_word_eq (w : BitVec 32) : beWordStored w = w := beWordStored_eq w

/-- **A byte-reversed store is refuted, as a word inequality.** Instantiating `leWord4_ne` at the
four bytes `w`'s `toBEBytes` produces: reading them in the opposite order yields a different word.
`proved`. -/
theorem reversed_bytes_regrouped (w : BitVec 32) (hne : (beByte w 0).toNat ≠ (beByte w 3).toNat) :
    BitVec.ofNat 32
        ((beByte w 3).toNat * 16777216 + (beByte w 2).toNat * 65536
          + (beByte w 1).toNat * 256 + (beByte w 0).toNat)
      ≠ BitVec.ofNat 32
        ((beByte w 0).toNat * 16777216 + (beByte w 1).toNat * 65536
          + (beByte w 2).toNat * 256 + (beByte w 3).toNat) := by
  refine leWord4_ne _ _ _ _ ?_ ?_ ?_ ?_ hne
  · have := bnd8 (beByte w 0); simpa using this
  · have := bnd8 (beByte w 1); simpa using this
  · have := bnd8 (beByte w 2); simpa using this
  · have := bnd8 (beByte w 3); simpa using this

/-! ## §8 · Axiom cone -/

#print axioms beByte_getD
#print axioms beWordStored_eq
#print axioms beWordStored_beBytes
#print axioms be_lo
#print axioms le_lo
#print axioms regroup_diff
#print axioms leWord4_ne
#print axioms word4_bound
#print axioms index_mut_range_ok
#print axioms copy_from_slice_ok
#print axioms bind_tc_pair
#print axioms to_be_bytes_slice
#print axioms to_be_bytes_slice_len
#print axioms beBytes_len
#print axioms storeBody_step
#print axioms storeBody_done
#print axioms storeBody_total
#print axioms byte32_getD
#print axioms beWord_get
#print axioms storeInv_init
#print axioms storeInv_eight
#print axioms storeBody_step_pres
#print axioms sha256_loop2_eq
#print axioms storeInv_getD
#print axioms measure_dec
#print axioms storeInv_post
#print axioms store_loop2
#print axioms store_spec_of_ok
#print axioms store_spec_exists
#print axioms stored_word_eq
#print axioms reversed_bytes_regrouped

end Store
end Extraction
end Icarus