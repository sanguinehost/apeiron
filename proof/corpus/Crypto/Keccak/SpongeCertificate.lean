/-
  Crypto/Keccak/SpongeCertificate.lean

  Stage 1 of the `word_sha3_256` refinement: the state/byte conversion layer.

  `proof/ApeironWord.lean` carries the extraction of `apeiron_word::word_sha3_256`
  (`crypto.sha3_256` -> `crypto.sponge`), whose data lives in `Array Std.U64 25#usize`,
  `Array Std.U8` and `Slice Std.U8`. `Crypto/Sponge.lean` carries the canonical
  FIPS-202 sponge over `Vector (BitVec 64) 25` with messages as `List (BitVec 8)`.
  The two sides live at different types, so this file carries the one-way
  conversions between them and the arithmetic that pins the extracted constants
  to the canonical ones:

  * `byteView` / `sliceView`: `Array Std.U8` and `Slice Std.U8` read as
    `List (BitVec 8)` (elementwise `U8.bv`, the model's own byte view);
  * `stateView`: `Array Std.U64 25#usize` read as the canonical `State`, with
    `stateView_zero` fixing the `Array.repeat 25 0` initial state;
  * `laneBytes` / `foldLane` / `byteOf_eq_laneBytes_get`: the little-endian lane
    packing in both shapes - `byteOf`'s `extractLsb' (8 * j) 8` and the
    `List.range 8` fold of `xorBlock` - plus the extracted `U64::from_le_bytes`
    read, which the model defines as `BitVec.fromLEBytes` of the byte view, with
    `byteWindow` / `foldLane_eq_fromLEBytes` giving that equality for any offset
    (`List.extract` is start/stop and returns `List α` in v4.31.0, so `extract o 8`
    truncates for `o > 0` and is not the window statement);
  * `rate_256_denote`, `rate_512_denote`, `domain_denote`, `pad_final_denote`:
    the extracted `SHA3_256_RATE` / `SHA3_512_RATE` / `6#u8` / `128#u8` against
    `rate_sha3_256` / `rate_sha3_512` / `domain_sha3` / `0x80`, with the rate
    offsets (lanes per block, `rate % 8 = 0` for the `massert`, `rate ≤ 200` for
    the padding buffer, and the `32 ≤ 136` squeeze bound of stage 5);
  * `WordSha3_256Denotes`: the stage-6 headline, carried as a `def` of a `Prop`
    and deliberately not asserted here.

  `#guard` pins run the untrusted evaluator; they guard byte order, lane order
  and rate arithmetic against regressions and are not kernel proofs.

  Open obligations, next stages: the absorb loop against
  `absorb_block` with the lane index `x + 5 * y`, the 24-round permutation fold
  (stage 3, alongside `Crypto/Keccak/Certificate.lean`), the padding merge
  (stage 4) and the squeeze bound (stage 5).

  `absorb_body_step` and `absorb_loop_spec` carry a finite `maxHeartbeats`
  bound (not `0`) for the extracted loop-body reductions; the bound is
  documented here and not applied anywhere else in the file.
-/

import Crypto.Sponge
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Keccak (Lane State rate_sha3_256 rate_sha3_512 domain_sha3 byteOf xorBlock
  sha3_256)

/-! ## Byte views

`Array Std.U8` is a `ListN` wrapper (`Array.val`, `Array.property`), `Slice Std.U8` is a
length-bounded `ListN` (`Slice.val`, `Slice.len`), and `U8.bv` is the model's own byte view.
The digest of the headline is an `Array Std.U8 32#usize` and the input is a `Slice Std.U8`, so
both are read through the same elementwise `List (BitVec 8)` view. -/

/-- An extracted byte array read as the canonical message/digest byte list. -/
def byteView {n : Std.Usize} (a : Array Std.U8 n) : List (BitVec 8) :=
  a.val.map Std.U8.bv

/-- An extracted byte slice read as the canonical message byte list. -/
def sliceView (s : Slice Std.U8) : List (BitVec 8) :=
  s.val.map Std.U8.bv

theorem byteView_length {n : Std.Usize} (a : Array Std.U8 n) :
    (byteView a).length = n.val := by
  simp [byteView, Array.property]

/-- The digest array of the headline has canonical length 32. -/
theorem byteView_digest_length (a : Array Std.U8 32#usize) :
    (byteView a).length = 32 := by
  have h : (byteView a).length = (32#usize).val := byteView_length a
  rw [h]
  decide

/-- Pointwise read: the extracted byte at `i` is the canonical byte at `i`. -/
theorem byteView_getD {n : Std.Usize} (a : Array Std.U8 n) (i : Nat) :
    (byteView a).getD i (Std.U8.bv (0#u8)) = (a.val.getD i (0#u8)).bv :=
  List.getD_map _ _ _

theorem byteView_repeat (n : Std.Usize) (b : Std.U8) :
    byteView (Array.repeat n b) = List.replicate n.val b.bv := by
  simp [byteView, Array.repeat_val]

theorem sliceView_length (s : Slice Std.U8) :
    (sliceView s).length = s.val.length := by
  simp [sliceView]

/-- The extracted `Slice::len` is the canonical message length. -/
theorem sliceView_len (s : Slice Std.U8) :
    (Std.Slice.len s).val = (sliceView s).length := by
  simp [Std.Slice.len_val, sliceView]

theorem sliceView_of_list (l : List Std.U8) (h : l.length ≤ Std.Usize.max) :
    sliceView (Std.Slice.from l h) = l.map Std.U8.bv := by
  simp [sliceView, Std.Slice.from, Std.Slice.val, Aeneas.Data.ListN.ListN.from_to_inverse]

/-! ## State view

`Array Std.U64 25#usize` is read lane by lane; `Crypto.Keccak.Certificate` carries the same
conversion under the name `view` for the permutation certificate. -/

/-- The extracted 25-lane state read as the canonical `Vector (BitVec 64) 25`. -/
def stateView (a : Array Std.U64 25#usize) : State :=
  Vector.ofFn fun i : Fin 25 => (a.val.getD i.val (0#u64)).bv

theorem stateView_get (a : Array Std.U64 25#usize) (i : Fin 25) :
    (stateView a).get i = (a.val.getD i.val (0#u64)).bv := by
  show (stateView a)[i.val] = _
  simp only [stateView, Vector.getElem_ofFn]

/-- The initial state of the extracted sponge is the canonical zero state. -/
theorem stateView_zero : stateView (Array.repeat 25#usize 0#u64) = Vector.replicate 25 0 := by
  apply Vector.ext
  intro i hi
  show (stateView (Array.repeat 25#usize 0#u64))[i] = _
  simp only [stateView, Array.repeat_val, Vector.getElem_ofFn]
  rw [List.getD_replicate _ (by simpa using hi)]
  simp

/-! ## Little-endian lane packing

FIPS-202 §3.1.2: state byte `k` is byte `k % 8` of lane `k / 8`, least significant byte first.
`laneBytes` is that lane view, `foldLane` is the `List.range 8` fold shape used by `xorBlock` and
by the extracted `absorb_block`, and `core.num.U64.from_le_bytes` is the extracted little-endian
64-bit read, which the model defines as `BitVec.fromLEBytes` of the byte view. -/

/-- Byte `j` of a lane, least significant first (FIPS-202 §3.1.2). -/
def laneByte (l : Lane) (j : Nat) : BitVec 8 := l.extractLsb' (8 * j) 8

/-- The eight canonical bytes of a lane, least significant first. -/
def laneBytes (l : Lane) : List (BitVec 8) :=
  (List.range 8).map (laneByte l)

theorem laneBytes_length (l : Lane) : (laneBytes l).length = 8 := rfl

theorem laneBytes_getD (l : Lane) (j : Nat) (hj : j < 8) :
    (laneBytes l).getD j (laneByte l 0) = laneByte l j := by
  simp [laneBytes]
  rw [List.getElem?_range (by simpa using hj)]
  rfl

/-- Lane `(k / 8) % 25` of the canonical state, the lane `byteOf` reads at state byte `k`. -/
abbrev laneAt (s : State) (k : Nat) : Lane :=
  s.get ⟨(k / 8) % 25, Nat.mod_lt (k / 8) (by decide : (0 : Nat) < 25)⟩

/-- `byteOf` is the FIPS-202 §3.1.2 lane byte view of the canonical state: state byte `k` is byte
`k % 8` of lane `(k / 8) % 25`, least significant byte first. -/
theorem byteOf_eq_laneBytes_getD (s : State) (k : Nat) :
    byteOf s k = (laneBytes (laneAt s k)).getD (k % 8) (laneByte (laneAt s k) 0) := by
  simp only [byteOf, laneAt]
  exact (laneBytes_getD _ _ (Nat.mod_lt _ (by decide : (0 : Nat) < 8))).symm

/-- The `List.range 8` fold that packs a byte window into a lane, in the shape used by
`xorBlock` and by the extracted `absorb_block` (`U64::from_le_bytes`). -/
def foldLane (bs : List (BitVec 8)) (o : Nat) : Lane :=
  (List.range 8).foldl
    (fun acc j => acc ^^^ ((bs.getD (o + j) 0).setWidth 64 <<< (8 * j))) 0

/-- The eight-byte window of `bs` at offset `o`, read out of bounds as zero, as the `Fin 8`
function the `fromLEBytes` shape consumes. -/
def byteWindow (bs : List (BitVec 8)) (o : Nat) : List (BitVec 8) :=
  List.ofFn fun j : Fin 8 => bs.getD (o + j.val) 0

/-- `getElem!` on a byte list is the out-of-bounds-zero `getD`. -/
theorem byte_getElem_eq_getD (l : List (BitVec 8)) (k : Nat) :
    l[k]! = List.getD l k 0 := by
  simp only [List.getElem!_eq_getElem?_getD, List.getD_eq_getElem?_getD]
  rfl

/-- Pointwise read of a byte window: entry `k` is the byte of `bs` at `o + k`. -/
theorem byteWindow_getD (bs : List (BitVec 8)) (o k : Nat) (hk : k < 8) :
    List.getD (byteWindow bs o) k 0 = bs.getD (o + k) 0 := by
  simp only [byteWindow, List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  simp [hk]

/-- The `List.range 8` fold over a byte window at offset `o` is the little-endian read
`BitVec.fromLEBytes` of that window, cast to the canonical 64-bit lane. This is the general
form of `from_le_bytes_denote`, which pins only the eight-byte array. -/
theorem foldLane_eq_fromLEBytes (bs : List (BitVec 8)) (o : Nat) :
    foldLane bs o
      = (BitVec.fromLEBytes (byteWindow bs o)).cast (by simp [byteWindow]) := by
  rw [BitVec.eq_iff]
  intro i hi
  simp only [BitVec.getElem!_cast, BitVec.fromLEBytes_getElem!]
  rw [byte_getElem_eq_getD, byteWindow_getD _ _ _ (by omega)]
  -- A byte contributes no bit at index `8 ≤ k`: it is below `2 ^ 8`.
  have key : ∀ (x : BitVec 8) (k : Nat), 8 ≤ k → x.toNat.testBit k = false := by
    intro x k hk
    exact Nat.testBit_lt_two_pow
      (lt_of_lt_of_le x.isLt (Nat.pow_le_pow_right Nat.zero_lt_two hk))
  -- One fold step contributes the bits of byte `j` at indices `8 * j … 8 * j + 7`, so it survives
  -- only at `j = i / 8`, and there it contributes bit `i % 8`.
  have step : ∀ (j : Nat), j < 8 →
      (decide (i ≥ 8 * j) && (decide (i - 8 * j < 64)
        && (bs.getD (o + j) 0).toNat.testBit (i - 8 * j)))
        = if j = i / 8 then (bs.getD (o + i / 8) 0).toNat.testBit (i % 8) else false := by
    intro j hj
    rw [decide_eq_true (show i - 8 * j < 64 by omega), Bool.true_and]
    by_cases h : j = i / 8
    · rw [if_pos h, decide_eq_true (show i ≥ 8 * j by omega), Bool.true_and, h]
      rw [show i - 8 * (i / 8) = i % 8 by omega]
    · rw [if_neg h]
      by_cases h2 : i ≥ 8 * j
      · rw [decide_eq_true h2, Bool.true_and]
        exact key (bs.getD (o + j) 0) (i - 8 * j) (by omega)
      · rw [decide_eq_false h2, Bool.false_and]
  have hzero : BitVec.toNat (0 : BitVec 64) = 0 := rfl
  simp only [BitVec.getElem!_eq_testBit_toNat, Byte.testBit, foldLane, List.range_succ,
    List.foldl, BitVec.toNat_xor, BitVec.toNat_shiftLeft, BitVec.toNat_setWidth,
    Nat.testBit_shiftLeft, Nat.testBit_xor, List.foldl_append, Nat.mod_eq_of_lt, BitVec.isLt,
    Nat.lt_of_lt_of_le, Nat.pow_le_pow_right, Nat.testBit_mod_two_pow, List.range_zero,
    List.foldl_nil, hzero, Nat.zero_testBit, decide_eq_true hi, Bool.true_and,
    step 0 (by omega), step 1 (by omega), step 2 (by omega), step 3 (by omega),
    step 4 (by omega), step 5 (by omega), step 6 (by omega), step 7 (by omega)]
  -- Exactly one `j < 8` satisfies `j = i / 8`, so the eight-fold xor is that one contribution.
  obtain ⟨q, hq, hq8⟩ : ∃ q : Nat, i / 8 = q ∧ q < 8 := ⟨i / 8, rfl, by omega⟩
  have hd : ∀ k : Nat, (i / 8 = k) ↔ k = q := by
    intro k
    constructor
    · intro h
      omega
    · intro h
      omega
  have hcases : q = 0 ∨ q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 6 ∨ q = 7 := by omega
  rcases hcases with hq0 | hq0 | hq0 | hq0 | hq0 | hq0 | hq0 | hq0 <;>
    rcases hq0 with rfl <;>
    simp_all [Bool.false_xor, Bool.xor_false, hd 0, hd 1, hd 2, hd 3, hd 4, hd 5, hd 6, hd 7]

/-- The extracted little-endian 64-bit lane read is `BitVec.fromLEBytes` of the byte view: the
same value `foldLane` computes, at the definitional level of the model. -/
theorem from_le_bytes_denote (a : Array Std.U8 8#usize) :
    (core.num.U64.from_le_bytes a).bv =
      (BitVec.fromLEBytes (byteView a)).cast (by simp [byteView_length, Array.property]) := rfl

/-! ## The absorb lane read

`crypto.absorb_block_loop.body` (3133-3165) reads the eight block bytes at `o = 8 * i` and hands them to
`core.num.U64.from_le_bytes`. This section pins that read to `foldLane` - the `List.range 8` fold over the
same eight bytes that the canonical `xorBlock` runs per lane. Both sides read out of bounds as zero, so the
statement carries no in-bounds hypothesis; the block length belongs to the loop invariant. -/

/-- The eight block bytes the extracted loop reads at offset `o`, read out of bounds as zero: the
`Fin 8` window the little-endian lane read consumes. -/
def blockBytes (block : Slice Std.U8) (o : Nat) : List Std.U8 :=
  List.ofFn fun j : Fin 8 => block.val.getD (o + j.val) (0#u8)

/-- The window the extracted loop reads is eight bytes long. -/
theorem blockBytes_length (block : Slice Std.U8) (o : Nat) : (blockBytes block o).length = 8 := rfl

/-- The eight block bytes at `o`, read through the byte view, are the entries of the byte window at
`o`, in the same order. -/
theorem blockBytes_view (block : Slice Std.U8) (o : Nat) :
    List.map Std.U8.bv (blockBytes block o) = byteWindow (sliceView block) o := by
  unfold blockBytes byteWindow sliceView
  rw [← List.ofFn_comp', List.ofFn_inj]
  funext j
  exact (List.getD_map _ _ _).symm

/-- Two `BitVec.fromLEBytes` of one byte list, cast to one width, agree whatever their width proofs:
`BitVec.cast` reads its proof only through `isLt`, and proofs are irrelevant. -/
theorem fromLEBytes_cast_eq {a b : List (BitVec 8)} {p : 8 * a.length = 64} {q : 8 * b.length = 64}
    (h : a = b) :
    (BitVec.fromLEBytes a).cast p = (BitVec.fromLEBytes b).cast q := by
  subst h
  simp only [BitVec.cast]

/-- `from_le_bytes_denote` in list form. Its width proof mentions only the list, so it rewrites a `U64`
value rather than a term sitting under a `BitVec.cast` whose type depends on the array. -/
theorem from_le_bytes_denote_list (l : List Std.U8) (h : l.length = 8) :
    (core.num.U64.from_le_bytes (Array.make 8#usize l h)).bv
      = (BitVec.fromLEBytes (l.map Std.U8.bv)).cast (by rw [List.length_map, h]) := by
  rw [from_le_bytes_denote]
  simp only [byteView]
  exact fromLEBytes_cast_eq
    (a := List.map Std.U8.bv ((↑(Array.make 8#usize l h) : List Std.U8)))
    (b := l.map Std.U8.bv) (by rw [Array.make_val])

/-- The extracted little-endian lane read of one absorb iteration is the `foldLane` the canonical
`xorBlock` computes for that lane: the eight block bytes at `o`, least significant byte first. -/
theorem absorbLane_eq_foldLane (block : Slice Std.U8) (o : Nat) :
    (core.num.U64.from_le_bytes
      (Array.make 8#usize (blockBytes block o) (blockBytes_length block o))).bv
      = foldLane (sliceView block) o := by
  rw [from_le_bytes_denote_list, foldLane_eq_fromLEBytes]
  exact fromLEBytes_cast_eq (a := List.map Std.U8.bv (blockBytes block o))
    (b := byteWindow (sliceView block) o) (blockBytes_view block o)

/-! ## Reading a lane through a write

`Array.update` (3182) returns `Array.set`; `Array.set` replaces one element of the underlying list, so
the canonical state read `stateView` sees exactly that lane replaced. `Certificate.lean` carries the
same fact for `view`; it is repeated here for `stateView`, which reads through `List.getD`. -/

/-- A written list element reads as the written value and every other element as before. -/
theorem list_getD_set {α : Type} (l : List α) (i j : Nat) (v d : α) (hi : i < l.length) :
    (l.set i v).getD j d = if j = i then v else l.getD j d := by
  by_cases h : j = i
  · subst h
    rw [List.getD_eq_getElem?_getD, List.getElem?_set, if_pos rfl, if_pos hi, if_pos rfl]
    rfl
  · rw [List.getD_eq_getElem?_getD, List.getElem?_set, if_neg (Ne.symm h),
      List.getD_eq_getElem?_getD, if_neg h]

/-- Pointwise read of `stateView` after a lane write: the written lane reads as the written word's
bit vector, every other lane is unchanged. -/
theorem stateView_set_get (a : Array Std.U64 25#usize) (i : Std.Usize) (v : Std.U64)
    (hi : i.val < 25) (j : Fin 25) :
    (stateView (a.set i v)).get j = if j.val = i.val then v.bv else (stateView a).get j := by
  show (stateView (a.set i v))[j.val] = if j.val = i.val then v.bv else (stateView a)[j.val]
  simp only [stateView, Vector.getElem_ofFn]
  show ((a.set i v).val.getD j.val (0#u64)).bv
      = if j.val = i.val then v.bv else (a.val.getD j.val (0#u64)).bv
  have ha : a.val.length = 25 := by simpa using a.property
  have hset : ((↑(a.set i v) : List Std.U64).getD j.val (0#u64))
      = ((↑a : List Std.U64).set i.val v).getD j.val (0#u64) :=
    congrArg (fun l : List Std.U64 => l.getD j.val (0#u64)) (Array.set_val_eq a i v)
  rw [hset]
  have hg : ((↑a : List Std.U64).set i.val v).getD j.val (0#u64)
      = if j.val = i.val then v else (↑a : List Std.U64).getD j.val (0#u64) :=
    list_getD_set _ _ _ _ _ (by omega)
  rw [hg]
  by_cases h : j.val = i.val
  · rw [if_pos h, if_pos h]
  · rw [if_neg h, if_neg h]

/-- One iteration of the extracted absorb loop (`crypto.absorb_block_loop.body`, 3139-3164), read
through the state view: lane `i` takes its old value XOR the little-endian block window at `8 * i`
- the `foldLane` the canonical `xorBlock` computes for that lane (FIPS-202 §4) - and every other
lane is unchanged. The written word `x` is `i17 ^^^ i16` of the loop body, where `i17` is the
`Array.index_usize` read of lane `i` and `i16` the `from_le_bytes` read of the eight block bytes;
`absorbLane_eq_foldLane` pins `i16` to `foldLane`. -/
theorem absorb_step (a : Array Std.U64 25#usize) (block : Slice Std.U8) (i : Std.Usize)
    (hi : i.val < 25) (x : Std.U64)
    (hx : x.bv = (stateView a).get ⟨i.val, hi⟩ ^^^ (core.num.U64.from_le_bytes
      (Array.make 8#usize (blockBytes block (8 * i.val))
        (blockBytes_length block (8 * i.val)))).bv) :
    ∀ k : Fin 25, (stateView (a.set i x)).get k
      = if k.val = i.val then
          (stateView a).get ⟨i.val, hi⟩ ^^^ foldLane (sliceView block) (8 * i.val)
        else (stateView a).get k := by
  have hxl : (core.num.U64.from_le_bytes
      (Array.make 8#usize (blockBytes block (8 * i.val))
        (blockBytes_length block (8 * i.val)))).bv
      = foldLane (sliceView block) (8 * i.val) :=
    absorbLane_eq_foldLane block (8 * i.val)
  intro k
  rw [stateView_set_get a i x hi k]
  by_cases h : k.val = i.val
  · rw [if_pos h, if_pos h, ← hxl]
    exact hx
  · rw [if_neg h, if_neg h]

/-! ## The absorb loop

`crypto.absorb_block` (3181) reads `Slice.len` twice: the `massert` that the block length is a whole
number of lanes, then `lanes = |block| / 8`; `crypto.absorb_block_loop` (3170) XORs one
little-endian block lane per iteration into state lane `i`. This section carries the loop
invariant against the canonical `xorBlock`: after the loop, lane `k < rate / 8` is the old lane XOR
the eight block bytes at `8 * k` (the `foldLane` of `absorb_step`), and every other lane is
unchanged. -/

/-- One `rate`-byte block of a byte list, in the shape `xorBlock` reads: byte `k` of the block is
byte `k` of the list, out of bounds as zero. -/
def blockVec (rate : Nat) (bs : List (BitVec 8)) : Vector (BitVec 8) rate :=
  Vector.ofFn fun i : Fin rate => bs.getD i.val 0

/-- `xorBlock` is read lane by lane: the old lane XOR the `List.range 8` fold over the block
bytes at `8 * i`. -/
theorem xorBlock_get (rate : Nat) (s : State) (b : Vector (BitVec 8) rate) (i : Fin 25) :
    (xorBlock rate s b).get i
      = s.get i ^^^ (List.range 8).foldl
          (fun acc j => acc ^^^ ((b.getD (8 * i.val + j) 0).setWidth 64 <<< (8 * j))) 0 := by
  show (xorBlock rate s b)[i.val] = _
  simp only [xorBlock, Vector.getElem_ofFn]

/-- An in-bounds byte of a `blockVec` block is the byte of the list. -/
theorem blockVec_getD (rate : Nat) (bs : List (BitVec 8)) (k : Nat) (hk : k < rate) :
    (blockVec rate bs).getD k 0 = bs.getD k 0 := by
  simp [blockVec, Vector.getD, Array.getD, hk]

/-- A byte past the end of a `blockVec` block reads as zero. -/
theorem blockVec_getD_ge (rate : Nat) (bs : List (BitVec 8)) (k : Nat) (hk : rate ≤ k) :
    (blockVec rate bs).getD k 0 = 0 := by
  simp [blockVec, Vector.getD, Array.getD, Nat.not_lt.mpr hk]

/-- Two step functions that agree, at every accumulator, on the elements of the folded list fold
alike. -/
theorem foldl_congr {α β : Type} (f g : α → β → α) :
    ∀ (l : List β), (∀ b ∈ l, ∀ x, f x b = g x b) → ∀ x, l.foldl f x = l.foldl g x := by
  intro l
  induction l with
  | nil => intro _ x; rfl
  | cons a t ih =>
    intro h x
    simp only [List.foldl]
    rw [← h a (by simp) x]
    exact ih (fun b hb y => h b (by simp [hb]) y) _

/-- The eight bytes of a window past the end of a byte list fold to the zero lane. -/
theorem foldLane_eq_zero {bs : List (BitVec 8)} {o : Nat} (h : bs.length ≤ o) :
    foldLane bs o = 0 := by
  have key : ∀ j : Nat, j < 8 → (bs.getD (o + j) 0).setWidth 64 <<< (8 * j) = 0 := by
    intro j _
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none (by omega)]
    simp
  unfold foldLane
  rw [foldl_congr _ (fun x _ => x) _
    (fun j hj x => by rw [key j (List.mem_range.mp hj)]; simp [BitVec.xor_zero]) 0]
  rfl

/-- `xorBlock` of a byte-list block at a rate lane is the old lane XOR the eight block bytes at
`8 * k`: the `rate % 8 = 0` bound keeps every byte of that window inside the block. -/
theorem xorBlock_blockVec_foldLane (rate : Nat) (s : State) (bs : List (BitVec 8))
    (hrate : rate % 8 = 0) (i : Fin 25) (hi : i.val < rate / 8) :
    (xorBlock rate s (blockVec rate bs)).get i = s.get i ^^^ foldLane bs (8 * i.val) := by
  rw [xorBlock_get]
  have hr8 : rate = 8 * (rate / 8) := by omega
  have hfun : ∀ j : Nat, j < 8 →
      (blockVec rate bs).getD (8 * i.val + j) 0 = bs.getD (8 * i.val + j) 0 := by
    intro j hj
    exact blockVec_getD rate bs (8 * i.val + j) (by omega)
  unfold foldLane
  rw [foldl_congr _ _ _
    (fun j hj x => by rw [hfun j (List.mem_range.mp hj)]) 0]

/-- `xorBlock` of a byte-list block past the rate block leaves the lane unchanged: the block holds
`rate` bytes and reads out of bounds as zero. -/
theorem xorBlock_lane_ge (rate : Nat) (s : State) (bs : List (BitVec 8)) (hrate : rate % 8 = 0)
    (hlen : bs.length = rate) (i : Fin 25) (hi : rate / 8 ≤ i.val) :
    (xorBlock rate s (blockVec rate bs)).get i = s.get i := by
  rw [xorBlock_get]
  have hr8 : rate = 8 * (rate / 8) := by omega
  have hfun : ∀ j : Nat, j < 8 → (blockVec rate bs).getD (8 * i.val + j) 0 = 0 := by
    intro j hj
    exact blockVec_getD_ge rate bs (8 * i.val + j) (by omega)
  rw [foldl_congr _ (fun x _ => x) _
    (fun j hj x => by rw [hfun j (List.mem_range.mp hj)]; simp [BitVec.xor_zero]) 0]
  simp

/-- Loop invariant of the extracted absorb loop, read through `stateView`: lane `k` below the
counter holds its old value XOR the eight block bytes at `8 * k`, every other lane is unchanged. -/
def absorbInv (block : Slice Std.U8) (state a : Array Std.U64 25#usize) (i : Std.Usize) : Prop :=
  ∀ k : Fin 25, (stateView a).get k
    = if k.val < i.val then (stateView state).get k ^^^ foldLane (sliceView block) (8 * k.val)
      else (stateView state).get k

/-- The absorb-loop postcondition named opaquely: the final state is the canonical `xorBlock` of
the start state with the block bytes. Kept a named `def` so `loop.spec_decr_nat` unifies the post
without unfolding `xorBlock`/`blockVec`/`stateView`. -/
def absorbPost (rate : Nat) (state : Array Std.U64 25#usize) (block : Slice Std.U8)
    (a : Array Std.U64 25#usize) : Prop :=
  ∀ k : Fin 25, (stateView a).get k
    = (xorBlock rate (stateView state) (blockVec rate (sliceView block))).get k

/-- The absorb-loop invariant together with the counter bound that keeps the counter inside the
rate block. -/
def absorbLoopInv (block : Slice Std.U8) (lanes : Std.Usize)
    (state a : Array Std.U64 25#usize) (i : Std.Usize) : Prop :=
  i.val ≤ lanes.val ∧ absorbInv block state a i

/-- The extracted absorb-loop step function named opaquely, so `loop.spec_decr_nat` fixes it as the
loop body without the elaborator unfolding the whole `do` chain. -/
def absorbBody (block : Slice Std.U8) (lanes : Std.Usize) :
    Array Std.U64 25#usize × Std.Usize →
      Result (ControlFlow (Array Std.U64 25#usize × Std.Usize) (Array Std.U64 25#usize)) :=
  fun s => crypto.absorb_block_loop.body block lanes s.1 s.2

/-- The extracted loop with the opaque named body: definitional, and stated as an equality so
`loop.spec_decr_nat` can fix the body without the elaborator unfolding the `do` chain. -/
theorem absorb_block_loop_eq (state : Array Std.U64 25#usize) (block : Slice Std.U8)
    (lanes : Std.Usize) (i : Std.Usize) :
    crypto.absorb_block_loop state block lanes i = loop (absorbBody block lanes) (state, i) := rfl

/-- From the invariant and the counter run to the end of the rate block, the state is the
canonical `xorBlock` of the start state with the block bytes. -/
theorem absorbInv_post (block : Slice Std.U8) (rate : Nat) (state a : Array Std.U64 25#usize)
    (i : Std.Usize) (hrate : rate % 8 = 0) (hi : i.val = rate / 8)
    (hlen : (sliceView block).length = rate) (hinv : absorbInv block state a i) :
    absorbPost rate state block a := by
  unfold absorbPost
  intro k
  rw [hinv k]
  by_cases hlt : k.val < i.val
  · -- `k < i = rate / 8`, so `k` is inside the rate block and `xorBlock` reads the eight block
    -- bytes at `8 * k`.
    rw [if_pos hlt]
    rw [xorBlock_blockVec_foldLane rate (stateView state) (sliceView block) hrate k (by omega)]
  · -- `i = rate / 8 ≤ k`: the lane is past the rate block, and `xorBlock` leaves it alone.
    rw [if_neg hlt]
    rw [xorBlock_lane_ge rate (stateView state) (sliceView block) hrate hlen k (by omega)]

/-- The eight block bytes the extracted loop reads at `o`, as the eight-element list the
little-endian lane read consumes. -/
theorem blockBytes_eq_list (block : Slice Std.U8) (o : Nat) :
    blockBytes block o = [block.val.getD (o + 0) (0#u8), block.val.getD (o + 1) (0#u8),
      block.val.getD (o + 2) (0#u8), block.val.getD (o + 3) (0#u8),
      block.val.getD (o + 4) (0#u8), block.val.getD (o + 5) (0#u8),
      block.val.getD (o + 6) (0#u8), block.val.getD (o + 7) (0#u8)] := by
  unfold blockBytes
  rw [List.ofFn_succ, List.ofFn_succ, List.ofFn_succ, List.ofFn_succ, List.ofFn_succ,
    List.ofFn_succ, List.ofFn_succ, List.ofFn_succ]
  rfl

/-- One iteration's little-endian lane read, of the eight block bytes at `o = 8 * i`, is the
`foldLane` of those bytes at that offset - the same fold `xorBlock` runs for lane `i`. -/
theorem from_le_bytes_eight (block : Slice Std.U8) (o : Nat) (i : Nat)
    (b0 b1 b2 b3 b4 b5 b6 b7 : Std.U8)
    (hlen : ([b0, b1, b2, b3, b4, b5, b6, b7] : List Std.U8).length = 8)
    (hb0 : b0 = block.val.getD (o + 0) (0#u8)) (hb1 : b1 = block.val.getD (o + 1) (0#u8))
    (hb2 : b2 = block.val.getD (o + 2) (0#u8)) (hb3 : b3 = block.val.getD (o + 3) (0#u8))
    (hb4 : b4 = block.val.getD (o + 4) (0#u8)) (hb5 : b5 = block.val.getD (o + 5) (0#u8))
    (hb6 : b6 = block.val.getD (o + 6) (0#u8)) (hb7 : b7 = block.val.getD (o + 7) (0#u8))
    (ho : o = 8 * i) :
    (core.num.U64.from_le_bytes
      (Array.make 8#usize [b0, b1, b2, b3, b4, b5, b6, b7] hlen)).bv
      = foldLane (sliceView block) (8 * i) := by
  subst ho
  have hlist : ([b0, b1, b2, b3, b4, b5, b6, b7] : List Std.U8) = blockBytes block (8 * i) := by
    rw [blockBytes_eq_list block (8 * i)]
    simp only [hb0, hb1, hb2, hb3, hb4, hb5, hb6, hb7]
  rw [from_le_bytes_denote_list [b0, b1, b2, b3, b4, b5, b6, b7] hlen,
    foldLane_eq_fromLEBytes]
  exact fromLEBytes_cast_eq (by rw [hlist, blockBytes_view])

/-- In-bounds `Array.index_usize` reads the underlying list. -/
theorem array_index_usize_eq {α : Type} {n : Std.Usize} (a : Array α n) (i : Std.Usize)
    (h : i.val < a.val.length) : Array.index_usize a i = ok (a.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Array.index_usize_spec a i h)
  rw [hxv] at hx
  exact hx

/-- In-bounds `Slice.index_usize` reads the underlying list. -/
theorem slice_index_usize_eq {α : Type} (s : Slice α) (i : Std.Usize)
    (h : i.val < s.val.length) : Slice.index_usize s i = ok (s.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec s i h)
  rw [hxv] at hx
  exact hx

/-- The lane list of an `Array Std.U64 25#usize` has length 25, so an in-bounds index is a lane
index. -/
theorem array_u64_25_length (a : Array Std.U64 25#usize) : a.val.length = 25 := by
  simpa using a.property



set_option maxHeartbeats 1500000 in
/-- One iteration of the extracted absorb loop from the current state `a`, with the initial state
`state` fixed: lane `i` takes its old value XOR the little-endian block window at `8 * i`, the
counter advances, and the loop invariant relating `a` to `state` is preserved. -/
theorem absorb_body_step (state : Array Std.U64 25#usize) (a : Array Std.U64 25#usize)
    (block : Slice Std.U8)
    (lanes : Std.Usize) (rate : Nat) (i : Std.Usize) (hlt : i.val < lanes.val)
    (hrate : rate % 8 = 0) (hlanes : lanes.val = rate / 8) (hr25 : rate / 8 ≤ 25)
    (hlen : (sliceView block).length = rate) (hinv : absorbInv block state a i) :
    ∃ (a1 : Array Std.U64 25#usize) (i1 : Std.Usize),
      crypto.absorb_block_loop.body block lanes a i = ok (ControlFlow.cont (a1, i1)) ∧
      i1.val = i.val + 1 ∧ absorbInv block state a1 i1 := by
  have hblen : block.val.length = rate := by rw [← sliceView_length]; exact hlen
  have hi25 : i.val < 25 := by omega
  have h8i : 8 * i.val + 7 < rate := by omega
  have hmax8 : i.val * (8#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o, ho, hov⟩ := WP.spec_imp_exists (Usize.mul_spec (x := i) (y := 8#usize) hmax8)
  have hoval : o.val = i.val * 8 := by rw [hov]; rfl
  have h8o : o.val = 8 * i.val := by omega
  have hb0 : o.val < block.val.length := by rw [h8o]; omega
  have hidx0 : Slice.index_usize block o = ok (block.val[o.val]'hb0) :=
    slice_index_usize_eq block o hb0
  have hmax1 : o.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o1, ho1, hov1⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 1#usize) hmax1)
  have ho1val : o1.val = o.val + 1 := by rw [hov1]; rfl
  have hb1 : o1.val < block.val.length := by rw [ho1val, h8o]; omega
  have hidx1 : Slice.index_usize block o1 = ok (block.val[o1.val]'hb1) :=
    slice_index_usize_eq block o1 hb1
  have hmax2 : o.val + (2#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o2, ho2, hov2⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 2#usize) hmax2)
  have ho2val : o2.val = o.val + 2 := by rw [hov2]; rfl
  have hb2 : o2.val < block.val.length := by rw [ho2val, h8o]; omega
  have hidx2 : Slice.index_usize block o2 = ok (block.val[o2.val]'hb2) :=
    slice_index_usize_eq block o2 hb2
  have hmax3 : o.val + (3#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o3, ho3, hov3⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 3#usize) hmax3)
  have ho3val : o3.val = o.val + 3 := by rw [hov3]; rfl
  have hb3 : o3.val < block.val.length := by rw [ho3val, h8o]; omega
  have hidx3 : Slice.index_usize block o3 = ok (block.val[o3.val]'hb3) :=
    slice_index_usize_eq block o3 hb3
  have hmax4 : o.val + (4#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o4, ho4, hov4⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 4#usize) hmax4)
  have ho4val : o4.val = o.val + 4 := by rw [hov4]; rfl
  have hb4 : o4.val < block.val.length := by rw [ho4val, h8o]; omega
  have hidx4 : Slice.index_usize block o4 = ok (block.val[o4.val]'hb4) :=
    slice_index_usize_eq block o4 hb4
  have hmax5 : o.val + (5#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o5, ho5, hov5⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 5#usize) hmax5)
  have ho5val : o5.val = o.val + 5 := by rw [hov5]; rfl
  have hb5 : o5.val < block.val.length := by rw [ho5val, h8o]; omega
  have hidx5 : Slice.index_usize block o5 = ok (block.val[o5.val]'hb5) :=
    slice_index_usize_eq block o5 hb5
  have hmax6 : o.val + (6#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o6, ho6, hov6⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 6#usize) hmax6)
  have ho6val : o6.val = o.val + 6 := by rw [hov6]; rfl
  have hb6 : o6.val < block.val.length := by rw [ho6val, h8o]; omega
  have hidx6 : Slice.index_usize block o6 = ok (block.val[o6.val]'hb6) :=
    slice_index_usize_eq block o6 hb6
  have hmax7 : o.val + (7#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨o7, ho7, hov7⟩ := WP.spec_imp_exists (Usize.add_spec (x := o) (y := 7#usize) hmax7)
  have ho7val : o7.val = o.val + 7 := by rw [hov7]; rfl
  have hb7 : o7.val < block.val.length := by rw [ho7val, h8o]; omega
  have hidx7 : Slice.index_usize block o7 = ok (block.val[o7.val]'hb7) :=
    slice_index_usize_eq block o7 hb7
  have hi25b : i.val < a.val.length := by
    have hs : a.val.length = 25 := by simpa using a.property
    omega
  have hidx17 : Array.index_usize a i = ok (a.val[i.val]'hi25b) :=
    array_index_usize_eq a i hi25b
  have hmaxi1 : i.val + (1#usize).val ≤ Usize.max := by scalar_tac
  obtain ⟨i1, hi1, hovi1⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hmaxi1)
  have hi1val : i1.val = i.val + 1 := by rw [hovi1]; rfl
  have hlit : ([block.val[o.val]'hb0, block.val[o1.val]'hb1, block.val[o2.val]'hb2,
      block.val[o3.val]'hb3, block.val[o4.val]'hb4, block.val[o5.val]'hb5,
      block.val[o6.val]'hb6, block.val[o7.val]'hb7] : List Std.U8).length = 8 := rfl
  obtain ⟨a1, ha1, ha1v⟩ := WP.spec_imp_exists (Array.update_spec a i
    (a.val[i.val]'hi25b ^^^ core.num.U64.from_le_bytes
      (Array.make 8#usize [block.val[o.val]'hb0, block.val[o1.val]'hb1, block.val[o2.val]'hb2,
        block.val[o3.val]'hb3, block.val[o4.val]'hb4, block.val[o5.val]'hb5,
        block.val[o6.val]'hb6, block.val[o7.val]'hb7] hlit)) hi25b)
  have hb0' : block.val[o.val]'hb0 = block.val.getD (8 * i.val + 0) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), h8o]
    simp
  have hb1' : block.val[o1.val]'hb1 = block.val.getD (8 * i.val + 1) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho1val, h8o]
  have hb2' : block.val[o2.val]'hb2 = block.val.getD (8 * i.val + 2) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho2val, h8o]
  have hb3' : block.val[o3.val]'hb3 = block.val.getD (8 * i.val + 3) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho3val, h8o]
  have hb4' : block.val[o4.val]'hb4 = block.val.getD (8 * i.val + 4) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho4val, h8o]
  have hb5' : block.val[o5.val]'hb5 = block.val.getD (8 * i.val + 5) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho5val, h8o]
  have hb6' : block.val[o6.val]'hb6 = block.val.getD (8 * i.val + 6) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho6val, h8o]
  have hb7' : block.val[o7.val]'hb7 = block.val.getD (8 * i.val + 7) (0#u8) := by
    rw [List.getElem_eq_getD (0#u8), ho7val, h8o]
  have hlist : [block.val[o.val]'hb0, block.val[o1.val]'hb1, block.val[o2.val]'hb2,
      block.val[o3.val]'hb3, block.val[o4.val]'hb4, block.val[o5.val]'hb5,
      block.val[o6.val]'hb6, block.val[o7.val]'hb7] = blockBytes block (8 * i.val) := by
    rw [hb0', hb1', hb2', hb3', hb4', hb5', hb6', hb7', ← blockBytes_eq_list]
  refine ⟨a1, i1, ?_, hi1val, ?_⟩
  · unfold crypto.absorb_block_loop.body
    rw [if_pos ((UScalar.lt_equiv i lanes).mpr hlt)]
    rw [ho]
    simp only [bind_tc_ok]
    rw [hidx0]
    simp only [bind_tc_ok]
    rw [ho1]
    simp only [bind_tc_ok]
    rw [hidx1]
    simp only [bind_tc_ok]
    rw [ho2]
    simp only [bind_tc_ok]
    rw [hidx2]
    simp only [bind_tc_ok]
    rw [ho3]
    simp only [bind_tc_ok]
    rw [hidx3]
    simp only [bind_tc_ok]
    rw [ho4]
    simp only [bind_tc_ok]
    rw [hidx4]
    simp only [bind_tc_ok]
    rw [ho5]
    simp only [bind_tc_ok]
    rw [hidx5]
    simp only [bind_tc_ok]
    rw [ho6]
    simp only [bind_tc_ok]
    rw [hidx6]
    simp only [bind_tc_ok]
    rw [ho7]
    simp only [bind_tc_ok]
    rw [hidx7]
    simp only [lift, bind_tc_ok]
    rw [hidx17]
    simp only [lift, bind_tc_ok]
    rw [ha1]
    simp only [bind_tc_ok]
    rw [hi1]
    simp only [bind_tc_ok]
  · intro k
    have hx : (a.val[i.val]'hi25b ^^^ core.num.U64.from_le_bytes
        (Array.make 8#usize [block.val[o.val]'hb0, block.val[o1.val]'hb1, block.val[o2.val]'hb2,
          block.val[o3.val]'hb3, block.val[o4.val]'hb4, block.val[o5.val]'hb5,
          block.val[o6.val]'hb6, block.val[o7.val]'hb7] hlit)).bv
        = (stateView a).get ⟨i.val, hi25⟩ ^^^
          (core.num.U64.from_le_bytes
            (Array.make 8#usize (blockBytes block (8 * i.val))
              (blockBytes_length block (8 * i.val)))).bv := by
      rw [List.getElem_eq_getD (0#u64), stateView_get, UScalar.bv_xor]
      simp only [from_le_bytes_denote_list]
      congr 1
      exact fromLEBytes_cast_eq (by rw [hlist, blockBytes_view])
    rw [ha1v, absorb_step a block i hi25 _ hx k]
    by_cases h : k.val = i.val
    · have hk : k = ⟨i.val, hi25⟩ := Fin.ext h
      subst hk
      rw [if_pos rfl, if_pos (by omega)]
      rw [hinv ⟨i.val, hi25⟩, if_neg (by omega)]
    · rw [if_neg h, hinv k]
      by_cases hlt : k.val < i.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega)]

set_option maxHeartbeats 1500000 in
/-- The extracted absorb loop over one rate block: on success the state is the canonical
`xorBlock` of the start state with the block bytes. -/
theorem absorb_loop_spec (state : Array Std.U64 25#usize) (block : Slice Std.U8)
    (rate : Nat) (lanes : Std.Usize) (hrate : rate % 8 = 0) (hlanes : lanes.val = rate / 8)
    (hr25 : rate / 8 ≤ 25) (hlen : (sliceView block).length = rate) :
    crypto.absorb_block_loop state block lanes 0#usize ⦃ absorbPost rate state block ⦄ := by
  rw [absorb_block_loop_eq]
  apply loop.spec_decr_nat
    (measure := fun s : Array Std.U64 25#usize × Std.Usize => lanes.val - s.2.val)
    (inv := fun s => s.2.val ≤ lanes.val ∧ absorbInv block state s.1 s.2)
    (post := absorbPost rate state block)
  · intro s hs
    rcases s with ⟨a1, i1⟩
    have hs' : i1.val ≤ lanes.val ∧ absorbInv block state a1 i1 := hs
    obtain ⟨hi1, hinv1⟩ := hs'
    by_cases hlt : i1.val < lanes.val
    · obtain ⟨a2, i2, hbody, hi2v, hinv2⟩ :=
        absorb_body_step state a1 block lanes rate i1 hlt hrate hlanes hr25 hlen hinv1
      have hbody' : absorbBody block lanes (a1, i1) = ok (ControlFlow.cont (a2, i2)) := hbody
      rw [hbody']
      simp only [Aeneas.Std.WP.spec_ok]
      refine ⟨⟨by omega, ?_⟩, by omega⟩
      exact hinv2
    · have hnot : ¬ (i1 < lanes) := by
        intro hc'
        exact hlt ((UScalar.lt_equiv i1 lanes).mp hc')
      have hbody : absorbBody block lanes (a1, i1) = ok (ControlFlow.done a1) := by
        unfold absorbBody
        unfold crypto.absorb_block_loop.body
        rw [if_neg hnot]
      rw [hbody]
      simp only [Aeneas.Std.WP.spec_ok]
      exact absorbInv_post block rate state a1 i1 hrate (by omega) hlen hinv1
  · change (0#usize).val ≤ lanes.val ∧ absorbInv block state state 0#usize
    refine ⟨by simp, ?_⟩
    intro k
    rw [if_neg (by simp)]

/-! ## Rates, offsets and domain separation

`crypto.sponge` (3377-3380) requires `rate % 8 = 0` and `rate ≤ 200`; the rate is the byte rate, so
a block is `rate / 8` lanes, and the state byte offsets of a block run `0 .. rate - 1`. -/

theorem rate_256_denote : crypto.SHA3_256_RATE.val = rate_sha3_256 := by
  simp [rate_sha3_256, crypto.SHA3_256_RATE]

theorem rate_512_denote : crypto.SHA3_512_RATE.val = rate_sha3_512 := by
  simp [rate_sha3_512, crypto.SHA3_512_RATE]

theorem domain_denote : (6#u8).bv = domain_sha3 := by decide

theorem pad_final_denote : (128#u8).bv = 0x80 := by decide

/-- A rate block is a whole number of lanes. -/
theorem rate_lanes_256 : rate_sha3_256 / 8 = 17 := by decide

theorem rate_lanes_512 : rate_sha3_512 / 8 = 9 := by decide

/-- The `massert (rate % 8 = 0)` of `crypto.sponge` holds for both canonical rates. -/
theorem rate_mod_8_256 : rate_sha3_256 % 8 = 0 := by decide

theorem rate_mod_8_512 : rate_sha3_512 % 8 = 0 := by decide

/-- The `massert (rate ≤ 200)` of `crypto.sponge` - the `Array.repeat 200` padding buffer - holds
for both canonical rates. -/
theorem rate_le_200_256 : rate_sha3_256 ≤ 200 := by decide

theorem rate_le_200_512 : rate_sha3_512 ≤ 200 := by decide

/-- Bit rates, for the FIPS-202 parameter names. -/
theorem rate_bits_256 : rate_sha3_256 * 8 = 1088 := by decide

theorem rate_bits_512 : rate_sha3_512 * 8 = 576 := by decide

/-- The squeeze stage reads 32 output bytes out of one rate block: the stage-5 bound `32 ≤ 136`
that keeps the second permutation out of the picture. -/
theorem outLen_le_rate_256 : 32 ≤ rate_sha3_256 := by decide

theorem outLen_div_rate_256 : 32 / rate_sha3_256 = 0 := by decide

/-! ## Known-answer pins

These run the untrusted evaluator: they guard byte order, lane order and rate arithmetic against
regressions, they are not kernel proofs. -/

#guard (6#u8).bv == 0x06
#guard (128#u8).bv == 0x80
#guard crypto.SHA3_256_RATE.val == 136
#guard crypto.SHA3_512_RATE.val == 72

-- Little-endian packing: byte 0 is the least significant byte.
#guard BitVec.fromLEBytes [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] == 0x0807060504030201
#guard foldLane [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08] 0 == 0x0807060504030201

-- The fold shape agrees with the byte order at a nonzero window offset, and the byte window of a
-- lane reads back in the same order.
#guard foldLane ([0xff, 0xff] ++ [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]) 2 ==
  0x0807060504030201
#guard laneBytes 0x0807060504030201 == [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]

/-! ## The headline (stage 6)

`Result α` is `ITree RustEffect α` and `ok = Result.ok`; the model has no `Result.map`, so
`resultMap` is the `map` the headline statement needs. `crypto.sha3_256` returns
`Result (Array Std.U8 32#usize)` and the canonical `sha3_256` returns `Vector (BitVec 8) 32`, so
the statement is carried over the byte views, type-adapted from the plan. -/

/-- `Result α` is `ITree RustEffect α`, an irreducible definition, so the `map` is read through
`Result.match`; `ok` is dispatched by `Result.match.ok`. -/
def resultMap {α β : Type} (f : α → β) (r : Result α) : Result β :=
  match r.match with
  | .ok a => Result.ok (f a)
  | .vis e k => Result.vis e (fun o => Std.bind (k o) fun x => Result.ok (f x))
  | .div => Result.div

theorem resultMap_ok {α β : Type} (f : α → β) (a : α) :
    resultMap f (Result.ok a) = Result.ok (f a) := by
  simp [resultMap, Result.match.ok]

/-- The refinement target: the extracted `word_sha3_256` computes the canonical
`Apeiron.Crypto.Keccak.sha3_256` on the byte view of its input. Carried, not asserted; stages 2-5
supply the loop and conversion lemmas it is built from. -/
def WordSha3_256Denotes (msg : Slice Std.U8) : Prop :=
  resultMap (fun a => byteView a) (word_sha3_256 msg) = ok (sha3_256 (sliceView msg)).toList

end apeiron_word
