/-
  `Icarus/Extraction/LoadBlock.lean` — **O14b, the block-word loading loop of `sha256.compress`,
  as a typed value link.**

  Owned files: this one and the task receipt. No corpus edit, no extraction refresh, no commit.
  Read-only upstream: `Crypto/Sha2.lean`, `Crypto/Sha256Certificate.lean`, the generated
  `ApeironWord.lean`, and `Icarus/Composition/Sha256BE.lean`. The campaign staging source
  `work/proofs/AlgebraBE16Load.lean` (AR-69) lives outside this package and is cited, not imported.

  ## What this file links

  **Extracted end.** `sha256.compress_loop0` (`ApeironWord.lean:15269`, from
  `src/sha256.rs:37-41`) is the `while i < 16` loop that fills the schedule's first sixteen rows
  with `u32::from_be_bytes([b[4i], b[4i+1], b[4i+2], b[4i+3]])`. `loop` is a `partial_fixpoint`
  (`Aeneas/Std/Primitives.lean:415-420`), so the loop's value is not definitional: it is reached only
  through `Aeneas.Std.loop.spec_decr_nat` (`Aeneas/Std/WP.lean:979`). Every statement about it is
  therefore **success-conditioned**, and no totality or existence claim is made or implied.

  **Canonical end.** `apeiron_word.rowVec (bytesToWords block) 0`
  (`Crypto/Sha256Certificate.lean:313`, `:493`) — the sixteen big-endian block words, zero above,
  which is the invariant premise `apeiron_word.sched_loop1` already takes.

  The link, `load_loop0_repeat`: run at the schedule array `sha256.compress` actually starts from,
  `Array.repeat 64#usize 0#u32`, a successful `sha256.compress_loop0` yields exactly
  `wView w' = rowVec (bytesToWords block) 0`. This is the premise the O14 owner consumes in
  `Sha256AddBack`'s `hload`; that file is not edited here.

  ## What is proved

  1. **§1 one visit.** `loadBody_step`: for `i.val < 16` the extracted body is
     `ok (cont (w.set i v, i'))` with `i'.val = i.val + 1` and `v.bv = beWord16 block ⟨i.val, _⟩` —
     the four `Array.index_usize` reads at `4i … 4i+3` and the four-byte big-endian read, read off
     `Apeiron.Sha256BE.from_be_bytes_make4` and `Apeiron.Sha256BE.beWord16_eq_beWord4` (the corpus's
     own four-byte layer, which AR-69 restated verbatim because `work/proofs` is not a lake package
     root). `loadBody_done` is the exit branch; `loadBody_total` the absence of a failure path at
     any index.
  2. **§2 the invariant.** `loadVec`, `loadVec_getD`, `loadVec16_eq_rowVec0`, `loadInv`,
     `loadStep_inv` — one visit preserves "rows written so far are the block words, nothing above".
  3. **§3 the loop.** `load_loop0` at an arbitrary zero-extended start schedule, and
     `load_loop0_repeat` at the repeated-zero start `sha256.compress` uses, with the row-read
     corollaries `load_loop0_wGet`, `load_loop0_beWord16`, `load_loop0_getD`.

  ## Grades and non-claims

  `proved` for §§1–3 from the build gate recorded in the receipt: `#print axioms` rows inside
  `{propext, Classical.choice, Quot.sound}`, zero `sorryAx`.

  **Success-conditioned throughout**, at the pinned extracted representation of `compress_loop0`.
  No claim is made that a successful run exists, nor that the loop is the only way the pipeline
  reaches that schedule. The AR-40/AR-46 `blockArray` check of AR-69 §4 is **not** ported here: the
  corpus already carries it in `Icarus/Composition/AlgebraSha256dBlockArray.lean`, and it is not in
  this theorem's dependency closure. The 16-bit layer of AR-69 §1 is likewise not ported. Nothing
  here bears on security, collision or preimage resistance, differential cryptanalysis, side
  channels or constant-time behaviour, and no timing is measured.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate
import ApeironWord
import Icarus.Composition.Sha256BE

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace Icarus
namespace Extraction
namespace LoadBlock

open Aeneas Aeneas.Std Result
open Apeiron.Crypto.Sha2 (Word Block)
open apeiron_word
open apeiron_word (b64 beWord16 bytesToWords bytesToWords_getD rowVec rowVec_getD schedRow
  vec_getD_eq ofFn_getD ofFn_getD' wGet wGet_set wGet_zero wView wView_getD array_index_usize_eq
  getElem_getD)

/-! ## §1 · One visit of the loading loop

`sha256.compress_loop0.body` (`ApeironWord.lean:15243-15264`) reads `block[4i]`, `block[4i+1]`,
`block[4i+2]`, `block[4i+3]` through four `Array.index_usize` reads at `o = 4*i`, `o+1`, `o+2`,
`o+3`, reads them big-endian through `core.num.U32.from_be_bytes`, writes the word into slot `i` and
returns the counter `i+1`. -/

/-- **One block-word loading visit.** With `i.val < 16` the extracted body succeeds: it writes row
    `i` with the canonical big-endian word of bytes `4i … 4i+3` and returns the counter
    `i.val + 1`. The body never fails here: the guard bounds the read offsets below the block
    length and the write index below the schedule length. -/
theorem loadBody_step (block : Array Std.U8 64#usize) (w : Array Std.U32 64#usize)
    (i : Std.Usize) (hlt : i.val < 16) :
    ∃ v : Std.U32, ∃ i' : Std.Usize,
      sha256.compress_loop0.body block w i = Result.ok (ControlFlow.cont (w.set i v, i')) ∧
      i'.val = i.val + 1 ∧
      v.bv = beWord16 block ⟨i.val, by omega⟩ := by
  have h4v : ((4#usize : Std.Usize).val) = 4 := rfl
  have h1v : ((1#usize : Std.Usize).val) = 1 := rfl
  have h2v : ((2#usize : Std.Usize).val) = 2 := rfl
  have h3v : ((3#usize : Std.Usize).val) = 3 := rfl
  have hbl : block.val.length = 64 := by simpa using block.property
  have hwl : w.val.length = 64 := by simpa using w.property
  -- `Usize.max` is at least 32-bit wide, so every offset below is in range.
  have hmax : (4294967295 : Nat) ≤ Usize.max := (Usize.cMax_bound_concrete).1
  -- `o ← 4 * i`, `o1 ← o + 1`, `o2 ← o + 2`, `o3 ← o + 3`, each in range.
  obtain ⟨o, ho, hov⟩ :=
    WP.spec_imp_exists (Usize.mul_spec (x := 4#usize) (y := i) (by rw [h4v]; omega))
  have hov' : o.val = 4 * i.val := by rw [hov, h4v, Nat.mul_comm]
  obtain ⟨o1, ho1, ho1v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := o) (y := 1#usize) (by rw [h1v]; omega))
  have ho1v' : o1.val = 4 * i.val + 1 := by rw [ho1v, h1v, hov']
  obtain ⟨o2, ho2, ho2v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := o) (y := 2#usize) (by rw [h2v]; omega))
  have ho2v' : o2.val = 4 * i.val + 2 := by rw [ho2v, h2v, hov']
  obtain ⟨o3, ho3, ho3v⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := o) (y := 3#usize) (by rw [h3v]; omega))
  have ho3v' : o3.val = 4 * i.val + 3 := by rw [ho3v, h3v, hov']
  have hb0 : o.val < block.val.length := by rw [hov']; omega
  have hb1 : o1.val < block.val.length := by rw [ho1v']; omega
  have hb2 : o2.val < block.val.length := by rw [ho2v']; omega
  have hb3 : o3.val < block.val.length := by rw [ho3v']; omega
  -- The four reads, at the bytes of the one word being written.
  let b0 : Std.U8 := block.val[o.val]'hb0
  let b1 : Std.U8 := block.val[o1.val]'hb1
  let b2 : Std.U8 := block.val[o2.val]'hb2
  let b3 : Std.U8 := block.val[o3.val]'hb3
  have hread0 : Array.index_usize block o = ok b0 := by
    rw [array_index_usize_eq block o hb0]
  have hread1 : Array.index_usize block o1 = ok b1 := by
    rw [array_index_usize_eq block o1 hb1]
  have hread2 : Array.index_usize block o2 = ok b2 := by
    rw [array_index_usize_eq block o2 hb2]
  have hread3 : Array.index_usize block o3 = ok b3 := by
    rw [array_index_usize_eq block o3 hb3]
  -- The written word is the big-endian read of those four bytes.
  let v : Std.U32 := core.num.U32.from_be_bytes (Array.make 4#usize [b0, b1, b2, b3])
  have hb0v : (b0 : Std.U8).bv = b64 block (4 * i.val) := by
    show (block.val[o.val]'hb0 : Std.U8).bv = _
    rw [getElem_getD _ _ hb0 (0#u8), hov']
    rfl
  have hb1v : (b1 : Std.U8).bv = b64 block (4 * i.val + 1) := by
    show (block.val[o1.val]'hb1 : Std.U8).bv = _
    rw [getElem_getD _ _ hb1 (0#u8), ho1v']
    rfl
  have hb2v : (b2 : Std.U8).bv = b64 block (4 * i.val + 2) := by
    show (block.val[o2.val]'hb2 : Std.U8).bv = _
    rw [getElem_getD _ _ hb2 (0#u8), ho2v']
    rfl
  have hb3v : (b3 : Std.U8).bv = b64 block (4 * i.val + 3) := by
    show (block.val[o3.val]'hb3 : Std.U8).bv = _
    rw [getElem_getD _ _ hb3 (0#u8), ho3v']
    rfl
  have hval : v.bv = beWord16 block ⟨i.val, by omega⟩ := by
    calc (core.num.U32.from_be_bytes (Array.make 4#usize [b0, b1, b2, b3])).bv
        = Apeiron.Sha256BE.beWord4 b0.bv b1.bv b2.bv b3.bv :=
          Apeiron.Sha256BE.from_be_bytes_make4 b0 b1 b2 b3
      _ = Apeiron.Sha256BE.beWord4 (b64 block (4 * i.val)) (b64 block (4 * i.val + 1))
            (b64 block (4 * i.val + 2)) (b64 block (4 * i.val + 3)) :=
        by rw [hb0v, hb1v, hb2v, hb3v]
      _ = beWord16 block ⟨i.val, by omega⟩ :=
        (Apeiron.Sha256BE.beWord16_eq_beWord4 block ⟨i.val, by omega⟩).symm
  -- The write at slot `i`, and the counter increment.
  have hiwl : i.val < w.val.length := by rw [hwl]; omega
  have hupd : Array.update w i
        (core.num.U32.from_be_bytes (Array.make 4#usize [b0, b1, b2, b3]))
      = ok (w.set i v) := by
    obtain ⟨nv, hnv, hnv'⟩ := WP.spec_imp_exists (Array.update_spec w i v hiwl)
    rw [hnv'] at hnv
    exact hnv
  have hbound : i.val + (1#usize : Std.Usize).val ≤ Usize.max := by
    rw [h1v]; omega
  obtain ⟨i', hi', hiv⟩ :=
    WP.spec_imp_exists (Usize.add_spec (x := i) (y := 1#usize) hbound)
  have hiv' : i'.val = i.val + 1 := by rw [hiv, h1v]
  refine ⟨v, i', ?_, hiv', hval⟩
  -- The replay: the four reads, the write and the increment are their specifications, in one
  -- pass over the body.
  unfold sha256.compress_loop0.body
  rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt)]
  simp only [ho, ho1, ho2, ho3, hread0, hread1, hread2, hread3, hupd, hi', lift, bind_tc_ok]

/-- **The exit branch.** At `16 ≤ i.val` the body returns the incoming array unchanged. This is
    where the loop's invariant is read off. -/
theorem loadBody_done (block : Array Std.U8 64#usize) (w : Array Std.U32 64#usize)
    (i : Std.Usize) (hge : 16 ≤ i.val) :
    sha256.compress_loop0.body block w i = Result.ok (ControlFlow.done w) := by
  unfold sha256.compress_loop0.body
  rw [if_neg (by
    show ¬ i.val < (16#usize : Std.Usize).val
    have h16 : (16#usize : Std.Usize).val = 16 := by norm_num
    omega)]

/-- **The body has no failure path**, at any index: below the guard it continues with the written
    array, above it with `done`. -/
theorem loadBody_total (block : Array Std.U8 64#usize) (w : Array Std.U32 64#usize)
    (i : Std.Usize) :
    (∃ w' : Array Std.U32 64#usize, ∃ i' : Std.Usize,
        sha256.compress_loop0.body block w i = Result.ok (ControlFlow.cont (w', i')))
      ∨ sha256.compress_loop0.body block w i = Result.ok (ControlFlow.done w) := by
  by_cases hlt : i.val < 16
  · obtain ⟨v, i', h, _, _⟩ := loadBody_step block w i hlt
    exact Or.inl ⟨w.set i v, i', h⟩
  · exact Or.inr (loadBody_done block w i (by omega))

/-! ## §2 · The invariant

`loadVec block n` is the schedule the loop holds after `n` visits: the first `n` big-endian words of
the block, zero above. `rowVec (bytesToWords block) 0` is the corpus's own spelling of the same
vector (`Crypto/Sha256Certificate.lean:313`), and it is the invariant premise
`apeiron_word.sched_loop1` takes. -/

/-- The sixty-four row schedule the loading loop has written after `n` visits. -/
def loadVec (block : Array Std.U8 64#usize) (n : Nat) : Vector Word 64 :=
  Vector.ofFn fun j : Fin 64 => if j.val < n then (bytesToWords block).getD j.val 0 else 0

theorem loadVec_getD (block : Array Std.U8 64#usize) (n i : Nat) (hi : i < 64) :
    (loadVec block n).getD i 0 = if i < n then (bytesToWords block).getD i 0 else 0 :=
  ofFn_getD _ i hi

/-- After sixteen visits the schedule is the corpus's `rowVec` at zero extension steps: the block
    words below row sixteen, zero above. -/
theorem loadVec16_eq_rowVec0 (block : Array Std.U8 64#usize) :
    loadVec block 16 = rowVec (bytesToWords block) 0 := by
  refine vec_getD_eq _ _ 0 fun i hi => ?_
  rw [loadVec_getD _ 16 i hi, rowVec_getD _ 0 i hi]
  by_cases h16 : i < 16
  · rw [if_pos h16, if_pos (by omega), schedRow, dif_pos h16,
      bytesToWords_getD block i (by omega)]
  · rw [if_neg h16, if_neg (by omega)]

/-- The loop invariant of the loading loop, at the measure `16 - i.val`: the rows written so far
    are the block's big-endian words, nothing above. -/
def loadInv (block : Array Std.U8 64#usize)
    (p : Array Std.U32 64#usize × Std.Usize) : Prop :=
  p.2.val ≤ 16 ∧ wView p.1 = loadVec block p.2.val

/-- **One loading step preserves the invariant.** Writing row `i` with the big-endian word of bytes
    `4i … 4i+3` makes the schedule read as `loadVec` at the next counter, because the write touches
    row `i` and no other row. -/
theorem loadStep_inv (block : Array Std.U8 64#usize) (w : Array Std.U32 64#usize)
    (i : Std.Usize) (v : Std.U32) (hlt : i.val < 16)
    (hinv : wView w = loadVec block i.val)
    (hv : v.bv = beWord16 block ⟨i.val, by omega⟩) :
    wView (w.set i v) = loadVec block (i.val + 1) := by
  refine vec_getD_eq _ _ 0 fun j hj => ?_
  have hgj : wGet w j = (loadVec block i.val).getD j 0 := by
    rw [← wView_getD w j hj, hinv]
  have hlt64 : i.val < 64 := by omega
  rw [wView_getD _ j hj, wGet_set w i v hlt64 j,
    loadVec_getD block (i.val + 1) j hj, hgj, loadVec_getD block i.val j hj]
  by_cases hji : i.val = j
  · subst hji
    rw [if_pos rfl, if_pos (by omega)]
    rw [hv, bytesToWords_getD block i.val (by omega)]
  · simp only [if_neg hji]
    by_cases hjlt : j < i.val + 1
    · rw [if_pos hjlt, if_pos (by omega)]
    · rw [if_neg hjlt, if_neg (by omega)]

/-! ## §3 · The loop

`load_loop0` is the loop at an arbitrary zero-extended start schedule; `load_loop0_repeat` is the
same value link at the schedule array `sha256.compress` actually starts from. Both are
**success-conditioned** — `loop` is a `partial_fixpoint`, so no statement about it can be stronger
than this. -/

/-- **The loading loop.** Run from the all-zero schedule at counter zero, a successful
    `sha256.compress_loop0 block w0 0#usize` yields the schedule whose rows are the block's sixteen
    consecutive big-endian words, read through `bytesToWords`, and zero above: `loadVec block 16`.

    Success-conditioned: it says what a successful run computes, not that one exists. -/
theorem load_loop0 (block : Array Std.U8 64#usize) (w0 : Array Std.U32 64#usize)
    (w' : Array Std.U32 64#usize) (hinv : wView w0 = loadVec block 0)
    (h : sha256.compress_loop0 block w0 0#usize = Result.ok w') :
    wView w' = loadVec block 16 := by
  have hloop :
      loop (fun p : Array Std.U32 64#usize × Std.Usize =>
        sha256.compress_loop0.body block p.1 p.2) (w0, 0#usize)
        ⦃ fun y => wView y = loadVec block 16 ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : Array Std.U32 64#usize × Std.Usize => 16 - p.2.val)
      (inv := loadInv block)
      (post := fun y => wView y = loadVec block 16)
    · intro p hp
      obtain ⟨hle, hw⟩ := hp
      by_cases hlt : p.2.val < 16
      · obtain ⟨v, i24, hb, hi24v, hv⟩ := loadBody_step block p.1 p.2 hlt
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        show (i24.val ≤ 16 ∧ wView (p.1.set p.2 v) = loadVec block i24.val) ∧
          16 - i24.val < 16 - p.2.val
        refine ⟨⟨?_, ?_⟩, ?_⟩
        · rw [hi24v]; omega
        · have hstep := loadStep_inv block p.1 p.2 v hlt hw hv
          rw [← hi24v] at hstep
          exact hstep
        · rw [hi24v]; omega
      · rw [WP.spec_equiv_exists]
        refine ⟨ControlFlow.done p.1, loadBody_done block p.1 p.2 (by omega), ?_⟩
        show wView p.1 = loadVec block 16
        refine vec_getD_eq _ _ 0 fun j hj => ?_
        have hle16 : p.2.val = 16 := by omega
        rw [hle16] at hw
        rw [hw]
    · refine ⟨?_, ?_⟩
      · show (0 : Nat) ≤ 16
        omega
      · exact hinv
  have hloop' :
      loop (fun p : Array Std.U32 64#usize × Std.Usize =>
        sha256.compress_loop0.body block p.1 p.2) (w0, 0#usize) = Result.ok w' := by
    simpa only [sha256.compress_loop0] using h
  obtain ⟨y, hy, hPy⟩ := WP.spec_imp_exists hloop
  rw [hloop'] at hy
  have hw'y : w' = y := Result.ok_injective hy
  rw [← hw'y] at hPy
  exact hPy

/-- **At the schedule `sha256.compress` starts from.** `sha256.compress`
    (`ApeironWord.lean:15396`) calls the loading loop on `Array.repeat 64#usize 0#u32` at counter
    `0#usize`; a successful run yields `rowVec (bytesToWords block) 0`, the vector
    `apeiron_word.sched_loop1` takes as its invariant. -/
theorem load_loop0_repeat (block : Array Std.U8 64#usize)
    (w' : Array Std.U32 64#usize)
    (h : sha256.compress_loop0 block (Array.repeat 64#usize 0#u32) 0#usize = Result.ok w') :
    wView w' = rowVec (bytesToWords block) 0 := by
  have hzero : wView (Array.repeat 64#usize 0#u32) = loadVec block 0 := by
    refine vec_getD_eq _ _ 0 fun j hj => ?_
    rw [wView_getD _ j hj, wGet_zero j hj, loadVec_getD block 0 j hj, if_neg (by omega)]
  exact (load_loop0 block (Array.repeat 64#usize 0#u32) w' hzero h).trans
    (loadVec16_eq_rowVec0 block)

/-- **The loading loop as sixteen consecutive big-endian reads.** Row `k` of the returned schedule
    is the block's `k`-th word for every `k < 16`, and zero above: the block-to-word loading, one
    word per visit. -/
theorem load_loop0_wGet (block : Array Std.U8 64#usize)
    (w' : Array Std.U32 64#usize)
    (h : sha256.compress_loop0 block (Array.repeat 64#usize 0#u32) 0#usize = Result.ok w')
    (k : Nat) (hk : k < 64) :
    wGet w' k = if k < 16 then (bytesToWords block).getD k 0 else 0 := by
  have hmain := load_loop0_repeat block w' h
  have hrow : (wView w').getD k 0 = if k < 16 then (bytesToWords block).getD k 0 else 0 := by
    rw [hmain, rowVec_getD _ 0 k hk]
    by_cases hk16 : k < 16
    · rw [if_pos (by omega), if_pos hk16, schedRow, dif_pos hk16]
    · rw [if_neg (by omega), if_neg hk16]
  rw [← wView_getD w' k hk, hrow]

/-- The same read, at a block-word row: `k < 16` is the big-endian word of bytes `4k … 4k+3`. -/
theorem load_loop0_beWord16 (block : Array Std.U8 64#usize)
    (w' : Array Std.U32 64#usize)
    (h : sha256.compress_loop0 block (Array.repeat 64#usize 0#u32) 0#usize = Result.ok w')
    (k : Nat) (hk16 : k < 16) :
    wGet w' k = beWord16 block ⟨k, hk16⟩ := by
  have hmain := load_loop0_wGet block w' h k (by omega)
  rw [hmain, if_pos hk16, bytesToWords_getD block k hk16]

/-- The same, read through the array's own list read at `j`. -/
theorem load_loop0_getD (block : Array Std.U8 64#usize)
    (w' : Array Std.U32 64#usize)
    (h : sha256.compress_loop0 block (Array.repeat 64#usize 0#u32) 0#usize = Result.ok w')
    (j : Nat) (hj : j < 64) :
    (w'.val.getD j (0#u32) : Std.U32).bv
      = if j < 16 then (bytesToWords block).getD j 0 else 0 := by
  have hk := load_loop0_wGet block w' h j hj
  rw [← hk]
  rfl

/-! ## §4 · Kernel checks: axiom cones -/

#print axioms Icarus.Extraction.LoadBlock.loadBody_step
#print axioms Icarus.Extraction.LoadBlock.loadBody_done
#print axioms Icarus.Extraction.LoadBlock.loadBody_total
#print axioms Icarus.Extraction.LoadBlock.loadVec_getD
#print axioms Icarus.Extraction.LoadBlock.loadVec16_eq_rowVec0
#print axioms Icarus.Extraction.LoadBlock.loadStep_inv
#print axioms Icarus.Extraction.LoadBlock.load_loop0
#print axioms Icarus.Extraction.LoadBlock.load_loop0_repeat
#print axioms Icarus.Extraction.LoadBlock.load_loop0_wGet
#print axioms Icarus.Extraction.LoadBlock.load_loop0_beWord16
#print axioms Icarus.Extraction.LoadBlock.load_loop0_getD

end LoadBlock
end Extraction
end Icarus