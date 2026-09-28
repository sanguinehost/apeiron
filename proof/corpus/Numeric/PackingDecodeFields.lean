/-
Refinement of the extracted canonical packing decoder's field reader
.

`Numeric/PackingRefinement.lean` names the structural identity of `decode_block_fields` as residual
item 1: the byte-level nibble/bit-plane unpack, which calls the generic `unpack_code` per weight and
not `word_q4_k_codes`/`word_q6_k_codes`. Its per-code reference law is `unpack_code_ref`
(`Numeric/PackingUnpack.lean`); this module lifts it across the extracted loops of
`packing.Packing.decode_block_fields` (`proof/ApeironWord.lean`).

Proved here:

* `unpackRefNat` / `codeSumNat`: the `unpackRef` reference re-indexed by `Nat` (definitionally
  `unpackRef` at `.val`), so the `List.range`-map invariants need no `Usize` coercions;
  `unpackRef_eq_nat` is the bridge;
* `codes_loop2_spec`: the extracted code-unpack loop `decode_block_fields_loop2` on the success
  domain returns `codes0.value ++ (List.range block).map (unpackRefNat …)`, one `unpack_code` per
  weight, via `unpack_code_ref`; the identical bodies of `loop3`/`loop7`/`loop8` transfer by
  `rfl` (`codes_loop3_spec`, `codes_loop7_spec`, `codes_loop8_spec`);
* `copy_loop5_spec` / `copy_loop0_spec`: the byte-copy loops `decode_block_fields_loop5` /
  `decode_block_fields_loop0` append exactly `nbytes` consecutive payload bytes;
* `rot_loop1_spec` / `rot_loop6_spec`: the nibble-swap loops `decode_block_fields_loop1` /
  `decode_block_fields_loop6` rotate every byte left by four bits in place;
* `coeff_loop4_spec`: the coefficient loop `decode_block_fields_loop4` appends exactly `groups`
  `(a, b)` pairs, each the signed (`Int.bmod · 256`) or unsigned coefficient byte at the group's
  strand position, and advances the strand by `groups·coeffStep`.

All nine `decode_block_fields_loop{0..8}` bodies are therefore covered.

* `decode_block_fields_spec`: the top-level assembly on the non-terminal face
  (`is_terminal self = ok false`), composing the scale/min/coefficient reads and the four loop calls
  across the extracted `match`/`if` structure into the `CanonicalBlock` whose `codes` are
  `(List.range block).map (unpackRef …)`. Its sub-call results are the loop specs above, so the
  theorem assumes the loop-spec conclusions (the value laws) and discharges the branch structure.

## Residual

`decode_block_spec` — the composition of `decode_block_fields_spec` with the `epilogue` folds of
`Numeric/PackingEpilogueFold.lean` for the `Fixed` width — is **not** closed here: `epilogue`'s own
top-level `match` assembly (the `scale_to_fixed`/`has_min`/width dispatch around
`epilogue_loop1`/`epilogue_loop2`) is still residual in `Numeric/PackingRefinement.lean`, and the
`Terminal` (`F16`/`F32`) epilogue loop `epilogue_loop0` is uncovered. The open goal is recorded at
the end of this file.

## Complexity

Each loop visits its `block`/`nbytes`/`groups` index once with `O(1)` work: `O(block·total/8)` and
`O(block)` time for the copy/code loops, `O(nbytes)` for the nibble swap. The figures name the loops.

## Trust boundary

Reaches the extracted `packing.Packing.decode_block_fields_loop{0..8}(.body)` of
`proof/ApeironWord.lean`. Does not cover rustc/LLVM codegen of `src/format/packing.rs`, the kernel
ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.PackingUnpack
import Numeric.PackingEpilogueFold

namespace apeiron_word

open Aeneas Aeneas.Std Result WP
open scoped BigOperators

/-! ## The `Nat`-indexed reference and its bridge to `unpackRef` -/

/-- The bit-plane sum of weight `i` with `Nat` indices; `codeSum` at `.val`. -/
def codeSumNat (self : packing.Packing) (area : Slice U8) (i total : Nat) : Nat :=
  codeBits self.plane_order area i total total

/-- The reference code value with `Nat` indices; `unpackRef` at `.val`. -/
def unpackRefNat (self : packing.Packing) (area : Slice U8) (i total : Nat) : I64 :=
  if self.signed_code = true ∧ total < 64
      ∧ (codeSumNat self area i total).testBit (total - 1) = true
  then ⟨BitVec.ofInt 64 ((codeSumNat self area i total : Int) - 2 ^ total)⟩
  else ⟨BitVec.ofNat 64 (codeSumNat self area i total)⟩

/-- `unpackRef` at a `Usize` index is `unpackRefNat` at its value. -/
lemma unpackRef_eq_nat (self : packing.Packing) (area : Slice U8) (i total : Usize) :
    unpackRef self area i total = unpackRefNat self area i.val total.val := by
  simp only [unpackRef, unpackRefNat, codeSum, codeSumNat]

/-- The descriptor of the code-unpack loop body. -/
def codePacking (cw : packing.CodeWidth) (i : U8) (i1 : U32) (i2 : U32)
    (sk : packing.ScaleKind) (b b1 : Bool) (i3 : I32) (b2 : Bool)
    (po : packing.PlaneOrder) (no : packing.NibbleOrder) : packing.Packing :=
  { width := cw, high_planes := i, block := i1, group := i2, scale_kind := sk, has_min := b,
    signed_code := b1, bias := i3, signed_coeff := b2, plane_order := po, nibble_order := no }

@[simp] lemma codePacking_signed_code (cw i i1 i2 sk b b1 i3 b2 po no) :
    (codePacking cw i i1 i2 sk b b1 i3 b2 po no).signed_code = b1 := rfl

@[simp] lemma Vec_deref_val (v : alloc.vec.Vec U8) :
    (alloc.vec.Vec.deref v).val = v.val := by
  simp [alloc.vec.Vec.deref]

@[simp] lemma Vec_deref_length (v : alloc.vec.Vec U8) :
    (alloc.vec.Vec.deref v).val.length = v.val.length := by
  rw [Vec_deref_val]

/-! ## The code-unpack loop `decode_block_fields_loop2` -/

/-- The code-unpack loop invariant: `i4` codes appended, each the `unpackRefNat` reference of its
weight. -/
def CodesLoopInv (self : packing.Packing) (area : alloc.vec.Vec U8) (total block : Usize)
    (codes0 codes : alloc.vec.Vec I64) (i4 : Usize) : Prop :=
  i4.val ≤ block.val ∧
  codes.val = codes0.val ++
    (List.range i4.val).map
      (fun k => unpackRefNat self (alloc.vec.Vec.deref area) k total.val)

/-- One extracted code-unpack iteration: `unpack_code` at weight `i4` is the reference value,
appended; the index advances. -/
theorem codes_loop2_body_step
    (cw : packing.CodeWidth) (i : U8) (i1 : U32) (i2 : U32) (sk : packing.ScaleKind)
    (b b1 : Bool) (i3 : I32) (b2 : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (total block : Usize) (area : alloc.vec.Vec U8) (codes0 codes : alloc.vec.Vec I64)
    (i4 : Usize)
    (htot : total.val ≤ 64)
    (hsigned : b1 = true → 0 < total.val ∧ (total.val < 64 → total.val ≤ 62))
    (hmul : i4.val * total.val + total.val ≤ Usize.max)
    (harea : (i4.val * total.val + total.val - 1) / 8 < area.val.length)
    (hcap : codes0.val.length + block.val ≤ Usize.max)
    (hinv : CodesLoopInv (codePacking cw i i1 i2 sk b b1 i3 b2 po no) area total block codes0
      codes i4)
    (hlt : i4.val < block.val) :
    ∃ (codes2 : alloc.vec.Vec I64) (i5 : Usize),
      packing.Packing.decode_block_fields_loop2.body cw i i1 i2 sk b b1 i3 b2 po no total block
          area codes i4 = ok (ControlFlow.cont (codes2, i5)) ∧
      CodesLoopInv (codePacking cw i i1 i2 sk b b1 i3 b2 po no) area total block codes0 codes2 i5 ∧
      i5.val = i4.val + 1 := by
  obtain ⟨-, hmap⟩ := hinv
  have hP : codePacking cw i i1 i2 sk b b1 i3 b2 po no
      = ({ width := cw, high_planes := i, block := i1, group := i2, scale_kind := sk,
           has_min := b, signed_code := b1, bias := i3, signed_coeff := b2, plane_order := po,
           nibble_order := no } : packing.Packing) := rfl
  have hunpack : packing.Packing.unpack_code
      (codePacking cw i i1 i2 sk b b1 i3 b2 po no) (alloc.vec.Vec.deref area) i4 total
      = ok (unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
          (alloc.vec.Vec.deref area) i4.val total.val) := by
    rw [← unpackRef_eq_nat]
    exact unpack_code_ref (codePacking cw i i1 i2 sk b b1 i3 b2 po no) (alloc.vec.Vec.deref area)
      i4 total (by simpa [codePacking_signed_code] using hsigned) htot hmul (by simpa using harea)
  have hclen : codes.val.length = codes0.val.length + i4.val := by
    rw [hmap]; simp
  have hcodelen : codes.val.length < Usize.max := by
    rw [hclen]
    calc codes0.val.length + i4.val < codes0.val.length + block.val :=
          Nat.add_lt_add_left hlt _
      _ ≤ Usize.max := hcap
  obtain ⟨codes2, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec codes
      (unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no) (alloc.vec.Vec.deref area)
        i4.val total.val) hcodelen)
  obtain ⟨i5, hadd, hi5v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i4) (y := 1#usize) (by scalar_tac))
  have hi5' : i5.val = i4.val + 1 := by rw [hi5v]; simp
  have hbody : packing.Packing.decode_block_fields_loop2.body cw i i1 i2 sk b b1 i3 b2 po no
      total block area codes i4 = ok (ControlFlow.cont (codes2, i5)) := by
    unfold packing.Packing.decode_block_fields_loop2.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), ← hP]
    dsimp only
    rw [hunpack]
    simp only [bind_tc_ok]
    rw [hpush]
    simp only [bind_tc_ok]
    rw [hadd]
    simp only [bind_tc_ok]
  refine ⟨codes2, i5, hbody, ?_, hi5'⟩
  refine ⟨by omega, ?_⟩
  have hsp : (List.range (i4.val + 1)).map
        (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
          (alloc.vec.Vec.deref area) k total.val)
      = (List.range i4.val).map
          (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
            (alloc.vec.Vec.deref area) k total.val)
        ++ [unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
            (alloc.vec.Vec.deref area) i4.val total.val] := by
    rw [List.range_succ, List.map_append]
    simp
  rw [hpushv, hmap, hi5', hsp, List.append_assoc]

/-- The extracted code-unpack loop on the success domain: `block` `unpack_code` references. -/
theorem codes_loop2_spec
    (cw : packing.CodeWidth) (i : U8) (i1 : U32) (i2 : U32) (sk : packing.ScaleKind)
    (b b1 : Bool) (i3 : I32) (b2 : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (total block : Usize) (area : alloc.vec.Vec U8) (codes0 : alloc.vec.Vec I64)
    (htot : total.val ≤ 64)
    (hsigned : b1 = true → 0 < total.val ∧ (total.val < 64 → total.val ≤ 62))
    (hmul : ∀ k, k < block.val → k * total.val + total.val ≤ Usize.max)
    (harea : ∀ k, k < block.val → (k * total.val + total.val - 1) / 8 < area.val.length)
    (hcap : codes0.val.length + block.val ≤ Usize.max) :
    WP.spec
      (packing.Packing.decode_block_fields_loop2 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes0 0#usize)
      (fun codes => ∃ codes' : alloc.vec.Vec I64, codes = codes' ∧
        codes'.val = codes0.val ++
          (List.range block.val).map
            (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
              (alloc.vec.Vec.deref area) k total.val)) := by
  have hinv0 : CodesLoopInv (codePacking cw i i1 i2 sk b b1 i3 b2 po no) area total block codes0
      codes0 0#usize := by
    refine ⟨by simp, by simp⟩
  unfold packing.Packing.decode_block_fields_loop2
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize => block.val - s.2.val)
    (inv := fun s => CodesLoopInv (codePacking cw i i1 i2 sk b b1 i3 b2 po no) area total block
      codes0 s.1 s.2)
    (post := fun (codes : alloc.vec.Vec I64) => ∃ codes' : alloc.vec.Vec I64,
      codes = codes' ∧ codes'.val = codes0.val ++
      (List.range block.val).map
        (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
          (alloc.vec.Vec.deref area) k total.val))
  · intro x hx
    rcases x with ⟨codes, i4⟩
    dsimp only
    dsimp only at hx
    by_cases hlt : i4.val < block.val
    · obtain ⟨codes2, i5, hbody, hinv2, hi5⟩ :=
        codes_loop2_body_step cw i i1 i2 sk b b1 i3 b2 po no total block area codes0 codes i4
          htot hsigned (hmul i4.val hlt) (harea i4.val hlt) hcap hx hlt
      rw [hbody, WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show block.val - i5.val < block.val - i4.val
      rw [hi5]
      omega
    · have hnot : ¬ (i4 < block) := by rw [UScalar.lt_equiv]; exact hlt
      have hbody : packing.Packing.decode_block_fields_loop2.body cw i i1 i2 sk b b1 i3 b2 po no
          total block area codes i4 = ok (ControlFlow.done codes) := by
        unfold packing.Packing.decode_block_fields_loop2.body
        rw [if_neg hnot]
      rw [hbody, WP.spec_ok]
      have hi4eq : i4.val = block.val := by
        have := hx.1; omega
      obtain ⟨-, hmap⟩ := hx
      rw [hi4eq] at hmap
      exact ⟨codes, rfl, hmap⟩
  · exact hinv0

/-! ### The four identical code-unpack loops

`decode_block_fields_loop3`, `loop7` and `loop8` have textually identical bodies to `loop2`, so the
same spec transfers by `rfl`. -/

theorem codes_loop3_eq (cw i i1 i2 sk b b1 i3 b2 po no total block area codes i4) :
    packing.Packing.decode_block_fields_loop3 cw i i1 i2 sk b b1 i3 b2 po no total block area codes
        i4
      = packing.Packing.decode_block_fields_loop2 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes i4 := rfl

theorem codes_loop7_eq (cw i i1 i2 sk b b1 i3 b2 po no total block area codes i4) :
    packing.Packing.decode_block_fields_loop7 cw i i1 i2 sk b b1 i3 b2 po no total block area codes
        i4
      = packing.Packing.decode_block_fields_loop2 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes i4 := rfl

theorem codes_loop8_eq (cw i i1 i2 sk b b1 i3 b2 po no total block area codes i4) :
    packing.Packing.decode_block_fields_loop8 cw i i1 i2 sk b b1 i3 b2 po no total block area codes
        i4
      = packing.Packing.decode_block_fields_loop2 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes i4 := rfl

/-- The `decode_block_fields_loop3` spec — `loop2`'s law on the identical body. -/
theorem codes_loop3_spec
    (cw : packing.CodeWidth) (i : U8) (i1 : U32) (i2 : U32) (sk : packing.ScaleKind)
    (b b1 : Bool) (i3 : I32) (b2 : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (total block : Usize) (area : alloc.vec.Vec U8) (codes0 : alloc.vec.Vec I64)
    (htot : total.val ≤ 64)
    (hsigned : b1 = true → 0 < total.val ∧ (total.val < 64 → total.val ≤ 62))
    (hmul : ∀ k, k < block.val → k * total.val + total.val ≤ Usize.max)
    (harea : ∀ k, k < block.val → (k * total.val + total.val - 1) / 8 < area.val.length)
    (hcap : codes0.val.length + block.val ≤ Usize.max) :
    WP.spec
      (packing.Packing.decode_block_fields_loop3 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes0 0#usize)
      (fun codes => ∃ codes' : alloc.vec.Vec I64, codes = codes' ∧
        codes'.val = codes0.val ++
          (List.range block.val).map
            (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
              (alloc.vec.Vec.deref area) k total.val)) := by
  rw [codes_loop3_eq]
  exact codes_loop2_spec cw i i1 i2 sk b b1 i3 b2 po no total block area codes0 htot hsigned hmul
    harea hcap

/-- The `decode_block_fields_loop7` spec — `loop2`'s law on the identical body. -/
theorem codes_loop7_spec
    (cw : packing.CodeWidth) (i : U8) (i1 : U32) (i2 : U32) (sk : packing.ScaleKind)
    (b b1 : Bool) (i3 : I32) (b2 : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (total block : Usize) (area : alloc.vec.Vec U8) (codes0 : alloc.vec.Vec I64)
    (htot : total.val ≤ 64)
    (hsigned : b1 = true → 0 < total.val ∧ (total.val < 64 → total.val ≤ 62))
    (hmul : ∀ k, k < block.val → k * total.val + total.val ≤ Usize.max)
    (harea : ∀ k, k < block.val → (k * total.val + total.val - 1) / 8 < area.val.length)
    (hcap : codes0.val.length + block.val ≤ Usize.max) :
    WP.spec
      (packing.Packing.decode_block_fields_loop7 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes0 0#usize)
      (fun codes => ∃ codes' : alloc.vec.Vec I64, codes = codes' ∧
        codes'.val = codes0.val ++
          (List.range block.val).map
            (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
              (alloc.vec.Vec.deref area) k total.val)) := by
  rw [codes_loop7_eq]
  exact codes_loop2_spec cw i i1 i2 sk b b1 i3 b2 po no total block area codes0 htot hsigned hmul
    harea hcap

/-- The `decode_block_fields_loop8` spec — `loop2`'s law on the identical body. -/
theorem codes_loop8_spec
    (cw : packing.CodeWidth) (i : U8) (i1 : U32) (i2 : U32) (sk : packing.ScaleKind)
    (b b1 : Bool) (i3 : I32) (b2 : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (total block : Usize) (area : alloc.vec.Vec U8) (codes0 : alloc.vec.Vec I64)
    (htot : total.val ≤ 64)
    (hsigned : b1 = true → 0 < total.val ∧ (total.val < 64 → total.val ≤ 62))
    (hmul : ∀ k, k < block.val → k * total.val + total.val ≤ Usize.max)
    (harea : ∀ k, k < block.val → (k * total.val + total.val - 1) / 8 < area.val.length)
    (hcap : codes0.val.length + block.val ≤ Usize.max) :
    WP.spec
      (packing.Packing.decode_block_fields_loop8 cw i i1 i2 sk b b1 i3 b2 po no total block area
        codes0 0#usize)
      (fun codes => ∃ codes' : alloc.vec.Vec I64, codes = codes' ∧
        codes'.val = codes0.val ++
          (List.range block.val).map
            (fun k => unpackRefNat (codePacking cw i i1 i2 sk b b1 i3 b2 po no)
              (alloc.vec.Vec.deref area) k total.val)) := by
  rw [codes_loop8_eq]
  exact codes_loop2_spec cw i i1 i2 sk b b1 i3 b2 po no total block area codes0 htot hsigned hmul
    harea hcap

/-! ## The byte-copy loops `decode_block_fields_loop5` / `decode_block_fields_loop0` -/

/-- The copy-loop invariant: `j` consecutive payload bytes appended from `start`. -/
def CopyLoopInv (raw : Slice U8) (start nbytes : Usize) (area0 area : alloc.vec.Vec U8)
    (j : Usize) : Prop :=
  j.val ≤ nbytes.val ∧
  area.val = area0.val ++
    (List.range j.val).map (fun k => raw.val.getD (start.val + k) 0#u8)

/-- One extracted copy iteration: `raw[start + j]` is appended, the index advances. -/
theorem copy_loop5_body_step (raw : Slice U8) (start nbytes : Usize)
    (area0 area : alloc.vec.Vec U8) (j : Usize)
    (hbound : start.val + j.val < raw.val.length)
    (hcap : area0.val.length + nbytes.val ≤ Usize.max)
    (hinv : CopyLoopInv raw start nbytes area0 area j)
    (hlt : j.val < nbytes.val) :
    ∃ (area2 : alloc.vec.Vec U8) (j1 : Usize),
      packing.Packing.decode_block_fields_loop5.body raw start nbytes area j
          = ok (ControlFlow.cont (area2, j1)) ∧
      CopyLoopInv raw start nbytes area0 area2 j1 ∧ j1.val = j.val + 1 := by
  obtain ⟨hjle, hmap⟩ := hinv
  obtain ⟨i, hadd, hiv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := start) (y := j) (by scalar_tac))
  have hb : i.val < raw.val.length := by rw [hiv]; exact hbound
  obtain ⟨x, hindex, hxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw i hb)
  have halen : area.val.length = area0.val.length + j.val := by
    rw [hmap]; simp
  obtain ⟨area2, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec area x (by rw [halen]; omega))
  obtain ⟨j1, hadd1, hj1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
  have hj1' : j1.val = j.val + 1 := by rw [hj1v]; simp
  have hbody : packing.Packing.decode_block_fields_loop5.body raw start nbytes area j
      = ok (ControlFlow.cont (area2, j1)) := by
    unfold packing.Packing.decode_block_fields_loop5.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), hadd]
    simp only [bind_tc_ok]
    rw [hindex]
    simp only [bind_tc_ok]
    rw [hpush]
    simp only [bind_tc_ok]
    rw [hadd1]
    simp only [bind_tc_ok]
  refine ⟨area2, j1, hbody, ?_, hj1'⟩
  refine ⟨by omega, ?_⟩
  have hxval : x = raw.val.getD (start.val + j.val) 0#u8 := by
    rw [hxv]
    simp only [hiv]
    exact (getD_eq_getElem raw.val (start.val + j.val) 0#u8 hbound).symm
  rw [hpushv, hmap, hj1', List.range_succ, List.map_append, hxval]
  simp only [List.append_assoc, List.map_cons, List.map_nil]

/-- The extracted copy loop on the success domain: `nbytes` payload bytes appended. -/
theorem copy_loop5_spec (raw : Slice U8) (start nbytes : Usize) (area0 : alloc.vec.Vec U8)
    (hbound : ∀ k, k < nbytes.val → start.val + k < raw.val.length)
    (hcap : area0.val.length + nbytes.val ≤ Usize.max) :
    WP.spec (packing.Packing.decode_block_fields_loop5 raw start nbytes area0 0#usize)
      (fun area => ∃ area' : alloc.vec.Vec U8, area = area' ∧
        area'.val = area0.val ++
          (List.range nbytes.val).map (fun k => raw.val.getD (start.val + k) 0#u8)) := by
  have hinv0 : CopyLoopInv raw start nbytes area0 area0 0#usize := by
    refine ⟨by simp, by simp⟩
  unfold packing.Packing.decode_block_fields_loop5
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec U8 × Usize => nbytes.val - s.2.val)
    (inv := fun s => CopyLoopInv raw start nbytes area0 s.1 s.2)
    (post := fun (area : alloc.vec.Vec U8) => ∃ area' : alloc.vec.Vec U8, area = area' ∧
      area'.val = area0.val ++
        (List.range nbytes.val).map (fun k => raw.val.getD (start.val + k) 0#u8))
  · intro x hx
    rcases x with ⟨area, j⟩
    dsimp only
    dsimp only at hx
    by_cases hlt : j.val < nbytes.val
    · obtain ⟨area2, j1, hbody, hinv2, hj1⟩ :=
        copy_loop5_body_step raw start nbytes area0 area j (hbound j.val hlt) hcap hx hlt
      rw [hbody, WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show nbytes.val - j1.val < nbytes.val - j.val
      rw [hj1]
      omega
    · have hnot : ¬ (j < nbytes) := by rw [UScalar.lt_equiv]; exact hlt
      have hbody : packing.Packing.decode_block_fields_loop5.body raw start nbytes area j
          = ok (ControlFlow.done area) := by
        unfold packing.Packing.decode_block_fields_loop5.body
        rw [if_neg hnot]
      rw [hbody, WP.spec_ok]
      have hjeq : j.val = nbytes.val := by have := hx.1; omega
      obtain ⟨-, hmap⟩ := hx
      rw [hjeq] at hmap
      exact ⟨area, rfl, hmap⟩
  · exact hinv0

/-- The zero-offset copy loop `decode_block_fields_loop0` is the copy loop at `start = 0`. -/
theorem copy_loop0_eq (raw : Slice U8) (nbytes : Usize) (area : alloc.vec.Vec U8) (j : Usize) :
    packing.Packing.decode_block_fields_loop0 raw nbytes area j
      = packing.Packing.decode_block_fields_loop5 raw 0#usize nbytes area j := rfl

/-- The extracted zero-offset copy loop on the success domain. -/
theorem copy_loop0_spec (raw : Slice U8) (nbytes : Usize) (area0 : alloc.vec.Vec U8)
    (hbound : ∀ k, k < nbytes.val → k < raw.val.length)
    (hcap : area0.val.length + nbytes.val ≤ Usize.max) :
    WP.spec (packing.Packing.decode_block_fields_loop0 raw nbytes area0 0#usize)
      (fun area => ∃ area' : alloc.vec.Vec U8, area = area' ∧
        area'.val = area0.val ++
          (List.range nbytes.val).map (fun k => raw.val.getD k 0#u8)) := by
  have h := copy_loop5_spec raw 0#usize nbytes area0 (by simpa using hbound) hcap
  rw [copy_loop0_eq]
  simpa using h

/-! ## The nibble-swap loops `decode_block_fields_loop1` / `decode_block_fields_loop6` -/

/-- The nibble swap: `u8::rotate_left(4)`, the byte's two halves exchanged. -/
def nibbleRot (b : U8) : U8 := core.num.U8.rotate_left b 4#u32

lemma nibbleRot_eq (b : U8) : core.num.U8.rotate_left b 4#u32 = nibbleRot b := rfl

/-- Reading weight `j < A.length` of the prefix-rotated list is reading `A[j]`. -/
lemma map_take_drop_getElem (A : List U8) (rot : U8 → U8) {j : Nat} (hj : j < A.length)
    (h : j < ((A.take j).map rot ++ A.drop j).length) :
    ((A.take j).map rot ++ A.drop j)[j] = A[j] := by
  have hlen : ((A.take j).map rot).length = j := by
    simp [Nat.min_eq_left (le_of_lt hj)]
  rw [List.getElem_append_right (le_of_eq hlen)]
  rw [List.getElem_drop]
  congr 1
  omega

/-- Rotating the `j`-th byte of the prefix-rotated prefix adds the `j`-th rotated byte. -/
lemma map_take_succ (A : List U8) (rot : U8 → U8) {j : Nat} (hj : j < A.length) :
    (A.take (j + 1)).map rot = (A.take j).map rot ++ [rot A[j]] := by
  rw [List.take_add_one, List.getElem?_eq_getElem hj, List.map_append, Option.toList_some,
    List.map_singleton]

/-- `List.set` at `j` of the prefix-rotated list. -/
lemma rot_set_eq (A : List U8) {j : Nat} (hj : j < A.length) (i1 : U8) :
    ((A.take j).map nibbleRot ++ A.drop j).set j i1
      = (A.take j).map nibbleRot ++ i1 :: A.drop (j + 1) := by
  have hlen : ((A.take j).map nibbleRot).length = j := by
    simp [Nat.min_eq_left (le_of_lt hj)]
  have hbound : j < ((A.take j).map nibbleRot ++ A.drop j).length := by
    rw [List.length_append, hlen, List.length_drop]; omega
  rw [List.set_eq_take_cons_drop _ hbound]
  rw [List.take_append_of_le_length (by rw [hlen])]
  rw [List.take_of_length_le (le_of_eq hlen)]
  rw [List.drop_append]
  rw [show j + 1 - ((A.take j).map nibbleRot).length = 1 by rw [hlen]; omega]
  rw [List.drop_eq_nil_of_le (by rw [hlen]; omega)]
  simp only [List.nil_append]
  rw [List.drop_drop]

/-- The rotate-loop invariant: the first `j` bytes are the prefix-rotated payload. -/
def RotLoopInv (nbytes : Usize) (area0 area : alloc.vec.Vec U8) (j : Usize) : Prop :=
  j.val ≤ nbytes.val ∧
  area.val = (area0.val.take j.val).map nibbleRot ++ area0.val.drop j.val

/-- One extracted nibble-swap iteration: byte `j` is rotated left by four and stored. -/
theorem rot_loop1_body_step (nbytes : Usize) (area0 area : alloc.vec.Vec U8) (j : Usize)
    (hnle : nbytes.val ≤ area0.val.length)
    (hinv : RotLoopInv nbytes area0 area j)
    (hlt : j.val < nbytes.val) :
    ∃ (area2 : alloc.vec.Vec U8) (j1 : Usize),
      packing.Packing.decode_block_fields_loop1.body nbytes area j
          = ok (ControlFlow.cont (area2, j1)) ∧
      RotLoopInv nbytes area0 area2 j1 ∧ j1.val = j.val + 1 := by
  obtain ⟨hjle, hmap⟩ := hinv
  have hja : j.val < area0.val.length := by omega
  have hbound_area : j.val < area.val.length := by
    rw [hmap]
    simp only [List.length_append, List.length_map, List.length_take, List.length_drop]
    rw [Nat.min_eq_left (le_of_lt hja)]
    omega
  obtain ⟨i, hindex, hiv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_usize_spec area j (by simpa using hbound_area))
  have hiA : i = area0.val[j.val] := by
    rw [hiv]
    simpa only [hmap] using map_take_drop_getElem area0.val nibbleRot hja (by
      simp only [List.length_append, List.length_map, List.length_take, List.length_drop]
      rw [Nat.min_eq_left (le_of_lt hja)]
      omega)
  obtain ⟨p, hmut, hpost⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.index_mut_usize_spec area j (by simpa using hbound_area))
  obtain ⟨o, back⟩ := p
  obtain ⟨ho, hback⟩ := hpost
  obtain ⟨j1, hadd, hj1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := j) (y := 1#usize) (by scalar_tac))
  have hj1' : j1.val = j.val + 1 := by rw [hj1v]; simp
  refine ⟨back (nibbleRot i), j1, ?_, ?_, hj1'⟩
  · unfold packing.Packing.decode_block_fields_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt),
      alloc.vec.Vec.index_slice_index, hindex]
    simp only [bind_tc_ok, lift, nibbleRot_eq]
    rw [alloc.vec.Vec.index_mut_slice_index, hmut]
    simp only [bind_tc_ok]
    rw [hadd]
    simp only [bind_tc_ok]
    rfl
  · refine ⟨by omega, ?_⟩
    have hbapp : back (nibbleRot i) = area.set j (nibbleRot (area0.val[j.val])) := by
      rw [hback, hiA]
    show (back (nibbleRot i)).val
        = (area0.val.take j1.val).map nibbleRot ++ area0.val.drop j1.val
    rw [hbapp, alloc.vec.Vec.set_val_eq, hmap, hj1', rot_set_eq area0.val hja]
    rw [map_take_succ area0.val nibbleRot hja]
    simp [List.append_assoc]

/-- The extracted nibble-swap loop on the success domain. -/
theorem rot_loop1_spec (nbytes : Usize) (area0 : alloc.vec.Vec U8)
    (hnle : nbytes.val ≤ area0.val.length) :
    WP.spec (packing.Packing.decode_block_fields_loop1 nbytes area0 0#usize)
      (fun area => ∃ area' : alloc.vec.Vec U8, area = area' ∧
        area'.val = (area0.val.take nbytes.val).map nibbleRot ++ area0.val.drop nbytes.val) := by
  have hinv0 : RotLoopInv nbytes area0 area0 0#usize := by
    refine ⟨by simp, by simp⟩
  unfold packing.Packing.decode_block_fields_loop1
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec U8 × Usize => nbytes.val - s.2.val)
    (inv := fun s => RotLoopInv nbytes area0 s.1 s.2)
    (post := fun (area : alloc.vec.Vec U8) => ∃ area' : alloc.vec.Vec U8, area = area' ∧
      area'.val = (area0.val.take nbytes.val).map nibbleRot ++ area0.val.drop nbytes.val)
  · intro x hx
    rcases x with ⟨area, j⟩
    dsimp only
    dsimp only at hx
    by_cases hlt : j.val < nbytes.val
    · obtain ⟨area2, j1, hbody, hinv2, hj1⟩ :=
        rot_loop1_body_step nbytes area0 area j hnle hx hlt
      rw [hbody, WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show nbytes.val - j1.val < nbytes.val - j.val
      rw [hj1]
      omega
    · have hnot : ¬ (j < nbytes) := by rw [UScalar.lt_equiv]; exact hlt
      have hbody : packing.Packing.decode_block_fields_loop1.body nbytes area j
          = ok (ControlFlow.done area) := by
        unfold packing.Packing.decode_block_fields_loop1.body
        rw [if_neg hnot]
      rw [hbody, WP.spec_ok]
      have hjeq : j.val = nbytes.val := by have := hx.1; omega
      obtain ⟨-, hmap⟩ := hx
      rw [hjeq] at hmap
      exact ⟨area, rfl, hmap⟩
  · exact hinv0

/-- `decode_block_fields_loop6` is the rotate loop. -/
theorem rot_loop6_eq (nbytes : Usize) (area : alloc.vec.Vec U8) (j : Usize) :
    packing.Packing.decode_block_fields_loop6 nbytes area j
      = packing.Packing.decode_block_fields_loop1 nbytes area j := rfl

/-- The extracted `decode_block_fields_loop6` on the success domain. -/
theorem rot_loop6_spec (nbytes : Usize) (area0 : alloc.vec.Vec U8)
    (hnle : nbytes.val ≤ area0.val.length) :
    WP.spec (packing.Packing.decode_block_fields_loop6 nbytes area0 0#usize)
      (fun area => ∃ area' : alloc.vec.Vec U8, area = area' ∧
        area'.val = (area0.val.take nbytes.val).map nibbleRot ++ area0.val.drop nbytes.val) := by
  have h := rot_loop1_spec nbytes area0 hnle
  rw [rot_loop6_eq]
  exact h

/-! ## The coefficient loop `decode_block_fields_loop4` -/

/-- One coefficient byte as an `i32`: signed `i8` (balanced mod 256) or unsigned. -/
def readCoeff (sco : Bool) (x : U8) : ℤ :=
  if sco then Int.bmod (x.val : ℤ) 256 else (x.val : ℤ)

/-- The integer values of a coefficient vector. -/
def coeffVals (coeffs : alloc.vec.Vec (I32 × I32)) : List (ℤ × ℤ) :=
  coeffs.val.map (fun p => (p.1.val, p.2.val))

/-- The byte stride of one coefficient group. -/
def coeffStep (b : Bool) : Nat := if b then 2 else 1

/-- The reference `(a, b)` pair of group `k`, read at `at0 + k·step`. -/
def coeffRef (b b1 : Bool) (raw : Slice U8) (at0 : Nat) (k : Nat) : ℤ × ℤ :=
  (readCoeff b1 (raw.val.getD (at0 + k * coeffStep b) 0#u8),
   if b then readCoeff b1 (raw.val.getD (at0 + k * coeffStep b + 1) 0#u8) else 0)

lemma readSigned_val (x : U8) :
    (IScalar.cast IScalarTy.I32 (UScalar.hcast IScalarTy.I8 x)).val = readCoeff true x := by
  rw [IScalar.val_mod_pow_greater_numBits IScalarTy.I32 _ (by norm_num)]
  simp [readCoeff, UScalar.hcast_val_eq, IScalarTy.I8_numBits_eq]

lemma readUnsigned_val (x : U8) :
    (UScalar.hcast IScalarTy.I32 x).val = readCoeff false x := by
  simp only [readCoeff]
  rw [UScalar.hcast_val_eq, IScalarTy.I32_numBits_eq]
  apply Int.bmod_eq_of_le
  · norm_num
  · have := U8.lt_succ_max x
    norm_num
    omega

/-- The signed-coefficient read block returns the reference value. -/
lemma readBlock_signed_ok (raw : Slice U8) (pos : Usize) (h : pos.val < raw.val.length) :
    ∃ x : I32,
      (do let i ← Slice.index_usize raw pos
          let i1 ← lift (UScalar.hcast IScalarTy.I8 i)
          ok (IScalar.cast IScalarTy.I32 i1)) = ok x ∧
      x.val = readCoeff true (raw.val.getD pos.val 0#u8) := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  have hiD : i = raw.val.getD pos.val 0#u8 := by
    rw [hiv]; exact (getD_eq_getElem raw.val pos.val 0#u8 h).symm
  refine ⟨IScalar.cast IScalarTy.I32 (UScalar.hcast IScalarTy.I8 i), ?_, ?_⟩
  · rw [hi]; simp only [bind_tc_ok, lift]
  · rw [readSigned_val, hiD]

/-- The unsigned-coefficient read block returns the reference value. -/
lemma readBlock_unsigned_ok (raw : Slice U8) (pos : Usize) (h : pos.val < raw.val.length) :
    ∃ x : I32,
      (do let i ← Slice.index_usize raw pos
          ok (UScalar.hcast IScalarTy.I32 i)) = ok x ∧
      x.val = readCoeff false (raw.val.getD pos.val 0#u8) := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  have hiD : i = raw.val.getD pos.val 0#u8 := by
    rw [hiv]; exact (getD_eq_getElem raw.val pos.val 0#u8 h).symm
  refine ⟨UScalar.hcast IScalarTy.I32 i, ?_, ?_⟩
  · rw [hi]; simp only [bind_tc_ok]
  · rw [readUnsigned_val, hiD]

/-- The inlined signed-coefficient read block (`let i1 ← lift` folded). -/
lemma readBlock_signed_ok_inl (raw : Slice U8) (pos : Usize) (h : pos.val < raw.val.length) :
    ∃ x : I32,
      (do let i ← Slice.index_usize raw pos
          ok (IScalar.cast IScalarTy.I32 (UScalar.hcast IScalarTy.I8 i))) = ok x ∧
      x.val = readCoeff true (raw.val.getD pos.val 0#u8) := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  have hiD : i = raw.val.getD pos.val 0#u8 := by
    rw [hiv]; exact (getD_eq_getElem raw.val pos.val 0#u8 h).symm
  refine ⟨IScalar.cast IScalarTy.I32 (UScalar.hcast IScalarTy.I8 i), ?_, ?_⟩
  · rw [hi]; simp only [bind_tc_ok]
  · rw [readSigned_val, hiD]

/-- The inlined unsigned-coefficient read block (`let i1 ← lift` folded). -/
lemma readBlock_unsigned_ok_inl (raw : Slice U8) (pos : Usize) (h : pos.val < raw.val.length) :
    ∃ x : I32,
      (do let i ← Slice.index_usize raw pos
          ok (UScalar.hcast IScalarTy.I32 i)) = ok x ∧
      x.val = readCoeff false (raw.val.getD pos.val 0#u8) := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Slice.index_usize_spec raw pos h)
  have hiD : i = raw.val.getD pos.val 0#u8 := by
    rw [hiv]; exact (getD_eq_getElem raw.val pos.val 0#u8 h).symm
  refine ⟨UScalar.hcast IScalarTy.I32 i, ?_, ?_⟩
  · rw [hi]; simp only [bind_tc_ok]
  · rw [readUnsigned_val, hiD]

/-- The coefficient-loop invariant: `g` groups appended, the strand advanced, `signed_coeff`
unchanged. -/
def CoeffLoopInv (b b1 : Bool) (raw : Slice U8) (at0 : Usize) (groups : Usize)
    (coeffs0 coeffs : alloc.vec.Vec (I32 × I32)) (b2 : Bool) (pos : Usize) (g : Usize) : Prop :=
  g.val ≤ groups.val ∧ b2 = b1 ∧
  pos.val = at0.val + g.val * coeffStep b ∧
  coeffVals coeffs = coeffVals coeffs0 ++ (List.range g.val).map (coeffRef b b1 raw at0.val)

/-- One extracted coefficient-group iteration. -/
theorem coeff_loop4_body_step (b b1 : Bool) (raw : Slice U8) (groups : Usize)
    (at0 : Usize) (coeffs0 coeffs : alloc.vec.Vec (I32 × I32)) (b2 : Bool) (pos : Usize)
    (g : Usize)
    (hinv : CoeffLoopInv b b1 raw at0 groups coeffs0 coeffs b2 pos g)
    (hlt : g.val < groups.val)
    (hA : pos.val < raw.val.length)
    (hB : b = true → pos.val + 1 < raw.val.length)
    (hcap : coeffs0.val.length + groups.val ≤ Usize.max) :
    ∃ (b2' : Bool) (pos' : Usize) (coeffs' : alloc.vec.Vec (I32 × I32)) (g' : Usize),
      packing.Packing.decode_block_fields_loop4.body b raw groups b2 pos coeffs g
          = ok (ControlFlow.cont (b2', pos', coeffs', g')) ∧
      CoeffLoopInv b b1 raw at0 groups coeffs0 coeffs' b2' pos' g' ∧ g'.val = g.val + 1 := by
  obtain ⟨hg, hb2, hpos, hmap⟩ := hinv
  rw [hb2]
  have hlen : coeffs.val.length = coeffs0.val.length + g.val := by
    have := congrArg List.length hmap
    simpa [coeffVals] using this
  have hclen : coeffs.val.length < Usize.max := by
    rw [hlen]
    calc coeffs0.val.length + g.val < coeffs0.val.length + groups.val := Nat.add_lt_add_left hlt _
      _ ≤ Usize.max := hcap
  obtain ⟨at1, hat1, hat1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := pos) (y := 1#usize) (by scalar_tac))
  have hat1' : at1.val = pos.val + 1 := by rw [hat1v]; simp
  have hsp : (List.range (g.val + 1)).map (coeffRef b b1 raw at0.val)
      = (List.range g.val).map (coeffRef b b1 raw at0.val)
        ++ [coeffRef b b1 raw at0.val g.val] := by
    rw [List.range_succ, List.map_append]; simp
  by_cases hb1 : b1 = true
  · by_cases hb : b = true
    · -- signed, with min
      have hA1 : at1.val < raw.val.length := by rw [hat1']; exact hB hb
      obtain ⟨xa, hxa, hxav⟩ := readBlock_signed_ok raw pos hA
      obtain ⟨xb, hxb, hxbv⟩ := readBlock_signed_ok_inl raw at1 hA1
      obtain ⟨at3, hat3, hat3v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := at1) (y := 1#usize) (by scalar_tac))
      have hat3' : at3.val = at1.val + 1 := by rw [hat3v]; simp
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, xb) hclen)
      obtain ⟨g1, hgadd, hg1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := g) (y := 1#usize) (by scalar_tac))
      have hg1' : g1.val = g.val + 1 := by rw [hg1v]; simp
      have hpos2 : pos.val = at0.val + g.val * 2 := by rw [hpos, coeffStep, if_pos hb]
      have hat1'' : at1.val = at0.val + g.val * 2 + 1 := by rw [hat1', hpos2]
      have href : coeffRef b b1 raw at0.val g.val
          = (readCoeff true (raw.val.getD pos.val 0#u8),
             readCoeff true (raw.val.getD at1.val 0#u8)) := by
        unfold coeffRef
        simp only [coeffStep, if_pos hb, hb1, hpos2, hat1'']
      refine ⟨b1, at3, coeffs2, g1, ?_, ?_, hg1'⟩
      · unfold packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_pos hb1, hxa]
        simp only [bind_tc_ok, lift]
        rw [hat1]; simp only [bind_tc_ok]
        rw [if_pos hb, if_pos hb1, hxb]
        simp only [bind_tc_ok]
        rw [hat3]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, xb)
                   let g1 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at3, coeffs1, g1)))
            = ok (ControlFlow.cont (b1, at3, coeffs2, g1))
        rw [hpush, hgadd]
        simp only [bind_tc_ok]
      · refine ⟨by omega, rfl, ?_, ?_⟩
        · rw [hat3', hat1', hpos, coeffStep, if_pos hb]; omega
        · simp only [coeffVals] at hmap ⊢
          rw [hpushv, List.map_append, hmap, hg1', hsp, List.append_assoc]
          simp only [List.map_cons, List.map_nil]
          rw [href]
          simp [hxav, hxbv]
    · -- signed, no min
      obtain ⟨xa, hxa, hxav⟩ := readBlock_signed_ok raw pos hA
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, 0#i32) hclen)
      obtain ⟨g1, hgadd, hg1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := g) (y := 1#usize) (by scalar_tac))
      have hg1' : g1.val = g.val + 1 := by rw [hg1v]; simp
      have hpos2 : pos.val = at0.val + g.val := by rw [hpos, coeffStep, if_neg hb, Nat.mul_one]
      have href : coeffRef b b1 raw at0.val g.val
          = (readCoeff true (raw.val.getD pos.val 0#u8), 0) := by
        unfold coeffRef
        simp only [coeffStep, if_neg hb, Nat.mul_one, hb1, hpos2]
      refine ⟨b1, at1, coeffs2, g1, ?_, ?_, hg1'⟩
      · unfold packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_pos hb1, hxa]
        simp only [bind_tc_ok, lift]
        rw [hat1]; simp only [bind_tc_ok]
        rw [if_neg hb]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, 0#i32)
                   let g1 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at1, coeffs1, g1)))
            = ok (ControlFlow.cont (b1, at1, coeffs2, g1))
        rw [hpush, hgadd]
        simp only [bind_tc_ok]
      · refine ⟨by omega, rfl, ?_, ?_⟩
        · rw [hat1', hpos, coeffStep, if_neg hb]; omega
        · simp only [coeffVals] at hmap ⊢
          rw [hpushv, List.map_append, hmap, hg1', hsp, List.append_assoc]
          simp only [List.map_cons, List.map_nil]
          rw [href]
          simp [hxav]
  · by_cases hb : b = true
    · -- unsigned, with min
      have hA1 : at1.val < raw.val.length := by rw [hat1']; exact hB hb
      obtain ⟨xa, hxa, hxav⟩ := readBlock_unsigned_ok raw pos hA
      obtain ⟨xb, hxb, hxbv⟩ := readBlock_unsigned_ok_inl raw at1 hA1
      obtain ⟨at3, hat3, hat3v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := at1) (y := 1#usize) (by scalar_tac))
      have hat3' : at3.val = at1.val + 1 := by rw [hat3v]; simp
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, xb) hclen)
      obtain ⟨g1, hgadd, hg1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := g) (y := 1#usize) (by scalar_tac))
      have hg1' : g1.val = g.val + 1 := by rw [hg1v]; simp
      have hpos2 : pos.val = at0.val + g.val * 2 := by rw [hpos, coeffStep, if_pos hb]
      have hat1'' : at1.val = at0.val + g.val * 2 + 1 := by rw [hat1', hpos2]
      have href : coeffRef b b1 raw at0.val g.val
          = (readCoeff false (raw.val.getD pos.val 0#u8),
             readCoeff false (raw.val.getD at1.val 0#u8)) := by
        unfold coeffRef
        simp only [coeffStep, if_pos hb, hb1, hpos2, hat1'']
      refine ⟨b1, at3, coeffs2, g1, ?_, ?_, hg1'⟩
      · unfold packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_neg hb1, hxa]
        simp only [bind_tc_ok]
        rw [hat1]; simp only [bind_tc_ok]
        rw [if_pos hb, if_neg hb1, hxb]
        simp only [bind_tc_ok]
        rw [hat3]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, xb)
                   let g1 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at3, coeffs1, g1)))
            = ok (ControlFlow.cont (b1, at3, coeffs2, g1))
        rw [hpush, hgadd]
        simp only [bind_tc_ok]
      · refine ⟨by omega, rfl, ?_, ?_⟩
        · rw [hat3', hat1', hpos, coeffStep, if_pos hb]; omega
        · simp only [coeffVals] at hmap ⊢
          rw [hpushv, List.map_append, hmap, hg1', hsp, List.append_assoc]
          simp only [List.map_cons, List.map_nil]
          rw [href]
          simp [hxav, hxbv]
    · -- unsigned, no min
      obtain ⟨xa, hxa, hxav⟩ := readBlock_unsigned_ok raw pos hA
      obtain ⟨coeffs2, hpush, hpushv⟩ := WP.spec_imp_exists
        (alloc.vec.Vec.push_spec coeffs (xa, 0#i32) hclen)
      obtain ⟨g1, hgadd, hg1v⟩ := WP.spec_imp_exists
        (Usize.add_spec (x := g) (y := 1#usize) (by scalar_tac))
      have hg1' : g1.val = g.val + 1 := by rw [hg1v]; simp
      have hpos2 : pos.val = at0.val + g.val := by rw [hpos, coeffStep, if_neg hb, Nat.mul_one]
      have href : coeffRef b b1 raw at0.val g.val
          = (readCoeff false (raw.val.getD pos.val 0#u8), 0) := by
        unfold coeffRef
        simp only [coeffStep, if_neg hb, Nat.mul_one, hb1, hpos2]
      refine ⟨b1, at1, coeffs2, g1, ?_, ?_, hg1'⟩
      · unfold packing.Packing.decode_block_fields_loop4.body
        rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt), if_neg hb1, hxa]
        simp only [bind_tc_ok]
        rw [hat1]; simp only [bind_tc_ok]
        rw [if_neg hb]
        simp only [bind_tc_ok]
        change (do let coeffs1 ← coeffs.push (xa, 0#i32)
                   let g1 ← g + 1#usize
                   ok (ControlFlow.cont (b1, at1, coeffs1, g1)))
            = ok (ControlFlow.cont (b1, at1, coeffs2, g1))
        rw [hpush, hgadd]
        simp only [bind_tc_ok]
      · refine ⟨by omega, rfl, ?_, ?_⟩
        · rw [hat1', hpos, coeffStep, if_neg hb]; omega
        · simp only [coeffVals] at hmap ⊢
          rw [hpushv, List.map_append, hmap, hg1', hsp, List.append_assoc]
          simp only [List.map_cons, List.map_nil]
          rw [href]
          simp [hxav]

/-- The extracted coefficient loop on the success domain. -/
theorem coeff_loop4_spec (b b1 : Bool) (raw : Slice U8) (at0 : Usize) (groups : Usize)
    (coeffs0 : alloc.vec.Vec (I32 × I32))
    (hbound : ∀ k, k < groups.val →
      at0.val + k * coeffStep b + (if b then 1 else 0) < raw.val.length)
    (hcap : coeffs0.val.length + groups.val ≤ Usize.max) :
    WP.spec (packing.Packing.decode_block_fields_loop4 b b1 raw at0 coeffs0 groups 0#usize)
      (fun r => ∃ (b2' : Bool) (pos' : Usize) (coeffs' : alloc.vec.Vec (I32 × I32)),
        r = (b2', pos', coeffs') ∧ b2' = b1 ∧
        coeffVals coeffs' = coeffVals coeffs0 ++
          (List.range groups.val).map (coeffRef b b1 raw at0.val) ∧
        pos'.val = at0.val + groups.val * coeffStep b) := by
  have hinv0 : CoeffLoopInv b b1 raw at0 groups coeffs0 coeffs0 b1 at0 0#usize := by
    refine ⟨Nat.zero_le _, rfl, by simp, by simp⟩
  unfold packing.Packing.decode_block_fields_loop4
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Usize × alloc.vec.Vec (I32 × I32) × Usize =>
      groups.val - s.2.2.2.val)
    (inv := fun s => CoeffLoopInv b b1 raw at0 groups coeffs0 s.2.2.1 s.1 s.2.1 s.2.2.2)
    (post := fun (r : Bool × Usize × alloc.vec.Vec (I32 × I32)) =>
      ∃ (b2' : Bool) (pos' : Usize) (coeffs' : alloc.vec.Vec (I32 × I32)),
        r = (b2', pos', coeffs') ∧ b2' = b1 ∧
        coeffVals coeffs' = coeffVals coeffs0 ++
          (List.range groups.val).map (coeffRef b b1 raw at0.val) ∧
        pos'.val = at0.val + groups.val * coeffStep b)
  · intro x hx
    obtain ⟨b2, pos, coeffs, g⟩ := x
    dsimp only at hx ⊢
    by_cases hlt : g.val < groups.val
    · have hpos : pos.val = at0.val + g.val * coeffStep b := hx.2.2.1
      have hA : pos.val < raw.val.length := by
        by_cases hb : b = true
        · have h := hbound g.val hlt
          simp only [coeffStep, if_pos hb] at h
          rw [hpos]; simp only [coeffStep, if_pos hb]
          omega
        · have h := hbound g.val hlt
          simp only [coeffStep, if_neg hb] at h
          rw [hpos]; simp only [coeffStep, if_neg hb, Nat.mul_one]
          omega
      have hB : b = true → pos.val + 1 < raw.val.length := by
        intro hb
        have h := hbound g.val hlt
        simp only [coeffStep, if_pos hb] at h
        rw [hpos]; simp only [coeffStep, if_pos hb]
        omega
      obtain ⟨b2', pos', coeffs', g', hbody, hinv2, hg'⟩ :=
        coeff_loop4_body_step b b1 raw groups at0 coeffs0 coeffs b2 pos g hx hlt hA hB hcap
      rw [hbody, WP.spec_ok]
      refine ⟨hinv2, ?_⟩
      show groups.val - g'.val < groups.val - g.val
      rw [hg']
      omega
    · have hnot : ¬ (g < groups) := by rw [UScalar.lt_equiv]; exact hlt
      have hbody : packing.Packing.decode_block_fields_loop4.body b raw groups b2 pos coeffs g
          = ok (ControlFlow.done (b2, pos, coeffs)) := by
        unfold packing.Packing.decode_block_fields_loop4.body
        rw [if_neg hnot]
      rw [hbody, WP.spec_ok]
      have hgeq : g.val = groups.val := by have := hx.1; omega
      refine ⟨b2, pos, coeffs, rfl, hx.2.1, ?_, ?_⟩
      · obtain ⟨-, -, -, hmap⟩ := hx
        rw [hgeq] at hmap
        exact hmap
      · rw [hx.2.2.1, hgeq]
  · exact hinv0

/-! ## The top-level non-terminal assembly

`decode_block_fields_spec` composes the nine loop specs across the extracted branch structure for the
non-terminal face (`is_terminal self = ok false`). Every sub-call result and its value law is the
conclusion of the corresponding loop spec above (`copy_loop5_spec`, `rot_loop6_spec`,
`coeff_loop4_spec`, `codes_loop7_spec`/`codes_loop8_spec`), so no mathematics is assumed: the
remaining work is the `match`/`if` assembly around them. The `Terminal` (`F16`/`F32`) branch is not
covered. -/
theorem decode_block_fields_spec
    (self : packing.Packing) (raw : Slice U8)
    (bb : Usize) (hbb : packing.Packing.block_bytes self = ok (core.result.Result.Ok bb))
    (hshort : ¬ (Slice.len raw < bb))
    (sfh : Usize) (hsf : packing.Packing.scale_field_bytes self = ok sfh)
    (ht : packing.Packing.is_terminal self = ok false)
    (scaleh : U16) (hscale : packing.read_field raw 0#usize sfh = ok scaleh)
    (atq : Usize) (hatq : 0#usize + sfh = ok atq)
    (bm : Bool) (atm : Usize) (minm : U16)
    (hminbr : (if self.has_min then
        (do let min1 ← packing.read_field raw atq sfh
            let at2h ← atq + sfh
            ok (true, at2h, min1))
      else ok (false, atq, 0#u16)) = ok (bm, atm, minm))
    (val1h : Usize) (hgroups : packing.Packing.groups { self with has_min := bm } =
      ok (core.result.Result.Ok val1h))
    (b2h : Bool) (at2h : Usize) (coeffs1h : alloc.vec.Vec (I32 × I32))
    (hloop4 : packing.Packing.decode_block_fields_loop4 bm self.signed_coeff raw atm
        (alloc.vec.Vec.with_capacity (I32 × I32) val1h) val1h 0#usize = ok (b2h, at2h, coeffs1h))
    (hcoeffs : coeffVals coeffs1h =
      coeffVals (alloc.vec.Vec.with_capacity (I32 × I32) val1h) ++
        (List.range val1h.val).map (coeffRef bm self.signed_coeff raw atm.val))
    (hcoeffsAt : at2h.val = atm.val + val1h.val * coeffStep bm)
    (val2h : Usize) (htotal : packing.Packing.total_code_bits
        { self with has_min := bm, signed_coeff := b2h } = ok (core.result.Result.Ok val2h))
    (val3h : Usize) (hblock : packing.Packing.block_size
        { self with has_min := bm, signed_coeff := b2h } = ok (core.result.Result.Ok val3h))
    (bitsh : Usize) (hbits : val3h * val2h = ok bitsh)
    (nb1h : Usize) (hdiv : bitsh / 8#usize = ok nb1h)
    (nb2h : Usize) (hmod : bitsh % 8#usize = ok nb2h)
    (nb3h : Usize) (hcb : lift (UScalar.cast_fromBool UScalarTy.Usize (nb2h != 0#usize)) = ok nb3h)
    (nbytesh : Usize) (hnbytes : nb1h + nb3h = ok nbytesh)
    (area1h : alloc.vec.Vec U8)
    (hloop5 : packing.Packing.decode_block_fields_loop5 raw at2h nbytesh
        (alloc.vec.Vec.with_capacity U8 nbytesh) 0#usize = ok area1h)
    (harea1 : area1h.val =
      (List.range nbytesh.val).map (fun k => raw.val.getD (at2h.val + k) 0#u8))
    (b3h : Bool) (hnib : packing.NibbleOrder.Insts.CoreCmpPartialEqNibbleOrder.eq
        self.nibble_order packing.NibbleOrder.HighLow = ok b3h)
    (area2h : alloc.vec.Vec U8)
    (hloop6 : b3h = true → packing.Packing.decode_block_fields_loop6 nbytesh area1h 0#usize = ok area2h)
    (harea2 : b3h = true → area2h.val = (area1h.val.take nbytesh.val).map nibbleRot ++
      area1h.val.drop nbytesh.val)
    (codesH codesL : alloc.vec.Vec I64)
    (hloop7 : packing.Packing.decode_block_fields_loop7 self.width self.high_planes self.block
        self.group self.scale_kind bm self.signed_code self.bias b2h self.plane_order
        self.nibble_order val2h val3h area2h (alloc.vec.Vec.with_capacity I64 val3h) 0#usize = ok codesH)
    (hloop8 : packing.Packing.decode_block_fields_loop8 self.width self.high_planes self.block
        self.group self.scale_kind bm self.signed_code self.bias b2h self.plane_order
        self.nibble_order val2h val3h area1h (alloc.vec.Vec.with_capacity I64 val3h) 0#usize = ok codesL)
    (hcodesH : codesH.val = (List.range val3h.val).map (fun k => unpackRefNat
        (codePacking self.width self.high_planes self.block self.group self.scale_kind bm
          self.signed_code self.bias b2h self.plane_order self.nibble_order)
        (alloc.vec.Vec.deref area2h) k val2h.val))
    (hcodesL : codesL.val = (List.range val3h.val).map (fun k => unpackRefNat
        (codePacking self.width self.high_planes self.block self.group self.scale_kind bm
          self.signed_code self.bias b2h self.plane_order self.nibble_order)
        (alloc.vec.Vec.deref area1h) k val2h.val)) :
    packing.Packing.decode_block_fields self raw =
      ok (core.result.Result.Ok
        (if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
         else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })) := by
  have _ := hcoeffs
  have _ := hcoeffsAt
  have _ := harea1
  have _ := harea2
  have _ := hcodesH
  have _ := hcodesL
  unfold packing.Packing.decode_block_fields
  rw [hbb]
  simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
  rw [if_neg hshort]
  rw [hsf]
  simp only [bind_tc_ok]
  rw [ht]
  simp only [bind_tc_ok]
  rw [if_neg (by decide : ¬ ((false : Bool) = true))]
  rw [hscale]
  simp only [bind_tc_ok]
  rw [hatq]
  simp only [bind_tc_ok]
  simp [hminbr, bind_tc_ok]
  rw [hgroups]
  simp only [bind_tc_ok]
  rw [hloop4]
  simp [bind_tc_ok]
  rw [htotal]
  simp only [bind_tc_ok]
  rw [hblock]
  simp only [bind_tc_ok]
  rw [hbits]; simp only [bind_tc_ok]
  rw [hdiv]; simp only [bind_tc_ok]
  rw [hmod]; simp only [bind_tc_ok]
  rw [hcb]; simp only [bind_tc_ok]
  rw [hnbytes]; simp only [bind_tc_ok]
  rw [hloop5]
  simp only [bind_tc_ok]
  rw [hnib]
  simp only [bind_tc_ok]
  by_cases hb3 : b3h = true
  · rw [if_pos hb3]
    rw [hloop6 hb3]
    simp only [bind_tc_ok]
    rw [hloop7]
    simp only [bind_tc_ok]
    rw [if_pos hb3]
  · rw [if_neg hb3]
    rw [hloop8]
    simp only [bind_tc_ok]
    rw [if_neg hb3]

/-! ## Residual: `decode_block_spec`

`decode_block_fields_spec` closes the non-terminal field reader. The remaining obligation is
`decode_block_spec`: `packing_decode_block_eq_epilogue` (`Numeric/PackingRefinement.lean`) factors
`packing.Packing.decode_block self raw` into `decode_block_fields` followed by `epilogue`, so on the
`decode_block_fields_spec` domain it reduces to the `epilogue` law. That top-level assembly
(`block_size`/`codes.length`/`coeffs.length` checks, the `scale_to_fixed` success and the
`has_min`/width dispatch around `epilogue_loop1`/`epilogue_loop2`) is not proved in this module; its
`Fixed`-width leaves are `epilogue_loop2_fixed_spec`/`epilogue_loop1_fixed_spec`
(`Numeric/PackingEpilogueFold.lean`). The `Terminal` (`F16`/`F32`) `epilogue_loop0` and the
non-`Fixed` widths remain outside. -/

#print axioms apeiron_word.unpackRef_eq_nat
#print axioms apeiron_word.codes_loop2_spec
#print axioms apeiron_word.copy_loop5_spec
#print axioms apeiron_word.copy_loop0_spec
#print axioms apeiron_word.rot_loop1_spec
#print axioms apeiron_word.rot_loop6_spec
#print axioms apeiron_word.coeff_loop4_spec
#print axioms apeiron_word.codes_loop3_spec
#print axioms apeiron_word.codes_loop7_spec
#print axioms apeiron_word.codes_loop8_spec
#print axioms apeiron_word.decode_block_fields_spec

end apeiron_word

