/-
Refinement of the extracted canonical packing decoder's top-level assembly
.

`Numeric/PackingDecodeFields.lean` closes the non-terminal byte-level field reader
(`decode_block_fields_spec`); `Numeric/PackingEpilogueFold.lean` closes the two `Fixed`-width
epilogue folds (`epilogue_loop2_fixed_spec`, `epilogue_loop1_fixed_spec`). The residual named in
both files is the top-level `epilogue` assembly (`block_size`/length checks, the terminal dispatch,
`scale_to_fixed` and the `has_min` split) and its composition with the field reader through
`packing_decode_block_eq_epilogue`. This module discharges the `Fixed`-width part of that residual.

Proved here:

* `decodeRef`: the single affine reference of a decoded block, `epilogueMinRef` on the `has_min`
  face and `epilogueNominRef` otherwise, matching the `has_min` split of the extracted `epilogue`;
* `epilogue_fixed_nomin_spec` / `epilogue_fixed_min_spec`: on the `Fixed` width, the extracted
  `packing.Packing.epilogue` returns `Ok` exactly when its two length checks pass and the
  branch's `scale_to_fixed` call succeeds, and its values are the `List.range`-map of the reference
  — the two branch laws are the fold specs above;
* `epilogue_fixed_spec`: the top-level `epilogue` on the `Fixed` width, dispatching to loop1/loop2
  by `has_min` (`decodeRef`);
* `decode_block_spec`: on the success domain (`decode_block_fields self raw = ok (Ok b)`), the
  public decoder `word_packing_decode_block self raw = ok (Ok vals)` with `vals` the reference
  values of the block the field reader unpacked — the byte-level decoder refines the canonical
  affine reading of the unpacked code.

## Domain and residual

The statement is restricted to the `Fixed` width (`self.width = CodeWidth.Fixed w`); the `Terminal`
(`F16`/`F32`) `epilogue_loop0` and the `Trit` width are not covered. The domain conditions
(`block_size`/`groups`/`scale_to_fixed` results, the two length equalities and the per-weight
headroom windows) are exactly what the checked operations need; the statements take them as
hypotheses. `decode_block_spec` is stated over an arbitrary successful
`decode_block_fields` result; instantiating it at the concrete `decode_block_fields_spec` block is a
separate step (see the residual note at the end of the file).

## Complexity

`epilogue` visits the `block` weights once through `epilogue_loop1`/`epilogue_loop2`: `O(block)`
time and output, counted in loop iterations and `i64` words. The checks are `O(1)`. The figures name
the loops.

## Trust boundary

Reaches the extracted `packing.Packing.epilogue(.body)` and the decoded `Vec` values of
`proof/ApeironWord.lean`. Does not cover rustc/LLVM codegen of `src/format/packing.rs`, the kernel
ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.PackingDecodeFields

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The `has_min` dispatch reference -/

/-- The affine reading of a decoded block: the min reference on the `has_min` face, the no-min
reference otherwise. -/
def decodeRef (hasMin : Bool) (scale bias minScale : ℤ)
    (coeffs : List (Std.I32 × Std.I32)) (codes : List Std.I64) (group j : ℕ) : ℤ :=
  if hasMin then epilogueMinRef scale bias minScale coeffs codes group j
  else epilogueNominRef scale bias coeffs codes group j

@[simp] lemma decodeRef_true (scale bias minScale : ℤ) (coeffs : List (Std.I32 × Std.I32))
    (codes : List Std.I64) (group j : ℕ) :
    decodeRef true scale bias minScale coeffs codes group j
      = epilogueMinRef scale bias minScale coeffs codes group j := rfl

@[simp] lemma decodeRef_false (scale bias minScale : ℤ) (coeffs : List (Std.I32 × Std.I32))
    (codes : List Std.I64) (group j : ℕ) :
    decodeRef false scale bias minScale coeffs codes group j
      = epilogueNominRef scale bias coeffs codes group j := rfl

/-- The `with_capacity` value law the epilogue starts its folds from. -/
@[simp] lemma with_capacity_val (T : Type) (n : Std.Usize) :
    (alloc.vec.Vec.with_capacity T n).val = ([] : List T) := by
  simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]

/-! ## The `has_min = false` epilogue -/

/-- **`epilogue` on the `Fixed`/`has_min = false` face.** On the two length equalities, the
evaluated `block_size`/`groups`/`scale_to_fixed` results and the per-weight window, the extracted
`epilogue` returns the `List.range`-map of the no-min affine reference. -/
theorem epilogue_fixed_nomin_spec
    (w : Std.U8) (self : packing.Packing) (b : packing.CanonicalBlock)
    (hw : self.width = packing.CodeWidth.Fixed w)
    (block : Std.Usize)
    (hbs : packing.Packing.block_size self = ok (core.result.Result.Ok block))
    (hcodeslen : b.codes.val.length = block.val)
    (val1 : Std.Usize)
    (hgrp : packing.Packing.groups self = ok (core.result.Result.Ok val1))
    (hcoeffslen : b.coeffs.val.length = val1.val)
    (scale : Std.I64)
    (hscale : packing.Packing.scale_to_fixed self b.scale = ok (core.result.Result.Ok scale))
    (hmin : self.has_min = false)
    (hwin : EpilogueNominWindow b.coeffs.val b.codes.val
      (UScalar.cast UScalarTy.Usize self.group) scale self.bias block.val) :
    ∃ out' : alloc.vec.Vec Std.I64,
      packing.Packing.epilogue self b = ok (core.result.Result.Ok out') ∧
      out'.val.map (fun x => x.val) =
        (List.range block.val).map (epilogueNominRef scale.val self.bias.val b.coeffs.val
          b.codes.val (UScalar.cast UScalarTy.Usize self.group).val) := by
  obtain ⟨out', hloop, hval⟩ :=
    epilogue_loop2_fixed_spec w self.high_planes self.block self.group self.scale_kind
      self.signed_code self.bias self.signed_coeff self.plane_order self.nibble_order b block
      (UScalar.cast UScalarTy.Usize self.group) scale (alloc.vec.Vec.with_capacity Std.I64 block)
      packing.PackingError.Overflow hwin (by omega)
      (by simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]; scalar_tac)
  have hterm : packing.Packing.is_terminal self = ok false := by
    simp [packing.Packing.is_terminal, hw]
  have hc1 : b.codes.len = block :=
    UScalar.val_eq_imp (b.codes.len) block (by rw [alloc.vec.Vec.len_val]; exact hcodeslen)
  have hc2 : b.coeffs.len = val1 :=
    UScalar.val_eq_imp (b.coeffs.len) val1 (by rw [alloc.vec.Vec.len_val]; exact hcoeffslen)
  refine ⟨out', ?_, ?_⟩
  · unfold packing.Packing.epilogue
    rw [hbs]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch, lift]
    rw [if_neg (by rw [hc1]; simp)]
    rw [hgrp]
    simp only [bind_tc_ok]
    rw [if_neg (by rw [hc2]; simp)]
    rw [hterm]
    simp only [bind_tc_ok]
    rw [if_neg (by decide : ¬ ((false : Bool) = true))]
    rw [hscale]
    simp only [bind_tc_ok]
    rw [if_neg (by rw [hmin]; decide)]
    rw [hw, hloop]
    simp only [bind_tc_ok]
    rfl
  · rw [hval]
    simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]

/-! ## The `has_min = true` epilogue -/

/-- **`epilogue` on the `Fixed`/`has_min = true` face.** The min mirror of
`epilogue_fixed_nomin_spec`: the evaluated `scale_to_fixed self b.min` supplies `min_scale`. -/
theorem epilogue_fixed_min_spec
    (w : Std.U8) (self : packing.Packing) (b : packing.CanonicalBlock)
    (hw : self.width = packing.CodeWidth.Fixed w)
    (block : Std.Usize)
    (hbs : packing.Packing.block_size self = ok (core.result.Result.Ok block))
    (hcodeslen : b.codes.val.length = block.val)
    (val1 : Std.Usize)
    (hgrp : packing.Packing.groups self = ok (core.result.Result.Ok val1))
    (hcoeffslen : b.coeffs.val.length = val1.val)
    (scale : Std.I64)
    (hscale : packing.Packing.scale_to_fixed self b.scale = ok (core.result.Result.Ok scale))
    (minScale : Std.I64)
    (hminScale : packing.Packing.scale_to_fixed self b.min = ok (core.result.Result.Ok minScale))
    (hmin : self.has_min = true)
    (hwin : EpilogueMinWindow b.coeffs.val b.codes.val
      (UScalar.cast UScalarTy.Usize self.group) scale minScale self.bias block.val) :
    ∃ out' : alloc.vec.Vec Std.I64,
      packing.Packing.epilogue self b = ok (core.result.Result.Ok out') ∧
      out'.val.map (fun x => x.val) =
        (List.range block.val).map (epilogueMinRef scale.val self.bias.val minScale.val b.coeffs.val
          b.codes.val (UScalar.cast UScalarTy.Usize self.group).val) := by
  obtain ⟨out', hloop, hval⟩ :=
    epilogue_loop1_fixed_spec w self.high_planes self.block self.group self.scale_kind
      self.signed_code self.bias self.signed_coeff self.plane_order self.nibble_order b block
      (UScalar.cast UScalarTy.Usize self.group) scale minScale
      (alloc.vec.Vec.with_capacity Std.I64 block) packing.PackingError.Overflow hwin (by omega)
      (by simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]; scalar_tac)
  have hterm : packing.Packing.is_terminal self = ok false := by
    simp [packing.Packing.is_terminal, hw]
  have hc1 : b.codes.len = block :=
    UScalar.val_eq_imp (b.codes.len) block (by rw [alloc.vec.Vec.len_val]; exact hcodeslen)
  have hc2 : b.coeffs.len = val1 :=
    UScalar.val_eq_imp (b.coeffs.len) val1 (by rw [alloc.vec.Vec.len_val]; exact hcoeffslen)
  refine ⟨out', ?_, ?_⟩
  · unfold packing.Packing.epilogue
    rw [hbs]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch, lift]
    rw [if_neg (by rw [hc1]; simp)]
    rw [hgrp]
    simp only [bind_tc_ok]
    rw [if_neg (by rw [hc2]; simp)]
    rw [hterm]
    simp only [bind_tc_ok]
    rw [if_neg (by decide : ¬ ((false : Bool) = true))]
    rw [hscale]
    simp only [bind_tc_ok]
    rw [if_pos (by rw [hmin])]
    rw [hminScale]
    simp only [bind_tc_ok]
    rw [hw, hloop]
    simp only [bind_tc_ok]
    rfl
  · rw [hval]
    simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]

/-! ## The top-level epilogue dispatch -/

/-- **`epilogue` on the `Fixed` width.** The extracted top-level assembly: the two length checks,
the terminal dispatch (false for the `Fixed` width), the scale reads and the `has_min` split into
`epilogue_loop1`/`epilogue_loop2`. The conclusion is the `decodeRef` reference, which selects the
face. -/
theorem epilogue_fixed_spec
    (w : Std.U8) (self : packing.Packing) (b : packing.CanonicalBlock)
    (hw : self.width = packing.CodeWidth.Fixed w)
    (block : Std.Usize)
    (hbs : packing.Packing.block_size self = ok (core.result.Result.Ok block))
    (hcodeslen : b.codes.val.length = block.val)
    (val1 : Std.Usize)
    (hgrp : packing.Packing.groups self = ok (core.result.Result.Ok val1))
    (hcoeffslen : b.coeffs.val.length = val1.val)
    (scale : Std.I64)
    (hscale : packing.Packing.scale_to_fixed self b.scale = ok (core.result.Result.Ok scale))
    (minScale : Std.I64)
    (hminScale : packing.Packing.scale_to_fixed self b.min = ok (core.result.Result.Ok minScale))
    (hwinn : EpilogueNominWindow b.coeffs.val b.codes.val
      (UScalar.cast UScalarTy.Usize self.group) scale self.bias block.val)
    (hwinm : EpilogueMinWindow b.coeffs.val b.codes.val
      (UScalar.cast UScalarTy.Usize self.group) scale minScale self.bias block.val) :
    ∃ out' : alloc.vec.Vec Std.I64,
      packing.Packing.epilogue self b = ok (core.result.Result.Ok out') ∧
      out'.val.map (fun x => x.val) =
        (List.range block.val).map (decodeRef self.has_min scale.val self.bias.val minScale.val
          b.coeffs.val b.codes.val (UScalar.cast UScalarTy.Usize self.group).val) := by
  by_cases hmin : self.has_min = true
  · obtain ⟨out', hep, hval⟩ :=
      epilogue_fixed_min_spec w self b hw block hbs hcodeslen val1 hgrp hcoeffslen scale hscale
        minScale hminScale hmin hwinm
    exact ⟨out', hep, by rw [hval]; simp [decodeRef, hmin]⟩
  · have hmin' : self.has_min = false := by cases h : self.has_min <;> simp_all
    obtain ⟨out', hep, hval⟩ :=
      epilogue_fixed_nomin_spec w self b hw block hbs hcodeslen val1 hgrp hcoeffslen scale hscale
        hmin' hwinn
    exact ⟨out', hep, by rw [hval]; simp [decodeRef, hmin']⟩

/-! ## The decoder value law -/

/-- **The decoded block's affine value law.** On the success domain (`decode_block_fields` returns
the block `b`), with the `Fixed`-width epilogue domain, the public decoder returns the reference
values `decodeRef` of the block the field reader unpacked: `O(block)` `i64` words, the affine
reading of the unpacked code and coefficients. -/
theorem decode_block_spec
    (w : Std.U8) (self : packing.Packing) (raw : Slice Std.U8) (b : packing.CanonicalBlock)
    (hw : self.width = packing.CodeWidth.Fixed w)
    (hfields : packing.Packing.decode_block_fields self raw = ok (core.result.Result.Ok b))
    (block : Std.Usize)
    (hbs : packing.Packing.block_size self = ok (core.result.Result.Ok block))
    (hcodeslen : b.codes.val.length = block.val)
    (val1 : Std.Usize)
    (hgrp : packing.Packing.groups self = ok (core.result.Result.Ok val1))
    (hcoeffslen : b.coeffs.val.length = val1.val)
    (scale : Std.I64)
    (hscale : packing.Packing.scale_to_fixed self b.scale = ok (core.result.Result.Ok scale))
    (minScale : Std.I64)
    (hminScale : packing.Packing.scale_to_fixed self b.min = ok (core.result.Result.Ok minScale))
    (hwinn : EpilogueNominWindow b.coeffs.val b.codes.val
      (UScalar.cast UScalarTy.Usize self.group) scale self.bias block.val)
    (hwinm : EpilogueMinWindow b.coeffs.val b.codes.val
      (UScalar.cast UScalarTy.Usize self.group) scale minScale self.bias block.val) :
    ∃ vals : alloc.vec.Vec Std.I64,
      word_packing_decode_block self raw = ok (core.result.Result.Ok vals) ∧
      vals.val.map (fun x => x.val) =
        (List.range block.val).map (decodeRef self.has_min scale.val self.bias.val minScale.val
          b.coeffs.val b.codes.val (UScalar.cast UScalarTy.Usize self.group).val) := by
  obtain ⟨out', hep, hval⟩ :=
    epilogue_fixed_spec w self b hw block hbs hcodeslen val1 hgrp hcoeffslen scale hscale minScale
      hminScale hwinn hwinm
  exact ⟨out', by rw [packing_decode_block_eq_epilogue self raw b hfields]; exact hep, hval⟩

/-! ## The composition with the field reader

`decode_block_spec` factors the decoder through an arbitrary successful
`decode_block_fields`. The two transport lemmas below let the concrete block `decode_block_fields`
returns from `decode_block_fields_spec` feed the same statement: `block_size` reads only the width
and `block`, and `groups` only `block_size` and `group`. -/

/-- `block_size` is unchanged by the `has_min`/`signed_coeff` update the field reader's block
carries. -/
lemma block_size_update_hasmin_sco (self : packing.Packing) (hm sc : Bool) :
    packing.Packing.block_size { self with has_min := hm, signed_coeff := sc }
      = packing.Packing.block_size self := rfl

/-- `groups` is unchanged by the `has_min` update. -/
lemma groups_update_hasmin (self : packing.Packing) (hm : Bool) :
    packing.Packing.groups { self with has_min := hm } = packing.Packing.groups self := rfl

/-- **The decoder value law on the byte-level domain.** The composition of `decode_block_spec` with
`decode_block_fields_spec`: the block `decode_block_fields` unpacks (its coefficient list and its
`codesH`/`codesL` code list, each the `unpackRefNat` map of the nibble/bit-plane area) is the block
whose affine reading `decodeRef` the decoder returns. `hbm : bm = self.has_min` records the flag the
reader's own `if` threaded (the reader's both arms carry `self.has_min` through unchanged); the
reference then dispatches on the same flag. The two windows are the per-weight checked-`i64`
headroom of the chosen epilogue face. -/
theorem decode_block_spec_fields
    (self : packing.Packing) (raw : Slice Std.U8) (w : Std.U8)
    (hw : self.width = packing.CodeWidth.Fixed w)
    (bb : Std.Usize) (hbb : packing.Packing.block_bytes self = ok (core.result.Result.Ok bb))
    (hshort : ¬ (Slice.len raw < bb))
    (sfh : Std.Usize) (hsf : packing.Packing.scale_field_bytes self = ok sfh)
    (ht : packing.Packing.is_terminal self = ok false)
    (scaleh : Std.U16) (hscale : packing.read_field raw 0#usize sfh = ok scaleh)
    (atq : Std.Usize) (hatq : 0#usize + sfh = ok atq)
    (bm : Bool) (atm : Std.Usize) (minm : Std.U16)
    (hminbr : (if self.has_min then
        (do let min1 ← packing.read_field raw atq sfh
            let at2h ← atq + sfh
            ok (true, at2h, min1))
      else ok (false, atq, 0#u16)) = ok (bm, atm, minm))
    (hbm : bm = self.has_min)
    (val1h : Std.Usize) (hgroups : packing.Packing.groups { self with has_min := bm } =
      ok (core.result.Result.Ok val1h))
    (b2h : Bool) (at2h : Std.Usize) (coeffs1h : alloc.vec.Vec (Std.I32 × Std.I32))
    (hloop4 : packing.Packing.decode_block_fields_loop4 bm self.signed_coeff raw atm
        (alloc.vec.Vec.with_capacity (Std.I32 × Std.I32) val1h) val1h 0#usize
        = ok (b2h, at2h, coeffs1h))
    (hcoeffs : coeffVals coeffs1h =
      coeffVals (alloc.vec.Vec.with_capacity (Std.I32 × Std.I32) val1h) ++
        (List.range val1h.val).map (coeffRef bm self.signed_coeff raw atm.val))
    (hcoeffsAt : at2h.val = atm.val + val1h.val * coeffStep bm)
    (val2h : Std.Usize) (htotal : packing.Packing.total_code_bits
        { self with has_min := bm, signed_coeff := b2h } = ok (core.result.Result.Ok val2h))
    (val3h : Std.Usize) (hblock : packing.Packing.block_size
        { self with has_min := bm, signed_coeff := b2h } = ok (core.result.Result.Ok val3h))
    (bitsh : Std.Usize) (hbits : val3h * val2h = ok bitsh)
    (nb1h : Std.Usize) (hdiv : bitsh / 8#usize = ok nb1h)
    (nb2h : Std.Usize) (hmod : bitsh % 8#usize = ok nb2h)
    (nb3h : Std.Usize)
    (hcb : lift (UScalar.cast_fromBool UScalarTy.Usize (nb2h != 0#usize)) = ok nb3h)
    (nbytesh : Std.Usize) (hnbytes : nb1h + nb3h = ok nbytesh)
    (area1h : alloc.vec.Vec Std.U8)
    (hloop5 : packing.Packing.decode_block_fields_loop5 raw at2h nbytesh
        (alloc.vec.Vec.with_capacity Std.U8 nbytesh) 0#usize = ok area1h)
    (harea1 : area1h.val =
      (List.range nbytesh.val).map (fun k => raw.val.getD (at2h.val + k) 0#u8))
    (b3h : Bool) (hnib : packing.NibbleOrder.Insts.CoreCmpPartialEqNibbleOrder.eq
        self.nibble_order packing.NibbleOrder.HighLow = ok b3h)
    (area2h : alloc.vec.Vec Std.U8)
    (hloop6 : b3h = true →
      packing.Packing.decode_block_fields_loop6 nbytesh area1h 0#usize = ok area2h)
    (harea2 : b3h = true → area2h.val = (area1h.val.take nbytesh.val).map nibbleRot ++
      area1h.val.drop nbytesh.val)
    (codesH codesL : alloc.vec.Vec Std.I64)
    (hloop7 : packing.Packing.decode_block_fields_loop7 self.width self.high_planes self.block
        self.group self.scale_kind bm self.signed_code self.bias b2h self.plane_order
        self.nibble_order val2h val3h area2h (alloc.vec.Vec.with_capacity Std.I64 val3h) 0#usize
        = ok codesH)
    (hloop8 : packing.Packing.decode_block_fields_loop8 self.width self.high_planes self.block
        self.group self.scale_kind bm self.signed_code self.bias b2h self.plane_order
        self.nibble_order val2h val3h area1h (alloc.vec.Vec.with_capacity Std.I64 val3h) 0#usize
        = ok codesL)
    (hcodesH : codesH.val = (List.range val3h.val).map (fun k => unpackRefNat
        (codePacking self.width self.high_planes self.block self.group self.scale_kind bm
          self.signed_code self.bias b2h self.plane_order self.nibble_order)
        (alloc.vec.Vec.deref area2h) k val2h.val))
    (hcodesL : codesL.val = (List.range val3h.val).map (fun k => unpackRefNat
        (codePacking self.width self.high_planes self.block self.group self.scale_kind bm
          self.signed_code self.bias b2h self.plane_order self.nibble_order)
        (alloc.vec.Vec.deref area1h) k val2h.val))
    (scale : Std.I64)
    (hscale' : packing.Packing.scale_to_fixed self scaleh = ok (core.result.Result.Ok scale))
    (minScale : Std.I64)
    (hminScale : packing.Packing.scale_to_fixed self minm = ok (core.result.Result.Ok minScale))
    (hwinn : EpilogueNominWindow coeffs1h.val (if b3h then codesH.val else codesL.val)
      (UScalar.cast UScalarTy.Usize self.group) scale self.bias val3h.val)
    (hwinm : EpilogueMinWindow coeffs1h.val (if b3h then codesH.val else codesL.val)
      (UScalar.cast UScalarTy.Usize self.group) scale minScale self.bias val3h.val) :
    ∃ vals : alloc.vec.Vec Std.I64,
      word_packing_decode_block self raw = ok (core.result.Result.Ok vals) ∧
      vals.val.map (fun x => x.val) =
        (List.range val3h.val).map (decodeRef bm scale.val self.bias.val minScale.val coeffs1h.val
          (if b3h then codesH.val else codesL.val) (UScalar.cast UScalarTy.Usize self.group).val) := by
  have hfields :=
    decode_block_fields_spec self raw bb hbb hshort sfh hsf ht scaleh hscale atq hatq bm atm minm
      hminbr val1h hgroups b2h at2h coeffs1h hloop4 hcoeffs hcoeffsAt val2h htotal val3h hblock
      bitsh hbits nb1h hdiv nb2h hmod nb3h hcb nbytesh hnbytes area1h hloop5 harea1 b3h hnib area2h
      hloop6 harea2 codesH codesL hloop7 hloop8 hcodesH hcodesL
  have hbs : packing.Packing.block_size self = ok (core.result.Result.Ok val3h) := by
    rw [← block_size_update_hasmin_sco self bm b2h]
    exact hblock
  have hgrp : packing.Packing.groups self = ok (core.result.Result.Ok val1h) := by
    rw [← groups_update_hasmin self bm]
    exact hgroups
  have hcl : coeffs1h.val.length = val1h.val := by
    have h := congrArg List.length hcoeffs
    simp [coeffVals] at h
    simpa using h
  have hcoefval : (↑(((if b3h then
        { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
      else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
      : packing.CanonicalBlock)).coeffs) = coeffs1h.val := by
    by_cases hb3 : b3h = true <;> simp [hb3]
  have hcodeval : (↑(((if b3h then
        { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
      else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
      : packing.CanonicalBlock)).codes) = (if b3h then codesH.val else codesL.val) := by
    by_cases hb3 : b3h = true <;> simp [hb3]
  have hcodeslen : (↑(((if b3h then
        { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
      else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
      : packing.CanonicalBlock)).codes : List Std.I64).length = val3h.val := by
    rw [hcodeval]
    by_cases hb3 : b3h = true <;> simp [hb3, hcodesH, hcodesL]
  have hcoeffslen : (↑(((if b3h then
        { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
      else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
      : packing.CanonicalBlock)).coeffs : List (Std.I32 × Std.I32)).length = val1h.val := by
    rw [hcoefval]
    exact hcl
  have hscale'' : packing.Packing.scale_to_fixed self
      (((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)).scale = ok (core.result.Result.Ok scale) := by
    by_cases hb3 : b3h = true <;> simpa [hb3] using hscale'
  have hminScale'' : packing.Packing.scale_to_fixed self
      (((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)).min = ok (core.result.Result.Ok minScale) := by
    by_cases hb3 : b3h = true <;> simpa [hb3] using hminScale
  have hwinn' : EpilogueNominWindow
      (↑(((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)).coeffs)
      (↑(((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)).codes)
      (UScalar.cast UScalarTy.Usize self.group) scale self.bias val3h.val := by
    rw [hcoefval, hcodeval]
    exact hwinn
  have hwinm' : EpilogueMinWindow
      (↑(((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)).coeffs)
      (↑(((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)).codes)
      (UScalar.cast UScalarTy.Usize self.group) scale minScale self.bias val3h.val := by
    rw [hcoefval, hcodeval]
    exact hwinm
  obtain ⟨vals, hv, hval'⟩ :=
    decode_block_spec w self raw
      ((if b3h then { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesH }
        else { scale := scaleh, min := minm, coeffs := coeffs1h, codes := codesL })
        : packing.CanonicalBlock)
      hw hfields val3h hbs hcodeslen val1h hgrp hcoeffslen scale hscale'' minScale hminScale''
      hwinn' hwinm'
  refine ⟨vals, hv, ?_⟩
  rw [hval', hbm]
  simp only [hcoefval, hcodeval]

/-! ## Residual

Covered: the `Fixed`-width top-level `epilogue`, the composed decoder value law over an arbitrary
successful `decode_block_fields` result (`decode_block_spec`), and its composition with
`decode_block_fields_spec` (`decode_block_spec_fields`). Not covered: the `Terminal` (`F16`/`F32`)
`epilogue_loop0` face and the `Trit` width. The concrete reference stays over the returned
`coeffs1h`/`codesH`/`codesL` lists; rewriting those to `coeffRef`/`unpackRefNat` pointwise (which
`hcoeffs`/`hcodesH`/`hcodesL` license) is a separate index step. -/

#print axioms apeiron_word.decodeRef
#print axioms apeiron_word.epilogue_fixed_nomin_spec
#print axioms apeiron_word.epilogue_fixed_min_spec
#print axioms apeiron_word.epilogue_fixed_spec
#print axioms apeiron_word.decode_block_spec
#print axioms apeiron_word.decode_block_spec_fields

end

end apeiron_word
