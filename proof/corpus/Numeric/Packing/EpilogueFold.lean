/-
Loop-level refinement of the extracted canonical packing epilogue's no-min `Fixed` fold
(`APEIRON-ALGEBRA-G3-PACK-W3-B7`).

`Numeric/Packing/Refinement.lean` proves the per-element value law `epilogue_entry_fixed_nomin` and
names the `Vec`-level fold of `packing.Packing.epilogue_loop2` as residual item 2. This module lifts
that per-element law across the loop on the same real extracted body
(`proof/ApeironWord.lean`, `packing.Packing.epilogue_loop2` at lines 10900-10960):

* `epilogueNominRef`: the closed-form reference value of weight `j`, the `getD`-based
  `affineWeight` the entry law computes (`has_min = false`);
* `EpilogueNominWindow`: the per-weight checked-`i64` headroom `epilogue_entry_fixed_nomin` asks for
  at every index below `block` — the coefficient-index bound and the four fits of
  `(scale·a)·(q + bias)` — stated as one predicate, with the code bound supplied separately by
  `block ≤ codes.length`;
* `epilogue_loop2_fixed_spec`: on the window, `epilogue_loop2` at `width = Fixed w`,
  `has_min = false` from `(out, false, error, 0)` returns `out` extended by the `block` reference
  weights, the flag clear and the error unchanged, as values (`Vec.val.map (·.val)`);
* `epilogue_entry_fixed_min`: the `has_min = true` mirror of the no-min per-element law, the
  `packing.mul`/`packing.sub` chain realizing `(scale·a)·(q + bias) − min_scale·b`;
* `epilogue_loop1_fixed_spec`: the same loop fold on the `has_min = true` face
  (`EpilogueMinWindow`, `epilogueMinRef`).

## Domain and residual

Both `Fixed` faces: the per-element arithmetic is `epilogue_entry_fixed_nomin` (no min) and
`epilogue_entry_fixed_min` (min); this file supplies the `List.range`-map fold and the top-level
loop equality for `epilogue_loop2` and `epilogue_loop1`. Not claimed here: the `Terminal`
(`F16`/`F32`) loop (`epilogue_loop0`), the non-`Fixed` widths, and the byte-level
`decode_block_fields` identity, all still residual in `Numeric/Packing/Refinement.lean`'s header.

## Complexity

`epilogue_loop2` visits `block` weights once with `O(1)` checked `i64` arithmetic and one push each:
`O(block)` time and space, counted in loop iterations and `i64` words. The figures name the loop.

## Trust boundary

Reaches the extracted `packing.Packing.epilogue_loop2(.body)` of `proof/ApeironWord.lean` at the
revision `Numeric.PackingRefinement` binds to. Does not cover rustc/LLVM codegen of
`src/format/packing.rs`, the kernel ABI, GPU/SPIR-V, or the `unsafe` container.
-/
import Numeric.Packing.Refinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The reference value, window and invariant -/

/-- The affine value of weight `j` on the no-min `Fixed` face, read off the block's coefficient and
code lists with the same `getD` defaults `epilogue_entry_fixed_nomin` uses. -/
def epilogueNominRef (scale bias : ℤ) (coeffs : List (Std.I32 × Std.I32)) (codes : List Std.I64)
    (group : ℕ) (j : ℕ) : ℤ :=
  affineWeight scale
    ((coeffs.getD (j / group) (0#i32, 0#i32)).1.val)
    ((codes.getD j 0#i64).val) bias 0 0 false

/-- The checked-`i64` headroom `epilogue_entry_fixed_nomin` asks for at weight `i`: in-bounds
coefficient and the four fits of `(scale·a)·(q + bias)`. The code bound `i < codes.length` is the
caller's `block ≤ codes.length`, which covers every `i < block`. -/
def EpilogueNominWindow (coeffs : List (Std.I32 × Std.I32)) (codes : List Std.I64)
    (group : Std.Usize) (scale : Std.I64) (bias : Std.I32) (block : ℕ) : Prop :=
  0 < group.val ∧
  ∀ i, i < block →
    i / group.val < coeffs.length ∧
    I64.min ≤ (codes.getD i 0#i64).val + bias.val ∧
    (codes.getD i 0#i64).val + bias.val ≤ I64.max ∧
    I64.min ≤ scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val ∧
    scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val ≤ I64.max ∧
    I64.min ≤ (scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val)
      * ((codes.getD i 0#i64).val + bias.val) ∧
    (scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val)
      * ((codes.getD i 0#i64).val + bias.val) ≤ I64.max

/-- Loop invariant: `i` weights appended to `out0`, each equal to its reference value as a value. -/
def EpilogueNominLoopInv (coeffs : List (Std.I32 × Std.I32)) (codes : List Std.I64)
    (group : Std.Usize) (scale bias : ℤ) (block : ℕ) (out0 out : alloc.vec.Vec Std.I64)
    (i : ℕ) : Prop :=
  i ≤ block ∧
  out.val.length = out0.val.length + i ∧
  out.val.map (fun x => x.val) =
    out0.val.map (fun x => x.val) ++
      (List.range i).map (epilogueNominRef scale bias coeffs codes group.val)

/-! ## The per-element value law, `Fixed`, with a min

The `has_min = true` mirror of `epilogue_entry_fixed_nomin`: the extracted body takes the
`packing.sub` branch, subtracting `min_scale·b` from `(scale·a)·(q + bias)`. -/

/-- **Per-element value law, `Fixed`, with min.** On the success domain (`has_min = true`), the
extracted `epilogue_entry` returns the checked-`i64` value of `affineWeight` at that weight: two
`packing.mul`s then the `packing.sub` of `min_scale·b`. The casts are the widening `I32 → I64`
casts, and the range hypotheses are exactly what makes the checked operations succeed. -/
theorem epilogue_entry_fixed_min
    (w : Std.U8) (hp : Std.U8) (blk : Std.U32) (grp : Std.U32) (sk : packing.ScaleKind)
    (sc : Bool) (bias : Std.I32) (sco : Bool) (po : packing.PlaneOrder)
    (no : packing.NibbleOrder) (b : packing.CanonicalBlock) (i : Std.Usize)
    (scale minScale : Std.I64) (group : Std.Usize)
    (hc : i.val < b.codes.val.length)
    (hg : 0 < group.val)
    (hj : i.val / group.val < b.coeffs.val.length)
    (ha : I64.min ≤ (b.codes.val.getD i.val 0#i64).val + bias.val)
    (ha' : (b.codes.val.getD i.val 0#i64).val + bias.val ≤ I64.max)
    (hb : I64.min ≤ scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
    (hb' : scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val ≤ I64.max)
    (hd : I64.min ≤ (scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
        * ((b.codes.val.getD i.val 0#i64).val + bias.val))
    (hd' : (scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
        * ((b.codes.val.getD i.val 0#i64).val + bias.val) ≤ I64.max)
    (hm : I64.min ≤ minScale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).2.val)
    (hm' : minScale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).2.val ≤ I64.max)
    (hs : I64.min ≤ (scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
        * ((b.codes.val.getD i.val 0#i64).val + bias.val)
        - minScale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).2.val)
    (hs' : (scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
        * ((b.codes.val.getD i.val 0#i64).val + bias.val)
        - minScale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).2.val ≤ I64.max) :
    packing.Packing.epilogue_entry
      { width := packing.CodeWidth.Fixed w, high_planes := hp, block := blk, group := grp,
        scale_kind := sk, has_min := true, signed_code := sc, bias := bias,
        signed_coeff := sco, plane_order := po, nibble_order := no }
      b i scale minScale group ⦃ r =>
        ∃ z, r = core.result.Result.Ok z ∧
          z.val = affineWeight scale.val
            (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val
            (b.codes.val.getD i.val 0#i64).val bias.val minScale.val
            (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).2.val true ⦄ := by
  unfold packing.Packing.epilogue_entry
  dsimp only
  refine WP.spec_bind (Usize.div_spec i (y := group) (by omega)) ?_
  intro i1 hi1v
  have hci1 : i1.val < b.coeffs.val.length := by rw [hi1v]; exact hj
  rw [alloc.vec.Vec.index_slice_index]
  refine WP.spec_bind (alloc.vec.Vec.index_usize_spec b.coeffs i1 hci1) ?_
  intro am ham
  obtain ⟨a, mb⟩ := am
  have hav : a.val = (b.coeffs.val.getD i1.val (0#i32, 0#i32)).1.val := by
    rw [getD_eq_getElem (b.coeffs.val) i1.val (0#i32, 0#i32) hci1]
    simpa using congrArg (fun p : Std.I32 × Std.I32 => p.1.val) ham
  have hmv : mb.val = (b.coeffs.val.getD i1.val (0#i32, 0#i32)).2.val := by
    rw [getD_eq_getElem (b.coeffs.val) i1.val (0#i32, 0#i32) hci1]
    simpa using congrArg (fun p : Std.I32 × Std.I32 => p.2.val) ham
  rw [alloc.vec.Vec.index_slice_index]
  refine WP.spec_bind (alloc.vec.Vec.index_usize_spec b.codes i hc) ?_
  intro i2 hi2
  have hi2d : i2.val = (b.codes.val.getD i.val 0#i64).val := by
    rw [hi2, getD_eq_getElem (b.codes.val) i.val 0#i64 hc]
  simp only [lift]
  refine WP.spec_bind ((WP.spec_ok (IScalar.cast IScalarTy.I64 bias)).mpr rfl) ?_
  intro i3 hi3
  have hcast : (IScalar.cast IScalarTy.I64 bias).val = bias.val := by simp
  have hi3v : i3.val = bias.val := by rw [← hi3, hcast]
  have hloa : I64.min ≤ i2.val + i3.val := by rw [hi2d, hi3v]; exact ha
  have hhia : i2.val + i3.val ≤ I64.max := by rw [hi2d, hi3v]; exact ha'
  obtain ⟨cadd, hcadd, hcaddv⟩ := I64_checked_add_some i2 i3 hloa hhia
  refine WP.spec_bind
    (by rw [hcadd]; exact (WP.spec_ok (some cadd)).mpr rfl) ?_
  intro o ho
  refine WP.spec_bind
    (by rw [← ho, core.option.Option.ok_or_some]
        exact (WP.spec_ok (core.result.Result.Ok cadd)).mpr rfl) ?_
  intro r hr
  refine WP.spec_bind
    (by rw [← hr]; exact core.result.Result.Insts.CoreOpsTry.branch_Ok.spec cadd) ?_
  intro cf hcf
  simp only [hcf]
  refine WP.spec_bind ((WP.spec_ok (IScalar.cast IScalarTy.I64 a)).mpr rfl) ?_
  intro i4 hi4
  have hcasta : (IScalar.cast IScalarTy.I64 a).val = a.val := by simp
  have hb1 : I64.min ≤ scale.val * i4.val := by
    rw [← hi4, hcasta, hav, hi1v]; exact hb
  have hb1' : scale.val * i4.val ≤ I64.max := by
    rw [← hi4, hcasta, hav, hi1v]; exact hb'
  obtain ⟨sa, hsa, hsav⟩ := packing_mul_ok scale i4 hb1 hb1'
  refine WP.spec_bind
    (by rw [hsa]; exact (WP.spec_ok (core.result.Result.Ok sa)).mpr rfl) ?_
  intro r1 hr1
  refine WP.spec_bind
    (by rw [← hr1]; exact core.result.Result.Insts.CoreOpsTry.branch_Ok.spec sa) ?_
  intro cf1 hcf1
  simp only [hcf1]
  have hd1 : I64.min ≤ sa.val * cadd.val := by
    rw [hsav, hcaddv, hi2d, hi3v, ← hi4, hcasta, hav, hi1v]; exact hd
  have hd1' : sa.val * cadd.val ≤ I64.max := by
    rw [hsav, hcaddv, hi2d, hi3v, ← hi4, hcasta, hav, hi1v]; exact hd'
  obtain ⟨z, hz, hzv⟩ := packing_mul_ok sa cadd hd1 hd1'
  refine WP.spec_bind
    (by rw [hz]; exact (WP.spec_ok (core.result.Result.Ok z)).mpr rfl) ?_
  intro r2 hr2
  refine WP.spec_bind
    (by rw [← hr2]; exact core.result.Result.Insts.CoreOpsTry.branch_Ok.spec z) ?_
  intro cf2 hcf2
  simp only [hcf2]
  simp only [if_pos trivial]
  refine WP.spec_bind ((WP.spec_ok (IScalar.cast IScalarTy.I64 mb)).mpr rfl) ?_
  intro i5 hi5
  have hcastmb : (IScalar.cast IScalarTy.I64 mb).val = mb.val := by simp
  have hi5v : i5.val = mb.val := by rw [← hi5, hcastmb]
  have hm1 : I64.min ≤ minScale.val * i5.val := by rw [hi5v, hmv, hi1v]; exact hm
  have hm1' : minScale.val * i5.val ≤ I64.max := by rw [hi5v, hmv, hi1v]; exact hm'
  obtain ⟨msb, hmsb, hmsbv⟩ := packing_mul_ok minScale i5 hm1 hm1'
  refine WP.spec_bind
    (by rw [hmsb]; exact (WP.spec_ok (core.result.Result.Ok msb)).mpr rfl) ?_
  intro r3 hr3
  refine WP.spec_bind
    (by rw [← hr3]; exact core.result.Result.Insts.CoreOpsTry.branch_Ok.spec msb) ?_
  intro cf3 hcf3
  simp only [hcf3]
  have hzeq : z.val = (scale.val
        * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
      * ((b.codes.val.getD i.val 0#i64).val + bias.val) := by
    rw [hzv, hsav, hcaddv, hi2d, hi3v, ← hi4, hcasta, hav, hi1v]
  have hmbeq : msb.val = minScale.val
      * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).2.val := by
    rw [hmsbv, hi5v, hmv, hi1v]
  have hs1 : I64.min ≤ z.val - msb.val := by rw [hzeq, hmbeq]; exact hs
  have hs1' : z.val - msb.val ≤ I64.max := by rw [hzeq, hmbeq]; exact hs'
  obtain ⟨val4, hval4, hval4v⟩ := packing_sub_ok z msb hs1 hs1'
  refine WP.spec_bind
    (by rw [hval4]; exact (WP.spec_ok (core.result.Result.Ok val4)).mpr rfl) ?_
  intro r4 hr4
  refine WP.spec_bind
    (by rw [← hr4]; exact core.result.Result.Insts.CoreOpsTry.branch_Ok.spec val4) ?_
  intro cf4 hcf4
  simp only [hcf4]
  rw [WP.spec_ok]
  refine ⟨val4, rfl, ?_⟩
  rw [hval4v, hzeq, hmbeq]
  unfold affineWeight
  simp

/-! ## One loop iteration -/

/-- **One `epilogue_loop2` iteration, no min.** With the window at `i6 < block` and the invariant,
the body computes `epilogue_entry_fixed_nomin`'s value, pushes it and continues at `i6 + 1` with the
invariant restored (flag clear, error unchanged). -/
theorem epilogue_loop2_body_step_fixed
    (w : Std.U8) (hp : Std.U8) (blk : Std.U32) (grp : Std.U32) (sk : packing.ScaleKind)
    (sc : Bool) (bias : Std.I32) (sco : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (i4 i5 : Std.U16) (v : alloc.vec.Vec (Std.I32 × Std.I32)) (v1 : alloc.vec.Vec Std.I64)
    (block group : Std.Usize) (scale : Std.I64) (out0 out : alloc.vec.Vec Std.I64)
    (error : packing.PackingError) (i6 : Std.Usize)
    (hwin : EpilogueNominWindow v.val v1.val group scale bias block.val)
    (hcd : block.val ≤ v1.val.length)
    (hcap : out0.val.length + block.val ≤ Usize.max)
    (hinv : EpilogueNominLoopInv v.val v1.val group scale.val bias.val block.val out0 out i6.val)
    (hlt : i6.val < block.val) :
    ∃ (out2 : alloc.vec.Vec Std.I64) (i7 : Std.Usize),
      packing.Packing.epilogue_loop2.body (packing.CodeWidth.Fixed w) hp blk grp sk sc bias sco po no
          i4 i5 v v1 block group scale out false error i6
        = ok (ControlFlow.cont (out2, false, error, i7)) ∧
      EpilogueNominLoopInv v.val v1.val group scale.val bias.val block.val out0 out2 i7.val ∧
      i7.val = i6.val + 1 := by
  obtain ⟨hgroup, hw⟩ := hwin
  obtain ⟨_, hlen, hmap⟩ := hinv
  obtain ⟨hj, ha, ha', hb, hb', hd, hd'⟩ := hw i6.val hlt
  have hc : i6.val < v1.val.length := by omega
  obtain ⟨cf, hentry, hpost⟩ := WP.spec_imp_exists
    (epilogue_entry_fixed_nomin w hp blk grp sk sc bias sco po no
      ({ scale := i4, min := i5, coeffs := v, codes := v1 } : packing.CanonicalBlock) i6 scale
      0#i64 group hc hgroup hj ha ha' hb hb' hd hd')
  obtain ⟨z, hcf, hzv⟩ := hpost
  have houtlt : out.val.length < Usize.max := by rw [hlen]; omega
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out z houtlt)
  obtain ⟨i7, hadd, hi7v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i6) (y := 1#usize) (by scalar_tac))
  have hi7' : i7.val = i6.val + 1 := by rw [hi7v]; simp
  have hbody : packing.Packing.epilogue_loop2.body (packing.CodeWidth.Fixed w) hp blk grp sk sc bias
      sco po no i4 i5 v v1 block group scale out false error i6
      = ok (ControlFlow.cont (out2, false, error, i7)) := by
    unfold packing.Packing.epilogue_loop2.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt),
      if_neg (by decide : ¬ ((false : Bool) = true))]
    rw [hentry, hcf]
    simp only [bind_tc_ok]
    rw [hpush, hadd]
    simp only [bind_tc_ok]
    rfl
  refine ⟨out2, i7, hbody, ?_, hi7'⟩
  refine ⟨by omega, ?_, ?_⟩
  · rw [hpushv, List.length_append, List.length_singleton, hlen, hi7']
    omega
  · have hsp : (List.range (i6.val + 1)).map
          (epilogueNominRef scale.val bias.val v.val v1.val group.val)
        = (List.range i6.val).map (epilogueNominRef scale.val bias.val v.val v1.val group.val)
          ++ [epilogueNominRef scale.val bias.val v.val v1.val group.val i6.val] := by
      rw [List.range_succ, List.map_append]
      simp
    rw [hpushv, List.map_append, List.map_singleton, hmap, hi7', hsp, hzv, List.append_assoc]
    simp only [epilogueNominRef]

/-! ## The loop -/

/-- **`epilogue_loop2` on the `Fixed`/`has_min = false` face.** On the per-weight window, from
`(out, false, error, 0)` the loop returns `out` extended by the `block` reference weights, the flag
clear and the error unchanged, as values. -/
theorem epilogue_loop2_fixed_spec
    (w : Std.U8) (hp : Std.U8) (blk : Std.U32) (grp : Std.U32) (sk : packing.ScaleKind)
    (sc : Bool) (bias : Std.I32) (sco : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (b : packing.CanonicalBlock) (block group : Std.Usize) (scale : Std.I64)
    (out0 : alloc.vec.Vec Std.I64) (error : packing.PackingError)
    (hwin : EpilogueNominWindow b.coeffs.val b.codes.val group scale bias block.val)
    (hcodes : block.val ≤ b.codes.val.length)
    (hcap : out0.val.length + block.val ≤ Usize.max) :
    ∃ out' : alloc.vec.Vec Std.I64,
      packing.Packing.epilogue_loop2 (packing.CodeWidth.Fixed w) hp blk grp sk sc bias sco po no
          b.scale b.min b.coeffs b.codes block group scale out0 false error 0#usize
        = ok (out', false, error) ∧
      out'.val.map (fun x => x.val) =
        out0.val.map (fun x => x.val) ++
          (List.range block.val).map
            (epilogueNominRef scale.val bias.val b.coeffs.val b.codes.val group.val) := by
  have hinv0 : EpilogueNominLoopInv b.coeffs.val b.codes.val group scale.val bias.val block.val
      out0 out0 0 := by
    refine ⟨Nat.zero_le _, by simp, by simp⟩
  have hspec : WP.spec
      (packing.Packing.epilogue_loop2 (packing.CodeWidth.Fixed w) hp blk grp sk sc bias sco po no
        b.scale b.min b.coeffs b.codes block group scale out0 false error 0#usize)
      (fun r : alloc.vec.Vec Std.I64 × Bool × packing.PackingError =>
        r.2.1 = false ∧ r.2.2 = error ∧
        r.1.val.map (fun x => x.val) =
          out0.val.map (fun x => x.val) ++
            (List.range block.val).map
              (epilogueNominRef scale.val bias.val b.coeffs.val b.codes.val group.val)) := by
    unfold packing.Packing.epilogue_loop2
    apply loop.spec_decr_nat
      (measure := fun st : alloc.vec.Vec Std.I64 × Bool × packing.PackingError × Std.Usize =>
        block.val - st.2.2.2.val)
      (inv := fun st : alloc.vec.Vec Std.I64 × Bool × packing.PackingError × Std.Usize =>
        st.2.1 = false ∧ st.2.2.1 = error ∧
        EpilogueNominLoopInv b.coeffs.val b.codes.val group scale.val bias.val block.val out0 st.1
          st.2.2.2.val)
      (post := fun r : alloc.vec.Vec Std.I64 × Bool × packing.PackingError =>
        r.2.1 = false ∧ r.2.2 = error ∧
        r.1.val.map (fun x => x.val) =
          out0.val.map (fun x => x.val) ++
            (List.range block.val).map
              (epilogueNominRef scale.val bias.val b.coeffs.val b.codes.val group.val))
    · rintro ⟨outX, flgX, errX, iX⟩ ⟨hfl, herr, hinv⟩
      subst hfl
      subst herr
      dsimp only
      by_cases hlt : iX.val < block.val
      · obtain ⟨out2, i7, hbody, hinv2, hi7⟩ :=
          epilogue_loop2_body_step_fixed w hp blk grp sk sc bias sco po no b.scale b.min b.coeffs
            b.codes block group scale out0 outX errX iX hwin hcodes hcap hinv hlt
        rw [hbody, WP.spec_ok]
        refine ⟨⟨rfl, rfl, hinv2⟩, ?_⟩
        show block.val - i7.val < block.val - iX.val
        rw [hi7]
        omega
      · have hnot : ¬ (iX < block) := by rw [UScalar.lt_equiv]; exact hlt
        have hbody : packing.Packing.epilogue_loop2.body (packing.CodeWidth.Fixed w) hp blk grp sk sc
            bias sco po no b.scale b.min b.coeffs b.codes block group scale outX false errX iX
            = ok (ControlFlow.done (outX, false, errX)) := by
          unfold packing.Packing.epilogue_loop2.body
          rw [if_neg hnot]
        rw [hbody, WP.spec_ok]
        have hiXeq : iX.val = block.val := by
          have hle : iX.val ≤ block.val := by simpa using hinv.1
          omega
        refine ⟨rfl, rfl, ?_⟩
        obtain ⟨_, hlen, hmap⟩ := hinv
        rw [hiXeq] at hmap
        exact hmap
    · exact ⟨rfl, rfl, hinv0⟩
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  obtain ⟨hfl, herr, hmap⟩ := hpost
  refine ⟨r.1, ?_, hmap⟩
  rw [hr]
  congr 1
  exact Prod.ext rfl (Prod.ext hfl herr)

/-! ## The `has_min = true` loop

`epilogue_loop1` is `epilogue_loop2` with the min branch: the per-element law is
`epilogue_entry_fixed_min`, and the reference subtracts `min_scale·b`. -/

/-- The affine value of weight `j` on the `Fixed`/`has_min = true` face. -/
def epilogueMinRef (scale bias minScale : ℤ) (coeffs : List (Std.I32 × Std.I32))
    (codes : List Std.I64) (group : ℕ) (j : ℕ) : ℤ :=
  affineWeight scale
    ((coeffs.getD (j / group) (0#i32, 0#i32)).1.val)
    ((codes.getD j 0#i64).val) bias minScale
    ((coeffs.getD (j / group) (0#i32, 0#i32)).2.val) true

/-- The checked-`i64` headroom `epilogue_entry_fixed_min` asks for at weight `i`: the
coefficient-index bound, the four fits of `(scale·a)·(q + bias)`, the min-multiply fit and the
subtract fit. The code bound is the caller's `block ≤ codes.length`. -/
def EpilogueMinWindow (coeffs : List (Std.I32 × Std.I32)) (codes : List Std.I64)
    (group : Std.Usize) (scale minScale : Std.I64) (bias : Std.I32) (block : ℕ) : Prop :=
  0 < group.val ∧
  ∀ i, i < block →
    i / group.val < coeffs.length ∧
    I64.min ≤ (codes.getD i 0#i64).val + bias.val ∧
    (codes.getD i 0#i64).val + bias.val ≤ I64.max ∧
    I64.min ≤ scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val ∧
    scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val ≤ I64.max ∧
    I64.min ≤ (scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val)
      * ((codes.getD i 0#i64).val + bias.val) ∧
    (scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val)
      * ((codes.getD i 0#i64).val + bias.val) ≤ I64.max ∧
    I64.min ≤ minScale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).2.val ∧
    minScale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).2.val ≤ I64.max ∧
    I64.min ≤ (scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val)
      * ((codes.getD i 0#i64).val + bias.val)
      - minScale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).2.val ∧
    (scale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).1.val)
      * ((codes.getD i 0#i64).val + bias.val)
      - minScale.val * (coeffs.getD (i / group.val) (0#i32, 0#i32)).2.val ≤ I64.max

/-- The `has_min = true` loop invariant. -/
def EpilogueMinLoopInv (coeffs : List (Std.I32 × Std.I32)) (codes : List Std.I64)
    (group : Std.Usize) (scale bias minScale : ℤ) (block : ℕ)
    (out0 out : alloc.vec.Vec Std.I64) (i : ℕ) : Prop :=
  i ≤ block ∧
  out.val.length = out0.val.length + i ∧
  out.val.map (fun x => x.val) =
    out0.val.map (fun x => x.val) ++
      (List.range i).map (epilogueMinRef scale bias minScale coeffs codes group.val)

/-- **One `epilogue_loop1` iteration, with min.** -/
theorem epilogue_loop1_body_step_fixed
    (w : Std.U8) (hp : Std.U8) (blk : Std.U32) (grp : Std.U32) (sk : packing.ScaleKind)
    (sc : Bool) (bias : Std.I32) (sco : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (i4 i5 : Std.U16) (v : alloc.vec.Vec (Std.I32 × Std.I32)) (v1 : alloc.vec.Vec Std.I64)
    (block group : Std.Usize) (scale minScale : Std.I64) (out0 out : alloc.vec.Vec Std.I64)
    (error : packing.PackingError) (i6 : Std.Usize)
    (hwin : EpilogueMinWindow v.val v1.val group scale minScale bias block.val)
    (hcd : block.val ≤ v1.val.length)
    (hcap : out0.val.length + block.val ≤ Usize.max)
    (hinv : EpilogueMinLoopInv v.val v1.val group scale.val bias.val minScale.val block.val out0 out
      i6.val)
    (hlt : i6.val < block.val) :
    ∃ (out2 : alloc.vec.Vec Std.I64) (i7 : Std.Usize),
      packing.Packing.epilogue_loop1.body (packing.CodeWidth.Fixed w) hp blk grp sk sc bias sco po no
          i4 i5 v v1 block group scale minScale out false error i6
        = ok (ControlFlow.cont (out2, false, error, i7)) ∧
      EpilogueMinLoopInv v.val v1.val group scale.val bias.val minScale.val block.val out0 out2
        i7.val ∧
      i7.val = i6.val + 1 := by
  obtain ⟨hgroup, hw⟩ := hwin
  obtain ⟨_, hlen, hmap⟩ := hinv
  obtain ⟨hj, ha, ha', hb, hb', hd, hd', hm, hm', hs, hs'⟩ := hw i6.val hlt
  have hc : i6.val < v1.val.length := by omega
  obtain ⟨cf, hentry, hpost⟩ := WP.spec_imp_exists
    (epilogue_entry_fixed_min w hp blk grp sk sc bias sco po no
      ({ scale := i4, min := i5, coeffs := v, codes := v1 } : packing.CanonicalBlock) i6 scale
      minScale group hc hgroup hj ha ha' hb hb' hd hd' hm hm' hs hs')
  obtain ⟨z, hcf, hzv⟩ := hpost
  have houtlt : out.val.length < Usize.max := by rw [hlen]; omega
  obtain ⟨out2, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out z houtlt)
  obtain ⟨i7, hadd, hi7v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i6) (y := 1#usize) (by scalar_tac))
  have hi7' : i7.val = i6.val + 1 := by rw [hi7v]; simp
  have hbody : packing.Packing.epilogue_loop1.body (packing.CodeWidth.Fixed w) hp blk grp sk sc bias
      sco po no i4 i5 v v1 block group scale minScale out false error i6
      = ok (ControlFlow.cont (out2, false, error, i7)) := by
    unfold packing.Packing.epilogue_loop1.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hlt),
      if_neg (by decide : ¬ ((false : Bool) = true))]
    rw [hentry, hcf]
    simp only [bind_tc_ok]
    rw [hpush, hadd]
    simp only [bind_tc_ok]
    rfl
  refine ⟨out2, i7, hbody, ?_, hi7'⟩
  refine ⟨by omega, ?_, ?_⟩
  · rw [hpushv, List.length_append, List.length_singleton, hlen, hi7']
    omega
  · have hsp : (List.range (i6.val + 1)).map
          (epilogueMinRef scale.val bias.val minScale.val v.val v1.val group.val)
        = (List.range i6.val).map
            (epilogueMinRef scale.val bias.val minScale.val v.val v1.val group.val)
          ++ [epilogueMinRef scale.val bias.val minScale.val v.val v1.val group.val i6.val] := by
      rw [List.range_succ, List.map_append]
      simp
    rw [hpushv, List.map_append, List.map_singleton, hmap, hi7', hsp, hzv, List.append_assoc]
    simp only [epilogueMinRef]

/-- **`epilogue_loop1` on the `Fixed`/`has_min = true` face.** On the per-weight min window, from
`(out, false, error, 0)` the loop returns `out` extended by the `block` min-reference weights, the
flag clear and the error unchanged, as values. -/
theorem epilogue_loop1_fixed_spec
    (w : Std.U8) (hp : Std.U8) (blk : Std.U32) (grp : Std.U32) (sk : packing.ScaleKind)
    (sc : Bool) (bias : Std.I32) (sco : Bool) (po : packing.PlaneOrder) (no : packing.NibbleOrder)
    (b : packing.CanonicalBlock) (block group : Std.Usize) (scale minScale : Std.I64)
    (out0 : alloc.vec.Vec Std.I64) (error : packing.PackingError)
    (hwin : EpilogueMinWindow b.coeffs.val b.codes.val group scale minScale bias block.val)
    (hcodes : block.val ≤ b.codes.val.length)
    (hcap : out0.val.length + block.val ≤ Usize.max) :
    ∃ out' : alloc.vec.Vec Std.I64,
      packing.Packing.epilogue_loop1 (packing.CodeWidth.Fixed w) hp blk grp sk sc bias sco po no
          b.scale b.min b.coeffs b.codes block group scale minScale out0 false error 0#usize
        = ok (out', false, error) ∧
      out'.val.map (fun x => x.val) =
        out0.val.map (fun x => x.val) ++
          (List.range block.val).map
            (epilogueMinRef scale.val bias.val minScale.val b.coeffs.val b.codes.val group.val) := by
  have hinv0 : EpilogueMinLoopInv b.coeffs.val b.codes.val group scale.val bias.val minScale.val
      block.val out0 out0 0 := by
    refine ⟨Nat.zero_le _, by simp, by simp⟩
  have hspec : WP.spec
      (packing.Packing.epilogue_loop1 (packing.CodeWidth.Fixed w) hp blk grp sk sc bias sco po no
        b.scale b.min b.coeffs b.codes block group scale minScale out0 false error 0#usize)
      (fun r : alloc.vec.Vec Std.I64 × Bool × packing.PackingError =>
        r.2.1 = false ∧ r.2.2 = error ∧
        r.1.val.map (fun x => x.val) =
          out0.val.map (fun x => x.val) ++
            (List.range block.val).map
              (epilogueMinRef scale.val bias.val minScale.val b.coeffs.val b.codes.val group.val)) := by
    unfold packing.Packing.epilogue_loop1
    apply loop.spec_decr_nat
      (measure := fun st : alloc.vec.Vec Std.I64 × Bool × packing.PackingError × Std.Usize =>
        block.val - st.2.2.2.val)
      (inv := fun st : alloc.vec.Vec Std.I64 × Bool × packing.PackingError × Std.Usize =>
        st.2.1 = false ∧ st.2.2.1 = error ∧
        EpilogueMinLoopInv b.coeffs.val b.codes.val group scale.val bias.val minScale.val block.val
          out0 st.1 st.2.2.2.val)
      (post := fun r : alloc.vec.Vec Std.I64 × Bool × packing.PackingError =>
        r.2.1 = false ∧ r.2.2 = error ∧
        r.1.val.map (fun x => x.val) =
          out0.val.map (fun x => x.val) ++
            (List.range block.val).map
              (epilogueMinRef scale.val bias.val minScale.val b.coeffs.val b.codes.val group.val))
    · rintro ⟨outX, flgX, errX, iX⟩ ⟨hfl, herr, hinv⟩
      subst hfl
      subst herr
      dsimp only
      by_cases hlt : iX.val < block.val
      · obtain ⟨out2, i7, hbody, hinv2, hi7⟩ :=
          epilogue_loop1_body_step_fixed w hp blk grp sk sc bias sco po no b.scale b.min b.coeffs
            b.codes block group scale minScale out0 outX errX iX hwin hcodes hcap hinv hlt
        rw [hbody, WP.spec_ok]
        refine ⟨⟨rfl, rfl, hinv2⟩, ?_⟩
        show block.val - i7.val < block.val - iX.val
        rw [hi7]
        omega
      · have hnot : ¬ (iX < block) := by rw [UScalar.lt_equiv]; exact hlt
        have hbody : packing.Packing.epilogue_loop1.body (packing.CodeWidth.Fixed w) hp blk grp sk sc
            bias sco po no b.scale b.min b.coeffs b.codes block group scale minScale outX false errX
            iX = ok (ControlFlow.done (outX, false, errX)) := by
          unfold packing.Packing.epilogue_loop1.body
          rw [if_neg hnot]
        rw [hbody, WP.spec_ok]
        have hiXeq : iX.val = block.val := by
          have hle : iX.val ≤ block.val := by simpa using hinv.1
          omega
        refine ⟨rfl, rfl, ?_⟩
        obtain ⟨_, hlen, hmap⟩ := hinv
        rw [hiXeq] at hmap
        exact hmap
    · exact ⟨rfl, rfl, hinv0⟩
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  obtain ⟨hfl, herr, hmap⟩ := hpost
  refine ⟨r.1, ?_, hmap⟩
  rw [hr]
  congr 1
  exact Prod.ext rfl (Prod.ext hfl herr)

end

end apeiron_word
