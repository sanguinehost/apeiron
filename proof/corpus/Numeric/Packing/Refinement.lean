/-
Refinement of the extracted canonical packing decoder.

The Rust body is `src/format/packing.rs` (the canonical pack/decode grammar; GGUF is only the
tested import boundary). The extraction root is `apeiron_word.word_packing_decode_block`
(`proof/ApeironWord.lean`, `packing.Packing.decode_block`), which is

  `decode_block p raw = epilogue p (decode_block_fields p raw)`.

The documented specification (`src/format/packing.rs:634-738`), at the canonical-block level:

* a block is `(scale, min, coeffs, codes)`; `codes` holds one interpreted code per weight;
* the affine epilogue law is `w_i = scale·a_g·(codes_i + bias) - min_scale·b_g` with
  `g = i / group`, `scale = scale_to_fixed p.scale`, `min_scale = scale_to_fixed p.min`;
* the terminal `F16`/`F32` faces are element-wise instead: `F16` exact, `F32` one
  round-to-nearest reduction (`f32_bits_to_fixed`) and refusals on non-finite/overflow/underflow;
* the refusals are the named `PackingError` constructors: `Truncated` when `raw` is shorter than
  `block_bytes p`, `BadWidth`/`BadBlock`/`BadGeometry` from the geometry validation, and `Overflow`
  when a checked `i64` product/add/sub leaves range. No value is clamped.

## What this file discharges, exactly, and what stays residual

Proved here:

* `word_packing_decode_block_truncated`: for any packing whose `block_bytes` evaluates to `bb`, a
  payload shorter than `bb` returns `Err (Truncated bb raw.len)` exactly;
* `word_packing_decode_block_block_error`: a `block_bytes` refusal `Err e` propagates as `Err e`;
* `word_packing_decode_block_eq_epilogue`: the success path is the epilogue of the decoded fields
  (the composition the spec names), so the top-level obligation factors into `decode_block_fields`
  and `epilogue`;
* the checked-`i64` witness lemmas the epilogue law needs (`I64_checked_mul_some`,
  `I64_checked_sub_some`, `I64_checked_add_some`), restated only because the corpus exports the
  `i128` and `i64` add variants and not these two;
* `epilogue_entry_fixed_nomin`: the per-element value law on the `Fixed`/`has_min = false` face —
  the extracted `packing.Packing.epilogue_entry` returns `Ok z` with `z.val` exactly
  `affineWeight` at that weight (`blockWeight`), the two checked multiplies realizing
  `(scale·a)·(q + bias)`.

Residual and **not** claimed: the structural identity of `decode_block_fields` (the byte-level
nibble/bit-plane unpack, which does *not* call `word_q4_k_codes`/`word_q6_k_codes` but the generic
`unpack_code`; its per-code reference law is `unpack_code_ref` in `Numeric/Packing/Unpack.lean`, and the
loop over the code area is not yet composed), and the top-level `epilogue` assembly. The
`Vec`-level folds of `epilogue_loop2` (no-min) and `epilogue_loop1` (`has_min`) on the `Fixed` width
are `epilogue_loop2_fixed_spec`/`epilogue_loop1_fixed_spec` in `Numeric/Packing/EpilogueFold.lean`. Per the plan, the code
extraction is already proved for Q4_K/Q6_K; this file supplies the affine fold and the refusals on
the same real body.

## Complexity

`decode_block_fields` visits the `block` weights once, reading `total_code_bits` bits each and
writing `ceil(block·total/8)` code bytes: `O(block·total/8)` time and space. The affine fold
`epilogue` visits the `block` weights once with `O(1)` checked `i64` arithmetic each: `O(block)`
time, `O(block)` output. The refusal test is `O(1)`. `block ∈ {1, 32, 64, 256}` is fixed by the
grammar, so both are O(1) in the served model's step count; the figures name the loops.
-/
import Numeric.Packing.Format

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## The pure affine epilogue law

The reference integer value of one weight, matching the operation order of
`packing.Packing.epilogue_entry`: `(scale·a)·(codes_i + bias)` minus `min_scale·b` when the packing
carries a min. Exact over `ℤ`; the extracted body computes the same value whenever no checked
`i64` operation leaves range. -/

/-- The affine epilogue value of one weight: `(scale·a)·(q + bias) − (has_min ? min_scale·b : 0)`. -/
def affineWeight (scale a q bias minScale mb : ℤ) (hasMin : Bool) : ℤ :=
  (scale * a) * (q + bias) - (if hasMin then minScale * mb else 0)

/-! ## Checked-`i64` witnesses

`packing.mul`/`packing.sub` are `I64.checked_mul`/`I64.checked_sub` narrowed through
`Option.ok_or Overflow`. The corpus already exports `I64_checked_add_some` (`EngineSiluRefinement`)
and the `i128` variants; the multiply and subtract witnesses are restated here, from the same
`*_bv_spec` shape, because no numeric module exports them. -/

theorem I64_checked_mul_some (x y : I64)
    (hlo : I64.min ≤ x.val * y.val) (hhi : x.val * y.val ≤ I64.max) :
    ∃ z, x.checked_mul y = some z ∧ z.val = x.val * y.val := by
  have hspec := I64.checked_mul_bv_spec x y
  cases hc : core.num.checked_mul_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_mul] using hc, hspec.2.2.1⟩

theorem I64_checked_sub_some (x y : I64)
    (hlo : I64.min ≤ x.val - y.val) (hhi : x.val - y.val ≤ I64.max) :
    ∃ z, x.checked_sub y = some z ∧ z.val = x.val - y.val := by
  have hspec := I64.checked_sub_bv_spec x y
  cases hc : core.num.checked_sub_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_sub] using hc, hspec.2.2.1⟩

/-! ## The top-level composition

`word_packing_decode_block` is the public wrapper over `packing.Packing.decode_block`, and
`decode_block` is `decode_block_fields` bound into `epilogue`. The success path therefore factors
exactly; the residual is the two halves' structural identities, not the composition. -/

/-- The public wrapper unfolds to the extracted `decode_block`. -/
theorem word_packing_decode_block_eq (p : packing.Packing) (raw : Slice Std.U8) :
    word_packing_decode_block p raw = packing.Packing.decode_block p raw := rfl

/-- The success path of the decoder is the epilogue of the decoded canonical block. -/
theorem packing_decode_block_eq_epilogue (p : packing.Packing) (raw : Slice Std.U8)
    (b : packing.CanonicalBlock)
    (hfields : packing.Packing.decode_block_fields p raw = ok (core.result.Result.Ok b)) :
    word_packing_decode_block p raw = packing.Packing.epilogue p b := by
  rw [word_packing_decode_block_eq]
  unfold packing.Packing.decode_block
  rw [hfields]
  simp [core.result.Result.Insts.CoreOpsTry.branch]

/-! ## Refusal equalities -/

/-- **Truncation refusal.** For any packing whose `block_bytes` evaluates to `bb`, a payload
shorter than `bb` returns `Err (Truncated bb raw.len)` exactly; nothing is read. -/
theorem word_packing_decode_block_truncated (p : packing.Packing) (raw : Slice Std.U8)
    (bb : Std.Usize) (hbb : packing.Packing.block_bytes p = ok (core.result.Result.Ok bb))
    (hshort : Slice.len raw < bb) :
    word_packing_decode_block p raw =
      ok (core.result.Result.Err
        (packing.PackingError.Truncated bb (Slice.len raw))) := by
  rw [word_packing_decode_block_eq]
  unfold packing.Packing.decode_block packing.Packing.decode_block_fields
  rw [hbb]
  simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
  rw [if_pos hshort]
  simp [core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]

/-- **Geometry refusal propagation.** A `block_bytes` refusal `Err e` propagates unchanged through
`decode_block_fields` and the top-level decoder; no code is unpacked. -/
theorem word_packing_decode_block_block_error (p : packing.Packing) (raw : Slice Std.U8)
    (e : packing.PackingError)
    (hbb : packing.Packing.block_bytes p = ok (core.result.Result.Err e)) :
    word_packing_decode_block p raw = ok (core.result.Result.Err e) := by
  rw [word_packing_decode_block_eq]
  unfold packing.Packing.decode_block packing.Packing.decode_block_fields
  rw [hbb]
  simp [core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual]

/-! ## The affine fold's arithmetic primitives

`packing.mul` and `packing.sub` are the checked `i64` operations the epilogue applies in order. Their
success values are the exact integer products/differences, and their refusals are exactly
`Overflow`; `packing_mul2_affine` composes two multiplies into the no-min/no-bias affine law
`affineWeight scale a q 0 0 0 false`. -/

theorem packing_mul_ok (a b : I64)
    (hlo : I64.min ≤ a.val * b.val) (hhi : a.val * b.val ≤ I64.max) :
    ∃ z, packing.mul a b = ok (core.result.Result.Ok z) ∧ z.val = a.val * b.val := by
  obtain ⟨z, hz, hzv⟩ := I64_checked_mul_some a b hlo hhi
  refine ⟨z, ?_, hzv⟩
  unfold packing.mul
  rw [hz]
  simp only [lift, bind_tc_ok]
  rw [core.option.Option.ok_or_some]

theorem packing_mul_overflow (a b : I64) (h : I64.checked_mul a b = none) :
    packing.mul a b = ok (core.result.Result.Err packing.PackingError.Overflow) := by
  unfold packing.mul
  rw [h]
  simp only [lift, bind_tc_ok]
  rw [core.option.Option.ok_or_none]

theorem packing_sub_ok (a b : I64)
    (hlo : I64.min ≤ a.val - b.val) (hhi : a.val - b.val ≤ I64.max) :
    ∃ z, packing.sub a b = ok (core.result.Result.Ok z) ∧ z.val = a.val - b.val := by
  obtain ⟨z, hz, hzv⟩ := I64_checked_sub_some a b hlo hhi
  refine ⟨z, ?_, hzv⟩
  unfold packing.sub
  rw [hz]
  simp only [lift, bind_tc_ok]
  rw [core.option.Option.ok_or_some]

theorem packing_sub_overflow (a b : I64) (h : I64.checked_sub a b = none) :
    packing.sub a b = ok (core.result.Result.Err packing.PackingError.Overflow) := by
  unfold packing.sub
  rw [h]
  simp only [lift, bind_tc_ok]
  rw [core.option.Option.ok_or_none]

/-- Two checked multiplies in epilogue order: `(scale·a)·q` on the success path. -/
theorem packing_mul2_ok (scale a q : I64)
    (h1lo : I64.min ≤ scale.val * a.val) (h1hi : scale.val * a.val ≤ I64.max)
    (h2lo : I64.min ≤ (scale.val * a.val) * q.val)
    (h2hi : (scale.val * a.val) * q.val ≤ I64.max) :
    ∃ sa z, packing.mul scale a = ok (core.result.Result.Ok sa) ∧
      packing.mul sa q = ok (core.result.Result.Ok z) ∧
      z.val = (scale.val * a.val) * q.val := by
  obtain ⟨sa, hsa, hsav⟩ := packing_mul_ok scale a h1lo h1hi
  obtain ⟨z, hz, hzv⟩ := packing_mul_ok sa q (by rw [hsav]; exact h2lo)
    (by rw [hsav]; exact h2hi)
  exact ⟨sa, z, hsa, hz, by rw [hzv, hsav]⟩

/-- **Exactness of the affine fold** on the no-min/no-bias entry (`has_min = false`, `bias = 0`,
covering Q8_0): the extracted two-multiply composition realizes `affineWeight`. -/
theorem packing_mul2_affine (scale a q : I64)
    (h1lo : I64.min ≤ scale.val * a.val) (h1hi : scale.val * a.val ≤ I64.max)
    (h2lo : I64.min ≤ (scale.val * a.val) * q.val)
    (h2hi : (scale.val * a.val) * q.val ≤ I64.max) :
    ∃ sa z, packing.mul scale a = ok (core.result.Result.Ok sa) ∧
      packing.mul sa q = ok (core.result.Result.Ok z) ∧
      z.val = affineWeight scale.val a.val q.val 0 0 0 false := by
  obtain ⟨sa, z, hsa, hz, hzv⟩ := packing_mul2_ok scale a q h1lo h1hi h2lo h2hi
  refine ⟨sa, z, hsa, hz, ?_⟩
  rw [hzv]
  unfold affineWeight
  simp

/-! ## The canonical affine value law

`blockWeight` is the reference value of weight `j` of a decoded block: the checked-arithmetic
epilogue applied to that weight's coefficient pair and code. `epilogueInv` states the output vector
is exactly the `List.range`-map of the reference; the per-element step is the checked `i64`
arithmetic proved above. -/

/-- The canonical reference value of weight `j` of a decoded block, using the same `List.getD`
defaults the reference lists use. -/
def blockWeight (scale minScale : ℤ) (bias : ℤ) (hasMin : Bool)
    (b : packing.CanonicalBlock) (j : Nat) : ℤ :=
  affineWeight scale
    ((b.coeffs.val.getD j (0#i32, 0#i32)).1.val)
    ((b.codes.val.getD j 0#i64).val)
    bias minScale ((b.coeffs.val.getD j (0#i32, 0#i32)).2.val) hasMin

/-- The loop invariant: the values of the output vector are the reference prefix of length `i`. -/
def epilogueInv (scale minScale : ℤ) (bias : ℤ) (hasMin : Bool)
    (b : packing.CanonicalBlock) (block : Nat)
    (out : alloc.vec.Vec Std.I64) (i : Nat) : Prop :=
  i ≤ block ∧ out.val.map (fun x => x.val) =
    (List.range i).map (fun j => blockWeight scale minScale bias hasMin b j)

lemma getD_eq_getElem {α : Type} (l : List α) (i : Nat) (d : α) (h : i < l.length) :
    l.getD i d = l[i] := by
  simp only [List.getD, List.getElem?_eq_getElem h]
  rfl

theorem I64_checked_add_some (x y : I64)
    (hlo : I64.min ≤ x.val + y.val) (hhi : x.val + y.val ≤ I64.max) :
    ∃ z, x.checked_add y = some z ∧ z.val = x.val + y.val := by
  have hspec := I64.checked_add_bv_spec x y
  cases hc : core.num.checked_add_IScalar x y with
  | none =>
      rw [hc] at hspec
      exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      exact ⟨z, by simpa only [I64.checked_add] using hc, hspec.2.2.1⟩

/-- **Per-element value law, `Fixed`, no min.** On the success domain (`has_min = false`), the
extracted `epilogue_entry` returns the checked-`i64` value of `affineWeight` at that weight: the
two `packing.mul`s realize `(scale·a)·(q + bias)`. The cast values are the widening `I32 → I64`
casts, and the range hypotheses are exactly what makes the checked operations succeed. -/
theorem epilogue_entry_fixed_nomin
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
    (hd : I64.min ≤ (scale.val
          * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
        * ((b.codes.val.getD i.val 0#i64).val + bias.val))
    (hd' : (scale.val * (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val)
        * ((b.codes.val.getD i.val 0#i64).val + bias.val) ≤ I64.max) :
    packing.Packing.epilogue_entry
      { width := packing.CodeWidth.Fixed w, high_planes := hp, block := blk, group := grp,
        scale_kind := sk, has_min := false, signed_code := sc, bias := bias,
        signed_coeff := sco, plane_order := po, nibble_order := no }
      b i scale minScale group ⦃ r =>
        ∃ z, r = core.result.Result.Ok z ∧
          z.val = affineWeight scale.val
            (b.coeffs.val.getD (i.val / group.val) (0#i32, 0#i32)).1.val
            (b.codes.val.getD i.val 0#i64).val bias.val 0 0 false ⦄ := by
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
  simp only [if_neg (by decide : ¬ ((false : Bool) = true))]
  rw [WP.spec_ok]
  refine ⟨z, rfl, ?_⟩
  rw [hzv, hsav, hcaddv, hi2d, hi3v, ← hi4, hcasta, hav, hi1v]
  unfold affineWeight
  simp

/-! ## Residual obligation

Two pieces of the full decoder identity remain open and are **not** claimed by this file; they are
the exact residual named in the header:

1. the byte-level `decode_block_fields` structural identity — the generic `unpack_code` bit-plane
   unpack producing `b.codes`. It does not call `word_q4_k_codes`/`word_q6_k_codes` (those are the
   `import_row` accessors), so the existing nibble Q4_K/Q6_K equalities do not compose directly. The
   per-code law is `unpack_code_ref` (`Numeric/Packing/Unpack.lean`); its composition over the packed
   code area is open.
2. the top-level `epilogue` assembly and the non-`Fixed` widths. The `Fixed`-width loop folds are
   proved in `Numeric/Packing/EpilogueFold.lean` (`epilogue_loop2_fixed_spec`,
   `epilogue_loop1_fixed_spec`, with the `has_min` entry law `epilogue_entry_fixed_min`).

The shape below records only the *totality* on the success path (fields return a block and the
decoder returns a vector). It is deliberately not the value law; the value law is items 1 and 2. -/

/-- Shape of the residual success path: a long-enough payload yields a canonical block and a decoded
vector (no overflow). Not the value law; see the section comment. -/
def packing_decode_shape : Prop :=
  ∀ (p : packing.Packing) (raw : Slice Std.U8),
    (∃ bb, packing.Packing.block_bytes p = ok (core.result.Result.Ok bb) ∧
      bb ≤ Slice.len raw) →
    ∃ (b : packing.CanonicalBlock) (v : alloc.vec.Vec Std.I64),
      packing.Packing.decode_block_fields p raw = ok (core.result.Result.Ok b) ∧
      word_packing_decode_block p raw = ok (core.result.Result.Ok v)

end apeiron_word
