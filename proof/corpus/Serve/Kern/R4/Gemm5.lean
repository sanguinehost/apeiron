/-
Integer, index and layout refinement of the extracted `kern::gemm` quad packer, fifth group.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::pack_group`. This module
refines the portable (non-AVX2, non-`unsafe`) four-code quad packer that `Serve/Kern/Gemm4.lean`
left open:

* `pack_group_loop0_loop1_loop0` (the four-code quad packer): for each `quad < 8` it reads the four
  `i8` codes at `b*ACT_QK + 4*quad`, packs them little-endian into a `u32`, xors the sign mask
  `0x80808080`, and writes the word at `(b*8 + quad)*8 + lane`.

The reference is plain Lean over `Nat`/`List`: a fold of `Slice.set` writes and an index arithmetic.
It states WHAT the body computes, not how the loop runs. The quad packer refuses only through
`Slice.index_usize`/`Slice.update` out of bounds and the checked `usize` multiplications; the premise
bundle excludes both.

NOT refined here, with the reason stated rather than skirted:

* the enclosing loops `pack_group_loop0_loop1` (block loop) and `pack_group_loop0` (column loop):
  `pack_group_loop0_loop1`'s body reads `scales[b]`, takes `I64.trailing_zeros`, converts through
  `I64.Insts.CoreConvertFromU32.from` and folds with `core.cmp.Ord.max.default`, and
  `pack_group_loop0`'s body calls the `chart.QuantAct` accessors (`chart.QuantAct.impl.scales`,
  `chart.QuantAct.impl.codes`, `Slice.len scales`, `core.cmp.Ord.min.trait_default`). Those step
  rules come from the `chart` group and its `i64`/`ord` companions; the goals stay open rather than
  weakened.
* `column_ptrs` / `column_ptrs_loop`: their extracted bodies dereference
  `core.slice.Slice.as_mut_ptr` and `alloc.vec.Vec.spare_capacity_mut`, both emitted as
  `axiom : Result _`, so no `wp` spec and no raw pointer value are derivable.
* `fast_columns_ok` / `fast_columns_ok_loop`: their bodies call `chart.QuantAct.len`,
  `chart.QuantAct.impl.scales`, `chart.QuantAct.magnitude` and `chart.q8act_fast_bound_ok`. Those
  accessors are refined elsewhere in the tree, but `kern.gemm.fast_columns_ok` is its own body (not
  the `chart`/`kern.matmat` one), so the `step` rules that walk it come from that group; the goal
  stays open rather than weakened.
* the tile value itself: every output word of the AVX2 path is written by `kern.gemm.x86.rows`,
  emitted by the extraction as `axiom`; `pack_group` also reaches the `axiom` pool driver
  `chart.pool.chunks.for_chunks4`, so `pack_group`'s own composition has no `wp` spec and is not
  refined here.

Trust boundary: rustc/LLVM codegen, the `F32`/AVX2 layers, the `axiom` pool driver and the
`R[x]/(x²−σ)` word kernels are not covered here. No `sorry`/`admit`/`axiom`/`native_decide`/
`bv_decide`. `decide` is not used on an extracted body or on a whole source; it appears only on
literal lists and one literal integer bound.
-/
import Serve.Kern.Gemm3

set_option autoImplicit false
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## References (plain Lean) -/

/-- The byte a code lane contributes to a packed word: `codes[i] as u8`, i.e. `IScalar.hcast .U8`
of `codes[i]` followed by the `u8 -> u32` widening. -/
def gemm5_byte (codes : List Std.I8) (off : Nat) : Nat :=
  ((codes.getD off 0#i8).val % 256).toNat

/-- The little-endian 32-bit word of the four code bytes at `off .. off + 3`. -/
def gemm5_le32 (codes : List Std.I8) (off : Nat) : Nat :=
  gemm5_byte codes off + gemm5_byte codes (off + 1) * 256 +
    gemm5_byte codes (off + 2) * 65536 + gemm5_byte codes (off + 3) * 16777216

/-- The packed word the quad packer writes: the little-endian code word xored with the sign mask
`0x80808080`. -/
def gemm5_quadWord (codes : List Std.I8) (off : Nat) : Nat :=
  gemm5_le32 codes off ^^^ 2155905152

/-- The words after the quad loop has written the first `q` quads of block `b` at lane `lane`:
`q` `Slice.set` writes at `(b*8 + k)*8 + lane`, each to the packed word of the codes
`b*32 + 4k .. b*32 + 4k + 3`. -/
def gemm5_packQuads (w0 : List Std.U32) (lane : Nat) (codes : List Std.I8) (b q : Nat) :
    List Std.U32 :=
  (List.range q).foldl (fun w k =>
    w.set ((b * 8 + k) * 8 + lane)
      (⟨BitVec.ofNat _ (gemm5_quadWord codes (b * 32 + 4 * k))⟩ : Std.U32)) w0

/-! ## Value lemmas -/

/-- `kern.gemm.ACT_QK` is 32, the quad stride. -/
private lemma gemm5_ACT_QK_val : (kern.gemm.ACT_QK : Std.Usize).val = 32 := by
  unfold kern.gemm.ACT_QK
  rfl

/-- `kern.gemm.LANES` is 8, the lane stride. -/
private lemma gemm5_LANES_val : (kern.gemm.LANES : Std.Usize).val = 8 := by
  unfold kern.gemm.LANES
  rfl

/-- The `u32` literal `⟨BitVec.ofNat _ n⟩` has value `n` below `2^32`. -/
private lemma gemm5_u32_val_ofNat {n : Nat} (h : n < 2 ^ 32) :
    (⟨BitVec.ofNat _ n⟩ : Std.U32).val = n := by
  show (BitVec.ofNat _ n).toNat = n
  rw [BitVec.toNat_ofNat, UScalarTy.U32_numBits_eq]
  exact Nat.mod_eq_of_lt h

/-- In-bounds `List.getElem` of an `I8` list is its `getD`. -/
private lemma gemm5_getD_val (l : List Std.I8) (i : Nat) (h : i < l.length) :
    (l.getD i 0#i8).val = (l[i]'h).val := by
  rw [List.getD, List.getElem?_eq_getElem h]
  rfl

/-- The reference byte at an in-bounds index, in `hcast`/widening form. -/
private lemma gemm5_byte_of_val (codes : List Std.I8) (i : Nat) (h : i < codes.length) :
    (((codes[i]'h).val % 256).toNat) = gemm5_byte codes i := by
  unfold gemm5_byte
  rw [gemm5_getD_val codes i h]

/-- Each reference byte is below `256`. -/
private lemma gemm5_byte_lt (codes : List Std.I8) (i : Nat) : gemm5_byte codes i < 256 := by
  unfold gemm5_byte
  exact (Int.toNat_lt_of_ne_zero (m := (codes.getD i 0#i8).val % 256) (n := 256)
    (by norm_num)).mpr (Int.emod_lt_of_pos _ (by norm_num))

/-- The `i8 -> u8` cast followed by the `u8 -> u32` widening, as a `step` rule. -/
@[step]
theorem gemm5_fromU32U8_spec (x : Std.U8) :
    lift (core.convert.num.FromU32U8.from x) ⦃ y => y.val = x.val ⦄ := by
  simp only [lift, WP.spec_ok, core.convert.num.FromU32U8.from_val_eq]

/-- The composed `hcast .U8` then `FromU32U8.from` value. -/
private lemma gemm5_hcast_from_val (x : Std.I8) :
    (core.convert.num.FromU32U8.from (IScalar.hcast UScalarTy.U8 x)).val =
      ((x.val % 256).toNat) := by
  rw [core.convert.num.FromU32U8.from_val_eq, IScalar.hcast_val_eq, UScalarTy.U8_numBits_eq]
  norm_num

/-- A `u32` shift of a byte by `k ≤ 24` does not wrap modulo `2^32`. -/
private lemma gemm5_shl_mod (b k : Nat) (hb : b < 2 ^ 8) (hk : k ≤ 24) :
    (b <<< k) % (2 : Nat) ^ 32 = b <<< k := by
  apply Nat.mod_eq_of_lt
  rw [Nat.shiftLeft_eq]
  calc b * 2 ^ k < 2 ^ 8 * 2 ^ k := (Nat.mul_lt_mul_right (Nat.two_pow_pos k)).mpr hb
    _ = 2 ^ (8 + k) := by rw [← Nat.pow_add]
    _ ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) (by omega)

/-- OR with a fresh high part `w·2^k` is addition when `a < 2^k`. -/
private lemma gemm5_or_add_of_lt {a k w : Nat} (ha : a < 2 ^ k) :
    a ||| (w * 2 ^ k) = a + w * 2 ^ k := by
  have h := Nat.two_pow_add_eq_or_of_lt ha w
  rw [Nat.mul_comm (2 ^ k) w, Nat.or_comm, Nat.add_comm] at h
  exact h.symm

/-- Two bytes occupy 16 bits. -/
private lemma gemm5_lt16 {a b : Nat} (ha : a < 2 ^ 8) (hb : b < 2 ^ 8) :
    a + b * 2 ^ 8 < 2 ^ 16 := by
  have h8 : (2 : Nat) ^ 8 = 256 := by norm_num
  have h16 : (2 : Nat) ^ 16 = 65536 := by norm_num
  rw [h8] at ha hb
  rw [h16, h8]
  omega

/-- Three bytes occupy 24 bits. -/
private lemma gemm5_lt24 {a b c : Nat} (ha : a < 2 ^ 8) (hb : b < 2 ^ 8) (hc : c < 2 ^ 8) :
    a + b * 2 ^ 8 + c * 2 ^ 16 < 2 ^ 24 := by
  have h8 : (2 : Nat) ^ 8 = 256 := by norm_num
  have h16 : (2 : Nat) ^ 16 = 65536 := by norm_num
  have h24 : (2 : Nat) ^ 24 = 16777216 := by norm_num
  rw [h8] at ha hb hc
  rw [h24, h8, h16]
  omega

/-- The extracted four-byte OR chain, on the four byte values. -/
private lemma gemm5_chain (b0 b1 b2 b3 : Nat)
    (h0 : b0 < 2 ^ 8) (h1 : b1 < 2 ^ 8) (h2 : b2 < 2 ^ 8) (h3 : b3 < 2 ^ 8) :
    (((b0 ||| (b1 <<< 8) % (2 : Nat) ^ 32) ||| (b2 <<< 16) % (2 : Nat) ^ 32) |||
        (b3 <<< 24) % (2 : Nat) ^ 32) =
      b0 + b1 * 256 + b2 * 65536 + b3 * 16777216 := by
  rw [gemm5_shl_mod b1 8 h1 (by norm_num), gemm5_shl_mod b2 16 h2 (by norm_num),
    gemm5_shl_mod b3 24 h3 (by norm_num)]
  have e1 : b0 ||| (b1 <<< 8) = b0 + b1 * 2 ^ 8 := by
    rw [Nat.shiftLeft_eq]
    exact gemm5_or_add_of_lt (k := 8) (w := b1) (a := b0) h0
  have e2 : (b0 + b1 * 2 ^ 8) ||| (b2 <<< 16) = b0 + b1 * 2 ^ 8 + b2 * 2 ^ 16 := by
    rw [Nat.shiftLeft_eq]
    exact gemm5_or_add_of_lt (k := 16) (w := b2) (a := b0 + b1 * 2 ^ 8) (gemm5_lt16 h0 h1)
  have e3 : (b0 + b1 * 2 ^ 8 + b2 * 2 ^ 16) ||| (b3 <<< 24) =
      b0 + b1 * 2 ^ 8 + b2 * 2 ^ 16 + b3 * 2 ^ 24 := by
    rw [Nat.shiftLeft_eq]
    exact gemm5_or_add_of_lt (k := 24) (w := b3) (a := b0 + b1 * 2 ^ 8 + b2 * 2 ^ 16)
      (gemm5_lt24 h0 h1 h2)
  have h8 : (2 : Nat) ^ 8 = 256 := by norm_num
  have h16 : (2 : Nat) ^ 16 = 65536 := by norm_num
  have h24 : (2 : Nat) ^ 24 = 16777216 := by norm_num
  rw [e1, e2, e3, h8, h16, h24]

/-- The same chain with the Aeneas `U32.size` in place of `2^32`. -/
private lemma gemm5_chainU (b0 b1 b2 b3 : Nat)
    (h0 : b0 < 2 ^ 8) (h1 : b1 < 2 ^ 8) (h2 : b2 < 2 ^ 8) (h3 : b3 < 2 ^ 8) :
    (((b0 ||| (b1 <<< 8) % U32.size) ||| (b2 <<< 16) % U32.size) |||
        (b3 <<< 24) % U32.size) =
      b0 + b1 * 256 + b2 * 65536 + b3 * 16777216 := by
  have hs : U32.size = (2 : Nat) ^ 32 := by rw [U32.size_eq]; norm_num
  rw [hs]
  exact gemm5_chain b0 b1 b2 b3 h0 h1 h2 h3

/-- The little-endian reference word is below `2^32`. -/
private lemma gemm5_le32_lt (codes : List Std.I8) (off : Nat) : gemm5_le32 codes off < 2 ^ 32 := by
  unfold gemm5_le32
  have h0 := gemm5_byte_lt codes off
  have h1 := gemm5_byte_lt codes (off + 1)
  have h2 := gemm5_byte_lt codes (off + 2)
  have h3 := gemm5_byte_lt codes (off + 3)
  omega

/-- The packed quad word is below `2^32`, so the `u32` literal does not truncate. -/
private lemma gemm5_quadWord_lt (codes : List Std.I8) (off : Nat) :
    gemm5_quadWord codes off < 2 ^ 32 := by
  unfold gemm5_quadWord
  exact Nat.xor_lt_two_pow (gemm5_le32_lt codes off) (by norm_num)

/-- `gemm5_packQuads` one step. -/
private lemma gemm5_packQuads_succ (w0 : List Std.U32) (lane : Nat) (codes : List Std.I8)
    (b q : Nat) :
    gemm5_packQuads w0 lane codes b (q + 1) =
      (gemm5_packQuads w0 lane codes b q).set ((b * 8 + q) * 8 + lane)
        (⟨BitVec.ofNat _ (gemm5_quadWord codes (b * 32 + 4 * q))⟩ : Std.U32) := by
  unfold gemm5_packQuads
  rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]

/-! ## `pack_group_loop0_loop1_loop0`: the four-code quad packer -/

/-- The four-code quad packer writes the eight packed words of block `b` at lane `lane`.

The loop's measure is `8 - quad.val`; the invariant is the running fold `gemm5_packQuads` over the
words plus the length preservation that keeps the `Slice.update` index in bounds. -/
@[step]
theorem word_pack_group_loop0_loop1_loop0_ok (lane : Std.Usize) (codes : Slice Std.I8)
    (b : Std.Usize) (words0 : Slice Std.U32)
    (hno : (b.val * 8 + 8) * 8 + lane.val ≤ Usize.max)
    (hcode : ∀ (q : Nat), q < 8 → b.val * 32 + 4 * q + 3 < codes.length)
    (hbound : ∀ (q : Nat), q < 8 → (b.val * 8 + q) * 8 + lane.val < words0.length) :
    kern.gemm.pack_group_loop0_loop1_loop0 words0 lane codes b 0#usize ⦃ r =>
      r.val = gemm5_packQuads words0.val lane.val codes.val b.val 8 ⦄ := by
  unfold kern.gemm.pack_group_loop0_loop1_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Slice Std.U32 × Std.Usize => 8 - st.2.val)
    (inv := fun st : Slice Std.U32 × Std.Usize =>
      st.2.val ≤ 8 ∧ st.1.length = words0.length ∧
      st.1.val = gemm5_packQuads words0.val lane.val codes.val b.val st.2.val)
    (post := fun r : Slice Std.U32 =>
      r.val = gemm5_packQuads words0.val lane.val codes.val b.val 8)
  · rintro ⟨words, quad⟩ ⟨hq8, hwlen, hinvw⟩
    dsimp only at *
    unfold kern.gemm.pack_group_loop0_loop1_loop0.body
    by_cases hlt : quad < 8#usize
    · rw [if_pos hlt]
      have hq8' : quad.val < 8 := by simpa [UScalar.lt_equiv] using hlt
      simp only [lift, bind_tc_ok]
      step*
      · rw [gemm5_ACT_QK_val]; omega
      · rw [gemm5_ACT_QK_val] at i_post; rw [i_post, i1_post]; omega
      · rw [at_post, i_post, i1_post, gemm5_ACT_QK_val]
        have h := hcode quad.val hq8'; omega
      · rw [at_post, i_post, i1_post, gemm5_ACT_QK_val]
        simp only [UScalar.ofNatCore_val_eq]
        have h := hno; omega
      · rw [i5_post, at_post, i_post, i1_post, gemm5_ACT_QK_val]
        have h := hcode quad.val hq8'; omega
      · rw [at_post, i_post, i1_post, gemm5_ACT_QK_val]
        simp only [UScalar.ofNatCore_val_eq]
        have h := hno; omega
      · rw [i11_post, at_post, i_post, i1_post, gemm5_ACT_QK_val]
        have h := hcode quad.val hq8'; omega
      · rw [at_post, i_post, i1_post, gemm5_ACT_QK_val]
        simp only [UScalar.ofNatCore_val_eq]
        have h := hno; omega
      · rw [i17_post, at_post, i_post, i1_post, gemm5_ACT_QK_val]
        have h := hcode quad.val hq8'; omega
      · rw [i23_post, i22_post, gemm5_LANES_val]; have h := hno; omega
      · rw [i24_post, i23_post, i22_post, gemm5_LANES_val]; have h := hno; omega
      · rw [i25_post, i24_post, i23_post, i22_post, gemm5_LANES_val, hwlen]
        exact hbound quad.val hq8'
      · refine ⟨?_, ?_, ?_, ?_⟩
        · rw [quad1_post]; omega
        · rw [s_post, Slice.set_length]
          exact hwlen
        · have hat : (↑«at» : Nat) = b.val * 32 + 4 * quad.val := by
            rw [at_post, i_post, i1_post, gemm5_ACT_QK_val]
          have hbnd_at : (↑«at» : Nat) < codes.length := by
            rw [hat]; have h := hcode quad.val hq8'; omega
          have hbnd5 : (↑i5 : Nat) < codes.length := by
            rw [i5_post, hat]; have h := hcode quad.val hq8'; omega
          have hbnd11 : (↑i11 : Nat) < codes.length := by
            rw [i11_post, hat]; have h := hcode quad.val hq8'; omega
          have hbnd17 : (↑i17 : Nat) < codes.length := by
            rw [i17_post, hat]; have h := hcode quad.val hq8'; omega
          have hA0 : (core.convert.num.FromU32U8.from (IScalar.hcast UScalarTy.U8 i2)).val
              = gemm5_byte codes.val (b.val * 32 + 4 * quad.val) := by
            rw [gemm5_hcast_from_val, ← hat, i2_post]
            exact gemm5_byte_of_val codes.val ↑«at» hbnd_at
          have hA1 : (core.convert.num.FromU32U8.from (IScalar.hcast UScalarTy.U8 i6)).val
              = gemm5_byte codes.val (b.val * 32 + 4 * quad.val + 1) := by
            rw [gemm5_hcast_from_val]
            have h6 : ((i6.val % 256).toNat) = gemm5_byte codes.val ↑i5 := by
              rw [i6_post]; exact gemm5_byte_of_val codes.val ↑i5 hbnd5
            rw [h6, i5_post, hat]
          have hA2 : (core.convert.num.FromU32U8.from (IScalar.hcast UScalarTy.U8 i12)).val
              = gemm5_byte codes.val (b.val * 32 + 4 * quad.val + 2) := by
            rw [gemm5_hcast_from_val]
            have h12 : ((i12.val % 256).toNat) = gemm5_byte codes.val ↑i11 := by
              rw [i12_post]; exact gemm5_byte_of_val codes.val ↑i11 hbnd11
            rw [h12, i11_post, hat]
          have hA3 : (core.convert.num.FromU32U8.from (IScalar.hcast UScalarTy.U8 i18)).val
              = gemm5_byte codes.val (b.val * 32 + 4 * quad.val + 3) := by
            rw [gemm5_hcast_from_val]
            have h18 : ((i18.val % 256).toNat) = gemm5_byte codes.val ↑i17 := by
              rw [i18_post]; exact gemm5_byte_of_val codes.val ↑i17 hbnd17
            rw [h18, i17_post, hat]
          have hEl : ((core.convert.num.FromU32U8.from (IScalar.hcast UScalarTy.U8 i2) ||| i9 |||
                i15 ||| i21) ^^^ 2155905152#u32)
              = (⟨BitVec.ofNat _ (gemm5_quadWord codes.val (b.val * 32 + 4 * quad.val))⟩ :
                  Std.U32) := by
            apply UScalar.eq_of_val_eq
            rw [gemm5_u32_val_ofNat (gemm5_quadWord_lt codes.val (b.val * 32 + 4 * quad.val))]
            simp only [UScalar.val_xor, UScalar.val_or, UScalar.ofNatCore_val_eq]
            rw [i9_post, i15_post, i21_post, hA0, hA1, hA2, hA3]
            rw [gemm5_chainU (gemm5_byte codes.val (b.val * 32 + 4 * quad.val))
              (gemm5_byte codes.val (b.val * 32 + 4 * quad.val + 1))
              (gemm5_byte codes.val (b.val * 32 + 4 * quad.val + 2))
              (gemm5_byte codes.val (b.val * 32 + 4 * quad.val + 3))
              (gemm5_byte_lt codes.val (b.val * 32 + 4 * quad.val))
              (gemm5_byte_lt codes.val (b.val * 32 + 4 * quad.val + 1))
              (gemm5_byte_lt codes.val (b.val * 32 + 4 * quad.val + 2))
              (gemm5_byte_lt codes.val (b.val * 32 + 4 * quad.val + 3))]
            unfold gemm5_quadWord gemm5_le32
            rfl
          have hidx : (↑i25 : Nat) = (↑b * 8 + ↑quad) * 8 + ↑lane := by
            rw [i25_post, i24_post, i23_post, i22_post, gemm5_LANES_val]
          rw [s_post, Slice.set_val_eq, hidx, quad1_post, gemm5_packQuads_succ, ← hinvw, hEl]
        · rw [quad1_post]; omega
    · rw [if_neg hlt]
      have hge : 8 ≤ quad.val := by
        have h : ¬ (quad.val < 8) := by simpa [UScalar.lt_equiv] using hlt
        omega
      have hqk : quad.val = 8 := le_antisymm hq8 hge
      simp only [WP.spec_ok]
      simpa [hqk] using hinvw
  · refine ⟨by simp, rfl, ?_⟩
    simp [gemm5_packQuads]

/-! ## Non-vacuity -/

/-- Non-vacuity of the byte reference: the `% 256` mask maps a negative code to its low byte. -/
example : gemm5_byte ([Std.I8.ofInt (-1), 0#i8, 0#i8, 0#i8] : List Std.I8) 0 = 255 := by decide

/-- Non-vacuity: the little-endian word of four literal codes. -/
example : gemm5_le32 ([1#i8, 2#i8, 3#i8, 4#i8] : List Std.I8) 0
    = 1 + 2 * 256 + 3 * 65536 + 4 * 16777216 := by decide

/-- Non-vacuity: the sign-mask xor of the packed word. -/
example : gemm5_quadWord ([0#i8, 0#i8, 0#i8, 0#i8] : List Std.I8) 0 = 2155905152 := by decide

/-- Non-vacuity: the packer writes the packed quad word at the addressed lane, leaving the rest. -/
example : gemm5_packQuads ([0#u32, 0#u32] : List Std.U32) 1 ([0#i8, 1#i8] : List Std.I8) 0 1
    = [0#u32, ⟨BitVec.ofNat _ (gemm5_quadWord ([0#i8, 1#i8] : List Std.I8) 0)⟩] := by
  simp [gemm5_packQuads]

/-- Non-vacuity of the hypothesis bundle: `codes` of length 32, `words0` of length 64 satisfy
`hno`, `hcode` and `hbound` jointly at `b = 0`, `lane = 0`. -/
example : ∃ (words0 : Slice Std.U32) (codes : Slice Std.I8),
    (64 : Nat) ≤ Std.Usize.max ∧
    (∀ (q : Nat), q < 8 → 4 * q + 3 < codes.length) ∧
    (∀ (q : Nat), q < 8 → q * 8 < words0.length) := by
  refine ⟨Slice.from (List.replicate 64 0#u32) (by simp only [List.length_replicate]; scalar_tac),
    Slice.from (List.replicate 32 0#i8) (by simp only [List.length_replicate]; scalar_tac),
    ?_, ?_, ?_⟩
  · simpa using Slice.property (Slice.from (List.replicate 64 0#u32)
      (by simp only [List.length_replicate]; scalar_tac))
  · intro q hq
    simp only [Slice.length, Slice.from_val, List.length_replicate]
    omega
  · intro q hq
    simp only [Slice.length, Slice.from_val, List.length_replicate]
    omega

/-! ## Axiom cone -/

end apeiron_word
