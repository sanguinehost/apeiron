/-
Refinement of the extracted `Q2_K` dequantizer (`APEIRON-ALGEBRA-G3-DEQUANT-W3-T3`).

The Rust bodies are `dequant::q2_k_block` and `dequant::dequant_q2_k_row` in
`src/engine/dequant.rs`; the block accessors `quant::q2_k_d`, `quant::q2_k_dmin`,
`quant::q2_k_scales`, `quant::q2_k_scale_min` and `quant::q2_k_codes` are in
`src/format/quant.rs`. The decode is exact integer arithmetic over Q32.32: `d` and `dmin` are
finite `f16` values decoded exactly by `f16_to_fixed_exact`, the 16 packed `(scale, min)` bytes
are split by `q2_k_scale_min` into low/high nibbles, the 256 two-bit codes are unpacked from the
64 `qs` bytes, and each weight is `d·(sc & 0xF)·q − dmin·(sc >> 4)` with checked `i64` multiply
and subtract, so the law is an `ℤ` equality, not an error bound.

The block layout is `scales[16] (0..16); qs[64] (16..80); f16 d (80); f16 dmin (82)`, 84 bytes.

## Proved

* `quant_Q2_K_BLOCK_BYTES_eq`: the extracted block length evaluates to 84.
* `q2_k_scale_min_ref`: the extracted `quant.q2_k_scale_min` splits a byte into its low nibble
  (`&&& 15`) and high nibble (`>>> 4`).
* `q2_k_scales_ok`: on a whole block, `q2_k_scales` copies the first sixteen bytes exactly.
* `q2_k_d_ok`, `q2_k_dmin_ok`: the two `f16` fields are read at bytes 80 and 82.
* `q2_codes_loop_spec` (`q2Lane`/`q2Plane`/`q2Round`/`q2Half`): the four nested `q2_k_codes`
  loops unpack `q2QsRef`.
* `q2_k_codes_ok`: on a whole block, `q2_k_codes` returns the 256 codes pointwise `q2CodesRef`.
* `q2_k_block_spec`: on a whole 84-byte block with finite `d`/`dmin`, `q2_k_block` returns
  `(d, dmin, sc, codes)` with `d`/`dmin` the exact Q32.32 `f16` values, `sc` the sixteen scale
  bytes and `codes` pointwise `q2CodesRef`.
* `dequant_row_fixed_q2_k`: `dequant_row_fixed` at `ggml_type = 10` (`GGML_TYPE_Q2_K`) is
  `dequant_q2_k_row`.

## Residual

The Q2_K row-level refusals (short payload, non-finite scale, overflowing product) are not
characterized here: `q2kBlockDomain` is a sufficient success hypothesis. The lane/group/row laws
are stated in the following section.

## Trust boundary

Kernel-checked against the Aeneas model of `i64::checked_mul`/`checked_sub`, `u16::from_le_bytes`,
slice range `get`, `Array` index/update and `Vec::push` (`target/aeneas-model/lean`);
rustc/LLVM codegen and the GGUF file reader upstream of the row bytes are outside this module.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.Dequant.K.Q6K
import Numeric.Dequant.Refinement
import Numeric.Packing.Format

namespace apeiron_word
open Aeneas Aeneas.Std Result

/-! ## Block length -/

/-- The Q2_K block length evaluates to 84 bytes. -/
theorem quant_Q2_K_BLOCK_BYTES_eq : quant.Q2_K_BLOCK_BYTES = ok (84#usize) := by
  have h : quant.Q2_K_BLOCK_BYTES ⦃ z => z = (84#usize) ⦄ := by
    unfold quant.Q2_K_BLOCK_BYTES
    simp only [quant.QK_K]
    step*
  rcases (Aeneas.Std.WP.spec_equiv_exists _ _).1 h with ⟨y, hy, hyv⟩
  rw [hyv] at hy
  exact hy

/-! ## Q2_K references -/

/-- The `g`-th packed `(scale, min)` byte of a block, read at byte `g`. -/
def q2kScalesRef (blk : List Std.U8) (g : Nat) : Nat := (blk.getD g 0#u8).val

/-- The finite `f16` scale `d` of block `b`, read at byte `84·b + 80`. -/
def q2kD (raw : Slice Std.U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (84 * b + 80))

/-- The finite `f16` minimum scale `dmin` of block `b`, read at byte `84·b + 82`. -/
def q2kDmin (raw : Slice Std.U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (84 * b + 82))

/-- Q2_K reference code `i` on the post-offset `qs` bytes: the two-bit field
`(qs[32·(i/128) + 16·((i%32)/16) + i%16] >>> 2·((i%128)/32)) &&& 3`. -/
def q2QsRef (qs : List Std.U8) (i : Nat) : Nat :=
  (qs.getD (32 * (i / 128) + 16 * ((i % 32) / 16) + i % 16) 0#u8).val
    >>> (2 * ((i % 128) / 32)) &&& 3

/-- Q2_K two-bit reference code `i` of a whole block: `q2QsRef` of the `qs` window (byte 16). -/
def q2CodesRef (blk : List Std.U8) (i : Nat) : Nat := q2QsRef (blk.drop 16) i

/-- The scale/min byte list of a block: bytes `0..16`. -/
def q2kScList (raw : Slice Std.U8) (b : Nat) : List Std.U8 :=
  (raw.val.drop (84 * b)).take 16

/-- The block's 84 raw bytes, read from the row at byte `84·b`. -/
def q2kBlockBytes (raw : Slice Std.U8) (b : Nat) : List Std.U8 :=
  (raw.val.drop (84 * b)).take 84

/-- `q2CodesRef` is below four. -/
lemma q2CodesRef_lt_4 (blk : List Std.U8) (i : Nat) : q2CodesRef blk i < 4 := by
  unfold q2CodesRef q2QsRef
  have h : (((blk.drop 16).getD
      (32 * (i / 128) + 16 * ((i % 32) / 16) + i % 16) 0#u8).val
        >>> (2 * ((i % 128) / 32))) &&& 3 ≤ 3 := Nat.and_le_right
  omega

/-- The 256 decoded two-bit codes of block `b`, entry `i` being `q2CodesRef` of the block bytes. -/
def q2kCodesList (raw : Slice Std.U8) (b : Nat) : List Std.U8 :=
  (List.range 256).map (fun i =>
    U8.ofNatCore (q2CodesRef (q2kBlockBytes raw b) i) (by
      have h := q2CodesRef_lt_4 (q2kBlockBytes raw b) i
      have h' : 2 ^ UScalarTy.U8.numBits = 256 := by norm_num [UScalarTy.U8_numBits_eq]
      omega))

/-- The exact `ℤ` value of within-block index `i`: `d·(sc & 0xF)·q − dmin·(sc >> 4)`. -/
def q2kBlockVal (d dmin : ℤ) (sc codes : List Std.U8) (i : Nat) : ℤ :=
  d * (((sc.getD (i / 16) 0#u8).val &&& 15 : Nat) : ℤ) * ((codes.getD i 0#u8).val : ℤ)
    - dmin * (((sc.getD (i / 16) 0#u8).val >>> 4 : Nat) : ℤ)

/-- The 256 exact weights of block `b`. -/
def q2kBlockVals (raw : Slice Std.U8) (b : Nat) : List ℤ :=
  (List.range 256).map (fun i =>
    q2kBlockVal (q2kD raw b) (q2kDmin raw b) (q2kScList raw b) (q2kCodesList raw b) i)

/-- The exact weight of row index `j`, block `j / 256`, within-block index `j % 256`. -/
def q2kWeight (raw : Slice Std.U8) (j : Nat) : ℤ :=
  (q2kBlockVals raw (j / 256)).getD (j % 256) 0

/-- The first `n` exact Q32.32 integers of a Q2_K row. -/
def q2kRowValues (raw : Slice Std.U8) (n : Nat) : List ℤ :=
  (List.range n).map (q2kWeight raw)

/-! ## `quant::q2_k_scale_min` -/

/-- The extracted `q2_k_scale_min` splits `sc` into its low nibble and high nibble. -/
theorem q2_k_scale_min_ref (sc : Std.U8) :
    ∃ a b : Std.U8, quant.q2_k_scale_min sc = ok (a, b) ∧
      a.val = (sc.val &&& 15) ∧ b.val = (sc.val >>> 4) := by
  obtain ⟨z, hz, hzv, -⟩ := WP.spec_imp_exists
    (U8.ShiftRight_IScalar_spec sc 4#i32 (by simp) (by simp))
  refine ⟨sc &&& 15#u8, z, ?_, ?_, ?_⟩
  · simp only [quant.q2_k_scale_min, lift, bind_tc_ok, hz]
  · rw [UScalar.val_and]; norm_num
  · rw [hzv]; simp

/-! ## `quant::q2_k_d` and `quant::q2_k_dmin` -/

/-- `quant::q2_k_d` succeeds on a whole block and reads the `f16` bits at byte 80. -/
theorem q2_k_d_ok (blk : Slice U8) (hlen : 84 ≤ blk.val.length) :
    quant.q2_k_d blk = ok (.Ok (f16LeHalf blk 80)) := by
  have hneed := quant_need_ok blk 84#usize (by simpa using hlen)
  have hu := quant_u16le_ok blk 80#usize (by norm_num; omega)
  rw [quant.q2_k_d, quant_Q2_K_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, hneed]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, quant.QK_K]
  obtain ⟨a, ha, hav⟩ := UScalar.div_spec (256#usize) (y := 16#usize) (by decide)
  obtain ⟨b, hb, hbv⟩ := UScalar.div_spec (256#usize) (y := 4#usize) (by decide)
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (Usize.add_spec (x := a) (y := b) (by rw [hav, hbv]; scalar_tac))
  have hc80 : c = 80#usize := UScalar.eq_of_val_eq (by rw [hcv, hav, hbv]; norm_num)
  rw [ha]; simp only [bind_tc_ok]
  rw [hb]; simp only [bind_tc_ok]
  rw [hc]; simp only [bind_tc_ok]
  rw [hc80, hu]
  simp only [bind_tc_ok]
  rfl

/-- `quant::q2_k_dmin` succeeds on a whole block and reads the `f16` bits at byte 82. -/
theorem q2_k_dmin_ok (blk : Slice U8) (hlen : 84 ≤ blk.val.length) :
    quant.q2_k_dmin blk = ok (.Ok (f16LeHalf blk 82)) := by
  have hneed := quant_need_ok blk 84#usize (by simpa using hlen)
  have hu := quant_u16le_ok blk 82#usize (by norm_num; omega)
  rw [quant.q2_k_dmin, quant_Q2_K_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, hneed]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, quant.QK_K]
  obtain ⟨a, ha, hav⟩ := UScalar.div_spec (256#usize) (y := 16#usize) (by decide)
  obtain ⟨b, hb, hbv⟩ := UScalar.div_spec (256#usize) (y := 4#usize) (by decide)
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (Usize.add_spec (x := a) (y := b) (by rw [hav, hbv]; scalar_tac))
  obtain ⟨d, hd, hdv⟩ := WP.spec_imp_exists (Usize.add_spec (x := c) (y := 2#usize) (by rw [hcv]; scalar_tac))
  have hd82 : d = 82#usize := UScalar.eq_of_val_eq (by rw [hdv, hcv, hav, hbv]; norm_num)
  rw [ha]; simp only [bind_tc_ok]
  rw [hb]; simp only [bind_tc_ok]
  rw [hc]; simp only [bind_tc_ok]
  rw [hd]; simp only [bind_tc_ok]
  rw [hd82, hu]
  simp only [bind_tc_ok]
  rfl

/-! ## `quant::q2_k_scales`

`q2_k_scales` copies the first sixteen bytes into a fresh `[u8; 16]` via `copy_from_slice`; the
copied window is exactly `blk[0..16]`. -/

/-- **`Q2_K` scale reader.** On a whole block, `q2_k_scales` returns the sixteen bytes `0..16`. -/
theorem q2_k_scales_ok (blk : Slice U8) (hlen : 16 ≤ blk.val.length) :
    ∃ v : Array U8 16#usize,
      quant.q2_k_scales blk = ok (core.result.Result.Ok v) ∧
      ∀ k, k < 16 → (v.val.getD k 0#u8).val = q2kScalesRef blk.val k := by
  have hneed := quant_need_ok blk 16#usize (by simpa using hlen)
  have hspec : quant.q2_k_scales blk ⦃ r => ∃ v : Array U8 16#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array U8 16#usize) quant.QuantError) ∧
      ∀ k, k < 16 → (v.val.getD k 0#u8).val = q2kScalesRef blk.val k ⦄ := by
    unfold quant.q2_k_scales
    simp only [quant.QK_K]
    obtain ⟨i, hi, hiv⟩ := UScalar.div_spec (256#usize) (y := 16#usize) (by decide)
    rw [hi]; simp only [bind_tc_ok]
    have hi16 : i = 16#usize := UScalar.eq_of_val_eq (by rw [hiv]; norm_num)
    rw [hi16, hneed]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    step*
    refine ⟨to_slice_mut_back s2, rfl, ?_⟩
    intro k hk
    have hs2len : s2.val.length = 16 := by rw [s2_post]; exact s1_post1
    rw [to_slice_mut_back_post, Array.from_slice_val _ _ hs2len, s2_post, s1_post]
    unfold q2kScalesRef
    rw [getD_slice blk.val 0 16 k (by omega) hk]
    simp
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  exact ⟨v, by rw [hr, hrv], hpt⟩

/-! ## `quant::q2_k_codes`

The four nested loops unpack the 64 `qs` bytes into 256 two-bit codes: `half ∈ {0,1}` selects the
32-byte half, `round ∈ 0..4` the 2-bit shift, `plane ∈ {0,1}` the 16-byte plane and `lane ∈ 0..16`
the byte. Output `y` runs `0..256` in the same order, so
`y = 128·half + 32·round + 16·plane + lane` and code `y` is
`(qs[32·half + 16·plane + lane] >>> 2·round) &&& 3`. -/

/-- The accumulator equals the reference on the first `thr` outputs and zero elsewhere. -/
def q2Pointwise (qs : Slice U8) (out : Array U8 256#usize) (thr : Nat) : Prop :=
  ∀ i, i < 256 → (out.val.getD i 0#u8).val = if i < thr then q2QsRef qs.val i else 0

/-- The reference index arithmetic: `q2QsRef` at the loop position `128h + 32r + 16p + l`. -/
lemma q2QsRef_at (qs : List Std.U8) (hN rN pN l : Nat)
    (_hh : hN < 2) (hr : rN < 4) (hp : pN < 2) (hl : l < 16) :
    q2QsRef qs (128 * hN + 32 * rN + 16 * pN + l)
      = (qs.getD (32 * hN + 16 * pN + l) 0#u8).val >>> (2 * rN) &&& 3 := by
  unfold q2QsRef
  have h1 : (128 * hN + 32 * rN + 16 * pN + l) / 128 = hN := by omega
  have h2 : ((128 * hN + 32 * rN + 16 * pN + l) % 128) / 32 = rN := by omega
  have h3 : ((128 * hN + 32 * rN + 16 * pN + l) % 32) / 16 = pN := by omega
  have h4 : (128 * hN + 32 * rN + 16 * pN + l) % 16 = l := by omega
  rw [h1, h2, h3, h4]

/-- The lane-loop (innermost) invariant: `l` lanes of plane `p` of half `h`, shift `2·r`. -/
def q2LaneInv (qs : Slice U8) (hN rN pN : Nat) (base : Usize) (shift : Std.U32) (lo : Usize)
    (out : Array U8 256#usize) (y : Usize) (l : Nat) : Prop :=
  hN < 2 ∧ rN < 4 ∧ pN < 2 ∧ l ≤ 16 ∧
  base.val = 32 * hN ∧ shift.val = 2 * rN ∧ lo.val = 16 * pN ∧
  y.val = 128 * hN + 32 * rN + 16 * pN + l ∧
  q2Pointwise qs out y.val

/-- One innermost-lane body step: read `qs[32·h + 16·p + l]`, shift, mask, write output `y`. -/
theorem q2Lane_body_step (qs : Slice U8) (hN rN pN : Nat) (base : Usize) (shift : Std.U32)
    (lo : Usize) (out : Array U8 256#usize) (y l : Usize)
    (hbar : 64 ≤ qs.val.length) (hh : hN < 2) (hr : rN < 4) (hp : pN < 2)
    (hbase : base.val = 32 * hN) (hshift : shift.val = 2 * rN) (hlo : lo.val = 16 * pN)
    (hy : y.val = 128 * hN + 32 * rN + 16 * pN + l.val) (hl : l.val < 16)
    (hout : q2Pointwise qs out y.val) :
    ∃ (out2 : Array U8 256#usize) (y2 l2 : Usize),
      quant.q2_k_codes_loop0_loop0_loop0_loop0.body qs base shift lo out y l
        = ok (ControlFlow.cont (out2, y2, l2)) ∧
      y2.val = 128 * hN + 32 * rN + 16 * pN + l2.val ∧
      l2.val = l.val + 1 ∧ l2.val ≤ 16 ∧ q2Pointwise qs out2 y2.val := by
  obtain ⟨i, hi, hiv⟩ := WP.spec_imp_exists (Usize.add_spec (x := base) (y := l) (by rw [hbase]; scalar_tac))
  have hiv' : i.val = 32 * hN + l.val := by rw [hiv, hbase]
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := lo) (by rw [hiv', hlo]; scalar_tac))
  have hi1v' : i1.val = 32 * hN + 16 * pN + l.val := by rw [hi1v, hiv', hlo]; ring
  have hi1lt : i1.val < qs.val.length := by
    have hl15 : l.val ≤ 15 := by omega
    have hb1 : 32 * hN ≤ 32 := by omega
    have hp1 : 16 * pN ≤ 16 := by omega
    rw [hi1v']; omega
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (Slice.index_usize_spec qs i1 (by exact hi1lt))
  have hcv' : c.val = (qs.val.getD i1.val 0#u8).val := by
    rw [hcv]
    exact congrArg (fun u : Std.U8 => u.val) (getD_of_lt hi1lt).symm
  obtain ⟨sh, hsh, hshv, -⟩ := WP.spec_imp_exists
    (UScalar.ShiftRight_spec c shift (by rw [hshift]; scalar_tac))
  obtain ⟨a4, ha4, ha4v, -⟩ := WP.spec_imp_exists (UScalar.and_spec sh 3#u8)
  have ha4eq : (sh &&& 3#u8) = a4 := by
    have := ha4
    simp only [lift, Result.ok.injEq] at this
    exact this
  have hlen : out.val.length = 256 := Array.property out
  have hylen : y.val < 256 := by omega
  obtain ⟨out2, hupd, hout2⟩ := WP.spec_imp_exists
    (Array.update_spec out y a4 (by simpa using hylen))
  obtain ⟨y2, hy2, hy2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := y) (y := 1#usize) (by scalar_tac))
  obtain ⟨l2, hl2, hl2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := l) (y := 1#usize) (by scalar_tac))
  have hy2v' : y2.val = y.val + 1 := hy2v
  have hl2v' : l2.val = l.val + 1 := hl2v
  have hy2abs : y2.val = 128 * hN + 32 * rN + 16 * pN + l2.val := by
    rw [hy2v', hy, hl2v']; ring
  have hl2le : l2.val ≤ 16 := by rw [hl2v']; omega
  refine ⟨out2, y2, l2, ?_, hy2abs, hl2v', hl2le, ?_⟩
  · unfold quant.q2_k_codes_loop0_loop0_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hl)]
    simp only [hi, hi1, hc, hsh, lift, bind_tc_ok]
    rw [ha4eq, hupd]
    simp only [hy2, hl2, bind_tc_ok]
  · intro i0 hi0
    rw [hout2, Array.set_val_eq, getD_set]
    by_cases hey : i0 = y.val
    · subst hey
      rw [if_pos ⟨rfl, by simpa using hylen⟩, hy2v', if_pos (by omega)]
      rw [ha4v, UScalar.val_and, hshv, hcv', hi1v', hshift, hy]
      norm_num
      exact (q2QsRef_at qs.val hN rN pN l.val hh hr hp hl).symm
    · rw [if_neg (fun h => hey h.1), hout i0 hi0]
      by_cases hlt : i0 < y.val
      · rw [if_pos hlt, if_pos (by rw [hy2v']; omega)]
      · rw [if_neg hlt, if_neg (by rw [hy2v']; omega)]

/-- The innermost lane loop: `loop.spec_decr_nat` with `q2LaneInv` and measure `16 − l`.

Complexity: 16 iterations, each O(1) (one byte read, one shift, one mask, one array write). -/
theorem q2Lane_loop_spec (qs : Slice U8) (hN rN pN : Nat) (base : Usize) (shift : Std.U32) (lo : Usize)
    (out : Array U8 256#usize) (y : Usize)
    (hbar : 64 ≤ qs.val.length) (hh : hN < 2) (hr : rN < 4) (hp : pN < 2)
    (hbase : base.val = 32 * hN) (hshift : shift.val = 2 * rN) (hlo : lo.val = 16 * pN)
    (hy : y.val = 128 * hN + 32 * rN + 16 * pN) (hout : q2Pointwise qs out y.val) :
    ∃ out1 y1,
      quant.q2_k_codes_loop0_loop0_loop0_loop0 qs out y base shift lo 0#usize
        = ok (out1, y1) ∧
      y1.val = 128 * hN + 32 * rN + 16 * pN + 16 ∧ q2Pointwise qs out1 y1.val := by
  have hspec : quant.q2_k_codes_loop0_loop0_loop0_loop0 qs out y base shift lo 0#usize
      ⦃ r => r.2.val = 128 * hN + 32 * rN + 16 * pN + 16 ∧ q2Pointwise qs r.1 r.2.val ⦄ := by
    unfold quant.q2_k_codes_loop0_loop0_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Array U8 256#usize × Usize × Usize => 16 - s.2.2.val)
      (inv := fun s => q2LaneInv qs hN rN pN base shift lo s.1 s.2.1 s.2.2.val)
      (post := fun r : Array U8 256#usize × Usize =>
        r.2.val = 128 * hN + 32 * rN + 16 * pN + 16 ∧ q2Pointwise qs r.1 r.2.val)
    · rintro ⟨outX, yX, lX⟩ hinv
      obtain ⟨hh', hr', hp', hl16, hbase', hshift', hlo', hyX, houtX⟩ := hinv
      dsimp only
      by_cases hlt : lX.val < 16
      · obtain ⟨out2, y2, l2, hbody, hy2, hl2, hl2le, hout2⟩ :=
          q2Lane_body_step qs hN rN pN base shift lo outX yX lX hbar hh' hr' hp'
            hbase' hshift' hlo' hyX hlt houtX
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        exact ⟨⟨hh', hr', hp', hl2le, hbase', hshift', hlo', hy2, hout2⟩, by omega⟩
      · have hbody : quant.q2_k_codes_loop0_loop0_loop0_loop0.body qs base shift lo outX yX lX
            = ok (ControlFlow.done (outX, yX)) := by
          unfold quant.q2_k_codes_loop0_loop0_loop0_loop0.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hl16' : lX.val = 16 := le_antisymm hl16 (Nat.le_of_not_lt hlt)
        exact ⟨by rw [hyX, hl16'], houtX⟩
    · refine ⟨hh, hr, hp, by simp, hbase, hshift, hlo, ?_, hout⟩
      rw [hy]
      simp
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r.1, r.2, hr, hpost.1, hpost.2⟩

/-- One plane body step: run the lane loop over the 16 bytes of plane `plane`. -/
theorem q2Plane_body_step (qs : Slice U8) (hN rN : Nat) (base : Usize) (shift : Std.U32)
    (out : Array U8 256#usize) (y plane : Usize)
    (hbar : 64 ≤ qs.val.length) (hh : hN < 2) (hr : rN < 4)
    (hbase : base.val = 32 * hN) (hshift : shift.val = 2 * rN) (hplane : plane.val < 2)
    (hy : y.val = 128 * hN + 32 * rN + 16 * plane.val) (hout : q2Pointwise qs out y.val) :
    ∃ out2 y2 plane2,
      quant.q2_k_codes_loop0_loop0_loop0.body qs base shift out y plane
        = ok (ControlFlow.cont (out2, y2, plane2)) ∧
      y2.val = 128 * hN + 32 * rN + 16 * plane2.val ∧ plane2.val = plane.val + 1 ∧
      q2Pointwise qs out2 y2.val := by
  obtain ⟨lo, hlo, hlov⟩ := WP.spec_imp_exists (Usize.mul_spec (x := plane) (y := 16#usize) (by scalar_tac))
  have hlov' : lo.val = 16 * plane.val := by
    rw [hlov, show (16#usize : Usize).val = 16 from rfl, Nat.mul_comm]
  obtain ⟨out1, y1, hloop, hy1, hout1⟩ :=
    q2Lane_loop_spec qs hN rN plane.val base shift lo out y hbar hh hr hplane hbase hshift hlov'
      hy hout
  obtain ⟨plane2, hp2, hp2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := plane) (y := 1#usize) (by scalar_tac))
  have hp2v' : plane2.val = plane.val + 1 := by rw [hp2v]; simp
  refine ⟨out1, y1, plane2, ?_, ?_, hp2v', hout1⟩
  · unfold quant.q2_k_codes_loop0_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hplane)]
    simp only [hlo, bind_tc_ok]
    change (do
        let (out1', y1') ← quant.q2_k_codes_loop0_loop0_loop0_loop0 qs out y base shift lo 0#usize
        let plane1 ← plane + 1#usize
        ok (ControlFlow.cont (out1', y1', plane1)))
      = ok (ControlFlow.cont (out1, y1, plane2))
    rw [hloop, hp2]
    simp only [bind_tc_ok]
    rfl
  · rw [hy1, hp2v']; ring

/-- The plane loop: `loop.spec_decr_nat` with measure `2 − plane`. -/
theorem q2Plane_loop_spec (qs : Slice U8) (hN rN : Nat) (base : Usize) (shift : Std.U32)
    (out : Array U8 256#usize) (y plane : Usize)
    (hbar : 64 ≤ qs.val.length) (hh : hN < 2) (hr : rN < 4)
    (hbase : base.val = 32 * hN) (hshift : shift.val = 2 * rN) (hplane : plane.val ≤ 2)
    (hy : y.val = 128 * hN + 32 * rN + 16 * plane.val) (hout : q2Pointwise qs out y.val) :
    ∃ out1 y1,
      quant.q2_k_codes_loop0_loop0_loop0 qs out y base shift plane = ok (out1, y1) ∧
      y1.val = 128 * hN + 32 * rN + 32 ∧ q2Pointwise qs out1 y1.val := by
  have hspec : quant.q2_k_codes_loop0_loop0_loop0 qs out y base shift plane
      ⦃ r => r.2.val = 128 * hN + 32 * rN + 32 ∧ q2Pointwise qs r.1 r.2.val ⦄ := by
    unfold quant.q2_k_codes_loop0_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Array U8 256#usize × Usize × Usize => 2 - s.2.2.val)
      (inv := fun s => hN < 2 ∧ rN < 4 ∧ s.2.2.val ≤ 2 ∧
        base.val = 32 * hN ∧ shift.val = 2 * rN ∧
        s.2.1.val = 128 * hN + 32 * rN + 16 * s.2.2.val ∧ q2Pointwise qs s.1 s.2.1.val)
      (post := fun r : Array U8 256#usize × Usize =>
        r.2.val = 128 * hN + 32 * rN + 32 ∧ q2Pointwise qs r.1 r.2.val)
    · rintro ⟨outX, yX, plX⟩ hinv
      obtain ⟨hh', hr', hpl2, hbase', hshift', hyX, houtX⟩ := hinv
      dsimp only
      by_cases hlt : plX.val < 2
      · obtain ⟨out2, y2, pl2, hbody, hy2, hpl2v, hout2⟩ :=
          q2Plane_body_step qs hN rN base shift outX yX plX hbar hh' hr' hbase' hshift' hlt hyX houtX
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        exact ⟨⟨hh', hr', by rw [hpl2v]; omega, hbase', hshift',
          by omega, hout2⟩, by rw [hpl2v]; omega⟩
      · have hbody : quant.q2_k_codes_loop0_loop0_loop0.body qs base shift outX yX plX
            = ok (ControlFlow.done (outX, yX)) := by
          unfold quant.q2_k_codes_loop0_loop0_loop0.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hpl2' : plX.val = 2 := le_antisymm hpl2 (Nat.le_of_not_lt hlt)
        exact ⟨by rw [hyX, hpl2'], houtX⟩
    · exact ⟨hh, hr, hplane, hbase, hshift, hy, hout⟩
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r.1, r.2, hr, hpost.1, hpost.2⟩

/-- One round body step: run the plane loop for shift `2·round`, then advance. -/
theorem q2Round_body_step (qs : Slice U8) (hN : Nat) (base : Usize)
    (out : Array U8 256#usize) (y : Usize) (shiftv : Std.U32) (round : Usize)
    (hbar : 64 ≤ qs.val.length) (hh : hN < 2)
    (hbase : base.val = 32 * hN) (hround : round.val < 4)
    (hshift : shiftv.val = 2 * round.val)
    (hy : y.val = 128 * hN + 32 * round.val) (hout : q2Pointwise qs out y.val) :
    ∃ out2 y2 shift2 round2,
      quant.q2_k_codes_loop0_loop0.body qs base out y shiftv round
        = ok (ControlFlow.cont (out2, y2, shift2, round2)) ∧
      y2.val = 128 * hN + 32 * round2.val ∧ shift2.val = 2 * round2.val ∧
      round2.val = round.val + 1 ∧ q2Pointwise qs out2 y2.val := by
  obtain ⟨out1, y1, hloop, hy1, hout1⟩ :=
    q2Plane_loop_spec qs hN round.val base shiftv out y 0#usize hbar hh hround hbase hshift
      (by simp) (by rw [hy, show (0#usize : Usize).val = 0 from rfl]; ring) hout
  obtain ⟨shift2, hs2, hs2v⟩ := WP.spec_imp_exists (U32.add_spec (x := shiftv) (y := 2#u32) (by scalar_tac))
  obtain ⟨round2, hr2, hr2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := round) (y := 1#usize) (by scalar_tac))
  have hs2v' : shift2.val = shiftv.val + 2 := hs2v
  have hr2v' : round2.val = round.val + 1 := by rw [hr2v]; simp
  refine ⟨out1, y1, shift2, round2, ?_, ?_, ?_, hr2v', hout1⟩
  · unfold quant.q2_k_codes_loop0_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hround)]
    change (do
        let (out1', y1') ← quant.q2_k_codes_loop0_loop0_loop0 qs out y base shiftv 0#usize
        let shift1 ← shiftv + 2#u32
        let round1 ← round + 1#usize
        ok (ControlFlow.cont (out1', y1', shift1, round1)))
      = ok (ControlFlow.cont (out1, y1, shift2, round2))
    rw [hloop, hs2, hr2]
    simp only [bind_tc_ok]
    rfl
  · rw [hy1, hr2v']; ring
  · rw [hs2v', hr2v', hshift]; ring

/-- The round loop: `loop.spec_decr_nat` with measure `4 − round`. -/
theorem q2Round_loop_spec (qs : Slice U8) (hN : Nat) (base : Usize) (shift : Std.U32)
    (out : Array U8 256#usize) (y round : Usize)
    (hbar : 64 ≤ qs.val.length) (hh : hN < 2)
    (hbase : base.val = 32 * hN) (hround : round.val ≤ 4) (hshift : shift.val = 2 * round.val)
    (hy : y.val = 128 * hN + 32 * round.val) (hout : q2Pointwise qs out y.val) :
    ∃ out1 y1,
      quant.q2_k_codes_loop0_loop0 qs out y base shift round = ok (out1, y1) ∧
      y1.val = 128 * hN + 128 ∧ q2Pointwise qs out1 y1.val := by
  have hspec : quant.q2_k_codes_loop0_loop0 qs out y base shift round
      ⦃ r => r.2.val = 128 * hN + 128 ∧ q2Pointwise qs r.1 r.2.val ⦄ := by
    unfold quant.q2_k_codes_loop0_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Array U8 256#usize × Usize × U32 × Usize => 4 - s.2.2.2.val)
      (inv := fun s => hN < 2 ∧ s.2.2.2.val ≤ 4 ∧ base.val = 32 * hN ∧
        s.2.2.1.val = 2 * s.2.2.2.val ∧
        s.2.1.val = 128 * hN + 32 * s.2.2.2.val ∧ q2Pointwise qs s.1 s.2.1.val)
      (post := fun r : Array U8 256#usize × Usize =>
        r.2.val = 128 * hN + 128 ∧ q2Pointwise qs r.1 r.2.val)
    · rintro ⟨outX, yX, shX, rnX⟩ hinv
      obtain ⟨hh', hrn4, hbase', hshiftX, hyX, houtX⟩ := hinv
      dsimp only
      by_cases hlt : rnX.val < 4
      · obtain ⟨out2, y2, sh2, rn2, hbody, hy2, hsh2, hrn2, hout2⟩ :=
          q2Round_body_step qs hN base outX yX shX rnX hbar hh' hbase' hlt hshiftX hyX houtX
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        exact ⟨⟨hh', by rw [hrn2]; omega, hbase', hsh2, by omega, hout2⟩,
          by rw [hrn2]; omega⟩
      · have hbody : quant.q2_k_codes_loop0_loop0.body qs base outX yX shX rnX
            = ok (ControlFlow.done (outX, yX)) := by
          unfold quant.q2_k_codes_loop0_loop0.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hrn4' : rnX.val = 4 := le_antisymm hrn4 (Nat.le_of_not_lt hlt)
        exact ⟨by rw [hyX, hrn4'], houtX⟩
    · exact ⟨hh, hround, hbase, hshift, hy, hout⟩
  obtain ⟨r, hr, hpost⟩ := WP.spec_imp_exists hspec
  exact ⟨r.1, r.2, hr, hpost.1, hpost.2⟩

/-- One half body step: run the round loop for the 32-byte half `half`, then advance. -/
theorem q2Half_body_step (qs : Slice U8) (out : Array U8 256#usize) (y half : Usize)
    (hbar : 64 ≤ qs.val.length) (hhalf : half.val < 2)
    (hy : y.val = 128 * half.val) (hout : q2Pointwise qs out y.val) :
    ∃ out2 y2 half2,
      quant.q2_k_codes_loop0.body qs out y half = ok (ControlFlow.cont (out2, y2, half2)) ∧
      y2.val = 128 * half2.val ∧ half2.val = half.val + 1 ∧ q2Pointwise qs out2 y2.val := by
  obtain ⟨base, hb, hbv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := half) (y := 32#usize) (by scalar_tac))
  have hbv' : base.val = 32 * half.val := by
    rw [hbv, show (32#usize : Usize).val = 32 from rfl, Nat.mul_comm]
  obtain ⟨out1, y1, hloop, hy1, hout1⟩ :=
    q2Round_loop_spec qs half.val base 0#u32 out y 0#usize hbar hhalf hbv'
      (by simp) (by simp) (by rw [hy, show (0#usize : Usize).val = 0 from rfl]; ring) hout
  obtain ⟨half2, hh2, hh2v⟩ := WP.spec_imp_exists (Usize.add_spec (x := half) (y := 1#usize) (by scalar_tac))
  have hh2v' : half2.val = half.val + 1 := by rw [hh2v]; simp
  refine ⟨out1, y1, half2, ?_, ?_, hh2v', hout1⟩
  · unfold quant.q2_k_codes_loop0.body
    rw [if_pos (by rw [UScalar.lt_equiv]; exact hhalf)]
    change (do
        let base1 ← half * 32#usize
        let (out1', y1') ← quant.q2_k_codes_loop0_loop0 qs out y base1 0#u32 0#usize
        let half1 ← half + 1#usize
        ok (ControlFlow.cont (out1', y1', half1)))
      = ok (ControlFlow.cont (out1, y1, half2))
    simp only [hb, bind_tc_ok]
    rw [hloop, hh2]
    simp only [bind_tc_ok]
    rfl
  · rw [hy1, hh2v']; ring

/-- The half loop: `loop.spec_decr_nat` with measure `2 − half`. -/
theorem q2Half_loop_spec (qs : Slice U8) (out : Array U8 256#usize) (y half : Usize)
    (hbar : 64 ≤ qs.val.length) (hhalf : half.val ≤ 2)
    (hy : y.val = 128 * half.val) (hout : q2Pointwise qs out y.val) :
    WP.spec (quant.q2_k_codes_loop0 qs out y half) (fun out1 => q2Pointwise qs out1 256) := by
  have hspec : quant.q2_k_codes_loop0 qs out y half
      ⦃ out1 => q2Pointwise qs out1 256 ⦄ := by
    unfold quant.q2_k_codes_loop0
    apply loop.spec_decr_nat
      (measure := fun s : Array U8 256#usize × Usize × Usize => 2 - s.2.2.val)
      (inv := fun s => s.2.2.val ≤ 2 ∧ s.2.1.val = 128 * s.2.2.val ∧ q2Pointwise qs s.1 s.2.1.val)
      (post := fun out1 : Array U8 256#usize => q2Pointwise qs out1 256)
    · rintro ⟨outX, yX, hfX⟩ hinv
      obtain ⟨hhf2, hyX, houtX⟩ := hinv
      dsimp only
      by_cases hlt : hfX.val < 2
      · obtain ⟨out2, y2, hf2, hbody, hy2, hf2v, hout2⟩ :=
          q2Half_body_step qs outX yX hfX hbar hlt hyX houtX
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        exact ⟨⟨by omega, by omega, hout2⟩, by omega⟩
      · have hbody : quant.q2_k_codes_loop0.body qs outX yX hfX
            = ok (ControlFlow.done outX) := by
          unfold quant.q2_k_codes_loop0.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        simp only [hbody, Aeneas.Std.WP.spec_ok]
        have hf2' : hfX.val = 2 := le_antisymm hhf2 (Nat.le_of_not_lt hlt)
        intro i hi
        rw [houtX i hi, if_pos (by rw [hyX, hf2']; omega), if_pos hi]
    · exact ⟨hhalf, hy, hout⟩
  exact hspec

/-- `q2QsRef` of a slice window is the same on the dropped window. -/
lemma q2QsRef_slice (blk : List Std.U8) (i : Nat) (hi : i < 256) :
    q2QsRef (blk.slice 16 80) i = q2QsRef (blk.drop 16) i := by
  unfold q2QsRef
  have hk : 32 * (i / 128) + 16 * ((i % 32) / 16) + i % 16 < 64 := by omega
  have hs : blk.slice 16 80 = (blk.drop 16).take 64 := rfl
  rw [hs, getD_take (blk.drop 16) 64 _ 0#u8 hk]

/-- **`Q2_K` code reader.** On a whole block, `q2_k_codes` returns the 256 two-bit codes
pointwise `q2QsRef` of the `qs` window. -/
theorem q2_k_codes_ok (blk : Slice U8) (hlen : 80 ≤ blk.val.length) :
    ∃ v : Array U8 256#usize,
      quant.q2_k_codes blk = ok (core.result.Result.Ok v) ∧
      ∀ i, i < 256 → (v.val.getD i 0#u8).val = q2QsRef (blk.val.drop 16) i := by
  have hneed := quant_need_ok blk 80#usize (by simpa using hlen)
  have hspec : quant.q2_k_codes blk ⦃ r => ∃ v : Array U8 256#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array U8 256#usize) quant.QuantError) ∧
      ∀ i, i < 256 → (v.val.getD i 0#u8).val = q2QsRef (blk.val.drop 16) i ⦄ := by
    unfold quant.q2_k_codes
    simp only [quant.QK_K]
    obtain ⟨i, hi, hiv⟩ := UScalar.div_spec (256#usize) (y := 16#usize) (by decide)
    rw [hi]; simp only [bind_tc_ok]
    have hi16 : i = 16#usize := UScalar.eq_of_val_eq (by rw [hiv]; norm_num)
    rw [hi16]
    obtain ⟨i1, hi1, hi1v⟩ := UScalar.div_spec (256#usize) (y := 4#usize) (by decide)
    rw [hi1]; simp only [bind_tc_ok]
    have hi164 : i1 = 64#usize := UScalar.eq_of_val_eq (by rw [hi1v]; norm_num)
    rw [hi164]
    obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
      (Usize.add_spec (x := 16#usize) (y := 64#usize) (by scalar_tac))
    rw [hi2]; simp only [bind_tc_ok]
    have hi280 : i2 = 80#usize := UScalar.eq_of_val_eq (by rw [hi2v]; norm_num)
    rw [hi280, hneed]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    step*
    have hbar : 64 ≤ qs.val.length := by
      have h := qs_post1
      simp only [Slice.length] at h
      rw [h]
    have hout0 : q2Pointwise qs (Array.repeat 256#usize 0#u8) 0 := by
      intro i hi
      rw [Array.repeat_val]
      simp only [List.getD, List.getElem?_replicate]
      rw [if_pos (by simpa using hi), if_neg (by omega)]
      rfl
    refine WP.spec_bind (q2Half_loop_spec qs (Array.repeat 256#usize 0#u8) 0#usize 0#usize hbar
      (by decide) (by decide) hout0) ?_
    intro out1 hpoint
    refine ⟨out1, rfl, ?_⟩
    intro i hi
    rw [hpoint i hi, if_pos hi, qs_post, q2QsRef_slice blk.val i hi]
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  exact ⟨v, by rw [hr, hrv], hpt⟩

/-! ## `dequant::q2_k_block` -/

/-- **`Q2_K` block law.** On the block window returned by `block_slice` with finite `d`/`dmin`,
`q2_k_block` returns `(d, dmin, sc, codes)` with the exact Q32.32 `f16` values, the sixteen scale
bytes and the 256 codes `q2CodesRef`. -/
theorem q2_k_block_spec (raw : Slice U8) (block : Std.Usize) (blk : Slice Std.U8)
    (hblk : dequant.block_slice raw block 84#usize = ok (.Ok blk))
    (h84 : 84 ≤ blk.val.length)
    (hd : f16Exp (f16LeHalf blk 80) ≠ 31)
    (hdmin : f16Exp (f16LeHalf blk 82) ≠ 31) :
    ∃ d dmin sc codes,
      dequant.q2_k_block raw block = ok (.Ok (d, dmin, sc, codes)) ∧
      d.val = f16Value (f16LeHalf blk 80) ∧
      dmin.val = f16Value (f16LeHalf blk 82) ∧
      (∀ k, k < 16 → (sc.val.getD k 0#u8).val = q2kScalesRef blk.val k) ∧
      (∀ i, i < 256 → (codes.val.getD i 0#u8).val = q2CodesRef blk.val i) := by
  have hdok := q2_k_d_ok blk h84
  have hdminok := q2_k_dmin_ok blk h84
  obtain ⟨d, hd_eq, hdv⟩ := f16_to_fixed_exact (f16LeHalf blk 80) hd
  obtain ⟨dmin, hdmin_eq, hdminv⟩ := f16_to_fixed_exact (f16LeHalf blk 82) hdmin
  obtain ⟨sc, hsc_eq, hsc⟩ := q2_k_scales_ok blk (by omega)
  obtain ⟨codes, hcodes_eq, hcodes⟩ := q2_k_codes_ok blk (by omega)
  have hcodes_eq' : quant.q2_k_codes blk = ok (.Ok codes) := hcodes_eq
  refine ⟨d, dmin, sc, codes, ?_, hdv, hdminv, hsc, ?_⟩
  · simp only [dequant.q2_k_block, quant_Q2_K_BLOCK_BYTES_eq, hblk, hdok, hd_eq, hdminok,
      hdmin_eq, hsc_eq, hcodes_eq', core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  · intro i hi
    rw [hcodes i hi]
    simp [q2CodesRef]

/-- **`Q2_K` block dispatch.** `dequant_row_fixed` at `ggml_type = 10` (`GGML_TYPE_Q2_K`) is
`dequant_q2_k_row`. -/
theorem dequant_row_fixed_q2_k (raw : Slice Std.U8) (n : Std.Usize) (out : alloc.vec.Vec Std.I64) :
    dequant.dequant_row_fixed 10#u32 raw n out = dequant.dequant_q2_k_row raw n out := rfl

/-! ## Checked `i64` and `f16` bounds

A finite `f16` scale is `< 2^48` in Q32.32, a nibble code is `≤ 15`, a two-bit code is `≤ 3`, so
every product and difference stays inside `i64`. -/

lemma q2f16Mag_le (h : U16) (he : f16Exp h ≠ 31) : f16Mag h ≤ 2047 * 2 ^ 37 := by
  unfold f16Mag
  have hm : f16Man h ≤ 1023 := by have := f16_man_lt h; omega
  split
  · have h1 : (f16Man h : ℤ) * 2 ^ 8 ≤ 1023 * 2 ^ 8 :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hm) (by positivity)
    have h2 : (1023 : ℤ) * 2 ^ 8 ≤ 2047 * 2 ^ 37 := by norm_num
    linarith
  · have he' : f16Exp h + 7 ≤ 37 := by have := f16_exp_lt h; omega
    have h1 : ((1024 + f16Man h : ℕ) : ℤ) ≤ 2047 := by push_cast; omega
    have h2 : (2 : ℤ) ^ (f16Exp h + 7) ≤ 2 ^ 37 := pow_le_pow_right₀ (by norm_num) he'
    have h3 : ((1024 + f16Man h : ℕ) : ℤ) * 2 ^ (f16Exp h + 7) ≤ 2047 * 2 ^ 37 :=
      mul_le_mul h1 h2 (by positivity) (by norm_num)
    linarith

lemma q2f16Mag_nonneg (h : U16) : 0 ≤ f16Mag h := by
  unfold f16Mag
  split <;> positivity

lemma q2f16Value_abs_lt (h : U16) (he : f16Exp h ≠ 31) : |f16Value h| < 2 ^ 48 := by
  have hm := q2f16Mag_le h he
  have h48 : (2047 : ℤ) * 2 ^ 37 < 2 ^ 48 := by norm_num
  have hnn := q2f16Mag_nonneg h
  unfold f16Value
  split
  · rw [abs_neg, abs_of_nonneg hnn]; linarith
  · rw [abs_of_nonneg hnn]; linarith

theorem q2i_mul_fits15 {x y : ℤ} (hx : |x| ≤ 2 ^ 48) (hy : |y| ≤ 15) :
    I64.min ≤ x * y ∧ x * y ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 48 * 15 < 2 ^ 63 := by norm_num
  have hxy : |x * y| < 2 ^ 63 := by
    rw [abs_mul]
    have h : |x| * |y| ≤ (2 : ℤ) ^ 48 * 15 :=
      mul_le_mul hx hy (abs_nonneg y) (by positivity)
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  omega

theorem q2i_mul_fits3 {x y : ℤ} (hx : |x| ≤ 2 ^ 54) (hy : |y| ≤ 3) :
    I64.min ≤ x * y ∧ x * y ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 54 * 3 < 2 ^ 63 := by norm_num
  have hxy : |x * y| < 2 ^ 63 := by
    rw [abs_mul]
    have h : |x| * |y| ≤ (2 : ℤ) ^ 54 * 3 :=
      mul_le_mul hx hy (abs_nonneg y) (by positivity)
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  omega

theorem q2i_sub_fits {x y : ℤ} (hx : |x| ≤ 2 ^ 54 * 3) (hy : |y| ≤ 2 ^ 54) :
    I64.min ≤ x - y ∧ x - y ≤ I64.max := by
  have h2 : (2 : ℤ) ^ 54 * 3 + 2 ^ 54 < 2 ^ 63 := by norm_num
  have hxy : |x - y| < 2 ^ 63 := by
    have h1 : |x - y| ≤ |x| + |y| := by simpa [sub_eq_add_neg] using abs_add_le x (-y)
    have h : |x| + |y| ≤ (2 : ℤ) ^ 54 * 3 + 2 ^ 54 := add_le_add hx hy
    linarith
  have hb := abs_lt.mp hxy
  rw [I64.min_eq, I64.max_eq]
  omega

/-- `dequant::sub` on the success path returns the exact `i64` difference. -/
theorem q2k_sub_ok (a b : I64)
    (hlo : I64.min ≤ a.val - b.val) (hhi : a.val - b.val ≤ I64.max) :
    ∃ z : I64, dequant.sub a b = ok (.Ok z) ∧ z.val = a.val - b.val := by
  have hspec := I64.checked_sub_bv_spec a b
  cases hc : core.num.checked_sub_IScalar a b with
  | none => rw [hc] at hspec; exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      refine ⟨z, ?_, hspec.2.2.1⟩
      unfold dequant.sub
      rw [show I64.checked_sub a b = some z by simpa only [I64.checked_sub] using hc]
      simp only [lift, bind_tc_ok]
      rw [core.option.Option.ok_or_some]

/-- An unsigned-`u8` code cast to `i64` keeps its value. -/
lemma q2k_hcast_u8_val (c : Std.U8) : (UScalar.hcast .I64 c).val = (c.val : ℤ) := by
  rw [UScalar.hcast_val_eq, IScalarTy.I64_numBits_eq]
  apply Int.bmod_eq_of_le
  · norm_num
  · have h := U8.lt_succ_max c
    norm_num
    omega

/-! ## Q2_K lane loop

The innermost loop writes `slope·code − intercept` at `vals[16·group + lane]`, `lane < 16`. -/

/-- The exact `ℤ` value written at lane `l` of group `g`. -/
def q2kLaneVal (codes : Array Std.U8 256#usize) (g l : Nat) (slope intercept : ℤ) : ℤ :=
  slope * ((codes.val.getD (16 * g + l) 0#u8).val : ℤ) - intercept

/-- Lane-loop invariant: lanes `0..lane-1` of `group` hold `q2kLaneVal`, every other entry is
`vals0`. -/
def q2kLaneInv (vals0 vals : Array I64 256#usize) (codes : Array Std.U8 256#usize)
    (slope intercept : ℤ) (group lane : Nat) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane ≤ 16 ∧
  (∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j / 16 = group ∧ j % 16 < lane then q2kLaneVal codes group (j % 16) slope intercept
    else (vals0.val.getD j 0#i64).val)

/-- One lane-loop body step on the success path: the checked multiply and subtract succeed and the
write advances the invariant by one lane. -/
theorem q2kLane_body_step (vals0 vals : Array I64 256#usize) (codes : Array Std.U8 256#usize)
    (slope intercept : I64) (group lane : Usize)
    (hgroup : group.val < 16) (hlane : lane.val < 16)
    (hcode : (codes.val.getD (16 * group.val + lane.val) 0#u8).val ≤ 3)
    (hs : |slope.val| ≤ 2 ^ 54) (hi : |intercept.val| ≤ 2 ^ 54)
    (hinv : q2kLaneInv vals0 vals codes slope.val intercept.val group.val lane.val true none) :
    ∃ (ok2 : Bool) (err1 : Option error.EngineError) (vals1 : Array I64 256#usize) (lane2 : Usize),
      dequant.dequant_q2_k_row_loop0_loop0_loop0.body codes group slope intercept true none vals lane
          = ok (ControlFlow.cont (ok2, err1, vals1, lane2)) ∧
      ok2 = true ∧ err1 = none ∧ lane2.val = lane.val + 1 ∧
      q2kLaneInv vals0 vals1 codes slope.val intercept.val group.val (lane.val + 1) ok2 err1 := by
  obtain ⟨i, hm16, hiv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 16#usize) (y := group) (by scalar_tac))
  have hiv' : i.val = 16 * group.val := by rw [hiv]; simp
  obtain ⟨i1, hadd, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := lane) (by rw [hiv']; scalar_tac))
  have hi1' : i1.val = 16 * group.val + lane.val := by rw [hi1v, hiv']
  obtain ⟨i2, hidx, hi2v⟩ := WP.spec_imp_exists
    (Array.index_usize_spec codes i1
      (by show i1.val < codes.val.length; rw [Array.property codes, hi1']; scalar_tac))
  have hi2val : i2.val = (codes.val.getD (16 * group.val + lane.val) 0#u8).val := by
    have hlt : i1.val < codes.val.length := by rw [Array.property codes, hi1']; scalar_tac
    have h : i2.val = (codes.val.getD i1.val 0#u8).val := by
      rw [hi2v]
      exact congrArg (fun x : U8 => x.val) (getD_of_lt (l := codes.val) hlt).symm
    rwa [hi1'] at h
  let i3 : I64 := UScalar.hcast .I64 i2
  have hcast : I64.Insts.CoreConvertFromU8.from i2 = ok i3 := rfl
  have hi3v : i3.val = (i2.val : ℤ) := q2k_hcast_u8_val i2
  have hcle : i2.val ≤ 3 := by rw [hi2val]; exact hcode
  have hi3abs : |i3.val| ≤ 3 := by
    rw [hi3v, abs_of_nonneg (by positivity)]
    exact_mod_cast hcle
  obtain ⟨prod, hmulv, hprodv⟩ := q6k_mul_ok slope i3
    (q2i_mul_fits3 hs hi3abs).1 (q2i_mul_fits3 hs hi3abs).2
  have hprodabs : |prod.val| ≤ (2 : ℤ) ^ 54 * 3 := by
    rw [hprodv, abs_mul]
    exact mul_le_mul hs hi3abs (abs_nonneg _) (by positivity)
  obtain ⟨v, hsubv, hvval⟩ := q2k_sub_ok prod intercept
    (q2i_sub_fits hprodabs hi).1 (q2i_sub_fits hprodabs hi).2
  obtain ⟨a, hupd, hau⟩ := WP.spec_imp_exists
    (Array.update_spec vals i1 v
      (by show i1.val < vals.val.length; rw [Array.property vals, hi1']; scalar_tac))
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  have hdiv : i1.val / 16 = group.val := by rw [hi1']; exact div16_mul_add group.val lane.val hlane
  have hmod : i1.val % 16 = lane.val := by rw [hi1']; exact mod16_mul_add group.val lane.val hlane
  refine ⟨true, none, a, lane2, ?_, rfl, rfl, hlane2, ?_⟩
  · unfold dequant.dequant_q2_k_row_loop0_loop0_loop0.body
    rw [if_pos (by scalar_tac : lane < 16#usize), if_pos (by simp : (true : Bool) = true)]
    simp only [hm16, hadd, hidx, hcast, hmulv, hsubv, hupd, haddl, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    have hvals_len : vals.val.length = 256 := Array.property vals
    rw [hau, Array.set_val_eq, getD_set]
    by_cases hjk : j = i1.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [hvals_len]; rw [hi1']; omega⟩,
        if_pos ⟨hdiv, by omega⟩, hmod]
      unfold q2kLaneVal
      rw [hvval, hprodv, hi3v, hi2val]
    · rw [if_neg (fun h => hjk h.1), hinv.2.2.2 j hj]
      by_cases hcond : j / 16 = group.val ∧ j % 16 < lane.val
      · rw [if_pos hcond, if_pos ⟨hcond.1, by omega⟩]
      · rw [if_neg hcond]
        have hnot : ¬ (j / 16 = group.val ∧ j % 16 < lane.val + 1) := by
          rintro ⟨h1, h2⟩
          refine hcond ⟨h1, ?_⟩
          by_contra hge
          have hge' : lane.val ≤ j % 16 := not_lt.mp hge
          have hj16 : j = 16 * group.val + j % 16 := eq_of_div16 j group.val h1
          have : j % 16 = lane.val := by omega
          exact hjk (by rw [hj16, this, hi1'])
        rw [if_neg hnot]

/-- The lane loop: `loop.spec_decr_nat` with `q2kLaneInv` and measure `16 − lane`.

Complexity: 16 iterations, each O(1) (one array read, one checked multiply, one checked subtract,
one array write). -/
theorem q2kLane_loop_spec (vals0 : Array I64 256#usize) (codes : Array Std.U8 256#usize)
    (slope intercept : I64) (group : Usize) (hgroup : group.val < 16)
    (hcode : ∀ l, l < 16 → (codes.val.getD (16 * group.val + l) 0#u8).val ≤ 3)
    (hs : |slope.val| ≤ 2 ^ 54) (hi : |intercept.val| ≤ 2 ^ 54)
    (hinv0 : q2kLaneInv vals0 vals0 codes slope.val intercept.val group.val 0 true none) :
    WP.spec (dequant.dequant_q2_k_row_loop0_loop0_loop0 true none codes vals0 group slope intercept 0#usize)
      (fun b => ∃ vals : Array I64 256#usize,
        b = (true, none, vals) ∧
        q2kLaneInv vals0 vals codes slope.val intercept.val group.val 16 true none) := by
  unfold dequant.dequant_q2_k_row_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Option error.EngineError × Array I64 256#usize × Usize =>
      16 - s.2.2.2.val)
    (inv := fun s => q2kLaneInv vals0 s.2.2.1 codes slope.val intercept.val group.val s.2.2.2.val
      s.1 s.2.1)
  · intro s hs'
    rcases s with ⟨ok1, err, vals, lane⟩
    dsimp only at hs' ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs'
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    rw [hok', herr']
    by_cases hlt : lane.val < 16
    · obtain ⟨ok2, err1, vals1, lane2, hbody, hok2, herr1, hlane2, hinv2⟩ :=
        q2kLane_body_step vals0 vals codes slope intercept group lane hgroup hlt
          (hcode lane.val hlt) hs hi ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane2] using hinv2, ?_⟩
      show 16 - lane2.val < 16 - lane.val
      omega
    · have hnot : ¬ (lane < 16#usize) := fun hc => hlt ((UScalar.lt_equiv lane 16#usize).1 hc)
      have hbody : dequant.dequant_q2_k_row_loop0_loop0_loop0.body codes group slope intercept
            true none vals lane = ok (ControlFlow.done (true, none, vals)) := by
        rw [dequant.dequant_q2_k_row_loop0_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨vals, rfl, ?_⟩
      have hlane16 : lane.val = 16 := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      intro j hj
      rw [hrec j hj, hlane16]
  · exact hinv0

/-! ## Q2_K group loop -/

/-- Group-loop invariant: every entry whose group is below `group` holds `d·sc·q − dmin·m`; every
other entry is `vals0`. -/
def q2kGroupInv (d dmin : I64) (scales : Array Std.U8 16#usize) (codes : Array Std.U8 256#usize)
    (vals0 vals : Array I64 256#usize) (group : Nat) (ok1 : Bool) (err : Option error.EngineError) :
    Prop :=
  ok1 = true ∧ err = none ∧ group ≤ 16 ∧
  (∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j / 16 < group then q2kBlockVal d.val dmin.val scales.val codes.val j
    else (vals0.val.getD j 0#i64).val)

/-- Destructuring a successful pair bind reduces to the direct application. -/
theorem q2k_let_pair {α β γ : Type} (x : α) (y : β) (f : α → β → Result γ) :
    (do let (a, b) ← (ok (x, y) : Result (α × β)); f a b) = f x y := by
  simp only [bind_tc_ok]
  rfl

/-- One group-loop body step on the success path: split the packed scale byte, take the two checked
products `d·sc`, `dmin·m`, run the lane loop and advance. -/
theorem q2kGroup_body_step (d dmin : I64) (scales : Array Std.U8 16#usize)
    (codes : Array Std.U8 256#usize) (vals0 vals : Array I64 256#usize) (group : Usize)
    (hgroup : group.val < 16) (hsc : group.val < scales.val.length)
    (hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 3)
    (hd : |d.val| ≤ 2 ^ 48) (hm : |dmin.val| ≤ 2 ^ 48)
    (hinv : q2kGroupInv d dmin scales codes vals0 vals group.val true none) :
    ∃ (ok3 : Bool) (err2 : Option error.EngineError) (vals1 : Array I64 256#usize) (group2 : Usize),
      dequant.dequant_q2_k_row_loop0_loop0.body d dmin scales codes true none vals group
          = ok (ControlFlow.cont (ok3, err2, vals1, group2)) ∧
      ok3 = true ∧ err2 = none ∧ group2.val = group.val + 1 ∧
      q2kGroupInv d dmin scales codes vals0 vals1 (group.val + 1) ok3 err2 := by
  obtain ⟨sc, hidx, hscv⟩ := WP.spec_imp_exists (Array.index_usize_spec scales group hsc)
  have hscval : sc.val = (scales.val.getD group.val 0#u8).val := by
    rw [hscv]
    exact congrArg (fun x : U8 => x.val) (getD_of_lt (l := scales.val) hsc).symm
  obtain ⟨scc, mcc, hscm, hsccv, hmccv⟩ := q2_k_scale_min_ref sc
  have hscc_le : scc.val ≤ 15 := by rw [hsccv]; exact Nat.and_le_right
  have hmcc_le : mcc.val ≤ 15 := by
    rw [hmccv]
    have := U8.lt_succ_max sc
    omega
  let i1 : I64 := UScalar.hcast .I64 scc
  have hcast1 : I64.Insts.CoreConvertFromU8.from scc = ok i1 := rfl
  have hi1v : i1.val = (scc.val : ℤ) := q2k_hcast_u8_val scc
  have hi1abs : |i1.val| ≤ 15 := by
    rw [hi1v, abs_of_nonneg (by positivity)]; exact_mod_cast hscc_le
  obtain ⟨slope, hmulv, hslopev⟩ := q6k_mul_ok d i1
    (q2i_mul_fits15 hd hi1abs).1 (q2i_mul_fits15 hd hi1abs).2
  have hslope : slope.val = d.val * ((scales.val.getD group.val 0#u8).val &&& 15 : Nat) := by
    rw [hslopev, hi1v, hsccv, hscval]
  let i2 : I64 := UScalar.hcast .I64 mcc
  have hcast2 : I64.Insts.CoreConvertFromU8.from mcc = ok i2 := rfl
  have hi2v : i2.val = (mcc.val : ℤ) := q2k_hcast_u8_val mcc
  have hi2abs : |i2.val| ≤ 15 := by
    rw [hi2v, abs_of_nonneg (by positivity)]; exact_mod_cast hmcc_le
  obtain ⟨intercept, hmul2, hinterv⟩ := q6k_mul_ok dmin i2
    (q2i_mul_fits15 hm hi2abs).1 (q2i_mul_fits15 hm hi2abs).2
  have hinter : intercept.val = dmin.val * ((scales.val.getD group.val 0#u8).val >>> 4 : Nat) := by
    rw [hinterv, hi2v, hmccv, hscval]
  have hm15 : |((((scales.val.getD group.val 0#u8).val &&& 15 : Nat) : ℤ))| ≤ 15 := by
    rw [abs_of_nonneg (by positivity)]
    have h : ((scales.val.getD group.val 0#u8).val &&& 15 : Nat) ≤ 15 := Nat.and_le_right
    exact_mod_cast h
  have hm4 : |((((scales.val.getD group.val 0#u8).val >>> 4 : Nat) : ℤ))| ≤ 15 := by
    rw [abs_of_nonneg (by positivity)]
    have hb : (scales.val.getD group.val 0#u8).val < 256 := U8.lt_succ_max _
    have h : ((scales.val.getD group.val 0#u8).val >>> 4 : Nat) ≤ 15 := by omega
    exact_mod_cast h
  have hslope54 : |slope.val| ≤ 2 ^ 54 := by
    rw [hslope, abs_mul]
    calc |d.val| * _ ≤ 2 ^ 48 * 15 := mul_le_mul hd hm15 (abs_nonneg _) (by norm_num)
      _ ≤ 2 ^ 54 := by norm_num
  have hinter54 : |intercept.val| ≤ 2 ^ 54 := by
    rw [hinter, abs_mul]
    calc |dmin.val| * _ ≤ 2 ^ 48 * 15 := mul_le_mul hm hm4 (abs_nonneg _) (by norm_num)
      _ ≤ 2 ^ 54 := by norm_num
  have hinv0 : q2kLaneInv vals vals codes slope.val intercept.val group.val 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ (j / 16 = group.val ∧ j % 16 < 0))]
  obtain ⟨b1, hloop2, vals1, hb1, hlaneinv⟩ := WP.spec_imp_exists
    (q2kLane_loop_spec vals codes slope intercept group hgroup
      (fun l hl => hcode (16 * group.val + l) (by omega)) hslope54 hinter54 hinv0)
  have hloop2' : dequant.dequant_q2_k_row_loop0_loop0_loop0 true none codes vals group slope
      intercept 0#usize = ok (true, none, vals1) := by rw [hloop2, hb1]
  obtain ⟨group2, haddg, hgroup2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := group) (y := 1#usize) (by scalar_tac))
  have hgroup2 : group2.val = group.val + 1 := by rw [hgroup2v]; simp
  refine ⟨true, none, vals1, group2, ?_, rfl, rfl, hgroup2, ?_⟩
  · unfold dequant.dequant_q2_k_row_loop0_loop0.body
    rw [if_pos (by scalar_tac : group < 16#usize), if_pos (by simp : (true : Bool) = true)]
    rw [hidx, bind_tc_ok, hscm, q2k_let_pair]
    simp only [hcast1, bind_tc_ok]
    simp only [hmulv, bind_tc_ok]
    simp only [hcast2, bind_tc_ok]
    simp only [hmul2, bind_tc_ok]
    change (do
        let (ok2, err1, valsX) ← dequant.dequant_q2_k_row_loop0_loop0_loop0 true none codes vals
          group slope intercept 0#usize
        let group1 ← group + 1#usize
        ok (ControlFlow.cont (ok2, err1, valsX, group1)))
      = ok (ControlFlow.cont (true, none, vals1, group2))
    rw [hloop2', haddg]
    simp only [bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    have hb := hlaneinv.2.2.2 j hj
    by_cases hlt : j / 16 < group.val
    · rw [hb, if_neg (by omega : ¬ (j / 16 = group.val ∧ j % 16 < 16)),
        hinv.2.2.2 j hj, if_pos hlt, if_pos (by omega)]
    · rw [hb]
      by_cases heq : j / 16 = group.val
      · rw [if_pos ⟨heq, by omega⟩, if_pos (by omega)]
        unfold q2kLaneVal q2kBlockVal
        have hidx : 16 * (j / 16) + j % 16 = j := by omega
        rw [hslope, hinter, ← heq, hidx]
      · rw [if_neg (fun h => heq h.1), hinv.2.2.2 j hj, if_neg hlt, if_neg (by omega)]

/-- The group loop: `loop.spec_decr_nat` with `q2kGroupInv` and measure `16 − group`.

Complexity: 16 groups, each with two checked products and one 16-lane pass; O(1) array writes. -/
theorem q2kGroup_loop_spec (d dmin : I64) (scales : Array Std.U8 16#usize)
    (codes : Array Std.U8 256#usize) (vals0 : Array I64 256#usize)
    (hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 3)
    (hd : |d.val| ≤ 2 ^ 48) (hm : |dmin.val| ≤ 2 ^ 48)
    (hinv0 : q2kGroupInv d dmin scales codes vals0 vals0 0 true none) :
    WP.spec (dequant.dequant_q2_k_row_loop0_loop0 true none d dmin scales codes vals0 0#usize)
      (fun b => ∃ vals : Array I64 256#usize,
        b = (true, none, vals) ∧ q2kGroupInv d dmin scales codes vals0 vals 16 true none) := by
  unfold dequant.dequant_q2_k_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Option error.EngineError × Array I64 256#usize × Usize =>
      16 - s.2.2.2.val)
    (inv := fun s => q2kGroupInv d dmin scales codes vals0 s.2.2.1 s.2.2.2.val s.1 s.2.1)
  · intro s hs'
    rcases s with ⟨ok1, err, vals, group⟩
    dsimp only at hs' ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs'
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    rw [hok', herr']
    by_cases hlt : group.val < 16
    · have hsclt : group.val < scales.val.length := by rw [Array.property scales]; exact hlt
      obtain ⟨ok3, err2, vals1, group2, hbody, hok3, herr2, hgroup2, hinv2⟩ :=
        q2kGroup_body_step d dmin scales codes vals0 vals group hlt hsclt hcode hd hm
          ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hgroup2] using hinv2, ?_⟩
      show 16 - group2.val < 16 - group.val
      omega
    · have hnot : ¬ (group < 16#usize) := fun hc => hlt ((UScalar.lt_equiv group 16#usize).1 hc)
      have hbody : dequant.dequant_q2_k_row_loop0_loop0.body d dmin scales codes true none vals group
          = ok (ControlFlow.done (true, none, vals)) := by
        rw [dequant.dequant_q2_k_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨vals, rfl, ?_⟩
      have hgroup16 : group.val = 16 := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      intro j hj
      rw [hrec j hj, hgroup16]
  · exact hinv0

/-! ## Q2_K block-window byte reading -/

/-- The 84-byte window is `(drop (84·b)).take 84`. -/
lemma q2kBlockBytes_slice (raw : Slice U8) (b : Nat) :
    q2kBlockBytes raw b = raw.val.slice (84 * b) (84 * b + 84) := by
  unfold q2kBlockBytes List.slice
  congr 1
  omega

/-- `f16LeHalf` of the 84-byte window at byte `80` is the raw half-word at `s + 80`. -/
lemma q2f16LeHalf_slice (raw : Slice U8) (s : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 84)) (he : s + 84 ≤ raw.val.length) (j : Nat)
    (hj : j + 1 < 84) :
    f16LeHalf blk j = f16LeHalf raw (s + j) := by
  unfold f16LeHalf
  congr 2
  rw [h, getD_slice raw.val s (s + 84) j he (by omega),
    getD_slice raw.val s (s + 84) (j + 1) he (by omega)]
  simp only [Nat.add_assoc]

/-- The scale byte `m` of a block window is the raw byte at `84·b + m`. -/
lemma q2kScList_getD (raw : Slice U8) (b m : Nat) (hm : m < 16) :
    ((q2kScList raw b).getD m 0#u8).val = (raw.val.getD (84 * b + m) 0#u8).val := by
  unfold q2kScList
  rw [getD_drop_take raw.val (84 * b) 16 m 0#u8 hm]

/-- The decoded codes list's entry `i` is `q2CodesRef` of the block bytes. -/
lemma q2kCodesList_getD (raw : Slice U8) (b i : Nat) (hi : i < 256) :
    ((q2kCodesList raw b).getD i 0#u8).val = q2CodesRef (q2kBlockBytes raw b) i := by
  unfold q2kCodesList
  rw [List.getD_eq_getElem?_getD, List.getElem?_map]
  have h : (List.range 256)[i]? = some i := by
    rw [List.getElem?_eq_getElem (by simpa using hi)]
    simp
  rw [h]
  simp only [Option.map_some, Option.getD_some, U8.ofNatCore_val_eq]

/-- The scale array entry `k` of the window is the raw byte at `84·b + k`. -/
lemma q2kScalesRef_slice (raw : Slice U8) (b : Nat) (blk : Slice U8) (k : Nat)
    (h : blk.val = raw.val.slice (84 * b) (84 * b + 84))
    (he : 84 * b + 84 ≤ raw.val.length) (hk : k < 16) :
    q2kScalesRef blk.val k = (raw.val.getD (84 * b + k) 0#u8).val := by
  unfold q2kScalesRef
  rw [h, getD_slice raw.val (84 * b) (84 * b + 84) k he (by omega)]

/-- The raw-level `q2kBlockVal` on the block window equals the decoded-array reference. -/
lemma q2kBlockVal_bridge (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (hslice : blk.val = raw.val.slice (84 * b) (84 * b + 84))
    (he : 84 * b + 84 ≤ raw.val.length)
    (d dmin : I64) (sc : Array U8 16#usize) (codes : Array U8 256#usize)
    (hd : d.val = q2kD raw b) (hdmin : dmin.val = q2kDmin raw b)
    (hsc : ∀ k, k < 16 → (sc.val.getD k 0#u8).val = q2kScalesRef blk.val k)
    (hcv : ∀ i, i < 256 → (codes.val.getD i 0#u8).val = q2CodesRef blk.val i)
    (j : Nat) (hj : j < 256) :
    q2kBlockVal d.val dmin.val sc.val codes.val j =
      q2kBlockVal (q2kD raw b) (q2kDmin raw b) (q2kScList raw b) (q2kCodesList raw b) j := by
  have hsc' : ∀ m, m < 16 → (sc.val.getD m 0#u8).val = ((q2kScList raw b).getD m 0#u8).val := by
    intro m hm
    rw [hsc m hm, q2kScalesRef_slice raw b blk m hslice he hm, q2kScList_getD raw b m hm]
  have hcodes' : ∀ i, i < 256 →
      (codes.val.getD i 0#u8).val = ((q2kCodesList raw b).getD i 0#u8).val := by
    intro i hi
    rw [hcv i hi, q2kCodesList_getD raw b i hi, q2kBlockBytes_slice raw b, ← hslice]
  have hsub16 : j / 16 < 16 := by omega
  unfold q2kBlockVal
  rw [hd, hdmin, hsc' (j / 16) hsub16, hcodes' j hj]

/-- The block-slice window: `q2kBlockVals` entry `i` is `q2kBlockVal` at `i`. -/
lemma q2kBlockVals_getD (raw : Slice U8) (b i : Nat) (hi : i < 256) :
    (q2kBlockVals raw b).getD i 0 =
      q2kBlockVal (q2kD raw b) (q2kDmin raw b) (q2kScList raw b) (q2kCodesList raw b) i := by
  unfold q2kBlockVals
  rw [List.getD_eq_getElem?_getD]
  have h : (List.range 256)[i]? = some i := by
    rw [List.getElem?_eq_getElem (by simpa using hi)]
    simp
  rw [List.getElem?_map, h]
  rfl

/-- `q2kWeight` at the global index of block `b`, entry `i`. -/
lemma q2kWeight_eq (raw : Slice U8) (b i : Nat) (hi : i < 256) :
    q2kWeight raw (256 * b + i) = (q2kBlockVals raw b).getD i 0 := by
  unfold q2kWeight
  rw [show (256 * b + i) / 256 = b by omega, show (256 * b + i) % 256 = i by omega]

/-- Appending the next `take` raw weights advances the row prefix by `take`. -/
lemma q2kRowValues_append (raw : Slice U8) (m take : Nat) :
    q2kRowValues raw m ++ (List.range take).map (fun j => q2kWeight raw (m + j)) =
      q2kRowValues raw (m + take) := by
  unfold q2kRowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-! ## The Q2_K row success domain and loop invariant -/

/-- The success domain of one 84-byte block `b`: the window is the raw slice `[84·b, 84·b+84)`
and `d`/`dmin` are finite. The checked-`i64` fits follow from `q2f16Value_abs_lt` and
`q2CodesRef_lt_4` in the body step. -/
def q2kBlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  ∃ blk : Slice U8,
    dequant.block_slice raw b 84#usize = ok (.Ok blk) ∧
    blk.val = raw.val.slice (84 * b.val) (84 * b.val + 84) ∧
    84 * b.val + 84 ≤ raw.val.length ∧
    84 ≤ blk.val.length ∧
    f16Exp (f16LeHalf blk 80) ≠ 31 ∧
    f16Exp (f16LeHalf blk 82) ≠ 31

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (256·block) n` raw weights, with the success flag set. -/
def q2kRowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q2kRowValues raw (Nat.min (256 * block.val) n.val)

/-- One block-loop body step on the success domain: decode block `block`, run the group loop,
append `take = min 256 (n - 256·block)` values and advance. -/
theorem q2kBlock_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : q2kBlockDomain raw block)
    (hcap : out.val.length + (n.val - 256 * block.val) ≤ Usize.max)
    (hinv : q2kRowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.dequant_q2_k_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q2kRowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨blk, hblk, hslice, hbnd, h84, hd, hdmin⟩ := hdom
  obtain ⟨d, dmin, sc, codes, hq2, hdv, hdminv, hscv, hcv⟩ :=
    q2_k_block_spec raw block blk hblk h84 hd hdmin
  have hdv' : d.val = q2kD raw block.val := by
    rw [hdv]
    unfold q2kD
    rw [q2f16LeHalf_slice raw (84 * block.val) blk hslice hbnd 80 (by norm_num)]
  have hdminv' : dmin.val = q2kDmin raw block.val := by
    rw [hdminv]
    unfold q2kDmin
    rw [q2f16LeHalf_slice raw (84 * block.val) blk hslice hbnd 82 (by norm_num)]
  have hfin0 : f16Exp (f16LeHalf raw (84 * block.val + 80)) ≠ 31 := by
    have h := q2f16LeHalf_slice raw (84 * block.val) blk hslice hbnd 80 (by norm_num)
    rw [← h]; exact hd
  have hfin2 : f16Exp (f16LeHalf raw (84 * block.val + 82)) ≠ 31 := by
    have h := q2f16LeHalf_slice raw (84 * block.val) blk hslice hbnd 82 (by norm_num)
    rw [← h]; exact hdmin
  have hd48 : |d.val| ≤ 2 ^ 48 := by
    rw [hdv']; unfold q2kD
    exact le_of_lt (q2f16Value_abs_lt _ hfin0)
  have hm48 : |dmin.val| ≤ 2 ^ 48 := by
    rw [hdminv']; unfold q2kDmin
    exact le_of_lt (q2f16Value_abs_lt _ hfin2)
  have hcode : ∀ i, i < 256 → (codes.val.getD i 0#u8).val ≤ 3 := by
    intro i hi
    rw [hcv i hi]
    have := q2CodesRef_lt_4 blk.val i
    omega
  have hltmul : 256 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hblock)
  have hinv0 : q2kGroupInv d dmin sc codes (Array.repeat 256#usize 0#i64)
      (Array.repeat 256#usize 0#i64) 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ j / 16 < 0)]
  obtain ⟨b1, hloop, vals1, hb1, hgrp⟩ := WP.spec_imp_exists
    (q2kGroup_loop_spec d dmin sc codes (Array.repeat 256#usize 0#i64) hcode hd48 hm48 hinv0)
  have hloop' : dequant.dequant_q2_k_row_loop0_loop0 true none d dmin sc codes
      (Array.repeat 256#usize 0#i64) 0#usize = ok (true, none, vals1) := by rw [hloop, hb1]
  obtain ⟨i, hiv, hivv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := block) (y := quant.QK_K) (by simp only [quant.QK_K]; scalar_tac))
  have hivv' : i.val = 256 * block.val := by rw [hivv]; simp [quant.QK_K]; ring
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists
    (Usize.sub_spec (x := n) (y := i) (by rw [hivv']; omega))
  have hi1v' : i1.val = n.val - 256 * block.val := by rw [hi1v.1, hivv']
  obtain ⟨take, htake, htakev⟩ := q6k_min_ok quant.QK_K i1
  have htakev' : take.val = Nat.min 256 (n.val - 256 * block.val) := by
    rw [htakev, hi1v']; simp [quant.QK_K]
  obtain ⟨block1, hblock1a, hblock1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := block) (y := 1#usize) (by scalar_tac))
  have hblock1v' : block1.val = block.val + 1 := by rw [hblock1v]; simp
  have htake_le : take.val ≤ 256 := by rw [htakev']; exact min_le_left _ _
  have htake_X : take.val ≤ n.val - 256 * block.val := by rw [htakev']; exact min_le_right _ _
  have hcap' : out.val.length + take.val ≤ Usize.max := by omega
  obtain ⟨out1, hpush, hpushv⟩ := WP.spec_imp_exists
    (q6k_push_prefix_spec out vals1 take htake_le hcap')
  have hgrp_j : ∀ j, j < 256 → (vals1.val.getD j 0#i64).val =
      q2kBlockVal d.val dmin.val sc.val codes.val j := by
    intro j hj
    have h := hgrp.2.2.2 j hj
    rwa [if_pos (by omega : j / 16 < 16)] at h
  have hgrp_raw : ∀ j, j < 256 → (vals1.val.getD j 0#i64).val =
      (q2kBlockVals raw block.val).getD j 0 := by
    intro j hj
    rw [hgrp_j j hj, q2kBlockVals_getD raw block.val j hj]
    exact q2kBlockVal_bridge raw block.val blk hslice hbnd d dmin sc codes hdv' hdminv' hscv hcv j hj
  have hmap : (List.range take.val).map (fun j => (vals1.val.getD j 0#i64).val) =
      (List.range take.val).map (fun j => q2kWeight raw (256 * block.val + j)) := by
    apply List.map_congr_left
    intro j hj
    have hj' : j < take.val := List.mem_range.mp hj
    have hj256 : j < 256 := by omega
    rw [hgrp_raw j hj256, q2kWeight_eq raw block.val j hj256]
  have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 256 * block.val + take.val = Nat.min (256 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals out1.val = dequantVals out0.val ++
      q2kRowValues raw (Nat.min (256 * block1.val) n.val) := by
    rw [hblock1v']
    rw [hpushv, hmap, hinv.2.2.2, hmin, List.append_assoc, q2kRowValues_append, hsum]
  refine ⟨out1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.dequant_q2_k_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq2]
  simp only [bind_tc_ok]
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  change (do
    let (ok2, err1, vals1) ←
      dequant.dequant_q2_k_row_loop0_loop0 true none d dmin sc codes
        (Array.repeat 256#usize 0#i64) 0#usize
    let out1 ← if ok2 = true then dequant.push_prefix out vals1 take else ok out
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, ok2, err1)))
    = ok (ControlFlow.cont (out1, block1, true, none))
  rw [hloop']
  simp only [bind_tc_ok]
  change (do
    let out1 ← if (true : Bool) = true then dequant.push_prefix out vals1 take else ok out
    let block1 ← block + 1#usize
    ok (ControlFlow.cont (out1, block1, true, none)))
    = ok (ControlFlow.cont (out1, block1, true, none))
  rw [hpush]
  rw [hblock1a]
  rw [if_pos rfl]
  simp only [bind_tc_ok]

/-- The block loop: `loop.spec_decr_nat` with `q2kRowInv` and measure `blocks - block`, on the
success domain `q2kBlockDomain` for every block below `blocks`.

Complexity: `blocks` iterations, each an 84-byte read plus 16 groups of 16 lanes; the measure is
`blocks - block`. -/
theorem q2kRow_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : ∀ b : Usize, b.val < blocks.val → q2kBlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_q2_k_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q2kRowValues raw n.val) := by
  unfold dequant.dequant_q2_k_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q2kRowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
  · intro s hs
    rcases s with ⟨out, block, ok1, err⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    rw [hok, herr]
    by_cases hlt : block.val < blocks.val
    · have hltmul : 256 * block.val < n.val :=
        lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hlt)
      have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
        Nat.min_eq_left (le_of_lt hltmul)
      have hlen_out : out.val.length = out0.val.length + 256 * block.val := by
        have h := congrArg List.length hrec
        rw [hmin] at h
        simpa [dequantVals, q2kRowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 256 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q2kBlock_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.dequant_q2_k_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_q2_k_row_loop0.body, if_neg hnot]
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨trivial, ?_⟩
      have hblock_eq : block.val = blocks.val := by omega
      rw [hrec, hblock_eq, hblocks]
      have hminend : Nat.min (256 * ((n.val + 255) / 256)) n.val = n.val :=
        Nat.min_eq_right (le_mul_div_ceil_256 n.val)
      rw [hminend]
  · refine ⟨rfl, rfl, by simp, ?_⟩
    have h0 : Nat.min (256 * (0#usize).val) n.val = 0 := by simp
    rw [h0]
    simp [q2kRowValues]

/-- **`Q2_K` row law.** On the block success domain, the extracted `dequant_q2_k_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `q2kRowValues raw n` to the output. -/
theorem dequant_q2_k_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 255) / 256 → q2kBlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q2_k_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q2kRowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_256_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q2kRow_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.dequant_q2_k_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## Joint non-vacuity witness

A concrete 84-byte block with `d = 1.0` (`0x3C00` little-endian at byte 80) and `dmin = 0.0`
(bytes 82–83 zero); the scales and codes are zero. It satisfies every `q2kBlockDomain` conjunct at
block `0`, so the row domain is not vacuous. -/

/-- The witness block bytes: 84 bytes, `d = 0x3C00`, `dmin = 0`, zero scales and codes. -/
def q2kWitBytes : List U8 :=
  List.replicate 80 0#u8 ++ [0#u8, 60#u8, 0#u8, 0#u8]

theorem q2kWitBytes_length : q2kWitBytes.length = 84 := by
  simp [q2kWitBytes]

/-- The witness block as a slice. -/
def q2kWitRaw : Slice U8 := Slice.from q2kWitBytes (by rw [q2kWitBytes_length]; scalar_tac)

theorem q2kWitRaw_val : q2kWitRaw.val = q2kWitBytes := Slice.from_val _ _

theorem q2kWit_f16LeHalf_80 : f16LeHalf q2kWitRaw 80 = 15360#u16 := by
  unfold f16LeHalf
  rw [q2kWitRaw_val]
  decide

theorem q2kWit_f16LeHalf_82 : f16LeHalf q2kWitRaw 82 = 0#u16 := by
  unfold f16LeHalf
  rw [q2kWitRaw_val]
  decide

/-- **Joint non-vacuity of `q2kBlockDomain` at block `0`.** -/
theorem q2kBlockDomain_witness : q2kBlockDomain q2kWitRaw 0#usize := by
  obtain ⟨blk, hblk, hslice⟩ := block_slice_ok q2kWitRaw 0#usize 84#usize (by
    rw [q2kWitRaw_val, q2kWitBytes_length]; norm_num)
  refine ⟨blk, hblk, ?_, ?_, ?_, ?_, ?_⟩
  · simpa using hslice
  · rw [q2kWitRaw_val, q2kWitBytes_length]; norm_num
  · rw [hslice, List.slice_length]; simp [q2kWitRaw_val, q2kWitBytes_length]
  · intro hc
    have hs : f16LeHalf blk 80 = f16LeHalf q2kWitRaw 80 := by
      simpa using q2f16LeHalf_slice q2kWitRaw 0 blk (by simpa using hslice)
        (by rw [q2kWitRaw_val, q2kWitBytes_length]) 80 (by norm_num)
    rw [hs, q2kWit_f16LeHalf_80] at hc
    unfold f16Exp at hc
    norm_num at hc
  · intro hc
    have hs : f16LeHalf blk 82 = f16LeHalf q2kWitRaw 82 := by
      simpa using q2f16LeHalf_slice q2kWitRaw 0 blk (by simpa using hslice)
        (by rw [q2kWitRaw_val, q2kWitBytes_length]) 82 (by norm_num)
    rw [hs, q2kWit_f16LeHalf_82] at hc
    unfold f16Exp at hc
    norm_num at hc

end apeiron_word
