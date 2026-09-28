/-
Refinement of the extracted `Q6_K` dequantizer (`APEIRON-ALGEBRA-G3-DEQUANT-W2-T2`).

The Rust bodies are `dequant::q6_k_block` and `dequant::dequant_q6_k_row` in
`src/engine/dequant.rs`; the block accessors `quant::q6_k_d`, `quant::q6_k_scales` and
`quant::q6_k_codes` are in `src/format/quant.rs`. The decode is exact integer arithmetic over
Q32.32: `d` is a finite `f16` decoded exactly by `f16_to_fixed_exact`, each of the sixteen signed
8-bit sub-block scales and the 256 signed 6-bit codes (`q - 32`) is read as a signed byte, and each
weight is the checked `i64` product `d·scale_{i/16}·q_i` with no rounding.

The sub-block width is 16: `dequant_q6_k_row` writes `vals[16·group + lane]` for `group < 16`,
`lane < 16`, with `slope = d·scales[group]`, so weight `i` (of the 256 in a block) uses
`scale_{i/16}` and `codes[i]`. That is the law below; the block-level reader is the proved code
reader `Numeric.FormatRefinement.word_q6_k_codes_ok` plus the block scale reader proved here.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`. `decide` is used only on closed
literal facts, never on an extracted body.
-/
import Numeric.FormatRefinement
import Numeric.DequantRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Block-slice success

`dequant::block_slice` is `start = block·bb`, `end = start + bb`, then the range read; it refuses
with `QuantError.Truncated end (len raw)` only when `end` is past the payload. -/

/-- `checked_add` succeeds with the exact sum inside `usize`. -/
theorem dequant_usize_checked_add_some (x y : Usize) (h : x.val + y.val ≤ Usize.max) :
    ∃ z : Usize, Usize.checked_add x y = some z ∧ z.val = x.val + y.val := by
  have hs := Usize.checked_add_bv_spec x y
  cases hc : Usize.checked_add x y with
  | none => rw [hc] at hs; simp only at hs; omega
  | some z => rw [hc] at hs; exact ⟨z, rfl, hs.2.1⟩

/-- `block_slice` succeeds and returns exactly the `[block·bb, block·bb+bb)` window. -/
theorem block_slice_ok (raw : Slice U8) (block bb : Usize)
    (hlen : block.val * bb.val + bb.val ≤ raw.val.length) :
    ∃ blk : Slice U8,
      dequant.block_slice raw block bb = ok (.Ok blk) ∧
      blk.val = raw.val.slice (block.val * bb.val) (block.val * bb.val + bb.val) := by
  obtain ⟨m, hm, hmv⟩ := dequant_usize_checked_mul_some block bb (by
    have := raw.property; scalar_tac)
  obtain ⟨e, hce, hev⟩ := dequant_usize_checked_add_some m bb (by rw [hmv]; scalar_tac)
  have hev' : e.val = block.val * bb.val + bb.val := by rw [hev, hmv]
  have hcond : m ≤ e ∧ e.val ≤ raw.val.length := by
    refine ⟨?_, ?_⟩
    · rw [UScalar.le_equiv, hmv, hev']; omega
    · rw [hev']; exact hlen
  rw [dequant.block_slice, hm]
  simp only [lift, bind_tc_ok, core.option.Option.ok_or_some,
    core.result.Result.Insts.CoreOpsTry.branch]
  rw [hce]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  simp only [core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
  rw [if_pos hcond]
  simp only [bind_tc_ok, core.option.Option.ok_or_some]
  refine ⟨_, rfl, ?_⟩
  rw [Slice.from_val, hmv, hev']

/-! ## Q6_K references

| symbol | byte source (relative to block start) |
|---|---|
| `q6kD` | finite `f16` at 192 |
| `q6kScaleRef k` | signed byte at `192 + k`, `k < 16` |
| `q6kCodeRef i` | `q6CodesRef` of the 210-byte block at `i`, `i < 256` |

The row references read them through the 210-byte stride. -/

/-- The finite `f16` scale `d` of block `b`, read at byte `210·b + 208`. -/
def q6kD (raw : Slice U8) (b : Nat) : ℤ := f16Value (f16LeHalf raw (210 * b + 208))

/-- The `k`-th signed 8-bit sub-block scale of block `b`, byte `210·b + 192 + k`. -/
def q6kScaleRef (raw : Slice U8) (b k : Nat) : ℤ :=
  Int.bmod ((raw.val.getD (210 * b + 192 + k) 0#u8).val : ℤ) 256

/-- The `i`-th signed 6-bit code of block `b`, as `q6CodesRef` on the block bytes. -/
def q6kCodeRef (raw : Slice U8) (b i : Nat) : ℤ :=
  q6CodesRef ((raw.val.drop (210 * b)).take 210) i

/-- The exact weight `j` of a Q6_K row: `d_{j/256}·scale_{(j%256)/16}·q_{j%256}`. -/
def q6kWeight (raw : Slice U8) (j : Nat) : ℤ :=
  q6kD raw (j / 256) * q6kScaleRef raw (j / 256) ((j % 256) / 16) * q6kCodeRef raw (j / 256) (j % 256)

/-- The first `n` exact Q32.32 integers of a Q6_K row. -/
def q6kRowValues (raw : Slice U8) (n : Nat) : List ℤ :=
  (List.range n).map (q6kWeight raw)

/-! ## Block readers

`quant::q6_k_d` reads the little-endian `f16` at byte 208; `quant::q6_k_scales` writes the sixteen
signed bytes at 192..208. `quant::q6_k_codes` is the proved reader `word_q6_k_codes_ok`. -/

lemma getD_of_lt {l : List U8} {i : Nat} (h : i < l.length) : l.getD i 0#u8 = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- `quant::u16le` reads the little-endian `u16` at byte `i`. -/
theorem quant_u16le_ok (blk : Slice U8) (i : Usize) (h : i.val + 1 < blk.val.length) :
    quant.u16le blk i = ok (f16LeHalf blk i.val) := by
  obtain ⟨x, hx, hxv⟩ := WP.spec_imp_exists
    (Slice.index_usize_spec blk i (by show i.val < blk.val.length; omega))
  obtain ⟨i2, hi2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) (by scalar_tac))
  have hi2' : i2.val = i.val + 1 := by rw [hi2v]; simp
  obtain ⟨y, hy, hyv⟩ := WP.spec_imp_exists
    (Slice.index_usize_spec blk i2 (by show i2.val < blk.val.length; omega))
  have hgd1 : blk.val.getD i.val 0#u8 = blk.val[i.val] := getD_of_lt (by omega)
  have hgd2 : blk.val.getD (i.val + 1) 0#u8 = blk.val[i2.val] := by
    rw [← hi2']; exact getD_of_lt (by omega)
  rw [quant.u16le, hx]
  simp only [bind_tc_ok]
  rw [hi2]
  simp only [bind_tc_ok]
  rw [hy]
  simp only [bind_tc_ok]
  unfold f16LeHalf
  congr 2
  apply Aeneas.Std.Array.ext
  simp only [Array.make_val]
  rw [hxv, hyv, ← hgd1, ← hgd2]

/-- `quant::need` succeeds when the block has at least `n` bytes. -/
theorem quant_need_ok (blk : Slice U8) (n : Usize) (h : n.val ≤ blk.val.length) :
    quant.need blk n = ok (.Ok ()) := by
  unfold quant.need
  by_cases hc : Slice.len blk < n
  · have hlt : blk.val.length < n.val := by
      have h' := (UScalar.lt_equiv (Slice.len blk) n).1 hc
      simpa only [Slice.len_val, Slice.length] using h'
    simp only [hc, if_true]
    exact absurd hlt (not_lt.mpr h)
  · simp only [hc, if_false]

/-- `quant::q6_k_d` succeeds on a whole block and reads the `f16` bits at byte 208. -/
theorem q6_k_d_ok (blk : Slice U8) (hlen : 210 ≤ blk.val.length) :
    quant.q6_k_d blk = ok (.Ok (f16LeHalf blk 208)) := by
  have hneed := quant_need_ok blk 210#usize (by simpa using hlen)
  have hu := quant_u16le_ok blk 208#usize (by norm_num; omega)
  rw [quant.q6_k_d, quant_Q6_K_BLOCK_BYTES_eq]
  simp only [bind_tc_ok, hneed]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, quant.QK_K]
  obtain ⟨a, ha, hav⟩ := UScalar.div_spec (256#usize) (y := 2#usize) (by decide)
  obtain ⟨b, hb, hbv⟩ := UScalar.div_spec (256#usize) (y := 4#usize) (by decide)
  obtain ⟨c, hc, hcv⟩ := WP.spec_imp_exists (Usize.add_spec (x := a) (y := b) (by rw [hav, hbv]; scalar_tac))
  obtain ⟨d, hd, hdv⟩ := UScalar.div_spec (256#usize) (y := 16#usize) (by decide)
  obtain ⟨e', he', he'v⟩ := WP.spec_imp_exists (Usize.add_spec (x := c) (y := d) (by rw [hcv, hdv]; scalar_tac))
  have he208 : e' = 208#usize :=
    UScalar.eq_of_val_eq (by rw [he'v, hcv, hbv, hav, hdv]; norm_num)
  rw [ha]; simp only [bind_tc_ok]
  rw [hb]; simp only [bind_tc_ok]
  rw [hc]; simp only [bind_tc_ok]
  rw [hd]; simp only [bind_tc_ok]
  rw [he']; simp only [bind_tc_ok]
  rw [he208, hu]
  simp only [bind_tc_ok]
  rfl

/-! ## `quant::q6_k_scales`

`q6_k_scales` writes the sixteen signed bytes at block offset `192..208`. The loop-body read is
`blk[QK_K/2 + QK_K/4 + k]` and `QK_K/2 + QK_K/4 = 128 + 64 = 192`, so entry `k` is exactly
`q6kScaleRef blk 0 k` (`Int.bmod` of the byte). The cast `UScalar.hcast .I8` is the signed 8-bit
reading `Int.bmod x 256`. -/

/-- The reference value written by the scale loop at entry `k`. -/
lemma q6kScaleRef_eq_getD (blk : Slice U8) (k : Nat) :
    q6kScaleRef blk 0 k = Int.bmod ((blk.val.getD (192 + k) 0#u8).val : ℤ) 256 := rfl

/-- After `k` steps the accumulator holds the first `k` signed scales and zeros elsewhere. -/
def q6kScalesInv (blk : Slice U8) (out : Array I8 16#usize) (k : Nat) : Prop :=
  k ≤ 16 ∧ ∀ j, j < 16 → (out.val.getD j 0#i8).val = if j < k then q6kScaleRef blk 0 j else 0

/-- One scale-loop body step: read `blk[192 + k]`, signed-cast it, write entry `k`, advance. -/
theorem q6kScales_body_step (blk : Slice U8) (out : Array I8 16#usize) (k : Usize)
    (hlen : 210 ≤ blk.val.length) (hk : k.val < 16) (hinv : q6kScalesInv blk out k.val) :
    ∃ (out2 : Array I8 16#usize) (k2 : Usize),
      quant.q6_k_scales_loop.body blk out k = ok (ControlFlow.cont (out2, k2)) ∧
      q6kScalesInv blk out2 (k.val + 1) ∧ k2.val = k.val + 1 := by
  obtain ⟨i, hdiv2, hdiv2v⟩ := UScalar.div_spec (256#usize) (y := 2#usize) (by decide)
  obtain ⟨i1, hdiv4, hdiv4v⟩ := UScalar.div_spec (256#usize) (y := 4#usize) (by decide)
  have hi : i.val = 128 := by rw [hdiv2v]; norm_num
  have hi1 : i1.val = 64 := by rw [hdiv4v]; norm_num
  obtain ⟨i2, hadd1, hadd1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := i1) (by rw [hi, hi1]; scalar_tac))
  have hi2 : i2.val = 192 := by rw [hadd1v, hi, hi1]
  obtain ⟨i3, haddk, haddkv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i2) (y := k) (by rw [hi2]; scalar_tac))
  have hi3 : i3.val = 192 + k.val := by rw [haddkv, hi2]
  have hbound : i3.val < blk.val.length := by rw [hi3]; omega
  obtain ⟨i4, hidx, hidxv⟩ := WP.spec_imp_exists (Slice.index_usize_spec blk i3 hbound)
  have hi4v : i4.val = (blk.val.getD (192 + k.val) 0#u8).val := by
    have h' : i4.val = (blk.val.getD i3.val 0#u8).val := by
      rw [hidxv]
      exact congrArg UScalar.val (getD_of_lt hbound).symm
    rwa [hi3] at h'
  obtain ⟨out2, hupd, hout2⟩ := WP.spec_imp_exists
    (Array.update_spec out k (UScalar.hcast .I8 i4)
      (by show k.val < out.val.length; rw [Array.property out]; exact hk))
  obtain ⟨k2, hkadd, hk2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := k) (y := 1#usize) (by scalar_tac))
  have hk2 : k2.val = k.val + 1 := by rw [hk2v]; simp
  refine ⟨out2, k2, ?_, ?_, hk2⟩
  · rw [quant.q6_k_scales_loop.body]
    rw [if_pos (by scalar_tac)]
    simp only [quant.QK_K, hdiv2, hdiv4, hadd1, haddk, hidx, lift, hupd, hkadd, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    intro j hj
    have hout_prop : out.val.length = 16 := Array.property out
    rw [hout2, Array.set_val_eq, getD_set]
    by_cases hjk : j = k.val
    · rw [if_pos ⟨hjk, by rw [hout_prop]; exact hk⟩, hjk, if_pos (by omega)]
      rw [UScalar.hcast_val_eq, IScalarTy.I8_numBits_eq, hi4v, q6kScaleRef_eq_getD]
      norm_num
    · rw [if_neg (fun h => hjk h.1), hinv.2 j hj]
      by_cases hlt : j < k.val
      · rw [if_pos hlt, if_pos (by omega)]
      · rw [if_neg hlt, if_neg (by omega)]

/-- The scale loop: `loop.spec_decr_nat` with `q6kScalesInv` and measure `16 - k`.

Complexity: 16 iterations, each O(1) (one byte read, one signed cast, one array write); O(1) writes. -/
theorem q6kScales_loop_spec (blk : Slice U8) (out0 : Array I8 16#usize)
    (hlen : 210 ≤ blk.val.length) (hinv0 : q6kScalesInv blk out0 0) :
    WP.spec (quant.q6_k_scales_loop blk out0 0#usize) (fun out => q6kScalesInv blk out 16) := by
  unfold quant.q6_k_scales_loop
  apply loop.spec_decr_nat
    (measure := fun s : Array I8 16#usize × Usize => 16 - s.2.val)
    (inv := fun s => q6kScalesInv blk s.1 s.2.val)
  · rintro ⟨out, k⟩ hinv'
    dsimp only at hinv' ⊢
    by_cases hlt : k.val < 16
    · obtain ⟨out2, k2, hbody, hinv2, hk2⟩ := q6kScales_body_step blk out k hlen hlt hinv'
      simp only [hbody]
      refine ⟨by simpa [hk2] using hinv2, ?_⟩
      show 16 - k2.val < 16 - k.val
      omega
    · have hnot : ¬ (k < 16#usize) := fun hc => hlt ((UScalar.lt_equiv k 16#usize).1 hc)
      have hbody : quant.q6_k_scales_loop.body blk out k = ok (ControlFlow.done out) := by
        rw [quant.q6_k_scales_loop.body, if_neg hnot]
      simp only [hbody]
      have hk16 : k.val = 16 := by have := hinv'.1; omega
      simpa [hk16] using hinv'
  · exact hinv0

/-- **`Q6_K` scale reader.** On a whole block the extracted `q6_k_scales` succeeds and returns the
sixteen signed bytes at block offset `192 + k` as `i8` words. -/
theorem q6_k_scales_ok (blk : Slice U8) (hlen : 210 ≤ blk.val.length) :
    ∃ v : Array I8 16#usize,
      quant.q6_k_scales blk = ok (core.result.Result.Ok v) ∧
      ∀ k, k < 16 → (v.val.getD k 0#i8).val = q6kScaleRef blk 0 k := by
  have hneed := quant_need_ok blk 210#usize (by simpa using hlen)
  have hspec : quant.q6_k_scales blk ⦃ r => ∃ v : Array I8 16#usize,
      r = (core.result.Result.Ok v : core.result.Result (Array I8 16#usize) quant.QuantError) ∧
      ∀ k, k < 16 → (v.val.getD k 0#i8).val = q6kScaleRef blk 0 k ⦄ := by
    have hrepeat : q6kScalesInv blk (Array.repeat 16#usize 0#i8) 0 := by
      refine ⟨by norm_num, ?_⟩
      intro j hj
      rw [Array.repeat_val]
      simp only [List.getD, List.getElem?_replicate]
      rw [if_pos (by simpa using hj), if_neg (by omega : ¬ j < 0)]
      rfl
    unfold quant.q6_k_scales
    rw [quant_Q6_K_BLOCK_BYTES_eq]
    simp only [bind_tc_ok, hneed]
    simp only [core.result.Result.Insts.CoreOpsTry.branch]
    step*
    refine WP.spec_bind (q6kScales_loop_spec blk (Array.repeat 16#usize 0#i8) hlen hrepeat) ?_
    intro out1 hout1
    refine ⟨out1, rfl, ?_⟩
    intro k hk
    rw [hout1.2 k hk, if_pos hk]
  rcases WP.spec_imp_exists hspec with ⟨r, hr, hv⟩
  rcases hv with ⟨v, hrv, hpt⟩
  exact ⟨v, by rw [hr, hrv], hpt⟩

/-! ## `dequant::q6_k_block`

On a whole 210-byte block with finite `d` the block returns the exact Q32.32 scale, the sixteen
signed sub-block scales and the 256 signed codes. Mirrors `Numeric.DequantQ4K.q4_k_block_spec`:
the block is passed by its slice, so the references are stated over `blk`, not `raw`. -/

/-- **`Q6_K` block law.** On the block window returned by `block_slice` with finite `d`,
`q6_k_block` returns `(d, scales, codes)` with `d` the exact Q32.32 `f16` value, the sixteen
signed scales `q6kScaleRef blk 0`, and the 256 codes `q6CodesRef blk.val`. -/
theorem q6_k_block_ok (raw : Slice U8) (block : Usize) (blk : Slice U8)
    (hblk : dequant.block_slice raw block 210#usize = ok (.Ok blk))
    (h210 : 210 ≤ blk.val.length)
    (hd : f16Exp (f16LeHalf blk 208) ≠ 31) :
    ∃ (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize),
      dequant.q6_k_block raw block = ok (.Ok (d, scales, codes)) ∧
      d.val = f16Value (f16LeHalf blk 208) ∧
      (∀ k, k < 16 → (scales.val.getD k 0#i8).val = q6kScaleRef blk 0 k) ∧
      (∀ i, i < 256 → (codes.val.getD i 0#i8).val = q6CodesRef blk.val i) := by
  have hdok := q6_k_d_ok blk h210
  obtain ⟨d, hd_eq, hdv⟩ := f16_to_fixed_exact (f16LeHalf blk 208) hd
  obtain ⟨scales, hsc_eq, hsc_inv⟩ := q6_k_scales_ok blk h210
  obtain ⟨codes, hcodes_eq, hcodes⟩ := word_q6_k_codes_ok blk (by
    rw [UScalar.le_equiv]; simpa [Slice.len] using h210)
  have hcodes_eq' : quant.q6_k_codes blk = ok (.Ok codes) := hcodes_eq
  refine ⟨d, scales, codes, ?_, hdv, ?_, hcodes⟩
  · simp only [dequant.q6_k_block, quant_Q6_K_BLOCK_BYTES_eq, hblk, hdok, hd_eq, hsc_eq,
      hcodes_eq', core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  · intro k hk
    exact hsc_inv k hk

/-! ## `dequant::q6_k_block` loop scaffolding

The lane loop (`dequant_q6_k_row_loop0_loop0_loop0`) fills the sixteen entries `16·group + lane`
of the 256-entry value array with `slope·codes[16·group + lane]`, from `slope = d·scales[group]`.
`q6kLaneWeight` is the reference value, `q6kLaneInv` the accumulator invariant. -/

/-- Reference value of lane `l` of `group`: `slope` times the signed code at `16·group + l`. -/
def q6kLaneWeight (slope : I64) (codes : Array I8 256#usize) (group l : Nat) : ℤ :=
  slope.val * (codes.val.getD (16 * group + l) 0#i8).val

/-- The signed-`i8` default `getD` lemma for `List I8`. -/
lemma getD_of_lt_i8 {l : List I8} {i : Nat} (h : i < l.length) : l.getD i 0#i8 = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- Lane-loop invariant: lanes `0..lane-1` of `group` hold `q6kLaneWeight`, every other entry is
`vals0`. -/
def q6kLaneInv (vals0 vals : Array I64 256#usize) (codes : Array I8 256#usize) (slope : I64)
    (group lane : Nat) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ lane ≤ 16 ∧
  (∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j / 16 = group ∧ j % 16 < lane then q6kLaneWeight slope codes group (j % 16)
    else (vals0.val.getD j 0#i64).val)

lemma div16_mul_add (g l : Nat) (hl : l < 16) : (16 * g + l) / 16 = g := by omega
lemma mod16_mul_add (g l : Nat) (hl : l < 16) : (16 * g + l) % 16 = l := by omega

lemma eq_of_div16 (j g : Nat) (h : j / 16 = g) : j = 16 * g + j % 16 := by
  have := Nat.div_add_mod j 16
  omega

/-- `dequant::mul` on the success path returns the exact `i64` product. -/
theorem q6k_mul_ok (a b : I64)
    (hlo : I64.min ≤ a.val * b.val) (hhi : a.val * b.val ≤ I64.max) :
    ∃ z : I64, dequant.mul a b = ok (.Ok z) ∧ z.val = a.val * b.val := by
  have hspec := I64.checked_mul_bv_spec a b
  cases hc : core.num.checked_mul_IScalar a b with
  | none => rw [hc] at hspec; exact absurd ⟨hlo, hhi⟩ hspec
  | some z =>
      rw [hc] at hspec
      refine ⟨z, ?_, hspec.2.2.1⟩
      unfold dequant.mul
      rw [show I64.checked_mul a b = some z by simpa only [I64.checked_mul] using hc]
      simp only [lift, bind_tc_ok]
      rw [core.option.Option.ok_or_some]

/-- One lane-loop body step on the success path. -/
theorem q6kLane_body_step (vals0 vals : Array I64 256#usize) (codes : Array I8 256#usize)
    (slope : I64) (group lane : Usize) (hgroup : group.val < 16) (hlane : lane.val < 16)
    (hmul : I64.min ≤ slope.val * (codes.val.getD (16 * group.val + lane.val) 0#i8).val ∧
      slope.val * (codes.val.getD (16 * group.val + lane.val) 0#i8).val ≤ I64.max)
    (hinv : q6kLaneInv vals0 vals codes slope group.val lane.val true none) :
    ∃ (ok2 : Bool) (err1 : Option error.EngineError) (vals1 : Array I64 256#usize) (lane2 : Usize),
      dequant.dequant_q6_k_row_loop0_loop0_loop0.body codes group slope true none vals lane
          = ok (ControlFlow.cont (ok2, err1, vals1, lane2)) ∧
      ok2 = true ∧ err1 = none ∧ lane2.val = lane.val + 1 ∧
      q6kLaneInv vals0 vals1 codes slope group.val (lane.val + 1) ok2 err1 := by
  obtain ⟨i, hm16, hiv⟩ := WP.spec_imp_exists
    (Usize.mul_spec (x := 16#usize) (y := group) (by scalar_tac))
  have hiv' : i.val = 16 * group.val := by rw [hiv]; simp
  obtain ⟨i1, hadd, hi1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := lane) (by rw [hiv']; scalar_tac))
  have hi1' : i1.val = 16 * group.val + lane.val := by rw [hi1v, hiv']
  obtain ⟨i2, hidx, hi2v⟩ := WP.spec_imp_exists
    (Array.index_usize_spec codes i1
      (by show i1.val < codes.val.length; rw [Array.property codes, hi1']; scalar_tac))
  have hi2val : i2.val = (codes.val.getD (16 * group.val + lane.val) 0#i8).val := by
    have hlt : i1.val < codes.val.length := by rw [Array.property codes, hi1']; scalar_tac
    have h : i2.val = (codes.val.getD i1.val 0#i8).val := by
      rw [hi2v]
      exact congrArg (fun x : I8 => x.val) (getD_of_lt_i8 (l := codes.val) hlt).symm
    rwa [hi1'] at h
  obtain ⟨v, hmulv, hvval⟩ := q6k_mul_ok slope (core.convert.num.FromI64I8.from i2)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hi2val]; exact hmul.1)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hi2val]; exact hmul.2)
  obtain ⟨a, hupd, hau⟩ := WP.spec_imp_exists
    (Array.update_spec vals i1 v
      (by show i1.val < vals.val.length; rw [Array.property vals, hi1']; scalar_tac))
  obtain ⟨lane2, haddl, hlane2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := lane) (y := 1#usize) (by scalar_tac))
  have hlane2 : lane2.val = lane.val + 1 := by rw [hlane2v]; simp
  have hdiv : i1.val / 16 = group.val := by rw [hi1']; exact div16_mul_add group.val lane.val hlane
  have hmod : i1.val % 16 = lane.val := by rw [hi1']; exact mod16_mul_add group.val lane.val hlane
  refine ⟨true, none, a, lane2, ?_, rfl, rfl, hlane2, ?_⟩
  · rw [dequant.dequant_q6_k_row_loop0_loop0_loop0.body]
    rw [if_pos (by scalar_tac : lane < 16#usize)]
    simp only [if_true, hm16, hadd, hidx, lift, hmulv, hupd, haddl, bind_tc_ok]
    rfl
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    have hvals_len : vals.val.length = 256 := Array.property vals
    rw [hau, Array.set_val_eq, getD_set]
    by_cases hjk : j = i1.val
    · subst hjk
      rw [if_pos ⟨rfl, by rw [hvals_len]; rw [hi1']; omega⟩,
        if_pos ⟨hdiv, by omega⟩, hmod]
      unfold q6kLaneWeight
      rw [hvval, core.convert.num.FromI64I8.from_val_eq, hi2val]
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

/-- The lane loop: `loop.spec_decr_nat` with `q6kLaneInv` and measure `16 - lane`, on the
no-overflow path (`hmul` at every lane).

Complexity: 16 iterations, each O(1) (one array read, one checked multiply, one array write); the
array is 256 words, never resized. -/
theorem q6kLane_loop_spec (vals0 : Array I64 256#usize) (codes : Array I8 256#usize)
    (slope : I64) (group : Usize) (hgroup : group.val < 16)
    (hmul : ∀ l, l < 16 → I64.min ≤ slope.val * (codes.val.getD (16 * group.val + l) 0#i8).val ∧
      slope.val * (codes.val.getD (16 * group.val + l) 0#i8).val ≤ I64.max)
    (hinv0 : q6kLaneInv vals0 vals0 codes slope group.val 0 true none) :
    WP.spec (dequant.dequant_q6_k_row_loop0_loop0_loop0 true none codes vals0 group slope 0#usize)
      (fun b => ∃ vals : Array I64 256#usize,
        b = (true, none, vals) ∧ q6kLaneInv vals0 vals codes slope group.val 16 true none) := by
  unfold dequant.dequant_q6_k_row_loop0_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Option error.EngineError × Array I64 256#usize × Usize =>
      16 - s.2.2.2.val)
    (inv := fun s => q6kLaneInv vals0 s.2.2.1 codes slope group.val s.2.2.2.val s.1 s.2.1)
  · intro s hs
    rcases s with ⟨ok1, err, vals, lane⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    rw [hok', herr']
    by_cases hlt : lane.val < 16
    · obtain ⟨ok2, err1, vals1, lane2, hbody, hok2, herr1, hlane2, hinv2⟩ :=
        q6kLane_body_step vals0 vals codes slope group lane hgroup hlt (hmul lane.val hlt)
          ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hlane2] using hinv2, ?_⟩
      show 16 - lane2.val < 16 - lane.val
      omega
    · have hnot : ¬ (lane < 16#usize) := fun hc => hlt ((UScalar.lt_equiv lane 16#usize).1 hc)
      have hbody : dequant.dequant_q6_k_row_loop0_loop0_loop0.body codes group slope true none vals lane
          = ok (ControlFlow.done (true, none, vals)) := by
        rw [dequant.dequant_q6_k_row_loop0_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨vals, rfl, ?_⟩
      have hlane16 : lane.val = 16 := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      intro j hj
      rw [hrec j hj, hlane16]
  · exact hinv0

/-! ## `dequant::q6_k_block` group loop

`dequant_q6_k_row_loop0_loop0` runs the sixteen groups; group `g` computes `slope = d·scales[g]`
and invokes the lane loop. `q6kGroupInv` accumulates the reference value
`d·scales[j/16]·codes[j]` for every `j` whose group is complete. -/

/-- Group-loop invariant: every entry whose group is below `group` holds
`d·scales[j/16]·codes[j]`; every other entry is `vals0`. -/
def q6kGroupInv (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize)
    (vals0 vals : Array I64 256#usize) (group : Nat) (ok1 : Bool) (err : Option error.EngineError) :
    Prop :=
  ok1 = true ∧ err = none ∧ group ≤ 16 ∧
  (∀ j, j < 256 → (vals.val.getD j 0#i64).val =
    if j / 16 < group then
      d.val * (scales.val.getD (j / 16) 0#i8).val * (codes.val.getD j 0#i8).val
    else (vals0.val.getD j 0#i64).val)

/-- One group-loop body step on the no-overflow path. -/
theorem q6kGroup_body_step (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize)
    (vals0 vals : Array I64 256#usize) (group : Usize) (hgroup : group.val < 16)
    (hsclt : group.val < scales.val.length)
    (hmul : I64.min ≤ d.val * (scales.val.getD group.val 0#i8).val ∧
      d.val * (scales.val.getD group.val 0#i8).val ≤ I64.max)
    (hlane : ∀ l, l < 16 →
      I64.min ≤ d.val * (scales.val.getD group.val 0#i8).val
          * (codes.val.getD (16 * group.val + l) 0#i8).val ∧
      d.val * (scales.val.getD group.val 0#i8).val
          * (codes.val.getD (16 * group.val + l) 0#i8).val ≤ I64.max)
    (hinv : q6kGroupInv d scales codes vals0 vals group.val true none) :
    ∃ (ok3 : Bool) (err2 : Option error.EngineError) (vals1 : Array I64 256#usize) (group2 : Usize),
      dequant.dequant_q6_k_row_loop0_loop0.body d scales codes true none vals group
          = ok (ControlFlow.cont (ok3, err2, vals1, group2)) ∧
      ok3 = true ∧ err2 = none ∧ group2.val = group.val + 1 ∧
      q6kGroupInv d scales codes vals0 vals1 (group.val + 1) ok3 err2 := by
  obtain ⟨sc, hidx, hscv⟩ := WP.spec_imp_exists (Array.index_usize_spec scales group hsclt)
  have hscval : sc.val = (scales.val.getD group.val 0#i8).val := by
    rw [hscv]
    exact congrArg (fun x : I8 => x.val) (getD_of_lt_i8 (l := scales.val) hsclt).symm
  obtain ⟨slope, hmulv, hslopev⟩ := q6k_mul_ok d (core.convert.num.FromI64I8.from sc)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hscval]; exact hmul.1)
    (by rw [core.convert.num.FromI64I8.from_val_eq, hscval]; exact hmul.2)
  have hslope : slope.val = d.val * (scales.val.getD group.val 0#i8).val := by
    rw [hslopev, core.convert.num.FromI64I8.from_val_eq, hscval]
  have hinv0 : q6kLaneInv vals vals codes slope group.val 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ (j / 16 = group.val ∧ j % 16 < 0))]
  obtain ⟨b1, hloop2, vals1, hb1, hlaneinv⟩ := WP.spec_imp_exists
    (q6kLane_loop_spec vals codes slope group hgroup
      (fun l hl => by rw [hslope]; exact hlane l hl) hinv0)
  have hloop2' : dequant.dequant_q6_k_row_loop0_loop0_loop0 true none codes vals group slope 0#usize
      = ok (true, none, vals1) := by rw [hloop2, hb1]
  obtain ⟨group2, haddg, hgroup2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := group) (y := 1#usize) (by scalar_tac))
  have hgroup2 : group2.val = group.val + 1 := by rw [hgroup2v]; simp
  refine ⟨true, none, vals1, group2, ?_, rfl, rfl, hgroup2, ?_⟩
  · rw [dequant.dequant_q6_k_row_loop0_loop0.body]
    rw [if_pos (by scalar_tac : group < 16#usize)]
    simp only [if_true, hidx, lift, hmulv, haddg, bind_tc_ok]
    change (dequant.dequant_q6_k_row_loop0_loop0_loop0 true none codes vals group slope 0#usize
        >>= fun b => ok (ControlFlow.cont (b.1, b.2.1, b.2.2, group2)))
      = ok (ControlFlow.cont (true, none, vals1, group2))
    rw [hloop2']
    simp only [bind_tc_ok]
  · refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    have hb := hlaneinv.2.2.2 j hj
    by_cases hlt : j / 16 < group.val
    · rw [hb, if_neg (by omega : ¬ (j / 16 = group.val ∧ j % 16 < 16)),
        hinv.2.2.2 j hj, if_pos hlt, if_pos (by omega)]
    · rw [hb]
      by_cases heq : j / 16 = group.val
      · rw [if_pos ⟨heq, by omega⟩, if_pos (by omega)]
        unfold q6kLaneWeight
        rw [hslope, heq, ← eq_of_div16 j group.val heq]
      · rw [if_neg (fun h => heq h.1), hinv.2.2.2 j hj, if_neg hlt, if_neg (by omega)]

/-- The group loop: `loop.spec_decr_nat` with `q6kGroupInv` and measure `16 - group`, on the
no-overflow path.

Complexity: 16 groups, each running the 16-lane loop; O(1) array writes in total per group. -/
theorem q6kGroup_loop_spec (d : I64) (scales : Array I8 16#usize) (codes : Array I8 256#usize)
    (vals0 : Array I64 256#usize)
    (hmul : ∀ g, g < 16 → I64.min ≤ d.val * (scales.val.getD g 0#i8).val ∧
      d.val * (scales.val.getD g 0#i8).val ≤ I64.max)
    (hlane : ∀ g, g < 16 → ∀ l, l < 16 →
      I64.min ≤ d.val * (scales.val.getD g 0#i8).val
          * (codes.val.getD (16 * g + l) 0#i8).val ∧
      d.val * (scales.val.getD g 0#i8).val
          * (codes.val.getD (16 * g + l) 0#i8).val ≤ I64.max)
    (hinv0 : q6kGroupInv d scales codes vals0 vals0 0 true none) :
    WP.spec (dequant.dequant_q6_k_row_loop0_loop0 true none d scales codes vals0 0#usize)
      (fun b => ∃ vals : Array I64 256#usize,
        b = (true, none, vals) ∧ q6kGroupInv d scales codes vals0 vals 16 true none) := by
  unfold dequant.dequant_q6_k_row_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun s : Bool × Option error.EngineError × Array I64 256#usize × Usize =>
      16 - s.2.2.2.val)
    (inv := fun s => q6kGroupInv d scales codes vals0 s.2.2.1 s.2.2.2.val s.1 s.2.1)
  · intro s hs
    rcases s with ⟨ok1, err, vals, group⟩
    dsimp only at hs ⊢
    obtain ⟨hok, herr, hle, hrec⟩ := hs
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    rw [hok', herr']
    by_cases hlt : group.val < 16
    · have hsclt : group.val < scales.val.length := by
        rw [Array.property scales]; exact hlt
      obtain ⟨ok3, err2, vals1, group2, hbody, hok3, herr2, hgroup2, hinv2⟩ :=
        q6kGroup_body_step d scales codes vals0 vals group hlt hsclt (hmul group.val hlt)
          (hlane group.val hlt) ⟨rfl, rfl, hle, hrec⟩
      simp only [hbody]
      refine ⟨by simpa [hgroup2] using hinv2, ?_⟩
      show 16 - group2.val < 16 - group.val
      omega
    · have hnot : ¬ (group < 16#usize) := fun hc => hlt ((UScalar.lt_equiv group 16#usize).1 hc)
      have hbody : dequant.dequant_q6_k_row_loop0_loop0.body d scales codes true none vals group
          = ok (ControlFlow.done (true, none, vals)) := by
        rw [dequant.dequant_q6_k_row_loop0_loop0.body, if_neg hnot]
      simp only [hbody]
      refine ⟨vals, rfl, ?_⟩
      have hgroup16 : group.val = 16 := by omega
      refine ⟨rfl, rfl, by omega, ?_⟩
      intro j hj
      rw [hrec j hj, hgroup16]
  · exact hinv0

/-! ## `dequant::push_prefix`

`dequant::push_prefix` appends the first `take` entries of a 256-word value array to the output
vector. It is the appender every block dequantizer uses to write its `min 256 (n - block·256)`
values. Self-contained: no block or slice reasoning here. -/

/-- The signed-`i64` default `getD` lemma. -/
lemma getD_of_lt_i64 {l : List I64} {i : Nat} (h : i < l.length) : l.getD i 0#i64 = l[i] := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem h]
  rfl

/-- `push_prefix` accumulator invariant: the output is the initial output followed by the first
`idx` reference values. -/
def q6kPushInv (vals : Array I64 256#usize) (out0 out : alloc.vec.Vec I64) (take idx : Usize) :
    Prop :=
  idx.val ≤ take.val ∧
  dequantVals out.val = dequantVals out0.val ++
    (List.range idx.val).map (fun j => (vals.val.getD j 0#i64).val)

/-- One `push_prefix` loop-body step. -/
theorem q6kPush_body_step (vals : Array I64 256#usize) (out0 out : alloc.vec.Vec I64)
    (take idx : Usize) (htake : take.val ≤ 256) (hlt : idx.val < take.val)
    (hcap : out.val.length < Usize.max) (hinv : q6kPushInv vals out0 out take idx) :
    ∃ (out1 : alloc.vec.Vec I64) (idx1 : Usize),
      dequant.push_prefix_loop.body vals take out idx = ok (ControlFlow.cont (out1, idx1)) ∧
      q6kPushInv vals out0 out1 take idx1 ∧ idx1.val = idx.val + 1 := by
  obtain ⟨i, hidx, hiv⟩ := WP.spec_imp_exists
    (Array.index_usize_spec vals idx
      (by show idx.val < vals.val.length; rw [Array.property vals]; scalar_tac))
  have hival : i.val = (vals.val.getD idx.val 0#i64).val := by
    rw [hiv]
    exact congrArg (fun x : I64 => x.val)
      (getD_of_lt_i64 (l := vals.val)
        (by show idx.val < vals.val.length; rw [Array.property vals]; scalar_tac)).symm
  obtain ⟨out1, hpush, hpushv⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out i hcap)
  obtain ⟨idx1, hadd, hidx1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := idx) (y := 1#usize) (by scalar_tac))
  have hidx1 : idx1.val = idx.val + 1 := by rw [hidx1v]; simp
  refine ⟨out1, idx1, ?_, ?_, hidx1⟩
  · rw [dequant.push_prefix_loop.body, if_pos (by scalar_tac : idx < take)]
    simp only [hidx, hpush, hadd, bind_tc_ok]
  · refine ⟨by omega, ?_⟩
    rw [hpushv, hidx1, dequantVals, List.map_append, ← dequantVals, hinv.2,
      List.range_succ, List.map_append]
    simp only [List.map_cons, List.map_nil, hival, List.append_assoc]

/-- **`push_prefix` law.** On room for `take` more words and `take ≤ 256`, the extracted
`push_prefix` succeeds and appends exactly the first `take` reference values of `vals`. -/
theorem q6k_push_prefix_spec (out : alloc.vec.Vec I64) (vals : Array I64 256#usize) (take : Usize)
    (htake : take.val ≤ 256) (hcap : out.val.length + take.val ≤ Usize.max) :
    WP.spec (dequant.push_prefix out vals take)
      (fun out' => dequantVals out'.val = dequantVals out.val ++
        (List.range take.val).map (fun j => (vals.val.getD j 0#i64).val)) := by
  unfold dequant.push_prefix dequant.push_prefix_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize => take.val - s.2.val)
    (inv := fun s => q6kPushInv vals out s.1 take s.2)
  · intro s hs
    rcases s with ⟨o, idx⟩
    dsimp only at hs ⊢
    by_cases hlt : idx.val < take.val
    · have ho_len : o.val.length = out.val.length + idx.val := by
        have := congrArg List.length hs.2
        simp [dequantVals, List.length_append, List.length_map, List.length_range] at this
        omega
      have hcapo : o.val.length < Usize.max := by omega
      obtain ⟨o1, idx1, hbody, hinv2, hidx1⟩ :=
        q6kPush_body_step vals out o take idx htake hlt hcapo hs
      simp only [hbody]
      refine ⟨hinv2, ?_⟩
      show take.val - idx1.val < take.val - idx.val
      omega
    · have hnot : ¬ (idx < take) := fun hc => hlt ((UScalar.lt_equiv idx take).1 hc)
      have hbody : dequant.push_prefix_loop.body vals take o idx = ok (ControlFlow.done o) := by
        rw [dequant.push_prefix_loop.body, if_neg hnot]
      simp only [hbody]
      have hidx_take : idx.val = take.val := by have := hs.1; omega
      simpa [hidx_take] using hs.2
  · refine ⟨by simp, ?_⟩
    simp

/-! ## Block-window byte reading

`block_slice_ok` returns `blk` with `blk.val = raw.val.slice s (s + 210)`. The `getD` of a
`List.slice` is the `getD` of the payload at the shifted index, which is what the raw-level
references need. -/

/-- `getD` of a `List.slice` is the payload `getD` at the shifted index. -/
lemma getD_slice (l : List U8) (s e i : Nat) (he : e ≤ l.length) (hi : i < e - s) :
    (l.slice s e).getD i 0#u8 = l.getD (s + i) 0#u8 := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_slice s e i l ⟨he, by omega⟩,
    List.getD_eq_getElem?_getD]

/-- `core.cmp.Ord.min` on `usize` returns the smaller operand. -/
theorem q6k_min_ok (x y : Usize) :
    ∃ z : Usize, core.cmp.Ord.min.trait_default core.cmp.OrdUsize x y = ok z ∧
      z.val = Nat.min x.val y.val := by
  unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body
  simp only [core.cmp.impls.PartialOrdUsize.lt, liftFun2, bind_tc_ok]
  by_cases h : y.val < x.val
  · refine ⟨y, by simp [h], ?_⟩
    exact (Nat.min_eq_right (Nat.le_of_lt h)).symm
  · refine ⟨x, by simp [h], ?_⟩
    exact (Nat.min_eq_left (Nat.le_of_not_lt h)).symm

/-- `f16LeHalf` of a 210-byte window at byte 208 is the raw half-word at `s + 208`. -/
lemma f16LeHalf_slice (raw : Slice U8) (s : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice s (s + 210)) (he : s + 210 ≤ raw.val.length) :
    f16LeHalf blk 208 = f16LeHalf raw (s + 208) := by
  unfold f16LeHalf
  congr 2
  rw [h, getD_slice raw.val s (s + 210) 208 he (by omega),
    getD_slice raw.val s (s + 210) 209 he (by omega)]

/-- The signed sub-block scale `k` of a 210-byte window is the raw-level `q6kScaleRef`. -/
lemma q6kScaleRef_slice (raw : Slice U8) (b : Nat) (blk : Slice U8) (k : Nat)
    (h : blk.val = raw.val.slice (210 * b) (210 * b + 210))
    (he : 210 * b + 210 ≤ raw.val.length) (hk : k < 16) :
    q6kScaleRef blk 0 k = q6kScaleRef raw b k := by
  rw [q6kScaleRef_eq_getD, h,
    getD_slice raw.val (210 * b) (210 * b + 210) (192 + k) he (by omega)]
  unfold q6kScaleRef
  simp only [Nat.add_assoc]

/-- The 210-byte window's bytes are `(drop (210·b)).take 210`, so `q6CodesRef` on the window is
the raw-level `q6kCodeRef`. -/
lemma q6kCodeRef_slice (raw : Slice U8) (b : Nat) (blk : Slice U8)
    (h : blk.val = raw.val.slice (210 * b) (210 * b + 210)) :
    ∀ i, q6CodesRef blk.val i = q6kCodeRef raw b i := by
  intro i
  have hsl : raw.val.slice (210 * b) (210 * b + 210) = (raw.val.drop (210 * b)).take 210 := by
    unfold List.slice
    congr 1
    omega
  unfold q6kCodeRef
  rw [h, hsl]

/-- The `j`-th entry of block `b`'s 256-value array is raw weight `256·b + j`. -/
lemma q6kWeight_block (raw : Slice U8) (b j : Nat) (hj : j < 256) :
    q6kWeight raw (256 * b + j) = q6kD raw b * q6kScaleRef raw b (j / 16) * q6kCodeRef raw b j := by
  unfold q6kWeight
  have h1 : (256 * b + j) / 256 = b := by omega
  have h2 : (256 * b + j) % 256 = j := by omega
  rw [h1, h2]

/-- Appending the next `take` raw weights advances the row prefix by `take`. -/
lemma q6kRowValues_append (raw : Slice U8) (m take : Nat) :
    q6kRowValues raw m ++ (List.range take).map (fun j => q6kWeight raw (m + j)) =
      q6kRowValues raw (m + take) := by
  unfold q6kRowValues
  rw [List.range_add, List.map_append, List.map_map]
  congr 1

/-- `block < (n + 255)/256` gives `256·block < n`: the block loop's divisibility bound. -/
lemma lt_mul_of_lt_div_ceil_256 {b n : Nat} (hb : b < (n + 255) / 256) : 256 * b < n := by
  by_contra h
  have hle : n ≤ 256 * b := not_lt.mp h
  have h1 : (n + 255) / 256 ≤ (256 * b + 255) / 256 := Nat.div_le_div_right (by omega)
  have h2 : (256 * b + 255) / 256 = b := by omega
  omega

/-- `256·b + min 256 (n - 256·b) = min (256·(b+1)) n` when `256·b ≤ n`. -/
lemma mul_block_add_min (b n : Nat) (h : 256 * b ≤ n) :
    256 * b + Nat.min 256 (n - 256 * b) = Nat.min (256 * (b + 1)) n := by
  change 256 * b + min 256 (n - 256 * b) = min (256 * (b + 1)) n
  by_cases h1 : n - 256 * b ≤ 256
  · rw [Nat.min_eq_right h1, Nat.min_eq_right (by omega)]
    omega
  · have h1' : 256 < n - 256 * b := not_le.mp h1
    rw [Nat.min_eq_left (le_of_lt h1'), Nat.min_eq_left (by omega)]
    omega

/-- `n ≤ 256 · ceil(n/256)`. -/
lemma le_mul_div_ceil_256 (n : Nat) : n ≤ 256 * ((n + 255) / 256) := by omega

/-- The extracted `div_ceil n QK_K` succeeds and its value is `(n + 255)/256`. -/
theorem dequant_div_ceil_256_ok (n : Usize) :
    ∃ blocks : Usize, core.num.Usize.div_ceil n quant.QK_K = ok blocks ∧
      blocks.val = (n.val + 255) / 256 := by
  obtain ⟨q, hq, hqv⟩ := UScalar.div_spec n (y := quant.QK_K) (by simp only [quant.QK_K]; decide)
  obtain ⟨r, hr, hrv⟩ := WP.spec_imp_exists
    (UScalar.rem_spec n (y := quant.QK_K) (by simp only [quant.QK_K]; decide))
  have hqv' : q.val = n.val / 256 := by rw [hqv]; simp [quant.QK_K]
  have hrv' : r.val = n.val % 256 := by rw [hrv]; simp [quant.QK_K]
  have hq' : UScalar.div n quant.QK_K = ok q := hq
  have hr' : UScalar.rem n quant.QK_K = ok r := hr
  unfold core.num.Usize.div_ceil
  rw [hq']
  simp only [bind_tc_ok]
  rw [hr']
  simp only [bind_tc_ok]
  by_cases hr0 : r.val = 0
  · refine ⟨q, ?_, ?_⟩
    · rw [if_pos hr0]
    · rw [hqv']
      have hr0' : n.val % 256 = 0 := by rw [← hrv']; exact hr0
      omega
  · obtain ⟨q1, hq1, hq1v⟩ := WP.spec_imp_exists
      (UScalar.add_spec (x := q) (y := (UScalar.one : Usize)) (by
        rw [hqv']; scalar_tac))
    refine ⟨q1, ?_, ?_⟩
    · rw [if_neg hr0]
      exact hq1
    · rw [hq1v, hqv']
      have hr0' : n.val % 256 ≠ 0 := by rw [← hrv']; exact hr0
      have hone : (UScalar.one : Usize).val = 1 := by simp [UScalar.one]
      rw [hone]
      omega

/-! ## The Q6_K row success domain and loop invariant -/

/-- The success domain of one 210-byte block `b`: the window is the raw slice `[210·b, 210·b+210)`,
`d` is finite, and every sub-block scale and weight product is inside `i64`. The pointwise
`d`/scale/code values are stated at the raw level (`q6kD`, `q6kScaleRef`, `q6kCodeRef`). -/
def q6kBlockDomain (raw : Slice U8) (b : Usize) : Prop :=
  ∃ blk : Slice U8,
    dequant.block_slice raw b 210#usize = ok (.Ok blk) ∧
    blk.val = raw.val.slice (210 * b.val) (210 * b.val + 210) ∧
    210 * b.val + 210 ≤ raw.val.length ∧
    210 ≤ blk.val.length ∧
    f16Exp (f16LeHalf blk 208) ≠ 31 ∧
    (∀ g, g < 16 → I64.min ≤ q6kD raw b.val * q6kScaleRef raw b.val g ∧
      q6kD raw b.val * q6kScaleRef raw b.val g ≤ I64.max) ∧
    (∀ g, g < 16 → ∀ l, l < 16 →
      I64.min ≤ q6kD raw b.val * q6kScaleRef raw b.val g * q6kCodeRef raw b.val (16 * g + l) ∧
      q6kD raw b.val * q6kScaleRef raw b.val g * q6kCodeRef raw b.val (16 * g + l) ≤ I64.max)

/-- Block-loop invariant: after `block` completed blocks the output is the initial output followed
by the first `min (256·block) n` raw weights, with the success flag set. -/
def q6kRowInv (raw : Slice U8) (n : Usize) (out0 out : alloc.vec.Vec I64)
    (blocks block : Usize) (ok1 : Bool) (err : Option error.EngineError) : Prop :=
  ok1 = true ∧ err = none ∧ block.val ≤ blocks.val ∧
  dequantVals out.val = dequantVals out0.val ++ q6kRowValues raw (Nat.min (256 * block.val) n.val)

/-- One block-loop body step on the success domain: decode block `block`, run the group loop,
append `take = min 256 (n - 256·block)` values and advance. -/
theorem q6kBlock_body_step (raw : Slice U8) (n blocks : Usize) (out0 out : alloc.vec.Vec I64)
    (block : Usize) (hblock : block.val < blocks.val)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : q6kBlockDomain raw block)
    (hcap : out.val.length + (n.val - 256 * block.val) ≤ Usize.max)
    (hinv : q6kRowInv raw n out0 out blocks block true none) :
    ∃ (out1 : alloc.vec.Vec I64) (block1 : Usize),
      dequant.dequant_q6_k_row_loop0.body raw n blocks out block true none =
        ok (ControlFlow.cont (out1, block1, true, none)) ∧
      q6kRowInv raw n out0 out1 blocks block1 true none ∧ block1.val = block.val + 1 := by
  obtain ⟨blk, hblk, hslice, hbnd, h210, hfin, hnol, hnol2⟩ := hdom
  obtain ⟨d, scales, codes, hq6, hdv, hscv, hcv⟩ := q6_k_block_ok raw block blk hblk h210 hfin
  have hdv' : d.val = q6kD raw block.val := by
    rw [hdv, f16LeHalf_slice raw (210 * block.val) blk hslice hbnd]; rfl
  have hscv' : ∀ k, k < 16 → (scales.val.getD k 0#i8).val = q6kScaleRef raw block.val k := by
    intro k hk; rw [hscv k hk]; exact q6kScaleRef_slice raw block.val blk k hslice hbnd hk
  have hcv' : ∀ i, i < 256 → (codes.val.getD i 0#i8).val = q6kCodeRef raw block.val i := by
    intro i hi; rw [hcv i hi]; exact q6kCodeRef_slice raw block.val blk hslice i
  have hmul : ∀ g, g < 16 → I64.min ≤ d.val * (scales.val.getD g 0#i8).val ∧
      d.val * (scales.val.getD g 0#i8).val ≤ I64.max := by
    intro g hg
    obtain ⟨h1, h2⟩ := hnol g hg
    rw [hdv', hscv' g hg]
    exact ⟨h1, h2⟩
  have hlane : ∀ g, g < 16 → ∀ l, l < 16 →
      I64.min ≤ d.val * (scales.val.getD g 0#i8).val * (codes.val.getD (16 * g + l) 0#i8).val ∧
      d.val * (scales.val.getD g 0#i8).val * (codes.val.getD (16 * g + l) 0#i8).val ≤ I64.max := by
    intro g hg l hl
    obtain ⟨h1, h2⟩ := hnol2 g hg l hl
    rw [hdv', hscv' g hg, hcv' (16 * g + l) (by omega)]
    exact ⟨h1, h2⟩
  have hltmul : 256 * block.val < n.val :=
    lt_mul_of_lt_div_ceil_256 (by rw [← hblocks]; exact hblock)
  have hinv0 : q6kGroupInv d scales codes (Array.repeat 256#usize 0#i64)
      (Array.repeat 256#usize 0#i64) 0 true none := by
    refine ⟨rfl, rfl, by omega, ?_⟩
    intro j hj
    rw [if_neg (by omega : ¬ j / 16 < 0)]
  obtain ⟨b1, hloop, vals1, hb1, hgrp⟩ := WP.spec_imp_exists
    (q6kGroup_loop_spec d scales codes (Array.repeat 256#usize 0#i64) hmul hlane hinv0)
  have hloop' : dequant.dequant_q6_k_row_loop0_loop0 true none d scales codes
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
      d.val * (scales.val.getD (j / 16) 0#i8).val * (codes.val.getD j 0#i8).val := by
    intro j hj
    have h := hgrp.2.2.2 j hj
    rwa [if_pos (by omega : j / 16 < 16)] at h
  have hmap : (List.range take.val).map (fun j => (vals1.val.getD j 0#i64).val) =
      (List.range take.val).map (fun j => q6kWeight raw (256 * block.val + j)) := by
    apply List.map_congr_left
    intro j hj
    have hj' : j < take.val := List.mem_range.mp hj
    have hj256 : j < 256 := by omega
    rw [hgrp_j j hj256, hdv', hscv' (j / 16) (by omega), hcv' j hj256,
      q6kWeight_block raw block.val j hj256]
  have hmin : Nat.min (256 * block.val) n.val = 256 * block.val :=
    Nat.min_eq_left (le_of_lt hltmul)
  have hsum : 256 * block.val + take.val = Nat.min (256 * (block.val + 1)) n.val := by
    rw [htakev']; exact mul_block_add_min block.val n.val (le_of_lt hltmul)
  have hinv_new : dequantVals out1.val = dequantVals out0.val ++
      q6kRowValues raw (Nat.min (256 * block1.val) n.val) := by
    rw [hblock1v']
    rw [hpushv, hmap, hinv.2.2.2, hmin, List.append_assoc, q6kRowValues_append, hsum]
  refine ⟨out1, block1, ?_, ⟨rfl, rfl, by omega, hinv_new⟩, hblock1v'⟩
  rw [dequant.dequant_q6_k_row_loop0.body]
  rw [if_pos (by scalar_tac : block < blocks)]
  simp only [if_true]
  rw [hq6]
  simp only [bind_tc_ok]
  rw [hiv]
  simp only [bind_tc_ok]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [htake]
  simp only [bind_tc_ok]
  change (do
    let (ok2, err1, vals1) ←
      dequant.dequant_q6_k_row_loop0_loop0 true none d scales codes
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

/-- The block loop: `loop.spec_decr_nat` with `q6kRowInv` and measure `blocks - block`, on the
success domain `q6kBlockDomain` for every block below `blocks`.

Complexity: `blocks` iterations, each a 210-byte read plus 16 groups of 16 lanes; the measure is
`blocks - block`. -/
theorem q6kRow_loop0_spec (raw : Slice U8) (n blocks : Usize) (out0 : alloc.vec.Vec I64)
    (hblocks : blocks.val = (n.val + 255) / 256)
    (hdom : ∀ b : Usize, b.val < blocks.val → q6kBlockDomain raw b)
    (hcap : out0.val.length + n.val ≤ Usize.max) :
    WP.spec (dequant.dequant_q6_k_row_loop0 raw n out0 blocks 0#usize true none)
      (fun p => p.2 = none ∧
        dequantVals p.1.val = dequantVals out0.val ++ q6kRowValues raw n.val) := by
  unfold dequant.dequant_q6_k_row_loop0
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
      blocks.val - s.2.1.val)
    (inv := fun s => q6kRowInv raw n out0 s.1 blocks s.2.1 s.2.2.1 s.2.2.2)
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
        simpa [dequantVals, q6kRowValues, List.length_append, List.length_map,
          List.length_range] using h
      have hcap' : out.val.length + (n.val - 256 * block.val) ≤ Usize.max := by
        rw [hlen_out]; omega
      obtain ⟨out1, block1, hbody, hinv1, hblock1⟩ :=
        q6kBlock_body_step raw n blocks out0 out block hlt hblocks (hdom block hlt) hcap'
          ⟨rfl, rfl, hle, hrec⟩
      rw [hbody]
      simp only [WP.spec_ok]
      refine ⟨hinv1, ?_⟩
      show blocks.val - block1.val < blocks.val - block.val
      omega
    · have hnot : ¬ (block < blocks) := fun hc => hlt ((UScalar.lt_equiv block blocks).1 hc)
      have hbody : dequant.dequant_q6_k_row_loop0.body raw n blocks out block true none
          = ok (ControlFlow.done (out, none)) := by
        rw [dequant.dequant_q6_k_row_loop0.body, if_neg hnot]
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
    simp [q6kRowValues]

/-- **`Q6_K` row law.** On the block success domain, the extracted `dequant_q6_k_row` succeeds
with `Ok ()` and appends exactly the first `n` raw weights `q6kRowValues raw n` to the output. -/
theorem dequant_q6_k_row_spec (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hdom : ∀ b : Usize, b.val < (n.val + 255) / 256 → q6kBlockDomain raw b)
    (hcap : out.val.length + n.val ≤ Usize.max) :
    ∃ out1 : alloc.vec.Vec I64,
      dequant.dequant_q6_k_row raw n out = ok (.Ok (), out1) ∧
      dequantVals out1.val = dequantVals out.val ++ q6kRowValues raw n.val := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_div_ceil_256_ok n
  obtain ⟨p, hloop, hp⟩ := WP.spec_imp_exists
    (q6kRow_loop0_spec raw n blocks out hblocks
      (fun b hb => hdom b (by rw [← hblocks]; exact hb)) hcap)
  obtain ⟨out1, err⟩ := p
  obtain ⟨herr, hvals⟩ := hp
  have herr' : err = none := herr
  subst err
  refine ⟨out1, ?_, hvals⟩
  unfold dequant.dequant_q6_k_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-! ## `dequant_row_fixed` at `GGML_TYPE_Q6_K` -/

/-- `dequant_row_fixed` at `ggml_type = 14` (`GGML_TYPE_Q6_K`) is `dequant_q6_k_row`. -/
theorem dequant_row_fixed_q6_k (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64) :
    dequant.dequant_row_fixed 14#u32 raw n out = dequant.dequant_q6_k_row raw n out := rfl

end apeiron_word
