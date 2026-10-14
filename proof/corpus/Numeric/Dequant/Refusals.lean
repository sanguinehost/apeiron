/-
Row-level refusals of the extracted `F16`, `Q4_K` and `Q6_K` dequantizers
.

Each theorem below is an equality stating the *first checked refusal* of the extracted row body in
program order, at the error variant the Rust source produces:

* short payload: a raw shorter than the first element/block needs refuses through the first length
  test (`dequant::read_u16` for `F16`, `dequant::block_slice` for `Q4_K`/`Q6_K`) with
  `QuantError::Truncated`;
* non-finite scale: a finite-length first block whose `f16` `d` (or, for `Q4_K`, whose `dmin` after
  a finite `d`) has an all-ones exponent refuses with `EngineError::NonFiniteScale bits`
  (`dequant::f16_to_fixed`, via the proved `f16_to_fixed_nonfinite`).

The output vector is unchanged in every refusal arm: the extracted loops carry an `ok` flag, stop on
the first refusal and return the accumulator untouched, so the row returns `Err` with the caller's
pre-state.

## Arm 3 (i64 weight-product overflow) — absent at block 0 lane 0

The Rust body has a checked multiply (`dequant::mul`, `i64::checked_mul`) with the distinct error
`EngineError::FixedOverflow`. It is not reachable at block 0 lane 0 of either `K` row: the proved
factor bounds leave the product inside `i64`. For `Q4_K`, `d` and `dmin` are finite `f16` magnitudes
with `|v| < 2^48` (`f16Value_abs_lt`, `Numeric.DequantQ4K`), a 6-bit scale/min code is `< 64`
(`q4kScaleMinRef_fst_lt`/`_snd_lt`) and a 4-bit code is `< 16` (`q4CodesRef_lt_16`), so at lane 0
`|slope| < 2^48·64 < 2^54` and `|slope·q| < 2^54·16 < 2^58 < 2^63`, with `|dmin·m| < 2^54` for the
checked subtract; every product and difference fits. The `Q6_K` lane product is `d·scales[0]·codes[0]`
with `|d| < 2^48` and a signed 8-bit scale in `[-128, 128]`; the codes are six-bit signed `q - 32`, so
`|d·sc·q| < 2^48·128·32 ≤ 2^60 < 2^63`. No first-block refusal can therefore be the overflow arm; it
does not exist and no equality is stated.

## Trust boundary

Kernel-checked against the Aeneas model of `usize::checked_mul`/`checked_add`, slice range `get`,
`Result::branch`/`FromResidual` and the extracted loop bodies (`target/aeneas-model/lean`);
rustc/LLVM codegen, the GGUF reader and the loader are outside this module. `FixedOverflow` is not
covered by an extracted equality here.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`/`axiom`/`opaque`. `decide` is not used
on extracted bodies.
-/
import Numeric.Dequant.Refinement
import Numeric.Dequant.K.Q4K
import Numeric.Dequant.K.Q6K

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Shared refusal helpers -/

/-- **`read_u16` short payload.** If the two-byte range starting at `i` leaves the payload
(`i + 2 = ok i1` and `raw.len < i1`), `read_u16` refuses with `Truncated i1 raw.len`. -/
theorem read_u16_refuse (raw : Slice U8) (i i1 : Usize)
    (hadd : i + 2#usize = ok i1) (h : raw.val.length < i1.val) :
    dequant.read_u16 raw i =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated i1 (Slice.len raw)))) := by
  rw [dequant.read_u16, hadd]
  simp only [bind_tc_ok, core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get]
  rw [if_neg (by intro hc; have hc2 : i1.val ≤ raw.val.length := hc.2; omega)]
  simp only [bind_tc_ok, core.option.Option.and_then, core.option.Option.ok_or_none,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`block_slice` at block `0` short payload.** A payload shorter than one `bb`-byte block
refuses through the range read with `Truncated bb raw.len`. -/
theorem block_slice_zero_refuse (raw : Slice U8) (bb : Usize) (h : raw.val.length < bb.val) :
    dequant.block_slice raw 0#usize bb =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated bb (Slice.len raw)))) := by
  obtain ⟨m, hm, hmv⟩ := dequant_usize_checked_mul_some 0#usize bb (by simp)
  obtain ⟨e, he, hev⟩ := dequant_usize_checked_add_some m bb (by rw [hmv]; simp; scalar_tac)
  have hebb : e = bb := UScalar.eq_of_val_eq (by rw [hev, hmv]; simp)
  have hcond : ¬ (m ≤ bb ∧ bb.val ≤ raw.val.length) := by
    intro hc; exact absurd hc.2 (not_le.mpr h)
  simp only [dequant.block_slice, hm, he, lift, bind_tc_ok,
    core.option.Option.ok_or_some, core.result.Result.Insts.CoreOpsTry.branch,
    core.slice.Slice.get, core.slice.index.SliceIndexRangeUsizeSlice.get,
    if_neg hcond, core.option.Option.ok_or_none, hebb,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- The extracted `div_ceil n QK_K` succeeds with value `(n + 255)/256` (local copy of the two
`Numeric.DequantQ{4,6}K` declarations, so this module does not depend on the duplicated name). -/
theorem dequant_refuse_div_ceil_256 (n : Usize) :
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

/-! ## `F16` refusals -/

/-- **`F16` row loop short payload.** At `idx = 0`, a payload shorter than two bytes refuses in
`read_u16` with `Truncated 2 raw.len`; the loop's `false` flag drains to `done` and returns the
accumulator. -/
theorem f16_row_loop_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 2) :
    dequant.dequant_f16_row_loop raw n out 0#usize true none =
      ok (out, some
        (error.EngineError.Quant (quant.QuantError.Truncated 2#usize (Slice.len raw)))) := by
  have hspec : dequant.dequant_f16_row_loop raw n out 0#usize true none
      ⦃ p => p = (out, some
        (error.EngineError.Quant (quant.QuantError.Truncated 2#usize (Slice.len raw)))) ⦄ := by
    unfold dequant.dequant_f16_row_loop
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
        n.val - s.2.1.val)
      (inv := fun s =>
        (s.2.2.1 = true ∧ s.2.2.2 = none ∧ s.2.1.val = 0 ∧ s.1 = out) ∨
        (s.2.2.1 = false ∧ s.2.2.2 = some
          (error.EngineError.Quant (quant.QuantError.Truncated 2#usize (Slice.len raw))) ∧
          s.1 = out))
    · intro s hs
      rcases s with ⟨out1, idx, ok1, err⟩
      dsimp only at hs ⊢
      rcases hs with ⟨hok, herr, hidx, hout⟩ | ⟨hok, herr, hout⟩
      · have hidxeq : idx = 0#usize := UScalar.eq_of_val_eq hidx
        obtain ⟨i, hmul, hiv⟩ := WP.spec_imp_exists
          (Usize.mul_spec (x := 2#usize) (y := idx) (by rw [hidx]; simp))
        have hiv' : i.val = 0 := by rw [hiv, hidxeq]; simp
        have hi2 : i.val + 2 ≤ Usize.max := by rw [hiv']; simp; scalar_tac
        obtain ⟨i1, hadd, hi1v⟩ := WP.spec_imp_exists (Usize.add_spec (x := i) (y := 2#usize) hi2)
        have hi1v' : i1.val = i.val + 2 := by rw [hi1v]; simp
        have hshort' : raw.val.length < i1.val := by rw [hi1v', hiv']; exact hshort
        have hread := read_u16_refuse raw i i1 hadd hshort'
        have hi1eq : i1 = 2#usize := UScalar.eq_of_val_eq (by rw [hi1v', hiv']; norm_num)
        rw [hi1eq] at hread
        obtain ⟨idx1, hidx1, hidx1v⟩ := WP.spec_imp_exists
          (Usize.add_spec (x := idx) (y := 1#usize) (by rw [hidx]; simp; scalar_tac))
        have hidx1v' : idx1.val = idx.val + 1 := by rw [hidx1v]; simp
        have hlt : idx < n := (UScalar.lt_equiv idx n).2 (by rw [hidx]; exact hn)
        have hbody : dequant.dequant_f16_row_loop.body raw n out1 idx ok1 err =
            ok (ControlFlow.cont (out1, idx1, false, some
              (error.EngineError.Quant (quant.QuantError.Truncated 2#usize (Slice.len raw))))) := by
          rw [dequant.dequant_f16_row_loop.body]
          simp only [if_pos hlt, if_pos hok, hmul, bind_tc_ok, hread, hidx1]
          rfl
        rw [hbody, Aeneas.Std.WP.spec_ok]
        dsimp only
        refine ⟨Or.inr ⟨rfl, rfl, hout⟩, ?_⟩
        show n.val - idx1.val < n.val - idx.val
        rw [hidx1v', hidx]
        omega
      · have hbody : dequant.dequant_f16_row_loop.body raw n out1 idx ok1 err =
            ok (ControlFlow.done (out1, err)) := by
          rw [dequant.dequant_f16_row_loop.body]
          by_cases hlt : idx < n
          · rw [if_pos hlt, if_neg (by rw [hok]; decide)]
          · rw [if_neg hlt]
        rw [hbody, Aeneas.Std.WP.spec_ok]
        dsimp only
        rw [hout, herr]
    · left; exact ⟨rfl, rfl, by simp, rfl⟩
  obtain ⟨b, hb, hpost⟩ := WP.spec_imp_exists hspec
  rw [hpost] at hb
  exact hb

/-- **`F16` row short payload.** A payload shorter than one `f16` (`raw.len < 2`) with at least one
element refuses with `Truncated 2 raw.len` and leaves `out` unchanged. -/
theorem dequant_f16_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 2) :
    dequant.dequant_f16_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 2#usize (Slice.len raw))), out) := by
  unfold dequant.dequant_f16_row
  rw [f16_row_loop_short raw n out hn hshort]
  simp only [bind_tc_ok]
  rfl

/-! ## `Q4_K` refusals -/

/-- **`Q4_K` block short payload.** A payload shorter than one 144-byte block refuses in
`block_slice` with `Truncated 144 raw.len`. -/
theorem q4_k_block_short (raw : Slice U8) (h : raw.val.length < 144) :
    dequant.q4_k_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 144#usize (Slice.len raw)))) := by
  simp only [dequant.q4_k_block, quant_Q4_K_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 144#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`Q4_K` block non-finite `d`.** On a whole first block whose `f16` `d` has an all-ones
exponent, `read_scale` refuses with `NonFiniteScale (f16LeHalf raw 0)` before any code is read. -/
theorem q4_k_block_nonfinite_d (raw : Slice U8) (hlen : 144 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.q4_k_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 144#usize (by simpa using hlen)
  have h144 : 144 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 0 = f16LeHalf raw 0 := by
    have h := f16LeHalf_block raw 0 0 blk hslice (by simpa using hlen) (by norm_num)
    simpa using h
  have hu := read_u16_ok blk 0#usize (by simp; omega)
  have hread : dequant.read_scale blk 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 0))) := by
    rw [dequant.read_scale, hu]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 0) (by rw [hbits]; exact hd)
  simp only [dequant.q4_k_block, quant_Q4_K_BLOCK_BYTES_eq, hblk, hread, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits]

/-- **`Q4_K` block non-finite `dmin`.** On a whole first block whose second `f16` scale (`dmin`,
byte 2) has an all-ones exponent and whose `d` (byte 0) is finite, the second `read_scale` refuses
with `NonFiniteScale (f16LeHalf raw 2)`. This is the second arm in program order after the `d`
read succeeds. -/
theorem q4_k_block_nonfinite_dmin (raw : Slice U8) (hlen : 144 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 0) ≠ 31) (hdmin : f16Exp (f16LeHalf raw 2) = 31) :
    dequant.q4_k_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 2))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 144#usize (by simpa using hlen)
  have h144 : 144 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits0 : f16LeHalf blk 0 = f16LeHalf raw 0 := by
    have h := f16LeHalf_block raw 0 0 blk hslice (by simpa using hlen) (by norm_num)
    simpa using h
  have hbits2 : f16LeHalf blk 2 = f16LeHalf raw 2 := by
    have h := f16LeHalf_block raw 0 2 blk hslice (by simpa using hlen) (by norm_num)
    simpa using h
  have hu0 := read_u16_ok blk 0#usize (by simp; omega)
  obtain ⟨d, hd0, -⟩ := f16_to_fixed_exact (f16LeHalf blk 0) (by rw [hbits0]; exact hd)
  have hd_eq : dequant.read_scale blk 0#usize = ok (.Ok d) := by
    rw [dequant.read_scale, hu0]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact hd0
  have hu2 := read_u16_ok blk 2#usize (by simp; omega)
  have hdmin_eq : dequant.read_scale blk 2#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 2))) := by
    rw [dequant.read_scale, hu2]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 2) (by rw [hbits2]; exact hdmin)
  simp only [dequant.q4_k_block, quant_Q4_K_BLOCK_BYTES_eq, hblk, hd_eq, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch, hdmin_eq,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits2]

/-- **`Q4_K` row block loop refusal.** If block 0 refuses with `e`, the block loop stops, leaves
`out` untouched and returns `some e`. -/
theorem q4kRow_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError)
    (hblocks : blocks.val = (n.val + 255) / 256) (hn : 0 < n.val)
    (hblock0 : dequant.q4_k_block raw 0#usize = ok (.Err e)) :
    dequant.dequant_q4_k_row_loop0 raw n out blocks 0#usize true none = ok (out, some e) := by
  have hbpos : 0 < blocks.val := by rw [hblocks]; omega
  have hspec : dequant.dequant_q4_k_row_loop0 raw n out blocks 0#usize true none
      ⦃ p => p = (out, some e) ⦄ := by
    unfold dequant.dequant_q4_k_row_loop0
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
        blocks.val - s.2.1.val)
      (inv := fun s =>
        (s.2.2.1 = true ∧ s.2.2.2 = none ∧ s.2.1.val = 0 ∧ s.1 = out) ∨
        (s.2.2.1 = false ∧ s.2.2.2 = some e ∧ s.1 = out))
    · intro s hs
      rcases s with ⟨out1, block, ok1, err⟩
      dsimp only at hs ⊢
      rcases hs with ⟨hok, herr, hblock, hout⟩ | ⟨hok, herr, hout⟩
      · have hbeq : block = 0#usize := UScalar.eq_of_val_eq hblock
        have hblock0' : dequant.q4_k_block raw block = ok (.Err e) := by rw [hbeq]; exact hblock0
        have hlt : block < blocks :=
          (UScalar.lt_equiv block blocks).2 (by rw [hblock]; exact hbpos)
        obtain ⟨block1, hb1, hb1v⟩ := WP.spec_imp_exists
          (Usize.add_spec (x := block) (y := 1#usize) (by rw [hblock]; simp; scalar_tac))
        have hb1v' : block1.val = block.val + 1 := by rw [hb1v]; simp
        have hbody : dequant.dequant_q4_k_row_loop0.body raw n blocks out1 block ok1 err =
            ok (ControlFlow.cont (out1, block1, false, some e)) := by
          rw [dequant.dequant_q4_k_row_loop0.body]
          simp only [if_pos hlt, if_pos hok, hblock0', bind_tc_ok, hb1]
        rw [hbody, Aeneas.Std.WP.spec_ok]
        dsimp only
        refine ⟨Or.inr ⟨rfl, rfl, hout⟩, ?_⟩
        show blocks.val - block1.val < blocks.val - block.val
        rw [hb1v', hblock]
        omega
      · have hbody : dequant.dequant_q4_k_row_loop0.body raw n blocks out1 block ok1 err =
            ok (ControlFlow.done (out1, err)) := by
          rw [dequant.dequant_q4_k_row_loop0.body]
          by_cases hlt : block < blocks
          · rw [if_pos hlt, if_neg (by rw [hok]; decide)]
          · rw [if_neg hlt]
        rw [hbody, Aeneas.Std.WP.spec_ok]
        dsimp only
        rw [hout, herr]
    · left; exact ⟨rfl, rfl, by simp, rfl⟩
  obtain ⟨b, hb, hpost⟩ := WP.spec_imp_exists hspec
  rw [hpost] at hb
  exact hb

/-- **`Q4_K` row refusal from a refusing first block.** With at least one element and block 0
refusing with `e`, the row returns `Err e` and the untouched `out`. -/
theorem dequant_q4_k_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.q4_k_block raw 0#usize = ok (.Err e)) :
    dequant.dequant_q4_k_row raw n out = ok (.Err e, out) := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_refuse_div_ceil_256 n
  have hloop := q4kRow_loop0_refuse raw n out blocks e hblocks hn hblock0
  unfold dequant.dequant_q4_k_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`Q4_K` row short payload.** A payload shorter than one 144-byte block with at least one
element refuses with `Truncated 144 raw.len`. -/
theorem dequant_q4_k_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 144) :
    dequant.dequant_q4_k_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 144#usize (Slice.len raw))), out) :=
  dequant_q4_k_row_refuse raw n out _ hn (q4_k_block_short raw hshort)

/-- **`Q4_K` row non-finite `d`.** On a whole first block (`raw.len ≥ 144`) whose `f16` `d` has an
all-ones exponent, the row refuses with `NonFiniteScale (f16LeHalf raw 0)`. -/
theorem dequant_q4_k_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 144 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.dequant_q4_k_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0)), out) :=
  dequant_q4_k_row_refuse raw n out _ hn (q4_k_block_nonfinite_d raw hlen hd)

/-- **`Q4_K` row non-finite `dmin`.** On a whole first block whose `d` (byte 0) is finite and whose
`dmin` (byte 2) has an all-ones exponent, the row refuses with `NonFiniteScale (f16LeHalf raw 2)`.
This is the second scale arm in program order after the `d` read succeeds. -/
theorem dequant_q4_k_row_nonfinite_dmin (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 144 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 0) ≠ 31) (hdmin : f16Exp (f16LeHalf raw 2) = 31) :
    dequant.dequant_q4_k_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 2)), out) :=
  dequant_q4_k_row_refuse raw n out _ hn (q4_k_block_nonfinite_dmin raw hlen hd hdmin)

/-! ## `Q6_K` refusals -/

/-- **`Q6_K` block short payload.** A payload shorter than one 210-byte block refuses in
`block_slice` with `Truncated 210 raw.len`. -/
theorem q6_k_block_short (raw : Slice U8) (h : raw.val.length < 210) :
    dequant.q6_k_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 210#usize (Slice.len raw)))) := by
  simp only [dequant.q6_k_block, quant_Q6_K_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 210#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`Q6_K` block non-finite `d`.** On a whole first block whose `f16` `d` (byte 208) has an
all-ones exponent, the block refuses with `NonFiniteScale (f16LeHalf raw 208)`. -/
theorem q6_k_block_nonfinite_d (raw : Slice U8) (hlen : 210 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 208) = 31) :
    dequant.q6_k_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 208))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 210#usize (by simpa using hlen)
  have h210 : 210 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 208 = f16LeHalf raw 208 := by
    have h := f16LeHalf_slice raw 0 blk hslice (by simpa using hlen)
    simpa using h
  have hdok := q6_k_d_ok blk h210
  have hread : dequant.f16_to_fixed (f16LeHalf raw 208) =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 208))) := by
    rw [← hbits]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 208) (by rw [hbits]; exact hd)
  simp only [dequant.q6_k_block, quant_Q6_K_BLOCK_BYTES_eq, hblk, hdok, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch, hbits, hread,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`Q6_K` row block loop refusal.** If block 0 refuses with `e`, the block loop stops, leaves
`out` untouched and returns `some e`. -/
theorem q6kRow_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError)
    (hblocks : blocks.val = (n.val + 255) / 256) (hn : 0 < n.val)
    (hblock0 : dequant.q6_k_block raw 0#usize = ok (.Err e)) :
    dequant.dequant_q6_k_row_loop0 raw n out blocks 0#usize true none = ok (out, some e) := by
  have hbpos : 0 < blocks.val := by rw [hblocks]; omega
  have hspec : dequant.dequant_q6_k_row_loop0 raw n out blocks 0#usize true none
      ⦃ p => p = (out, some e) ⦄ := by
    unfold dequant.dequant_q6_k_row_loop0
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
        blocks.val - s.2.1.val)
      (inv := fun s =>
        (s.2.2.1 = true ∧ s.2.2.2 = none ∧ s.2.1.val = 0 ∧ s.1 = out) ∨
        (s.2.2.1 = false ∧ s.2.2.2 = some e ∧ s.1 = out))
    · intro s hs
      rcases s with ⟨out1, block, ok1, err⟩
      dsimp only at hs ⊢
      rcases hs with ⟨hok, herr, hblock, hout⟩ | ⟨hok, herr, hout⟩
      · have hbeq : block = 0#usize := UScalar.eq_of_val_eq hblock
        have hblock0' : dequant.q6_k_block raw block = ok (.Err e) := by rw [hbeq]; exact hblock0
        have hlt : block < blocks :=
          (UScalar.lt_equiv block blocks).2 (by rw [hblock]; exact hbpos)
        obtain ⟨block1, hb1, hb1v⟩ := WP.spec_imp_exists
          (Usize.add_spec (x := block) (y := 1#usize) (by rw [hblock]; simp; scalar_tac))
        have hb1v' : block1.val = block.val + 1 := by rw [hb1v]; simp
        have hbody : dequant.dequant_q6_k_row_loop0.body raw n blocks out1 block ok1 err =
            ok (ControlFlow.cont (out1, block1, false, some e)) := by
          rw [dequant.dequant_q6_k_row_loop0.body]
          simp only [if_pos hlt, if_pos hok, hblock0', bind_tc_ok, hb1]
        rw [hbody, Aeneas.Std.WP.spec_ok]
        dsimp only
        refine ⟨Or.inr ⟨rfl, rfl, hout⟩, ?_⟩
        show blocks.val - block1.val < blocks.val - block.val
        rw [hb1v', hblock]
        omega
      · have hbody : dequant.dequant_q6_k_row_loop0.body raw n blocks out1 block ok1 err =
            ok (ControlFlow.done (out1, err)) := by
          rw [dequant.dequant_q6_k_row_loop0.body]
          by_cases hlt : block < blocks
          · rw [if_pos hlt, if_neg (by rw [hok]; decide)]
          · rw [if_neg hlt]
        rw [hbody, Aeneas.Std.WP.spec_ok]
        dsimp only
        rw [hout, herr]
    · left; exact ⟨rfl, rfl, by simp, rfl⟩
  obtain ⟨b, hb, hpost⟩ := WP.spec_imp_exists hspec
  rw [hpost] at hb
  exact hb

/-- **`Q6_K` row refusal from a refusing first block.** With at least one element and block 0
refusing with `e`, the row returns `Err e` and the untouched `out`. -/
theorem dequant_q6_k_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.q6_k_block raw 0#usize = ok (.Err e)) :
    dequant.dequant_q6_k_row raw n out = ok (.Err e, out) := by
  obtain ⟨blocks, hdiv, hblocks⟩ := dequant_refuse_div_ceil_256 n
  have hloop := q6kRow_loop0_refuse raw n out blocks e hblocks hn hblock0
  unfold dequant.dequant_q6_k_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`Q6_K` row short payload.** A payload shorter than one 210-byte block with at least one
element refuses with `Truncated 210 raw.len`. -/
theorem dequant_q6_k_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 210) :
    dequant.dequant_q6_k_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 210#usize (Slice.len raw))), out) :=
  dequant_q6_k_row_refuse raw n out _ hn (q6_k_block_short raw hshort)

/-- **`Q6_K` row non-finite `d`.** On a whole first block (`raw.len ≥ 210`) whose `f16` `d` at byte
208 has an all-ones exponent, the row refuses with `NonFiniteScale (f16LeHalf raw 208)`. -/
theorem dequant_q6_k_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 210 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 208) = 31) :
    dequant.dequant_q6_k_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 208)), out) :=
  dequant_q6_k_row_refuse raw n out _ hn (q6_k_block_nonfinite_d raw hlen hd)

end

end apeiron_word
