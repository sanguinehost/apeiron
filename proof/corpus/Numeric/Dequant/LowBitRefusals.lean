/-
Row-level refusals of the six extracted low-bit dequantizers
(`APEIRON-TERNARY-W3-T7`).

Each headline below is an equality stating the *first checked refusal* of the extracted row body in
program order, at the error variant the Rust source produces:

* short payload: a raw shorter than one block refuses through the first length test
  (`dequant::block_slice`, called at the head of each block reader) with `QuantError::Truncated` and
  the block byte count as the required length;
* non-finite scale: a whole first block whose `f16` `d` has an all-ones exponent refuses with
  `EngineError::NonFiniteScale bits` (`dequant::f16_to_fixed`, via the proved `f16_to_fixed_nonfinite`
  at the scale offsets `0` (`Q1_0`/`Q2_0`/`PQ2_0`), `64` (`TQ2_0`), `52` (`TQ1_0`) and `26`
  (`PTQ1_0`)).

The output vector is unchanged in every refusal arm: each extracted block loop carries an `ok` flag,
stops on the first refusal of block 0 and returns the accumulator untouched, so the row returns
`Err` with the caller's pre-state. All six block loops share the state type
`Vec I64 × Usize × Bool × Option EngineError`, so one generic drain lemma
(`lowbit_row_loop0_refuse`) covers them.

## Overflow arm — absent on the success path

`dequant::mul` is a checked `i64` multiply with the distinct error `EngineError::FixedOverflow`. It
is not reachable on any success path of these six faces. Every weight is `(c − 1)·d` with a four-level
code `c < 4` (or `±d` for `Q1_0`, `(trit − 1)·d` with `trit < 3` for `TQ1_0`/`PTQ1_0`), and a finite
`f16` magnitude `|d| < 2^48` (`f16Value_abs_lt`); hence `|value| ≤ 2·|d| < 2^49` (the spec bound in
`docs/epics/ternary-hadamard.md`), and `|value| < 2^49 ≪ 2^63`. No fixed-overflow refusal equality is
stated for these formats.

## Trust boundary

Kernel-checked against the Aeneas model of `usize` checked arithmetic, slice range `get`,
`Result::branch`/`FromResidual` and the extracted block readers and row loops
(`target/aeneas-model/lean`); rustc/LLVM codegen, the GGUF reader and the loader are outside this
module. `FixedOverflow` is not covered by an extracted equality here.

No `sorry`/`admit`/`set_option`/`native_decide`/`bv_decide`/`axiom`/`opaque`. `decide` is not used on
extracted bodies.
-/
import Numeric.Dequant.Refusals
import Numeric.Dequant.PTQ1_0
import Numeric.Dequant.TQ2_0
import Numeric.Dequant.Q1Q2

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## Shared refusal lemmas -/

/-- **Generic block-loop refusal.** If the first block state steps to a draining state carrying `e`,
and every draining state returns, the loop returns `some e` with the accumulator untouched. All six
low-bit row block loops have this body shape. -/
theorem lowbit_row_loop0_refuse
    (body : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError →
      Result (ControlFlow (alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError)
                          (alloc.vec.Vec I64 × Option error.EngineError)))
    (out : alloc.vec.Vec I64) (e : error.EngineError)
    (h0 : body (out, 0#usize, true, none) =
      ok (ControlFlow.cont (out, 1#usize, false, some e)))
    (hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      body (out1, block, false, err) = ok (ControlFlow.done (out1, err))) :
    loop body (out, 0#usize, true, none) = ok (out, some e) := by
  have hspec : loop body (out, 0#usize, true, none) ⦃ p => p = (out, some e) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option error.EngineError =>
        if s.2.1.val = 0 then 1 else 0)
      (inv := fun s =>
        (s.2.2.1 = true ∧ s.2.2.2 = none ∧ s.2.1 = 0#usize ∧ s.1 = out) ∨
        (s.2.2.1 = false ∧ s.2.2.2 = some e ∧ s.1 = out))
    · intro s hs
      rcases s with ⟨out1, block, ok1, err⟩
      dsimp only at hs ⊢
      rcases hs with ⟨hok, herr, hblock, hout⟩ | ⟨hok, herr, hout⟩
      · rw [hok, herr, hblock, hout, h0, Aeneas.Std.WP.spec_ok]
        refine ⟨Or.inr ⟨rfl, rfl, rfl⟩, ?_⟩
        simp
      · rw [hok, herr, hout, hdone out block (some e), Aeneas.Std.WP.spec_ok]
    · exact Or.inl ⟨rfl, rfl, rfl, rfl⟩
  obtain ⟨b, hb, hpost⟩ := WP.spec_imp_exists hspec
  rw [hpost] at hb
  exact hb

/-- `f16LeHalf` of a window at byte `j` is the raw half-word at `s + j`. -/
lemma low_f16LeHalf_slice_off (raw blk : Slice U8) (s e j : Nat)
    (h : blk.val = raw.val.slice s e) (he : e ≤ raw.val.length) (hj : s + j + 2 ≤ e) :
    f16LeHalf blk j = f16LeHalf raw (s + j) := by
  unfold f16LeHalf
  have h0 : blk.val.getD j 0#u8 = raw.val.getD (s + j) 0#u8 := by
    rw [h, getD_slice raw.val s e j he (by omega)]
  have h1 : blk.val.getD (j + 1) 0#u8 = raw.val.getD (s + j + 1) 0#u8 := by
    rw [h, getD_slice raw.val s e (j + 1) he (by omega)]
    congr 1
  rw [h0, h1]

/-- Every finite `f16` scale magnitude is below `2^49` (the spec refusal bound). -/
lemma lowbit_f16Value_abs_lt_49 (h : U16) (he : f16Exp h ≠ 31) : |f16Value h| < 2 ^ 49 :=
  lt_trans (f16Value_abs_lt h he) (by norm_num)

/-! ## `Q1_0` / `Q2_0` refusals (shared 18-byte `q1_q2_block`) -/

/-- **`Q1_0`/`Q2_0` block short payload.** A payload shorter than one 18-byte block refuses through
the range read with `Truncated 18 raw.len`. -/
theorem q1_q2_block_short (raw : Slice U8) (h : raw.val.length < 18) :
    dequant.lowbit.q1_q2_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 18#usize (Slice.len raw)))) := by
  simp only [dequant.lowbit.q1_q2_block, quant_Q1_0_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 18#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`Q1_0`/`Q2_0` block non-finite `d`.** On a whole first block (`raw.len ≥ 18`) whose `f16` `d`
(byte 0) has an all-ones exponent, `read_scale` refuses with `NonFiniteScale (f16LeHalf raw 0)`
before any code byte is read. -/
theorem q1_q2_block_nonfinite_d (raw : Slice U8) (hlen : 18 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.lowbit.q1_q2_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 18#usize (by simpa using hlen)
  have hblen : 18 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 0 = f16LeHalf raw 0 :=
    low_f16LeHalf_slice raw blk 0 18 hslice (by simpa using hlen) (by norm_num)
  have hu := read_u16_ok blk 0#usize (by simp; omega)
  have hread : dequant.read_scale blk 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 0))) := by
    rw [dequant.read_scale, hu]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 0) (by rw [hbits]; exact hd)
  simp only [dequant.lowbit.q1_q2_block, quant_Q1_0_BLOCK_BYTES_eq, hblk, hread, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits]

/-- **`Q1_0` row block-loop refusal.** If block 0 refuses with `e`, the block loop stops, leaves
`out` untouched and returns `some e`. -/
theorem q1_0_row_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError) (hblocks : 0 < blocks.val)
    (hblock0 : dequant.lowbit.q1_q2_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_q1_0_row_loop0 raw n out blocks 0#usize true none =
      ok (out, some e) := by
  obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have hb1eq : b1 = 1#usize := UScalar.eq_of_val_eq (by rw [hb1v]; simp)
  have h0 : dequant.lowbit.dequant_q1_0_row_loop0.body raw n blocks out 0#usize true none =
      ok (ControlFlow.cont (out, 1#usize, false, some e)) := by
    rw [dequant.lowbit.dequant_q1_0_row_loop0.body]
    simp only [hblock0, bind_tc_ok, hb1, hb1eq]
    split_ifs <;> simp_all
  have hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      dequant.lowbit.dequant_q1_0_row_loop0.body raw n blocks out1 block false err =
        ok (ControlFlow.done (out1, err)) := by
    intro out1 block err
    rw [dequant.lowbit.dequant_q1_0_row_loop0.body]
    by_cases hlt : block < blocks <;> simp [hlt]
  exact lowbit_row_loop0_refuse
    (body := fun (out1, block1, ok2, err1) =>
      dequant.lowbit.dequant_q1_0_row_loop0.body raw n blocks out1 block1 ok2 err1)
    out e h0 hdone

/-- **`Q1_0` row refusal from a refusing first block.** -/
theorem dequant_q1_0_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.lowbit.q1_q2_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_q1_0_row raw n out = ok (.Err e, out) := by
  obtain ⟨blocks, hdiv, hblocksv⟩ := dequant_div_ceil_128_ok n
  have hbpos : 0 < blocks.val := by rw [hblocksv]; omega
  have hloop := q1_0_row_loop0_refuse raw n out blocks e hbpos hblock0
  unfold dequant.lowbit.dequant_q1_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`Q1_0` row short payload.** -/
theorem dequant_q1_0_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 18) :
    dequant.lowbit.dequant_q1_0_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 18#usize (Slice.len raw))),
        out) :=
  dequant_q1_0_row_refuse raw n out _ hn (q1_q2_block_short raw hshort)

/-- **`Q1_0` row non-finite `d`.** -/
theorem dequant_q1_0_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 18 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.lowbit.dequant_q1_0_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0)), out) :=
  dequant_q1_0_row_refuse raw n out _ hn (q1_q2_block_nonfinite_d raw hlen hd)

/-- **`Q2_0` row block-loop refusal.** -/
theorem q2_0_row_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError) (hblocks : 0 < blocks.val)
    (hblock0 : dequant.lowbit.q1_q2_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_q2_0_row_loop0 raw n out blocks 0#usize true none =
      ok (out, some e) := by
  obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have hb1eq : b1 = 1#usize := UScalar.eq_of_val_eq (by rw [hb1v]; simp)
  have h0 : dequant.lowbit.dequant_q2_0_row_loop0.body raw n blocks out 0#usize true none =
      ok (ControlFlow.cont (out, 1#usize, false, some e)) := by
    rw [dequant.lowbit.dequant_q2_0_row_loop0.body]
    simp only [hblock0, bind_tc_ok, hb1, hb1eq]
    split_ifs <;> simp_all
  have hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      dequant.lowbit.dequant_q2_0_row_loop0.body raw n blocks out1 block false err =
        ok (ControlFlow.done (out1, err)) := by
    intro out1 block err
    rw [dequant.lowbit.dequant_q2_0_row_loop0.body]
    by_cases hlt : block < blocks <;> simp [hlt]
  exact lowbit_row_loop0_refuse
    (body := fun (out1, block1, ok2, err1) =>
      dequant.lowbit.dequant_q2_0_row_loop0.body raw n blocks out1 block1 ok2 err1)
    out e h0 hdone

/-- **`Q2_0` row refusal from a refusing first block.** -/
theorem dequant_q2_0_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.lowbit.q1_q2_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_q2_0_row raw n out = ok (.Err e, out) := by
  obtain ⟨blocks, hdiv, hblocksv⟩ := dequant_div_ceil_64_ok n
  have hbpos : 0 < blocks.val := by rw [hblocksv]; omega
  have hloop := q2_0_row_loop0_refuse raw n out blocks e hbpos hblock0
  unfold dequant.lowbit.dequant_q2_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`Q2_0` row short payload.** -/
theorem dequant_q2_0_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 18) :
    dequant.lowbit.dequant_q2_0_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 18#usize (Slice.len raw))),
        out) :=
  dequant_q2_0_row_refuse raw n out _ hn (q1_q2_block_short raw hshort)

/-- **`Q2_0` row non-finite `d`.** -/
theorem dequant_q2_0_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 18 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.lowbit.dequant_q2_0_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0)), out) :=
  dequant_q2_0_row_refuse raw n out _ hn (q1_q2_block_nonfinite_d raw hlen hd)

/-! ## `PQ2_0` refusals (34-byte `pq2_0_block`, `d`@0) -/

/-- **`PQ2_0` block short payload.** -/
theorem pq2_0_block_short (raw : Slice U8) (h : raw.val.length < 34) :
    dequant.lowbit.pq2_0_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 34#usize (Slice.len raw)))) := by
  simp only [dequant.lowbit.pq2_0_block, quant_PQ2_0_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 34#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`PQ2_0` block non-finite `d`.** On a whole first block (`raw.len ≥ 34`) whose `f16` `d` (byte 0)
has an all-ones exponent, the block refuses with `NonFiniteScale (f16LeHalf raw 0)`. -/
theorem pq2_0_block_nonfinite_d (raw : Slice U8) (hlen : 34 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.lowbit.pq2_0_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 34#usize (by simpa using hlen)
  have hblen : 34 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 0 = f16LeHalf raw 0 :=
    low_f16LeHalf_slice raw blk 0 34 hslice (by simpa using hlen) (by norm_num)
  have hu := read_u16_ok blk 0#usize (by simp; omega)
  have hread : dequant.read_scale blk 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 0))) := by
    rw [dequant.read_scale, hu]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 0) (by rw [hbits]; exact hd)
  simp only [dequant.lowbit.pq2_0_block, quant_PQ2_0_BLOCK_BYTES_eq, hblk, hread, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits]

/-- **`PQ2_0` row block-loop refusal.** -/
theorem pq2_0_row_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError) (hblocks : 0 < blocks.val)
    (hblock0 : dequant.lowbit.pq2_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_pq2_0_row_loop0 raw n out blocks 0#usize true none =
      ok (out, some e) := by
  obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have hb1eq : b1 = 1#usize := UScalar.eq_of_val_eq (by rw [hb1v]; simp)
  have h0 : dequant.lowbit.dequant_pq2_0_row_loop0.body raw n blocks out 0#usize true none =
      ok (ControlFlow.cont (out, 1#usize, false, some e)) := by
    rw [dequant.lowbit.dequant_pq2_0_row_loop0.body]
    simp only [hblock0, bind_tc_ok, hb1, hb1eq]
    split_ifs <;> simp_all
  have hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      dequant.lowbit.dequant_pq2_0_row_loop0.body raw n blocks out1 block false err =
        ok (ControlFlow.done (out1, err)) := by
    intro out1 block err
    rw [dequant.lowbit.dequant_pq2_0_row_loop0.body]
    by_cases hlt : block < blocks <;> simp [hlt]
  exact lowbit_row_loop0_refuse
    (body := fun (out1, block1, ok2, err1) =>
      dequant.lowbit.dequant_pq2_0_row_loop0.body raw n blocks out1 block1 ok2 err1)
    out e h0 hdone

/-- **`PQ2_0` row refusal from a refusing first block.** -/
theorem dequant_pq2_0_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.lowbit.pq2_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_pq2_0_row raw n out = ok (.Err e, out) := by
  obtain ⟨blocks, hdiv, hblocksv⟩ := dequant_div_ceil_128pq_ok n
  have hbpos : 0 < blocks.val := by rw [hblocksv]; omega
  have hloop := pq2_0_row_loop0_refuse raw n out blocks e hbpos hblock0
  unfold dequant.lowbit.dequant_pq2_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`PQ2_0` row short payload.** -/
theorem dequant_pq2_0_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 34) :
    dequant.lowbit.dequant_pq2_0_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 34#usize (Slice.len raw))),
        out) :=
  dequant_pq2_0_row_refuse raw n out _ hn (pq2_0_block_short raw hshort)

/-- **`PQ2_0` row non-finite `d`.** -/
theorem dequant_pq2_0_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 34 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 0) = 31) :
    dequant.lowbit.dequant_pq2_0_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 0)), out) :=
  dequant_pq2_0_row_refuse raw n out _ hn (pq2_0_block_nonfinite_d raw hlen hd)

/-! ## `TQ2_0` refusals (66-byte `tq2_0_block`, `d`@64) -/

/-- **`TQ2_0` block short payload.** -/
theorem tq2_0_block_short (raw : Slice U8) (h : raw.val.length < 66) :
    dequant.lowbit.tq2_0_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 66#usize (Slice.len raw)))) := by
  simp only [dequant.lowbit.tq2_0_block, quant_TQ2_0_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 66#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`TQ2_0` block non-finite `d`.** On a whole first block (`raw.len ≥ 66`) whose `f16` `d` (byte 64)
has an all-ones exponent, the block refuses with `NonFiniteScale (f16LeHalf raw 64)`. -/
theorem tq2_0_block_nonfinite_d (raw : Slice U8) (hlen : 66 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 64) = 31) :
    dequant.lowbit.tq2_0_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 64))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 66#usize (by simpa using hlen)
  have hblen : 66 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 64 = f16LeHalf raw 64 :=
    low_f16LeHalf_slice_off raw blk 0 66 64 hslice (by simpa using hlen) (by norm_num)
  have hu := read_u16_ok blk 64#usize (by simp; omega)
  have hread : dequant.read_scale blk 64#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 64))) := by
    rw [dequant.read_scale, hu]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 64) (by rw [hbits]; exact hd)
  simp only [dequant.lowbit.tq2_0_block, quant_TQ2_0_BLOCK_BYTES_eq, hblk, hread, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits]

/-- **`TQ2_0` row block-loop refusal.** -/
theorem tq2_0_row_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError) (hblocks : 0 < blocks.val)
    (hblock0 : dequant.lowbit.tq2_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_tq2_0_row_loop0 raw n out blocks 0#usize true none =
      ok (out, some e) := by
  obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have hb1eq : b1 = 1#usize := UScalar.eq_of_val_eq (by rw [hb1v]; simp)
  have h0 : dequant.lowbit.dequant_tq2_0_row_loop0.body raw n blocks out 0#usize true none =
      ok (ControlFlow.cont (out, 1#usize, false, some e)) := by
    rw [dequant.lowbit.dequant_tq2_0_row_loop0.body]
    simp only [hblock0, bind_tc_ok, hb1, hb1eq]
    split_ifs <;> simp_all
  have hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      dequant.lowbit.dequant_tq2_0_row_loop0.body raw n blocks out1 block false err =
        ok (ControlFlow.done (out1, err)) := by
    intro out1 block err
    rw [dequant.lowbit.dequant_tq2_0_row_loop0.body]
    by_cases hlt : block < blocks <;> simp [hlt]
  exact lowbit_row_loop0_refuse
    (body := fun (out1, block1, ok2, err1) =>
      dequant.lowbit.dequant_tq2_0_row_loop0.body raw n blocks out1 block1 ok2 err1)
    out e h0 hdone

/-- **`TQ2_0` row refusal from a refusing first block.** -/
theorem dequant_tq2_0_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.lowbit.tq2_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_tq2_0_row raw n out = ok (.Err e, out) := by
  obtain ⟨blocks, hdiv, hblocksv⟩ := tq2_0_div_ceil_ok n
  have hbpos : 0 < blocks.val := by rw [hblocksv]; omega
  have hloop := tq2_0_row_loop0_refuse raw n out blocks e hbpos hblock0
  unfold dequant.lowbit.dequant_tq2_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`TQ2_0` row short payload.** -/
theorem dequant_tq2_0_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 66) :
    dequant.lowbit.dequant_tq2_0_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 66#usize (Slice.len raw))),
        out) :=
  dequant_tq2_0_row_refuse raw n out _ hn (tq2_0_block_short raw hshort)

/-- **`TQ2_0` row non-finite `d`.** -/
theorem dequant_tq2_0_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 66 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 64) = 31) :
    dequant.lowbit.dequant_tq2_0_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 64)), out) :=
  dequant_tq2_0_row_refuse raw n out _ hn (tq2_0_block_nonfinite_d raw hlen hd)

/-! ## `TQ1_0` refusals (54-byte `tq1_0_block`, `d`@52) -/

/-- **`TQ1_0` block short payload.** -/
theorem tq1_0_block_short (raw : Slice U8) (h : raw.val.length < 54) :
    dequant.lowbit.tq1_0_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 54#usize (Slice.len raw)))) := by
  simp only [dequant.lowbit.tq1_0_block, quant_TQ1_0_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 54#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`TQ1_0` block non-finite `d`.** On a whole first block (`raw.len ≥ 54`) whose `f16` `d` (byte 52)
has an all-ones exponent, the block refuses with `NonFiniteScale (f16LeHalf raw 52)`. -/
theorem tq1_0_block_nonfinite_d (raw : Slice U8) (hlen : 54 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 52) = 31) :
    dequant.lowbit.tq1_0_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 52))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 54#usize (by simpa using hlen)
  have hblen : 54 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 52 = f16LeHalf raw 52 :=
    f16LeHalf_block54 raw 0 52 blk hslice (by simpa using hlen) (by norm_num)
  have hu := read_u16_ok blk 52#usize (by simp; omega)
  have hread : dequant.read_scale blk 52#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 52))) := by
    rw [dequant.read_scale, hu]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 52) (by rw [hbits]; exact hd)
  simp only [dequant.lowbit.tq1_0_block, quant_TQ1_0_BLOCK_BYTES_eq, hblk, hread, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits]

/-- **`TQ1_0` row block-loop refusal.** -/
theorem tq1_0_row_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError) (hblocks : 0 < blocks.val)
    (hblock0 : dequant.lowbit.tq1_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_tq1_0_row_loop0 raw n out blocks 0#usize true none =
      ok (out, some e) := by
  obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have hb1eq : b1 = 1#usize := UScalar.eq_of_val_eq (by rw [hb1v]; simp)
  have h0 : dequant.lowbit.dequant_tq1_0_row_loop0.body raw n blocks out 0#usize true none =
      ok (ControlFlow.cont (out, 1#usize, false, some e)) := by
    rw [dequant.lowbit.dequant_tq1_0_row_loop0.body]
    simp only [hblock0, bind_tc_ok, hb1, hb1eq]
    split_ifs <;> simp_all
  have hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      dequant.lowbit.dequant_tq1_0_row_loop0.body raw n blocks out1 block false err =
        ok (ControlFlow.done (out1, err)) := by
    intro out1 block err
    rw [dequant.lowbit.dequant_tq1_0_row_loop0.body]
    by_cases hlt : block < blocks <;> simp [hlt]
  exact lowbit_row_loop0_refuse
    (body := fun (out1, block1, ok2, err1) =>
      dequant.lowbit.dequant_tq1_0_row_loop0.body raw n blocks out1 block1 ok2 err1)
    out e h0 hdone

/-- **`TQ1_0` row refusal from a refusing first block.** -/
theorem dequant_tq1_0_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.lowbit.tq1_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_tq1_0_row raw n out = ok (.Err e, out) := by
  have hQK : quant.TQ1_0_QK = quant.QK_K := by with_unfolding_all rfl
  obtain ⟨blocks, hdiv0, hblocksv⟩ := dequant_refuse_div_ceil_256 n
  have hdiv : core.num.Usize.div_ceil n quant.TQ1_0_QK = ok blocks := by rw [hQK]; exact hdiv0
  have hbpos : 0 < blocks.val := by rw [hblocksv]; omega
  have hloop := tq1_0_row_loop0_refuse raw n out blocks e hbpos hblock0
  unfold dequant.lowbit.dequant_tq1_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`TQ1_0` row short payload.** -/
theorem dequant_tq1_0_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 54) :
    dequant.lowbit.dequant_tq1_0_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 54#usize (Slice.len raw))),
        out) :=
  dequant_tq1_0_row_refuse raw n out _ hn (tq1_0_block_short raw hshort)

/-- **`TQ1_0` row non-finite `d`.** -/
theorem dequant_tq1_0_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 54 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 52) = 31) :
    dequant.lowbit.dequant_tq1_0_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 52)), out) :=
  dequant_tq1_0_row_refuse raw n out _ hn (tq1_0_block_nonfinite_d raw hlen hd)

/-! ## `PTQ1_0` refusals (28-byte `ptq1_0_block`, `d`@26) -/

/-- **`PTQ1_0` block short payload.** -/
theorem ptq1_0_block_short (raw : Slice U8) (h : raw.val.length < 28) :
    dequant.lowbit.ptq1_0_block raw 0#usize =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 28#usize (Slice.len raw)))) := by
  simp only [dequant.lowbit.ptq1_0_block, quant_PTQ1_0_BLOCK_BYTES_eq,
    block_slice_zero_refuse raw 28#usize h, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from]

/-- **`PTQ1_0` block non-finite `d`.** On a whole first block (`raw.len ≥ 28`) whose `f16` `d` (byte 26)
has an all-ones exponent, the block refuses with `NonFiniteScale (f16LeHalf raw 26)`. -/
theorem ptq1_0_block_nonfinite_d (raw : Slice U8) (hlen : 28 ≤ raw.val.length)
    (hd : f16Exp (f16LeHalf raw 26) = 31) :
    dequant.lowbit.ptq1_0_block raw 0#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 26))) := by
  obtain ⟨blk, hblk, hslice⟩ :=
    block_slice_ok raw 0#usize 28#usize (by simpa using hlen)
  have hblen : 28 ≤ blk.val.length := by rw [hslice]; simp; omega
  have hbits : f16LeHalf blk 26 = f16LeHalf raw 26 :=
    f16LeHalf_block28 raw 0 26 blk hslice (by simpa using hlen) (by norm_num)
  have hu := read_u16_ok blk 26#usize (by simp; omega)
  have hread : dequant.read_scale blk 26#usize =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf blk 26))) := by
    rw [dequant.read_scale, hu]
    simp only [bind_tc_ok, core.result.Result.Insts.CoreOpsTry.branch]
    exact f16_to_fixed_nonfinite (f16LeHalf blk 26) (by rw [hbits]; exact hd)
  simp only [dequant.lowbit.ptq1_0_block, quant_PTQ1_0_BLOCK_BYTES_eq, hblk, hread, bind_tc_ok,
    core.result.Result.Insts.CoreOpsTry.branch,
    core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual,
    core.convert.FromSame.from, hbits]

/-- **`PTQ1_0` row block-loop refusal.** -/
theorem ptq1_0_row_loop0_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (blocks : Usize) (e : error.EngineError) (hblocks : 0 < blocks.val)
    (hblock0 : dequant.lowbit.ptq1_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_ptq1_0_row_loop0 raw n out blocks 0#usize true none =
      ok (out, some e) := by
  obtain ⟨b1, hb1, hb1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  have hb1eq : b1 = 1#usize := UScalar.eq_of_val_eq (by rw [hb1v]; simp)
  have h0 : dequant.lowbit.dequant_ptq1_0_row_loop0.body raw n blocks out 0#usize true none =
      ok (ControlFlow.cont (out, 1#usize, false, some e)) := by
    rw [dequant.lowbit.dequant_ptq1_0_row_loop0.body]
    simp only [hblock0, bind_tc_ok, hb1, hb1eq]
    split_ifs <;> simp_all
  have hdone : ∀ (out1 : alloc.vec.Vec I64) (block : Usize) (err : Option error.EngineError),
      dequant.lowbit.dequant_ptq1_0_row_loop0.body raw n blocks out1 block false err =
        ok (ControlFlow.done (out1, err)) := by
    intro out1 block err
    rw [dequant.lowbit.dequant_ptq1_0_row_loop0.body]
    by_cases hlt : block < blocks <;> simp [hlt]
  exact lowbit_row_loop0_refuse
    (body := fun (out1, block1, ok2, err1) =>
      dequant.lowbit.dequant_ptq1_0_row_loop0.body raw n blocks out1 block1 ok2 err1)
    out e h0 hdone

/-- **`PTQ1_0` row refusal from a refusing first block.** -/
theorem dequant_ptq1_0_row_refuse (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (e : error.EngineError) (hn : 0 < n.val)
    (hblock0 : dequant.lowbit.ptq1_0_block raw 0#usize = ok (.Err e)) :
    dequant.lowbit.dequant_ptq1_0_row raw n out = ok (.Err e, out) := by
  have hQK : quant.PTQ1_0_QK = quant.Q1_0_QK := by with_unfolding_all rfl
  obtain ⟨blocks, hdiv0, hblocksv⟩ := dequant_div_ceil_128_ok n
  have hdiv : core.num.Usize.div_ceil n quant.PTQ1_0_QK = ok blocks := by rw [hQK]; exact hdiv0
  have hbpos : 0 < blocks.val := by rw [hblocksv]; omega
  have hloop := ptq1_0_row_loop0_refuse raw n out blocks e hbpos hblock0
  unfold dequant.lowbit.dequant_ptq1_0_row
  rw [hdiv]
  simp only [bind_tc_ok]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **`PTQ1_0` row short payload.** -/
theorem dequant_ptq1_0_row_short (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hshort : raw.val.length < 28) :
    dequant.lowbit.dequant_ptq1_0_row raw n out =
      ok (.Err (error.EngineError.Quant (quant.QuantError.Truncated 28#usize (Slice.len raw))),
        out) :=
  dequant_ptq1_0_row_refuse raw n out _ hn (ptq1_0_block_short raw hshort)

/-- **`PTQ1_0` row non-finite `d`.** -/
theorem dequant_ptq1_0_row_nonfinite_d (raw : Slice U8) (n : Usize) (out : alloc.vec.Vec I64)
    (hn : 0 < n.val) (hlen : 28 ≤ raw.val.length) (hd : f16Exp (f16LeHalf raw 26) = 31) :
    dequant.lowbit.dequant_ptq1_0_row raw n out =
      ok (.Err (error.EngineError.NonFiniteScale (f16LeHalf raw 26)), out) :=
  dequant_ptq1_0_row_refuse raw n out _ hn (ptq1_0_block_nonfinite_d raw hlen hd)

end

end apeiron_word
