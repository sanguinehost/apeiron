/-
Refinement of the extracted fixed-point projection kernel.

The Rust body is `src/engine/matvec.rs`; `i64` Q32.32. The public entry point is
`apeiron_word.word_matvec_fixed`, the extracted wrapper over `matvec.matvec_fixed`. The kernel
dequantizes each of `n_out` weight rows to Q32.32 and reduces it against the activation `x` by the
row-dot helper `matvec.row_dot_fixed`, giving `y[o] = round(Σ_i W[o,i]·x_i / 2^32)`.

## Dominant op

`matvec_fixed` is the dominant inference kernel: `O(n_out·n_in)` checked `i128` multiply-adds plus
one dequantization per weight element, `O(n_in)` scratch for the current row and `O(n_out)` output.
On the served projections — Q/K/V with the GQA head counts, the FFN and the LM head — the caller
fixes `n_out` and `n_in`; there is no float on the executing path. This module fixes the statement
of the per-coordinate rounding bound for the composed margin: with the dequantized row `deq r`
fixed by the class-(b) tested converter, each output coordinate is within `1/(2·q32)` of the real
dot `<deq r, x> = rowDotReal (deq r) x`.

## The dequant relation

The dequant-to-real relation is **not** re-proved here and is not assumed as a bound. It is the
class-(b) `tested` boundary (`dequant.dequant_row_fixed`, exercised by the `cargo test` suite, with
the per-quant code equalities of `Numeric.FormatRefinement` proving the block readers): the domain
`matvec_domain` carries the exact success result of the dequantizer on each row and the identity of
the produced row as `deq r`, and only the range windows `row_dot_domain (deq r) x`. The coordinate
bound is *derived* per row by `row_dot_refines_proved`; the dequant-to-ideal-weight reading remains
`tested`.

## Landed

The domain (`matvec_domain`), the refinement statement (`matvec_refines`) and the loop invariant
(`matvecLoopInv`) were on disk; this module proves the prologue reduction
(`word_matvec_fixed_unfold`), the refusal equalities (`word_matvec_fixed_err_refuses` and the named
`RowLength` / `Dequant` / `AccumulatorOverflow` / `OutputOverflow` wrappers, with the body-level
production equalities `matvec_fixed_loop_body_row_length_refuses`,
`matvec_fixed_loop_body_dequant_refuses`, `matvec_fixed_loop_body_dot_refuses`), the row-loop body
step (`matvec_fixed_loop_body_step`), the `loop.spec_decr_nat` assembly
(`matvec_fixed_loop_spec`) and the refinement (`matvec_refines_proved`).

## The pair-pattern reduction

The extracted body destructures the dequantizer result with a pair-pattern bind
(`let (r, row_buf3) ← dequant.dequant_row_fixed …`), an `Aeneas.Std.uncurry` whose `Prod` match
`dsimp`/`beta_reduce` do not iota (and `dsimp` additionally aborts on the adjacent scalar literal
proofs with the `instances`-transparency diagnostic). The reduction used here is the first-order
one: `simp only [ …, uncurry_apply_pair, … ]` rewrites `uncurry f (a, b)` to `f a b` on the
constructor pair and reduces the whole body in one pass; `rfl`-level defeq closes the residue. The
equivalent bind-by-bind form (`WP.spec_bind` with `intro ⟨r, row_buf3⟩`) reaches the same term.

No `sorry`/`admit`/`set_option`/`native_decide`/`maxHeartbeats 0`.
-/
import Numeric.EngineDotRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-- The stated success domain of `word_matvec_fixed`:

* `block_geom ggml_type = some (qk, bb)`, `x.len = n_in`, `div_ceil n_in qk = blocks` and the two
  checked strides are `row_bytes`/`need` with `need ≤ raw.len`;
* `deq r` is the dequantized row for every `r < n_out`, of length `n_in` and inside
  `row_dot_domain` (the range windows, not the bound);
* for every row index `o < n_out` the row slice and its dequantizer result are the loop's own
  (the class-(b) `tested` dequant boundary). -/
def matvec_domain
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64) : Prop :=
  quant.block_geom ggml_type = ok (some (qk, bb)) ∧
  x.val.length = n_in.val ∧
  core.num.Usize.div_ceil n_in qk = ok blocks ∧
  Usize.checked_mul blocks bb = some row_bytes ∧
  Usize.checked_mul n_out row_bytes = some need ∧
  need.val ≤ raw.val.length ∧
  (∀ (r : Nat), r < n_out.val → (deq r).val.length = n_in.val) ∧
  (∀ (r : Nat), r < n_out.val → row_dot_domain (deq r) x) ∧
  (∀ (o : Usize), o.val < n_out.val →
    ∃ (start i : Usize) (row : Slice U8),
      Usize.checked_mul o row_bytes = some start ∧
      start + row_bytes = ok i ∧
      core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) raw
        { start := start, «end» := i } = ok (some row)) ∧
  (∀ (o start i : Usize) (row : Slice U8) (rb : alloc.vec.Vec I64),
    o.val < n_out.val →
    Usize.checked_mul o row_bytes = some start →
    start + row_bytes = ok i →
    core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) raw
      { start := start, «end» := i } = ok (some row) →
    rb.val = [] →
    ∃ rb' : alloc.vec.Vec I64,
      dequant.dequant_row_fixed ggml_type row n_in rb = ok (core.result.Result.Ok (), rb') ∧
      rb'.val = (deq o.val).val)

/-- The target refinement statement. On `matvec_domain` the extracted `word_matvec_fixed` returns
a length-`n_out` vector whose `r`-th Q32.32 value is within `1/(2·q32)` of the real dot of the
dequantized row `deq r` with `x`. -/
def matvec_refines : Prop :=
  ∀ (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64),
    matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq →
      ∃ out : alloc.vec.Vec I64,
        word_matvec_fixed ggml_type raw n_out n_in x = ok (core.result.Result.Ok out) ∧
        out.val.length = n_out.val ∧
        ∀ (r : Nat) (hr : r < out.val.length),
          |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32)

/-- The row-loop invariant after `o` completed rows: the failure flag is clear, the error is
`none`, `o ≤ n_out`, and the accumulated output is exactly `o` values each within the row-dot
bound. This is the invariant the `loop.spec_decr_nat` assembly preserves. -/
def matvecLoopInv (n_out : Usize) (x : Slice I64) (deq : Nat → Slice I64)
    (out _row_buf : alloc.vec.Vec I64) (o : Usize) (ok : Bool)
    (err : Option matvec.MatvecError) : Prop :=
  ok = true ∧ err = none ∧ o.val ≤ n_out.val ∧
  out.val.length = o.val ∧
  (∀ (r : Nat) (hr : r < out.val.length),
    |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32))

/-! ## Shared bounds -/

/-- `n_out.val` is at most the `usize` maximum. -/
theorem n_out_le_max (n_out : Usize) : n_out.val ≤ Usize.max := by
  have h := UScalar.hBounds n_out
  rw [UScalarTy.Usize_numBits_eq] at h
  have hmax : (Usize.max : Nat) = 2 ^ System.Platform.numBits - 1 := by
    simp [Usize.max, Usize.numBits]
  omega

/-- The successor of any `o < n_out.val` is within `usize`. -/
theorem o_succ_le_max (n_out o : Usize) (ho : o.val < n_out.val) :
    o.val + (1#usize).val ≤ Usize.max := by
  have h1 : (1#usize).val = 1 := by norm_num
  have hnmax := n_out_le_max n_out
  rw [h1]
  omega

/-! ## Prologue reduction -/

/-- The extracted `word_matvec_fixed` after the prologue checks: on the domain equations the body
reduces to the row loop followed by the error epilogue. -/
theorem word_matvec_fixed_unfold
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length) :
    word_matvec_fixed ggml_type raw n_out n_in x =
      (do
        let (out1, err) ←
          matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
            (alloc.vec.Vec.with_capacity Std.I64 n_out)
            (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
        match err with
        | none => ok (core.result.Result.Ok out1)
        | some e => ok (core.result.Result.Err e)) := by
  have hlenx : Slice.len x = n_in := UScalar.eq_of_val_eq (by rw [Slice.len_val]; exact hxlen)
  have hcond : (Slice.len x != n_in) = false := by simp [hlenx]
  have hlenraw : ¬ (Slice.len raw < need) := by
    intro hc
    have h := (UScalar.lt_equiv (Slice.len raw) need).1 hc
    simp only [Slice.len_val, Slice.length] at h
    omega
  rw [word_matvec_fixed, matvec.matvec_fixed]
  simp only [hgeom, hcond, Bool.false_eq_true, if_false, hdiv, hrowb, hneed,
    if_neg hlenraw, lift, bind_tc_ok, uncurry_apply_pair,
    core.option.Option.ok_or_some, core.result.Result.Insts.CoreOpsTry.branch]
  rfl

/-! ## Refusal equalities -/

/-- **Any recorded refusal.** If the row loop terminates with `err = some e`, the wrapper returns
`Err e` exactly: no value is clamped and no error is rewritten. -/
theorem word_matvec_fixed_err_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length)
    (out1 : alloc.vec.Vec Std.I64) (e : matvec.MatvecError)
    (hloop : matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
      = ok (out1, some e)) :
    word_matvec_fixed ggml_type raw n_out n_in x = ok (core.result.Result.Err e) := by
  rw [word_matvec_fixed_unfold ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff]
  rw [hloop]
  simp only [bind_tc_ok, uncurry_apply_pair]

/-- **Row-length refusal.** A dequantized weight row whose length differs from `n_in` refuses with
`MatvecError.RowLength` carrying the loop's own row index and the two lengths. -/
theorem word_matvec_fixed_row_length_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length)
    (out1 : alloc.vec.Vec Std.I64) (row weights : Usize)
    (hloop : matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
      = ok (out1, some (matvec.MatvecError.RowLength row weights n_in))) :
    word_matvec_fixed ggml_type raw n_out n_in x =
      ok (core.result.Result.Err (matvec.MatvecError.RowLength row weights n_in)) :=
  word_matvec_fixed_err_refuses ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff out1 (matvec.MatvecError.RowLength row weights n_in) hloop

/-- **Dequant refusal.** A malformed or overflowing weight block refuses with
`MatvecError.Dequant e`, the dequantizer's own `EngineError`, unchanged. -/
theorem word_matvec_fixed_dequant_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length)
    (out1 : alloc.vec.Vec Std.I64) (e : error.EngineError)
    (hloop : matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
      = ok (out1, some (matvec.MatvecError.Dequant e))) :
    word_matvec_fixed ggml_type raw n_out n_in x =
      ok (core.result.Result.Err (matvec.MatvecError.Dequant e)) :=
  word_matvec_fixed_err_refuses ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff out1 (matvec.MatvecError.Dequant e) hloop

/-- **Accumulator refusal.** A row dot whose `i128` accumulator left its range refuses with
`MatvecError.AccumulatorOverflow` exactly (the `row_dot_fixed` refusal lifted unchanged). -/
theorem word_matvec_fixed_acc_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length)
    (out1 : alloc.vec.Vec Std.I64)
    (hloop : matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
      = ok (out1, some matvec.MatvecError.AccumulatorOverflow)) :
    word_matvec_fixed ggml_type raw n_out n_in x =
      ok (core.result.Result.Err matvec.MatvecError.AccumulatorOverflow) :=
  word_matvec_fixed_err_refuses ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff out1 matvec.MatvecError.AccumulatorOverflow hloop

/-- **Output refusal.** A rounded row value that does not fit `i64` refuses with
`MatvecError.OutputOverflow` exactly (the `row_dot_fixed` refusal lifted unchanged). -/
theorem word_matvec_fixed_out_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize)
    (hgeom : quant.block_geom ggml_type = ok (some (qk, bb)))
    (hxlen : x.val.length = n_in.val)
    (hdiv : core.num.Usize.div_ceil n_in qk = ok blocks)
    (hrowb : Usize.checked_mul blocks bb = some row_bytes)
    (hneed : Usize.checked_mul n_out row_bytes = some need)
    (hsuff : need.val ≤ raw.val.length)
    (out1 : alloc.vec.Vec Std.I64)
    (hloop : matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none
      = ok (out1, some matvec.MatvecError.OutputOverflow)) :
    word_matvec_fixed ggml_type raw n_out n_in x =
      ok (core.result.Result.Err matvec.MatvecError.OutputOverflow) :=
  word_matvec_fixed_err_refuses ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff out1 matvec.MatvecError.OutputOverflow hloop

/-- **Row-length production.** The loop body takes the `RowLength` branch exactly when a
successfully dequantized row has a length different from `n_in`: the flag drops, the error is
`some (RowLength o (Vec.len rb') n_in)`, and the row index advances as on the success path. -/
theorem matvec_fixed_loop_body_row_length_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (start i : Usize) (row : Slice U8) (rb' : alloc.vec.Vec I64)
    (ho : o.val < n_out.val)
    (hstart : Usize.checked_mul o row_bytes = some start)
    (hadd : start + row_bytes = ok i)
    (hget : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) raw
      { start := start, «end» := i } = ok (some row))
    (hdeq : dequant.dequant_row_fixed ggml_type row n_in
        { row_buf with slice := Slice.new Std.I64 }
      = ok (core.result.Result.Ok (), rb'))
    (hdif : alloc.vec.Vec.len rb' ≠ n_in) :
    ∃ o2 : Usize,
      matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o true none
        = ok (ControlFlow.cont (out, rb', o2, false,
          some (matvec.MatvecError.RowLength o (alloc.vec.Vec.len rb') n_in))) := by
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  obtain ⟨o2, hadd1, _⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (o_succ_le_max n_out o ho))
  refine ⟨o2, ?_⟩
  rw [matvec.matvec_fixed_loop.body]
  simp only [if_pos ((UScalar.lt_equiv o n_out).2 ho),
    hstart, hadd, hget, hclear, hdeq, if_neg hdif, hadd1,
    lift, bind_tc_ok, uncurry_apply_pair]
  rfl

/-- **Dequant production.** The loop body takes the `Dequant` branch exactly when the dequantizer
refuses a row: the flag drops and the error is `some (Dequant e)`. -/
theorem matvec_fixed_loop_body_dequant_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (start i : Usize) (row : Slice U8) (e : error.EngineError)
    (rb' : alloc.vec.Vec I64)
    (ho : o.val < n_out.val)
    (hstart : Usize.checked_mul o row_bytes = some start)
    (hadd : start + row_bytes = ok i)
    (hget : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) raw
      { start := start, «end» := i } = ok (some row))
    (hdeq : dequant.dequant_row_fixed ggml_type row n_in
        { row_buf with slice := Slice.new Std.I64 }
      = ok (core.result.Result.Err e, rb')) :
    ∃ o2 : Usize,
      matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o true none
        = ok (ControlFlow.cont (out, rb', o2, false,
          some (matvec.MatvecError.Dequant e))) := by
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  obtain ⟨o2, hadd1, _⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (o_succ_le_max n_out o ho))
  refine ⟨o2, ?_⟩
  rw [matvec.matvec_fixed_loop.body]
  simp only [if_pos ((UScalar.lt_equiv o n_out).2 ho),
    hstart, hadd, hget, hclear, hdeq, hadd1,
    lift, bind_tc_ok, uncurry_apply_pair]
  rfl

/-- **Dot refusal production.** When the row dot refuses with `e` (its two refusals are
`AccumulatorOverflow` and `OutputOverflow`, proved in `Numeric.EngineDotRefinement` as
`word_row_dot_fixed_acc_refuses` / `word_row_dot_fixed_out_high_refuses`), the loop body records
`some e` unchanged and drops the flag. -/
theorem matvec_fixed_loop_body_dot_refuses
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (row_bytes need : Usize) (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (start i : Usize) (row : Slice U8) (rb' : alloc.vec.Vec I64)
    (e : matvec.MatvecError)
    (ho : o.val < n_out.val)
    (hstart : Usize.checked_mul o row_bytes = some start)
    (hadd : start + row_bytes = ok i)
    (hget : core.slice.Slice.get (core.slice.index.SliceIndexRangeUsizeSlice Std.U8) raw
      { start := start, «end» := i } = ok (some row))
    (hdeq : dequant.dequant_row_fixed ggml_type row n_in
        { row_buf with slice := Slice.new Std.I64 }
      = ok (core.result.Result.Ok (), rb'))
    (hlenrow : alloc.vec.Vec.len rb' = n_in)
    (hdot : matvec.row_dot_fixed (alloc.vec.Vec.deref rb') x
      = ok (core.result.Result.Err e)) :
    ∃ o2 : Usize,
      matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o true none
        = ok (ControlFlow.cont (out, rb', o2, false, some e)) := by
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  obtain ⟨o2, hadd1, _⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (o_succ_le_max n_out o ho))
  refine ⟨o2, ?_⟩
  rw [matvec.matvec_fixed_loop.body]
  simp only [if_pos ((UScalar.lt_equiv o n_out).2 ho),
    hstart, hadd, hget, hclear, hdeq, if_pos hlenrow, hdot, hadd1,
    lift, bind_tc_ok, uncurry_apply_pair]
  rfl

/-! ## The bound survives one append -/

/-- The per-coordinate row-dot bound is preserved by appending the next rounded dot: the first
`out.val.length` entries are the old ones and the last entry is the new bound. -/
theorem bound_cons (deq : Nat → Slice I64) (x : Slice I64)
    (out out2 : alloc.vec.Vec Std.I64) (y : I64) (o : Usize)
    (hlen : out.val.length = o.val)
    (hpush : out2.val = out.val ++ [y])
    (hbound : ∀ (r : Nat) (hr : r < out.val.length),
      |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32))
    (hy : |toReal y - rowDotReal (deq o.val) x| ≤ 1 / (2 * q32)) :
    ∀ (r : Nat) (hr : r < out2.val.length),
      |toReal (out2.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32) := by
  have hlen2 : out2.val.length = out.val.length + 1 := by
    rw [hpush]
    simp
  intro r hr
  by_cases hrlt : r < out.val.length
  · have hq : out2.val[r]? = out.val[r]? := by
      rw [hpush, List.getElem?_append_left hrlt]
    have hget : out2.val[r]'hr = out.val[r]'hrlt := by
      have h1 := List.getElem?_eq_getElem (l := out2.val) (i := r) hr
      have h2 := List.getElem?_eq_getElem (l := out.val) (i := r) hrlt
      have h3 : some (out2.val[r]'hr) = some (out.val[r]'hrlt) :=
        h1.symm.trans (hq.trans h2)
      exact Option.some.inj h3
    rw [hget]
    exact hbound r hrlt
  · have hreq : r = out.val.length := by omega
    subst hreq
    have hq : out2.val[out.val.length]? = some y := by
      rw [hpush]
      simp
    have hget : out2.val[out.val.length]'hr = y := by
      have h1 := List.getElem?_eq_getElem (l := out2.val) (i := out.val.length) hr
      have h3 : some (out2.val[out.val.length]'hr) = some y := h1.symm.trans hq
      exact Option.some.inj h3
    rw [hget, hlen]
    exact hy

/-! ## The row-loop body step -/

/-- One row-loop body step on the success domain. With `o.val < n_out.val` and `matvecLoopInv`
holding, the body reads row `o`, dequantizes it to `deq o.val` (the class-(b) `tested` boundary),
reduces it against `x` through `row_dot_fixed`, appends the rounded dot and continues at `o + 1`
with the invariant restored.

The statement is the `WP.spec` over the extracted `matvec.matvec_fixed_loop.body` that
`loop.spec_decr_nat` consumes; its proof reduces the body in one first-order pass through
`uncurry_apply_pair`, destructing the dequantizer pair on its constructor value. -/
theorem matvec_fixed_loop_body_step
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (out row_buf : alloc.vec.Vec I64) (o : Usize)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq)
    (ho : o.val < n_out.val)
    (hinv : matvecLoopInv n_out x deq out row_buf o true none) :
    WP.spec (matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
        out row_buf o true none)
      (fun cf =>
        ∃ (out2 : alloc.vec.Vec Std.I64) (row_buf3 : alloc.vec.Vec Std.I64) (o2 : Usize),
          cf = ControlFlow.cont (out2, row_buf3, o2, true, none) ∧
          matvecLoopInv n_out x deq out2 row_buf3 o2 true none ∧
          o2.val = o.val + 1) := by
  obtain ⟨_hgeom, _hxlen, _hdiv, _hrowb, _hneed, _hsuff, hdeqlen, hdotdom, hslice, hdeq⟩ := hdom
  obtain ⟨_hok, _herr, _ho', hlen, hbound⟩ := hinv
  obtain ⟨start, i, row, hstart, hadd, hget⟩ := hslice o ho
  -- the dequantized row of this iteration, on the cleared scratch
  have hclear : (alloc.vec.Vec.clear Global row_buf
      : Result (alloc.vec.Vec Std.I64)) = ok { row_buf with slice := Slice.new Std.I64 } := rfl
  have hcleared : ({ row_buf with slice := Slice.new Std.I64 } : alloc.vec.Vec Std.I64).val = [] := by
    simp [alloc.vec.Vec.val, Slice.new_val]
  obtain ⟨rb', hdeq_eq, hdeq_val⟩ :=
    hdeq o start i row { row_buf with slice := Slice.new Std.I64 } ho hstart hadd hget hcleared
  -- the row-dot result on the dequantized row
  have hsval : (alloc.vec.Vec.deref rb').val = rb'.val := by
    simp [alloc.vec.Vec.deref]
  have hseq : alloc.vec.Vec.deref rb' = deq o.val := by
    apply Slice.ext
    rw [hsval, hdeq_val]
  have hdotdom' : row_dot_domain (alloc.vec.Vec.deref rb') x := by
    rw [hseq]
    exact hdotdom o.val ho
  obtain ⟨y, hy, hboundy⟩ := row_dot_refines_proved (alloc.vec.Vec.deref rb') x hdotdom'
  have hy' : matvec.row_dot_fixed (alloc.vec.Vec.deref rb') x
      = ok (core.result.Result.Ok y) := hy
  have hboundy' : |toReal y - rowDotReal (deq o.val) x| ≤ 1 / (2 * q32) := by
    rw [← hseq]
    exact hboundy
  have hlenrow : alloc.vec.Vec.len rb' = n_in := by
    apply UScalar.eq_of_val_eq
    rw [alloc.vec.Vec.len_val]
    show rb'.val.length = n_in.val
    rw [hdeq_val]
    exact hdeqlen o.val ho
  -- push and index witnesses
  have hnmax := n_out_le_max n_out
  have hpushmax : out.val.length < Usize.max := by
    rw [hlen]
    omega
  obtain ⟨out2, hpush, hpushv⟩ :=
    WP.spec_imp_exists (alloc.vec.Vec.push_spec out y hpushmax)
  obtain ⟨o2, hadd1, ho2⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := o) (y := 1#usize) (o_succ_le_max n_out o ho))
  -- the body step itself
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (out2, rb', o2, true, none), ?_, ?_⟩
  · rw [matvec.matvec_fixed_loop.body]
    simp only [if_pos ((UScalar.lt_equiv o n_out).2 ho),
      hstart, hadd, hget, hclear, hdeq_eq, if_pos hlenrow, hy', hpush, hadd1,
      lift, bind_tc_ok, uncurry_apply_pair]
    rfl
  · refine ⟨out2, rb', o2, rfl, ?_, ho2⟩
    refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · have h1 : (1#usize).val = 1 := by norm_num
      omega
    · have h1 : (1#usize).val = 1 := by norm_num
      have hlen2 : out2.val.length = out.val.length + 1 := by rw [hpushv]; simp
      omega
    · exact bound_cons deq x out out2 y o hlen hpushv hbound hboundy'

/-! ## The loop assembly -/

/-- The row loop on `matvec_domain`: `loop.spec_decr_nat` with `matvecLoopInv` and the measure
`n_out - o`. The loop returns the length-`n_out` output whose coordinates are all within the
row-dot bound. -/
theorem matvec_fixed_loop_spec
    (ggml_type : U32) (raw : Slice U8) (n_out n_in : Usize) (x : Slice I64)
    (qk bb blocks row_bytes need : Usize) (deq : Nat → Slice I64)
    (hdom : matvec_domain ggml_type raw n_out n_in x qk bb blocks row_bytes need deq) :
    WP.spec (matvec.matvec_fixed_loop ggml_type raw n_out n_in x row_bytes need
        (alloc.vec.Vec.with_capacity Std.I64 n_out)
        (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none)
      (fun b =>
        ∃ out : alloc.vec.Vec I64, b = (out, none) ∧
          out.val.length = n_out.val ∧
          ∀ (r : Nat) (hr : r < out.val.length),
            |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32)) := by
  have hinv0 : matvecLoopInv n_out x deq
      (alloc.vec.Vec.with_capacity Std.I64 n_out)
      (alloc.vec.Vec.with_capacity Std.I64 n_in) 0#usize true none := by
    refine ⟨rfl, rfl, ?_, ?_, ?_⟩
    · show (0#usize).val ≤ n_out.val
      have h1 : (0#usize).val = 0 := by norm_num
      omega
    · show (alloc.vec.Vec.with_capacity Std.I64 n_out).val.length = (0#usize).val
      have h1 : (0#usize).val = 0 := by norm_num
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val, h1]
    · show ∀ (r : Nat) (hr : r < (alloc.vec.Vec.with_capacity Std.I64 n_out).val.length),
        |toReal ((alloc.vec.Vec.with_capacity Std.I64 n_out).val[r]'hr)
          - rowDotReal (deq r) x| ≤ 1 / (2 * q32)
      intro r hr
      simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val] at hr
  unfold matvec.matvec_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun s : (alloc.vec.Vec Std.I64) × (alloc.vec.Vec Std.I64) ×
        (Std.Usize × Bool × (Option matvec.MatvecError)) =>
      n_out.val - s.2.2.1.val)
    (inv := fun s => matvecLoopInv n_out x deq s.1 s.2.1 s.2.2.1 s.2.2.2.1 s.2.2.2.2)
    (post := fun b : (alloc.vec.Vec Std.I64) × (Option matvec.MatvecError) =>
      ∃ out : alloc.vec.Vec I64, b = (out, none) ∧
        out.val.length = n_out.val ∧
        ∀ (r : Nat) (hr : r < out.val.length),
          |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32))
  · intro s hs
    rcases s with ⟨out, row_buf, o, ok1, err⟩
    obtain ⟨hok, herr, ho, hlen, hbound⟩ := hs
    have hok' : ok1 = true := hok
    have herr' : err = none := herr
    have ho' : o.val ≤ n_out.val := ho
    have hlen' : out.val.length = o.val := hlen
    have hbound' : ∀ (r : Nat) (hr : r < out.val.length),
        |toReal (out.val[r]'hr) - rowDotReal (deq r) x| ≤ 1 / (2 * q32) := hbound
    rw [hok', herr']
    by_cases hlt : o.val < n_out.val
    · obtain ⟨cf, hcf, hpost⟩ := WP.spec_imp_exists
        (matvec_fixed_loop_body_step ggml_type raw n_out n_in x qk bb blocks row_bytes need deq
          out row_buf o hdom hlt ⟨rfl, rfl, ho', hlen', hbound'⟩)
      obtain ⟨out2, row_buf2, o2, hcfg, hinv2, ho2⟩ := hpost
      rw [hcfg] at hcf
      simp only [hcf]
      refine ⟨hinv2, ?_⟩
      show n_out.val - o2.val < n_out.val - o.val
      omega
    · have hnot : ¬ (o < n_out) := by
        intro hc
        exact hlt ((UScalar.lt_equiv o n_out).1 hc)
      have hbody : matvec.matvec_fixed_loop.body ggml_type raw n_out n_in x row_bytes need
          out row_buf o true none
          = ok (ControlFlow.done (out, none)) := by
        rw [matvec.matvec_fixed_loop.body, if_neg hnot]
      simp only [hbody]
      refine ⟨out, rfl, ?_, hbound'⟩
      rw [hlen']
      omega
  · exact hinv0

/-! ## The refinement -/

/-- **Matvec refinement.** On `matvec_domain` the extracted `word_matvec_fixed` returns the
length-`n_out` output whose `r`-th Q32.32 value is within `1/(2·q32)` of the real dot
`rowDotReal (deq r) x` of the dequantized row with the activation. The bound is derived per row by
`row_dot_refines_proved`; the dequant-to-ideal-weight reading stays the class-(b) `tested`
boundary. -/
theorem matvec_refines_proved : matvec_refines := by
  intro ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom
  obtain ⟨b, hb, out, hbout, hlenout, hbound⟩ := WP.spec_imp_exists
    (matvec_fixed_loop_spec ggml_type raw n_out n_in x qk bb blocks row_bytes need deq hdom)
  obtain ⟨hgeom, hxlen, hdiv, hrowb, hneed, hsuff, _hdeqlen, _hdotdom, _hslice, _hdeq⟩ := hdom
  rw [hbout] at hb
  refine ⟨out, ?_, hlenout, hbound⟩
  rw [word_matvec_fixed_unfold ggml_type raw n_out n_in x qk bb blocks row_bytes need
    hgeom hxlen hdiv hrowb hneed hsuff]
  rw [hb]
  simp only [bind_tc_ok, uncurry_apply_pair]

end

end apeiron_word
