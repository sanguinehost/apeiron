/-
Refinement of the extracted fixed-point row dot kernel.

The Rust body is `src/engine/matvec.rs`; `i64` Q32.32. The public entry point is
`apeiron_word.word_row_dot_fixed`, the extracted wrapper over `matvec.row_dot_fixed`.
The documented specification (`matvec.rs:56-67`) is

* `w_row` and `x` are Q32.32 rows of equal length `n`;
* every product `w_i·x_i` is an `i64` square-scale quantity, so the dot
  `Σ_i w_i·x_i` is accumulated exactly in `i128`;
* the single reduction is the value-path epilogue `value.round_shift_32`, one
  round-to-nearest `>> 32` (ties upward): `y = round(Σ_i w_i·x_i / 2^32)`;
* the refusals are `Err RowLength` when the lengths differ, `Err AccumulatorOverflow`
  when the `i128` accumulator leaves its range, and `Err OutputOverflow` when the
  rounded value does not fit `i64`. No value is clamped.

`Numeric.EngineNormRefinement` is imported for the proved `round_shift_32_spec` epilogue
lemma and the checked-`i128` witnesses (`I128_checked_mul_some`); the epilogue lemma is
reused, not restated.

## Dominant-kernel complexity

`matvec_fixed` (`src/engine/matvec.rs:143`) calls this helper once per output row, so on the
served projections — Q/K/V at the GQA head counts (`n_head·head_dim` and `n_head·kv_head·head_dim`
rows over `n_embd`), the FFN (`n_ff` rows) and the LM head (`vocab` rows) — the dot discharges
the dominant arithmetic: `O(n_out·n_in)` checked `i128` multiply-adds plus one dequantization
per weight element, with `O(n_in)` scratch for the current row and `O(n_out)` output. The
row loop of this module visits each of the `n` positions once: the recurrence
`acc_{k+1} = acc_k + w_k·x_k` closes after `n` steps, `O(n)` time in checked multiply-adds,
`O(1)` extra space. `round_shift_32` is `O(1)` word operations.

This module lands the domain (`row_dot_domain`), the real reference (`rowDotReal`), the
refinement statement/proof (`row_dot_refines`, `row_dot_refines_proved`), the accumulator's
exact-integer identity (`row_dot_fixed_loop_eq`), and the refusal equalities
(`word_row_dot_fixed_row_length`, `word_row_dot_fixed_acc_refuses`,
`word_row_dot_fixed_out_high_refuses`, `word_row_dot_fixed_out_low_refuses`) available by
structural reduction of the extracted body.
-/
import Numeric.EngineNormRefinement

namespace apeiron_word

open Aeneas Aeneas.Std Result

noncomputable section

/-! ## The exact integer dot and the real reference

`dotPrefix` is the exact ℤ-valued prefix `Σ_{j<k} w_j·x_j` of the row pair; `rowDotReal` is the
real dot product `<w,x> = Σ_i toReal w_i · toReal x_i` over the shorter of the two rows (both, on
the domain). `toReal` is the signed Q32.32 denotation from `Numeric.Scale`. -/

/-- The exact ℤ prefix `Σ_{j<k} w_j·x_j` of the two rows, `0` for `k = 0`. -/
def dotPrefix (w_row x : Slice I64) (k : Nat) : ℤ :=
  (((w_row.val.take k).zip (x.val.take k)).map (fun p => p.1.val * p.2.val)).sum

/-- The real dot product `Σ_i toReal w_i · toReal x_i`. -/
def rowDotReal (w_row x : Slice I64) : ℝ :=
  ((w_row.val.zip x.val).map (fun p => toReal p.1 * toReal p.2)).sum

/-- The exact ℤ prefix does not depend on terms past the common length: at `k = n` both
`w_row` and `x` are consumed whole when the lengths agree. -/
theorem dotPrefix_full (w_row x : Slice I64) (h : w_row.val.length = x.val.length) :
    dotPrefix w_row x w_row.val.length =
      ((w_row.val.zip x.val).map (fun p => p.1.val * p.2.val)).sum := by
  unfold dotPrefix
  rw [List.take_length]
  have hx : x.val.take w_row.val.length = x.val := by rw [h, List.take_length]
  rw [hx]

/-- Single-term real/integer bridge: `toReal a · toReal b = (a.val·b.val)/2^64`. -/
theorem toReal_mul_eq (a b : I64) :
    toReal a * toReal b = ((a.val : ℝ) * (b.val : ℝ)) / q32^2 := by
  unfold toReal q32
  ring_nf

/-- Per-pair bridge over a zipped list: the sum of real products is the exact integer sum
divided by `2^64`. -/
theorem sum_toReal_zip (l : List (I64 × I64)) :
    (l.map (fun p => toReal p.1 * toReal p.2)).sum =
      (((l.map (fun p => p.1.val * p.2.val)).sum : ℤ) : ℝ) / q32^2 := by
  induction l with
  | nil => simp
  | cons p tl ih =>
      obtain ⟨a, b⟩ := p
      simp only [List.map_cons, List.sum_cons]
      rw [toReal_mul_eq, ih, Int.cast_add, Int.cast_mul]
      ring

/-- The real reference is the exact integer dot over `2^64`. -/
theorem rowDotReal_eq (w_row x : Slice I64) (h : w_row.val.length = x.val.length) :
    rowDotReal w_row x = (dotPrefix w_row x w_row.val.length : ℝ) / q32^2 := by
  unfold rowDotReal
  rw [dotPrefix_full w_row x h]
  exact sum_toReal_zip (w_row.val.zip x.val)

/-! ## The documented domain and reference

The domain carries the equal-length precondition, the per-prefix `i128` window that keeps the
checked accumulator on its success path, and the `i64` output window that keeps the epilogue
narrowing on the `Ok` arm. -/

/-- The documented domain of `row_dot_fixed`: equal lengths; every nonempty prefix sum of
products inside `i128` (the accumulator never refuses); and the single epilogue rounding inside
`i64` (the output never refuses). -/
def row_dot_domain (w_row x : Slice I64) : Prop :=
  w_row.val.length = x.val.length ∧
  (∀ k, 1 ≤ k → k ≤ w_row.val.length →
      I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) ∧
  -((2 : ℤ) ^ (63 : ℕ)) ≤
      divRoundNearest (dotPrefix w_row x w_row.val.length) ((2 : ℤ) ^ (32 : ℕ)) ∧
  divRoundNearest (dotPrefix w_row x w_row.val.length) ((2 : ℤ) ^ (32 : ℕ)) ≤
      (2 : ℤ) ^ (63 : ℕ) - 1

/-- The target refinement statement: on `row_dot_domain` the extracted body returns an `i64`
within the single round-to-nearest epilogue bound `1/(2·2^32)` of the real dot product. -/
def row_dot_refines : Prop :=
  ∀ (w_row x : Slice I64), row_dot_domain w_row x →
    ∃ y : I64,
      word_row_dot_fixed w_row x = ok (core.result.Result.Ok y) ∧
      |toReal y - rowDotReal w_row x| ≤ 1 / (2 * q32)

/-! ## Product and prefix lemmas -/

/-- A product of two `i64`-range integers fits `i128`. -/
theorem i64_prod_range (a b : ℤ)
    (ha : -((2 : ℤ) ^ (63 : ℕ)) ≤ a) (ha' : a ≤ (2 : ℤ) ^ (63 : ℕ) - 1)
    (hb : -((2 : ℤ) ^ (63 : ℕ)) ≤ b) (hb' : b ≤ (2 : ℤ) ^ (63 : ℕ) - 1) :
    -((2 : ℤ) ^ (127 : ℕ)) ≤ a * b ∧ a * b ≤ (2 : ℤ) ^ (127 : ℕ) - 1 := by
  have habs : |a * b| ≤ (2 : ℤ) ^ (126 : ℕ) := by
    rw [abs_mul]
    have ha2 : |a| ≤ (2 : ℤ) ^ (63 : ℕ) := abs_le.mpr ⟨by linarith, by linarith⟩
    have hb2 : |b| ≤ (2 : ℤ) ^ (63 : ℕ) := abs_le.mpr ⟨by linarith, by linarith⟩
    calc |a| * |b| ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (63 : ℕ) :=
          mul_le_mul ha2 hb2 (abs_nonneg _) (by positivity)
      _ = (2 : ℤ) ^ (126 : ℕ) := by norm_num
  have hle := abs_le.mp habs
  constructor <;> linarith [hle.1, hle.2]

/-- The exact prefix steps by the term at the index. -/
theorem dotPrefix_succ (w_row x : Slice I64) {i : Nat}
    (hiw : i < w_row.val.length) (hix : i < x.val.length) :
    dotPrefix w_row x (i + 1) =
      dotPrefix w_row x i + w_row.val[i].val * x.val[i].val := by
  unfold dotPrefix
  rw [List.take_add_one, List.take_add_one]
  rw [List.getElem?_eq_getElem hiw, List.getElem?_eq_getElem hix]
  simp only [Option.toList_some]
  have hlen : (w_row.val.take i).length = (x.val.take i).length := by
    rw [List.length_take, List.length_take, Nat.min_eq_left (le_of_lt hiw),
      Nat.min_eq_left (le_of_lt hix)]
  rw [List.zip_append hlen]
  simp only [List.zip_cons_cons, List.zip_nil_right, List.map_append, List.sum_append,
    List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  ring

/-! ## Refusal equalities -/

/-- **Length refusal.** Unequal row lengths return `Err RowLength` exactly. -/
theorem word_row_dot_fixed_row_length (w_row x : Slice I64)
    (h : w_row.val.length ≠ x.val.length) :
    word_row_dot_fixed w_row x =
      ok (core.result.Result.Err
        (matvec.MatvecError.RowLength 0#usize (Slice.len w_row) (Slice.len x))) := by
  have hlen' : Slice.len w_row ≠ Slice.len x := by
    intro hc
    have hv : (Slice.len w_row).val = (Slice.len x).val :=
      congrArg (fun u : Usize => u.val) hc
    rw [Slice.len_val w_row, Slice.len_val x] at hv
    exact h hv
  have hcond : (Slice.len w_row != Slice.len x) = true := by simpa using hlen'
  rw [word_row_dot_fixed, matvec.row_dot_fixed]
  dsimp only
  rw [if_pos hcond]

/-- **`i128` accumulator refusal.** A loop that stops with the failure flag set returns
`Err AccumulatorOverflow` exactly. -/
theorem word_row_dot_fixed_acc_refuses (w_row x : Slice I64) (S : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, false)) :
    word_row_dot_fixed w_row x =
      ok (core.result.Result.Err matvec.MatvecError.AccumulatorOverflow) := by
  have hlen' : Slice.len w_row = Slice.len x := by
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val w_row, Slice.len_val x]
    exact hlen
  have hcond : (Slice.len w_row != Slice.len x) = false := by simp [hlen']
  rw [word_row_dot_fixed, matvec.row_dot_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [hloop]
  simp

/-- The extracted body after the length test and a successful accumulator loop, as a `do`
block. -/
theorem word_row_dot_fixed_unfold (w_row x : Slice I64) (S : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true)) :
    word_row_dot_fixed w_row x =
      (do
        let rounded ← value.round_shift_32 S
        let i2 ← lift (IScalar.cast .I128 core.num.I64.MAX)
        if rounded > i2
        then ok (core.result.Result.Err matvec.MatvecError.OutputOverflow)
        else
          let i3 ← lift (IScalar.cast .I128 core.num.I64.MIN)
          if rounded < i3
          then ok (core.result.Result.Err matvec.MatvecError.OutputOverflow)
          else
            let i4 ← lift (IScalar.cast .I64 rounded)
            ok (core.result.Result.Ok i4)) := by
  have hlen' : Slice.len w_row = Slice.len x := by
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val w_row, Slice.len_val x]
    exact hlen
  have hcond : (Slice.len w_row != Slice.len x) = false := by simp [hlen']
  rw [word_row_dot_fixed, matvec.row_dot_fixed]
  dsimp only
  rw [hcond]
  simp only [Bool.false_eq_true, if_false]
  rw [hloop]
  simp only [bind_tc_ok]
  rfl

/-- **Upper `i64` narrowing refusal.** A rounded value above `i64::MAX` returns
`Err OutputOverflow` exactly. -/
theorem word_row_dot_fixed_out_high_refuses (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : value.round_shift_32 S = ok rounded)
    (hhigh : rounded > IScalar.cast .I128 core.num.I64.MAX) :
    word_row_dot_fixed w_row x =
      ok (core.result.Result.Err matvec.MatvecError.OutputOverflow) := by
  rw [word_row_dot_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_pos hhigh]

/-- **Lower `i64` narrowing refusal.** A rounded value below `i64::MIN` returns
`Err OutputOverflow` exactly. -/
theorem word_row_dot_fixed_out_low_refuses (w_row x : Slice I64) (S rounded : I128)
    (hlen : w_row.val.length = x.val.length)
    (hloop : matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true))
    (hround : value.round_shift_32 S = ok rounded)
    (hhigh : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX))
    (hlow : rounded < IScalar.cast .I128 core.num.I64.MIN) :
    word_row_dot_fixed w_row x =
      ok (core.result.Result.Err matvec.MatvecError.OutputOverflow) := by
  rw [word_row_dot_fixed_unfold w_row x S hlen hloop]
  rw [hround]
  simp only [lift, bind_tc_ok]
  rw [if_neg hhigh, if_pos hlow]

/-! ## The exact accumulator identity

`matvec.row_dot_fixed_loop` is the extracted loop; on the prefix window it terminates with
`(S, true)` and `S.val` is the exact ℤ prefix at `n`. -/

/-- The loop invariant: the failure flag is clear, the index stays at most `n`, and the
accumulator is the exact ℤ prefix at the index. -/
def dotLoopInv (w_row x : Slice I64) (n : Usize) (s : I128 × Usize × Bool) : Prop :=
  s.2.2 = true ∧ s.2.1.val ≤ n.val ∧ s.1.val = dotPrefix w_row x s.2.1.val

/-- One loop-body step: with `i < n` the body reads both rows, multiplies in `i128`, adds into
the accumulator and advances; the checked operations stay on their `some` arms exactly when the
next prefix is inside the `i128` window. -/
theorem row_dot_fixed_loop_body_step (w_row x : Slice I64) (n : Usize) (acc : I128)
    (i : Usize)
    (hilt : i.val < n.val) (hn : n.val = w_row.val.length)
    (hxlen : x.val.length = w_row.val.length)
    (hacc : acc.val = dotPrefix w_row x i.val)
    (hfit : ∀ k, 1 ≤ k → k ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) :
    ∃ total inext, matvec.row_dot_fixed_loop.body w_row x acc i true
        = ok (ControlFlow.cont (total, inext, true)) ∧
      total.val = dotPrefix w_row x inext.val ∧ inext.val = i.val + 1 := by
  have hbnd : i.val < w_row.val.length := by omega
  have hbndx : i.val < x.val.length := by omega
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  obtain ⟨iw, hiw, hiwv⟩ := WP.spec_imp_exists (Slice.index_usize_spec w_row i hbnd)
  have hiw' : iw.val = w_row.val[i.val].val := by rw [hiwv]
  obtain ⟨ix, hix, hixv⟩ := WP.spec_imp_exists (Slice.index_usize_spec x i hbndx)
  have hix' : ix.val = x.val[i.val].val := by rw [hixv]
  have haw : (core.convert.num.FromI128I64.from iw).val = iw.val :=
    core.convert.num.FromI128I64.from_val_eq iw
  have hbx : (core.convert.num.FromI128I64.from ix).val = ix.val :=
    core.convert.num.FromI128I64.from_val_eq ix
  have hbw := iw.hBounds
  rw [IScalarTy.I64_numBits_eq] at hbw
  norm_num at hbw
  have hbx' := ix.hBounds
  rw [IScalarTy.I64_numBits_eq] at hbx'
  norm_num at hbx'
  have hwb1 : -((2 : ℤ) ^ (63 : ℕ)) ≤ iw.val := by linarith [hbw.1]
  have hwb2 : iw.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by linarith [hbw.2]
  have hxb1 : -((2 : ℤ) ^ (63 : ℕ)) ≤ ix.val := by linarith [hbx'.1]
  have hxb2 : ix.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by linarith [hbx'.2]
  have hprange := i64_prod_range iw.val ix.val hwb1 hwb2 hxb1 hxb2
  have hmullo : I128.min ≤ (core.convert.num.FromI128I64.from iw).val *
      (core.convert.num.FromI128I64.from ix).val := by
    rw [hI128min, haw, hbx]; exact hprange.1
  have hmulhi : (core.convert.num.FromI128I64.from iw).val *
      (core.convert.num.FromI128I64.from ix).val ≤ I128.max := by
    rw [hI128max, haw, hbx]; exact hprange.2
  obtain ⟨prod, hprod_eq, hprodv⟩ := I128_checked_mul_some _ _ hmullo hmulhi
  have hprodval : prod.val = w_row.val[i.val].val * x.val[i.val].val := by
    rw [hprodv, haw, hbx, hiw', hix']
  have hsuc : acc.val + prod.val = dotPrefix w_row x (i.val + 1) := by
    rw [hacc, hprodval, dotPrefix_succ w_row x hbnd hbndx]
  have hk1 : 1 ≤ i.val + 1 := by omega
  have hk2 : i.val + 1 ≤ w_row.val.length := by omega
  have hpb := hfit (i.val + 1) hk1 hk2
  have hsumlo : I128.min ≤ acc.val + prod.val := by rw [hsuc]; exact hpb.1
  have hsumhi : acc.val + prod.val ≤ I128.max := by rw [hsuc]; exact hpb.2
  obtain ⟨total, htot_eq, htotv⟩ := I128_checked_add_some acc prod hsumlo hsumhi
  have hiaddmax : i.val + (1#usize).val ≤ Usize.max := by
    have h1 : (1#usize).val = 1 := by norm_num
    have hb := w_row.property
    rw [h1]; omega
  obtain ⟨inext, hinext, hinextv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) hiaddmax)
  refine ⟨total, inext, ?_, ?_, hinextv⟩
  · rw [matvec.row_dot_fixed_loop.body, hiw, hix]
    simp only [lift, bind_tc_ok]
    rw [hprod_eq]
    dsimp only
    rw [htot_eq]
    dsimp only
    rw [hinext]
    simp only [bind_tc_ok]
    have hilt' : i < Slice.len w_row :=
      (UScalar.lt_equiv i (Slice.len w_row)).2 (by rw [Slice.len_val]; exact hbnd)
    rw [if_pos hilt', if_pos trivial]
    rfl
  · rw [htotv, hsuc]
    have h1 : (1#usize).val = 1 := by norm_num
    rw [hinextv, h1]

/-- On the prefix window the loop terminates with the exact ℤ prefix at `n`. -/
theorem row_dot_fixed_loop_spec (w_row x : Slice I64)
    (hxlen : x.val.length = w_row.val.length)
    (hfit : ∀ k, 1 ≤ k → k ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) :
    matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true ⦃ b =>
      ∃ S : I128, b = (S, true) ∧ S.val = dotPrefix w_row x w_row.val.length ⦄ := by
  unfold matvec.row_dot_fixed_loop
  apply loop.spec_decr_nat
    (measure := fun s : I128 × Usize × Bool => (Slice.len w_row).val - s.2.1.val)
    (inv := dotLoopInv w_row x (Slice.len w_row))
    (post := fun b : I128 × Bool =>
      ∃ S : I128, b = (S, true) ∧ S.val = dotPrefix w_row x w_row.val.length)
  · intro s hs
    rcases s with ⟨acc, i, acc_ok⟩
    dsimp only at hs ⊢
    obtain ⟨hok, hile, hsum⟩ := hs
    dsimp only at hok hile hsum
    rw [hok]
    by_cases hlt : i.val < (Slice.len w_row).val
    · obtain ⟨total, inext, hbody, htot, hinext⟩ :=
        row_dot_fixed_loop_body_step w_row x (Slice.len w_row) acc i hlt
          (by rw [Slice.len_val]) hxlen hsum hfit
      rw [hbody]
      have hinextle : inext.val ≤ (Slice.len w_row).val := by
        rw [hinext]; omega
      refine ⟨⟨rfl, hinextle, htot⟩, ?_⟩
      show (Slice.len w_row).val - inext.val < (Slice.len w_row).val - i.val
      omega
    · have hnot : ¬ (i < Slice.len w_row) := by
        intro hc; exact hlt ((UScalar.lt_equiv i (Slice.len w_row)).1 hc)
      have hie : i.val = (Slice.len w_row).val := by omega
      have hlen : (Slice.len w_row).val = w_row.val.length := Slice.len_val w_row
      have hbody : matvec.row_dot_fixed_loop.body w_row x acc i true
          = ok (ControlFlow.done (acc, true)) := by
        rw [matvec.row_dot_fixed_loop.body, if_neg hnot]
      rw [hbody]
      exact ⟨acc, rfl, by rw [hsum, hie, hlen]⟩
  · refine ⟨rfl, ?_, ?_⟩
    · simp
    · show (0#i128).val = dotPrefix w_row x (0#usize).val
      rw [dotPrefix]
      simp

/-- Existential form of the loop identity. -/
theorem row_dot_fixed_loop_eq (w_row x : Slice I64)
    (hxlen : x.val.length = w_row.val.length)
    (hfit : ∀ k, 1 ≤ k → k ≤ w_row.val.length →
        I128.min ≤ dotPrefix w_row x k ∧ dotPrefix w_row x k ≤ I128.max) :
    ∃ S : I128,
      S.val = dotPrefix w_row x w_row.val.length ∧
      matvec.row_dot_fixed_loop w_row x 0#i128 0#usize true = ok (S, true) := by
  obtain ⟨y, hy, S, hyS, hS⟩ :=
    WP.spec_imp_exists (row_dot_fixed_loop_spec w_row x hxlen hfit)
  exact ⟨S, hS, by rw [hy, hyS]⟩

/-! ## The epilogue bound -/

/-- The single round-to-nearest reduction against the exact integer dot: for
`y.val = divRoundNearest S 2^32`, the real output is within `1/(2·2^32)` of `S/2^64`. -/
theorem dot_rounding_bound (S : ℤ) (y : I64)
    (hy : y.val = divRoundNearest S ((2 : ℤ) ^ (32 : ℕ))) :
    |toReal y - (S : ℝ) / q32^2| ≤ 1 / (2 * q32) := by
  set Z : ℤ := y.val * (2 : ℤ) ^ (32 : ℕ) - S with hZdef
  have hint : |Z| ≤ (2 : ℤ) ^ (31 : ℕ) := by
    have h := div_round_nearest_abs_bound S ((2 : ℤ) ^ (32 : ℕ)) (by norm_num)
    rw [show (2 : ℤ) ^ (32 : ℕ) / 2 = (2 : ℤ) ^ (31 : ℕ) from by norm_num] at h
    rw [hZdef, hy]
    exact h
  have hZid : toReal y - (S : ℝ) / q32^2 = ((Z : ℤ) : ℝ) / ((2 : ℝ) ^ (64 : ℕ)) := by
    rw [hZdef]
    unfold toReal q32
    push_cast
    ring
  have hZR : |((Z : ℤ) : ℝ)| ≤ (2 : ℝ) ^ (31 : ℕ) := by exact_mod_cast hint
  rw [hZid, abs_div, abs_of_pos (by positivity : (0 : ℝ) < (2 : ℝ) ^ (64 : ℕ))]
  calc |((Z : ℤ) : ℝ)| / (2 : ℝ) ^ (64 : ℕ)
      ≤ (2 : ℝ) ^ (31 : ℕ) / (2 : ℝ) ^ (64 : ℕ) :=
        div_le_div_of_nonneg_right hZR (by positivity)
    _ = 1 / (2 * q32) := by rw [show q32 = (2 : ℝ) ^ (32 : ℕ) from rfl]; norm_num

/-! ## The refinement -/

/-- **Row dot refinement.** On `row_dot_domain` the extracted `word_row_dot_fixed` returns an
`i64` within `1/(2·2^32)` of the real dot product. -/
theorem row_dot_refines_proved : row_dot_refines := by
  intro w_row x hdom
  obtain ⟨hlen, hfit, houtlo, houthi⟩ := hdom
  have hxlen : x.val.length = w_row.val.length := hlen.symm
  obtain ⟨S, hSval, hloop⟩ := row_dot_fixed_loop_eq w_row x hxlen hfit
  have hI128min : I128.min = -(2 : ℤ) ^ (127 : ℕ) := by
    norm_num [I128.min, IScalar.min, I128.numBits]
  have hI128max : I128.max = (2 : ℤ) ^ (127 : ℕ) - 1 := by
    norm_num [I128.max, IScalar.max, I128.numBits]
  set yv : ℤ := divRoundNearest S.val ((2 : ℤ) ^ (32 : ℕ)) with hyvdef
  have hyvb : -((2 : ℤ) ^ (63 : ℕ)) ≤ yv ∧ yv ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [hyvdef, hSval]; exact ⟨houtlo, houthi⟩
  have hZ := div_round_nearest_abs_bound S.val ((2 : ℤ) ^ (32 : ℕ)) (by norm_num)
  rw [show (2 : ℤ) ^ (32 : ℕ) / 2 = (2 : ℤ) ^ (31 : ℕ) from by norm_num] at hZ
  have hZ' : |yv * (2 : ℤ) ^ (32 : ℕ) - S.val| ≤ (2 : ℤ) ^ (31 : ℕ) := by
    rw [hyvdef]; exact hZ
  have hSbound : |S.val| ≤ (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (31 : ℕ) := by
    have hyabs : |yv| ≤ (2 : ℤ) ^ (63 : ℕ) := by rw [abs_le]; omega
    have h1 : |yv * (2 : ℤ) ^ (32 : ℕ)| ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (32 : ℕ) := by
      rw [abs_mul, abs_of_pos (by norm_num : (0 : ℤ) < (2 : ℤ) ^ (32 : ℕ))]
      exact mul_le_mul_of_nonneg_right hyabs (by norm_num)
    have h2 : S.val = yv * (2 : ℤ) ^ (32 : ℕ) -
        (yv * (2 : ℤ) ^ (32 : ℕ) - S.val) := by ring
    rw [h2]
    calc |yv * (2 : ℤ) ^ (32 : ℕ) - (yv * (2 : ℤ) ^ (32 : ℕ) - S.val)|
        ≤ |yv * (2 : ℤ) ^ (32 : ℕ)| + |yv * (2 : ℤ) ^ (32 : ℕ) - S.val| := abs_sub _ _
      _ ≤ (2 : ℤ) ^ (63 : ℕ) * (2 : ℤ) ^ (32 : ℕ) + (2 : ℤ) ^ (31 : ℕ) :=
            add_le_add h1 hZ'
      _ = (2 : ℤ) ^ (95 : ℕ) + (2 : ℤ) ^ (31 : ℕ) := by norm_num
  have hlo31 : I128.min ≤ S.val + 2 ^ (31 : ℕ) := by
    rw [hI128min]
    have h := (abs_le.mp hSbound).1
    linarith
  have hhi31 : S.val + 2 ^ (31 : ℕ) ≤ I128.max := by
    rw [hI128max]
    have h := (abs_le.mp hSbound).2
    linarith
  obtain ⟨rounded, hround_eq, hroundval⟩ := round_shift_32_spec S hlo31 hhi31
  have hroundb : -((2 : ℤ) ^ (63 : ℕ)) ≤ rounded.val ∧
      rounded.val ≤ (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [hroundval, hSval]
    exact ⟨houtlo, houthi⟩
  have hmaxv : (IScalar.cast .I128 core.num.I64.MAX).val = (2 : ℤ) ^ (63 : ℕ) - 1 := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MAX.val = (2 : ℤ) ^ (63 : ℕ) - 1 := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 ((2 : ℤ) ^ (63 : ℕ) - 1)
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hminv : (IScalar.cast .I128 core.num.I64.MIN).val = -(2 : ℤ) ^ (63 : ℕ) := by
    rw [IScalar.cast_val_eq]
    have h : core.num.I64.MIN.val = -(2 : ℤ) ^ (63 : ℕ) := by decide
    rw [h]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I128 (-(2 : ℤ) ^ (63 : ℕ))
      (by norm_num [IScalarTy.numBits]) (by norm_num [IScalarTy.numBits])
  have hmax : ¬ (rounded > IScalar.cast .I128 core.num.I64.MAX) := by
    intro hc
    rw [gt_iff_lt, IScalar.lt_equiv, hmaxv] at hc
    omega
  have hmin : ¬ (rounded < IScalar.cast .I128 core.num.I64.MIN) := by
    intro hc
    rw [IScalar.lt_equiv, hminv] at hc
    omega
  have hcastval : (IScalar.cast .I64 rounded).val = rounded.val := by
    rw [IScalar.cast_val_eq]
    norm_num [IScalarTy.numBits]
    exact bmod_pow_numBits_eq_of_lt IScalarTy.I64 rounded.val
      (by norm_num [IScalarTy.numBits]; omega) (by norm_num [IScalarTy.numBits]; omega)
  refine ⟨IScalar.cast .I64 rounded, ?_, ?_⟩
  · rw [word_row_dot_fixed_unfold w_row x S hlen hloop]
    rw [hround_eq]
    simp only [lift, bind_tc_ok]
    rw [if_neg hmax]
    rw [if_neg hmin]
  · rw [rowDotReal_eq w_row x hlen]
    have hyv : (IScalar.cast .I64 rounded).val =
        divRoundNearest (dotPrefix w_row x w_row.val.length) ((2 : ℤ) ^ (32 : ℕ)) := by
      rw [hcastval, hroundval, hSval]
    exact dot_rounding_bound (dotPrefix w_row x w_row.val.length) (IScalar.cast .I64 rounded) hyv

end

end apeiron_word
