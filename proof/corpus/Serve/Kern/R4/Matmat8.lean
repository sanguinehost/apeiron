import Serve.Kern.R4.Matmat7

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false
set_option linter.unusedTactic false
set_option linter.unreachableTactic false

namespace apeiron_word

open Aeneas Aeneas.Std Result ControlFlow

/-!
# Serving kernel group `kern.matmat` (part 8): the `row_accs` bail law

`src/engine/kern/matmat.rs:504-519` is `row_accs`' inner loop. `Matmat7` refines the `Live`-column
step and the all-`Live` accumulation law; `Matmat6` refines the non-`Live` skip. This file states the
remaining behaviour of the same extracted body:

* §0 the two bail arms not yet stated — the `i128` checked multiply and the `i128` checked add —
  and the column bail predicate that is their disjunction with the `i64` checked multiply arm
  (`matmat7_rowaccs_body_bail_mul`).
* §1 **`word_matmat8_rowaccs_bail_ok`**: if the columns `[c, j0)` are `Live` with in-range checked
  operations and column `j0` (`c ≤ j0 < k ≤ |state|`) satisfies the bail predicate, the loop stops
  with `some RowOutcome.Bail` at column `j0` (cursor `j0+1`), leaving each accumulator slot at its
  initial value plus the contributions of the `Live` columns of `[c, j0)`.
* §2 the total law `word_matmat8_rowaccs_total_ok` over every `ColumnState`: a `Live` column with
  in-range operations accumulates, every other column is skipped, and no column bails.
* §3 non-vacuity: the bail predicate at a concrete overflowing `i64` product.

The bail predicate is the disjunction of the three arms the extracted body actually has, so §1's
hypotheses are exactly the body's bail condition at `j0`. The complementary direction — no `Live`
column with in-range checked operations bails — is `word_matmat8_rowaccs_total_ok`, which shows the
loop returns `early = none` on that whole domain.

Trust boundaries: rustc/LLVM codegen; the `F32`/AVX2 layers; `dequant.f16_to_fixed` and the
`q8_0_block_kdots` drivers of the outer `row_accs_loop0`, which choose `d` and `dots`.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`; `decide` is only used on literal
`isize`/`usize` values, never on an extracted body.
-/

/-! ## §0 · The bail predicate and the two remaining body bail arms -/

/-- The bail condition of the extracted `row_accs_loop0_loop0.body` at column `j`, given the
accumulator slot `t` the body reads: the checked `i64` multiply of the column's scale with its
activation lane overflows, or it holds and the checked `i128` multiply by `d` overflows, or both
hold and the checked `i128` add into the accumulator slot overflows. `j` beyond the slices' domains
is totalised by `matmat7_scale`/`matmat7_dotv`. -/
def matmat8_bails (acts : Slice chart.QuantAct) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (t : I128) (j : Nat) : Prop :=
  (I64.checked_mul (matmat7_scale acts blk j)
      (core.convert.num.FromI64I32.from (matmat7_dotv dots j)) = none) ∨
  (∃ scaled : I64,
      I64.checked_mul (matmat7_scale acts blk j)
        (core.convert.num.FromI64I32.from (matmat7_dotv dots j)) = some scaled ∧
      I128.checked_mul (core.convert.num.FromI128I64.from d)
        (core.convert.num.FromI128I64.from scaled) = none) ∨
  (∃ scaled : I64, ∃ term : I128,
      I64.checked_mul (matmat7_scale acts blk j)
        (core.convert.num.FromI64I32.from (matmat7_dotv dots j)) = some scaled ∧
      I128.checked_mul (core.convert.num.FromI128I64.from d)
        (core.convert.num.FromI128I64.from scaled) = some term ∧
      I128.checked_add t term = none)

/-- `row_accs_loop0_loop0.body`, `early = none`, on a `Live` in-range column whose checked `i64`
multiply succeeds and whose checked `i128` multiply overflows: `totals` is left unchanged, `early`
becomes `some Bail`, and the cursor advances by one (the next body call then stops the loop). -/
theorem matmat8_rowaccs_body_bail_mul128 (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (totals : Array I128 16#usize) (c : Std.Usize)
    (hlt : c.val < k.val) (hlen : c.val < state.val.length)
    (hlive : state.val[c.val] = kern.matmat.ColumnState.Live)
    (hacts : c.val < acts.val.length)
    (hscales : blk.val < (acts.val[c.val]).scales.val.length)
    (hdot : c.val < dots.val.length)
    (scaled : I64)
    (hmul : I64.checked_mul ((acts.val[c.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c.val])) = some scaled)
    (hmul128 : I128.checked_mul (core.convert.num.FromI128I64.from d)
              (core.convert.num.FromI128I64.from scaled) = none) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals none c ⦃ r =>
      ∃ c1 : Std.Usize,
        r = ControlFlow.cont (totals, some kern.matmat.RowOutcome.Bail, c1) ∧
        c1.val = c.val + 1 ⦄ := by
  have hstate : Slice.index_usize state c = ok (state.val[c.val]) :=
    slice_index_usize_ok state c hlen
  have heq : kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq (state.val[c.val])
      kern.matmat.ColumnState.Live = ok true := matmat6_colstate_eq_live _ hlive
  have hscales' : blk.val < (alloc.vec.Vec.deref (acts.val[c.val]).scales).val.length := by
    simpa only [alloc.vec.Vec.deref, Slice.from_val] using hscales
  have hscaleeq : Slice.index_usize (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk
      = ok ((acts.val[c.val]).scales.val[blk.val]) := by
    rw [slice_index_usize_ok (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk hscales']
    simp only [alloc.vec.Vec.deref, Slice.from_val]
  obtain ⟨c1, hc1, _hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  unfold kern.matmat.row_accs_loop0_loop0.body
  rw [if_pos ((UScalar.lt_equiv c k).2 hlt)]
  simp only [core.option.Option.is_none, Option.isNone_none, if_true, hstate, heq, bind_tc_ok,
    matmat7_scales_eq, hscaleeq, slice_index_usize_ok acts c hacts,
    matmat7_array_index_usize_ok dots c hdot, lift, hmul, hmul128, hc1]
  step*

/-- `row_accs_loop0_loop0.body`, `early = none`, on a `Live` in-range column whose checked `i64`
multiply and checked `i128` multiply succeed and whose checked `i128` add overflows: `totals` is left
unchanged, `early` becomes `some Bail`, and the cursor advances by one. -/
theorem matmat8_rowaccs_body_bail_add (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (totals : Array I128 16#usize) (c : Std.Usize)
    (hlt : c.val < k.val) (hlen : c.val < state.val.length)
    (hlive : state.val[c.val] = kern.matmat.ColumnState.Live)
    (hacts : c.val < acts.val.length)
    (hscales : blk.val < (acts.val[c.val]).scales.val.length)
    (hdot : c.val < dots.val.length)
    (htot : c.val < totals.val.length)
    (scaled : I64) (term : I128)
    (hmul : I64.checked_mul ((acts.val[c.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c.val])) = some scaled)
    (hmul128 : I128.checked_mul (core.convert.num.FromI128I64.from d)
              (core.convert.num.FromI128I64.from scaled) = some term)
    (hadd : I128.checked_add (totals.val[c.val]) term = none) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals none c ⦃ r =>
      ∃ c1 : Std.Usize,
        r = ControlFlow.cont (totals, some kern.matmat.RowOutcome.Bail, c1) ∧
        c1.val = c.val + 1 ⦄ := by
  have hstate : Slice.index_usize state c = ok (state.val[c.val]) :=
    slice_index_usize_ok state c hlen
  have heq : kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq (state.val[c.val])
      kern.matmat.ColumnState.Live = ok true := matmat6_colstate_eq_live _ hlive
  have hscales' : blk.val < (alloc.vec.Vec.deref (acts.val[c.val]).scales).val.length := by
    simpa only [alloc.vec.Vec.deref, Slice.from_val] using hscales
  have hscaleeq : Slice.index_usize (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk
      = ok ((acts.val[c.val]).scales.val[blk.val]) := by
    rw [slice_index_usize_ok (alloc.vec.Vec.deref (acts.val[c.val]).scales) blk hscales']
    simp only [alloc.vec.Vec.deref, Slice.from_val]
  obtain ⟨c1, hc1, _hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  unfold kern.matmat.row_accs_loop0_loop0.body
  rw [if_pos ((UScalar.lt_equiv c k).2 hlt)]
  simp only [core.option.Option.is_none, Option.isNone_none, if_true, hstate, heq, bind_tc_ok,
    matmat7_scales_eq, hscaleeq, slice_index_usize_ok acts c hacts,
    matmat7_array_index_usize_ok dots c hdot, lift, hmul, hmul128,
    matmat7_array_index_usize_ok totals c htot, hadd, hc1]
  step*

/-! ## §1 · The `row_accs` bail law -/

/-- The accumulator reference: `T` has `t0`'s length and each slot is `t0`'s value plus the
contributions of the `Live` columns of `[c0, stop)`. -/
def matmat8_acc (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (t0 : Array I128 16#usize)
    (c0 stop : Nat) (T : Array I128 16#usize) : Prop :=
  T.val.length = t0.val.length ∧
  ∀ j, j < t0.val.length →
    (T.val.getD j (0#i128)).val = (t0.val.getD j (0#i128)).val +
      (if c0 ≤ j ∧ j < stop then matmat7_delta acts state blk d dots j else 0)

/-- The bail invariant of `row_accs_loop0_loop0`: either the loop is still accumulating at cursor
`cur ∈ [c0, j0]` with `early = none`, or column `j0` has bailed and the next body call stops the
loop with `(totals, some Bail)`. -/
def matmat8_bailInv (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (t0 : Array I128 16#usize)
    (c0 j0 : Nat) (st : Matmat7RowAccsSt) : Prop :=
  (st.2.1 = none ∧ c0 ≤ st.2.2.val ∧ st.2.2.val ≤ j0 ∧
      matmat8_acc acts state blk d dots t0 c0 st.2.2.val st.1) ∨
  (st.2.1 = some kern.matmat.RowOutcome.Bail ∧ st.2.2.val = j0 + 1 ∧
      matmat8_acc acts state blk d dots t0 c0 j0 st.1)

/-- **`row_accs` bail law.** With `c ≤ j0 < k ≤ |state|`, if every column `j ∈ [c, j0)` is `Live`
with in-range checked operations and column `j0` satisfies the bail predicate against the initial
accumulator slot `totals[j0]`, then `row_accs_loop0_loop0` stops with `some RowOutcome.Bail`: each
slot of the returned accumulator is its initial value plus the contributions of the `Live` columns of
`[c, j0)`, and the columns from `j0` on are not accumulated. The stop is exact — the bail predicate
is the disjunction of the body's three bail arms, and no `Live` column of `[c, j0)` bails. -/
theorem word_matmat8_rowaccs_bail_ok
    (acts : Slice chart.QuantAct) (k : Std.Usize) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize)
    (totals : Array I128 16#usize) (c : Std.Usize) (j0 : Nat)
    (hc : c.val ≤ j0) (hj0 : j0 < k.val) (hk : k.val ≤ state.val.length)
    (hj0state : j0 < state.val.length) (hacts : j0 < acts.val.length)
    (hdots : j0 < dots.val.length) (hlent : j0 < totals.val.length)
    (hblk : ∀ j (hj : j < acts.val.length), c.val ≤ j → j < j0 →
      blk.val < (acts.val[j]).scales.val.length)
    (hlive : ∀ j (hj : j < state.val.length), c.val ≤ j → j < j0 →
      state.val[j] = kern.matmat.ColumnState.Live)
    (hj0blk : blk.val < (acts.val[j0]).scales.val.length)
    (hj0live : state.val[j0] = kern.matmat.ColumnState.Live)
    (hdom : ∀ j, c.val ≤ j → j < j0 →
      I64.min ≤ (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ∧
      (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ≤ I64.max ∧
      I128.min ≤ d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
      d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max ∧
      I128.min ≤ (totals.val.getD j (0#i128)).val +
        d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
      (totals.val.getD j (0#i128)).val +
        d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max)
    (hbail : matmat8_bails acts blk d dots (totals.val[j0]) j0) :
    kern.matmat.row_accs_loop0_loop0 acts k state totals none blk d dots c ⦃ r =>
      ∃ totals' : Array Std.I128 16#usize,
        r = (totals', some kern.matmat.RowOutcome.Bail) ∧
        matmat8_acc acts state blk d dots totals c.val j0 totals' ⦄ := by
  unfold kern.matmat.row_accs_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Matmat7RowAccsSt => k.val - st.2.2.val)
    (inv := matmat8_bailInv acts state blk d dots totals c.val j0)
    (post := fun y : (Array Std.I128 16#usize) × (Option kern.matmat.RowOutcome) =>
      ∃ totals' : Array Std.I128 16#usize,
        y = (totals', some kern.matmat.RowOutcome.Bail) ∧
        matmat8_acc acts state blk d dots totals c.val j0 totals')
  · rintro ⟨totals', early', c'⟩ hinv
    rcases hinv with ⟨he, hge, hle, hacc⟩ | ⟨he, hcur, hacc⟩
    · dsimp only at he hge hle hacc
      subst he
      simp only []
      have hlenacc := hacc.1
      have hpt := hacc.2
      by_cases hlt : c'.val < j0
      · -- the `Live` accumulation step
        have hstate_len : c'.val < state.val.length := lt_trans hlt hj0state
        have hlive' : state.val[c'.val] = kern.matmat.ColumnState.Live :=
          hlive c'.val hstate_len hge hlt
        have hliveD : state.val.getD c'.val kern.matmat.ColumnState.Live =
            kern.matmat.ColumnState.Live := by
          rw [List.getD_eq_getElem state.val kern.matmat.ColumnState.Live hstate_len, hlive']
        have hacts' : c'.val < acts.val.length := lt_trans hlt hacts
        have hdots' : c'.val < dots.val.length := lt_trans hlt hdots
        have hblk' : blk.val < (acts.val[c'.val]).scales.val.length := hblk c'.val hacts' hge hlt
        have hlent' : c'.val < totals.val.length := lt_trans hlt hlent
        have htot'' : c'.val < totals'.val.length := by rw [hlenacc]; exact hlent'
        have hsc' : matmat7_scale acts blk c'.val = (acts.val[c'.val]).scales.val[blk.val] :=
          matmat7_scale_eq acts blk c'.val hacts' hblk'
        have hdv' : matmat7_dotv dots c'.val = dots.val[c'.val] :=
          matmat7_dotv_eq dots c'.val hdots'
        have hval : ((acts.val[c'.val]).scales.val[blk.val]).val * (dots.val[c'.val]).val =
            (matmat7_scale acts blk c'.val).val * (matmat7_dotv dots c'.val).val := by
          rw [hsc', hdv']
        have hpc : (totals'.val.getD c'.val (0#i128)).val =
            (totals.val.getD c'.val (0#i128)).val := by
          have h := hpt c'.val hlent'
          rw [if_neg (by rintro ⟨_, h2⟩; omega), add_zero] at h
          exact h
        obtain ⟨scaled, hmul, hscaledv⟩ := matmat7_i64_checked_mul_some
          ((acts.val[c'.val]).scales.val[blk.val]) (core.convert.num.FromI64I32.from (dots.val[c'.val]))
          (by rw [core.convert.num.FromI64I32.from_val_eq, hval]
              exact (hdom c'.val hge hlt).1)
          (by rw [core.convert.num.FromI64I32.from_val_eq, hval]
              exact (hdom c'.val hge hlt).2.1)
        obtain ⟨term, hterm, htermv⟩ := matmat7_i128_checked_mul_some
          (core.convert.num.FromI128I64.from d) (core.convert.num.FromI128I64.from scaled)
          (by rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact (hdom c'.val hge hlt).2.2.1)
          (by rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact (hdom c'.val hge hlt).2.2.2.1)
        obtain ⟨total, hadd, htotalv⟩ := matmat7_i128_checked_add_some
          (totals'.val[c'.val]'htot'') term
          (by rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, htermv,
                core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact (hdom c'.val hge hlt).2.2.2.2.1)
          (by rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, htermv,
                core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact (hdom c'.val hge hlt).2.2.2.2.2)
        obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
          (matmat7_rowaccs_body_live acts k state blk d dots totals' c' (lt_trans hlt hj0)
            hstate_len hlive' hacts' hblk' hdots' htot'' scaled term total hmul hterm hadd)
        obtain ⟨c1, hr, hc1v, htotv⟩ := hpost
        rw [hbody, hr]
        simp only [WP.spec_ok]
        show matmat8_bailInv acts state blk d dots totals c.val j0
            (totals'.set c' total, none, c1) ∧ k.val - c1.val < k.val - c'.val
        simp only [matmat8_bailInv, matmat8_acc]
        refine ⟨Or.inl ⟨trivial, ?_, ?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · rw [Array.set_val_eq, List.length_set]; exact hlenacc
        · intro j hj
          have hjT : j < totals'.val.length := by rw [hlenacc]; exact hj
          have hset := matmat7_list_getD_set totals'.val c'.val j total hjT
          rw [Array.set_val_eq]
          by_cases hjc : j = c'.val
          · rw [if_pos hjc] at hset
            rw [hset, if_pos (by rw [hjc]; exact ⟨hge, by omega⟩), hjc]
            rw [htotv, ← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc,
              matmat7_delta_live acts state blk d dots c'.val hliveD,
              matmat7_term_eq acts blk d dots c'.val hacts' hblk' hdots']
          · rw [if_neg hjc] at hset
            rw [hset, hpt j hj]
            congr 1
            by_cases hcond : c.val ≤ j ∧ j < c'.val
            · rw [if_pos hcond, if_pos ⟨hcond.1, by omega⟩]
            · rw [if_neg hcond,
                if_neg (fun hc'' => hcond ⟨hc''.1, by omega⟩)]
        · omega
      · -- the cursor has reached the bail column
        have hcj : c'.val = j0 := le_antisymm hle (by omega)
        rw [hcj] at hacc
        have hklt : c'.val < k.val := by omega
        have hstate_len : c'.val < state.val.length := by omega
        have hlive' : state.val[c'.val] = kern.matmat.ColumnState.Live :=
          by simpa only [hcj] using hj0live
        have hacts' : c'.val < acts.val.length := by omega
        have hdots' : c'.val < dots.val.length := by omega
        have hblk' : blk.val < (acts.val[c'.val]).scales.val.length :=
          by simpa only [hcj] using hj0blk
        have hlent' : c'.val < totals.val.length := by omega
        have htot'' : c'.val < totals'.val.length := by rw [hlenacc]; exact hlent'
        have hsc' : matmat7_scale acts blk c'.val = (acts.val[c'.val]).scales.val[blk.val] :=
          matmat7_scale_eq acts blk c'.val hacts' hblk'
        have hdv' : matmat7_dotv dots c'.val = dots.val[c'.val] :=
          matmat7_dotv_eq dots c'.val hdots'
        have hscj : matmat7_scale acts blk j0 = (acts.val[c'.val]).scales.val[blk.val] := by
          rw [← hcj]; exact hsc'
        have hdvj : matmat7_dotv dots j0 = dots.val[c'.val] := by
          rw [← hcj]; exact hdv'
        have hpc : (totals'.val.getD c'.val (0#i128)).val =
            (totals.val.getD c'.val (0#i128)).val := by
          have h := hpt c'.val hlent'
          rw [if_neg (by rintro ⟨_, h2⟩; omega), add_zero] at h
          exact h
        rcases hbail with hb | ⟨scaled, hm1, hm2⟩ | ⟨scaled, term, hm1, hm2, ha1⟩
        · have hm' : I64.checked_mul ((acts.val[c'.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c'.val])) = none := by
            rw [← hscj, ← hdvj]; exact hb
          obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
            (matmat7_rowaccs_body_bail_mul acts k state blk d dots totals' c' hklt hstate_len
              hlive' hacts' hblk' hdots' hm')
          obtain ⟨c1, hr, hc1v⟩ := hpost
          rw [hbody, hr]
          simp only [WP.spec_ok]
          show matmat8_bailInv acts state blk d dots totals c.val j0
              (totals', some kern.matmat.RowOutcome.Bail, c1) ∧ k.val - c1.val < k.val - c'.val
          simp only [matmat8_bailInv, matmat8_acc]
          refine ⟨Or.inr ⟨trivial, ?_, hacc⟩, by omega⟩
          rw [hc1v, hcj]
        · have hm' : I64.checked_mul ((acts.val[c'.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c'.val])) = some scaled := by
            rw [← hscj, ← hdvj]; exact hm1
          obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
            (matmat8_rowaccs_body_bail_mul128 acts k state blk d dots totals' c' hklt hstate_len
              hlive' hacts' hblk' hdots' scaled hm' hm2)
          obtain ⟨c1, hr, hc1v⟩ := hpost
          rw [hbody, hr]
          simp only [WP.spec_ok]
          show matmat8_bailInv acts state blk d dots totals c.val j0
              (totals', some kern.matmat.RowOutcome.Bail, c1) ∧ k.val - c1.val < k.val - c'.val
          simp only [matmat8_bailInv, matmat8_acc]
          refine ⟨Or.inr ⟨trivial, ?_, hacc⟩, by omega⟩
          rw [hc1v, hcj]
        · have hm' : I64.checked_mul ((acts.val[c'.val]).scales.val[blk.val])
              (core.convert.num.FromI64I32.from (dots.val[c'.val])) = some scaled := by
            rw [← hscj, ← hdvj]; exact hm1
          have ha' : I128.checked_add (totals'.val[c'.val]) term = none := by
            rw [show totals'.val[c'.val]'htot'' = totals.val[j0] from
              IScalar.eq_of_val_eq (by
                rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, hcj,
                  List.getD_eq_getElem totals.val (0#i128) hlent])]
            exact ha1
          obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
            (matmat8_rowaccs_body_bail_add acts k state blk d dots totals' c' hklt hstate_len
              hlive' hacts' hblk' hdots' htot'' scaled term hm' hm2 ha')
          obtain ⟨c1, hr, hc1v⟩ := hpost
          rw [hbody, hr]
          simp only [WP.spec_ok]
          show matmat8_bailInv acts state blk d dots totals c.val j0
              (totals', some kern.matmat.RowOutcome.Bail, c1) ∧ k.val - c1.val < k.val - c'.val
          simp only [matmat8_bailInv, matmat8_acc]
          refine ⟨Or.inr ⟨trivial, ?_, hacc⟩, by omega⟩
          rw [hc1v, hcj]
    · -- a previous column already bailed: the body stops the loop
      dsimp only at he hcur hacc
      simp only []
      rw [he]
      have hbody := matmat6_rowaccs_body_early acts k state blk d dots totals'
        kern.matmat.RowOutcome.Bail c'
      rw [hbody]
      simp only [WP.spec_ok]
      exact ⟨totals', rfl, hacc⟩
  · refine Or.inl ⟨rfl, le_rfl, hc, rfl, ?_⟩
    intro j hj
    dsimp only
    rw [if_neg (by rintro ⟨_, h2⟩; omega), add_zero]

/-! ## §2 · The total law over every `ColumnState` -/

/-- A non-`Live` column contributes nothing. -/
theorem matmat8_delta_ne_live (acts : Slice chart.QuantAct) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize) (j : Nat)
    (h : state.val.getD j kern.matmat.ColumnState.Live ≠ kern.matmat.ColumnState.Live) :
    matmat7_delta acts state blk d dots j = 0 := by
  unfold matmat7_delta
  split <;> simp_all

/-- The in-range condition of a `Live` column `j` read against accumulator slot `t`: the checked
`i64` multiply `scale·dot`, the checked `i128` multiply by `d`, and the checked `i128` add into `t`
all stay in range. -/
def matmat8_colDom (acts : Slice chart.QuantAct) (blk : Std.Usize) (d : I64)
    (dots : Array I32 16#usize) (t : I128) (j : Nat) : Prop :=
  I64.min ≤ (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ∧
  (matmat7_scale acts blk j).val * (matmat7_dotv dots j).val ≤ I64.max ∧
  I128.min ≤ d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
  d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max ∧
  I128.min ≤ t.val + d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ∧
  t.val + d.val * ((matmat7_scale acts blk j).val * (matmat7_dotv dots j).val) ≤ I128.max

/-- **`row_accs` total law, every `ColumnState`.** With `c ≤ k ≤ |state|`, if each `Live` column
`j ∈ [c, k)` has an in-range scale index and in-range checked operations, then the loop never bails:
it returns `(totals', none)` with each slot of `totals'` equal to its initial value plus `d·(scale·dot)`
over exactly the `Live` columns of `[c, k)`. A non-`Live` column (`Overflow`, `Fallback`) is skipped
and contributes `0` (`matmat7_delta`), and requires nothing of `acts`, `dots` or its checked
operations — the `Live`-only hypothesis is the whole of the domain (the index witness
`j < acts.val.length` in `hlive` is implied by `j < k.val ≤ acts.val.length`). -/
theorem word_matmat8_rowaccs_total_ok
    (acts : Slice chart.QuantAct) (k : Std.Usize) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : I64) (dots : Array I32 16#usize)
    (totals : Array I128 16#usize) (c : Std.Usize)
    (hc : c.val ≤ k.val) (hk : k.val ≤ state.val.length)
    (hacts : k.val ≤ acts.val.length) (hdots : k.val ≤ dots.val.length)
    (hlent : k.val ≤ totals.val.length)
    (hlive : ∀ j (hj : j < state.val.length) (hjA : j < acts.val.length),
      c.val ≤ j → j < k.val →
      state.val[j] = kern.matmat.ColumnState.Live →
      blk.val < (acts.val[j]).scales.val.length ∧
      matmat8_colDom acts blk d dots (totals.val.getD j (0#i128)) j) :
    kern.matmat.row_accs_loop0_loop0 acts k state totals none blk d dots c ⦃ r =>
      ∃ totals' : Array Std.I128 16#usize, r = (totals', none) ∧
        (∀ j, j < totals.val.length →
          (totals'.val.getD j (0#i128)).val = (totals.val.getD j (0#i128)).val +
            (if c.val ≤ j ∧ j < k.val then matmat7_delta acts state blk d dots j else 0)) ⦄ := by
  unfold kern.matmat.row_accs_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Matmat7RowAccsSt => k.val - st.2.2.val)
    (inv := matmat7_loopInv acts state blk d dots totals c.val k.val)
    (post := fun y : (Array Std.I128 16#usize) × (Option kern.matmat.RowOutcome) =>
      ∃ totals' : Array Std.I128 16#usize, y = (totals', none) ∧
        (∀ j, j < totals.val.length →
          (totals'.val.getD j (0#i128)).val = (totals.val.getD j (0#i128)).val +
            (if c.val ≤ j ∧ j < k.val then matmat7_delta acts state blk d dots j else 0)))
  · rintro ⟨totals', early', c'⟩ hinv
    obtain ⟨he, hge, hle, hlen, hpoint⟩ := hinv
    dsimp only at he hge hle hlen hpoint
    subst he
    simp only []
    by_cases hlt : c'.val < k.val
    · have hstate_len : c'.val < state.val.length := lt_of_lt_of_le hlt hk
      have hlent' : c'.val < totals.val.length := lt_of_lt_of_le hlt hlent
      have htot'' : c'.val < totals'.val.length := by rw [hlen]; exact hlent'
      have hpc : (totals'.val.getD c'.val (0#i128)).val =
          (totals.val.getD c'.val (0#i128)).val := by
        have h := hpoint c'.val hlent'
        rw [if_neg (by rintro ⟨_, h2⟩; omega), add_zero] at h
        exact h
      by_cases hl : state.val[c'.val] = kern.matmat.ColumnState.Live
      · -- a `Live` column: accumulate
        have hacts' : c'.val < acts.val.length := lt_of_lt_of_le hlt hacts
        have hdots' : c'.val < dots.val.length := lt_of_lt_of_le hlt hdots
        obtain ⟨hblk', hcd⟩ := hlive c'.val hstate_len hacts' hge hlt hl
        have hliveD : state.val.getD c'.val kern.matmat.ColumnState.Live =
            kern.matmat.ColumnState.Live := by
          rw [List.getD_eq_getElem state.val kern.matmat.ColumnState.Live hstate_len, hl]
        have hsc' : matmat7_scale acts blk c'.val = (acts.val[c'.val]).scales.val[blk.val] :=
          matmat7_scale_eq acts blk c'.val hacts' hblk'
        have hdv' : matmat7_dotv dots c'.val = dots.val[c'.val] :=
          matmat7_dotv_eq dots c'.val hdots'
        have hval : ((acts.val[c'.val]).scales.val[blk.val]).val * (dots.val[c'.val]).val =
            (matmat7_scale acts blk c'.val).val * (matmat7_dotv dots c'.val).val := by
          rw [hsc', hdv']
        obtain ⟨scaled, hmul, hscaledv⟩ := matmat7_i64_checked_mul_some
          ((acts.val[c'.val]).scales.val[blk.val]) (core.convert.num.FromI64I32.from (dots.val[c'.val]))
          (by rw [core.convert.num.FromI64I32.from_val_eq, hval]
              exact hcd.1)
          (by rw [core.convert.num.FromI64I32.from_val_eq, hval]
              exact hcd.2.1)
        obtain ⟨term, hterm, htermv⟩ := matmat7_i128_checked_mul_some
          (core.convert.num.FromI128I64.from d) (core.convert.num.FromI128I64.from scaled)
          (by rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact hcd.2.2.1)
          (by rw [core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact hcd.2.2.2.1)
        obtain ⟨total, hadd, htotalv⟩ := matmat7_i128_checked_add_some
          (totals'.val[c'.val]'htot'') term
          (by rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, htermv,
                core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact hcd.2.2.2.2.1)
          (by rw [← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc, htermv,
                core.convert.num.FromI128I64.from_val_eq, core.convert.num.FromI128I64.from_val_eq,
                hscaledv, core.convert.num.FromI64I32.from_val_eq, hval]
              exact hcd.2.2.2.2.2)
        obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
          (matmat7_rowaccs_body_live acts k state blk d dots totals' c' hlt hstate_len hl
            hacts' hblk' hdots' htot'' scaled term total hmul hterm hadd)
        obtain ⟨c1, hr, hc1v, htotv⟩ := hpost
        rw [hbody, hr]
        simp only [WP.spec_ok]
        show matmat7_loopInv acts state blk d dots totals c.val k.val
            (totals'.set c' total, none, c1) ∧ k.val - c1.val < k.val - c'.val
        simp only [matmat7_loopInv]
        refine ⟨⟨trivial, by omega, by omega, ?_, ?_⟩, by omega⟩
        · rw [Array.set_val_eq, List.length_set]; exact hlen
        · intro j hj
          have hjT : j < totals'.val.length := by rw [hlen]; exact hj
          have hset := matmat7_list_getD_set totals'.val c'.val j total hjT
          rw [Array.set_val_eq]
          by_cases hjc : j = c'.val
          · rw [if_pos hjc] at hset
            rw [hset, if_pos (by rw [hjc]; exact ⟨hge, by omega⟩), hjc]
            rw [htotv, ← List.getD_eq_getElem totals'.val (0#i128) htot'', hpc,
              matmat7_delta_live acts state blk d dots c'.val hliveD,
              matmat7_term_eq acts blk d dots c'.val hacts' hblk' hdots']
          · rw [if_neg hjc] at hset
            rw [hset, hpoint j hj]
            congr 1
            by_cases hcond : c.val ≤ j ∧ j < c'.val
            · rw [if_pos hcond, if_pos ⟨hcond.1, by omega⟩]
            · rw [if_neg hcond,
                if_neg (fun hc'' => hcond ⟨hc''.1, by omega⟩)]
      · -- a non-`Live` column: skip
        have hneq : state.val[c'.val] ≠ kern.matmat.ColumnState.Live := hl
        have hneqD : state.val.getD c'.val kern.matmat.ColumnState.Live ≠
            kern.matmat.ColumnState.Live := by
          rw [List.getD_eq_getElem state.val kern.matmat.ColumnState.Live hstate_len]
          exact hneq
        obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
          (matmat6_rowaccs_body_skip acts k state blk d dots totals' c' hlt hstate_len hneq)
        obtain ⟨c1, hr, hc1v⟩ := hpost
        rw [hbody, hr]
        simp only [WP.spec_ok]
        show matmat7_loopInv acts state blk d dots totals c.val k.val
            (totals', none, c1) ∧ k.val - c1.val < k.val - c'.val
        simp only [matmat7_loopInv]
        refine ⟨⟨trivial, by omega, by omega, hlen, ?_⟩, by omega⟩
        intro j hj
        rw [hpoint j hj]
        have hg : (if c.val ≤ j ∧ j < c'.val then matmat7_delta acts state blk d dots j else 0)
            = (if c.val ≤ j ∧ j < c1.val then matmat7_delta acts state blk d dots j else 0) := by
          by_cases hjc : j = c'.val
          · rw [hjc, if_neg (by omega), if_pos (by omega),
              matmat8_delta_ne_live acts state blk d dots c'.val hneqD]
          · by_cases hcond : c.val ≤ j ∧ j < c'.val
            · rw [if_pos hcond, if_pos ⟨hcond.1, by omega⟩]
            · rw [if_neg hcond,
                if_neg (fun hc'' => hcond ⟨hc''.1, by omega⟩)]
        rw [hg]
    · have hck : c'.val = k.val := le_antisymm hle (by omega)
      unfold kern.matmat.row_accs_loop0_loop0.body
      rw [if_neg (fun hc' => hlt ((UScalar.lt_equiv c' k).1 hc'))]
      simp only [WP.spec_ok]
      refine ⟨totals', rfl, ?_⟩
      intro j hj
      rw [hpoint j hj, hck]
  · refine ⟨rfl, le_rfl, hc, rfl, ?_⟩
    intro j hj
    dsimp only
    rw [if_neg (by rintro ⟨_, h2⟩; omega), add_zero]

/-! ## §3 · Non-vacuity of the bail predicate -/

/-- A concrete activation block whose scale `0` is `2^63-1`. -/
def matmat8_qa : chart.QuantAct :=
  { scales := alloc.vec.Vec.from ([9223372036854775807#i64] : List I64) (by scalar_tac)
    codes := alloc.vec.Vec.from ([] : List I8) (by scalar_tac) }

/-- The one-entry activation slice holding `matmat8_qa`. -/
def matmat8_acts : Slice chart.QuantAct := Slice.from [matmat8_qa] (by scalar_tac)

/-- A 16-lane `dots` vector with lane `0` equal to `2`. -/
def matmat8_dots : Array I32 16#usize :=
  Array.from
    ([2#i32, 0#i32, 0#i32, 0#i32,
      0#i32, 0#i32, 0#i32, 0#i32,
      0#i32, 0#i32, 0#i32, 0#i32,
      0#i32, 0#i32, 0#i32, 0#i32] : List I32)
    (by decide)

/-- `matmat8_bails` is inhabited: at column `0` of the witness slice, with `blk = 0` and the witness
`dots`, the checked `i64` multiply is `(2^63-1)·2 = 2^64-2 > I64.max`, so the predicate's first arm
fires. -/
example : matmat8_bails matmat8_acts 0#usize 1#i64 matmat8_dots 0#i128 0 := by
  refine Or.inl ?_
  have hsc : matmat7_scale matmat8_acts 0#usize 0 = 9223372036854775807#i64 := by
    simp [matmat7_scale, matmat8_acts, matmat8_qa]
  have hdv : matmat7_dotv matmat8_dots 0 = 2#i32 := by
    simp [matmat7_dotv, matmat8_dots]
  rw [hsc, hdv]
  refine matmat7_i64_checked_mul_none ?_
  rw [core.convert.num.FromI64I32.from_val_eq, I64.min_eq, I64.max_eq]
  first
    | decide
    | norm_num

end apeiron_word
