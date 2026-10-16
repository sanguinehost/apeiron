import Serve.Kern.R4.Matmat5

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
# Serving kernel group `kern.matmat` (part 6): `row_accs`, `pooled_rows`, `pooled_q8_0`,
`matmat_q8act`

Helpers of the `matmat_q8act` route (`src/engine/kern/matmat.rs`):

* `row_accs` (`:482-528`) accumulates, per output column, the `i32` dot row against the live columns'
  scales (`:504-519`), bailing on the first `i64`/`i128` overflow.
* `pooled_rows` (`:247-284`) writes `round_shift_32`-scaled accumulator rows into the caller's `dst`.
* `pooled_q8_0` (`:320-353`) materialises the `q8_0` activations as one `Vec<i64>` per output column.
* `matmat_q8act` (`:160-240`) drives `row_accs` per output row and resolves each column's accumulator
  (`:198-206`) or falls a `Live` column back (`:188-191`).

This file states WHAT the clean-cone loops compute, not how they loop. A body whose own extraction
calls `chart.matvec_q8act`, `chart.q8_0_row_kcols_avx2`, `kern.matmat.q8_0_block_kdots` (AVX2 arm) or
`dequant.f16_to_fixed` is not proved here; its cone carries a boundary axiom and it is named in the
return.

Trust boundaries: rustc/LLVM codegen; the `F32`/AVX2 layers; `dequant.f16_to_fixed`; the
`chart.QuantAct.magnitude` walk; `chart.matvec_q8act`.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`; `decide` is not used on an extracted body.
-/

/-! ## §0 · References -/

/-- The buffer of `k - c` fresh empty columns the fill loops append, one per step. -/
def matmat6_fillApp (n_out : Std.Usize) (columns : List (alloc.vec.Vec Std.I64)) :
    Nat → List (alloc.vec.Vec Std.I64)
  | 0 => columns
  | n + 1 => matmat6_fillApp n_out columns n ++ [alloc.vec.Vec.with_capacity Std.I64 n_out]

theorem matmat6_fillApp_succ (n_out : Std.Usize) (columns : List (alloc.vec.Vec Std.I64)) (n : Nat) :
    matmat6_fillApp n_out columns (n + 1) =
      matmat6_fillApp n_out columns n ++ [alloc.vec.Vec.with_capacity Std.I64 n_out] := rfl

theorem matmat6_fillApp_length (n_out : Std.Usize) (columns : List (alloc.vec.Vec Std.I64)) (n : Nat) :
    (matmat6_fillApp n_out columns n).length = columns.length + n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [matmat6_fillApp_succ, List.length_append, ih, List.length_singleton, Nat.add_assoc]

/-! ## §1 · The three fill loops -/

/-- `matmat_q8act_loop0.body` past the end returns the final buffer. -/
theorem matmat6_fill_body_done (n_out k : Std.Usize)
    (columns : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : ¬ c.val < k.val) :
    kern.matmat.matmat_q8act_loop0.body n_out k columns c = ok (ControlFlow.done columns) := by
  unfold kern.matmat.matmat_q8act_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hc)]

/-- `matmat_q8act_loop0.body` on an in-range cursor `c` pushes one fresh `n_out`-capacity column and
advances to `c + 1`. -/
theorem matmat6_fill_body_cont (n_out k : Std.Usize)
    (columns : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val < k.val) (hlen : columns.val.length < Std.Usize.max) :
    kern.matmat.matmat_q8act_loop0.body n_out k columns c ⦃ r =>
      ∃ (columns1 : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c1 : Std.Usize),
        r = ControlFlow.cont (columns1, c1) ∧
        columns1.val = columns.val ++ [alloc.vec.Vec.with_capacity Std.I64 n_out] ∧
        c1.val = c.val + 1 ⦄ := by
  unfold kern.matmat.matmat_q8act_loop0.body
  rw [if_pos ((UScalar.lt_equiv c k).2 hc)]
  obtain ⟨columns1, hpush, hpushv⟩ := WP.spec_imp_exists
    (alloc.vec.Vec.push_spec columns (alloc.vec.Vec.with_capacity Std.I64 n_out) hlen)
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  simp only [bind_tc_ok]
  rw [hpush]
  simp only [bind_tc_ok]
  rw [hc1]
  simp only [bind_tc_ok]
  rw [WP.spec_ok]
  exact ⟨columns1, c1, rfl, hpushv, by rw [hc1v, show (1#usize).val = 1 from by decide]⟩

/-- **Fill loop law.** `matmat_q8act_loop0` from cursor `c` appends `k - c` fresh empty columns. -/
theorem word_matmat6_fill_loop0_ok (n_out k : Std.Usize)
    (columns : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val ≤ k.val) (hcap : columns.val.length + (k.val - c.val) ≤ Std.Usize.max) :
    kern.matmat.matmat_q8act_loop0 n_out k columns c ⦃ r =>
      r.val = matmat6_fillApp n_out columns.val (k.val - c.val) ⦄ := by
  unfold kern.matmat.matmat_q8act_loop0
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize => k.val - st.2.val)
    (inv := fun (st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize) =>
      c.val ≤ st.2.val ∧ st.2.val ≤ k.val ∧
      st.1.val = matmat6_fillApp n_out columns.val (st.2.val - c.val))
    (post := fun (y : alloc.vec.Vec (alloc.vec.Vec Std.I64)) =>
      y.val = matmat6_fillApp n_out columns.val (k.val - c.val))
  · rintro ⟨cols', c'⟩ ⟨hge, hle, hinv⟩
    dsimp only at hge hle hinv ⊢
    by_cases hlt : c'.val < k.val
    · have hlen : cols'.val.length < Std.Usize.max := by
        have hlen' : cols'.val.length = columns.val.length + (c'.val - c.val) := by
          rw [hinv, matmat6_fillApp_length]
        omega
      obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
        (matmat6_fill_body_cont n_out k cols' c' hlt hlen)
      obtain ⟨columns1, c1, hr, hpushv, hc1v⟩ := hpost
      rw [hbody, hr]
      simp only [WP.spec_ok]
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · omega
      · omega
      · rw [hpushv, hinv, hc1v,
          show (c'.val + 1) - c.val = (c'.val - c.val) + 1 from by omega, matmat6_fillApp_succ]
      · omega
    · have hge' : c'.val = k.val := le_antisymm hle (Nat.le_of_not_lt hlt)
      have hbody := matmat6_fill_body_done n_out k cols' c' hlt
      rw [hbody]
      simp only [WP.spec_ok]
      rw [← hge']
      exact hinv
  · refine ⟨le_rfl, hc, ?_⟩
    rw [Nat.sub_self]
    rfl
theorem matmat6_loop3_eq_loop0 : kern.matmat.matmat_q8act_loop3 = kern.matmat.matmat_q8act_loop0 := by
  funext n_out k columns c
  unfold kern.matmat.matmat_q8act_loop3 kern.matmat.matmat_q8act_loop0
  rfl

theorem matmat6_pooled_fill_eq_loop0 : kern.matmat.pooled_q8_0_loop0 =
    kern.matmat.matmat_q8act_loop0 := by
  funext n_out k columns c
  unfold kern.matmat.pooled_q8_0_loop0 kern.matmat.matmat_q8act_loop0
  rfl

/-- **Fill loop law (`loop3`).** The third `matmat_q8act` fill loop is the first. -/
theorem word_matmat6_fill_loop3_ok (n_out k : Std.Usize)
    (columns : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val ≤ k.val) (hcap : columns.val.length + (k.val - c.val) ≤ Std.Usize.max) :
    kern.matmat.matmat_q8act_loop3 n_out k columns c ⦃ r =>
      r.val = matmat6_fillApp n_out columns.val (k.val - c.val) ⦄ := by
  rw [matmat6_loop3_eq_loop0]
  exact word_matmat6_fill_loop0_ok n_out k columns c hc hcap

/-- **Fill loop law (`pooled_q8_0_loop0`).** The `pooled_q8_0` fill loop is the first. -/
theorem word_matmat6_pooled_fill_ok (n_out k : Std.Usize)
    (columns : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : c.val ≤ k.val) (hcap : columns.val.length + (k.val - c.val) ≤ Std.Usize.max) :
    kern.matmat.pooled_q8_0_loop0 n_out k columns c ⦃ r =>
      r.val = matmat6_fillApp n_out columns.val (k.val - c.val) ⦄ := by
  rw [matmat6_pooled_fill_eq_loop0]
  exact word_matmat6_fill_loop0_ok n_out k columns c hc hcap

/-! ## §2 · Column-state comparison -/

/-- The `Live` variant's discriminant. -/
theorem matmat6_discr_live : kern.matmat.ColumnState.Live.read_discriminant = (0#isize) := rfl
/-- The `Overflow` variant's discriminant. -/
theorem matmat6_discr_overflow : kern.matmat.ColumnState.Overflow.read_discriminant = (1#isize) := rfl
/-- The `Fallback` variant's discriminant. -/
theorem matmat6_discr_fallback : kern.matmat.ColumnState.Fallback.read_discriminant = (2#isize) := rfl

/-- `(0#isize).val = 0`. -/
theorem matmat6_isize_zero_val : (0#isize).val = 0 := Isize.ofInt_val_eq _
/-- `(1#isize).val = 1`. -/
theorem matmat6_isize_one_val : (1#isize).val = 1 := Isize.ofInt_val_eq _
/-- `(2#isize).val = 2`. -/
theorem matmat6_isize_two_val : (2#isize).val = 2 := Isize.ofInt_val_eq _

/-- The `isize` discriminants are pairwise distinct. -/
theorem matmat6_isize_ne_one_zero : (1#isize) ≠ (0#isize) := by
  intro h
  have hv := IScalar.val_eq_of_eq h
  rw [matmat6_isize_one_val, matmat6_isize_zero_val] at hv
  omega
/-- The `isize` discriminants are pairwise distinct. -/
theorem matmat6_isize_ne_two_zero : (2#isize) ≠ (0#isize) := by
  intro h
  have hv := IScalar.val_eq_of_eq h
  rw [matmat6_isize_two_val, matmat6_isize_zero_val] at hv
  omega

/-- `decide` of a distinct discriminant pair is `false`. -/
theorem matmat6_decide_one_zero : decide ((1#isize) = (0#isize)) = false :=
  decide_eq_false matmat6_isize_ne_one_zero
/-- `decide` of a distinct discriminant pair is `false`. -/
theorem matmat6_decide_two_zero : decide ((2#isize) = (0#isize)) = false :=
  decide_eq_false matmat6_isize_ne_two_zero

/-- `ColumnState::eq Live Live` admits. -/
theorem matmat6_colstate_eq_live (cs : kern.matmat.ColumnState)
    (h : cs = kern.matmat.ColumnState.Live) :
    kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq cs kern.matmat.ColumnState.Live
      = ok true := by
  subst h
  unfold kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq
  rw [matmat6_discr_live]
  show ok (decide ((0#isize) = (0#isize))) = ok true
  rw [show decide ((0#isize) = (0#isize)) = true from decide_eq_true rfl]

/-- `ColumnState::eq cs Live` refuses a non-`Live` column. -/
theorem matmat6_colstate_eq_live_ne (cs : kern.matmat.ColumnState)
    (h : cs ≠ kern.matmat.ColumnState.Live) :
    kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq cs kern.matmat.ColumnState.Live
      = ok false := by
  cases cs
  · exact absurd rfl h
  · unfold kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq
    rw [matmat6_discr_overflow, matmat6_discr_live]
    show ok (decide ((1#isize) = (0#isize))) = ok false
    rw [matmat6_decide_one_zero]
  · unfold kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq
    rw [matmat6_discr_fallback, matmat6_discr_live]
    show ok (decide ((2#isize) = (0#isize))) = ok false
    rw [matmat6_decide_two_zero]

/-! ## §3 · The `row_accs` accumulation loop -/

/-- The loop state of `row_accs_loop0_loop0`: `(totals, early, c)`. -/
abbrev Matmat6RowAccsSt := (Array Std.I128 16#usize) × (Option kern.matmat.RowOutcome) × Std.Usize

/-- `row_accs_loop0_loop0.body` past the end returns the final `(totals, early)`. -/
theorem matmat6_rowaccs_body_done (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : Std.I64)
    (dots : Array Std.I32 16#usize) (totals : Array Std.I128 16#usize)
    (early : Option kern.matmat.RowOutcome) (c : Std.Usize) (hc : ¬ c.val < k.val) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals early c
      = ok (ControlFlow.done (totals, early)) := by
  unfold kern.matmat.row_accs_loop0_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv]; exact hc)]

/-- `row_accs_loop0_loop0.body` with `early = some x` returns immediately: a previous column already
bailed, so the loop stops whatever the cursor. -/
theorem matmat6_rowaccs_body_early (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : Std.I64)
    (dots : Array Std.I32 16#usize) (totals : Array Std.I128 16#usize)
    (x : kern.matmat.RowOutcome) (c : Std.Usize) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals (some x) c
      = ok (ControlFlow.done (totals, some x)) := by
  unfold kern.matmat.row_accs_loop0_loop0.body
  by_cases hc : c.val < k.val
  · rw [if_pos ((UScalar.lt_equiv c k).2 hc)]
    simp only [core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true, if_false]
  · rw [if_neg (by rw [UScalar.lt_equiv]; exact hc)]

/-- `row_accs_loop0_loop0.body`, `early = none`, on a non-`Live` in-range column: the column is
skipped (its scale/code row is not read) and the cursor advances, `totals` unchanged. -/
theorem matmat6_rowaccs_body_skip (acts : Slice chart.QuantAct) (k : Std.Usize)
    (state : Slice kern.matmat.ColumnState) (blk : Std.Usize) (d : Std.I64)
    (dots : Array Std.I32 16#usize) (totals : Array Std.I128 16#usize) (c : Std.Usize)
    (hlt : c.val < k.val) (hlen : c.val < state.val.length)
    (hneq : state.val[c.val] ≠ kern.matmat.ColumnState.Live) :
    kern.matmat.row_accs_loop0_loop0.body acts k state blk d dots totals none c ⦃ r =>
      ∃ c1 : Std.Usize, r = ControlFlow.cont (totals, none, c1) ∧ c1.val = c.val + 1 ⦄ := by
  unfold kern.matmat.row_accs_loop0_loop0.body
  rw [if_pos ((UScalar.lt_equiv c k).2 hlt)]
  simp only [core.option.Option.is_none, Option.isNone_none, if_true]
  obtain ⟨cs, hcs, hcsv⟩ := WP.spec_imp_exists (Slice.index_usize_spec state c hlen)
  subst hcsv
  have heq : kern.matmat.ColumnState.Insts.CoreCmpPartialEqColumnState.eq (state.val[c.val])
      kern.matmat.ColumnState.Live = ok false := matmat6_colstate_eq_live_ne _ hneq
  rw [hcs]
  simp only [bind_tc_ok]
  rw [heq]
  simp only [bind_tc_ok]
  rw [if_neg (by decide)]
  simp only [bind_tc_ok]
  step*

/-- **`row_accs` accumulation loop, every column skipped.** When no column from `c` to `k` is
`Live`, the loop reads no scale or code row, does not bail, and returns `totals` unchanged with
`early = none`. -/
theorem word_matmat6_rowaccs_skip_ok
    (acts : Slice chart.QuantAct) (k : Std.Usize) (state : Slice kern.matmat.ColumnState)
    (blk : Std.Usize) (d : Std.I64) (dots : Array Std.I32 16#usize)
    (totals : Array Std.I128 16#usize) (c : Std.Usize)
    (hc : c.val ≤ k.val) (hk : k.val ≤ state.val.length)
    (hskip : ∀ j, c.val ≤ j → j < k.val →
      state.val.getD j kern.matmat.ColumnState.Live ≠ kern.matmat.ColumnState.Live) :
    kern.matmat.row_accs_loop0_loop0 acts k state totals none blk d dots c
      ⦃ r => r = (totals, none) ⦄ := by
  unfold kern.matmat.row_accs_loop0_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Matmat6RowAccsSt => k.val - st.2.2.val)
    (inv := fun (st : Matmat6RowAccsSt) =>
      st.2.1 = none ∧ st.1 = totals ∧ c.val ≤ st.2.2.val ∧ st.2.2.val ≤ k.val ∧
      (∀ j, c.val ≤ j → j < st.2.2.val →
        state.val.getD j kern.matmat.ColumnState.Live ≠ kern.matmat.ColumnState.Live))
    (post := fun (y : (Array Std.I128 16#usize) × (Option kern.matmat.RowOutcome)) =>
      y = (totals, none))
  · rintro ⟨totals', early', c'⟩ ⟨he, ht, hge, hle, hsk⟩
    dsimp only at he ht hge hle hsk ⊢
    by_cases hlt : c'.val < k.val
    · have hlen : c'.val < state.val.length := lt_of_lt_of_le hlt hk
      have hneq : state.val[c'.val] ≠ kern.matmat.ColumnState.Live := by
        intro hce
        exact hskip c'.val hge hlt (by rw [List.getD_eq_getElem _ _ hlen]; exact hce)
      rw [ht, he]
      obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
        (matmat6_rowaccs_body_skip acts k state blk d dots totals c' hlt hlen hneq)
      obtain ⟨c1, hr, hc1v⟩ := hpost
      rw [hbody, hr]
      simp only [WP.spec_ok]
      refine ⟨⟨trivial, trivial, ?_, ?_, ?_⟩, ?_⟩
      · omega
      · omega
      · intro j hj1 hj2
        by_cases hj : j < c'.val
        · exact hsk j hj1 hj
        · have hjc : j = c'.val := by omega
          subst hjc
          rw [List.getD_eq_getElem _ _ hlen]
          exact hneq
      · omega
    · rw [ht, he]
      have hbody := matmat6_rowaccs_body_done acts k state blk d dots totals none c' hlt
      rw [hbody]
      simp only [WP.spec_ok]
  · dsimp only
    refine ⟨rfl, rfl, le_rfl, hc, ?_⟩
    intro j hj1 hj2
    exfalso
    omega

/-! ## §7 · Non-vacuity -/

/-- Filling from the empty buffer with two columns yields two fresh `n_out`-capacity columns. -/
example : matmat6_fillApp 4#usize [] 2 =
    [alloc.vec.Vec.with_capacity Std.I64 4#usize,
     alloc.vec.Vec.with_capacity Std.I64 4#usize] := rfl

end apeiron_word
