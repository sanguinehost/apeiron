import Serve.Kern.Matmat4

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
# Serving kernel group `kern.matmat` (part 5): `fast_family` and `per_column`

`src/engine/kern/matmat.rs:362-470` carries two drivers of the `matmat_q8act` route:

* `fast_family` (`:408-440`) decides whether every activation column can take the fast kernel: a
  `Q8_0`/`Q5_0` type, a whole number of `32`-wide blocks, `1 ≤ k ≤ MAX_K`, and each column with
  `n_in` codes, one scale per block and a passing `fast_bound_ok` (`top = 128` for `Q8_0`, `16` for
  `Q5_0`). It returns `Some family` exactly when every column admits, `None` on the first refusal or
  an outer shape refusal.
* `per_column` (`:362-376`) is the reference fallback: it maps `matvec_q8act` over the columns in
  order, refusing with the lowest-indexed column's error.

This file states WHAT those loops compute, not how they loop:

* `matmat5_takeB A i k` is the conjunction `A i && … && A (i+k-1)`; the prefix invariant of the
  family loops is `admitted && matmat5_takeB (colB) c (c' - c)` — an accepted prefix, a refusal at a
  column index, or exhaustion (`c' = |qs|`).
* `matmat5_prefF F c k` is the per-column list `[F c, …, F (c+k-1)]` the `per_column` output buffer
  accumulates.

Trust boundaries: rustc/LLVM codegen; the `F32`/AVX2 layers (`q8_0_block_kdots_avx2`,
`hsum128/256`); the `chart.QuantAct.magnitude` walk and `chart.matvec_q8act` are entered by their
values per column, not refined here; the `row_accs` / `pooled_rows` / `matmat_q8act` loops are not
covered.

No `sorry`/`admit`/`axiom`/`native_decide`/`bv_decide`; `decide` is not used on an extracted body.
-/

/-! ## §0 · References -/

/-- `fast_bound_ok n_in top q` reduced to `true`: the per-column bound test as a computable `Bool`.
The extracted function is kept as-is; `Result.reducesTo` reads its `ok` payload. -/
def matmat5_fbB (n_in : Std.Usize) (top : Std.I128) (q : chart.QuantAct) : Bool :=
  (kern.matmat.fast_bound_ok n_in top q).reducesTo true

/-- A column is admitted when its code count is `n_in`, its scale-block count is `blocks`, and
`fast_bound_ok` holds. -/
def matmat5_admitsB (n_in blocks : Std.Usize) (top : Std.I128) (q : chart.QuantAct) : Bool :=
  decide (q.codes.val.length = n_in.val) && decide (q.scales.val.length = blocks.val) &&
    matmat5_fbB n_in top q

/-- `fast_bound_ok` returns a definite `Bool` on `q` (it neither panics nor diverges). -/
def matmat5_boundTotal (n_in : Std.Usize) (top : Std.I128) (q : chart.QuantAct) : Prop :=
  ∃ b : Bool, kern.matmat.fast_bound_ok n_in top q = ok b

/-- The admission predicate of column `j`, totalised past the slice end with `qaDefault`. -/
def matmat5_colB (n_in blocks : Std.Usize) (top : Std.I128)
    (qs : Slice chart.QuantAct) : Nat → Bool :=
  fun j => matmat5_admitsB n_in blocks top (qs.val.getD j qaDefault)

/-- `matmat5_takeB A i k` is the conjunction `A i && A (i+1) && … && A (i+k-1)` (`true` at `k = 0`). -/
def matmat5_takeB (A : Nat → Bool) : Nat → Nat → Bool
  | _, 0 => true
  | i, k + 1 => matmat5_takeB A i k && A (i + k)

/-- The `Q8_0` family decision of the whole column vector: every column admits at `top = 128`. -/
def matmat5_fam8B (n_in blocks : Std.Usize) (qs : Slice chart.QuantAct) : Bool :=
  matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) 0 qs.val.length

/-- The `Q5_0` family decision of the whole column vector: every column admits at `top = 16`. -/
def matmat5_fam5B (n_in blocks : Std.Usize) (qs : Slice chart.QuantAct) : Bool :=
  matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) 0 qs.val.length

/-! ## §1 · The `fast_family` prefix reference -/

theorem matmat5_takeB_succ (A : Nat → Bool) (i k : Nat) :
    matmat5_takeB A i (k + 1) = (matmat5_takeB A i k && A (i + k)) := rfl

theorem matmat5_takeB_split (A : Nat → Bool) (i k m : Nat) :
    matmat5_takeB A i (k + m) = (matmat5_takeB A i k && matmat5_takeB A (i + k) m) := by
  induction m with
  | zero => simp [matmat5_takeB]
  | succ m ih => rw [Nat.add_succ, matmat5_takeB_succ, ih, matmat5_takeB_succ, Bool.and_assoc,
      Nat.add_assoc]

/-- `matmat5_takeB` at length one is the predicate itself. -/
theorem matmat5_takeB_one (A : Nat → Bool) (i : Nat) : matmat5_takeB A i 1 = A i := by
  rw [matmat5_takeB_succ A i 0]
  rfl

/-- A conjunction over a longer prefix stays `false` once a shorter one has. -/
theorem matmat5_takeB_mono_false (A : Nat → Bool) (i k m : Nat)
    (h : matmat5_takeB A i k = false) : matmat5_takeB A i (k + m) = false := by
  rw [matmat5_takeB_split, h, Bool.false_and]

/-- A conjunction over a prefix is `true` when every member is. -/
theorem matmat5_takeB_true (A : Nat → Bool) (i : Nat) :
    ∀ k, (∀ j, j < k → A (i + j) = true) → matmat5_takeB A i k = true
  | 0, _ => rfl
  | k + 1, h => by
    rw [matmat5_takeB_succ, matmat5_takeB_true A i k (fun j hj => h j (Nat.lt_trans hj (Nat.lt_succ_self k))),
      h k (Nat.lt_succ_self k)]
    rfl

/-- A member of a prefix at a concrete offset makes the conjunction `false`. -/
theorem matmat5_takeB_false_of (A : Nat → Bool) (i k : Nat) (h : A i = false) :
    matmat5_takeB A i (k + 1) = false := by
  rw [Nat.add_comm k 1, matmat5_takeB_split A i 1 k, matmat5_takeB_one, h, Bool.false_and]

/-- `Result.reducesTo true` on a definite `ok b` is `b`. -/
theorem matmat5_fbB_ok (n_in : Std.Usize) (top : Std.I128) (q : chart.QuantAct) (b : Bool)
    (h : kern.matmat.fast_bound_ok n_in top q = ok b) : matmat5_fbB n_in top q = b := by
  unfold matmat5_fbB Result.reducesTo
  rw [h]
  cases b <;> rfl

/-- The admission predicate, unfolded around a definite bound test. -/
theorem matmat5_admitsB_eq (n_in blocks : Std.Usize) (top : Std.I128) (q : chart.QuantAct)
    (b : Bool) (hfb : kern.matmat.fast_bound_ok n_in top q = ok b) :
    matmat5_admitsB n_in blocks top q =
      ((decide (q.codes.val.length = n_in.val) && decide (q.scales.val.length = blocks.val)) && b) := by
  rw [matmat5_admitsB, matmat5_fbB_ok n_in top q b hfb]

/-- In-bounds `getD` of the column slice is the indexed column. -/
theorem matmat5_colB_getElem (n_in blocks : Std.Usize) (top : Std.I128)
    (qs : Slice chart.QuantAct) (j : Nat) (hj : j < qs.val.length) :
    matmat5_colB n_in blocks top qs j = matmat5_admitsB n_in blocks top (qs.val[j]'hj) := by
  rw [matmat5_colB, List.getD_eq_getElem qs.val qaDefault hj]

/-- A column whose code count is not `n_in` is not admitted. -/
theorem matmat5_admitsB_false_of_len (n_in blocks : Std.Usize) (top : Std.I128)
    (q : chart.QuantAct) (h : ¬ q.codes.val.length = n_in.val) :
    matmat5_admitsB n_in blocks top q = false := by
  rw [matmat5_admitsB, decide_eq_false_iff_not.mpr h]
  simp

/-- A column without one scale per block is not admitted. -/
theorem matmat5_admitsB_false_of_blk (n_in blocks : Std.Usize) (top : Std.I128)
    (q : chart.QuantAct) (h : ¬ q.scales.val.length = blocks.val) :
    matmat5_admitsB n_in blocks top q = false := by
  rw [matmat5_admitsB, decide_eq_false_iff_not.mpr h]
  simp

/-! ## §2 · `fast_family_loop0`: the three-outcome body -/

/-- The `Q8_0` family body with `admitted = false` returns `false` without reading a column. -/
theorem matmat5_fast_family_loop0_body_done_false (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) :
    kern.matmat.fast_family_loop0.body n_in qs blocks c false = ok (ControlFlow.done false) := by
  unfold kern.matmat.fast_family_loop0.body
  by_cases hlt : c < Slice.len qs
  · rw [if_pos hlt, if_neg (by decide : ¬ (false = true))]
  · rw [if_neg hlt]

/-- The `Q8_0` family body past the end returns the carried admission flag. -/
theorem matmat5_fast_family_loop0_body_done (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) (adm : Bool) (hc : ¬ c.val < qs.val.length) :
    kern.matmat.fast_family_loop0.body n_in qs blocks c adm = ok (ControlFlow.done adm) := by
  unfold kern.matmat.fast_family_loop0.body
  rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hc)]

/-- The `Q8_0` family body in the admitting state: reading column `c`, the body advances to
`(c + 1, admitsB column c)`. The three outcomes are the `cont` here and the two `done` lemmas
above; the `magnitude` walk enters by its value (`hmag`), keyed on the column's admission. -/
theorem matmat5_fast_family_loop0_body_cont (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) (b : Bool) (hc : c.val < qs.val.length)
    (hfb : kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[c.val]'hc) = ok b)
    (hmag : matmat5_admitsB n_in blocks 128#i128 (qs.val[c.val]'hc) = true →
      chart.QuantAct.magnitude (qs.val[c.val]'hc) = ok (chart.ActMagnitude.Exact 0#i128))
    (hinc : c.val + 1 ≤ Std.Usize.max) :
    kern.matmat.fast_family_loop0.body n_in qs blocks c true ⦃ r =>
      ∃ c1 : Std.Usize, r = ControlFlow.cont
        (c1, matmat5_admitsB n_in blocks 128#i128 (qs.val[c.val]'hc)) ∧
        c1.val = c.val + 1 ⦄ := by
  have hadmEq := matmat5_admitsB_eq n_in blocks 128#i128 (qs.val[c.val]'hc) b hfb
  have hlc : c < Slice.len qs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hc
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  have hc1val : c1.val = c.val + 1 := by rw [hc1v, show (1#usize).val = 1 from by decide]
  have hnot : ¬ ((false : Bool) = true) := by decide
  unfold kern.matmat.fast_family_loop0.body
  rw [if_pos hlc, if_pos rfl, slice_index_usize_ok qs c hc]
  simp only [chart.QuantAct.len, bind_tc_ok]
  by_cases hcl : (qs.val[c.val]'hc).codes.val.length = n_in.val
  · rw [if_pos (UScalar.eq_of_val_eq (by rw [alloc.vec.Vec.len_val]; exact hcl))]
    simp only [chart.QuantAct.impl.scales, bind_tc_ok]
    by_cases hsl : (qs.val[c.val]'hc).scales.val.length = blocks.val
    · rw [if_pos (UScalar.eq_of_val_eq (by
        rw [Slice.len_val]; simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.length]
        exact hsl))]
      rw [hfb]
      simp only [bind_tc_ok]
      by_cases hb : b = true
      · rw [if_pos hb]
        have hA : matmat5_admitsB n_in blocks 128#i128 (qs.val[c.val]'hc) = true := by
          simp [hadmEq, hcl, hsl, hb]
        simp only [hmag hA, bind_tc_ok, massert_True]
        rw [hc1]
        simp only [bind_tc_ok]
        rw [WP.spec_ok]
        exact ⟨c1, by rw [hA], hc1val⟩
      · rw [if_neg hb]
        simp only [bind_tc_ok]
        rw [hc1]
        simp only [bind_tc_ok]
        rw [WP.spec_ok]
        have hbf : b = false := by cases b <;> simp_all
        exact ⟨c1, by rw [hadmEq, hbf]; simp, hc1val⟩
    · rw [if_neg (by
        intro hcc
        exact hsl (by
          have h := UScalar.val_eq_of_eq hcc
          rw [Slice.len_val] at h
          simpa only [alloc.vec.Vec.deref, Slice.from_val, Slice.length] using h))]
      simp only [bind_tc_ok]
      rw [if_neg hnot]
      simp only [bind_tc_ok]
      rw [hc1]
      simp only [bind_tc_ok]
      rw [WP.spec_ok]
      exact ⟨c1, by rw [matmat5_admitsB_false_of_blk n_in blocks 128#i128 _ hsl], hc1val⟩
  · rw [if_neg (by
      intro hcc
      exact hcl (by
        have h := UScalar.val_eq_of_eq hcc
        rw [alloc.vec.Vec.len_val] at h
        exact h))]
    simp only [bind_tc_ok]
    rw [if_neg hnot]
    simp only [bind_tc_ok]
    rw [hc1]
    simp only [bind_tc_ok]
    rw [WP.spec_ok]
    exact ⟨c1, by rw [matmat5_admitsB_false_of_len n_in blocks 128#i128 _ hcl], hc1val⟩

/-! ## §3 · `fast_family_loop0`: the three-outcome prefix invariant -/

/-- The `Q8_0` family loop from any cursor `c` and flag `admitted`: the return is
`admitted && takeB(colB) c (|qs| - c)` — an accepted prefix, a refusal at a column index, or
exhaustion, all read off the same conjunction. Domain: `fast_bound_ok` returns a definite `Bool` on
every column (`hbT`) and every admitting column has an `Exact` magnitude (`hM`); a column whose code
count or scale count is wrong needs neither and refuses. -/
theorem matmat5_fast_family_loop0_spec (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) (admitted : Bool) (hc : c.val ≤ qs.val.length)
    (hbT : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[j]'hj) = ok b)
    (hM : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 128#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop0 n_in qs blocks c admitted ⦃ r =>
      r = (admitted && matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) c.val
        (qs.val.length - c.val)) ⦄ := by
  unfold kern.matmat.fast_family_loop0
  apply loop.spec_decr_nat
    (measure := fun st : Std.Usize × Bool => qs.val.length - st.1.val)
    (inv := fun st => c.val ≤ st.1.val ∧ st.1.val ≤ qs.val.length ∧
      st.2 = (admitted && matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) c.val
        (st.1.val - c.val)))
    (post := fun y : Bool => y = (admitted && matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs)
      c.val (qs.val.length - c.val)))
  · rintro ⟨c', adm'⟩ ⟨hge, hle, hinv⟩
    dsimp only at hge hle hinv ⊢
    by_cases hlt : c'.val < qs.val.length
    · by_cases hadm : adm' = true
      · rw [hadm]
        obtain ⟨b, hb⟩ := hbT c'.val hlt
        obtain ⟨r, hbody, c1, hr, hc1v⟩ := WP.spec_imp_exists
          (matmat5_fast_family_loop0_body_cont n_in qs blocks c' b hlt hb
            (hM c'.val hlt) (by have := Slice.length_ineq qs; omega))
        rw [hbody, hr]
        simp only [WP.spec_ok]
        have hX : (admitted && matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) c.val
            (c'.val - c.val)) = true := by rw [← hinv]; exact hadm
        have hsplit : matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) c.val
            (c1.val - c.val) =
            (matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) c.val (c'.val - c.val) &&
              matmat5_colB n_in blocks 128#i128 qs c'.val) := by
          rw [show c1.val - c.val = (c'.val - c.val) + 1 from by omega, matmat5_takeB_succ,
            show c.val + (c'.val - c.val) = c'.val from by omega]
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · rw [hsplit, matmat5_colB_getElem n_in blocks 128#i128 qs c'.val hlt,
            ← Bool.and_assoc, hX, Bool.true_and]
        · omega
      · have hfalse : adm' = false := by cases adm' <;> simp_all
        rw [hfalse]
        have hbody := matmat5_fast_family_loop0_body_done_false n_in qs blocks c'
        rw [hbody]
        simp only [WP.spec_ok]
        rw [show qs.val.length - c.val = (c'.val - c.val) + (qs.val.length - c'.val) from by omega,
          matmat5_takeB_split, ← Bool.and_assoc, ← hinv, hfalse, Bool.false_and]
    · have hge' : c'.val = qs.val.length := le_antisymm hle (Nat.le_of_not_lt hlt)
      have hbody := matmat5_fast_family_loop0_body_done n_in qs blocks c' adm' hlt
      rw [hbody]
      simp only [WP.spec_ok]
      rw [← hge']
      exact hinv
  · exact ⟨le_rfl, hc, by
      rw [Nat.sub_self, show matmat5_takeB (matmat5_colB n_in blocks 128#i128 qs) c.val 0 = true from rfl,
        Bool.and_true]⟩

/-- **Success law.** From `(0, true)` every admitting column makes the `Q8_0` family loop return
`true`; a non-admitting column makes it return `false`. The reference `matmat5_fam8B` is the
conjunction over all columns. -/
theorem word_matmat5_fast_family_loop0_eq (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize)
    (hbT : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[j]'hj) = ok b)
    (hM : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 128#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop0 n_in qs blocks 0#usize true ⦃ r =>
      r = matmat5_fam8B n_in blocks qs ⦄ := by
  obtain ⟨y, hy, hpost⟩ := WP.spec_imp_exists
    (matmat5_fast_family_loop0_spec n_in qs blocks 0#usize true (Nat.zero_le _) hbT hM)
  rw [hy]
  simp only [WP.spec_ok]
  rw [hpost, show (0#usize).val = 0 from rfl, Nat.sub_zero]
  rfl
/-! ## §4a · `fast_family_loop1`: the three-outcome body -/

/-- The `Q5_0` family body with `admitted = false` returns `false` without reading a column. -/
theorem matmat5_fast_family_loop1_body_done_false (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) :
    kern.matmat.fast_family_loop1.body n_in qs blocks c false = ok (ControlFlow.done false) := by
  unfold kern.matmat.fast_family_loop1.body
  by_cases hlt : c < Slice.len qs
  · rw [if_pos hlt, if_neg (by decide : ¬ (false = true))]
  · rw [if_neg hlt]

/-- The `Q5_0` family body past the end returns the carried admission flag. -/
theorem matmat5_fast_family_loop1_body_done (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) (adm : Bool) (hc : ¬ c.val < qs.val.length) :
    kern.matmat.fast_family_loop1.body n_in qs blocks c adm = ok (ControlFlow.done adm) := by
  unfold kern.matmat.fast_family_loop1.body
  rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hc)]

/-- The `Q5_0` family body in the admitting state: reading column `c`, the body advances to
`(c + 1, admitsB column c)`. The three outcomes are the `cont` here and the two `done` lemmas
above; the `magnitude` walk enters by its value (`hmag`), keyed on the column's admission. -/
theorem matmat5_fast_family_loop1_body_cont (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) (b : Bool) (hc : c.val < qs.val.length)
    (hfb : kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[c.val]'hc) = ok b)
    (hmag : matmat5_admitsB n_in blocks 16#i128 (qs.val[c.val]'hc) = true →
      chart.QuantAct.magnitude (qs.val[c.val]'hc) = ok (chart.ActMagnitude.Exact 0#i128))
    (hinc : c.val + 1 ≤ Std.Usize.max) :
    kern.matmat.fast_family_loop1.body n_in qs blocks c true ⦃ r =>
      ∃ c1 : Std.Usize, r = ControlFlow.cont
        (c1, matmat5_admitsB n_in blocks 16#i128 (qs.val[c.val]'hc)) ∧
        c1.val = c.val + 1 ⦄ := by
  have hadmEq := matmat5_admitsB_eq n_in blocks 16#i128 (qs.val[c.val]'hc) b hfb
  have hlc : c < Slice.len qs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hc
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  have hc1val : c1.val = c.val + 1 := by rw [hc1v, show (1#usize).val = 1 from by decide]
  have hnot : ¬ ((false : Bool) = true) := by decide
  unfold kern.matmat.fast_family_loop1.body
  rw [if_pos hlc, if_pos rfl, slice_index_usize_ok qs c hc]
  simp only [chart.QuantAct.len, bind_tc_ok]
  by_cases hcl : (qs.val[c.val]'hc).codes.val.length = n_in.val
  · rw [if_pos (UScalar.eq_of_val_eq (by rw [alloc.vec.Vec.len_val]; exact hcl))]
    simp only [chart.QuantAct.impl.scales, bind_tc_ok]
    by_cases hsl : (qs.val[c.val]'hc).scales.val.length = blocks.val
    · rw [if_pos (UScalar.eq_of_val_eq (by
        rw [Slice.len_val]; simp only [alloc.vec.Vec.deref, Slice.from_val, Slice.length]
        exact hsl))]
      rw [hfb]
      simp only [bind_tc_ok]
      by_cases hb : b = true
      · rw [if_pos hb]
        have hA : matmat5_admitsB n_in blocks 16#i128 (qs.val[c.val]'hc) = true := by
          simp [hadmEq, hcl, hsl, hb]
        simp only [hmag hA, bind_tc_ok, massert_True]
        rw [hc1]
        simp only [bind_tc_ok]
        rw [WP.spec_ok]
        exact ⟨c1, by rw [hA], hc1val⟩
      · rw [if_neg hb]
        simp only [bind_tc_ok]
        rw [hc1]
        simp only [bind_tc_ok]
        rw [WP.spec_ok]
        have hbf : b = false := by cases b <;> simp_all
        exact ⟨c1, by rw [hadmEq, hbf]; simp, hc1val⟩
    · rw [if_neg (by
        intro hcc
        exact hsl (by
          have h := UScalar.val_eq_of_eq hcc
          rw [Slice.len_val] at h
          simpa only [alloc.vec.Vec.deref, Slice.from_val, Slice.length] using h))]
      simp only [bind_tc_ok]
      rw [if_neg hnot]
      simp only [bind_tc_ok]
      rw [hc1]
      simp only [bind_tc_ok]
      rw [WP.spec_ok]
      exact ⟨c1, by rw [matmat5_admitsB_false_of_blk n_in blocks 16#i128 _ hsl], hc1val⟩
  · rw [if_neg (by
      intro hcc
      exact hcl (by
        have h := UScalar.val_eq_of_eq hcc
        rw [alloc.vec.Vec.len_val] at h
        exact h))]
    simp only [bind_tc_ok]
    rw [if_neg hnot]
    simp only [bind_tc_ok]
    rw [hc1]
    simp only [bind_tc_ok]
    rw [WP.spec_ok]
    exact ⟨c1, by rw [matmat5_admitsB_false_of_len n_in blocks 16#i128 _ hcl], hc1val⟩

/-! ## §4b · `fast_family_loop1`: the three-outcome prefix invariant -/

/-- The `Q5_0` family loop from any cursor `c` and flag `admitted`: the return is
`admitted && takeB(colB) c (|qs| - c)` — an accepted prefix, a refusal at a column index, or
exhaustion, all read off the same conjunction. Domain: `fast_bound_ok` returns a definite `Bool` on
every column (`hbT`) and every admitting column has an `Exact` magnitude (`hM`); a column whose code
count or scale count is wrong needs neither and refuses. -/
theorem matmat5_fast_family_loop1_spec (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize) (c : Std.Usize) (admitted : Bool) (hc : c.val ≤ qs.val.length)
    (hbT : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[j]'hj) = ok b)
    (hM : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 16#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop1 n_in qs blocks c admitted ⦃ r =>
      r = (admitted && matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) c.val
        (qs.val.length - c.val)) ⦄ := by
  unfold kern.matmat.fast_family_loop1
  apply loop.spec_decr_nat
    (measure := fun st : Std.Usize × Bool => qs.val.length - st.1.val)
    (inv := fun st => c.val ≤ st.1.val ∧ st.1.val ≤ qs.val.length ∧
      st.2 = (admitted && matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) c.val
        (st.1.val - c.val)))
    (post := fun y : Bool => y = (admitted && matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs)
      c.val (qs.val.length - c.val)))
  · rintro ⟨c', adm'⟩ ⟨hge, hle, hinv⟩
    dsimp only at hge hle hinv ⊢
    by_cases hlt : c'.val < qs.val.length
    · by_cases hadm : adm' = true
      · rw [hadm]
        obtain ⟨b, hb⟩ := hbT c'.val hlt
        obtain ⟨r, hbody, c1, hr, hc1v⟩ := WP.spec_imp_exists
          (matmat5_fast_family_loop1_body_cont n_in qs blocks c' b hlt hb
            (hM c'.val hlt) (by have := Slice.length_ineq qs; omega))
        rw [hbody, hr]
        simp only [WP.spec_ok]
        have hX : (admitted && matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) c.val
            (c'.val - c.val)) = true := by rw [← hinv]; exact hadm
        have hsplit : matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) c.val
            (c1.val - c.val) =
            (matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) c.val (c'.val - c.val) &&
              matmat5_colB n_in blocks 16#i128 qs c'.val) := by
          rw [show c1.val - c.val = (c'.val - c.val) + 1 from by omega, matmat5_takeB_succ,
            show c.val + (c'.val - c.val) = c'.val from by omega]
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · rw [hsplit, matmat5_colB_getElem n_in blocks 16#i128 qs c'.val hlt,
            ← Bool.and_assoc, hX, Bool.true_and]
        · omega
      · have hfalse : adm' = false := by cases adm' <;> simp_all
        rw [hfalse]
        have hbody := matmat5_fast_family_loop1_body_done_false n_in qs blocks c'
        rw [hbody]
        simp only [WP.spec_ok]
        rw [show qs.val.length - c.val = (c'.val - c.val) + (qs.val.length - c'.val) from by omega,
          matmat5_takeB_split, ← Bool.and_assoc, ← hinv, hfalse, Bool.false_and]
    · have hge' : c'.val = qs.val.length := le_antisymm hle (Nat.le_of_not_lt hlt)
      have hbody := matmat5_fast_family_loop1_body_done n_in qs blocks c' adm' hlt
      rw [hbody]
      simp only [WP.spec_ok]
      rw [← hge']
      exact hinv
  · exact ⟨le_rfl, hc, by
      rw [Nat.sub_self, show matmat5_takeB (matmat5_colB n_in blocks 16#i128 qs) c.val 0 = true from rfl,
        Bool.and_true]⟩

/-- **Success law.** From `(0, true)` every admitting column makes the `Q5_0` family loop return
`true`; a non-admitting column makes it return `false`. The reference `matmat5_fam5B` is the
conjunction over all columns. -/
theorem word_matmat5_fast_family_loop1_eq (n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (blocks : Std.Usize)
    (hbT : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[j]'hj) = ok b)
    (hM : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 16#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop1 n_in qs blocks 0#usize true ⦃ r =>
      r = matmat5_fam5B n_in blocks qs ⦄ := by
  obtain ⟨y, hy, hpost⟩ := WP.spec_imp_exists
    (matmat5_fast_family_loop1_spec n_in qs blocks 0#usize true (Nat.zero_le _) hbT hM)
  rw [hy]
  simp only [WP.spec_ok]
  rw [hpost, show (0#usize).val = 0 from rfl, Nat.sub_zero]
  rfl

/-! ## §5 · `fast_family`: the family decision -/

/-- The `Q8_0` family loop from `(0, true)` returns `ok` of the whole-vector conjunction. -/
theorem matmat5_loop0_eq (n_in : Std.Usize) (qs : Slice chart.QuantAct) (blocks : Std.Usize)
    (hbT : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[j]'hj) = ok b)
    (hM : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 128#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop0 n_in qs blocks 0#usize true =
      ok (matmat5_fam8B n_in blocks qs) := by
  obtain ⟨y, hy, hpost⟩ := WP.spec_imp_exists
    (word_matmat5_fast_family_loop0_eq n_in qs blocks hbT hM)
  rw [hy, hpost]

/-- The `Q5_0` family loop from `(0, true)` returns `ok` of the whole-vector conjunction. -/
theorem matmat5_loop1_eq (n_in : Std.Usize) (qs : Slice chart.QuantAct) (blocks : Std.Usize)
    (hbT : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[j]'hj) = ok b)
    (hM : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 16#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family_loop1 n_in qs blocks 0#usize true =
      ok (matmat5_fam5B n_in blocks qs) := by
  obtain ⟨y, hy, hpost⟩ := WP.spec_imp_exists
    (word_matmat5_fast_family_loop1_eq n_in qs blocks hbT hM)
  rw [hy, hpost]

/-- The `fast_family` outcome: a `Q8_0`/`Q5_0` family exactly when `n_in` is a whole number of
`ACT_QK` blocks, `qs` is non-empty, holds at most `MAX_K` columns and every column admits; `none`
otherwise or on an unhandled type. -/
def matmat5_familyRef (ggml_type : Std.U32) (n_in blocks : Std.Usize) (qs : Slice chart.QuantAct) :
    Option kern.matmat.Family :=
  if ggml_type = 8#uscalar then
    (if n_in.val % kern.matmat.ACT_QK.val = 0 then
       (if qs.val.length = 0 then none
        else if qs.val.length > kern.matmat.MAX_K.val then none
        else if matmat5_fam8B n_in blocks qs then some kern.matmat.Family.Q8_0 else none)
     else none)
  else if ggml_type = 6#uscalar then
    (if n_in.val % kern.matmat.ACT_QK.val = 0 then
       (if qs.val.length = 0 then none
        else if qs.val.length > kern.matmat.MAX_K.val then none
        else if matmat5_fam5B n_in blocks qs then some kern.matmat.Family.Q5_0 else none)
     else none)
  else none

/-- **Driver law.** `fast_family` returns exactly `matmat5_familyRef` at the division witness
`blocks = n_in / ACT_QK`: the `Q8_0`/`Q5_0` family when the shape checks and every column's admission
pass, and `none` otherwise; an unhandled `ggml_type` is `none`. Domain: the bound test returns a
definite `Bool` on every column under both tops (`hb8`, `hb5`) and every admitting column has an
`Exact` magnitude (`hM8`, `hM5`). -/
theorem word_matmat5_fast_family_eq (ggml_type : Std.U32) (n_in : Std.Usize)
    (qs : Slice chart.QuantAct) (blocks : Std.Usize)
    (hblk : n_in / kern.matmat.ACT_QK = ok blocks)
    (hb8 : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 128#i128 (qs.val[j]'hj) = ok b)
    (hM8 : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 128#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128))
    (hb5 : ∀ (j : Nat) (hj : j < qs.val.length),
      ∃ b : Bool, kern.matmat.fast_bound_ok n_in 16#i128 (qs.val[j]'hj) = ok b)
    (hM5 : ∀ (j : Nat) (hj : j < qs.val.length),
      matmat5_admitsB n_in blocks 16#i128 (qs.val[j]'hj) = true →
      chart.QuantAct.magnitude (qs.val[j]'hj) = ok (chart.ActMagnitude.Exact 0#i128)) :
    kern.matmat.fast_family ggml_type n_in qs = ok (matmat5_familyRef ggml_type n_in blocks qs) := by
  have h0 := matmat5_loop0_eq n_in qs blocks hb8 hM8
  have h1 := matmat5_loop1_eq n_in qs blocks hb5 hM5
  unfold kern.matmat.fast_family matmat5_familyRef
  rw [hblk]
  simp only [bind_tc_ok, core.num.Usize.is_multiple_of, UScalar.is_multiple_of,
    core.slice.Slice.is_empty, Slice.length, decide_eq_true_eq, beq_iff_eq, Slice.len_val,
    gt_iff_lt, UScalar.lt_equiv, if_true, if_false]
  rw [h0, h1]
  by_cases h8 : ggml_type = 8#uscalar
  · subst ggml_type
    simp only [bind_tc_ok, if_true]
    by_cases hmod : n_in.val % kern.matmat.ACT_QK.val = 0
    · simp only [if_pos hmod]
      by_cases hem : qs.val.length = 0
      · simp only [if_pos hem]; rfl
      · simp only [if_neg hem]
        by_cases hk : qs.val.length > kern.matmat.MAX_K.val
        · simp only [if_pos hk]; rfl
        · simp only [if_neg hk]
          by_cases hfam : matmat5_fam8B n_in blocks qs = true
          · simp only [if_pos hfam]; rfl
          · simp only [if_neg hfam]; rfl
    · simp only [if_neg hmod]; rfl
  · by_cases h6 : ggml_type = 6#uscalar
    · subst ggml_type
      simp only [bind_tc_ok, if_true, if_false]
      by_cases hmod : n_in.val % kern.matmat.ACT_QK.val = 0
      · simp only [if_pos hmod]
        by_cases hem : qs.val.length = 0
        · simp only [if_pos hem]; rfl
        · simp only [if_neg hem]
          by_cases hk : qs.val.length > kern.matmat.MAX_K.val
          · simp only [if_pos hk]; rfl
          · simp only [if_neg hk]
            by_cases hfam : matmat5_fam5B n_in blocks qs = true
            · simp only [if_pos hfam]; rfl
            · simp only [if_neg hfam]; rfl
      · simp only [if_neg hmod]; rfl
    · rw [if_neg h8, if_neg h6]
      split
      · exact absurd rfl h8
      · exact absurd rfl h6
      · rfl

/-! ## §6 · `per_column`: the ordered fallback -/

/-- The per-column value prefix `F c, F (c+1), …, F (c+k-1)`. -/
def matmat5_prefF (F : Nat → alloc.vec.Vec Std.I64) (c : Nat) : Nat → List (alloc.vec.Vec Std.I64)
  | 0 => []
  | k + 1 => matmat5_prefF F c k ++ [F (c + k)]

theorem matmat5_prefF_succ (F : Nat → alloc.vec.Vec Std.I64) (c k : Nat) :
    matmat5_prefF F c (k + 1) = matmat5_prefF F c k ++ [F (c + k)] := rfl

theorem matmat5_prefF_length (F : Nat → alloc.vec.Vec Std.I64) (c k : Nat) :
    (matmat5_prefF F c k).length = k := by
  induction k with
  | zero => rfl
  | succ k ih => rw [matmat5_prefF_succ, List.length_append, ih, List.length_singleton]

theorem matmat5_with_capacity_val (T : Type) (n : Std.Usize) :
    (alloc.vec.Vec.with_capacity T n).val = ([] : List T) := by
  simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]

/-- The `per_column` body past the slice end returns the finished buffer. -/
theorem matmat5_per_column_loop_body_done (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (out : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (hc : ¬ c.val < qs.val.length) :
    kern.matmat.per_column_loop.body ggml_type raw n_out n_in qs out c =
      ok (ControlFlow.done (core.result.Result.Ok out)) := by
  unfold kern.matmat.per_column_loop.body
  rw [if_neg (by rw [UScalar.lt_equiv, Slice.len_val]; exact hc)]

/-- The `per_column` body on an `Ok` column: push the column value and advance. -/
theorem matmat5_per_column_loop_body_cont (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (out : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (v : alloc.vec.Vec Std.I64) (hc : c.val < qs.val.length)
    (hr : chart.matvec_q8act ggml_type raw n_out n_in (qs.val[c.val]'hc) =
      ok (core.result.Result.Ok v))
    (hcap : out.val.length < Std.Usize.max) :
    kern.matmat.per_column_loop.body ggml_type raw n_out n_in qs out c ⦃ r =>
      ∃ (out1 : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c1 : Std.Usize),
        r = ControlFlow.cont (out1, c1) ∧ out1.val = out.val ++ [v] ∧ c1.val = c.val + 1 ⦄ := by
  unfold kern.matmat.per_column_loop.body
  have hlc : c < Slice.len qs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hc
  rw [if_pos hlc, slice_index_usize_ok qs c hc]
  simp only [bind_tc_ok]
  rw [hr]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  obtain ⟨out1, hout1, hout1v⟩ := WP.spec_imp_exists (alloc.vec.Vec.push_spec out v hcap)
  obtain ⟨c1, hc1, hc1v⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := c) (y := 1#usize) (by scalar_tac))
  rw [hout1]
  simp only [bind_tc_ok]
  rw [hc1]
  simp only [bind_tc_ok]
  rw [WP.spec_ok]
  refine ⟨out1, c1, rfl, hout1v, ?_⟩
  rw [hc1v, show (1#usize).val = 1 from by decide]

/-- The `per_column` loop: from any cursor `c` and buffer `out`, when every column from `c` on
returns `Ok`, the loop returns `Ok out'` where `out'` appends the column values `F c …` to `out`.
The `matvec_q8act` kernel enters by its value per column (`hF`); the `FromSame` error lift is
unchanged. Domain: `c ≤ |qs|` and the buffer cannot exceed the vector capacity (`hcap`). -/
theorem matmat5_per_column_loop_spec (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (out : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (F : Nat → alloc.vec.Vec Std.I64) (hc : c.val ≤ qs.val.length)
    (hF : ∀ (j : Nat) (hj : j < qs.val.length),
      chart.matvec_q8act ggml_type raw n_out n_in (qs.val[j]'hj) =
        ok (core.result.Result.Ok (F j)))
    (hcap : out.val.length + qs.val.length < Std.Usize.max) :
    kern.matmat.per_column_loop ggml_type raw n_out n_in qs out c ⦃ r =>
      ∃ out', r = core.result.Result.Ok out' ∧
        out'.val = out.val ++ matmat5_prefF F c.val (qs.val.length - c.val) ⦄ := by
  unfold kern.matmat.per_column_loop
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      qs.val.length - st.2.val)
    (inv := fun st => c.val ≤ st.2.val ∧ st.2.val ≤ qs.val.length ∧
      st.1.val = out.val ++ matmat5_prefF F c.val (st.2.val - c.val))
    (post := fun y => ∃ out' : alloc.vec.Vec (alloc.vec.Vec Std.I64),
      y = core.result.Result.Ok out' ∧
      out'.val = out.val ++ matmat5_prefF F c.val (qs.val.length - c.val))
  · rintro ⟨out', c'⟩ ⟨hge, hle, hinv⟩
    dsimp only at hge hle hinv ⊢
    by_cases hlt : c'.val < qs.val.length
    · have hcap' : out'.val.length < Std.Usize.max := by
        rw [hinv, List.length_append, matmat5_prefF_length]
        omega
      obtain ⟨r, hbody, hpost⟩ := WP.spec_imp_exists
        (matmat5_per_column_loop_body_cont ggml_type raw n_out n_in qs out' c'
          (F c'.val) hlt (hF c'.val hlt) hcap')
      obtain ⟨out1, c1, hr, hout1v, hc1v⟩ := hpost
      rw [hbody, hr]
      simp only [WP.spec_ok]
      refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
      · omega
      · omega
      · calc out1.val = (out.val ++ matmat5_prefF F c.val (c'.val - c.val)) ++ [F c'.val] := by
              rw [hout1v, hinv]
          _ = out.val ++ (matmat5_prefF F c.val (c'.val - c.val) ++ [F c'.val]) :=
              List.append_assoc ..
          _ = out.val ++ matmat5_prefF F c.val ((c'.val - c.val) + 1) := by
              rw [matmat5_prefF_succ, show c.val + (c'.val - c.val) = c'.val from by omega]
          _ = out.val ++ matmat5_prefF F c.val (c1.val - c.val) := by
              rw [hc1v, show (c'.val + 1) - c.val = (c'.val - c.val) + 1 from by omega]
      · omega
    · have hge' : c'.val = qs.val.length := le_antisymm hle (Nat.le_of_not_lt hlt)
      rw [matmat5_per_column_loop_body_done ggml_type raw n_out n_in qs out' c' hlt]
      simp only [WP.spec_ok]
      exact ⟨out', rfl, by rw [hinv, ← hge']⟩
  · exact ⟨le_rfl, hc, by
      rw [Nat.sub_self, show matmat5_prefF F c.val 0 = [] from rfl, List.append_nil]⟩

/-- **Success law.** When every column returns `Ok`, `per_column` returns `Ok out'` with
`out'.val = matmat5_prefF F 0 |qs|`, i.e. the columns in order. -/
theorem word_matmat5_per_column_ok (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct) (F : Nat → alloc.vec.Vec Std.I64)
    (hF : ∀ (j : Nat) (hj : j < qs.val.length),
      chart.matvec_q8act ggml_type raw n_out n_in (qs.val[j]'hj) =
        ok (core.result.Result.Ok (F j)))
    (hcap : qs.val.length < Std.Usize.max) :
    kern.matmat.per_column ggml_type raw n_out n_in qs ⦃ r =>
      ∃ out' : alloc.vec.Vec (alloc.vec.Vec Std.I64), r = core.result.Result.Ok out' ∧
        out'.val = matmat5_prefF F 0 qs.val.length ⦄ := by
  unfold kern.matmat.per_column
  have hw : (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len qs)).val.length +
      qs.val.length < Std.Usize.max := by
    rw [matmat5_with_capacity_val]; simp only [List.length_nil, Nat.zero_add]; exact hcap
  obtain ⟨y, hy, hpost⟩ := WP.spec_imp_exists
    (matmat5_per_column_loop_spec ggml_type raw n_out n_in qs
      (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len qs)) 0#usize F
      (Nat.zero_le _) hF hw)
  rw [hy]
  simp only [WP.spec_ok]
  obtain ⟨out', hout', hout'v⟩ := hpost
  refine ⟨out', ?_, ?_⟩
  · rw [hout']
  · rw [hout'v, matmat5_with_capacity_val, List.nil_append]
    rw [show (0#usize).val = 0 from rfl, Nat.sub_zero]

/-! ## §6b · `per_column`: refusal propagation -/

/-- The `per_column` body on an `Err` column: `Result::branch` breaks with the residual, the
`FromSame` lift carries the error unchanged, and the body returns the finished buffer holding the
lowest-indexed error. -/
theorem matmat5_per_column_loop_body_err (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (out : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (e : matvec.MatvecError) (hc : c.val < qs.val.length)
    (hr : chart.matvec_q8act ggml_type raw n_out n_in (qs.val[c.val]'hc) =
      ok (core.result.Result.Err e)) :
    kern.matmat.per_column_loop.body ggml_type raw n_out n_in qs out c =
      ok (ControlFlow.done (core.result.Result.Err e)) := by
  unfold kern.matmat.per_column_loop.body
  have hlc : c < Slice.len qs := by rw [UScalar.lt_equiv, Slice.len_val]; exact hc
  rw [if_pos hlc, slice_index_usize_ok qs c hc]
  simp only [bind_tc_ok]
  rw [hr]
  simp only [core.result.Result.Insts.CoreOpsTry.branch, bind_tc_ok]
  obtain ⟨r1, hr1, hr1v⟩ := WP.spec_imp_exists
    (core.result.Result.Insts.CoreOpsTryTraitFromResidualResultInfallible.from_residual_Err.spec
      (alloc.vec.Vec (alloc.vec.Vec Std.I64)) (core.convert.FromSame matvec.MatvecError) e e
      (by rfl))
  rw [hr1, hr1v]
  simp only [bind_tc_ok]

/-- **Refusal law.** When the column at the cursor refuses, the `per_column` loop returns that error
without reading a later column: the measure `|qs| - c'` decreases only on `Continue`, and the body
`done`s on the `Err` column. -/
theorem matmat5_per_column_loop_refuses (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct)
    (out : alloc.vec.Vec (alloc.vec.Vec Std.I64)) (c : Std.Usize)
    (e : matvec.MatvecError) (hc : c.val < qs.val.length)
    (hr : chart.matvec_q8act ggml_type raw n_out n_in (qs.val[c.val]'hc) =
      ok (core.result.Result.Err e)) :
    kern.matmat.per_column_loop ggml_type raw n_out n_in qs out c ⦃ r =>
      r = core.result.Result.Err e ⦄ := by
  unfold kern.matmat.per_column_loop
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      qs.val.length - st.2.val)
    (inv := fun st => st.2.val = c.val)
    (post := fun y => y = core.result.Result.Err e)
  · rintro ⟨out', c'⟩ hc'
    dsimp only at hc' ⊢
    have hc'e : c' = c := UScalar.eq_of_val_eq hc'
    rw [hc'e]
    rw [matmat5_per_column_loop_body_err ggml_type raw n_out n_in qs out' c e hc hr]
    simp only [WP.spec_ok]
  · rfl

/-- **Driver refusal.** When the cursor-0 column of a non-empty `qs` refuses with `e`, `per_column`
returns `Err e`. -/
theorem word_matmat5_per_column_refuses (ggml_type : Std.U32) (raw : Slice Std.U8)
    (n_out n_in : Std.Usize) (qs : Slice chart.QuantAct) (e : matvec.MatvecError)
    (hne : (0#usize).val < qs.val.length)
    (hr : chart.matvec_q8act ggml_type raw n_out n_in (qs.val[(0#usize).val]'hne) =
      ok (core.result.Result.Err e)) :
    kern.matmat.per_column ggml_type raw n_out n_in qs ⦃ r =>
      r = core.result.Result.Err e ⦄ := by
  unfold kern.matmat.per_column
  exact matmat5_per_column_loop_refuses ggml_type raw n_out n_in qs
    (alloc.vec.Vec.with_capacity (alloc.vec.Vec Std.I64) (Slice.len qs)) 0#usize e hne hr

/-! ## §7 · Non-vacuity -/

/-- The prefix conjunction at length three over `j < 3` is `true`. -/
example : matmat5_takeB (fun j => decide (j < 3)) 0 3 = true := by decide

/-- The prefix conjunction at length five over `j < 3` is `false`: the predicate is not constantly
`true`. -/
example : matmat5_takeB (fun j => decide (j < 3)) 0 5 = false := by decide

/-- An empty prefix is the empty list at any `F`, `c`. -/
example (F : Nat → alloc.vec.Vec Std.I64) (c : Nat) : matmat5_prefF F c 0 = [] := rfl

/-- With `qs` empty the division witness `0 / 32 = 0` holds, the per-column hypotheses of
`word_matmat5_fast_family_eq` are vacuous and the hypotheses hold jointly. -/
example (ggml_type : Std.U32) :
    kern.matmat.fast_family ggml_type 0#usize (Slice.new chart.QuantAct) =
      ok (matmat5_familyRef ggml_type 0#usize 0#usize (Slice.new chart.QuantAct)) := by
  have hblk : 0#usize / kern.matmat.ACT_QK = ok (0#usize) := by
    obtain ⟨z, hz, hzv⟩ := UScalar.div_spec (0#usize) (y := kern.matmat.ACT_QK) (by
      rw [kern.matmat.ACT_QK]; decide)
    rw [hz]
    congr 1
    exact UScalar.eq_of_val_eq (by rw [hzv, kern.matmat.ACT_QK]; rfl)
  exact word_matmat5_fast_family_eq ggml_type 0#usize (Slice.new chart.QuantAct) 0#usize hblk
    (fun _ hj => by simp [Slice.new_val] at hj)
    (fun _ hj => by simp [Slice.new_val] at hj)
    (fun _ hj => by simp [Slice.new_val] at hj)
    (fun _ hj => by simp [Slice.new_val] at hj)

/-- With `qs` empty the `per_column` output buffer is empty for any per-column value map: the
`hcap` domain hypothesis holds and the per-column hypothesis is vacuous. -/
example (ggml_type : Std.U32) (raw : Slice Std.U8) (n_out n_in : Std.Usize)
    (F : Nat → alloc.vec.Vec Std.I64) :
    kern.matmat.per_column ggml_type raw n_out n_in (Slice.new chart.QuantAct) ⦃ r =>
      ∃ out' : alloc.vec.Vec (alloc.vec.Vec Std.I64), r = core.result.Result.Ok out' ∧
        out'.val = [] ⦄ :=
  word_matmat5_per_column_ok ggml_type raw n_out n_in (Slice.new chart.QuantAct) F
    (fun _ hj => by simp [Slice.new_val] at hj)
    (by rw [Slice.new_val]; exact Nat.lt_of_lt_of_le Nat.zero_lt_one usize_max_ge_one)

end apeiron_word
