/-
Integer and layout refinement of the extracted `kern::gemm` bodies, continued from `Serve/Gemm.lean`.

Serving path: `Engine::next_token_state` -> `step::cols` -> `kern::gemm::gemm_q8_0_group`. This
module refines the portable (non-AVX2, non-`unsafe`) layout bodies of `src/engine/kern/gemm.rs`:

* the layout bodies `empty_columns`, `split_part`, `part_row_starts`.

NOT refined here: the activation packer `pack_group`, the integer compute body that turns each
column group into its `x + 128` lane words and its per-block exponent shifts. Its extracted body
carries seven nested loop bodies, `qs[c].scales()`/`codes()` trait-method calls and `&mut` slices,
and this module states no theorem about it.

Each reference is plain Lean over `Nat`/`List`: it states WHAT the body computes (index arithmetic
on a partition of rows into parts), not how any loop runs.
`word_<fn>_ok` is the refinement; `word_<fn>_refuses` is the exact refusal condition.

Boundaries, stated rather than proved here: the AVX2/AVX-VNNI core (`kern.gemm.x86.rows`,
`kern.gemm.x86.rows_swiglu`, emitted by the extraction as `axiom`), the pool drivers
(`chart.pool.chunks.for_chunks4`, `chart.pool.chunks.for_pbands`, `chart.pool.run_global`,
`chart.pool.Flag`) and the `unsafe`/raw-pointer containers (`column_ptrs`, `ColPtrs`,
`SwigluPtrs`). The scalar dot-product bodies `exact_dot`, `row_scales` and `swiglu_one` are not
extracted: they are reachable only from the AVX2 `axiom`, so the extraction emits no body for them.
-/
import ApeironWord

namespace apeiron_word

open Aeneas Aeneas.Std Result

/-! ## Generic list scaffolding -/

private lemma replicate_append_one {α : Type _} (n : Nat) (a : α) :
    List.replicate n a ++ [a] = List.replicate (n + 1) a := by
  induction n with
  | zero => rfl
  | succ m ih =>
      show a :: (List.replicate m a ++ [a]) = List.replicate (m + 1 + 1) a
      rw [ih]
      show a :: List.replicate (m + 1) a = a :: List.replicate (m + 1) a
      rfl

/-- Empty value list of a capacity-only vector. -/
private lemma withCapacity_val (T : Type) (n : Std.Usize) :
    (alloc.vec.Vec.with_capacity T n).val = [] := rfl

/-- The word maximum of `usize`. -/
private lemma usize_max_eq' : Usize.max = 2 ^ System.Platform.numBits - 1 := by
  rw [Usize.max_def, Usize.numBits_def, UScalarTy.Usize_numBits_eq]

private lemma usize_val_le_max' (x : Std.Usize) : x.val ≤ Usize.max := by
  have h : x.bv.toNat < 2 ^ System.Platform.numBits := by
    simpa only [UScalarTy.Usize_numBits_eq] using x.bv.isLt
  show x.bv.toNat ≤ Usize.max
  rw [usize_max_eq']
  omega

/-! ## `empty_columns` -/

/-- `k` empty columns: the value list of each column is empty. -/
def emptyColumnsRef (k : Nat) : List (List Std.I64) := List.replicate k []

/-- The value lists of a vector-of-vectors, flattened one level: `fun v => v.val`. -/
def vecVals {α : Type} (v : alloc.vec.Vec (alloc.vec.Vec α)) : List (List α) :=
  v.val.map (fun x => x.val)

theorem empty_columns_loop_spec (k cap : Std.Usize) (c : Std.Usize)
    (cols : alloc.vec.Vec (alloc.vec.Vec Std.I64))
    (hc : c.val ≤ k.val) (hcols : vecVals cols = List.replicate c.val []) :
    kern.gemm.empty_columns_loop k cap cols c ⦃ r =>
      vecVals r = emptyColumnsRef k.val ⦄ := by
  unfold kern.gemm.empty_columns_loop
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      k.val - st.2.val)
    (inv := fun st => st.2.val ≤ k.val ∧ vecVals st.1 = List.replicate st.2.val [])
    (post := fun r => vecVals r = emptyColumnsRef k.val)
  · rintro ⟨cols, c⟩ ⟨hc', hcols'⟩
    dsimp only
    unfold kern.gemm.empty_columns_loop.body
    have hcols'' : vecVals cols = List.replicate c.val [] := hcols'
    by_cases hlt : c < k
    · rw [if_pos hlt]
      have hlt'' : c.val < k.val := by simpa [UScalar.lt_equiv] using hlt
      have hk := usize_val_le_max' k
      have hclen : (↑cols : List (alloc.vec.Vec Std.I64)).length = c.val := by
        have := congrArg List.length hcols''
        simpa only [vecVals, List.length_map, List.length_replicate] using this
      step*
      · refine ⟨by omega, ?_, ?_⟩
        · simp only [vecVals] at hcols'' ⊢
          rw [cols1_post, List.map_append, hcols'', c1_post]
          simp only [List.map_cons, List.map_nil, withCapacity_val]
          rw [replicate_append_one]
        · rw [c1_post]
          omega
    · rw [if_neg hlt]
      step*
      have hlt' : ¬ (c.val < k.val) := by simpa [UScalar.lt_equiv] using hlt
      have hge : k.val ≤ c.val := Nat.le_of_not_lt hlt'
      have hcv : c.val ≤ k.val := hc'
      have hck : c.val = k.val := le_antisymm hcv hge
      unfold emptyColumnsRef
      rw [← hck]
      exact hcols''
  · exact ⟨hc, hcols⟩

theorem word_empty_columns_ok (k cap : Std.Usize) :
    kern.gemm.empty_columns k cap ⦃ r => vecVals r = emptyColumnsRef k.val ⦄ := by
  unfold kern.gemm.empty_columns
  apply empty_columns_loop_spec
  · simp
  · simp [vecVals, withCapacity_val]

/-- Non-vacuity of the reference: two columns, both empty. -/
example : emptyColumnsRef 2 = [[], []] := by decide

/-! ## `split_part` -/

/-- One column's rows `start .. start + n_out` cut from the flattened column. -/
def splitPartRef (start n_out : Nat) (v : alloc.vec.Vec Std.I64) : List Std.I64 :=
  v.val.slice start (start + n_out)

private lemma take_succ_eq {α : Type _} (l : List α) (n : Nat) (h : n < l.length) :
    l.take (n + 1) = l.take n ++ [l[n]] := by
  induction l generalizing n with
  | nil => simp at h
  | cons a t ih =>
      cases n with
      | zero => rfl
      | succ m =>
          have hm : m < t.length := by simpa using h
          show a :: t.take (m + 1) = a :: (t.take m ++ [t[m]])
          rw [ih m hm]

private lemma map_take_succ {α β : Type _} (l : List α) (f : α → β) (n : Nat) (h : n < l.length) :
    (l.take (n + 1)).map f = (l.take n).map f ++ [f l[n]] := by
  rw [take_succ_eq l n h, List.map_append, List.map_cons, List.map_nil]

/-- `usize` is positive (the model's `Usize.max` is a large literal). -/
private lemma usize_max_pos : (0 : Nat) < Usize.max := by
  have h : (1#usize).val ≤ Usize.max := usize_val_le_max' 1#usize
  have h1 : (1#usize).val = 1 := by scalar_tac
  omega

theorem split_part_loop_spec (flat : Slice (alloc.vec.Vec Std.I64)) (start n_out : Std.Usize)
    (c : Std.Usize) (cols : alloc.vec.Vec (alloc.vec.Vec Std.I64))
    (hno : start.val + n_out.val ≤ Usize.max)
    (hcap : flat.val.length < Usize.max)
    (hlen : ∀ (i : Nat) (h : i < flat.val.length),
      start.val + n_out.val ≤ (flat.val[i]'h).val.length)
    (hc : c.val ≤ flat.val.length)
    (hcols : vecVals cols = (flat.val.take c.val).map (splitPartRef start.val n_out.val)) :
    kern.gemm.split_part_loop flat start n_out cols c ⦃ r =>
      vecVals r = flat.val.map (splitPartRef start.val n_out.val) ⦄ := by
  unfold kern.gemm.split_part_loop
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec (alloc.vec.Vec Std.I64) × Std.Usize =>
      flat.val.length - st.2.val)
    (inv := fun st => st.2.val ≤ flat.val.length ∧
      vecVals st.1 = (flat.val.take st.2.val).map (splitPartRef start.val n_out.val))
    (post := fun r => vecVals r = flat.val.map (splitPartRef start.val n_out.val))
  · rintro ⟨cols, c⟩ ⟨hc', hcols'⟩
    dsimp only
    unfold kern.gemm.split_part_loop.body
    have hcols'' : vecVals cols = (flat.val.take c.val).map (splitPartRef start.val n_out.val) :=
      hcols'
    simp only [vecVals] at hcols''
    by_cases hlt : c < Slice.len flat
    · rw [if_pos hlt]
      have hltc : c.val < flat.val.length := by
        have h := hlt
        rw [UScalar.lt_equiv, Slice.len_val] at h
        exact h
      step*
      · -- push precondition: `cols.val.length < Usize.max`
        have hclen : (↑cols : List (alloc.vec.Vec Std.I64)).length = c.val := by
          have h := congrArg List.length hcols''
          simp only [List.length_map, List.length_take] at h
          rw [Nat.min_eq_left hltc.le] at h
          exact h
        omega
      · refine ⟨?_, ?_, ?_⟩
        · rw [c1_post]; omega
        · have hv1 : (↑v1 : List Std.I64) = ↑s := by rw [alloc.vec.Vec.val, ← s_post]
          have hv : (↑v1 : List Std.I64)
              = List.slice start.val (start.val + n_out.val) (↑v) := by
            have hsrange : (↑s : List Std.I64)
                = List.slice start.val i1.val (↑v) := by assumption
            rw [hv1, hsrange, i1_post]
          simp only [vecVals]
          rw [cols1_post, c1_post]
          simp only [List.map_append, List.map_cons, List.map_nil]
          rw [hcols'']
          rw [map_take_succ ↑flat (splitPartRef start.val n_out.val) c.val hltc]
          simp only [splitPartRef]
          rw [← v_post, hv]
        · rw [c1_post]; omega
    · rw [if_neg hlt]
      step*
      have hltc : ¬ ((↑c : Nat) < flat.val.length) := by
        intro h
        exact hlt (by rw [UScalar.lt_equiv, Slice.len_val]; exact h)
      have hck : (↑c : Nat) = flat.val.length := le_antisymm hc' (Nat.le_of_not_lt hltc)
      simp only [vecVals]
      rw [hcols'', hck, List.take_length]
  · exact ⟨hc, hcols⟩

theorem word_split_part_ok (flat : Slice (alloc.vec.Vec Std.I64)) (start n_out : Std.Usize)
    (hno : start.val + n_out.val ≤ Usize.max)
    (hcap : flat.val.length < Usize.max)
    (hlen : ∀ (i : Nat) (h : i < flat.val.length),
      start.val + n_out.val ≤ (flat.val[i]'h).val.length) :
    kern.gemm.split_part flat start n_out ⦃ r =>
      vecVals r = flat.val.map (splitPartRef start.val n_out.val) ⦄ := by
  unfold kern.gemm.split_part
  apply split_part_loop_spec
  · exact hno
  · exact hcap
  · exact hlen
  · simp
  · simp [vecVals, withCapacity_val]

/-- Non-vacuity of the reference: rows `1..3` of `[1, 2, 3, 4]` are `[2, 3]`. -/
example : (List.slice 1 3 [1, 2, 3, 4] : List Nat) = [2, 3] := by decide

/-- Non-vacuity of the joint preconditions at `flat = Slice.new`, `start = n_out = 0`. -/
example : ∃ (flat : Slice (alloc.vec.Vec Std.I64)) (start n_out : Std.Usize),
    start.val + n_out.val ≤ Usize.max ∧ flat.val.length < Usize.max ∧
    (∀ (i : Nat) (h : i < flat.val.length),
      start.val + n_out.val ≤ (flat.val[i]'h).val.length) :=
  ⟨Slice.new (alloc.vec.Vec Std.I64), 0#usize, 0#usize, by simp,
    by simp [Slice.new_val, usize_max_pos],
    by intro i h; simp [Slice.new_val] at h⟩

/-! ## `part_row_starts` -/

/-- Running partial sums with an accumulator: `rowStartsAux acc ns` is `acc` followed by the
running sum after each entry of `ns`. -/
private def rowStartsAux : Nat → List Nat → List Nat
  | acc, [] => [acc]
  | acc, n :: rest => acc :: rowStartsAux (acc + n) rest

/-- The flattened row starts of a part list with output sizes `ns`: entry `q` is the sum of the
first `q` sizes, so entry `q` is the first row index of part `q`. -/
def rowStarts (ns : List Nat) : List Nat := rowStartsAux 0 ns

/-- The output sizes of a part slice. -/
def nsOf (parts : Slice kern.gemm.GemmPart) : List Nat := parts.val.map (fun gp => gp.n_out.val)

/-- The output sizes of the first `p` parts. -/
def nsTake (parts : Slice kern.gemm.GemmPart) (p : Std.Usize) : List Nat :=
  (parts.val.take p.val).map (fun gp => gp.n_out.val)

/-- The overflow scan of `part_row_starts`: the accumulator is the running row start; the scan
turns true at the first size whose running sum leaves `usize`, or whose running sum times the
column count `k` does. -/
private def prsScanAux (k : Nat) : Nat → List Nat → Bool
  | _, [] => false
  | acc, n :: rest =>
      decide (Usize.max < acc + n) || decide (Usize.max < (acc + n) * k) ||
        prsScanAux k (acc + n) rest

/-- `part_row_starts`'s overflow flag over a size list: the result is `none` exactly when this
flag is `true`. -/
def prsFlag (k : Nat) (ns : List Nat) : Bool := prsScanAux k 0 ns

/-- The value words of a `usize` vector, as naturals. -/
def vecUsizes (v : alloc.vec.Vec Std.Usize) : List Nat := v.val.map (fun x => x.val)

private lemma rowStartsAux_length (acc : Nat) (l : List Nat) :
    (rowStartsAux acc l).length = l.length + 1 := by
  induction l generalizing acc with
  | nil => rfl
  | cons n rest ih => simp [rowStartsAux, ih]

private lemma rowStartsAux_append_one (acc : Nat) (l : List Nat) (x : Nat) :
    rowStartsAux acc (l ++ [x]) = rowStartsAux acc l ++ [acc + l.sum + x] := by
  induction l generalizing acc with
  | nil => simp [rowStartsAux]
  | cons n rest ih =>
      show acc :: rowStartsAux (acc + n) (rest ++ [x])
        = (acc :: rowStartsAux (acc + n) rest) ++ [acc + (n + rest.sum) + x]
      rw [ih, List.cons_append]
      congr 1
      rw [List.append_right_inj, List.singleton_inj]
      omega

private lemma prsScanAux_append (k acc : Nat) (l ms : List Nat) :
    prsScanAux k acc (l ++ ms) = (prsScanAux k acc l || prsScanAux k (acc + l.sum) ms) := by
  induction l generalizing acc with
  | nil => simp [prsScanAux]
  | cons n rest ih =>
      show (decide (Usize.max < acc + n) || decide (Usize.max < (acc + n) * k) ||
              prsScanAux k (acc + n) (rest ++ ms))
          = ((decide (Usize.max < acc + n) || decide (Usize.max < (acc + n) * k) ||
              prsScanAux k (acc + n) rest) || prsScanAux k (acc + (n + rest.sum)) ms)
      rw [ih, Nat.add_assoc]
      simp only [Bool.or_assoc]

/-- Non-vacuity of the reference: the row starts of parts with 3 and 4 outputs. -/
example : rowStarts [3, 4] = [0, 3, 7] := by decide

/-- Non-vacuity of the reference at the empty part list: no rows, one start. -/
example : rowStarts [] = [0] := rfl

/-- Non-vacuity of the joint preconditions of the `part_row_starts` theorems at the empty part
slice: no part means no row start beyond `0`, so the overflow scan is `false`. -/
example : ∃ (parts : Slice kern.gemm.GemmPart) (k : Std.Usize),
    parts.val.length < Usize.max ∧ prsFlag k.val (nsOf parts) = false :=
  ⟨Slice.new kern.gemm.GemmPart, 0#usize,
    by simpa [Slice.new_val] using usize_max_pos,
    rfl⟩

/-! ## `part_row_starts` list scaffolding -/

/-- The output sizes of the first `n` parts, as naturals. -/
private def nsTakeNat (parts : Slice kern.gemm.GemmPart) (n : Nat) : List Nat :=
  (parts.val.take n).map (fun gp => gp.n_out.val)

private lemma nsTake_eq (parts : Slice kern.gemm.GemmPart) (p : Std.Usize) :
    nsTake parts p = nsTakeNat parts p.val := rfl

private lemma nsTakeNat_zero (parts : Slice kern.gemm.GemmPart) : nsTakeNat parts 0 = [] := by
  simp [nsTakeNat]

private lemma nsTakeNat_succ (parts : Slice kern.gemm.GemmPart) (n : Nat)
    (h : n < parts.val.length) :
    nsTakeNat parts (n + 1) = nsTakeNat parts n ++ [(parts.val[n]).n_out.val] := by
  simp only [nsTakeNat]
  rw [List.take_succ_eq_append_getElem h, List.map_append, List.map_cons, List.map_nil]

private lemma nsTakeNat_length (parts : Slice kern.gemm.GemmPart) (n : Nat)
    (h : n ≤ parts.val.length) : (nsTakeNat parts n).length = n := by
  simp only [nsTakeNat, List.length_map, List.length_take, Nat.min_eq_left h]

private lemma nsOf_eq_nsTakeNat (parts : Slice kern.gemm.GemmPart) :
    nsOf parts = nsTakeNat parts parts.val.length := by
  simp only [nsOf, nsTakeNat, List.take_length]

private lemma take_eq_of_length {α : Type} (l : List α) (n : Nat) (h : l.length = n) :
    l.take n = l := by rw [← h, List.take_length]

private lemma rowStartsAux_getElem? (acc : Nat) (l : List Nat) (q : Nat) :
    (rowStartsAux acc l)[q]? =
      (if q ≤ l.length then some (acc + (l.take q).sum) else none) := by
  induction l generalizing acc q with
  | nil =>
      cases q with
      | zero => simp [rowStartsAux]
      | succ k => simp [rowStartsAux]
  | cons n rest ih =>
      cases q with
      | zero => simp [rowStartsAux]
      | succ k =>
          show (rowStartsAux (acc + n) rest)[k]? =
            (if k + 1 ≤ (n :: rest).length then
              some (acc + ((n :: rest).take (k + 1)).sum) else none)
          rw [ih (acc + n) k]
          have hcond : (k + 1 ≤ (n :: rest).length) ↔ k ≤ rest.length := by
            rw [List.length_cons]
            omega
          by_cases hk : k ≤ rest.length
          · rw [if_pos hk, if_pos (hcond.mpr hk), List.take_succ_cons, List.sum_cons]
            congr 1
            omega
          · rw [if_neg hk, if_neg (fun h => hk (hcond.mp h))]

private lemma rowStarts_getElem? (ns : List Nat) (q : Nat) :
    (rowStarts ns)[q]? = (if q ≤ ns.length then some (ns.take q).sum else none) := by
  simpa only [rowStarts, Nat.zero_add] using rowStartsAux_getElem? 0 ns q

private lemma rowStarts_append_one (ns : List Nat) (n : Nat) :
    rowStarts (ns ++ [n]) = rowStarts ns ++ [ns.sum + n] := by
  show rowStartsAux 0 (ns ++ [n]) = rowStartsAux 0 ns ++ [ns.sum + n]
  rw [rowStartsAux_append_one, Nat.zero_add]

/-- Row starts are monotone in the size list is not needed: the scan becomes `true` at the first
size whose running sum leaves `usize`. -/
private lemma prsScanAux_snoc_true (k acc : Nat) (l : List Nat) (n : Nat)
    (h : prsScanAux k acc l = true) : prsScanAux k acc (l ++ [n]) = true := by
  rw [prsScanAux_append, h]
  simp

private lemma prsScanAux_snoc_of_lt (k acc : Nat) (l : List Nat) (n : Nat)
    (h0 : prsScanAux k acc l = false) (h : Usize.max < acc + l.sum + n) :
    prsScanAux k acc (l ++ [n]) = true := by
  rw [prsScanAux_append, h0, Bool.false_or]
  simp only [prsScanAux, Bool.or_false]
  have hA : decide (Usize.max < (acc + l.sum) + n) = true := by
    rw [decide_eq_true_eq]
    omega
  rw [hA]
  simp

/-- `usize`'s checked multiply is `none` exactly when the product leaves `usize`. -/
private lemma checked_mul_isNone (x k : Std.Usize) :
    (Usize.checked_mul x k).isNone = decide (Usize.max < x.val * k.val) := by
  have hs := Usize.checked_mul_bv_spec x k
  cases hc : Usize.checked_mul x k with
  | none =>
      rw [hc] at hs
      simp [hs]
  | some w =>
      rw [hc] at hs
      have hf : ¬ (Usize.max < x.val * k.val) := by omega
      simp [hf]

private lemma prsScanAux_snoc_eq_isNone (acc : Nat) (l : List Nat) (n : Nat) (k z : Std.Usize)
    (h0 : prsScanAux k.val acc l = false) (hz : z.val = acc + l.sum + n)
    (hle : z.val ≤ Usize.max) :
    prsScanAux k.val acc (l ++ [n]) = (Usize.checked_mul z k).isNone := by
  rw [prsScanAux_append, h0, Bool.false_or]
  simp only [prsScanAux, Bool.or_false]
  have hlt : ¬ (Usize.max < (acc + l.sum) + n) := by
    rw [← hz]
    omega
  have hdec : decide (Usize.max < (acc + l.sum) + n) = false := by simp [hlt]
  rw [hdec, Bool.false_or, checked_mul_isNone, hz]

theorem part_row_starts_loop_spec (parts : Slice kern.gemm.GemmPart) (k : Std.Usize)
    (hcap : parts.val.length < Usize.max)
    (rstarts : alloc.vec.Vec Std.Usize) (overflow : Bool) (p : Std.Usize)
    (hp : p.val ≤ parts.val.length)
    (hlen : rstarts.val.length = p.val + 1)
    (hov : overflow = prsFlag k.val (nsTake parts p))
    (hval : overflow = false → vecUsizes rstarts = rowStarts (nsTake parts p)) :
    kern.gemm.part_row_starts_loop parts k rstarts overflow p ⦃ r =>
      r.2 = prsFlag k.val (nsOf parts) ∧
      (r.2 = false → vecUsizes r.1 = rowStarts (nsOf parts)) ⦄ := by
  unfold kern.gemm.part_row_starts_loop
  apply loop.spec_decr_nat
    (measure := fun st : alloc.vec.Vec Std.Usize × Bool × Std.Usize =>
      parts.val.length - st.2.2.val)
    (inv := fun st : alloc.vec.Vec Std.Usize × Bool × Std.Usize =>
      st.2.2.val ≤ parts.val.length ∧ st.1.val.length = st.2.2.val + 1 ∧
      st.2.1 = prsFlag k.val (nsTake parts st.2.2) ∧
      (st.2.1 = false → vecUsizes st.1 = rowStarts (nsTake parts st.2.2)))
    (post := fun r : alloc.vec.Vec Std.Usize × Bool =>
      r.2 = prsFlag k.val (nsOf parts) ∧
      (r.2 = false → vecUsizes r.1 = rowStarts (nsOf parts)))
  · rintro ⟨rstarts, overflow, p⟩ ⟨hp', hlen', hov', hval'⟩
    dsimp only at *
    unfold kern.gemm.part_row_starts_loop.body
    by_cases hlt : p < Slice.len parts
    · rw [if_pos hlt]
      have hpLt : p.val < parts.val.length := by
        simpa [Slice.len_val, UScalar.lt_equiv] using hlt
      have hlenp : p.val < rstarts.val.length := by rw [hlen']; omega
      have h1v : (1#usize).val = 1 := by scalar_tac
      have hlt1 : p.val + 1 < Usize.max := by omega
      have hpush : rstarts.val.length < Usize.max := by rw [hlen']; exact hlt1
      have hadd : p.val + (1#usize).val ≤ Usize.max := by rw [h1v]; omega
      step*
      cases next with
      | none =>
        cases overflow with
        | false =>
          simp only [lift, Option.isNone_none, core.option.Option.unwrap_or, ↓reduceIte]
          step*
          have hvalF : vecUsizes rstarts = rowStarts (nsTake parts p) := hval' rfl
          have hsum : i1.val = (nsTakeNat parts p.val).sum := by
            have h1 : (vecUsizes rstarts)[p.val]? = some i1.val := by
              rw [vecUsizes, List.getElem?_map, List.getElem?_eq_getElem hlenp, ← i1_post]
              rfl
            have h2 : (rowStarts (nsTake parts p))[p.val]? = some (nsTakeNat parts p.val).sum := by
              have hl : (nsTakeNat parts p.val).length = p.val := nsTakeNat_length parts p.val hp'
              rw [rowStarts_getElem?, nsTake_eq, if_pos (by rw [hl]),
                take_eq_of_length (nsTakeNat parts p.val) p.val hl]
            rw [hvalF] at h1
            rw [h2] at h1
            exact (Option.some.inj h1).symm
          have hns : Usize.max < (nsTakeNat parts p.val).sum + (parts.val[p.val]).n_out.val := by
            have h := next_post
            rw [hsum, gp_post] at h
            exact h
          have hflag0 : prsScanAux k.val 0 (nsTakeNat parts p.val) = false := by
            have h := hov'
            rw [prsFlag, nsTake_eq] at h
            exact h.symm
          have htake : nsTake parts p1 = nsTakeNat parts (p.val + 1) := by
            rw [nsTake_eq, p1_post]
          refine ⟨?_, ?_, ?_, ?_, ?_⟩
          · rw [p1_post]; omega
          · rw [rstarts1_post, List.length_append, hlen', p1_post]; simp
          · rw [htake, prsFlag, nsTakeNat_succ parts p.val hpLt]
            exact (prsScanAux_snoc_of_lt k.val 0 (nsTakeNat parts p.val)
              (parts.val[p.val]).n_out.val hflag0 (by omega)).symm
          · intro h
            exact absurd h (by decide)
          · rw [p1_post]; omega
        | true =>
          simp only [lift, core.option.Option.unwrap_or, ↓reduceIte]
          step*
          have hflag0 : prsScanAux k.val 0 (nsTakeNat parts p.val) = true := by
            have h := hov'
            rw [prsFlag, nsTake_eq] at h
            exact h.symm
          have htake : nsTake parts p1 = nsTakeNat parts (p.val + 1) := by
            rw [nsTake_eq, p1_post]
          refine ⟨?_, ?_, ?_, ?_, ?_⟩
          · rw [p1_post]; omega
          · rw [rstarts1_post, List.length_append, hlen', p1_post]; simp
          · rw [htake, prsFlag, nsTakeNat_succ parts p.val hpLt]
            exact (prsScanAux_snoc_true k.val 0 (nsTakeNat parts p.val)
              (parts.val[p.val]).n_out.val hflag0).symm
          · intro h
            exact absurd h (by decide)
          · rw [p1_post]; omega
      | some z =>
        cases overflow with
        | false =>
          simp only [lift, Option.isNone_some, core.option.Option.and_then,
            core.option.Option.unwrap_or, ↓reduceIte, Bool.false_eq_true,
            kern.gemm.part_row_starts.closure.Insts.CoreOpsFunctionFnOnceTupleUsizeOptionUsize.call_once]
          step*
          have hvalF : vecUsizes rstarts = rowStarts (nsTake parts p) := hval' rfl
          have hsum : i1.val = (nsTakeNat parts p.val).sum := by
            have h1 : (vecUsizes rstarts)[p.val]? = some i1.val := by
              rw [vecUsizes, List.getElem?_map, List.getElem?_eq_getElem hlenp, ← i1_post]
              rfl
            have h2 : (rowStarts (nsTake parts p))[p.val]? = some (nsTakeNat parts p.val).sum := by
              have hl : (nsTakeNat parts p.val).length = p.val := nsTakeNat_length parts p.val hp'
              rw [rowStarts_getElem?, nsTake_eq, if_pos (by rw [hl]),
                take_eq_of_length (nsTakeNat parts p.val) p.val hl]
            rw [hvalF] at h1
            rw [h2] at h1
            exact (Option.some.inj h1).symm
          have hz' : z.val = (nsTakeNat parts p.val).sum + (parts.val[p.val]).n_out.val := by
            rw [next_post.2.1, hsum, gp_post]
          have hz : z.val = 0 + (nsTakeNat parts p.val).sum + (parts.val[p.val]).n_out.val := by
            rw [hz']
            omega
          have hle : z.val ≤ Usize.max := by
            rw [next_post.2.1]
            exact next_post.1
          have hflag0 : prsScanAux k.val 0 (nsTakeNat parts p.val) = false := by
            have h := hov'
            rw [prsFlag, nsTake_eq] at h
            exact h.symm
          have htake : nsTake parts p1 = nsTakeNat parts (p.val + 1) := by
            rw [nsTake_eq, p1_post]
          refine ⟨?_, ?_, ?_, ?_, ?_⟩
          · rw [p1_post]; omega
          · rw [rstarts1_post, List.length_append, hlen', p1_post]; simp
          · rw [htake, prsFlag, nsTakeNat_succ parts p.val hpLt]
            exact (prsScanAux_snoc_eq_isNone 0 (nsTakeNat parts p.val)
              (parts.val[p.val]).n_out.val k z hflag0 hz hle).symm
          · intro _
            rw [show vecUsizes rstarts1 = vecUsizes rstarts ++ [z.val] from by
              simp only [vecUsizes, rstarts1_post, List.map_append, List.map_cons, List.map_nil]]
            rw [hvalF, nsTake_eq, htake, nsTakeNat_succ parts p.val hpLt, rowStarts_append_one,
              List.append_right_inj, List.singleton_inj]
            exact hz'
          · rw [p1_post]; omega
        | true =>
          simp only [lift, core.option.Option.unwrap_or, ↓reduceIte]
          step*
          have hflag0 : prsScanAux k.val 0 (nsTakeNat parts p.val) = true := by
            have h := hov'
            rw [prsFlag, nsTake_eq] at h
            exact h.symm
          have htake : nsTake parts p1 = nsTakeNat parts (p.val + 1) := by
            rw [nsTake_eq, p1_post]
          refine ⟨?_, ?_, ?_, ?_, ?_⟩
          · rw [p1_post]; omega
          · rw [rstarts1_post, List.length_append, hlen', p1_post]; simp
          · rw [htake, prsFlag, nsTakeNat_succ parts p.val hpLt]
            exact (prsScanAux_snoc_true k.val 0 (nsTakeNat parts p.val)
              (parts.val[p.val]).n_out.val hflag0).symm
          · intro h
            exact absurd h (by decide)
          · rw [p1_post]; omega
    · rw [if_neg hlt]
      step*
      have hpk : p.val = parts.val.length := by
        have h : ¬ (p.val < parts.val.length) := by
          simpa [Slice.len_val, UScalar.lt_equiv] using hlt
        omega
      have htake : nsTake parts p = nsOf parts := by
        rw [nsTake_eq, nsOf_eq_nsTakeNat, hpk]
      refine ⟨?_, ?_⟩
      · rw [hov', htake]
      · intro hf
        rw [hval' hf, htake]
  · exact ⟨hp, hlen, hov, hval⟩

/-- The trailing branch of `part_row_starts` on the success case: a `false` overflow flag yields
`some` of the accumulated row starts. Stated over the raw loop result so the pattern bind is
discharged by `obtain`, not by rewriting a `let`. -/
private theorem part_row_starts_finish_ok (parts : Slice kern.gemm.GemmPart) (k : Std.Usize)
    (hflag : prsFlag k.val (nsOf parts) = false) (r : alloc.vec.Vec Std.Usize × Bool)
    (hr : r.2 = prsFlag k.val (nsOf parts) ∧
      (r.2 = false → vecUsizes r.1 = rowStarts (nsOf parts))) :
    (let (rstarts2, overflow) := r; if overflow = true then ok none else ok (some rstarts2)) ⦃
      fun r => ∃ v, r = some v ∧ vecUsizes v = rowStarts (nsOf parts) ⦄ := by
  obtain ⟨rstarts2, overflow⟩ := r
  have ho' : overflow = prsFlag k.val (nsOf parts) := hr.1
  have hv' : overflow = false → vecUsizes rstarts2 = rowStarts (nsOf parts) := hr.2
  have hf : overflow = false := by rw [ho', hflag]
  simp only [hf]
  exact ⟨rstarts2, rfl, hv' hf⟩

/-- The trailing branch of `part_row_starts` on the refusal case: a `true` overflow flag yields
`none`. -/
private theorem part_row_starts_finish_none (parts : Slice kern.gemm.GemmPart) (k : Std.Usize)
    (hflag : prsFlag k.val (nsOf parts) = true) (r : alloc.vec.Vec Std.Usize × Bool)
    (hr : r.2 = prsFlag k.val (nsOf parts)) :
    (let (rstarts2, overflow) := r; if overflow = true then ok none else ok (some rstarts2)) ⦃
      fun r => r = none ⦄ := by
  obtain ⟨rstarts2, overflow⟩ := r
  have ho' : overflow = prsFlag k.val (nsOf parts) := hr
  have ht : overflow = true := by rw [ho', hflag]
  simp only [ht, ↓reduceIte, WP.spec_ok]

theorem word_part_row_starts_ok (parts : Slice kern.gemm.GemmPart) (k : Std.Usize)
    (hcap : parts.val.length < Usize.max)
    (hflag : prsFlag k.val (nsOf parts) = false) :
    kern.gemm.part_row_starts parts k ⦃ r =>
      ∃ v, r = some v ∧ vecUsizes v = rowStarts (nsOf parts) ⦄ := by
  unfold kern.gemm.part_row_starts
  have hmax : (0 : Nat) < Usize.max := usize_max_pos
  have h1v : (1#usize).val = 1 := by scalar_tac
  have hadd : (Slice.len parts).val + (1#usize).val ≤ Usize.max := by
    have h : (Slice.len parts).val < Usize.max := by simpa [Slice.len_val] using hcap
    rw [h1v]
    omega
  have hpush : ([] : List Std.Usize).length < Usize.max := by simpa using hmax
  step*
  · simpa [withCapacity_val] using hpush
  refine WP.spec_bind (part_row_starts_loop_spec parts k hcap rstarts1 false 0#usize ?_ ?_ ?_ ?_) ?_
  · simp
  · rw [rstarts1_post]; simp [withCapacity_val]
  · simp [nsTake, prsFlag, prsScanAux]
  · intro _
    have hv0 : (0#usize).val = 0 := by scalar_tac
    have hwcv : (↑(alloc.vec.Vec.with_capacity Std.Usize i1) : List Std.Usize) = [] := rfl
    have hval1 : rstarts1.val = [0#usize] := by
      have h2 : (↑rstarts1 : List Std.Usize) = [0#usize] := by
        rw [rstarts1_post, hwcv, List.nil_append]
      exact h2
    rw [vecUsizes, hval1]
    simp [hv0, nsTake, rowStarts, rowStartsAux]
  · intro r hr
    exact part_row_starts_finish_ok parts k hflag r hr

theorem word_part_row_starts_refuses (parts : Slice kern.gemm.GemmPart) (k : Std.Usize)
    (hcap : parts.val.length < Usize.max)
    (hflag : prsFlag k.val (nsOf parts) = true) :
    kern.gemm.part_row_starts parts k ⦃ r => r = none ⦄ := by
  unfold kern.gemm.part_row_starts
  have hmax : (0 : Nat) < Usize.max := usize_max_pos
  have h1v : (1#usize).val = 1 := by scalar_tac
  have hadd : (Slice.len parts).val + (1#usize).val ≤ Usize.max := by
    have h : (Slice.len parts).val < Usize.max := by simpa [Slice.len_val] using hcap
    rw [h1v]
    omega
  have hpush : ([] : List Std.Usize).length < Usize.max := by simpa using hmax
  step*
  · simpa [withCapacity_val] using hpush
  refine WP.spec_bind (part_row_starts_loop_spec parts k hcap rstarts1 false 0#usize ?_ ?_ ?_ ?_) ?_
  · simp
  · rw [rstarts1_post]; simp [withCapacity_val]
  · simp [nsTake, prsFlag, prsScanAux]
  · intro _
    have hv0 : (0#usize).val = 0 := by scalar_tac
    have hwcv : (↑(alloc.vec.Vec.with_capacity Std.Usize i1) : List Std.Usize) = [] := rfl
    have hval1 : rstarts1.val = [0#usize] := by
      have h2 : (↑rstarts1 : List Std.Usize) = [0#usize] := by
        rw [rstarts1_post, hwcv, List.nil_append]
      exact h2
    rw [vecUsizes, hval1]
    simp [hv0, nsTake, rowStarts, rowStartsAux]
  · intro r hr
    exact part_row_starts_finish_none parts k hflag r hr.1

/-- Non-vacuity of the refusal hypothesis: a running sum that leaves `usize` turns the scan on. -/
example : prsFlag 0 [Usize.max, 1] = true := by
  have h1 : ¬ (Usize.max < Usize.max) := Nat.lt_irrefl _
  have h2 : ¬ (Usize.max < 0) := Nat.not_lt_zero _
  have h3 : Usize.max < Usize.max + 1 := Nat.lt_succ_self _
  simp only [prsFlag, prsScanAux, Nat.zero_add, Nat.mul_zero]
  rw [decide_eq_false_iff_not.mpr h1, decide_eq_false_iff_not.mpr h2, decide_eq_true h3]
  simp

#print axioms apeiron_word.empty_columns_loop_spec
#print axioms apeiron_word.word_empty_columns_ok
#print axioms apeiron_word.split_part_loop_spec
#print axioms apeiron_word.word_split_part_ok
#print axioms apeiron_word.part_row_starts_loop_spec
#print axioms apeiron_word.word_part_row_starts_ok
#print axioms apeiron_word.word_part_row_starts_refuses

end apeiron_word
