import ApeironWord
import Apeiron.Linter.Notation
import Apeiron.Linter.Citation
import Substrate.Algebra.Neural.TrainingCollapse

/-!
# The extracted exact-integer Gram fold

`gram.gram_fold_i8` (`src/learn/gram.rs`) is the exact checked `i64` fold `g += XᵀX` over a batch of
`i8` rows. This module refines the extracted body against its closed form, the integer sum
`g[i,j] + Σ_r x[r,i]·x[r,j]`. The link from that closed form to the canonical
`Sanguine.TrainingCollapse.gram_stack` (two folds equal the fold of the stacked batch) is not yet
stated here.

The three extracted loops are `gram.rows_loop` (over rows) calling `gram.row_loop` (over features)
calling `gram.row_pair_loop` (over features). Each is proved against a pure `List ℤ` model by
`loop.spec_decr_nat` with an explicit invariant, following `Map/Executed*.lean`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace apeiron_gram

open apeiron_word

open Aeneas Aeneas.Std Result
open List
open Matrix

/-! ## 1. List and Aeneas index helpers -/

/-- Reading a list after one element is written. (Local name; `apeiron_word.getD_set` is the
array variant.) -/
theorem gram_getD_set {α : Type} (l : List α) (i j : ℕ) (v : α) (h : i < l.length) (d : α) :
    (l.set i v).getD j d = if i = j then v else l.getD j d := by
  by_cases heq : i = j
  · subst heq
    rw [List.getD_eq_getElem?_getD, List.getElem?_set_self h, Option.getD_some]
    simp
  · rw [List.getD_eq_getElem?_getD, List.getElem?_set, if_neg heq, ← List.getD_eq_getElem?_getD]
    simp [heq]

/-- `(l.map f).getD i 0 = f (l.getD i default)` for an in-range index. -/
theorem list_map_getD {α : Type} [Inhabited α] (f : α → ℤ) (l : List α) {i : ℕ}
    (hi : i < l.length) : (l.map f).getD i 0 = f (l.getD i default) := by
  rw [List.getD_eq_getElem _ _ (by rwa [List.length_map]), List.getElem_map,
    List.getD_eq_getElem _ _ hi]

/-- `Slice.index_usize` reads the backing list at the in-range index. -/
theorem slice_index_getD {α : Type} [Inhabited α] (x : Slice α) (k : Usize)
    (h : k.val < x.val.length) :
    Slice.index_usize x k = ok (x.val.getD k.val default) := by
  unfold Slice.index_usize
  rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

/-- `Vec.index` reads the backing list at the in-range index. -/
theorem vec_index_getD (g : alloc.vec.Vec I64) (k : Usize) (h : k.val < g.val.length) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice I64) g k
      = ok (g.val.getD k.val 0#i64) := by
  rw [alloc.vec.Vec.index_slice_index]
  unfold alloc.vec.Vec.index_usize
  rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]

/-- `Vec.index_mut` returns the old value and the setter. -/
theorem vec_index_mut_getD (g : alloc.vec.Vec I64) (k : Usize) (h : k.val < g.val.length) :
    alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice I64) g k
      = ok (g.val.getD k.val 0#i64, g.set k) := by
  rw [alloc.vec.Vec.index_mut_slice_index]
  unfold alloc.vec.Vec.index_mut_usize
  have hindex : alloc.vec.Vec.index_usize g k = ok (g.val.getD k.val 0#i64) := by
    unfold alloc.vec.Vec.index_usize
    rw [alloc.vec.Vec.getElem?_Nat_eq, List.getElem?_eq_getElem h, List.getD_eq_getElem _ _ h]
  rw [hindex]
  simp only [bind_tc_ok]

/-! ## 2. `i64` checked-arithmetic helpers -/

/-- The `i8 → i64` widening preserves the value. -/
theorem fromI8_val (z : I8) : (core.convert.num.FromI64I8.from z).val = z.val :=
  core.convert.num.FromI64I8.from_val_eq z

/-- A `checked_mul` that fits returns its mathematical product. -/
theorem checked_mul_some {a b : I64}
    (h : I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max) :
    ∃ z, I64.checked_mul a b = some z ∧ z.val = a.val * b.val := by
  have hspec : (match I64.checked_mul a b with
      | some z => I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max
          ∧ z.val = a.val * b.val ∧ z.bv = a.bv * b.bv
      | none => ¬ (I64.min ≤ a.val * b.val ∧ a.val * b.val ≤ I64.max)) :=
    I64.checked_mul_bv_spec a b
  cases hm : I64.checked_mul a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- A `checked_add` that fits returns its mathematical sum. -/
theorem checked_add_some {a b : I64}
    (h : I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max) :
    ∃ z, I64.checked_add a b = some z ∧ z.val = a.val + b.val := by
  have hspec : (match I64.checked_add a b with
      | some z => I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max
          ∧ z.val = a.val + b.val ∧ z.bv = a.bv + b.bv
      | none => ¬ (I64.min ≤ a.val + b.val ∧ a.val + b.val ≤ I64.max)) :=
    I64.checked_add_bv_spec a b
  cases hm : I64.checked_add a b with
  | none => rw [hm] at hspec; exact absurd h hspec
  | some z => rw [hm] at hspec; exact ⟨z, rfl, hspec.2.2.1⟩

/-- The magnitude of an `i8` value is at most `128`. -/
theorem i8_abs_le (z : I8) : |z.val| ≤ 128 := by
  have h := z.hBounds
  have hbits : IScalarTy.I8.numBits = 8 := by decide
  rw [hbits] at h
  norm_num at h
  rw [abs_le]
  omega

/-- The **value list** of a word vector: each entry's `.val`. -/
def gvals (g : alloc.vec.Vec I64) : List ℤ := g.val.map (fun z => z.val)

/-- The **value list** of an `i8` slice: each entry's `.val`. -/
def xvals (x : Slice I8) : List ℤ := x.val.map (fun z => z.val)

/-- Every entry of a value list lies within `c` in absolute value. -/
def boundAt (g : List ℤ) (c : ℤ) : Prop := ∀ k, k < g.length → |g.getD k 0| ≤ c

theorem gvals_length (g : alloc.vec.Vec I64) : (gvals g).length = g.val.length := by
  simp [gvals]

theorem xvals_getD (x : Slice I8) {k : ℕ} (h : k < x.val.length) :
    (xvals x).getD k 0 = (x.val.getD k 0#i8).val := by
  unfold xvals
  exact list_map_getD (fun z => z.val) x.val h

theorem gvals_getD (g : alloc.vec.Vec I64) {k : ℕ} (h : k < g.val.length) :
    (gvals g).getD k 0 = (g.val.getD k 0#i64).val := by
  unfold gvals
  have hlen : k < (g.val.map (fun z => z.val)).length := by simpa using h
  rw [List.getD_eq_getElem _ _ hlen, List.getElem_map, List.getD_eq_getElem _ _ h]

theorem gvals_set (g : alloc.vec.Vec I64) (k : Usize) (v : I64) :
    gvals (g.set k v) = (gvals g).set k.val v.val := by
  simp only [gvals, alloc.vec.Vec.set_val_eq, List.map_set]

/-! ## 3. The pure value models of the three loops -/

/-- The pure value model of `row_pair_loop`: fold `j` from `0`, writing
`g[iw+j] := g[iw+j] + a * xv[rd+j]`. -/
def pairVals (g xv : List ℤ) (iw rd : ℕ) (a : ℤ) : ℕ → List ℤ
  | 0 => g
  | (j + 1) =>
      (pairVals g xv iw rd a j).set (iw + j)
        ((pairVals g xv iw rd a j).getD (iw + j) 0 + a * xv.getD (rd + j) 0)

theorem pairVals_step (g xv : List ℤ) (iw rd : ℕ) (a : ℤ) (j : ℕ) :
    pairVals g xv iw rd a (j + 1) =
      (pairVals g xv iw rd a j).set (iw + j)
        ((pairVals g xv iw rd a j).getD (iw + j) 0 + a * xv.getD (rd + j) 0) := rfl

/-- The pure value model of `row_loop`: fold `i` from `0`, calling `pairVals` at write base
`i * dd` with `a = xv[rd+i]`. -/
def rowVals (g xv : List ℤ) (rd dd : ℕ) : ℕ → List ℤ
  | 0 => g
  | (i + 1) =>
      pairVals (rowVals g xv rd dd i) xv (i * dd) rd (xv.getD (rd + i) 0) dd

/-- The pure value model of `rows_loop`: fold `r` from `0`, calling `rowVals` with row base
`r * dd`. -/
def rowsVals (g xv : List ℤ) (dd : ℕ) : ℕ → List ℤ
  | 0 => g
  | (r + 1) => rowVals (rowsVals g xv dd r) xv (r * dd) dd dd

theorem pairVals_length (g xv : List ℤ) (iw rd : ℕ) (a : ℤ) (n : ℕ) :
    (pairVals g xv iw rd a n).length = g.length := by
  induction n with
  | zero => rfl
  | succ j ih => rw [pairVals, List.length_set, ih]

/-- After `n` steps of `pairVals`, an index `iw + j` with `n ≤ j` is untouched. -/
theorem pairVals_getD_ge (g xv : List ℤ) (iw rd : ℕ) (a : ℤ) :
    ∀ n, iw + n ≤ g.length → ∀ j, n ≤ j →
      (pairVals g xv iw rd a n).getD (iw + j) 0 = g.getD (iw + j) 0 := by
  intro n
  induction n with
  | zero => intro _ j _; rfl
  | succ k ih =>
    intro hk j hkj
    have hkg : iw + k ≤ g.length := by omega
    have hklt : iw + k < g.length := by omega
    have hne : iw + k ≠ iw + j := by omega
    rw [pairVals, gram_getD_set _ _ _ _ (by rw [pairVals_length]; exact hklt) 0, if_neg hne]
    exact ih hkg j (by omega)

/-- After `n` steps of `pairVals`, the index `iw + j` with `j < n` carries its added term. -/
theorem pairVals_getD_lt (g xv : List ℤ) (iw rd : ℕ) (a : ℤ) :
    ∀ n, iw + n ≤ g.length → ∀ j, j < n →
      (pairVals g xv iw rd a n).getD (iw + j) 0
        = g.getD (iw + j) 0 + a * xv.getD (rd + j) 0 := by
  intro n
  induction n with
  | zero => intro _ j hj; omega
  | succ k ih =>
    intro hk j hj
    have hkg : iw + k ≤ g.length := by omega
    have hklt : iw + k < g.length := by omega
    rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hjlt | hjeq
    · have hne : iw + k ≠ iw + j := by omega
      rw [pairVals, gram_getD_set _ _ _ _ (by rw [pairVals_length]; exact hklt) 0, if_neg hne]
      exact ih hkg j hjlt
    · rw [hjeq]
      rw [pairVals, gram_getD_set _ _ _ _ (by rw [pairVals_length]; exact hklt) 0, if_pos rfl]
      rw [pairVals_getD_ge g xv iw rd a k hkg k (le_refl k)]
theorem pairVals_getD_untouched (g xv : List ℤ) (iw rd : ℕ) (a : ℤ) :
    ∀ n, iw + n ≤ g.length → ∀ k, (∀ j, j < n → k ≠ iw + j) →
      (pairVals g xv iw rd a n).getD k 0 = g.getD k 0 := by
  intro n
  induction n with
  | zero => intro _ k _; rfl
  | succ j ih =>
    intro hn k hnot
    have hjlt : iw + j < g.length := by omega
    have hne : iw + j ≠ k := fun h => hnot j (by omega) h.symm
    rw [pairVals, gram_getD_set _ _ _ _ (by rw [pairVals_length]; exact hjlt) 0, if_neg hne]
    exact ih (by omega) k (fun j' hj' => hnot j' (by omega))

/-- Two indices in distinct length-`dd` blocks are distinct (`b < dd` is required for the left one). -/
theorem block_ne {dd a b c d : ℕ} (h : a < c) (hb : b < dd) :
    a * dd + b ≠ c * dd + d := by
  have h1 : (a + 1) * dd ≤ c * dd := Nat.mul_le_mul_right dd (Nat.succ_le_of_lt h)
  have h2 : a * dd + b < (a + 1) * dd := by
    rw [Nat.succ_mul]; exact Nat.add_lt_add_left hb (a * dd)
  have h3 : a * dd + b < c * dd := lt_of_lt_of_le h2 h1
  have h4 : c * dd ≤ c * dd + d := Nat.le_add_right _ _
  omega

theorem rowVals_length (g xv : List ℤ) (rd dd : ℕ) (n : ℕ) :
    (rowVals g xv rd dd n).length = g.length := by
  induction n with
  | zero => rfl
  | succ i ih => rw [rowVals, pairVals_length, ih]

theorem rowsVals_length (g xv : List ℤ) (dd : ℕ) (n : ℕ) :
    (rowsVals g xv dd n).length = g.length := by
  induction n with
  | zero => rfl
  | succ r ih => rw [rowsVals, rowVals_length, ih]

/-- After `n` steps of `rowVals`, a feature block `i` with `n ≤ i` is untouched. -/
theorem rowVals_getD_ge (g xv : List ℤ) (rd dd : ℕ) :
    ∀ n, n * dd ≤ g.length → ∀ i j, n ≤ i → j < dd →
      (rowVals g xv rd dd n).getD (i * dd + j) 0 = g.getD (i * dd + j) 0 := by
  intro n
  induction n with
  | zero => intro _ i j _ _; rfl
  | succ k ih =>
    intro hn i j hki hj
    have hk : k * dd + dd ≤ g.length := by have := hn; rw [Nat.succ_mul] at this; omega
    have hki' : k < i := by omega
    rw [rowVals]
    rw [pairVals_getD_untouched (rowVals g xv rd dd k) xv (k * dd) rd (xv.getD (rd + k) 0) dd
          (by rw [rowVals_length]; omega) (i * dd + j)
          (fun j' hj' => (block_ne (a := k) (b := j') (c := i) (d := j) hki' hj').symm)]
    exact ih (by omega) i j (by omega) hj

/-- After `n` steps of `rowVals`, feature block `i` with `i < n` carries row `rd`'s product. -/
theorem rowVals_getD_lt (g xv : List ℤ) (rd dd : ℕ) :
    ∀ n, n * dd ≤ g.length → ∀ i j, i < n → j < dd →
      (rowVals g xv rd dd n).getD (i * dd + j) 0
        = g.getD (i * dd + j) 0 + xv.getD (rd + i) 0 * xv.getD (rd + j) 0 := by
  intro n
  induction n with
  | zero => intro _ i j hi _; omega
  | succ k ih =>
    intro hn i j hi hj
    have hk : k * dd + dd ≤ g.length := by have := hn; rw [Nat.succ_mul] at this; omega
    rw [rowVals]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hi with hil | hieq
    · rw [pairVals_getD_untouched (rowVals g xv rd dd k) xv (k * dd) rd (xv.getD (rd + k) 0) dd
            (by rw [rowVals_length]; omega) (i * dd + j)
            (fun j' _ => block_ne (a := i) (b := j) (c := k) (d := j') hil hj)]
      exact ih (by omega) i j hil hj
    · rw [hieq]
      rw [pairVals_getD_lt (rowVals g xv rd dd k) xv (k * dd) rd (xv.getD (rd + k) 0) dd
            (by rw [rowVals_length]; omega) j hj]
      rw [rowVals_getD_ge g xv rd dd k (by omega) k j (le_refl k) hj]

/-- **The closed form of the row-fold model.** After `n` rows, entry `(i,j)` is its initial value
plus the sum of the per-row products. -/
theorem rowsVals_getD (g xv : List ℤ) (dd : ℕ) :
    ∀ n, dd * dd ≤ g.length → ∀ i j, i < dd → j < dd →
      (rowsVals g xv dd n).getD (i * dd + j) 0
        = g.getD (i * dd + j) 0
          + Finset.sum (Finset.range n)
              (fun r => xv.getD (r * dd + i) 0 * xv.getD (r * dd + j) 0) := by
  intro n
  induction n with
  | zero => intro _ i j _ _; simp [rowsVals]
  | succ k ih =>
    intro hd i j hi hj
    rw [rowsVals]
    rw [rowVals_getD_lt (rowsVals g xv dd k) xv (k * dd) dd dd
          (by rw [rowsVals_length]; exact hd) i j hi hj]
    rw [ih hd i j hi hj]
    rw [Finset.sum_range_succ]
    ring

theorem abs_mul_le_16384 {a b : ℤ} (ha : |a| ≤ 128) (hb : |b| ≤ 128) : |a * b| ≤ 16384 := by
  rw [abs_mul]
  have ha' : 0 ≤ |a| := abs_nonneg _
  have hb' : 0 ≤ |b| := abs_nonneg _
  nlinarith

theorem xvals_bound (x : Slice I8) (k : ℕ) : |(xvals x).getD k 0| ≤ 128 := by
  by_cases hk : k < x.val.length
  · rw [xvals_getD x hk]
    exact i8_abs_le _
  · have hlen : (xvals x).length = x.val.length := by simp [xvals]
    have hkg : (xvals x).length ≤ k := by omega
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_none hkg]
    dsimp
    decide

theorem rowVals_bound (g xv : List ℤ) (rd dd : ℕ) (C : ℤ)
    (hg : ∀ i < dd, ∀ j < dd, |g.getD (i * dd + j) 0| ≤ C)
    (hlen : dd * dd ≤ g.length)
    (hxv : ∀ k, |xv.getD k 0| ≤ 128) :
    ∀ i < dd, ∀ j < dd, |(rowVals g xv rd dd dd).getD (i * dd + j) 0| ≤ C + 16384 := by
  intro i hi j hj
  rw [rowVals_getD_lt g xv rd dd dd hlen i j hi hj]
  have h1 := abs_add_le (g.getD (i * dd + j) 0) (xv.getD (rd + i) 0 * xv.getD (rd + j) 0)
  have h2 := abs_mul_le_16384 (hxv (rd + i)) (hxv (rd + j))
  have h3 := hg i hi j hj
  omega

/-! ## 4. The `row_pair_loop` loop -/

/-- The loop invariant of `gram.row_pair_loop`. -/
def pairLoopInv (x : Slice I8) (g0 : alloc.vec.Vec I64) (dd rd i : Usize) (a : I64)
    (g : alloc.vec.Vec I64) (j : Usize) (ok1 : Bool) (err : Option gram.GramError) : Prop :=
  ok1 = true ∧ err = none ∧ g.val.length = g0.val.length ∧ j.val ≤ dd.val ∧
  gvals g = pairVals (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val j.val

theorem row_pair_loop_body_exec
    (x : Slice I8) (dd rd i : Usize) (a : I64)
    (g : alloc.vec.Vec I64) (j : Usize) (err : Option gram.GramError)
    (rdv i2v i4v idxv j1 : Usize)
    (v_cur prod sum : I64)
    (hj : j < dd)
    (hrd : rd * dd = ok rdv)
    (hi2 : rdv + j = ok i2v)
    (hi2lt : i2v.val < x.val.length)
    (hi4 : i * dd = ok i4v)
    (hidx : i4v + j = ok idxv)
    (_hidxlt : idxv.val < g.val.length)
    (hmul : a.checked_mul (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)) = some prod)
    (hvi : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice I64) g idxv = ok v_cur)
    (hadd : v_cur.checked_add prod = some sum)
    (hvimut : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice I64) g idxv = ok (v_cur, g.set idxv))
    (hj1 : j + 1#usize = ok j1) :
    gram.row_pair_loop_loop.body x dd rd i a g j true err =
      ok (ControlFlow.cont (g.set idxv sum, j1, true, err)) := by
  have hi3 : Slice.index_usize x i2v = ok (x.val.getD i2v.val 0#i8) := slice_index_getD x i2v hi2lt
  rw [gram.row_pair_loop_loop.body]
  rw [if_pos hj, if_pos rfl]
  simp only [hrd, hi2, hi3, lift, hi4, hidx, hmul, hvi, hadd, hvimut, hj1, bind_tc_ok]
  change (do
      let (g1, ok2, err1) ← ok (g.set idxv sum, true, err)
      ok (ControlFlow.cont (g1, j1, ok2, err1))) = _
  simp only [bind_tc_ok]

/-- One iteration of `gram.row_pair_loop` advances the pure model and preserves the invariant. -/
theorem row_pair_loop_body_step
    (x : Slice I8) (dd rd i : Usize) (a : I64)
    (g0 : alloc.vec.Vec I64) (g : alloc.vec.Vec I64) (j : Usize) (ok1 : Bool)
    (err : Option gram.GramError) (C : ℤ)
    (hjn : j.val < dd.val) (hi : i.val < dd.val)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hrdd : rd.val * dd.val + dd.val ≤ x.val.length)
    (ha : a.val = (x.val.getD (rd.val * dd.val + i.val) 0#i8).val)
    (hC0 : 0 ≤ C)
    (hC : C + 16384 ≤ I64.max)
    (hcur0 : |(gvals g0).getD (i.val * dd.val + j.val) 0| ≤ C)
    (hinv : pairLoopInv x g0 dd rd i a g j ok1 err) :
    WP.spec (gram.row_pair_loop_loop.body x dd rd i a g j ok1 err)
      (fun cf => ∃ g' j', cf = ControlFlow.cont (g', j', true, err) ∧
        j'.val = j.val + 1 ∧ pairLoopInv x g0 dd rd i a g' j' true err) := by
  obtain ⟨hok, herr, hglen, hjle, hgvals⟩ := hinv
  have hlen : g.val.length = dd.val * dd.val := by rw [hglen, hlen0]
  have hxmax : x.val.length ≤ Usize.max := Slice.property x
  have hg0max : g0.val.length ≤ Usize.max := Slice.property g0.slice
  have hdd_sq : dd.val * dd.val ≤ Usize.max := by rw [← hlen0]; exact hg0max
  have hi_mul : i.val * dd.val + dd.val ≤ dd.val * dd.val := by
    calc i.val * dd.val + dd.val = (i.val + 1) * dd.val := by rw [Nat.succ_mul]
      _ ≤ dd.val * dd.val := Nat.mul_le_mul_right dd.val (Nat.succ_le_of_lt hi)
  have hrd_max : rd.val * dd.val ≤ Usize.max := by omega
  have hi_max : i.val * dd.val ≤ Usize.max := by omega
  have hj_succ : j.val + 1 ≤ Usize.max := by
    have : j.val + 1 ≤ dd.val := Nat.succ_le_of_lt hjn
    have : dd.val ≤ dd.val * dd.val := by
      have hdd_pos : 1 ≤ dd.val := by omega
      calc dd.val = 1 * dd.val := by rw [one_mul]
        _ ≤ dd.val * dd.val := Nat.mul_le_mul_right dd.val hdd_pos
    omega
  obtain ⟨rdv, hrd, hrdv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := rd) (y := dd) hrd_max)
  obtain ⟨i2v, hi2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := rdv) (y := j) (by rw [hrdv]; omega))
  obtain ⟨i4v, hi4, hi4v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := i) (y := dd) hi_max)
  obtain ⟨idxv, hidx, hidxv⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i4v) (y := j) (by rw [hi4v]; omega))
  obtain ⟨j1, hj1, hj1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := j) (y := 1#usize) (by show j.val + (1#usize).val ≤ Usize.max; rw [show (1#usize).val = 1 from rfl]; exact hj_succ))
  have hj1_val : j1.val = j.val + 1 := by rw [hj1v]; rfl
  have hi2lt : i2v.val < x.val.length := by rw [hi2v, hrdv]; omega
  have hidxlt : idxv.val < g.val.length := by rw [hidxv, hi4v, hlen]; omega
  have halt : |a.val| ≤ 128 := by rw [ha]; exact i8_abs_le _
  have hblt : |(core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)).val| ≤ 128 := by
    rw [fromI8_val]; exact i8_abs_le _
  have hprod_abs : |a.val * (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)).val| ≤ 16384 :=
    abs_mul_le_16384 halt hblt
  have hprod_bounds : I64.min ≤ a.val * (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)).val ∧
      a.val * (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)).val ≤ I64.max := by
    rw [I64.min_eq, I64.max_eq]
    have := abs_le.mp hprod_abs
    omega
  obtain ⟨prod, hmul, hprodv⟩ := checked_mul_some hprod_bounds
  have hprod_val_abs : |prod.val| ≤ 16384 := by rw [hprodv]; exact hprod_abs
  have hcur_eq : (g.val.getD idxv.val 0#i64).val = (gvals g0).getD (i.val * dd.val + j.val) 0 := by
    have hidx_eq : idxv.val = i.val * dd.val + j.val := by omega
    rw [← gvals_getD g hidxlt, hidx_eq, hgvals]
    have hle_len : i.val * dd.val + j.val ≤ (gvals g0).length := by
      rw [gvals_length, hlen0]; omega
    exact pairVals_getD_ge (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val j.val hle_len j.val (le_refl j.val)
  have hcur_abs : |(g.val.getD idxv.val 0#i64).val| ≤ C := by
    rw [hcur_eq]; exact hcur0
  have hC' : C + 16384 ≤ 9223372036854775807 := by rw [I64.max_eq] at hC; exact hC
  have hsum_bounds : I64.min ≤ (g.val.getD idxv.val 0#i64).val + prod.val ∧
      (g.val.getD idxv.val 0#i64).val + prod.val ≤ I64.max := by
    rw [I64.min_eq, I64.max_eq]
    have hc := abs_le.mp hcur_abs
    have hp := abs_le.mp hprod_val_abs
    omega
  obtain ⟨sum, hadd, hsumv⟩ := checked_add_some hsum_bounds
  have hvi : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice I64) g idxv
      = ok (g.val.getD idxv.val 0#i64) := vec_index_getD g idxv hidxlt
  have hvimut : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice I64) g idxv
      = ok (g.val.getD idxv.val 0#i64, g.set idxv) := vec_index_mut_getD g idxv hidxlt
  have hjProp : j < dd := (UScalar.lt_equiv j dd).2 hjn
  have hexec := row_pair_loop_body_exec x dd rd i a g j err rdv i2v i4v idxv j1
    (g.val.getD idxv.val 0#i64) prod sum
    hjProp hrd hi2 hi2lt hi4 hidx hidxlt hmul hvi hadd hvimut hj1
  have hsum_val : sum.val = (pairVals (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val j.val).getD (i.val * dd.val + j.val) 0 + a.val * (xvals x).getD (rd.val * dd.val + j.val) 0 := by
    rw [hsumv, hprodv, fromI8_val]
    have : (g.val.getD idxv.val 0#i64).val = (gvals g).getD (i.val * dd.val + j.val) 0 := by
      have : idxv.val = i.val * dd.val + j.val := by omega
      rw [← this]
      exact (gvals_getD g hidxlt).symm
    rw [this, hgvals]
    have : (x.val.getD i2v.val 0#i8).val = (xvals x).getD (rd.val * dd.val + j.val) 0 := by
      have : i2v.val = rd.val * dd.val + j.val := by omega
      rw [← this]
      exact (xvals_getD x hi2lt).symm
    rw [this]
  have hgvals' : gvals (g.set idxv sum) = pairVals (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val j1.val := by
    rw [hj1_val, pairVals_step, ← hgvals, gvals_set]
    have : idxv.val = i.val * dd.val + j.val := by omega
    rw [this, hsum_val, hgvals]
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (g.set idxv sum, j1, true, err), hexec, g.set idxv sum, j1, rfl, hj1_val, ?_⟩
  refine ⟨rfl, herr, ?_, ?_, hgvals'⟩
  · rw [alloc.vec.Vec.set_val_eq, List.length_set]; exact hglen
  · omega


/-- Full specification of `gram.row_pair_loop_loop`. -/
theorem row_pair_loop_loop_spec
    (x : Slice I8) (dd rd i : Usize) (a : I64)
    (g0 : alloc.vec.Vec I64) (g : alloc.vec.Vec I64) (j : Usize) (C : ℤ)
    (hi : i.val < dd.val)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hrdd : rd.val * dd.val + dd.val ≤ x.val.length)
    (ha : a.val = (x.val.getD (rd.val * dd.val + i.val) 0#i8).val)
    (hC0 : 0 ≤ C)
    (hC : C + 16384 ≤ I64.max)
    (hbound : ∀ j' < dd.val, |(gvals g0).getD (i.val * dd.val + j') 0| ≤ C)
    (hinv0 : pairLoopInv x g0 dd rd i a g j true none) :
    WP.spec (gram.row_pair_loop_loop g x dd rd i a j true none)
      (fun (g', err') => err' = none ∧ g'.val.length = g0.val.length ∧
        gvals g' = pairVals (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val dd.val) := by
  unfold gram.row_pair_loop_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option gram.GramError => dd.val - s.2.1.val)
    (inv := fun s => pairLoopInv x g0 dd rd i a s.1 s.2.1 s.2.2.1 s.2.2.2)
    (post := fun b : alloc.vec.Vec I64 × Option gram.GramError =>
      b.2 = none ∧ b.1.val.length = g0.val.length ∧
      gvals b.1 = pairVals (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val dd.val)
  · rintro ⟨g', j', ok', err'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', herr', hlen', hj'_le, hgvals'⟩ := hinv
    rw [hok', herr']
    by_cases hjlt : j'.val < dd.val
    · have hcur0 : |(gvals g0).getD (i.val * dd.val + j'.val) 0| ≤ C := hbound j'.val hjlt
      obtain ⟨cf, hstep, g'', j'', hcf, hj''_val, hinv''⟩ := WP.spec_imp_exists
        (row_pair_loop_body_step x dd rd i a g0 g' j' true none C hjlt hi hlen0 hrdd ha hC0 hC hcur0
          ⟨rfl, rfl, hlen', hj'_le, hgvals'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (j' < dd) := fun hc => hjlt ((UScalar.lt_equiv j' dd).1 hc)
      have hbody : gram.row_pair_loop_loop.body x dd rd i a g' j' true none =
          ok (ControlFlow.done (g', none)) := by
        rw [gram.row_pair_loop_loop.body, if_neg hnot]
      simp only [hbody]
      have hj'_eq : j'.val = dd.val := by omega
      rw [hj'_eq] at hgvals'
      exact ⟨rfl, hlen', hgvals'⟩
  · exact hinv0

/-- Specification of `gram.row_pair_loop`. -/
theorem row_pair_loop_spec
    (x : Slice I8) (dd rd i : Usize) (a : I64)
    (g0 : alloc.vec.Vec I64) (C : ℤ)
    (hi : i.val < dd.val)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hrdd : rd.val * dd.val + dd.val ≤ x.val.length)
    (ha : a.val = (x.val.getD (rd.val * dd.val + i.val) 0#i8).val)
    (hC0 : 0 ≤ C)
    (hC : C + 16384 ≤ I64.max)
    (hbound : ∀ j' < dd.val, |(gvals g0).getD (i.val * dd.val + j') 0| ≤ C) :
    WP.spec (gram.row_pair_loop g0 x dd rd i a)
      (fun (g', err') => err' = none ∧ g'.val.length = g0.val.length ∧
        gvals g' = pairVals (gvals g0) (xvals x) (i.val * dd.val) (rd.val * dd.val) a.val dd.val) := by
  unfold gram.row_pair_loop
  have hinv0 : pairLoopInv x g0 dd rd i a g0 0#usize true none := by
    refine ⟨rfl, rfl, rfl, by show (0#usize).val ≤ dd.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    simp [pairVals]
  exact row_pair_loop_loop_spec x dd rd i a g0 g0 0#usize C hi hlen0 hrdd ha hC0 hC hbound hinv0

/-! ## 5. The `row_loop` loop -/

theorem row_loop_body_exec
    (x : Slice I8) (dd rd : Usize)
    (g : alloc.vec.Vec I64) (i : Usize) (err : Option gram.GramError)
    (rdv i2v i4 : Usize) (g1 : alloc.vec.Vec I64)
    (hi : i < dd)
    (hrd : rd * dd = ok rdv)
    (hi2 : rdv + i = ok i2v)
    (hi2lt : i2v.val < x.val.length)
    (hpair : gram.row_pair_loop g x dd rd i (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)) = ok (g1, none))
    (hi4 : i + 1#usize = ok i4) :
    gram.row_loop_loop.body x dd rd g i true err =
      ok (ControlFlow.cont (g1, i4, true, err)) := by
  have hi3 : Slice.index_usize x i2v = ok (x.val.getD i2v.val 0#i8) := slice_index_getD x i2v hi2lt
  rw [gram.row_loop_loop.body]
  rw [if_pos hi, if_pos rfl]
  simp only [hrd, hi2, hi3, lift, hpair, hi4, bind_tc_ok]
  simp

/-- The loop invariant of `gram.row_loop`. -/
def rowLoopInv (x : Slice I8) (g0 : alloc.vec.Vec I64) (dd rd : Usize)
    (g : alloc.vec.Vec I64) (i : Usize) (ok1 : Bool) (err : Option gram.GramError) : Prop :=
  ok1 = true ∧ err = none ∧ g.val.length = g0.val.length ∧ i.val ≤ dd.val ∧
  gvals g = rowVals (gvals g0) (xvals x) (rd.val * dd.val) dd.val i.val

/-- One iteration of `gram.row_loop` advances the pure model and preserves the invariant. -/
theorem row_loop_body_step
    (x : Slice I8) (dd rd : Usize)
    (g0 : alloc.vec.Vec I64) (g : alloc.vec.Vec I64) (i : Usize) (ok1 : Bool)
    (err : Option gram.GramError) (C : ℤ)
    (hin : i.val < dd.val)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hrdd : rd.val * dd.val + dd.val ≤ x.val.length)
    (hC0 : 0 ≤ C)
    (hC : C + 16384 ≤ I64.max)
    (hbound : ∀ j' < dd.val, |(gvals g0).getD (i.val * dd.val + j') 0| ≤ C)
    (hinv : rowLoopInv x g0 dd rd g i ok1 err) :
    WP.spec (gram.row_loop_loop.body x dd rd g i ok1 err)
      (fun cf => ∃ g' i', cf = ControlFlow.cont (g', i', true, err) ∧
        i'.val = i.val + 1 ∧ rowLoopInv x g0 dd rd g' i' true err) := by
  obtain ⟨hok, herr, hglen, _hile, hgvals⟩ := hinv
  have hlen : g.val.length = dd.val * dd.val := by rw [hglen, hlen0]
  have hrd_max : rd.val * dd.val ≤ Usize.max := by
    have hxmax : x.val.length ≤ Usize.max := Slice.property x
    omega
  have hi_succ : i.val + 1 ≤ Usize.max := by
    have : i.val + 1 ≤ dd.val := Nat.succ_le_of_lt hin
    have : dd.val ≤ dd.val * dd.val := by
      have hdd_pos : 1 ≤ dd.val := by omega
      calc dd.val = 1 * dd.val := by rw [one_mul]
        _ ≤ dd.val * dd.val := Nat.mul_le_mul_right dd.val hdd_pos
    have hg0max : g0.val.length ≤ Usize.max := Slice.property g0.slice
    have : dd.val * dd.val ≤ Usize.max := by rw [← hlen0]; exact hg0max
    omega
  obtain ⟨rdv, hrd, hrdv⟩ := WP.spec_imp_exists (Usize.mul_spec (x := rd) (y := dd) hrd_max)
  obtain ⟨i2v, hi2, hi2v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := rdv) (y := i) (by rw [hrdv]; have hxmax := Slice.property x; omega))
  obtain ⟨i4, hi4, hi4v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := i) (y := 1#usize) (by show i.val + (1#usize).val ≤ Usize.max; rw [show (1#usize).val = 1 from rfl]; exact hi_succ))
  have hi4_val : i4.val = i.val + 1 := by rw [hi4v]; rfl
  have hi2lt : i2v.val < x.val.length := by rw [hi2v, hrdv]; omega
  have hbound_g : ∀ j' < dd.val, |(gvals g).getD (i.val * dd.val + j') 0| ≤ C := by
    intro j' hj'
    rw [hgvals]
    have hle_len : i.val * dd.val ≤ (gvals g0).length := by
      rw [gvals_length, hlen0]
      exact Nat.mul_le_mul_right dd.val (Nat.le_of_lt hin)
    rw [rowVals_getD_ge (gvals g0) (xvals x) (rd.val * dd.val) dd.val i.val hle_len i.val j' (le_refl i.val) hj']
    exact hbound j' hj'
  have ha_eq : (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8)).val =
      (x.val.getD (rd.val * dd.val + i.val) 0#i8).val := by
    rw [fromI8_val]
    have : i2v.val = rd.val * dd.val + i.val := by rw [hi2v, hrdv]
    rw [this]
  have hspec := row_pair_loop_spec x dd rd i (core.convert.num.FromI64I8.from (x.val.getD i2v.val 0#i8))
    g C hin hlen hrdd ha_eq hC0 hC hbound_g
  obtain ⟨⟨g1, e1⟩, hpair_exec, he1, hg1_len, hg1_vals⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst he1
  have hiProp : i < dd := (UScalar.lt_equiv i dd).2 hin
  have hexec := row_loop_body_exec x dd rd g i err rdv i2v i4 g1 hiProp hrd hi2 hi2lt hpair_exec hi4
  have hgvals' : gvals g1 = rowVals (gvals g0) (xvals x) (rd.val * dd.val) dd.val i4.val := by
    rw [hi4_val, rowVals]
    rw [hg1_vals, hgvals]
    congr 1
    have hidx_eq : i2v.val = rd.val * dd.val + i.val := by rw [hi2v, hrdv]
    rw [fromI8_val, ← hidx_eq]
    exact (xvals_getD x hi2lt).symm
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (g1, i4, true, err), hexec, g1, i4, rfl, hi4_val, ?_⟩
  refine ⟨rfl, herr, ?_, ?_, hgvals'⟩
  · rw [hg1_len, hglen]
  · omega

/-- Full specification of `gram.row_loop_loop`. -/
theorem row_loop_loop_spec
    (x : Slice I8) (dd rd : Usize)
    (g0 : alloc.vec.Vec I64) (g : alloc.vec.Vec I64) (i : Usize) (C : ℤ)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hrdd : rd.val * dd.val + dd.val ≤ x.val.length)
    (hC0 : 0 ≤ C)
    (hC : C + 16384 ≤ I64.max)
    (hbound : ∀ i' < dd.val, ∀ j' < dd.val, |(gvals g0).getD (i' * dd.val + j') 0| ≤ C)
    (hinv0 : rowLoopInv x g0 dd rd g i true none) :
    WP.spec (gram.row_loop_loop g x dd rd i true none)
      (fun (g', err') => err' = none ∧ g'.val.length = g0.val.length ∧
        gvals g' = rowVals (gvals g0) (xvals x) (rd.val * dd.val) dd.val dd.val) := by
  unfold gram.row_loop_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option gram.GramError => dd.val - s.2.1.val)
    (inv := fun s => rowLoopInv x g0 dd rd s.1 s.2.1 s.2.2.1 s.2.2.2)
    (post := fun b : alloc.vec.Vec I64 × Option gram.GramError =>
      b.2 = none ∧ b.1.val.length = g0.val.length ∧
      gvals b.1 = rowVals (gvals g0) (xvals x) (rd.val * dd.val) dd.val dd.val)
  · rintro ⟨g', i', ok', err'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', herr', hlen', hi'_le, hgvals'⟩ := hinv
    rw [hok', herr']
    by_cases hilt : i'.val < dd.val
    · have hbound_i : ∀ j' < dd.val, |(gvals g0).getD (i'.val * dd.val + j') 0| ≤ C := fun j' hj' => hbound i'.val hilt j' hj'
      obtain ⟨cf, hstep, g'', i'', hcf, hi''_val, hinv''⟩ := WP.spec_imp_exists
        (row_loop_body_step x dd rd g0 g' i' true none C hilt hlen0 hrdd hC0 hC hbound_i
          ⟨rfl, rfl, hlen', hi'_le, hgvals'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (i' < dd) := fun hc => hilt ((UScalar.lt_equiv i' dd).1 hc)
      have hbody : gram.row_loop_loop.body x dd rd g' i' true none =
          ok (ControlFlow.done (g', none)) := by
        rw [gram.row_loop_loop.body, if_neg hnot]
      simp only [hbody]
      have hi'_eq : i'.val = dd.val := by omega
      rw [hi'_eq] at hgvals'
      exact ⟨rfl, hlen', hgvals'⟩
  · exact hinv0

/-- Specification of `gram.row_loop`. -/
theorem row_loop_spec
    (x : Slice I8) (dd rd : Usize) (g0 : alloc.vec.Vec I64) (C : ℤ)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hrdd : rd.val * dd.val + dd.val ≤ x.val.length)
    (hC0 : 0 ≤ C)
    (hC : C + 16384 ≤ I64.max)
    (hbound : ∀ i' < dd.val, ∀ j' < dd.val, |(gvals g0).getD (i' * dd.val + j') 0| ≤ C) :
    WP.spec (gram.row_loop g0 x dd rd)
      (fun (g', err') => err' = none ∧ g'.val.length = g0.val.length ∧
        gvals g' = rowVals (gvals g0) (xvals x) (rd.val * dd.val) dd.val dd.val) := by
  unfold gram.row_loop
  have hinv0 : rowLoopInv x g0 dd rd g0 0#usize true none := by
    refine ⟨rfl, rfl, rfl, by show (0#usize).val ≤ dd.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_⟩
    simp [rowVals]
  exact row_loop_loop_spec x dd rd g0 g0 0#usize C hlen0 hrdd hC0 hC hbound hinv0

/-! ## 6. The `rows_loop` loop -/

theorem rows_loop_body_exec
    (x : Slice I8) (dd rows : Usize)
    (g : alloc.vec.Vec I64) (r : Usize) (err : Option gram.GramError)
    (r1 : Usize) (g1 : alloc.vec.Vec I64)
    (hr : r < rows)
    (hrow : gram.row_loop g x dd r = ok (g1, none))
    (hr1 : r + 1#usize = ok r1) :
    gram.rows_loop_loop.body x dd rows g r true err =
      ok (ControlFlow.cont (g1, r1, true, err)) := by
  rw [gram.rows_loop_loop.body]
  rw [if_pos hr, if_pos rfl]
  simp only [hrow, hr1, bind_tc_ok]
  simp

/-- The loop invariant of `gram.rows_loop`. -/
def rowsLoopInv (x : Slice I8) (g0 : alloc.vec.Vec I64) (dd rows : Usize) (C0 : ℤ)
    (g : alloc.vec.Vec I64) (r : Usize) (ok1 : Bool) (err : Option gram.GramError) : Prop :=
  ok1 = true ∧ err = none ∧ g.val.length = g0.val.length ∧ r.val ≤ rows.val ∧
  gvals g = rowsVals (gvals g0) (xvals x) dd.val r.val ∧
  (∀ i < dd.val, ∀ j < dd.val, |(gvals g).getD (i * dd.val + j) 0| ≤ C0 + (r.val : ℤ) * 16384)

/-- One iteration of `gram.rows_loop` advances the pure model and preserves the invariant. -/
theorem rows_loop_body_step
    (x : Slice I8) (dd rows : Usize)
    (g0 : alloc.vec.Vec I64) (g : alloc.vec.Vec I64) (r : Usize) (ok1 : Bool)
    (err : Option gram.GramError) (C0 : ℤ)
    (hrn : r.val < rows.val)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hxlen : rows.val * dd.val ≤ x.val.length)
    (hC0 : 0 ≤ C0)
    (hC : C0 + (rows.val : ℤ) * 16384 ≤ I64.max)
    (hinv : rowsLoopInv x g0 dd rows C0 g r ok1 err) :
    WP.spec (gram.rows_loop_loop.body x dd rows g r ok1 err)
      (fun cf => ∃ g' r', cf = ControlFlow.cont (g', r', true, err) ∧
        r'.val = r.val + 1 ∧ rowsLoopInv x g0 dd rows C0 g' r' true err) := by
  obtain ⟨hok, herr, hglen, _hrle, hgvals, hbound_r⟩ := hinv
  have hlen : g.val.length = dd.val * dd.val := by rw [hglen, hlen0]
  have hr_succ : r.val + 1 ≤ Usize.max := by
    have hrows_max : rows.val ≤ Usize.max := by
      have h := UScalar.hrBounds rows
      rw [UScalar.rMax_eq_max, UScalar.max_USize_eq] at h
      exact h
    omega
  obtain ⟨r1, hr1, hr1v⟩ := WP.spec_imp_exists
    (Usize.add_spec (x := r) (y := 1#usize) (by show r.val + (1#usize).val ≤ Usize.max; rw [show (1#usize).val = 1 from rfl]; exact hr_succ))
  have hr1_val : r1.val = r.val + 1 := by rw [hr1v]; rfl
  have hCr0 : 0 ≤ C0 + (r.val : ℤ) * 16384 := by omega
  have hCr : C0 + (r.val : ℤ) * 16384 + 16384 ≤ I64.max := by
    have : (r.val : ℤ) + 1 ≤ (rows.val : ℤ) := by omega
    calc C0 + (r.val : ℤ) * 16384 + 16384 = C0 + ((r.val : ℤ) + 1) * 16384 := by ring
      _ ≤ C0 + (rows.val : ℤ) * 16384 := by nlinarith
      _ ≤ I64.max := hC
  have hrdd : r.val * dd.val + dd.val ≤ x.val.length := by
    calc r.val * dd.val + dd.val = (r.val + 1) * dd.val := by rw [Nat.succ_mul]
      _ ≤ rows.val * dd.val := Nat.mul_le_mul_right dd.val (Nat.succ_le_of_lt hrn)
      _ ≤ x.val.length := hxlen
  have hspec := row_loop_spec x dd r g (C0 + (r.val : ℤ) * 16384) hlen hrdd hCr0 hCr hbound_r
  obtain ⟨⟨g1, e1⟩, hrow_exec, he1, hg1_len, hg1_vals⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst he1
  have hrProp : r < rows := (UScalar.lt_equiv r rows).2 hrn
  have hexec := rows_loop_body_exec x dd rows g r err r1 g1 hrProp hrow_exec hr1
  have hgvals' : gvals g1 = rowsVals (gvals g0) (xvals x) dd.val r1.val := by
    rw [hr1_val, rowsVals]
    rw [hg1_vals, hgvals]
  have hlen_g : dd.val * dd.val ≤ (gvals g).length := by rw [gvals_length, hlen]
  have hbound_next := rowVals_bound (gvals g) (xvals x) (r.val * dd.val) dd.val (C0 + (r.val : ℤ) * 16384)
    hbound_r hlen_g (xvals_bound x)
  have hbound_r1 : ∀ i < dd.val, ∀ j < dd.val, |(gvals g1).getD (i * dd.val + j) 0| ≤ C0 + (r1.val : ℤ) * 16384 := by
    intro i hi j hj
    have := hbound_next i hi j hj
    rw [← hg1_vals] at this
    rw [hr1_val]
    push_cast
    linarith
  rw [hok]
  rw [WP.spec_equiv_exists]
  refine ⟨ControlFlow.cont (g1, r1, true, err), hexec, g1, r1, rfl, hr1_val, ?_⟩
  refine ⟨rfl, herr, ?_, ?_, hgvals', hbound_r1⟩
  · rw [hg1_len, hglen]
  · omega

/-- Full specification of `gram.rows_loop_loop`. -/
theorem rows_loop_loop_spec
    (x : Slice I8) (dd rows : Usize)
    (g0 : alloc.vec.Vec I64) (g : alloc.vec.Vec I64) (r : Usize) (C0 : ℤ)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hxlen : rows.val * dd.val ≤ x.val.length)
    (hC0 : 0 ≤ C0)
    (hC : C0 + (rows.val : ℤ) * 16384 ≤ I64.max)
    (hinv0 : rowsLoopInv x g0 dd rows C0 g r true none) :
    WP.spec (gram.rows_loop_loop g x dd rows r true none)
      (fun (g', err') => err' = none ∧ g'.val.length = g0.val.length ∧
        gvals g' = rowsVals (gvals g0) (xvals x) dd.val rows.val ∧
        (∀ i < dd.val, ∀ j < dd.val, |(gvals g').getD (i * dd.val + j) 0| ≤ C0 + (rows.val : ℤ) * 16384)) := by
  unfold gram.rows_loop_loop
  apply loop.spec_decr_nat
    (measure := fun s : alloc.vec.Vec I64 × Usize × Bool × Option gram.GramError => rows.val - s.2.1.val)
    (inv := fun s => rowsLoopInv x g0 dd rows C0 s.1 s.2.1 s.2.2.1 s.2.2.2)
    (post := fun b : alloc.vec.Vec I64 × Option gram.GramError =>
      b.2 = none ∧ b.1.val.length = g0.val.length ∧
      gvals b.1 = rowsVals (gvals g0) (xvals x) dd.val rows.val ∧
      (∀ i < dd.val, ∀ j < dd.val, |(gvals b.1).getD (i * dd.val + j) 0| ≤ C0 + (rows.val : ℤ) * 16384))
  · rintro ⟨g', r', ok', err'⟩ hinv
    dsimp only at hinv ⊢
    obtain ⟨hok', herr', hlen', hr'_le, hgvals', hbound'⟩ := hinv
    rw [hok', herr']
    by_cases hrlt : r'.val < rows.val
    · obtain ⟨cf, hstep, g'', r'', hcf, hr''_val, hinv''⟩ := WP.spec_imp_exists
        (rows_loop_body_step x dd rows g0 g' r' true none C0 hrlt hlen0 hxlen hC0 hC
          ⟨rfl, rfl, hlen', hr'_le, hgvals', hbound'⟩)
      rw [hcf] at hstep
      simp only [hstep]
      refine ⟨hinv'', ?_⟩
      dsimp only
      omega
    · have hnot : ¬ (r' < rows) := fun hc => hrlt ((UScalar.lt_equiv r' rows).1 hc)
      have hbody : gram.rows_loop_loop.body x dd rows g' r' true none =
          ok (ControlFlow.done (g', none)) := by
        rw [gram.rows_loop_loop.body, if_neg hnot]
      simp only [hbody]
      have hr'_eq : r'.val = rows.val := by omega
      rw [hr'_eq] at hgvals' hbound'
      exact ⟨rfl, hlen', hgvals', hbound'⟩
  · exact hinv0

/-- Specification of `gram.rows_loop`. -/
theorem rows_loop_spec
    (x : Slice I8) (dd rows : Usize) (g0 : alloc.vec.Vec I64) (C0 : ℤ)
    (hlen0 : g0.val.length = dd.val * dd.val)
    (hxlen : rows.val * dd.val ≤ x.val.length)
    (hC0 : 0 ≤ C0)
    (hC : C0 + (rows.val : ℤ) * 16384 ≤ I64.max)
    (hbound : ∀ i < dd.val, ∀ j < dd.val, |(gvals g0).getD (i * dd.val + j) 0| ≤ C0) :
    WP.spec (gram.rows_loop g0 x dd rows)
      (fun (g', err') => err' = none ∧ g'.val.length = g0.val.length ∧
        gvals g' = rowsVals (gvals g0) (xvals x) dd.val rows.val ∧
        (∀ i < dd.val, ∀ j < dd.val, |(gvals g').getD (i * dd.val + j) 0| ≤ C0 + (rows.val : ℤ) * 16384)) := by
  unfold gram.rows_loop
  have hinv0 : rowsLoopInv x g0 dd rows C0 g0 0#usize true none := by
    refine ⟨rfl, rfl, rfl, by show (0#usize).val ≤ rows.val; rw [show (0#usize).val = 0 from rfl]; omega, ?_, ?_⟩
    · simp [rowsVals]
    · intro i hi j hj
      have := hbound i hi j hj
      show |(gvals g0).getD (i * dd.val + j) 0| ≤ C0 + (0 : ℤ) * 16384
      simpa using this
  exact rows_loop_loop_spec x dd rows g0 g0 0#usize C0 hlen0 hxlen hC0 hC hinv0

/-! ## 7. Headlines -/

/-- Under shape agreement and an explicit no-overflow bound, `gram.gram_fold_i8` returns `Ok g'`
where every entry equals its initial value plus the exact integer Gram sum. -/
theorem gram_fold_i8_ok
    (g : alloc.vec.Vec I64) (x : Slice I8) (rows d : Usize) (C0 : ℤ)
    (hglen : g.val.length = d.val * d.val)
    (hxlen : x.val.length = rows.val * d.val)
    (hC0 : 0 ≤ C0)
    (hC : C0 + (rows.val : ℤ) * 16384 ≤ I64.max)
    (hbound : ∀ i < d.val, ∀ j < d.val, |(gvals g).getD (i * d.val + j) 0| ≤ C0) :
    ∃ g', gram.gram_fold_i8 g x rows d = ok (core.result.Result.Ok g') ∧
      ∀ i < d.val, ∀ j < d.val,
        (gvals g').getD (i * d.val + j) 0 =
          (gvals g).getD (i * d.val + j) 0 +
            Finset.sum (Finset.range rows.val)
              (fun r => (x.val.getD (r * d.val + i) 0#i8).val * (x.val.getD (r * d.val + j) 0#i8).val) := by
  have hdd_max : d.val * d.val ≤ Usize.max := by
    rw [← hglen]; exact Slice.property g.slice
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := d) (y := d) hdd_max)
  have hi_eq : alloc.vec.Vec.len g = i1 := by
    apply UScalar.eq_of_val_eq
    rw [alloc.vec.Vec.len_val, hi1v]
    exact hglen
  have hrd_max : rows.val * d.val ≤ Usize.max := by
    rw [← hxlen]; exact Slice.property x
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := rows) (y := d) hrd_max)
  have hi2_eq : Slice.len x = i3 := by
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val, hi3v]
    exact hxlen
  have hspec := rows_loop_spec x d rows g C0 hglen (by omega) hC0 hC hbound
  obtain ⟨⟨g1, err⟩, hrows_exec, herr, _hg1_len, hg1_vals, _hg1_bound⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst herr
  refine ⟨g1, ?_, ?_⟩
  · rw [gram.gram_fold_i8]
    simp only [hi1, hi3, hrows_exec]
    have h1 : g.val.length = i1.val := by
      simpa only [alloc.vec.Vec.len_val] using congrArg (fun u : Usize => u.val) hi_eq
    have h2 : x.val.length = i3.val := by
      simpa only [Slice.len_val] using congrArg (fun u : Usize => u.val) hi2_eq
    simp
    rw [if_pos h1, if_pos h2]
  · intro i hi j hj
    have hg1_v : (gvals g1).getD (i * d.val + j) 0 =
        (rowsVals (gvals g) (xvals x) d.val rows.val).getD (i * d.val + j) 0 := by
      rw [hg1_vals]
    rw [hg1_v]
    have hlen_g : d.val * d.val ≤ (gvals g).length := by rw [gvals_length, hglen]
    rw [rowsVals_getD (gvals g) (xvals x) d.val rows.val hlen_g i j hi hj]
    congr 1
    apply Finset.sum_congr rfl
    intro r hr
    have hrlt : r < rows.val := Finset.mem_range.mp hr
    have hr_i : r * d.val + i < x.val.length := by
      rw [hxlen]
      calc r * d.val + i < r * d.val + d.val := by omega
        _ = (r + 1) * d.val := by rw [Nat.succ_mul]
        _ ≤ rows.val * d.val := Nat.mul_le_mul_right d.val (Nat.succ_le_of_lt hrlt)
    have hr_j : r * d.val + j < x.val.length := by
      rw [hxlen]
      calc r * d.val + j < r * d.val + d.val := by omega
        _ = (r + 1) * d.val := by rw [Nat.succ_mul]
        _ ≤ rows.val * d.val := Nat.mul_le_mul_right d.val (Nat.succ_le_of_lt hrlt)
    rw [xvals_getD x hr_i, xvals_getD x hr_j]

/-- Shape mismatch refusal: if the accumulator vector or data slice length does not match
the dimensions, `gram.gram_fold_i8` returns `Ok (Err ShapeMismatch)`. -/
theorem gram_fold_i8_refusal
    (g : alloc.vec.Vec I64) (x : Slice I8) (rows d : Usize)
    (hdd : d.val * d.val ≤ Usize.max)
    (hrows : rows.val * d.val ≤ Usize.max)
    (hne : g.val.length ≠ d.val * d.val ∨ x.val.length ≠ rows.val * d.val) :
    gram.gram_fold_i8 g x rows d = ok (core.result.Result.Err gram.GramError.ShapeMismatch) := by
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := d) (y := d) hdd)
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := rows) (y := d) hrows)
  rw [gram.gram_fold_i8]
  simp only [hi1, hi3]
  simp
  intro hc1 hc2
  exfalso
  rcases hne with hh | hh
  · exact hh (by rw [hc1, hi1v])
  · exact hh (by rw [hc2, hi3v])

/-! ## 8. The link to the canonical `gram_stack`

`toMat` reads the `i8` batch as a real matrix; `gram_fold_i8_entry` identifies one fold's entry with
an entry of `XᵀX`; `gram_fold_i8_stack` chains two folds and rewrites through the canonical
`Sanguine.TrainingCollapse.gram_stack` (the stacked batch's `XᵀX` is the sum of the blocks'). -/

/-- The real matrix of the `rows` rows of `x` at stride `d`. -/
def toMat (x : Slice I8) (rows d : ℕ) : Matrix (Fin rows) (Fin d) ℝ :=
  Matrix.of fun r c => ((x.val.getD (r.val * d + c.val) 0#i8).val : ℝ)

/-- The zero `i64` accumulator of a `d × d` Gram buffer. -/
def zeroGram (d : Usize) (h : d.val * d.val ≤ Usize.max) : alloc.vec.Vec I64 :=
  alloc.vec.Vec.from (List.replicate (d.val * d.val) 0#i64) (by simpa using h)

theorem zeroGram_val (d : Usize) (h : d.val * d.val ≤ Usize.max) :
    (zeroGram d h).val = List.replicate (d.val * d.val) 0#i64 := by
  simp [zeroGram, alloc.vec.Vec.from_val]

theorem zeroGram_getD (d : Usize) (h : d.val * d.val ≤ Usize.max) {k : ℕ}
    (hk : k < d.val * d.val) : (gvals (zeroGram d h)).getD k 0 = 0 := by
  have hg : gvals (zeroGram d h) = List.replicate (d.val * d.val) (0 : ℤ) := by
    simp [gvals, zeroGram_val, List.map_replicate]
  rw [hg]
  exact List.getD_replicate (0 : ℤ) hk

/-- An in-range Gram index stays in range: `i·d + j < d·d` for `i, j < d`. -/
theorem gram_idx_lt {dd i j : ℕ} (hi : i < dd) (hj : j < dd) : i * dd + j < dd * dd := by
  have h1 : (i + 1) * dd ≤ dd * dd := Nat.mul_le_mul_right dd (Nat.succ_le_of_lt hi)
  rw [Nat.succ_mul] at h1
  omega

/-- The `range rows` Gram sum, cast to `ℝ`, is the matching entry of `XᵀX`. -/
theorem sum_range_eq_gram (x : Slice I8) (rows d : ℕ) (i j : Fin d) :
    ((Finset.sum (Finset.range rows)
        (fun r => (x.val.getD (r * d + i.val) 0#i8).val *
                  (x.val.getD (r * d + j.val) 0#i8).val) : ℤ) : ℝ)
      = ((toMat x rows d)ᵀ * toMat x rows d) i j := by
  rw [Int.cast_sum, Matrix.mul_apply]
  rw [show (∑ k : Fin rows, (toMat x rows d)ᵀ i k * toMat x rows d k j)
        = ∑ k : Fin rows,
            ((x.val.getD (k.val * d + i.val) 0#i8).val : ℝ) *
            ((x.val.getD (k.val * d + j.val) 0#i8).val : ℝ) by
      apply Finset.sum_congr rfl
      intro k _
      rw [Matrix.transpose_apply]
      simp [toMat, Matrix.of_apply]]
  rw [Fin.sum_univ_eq_sum_range
      (fun r => ((x.val.getD (r * d + i.val) 0#i8).val : ℝ) *
                ((x.val.getD (r * d + j.val) 0#i8).val : ℝ)) rows]
  apply Finset.sum_congr rfl
  intro r _
  rw [Int.cast_mul]

/-- `gram_fold_i8_ok` strengthened with length preservation and the accumulator bound, so a fold's
result can be chained into a second fold. -/
theorem gram_fold_i8_spec
    (g : alloc.vec.Vec I64) (x : Slice I8) (rows d : Usize) (C0 : ℤ)
    (hglen : g.val.length = d.val * d.val)
    (hxlen : x.val.length = rows.val * d.val)
    (hC0 : 0 ≤ C0)
    (hC : C0 + (rows.val : ℤ) * 16384 ≤ I64.max)
    (hbound : ∀ i < d.val, ∀ j < d.val, |(gvals g).getD (i * d.val + j) 0| ≤ C0) :
    ∃ g', gram.gram_fold_i8 g x rows d = ok (core.result.Result.Ok g') ∧
      g'.val.length = g.val.length ∧
      (∀ i < d.val, ∀ j < d.val,
        (gvals g').getD (i * d.val + j) 0 =
          (gvals g).getD (i * d.val + j) 0 +
            Finset.sum (Finset.range rows.val)
              (fun r => (x.val.getD (r * d.val + i) 0#i8).val * (x.val.getD (r * d.val + j) 0#i8).val)) ∧
      (∀ i < d.val, ∀ j < d.val, |(gvals g').getD (i * d.val + j) 0| ≤ C0 + (rows.val : ℤ) * 16384) := by
  have hdd_max : d.val * d.val ≤ Usize.max := by
    rw [← hglen]; exact Slice.property g.slice
  obtain ⟨i1, hi1, hi1v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := d) (y := d) hdd_max)
  have hi_eq : alloc.vec.Vec.len g = i1 := by
    apply UScalar.eq_of_val_eq
    rw [alloc.vec.Vec.len_val, hi1v]
    exact hglen
  have hrd_max : rows.val * d.val ≤ Usize.max := by
    rw [← hxlen]; exact Slice.property x
  obtain ⟨i3, hi3, hi3v⟩ := WP.spec_imp_exists (Usize.mul_spec (x := rows) (y := d) hrd_max)
  have hi2_eq : Slice.len x = i3 := by
    apply UScalar.eq_of_val_eq
    rw [Slice.len_val, hi3v]
    exact hxlen
  have hspec := rows_loop_spec x d rows g C0 hglen (by omega) hC0 hC hbound
  obtain ⟨⟨g1, err⟩, hrows_exec, herr, hg1_len, hg1_vals, hg1_bound⟩ := (WP.spec_equiv_exists _ _).mp hspec
  subst herr
  refine ⟨g1, ?_, hg1_len, ?_, hg1_bound⟩
  · rw [gram.gram_fold_i8]
    simp only [hi1, hi3, hrows_exec]
    have h1 : g.val.length = i1.val := by
      simpa only [alloc.vec.Vec.len_val] using congrArg (fun u : Usize => u.val) hi_eq
    have h2 : x.val.length = i3.val := by
      simpa only [Slice.len_val] using congrArg (fun u : Usize => u.val) hi2_eq
    simp
    rw [if_pos h1, if_pos h2]
  · intro i hi j hj
    have hg1_v : (gvals g1).getD (i * d.val + j) 0 =
        (rowsVals (gvals g) (xvals x) d.val rows.val).getD (i * d.val + j) 0 := by
      rw [hg1_vals]
    rw [hg1_v]
    have hlen_g : d.val * d.val ≤ (gvals g).length := by rw [gvals_length, hglen]
    rw [rowsVals_getD (gvals g) (xvals x) d.val rows.val hlen_g i j hi hj]
    congr 1
    apply Finset.sum_congr rfl
    intro r hr
    have hrlt : r < rows.val := Finset.mem_range.mp hr
    have hr_i : r * d.val + i < x.val.length := by
      rw [hxlen]
      calc r * d.val + i < r * d.val + d.val := by omega
        _ = (r + 1) * d.val := by rw [Nat.succ_mul]
        _ ≤ rows.val * d.val := Nat.mul_le_mul_right d.val (Nat.succ_le_of_lt hrlt)
    have hr_j : r * d.val + j < x.val.length := by
      rw [hxlen]
      calc r * d.val + j < r * d.val + d.val := by omega
        _ = (r + 1) * d.val := by rw [Nat.succ_mul]
        _ ≤ rows.val * d.val := Nat.mul_le_mul_right d.val (Nat.succ_le_of_lt hrlt)
    rw [xvals_getD x hr_i, xvals_getD x hr_j]

/-- From the zero accumulator, one fold's entry cast to `ℝ` is the matching entry of `XᵀX`.
Derived from `gram_fold_i8_ok` with `C0 = 0`. -/
theorem gram_fold_i8_entry
    (x : Slice I8) (rows d : Usize) (hd : d.val * d.val ≤ Usize.max)
    (hxlen : x.val.length = rows.val * d.val)
    (hC : (rows.val : ℤ) * 16384 ≤ I64.max) :
    ∃ g', gram.gram_fold_i8 (zeroGram d hd) x rows d = ok (core.result.Result.Ok g') ∧
      ∀ i j : Fin d.val,
        ((gvals g').getD (i.val * d.val + j.val) 0 : ℝ)
          = ((toMat x rows.val d.val)ᵀ * toMat x rows.val d.val) i j := by
  have hglen : (zeroGram d hd).val.length = d.val * d.val := by
    rw [zeroGram_val, List.length_replicate]
  have hbound : ∀ i < d.val, ∀ j < d.val,
      |(gvals (zeroGram d hd)).getD (i * d.val + j) 0| ≤ (0 : ℤ) := by
    intro i hi j hj
    rw [zeroGram_getD d hd (gram_idx_lt hi hj)]
    simp
  obtain ⟨g', hfold, hentry⟩ :=
    gram_fold_i8_ok (zeroGram d hd) x rows d 0 hglen hxlen (le_refl 0) (by simpa using hC) hbound
  refine ⟨g', hfold, ?_⟩
  intro i j
  have hz : (gvals (zeroGram d hd)).getD (i.val * d.val + j.val) 0 = 0 :=
    zeroGram_getD d hd (gram_idx_lt i.isLt j.isLt)
  have hcast : ((gvals g').getD (i.val * d.val + j.val) 0 : ℝ)
      = ((Finset.sum (Finset.range rows.val)
          (fun r => (x.val.getD (r * d.val + i.val) 0#i8).val *
                    (x.val.getD (r * d.val + j.val) 0#i8).val) : ℤ) : ℝ) := by
    rw [hentry i.val i.isLt j.val j.isLt, hz, zero_add]
  rw [hcast]
  exact sum_range_eq_gram x rows.val d.val i j

/-- **The extracted fold composes**: folding `x₁` (`n` rows) then `x₂` (`m` rows) from the zero
accumulator gives the stacked batch's `XᵀX`, by the canonical
`Sanguine.TrainingCollapse.gram_stack`. -/
theorem gram_fold_i8_stack
    (x₁ : Slice I8) (n : Usize) (x₂ : Slice I8) (m : Usize) (d : Usize)
    (hd : d.val * d.val ≤ Usize.max)
    (hx₁ : x₁.val.length = n.val * d.val)
    (hx₂ : x₂.val.length = m.val * d.val)
    (hC : ((n.val : ℤ) + (m.val : ℤ)) * 16384 ≤ I64.max) :
    ∃ g₁ g₂,
      gram.gram_fold_i8 (zeroGram d hd) x₁ n d = ok (core.result.Result.Ok g₁) ∧
      gram.gram_fold_i8 g₁ x₂ m d = ok (core.result.Result.Ok g₂) ∧
      ∀ i j : Fin d.val,
        ((gvals g₂).getD (i.val * d.val + j.val) 0 : ℝ)
          = ((Sanguine.TrainingCollapse.stack (toMat x₁ n.val d.val) (toMat x₂ m.val d.val))ᵀ *
              Sanguine.TrainingCollapse.stack (toMat x₁ n.val d.val) (toMat x₂ m.val d.val)) i j := by
  have hglen0 : (zeroGram d hd).val.length = d.val * d.val := by
    rw [zeroGram_val, List.length_replicate]
  have hbound0 : ∀ i < d.val, ∀ j < d.val,
      |(gvals (zeroGram d hd)).getD (i * d.val + j) 0| ≤ (0 : ℤ) := by
    intro i hi j hj
    rw [zeroGram_getD d hd (gram_idx_lt hi hj)]
    simp
  have hnC : (n.val : ℤ) * 16384 ≤ I64.max := by
    have hm : (0 : ℤ) ≤ (m.val : ℤ) * 16384 := by positivity
    nlinarith [hC]
  have hC1 : (0 : ℤ) + (n.val : ℤ) * 16384 ≤ I64.max := by linarith
  obtain ⟨g₁, hfold1, hlen1, hentry1, hbound1⟩ :=
    gram_fold_i8_spec (zeroGram d hd) x₁ n d 0 hglen0 hx₁ (le_refl 0) hC1 hbound0
  have hglen1 : g₁.val.length = d.val * d.val := by rw [hlen1, hglen0]
  have hC2 : (n.val : ℤ) * 16384 + (m.val : ℤ) * 16384 ≤ I64.max := by nlinarith [hC]
  obtain ⟨g₂, hfold2, hentry2⟩ :=
    gram_fold_i8_ok g₁ x₂ m d ((n.val : ℤ) * 16384) hglen1 hx₂
      (by positivity) hC2 (by intro i hi j hj; simpa using hbound1 i hi j hj)
  refine ⟨g₁, g₂, hfold1, hfold2, ?_⟩
  intro i j
  have hz : (gvals (zeroGram d hd)).getD (i.val * d.val + j.val) 0 = 0 :=
    zeroGram_getD d hd (gram_idx_lt i.isLt j.isLt)
  have hg1 : ((gvals g₁).getD (i.val * d.val + j.val) 0 : ℝ)
      = ((toMat x₁ n.val d.val)ᵀ * toMat x₁ n.val d.val) i j := by
    have e1 : (gvals g₁).getD (i.val * d.val + j.val) 0 =
        (gvals (zeroGram d hd)).getD (i.val * d.val + j.val) 0 +
          Finset.sum (Finset.range n.val)
            (fun r => (x₁.val.getD (r * d.val + i.val) 0#i8).val *
                      (x₁.val.getD (r * d.val + j.val) 0#i8).val) :=
      hentry1 i.val i.isLt j.val j.isLt
    rw [e1, hz, zero_add]
    exact sum_range_eq_gram x₁ n.val d.val i j
  have hcast : ((gvals g₂).getD (i.val * d.val + j.val) 0 : ℝ)
      = ((gvals g₁).getD (i.val * d.val + j.val) 0 : ℝ) +
        ((Finset.sum (Finset.range m.val)
          (fun r => (x₂.val.getD (r * d.val + i.val) 0#i8).val *
                    (x₂.val.getD (r * d.val + j.val) 0#i8).val) : ℤ) : ℝ) := by
    rw [hentry2 i.val i.isLt j.val j.isLt, Int.cast_add]
  rw [hcast, hg1]
  rw [sum_range_eq_gram x₂ m.val d.val i j]
  rw [Sanguine.TrainingCollapse.gram_stack (toMat x₁ n.val d.val) (toMat x₂ m.val d.val),
      Matrix.add_apply]

/-- Non-vacuity: `gram_fold_i8_stack`'s joint hypotheses are discharged at `d = n = m = 1`. -/
example :
    ∃ (x₁ x₂ : Slice I8) (n m d : Usize),
      d.val * d.val ≤ Usize.max ∧
      x₁.val.length = n.val * d.val ∧
      x₂.val.length = m.val * d.val ∧
      ((n.val : ℤ) + (m.val : ℤ)) * 16384 ≤ I64.max :=
  ⟨Slice.from [3#i8] (by scalar_tac), Slice.from [5#i8] (by scalar_tac),
   1#usize, 1#usize, 1#usize, by scalar_tac, by simp, by simp, by scalar_tac⟩

end apeiron_gram

