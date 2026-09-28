import Spirv.CheckScanState

/-!
# `apply_shape` used-id scan loops

Exact specs for the used-id scan loops of `src/spirv/check.rs::apply_shape`, over the extracted
bodies in `proof/ApeironWord.lean`:

* `firstUndef` — the pure first-undefined used id: the leftmost element of the scanned operand
  range absent from `defs`. The extracted body errors on the *first* such id
  (`CheckError.UndefinedId index used`), so this is the payload selector for the refusal.
* `apply_shape_scan_spec` — one generic scan lemma: for any body that continues at `u + 1` while
  `v[u] ∈ defs`, returns `errVal v[u]` at the first `v[u] ∉ defs`, and returns `exitVal` once past
  the range, the whole loop returns `errVal used` for `used = firstUndef v defs` when one exists and
  `exitVal` otherwise. Proved by `loop.spec_decr_nat` on the measure `v.length - i.val` and the
  prefix invariant `i.val ≤ v.length ∧ ∀ x ∈ v.take i.val, x ∈ defs`.
* `apply_shape_loopN_spec` (`N = 0..7`) — the loop from its initial index `0#usize`, instantiated
  from the generic scan lemma. `apply_shape_loop0_body_cont`/`_undefined` (CheckScanState) supply
  the two in-range body steps and `apply_shape_loop0_body_exit` the past-the-range step.

The extraction contains **eight** copies of the source loop (`apply_shape_loop0` … `apply_shape_loop7`,
one per `storage_class`/`decoration`/`sudot` branch of `apply_shape`); all eight bodies are
definitionally equal, so `apply_shape_loopN_spec` for `N = 1..7` is the `N = 0` statement read
through `rfl`. `applyShapeExit` is the pure image of the extracted post-scan branch (`o = none` is a
no-op; `o = some result` refuses a duplicate result id, otherwise pushes `result` and records its
`resolveTagPure` tag through `tyInsertVec`).

## Residual obligation (component 2 of `Spirv/Check.lean`, not proved here)

* `apply_shape` itself: the assembly of the eight branch calls with their `storage_class`/`decoration`
  admission (`storage_class_admitted_spec`, `decoration_admitted_spec`, CheckScanState), the
  `sudot` result/operand type checks and the `TypeMismatch`/`BadStorageClass`/`BadDecoration`
  refusals are outside this module. The two `Vec` capacity bounds (`defs.val.length < Usize.max`,
  `tys.val.length < Usize.max`) are hypotheses of `apply_shape_loopN_spec`; they are not discharged
  here.
* `check_fragment_loop` and its scan invariant `defs.toList = st.defs.reverse`, and the composition
  to `decode`, remain open.
* The `TyTag`/key translation `τ` to `Spirv.Decode`'s `lookupTy`/`resultTag` (CheckScan.lean's
  header) remains open.
-/

namespace Spirv

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## The pure first-undefined used id -/

/-- Left-to-right first element of `v` absent from `defs`; `none` when every element of `v` is in
`defs`. Mirrors the extracted scan's stop position: it errors at the first absent id. -/
def firstUndef : List Std.U32 → List Std.U32 → Option Std.U32
  | [], _ => none
  | x :: t, defs => if x ∈ defs then firstUndef t defs else some x

/-- A prefix with no absent id leaves the scan of the appended tail unchanged. -/
theorem firstUndef_append_of_none {l₁ l₂ defs : List Std.U32}
    (h : firstUndef l₁ defs = none) : firstUndef (l₁ ++ l₂) defs = firstUndef l₂ defs := by
  induction l₁ with
  | nil => simp [firstUndef]
  | cons x t ih =>
      simp only [List.cons_append]
      by_cases hx : x ∈ defs
      · simp only [firstUndef, if_pos hx] at h ⊢
        exact ih h
      · simp only [firstUndef, if_neg hx] at h
        exact absurd h (by simp)

/-- Every element in `defs` means no first-undefined id. -/
theorem firstUndef_eq_none_of_all {l defs : List Std.U32} (h : ∀ x ∈ l, x ∈ defs) :
    firstUndef l defs = none := by
  induction l with
  | nil => rfl
  | cons x t ih =>
      simp only [firstUndef, if_pos (h x (by simp))]
      exact ih (fun y hy => h y (by simp [hy]))

/-- With the prefix defined, the first absent id is the one at the current index. -/
theorem firstUndef_eq_some_of_lt {v defs : List Std.U32} {i : Nat}
    (hlt : i < v.length) (hpre : ∀ x ∈ v.take i, x ∈ defs) (hnot : v[i]'hlt ∉ defs) :
    firstUndef v defs = some (v[i]'hlt) := by
  conv_lhs => rw [← List.take_append_drop i v]
  rw [firstUndef_append_of_none (defs := defs)
    (firstUndef_eq_none_of_all (defs := defs) (l := v.take i) hpre)]
  rw [List.drop_eq_getElem_cons hlt]
  simp only [firstUndef, if_neg hnot]

/-! ## The generic used-id scan lemma -/

/-- **Generic used-id scan.** A loop whose body continues at `u + 1` while `v[u] ∈ defs`, returns
`errVal v[u]` at the first `v[u] ∉ defs`, and returns `exitVal` once `u` is past the range, returns
`errVal used` for `used = firstUndef v defs` when one exists and `exitVal` otherwise. Proved by
`loop.spec_decr_nat` on the measure `v.length - i.val` and the prefix invariant
`i.val ≤ v.length ∧ ∀ x ∈ v.take i.val, x ∈ defs`. -/
theorem apply_shape_scan_spec {α : Type}
    (body : Std.Usize → Result (ControlFlow Std.Usize α))
    (v defs : List Std.U32) (errVal : Std.U32 → α) (exitVal : α)
    (hbound : v.length ≤ Usize.max)
    (hcont : ∀ (u : Std.Usize) (hlt : u.val < v.length),
      v[u.val]'hlt ∈ defs →
      body u = (do let u' ← u + 1#usize; ok (ControlFlow.cont u')))
    (herr : ∀ (u : Std.Usize) (hlt : u.val < v.length),
      v[u.val]'hlt ∉ defs →
      body u = ok (ControlFlow.done (errVal (v[u.val]'hlt))))
    (hexit : ∀ u : Std.Usize, u.val = v.length → body u = ok (ControlFlow.done exitVal)) :
    loop body 0#usize =
      ok (match firstUndef v defs with
          | some used => errVal used
          | none => exitVal) := by
  have hspec : loop body 0#usize ⦃ r => r = (match firstUndef v defs with
      | some used => errVal used
      | none => exitVal) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Std.Usize) => v.length - i.val)
      (inv := fun (i : Std.Usize) => i.val ≤ v.length ∧ ∀ x ∈ v.take i.val, x ∈ defs)
    · intro i hi
      obtain ⟨hile, hpre⟩ := hi
      by_cases hlt : i.val < v.length
      · by_cases hmem : v[i.val]'hlt ∈ defs
        · have h1 : (1#usize).val = 1 := rfl
          have hmax : i.val + (1#usize).val ≤ Usize.max := by
            rw [h1]; omega
          obtain ⟨u', hu', hu'v⟩ := Aeneas.Std.WP.spec_imp_exists
            (Usize.add_spec (x := i) (y := 1#usize) hmax)
          have hu'v' : u'.val = i.val + 1 := by rw [hu'v, h1]
          have hbody : body i = ok (ControlFlow.cont u') := by
            rw [hcont i hlt hmem, hu']
            simp only [bind_tc_ok]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          refine ⟨⟨?_, ?_⟩, ?_⟩
          · rw [hu'v']; omega
          · intro x hx
            rw [hu'v', List.take_succ_eq_append_getElem hlt] at hx
            rcases List.mem_append.mp hx with h | h
            · exact hpre x h
            · rw [List.mem_singleton] at h
              rw [h]; exact hmem
          · rw [hu'v']; omega
        · rw [herr i hlt hmem]
          simp only [Aeneas.Std.WP.spec_ok]
          rw [firstUndef_eq_some_of_lt hlt hpre hmem]
      · rw [hexit i (by omega)]
        simp only [Aeneas.Std.WP.spec_ok]
        have htake : v.take i.val = v := List.take_of_length_le (by omega)
        have hall : ∀ x ∈ v, x ∈ defs := by
          intro x hx
          exact hpre x (by rw [htake]; exact hx)
        rw [firstUndef_eq_none_of_all hall]
    · exact ⟨Nat.zero_le _, by simp⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hpost] at hr
  exact hr

/-! ## The extracted post-scan branch -/

/-- The pure image of the post-scan branch of `apply_shape_loopN.body`: `o = none` is a no-op;
`o = some result` refuses a duplicate result id, otherwise pushes `result` and records its
`resolveTagPure` tag through `tyInsertVec`. `hdefs`/`htys` are the `Vec` capacity bounds the push
and insert arms need; they do not appear in the result when those arms are not taken. -/
def applyShapeExit (index : Std.Usize) (o : Option Std.U32) (ts : spirv.check.TagSpec)
    (defs : alloc.vec.Vec Std.U32) (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    (core.result.Result Unit spirv.check.CheckError) × (alloc.vec.Vec Std.U32) ×
      (alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) :=
  match o with
  | none => (core.result.Result.Ok (), defs, tys)
  | some result =>
      if result ∈ defs.val then
        (core.result.Result.Err (spirv.check.CheckError.DuplicateResultId index result), defs, tys)
      else
        let defs1 := alloc.vec.Vec.from (defs.val ++ [result])
          (by rw [List.length_append, List.length_singleton]; omega)
        match resolveTagPure ts tys.val with
        | none => (core.result.Result.Ok (), defs1, tys)
        | some tag => (core.result.Result.Ok (), defs1, tyInsertVec tys result tag htys)

/-- `defs_contains` on a dereferenced `Vec` reads through to `defs.val` for an absent id. -/
theorem defs_contains_vec_deref_false (defs : alloc.vec.Vec Std.U32) (id : Std.U32)
    (h : id ∉ defs.val) :
    spirv.check.defs_contains (alloc.vec.Vec.deref defs) id = ok false := by
  rw [defs_contains_spec]
  simp [alloc.vec.Vec.deref, Slice.from_val, h]

/-- **`apply_shape_loop0` past-the-range body step.** With the scan index past `uses.len()`, the
body runs the post-scan branch: its result is `applyShapeExit`. -/
theorem apply_shape_loop0_body_exit (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (u : Usize)
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max)
    (h : u.val = v.val.length) :
    spirv.check.apply_shape_loop0.body index o v ts defs tys u =
      ok (ControlFlow.done (applyShapeExit index o ts defs tys hdefs htys)) := by
  dsimp only [spirv.check.apply_shape_loop0.body]
  rw [if_neg (by
    intro hc
    have hc' : u.val < (alloc.vec.Vec.len v).val := (UScalar.lt_equiv u _).1 hc
    have hlen : (alloc.vec.Vec.len v).val = v.val.length := alloc.vec.Vec.len_val v
    rw [hlen] at hc'
    omega)]
  cases o with
  | none => rfl
  | some result =>
      dsimp only
      by_cases hmem : result ∈ defs.val
      · rw [defs_contains_vec_deref defs result hmem]
        simp only [bind_tc_ok]
        unfold applyShapeExit
        simp only [if_pos hmem]
        simp
      · rw [defs_contains_vec_deref_false defs result hmem]
        simp only [bind_tc_ok]
        rw [vec_push_eq defs result hdefs]
        simp only [bind_tc_ok]
        rw [resolve_tag_spec ts (alloc.vec.Vec.deref tys)]
        simp only [bind_tc_ok]
        have hderef : (alloc.vec.Vec.deref tys).val = tys.val := by
          simp [alloc.vec.Vec.deref, Slice.from_val]
        rw [hderef]
        unfold applyShapeExit
        simp only [if_neg hmem]
        cases hrt : resolveTagPure ts tys.val with
        | none => rfl
        | some tag =>
            dsimp only
            rw [ty_insert_spec tys result tag htys]
            simp only [bind_tc_ok]
            rw [if_neg (by simp)]

/-! ## The loop specs -/

/-- **`apply_shape_loop0` spec.** The used-id scan from the initial index returns the refusal
`Err (UndefinedId index used)` for the *first* absent used id `used`, and otherwise the post-scan
`applyShapeExit` result. `loop.spec_decr_nat` on the prefix invariant
`i.val ≤ uses.val.length ∧ ∀ x ∈ uses.val.take i.val, x ∈ defs.val`. -/
theorem apply_shape_loop0_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop0 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) := by
  unfold spirv.check.apply_shape_loop0
  exact apply_shape_scan_spec
    (body := fun u => spirv.check.apply_shape_loop0.body index o v ts defs tys u)
    v.val defs.val
    (fun used => (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys))
    (applyShapeExit index o ts defs tys hdefs htys)
    (alloc.vec.Vec.property v)
    (fun u hlt hmem => apply_shape_loop0_body_cont index o v ts defs tys u hlt hmem)
    (fun u hlt hnot => apply_shape_loop0_body_undefined index o v ts defs tys u hlt hnot)
    (fun u hu => apply_shape_loop0_body_exit index o v ts defs tys u hdefs htys hu)

/-- `apply_shape_loop1` is `apply_shape_loop0`: the extracted bodies are definitionally equal. -/
theorem apply_shape_loop1_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop1 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

/-- `apply_shape_loop2` is `apply_shape_loop0`. -/
theorem apply_shape_loop2_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop2 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

/-- `apply_shape_loop3` is `apply_shape_loop0`. -/
theorem apply_shape_loop3_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop3 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

/-- `apply_shape_loop4` is `apply_shape_loop0`. -/
theorem apply_shape_loop4_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop4 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

/-- `apply_shape_loop5` is `apply_shape_loop0`. -/
theorem apply_shape_loop5_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop5 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

/-- `apply_shape_loop6` is `apply_shape_loop0`. -/
theorem apply_shape_loop6_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop6 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

/-- `apply_shape_loop7` is `apply_shape_loop0`. -/
theorem apply_shape_loop7_spec (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag))
    (hdefs : defs.val.length < Usize.max) (htys : tys.val.length < Usize.max) :
    spirv.check.apply_shape_loop7 index o v ts defs tys 0#usize =
      ok (match firstUndef v.val defs.val with
          | some used =>
            (core.result.Result.Err (spirv.check.CheckError.UndefinedId index used), defs, tys)
          | none => applyShapeExit index o ts defs tys hdefs htys) :=
  apply_shape_loop0_spec index o v ts defs tys hdefs htys

end Spirv
