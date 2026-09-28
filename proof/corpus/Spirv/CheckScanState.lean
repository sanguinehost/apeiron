import Spirv.CheckScan

/-!
# Checker scan-state primitives (APEIRON-GPU-SPV-W3-B5a)

Exact specs for the remaining scalar scans of `src/spirv/check.rs`, over the extracted bodies in
`proof/ApeironWord.lean`:

* `decoration_admitted_spec`, `storage_class_admitted_spec` — the fixed-table membership scans
  (`VALID_DECORATIONS`, `VALID_STORAGE_CLASSES`) return exactly
  `decide (x.val ∈ list)` for the `Nat` literal of the table, proved by `loop.spec_decr_nat` on the
  prefix invariant `i.val ≤ table.length ∧ x ∉ table.take i.val` (mirroring `defs_contains_spec`);
* `ty_insert_spec` — the id-to-tag table insert: `ty_insert` scans oldest-first and, at the first
  entry whose key is `id`, **overwrites** its value (`Vec.set`, the capacity is preserved); when `id`
  is absent it appends `(id, tag)` (`Vec.push`, the capacity becomes the wrapped length). There is no
  refusal arm: the duplicate-result-id refusal lives in `apply_shape` (`defs_contains defs result`
  before the insert), not in `ty_insert`. The pure image is
  `tyInsertVec`, whose value is `tyInsertList` and whose absent-branch capacity is `Vec.from`'s;
* `resolve_tag_spec` — `resolve_tag` is `resolveTagPure`, i.e. `ty_get`'s `firstLookup` under a case
  split on `TagSpec` (`ty_get_spec`); the `Vector`/`RuntimeArray`/`Pointer` cases map a present
  lookup to the tag and an absent one to `none`, and `OfType` is the lookup itself;
* `apply_shape_loop0_body_cont`, `apply_shape_loop0_body_undefined` — one body step of the
  id-must-be-defined scan: an in-bounds, defined used-id continues at `u + 1`; an in-bounds
  undefined used-id returns `CheckError.UndefinedId index used` with `defs`/`tys` unchanged.

## Residual obligation (component 2 of `Spirv/Check.lean`, not proved here)

The loop invariant `defs.toList = st.defs.reverse` for `check_fragment_loop`, the `loop.spec_decr_nat`
assembly of `apply_shape_loop0`–`apply_shape_loop3` from the per-body steps above, and the
`TyTag`/key correspondence to `Spirv.Decode`'s `lookupTy`/`resultTag` (the translation `τ` of
`Spirv/CheckScan.lean`'s header) remain open; `apply_shape_loop4` and the composition to
`wfCheckB` are likewise outside this module.

## Controls

The decoration-list literal is the only input of `decoration_admitted_spec`: dropping one element
from `validDecorationsList` in a throwaway copy makes `VALID_DECORATIONS_val` fail (`simp` leaves
`[2, 6, 11, 33, 34, 35] = [2, 6, 11, 33, 34]`), and restoring the copy compiles.
-/

namespace Spirv

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false

open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## The fixed admission tables -/

/-- The `Nat` literal of the extracted `VALID_DECORATIONS` table: the input of
`decoration_admitted_spec`. -/
def validDecorationsList : List Nat := [2, 6, 11, 33, 34, 35]

theorem VALID_DECORATIONS_val :
    spirv.check.VALID_DECORATIONS.val.map (fun u : Std.U32 => u.val) = validDecorationsList := by
  unfold spirv.check.VALID_DECORATIONS validDecorationsList
  simp

/-- The `Nat` literal of the extracted `VALID_STORAGE_CLASSES` table: the input of
`storage_class_admitted_spec`. -/
def validStorageClassesList : List Nat := [1, 4, 7, 12]

theorem VALID_STORAGE_CLASSES_val :
    spirv.check.VALID_STORAGE_CLASSES.val.map (fun u : Std.U32 => u.val) =
      validStorageClassesList := by
  unfold spirv.check.VALID_STORAGE_CLASSES validStorageClassesList
  simp

/-- In-bounds `Array.index_usize` reads the underlying list. -/
theorem array_index_usize_eq {α : Type} {n : Std.Usize} (a : Array α n) (i : Usize)
    (h : i.val < a.val.length) : Array.index_usize a i = ok (a.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := Aeneas.Std.WP.spec_imp_exists (Array.index_usize_spec a i h)
  rw [hxv] at hx
  exact hx

/-- Reading a `U32` value as `Nat` is injective, so absence is preserved. -/
theorem mem_map_val_of_not_mem {l : List Std.U32} {u : Std.U32} (h : u ∉ l) :
    u.val ∉ l.map (fun x => x.val) := by
  intro hc
  obtain ⟨b, hb, hbv⟩ := List.mem_map.mp hc
  exact h (UScalar.eq_of_val_eq hbv ▸ hb)

/-- **`decoration_admitted` spec.** The extracted table scan is `decide` membership in the `Nat`
literal, proved by `loop.spec_decr_nat` on the prefix invariant
`i.val ≤ VALID_DECORATIONS.val.length ∧ decoration ∉ (VALID_DECORATIONS.val.take i.val)`. -/
theorem decoration_admitted_spec (decoration : Std.U32) :
    spirv.check.decoration_admitted decoration =
      ok (decide (decoration.val ∈ validDecorationsList)) := by
  unfold spirv.check.decoration_admitted spirv.check.decoration_admitted_loop
  have hlen : (Slice.len (Array.to_slice spirv.check.VALID_DECORATIONS)).val =
      spirv.check.VALID_DECORATIONS.val.length := by
    simp [Array.to_slice, Slice.from_val]
  have hspec : loop (fun i => spirv.check.decoration_admitted_loop.body decoration i) 0#usize
      ⦃ r => r = decide (decoration.val ∈ validDecorationsList) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Std.Usize) => spirv.check.VALID_DECORATIONS.val.length - i.val)
      (inv := fun (i : Std.Usize) =>
        i.val ≤ spirv.check.VALID_DECORATIONS.val.length ∧
        decoration ∉ (spirv.check.VALID_DECORATIONS.val.take i.val))
      (post := fun (r : Bool) => r = decide (decoration.val ∈ validDecorationsList))
    · intro i hi
      obtain ⟨hile, hnotin⟩ := hi
      by_cases hlt : i.val < spirv.check.VALID_DECORATIONS.val.length
      · have hlt' : i < Slice.len (Array.to_slice spirv.check.VALID_DECORATIONS) :=
          (UScalar.lt_equiv i _).2 (by rw [hlen]; exact hlt)
        have hidx : Array.index_usize spirv.check.VALID_DECORATIONS i =
            ok (spirv.check.VALID_DECORATIONS.val[i.val]'hlt) :=
          array_index_usize_eq _ i hlt
        by_cases heq : spirv.check.VALID_DECORATIONS.val[i.val]'hlt = decoration
        · have hbody : spirv.check.decoration_admitted_loop.body decoration i =
              ok (ControlFlow.done true) := by
            unfold spirv.check.decoration_admitted_loop.body
            simp only [lift, bind_tc_ok]
            rw [if_pos hlt', hidx]
            simp only [bind_tc_ok]
            rw [if_pos heq]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have hmem : decoration.val ∈ validDecorationsList := by
            rw [← VALID_DECORATIONS_val]
            exact List.mem_map.mpr ⟨decoration, heq ▸ List.getElem_mem hlt, rfl⟩
          simp [hmem]
        · have hnotin' : decoration ∉ (spirv.check.VALID_DECORATIONS.val.take (i.val + 1)) := by
            rw [List.take_succ_eq_append_getElem hlt]
            simp only [List.mem_append, List.mem_singleton, not_or]
            exact ⟨hnotin, fun h => heq h.symm⟩
          have hmax : i.val + (1#usize).val ≤ Usize.max := by
            have h1 : (1#usize).val = 1 := rfl
            have hlen6 : spirv.check.VALID_DECORATIONS.val.length = 6 := by simp
            have hilt6 : i.val < 6 := hlt.trans_eq hlen6
            have h6 : (6 : Nat) ≤ Usize.max := by scalar_tac
            rw [h1]; omega
          obtain ⟨inext, hinext, hinextv⟩ := Aeneas.Std.WP.spec_imp_exists
            (Usize.add_spec (x := i) (y := 1#usize) hmax)
          have hbody : spirv.check.decoration_admitted_loop.body decoration i =
              ok (ControlFlow.cont inext) := by
            unfold spirv.check.decoration_admitted_loop.body
            simp only [lift, bind_tc_ok]
            rw [if_pos hlt', hidx]
            simp only [bind_tc_ok]
            rw [if_neg heq, hinext]
            simp only [bind_tc_ok]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have h1 : (1#usize).val = 1 := rfl
          have hinextv' : inext.val = i.val + 1 := by rw [hinextv, h1]
          refine ⟨⟨by rw [hinextv']; omega, by rw [hinextv']; exact hnotin'⟩, by
            rw [hinextv']; omega⟩
      · have hbody : spirv.check.decoration_admitted_loop.body decoration i =
            ok (ControlFlow.done false) := by
          unfold spirv.check.decoration_admitted_loop.body
          simp only [lift, bind_tc_ok]
          rw [if_neg (by
            intro hc
            have hc' : i.val < (Slice.len (Array.to_slice spirv.check.VALID_DECORATIONS)).val :=
              (UScalar.lt_equiv i _).1 hc
            rw [hlen] at hc'
            exact hlt hc')]
        rw [hbody]
        simp only [Aeneas.Std.WP.spec_ok]
        have hnotmem : decoration.val ∉ validDecorationsList := by
          rw [← VALID_DECORATIONS_val]
          exact mem_map_val_of_not_mem (by
            rw [← List.take_of_length_le (l := spirv.check.VALID_DECORATIONS.val)
              (Nat.le_of_not_lt hlt)]
            exact hnotin)
        simp [hnotmem]
    · exact ⟨Nat.zero_le _, by simp⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hpost] at hr
  exact hr

/-- **`storage_class_admitted` spec.** The same scan over `VALID_STORAGE_CLASSES`, with the prefix
invariant `i.val ≤ VALID_STORAGE_CLASSES.val.length ∧
storage_class ∉ (VALID_STORAGE_CLASSES.val.take i.val)`. -/
theorem storage_class_admitted_spec (storage_class : Std.U32) :
    spirv.check.storage_class_admitted storage_class =
      ok (decide (storage_class.val ∈ validStorageClassesList)) := by
  unfold spirv.check.storage_class_admitted spirv.check.storage_class_admitted_loop
  have hlen : (Slice.len (Array.to_slice spirv.check.VALID_STORAGE_CLASSES)).val =
      spirv.check.VALID_STORAGE_CLASSES.val.length := by
    simp [Array.to_slice, Slice.from_val]
  have hspec : loop (fun i => spirv.check.storage_class_admitted_loop.body storage_class i) 0#usize
      ⦃ r => r = decide (storage_class.val ∈ validStorageClassesList) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Std.Usize) => spirv.check.VALID_STORAGE_CLASSES.val.length - i.val)
      (inv := fun (i : Std.Usize) =>
        i.val ≤ spirv.check.VALID_STORAGE_CLASSES.val.length ∧
        storage_class ∉ (spirv.check.VALID_STORAGE_CLASSES.val.take i.val))
      (post := fun (r : Bool) => r = decide (storage_class.val ∈ validStorageClassesList))
    · intro i hi
      obtain ⟨hile, hnotin⟩ := hi
      by_cases hlt : i.val < spirv.check.VALID_STORAGE_CLASSES.val.length
      · have hlt' : i < Slice.len (Array.to_slice spirv.check.VALID_STORAGE_CLASSES) :=
          (UScalar.lt_equiv i _).2 (by rw [hlen]; exact hlt)
        have hidx : Array.index_usize spirv.check.VALID_STORAGE_CLASSES i =
            ok (spirv.check.VALID_STORAGE_CLASSES.val[i.val]'hlt) :=
          array_index_usize_eq _ i hlt
        by_cases heq : spirv.check.VALID_STORAGE_CLASSES.val[i.val]'hlt = storage_class
        · have hbody : spirv.check.storage_class_admitted_loop.body storage_class i =
              ok (ControlFlow.done true) := by
            unfold spirv.check.storage_class_admitted_loop.body
            simp only [lift, bind_tc_ok]
            rw [if_pos hlt', hidx]
            simp only [bind_tc_ok]
            rw [if_pos heq]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have hmem : storage_class.val ∈ validStorageClassesList := by
            rw [← VALID_STORAGE_CLASSES_val]
            exact List.mem_map.mpr ⟨storage_class, heq ▸ List.getElem_mem hlt, rfl⟩
          simp [hmem]
        · have hnotin' : storage_class ∉
              (spirv.check.VALID_STORAGE_CLASSES.val.take (i.val + 1)) := by
            rw [List.take_succ_eq_append_getElem hlt]
            simp only [List.mem_append, List.mem_singleton, not_or]
            exact ⟨hnotin, fun h => heq h.symm⟩
          have hmax : i.val + (1#usize).val ≤ Usize.max := by
            have h1 : (1#usize).val = 1 := rfl
            have hlen4 : spirv.check.VALID_STORAGE_CLASSES.val.length = 4 := by simp
            have hilt4 : i.val < 4 := hlt.trans_eq hlen4
            have h4 : (4 : Nat) ≤ Usize.max := by scalar_tac
            rw [h1]; omega
          obtain ⟨inext, hinext, hinextv⟩ := Aeneas.Std.WP.spec_imp_exists
            (Usize.add_spec (x := i) (y := 1#usize) hmax)
          have hbody : spirv.check.storage_class_admitted_loop.body storage_class i =
              ok (ControlFlow.cont inext) := by
            unfold spirv.check.storage_class_admitted_loop.body
            simp only [lift, bind_tc_ok]
            rw [if_pos hlt', hidx]
            simp only [bind_tc_ok]
            rw [if_neg heq, hinext]
            simp only [bind_tc_ok]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have h1 : (1#usize).val = 1 := rfl
          have hinextv' : inext.val = i.val + 1 := by rw [hinextv, h1]
          refine ⟨⟨by rw [hinextv']; omega, by rw [hinextv']; exact hnotin'⟩, by
            rw [hinextv']; omega⟩
      · have hbody : spirv.check.storage_class_admitted_loop.body storage_class i =
            ok (ControlFlow.done false) := by
          unfold spirv.check.storage_class_admitted_loop.body
          simp only [lift, bind_tc_ok]
          rw [if_neg (by
            intro hc
            have hc' : i.val < (Slice.len (Array.to_slice spirv.check.VALID_STORAGE_CLASSES)).val :=
              (UScalar.lt_equiv i _).1 hc
            rw [hlen] at hc'
            exact hlt hc')]
        rw [hbody]
        simp only [Aeneas.Std.WP.spec_ok]
        have hnotmem : storage_class.val ∉ validStorageClassesList := by
          rw [← VALID_STORAGE_CLASSES_val]
          exact mem_map_val_of_not_mem (by
            rw [← List.take_of_length_le (l := spirv.check.VALID_STORAGE_CLASSES.val)
              (Nat.le_of_not_lt hlt)]
            exact hnotin)
        simp [hnotmem]
    · exact ⟨Nat.zero_le _, by simp⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hpost] at hr
  exact hr

/-! ## The pure image of `resolve_tag` -/

/-- The pure image of the extracted `resolve_tag`: `ty_get` becomes `firstLookup` over `tys.val`. -/
def resolveTagPure : spirv.check.TagSpec → List (Std.U32 × spirv.check.TyTag) → Option spirv.check.TyTag
  | .None, _ => none
  | .Void, _ => some .Void
  | .Bool, _ => some .Bool
  | .Int width, _ => some (.Int width)
  | .Float width, _ => some (.Float width)
  | .Vector elem, l => (firstLookup l elem).map (fun _ => .Vec)
  | .RuntimeArray elem, l => (firstLookup l elem).map (fun _ => .RuntimeArray)
  | .Pointer pointee, l => (firstLookup l pointee).map (fun _ => .Pointer)
  | .Function, _ => some .Function
  | .Struct, _ => some .Struct
  | .OfType ty, l => firstLookup l ty

/-- **`resolve_tag` spec.** The extracted resolver is `resolveTagPure`; the three resolving cases
read the type table through `ty_get_spec` (`firstLookup`), and `OfType` is that lookup itself. -/
theorem resolve_tag_spec (spec : spirv.check.TagSpec)
    (tys : Slice (Std.U32 × spirv.check.TyTag)) :
    spirv.check.resolve_tag spec tys = ok (resolveTagPure spec tys.val) := by
  cases spec with
  | None => rfl
  | Void => rfl
  | Bool => rfl
  | Int width => rfl
  | Float width => rfl
  | Vector elem =>
      simp only [spirv.check.resolve_tag, resolveTagPure, ty_get_spec, bind_tc_ok]
      cases h : firstLookup tys.val elem <;> simp [h]
  | RuntimeArray elem =>
      simp only [spirv.check.resolve_tag, resolveTagPure, ty_get_spec, bind_tc_ok]
      cases h : firstLookup tys.val elem <;> simp [h]
  | Pointer pointee =>
      simp only [spirv.check.resolve_tag, resolveTagPure, ty_get_spec, bind_tc_ok]
      cases h : firstLookup tys.val pointee <;> simp [h]
  | Function => rfl
  | Struct => rfl
  | OfType ty =>
      simp only [spirv.check.resolve_tag, resolveTagPure, ty_get_spec, bind_tc_ok]

/-! ## `ty_insert`: oldest-first overwrite-or-append -/

/-- Oldest-first overwrite-or-append: at the first entry whose key is `id`, replace its value;
otherwise append `(id, tag)`. -/
def tyInsertList : List (Std.U32 × spirv.check.TyTag) → Std.U32 → spirv.check.TyTag →
    List (Std.U32 × spirv.check.TyTag)
  | [], id, tag => [(id, tag)]
  | (k, v) :: t, id, tag => if k = id then (k, tag) :: t else (k, v) :: tyInsertList t id tag

theorem tyInsertList_prefix (l₁ l₂ : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (h : id ∉ l₁.map Prod.fst) :
    tyInsertList (l₁ ++ l₂) id tag = l₁ ++ tyInsertList l₂ id tag := by
  induction l₁ with
  | nil => simp
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.map_cons, List.mem_cons, not_or] at h
      obtain ⟨hk, ht⟩ := h
      simp only [List.cons_append, tyInsertList, if_neg (Ne.symm hk), ih ht]

theorem tyInsertList_of_not_mem (l : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (h : id ∉ l.map Prod.fst) :
    tyInsertList l id tag = l ++ [(id, tag)] := by
  induction l with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.map_cons, List.mem_cons, not_or] at h
      obtain ⟨hk, ht⟩ := h
      simp only [tyInsertList, if_neg (Ne.symm hk), ih ht, List.cons_append]

theorem tyInsertList_length_of_not_mem (l : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (h : id ∉ l.map Prod.fst) :
    (tyInsertList l id tag).length = l.length + 1 := by
  rw [tyInsertList_of_not_mem l id tag h, List.length_append]
  simp

theorem tyInsertList_length_of_mem (l : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (h : id ∈ l.map Prod.fst) :
    (tyInsertList l id tag).length = l.length := by
  induction l with
  | nil => simp at h
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.map_cons, List.mem_cons] at h
      by_cases hk : k = id
      · simp [tyInsertList, hk]
      · have ht : id ∈ t.map Prod.fst := h.resolve_left (fun h' => hk h'.symm)
        simp only [tyInsertList, if_neg hk, List.length_cons, ih ht]

/-- Replacing the element at index `i` splits the list at `i`. -/
theorem set_eq_take_append_drop {α : Type} (l : List α) (i : Nat) (x : α) (h : i < l.length) :
    l.take i ++ x :: l.drop (i + 1) = l.set i x := by
  induction l generalizing i with
  | nil => simp at h
  | cons a t ih =>
      cases i with
      | zero => rfl
      | succ k =>
          simp only [List.take_succ_cons, List.drop_succ_cons, List.set_cons_succ, List.cons_append]
          congr 1
          exact ih k (by simpa using h)

/-- With no earlier key match, `tyInsertList` overwrites the first match in place. -/
theorem tyInsertList_eq_set_of_first (l : List (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (i : Nat) (hlt : i < l.length)
    (hnotin : id ∉ (l.take i).map Prod.fst) (heq : (l[i]'hlt).1 = id) :
    tyInsertList l id tag = l.set i ((l[i]'hlt).1, tag) := by
  conv_lhs => rw [← List.take_append_drop i l]
  rw [tyInsertList_prefix (l₁ := l.take i) (l₂ := l.drop i) id tag hnotin]
  rw [List.drop_eq_getElem_cons hlt]
  simp only [tyInsertList]
  rw [if_pos heq]
  rw [set_eq_take_append_drop l i ((l[i]'hlt).1, tag) hlt]

/-- The pure image of `ty_insert`: the found branch keeps `cap` and `tyInsertList`'s value, the
absent branch is the push result (`tyInsertList` equals `l ++ [(id, tag)]` there and `Vec.from`
resets `cap` to the wrapped length). `hlen` only discharges the push arm's bound. -/
def tyInsertVec (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (hlen : tys.val.length < Usize.max) :
    alloc.vec.Vec (Std.U32 × spirv.check.TyTag) :=
  if h : id ∈ tys.val.map Prod.fst then
    { slice := Slice.from (tyInsertList tys.val id tag)
        (by rw [tyInsertList_length_of_mem tys.val id tag h]; exact tys.property)
    , cap := tys.cap }
  else
    alloc.vec.Vec.from (tys.val ++ [(id, tag)])
      (by rw [List.length_append, List.length_singleton]; omega)

/-! ## The extracted `Vec` steps -/

theorem vec_index_usize_eq {α : Type} (v : alloc.vec.Vec α) (i : Usize)
    (h : i.val < v.val.length) : v.index_usize i = ok (v.val[i.val]'h) := by
  unfold alloc.vec.Vec.index_usize
  simp [alloc.vec.Vec.getElem?_Usize_eq, List.getElem?_eq_getElem h]

theorem vec_index_eq {α : Type} (v : alloc.vec.Vec α) (i : Usize)
    (h : i.val < v.val.length) :
    alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v i =
      ok (v.val[i.val]'h) := by
  rw [alloc.vec.Vec.index_slice_index]
  exact vec_index_usize_eq v i h

theorem vec_index_mut_eq {α : Type} (v : alloc.vec.Vec α) (i : Usize)
    (h : i.val < v.val.length) :
    alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i =
      ok (v.val[i.val]'h, v.set i) := by
  rw [alloc.vec.Vec.index_mut_slice_index]
  unfold alloc.vec.Vec.index_mut_usize
  rw [vec_index_usize_eq v i h]
  simp only [bind_tc_ok]

theorem vec_push_eq {α : Type} (v : alloc.vec.Vec α) (x : α) (h : v.val.length < Usize.max) :
    alloc.vec.Vec.push v x =
      ok (alloc.vec.Vec.from (v.val ++ [x])
        (by have := alloc.vec.Vec.property v; simp; omega)) := by
  unfold alloc.vec.Vec.push
  dsimp only
  rw [dif_pos (by
    have h' : List.length v.val + 1 ≤ Usize.max := by omega
    simp [h'])]
  congr 1
  rw [alloc.vec.Vec.eq_iff]
  constructor
  · simp [alloc.vec.Vec.from_val, List.concat_eq_append]
  · simp [alloc.vec.Vec.from]

/-! ## The `ty_insert` scan -/

/-- **`ty_insert` spec.** The extracted `ty_insert` is `tyInsertVec`: a found key is overwritten in
place with `cap` preserved (no refusal arm), an absent key is appended with `cap` the wrapped
length. `loop.spec_decr_nat` on `i.val ≤ tys.val.length ∧ id ∉ (tys.val.take i.val)`; `hlen` is the
push arm's bound, the only hypothesis beyond the arm conditions. -/
theorem ty_insert_spec (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (tag : spirv.check.TyTag) (hlen : tys.val.length < Usize.max) :
    spirv.check.ty_insert tys id tag = ok (tyInsertVec tys id tag hlen) := by
  unfold spirv.check.ty_insert spirv.check.ty_insert_loop
  have hspec : loop (fun i => spirv.check.ty_insert_loop.body tys id tag i) 0#usize
      ⦃ r => r = tyInsertVec tys id tag hlen ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Std.Usize) => tys.val.length - i.val)
      (inv := fun (i : Std.Usize) =>
        i.val ≤ tys.val.length ∧ id ∉ (tys.val.take i.val).map Prod.fst)
      (post := fun (r : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) =>
        r = tyInsertVec tys id tag hlen)
    · intro i hi
      obtain ⟨hile, hnotin⟩ := hi
      by_cases hlt : i.val < tys.val.length
      · have hlt' : i < alloc.vec.Vec.len tys :=
          (UScalar.lt_equiv i _).2 (by rw [alloc.vec.Vec.len_val]; exact hlt)
        have hidx : alloc.vec.Vec.index
              (core.slice.index.SliceIndexUsizeSlice (Std.U32 × spirv.check.TyTag)) tys i =
            ok (tys.val[i.val]'hlt) := vec_index_eq tys i hlt
        by_cases heq : (tys.val[i.val]'hlt).1 = id
        · have hidxmut : alloc.vec.Vec.index_mut
                (core.slice.index.SliceIndexUsizeSlice (Std.U32 × spirv.check.TyTag)) tys i =
              ok (tys.val[i.val]'hlt, tys.set i) := vec_index_mut_eq tys i hlt
          have hbody : spirv.check.ty_insert_loop.body tys id tag i =
              ok (ControlFlow.done (tys.set i ((tys.val[i.val]'hlt).1, tag))) := by
            dsimp only [spirv.check.ty_insert_loop.body]
            rw [if_pos hlt']
            rw [hidx]
            simp only [bind_tc_ok]
            show (if (tys.val[i.val]'hlt).1 = id then
                    (do let ((i3, _), index_mut_back) ← alloc.vec.Vec.index_mut
                          (core.slice.index.SliceIndexUsizeSlice (Std.U32 × spirv.check.TyTag)) tys i
                        ok (ControlFlow.done (index_mut_back (i3, tag))))
                  else
                    (do let i3 ← i + 1#usize
                        ok (ControlFlow.cont i3))) =
              ok (ControlFlow.done (tys.set i ((tys.val[i.val]'hlt).1, tag)))
            rw [if_pos heq]
            rw [hidxmut]
            simp only [bind_tc_ok]
            rw [← Prod.mk.eta (p := tys.val[i.val]'hlt)]
            rfl
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have hmem : id ∈ tys.val.map Prod.fst :=
            List.mem_map.mpr ⟨tys.val[i.val]'hlt, List.getElem_mem hlt, heq⟩
          unfold tyInsertVec
          rw [dif_pos hmem]
          exact alloc.vec.Vec.ext _ _
            (by rw [alloc.vec.Vec.set_val_eq]
                simp only [alloc.vec.Vec.val, Slice.from_val]
                exact (tyInsertList_eq_set_of_first tys.val id tag i.val hlt hnotin heq).symm)
            (by rfl)
        · have hnotin' : id ∉ (tys.val.take (i.val + 1)).map Prod.fst := by
            intro hc
            obtain ⟨kv, hkv, hkvid⟩ := List.mem_map.mp hc
            rw [List.take_succ_eq_append_getElem hlt] at hkv
            rcases List.mem_append.mp hkv with h1 | h2
            · exact hnotin (List.mem_map.mpr ⟨kv, h1, hkvid⟩)
            · rw [List.mem_singleton] at h2
              exact heq (h2 ▸ hkvid)
          have hmax : i.val + (1#usize).val ≤ Usize.max := by
            have h1 : (1#usize).val = 1 := rfl
            have hp := alloc.vec.Vec.property tys
            rw [h1]; omega
          obtain ⟨inext, hinext, hinextv⟩ := Aeneas.Std.WP.spec_imp_exists
            (Usize.add_spec (x := i) (y := 1#usize) hmax)
          have hbody : spirv.check.ty_insert_loop.body tys id tag i =
              ok (ControlFlow.cont inext) := by
            dsimp only [spirv.check.ty_insert_loop.body]
            rw [if_pos hlt']
            rw [hidx]
            simp only [bind_tc_ok]
            show (if (tys.val[i.val]'hlt).1 = id then
                    (do let ((i3, _), index_mut_back) ← alloc.vec.Vec.index_mut
                          (core.slice.index.SliceIndexUsizeSlice (Std.U32 × spirv.check.TyTag)) tys i
                        ok (ControlFlow.done (index_mut_back (i3, tag))))
                  else
                    (do let i3 ← i + 1#usize
                        ok (ControlFlow.cont i3))) =
              ok (ControlFlow.cont inext)
            rw [if_neg heq, hinext]
            simp only [bind_tc_ok]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have h1 : (1#usize).val = 1 := rfl
          have hinextv' : inext.val = i.val + 1 := by rw [hinextv, h1]
          exact ⟨⟨by rw [hinextv']; omega, by rw [hinextv']; exact hnotin'⟩,
            by rw [hinextv']; omega⟩
      · have hnotmem : id ∉ tys.val.map Prod.fst := by
          rw [← List.take_of_length_le (l := tys.val) (Nat.le_of_not_lt hlt)]
          exact hnotin
        have hpush : alloc.vec.Vec.push tys (id, tag) =
            ok (alloc.vec.Vec.from (tys.val ++ [(id, tag)])
              (by have := alloc.vec.Vec.property tys; simp; omega)) :=
          vec_push_eq tys (id, tag) hlen
        have hbody : spirv.check.ty_insert_loop.body tys id tag i =
            ok (ControlFlow.done (alloc.vec.Vec.from (tys.val ++ [(id, tag)])
              (by have := alloc.vec.Vec.property tys; simp; omega))) := by
          dsimp only [spirv.check.ty_insert_loop.body]
          rw [if_neg (by
            intro hc
            have hc' : i.val < (alloc.vec.Vec.len tys).val := (UScalar.lt_equiv i _).1 hc
            rw [alloc.vec.Vec.len_val] at hc'
            exact hlt hc')]
          rw [hpush]
          simp only [bind_tc_ok]
        rw [hbody]
        simp only [Aeneas.Std.WP.spec_ok]
        unfold tyInsertVec
        rw [dif_neg hnotmem]
    · exact ⟨Nat.zero_le _, by simp⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hpost] at hr
  exact hr

/-! ## One `apply_shape_loop0` body step -/

/-- `Vec.deref` reads through to `defs.val`, so the extracted membership scan is the pure
membership on `defs.val`. -/
theorem defs_contains_vec_deref (defs : alloc.vec.Vec Std.U32) (id : Std.U32)
    (h : id ∈ defs.val) :
    spirv.check.defs_contains (alloc.vec.Vec.deref defs) id = ok true := by
  rw [defs_contains_spec]
  simp [alloc.vec.Vec.deref, Slice.from_val, h]

/-- **`apply_shape_loop0` defined-used-id step.** With the used id in bounds and defined in `defs`,
the body returns the continuation at `u + 1` (`defs`/`tys` unchanged). -/
theorem apply_shape_loop0_body_cont (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (u : Usize)
    (hlt : u.val < v.val.length) (hdef : v.val[u.val]'hlt ∈ defs.val) :
    spirv.check.apply_shape_loop0.body index o v ts defs tys u =
      (do let u1 ← u + 1#usize
          ok (ControlFlow.cont u1)) := by
  dsimp only [spirv.check.apply_shape_loop0.body]
  rw [if_pos ((UScalar.lt_equiv u _).2 (by rw [alloc.vec.Vec.len_val]; exact hlt))]
  rw [vec_index_eq v u hlt]
  simp only [bind_tc_ok]
  rw [defs_contains_vec_deref defs (v.val[u.val]'hlt) hdef]
  simp only [bind_tc_ok]
  simp

/-- **`apply_shape_loop0` undefined-used-id step.** With the used id in bounds but absent from
`defs`, the body returns `CheckError.UndefinedId index used` and leaves `defs`/`tys` unchanged. -/
theorem apply_shape_loop0_body_undefined (index : Std.Usize) (o : Option Std.U32)
    (v : alloc.vec.Vec Std.U32) (ts : spirv.check.TagSpec) (defs : alloc.vec.Vec Std.U32)
    (tys : alloc.vec.Vec (Std.U32 × spirv.check.TyTag)) (u : Usize)
    (hlt : u.val < v.val.length) (hnot : v.val[u.val]'hlt ∉ defs.val) :
    spirv.check.apply_shape_loop0.body index o v ts defs tys u =
      ok (ControlFlow.done (core.result.Result.Err
        (spirv.check.CheckError.UndefinedId index (v.val[u.val]'hlt)), defs, tys)) := by
  dsimp only [spirv.check.apply_shape_loop0.body]
  rw [if_pos ((UScalar.lt_equiv u _).2 (by rw [alloc.vec.Vec.len_val]; exact hlt))]
  rw [vec_index_eq v u hlt]
  simp only [bind_tc_ok]
  have hdc : spirv.check.defs_contains (alloc.vec.Vec.deref defs) (v.val[u.val]'hlt) =
      ok false := by
    rw [defs_contains_spec]
    simp [alloc.vec.Vec.deref, Slice.from_val, hnot]
  rw [hdc]
  simp only [bind_tc_ok]
  simp

end Spirv
