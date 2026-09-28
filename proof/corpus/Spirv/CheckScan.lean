import Spirv.Check

/-!
# Checker scan lookup lemmas

The linear scans `check.rs::ty_get` and `check.rs::defs_contains` (`slice_scan` in
`proof/ApeironWord.lean`, lines around 11192–11360) walk their slice oldest entry first and stop at
the first key match. This module proves that reading:

* `ty_get_spec` — the extracted `spirv.check.ty_get` returns exactly `firstLookup tys.val id`, the
  pure left-to-right association-list lookup, via `loop.spec_decr_nat` on the prefix invariant
  `i.val ≤ tys.val.length ∧ id ∉ (tys.val.take i.val).map Prod.fst`;
* `defs_contains_spec` — the same scan over a `Slice Std.U32` returns
  `decide (id ∈ defs.val)`;
* `firstLookup_reverse_lookupTy` — the pure scan-order lemma: on a key-`Nodup` list the oldest-first
  `firstLookup` agrees with `lookupTy` over the reversed list (`Spirv.Decode.lookupTy`, newest first,
  `Decode.lean` around line 223);
* `ty_get_eq_lookupTy_of_nodup` — the same agreement for `ty_get` under an explicit tag translation
  `τ : spirv.check.TyTag → Spirv.TyTag`; key agreement uses `UScalar.val`'s injectivity.

## Residual obligation (not proved here)

No `TyTag`/key correspondence between the extracted `spirv.check.TyTag`/`Std.U32` and the decoder's
`Spirv.TyTag`/`SpvId` exists in `Check.lean` or `Decode.lean`; `ty_get_eq_lookupTy_of_nodup` therefore
carries the value translation `τ` as a parameter and maps through `.val` on keys. The established
correspondence must supply `τ` and show it commutes with `resultTag`/`stepInst`. Also open: the
`check_fragment_loop` scan invariant `defs.toList = st.defs.reverse` and the composition to
`decode ws = some m`.
-/

namespace Spirv

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result
open apeiron_word

/-! ## The oldest-first association-list lookup -/

/-- Left-to-right association-list lookup: the value of the first entry whose key is `x`, `none` if
`x` is absent. `Spirv.lookupTy` scans the same list right-to-left. -/
def firstLookup {α β : Type} [DecidableEq α] : List (α × β) → α → Option β
  | [], _ => none
  | (k, v) :: t, x => if k = x then some v else firstLookup t x

theorem firstLookup_append {α β : Type} [DecidableEq α] (l₁ l₂ : List (α × β)) (x : α) :
    firstLookup (l₁ ++ l₂) x =
      (match firstLookup l₁ x with | some v => some v | none => firstLookup l₂ x) := by
  induction l₁ with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.cons_append, firstLookup]
      by_cases h : k = x
      · simp [h]
      · simp [h, ih]

/-- A key absent from the whole list is absent from the scan: the lookup is `none`. -/
theorem firstLookup_of_not_mem {α β : Type} [DecidableEq α] {l : List (α × β)} {x : α}
    (h : x ∉ l.map Prod.fst) : firstLookup l x = none := by
  induction l with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.map_cons, List.mem_cons, not_or] at h
      obtain ⟨hk, ht⟩ := h
      simp only [firstLookup]
      rw [if_neg (Ne.symm hk), ih ht]

/-- With the prefix key-free, the scan stops at the first occurrence: the value at index `i`. -/
theorem firstLookup_eq_some_of_lt {α β : Type} [DecidableEq α] {l : List (α × β)} {x : α}
    {i : Nat} (hlt : i < l.length) (hnotin : x ∉ (l.take i).map Prod.fst)
    (heq : (l[i]'hlt).1 = x) : firstLookup l x = some (l[i]'hlt).2 := by
  have htake : firstLookup (l.take i) x = none := firstLookup_of_not_mem hnotin
  conv_lhs => rw [← List.take_append_drop i l]
  rw [firstLookup_append, htake]
  rw [List.drop_eq_getElem_cons hlt]
  simp only [firstLookup]
  rw [if_pos heq]

/-- A key absent from the scanned prefix is absent from the whole list when the prefix is the
whole list: the lookup is `none`. -/
theorem firstLookup_eq_none_of_ge {α β : Type} [DecidableEq α] {l : List (α × β)} {x : α}
    {i : Nat} (hlen : l.length ≤ i) (hnotin : x ∉ (l.take i).map Prod.fst) :
    firstLookup l x = none := by
  have htake : l.take i = l := List.take_of_length_le hlen
  rw [← htake]
  exact firstLookup_of_not_mem hnotin

/-! ## Scan-order agreement with `Spirv.lookupTy` -/

theorem lookupTy_append (l₁ l₂ : List (SpvId × TyTag)) (x : SpvId) :
    lookupTy (l₁ ++ l₂) x =
      (match lookupTy l₁ x with | some v => some v | none => lookupTy l₂ x) := by
  induction l₁ with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.cons_append, lookupTy]
      by_cases h : k = x
      · simp [h]
      · simp [h, ih]

theorem lookupTy_of_not_mem {l : List (SpvId × TyTag)} {x : SpvId}
    (h : x ∉ l.map Prod.fst) : lookupTy l x = none := by
  induction l with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.map_cons, List.mem_cons, not_or] at h
      obtain ⟨hk, ht⟩ := h
      simp only [lookupTy]
      rw [if_neg (Ne.symm hk), ih ht]

/-- **Scan-order lemma.** On a list with pairwise-distinct keys, the oldest-first `firstLookup` and
the newest-first `Spirv.lookupTy` over the reversed list return the same value: uniqueness makes the
order immaterial. -/
theorem firstLookup_reverse_lookupTy (l : List (SpvId × TyTag))
    (hnodup : (l.map Prod.fst).Nodup) (x : SpvId) :
    firstLookup l x = lookupTy l.reverse x := by
  induction l with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      rw [List.map_cons, List.nodup_cons] at hnodup
      obtain ⟨hk, ht⟩ := hnodup
      rw [List.reverse_cons, lookupTy_append, ← ih ht]
      simp only [firstLookup, lookupTy]
      by_cases hkx : k = x
      · have hxk : x ∉ t.map Prod.fst := fun hmem => hk (hkx.symm ▸ hmem)
        simp [firstLookup_of_not_mem hxk, hkx]
      · cases hft : firstLookup t x <;> simp [hkx]

/-! ## `ty_get` under a tag translation -/

/-- Keys of a `Std.U32`-keyed list stay distinct after reading them as `Nat`. -/
theorem nodup_val_of_nodup_fst {β : Type} {l : List (Std.U32 × β)}
    (h : (l.map Prod.fst).Nodup) : (l.map (fun kv => kv.1.val)).Nodup := by
  have hmap : l.map (fun kv => kv.1.val) =
      (l.map Prod.fst).map (fun u : Std.U32 => u.val) := by
    rw [List.map_map]; rfl
  rw [hmap]
  exact List.Nodup.map (fun _ _ hv => UScalar.eq_of_val_eq hv) h

/-- Reading the key as `.val` and the value through `f` commutes with the oldest-first scan. -/
theorem firstLookup_map_key_val {β γ : Type} (f : β → γ) (l : List (Std.U32 × β)) (x : Std.U32) :
    (firstLookup l x).map f =
      firstLookup (l.map (fun kv => (kv.1.val, f kv.2))) x.val := by
  induction l with
  | nil => rfl
  | cons kv t ih =>
      obtain ⟨k, v⟩ := kv
      simp only [List.map_cons, firstLookup]
      by_cases h : k = x
      · simp [h]
      · have h' : k.val ≠ x.val := fun hv => h (UScalar.eq_of_val_eq hv)
        rw [if_neg h, if_neg h']
        exact ih

/-! ## The extracted scans -/

/-- In-bounds `Slice.index_usize` reads the underlying list. -/
theorem index_usize_eq {α : Type} (s : Slice α) (i : Usize)
    (h : i.val < s.val.length) : Slice.index_usize s i = ok (s.val[i.val]'h) := by
  obtain ⟨x, hx, hxv⟩ := Aeneas.Std.WP.spec_imp_exists (Slice.index_usize_spec s i h)
  rw [hxv] at hx
  exact hx

/-- **`ty_get` spec.** The extracted `spirv.check.ty_get` is the oldest-first pure lookup.
Proved by `loop.spec_decr_nat` on the prefix invariant `i.val ≤ tys.val.length` plus
`id ∉ (tys.val.take i.val).map Prod.fst`; the measure is `tys.val.length - i.val`. -/
theorem ty_get_spec (tys : Slice (Std.U32 × spirv.check.TyTag)) (id : Std.U32) :
    spirv.check.ty_get tys id = ok (firstLookup tys.val id) := by
  unfold spirv.check.ty_get spirv.check.ty_get_loop
  have hspec : loop (fun i => spirv.check.ty_get_loop.body tys id i) 0#usize
      ⦃ r => r = firstLookup tys.val id ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Std.Usize) => tys.val.length - i.val)
      (inv := fun (i : Std.Usize) =>
        i.val ≤ tys.val.length ∧ id ∉ (tys.val.take i.val).map Prod.fst)
      (post := fun (r : Option spirv.check.TyTag) => r = firstLookup tys.val id)
    · intro i hi
      obtain ⟨hile, hnotin⟩ := hi
      by_cases hlt : i.val < tys.val.length
      · have hidx : Slice.index_usize tys i = ok (tys.val[i.val]'hlt) := index_usize_eq tys i hlt
        have hmax : i.val + (1#usize).val ≤ Usize.max := by
          have h1 : (1#usize).val = 1 := rfl
          have hp := tys.property
          rw [h1]; omega
        obtain ⟨inext, hinext, hinextv⟩ := Aeneas.Std.WP.spec_imp_exists
          (Usize.add_spec (x := i) (y := 1#usize) hmax)
        have hlt' : i < Slice.len tys :=
          (UScalar.lt_equiv i (Slice.len tys)).2 (by rw [Slice.len_val]; exact hlt)
        by_cases heq : (tys.val[i.val]'hlt).1 = id
        · have hbody : spirv.check.ty_get_loop.body tys id i =
              ok (ControlFlow.done (some (tys.val[i.val]'hlt).2)) := by
            unfold spirv.check.ty_get_loop.body
            rw [if_pos hlt', hidx, hinext]
            simp only [bind_tc_ok]
            show (if (tys.val[i.val]'hlt).1 = id
                then ok (ControlFlow.done (some (tys.val[i.val]'hlt).2))
                else ok (ControlFlow.cont inext)) =
              ok (ControlFlow.done (some (tys.val[i.val]'hlt).2))
            rw [if_pos heq]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          exact (firstLookup_eq_some_of_lt (l := tys.val) (x := id) hlt hnotin heq).symm
        · have hnotin' : id ∉ (tys.val.take (i.val + 1)).map Prod.fst := by
            have htake : (tys.val.take (i.val + 1)).map Prod.fst =
                (tys.val.take i.val).map Prod.fst ++ [(tys.val[i.val]'hlt).1] := by
              rw [List.take_succ_eq_append_getElem hlt, List.map_append]; rfl
            rw [htake]
            simp only [List.mem_append, List.mem_singleton, not_or]
            exact ⟨hnotin, fun h => heq h.symm⟩
          have hbody : spirv.check.ty_get_loop.body tys id i =
              ok (ControlFlow.cont inext) := by
            unfold spirv.check.ty_get_loop.body
            rw [if_pos hlt', hidx, hinext]
            simp only [bind_tc_ok]
            show (if (tys.val[i.val]'hlt).1 = id
                then ok (ControlFlow.done (some (tys.val[i.val]'hlt).2))
                else ok (ControlFlow.cont inext)) = ok (ControlFlow.cont inext)
            rw [if_neg heq]
          have h1 : (1#usize).val = 1 := rfl
          have hinextv' : inext.val = i.val + 1 := by rw [hinextv, h1]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          refine ⟨⟨by rw [hinextv']; omega, by rw [hinextv']; exact hnotin'⟩,
            by rw [hinextv']; omega⟩
      · have hbody : spirv.check.ty_get_loop.body tys id i = ok (ControlFlow.done none) := by
          unfold spirv.check.ty_get_loop.body
          rw [if_neg (by
            intro hc
            have hc' : i.val < (Slice.len tys).val := (UScalar.lt_equiv i (Slice.len tys)).1 hc
            rw [Slice.len_val] at hc'
            exact hlt hc')]
        rw [hbody]
        simp only [Aeneas.Std.WP.spec_ok]
        exact (firstLookup_eq_none_of_ge (l := tys.val) (x := id) (i := i.val)
          (Nat.le_of_not_lt hlt) hnotin).symm
    · exact ⟨Nat.zero_le _, by simp⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hpost] at hr
  exact hr

/-- **`defs_contains` spec.** The extracted `spirv.check.defs_contains` is
`decide (id ∈ defs.val)`, by `loop.spec_decr_nat` on the prefix invariant
`i.val ≤ defs.val.length` plus `id ∉ defs.val.take i.val`. -/
theorem defs_contains_spec (defs : Slice Std.U32) (id : Std.U32) :
    spirv.check.defs_contains defs id = ok (decide (id ∈ defs.val)) := by
  unfold spirv.check.defs_contains spirv.check.defs_contains_loop
  have hspec : loop (fun i => spirv.check.defs_contains_loop.body defs id i) 0#usize
      ⦃ r => r = decide (id ∈ defs.val) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun (i : Std.Usize) => defs.val.length - i.val)
      (inv := fun (i : Std.Usize) => i.val ≤ defs.val.length ∧ id ∉ defs.val.take i.val)
      (post := fun (r : Bool) => r = decide (id ∈ defs.val))
    · intro i hi
      obtain ⟨hile, hnotin⟩ := hi
      by_cases hlt : i.val < defs.val.length
      · have hidx : Slice.index_usize defs i = ok (defs.val[i.val]'hlt) := index_usize_eq defs i hlt
        have hmax : i.val + (1#usize).val ≤ Usize.max := by
          have h1 : (1#usize).val = 1 := rfl
          have hp := defs.property
          rw [h1]; omega
        obtain ⟨inext, hinext, hinextv⟩ := Aeneas.Std.WP.spec_imp_exists
          (Usize.add_spec (x := i) (y := 1#usize) hmax)
        have hlt' : i < Slice.len defs :=
          (UScalar.lt_equiv i (Slice.len defs)).2 (by rw [Slice.len_val]; exact hlt)
        by_cases heq : defs.val[i.val]'hlt = id
        · have hbody : spirv.check.defs_contains_loop.body defs id i =
              ok (ControlFlow.done true) := by
            unfold spirv.check.defs_contains_loop.body
            rw [if_pos hlt', hidx, hinext]
            simp only [bind_tc_ok]
            simp [heq]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          have hmem : id ∈ defs.val := by
            rw [← List.take_append_drop i.val defs.val, List.drop_eq_getElem_cons hlt]
            simp only [List.mem_append, List.mem_cons]
            exact Or.inr (Or.inl heq.symm)
          simp [hmem]
        · have hnotin' : id ∉ defs.val.take (i.val + 1) := by
            rw [List.take_succ_eq_append_getElem hlt]
            simp only [List.mem_append, List.mem_singleton, not_or]
            exact ⟨hnotin, fun h => heq h.symm⟩
          have hbody : spirv.check.defs_contains_loop.body defs id i =
              ok (ControlFlow.cont inext) := by
            unfold spirv.check.defs_contains_loop.body
            rw [if_pos hlt', hidx, hinext]
            simp only [bind_tc_ok]
            simp [heq]
          have h1 : (1#usize).val = 1 := rfl
          have hinextv' : inext.val = i.val + 1 := by rw [hinextv, h1]
          rw [hbody]
          simp only [Aeneas.Std.WP.spec_ok]
          refine ⟨⟨by rw [hinextv']; omega, by rw [hinextv']; exact hnotin'⟩,
            by rw [hinextv']; omega⟩
      · have hbody : spirv.check.defs_contains_loop.body defs id i = ok (ControlFlow.done false) := by
          unfold spirv.check.defs_contains_loop.body
          rw [if_neg (by
            intro hc
            have hc' : i.val < (Slice.len defs).val := (UScalar.lt_equiv i (Slice.len defs)).1 hc
            rw [Slice.len_val] at hc'
            exact hlt hc')]
        rw [hbody]
        simp only [Aeneas.Std.WP.spec_ok]
        have hnotmem : id ∉ defs.val := by
          rw [← List.take_of_length_le (l := defs.val) (Nat.le_of_not_lt hlt)]
          exact hnotin
        simp [hnotmem]
    · exact ⟨Nat.zero_le _, by simp⟩
  obtain ⟨r, hr, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hspec
  rw [hpost] at hr
  exact hr

/-- **`ty_get` scan agreement.** With `tys` key-unique, the extracted `ty_get` returns the value the
decoder's newest-first `lookupTy` returns over the reversed, `τ`-translated list, read through `τ`.
`τ` is the unestablished `spirv.check.TyTag → Spirv.TyTag` correspondence (see the module header). -/
theorem ty_get_eq_lookupTy_of_nodup
    (tys : Slice (Std.U32 × spirv.check.TyTag)) (id : Std.U32)
    (τ : spirv.check.TyTag → Spirv.TyTag)
    (hnodup : (tys.val.map Prod.fst).Nodup) :
    (do let o ← spirv.check.ty_get tys id; ok (Option.map τ o)) =
      ok (lookupTy ((tys.val.map (fun kv => (kv.1.val, τ kv.2))).reverse) id.val) := by
  rw [ty_get_spec]
  simp only [bind_tc_ok]
  congr 1
  rw [firstLookup_map_key_val]
  have hnodup' : ((tys.val.map (fun kv => (kv.1.val, τ kv.2))).map Prod.fst).Nodup := by
    have hmap : (tys.val.map (fun kv => (kv.1.val, τ kv.2))).map Prod.fst =
        tys.val.map (fun kv => kv.1.val) := by
      rw [List.map_map]; rfl
    rw [hmap]
    exact nodup_val_of_nodup_fst (l := tys.val) hnodup
  exact firstLookup_reverse_lookupTy _ hnodup' id.val

end Spirv
