-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (HT-INSERT-W7): totality of the
-- executed insert and resize.
import Map.ExecutedResize
import Map.ExecutedBuild
import Map.ExecutedReadTotal
import Map.ExecutedReadCoordTotal
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed insert and resize

`insert_refines` / `coord_insert_refines` (`Map/ExecutedResize.lean`) take the premise
`word_hashtrinity_insert s k v = ok s'`.  This module discharges that premise: under the
representation the refinement already carries, the executed bodies return `ok`.

The technique is read totality (`Map/ExecutedReadTotal.lean`): walk the extracted body with the
representation invariant, closing each `fail`, `panic` and checked-arithmetic branch.

* Stage 1 (`resize_ok`): `Table.resize` returns `ok` under `RepW ∧ KC ∧ arenaLen + mainCount < nilN`
  and a bound on the bit count.
* Stage 2 (`grow_for_one_ok`): `grow_for_one` returns `ok` under the same premises.
* Stage 3 (`insert_ok` / `coord_insert_ok`): `∃ s', word_hashtrinity_insert s k v = ok s'`.
* Stage 4 (`insert_refines_total` / `execBuild_ok`): the refinement conclusion without the `ok`
  premise, and totality of the executed build.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-! ## Index totality -/

/-- Checked `Vec::index_mut` returns `ok` on an in-range index. -/
theorem vec_index_mut_ex {α : Type} (v : alloc.vec.Vec α) (i : Usize) (h : i.val < v.val.length) :
    ∃ x back, alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice α) v i = ok (x, back) := by
  rw [alloc.vec.Vec.index_mut_slice_index]
  obtain ⟨p, hp, -⟩ := WP.spec_imp_exists (alloc.vec.Vec.index_mut_usize_spec v i h)
  exact ⟨p.1, p.2, hp⟩

/-- A `u32` value is below `2^32`. -/
theorem u32_val_lt (x : U32) : x.val < 2 ^ 32 := by
  have := x.hBounds
  rwa [UScalarTy.U32_numBits_eq] at this

/-! ## `h` and `update` totality -/

/-- `Overflow::h` is total on an embedded tree. -/
theorem ov_h_ok (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (x : U32) (t : OTree)
    (hE : Emb L x t) :
    ∃ r, hashtrinity.overflow.Overflow.h CI o x = ok r := by
  unfold hashtrinity.overflow.Overflow.h
  by_cases hx : x = hashtrinity.overflow.NIL
  · rw [if_pos hx]
    exact ⟨0#u32, rfl⟩
  · rw [if_neg hx]
    cases t with
    | leaf => exact absurd ((NIL_eq_iff x).mpr hE) hx
    | node l i r =>
      simp only [lift, bind_tc_ok]
      obtain ⟨hxi, _, nd, hnd, _, _⟩ := hE
      have hxlt : x.val < L.length := by rw [hxi]; exact (List.getElem?_eq_some_iff.mp hnd).1
      obtain ⟨n, hn⟩ := vec_index_ex o.nodes (UScalar.cast .Usize x)
        (by rw [cast_usize_u32, hL]; exact hxlt)
      rw [hn]
      simp only [bind_tc_ok]
      exact ⟨_, rfl⟩

/-- `Overflow::update` is total on an embedded node whose stored height is current. -/
theorem ov_update_ok (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (i : U32)
    (l r : OTree) (ii : Nat) (hE : Emb L i (.node l ii r)) (hH : HOK L (.node l ii r)) :
    ∃ o', hashtrinity.overflow.Overflow.update CI o i = ok o' := by
  obtain ⟨hiv, hii, nd, hnd, hEl, hEr⟩ := hE
  obtain ⟨hl, hr, ndH, hndH, hndHv⟩ := hH
  have hilt : i.val < L.length := by rw [hiv]; exact (List.getElem?_eq_some_iff.mp hnd).1
  have hcastl : (UScalar.cast .Usize i).val < o.nodes.val.length := by
    rw [cast_usize_u32, hL]; exact hilt
  have hmax : max (ht l) (ht r) + 1 ≤ 2 ^ 32 - 1 := by
    have := u32_val_lt ndH.height
    omega
  unfold hashtrinity.overflow.Overflow.update
  simp only [lift, bind_tc_ok]
  obtain ⟨n, hn⟩ := vec_index_ex o.nodes (UScalar.cast .Usize i) hcastl
  rw [hn]; simp only [bind_tc_ok]
  have hnnd : n = nd := by
    have h' := vec_index_ok o.nodes (UScalar.cast .Usize i) n hn
    rw [cast_usize_u32, hL, hiv, hnd] at h'
    exact (Option.some.inj h').symm
  subst hnnd
  have hEl' : Emb L n.left l := hEl
  have hEr' : Emb L n.right r := hEr
  obtain ⟨a, ha⟩ := ov_h_ok o L hL n.left l hEl'
  have hav : a.val = ht l := h_spec o L hL n.left l hEl' hl a ha
  rw [ha]; simp only [bind_tc_ok]
  obtain ⟨b, hb⟩ := ov_h_ok o L hL n.right r hEr'
  have hbv : b.val = ht r := h_spec o L hL n.right r hEr' hr b hb
  rw [hb]; simp only [bind_tc_ok]
  have hab : a.val + 1 < 2 ^ UScalarTy.U32.numBits := by
    rw [UScalarTy.U32_numBits_eq, hav]
    have h1 : ht l ≤ max (ht l) (ht r) := le_max_left _ _
    omega
  by_cases hge : a >= b
  · rw [if_pos hge]; simp only [bind_tc_ok]
    obtain ⟨hh, hhh, -⟩ := uadd_tot a 1#u32 hab
    rw [hhh]; simp only [bind_tc_ok]
    obtain ⟨n2, back, hn2⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize i) hcastl
    rw [hn2]; simp only [bind_tc_ok]
    exact ⟨_, rfl⟩
  · rw [if_neg hge]; simp only [bind_tc_ok]
    have hba : b.val + 1 < 2 ^ UScalarTy.U32.numBits := by
      rw [UScalarTy.U32_numBits_eq, hbv]
      have h1 : ht r ≤ max (ht l) (ht r) := le_max_right _ _
      omega
    obtain ⟨hh, hhh, -⟩ := uadd_tot b 1#u32 hba
    rw [hhh]; simp only [bind_tc_ok]
    obtain ⟨n2, back, hn2⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize i) hcastl
    rw [hn2]; simp only [bind_tc_ok]
    exact ⟨_, rfl⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.ov_h_ok
#audit Apeiron.MapExecuted.ov_update_ok
