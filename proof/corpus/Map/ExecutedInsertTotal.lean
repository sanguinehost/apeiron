-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W7): totality of the
-- executed insert and resize.
import Map.ExecutedResize
import Map.ExecutedBuild
import Map.ExecutedReadTotal
import Map.ExecutedReadCoordTotal
import Map.ExecutedOverflowAvl
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

/-! ## `update` with an explicit fit bound -/

/-- `Overflow::update` is total on an embedded node whose recomputed height fits: the only failing
step is `taller + 1`, so a bound on `max (ht l) (ht r) + 1` suffices.  This is the totality twin of
`update_spec` with the `HOK` hypothesis on the *children* kept and the `HOK` on the updated node
replaced by the explicit fit bound, so that it applies to a node whose stored height is stale (the
rotations refresh `y` and `x` before their subtrees have current heights). -/
theorem ov_update_ok_hmax (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (i : U32)
    (l r : OTree) (ii : Nat) (hE : Emb L i (.node l ii r)) (hl : HOK L l) (hr : HOK L r)
    (hmax : max (ht l) (ht r) + 1 ≤ 2 ^ 32 - 1) :
    ∃ o', hashtrinity.overflow.Overflow.update CI o i = ok o' := by
  obtain ⟨hiv, hii, nd, hnd, hEl, hEr⟩ := hE
  have hilt : i.val < L.length := by rw [hiv]; exact (List.getElem?_eq_some_iff.mp hnd).1
  have hcastl : (UScalar.cast .Usize i).val < o.nodes.val.length := by
    rw [cast_usize_u32, hL]; exact hilt
  unfold hashtrinity.overflow.Overflow.update
  simp only [lift, bind_tc_ok]
  obtain ⟨n, hn⟩ := vec_index_ex o.nodes (UScalar.cast .Usize i) hcastl
  rw [hn]; simp only [bind_tc_ok]
  have hnnd : n = nd := by
    have h' := vec_index_ok o.nodes (UScalar.cast .Usize i) n hn
    rw [cast_usize_u32, hL, hiv, hnd] at h'
    exact (Option.some.inj h').symm
  rw [hnnd]
  obtain ⟨a, ha⟩ := ov_h_ok o L hL nd.left l hEl
  have hav : a.val = ht l := h_spec o L hL nd.left l hEl hl a ha
  rw [ha]; simp only [bind_tc_ok]
  obtain ⟨b, hb⟩ := ov_h_ok o L hL nd.right r hEr
  have hbv : b.val = ht r := h_spec o L hL nd.right r hEr hr b hb
  rw [hb]; simp only [bind_tc_ok]
  have hab : a.val + 1 < 2 ^ UScalarTy.U32.numBits := by
    rw [UScalarTy.U32_numBits_eq, hav]; omega
  by_cases hge : a >= b
  · rw [if_pos hge]; simp only [bind_tc_ok]
    obtain ⟨hh, hhh, -⟩ := uadd_tot a 1#u32 hab
    rw [hhh]; simp only [bind_tc_ok]
    obtain ⟨n2, back, hn2⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize i) hcastl
    rw [hn2]; simp only [bind_tc_ok]
    exact ⟨_, rfl⟩
  · rw [if_neg hge]; simp only [bind_tc_ok]
    have hba : b.val + 1 < 2 ^ UScalarTy.U32.numBits := by
      rw [UScalarTy.U32_numBits_eq, hbv]; omega
    obtain ⟨hh, hhh, -⟩ := uadd_tot b 1#u32 hba
    rw [hhh]; simp only [bind_tc_ok]
    obtain ⟨n2, back, hn2⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize i) hcastl
    rw [hn2]; simp only [bind_tc_ok]
    exact ⟨_, rfl⟩

/-! ## Rotation totality -/

/-- `Overflow::rotate_right` is total: the totality twin of `rotate_right_spec`.  The spec's
hypothesis bundle carries over (`Emb`, `ND`, and the `HOK` of the whole tree in place of the
per-subtree `HOK`); the extra `hroom` states that the one-step growth of the refreshed nodes still
fits the word.  It is the fit of the second `update`, whose node `x` regains height
`max (ht a) (ht (.node b yi c)) + 1`; without it a degenerate chain of height `2^32 - 1` would
overflow the `u32` height. -/
theorem ov_rotate_right_ok (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (y : U32)
    (a b c : OTree) (xi yi : Nat)
    (hE : Emb L y (.node (.node a xi b) yi c))
    (hND : ND (.node (.node a xi b) yi c))
    (hH : HOK L (.node (.node a xi b) yi c))
    (hroom : ht (.node (.node a xi b) yi c) + 1 ≤ 2 ^ 32 - 1) :
    ∃ x' o', hashtrinity.overflow.Overflow.rotate_right CI o y = ok (x', o') := by
  obtain ⟨hyv, hyi, ny, hny, hEL, hEc⟩ := hE
  obtain ⟨hxv, hxi, nx, hnx, hEa, hEb⟩ := hEL
  obtain ⟨⟨hNDa, hNDb, hxa, hxb, hab⟩, hNDc, hyl, hyc, hlc⟩ := hND
  obtain ⟨hHAB, hHc, nyH, hnyH, hnyHv⟩ := hH
  obtain ⟨hHa, hHb, nxH, hnxH, hnxHv⟩ := hHAB
  have hyl' : yi ∉ idxs a ∧ ¬ yi = xi ∧ yi ∉ idxs b := by
    simpa only [idxs, List.mem_append, List.mem_cons, not_or] using hyl
  have hxy : xi ≠ yi := fun e => hyl'.2.1 e.symm
  have hyx : yi ≠ xi := hyl'.2.1
  have hxlt : xi < L.length := (List.getElem?_eq_some_iff.mp hnx).1
  have hylt : yi < L.length := (List.getElem?_eq_some_iff.mp hny).1
  have hmax1 : max (ht b) (ht c) + 1 ≤ 2 ^ 32 - 1 := by
    simp only [ht] at hroom ⊢; omega
  have hmax2 : max (ht a) (ht (.node b yi c)) + 1 ≤ 2 ^ 32 - 1 := by
    simp only [ht] at hroom ⊢; omega
  unfold hashtrinity.overflow.Overflow.rotate_right
  simp only [lift, bind_tc_ok]
  obtain ⟨N0, hN0ok⟩ := vec_index_ex o.nodes (UScalar.cast .Usize y)
    (by rw [cast_usize_u32, hL, hyv]; exact hylt)
  have hN0 : N0 = ny := by
    have h' := vec_index_ok o.nodes (UScalar.cast .Usize y) N0 hN0ok
    rw [cast_usize_u32, hL, hyv, hny] at h'
    exact (Option.some.inj h').symm
  obtain ⟨N1, hN1ok⟩ := vec_index_ex o.nodes (UScalar.cast .Usize ny.left)
    (by rw [cast_usize_u32, hL, hxv]; exact hxlt)
  have hN1 : N1 = nx := by
    have h' := vec_index_ok o.nodes (UScalar.cast .Usize ny.left) N1 hN1ok
    rw [cast_usize_u32, hL, hxv, hnx] at h'
    exact (Option.some.inj h').symm
  obtain ⟨N2, back, hN2ok⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize ny.left)
    (by rw [cast_usize_u32, hL, hxv]; exact hxlt)
  obtain ⟨hN2', hback⟩ := idxm_ok hN2ok
  rw [hL, hxv, hnx] at hN2'
  have hN2 : N2 = nx := (Option.some.inj hN2').symm
  have hv : (back {nx with right := y}).val = L.set xi {nx with right := y} := by
    rw [hback, hL, hxv]
  obtain ⟨N3, back1, hN3ok⟩ := vec_index_mut_ex (back {nx with right := y}) (UScalar.cast .Usize y)
    (by rw [hv, List.length_set, cast_usize_u32, hyv]; exact hylt)
  obtain ⟨hN3', hback1⟩ := idxm_ok hN3ok
  rw [hv, hyv, List.getElem?_set_ne hxy, hny] at hN3'
  have hN3 : N3 = ny := (Option.some.inj hN3').symm
  have hv1 : (back1 {ny with left := nx.right}).val =
      (L.set xi {nx with right := y}).set yi {ny with left := nx.right} := by
    rw [hback1, hv, hyv]
  set A1 : List ONode := L.set xi {nx with right := y} with hA1def
  set A2 : List ONode := A1.set yi {ny with left := nx.right} with hA2def
  have hA2o : ∀ j, j ≠ xi → j ≠ yi → A2[j]? = L[j]? := by
    intro j hjx hjy
    rw [hA2def, hA1def, List.getElem?_set_ne (Ne.symm hjy), List.getElem?_set_ne (Ne.symm hjx)]
  have hA2yi : A2[yi]? = some {ny with left := nx.right} := by
    rw [hA2def]
    exact List.getElem?_set_self (by rw [hA1def, List.length_set]; exact hylt)
  have hEb2 : Emb A2 nx.right b :=
    emb_congr hEb (fun j hj => hA2o j (fun e => hxb (e ▸ hj)) (fun e => hyl'.2.2 (e ▸ hj)))
  have hEc2 : Emb A2 ny.right c :=
    emb_congr hEc (fun j hj => hA2o j (fun e => hlc xi (by simp [idxs]) (e ▸ hj))
      (fun e => hyc (e ▸ hj)))
  have hE2 : Emb A2 y (.node b yi c) :=
    ⟨hyv, hyi, {ny with left := nx.right}, hA2yi, hEb2, hEc2⟩
  have hHb2 : HOK A2 b :=
    hok_congr hHb (fun j hj => hA2o j (fun e => hxb (e ▸ hj)) (fun e => hyl'.2.2 (e ▸ hj)))
  have hHc2 : HOK A2 c :=
    hok_congr hHc (fun j hj => hA2o j (fun e => hlc xi (by simp [idxs]) (e ▸ hj))
      (fun e => hyc (e ▸ hj)))
  obtain ⟨o1, hu1⟩ := ov_update_ok_hmax { o with nodes := back1 {ny with left := nx.right} } A2 hv1
    y b c yi hE2 hHb2 hHc2 hmax1
  obtain ⟨nd1, hh1, hnd1, hhv1, hnodes1, -, -, -⟩ :=
    update_spec { o with nodes := back1 {ny with left := nx.right} } o1 A2 hv1 y b c yi hE2 hHb2 hHc2 hu1
  have hnd1eq : nd1 = {ny with left := nx.right} := Option.some.inj (hnd1.symm.trans hA2yi)
  rw [hnd1eq] at hnodes1
  set A3 : List ONode := A2.set yi {({ny with left := nx.right} : ONode) with height := hh1} with hA3def
  have hL3 : o1.nodes.val = A3 := hnodes1
  have hA3o : ∀ j, j ≠ yi → A3[j]? = A2[j]? := by
    intro j hj; rw [hA3def, List.getElem?_set_ne (Ne.symm hj)]
  have hA3yi : A3[yi]? = some {({ny with left := nx.right} : ONode) with height := hh1} := by
    rw [hA3def]
    exact List.getElem?_set_self (by rw [hA2def, hA1def, List.length_set, List.length_set]; exact hylt)
  have hA3xi : A3[xi]? = some {nx with right := y} := by
    rw [hA3o xi hxy, hA2def, List.getElem?_set_ne hyx, hA1def]
    exact List.getElem?_set_self (by simpa using hxlt)
  have hA3oL : ∀ j, j ≠ xi → j ≠ yi → A3[j]? = L[j]? := by
    intro j hjx hjy; rw [hA3o j hjy, hA2o j hjx hjy]
  have hEa3 : Emb A3 nx.left a :=
    emb_congr hEa (fun j hj => hA3oL j (fun e => hxa (e ▸ hj)) (fun e => hyl'.1 (e ▸ hj)))
  have hEb3 : Emb A3 nx.right b :=
    emb_congr hEb (fun j hj => hA3oL j (fun e => hxb (e ▸ hj)) (fun e => hyl'.2.2 (e ▸ hj)))
  have hEc3 : Emb A3 ny.right c :=
    emb_congr hEc (fun j hj => hA3oL j (fun e => hlc xi (by simp [idxs]) (e ▸ hj))
      (fun e => hyc (e ▸ hj)))
  have hHa3 : HOK A3 a :=
    hok_congr hHa (fun j hj => hA3oL j (fun e => hxa (e ▸ hj)) (fun e => hyl'.1 (e ▸ hj)))
  have hHb3 : HOK A3 b :=
    hok_congr hHb (fun j hj => hA3oL j (fun e => hxb (e ▸ hj)) (fun e => hyl'.2.2 (e ▸ hj)))
  have hHc3 : HOK A3 c :=
    hok_congr hHc (fun j hj => hA3oL j (fun e => hlc xi (by simp [idxs]) (e ▸ hj))
      (fun e => hyc (e ▸ hj)))
  have hEy3 : Emb A3 y (.node b yi c) :=
    ⟨hyv, hyi, {({ny with left := nx.right} : ONode) with height := hh1}, hA3yi, hEb3, hEc3⟩
  have hE3 : Emb A3 ny.left (.node a xi (.node b yi c)) :=
    ⟨hxv, hxi, {nx with right := y}, hA3xi, hEa3, hEy3⟩
  have hHbc3 : HOK A3 (.node b yi c) :=
    ⟨hHb3, hHc3, {({ny with left := nx.right} : ONode) with height := hh1}, hA3yi, hhv1⟩
  obtain ⟨o2, hu2⟩ := ov_update_ok_hmax o1 A3 hL3 ny.left a (.node b yi c) xi hE3 hHa3 hHbc3 hmax2
  refine ⟨ny.left, o2, ?_⟩
  simp only [hN0ok, hN1ok, hN2ok, hN0, hN1, hN2, bind_tc_ok]
  change (do
    let (n3, index_mut_back1) ←
      alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice (hashtrinity.overflow.Node U64 I64))
        (back { key := nx.key, entries := nx.entries, left := nx.left, right := y, height := nx.height,
                alive := nx.alive })
        (UScalar.cast UScalarTy.Usize y)
    let self1 ←
      hashtrinity.overflow.Overflow.update CI
        { nodes := index_mut_back1 { key := n3.key, entries := n3.entries, left := nx.right, right := n3.right,
                                       height := n3.height, alive := n3.alive },
          free := o.free, root := o.root, len := o.len }
        y
    let self2 ← hashtrinity.overflow.Overflow.update CI self1 ny.left
    ok (ny.left, self2)) = ok (ny.left, o2)
  rw [hN3ok]; simp only [bind_tc_ok]; rw [hN3]
  change (do
    let self1 ←
      hashtrinity.overflow.Overflow.update CI
        { nodes := back1 { key := ny.key, entries := ny.entries, left := nx.right, right := ny.right,
                            height := ny.height, alive := ny.alive },
          free := o.free, root := o.root, len := o.len }
        y
    let self2 ← hashtrinity.overflow.Overflow.update CI self1 ny.left
    ok (ny.left, self2)) = ok (ny.left, o2)
  rw [hu1]; simp only [bind_tc_ok]
  rw [hu2]; simp only [bind_tc_ok]

/-- `Overflow::rotate_left` is total: the totality twin of `rotate_left_spec`, obtained from
`ov_rotate_right_ok` by the mirrored case.  Same hypothesis bundle (`Emb`, `ND`, `HOK` of the whole
tree) plus the fit bound `hroom` for the one-step growth of the two refreshed nodes. -/
theorem ov_rotate_left_ok (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (x : U32)
    (a b c : OTree) (xi yi : Nat)
    (hE : Emb L x (.node a xi (.node b yi c)))
    (hND : ND (.node a xi (.node b yi c)))
    (hH : HOK L (.node a xi (.node b yi c)))
    (hroom : ht (.node a xi (.node b yi c)) + 1 ≤ 2 ^ 32 - 1) :
    ∃ y' o', hashtrinity.overflow.Overflow.rotate_left CI o x = ok (y', o') := by
  obtain ⟨hxv, hxi, nx, hnx, hEa, hEy⟩ := hE
  obtain ⟨hyv, hyi, ny, hny, hEb, hEc⟩ := hEy
  obtain ⟨hNDa, ⟨hNDb, hNDc, hyb, hyc, hbc⟩, hxa, hxr, har⟩ := hND
  obtain ⟨hHa, hHab, nxH, hnxH, hnxHv⟩ := hH
  obtain ⟨hHb, hHc, nyH, hnyH, hnyHv⟩ := hHab
  have hxr' : xi ∉ idxs b ∧ xi ≠ yi ∧ xi ∉ idxs c := by
    simpa only [idxs, List.mem_append, List.mem_cons, not_or] using hxr
  have hxy : xi ≠ yi := hxr'.2.1
  have hyx : yi ≠ xi := fun e => hxy e.symm
  have hxlt : xi < L.length := (List.getElem?_eq_some_iff.mp hnx).1
  have hylt : yi < L.length := (List.getElem?_eq_some_iff.mp hny).1
  have hmax1 : max (ht a) (ht b) + 1 ≤ 2 ^ 32 - 1 := by
    simp only [ht] at hroom ⊢; omega
  have hmax2 : max (ht (.node a xi b)) (ht c) + 1 ≤ 2 ^ 32 - 1 := by
    simp only [ht] at hroom ⊢; omega
  have hya : yi ∉ idxs a := fun hh => har yi hh (by simp [idxs])
  unfold hashtrinity.overflow.Overflow.rotate_left
  simp only [lift, bind_tc_ok]
  obtain ⟨N0, hN0ok⟩ := vec_index_ex o.nodes (UScalar.cast .Usize x)
    (by rw [cast_usize_u32, hL, hxv]; exact hxlt)
  have hN0 : N0 = nx := by
    have h' := vec_index_ok o.nodes (UScalar.cast .Usize x) N0 hN0ok
    rw [cast_usize_u32, hL, hxv, hnx] at h'
    exact (Option.some.inj h').symm
  obtain ⟨N1, hN1ok⟩ := vec_index_ex o.nodes (UScalar.cast .Usize nx.right)
    (by rw [cast_usize_u32, hL, hyv]; exact hylt)
  have hN1 : N1 = ny := by
    have h' := vec_index_ok o.nodes (UScalar.cast .Usize nx.right) N1 hN1ok
    rw [cast_usize_u32, hL, hyv, hny] at h'
    exact (Option.some.inj h').symm
  obtain ⟨N2, back, hN2ok⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize nx.right)
    (by rw [cast_usize_u32, hL, hyv]; exact hylt)
  obtain ⟨hN2', hback⟩ := idxm_ok hN2ok
  rw [hL, hyv, hny] at hN2'
  have hN2 : N2 = ny := (Option.some.inj hN2').symm
  have hv : (back {ny with left := x}).val = L.set yi {ny with left := x} := by
    rw [hback, hL, hyv]
  obtain ⟨N3, back1, hN3ok⟩ := vec_index_mut_ex (back {ny with left := x}) (UScalar.cast .Usize x)
    (by rw [hv, List.length_set, cast_usize_u32, hxv]; exact hxlt)
  obtain ⟨hN3', hback1⟩ := idxm_ok hN3ok
  rw [hv, hxv, List.getElem?_set_ne hyx, hnx] at hN3'
  have hN3 : N3 = nx := (Option.some.inj hN3').symm
  have hv1 : (back1 {nx with right := ny.left}).val =
      (L.set yi {ny with left := x}).set xi {nx with right := ny.left} := by
    rw [hback1, hv, hxv]
  set A1 : List ONode := L.set yi {ny with left := x} with hA1def
  set A2 : List ONode := A1.set xi {nx with right := ny.left} with hA2def
  have hA2o : ∀ j, j ≠ xi → j ≠ yi → A2[j]? = L[j]? := by
    intro j hjx hjy
    rw [hA2def, hA1def, List.getElem?_set_ne (Ne.symm hjx), List.getElem?_set_ne (Ne.symm hjy)]
  have hA2xi : A2[xi]? = some {nx with right := ny.left} := by
    rw [hA2def]
    exact List.getElem?_set_self (by rw [hA1def, List.length_set]; exact hxlt)
  have hEa2 : Emb A2 nx.left a :=
    emb_congr hEa (fun j hj => hA2o j (fun e => hxa (e ▸ hj)) (fun e => hya (e ▸ hj)))
  have hEb2 : Emb A2 ny.left b :=
    emb_congr hEb (fun j hj => hA2o j (fun e => hxr'.1 (e ▸ hj)) (fun e => hyb (e ▸ hj)))
  have hE2 : Emb A2 x (.node a xi b) :=
    ⟨hxv, hxi, {nx with right := ny.left}, hA2xi, hEa2, hEb2⟩
  have hHa2 : HOK A2 a :=
    hok_congr hHa (fun j hj => hA2o j (fun e => hxa (e ▸ hj)) (fun e => hya (e ▸ hj)))
  have hHb2 : HOK A2 b :=
    hok_congr hHb (fun j hj => hA2o j (fun e => hxr'.1 (e ▸ hj)) (fun e => hyb (e ▸ hj)))
  obtain ⟨o1, hu1⟩ := ov_update_ok_hmax { o with nodes := back1 {nx with right := ny.left} } A2 hv1
    x a b xi hE2 hHa2 hHb2 hmax1
  obtain ⟨nd1, hh1, hnd1, hhv1, hnodes1, -, -, -⟩ :=
    update_spec { o with nodes := back1 {nx with right := ny.left} } o1 A2 hv1 x a b xi hE2 hHa2 hHb2 hu1
  have hnd1eq : nd1 = {nx with right := ny.left} := Option.some.inj (hnd1.symm.trans hA2xi)
  rw [hnd1eq] at hnodes1
  set A3 : List ONode := A2.set xi {({nx with right := ny.left} : ONode) with height := hh1} with hA3def
  have hL3 : o1.nodes.val = A3 := hnodes1
  have hA3o : ∀ j, j ≠ xi → A3[j]? = A2[j]? := by
    intro j hj; rw [hA3def, List.getElem?_set_ne (Ne.symm hj)]
  have hA3xi : A3[xi]? = some {({nx with right := ny.left} : ONode) with height := hh1} := by
    rw [hA3def]
    exact List.getElem?_set_self (by rw [hA2def, hA1def, List.length_set, List.length_set]; exact hxlt)
  have hA3yi : A3[yi]? = some {ny with left := x} := by
    rw [hA3o yi hyx, hA2def, List.getElem?_set_ne hxy]
    exact List.getElem?_set_self hylt
  have hA3oL : ∀ j, j ≠ xi → j ≠ yi → A3[j]? = L[j]? := by
    intro j hjx hjy; rw [hA3o j hjx, hA2o j hjx hjy]
  have hEa3 : Emb A3 nx.left a :=
    emb_congr hEa (fun j hj => hA3oL j (fun e => hxa (e ▸ hj)) (fun e => hya (e ▸ hj)))
  have hEb3 : Emb A3 ny.left b :=
    emb_congr hEb (fun j hj => hA3oL j (fun e => hxr'.1 (e ▸ hj)) (fun e => hyb (e ▸ hj)))
  have hEc3 : Emb A3 ny.right c :=
    emb_congr hEc (fun j hj => hA3oL j (fun e => hxr'.2.2 (e ▸ hj)) (fun e => hyc (e ▸ hj)))
  have hHa3 : HOK A3 a :=
    hok_congr hHa (fun j hj => hA3oL j (fun e => hxa (e ▸ hj)) (fun e => hya (e ▸ hj)))
  have hHb3 : HOK A3 b :=
    hok_congr hHb (fun j hj => hA3oL j (fun e => hxr'.1 (e ▸ hj)) (fun e => hyb (e ▸ hj)))
  have hHc3 : HOK A3 c :=
    hok_congr hHc (fun j hj => hA3oL j (fun e => hxr'.2.2 (e ▸ hj)) (fun e => hyc (e ▸ hj)))
  have hEx3 : Emb A3 x (.node a xi b) :=
    ⟨hxv, hxi, {({nx with right := ny.left} : ONode) with height := hh1}, hA3xi, hEa3, hEb3⟩
  have hHx3 : HOK A3 (.node a xi b) :=
    ⟨hHa3, hHb3, {({nx with right := ny.left} : ONode) with height := hh1}, hA3xi, hhv1⟩
  have hE3 : Emb A3 nx.right (.node (.node a xi b) yi c) :=
    ⟨hyv, hyi, {ny with left := x}, hA3yi, hEx3, hEc3⟩
  obtain ⟨o2, hu2⟩ := ov_update_ok_hmax o1 A3 hL3 nx.right (.node a xi b) c yi hE3 hHx3 hHc3 hmax2
  refine ⟨nx.right, o2, ?_⟩
  simp only [hN0ok, hN1ok, hN2ok, hN0, hN1, hN2, bind_tc_ok]
  change (do
    let (n3, index_mut_back1) ←
      alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice (hashtrinity.overflow.Node U64 I64))
        (back { key := ny.key, entries := ny.entries, left := x, right := ny.right, height := ny.height,
                alive := ny.alive })
        (UScalar.cast UScalarTy.Usize x)
    let self1 ←
      hashtrinity.overflow.Overflow.update CI
        { nodes := index_mut_back1 { key := n3.key, entries := n3.entries, left := n3.left, right := ny.left,
                                       height := n3.height, alive := n3.alive },
          free := o.free, root := o.root, len := o.len }
        x
    let self2 ← hashtrinity.overflow.Overflow.update CI self1 nx.right
    ok (nx.right, self2)) = ok (nx.right, o2)
  rw [hN3ok]; simp only [bind_tc_ok]; rw [hN3]
  change (do
    let self1 ←
      hashtrinity.overflow.Overflow.update CI
        { nodes := back1 { key := nx.key, entries := nx.entries, left := nx.left, right := ny.left,
                            height := nx.height, alive := nx.alive },
          free := o.free, root := o.root, len := o.len }
        x
    let self2 ← hashtrinity.overflow.Overflow.update CI self1 nx.right
    ok (nx.right, self2)) = ok (nx.right, o2)
  rw [hu1]; simp only [bind_tc_ok]
  rw [hu2]; simp only [bind_tc_ok]

/-! ## `rebalance` totality -/

/-- Reduce a `let (x,y) ← ok (a,b); k x y` monadic pair-bind left by the literal pair (the
extraction's pair-bind redex that `bind_tc_ok` leaves behind). -/
theorem bind_pair_ok {α β γ : Type} (a : α) (b : β) (k : α → β → Result γ) :
    (do let (x, y) ← (ok (a, b) : Result (α × β)); k x y) = k a b := by
  simp only [bind_tc_ok]
  rfl

/-- `Overflow::rebalance` is total: the totality twin of `rebalance_spec`.  It carries the spec's
`Bal`/`HOK` premises — a double rotation relinks the parent to a subtree whose freshly stored height
is read by the following `update`, and `Bal` is what makes that stored height current again — and the
height fit `hroom` of the deepest intermediate tree (the parent after a double rotation reaches
height `max (ht l) (ht r) + 2`), which the spec's partial-correctness conclusion excludes. -/
theorem ov_rebalance_ok (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (i : U32)
    (l r : OTree) (ii : Nat)
    (hE : Emb L i (.node l ii r)) (hND : ND (.node l ii r)) (hB : BST L (.node l ii r))
    (hBl : Bal l) (hBr : Bal r) (hl : HOK L l) (hr : HOK L r)
    (hroom : max (ht l) (ht r) + 2 ≤ 2 ^ 32 - 1) :
    ∃ x' o', hashtrinity.overflow.Overflow.rebalance CI o i = ok (x', o') := by
  have hE0 := hE
  have hND0 := hND
  obtain ⟨hiv, hii, nd, hnd, hEl, hEr⟩ := hE0
  obtain ⟨hNDl, hNDr, hiil, hiir, hlr'⟩ := hND0
  have hiiL : ii < L.length := (List.getElem?_eq_some_iff.mp hnd).1
  have hmax1 : max (ht l) (ht r) + 1 ≤ 2 ^ 32 - 1 := by omega
  unfold hashtrinity.overflow.Overflow.rebalance
  simp only [lift, bind_tc_ok]
  obtain ⟨o1, hu1⟩ := ov_update_ok_hmax o L hL i l r ii hE hl hr hmax1
  rw [hu1]; simp only [bind_tc_ok]
  obtain ⟨nd1, hh, hnd1, hhv, hnodes1, hf1, hr1, hl1⟩ :=
    update_spec o o1 L hL i l r ii hE hl hr hu1
  have hndeq : nd = nd1 := Option.some.inj (hnd.symm.trans hnd1)
  subst hndeq
  obtain ⟨L1, hL1def⟩ : ∃ L1, L1 = L.set ii {nd with height := hh} := ⟨_, rfl⟩
  have hL1 : o1.nodes.val = L1 := by rw [hnodes1, hL1def]
  have hL1ii : L1[ii]? = some {nd with height := hh} := by
    rw [hL1def]; exact List.getElem?_set_self hiiL
  have hL1o : ∀ j, j ≠ ii → L1[j]? = L[j]? := by
    intro j hj; rw [hL1def, List.getElem?_set_ne (Ne.symm hj)]
  have hrel1 : Rel [ii] L L1 := by rw [hL1def]; exact Rel_set hnd ⟨rfl, rfl, rfl⟩
  have hnel : ∀ j ∈ idxs l, j ≠ ii := fun j hj e => hiil (e ▸ hj)
  have hner : ∀ j ∈ idxs r, j ≠ ii := fun j hj e => hiir (e ▸ hj)
  have hEl1 : Emb L1 nd.left l := emb_congr hEl (fun j hj => hL1o j (hnel j hj))
  have hEr1 : Emb L1 nd.right r := emb_congr hEr (fun j hj => hL1o j (hner j hj))
  have hHl1 : HOK L1 l := hok_congr hl (fun j hj => hL1o j (hnel j hj))
  have hHr1 : HOK L1 r := hok_congr hr (fun j hj => hL1o j (hner j hj))
  have hB1 : BST L1 (.node l ii r) := bst_rel hrel1 hB
  have hEi1 : Emb L1 i (.node l ii r) := ⟨hiv, hii, {nd with height := hh}, hL1ii, hEl1, hEr1⟩
  have hH1 : HOK L1 (.node l ii r) := ⟨hHl1, hHr1, {nd with height := hh}, hL1ii, hhv⟩
  obtain ⟨n, hn⟩ := vec_index_ex o1.nodes (UScalar.cast .Usize i)
    (by rw [cast_usize_u32, hiv, hL1, hL1def, List.length_set]; exact hiiL)
  rw [hn]; simp only [bind_tc_ok]
  have hn_eq : n = ({nd with height := hh} : ONode) := by
    have h' := vec_index_ok o1.nodes (UScalar.cast .Usize i) n hn
    rw [cast_usize_u32, hiv, hL1] at h'
    exact Option.some.inj (h'.symm.trans hL1ii)
  have hnL : n.left = nd.left := by rw [hn_eq]
  have hnR : n.right = nd.right := by rw [hn_eq]
  obtain ⟨a, ha⟩ := ov_h_ok o1 L1 hL1 n.left l (by rw [hnL]; exact hEl1)
  rw [ha]; simp only [bind_tc_ok]
  obtain ⟨b, hb⟩ := ov_h_ok o1 L1 hL1 n.right r (by rw [hnR]; exact hEr1)
  rw [hb]; simp only [bind_tc_ok]
  have hav : a.val = ht l := h_spec o1 L1 hL1 n.left l (by rw [hnL]; exact hEl1) hHl1 a ha
  have hbv : b.val = ht r := h_spec o1 L1 hL1 n.right r (by rw [hnR]; exact hEr1) hHr1 b hb
  have hb1 : b.val + 1 < 2 ^ UScalarTy.U32.numBits := by
    rw [UScalarTy.U32_numBits_eq, hbv]; omega
  obtain ⟨i3, hi3, hi3v⟩ := uadd_tot b 1#u32 hb1
  rw [hi3]; simp only [bind_tc_ok]
  by_cases hA : a > i3
  · -- left-heavy
    have hA' : ht r + 1 < ht l := by
      have h := (UScalar.lt_equiv i3 a).mp hA
      rw [hi3v, hbv, u32_one, hav] at h
      exact h
    rw [if_pos hA]
    cases l with
    | leaf => simp only [ht] at hA'; omega
    | node ll li lr =>
      have hEl1c := hEl1
      have hHl1c := hHl1
      obtain ⟨hlv, hli, nl, hnl, hEll, hElr⟩ := hEl1
      obtain ⟨hHll, hHlr, _, _, _⟩ := hHl1
      obtain ⟨n2, hn2⟩ := vec_index_ex o1.nodes (UScalar.cast .Usize n.left)
        (by rw [cast_usize_u32, hnL, hlv, hL1]; exact (List.getElem?_eq_some_iff.mp hnl).1)
      rw [hn2]; simp only [bind_tc_ok]
      have hn2_eq : n2 = nl := by
        have h' := vec_index_ok o1.nodes (UScalar.cast .Usize n.left) n2 hn2
        rw [cast_usize_u32, hnL, hlv, hL1] at h'
        exact Option.some.inj (h'.symm.trans hnl)
      have hn2L : n2.left = nl.left := by rw [hn2_eq]
      have hn2R : n2.right = nl.right := by rw [hn2_eq]
      obtain ⟨c, hc⟩ := ov_h_ok o1 L1 hL1 n2.right lr (by rw [hn2R]; exact hElr)
      rw [hc]; simp only [bind_tc_ok]
      have hcv : c.val = ht lr :=
        h_spec o1 L1 hL1 n2.right lr (by rw [hn2R]; exact hElr) hHlr c hc
      obtain ⟨d, hd⟩ := ov_h_ok o1 L1 hL1 n2.left ll (by rw [hn2L]; exact hEll)
      rw [hd]; simp only [bind_tc_ok]
      have hdv : d.val = ht ll :=
        h_spec o1 L1 hL1 n2.left ll (by rw [hn2L]; exact hEll) hHll d hd
      by_cases hrot : c > d
      · -- left-right double rotation
        have hrot' : ht ll < ht lr := by
          have h := (UScalar.lt_equiv d c).mp hrot
          rw [hcv, hdv] at h
          exact h
        rw [if_pos hrot]
        cases lr with
        | leaf => simp only [ht] at hrot'; omega
        | node lrl lri lrr =>
          obtain ⟨hHlrl, hHlrr, _, _, _⟩ := hHlr
          obtain ⟨-, hBlr, hb1', hb2'⟩ := hBl
          obtain ⟨-, -, hb3', hb4'⟩ := hBlr
          have hhlr : ht (.node (.node ll li lrl) lri lrr) = ht (.node ll li (.node lrl lri lrr)) := by
            simp only [ht] at *; omega
          have hroomL : ht (.node ll li (.node lrl lri lrr)) + 1 ≤ 2 ^ 32 - 1 := by
            simp only [ht] at hroom ⊢; omega
          obtain ⟨nl2, o2, hrot_left⟩ := ov_rotate_left_ok o1 L1 hL1 n.left ll lrl lrr li lri
            (by rw [hnL]; exact hEl1c) hNDl hHl1c hroomL
          rw [hrot_left]; rw [bind_pair_ok]
          obtain ⟨A2, hA2, hnl2v, hEA2, hRel2, hB2, hH2, hf2, hr2, hl2⟩ :=
            rotate_left_spec o1 o2 L1 hL1 n.left nl2 ll lrl lrr li lri
              (by rw [hnL]; exact hEl1c) hNDl hB1.1 hHll hHlrl hHlrr hrot_left
          have hA2ii : A2[ii]? = some {nd with height := hh} := by
            rw [rel_out hRel2 hiil]; exact hL1ii
          have hA2lt : ii < A2.length := (List.getElem?_eq_some_iff.mp hA2ii).1
          obtain ⟨n4, back, hmut⟩ := vec_index_mut_ex o2.nodes (UScalar.cast .Usize i)
            (by rw [cast_usize_u32, hiv, hA2]; exact hA2lt)
          rw [hmut]; rw [bind_pair_ok]
          obtain ⟨hn4h1, hback⟩ := idxm_ok hmut
          rw [hA2, hiv] at hn4h1
          have hn4eq : n4 = ({nd with height := hh} : ONode) :=
            Option.some.inj (hn4h1.symm.trans hA2ii)
          have hback' : ∀ x, (back x).val = A2.set ii x := by
            intro x; rw [hback x, hA2, hiv]
          obtain ⟨A3, hA3def⟩ : ∃ A3, A3 =
              A2.set ii {({nd with height := hh} : ONode) with left := nl2} := ⟨_, rfl⟩
          have hA3ii : A3[ii]? = some {({nd with height := hh} : ONode) with left := nl2} := by
            rw [hA3def]; exact List.getElem?_set_self hA2lt
          have hA3o : ∀ j, j ≠ ii → A3[j]? = A2[j]? := by
            intro j hj; rw [hA3def, List.getElem?_set_ne (Ne.symm hj)]
          have hne : ∀ j ∈ idxs (.node (.node ll li lrl) lri lrr), j ≠ ii := by
            intro j hj e
            rw [idxs_rotL] at hj
            exact hiil (e ▸ hj)
          have hEr3 : Emb A3 nd.right r :=
            emb_congr (emb_rel hRel2 hEr1 (fun j hj hS => hlr' j hS hj))
              (fun j hj => hA3o j (hner j hj))
          have hE3 : Emb A3 i (.node (.node (.node ll li lrl) lri lrr) ii r) :=
            ⟨hiv, hii, {({nd with height := hh} : ONode) with left := nl2}, hA3ii,
              emb_congr hEA2 (fun j hj => hA3o j (hne j hj)), hEr3⟩
          have hND3 : ND (.node (.node (.node ll li lrl) lri lrr) ii r) :=
            nd_repl_l hND (nd_rotL hNDl) idxs_rotL
          have hH3 : HOK A3 (.node (.node (.node ll li lrl) lri lrr) ii r) :=
            ⟨hok_congr hH2 (fun j hj => hA3o j (hne j hj)),
              hok_congr (hok_rel hRel2 hHr1 (fun j hj hS => hlr' j hS hj))
                (fun j hj => hA3o j (hner j hj)),
              {({nd with height := hh} : ONode) with left := nl2}, hA3ii,
              by rw [hhlr]; exact hhv⟩
          have hroom3 : ht (.node (.node (.node ll li lrl) lri lrr) ii r) + 1 ≤ 2 ^ 32 - 1 := by
            have hh2 : ht (.node (.node ll li lrl) lri lrr) = ht (.node ll li (.node lrl lri lrr)) := hhlr
            simp only [ht] at hroom hh2 ⊢; omega
          have hA3 : (back {n4 with left := nl2}).val = A3 := by
            rw [hn4eq, hback', hA3def]
          obtain ⟨x', o', hrot2⟩ := ov_rotate_right_ok
            { nodes := back {n4 with left := nl2}, free := o2.free, root := o2.root, len := o2.len }
            A3 hA3 i (.node ll li lrl) lrr r lri ii hE3 hND3 hH3 hroom3
          refine ⟨x', o', ?_⟩
          simp only [bind_tc_ok]
          change hashtrinity.overflow.Overflow.rotate_right CI
            { nodes := back {n4 with left := nl2}, free := o2.free, root := o2.root, len := o2.len } i
            = ok (x', o')
          exact hrot2
      · -- left-left single rotation
        have hrot' : ht lr ≤ ht ll := by
          have h := (not_congr (UScalar.lt_equiv d c)).mp hrot
          rw [hcv, hdv] at h
          omega
        rw [if_neg hrot]; simp only [bind_tc_ok]
        have hroomLL : ht (.node (.node ll li lr) ii r) + 1 ≤ 2 ^ 32 - 1 := by
          simp only [ht] at hroom ⊢; omega
        obtain ⟨x', o', hrot2⟩ :=
          ov_rotate_right_ok o1 L1 hL1 i ll lr r li ii hEi1 hND hH1 hroomLL
        refine ⟨x', o', ?_⟩
        change hashtrinity.overflow.Overflow.rotate_right CI o1 i = ok (x', o')
        exact hrot2
  · -- not left-heavy
    rw [if_neg hA]
    have hA'' : ht l ≤ ht r + 1 := by
      have h := (not_congr (UScalar.lt_equiv i3 a)).mp hA
      rw [hi3v, hbv, u32_one, hav] at h
      omega
    have ha1 : a.val + 1 < 2 ^ UScalarTy.U32.numBits := by
      rw [UScalarTy.U32_numBits_eq, hav]; omega
    obtain ⟨i4, hi4, hi4v⟩ := uadd_tot a 1#u32 ha1
    rw [hi4]; simp only [bind_tc_ok]
    by_cases hB' : b > i4
    · -- right-heavy
      have hB'' : ht l + 1 < ht r := by
        have h := (UScalar.lt_equiv i4 b).mp hB'
        rw [hi4v, hav, u32_one, hbv] at h
        exact h
      rw [if_pos hB']
      cases r with
      | leaf => simp only [ht] at hB''; omega
      | node rl ri rr =>
        have hEr1c := hEr1
        have hHr1c := hHr1
        obtain ⟨hrv, hri, nr, hnr, hErl, hErr⟩ := hEr1
        obtain ⟨hHrl, hHrr, _, _, _⟩ := hHr1
        obtain ⟨n2, hn2⟩ := vec_index_ex o1.nodes (UScalar.cast .Usize n.right)
          (by rw [cast_usize_u32, hnR, hrv, hL1]; exact (List.getElem?_eq_some_iff.mp hnr).1)
        rw [hn2]; simp only [bind_tc_ok]
        have hn2_eq : n2 = nr := by
          have h' := vec_index_ok o1.nodes (UScalar.cast .Usize n.right) n2 hn2
          rw [cast_usize_u32, hnR, hrv, hL1] at h'
          exact Option.some.inj (h'.symm.trans hnr)
        have hn2L : n2.left = nr.left := by rw [hn2_eq]
        have hn2R : n2.right = nr.right := by rw [hn2_eq]
        obtain ⟨c, hc⟩ := ov_h_ok o1 L1 hL1 n2.left rl (by rw [hn2L]; exact hErl)
        rw [hc]; simp only [bind_tc_ok]
        have hcv : c.val = ht rl :=
          h_spec o1 L1 hL1 n2.left rl (by rw [hn2L]; exact hErl) hHrl c hc
        obtain ⟨d, hd⟩ := ov_h_ok o1 L1 hL1 n2.right rr (by rw [hn2R]; exact hErr)
        rw [hd]; simp only [bind_tc_ok]
        have hdv : d.val = ht rr :=
          h_spec o1 L1 hL1 n2.right rr (by rw [hn2R]; exact hErr) hHrr d hd
        by_cases hrot : c > d
        · -- right-left double rotation
          have hrot' : ht rr < ht rl := by
            have h := (UScalar.lt_equiv d c).mp hrot
            rw [hcv, hdv] at h
            exact h
          rw [if_pos hrot]
          cases rl with
          | leaf => simp only [ht] at hrot'; omega
          | node rll rli rlr =>
            obtain ⟨hHrll, hHrlr, _, _, _⟩ := hHrl
            obtain ⟨hBrl, -, hc1, hc2⟩ := hBr
            obtain ⟨-, -, hc3, hc4⟩ := hBrl
            have hhrl : ht (.node rll rli (.node rlr ri rr)) = ht (.node (.node rll rli rlr) ri rr) := by
              simp only [ht] at *; omega
            have hroomR : ht (.node (.node rll rli rlr) ri rr) + 1 ≤ 2 ^ 32 - 1 := by
              simp only [ht] at hroom ⊢; omega
            obtain ⟨nr2, o2, hrot_right⟩ :=
              ov_rotate_right_ok o1 L1 hL1 n.right rll rlr rr rli ri
                (by rw [hnR]; exact hEr1c) hNDr hHr1c hroomR
            rw [hrot_right]; rw [bind_pair_ok]
            obtain ⟨A2, hA2, hnr2v, hEA2, hRel2, hB2, hH2, hf2, hr2, hl2⟩ :=
              rotate_right_spec o1 o2 L1 hL1 n.right nr2 rll rlr rr rli ri
                (by rw [hnR]; exact hEr1c) hNDr hB1.2.1 hHrll hHrlr hHrr hrot_right
            have hA2ii : A2[ii]? = some {nd with height := hh} := by
              rw [rel_out hRel2 hiir]; exact hL1ii
            have hA2lt : ii < A2.length := (List.getElem?_eq_some_iff.mp hA2ii).1
            obtain ⟨n4, back, hmut⟩ := vec_index_mut_ex o2.nodes (UScalar.cast .Usize i)
              (by rw [cast_usize_u32, hiv, hA2]; exact hA2lt)
            rw [hmut]; rw [bind_pair_ok]
            obtain ⟨hn4h1, hback⟩ := idxm_ok hmut
            rw [hA2, hiv] at hn4h1
            have hn4eq : n4 = ({nd with height := hh} : ONode) :=
              Option.some.inj (hn4h1.symm.trans hA2ii)
            have hback' : ∀ x, (back x).val = A2.set ii x := by
              intro x; rw [hback x, hA2, hiv]
            obtain ⟨A3, hA3def⟩ : ∃ A3, A3 =
                A2.set ii {({nd with height := hh} : ONode) with right := nr2} := ⟨_, rfl⟩
            have hA3ii : A3[ii]? = some {({nd with height := hh} : ONode) with right := nr2} := by
              rw [hA3def]; exact List.getElem?_set_self hA2lt
            have hA3o : ∀ j, j ≠ ii → A3[j]? = A2[j]? := by
              intro j hj; rw [hA3def, List.getElem?_set_ne (Ne.symm hj)]
            have hne : ∀ j ∈ idxs (.node rll rli (.node rlr ri rr)), j ≠ ii := by
              intro j hj e
              rw [idxs_rotR] at hj
              exact hiir (e ▸ hj)
            have hEl3 : Emb A3 nd.left l :=
              emb_congr (emb_rel hRel2 hEl1 (fun j hj hS => hlr' j hj hS))
                (fun j hj => hA3o j (hnel j hj))
            have hE3 : Emb A3 i (.node l ii (.node rll rli (.node rlr ri rr))) :=
              ⟨hiv, hii, {({nd with height := hh} : ONode) with right := nr2}, hA3ii, hEl3,
                emb_congr hEA2 (fun j hj => hA3o j (hne j hj))⟩
            have hND3 : ND (.node l ii (.node rll rli (.node rlr ri rr))) :=
              nd_repl_r hND (nd_rotR hNDr) idxs_rotR
            have hH3 : HOK A3 (.node l ii (.node rll rli (.node rlr ri rr))) :=
              ⟨hok_congr (hok_rel hRel2 hHl1 (fun j hj hS => hlr' j hj hS))
                 (fun j hj => hA3o j (hnel j hj)),
               hok_congr hH2 (fun j hj => hA3o j (hne j hj)),
               {({nd with height := hh} : ONode) with right := nr2}, hA3ii,
               by rw [hhrl]; exact hhv⟩
            have hroom3 : ht (.node l ii (.node rll rli (.node rlr ri rr))) + 1 ≤ 2 ^ 32 - 1 := by
              have hh2 : ht (.node rll rli (.node rlr ri rr)) = ht (.node (.node rll rli rlr) ri rr) := hhrl
              simp only [ht] at hroom hh2 ⊢; omega
            have hA3 : (back {n4 with right := nr2}).val = A3 := by
              rw [hn4eq, hback', hA3def]
            obtain ⟨y', o', hrot2⟩ := ov_rotate_left_ok
              { nodes := back {n4 with right := nr2}, free := o2.free, root := o2.root, len := o2.len }
              A3 hA3 i l rll (.node rlr ri rr) ii rli hE3 hND3 hH3 hroom3
            refine ⟨y', o', ?_⟩
            simp only [bind_tc_ok]
            change hashtrinity.overflow.Overflow.rotate_left CI
              { nodes := back {n4 with right := nr2}, free := o2.free, root := o2.root, len := o2.len } i
              = ok (y', o')
            exact hrot2
        · -- right-right single rotation
          have hrot' : ht rl ≤ ht rr := by
            have h := (not_congr (UScalar.lt_equiv d c)).mp hrot
            rw [hcv, hdv] at h
            omega
          rw [if_neg hrot]; simp only [bind_tc_ok]
          have hroomRR : ht (.node l ii (.node rl ri rr)) + 1 ≤ 2 ^ 32 - 1 := by
            simp only [ht] at hroom ⊢; omega
          obtain ⟨y', o', hrot2⟩ :=
            ov_rotate_left_ok o1 L1 hL1 i l rl rr ii ri hEi1 hND hH1 hroomRR
          refine ⟨y', o', ?_⟩
          change hashtrinity.overflow.Overflow.rotate_left CI o1 i = ok (y', o')
          exact hrot2
    · -- balanced
      rw [if_neg hB']
      exact ⟨i, o1, rfl⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.ov_h_ok
#audit Apeiron.MapExecuted.ov_update_ok
#audit Apeiron.MapExecuted.ov_update_ok_hmax
#audit Apeiron.MapExecuted.ov_rotate_right_ok
#audit Apeiron.MapExecuted.ov_rotate_left_ok
#audit Apeiron.MapExecuted.ov_rebalance_ok
#audit Apeiron.MapExecuted.bind_pair_ok
