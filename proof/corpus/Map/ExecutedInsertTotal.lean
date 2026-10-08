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

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.ov_h_ok
#audit Apeiron.MapExecuted.ov_update_ok
#audit Apeiron.MapExecuted.ov_update_ok_hmax
#audit Apeiron.MapExecuted.ov_rotate_right_ok
#audit Apeiron.MapExecuted.ov_rotate_left_ok
