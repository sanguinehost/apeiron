-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W7): totality of the
-- executed insert and resize.
import Map.ExecutedResize
import Map.ExecutedBuild
import Map.ExecutedReadTotal
import Map.ExecutedReadCoordTotal
import Map.ExecutedOverflowAvl
import Map.ExecutedOverflowChain
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

/-! ## Stage 4: the chain rebalance is total

`rebalance_chain` walks the recorded root-to-slot path from the innermost ancestor outward.  The only
possibly-failing call is `rebalance` at one frame; the AVL shape facts it returns are then used to
relink the parent (`chain_relink`), exactly as the partial spec `rebalance_chain_loop_spec` does.  The
height fit that `rebalance` needs follows from the absolute arena bound `o.nodes.val.length + 3 ≤
2 ^ 32 - 1`, which is preserved because every write keeps the arena length (`Rel`). -/

/-- `ChainInv` fixes the arena length: its `Rel` conjunct identifies the arena with `ns0`. -/
theorem ChainInv.len_eq {ns0 : List ONode} {U : List Nat} {F0 : alloc.vec.Vec U32} {N0 : Usize}
    {chain : Slice U32} {o : Ov} {idx : Usize} (h : ChainInv ns0 U F0 N0 chain o idx) :
    o.nodes.val.length = ns0.length := by
  obtain ⟨G, S, s, sc, _, _, _, _, _, _, _, _, _, _, _, _, hRel, _, _⟩ := h
  exact hRel.1.symm

/-- Every embedded subtree height is below the ambient arena length: `ht_le_len` bounds it by the
number of node indices of the shape, `Emb` places those indices below the arena, and `ND` makes them
distinct. -/
theorem ht_le_arena_len {L : List ONode} {i : U32} {t : OTree}
    (hE : Emb L i t) (hND : ND t) : ht t ≤ L.length := by
  have h1 : ht t ≤ (idxs t).length := ht_le_len t
  have hlt : ∀ j ∈ idxs t, j < L.length := by
    intro j hj
    obtain ⟨nd, hnd, _⟩ := emb_mem hE j hj
    exact (List.getElem?_eq_some_iff.mp hnd).1
  have h2 : (idxs t).length ≤ L.length := idxs_length_le (nd_nodup t hND) hlt
  omega

/-- `ov_rebalance_ok` in the `frT` frame shape: `rebalance` at the innermost ancestor of a path. -/
theorem ov_rebalance_frT_ok (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (n : U32)
    (g : Fr) (S : OTree)
    (hEn : Emb L n (frT g S)) (hND : ND (frT g S)) (hB : BST L (frT g S))
    (hBS : Bal S) (hHS : HOK L S) (hBg : Bal g.o) (hHg : HOK L g.o)
    (hroom : max (ht S) (ht g.o) + 2 ≤ 2 ^ 32 - 1) :
    ∃ x' o', hashtrinity.overflow.Overflow.rebalance CI o n = ok (x', o') := by
  cases hd : g.d with
  | true =>
    rw [frT_true hd] at hEn hND hB
    exact ov_rebalance_ok o L hL n S g.o g.a hEn hND hB hBS hBg hHS hHg hroom
  | false =>
    rw [frT_false hd] at hEn hND hB
    refine ov_rebalance_ok o L hL n g.o S g.a hEn hND hB hBg hBS hHg hHS ?_
    rw [Nat.max_comm]; exact hroom

/-- One step of `rebalance_chain_loop` is total, preserves `ChainInv`, and strictly decreases the
loop index.  This is the `loop_tot_inv` body obligation, mirroring the body walk of
`rebalance_chain_loop_spec` with `ov_rebalance_ok` supplying the single potentially failing call and
`bind_pair_ok` handling the pair-binds. -/
theorem ov_rebalance_chain_body_ok (ns0 : List ONode) (U : List Nat) (F0 : alloc.vec.Vec U32)
    (N0 : Usize) (chain : Slice U32) (o : Ov) (idx : Usize)
    (hinv : ChainInv ns0 U F0 N0 chain.val o idx)
    (hbound : o.nodes.val.length + 3 ≤ 2 ^ 32 - 1) :
    ∃ cf, hashtrinity.overflow.Overflow.rebalance_chain_loop.body CI chain o idx = ok cf ∧
      (match cf with
        | .done _ => True
        | .cont x' => ChainInv ns0 U F0 N0 chain.val x'.1 x'.2 ∧ x'.2.val < idx.val) := by
  obtain ⟨G, S, s, sc, hidx, hch, hES, hBalS, hHS, hBST, hND, hU, hSp, hAnc, hsc1, hsc2, hRel,
    hfree, hlen⟩ := hinv
  unfold hashtrinity.overflow.Overflow.rebalance_chain_loop.body
  by_cases hpos : idx > 0#usize
  · rw [if_pos hpos]
    have hpos' : 0 < idx.val := by
      have := (UScalar.lt_equiv (0#usize) idx).mp hpos
      omega
    obtain ⟨idx1, hidx1, hidx1v⟩ := usub_tot idx 1#usize (by rw [val_1_usize]; omega)
    rw [val_1_usize] at hidx1v
    rw [hidx1]; simp only [bind_tc_ok]
    cases G with
    | nil => simp at hidx; omega
    | cons g rest =>
      have hidx' : idx.val = rest.length + 1 := by
        simpa only [List.length_cons] using hidx
      have hrl : rest.length = idx1.val := by rw [hidx1v]; omega
      obtain ⟨⟨c0, hc0, hc0v⟩, hchrest⟩ := hch
      have hidx1lt : idx1.val < chain.length := by
        rw [← hrl]; exact (List.getElem?_eq_some_iff.mp hc0).1
      obtain ⟨n, hn⟩ := Apeiron.PrefixMasks.slice_index_ex chain idx1 hidx1lt
      rw [hn]; simp only [bind_tc_ok]
      have hn' : chain.val[idx1.val]? = some n := slice_index_ok chain idx1 n hn
      rw [← hrl, hc0] at hn'
      have hnc : c0 = n := Option.some.inj hn'
      have hnv : n.val = g.a := by rw [← hnc]; exact hc0v
      obtain ⟨c, hSpc, hcv, hga, nd, hnd, hEo, hch'⟩ := hSp
      obtain ⟨hBg, hHg, hg1, hg2, _, hAncr⟩ := hAnc
      have hEn : Emb o.nodes.val n (frT g S) := emb_frT hnv hga hnd hEo hch' hES
      have hNDS : ND S := nd_plugI_sub (g :: rest) hND
      have hNDo : ND g.o := (nd_frT (nd_plugI_sub rest hND)).2.1
      have hroom : max (ht S) (ht g.o) + 2 ≤ 2 ^ 32 - 1 := by
        have hs := ht_le_arena_len hES hNDS
        have ho := ht_le_arena_len hEo hNDo
        omega
      obtain ⟨newroot, self1, hreb⟩ :=
        ov_rebalance_frT_ok o o.nodes.val rfl n g S hEn (nd_plugI_sub rest hND)
          (bst_plugI_sub rest hBST) hBalS hHS hBg hHg hroom
      rw [hreb]; rw [bind_pair_ok]
      obtain ⟨T', A, hA, hEA, hid, hndT, hbA, hhA, hbal, hrel, hf, hr, hl, hlo, hhi⟩ :=
        rebalance_frT o self1 o.nodes.val rfl n newroot g S sc hEn (nd_plugI_sub rest hND)
          (bst_plugI_sub rest hBST) hBalS hHS hBg hHg hsc1 hsc2 hg1 hg2 hreb
      have hUm : ∀ j ∈ idxs (frT g S), j ∈ U := fun j hj => by
        rw [← hU]; exact (mem_plugI rest (frT g S)).mpr (Or.inl hj)
      cases rest with
      | nil =>
        have hidx0 : idx1 = 0#usize := UScalar.eq_of_val_eq (by rw [← hrl]; rfl)
        rw [if_pos hidx0]
        refine ⟨ControlFlow.cont ({ self1 with root := newroot }, idx1), rfl, ?_⟩
        refine ⟨?_, ?_⟩
        · refine ⟨[], T', newroot, max sc (ht g.o) + 1, ?_, trivial, ?_, hbal, ?_, ?_,
            hndT, hid.trans hU, rfl, trivial, hlo, hhi, ?_, ?_, ?_⟩
          · rw [hidx0]; rfl
          · dsimp only; rw [hA]; exact hEA
          · dsimp only; rw [hA]; exact hhA
          · dsimp only; rw [hA]; exact hbA
          · dsimp only; rw [hA]
            exact Rel.trans' (U := U) hRel hrel (fun j hj => hj) hUm
          · dsimp only; rw [hf, hfree]
          · dsimp only; rw [hl, hlen]
        · rw [hidx1v]; omega
      | cons h' rest' =>
        have hrl' : rest'.length + 1 = idx1.val := by rw [← hrl]; rfl
        have hidx0 : ¬ idx1 = 0#usize := fun e => by rw [e] at hrl'; simp at hrl'
        rw [if_neg hidx0]
        obtain ⟨i, hi, hiv⟩ := usub_tot idx1 1#usize (by rw [val_1_usize]; omega)
        rw [val_1_usize] at hiv
        rw [hi]; simp only [bind_tc_ok]
        have hirl : i.val = rest'.length := by rw [hiv]; omega
        obtain ⟨⟨c1, hc1, hc1v⟩, hchrest'⟩ := hchrest
        have hilt : i.val < chain.length := by
          rw [hirl]; exact (List.getElem?_eq_some_iff.mp hc1).1
        obtain ⟨p, hp⟩ := Apeiron.PrefixMasks.slice_index_ex chain i hilt
        rw [hp]; simp only [bind_tc_ok]
        have hp' : chain.val[i.val]? = some p := slice_index_ok chain i p hp
        rw [hirl, hc1] at hp'
        have hpc : c1 = p := Option.some.inj hp'
        have hpv : p.val = h'.a := by rw [← hpc]; exact hc1v
        obtain ⟨c', hSpc', hc'v, hha, ndh, hndh, hEoh, hchh⟩ := hSpc
        simp only [lift, bind_tc_ok]
        have hND2 : ND (plugI (frT g S) (h' :: rest')) := hND
        have hND3 : ND (plugI (frT h' (frT g S)) rest') := hND
        have hhaX : h'.a ∉ idxs (frT g S) := (nd_frT (nd_plugI_sub rest' hND3)).2.2.1
        have hAh : A[h'.a]? = some ndh := by rw [rel_out hrel hhaX]; exact hndh
        have hAlt : h'.a < A.length := (List.getElem?_eq_some_iff.mp hAh).1
        have hcastv : (UScalar.cast .Usize p).val = h'.a := by rw [cast_usize_u32, hpv]
        obtain ⟨n1, hn1⟩ := vec_index_ex self1.nodes (UScalar.cast .Usize p)
          (by rw [hA, hcastv]; exact hAlt)
        rw [hn1]; simp only [bind_tc_ok]
        have hn1' : A[h'.a]? = some n1 := by
          have h := idx_ok hn1
          rw [hA, hpv] at h
          exact h
        have hn1n : n1 = ndh := Option.some.inj (hn1'.symm.trans hAh)
        rw [hn1n]
        have hcnv : c.val = n.val := by rw [hcv, hnv]
        have hcn' : c = n := UScalar.eq_of_val_eq hcnv
        have hnX : n.val ∈ idxs (frT g S) := by
          rw [hnv]; exact mem_frT.mpr (Or.inr (Or.inl rfl))
        have hdisj := nd_plugI_disj (h' :: rest') hND2
        have hne : ∀ j ∈ idxs T', (A.set h'.a (setChild h'.d ndh newroot))[j]? = A[j]? :=
          fun j hj => List.getElem?_set_ne
            (show h'.a ≠ j from fun e => hhaX (by rw [e, ← hid]; exact hj))
        have hBT : BST o.nodes.val T' := bst_congr (fun j => (keyAt_rel hrel j).symm) hbA
        have hBsw : BST o.nodes.val (plugI T' (h' :: rest')) :=
          bst_plugI_swap (h' :: rest') hBST hBT hid
        have hUh : h'.a ∈ U := by
          rw [← hU]
          exact (mem_plugI rest' (frT h' (frT g S))).mpr
            (Or.inl (mem_frT.mpr (Or.inr (Or.inl rfl))))
        obtain ⟨hSp'', hEA'', hrel'', hHAg, _hout⟩ :=
          chain_relink (X := frT g S) (h := h') (rest' := rest') (ndh := ndh)
            (r := o.root) (x' := newroot) (T' := T')
            hND2 hrel hndh hSpc' hc'v hha hEoh hEA hid
        have key : ∀ v : alloc.vec.Vec ONode, v.val = A.set h'.a (setChild h'.d ndh newroot) →
            ChainInv ns0 U F0 N0 chain.val
              { nodes := v, free := self1.free, root := self1.root, len := self1.len } idx1 := by
          intro v hv
          refine ⟨h' :: rest', T', newroot, max sc (ht g.o) + 1, ?_, ⟨⟨c1, hc1, hc1v⟩, hchrest'⟩,
            ?_, hbal, ?_, ?_, ?_, ?_, ?_, ?_, hlo, hhi, ?_, ?_, ?_⟩
          · show idx1.val = (h' :: rest').length
            simp only [List.length_cons]; omega
          · show Emb v.val newroot T'
            rw [hv]; exact hEA''
          · show HOK v.val T'
            rw [hv]; exact hok_congr hhA (fun j hj => hne j hj)
          · show BST v.val (plugI T' (h' :: rest'))
            rw [hv]; exact bst_rel (hrel.trans hrel'') hBsw
          · exact nd_plugI_swap (h' :: rest') hND2 hndT hid
          · exact (idxs_plugI_eq (h' :: rest') hid).trans hU
          · show Spine v.val self1.root newroot (h' :: rest')
            rw [hv, hr]; exact hSp''
          · show AncOld v.val (h' :: rest') (max sc (ht g.o) + 1)
            rw [hv]; exact ancOld_hcongr (h' :: rest') hAncr hHAg
          · show Rel U ns0 v.val
            rw [hv]
            exact Rel.trans' (U := U) (Rel.trans' (U := U) hRel hrel (fun j hj => hj) hUm) hrel''
              (fun j hj => hj) (fun j hj => by rw [List.mem_singleton] at hj; rw [hj]; exact hUh)
          · show self1.free = F0
            rw [hf, hfree]
          · show self1.len = N0
            rw [hl, hlen]
        cases hhd : h'.d with
        | true =>
          have hcond : ndh.left = n := by
            have h1 := hchh
            rw [hhd] at h1
            simp [childOf] at h1
            rw [h1, hcn']
          rw [if_pos hcond]
          obtain ⟨n2, back, hn2⟩ := vec_index_mut_ex self1.nodes (UScalar.cast .Usize p)
            (by rw [hA, hcastv]; exact hAlt)
          rw [hn2]; rw [bind_pair_ok]
          obtain ⟨hn2', hback⟩ := idxm_ok hn2
          rw [hA, hpv] at hn2'
          have hn2n : n2 = ndh := Option.some.inj (hn2'.symm.trans hAh)
          rw [hn2n]
          refine ⟨ControlFlow.cont
            ({ self1 with nodes := back {ndh with left := newroot} }, idx1), rfl, ?_⟩
          refine ⟨key _ ?_, ?_⟩
          · rw [hback, hA, hpv, hhd]; rfl
          · rw [hidx1v]; omega
        | false =>
          have hcond : ¬ ndh.left = n := by
            intro hcl
            have h1 := hEoh
            rw [hhd] at h1
            simp [childOf] at h1
            have hnn : n.val ≠ nilN := by rw [hnv]; exact hga
            have hmem := emb_root_mem h1 (by rw [hcl]; exact hnn)
            rw [hcl] at hmem
            exact hdisj n.val (by
              simp only [spineIdx, List.mem_cons, List.mem_append]
              exact Or.inr (Or.inl hmem)) hnX
          rw [if_neg hcond]
          obtain ⟨n2, back, hn2⟩ := vec_index_mut_ex self1.nodes (UScalar.cast .Usize p)
            (by rw [hA, hcastv]; exact hAlt)
          rw [hn2]; rw [bind_pair_ok]
          obtain ⟨hn2', hback⟩ := idxm_ok hn2
          rw [hA, hpv] at hn2'
          have hn2n : n2 = ndh := Option.some.inj (hn2'.symm.trans hAh)
          rw [hn2n]
          refine ⟨ControlFlow.cont
            ({ self1 with nodes := back {ndh with right := newroot} }, idx1), rfl, ?_⟩
          refine ⟨key _ ?_, ?_⟩
          · rw [hback, hA, hpv, hhd]; rfl
          · rw [hidx1v]; omega
  · rw [if_neg hpos]
    exact ⟨ControlFlow.done o, rfl, trivial⟩

/-- `rebalance_chain_loop` is total on a `ChainInv` state under the absolute arena bound. -/
theorem ov_rebalance_chain_loop_ok (ns0 : List ONode) (U : List Nat) (F0 : alloc.vec.Vec U32)
    (N0 : Usize) (o0 : Ov) (chain : Slice U32) (idx0 : Usize)
    (hinv : ChainInv ns0 U F0 N0 chain.val o0 idx0)
    (hbound : o0.nodes.val.length + 3 ≤ 2 ^ 32 - 1) :
    ∃ o', hashtrinity.overflow.Overflow.rebalance_chain_loop CI o0 chain idx0 = ok o' := by
  have hb : ∀ st : Ov × Usize, ChainInv ns0 U F0 N0 chain.val st.1 st.2 →
      st.1.nodes.val.length + 3 ≤ 2 ^ 32 - 1 := by
    intro st hst
    rw [ChainInv.len_eq hst, ← ChainInv.len_eq hinv]
    exact hbound
  unfold hashtrinity.overflow.Overflow.rebalance_chain_loop
  refine loop_tot_inv
    (fun st : Ov × Usize =>
      hashtrinity.overflow.Overflow.rebalance_chain_loop.body CI chain st.1 st.2)
    (fun st => ChainInv ns0 U F0 N0 chain.val st.1 st.2) (fun st => st.2.val) ?_ ?_ (o0, idx0)
    hinv
  · rintro ⟨o, idx⟩ hinv'
    obtain ⟨cf, hcf, _⟩ := ov_rebalance_chain_body_ok ns0 U F0 N0 chain o idx hinv' (hb _ hinv')
    exact ⟨cf, hcf⟩
  · rintro ⟨o, idx⟩ ⟨o', idx'⟩ hinv' hcf
    obtain ⟨cf, hcf0, hpost⟩ :=
      ov_rebalance_chain_body_ok ns0 U F0 N0 chain o idx hinv' (hb _ hinv')
    rw [hcf0] at hcf
    have hcfeq : cf = ControlFlow.cont (o', idx') := Result.ok_injective hcf
    rw [hcfeq] at hpost
    exact hpost

/-- `rebalance_chain` is total on a `ChainInv` state at the full chain length, under the absolute
arena bound. -/
theorem ov_rebalance_chain_ok (ns0 : List ONode) (U : List Nat) (F0 : alloc.vec.Vec U32)
    (N0 : Usize) (o0 : Ov) (chain : Slice U32)
    (hinv : ChainInv ns0 U F0 N0 chain.val o0 (Slice.len chain))
    (hbound : o0.nodes.val.length + 3 ≤ 2 ^ 32 - 1) :
    ∃ o', hashtrinity.overflow.Overflow.rebalance_chain CI o0 chain = ok o' := by
  unfold hashtrinity.overflow.Overflow.rebalance_chain
  exact ov_rebalance_chain_loop_ok ns0 U F0 N0 o0 chain (Slice.len chain) hinv hbound

/-! ## Stage 5: `alloc` and the descent of `insert_node` are total -/

/-- `2 ^ 32 - 1 ≤ Usize.max`. -/
theorem nilN_le_usize_max : nilN ≤ Usize.max := by
  unfold nilN
  rw [Usize.max_def, Usize.numBits_def]
  rcases System.Platform.numBits_eq with h | h
  · show 2 ^ 32 - 1 ≤ 2 ^ System.Platform.numBits - 1; rw [h]
  · show 2 ^ 32 - 1 ≤ 2 ^ System.Platform.numBits - 1; rw [h]; decide

/-- `Vec::push` is total when the length is below `Usize.max`. -/
theorem vec_push_ex' {α : Type} (v : alloc.vec.Vec α) (x : α) (h : v.val.length < Usize.max) :
    ∃ v1, alloc.vec.Vec.push v x = ok v1 := by
  unfold alloc.vec.Vec.push
  dsimp only
  rw [dif_pos (by simp only [Bool.or_eq_true, decide_eq_true_eq]; right; omega)]
  exact ⟨_, rfl⟩

/-- `Vec::pop` is total. -/
theorem vec_pop_ex {α : Type} (v : alloc.vec.Vec α) :
    ∃ x v1, alloc.vec.Vec.pop Global v = ok (x, v1) := by
  unfold alloc.vec.Vec.pop
  by_cases h : v.val.length = 0
  · exact ⟨none, v, by rw [dif_pos h]⟩
  · exact ⟨_, _, by rw [dif_neg h]⟩

/-- **`ov_alloc_ok`.**  `Overflow::alloc` returns `ok` under `alloc_spec`'s hypotheses: the free-list
branch reads an in-range slot, and the push branch has `Usize` room because
`o.nodes.val.length < nilN ≤ Usize.max`. -/
theorem ov_alloc_ok (o : Ov) (T : OTree) (coord k : U64) (v : I64)
    (_hE : Emb o.nodes.val o.root T)
    (_hfn : (o.free.val.map (fun x => x.val)).Nodup)
    (hfl : ∀ j ∈ o.free.val, j.val < o.nodes.val.length ∧ j.val ∉ idxs T)
    (hcap : o.nodes.val.length < nilN) :
    ∃ i o', hashtrinity.overflow.Overflow.alloc CI o coord k v = ok (i, o') := by
  obtain ⟨popped, v1, hpop⟩ := vec_pop_ex o.free
  unfold hashtrinity.overflow.Overflow.alloc
  simp only [lift, bind_tc_ok]
  set nd : ONode :=
    { key := coord,
      entries := alloc.slice.Slice.into_vec
        (Std.Array.to_slice (Array.make 1#usize [(k, some v)] :
          Array (U64 × Option I64) 1#usize)),
      left := hashtrinity.overflow.NIL, right := hashtrinity.overflow.NIL,
      height := 1#u32, alive := true } with hnd
  rw [hpop]
  rw [bind_tc_ok]
  change ∃ i' o', (match popped with
    | none => (do let v2 ← alloc.vec.Vec.push o.nodes nd; let i1 ← alloc.vec.Vec.len v2 - 1#usize; ok (UScalar.cast .U32 i1, { o with nodes := v2, free := v1 }))
    | some i => (do let (_, back) ← alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice ONode) o.nodes (UScalar.cast .Usize i); ok (i, { o with nodes := back nd, free := v1 }))) = ok (i', o')
  cases popped with
  | none =>
    have hlenu : o.nodes.val.length < Usize.max := by
      have := nilN_le_usize_max; unfold nilN at hcap this; omega
    obtain ⟨v2, hpush⟩ := vec_push_ex' o.nodes nd hlenu
    rw [hpush]
    simp only [bind_tc_ok]
    have hlen2 : v2.val.length = o.nodes.val.length + 1 := by
      rw [vec_push_ok _ _ _ hpush]; simp
    obtain ⟨i1, hi1, hi1v⟩ := usub_tot (alloc.vec.Vec.len v2) 1#usize (by
      have h1 : (alloc.vec.Vec.len v2).val = v2.val.length := alloc.vec.Vec.len_val v2
      rw [h1, hlen2, val_1_usize]; omega)
    rw [hi1]; simp only [bind_tc_ok]
    exact ⟨_, _, rfl⟩
  | some i0 =>
    rcases pop_spec o.free v1 (some i0) hpop with hbad | ⟨y, hfy, hyeq⟩
    · exact absurd hbad.2.1 (by simp)
    · have hy : y = i0 := (Option.some.inj hyeq).symm
      rw [hy] at hfy
      have hi0 : i0 ∈ o.free.val := by rw [hfy]; simp
      have hi0lt : i0.val < o.nodes.val.length := (hfl i0 hi0).1
      have hcast : (UScalar.cast .Usize i0).val < o.nodes.val.length := by
        rw [cast_usize_u32]; exact hi0lt
      obtain ⟨n, back, hmut⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize i0) hcast
      change ∃ i' o', (do let (_, back') ← alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice ONode) o.nodes (UScalar.cast .Usize i0); ok (i0, { o with nodes := back' nd, free := v1 })) = ok (i', o')
      rw [hmut]; simp only [bind_tc_ok]
      exact ⟨i0, _, rfl⟩

/-- `Overflow::insert_node_loop` terminates: each `cont` descends to a strictly smaller subtree, and
the growing path stays below `Usize.max` because it grows by one per descent while `ht` drops by at
least one. -/
theorem insert_node_loop_total (v : alloc.vec.Vec ONode) (i : U32) (key : U64) :
    ∀ (S : OTree) (chain : alloc.vec.Vec U32) (cur : U32),
      Emb v.val cur S → cur.val ≠ nilN → chain.val.length + ht S < Usize.max →
      ∃ r, hashtrinity.overflow.Overflow.insert_node_loop v i key chain cur = ok r := by
  intro S
  induction S with
  | leaf =>
    intro chain cur hE hnil hclen
    exact absurd hE hnil
  | node Sl ii Sr ihl ihr =>
    intro chain cur hE hnil hclen
    obtain ⟨hcv, hii, nd, hnd, hEl, hEr⟩ := hE
    have hcurlt : cur.val < v.val.length := by
      rw [hcv]; exact (List.getElem?_eq_some_iff.mp hnd).1
    have hcastlt : (UScalar.cast .Usize cur).val < v.val.length := by
      rw [cast_usize_u32]; exact hcurlt
    have nd_eq : ∀ n, alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v
        (UScalar.cast .Usize cur) = ok n → n = nd := by
      intro n hn
      have h := vec_index_ok v (UScalar.cast .Usize cur) n hn
      rw [cast_usize_u32, hcv, hnd] at h
      exact (Option.some.inj h).symm
    have hpushlt : chain.val.length < Usize.max := by
      have : 0 < ht (.node Sl ii Sr) := by simp only [ht]; omega
      omega
    obtain ⟨chain1, hpush⟩ := vec_push_ex' chain cur hpushlt
    unfold hashtrinity.overflow.Overflow.insert_node_loop
    rw [loop.eq_def]
    unfold hashtrinity.overflow.Overflow.insert_node_loop.body
    simp only [lift, bind_tc_ok]
    rw [hpush]
    simp only [bind_tc_ok]
    obtain ⟨n, hn⟩ := vec_index_ex v (UScalar.cast .Usize cur) hcastlt
    rw [hn]
    simp only [bind_tc_ok]
    have hnnd : n = nd := nd_eq n hn
    rw [hnnd]
    by_cases hlt : key < nd.key
    · rw [if_pos hlt]
      by_cases hnl : nd.left = hashtrinity.overflow.NIL
      · rw [if_pos hnl]
        obtain ⟨p, back, hm⟩ := vec_index_mut_ex v (UScalar.cast .Usize cur) hcastlt
        rw [hm]
        simp only [bind_tc_ok]
        change ∃ r, (Std.bind (ok (ControlFlow.done (back { p with left := i }, chain1))) _) = ok r
        rw [bind_ok]
        exact ⟨_, rfl⟩
      · rw [if_neg hnl]
        rw [bind_ok]
        have hnlv : nd.left.val ≠ nilN := fun hh => hnl ((NIL_eq_iff _).mpr hh)
        have hclen' : chain1.val.length + ht Sl < Usize.max := by
          have hch1 : chain1.val.length = chain.val.length + 1 := by
            rw [vec_push_ok chain chain1 cur hpush]; simp
          have hht : ht Sl + 1 ≤ ht (.node Sl ii Sr) := by simp only [ht]; omega
          omega
        simpa only [hashtrinity.overflow.Overflow.insert_node_loop,
          hashtrinity.overflow.Overflow.insert_node_loop.body, lift, bind_tc_ok]
          using ihl chain1 nd.left hEl hnlv hclen'
    · rw [if_neg hlt]
      by_cases hnr : nd.right = hashtrinity.overflow.NIL
      · rw [if_pos hnr]
        obtain ⟨p, back, hm⟩ := vec_index_mut_ex v (UScalar.cast .Usize cur) hcastlt
        rw [hm]
        simp only [bind_tc_ok]
        change ∃ r, (Std.bind (ok (ControlFlow.done (back { p with right := i }, chain1))) _) = ok r
        rw [bind_ok]
        exact ⟨_, rfl⟩
      · rw [if_neg hnr]
        rw [bind_ok]
        have hnrv : nd.right.val ≠ nilN := fun hh => hnr ((NIL_eq_iff _).mpr hh)
        have hclen' : chain1.val.length + ht Sr < Usize.max := by
          have hch1 : chain1.val.length = chain.val.length + 1 := by
            rw [vec_push_ok chain chain1 cur hpush]; simp
          have hht : ht Sr + 1 ≤ ht (.node Sl ii Sr) := by simp only [ht]; omega
          omega
        simpa only [hashtrinity.overflow.Overflow.insert_node_loop,
          hashtrinity.overflow.Overflow.insert_node_loop.body, lift, bind_tc_ok]
          using ihr chain1 nd.right hEr hnrv hclen'

/-- **`ov_insert_node_ok`.**  `Overflow::insert_node` returns `ok` under `insert_node_spec`'s
hypotheses: the descent `insert_node_loop` terminates (`insert_node_loop_total`), and the closing
`rebalance_chain` terminates (`ov_rebalance_chain_ok`) on the `ChainInv` the descent produces. -/
theorem ov_insert_node_ok (o : Ov) (T : OTree) (i : U32) (ni : ONode)
    (hE : Emb o.nodes.val o.root T) (hB : BST o.nodes.val T) (hND : ND T)
    (hH : HOK o.nodes.val T) (hBal : Bal T)
    (hi : i.val ∉ idxs T) (hii : i.val ≠ nilN) (hni : o.nodes.val[i.val]? = some ni)
    (hnl : ni.left.val = nilN) (hnr : ni.right.val = nilN) (hnh : ni.height.val = 1)
    (hkey : ∀ j ∈ idxs T, keyAt o.nodes.val j ≠ ni.key.val)
    (hbound : o.nodes.val.length + 3 ≤ 2 ^ 32 - 1) :
    ∃ o', hashtrinity.overflow.Overflow.insert_node CI o i = ok o' := by
  unfold hashtrinity.overflow.Overflow.insert_node
  by_cases hroot : o.root = hashtrinity.overflow.NIL
  · rw [if_pos hroot]
    exact ⟨_, rfl⟩
  · rw [if_neg hroot]
    simp only [lift, bind_tc_ok]
    have hilt : i.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hni).1
    have hcast : (UScalar.cast .Usize i).val < o.nodes.val.length := by
      rw [cast_usize_u32]; exact hilt
    obtain ⟨n, hn⟩ := vec_index_ex o.nodes (UScalar.cast .Usize i) hcast
    rw [hn]; simp only [bind_tc_ok]
    have hnni : n = ni := by
      have h := vec_index_ok o.nodes (UScalar.cast .Usize i) n hn
      rw [cast_usize_u32, hni] at h
      exact (Option.some.inj h).symm
    rw [hnni]
    have hroot' : o.root.val ≠ nilN := fun hh => hroot ((NIL_eq_iff _).mpr hh)
    have hclen : (alloc.vec.Vec.with_capacity U32 64#usize).val.length + ht T < Usize.max := by
      have h0 : (alloc.vec.Vec.with_capacity U32 64#usize).val.length = 0 := by
        simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.val, Slice.new_val]
      rw [h0]
      have h1 := ht_le_arena_len hE hND
      have h2 := nilN_le_usize_max
      unfold nilN at h2
      omega
    obtain ⟨⟨v, chain1⟩, hloop⟩ :=
      insert_node_loop_total o.nodes i ni.key T (alloc.vec.Vec.with_capacity U32 64#usize) o.root
        hE hroot' hclen
    rw [hloop]; simp only [bind_tc_ok]
    have hpost := insert_node_loop_spec o.nodes i ni.key o.root T hND hkey
      (alloc.vec.Vec.with_capacity U32 64#usize) o.root
      ⟨[], T, rfl, trivial, rfl, hE, hroot', rfl, fun g hg => absurd hg (by simp)⟩ (v, chain1) hloop
    obtain ⟨g0, G, nd, hlenC, hchC, hDescC, hplugC, hnd, hv, hSpC⟩ := hpost
    have hlenC' : chain1.val.length = G.length + 1 := hlenC
    have hv' : v.val = o.nodes.val.set g0.a (setChild g0.d nd i) := hv
    have hg0T : g0.a ∈ idxs T := by
      rw [← hplugC]; exact (mem_plugI (g0 :: G) .leaf).mpr (Or.inr (by simp [spineIdx]))
    have hig : i.val ≠ g0.a := fun e => hi (by rw [e]; exact hg0T)
    have hvi : v.val[i.val]? = some ni := by
      rw [hv', List.getElem?_set_ne (Ne.symm hig)]; exact hni
    have hrelv : Rel [g0.a] o.nodes.val v.val := by
      rw [hv']; exact Rel_set hnd (sameD_setChild _ _ _)
    have hkv : ∀ j, keyAt v.val j = keyAt o.nodes.val j := keyAt_rel hrelv
    have hhag : ∀ j, HAg o.nodes.val v.val j := by
      intro j nd0 hnd0
      by_cases hj : j = g0.a
      · subst hj
        rw [hnd] at hnd0
        have := Option.some.inj hnd0
        subst this
        exact ⟨setChild g0.d nd i, by
          rw [hv']; exact List.getElem?_set_self (List.getElem?_eq_some_iff.mp hnd).1,
          height_setChild _ _ _⟩
      · exact ⟨nd0, by rw [hv', List.getElem?_set_ne (Ne.symm hj)]; exact hnd0, rfl⟩
    have hHv : HOK v.val (plugI .leaf (g0 :: G)) := by
      rw [hplugC]; exact hok_hcongr hH (fun j _ => hhag j)
    have hmemN : ∀ j, j ∈ idxs (.node .leaf i.val .leaf) ↔ j ∈ idxs .leaf ∨ j = i.val := by
      intro j; simp [idxs]
    have hNDN : ND (.node .leaf i.val .leaf) :=
      ⟨trivial, trivial, by simp [idxs], by simp [idxs], fun j hj => by simp [idxs] at hj⟩
    have hBN : BST v.val (.node .leaf i.val .leaf) :=
      ⟨trivial, trivial, fun j hj => by simp [idxs] at hj, fun j hj => by simp [idxs] at hj⟩
    have hinv : ChainInv v.val (idxs (plugI (.node .leaf i.val .leaf) (g0 :: G))) o.free o.len
        (alloc.vec.Vec.deref chain1).val { o with nodes := v }
        (Slice.len (alloc.vec.Vec.deref chain1)) := by
      refine ⟨g0 :: G, .node .leaf i.val .leaf, i, 0, ?_, ?_, ⟨rfl, hii, ni, hvi, hnl, hnr⟩,
        ⟨trivial, trivial, by simp [ht], by simp [ht]⟩, ⟨trivial, trivial, ni, hvi, by simp [ht, hnh]⟩,
        ?_, ?_, rfl, hSpC, ?_, Nat.zero_le _, by simp [ht], Rel.refl _ _, rfl, rfl⟩
      · rw [Slice.len_val]
        show (alloc.vec.Vec.deref chain1).val.length = G.length + 1
        rw [deref_val]; exact hlenC'
      · rw [deref_val]; exact hchC
      · exact bst_plugI_ins (keyAt_get hvi) (g0 :: G) (X := .leaf)
          (X' := .node .leaf i.val .leaf) (descK_congr hkv hDescC) hmemN
          (by rw [hplugC]; exact bst_rel hrelv hB) hBN
      · exact nd_plugI_ins (g0 :: G) (X := .leaf) (X' := .node .leaf i.val .leaf) hmemN
          (by rw [hplugC]; exact hND) hNDN (by rw [hplugC]; exact hi)
      · exact ancOld_of_valid (g0 :: G) (X := .leaf) (by rw [hplugC]; exact hBal) hHv
    have hbound' : v.val.length + 3 ≤ 2 ^ 32 - 1 := by
      rw [hv', List.length_set]; exact hbound
    exact ov_rebalance_chain_ok v.val (idxs (plugI (.node .leaf i.val .leaf) (g0 :: G))) o.free o.len
      { o with nodes := v } (alloc.vec.Vec.deref chain1) hinv hbound'

/-! ## Stage 6: `Overflow::insert_new_at` is total

The lead-recorded domain (W11 finding): `insert_new_at_spec`'s premises do not make the executed body
total.  Three extra premises are the smallest real bounds of the body — the `Usize` length room
(`self.len + 1#usize`), the `Usize` entry-vector room at every node of the tree (the non-`NIL`
branch's `Vec::push`), and the arena `+4` bound (`hcap`'s `+3` for the closing `insert_node`
rebalance plus one slot for the fresh node).  The proof mirrors `insert_new_at_spec` but closes each
branch with the totality lemmas `ov_alloc_ok` / `ov_insert_node_ok` instead of taking a successful
run as a hypothesis. -/

/-- The common tail of `Overflow::insert_new_at`: read the entry vector of the returned node and take
its last index.  The returned node always holds at least the pushed entry, so the subtraction does not
underflow. -/
theorem ov_insert_new_at_tail_ok (S : Ov) (node : U32)
    (hlt : node.val < S.nodes.val.length)
    (hent : ∀ nd, S.nodes.val[node.val]? = some nd → 0 < nd.entries.val.length) :
    ∃ nd e o', (do
      let i1 ← lift (UScalar.cast .Usize node)
      let n1 ← alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) S.nodes i1
      let i2 := alloc.vec.Vec.len n1.entries
      let e ← i2 - 1#usize
      ok ((node, e), S)) = ok ((nd, e), o') := by
  obtain ⟨n1, hn1⟩ := vec_index_ex S.nodes (UScalar.cast .Usize node)
    (by rw [cast_usize_u32]; exact hlt)
  have hnd : S.nodes.val[node.val]? = some n1 := by
    have h := vec_index_ok S.nodes (UScalar.cast .Usize node) n1 hn1
    rwa [cast_usize_u32] at h
  have hpos : 1 ≤ (alloc.vec.Vec.len n1.entries).val := by
    rw [alloc.vec.Vec.len_val]; exact hent n1 hnd
  obtain ⟨e, he, _⟩ := usub_tot (alloc.vec.Vec.len n1.entries) 1#usize (by simpa using hpos)
  refine ⟨node, e, S, ?_⟩
  simp only [lift, bind_tc_ok]
  rw [hn1]; simp only [bind_tc_ok]
  rw [he]; simp only [bind_tc_ok]

/-- The `NIL` branch of `Overflow::insert_new_at`: allocate a fresh node and link it.  The returned
node is the fresh index; it carries the pushed entry. -/
theorem ov_insert_new_at_branch_nil (o : Ov) (T : OTree) (i : Usize) (coord k : U64) (v : I64)
    (hI : OInv o T) (hcap' : o.nodes.val.length + 4 ≤ 2 ^ 32 - 1)
    (hkey0 : ∀ j ∈ idxs T, keyAt o.nodes.val j ≠ coord.val) :
    ∃ S node, (do
      let (i1, self2) ← hashtrinity.overflow.Overflow.alloc CI { o with len := i } coord k v
      let self3 ← hashtrinity.overflow.Overflow.insert_node CI self2 i1
      ok (self3, i1)) = ok (S, node) ∧
      node.val < S.nodes.val.length ∧
      (∀ nd, S.nodes.val[node.val]? = some nd → 0 < nd.entries.val.length) := by
  obtain ⟨hE, hB, hND, hH, hBal, hfn, hfl⟩ := hI
  have hcap : o.nodes.val.length < nilN := by unfold nilN; omega
  obtain ⟨i1, o2, halloc⟩ := ov_alloc_ok { o with len := i } T coord k v hE hfn hfl hcap
  obtain ⟨ni, hni, hnik, hnie, hnia, hnl, hnr, hnh, hninil, hnT, hnlt, hother, hroot2, hlen2, hfn2,
    hfl2, hlenA⟩ :=
    alloc_spec { o with len := i } o2 T i1 coord k v hE hfn hfl hcap halloc
  have hneT : ∀ j ∈ idxs T, j ≠ i1.val := fun j hj e => hnT (by rw [← e]; exact hj)
  have hagree : ∀ j ∈ idxs T, o2.nodes.val[j]? = o.nodes.val[j]? := fun j hj => hother j (hneT j hj)
  have hE2 : Emb o2.nodes.val o2.root T := by rw [hroot2]; exact emb_congr hE hagree
  have hB2 : BST o2.nodes.val T :=
    bst_congr' (fun j hj => by unfold keyAt; rw [hagree j hj]) hB
  have hH2 : HOK o2.nodes.val T := hok_congr hH hagree
  have hkey' : ∀ j ∈ idxs T, keyAt o2.nodes.val j ≠ ni.key.val := by
    intro j hj
    have h1 : keyAt o2.nodes.val j = keyAt o.nodes.val j := by unfold keyAt; rw [hagree j hj]
    rw [h1, hnik]; exact hkey0 j hj
  have hlenA' : o2.nodes.val.length ≤ o.nodes.val.length + 1 := hlenA
  have hbound : o2.nodes.val.length + 3 ≤ 2 ^ 32 - 1 := by omega
  obtain ⟨o3, hins⟩ :=
    ov_insert_node_ok o2 T i1 ni hE2 hB2 hND hH2 hBal hnT hninil hni hnl hnr hnh hkey' hbound
  obtain ⟨T', hE3, hB3, hND3, hH3, hBal3, hperm, hrel3, hf3, hl3⟩ :=
    insert_node_spec o2 o3 T i1 ni hE2 hB2 hND hH2 hBal hnT hninil hni hnl hnr hnh hkey' hins
  have hi1mem : i1.val ∈ idxs T' := by
    rw [hperm.mem_iff]; exact List.mem_cons_self ..
  have hlt : i1.val < o3.nodes.val.length := by
    obtain ⟨nd, hnd, _⟩ := emb_mem hE3 i1.val hi1mem
    exact (List.getElem?_eq_some_iff.mp hnd).1
  have hent : ∀ nd, o3.nodes.val[i1.val]? = some nd → 0 < nd.entries.val.length := by
    intro nd hnd
    obtain ⟨nd', hnd', hsd, _⟩ := hrel3.2 i1.val ni hni
    have hndeq : nd = nd' := Option.some.inj (hnd.symm.trans hnd')
    subst hndeq
    rw [← hsd.2.1, hnie]; simp
  refine ⟨o3, i1, ?_, hlt, hent⟩
  rw [halloc]; rw [bind_pair_ok]
  rw [hins]; simp only [bind_tc_ok]

/-- The non-`NIL` branch of `Overflow::insert_new_at`: read the matching node and push the entry onto
its entry vector.  The returned node is the matched index. -/
theorem ov_insert_new_at_branch_else (o : Ov) (T : OTree) (n : U32) (i : Usize) (_coord k : U64)
    (v : I64) (hE : Emb o.nodes.val o.root T) (hnmem : n.val ∈ idxs T)
    (hentries : ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.entries.val.length < Usize.max) :
    ∃ S, (do
      let i1 ← lift (UScalar.cast .Usize n)
      let (n1, index_mut_back) ←
        alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice ONode) o.nodes i1
      let v1 ← alloc.vec.Vec.push n1.entries (k, some v)
      let v2 := index_mut_back { n1 with entries := v1 }
      ok ({ o with nodes := v2, len := i }, n)) = ok (S, n) ∧
      n.val < S.nodes.val.length ∧
      (∀ nd, S.nodes.val[n.val]? = some nd → 0 < nd.entries.val.length) := by
  obtain ⟨nd0, hnd0, _⟩ := emb_mem hE n.val hnmem
  have hnlt : n.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd0).1
  have hcastlt : (UScalar.cast .Usize n).val < o.nodes.val.length := by
    rw [cast_usize_u32]; exact hnlt
  obtain ⟨n1, back, hmut⟩ := vec_index_mut_ex o.nodes (UScalar.cast .Usize n) hcastlt
  obtain ⟨hn1, hback⟩ := idxm_ok hmut
  have hent : n1.entries.val.length < Usize.max := hentries n.val hnmem n1 hn1
  obtain ⟨v1, hpush⟩ := vec_push_ex' n1.entries (k, some v) hent
  have hv1 : v1.val = n1.entries.val ++ [(k, some v)] := vec_push_ok _ _ _ hpush
  refine ⟨{ o with nodes := back { n1 with entries := v1 }, len := i }, ?_, ?_, ?_⟩
  · simp only [lift, bind_tc_ok]
    rw [hmut]; rw [bind_pair_ok]
    rw [hpush]; simp only [bind_tc_ok]
  · rw [hback]; simp only [List.length_set]; exact hnlt
  · intro nd hnd
    rw [hback] at hnd
    rw [List.getElem?_set_self hnlt] at hnd
    have hndeq : ({ n1 with entries := v1 } : ONode) = nd := Option.some.inj hnd
    subst hndeq
    simp [hv1]

/-- **`ov_insert_new_at_ok`.**  `Overflow::insert_new_at` returns `ok` under the stated domain: the
`Usize` length room (`hlen`), `Usize` entry-vector room at every node of the tree (`hentries`), the
arena bound `hcap'` (`length + 4 ≤ 2 ^ 32 - 1`; `+3` for the closing `insert_node` rebalance and one
slot for the fresh node), and the absence of the key (`habs`).  These three are the lead-recorded
smallest real bounds of the extracted body; `insert_new_at_spec`'s premises do not carry them. -/
theorem ov_insert_new_at_ok (o : Ov) (T : OTree) (coord k : U64) (v : I64)
    (hI : OInv o T) (hlen : o.len.val < Usize.max)
    (hentries : ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.entries.val.length < Usize.max)
    (hcap' : o.nodes.val.length + 4 ≤ 2 ^ 32 - 1)
    (_habs : ¬ Present o.nodes.val T coord k) :
    ∃ nd e o', hashtrinity.overflow.Overflow.insert_new_at CI o coord k v = ok ((nd, e), o') := by
  have hlen_lt : o.len.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
    rw [val_1_usize]
    have hlen' := hlen
    rw [Usize.max_def, Usize.numBits_def] at hlen'
    have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
    omega
  obtain ⟨i, hi, _⟩ := uadd_tot o.len 1#usize hlen_lt
  unfold hashtrinity.overflow.Overflow.insert_new_at
  rw [hi]
  simp only [bind_tc_ok]
  obtain ⟨n, hn⟩ := find_node_loop_total o.nodes coord T o.root hI.1
  have hnfn : hashtrinity.overflow.Overflow.find_node CI { o with len := i } coord = ok n := hn
  rw [hnfn]; simp only [bind_tc_ok]
  have hspec := find_node_spec o.nodes coord T o.root hI.1 hI.2.1 n hn
  by_cases hnil : n = hashtrinity.overflow.NIL
  · rw [if_pos hnil]
    have hkey0 : ∀ j ∈ idxs T, keyAt o.nodes.val j ≠ coord.val := by
      rcases hspec with ⟨_, hno⟩ | ⟨hm, _⟩
      · exact hno
      · rw [(NIL_eq_iff n).mp hnil] at hm
        obtain ⟨_, _, hne⟩ := emb_mem hI.1 _ hm
        exact absurd rfl hne
    obtain ⟨S, node, hbr, hlt, hent⟩ := ov_insert_new_at_branch_nil o T i coord k v hI hcap' hkey0
    rw [hbr]; rw [bind_pair_ok]
    exact ov_insert_new_at_tail_ok S node hlt hent
  · rw [if_neg hnil]
    have hnmem : n.val ∈ idxs T := by
      rcases hspec with ⟨hm, _⟩ | ⟨hm, _⟩
      · exact absurd (UScalar.eq_of_val_eq (hm.trans NIL_val.symm)) hnil
      · exact hm
    obtain ⟨S, hbr, hlt, hent⟩ := ov_insert_new_at_branch_else o T n i coord k v hI.1 hnmem hentries
    rw [hbr]; rw [bind_pair_ok]
    exact ov_insert_new_at_tail_ok S n hlt hent

/-- The premise set of `ov_insert_new_at_ok` is jointly satisfiable at the empty overflow tree
(`nodes = []`, `free = []`, root `NIL`, `len = 0`). -/
example (coord k : U64) (_v : I64) :
    ∃ (o : Ov) (T : OTree), OInv o T ∧ o.len.val < Usize.max ∧
      (∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.entries.val.length < Usize.max) ∧
      o.nodes.val.length + 4 ≤ 2 ^ 32 - 1 ∧ ¬ Present o.nodes.val T coord k := by
  refine ⟨{ nodes := alloc.vec.Vec.from ([] : List (hashtrinity.overflow.Node U64 I64)) (by simp),
            free := alloc.vec.Vec.from ([] : List U32) (by simp),
            root := hashtrinity.overflow.NIL, len := 0#usize }, .leaf,
    ⟨NIL_val, trivial, trivial, trivial, trivial, ?_, ?_⟩, ?_, ?_, ?_, ?_⟩
  · simp [alloc.vec.Vec.from_val]
  · intro j hj; simp [alloc.vec.Vec.from_val] at hj
  · have h0 : (0#usize).val = 0 := rfl
    rw [h0]
    have hm := nilN_le_usize_max
    unfold nilN at hm
    omega
  · intro j hj; simp [idxs] at hj
  · rw [show (alloc.vec.Vec.from ([] : List (hashtrinity.overflow.Node U64 I64)) (by simp)).val.length = 0
        by simp [alloc.vec.Vec.from_val]]
    norm_num
  · intro h; obtain ⟨j, hj, _⟩ := h; simp [idxs] at hj

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.ov_h_ok
#audit Apeiron.MapExecuted.ov_update_ok
#audit Apeiron.MapExecuted.ov_update_ok_hmax
#audit Apeiron.MapExecuted.ov_rotate_right_ok
#audit Apeiron.MapExecuted.ov_rotate_left_ok
#audit Apeiron.MapExecuted.ov_rebalance_ok
#audit Apeiron.MapExecuted.bind_pair_ok
#audit Apeiron.MapExecuted.ChainInv.len_eq
#audit Apeiron.MapExecuted.ht_le_arena_len
#audit Apeiron.MapExecuted.ov_rebalance_frT_ok
#audit Apeiron.MapExecuted.ov_rebalance_chain_body_ok
#audit Apeiron.MapExecuted.ov_rebalance_chain_loop_ok
#audit Apeiron.MapExecuted.ov_rebalance_chain_ok
#audit Apeiron.MapExecuted.nilN_le_usize_max
#audit Apeiron.MapExecuted.vec_push_ex'
#audit Apeiron.MapExecuted.vec_pop_ex
#audit Apeiron.MapExecuted.ov_alloc_ok
#audit Apeiron.MapExecuted.insert_node_loop_total
#audit Apeiron.MapExecuted.ov_insert_node_ok
#audit Apeiron.MapExecuted.ov_insert_new_at_tail_ok
#audit Apeiron.MapExecuted.ov_insert_new_at_branch_nil
#audit Apeiron.MapExecuted.ov_insert_new_at_branch_else
#audit Apeiron.MapExecuted.ov_insert_new_at_ok
