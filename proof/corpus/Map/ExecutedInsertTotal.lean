-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W7): totality of the
-- executed insert and resize.
import Map.ExecutedResize
import Map.ExecutedBuild
import Map.ExecutedReadTotal
import Map.ExecutedReadCoordTotal
import Map.ExecutedOverflowAvl
import Map.ExecutedOverflowChain
-- Reuse the tested `with_bits` construction helpers (`from_elem_ok`, `alloc_uninit_ok`,
-- `usize_max_ge`); `Map.ExecutedBuildWitness` imports `Map.ExecutedBuild` and does not depend on
-- this module.
import Map.ExecutedBuildWitness
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
slot for the fresh node).  These are the lead-recorded smallest real bounds of the extracted body;
`insert_new_at_spec`'s premises do not carry them.  No absence premise is required: the body's
`find_node` result already splits the `alloc` branch, so the premise is vestigial (W14). -/
theorem ov_insert_new_at_ok (o : Ov) (T : OTree) (coord k : U64) (v : I64)
    (hI : OInv o T) (hlen : o.len.val < Usize.max)
    (hentries : ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.entries.val.length < Usize.max)
    (hcap' : o.nodes.val.length + 4 ≤ 2 ^ 32 - 1) :
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

/-! ## Stage 7: the keyed capped probe is total under the weak `matches` premise

W13's premise correction: W12's draft stated `hm : ∀ keys i c k, ks.«matches» keys i c k = ok true`,
which forces the first confirmation and collapses the probe.  Totality needs only that `matches`
returns *some* `Bool`.  The existing cold-probe totality lemmas
(`probe_rest_with_total`, `probe_capped_cold_loop_total`, `probe_capped_cold_total`) are proved
against `= ok true`; under the weak premise the loop can take its `cont` branch, so this module
carries weak-premise twins.  The `cont` obligations become measure-decrease obligations: the rest
loop increases its index, the cold loop clears the lowest set bit of `hits` (`hits &&& (hits - 1) ≤
hits - 1 < hits`). -/

/-- `Confirm::confirm` returns *some* `Bool` under the weak `matches` premise (cf. `confirm_ok'`,
which needs `= ok true` to force the result). -/
theorem confirm_ex_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (q : hashtrinity.overflow.ByKey U64) (keys : KS) (i : Usize) :
    ∃ b, hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm.confirm
      core.cmp.EqU64 CI ks q keys i = ok b := by
  obtain ⟨b, hb⟩ := hm keys i q.coord q.k
  exact ⟨b, by
    unfold hashtrinity.overflow.ByKey.Insts.Apeiron_wordHashtrinityOverflowConfirm.confirm
    exact hb⟩

/-- The scalar tail of the capped probe is total under the weak `matches` premise; the loop's
`cont` branch is closed by the decreasing `limit - j` measure instead of the `= ok true`
contradiction. -/
theorem probe_rest_with_total_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (s : Tbl KS) (c : U64) (j limit : U32) (q : hashtrinity.overflow.ByKey U64)
    (hb : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hjl : j.val ≤ limit.val) (hlim : limit.val ≤ limN s.bits.val) :
    ∃ r, hashtrinity.overflow.Table.probe_rest_with core.cmp.EqU64 CI ks (ConfI ks) s c j limit q
      = ok r := by
  unfold hashtrinity.overflow.Table.probe_rest_with hashtrinity.overflow.Table.probe_rest_with_loop
  refine loop_tot_inv
    (fun j => hashtrinity.overflow.Table.probe_rest_with_loop.body core.cmp.EqU64 CI ks
      (ConfI ks) s c limit q j)
    (fun j => j.val ≤ limit.val) (fun j => limit.val - j.val) ?_ ?_ j hjl
  · intro j hj
    unfold hashtrinity.overflow.Table.probe_rest_with_loop.body
    by_cases hjl' : j < limit
    · rw [if_pos hjl']
      have hjlt : j.val < limit.val := (UScalar.lt_equiv _ _).mp hjl'
      obtain ⟨i, hid, hilt⟩ := descend_total ks s c j hb hb64 hlen (lt_of_lt_of_le hjlt hlim)
      rw [hid]
      simp only [bind_tc_ok]
      obtain ⟨sc, hsc⟩ := get_unchecked_ex s.coords i hilt
      rw [hsc]
      simp only [bind_tc_ok]
      by_cases hsce : sc = c
      · rw [if_pos hsce]
        obtain ⟨b, hb'⟩ := confirm_ex_weak ks hm q s.keys i
        rw [hb']
        simp only [bind_tc_ok]
        by_cases hbt : b = true
        · rw [if_pos hbt]; exact ⟨_, rfl⟩
        · rw [if_neg hbt]
          by_cases hse : sc = hashtrinity.EMPTY
          · rw [if_pos hse]; exact ⟨_, rfl⟩
          · rw [if_neg hse]
            obtain ⟨j1, hj1, _⟩ := uadd_tot j 1#u32
              (by rw [val_1_u32]; exact u32_sum_lt j.val 1 (by have := limN_le_16 s.bits.val; omega) (by norm_num))
            rw [hj1]
            simp only [bind_tc_ok]
            exact ⟨_, rfl⟩
      · rw [if_neg hsce]
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse]; exact ⟨_, rfl⟩
        · rw [if_neg hse]
          obtain ⟨j1, hj1, _⟩ := uadd_tot j 1#u32
            (by rw [val_1_u32]; exact u32_sum_lt j.val 1 (by have := limN_le_16 s.bits.val; omega) (by norm_num))
          rw [hj1]
          simp only [bind_tc_ok]
          exact ⟨_, rfl⟩
    · rw [if_neg hjl']
      exact ⟨_, rfl⟩
  · intro j j' hj hcf
    unfold hashtrinity.overflow.Table.probe_rest_with_loop.body at hcf
    by_cases hjl' : j < limit
    · rw [if_pos hjl'] at hcf
      obtain ⟨i, _, hcf⟩ := bind_inv hcf
      obtain ⟨sc, _, hcf⟩ := bind_inv hcf
      by_cases hsce : sc = c
      · rw [if_pos hsce] at hcf
        obtain ⟨b, hb, hcf⟩ := bind_inv hcf
        by_cases hbt : b = true
        · rw [if_pos hbt] at hcf
          exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
        · rw [if_neg hbt] at hcf
          by_cases hse : sc = hashtrinity.EMPTY
          · rw [if_pos hse] at hcf
            exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
          · rw [if_neg hse] at hcf
            obtain ⟨j1, hj1, hcf⟩ := bind_inv hcf
            have hh := Result.ok_injective hcf
            simp only [ControlFlow.cont.injEq] at hh
            subst hh
            have hj1v := (uadd_ok hj1).1
            rw [val_1_u32] at hj1v
            have hjlt : j.val < limit.val := (UScalar.lt_equiv _ _).mp hjl'
            exact ⟨by omega, by omega⟩
      · rw [if_neg hsce] at hcf
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse] at hcf
          exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
        · rw [if_neg hse] at hcf
          obtain ⟨j1, hj1, hcf⟩ := bind_inv hcf
          have hh := Result.ok_injective hcf
          simp only [ControlFlow.cont.injEq] at hh
          subst hh
          have hj1v := (uadd_ok hj1).1
          rw [val_1_u32] at hj1v
          have hjlt : j.val < limit.val := (UScalar.lt_equiv _ _).mp hjl'
          exact ⟨by omega, by omega⟩
    · rw [if_neg hjl'] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

/-- The cold capped-probe loop is total under the weak `matches` premise; its `cont` branch clears
the lowest set bit of `hits`, so `hits` strictly decreases. -/
theorem probe_capped_cold_loop_total_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (s : Slice U64) (s1 : Slice (core.mem.maybe_uninit.MaybeUninit I64)) (t : KS)
    (ss : hashtrinity.side.SideStore U64 I64 KS) (o : Option Ov) (i : U32) (i1 i2 : Usize)
    (pd : core.marker.PhantomData U64) (coord : U64) (home : Usize)
    (q : hashtrinity.overflow.ByKey U64) (limit : U32)
    (hbits : 0 < i.val) (hlen : s.length = 2 ^ i.val) (hhome8 : home.val + 8 ≤ s.length)
    (hlim8 : 8 ≤ limit.val) (hlim : limit.val ≤ limN i.val)
    (hits empties : U32) (hhits : hits.val < 256) (hemp : empties.val < 256) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_cold_loop core.cmp.EqU64 CI ks (ConfI ks)
      s s1 t ss o i i1 i2 pd coord home q limit hits empties = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold_loop
  refine loop_tot_inv _ (fun h => h.val < 256) (fun h => h.val) ?_ ?_ hits hhits
  · intro h1 hlt256
    unfold hashtrinity.overflow.Table.probe_capped_cold_loop.body
    simp only [lift, bind_tc_ok]
    have hblt := bits_lt_numBits s i hlen
    by_cases hnz : (h1 != 0#u32) = true
    · rw [if_pos hnz]
      obtain ⟨tz, htz⟩ := u32_tz_ex h1
      rw [htz]
      simp only [bind_tc_ok]
      have htz7 : tz.val ≤ 7 := u32_tz_le7 h1 hlt256 (ne_zero_of_bne h1 hnz) tz htz
      obtain ⟨slot, hslot, _⟩ := uadd_tot home (UScalar.cast .Usize tz)
        (by rw [cast_usize_u32]; exact add_lt_numBits home.val tz.val s.length i.val hlen hblt (by omega) (by omega))
      rw [hslot]
      simp only [bind_tc_ok]
      obtain ⟨b, hb⟩ := confirm_ex_weak ks hm q t slot
      rw [hb]
      simp only [bind_tc_ok]
      by_cases hbt : b = true
      · rw [if_pos hbt]; exact ⟨_, rfl⟩
      · rw [if_neg hbt]
        obtain ⟨i5, hi5, _⟩ := usub_tot h1 1#u32
          (by rw [val_1_u32]; exact Nat.one_le_iff_ne_zero.mpr (ne_zero_of_bne h1 hnz))
        rw [hi5]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
    · rw [if_neg hnz]
      by_cases hne : (empties != 0#u32) = true
      · rw [if_pos hne]
        obtain ⟨tz, htz⟩ := u32_tz_ex empties
        rw [htz]
        simp only [bind_tc_ok]
        have htz7 : tz.val ≤ 7 := u32_tz_le7 empties hemp (ne_zero_of_bne empties hne) tz htz
        obtain ⟨slot, hslot, _⟩ := uadd_tot home (UScalar.cast .Usize tz)
          (by rw [cast_usize_u32]; exact add_lt_numBits home.val tz.val s.length i.val hlen hblt (by omega) (by omega))
        rw [hslot]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
      · rw [if_neg hne]
        have hb64 : i.val ≤ 64 := by
          have := slice_len_lt s
          rw [hlen] at this
          have := (Nat.pow_lt_pow_iff_right (by norm_num : 1 < 2)).mp this
          rcases System.Platform.numBits_eq with h | h <;> omega
        obtain ⟨r, hr⟩ := probe_rest_with_total_weak ks hm
          { coords := s, vals := s1, keys := t, side := ss, overflow := o, bits := i,
            len := i1, tombs := i2, _k := pd } coord hashtrinity.LINEAR_G limit q
          hbits hb64 hlen (by rw [LINEAR_G_val]; exact hlim8) hlim
        rw [hr]
        simp only [bind_tc_ok]
        exact ⟨_, rfl⟩
  · intro h1 h1' hlt256 hcf
    unfold hashtrinity.overflow.Table.probe_capped_cold_loop.body at hcf
    simp only [lift, bind_tc_ok] at hcf
    by_cases hnz : (h1 != 0#u32) = true
    · rw [if_pos hnz] at hcf
      obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
      obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
      obtain ⟨b, hb, hcf⟩ := bind_inv hcf
      by_cases hbt : b = true
      · rw [if_pos hbt] at hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg hbt] at hcf
        obtain ⟨i5, hi5, hcf⟩ := bind_inv hcf
        have hh := Result.ok_injective hcf
        simp only [ControlFlow.cont.injEq] at hh
        subst hh
        have hi5v : i5.val = h1.val - 1 := by
          have := usub_ok hi5
          rw [val_1_u32] at this
          omega
        have hval : (h1 &&& i5).val = h1.val &&& i5.val := UScalar.val_and h1 i5
        refine ⟨?_, ?_⟩
        · rw [hval, hi5v]
          have hle : h1.val &&& (h1.val - 1) ≤ h1.val - 1 := Nat.and_le_right
          omega
        · rw [hval, hi5v]
          have hne := ne_zero_of_bne h1 hnz
          have hle : h1.val &&& (h1.val - 1) ≤ h1.val - 1 := Nat.and_le_right
          omega
    · rw [if_neg hnz] at hcf
      by_cases hne : (empties != 0#u32) = true
      · rw [if_pos hne] at hcf
        obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
        obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)
      · rw [if_neg hne] at hcf
        obtain ⟨r, hr, hcf⟩ := bind_inv hcf
        exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

/-- The cold capped probe is total under the weak `matches` premise. -/
theorem probe_capped_cold_total_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (hbits : 0 < s.bits.val) (hb64 : s.bits.val ≤ 64) (hlen : s.coords.length = 2 ^ s.bits.val)
    (hhome : home.val < s.coords.length) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_cold core.cmp.EqU64 CI ks (ConfI ks) s c home q
      = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold
  simp only [lift, bind_tc_ok]
  obtain ⟨lim, hlimok⟩ := probe_limit_total ks s hb64
  rw [hlimok]
  simp only [bind_tc_ok]
  have hlimv : lim.val = limN s.bits.val := probe_limit_val ks s lim hbits hlimok
  have hlim8 : 8 ≤ limN s.bits.val := by
    unfold limN
    have hd : 1 ≤ 64 / s.bits.val := Nat.div_pos hb64 hbits
    omega
  by_cases h8 : lim >= hashtrinity.LINEAR_G
  · rw [if_pos h8]
    have hblt := bits_lt_numBits s.coords s.bits hlen
    obtain ⟨i, hi, _⟩ := uadd_tot home (UScalar.cast .Usize hashtrinity.LINEAR_G)
      (by rw [cast_usize_u32, LINEAR_G_val]
          exact add_lt_numBits home.val 8 s.coords.length s.bits.val hlen hblt hhome (by norm_num))
    rw [hi]
    simp only [bind_tc_ok]
    by_cases hle : i <= Slice.len s.coords
    · rw [if_pos hle]
      have hivi : i.val = home.val + 8 := by
        rw [(uadd_ok hi).1, cast_usize_u32, LINEAR_G_val]
      have hle' : home.val + 8 ≤ s.coords.length := by
        rw [UScalar.le_equiv, Slice.len_val] at hle
        rw [hivi] at hle; exact hle
      obtain ⟨hits, empties, hpm, hh256, he256, _⟩ :=
        Apeiron.PrefixMasks.prefix_masks_spec s.coords home c hle'
      rw [hpm]
      simp only [bind_tc_ok]
      obtain ⟨r, hr⟩ := probe_capped_cold_loop_total_weak ks hm s.coords s.vals s.keys s.side s.overflow
        s.bits s.len s.tombs s._k c home q lim hbits hlen hle'
        (by rw [hlimv]; exact hlim8) (by rw [hlimv]) hits empties hh256 he256
      exact ⟨r, hr⟩
    · rw [if_neg hle]
      obtain ⟨r, hr⟩ := probe_rest_with_total_weak ks hm s c 0#u32 lim q hbits hb64 hlen
        (by simp) (by rw [hlimv])
      exact ⟨r, hr⟩
  · rw [if_neg h8]
    obtain ⟨r, hr⟩ := probe_rest_with_total_weak ks hm s c 0#u32 lim q hbits hb64 hlen
      (by simp) (by rw [hlimv])
    exact ⟨r, hr⟩

/-- **`probe_capped_keyed_ok`.**  `hashtrinity.overflow.Table.probe_capped_keyed` returns `ok`
under the weak `matches` premise: the `Table.home` and `Slice.get_unchecked` preamble are total
(`home_total`, `get_unchecked_ex`), the home-slot confirmation is total (`confirm_ex_weak`), and
every cold branch defers to `probe_capped_cold_total_weak`. -/
theorem probe_capped_keyed_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (_hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl KS) (c k : U64) (hbits : 0 < s.bits.val) (hlen : s.coords.length = 2 ^ s.bits.val) :
    ∃ r, hashtrinity.overflow.Table.probe_capped_keyed core.cmp.EqU64 CI ks s c k = ok r := by
  unfold hashtrinity.overflow.Table.probe_capped_keyed
  have hb64 : s.bits.val ≤ 64 := bits_le_64 ks s hlen
  obtain ⟨home, hhome, hhomelt⟩ := home_total ks s c hbits hb64 hlen
  rw [hhome]
  simp only [bind_tc_ok]
  obtain ⟨sc, hsc⟩ := get_unchecked_ex s.coords home hhomelt
  rw [hsc]
  simp only [bind_tc_ok]
  by_cases hc : sc = c
  · rw [if_pos hc]
    obtain ⟨b, hb⟩ := hm s.keys home c k
    rw [hb]
    simp only [bind_tc_ok]
    by_cases hbt : b = true
    · rw [if_pos hbt]; exact ⟨_, rfl⟩
    · rw [if_neg hbt]
      by_cases he : sc = hashtrinity.EMPTY
      · rw [if_pos he]; exact ⟨_, rfl⟩
      · rw [if_neg he]
        exact probe_capped_cold_total_weak ks hm s c home { coord := c, k := k }
          hbits hb64 hlen hhomelt
  · rw [if_neg hc]
    by_cases he : sc = hashtrinity.EMPTY
    · rw [if_pos he]; exact ⟨_, rfl⟩
    · rw [if_neg he]
      exact probe_capped_cold_total_weak ks hm s c home { coord := c, k := k }
        hbits hb64 hlen hhomelt

/-! ### Stage 7b: the weak `matches` slot contract

`probe_capped_keyed` totality alone does not supply the slot index bound the `place_capped`
`.Hit`/`.Vacant` branches need: `Capped.Hit i` must imply `t.coords[i] = c` (hence `i < vals.length`
via `vals_len`) and `Capped.Vacant i` must imply `t.coords[i]` is `EMPTY`.  Under the weak premise
the Pass clauses of `CappedOK` (`pcc_spec`) can fail, but these slot facts hold, because a `Hit`/cold
`Vacant` is only emitted at a slot whose coordinate the prefix masks record. -/

/-- The slot facts of a capped-probe verdict. -/
def SlotBelow {KS : Type} (s : Tbl KS) (c : U64) : hashtrinity.Capped → Prop
  | .Hit i => s.coords.val[i.val]? = some c
  | .Vacant i => ∃ x, s.coords.val[i.val]? = some x ∧ x.val = EMPTYN
  | .Blocked => True

/-- Checked `Slice::get_unchecked_mut` returns `ok` on an in-range index (its model is the checked
`index_mut`). -/
theorem slice_get_unchecked_mut_ex {α : Type} (s : Slice α) (i : Usize) (h : i.val < s.length) :
    ∃ x back, core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice α) s i
      = ok (x, back) := by
  unfold core.slice.Slice.get_unchecked_mut
  simp only [core.slice.index.Usize.index_mut, Slice.index_mut_usize]
  obtain ⟨y, hy⟩ := Apeiron.PrefixMasks.slice_index_ex s i h
  rw [hy]
  exact ⟨y, s.set i, by simp⟩

/-- The scalar tail's slot facts, under the weak `matches` premise. -/
theorem probe_rest_with_slot_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Tbl KS) (c : U64) (j limit : U32) (q : hashtrinity.overflow.ByKey U64)
    (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_rest_with core.cmp.EqU64 CI ks (ConfI ks) s c j limit q
      = ok r) :
    SlotBelow s c r := by
  unfold hashtrinity.overflow.Table.probe_rest_with hashtrinity.overflow.Table.probe_rest_with_loop at h
  refine loop_pc (fun _ => True) (fun r => SlotBelow s c r) _ ?_ j trivial r h
  intro j _ cf hcf
  unfold hashtrinity.overflow.Table.probe_rest_with_loop.body at hcf
  by_cases hjl' : j < limit
  · rw [if_pos hjl'] at hcf
    obtain ⟨i, _, hcf⟩ := bind_inv hcf
    obtain ⟨sc, hsc, hcf⟩ := bind_inv hcf
    have hsc' := get_unchecked_ok _ _ _ hsc
    by_cases hsce : sc = c
    · rw [if_pos hsce] at hcf
      obtain ⟨b, _, hcf⟩ := bind_inv hcf
      by_cases hbt : b = true
      · rw [if_pos hbt] at hcf
        have hh := (Result.ok_injective hcf).symm
        subst hh
        show s.coords.val[i.val]? = some c
        rw [hsc', hsce]
      · rw [if_neg hbt] at hcf
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse] at hcf
          have hh := (Result.ok_injective hcf).symm
          subst hh
          exact ⟨sc, hsc', (EMPTY_eq_iff sc).mp hse⟩
        · rw [if_neg hse] at hcf
          obtain ⟨j1, _, hcf⟩ := bind_inv hcf
          have hh := (Result.ok_injective hcf).symm
          subst hh
          trivial
    · rw [if_neg hsce] at hcf
      by_cases hse : sc = hashtrinity.EMPTY
      · rw [if_pos hse] at hcf
        have hh := (Result.ok_injective hcf).symm
        subst hh
        exact ⟨sc, hsc', (EMPTY_eq_iff sc).mp hse⟩
      · rw [if_neg hse] at hcf
        obtain ⟨j1, _, hcf⟩ := bind_inv hcf
        have hh := (Result.ok_injective hcf).symm
        subst hh
        trivial
  · rw [if_neg hjl'] at hcf
    have hh := (Result.ok_injective hcf).symm
    subst hh
    trivial

/-- The cold capped-probe loop's slot facts, under the weak `matches` premise: `hits`-set slots
hold `c` and `empties`-set slots hold `EMPTY`, and the `cont` step only clears a `hits` bit. -/
theorem probe_capped_cold_loop_slot_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64) (lim : U32)
    (hits empties : U32) (hhits : hits.val < 256) (hemp256 : empties.val < 256)
    (hhit : ∀ k : Nat, k < 8 → hits.val.testBit k = true → s.coords.val[home.val + k]? = some c)
    (hemp : ∀ k : Nat, k < 8 → empties.val.testBit k = true →
      ∃ x, s.coords.val[home.val + k]? = some x ∧ x.val = EMPTYN)
    (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_cold_loop core.cmp.EqU64 CI ks (ConfI ks)
      s.coords s.vals s.keys s.side s.overflow s.bits s.len s.tombs s._k c home q lim hits empties
      = ok r) :
    SlotBelow s c r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold_loop at h
  refine loop_pc
    (fun (h1 : U32) => h1.val < 256 ∧
      ∀ k : Nat, k < 8 → h1.val.testBit k = true → s.coords.val[home.val + k]? = some c)
    (fun r => SlotBelow s c r) _ ?_ hits ⟨hhits, hhit⟩ r h
  intro h1 hinv cf hcf
  unfold hashtrinity.overflow.Table.probe_capped_cold_loop.body at hcf
  simp only [lift, bind_tc_ok] at hcf
  by_cases hnz : (h1 != 0#u32) = true
  · rw [if_pos hnz] at hcf
    obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
    obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
    obtain ⟨b, _, hcf⟩ := bind_inv hcf
    have hslotv : slot.val = home.val + tz.val := by
      have hh := (uadd_ok hslot).1; rw [cast_usize_u32] at hh; exact hh
    have htzbit : h1.val.testBit tz.val = true :=
      (u32_tz_spec h1 (ne_zero_of_bne h1 hnz) tz htz).2.1
    have htz7 : tz.val ≤ 7 := u32_tz_le7 h1 hinv.1 (ne_zero_of_bne h1 hnz) tz htz
    by_cases hbt : b = true
    · rw [if_pos hbt] at hcf
      have hh := (Result.ok_injective hcf).symm
      subst hh
      show s.coords.val[slot.val]? = some c
      rw [hslotv]
      exact hinv.2 tz.val (by omega) htzbit
    · rw [if_neg hbt] at hcf
      obtain ⟨i5, hi5, hcf⟩ := bind_inv hcf
      have hh := (Result.ok_injective hcf).symm
      subst hh
      refine ⟨?_, ?_⟩
      · rw [UScalar.val_and]
        have hi5v : i5.val = h1.val - 1 := by
          have h5 := (usub_ok hi5); rw [val_1_u32] at h5; omega
        have hle : h1.val &&& i5.val ≤ i5.val := Nat.and_le_right
        omega
      · intro k hk8 hbit
        have hv : (h1 &&& i5).val.testBit k = (h1.val.testBit k && i5.val.testBit k) := by
          rw [UScalar.val_and]
          exact Nat.testBit_and h1.val i5.val k
        rw [hv] at hbit
        exact hinv.2 k hk8 (Bool.and_eq_true_iff.mp hbit).1
  · rw [if_neg hnz] at hcf
    by_cases hne : (empties != 0#u32) = true
    · rw [if_pos hne] at hcf
      obtain ⟨tz, htz, hcf⟩ := bind_inv hcf
      obtain ⟨slot, hslot, hcf⟩ := bind_inv hcf
      have hslotv : slot.val = home.val + tz.val := by
        have hh := (uadd_ok hslot).1; rw [cast_usize_u32] at hh; exact hh
      have htzbit : empties.val.testBit tz.val = true :=
        (u32_tz_spec empties (ne_zero_of_bne empties hne) tz htz).2.1
      have htz7 : tz.val ≤ 7 := u32_tz_le7 empties hemp256 (ne_zero_of_bne empties hne) tz htz
      have hh := (Result.ok_injective hcf).symm
      subst hh
      obtain ⟨x, hx, hxe⟩ := hemp tz.val (by omega) htzbit
      exact ⟨x, by rw [hslotv]; exact hx, hxe⟩
    · rw [if_neg hne] at hcf
      obtain ⟨r', hr', hcf⟩ := bind_inv hcf
      have hh := (Result.ok_injective hcf).symm
      subst hh
      exact probe_rest_with_slot_weak ks s c hashtrinity.LINEAR_G lim q r' hr'

/-- The cold capped probe's slot facts, under the weak `matches` premise. -/
theorem probe_capped_cold_slot_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Tbl KS) (c : U64) (home : Usize) (q : hashtrinity.overflow.ByKey U64)
    (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_cold core.cmp.EqU64 CI ks (ConfI ks) s c home q
      = ok r) :
    SlotBelow s c r := by
  unfold hashtrinity.overflow.Table.probe_capped_cold at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨lim, _, h⟩ := bind_inv h
  by_cases h8 : lim >= hashtrinity.LINEAR_G
  · rw [if_pos h8] at h
    obtain ⟨i, hi, h⟩ := bind_inv h
    have hiv : i.val = home.val + 8 := by rw [(uadd_ok hi).1, cast_usize_u32, LINEAR_G_val]
    by_cases hle : i <= Slice.len s.coords
    · rw [if_pos hle] at h
      have hle' : home.val + 8 ≤ s.coords.length := by
        rw [UScalar.le_equiv, Slice.len_val] at hle
        rw [hiv] at hle; exact hle
      obtain ⟨hits, empties, hpm, hh256, he256, hs3⟩ :=
        Apeiron.PrefixMasks.prefix_masks_spec s.coords home c hle'
      rw [hpm] at h
      simp only [bind_tc_ok] at h
      have hhit : ∀ k : Nat, k < 8 → hits.val.testBit k = true →
          s.coords.val[home.val + k]? = some c := by
        intro k hk hb
        obtain ⟨x, hx, hb1, _⟩ := hs3 k hk
        rw [hx, hb1.mp hb]
      have hemp : ∀ k : Nat, k < 8 → empties.val.testBit k = true →
          ∃ x, s.coords.val[home.val + k]? = some x ∧ x.val = EMPTYN := by
        intro k hk hb
        obtain ⟨x, hx, _, hb2⟩ := hs3 k hk
        exact ⟨x, hx, (EMPTY_eq_iff x).mp (hb2.mp hb)⟩
      exact probe_capped_cold_loop_slot_weak ks s c home q lim hits empties hh256 he256 hhit hemp r h
    · rw [if_neg hle] at h
      exact probe_rest_with_slot_weak ks s c 0#u32 lim q r h
  · rw [if_neg h8] at h
    exact probe_rest_with_slot_weak ks s c 0#u32 lim q r h

/-- `probe_capped_keyed`'s slot facts, under the weak `matches` premise. -/
theorem probe_capped_keyed_slot_weak {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (s : Tbl KS) (c k : U64) (r : hashtrinity.Capped)
    (h : hashtrinity.overflow.Table.probe_capped_keyed core.cmp.EqU64 CI ks s c k = ok r) :
    SlotBelow s c r := by
  unfold hashtrinity.overflow.Table.probe_capped_keyed at h
  obtain ⟨home, _, h⟩ := bind_inv h
  obtain ⟨sc, hsc, h⟩ := bind_inv h
  have hsc' := get_unchecked_ok _ _ _ hsc
  by_cases hc : sc = c
  · rw [if_pos hc] at h
    obtain ⟨b, _, h⟩ := bind_inv h
    by_cases hbt : b = true
    · rw [if_pos hbt] at h
      have hh := (Result.ok_injective h).symm
      subst hh
      show s.coords.val[home.val]? = some c
      rw [hsc', hc]
    · rw [if_neg hbt] at h
      by_cases he : sc = hashtrinity.EMPTY
      · rw [if_pos he] at h
        have hh := (Result.ok_injective h).symm
        subst hh
        exact ⟨sc, hsc', (EMPTY_eq_iff sc).mp he⟩
      · rw [if_neg he] at h
        exact probe_capped_cold_slot_weak ks s c home _ r h
  · rw [if_neg hc] at h
    by_cases he : sc = hashtrinity.EMPTY
    · rw [if_pos he] at h
      have hh := (Result.ok_injective h).symm
      subst hh
      exact ⟨sc, hsc', (EMPTY_eq_iff sc).mp he⟩
    · rw [if_neg he] at h
      exact probe_capped_cold_slot_weak ks s c home _ r h

/-! ### Stage 7c: `place_capped` is total under the weak `matches` premise

`place_capped` runs the keyed capped probe and acts on its verdict.  The slot contract
(`probe_capped_keyed_slot_weak`) supplies the index and the `EMPTY` fact the `.Hit`/`.Vacant`
branches read; the domain premises (`hsl`, `hinit`, `hset`, `hlenmax`, `hov`) close every
`get_unchecked_mut`, `assume_init_read`, `set`, `+ 1#usize` and overflow step. -/

/-- The `Regen<u64>` key-store size carrier: every `Usize` index fits. -/
def regenSz (_ : hashtrinity.algebra.Regen U64) : Nat := 2 ^ 64

/-- Every `Usize` value is at most `2 ^ 64`. -/
theorem usize_val_le_two_pow64 (n : Usize) : n.val ≤ 2 ^ 64 := by
  have h : n.val < 2 ^ UScalarTy.Usize.numBits := UScalar.hBounds n
  have h64 : UScalarTy.Usize.numBits ≤ 64 := by
    rw [show UScalarTy.Usize.numBits = System.Platform.numBits from rfl]
    rcases System.Platform.numBits_eq with h | h <;> omega
  have hp : 2 ^ UScalarTy.Usize.numBits ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) h64
  omega

/-- Every `Slice U64` is at most `2 ^ 64` entries long. -/
theorem slice_len_le_two_pow64 (s : Slice U64) : s.val.length ≤ 2 ^ 64 := by
  have h : s.val.length < 2 ^ System.Platform.numBits := by simpa [Slice.length] using slice_len_lt s
  have h64 : System.Platform.numBits ≤ 64 := by
    rcases System.Platform.numBits_eq with h | h <;> omega
  have hp : 2 ^ System.Platform.numBits ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) h64
  omega

/-- Every `Slice U64` is strictly shorter than `2 ^ 64`. -/
theorem slice_len_lt_two_pow64 (s : Slice U64) : s.val.length < 2 ^ 64 := by
  have h : s.val.length < 2 ^ System.Platform.numBits := by simpa [Slice.length] using slice_len_lt s
  have h64 : System.Platform.numBits ≤ 64 := by
    rcases System.Platform.numBits_eq with h | h <;> omega
  have hp : 2 ^ System.Platform.numBits ≤ 2 ^ 64 := Nat.pow_le_pow_right (by norm_num) h64
  omega

/-- The `Regen<u64>` size invariant holds at every table. -/
theorem regen_sz_inv (t : Tbl (hashtrinity.algebra.Regen U64)) :
    t.coords.val.length ≤ regenSz t.keys := by
  simpa [regenSz] using slice_len_le_two_pow64 t.coords

/-- **`place_capped_ok`.**  `Table::place_capped` returns `ok` under the weak `matches` premise and
the table domain the three verdicts read.  `hov` states the overflow domain of a present store: an
`OInv` shape `T`, the `+4` arena bound, the `Usize` length room and the entry-vector room at every
node of `T`. -/
theorem place_capped_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (t : Tbl KS) (c : U64) (v : I64)
    (hbits : 0 < t.bits.val) (hlen : t.coords.length = 2 ^ t.bits.val)
    (hsl : t.vals.length = t.coords.length)
    (hinit : ∀ (i : Nat), t.coords.val[i]? = some c → ∃ z, t.vals.val[i]? = some (some z))
    (hlenmax : t.len.val < Usize.max)
    (hov : ∀ ov, t.overflow = some ov →
      ∃ T, OInv ov T ∧ ov.nodes.val.length + 4 ≤ 2 ^ 32 - 1 ∧ ov.len.val < Usize.max ∧
      (∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd → nd.entries.val.length < Usize.max))
    (hsz : t.coords.val.length ≤ sz t.keys) :
    ∃ t', hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c v = ok t' := by
  obtain ⟨r, hr⟩ := probe_capped_keyed_ok ks hm hA t c c hbits hlen
  have hslot := probe_capped_keyed_slot_weak ks t c c r hr
  unfold hashtrinity.Table.place_capped
  rw [hr]
  simp only [bind_tc_ok]
  cases r with
  | Hit i =>
    -- `.Hit i`: overwrite the live slot
    have hcx : t.coords.val[i.val]? = some c := hslot
    have hltc : i.val < t.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < t.vals.length := by rw [hsl]; exact hltc
    obtain ⟨mu, back, hget⟩ := slice_get_unchecked_mut_ex t.vals i hltv
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok t.vals i mu back hget
    obtain ⟨z, hz⟩ := hinit i.val hcx
    have hmuz : mu = some z := Option.some.inj (hmu.symm.trans hz)
    have hread : core.mem.maybe_uninit.MaybeUninit.assume_init_read (some z) = ok z := rfl
    have hltv2 : i.val < (t.vals.set i (some z)).length := by rw [slice_set_length]; exact hltv
    obtain ⟨_, back1, hget2⟩ := slice_get_unchecked_mut_ex (t.vals.set i (some z)) i hltv2
    refine ⟨{t with vals := back1 (some v)}, ?_⟩
    simp only [hget, hmuz, hback, bind_tc_ok]
    change (do
      let old ← core.mem.maybe_uninit.MaybeUninit.assume_init_read (some z)
      core.mem.drop old
      let mu1 ← core.mem.maybe_uninit.MaybeUninit.new v
      let back1' ← core.slice.Slice.get_unchecked_mut
        (core.slice.index.SliceIndexUsizeSlice (core.mem.maybe_uninit.MaybeUninit I64))
        (t.vals.set i (some z)) i
      ok { t with vals := back1'.2 mu1 }) = ok { t with vals := back1 (some v) }
    simp only [hread, hget2, core.mem.drop, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
  | Vacant i =>
    -- `.Vacant i`: fill the `EMPTY` slot
    obtain ⟨_, hcx, _⟩ := hslot
    have hltc : i.val < t.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < t.vals.length := by rw [hsl]; exact hltc
    obtain ⟨_, backc, hgetc⟩ := slice_get_unchecked_mut_ex t.coords i hltc
    obtain ⟨_, backv, hgetv⟩ := slice_get_unchecked_mut_ex t.vals i hltv
    obtain ⟨keys', hset', -⟩ := hset t.keys i c (by omega)
    have hltmax : t.len.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
      rw [val_1_usize]
      have hl := hlenmax
      rw [Usize.max_def, Usize.numBits_def] at hl
      have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
      omega
    obtain ⟨i1, hi1, -⟩ := uadd_tot t.len 1#usize hltmax
    refine ⟨{t with coords := backc c, vals := backv (some v), keys := keys', len := i1}, ?_⟩
    simp only [hgetc, hgetv, hset', hi1, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
    rfl
  | Blocked =>
    -- `.Blocked`: the overflow store
    cases hto : t.overflow with
    | none =>
      let OV : Ov := { nodes := alloc.vec.Vec.new (hashtrinity.overflow.Node U64 I64),
                       free := alloc.vec.Vec.new U32,
                       root := hashtrinity.overflow.NIL,
                       len := 0#usize }
      have hnew : hashtrinity.overflow.Overflow.new I64 CI = ok OV := rfl
      obtain ⟨hF, -, hl0⟩ := ov_full_new ks t (fun _ => 0#i64) OV hnew
      have hlen0 : OV.len.val < Usize.max := by
        have h0 : OV.len.val = 0 := rfl
        rw [h0]
        have hge := nilN_le_usize_max
        unfold nilN at hge
        omega
      have hent0 : ∀ j ∈ idxs OTree.leaf, ∀ nd, OV.nodes.val[j]? = some nd →
          nd.entries.val.length < Usize.max := by
        intro j hj; simp [idxs] at hj
      have hcap0 : OV.nodes.val.length + 4 ≤ 2 ^ 32 - 1 := by rw [hl0]; norm_num
      obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok OV .leaf c c v hF.1 hlen0 hent0 hcap0
      refine ⟨{t with overflow := some ov1}, ?_⟩
      simp only [core.option.Option.is_none, Option.isNone, hnew,
        core.option.Option.as_mut, bind_tc_ok]
      rw [if_pos trivial]
      simp only [bind_tc_ok]
      change (do
        let (_, ov1') ← hashtrinity.overflow.Overflow.insert_new_at CI OV c c v
        ok { t with overflow := some ov1' }) = ok { t with overflow := some ov1 }
      rw [hins]
      simp only [bind_tc_ok]
      rfl
    | some ov0 =>
      obtain ⟨T, hI, hcap, hlenov, hentries⟩ := hov ov0 hto
      obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok ov0 T c c v hI hlenov hentries hcap
      refine ⟨{t with overflow := some ov1}, ?_⟩
      simp only [core.option.Option.is_none, Option.isNone,
        core.option.Option.as_mut, bind_tc_ok]
      rw [if_neg (by simp)]
      simp only [bind_tc_ok]
      change (do
        let (_, ov1') ← hashtrinity.overflow.Overflow.insert_new_at CI ov0 c c v
        ok { t with overflow := some ov1' }) = ok { t with overflow := some ov1 }
      rw [hins]
      simp only [bind_tc_ok]
      rfl

/-! ### Joint witness: the premises of `place_capped_ok` at one concrete instance

The `Regen<u64>` key store has total `set` (`ok self`) and `matches` (`ok true`); `with_bits 1` gives
a fresh table whose coordinates are all `EMPTY` and whose values are all uninitialised, with
`overflow = none`; the non-reserved coordinate `0` is held by no slot.  Every premise of
`place_capped_ok` therefore holds at this instance. -/

/-- `1#usize <<< 1#u32 = ok 2#usize`. -/
theorem shl_1_1 : (1#usize <<< 1#u32) = ok 2#usize := by
  show UScalar.shiftLeft_UScalar 1#usize 1#u32 = _
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft
  have h1 : (1#u32).val < UScalarTy.Usize.numBits := by
    show 1 < System.Platform.numBits
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide
  simp only [h1, if_true]
  congr 1
  apply UScalar.eq_of_val_eq
  have := usize_max_ge
  have hm := Usize.max_def
  rw [Usize.numBits_def] at hm
  have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
  show ((1#usize).bv.shiftLeft (1#u32).val).toNat = 2
  rw [BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
  have h1v : (1#usize).bv.toNat = 1 := rfl
  have h1w : (1#u32).val = 1 := rfl
  rw [h1v, h1w, Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega)]

/-- `with_bits 1` at the `Regen<u64>` key store returns a table with `len = 0`. -/
theorem with_bits1_RKS_ex :
    ∃ t : Tbl (hashtrinity.algebra.Regen U64),
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI RKS 1#u32 = ok t ∧ t.len.val = 0 := by
  unfold hashtrinity.Table.with_bits
  rw [shl_1_1]
  simp only [bind_tc_ok]
  obtain ⟨v, hv⟩ := from_elem_ok 2#usize
  rw [hv]; simp only [bind_tc_ok]
  unfold alloc.vec.Vec.into_boxed_slice
  simp only [bind_tc_ok]
  obtain ⟨s1, hs1⟩ := alloc_uninit_ok I64 2#usize
  rw [hs1]; simp only [bind_tc_ok]
  simp only [hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc,
    bind_tc_ok]
  unfold hashtrinity.side.SideStore.empty
  simp only [hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc,
    bind_tc_ok]
  exact ⟨_, rfl, rfl⟩

/-- **Joint witness for `place_capped_ok`.**  Every hypothesis of `place_capped_ok` holds at one
instance: the `Regen<u64>` key store, a fresh `with_bits 1` table (`overflow = none`), and the
non-reserved coordinate `0` held by no slot. -/
example : ∃ (t t' : Tbl (hashtrinity.algebra.Regen U64)),
    hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI RKS 1#u32 = ok t ∧
    hashtrinity.Table.place_capped core.cmp.EqU64 CI RKS t 0#u64 0#u64 0#i64 = ok t' := by
  obtain ⟨t, hwb, -⟩ := with_bits1_RKS_ex
  have hR := with_bits_rep RKS 1#u32 (by decide) t hwb
  have hovt := with_bits_overflow RKS 1#u32 t hwb
  have hsum : arenaLen t + mainCount t = 0 := (with_bits_wf RKS 1#u32 (by decide) t hwb).2
  have hmc : mainCount t = 0 := by omega
  have hlenmax : t.len.val < Usize.max := by
    rw [hR.len_count, hmc]
    have h := nilN_le_usize_max
    unfold nilN at h
    omega
  have hns0 : ¬ IsSent (0#u64) := by
    intro h; rcases h with h | h
    · have hv : (0#u64).val = 0 := rfl
      rw [hv] at h; unfold EMPTYN at h; omega
    · have hv : (0#u64).val = 0 := rfl
      rw [hv] at h; unfold TOMBN at h; omega
  obtain ⟨t', ht'⟩ := place_capped_ok RKS regenSz
    (fun keys i c k => ⟨true, RKS_matches keys i c k⟩)
    (fun _ => avx2_spec_holds)
    (fun keys i k _ => ⟨keys, rfl, rfl⟩)
    t 0#u64 0#i64
    hR.bits_pos hR.len_pow hR.vals_len
    (fun i hc => ⟨0#i64, (hR.main_val i 0#u64 hc hns0).2⟩)
    hlenmax
    (by intro ov hsome; rw [hovt] at hsome; exact absurd hsome (by simp))
    (regen_sz_inv t)
  exact ⟨t, t', hwb, ht'⟩

/-! ### `place_capped` preserves the initialised-value domain

`resize`'s main-slot loop calls `place_capped` on the growing table once per live key, and the call
needs `place_capped_ok`'s `hinit` premise on the table at that point.  `BigInit` is that premise in a
coordinate-independent form — every non-reserved main slot holds an initialised value — and
`place_capped_bigInit` carries it across one call.  A non-reserved slot can only ever be written with
an initialised value, so the property is preserved by all three verdicts. -/

/-- Every non-reserved main coordinate of `t` sits at an initialised value. -/
def BigInit {KS : Type} (t : Tbl KS) : Prop :=
  ∀ (i : Nat) (c : U64), t.coords.val[i]? = some c → ¬ IsSent c →
    ∃ z, t.vals.val[i]? = some (some z)

theorem place_capped_bigInit {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (t t' : Tbl KS) (c : U64) (v : I64)
    (h : hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c v = ok t')
    (hI : BigInit t) : BigInit t' := by
  unfold hashtrinity.Table.place_capped at h
  obtain ⟨r, hr, h⟩ := bind_inv h
  cases r with
  | Blocked =>
    obtain ⟨bo, -, h⟩ := bind_inv h
    unfold core.option.Option.as_mut at h
    obtain ⟨⟨o1, backo⟩, -, h⟩ := bind_inv h
    cases o1 with
    | none => exact absurd h (by intro hh; exact ok_not_fail hh.symm)
    | some ov =>
      change (do
        let (_, ov1) ← hashtrinity.overflow.Overflow.insert_new_at CI ov c c v
        ok ({ t with overflow := backo (some ov1) } : Tbl KS)) = ok t' at h
      obtain ⟨a, hins, h⟩ := bind_inv h
      rcases a with ⟨⟨nd, e⟩, ov1⟩
      have ht'' : t' = {t with overflow := backo (some ov1)} := (Result.ok_injective h).symm
      subst ht''
      exact hI
  | Hit i =>
    have hcx : t.coords.val[i.val]? = some c :=
      probe_capped_keyed_slot_weak ks t c c (hashtrinity.Capped.Hit i) hr
    obtain ⟨⟨mu, back⟩, hg, h⟩ := bind_inv h
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok t.vals i mu back hg
    obtain ⟨old, -, h⟩ := bind_inv h
    obtain ⟨-, -, h⟩ := bind_inv h
    obtain ⟨mu1, hnew, h⟩ := bind_inv h
    obtain ⟨⟨x2, back1⟩, hg1, h⟩ := bind_inv h
    obtain ⟨hx2, hback1⟩ := get_unchecked_mut_ok (back mu) i x2 back1 hg1
    have hmu1 : mu1 = some v := (Result.ok_injective hnew).symm
    have ht' : t' = {t with vals := back1 mu1} := (Result.ok_injective h).symm
    subst ht'
    have hlt : i.val < t.vals.length := by
      have h1 : i.val < (back mu).val.length := (List.getElem?_eq_some_iff.mp hx2).1
      have h2 : (back mu).val.length = t.vals.length := by
        rw [hback, Slice.set_val_eq, List.length_set]
      omega
    have hlt' : i.val < (t.vals.set i mu).length := by
      rw [slice_set_length]; exact hlt
    intro j c' hc' hns'
    have hc0 : t.coords.val[j]? = some c' := by simpa using hc'
    by_cases hji : j = i.val
    · subst hji
      refine ⟨v, ?_⟩
      have hb1 : ({t with vals := back1 mu1} : Tbl KS).vals.val =
          (Slice.set (t.vals.set i mu) i mu1).val := by
        show (back1 mu1).val = _
        rw [hback1, hback]
      rw [hb1, slice_set_getElem, if_pos ⟨rfl, hlt'⟩, hmu1]
    · refine ?_
      have hb1 : ({t with vals := back1 mu1} : Tbl KS).vals.val =
          (Slice.set (t.vals.set i mu) i mu1).val := by
        show (back1 mu1).val = _
        rw [hback1, hback]
      rw [hb1, slice_set_getElem, if_neg (by rintro ⟨h1, _⟩; exact hji h1),
        slice_set_getElem, if_neg (by rintro ⟨h1, _⟩; exact hji h1)]
      exact hI j c' hc0 hns'
  | Vacant i =>
    obtain ⟨x1, hcx, hx1e⟩ :=
      probe_capped_keyed_slot_weak ks t c c (hashtrinity.Capped.Vacant i) hr
    obtain ⟨⟨x0, backc⟩, hgc, h⟩ := bind_inv h
    obtain ⟨mu, hnew, h⟩ := bind_inv h
    obtain ⟨⟨y, backv⟩, hgv, h⟩ := bind_inv h
    obtain ⟨tk, hset, h⟩ := bind_inv h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    obtain ⟨hx0, hbackc⟩ := get_unchecked_mut_ok t.coords i x0 backc hgc
    obtain ⟨hy, hbackv⟩ := get_unchecked_mut_ok t.vals i y backv hgv
    have hmu : mu = some v := (Result.ok_injective hnew).symm
    have ht' : t' = {t with coords := backc c, vals := backv mu, keys := tk, len := i1} :=
      (Result.ok_injective h).symm
    subst ht'
    have hltv : i.val < t.vals.length := (List.getElem?_eq_some_iff.mp hy).1
    have hltc : i.val < t.coords.length := (List.getElem?_eq_some_iff.mp hx0).1
    intro j c' hc' hns'
    by_cases hji : j = i.val
    · subst hji
      refine ⟨v, ?_⟩
      have hb : ({t with coords := backc c, vals := backv mu, keys := tk, len := i1} : Tbl KS).vals.val =
          (Slice.set t.vals i mu).val := by
        show (backv mu).val = _
        rw [hbackv]
      rw [hb, slice_set_getElem, if_pos ⟨rfl, hltv⟩, hmu]
    · refine ?_
      have hcj : t.coords.val[j]? = some c' := by
        have h1 := hc'
        rw [hbackc, slice_set_getElem, if_neg (by rintro ⟨h1, _⟩; exact hji h1)] at h1
        exact h1
      obtain ⟨z, hz⟩ := hI j c' hcj hns'
      refine ⟨z, ?_⟩
      have hb : ({t with coords := backc c, vals := backv mu, keys := tk, len := i1} : Tbl KS).vals.val =
          (Slice.set t.vals i mu).val := by
        show (backv mu).val = _
        rw [hbackv]
      rw [hb, slice_set_getElem, if_neg (by rintro ⟨h1, _⟩; exact hji h1)]
      exact hz

/-! ### `insert_new_at` preserves the overflow domain with no absence premise

`insert_new_at_spec` establishes `OInv o' T'` but requires `habs : ¬ Present …`, which the resize
loop's weak `matches` context cannot supply.  The domain part of that proof does not use `habs`: it
is the `Emb`/`BST`/`ND`/`HOK`/`Bal`/free-list transport through `alloc` + `insert_node` (fresh
coordinate) or the entry push (existing coordinate).  This section restates it with `habs` dropped
and adds the per-insert entry-vector slack the consuming loop needs. -/

/-- **`ov_insert_new_at_domain`.**  `Overflow::insert_new_at` preserves `OInv` with no absence
premise: it increments `len` by one, grows the arena by at most one node, and raises the maximum
entry-vector length by at most one (`M → M + 1`).  `hentries` bounds the old entry-vector lengths. -/
theorem ov_insert_new_at_domain (o o' : Ov) (T : OTree) (coord k : U64) (v : I64) (nd : U32)
    (e : Usize) (M : Nat)
    (hI : OInv o T) (hcap : o.nodes.val.length < nilN)
    (hentries : ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.entries.val.length ≤ M)
    (h : hashtrinity.overflow.Overflow.insert_new_at CI o coord k v = ok ((nd, e), o')) :
    ∃ T', OInv o' T' ∧ o'.len.val = o.len.val + 1 ∧ o'.nodes.val.length ≤ o.nodes.val.length + 1 ∧
      (∀ j ∈ idxs T', ∀ nd', o'.nodes.val[j]? = some nd' → nd'.entries.val.length ≤ M + 1) := by
  obtain ⟨hE, hB, hND, hH, hBal, hfn, hfl⟩ := hI
  unfold hashtrinity.overflow.Overflow.insert_new_at at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  have hiv := uadd_ok hi
  have hu1 : (1#usize).val = 1 := rfl
  rw [hu1] at hiv
  obtain ⟨n, hn, h⟩ := bind_inv h
  have hspec := find_node_spec o.nodes coord T o.root hE hB n hn
  obtain ⟨⟨self1, node⟩, hif, h⟩ := bind_inv h
  by_cases hnil : n = hashtrinity.overflow.NIL
  · rw [if_pos hnil] at hif
    obtain ⟨⟨i1, o2⟩, halloc, hif⟩ := bind_inv hif
    obtain ⟨ni, hni, hnik, hnie, hnia, hnl, hnr, hnh, hnii, hnT, hnlt, hother, hroot2, hlen2, hfn2, hfl2, hlenA⟩ :=
      alloc_spec { o with len := i } o2 T i1 coord k v hE hfn hfl hcap halloc
    have hroot2' : o2.root = o.root := hroot2
    have hlen2' : o2.len = i := hlen2
    obtain ⟨self3, hins, hif⟩ := bind_inv hif
    have hneT : ∀ j ∈ idxs T, j ≠ i1.val := fun j hj e => hnT (by rw [← e]; exact hj)
    have hagree : ∀ j ∈ idxs T, o2.nodes.val[j]? = o.nodes.val[j]? :=
      fun j hj => hother j (hneT j hj)
    have hE2 : Emb o2.nodes.val o2.root T := by
      rw [hroot2']; exact emb_congr hE hagree
    have hB2 : BST o2.nodes.val T :=
      bst_congr' (fun j hj => by unfold keyAt; rw [hagree j hj]) hB
    have hH2 : HOK o2.nodes.val T := hok_congr hH hagree
    have hspecL : ∀ j ∈ idxs T, keyAt o.nodes.val j ≠ coord.val := by
      rcases hspec with ⟨_, hno⟩ | ⟨hm, _⟩
      · exact hno
      · rw [(NIL_eq_iff n).mp hnil] at hm
        obtain ⟨_, _, hne⟩ := emb_mem hE _ hm
        exact absurd rfl hne
    have hkey' : ∀ j ∈ idxs T, keyAt o2.nodes.val j ≠ ni.key.val := by
      intro j hj
      have h1 : keyAt o2.nodes.val j = keyAt o.nodes.val j := by
        unfold keyAt; rw [hagree j hj]
      rw [h1, hnik]; exact hspecL j hj
    obtain ⟨T', hE3, hB3, hND3, hH3, hBal3, hperm, hrel3, hf3, hl3⟩ :=
      insert_node_spec o2 self3 T i1 ni hE2 hB2 hND hH2 hBal hnT hnii hni hnl hnr hnh hkey' hins
    have hs := Result.ok_injective hif
    simp only [Prod.mk.injEq] at hs
    obtain ⟨hs1, hs2⟩ := hs
    subst hs1 hs2
    obtain ⟨n1, hn1, h⟩ := bind_inv h
    have hn1' := idx_ok hn1
    obtain ⟨e1, he1, h⟩ := bind_inv h
    have he1v := usub_ok he1
    rw [hu1] at he1v
    have hlenv : (alloc.vec.Vec.len n1.entries).val = n1.entries.val.length :=
      alloc.vec.Vec.len_val _
    have hfin := Result.ok_injective h
    simp only [Prod.mk.injEq] at hfin
    obtain ⟨⟨hnd1, he2⟩, ho'⟩ := hfin
    subst hnd1 he2 ho'
    obtain ⟨nd', hnd', hsd, _⟩ := hrel3.2 i1.val ni hni
    rw [hnd'] at hn1'
    have hnn : nd' = n1 := Option.some.inj hn1'
    subst hnn
    have hnent : nd'.entries.val = [(k, some v)] := by rw [← hsd.2.1]; exact hnie
    have hnkey : nd'.key = coord := by rw [← hsd.1]; exact hnik
    refine ⟨T', ⟨hE3, hB3, hND3, hH3, hBal3, ?_, ?_⟩, ?_, ?_, ?_⟩
    · rw [hf3]; exact hfn2
    · intro j hj
      rw [hf3] at hj
      obtain ⟨h1, h2, h3⟩ := hfl2 j hj
      refine ⟨lt_of_lt_of_eq h1 hrel3.1, fun hm => ?_⟩
      rcases List.mem_cons.mp (hperm.mem_iff.mp hm) with e | e
      · exact h3 e
      · exact h2 e
    · rw [hl3, hlen2']; exact hiv.1
    · rw [← hrel3.1]; exact hlenA
    · intro j hjT' nde hnde
      rcases List.mem_cons.mp (hperm.mem_iff.mp hjT') with e | hjT
      · subst e
        have hnde_eq : nde = nd' := Option.some.inj (hnde.symm.trans hnd')
        subst hnde_eq
        rw [hnent]
        simp
      · obtain ⟨ndo, hndo, _⟩ := emb_mem hE j hjT
        have hndo2 : o2.nodes.val[j]? = some ndo := by rw [hagree j hjT]; exact hndo
        obtain ⟨nds, hnds, hsd2, _⟩ := hrel3.2 j ndo hndo2
        have hnde_eq : nde = nds := Option.some.inj (hnde.symm.trans hnds)
        subst hnde_eq
        have hlen := hentries j hjT ndo hndo
        rw [← hsd2.2.1]
        omega
  · rw [if_neg hnil] at hif
    obtain ⟨⟨n1, back⟩, hn1, hif⟩ := bind_inv hif
    obtain ⟨hn1', hback⟩ := idxm_ok hn1
    obtain ⟨v1, hpush, hif⟩ := bind_inv hif
    have hv1 := vec_push_ok _ _ _ hpush
    have hs := Result.ok_injective hif
    simp only [Prod.mk.injEq] at hs
    obtain ⟨hs1, hs2⟩ := hs
    subst hs1 hs2
    have hnv : n.val ≠ nilN := fun hh => hnil ((NIL_eq_iff n).mpr hh)
    rcases hspec with ⟨hm, _⟩ | ⟨hm, hk⟩
    · exact absurd hm hnv
    have hn1o : o.nodes.val[n.val]? = some n1 := hn1'
    have hkey : n1.key = coord := by
      apply UScalar.eq_of_val_eq
      rw [← keyAt_get hn1o]; exact hk
    have hlt : n.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hn1o).1
    have hnodes : (back {n1 with entries := v1}).val =
        o.nodes.val.set n.val {n1 with entries := v1} := by rw [hback]
    obtain ⟨n1b, hn1b, h⟩ := bind_inv h
    have hn1c := idx_ok hn1b
    obtain ⟨e1, he1, h⟩ := bind_inv h
    have he1v := usub_ok he1
    rw [hu1] at he1v
    have hfin := Result.ok_injective h
    simp only [Prod.mk.injEq] at hfin
    obtain ⟨⟨hnd1, he2⟩, ho'⟩ := hfin
    subst hnd1 he2 ho'
    have hset : (o.nodes.val.set n.val {n1 with entries := v1}).length = o.nodes.val.length :=
      List.length_set
    have hn1new : n1b = {n1 with entries := v1} := by
      have : (back {n1 with entries := v1}).val[n.val]? = some n1b := hn1c
      rw [hnodes, List.getElem?_set_self hlt] at this
      exact (Option.some.inj this).symm
    subst hn1new
    have hent : v1.val = n1.entries.val ++ [(k, some v)] := hv1
    have hl2 : v1.val.length = n1.entries.val.length + 1 := by rw [hent]; simp
    have hne : ∀ j, j ≠ n.val → (o.nodes.val.set n.val {n1 with entries := v1})[j]? =
        o.nodes.val[j]? := fun j hj => List.getElem?_set_ne (Ne.symm hj)
    have hptr : ∀ j ∈ idxs T, ∀ nd0, o.nodes.val[j]? = some nd0 →
        ∃ nd', (o.nodes.val.set n.val {n1 with entries := v1})[j]? = some nd' ∧
          nd'.left = nd0.left ∧ nd'.right = nd0.right := by
      intro j hj nd0 hnd0
      by_cases hjj : j = n.val
      · subst hjj
        rw [hn1o] at hnd0
        have := Option.some.inj hnd0
        subst this
        exact ⟨{n1 with entries := v1}, List.getElem?_set_self hlt, rfl, rfl⟩
      · exact ⟨nd0, by rw [hne j hjj]; exact hnd0, rfl, rfl⟩
    have hkeys : ∀ j, keyAt (o.nodes.val.set n.val {n1 with entries := v1}) j =
        keyAt o.nodes.val j := by
      intro j
      by_cases hjj : j = n.val
      · subst hjj
        rw [keyAt_get (List.getElem?_set_self hlt), keyAt_get hn1o]
      · unfold keyAt; rw [hne j hjj]
    have hhag : ∀ j, HAg o.nodes.val (o.nodes.val.set n.val {n1 with entries := v1}) j := by
      intro j nd0 hnd0
      by_cases hjj : j = n.val
      · subst hjj
        rw [hn1o] at hnd0
        have := Option.some.inj hnd0
        subst this
        exact ⟨{n1 with entries := v1}, List.getElem?_set_self hlt, rfl⟩
      · exact ⟨nd0, by rw [hne j hjj]; exact hnd0, rfl⟩
    refine ⟨T, ⟨?_, ?_, hND, ?_, hBal, hfn, ?_⟩, ?_, ?_, ?_⟩
    · show Emb (back {n1 with entries := v1}).val o.root T
      rw [hnodes]; exact emb_ptr_congr hE hptr
    · show BST (back {n1 with entries := v1}).val T
      rw [hnodes]; exact bst_congr' (fun j _ => hkeys j) hB
    · show HOK (back {n1 with entries := v1}).val T
      rw [hnodes]; exact hok_hcongr hH (fun j _ => hhag j)
    · intro j hj
      refine ⟨?_, (hfl j hj).2⟩
      show j.val < (back {n1 with entries := v1}).val.length
      rw [hnodes, hset]; exact (hfl j hj).1
    · show i.val = o.len.val + 1
      exact hiv.1
    · show (back {n1 with entries := v1}).val.length ≤ _
      rw [hnodes, hset]; omega
    · intro j hjT nde hnde
      by_cases hjj : j = n.val
      · subst hjj
        have hnde_eq : nde = {n1 with entries := v1} := by
          have hself : (back {n1 with entries := v1}).val[n.val]? =
              some ({n1 with entries := v1} : ONode) := by
            rw [hnodes]; exact List.getElem?_set_self hlt
          exact Option.some.inj (hnde.symm.trans hself)
        subst hnde_eq
        have hlen := hentries n.val hm n1 hn1o
        show v1.val.length ≤ M + 1
        rw [hl2]
        omega
      · have hnde' : o.nodes.val[j]? = some nde := by
          rw [hnodes] at hnde
          rw [hne j hjj] at hnde
          exact hnde
        have hlen := hentries j hjT nde hnde'
        omega

/-! ### `place_capped` advances the overflow domain by at most one

`place_capped_ok`'s `hov` fixes the arena, `len` and entry-vector bounds at `place_capped`'s
capacity constants.  Those constants are consumed one unit per overflowing `insert_new_at`, so the
resize loops must carry the bounds as a counter.  `Hov t n` bounds a present overflow's arena,
`len` and per-node entry-vector length by `n`; `place_capped_hov` advances the counter by one. -/

/-- The overflow domain of `t` with a size counter `n`: a present overflow has an `OInv` shape whose
arena length, `len` and per-node entry-vector length are each at most `n`. -/
def Hov {KS : Type} (t : Tbl KS) (n : Nat) : Prop :=
  ∀ ov, t.overflow = some ov →
    ∃ T, OInv ov T ∧ ov.nodes.val.length ≤ n ∧ ov.len.val ≤ n ∧
      (∀ j ∈ idxs T, ∀ nd, ov.nodes.val[j]? = some nd → nd.entries.val.length ≤ n)

/-- **`place_capped_domain`.**  `Table::place_capped` returns a table whose overflow domain is
`Hov … n` transported to `Hov … (n+1)`: the `Hit` and `Vacant` verdicts leave the overflow untouched;
the `Blocked` verdict runs one `insert_new_at`, which `ov_insert_new_at_domain` grows by at most one
in each of the three measures.  `hK` (`n + 4 ≤ 2^32 - 1`) supplies the arena capacity the `Blocked`
verdict reads. -/
theorem place_capped_domain {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (t : Tbl KS) (c : U64) (v : I64) (n : Nat)
    (hbits : 0 < t.bits.val) (hlen : t.coords.length = 2 ^ t.bits.val)
    (hsl : t.vals.length = t.coords.length)
    (hinit : ∀ (i : Nat), t.coords.val[i]? = some c → ∃ z, t.vals.val[i]? = some (some z))
    (hlenmax : t.len.val < Usize.max)
    (hov : Hov t n) (hK : n + 4 ≤ 2 ^ 32 - 1)
    (hsz : t.coords.val.length ≤ sz t.keys) :
    ∃ t', hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c v = ok t' ∧
      Hov t' (n + 1) ∧ sz t'.keys = sz t.keys := by
  obtain ⟨r, hr⟩ := probe_capped_keyed_ok ks hm hA t c c hbits hlen
  have hslot := probe_capped_keyed_slot_weak ks t c c r hr
  unfold hashtrinity.Table.place_capped
  rw [hr]
  simp only [bind_tc_ok]
  cases r with
  | Hit i =>
    have hcx : t.coords.val[i.val]? = some c := hslot
    have hltc : i.val < t.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < t.vals.length := by rw [hsl]; exact hltc
    obtain ⟨mu, back, hget⟩ := slice_get_unchecked_mut_ex t.vals i hltv
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok t.vals i mu back hget
    obtain ⟨z, hz⟩ := hinit i.val hcx
    have hmuz : mu = some z := Option.some.inj (hmu.symm.trans hz)
    have hread : core.mem.maybe_uninit.MaybeUninit.assume_init_read (some z) = ok z := rfl
    have hltv2 : i.val < (t.vals.set i (some z)).length := by rw [slice_set_length]; exact hltv
    obtain ⟨_, back1, hget2⟩ := slice_get_unchecked_mut_ex (t.vals.set i (some z)) i hltv2
    refine ⟨{t with vals := back1 (some v)}, ?_, ?_, ?_⟩
    · simp only [hget, hmuz, hback, bind_tc_ok]
      change (do
        let old ← core.mem.maybe_uninit.MaybeUninit.assume_init_read (some z)
        core.mem.drop old
        let mu1 ← core.mem.maybe_uninit.MaybeUninit.new v
        let back1' ← core.slice.Slice.get_unchecked_mut
          (core.slice.index.SliceIndexUsizeSlice (core.mem.maybe_uninit.MaybeUninit I64))
          (t.vals.set i (some z)) i
        ok { t with vals := back1'.2 mu1 }) = ok { t with vals := back1 (some v) }
      simp only [hread, hget2, core.mem.drop, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
    · intro ov hsome
      have hsome' : t.overflow = some ov := hsome
      obtain ⟨T, hI0, hnode0, hlenov, hentries0⟩ := hov ov hsome'
      exact ⟨T, hI0, by omega, by omega, fun j hj nd hnd => by have := hentries0 j hj nd hnd; omega⟩
    · rfl
  | Vacant i =>
    obtain ⟨_, hcx, _⟩ := hslot
    have hltc : i.val < t.coords.val.length := (List.getElem?_eq_some_iff.mp hcx).1
    have hltv : i.val < t.vals.length := by rw [hsl]; exact hltc
    obtain ⟨_, backc, hgetc⟩ := slice_get_unchecked_mut_ex t.coords i hltc
    obtain ⟨_, backv, hgetv⟩ := slice_get_unchecked_mut_ex t.vals i hltv
    obtain ⟨keys', hset', hsze⟩ := hset t.keys i c (by omega)
    have hltmax : t.len.val + (1#usize).val < 2 ^ UScalarTy.Usize.numBits := by
      rw [val_1_usize]
      have hl := hlenmax
      rw [Usize.max_def, Usize.numBits_def] at hl
      have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
      omega
    obtain ⟨i1, hi1, -⟩ := uadd_tot t.len 1#usize hltmax
    refine ⟨{t with coords := backc c, vals := backv (some v), keys := keys', len := i1}, ?_, ?_, ?_⟩
    · simp only [hgetc, hgetv, hset', hi1, core.mem.maybe_uninit.MaybeUninit.new, bind_tc_ok]
      rfl
    · intro ov hsome
      have hsome' : t.overflow = some ov := hsome
      obtain ⟨T, hI0, hnode0, hlenov, hentries0⟩ := hov ov hsome'
      exact ⟨T, hI0, by omega, by omega, fun j hj nd hnd => by have := hentries0 j hj nd hnd; omega⟩
    · exact hsze
  | Blocked =>
    cases hto : t.overflow with
    | none =>
      let OV : Ov := { nodes := alloc.vec.Vec.new (hashtrinity.overflow.Node U64 I64),
                       free := alloc.vec.Vec.new U32,
                       root := hashtrinity.overflow.NIL,
                       len := 0#usize }
      have hnew : hashtrinity.overflow.Overflow.new I64 CI = ok OV := rfl
      obtain ⟨hF, -, hl0⟩ := ov_full_new ks t (fun _ => 0#i64) OV hnew
      have hlen0 : OV.len.val < Usize.max := by
        have h0 : OV.len.val = 0 := rfl
        rw [h0]
        have hge := nilN_le_usize_max
        unfold nilN at hge
        omega
      have hent0 : ∀ j ∈ idxs OTree.leaf, ∀ nd, OV.nodes.val[j]? = some nd →
          nd.entries.val.length < Usize.max := by
        intro j hj; simp [idxs] at hj
      have hcap0 : OV.nodes.val.length + 4 ≤ 2 ^ 32 - 1 := by rw [hl0]; norm_num
      obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok OV .leaf c c v hF.1 hlen0 hent0 hcap0
      refine ⟨{t with overflow := some ov1}, ?_, ?_, ?_⟩
      · simp only [core.option.Option.is_none, Option.isNone, hnew,
          core.option.Option.as_mut, bind_tc_ok]
        rw [if_pos trivial]
        simp only [bind_tc_ok]
        change (do
          let (_, ov1') ← hashtrinity.overflow.Overflow.insert_new_at CI OV c c v
          ok { t with overflow := some ov1' }) = ok { t with overflow := some ov1 }
        rw [hins]
        simp only [bind_tc_ok]
        rfl
      · intro ov hsome
        have heq : ov = ov1 := (Option.some.inj hsome).symm
        rw [heq]
        obtain ⟨T', hI', hlen', hnodes', hent'⟩ :=
          ov_insert_new_at_domain OV ov1 OTree.leaf c c v nd e 0 hF.1
            (by rw [hl0]; unfold nilN; norm_num) (by intro j hj; simp [idxs] at hj) hins
        have hl0len : OV.len.val = 0 := rfl
        rw [hl0] at hnodes'
        rw [hl0len] at hlen'
        exact ⟨T', hI', by omega, by omega, fun j hj nd hnd => by have := hent' j hj nd hnd; omega⟩
      · rfl
    | some ov0 =>
      obtain ⟨T, hI0, hnode0, hlenov, hentries0⟩ := hov ov0 hto
      have hUsize : n < Usize.max := by
        have h := nilN_le_usize_max; unfold nilN at h; omega
      have hcap : ov0.nodes.val.length + 4 ≤ 2 ^ 32 - 1 := by omega
      obtain ⟨nd, e, ov1, hins⟩ := ov_insert_new_at_ok ov0 T c c v hI0
        (by omega) (fun j hj nd hnd => by have := hentries0 j hj nd hnd; omega) hcap
      refine ⟨{t with overflow := some ov1}, ?_, ?_, ?_⟩
      · simp only [core.option.Option.is_none, Option.isNone,
          core.option.Option.as_mut, bind_tc_ok]
        rw [if_neg (by simp)]
        simp only [bind_tc_ok]
        change (do
          let (_, ov1') ← hashtrinity.overflow.Overflow.insert_new_at CI ov0 c c v
          ok { t with overflow := some ov1' }) = ok { t with overflow := some ov1 }
        rw [hins]
        simp only [bind_tc_ok]
        rfl
      · intro ov hsome
        have heq : ov = ov1 := (Option.some.inj hsome).symm
        rw [heq]
        obtain ⟨T', hI', hlen', hnodes', hent'⟩ :=
          ov_insert_new_at_domain ov0 ov1 T c c v nd e n hI0 (by unfold nilN; omega) hentries0 hins
        refine ⟨T', hI', by omega, by omega, fun j hj nd hnd => by have := hent' j hj nd hnd; omega⟩
      · rfl

/-- **`place_capped_hov`.**  One `place_capped` call carries `Hov`'s bounds across the call: the
result's counter is `n + 1`, so the three bounds advance by at most one. -/
theorem place_capped_hov {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (sz : KS → Nat)
    (hm : ∀ keys i c k, ∃ b, ks.«matches» keys i c k = ok b)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), i.val < sz keys →
      ∃ keys', ks.set keys i k = ok keys' ∧ sz keys' = sz keys)
    (t t' : Tbl KS) (c : U64) (v : I64) (n : Nat)
    (hbits : 0 < t.bits.val) (hlen : t.coords.length = 2 ^ t.bits.val)
    (hsl : t.vals.length = t.coords.length)
    (hinit : ∀ (i : Nat), t.coords.val[i]? = some c → ∃ z, t.vals.val[i]? = some (some z))
    (hlenmax : t.len.val < Usize.max)
    (hov : Hov t n) (hK : n + 4 ≤ 2 ^ 32 - 1)
    (hsz : t.coords.val.length ≤ sz t.keys)
    (h : hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c v = ok t') :
    Hov t' (n + 1) ∧ sz t'.keys = sz t.keys := by
  obtain ⟨t'', h'', hov'', hsze''⟩ :=
    place_capped_domain ks sz hm hA hset t c v n hbits hlen hsl hinit hlenmax hov hK hsz
  have heq : t' = t'' := Result.ok_injective (h.symm.trans h'')
  subst heq
  exact ⟨hov'', hsze''⟩

/-- **`place_capped_frame`.**  `place_capped` preserves the table's shape fields (`bits`,
`coords.length`, `vals.length`) and raises `len` by at most one (only the `Vacant` verdict writes it).
This is the shape half of the resize-loop invariant, alongside `place_capped_bigInit` (`hinit`) and
`place_capped_hov` (`hov`). -/
theorem place_capped_frame {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (t t' : Tbl KS) (c : U64) (v : I64)
    (h : hashtrinity.Table.place_capped core.cmp.EqU64 CI ks t c c v = ok t') :
    t'.bits = t.bits ∧ t'.coords.length = t.coords.length ∧
      t'.vals.length = t.vals.length ∧ t'.len.val ≤ t.len.val + 1 := by
  unfold hashtrinity.Table.place_capped at h
  obtain ⟨r, hr, h⟩ := bind_inv h
  cases r with
  | Blocked =>
    obtain ⟨bo, -, h⟩ := bind_inv h
    unfold core.option.Option.as_mut at h
    obtain ⟨⟨o1, backo⟩, -, h⟩ := bind_inv h
    cases o1 with
    | none => exact absurd h (by intro hh; exact ok_not_fail hh.symm)
    | some ov =>
      change (do
        let (_, ov1) ← hashtrinity.overflow.Overflow.insert_new_at CI ov c c v
        ok ({ t with overflow := backo (some ov1) } : Tbl KS)) = ok t' at h
      obtain ⟨a, hins, h⟩ := bind_inv h
      rcases a with ⟨⟨nd, e⟩, ov1⟩
      have ht'' : t' = {t with overflow := backo (some ov1)} := (Result.ok_injective h).symm
      subst ht''
      refine ⟨rfl, rfl, rfl, ?_⟩
      show t.len.val ≤ t.len.val + 1
      exact Nat.le_succ _
  | Hit i =>
    obtain ⟨⟨mu, back⟩, hg, h⟩ := bind_inv h
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok t.vals i mu back hg
    obtain ⟨old, -, h⟩ := bind_inv h
    obtain ⟨-, -, h⟩ := bind_inv h
    obtain ⟨mu1, hnew, h⟩ := bind_inv h
    obtain ⟨⟨x2, back1⟩, hg1, h⟩ := bind_inv h
    obtain ⟨hx2, hback1⟩ := get_unchecked_mut_ok (back mu) i x2 back1 hg1
    have ht' : t' = {t with vals := back1 mu1} := (Result.ok_injective h).symm
    subst ht'
    refine ⟨rfl, rfl, ?_, ?_⟩
    · show (back1 mu1).length = t.vals.length
      rw [hback1, hback]
      rw [slice_set_length, slice_set_length]
    · show t.len.val ≤ t.len.val + 1
      exact Nat.le_succ _
  | Vacant i =>
    obtain ⟨⟨x0, backc⟩, hgc, h⟩ := bind_inv h
    obtain ⟨mu, hnew, h⟩ := bind_inv h
    obtain ⟨⟨y, backv⟩, hgv, h⟩ := bind_inv h
    obtain ⟨tk, hset, h⟩ := bind_inv h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    obtain ⟨hx0, hbackc⟩ := get_unchecked_mut_ok t.coords i x0 backc hgc
    obtain ⟨hy, hbackv⟩ := get_unchecked_mut_ok t.vals i y backv hgv
    have ht' : t' = {t with coords := backc c, vals := backv mu, keys := tk, len := i1} :=
      (Result.ok_injective h).symm
    subst ht'
    refine ⟨rfl, ?_, ?_, ?_⟩
    · show (backc c).length = t.coords.length
      rw [hbackc, slice_set_length]
    · show (backv mu).length = t.vals.length
      rw [hbackv, slice_set_length]
    · show i1.val ≤ t.len.val + 1
      have hi1v : i1.val = t.len.val + 1 := by
        have := (uadd_ok hi1).1
        rw [val_1_usize] at this
        exact this
      omega

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
#audit Apeiron.MapExecuted.ov_insert_new_at_domain
#audit Apeiron.MapExecuted.confirm_ex_weak
#audit Apeiron.MapExecuted.probe_rest_with_total_weak
#audit Apeiron.MapExecuted.probe_capped_cold_loop_total_weak
#audit Apeiron.MapExecuted.probe_capped_cold_total_weak
#audit Apeiron.MapExecuted.probe_capped_keyed_ok
#audit Apeiron.MapExecuted.slice_get_unchecked_mut_ex
#audit Apeiron.MapExecuted.probe_rest_with_slot_weak
#audit Apeiron.MapExecuted.probe_capped_cold_loop_slot_weak
#audit Apeiron.MapExecuted.probe_capped_cold_slot_weak
#audit Apeiron.MapExecuted.probe_capped_keyed_slot_weak
#audit Apeiron.MapExecuted.place_capped_ok
#audit Apeiron.MapExecuted.place_capped_bigInit
#audit Apeiron.MapExecuted.place_capped_domain
#audit Apeiron.MapExecuted.place_capped_hov
#audit Apeiron.MapExecuted.place_capped_frame
#audit Apeiron.MapExecuted.shl_1_1
#audit Apeiron.MapExecuted.with_bits1_RKS_ex
#audit Apeiron.MapExecuted.usize_val_le_two_pow64
#audit Apeiron.MapExecuted.slice_len_le_two_pow64
#audit Apeiron.MapExecuted.slice_len_lt_two_pow64
#audit Apeiron.MapExecuted.regen_sz_inv
