-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the executed overflow tree's AVL write path.
import Map.ExecutedOverflow
import Map.ExecutedInsertSide
open Aeneas Aeneas.Std Result

/-!
# The executed overflow tree: AVL shape, `update` and the two rotations

`Map/ExecutedOverflow.lean` embeds the arena as an `OTree` shape (`Emb`) and orders it (`BST`).  The
write path also reads and writes the `height` field, so this file adds:

* `ht t` the height of a shape, `Bal t` the AVL balance (`|ht l - ht r| ≤ 1` at every node),
  `HOK ns t` the stored `height` field of every node of `t` equals `ht` of its subtree,
  `ND t` the node indices of `t` are pairwise distinct;
* `Rel S ns ns'`: the arena `ns'` differs from `ns` only in the `left`/`right`/`height` fields of
  the nodes in `S` (`sameD`: `key`, `entries` and `alive` are kept at every node);
* `update_spec`, `rotate_right_spec`, `rotate_left_spec`: the extracted `update`, `rotate_right`
  and `rotate_left` are partially correct against the tree rotations, in the arena.

All specs are partial correctness (`... = ok r → ...`): a failing call (index out of the arena,
`u32` overflow of a height) is excluded by the hypothesis.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-! ## Shape predicates -/

/-- The height of a shape. -/
def ht : OTree → Nat
  | .leaf => 0
  | .node l _ r => max (ht l) (ht r) + 1

/-- The AVL balance at every node. -/
def Bal : OTree → Prop
  | .leaf => True
  | .node l _ r => Bal l ∧ Bal r ∧ ht l ≤ ht r + 1 ∧ ht r ≤ ht l + 1

/-- The stored `height` field of every node equals the height of its subtree. -/
def HOK (ns : List ONode) : OTree → Prop
  | .leaf => True
  | .node l i r => HOK ns l ∧ HOK ns r ∧
      ∃ nd, ns[i]? = some nd ∧ nd.height.val = max (ht l) (ht r) + 1

/-- The node indices of a shape are pairwise distinct. -/
def ND : OTree → Prop
  | .leaf => True
  | .node l i r => ND l ∧ ND r ∧ i ∉ idxs l ∧ i ∉ idxs r ∧ ∀ j ∈ idxs l, j ∉ idxs r

theorem ht_le_len : ∀ t : OTree, ht t ≤ (idxs t).length
  | .leaf => by simp [ht, idxs]
  | .node l i r => by
    have h1 := ht_le_len l
    have h2 := ht_le_len r
    simp only [ht, idxs, List.length_append, List.length_cons]
    omega

/-! ## Transport along arena changes -/

theorem emb_congr {ns ns' : List ONode} : ∀ {x : U32} {t : OTree}, Emb ns x t →
    (∀ j ∈ idxs t, ns'[j]? = ns[j]?) → Emb ns' x t
  | _, .leaf, h, _ => h
  | x, .node l i r, h, hj => by
    obtain ⟨hx, hi, nd, hnd, hl, hr⟩ := h
    refine ⟨hx, hi, nd, ?_, emb_congr hl ?_, emb_congr hr ?_⟩
    · rw [hj i (by simp [idxs])]; exact hnd
    · intro j hj'; exact hj j (by simp [idxs, hj'])
    · intro j hj'; exact hj j (by simp [idxs, hj'])

theorem bst_congr {ns ns' : List ONode} (hk : ∀ j, keyAt ns' j = keyAt ns j) :
    ∀ {t : OTree}, BST ns t → BST ns' t
  | .leaf, _ => trivial
  | .node l i r, ⟨a, b, c, d⟩ =>
    ⟨bst_congr hk a, bst_congr hk b, fun j hj => by rw [hk, hk]; exact c j hj,
      fun j hj => by rw [hk, hk]; exact d j hj⟩

theorem hok_congr {ns ns' : List ONode} : ∀ {t : OTree}, HOK ns t →
    (∀ j ∈ idxs t, ns'[j]? = ns[j]?) → HOK ns' t
  | .leaf, _, _ => trivial
  | .node l i r, ⟨a, b, nd, hnd, hh⟩, hj => by
    refine ⟨hok_congr a ?_, hok_congr b ?_, nd, ?_, hh⟩
    · intro j hj'; exact hj j (by simp [idxs, hj'])
    · intro j hj'; exact hj j (by simp [idxs, hj'])
    · rw [hj i (by simp [idxs])]; exact hnd

/-- Node data kept by every write of the AVL path. -/
def sameD (a b : ONode) : Prop := a.key = b.key ∧ a.entries = b.entries ∧ a.alive = b.alive

theorem sameD_refl (a : ONode) : sameD a a := ⟨rfl, rfl, rfl⟩

theorem sameD_trans {a b c : ONode} (h1 : sameD a b) (h2 : sameD b c) : sameD a c :=
  ⟨h1.1.trans h2.1, h1.2.1.trans h2.2.1, h1.2.2.trans h2.2.2⟩

/-- `ns'` differs from `ns` only at the nodes `S`, and keeps `key`/`entries`/`alive` everywhere. -/
def Rel (S : List Nat) (ns ns' : List ONode) : Prop :=
  ns.length = ns'.length ∧ ∀ j nd, ns[j]? = some nd →
    ∃ nd', ns'[j]? = some nd' ∧ sameD nd nd' ∧ (j ∉ S → nd' = nd)

theorem Rel.refl (S : List Nat) (ns : List ONode) : Rel S ns ns :=
  ⟨rfl, fun _ nd h => ⟨nd, h, sameD_refl nd, fun _ => rfl⟩⟩

theorem Rel.trans {S S' : List Nat} {ns ns' ns'' : List ONode} (h1 : Rel S ns ns')
    (h2 : Rel S' ns' ns'') : Rel (S ++ S') ns ns'' := by
  refine ⟨h1.1.trans h2.1, fun j nd hj => ?_⟩
  obtain ⟨nd', hn', hs', ho'⟩ := h1.2 j nd hj
  obtain ⟨nd'', hn'', hs'', ho''⟩ := h2.2 j nd' hn'
  refine ⟨nd'', hn'', sameD_trans hs' hs'', fun hout => ?_⟩
  have hS : j ∉ S := fun hh => hout (List.mem_append.mpr (Or.inl hh))
  have hS' : j ∉ S' := fun hh => hout (List.mem_append.mpr (Or.inr hh))
  rw [ho'' hS', ho' hS]

theorem Rel.mono {S S' : List Nat} {ns ns' : List ONode} (hS : ∀ j ∈ S, j ∈ S')
    (h : Rel S ns ns') : Rel S' ns ns' := by
  refine ⟨h.1, fun j nd hj => ?_⟩
  obtain ⟨nd', hn', hs', ho'⟩ := h.2 j nd hj
  exact ⟨nd', hn', hs', fun hout => ho' (fun hh => hout (hS j hh))⟩

theorem rel_out {S : List Nat} {ns ns' : List ONode} (h : Rel S ns ns') {j : Nat} (hj : j ∉ S) :
    ns'[j]? = ns[j]? := by
  rcases Nat.lt_or_ge j ns.length with hlt | hge
  · obtain ⟨nd, hnd⟩ : ∃ nd, ns[j]? = some nd := ⟨ns[j], List.getElem?_eq_getElem hlt⟩
    obtain ⟨nd', h1, _, h3⟩ := h.2 j nd hnd
    rw [h1, hnd, h3 hj]
  · rw [List.getElem?_eq_none hge, List.getElem?_eq_none (by rw [← h.1]; exact hge)]

theorem keyAt_rel {S : List Nat} {ns ns' : List ONode} (h : Rel S ns ns') (j : Nat) :
    keyAt ns' j = keyAt ns j := by
  unfold keyAt
  rcases Nat.lt_or_ge j ns.length with hlt | hge
  · obtain ⟨nd, hnd⟩ : ∃ nd, ns[j]? = some nd := ⟨ns[j], List.getElem?_eq_getElem hlt⟩
    obtain ⟨nd', h1, hs, _⟩ := h.2 j nd hnd
    rw [h1, hnd]
    show nd'.key.val = nd.key.val
    rw [hs.1]
  · rw [List.getElem?_eq_none hge, List.getElem?_eq_none (by rw [← h.1]; exact hge)]

theorem keyAt_get {ns : List ONode} {j : Nat} {nd : ONode} (h : ns[j]? = some nd) :
    keyAt ns j = nd.key.val := by
  simp [keyAt, h]

theorem emb_rel {S : List Nat} {ns ns' : List ONode} (h : Rel S ns ns') {x : U32} {t : OTree}
    (hE : Emb ns x t) (hd : ∀ j ∈ idxs t, j ∉ S) : Emb ns' x t :=
  emb_congr hE (fun j hj => rel_out h (hd j hj))

theorem hok_rel {S : List Nat} {ns ns' : List ONode} (h : Rel S ns ns') {t : OTree}
    (hH : HOK ns t) (hd : ∀ j ∈ idxs t, j ∉ S) : HOK ns' t :=
  hok_congr hH (fun j hj => rel_out h (hd j hj))

theorem bst_rel {S : List Nat} {ns ns' : List ONode} (h : Rel S ns ns') {t : OTree}
    (hB : BST ns t) : BST ns' t :=
  bst_congr (keyAt_rel h) hB

theorem Rel_set {ns : List ONode} {i : Nat} {nd nd' : ONode} (h : ns[i]? = some nd)
    (hs : sameD nd nd') : Rel [i] ns (ns.set i nd') := by
  have hi : i < ns.length := (List.getElem?_eq_some_iff.mp h).1
  refine ⟨(List.length_set).symm, fun j n0 hj => ?_⟩
  by_cases hij : i = j
  · subst hij
    have : nd = n0 := by rw [h] at hj; exact Option.some.inj hj
    subst this
    exact ⟨nd', List.getElem?_set_self hi, hs, fun hh => absurd (by simp) hh⟩
  · exact ⟨n0, by rw [List.getElem?_set_ne hij]; exact hj, sameD_refl n0, fun _ => rfl⟩

/-! ## Two-node writes -/

theorem arena2_get (ns : List ONode) (p q : Nat) (hpq : p ≠ q) (P Q : ONode)
    (hp : p < ns.length) (hq : q < ns.length) :
    ((ns.set p P).set q Q)[p]? = some P ∧ ((ns.set p P).set q Q)[q]? = some Q ∧
      ∀ j, j ≠ p → j ≠ q → ((ns.set p P).set q Q)[j]? = ns[j]? := by
  refine ⟨?_, ?_, ?_⟩
  · rw [List.getElem?_set_ne (Ne.symm hpq)]
    exact List.getElem?_set_self hp
  · exact List.getElem?_set_self (by rw [List.length_set]; exact hq)
  · intro j hjp hjq
    rw [List.getElem?_set_ne (Ne.symm hjq), List.getElem?_set_ne (Ne.symm hjp)]

theorem arena2_rel (ns : List ONode) (p q : Nat) (hpq : p ≠ q) (P Q nx ny : ONode)
    (hp : ns[p]? = some nx) (hq : ns[q]? = some ny) (hP : sameD nx P) (hQ : sameD ny Q) :
    Rel ([p] ++ [q]) ns ((ns.set p P).set q Q) := by
  refine Rel.trans (Rel_set hp hP) (Rel_set ?_ hQ)
  rw [List.getElem?_set_ne hpq]; exact hq

theorem arena_collapse (ns : List ONode) (p q : Nat) (hpq : p ≠ q) (a1 a2 b1 b2 : ONode) :
    ((((ns.set p a1).set q b1).set q b2).set p a2) = (ns.set p a2).set q b2 := by
  rw [List.set_set, List.set_comm b2 a2 (Ne.symm hpq), List.set_set]

/-! ## Tree facts -/

theorem bst_rotR {ns : List ONode} {a b c : OTree} {x y : Nat}
    (h : BST ns (.node (.node a x b) y c)) : BST ns (.node a x (.node b y c)) := by
  obtain ⟨⟨hBa, hBb, hax, hbx⟩, hBc, hly, hyc⟩ := h
  refine ⟨hBa, ⟨hBb, hBc, fun j hj => hly j (by simp [idxs, hj]), hyc⟩, hax, ?_⟩
  intro j hj
  simp only [idxs, List.mem_append, List.mem_cons] at hj
  rcases hj with hj | hj | hj
  · exact hbx j hj
  · rw [hj]; exact hly x (by simp [idxs])
  · have h1 := hly x (by simp [idxs])
    have h2 := hyc j hj
    omega

theorem bst_rotL {ns : List ONode} {a b c : OTree} {x y : Nat}
    (h : BST ns (.node a x (.node b y c))) : BST ns (.node (.node a x b) y c) := by
  obtain ⟨hBa, ⟨hBb, hBc, hby, hyc⟩, hax, hxr⟩ := h
  refine ⟨⟨hBa, hBb, hax, fun j hj => hxr j (by simp [idxs, hj])⟩, hBc, ?_, hyc⟩
  intro j hj
  simp only [idxs, List.mem_append, List.mem_cons] at hj
  rcases hj with hj | hj | hj
  · have h1 := hax j hj
    have h2 := hxr y (by simp [idxs])
    omega
  · rw [hj]; exact hxr y (by simp [idxs])
  · exact hby j hj

theorem nd_rotR {a b c : OTree} {x y : Nat} (h : ND (.node (.node a x b) y c)) :
    ND (.node a x (.node b y c)) := by
  obtain ⟨⟨hNa, hNb, hxa, hxb, hab⟩, hNc, hyl, hyc, hlc⟩ := h
  simp only [idxs, List.mem_append, List.mem_cons, not_or] at hyl
  refine ⟨hNa, ⟨hNb, hNc, hyl.2.2, hyc, fun j hj hjc => hlc j (by simp [idxs, hj]) hjc⟩, hxa, ?_, ?_⟩
  · simp only [idxs, List.mem_append, List.mem_cons, not_or]
    exact ⟨hxb, fun e => hyl.2.1 e.symm, fun hc' => hlc x (by simp [idxs]) hc'⟩
  · intro j hj
    simp only [idxs, List.mem_append, List.mem_cons, not_or]
    exact ⟨hab j hj, fun e => hyl.1 (e ▸ hj), fun hc' => hlc j (by simp [idxs, hj]) hc'⟩

theorem nd_rotL {a b c : OTree} {x y : Nat} (h : ND (.node a x (.node b y c))) :
    ND (.node (.node a x b) y c) := by
  obtain ⟨hNa, ⟨hNb, hNc, hyb, hyc, hbc⟩, hxa, hxr, har⟩ := h
  simp only [idxs, List.mem_append, List.mem_cons, not_or] at hxr
  refine ⟨⟨hNa, hNb, hxa, hxr.1, fun j hj hjb => har j hj (by simp [idxs, hjb])⟩, hNc, ?_, hyc, ?_⟩
  · simp only [idxs, List.mem_append, List.mem_cons, not_or]
    exact ⟨fun hya => har y hya (by simp [idxs]), fun e => hxr.2.1 e.symm, hyb⟩
  · intro j hj
    simp only [idxs, List.mem_append, List.mem_cons] at hj
    rcases hj with hj | hj | hj
    · exact fun hjc => har j hj (by simp [idxs, hjc])
    · rw [hj]; exact hxr.2.2
    · exact hbc j hj

theorem idxs_rotR {a b c : OTree} {x y : Nat} :
    idxs (.node a x (.node b y c)) = idxs (.node (.node a x b) y c) := by
  simp [idxs]

theorem idxs_rotL {a b c : OTree} {x y : Nat} :
    idxs (.node (.node a x b) y c) = idxs (.node a x (.node b y c)) := by
  simp [idxs]

/-! ## Extracted-code access -/

theorem u32_one : (1#u32).val = 1 := by rfl
theorem u32_zero : (0#u32).val = 0 := by rfl

theorem vec_set_val {α : Type} (v : alloc.vec.Vec α) (i : Usize) (x : α) :
    (v.set i x).val = v.val.set i.val x := by
  simp only [alloc.vec.Vec.set_val_eq]

theorem idx_ok {ns : alloc.vec.Vec ONode} {i : U32} {n : ONode}
    (h : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) ns
      (UScalar.cast .Usize i) = ok n) : ns.val[i.val]? = some n := by
  have := vec_index_ok _ _ _ h
  rwa [cast_usize_u32] at this

theorem idxm_ok {ns : alloc.vec.Vec ONode} {i : U32} {n : ONode} {back : ONode → alloc.vec.Vec ONode}
    (h : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice ONode) ns
      (UScalar.cast .Usize i) = ok (n, back)) :
    ns.val[i.val]? = some n ∧ ∀ x, (back x).val = ns.val.set i.val x := by
  obtain ⟨h1, h2⟩ := vec_index_mut_ok _ _ _ _ h
  refine ⟨by rwa [cast_usize_u32] at h1, fun x => ?_⟩
  rw [h2, vec_set_val, cast_usize_u32]

/-- `h` reads the stored height, `0` on `NIL`. -/
theorem h_spec (o : Ov) (L : List ONode) (hL : o.nodes.val = L) (x : U32) (t : OTree)
    (hE : Emb L x t) (hH : HOK L t) (r : U32)
    (h : hashtrinity.overflow.Overflow.h CI o x = ok r) : r.val = ht t := by
  unfold hashtrinity.overflow.Overflow.h at h
  by_cases hx : x = hashtrinity.overflow.NIL
  · rw [if_pos hx] at h
    have hr : (0#u32) = r := Result.ok_injective h
    subst hr
    cases t with
    | leaf => simp [ht]
    | node l i rr =>
      obtain ⟨hxi, hi, _⟩ := hE
      exact absurd (by rw [← hxi]; exact (NIL_eq_iff x).mp hx) hi
  · rw [if_neg hx] at h
    simp only [lift, bind_tc_ok] at h
    obtain ⟨n, hn, h⟩ := bind_inv h
    have hn' := idx_ok hn
    rw [hL] at hn'
    have hr : n.height = r := Result.ok_injective h
    subst hr
    cases t with
    | leaf => exact absurd ((NIL_eq_iff x).mpr hE) hx
    | node l i rr =>
      obtain ⟨hxi, hi, nd, hnd, _, _⟩ := hE
      obtain ⟨_, _, nd', hnd', hh⟩ := hH
      rw [hxi, hnd] at hn'
      have hnn : nd = n := Option.some.inj hn'
      subst hnn
      rw [hnd] at hnd'
      have hnn' : nd = nd' := Option.some.inj hnd'
      subst hnn'
      simpa [ht] using hh

/-- `update`: the height of node `ii` becomes `1 + max (ht l) (ht r)`; nothing else changes. -/
theorem update_spec (o o' : Ov) (L : List ONode) (hL : o.nodes.val = L) (i : U32)
    (l r : OTree) (ii : Nat) (hE : Emb L i (.node l ii r)) (hl : HOK L l) (hr : HOK L r)
    (h : hashtrinity.overflow.Overflow.update CI o i = ok o') :
    ∃ nd hv, L[ii]? = some nd ∧ hv.val = max (ht l) (ht r) + 1 ∧
      o'.nodes.val = L.set ii {nd with height := hv} ∧ o'.free = o.free ∧ o'.root = o.root ∧
      o'.len = o.len := by
  obtain ⟨hxi, hi, nd, hnd, hEl, hEr⟩ := hE
  unfold hashtrinity.overflow.Overflow.update at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨n, hn, h⟩ := bind_inv h
  have hn' := idx_ok hn
  rw [hL, hxi, hnd] at hn'
  have hnn : nd = n := Option.some.inj hn'
  subst hnn
  obtain ⟨n1, hn1, h⟩ := bind_inv h
  have hn1' := idx_ok hn1
  rw [hL, hxi, hnd] at hn1'
  have hnn1 : nd = n1 := Option.some.inj hn1'
  subst hnn1
  obtain ⟨hl', hhl, h⟩ := bind_inv h
  have hvl := h_spec o L hL _ l hEl hl hl' hhl
  obtain ⟨hr', hhr, h⟩ := bind_inv h
  have hvr := h_spec o L hL _ r hEr hr hr' hhr
  obtain ⟨tl, htl, h⟩ := bind_inv h
  obtain ⟨hh, hhe, h⟩ := bind_inv h
  have hmax : tl.val = max (ht l) (ht r) := by
    by_cases hge : hl' >= hr'
    · rw [if_pos hge] at htl
      have h1 := (UScalar.le_equiv hr' hl').mp hge
      have h2 := Result.ok_injective htl
      rw [← h2]; omega
    · rw [if_neg hge] at htl
      have h1 := (not_congr (UScalar.le_equiv hr' hl')).mp hge
      have h2 := Result.ok_injective htl
      rw [← h2]; omega
  have hadd := uadd_ok hhe
  obtain ⟨⟨n2, back⟩, hn2, h⟩ := bind_inv h
  obtain ⟨hn2', hback⟩ := idxm_ok hn2
  rw [hL, hxi, hnd] at hn2'
  have hnn2 : nd = n2 := Option.some.inj hn2'
  subst hnn2
  have hfin := Result.ok_injective h
  subst hfin
  refine ⟨nd, hh, hnd, ?_, ?_, rfl, rfl, rfl⟩
  · rw [hadd.1, hmax, u32_one]
  · dsimp only
    rw [hback, hL, hxi]

/-- `rotate_right y`: `((a x b) y c) ↦ (a x (b y c))` in the arena, with the two heights refreshed. -/
theorem rotate_right_spec (o o' : Ov) (L : List ONode) (hL : o.nodes.val = L) (y x' : U32)
    (a b c : OTree) (xi yi : Nat)
    (hE : Emb L y (.node (.node a xi b) yi c))
    (hND : ND (.node (.node a xi b) yi c))
    (hB : BST L (.node (.node a xi b) yi c))
    (ha : HOK L a) (hb : HOK L b) (hc : HOK L c)
    (h : hashtrinity.overflow.Overflow.rotate_right CI o y = ok (x', o')) :
    ∃ A, o'.nodes.val = A ∧ x'.val = xi ∧ Emb A x' (.node a xi (.node b yi c)) ∧
      Rel (idxs (.node (.node a xi b) yi c)) L A ∧
      BST A (.node a xi (.node b yi c)) ∧ HOK A (.node a xi (.node b yi c)) ∧
      o'.free = o.free ∧ o'.root = o.root ∧ o'.len = o.len := by
  obtain ⟨hyv, hyi, ny, hny, hEL, hEc⟩ := hE
  obtain ⟨hxv, hxi, nx, hnx, hEa, hEb⟩ := hEL
  obtain ⟨⟨hNDa, hNDb, hxa, hxb, hab⟩, hNDc, hyl, hyc, hlc⟩ := hND
  have hyl' : yi ∉ idxs a ∧ ¬ yi = xi ∧ yi ∉ idxs b := by
    simpa only [idxs, List.mem_append, List.mem_cons, not_or] using hyl
  have hxy : xi ≠ yi := fun e => hyl'.2.1 e.symm
  have hxlt : xi < L.length := (List.getElem?_eq_some_iff.mp hnx).1
  have hylt : yi < L.length := (List.getElem?_eq_some_iff.mp hny).1
  have hav : ∀ j ∈ idxs a, j ≠ xi ∧ j ≠ yi :=
    fun j hj => ⟨fun e => hxa (e ▸ hj), fun e => hyl'.1 (e ▸ hj)⟩
  have hbv : ∀ j ∈ idxs b, j ≠ xi ∧ j ≠ yi :=
    fun j hj => ⟨fun e => hxb (e ▸ hj), fun e => hyl'.2.2 (e ▸ hj)⟩
  have hcv : ∀ j ∈ idxs c, j ≠ xi ∧ j ≠ yi :=
    fun j hj => ⟨fun e => hlc xi (by simp [idxs]) (e ▸ hj), fun e => hyc (e ▸ hj)⟩
  unfold hashtrinity.overflow.Overflow.rotate_right at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨n, hn, h⟩ := bind_inv h
  have hn' := idx_ok hn
  rw [hL, hyv, hny] at hn'
  have hnn : ny = n := Option.some.inj hn'
  subst hnn
  obtain ⟨n1, hn1, h⟩ := bind_inv h
  have hn1' := idx_ok hn1
  rw [hL, hxv, hnx] at hn1'
  have hnn1 : nx = n1 := Option.some.inj hn1'
  subst hnn1
  obtain ⟨⟨n2, back⟩, hn2, h⟩ := bind_inv h
  obtain ⟨hn2', hback⟩ := idxm_ok hn2
  rw [hL, hxv, hnx] at hn2'
  have hnn2 : nx = n2 := Option.some.inj hn2'
  subst hnn2
  obtain ⟨⟨n3, back1⟩, hn3, h⟩ := bind_inv h
  obtain ⟨hn3', hback1⟩ := idxm_ok hn3
  have hv : (back {nx with right := y}).val = L.set xi {nx with right := y} := by
    rw [hback, hL, hxv]
  rw [hv, hyv, List.getElem?_set_ne hxy, hny] at hn3'
  have hnn3 : ny = n3 := Option.some.inj hn3'
  subst hnn3
  have hv1 : (back1 {ny with left := nx.right}).val =
      (L.set xi {nx with right := y}).set yi {ny with left := nx.right} := by
    rw [hback1, hv, hyv]
  obtain ⟨L1, hL1def⟩ : ∃ L1, L1 =
      (L.set xi {nx with right := y}).set yi {ny with left := nx.right} := ⟨_, rfl⟩
  obtain ⟨hL1x, hL1y, hL1o⟩ := arena2_get L xi yi hxy {nx with right := y}
    {ny with left := nx.right} hxlt hylt
  rw [← hL1def] at hL1x hL1y hL1o
  -- first `update`, at `y`
  obtain ⟨o1, hu1, h⟩ := bind_inv h
  have hE1 : Emb L1 y (.node b yi c) := by
    refine ⟨hyv, hyi, {ny with left := nx.right}, hL1y, ?_, ?_⟩
    · exact emb_congr hEb (fun j hj => hL1o j (hbv j hj).1 (hbv j hj).2)
    · exact emb_congr hEc (fun j hj => hL1o j (hcv j hj).1 (hcv j hj).2)
  have hHb1 : HOK L1 b := hok_congr hb (fun j hj => hL1o j (hbv j hj).1 (hbv j hj).2)
  have hHc1 : HOK L1 c := hok_congr hc (fun j hj => hL1o j (hcv j hj).1 (hcv j hj).2)
  obtain ⟨nd1, hh1, hnd1, hhv1, hnodes1, hf1, hr1, hl1⟩ :=
    update_spec _ o1 L1 (by dsimp only; rw [hv1, hL1def]) y b c yi hE1 hHb1 hHc1 hu1
  have hnd1eq : ({ny with left := nx.right} : ONode) = nd1 :=
    Option.some.inj (hL1y.symm.trans hnd1)
  subst hnd1eq
  obtain ⟨L2, hL2def⟩ : ∃ L2, L2 = L1.set yi
      {({ny with left := nx.right} : ONode) with height := hh1} := ⟨_, rfl⟩
  have hylt1 : yi < L1.length := (List.getElem?_eq_some_iff.mp hL1y).1
  have hL2x : L2[xi]? = some {nx with right := y} := by
    rw [hL2def, List.getElem?_set_ne (Ne.symm hxy)]; exact hL1x
  have hL2y : L2[yi]? = some {({ny with left := nx.right} : ONode) with height := hh1} := by
    rw [hL2def]; exact List.getElem?_set_self hylt1
  have hL2o : ∀ j, j ≠ xi → j ≠ yi → L2[j]? = L[j]? := by
    intro j h1 h2
    rw [hL2def, List.getElem?_set_ne (Ne.symm h2)]
    exact hL1o j h1 h2
  -- second `update`, at `x`
  obtain ⟨o2, hu2, h⟩ := bind_inv h
  have hE2 : Emb L2 ny.left (.node a xi (.node b yi c)) := by
    refine ⟨hxv, hxi, {nx with right := y}, hL2x, ?_, ?_⟩
    · exact emb_congr hEa (fun j hj => hL2o j (hav j hj).1 (hav j hj).2)
    · refine ⟨hyv, hyi, {({ny with left := nx.right} : ONode) with height := hh1}, hL2y, ?_, ?_⟩
      · exact emb_congr hEb (fun j hj => hL2o j (hbv j hj).1 (hbv j hj).2)
      · exact emb_congr hEc (fun j hj => hL2o j (hcv j hj).1 (hcv j hj).2)
  have hHa2 : HOK L2 a := hok_congr ha (fun j hj => hL2o j (hav j hj).1 (hav j hj).2)
  have hHbc2 : HOK L2 (.node b yi c) :=
    ⟨hok_congr hb (fun j hj => hL2o j (hbv j hj).1 (hbv j hj).2),
     hok_congr hc (fun j hj => hL2o j (hcv j hj).1 (hcv j hj).2),
     _, hL2y, hhv1⟩
  obtain ⟨nd2, hh2, hnd2, hhv2, hnodes2, hf2, hr2, hl2⟩ :=
    update_spec _ o2 L2 (by rw [hnodes1, hL2def]) ny.left a (.node b yi c) xi hE2 hHa2 hHbc2 hu2
  have hnd2eq : ({nx with right := y} : ONode) = nd2 := Option.some.inj (hL2x.symm.trans hnd2)
  subst hnd2eq
  have hx : ny.left = x' := congrArg Prod.fst (Result.ok_injective h)
  have ho : o2 = o' := congrArg Prod.snd (Result.ok_injective h)
  subst hx
  subst ho
  have hA : o2.nodes.val = (L.set xi {({nx with right := y} : ONode) with height := hh2}).set yi
      {({ny with left := nx.right} : ONode) with height := hh1} := by
    rw [hnodes2, hL2def, hL1def]
    exact arena_collapse L xi yi hxy _ _ _ _
  obtain ⟨hAx, hAy, hAo⟩ := arena2_get L xi yi hxy
    {({nx with right := y} : ONode) with height := hh2}
    {({ny with left := nx.right} : ONode) with height := hh1} hxlt hylt
  have hRel : Rel (idxs (.node (.node a xi b) yi c)) L
      ((L.set xi {({nx with right := y} : ONode) with height := hh2}).set yi
        {({ny with left := nx.right} : ONode) with height := hh1}) := by
    refine Rel.mono ?_ (arena2_rel L xi yi hxy _ _ nx ny hnx hny ⟨rfl, rfl, rfl⟩ ⟨rfl, rfl, rfl⟩)
    intro j hj
    simp only [List.mem_append, List.mem_singleton] at hj
    rcases hj with hj | hj <;> rw [hj] <;> simp [idxs]
  refine ⟨_, hA, hxv, ?_, hRel, bst_rel hRel (bst_rotR hB), ?_, ?_, ?_, ?_⟩
  · refine ⟨hxv, hxi, {({nx with right := y} : ONode) with height := hh2}, hAx, ?_, ?_⟩
    · exact emb_congr hEa (fun j hj => hAo j (hav j hj).1 (hav j hj).2)
    · refine ⟨hyv, hyi, {({ny with left := nx.right} : ONode) with height := hh1}, hAy, ?_, ?_⟩
      · exact emb_congr hEb (fun j hj => hAo j (hbv j hj).1 (hbv j hj).2)
      · exact emb_congr hEc (fun j hj => hAo j (hcv j hj).1 (hcv j hj).2)
  · refine ⟨hok_congr ha (fun j hj => hAo j (hav j hj).1 (hav j hj).2),
      ⟨hok_congr hb (fun j hj => hAo j (hbv j hj).1 (hbv j hj).2),
       hok_congr hc (fun j hj => hAo j (hcv j hj).1 (hcv j hj).2), _, hAy, hhv1⟩, _, hAx, hhv2⟩
  · exact hf2.trans hf1
  · exact hr2.trans hr1
  · exact hl2.trans hl1

/-- `rotate_left x`: `(a x (b y c)) ↦ ((a x b) y c)` in the arena, with the two heights refreshed. -/
theorem rotate_left_spec (o o' : Ov) (L : List ONode) (hL : o.nodes.val = L) (x y' : U32)
    (a b c : OTree) (xi yi : Nat)
    (hE : Emb L x (.node a xi (.node b yi c)))
    (hND : ND (.node a xi (.node b yi c)))
    (hB : BST L (.node a xi (.node b yi c)))
    (ha : HOK L a) (hb : HOK L b) (hc : HOK L c)
    (h : hashtrinity.overflow.Overflow.rotate_left CI o x = ok (y', o')) :
    ∃ A, o'.nodes.val = A ∧ y'.val = yi ∧ Emb A y' (.node (.node a xi b) yi c) ∧
      Rel (idxs (.node a xi (.node b yi c))) L A ∧
      BST A (.node (.node a xi b) yi c) ∧ HOK A (.node (.node a xi b) yi c) ∧
      o'.free = o.free ∧ o'.root = o.root ∧ o'.len = o.len := by
  obtain ⟨hxv, hxi, nx, hnx, hEa, hEy⟩ := hE
  obtain ⟨hyv, hyi, ny, hny, hEb, hEc⟩ := hEy
  obtain ⟨hNDa, ⟨hNDb, hNDc, hyb, hyc, hbc⟩, hxa, hxr, har⟩ := hND
  have hxr' : xi ∉ idxs b ∧ ¬ xi = yi ∧ xi ∉ idxs c := by
    simpa only [idxs, List.mem_append, List.mem_cons, not_or] using hxr
  have hxy : xi ≠ yi := hxr'.2.1
  have hyx : yi ≠ xi := fun e => hxy e.symm
  have hxlt : xi < L.length := (List.getElem?_eq_some_iff.mp hnx).1
  have hylt : yi < L.length := (List.getElem?_eq_some_iff.mp hny).1
  have hav : ∀ j ∈ idxs a, j ≠ xi ∧ j ≠ yi :=
    fun j hj => ⟨fun e => hxa (e ▸ hj), fun e => har j hj (by rw [e]; simp [idxs])⟩
  have hbv : ∀ j ∈ idxs b, j ≠ xi ∧ j ≠ yi :=
    fun j hj => ⟨fun e => hxr'.1 (e ▸ hj), fun e => hyb (e ▸ hj)⟩
  have hcv : ∀ j ∈ idxs c, j ≠ xi ∧ j ≠ yi :=
    fun j hj => ⟨fun e => hxr'.2.2 (e ▸ hj), fun e => hyc (e ▸ hj)⟩
  unfold hashtrinity.overflow.Overflow.rotate_left at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨n, hn, h⟩ := bind_inv h
  have hn' := idx_ok hn
  rw [hL, hxv, hnx] at hn'
  have hnn : nx = n := Option.some.inj hn'
  subst hnn
  obtain ⟨n1, hn1, h⟩ := bind_inv h
  have hn1' := idx_ok hn1
  rw [hL, hyv, hny] at hn1'
  have hnn1 : ny = n1 := Option.some.inj hn1'
  subst hnn1
  obtain ⟨⟨n2, back⟩, hn2, h⟩ := bind_inv h
  obtain ⟨hn2', hback⟩ := idxm_ok hn2
  rw [hL, hyv, hny] at hn2'
  have hnn2 : ny = n2 := Option.some.inj hn2'
  subst hnn2
  obtain ⟨⟨n3, back1⟩, hn3, h⟩ := bind_inv h
  obtain ⟨hn3', hback1⟩ := idxm_ok hn3
  have hv : (back {ny with left := x}).val = L.set yi {ny with left := x} := by
    rw [hback, hL, hyv]
  rw [hv, hxv, List.getElem?_set_ne hyx, hnx] at hn3'
  have hnn3 : nx = n3 := Option.some.inj hn3'
  subst hnn3
  have hv1 : (back1 {nx with right := ny.left}).val =
      (L.set yi {ny with left := x}).set xi {nx with right := ny.left} := by
    rw [hback1, hv, hxv]
  obtain ⟨L1, hL1def⟩ : ∃ L1, L1 =
      (L.set yi {ny with left := x}).set xi {nx with right := ny.left} := ⟨_, rfl⟩
  obtain ⟨hL1y, hL1x, hL1o⟩ := arena2_get L yi xi hyx {ny with left := x}
    {nx with right := ny.left} hylt hxlt
  rw [← hL1def] at hL1x hL1y hL1o
  -- first `update`, at `x`
  obtain ⟨o1, hu1, h⟩ := bind_inv h
  have hE1 : Emb L1 x (.node a xi b) := by
    refine ⟨hxv, hxi, {nx with right := ny.left}, hL1x, ?_, ?_⟩
    · exact emb_congr hEa (fun j hj => hL1o j (hav j hj).2 (hav j hj).1)
    · exact emb_congr hEb (fun j hj => hL1o j (hbv j hj).2 (hbv j hj).1)
  have hHa1 : HOK L1 a := hok_congr ha (fun j hj => hL1o j (hav j hj).2 (hav j hj).1)
  have hHb1 : HOK L1 b := hok_congr hb (fun j hj => hL1o j (hbv j hj).2 (hbv j hj).1)
  obtain ⟨nd1, hh1, hnd1, hhv1, hnodes1, hf1, hr1, hl1⟩ :=
    update_spec _ o1 L1 (by dsimp only; rw [hv1, hL1def]) x a b xi hE1 hHa1 hHb1 hu1
  have hnd1eq : ({nx with right := ny.left} : ONode) = nd1 :=
    Option.some.inj (hL1x.symm.trans hnd1)
  subst hnd1eq
  obtain ⟨L2, hL2def⟩ : ∃ L2, L2 = L1.set xi
      {({nx with right := ny.left} : ONode) with height := hh1} := ⟨_, rfl⟩
  have hxlt1 : xi < L1.length := (List.getElem?_eq_some_iff.mp hL1x).1
  have hL2y : L2[yi]? = some {ny with left := x} := by
    rw [hL2def, List.getElem?_set_ne hxy]; exact hL1y
  have hL2x : L2[xi]? = some {({nx with right := ny.left} : ONode) with height := hh1} := by
    rw [hL2def]; exact List.getElem?_set_self hxlt1
  have hL2o : ∀ j, j ≠ xi → j ≠ yi → L2[j]? = L[j]? := by
    intro j h1 h2
    rw [hL2def, List.getElem?_set_ne (Ne.symm h1)]
    exact hL1o j h2 h1
  -- second `update`, at `y`
  obtain ⟨o2, hu2, h⟩ := bind_inv h
  have hE2 : Emb L2 nx.right (.node (.node a xi b) yi c) := by
    refine ⟨hyv, hyi, {ny with left := x}, hL2y, ?_, ?_⟩
    · refine ⟨hxv, hxi, {({nx with right := ny.left} : ONode) with height := hh1}, hL2x, ?_, ?_⟩
      · exact emb_congr hEa (fun j hj => hL2o j (hav j hj).1 (hav j hj).2)
      · exact emb_congr hEb (fun j hj => hL2o j (hbv j hj).1 (hbv j hj).2)
    · exact emb_congr hEc (fun j hj => hL2o j (hcv j hj).1 (hcv j hj).2)
  have hHc2 : HOK L2 c := hok_congr hc (fun j hj => hL2o j (hcv j hj).1 (hcv j hj).2)
  have hHab2 : HOK L2 (.node a xi b) :=
    ⟨hok_congr ha (fun j hj => hL2o j (hav j hj).1 (hav j hj).2),
     hok_congr hb (fun j hj => hL2o j (hbv j hj).1 (hbv j hj).2),
     _, hL2x, hhv1⟩
  obtain ⟨nd2, hh2, hnd2, hhv2, hnodes2, hf2, hr2, hl2⟩ :=
    update_spec _ o2 L2 (by rw [hnodes1, hL2def]) nx.right (.node a xi b) c yi hE2 hHab2 hHc2 hu2
  have hnd2eq : ({ny with left := x} : ONode) = nd2 := Option.some.inj (hL2y.symm.trans hnd2)
  subst hnd2eq
  have hx : nx.right = y' := congrArg Prod.fst (Result.ok_injective h)
  have ho : o2 = o' := congrArg Prod.snd (Result.ok_injective h)
  subst hx
  subst ho
  have hA : o2.nodes.val = (L.set yi {({ny with left := x} : ONode) with height := hh2}).set xi
      {({nx with right := ny.left} : ONode) with height := hh1} := by
    rw [hnodes2, hL2def, hL1def]
    exact arena_collapse L yi xi hyx _ _ _ _
  obtain ⟨hAy, hAx, hAo⟩ := arena2_get L yi xi hyx
    {({ny with left := x} : ONode) with height := hh2}
    {({nx with right := ny.left} : ONode) with height := hh1} hylt hxlt
  have hRel : Rel (idxs (.node a xi (.node b yi c))) L
      ((L.set yi {({ny with left := x} : ONode) with height := hh2}).set xi
        {({nx with right := ny.left} : ONode) with height := hh1}) := by
    refine Rel.mono ?_ (arena2_rel L yi xi hyx _ _ ny nx hny hnx ⟨rfl, rfl, rfl⟩ ⟨rfl, rfl, rfl⟩)
    intro j hj
    simp only [List.mem_append, List.mem_singleton] at hj
    rcases hj with hj | hj <;> rw [hj] <;> simp [idxs]
  refine ⟨_, hA, hyv, ?_, hRel, bst_rel hRel (bst_rotL hB), ?_, ?_, ?_, ?_⟩
  · refine ⟨hyv, hyi, {({ny with left := x} : ONode) with height := hh2}, hAy, ?_, ?_⟩
    · refine ⟨hxv, hxi, {({nx with right := ny.left} : ONode) with height := hh1}, hAx, ?_, ?_⟩
      · exact emb_congr hEa (fun j hj => hAo j (hav j hj).2 (hav j hj).1)
      · exact emb_congr hEb (fun j hj => hAo j (hbv j hj).2 (hbv j hj).1)
    · exact emb_congr hEc (fun j hj => hAo j (hcv j hj).2 (hcv j hj).1)
  · refine ⟨⟨hok_congr ha (fun j hj => hAo j (hav j hj).2 (hav j hj).1),
      hok_congr hb (fun j hj => hAo j (hbv j hj).2 (hbv j hj).1), _, hAx, hhv1⟩,
      hok_congr hc (fun j hj => hAo j (hcv j hj).2 (hcv j hj).1), _, hAy, hhv2⟩
  · exact hf2.trans hf1
  · exact hr2.trans hr1
  · exact hl2.trans hl1

end Apeiron.MapExecuted
