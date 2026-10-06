-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe`: the overflow tree's in-order walk and `into_entries`.
import Map.ExecutedInsertSlot
open Aeneas Aeneas.Std Result

/-!
# The overflow tree's in-order walk and `into_entries`

`resize` re-places every overflow entry through `Overflow::into_entries`
(`src/hashtrinity/overflow.rs`), which walks the tree with `in_order` (an explicit stack: push the
left spine, pop, emit the node when `alive`, continue at its right child) and takes each node's
entries.

* `in_order_spec`: on an embedded tree whose nodes are all `alive`, `in_order` from an empty output
  returns exactly the tree's arena indices in order (`idxs T`).
* `into_entries_spec`: on an `OvFull` store, `into_entries` returns `(key, key, m key)` for each node
  of the tree, in order.

The walk's invariant is `out ++ idxs(cur's subtree) ++ pendG G = idxs T`, with `G` the stack read top
first, each entry a node and its right subtree.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- The work still pending on the stack, top first: each node, then its right subtree. -/
def pendG : List (Nat × OTree) → List Nat
  | [] => []
  | (x, R) :: G => x :: idxs R ++ pendG G

/-- The stack `st` (bottom first) is the frame list `G` (top first), each frame a node of `ns` whose
right child embeds the frame's subtree. -/
def StackOK (ns : List ONode) (st : List U32) (G : List (Nat × OTree)) : Prop :=
  st.map (fun x => x.val) = (G.map Prod.fst).reverse ∧
    ∀ p ∈ G, ∃ nd, ns[p.1]? = some nd ∧ Emb ns nd.right p.2

theorem next_spec {α : Type} (it it' : alloc.vec.into_iter.IntoIter α) (o : Option α)
    (h : alloc.vec.into_iter.IteratorIntoIter.next it = ok (o, it')) :
    (it.val = [] ∧ o = none) ∨ (∃ hd, it.val = hd :: it'.val ∧ o = some hd) := by
  unfold alloc.vec.into_iter.IteratorIntoIter.next at h
  split at h
  · rename_i heq
    left
    have := Result.ok_injective h
    simp only [Prod.mk.injEq] at this
    exact ⟨heq, this.1.symm⟩
  · rename_i hd tl heq
    right
    have := Result.ok_injective h
    simp only [Prod.mk.injEq] at this
    obtain ⟨h1, h2⟩ := this
    subst h1 h2
    exact ⟨hd, by simpa using heq, rfl⟩

theorem stack_pop_frame {L : List Nat} {a : Nat} {G : List (Nat × OTree)}
    (h : (G.map Prod.fst).reverse = L ++ [a]) :
    ∃ R G', G = (a, R) :: G' ∧ (G'.map Prod.fst).reverse = L := by
  cases G with
  | nil => simp at h
  | cons p G' =>
    simp only [List.map_cons, List.reverse_cons] at h
    obtain ⟨h1, h2⟩ := List.append_inj' h rfl
    have : p.1 = a := by simpa using h2
    exact ⟨p.2, G', by rw [← this], h1⟩

theorem emb_ne_nil {ns : List ONode} {x : U32} {t : OTree} (h : Emb ns x t)
    (hx : x ≠ hashtrinity.overflow.NIL) :
    ∃ l i r, t = .node l i r ∧ x.val = i ∧ ∃ nd, ns[i]? = some nd ∧ Emb ns nd.left l ∧
      Emb ns nd.right r := by
  cases t with
  | leaf => exact absurd ((NIL_eq_iff x).mpr h) hx
  | node l i r =>
    obtain ⟨h1, _, nd, hnd, hl, hr⟩ := h
    exact ⟨l, i, r, rfl, h1, nd, hnd, hl, hr⟩

theorem emb_nil {ns : List ONode} {t : OTree} (h : Emb ns hashtrinity.overflow.NIL t) :
    t = .leaf := by
  cases t with
  | leaf => rfl
  | node l i r =>
    obtain ⟨h1, h2, _⟩ := h
    rw [NIL_val] at h1
    exact absurd h1.symm h2

/-- The left-spine push: from a subtree at `cur`, the stack gains the spine and `cur` ends at
`NIL`. -/
theorem loop1_eq (v : alloc.vec.Vec ONode) (st : alloc.vec.Vec U32) (c : U32) :
    hashtrinity.overflow.Overflow.in_order_loop0_loop1 v st c =
      hashtrinity.overflow.Overflow.in_order_loop0_loop0 v st c := by
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop1
    hashtrinity.overflow.Overflow.in_order_loop0_loop0
    hashtrinity.overflow.Overflow.in_order_loop0_loop1.body
    hashtrinity.overflow.Overflow.in_order_loop0_loop0.body
  rfl

theorem push_spine_spec (v : alloc.vec.Vec ONode) (pre : List Nat) (T : OTree)
    (stack : alloc.vec.Vec U32) (cur : U32)
    (hinv : ∃ Tc G, Emb v.val cur Tc ∧ StackOK v.val stack.val G ∧ pre ++ idxs Tc ++ pendG G = idxs T)
    (r : alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.in_order_loop0_loop0 v stack cur = ok r) :
    ∃ G, StackOK v.val r.val G ∧ pre ++ pendG G = idxs T := by
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0 at h
  refine loop_pc (fun (p : alloc.vec.Vec U32 × U32) => ∃ Tc G, Emb v.val p.2 Tc ∧
      StackOK v.val p.1.val G ∧ pre ++ idxs Tc ++ pendG G = idxs T)
    (fun r => ∃ G, StackOK v.val r.val G ∧ pre ++ pendG G = idxs T) _ ?_ (stack, cur) hinv r h
  rintro ⟨st, c⟩ ⟨Tc, G, hE, ⟨hst, hfr⟩, hp⟩ cf hcf
  dsimp only at hcf hE hst
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0.body at hcf
  by_cases hc : c = hashtrinity.overflow.NIL
  · rw [if_neg (by simp [hc])] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    rw [hc] at hE
    have := emb_nil hE
    subst this
    exact ⟨G, ⟨hst, hfr⟩, by simpa [idxs] using hp⟩
  · rw [if_pos (by simp [hc])] at hcf
    obtain ⟨st1, hst1, hcf⟩ := bind_inv hcf
    simp only [lift, bind_tc_ok] at hcf
    obtain ⟨n, hn, hcf⟩ := bind_inv hcf
    have := (Result.ok_injective hcf).symm
    subst this
    obtain ⟨l, i, r', rfl, hci, nd, hnd, hl, hr⟩ := emb_ne_nil hE hc
    have hn' := idx_ok hn
    rw [hci, hnd] at hn'
    have := Option.some.inj hn'
    subst this
    refine ⟨l, (i, r') :: G, hl, ⟨?_, ?_⟩, ?_⟩
    · show st1.val.map _ = _
      rw [vec_push_ok _ _ _ hst1, List.map_append, hst]; simp [hci]
    · intro p hp'
      rcases List.mem_cons.mp hp' with rfl | hp'
      · exact ⟨nd, hnd, hr⟩
      · exact hfr p hp'
    · rw [← hp]; simp [idxs, pendG]

/-- **`in_order_spec`.**  On an embedded tree whose nodes are all `alive`, the in-order walk from an
empty output returns the tree's arena indices in order. -/
theorem in_order_spec (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hal : ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.alive = true)
    (r : alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.in_order CI o (alloc.vec.Vec.new U32) = ok r) :
    r.val.map (fun x => x.val) = idxs T := by
  unfold hashtrinity.overflow.Overflow.in_order hashtrinity.overflow.Overflow.in_order_loop0 at h
  refine loop_pc (fun (p : alloc.vec.Vec U32 × alloc.vec.Vec U32 × U32) => ∃ Tc G,
      Emb o.nodes.val p.2.2 Tc ∧ StackOK o.nodes.val p.2.1.val G ∧
      p.1.val.map (fun x => x.val) ++ idxs Tc ++ pendG G = idxs T)
    (fun r => r.val.map (fun x => x.val) = idxs T) _ ?_ _ ⟨T, [], hE, ⟨by simp [alloc.vec.Vec.new],
      by simp⟩, by simp [alloc.vec.Vec.new, pendG]⟩ r h
  rintro ⟨out, st, c⟩ ⟨Tc, G, hEc, hS, hp⟩ cf hcf
  dsimp only at hcf hEc hS hp
  unfold hashtrinity.overflow.Overflow.in_order_loop0.body at hcf
  -- the pop step shared by both branches
  have pop_step : ∀ (st1 : alloc.vec.Vec U32) (G1 : List (Nat × OTree)),
      StackOK o.nodes.val st1.val G1 → out.val.map (fun x => x.val) ++ pendG G1 = idxs T →
      ∀ (o2 : Option U32) (st2 : alloc.vec.Vec U32),
      alloc.vec.Vec.pop Global st1 = ok (o2, st2) →
      ∀ (n : U32), core.option.Option.expect o2 (toStr "nonempty stack") = ok n →
      ∀ (n1 : ONode), alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) o.nodes
        (UScalar.cast .Usize n) = ok n1 →
      ∀ out1, (if n1.alive then alloc.vec.Vec.push out n else ok out) = ok out1 →
      ∀ (n2 : ONode), alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) o.nodes
        (UScalar.cast .Usize n) = ok n2 →
      ∃ Tc G, Emb o.nodes.val n2.right Tc ∧ StackOK o.nodes.val st2.val G ∧
        out1.val.map (fun x => x.val) ++ idxs Tc ++ pendG G = idxs T := by
    intro st1 G1 ⟨hst1, hfr1⟩ hp1 o2 st2 hpop n hexp n1 hn1 out1 hout1 n2 hn2
    rcases pop_spec _ _ _ hpop with ⟨_, rfl, _⟩ | ⟨y, hy, rfl⟩
    · simp [core.option.Option.expect, Result.ofOption] at hexp
    have hny : n = y := by
      simp only [core.option.Option.expect, Result.ofOption] at hexp
      exact (Result.ok_injective hexp).symm
    subst hny
    rw [hy, List.map_append] at hst1
    obtain ⟨R, G', rfl, hG'⟩ := stack_pop_frame hst1.symm
    obtain ⟨nd, hnd, hR⟩ := hfr1 _ (List.mem_cons_self ..)
    have hn1' := idx_ok hn1
    rw [hnd] at hn1'
    have := Option.some.inj hn1'
    subst this
    have hn2' := idx_ok hn2
    rw [hnd] at hn2'
    have := Option.some.inj hn2'
    subst this
    have hyT : n.val ∈ idxs T := by
      rw [← hp1]; simp [pendG]
    obtain ⟨nd', hnd', hal'⟩ := hal n.val hyT
    rw [hnd] at hnd'
    have := Option.some.inj hnd'
    subst this
    rw [if_pos hal'] at hout1
    refine ⟨R, G', hR, ⟨hG'.symm, fun p hp' => hfr1 p (List.mem_cons_of_mem _ hp')⟩, ?_⟩
    rw [vec_push_ok _ _ _ hout1, ← hp1]
    simp [pendG]
  by_cases hc : c = hashtrinity.overflow.NIL
  · rw [if_neg (by simp [hc])] at hcf
    rw [hc] at hEc
    have := emb_nil hEc
    subst this
    obtain ⟨b, hb, hcf⟩ := bind_inv hcf
    have hb' := (Result.ok_injective hb).symm
    subst hb'
    by_cases he : st.val.length = 0
    · rw [if_pos (by simp [he])] at hcf
      have := (Result.ok_injective hcf).symm
      subst this
      have hG : G = [] := by
        have h1 := congrArg List.length hS.1
        rw [List.length_map, List.length_reverse, List.length_map, he] at h1
        exact List.eq_nil_of_length_eq_zero h1.symm
      subst hG
      simpa [idxs, pendG] using hp
    · rw [if_neg (by simp [he])] at hcf
      obtain ⟨st1, hst1, hcf⟩ := bind_inv hcf
      rw [loop1_eq] at hst1
      obtain ⟨G1, hS1, hp1⟩ := push_spine_spec o.nodes (out.val.map (fun x => x.val)) T st c
        ⟨.leaf, G, by rw [hc]; exact NIL_val, hS, hp⟩ st1 hst1
      obtain ⟨⟨o2, st2⟩, hpop, hcf⟩ := bind_inv hcf
      obtain ⟨n, hexp, hcf⟩ := bind_inv hcf
      simp only [lift, bind_tc_ok] at hcf
      obtain ⟨n1, hn1, hcf⟩ := bind_inv hcf
      obtain ⟨out1, hout1, hcf⟩ := bind_inv hcf
      obtain ⟨n2, hn2, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      exact pop_step st1 G1 hS1 hp1 o2 st2 hpop n hexp n1 hn1 out1 hout1 n2 hn2
  · rw [if_pos (by simp [hc])] at hcf
    obtain ⟨st1, hst1, hcf⟩ := bind_inv hcf
    obtain ⟨G1, hS1, hp1⟩ := push_spine_spec o.nodes (out.val.map (fun x => x.val)) T st c
      ⟨Tc, G, hEc, hS, hp⟩ st1 hst1
    obtain ⟨⟨o2, st2⟩, hpop, hcf⟩ := bind_inv hcf
    obtain ⟨n, hexp, hcf⟩ := bind_inv hcf
    simp only [lift, bind_tc_ok] at hcf
    obtain ⟨n1, hn1, hcf⟩ := bind_inv hcf
    obtain ⟨out1, hout1, hcf⟩ := bind_inv hcf
    obtain ⟨n2, hn2, hcf⟩ := bind_inv hcf
    have := (Result.ok_injective hcf).symm
    subst this
    exact pop_step st1 G1 hS1 hp1 o2 st2 hpop n hexp n1 hn1 out1 hout1 n2 hn2

/-- The triple `into_entries` emits for arena node `j`: its coordinate, key and value. -/
def entOf (ns : List ONode) (m : Apeiron.Table.HMap U64 I64) (j : Nat) : U64 × U64 × I64 :=
  match ns[j]? with
  | some nd => (nd.key, nd.key, m nd.key)
  | none => (0#u64, 0#u64, 0#i64)

/-- The entry scan of one node whose only entry is `(k, some w)`. -/
theorem entries_loop_single (key k : U64) (w : I64)
    (iter : alloc.vec.into_iter.IntoIter (U64 × Option I64))
    (out out1 : alloc.vec.Vec (U64 × U64 × I64)) (hit : iter.val = [(k, some w)])
    (h : hashtrinity.overflow.Overflow.into_entries_loop0_loop0 iter out key = ok out1) :
    out1.val = out.val ++ [(key, k, w)] := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0 at h
  refine loop_pc (fun (p : alloc.vec.into_iter.IntoIter (U64 × Option I64) ×
      alloc.vec.Vec (U64 × U64 × I64)) =>
      (p.1.val = [(k, some w)] ∧ p.2.val = out.val) ∨
        (p.1.val = [] ∧ p.2.val = out.val ++ [(key, k, w)]))
    (fun r => r.val = out.val ++ [(key, k, w)]) _ ?_ (iter, out) (Or.inl ⟨hit, rfl⟩) out1 h
  rintro ⟨it, o⟩ hinv cf hcf
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0.body at hcf
  obtain ⟨⟨x, it1⟩, hn, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨hd, hcons, rfl⟩
  · have := (Result.ok_injective hcf).symm
    subst this
    rcases hinv with ⟨h1, _⟩ | ⟨_, h2⟩
    · rw [hnil] at h1; exact absurd h1 (by simp)
    · exact h2
  · rcases hinv with ⟨h1, h2⟩ | ⟨h1, _⟩
    · rw [h1] at hcons
      simp only [List.cons.injEq] at hcons
      obtain ⟨rfl, hrest⟩ := hcons
      obtain ⟨o1, hpush, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      exact Or.inr ⟨hrest.symm, by rw [vec_push_ok _ _ _ hpush, h2]⟩
    · rw [h1] at hcons; exact absurd hcons (by simp)

/-- **`into_entries_spec`.**  On a tree whose nodes are `alive` and hold the single entry
`(key, some (m key))`, `into_entries` returns `(key, key, m key)` for each node, in order. -/
theorem into_entries_spec (m : Apeiron.Table.HMap U64 I64) (o : Ov) (T : OTree)
    (hE : Emb o.nodes.val o.root T) (hND : ND T)
    (hall : ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧ nd.alive = true ∧
      nd.entries.val = [(nd.key, some (m nd.key))])
    (es : alloc.vec.Vec (U64 × U64 × I64))
    (h : hashtrinity.overflow.Overflow.into_entries CI o = ok es) :
    es.val = (idxs T).map (entOf o.nodes.val m) := by
  unfold hashtrinity.overflow.Overflow.into_entries at h
  obtain ⟨order, hord, h1⟩ := bind_inv h
  have hordv := in_order_spec o T hE (fun i hi => by
    obtain ⟨nd, hnd, ha, _⟩ := hall i hi; exact ⟨nd, hnd, ha⟩) order hord
  obtain ⟨iter, hiter, h2⟩ := bind_inv h1
  unfold alloc.vec.IntoIteratorVec.into_iter at hiter
  have hio : iter = order := (Result.ok_injective hiter).symm
  subst hio
  have hnd := nd_nodup T hND
  unfold hashtrinity.overflow.Overflow.into_entries_loop0 at h2
  refine loop_pc (fun (p : alloc.vec.into_iter.IntoIter U32 × alloc.vec.Vec ONode ×
      alloc.vec.Vec (U64 × U64 × I64)) => ∃ dn, dn ++ p.1.val.map (fun x => x.val) = idxs T ∧
      p.2.1.val.length = o.nodes.val.length ∧
      (∀ j ∈ p.1.val.map (fun x => x.val), p.2.1.val[j]? = o.nodes.val[j]?) ∧
      p.2.2.val = dn.map (entOf o.nodes.val m))
    (fun r => r.val = (idxs T).map (entOf o.nodes.val m)) _ ?_ _
    ⟨[], by simpa using hordv, rfl, fun _ _ => rfl, rfl⟩ es h2
  rintro ⟨it, v, out⟩ ⟨dn, hdn, hlen, hagree, hout⟩ cf hcf
  dsimp only at hcf hdn hlen hagree hout
  unfold hashtrinity.overflow.Overflow.into_entries_loop0.body at hcf
  obtain ⟨⟨x, it1⟩, hn, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hn with ⟨hnil, rfl⟩ | ⟨n, hcons, rfl⟩
  · have := (Result.ok_injective hcf).symm
    subst this
    rw [hnil] at hdn
    simp only [List.map_nil, List.append_nil] at hdn
    show out.val = _
    rw [hout, hdn]
  · simp only [lift, bind_tc_ok] at hcf
    obtain ⟨⟨node, back⟩, hidx, hcf⟩ := bind_inv hcf
    obtain ⟨hnode, hback⟩ := idxm_ok hidx
    obtain ⟨⟨ents, v1⟩, htake, hcf⟩ := bind_inv hcf
    unfold core.mem.take at htake
    obtain ⟨d, -, htake⟩ := bind_inv htake
    have htk := Result.ok_injective htake
    simp only [Prod.mk.injEq] at htk
    obtain ⟨rfl, rfl⟩ := htk
    obtain ⟨iter2, hit2, hcf⟩ := bind_inv hcf
    unfold alloc.vec.IntoIteratorVec.into_iter at hit2
    have := (Result.ok_injective hit2).symm
    subst this
    obtain ⟨out1, hout1, hcf⟩ := bind_inv hcf
    have := (Result.ok_injective hcf).symm
    subst this
    rw [hcons, List.map_cons] at hdn hagree
    have hnT : n.val ∈ idxs T := by rw [← hdn]; simp
    have hv : v.val[n.val]? = o.nodes.val[n.val]? := hagree n.val (List.mem_cons_self ..)
    obtain ⟨nd, hndo, _, hent⟩ := hall n.val hnT
    rw [hv, hndo] at hnode
    have := Option.some.inj hnode
    subst this
    have hout1v := entries_loop_single nd.key nd.key (m nd.key) _ out out1 hent hout1
    have hnodup : (dn ++ n.val :: it1.val.map (fun x => x.val)).Nodup := by rw [hdn]; exact hnd
    have hnotin : n.val ∉ it1.val.map (fun x => x.val) := by
      have := (List.nodup_append.mp hnodup).2.1
      exact (List.nodup_cons.mp this).1
    refine ⟨dn ++ [n.val], by rw [← hdn]; simp, ?_, fun j hj => ?_, ?_⟩
    · rw [hback, List.length_set, hlen]
    · rw [hback, List.getElem?_set_ne (fun (e : n.val = j) => hnotin (by rw [e]; exact hj))]
      exact hagree j (List.mem_cons_of_mem _ hj)
    · rw [hout1v, hout, List.map_append]
      simp [entOf, hndo]

end Apeiron.MapExecuted
