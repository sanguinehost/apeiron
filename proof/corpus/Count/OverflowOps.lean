-- Row: `apeiron-hashdos-cap-count-MUBqkA7OYTvluKSIRdbjC`: the overflow store as the count face uses it.
import ApeironWord
import Count.Helpers
import Count.FreqRep
import Map.ExecutedOverflowInsert
open Aeneas Aeneas.Std Result apeiron_word

/-!
# The overflow store under the count face

The count face keys its overflow tree by the frequency itself (`coord = k = f`), so every node holds
exactly one live entry `(key, some w)`.  This module proves, on the extracted
`hashtrinity.overflow.Overflow U64 I64`, the operations the count face reaches that the map face's
read/insert specs do not cover:

* `in_order_spec`, `into_entries_spec`: the traversal returns every node of the tree, in order;
* `new_ovok`: `Overflow::new` is the empty AVL tree;
* `update_spec`: `take_value` then `write_back` replaces the value of one entry;
* `insert_ovok`: `insert_new_at` of a fresh frequency preserves the single-entry invariant.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Apeiron.MapExecuted (ONode Ov OTree Emb BST OInv idxs nilN NIL_val NIL_eq_iff ND EntryOf keyAt
  FirstAt Present)

/-! ## Vector facts -/

theorem vec_push_ok {α : Type} (v v1 : alloc.vec.Vec α) (x : α)
    (h : alloc.vec.Vec.push v x = ok v1) : v1.val = v.val ++ [x] := by
  unfold alloc.vec.Vec.push at h
  dsimp only at h
  split at h
  · have := Result.ok_injective h
    subst this
    simp [alloc.vec.Vec.from_val]
  · exact absurd h (by intro hh; exact ok_not_fail hh.symm)

theorem ne_nil_iff (x : U32) : (x != hashtrinity.overflow.NIL) = true ↔ x.val ≠ nilN :=
  (bne_iff_ne).trans (not_congr (NIL_eq_iff x))

theorem nd_nodup : ∀ {T : OTree}, ND T → (idxs T).Nodup
  | .leaf, _ => List.nodup_nil
  | .node l i r, ⟨hl, hr, hil, hir, hlr⟩ => by
    simp only [idxs]
    rw [List.nodup_append]
    refine ⟨nd_nodup hl, ?_, ?_⟩
    · rw [List.nodup_cons]; exact ⟨hir, nd_nodup hr⟩
    · intro a ha b hb hab
      subst hab
      simp only [List.mem_cons] at hb
      rcases hb with hb | hb
      · subst hb; exact hil ha
      · exact hlr a ha hb

/-! ## `in_order` -/

/-- Pending ancestors: node `i`, whose right subtree is `R`. -/
def fl (fr : List (Nat × OTree)) : List Nat := fr.flatMap (fun p => p.1 :: idxs p.2)

theorem fl_cons (j : Nat) (R : OTree) (fr : List (Nat × OTree)) :
    fl ((j, R) :: fr) = j :: idxs R ++ fl fr := by
  simp [fl]

def FramesOK (ns : List ONode) : List (Nat × OTree) → Prop
  | [] => True
  | (i, R) :: rest => i ≠ nilN ∧ (∃ nd, ns[i]? = some nd ∧ Emb ns nd.right R) ∧ FramesOK ns rest

theorem emb_leaf_iff {ns : List ONode} {x : U32} : Emb ns x .leaf ↔ x.val = nilN := Iff.rfl

theorem in_order_descend (v : alloc.vec.Vec ONode) (stack : alloc.vec.Vec U32) (cur : U32)
    (T0 : OTree) (fr : List (Nat × OTree))
    (hE : Emb v.val cur T0) (hF : FramesOK v.val fr)
    (hS : stack.val.map (fun x => x.val) = (fr.map Prod.fst).reverse)
    (stack1 : alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.in_order_loop0_loop0 v stack cur = ok stack1) :
    ∃ fr', FramesOK v.val fr' ∧ stack1.val.map (fun x => x.val) = (fr'.map Prod.fst).reverse ∧
      fl fr' = idxs T0 ++ fl fr := by
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0 at h
  refine loop_pc (fun (st : alloc.vec.Vec U32 × U32) =>
      ∃ T1 fr1, Emb v.val st.2 T1 ∧ FramesOK v.val fr1 ∧
        st.1.val.map (fun x => x.val) = (fr1.map Prod.fst).reverse ∧
        idxs T1 ++ fl fr1 = idxs T0 ++ fl fr)
    (fun (y : alloc.vec.Vec U32) => ∃ fr', FramesOK v.val fr' ∧
      y.val.map (fun x => x.val) = (fr'.map Prod.fst).reverse ∧ fl fr' = idxs T0 ++ fl fr)
    _ ?_ (stack, cur) ⟨T0, fr, hE, hF, hS, rfl⟩ stack1 h
  intro st hst cf hcf
  obtain ⟨stk, c⟩ := st
  obtain ⟨T1, fr1, hE1, hF1, hS1, hI1⟩ := hst
  dsimp only at hE1 hS1 hcf
  unfold hashtrinity.overflow.Overflow.in_order_loop0_loop0.body at hcf
  by_cases hc : c != hashtrinity.overflow.NIL
  · rw [if_pos hc] at hcf
    have hcv := (ne_nil_iff c).mp hc
    obtain ⟨stack2, hpush, hcf⟩ := bind_inv hcf
    obtain ⟨i, hi, hcf⟩ := bind_inv hcf
    obtain ⟨n, hn, hcf⟩ := bind_inv hcf
    have hcf' : cf = ControlFlow.cont (stack2, n.left) := (Result.ok_injective hcf).symm
    subst hcf'
    have hp := vec_push_ok _ _ _ hpush
    have hn' := Apeiron.MapExecuted.vec_index_ok _ _ _ hn
    have hi' : i = UScalar.cast .Usize c := (Result.ok_injective hi).symm
    subst hi'
    rw [Apeiron.MapExecuted.cast_usize_u32] at hn'
    cases T1 with
    | leaf => exact absurd hE1 (by rw [emb_leaf_iff]; exact hcv)
    | node l j r =>
      obtain ⟨hcj, hj, nd, hnd, hl, hr⟩ := hE1
      rw [hcj, hnd] at hn'
      have hnn : nd = n := Option.some.inj hn'
      subst hnn
      refine ⟨l, (j, r) :: fr1, hl, ⟨hj, ⟨nd, hnd, hr⟩, hF1⟩, ?_, ?_⟩
      · rw [hp, List.map_append, hS1, ← hcj]
        simp
      · simp only [fl, List.flatMap_cons] at hI1 ⊢
        simp only [idxs, List.append_assoc, List.cons_append] at hI1 ⊢
        exact hI1
  · rw [if_neg hc] at hcf
    have hcf' : cf = ControlFlow.done stk := (Result.ok_injective hcf).symm
    subst hcf'
    have hcv : c.val = nilN := by
      by_contra hne
      exact hc ((ne_nil_iff c).mpr hne)
    have hT1 : T1 = .leaf := by
      cases T1 with
      | leaf => rfl
      | node l j r => exact absurd hcv (by obtain ⟨hcj, hj, _⟩ := hE1; rw [hcj]; exact hj)
    subst hT1
    exact ⟨fr1, hF1, hS1, by simpa [idxs] using hI1⟩

/-- Variant for the second copy of the loop (identical body). -/
theorem in_order_descend' (v : alloc.vec.Vec ONode) (stack : alloc.vec.Vec U32) (cur : U32)
    (T0 : OTree) (fr : List (Nat × OTree))
    (hE : Emb v.val cur T0) (hF : FramesOK v.val fr)
    (hS : stack.val.map (fun x => x.val) = (fr.map Prod.fst).reverse)
    (stack1 : alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.in_order_loop0_loop1 v stack cur = ok stack1) :
    ∃ fr', FramesOK v.val fr' ∧ stack1.val.map (fun x => x.val) = (fr'.map Prod.fst).reverse ∧
      fl fr' = idxs T0 ++ fl fr :=
  in_order_descend v stack cur T0 fr hE hF hS stack1 h


/-- The tail of an `in_order` iteration: pop the innermost pending ancestor, record it, step right. -/
theorem in_order_tail (v : alloc.vec.Vec ONode) (out stack1 stack2 out1 : alloc.vec.Vec U32)
    (o : Option U32) (n : U32) (i i1 : Usize) (n1 n2 : ONode)
    (fr' : List (Nat × OTree)) (hF : FramesOK v.val fr')
    (hS : stack1.val.map (fun x => x.val) = (fr'.map Prod.fst).reverse) (hne : fr' ≠ [])
    (hAl : ∀ j ∈ fl fr', ∃ nd, v.val[j]? = some nd ∧ nd.alive = true)
    (hpop : alloc.vec.Vec.pop Global stack1 = ok (o, stack2))
    (hexp : core.option.Option.expect o (toStr "nonempty stack") = ok n)
    (hi : (lift (UScalar.cast .Usize n) : Result Usize) = ok i)
    (hn1 : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v i = ok n1)
    (hout : (if n1.alive then alloc.vec.Vec.push out n else ok out) = ok out1)
    (hi1 : (lift (UScalar.cast .Usize n) : Result Usize) = ok i1)
    (hn2 : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice ONode) v i1 = ok n2) :
    ∃ j R fr'', fr' = (j, R) :: fr'' ∧ Emb v.val n2.right R ∧ FramesOK v.val fr'' ∧
      stack2.val.map (fun x => x.val) = (fr''.map Prod.fst).reverse ∧
      out1.val.map (fun x => x.val) = out.val.map (fun x => x.val) ++ [j] := by
  have hi' : i = UScalar.cast .Usize n := (Result.ok_injective hi).symm
  have hi1' : i1 = UScalar.cast .Usize n := (Result.ok_injective hi1).symm
  subst hi' hi1'
  rcases Apeiron.MapExecuted.pop_spec _ _ _ hpop with ⟨hv0, _, _⟩ | ⟨y, hy, hoy⟩
  · exfalso
    apply hne
    rw [hv0] at hS
    simp only [List.map_nil] at hS
    cases fr' with
    | nil => rfl
    | cons a rest => simp at hS
  · subst hoy
    have hny : n = y := by
      unfold core.option.Option.expect at hexp
      simpa [Result.ofOption] using hexp.symm
    subst hny
    cases fr' with
    | nil => simp at hne
    | cons hd rest =>
      obtain ⟨j, R⟩ := hd
      obtain ⟨hj, ⟨nd, hnd, hR⟩, hrest⟩ := hF
      have hS' : stack2.val.map (fun x => x.val) ++ [n.val] = (rest.map Prod.fst).reverse ++ [j] := by
        have := hS
        rw [hy, List.map_append] at this
        simpa using this
      have hjn : n.val = j := by
        have := List.append_inj' hS' rfl
        simpa using this.2
      have hstk : stack2.val.map (fun x => x.val) = (rest.map Prod.fst).reverse := by
        have := List.append_inj' hS' rfl
        exact this.1
      have hjAl := hAl j (by simp [fl])
      obtain ⟨ndj, hndj, halive⟩ := hjAl
      have hn1' := Apeiron.MapExecuted.vec_index_ok _ _ _ hn1
      rw [Apeiron.MapExecuted.cast_usize_u32, hjn, hndj] at hn1'
      have hn1n : ndj = n1 := Option.some.inj hn1'
      subst hn1n
      rw [if_pos halive] at hout
      have hp := vec_push_ok _ _ _ hout
      have hn2' := Apeiron.MapExecuted.vec_index_ok _ _ _ hn2
      rw [Apeiron.MapExecuted.cast_usize_u32, hjn, hnd] at hn2'
      have hn2n : nd = n2 := Option.some.inj hn2'
      subst hn2n
      refine ⟨j, R, rest, rfl, hR, hrest, hstk, ?_⟩
      rw [hp, List.map_append]
      simp [hjn]


/-- **`in_order`.**  On an embedded tree whose nodes are alive, the traversal returns exactly the
node indices of the tree, in order. -/
theorem in_order_spec (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hAl : ∀ j ∈ idxs T, ∃ nd, o.nodes.val[j]? = some nd ∧ nd.alive = true)
    (out : alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.in_order Apeiron.MapExecuted.CI o (alloc.vec.Vec.new U32) = ok out) :
    out.val.map (fun x => x.val) = idxs T := by
  unfold hashtrinity.overflow.Overflow.in_order at h
  unfold hashtrinity.overflow.Overflow.in_order_loop0 at h
  refine loop_pc (fun (st : alloc.vec.Vec U32 × alloc.vec.Vec U32 × U32) =>
      ∃ T0 fr, Emb o.nodes.val st.2.2 T0 ∧ FramesOK o.nodes.val fr ∧
        st.2.1.val.map (fun x => x.val) = (fr.map Prod.fst).reverse ∧
        idxs T = st.1.val.map (fun x => x.val) ++ idxs T0 ++ fl fr)
    (fun (y : alloc.vec.Vec U32) => y.val.map (fun x => x.val) = idxs T)
    _ ?_ (alloc.vec.Vec.new U32, alloc.vec.Vec.new U32, o.root)
    ⟨T, [], hE, trivial, rfl, by simp [fl]⟩ out h
  intro st hst cf hcf
  obtain ⟨ot, stk, c⟩ := st
  obtain ⟨T0, fr, hE0, hF0, hS0, hI0⟩ := hst
  dsimp only at hE0 hS0 hI0 hcf
  unfold hashtrinity.overflow.Overflow.in_order_loop0.body at hcf
  have hmemT : ∀ j, j ∈ idxs T0 ++ fl fr → j ∈ idxs T := by
    intro j hj
    rw [hI0]; simp only [List.mem_append]
    rcases List.mem_append.mp hj with h1 | h1
    · exact Or.inl (Or.inr h1)
    · exact Or.inr h1
  by_cases hc : c != hashtrinity.overflow.NIL
  · rw [if_pos hc] at hcf
    have hcv := (ne_nil_iff c).mp hc
    obtain ⟨stack1, hdesc, hcf⟩ := bind_inv hcf
    obtain ⟨⟨o1, stack2⟩, hpop, hcf⟩ := bind_inv hcf
    obtain ⟨n, hexp, hcf⟩ := bind_inv hcf
    obtain ⟨i, hi, hcf⟩ := bind_inv hcf
    obtain ⟨n1, hn1, hcf⟩ := bind_inv hcf
    obtain ⟨out1, hout, hcf⟩ := bind_inv hcf
    obtain ⟨i1, hi1, hcf⟩ := bind_inv hcf
    obtain ⟨n2, hn2, hcf⟩ := bind_inv hcf
    have hcf' : cf = ControlFlow.cont (out1, stack2, n2.right) := (Result.ok_injective hcf).symm
    subst hcf'
    obtain ⟨fr', hF', hS', hfl'⟩ := in_order_descend _ _ _ _ _ hE0 hF0 hS0 _ hdesc
    have hT0ne : idxs T0 ≠ [] := by
      cases T0 with
      | leaf => exact absurd hE0 (by rw [emb_leaf_iff]; exact hcv)
      | node l j r => simp [idxs]
    have hne : fr' ≠ [] := by
      intro hh
      rw [hh] at hfl'
      simp [fl] at hfl'
      exact hT0ne hfl'.1
    have hAl' : ∀ j ∈ fl fr', ∃ nd, o.nodes.val[j]? = some nd ∧ nd.alive = true := by
      intro j hj
      rw [hfl'] at hj
      exact hAl j (hmemT j hj)
    obtain ⟨j, R, fr'', hfr, hR, hF'', hS'', hout1⟩ :=
      in_order_tail _ _ _ _ _ _ _ _ _ _ _ _ hF' hS' hne hAl' hpop hexp hi hn1 hout hi1 hn2
    subst hfr
    refine ⟨R, fr'', hR, hF'', hS'', ?_⟩
    rw [hout1]
    have : idxs T = ot.val.map (fun x => x.val) ++ (j :: idxs R ++ fl fr'') := by
      rw [hI0, List.append_assoc, ← fl_cons, hfl']
    rw [this]
    simp
  · rw [if_neg hc] at hcf
    have hcv : c.val = nilN := by
      by_contra hne
      exact hc ((ne_nil_iff c).mpr hne)
    have hT0 : T0 = .leaf := by
      cases T0 with
      | leaf => rfl
      | node l j r => exact absurd hcv (by obtain ⟨hcj, hj, _⟩ := hE0; rw [hcj]; exact hj)
    subst hT0
    obtain ⟨b, hb, hcf⟩ := bind_inv hcf
    have hbv : b = decide (stk.val.length = 0) := (Result.ok_injective hb).symm
    by_cases hb' : b = true
    · rw [hb'] at hcf
      simp only [if_true] at hcf
      have hcf' : cf = ControlFlow.done ot := (Result.ok_injective hcf).symm
      subst hcf'
      have hlen : stk.val.length = 0 := by simpa [hbv] using hb'
      have hfr : fr = [] := by
        have h0 : stk.val = [] := List.length_eq_zero_iff.mp hlen
        rw [h0] at hS0
        cases fr with
        | nil => rfl
        | cons a rest => simp at hS0
      subst hfr
      simpa [fl, idxs] using hI0.symm
    · have hb0 : b = false := by simpa using hb'
      rw [hb0] at hcf
      simp only [Bool.false_eq_true, if_false] at hcf
      obtain ⟨stack1, hdesc, hcf⟩ := bind_inv hcf
      obtain ⟨⟨o1, stack2⟩, hpop, hcf⟩ := bind_inv hcf
      obtain ⟨n, hexp, hcf⟩ := bind_inv hcf
      obtain ⟨i, hi, hcf⟩ := bind_inv hcf
      obtain ⟨n1, hn1, hcf⟩ := bind_inv hcf
      obtain ⟨out1, hout, hcf⟩ := bind_inv hcf
      obtain ⟨i1, hi1, hcf⟩ := bind_inv hcf
      obtain ⟨n2, hn2, hcf⟩ := bind_inv hcf
      have hcf' : cf = ControlFlow.cont (out1, stack2, n2.right) := (Result.ok_injective hcf).symm
      subst hcf'
      obtain ⟨fr', hF', hS', hfl'⟩ := in_order_descend' _ _ _ _ _ hE0 hF0 hS0 _ hdesc
      have hfrne : fr ≠ [] := by
        intro hh
        subst hh
        have hlen : stk.val.length = 0 := by
          have : stk.val.map (fun x => x.val) = [] := by simpa using hS0
          simpa using this
        simp [hbv, hlen] at hb0
      have hne : fr' ≠ [] := by
        intro hh
        rw [hh] at hfl'
        cases fr with
        | nil => exact hfrne rfl
        | cons a rest => simp [fl, idxs] at hfl'
      have hAl' : ∀ j ∈ fl fr', ∃ nd, o.nodes.val[j]? = some nd ∧ nd.alive = true := by
        intro j hj
        rw [hfl'] at hj
        exact hAl j (hmemT j hj)
      obtain ⟨j, R, fr'', hfr, hR, hF'', hS'', hout1⟩ :=
        in_order_tail _ _ _ _ _ _ _ _ _ _ _ _ hF' hS' hne hAl' hpop hexp hi hn1 hout hi1 hn2
      subst hfr
      refine ⟨R, fr'', hR, hF'', hS'', ?_⟩
      rw [hout1]
      have : idxs T = ot.val.map (fun x => x.val) ++ (j :: idxs R ++ fl fr'') := by
        rw [hI0, List.append_assoc, ← fl_cons, hfl']
      rw [this]
      simp


/-! ## `into_entries` -/

/-- The remaining elements of a by-value vector iterator. -/
def ivl {α : Type} (it : alloc.vec.into_iter.IntoIter α) : List α := (it : alloc.vec.Vec α).val

theorem next_spec {α : Type} (it it' : alloc.vec.into_iter.IntoIter α) (o : Option α)
    (h : alloc.vec.into_iter.IteratorIntoIter.next it = ok (o, it')) :
    (ivl it = [] ∧ o = none ∧ ivl it' = []) ∨ ∃ hd tl, ivl it = hd :: tl ∧ o = some hd ∧ ivl it' = tl := by
  unfold alloc.vec.into_iter.IteratorIntoIter.next at h
  split at h
  · rename_i hnil
    left
    have := Result.ok_injective h
    simp only [Prod.mk.injEq] at this
    obtain ⟨h1, h2⟩ := this
    subst h2
    exact ⟨hnil, h1.symm, hnil⟩
  · rename_i hd tl hcons
    right
    have := Result.ok_injective h
    simp only [Prod.mk.injEq] at this
    obtain ⟨h1, h2⟩ := this
    refine ⟨hd, tl, hcons, h1.symm, ?_⟩
    rw [← h2]
    simp [ivl, alloc.vec.Vec.from_val]


theorem into_entries_inner (key : U64) (iter : alloc.vec.into_iter.IntoIter (U64 × Option I64))
    (out out1 : alloc.vec.Vec (U64 × U64 × I64))
    (h : hashtrinity.overflow.Overflow.into_entries_loop0_loop0 iter out key = ok out1) :
    out1.val = out.val ++ (ivl iter).filterMap (fun p => p.2.map (fun w => (key, p.1, w))) := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0 at h
  refine loop_pc (fun (st : alloc.vec.into_iter.IntoIter (U64 × Option I64) ×
        alloc.vec.Vec (U64 × U64 × I64)) =>
      out.val ++ (ivl iter).filterMap (fun p => p.2.map (fun w => (key, p.1, w))) =
        st.2.val ++ (ivl st.1).filterMap (fun p => p.2.map (fun w => (key, p.1, w))))
    (fun (y : alloc.vec.Vec (U64 × U64 × I64)) =>
      y.val = out.val ++ (ivl iter).filterMap (fun p => p.2.map (fun w => (key, p.1, w))))
    _ ?_ (iter, out) rfl out1 h
  intro st hst cf hcf
  obtain ⟨it, ou⟩ := st
  dsimp only at hst hcf
  unfold hashtrinity.overflow.Overflow.into_entries_loop0_loop0.body at hcf
  obtain ⟨⟨o, it1⟩, hnext, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hnext with ⟨h0, ho, h1⟩ | ⟨hd, tl, h0, ho, h1⟩
  · subst ho
    try simp only at hcf
    have hcf' : cf = ControlFlow.done ou := (Result.ok_injective hcf).symm
    subst hcf'
    rw [hst, h0]; simp
  · subst ho
    obtain ⟨k, w⟩ := hd
    cases w with
    | none =>
      try simp only at hcf
      have hcf' : cf = ControlFlow.cont (it1, ou) := (Result.ok_injective hcf).symm
      subst hcf'
      show _ = ou.val ++ (ivl it1).filterMap _
      rw [hst, h0, h1]; simp
    | some w1 =>
      try simp only at hcf
      obtain ⟨out2, hp, hcf⟩ := bind_inv hcf
      have hcf' : cf = ControlFlow.cont (it1, out2) := (Result.ok_injective hcf).symm
      subst hcf'
      have hpv := vec_push_ok _ _ _ hp
      show _ = out2.val ++ (ivl it1).filterMap _
      rw [hst, h0, h1, hpv]; simp


/-- Count-face node invariant: every tree node is alive and holds the single entry
`(key, some w)` (the overflow is keyed by the frequency, which is also the entry key). -/
def NodeOK (ns : List ONode) (T : OTree) : Prop :=
  ∀ j ∈ idxs T, ∃ nd, ns[j]? = some nd ∧ nd.alive = true ∧ ∃ w, nd.entries.val = [(nd.key, some w)]

/-- The `(coordinate, key, value)` triple `into_entries` yields for node `j`. -/
def EntryRel (ns : List ONode) (j : Nat) (t : U64 × U64 × I64) : Prop :=
  ∃ nd w, ns[j]? = some nd ∧ nd.entries.val = [(nd.key, some w)] ∧ t = (nd.key, nd.key, w)

theorem into_entries_loop_spec (ns : List ONode) (T : OTree) (hND : ND T) (hOK : NodeOK ns T)
    (iter0 : alloc.vec.into_iter.IntoIter U32) (v0 : alloc.vec.Vec ONode)
    (out0 : alloc.vec.Vec (U64 × U64 × I64)) (L : alloc.vec.Vec (U64 × U64 × I64))
    (hit : (ivl iter0).map (fun x => x.val) = idxs T) (hv : v0.val = ns) (hout : out0.val = [])
    (h : hashtrinity.overflow.Overflow.into_entries_loop0 iter0 v0 out0 = ok L) :
    List.Forall₂ (EntryRel ns) (idxs T) L.val := by
  unfold hashtrinity.overflow.Overflow.into_entries_loop0 at h
  have hnd := nd_nodup hND
  refine loop_pc (fun (st : alloc.vec.into_iter.IntoIter U32 × alloc.vec.Vec ONode ×
        alloc.vec.Vec (U64 × U64 × I64)) =>
      ∃ proc rem, idxs T = proc ++ rem ∧ (ivl st.1).map (fun x => x.val) = rem ∧
        List.Forall₂ (EntryRel ns) proc st.2.2.val ∧ ∀ j ∈ rem, st.2.1.val[j]? = ns[j]?)
    (fun (y : alloc.vec.Vec (U64 × U64 × I64)) => List.Forall₂ (EntryRel ns) (idxs T) y.val)
    _ ?_ (iter0, v0, out0) ⟨[], idxs T, rfl, hit, by rw [hout]; exact List.Forall₂.nil,
      fun j _ => by rw [hv]⟩ L h
  intro st hst cf hcf
  obtain ⟨it, vv, ou⟩ := st
  obtain ⟨proc, rem, hsplit, hrem, hF2, hvv⟩ := hst
  dsimp only at hrem hF2 hvv hcf
  unfold hashtrinity.overflow.Overflow.into_entries_loop0.body at hcf
  obtain ⟨⟨o, it1⟩, hnext, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hnext with ⟨h0, ho, h1⟩ | ⟨hd, tl, h0, ho, h1⟩
  · subst ho
    try simp only at hcf
    have hcf' : cf = ControlFlow.done ou := (Result.ok_injective hcf).symm
    subst hcf'
    have : rem = [] := by rw [← hrem, h0]; rfl
    subst this
    rw [hsplit]; simpa using hF2
  · subst ho
    try simp only at hcf
    obtain ⟨i, hi, hcf⟩ := bind_inv hcf
    obtain ⟨⟨node, back⟩, hnode, hcf⟩ := bind_inv hcf
    obtain ⟨⟨entries, v1⟩, htake, hcf⟩ := bind_inv hcf
    obtain ⟨iter2, hiter2, hcf⟩ := bind_inv hcf
    obtain ⟨out1, hinner, hcf⟩ := bind_inv hcf
    have hcf' : cf = ControlFlow.cont (it1, back {node with entries := v1}, out1) :=
      (Result.ok_injective hcf).symm
    subst hcf'
    have hrem' : (hd.val :: (ivl it1).map (fun x => x.val)) = rem := by
      rw [← hrem, h0, h1]; simp
    have hi' : i = UScalar.cast .Usize hd := (Result.ok_injective hi).symm
    subst hi'
    obtain ⟨hn1, hback⟩ := Apeiron.MapExecuted.idxm_ok hnode
    have hmem : hd.val ∈ rem := by rw [← hrem']; simp
    have hnodeeq : vv.val[hd.val]? = some node := hn1
    have hnsn : ns[hd.val]? = some node := by rw [← hvv _ hmem]; exact hnodeeq
    have hT : hd.val ∈ idxs T := by rw [hsplit]; exact List.mem_append_right _ hmem
    obtain ⟨nd0, hnd0, _, w, hw⟩ := hOK hd.val hT
    rw [hnsn] at hnd0
    have hnn : node = nd0 := Option.some.inj hnd0
    subst hnn
    -- the entries vector is taken whole
    have htake' : entries = node.entries ∧ v1 = alloc.vec.Vec.new (U64 × Option I64) := by
      unfold core.mem.take at htake
      obtain ⟨d, hd', htake⟩ := bind_inv htake
      have hdd : d = alloc.vec.Vec.new (U64 × Option I64) := by
        have := hd'
        simpa [alloc.vec.Vec.Insts.CoreDefaultDefault, alloc.vec.Vec.Insts.CoreDefaultDefault.default] using
          (Result.ok_injective this).symm
      have := Result.ok_injective htake
      simp only [Prod.mk.injEq] at this
      exact ⟨this.1.symm, by rw [← this.2, hdd]⟩
    obtain ⟨he, hv1⟩ := htake'
    subst he
    have hiter2' : iter2 = node.entries := (Result.ok_injective hiter2).symm
    have hin := into_entries_inner _ _ _ _ hinner
    have hiter2v : ivl iter2 = [(node.key, some w)] := by
      rw [hiter2']; exact hw
    rw [hiter2v] at hin
    simp only [List.filterMap_cons, List.filterMap_nil, Option.map_some] at hin
    refine ⟨proc ++ [hd.val], (ivl it1).map (fun x => x.val), ?_, rfl, ?_, ?_⟩
    · rw [hsplit, ← hrem']; simp
    · rw [hin]
      exact List.rel_append hF2 (List.Forall₂.cons ⟨node, w, hnsn, hw, rfl⟩ List.Forall₂.nil)
    · intro j hj
      have hjrem : j ∈ rem := by rw [← hrem']; exact List.mem_cons_of_mem _ hj
      have hne : j ≠ hd.val := by
        intro hjj
        subst hjj
        have hnd2 := hnd
        rw [hsplit, ← hrem'] at hnd2
        have := (List.nodup_append.mp hnd2).2.1
        rw [List.nodup_cons] at this
        exact this.1 hj
      show (back {node with entries := v1}).val[j]? = ns[j]?
      rw [hback, List.getElem?_set_ne (Ne.symm hne)]
      exact hvv j hjrem


/-- **`into_entries`.**  The traversal yields one `(key, key, value)` triple per tree node, in order. -/
theorem into_entries_spec (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T) (hND : ND T)
    (hOK : NodeOK o.nodes.val T) (L : alloc.vec.Vec (U64 × U64 × I64))
    (h : hashtrinity.overflow.Overflow.into_entries Apeiron.MapExecuted.CI o = ok L) :
    List.Forall₂ (EntryRel o.nodes.val) (idxs T) L.val := by
  unfold hashtrinity.overflow.Overflow.into_entries at h
  obtain ⟨order, hord, h⟩ := bind_inv h
  obtain ⟨iter, hiter, h⟩ := bind_inv h
  have hAl : ∀ j ∈ idxs T, ∃ nd, o.nodes.val[j]? = some nd ∧ nd.alive = true := by
    intro j hj
    obtain ⟨nd, hnd, ha, _⟩ := hOK j hj
    exact ⟨nd, hnd, ha⟩
  have hs := in_order_spec o T hE hAl order hord
  have hit' : ivl iter = order.val :=
    (congrArg (fun x : alloc.vec.Vec U32 => x.val) (Result.ok_injective hiter)).symm
  exact into_entries_loop_spec o.nodes.val T hND hOK iter o.nodes _ L (by rw [hit', hs]) rfl rfl h

/-- `Overflow::new` is the empty AVL tree. -/
theorem new_spec (o : Ov)
    (h : hashtrinity.overflow.Overflow.new I64 Apeiron.MapExecuted.CI = ok o) :
    OInv o .leaf ∧ NodeOK o.nodes.val .leaf ∧ o.nodes.val = [] := by
  unfold hashtrinity.overflow.Overflow.new at h
  have ho := (Result.ok_injective h).symm
  subst ho
  refine ⟨⟨?_, trivial, trivial, trivial, trivial, ?_, ?_⟩, fun j hj => by simp [idxs] at hj, rfl⟩
  · exact NIL_val
  · exact List.nodup_nil
  · intro j hj; exact absurd hj (by simp)

theorem oinv_node_update (o o2 : Ov) (T : OTree) (hI : OInv o T) (n : Nat) (nd : ONode)
    (hnd : o.nodes.val[n]? = some nd) (ents : alloc.vec.Vec (U64 × Option I64))
    (hn2 : o2.nodes.val = o.nodes.val.set n {nd with entries := ents})
    (hr : o2.root = o.root) (hf : o2.free = o.free) : OInv o2 T := by
  obtain ⟨hE, hB, hND, hH, hBal, hfn, hfl⟩ := hI
  have hlt : n < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
  have hne : ∀ j, j ≠ n → o2.nodes.val[j]? = o.nodes.val[j]? := fun j hj => by
    rw [hn2]; exact List.getElem?_set_ne (Ne.symm hj)
  have hself : o2.nodes.val[n]? = some {nd with entries := ents} := by
    rw [hn2]; exact List.getElem?_set_self hlt
  have hptr : ∀ j ∈ idxs T, ∀ nd0, o.nodes.val[j]? = some nd0 →
      ∃ nd', o2.nodes.val[j]? = some nd' ∧ nd'.left = nd0.left ∧ nd'.right = nd0.right := by
    intro j _ nd0 hnd0
    by_cases hjj : j = n
    · subst hjj
      rw [hnd] at hnd0
      have := Option.some.inj hnd0
      subst this
      exact ⟨_, hself, rfl, rfl⟩
    · exact ⟨nd0, by rw [hne j hjj]; exact hnd0, rfl, rfl⟩
  have hkeys : ∀ j, keyAt o2.nodes.val j = keyAt o.nodes.val j := by
    intro j
    by_cases hjj : j = n
    · subst hjj
      rw [Apeiron.MapExecuted.keyAt_get hself, Apeiron.MapExecuted.keyAt_get hnd]
    · unfold keyAt; rw [hne j hjj]
  have hhag : ∀ j, Apeiron.MapExecuted.HAg o.nodes.val o2.nodes.val j := by
    intro j nd0 hnd0
    by_cases hjj : j = n
    · subst hjj
      rw [hnd] at hnd0
      have := Option.some.inj hnd0
      subst this
      exact ⟨_, hself, rfl⟩
    · exact ⟨nd0, by rw [hne j hjj]; exact hnd0, rfl⟩
  refine ⟨?_, ?_, hND, ?_, hBal, ?_, ?_⟩
  · rw [hr]; exact Apeiron.MapExecuted.emb_ptr_congr hE hptr
  · exact Apeiron.MapExecuted.bst_congr' (fun j _ => hkeys j) hB
  · exact Apeiron.MapExecuted.hok_hcongr hH (fun j _ => hhag j)
  · rw [hf]; exact hfn
  · intro j hj
    rw [hf] at hj
    refine ⟨?_, (hfl j hj).2⟩
    rw [hn2, List.length_set]; exact (hfl j hj).1

theorem nodeok_node_update (ns ns2 : List ONode) (T : OTree) (hOK : NodeOK ns T) (n : Nat)
    (nd : ONode) (hnd : ns[n]? = some nd) (ents : alloc.vec.Vec (U64 × Option I64))
    (hn2 : ns2 = ns.set n {nd with entries := ents}) (w' : I64)
    (hents : ents.val = [(nd.key, some w')]) : NodeOK ns2 T := by
  have hlt : n < ns.length := (List.getElem?_eq_some_iff.mp hnd).1
  intro j hj
  by_cases hjj : j = n
  · subst hjj
    obtain ⟨nd0, hnd0, ha, _⟩ := hOK j hj
    rw [hnd] at hnd0
    have := Option.some.inj hnd0
    subst this
    exact ⟨{nd with entries := ents}, by rw [hn2]; exact List.getElem?_set_self hlt, ha, w', hents⟩
  · obtain ⟨nd0, hnd0, ha, w, hw⟩ := hOK j hj
    exact ⟨nd0, by rw [hn2, List.getElem?_set_ne (Ne.symm hjj)]; exact hnd0, ha, w, hw⟩

/-- `take_value` then `write_back` on the single entry of a node replaces its value. -/
theorem update_spec (o o1 o2 : Ov) (n e : U32) (old merged : I64) (nd : ONode) (w : I64)
    (hnd : o.nodes.val[n.val]? = some nd) (hent : nd.entries.val = [(nd.key, some w)])
    (he : e.val = 0)
    (htake : hashtrinity.overflow.Overflow.take_value Apeiron.MapExecuted.CI o n e = ok (old, o1))
    (hwb : hashtrinity.overflow.Overflow.write_back Apeiron.MapExecuted.CI o1 n e merged = ok o2) :
    old = w ∧ ∃ ents : alloc.vec.Vec (U64 × Option I64), ents.val = [(nd.key, some merged)] ∧
      o2.nodes.val = o.nodes.val.set n.val {nd with entries := ents} ∧ o2.root = o.root ∧
      o2.free = o.free ∧ o2.len = o.len := by
  unfold hashtrinity.overflow.Overflow.take_value at htake
  obtain ⟨i, hi, htake⟩ := bind_inv htake
  obtain ⟨⟨n1, back⟩, hn1, htake⟩ := bind_inv htake
  obtain ⟨i1, hi1, htake⟩ := bind_inv htake
  obtain ⟨⟨⟨t, ov⟩, back1⟩, hn2, htake⟩ := bind_inv htake
  have hi' : i = UScalar.cast .Usize n := (Result.ok_injective hi).symm
  have hi1' : i1 = UScalar.cast .Usize e := (Result.ok_injective hi1).symm
  subst hi' hi1'
  obtain ⟨hn1a, hback⟩ := Apeiron.MapExecuted.idxm_ok hn1
  rw [hnd] at hn1a
  have hnn : nd = n1 := Option.some.inj hn1a
  subst hnn
  obtain ⟨hn2a, hback1⟩ := Apeiron.MapExecuted.vec_index_mut_ok _ _ _ _ hn2
  rw [Apeiron.MapExecuted.cast_usize_u32, hent] at hn2a
  rw [he] at hn2a
  simp only [List.getElem?_cons_zero, Option.some.injEq, Prod.mk.injEq] at hn2a
  obtain ⟨ht, hov⟩ := hn2a
  subst ht hov
  obtain ⟨t1, hexp, htake⟩ := bind_inv htake
  have hres := Result.ok_injective htake
  simp only [Prod.mk.injEq] at hres
  obtain ⟨hold, ho1⟩ := hres
  have ht1 : t1 = w := by
    unfold core.option.Option.expect at hexp
    simpa [Result.ofOption, core.option.Option.take] using hexp.symm
  refine ⟨by rw [← hold, ht1], ?_⟩
  unfold hashtrinity.overflow.Overflow.write_back at hwb
  obtain ⟨j, hj, hwb1⟩ := bind_inv hwb
  obtain ⟨⟨n3, backw⟩, hn3, hwb2⟩ := bind_inv hwb1
  obtain ⟨j1, hj1, hwb3⟩ := bind_inv hwb2
  obtain ⟨⟨⟨t3, ov3⟩, backw1⟩, hn4, hwb4⟩ := bind_inv hwb3
  have hj' : j = UScalar.cast .Usize n := (Result.ok_injective hj).symm
  have hj1' : j1 = UScalar.cast .Usize e := (Result.ok_injective hj1).symm
  subst hj' hj1'
  have ho2 : o2 = { o1 with nodes := backw ({ n3 with entries := backw1 (t3, some merged) }) } :=
    (Result.ok_injective hwb4).symm
  -- the intermediate arena
  have ho1' : o1 = { o with nodes := back ({ nd with entries := back1 (nd.key, none) }) } := ho1.symm
  subst ho1'
  obtain ⟨hn3a, hback3⟩ := Apeiron.MapExecuted.idxm_ok hn3
  simp only at hn3a hback3
  have hlt : n.val < o.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
  rw [hback, List.getElem?_set_self hlt] at hn3a
  have hn3e : n3 = { nd with entries := back1 (nd.key, none) } := (Option.some.inj hn3a).symm
  subst hn3e
  obtain ⟨hn4a, hback4⟩ := Apeiron.MapExecuted.vec_index_mut_ok _ _ _ _ hn4
  rw [Apeiron.MapExecuted.cast_usize_u32, he] at hn4a
  have hb1val : (back1 (nd.key, none)).val = [(nd.key, none)] := by
    rw [hback1, Apeiron.MapExecuted.vec_set_val, hent, Apeiron.MapExecuted.cast_usize_u32, he]; rfl
  simp only [hb1val, List.getElem?_cons_zero, Option.some.injEq, Prod.mk.injEq] at hn4a
  obtain ⟨ht3, hov3⟩ := hn4a
  subst ht3
  refine ⟨backw1 (nd.key, some merged), ?_, ?_, by rw [ho2], by rw [ho2], by rw [ho2]⟩
  · rw [hback4, Apeiron.MapExecuted.vec_set_val, hb1val, Apeiron.MapExecuted.cast_usize_u32, he]; rfl
  · rw [ho2]
    show (backw _).val = _
    rw [hback3, hback, List.set_set]

/-- No tree node holds a coordinate absent from the tree. -/
theorem keys_absent_of (o : Ov) (T : OTree) (hOK : NodeOK o.nodes.val T) (f : U64)
    (hnp : ¬ Present o.nodes.val T f f) :
    ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.key ≠ f := by
  intro j hj nd hnd hk
  obtain ⟨nd0, hnd0, _, w, hw⟩ := hOK j hj
  have e1 : nd0 = nd := by rw [hnd] at hnd0; exact (Option.some.inj hnd0).symm
  rw [e1] at hw
  apply hnp
  refine ⟨j, hj, nd, hnd, hk, 0, (nd.key, some w), ?_, hk⟩
  rw [hw]; rfl

theorem keys_absent (o : Ov) (T : OTree) (hI : OInv o T) (hOK : NodeOK o.nodes.val T) (f : U64)
    (hfe : hashtrinity.overflow.Overflow.find_entry Apeiron.MapExecuted.CI core.cmp.EqU64 o f f = ok none) :
    ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.key ≠ f :=
  keys_absent_of o T hOK f ((Apeiron.MapExecuted.find_entry_none_iff o T hI.1 hI.2.1 f f none hfe).mp rfl)

/-- **`insert_new_at` of a fresh frequency.**  The store stays an AVL tree of single-entry nodes; the
old nodes keep their coordinate and entry, and one new node holds `(f, some v)`. -/
theorem insert_ovok (o o' : Ov) (T : OTree) (f : U64) (v : I64) (nd : U32) (e : Usize)
    (hI : OInv o T) (hOK : NodeOK o.nodes.val T) (hcap : o.nodes.val.length < nilN)
    (hnp : ¬ Present o.nodes.val T f f)
    (h : hashtrinity.overflow.Overflow.insert_new_at Apeiron.MapExecuted.CI o f f v = ok ((nd, e), o')) :
    ∃ T', OInv o' T' ∧ NodeOK o'.nodes.val T' ∧ o'.nodes.val.length ≤ o.nodes.val.length + 1 ∧
      (∀ j ∈ idxs T', ∃ nd', o'.nodes.val[j]? = some nd' ∧
        ((∃ nd0, j ∈ idxs T ∧ o.nodes.val[j]? = some nd0 ∧ nd0.key = nd'.key ∧
            nd0.entries.val = nd'.entries.val) ∨
         (nd'.key = f ∧ nd'.entries.val = [(f, some v)]))) ∧
      (∀ j ∈ idxs T, ∃ j' ∈ idxs T', ∃ nd0 nd', o.nodes.val[j]? = some nd0 ∧
        o'.nodes.val[j']? = some nd' ∧ nd'.key = nd0.key ∧ nd'.entries.val = nd0.entries.val) ∧
      (∃ j ∈ idxs T', ∃ nd', o'.nodes.val[j]? = some nd' ∧ nd'.key = f ∧
        nd'.entries.val = [(f, some v)]) := by
  obtain ⟨T', hI', -, hlen, hndT', ⟨nde, hnde, hkeyde, hfirst, hentde⟩, hiff, -, hC⟩ :=
    Apeiron.MapExecuted.insert_new_at_spec o o' T f f v nd e hI hcap hnp h
  have habs := keys_absent_of o T hOK f hnp
  rcases hC with ⟨nd0, hmT, hnd0, hk0⟩ | ⟨hnT, hC⟩
  · exact absurd hk0 (habs _ hmT nd0 hnd0)
  have hOK' : NodeOK o'.nodes.val T' := by
    intro j hj
    obtain ⟨nd', hnd', hcase⟩ := hC j hj
    rcases hcase with ⟨hjT, nd0, hnd0, hkk, hal, hent⟩ | ⟨_, hk, hal, hent⟩
    · obtain ⟨nd1, hnd1, ha1, w, hw⟩ := hOK j hjT
      have e1 : nd1 = nd0 := by rw [hnd0] at hnd1; exact (Option.some.inj hnd1).symm
      rw [e1] at ha1 hw
      exact ⟨nd', hnd', by rw [hal]; exact ha1, w, by rw [hent, hw, hkk]⟩
    · exact ⟨nd', hnd', hal, v, by rw [hent, hk]⟩
  refine ⟨T', hI', hOK', hlen, ?_, ?_, ?_⟩
  · intro j hj
    obtain ⟨nd', hnd', hcase⟩ := hC j hj
    refine ⟨nd', hnd', ?_⟩
    rcases hcase with ⟨hjT, nd0, hnd0, hkk, hal, hent⟩ | ⟨_, hk, hal, hent⟩
    · exact Or.inl ⟨nd0, hjT, hnd0, hkk.symm, hent.symm⟩
    · exact Or.inr ⟨hk, hent⟩
  · intro j hj
    obtain ⟨nd0, hnd0, _, w, hw⟩ := hOK j hj
    have hent : EntryOf o.nodes.val T nd0.key nd0.key (some w) :=
      ⟨j, hj, nd0, hnd0, rfl, by rw [hw]; exact List.mem_singleton_self _⟩
    obtain ⟨j', hj', nd', hnd', hkk, hmem⟩ := (hiff nd0.key nd0.key (some w)).mpr (Or.inl hent)
    obtain ⟨nd1, hnd1, _, w1, hw1⟩ := hOK' j' hj'
    have e1 : nd1 = nd' := by rw [hnd'] at hnd1; exact (Option.some.inj hnd1).symm
    rw [e1] at hw1
    refine ⟨j', hj', nd0, nd', hnd0, hnd', hkk, ?_⟩
    rw [hw1] at hmem
    simp only [List.mem_singleton, Prod.mk.injEq, Option.some.injEq] at hmem
    rw [hw1, hw, hkk, hmem.2]
  · obtain ⟨nd1, hnd1, _, w1, hw1⟩ := hOK' nd.val hndT'
    have e1 : nd1 = nde := by rw [hnde] at hnd1; exact (Option.some.inj hnd1).symm
    rw [e1] at hw1
    refine ⟨nd.val, hndT', nde, hnde, hkeyde, ?_⟩
    have h0 : nde.entries.val[e.val]? = some (f, some v) := hentde
    rw [hw1] at h0 ⊢
    have he0 : e.val = 0 := by
      by_contra hne
      rw [List.getElem?_eq_none (by simp; omega)] at h0
      exact absurd h0 (by simp)
    rw [he0] at h0
    simp only [List.getElem?_cons_zero, Option.some.injEq, Prod.mk.injEq] at h0
    rw [hkeyde, h0.2]

end Apeiron.CountAggregate
