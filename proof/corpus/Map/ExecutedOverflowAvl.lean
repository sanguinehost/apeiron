-- The executed overflow tree's AVL write path.
import Map.ExecutedOverflowAvlBase
open Aeneas Aeneas.Std Result

/-!
# The executed overflow tree: `rebalance`

`rebalance_spec`: on a node `i` whose two subtrees are AVL (`Bal`, `HOK`, `BST`) and differ in height
by at most two, the extracted `rebalance` returns the root of an AVL tree over the same node indices
in the same in-order sequence, the stored heights are current, and its height is
`max (ht l) (ht r)` or one more.  Partial correctness: a failing call is excluded by the hypothesis.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-! ## Pure shape facts -/

theorem Rel.trans' {U S S' : List Nat} {ns ns' ns'' : List ONode} (h1 : Rel S ns ns')
    (h2 : Rel S' ns' ns'') (hS : ∀ j ∈ S, j ∈ U) (hS' : ∀ j ∈ S', j ∈ U) : Rel U ns ns'' :=
  Rel.mono (fun j hj => by
    rcases List.mem_append.mp hj with hj | hj
    · exact hS j hj
    · exact hS' j hj) (h1.trans h2)

theorem nd_repl_l {l l' r : OTree} {i : Nat} (h : ND (.node l i r)) (hl : ND l')
    (e : idxs l' = idxs l) : ND (.node l' i r) := by
  obtain ⟨_, hNr, hil, hir, hlr⟩ := h
  exact ⟨hl, hNr, by rw [e]; exact hil, hir, by rw [e]; exact hlr⟩

theorem nd_repl_r {l r r' : OTree} {i : Nat} (h : ND (.node l i r)) (hr : ND r')
    (e : idxs r' = idxs r) : ND (.node l i r') := by
  obtain ⟨hNl, _, hil, hir, hlr⟩ := h
  exact ⟨hNl, hr, hil, by rw [e]; exact hir, fun j hj => by rw [e]; exact hlr j hj⟩

theorem bst_repl_l {ns : List ONode} {l l' r : OTree} {i : Nat} (h : BST ns (.node l i r))
    (hl : BST ns l') (e : idxs l' = idxs l) : BST ns (.node l' i r) := by
  obtain ⟨_, hBr, hlt, hgt⟩ := h
  exact ⟨hl, hBr, by rw [e]; exact hlt, hgt⟩

theorem bst_repl_r {ns : List ONode} {l r r' : OTree} {i : Nat} (h : BST ns (.node l i r))
    (hr : BST ns r') (e : idxs r' = idxs r) : BST ns (.node l i r') := by
  obtain ⟨hBl, _, hlt, hgt⟩ := h
  exact ⟨hBl, hr, hlt, by rw [e]; exact hgt⟩

theorem mem_idxs_node_l {j : Nat} {a b : OTree} {x : Nat} (h : j ∈ idxs a) :
    j ∈ idxs (.node a x b) := by
  simp only [idxs, List.mem_append]; exact Or.inl h

theorem mem_idxs_node_r {j : Nat} {a b : OTree} {x : Nat} (h : j ∈ idxs b) :
    j ∈ idxs (.node a x b) := by
  simp only [idxs, List.mem_append, List.mem_cons]; exact Or.inr (Or.inr h)

theorem idxs_repl_l {l l' r : OTree} {i : Nat} (e : idxs l' = idxs l) :
    idxs (.node l' i r) = idxs (.node l i r) := by
  simp [idxs, e]

theorem idxs_repl_r {l r r' : OTree} {i : Nat} (e : idxs r' = idxs r) :
    idxs (.node l i r') = idxs (.node l i r) := by
  simp [idxs, e]

/-! ## Height arithmetic of the four cases -/

theorem avl_LL {ll lr r : OTree} {li ii : Nat} (hBl : Bal (.node ll li lr)) (hBr : Bal r)
    (hh : ht (.node ll li lr) = ht r + 2) (hc : ht lr ≤ ht ll) :
    Bal (.node ll li (.node lr ii r)) ∧ ht r + 2 ≤ ht (.node ll li (.node lr ii r)) ∧
      ht (.node ll li (.node lr ii r)) ≤ ht r + 3 := by
  obtain ⟨hBll, hBlr, h1, h2⟩ := hBl
  refine ⟨⟨hBll, ⟨hBlr, hBr, ?_, ?_⟩, ?_, ?_⟩, ?_, ?_⟩ <;> (try simp only [ht] at hh h1 h2 ⊢) <;> omega

theorem avl_LR {ll lrl lrr r : OTree} {li lri ii : Nat}
    (hBl : Bal (.node ll li (.node lrl lri lrr))) (hBr : Bal r)
    (hh : ht (.node ll li (.node lrl lri lrr)) = ht r + 2)
    (hc : ht ll < ht (.node lrl lri lrr)) :
    Bal (.node (.node ll li lrl) lri (.node lrr ii r)) ∧
      ht (.node (.node ll li lrl) lri (.node lrr ii r)) = ht r + 2 := by
  obtain ⟨hBll, ⟨hBlrl, hBlrr, h3, h4⟩, h1, h2⟩ := hBl
  refine ⟨⟨⟨hBll, hBlrl, ?_, ?_⟩, ⟨hBlrr, hBr, ?_, ?_⟩, ?_, ?_⟩, ?_⟩ <;>
    (try simp only [ht] at hh h1 h2 h3 h4 hc ⊢) <;> omega

theorem avl_RR {l rl rr : OTree} {ri ii : Nat} (hBr : Bal (.node rl ri rr)) (hBl : Bal l)
    (hh : ht (.node rl ri rr) = ht l + 2) (hc : ht rl ≤ ht rr) :
    Bal (.node (.node l ii rl) ri rr) ∧ ht l + 2 ≤ ht (.node (.node l ii rl) ri rr) ∧
      ht (.node (.node l ii rl) ri rr) ≤ ht l + 3 := by
  obtain ⟨hBrl, hBrr, h1, h2⟩ := hBr
  refine ⟨⟨⟨hBl, hBrl, ?_, ?_⟩, hBrr, ?_, ?_⟩, ?_, ?_⟩ <;> (try simp only [ht] at hh h1 h2 ⊢) <;> omega

theorem avl_RL {l rll rlr rr : OTree} {rli ri ii : Nat}
    (hBr : Bal (.node (.node rll rli rlr) ri rr)) (hBl : Bal l)
    (hh : ht (.node (.node rll rli rlr) ri rr) = ht l + 2)
    (hc : ht rr < ht (.node rll rli rlr)) :
    Bal (.node (.node l ii rll) rli (.node rlr ri rr)) ∧
      ht (.node (.node l ii rll) rli (.node rlr ri rr)) = ht l + 2 := by
  obtain ⟨⟨hBrll, hBrlr, h3, h4⟩, hBrr, h1, h2⟩ := hBr
  refine ⟨⟨⟨hBl, hBrll, ?_, ?_⟩, ⟨hBrlr, hBrr, ?_, ?_⟩, ?_, ?_⟩, ?_⟩ <;>
    (try simp only [ht] at hh h1 h2 h3 h4 hc ⊢) <;> omega

/-! ## `rebalance` -/

theorem rebalance_spec (o o' : Ov) (L : List ONode) (hL : o.nodes.val = L) (i x' : U32)
    (l r : OTree) (ii : Nat)
    (hE : Emb L i (.node l ii r)) (hND : ND (.node l ii r)) (hB : BST L (.node l ii r))
    (hBl : Bal l) (hBr : Bal r) (hHl : HOK L l) (hHr : HOK L r)
    (hlr : ht l ≤ ht r + 2 ∧ ht r ≤ ht l + 2)
    (h : hashtrinity.overflow.Overflow.rebalance CI o i = ok (x', o')) :
    ∃ T' A, o'.nodes.val = A ∧ Emb A x' T' ∧ idxs T' = idxs (.node l ii r) ∧ ND T' ∧
      BST A T' ∧ HOK A T' ∧ Bal T' ∧ Rel (idxs (.node l ii r)) L A ∧
      o'.free = o.free ∧ o'.root = o.root ∧ o'.len = o.len ∧
      max (ht l) (ht r) ≤ ht T' ∧ ht T' ≤ max (ht l) (ht r) + 1 ∧
      (ht l ≤ ht r + 1 → ht r ≤ ht l + 1 → ht T' = max (ht l) (ht r) + 1) := by
  have hE0 := hE
  have hND0 := hND
  obtain ⟨hiv, hii, nd, hnd, hEl, hEr⟩ := hE0
  obtain ⟨hNDl, hNDr, hiil, hiir, hlr'⟩ := hND0
  unfold hashtrinity.overflow.Overflow.rebalance at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨o1, hu1, h⟩ := bind_inv h
  obtain ⟨nd1, hh, hnd1, hhv, hnodes1, hf1, hr1, hl1⟩ :=
    update_spec o o1 L hL i l r ii hE hHl hHr hu1
  have hndeq : nd = nd1 := Option.some.inj (hnd.symm.trans hnd1)
  subst hndeq
  obtain ⟨L1, hL1def⟩ : ∃ L1, L1 = L.set ii {nd with height := hh} := ⟨_, rfl⟩
  have hL1 : o1.nodes.val = L1 := by rw [hnodes1, hL1def]
  have hiiL : ii < L.length := (List.getElem?_eq_some_iff.mp hnd).1
  have hL1ii : L1[ii]? = some {nd with height := hh} := by
    rw [hL1def]; exact List.getElem?_set_self hiiL
  have hL1o : ∀ j, j ≠ ii → L1[j]? = L[j]? := by
    intro j hj; rw [hL1def, List.getElem?_set_ne (Ne.symm hj)]
  have hrel1 : Rel [ii] L L1 := by rw [hL1def]; exact Rel_set hnd ⟨rfl, rfl, rfl⟩
  have hnel : ∀ j ∈ idxs l, j ≠ ii := fun j hj e => hiil (e ▸ hj)
  have hner : ∀ j ∈ idxs r, j ≠ ii := fun j hj e => hiir (e ▸ hj)
  have hEl1 : Emb L1 nd.left l := emb_congr hEl (fun j hj => hL1o j (hnel j hj))
  have hEr1 : Emb L1 nd.right r := emb_congr hEr (fun j hj => hL1o j (hner j hj))
  have hHl1 : HOK L1 l := hok_congr hHl (fun j hj => hL1o j (hnel j hj))
  have hHr1 : HOK L1 r := hok_congr hHr (fun j hj => hL1o j (hner j hj))
  have hEi1 : Emb L1 i (.node l ii r) := ⟨hiv, hii, {nd with height := hh}, hL1ii, hEl1, hEr1⟩
  have hB1 : BST L1 (.node l ii r) := bst_rel hrel1 hB
  have hH1 : HOK L1 (.node l ii r) := ⟨hHl1, hHr1, _, hL1ii, hhv⟩
  have hB1l : BST L1 l := hB1.1
  have hB1r : BST L1 r := hB1.2.1
  have hiiT : ii ∈ idxs (.node l ii r) := by simp [idxs]
  have hmemL : ∀ j ∈ idxs l, j ∈ idxs (.node l ii r) := fun j hj => by simp [idxs, hj]
  have hmemR : ∀ j ∈ idxs r, j ∈ idxs (.node l ii r) := fun j hj => by simp [idxs, hj]
  have hmemI : ∀ j ∈ [ii], j ∈ idxs (.node l ii r) := fun j hj => by
    rw [List.mem_singleton] at hj; rw [hj]; exact hiiT
  obtain ⟨n, hn, h⟩ := bind_inv h
  have hn' := idx_ok hn
  rw [hL1, hiv, hL1ii] at hn'
  have hnn : ({nd with height := hh} : ONode) = n := Option.some.inj hn'
  have hnL : n.left = nd.left := by rw [← hnn]
  have hnR : n.right = nd.right := by rw [← hnn]
  obtain ⟨n1, hn1, h⟩ := bind_inv h
  have hn1' := idx_ok hn1
  rw [hL1, hiv, hL1ii] at hn1'
  have hnn1 : ({nd with height := hh} : ONode) = n1 := Option.some.inj hn1'
  have hn1L : n1.left = nd.left := by rw [← hnn1]
  have hn1R : n1.right = nd.right := by rw [← hnn1]
  obtain ⟨hl, hhl, h⟩ := bind_inv h
  have hvl := h_spec o1 L1 hL1 _ l (by rw [hnL]; exact hEl1) hHl1 hl hhl
  obtain ⟨hr, hhr, h⟩ := bind_inv h
  have hvr := h_spec o1 L1 hL1 _ r (by rw [hn1R]; exact hEr1) hHr1 hr hhr
  obtain ⟨i3, hi3, h⟩ := bind_inv h
  have hi3v := uadd_ok hi3
  rw [u32_one] at hi3v
  by_cases hA : hl > i3
  · -- left-heavy
    rw [if_pos hA] at h
    have hA' : ht r + 1 < ht l := by
      have := (UScalar.lt_equiv i3 hl).mp hA
      omega
    cases l with
    | leaf => simp [ht] at hA'
    | node ll li lr =>
      have hEl1c := hEl1
      obtain ⟨hlv, hli, nl, hnl, hEll, hElr⟩ := hEl1
      obtain ⟨hHll, hHlr, nlh, hnlh, hhlh⟩ := hHl1
      obtain ⟨n2, hn2, h⟩ := bind_inv h
      have hn2' := idx_ok hn2
      rw [hL1, hnL, hlv, hnl] at hn2'
      have hnn2 : nl = n2 := Option.some.inj hn2'
      subst hnn2
      obtain ⟨i5, hi5, h⟩ := bind_inv h
      have hv5 := h_spec o1 L1 hL1 _ lr hElr hHlr i5 hi5
      obtain ⟨n3, hn3, h⟩ := bind_inv h
      have hn3' := idx_ok hn3
      rw [hL1, hnL, hlv, hnl] at hn3'
      have hnn3 : nl = n3 := Option.some.inj hn3'
      subst hnn3
      obtain ⟨i7, hi7, h⟩ := bind_inv h
      have hv7 := h_spec o1 L1 hL1 _ ll hEll hHll i7 hi7
      obtain ⟨⟨v, v1, i8, i9⟩, hcond, h⟩ := bind_inv h
      have hhh : ht (.node ll li lr) = ht r + 2 := by omega
      by_cases hrot : i5 > i7
      · -- left-right double rotation
        rw [if_pos hrot] at hcond
        have hrot' : ht ll < ht lr := by
          have := (UScalar.lt_equiv i7 i5).mp hrot
          omega
        cases lr with
        | leaf => simp [ht] at hrot'
        | node lrl lri lrr =>
          obtain ⟨hHlrl, hHlrr, nlrh, hnlrh, hhlrh⟩ := hHlr
          obtain ⟨⟨nl2, o2⟩, hrl, hcond⟩ := bind_inv hcond
          obtain ⟨⟨n4, back⟩, hn4, hcond⟩ := bind_inv hcond
          obtain ⟨A2, hA2, hnl2v, hEA2, hRel2, hB2, hH2, hf2, hr2, hl2⟩ :=
            rotate_left_spec o1 o2 L1 hL1 n.left nl2 ll lrl lrr li lri
              (by rw [hnL]; exact hEl1c) hNDl hB1l hHll hHlrl hHlrr hrl
          obtain ⟨hn4', hback⟩ := idxm_ok hn4
          rw [hA2, hiv] at hn4'
          have hA2ii : A2[ii]? = some {nd with height := hh} := by
            rw [rel_out hRel2 hiil]; exact hL1ii
          have hn4eq : ({nd with height := hh} : ONode) = n4 :=
            Option.some.inj (hA2ii.symm.trans hn4')
          subst hn4eq
          have hc := Result.ok_injective hcond
          simp only [Prod.mk.injEq] at hc
          obtain ⟨hv, hv1, hi8, hi9⟩ := hc
          subst hv hv1 hi8 hi9
          obtain ⟨A3, hA3def⟩ : ∃ A3, A3 =
              A2.set ii {({nd with height := hh} : ONode) with left := nl2} := ⟨_, rfl⟩
          have hA2lt : ii < A2.length := (List.getElem?_eq_some_iff.mp hA2ii).1
          have hA3ii : A3[ii]? = some {({nd with height := hh} : ONode) with left := nl2} := by
            rw [hA3def]; exact List.getElem?_set_self hA2lt
          have hA3o : ∀ j, j ≠ ii → A3[j]? = A2[j]? := by
            intro j hj; rw [hA3def, List.getElem?_set_ne (Ne.symm hj)]
          have hrelA3 : Rel [ii] A2 A3 := by
            rw [hA3def]; exact Rel_set hA2ii ⟨rfl, rfl, rfl⟩
          have hk3 : ∀ j, keyAt A3 j = keyAt L1 j :=
            fun j => (keyAt_rel hrelA3 j).trans (keyAt_rel hRel2 j)
          have hne : ∀ j ∈ idxs (.node (.node ll li lrl) lri lrr), j ≠ ii := by
            intro j hj e
            rw [idxs_rotL] at hj
            exact hiil (e ▸ hj)
          have hEr3 : Emb A3 nd.right r :=
            emb_congr (emb_rel hRel2 hEr1 (fun j hj hS => hlr' j hS hj))
              (fun j hj => hA3o j (hner j hj))
          obtain ⟨hH2a, hH2b, _⟩ := hH2
          have hE3 : Emb A3 i (.node (.node (.node ll li lrl) lri lrr) ii r) :=
            ⟨hiv, hii, {({nd with height := hh} : ONode) with left := nl2}, hA3ii,
              emb_congr hEA2 (fun j hj => hA3o j (hne j hj)), hEr3⟩
          have hND3 : ND (.node (.node (.node ll li lrl) lri lrr) ii r) :=
            nd_repl_l hND (nd_rotL hNDl) idxs_rotL
          have hB3 : BST A3 (.node (.node (.node ll li lrl) lri lrr) ii r) :=
            bst_congr hk3 (bst_repl_l hB1 (bst_rotL hB1l) idxs_rotL)
          have hHa3 : HOK A3 (.node ll li lrl) :=
            hok_congr hH2a (fun j hj => hA3o j (hne j (mem_idxs_node_l hj)))
          have hHb3 : HOK A3 lrr :=
            hok_congr hH2b (fun j hj => hA3o j (hne j (by simp [idxs, hj])))
          have hHc3 : HOK A3 r :=
            hok_congr (hok_rel hRel2 hHr1 (fun j hj hS => hlr' j hS hj))
              (fun j hj => hA3o j (hner j hj))
          obtain ⟨A, hA, hxv, hEA, hRelA, hBA, hHA, hfA, hrA, hlA⟩ :=
            rotate_right_spec _ o' A3 (by dsimp only; rw [hback, hA2, hiv, hA3def]) i x'
              (.node ll li lrl) lrr r lri ii hE3 hND3 hB3 hHa3 hHb3 hHc3 h
          have hbal := avl_LR (ii := ii) hBl hBr hhh hrot'
          have hrel : Rel (idxs (.node (.node ll li (.node lrl lri lrr)) ii r)) L A := by
            refine Rel.trans' (Rel.trans' (Rel.trans' hrel1 hRel2 hmemI hmemL) hrelA3
              (fun j hj => hj) hmemI) hRelA (fun j hj => hj) ?_
            intro j hj
            rw [idxs_repl_l idxs_rotL] at hj
            exact hj
          refine ⟨.node (.node ll li lrl) lri (.node lrr ii r), A, hA, hEA,
            idxs_rotR.trans (idxs_repl_l idxs_rotL), nd_rotR hND3, hBA, hHA, hbal.1, hrel,
            hfA.trans (hf2.trans hf1), hrA.trans (hr2.trans hr1), hlA.trans (hl2.trans hl1),
            ?_, ?_, fun h1 _ => by omega⟩
          · omega
          · omega
      · -- left-left single rotation
        rw [if_neg hrot] at hcond
        have hc := Result.ok_injective hcond
        simp only [Prod.mk.injEq] at hc
        obtain ⟨hv, hv1, hi8, hi9⟩ := hc
        subst hv hv1 hi8 hi9
        have hrot' : ht lr ≤ ht ll := by
          have := (not_congr (UScalar.lt_equiv i7 i5)).mp hrot
          omega
        obtain ⟨A, hA, hxv, hEA, hRelA, hBA, hHA, hfA, hrA, hlA⟩ :=
          rotate_right_spec _ o' L1 hL1 i x' ll lr r li ii hEi1 hND hB1 hHll hHlr hHr1 h
        have hbal := avl_LL (ii := ii) hBl hBr hhh hrot'
        refine ⟨.node ll li (.node lr ii r), A, hA, hEA, idxs_rotR, nd_rotR hND, hBA, hHA, hbal.1,
          Rel.trans' hrel1 hRelA hmemI (fun j hj => hj), hfA.trans hf1, hrA.trans hr1,
          hlA.trans hl1, ?_, ?_, fun h1 _ => by omega⟩
        · omega
        · omega
  · rw [if_neg hA] at h
    have hA' : ht l ≤ ht r + 1 := by
      have := (not_congr (UScalar.lt_equiv i3 hl)).mp hA
      omega
    obtain ⟨i4, hi4, h⟩ := bind_inv h
    have hi4v := uadd_ok hi4
    rw [u32_one] at hi4v
    by_cases hB' : hr > i4
    · -- right-heavy
      rw [if_pos hB'] at h
      have hB'' : ht l + 1 < ht r := by
        have := (UScalar.lt_equiv i4 hr).mp hB'
        omega
      cases r with
      | leaf => simp [ht] at hB''
      | node rl ri rr =>
        have hEr1c := hEr1
        obtain ⟨hrv, hri, nr, hnr, hErl, hErr⟩ := hEr1
        obtain ⟨hHrl, hHrr, nrh, hnrh, hhrh⟩ := hHr1
        obtain ⟨n2, hn2, h⟩ := bind_inv h
        have hn2' := idx_ok hn2
        rw [hL1, hn1R, hrv, hnr] at hn2'
        have hnn2 : nr = n2 := Option.some.inj hn2'
        subst hnn2
        obtain ⟨i6, hi6, h⟩ := bind_inv h
        have hv6 := h_spec o1 L1 hL1 _ rl hErl hHrl i6 hi6
        obtain ⟨n3, hn3, h⟩ := bind_inv h
        have hn3' := idx_ok hn3
        rw [hL1, hn1R, hrv, hnr] at hn3'
        have hnn3 : nr = n3 := Option.some.inj hn3'
        subst hnn3
        obtain ⟨i8, hi8, h⟩ := bind_inv h
        have hv8 := h_spec o1 L1 hL1 _ rr hErr hHrr i8 hi8
        obtain ⟨⟨v, v1, i9, i10⟩, hcond, h⟩ := bind_inv h
        have hhh : ht (.node rl ri rr) = ht l + 2 := by omega
        by_cases hrot : i6 > i8
        · -- right-left double rotation
          rw [if_pos hrot] at hcond
          have hrot' : ht rr < ht rl := by
            have := (UScalar.lt_equiv i8 i6).mp hrot
            omega
          cases rl with
          | leaf => simp [ht] at hrot'
          | node rll rli rlr =>
            obtain ⟨hHrll, hHrlr, nrlh, hnrlh, hhrlh⟩ := hHrl
            obtain ⟨⟨nr2, o2⟩, hrr, hcond⟩ := bind_inv hcond
            obtain ⟨⟨n4, back⟩, hn4, hcond⟩ := bind_inv hcond
            obtain ⟨A2, hA2, hnr2v, hEA2, hRel2, hB2, hH2, hf2, hr2, hl2⟩ :=
              rotate_right_spec o1 o2 L1 hL1 n1.right nr2 rll rlr rr rli ri
                (by rw [hn1R]; exact hEr1c) hNDr hB1r hHrll hHrlr hHrr hrr
            obtain ⟨hn4', hback⟩ := idxm_ok hn4
            rw [hA2, hiv] at hn4'
            have hA2ii : A2[ii]? = some {nd with height := hh} := by
              rw [rel_out hRel2 hiir]; exact hL1ii
            have hn4eq : ({nd with height := hh} : ONode) = n4 :=
              Option.some.inj (hA2ii.symm.trans hn4')
            subst hn4eq
            have hc := Result.ok_injective hcond
            simp only [Prod.mk.injEq] at hc
            obtain ⟨hv, hv1, hi9, hi10⟩ := hc
            subst hv hv1 hi9 hi10
            obtain ⟨A3, hA3def⟩ : ∃ A3, A3 =
                A2.set ii {({nd with height := hh} : ONode) with right := nr2} := ⟨_, rfl⟩
            have hA2lt : ii < A2.length := (List.getElem?_eq_some_iff.mp hA2ii).1
            have hA3ii : A3[ii]? = some {({nd with height := hh} : ONode) with right := nr2} := by
              rw [hA3def]; exact List.getElem?_set_self hA2lt
            have hA3o : ∀ j, j ≠ ii → A3[j]? = A2[j]? := by
              intro j hj; rw [hA3def, List.getElem?_set_ne (Ne.symm hj)]
            have hrelA3 : Rel [ii] A2 A3 := by
              rw [hA3def]; exact Rel_set hA2ii ⟨rfl, rfl, rfl⟩
            have hk3 : ∀ j, keyAt A3 j = keyAt L1 j :=
              fun j => (keyAt_rel hrelA3 j).trans (keyAt_rel hRel2 j)
            have hne : ∀ j ∈ idxs (.node rll rli (.node rlr ri rr)), j ≠ ii := by
              intro j hj e
              rw [idxs_rotR] at hj
              exact hiir (e ▸ hj)
            have hEl3 : Emb A3 nd.left l :=
              emb_congr (emb_rel hRel2 hEl1 (fun j hj hS => hlr' j hj hS))
                (fun j hj => hA3o j (hnel j hj))
            obtain ⟨hH2a, hH2b, _⟩ := hH2
            have hE3 : Emb A3 i (.node l ii (.node rll rli (.node rlr ri rr))) :=
              ⟨hiv, hii, {({nd with height := hh} : ONode) with right := nr2}, hA3ii, hEl3,
                emb_congr hEA2 (fun j hj => hA3o j (hne j hj))⟩
            have hND3 : ND (.node l ii (.node rll rli (.node rlr ri rr))) :=
              nd_repl_r hND (nd_rotR hNDr) idxs_rotR
            have hB3 : BST A3 (.node l ii (.node rll rli (.node rlr ri rr))) :=
              bst_congr hk3 (bst_repl_r hB1 (bst_rotR hB1r) idxs_rotR)
            have hHl3 : HOK A3 l :=
              hok_congr (hok_rel hRel2 hHl1 (fun j hj hS => hlr' j hj hS))
                (fun j hj => hA3o j (hnel j hj))
            have hHa3 : HOK A3 rll :=
              hok_congr hH2a (fun j hj => hA3o j (hne j (by simp [idxs, hj])))
            have hHb3 : HOK A3 (.node rlr ri rr) :=
              hok_congr hH2b (fun j hj => hA3o j (hne j (mem_idxs_node_r hj)))
            obtain ⟨A, hA, hxv, hEA, hRelA, hBA, hHA, hfA, hrA, hlA⟩ :=
              rotate_left_spec _ o' A3 (by dsimp only; rw [hback, hA2, hiv, hA3def]) i x'
                l rll (.node rlr ri rr) ii rli hE3 hND3 hB3 hHl3 hHa3 hHb3 h
            have hbal := avl_RL (ii := ii) hBr hBl hhh hrot'
            have hrel : Rel (idxs (.node l ii (.node (.node rll rli rlr) ri rr))) L A := by
              refine Rel.trans' (Rel.trans' (Rel.trans' hrel1 hRel2 hmemI hmemR) hrelA3
                (fun j hj => hj) hmemI) hRelA (fun j hj => hj) ?_
              intro j hj
              rw [idxs_repl_r idxs_rotR] at hj
              exact hj
            refine ⟨.node (.node l ii rll) rli (.node rlr ri rr), A, hA, hEA,
              idxs_rotL.trans (idxs_repl_r idxs_rotR), nd_rotL hND3, hBA, hHA, hbal.1, hrel,
              hfA.trans (hf2.trans hf1), hrA.trans (hr2.trans hr1), hlA.trans (hl2.trans hl1),
              ?_, ?_, fun h1 _ => by omega⟩
            · omega
            · omega
        · -- right-right single rotation
          rw [if_neg hrot] at hcond
          have hc := Result.ok_injective hcond
          simp only [Prod.mk.injEq] at hc
          obtain ⟨hv, hv1, hi9, hi10⟩ := hc
          subst hv hv1 hi9 hi10
          have hrot' : ht rl ≤ ht rr := by
            have := (not_congr (UScalar.lt_equiv i8 i6)).mp hrot
            omega
          obtain ⟨A, hA, hxv, hEA, hRelA, hBA, hHA, hfA, hrA, hlA⟩ :=
            rotate_left_spec _ o' L1 hL1 i x' l rl rr ii ri hEi1 hND hB1 hHl1 hHrl hHrr h
          have hbal := avl_RR (ii := ii) hBr hBl hhh hrot'
          refine ⟨.node (.node l ii rl) ri rr, A, hA, hEA, idxs_rotL, nd_rotL hND, hBA, hHA, hbal.1,
            Rel.trans' hrel1 hRelA hmemI (fun j hj => hj), hfA.trans hf1, hrA.trans hr1,
            hlA.trans hl1, ?_, ?_, fun h1 _ => by omega⟩
          · omega
          · omega
    · -- balanced
      rw [if_neg hB'] at h
      have hc := Result.ok_injective h
      simp only [Prod.mk.injEq] at hc
      obtain ⟨hx, ho⟩ := hc
      subst hx
      subst ho
      have hB'' : ht r ≤ ht l + 1 := by
        have := (not_congr (UScalar.lt_equiv i4 hr)).mp hB'
        omega
      refine ⟨.node l ii r, L1, hL1, hEi1, rfl, hND, hB1, hH1, ⟨hBl, hBr, hA', hB''⟩,
        Rel.mono hmemI hrel1, hf1, hr1, hl1, ?_, ?_, fun _ _ => rfl⟩
      · simp only [ht]; omega
      · simp only [ht]; omega

end Apeiron.MapExecuted
