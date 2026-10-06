-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the executed overflow tree's insert path.
import Map.ExecutedOverflowAvl
open Aeneas Aeneas.Std Result

/-!
# The executed overflow tree: path contexts

A subtree sits in a *slot* below a path of ancestors.  `Fr` is one ancestor: its node index `a`, the
side `d` the slot is on (`true`: left) and the other subtree `o`.  The path `G` is listed
innermost first, and `plugI X G` is the tree obtained by hanging `X` in the slot.

* `Spine ns r s G`: in the arena `ns` the root of the whole tree is `r`, the ancestors of `G` are
  linked through their child pointers down to the slot, whose root is `s`.
* `AncOld ns G sc`: the old (pre-insert) heights of the ancestors are consistent with an old slot
  height `sc`, so that `rebalance` along the path sees an imbalance of at most two.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

structure Fr where
  a : Nat
  d : Bool
  o : OTree

/-- The ancestor `f` around a subtree `X` in its slot. -/
def frT (f : Fr) (X : OTree) : OTree := if f.d then .node X f.a f.o else .node f.o f.a X

/-- Hang `X` in the slot below the path `G`, innermost ancestor first. -/
def plugI : OTree → List Fr → OTree
  | X, [] => X
  | X, f :: rest => plugI (frT f X) rest

/-- The node indices of the ancestors and of their other subtrees. -/
def spineIdx : List Fr → List Nat
  | [] => []
  | f :: rest => f.a :: (idxs f.o ++ spineIdx rest)

def childOf (b : Bool) (nd : ONode) : U32 := if b then nd.left else nd.right

def Spine (ns : List ONode) : U32 → U32 → List Fr → Prop
  | r, s, [] => r = s
  | r, s, g :: rest => ∃ c : U32, Spine ns r c rest ∧ c.val = g.a ∧ g.a ≠ nilN ∧
      ∃ nd, ns[g.a]? = some nd ∧ Emb ns (childOf (!g.d) nd) g.o ∧ childOf g.d nd = s

def AncOld (ns : List ONode) : List Fr → Nat → Prop
  | [], _ => True
  | g :: rest, sc => Bal g.o ∧ HOK ns g.o ∧ ht g.o ≤ sc + 1 ∧ sc ≤ ht g.o + 1 ∧
      (∃ nd, ns[g.a]? = some nd ∧ nd.height.val = max sc (ht g.o) + 1) ∧
      AncOld ns rest (max sc (ht g.o) + 1)

/-- The path positions are consistent with the chain slice (innermost frame last in the slice). -/
def ChainOf (ch : List U32) : List Fr → Prop
  | [] => True
  | g :: rest => (∃ c, ch[rest.length]? = some c ∧ c.val = g.a) ∧ ChainOf ch rest

/-- Every ancestor was reached by the key's comparison with the ancestor's coordinate. -/
def DescK (ns : List ONode) (key : Nat) (G : List Fr) : Prop :=
  ∀ g ∈ G, (g.d = true → key < keyAt ns g.a) ∧ (g.d = false → keyAt ns g.a < key)

theorem frT_true {f : Fr} (h : f.d = true) (X : OTree) : frT f X = .node X f.a f.o := by
  simp [frT, h]

theorem frT_false {f : Fr} (h : f.d = false) (X : OTree) : frT f X = .node f.o f.a X := by
  simp [frT, h]

theorem mem_frT {f : Fr} {X : OTree} {j : Nat} :
    j ∈ idxs (frT f X) ↔ j ∈ idxs X ∨ j = f.a ∨ j ∈ idxs f.o := by
  cases hd : f.d with
  | false => rw [frT_false hd]; simp only [idxs, List.mem_append, List.mem_cons]; tauto
  | true => rw [frT_true hd]; simp only [idxs, List.mem_append, List.mem_cons]

theorem mem_plugI {j : Nat} : ∀ (G : List Fr) (X : OTree),
    j ∈ idxs (plugI X G) ↔ j ∈ idxs X ∨ j ∈ spineIdx G
  | [], X => by simp [plugI, spineIdx]
  | g :: rest, X => by
    rw [plugI, mem_plugI rest (frT g X), mem_frT]
    simp only [spineIdx, List.mem_cons, List.mem_append]
    tauto

theorem idxs_frT_eq {f : Fr} {X X' : OTree} (e : idxs X' = idxs X) :
    idxs (frT f X') = idxs (frT f X) := by
  cases hd : f.d with
  | false => rw [frT_false hd, frT_false hd]; simp [idxs, e]
  | true => rw [frT_true hd, frT_true hd]; simp [idxs, e]

theorem idxs_plugI_eq : ∀ (G : List Fr) {X X' : OTree}, idxs X' = idxs X →
    idxs (plugI X' G) = idxs (plugI X G)
  | [], _, _, e => e
  | _ :: rest, _, _, e => idxs_plugI_eq rest (idxs_frT_eq e)

theorem nd_frT {g : Fr} {X : OTree} (h : ND (frT g X)) :
    ND X ∧ ND g.o ∧ g.a ∉ idxs X ∧ g.a ∉ idxs g.o ∧ ∀ j ∈ idxs X, j ∉ idxs g.o := by
  cases hd : g.d with
  | true => rw [frT_true hd] at h; exact h
  | false =>
    rw [frT_false hd] at h
    obtain ⟨h1, h2, h3, h4, h5⟩ := h
    exact ⟨h2, h1, h4, h3, fun j hj hjo => h5 j hjo hj⟩

theorem nd_plugI_sub : ∀ (G : List Fr) {X : OTree}, ND (plugI X G) → ND X
  | [], _, h => h
  | _ :: rest, _, h => (nd_frT (nd_plugI_sub rest h)).1

theorem nd_plugI_disj : ∀ (G : List Fr) {X : OTree}, ND (plugI X G) → ∀ j ∈ spineIdx G, j ∉ idxs X
  | [], _, _, j, hj => by simp [spineIdx] at hj
  | g :: rest, X, h, j, hj => by
    have h1 := nd_plugI_disj rest h
    have h2 := nd_frT (nd_plugI_sub rest h)
    simp only [spineIdx, List.mem_cons, List.mem_append] at hj
    rcases hj with hj | hj | hj
    · rw [hj]; exact h2.2.2.1
    · exact fun hx => h2.2.2.2.2 j hx hj
    · intro hx
      exact h1 j hj (mem_frT.mpr (Or.inl hx))

theorem nd_frT_repl {g : Fr} {X X' : OTree} (h : ND (frT g X)) (hX : ND X')
    (e : idxs X' = idxs X) : ND (frT g X') := by
  cases hd : g.d with
  | false => rw [frT_false hd] at h ⊢; exact nd_repl_r h hX e
  | true => rw [frT_true hd] at h ⊢; exact nd_repl_l h hX e

theorem nd_plugI_swap : ∀ (G : List Fr) {X X' : OTree}, ND (plugI X G) → ND X' →
    idxs X' = idxs X → ND (plugI X' G)
  | [], _, _, _, hX', _ => hX'
  | g :: rest, X, X', h, hX', e =>
    nd_plugI_swap rest (X := frT g X) (X' := frT g X') h
      (nd_frT_repl (nd_plugI_sub rest h) hX' e) (idxs_frT_eq e)

theorem bst_frT_sub {ns : List ONode} {g : Fr} {X : OTree} (h : BST ns (frT g X)) : BST ns X := by
  cases hd : g.d with
  | true => rw [frT_true hd] at h; exact h.1
  | false => rw [frT_false hd] at h; exact h.2.1

theorem bst_plugI_sub {ns : List ONode} : ∀ (G : List Fr) {X : OTree}, BST ns (plugI X G) → BST ns X
  | [], _, h => h
  | _ :: rest, _, h => bst_frT_sub (bst_plugI_sub rest h)

theorem bst_frT_repl {ns : List ONode} {g : Fr} {X X' : OTree} (h : BST ns (frT g X))
    (hX : BST ns X') (e : idxs X' = idxs X) : BST ns (frT g X') := by
  cases hd : g.d with
  | false => rw [frT_false hd] at h ⊢; exact bst_repl_r h hX e
  | true => rw [frT_true hd] at h ⊢; exact bst_repl_l h hX e

theorem bst_plugI_swap {ns : List ONode} : ∀ (G : List Fr) {X X' : OTree}, BST ns (plugI X G) →
    BST ns X' → idxs X' = idxs X → BST ns (plugI X' G)
  | [], _, _, _, hX', _ => hX'
  | g :: rest, X, X', h, hX', e =>
    bst_plugI_swap rest (X := frT g X) (X' := frT g X') h
      (bst_frT_repl (bst_plugI_sub rest h) hX' e) (idxs_frT_eq e)

/-! ### Inserting a new node at a slot -/

theorem bst_frT_ins {ns : List ONode} {g : Fr} {X X' : OTree} {i key : Nat}
    (hk : keyAt ns i = key) (hd : (g.d = true → key < keyAt ns g.a) ∧
      (g.d = false → keyAt ns g.a < key))
    (hmem : ∀ j, j ∈ idxs X' ↔ j ∈ idxs X ∨ j = i) (h : BST ns (frT g X)) (hX : BST ns X') :
    BST ns (frT g X') := by
  cases hg : g.d with
  | true =>
    rw [frT_true hg] at h ⊢
    obtain ⟨_, hBo, hlt, hgt⟩ := h
    refine ⟨hX, hBo, fun j hj => ?_, hgt⟩
    rcases (hmem j).mp hj with hj | hj
    · exact hlt j hj
    · rw [hj, hk]; exact hd.1 hg
  | false =>
    rw [frT_false hg] at h ⊢
    obtain ⟨hBo, _, hlt, hgt⟩ := h
    refine ⟨hBo, hX, hlt, fun j hj => ?_⟩
    rcases (hmem j).mp hj with hj | hj
    · exact hgt j hj
    · rw [hj, hk]; exact hd.2 hg

theorem bst_plugI_ins {ns : List ONode} {i key : Nat} (hk : keyAt ns i = key) :
    ∀ (G : List Fr) {X X' : OTree}, DescK ns key G → (∀ j, j ∈ idxs X' ↔ j ∈ idxs X ∨ j = i) →
      BST ns (plugI X G) → BST ns X' → BST ns (plugI X' G)
  | [], _, _, _, _, _, hX' => hX'
  | g :: rest, X, X', hD, hmem, h, hX' =>
    bst_plugI_ins hk rest (X := frT g X) (X' := frT g X')
      (fun f hf => hD f (List.mem_cons_of_mem _ hf))
      (fun j => by rw [mem_frT, mem_frT, hmem j]; tauto)
      h (bst_frT_ins hk (hD g (List.mem_cons_self ..)) hmem (bst_plugI_sub rest h) hX')

theorem nd_frT_ins {g : Fr} {X X' : OTree} {i : Nat}
    (hmem : ∀ j, j ∈ idxs X' ↔ j ∈ idxs X ∨ j = i) (h : ND (frT g X)) (hX : ND X')
    (hi : i ∉ idxs (frT g X)) : ND (frT g X') := by
  have hio : i ∉ idxs g.o := fun hh => hi (mem_frT.mpr (Or.inr (Or.inr hh)))
  have hia : i ≠ g.a := fun hh => hi (mem_frT.mpr (Or.inr (Or.inl hh)))
  obtain ⟨_, hNo, hax, hao, hdis⟩ := nd_frT h
  have hax' : g.a ∉ idxs X' := by
    intro hh
    rcases (hmem _).mp hh with hh | hh
    · exact hax hh
    · exact hia hh.symm
  have hdis' : ∀ j ∈ idxs X', j ∉ idxs g.o := by
    intro j hj
    rcases (hmem j).mp hj with hj | hj
    · exact hdis j hj
    · rw [hj]; exact hio
  cases hd : g.d with
  | true => rw [frT_true hd]; exact ⟨hX, hNo, hax', hao, hdis'⟩
  | false => rw [frT_false hd]; exact ⟨hNo, hX, hao, hax', fun j hj hjx => hdis' j hjx hj⟩

theorem nd_plugI_ins {i : Nat} : ∀ (G : List Fr) {X X' : OTree},
    (∀ j, j ∈ idxs X' ↔ j ∈ idxs X ∨ j = i) → ND (plugI X G) → ND X' → i ∉ idxs (plugI X G) →
      ND (plugI X' G)
  | [], _, _, _, _, hX', _ => hX'
  | g :: rest, X, X', hmem, h, hX', hi =>
    nd_plugI_ins rest (X := frT g X) (X' := frT g X')
      (fun j => by rw [mem_frT, mem_frT, hmem j]; tauto) h
      (nd_frT_ins hmem (nd_plugI_sub rest h) hX'
        (fun hh => hi ((mem_plugI rest _).mpr (Or.inl hh)))) hi

theorem idxs_plugI_perm {i : Nat} : ∀ (G : List Fr) {X X' : OTree},
    (idxs X').Perm (i :: idxs X) → (idxs (plugI X' G)).Perm (i :: idxs (plugI X G))
  | [], _, _, h => h
  | g :: rest, X, X', h =>
    idxs_plugI_perm rest (X := frT g X) (X' := frT g X') (by
      cases hd : g.d with
      | true =>
        rw [frT_true hd, frT_true hd]
        simp only [idxs]
        exact List.Perm.append_right _ h
      | false =>
        rw [frT_false hd, frT_false hd]
        simp only [idxs]
        exact (List.Perm.append_left _ (List.Perm.cons _ h)).trans
          ((List.Perm.append_left _ (List.Perm.swap _ _ _)).trans List.perm_middle))

/-! ### Heights along a path -/

theorem ht_frT (g : Fr) (X : OTree) : ht (frT g X) = max (ht X) (ht g.o) + 1 := by
  cases hd : g.d with
  | true => rw [frT_true hd]; rfl
  | false => rw [frT_false hd]; simp only [ht]; omega

theorem bal_frT_sub {g : Fr} {X : OTree} (h : Bal (frT g X)) :
    Bal X ∧ Bal g.o ∧ ht X ≤ ht g.o + 1 ∧ ht g.o ≤ ht X + 1 := by
  cases hd : g.d with
  | true => rw [frT_true hd] at h; exact h
  | false =>
    rw [frT_false hd] at h
    obtain ⟨h1, h2, h3, h4⟩ := h
    exact ⟨h2, h1, h4, h3⟩

theorem hok_frT_sub {ns : List ONode} {g : Fr} {X : OTree} (h : HOK ns (frT g X)) :
    HOK ns X ∧ HOK ns g.o ∧ ∃ nd, ns[g.a]? = some nd ∧ nd.height.val = max (ht X) (ht g.o) + 1 := by
  cases hd : g.d with
  | true => rw [frT_true hd] at h; exact h
  | false =>
    rw [frT_false hd] at h
    obtain ⟨h1, h2, nd, hnd, hh⟩ := h
    exact ⟨h2, h1, nd, hnd, by omega⟩

theorem bal_plugI_sub : ∀ (G : List Fr) {X : OTree}, Bal (plugI X G) → Bal X
  | [], _, h => h
  | _ :: rest, _, h => (bal_frT_sub (bal_plugI_sub rest h)).1

theorem hok_plugI_sub {ns : List ONode} : ∀ (G : List Fr) {X : OTree}, HOK ns (plugI X G) → HOK ns X
  | [], _, h => h
  | _ :: rest, _, h => (hok_frT_sub (hok_plugI_sub rest h)).1

theorem ancOld_of_valid {ns : List ONode} : ∀ (G : List Fr) {X : OTree}, Bal (plugI X G) →
    HOK ns (plugI X G) → AncOld ns G (ht X)
  | [], _, _, _ => trivial
  | g :: rest, X, hb, hh => by
    have hbf := bal_frT_sub (bal_plugI_sub rest hb)
    have hhf := hok_frT_sub (hok_plugI_sub rest hh)
    have ih := ancOld_of_valid rest (X := frT g X) hb hh
    rw [ht_frT] at ih
    exact ⟨hbf.2.1, hhf.2.1, hbf.2.2.2, hbf.2.2.1, hhf.2.2, ih⟩

/-- Height agreement: `ns'` has a node at `j` of the same stored height wherever `ns` has one. -/
def HAg (ns ns' : List ONode) (j : Nat) : Prop :=
  ∀ nd, ns[j]? = some nd → ∃ nd', ns'[j]? = some nd' ∧ nd'.height = nd.height

theorem hok_hcongr {ns ns' : List ONode} : ∀ {t : OTree}, HOK ns t →
    (∀ j ∈ idxs t, HAg ns ns' j) → HOK ns' t
  | .leaf, _, _ => trivial
  | .node l i r, ⟨a, b, nd, hnd, hh⟩, hj => by
    obtain ⟨nd', hnd', hht⟩ := hj i (by simp [idxs]) nd hnd
    refine ⟨hok_hcongr a ?_, hok_hcongr b ?_, nd', hnd', by rw [hht]; exact hh⟩
    · intro j hj'; exact hj j (by simp [idxs, hj'])
    · intro j hj'; exact hj j (by simp [idxs, hj'])

theorem ancOld_hcongr {ns ns' : List ONode} : ∀ (G : List Fr) {sc : Nat}, AncOld ns G sc →
    (∀ j ∈ spineIdx G, HAg ns ns' j) → AncOld ns' G sc
  | [], _, _, _ => trivial
  | g :: rest, sc, ⟨hb, hh, h1, h2, ⟨nd, hnd, hht⟩, hrest⟩, hj => by
    obtain ⟨nd', hnd', hnh⟩ := hj g.a (by simp [spineIdx]) nd hnd
    refine ⟨hb, hok_hcongr hh (fun j hjo => hj j (by simp [spineIdx, hjo])), h1, h2,
      ⟨nd', hnd', by rw [hnh]; exact hht⟩, ancOld_hcongr rest hrest (fun j hjr => hj j ?_)⟩
    simp only [spineIdx, List.mem_cons, List.mem_append]
    exact Or.inr (Or.inr hjr)

theorem spine_congr {ns ns' : List ONode} (r s : U32) : ∀ (G : List Fr),
    (∀ j ∈ spineIdx G, ns'[j]? = ns[j]?) → Spine ns r s G → Spine ns' r s G
  | [], _, h => h
  | g :: rest, hj, ⟨c, hc, hcv, hga, nd, hnd, hEo, hch⟩ => by
    refine ⟨c, spine_congr r c rest (fun j hjr => hj j ?_) hc, hcv, hga, nd, ?_, ?_, hch⟩
    · simp only [spineIdx, List.mem_cons, List.mem_append]
      exact Or.inr (Or.inr hjr)
    · rw [hj g.a (by simp [spineIdx])]; exact hnd
    · exact emb_congr hEo (fun j hjo => hj j (by simp [spineIdx, hjo]))

theorem emb_root_mem {ns : List ONode} {x : U32} {t : OTree} (h : Emb ns x t) (hx : x.val ≠ nilN) :
    x.val ∈ idxs t := by
  cases t with
  | leaf => exact absurd h hx
  | node l i r => obtain ⟨hxi, _⟩ := h; rw [hxi]; simp [idxs]

theorem emb_root_nil {ns : List ONode} {x : U32} (h : Emb ns x .leaf) : x.val = nilN := h

/-- `rebalance` at the innermost ancestor of a path: the shape facts used by the chain loop. -/
theorem rebalance_frT (o o' : Ov) (L : List ONode) (hL : o.nodes.val = L) (n x' : U32)
    (g : Fr) (S : OTree) (sc : Nat)
    (hE : Emb L n (frT g S)) (hND : ND (frT g S)) (hB : BST L (frT g S))
    (hBS : Bal S) (hHS : HOK L S) (hBo : Bal g.o) (hHo : HOK L g.o)
    (h1 : sc ≤ ht S) (h2 : ht S ≤ sc + 1) (h3 : ht g.o ≤ sc + 1) (h4 : sc ≤ ht g.o + 1)
    (h : hashtrinity.overflow.Overflow.rebalance CI o n = ok (x', o')) :
    ∃ T' A, o'.nodes.val = A ∧ Emb A x' T' ∧ idxs T' = idxs (frT g S) ∧ ND T' ∧
      BST A T' ∧ HOK A T' ∧ Bal T' ∧ Rel (idxs (frT g S)) L A ∧
      o'.free = o.free ∧ o'.root = o.root ∧ o'.len = o.len ∧
      max sc (ht g.o) + 1 ≤ ht T' ∧ ht T' ≤ max sc (ht g.o) + 1 + 1 := by
  cases hd : g.d with
  | true =>
    rw [frT_true hd] at hE hND hB ⊢
    obtain ⟨T', A, hA, hEA, hid, hnd, hbA, hhA, hbal, hrel, hf, hr, hl, ha, hb, hc⟩ :=
      rebalance_spec o o' L hL n x' S g.o g.a hE hND hB hBS hBo hHS hHo
        ⟨by omega, by omega⟩ h
    refine ⟨T', A, hA, hEA, hid, hnd, hbA, hhA, hbal, hrel, hf, hr, hl, ?_, ?_⟩
    · by_cases hbb : ht S ≤ ht g.o + 1 ∧ ht g.o ≤ ht S + 1
      · have := hc hbb.1 hbb.2
        omega
      · omega
    · omega
  | false =>
    rw [frT_false hd] at hE hND hB ⊢
    obtain ⟨T', A, hA, hEA, hid, hnd, hbA, hhA, hbal, hrel, hf, hr, hl, ha, hb, hc⟩ :=
      rebalance_spec o o' L hL n x' g.o S g.a hE hND hB hBo hBS hHo hHS
        ⟨by omega, by omega⟩ h
    refine ⟨T', A, hA, hEA, hid, hnd, hbA, hhA, hbal, hrel, hf, hr, hl, ?_, ?_⟩
    · by_cases hbb : ht g.o ≤ ht S + 1 ∧ ht S ≤ ht g.o + 1
      · have := hc hbb.1 hbb.2
        omega
      · omega
    · omega

/-! ### The parent relink -/

/-- Point the `b` child (`true`: left) of `nd` at `x`. -/
def setChild (b : Bool) (nd : ONode) (x : U32) : ONode :=
  if b then {nd with left := x} else {nd with right := x}

theorem sameD_setChild (b : Bool) (nd : ONode) (x : U32) : sameD nd (setChild b nd x) := by
  cases b <;> exact ⟨rfl, rfl, rfl⟩

theorem height_setChild (b : Bool) (nd : ONode) (x : U32) : (setChild b nd x).height = nd.height := by
  cases b <;> rfl

theorem childOf_setChild (b : Bool) (nd : ONode) (x : U32) : childOf b (setChild b nd x) = x := by
  cases b <;> rfl

theorem childOf_setChild_other (b : Bool) (nd : ONode) (x : U32) :
    childOf (!b) (setChild b nd x) = childOf (!b) nd := by
  cases b <;> rfl

theorem emb_frT {ns : List ONode} {g : Fr} {S : OTree} {n s : U32} {nd : ONode}
    (hn : n.val = g.a) (hga : g.a ≠ nilN) (hnd : ns[g.a]? = some nd)
    (hEo : Emb ns (childOf (!g.d) nd) g.o) (hch : childOf g.d nd = s) (hES : Emb ns s S) :
    Emb ns n (frT g S) := by
  cases hd : g.d with
  | true =>
    rw [frT_true hd]
    rw [hd] at hEo hch
    exact ⟨hn, hga, nd, hnd, by rw [← hch] at hES; exact hES, hEo⟩
  | false =>
    rw [frT_false hd]
    rw [hd] at hEo hch
    exact ⟨hn, hga, nd, hnd, hEo, by rw [← hch] at hES; exact hES⟩

theorem chain_relink {ns A : List ONode} {r c' x' : U32} {X T' : OTree} {h : Fr}
    {rest' : List Fr} {ndh : ONode}
    (hND : ND (plugI X (h :: rest'))) (hrel : Rel (idxs X) ns A)
    (hndh : ns[h.a]? = some ndh) (hSpc : Spine ns r c' rest') (hc'v : c'.val = h.a)
    (hha : h.a ≠ nilN) (hEoh : Emb ns (childOf (!h.d) ndh) h.o)
    (hEA : Emb A x' T') (hid : idxs T' = idxs X) :
    Spine (A.set h.a (setChild h.d ndh x')) r x' (h :: rest') ∧
    Emb (A.set h.a (setChild h.d ndh x')) x' T' ∧
    Rel [h.a] A (A.set h.a (setChild h.d ndh x')) ∧
    (∀ j, j ∈ spineIdx (h :: rest') → HAg ns (A.set h.a (setChild h.d ndh x')) j) ∧
    (∀ j, j ∈ spineIdx rest' → (A.set h.a (setChild h.d ndh x'))[j]? = ns[j]?) := by
  have hND3 : ND (plugI (frT h X) rest') := hND
  have hdisj := nd_plugI_disj (h :: rest') hND
  have hdisj' := nd_plugI_disj rest' hND3
  have hfh := nd_frT (nd_plugI_sub rest' hND3)
  have hhaX : h.a ∉ idxs X := hfh.2.2.1
  have hhao : h.a ∉ idxs h.o := hfh.2.2.2.1
  have hAh : A[h.a]? = some ndh := by rw [rel_out hrel hhaX]; exact hndh
  have hAlt : h.a < A.length := (List.getElem?_eq_some_iff.mp hAh).1
  have hrelh : Rel [h.a] A (A.set h.a (setChild h.d ndh x')) :=
    Rel_set hAh (sameD_setChild _ _ _)
  have hout : ∀ j, j ∈ spineIdx rest' → (A.set h.a (setChild h.d ndh x'))[j]? = ns[j]? := by
    intro j hj
    have hjne : j ≠ h.a := fun e => hdisj' j hj (by rw [e]; exact mem_frT.mpr (Or.inr (Or.inl rfl)))
    have hjX : j ∉ idxs X := fun hx => hdisj' j hj (mem_frT.mpr (Or.inl hx))
    rw [List.getElem?_set_ne (Ne.symm hjne), rel_out hrel hjX]
  refine ⟨?_, ?_, hrelh, ?_, hout⟩
  · obtain ⟨c'', hc''⟩ : ∃ c'', c'' = c' := ⟨_, rfl⟩
    refine ⟨c', spine_congr r c' rest' hout hSpc, hc'v, hha, setChild h.d ndh x', ?_, ?_,
      by rw [childOf_setChild]⟩
    · exact List.getElem?_set_self hAlt
    · rw [childOf_setChild_other]
      refine emb_congr (emb_rel hrel hEoh (fun j hj hjX => ?_)) (fun j hj => ?_)
      · exact hdisj j (by simp [spineIdx, hj]) hjX
      · exact List.getElem?_set_ne (show h.a ≠ j from fun e => hhao (e ▸ hj))
  · exact emb_congr hEA (fun j hj => List.getElem?_set_ne
      (show h.a ≠ j from fun e => hhaX (by rw [e, ← hid]; exact hj)))
  · intro j hj nd hnd
    simp only [spineIdx, List.mem_cons, List.mem_append] at hj
    rcases hj with hj | hj | hj
    · subst hj
      rw [hndh] at hnd
      have := Option.some.inj hnd
      subst this
      exact ⟨setChild h.d ndh x', List.getElem?_set_self hAlt, height_setChild _ _ _⟩
    · have hjne : j ≠ h.a := fun e => hhao (e ▸ hj)
      have hjX : j ∉ idxs X := fun hx => hdisj j (by simp [spineIdx, hj]) hx
      refine ⟨nd, ?_, rfl⟩
      rw [List.getElem?_set_ne (Ne.symm hjne), rel_out hrel hjX]; exact hnd
    · exact ⟨nd, by rw [hout j hj]; exact hnd, rfl⟩

/-! ### `rebalance_chain` -/

/-- The loop invariant of `rebalance_chain`: the path `G` is still above the slot holding the
rebalanced subtree `S`. -/
def ChainInv (ns0 : List ONode) (U : List Nat) (F0 : alloc.vec.Vec U32) (N0 : Usize)
    (chain : List U32) (o : Ov) (idx : Usize) : Prop :=
  ∃ (G : List Fr) (S : OTree) (s : U32) (sc : Nat),
    idx.val = G.length ∧ ChainOf chain G ∧
    Emb o.nodes.val s S ∧ Bal S ∧ HOK o.nodes.val S ∧
    BST o.nodes.val (plugI S G) ∧ ND (plugI S G) ∧ idxs (plugI S G) = U ∧
    Spine o.nodes.val o.root s G ∧ AncOld o.nodes.val G sc ∧ sc ≤ ht S ∧ ht S ≤ sc + 1 ∧
    Rel U ns0 o.nodes.val ∧ o.free = F0 ∧ o.len = N0

theorem rebalance_chain_loop_spec (ns0 : List ONode) (U : List Nat) (F0 : alloc.vec.Vec U32)
    (N0 : Usize) (chain : Slice U32) (o0 o' : Ov) (idx0 : Usize)
    (hinv : ChainInv ns0 U F0 N0 chain.val o0 idx0)
    (h : hashtrinity.overflow.Overflow.rebalance_chain_loop CI o0 chain idx0 = ok o') :
    ∃ T', Emb o'.nodes.val o'.root T' ∧ Bal T' ∧ HOK o'.nodes.val T' ∧ BST o'.nodes.val T' ∧
      ND T' ∧ idxs T' = U ∧ Rel U ns0 o'.nodes.val ∧ o'.free = F0 ∧ o'.len = N0 := by
  unfold hashtrinity.overflow.Overflow.rebalance_chain_loop at h
  refine loop_pc (fun (st : Ov × Usize) => ChainInv ns0 U F0 N0 chain.val st.1 st.2)
    (fun o' => ∃ T', Emb o'.nodes.val o'.root T' ∧ Bal T' ∧ HOK o'.nodes.val T' ∧
      BST o'.nodes.val T' ∧ ND T' ∧ idxs T' = U ∧ Rel U ns0 o'.nodes.val ∧ o'.free = F0 ∧
      o'.len = N0) _ ?_ (o0, idx0) hinv o' h
  intro x hx cf hcf
  obtain ⟨o, idx⟩ := x
  have hx' : ChainInv ns0 U F0 N0 chain.val o idx := hx
  have hcf' : hashtrinity.overflow.Overflow.rebalance_chain_loop.body CI chain o idx = ok cf := hcf
  unfold hashtrinity.overflow.Overflow.rebalance_chain_loop.body at hcf'
  obtain ⟨G, S, s, sc, hidx, hch, hES, hBalS, hHS, hBST, hND, hU, hSp, hAnc, hsc1, hsc2, hRel,
    hfree, hlen⟩ := hx'
  have hu0 : (0#usize).val = 0 := rfl
  have hu1 : (1#usize).val = 1 := rfl
  by_cases hpos : idx > 0#usize
  · rw [if_pos hpos] at hcf'
    have hpos' : 0 < idx.val := by
      have := (UScalar.lt_equiv (0#usize) idx).mp hpos
      omega
    obtain ⟨idx1, hidx1, hcf'⟩ := bind_inv hcf'
    have hidx1v := usub_ok hidx1
    rw [hu1] at hidx1v
    obtain ⟨n, hn, hcf'⟩ := bind_inv hcf'
    have hn' := slice_index_ok chain idx1 n hn
    cases G with
    | nil => simp at hidx; omega
    | cons g rest =>
      have hrl : rest.length = idx1.val := by simp only [List.length_cons] at hidx; omega
      obtain ⟨⟨c0, hc0, hc0v⟩, hchrest⟩ := hch
      rw [← hrl, hc0] at hn'
      have hnc : c0 = n := Option.some.inj hn'
      have hnv : n.val = g.a := by rw [← hnc]; exact hc0v
      obtain ⟨⟨newroot, o1⟩, hreb, hcf'⟩ := bind_inv hcf'
      obtain ⟨c, hSpc, hcv, hga, nd, hnd, hEo, hch'⟩ := hSp
      obtain ⟨hBg, hHg, hg1, hg2, _, hAncr⟩ := hAnc
      have hEn : Emb o.nodes.val n (frT g S) := emb_frT hnv hga hnd hEo hch' hES
      obtain ⟨T', A, hA, hEA, hid, hndT, hbA, hhA, hbal, hrel, hf, hr, hl, hlo, hhi⟩ :=
        rebalance_frT o o1 o.nodes.val rfl n newroot g S sc hEn (nd_plugI_sub rest hND)
          (bst_plugI_sub rest hBST) hBalS hHS hBg hHg hsc1 hsc2 hg1 hg2 hreb
      have hUm : ∀ j ∈ idxs (frT g S), j ∈ U := fun j hj => by
        rw [← hU]; exact (mem_plugI rest (frT g S)).mpr (Or.inl hj)
      cases rest with
      | nil =>
        have hidx0 : idx1 = 0#usize := UScalar.eq_of_val_eq (by rw [← hrl]; rfl)
        simp only [hidx0, if_true] at hcf'
        have := (Result.ok_injective hcf').symm
        subst this
        refine ⟨[], T', newroot, max sc (ht g.o) + 1, rfl, trivial, ?_, hbal, ?_, ?_,
          hndT, hid.trans hU, rfl, trivial, hlo, hhi, ?_, ?_, ?_⟩
        · dsimp only; rw [hA]; exact hEA
        · dsimp only; rw [hA]; exact hhA
        · dsimp only; rw [hA]; exact hbA
        · dsimp only; rw [hA]
          exact Rel.trans' (U := U) hRel hrel (fun j hj => hj) hUm
        · dsimp only; rw [hf, hfree]
        · dsimp only; rw [hl, hlen]
      | cons h' rest' =>
        have hrl' : rest'.length + 1 = idx1.val := by rw [← hrl]; rfl
        have hidx0 : ¬ idx1 = 0#usize := fun e => by rw [e] at hrl'; simp at hrl'
        simp only [hidx0, if_false] at hcf'
        obtain ⟨i, hi, hcf'⟩ := bind_inv hcf'
        have hiv := usub_ok hi
        rw [hu1] at hiv
        obtain ⟨p, hp, hcf'⟩ := bind_inv hcf'
        have hp' := slice_index_ok chain i p hp
        obtain ⟨⟨c1, hc1, hc1v⟩, hchrest'⟩ := hchrest
        have hirl : i.val = rest'.length := by omega
        rw [hirl, hc1] at hp'
        have hpc : c1 = p := Option.some.inj hp'
        have hpv : p.val = h'.a := by rw [← hpc]; exact hc1v
        simp only [lift, bind_tc_ok] at hcf'
        obtain ⟨n1, hn1, hcf'⟩ := bind_inv hcf'
        have hn1' := idx_ok hn1
        obtain ⟨c', hSpc', hc'v, hha, ndh, hndh, hEoh, hchh⟩ := hSpc
        have hND2 : ND (plugI (frT g S) (h' :: rest')) := hND
        have hND3 : ND (plugI (frT h' (frT g S)) rest') := hND
        have hdisj := nd_plugI_disj (h' :: rest') hND2
        have hhaX : h'.a ∉ idxs (frT g S) := (nd_frT (nd_plugI_sub rest' hND3)).2.2.1
        have hAh : A[h'.a]? = some ndh := by rw [rel_out hrel hhaX]; exact hndh
        rw [hA, hpv, hAh] at hn1'
        have hn1n : ndh = n1 := Option.some.inj hn1'
        subst hn1n
        obtain ⟨hSp'', hEA'', hrel'', hHAg, hout⟩ :=
          chain_relink (r := o.root) (x' := newroot) (T' := T') hND2 hrel hndh hSpc' hc'v hha hEoh hEA hid
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
        have key : ∀ v : alloc.vec.Vec ONode, v.val = A.set h'.a (setChild h'.d ndh newroot) →
            ChainInv ns0 U F0 N0 chain.val
              { nodes := v, free := o1.free, root := o1.root, len := o1.len } idx1 := by
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
          · show Spine v.val o1.root newroot (h' :: rest')
            rw [hv, hr]; exact hSp''
          · show AncOld v.val (h' :: rest') (max sc (ht g.o) + 1)
            rw [hv]; exact ancOld_hcongr (h' :: rest') hAncr hHAg
          · show Rel U ns0 v.val
            rw [hv]
            exact Rel.trans' (U := U) (Rel.trans' (U := U) hRel hrel (fun j hj => hj) hUm) hrel''
              (fun j hj => hj) (fun j hj => by rw [List.mem_singleton] at hj; rw [hj]; exact hUh)
          · show o1.free = F0
            rw [hf, hfree]
          · show o1.len = N0
            rw [hl, hlen]
        have hcnv : c.val = n.val := by rw [hcv, hnv]
        have hcn' : c = n := UScalar.eq_of_val_eq hcnv
        have hnX : n.val ∈ idxs (frT g S) := by
          rw [hnv]; exact mem_frT.mpr (Or.inr (Or.inl rfl))
        cases hhd : h'.d with
        | true =>
          have hcond : ndh.left = n := by
            have h1 := hchh
            rw [hhd] at h1
            simp [childOf] at h1
            rw [h1, hcn']
          rw [if_pos hcond] at hcf'
          obtain ⟨⟨n2, back⟩, hn2, hcf'⟩ := bind_inv hcf'
          obtain ⟨hn2', hback⟩ := idxm_ok hn2
          rw [hA, hpv, hAh] at hn2'
          have hn2n : ndh = n2 := Option.some.inj hn2'
          subst hn2n
          have hcf2 := (Result.ok_injective hcf').symm
          subst hcf2
          exact key _ (by rw [hback, hA, hpv, hhd]; rfl)
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
          rw [if_neg hcond] at hcf'
          obtain ⟨⟨n2, back⟩, hn2, hcf'⟩ := bind_inv hcf'
          obtain ⟨hn2', hback⟩ := idxm_ok hn2
          rw [hA, hpv, hAh] at hn2'
          have hn2n : ndh = n2 := Option.some.inj hn2'
          subst hn2n
          have hcf2 := (Result.ok_injective hcf').symm
          subst hcf2
          exact key _ (by rw [hback, hA, hpv, hhd]; rfl)
  · rw [if_neg hpos] at hcf'
    have := (Result.ok_injective hcf').symm
    subst this
    have hidx0 : idx.val = 0 := by
      have := (not_congr (UScalar.lt_equiv (0#usize) idx)).mp hpos
      omega
    have hG : G = [] := List.length_eq_zero_iff.mp (by omega)
    subst hG
    have hroot : o.root = s := hSp
    refine ⟨S, ?_, hBalS, hHS, hBST, hND, hU, hRel, hfree, hlen⟩
    rw [hroot]; exact hES

end Apeiron.MapExecuted
