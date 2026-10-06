-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe`: the executed overflow tree's read.
import Map.ExecutedProbe
open Aeneas Aeneas.Std Result

/-!
# The executed overflow tree: shape and read

`hashtrinity::overflow::Overflow` (`src/hashtrinity/overflow.rs`) is a binary search tree in a `Vec`
arena: node `i` is `nodes[i]`, children are `u32` indices, `NIL = u32::MAX` is the empty child.

* `OTree` is the abstract shape; `Emb ns x t` says the arena `ns` rooted at index `x` has shape `t`
  (`x = NIL` for the empty tree; otherwise `x` names an arena node whose children embed the
  subtrees).  `idxs t` lists the arena indices of `t`'s nodes in order.
* `BST ns t` is the search-tree order on the nodes' coordinates (`Node::key`): strictly smaller on
  the left, strictly larger on the right.
* `find_node_spec`: the extracted `find_node` loop returns a node of `t` whose coordinate is the
  query, or `NIL` exactly when no node of `t` has it.
* `ov_get_spec`: with every node holding the single entry `(key, some v)` (the shape every insert
  of a `u64` key leaves, the coordinate being the key), `Overflow::get` returns `some v` exactly
  for the coordinates present.

No height (AVL) property is used or claimed: the read is correct for any search tree, balanced or
not.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-- An overflow node at `u64` keys and `i64` values. -/
abbrev ONode := hashtrinity.overflow.Node U64 I64
/-- The overflow store at `u64` keys and `i64` values. -/
abbrev Ov := hashtrinity.overflow.Overflow U64 I64

/-- `NIL = u32::MAX`. -/
def nilN : Nat := 2^32 - 1

theorem NIL_val : hashtrinity.overflow.NIL.val = nilN := by
  unfold hashtrinity.overflow.NIL nilN; rfl

theorem NIL_eq_iff (x : U32) : x = hashtrinity.overflow.NIL ↔ x.val = nilN := by
  constructor
  · intro h; rw [h, NIL_val]
  · intro h; exact UScalar.eq_of_val_eq (by rw [h, NIL_val])

/-- The abstract shape of the overflow tree: arena indices at the nodes. -/
inductive OTree where
  | leaf : OTree
  | node : OTree → Nat → OTree → OTree

/-- The arena indices of a shape, in order. -/
def idxs : OTree → List Nat
  | .leaf => []
  | .node l i r => idxs l ++ i :: idxs r

/-- `Emb ns x t`: the arena `ns` rooted at index `x` has shape `t`. -/
def Emb (ns : List ONode) : U32 → OTree → Prop
  | x, .leaf => x.val = nilN
  | x, .node l i r => x.val = i ∧ i ≠ nilN ∧
      ∃ nd, ns[i]? = some nd ∧ Emb ns nd.left l ∧ Emb ns nd.right r

/-- The coordinate of arena node `i` (`0` off the arena; never read there). -/
def keyAt (ns : List ONode) (i : Nat) : Nat :=
  match ns[i]? with
  | some nd => nd.key.val
  | none => 0

/-- The search-tree order on the nodes' coordinates. -/
def BST (ns : List ONode) : OTree → Prop
  | .leaf => True
  | .node l i r => BST ns l ∧ BST ns r ∧ (∀ j ∈ idxs l, keyAt ns j < keyAt ns i) ∧
      (∀ j ∈ idxs r, keyAt ns i < keyAt ns j)

theorem emb_mem {ns : List ONode} : ∀ {x : U32} {t : OTree}, Emb ns x t →
    ∀ i ∈ idxs t, ∃ nd, ns[i]? = some nd ∧ i ≠ nilN
  | _, .leaf, _, i, hi => by simp [idxs] at hi
  | x, .node l j r, hE, i, hi => by
    obtain ⟨_, hj, nd, hnd, hl, hr⟩ := hE
    simp only [idxs, List.mem_append, List.mem_cons] at hi
    rcases hi with hi | hi | hi
    · exact emb_mem hl i hi
    · subst hi; exact ⟨nd, hnd, hj⟩
    · exact emb_mem hr i hi

theorem emb_unique {ns : List ONode} : ∀ {x : U32} {t t' : OTree}, Emb ns x t → Emb ns x t' →
    t = t'
  | _, .leaf, .leaf, _, _ => rfl
  | x, .leaf, .node _ i _, h1, h2 => by
    exfalso
    have h1' : x.val = nilN := h1
    obtain ⟨hx, hi, _⟩ := h2
    exact hi (by rw [← hx]; exact h1')
  | x, .node _ i _, .leaf, h1, h2 => by
    exfalso
    have h2' : x.val = nilN := h2
    obtain ⟨hx, hi, _⟩ := h1
    exact hi (by rw [← hx]; exact h2')
  | _, .node l i r, .node l' i' r', h1, h2 => by
    obtain ⟨hx, _, nd, hnd, hl, hr⟩ := h1
    obtain ⟨hx', _, nd', hnd', hl', hr'⟩ := h2
    have hii : i = i' := by rw [← hx, ← hx']
    subst hii
    rw [hnd] at hnd'
    have := Option.some.inj hnd'
    subst this
    rw [emb_unique hl hl', emb_unique hr hr']

theorem cast_usize_u32_idx (x : U32) : (UScalar.cast .Usize x).val = x.val := cast_usize_u32 x

/-- `find_node`: the node of the tree carrying the query coordinate, or `NIL` when none does. -/
theorem find_node_spec (ns : alloc.vec.Vec ONode) (c : U64) (T : OTree) (root : U32)
    (hE : Emb ns.val root T) (hB : BST ns.val T) (r : U32)
    (h : hashtrinity.overflow.Overflow.find_node_loop ns c root = ok r) :
    (r.val = nilN ∧ ∀ i ∈ idxs T, keyAt ns.val i ≠ c.val) ∨
      (r.val ∈ idxs T ∧ keyAt ns.val r.val = c.val) := by
  unfold hashtrinity.overflow.Overflow.find_node_loop at h
  refine loop_pc (fun (cur : U32) => ∃ S, Emb ns.val cur S ∧ BST ns.val S ∧
      (∀ i ∈ idxs S, i ∈ idxs T) ∧
      ((∃ i ∈ idxs T, keyAt ns.val i = c.val) → ∃ i ∈ idxs S, keyAt ns.val i = c.val))
    (fun r => (r.val = nilN ∧ ∀ i ∈ idxs T, keyAt ns.val i ≠ c.val) ∨
      (r.val ∈ idxs T ∧ keyAt ns.val r.val = c.val)) _ ?_ root
    ⟨T, hE, hB, fun i hi => hi, fun h => h⟩ r h
  intro cur ⟨S, hS, hBS, hsub, hfind⟩ cf hcf
  unfold hashtrinity.overflow.Overflow.find_node_loop.body at hcf
  by_cases hn : (cur != hashtrinity.overflow.NIL) = true
  · rw [if_pos hn] at hcf
    have hn' : cur.val ≠ nilN := by
      intro hh; have := (NIL_eq_iff cur).mpr hh; simp [this] at hn
    cases S with
    | leaf => exact absurd hS hn'
    | node l i rr =>
      obtain ⟨hci, _, nd, hnd, hl, hr⟩ := hS
      obtain ⟨hBl, hBr, hlt, hgt⟩ := hBS
      simp only [lift, bind_tc_ok] at hcf
      obtain ⟨n, hn1, hcf⟩ := bind_inv hcf
      have hn1' := vec_index_ok _ _ _ hn1
      rw [cast_usize_u32, hci, hnd] at hn1'
      have hnn : n = nd := (Option.some.inj hn1').symm
      subst hnn
      have hkey : keyAt ns.val i = n.key.val := by simp [keyAt, hnd]
      by_cases heq : c = n.key
      · rw [if_pos heq] at hcf
        have := (Result.ok_injective hcf).symm
        subst this
        right
        refine ⟨by rw [hci]; exact hsub i (by simp [idxs]), by rw [hci, hkey, heq]⟩
      · rw [if_neg heq] at hcf
        have hne : c.val ≠ n.key.val := fun hh => heq (UScalar.eq_of_val_eq hh)
        by_cases hlt' : c < n.key
        · rw [if_pos hlt'] at hcf
          obtain ⟨n1, hn2, hcf⟩ := bind_inv hcf
          have hn2' := vec_index_ok _ _ _ hn2
          rw [cast_usize_u32, hci, hnd] at hn2'
          have hnn : n = n1 := Option.some.inj hn2'
          subst hnn
          have := (Result.ok_injective hcf).symm
          subst this
          have hcv : c.val < n.key.val := (UScalar.lt_equiv _ _).mp hlt'
          refine ⟨l, hl, hBl, fun j hj => hsub j (by simp [idxs, hj]), ?_⟩
          intro hex
          obtain ⟨j, hj, hjk⟩ := hfind hex
          simp only [idxs, List.mem_append, List.mem_cons] at hj
          rcases hj with hj | hj | hj
          · exact ⟨j, hj, hjk⟩
          · subst hj; rw [hkey] at hjk; exact absurd hjk.symm hne
          · have := hgt j hj; rw [hkey, hjk] at this; omega
        · rw [if_neg hlt'] at hcf
          obtain ⟨n1, hn2, hcf⟩ := bind_inv hcf
          have hn2' := vec_index_ok _ _ _ hn2
          rw [cast_usize_u32, hci, hnd] at hn2'
          have hnn : n = n1 := Option.some.inj hn2'
          subst hnn
          have := (Result.ok_injective hcf).symm
          subst this
          have hcv : ¬ c.val < n.key.val := fun hh => hlt' ((UScalar.lt_equiv _ _).mpr hh)
          refine ⟨rr, hr, hBr, fun j hj => hsub j (by simp [idxs, hj]), ?_⟩
          intro hex
          obtain ⟨j, hj, hjk⟩ := hfind hex
          simp only [idxs, List.mem_append, List.mem_cons] at hj
          rcases hj with hj | hj | hj
          · have := hlt j hj; rw [hkey, hjk] at this; omega
          · subst hj; rw [hkey] at hjk; exact absurd hjk.symm hne
          · exact ⟨j, hj, hjk⟩
  · rw [if_neg hn] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    left
    refine ⟨NIL_val, ?_⟩
    have hcur : cur = hashtrinity.overflow.NIL := UScalar.eq_of_val_eq (by simpa using hn)
    cases S with
    | leaf =>
      intro i hi hk
      obtain ⟨j, hj, _⟩ := hfind ⟨i, hi, hk⟩
      simp [idxs] at hj
    | node l i rr =>
      obtain ⟨hci, hi, _⟩ := hS
      rw [hcur, NIL_val] at hci
      exact absurd hci.symm hi

/-- In a search tree, distinct node positions carry distinct coordinates. -/
theorem bst_key_inj {ns : List ONode} : ∀ {T : OTree}, BST ns T →
    ∀ {i j : Nat}, i ∈ idxs T → j ∈ idxs T → i ≠ j → keyAt ns i ≠ keyAt ns j
  | .leaf, _, _, _, hi, _, _ => by simp [idxs] at hi
  | .node l k r, hB, i, j, hi, hj, hij => by
    obtain ⟨hBl, hBr, hlt, hgt⟩ := hB
    simp only [idxs, List.mem_append, List.mem_cons] at hi hj
    rcases hi with hi | hi | hi <;> rcases hj with hj | hj | hj
    · exact bst_key_inj hBl hi hj hij
    · subst hj; have := hlt i hi; omega
    · have := hlt i hi; have := hgt j hj; omega
    · subst hi; have := hlt j hj; omega
    · subst hi; subst hj; exact absurd rfl hij
    · subst hi; have := hgt j hj; omega
    · have := hgt i hi; have := hlt j hj; omega
    · subst hj; have := hgt i hi; omega
    · exact bst_key_inj hBr hi hj hij

/-- Every node of the tree holds the single entry `(key, some v)`. -/
def SingleEntries (ns : List ONode) (T : OTree) : Prop :=
  ∀ i ∈ idxs T, ∃ nd, ns[i]? = some nd ∧ ∃ v, nd.entries.val = [(nd.key, some v)]

/-- The value an overflow tree holds at coordinate `c`. -/
def OvHas (ns : List ONode) (T : OTree) (c : U64) (v : I64) : Prop :=
  ∃ i ∈ idxs T, ∃ nd, ns[i]? = some nd ∧ nd.key = c ∧ nd.entries.val = [(c, some v)]

theorem eqU64_eq (a b : U64) : core.cmp.EqU64.partialEqInst.eq a b = ok (decide (a = b)) := rfl

/-- `Overflow::get` on a search tree of single entries: `some v` exactly for the coordinates the tree
holds, with their value. -/
theorem ov_get_spec (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T) (hB : BST o.nodes.val T)
    (hS : SingleEntries o.nodes.val T) (c : U64) (r : Option I64)
    (h : hashtrinity.overflow.Overflow.get CI core.cmp.EqU64 o c c = ok r) :
    ∀ v, r = some v ↔ OvHas o.nodes.val T c v := by
  unfold hashtrinity.overflow.Overflow.get at h
  obtain ⟨fe, hfe, h⟩ := bind_inv h
  unfold hashtrinity.overflow.Overflow.find_entry hashtrinity.overflow.Overflow.find_node at hfe
  obtain ⟨n, hn, hfe⟩ := bind_inv hfe
  have hspec := find_node_spec o.nodes c T o.root hE hB n hn
  by_cases hnil : n = hashtrinity.overflow.NIL
  · rw [if_pos hnil] at hfe
    have := (Result.ok_injective hfe).symm
    subst this
    simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok,
      core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual] at h
    have := (Result.ok_injective h).symm
    subst this
    intro v
    simp only [reduceCtorEq, false_iff]
    rintro ⟨i, hi, nd, hnd, hk, _⟩
    rcases hspec with ⟨_, hno⟩ | ⟨hm, _⟩
    · exact hno i hi (by simp [keyAt, hnd, hk])
    · rw [(NIL_eq_iff n).mp hnil] at hm
      obtain ⟨_, _, hne⟩ := emb_mem hE _ hm
      exact hne rfl
  · rw [if_neg hnil] at hfe
    have hnv : n.val ≠ nilN := fun hh => hnil ((NIL_eq_iff n).mpr hh)
    rcases hspec with ⟨hm, _⟩ | ⟨hm, hk⟩
    · exact absurd hm hnv
    simp only [lift, bind_tc_ok] at hfe
    obtain ⟨n1, hn1, hfe⟩ := bind_inv hfe
    have hn1' := vec_index_ok _ _ _ hn1
    rw [cast_usize_u32] at hn1'
    obtain ⟨nd, hnd, v0, hent⟩ := hS n.val hm
    rw [hnd] at hn1'
    have hnn : n1 = nd := (Option.some.inj hn1').symm
    subst hnn
    have hkey : n1.key = c := by
      apply UScalar.eq_of_val_eq; simpa [keyAt, hnd] using hk
    -- the entry scan finds entry 0
    unfold hashtrinity.overflow.Overflow.find_entry_loop at hfe
    rw [loop.eq_def] at hfe
    obtain ⟨cf, hcf, hfe⟩ := bind_inv hfe
    unfold hashtrinity.overflow.Overflow.find_entry_loop.body at hcf
    have hlen : (alloc.vec.Vec.len n1.entries).val = 1 := by simp [hent]
    have h0 : (0#usize) < alloc.vec.Vec.len n1.entries := by
      rw [UScalar.lt_equiv, hlen]; simp
    rw [if_pos h0] at hcf
    obtain ⟨⟨t, w⟩, ht, hcf⟩ := bind_inv hcf
    have ht' := vec_index_ok _ _ _ ht
    rw [hent] at ht'
    simp at ht'
    obtain ⟨rfl, rfl⟩ := ht'
    obtain ⟨b, hb, hcf⟩ := bind_inv hcf
    rw [eqU64_eq, hkey] at hb
    have hb' : b = true := by
      have := Result.ok_injective hb; simp at this; exact this
    subst hb'
    rw [if_pos rfl] at hcf
    simp only [lift, bind_tc_ok] at hcf
    have := (Result.ok_injective hcf).symm
    subst this
    simp only [] at hfe
    have := (Result.ok_injective hfe).symm
    subst this
    simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok] at h
    simp only [lift, bind_tc_ok] at h
    obtain ⟨n2, hn2, h⟩ := bind_inv h
    have hn2' := vec_index_ok _ _ _ hn2
    rw [cast_usize_u32, hnd] at hn2'
    have hnn : n1 = n2 := Option.some.inj hn2'
    subst hnn
    obtain ⟨⟨t2, o1⟩, he2, h⟩ := bind_inv h
    have he2' := vec_index_ok _ _ _ he2
    rw [hent] at he2'
    have h00 : (UScalar.cast .Usize (UScalar.cast .U32 (0#usize))).val = 0 := by
      rw [cast_usize_u32, UScalar.cast_val_eq]; simp
    rw [h00] at he2'
    simp at he2'
    obtain ⟨rfl, rfl⟩ := he2'
    simp only [core.option.Option.as_ref] at h
    have := (Result.ok_injective h).symm
    subst this
    intro v
    constructor
    · intro hv
      have := Option.some.inj hv
      subst this
      exact ⟨n.val, hm, n1, hnd, hkey, by rw [hent, hkey]⟩
    · rintro ⟨i, hi, nd', hnd', hk', hent'⟩
      -- the node holding `c` is unique: the search tree has distinct coordinates on its nodes
      have hsame : i = n.val := by
        by_contra hne
        exact absurd (bst_key_inj hB hi hm hne) (by
          simp [keyAt, hnd, hnd', hk', hkey])
      subst hsame
      rw [hnd] at hnd'
      have hnn : n1 = nd' := Option.some.inj hnd'
      subst hnn
      rw [hent, hkey] at hent'
      simp at hent'
      rw [hent']

end Apeiron.MapExecuted
