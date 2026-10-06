-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-READ): the executed overflow tree's insert.
import Map.ExecutedOverflowDescend
open Aeneas Aeneas.Std Result

/-!
# The executed overflow tree: `find_entry` and `insert_new_at`

* `find_entry_none_iff`, `find_entry_spec`: `find_entry` returns `none` exactly when no node of the
  tree holds the coordinate with an entry of that key; it returns `some (n, e)` exactly when node `n`
  holds the coordinate and `e` is the first entry of that key.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

/-! ## `find_entry` -/

/-- The entry scan: the first entry of the node whose key is `k`. -/
theorem find_entry_loop_spec (k : U64) (n : U32) (entries : alloc.vec.Vec (U64 × Option I64))
    (e0 : Usize) (r : Option (U32 × U32))
    (h : hashtrinity.overflow.Overflow.find_entry_loop CI core.cmp.EqU64 k n entries e0 = ok r) :
    (r = none ∧ ∀ e', e0.val ≤ e' → ∀ x, entries.val[e']? = some x → x.1 ≠ k) ∨
    (∃ e : Usize, e0.val ≤ e.val ∧ r = some (n, UScalar.cast .U32 e) ∧
      ∃ x, entries.val[e.val]? = some x ∧ x.1 = k ∧
        ∀ e', e0.val ≤ e' → e' < e.val → ∀ y, entries.val[e']? = some y → y.1 ≠ k) := by
  unfold hashtrinity.overflow.Overflow.find_entry_loop at h
  refine loop_pc (fun (cur : Usize) => e0.val ≤ cur.val ∧
      ∀ e', e0.val ≤ e' → e' < cur.val → ∀ y, entries.val[e']? = some y → y.1 ≠ k)
    (fun r => (r = none ∧ ∀ e', e0.val ≤ e' → ∀ x, entries.val[e']? = some x → x.1 ≠ k) ∨
      (∃ e : Usize, e0.val ≤ e.val ∧ r = some (n, UScalar.cast .U32 e) ∧
        ∃ x, entries.val[e.val]? = some x ∧ x.1 = k ∧
          ∀ e', e0.val ≤ e' → e' < e.val → ∀ y, entries.val[e']? = some y → y.1 ≠ k)) _ ?_
    e0 ⟨le_refl _, fun e' h1 h2 => absurd h2 (by omega)⟩ r h
  intro cur ⟨hc0, hinv⟩ cf hcf
  unfold hashtrinity.overflow.Overflow.find_entry_loop.body at hcf
  have hlenv : (alloc.vec.Vec.len entries).val = entries.val.length := alloc.vec.Vec.len_val entries
  by_cases hlt : cur < alloc.vec.Vec.len entries
  · rw [if_pos hlt] at hcf
    have hlt' := (UScalar.lt_equiv _ _).mp hlt
    obtain ⟨⟨t, w⟩, ht, hcf⟩ := bind_inv hcf
    have ht' := vec_index_ok _ _ _ ht
    obtain ⟨b, hb, hcf⟩ := bind_inv hcf
    rw [eqU64_eq] at hb
    have hb' : decide (t = k) = b := Result.ok_injective hb
    by_cases hbb : b = true
    · rw [if_pos hbb] at hcf
      simp only [lift, bind_tc_ok] at hcf
      have := (Result.ok_injective hcf).symm
      subst this
      right
      refine ⟨cur, hc0, rfl, (t, w), ht', ?_, fun e' h1 h2 => hinv e' h1 h2⟩
      rw [← hb'] at hbb
      exact of_decide_eq_true hbb
    · rw [if_neg hbb] at hcf
      obtain ⟨e1, he1, hcf⟩ := bind_inv hcf
      have := (Result.ok_injective hcf).symm
      subst this
      have hadd := uadd_ok he1
      have hone : (1#usize).val = 1 := rfl
      refine ⟨by omega, fun e' h1 h2 y hy => ?_⟩
      by_cases hec : e' < cur.val
      · exact hinv e' h1 hec y hy
      · have : e' = cur.val := by omega
        subst this
        rw [ht'] at hy
        have := Option.some.inj hy
        subst this
        have hbf : b = false := by simpa using hbb
        rw [← hb'] at hbf
        exact of_decide_eq_false hbf
  · rw [if_neg hlt] at hcf
    have hlt' : ¬ cur.val < entries.val.length := fun hh =>
      hlt ((UScalar.lt_equiv _ _).mpr (by rw [hlenv]; exact hh))
    have := (Result.ok_injective hcf).symm
    subst this
    left
    refine ⟨rfl, fun e' h1 x hx => ?_⟩
    by_cases hec : e' < cur.val
    · exact hinv e' h1 hec x hx
    · exfalso
      have : entries.val.length ≤ e' := by omega
      rw [List.getElem?_eq_none this] at hx
      exact absurd hx (by simp)

/-- Some node of the tree holds the coordinate `c` with an entry whose key is `k`. -/
def Present (ns : List ONode) (T : OTree) (c k : U64) : Prop :=
  ∃ j ∈ idxs T, ∃ nd, ns[j]? = some nd ∧ nd.key = c ∧
    ∃ (e : Nat) (x : U64 × Option I64), nd.entries.val[e]? = some x ∧ x.1 = k

/-- `e` is the first entry of `nd` whose key is `k`. -/
def FirstAt (nd : ONode) (e : Nat) (k : U64) : Prop :=
  (∃ x : U64 × Option I64, nd.entries.val[e]? = some x ∧ x.1 = k) ∧
    ∀ e' < e, ∀ y : U64 × Option I64, nd.entries.val[e']? = some y → y.1 ≠ k

theorem firstAt_unique {nd : ONode} {e1 e2 : Nat} {k : U64} (h1 : FirstAt nd e1 k)
    (h2 : FirstAt nd e2 k) : e1 = e2 := by
  obtain ⟨⟨x1, hx1, hk1⟩, hf1⟩ := h1
  obtain ⟨⟨x2, hx2, hk2⟩, hf2⟩ := h2
  by_contra hne
  rcases Nat.lt_or_gt_of_ne hne with hlt | hlt
  · exact hf2 e1 hlt x1 hx1 hk1
  · exact hf1 e2 hlt x2 hx2 hk2

theorem find_entry_core (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hB : BST o.nodes.val T) (c k : U64) (r : Option (U32 × U32))
    (h : hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 o c k = ok r) :
    (r = none ∧ ¬ Present o.nodes.val T c k) ∨
    (∃ (n : U32) (e : Usize), r = some (n, UScalar.cast .U32 e) ∧ n.val ∈ idxs T ∧
      ∃ nd, o.nodes.val[n.val]? = some nd ∧ nd.key = c ∧ FirstAt nd e.val k) := by
  unfold hashtrinity.overflow.Overflow.find_entry at h
  obtain ⟨nn, hnn, h⟩ := bind_inv h
  have hspec := find_node_spec o.nodes c T o.root hE hB nn hnn
  by_cases hnil : nn = hashtrinity.overflow.NIL
  · rw [if_pos hnil] at h
    have := (Result.ok_injective h).symm
    subst this
    left
    refine ⟨rfl, ?_⟩
    rintro ⟨j, hj, nd, hnd, hk, _⟩
    rcases hspec with ⟨_, hno⟩ | ⟨hm, _⟩
    · exact hno j hj (by rw [keyAt_get hnd, hk])
    · rw [(NIL_eq_iff nn).mp hnil] at hm
      obtain ⟨_, _, hne⟩ := emb_mem hE _ hm
      exact hne rfl
  · rw [if_neg hnil] at h
    have hnv : nn.val ≠ nilN := fun hh => hnil ((NIL_eq_iff nn).mpr hh)
    rcases hspec with ⟨hm, _⟩ | ⟨hm, hk⟩
    · exact absurd hm hnv
    simp only [lift, bind_tc_ok] at h
    obtain ⟨n1, hn1, h⟩ := bind_inv h
    have hn1' := idx_ok hn1
    have hkey : n1.key = c := by
      apply UScalar.eq_of_val_eq
      rw [← keyAt_get hn1']; exact hk
    have hu0 : (0#usize).val = 0 := rfl
    rcases find_entry_loop_spec k nn n1.entries 0#usize r h with ⟨hr, hnone⟩ | ⟨e, _, hr, x, hx, hxk, hfirst⟩
    · left
      refine ⟨hr, ?_⟩
      rintro ⟨j, hj, nd', hnd', hk', e, x, hx, hxk⟩
      have hjn : j = nn.val := by
        by_contra hne
        exact bst_key_inj hB hj hm hne (by rw [keyAt_get hnd', keyAt_get hn1', hk', hkey])
      subst hjn
      rw [hn1'] at hnd'
      have := Option.some.inj hnd'
      subst this
      exact hnone e (by rw [hu0]; exact Nat.zero_le _) x hx hxk
    · right
      refine ⟨nn, e, hr, hm, n1, hn1', hkey, ⟨x, hx, hxk⟩, fun e' h1 y hy => ?_⟩
      exact hfirst e' (by rw [hu0]; exact Nat.zero_le _) h1 y hy

theorem cast_u32_val (e : Usize) (h : e.val < 2 ^ 32) : (UScalar.cast .U32 e).val = e.val := by
  rw [UScalar.cast_val_eq]
  exact Nat.mod_eq_of_lt h

/-- `find_entry` returns `none` exactly when no node holds the coordinate with an entry of key `k`. -/
theorem find_entry_none_iff (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hB : BST o.nodes.val T) (c k : U64) (r : Option (U32 × U32))
    (h : hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 o c k = ok r) :
    r = none ↔ ¬ Present o.nodes.val T c k := by
  rcases find_entry_core o T hE hB c k r h with ⟨hr, hnp⟩ | ⟨n, e, hr, hn, nd, hnd, hkey, hfirst⟩
  · exact ⟨fun _ => hnp, fun _ => hr⟩
  · refine ⟨fun hh => ?_, fun hnp => ?_⟩
    · rw [hr] at hh; exact absurd hh (by simp)
    · exact absurd ⟨n.val, hn, nd, hnd, hkey, e.val, hfirst.1⟩ hnp

/-- **`find_entry_spec`.**  With every entry list shorter than `2^32` (the `u32` entry index does not
wrap), `find_entry` returns `some (n, e)` exactly when node `n` of the tree holds the coordinate and
`e` is the first entry of key `k`; `none` exactly when no node does. -/
theorem find_entry_spec (o : Ov) (T : OTree) (hE : Emb o.nodes.val o.root T)
    (hB : BST o.nodes.val T)
    (hlen : ∀ j ∈ idxs T, ∀ nd, o.nodes.val[j]? = some nd → nd.entries.val.length < 2 ^ 32)
    (c k : U64) (r : Option (U32 × U32))
    (h : hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 o c k = ok r) :
    (r = none ↔ ¬ Present o.nodes.val T c k) ∧
    (∀ (n e : U32), r = some (n, e) ↔ n.val ∈ idxs T ∧ ∃ nd, o.nodes.val[n.val]? = some nd ∧
      nd.key = c ∧ FirstAt nd e.val k) := by
  refine ⟨find_entry_none_iff o T hE hB c k r h, fun n' e' => ?_⟩
  rcases find_entry_core o T hE hB c k r h with ⟨hr, hnp⟩ | ⟨n, e, hr, hn, nd, hnd, hkey, hfirst⟩
  · refine ⟨fun hh => by rw [hr] at hh; exact absurd hh (by simp), ?_⟩
    rintro ⟨hn', nd', hnd', hkey', hfirst'⟩
    exact absurd ⟨n'.val, hn', nd', hnd', hkey', e'.val, hfirst'.1⟩ hnp
  · have he : e.val < 2 ^ 32 := by
      obtain ⟨⟨x, hx, _⟩, _⟩ := hfirst
      have h1 := (List.getElem?_eq_some_iff.mp hx).1
      have h2 := hlen n.val hn nd hnd
      omega
    have hcv := cast_u32_val e he
    constructor
    · intro hh
      rw [hr] at hh
      have hh' := Option.some.inj hh
      simp only [Prod.mk.injEq] at hh'
      obtain ⟨rfl, rfl⟩ := hh'
      exact ⟨hn, nd, hnd, hkey, by rw [hcv]; exact hfirst⟩
    · rintro ⟨hn', nd', hnd', hkey', hfirst'⟩
      have hnn : n = n' := by
        apply UScalar.eq_of_val_eq
        by_contra hne
        exact bst_key_inj hB hn hn' hne (by rw [keyAt_get hnd, keyAt_get hnd', hkey, hkey'])
      subst hnn
      rw [hnd] at hnd'
      have := Option.some.inj hnd'
      subst this
      have hee : e.val = e'.val := firstAt_unique hfirst hfirst'
      rw [hr]
      congr 2
      apply UScalar.eq_of_val_eq
      rw [hcv, hee]

/-! ## `alloc` -/

theorem pop_spec {α : Type} (v v' : alloc.vec.Vec α) (x : Option α)
    (h : alloc.vec.Vec.pop Global v = ok (x, v')) :
    (v.val = [] ∧ x = none ∧ v'.val = []) ∨ (∃ y, v.val = v'.val ++ [y] ∧ x = some y) := by
  unfold alloc.vec.Vec.pop at h
  rcases List.eq_nil_or_concat v.val with hnil | ⟨l', y, hl⟩
  · left
    have hz : v.val.length = 0 := by rw [hnil]; rfl
    rw [dif_pos hz] at h
    have := Result.ok_injective h
    simp only [Prod.mk.injEq] at this
    obtain ⟨hx, hv⟩ := this
    exact ⟨hnil, hx.symm, by rw [← hv]; exact hnil⟩
  · right
    have hz : ¬ v.val.length = 0 := by rw [hl]; simp
    rw [dif_neg hz] at h
    have := Result.ok_injective h
    simp only [Prod.mk.injEq] at this
    obtain ⟨hx, hv⟩ := this
    have hv' : v'.val = l' := by
      rw [← hv]
      show Slice.val (Slice.from _ _) = l'
      rw [Slice.from_val, hl]; simp
    refine ⟨y, by rw [hv', hl]; simp, ?_⟩
    rw [← hx, hl]; simp

/-- `alloc`: a fresh node, from the free list when it is non-empty, else pushed at the end. -/
theorem alloc_spec (o o' : Ov) (T : OTree) (i : U32) (coord k : U64) (v : I64)
    (hE : Emb o.nodes.val o.root T)
    (hfn : (o.free.val.map (fun x => x.val)).Nodup)
    (hfl : ∀ j ∈ o.free.val, j.val < o.nodes.val.length ∧ j.val ∉ idxs T)
    (hcap : o.nodes.val.length < nilN)
    (h : hashtrinity.overflow.Overflow.alloc CI o coord k v = ok (i, o')) :
    ∃ ni, o'.nodes.val[i.val]? = some ni ∧ ni.key = coord ∧ ni.entries.val = [(k, some v)] ∧
      ni.alive = true ∧ ni.left.val = nilN ∧ ni.right.val = nilN ∧ ni.height.val = 1 ∧
      i.val ≠ nilN ∧ i.val ∉ idxs T ∧ i.val < o'.nodes.val.length ∧
      (∀ j, j ≠ i.val → o'.nodes.val[j]? = o.nodes.val[j]?) ∧
      o'.root = o.root ∧ o'.len = o.len ∧
      (o'.free.val.map (fun x => x.val)).Nodup ∧
      (∀ j ∈ o'.free.val, j.val < o'.nodes.val.length ∧ j.val ∉ idxs T ∧ j.val ≠ i.val) ∧
      o'.nodes.val.length ≤ o.nodes.val.length + 1 := by
  have hTlt : ∀ j ∈ idxs T, j < o.nodes.val.length := fun j hj => by
    obtain ⟨nd, hnd, _⟩ := emb_mem hE j hj
    exact (List.getElem?_eq_some_iff.mp hnd).1
  unfold hashtrinity.overflow.Overflow.alloc at h
  simp only [lift, bind_tc_ok] at h
  obtain ⟨⟨popped, v1⟩, hpop, h⟩ := bind_inv h
  rcases pop_spec _ _ _ hpop with ⟨hf0, hx, hv1⟩ | ⟨y, hfy, hx⟩
  · subst hx
    obtain ⟨v2, hpush, h⟩ := bind_inv h
    have hv2 := vec_push_ok _ _ _ hpush
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    have hi1v := usub_ok hi1
    have hlenv : (alloc.vec.Vec.len v2).val = v2.val.length := alloc.vec.Vec.len_val v2
    have hu1 : (1#usize).val = 1 := rfl
    have hlen2 : v2.val.length = o.nodes.val.length + 1 := by rw [hv2]; simp
    rw [hu1] at hi1v
    have hhf := Result.ok_injective h
    have hii : i = UScalar.cast .U32 i1 := (congrArg Prod.fst hhf).symm
    have ho : o' = { o with nodes := v2, free := v1 } := (congrArg Prod.snd hhf).symm
    have hi1n : i1.val = o.nodes.val.length := by omega
    have hcast : i.val = o.nodes.val.length := by
      rw [hii, UScalar.cast_val_eq, hi1n]
      apply Nat.mod_eq_of_lt
      have hn32 : UScalarTy.U32.numBits = 32 := rfl
      rw [hn32]; unfold nilN at hcap; omega
    subst ho
    obtain ⟨X, hX, hXk, hXe, hXa, hXl, hXr, hXh⟩ : ∃ X : ONode, v2.val = o.nodes.val ++ [X] ∧
        X.key = coord ∧ X.entries.val = [(k, some v)] ∧ X.alive = true ∧
        X.left = hashtrinity.overflow.NIL ∧ X.right = hashtrinity.overflow.NIL ∧ X.height = 1#u32 :=
      ⟨_, hv2, rfl, rfl, rfl, rfl, rfl, rfl⟩
    refine ⟨X, ?_, hXk, hXe, hXa, by rw [hXl]; exact NIL_val, by rw [hXr]; exact NIL_val,
      by rw [hXh]; exact u32_one, ?_, ?_, ?_, ?_, rfl, rfl, ?_, ?_, ?_⟩
    · dsimp only
      rw [hX, hcast]; exact List.getElem?_concat_length
    · rw [hcast]; omega
    · intro hm
      have := hTlt _ hm
      omega
    · dsimp only
      rw [hlen2]; omega
    · intro j hj
      dsimp only
      rw [hX]
      rcases Nat.lt_or_ge j o.nodes.val.length with h1 | h1
      · rw [List.getElem?_append_left h1]
      · rw [List.getElem?_eq_none (by simp; omega), List.getElem?_eq_none h1]
    · dsimp only
      rw [hv1]; exact List.nodup_nil
    · intro j hj
      dsimp only at hj
      rw [hv1] at hj
      exact absurd hj (by simp)
    · dsimp only
      rw [hlen2]
  · subst hx
    obtain ⟨⟨n0, back⟩, hn0, h⟩ := bind_inv h
    obtain ⟨hn0', hback⟩ := idxm_ok hn0
    have hhf := Result.ok_injective h
    have hii : y = i := congrArg Prod.fst hhf
    subst hii
    have hylt : y.val < o.nodes.val.length := (hfl y (by rw [hfy]; simp)).1
    have hyT : y.val ∉ idxs T := (hfl y (by rw [hfy]; simp)).2
    have hfn' : (v1.val.map (fun x => x.val)).Nodup ∧ [y.val].Nodup ∧
        ∀ a ∈ v1.val.map (fun x => x.val), ∀ b ∈ [y.val], a ≠ b := by
      have := hfn
      rw [hfy, List.map_append, List.nodup_append] at this
      simpa using this
    obtain ⟨X, ho, hXk, hXe, hXa, hXl, hXr, hXh⟩ : ∃ X : ONode,
        o' = { o with nodes := back X, free := v1 } ∧
        X.key = coord ∧ X.entries.val = [(k, some v)] ∧ X.alive = true ∧
        X.left = hashtrinity.overflow.NIL ∧ X.right = hashtrinity.overflow.NIL ∧ X.height = 1#u32 :=
      ⟨_, (congrArg Prod.snd hhf).symm, rfl, rfl, rfl, rfl, rfl, rfl⟩
    subst ho
    refine ⟨X, ?_, hXk, hXe, hXa, by rw [hXl]; exact NIL_val, by rw [hXr]; exact NIL_val,
      by rw [hXh]; exact u32_one, ?_, hyT, ?_, ?_, rfl, rfl, hfn'.1, ?_, ?_⟩
    · dsimp only
      rw [hback]; exact List.getElem?_set_self hylt
    · omega
    · dsimp only
      rw [hback]; simp [hylt]
    · intro j hj
      dsimp only
      rw [hback]
      exact List.getElem?_set_ne (Ne.symm hj)
    · intro j hj
      dsimp only at hj
      have hjf : j ∈ o.free.val := by rw [hfy]; exact List.mem_append_left _ hj
      refine ⟨?_, (hfl j hjf).2, ?_⟩
      · dsimp only
        rw [hback]; simp
        exact (hfl j hjf).1
      · intro e
        exact hfn'.2.2 j.val (List.mem_map.mpr ⟨j, hj, rfl⟩) y.val (by simp) e
    · dsimp only
      rw [hback]; simp

/-! ## `insert_node` -/

theorem deref_val {α : Type} (v : alloc.vec.Vec α) : (alloc.vec.Vec.deref v).val = v.val :=
  Slice.from_val _ _

theorem descK_congr {ns ns' : List ONode} {key : Nat} {G : List Fr}
    (hk : ∀ j, keyAt ns' j = keyAt ns j) (h : DescK ns key G) : DescK ns' key G :=
  fun g hg => by rw [hk]; exact h g hg

/-- `insert_node`: the fresh node `i` (key not in the tree) is linked at its coordinate position and
the path is rebalanced; the result is an AVL tree over the old indices and `i`. -/
theorem insert_node_spec (o o' : Ov) (T : OTree) (i : U32) (ni : ONode)
    (hE : Emb o.nodes.val o.root T) (hB : BST o.nodes.val T) (hND : ND T)
    (hH : HOK o.nodes.val T) (hBal : Bal T)
    (hi : i.val ∉ idxs T) (hii : i.val ≠ nilN) (hni : o.nodes.val[i.val]? = some ni)
    (hnl : ni.left.val = nilN) (hnr : ni.right.val = nilN) (hnh : ni.height.val = 1)
    (hkey : ∀ j ∈ idxs T, keyAt o.nodes.val j ≠ ni.key.val)
    (h : hashtrinity.overflow.Overflow.insert_node CI o i = ok o') :
    ∃ T', Emb o'.nodes.val o'.root T' ∧ BST o'.nodes.val T' ∧ ND T' ∧ HOK o'.nodes.val T' ∧
      Bal T' ∧ (idxs T').Perm (i.val :: idxs T) ∧ Rel (idxs T') o.nodes.val o'.nodes.val ∧
      o'.free = o.free ∧ o'.len = o.len := by
  unfold hashtrinity.overflow.Overflow.insert_node at h
  by_cases hroot : o.root = hashtrinity.overflow.NIL
  · rw [if_pos hroot] at h
    have ho := (Result.ok_injective h).symm
    subst ho
    have hT : T = .leaf := by
      cases T with
      | leaf => rfl
      | node l j r =>
        obtain ⟨hv, hj, _⟩ := hE
        exact absurd (by rw [← hv]; exact (NIL_eq_iff _).mp hroot) hj
    subst hT
    refine ⟨.node .leaf i.val .leaf, ?_, ?_, ?_, ?_, ?_, ?_, Rel.refl _ _, rfl, rfl⟩
    · exact ⟨rfl, hii, ni, hni, hnl, hnr⟩
    · exact ⟨trivial, trivial, fun j hj => by simp [idxs] at hj, fun j hj => by simp [idxs] at hj⟩
    · exact ⟨trivial, trivial, by simp [idxs], by simp [idxs], fun j hj => by simp [idxs] at hj⟩
    · exact ⟨trivial, trivial, ni, hni, by simp [ht, hnh]⟩
    · exact ⟨trivial, trivial, by simp [ht], by simp [ht]⟩
    · exact List.Perm.refl _
  · rw [if_neg hroot] at h
    simp only [lift, bind_tc_ok] at h
    obtain ⟨n, hn, h⟩ := bind_inv h
    have hn' := idx_ok hn
    rw [hni] at hn'
    have hnn : ni = n := Option.some.inj hn'
    subst hnn
    obtain ⟨⟨v, chain1⟩, hloop, h⟩ := bind_inv h
    have hroot' : o.root.val ≠ nilN := fun hh => hroot ((NIL_eq_iff _).mpr hh)
    have hpost := insert_node_loop_spec o.nodes i ni.key o.root T hND hkey
      (alloc.vec.Vec.with_capacity U32 64#usize) o.root
      ⟨[], T, rfl, trivial, rfl, hE, hroot', rfl, fun g hg => absurd hg (by simp)⟩ (v, chain1) hloop
    obtain ⟨g0, G, nd, hlenC, hchC, hDescC, hplugC, hnd, hv, hSpC⟩ := hpost
    have hlenC' : chain1.val.length = G.length + 1 := hlenC
    have hv' : v.val = o.nodes.val.set g0.a (setChild g0.d nd i) := hv
    have h2 : hashtrinity.overflow.Overflow.rebalance_chain_loop CI { o with nodes := v }
        (alloc.vec.Vec.deref chain1) (Slice.len (alloc.vec.Vec.deref chain1)) = ok o' := h
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
    obtain ⟨T', hE', hBal', hH', hB', hND', hU', hRel', hf', hl'⟩ :=
      rebalance_chain_loop_spec v.val _ o.free o.len (alloc.vec.Vec.deref chain1)
        { o with nodes := v } o' _ hinv h2
    refine ⟨T', hE', hB', hND', hH', hBal', ?_, ?_, hf', hl'⟩
    · have := idxs_plugI_perm (g0 :: G) (X := .leaf) (X' := .node .leaf i.val .leaf)
        (i := i.val) (List.Perm.refl _)
      rw [hplugC] at this
      rw [hU']; exact this
    · rw [hU']
      refine Rel.trans' (U := idxs (plugI (.node .leaf i.val .leaf) (g0 :: G))) hrelv hRel' ?_
        (fun j hj => hj)
      intro j hj
      rw [List.mem_singleton] at hj
      rw [hj]
      exact (mem_plugI (g0 :: G) _).mpr (Or.inr (by simp [spineIdx]))

/-! ## `insert_new_at` -/

/-- The representation invariant of the overflow store at tree shape `T`: the arena embeds `T`
as an AVL search tree with current heights, and the free list holds distinct arena slots that `T`
does not use. -/
def OInv (o : Ov) (T : OTree) : Prop :=
  Emb o.nodes.val o.root T ∧ BST o.nodes.val T ∧ ND T ∧ HOK o.nodes.val T ∧ Bal T ∧
  (o.free.val.map (fun x => x.val)).Nodup ∧
  (∀ j ∈ o.free.val, j.val < o.nodes.val.length ∧ j.val ∉ idxs T)

/-- The tree holds the entry `(k, w)` at coordinate `c`. -/
def EntryOf (ns : List ONode) (T : OTree) (c k : U64) (w : Option I64) : Prop :=
  ∃ j ∈ idxs T, ∃ nd, ns[j]? = some nd ∧ nd.key = c ∧ (k, w) ∈ nd.entries.val

theorem bst_congr' {ns ns' : List ONode} : ∀ {t : OTree},
    (∀ j ∈ idxs t, keyAt ns' j = keyAt ns j) → BST ns t → BST ns' t
  | .leaf, _, _ => trivial
  | .node l i r, hk, ⟨a, b, c, d⟩ =>
    ⟨bst_congr' (fun j hj => hk j (by simp [idxs, hj])) a,
     bst_congr' (fun j hj => hk j (by simp [idxs, hj])) b,
     fun j hj => by
       rw [hk j (by simp [idxs, hj]), hk i (by simp [idxs])]; exact c j hj,
     fun j hj => by
       rw [hk i (by simp [idxs]), hk j (by simp [idxs, hj])]; exact d j hj⟩

theorem emb_ptr_congr {ns ns' : List ONode} : ∀ {x : U32} {t : OTree}, Emb ns x t →
    (∀ j ∈ idxs t, ∀ nd, ns[j]? = some nd →
      ∃ nd', ns'[j]? = some nd' ∧ nd'.left = nd.left ∧ nd'.right = nd.right) → Emb ns' x t
  | _, .leaf, h, _ => h
  | x, .node l i r, h, hj => by
    obtain ⟨hx, hi, nd, hnd, hl, hr⟩ := h
    obtain ⟨nd', hnd', hl', hr'⟩ := hj i (by simp [idxs]) nd hnd
    refine ⟨hx, hi, nd', hnd', ?_, ?_⟩
    · rw [hl']; exact emb_ptr_congr hl (fun j hj' => hj j (by simp [idxs, hj']))
    · rw [hr']; exact emb_ptr_congr hr (fun j hj' => hj j (by simp [idxs, hj']))

theorem entryOf_ins {ns ns' : List ONode} {T T' : OTree} {i0 : Nat} {coord k : U64} {v : I64}
    (hperm : (idxs T').Perm (i0 :: idxs T))
    (hold : ∀ j ∈ idxs T, ∃ nd nd', ns[j]? = some nd ∧ ns'[j]? = some nd' ∧
      nd'.key = nd.key ∧ nd'.entries = nd.entries)
    (hnew : ∃ nd', ns'[i0]? = some nd' ∧ nd'.key = coord ∧ nd'.entries.val = [(k, some v)])
    (c k' : U64) (w : Option I64) :
    EntryOf ns' T' c k' w ↔ EntryOf ns T c k' w ∨ (c = coord ∧ k' = k ∧ w = some v) := by
  obtain ⟨nn, hnn, hnk, hne⟩ := hnew
  constructor
  · rintro ⟨j, hj, nd0, hnd0, hkc, hm⟩
    rcases List.mem_cons.mp (hperm.mem_iff.mp hj) with hj0 | hjT
    · subst hj0
      rw [hnn] at hnd0
      have := Option.some.inj hnd0
      subst this
      right
      rw [hne] at hm
      simp only [List.mem_singleton, Prod.mk.injEq] at hm
      exact ⟨by rw [← hkc, hnk], hm.1, hm.2⟩
    · left
      obtain ⟨nd, nd', hnd, hnd', hk1, he1⟩ := hold j hjT
      rw [hnd0] at hnd'
      have := Option.some.inj hnd'
      subst this
      exact ⟨j, hjT, nd, hnd, by rw [← hk1]; exact hkc, by rw [← he1]; exact hm⟩
  · rintro (⟨j, hj, nd, hnd, hkc, hm⟩ | ⟨hc, hk, hw⟩)
    · obtain ⟨nd1, nd', hnd1, hnd', hk1, he1⟩ := hold j hj
      rw [hnd] at hnd1
      have := Option.some.inj hnd1
      subst this
      exact ⟨j, hperm.mem_iff.mpr (List.mem_cons_of_mem _ hj), nd', hnd', by rw [hk1]; exact hkc,
        by rw [he1]; exact hm⟩
    · refine ⟨i0, hperm.mem_iff.mpr (List.mem_cons_self ..), nn, hnn, by rw [hnk, hc], ?_⟩
      rw [hne, hk, hw]; simp

theorem entryOf_push {ns ns' : List ONode} {T : OTree} {j0 : Nat} {coord k : U64} {v : I64}
    (hj0 : j0 ∈ idxs T) (hold : ∀ j ∈ idxs T, j ≠ j0 → ns'[j]? = ns[j]?)
    (hn : ∃ nd nd', ns[j0]? = some nd ∧ ns'[j0]? = some nd' ∧ nd.key = coord ∧ nd'.key = coord ∧
      nd'.entries.val = nd.entries.val ++ [(k, some v)])
    (c k' : U64) (w : Option I64) :
    EntryOf ns' T c k' w ↔ EntryOf ns T c k' w ∨ (c = coord ∧ k' = k ∧ w = some v) := by
  obtain ⟨nd, nd', hnd, hnd', hk, hk', he⟩ := hn
  constructor
  · rintro ⟨j, hj, nd0, hnd0, hkc, hm⟩
    by_cases hjj : j = j0
    · subst hjj
      rw [hnd'] at hnd0
      have := Option.some.inj hnd0
      subst this
      rw [he] at hm
      rcases List.mem_append.mp hm with hm | hm
      · left; exact ⟨j, hj, nd, hnd, by rw [hk, ← hkc, hk'], hm⟩
      · right
        simp only [List.mem_singleton, Prod.mk.injEq] at hm
        exact ⟨by rw [← hkc, hk'], hm.1, hm.2⟩
    · left
      exact ⟨j, hj, nd0, by rw [← hold j hj hjj]; exact hnd0, hkc, hm⟩
  · rintro (⟨j, hj, nd0, hnd0, hkc, hm⟩ | ⟨hc, hk1, hw⟩)
    · by_cases hjj : j = j0
      · subst hjj
        rw [hnd] at hnd0
        have := Option.some.inj hnd0
        subst this
        exact ⟨j, hj, nd', hnd', by rw [hk', ← hk]; exact hkc, by
          rw [he]; exact List.mem_append_left _ hm⟩
      · exact ⟨j, hj, nd0, by rw [hold j hj hjj]; exact hnd0, hkc, hm⟩
    · refine ⟨j0, hj0, nd', hnd', by rw [hk', hc], ?_⟩
      rw [he, hk1, hw]; simp

/-- **`insert_new_at_spec`.**  For a key not held under its coordinate (`habs`; this is
`find_entry_none_iff` for `find_entry = none`), `insert_new_at` leaves the store an AVL search tree
(`OInv`) over the old node indices, plus a fresh node when the coordinate was new; the entry set
gains exactly `(coord, k, some v)`, which sits at the returned `(node, entry)` as the first entry of
its key. -/
theorem insert_new_at_spec (o o' : Ov) (T : OTree) (coord k : U64) (v : I64) (nd : U32) (e : Usize)
    (hI : OInv o T) (hcap : o.nodes.val.length < nilN)
    (habs : ¬ Present o.nodes.val T coord k)
    (h : hashtrinity.overflow.Overflow.insert_new_at CI o coord k v = ok ((nd, e), o')) :
    ∃ T', OInv o' T' ∧ o'.len.val = o.len.val + 1 ∧ o'.nodes.val.length ≤ o.nodes.val.length + 1 ∧
      nd.val ∈ idxs T' ∧
      (∃ nde, o'.nodes.val[nd.val]? = some nde ∧ nde.key = coord ∧ FirstAt nde e.val k ∧
        nde.entries.val[e.val]? = some (k, some v)) ∧
      (∀ c k' w, EntryOf o'.nodes.val T' c k' w ↔
        EntryOf o.nodes.val T c k' w ∨ (c = coord ∧ k' = k ∧ w = some v)) ∧
      ((∀ j ∈ idxs T, keyAt o.nodes.val j ≠ coord.val) →
        (idxs T').Perm (nd.val :: idxs T) ∧
        (∃ nde, o'.nodes.val[nd.val]? = some nde ∧ nde.entries.val = [(k, some v)] ∧
          nde.alive = true) ∧
        ∀ j ∈ idxs T, ∀ a, o.nodes.val[j]? = some a → ∃ b, o'.nodes.val[j]? = some b ∧ sameD a b) := by
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
    have hlen1 : (alloc.vec.Vec.len nd'.entries).val = 1 := by rw [hlenv, hnent]; rfl
    have he0 : e1.val = 0 := by omega
    have hold : ∀ j ∈ idxs T, ∃ nd1 nd2, o.nodes.val[j]? = some nd1 ∧
        self3.nodes.val[j]? = some nd2 ∧ nd2.key = nd1.key ∧ nd2.entries = nd1.entries := by
      intro j hj
      obtain ⟨ndo, hndo, _⟩ := emb_mem hE j hj
      obtain ⟨nd'', hnd'', hs'', _⟩ := hrel3.2 j ndo (by rw [hagree j hj]; exact hndo)
      exact ⟨ndo, nd'', hndo, hnd'', hs''.1.symm, hs''.2.1.symm⟩
    refine ⟨T', ⟨hE3, hB3, hND3, hH3, hBal3, ?_, ?_⟩, ?_, ?_, ?_, ⟨nd', hnd', hnkey, ?_, ?_⟩, ?_, ?_⟩
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
    · exact hperm.mem_iff.mpr (List.mem_cons_self ..)
    · rw [he0]
      refine ⟨⟨(k, some v), by rw [hnent]; rfl, rfl⟩, fun e' h1 => absurd h1 (by omega)⟩
    · rw [he0, hnent]; rfl
    · exact entryOf_ins hperm hold ⟨nd', hnd', hnkey, hnent⟩
    · intro _
      refine ⟨hperm, ⟨nd', hnd', hnent, by rw [← hsd.2.2]; exact hnia⟩, fun j hj a ha => ?_⟩
      obtain ⟨b, hb, hsb, _⟩ := hrel3.2 j a (by rw [hagree j hj]; exact ha)
      exact ⟨b, hb, hsb⟩
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
    have hlv : (alloc.vec.Vec.len v1).val = v1.val.length := alloc.vec.Vec.len_val v1
    have he1v' : (alloc.vec.Vec.len v1).val = e1.val + 1 := he1v
    have hl2 : v1.val.length = n1.entries.val.length + 1 := by rw [hent]; simp
    have he : e1.val = n1.entries.val.length := by omega
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
    have hlast : v1.val[e1.val]? = some (k, some v) := by
      rw [hent, he]; exact List.getElem?_concat_length
    refine ⟨T, ⟨?_, ?_, hND, ?_, hBal, hfn, ?_⟩, ?_, ?_, hm, ⟨{n1 with entries := v1}, ?_, hkey, ?_, ?_⟩, ?_, ?_⟩
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
    · show (back {n1 with entries := v1}).val[n.val]? = some _
      rw [hnodes]; exact List.getElem?_set_self hlt
    · refine ⟨⟨(k, some v), hlast, rfl⟩, fun e' h1 y hy hk' => ?_⟩
      have hy' : v1.val[e']? = some y := hy
      rw [hent, List.getElem?_append_left (by omega)] at hy'
      exact habs ⟨n.val, hm, n1, hn1o, hkey, e', y, hy', hk'⟩
    · exact hlast
    · intro c k' w
      show EntryOf (back {n1 with entries := v1}).val T c k' w ↔ _
      rw [hnodes]
      exact entryOf_push hm (fun j _ hj => hne j hj)
        ⟨n1, {n1 with entries := v1}, hn1o, List.getElem?_set_self hlt, hkey, hkey, hent⟩ c k' w
    · intro hno
      exact absurd hk (hno n.val hm)

/-- `insert_new_at_spec` with the absence hypothesis stated as `find_entry = none`. -/
theorem insert_new_at_spec_fe (o o' : Ov) (T : OTree) (coord k : U64) (v : I64) (nd : U32)
    (e : Usize) (hI : OInv o T) (hcap : o.nodes.val.length < nilN)
    (hfe : hashtrinity.overflow.Overflow.find_entry CI core.cmp.EqU64 o coord k = ok none)
    (h : hashtrinity.overflow.Overflow.insert_new_at CI o coord k v = ok ((nd, e), o')) :
    ∃ T', OInv o' T' ∧ o'.len.val = o.len.val + 1 ∧ o'.nodes.val.length ≤ o.nodes.val.length + 1 ∧
      nd.val ∈ idxs T' ∧
      (∃ nde, o'.nodes.val[nd.val]? = some nde ∧ nde.key = coord ∧ FirstAt nde e.val k ∧
        nde.entries.val[e.val]? = some (k, some v)) ∧
      (∀ c k' w, EntryOf o'.nodes.val T' c k' w ↔
        EntryOf o.nodes.val T c k' w ∨ (c = coord ∧ k' = k ∧ w = some v)) ∧
      ((∀ j ∈ idxs T, keyAt o.nodes.val j ≠ coord.val) →
        (idxs T').Perm (nd.val :: idxs T) ∧
        (∃ nde, o'.nodes.val[nd.val]? = some nde ∧ nde.entries.val = [(k, some v)] ∧
          nde.alive = true) ∧
        ∀ j ∈ idxs T, ∀ a, o.nodes.val[j]? = some a → ∃ b, o'.nodes.val[j]? = some b ∧ sameD a b) :=
  insert_new_at_spec o o' T coord k v nd e hI hcap
    ((find_entry_none_iff o T hI.1 hI.2.1 coord k none hfe).mp rfl) h

end Apeiron.MapExecuted
