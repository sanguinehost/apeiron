-- The executed overflow tree's insert path.
import Map.ExecutedOverflowChain
open Aeneas Aeneas.Std Result

/-!
# The executed overflow tree: the descent of `insert_node`

`insert_node_loop` walks from the root to the empty child slot where the new node `i` belongs, pushing
every visited node on `chain`, and links `i` there.  `insert_node_loop_spec`: the visited nodes are
the ancestors of the slot (innermost last in the slice), the descent obeyed the coordinate comparison
at every ancestor (`DescK`), and the arena changed only in the child pointer of the last ancestor.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

theorem chainOf_snoc {ch : List U32} {x : U32} : ∀ (G : List Fr), ChainOf ch G → G.length ≤ ch.length →
    ChainOf (ch ++ [x]) G
  | [], _, _ => trivial
  | g :: rest, ⟨⟨c, hc, hcv⟩, h2⟩, hl => by
    have hlt : rest.length < ch.length := by simp only [List.length_cons] at hl; omega
    refine ⟨⟨c, ?_, hcv⟩, chainOf_snoc rest h2 (by omega)⟩
    rw [List.getElem?_append_left hlt]; exact hc

/-- The state of the descent: the path `G` above the slot holding `S`, rooted at `cur`. -/
def DescInv (L : List ONode) (root : U32) (T : OTree) (key : Nat) (chain : alloc.vec.Vec U32)
    (cur : U32) : Prop :=
  ∃ (G : List Fr) (S : OTree), chain.val.length = G.length ∧ ChainOf chain.val G ∧
    Spine L root cur G ∧ Emb L cur S ∧ cur.val ≠ nilN ∧ plugI S G = T ∧ DescK L key G

/-- The result of the descent: a non-empty path whose last ancestor now points at `i`. -/
def DescPost (L : List ONode) (root : U32) (T : OTree) (key : Nat) (i : U32)
    (r : alloc.vec.Vec ONode × alloc.vec.Vec U32) : Prop :=
  ∃ (g0 : Fr) (G : List Fr) (nd : ONode),
    r.2.val.length = G.length + 1 ∧ ChainOf r.2.val (g0 :: G) ∧ DescK L key (g0 :: G) ∧
    plugI .leaf (g0 :: G) = T ∧ L[g0.a]? = some nd ∧
    r.1.val = L.set g0.a (setChild g0.d nd i) ∧ Spine r.1.val root i (g0 :: G)

theorem insert_node_loop_spec (v : alloc.vec.Vec ONode) (i : U32) (key : U64) (root : U32)
    (T : OTree) (hND : ND T) (hkey : ∀ j ∈ idxs T, keyAt v.val j ≠ key.val)
    (chain : alloc.vec.Vec U32) (cur : U32)
    (hinv : DescInv v.val root T key.val chain cur)
    (r : alloc.vec.Vec ONode × alloc.vec.Vec U32)
    (h : hashtrinity.overflow.Overflow.insert_node_loop v i key chain cur = ok r) :
    DescPost v.val root T key.val i r := by
  unfold hashtrinity.overflow.Overflow.insert_node_loop at h
  refine loop_pc (fun (st : alloc.vec.Vec U32 × U32) => DescInv v.val root T key.val st.1 st.2)
    (fun r => DescPost v.val root T key.val i r) _ ?_ (chain, cur) hinv r h
  intro x hx cf hcf
  obtain ⟨chain, cur⟩ := x
  have hx' : DescInv v.val root T key.val chain cur := hx
  have hcf' : hashtrinity.overflow.Overflow.insert_node_loop.body v i key chain cur = ok cf := hcf
  unfold hashtrinity.overflow.Overflow.insert_node_loop.body at hcf'
  obtain ⟨G, S, hlen, hch, hSp, hES, hnil, hplug, hDesc⟩ := hx'
  obtain ⟨chain1, hpush, hcf'⟩ := bind_inv hcf'
  have hchain1 := vec_push_ok _ _ _ hpush
  simp only [lift, bind_tc_ok] at hcf'
  obtain ⟨n, hn, hcf'⟩ := bind_inv hcf'
  have hn' := idx_ok hn
  cases S with
  | leaf => exact absurd hES hnil
  | node Sl ii Sr =>
    obtain ⟨hcv, hii', nd, hnd, hEl, hEr⟩ := hES
    rw [hcv, hnd] at hn'
    have hnn : nd = n := Option.some.inj hn'
    subst hnn
    have hNDS : ND (.node Sl ii Sr) := nd_plugI_sub G (by rw [hplug]; exact hND)
    have hND' : ND (plugI (.node Sl ii Sr) G) := by rw [hplug]; exact hND
    have hdisj := nd_plugI_disj G hND'
    have hcurT : ii ∈ idxs T := by
      rw [← hplug]; exact (mem_plugI G _).mpr (Or.inl (by simp [idxs]))
    have hkeyc : nd.key.val ≠ key.val := by
      have := hkey ii hcurT
      rwa [keyAt_get hnd] at this
    have hch1 : ChainOf chain1.val ((⟨ii, true, Sr⟩ : Fr) :: G) ∧
        ChainOf chain1.val ((⟨ii, false, Sl⟩ : Fr) :: G) := by
      have hc2 := chainOf_snoc (x := cur) G hch (by omega)
      have hcur : chain1.val[G.length]? = some cur := by rw [hchain1, ← hlen]; simp
      exact ⟨⟨⟨cur, hcur, hcv⟩, by rw [hchain1]; exact hc2⟩,
        ⟨⟨cur, hcur, hcv⟩, by rw [hchain1]; exact hc2⟩⟩
    have hlen1 : chain1.val.length = G.length + 1 := by rw [hchain1]; simp [hlen]
    have hiiL : ii < v.val.length := (List.getElem?_eq_some_iff.mp hnd).1
    by_cases hlt : key < nd.key
    · have hlt' := (UScalar.lt_equiv key nd.key).mp hlt
      rw [if_pos hlt] at hcf'
      obtain ⟨n1, hn1, hcf'⟩ := bind_inv hcf'
      have hn1' := idx_ok hn1
      rw [hcv, hnd] at hn1'
      have hnn1 : nd = n1 := Option.some.inj hn1'
      subst hnn1
      have hDesc' : DescK v.val key.val ((⟨ii, true, Sr⟩ : Fr) :: G) := by
        intro g hg
        rcases List.mem_cons.mp hg with hg | hg
        · subst hg
          refine ⟨fun _ => ?_, fun hh => absurd hh (by simp)⟩
          show key.val < keyAt v.val ii
          rw [keyAt_get hnd]; exact hlt'
        · exact hDesc g hg
      by_cases hnl : nd.left = hashtrinity.overflow.NIL
      · rw [if_pos hnl] at hcf'
        obtain ⟨⟨n2, back⟩, hn2, hcf'⟩ := bind_inv hcf'
        obtain ⟨hn2', hback⟩ := idxm_ok hn2
        rw [hcv, hnd] at hn2'
        have hnn2 : nd = n2 := Option.some.inj hn2'
        subst hnn2
        have hcf2 := (Result.ok_injective hcf').symm
        subst hcf2
        have hSl : Sl = .leaf := by
          cases Sl with
          | leaf => rfl
          | node a b c =>
            obtain ⟨hv1, hv2, _⟩ := hEl
            exact absurd (by rw [← hv1]; exact (NIL_eq_iff _).mp hnl) hv2
        subst hSl
        have hv1 : (back {nd with left := i}).val = v.val.set ii (setChild true nd i) := by
          rw [hback, hcv]; rfl
        refine ⟨⟨ii, true, Sr⟩, G, nd, hlen1, hch1.1, hDesc', ?_, hnd, hv1, ?_⟩
        · rw [← hplug]; rfl
        · rw [hv1]
          refine ⟨cur, spine_congr root cur G (fun j hj => ?_) hSp, hcv, hii', setChild true nd i,
            List.getElem?_set_self hiiL, ?_, childOf_setChild _ _ _⟩
          · exact List.getElem?_set_ne (fun e => hdisj j hj (by rw [← e]; simp [idxs]))
          · show Emb _ nd.right Sr
            exact emb_congr hEr (fun j hj => List.getElem?_set_ne
              (show ii ≠ j from fun e => hNDS.2.2.2.1 (by rw [e]; exact hj)))
      · rw [if_neg hnl] at hcf'
        have hcf2 := (Result.ok_injective hcf').symm
        subst hcf2
        refine ⟨⟨ii, true, Sr⟩ :: G, Sl, hlen1, hch1.1, ?_, hEl, ?_, by rw [← hplug]; rfl, hDesc'⟩
        · exact ⟨cur, hSp, hcv, hii', nd, hnd, hEr, rfl⟩
        · exact fun hh => hnl ((NIL_eq_iff _).mpr hh)
    · have hge : nd.key.val ≤ key.val := by
        have := (not_congr (UScalar.lt_equiv key nd.key)).mp hlt
        omega
      have hlt' : nd.key.val < key.val := by omega
      rw [if_neg hlt] at hcf'
      obtain ⟨n1, hn1, hcf'⟩ := bind_inv hcf'
      have hn1' := idx_ok hn1
      rw [hcv, hnd] at hn1'
      have hnn1 : nd = n1 := Option.some.inj hn1'
      subst hnn1
      have hDesc' : DescK v.val key.val ((⟨ii, false, Sl⟩ : Fr) :: G) := by
        intro g hg
        rcases List.mem_cons.mp hg with hg | hg
        · subst hg
          refine ⟨fun hh => absurd hh (by simp), fun _ => ?_⟩
          show keyAt v.val ii < key.val
          rw [keyAt_get hnd]; exact hlt'
        · exact hDesc g hg
      by_cases hnr : nd.right = hashtrinity.overflow.NIL
      · rw [if_pos hnr] at hcf'
        obtain ⟨⟨n2, back⟩, hn2, hcf'⟩ := bind_inv hcf'
        obtain ⟨hn2', hback⟩ := idxm_ok hn2
        rw [hcv, hnd] at hn2'
        have hnn2 : nd = n2 := Option.some.inj hn2'
        subst hnn2
        have hcf2 := (Result.ok_injective hcf').symm
        subst hcf2
        have hSr : Sr = .leaf := by
          cases Sr with
          | leaf => rfl
          | node a b c =>
            obtain ⟨hv1, hv2, _⟩ := hEr
            exact absurd (by rw [← hv1]; exact (NIL_eq_iff _).mp hnr) hv2
        subst hSr
        have hv1 : (back {nd with right := i}).val = v.val.set ii (setChild false nd i) := by
          rw [hback, hcv]; rfl
        refine ⟨⟨ii, false, Sl⟩, G, nd, hlen1, hch1.2, hDesc', ?_, hnd, hv1, ?_⟩
        · rw [← hplug]; rfl
        · rw [hv1]
          refine ⟨cur, spine_congr root cur G (fun j hj => ?_) hSp, hcv, hii', setChild false nd i,
            List.getElem?_set_self hiiL, ?_, childOf_setChild _ _ _⟩
          · exact List.getElem?_set_ne (fun e => hdisj j hj (by rw [← e]; simp [idxs]))
          · show Emb _ nd.left Sl
            exact emb_congr hEl (fun j hj => List.getElem?_set_ne
              (show ii ≠ j from fun e => hNDS.2.2.1 (by rw [e]; exact hj)))
      · rw [if_neg hnr] at hcf'
        have hcf2 := (Result.ok_injective hcf').symm
        subst hcf2
        refine ⟨⟨ii, false, Sl⟩ :: G, Sr, hlen1, hch1.2, ?_, hEr, ?_, by rw [← hplug]; rfl, hDesc'⟩
        · exact ⟨cur, hSp, hcv, hii', nd, hnd, hEl, rfl⟩
        · exact fun hh => hnr ((NIL_eq_iff _).mpr hh)

end Apeiron.MapExecuted
