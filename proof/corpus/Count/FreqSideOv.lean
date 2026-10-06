-- Row: `apeiron-hashdos-cap-count-MUBqkA7OYTvluKSIRdbjC` (APEIRON-HASHDOS-CAP-COUNT): the count face's side store and overflow branches of `insert_with`.
import ApeironWord
import Count.Helpers
import Count.FreqRep
import Count.OverflowOps
import Count.FreqAdd
open Aeneas Aeneas.Std Result apeiron_word

/-!
# The side store and overflow branches of the count face

`FreqSide::find` and the two side-store branches of `FreqLine::insert_with` (a new reserved
frequency pushed; an existing one merged), then the overflow branches (a blocked frequency
inserted at a fresh node, or merged into its node), each as a `LineBody` transformer.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive
open Apeiron.MapExecuted (ONode Ov OTree Emb BST OInv idxs nilN ND EntryOf keyAt FirstAt Present)

/-! ## `FreqSide::find` -/

theorem find_spec (sd : SideI) (coord : U64) (r : Option Usize)
    (h : hashtrinity.freq.FreqSide.find sd coord = ok r) :
    (r = none ∧ ∀ x ∈ sd.coords.val, x ≠ coord) ∨
    ∃ i : Usize, r = some i ∧ sd.coords.val[i.val]? = some coord := by
  unfold hashtrinity.freq.FreqSide.find at h
  unfold hashtrinity.freq.FreqSide.find_loop at h
  refine loop_pc (fun (i : Usize) => ∀ k, k < i.val → sd.coords.val[k]? ≠ some coord)
    (fun (y : Option Usize) => (y = none ∧ ∀ x ∈ sd.coords.val, x ≠ coord) ∨
      ∃ i : Usize, y = some i ∧ sd.coords.val[i.val]? = some coord)
    _ ?_ 0#usize (fun k hk => absurd hk (by simp)) r h
  intro i hi cf hcf
  unfold hashtrinity.freq.FreqSide.find_loop.body at hcf
  by_cases hlt : i < alloc.vec.Vec.len sd.coords
  · rw [if_pos hlt] at hcf
    obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
    have hget := Apeiron.MapExecuted.vec_index_ok _ _ _ hi2
    have hlt' : i.val < sd.coords.val.length := by
      have := (UScalar.lt_equiv i _).mp hlt
      rw [alloc.vec.Vec.len_val] at this
      exact this
    by_cases heq : i2 = coord
    · rw [if_pos heq] at hcf
      have hcf' : cf = ControlFlow.done (some i) := (Result.ok_injective hcf).symm
      subst hcf'
      exact Or.inr ⟨i, rfl, by rw [hget, heq]⟩
    · rw [if_neg heq] at hcf
      obtain ⟨i3, hi3, hcf⟩ := bind_inv hcf
      have hcf' : cf = ControlFlow.cont i3 := (Result.ok_injective hcf).symm
      subst hcf'
      have hi3v := (uadd_ok hi3).1
      have e1 : (1#usize).val = 1 := rfl
      rw [e1] at hi3v
      intro k hk
      by_cases hkk : k = i.val
      · subst hkk; rw [hget]; intro hh; exact heq (Option.some.inj hh)
      · exact hi k (by omega)
  · rw [if_neg hlt] at hcf
    have hcf' : cf = ControlFlow.done none := (Result.ok_injective hcf).symm
    subst hcf'
    refine Or.inl ⟨rfl, fun x hx => ?_⟩
    obtain ⟨k, hk⟩ := List.mem_iff_getElem?.mp hx
    have hlt' : ¬ i.val < sd.coords.val.length := by
      intro hh
      apply hlt
      rw [UScalar.lt_equiv, alloc.vec.Vec.len_val]
      exact hh
    have hk' : k < sd.coords.val.length := (List.getElem?_eq_some_iff.mp hk).1
    intro hxc
    exact hi k (by omega) (by rw [hk, hxc])

theorem nodup_idx_inj {α : Type} {l : List α} (hnd : l.Nodup) {i j : Nat} {a : α}
    (hi : l[i]? = some a) (hj : l[j]? = some a) : i = j := by
  obtain ⟨hi1, hi2⟩ := List.getElem?_eq_some_iff.mp hi
  obtain ⟨hj1, hj2⟩ := List.getElem?_eq_some_iff.mp hj
  exact (hnd.getElem_inj_iff).mp (hi2.trans hj2.symm)

theorem side_push_rep (sd sd' : SideI) (c c' : Nat → Nat) (f : U64) (new : Std.I64)
    (hS : SideRep sd c) (hf : Apeiron.MapExecuted.IsSent f) (hnf : ∀ x ∈ sd.coords.val, x ≠ f)
    (h1 : sd'.coords.val = sd.coords.val ++ [f]) (h2 : sd'.vals.val = sd.vals.val ++ [some new])
    (h3 : sd'.live.val = sd.live.val ++ [true])
    (hc'f : (c' f.val : Int) = new.val) (hpos : 0 < c' f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x) :
    SideRep sd' c' := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [h2, h1]; simp [hS.lv]
  · rw [h3, h1]; simp [hS.ll]
  · rw [h1, List.nodup_append]
    exact ⟨hS.nodup, List.nodup_singleton f, fun a ha b hb => by
      rw [List.mem_singleton] at hb; subst hb; exact hnf a ha⟩
  · intro i x hx
    rw [h1] at hx
    by_cases hi : i < sd.coords.val.length
    · rw [List.getElem?_append_left hi] at hx
      obtain ⟨hs, hl, v, hv, hcv, hp⟩ := hS.ent i x hx
      have hxne : x.val ≠ f.val := fun h => hnf x (List.mem_of_getElem? hx) (UScalar.eq_of_val_eq h)
      have hil : i < sd.live.val.length := by rw [hS.ll]; exact hi
      have hiv : i < sd.vals.val.length := by rw [hS.lv]; exact hi
      refine ⟨hs, ?_, v, ?_, ?_, ?_⟩
      · rw [h3, List.getElem?_append_left hil]; exact hl
      · rw [h2, List.getElem?_append_left hiv]; exact hv
      · rw [hc'o _ hxne]; exact hcv
      · rw [hc'o _ hxne]; exact hp
    · have hge : sd.coords.val.length ≤ i := by omega
      rw [List.getElem?_append_right hge] at hx
      have hi0 : i - sd.coords.val.length = 0 := by
        by_contra hne
        have : (([f] : List U64))[i - sd.coords.val.length]? = none :=
          List.getElem?_eq_none (by simp; omega)
        rw [this] at hx; exact absurd hx (by simp)
      rw [hi0] at hx
      simp only [List.getElem?_cons_zero, Option.some.injEq] at hx
      subst hx
      have hie : i = sd.coords.val.length := by omega
      refine ⟨hf, ?_, new, ?_, hc'f, hpos⟩
      · rw [h3, hie, ← hS.ll]; exact List.getElem?_concat_length
      · rw [h2, hie, ← hS.lv]; exact List.getElem?_concat_length

theorem side_update_rep (sd sd' : SideI) (c c' : Nat → Nat) (f : U64) (i : Nat) (merged : Std.I64)
    (hS : SideRep sd c) (hi : sd.coords.val[i]? = some f)
    (h1 : sd'.coords.val = sd.coords.val) (h2 : sd'.vals.val = sd.vals.val.set i (some merged))
    (h3 : sd'.live.val = sd.live.val)
    (hc'f : (c' f.val : Int) = merged.val) (hpos : 0 < c' f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x) :
    SideRep sd' c' := by
  have hilt : i < sd.vals.val.length := by
    rw [hS.lv]; exact (List.getElem?_eq_some_iff.mp hi).1
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [h2, h1, List.length_set]; exact hS.lv
  · rw [h3, h1]; exact hS.ll
  · rw [h1]; exact hS.nodup
  · intro i' x hx
    rw [h1] at hx
    obtain ⟨hs, hl, v, hv, hcv, hp⟩ := hS.ent i' x hx
    by_cases hii : i' = i
    · subst hii
      have hxf : x = f := by rw [hi] at hx; exact (Option.some.inj hx).symm
      subst hxf
      refine ⟨hs, by rw [h3]; exact hl, merged, ?_, hc'f, hpos⟩
      rw [h2]; exact List.getElem?_set_self hilt
    · have hxne : x.val ≠ f.val := fun h => hii (nodup_idx_inj hS.nodup hx (by
        have hxf : x = f := UScalar.eq_of_val_eq h
        rw [hxf]; exact hi))
      refine ⟨hs, by rw [h3]; exact hl, v, ?_, ?_, ?_⟩
      · rw [h2, List.getElem?_set_ne (Ne.symm hii)]; exact hv
      · rw [hc'o _ hxne]; exact hcv
      · rw [hc'o _ hxne]; exact hp

/-! ## Changes that leave the primary lanes alone -/

/-- `l'` has the primary lanes, geometry and length of `l` (only `side`/`overflow` may differ). -/
structure SameMain {V : Type} (l' l : hashtrinity.freq.FreqLine V) : Prop where
  slots : l'.slots = l.slots
  ctrl : l'.ctrl = l.ctrl
  bits : l'.bits = l.bits
  gshift : l'.gshift = l.gshift
  gmask : l'.gmask = l.gmask
  len : l'.len = l.len

theorem SameMain.foN {V : Type} {l' l : hashtrinity.freq.FreqLine V} (h : SameMain l' l) (i : Nat) :
    foN l' i = foN l i := foN_slots_eq l l' h.slots i

theorem SameMain.ctN {V : Type} {l' l : hashtrinity.freq.FreqLine V} (h : SameMain l' l) (i : Nat) :
    ctN l' i = ctN l i := ctN_ctrl_eq l l' h.ctrl i

theorem SameMain.liveSet {V : Type} {l' l : hashtrinity.freq.FreqLine V} (h : SameMain l' l) :
    liveSet l' = liveSet l := by
  ext i; rw [mem_liveSet, mem_liveSet, h.foN, h.slots]

theorem SameMain.geom {V : Type} {l' l : hashtrinity.freq.FreqLine V} (h : SameMain l' l)
    (hG : LineGeom l) : LineGeom l' :=
  geom_congr l l' hG h.foN h.ctN (by rw [h.slots]) (by rw [h.ctrl]) h.bits h.gshift h.gmask

theorem SameMain.blocked {l' l : Line} (h : SameMain l' l) (g : Nat) : Blocked l g → Blocked l' g := by
  intro hb
  refine ⟨fun j hj => ?_, fun i hi => ?_⟩
  · rw [h.bits]; intro k hk; rw [h.ctN]; exact hb.1 j hj k hk
  · rw [h.foN] at hi ⊢; exact hb.2 i hi

theorem SameMain.inMain {V : Type} {l' l : hashtrinity.freq.FreqLine V} (h : SameMain l' l) (x : Nat) :
    InMain l' x ↔ InMain l x := inMain_congr h.foN x

/-- Reserved frequencies are never in a primary lane. -/
theorem LineBody.not_inMain_sent {l : Line} {c : Nat → Nat} (hB : LineBody l c) {x : Nat}
    (hx : SentN x) : ¬ InMain l x := by
  rintro ⟨i, hi, hix⟩
  exact hB.nosent i hi (by rw [hix]; exact hx)

/-- A change confined to the side store. -/
theorem body_side_change (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (hsm : SameMain l' l)
    (hov : l'.overflow = l.overflow) (f : U64) (hf : Apeiron.MapExecuted.IsSent f)
    (hS' : SideRep l'.side c') (hc'o : ∀ x, x ≠ f.val → c' x = c x)
    (hin : InSide l' f.val)
    (hsup : ∀ x, InSide l x → InSide l' x) : LineBody l' c' := by
  have hfn : SentN f.val := (isSent_iff f).mp hf
  refine ⟨hsm.geom hB.geom, ?_, ?_, ?_, hS', ?_, ?_⟩
  · intro i hi; rw [hsm.foN] at hi ⊢; exact hB.nosent i hi
  · intro i hi
    rw [hsm.foN] at hi ⊢
    obtain ⟨v, hv, hcv, hp⟩ := hB.vals i hi
    have hne : foN l i ≠ f.val := fun h => hB.nosent i hi (by rw [h]; exact hfn)
    refine ⟨v, ?_, ?_, ?_⟩
    · unfold vlN at hv ⊢; rw [hsm.slots]; exact hv
    · rw [hc'o _ hne]; exact hcv
    · rw [hc'o _ hne]; exact hp
  · rw [hsm.len, hsm.liveSet]; exact hB.len_eq
  · intro o ho
    rw [hov] at ho
    obtain ⟨T, hO⟩ := hB.ov o ho
    refine ⟨T, hO.congr (fun j hj nd hnd hbl => hsm.blocked _ hbl) (fun j hj nd hnd hbl => ?_)⟩
    apply hc'o
    intro hk
    obtain ⟨nd', w, hnd', hent, hc, hpos, hns, hb⟩ := hO.2.2 j hj
    have e1 : nd' = nd := by rw [hnd] at hnd'; exact (Option.some.inj hnd').symm
    rw [e1] at hns
    exact hns ((isSent_iff nd.key).mpr (by rw [hk]; exact hfn))
  · intro x hxm hxs hxo
    by_cases hx : x = f.val
    · exact absurd (hx ▸ hin) hxs
    · rw [hc'o x hx]
      apply hB.absent x
      · exact fun h => hxm ((hsm.inMain x).mpr h)
      · exact fun h => hxs (hsup x h)
      · exact fun h => hxo ((inOv_congr hov x).mpr h)

/-! ## Changes confined to the overflow tree -/

/-- `x` is the coordinate of a node of the tree `T` in the arena of `o`. -/
def OvKeys (o : Ov) (T : OTree) (x : Nat) : Prop := ∃ j ∈ idxs T, keyAt o.nodes.val j = x

theorem inOv_iff_keys {l : Line} {o : Ov} {T : OTree} (ho : l.overflow = some o)
    (hE : Emb o.nodes.val o.root T) (x : Nat) : InOv l x ↔ OvKeys o T x := by
  constructor
  · rintro ⟨o1, T1, ho1, hE1, hk⟩
    rw [ho] at ho1
    have := Option.some.inj ho1
    subst this
    have := Apeiron.MapExecuted.emb_unique hE1 hE
    subst this
    exact hk
  · intro hk
    exact ⟨o, T, ho, hE, hk⟩

/-- The overflow store an `insert_with` works on: the existing one, or a fresh empty one. -/
theorem ov_start (l : Line) (c : Nat → Nat) (hB : LineBody l c) (ov : Ov)
    (hov : l.overflow = some ov ∨
      (l.overflow = none ∧ hashtrinity.overflow.Overflow.new I64 Apeiron.MapExecuted.CI = ok ov)) :
    ∃ T0, OvInfo l c ov T0 ∧ ∀ x, InOv l x ↔ OvKeys ov T0 x := by
  rcases hov with ho | ⟨hn, hnew⟩
  · obtain ⟨T, hO⟩ := hB.ov ov ho
    exact ⟨T, hO, inOv_iff_keys ho hO.1.1⟩
  · obtain ⟨hI, hOK, hn0⟩ := new_spec ov hnew
    refine ⟨.leaf, ⟨hI, hOK, fun j hj => by simp [idxs] at hj⟩, fun x => ?_⟩
    constructor
    · rintro ⟨o1, T1, ho1, _⟩
      rw [hn] at ho1; exact absurd ho1 (by simp)
    · rintro ⟨j, hj, _⟩
      simp [idxs] at hj

/-- A change confined to the overflow tree: one non-reserved frequency `f` (in no primary lane) gets
count `c' f`, every other count is unchanged, and the old overflow keys stay. -/
theorem body_ov_change (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (hsm : SameMain l' l)
    (hside : l'.side = l.side) (f : U64) (hfns : ¬ Apeiron.MapExecuted.IsSent f)
    (hfm : ¬ InMain l f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x)
    (o' : Ov) (T' : OTree) (ho' : l'.overflow = some o') (hO' : OvInfo l' c' o' T')
    (hin : InOv l' f.val) (hsup : ∀ x, InOv l x → InOv l' x) : LineBody l' c' := by
  have hfn : ¬ SentN f.val := fun h => hfns ((isSent_iff f).mpr h)
  refine ⟨hsm.geom hB.geom, ?_, ?_, ?_, ?_, fun o ho => ?_, ?_⟩
  · intro i hi; rw [hsm.foN] at hi ⊢; exact hB.nosent i hi
  · intro i hi
    rw [hsm.foN] at hi ⊢
    obtain ⟨v, hv, hcv, hp⟩ := hB.vals i hi
    have hne : foN l i ≠ f.val := fun h => hfm ⟨i, hi, h⟩
    refine ⟨v, ?_, ?_, ?_⟩
    · unfold vlN at hv ⊢; rw [hsm.slots]; exact hv
    · rw [hc'o _ hne]; exact hcv
    · rw [hc'o _ hne]; exact hp
  · rw [hsm.len, hsm.liveSet]; exact hB.len_eq
  · rw [hside]
    refine hB.side.congr (fun x hx => hc'o _ ?_)
    intro hxe
    have := hB.side.sent hx
    rw [isSent_iff] at this
    exact hfn (hxe ▸ this)
  · rw [ho'] at ho
    have := Option.some.inj ho
    subst this
    exact ⟨T', hO'⟩
  · intro x hxm hxs hxo
    by_cases hx : x = f.val
    · exact absurd (hx ▸ hin) hxo
    · rw [hc'o x hx]
      apply hB.absent x
      · exact fun h => hxm ((hsm.inMain x).mpr h)
      · exact fun h => hxs ((inSide_congr hside x).mpr h)
      · exact fun h => hxo (hsup x h)

/-- A blocked frequency absent from the overflow is inserted at a fresh node. -/
theorem body_ov_insert (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (hsm : SameMain l' l)
    (hside : l'.side = l.side) (f : U64) (v : Std.I64) (ov : Ov) (T0 : OTree)
    (hO0 : OvInfo l c ov T0) (hk0 : ∀ x, InOv l x ↔ OvKeys ov T0 x)
    (hfns : ¬ Apeiron.MapExecuted.IsSent f) (hblk : Blocked l f.val)
    (hv : (c' f.val : Int) = v.val) (hpos : 0 < c' f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x)
    (hcap : ov.nodes.val.length < nilN)
    (hnp : ¬ Present ov.nodes.val T0 f f)
    (nd : U32) (e : Usize) (o' : Ov)
    (hins : hashtrinity.overflow.Overflow.insert_new_at Apeiron.MapExecuted.CI ov f f v = ok ((nd, e), o'))
    (ho' : l'.overflow = some o') :
    LineBody l' c' ∧ o'.nodes.val.length ≤ ov.nodes.val.length + 1 := by
  obtain ⟨T', hI', hOK', hlen, hnodes, hold, j0, hj0, nd0, hnd0, hk0', hent0⟩ :=
    insert_ovok ov o' T0 f v nd e hO0.1 hO0.2.1 hcap hnp hins
  have habs := keys_absent_of ov T0 hO0.2.1 f hnp
  have hkeyne : ∀ j ∈ idxs T0, ∀ nd1, ov.nodes.val[j]? = some nd1 → nd1.key.val ≠ f.val := by
    intro j hj nd1 hnd1 h
    exact habs j hj nd1 hnd1 (UScalar.eq_of_val_eq h)
  have hOv' : OvInfo l' c' o' T' := by
    refine ⟨hI', hOK', fun j hj => ?_⟩
    obtain ⟨nd', hnd', hcase⟩ := hnodes j hj
    rcases hcase with ⟨nd1, hj1, hnd1, hkk, hent⟩ | ⟨hk, hent⟩
    · obtain ⟨nd2, w, hnd2, hent2, hc2, hp2, hns2, hb2⟩ := hO0.2.2 j hj1
      have e1 : nd2 = nd1 := by rw [hnd1] at hnd2; exact (Option.some.inj hnd2).symm
      rw [e1] at hent2 hc2 hp2 hns2 hb2
      have hne := hkeyne j hj1 nd1 hnd1
      rw [hkk] at hne hc2 hp2 hns2 hb2
      refine ⟨nd', w, hnd', ?_, ?_, ?_, hns2, hsm.blocked _ hb2⟩
      · rw [← hent, hent2, hkk]
      · rw [hc'o _ hne]; exact hc2
      · rw [hc'o _ hne]; exact hp2
    · refine ⟨nd', v, hnd', ?_, ?_, ?_, ?_, ?_⟩
      · rw [hent, hk]
      · rw [hk]; exact hv
      · rw [hk]; exact hpos
      · rw [hk]; exact hfns
      · rw [hk]; exact hsm.blocked _ hblk
  have hKiff : ∀ x, OvKeys o' T' x ↔ OvKeys ov T0 x ∨ x = f.val := by
    intro x
    constructor
    · rintro ⟨j, hj, hx⟩
      obtain ⟨nd', hnd', hcase⟩ := hnodes j hj
      rcases hcase with ⟨nd1, hj1, hnd1, hkk, _⟩ | ⟨hk, _⟩
      · left
        refine ⟨j, hj1, ?_⟩
        rw [Apeiron.MapExecuted.keyAt_get hnd1, hkk, ← Apeiron.MapExecuted.keyAt_get hnd']
        exact hx
      · right; rw [← hx, Apeiron.MapExecuted.keyAt_get hnd', hk]
    · rintro (⟨j, hj, hx⟩ | hx)
      · obtain ⟨j', hj', nd1, nd', hnd1, hnd', hkk, _⟩ := hold j hj
        refine ⟨j', hj', ?_⟩
        rw [Apeiron.MapExecuted.keyAt_get hnd', hkk, ← Apeiron.MapExecuted.keyAt_get hnd1, hx]
      · exact ⟨j0, hj0, by rw [hx, Apeiron.MapExecuted.keyAt_get hnd0, hk0']⟩
  refine ⟨body_ov_change l l' c c' hB hsm hside f hfns (fun ⟨i, hi, hix⟩ => hblk.2 i hi hix) hc'o
    o' T' ho' hOv' ?_ ?_, hlen⟩
  · exact (inOv_iff_keys ho' hI'.1 _).mpr ((hKiff _).mpr (Or.inr rfl))
  · intro x hx
    exact (inOv_iff_keys ho' hI'.1 x).mpr ((hKiff x).mpr (Or.inl ((hk0 x).mp hx)))

/-- A blocked frequency already in the overflow has its value replaced (`take_value`, then
`write_back` of the merged value). -/
theorem body_ov_update (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (hsm : SameMain l' l)
    (hside : l'.side = l.side) (f : U64) (ov : Ov) (T0 : OTree)
    (hO0 : OvInfo l c ov T0) (hk0 : ∀ x, InOv l x ↔ OvKeys ov T0 x)
    (n e : U32)
    (hfe : hashtrinity.overflow.Overflow.find_entry Apeiron.MapExecuted.CI core.cmp.EqU64 ov f f
      = ok (some (n, e)))
    (old merged : Std.I64) (o1 o2 : Ov)
    (htake : hashtrinity.overflow.Overflow.take_value Apeiron.MapExecuted.CI ov n e = ok (old, o1))
    (hwb : hashtrinity.overflow.Overflow.write_back Apeiron.MapExecuted.CI o1 n e merged = ok o2)
    (hmerge : merged.val = old.val + 1)
    (hc'f : c' f.val = c f.val + 1) (hc'o : ∀ x, x ≠ f.val → c' x = c x)
    (ho2 : l'.overflow = some o2) : LineBody l' c' ∧ o2.nodes.val.length = ov.nodes.val.length := by
  obtain ⟨hI0, hOK0, hv0⟩ := hO0
  have hlen : ∀ j ∈ idxs T0, ∀ nd, ov.nodes.val[j]? = some nd → nd.entries.val.length < 2 ^ 32 := by
    intro j hj nd hnd
    obtain ⟨nd1, hnd1, _, w, hw⟩ := hOK0 j hj
    have e1 : nd1 = nd := by rw [hnd] at hnd1; exact (Option.some.inj hnd1).symm
    rw [e1] at hw
    rw [hw]; simp
  have hspec := (Apeiron.MapExecuted.find_entry_spec ov T0 hI0.1 hI0.2.1 hlen f f _ hfe).2 n e
  obtain ⟨hn, nd, hnd, hkey, hfirst⟩ := hspec.mp rfl
  obtain ⟨nd1, w, hnd1, hent, hcw, hpos, hns, hblk⟩ := hv0 n.val hn
  have e1 : nd1 = nd := by rw [hnd] at hnd1; exact (Option.some.inj hnd1).symm
  rw [e1] at hent hcw hpos hns hblk
  rw [hkey] at hcw hpos hns hblk
  have he0 : e.val = 0 := by
    obtain ⟨⟨x, hx, _⟩, _⟩ := hfirst
    rw [hent] at hx
    by_contra hne
    rw [List.getElem?_eq_none (by simp; omega)] at hx
    exact absurd hx (by simp)
  obtain ⟨hold, ents, hents, hnodes2, hroot2, hfree2, hlen2⟩ :=
    update_spec ov o1 o2 n e old merged nd w hnd hent he0 htake hwb
  have hI2 : OInv o2 T0 := oinv_node_update ov o2 T0 hI0 n.val nd hnd ents hnodes2 hroot2 hfree2
  have hOK2 : NodeOK o2.nodes.val T0 :=
    nodeok_node_update ov.nodes.val o2.nodes.val T0 hOK0 n.val nd hnd ents hnodes2 merged hents
  have hlt : n.val < ov.nodes.val.length := (List.getElem?_eq_some_iff.mp hnd).1
  have hself : o2.nodes.val[n.val]? = some {nd with entries := ents} := by
    rw [hnodes2]; exact List.getElem?_set_self hlt
  have hne : ∀ j, j ≠ n.val → o2.nodes.val[j]? = ov.nodes.val[j]? := fun j hj => by
    rw [hnodes2]; exact List.getElem?_set_ne (Ne.symm hj)
  have hkeyAt : ∀ j, keyAt o2.nodes.val j = keyAt ov.nodes.val j := by
    intro j
    by_cases hj : j = n.val
    · subst hj
      rw [Apeiron.MapExecuted.keyAt_get hself, Apeiron.MapExecuted.keyAt_get hnd]
    · unfold keyAt; rw [hne j hj]
  have hmergeInt : (c' f.val : Int) = merged.val := by
    rw [hc'f, hmerge, hold]
    push_cast
    rw [hcw]
  have hOv2 : OvInfo l' c' o2 T0 := by
    refine ⟨hI2, hOK2, fun j hj => ?_⟩
    by_cases hjn : j = n.val
    · subst hjn
      refine ⟨{nd with entries := ents}, merged, hself, hents, ?_, ?_, ?_, ?_⟩
      · show (c' nd.key.val : Int) = merged.val
        rw [hkey]; exact hmergeInt
      · show 0 < c' nd.key.val
        rw [hkey, hc'f]; omega
      · show ¬ Apeiron.MapExecuted.IsSent nd.key
        rw [hkey]; exact hns
      · show Blocked l' nd.key.val
        rw [hkey]; exact hsm.blocked _ hblk
    · obtain ⟨nd2, w2, hnd2, hent2, hc2, hp2, hns2, hb2⟩ := hv0 j hj
      refine ⟨nd2, w2, by rw [hne j hjn]; exact hnd2, hent2, ?_, ?_, hns2, hsm.blocked _ hb2⟩
      · have hneq : nd2.key.val ≠ f.val := by
          intro h
          have hk1 := Apeiron.MapExecuted.bst_key_inj hI0.2.1 hj hn hjn
          apply hk1
          rw [Apeiron.MapExecuted.keyAt_get hnd2, Apeiron.MapExecuted.keyAt_get hnd, h, hkey]
        rw [hc'o _ hneq]; exact hc2
      · have hneq : nd2.key.val ≠ f.val := by
          intro h
          have hk1 := Apeiron.MapExecuted.bst_key_inj hI0.2.1 hj hn hjn
          apply hk1
          rw [Apeiron.MapExecuted.keyAt_get hnd2, Apeiron.MapExecuted.keyAt_get hnd, h, hkey]
        rw [hc'o _ hneq]; exact hp2
  have hfns : ¬ Apeiron.MapExecuted.IsSent f := hns
  refine ⟨body_ov_change l l' c c' hB hsm hside f hfns
    (fun ⟨i, hi, hix⟩ => hblk.2 i hi hix) hc'o o2 T0 ho2 hOv2 ?_ ?_, ?_⟩
  · exact (inOv_iff_keys ho2 hI2.1 _).mpr ⟨n.val, hn, by rw [hkeyAt, Apeiron.MapExecuted.keyAt_get hnd, hkey]⟩
  · intro x hx
    obtain ⟨j, hj, hjx⟩ := (hk0 x).mp hx
    exact (inOv_iff_keys ho2 hI2.1 x).mpr ⟨j, hj, by rw [hkeyAt]; exact hjx⟩
  · rw [hnodes2, List.length_set]

end Apeiron.CountAggregate
