-- Row: `apeiron-hashdos-cap-count-MUBqkA7OYTvluKSIRdbjC` (APEIRON-HASHDOS-CAP-COUNT): `insert_with`, `add` and the representation of the capped count face.
import ApeironWord
import Count.Helpers
import Count.FreqRep
import Count.OverflowOps
import Count.FreqAdd
import Count.FreqSideOv
import Count.FreqGrow
open Aeneas Aeneas.Std Result apeiron_word

/-!
# `insert_with` and `add` for the capped count face

`insert_with_spec` covers all four branches of `FreqLine::insert_with`: a reserved frequency pushed
to or merged in the side store; a frequency merged at its primary lane; a blocked frequency merged
in, or inserted at a fresh node of, the overflow tree; a fresh frequency placed at a vacant lane
after at most one `grow`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive
open Apeiron.MapExecuted (ONode Ov OTree Emb BST OInv idxs nilN ND EntryOf keyAt FirstAt Present)

theorem list_set_get_self {α : Type} (l : List α) (i : Nat) (x : α) (h : l[i]? = some x) :
    l.set i x = l := by
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_set]
  by_cases hk : i = k
  · subst hk
    obtain ⟨hlt, heq⟩ := List.getElem?_eq_some_iff.mp h
    simp [hlt, heq]
  · simp [hk]

/-- The vectors of the side-store merge branch: `vals[i] := some merged`, `live` unchanged. -/
theorem side_upd_exec (sd : SideI) (i : Usize) (merged : Std.I64)
    (mu mu1 m2 : core.mem.maybe_uninit.MaybeUninit Std.I64)
    (s s2 : Slice (core.mem.maybe_uninit.MaybeUninit Std.I64))
    (dmb dmb1 : Slice (core.mem.maybe_uninit.MaybeUninit Std.I64) →
      alloc.vec.Vec (core.mem.maybe_uninit.MaybeUninit Std.I64))
    (gb gb1 : core.mem.maybe_uninit.MaybeUninit Std.I64 →
      Slice (core.mem.maybe_uninit.MaybeUninit Std.I64))
    (hdm : (lift (alloc.vec.Vec.deref_mut sd.vals) : Result _) = ok (s, dmb))
    (hg : core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice
      (core.mem.maybe_uninit.MaybeUninit Std.I64)) s i = ok (mu, gb))
    (hmu1 : core.mem.maybe_uninit.MaybeUninit.new merged = ok mu1)
    (hdm1 : (lift (alloc.vec.Vec.deref_mut (dmb (gb mu))) : Result _) = ok (s2, dmb1))
    (hg1 : core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice
      (core.mem.maybe_uninit.MaybeUninit Std.I64)) s2 i = ok (m2, gb1)) :
    (dmb1 (gb1 mu1)).val = sd.vals.val.set i.val (some merged) := by
  have e0 : alloc.vec.Vec.deref_mut sd.vals = (s, dmb) := Result.ok_injective hdm
  have hs : s = (alloc.vec.Vec.deref_mut sd.vals).1 := by rw [e0]
  have hdmb : dmb = (alloc.vec.Vec.deref_mut sd.vals).2 := by rw [e0]
  subst hs hdmb
  obtain ⟨hg0, hgb⟩ := get_unchecked_mut_ok _ _ _ _ hg
  have hmu1' : mu1 = some merged := (Result.ok_injective hmu1).symm
  have e1 : gb mu = (alloc.vec.Vec.deref_mut sd.vals).1 := by
    rw [hgb]; exact slice_set_same _ _ _ hg0
  rw [e1] at hdm1
  have e2 : alloc.vec.Vec.deref_mut ((alloc.vec.Vec.deref_mut sd.vals).2
      (alloc.vec.Vec.deref_mut sd.vals).1) = (s2, dmb1) := Result.ok_injective hdm1
  have hs2 : s2 = (alloc.vec.Vec.deref_mut ((alloc.vec.Vec.deref_mut sd.vals).2
      (alloc.vec.Vec.deref_mut sd.vals).1)).1 := by rw [e2]
  have hdmb1 : dmb1 = (alloc.vec.Vec.deref_mut ((alloc.vec.Vec.deref_mut sd.vals).2
      (alloc.vec.Vec.deref_mut sd.vals).1)).2 := by rw [e2]
  subst hs2 hdmb1
  obtain ⟨hg10, hgb1⟩ := get_unchecked_mut_ok _ _ _ _ hg1
  rw [hgb1, hmu1']
  show ((sd.vals.slice.set i (some merged))).val = _
  rw [Slice.set_val_eq]
  rfl

theorem side_live_exec (sd : SideI) (i : Usize) (b0 b1 : Bool) (imb imb1 : Bool → alloc.vec.Vec Bool)
    (hlive : sd.live.val[i.val]? = some true)
    (him : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice Bool) sd.live i = ok (b0, imb))
    (him1 : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice Bool) (imb false) i = ok (b1, imb1)) :
    (imb1 true).val = sd.live.val := by
  obtain ⟨h0, hb⟩ := Apeiron.MapExecuted.vec_index_mut_ok _ _ _ _ him
  obtain ⟨h1, hb1⟩ := Apeiron.MapExecuted.vec_index_mut_ok _ _ _ _ him1
  rw [hb1, Apeiron.MapExecuted.vec_set_val, hb, Apeiron.MapExecuted.vec_set_val, List.set_set]
  exact list_set_get_self _ _ _ hlive

/-- No overflow node is keyed by a reserved frequency. -/
theorem LineBody.not_inOv_sent {l : Line} {c : Nat → Nat} (hB : LineBody l c) {x : Nat}
    (hx : SentN x) : ¬ InOv l x := by
  rintro ⟨o, T, ho, hE, j, hj, hk⟩
  obtain ⟨T0, hO⟩ := hB.ov o ho
  have := Apeiron.MapExecuted.emb_unique hE hO.1.1
  subst this
  obtain ⟨nd, hnd, _⟩ := Apeiron.MapExecuted.emb_mem hE j hj
  obtain ⟨nd', w, hnd', _, _, _, hns, _⟩ := hO.2.2 j hj
  have e1 : nd' = nd := by rw [hnd] at hnd'; exact (Option.some.inj hnd').symm
  rw [e1] at hns
  rw [Apeiron.MapExecuted.keyAt_get hnd] at hk
  exact hns ((isSent_iff nd.key).mpr (by rw [hk]; exact hx))

/-- A new reserved frequency is pushed to the side store. -/
theorem side_new_spec (l l' : Line) (c : Nat → Nat) (hB : LineBody l c) (f : U64) (new : Std.I64)
    (hf : Apeiron.MapExecuted.IsSent f) (hnew : new.val = 1)
    (hfind : ∀ x ∈ l.side.coords.val, x ≠ f)
    (v v1 v2 : _) (mu : core.mem.maybe_uninit.MaybeUninit Std.I64)
    (hv : alloc.vec.Vec.push l.side.coords f = ok v)
    (hmu : core.mem.maybe_uninit.MaybeUninit.new new = ok mu)
    (hv1 : alloc.vec.Vec.push l.side.vals mu = ok v1)
    (hv2 : alloc.vec.Vec.push l.side.live true = ok v2)
    (hl' : l' = { l with side := { coords := v, vals := v1, live := v2 } }) :
    LineBody l' (fun x => if x = f.val then c x + 1 else c x) ∧ Pot l' = Pot l := by
  have hfn := (isSent_iff f).mp hf
  have hc0 : c f.val = 0 := by
    apply hB.absent f.val (hB.not_inMain_sent hfn) _ (hB.not_inOv_sent hfn)
    rintro ⟨x, hx, hxf⟩
    exact hfind x hx (UScalar.eq_of_val_eq hxf)
  have hmu' : mu = some new := (Result.ok_injective hmu).symm
  have hsm : SameMain l' l := by subst hl'; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  have hp1 := vec_push_ok _ _ _ hv
  have hp2 := vec_push_ok _ _ _ hv1
  have hp3 := vec_push_ok _ _ _ hv2
  subst hl'
  refine ⟨body_side_change l _ c _ hB hsm rfl f hf ?_ (fun x hx => by simp [hx]) ?_ ?_, rfl⟩
  · refine side_push_rep l.side _ c _ f new hB.side hf hfind hp1 (by rw [hp2, hmu']) hp3 ?_ ?_
      (fun x hx => by simp [hx])
    · simp [hc0, hnew]
    · simp [hc0]
  · exact ⟨f, by show f ∈ v.val; rw [hp1]; simp, rfl⟩
  · rintro x ⟨y, hy, hyx⟩
    exact ⟨y, by show y ∈ v.val; rw [hp1]; simp [hy], hyx⟩

/-- A reserved frequency already in the side store is merged in place. -/
theorem side_upd_spec (l l' : Line) (c : Nat → Nat) (hB : LineBody l c) (f : U64) (new : Std.I64)
    (hf : Apeiron.MapExecuted.IsSent f) (hnew : new.val = 1) (i : Usize)
    (hfind : l.side.coords.val[i.val]? = some f)
    (mu mu1 m2 : core.mem.maybe_uninit.MaybeUninit Std.I64) (old merged : Std.I64)
    (s s2 : Slice (core.mem.maybe_uninit.MaybeUninit Std.I64))
    (dmb dmb1 : Slice (core.mem.maybe_uninit.MaybeUninit Std.I64) →
      alloc.vec.Vec (core.mem.maybe_uninit.MaybeUninit Std.I64))
    (gb gb1 : core.mem.maybe_uninit.MaybeUninit Std.I64 →
      Slice (core.mem.maybe_uninit.MaybeUninit Std.I64))
    (b0 b1 : Bool) (imb imb1 : Bool → alloc.vec.Vec Bool)
    (hdm : (lift (alloc.vec.Vec.deref_mut l.side.vals) : Result _) = ok (s, dmb))
    (hg : core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice
      (core.mem.maybe_uninit.MaybeUninit Std.I64)) s i = ok (mu, gb))
    (hold : core.mem.maybe_uninit.MaybeUninit.assume_init_read mu = ok old)
    (him : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice Bool) l.side.live i
      = ok (b0, imb))
    (hmerge : merged.val = new.val + old.val)
    (hmu1 : core.mem.maybe_uninit.MaybeUninit.new merged = ok mu1)
    (hdm1 : (lift (alloc.vec.Vec.deref_mut (dmb (gb mu))) : Result _) = ok (s2, dmb1))
    (hg1 : core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice
      (core.mem.maybe_uninit.MaybeUninit Std.I64)) s2 i = ok (m2, gb1))
    (him1 : alloc.vec.Vec.index_mut (core.slice.index.SliceIndexUsizeSlice Bool) (imb false) i
      = ok (b1, imb1))
    (hl' : l' = { l with side := { l.side with vals := dmb1 (gb1 mu1), live := imb1 true } }) :
    LineBody l' (fun x => if x = f.val then c x + 1 else c x) ∧ Pot l' = Pot l := by
  have hfn := (isSent_iff f).mp hf
  obtain ⟨-, hlive, v, hv, hcv, hpos⟩ := hB.side.ent i.val f hfind
  have hvals := side_upd_exec l.side i merged mu mu1 m2 s s2 dmb dmb1 gb gb1 hdm hg hmu1 hdm1 hg1
  have hlivev := side_live_exec l.side i b0 b1 imb imb1 hlive him him1
  -- the value read is the stored one
  have e0 : (alloc.vec.Vec.deref_mut l.side.vals) = (s, dmb) := Result.ok_injective hdm
  have hs : s = (alloc.vec.Vec.deref_mut l.side.vals).1 := by rw [e0]
  obtain ⟨hg0, -⟩ := get_unchecked_mut_ok _ _ _ _ hg
  have hmu : mu = some v := by
    rw [hs] at hg0
    have : l.side.vals.val[i.val]? = some mu := hg0
    rw [hv] at this
    exact (Option.some.inj this).symm
  have hold' : old = v := by
    have := assume_init_read_ok _ _ hold
    rw [hmu] at this
    exact (Option.some.inj this).symm
  subst hold'
  have hmu1' : mu1 = some merged := (Result.ok_injective hmu1).symm
  have hsm : SameMain l' l := by subst hl'; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  have hmc : ((c f.val + 1 : Nat) : Int) = merged.val := by
    rw [hmerge, hnew]; push_cast; rw [hcv]; ring
  subst hl'
  refine ⟨body_side_change l _ c _ hB hsm rfl f hf ?_ (fun x hx => by simp [hx]) ?_ ?_, rfl⟩
  · refine side_update_rep l.side _ c _ f i.val merged hB.side hfind rfl ?_ ?_ ?_ ?_
      (fun x hx => by simp [hx])
    · exact hvals
    · show (imb1 true).val = _
      exact hlivev
    · simpa using hmc
    · simp
  · exact ⟨f, by show f ∈ l.side.coords.val; exact List.mem_of_getElem? hfind, rfl⟩
  · exact fun x hx => hx

theorem iadd_ok {ty} {x y z : IScalar ty} (h : x + y = ok z) :
    z.val = x.val + y.val := by
  have := IScalar.add_equiv x y
  simp only [h, Result.match.ok] at this
  exact this.2.1

abbrev AddInst (inst : hashtrinity.algebra.Coord U64) :=
  hashtrinity.freq.FreqMapKI64.add.closure.Insts.CoreOpsFunctionFnOncePairI64I64I64 inst

theorem insert_with_spec (inst : hashtrinity.algebra.Coord U64) (l : Line) (c : Nat → Nat)
    (hR : LineRep l c) (f : U64) (new : Std.I64) (hnew : new.val = 1) (hP : Pot l < nilN)
    (l' : Line)
    (h : hashtrinity.freq.FreqLine.insert_with (AddInst inst) l f new () = ok l') :
    LineRep l' (fun x => if x = f.val then c x + 1 else c x) ∧ Pot l' ≤ Pot l + 1 := by
  have hB : LineBody l c := hR
  unfold hashtrinity.freq.FreqLine.insert_with at h
  obtain ⟨b, hb, h⟩ := bind_inv h
  rw [Apeiron.MapExecuted.is_sentinel_ok] at hb
  have hbd : b = decide (Apeiron.MapExecuted.IsSent f) := (Result.ok_injective hb).symm
  by_cases hs : Apeiron.MapExecuted.IsSent f
  · have hbt : b = true := by simp [hbd, hs]
    rw [if_pos hbt] at h
    obtain ⟨o, ho, h⟩ := bind_inv h
    rcases find_spec _ _ _ ho with ⟨hon, hnf⟩ | ⟨i, hoi, hfi⟩
    · subst hon
      dsimp only at h
      obtain ⟨v, hv, h⟩ := bind_inv h
      obtain ⟨mu, hmu, h⟩ := bind_inv h
      obtain ⟨v1, hv1, h⟩ := bind_inv h
      obtain ⟨v2, hv2, h⟩ := bind_inv h
      have hl' : l' = { l with side := { coords := v, vals := v1, live := v2 } } :=
        (Result.ok_injective h).symm
      obtain ⟨hbody, hpot⟩ := side_new_spec l l' c hB f new hs hnew hnf v v1 v2 mu hv hmu hv1 hv2 hl'
      exact ⟨hbody, by omega⟩
    · subst hoi
      dsimp only at h
      obtain ⟨⟨s, dmb⟩, hdm, h⟩ := bind_inv h
      obtain ⟨⟨mu, gb⟩, hg, h⟩ := bind_inv h
      obtain ⟨old, hold, h⟩ := bind_inv h
      obtain ⟨⟨b0, imb⟩, him, h⟩ := bind_inv h
      obtain ⟨merged, hmerge, h⟩ := bind_inv h
      obtain ⟨mu1, hmu1, h⟩ := bind_inv h
      obtain ⟨⟨s2, dmb1⟩, hdm1, h⟩ := bind_inv h
      obtain ⟨⟨m2, gb1⟩, hg1, h⟩ := bind_inv h
      obtain ⟨⟨b1, imb1⟩, him1, h⟩ := bind_inv h
      have hl' : l' = { l with side := { l.side with vals := dmb1 (gb1 mu1), live := imb1 true } } :=
        (Result.ok_injective h).symm
      have hmerge' : merged.val = new.val + old.val := by
        have hh : new + old = ok merged := hmerge
        exact iadd_ok hh
      obtain ⟨hbody, hpot⟩ := side_upd_spec l l' c hB f new hs hnew i hfi mu mu1 m2 old merged s s2 dmb dmb1
        gb gb1 b0 b1 imb imb1 hdm hg hold him hmerge' hmu1 hdm1 hg1 him1 hl'
      exact ⟨hbody, by omega⟩
  · have hbf : b = false := by simp [hbd, hs]
    rw [hbf] at h
    simp only [Bool.false_eq_true, if_false] at h
    have hfn' : ¬ SentN f.val := fun hh => hs ((isSent_iff f).mpr hh)
    obtain ⟨i, hi, h⟩ := bind_inv h
    obtain ⟨i1, hi1, h⟩ := bind_inv h
    obtain ⟨i3, hi3, h⟩ := bind_inv h
    obtain ⟨self1, hself1, h⟩ := bind_inv h
    have hS1 : LineBody self1 c ∧ Pot self1 ≤ Pot l := by
      by_cases hge : i1 >= i3
      · rw [if_pos hge] at hself1
        obtain ⟨hgb, -, hgp⟩ := grow_spec l self1 c hB hP hself1
        exact ⟨hgb, hgp⟩
      · rw [if_neg hge] at hself1
        have := Result.ok_injective hself1
        subst this
        exact ⟨hB, le_refl _⟩
    obtain ⟨hB1, hP1'⟩ := hS1
    have hP1 : Pot self1 < nilN := by omega
    obtain ⟨⟨i4, found, blocked, tag⟩, hprobe, h⟩ := bind_inv h
    obtain ⟨htagv, hpost⟩ := probe_tagged_spec self1 hB1.geom f i4 found blocked tag hprobe
    change (if found = true then _ else _) = _ at h
    by_cases hfound : found = true
    · rw [if_pos hfound] at h
      obtain ⟨⟨⟨i5, mu⟩, back⟩, hgm, h⟩ := bind_inv h
      obtain ⟨hgm1, hgm2⟩ := get_unchecked_mut_ok _ _ _ _ hgm
      obtain ⟨old, hold, h⟩ := bind_inv h
      have hmu : mu = some old := assume_init_read_ok _ _ hold
      obtain ⟨bd, hbd', h⟩ := bind_inv h
      have hbdf : bd = false := by
        rw [needs_drop_i64] at hbd'; exact (Result.ok_injective hbd').symm
      change (if bd = true then _ else _) = _ at h
      rw [if_neg (by simp [hbdf])] at h
      obtain ⟨t, ht, h⟩ := bind_inv h
      have htv := iadd_ok (show new + old = ok t from ht)
      obtain ⟨mu1, hmu1, h⟩ := bind_inv h
      have hmu1' : mu1 = some t := (Result.ok_injective hmu1).symm
      have hl' : l' = { self1 with slots := back (i5, mu1) } := (Result.ok_injective h).symm
      obtain ⟨-, hfo, hfne⟩ := hpost.1 hfound
      have hfoi5 : foN self1 i4.val = i5.val := foN_of_get _ _ _ hgm1
      have hi5f : i5 = f := UScalar.eq_of_val_eq (by rw [← hfoi5, hfo])
      obtain ⟨v, hv, hcv, hpv⟩ := hB1.vals i4.val hfne
      have hvl : vlN self1 i4.val = mu := by unfold vlN; rw [hgm1]
      have hvo : v = old := by
        rw [hvl, hmu] at hv; exact (Option.some_injective _ hv).symm
      subst hvo
      rw [hfo] at hcv hpv
      have hsl' : l'.slots = self1.slots.set i4 (f, some t) := by
        rw [hl']; show back (i5, mu1) = _; rw [hgm2, hmu1', hi5f]
      have hc'f : ((c f.val + 1 : Nat) : Int) = t.val := by
        rw [htv, hnew]; push_cast; rw [hcv]; ring
      have hbody := body_update self1 l' c (fun x => if x = f.val then c x + 1 else c x) hB1
        f i4 hfo (fun hh => hfn' (Or.inl hh)) t hsl' (by rw [hl']) (by rw [hl']) (by rw [hl'])
        (by rw [hl']) (by rw [hl']) (by rw [hl']) (by rw [hl'])
        (by simpa using hc'f) (by simp) (fun x hx => by simp [hx])
      refine ⟨hbody, ?_⟩
      have : Pot l' = Pot self1 := by rw [hl']; rfl
      omega
    · rw [if_neg hfound] at h
      have hfalse : found = false := by simpa using hfound
      rw [hfalse] at hpost
      change (if blocked = true then _ else _) = _ at h
      by_cases hblk : blocked = true
      · rw [if_pos hblk] at h
        obtain ⟨hf0, hblk1, hnotin⟩ := hpost.2.2 hblk
        have hblk' : Blocked self1 f.val := ⟨hblk1, hnotin⟩
        obtain ⟨o, ho, h⟩ := bind_inv h
        have hovx : ∃ ov, o = some ov ∧ (self1.overflow = some ov ∨
            (self1.overflow = none ∧ hashtrinity.overflow.Overflow.new I64 Apeiron.MapExecuted.CI = ok ov)) := by
          rcases hov : self1.overflow with _ | ov0
          · simp only [hov, core.option.Option.is_none, Option.isNone_none, if_true] at ho
            obtain ⟨o1, hnew', ho⟩ := bind_inv ho
            exact ⟨o1, (Result.ok_injective ho).symm, Or.inr ⟨rfl, hnew'⟩⟩
          · simp only [hov, core.option.Option.is_none, Option.isNone_some, Bool.false_eq_true, if_false] at ho
            exact ⟨ov0, (Result.ok_injective ho).symm, Or.inl rfl⟩
        obtain ⟨ov, hoo, hov'⟩ := hovx
        subst hoo
        obtain ⟨⟨o1, back⟩, hasm, h⟩ := bind_inv h
        have hasm' : o1 = some ov ∧ back = fun y => y := by
          rw [as_mut_ok] at hasm
          have := Result.ok_injective hasm
          simp only [Prod.mk.injEq] at this
          exact ⟨this.1.symm, this.2.symm⟩
        obtain ⟨rfl, rfl⟩ := hasm'
        try dsimp only at h
        obtain ⟨T0, hO0, hk0⟩ := ov_start self1 c hB1 ov hov'
        have hlenov : ov.nodes.val.length < nilN := by
          rcases hov' with ho | ⟨hn, hnew'⟩
          · have : ovArena self1 = ov.nodes.val.length := by unfold ovArena; rw [ho]
            have : ovArena self1 ≤ Pot self1 := by unfold Pot; omega
            omega
          · obtain ⟨_, _, hn0⟩ := new_spec ov hnew'
            rw [hn0]; unfold nilN; norm_num
        have hlenov2 : ov.nodes.val.length ≤ ovArena self1 := by
          rcases hov' with ho | ⟨hn, hnew'⟩
          · unfold ovArena; rw [ho]
          · obtain ⟨_, _, hn0⟩ := new_spec ov hnew'
            rw [hn0]; exact Nat.zero_le _
        obtain ⟨o2, hfe, h⟩ := bind_inv h
        have hsm : ∀ ov' : Ov, SameMain ({ self1 with overflow := some ov' } : Line) self1 :=
          fun ov' => ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
        cases o2 with
        | none =>
          try dsimp only at h
          obtain ⟨⟨⟨nd, e⟩, ov1⟩, hins, h⟩ := bind_inv h
          have hl' : l' = { self1 with overflow := some ov1 } := (Result.ok_injective h).symm
          have hnp : ¬ Present ov.nodes.val T0 f f :=
            (Apeiron.MapExecuted.find_entry_none_iff ov T0 hO0.1.1 hO0.1.2.1 f f none hfe).mp rfl
          have hc0 : c f.val = 0 := by
            apply hB1.absent
            · rintro ⟨i, hi, hix⟩; exact hnotin i hi hix
            · rintro ⟨y, hy, hyf⟩
              have := hB1.side.sent hy
              rw [isSent_iff] at this
              exact hfn' (hyf ▸ this)
            · intro hio
              obtain ⟨j, hj, hk⟩ := (hk0 f.val).mp hio
              obtain ⟨nd1, hnd1, _⟩ := Apeiron.MapExecuted.emb_mem hO0.1.1 j hj
              apply keys_absent_of ov T0 hO0.2.1 f hnp j hj nd1 hnd1
              apply UScalar.eq_of_val_eq
              rw [← Apeiron.MapExecuted.keyAt_get hnd1]; exact hk
          obtain ⟨hbody, harena⟩ := body_ov_insert self1 l' c (fun x => if x = f.val then c x + 1 else c x)
            hB1 (by subst hl'; exact hsm ov1) (by subst hl'; rfl) f new ov T0 hO0 hk0 hs hblk'
            (by simp [hc0, hnew]) (by simp) (fun x hx => by simp [hx]) hlenov hnp nd e ov1 hins
            (by subst hl'; rfl)
          refine ⟨hbody, ?_⟩
          have hPl' : Pot l' = self1.len.val + ov1.nodes.val.length := by subst hl'; rfl
          have hPself : Pot self1 = self1.len.val + ovArena self1 := rfl
          omega
        | some p =>
          obtain ⟨node, entry⟩ := p
          try dsimp only at h
          obtain ⟨⟨old, ov1⟩, htake, h⟩ := bind_inv h
          obtain ⟨merged, hmerge, h⟩ := bind_inv h
          obtain ⟨ov2, hwb, h⟩ := bind_inv h
          have hl' : l' = { self1 with overflow := some ov2 } := (Result.ok_injective h).symm
          have hmerge' : merged.val = new.val + old.val := by
            have hh : new + old = ok merged := hmerge
            exact iadd_ok hh
          obtain ⟨hbody, harena⟩ := body_ov_update self1 l' c (fun x => if x = f.val then c x + 1 else c x) hB1
            (by subst hl'; exact hsm ov2) (by subst hl'; rfl) f ov T0 hO0 hk0 node entry hfe old merged
            ov1 ov2 htake hwb (by rw [hmerge', hnew]; omega) (by simp) (fun x hx => by simp [hx])
            (by subst hl'; rfl)
          refine ⟨hbody, ?_⟩
          have hPl' : Pot l' = self1.len.val + ov2.nodes.val.length := by subst hl'; rfl
          have hPself : Pot self1 = self1.len.val + ovArena self1 := rfl
          omega
      · rw [if_neg hblk] at h
        have hblkf : blocked = false := by simpa using hblk
        rw [hblkf] at hpost
        obtain ⟨⟨⟨_, _⟩, back⟩, hgm, h⟩ := bind_inv h
        obtain ⟨hgm1, hgm2⟩ := get_unchecked_mut_ok _ _ _ _ hgm
        obtain ⟨mu, hmu, h⟩ := bind_inv h
        have hmu' : mu = some new := (Result.ok_injective hmu).symm
        obtain ⟨self2, hsl, h⟩ := bind_inv h
        obtain ⟨i5, hi5, h⟩ := bind_inv h
        have hl' : l' = { self2 with len := i5 } := (Result.ok_injective h).symm
        obtain ⟨hsl1, hsl2⟩ := set_lane_ok _ _ _ _ hsl
        have hi5v := len_ok _ _ hi5
        have hs2 : back (f, mu) = self1.slots.set i4 (f, some new) := by rw [hgm2, hmu']
        have hpost' : ProbePost self1 f.val false false i4.val := hpost
        have hnb := not_blocked_of_vacant self1 f.val i4.val hpost'
        have hc0 : c f.val = 0 := by
          apply hB1.absent
          · rintro ⟨i, hi, hix⟩
            exact (hpost.2.1 rfl rfl).2.2.2 i hi hix
          · rintro ⟨y, hy, hyf⟩
            have := hB1.side.sent hy
            rw [isSent_iff] at this
            exact hfn' (hyf ▸ this)
          · rintro ⟨o, T, ho, hE, j, hj, hk⟩
            obtain ⟨T0, hO⟩ := hB1.ov o ho
            have := Apeiron.MapExecuted.emb_unique hE hO.1.1
            subst this
            obtain ⟨nd, hnd, _⟩ := Apeiron.MapExecuted.emb_mem hE j hj
            obtain ⟨nd', w, hnd', _, _, _, _, hb⟩ := hO.2.2 j hj
            have e1 : nd' = nd := by rw [hnd] at hnd'; exact (Option.some.inj hnd').symm
            rw [e1] at hb
            rw [Apeiron.MapExecuted.keyAt_get hnd] at hk
            rw [hk] at hb
            exact hnb hb
        have hlen' : l'.len.val = self1.len.val + 1 := by
          rw [hl']; show i5.val = _; rw [hi5v, hsl2]
        have hside : l'.side = self1.side := by rw [hl', hsl2]
        have hov' : l'.overflow = self1.overflow := by rw [hl', hsl2]
        have hbody := body_insert_fresh self1 l' c (fun x => if x = f.val then c x + 1 else c x) hB1 f hfn'
          i4 new tag htagv hpost'
          (by rw [hl', hsl2]; exact hs2) (by rw [hl', hsl2]) (by rw [hl', hsl2]) (by rw [hl', hsl2])
          (by rw [hl', hsl2]) hlen' hside hov' (by simp [hc0, hnew]) (by simp) (fun x hx => by simp [hx])
        refine ⟨hbody, ?_⟩
        have : Pot l' = Pot self1 + 1 := by
          unfold Pot ovArena; rw [hlen', hov']; omega
        omega

/-! ## `add` -/

/-- `FreqMap::freq` of a `u64` key is the key itself: no clamp, so every key counts exactly. -/
theorem freq_spec (k i : U64)
    (h : hashtrinity.freq.FreqMap.freq Std.I64 U64.Insts.Apeiron_wordHashtrinityAlgebraCoord k = ok i) :
    i.val = k.val := by
  unfold hashtrinity.freq.FreqMap.freq at h
  have : i = k := (Result.ok_injective h).symm
  rw [this]

theorem add_spec (m m' : hashtrinity.freq.FreqMap U64 Std.I64) (c : Nat → Nat)
    (hR : LineRep m.inner c) (hP : Pot m.inner < nilN) (k : U64)
    (h : hashtrinity.freq.FreqMapKI64.add U64.Insts.Apeiron_wordHashtrinityAlgebraCoord m k 1#i64 = ok m') :
    LineRep m'.inner (fun x => if x = k.val then c x + 1 else c x) ∧
      Pot m'.inner ≤ Pot m.inner + 1 := by
  unfold hashtrinity.freq.FreqMapKI64.add at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨fl, hfl, h⟩ := bind_inv h
  have hm' : m' = { m with inner := fl } := (Result.ok_injective h).symm
  subst hm'
  have hiv := freq_spec _ _ hi
  have := insert_with_spec U64.Insts.Apeiron_wordHashtrinityAlgebraCoord m.inner c hR i 1#i64 rfl hP fl hfl
  rw [hiv] at this
  exact this

/-! ## The representation, stated at the brief's names -/

/-- `m` represents the count function `c`: `c f` is the number of increments whose key has
frequency `f` (`FreqMap::freq`, the key itself).  Every frequency is held in exactly one of the
primary lanes, the side store (reserved frequencies) and the overflow tree (frequencies blocked
past the probe cap); a frequency in none has count `0`. -/
def FreqRep (m : hashtrinity.freq.FreqMap U64 Std.I64) (c : Nat → Nat) : Prop :=
  LineRep m.inner c

/-- The empty table `FreqMap::with_capacity` returns satisfies `FreqRep` with the zero count. -/
theorem freqRep_empty (cap : Usize) (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (h : hashtrinity.freq.FreqMap.with_capacity Std.I64 U64.Insts.Apeiron_wordHashtrinityAlgebraCoord cap
      = ok m) :
    FreqRep m (fun _ => 0) :=
  with_capacity_rep _ cap m h

/-- The empty table holds no primary lane and no overflow node. -/
theorem pot_with_capacity (cap : Usize) (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (h : hashtrinity.freq.FreqMap.with_capacity Std.I64 U64.Insts.Apeiron_wordHashtrinityAlgebraCoord cap
      = ok m) : Pot m.inner = 0 := by
  unfold hashtrinity.freq.FreqMap.with_capacity at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  obtain ⟨i2, hi2, h⟩ := bind_inv h
  obtain ⟨fl, hfl, h⟩ := bind_inv h
  have hm : m = { inner := fl, _k := () } := (Result.ok_injective h).symm
  subst hm
  obtain ⟨hE, -⟩ := with_bits_spec _ _ _ hfl
  unfold Pot ovArena
  show fl.len.val + (match fl.overflow with | none => 0 | some o => o.nodes.val.length) = 0
  rw [hE.len0, hE.ov0]
  rfl

theorem add_preserves (m m' : hashtrinity.freq.FreqMap U64 Std.I64) (c : Nat → Nat)
    (hR : FreqRep m c) (hP : Pot m.inner < nilN) (k : U64)
    (h : hashtrinity.freq.FreqMapKI64.add U64.Insts.Apeiron_wordHashtrinityAlgebraCoord m k 1#i64 = ok m') :
    FreqRep m' (Function.update c k.val (c k.val + 1)) ∧ Pot m'.inner ≤ Pot m.inner + 1 := by
  have := add_spec m m' c hR hP k h
  have e : (fun x => if x = k.val then c x + 1 else c x)
      = Function.update c k.val (c k.val + 1) := by
    funext x
    by_cases hx : x = k.val
    · subst hx; simp
    · simp [hx]
  rw [e] at this
  exact this

end Apeiron.CountAggregate
