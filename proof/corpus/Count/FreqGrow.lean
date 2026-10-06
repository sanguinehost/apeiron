-- Row: `apeiron-hashdos-cap-count-MUBqkA7OYTvluKSIRdbjC`: the capped re-placement and growth of the count face.
import ApeironWord
import Count.Helpers
import Count.FreqRep
import Count.OverflowOps
import Count.FreqAdd
import Count.FreqSideOv
open Aeneas Aeneas.Std Result apeiron_word

/-!
# `place_capped` and `grow` for the count face

`place_capped` re-places one distinct non-reserved frequency through the capped probe (a vacant lane
within `PROBE_CAP` groups, else a fresh overflow node); `grow` runs it over every primary lane, then
over every overflow entry (`into_entries`), carrying the side store over unchanged.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive
open Apeiron.MapExecuted (ONode Ov OTree Emb BST OInv idxs nilN ND EntryOf keyAt FirstAt Present)

/-- The arena length of the overflow store (`0` when there is none). -/
def ovArena (l : Line) : Nat :=
  match l.overflow with
  | none => 0
  | some o => o.nodes.val.length

/-- Distinct frequencies placed in primary lanes, plus overflow arena slots; each placement adds at
most one. -/
def Pot (l : Line) : Nat := l.len.val + ovArena l

theorem is_sentinel_false (f : U64) (hf : ¬ Apeiron.MapExecuted.IsSent f) :
    hashtrinity.is_sentinel f = ok false := by
  rw [Apeiron.MapExecuted.is_sentinel_ok]
  simp [hf]

theorem as_mut_ok {α : Type} (x : Option α) :
    core.option.Option.as_mut x = ok (x, fun y => y) := rfl

/-- The overflow branch of `place_capped`, once the store `ov` it works on is fixed. -/
theorem place_blocked_tail (l l' : Line) (c : Nat → Nat) (hB : LineBody l c) (f : U64) (v : Std.I64)
    (hfns : ¬ SentN f.val) (hc0 : c f.val = 0) (hvpos : 0 < v.val) (hP : Pot l < nilN)
    (hblk : Blocked l f.val) (ov : Ov)
    (hov : l.overflow = some ov ∨
      (l.overflow = none ∧ hashtrinity.overflow.Overflow.new I64 Apeiron.MapExecuted.CI = ok ov))
    (nd : U32) (e : Usize) (ov1 : Ov)
    (hins : hashtrinity.overflow.Overflow.insert_new_at Apeiron.MapExecuted.CI ov f f v = ok ((nd, e), ov1))
    (hl' : l' = { l with overflow := some ov1 }) :
    LineBody l' (fun x => if x = f.val then v.val.toNat else c x) ∧ l'.bits = l.bits ∧
      l'.slots.length = l.slots.length ∧ l'.side = l.side ∧ Pot l' ≤ Pot l + 1 := by
  have hfn' : ¬ Apeiron.MapExecuted.IsSent f := fun hh => hfns ((isSent_iff f).mp hh)
  have hvint : ((v.val.toNat : Nat) : Int) = v.val := Int.toNat_of_nonneg (by omega)
  obtain ⟨T0, hO0, hk0⟩ := ov_start l c hB ov hov
  have hnp : ¬ Present ov.nodes.val T0 f f := by
    rintro ⟨j, hj, nd1, hnd1, hk, _⟩
    obtain ⟨nd2, w, hnd2, _, hc2, hp2, _⟩ := hO0.2.2 j hj
    have e1 : nd2 = nd1 := by rw [hnd1] at hnd2; exact (Option.some.inj hnd2).symm
    rw [e1, hk] at hp2
    omega
  have hlenov : ov.nodes.val.length ≤ ovArena l ∧ ov.nodes.val.length < nilN := by
    rcases hov with ho | ⟨hn, hnew⟩
    · have : ovArena l = ov.nodes.val.length := by unfold ovArena; rw [ho]
      constructor
      · omega
      · have : ovArena l ≤ Pot l := by unfold Pot; omega
        omega
    · obtain ⟨_, _, hn0⟩ := new_spec ov hnew
      rw [hn0]
      exact ⟨by simp, by unfold nilN; norm_num⟩
  have hsm : SameMain l' l := by subst hl'; exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  have hv : ((if f.val = f.val then v.val.toNat else c f.val : Nat) : Int) = v.val := by
    simp [hvint]
  obtain ⟨hbody, harena⟩ := body_ov_insert l l' c (fun x => if x = f.val then v.val.toNat else c x) hB
    hsm (by subst hl'; rfl) f v ov T0 hO0 hk0 hfn' hblk (by simpa using hvint)
    (by simp; omega) (fun x hx => by simp [hx]) hlenov.2 hnp nd e ov1 hins (by subst hl'; rfl)
  refine ⟨hbody, hsm.bits, by rw [hsm.slots], by subst hl'; rfl, ?_⟩
  unfold Pot ovArena
  rw [hsm.len, hl']
  simp only
  unfold Pot ovArena at *
  omega

/-- `place_capped` of a fresh non-reserved frequency: the count function gains `f ↦ v`. -/
theorem place_capped_spec (l l' : Line) (c : Nat → Nat) (hB : LineBody l c) (f : U64) (v : Std.I64)
    (hfns : ¬ SentN f.val) (hc0 : c f.val = 0) (hvpos : 0 < v.val) (hP : Pot l < nilN)
    (h : hashtrinity.freq.FreqLine.place_capped l f v = ok l') :
    LineBody l' (fun x => if x = f.val then v.val.toNat else c x) ∧ l'.bits = l.bits ∧
      l'.slots.length = l.slots.length ∧ l'.side = l.side ∧ Pot l' ≤ Pot l + 1 := by
  have hfn' : ¬ Apeiron.MapExecuted.IsSent f := fun hh => hfns ((isSent_iff f).mp hh)
  have hvint : ((v.val.toNat : Nat) : Int) = v.val := Int.toNat_of_nonneg (by omega)
  unfold hashtrinity.freq.FreqLine.place_capped at h
  obtain ⟨b, hb, h⟩ := bind_inv h
  rw [is_sentinel_false f hfn'] at hb
  have hbf : b = false := (Result.ok_injective hb).symm
  subst hbf
  simp only [Bool.false_eq_true, if_false] at h
  obtain ⟨⟨i, found, blocked, tag⟩, hprobe, h⟩ := bind_inv h
  obtain ⟨htagv, hpost⟩ := probe_tagged_spec l hB.geom f i found blocked tag hprobe
  change (if blocked = true then _ else _) = _ at h
  by_cases hblk : blocked = true
  · rw [if_pos hblk] at h
    obtain ⟨hf0, hblk1, hnotin⟩ := hpost.2.2 hblk
    have hblk' : Blocked l f.val := ⟨hblk1, hnotin⟩
    obtain ⟨o, ho, h⟩ := bind_inv h
    have hovx : ∃ ov, o = some ov ∧ (l.overflow = some ov ∨
        (l.overflow = none ∧ hashtrinity.overflow.Overflow.new I64 Apeiron.MapExecuted.CI = ok ov)) := by
      rcases hov : l.overflow with _ | ov0
      · simp only [hov, core.option.Option.is_none, Option.isNone_none, if_true] at ho
        obtain ⟨o1, hnew, ho⟩ := bind_inv ho
        exact ⟨o1, (Result.ok_injective ho).symm, Or.inr ⟨rfl, hnew⟩⟩
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
    obtain ⟨⟨⟨nd, e⟩, ov1⟩, hins, h⟩ := bind_inv h
    have hl' : l' = { l with overflow := some ov1 } := (Result.ok_injective h).symm
    exact place_blocked_tail l l' c hB f v hfns hc0 hvpos hP hblk' ov hov' nd e ov1 hins hl'
  · rw [if_neg hblk] at h
    have hblkf : blocked = false := by simpa using hblk
    rw [hblkf] at hpost
    change (if found = true then _ else _) = _ at h
    by_cases hfound : found = true
    · exfalso
      obtain ⟨-, hfo, hfne⟩ := hpost.1 hfound
      obtain ⟨v1, -, hcv, hp1⟩ := hB.vals i.val hfne
      rw [hfo, hc0] at hp1
      exact absurd hp1 (by omega)
    · rw [if_neg hfound] at h
      have hfalse : found = false := by simpa using hfound
      rw [hfalse] at hpost
      obtain ⟨⟨⟨_, _⟩, back⟩, hgm, h⟩ := bind_inv h
      obtain ⟨hgm1, hgm2⟩ := get_unchecked_mut_ok _ _ _ _ hgm
      obtain ⟨mu, hmu, h⟩ := bind_inv h
      have hmu' : mu = some v := (Result.ok_injective hmu).symm
      obtain ⟨self1, hsl, h⟩ := bind_inv h
      obtain ⟨i1, hi1, h⟩ := bind_inv h
      have hl' : l' = { self1 with len := i1 } := (Result.ok_injective h).symm
      obtain ⟨hsl1, hsl2⟩ := set_lane_ok _ _ _ _ hsl
      have hi1v := len_ok _ _ hi1
      have hs2 : back (f, mu) = l.slots.set i (f, some v) := by rw [hgm2, hmu']
      have hpost' : ProbePost l f.val false false i.val := hpost
      have hlen' : l'.len.val = l.len.val + 1 := by
        rw [hl']; show i1.val = _; rw [hi1v, hsl2]
      have hside : l'.side = l.side := by rw [hl', hsl2]
      have hov' : l'.overflow = l.overflow := by rw [hl', hsl2]
      have hbody := body_insert_fresh l l' c (fun x => if x = f.val then v.val.toNat else c x) hB f hfns
        i v tag htagv hpost'
        (by rw [hl', hsl2]; exact hs2) (by rw [hl', hsl2]) (by rw [hl', hsl2]) (by rw [hl', hsl2])
        (by rw [hl', hsl2]) hlen' hside hov' (by simpa using hvint) (by simp; omega)
        (fun x hx => by simp [hx])
      refine ⟨hbody, by rw [hl', hsl2], ?_, hside, ?_⟩
      · rw [hl', hsl2]
        show (back (f, mu)).val.length = _
        rw [hs2, Slice.set_val_eq]; simp
      · unfold Pot ovArena
        rw [hlen', hov']
        omega

/-! ## `grow`: the primary-lane loop -/

open Classical in
/-- The count function restricted to the frequencies in the first `i` slots of `old`. -/
noncomputable def ci (old : Line) (c : Nat → Nat) (i x : Nat) : Nat :=
  if ∃ k, k < i ∧ foN old k ≠ EMPTYN ∧ foN old k = x then c x else 0

def Pset (old : Line) (i : Nat) : Finset Nat :=
  (Finset.range i).filter (fun k => foN old k ≠ EMPTYN)

theorem Pset_sub (old : Line) (i : Nat) (hi : i ≤ old.slots.length) : Pset old i ⊆ liveSet old := by
  intro k hk
  simp only [Pset, Finset.mem_filter, Finset.mem_range] at hk
  rw [mem_liveSet]
  exact ⟨by omega, hk.2⟩

theorem Pset_full (old : Line) : Pset old old.slots.length = liveSet old := by
  ext k; simp [Pset, liveSet]

theorem EMPTY_ne_val (cc : U64) (h : (cc != hashtrinity.EMPTY) = true) : cc.val ≠ EMPTYN := by
  intro hh
  have : cc = hashtrinity.EMPTY := UScalar.eq_of_val_eq (by rw [EMPTY_val]; exact hh)
  simp [this] at h

theorem EMPTY_of_not_ne (cc : U64) (h : ¬ (cc != hashtrinity.EMPTY) = true) : cc.val = EMPTYN := by
  have : cc = hashtrinity.EMPTY := UScalar.eq_of_val_eq (by simpa using h)
  rw [this, EMPTY_val]

theorem index_mut_usize_ok {α : Type} (s : Slice α) (i : Usize) (x : α) (back : α → Slice α)
    (h : Slice.index_mut_usize s i = ok (x, back)) : s.val[i.val]? = some x ∧ back = s.set i := by
  unfold Slice.index_mut_usize at h
  obtain ⟨y, hy, h⟩ := bind_inv h
  have := Result.ok_injective h
  simp only [Prod.mk.injEq] at this
  obtain ⟨h1, h2⟩ := this
  subst h1
  exact ⟨slice_index_ok s i y hy, h2.symm⟩

theorem slice_set_same {α : Type} (s : Slice α) (i : Usize) (x : α) (h : s.val[i.val]? = some x) :
    s.set i x = s := by
  apply Slice.ext
  rw [Slice.set_val_eq]
  apply List.ext_getElem?
  intro k
  rw [List.getElem?_set]
  by_cases hk : i.val = k
  · subst hk
    obtain ⟨hlt, heq⟩ := List.getElem?_eq_some_iff.mp h
    simp [hlt, heq]
  · simp [hk]

theorem slice_set_getElem {α : Type} (s : Slice α) (i : Usize) (x : α) (k : Nat)
    (hi : i.val < s.length) :
    (s.set i x).val[k]? = if k = i.val then some x else s.val[k]? := by
  rw [Slice.set_val_eq, List.getElem?_set]
  by_cases hk : i.val = k
  · subst hk; simp [hi]
  · have : ¬ k = i.val := fun h => hk h.symm
    simp [hk, this]

theorem ci_zero (old : Line) (c : Nat → Nat) : ci old c 0 = fun _ => 0 := by
  funext x; unfold ci; simp

def GrowInv (old : Line) (c : Nat → Nat) (n : Usize) (bg0 : Line)
    (st : Slice (U64 × core.mem.maybe_uninit.MaybeUninit Std.I64) × Line × Usize) : Prop :=
  st.2.2.val ≤ n.val ∧ n.val = old.slots.length ∧
  (∀ k, st.1.val[k]? = if k < st.2.2.val
      then (old.slots.val[k]?).map (fun y => (hashtrinity.EMPTY, y.2)) else old.slots.val[k]?) ∧
  LineBody st.2.1 (ci old c st.2.2.val) ∧ st.2.1.bits.val = old.bits.val + 1 ∧
  st.2.1.side = bg0.side ∧ Pot st.2.1 ≤ (Pset old st.2.2.val).card

theorem grow_body_spec (old : Line) (c : Nat → Nat) (hB : LineBody old c) (hPo : Pot old < nilN)
    (n : Usize) (bg0 : Line)
    (st : Slice (U64 × core.mem.maybe_uninit.MaybeUninit Std.I64) × Line × Usize)
    (hI : GrowInv old c n bg0 st) (cf : ControlFlow
      (Slice (U64 × core.mem.maybe_uninit.MaybeUninit Std.I64) × Line × Usize) Line)
    (h : hashtrinity.freq.FreqLine.grow_loop0.body n st.1 st.2.1 st.2.2 = ok cf) :
    match cf with
    | .done bg => LineBody bg (ci old c old.slots.length) ∧ bg.bits.val = old.bits.val + 1 ∧
        bg.side = bg0.side ∧ Pot bg ≤ old.len.val
    | .cont st' => GrowInv old c n bg0 st' := by
  obtain ⟨s, bg, i⟩ := st
  obtain ⟨hi1, hn, hs, hbg, hbits, hside, hpot⟩ := hI
  dsimp only at hi1 hn hs hbg hbits hside hpot h
  unfold hashtrinity.freq.FreqLine.grow_loop0.body at h
  by_cases hlt : i < n
  · rw [if_pos hlt] at h
    have hlt' : i.val < n.val := (UScalar.lt_equiv i n).mp hlt
    obtain ⟨⟨cc, mu0⟩, hidx, h⟩ := bind_inv h
    have hsi := slice_index_ok _ _ _ hidx
    have hold : old.slots.val[i.val]? = some (cc, mu0) := by
      rw [hs i.val] at hsi
      simpa using hsi
    have hfoi : foN old i.val = cc.val := foN_of_get _ _ _ hold
    have hvli : vlN old i.val = mu0 := by unfold vlN; rw [hold]
    change (if (cc != hashtrinity.EMPTY) = true then _ else _) = _ at h
    by_cases hc : (cc != hashtrinity.EMPTY) = true
    · rw [if_pos hc] at h
      have hccne := EMPTY_ne_val cc hc
      obtain ⟨⟨⟨i1, mu⟩, back⟩, hgm, h⟩ := bind_inv h
      obtain ⟨hgm1, hgm2⟩ := get_unchecked_mut_ok _ _ _ _ hgm
      have himu : (i1, mu) = (cc, mu0) := by
        rw [hsi] at hgm1; exact (Option.some_injective _ hgm1).symm
      simp only [Prod.mk.injEq] at himu
      obtain ⟨him1, him2⟩ := himu
      obtain ⟨v, hv, h⟩ := bind_inv h
      have hmu : mu = some v := assume_init_read_ok _ _ hv
      obtain ⟨⟨⟨_, mu1⟩, imb⟩, him, h⟩ := bind_inv h
      obtain ⟨him1', him2'⟩ := index_mut_usize_ok _ _ _ _ him
      have hs1 : back (i1, mu) = s := by
        rw [hgm2, him1, him2]; exact slice_set_same s i (cc, mu0) hsi
      rw [hs1] at him1'
      have hmu1 : mu1 = mu0 := by
        rw [hsi] at him1'
        have := Option.some_injective _ him1'
        simp only [Prod.mk.injEq] at this
        exact this.2.symm
      obtain ⟨bigger1, hpl, h⟩ := bind_inv h
      obtain ⟨i2, hi2, h⟩ := bind_inv h
      have hcf : cf = ControlFlow.cont (imb (hashtrinity.EMPTY, mu1), bigger1, i2) :=
        (Result.ok_injective h).symm
      subst hcf
      have hi2v : i2.val = i.val + 1 := by
        have := (uadd_ok hi2).1
        have e : (1#usize).val = 1 := rfl
        rw [e] at this; exact this
      have hmu0 : mu0 = some v := by rw [← him2, hmu]
      obtain ⟨v0, hv0, hcv0, hpos0⟩ := hB.vals i.val (by rw [hfoi]; exact hccne)
      have hv0v : v0 = v := by
        rw [hvli, hmu0] at hv0; exact (Option.some_injective _ hv0).symm
      rw [hv0v, hfoi] at hcv0
      rw [hfoi] at hpos0
      have hfns : ¬ SentN cc.val := by
        have := hB.nosent i.val (by rw [hfoi]; exact hccne)
        rwa [hfoi] at this
      have hc0 : ci old c i.val cc.val = 0 := by
        unfold ci
        rw [if_neg]
        rintro ⟨k, hk1, hk2, hk3⟩
        have : k = i.val := hB.geom.distinct k i.val hk2 (by rw [hk3, hfoi])
        omega
      have hvpos : 0 < v.val := by rw [← hcv0]; exact_mod_cast hpos0
      have hPset_sub := Pset_sub old (i.val + 1) (by omega)
      have hPi : Pset old (i.val + 1) = insert i.val (Pset old i.val) := by
        ext k
        simp only [Pset, Finset.mem_filter, Finset.mem_range, Finset.mem_insert]
        constructor
        · rintro ⟨hk1, hk2⟩
          by_cases hki : k = i.val
          · exact Or.inl hki
          · exact Or.inr ⟨by omega, hk2⟩
        · rintro (hk | ⟨hk1, hk2⟩)
          · subst hk; exact ⟨by omega, by rw [hfoi]; exact hccne⟩
          · exact ⟨by omega, hk2⟩
      have hPn : i.val ∉ Pset old i.val := by simp [Pset]
      have hcard : (Pset old (i.val + 1)).card = (Pset old i.val).card + 1 := by
        rw [hPi, Finset.card_insert_of_notMem hPn]
      have hcard_le : (Pset old (i.val + 1)).card ≤ old.len.val := by
        rw [hB.len_eq]; exact Finset.card_le_card hPset_sub
      have hoPot : old.len.val ≤ Pot old := by unfold Pot; omega
      have hPbg : Pot bg < nilN := by omega
      obtain ⟨hbody, hbb, -, hsd, hpt⟩ := place_capped_spec bg bigger1 (ci old c i.val) hbg cc v hfns hc0
        hvpos hPbg hpl
      have hfun : (fun x => if x = cc.val then v.val.toNat else ci old c i.val x) =
          ci old c (i.val + 1) := by
        funext x
        unfold ci
        by_cases hx : x = cc.val
        · subst hx
          have : ∃ k, k < i.val + 1 ∧ foN old k ≠ EMPTYN ∧ foN old k = cc.val :=
            ⟨i.val, by omega, by rw [hfoi]; exact hccne, hfoi⟩
          rw [if_pos rfl, if_pos this]
          have : ((c cc.val : Nat) : Int) = v.val := hcv0
          omega
        · rw [if_neg hx]
          by_cases hx' : ∃ k, k < i.val ∧ foN old k ≠ EMPTYN ∧ foN old k = x
          · have : ∃ k, k < i.val + 1 ∧ foN old k ≠ EMPTYN ∧ foN old k = x := by
              obtain ⟨k, hk1, hk2⟩ := hx'; exact ⟨k, by omega, hk2⟩
            rw [if_pos hx', if_pos this]
          · have : ¬ ∃ k, k < i.val + 1 ∧ foN old k ≠ EMPTYN ∧ foN old k = x := by
              rintro ⟨k, hk1, hk2, hk3⟩
              by_cases hki : k = i.val
              · subst hki; rw [hfoi] at hk3; exact hx hk3.symm
              · exact hx' ⟨k, by omega, hk2, hk3⟩
            rw [if_neg hx', if_neg this]
      rw [hfun] at hbody
      unfold GrowInv
      dsimp only
      refine ⟨by omega, hn, ?_, ?_, ?_, ?_, ?_⟩
      · intro k
        have hsi' : i.val < s.length := (List.getElem?_eq_some_iff.mp hsi).1
        have hb3 : imb (hashtrinity.EMPTY, mu1) = s.set i (hashtrinity.EMPTY, mu1) := by
          rw [him2', hs1]
        rw [hi2v, hb3, slice_set_getElem s i _ k hsi', hs k]
        by_cases hk : k = i.val
        · subst hk; simp [hold, hmu1]
        · by_cases hlt2 : k < i.val
          · simp [hk, hlt2, show k < i.val + 1 by omega]
          · simp [hk, hlt2, show ¬ k < i.val + 1 by omega]
      · rw [hi2v]; exact hbody
      · rw [hbb]; exact hbits
      · rw [hsd]; exact hside
      · rw [hi2v, hcard]; omega
    · rw [if_neg hc] at h
      obtain ⟨i1, hi1', h⟩ := bind_inv h
      have hcf : cf = ControlFlow.cont (s, bg, i1) := (Result.ok_injective h).symm
      subst hcf
      have hccE := EMPTY_of_not_ne cc hc
      have hi1v : i1.val = i.val + 1 := by
        have := (uadd_ok hi1').1
        have e : (1#usize).val = 1 := rfl
        rw [e] at this; exact this
      have hci : ci old c i1.val = ci old c i.val := by
        funext x
        unfold ci
        rw [hi1v]
        by_cases hx : ∃ k, k < i.val ∧ foN old k ≠ EMPTYN ∧ foN old k = x
        · have : ∃ k, k < i.val + 1 ∧ foN old k ≠ EMPTYN ∧ foN old k = x := by
            obtain ⟨k, hk1, hk2⟩ := hx; exact ⟨k, by omega, hk2⟩
          rw [if_pos hx, if_pos this]
        · have : ¬ ∃ k, k < i.val + 1 ∧ foN old k ≠ EMPTYN ∧ foN old k = x := by
            rintro ⟨k, hk1, hk2, hk3⟩
            by_cases hki : k = i.val
            · subst hki; rw [hfoi] at hk2; exact hk2 hccE
            · exact hx ⟨k, by omega, hk2, hk3⟩
          rw [if_neg hx, if_neg this]
      have hP : Pset old i1.val = Pset old i.val := by
        ext k
        simp only [Pset, Finset.mem_filter, Finset.mem_range, hi1v]
        constructor
        · rintro ⟨hk1, hk2⟩
          refine ⟨?_, hk2⟩
          by_cases hki : k = i.val
          · subst hki; rw [hfoi] at hk2; exact absurd hccE hk2
          · omega
        · rintro ⟨hk1, hk2⟩; exact ⟨by omega, hk2⟩
      unfold GrowInv
      dsimp only
      refine ⟨by omega, hn, ?_, ?_, hbits, hside, ?_⟩
      · intro k
        rw [hs k, hi1v]
        by_cases hki : k = i.val
        · subst hki
          simp only [lt_irrefl, if_false, Nat.lt_succ_self, if_true]
          rw [hold]
          simp
          have : cc = hashtrinity.EMPTY := UScalar.eq_of_val_eq (by rw [hccE, EMPTY_val])
          rw [this]
        · by_cases hk : k < i.val
          · simp [hk, show k < i.val + 1 by omega]
          · simp [hk, show ¬ k < i.val + 1 by omega]
      · rw [hci]; exact hbg
      · rw [hP]; exact hpot
  · rw [if_neg hlt] at h
    have hcf : cf = ControlFlow.done bg := (Result.ok_injective h).symm
    subst hcf
    have hlt' : ¬ i.val < n.val := fun hh => hlt ((UScalar.lt_equiv i n).mpr hh)
    have hin : i.val = old.slots.length := by omega
    refine ⟨?_, hbits, hside, ?_⟩
    · rw [hin] at hbg; exact hbg
    · rw [hin, Pset_full, ← hB.len_eq] at hpot
      exact hpot

theorem grow_loop_spec (old : Line) (c : Nat → Nat) (hB : LineBody old c) (hPo : Pot old < nilN)
    (bg0 : Line) (hE : EmptyLine bg0) (hbits : bg0.bits.val = old.bits.val + 1) (n : Usize)
    (hn : n.val = old.slots.length) (r : Line)
    (h : hashtrinity.freq.FreqLine.grow_loop0 old.slots bg0 0#usize n = ok r) :
    LineBody r (ci old c old.slots.length) ∧ r.bits.val = old.bits.val + 1 ∧ r.side = bg0.side ∧
      Pot r ≤ old.len.val := by
  unfold hashtrinity.freq.FreqLine.grow_loop0 at h
  refine loop_pc (GrowInv old c n bg0)
    (fun (y : Line) => LineBody y (ci old c old.slots.length) ∧ y.bits.val = old.bits.val + 1 ∧
      y.side = bg0.side ∧ Pot y ≤ old.len.val)
    _ ?_ (old.slots, bg0, 0#usize) ?_ r h
  · intro st hst cf hcf
    obtain ⟨s, bg, i⟩ := st
    dsimp only at hcf
    cases cf with
    | done y => exact grow_body_spec old c hB hPo n bg0 (s, bg, i) hst (.done y) hcf
    | cont x => exact grow_body_spec old c hB hPo n bg0 (s, bg, i) hst (.cont x) hcf
  · have hbl := (empty_rep bg0 hE)
    unfold GrowInv
    dsimp only
    refine ⟨by simp, hn, ?_, ?_, hbits, rfl, ?_⟩
    · intro k; simp
    · have e : (0#usize).val = 0 := rfl
      rw [e, ci_zero]; exact hbl
    · have e : (0#usize).val = 0 := rfl
      rw [e]
      unfold Pot ovArena
      rw [hE.len0, hE.ov0]
      simp [Pset]

/-! ## `grow`: attaching the side store -/

open Classical in
/-- The count function restricted to the frequencies in a primary lane or the side store of `old`. -/
noncomputable def cS (old : Line) (c : Nat → Nat) (x : Nat) : Nat :=
  if InMain old x ∨ InSide old x then c x else 0

theorem inMain_iff_lt (old : Line) (x : Nat) :
    InMain old x ↔ ∃ k, k < old.slots.length ∧ foN old k ≠ EMPTYN ∧ foN old k = x := by
  constructor
  · rintro ⟨i, hi, hix⟩
    refine ⟨i, ?_, hi, hix⟩
    by_contra hh
    exact hi (foN_out_of_range old i (by omega))
  · rintro ⟨k, _, hk1, hk2⟩
    exact ⟨k, hk1, hk2⟩

theorem cS_eq_ci (old : Line) (c : Nat → Nat) (hB : LineBody old c) (x : Nat) (hx : ¬ SentN x) :
    cS old c x = ci old c old.slots.length x := by
  unfold cS ci
  have h1 : InSide old x → False := by
    rintro ⟨y, hy, hyx⟩
    have := hB.side.sent hy
    rw [isSent_iff] at this
    exact hx (hyx ▸ this)
  by_cases hm : InMain old x
  · rw [if_pos (Or.inl hm), if_pos ((inMain_iff_lt old x).mp hm)]
  · have : ¬ ∃ k, k < old.slots.length ∧ foN old k ≠ EMPTYN ∧ foN old k = x :=
      fun h => hm ((inMain_iff_lt old x).mpr h)
    rw [if_neg this, if_neg]
    rintro (h | h)
    · exact hm h
    · exact h1 h

theorem body_attach_side (old bg l0 : Line) (c : Nat → Nat) (hB : LineBody old c)
    (hbg : LineBody bg (ci old c old.slots.length)) (hsm : SameMain l0 bg)
    (hov : l0.overflow = bg.overflow) (hside : l0.side = old.side)
    (hbgside : bg.side.coords.val = []) : LineBody l0 (cS old c) := by
  have hcsc := cS_eq_ci old c hB
  refine ⟨hsm.geom hbg.geom, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i hi; rw [hsm.foN] at hi ⊢; exact hbg.nosent i hi
  · intro i hi
    rw [hsm.foN] at hi ⊢
    obtain ⟨v, hv, hcv, hp⟩ := hbg.vals i hi
    have hns := hbg.nosent i hi
    rw [hcsc _ hns]
    refine ⟨v, ?_, hcv, hp⟩
    unfold vlN at hv ⊢; rw [hsm.slots]; exact hv
  · rw [hsm.len, hsm.liveSet]; exact hbg.len_eq
  · rw [hside]
    refine hB.side.congr (fun x hx => ?_)
    unfold cS
    rw [if_pos (Or.inr ⟨x, hx, rfl⟩)]
  · intro o ho
    rw [hov] at ho
    obtain ⟨T, hO⟩ := hbg.ov o ho
    refine ⟨T, hO.congr (fun j hj nd hnd hbl => hsm.blocked _ hbl) (fun j hj nd hnd hbl => ?_)⟩
    obtain ⟨nd', w, hnd', _, _, _, hns, _⟩ := hO.2.2 j hj
    have e1 : nd' = nd := by rw [hnd] at hnd'; exact (Option.some.inj hnd').symm
    rw [e1] at hns
    exact hcsc _ (fun hh => hns ((isSent_iff nd.key).mpr hh))
  · intro x hxm hxs hxo
    unfold cS
    by_cases hc : InMain old x ∨ InSide old x
    · exfalso
      rcases hc with hm | hs
      · have hpos : 0 < c x := by
          obtain ⟨i, hi, hix⟩ := hm
          obtain ⟨v, _, hcv, hp⟩ := hB.vals i hi
          rw [hix] at hp; exact hp
        have hci : ci old c old.slots.length x = c x := by
          unfold ci; rw [if_pos ((inMain_iff_lt old x).mp hm)]
        have := hbg.absent x
        by_cases h1 : InMain bg x
        · exact hxm ((hsm.inMain x).mpr h1)
        · by_cases h2 : InSide bg x
          · obtain ⟨y, hy, _⟩ := h2
            rw [hbgside] at hy; exact absurd hy (by simp)
          · by_cases h3 : InOv bg x
            · exact hxo ((inOv_congr hov x).mpr h3)
            · have := this h1 h2 h3
              omega
      · exact hxs ((inSide_congr hside x).mpr hs)
    · rw [if_neg hc]

/-! ## `grow`: re-placing the overflow entries -/

/-- What `OvInfo` says about one `(coordinate, key, value)` triple `into_entries` yields. -/
def EntInfo (old : Line) (c : Nat → Nat) (t : U64 × U64 × Std.I64) : Prop :=
  t.1 = t.2.1 ∧ ¬ Apeiron.MapExecuted.IsSent t.1 ∧ Blocked old t.1.val ∧ 0 < c t.1.val ∧
    (c t.1.val : Int) = t.2.2.val

theorem entries_aux (old : Line) (c : Nat → Nat) (ov : Ov) (T : OTree) (hO : OvInfo old c ov T) :
    ∀ (J : List Nat) (L : List (U64 × U64 × Std.I64)),
      List.Forall₂ (EntryRel ov.nodes.val) J L → (∀ j ∈ J, j ∈ idxs T) → J.Nodup →
      (∀ t ∈ L, EntInfo old c t) ∧ (L.map (fun t => t.1.val)).Nodup ∧
        (∀ x, (∃ j ∈ J, keyAt ov.nodes.val j = x) ↔ ∃ t ∈ L, t.1.val = x) := by
  intro J L hL
  induction hL with
  | nil =>
    intro _ _
    refine ⟨fun t ht => by simp at ht, List.nodup_nil, fun x => ?_⟩
    simp
  | @cons j t J' L' hjt hrest ih =>
    intro hJ hnd
    have hJ' : ∀ j ∈ J', j ∈ idxs T := fun j' hj' => hJ j' (List.mem_cons_of_mem _ hj')
    have hnd' := (List.nodup_cons.mp hnd)
    obtain ⟨ih1, ih2, ih3⟩ := ih hJ' hnd'.2
    obtain ⟨nd, w, hnd1, hent, rfl⟩ := hjt
    obtain ⟨nd', w', hnd2, hent2, hc, hp, hns, hb⟩ := hO.2.2 j (hJ j (List.mem_cons_self ..))
    have e1 : nd' = nd := by rw [hnd1] at hnd2; exact (Option.some.inj hnd2).symm
    rw [e1] at hent2 hc hp hns hb
    have e2 : w' = w := by
      rw [hent] at hent2
      have := List.cons.inj hent2
      exact (Option.some.inj (Prod.mk.inj this.1).2).symm
    rw [e2] at hc
    refine ⟨?_, ?_, ?_⟩
    · intro t ht
      rcases List.mem_cons.mp ht with rfl | ht
      · exact ⟨rfl, hns, hb, hp, hc⟩
      · exact ih1 t ht
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨fun hmem => ?_, ih2⟩
      obtain ⟨t', ht', hxt⟩ := List.mem_map.mp hmem
      obtain ⟨j', hj', hkj⟩ := (ih3 t'.1.val).mpr ⟨t', ht', rfl⟩
      have hne : j ≠ j' := fun hh => hnd'.1 (hh ▸ hj')
      have hinj := Apeiron.MapExecuted.bst_key_inj hO.1.2.1 (hJ j (List.mem_cons_self ..))
        (hJ j' (List.mem_cons_of_mem _ hj')) hne
      apply hinj
      rw [Apeiron.MapExecuted.keyAt_get hnd1, hkj, ← hxt]
    · intro x
      constructor
      · rintro ⟨j2, hj2, hkx⟩
        rcases List.mem_cons.mp hj2 with rfl | hj2
        · exact ⟨_, List.mem_cons_self .., by rw [← hkx, Apeiron.MapExecuted.keyAt_get hnd1]⟩
        · obtain ⟨t2, ht2, hx2⟩ := (ih3 x).mp ⟨j2, hj2, hkx⟩
          exact ⟨t2, List.mem_cons_of_mem _ ht2, hx2⟩
      · rintro ⟨t2, ht2, hx2⟩
        rcases List.mem_cons.mp ht2 with rfl | ht2
        · exact ⟨j, List.mem_cons_self .., by rw [Apeiron.MapExecuted.keyAt_get hnd1]; exact hx2⟩
        · obtain ⟨j2, hj2, hk2⟩ := (ih3 x).mpr ⟨t2, ht2, hx2⟩
          exact ⟨j2, List.mem_cons_of_mem _ hj2, hk2⟩

open Classical in
/-- The count function restricted to `old`'s primary lanes, side store and the entries `done`. -/
noncomputable def cE (old : Line) (c : Nat → Nat) (done : List (U64 × U64 × Std.I64)) (x : Nat) : Nat :=
  if InMain old x ∨ InSide old x ∨ (∃ t ∈ done, t.1.val = x) then c x else 0

theorem cE_nil (old : Line) (c : Nat → Nat) : cE old c [] = cS old c := by
  funext x; unfold cE cS
  by_cases h : InMain old x ∨ InSide old x
  · rw [if_pos h, if_pos (by rcases h with h | h; exact Or.inl h; exact Or.inr (Or.inl h))]
  · rw [if_neg h, if_neg]
    rintro (h1 | h1 | ⟨t, ht, _⟩)
    · exact h (Or.inl h1)
    · exact h (Or.inr h1)
    · simp at ht

theorem list_len_le_arena (ov : Ov) (T : OTree) (hI : OInv ov T) : (idxs T).length ≤ ov.nodes.val.length := by
  have hnd := nd_nodup hI.2.2.1
  have hsub : idxs T ⊆ List.range ov.nodes.val.length := by
    intro j hj
    obtain ⟨nd, hnd1, _⟩ := Apeiron.MapExecuted.emb_mem hI.1 j hj
    simpa using (List.getElem?_eq_some_iff.mp hnd1).1
  have := (hnd.subperm hsub).length_le
  simpa using this

theorem grow_loop1_spec (old : Line) (c : Nat → Nat) (hB : LineBody old c) (hPo : Pot old < nilN)
    (ov : Ov) (T : OTree) (hovo : old.overflow = some ov) (hO : OvInfo old c ov T)
    (L : List (U64 × U64 × Std.I64)) (hL : List.Forall₂ (EntryRel ov.nodes.val) (idxs T) L)
    (iter : alloc.vec.into_iter.IntoIter (U64 × U64 × Std.I64)) (hit : ivl iter = L)
    (bg r : Line) (hbg : LineBody bg (cE old c [])) (hside : bg.side = old.side)
    (hpot : Pot bg ≤ old.len.val)
    (h : hashtrinity.freq.FreqLine.grow_loop1 iter bg = ok r) :
    LineBody r c ∧ r.bits = bg.bits ∧ r.side = old.side ∧ Pot r ≤ old.len.val + L.length := by
  obtain ⟨hinfo, hknd, hkeys⟩ := entries_aux old c ov T hO (idxs T) L hL (fun j hj => hj)
    (nd_nodup hO.1.2.2.1)
  have hLlen : L.length ≤ ov.nodes.val.length := by
    rw [← hL.length_eq]; exact list_len_le_arena ov T hO.1
  have hArena : ovArena old = ov.nodes.val.length := by unfold ovArena; rw [hovo]
  have hPoLe : old.len.val + ov.nodes.val.length ≤ Pot old := by unfold Pot; omega
  unfold hashtrinity.freq.FreqLine.grow_loop1 at h
  refine loop_pc (fun (st : alloc.vec.into_iter.IntoIter (U64 × U64 × Std.I64) × Line) =>
      ∃ proc rem, L = proc ++ rem ∧ ivl st.1 = rem ∧ LineBody st.2 (cE old c proc) ∧
        st.2.bits = bg.bits ∧ st.2.side = old.side ∧ Pot st.2 ≤ old.len.val + proc.length)
    (fun (y : Line) => LineBody y c ∧ y.bits = bg.bits ∧ y.side = old.side ∧
      Pot y ≤ old.len.val + L.length)
    _ ?_ (iter, bg) ⟨[], L, rfl, hit, hbg, rfl, hside, by simpa using hpot⟩ r h
  intro st hst cf hcf
  obtain ⟨it, b⟩ := st
  obtain ⟨proc, rem, hsplit, hrem, hbody, hbits, hsd, hpt⟩ := hst
  dsimp only at hrem hbody hbits hsd hpt hcf
  unfold hashtrinity.freq.FreqLine.grow_loop1.body at hcf
  obtain ⟨⟨o, it1⟩, hnext, hcf⟩ := bind_inv hcf
  rcases next_spec _ _ _ hnext with ⟨h0, ho, h1⟩ | ⟨hd, tl, h0, ho, h1⟩
  · subst ho
    try simp only at hcf
    have hcf' : cf = ControlFlow.done b := (Result.ok_injective hcf).symm
    subst hcf'
    have hrem0 : rem = [] := by rw [← hrem, h0]
    subst hrem0
    have hLp : L = proc := by rw [hsplit]; simp
    subst hLp
    have hfun : cE old c L = c := by
      funext x
      unfold cE
      by_cases hcond : InMain old x ∨ InSide old x ∨ (∃ t ∈ L, t.1.val = x)
      · rw [if_pos hcond]
      · rw [if_neg hcond]
        symm
        apply hB.absent x
        · exact fun h => hcond (Or.inl h)
        · exact fun h => hcond (Or.inr (Or.inl h))
        · rintro ⟨o2, T2, ho2, hE2, j, hj, hk⟩
          rw [hovo] at ho2
          have := Option.some.inj ho2
          subst this
          have hTT := Apeiron.MapExecuted.emb_unique hE2 hO.1.1
          subst hTT
          exact hcond (Or.inr (Or.inr ((hkeys x).mp ⟨j, hj, hk⟩)))
    rw [hfun] at hbody
    exact ⟨hbody, hbits, hsd, by omega⟩
  · subst ho
    try simp only at hcf
    obtain ⟨cc, k, v⟩ := hd
    have hrem' : (cc, k, v) :: tl = rem := by rw [← hrem, h0]
    have hltl : ivl it1 = tl := by rw [h1]
    obtain ⟨bigger1, hpl, hcf⟩ := bind_inv hcf
    have hcf' : cf = ControlFlow.cont (it1, bigger1) := (Result.ok_injective hcf).symm
    subst hcf'
    have hsplit' : L = proc ++ (cc, k, v) :: tl := by rw [hsplit, ← hrem']
    have hmemL : (cc, k, v) ∈ L := by rw [hsplit']; simp
    obtain ⟨hk1, hns, hbl, hpos, hcv⟩ := hinfo _ hmemL
    simp only at hk1 hns hbl hpos hcv
    have hfns : ¬ SentN cc.val := fun hh => hns ((isSent_iff cc).mpr hh)
    have hnodup2 := hknd
    rw [hsplit', List.map_append, List.nodup_append] at hnodup2
    have hnotin : ∀ t ∈ proc, t.1.val ≠ cc.val := fun t ht heq =>
      hnodup2.2.2 t.1.val (List.mem_map_of_mem ht) cc.val (by simp) heq
    have hc0 : cE old c proc cc.val = 0 := by
      unfold cE
      rw [if_neg]
      rintro (⟨i, hi, hix⟩ | ⟨y, hy, hyx⟩ | ⟨t, ht, htx⟩)
      · exact hbl.2 i hi hix
      · have := hB.side.sent hy
        rw [isSent_iff] at this
        exact hfns (hyx ▸ this)
      · exact hnotin t ht htx
    have hvpos : 0 < v.val := by rw [← hcv]; exact_mod_cast hpos
    have hlenlt : proc.length < L.length := by rw [hsplit']; simp
    have hPbg : Pot b < nilN := by omega
    obtain ⟨hbody', hbb, -, hsd', hpt'⟩ := place_capped_spec b bigger1 (cE old c proc) hbody cc v hfns hc0
      hvpos hPbg hpl
    have hfun : (fun x => if x = cc.val then v.val.toNat else cE old c proc x) =
        cE old c (proc ++ [(cc, k, v)]) := by
      funext x
      unfold cE
      by_cases hx : x = cc.val
      · subst hx
        rw [if_pos rfl, if_pos (Or.inr (Or.inr ⟨(cc, k, v), by simp, rfl⟩))]
        have : ((c cc.val : Nat) : Int) = v.val := hcv
        omega
      · rw [if_neg hx]
        by_cases hcond : InMain old x ∨ InSide old x ∨ (∃ t ∈ proc, t.1.val = x)
        · rw [if_pos hcond, if_pos]
          rcases hcond with h1 | h1 | ⟨t, ht, htx⟩
          · exact Or.inl h1
          · exact Or.inr (Or.inl h1)
          · exact Or.inr (Or.inr ⟨t, List.mem_append_left _ ht, htx⟩)
        · rw [if_neg hcond, if_neg]
          rintro (h1 | h1 | ⟨t, ht, htx⟩)
          · exact hcond (Or.inl h1)
          · exact hcond (Or.inr (Or.inl h1))
          · rcases List.mem_append.mp ht with ht | ht
            · exact hcond (Or.inr (Or.inr ⟨t, ht, htx⟩))
            · simp at ht; subst ht; exact hx htx.symm
    rw [hfun] at hbody'
    refine ⟨proc ++ [(cc, k, v)], tl, by rw [hsplit']; simp, hltl, hbody', ?_, ?_, ?_⟩
    · rw [hbb]; exact hbits
    · rw [hsd']; exact hsd
    · rw [List.length_append]; simp; omega

/-- **`grow`.**  The doubled line holds the same count function; the potential does not increase. -/
theorem grow_spec (l l1 : Line) (c : Nat → Nat) (hR : LineBody l c) (hP : Pot l < nilN)
    (h : hashtrinity.freq.FreqLine.grow l = ok l1) :
    LineBody l1 c ∧ l1.bits.val = l.bits.val + 1 ∧ Pot l1 ≤ Pot l := by
  unfold hashtrinity.freq.FreqLine.grow at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨bigger, hbigger, h⟩ := bind_inv h
  obtain ⟨hE, hbb⟩ := with_bits_spec _ _ _ hbigger
  have hiv := (uadd_ok hi).1
  have e : (1#u32).val = 1 := rfl
  rw [e] at hiv
  have hb4 := hR.geom.bits4
  have hbits : bigger.bits.val = l.bits.val + 1 := by
    rw [hbb, hiv]; omega
  obtain ⟨bigger1, hloop, h⟩ := bind_inv h
  obtain ⟨hbody0, hbits1, hside1, hpot1⟩ := grow_loop_spec l c hR hP bigger hE hbits
    (Slice.len l.slots) (Slice.len_val _) bigger1 hloop
  obtain ⟨fs, hfs, h⟩ := bind_inv h
  have hbs : bigger1.side.coords.val = [] := by rw [hside1]; exact hE.side0.1
  simp only [core.mem.replace, core.option.Option.take] at h
  have hl0 : LineBody ({ bigger1 with side := l.side } : Line) (cS l c) :=
    body_attach_side l bigger1 _ c hR hbody0 ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ rfl rfl hbs
  have hpl0 : Pot ({ bigger1 with side := l.side } : Line) ≤ l.len.val := hpot1
  have hlen0 : ovArena l ≥ 0 := Nat.zero_le _
  rcases hov : l.overflow with _ | ov
  · simp only [hov] at h
    have hl1 : l1 = { bigger1 with side := l.side } := (Result.ok_injective h).symm
    subst hl1
    have hcS : cS l c = c := by
      funext x
      unfold cS
      by_cases hx : InMain l x ∨ InSide l x
      · rw [if_pos hx]
      · rw [if_neg hx]
        symm
        apply hR.absent x
        · exact fun hh => hx (Or.inl hh)
        · exact fun hh => hx (Or.inr hh)
        · rintro ⟨o, T, ho, _⟩
          rw [hov] at ho; exact absurd ho (by simp)
    rw [hcS] at hl0
    refine ⟨hl0, hbits1, ?_⟩
    have : ovArena l = 0 := by unfold ovArena; rw [hov]
    unfold Pot at hpl0 ⊢
    omega
  · simp only [hov] at h
    obtain ⟨v, hv, h⟩ := bind_inv h
    obtain ⟨iter, hiter, h⟩ := bind_inv h
    obtain ⟨T, hO⟩ := hR.ov ov hov
    have hL := into_entries_spec ov T hO.1.1 hO.1.2.2.1 hO.2.1 v hv
    have hit : ivl iter = v.val :=
      (congrArg (fun x : alloc.vec.Vec (U64 × U64 × Std.I64) => x.val) (Result.ok_injective hiter)).symm
    rw [← cE_nil l c] at hl0
    obtain ⟨hbody1, hb1, -, hp1⟩ := grow_loop1_spec l c hR hP ov T hov hO v.val hL iter hit
      ({ bigger1 with side := l.side } : Line) l1 hl0 rfl hpl0 h
    refine ⟨hbody1, by rw [hb1]; exact hbits1, ?_⟩
    have hLlen : v.val.length ≤ ov.nodes.val.length := by
      rw [← hL.length_eq]; exact list_len_le_arena ov T hO.1
    have : ovArena l = ov.nodes.val.length := by unfold ovArena; rw [hov]
    unfold Pot at hp1 ⊢
    omega

end Apeiron.CountAggregate
