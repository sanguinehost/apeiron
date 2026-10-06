-- Row: `p0-count-body-get-the-a-EF9n3OLyqnX2ouQQWL8w9C` (COUNT-BODY step 2d): the count table's representation and `count_aggregate`'s refinement.
import ApeironWord
import Count.Helpers
import Count.FreqRep
import Count.OverflowOps
import Apeiron.Table.Basic
import Mathlib.Data.BitVec
import Mathlib.Tactic.IntervalCases
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
open Aeneas Aeneas.Std Result apeiron_word

/-!
# Constructors, insertion, growth, and add for the frequency table

This module proves, on the extracted Lean:
* Line initialization with capacity (`with_bits_spec`, `empty_rep`, `with_capacity_rep`).
* Insertion and slot update preserve line geometry and line body (`geom_insert`, `body_insert_fresh`, `body_update`).
* Growth preserves line body and doubles capacity (`grow_spec`).
* `insert_with` and `add` preserve representation (`insert_with_spec`, `add_spec`, `add_preserves`).
* Representation definitions: `LineBody`, `LineRep`, `FreqRep`, `freqRep_empty`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.CountAggregate
open Lean.Order Aeneas.Data.Coinductive
open Apeiron.MapExecuted (ONode Ov OTree Emb BST OInv idxs nilN ND EntryOf keyAt FirstAt Present)



/-! ## Constructors -/

theorem EMPTY_val : hashtrinity.EMPTY.val = EMPTYN := by
  unfold hashtrinity.EMPTY EMPTYN
  rfl

theorem EMPTY_CTRL_val : hashtrinity.freq.EMPTY_CTRL.val = 128 := by
  unfold hashtrinity.freq.EMPTY_CTRL
  rfl

theorem max_u32_ok (a b c : U32) (h : core.cmp.Ord.max.default core.cmp.OrdU32 a b = ok c) :
    c.val = max a.val b.val := by
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body at h
  simp [core.cmp.impls.PartialOrdU32.lt, liftFun2] at h
  by_cases hlt : b.val < a.val
  · rw [if_pos hlt] at h
    have := Result.ok_injective h
    subst this; omega
  · rw [if_neg hlt] at h
    have := Result.ok_injective h
    subst this; omega

theorem max_usize_ok (a b c : Usize) (h : core.cmp.Ord.max.default core.cmp.OrdUsize a b = ok c) :
    c.val = max a.val b.val := by
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body at h
  simp [core.cmp.impls.PartialOrdUsize.lt, liftFun2] at h
  by_cases hlt : b.val < a.val
  · rw [if_pos hlt] at h
    have := Result.ok_injective h
    subst this; omega
  · rw [if_neg hlt] at h
    have := Result.ok_injective h
    subst this; omega

theorem min_u64_ok (a b c : U64) (h : core.cmp.Ord.min.trait_default core.cmp.OrdU64 a b = ok c) :
    c.val = min a.val b.val := by
  unfold core.cmp.Ord.min.trait_default core.cmp.Ord.min.default core.cmp.Ord.min_body at h
  simp [core.cmp.impls.PartialOrdU64.lt, liftFun2] at h
  by_cases hlt : b.val < a.val
  · rw [if_pos hlt] at h
    have := Result.ok_injective h
    subst this; omega
  · rw [if_neg hlt] at h
    have := Result.ok_injective h
    subst this; omega

theorem with_bits_loop_spec {V : Type} (n : Usize)
    (slots : alloc.vec.Vec (U64 × core.mem.maybe_uninit.MaybeUninit V)) (i : Usize)
    (hs : slots.val = List.replicate i.val (hashtrinity.EMPTY, (none : Option V))) (hi : i.val ≤ n.val)
    (r : alloc.vec.Vec (U64 × core.mem.maybe_uninit.MaybeUninit V))
    (h : hashtrinity.freq.FreqLine.with_bits_loop n slots i = ok r) :
    r.val = List.replicate n.val (hashtrinity.EMPTY, (none : Option V)) := by
  unfold hashtrinity.freq.FreqLine.with_bits_loop at h
  refine loop_pc (fun (st : alloc.vec.Vec (U64 × core.mem.maybe_uninit.MaybeUninit V) × Usize) =>
      st.1.val = List.replicate st.2.val (hashtrinity.EMPTY, (none : Option V)) ∧ st.2.val ≤ n.val)
    (fun (y : alloc.vec.Vec (U64 × core.mem.maybe_uninit.MaybeUninit V)) =>
      y.val = List.replicate n.val (hashtrinity.EMPTY, (none : Option V))) _ ?_ (slots, i) ⟨hs, hi⟩ r h
  intro st hst cf hcf
  obtain ⟨sl, k⟩ := st
  obtain ⟨hst1, hst2⟩ := hst
  simp only at hst1 hst2
  unfold hashtrinity.freq.FreqLine.with_bits_loop.body at hcf
  dsimp only at hcf
  by_cases hk : k < n
  · rw [if_pos hk] at hcf
    have hk' : k.val < n.val := (UScalar.lt_equiv k n).mp hk
    obtain ⟨mu, hmu, hcf⟩ := bind_inv hcf
    obtain ⟨sl1, hsl1, hcf⟩ := bind_inv hcf
    obtain ⟨k1, hk1, hcf⟩ := bind_inv hcf
    have hcf' : cf = ControlFlow.cont (sl1, k1) := (Result.ok_injective hcf).symm
    subst hcf'
    have hmu' : mu = none := (Result.ok_injective hmu).symm
    subst hmu'
    have h1 := vec_push_ok _ _ _ hsl1
    have h2 := (uadd_ok hk1).1
    have e1 : (1#usize).val = 1 := rfl
    rw [e1] at h2
    refine ⟨?_, show k1.val ≤ n.val by omega⟩
    show sl1.val = List.replicate k1.val _
    rw [h1, hst1, h2, List.replicate_succ']
  · rw [if_neg hk] at hcf
    have hcf' : cf = ControlFlow.done sl := (Result.ok_injective hcf).symm
    subst hcf'
    have hk' : ¬ k.val < n.val := fun hh => hk ((UScalar.lt_equiv k n).mpr hh)
    have : k.val = n.val := by omega
    show sl.val = _
    rw [hst1, this]

theorem two_pow_lt_of_bits (b : Nat) (hb : b < System.Platform.numBits) : 2^b < 2^System.Platform.numBits :=
  Nat.pow_lt_pow_right (by norm_num) hb


structure EmptyLine {V : Type} (l : hashtrinity.freq.FreqLine V) : Prop where
  geom : LineGeom l
  len0 : l.len.val = 0
  foE : ∀ i, foN l i = EMPTYN
  ctE : ∀ i, ctN l i = 128
  side0 : l.side.coords.val = [] ∧ l.side.vals.val = [] ∧ l.side.live.val = []
  ov0 : l.overflow = none

theorem with_bits_spec (V : Type) (bits : U32) (l : hashtrinity.freq.FreqLine V)
    (h : hashtrinity.freq.FreqLine.with_bits V bits = ok l) :
    EmptyLine l ∧ l.bits.val = max bits.val 4 := by
  unfold hashtrinity.freq.FreqLine.with_bits at h
  obtain ⟨bits1, hbits1, h⟩ := bind_inv h
  obtain ⟨n, hn, h⟩ := bind_inv h
  obtain ⟨gbits, hgbits, h⟩ := bind_inv h
  obtain ⟨slots1, hslots1, h⟩ := bind_inv h
  obtain ⟨v, hv, h⟩ := bind_inv h
  obtain ⟨s, hs, h⟩ := bind_inv h
  obtain ⟨s1, hs1, h⟩ := bind_inv h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  obtain ⟨i2, hi2, h⟩ := bind_inv h
  obtain ⟨fs, hfs, h⟩ := bind_inv h
  have hl : l = { ctrl := s, slots := s1, gshift := i, gmask := i2, bits := bits1, len := (0#usize),
                   side := fs, overflow := none } :=
    (Result.ok_injective h).symm
  have hfs' : fs = { coords := alloc.vec.Vec.new Std.U64,
                     vals := alloc.vec.Vec.new (core.mem.maybe_uninit.MaybeUninit V),
                     live := alloc.vec.Vec.new Bool } := (Result.ok_injective hfs).symm
  have hb1 := max_u32_ok _ _ _ hbits1
  have hnv := ushl_ok hn
  have hgv := usub_ok hgbits
  rw [four_val] at hgv
  have e1 : (1#usize).val = 1 := rfl
  rw [e1, one_mul] at hnv
  have hbl : bits1.val < System.Platform.numBits := by
    have := hnv.1
    simpa [UScalarTy.numBits] using this
  have hb4 : 4 ≤ bits1.val := by
    rw [hb1]; omega
  have hnval : n.val = 2^bits1.val := by
    rw [hnv.2]
    apply Nat.mod_eq_of_lt
    simpa [UScalarTy.numBits] using two_pow_lt_of_bits _ hbl
  -- the slot vector
  have hwc : (alloc.vec.Vec.with_capacity (U64 × core.mem.maybe_uninit.MaybeUninit V) n).val = [] := rfl
  have hslots := with_bits_loop_spec n _ 0#usize
    (by rw [hwc]; rfl) (by simp) slots1 hslots1
  -- the control vector
  have hvv : v.val = List.replicate n.val hashtrinity.freq.EMPTY_CTRL ∧ v.length = n.val := by
    obtain ⟨vv, hvv, hvvp⟩ := WP.spec_imp_exists (alloc.vec.from_elem_spec core.clone.CloneU8
      hashtrinity.freq.EMPTY_CTRL n (by rfl))
    rw [hvv] at hv
    have := Result.ok_injective hv
    subst this
    exact hvvp
  have hs' : s = v.slice := (Result.ok_injective hs).symm
  have hs1' : s1 = slots1.slice := (Result.ok_injective hs1).symm
  have hctrl : l.ctrl.val = List.replicate n.val hashtrinity.freq.EMPTY_CTRL := by
    rw [hl]; show s.val = _; rw [hs']; exact hvv.1
  have hsl : l.slots.val = List.replicate n.val (hashtrinity.EMPTY, (none : Option V)) := by
    rw [hl]; show s1.val = _; rw [hs1']; exact hslots
  have hbits : l.bits = bits1 := by rw [hl]
  have hlen : l.len = 0#usize := by rw [hl]
  have hfo : ∀ i, foN l i = EMPTYN := by
    intro k
    unfold foN
    rw [hsl, List.getElem?_replicate]
    by_cases hk : k < n.val
    · simp [hk, EMPTY_val]
    · simp [hk]
  have hct : ∀ k, ctN l k = 128 := by
    intro k
    unfold ctN
    rw [hctrl, List.getElem?_replicate]
    by_cases hk : k < n.val
    · simp [hk, EMPTY_CTRL_val]
    · simp [hk]
  have hgshift : l.gshift = i := by rw [hl]
  have hgmask : l.gmask = i2 := by rw [hl]
  refine ⟨⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, hfo, hct, ?_, ?_⟩, ?_⟩
  · show l.ctrl.val.length = l.slots.val.length
    rw [hctrl, hsl]; simp
  · show l.slots.val.length = _
    rw [hsl, List.length_replicate, hbits, hnval]
  · rw [hbits]; exact hb4
  · -- gshift
    rw [hgshift, hbits]
    by_cases hz : gbits = 0#u32
    · rw [if_pos hz] at hi
      have : i = 0#u32 := (Result.ok_injective hi).symm
      have hg0 : gbits.val = 0 := by rw [hz]; rfl
      have : bits1.val = 4 := by omega
      rw [this, if_pos rfl]
      subst_vars
      rfl
    · rw [if_neg hz] at hi
      have hg0 : gbits.val ≠ 0 := by
        intro h0; apply hz; apply UScalar.eq_of_val_eq; simpa using h0
      have := usub_ok hi
      rw [u64_64_val] at this
      have hne : bits1.val ≠ 4 := by omega
      rw [if_neg hne]
      omega
  · -- gmask
    rw [hgmask, hbits]
    have h16 := (udiv_ok hi1)
    rw [group_val] at h16
    have h2 := usub_ok hi2
    have e : (1#usize).val = 1 := rfl
    rw [e] at h2
    have hpow : 2^bits1.val / 16 = 2^(bits1.val - 4) := by
      have : bits1.val = (bits1.val - 4) + 4 := by omega
      conv_lhs => rw [this, pow_add]
      norm_num
    rw [hnval, hpow] at h16
    have := Nat.two_pow_pos (bits1.val - 4)
    omega
  · intro k _; exact hct k
  · intro k hk; rw [hfo] at hk; exact absurd rfl hk
  · intro k k' hk; rw [hfo] at hk; exact absurd rfl hk
  · intro k hk; rw [hfo] at hk; exact absurd rfl hk
  · rw [hlen]; rfl
  · rw [hl, hfs']; exact ⟨rfl, rfl, rfl⟩
  · rw [hl]
  · rw [hbits, hb1, four_val]


/-! ## Slot and control updates -/

theorem get_unchecked_mut_ok {α : Type} (s : Slice α) (i : Usize) (x : α) (back : α → Slice α)
    (h : core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice α) s i = ok (x, back)) :
    s.val[i.val]? = some x ∧ back = s.set i := by
  unfold core.slice.Slice.get_unchecked_mut at h
  simp only [core.slice.index.Usize.index_mut, Slice.index_mut_usize] at h
  obtain ⟨y, hy, h⟩ := bind_inv h
  have := Result.ok_injective h
  simp only [Prod.mk.injEq] at this
  obtain ⟨h1, h2⟩ := this
  subst h1
  exact ⟨slice_index_ok s i y hy, h2.symm⟩

theorem set_lane_ok {V : Type} (l l' : hashtrinity.freq.FreqLine V) (i : Usize) (b : U8)
    (h : hashtrinity.freq.FreqLine.set_lane l i b = ok l') :
    i.val < l.ctrl.length ∧ l' = { l with ctrl := l.ctrl.set i b } := by
  unfold hashtrinity.freq.FreqLine.set_lane at h
  obtain ⟨⟨x, back⟩, hx, h⟩ := bind_inv h
  obtain ⟨h1, h2⟩ := get_unchecked_mut_ok _ _ _ _ hx
  have hl : l' = { l with ctrl := back b } := (Result.ok_injective h).symm
  refine ⟨?_, ?_⟩
  · exact (List.getElem?_eq_some_iff.mp h1).1
  · rw [hl, h2]

theorem foN_slots_set {V : Type} (l l' : hashtrinity.freq.FreqLine V) (s : Usize)
    (x : U64 × core.mem.maybe_uninit.MaybeUninit V) (hs : s.val < l.slots.length)
    (h : l'.slots = l.slots.set s x) (i : Nat) :
    foN l' i = if i = s.val then x.1.val else foN l i := by
  unfold foN
  rw [h, Slice.set_val_eq, List.getElem?_set]
  by_cases hi : i = s.val
  · subst hi; simp [hs]
  · have : ¬ s.val = i := fun h => hi h.symm
    simp [hi, this]

def vlN {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) : core.mem.maybe_uninit.MaybeUninit V :=
  match l.slots.val[i]? with
  | some y => y.2
  | none => none

theorem vlN_slots_set {V : Type} (l l' : hashtrinity.freq.FreqLine V) (s : Usize)
    (x : U64 × core.mem.maybe_uninit.MaybeUninit V) (hs : s.val < l.slots.length)
    (h : l'.slots = l.slots.set s x) (i : Nat) :
    vlN l' i = if i = s.val then x.2 else vlN l i := by
  unfold vlN
  rw [h, Slice.set_val_eq, List.getElem?_set]
  by_cases hi : i = s.val
  · subst hi; simp [hs]
  · have : ¬ s.val = i := fun h => hi h.symm
    simp [hi, this]

theorem ctN_ctrl_set {V : Type} (l l' : hashtrinity.freq.FreqLine V) (s : Usize) (b : U8)
    (hs : s.val < l.ctrl.length) (h : l'.ctrl = l.ctrl.set s b) (i : Nat) :
    ctN l' i = if i = s.val then b.val else ctN l i := by
  unfold ctN
  rw [h, Slice.set_val_eq, List.getElem?_set]
  by_cases hi : i = s.val
  · subst hi; simp [hs]
  · have : ¬ s.val = i := fun h => hi h.symm
    simp [hi, this]

theorem foN_slots_eq {V : Type} (l l' : hashtrinity.freq.FreqLine V) (h : l'.slots = l.slots) (i : Nat) :
    foN l' i = foN l i := by
  unfold foN; rw [h]

theorem ctN_ctrl_eq {V : Type} (l l' : hashtrinity.freq.FreqLine V) (h : l'.ctrl = l.ctrl) (i : Nat) :
    ctN l' i = ctN l i := by
  unfold ctN; rw [h]

theorem fullG_mono {V : Type} (l l' : hashtrinity.freq.FreqLine V) (g : Nat)
    (h : ∀ i, ctN l i < 128 → ctN l' i < 128) : fullG l g → fullG l' g := by
  intro hf k hk
  exact h _ (hf k hk)

/-- Placing a fresh fold at the vacancy `s` the probe returned preserves the probe's geometry. -/
theorem geom_insert {V : Type} (l l' : hashtrinity.freq.FreqLine V) (hG : LineGeom l) (f : U64)
    (hf : f.val ≠ EMPTYN) (s : Usize) (mv : core.mem.maybe_uninit.MaybeUninit V) (tag : U8)
    (htag : tag.val = tagN f.val) (hpost : ProbePost l f.val false false s.val)
    (hsl : l'.slots = l.slots.set s (f, mv)) (hct : l'.ctrl = l.ctrl.set s tag)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask) :
    LineGeom l' ∧ (∀ i, foN l' i = if i = s.val then f.val else foN l i) ∧
      (∀ i, ctN l' i = if i = s.val then tag.val else ctN l i) := by
  obtain ⟨-, hp2, -⟩ := hpost
  obtain ⟨hc128, hslt, ⟨js, hjs0, hjs1, hjs2⟩, habs⟩ := hp2 rfl rfl
  have hlen : l.ctrl.length = l.slots.length := hG.len_eq
  have hfo := foN_slots_set l l' s (f, mv) hslt hsl
  have hcn := ctN_ctrl_set l l' s tag (by omega) hct
  have htl := tagN_lt f.val
  have hmono : ∀ i, ctN l i < 128 → ctN l' i < 128 := by
    intro i hi
    rw [hcn i]
    by_cases hh : i = s.val
    · simp [hh]; omega
    · simp [hh]; exact hi
  refine ⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, hfo, hcn⟩
  · show l'.ctrl.val.length = l'.slots.val.length
    rw [hct, hsl, Slice.set_val_eq, Slice.set_val_eq]
    simp only [List.length_set]
    exact hlen
  · have : l'.slots.length = l.slots.length := by
      show l'.slots.val.length = _
      rw [hsl, Slice.set_val_eq]; simp
    rw [this, hb]; exact hG.size
  · rw [hb]; exact hG.bits4
  · rw [hgs, hb]; exact hG.gshift
  · rw [hgm, hb]; exact hG.gmask
  · intro i hi
    rw [hfo i] at hi
    by_cases hh : i = s.val
    · simp [hh] at hi; exact absurd hi hf
    · simp [hh] at hi
      rw [hcn i]; simp [hh]; exact hG.empty_ctrl i hi
  · intro i hi
    rw [hfo i] at hi ⊢
    rw [hcn i]
    by_cases hh : i = s.val
    · simp [hh, htag]
    · simp [hh] at hi ⊢; exact hG.full_ctrl i hi
  · intro i j hi hij
    rw [hfo i] at hi hij
    rw [hfo j] at hij
    have hnf : ∀ x, foN l x ≠ EMPTYN → foN l x ≠ f.val := habs
    by_cases hhi : i = s.val <;> by_cases hhj : j = s.val
    · rw [hhi, hhj]
    · simp [hhi, hhj] at hij
      exfalso
      by_cases hz : foN l j = EMPTYN
      · rw [hz] at hij; exact hf hij
      · exact hnf j hz hij.symm
    · simp [hhi, hhj] at hij
      exfalso
      by_cases hz : foN l i = EMPTYN
      · simp [hhi] at hi; exact hi hz
      · exact hnf i hz hij
    · simp [hhi, hhj] at hi hij
      exact hG.distinct i j hi hij
  · intro i hi
    rw [hfo i] at hi
    rw [hb]
    by_cases hh : i = s.val
    · subst hh
      refine ⟨js, hjs0, ?_, ?_⟩
      · rw [hfo, if_pos rfl]; rw [← hjs1]
      · intro j' hj'
        rw [hfo, if_pos rfl]
        exact fullG_mono l l' _ hmono (hjs2 j' hj')
    · simp [hh] at hi
      obtain ⟨jj, hjj0, hjj1, hjj2⟩ := hG.placed i hi
      refine ⟨jj, hjj0, ?_, ?_⟩
      · rw [hfo i, if_neg hh]; exact hjj1
      · intro j' hj'
        rw [hfo i, if_neg hh]
        exact fullG_mono l l' _ hmono (hjj2 j' hj')



/-! ## The count representation -/

abbrev Line := hashtrinity.freq.FreqLine Std.I64
abbrev SideI := hashtrinity.freq.FreqSide Std.I64

/-- The reserved frequencies `EMPTY = 2^64 - 1` and `TOMB = 2^64 - 2`, held out of band. -/
def SentN (f : Nat) : Prop := f = 2^64 - 1 ∨ f = 2^64 - 2

theorem isSent_iff (c : U64) : Apeiron.MapExecuted.IsSent c ↔ SentN c.val := Iff.rfl

def liveSet {V : Type} (l : hashtrinity.freq.FreqLine V) : Finset Nat :=
  (Finset.range l.slots.length).filter (fun i => foN l i ≠ EMPTYN)

/-- Frequency `f` has no primary lane and its probe prefix is full: it lives in the overflow. -/
def Blocked (l : Line) (f : Nat) : Prop :=
  (∀ j, j < 16 → fullG l (gseq (l.bits.val - 4) (pN f) j)) ∧
    ∀ i, foN l i ≠ EMPTYN → foN l i ≠ f

/-- `f` occupies a primary lane. -/
def InMain {V : Type} (l : hashtrinity.freq.FreqLine V) (f : Nat) : Prop :=
  ∃ i, foN l i ≠ EMPTYN ∧ foN l i = f

/-- `f` is a coordinate of the side store. -/
def InSide {V : Type} (l : hashtrinity.freq.FreqLine V) (f : Nat) : Prop :=
  ∃ x ∈ l.side.coords.val, x.val = f

/-- `f` is the key of a node of the overflow tree. -/
def InOv (l : Line) (f : Nat) : Prop :=
  ∃ o T, l.overflow = some o ∧ Apeiron.MapExecuted.Emb o.nodes.val o.root T ∧
    ∃ j ∈ Apeiron.MapExecuted.idxs T, Apeiron.MapExecuted.keyAt o.nodes.val j = f

/-- The side store holds, for each reserved frequency it names, the live count `c f > 0`. -/
structure SideRep (sd : SideI) (c : Nat → Nat) : Prop where
  lv : sd.vals.val.length = sd.coords.val.length
  ll : sd.live.val.length = sd.coords.val.length
  nodup : sd.coords.val.Nodup
  ent : ∀ (i : Nat) (x : U64), sd.coords.val[i]? = some x → Apeiron.MapExecuted.IsSent x ∧
    sd.live.val[i]? = some true ∧
    ∃ v : Std.I64, sd.vals.val[i]? = some (some v) ∧ (c x.val : Int) = v.val ∧ 0 < c x.val

/-- The overflow tree at shape `T`: an AVL tree of single-entry nodes keyed by non-reserved
frequencies, each blocked and carrying its count. -/
def OvInfo (l : Line) (c : Nat → Nat) (o : Ov) (T : OTree) : Prop :=
  OInv o T ∧ NodeOK o.nodes.val T ∧
  ∀ j ∈ Apeiron.MapExecuted.idxs T, ∃ nd w, o.nodes.val[j]? = some nd ∧
    nd.entries.val = [(nd.key, some w)] ∧ (c nd.key.val : Int) = w.val ∧ 0 < c nd.key.val ∧
    ¬ Apeiron.MapExecuted.IsSent nd.key ∧ Blocked l nd.key.val

/-- The line holds the count function `c` jointly in its primary lanes, its side store and its
overflow tree: every counted frequency is in exactly one of the three (the primary lanes and the
overflow are disjoint by `Blocked`, the side store holds only reserved frequencies), and a
frequency in none has count `0`; `len` is the number of live primary lanes. -/
structure LineBody (l : Line) (c : Nat → Nat) : Prop where
  geom : LineGeom l
  nosent : ∀ i, foN l i ≠ EMPTYN → ¬ SentN (foN l i)
  vals : ∀ i, foN l i ≠ EMPTYN → ∃ v, vlN l i = some v ∧ (c (foN l i) : Int) = v.val ∧ 0 < c (foN l i)
  len_eq : l.len.val = (liveSet l).card
  side : SideRep l.side c
  ov : ∀ o, l.overflow = some o → ∃ T, OvInfo l c o T
  absent : ∀ f, ¬ InMain l f → ¬ InSide l f → ¬ InOv l f → c f = 0

/-- The line represents the count function `c`.  The previous `8·len < 7·slots` load bound is not
part of the relation: `grow` re-places every overflow entry, so the primary count after a grow can
reach `len + |overflow|`, and only `len` (not the overflow size) triggers a grow. -/
def LineRep (l : Line) (c : Nat → Nat) : Prop := LineBody l c

theorem mem_liveSet {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) :
    i ∈ liveSet l ↔ i < l.slots.length ∧ foN l i ≠ EMPTYN := by
  simp [liveSet]

theorem foN_out_of_range {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) (h : l.slots.length ≤ i) :
    foN l i = EMPTYN := by
  unfold foN
  rw [List.getElem?_eq_none h]

theorem SideRep.congr {sd : SideI} {c c' : Nat → Nat} (h : SideRep sd c)
    (hcc : ∀ x ∈ sd.coords.val, c' x.val = c x.val) : SideRep sd c' := by
  refine ⟨h.lv, h.ll, h.nodup, ?_⟩
  intro i x hx
  obtain ⟨h1, h2, v, hv, hcv, hpos⟩ := h.ent i x hx
  have hmem : x ∈ sd.coords.val := List.mem_of_getElem? hx
  rw [hcc x hmem]
  exact ⟨h1, h2, v, hv, hcv, hpos⟩

theorem OvInfo.congr {l l' : Line} {c c' : Nat → Nat} {o : Ov} {T : OTree} (hO : OvInfo l c o T)
    (hbl : ∀ j ∈ Apeiron.MapExecuted.idxs T, ∀ nd, o.nodes.val[j]? = some nd →
      Blocked l nd.key.val → Blocked l' nd.key.val)
    (hcc : ∀ j ∈ Apeiron.MapExecuted.idxs T, ∀ nd, o.nodes.val[j]? = some nd →
      Blocked l nd.key.val → c' nd.key.val = c nd.key.val) : OvInfo l' c' o T := by
  obtain ⟨hI, hOK, hv⟩ := hO
  refine ⟨hI, hOK, fun j hj => ?_⟩
  obtain ⟨nd, w, hnd, hent, hc, hpos, hns, hb⟩ := hv j hj
  refine ⟨nd, w, hnd, hent, ?_, ?_, hns, hbl j hj nd hnd hb⟩
  · rw [hcc j hj nd hnd hb]; exact hc
  · rw [hcc j hj nd hnd hb]; exact hpos

/-- A frequency whose probe has a vacant lane within the cap is not blocked. -/
theorem not_blocked_of_vacant (l : Line) (f s : Nat) (hpost : ProbePost l f false false s) :
    ¬ Blocked l f := by
  intro hb
  obtain ⟨-, hp2, -⟩ := hpost
  obtain ⟨hc128, hslt, ⟨js, hjs0, hjs1, hjs2⟩, habs⟩ := hp2 rfl rfl
  have hfull := hb.1 js hjs0
  have h := hfull (s % 16) (by omega)
  have e : 16 * (gseq (l.bits.val - 4) (pN f) js) + s % 16 = s := by rw [← hjs1]; omega
  rw [e] at h
  omega

/-- Placing a fresh frequency in a vacant lane keeps every other frequency blocked. -/
theorem blocked_insert {l l' : Line} (s fv : Nat)
    (hfo : ∀ i, foN l' i = if i = s then fv else foN l i)
    (hmono : ∀ i, ctN l i < 128 → ctN l' i < 128) (hbits : l'.bits = l.bits)
    (g : Nat) (hg : g ≠ fv) (hb : Blocked l g) : Blocked l' g := by
  refine ⟨fun j hj => ?_, fun i hi => ?_⟩
  · rw [hbits]; exact fullG_mono l l' _ hmono (hb.1 j hj)
  · rw [hfo i] at hi ⊢
    by_cases hh : i = s
    · simp [hh] at hi ⊢; exact fun e => hg e.symm
    · simp [hh] at hi ⊢; exact hb.2 i hi

theorem SideRep.sent {sd : SideI} {c : Nat → Nat} (h : SideRep sd c) {x : U64}
    (hx : x ∈ sd.coords.val) : Apeiron.MapExecuted.IsSent x := by
  obtain ⟨i, hi⟩ := List.mem_iff_getElem?.mp hx
  exact (h.ent i x hi).1

theorem inSide_congr {V : Type} {l l' : hashtrinity.freq.FreqLine V} (h : l'.side = l.side) (f : Nat) :
    InSide l' f ↔ InSide l f := by
  unfold InSide; rw [h]

theorem inOv_congr {l l' : Line} (h : l'.overflow = l.overflow) (f : Nat) :
    InOv l' f ↔ InOv l f := by
  unfold InOv; rw [h]

theorem inMain_congr {V : Type} {l l' : hashtrinity.freq.FreqLine V} (h : ∀ i, foN l' i = foN l i)
    (f : Nat) : InMain l' f ↔ InMain l f := by
  unfold InMain; simp only [h]

theorem geom_congr {V : Type} (l l' : hashtrinity.freq.FreqLine V) (hG : LineGeom l)
    (hfo : ∀ i, foN l' i = foN l i) (hct : ∀ i, ctN l' i = ctN l i)
    (hs : l'.slots.length = l.slots.length) (hc : l'.ctrl.length = l.ctrl.length)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask) : LineGeom l' := by
  have hfull : ∀ g, fullG l g → fullG l' g := fun g hg k hk => by rw [hct]; exact hg k hk
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hc, hs]; exact hG.len_eq
  · rw [hs, hb]; exact hG.size
  · rw [hb]; exact hG.bits4
  · rw [hgs, hb]; exact hG.gshift
  · rw [hgm, hb]; exact hG.gmask
  · intro i hi; rw [hfo] at hi; rw [hct]; exact hG.empty_ctrl i hi
  · intro i hi; rw [hfo] at hi ⊢; rw [hct]; exact hG.full_ctrl i hi
  · intro i j hi hij; rw [hfo] at hi hij; rw [hfo] at hij; exact hG.distinct i j hi hij
  · intro i hi
    rw [hfo] at hi
    obtain ⟨j, hj0, hj1, hj2⟩ := hG.placed i hi
    rw [hb, hfo]
    exact ⟨j, hj0, hj1, fun j' hj' => hfull _ (hj2 j' hj')⟩

theorem liveSet_insert {V : Type} (l l' : hashtrinity.freq.FreqLine V) (s : Nat) (f : Nat)
    (hlen : l'.slots.length = l.slots.length) (hs : s < l.slots.length)
    (hfo : ∀ i, foN l' i = if i = s then f else foN l i) (hf : f ≠ EMPTYN)
    (hvac : foN l s = EMPTYN) : liveSet l' = insert s (liveSet l) := by
  ext i
  rw [mem_liveSet, Finset.mem_insert, mem_liveSet, hfo i, hlen]
  by_cases hh : i = s
  · subst hh; simp [hf, hs]
  · simp [hh]

theorem hmono_of_ctrl {l l' : Line} (s : Nat) (tag : Nat) (htag : tag < 128)
    (hcn : ∀ i, ctN l' i = if i = s then tag else ctN l i) :
    ∀ i, ctN l i < 128 → ctN l' i < 128 := by
  intro i hi
  rw [hcn i]
  by_cases hh : i = s
  · simp [hh]; omega
  · simp [hh]; exact hi

theorem body_insert_fresh (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (f : U64)
    (hfs : ¬ SentN f.val) (s : Usize) (v : Std.I64) (tag : U8)
    (htag : tag.val = tagN f.val) (hpost : ProbePost l f.val false false s.val)
    (hsl : l'.slots = l.slots.set s (f, some v)) (hct : l'.ctrl = l.ctrl.set s tag)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask)
    (hlen : l'.len.val = l.len.val + 1) (hside : l'.side = l.side)
    (hov : l'.overflow = l.overflow)
    (hc'f : (c' f.val : Int) = v.val) (hc'pos : 0 < c' f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x) :
    LineBody l' c' := by
  have hf : f.val ≠ EMPTYN := fun h => hfs (Or.inl h)
  obtain ⟨hG', hfo, hcn⟩ := geom_insert l l' hB.geom f hf s (some v) tag htag hpost hsl hct hb hgs hgm
  obtain ⟨-, hp2, -⟩ := hpost
  obtain ⟨hc128, hslt, _, habs⟩ := hp2 rfl rfl
  have hslen : l'.slots.length = l.slots.length := by
    show l'.slots.val.length = _; rw [hsl, Slice.set_val_eq]; simp
  have hvac : foN l s.val = EMPTYN := by
    by_contra hh
    have := hB.geom.full_ctrl _ hh
    have := tagN_lt (foN l s.val)
    omega
  have hvl := vlN_slots_set l l' s (f, some v) hslt hsl
  have hmono := hmono_of_ctrl (l := l) (l' := l') s.val tag.val (by rw [htag]; exact tagN_lt _) hcn
  have hnb := not_blocked_of_vacant l f.val s.val ⟨fun h => by simp at h, fun _ _ => ⟨hc128, hslt, by assumption, habs⟩, fun h => by simp at h⟩
  have hfo' : ∀ i, foN l' i = if i = s.val then f.val else foN l i := hfo
  have hgne : ∀ g, Blocked l g → g ≠ f.val := fun g hg hge => hnb (hge ▸ hg)
  refine ⟨hG', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i hi
    rw [hfo i] at hi ⊢
    by_cases hh : i = s.val
    · simp only [hh, if_true]; exact hfs
    · simp only [hh, if_false] at hi ⊢; exact hB.nosent i hi
  · intro i hi
    rw [hfo i] at hi ⊢
    by_cases hh : i = s.val
    · subst hh
      refine ⟨v, ?_, ?_, ?_⟩
      · rw [hvl]; simp
      · simpa using hc'f
      · simpa using hc'pos
    · simp only [hh, if_false] at hi ⊢
      obtain ⟨v0, hv0, hc0', hpos0⟩ := hB.vals i hi
      have hne : foN l i ≠ f.val := habs i hi
      refine ⟨v0, ?_, ?_, ?_⟩
      · rw [hvl]; simpa [hh] using hv0
      · rw [hc'o _ hne]; exact hc0'
      · rw [hc'o _ hne]; exact hpos0
  · rw [hlen, hB.len_eq, liveSet_insert l l' s.val f.val hslen hslt hfo hf hvac]
    rw [Finset.card_insert_of_notMem]
    rw [mem_liveSet]
    intro ⟨_, h2⟩
    exact h2 hvac
  · rw [hside]
    refine hB.side.congr (fun x hx => hc'o _ ?_)
    intro hxe
    have := hB.side.sent hx
    rw [isSent_iff] at this
    exact hfs (hxe ▸ this)
  · intro o ho
    rw [hov] at ho
    obtain ⟨T, hO⟩ := hB.ov o ho
    refine ⟨T, hO.congr (fun j hj nd hnd hbl => ?_) (fun j hj nd hnd hbl => ?_)⟩
    · exact blocked_insert s.val f.val hfo' hmono (by rw [hb]) _ (hgne _ hbl) hbl
    · exact hc'o _ (hgne _ hbl)
  · intro x hxm hxs hxo
    by_cases hx : x = f.val
    · exfalso
      apply hxm
      refine ⟨s.val, ?_, ?_⟩
      · rw [hfo s.val, if_pos rfl]; exact hf
      · rw [hfo s.val, if_pos rfl]; exact hx.symm
    · rw [hc'o x hx]
      apply hB.absent x
      · rintro ⟨i, hi, hix⟩
        apply hxm
        have hi's : i ≠ s.val := by
          intro hh; subst hh; exact hi hvac
        exact ⟨i, by rw [hfo i, if_neg hi's]; exact hi, by rw [hfo i, if_neg hi's]; exact hix⟩
      · exact fun h => hxs ((inSide_congr hside x).mpr h)
      · exact fun h => hxo ((inOv_congr hov x).mpr h)

theorem body_update (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (f : U64) (s : Usize)
    (hs : foN l s.val = f.val) (hlive : f.val ≠ EMPTYN) (v : Std.I64)
    (hsl : l'.slots = l.slots.set s (f, some v)) (hct : l'.ctrl = l.ctrl)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask)
    (hlen : l'.len = l.len) (hside : l'.side = l.side) (hov : l'.overflow = l.overflow)
    (hc'f : (c' f.val : Int) = v.val) (hc'pos : 0 < c' f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x) :
    LineBody l' c' := by
  have hslt : s.val < l.slots.length := by
    by_contra hh
    have := foN_out_of_range l s.val (by omega)
    rw [this] at hs; exact hlive hs.symm
  have hfo := foN_slots_set l l' s (f, some v) hslt hsl
  have hfo' : ∀ i, foN l' i = foN l i := by
    intro i; rw [hfo i]
    by_cases hh : i = s.val
    · subst hh; simp [hs]
    · simp [hh]
  have hslen : l'.slots.length = l.slots.length := by
    show l'.slots.val.length = _; rw [hsl, Slice.set_val_eq]; simp
  have hvl := vlN_slots_set l l' s (f, some v) hslt hsl
  have hct' : ∀ i, ctN l' i = ctN l i := fun i => ctN_ctrl_eq l l' hct i
  have hG' := geom_congr l l' hB.geom hfo' hct' hslen (by rw [hct]) hb hgs hgm
  have hls : liveSet l' = liveSet l := by
    ext i; rw [mem_liveSet, mem_liveSet, hfo', hslen]
  have hgne : ∀ g, Blocked l g → g ≠ f.val := by
    intro g hg hge
    exact hg.2 s.val (by rw [hs]; exact hlive) (by rw [hs]; exact hge.symm)
  refine ⟨hG', ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro i hi; rw [hfo'] at hi ⊢; exact hB.nosent i hi
  · intro i hi
    rw [hfo'] at hi ⊢
    by_cases hh : i = s.val
    · subst hh
      refine ⟨v, ?_, ?_, ?_⟩
      · rw [hvl]; simp
      · rw [hs]; exact hc'f
      · rw [hs]; exact hc'pos
    · obtain ⟨v0, hv0, hc0', hpos0⟩ := hB.vals i hi
      have hne : foN l i ≠ f.val := by
        intro heq
        exact hh (hB.geom.distinct i s.val hi (by rw [heq, hs]))
      refine ⟨v0, ?_, ?_, ?_⟩
      · rw [hvl]; simpa [hh] using hv0
      · rw [hc'o _ hne]; exact hc0'
      · rw [hc'o _ hne]; exact hpos0
  · rw [hlen, hls]; exact hB.len_eq
  · rw [hside]
    refine hB.side.congr (fun x hx => hc'o _ ?_)
    intro hxe
    have := hB.side.sent hx
    rw [isSent_iff] at this
    exact hB.nosent s.val (by rw [hs]; exact hlive) (by rw [hs, ← hxe]; exact this)
  · intro o ho
    rw [hov] at ho
    obtain ⟨T, hO⟩ := hB.ov o ho
    refine ⟨T, hO.congr (fun j hj nd hnd hbl => ?_) (fun j hj nd hnd hbl => ?_)⟩
    · refine ⟨fun j' hj' => ?_, fun i hi => ?_⟩
      · rw [hb]; intro k hk; rw [hct']; exact hbl.1 j' hj' k hk
      · rw [hfo'] at hi ⊢; exact hbl.2 i hi
    · exact hc'o _ (hgne _ hbl)
  · intro x hxm hxs hxo
    by_cases hx : x = f.val
    · exfalso
      apply hxm
      refine ⟨s.val, ?_, ?_⟩
      · rw [hfo', hs]; exact hlive
      · rw [hfo', hs]; exact hx.symm
    · rw [hc'o x hx]
      apply hB.absent x
      · rintro ⟨i, hi, hix⟩
        apply hxm
        exact ⟨i, by rw [hfo']; exact hi, by rw [hfo']; exact hix⟩
      · exact fun h => hxs ((inSide_congr hside x).mpr h)
      · exact fun h => hxo ((inOv_congr hov x).mpr h)

theorem assume_init_read_ok {T : Type} (m : core.mem.maybe_uninit.MaybeUninit T) (v : T)
    (h : core.mem.maybe_uninit.MaybeUninit.assume_init_read m = ok v) : m = some v := by
  unfold core.mem.maybe_uninit.MaybeUninit.assume_init_read at h
  cases m with
  | none => exact absurd h (by intro hh; exact ok_not_fail hh.symm)
  | some x => simp at h; rw [h]

theorem needs_drop_i64 : core.mem.needs_drop Std.I64 = ok false := by
  unfold core.mem.needs_drop
  have : NoDropGlue Std.I64 := Or.inr (Or.inl ⟨.I64, rfl⟩)
  simp [this]

theorem len_ok {V : Type} (l : hashtrinity.freq.FreqLine V) (n : Usize) (h : l.len + 1#usize = ok n) :
    n.val = l.len.val + 1 := by
  have := (uadd_ok h).1
  have e : (1#usize).val = 1 := rfl
  rw [e] at this; exact this

theorem empty_rep (l : Line) (hE : EmptyLine l) : LineRep l (fun _ => 0) := by
  unfold LineRep
  have hls : liveSet l = ∅ := by
    ext i
    rw [mem_liveSet]
    simp [hE.foE]
  obtain ⟨hc, hv, hl⟩ := hE.side0
  refine ⟨hE.geom, ?_, ?_, ?_, ⟨?_, ?_, ?_, ?_⟩, ?_, ?_⟩
  · intro i hi; exact absurd (hE.foE i) hi
  · intro i hi; exact absurd (hE.foE i) hi
  · rw [hE.len0, hls]; rfl
  · rw [hv, hc]; rfl
  · rw [hl, hc]; rfl
  · rw [hc]; exact List.nodup_nil
  · intro i x hx; rw [hc] at hx; exact absurd hx (by simp)
  · intro o ho; rw [hE.ov0] at ho; exact absurd ho (by simp)
  · intro f _ _ _; rfl

theorem with_capacity_rep {K : Type} (inst : hashtrinity.algebra.Coord K) (cap : Usize)
    (m : hashtrinity.freq.FreqMap K Std.I64)
    (h : hashtrinity.freq.FreqMap.with_capacity Std.I64 inst cap = ok m) :
    LineRep m.inner (fun _ => 0) := by
  unfold hashtrinity.freq.FreqMap.with_capacity at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  obtain ⟨i2, hi2, h⟩ := bind_inv h
  obtain ⟨fl, hfl, h⟩ := bind_inv h
  have hm : m = { inner := fl, _k := () } := (Result.ok_injective h).symm
  subst hm
  exact empty_rep fl (with_bits_spec _ _ _ hfl).1

end Apeiron.CountAggregate
