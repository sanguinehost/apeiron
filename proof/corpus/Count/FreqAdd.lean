-- Row: `p0-count-body-get-the-a-EF9n3OLyqnX2ouQQWL8w9C` (COUNT-BODY step 2d): the count table's representation and `count_aggregate`'s refinement.
import ApeironWord
import Count.Helpers
import Count.FreqRep
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

theorem vec_push_ok {α : Type} (v v1 : alloc.vec.Vec α) (x : α)
    (h : alloc.vec.Vec.push v x = ok v1) : v1.val = v.val ++ [x] := by
  unfold alloc.vec.Vec.push at h
  dsimp only at h
  split at h
  · have := Result.ok_injective h
    subst this
    simp [alloc.vec.Vec.from_val]
  · exact absurd h (by intro hh; exact ok_not_fail hh.symm)

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
  have hl : l = { ctrl := s, slots := s1, gshift := i, gmask := i2, bits := bits1, len := 0#usize } :=
    (Result.ok_injective h).symm
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
  refine ⟨⟨⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩, ?_, hfo, hct⟩, ?_⟩
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
    (htag : tag.val = tagN f.val) (hpost : ProbePost l f.val false s.val)
    (hsl : l'.slots = l.slots.set s (f, mv)) (hct : l'.ctrl = l.ctrl.set s tag)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask) :
    LineGeom l' ∧ (∀ i, foN l' i = if i = s.val then f.val else foN l i) ∧
      (∀ i, ctN l' i = if i = s.val then tag.val else ctN l i) := by
  obtain ⟨-, hp2⟩ := hpost
  obtain ⟨hc128, hslt, ⟨js, hjs1, hjs2⟩, habs⟩ := hp2 rfl
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
      refine ⟨js, ?_, ?_⟩
      · rw [hfo, if_pos rfl]; rw [← hjs1]
      · intro j' hj'
        rw [hfo, if_pos rfl]
        exact fullG_mono l l' _ hmono (hjs2 j' hj')
    · simp [hh] at hi
      obtain ⟨jj, hjj1, hjj2⟩ := hG.placed i hi
      refine ⟨jj, ?_, ?_⟩
      · rw [hfo i, if_neg hh]; exact hjj1
      · intro j' hj'
        rw [hfo i, if_neg hh]
        exact fullG_mono l l' _ hmono (hjj2 j' hj')


/-! ## The count representation -/

abbrev Line := hashtrinity.freq.FreqLine Std.I64

def liveSet {V : Type} (l : hashtrinity.freq.FreqLine V) : Finset Nat :=
  (Finset.range l.slots.length).filter (fun i => foN l i ≠ EMPTYN)

/-- The line holds, for every live slot, a distinct fold with its count `c fold`, and absent folds
have count `0`; `len` is the number of live slots. -/
structure LineBody (l : Line) (c : Nat → Nat) : Prop where
  geom : LineGeom l
  vals : ∀ i, foN l i ≠ EMPTYN → ∃ v, vlN l i = some v ∧ (c (foN l i) : Int) = v.val ∧ 0 < c (foN l i)
  absent : ∀ f, (∀ i, foN l i ≠ EMPTYN → foN l i ≠ f) → c f = 0
  len_eq : l.len.val = (liveSet l).card

/-- `LineBody` and the load stays below the growth threshold. -/
structure LineRep (l : Line) (c : Nat → Nat) : Prop extends LineBody l c where
  load : 8 * l.len.val < 7 * l.slots.length

theorem mem_liveSet {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) :
    i ∈ liveSet l ↔ i < l.slots.length ∧ foN l i ≠ EMPTYN := by
  simp [liveSet]

theorem foN_out_of_range {V : Type} (l : hashtrinity.freq.FreqLine V) (i : Nat) (h : l.slots.length ≤ i) :
    foN l i = EMPTYN := by
  unfold foN
  rw [List.getElem?_eq_none h]

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
    obtain ⟨j, hj1, hj2⟩ := hG.placed i hi
    rw [hb, hfo]
    exact ⟨j, hj1, fun j' hj' => hfull _ (hj2 j' hj')⟩

theorem liveSet_insert {V : Type} (l l' : hashtrinity.freq.FreqLine V) (s : Nat) (f : Nat)
    (hlen : l'.slots.length = l.slots.length) (hs : s < l.slots.length)
    (hfo : ∀ i, foN l' i = if i = s then f else foN l i) (hf : f ≠ EMPTYN)
    (hvac : foN l s = EMPTYN) : liveSet l' = insert s (liveSet l) := by
  ext i
  rw [mem_liveSet, Finset.mem_insert, mem_liveSet, hfo i, hlen]
  by_cases hh : i = s
  · subst hh; simp [hf, hs]
  · simp [hh]

theorem body_insert_fresh (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (f : U64)
    (hf : f.val ≠ EMPTYN) (s : Usize) (v : Std.I64) (tag : U8)
    (htag : tag.val = tagN f.val) (hpost : ProbePost l f.val false s.val)
    (hsl : l'.slots = l.slots.set s (f, some v)) (hct : l'.ctrl = l.ctrl.set s tag)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask)
    (hlen : l'.len.val = l.len.val + 1)
    (hc'f : (c' f.val : Int) = v.val) (hc'pos : 0 < c' f.val) (hc'o : ∀ x, x ≠ f.val → c' x = c x) :
    LineBody l' c' := by
  obtain ⟨hG', hfo, hcn⟩ := geom_insert l l' hB.geom f hf s (some v) tag htag hpost hsl hct hb hgs hgm
  obtain ⟨-, hp2⟩ := hpost
  obtain ⟨hc128, hslt, _, habs⟩ := hp2 rfl
  have hslen : l'.slots.length = l.slots.length := by
    show l'.slots.val.length = _; rw [hsl, Slice.set_val_eq]; simp
  have hvac : foN l s.val = EMPTYN := by
    by_contra hh
    have := hB.geom.full_ctrl _ hh
    have := tagN_lt (foN l s.val)
    omega
  have hvl := vlN_slots_set l l' s (f, some v) hslt hsl
  refine ⟨hG', ?_, ?_, ?_⟩
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
  · intro x hx
    by_cases hxf : x = f.val
    · exfalso
      apply hx s.val
      · rw [hfo s.val, if_pos rfl]; exact hf
      · rw [hfo s.val, if_pos rfl]; exact hxf.symm
    · rw [hc'o x hxf]
      apply hB.absent
      intro i hi
      have hne : i ≠ s.val := by
        intro hh; subst hh; exact hi hvac
      have := hx i (by rw [hfo i]; simp [hne]; exact hi)
      rw [hfo i] at this
      simpa [hne] using this
  · rw [hlen, hB.len_eq, liveSet_insert l l' s.val f.val hslen hslt hfo hf hvac]
    rw [Finset.card_insert_of_notMem]
    rw [mem_liveSet]
    intro ⟨_, h2⟩
    exact h2 hvac

theorem body_update (l l' : Line) (c c' : Nat → Nat) (hB : LineBody l c) (f : U64) (s : Usize)
    (hs : foN l s.val = f.val) (hlive : f.val ≠ EMPTYN) (v : Std.I64)
    (hsl : l'.slots = l.slots.set s (f, some v)) (hct : l'.ctrl = l.ctrl)
    (hb : l'.bits = l.bits) (hgs : l'.gshift = l.gshift) (hgm : l'.gmask = l.gmask)
    (hlen : l'.len = l.len)
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
  have hG' := geom_congr l l' hB.geom hfo' (fun i => ctN_ctrl_eq l l' hct i) hslen
    (by rw [hct]) hb hgs hgm
  have hls : liveSet l' = liveSet l := by
    ext i; rw [mem_liveSet, mem_liveSet, hfo', hslen]
  refine ⟨hG', ?_, ?_, ?_⟩
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
  · intro x hx
    by_cases hxf : x = f.val
    · exfalso
      apply hx s.val
      · rw [hfo', hs]; exact hlive
      · rw [hfo', hs]; exact hxf.symm
    · rw [hc'o x hxf]
      apply hB.absent
      intro i hi
      have := hx i (by rw [hfo']; exact hi)
      rwa [hfo'] at this
  · rw [hlen, hls]; exact hB.len_eq



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
  have hls : liveSet l = ∅ := by
    ext i
    rw [mem_liveSet]
    simp [hE.foE]
  refine ⟨⟨hE.geom, ?_, ?_, ?_⟩, ?_⟩
  · intro i hi; exact absurd (hE.foE i) hi
  · intro f _; rfl
  · rw [hE.len0, hls]; rfl
  · rw [hE.len0]
    have := hE.geom.size
    have := Nat.two_pow_pos l.bits.val
    omega

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


/-! ## `grow` -/

open Classical in
noncomputable def ci (old : Line) (c : Nat → Nat) (i x : Nat) : Nat :=
  if ∃ k, k < i ∧ foN old k ≠ EMPTYN ∧ foN old k = x then c x else 0

def Pset (old : Line) (i : Nat) : Finset Nat :=
  (Finset.range i).filter (fun k => foN old k ≠ EMPTYN)

def GrowInv (old : Line) (c : Nat → Nat) (n : Usize)
    (st : Slice (U64 × core.mem.maybe_uninit.MaybeUninit Std.I64) × Line × Usize) : Prop :=
  st.2.2.val ≤ n.val ∧ n.val = old.slots.length ∧
  (∀ k, st.1.val[k]? = if k < st.2.2.val
      then (old.slots.val[k]?).map (fun y => (hashtrinity.EMPTY, y.2)) else old.slots.val[k]?) ∧
  LineBody st.2.1 (ci old c st.2.2.val) ∧ st.2.1.bits.val = old.bits.val + 1 ∧
  st.2.1.len.val = (Pset old st.2.2.val).card ∧
  (∀ j, foN st.2.1 j ≠ EMPTYN → ∃ k, k < st.2.2.val ∧ foN old k = foN st.2.1 j ∧
    vlN st.2.1 j = vlN old k)

theorem ci_zero (old : Line) (c : Nat → Nat) : ci old c 0 = fun _ => 0 := by
  funext x; unfold ci; simp

theorem ci_full (old : Line) (c : Nat → Nat) (hB : LineBody old c) :
    ci old c old.slots.length = c := by
  funext x
  unfold ci
  by_cases hx : ∃ k, k < old.slots.length ∧ foN old k ≠ EMPTYN ∧ foN old k = x
  · simp [hx]
  · simp only [hx, if_false]
    symm
    apply hB.absent
    intro i hi hix
    apply hx
    refine ⟨i, ?_, hi, hix⟩
    by_contra hh
    exact hi (foN_out_of_range old i (by omega))

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

theorem grow_body_spec (old : Line) (c : Nat → Nat) (hB : LineBody old c) (n : Usize)
    (st : Slice (U64 × core.mem.maybe_uninit.MaybeUninit Std.I64) × Line × Usize)
    (hI : GrowInv old c n st) (cf : ControlFlow
      (Slice (U64 × core.mem.maybe_uninit.MaybeUninit Std.I64) × Line × Usize) Line)
    (h : hashtrinity.freq.FreqLine.grow_loop.body n st.1 st.2.1 st.2.2 = ok cf) :
    match cf with
    | .done bg => LineBody bg c ∧ bg.bits.val = old.bits.val + 1 ∧ bg.len.val = old.len.val
    | .cont st' => GrowInv old c n st' := by
  obtain ⟨s, bg, i⟩ := st
  obtain ⟨hi1, hn, hs, hbg, hbits, hlen, hsound⟩ := hI
  dsimp only at hi1 hn hs hbg hbits hlen hsound h
  unfold hashtrinity.freq.FreqLine.grow_loop.body at h
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
      obtain ⟨⟨j, found⟩, hprobe, h⟩ := bind_inv h
      have hpost := probe_spec bg hbg.geom cc j found hprobe
      have hfound : found = false := by
        by_contra hf
        have hft : found = true := by simpa using hf
        obtain ⟨hp1, -⟩ := hpost
        obtain ⟨hfj, hfj'⟩ := hp1 hft
        obtain ⟨k, hk1, hk2, _⟩ := hsound j.val hfj'
        have : k = i.val := hB.geom.distinct k i.val (by rw [hk2, hfj]; exact hccne)
          (by rw [hk2, hfj, hfoi])
        omega
      rw [hfound] at hpost
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
      obtain ⟨⟨⟨_, _⟩, back1⟩, hgm', h⟩ := bind_inv h
      obtain ⟨hgm'1, hgm'2⟩ := get_unchecked_mut_ok _ _ _ _ hgm'
      obtain ⟨mu2, hmu2, h⟩ := bind_inv h
      have hmu2' : mu2 = some v := (Result.ok_injective hmu2).symm
      obtain ⟨i2, hi2, h⟩ := bind_inv h
      obtain ⟨bigger1, hsl, h⟩ := bind_inv h
      obtain ⟨i3, hi3, h⟩ := bind_inv h
      obtain ⟨i4, hi4, h⟩ := bind_inv h
      have hcf : cf = ControlFlow.cont (imb (hashtrinity.EMPTY, mu1), { bigger1 with len := i3 }, i4) :=
        (Result.ok_injective h).symm
      subst hcf
      obtain ⟨hsl1, hsl2⟩ := set_lane_ok _ _ _ _ hsl
      have htag := tag_spec _ _ _ hi2
      have hi3v := len_ok _ _ hi3
      have hi4v : i4.val = i.val + 1 := by
        have := (uadd_ok hi4).1
        have e : (1#usize).val = 1 := rfl
        rw [e] at this; exact this
      have hs2 : back1 (cc, mu2) = bg.slots.set j (cc, some v) := by rw [hgm'2, hmu2']
      have hb1_slots : bigger1.slots = bg.slots.set j (cc, some v) := by
        rw [hsl2]; exact hs2
      have hb1_ctrl : bigger1.ctrl = bg.ctrl.set j i2 := by rw [hsl2]
      have hb1_bits : bigger1.bits = bg.bits := by rw [hsl2]
      have hb1_gs : bigger1.gshift = bg.gshift := by rw [hsl2]
      have hb1_gm : bigger1.gmask = bg.gmask := by rw [hsl2]
      have hb1_len : bigger1.len = bg.len := by rw [hsl2]
      -- facts about the old slot `i`
      have hmu0 : mu0 = some v := by rw [← him2, hmu]
      have hold_vals := hB.vals i.val (by rw [hfoi]; exact hccne)
      obtain ⟨v0, hv0, hcv0, hpos0⟩ := hold_vals
      have hv0v : v0 = v := by
        rw [hvli, hmu0] at hv0; exact (Option.some_injective _ hv0).symm
      have hcond : ∃ k, k < i.val + 1 ∧ foN old k ≠ EMPTYN ∧ foN old k = cc.val :=
        ⟨i.val, by omega, by rw [hfoi]; exact hccne, hfoi⟩
      have hc'f : (ci old c (i.val + 1) cc.val : Int) = v.val := by
        unfold ci; rw [if_pos hcond, ← hfoi]; rw [hv0v] at hcv0; exact hcv0
      have hc'pos : 0 < ci old c (i.val + 1) cc.val := by
        unfold ci; rw [if_pos hcond, ← hfoi]; exact hpos0
      have hc'o : ∀ x, x ≠ cc.val → ci old c (i.val + 1) x = ci old c i.val x := by
        intro x hx
        unfold ci
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
      have hBL : LineBody ({ bigger1 with len := i3 } : Line) (ci old c (i.val + 1)) :=
        body_insert_fresh bg { bigger1 with len := i3 } _ _ hbg cc hccne j v i2 htag hpost
          hb1_slots hb1_ctrl hb1_bits hb1_gs hb1_gm (by show i3.val = _; rw [hi3v, hb1_len])
          hc'f hc'pos hc'o
      obtain ⟨hG', hfo', hcn'⟩ := geom_insert bg { bigger1 with len := i3 } hbg.geom cc hccne j
        (some v) i2 htag hpost hb1_slots hb1_ctrl hb1_bits hb1_gs hb1_gm
      have hjslt : j.val < bg.slots.length := hpost.2 rfl |>.2.1
      have hvl := vlN_slots_set bg { bigger1 with len := i3 } j (cc, some v) hjslt hb1_slots
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
      unfold GrowInv
      dsimp only
      refine ⟨by omega, hn, ?_, ?_, ?_, ?_, ?_⟩
      · intro k
        have hsi' : i.val < s.length := (List.getElem?_eq_some_iff.mp hsi).1
        have hb3 : imb (hashtrinity.EMPTY, mu1) = s.set i (hashtrinity.EMPTY, mu1) := by
          rw [him2', hs1]
        rw [hi4v, hb3, slice_set_getElem s i _ k hsi', hs k]
        by_cases hk : k = i.val
        · subst hk; simp [hold, hmu1]
        · by_cases hlt2 : k < i.val
          · simp [hk, hlt2, show k < i.val + 1 by omega]
          · simp [hk, hlt2, show ¬ k < i.val + 1 by omega]
      · rw [hi4v]; exact hBL
      · show bigger1.bits.val = _; rw [hb1_bits]; exact hbits
      · show i3.val = _
        rw [hi4v, hi3v, hb1_len, hlen, hPi, Finset.card_insert_of_notMem hPn]
      · intro j' hj'
        rw [hi4v]
        by_cases hh : j' = j.val
        · subst hh
          refine ⟨i.val, by omega, ?_, ?_⟩
          · rw [hfo' j.val, if_pos rfl]; exact hfoi
          · rw [hvl, if_pos rfl, hvli, hmu0]
        · have hj'' : foN bg j' ≠ EMPTYN := by
            rw [hfo' j', if_neg hh] at hj'; exact hj'
          obtain ⟨k, hk1, hk2, hk3⟩ := hsound j' hj''
          refine ⟨k, by omega, ?_, ?_⟩
          · rw [hfo' j', if_neg hh]; exact hk2
          · rw [hvl, if_neg hh]; exact hk3
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
      refine ⟨by omega, hn, ?_, ?_, hbits, ?_, ?_⟩
      · intro k
        rw [hs k, hi1v]
        by_cases hki : k = i.val
        · subst hki
          simp only [lt_irrefl, if_false, Nat.lt_succ_self, if_true]
          rw [hold]
          simp
          -- the blanked slot of a dead slot is itself
          have : cc = hashtrinity.EMPTY := UScalar.eq_of_val_eq (by rw [hccE, EMPTY_val])
          rw [this]
        · by_cases hk : k < i.val
          · simp [hk, show k < i.val + 1 by omega]
          · simp [hk, show ¬ k < i.val + 1 by omega]
      · rw [hci]; exact hbg
      · rw [hP]; exact hlen
      · intro j hj
        obtain ⟨k, hk1, hk2, hk3⟩ := hsound j hj
        exact ⟨k, by omega, hk2, hk3⟩
  · rw [if_neg hlt] at h
    have hcf : cf = ControlFlow.done bg := (Result.ok_injective h).symm
    subst hcf
    have hlt' : ¬ i.val < n.val := fun hh => hlt ((UScalar.lt_equiv i n).mpr hh)
    have hin : i.val = old.slots.length := by omega
    refine ⟨?_, hbits, ?_⟩
    · rw [hin, ci_full old c hB] at hbg; exact hbg
    · rw [hlen, hin, Pset_full, ← hB.len_eq]


theorem grow_loop_spec (old : Line) (c : Nat → Nat) (hB : LineBody old c) (bg0 : Line)
    (hE : EmptyLine bg0) (hbits : bg0.bits.val = old.bits.val + 1) (n : Usize)
    (hn : n.val = old.slots.length) (r : Line)
    (h : hashtrinity.freq.FreqLine.grow_loop old.slots bg0 0#usize n = ok r) :
    LineBody r c ∧ r.bits.val = old.bits.val + 1 ∧ r.len.val = old.len.val := by
  unfold hashtrinity.freq.FreqLine.grow_loop at h
  refine loop_pc (GrowInv old c n)
    (fun (y : Line) => LineBody y c ∧ y.bits.val = old.bits.val + 1 ∧ y.len.val = old.len.val)
    _ ?_ (old.slots, bg0, 0#usize) ?_ r h
  · intro st hst cf hcf
    obtain ⟨s, bg, i⟩ := st
    dsimp only at hcf
    cases cf with
    | done y => exact grow_body_spec old c hB n (s, bg, i) hst (.done y) hcf
    | cont x => exact grow_body_spec old c hB n (s, bg, i) hst (.cont x) hcf
  · have hbl := (empty_rep bg0 hE).toLineBody
    unfold GrowInv
    dsimp only
    refine ⟨by simp, hn, ?_, ?_, hbits, ?_, ?_⟩
    · intro k; simp
    · have e : (0#usize).val = 0 := rfl
      rw [e, ci_zero]; exact hbl
    · have e : (0#usize).val = 0 := rfl
      rw [e, hE.len0]; simp [Pset]
    · intro j hj; exact absurd (hE.foE j) hj

theorem grow_spec (l : Line) (c : Nat → Nat) (hR : LineBody l c) (l1 : Line)
    (h : hashtrinity.freq.FreqLine.grow l = ok l1) :
    LineBody l1 c ∧ l1.bits.val = l.bits.val + 1 ∧ l1.len.val = l.len.val := by
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
  exact grow_loop_spec l c hR bigger hE hbits (Slice.len l.slots) (Slice.len_val _) l1 h


/-! ## `insert_with` -/

theorem iadd_ok {ty} {x y z : IScalar ty} (h : x + y = ok z) :
    z.val = x.val + y.val := by
  have := IScalar.add_equiv x y
  simp only [h, Result.match.ok] at this
  exact this.2.1

theorem ge_val {ty} (x y : UScalar ty) : (x >= y) ↔ y.val ≤ x.val := by
  exact UScalar.le_equiv y x

theorem insert_with_spec {K : Type} (inst : hashtrinity.algebra.Coord K) (l : Line)
    (c : Nat → Nat) (hR : LineRep l c) (f : U64) (hf : f.val ≠ EMPTYN) (new : Std.I64)
    (hnew : new.val = 1) (l' : Line)
    (h : hashtrinity.freq.FreqLine.insert_with
      (hashtrinity.freq.FreqMapKI64.add.closure.Insts.CoreOpsFunctionFnOncePairI64I64I64 inst)
      l f new () = ok l') :
    LineRep l' (fun x => if x = f.val then c x + 1 else c x) := by
  unfold hashtrinity.freq.FreqLine.insert_with at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨i1, hi1, h⟩ := bind_inv h
  obtain ⟨i3, hi3, h⟩ := bind_inv h
  obtain ⟨self1, hself1, h⟩ := bind_inv h
  have hiv := len_ok _ _ hi
  have hi1v := (umul_ok hi1).1
  have hi3v := (umul_ok hi3).1
  have e8 : (8#usize).val = 8 := rfl
  have e7 : (7#usize).val = 7 := rfl
  rw [e8] at hi1v
  rw [e7, Slice.len_val] at hi3v
  have hS1 : LineRep self1 c ∧ 8 * (self1.len.val + 1) < 7 * self1.slots.length := by
    by_cases hge : i1 >= i3
    · rw [if_pos hge] at hself1
      have hge' := (ge_val i1 i3).mp hge
      obtain ⟨hgb, hgbits, hglen⟩ := grow_spec l c hR.toLineBody self1 hself1
      have hsz1 := hgb.geom.size
      have hsz := hR.geom.size
      have hld := hR.load
      have hb4 := hR.geom.bits4
      have hpow : 2 ^ self1.bits.val = 2 * 2 ^ l.bits.val := by rw [hgbits]; ring
      have h16 : 16 ≤ 2 ^ l.bits.val := by
        calc 16 = 2^4 := by norm_num
          _ ≤ 2 ^ l.bits.val := Nat.pow_le_pow_right (by norm_num) hb4
      refine ⟨⟨hgb, ?_⟩, ?_⟩
      · rw [hglen, hsz1, hpow, ← hsz]; omega
      · rw [hglen, hsz1, hpow, ← hsz]; omega
    · rw [if_neg hge] at hself1
      have := Result.ok_injective hself1
      subst this
      have hlt : ¬ i3.val ≤ i1.val := fun hh => hge ((ge_val i1 i3).mpr hh)
      exact ⟨hR, by omega⟩
  obtain ⟨hR1, hroom⟩ := hS1
  obtain ⟨⟨i4, found, tag⟩, hprobe, h⟩ := bind_inv h
  obtain ⟨htagv, hpost⟩ := probe_tagged_spec self1 hR1.geom f i4 found tag hprobe
  have hslen1 : 8 * (self1.len.val + 1) < 7 * self1.slots.length := hroom
  change (if found = true then _ else _) = _ at h
  by_cases hfound : found = true
  · rw [if_pos hfound] at h
    obtain ⟨⟨⟨i5, mu⟩, back⟩, hgm, h⟩ := bind_inv h
    obtain ⟨hgm1, hgm2⟩ := get_unchecked_mut_ok _ _ _ _ hgm
    obtain ⟨old, hold, h⟩ := bind_inv h
    have hmu : mu = some old := assume_init_read_ok _ _ hold
    obtain ⟨b, hb, h⟩ := bind_inv h
    have hbf : b = false := by
      rw [needs_drop_i64] at hb; exact (Result.ok_injective hb).symm
    change (if b = true then _ else _) = _ at h
    rw [if_neg (by simp [hbf])] at h
    obtain ⟨t, ht, h⟩ := bind_inv h
    have ht' : new + old = ok t := ht
    have htv := iadd_ok ht'
    obtain ⟨mu1, hmu1, h⟩ := bind_inv h
    have hmu1' : mu1 = some t := (Result.ok_injective hmu1).symm
    have hl' : l' = { self1 with slots := back (i5, mu1) } := (Result.ok_injective h).symm
    obtain ⟨hfo, hfne⟩ := hpost.1 hfound
    have hfoi5 : foN self1 i4.val = i5.val := foN_of_get _ _ _ hgm1
    have hi5f : i5 = f := UScalar.eq_of_val_eq (by rw [← hfoi5, hfo])
    obtain ⟨v, hv, hcv, hpv⟩ := hR1.vals i4.val hfne
    have hvl : vlN self1 i4.val = mu := by unfold vlN; rw [hgm1]
    have hvo : v = old := by
      rw [hvl, hmu] at hv; exact (Option.some_injective _ hv).symm
    subst hvo
    rw [hfo] at hcv hpv
    have hsl' : l'.slots = self1.slots.set i4 (f, some t) := by
      rw [hl']; show back (i5, mu1) = _; rw [hgm2, hmu1', hi5f]
    have hbody := body_update self1 l' c (fun x => if x = f.val then c x + 1 else c x) hR1.toLineBody
      f i4 hfo hf t hsl' (by rw [hl']) (by rw [hl']) (by rw [hl']) (by rw [hl']) (by rw [hl'])
      (by simp [htv, hnew, hcv]; omega) (by simp) (fun x hx => by simp [hx])
    refine ⟨hbody, ?_⟩
    have hlen' : l'.len = self1.len := by rw [hl']
    have hsln : l'.slots.length = self1.slots.length := by
      show l'.slots.val.length = _; rw [hsl', Slice.set_val_eq]; simp
    rw [hlen', hsln]; exact hR1.load
  · rw [if_neg hfound] at h
    have hfalse : found = false := by simpa using hfound
    rw [hfalse] at hpost
    obtain ⟨⟨⟨_, _⟩, back⟩, hgm, h⟩ := bind_inv h
    obtain ⟨hgm1, hgm2⟩ := get_unchecked_mut_ok _ _ _ _ hgm
    obtain ⟨mu, hmu, h⟩ := bind_inv h
    have hmu' : mu = some new := (Result.ok_injective hmu).symm
    obtain ⟨self2, hsl, h⟩ := bind_inv h
    obtain ⟨i5, hi5, h⟩ := bind_inv h
    have hl' : l' = { self2 with len := i5 } := (Result.ok_injective h).symm
    obtain ⟨hsl1, hsl2⟩ := set_lane_ok _ _ _ _ hsl
    have hi5v := len_ok _ _ hi5
    have hc0 : c f.val = 0 := by
      apply hR1.absent
      exact (hpost.2 rfl).2.2.2
    have hs2 : back (f, mu) = self1.slots.set i4 (f, some new) := by rw [hgm2, hmu']
    have hbody := body_insert_fresh self1 l' c (fun x => if x = f.val then c x + 1 else c x)
      hR1.toLineBody f hf i4 new tag htagv hpost
      (by rw [hl', hsl2]; exact hs2) (by rw [hl', hsl2]) (by rw [hl', hsl2]) (by rw [hl', hsl2])
      (by rw [hl', hsl2]) (by rw [hl']; show i5.val = _; rw [hi5v]; show self2.len.val + 1 = _; rw [hsl2])
      (by simp [hnew, hc0]) (by simp) (fun x hx => by simp [hx])
    refine ⟨hbody, ?_⟩
    have hlen' : l'.len.val = self1.len.val + 1 := by
      rw [hl']; show i5.val = _; rw [hi5v, hsl2]
    have hsln : l'.slots.length = self1.slots.length := by
      rw [hl', hsl2]
      show (back (f, mu)).val.length = _
      rw [hs2, Slice.set_val_eq]; simp
    rw [hlen', hsln]; exact hslen1


/-! ## `add` -/

/-- The fold `FreqMap::freq` applies to a `u64` key: `min(k, EMPTY - 2)`.  It is the identity on
`k ≤ 2^64 - 3` and sends the three keys `2^64 - 3, 2^64 - 2, 2^64 - 1` to `2^64 - 3`. -/
def foldN (k : Nat) : Nat := min k (2^64 - 3)

theorem foldN_ne_empty (k : Nat) : foldN k ≠ EMPTYN := by
  unfold foldN EMPTYN; omega

theorem freq_spec (k i : U64)
    (h : hashtrinity.freq.FreqMap.freq Std.I64 U64.Insts.Apeiron_wordHashtrinityAlgebraCoord k = ok i) :
    i.val = foldN k.val := by
  unfold hashtrinity.freq.FreqMap.freq at h
  obtain ⟨x, hx, h2⟩ := bind_inv h
  obtain ⟨i1, hi1, h3⟩ := bind_inv h2
  have hx' : x = k := (Result.ok_injective hx).symm
  subst hx'
  have e2 : (2#u64).val = 2 := rfl
  have h1 := usub_ok hi1
  rw [e2, EMPTY_val] at h1
  have := min_u64_ok _ _ _ h3
  rw [this]
  unfold foldN EMPTYN at *
  omega

theorem add_spec (m m' : hashtrinity.freq.FreqMap U64 Std.I64) (c : Nat → Nat)
    (hR : LineRep m.inner c) (k : U64)
    (h : hashtrinity.freq.FreqMapKI64.add U64.Insts.Apeiron_wordHashtrinityAlgebraCoord m k 1#i64 = ok m') :
    LineRep m'.inner (fun x => if x = foldN k.val then c x + 1 else c x) := by
  unfold hashtrinity.freq.FreqMapKI64.add at h
  obtain ⟨i, hi, h⟩ := bind_inv h
  obtain ⟨fl, hfl, h⟩ := bind_inv h
  have hm' : m' = { m with inner := fl } := (Result.ok_injective h).symm
  subst hm'
  have hiv := freq_spec _ _ hi
  have hf : i.val ≠ EMPTYN := by rw [hiv]; exact foldN_ne_empty _
  have := insert_with_spec U64.Insts.Apeiron_wordHashtrinityAlgebraCoord m.inner c hR i hf 1#i64 rfl fl hfl
  rw [hiv] at this
  exact this



/-! ## The representation, stated at the brief's names -/

/-- `m` represents the count function `c` on folds: `c f` is the number of increments whose key has
fold `f` (`FreqMap::freq`); every live slot holds a distinct fold with value `c fold > 0`, the folds
with no live slot have count `0`, `len` is the number of live slots, the control bytes agree with the
slots, every live fold sits at the end of a run of full groups on its probe sequence, and the load is
below the growth threshold `8·len < 7·slots`.  `c` is indexed by the fold, not the key. -/
def FreqRep (m : hashtrinity.freq.FreqMap U64 Std.I64) (c : Nat → Nat) : Prop :=
  LineRep m.inner c

/-- The empty table `FreqMap::with_capacity` returns satisfies `FreqRep` with the zero count. -/
theorem freqRep_empty (cap : Usize) (m : hashtrinity.freq.FreqMap U64 Std.I64)
    (h : hashtrinity.freq.FreqMap.with_capacity Std.I64 U64.Insts.Apeiron_wordHashtrinityAlgebraCoord cap
      = ok m) :
    FreqRep m (fun _ => 0) :=
  with_capacity_rep _ cap m h

theorem add_preserves (m m' : hashtrinity.freq.FreqMap U64 Std.I64) (c : Nat → Nat)
    (hR : FreqRep m c) (k : U64)
    (h : hashtrinity.freq.FreqMapKI64.add U64.Insts.Apeiron_wordHashtrinityAlgebraCoord m k 1#i64 = ok m') :
    FreqRep m' (Function.update c (foldN k.val) (c (foldN k.val) + 1)) := by
  have := add_spec m m' c hR k h
  have e : (fun x => if x = foldN k.val then c x + 1 else c x)
      = Function.update c (foldN k.val) (c (foldN k.val) + 1) := by
    funext x
    by_cases hx : x = foldN k.val
    · subst hx; simp
    · simp [hx]
  rw [e] at this
  exact this


end Apeiron.CountAggregate
