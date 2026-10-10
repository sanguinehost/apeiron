-- Non-vacuity of `build_refines`.
import Map.ExecutedBuild
open Aeneas Aeneas.Std Result
set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

/-!
# Non-vacuity of `build_refines`

`build_refines` assumes `xs.length < nilN` and `execBuild n xs = ok s`.  This file exhibits both at
one instance and the extracted execution returning `ok`: `n = 4`, `xs = [(EMPTY, 5), (TOMB, 7)]`
(distinct reserved keys, so both inserts take the side-store path).

* `bits_for_four`: the extracted `bits_for 4` is `ok 3` (`numBits ∈ {32, 64}` handled by val-level
  reasoning, no `decide` over a word).
* `with_bits3_ex`: `with_bits 3` returns `ok s0` with an empty side store (`coords = []`, `cap = 0`).
* `push1_spec` / `push2_spec`: the first `SideStore::push` grows (`cap = 4`), the second does not.
* `insert_sent_ex`: a reserved key whose `find` is `none` and whose `push` is `ok ss` inserts `ok`.
* `build_witness_exec`: `execBuild 4 [(EMPTY, 5), (TOMB, 7)] = ok s`.
* `build_refines_witness`: with `hlen` and distinct keys, `build_refines` yields `MapRep` of the
  built map.

Partial-correctness statement of `build_refines` is thereby shown non-vacuous at this instance.  Not
covered: a main-table (non-reserved) key, a resize, or an overflow node.
-/

namespace Apeiron.MapExecuted
open apeiron_word

open apeiron_word

theorem usize_mk_val (n : Nat) (h : n ≤ Usize.max) :
    (⟨BitVec.ofNat _ n⟩ : Usize).val = n := by
  show (BitVec.ofNat _ n).toNat = n
  rw [BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  have h1 := Usize.max_def
  rw [Usize.numBits_def] at h1
  have : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
  omega

theorem usize_max_ge : 2 ^ 32 - 1 ≤ Usize.max := by
  rw [Usize.max_def, Usize.numBits_def]
  rcases System.Platform.numBits_eq with h | h
  · show _ ≤ 2 ^ System.Platform.numBits - 1; rw [h]
  · show _ ≤ 2 ^ System.Platform.numBits - 1; rw [h]; decide

theorem sat_mul_4_4 : core.num.Usize.saturating_mul 4#usize 4#usize = ok 16#usize := by
  unfold core.num.Usize.saturating_mul
  congr 1
  apply UScalar.eq_of_val_eq
  rw [usize_mk_val _ (by have := usize_max_ge; omega)]
  simp
  have := usize_max_ge; omega

theorem max_usize (a b : Usize) :
    core.cmp.Ord.max.default core.cmp.OrdUsize a b = ok (if b.val < a.val then a else b) := by
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body
  simp [core.cmp.impls.PartialOrdUsize.lt, liftFun2]
  split <;> rfl

theorem max_u32 (a b : U32) :
    core.cmp.Ord.max.default core.cmp.OrdU32 a b = ok (if b.val < a.val then a else b) := by
  unfold core.cmp.Ord.max.default core.cmp.Ord.max_body
  simp [core.cmp.impls.PartialOrdU32.lt, liftFun2]
  split <;> rfl

theorem npot_8 : core.num.Usize.next_power_of_two 8#usize = ok 8#usize := by
  unfold core.num.Usize.next_power_of_two
  simp
  have := usize_max_ge
  rw [if_pos (by omega)]
  congr 1
  apply UScalar.eq_of_val_eq
  show (⟨BitVec.ofNat _ 8⟩ : Usize).val = (8#usize).val
  rw [usize_mk_val 8 (by omega)]
  simp

theorem lz_8 : (core.num.Usize.leading_zeros 8#usize).val = System.Platform.numBits - 4 := by
  unfold core.num.Usize.leading_zeros
  simp [BitVec.leadingZeros]
  have hne : (8#System.Platform.numBits ≠ 0#System.Platform.numBits) := by
    intro hh
    have := congrArg BitVec.toNat hh
    rw [BitVec.toNat_ofNat, BitVec.toNat_ofNat] at this
    rcases System.Platform.numBits_eq with h | h <;> rw [h] at this <;> norm_num at this
  have hlog : Nat.log 2 8 = 3 := by
    rw [show (8:Nat) = 2 ^ 3 by norm_num]; exact Nat.log_pow (by norm_num) 3
  have h8 : 8 % 2 ^ System.Platform.numBits = 8 := by
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> norm_num
  rw [if_neg hne, h8, hlog]
  rcases System.Platform.numBits_eq with h | h
  · simp [h]; rfl
  · simp [h]; rfl


theorem bits_for_four_spec : hashtrinity.bits_for 4#usize ⦃ b => b.val = 3 ⦄ := by
  unfold hashtrinity.bits_for
  unfold hashtrinity.LOAD_DEN hashtrinity.LOAD_NUM
  rw [sat_mul_4_4]
  step*
  have hneed : need = 6#usize := by
    apply UScalar.eq_of_val_eq; simp; omega
  subst hneed
  rw [max_usize]
  have h68 : ¬ ((8#usize).val < (6#usize).val) := by simp
  simp only [h68, if_false, bind_tc_ok]
  rw [npot_8]
  simp only [bind_tc_ok, lift]
  have hbits : core.num.Usize.BITS.val = System.Platform.numBits := by
    rfl
  have hlz := lz_8
  have hnb : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
  have hge : 32 ≤ System.Platform.numBits := by
    rcases System.Platform.numBits_eq with h | h <;> omega
  step*
  all_goals (try simp only [hnb] at *)
  rw [max_u32]
  have h6 : i6.val = 3 := by omega
  have hc : ¬ ((3#u32).val < i6.val) := by simp; omega
  simp only [hc, if_false]
  rw [WP.spec_ok]; rfl


theorem bits_for_four : hashtrinity.bits_for 4#usize = ok 3#u32 := by
  obtain ⟨b, hb, hv⟩ := WP.spec_imp_exists bits_for_four_spec
  rw [hb]; congr 1
  apply UScalar.eq_of_val_eq; rw [hv]; rfl

theorem alloc_uninit_ok (T : Type) (n : Usize) :
    ∃ s, hashtrinity.algebra.alloc_uninit_boxed T n = ok s := by
  unfold hashtrinity.algebra.alloc_uninit_boxed
  have hcap : n.val ≤ (alloc.vec.Vec.with_capacity (core.mem.maybe_uninit.MaybeUninit T) n).cap.val := by
    simp [alloc.vec.Vec.with_capacity]
  simp only [alloc.vec.Vec.set_len, dif_pos hcap, alloc.vec.Vec.into_boxed_slice, bind_tc_ok]
  exact ⟨_, rfl⟩

theorem shl_1_3 : (1#usize <<< 3#u32) = ok 8#usize := by
  show UScalar.shiftLeft_UScalar 1#usize 3#u32 = _
  unfold UScalar.shiftLeft_UScalar UScalar.shiftLeft
  have h3 : (3#u32).val < UScalarTy.Usize.numBits := by
    show 3 < System.Platform.numBits
    rcases System.Platform.numBits_eq with h | h <;> rw [h] <;> decide
  simp only [h3, if_true]
  congr 1
  apply UScalar.eq_of_val_eq
  have := usize_max_ge
  have hm := Usize.max_def
  rw [Usize.numBits_def] at hm
  have hp : 0 < 2 ^ UScalarTy.Usize.numBits := Nat.two_pow_pos _
  show ((1#usize).bv.shiftLeft (3#u32).val).toNat = 8
  rw [BitVec.shiftLeft_eq, BitVec.toNat_shiftLeft]
  have h1 : (1#usize).bv.toNat = 1 := rfl
  have h3v : (3#u32).val = 3 := rfl
  rw [h1, h3v, Nat.shiftLeft_eq, Nat.mod_eq_of_lt (by omega)]

theorem from_elem_ok (n : Usize) :
    ∃ v, alloc.vec.from_elem core.clone.CloneU64 hashtrinity.EMPTY n = ok v := by
  obtain ⟨v', h1, hP, _⟩ := WP.spec_imp_exists (alloc.vec.from_elem_spec core.clone.CloneU64 hashtrinity.EMPTY n rfl)
  exact ⟨v', h1⟩


theorem slice_index_ex {α : Type} (s : Slice α) (i : Usize) (h : i.val < s.val.length) :
    ∃ x, s.index_usize i = ok x := by
  unfold Slice.index_usize
  have : s[i]? = some s.val[i.val] := by
    rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem h]
  rw [this]; exact ⟨_, rfl⟩

theorem vec_index_ex {α : Type} (v : alloc.vec.Vec α) (i : Usize) (h : i.val < v.val.length) :
    ∃ x, alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice α) v i = ok x := by
  unfold alloc.vec.Vec.index
  exact slice_index_ex v.slice i h

theorem gum_ex {α : Type} (s : Slice α) (i : Usize) (h : i.val < s.val.length) :
    ∃ x back, core.slice.Slice.get_unchecked_mut (core.slice.index.SliceIndexUsizeSlice α) s i
      = ok (x, back) := by
  unfold core.slice.Slice.get_unchecked_mut
  simp only [core.slice.index.Usize.index_mut, Slice.index_mut_usize]
  obtain ⟨y, hy⟩ := slice_index_ex s i h
  rw [hy]
  exact ⟨y, s.set i, by simp⟩

theorem vec_push_ex {α : Type} (v : alloc.vec.Vec α) (x : α) (h : v.val.length < Usize.max) :
    ∃ v1, alloc.vec.Vec.push v x = ok v1 := by
  unfold alloc.vec.Vec.push
  dsimp only
  rw [dif_pos (by simp only [Bool.or_eq_true, decide_eq_true_eq]; right; omega)]
  exact ⟨_, rfl⟩

theorem find_none_empty {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (ss : hashtrinity.side.SideStore U64 I64 KS) (c k : U64) (h : ss.coords.val = []) :
    hashtrinity.side.SideStore.find CI ks ss c k = ok none := by
  unfold hashtrinity.side.SideStore.find hashtrinity.side.SideStore.find_loop
  rw [loop]
  unfold hashtrinity.side.SideStore.find_loop.body
  have hl : ¬ (0#usize < alloc.vec.Vec.len ss.coords) := by
    simp [alloc.vec.Vec.len, h]
  dsimp only
  rw [if_neg hl]
  simp only [bind_ok]


theorem stored_alloc_ex (n : Usize) :
    ∃ g, hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc CI
      core.cmp.EqU64 n = ok g ∧ g.val.length = n.val := by
  obtain ⟨s, hs⟩ := alloc_uninit_ok U64 n
  refine ⟨s, ?_, ?_⟩
  · unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.alloc
    rw [hs]; simp only [bind_tc_ok]
  · rw [alloc_uninit_val U64 n s hs]; simp

theorem push_loop_empty (v : alloc.vec.Vec U64) (g g1 : hashtrinity.algebra.Stored U64)
    (h : v.val = []) :
    hashtrinity.side.SideStore.push_loop SKS v g g1 0#usize = ok g1 := by
  unfold hashtrinity.side.SideStore.push_loop
  rw [loop]
  unfold hashtrinity.side.SideStore.push_loop.body
  have hl : ¬ (0#usize < alloc.vec.Vec.len v) := by
    simp [alloc.vec.Vec.len, h]
  dsimp only
  rw [if_neg hl]
  simp only [bind_ok]

theorem stored_set_ex (g : hashtrinity.algebra.Stored U64) (i : Usize) (k : U64)
    (h : i.val < g.val.length) :
    ∃ g2, hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set CI
      core.cmp.EqU64 g i k = ok g2 ∧ g2.val.length = g.val.length := by
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set
  obtain ⟨x, back, hx⟩ := gum_ex g i h
  have hb := (get_unchecked_mut_ok g i x back hx).2
  have hmu : core.mem.maybe_uninit.MaybeUninit.new k = ok (some k) := rfl
  rw [hmu]
  simp only [bind_tc_ok]
  rw [hx]
  simp only [bind_tc_ok]
  refine ⟨_, rfl, ?_⟩
  rw [hb]
  have := slice_set_length g i (some k)
  simp [Slice.length] at this ⊢

theorem push1_spec (ss : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64)) (c k : U64) (v : I64)
    (h0 : ss.coords.val = []) (h0v : ss.vals.val = []) (hc : ss.cap = 0#usize) :
    hashtrinity.side.SideStore.push CI SKS ss c k v ⦃ ss' =>
      ss'.coords.val = [c] ∧ ss'.cap = 4#usize ∧ ss'.keys.val.length = 4 ∧
        ss'.vals.val.length = 1 ⦄ := by
  unfold hashtrinity.side.SideStore.push
  step*
  have hl0 : ss.coords.len.val = 0 := by simp [alloc.vec.Vec.len, h0]
  have hi1' : i1.val = 0 := by rw [i1_post, hc]; rfl
  have hn' : need.val = 1 := by rw [need_post, hl0]
  rw [max_usize]
  have h1 : ¬ ((4#usize).val < i1.val) := by rw [hi1']; decide
  rw [if_neg h1]
  simp only [bind_tc_ok]
  rw [max_usize]
  have h2 : (need.val < (4#usize).val) := by rw [hn']; decide
  rw [if_pos h2]
  simp only [bind_tc_ok]
  obtain ⟨g, hg, hgl⟩ := stored_alloc_ex 4#usize
  rw [hg]
  simp only [bind_tc_ok]
  rw [push_loop_empty _ _ _ h0]
  simp only [bind_tc_ok]
  obtain ⟨g2, hg2, hg2l⟩ := stored_set_ex g ss.coords.len k (by rw [hgl, hl0]; decide)
  rw [hg2]
  simp only [bind_tc_ok]
  obtain ⟨v1, hv1⟩ := vec_push_ex ss.coords c (by
    show ss.coords.val.length < Usize.max
    rw [h0]; have := usize_max_ge; simp; omega)
  rw [hv1]
  simp only [bind_tc_ok]
  have hmu : core.mem.maybe_uninit.MaybeUninit.new v = ok (some v) := rfl
  rw [hmu]
  simp only [bind_tc_ok]
  obtain ⟨v2, hv2⟩ := vec_push_ex ss.vals (some v) (by
    show ss.vals.val.length < Usize.max
    rw [h0v]; have := usize_max_ge; simp; omega)
  rw [hv2]
  simp only [bind_tc_ok]
  rw [WP.spec_ok]
  refine ⟨?_, rfl, ?_, ?_⟩
  · rw [vec_push_ok _ _ _ hv1, h0]; rfl
  · rw [hg2l, hgl]; rfl
  · show v2.val.length = 1
    rw [vec_push_ok _ _ _ hv2, h0v]; rfl


theorem push2_spec (ss : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64)) (c k : U64) (v : I64)
    (h1 : ss.coords.val.length = 1) (h1v : ss.vals.val.length = 1) (hc : ss.cap = 4#usize)
    (hk : ss.keys.val.length = 4) :
    hashtrinity.side.SideStore.push CI SKS ss c k v ⦃ _ => True ⦄ := by
  unfold hashtrinity.side.SideStore.push
  step*
  have hl1 : ss.coords.len.val = 1 := by simp [alloc.vec.Vec.len, h1]
  obtain ⟨g2, hg2, -⟩ := stored_set_ex ss.keys ss.coords.len k (by rw [hk, hl1]; decide)
  rw [hg2]
  simp only [bind_tc_ok]
  obtain ⟨v1, hv1⟩ := vec_push_ex ss.coords c (by
    show ss.coords.val.length < Usize.max
    rw [h1]; have := usize_max_ge; omega)
  rw [hv1]
  simp only [bind_tc_ok]
  have hmu : core.mem.maybe_uninit.MaybeUninit.new v = ok (some v) := rfl
  rw [hmu]
  simp only [bind_tc_ok]
  obtain ⟨v2, hv2⟩ := vec_push_ex ss.vals (some v) (by
    show ss.vals.val.length < Usize.max
    rw [h1v]; have := usize_max_ge; omega)
  rw [hv2]
  simp only [bind_tc_ok]
  rw [WP.spec_ok]
  trivial


theorem with_bits3_ex :
    ∃ s0 : Tbl (hashtrinity.algebra.Stored U64),
      hashtrinity.Table.with_bits I64 core.cmp.EqU64 CI SKS 3#u32 = ok s0 ∧
        s0.side.coords.val = [] ∧ s0.side.vals.val = [] ∧ s0.side.cap = 0#usize := by
  unfold hashtrinity.Table.with_bits
  rw [shl_1_3]
  simp only [bind_tc_ok]
  obtain ⟨v, hv⟩ := from_elem_ok 8#usize
  rw [hv]
  simp only [bind_tc_ok]
  unfold alloc.vec.Vec.into_boxed_slice
  simp only [bind_tc_ok]
  obtain ⟨s1, hs1⟩ := alloc_uninit_ok I64 8#usize
  rw [hs1]
  simp only [bind_tc_ok]
  obtain ⟨t, ht, -⟩ := stored_alloc_ex 8#usize
  rw [ht]
  simp only [bind_tc_ok]
  unfold hashtrinity.side.SideStore.empty
  obtain ⟨t0, ht0, -⟩ := stored_alloc_ex 0#usize
  have ht0' : SKS.alloc 0#usize = ok t0 := ht0
  rw [ht0']
  simp only [bind_tc_ok]
  exact ⟨_, rfl, rfl, rfl, rfl⟩

theorem insert_sent_ex (s : Tbl (hashtrinity.algebra.Stored U64)) (k : U64) (v : I64)
    (hk : IsSent k) (ss : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64))
    (hf : hashtrinity.side.SideStore.find CI SKS s.side k k = ok none)
    (hp : hashtrinity.side.SideStore.push CI SKS s.side k k v = ok ss) :
    word_hashtrinity_insert s k v = ok { s with side := ss } := by
  unfold word_hashtrinity_insert hashtrinity.Table.insert_merge_last_write hashtrinity.Table.insert_core
  rw [coord_of_ok]
  simp only [bind_tc_ok]
  rw [is_sentinel_ok]
  simp only [bind_tc_ok]
  rw [if_pos (by simpa using hk)]
  rw [hf]
  simp only [bind_tc_ok]
  rw [hp]
  simp only [bind_tc_ok]
  rfl


theorem add_0_1 : (0#usize + 1#usize) = ok 1#usize := by
  obtain ⟨z, hz, hv⟩ := WP.spec_imp_exists
    (UScalar.add_spec (x := 0#usize) (y := 1#usize) (by scalar_tac))
  rw [hz]; congr 1
  apply UScalar.eq_of_val_eq
  rw [hv]; rfl

theorem find_none_one (ss : hashtrinity.side.SideStore U64 I64 (hashtrinity.algebra.Stored U64))
    (c1 c k : U64) (h : ss.coords.val = [c1]) (hne : c1 ≠ c) :
    hashtrinity.side.SideStore.find CI SKS ss c k = ok none := by
  unfold hashtrinity.side.SideStore.find hashtrinity.side.SideStore.find_loop
  rw [loop]
  unfold hashtrinity.side.SideStore.find_loop.body
  have hl : (0#usize < alloc.vec.Vec.len ss.coords) := by
    simp [alloc.vec.Vec.len, h]
  obtain ⟨x, hx⟩ := vec_index_ex ss.coords 0#usize (by simp [h])
  have hx' := vec_index_ok _ _ _ hx
  rw [h] at hx'
  simp at hx'
  subst hx'
  dsimp only
  rw [if_pos hl, hx]
  simp only [bind_tc_ok]
  rw [if_neg hne]
  simp only [add_0_1, bind_tc_ok, bind_ok]
  rw [loop]
  have hl2 : ¬ (1#usize < alloc.vec.Vec.len ss.coords) := by
    simp [alloc.vec.Vec.len, h]
  rw [if_neg hl2]
  simp only [bind_ok]


theorem EMPTY_ne_TOMB (t : U64) (ht : t.val = TOMBN) : hashtrinity.EMPTY ≠ t := by
  intro h
  have := congrArg UScalar.val h
  rw [EMPTY_val, ht] at this
  unfold EMPTYN TOMBN at this
  omega

/-- The joint witness: `n = 4`, two distinct reserved keys. -/
theorem build_witness_exec :
    ∃ (t : U64) (s : Tbl (hashtrinity.algebra.Stored U64)),
      hashtrinity.TOMB = ok t ∧ t.val = TOMBN ∧
      execBuild 4#usize [(hashtrinity.EMPTY, 5#i64), (t, 7#i64)] = ok s := by
  obtain ⟨t, htt, htv⟩ := TOMB_ok
  obtain ⟨s0, hs0, h0c, h0v, h0cap⟩ := with_bits3_ex
  have hsE : IsSent hashtrinity.EMPTY := Or.inl EMPTY_val
  have hsT : IsSent t := Or.inr htv
  -- first insert
  obtain ⟨p1, hp1e, hp1c, hp1cap, hp1k, hp1v⟩ := WP.spec_imp_exists
    (push1_spec s0.side hashtrinity.EMPTY hashtrinity.EMPTY 5#i64 h0c h0v h0cap)
  have hf1 := find_none_empty SKS s0.side hashtrinity.EMPTY hashtrinity.EMPTY h0c
  have hi1 := insert_sent_ex s0 hashtrinity.EMPTY 5#i64 hsE p1 hf1 hp1e
  -- second insert
  obtain ⟨p2, hp2e, -⟩ := WP.spec_imp_exists
    (push2_spec ({ s0 with side := p1 }).side t t 7#i64 (by simp [hp1c]) (by simp [hp1v]) hp1cap hp1k)
  have hf2 := find_none_one ({ s0 with side := p1 }).side hashtrinity.EMPTY t t hp1c (EMPTY_ne_TOMB t htv)
  have hi2 := insert_sent_ex { s0 with side := p1 } t 7#i64 hsT p2 hf2 hp2e
  refine ⟨t, { { s0 with side := p1 } with side := p2 }, htt, htv, ?_⟩
  unfold execBuild word_hashtrinity_with_capacity hashtrinity.TableKVStored.with_capacity
  rw [bits_for_four]
  simp only [bind_tc_ok]
  rw [hs0]
  simp only [bind_tc_ok, execFold, List.foldlM_cons, List.foldlM_nil]
  rw [hi1]
  simp only [bind_tc_ok]
  rw [hi2]
  simp only [bind_tc_ok]
  rfl

/-- Non-vacuity of `build_refines`: at `n = 4` and two distinct reserved keys, `hlen` holds, the
extracted build returns `ok s`, and the theorem's conclusion is then a `MapRep` of the built map. -/
theorem build_refines_witness :
    ∃ (xs : List (U64 × I64)) (s : Tbl (hashtrinity.algebra.Stored U64)),
      xs.length = 2 ∧ xs.length < nilN ∧ (xs.map Prod.fst).Nodup ∧
      execBuild 4#usize xs = ok s ∧
      MapRep s (Apeiron.Table.build (fun new _ => new) 0#i64 xs) (fun c => c ∈ xs.map Prod.fst) ∧
        arenaLen s + mainCount s ≤ xs.length := by
  obtain ⟨t, s, -, htv, hs⟩ := build_witness_exec
  have hlen : ([(hashtrinity.EMPTY, 5#i64), (t, 7#i64)] : List (U64 × I64)).length < nilN := by
    show 2 < nilN
    unfold nilN; decide
  refine ⟨_, s, rfl, hlen, ?_, hs, build_refines _ _ hlen s hs⟩
  have hne := EMPTY_ne_TOMB t htv
  simp [hne]

end Apeiron.MapExecuted
