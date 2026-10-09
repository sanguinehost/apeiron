-- Row: `apeiron-ht-insert-DRPGM5Xc97TdAjYF3GVYBs` (APEIRON-HT-INSERT, HT-INSERT-W27): the executed
-- insert is total on a reserved key (the side-store path).
import Map.ExecutedInsertOk
import Map.ExecutedInsertSide
import Map.ExecutedBuildWitness
import Map.ExecutedReadCoordTotal
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed insert on a reserved key

`insert_core_ns_ok` (`Map/ExecutedInsertOk.lean`) discharges the `¬ IsSent` branch of
`insert_core`.  This module discharges the other branch, `IsSent k`: `insert_core` takes
`is_sentinel coord = true`, runs `SideStore::find`, and either appends (`SideStore::push`) or
overwrites a side value in place.

* `side_coords_length_le_two`: from `RepW` the side store holds at most two coordinates — every
  entry is `IsSent` (two possible values) and `side_uniq` forbids duplicates.  This bounds the
  capacity/overflow arithmetic of `push`.
* `side_push_loop_total` / `side_push_total`: `SideStore::push` is total under total `matches`,
  `alloc`, `set` and `take_key`.
* `insert_core_sent_ok`: on a reserved key, `insert_core k v true` returns `(none, s')`.
* `coord_insert_sent_ok` / `coord_insert_total`: the `Regen<u64>` face, on a reserved key and on
  every key (the union with `coord_insert_ok`).
* `coord_insert_sent_witness`: the `grow_witness` state with the key `EMPTY = 2^64 - 1`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-! ## The side store holds at most two coordinates -/

/-- A list with no repeated element, all of whose elements lie in a two-element set, has length at
most two. -/
theorem nodup_of_uniq {α : Type} {l : List α}
    (h : ∀ (i j : Nat) (c : α), l[i]? = some c → l[j]? = some c → i = j) : l.Nodup := by
  induction l with
  | nil => exact List.nodup_nil
  | cons x t ih =>
    rw [List.nodup_cons]
    refine ⟨?_, ih ?_⟩
    · intro hxmem
      obtain ⟨j, hj⟩ := List.mem_iff_getElem?.mp hxmem
      have h0 : (x :: t)[0]? = some x := rfl
      have hj' : (x :: t)[j + 1]? = some x := by rw [List.getElem?_cons_succ]; exact hj
      have := h 0 (j + 1) x h0 hj'
      omega
    · intro i j c hi hj
      have hi' : (x :: t)[i + 1]? = some c := by rw [List.getElem?_cons_succ]; exact hi
      have hj' : (x :: t)[j + 1]? = some c := by rw [List.getElem?_cons_succ]; exact hj
      have := h (i + 1) (j + 1) c hi' hj'
      omega

/-- **`side_coords_length_le_two`.**  Every side coordinate is `IsSent` (`side_val`) and distinct
(`side_uniq`); with only two reserved values the side store holds at most two coordinates. -/
theorem side_coords_length_le_two {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) : s.side.coords.val.length ≤ 2 := by
  obtain ⟨t, ht, htv⟩ := TOMB_ok
  have hnd : s.side.coords.val.Nodup :=
    nodup_of_uniq fun i j c hi hj => hR.side_uniq i j c hi hj
  have hsub : s.side.coords.val ⊆ [hashtrinity.EMPTY, t] := by
    intro x hx
    obtain ⟨i, hi⟩ := List.mem_iff_getElem?.mp hx
    have hx' : IsSent x := (hR.side_val i x hi).1
    rw [List.mem_cons, List.mem_singleton]
    rcases hx' with h | h
    · exact Or.inl ((EMPTY_eq_iff x).mpr h)
    · exact Or.inr (UScalar.eq_of_val_eq (by rw [h, htv]))
  have := (hnd.subperm hsub).length_le
  simpa using this

/-- `32 ≤` the width of `Usize` on this platform. -/
theorem usize_numBits_ge32 : 32 ≤ UScalarTy.Usize.numBits := by
  have h : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
  rw [h]
  rcases System.Platform.numBits_eq with h32 | h64
  · omega
  · omega

/-- `2 < Usize.max` on both supported platforms. -/
theorem usize_max_gt_two : 2 < Usize.max := by
  have := usize_max_ge
  omega

/-! ## Totality of `SideStore::push` -/

/-- `SideStore::push_loop` is total: it walks `v` by an increasing index, reading each key with
`take_key` and writing it with `set`. -/
theorem side_push_loop_total
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (htk : ∀ (keys : KS) (i : Usize) (c : U64), ∃ (k' : U64) (keys' : KS),
      ks.take_key keys i c = ok (k', keys'))
    (v : alloc.vec.Vec U64) (grown grown1 : KS) (i0 : Usize) :
    ∃ r, hashtrinity.side.SideStore.push_loop ks v grown grown1 i0 = ok r := by
  unfold hashtrinity.side.SideStore.push_loop
  refine loop_tot
    (fun (g : KS × KS × Usize) =>
      hashtrinity.side.SideStore.push_loop.body ks v g.1 g.2.1 g.2.2)
    (fun g => v.val.length - g.2.2.val) ?_ ?_ (grown, grown1, i0)
  · rintro ⟨g2, g3, i⟩
    unfold hashtrinity.side.SideStore.push_loop.body
    dsimp only
    by_cases hl : i < alloc.vec.Vec.len v
    · rw [if_pos hl]
      have hv : i.val < v.val.length := by
        have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
      obtain ⟨c, hc⟩ := vec_index_ex v i hv
      rw [hc]; simp only [bind_tc_ok]
      obtain ⟨key, g2', hkey⟩ := htk g2 i c
      rw [hkey]; simp only [bind_tc_ok, uncurry_apply_pair]
      obtain ⟨g3', hset'⟩ := hset g3 i key
      rw [hset']; simp only [bind_tc_ok]
      obtain ⟨i2, hi2, -⟩ := uadd_tot i 1#usize (by
        rw [val_1_usize]
        have hnb : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
        rw [hnb]
        have := vec_len_lt v
        omega)
      rw [hi2]; simp only [bind_tc_ok]
      exact ⟨_, rfl⟩
    · rw [if_neg hl]; exact ⟨_, rfl⟩
  · rintro ⟨g2, g3, i⟩ ⟨g2', g3', i1'⟩ hcf
    unfold hashtrinity.side.SideStore.push_loop.body at hcf
    dsimp only at hcf
    by_cases hl : i < alloc.vec.Vec.len v
    · rw [if_pos hl] at hcf
      obtain ⟨c, hc, hcf⟩ := bind_inv hcf
      obtain ⟨key, hkey, hcf⟩ := bind_inv hcf
      obtain ⟨g3'', hset', hcf⟩ := bind_inv hcf
      obtain ⟨i2, hi2, hcf⟩ := bind_inv hcf
      have hh := Result.ok_injective hcf
      simp only [ControlFlow.cont.injEq, Prod.mk.injEq] at hh
      obtain ⟨-, -, hi2eq⟩ := hh
      rw [← hi2eq]
      show v.val.length - i2.val < v.val.length - i.val
      have hv : i.val < v.val.length := by
        have := (UScalar.lt_equiv _ _).mp hl; rwa [alloc.vec.Vec.len_val] at this
      have hi2v : i2.val = i.val + 1 := (uadd_ok hi2).1
      omega
    · rw [if_neg hl] at hcf
      exact absurd (Result.ok_injective hcf) (by intro hh; cases hh)

/-- **`side_push_total`.**  `SideStore::push` is total on a side store of at most two entries: the
single-entry growth branch's capacity arithmetic is discharged by that bound, `push_loop` copies the
old keys, and the two `Vec::push`es need only the vector bound. -/
theorem side_push_total
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks')
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (htk : ∀ (keys : KS) (i : Usize) (c : U64), ∃ (k' : U64) (keys' : KS),
      ks.take_key keys i c = ok (k', keys'))
    (ss : hashtrinity.side.SideStore U64 I64 KS) (c k : U64) (v : I64)
    (hlen : ss.coords.val.length ≤ 2) (hvl : ss.vals.val.length ≤ 2) :
    ∃ ss', hashtrinity.side.SideStore.push CI ks ss c k v = ok ss' := by
  unfold hashtrinity.side.SideStore.push
  dsimp only
  have hlenv : (alloc.vec.Vec.len ss.coords).val ≤ 2 := by
    rw [alloc.vec.Vec.len_val]; exact hlen
  have hnumBits : 3 < 2 ^ UScalarTy.Usize.numBits := by
    have h : UScalarTy.Usize.numBits = System.Platform.numBits := rfl
    rw [h]
    rcases System.Platform.numBits_eq with h32 | h64
    · rw [h32]; norm_num
    · rw [h64]; norm_num
  obtain ⟨need, hneed, hneedv⟩ := uadd_tot (alloc.vec.Vec.len ss.coords) 1#usize (by
    rw [val_1_usize]; omega)
  rw [hneed]; simp only [bind_tc_ok]
  rw [val_1_usize] at hneedv
  by_cases hg : need > ss.cap
  · rw [if_pos hg]
    have hcap2 : ss.cap.val ≤ 2 := by
      have h1 : ss.cap.val < need.val := (UScalar.lt_equiv _ _).mp hg
      omega
    obtain ⟨i1, hi1, -⟩ := umul_tot ss.cap 2#usize (by
      rw [show (2#usize).val = 2 from rfl, UScalar.max_USize_eq]
      have := usize_max_ge
      omega)
    rw [hi1]; simp only [bind_tc_ok]
    rw [max_usize i1 4#usize]; simp only [bind_tc_ok]
    rw [max_usize _ need]; simp only [bind_tc_ok]
    obtain ⟨grown, hgrown⟩ := halloc _
    rw [hgrown]; simp only [bind_tc_ok]
    obtain ⟨grown1, hgl⟩ := side_push_loop_total ks hset htk ss.coords ss.keys grown 0#usize
    rw [hgl]; simp only [bind_tc_ok]
    obtain ⟨g2', hset'⟩ := hset grown1 (alloc.vec.Vec.len ss.coords) k
    rw [hset']; simp only [bind_tc_ok]
    obtain ⟨v1, hv1⟩ := vec_push_ex ss.coords c (by have := usize_max_gt_two; omega)
    rw [hv1]; simp only [bind_tc_ok]
    rw [show core.mem.maybe_uninit.MaybeUninit.new v = ok (some v) from rfl]
    simp only [bind_tc_ok]
    obtain ⟨v2, hv2⟩ := vec_push_ex ss.vals (some v) (by have := usize_max_gt_two; omega)
    rw [hv2]; simp only [bind_tc_ok]
    exact ⟨_, rfl⟩
  · rw [if_neg hg]
    obtain ⟨g', hset'⟩ := hset ss.keys (alloc.vec.Vec.len ss.coords) k
    rw [hset']; simp only [bind_tc_ok]
    obtain ⟨v1, hv1⟩ := vec_push_ex ss.coords c (by have := usize_max_gt_two; omega)
    rw [hv1]; simp only [bind_tc_ok]
    rw [show core.mem.maybe_uninit.MaybeUninit.new v = ok (some v) from rfl]
    simp only [bind_tc_ok]
    obtain ⟨v2, hv2⟩ := vec_push_ex ss.vals (some v) (by have := usize_max_gt_two; omega)
    rw [hv2]; simp only [bind_tc_ok]
    exact ⟨_, rfl⟩

end

/-! ## Totality of `insert_core` on the reserved path -/

/-- **`insert_core_sent_ok`.**  On a reserved key, `insert_core k v true` returns `(none, s')`:
`SideStore::find` is total (`side_find_total_KS`) and its `none` arm pushes (`side_push_total`),
while its `some i` arm overwrites the in-range side value (`slice_get_unchecked_mut_ex`). -/
theorem insert_core_sent_ok {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)
    (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (halloc : ∀ n : Usize, ∃ ks', ks.alloc n = ok ks')
    (hset : ∀ (keys : KS) (i : Usize) (k : U64), ∃ keys', ks.set keys i k = ok keys')
    (htk : ∀ (keys : KS) (i : Usize) (c : U64), ∃ (k' : U64) (keys' : KS),
      ks.take_key keys i c = ok (k', keys'))
    {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live)
    {k : U64} (hk : IsSent k) (v : I64) :
    ∃ s', hashtrinity.Table.insert_core core.cmp.EqU64 CI ks s k v true = ok (none, s') := by
  unfold hashtrinity.Table.insert_core
  rw [coord_of_ok]
  simp only [bind_tc_ok]
  rw [is_sentinel_ok]
  simp only [bind_tc_ok]
  rw [if_pos (by simpa using hk)]
  obtain ⟨o, hfind⟩ := side_find_total_KS ks hm s.side k k
  rw [hfind]
  simp only [bind_tc_ok]
  have hlen := side_coords_length_le_two ks hR.1
  have hvl0 : s.side.vals.length ≤ 2 := by
    rw [hR.1.side_len_vals]; exact hlen
  have hvl : s.side.vals.val.length ≤ 2 := hvl0
  cases o with
  | none =>
    obtain ⟨ss, hpush⟩ := side_push_total ks halloc hset htk s.side k k v hlen hvl
    refine ⟨{ s with side := ss }, ?_⟩
    rw [hpush]; simp only [bind_tc_ok]
  | some i =>
    have hik : s.side.coords.val[i.val]? = some k :=
      (side_find_spec ks hm s.side k k (some i) hfind).1 i rfl
    have hltc : i.val < s.side.coords.val.length := (List.getElem?_eq_some_iff.mp hik).1
    have hltv0 : i.val < s.side.vals.val.length := by
      rw [show s.side.vals.val.length = s.side.coords.val.length from hR.1.side_len_vals]
      exact hltc
    have hltv : i.val < (s.side.vals.slice).length := hltv0
    obtain ⟨mu, back, hg⟩ := slice_get_unchecked_mut_ex s.side.vals.slice i hltv
    obtain ⟨hmu, hback⟩ := get_unchecked_mut_ok s.side.vals.slice i mu back hg
    have hval : (s.side.vals.slice).val[i.val]? = some (some (m k)) := (hR.1.side_val i k hik).2.2
    have hmu' : mu = some (m k) := by
      rw [hval] at hmu
      exact (Option.some.inj hmu).symm
    subst hmu'
    have hread : core.mem.maybe_uninit.MaybeUninit.assume_init_read (some (m k)) = ok (m k) := rfl
    have hsetlen : ((s.side.vals.slice).set i (some (m k))).length =
        (s.side.vals.slice).length := slice_set_length _ _ _
    obtain ⟨mu2, back2, hg2⟩ := slice_get_unchecked_mut_ex
      ((s.side.vals.slice).set i (some (m k))) i (by rw [hsetlen]; exact hltv)
    let vals' : alloc.vec.Vec (core.mem.maybe_uninit.MaybeUninit I64) :=
      alloc.vec.Vec.ofSlice (back2 (some v))
    let side' : hashtrinity.side.SideStore U64 I64 KS := { s.side with vals := vals' }
    refine ⟨{ s with side := side' }, ?_⟩
    simp only [core.mem.drop, alloc.vec.Vec.deref_mut, lift, bind_tc_ok, uncurry_apply_pair,
      core.mem.maybe_uninit.MaybeUninit.new, ite_true, hg, hread, hback, hg2,
      alloc.vec.Vec.ofSlice]
    rfl

/-! ## The `Regen<u64>` face -/

/-- **`coord_insert_sent_ok`.**  `word_coordmap_insert` (the `Regen<u64>` face) returns `ok` on a
reserved key: `insert_core_sent_ok` at `RKS`, whose `alloc`, `set` and `take_key` are total, then the
`none` arm of `insert_merge_last_write`.  This discharges the premise
`word_coordmap_insert s k v = ok s'` of `coord_insert_side_refines` on that domain. -/
theorem coord_insert_sent_ok {s : Tbl (hashtrinity.algebra.Regen U64)} {m : Apeiron.Table.HMap U64 I64}
    {live : U64 → Prop} (hR : RepW RKS s m live)
    {k : U64} (hk : IsSent k) (v : I64) :
    ∃ s', word_coordmap_insert s k v = ok s' := by
  obtain ⟨s', h⟩ := insert_core_sent_ok RKS RKS_matches
    (fun n => ⟨(), rfl⟩)
    (fun keys i k => ⟨keys, rfl⟩)
    (fun keys i c => ⟨c, keys, by
      change hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
        U64.Insts.Apeiron_wordHashtrinityAlgebraCoordInv core.cmp.EqU64 keys i c = ok (c, keys)
      unfold hashtrinity.algebra.Regen.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.take_key
      change (do let t ← (ok c : Result U64); ok (t, keys)) = ok (c, keys)
      simp only [bind_tc_ok]⟩)
    hR hk v
  refine ⟨s', ?_⟩
  unfold word_coordmap_insert hashtrinity.Table.insert_merge_last_write
  rw [h]
  simp only [bind_tc_ok, uncurry_apply_pair]

/-- **`coord_insert_total`.**  `word_coordmap_insert` returns `ok` for every key: the reserved case
is `coord_insert_sent_ok`, the rest `coord_insert_ok`, on the union of their premises. -/
theorem coord_insert_total {s : Tbl (hashtrinity.algebra.Regen U64)} {m : Apeiron.Table.HMap U64 I64}
    {live : U64 → Prop} (hR : RepW RKS s m live) (hcap : arenaLen s + mainCount s + 4 ≤ nilN)
    (n : Nat) (hHov : Hov s n) (hmain : mainCount s ≤ n) (hn : n * n + n + 4 ≤ nilN)
    (hbits : s.bits.val + 3 ≤ UScalarTy.Usize.numBits)
    (hocc : s.len.val + s.tombs.val ≤ s.coords.length)
    (k : U64) (v : I64) :
    ∃ s', word_coordmap_insert s k v = ok s' := by
  by_cases hk : IsSent k
  · exact coord_insert_sent_ok hR hk v
  · exact coord_insert_ok hR hcap n hHov hmain hn hbits hocc hk v

/-- **`coord_insert_sent_witness`.**  The premises of `coord_insert_sent_ok` hold jointly at a
reachable state — the table `with_bits 1` builds (`grow_witness`), with the reserved key
`EMPTY = 18446744073709551615`. -/
theorem coord_insert_sent_witness :
    ∃ (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
      (live : U64 → Prop) (n : Nat) (s' : Tbl (hashtrinity.algebra.Regen U64)),
      RepW RKS s m live ∧
      arenaLen s + mainCount s + 4 ≤ nilN ∧
      Hov s n ∧ mainCount s ≤ n ∧ n * n + n + 4 ≤ nilN ∧
      s.bits.val + 3 ≤ UScalarTy.Usize.numBits ∧
      s.len.val + s.tombs.val ≤ s.coords.length ∧
      IsSent 18446744073709551615#u64 ∧
      word_coordmap_insert s 18446744073709551615#u64 0#i64 = ok s' := by
  obtain ⟨s, m, live, n, -, hR, hcap, hHov, hmain, hn, hbits, hocc, -⟩ := grow_witness
  have hk : IsSent 18446744073709551615#u64 := by
    unfold IsSent EMPTYN TOMBN
    left
    decide
  obtain ⟨s', h⟩ := coord_insert_sent_ok hR hk 0#i64
  exact ⟨s, m, live, n, s', hR, hcap, hHov, hmain, hn, hbits, hocc, hk, h⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.nodup_of_uniq
#audit Apeiron.MapExecuted.side_coords_length_le_two
#audit Apeiron.MapExecuted.usize_numBits_ge32
#audit Apeiron.MapExecuted.usize_max_gt_two
#audit Apeiron.MapExecuted.side_push_loop_total
#audit Apeiron.MapExecuted.side_push_total
#audit Apeiron.MapExecuted.insert_core_sent_ok
#audit Apeiron.MapExecuted.coord_insert_sent_ok
#audit Apeiron.MapExecuted.coord_insert_total
#audit Apeiron.MapExecuted.coord_insert_sent_witness
