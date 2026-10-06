-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe` (APEIRON-HT-INSERT): the executed insert refines `insertWith`.
import Map.ExecutedInsertSlot
import Map.ExecutedReadUncond
open Aeneas Aeneas.Std Result

/-!
# The executed insert refines `Apeiron.Table.insertWith`

`word_hashtrinity_insert` / `word_coordmap_insert` are `insert_merge_last_write`, which is
`insert_core(k, v, last_write = true)` (`src/hashtrinity/mod.rs`).  On a non-reserved key `insert_core`
runs `grow_for_one` (a `resize` to `bits` or `bits + 1` slots when `len + tombs + 1` reaches `3/4` of
the slots), then the capped probe, then the verdict's write (`ExecutedInsertSlot.lean`).

* `grow_rep`: `grow_for_one` preserves the representation, given `ResizeOK` — the statement that
  `resize` does.  `ResizeOK` is discharged in `Map/ExecutedResize.lean` (`resizeOK_stored`,
  `resizeOK_regen`), which also states the unconditional headlines.
* `insert_ns_rep`: on a non-reserved key, at any key store whose key column `KC` is kept by
  `KeyStore::set`, the insert represents `upd k v m` on `live ∪ {k}`.
* `insert_preserves_of_resize` / `coord_insert_preserves_of_resize`: the `Stored<u64>` and
  `Regen<u64>` headlines; `insert_refines_of_resize` / `coord_insert_refines_of_resize` cover every
  key (reserved keys go through `insert_side_refines`).

The representation is `TableRep ∧ OvWF` (`CoordRep ∧ OvWF`): `TableRep` alone is not preserved —
a tree node with `alive = false` is admitted by `OvInv` and dropped by `resize`'s `into_entries`, and
`insert_new_at` is specified only on an AVL arena (`OInv`).  The hypothesis
`arenaLen s + mainCount s < nilN` keeps the `u32` arena index of a pushed overflow node below `NIL`.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- `resize` to a positive bit count preserves `RepW` and the key column, and does not raise the arena
plus the main count (proved in `Map/ExecutedResize.lean`). -/
def ResizeOK (KC : Slice U64 → KS → Prop) : Prop :=
  ∀ (s s' : Tbl KS) (nb : U32) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop),
    RepW ks s m live → KC s.coords s.keys → arenaLen s + mainCount s < nilN → 0 < nb.val →
    hashtrinity.Table.resize core.cmp.EqU64 CI ks s nb = ok s' →
    RepW ks s' m live ∧ KC s'.coords s'.keys ∧
      arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s

/-- `grow_for_one` either returns the table or resizes it to `bits` or `bits + 1` slots. -/
theorem grow_rep (KC : Slice U64 → KS → Prop) (hRes : ResizeOK ks KC) {s s1 : Tbl KS}
    {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop} (hR : RepW ks s m live)
    (hK : KC s.coords s.keys) (hsmall : arenaLen s + mainCount s < nilN)
    (h : hashtrinity.Table.grow_for_one core.cmp.EqU64 CI ks s = ok s1) :
    RepW ks s1 m live ∧ KC s1.coords s1.keys ∧
      arenaLen s1 + mainCount s1 ≤ arenaLen s + mainCount s := by
  unfold hashtrinity.Table.grow_for_one at h
  obtain ⟨i, -, h⟩ := bind_inv h
  obtain ⟨i1, -, h⟩ := bind_inv h
  obtain ⟨i2, -, h⟩ := bind_inv h
  obtain ⟨i3, -, h⟩ := bind_inv h
  by_cases hge : i2 >= i3
  · rw [if_pos hge] at h
    obtain ⟨i4, -, h⟩ := bind_inv h
    obtain ⟨nb, hnb, h⟩ := bind_inv h
    have hpos : 0 < nb.val := by
      by_cases hlt : s.len < i4
      · rw [if_pos hlt] at hnb
        have := (Result.ok_injective hnb).symm
        subst this
        exact hR.1.bits_pos
      · rw [if_neg hlt] at hnb
        have := (uadd_ok hnb).1
        have h1 : (1#u32).val = 1 := rfl
        omega
    exact hRes s s1 nb m live hR hK hsmall hpos h
  · rw [if_neg hge] at h
    have := (Result.ok_injective h).symm
    subst this
    exact ⟨hR, hK, Nat.le_refl _⟩

/-- Overwriting one list position raises a `countP` by at most one. -/
theorem countP_set_le {α : Type} (p : α → Bool) :
    ∀ (l : List α) (i : Nat) (a : α), (l.set i a).countP p ≤ l.countP p + 1
  | [], _, _ => by simp
  | x :: xs, 0, a => by
    simp only [List.set_cons_zero, List.countP_cons]
    split <;> split <;> omega
  | x :: xs, i + 1, a => by
    simp only [List.set_cons_succ, List.countP_cons]
    have := countP_set_le p xs i a
    split <;> omega

theorem slice_set_set_val {α : Type} (sl : Slice α) (i : Usize) (a b : α) :
    ((sl.set i a).set i b).val = sl.val.set i.val b := by
  rw [Slice.set_val_eq, Slice.set_val_eq, List.set_set]

/-- **The non-reserved insert.**  At a key store whose key column `KC` is kept by `KeyStore::set`
on an `EMPTY` slot, `insert_merge_last_write` on a non-reserved key represents `upd k v m` on
`live ∪ {k}`, given `ResizeOK`. -/
theorem insert_ns_rep (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (KC : Slice U64 → KS → Prop)
    (hKC : ∀ (coords coords' : Slice U64) (keys keys' : KS) (i : Usize) (k : U64),
      KC coords keys → coords'.val = coords.val.set i.val k → ks.set keys i k = ok keys' →
      KC coords' keys')
    (hRes : ResizeOK ks KC) {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : RepW ks s m live) (hK : KC s.coords s.keys) (hsmall : arenaLen s + mainCount s < nilN)
    {k : U64} (hns : ¬ IsSent k) (v : I64) {s' : Tbl KS}
    (h : hashtrinity.Table.insert_merge_last_write core.cmp.EqU64 CI ks s k v = ok s') :
    RepW ks s' (upd k v m) (fun c => live c ∨ c = k) ∧ KC s'.coords s'.keys ∧
      arenaLen s' + mainCount s' ≤ arenaLen s + mainCount s + 1 := by
  unfold hashtrinity.Table.insert_merge_last_write at h
  obtain ⟨⟨o, s1⟩, hic, h⟩ := bind_inv h
  unfold hashtrinity.Table.insert_core at hic
  obtain ⟨coord, hco, hic⟩ := bind_inv hic
  rw [coord_of_ok] at hco
  have hco' := Result.ok_injective hco
  subst hco'
  obtain ⟨b, hb, hic⟩ := bind_inv hic
  rw [is_sentinel_ok] at hb
  have hb' := (Result.ok_injective hb).symm
  subst hb'
  rw [if_neg (by simpa using hns)] at hic
  obtain ⟨t, hg, hic⟩ := bind_inv hic
  obtain ⟨hRt, hKt, hle⟩ := grow_rep ks KC hRes hR hK hsmall hg
  have hst : arenaLen t + mainCount t < nilN := by omega
  obtain ⟨c, hc, hic⟩ := bind_inv hic
  have hcok := pck_spec ks hm (fun _ => avx2_spec_holds) t k k hRt.1.bits_pos hRt.1.len_pow c hc
  have hms : ∀ c, IsSent c → upd k v m c = m c := fun c hc =>
    upd_ne v m (fun (e : c = k) => hns (e ▸ hc))
  rcases c with i | i | _
  · -- `Hit i`: overwrite in place.
    obtain ⟨-, -, hic⟩ := bind_inv hic
    obtain ⟨⟨mu, back⟩, hg1, hic⟩ := bind_inv hic
    obtain ⟨-, hback⟩ := get_unchecked_mut_ok _ _ _ _ hg1
    obtain ⟨old, -, hic⟩ := bind_inv hic
    rw [if_pos rfl] at hic
    obtain ⟨mu1, hmu1, hic⟩ := bind_inv hic
    obtain ⟨⟨mu2, back1⟩, hg2, hic⟩ := bind_inv hic
    obtain ⟨-, hback1⟩ := get_unchecked_mut_ok _ _ _ _ hg2
    obtain ⟨-, -, hic⟩ := bind_inv hic
    have hic' := Result.ok_injective hic
    simp only [Prod.mk.injEq] at hic'
    obtain ⟨rfl, rfl⟩ := hic'
    have hs' := (Result.ok_injective h).symm
    subst hs'
    have hmu1' : mu1 = some v := (Result.ok_injective hmu1).symm
    have hv : (back1 mu1).val = t.vals.val.set i.val (some v) := by
      rw [hback1, hback, hmu1', slice_set_set_val]
    exact ⟨⟨rep_hit ks hRt.1 hns hcok v _ hv, hRt.2⟩, hKt,
      by show arenaLen t + mainCount t ≤ _; omega⟩
  · -- `Vacant i`: fill the `EMPTY` slot.
    obtain ⟨⟨x1, back⟩, hg1, hic⟩ := bind_inv hic
    obtain ⟨-, hback⟩ := get_unchecked_mut_ok _ _ _ _ hg1
    obtain ⟨mu, hmu, hic⟩ := bind_inv hic
    obtain ⟨⟨x2, back1⟩, hg2, hic⟩ := bind_inv hic
    obtain ⟨-, hback1⟩ := get_unchecked_mut_ok _ _ _ _ hg2
    obtain ⟨keys', hset, hic⟩ := bind_inv hic
    obtain ⟨len', hlen', hic⟩ := bind_inv hic
    have hic' := Result.ok_injective hic
    simp only [Prod.mk.injEq] at hic'
    obtain ⟨rfl, rfl⟩ := hic'
    have hs' := (Result.ok_injective h).symm
    subst hs'
    have hmu' : mu = some v := (Result.ok_injective hmu).symm
    have hco : (back k).val = t.coords.val.set i.val k := by rw [hback, Slice.set_val_eq]
    have hv : (back1 mu).val = t.vals.val.set i.val (some v) := by
      rw [hback1, hmu', Slice.set_val_eq]
    have hl : len'.val = t.len.val + 1 := by
      have := (uadd_ok hlen').1
      rw [this]; rfl
    refine ⟨⟨rep_fill ks hRt.1 hns i hcok v _ _ keys' len' hco hv hl, hRt.2⟩,
      hKC t.coords (back k) t.keys keys' i k hKt hco hset, ?_⟩
    show arenaLen t + (back k).val.countP (fun c => decide (¬ IsSent c)) ≤ _
    rw [hco]
    have h1 := countP_set_le (fun c => decide (¬ IsSent c)) t.coords.val i.val k
    unfold mainCount at hle ⊢
    omega
  · -- `Blocked`: the overflow store.
    have hnm := blocked_not_main ks hRt.1 hcok
    have hmm : ∀ c, InMain t c → upd k v m c = m c := fun c hc =>
      upd_ne v m (fun (e : c = k) => hnm (e ▸ hc))
    obtain ⟨oo, hoo, hic⟩ := bind_inv hic
    obtain ⟨⟨o1, asb⟩, has, hic⟩ := bind_inv hic
    unfold core.option.Option.as_mut at has
    have has' := Result.ok_injective has
    simp only [Prod.mk.injEq] at has'
    obtain ⟨rfl, rfl⟩ := has'
    have hov : ∃ ov T, oo = some ov ∧ OvFull ks t m ov T ∧ (∀ c, InT ov T c ↔ InOv t c) ∧
        ov.nodes.val.length < nilN ∧ ov.nodes.val.length ≤ arenaLen t := by
      cases hto : t.overflow with
      | none =>
        rw [if_pos (by simp [hto])] at hoo
        obtain ⟨o2, ho2, hoo⟩ := bind_inv hoo
        have := (Result.ok_injective hoo).symm
        subst this
        obtain ⟨hF, hnone, hl0⟩ := ov_full_new ks t m o2 ho2
        refine ⟨o2, .leaf, rfl, hF, fun c => ⟨fun h => absurd h (hnone c), ?_⟩, ?_, ?_⟩
        · rintro ⟨o', ho', _⟩
          rw [hto] at ho'
          exact absurd ho' (by simp)
        · rw [hl0]; unfold nilN; omega
        · rw [hl0]; exact Nat.zero_le _
      | some ov =>
        rw [if_neg (by simp [hto])] at hoo
        have := (Result.ok_injective hoo).symm
        rw [hto] at this
        subst this
        obtain ⟨T, hF, hiff⟩ := ov_full_of_some ks hRt hto
        have : arenaLen t = ov.nodes.val.length := by unfold arenaLen; rw [hto]
        exact ⟨ov, T, rfl, hF, hiff, by omega, by omega⟩
    obtain ⟨ov, T, rfl, hF, hiff, hcap, hovt⟩ := hov
    obtain ⟨fe, hfe, hic⟩ := bind_inv hic
    rcases fe with _ | ⟨nn, ee⟩
    · -- absent: `insert_new_at`
      obtain ⟨⟨⟨nd, e⟩, ov1⟩, hins, hic⟩ := bind_inv hic
      have hic' := Result.ok_injective hic
      simp only [Prod.mk.injEq] at hic'
      obtain ⟨rfl, rfl⟩ := hic'
      have hs' := (Result.ok_injective h).symm
      subst hs'
      obtain ⟨T', hF', hT', -, hl1⟩ :=
        blocked_new_full ks ov T hF hns hcok hcap hfe v nd e ov1 hins
      refine ⟨rep_set_ov ks hRt.1 ov1 T' hF' hmm hms (fun c => ?_), hKt,
        by show ov1.nodes.val.length + mainCount t ≤ _; omega⟩
      rw [hT', hiff, hRt.1.live_iff c]
      constructor
      · rintro ((h | h | h) | h)
        · exact Or.inl h
        · exact Or.inr (Or.inl h)
        · exact Or.inr (Or.inr (Or.inl h))
        · exact Or.inr (Or.inr (Or.inr h))
      · rintro (h | h | h | h)
        · exact Or.inl (Or.inl h)
        · exact Or.inl (Or.inr (Or.inl h))
        · exact Or.inl (Or.inr (Or.inr h))
        · exact Or.inr h
    · -- present: `take_value` then `write_back`
      obtain ⟨-, -, hic⟩ := bind_inv hic
      obtain ⟨⟨old, ov1⟩, htv, hic⟩ := bind_inv hic
      simp only at hic
      obtain ⟨ov2, hwb, hic⟩ := bind_inv hic
      obtain ⟨-, -, hic⟩ := bind_inv hic
      have hic' := Result.ok_injective hic
      simp only [Prod.mk.injEq] at hic'
      obtain ⟨rfl, rfl⟩ := hic'
      have hs' := (Result.ok_injective h).symm
      subst hs'
      obtain ⟨hF2, hT2, hTk, hl2⟩ := blocked_hit_full ks ov T hF nn ee hfe old ov1 htv v ov2 hwb
      have hlk : live k := (hRt.1.live_iff k).mpr (Or.inr (Or.inr ((hiff k).mp hTk)))
      refine ⟨rep_set_ov ks hRt.1 ov2 T hF2 hmm hms (fun c => ?_), hKt,
        by show ov2.nodes.val.length + mainCount t ≤ _; omega⟩
      rw [hT2, hiff, ← hRt.1.live_iff c]
      constructor
      · rintro (h | rfl)
        · exact h
        · exact hlk
      · exact Or.inl

end

/-! ## The two key stores -/

/-- The `Stored<u64>` key column: `keys[i] = some c` at every non-reserved main coordinate `c`. -/
def SKC (coords : Slice U64) (keys : hashtrinity.algebra.Stored U64) : Prop :=
  keys.length = coords.length ∧
    ∀ (i : Nat) (c : U64), coords.val[i]? = some c → ¬ IsSent c → keys.val[i]? = some (some c)

theorem skc_set (coords coords' : Slice U64) (keys keys' : hashtrinity.algebra.Stored U64)
    (i : Usize) (k : U64) (hK : SKC coords keys) (hco : coords'.val = coords.val.set i.val k)
    (h : (SKS).set keys i k = ok keys') : SKC coords' keys' := by
  change hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set CI
    core.cmp.EqU64 keys i k = ok keys' at h
  unfold hashtrinity.algebra.Stored.Insts.Apeiron_wordHashtrinityAlgebraKeyStore.set at h
  obtain ⟨mu, hmu, h⟩ := bind_inv h
  obtain ⟨⟨x, back⟩, hg, h⟩ := bind_inv h
  obtain ⟨-, hback⟩ := get_unchecked_mut_ok _ _ _ _ hg
  have := (Result.ok_injective h).symm
  subst this
  have hmu' : mu = some k := (Result.ok_injective hmu).symm
  have hkv : (back mu).val = keys.val.set i.val (some k) := by
    rw [hback, hmu', Slice.set_val_eq]
  obtain ⟨hl, hall⟩ := hK
  have hl' : keys.val.length = coords.val.length := hl
  refine ⟨?_, fun d c hd hcns => ?_⟩
  · show (back mu).val.length = coords'.val.length
    rw [hkv, hco, List.length_set, List.length_set, hl']
  · rw [hkv]
    rw [hco] at hd
    by_cases hdi : d = i.val
    · rw [hdi] at hd ⊢
      have hlt : i.val < coords.val.length := by
        by_contra hge
        rw [List.getElem?_eq_none (by simp; omega)] at hd
        exact absurd hd (by simp)
      rw [List.getElem?_set_self hlt] at hd
      rw [List.getElem?_set_self (by omega), ← Option.some.inj hd]
    · rw [List.getElem?_set_ne (Ne.symm hdi)] at hd ⊢
      exact hall d c hd hcns

/-- **`insert_preserves_of_resize` (`Stored<u64>`).**  Given `ResizeOK`, `word_hashtrinity_insert`
on a non-reserved key preserves `TableRep ∧ OvWF` and updates the canonical map by
`insertWith (fun new _ => new)`. -/
theorem insert_preserves_of_resize (hRes : ResizeOK SKS SKC)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
    (hR : TableRep s m live) (hW : OvWF s) (hsmall : arenaLen s + mainCount s < nilN)
    (k : U64) (hns : ¬ IsSent k) (v : I64) (s' : Tbl (hashtrinity.algebra.Stored U64))
    (h : word_hashtrinity_insert s k v = ok s') :
    TableRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' := by
  obtain ⟨⟨hC, hW'⟩, hK, -⟩ := insert_ns_rep SKS SKS_matches SKC skc_set hRes ⟨hR.1, hW⟩ hR.2
    hsmall hns v h
  exact ⟨⟨hC, hK⟩, hW'⟩

/-- **`coord_insert_preserves_of_resize` (`Regen<u64>`).**  The same for `word_coordmap_insert`. -/
theorem coord_insert_preserves_of_resize (hRes : ResizeOK RKS (fun _ _ => True))
    (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
    (hR : CoordRep s m live) (hW : OvWF s) (hsmall : arenaLen s + mainCount s < nilN)
    (k : U64) (hns : ¬ IsSent k) (v : I64) (s' : Tbl (hashtrinity.algebra.Regen U64))
    (h : word_coordmap_insert s k v = ok s') :
    CoordRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' :=
  (insert_ns_rep RKS RKS_matches (fun _ _ => True) (fun _ _ _ _ _ _ _ _ _ => trivial) hRes
    ⟨hR, hW⟩ trivial hsmall hns v h).1

/-- **`insert_refines_of_resize` (`Stored<u64>`, every key).** -/
theorem insert_refines_of_resize (hRes : ResizeOK SKS SKC)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
    (hR : TableRep s m live) (hW : OvWF s) (hsmall : arenaLen s + mainCount s < nilN)
    (k : U64) (v : I64) (s' : Tbl (hashtrinity.algebra.Stored U64))
    (h : word_hashtrinity_insert s k v = ok s') :
    TableRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' := by
  by_cases hk : IsSent k
  · obtain ⟨ss, hs, hc⟩ := insert_sent_rep SKS SKS_matches hR.1 k v hk s' h
    subst hs
    exact ⟨⟨hc, hR.2.1, hR.2.2⟩, hW⟩
  · exact insert_preserves_of_resize hRes s m live hR hW hsmall k hk v s' h

/-- **`coord_insert_refines_of_resize` (`Regen<u64>`, every key).** -/
theorem coord_insert_refines_of_resize (hRes : ResizeOK RKS (fun _ _ => True))
    (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
    (hR : CoordRep s m live) (hW : OvWF s) (hsmall : arenaLen s + mainCount s < nilN)
    (k : U64) (v : I64) (s' : Tbl (hashtrinity.algebra.Regen U64))
    (h : word_coordmap_insert s k v = ok s') :
    CoordRep s' (Apeiron.Table.insertWith (fun new _ => new) k v m) (fun c => live c ∨ c = k) ∧
      OvWF s' := by
  by_cases hk : IsSent k
  · obtain ⟨ss, hs, hc⟩ := insert_sent_rep RKS RKS_matches hR k v hk s' h
    subst hs
    exact ⟨hc, hW⟩
  · exact coord_insert_preserves_of_resize hRes s m live hR hW hsmall k hk v s' h

end Apeiron.MapExecuted
