-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe`: the executed map face's read refines `Apeiron.Table.get`.
import Map.ExecutedRep
open Aeneas Aeneas.Std Result

/-!
# The executed map face's read refines `Apeiron.Table.get`

`get_refines` (`Stored<u64>`, `word_hashtrinity_get`) and `coord_get_refines` (`Regen<u64>`,
`word_coordmap_get`): under the face's representation, the executed read returns `some v` exactly when
the key is live and `v = Apeiron.Table.get m key`, and `none` off the live set.  Both are corollaries of
`get_coord_spec`, proved once over any key store whose `matches` is `true` on `u64` keys.

The read's branches, as the extracted `hashtrinity.Table.get_coord` takes them:

* reserved coordinate (`EMPTY`/`TOMB`): `SideStore::find` (`side_find_spec`);
* home slot holds the key: the slot's value (`CoreRep.main_val`);
* home slot `EMPTY`: absent — the capped placement puts a main key at the first non-`EMPTY`-preceded
  step of its probe sequence, and an overflow key's whole capped prefix is non-`EMPTY`;
* otherwise the capped probe (`pcw_spec`): `Hit` reads the slot, `Vacant` is absent by the same
  placement argument, `Blocked` excludes the main array and reads the overflow tree (`ov_get_spec`).

The AVX2 branch of `prefix_masks` enters through the hypothesis `avx2_probe = true → Avx2Spec`
(`Map/ExecutedProbe.lean`); with `avx2_probe = false` it is vacuous.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

theorem not_side_of_ns {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) : ¬ InSide s k := by
  rintro ⟨i, hi⟩
  exact hns (hR.side_val i k hi).1

theorem ovInv_some {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {o : Ov} (ho : s.overflow = some o) :
    ∃ T, Emb o.nodes.val o.root T ∧ BST o.nodes.val T ∧ (idxs T).Nodup ∧
      (∀ f ∈ o.free.val, f.val ∉ idxs T) ∧
      ∀ i ∈ idxs T, ∃ nd, o.nodes.val[i]? = some nd ∧
        nd.entries.val = [(nd.key, some (m nd.key))] ∧ ¬ IsSent nd.key ∧
        ∀ j : U32, j.val < limN s.bits.val → Pass ks s nd.key j := by
  have := hR.ov
  unfold OvInv at this
  rw [ho] at this
  exact this

/-- An overflow coordinate passes every capped probe step. -/
theorem ov_pass {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hov : InOv s k) :
    ¬ IsSent k ∧ ∀ j : U32, j.val < limN s.bits.val → Pass ks s k j := by
  obtain ⟨o, ho, T, hE, i, hi, nd, hnd, hk⟩ := hov
  obtain ⟨T', hE', _, _, _, hall⟩ := ovInv_some ks hR ho
  have hTT := emb_unique hE hE'
  subst hTT
  obtain ⟨nd', hnd', _, hns, hpass⟩ := hall i hi
  rw [hnd] at hnd'
  have hnn : nd = nd' := Option.some.inj hnd'
  subst hnn
  rw [← hk]
  exact ⟨hns, hpass⟩

/-- A capped `Vacant` verdict means the key is not live. -/
theorem vacant_not_live {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) (i : Usize)
    (hc : CappedOK ks s k (.Vacant i)) : ¬ live k := by
  obtain ⟨⟨x, hx, hxe⟩, j, hj, hd, hpre⟩ := hc
  rw [hR.live_iff]
  rintro (hside | ⟨_, i', hi'⟩ | hov)
  · exact not_side_of_ns ks hR hns hside
  · obtain ⟨j1, hj1, d1, hd1, hd1v, hpre1⟩ := hR.main_placed i' k hi' hns
    rcases Nat.lt_trichotomy j1.val j.val with hlt | heq | hgt
    · obtain ⟨d, hdd, y, hy, hyk, _⟩ := hpre j1 hlt
      have := desc_det ks hdd hd1
      subst this
      rw [hd1v, hi'] at hy
      exact hyk (Option.some.inj hy).symm
    · have hjj : j1 = j := UScalar.eq_of_val_eq heq
      subst hjj
      have := desc_det ks hd hd1
      subst this
      rw [hd1v, hi'] at hx
      have hkx := Option.some.inj hx
      subst hkx
      exact hns (Or.inl hxe)
    · obtain ⟨d', hd', y, hy, hye⟩ := hpre1 j hgt
      have := desc_det ks hd' hd
      subst this
      rw [hx] at hy
      have := Option.some.inj hy
      subst this
      exact hye hxe
  · obtain ⟨_, hpass⟩ := ov_pass ks hR hov
    obtain ⟨d, hdd, y, hy, _, hye⟩ := hpass j hj
    have := desc_det ks hdd hd
    subst this
    rw [hx] at hy
    have := Option.some.inj hy
    subst this
    exact hye hxe

/-- A capped `Blocked` verdict excludes the main array. -/
theorem blocked_not_main {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64}
    (hc : CappedOK ks s k .Blocked) : ¬ InMain s k := by
  rintro ⟨hns, i', hi'⟩
  obtain ⟨j1, hj1, d1, hd1, hd1v, _⟩ := hR.main_placed i' k hi' hns
  obtain ⟨d, hdd, y, hy, hyk, _⟩ := hc j1 hj1
  have := desc_det ks hdd hd1
  subst this
  rw [hd1v, hi'] at hy
  exact hyk (Option.some.inj hy).symm

theorem res_some {live : U64 → Prop} {m : Apeiron.Table.HMap U64 I64} {k : U64} (hl : live k) :
    ∀ v, some (m k) = some v ↔ live k ∧ v = Apeiron.Table.get m k := by
  intro v
  simp only [Option.some.injEq, Apeiron.Table.get]
  exact ⟨fun h => ⟨hl, h.symm⟩, fun h => h.2.symm⟩

theorem res_none {live : U64 → Prop} {m : Apeiron.Table.HMap U64 I64} {k : U64} (hl : ¬ live k) :
    ∀ v, (none : Option I64) = some v ↔ live k ∧ v = Apeiron.Table.get m k := by
  intro v
  simp only [reduceCtorEq, false_iff, not_and]
  exact fun h => absurd h hl

theorem coord_of_ok (k : U64) :
    hashtrinity.Table.coord_of I64 core.cmp.EqU64 CI ks k = ok k := rfl

theorem read_main_slot {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) (i : Usize)
    (hi : s.coords.val[i.val]? = some k) (mu : Option I64)
    (hmu : core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice
      (core.mem.maybe_uninit.MaybeUninit I64)) s.vals i = ok mu) (t : I64)
    (ht : core.mem.maybe_uninit.MaybeUninit.assume_init_ref mu = ok t) :
    live k ∧ t = m k := by
  obtain ⟨hl, hv⟩ := hR.main_val i.val k hi hns
  have hmu' := get_unchecked_ok _ _ _ hmu
  rw [hv] at hmu'
  have := Option.some.inj hmu'
  subst this
  have := assume_init_ref_ok _ _ ht
  exact ⟨hl, (Option.some.inj this).symm⟩

/-- The read of a `Blocked` verdict with no overflow tree: absent. -/
theorem blocked_none_not_live {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) (ho : s.overflow = none)
    (hc : CappedOK ks s k .Blocked) : ¬ live k := by
  have hnm := blocked_not_main ks hR hc
  rw [hR.live_iff]
  rintro (hside | hmain | ⟨o, ho', _⟩)
  · exact not_side_of_ns ks hR hns hside
  · exact hnm hmain
  · rw [ho] at ho'; exact absurd ho' (by simp)

/-- The read of a `Blocked` verdict with an overflow tree: the tree's `get`. -/
theorem blocked_some_spec {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) {o : Ov} (ho : s.overflow = some o)
    (hc : CappedOK ks s k .Blocked) (r : Option I64)
    (h : hashtrinity.overflow.Overflow.get CI core.cmp.EqU64 o k k = ok r) :
    ∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k := by
  have hnm := blocked_not_main ks hR hc
  obtain ⟨T, hE, hB, _, _, hall⟩ := ovInv_some ks hR ho
  have hS : SingleEntries o.nodes.val T := by
    intro i hi
    obtain ⟨nd, hnd, hent, _⟩ := hall i hi
    exact ⟨nd, hnd, _, hent⟩
  have hspec := ov_get_spec o T hE hB hS k r h
  intro v
  rw [hspec v]
  constructor
  · rintro ⟨i, hi, nd, hnd, hk, hent⟩
    obtain ⟨nd', hnd', hent', _⟩ := hall i hi
    rw [hnd] at hnd'
    have hnn : nd = nd' := Option.some.inj hnd'
    subst hnn
    rw [hent', hk] at hent
    simp only [List.cons.injEq, Prod.mk.injEq, Option.some.injEq, and_true, true_and] at hent
    refine ⟨(hR.live_iff k).mpr (Or.inr (Or.inr ⟨o, ho, T, hE, i, hi, nd, hnd, hk⟩)), ?_⟩
    simp only [Apeiron.Table.get]; exact hent.symm
  · rintro ⟨hl, hv⟩
    rcases (hR.live_iff k).mp hl with hside | hmain | ⟨o', ho', T', hE', i, hi, nd, hnd, hk⟩
    · exact absurd hside (not_side_of_ns ks hR hns)
    · exact absurd hmain hnm
    · have hoo : o = o' := Option.some.inj (ho.symm.trans ho')
      subst hoo
      have hTT := emb_unique hE hE'
      subst hTT
      obtain ⟨nd', hnd', hent', _⟩ := hall i hi
      rw [hnd] at hnd'
      have hnn : nd = nd' := Option.some.inj hnd'
      subst hnn
      refine ⟨i, hi, nd, hnd, hk, ?_⟩
      rw [hent', hk, hv]
      rfl

/-- The capped-probe tail of the read, one verdict at a time. -/
theorem read_tail_tac_hit {s : Tbl KS} {m : Apeiron.Table.HMap U64 I64} {live : U64 → Prop}
    (hR : CoreRep ks s m live) {k : U64} (hns : ¬ IsSent k) (i : Usize)
    (hc : CappedOK ks s k (.Hit i)) (r : Option I64)
    (h : (do
        let mu ← core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice
          (core.mem.maybe_uninit.MaybeUninit I64)) s.vals i
        let t ← core.mem.maybe_uninit.MaybeUninit.assume_init_ref mu
        ok (some t)) = ok r) :
    ∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k := by
  obtain ⟨mu, hmu, h⟩ := bind_inv h
  obtain ⟨t, ht, h⟩ := bind_inv h
  have := (Result.ok_injective h).symm
  subst this
  obtain ⟨hl, htv⟩ := read_main_slot ks hR hns i hc mu hmu t ht
  rw [htv]
  exact res_some hl

/-- **The executed read, over any key store whose `matches` is `true` on `u64`.** -/
theorem get_coord_spec (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
    (hR : CoreRep ks s m live) (k : U64) (r : Option I64)
    (h : hashtrinity.Table.get_coord core.cmp.EqU64 CI ks s k = ok r) :
    ∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k := by
  unfold hashtrinity.Table.get_coord at h
  rw [coord_of_ok, bind_tc_ok, is_sentinel_ok, bind_tc_ok] at h
  by_cases hs : IsSent k
  · rw [if_pos (by simp [hs])] at h
    obtain ⟨o, ho, h⟩ := bind_inv h
    obtain ⟨hsome, hnone⟩ := side_find_spec ks hm s.side k k o ho
    cases o with
    | none =>
      simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok,
        core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual] at h
      have := (Result.ok_injective h).symm
      subst this
      apply res_none
      rw [hR.live_iff]
      rintro (⟨i, hi⟩ | ⟨hns, _⟩ | hov)
      · exact hnone rfl i hi
      · exact hns hs
      · exact (ov_pass ks hR hov).1 hs
    | some i =>
      simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok] at h
      obtain ⟨mu, hmu, h⟩ := bind_inv h
      obtain ⟨t, ht, h⟩ := bind_inv h
      have := (Result.ok_injective h).symm
      subst this
      have hci := hsome i rfl
      obtain ⟨_, hl, hv, _⟩ := hR.side_val i.val k hci
      have hmu' := get_unchecked_ok _ _ _ hmu
      have hmu'' : s.side.vals.val[i.val]? = some mu := by
        simpa [alloc.vec.Vec.deref] using hmu'
      rw [hv] at hmu''
      have := Option.some.inj hmu''
      subst this
      have := assume_init_ref_ok _ _ ht
      have htv : t = m k := (Option.some.inj this).symm
      rw [htv]
      exact res_some hl
  · rw [if_neg (by simp [hs])] at h
    obtain ⟨home, hhome, h⟩ := bind_inv h
    obtain ⟨sc, hsc, h⟩ := bind_inv h
    have hsc' := get_unchecked_ok _ _ _ hsc
    have hhlt : home.val < s.coords.length := (List.getElem?_eq_some_iff.mp hsc').1
    have hvac : sc = hashtrinity.EMPTY → CappedOK ks s k (hashtrinity.Capped.Vacant home) := by
      intro he
      obtain ⟨i, hd, hv⟩ := desc_lin ks s k 0#u32 home (by simp) hhome hR.len_pow
        (by simpa using hhlt)
      have : i = home := UScalar.eq_of_val_eq (by rw [hv]; simp)
      subst this
      refine ⟨⟨sc, hsc', (EMPTY_eq_iff sc).mp he⟩, 0#u32, ?_, hd, ?_⟩
      · have := limN_pos s.bits.val; simp; omega
      · intro j' hj'; exact absurd hj' (by simp)
    by_cases hc : sc = k
    · rw [if_pos hc] at h
      obtain ⟨b, hb, h⟩ := bind_inv h
      rw [hm] at hb
      have hb' : b = true := (Result.ok_injective hb).symm
      subst hb'
      rw [if_pos rfl] at h
      obtain ⟨mu, hmu, h⟩ := bind_inv h
      obtain ⟨t, ht, h⟩ := bind_inv h
      have := (Result.ok_injective h).symm
      subst this
      obtain ⟨hl, htv⟩ := read_main_slot ks hR hs home (by rw [hsc', hc]) mu hmu t ht
      rw [htv]
      exact res_some hl
    · rw [if_neg hc] at h
      by_cases he : sc = hashtrinity.EMPTY
      · rw [if_pos he] at h
        have := (Result.ok_injective h).symm
        subst this
        exact res_none (vacant_not_live ks hR hs home (hvac he))
      · rw [if_neg he] at h
        obtain ⟨c, hcp, h⟩ := bind_inv h
        have hcok := pcw_spec ks hm hA s k home _ hhome hR.len_pow c hcp
        cases c with
        | Hit i => exact read_tail_tac_hit ks hR hs i hcok r h
        | Vacant i =>
          have := (Result.ok_injective h).symm
          subst this
          exact res_none (vacant_not_live ks hR hs i hcok)
        | Blocked =>
          simp only [core.option.Option.as_ref, bind_tc_ok] at h
          cases ho : s.overflow with
          | none =>
            rw [ho] at h
            have := (Result.ok_injective h).symm
            subst this
            exact res_none (blocked_none_not_live ks hR hs ho hcok)
          | some o =>
            rw [ho] at h
            exact blocked_some_spec ks hR hs ho hcok r h

end

/-- **`get_refines` (`Stored<u64>`, `word_hashtrinity_get`).**  Under `TableRep s m live`, the executed
read returns `some v` exactly when `live k` and `v = Apeiron.Table.get m k`; off the live set it
returns `none`. -/
theorem get_refines (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl (hashtrinity.algebra.Stored U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : TableRep s m live) (k : U64) (r : Option I64)
    (h : word_hashtrinity_get s k = ok r) :
    (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) := by
  unfold word_hashtrinity_get hashtrinity.Table.get at h
  obtain ⟨o, ho, h⟩ := bind_inv h
  have hspec := get_coord_spec SKS SKS_matches hA s m live hR.1 k o ho
  have hro : r = o := by
    cases o with
    | none => exact (Result.ok_injective h).symm
    | some _ => exact (Result.ok_injective h).symm
  subst hro
  refine ⟨hspec, fun hl => ?_⟩
  cases hr : r with
  | none => rfl
  | some v => exact absurd ((hspec v).mp hr).1 hl

/-- **`coord_get_refines` (`Regen<u64>`, `word_coordmap_get`).**  The same refinement for the
`CoordMap` face. -/
theorem coord_get_refines (hA : std_detect.detect.arch.x86.avx2_probe = true → Avx2Spec)
    (s : Tbl (hashtrinity.algebra.Regen U64)) (m : Apeiron.Table.HMap U64 I64)
    (live : U64 → Prop) (hR : CoordRep s m live) (k : U64) (r : Option I64)
    (h : word_coordmap_get s k = ok r) :
    (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) := by
  unfold word_coordmap_get hashtrinity.Table.get at h
  obtain ⟨o, ho, h⟩ := bind_inv h
  have hspec := get_coord_spec RKS RKS_matches hA s m live hR k o ho
  have hro : r = o := by
    cases o with
    | none => exact (Result.ok_injective h).symm
    | some _ => exact (Result.ok_injective h).symm
  subst hro
  refine ⟨hspec, fun hl => ?_⟩
  cases hr : r with
  | none => rfl
  | some v => exact absurd ((hspec v).mp hr).1 hl

end Apeiron.MapExecuted
