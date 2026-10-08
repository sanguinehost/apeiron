-- Row: `apeiron-ht-read-BgVD7Mjq4dXA4grupEYPDe`: the `CoordMap` face's read totality.
import Map.ExecutedRead
import Map.ExecutedReadUncond
import Map.ExecutedReadTotal
import Map.Rep
open Aeneas Aeneas.Std Result

/-!
# Totality of the executed `CoordMap` read

`get_coord_total` (`Map/ExecutedReadTotal.lean`) proves the read's failure-freedom for the
`Stored<u64>` face, at the store `SKS`; `coord_get_refines` (`Map/ExecutedRead.lean`) leaves the
partial-correctness premise `h : word_coordmap_get s k = ok r` for the `Regen<u64>` face.  This module
supplies the `Regen<u64>` twin of the W3c totality result: the same `get_coord` body with the store
`RKS`, so the `ok` premise of `coord_get_refines` is discharged.

The key-store-independent leaves (`home_total`, `descend_total`, `probe_capped_with_total`,
`side_find_loop_total` and the overflow-tree loops) are stated over any `KS` and are reused.  The
leaves specialised to `Stored` in `ExecutedReadTotal.lean` (`side_find_total`, `read_val_total`,
`get_coord_total`) are restated over any key store here; `ExecutedReadTotal.lean` is not edited.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Apeiron.MapExecuted
open apeiron_word

section
variable {KS : Type} (ks : hashtrinity.algebra.KeyStore KS U64)

/-- `SideStore::find` is total, over any key store. -/
theorem side_find_total_KS (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (ss : hashtrinity.side.SideStore U64 I64 KS) (c k : U64) :
    ∃ o, hashtrinity.side.SideStore.find CI ks ss c k = ok o := by
  unfold hashtrinity.side.SideStore.find
  exact side_find_loop_total ks hm ss c k 0#usize

/-- The main-slot value read is total whenever the slot holds an initialised value, over any key
store. -/
theorem read_val_total_KS (s : Tbl KS) (i : Usize) (x : I64)
    (hv : s.vals.val[i.val]? = some (some x)) :
    ∃ r, (do
      let mu ← core.slice.Slice.get_unchecked (core.slice.index.SliceIndexUsizeSlice
        (core.mem.maybe_uninit.MaybeUninit I64)) s.vals i
      let t ← core.mem.maybe_uninit.MaybeUninit.assume_init_ref mu
      ok (some t)) = ok r := by
  obtain ⟨mu, hmu⟩ := get_unchecked_ex s.vals i (by
    have h1 : i.val < s.vals.val.length := (List.getElem?_eq_some_iff.mp hv).1
    simpa [Slice.length] using h1)
  rw [hmu]; simp only [bind_tc_ok]
  have hmu' := get_unchecked_ok s.vals i mu hmu
  have hmu_eq : mu = some x := (Option.some.inj (by rw [hv] at hmu'; exact hmu')).symm
  rw [hmu_eq]
  simp only [core.mem.maybe_uninit.MaybeUninit.assume_init_ref, bind_tc_ok]
  exact ⟨some x, rfl⟩

/-- The extracted coordinate read is total on a table representing `m`, over any key store whose
`matches` is `true` on `u64` keys: every branch returns `ok`. -/
theorem get_coord_total_KS (hm : ∀ keys i c k, ks.«matches» keys i c k = ok true)
    (s : Tbl KS) (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop)
    (hR : CoreRep ks s m live) (k : U64) :
    ∃ r, hashtrinity.Table.get_coord core.cmp.EqU64 CI ks s k = ok r := by
  have hb := hR.bits_pos
  have hlen := hR.len_pow
  have hb64 := bits_le_64 ks s hlen
  unfold hashtrinity.Table.get_coord
  rw [coord_of_ok, bind_tc_ok, is_sentinel_ok, bind_tc_ok]
  by_cases hs : IsSent k
  · rw [if_pos (by simp [hs])]
    obtain ⟨o, ho⟩ := side_find_total_KS ks hm s.side k k
    rw [ho]; simp only [bind_tc_ok]
    cases o with
    | none =>
      simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok,
        core.option.Option.Insts.CoreOpsTry_traitFromResidualOptionInfallible.from_residual]
      exact ⟨none, rfl⟩
    | some i =>
      obtain ⟨hsome, _⟩ := side_find_spec ks hm s.side k k (some i) ho
      simp only [core.option.Option.Insts.CoreOpsTry_traitTry.branch, bind_tc_ok]
      have hci := hsome i rfl
      obtain ⟨_, _, hv⟩ := hR.side_val i.val k hci
      obtain ⟨mu, hmu⟩ := get_unchecked_ex (alloc.vec.Vec.deref s.side.vals) i
        (by
          have h1 : i.val < s.side.vals.val.length := (List.getElem?_eq_some_iff.mp hv).1
          simpa [alloc.vec.Vec.deref, Slice.length] using h1)
      rw [hmu]; simp only [bind_tc_ok]
      have hmu' := get_unchecked_ok _ _ _ hmu
      have hmu'' : s.side.vals.val[i.val]? = some mu := by
        simpa [alloc.vec.Vec.deref] using hmu'
      have hmu_eq : mu = some (m k) := (Option.some.inj (by rw [hv] at hmu''; exact hmu'')).symm
      rw [hmu_eq]
      simp only [core.mem.maybe_uninit.MaybeUninit.assume_init_ref, bind_tc_ok]
      exact ⟨some (m k), rfl⟩
  · rw [if_neg (by simp [hs])]
    obtain ⟨home, hhome_ok, hhome⟩ := home_total ks s k hb hb64 hlen
    rw [hhome_ok]; simp only [bind_tc_ok]
    obtain ⟨sc, hsc⟩ := get_unchecked_ex s.coords home hhome
    rw [hsc]; simp only [bind_tc_ok]
    by_cases hc : sc = k
    · rw [if_pos hc]
      rw [hm]; simp only [bind_tc_ok]
      rw [if_pos trivial]
      have hscv : s.coords.val[home.val]? = some k := by
        have := get_unchecked_ok _ _ _ hsc
        rw [hc] at this; exact this
      obtain ⟨_, hv⟩ := hR.main_val home.val k hscv hs
      exact read_val_total_KS s home (m k) hv
    · rw [if_neg hc]
      simp only [lift, bind_tc_ok]
      by_cases hci : home = UScalar.cast .Usize k
      · rw [if_pos hci]
        by_cases hse : sc = hashtrinity.EMPTY
        · rw [if_pos hse]; exact ⟨none, rfl⟩
        · rw [if_neg hse]
          obtain ⟨c, hcp⟩ := probe_capped_with_total ks hm s k home { coord := k, k := k }
            hb hb64 hlen hhome
          have hcok := pcw_spec ks hm (fun _ => avx2_spec_holds) s k home { coord := k, k := k }
            hb hhome_ok hlen c hcp
          simp only [ConfI] at hcp
          rw [hcp]; simp only [bind_tc_ok]
          cases c with
          | Hit i1 =>
            have hcv : s.coords.val[i1.val]? = some k := hcok
            obtain ⟨_, hv⟩ := hR.main_val i1.val k hcv hs
            exact read_val_total_KS s i1 (m k) hv
          | Vacant _ => exact ⟨none, rfl⟩
          | Blocked =>
            simp only [core.option.Option.as_ref, bind_tc_ok]
            cases hov : s.overflow with
            | none => exact ⟨none, rfl⟩
            | some o1 =>
              obtain ⟨T, hE, hB, _, _, hall⟩ := ovInv_some ks hR hov
              have hS : SingleEntries o1.nodes.val T := by
                intro i hi
                obtain ⟨nd, hnd, hent, _⟩ := hall i hi
                exact ⟨nd, hnd, m nd.key, hent⟩
              obtain ⟨r, hr⟩ := Overflow.get_total o1 T hE hB hS k
              exact ⟨r, hr⟩
      · rw [if_neg hci]
        obtain ⟨c, hcp⟩ := probe_capped_with_total ks hm s k home { coord := k, k := k }
          hb hb64 hlen hhome
        have hcok := pcw_spec ks hm (fun _ => avx2_spec_holds) s k home { coord := k, k := k }
          hb hhome_ok hlen c hcp
        simp only [ConfI] at hcp
        rw [hcp]; simp only [bind_tc_ok]
        cases c with
        | Hit i1 =>
          have hcv : s.coords.val[i1.val]? = some k := hcok
          obtain ⟨_, hv⟩ := hR.main_val i1.val k hcv hs
          exact read_val_total_KS s i1 (m k) hv
        | Vacant _ => exact ⟨none, rfl⟩
        | Blocked =>
          simp only [core.option.Option.as_ref, bind_tc_ok]
          cases hov : s.overflow with
          | none => exact ⟨none, rfl⟩
          | some o1 =>
            obtain ⟨T, hE, hB, _, _, hall⟩ := ovInv_some ks hR hov
            have hS : SingleEntries o1.nodes.val T := by
              intro i hi
              obtain ⟨nd, hnd, hent, _⟩ := hall i hi
              exact ⟨nd, hnd, m nd.key, hent⟩
            obtain ⟨r, hr⟩ := Overflow.get_total o1 T hE hB hS k
            exact ⟨r, hr⟩

end

/-! ## The `Regen<u64>` face -/

/-- **Totality of the executed read (`Regen<u64>`, `word_coordmap_get`).**  On a `CoordRep` state,
the read returns `ok` for every key: the same `get_coord` body as the `Stored` face, at `RKS`, with
`RKS_matches` supplying the key-store contract.  No premise beyond the representation. -/
theorem coord_get_ok_of_coordRep (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordRep s m live) (k : U64) :
    ∃ r, word_coordmap_get s k = ok r := by
  obtain ⟨o, ho⟩ := get_coord_total_KS RKS RKS_matches s m live hR k
  unfold word_coordmap_get hashtrinity.Table.get
  rw [ho]; simp only [bind_tc_ok]
  cases o with
  | none => exact ⟨none, rfl⟩
  | some v => exact ⟨some v, rfl⟩

/-- **`coord_get_refines` with its `ok` premise discharged.**  For every key the `CoordMap` read
returns `ok r`, with `r = some v` exactly when `live k` and `v = Apeiron.Table.get m k`, and `none`
off the live set. -/
theorem coord_get_refines_total (s : Tbl (hashtrinity.algebra.Regen U64))
    (m : Apeiron.Table.HMap U64 I64) (live : U64 → Prop) (hR : CoordRep s m live) (k : U64) :
    ∃ r, word_coordmap_get s k = ok r ∧
      (∀ v, r = some v ↔ live k ∧ v = Apeiron.Table.get m k) ∧ (¬ live k → r = none) := by
  obtain ⟨r, hr⟩ := coord_get_ok_of_coordRep s m live hR k
  exact ⟨r, hr, coord_get_refines (fun _ => avx2_spec_holds) s m live hR k r hr⟩

end Apeiron.MapExecuted

#audit Apeiron.MapExecuted.coord_get_ok_of_coordRep
#audit Apeiron.MapExecuted.coord_get_refines_total
