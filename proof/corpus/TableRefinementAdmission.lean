import TableRefinementBuild

/-! # Stream-only admission for the executing fixed-shard word table -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- Physical keys in the one state carried after a generated stream prefix are covered by that
    prefix's distinct routed keys.  This predicate does not quantify over arbitrary representatives
    with the same extensional reads. -/
def ReachablePrefixCoverage (kvs : Slice (I64 × I64)) (i : Nat) (state : WordTable) : Prop :=
  TableInv2 state ∧
    ∀ (shard : Usize) (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat),
      Array.index_usize state.shards shard = ok (some slots) →
      k ≤ 16 → NoEmptyIn slots.val 0 k →
      ∀ (j : Nat) (key value : I64), j < k →
        slots.val[j]? = some (some (key, value)) →
        key ∈ routedStreamKeys kvs i shard

private theorem getElem?_eq_some_of_index_usize_eq_ok {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

private theorem index_usize_eq_ok_of_getElem?_eq_some {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : a.val[i.val]? = some x) :
    Array.index_usize a i = ok x := by
  unfold Array.index_usize
  rw [Array.getElem?_Usize_eq, h]

/-- The real empty constructor has no physical key outside the empty stream prefix. -/
theorem emptyWordTable_reachablePrefixCoverage (default : I64)
    (kvs : Slice (I64 × I64)) :
    ReachablePrefixCoverage kvs 0 (emptyWordTable default) := by
  set_option maxRecDepth 4096 in
  refine ⟨emptyWordTable_tableInv2 default, ?_⟩
  intro shard slots k houter
  have houter' := getElem?_eq_some_of_index_usize_eq_ok
    (emptyWordTable default).shards shard (some slots) houter
  change (List.replicate 256
    (none : Option (Array (Option (I64 × I64)) 16#usize)))[shard.val]? =
      some (some slots) at houter'
  rw [List.getElem?_eq_getElem (by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
    simp at houter'), List.getElem_replicate] at houter'
  simp at houter'

private theorem routedStreamKeys_mono (kvs : Slice (I64 × I64)) (i : Nat)
    (hi : i < (sliceEntries kvs).length) (shard : Usize) :
    routedStreamKeys kvs i shard ⊆ routedStreamKeys kvs (i + 1) shard := by
  classical
  intro key hkey
  simp only [routedStreamKeys, Finset.mem_filter, List.mem_toFinset,
    List.mem_map] at hkey ⊢
  refine ⟨?_, hkey.2⟩
  obtain ⟨kv, hkv, rfl⟩ := hkey.1
  refine ⟨kv, ?_, rfl⟩
  rw [List.take_succ_eq_append_getElem hi]
  exact List.mem_append_left _ hkv

private theorem routedStreamKeys_current (kvs : Slice (I64 × I64)) (i : Usize)
    (key value : I64) (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value)) (shard : Usize)
    (hroute : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok shard) :
    key ∈ routedStreamKeys kvs (i.val + 1) shard := by
  classical
  have hget : (sliceEntries kvs)[i.val]? = some (key, value) := by
    unfold Slice.index_usize at hindex
    rw [Slice.getElem?_Usize_eq] at hindex
    unfold sliceEntries
    split at hindex <;> simp_all
  simp only [routedStreamKeys, Finset.mem_filter, List.mem_toFinset, List.mem_map]
  refine ⟨⟨(key, value), ?_, rfl⟩, hroute⟩
  rw [List.mem_take_iff_getElem]
  exact ⟨i.val, by omega, (List.getElem?_eq_some_iff.mp hget).2⟩

/-- Stream coverage and `TableInv2` turn the distinct routed-key bound into strict load capacity for
    the currently reachable state. -/
theorem reachablePrefixCoverage_capacity (kvs : Slice (I64 × I64)) (i : Nat)
    (state : WordTable) (hcoverage : ReachablePrefixCoverage kvs i state)
    (hdistinct : DistinctKeysPerShard kvs) (hi : i < (sliceEntries kvs).length)
    (shard : Usize) (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hk : k ≤ 16) (hoccupied : NoEmptyIn slots.val 0 k) : k < 16 := by
  have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
    (some slots) houter
  obtain ⟨keys, hkeysInjective, hkeysSlots⟩ :=
    occupiedPrefix_distinct_key_count slots k hk hoccupied
      (hcoverage.1.2 shard.val slots houterGet)
  let intoStream : Fin k → { key // key ∈ routedStreamKeys kvs i shard } := fun j =>
    ⟨keys j, by
      obtain ⟨value, hslot⟩ := hkeysSlots j
      exact hcoverage.2 shard slots k houter hk hoccupied j.val (keys j) value j.isLt hslot⟩
  have hIntoInjective : Function.Injective intoStream := by
    intro a b hab
    apply hkeysInjective
    exact Subtype.ext_iff.mp hab
  have hcard : k ≤ (routedStreamKeys kvs i shard).card := by
    simpa [intoStream] using Fintype.card_le_of_injective intoStream hIntoInjective
  exact Nat.lt_of_le_of_lt hcard (hdistinct i shard hi)

private theorem present_reachablePrefixCoverage (kvs : Slice (I64 × I64)) (i : Nat)
    (state : WordTable) (key value : I64) (shard slot : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (old : I64)
    (hcoverage : ReachablePrefixCoverage kvs i state)
    (hi : i < (sliceEntries kvs).length)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 state shard key = ok (some slot))
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hslot : Array.index_usize slots slot = ok (some (key, old))) :
    ReachablePrefixCoverage kvs (i + 1)
      (replacePresent state shard slots slot key
        (core.num.I64.wrapping_add value old)) := by
  have hwp := tableInv2_insert_with state key value hcoverage.1
  rw [insert_with_present_exact state key value shard slot slots key old
    hshard hfind houter hslot] at hwp
  refine ⟨by simpa using hwp, ?_⟩
  intro s current k hcurrent hk hoccupied j stored storedValue hj hentry
  have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
    (some slots) houter
  have hshardBound : shard.val < state.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houterGet
    simp at houterGet
  by_cases hss : s.val = shard.val
  · have hs : s = shard := UScalar.eq_of_val_eq hss
    subst s
    have hcurrentGet := getElem?_eq_some_of_index_usize_eq_ok
      (replacePresent state shard slots slot key
        (core.num.I64.wrapping_add value old)).shards shard (some current) hcurrent
    have hcurrentEq : current = slots.set slot
        (some (key, core.num.I64.wrapping_add value old)) := by
      simp only [replacePresent] at hcurrentGet
      rw [Array.set_val_eq] at hcurrentGet
      rw [List.getElem?_set_eq_of_lt _ hshardBound] at hcurrentGet
      exact Option.some.inj (Option.some.inj hcurrentGet).symm
    subst current
    have hslotGet := getElem?_eq_some_of_index_usize_eq_ok slots slot
      (some (key, old)) hslot
    have hoccupiedOld : NoEmptyIn slots.val 0 k := by
      intro q hq hqk hempty
      by_cases hqslot : q = slot.val
      · subst q
        rw [EmptyAt, hslotGet] at hempty
        simp at hempty
      · apply hoccupied q hq hqk
        simpa [EmptyAt, Array.set_val_eq, hqslot] using hempty
    have hold : stored ∈ routedStreamKeys kvs i shard := by
      by_cases hjslot : j = slot.val
      · subst j
        have hnew : (slots.val.set slot.val
            (some (key, core.num.I64.wrapping_add value old)))[slot.val]? =
            some (some (key, core.num.I64.wrapping_add value old)) := by
          rw [List.getElem?_eq_getElem (by simpa [slots.property] using (show slot.val < 16 by
            have : slot.val < slots.val.length := by
              by_contra h
              rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hslotGet
              simp at hslotGet
            simpa [slots.property] using this)), List.getElem_set_self]
        rw [Array.set_val_eq, hnew] at hentry
        have hstored : stored = key :=
          (Prod.mk.inj (Option.some.inj (Option.some.inj hentry.symm))).1
        subst stored
        exact hcoverage.2 shard slots k houter hk hoccupiedOld slot.val key old hj hslotGet
      · have hentryOld : slots.val[j]? = some (some (stored, storedValue)) := by
          simpa [Array.set_val_eq, hjslot] using hentry
        exact hcoverage.2 shard slots k houter hk hoccupiedOld j stored storedValue hj
          hentryOld
    exact routedStreamKeys_mono kvs i hi shard hold
  · have hne : shard.val ≠ s.val := Ne.symm hss
    have houterOld : Array.index_usize state.shards s = ok (some current) := by
      simpa [replacePresent, Array.index_usize, Array.set_val_eq, hne] using hcurrent
    exact routedStreamKeys_mono kvs i hi s
      (hcoverage.2 s current k houterOld hk hoccupied j stored storedValue hj hentry)

private theorem vacant_reachablePrefixCoverage (kvs : Slice (I64 × I64)) (i : Usize)
    (state : WordTable) (key value : I64) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize)
    (hcoverage : ReachablePrefixCoverage kvs i.val state)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 state shard key = ok none)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hfirst : first < 16#usize) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    ReachablePrefixCoverage kvs (i.val + 1)
      (insertVacant state shard slots first key
        (core.num.I64.wrapping_add value state.default)) := by
  have hwp := tableInv2_insert_with state key value hcoverage.1
  rw [insert_with_vacant_exact state key value shard first slots hshard hfind houter
    hfirst hempty hnomatch hslot] at hwp
  refine ⟨by simpa using hwp, ?_⟩
  have hkeyRoute : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (emptyWordTable 0#i64) key = ok shard := by
    rw [← word_shard_index_state_independent state (emptyWordTable 0#i64)]
    exact hshard
  have hkeyMember := routedStreamKeys_current kvs i key value hi hindex shard hkeyRoute
  intro s current k hcurrent hk hoccupied j stored storedValue hj hentry
  have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
    (some slots) houter
  have hshardBound : shard.val < state.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houterGet
    simp at houterGet
  by_cases hss : s.val = shard.val
  · have hs : s = shard := UScalar.eq_of_val_eq hss
    subst s
    have hcurrentGet := getElem?_eq_some_of_index_usize_eq_ok
      (insertVacant state shard slots first key
        (core.num.I64.wrapping_add value state.default)).shards shard (some current) hcurrent
    have hcurrentEq : current = slots.set first
        (some (key, core.num.I64.wrapping_add value state.default)) := by
      simp only [insertVacant] at hcurrentGet
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hcurrentGet
      exact Option.some.inj (Option.some.inj hcurrentGet).symm
    subst current
    have hslotGet := getElem?_eq_some_of_index_usize_eq_ok slots first none hslot
    by_cases hjfirst : j = first.val
    · subst j
      have hnew : (slots.val.set first.val
          (some (key, core.num.I64.wrapping_add value state.default)))[first.val]? =
          some (some (key, core.num.I64.wrapping_add value state.default)) := by
        rw [List.getElem?_eq_getElem (by simpa [slots.property] using hfirst),
          List.getElem_set_self]
      rw [Array.set_val_eq, hnew] at hentry
      have hstored : stored = key :=
        (Prod.mk.inj (Option.some.inj (Option.some.inj hentry.symm))).1
      simpa [hstored] using hkeyMember
    · have hjBefore : j < first.val := by
        by_contra hnot
        have hjAfter : first.val < j := by omega
        obtain ⟨n, hn, hpresent, hvacant⟩ :=
          hcoverage.1.1 shard.val slots houterGet
        have hnEq : n = first.val := by
          have hnLe : n ≤ first.val := by
            by_contra h
            exact hpresent first.val (Nat.zero_le _) (by omega)
              (by simpa [EmptyAt] using hslotGet)
          have hfirstLe : first.val ≤ n := by
            by_contra h
            exact hempty n (Nat.zero_le _) (by omega)
              (by simpa [EmptyAt] using hvacant n le_rfl (by omega))
          exact Nat.le_antisymm hnLe hfirstLe
        subst n
        have holdVacant := hvacant j (Nat.le_of_lt hjAfter) (Nat.lt_of_lt_of_le hj hk)
        have hentryOld : slots.val[j]? = some (some (stored, storedValue)) := by
          simpa [Array.set_val_eq, hjfirst] using hentry
        rw [holdVacant] at hentryOld
        simp at hentryOld
      have hentryOld : slots.val[j]? = some (some (stored, storedValue)) := by
        simpa [Array.set_val_eq, hjfirst] using hentry
      exact routedStreamKeys_mono kvs i.val hi shard
        (hcoverage.2 shard slots first.val houter (Nat.le_of_lt hfirst) hempty
          j stored storedValue hjBefore hentryOld)
  · have hne : shard.val ≠ s.val := Ne.symm hss
    have houterOld : Array.index_usize state.shards s = ok (some current) := by
      simpa [insertVacant, Array.index_usize, Array.set_val_eq, hne] using hcurrent
    exact routedStreamKeys_mono kvs i.val hi s
      (hcoverage.2 s current k houterOld hk hoccupied j stored storedValue hj hentry)

private theorem emptyShard_reachablePrefixCoverage (kvs : Slice (I64 × I64)) (i : Usize)
    (state : WordTable) (key value : I64) (shard : Usize)
    (hcoverage : ReachablePrefixCoverage kvs i.val state)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 state shard key = ok none)
    (houter : Array.index_usize state.shards shard = ok none)
    (hshardBound : shard < 256#usize) :
    ReachablePrefixCoverage kvs (i.val + 1)
      (insertVacant state shard (Array.repeat 16#usize none) 0#usize key
        (core.num.I64.wrapping_add value state.default)) := by
  have hwp := tableInv2_insert_with state key value hcoverage.1
  rw [insert_with_empty_shard_exact state key value shard hshard hfind houter
    (by decide) hshardBound] at hwp
  refine ⟨by simpa using hwp, ?_⟩
  have hkeyRoute : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (emptyWordTable 0#i64) key = ok shard := by
    rw [← word_shard_index_state_independent state (emptyWordTable 0#i64)]
    exact hshard
  have hkeyMember := routedStreamKeys_current kvs i key value hi hindex shard hkeyRoute
  intro s current k hcurrent hk hoccupied j stored storedValue hj hentry
  by_cases hss : s.val = shard.val
  · have hs : s = shard := UScalar.eq_of_val_eq hss
    subst s
    have hcurrentGet := getElem?_eq_some_of_index_usize_eq_ok
      (insertVacant state shard (Array.repeat 16#usize none) 0#usize key
        (core.num.I64.wrapping_add value state.default)).shards shard (some current) hcurrent
    have hcurrentEq : current = (Array.repeat 16#usize none).set 0#usize
        (some (key, core.num.I64.wrapping_add value state.default)) := by
      simp only [insertVacant] at hcurrentGet
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ (by
        rw [state.shards.property]
        exact hshardBound)] at hcurrentGet
      exact Option.some.inj (Option.some.inj hcurrentGet).symm
    subst current
    have hj0 : j = 0 := by
      by_contra hne
      have hj16 : j < 16 := Nat.lt_of_lt_of_le hj hk
      have hentryOld : (List.replicate 16 (none : Option (I64 × I64)))[j]? =
          some (some (stored, storedValue)) := by
        simpa [Array.repeat_val, Array.set_val_eq, hne] using hentry
      rw [List.getElem?_eq_getElem (by simpa using hj16), List.getElem_replicate] at hentryOld
      simp at hentryOld
    subst j
    simp [Array.set_val_eq, Array.repeat_val] at hentry
    have hstored : stored = key := hentry.1.symm
    simpa [hstored] using hkeyMember
  · have hne : shard.val ≠ s.val := Ne.symm hss
    have houterOld : Array.index_usize state.shards s = ok (some current) := by
      simpa [insertVacant, Array.index_usize, Array.set_val_eq, hne] using hcurrent
    exact routedStreamKeys_mono kvs i.val hi s
      (hcoverage.2 s current k houterOld hk hoccupied j stored storedValue hj hentry)

/-- One generated insertion from a reachable prefix is successful under the stream-only distinct
    key bound, preserves the canonical reads, and advances physical stream coverage. -/
theorem reachablePrefix_insert (default : I64) (kvs : Slice (I64 × I64))
    (i : Usize) (state : WordTable) (key value : I64)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hrep : ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)))
    (hcoverage : ReachablePrefixCoverage kvs i.val state)
    (hdistinct : DistinctKeysPerShard kvs) :
    ∃ next, table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 state () key value =
        ok (core.result.Result.Ok (), next) ∧
      ReadRep next (insertWith core.num.I64.wrapping_add key value
        (orderedFold default ((sliceEntries kvs).take i.val))) ∧
      ReachablePrefixCoverage kvs (i.val + 1) next := by
  let locate := generatedLocator state
  have hroute : GeneratedRoute state key (locate key) := generatedLocator_routes state key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, slot⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      let next := replacePresent state shard slots slot key
        (core.num.I64.wrapping_add value old)
      have hinsert := insert_with_present_exact state key value shard slot slots key old
        hshard hfind houter hslot
      refine ⟨next, hinsert, ?_, ?_⟩
      · exact insert_with_present_readRep_preservation locate state next
          (orderedFold default ((sliceEntries kvs).take i.val)) key value shard slot slots old
          ((generatedFunctionalRep_iff_readRep state _).mpr hrep)
          (generatedLocator_routes state) hlocate hshard hfind houter hslot hinsert
      · exact present_reachablePrefixCoverage kvs i.val state key value shard slot slots old
          hcoverage hi hshard hfind houter hslot
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      obtain ⟨routedShard, hroutedShard, hshardBound⟩ := word_shard_index_spec state key
      have hshardEq : routedShard = shard :=
        Aeneas.Std.Result.ok_injective (hroutedShard.symm.trans hshard)
      subst routedShard
      have hshardLen : shard.val < state.shards.val.length := by
        rw [state.shards.property]
        exact hshardBound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec state.shards shard hshardLen)
      cases outer with
      | none =>
          let next := insertVacant state shard (Array.repeat 16#usize none) 0#usize key
            (core.num.I64.wrapping_add value state.default)
          have hinsert := insert_with_empty_shard_exact state key value shard hshard hfind
            houter (by decide) hshardBound
          refine ⟨next, hinsert, ?_, ?_⟩
          · have hpost := tableInv_insert_with state
              (orderedFold default ((sliceEntries kvs).take i.val)) key value
              hcoverage.1.1 hrep
            rw [hinsert] at hpost
            have hp : TableInv next ∧ ReadRep next
                (insertWith core.num.I64.wrapping_add key value
                  (orderedFold default ((sliceEntries kvs).take i.val))) := by
              simpa [BuildInsertInvResult] using hpost
            exact hp.2
          · exact emptyShard_reachablePrefixCoverage kvs i state key value shard hcoverage
              hi hindex hshard hfind houter hshardBound
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hcoverage.1.1 shard.val slots houterGet
          have hkLt : k < 16 := reachablePrefixCoverage_capacity kvs i.val state hcoverage
            hdistinct hi shard slots k houter hk hoccupied
          have hfindSlots : table.Table.find_slots core.cmp.EqI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64 slots key = ok none := by
            unfold table.Table.find at hfind
            rw [houter] at hfind
            simpa only [bind_tc_ok] using hfind
          obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
            (find_slots_spec_generic slots key 0#usize (Nat.zero_le _))
          unfold table.Table.find_slots at hfindSlots
          rw [hfindSlots] at hfound
          have hfoundNone : found = none := Aeneas.Std.Result.ok_injective hfound.symm
          subst found
          have hnomatch : NoMatchIn key slots.val 0 16 := by
            simpa [FindPostN] using hpost
          let first : Usize := ⟨k, by scalar_tac⟩
          have hfirst : first < 16#usize := hkLt
          have hslotGet : slots.val[first.val]? = some none := hvacant k le_rfl hkLt
          have hslot : Array.index_usize slots first = ok none :=
            index_usize_eq_ok_of_getElem?_eq_some slots first none hslotGet
          have hnomatchPrefix : NoMatchIn key slots.val 0 first.val :=
            fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt)
          let next := insertVacant state shard slots first key
            (core.num.I64.wrapping_add value state.default)
          have hinsert := insert_with_vacant_exact state key value shard first slots hshard
            hfind houter hfirst hoccupied hnomatchPrefix hslot
          refine ⟨next, hinsert, ?_, ?_⟩
          · exact insert_with_vacant_readRep_preservation_routed locate state next
              (orderedFold default ((sliceEntries kvs).take i.val)) key value shard first slots
              ((generatedFunctionalRep_iff_readRep state _).mpr hrep)
              (generatedLocator_routes state) hlocate hshard hfind houter hfirst hoccupied
              hnomatchPrefix hslot hinsert
          · exact vacant_reachablePrefixCoverage kvs i state key value shard first slots
              hcoverage hi hindex hshard hfind houter hfirst hoccupied hnomatchPrefix hslot

/-- The real generated build loop carries physical stream coverage together with its canonical
    prefix fold, so `DistinctKeysPerShard` rules out every `ShardFull` branch. -/
theorem table_build_loop_readRep_of_distinctKeysPerShard (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShard kvs) :
    ∃ final,
      table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
        (emptyWordTable default) 0#usize = ok (core.result.Result.Ok final) ∧
      ReadRep final (orderedFold default (sliceEntries kvs)) ∧
      ReachablePrefixCoverage kvs (sliceEntries kvs).length final := by
  have hinitial : ReadRep (emptyWordTable default) (orderedFold default []) := by
    apply (readRep_iff_Rep (emptyWordTable default) (orderedFold default [])).mpr
    intro key
    simpa [orderedFold] using emptyWordTable_read default key
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTable default, 0#usize)
      ⦃ r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)) ∧
        ReachablePrefixCoverage kvs (sliceEntries kvs).length final ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTable × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)) ∧
        ReachablePrefixCoverage kvs p.2.val p.1)
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)) ∧
        ReachablePrefixCoverage kvs (sliceEntries kvs).length final)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep, hcoverage⟩ := hp
      dsimp only at hi hrep hcoverage ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        have hiList : i.val < (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact hlt
        have hindex : Slice.index_usize kvs i = ok (key, value) := by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
        obtain ⟨next, hinsert, hrepNext, hcoverageNext⟩ :=
          reachablePrefix_insert default kvs i state key value hiList hindex hrep hcoverage
            hdistinct
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntries kvs).take (i.val + 1) =
            (sliceEntries kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post, orderedFold, htake] using hrepNext
        · simpa [r_post] using hcoverageNext
        · simp only [r_post]
          omega
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        exact ⟨state, rfl, by simpa [hieq] using hrep, by simpa [hieq] using hcoverage⟩
    · exact ⟨Nat.zero_le _, by simpa using hinitial,
        emptyWordTable_reachablePrefixCoverage default kvs⟩
  obtain ⟨result, hresult, final, hfinal, hrep, hcoverage⟩ :=
    Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, hrep, hcoverage⟩

/-- Stream-level distinct routed keys alone admit the complete real generated word build. -/
theorem word_table_build_readRep_of_distinctKeysPerShard_alone (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShard kvs) :
    ∃ final, word_table_build default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) := by
  obtain ⟨final, hloop, hrep, _⟩ :=
    table_build_loop_readRep_of_distinctKeysPerShard default kvs hdistinct
  refine ⟨final, ?_, ?_⟩
  · unfold word_table_build table.Table.build
    rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64
        default () = ok (emptyWordTable default) by rfl]
    simpa using hloop
  · simpa [orderedFold_eq_build] using hrep

/-! ## Exact reachable-prefix admission -/

/-- Bidirectional physical coverage for the unique states reached by successful generated prefixes.
    The locator records exactly the distinct routed keys in the processed prefix. -/
def ReachablePrefixExact (kvs : Slice (I64 × I64)) (i : Nat) (state : WordTable) : Prop :=
  ReachablePrefixCoverage kvs i state ∧
    ∃ locate, Routes state locate ∧
      ∀ (shard : Usize) (key : I64),
        (∃ slot, locate key = some (shard, slot)) ↔ key ∈ routedStreamKeys kvs i shard

private theorem emptyWordTable_routes_none (default : I64) :
    Routes (emptyWordTable default) (fun _ => none) := by
  set_option maxRecDepth 4096 in
    intro key
    obtain ⟨shard, hshard, hbound⟩ := word_shard_index_spec (emptyWordTable default) key
    refine ⟨shard, hshard, ?_⟩
    unfold table.Table.find
    have houter : Array.index_usize (emptyWordTable default).shards shard = ok none := by
      unfold Array.index_usize
      rw [Array.getElem?_Usize_eq]
      have hget : (emptyWordTable default).shards.val[shard.val]? = some none := by
        change (List.replicate 256
          (none : Option (Array (Option (I64 × I64)) 16#usize)))[shard.val]? = some none
        rw [List.getElem?_eq_getElem (by simpa using hbound), List.getElem_replicate]
      rw [hget]
    rw [houter]
    simp

/-- The generated empty constructor has exactly the empty routed-key set. -/
theorem emptyWordTable_reachablePrefixExact (default : I64)
    (kvs : Slice (I64 × I64)) : ReachablePrefixExact kvs 0 (emptyWordTable default) := by
  set_option maxRecDepth 4096 in
    refine ⟨emptyWordTable_reachablePrefixCoverage default kvs,
      fun _ => none, emptyWordTable_routes_none default, ?_⟩
    intro shard key
    simp [routedStreamKeys]

private theorem routedStreamKeys_succ_iff (kvs : Slice (I64 × I64)) (i : Usize)
    (key value : I64) (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value)) (shard routed : Usize)
    (hroute : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok routed) (x : I64) :
    x ∈ routedStreamKeys kvs (i.val + 1) shard ↔
      x ∈ routedStreamKeys kvs i.val shard ∨ (x = key ∧ shard = routed) := by
  classical
  have hget : (sliceEntries kvs)[i.val]? = some (key, value) := by
    unfold Slice.index_usize at hindex
    rw [Slice.getElem?_Usize_eq] at hindex
    unfold sliceEntries
    split at hindex <;> simp_all
  have hentry : (sliceEntries kvs)[i.val] = (key, value) :=
    (List.getElem?_eq_some_iff.mp hget).2
  have hkeys : (((sliceEntries kvs).take (i.val + 1)).map Prod.fst).toFinset =
      insert key ((((sliceEntries kvs).take i.val).map Prod.fst).toFinset) := by
    rw [List.take_succ_eq_append_getElem hi]
    simp [hentry]
  simp only [routedStreamKeys, Finset.mem_filter]
  rw [hkeys]
  simp only [Finset.mem_insert]
  constructor
  · rintro ⟨hx | hx, hxroute⟩
    · subst x
      exact Or.inr ⟨rfl, Aeneas.Std.Result.ok_injective (hxroute.symm.trans hroute)⟩
    · exact Or.inl ⟨hx, hxroute⟩
  · rintro (⟨hx, hxroute⟩ | ⟨rfl, rfl⟩)
    · exact ⟨Or.inr hx, hxroute⟩
    · exact ⟨Or.inl rfl, hroute⟩

private theorem routes_insertEmptyShard_exact
    (locate : I64 → Option (Usize × Usize)) (state : WordTable) (key value : I64)
    (shard : Usize) (hroutes : Routes state locate)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (houter : Array.index_usize state.shards shard = ok none)
    (hshardBound : shard < 256#usize) :
    Routes (insertVacant state shard (Array.repeat 16#usize none) 0#usize key value)
      (insertLocator locate key shard 0#usize) := by
  intro x
  let next := insertVacant state shard (Array.repeat 16#usize none) 0#usize key value
  have hnewOuter : Array.index_usize next.shards shard =
      ok (some ((Array.repeat 16#usize none).set 0#usize (some (key, value)))) := by
    apply index_usize_eq_ok_of_getElem?_eq_some
    simp only [next, insertVacant, Array.set_val_eq]
    have hsLen : shard.val < state.shards.val.length := by
      rw [state.shards.property]
      exact hshardBound
    exact List.getElem?_set_eq_of_lt _ hsLen
  have hrepeatSlot : Array.index_usize (Array.repeat 16#usize
      (none : Option (I64 × I64))) 0#usize = ok none := by
    apply index_usize_eq_ok_of_getElem?_eq_some
    simp [Array.repeat_val]
  by_cases hx : x = key
  · subst x
    simp only [insertLocator]
    apply generatedRoute_some_of_find next key shard 0#usize
    · rw [← word_shard_index_state_independent state next key]
      exact hshard
    · unfold table.Table.find
      rw [hnewOuter]
      simp only [bind_tc_ok]
      exact find_slots_insertVacant_self (Array.repeat 16#usize none) 0#usize key value
        (by decide) (fun j hj hjlt _ => (Nat.not_lt_of_ge hj) hjlt)
  · simp only [insertLocator, if_neg hx]
    have hroute := hroutes x
    cases hloc : locate x with
    | none =>
        rw [hloc] at hroute
        obtain ⟨s, hs, hf⟩ := hroute
        apply generatedRoute_none_of_find next x s
        · rw [← word_shard_index_state_independent state next x]
          exact hs
        · by_cases hsame : s = shard
          · subst s
            unfold table.Table.find
            rw [hnewOuter]
            simp only [bind_tc_ok]
            rw [find_slots_insertVacant_ne (Array.repeat 16#usize none) 0#usize key value x
              hrepeatSlot hx]
            obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
              (find_slots_spec_generic (Array.repeat 16#usize
                (none : Option (I64 × I64))) x 0#usize (Nat.zero_le _))
            cases found with
            | none => exact hfound
            | some slot =>
                obtain ⟨hslot, _, ⟨stored, hmatch⟩, _⟩ := hpost
                change (List.replicate 16 (none : Option (I64 × I64)))[slot.val]? =
                  some (some (x, stored)) at hmatch
                rw [List.getElem?_eq_getElem (by simpa using hslot),
                  List.getElem_replicate] at hmatch
                simp at hmatch
          · have hne : shard.val ≠ s.val := fun h => hsame (UScalar.eq_of_val_eq h.symm)
            simpa [next, insertVacant, table.Table.find, Array.index_usize,
              Array.set_val_eq, hne] using hf
    | some p =>
        rcases p with ⟨s, slot⟩
        rw [hloc] at hroute
        obtain ⟨oldSlots, oldValue, hs, hf, ho, hi⟩ := hroute
        have hsame : s ≠ shard := by
          intro heq
          subst s
          rw [houter] at ho
          simp at ho
        apply generatedRoute_some_of_find next x s slot
        · rw [← word_shard_index_state_independent state next x]
          exact hs
        · have hne : shard.val ≠ s.val := fun h => hsame (UScalar.eq_of_val_eq h.symm)
          simpa [next, insertVacant, table.Table.find, Array.index_usize,
            Array.set_val_eq, hne] using hf

private theorem reachablePrefixCoverage_capacity_of_card_lt
    (kvs : Slice (I64 × I64)) (i : Nat) (state : WordTable)
    (hcoverage : ReachablePrefixCoverage kvs i state) (shard : Usize)
    (hcard : (routedStreamKeys kvs i shard).card < 16)
    (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hk : k ≤ 16) (hoccupied : NoEmptyIn slots.val 0 k) : k < 16 := by
  have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
    (some slots) houter
  obtain ⟨keys, hkeysInjective, hkeysSlots⟩ :=
    occupiedPrefix_distinct_key_count slots k hk hoccupied
      (hcoverage.1.2 shard.val slots houterGet)
  let intoStream : Fin k → { key // key ∈ routedStreamKeys kvs i shard } := fun j =>
    ⟨keys j, by
      obtain ⟨stored, hslot⟩ := hkeysSlots j
      exact hcoverage.2 shard slots k houter hk hoccupied j.val (keys j) stored j.isLt hslot⟩
  have hinjective : Function.Injective intoStream := by
    intro a b hab
    apply hkeysInjective
    exact Subtype.ext_iff.mp hab
  have hle : k ≤ (routedStreamKeys kvs i shard).card := by
    simpa [intoStream] using Fintype.card_le_of_injective intoStream hinjective
  exact Nat.lt_of_le_of_lt hle hcard

private theorem insertLocator_exact_succ (kvs : Slice (I64 × I64)) (i : Usize)
    (key value : I64) (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (locate : I64 → Option (Usize × Usize)) (routed first : Usize)
    (hlocate : locate key = none)
    (hexact : ∀ (shard : Usize) (x : I64),
      (∃ slot, locate x = some (shard, slot)) ↔ x ∈ routedStreamKeys kvs i.val shard)
    (hroute : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok routed) :
    ∀ (shard : Usize) (x : I64),
      (∃ slot, insertLocator locate key routed first x = some (shard, slot)) ↔
        x ∈ routedStreamKeys kvs (i.val + 1) shard := by
  intro shard x
  rw [routedStreamKeys_succ_iff kvs i key value hi hindex shard routed hroute x]
  by_cases hx : x = key
  · subst x
    have hnot : key ∉ routedStreamKeys kvs i.val shard := by
      intro hm
      obtain ⟨slot, hslot⟩ := (hexact shard key).mpr hm
      rw [hlocate] at hslot
      simp at hslot
    simpa [insertLocator, hnot] using (@eq_comm Usize routed shard)
  · simp [insertLocator, hx, hexact shard x]

private theorem presentLocator_exact_succ (kvs : Slice (I64 × I64)) (i : Usize)
    (key value : I64) (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (locate : I64 → Option (Usize × Usize)) (routed presentSlot : Usize)
    (hlocate : locate key = some (routed, presentSlot))
    (hexact : ∀ (shard : Usize) (x : I64),
      (∃ slot, locate x = some (shard, slot)) ↔ x ∈ routedStreamKeys kvs i.val shard)
    (hroute : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok routed) :
    ∀ (shard : Usize) (x : I64),
      (∃ slot, locate x = some (shard, slot)) ↔
        x ∈ routedStreamKeys kvs (i.val + 1) shard := by
  intro shard x
  rw [routedStreamKeys_succ_iff kvs i key value hi hindex shard routed hroute x]
  constructor
  · intro h
    exact Or.inl ((hexact shard x).mp h)
  · rintro (h | ⟨rfl, rfl⟩)
    · exact (hexact shard x).mpr h
    · exact ⟨presentSlot, hlocate⟩

/-- A successful generated insertion preserves exact physical/prefix key-set equality.  A repeated
    key succeeds even at load sixteen; a fresh key requires strict capacity in its routed shard. -/
theorem reachablePrefixExact_insert (default : I64) (kvs : Slice (I64 × I64))
    (i : Usize) (state : WordTable) (key value : I64)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hrep : ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)))
    (hexact : ReachablePrefixExact kvs i.val state)
    (hadmit : ∀ routed,
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok routed →
        key ∈ routedStreamKeys kvs i.val routed ∨
          (routedStreamKeys kvs i.val routed).card < 16) :
    ∃ next, table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 state () key value =
        ok (core.result.Result.Ok (), next) ∧
      ReadRep next (insertWith core.num.I64.wrapping_add key value
        (orderedFold default ((sliceEntries kvs).take i.val))) ∧
      ReachablePrefixExact kvs (i.val + 1) next := by
  obtain ⟨hcoverage, locate, hroutes, hkeys⟩ := hexact
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, slot⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      let next := replacePresent state shard slots slot key
        (core.num.I64.wrapping_add value old)
      have hinsert := insert_with_present_exact state key value shard slot slots key old
        hshard hfind houter hslot
      have hrouted : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok shard := by
        rw [← word_shard_index_state_independent state (emptyWordTable 0#i64)]
        exact hshard
      refine ⟨next, hinsert, ?_, ?_, locate, ?_, ?_⟩
      · exact insert_with_present_readRep_preservation locate state next
          (orderedFold default ((sliceEntries kvs).take i.val)) key value shard slot slots old
          ((functionalRep_iff_readRep_of_routes locate state _ hroutes).mpr hrep)
          hroutes hlocate hshard hfind houter hslot hinsert
      · exact present_reachablePrefixCoverage kvs i.val state key value shard slot slots old
          hcoverage hi hshard hfind houter hslot
      · exact routes_replacePresent locate state key
          (core.num.I64.wrapping_add value old) shard slot slots old hroutes houter hslot
      · exact presentLocator_exact_succ kvs i key value hi hindex locate shard slot hlocate
          hkeys hrouted
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      have hrouted : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok shard := by
        rw [← word_shard_index_state_independent state (emptyWordTable 0#i64)]
        exact hshard
      have hnot : key ∉ routedStreamKeys kvs i.val shard := by
        intro hm
        obtain ⟨slot, hs⟩ := (hkeys shard key).mpr hm
        rw [hlocate] at hs
        simp at hs
      have hcard : (routedStreamKeys kvs i.val shard).card < 16 :=
        (hadmit shard hrouted).resolve_left hnot
      obtain ⟨routedShard, hroutedShard, hshardBound⟩ := word_shard_index_spec state key
      have hshardEq : routedShard = shard :=
        Aeneas.Std.Result.ok_injective (hroutedShard.symm.trans hshard)
      subst routedShard
      have hshardLen : shard.val < state.shards.val.length := by
        rw [state.shards.property]
        exact hshardBound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec state.shards shard hshardLen)
      cases outer with
      | none =>
          let next := insertVacant state shard (Array.repeat 16#usize none) 0#usize key
            (core.num.I64.wrapping_add value state.default)
          have hinsert := insert_with_empty_shard_exact state key value shard hshard hfind
            houter (by decide) hshardBound
          refine ⟨next, hinsert, ?_, ?_, insertLocator locate key shard 0#usize, ?_, ?_⟩
          · have hpost := tableInv_insert_with state
                (orderedFold default ((sliceEntries kvs).take i.val)) key value
                hcoverage.1.1 hrep
            rw [hinsert] at hpost
            have hp : TableInv next ∧ ReadRep next
                (insertWith core.num.I64.wrapping_add key value
                  (orderedFold default ((sliceEntries kvs).take i.val))) := by
              simpa [BuildInsertInvResult] using hpost
            exact hp.2
          · exact emptyShard_reachablePrefixCoverage kvs i state key value shard hcoverage
              hi hindex hshard hfind houter hshardBound
          · exact routes_insertEmptyShard_exact locate state key
              (core.num.I64.wrapping_add value state.default) shard hroutes hshard houter
              hshardBound
          · exact insertLocator_exact_succ kvs i key value hi hindex locate shard 0#usize
              hlocate hkeys hrouted
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hcoverage.1.1 shard.val slots houterGet
          have hkLt : k < 16 := reachablePrefixCoverage_capacity_of_card_lt kvs i.val state
            hcoverage shard hcard slots k houter hk hoccupied
          have hfindSlots : table.Table.find_slots core.cmp.EqI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64 slots key = ok none := by
            unfold table.Table.find at hfind
            rw [houter] at hfind
            simpa only [bind_tc_ok] using hfind
          obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
            (find_slots_spec_generic slots key 0#usize (Nat.zero_le _))
          unfold table.Table.find_slots at hfindSlots
          rw [hfindSlots] at hfound
          have : found = none := Aeneas.Std.Result.ok_injective hfound.symm
          subst found
          have hnomatch : NoMatchIn key slots.val 0 16 := by simpa [FindPostN] using hpost
          let first : Usize := ⟨k, by scalar_tac⟩
          have hfirst : first < 16#usize := hkLt
          have hslot : Array.index_usize slots first = ok none :=
            index_usize_eq_ok_of_getElem?_eq_some slots first none (hvacant k le_rfl hkLt)
          have hnomatchPrefix : NoMatchIn key slots.val 0 first.val :=
            fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt)
          let next := insertVacant state shard slots first key
            (core.num.I64.wrapping_add value state.default)
          have hinsert := insert_with_vacant_exact state key value shard first slots hshard
            hfind houter hfirst hoccupied hnomatchPrefix hslot
          refine ⟨next, hinsert, ?_, ?_, insertLocator locate key shard first, ?_, ?_⟩
          · exact insert_with_vacant_readRep_preservation_routed locate state next
              (orderedFold default ((sliceEntries kvs).take i.val)) key value shard first slots
              ((functionalRep_iff_readRep_of_routes locate state _ hroutes).mpr hrep)
              hroutes hlocate hshard hfind houter hfirst hoccupied hnomatchPrefix hslot hinsert
          · exact vacant_reachablePrefixCoverage kvs i state key value shard first slots
              hcoverage hi hindex hshard hfind houter hfirst hoccupied hnomatchPrefix hslot
          · exact routes_insertVacant locate state key
              (core.num.I64.wrapping_add value state.default) shard first slots hroutes hshard
              houter hfirst hnomatchPrefix hslot
          · exact insertLocator_exact_succ kvs i key value hi hindex locate shard first
              hlocate hkeys hrouted

/-- Exact stream condition needed at each generated prefix: the next key is already present in its
    routed shard, or that shard contains fewer than sixteen distinct prefix keys.  Unlike
    `DistinctKeysPerShard`, this permits load sixteen when all later writes to that shard repeat
    existing keys. -/
def BuildAdmissible (_default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (key value : I64) (shard : Usize),
    i.val < (sliceEntries kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok shard →
    key ∈ routedStreamKeys kvs i.val shard ∨
      (routedStreamKeys kvs i.val shard).card < 16

/-- The earlier strict-prefix condition is sufficient for exact admission, but is stronger than
    necessary because it rejects any later stream position after a shard first reaches load sixteen. -/
theorem buildAdmissible_of_distinctKeysPerShard (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShard kvs) :
    BuildAdmissible default kvs := by
  intro i key value shard hi _ _
  exact Or.inr (hdistinct i.val shard hi)

/-- Exact per-step admission makes the real generated build succeed and gives the canonical ordered
    read representation. -/
theorem word_table_build_readRep_of_buildAdmissible (default : I64)
    (kvs : Slice (I64 × I64)) (hadmissible : BuildAdmissible default kvs) :
    ∃ final, word_table_build default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) := by
  have hinitial : ReadRep (emptyWordTable default) (orderedFold default []) := by
    apply (readRep_iff_Rep (emptyWordTable default) (orderedFold default [])).mpr
    intro key
    simpa [orderedFold] using emptyWordTable_read default key
  unfold word_table_build table.Table.build
  rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64
      default () = ok (emptyWordTable default) by rfl]
  simp only [bind_tc_ok]
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTable default, 0#usize)
      ⦃ r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTable × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)) ∧
        ReachablePrefixExact kvs p.2.val p.1)
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)))
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep, hexact⟩ := hp
      dsimp only at hi hrep hexact ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        have hiList : i.val < (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact hlt
        have hindex : Slice.index_usize kvs i = ok (key, value) := by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
        obtain ⟨next, hinsert, hrepNext, hexactNext⟩ :=
          reachablePrefixExact_insert default kvs i state key value hiList hindex hrep
            hexact (fun routed hrouted => hadmissible i key value routed hiList hindex hrouted)
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntries kvs).take (i.val + 1) =
            (sliceEntries kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post, orderedFold, htake] using hrepNext
        · simpa [r_post] using hexactNext
        · simp only [r_post]
          omega
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        exact ⟨state, rfl, by simpa [hieq] using hrep⟩
    · exact ⟨Nat.zero_le _, by simpa using hinitial,
        emptyWordTable_reachablePrefixExact default kvs⟩
  obtain ⟨result, hresult, final, hfinal, hrep⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, by simpa [orderedFold_eq_build] using hrep⟩

/-- Sound refusal direction: a real generated `ShardFull` result contradicts exact per-step
    admission.  The converse requires the residual first-bad-prefix fullness lemma. -/
theorem word_table_build_refusal_implies_not_buildAdmissible (default : I64)
    (kvs : Slice (I64 × I64))
    (hrefusal : word_table_build default kvs =
      ok (core.result.Result.Err table.StorageError.ShardFull)) :
    ¬ BuildAdmissible default kvs := by
  intro hadmissible
  obtain ⟨final, hsuccess, _⟩ :=
    word_table_build_readRep_of_buildAdmissible default kvs hadmissible
  rw [hrefusal] at hsuccess
  simp at hsuccess

private theorem insert_with_loop0_full_direct {N : Usize}
    (back : Option (Array (Option (I64 × I64)) N) →
      Array (Option (Array (Option (I64 × I64)) N)) 256#usize)
    (slots : Array (Option (I64 × I64)) N) (default key value : I64)
    (start : Usize) (hstart : start.val ≤ N.val)
    (hfull : NoEmptyIn slots.val start.val N.val) :
    table.Table.insert_with_loop0 core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 back default () key value slots start
      ⦃ r => r = (core.result.Result.Err table.StorageError.ShardFull, back (some slots)) ⦄ := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => N.val - i.val)
    (inv := fun i => InsertFullInvN (N := N) slots.val start.val i)
    (post := fun r => r =
      (core.result.Result.Err table.StorageError.ShardFull, back (some slots)))
  · intro i hiInv
    obtain ⟨hile, hstartle, hno⟩ := hiInv
    unfold table.Table.insert_with_loop0.body
    by_cases hi : i < N
    · rw [if_pos hi]
      step
      rcases o with _ | occupied
      · have hib : i.val < slots.val.length := by rw [slots.property]; exact hi
        exact (hfull i.val hstartle hi (emptyAt_of_none hib o_post.symm)).elim
      · simp
        step
        have hin : i.val < N.val := hi
        have hnextBound : i.val + 1 ≤ N.val := by omega
        have hnextStart : start.val ≤ i.val + 1 := by omega
        have hdec : N.val - (i.val + 1) < N.val - i.val := by omega
        refine ⟨⟨by simpa [i1_post] using hnextBound, ?_, ?_⟩, ?_⟩
        · simpa [i1_post] using hnextStart
        · simpa [i1_post] using noEmptyIn_succ hno (hfull i.val hstartle hi)
        · simpa [i1_post] using hdec
    · rw [if_neg hi, Aeneas.Std.WP.spec_ok]
  · exact ⟨hstart, le_rfl, fun j hj hjlt => by omega⟩

private theorem insert_with_full_direct (t : WordTable) (key value : I64) (shard : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfull : NoEmptyIn slots.val 0 16) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Err table.StorageError.ShardFull, t) := by
  have hm : Array.index_mut_usize t.shards shard = ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (insert_with_loop0_full_direct (t.shards.set shard) slots t.default key value
      0#usize (Nat.zero_le _) hfull)
  rw [hp] at hr
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg (by decide), houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_neg (by simp), hm]
  simp only [bind_tc_ok]
  change (do
    let (result, shards) ← table.Table.insert_with_loop0 core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 (t.shards.set shard)
      t.default () key value slots 0#usize
    ok (result, { t with shards := shards })) = _
  rw [hr]
  have houter' := getElem?_eq_some_of_index_usize_eq_ok t.shards shard (some slots) houter
  have hsLen : shard.val < t.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
    simp at houter'
  have hset : t.shards.set shard (some slots) = t.shards := by
    apply Aeneas.Std.Array.ext
    simp only [Array.set_val_eq]
    apply List.ext_getElem?
    intro n
    rw [List.getElem?_set_of_lt' (some slots) t.shards.val hsLen]
    split
    · rename_i hn
      subst n
      exact houter'.symm
    · rfl
  simp [hset]

private theorem reachablePrefixExact_card_le_occupiedPrefix
    (kvs : Slice (I64 × I64)) (i : Nat) (state : WordTable)
    (hexact : ReachablePrefixExact kvs i state) (shard : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hvacant : ∀ j, k ≤ j → j < 16 → slots.val[j]? = some none) :
    (routedStreamKeys kvs i shard).card ≤ k := by
  classical
  obtain ⟨_, locate, hroutes, hkeys⟩ := hexact
  have hslotExists : ∀ x : { key // key ∈ routedStreamKeys kvs i shard },
      ∃ slot, locate x.val = some (shard, slot) := fun x => (hkeys shard x.val).mpr x.property
  let slotOf : { key // key ∈ routedStreamKeys kvs i shard } → Usize := fun x =>
    Classical.choose (hslotExists x)
  have hslotOf : ∀ x, locate x.val = some (shard, slotOf x) := fun x =>
    Classical.choose_spec (hslotExists x)
  have hslotLt : ∀ x, (slotOf x).val < k := by
    intro x
    have hroute := hroutes x.val
    rw [hslotOf x] at hroute
    obtain ⟨routedSlots, value, _, _, hrouteOuter, hrouteSlot⟩ := hroute
    have hroutedSlots : routedSlots = slots := by
      exact Option.some.inj (Aeneas.Std.Result.ok_injective (hrouteOuter.symm.trans houter))
    subst routedSlots
    have hslotGet := getElem?_eq_some_of_index_usize_eq_ok slots (slotOf x)
      (some (x.val, value)) hrouteSlot
    have hlt16 : (slotOf x).val < 16 := by
      by_contra h
      rw [List.getElem?_eq_none (by rw [slots.property]; exact Nat.not_lt.mp h)] at hslotGet
      simp at hslotGet
    by_contra h
    rw [hvacant (slotOf x).val (Nat.not_lt.mp h) hlt16] at hslotGet
    simp at hslotGet
  let toSlot : { key // key ∈ routedStreamKeys kvs i shard } → Fin k := fun x =>
    ⟨(slotOf x).val, hslotLt x⟩
  have hinjective : Function.Injective toSlot := by
    intro x y hxy
    have hslotEq : slotOf x = slotOf y := by
      apply UScalar.eq_of_val_eq
      simpa [toSlot] using congrArg (fun z : Fin k => z.val) hxy
    have hxroute := hroutes x.val
    have hyroute := hroutes y.val
    rw [hslotOf x] at hxroute
    rw [hslotOf y, ← hslotEq] at hyroute
    obtain ⟨xslots, xvalue, _, _, hxouter, hxslot⟩ := hxroute
    obtain ⟨yslots, yvalue, _, _, hyouter, hyslot⟩ := hyroute
    have hxslots : xslots = slots := by
      exact Option.some.inj (Aeneas.Std.Result.ok_injective (hxouter.symm.trans houter))
    have hyslots : yslots = slots := by
      exact Option.some.inj (Aeneas.Std.Result.ok_injective (hyouter.symm.trans houter))
    subst xslots
    subst yslots
    have hpairs : (x.val, xvalue) = (y.val, yvalue) := by
      exact Option.some.inj (Aeneas.Std.Result.ok_injective (hxslot.symm.trans hyslot))
    exact Subtype.ext (congrArg Prod.fst hpairs)
  simpa [toSlot] using Fintype.card_le_of_injective toSlot hinjective

/-- A successful insertion from an exact generated prefix satisfies the tight admission condition
    for that step. -/
theorem reachablePrefixExact_admissible_of_insert_success
    (_default : I64) (kvs : Slice (I64 × I64)) (i : Usize) (state next : WordTable)
    (key value : I64) (hexact : ReachablePrefixExact kvs i.val state)
    (hsuccess : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 state () key value =
      ok (core.result.Result.Ok (), next)) :
    ∀ shard,
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok shard →
        key ∈ routedStreamKeys kvs i.val shard ∨
          (routedStreamKeys kvs i.val shard).card < 16 := by
  obtain ⟨hcoverage, locate, hroutes, hkeys⟩ := hexact
  intro shard hrouted
  have hroute := hroutes key
  cases hlocate : locate key with
  | some location =>
      rcases location with ⟨locatedShard, slot⟩
      have hlocated : locatedShard = shard := by
        rw [hlocate] at hroute
        obtain ⟨_, _, hshard, _⟩ := hroute
        have hstateRoute : table.Table.shard_index
            table.WordHome.Insts.Apeiron_wordTableHomeI64 state key = ok shard := by
          rw [word_shard_index_state_independent state (emptyWordTable 0#i64)]
          exact hrouted
        exact Aeneas.Std.Result.ok_injective (hshard.symm.trans hstateRoute)
      subst locatedShard
      exact Or.inl ((hkeys shard key).mp ⟨slot, hlocate⟩)
  | none =>
      rw [hlocate] at hroute
      obtain ⟨locatedShard, hshard, hfind⟩ := hroute
      have hstateRoute : table.Table.shard_index
          table.WordHome.Insts.Apeiron_wordTableHomeI64 state key = ok shard := by
        rw [word_shard_index_state_independent state (emptyWordTable 0#i64)]
        exact hrouted
      have hlocated : locatedShard = shard :=
        Aeneas.Std.Result.ok_injective (hshard.symm.trans hstateRoute)
      subst locatedShard
      right
      obtain ⟨boundedShard, hboundedRoute, hshardBound⟩ := word_shard_index_spec state key
      have hbounded : boundedShard = shard :=
        Aeneas.Std.Result.ok_injective (hboundedRoute.symm.trans hstateRoute)
      subst boundedShard
      have hshardLen : shard.val < state.shards.val.length := by
        rw [state.shards.property]
        exact hshardBound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec state.shards shard hshardLen)
      cases outer with
      | none =>
          have hempty : routedStreamKeys kvs i.val shard = ∅ := by
            apply Finset.eq_empty_iff_forall_notMem.mpr
            intro x hx
            obtain ⟨slot, hslot⟩ := (hkeys shard x).mpr hx
            have hxroute := hroutes x
            rw [hslot] at hxroute
            obtain ⟨slots, stored, _, _, hxouter, _⟩ := hxroute
            rw [houter] at hxouter
            simp at hxouter
          simp [hempty]
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hcoverage.1.1 shard.val slots houterGet
          have hkLt : k < 16 := by
            by_contra h
            have hkEq : k = 16 := Nat.le_antisymm hk (Nat.not_lt.mp h)
            subst k
            have hfull := insert_with_full_direct state key value shard slots hstateRoute hfind
              houter hoccupied
            rw [hfull] at hsuccess
            simp at hsuccess
          exact Nat.lt_of_le_of_lt
            (reachablePrefixExact_card_le_occupiedPrefix kvs i.val state
              ⟨hcoverage, locate, hroutes, hkeys⟩ shard slots k houter hvacant)
            hkLt

private def PrefixAdmissible (kvs : Slice (I64 × I64)) (limit : Nat) : Prop :=
  ∀ (i : Usize) (key value : I64) (shard : Usize),
    i.val < limit →
    Slice.index_usize kvs i = ok (key, value) →
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok shard →
    key ∈ routedStreamKeys kvs i.val shard ∨
      (routedStreamKeys kvs i.val shard).card < 16

/-- Any successful result from the real generated word build satisfies every tight per-step
    admission condition. -/
theorem word_table_build_success_implies_buildAdmissible (default : I64)
    (kvs : Slice (I64 × I64)) (final : WordTable)
    (hsuccess : word_table_build default kvs = ok (core.result.Result.Ok final)) :
    BuildAdmissible default kvs := by
  have hinitial : ReadRep (emptyWordTable default) (orderedFold default []) := by
    apply (readRep_iff_Rep (emptyWordTable default) (orderedFold default [])).mpr
    intro key
    simpa [orderedFold] using emptyWordTable_read default key
  have hloopSuccess : table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
      (emptyWordTable default) 0#usize = ok (core.result.Result.Ok final) := by
    unfold word_table_build table.Table.build at hsuccess
    rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64
        default () = ok (emptyWordTable default) by rfl] at hsuccess
    simpa only [bind_tc_ok] using hsuccess
  unfold table.Table.build_loop at hloopSuccess
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (emptyWordTable default, 0#usize)
      ⦃ result => ∀ final, result = core.result.Result.Ok final →
        BuildAdmissible default kvs ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTable × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)) ∧
        ReachablePrefixExact kvs p.2.val p.1 ∧ PrefixAdmissible kvs p.2.val)
      (post := fun result => ∀ final, result = core.result.Result.Ok final →
        BuildAdmissible default kvs)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep, hexact, hprefix⟩ := hp
      dsimp only at hi hrep hexact hprefix ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        have hiList : i.val < (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact hlt
        have hindex : Slice.index_usize kvs i = ok (key, value) := by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
        obtain ⟨result, hinsert, hresult⟩ := Aeneas.Std.WP.spec_imp_exists
          (buildInsertResult_of_readRep state
            (orderedFold default ((sliceEntries kvs).take i.val)) key value hrep)
        rcases result with ⟨result, next⟩
        cases result with
        | Ok resultUnit =>
            cases resultUnit
            have hadmit := reachablePrefixExact_admissible_of_insert_success default kvs i
              state next key value hexact hinsert
            obtain ⟨advanced, hadvancedInsert, hrepNext, hexactNext⟩ :=
              reachablePrefixExact_insert default kvs i state key value hiList hindex hrep
                hexact hadmit
            have hadvanced : advanced = next := by
              rw [hinsert] at hadvancedInsert
              exact (congrArg Prod.snd
                (Aeneas.Std.Result.ok_injective hadvancedInsert)).symm
            subst advanced
            rw [hinsert]
            simp only [bind_tc_ok]
            have htake : (sliceEntries kvs).take (i.val + 1) =
                (sliceEntries kvs).take i.val ++ [(key, value)] := by
              rw [List.take_succ_eq_append_getElem hiList]
              congr
              exact key_post.symm
            step
            refine ⟨?_, ?_, ?_, ?_, ?_⟩
            · simpa [r_post] using Nat.succ_le_iff.mpr hiList
            · simpa [r_post, orderedFold, htake] using hrepNext
            · simpa [r_post] using hexactNext
            · intro j stored storedValue shard hj hentry hroute
              by_cases hji : j.val < i.val
              · exact hprefix j stored storedValue shard hji hentry hroute
              · have hjiEq : j.val = i.val := by omega
                have hjiUsize : j = i := UScalar.eq_of_val_eq hjiEq
                subst j
                have hpair : (stored, storedValue) = (key, value) :=
                  Aeneas.Std.Result.ok_injective (hentry.symm.trans hindex)
                have hstored : stored = key := congrArg Prod.fst hpair
                rw [hstored] at hroute ⊢
                exact hadmit shard hroute
            · simp only [r_post]
              omega
        | Err error =>
            cases error
            rw [hinsert]
            simp
      · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
        intro result _
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        intro j key value shard hj hindex hroute
        exact hprefix j key value shard (by simpa [hieq] using hj) hindex hroute
    · exact ⟨Nat.zero_le _, by simpa using hinitial,
        emptyWordTable_reachablePrefixExact default kvs, by
          intro i key value shard hi _ _
          change i.val < 0 at hi
          omega⟩
  obtain ⟨result, hloop, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  exact hpost final (Aeneas.Std.Result.ok_injective (hloop.symm.trans hloopSuccess))

/-- Failure of the tight per-step admission condition forces the real generated build to report
    `ShardFull`. -/
theorem word_table_build_refuses_of_not_buildAdmissible (default : I64)
    (kvs : Slice (I64 × I64)) (h : ¬ BuildAdmissible default kvs) :
    word_table_build default kvs =
      ok (core.result.Result.Err table.StorageError.ShardFull) := by
  rcases word_table_build_result_readRep default kvs with hrefusal | ⟨final, hsuccess, _⟩
  · exact hrefusal
  · exact (h (word_table_build_success_implies_buildAdmissible default kvs final hsuccess)).elim

/-- The real generated word build reports `ShardFull` exactly on streams outside the tight
    per-step admission condition. -/
theorem word_table_build_refuses_iff_not_buildAdmissible (default : I64)
    (kvs : Slice (I64 × I64)) :
    word_table_build default kvs =
        ok (core.result.Result.Err table.StorageError.ShardFull) ↔
      ¬ BuildAdmissible default kvs :=
  ⟨word_table_build_refusal_implies_not_buildAdmissible default kvs,
    word_table_build_refuses_of_not_buildAdmissible default kvs⟩

end Apeiron.TableRefinement
