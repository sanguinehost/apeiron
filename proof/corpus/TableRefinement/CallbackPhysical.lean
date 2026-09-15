import TableRefinement.Callback

/-! # Physical preservation for fresh generic callback insertion

The callback module proves exact generated states and present-key preservation.  This module
discharges the two fresh-key states: a vacant slot in a populated shard and slot zero in an
untouched shard.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

private theorem physical_getElem?_of_index {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

private theorem physical_index_of_getElem? {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : a.val[i.val]? = some x) :
    Array.index_usize a i = ok x := by
  unfold Array.index_usize
  rw [Array.getElem?_Usize_eq, h]

private theorem physical_outer_bound {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize)
    (entry : Option (Array (Option (K × V)) N))
    (hindex : Array.index_usize t.shards shard = ok entry) : shard.val < 256 := by
  have hget := physical_getElem?_of_index t.shards shard entry hindex
  have hlength : shard.val < t.shards.val.length := by
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at hget
    simp at hget
  simpa [t.shards.property] using hlength

private theorem physical_inner_bound {K V : Type} {N : Usize}
    (slots : Array (Option (K × V)) N) (i : Usize) (entry : Option (K × V))
    (hindex : Array.index_usize slots i = ok entry) : i < N := by
  have hget := physical_getElem?_of_index slots i entry hindex
  have hlength : i.val < slots.val.length := by
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at hget
    simp at hget
  simpa [slots.property] using hlength

private theorem replacePresentG_entry_empty_cases {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none)
    (s j : Usize) (current : Array (Option (K × V)) N) (stored : K) (storedValue : V)
    (hs : Array.index_usize (replacePresentG t shard slots i key new).shards s =
      ok (some current))
    (hj : Array.index_usize current j = ok (some (stored, storedValue))) :
    (s = shard ∧ j = i ∧ stored = key ∧ storedValue = new) ∨
      ∃ oldSlots oldValue,
        Array.index_usize t.shards s = ok (some oldSlots) ∧
        Array.index_usize oldSlots j = ok (some (stored, oldValue)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using physical_outer_bound t shard (some slots) houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots.set i (some (key, new)) := by
      have hs' := physical_getElem?_of_index
        (replacePresentG t shard slots i key new).shards shard (some current) hs
      simp only [replacePresentG] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    by_cases hji : j = i
    · subst j
      have hnew := physical_getElem?_of_index (slots.set i (some (key, new))) i
        (some (stored, storedValue)) hj
      have hi : i.val < slots.val.length := by
        simpa [slots.property] using physical_inner_bound slots i none hslot
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi] at hnew
      have hpair : (stored, storedValue) = (key, new) :=
        Option.some.inj (Option.some.inj hnew.symm)
      exact Or.inl ⟨rfl, rfl, congrArg Prod.fst hpair, congrArg Prod.snd hpair⟩
    · right
      have hjiVal : j.val ≠ i.val := fun h => hji (UScalar.eq_of_val_eq h)
      have hjOld : Array.index_usize slots j = ok (some (stored, storedValue)) := by
        simpa [Array.index_usize, Array.set_val_eq, hjiVal] using hj
      exact ⟨slots, storedValue, houter, hjOld⟩
  · right
    have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [replacePresentG, Array.index_usize, Array.set_val_eq, hne] using hs
    exact ⟨current, storedValue, hsOld, hj⟩

private theorem replacePresentG_updated_empty_position {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    Array.index_usize (replacePresentG t shard slots i key new).shards shard =
        ok (some (slots.set i (some (key, new)))) ∧
      Array.index_usize (slots.set i (some (key, new))) i = ok (some (key, new)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using physical_outer_bound t shard (some slots) houter
  have hi : i.val < slots.val.length := by
    simpa [slots.property] using physical_inner_bound slots i none hslot
  constructor
  · apply physical_index_of_getElem?
    simp [replacePresentG, Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound]
  · apply physical_index_of_getElem?
    simp [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi]

private theorem homePlacedG_replacePresent_absent {K V H : Type} {N : Usize}
    (HomeInst : table.Home H K) (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (hhome : HomePlacedG HomeInst t)
    (hroute : table.Table.shard_index HomeInst t key = ok shard)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    HomePlacedG HomeInst (replacePresentG t shard slots i key new) := by
  intro s j current stored storedValue hs hj
  rcases replacePresentG_entry_empty_cases t shard i slots key new houter hslot
    s j current stored storedValue hs hj with hnew | ⟨oldSlots, oldValue, hsOld, hjOld⟩
  · rcases hnew with ⟨rfl, rfl, rfl, _⟩
    exact hroute
  · change table.Table.shard_index HomeInst t stored = ok s
    exact hhome s j oldSlots stored oldValue hsOld hjOld

private theorem globalKeysDistinctG_replacePresent_absent {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (hdistinct : GlobalKeysDistinctG t) (habsent : KeyAbsentG t key)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    GlobalKeysDistinctG (replacePresentG t shard slots i key new) := by
  intro leftShard leftSlot rightShard rightSlot leftSlots rightSlots stored left right
    hleftOuter hleftSlot hrightOuter hrightSlot
  rcases replacePresentG_entry_empty_cases t shard i slots key new houter hslot leftShard
    leftSlot leftSlots stored left hleftOuter hleftSlot with hleft | ⟨leftOldSlots, leftOld,
      hleftOuterOld, hleftSlotOld⟩
  · rcases replacePresentG_entry_empty_cases t shard i slots key new houter hslot rightShard
      rightSlot rightSlots stored right hrightOuter hrightSlot with hright | ⟨rightOldSlots,
        rightOld, hrightOuterOld, hrightSlotOld⟩
    · exact ⟨hleft.1.trans hright.1.symm, hleft.2.1.trans hright.2.1.symm⟩
    · have hrightKey : Array.index_usize rightOldSlots rightSlot = ok (some (key, rightOld)) :=
        by simpa [hleft.2.2.1] using hrightSlotOld
      exact (habsent rightShard rightSlot rightOldSlots rightOld hrightOuterOld hrightKey).elim
  · rcases replacePresentG_entry_empty_cases t shard i slots key new houter hslot rightShard
      rightSlot rightSlots stored right hrightOuter hrightSlot with hright | ⟨rightOldSlots,
        rightOld, hrightOuterOld, hrightSlotOld⟩
    · have hleftKey : Array.index_usize leftOldSlots leftSlot = ok (some (key, leftOld)) :=
        by simpa [hright.2.2.1] using hleftSlotOld
      exact (habsent leftShard leftSlot leftOldSlots leftOld hleftOuterOld hleftKey).elim
    · exact hdistinct leftShard leftSlot rightShard rightSlot leftOldSlots rightOldSlots stored
        leftOld rightOld hleftOuterOld hleftSlotOld hrightOuterOld hrightSlotOld

private theorem keyAbsentG_replacePresent_other {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) (query : K)
    (habsent : KeyAbsentG t query) (hquery : query ≠ key) :
    KeyAbsentG (replacePresentG t shard slots i key new) query := by
  intro s j current value hs hj
  rcases replacePresentG_entry_empty_cases t shard i slots key new houter hslot
    s j current query value hs hj with hnew | ⟨oldSlots, oldValue, hsOld, hjOld⟩
  · exact hquery hnew.2.2.1
  · exact habsent s j oldSlots oldValue hsOld hjOld

private theorem replacePresentG_preserves_other_key_absent {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none)
    (otherKey : K) (s j : Usize)
    (current : Array (Option (K × V)) N) (currentValue : V)
    (hs : Array.index_usize t.shards s = ok (some current))
    (hj : Array.index_usize current j = ok (some (otherKey, currentValue))) :
    ∃ updatedSlots,
      Array.index_usize (replacePresentG t shard slots i key new).shards s =
        ok (some updatedSlots) ∧
      Array.index_usize updatedSlots j = ok (some (otherKey, currentValue)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using physical_outer_bound t shard (some slots) houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots := by
      exact Option.some.inj (Aeneas.Std.Result.ok_injective (hs.symm.trans houter))
    subst current
    have hji : j ≠ i := by
      intro heq
      subst j
      have hcontr := Aeneas.Std.Result.ok_injective (hj.symm.trans hslot)
      simp at hcontr
    have hjiVal : j.val ≠ i.val := fun h => hji (UScalar.eq_of_val_eq h)
    refine ⟨slots.set i (some (key, new)), ?_, ?_⟩
    · apply physical_index_of_getElem?
      simp [replacePresentG, Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound]
    · simpa [Array.index_usize, Array.set_val_eq, hjiVal] using hj
  · have hneShard : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    refine ⟨current, ?_, hj⟩
    simpa [replacePresentG, Array.index_usize, Array.set_val_eq, hneShard] using hs

/-- A fresh write into a populated vacant slot preserves the full generic physical invariant. -/
theorem physicalInvG_populated_vacant {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value : V) (shard first : Usize)
    (slots : Array (Option (K × V)) N)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h) (habsent : KeyAbsentG t key)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (_hfind : table.Table.find cmp HomeInst t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : FirstVacantG slots first) :
    PhysicalInvG cmp HomeInst
      (replacePresentG t shard slots first key (mergeFn value t.default))
      (insertWith mergeFn key value h) := by
  have hinvCopy := hinv
  obtain ⟨hhome, hdistinct, hread⟩ := hinv
  obtain ⟨hfirstBound, _, hslot⟩ := hfirst
  have hhome' := homePlacedG_replacePresent_absent HomeInst t shard first slots key
    (mergeFn value t.default) hhome hshard houter hslot
  have hdistinct' := globalKeysDistinctG_replacePresent_absent t shard first slots key
    (mergeFn value t.default) hdistinct habsent houter hslot
  refine ⟨hhome', hdistinct', ?_⟩
  let updated := replacePresentG t shard slots first key (mergeFn value t.default)
  have hupdated := replacePresentG_updated_empty_position t shard first slots key
    (mergeFn value t.default) houter hslot
  intro query
  by_cases hquery : query = key
  · subst query
    have holdGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst t merge
      homeFn mergeFn key laws habsent
    have hdefault : t.default = h key :=
      Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread key))
    have hnewGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst updated
      merge homeFn mergeFn key shard first (slots.set first
        (some (key, mergeFn value t.default))) (mergeFn value t.default) laws hhome'
        hdistinct' hupdated.1 hupdated.2
    simpa [updated, insertWith, hdefault] using hnewGet
  · obtain ⟨oldShard, _, hpresent | habsentOld⟩ :=
      table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst t merge homeFn mergeFn
        h query laws hinvCopy
    · obtain ⟨oldSlots, oldIndex, oldValue, _, holdOuter, holdSlot, _⟩ := hpresent
      obtain ⟨updatedSlots, hupdatedOuter, hupdatedSlot⟩ :=
        replacePresentG_preserves_other_key_absent t shard first slots key
          (mergeFn value t.default) houter hslot query oldShard oldIndex oldSlots oldValue
          holdOuter holdSlot
      have holdGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst t merge
        homeFn mergeFn query oldShard oldIndex oldSlots oldValue laws hhome hdistinct
        holdOuter holdSlot
      have holdEq : oldValue = h query :=
        Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread query))
      have hnewGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst updated
        merge homeFn mergeFn query oldShard oldIndex updatedSlots oldValue laws hhome' hdistinct'
        hupdatedOuter hupdatedSlot
      simpa [updated, insertWith, hquery, holdEq] using hnewGet
    · obtain ⟨habsentOld, _, _⟩ := habsentOld
      have habsent' := keyAbsentG_replacePresent_other t shard first slots key
        (mergeFn value t.default) houter hslot query habsentOld hquery
      have holdGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst t merge
        homeFn mergeFn query laws habsentOld
      have hdefault : t.default = h query :=
        Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread query))
      have hnewGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst updated merge
        homeFn mergeFn query laws habsent'
      calc
        table.Table.get cmp HomeInst updated query = ok updated.default := hnewGet
        _ = ok t.default := rfl
        _ = ok (h query) := congrArg ok hdefault
        _ = ok (insertWith mergeFn key value h query) := by simp [insertWith, hquery]

private theorem insertEmptyShardG_entry_cases {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V) (hN : 0#usize < N)
    (houter : Array.index_usize t.shards shard = ok none)
    (s j : Usize) (current : Array (Option (K × V)) N) (stored : K) (storedValue : V)
    (hs : Array.index_usize (insertEmptyShardG t shard key new).shards s =
      ok (some current))
    (hj : Array.index_usize current j = ok (some (stored, storedValue))) :
    (s = shard ∧ j = 0#usize ∧ stored = key ∧ storedValue = new) ∨
      ∃ oldSlots oldValue,
        Array.index_usize t.shards s = ok (some oldSlots) ∧
        Array.index_usize oldSlots j = ok (some (stored, oldValue)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using physical_outer_bound t shard none houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = (Array.repeat N none).set 0#usize (some (key, new)) := by
      have hs' := physical_getElem?_of_index
        (insertEmptyShardG t shard key new).shards shard (some current) hs
      simp only [insertEmptyShardG] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    have hjN : j.val < N.val := by
      exact physical_inner_bound ((Array.repeat N none).set 0#usize (some (key, new))) j
        (some (stored, storedValue)) hj
    have hj0 : j = 0#usize := by
      by_contra hjne
      have hjneVal : j.val ≠ 0 := fun he => hjne (UScalar.eq_of_val_eq he)
      have hg := physical_getElem?_of_index
        ((Array.repeat N none).set 0#usize (some (key, new))) j
        (some (stored, storedValue)) hj
      rw [Array.set_val_eq] at hg
      simp [Array.repeat_val, hjN, hjneVal] at hg
    subst j
    have hg := physical_getElem?_of_index
      ((Array.repeat N none).set 0#usize (some (key, new))) 0#usize
      (some (stored, storedValue)) hj
    rw [Array.set_val_eq, Array.repeat_val,
      List.getElem?_set_self (by simpa using hN)] at hg
    have hpair : (stored, storedValue) = (key, new) :=
      Option.some.inj (Option.some.inj hg.symm)
    exact Or.inl ⟨rfl, rfl, congrArg Prod.fst hpair, congrArg Prod.snd hpair⟩
  · right
    have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [insertEmptyShardG, Array.index_usize, Array.set_val_eq, hne] using hs
    exact ⟨current, storedValue, hsOld, hj⟩

private theorem insertEmptyShardG_updated_position {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V) (hN : 0#usize < N)
    (houter : Array.index_usize t.shards shard = ok none) :
    Array.index_usize (insertEmptyShardG t shard key new).shards shard =
        ok (some ((Array.repeat N none).set 0#usize (some (key, new)))) ∧
      Array.index_usize ((Array.repeat N none).set 0#usize (some (key, new))) 0#usize =
        ok (some (key, new)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using physical_outer_bound t shard none houter
  constructor
  · apply physical_index_of_getElem?
    simp [insertEmptyShardG, Array.set_val_eq,
      List.getElem?_set_eq_of_lt _ hshardBound]
  · apply physical_index_of_getElem?
    have hzero : (0#usize).val < (Array.repeat N (none : Option (K × V))).val.length := by
      simpa [Array.repeat_val] using hN
    rw [Array.set_val_eq, List.getElem?_set_self hzero]

private theorem homePlacedG_insertEmptyShard_absent {K V H : Type} {N : Usize}
    (HomeInst : table.Home H K) (t : table.Table K V H N) (shard : Usize)
    (key : K) (new : V) (hN : 0#usize < N) (hhome : HomePlacedG HomeInst t)
    (hroute : table.Table.shard_index HomeInst t key = ok shard)
    (houter : Array.index_usize t.shards shard = ok none) :
    HomePlacedG HomeInst (insertEmptyShardG t shard key new) := by
  intro s j current stored storedValue hs hj
  rcases insertEmptyShardG_entry_cases t shard key new hN houter s j current stored storedValue
    hs hj with hnew | ⟨oldSlots, oldValue, hsOld, hjOld⟩
  · rcases hnew with ⟨rfl, rfl, rfl, _⟩
    exact hroute
  · change table.Table.shard_index HomeInst t stored = ok s
    exact hhome s j oldSlots stored oldValue hsOld hjOld

private theorem globalKeysDistinctG_insertEmptyShard_absent {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V) (hN : 0#usize < N)
    (hdistinct : GlobalKeysDistinctG t) (habsent : KeyAbsentG t key)
    (houter : Array.index_usize t.shards shard = ok none) :
    GlobalKeysDistinctG (insertEmptyShardG t shard key new) := by
  intro leftShard leftSlot rightShard rightSlot leftSlots rightSlots stored left right
    hleftOuter hleftSlot hrightOuter hrightSlot
  rcases insertEmptyShardG_entry_cases t shard key new hN houter leftShard leftSlot leftSlots
    stored left hleftOuter hleftSlot with hleft | ⟨leftOldSlots, leftOld, hleftOuterOld,
      hleftSlotOld⟩
  · rcases insertEmptyShardG_entry_cases t shard key new hN houter rightShard rightSlot rightSlots
      stored right hrightOuter hrightSlot with hright | ⟨rightOldSlots, rightOld,
        hrightOuterOld, hrightSlotOld⟩
    · exact ⟨hleft.1.trans hright.1.symm, hleft.2.1.trans hright.2.1.symm⟩
    · have hrightKey : Array.index_usize rightOldSlots rightSlot = ok (some (key, rightOld)) :=
        by simpa [hleft.2.2.1] using hrightSlotOld
      exact (habsent rightShard rightSlot rightOldSlots rightOld hrightOuterOld hrightKey).elim
  · rcases insertEmptyShardG_entry_cases t shard key new hN houter rightShard rightSlot rightSlots
      stored right hrightOuter hrightSlot with hright | ⟨rightOldSlots, rightOld,
        hrightOuterOld, hrightSlotOld⟩
    · have hleftKey : Array.index_usize leftOldSlots leftSlot = ok (some (key, leftOld)) :=
        by simpa [hright.2.2.1] using hleftSlotOld
      exact (habsent leftShard leftSlot leftOldSlots leftOld hleftOuterOld hleftKey).elim
    · exact hdistinct leftShard leftSlot rightShard rightSlot leftOldSlots rightOldSlots stored
        leftOld rightOld hleftOuterOld hleftSlotOld hrightOuterOld hrightSlotOld

private theorem keyAbsentG_insertEmptyShard_other {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V) (hN : 0#usize < N)
    (houter : Array.index_usize t.shards shard = ok none) (query : K)
    (habsent : KeyAbsentG t query) (hquery : query ≠ key) :
    KeyAbsentG (insertEmptyShardG t shard key new) query := by
  intro s j current value hs hj
  rcases insertEmptyShardG_entry_cases t shard key new hN houter s j current query value hs hj
    with hnew | ⟨oldSlots, oldValue, hsOld, hjOld⟩
  · exact hquery hnew.2.2.1
  · exact habsent s j oldSlots oldValue hsOld hjOld

private theorem insertEmptyShardG_preserves_other_key_absent {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V)
    (houter : Array.index_usize t.shards shard = ok none)
    (otherKey : K) (s j : Usize)
    (current : Array (Option (K × V)) N) (currentValue : V)
    (hs : Array.index_usize t.shards s = ok (some current))
    (hj : Array.index_usize current j = ok (some (otherKey, currentValue))) :
    ∃ updatedSlots,
      Array.index_usize (insertEmptyShardG t shard key new).shards s = ok (some updatedSlots) ∧
      Array.index_usize updatedSlots j = ok (some (otherKey, currentValue)) := by
  have hneShard : shard.val ≠ s.val := by
    intro heq
    have hsNone : Array.index_usize t.shards shard = ok (some current) := by
      simpa [UScalar.eq_of_val_eq heq] using hs
    rw [houter] at hsNone
    simp at hsNone
  have hsne : s.val ≠ shard.val := Ne.symm hneShard
  refine ⟨current, ?_, hj⟩
  simpa [insertEmptyShardG, Array.index_usize, Array.set_val_eq, hsne] using hs

/-- A fresh write into an untouched positive-capacity shard preserves the full generic invariant. -/
theorem physicalInvG_empty_shard {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value : V) (shard : Usize)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h) (habsent : KeyAbsentG t key)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (_hfind : table.Table.find cmp HomeInst t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok none) (hN : 0#usize < N) :
    PhysicalInvG cmp HomeInst (insertEmptyShardG t shard key (mergeFn value t.default))
      (insertWith mergeFn key value h) := by
  have hinvCopy := hinv
  obtain ⟨hhome, hdistinct, hread⟩ := hinv
  have hhome' := homePlacedG_insertEmptyShard_absent HomeInst t shard key
    (mergeFn value t.default) hN hhome hshard houter
  have hdistinct' := globalKeysDistinctG_insertEmptyShard_absent t shard key
    (mergeFn value t.default) hN hdistinct habsent houter
  refine ⟨hhome', hdistinct', ?_⟩
  let updated := insertEmptyShardG t shard key (mergeFn value t.default)
  have hupdated := insertEmptyShardG_updated_position t shard key (mergeFn value t.default) hN
    houter
  intro query
  by_cases hquery : query = key
  · subst query
    have holdGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst t merge
      homeFn mergeFn key laws habsent
    have hdefault : t.default = h key :=
      Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread key))
    have hnewGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst updated
      merge homeFn mergeFn key shard 0#usize ((Array.repeat N none).set 0#usize
        (some (key, mergeFn value t.default))) (mergeFn value t.default) laws hhome' hdistinct'
        hupdated.1 hupdated.2
    simpa [updated, insertWith, hdefault] using hnewGet
  · obtain ⟨oldShard, _, hpresent | habsentOld⟩ :=
      table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst t merge homeFn mergeFn
        h query laws hinvCopy
    · obtain ⟨oldSlots, oldIndex, oldValue, _, holdOuter, holdSlot, _⟩ := hpresent
      obtain ⟨updatedSlots, hupdatedOuter, hupdatedSlot⟩ :=
        insertEmptyShardG_preserves_other_key_absent t shard key (mergeFn value t.default)
          houter query oldShard oldIndex oldSlots oldValue holdOuter holdSlot
      have holdGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst t merge
        homeFn mergeFn query oldShard oldIndex oldSlots oldValue laws hhome hdistinct
        holdOuter holdSlot
      have holdEq : oldValue = h query :=
        Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread query))
      have hnewGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst updated
        merge homeFn mergeFn query oldShard oldIndex updatedSlots oldValue laws hhome' hdistinct'
        hupdatedOuter hupdatedSlot
      simpa [updated, insertWith, hquery, holdEq] using hnewGet
    · obtain ⟨habsentOld, _, _⟩ := habsentOld
      have habsent' := keyAbsentG_insertEmptyShard_other t shard key (mergeFn value t.default)
        hN houter query habsentOld hquery
      have holdGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst t merge
        homeFn mergeFn query laws habsentOld
      have hdefault : t.default = h query :=
        Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread query))
      have hnewGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst updated merge
        homeFn mergeFn query laws habsent'
      calc
        table.Table.get cmp HomeInst updated query = ok updated.default := hnewGet
        _ = ok t.default := rfl
        _ = ok (h query) := congrArg ok hdefault
        _ = ok (insertWith mergeFn key value h query) := by simp [insertWith, hquery]

end Apeiron.TableRefinement
