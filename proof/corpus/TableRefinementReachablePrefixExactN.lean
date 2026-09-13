import TableRefinementBuildGeneric

/-! # Reachable-prefix exact coverage at general `N`

The lower layer of the general-`N` admission development: physical invariant preservation and the
bidirectional `ReachablePrefixExactN` key-set characterization of successful generated prefixes.
`TableRefinementAdmissionGeneric` imports this module and adds the admission/refusal headlines. -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

private theorem getElem?_eq_some_of_index_usize_eq_okN {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all
private theorem index_usize_eq_ok_of_getElem?_eq_someN {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : a.val[i.val]? = some x) :
    Array.index_usize a i = ok x := by
  unfold Array.index_usize
  rw [Array.getElem?_Usize_eq, h]
private theorem getElem?_set_eq_selfN {α : Type} (xs : List α) (i : Nat) (x : α)
    (hi : i < xs.length) : (xs.set i x)[i]? = some x := by
  rw [List.getElem?_eq_getElem (by simpa), List.getElem_set_self]

private theorem getElem?_set_eq_of_neN {α : Type} (xs : List α) (i j : Nat) (x : α)
    (hij : j ≠ i) : (xs.set i x)[j]? = xs[j]? := by
  simp [hij]

/-- Word shard routing is independent of the table, even across different array lengths. -/
private theorem word_shard_index_independent_N {N N' : Usize} (t : WordTableN N)
    (t' : WordTableN N') (key : I64) :
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t' key := by
  simp [table.Table.shard_index, table.WordHome.Insts.Apeiron_wordTableHomeI64.home]
/-! ## Generic-`N` invariant preservation -/
private theorem occupiedPrefix_set_presentN {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (i : Usize) (key old value : I64)
    (hp : OccupiedPrefixN N slots.val)
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    OccupiedPrefixN N (slots.set i (some (key, value))).val := by
  obtain ⟨k, hk, hoccupied, hvacant⟩ := hp
  have hslot' := getElem?_eq_some_of_index_usize_eq_okN slots i (some (key, old)) hslot
  have hi : i.val < N.val := by
    have hiLen : i.val < slots.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hslot'
      simp at hslot'
    simpa [slots.property] using hiLen
  have hbefore : i.val < k := by
    by_contra h
    have hsuffix := hvacant i.val (Nat.le_of_not_gt h) hi
    rw [hsuffix] at hslot'
    simp at hslot'
  refine ⟨k, hk, ?_, ?_⟩
  · intro j hj hjk
    by_cases hji : j = i.val
    · subst j
      simp [EmptyAt, Array.set_val_eq, hi]
    · simpa [EmptyAt, Array.set_val_eq, hji] using hoccupied j hj hjk
  · intro j hkj hj
    have hne : j ≠ i.val := by omega
    simpa [Array.set_val_eq, hne] using hvacant j hkj hj

private theorem occupiedPrefix_set_vacantN {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (k : Usize) (key value : I64)
    (hoccupied : NoEmptyIn slots.val 0 k.val)
    (hvacant : ∀ j, k.val ≤ j → j < N.val → slots.val[j]? = some none)
    (hk : k < N) :
    OccupiedPrefixN N (slots.set k (some (key, value))).val := by
  refine ⟨k.val + 1, Nat.succ_le_iff.mpr hk, ?_, ?_⟩
  · apply noEmptyIn_succ
    · intro j hj hjk
      have hne : j ≠ k.val := by omega
      simpa [EmptyAt, Array.set_val_eq, hne] using hoccupied j hj hjk
    · have hkLen : k.val < slots.val.length := by simpa [slots.property] using hk
      intro hempty
      have hnew : (slots.val.set k.val (some (key, value)))[k.val]? =
          some (some (key, value)) := by
        have hb : k.val < (slots.val.set k.val (some (key, value))).length := by
          simpa using hkLen
        rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
      rw [EmptyAt, Array.set_val_eq, hnew] at hempty
      simp at hempty
  · intro j hnext hj
    have hne : j ≠ k.val := by omega
    simpa [Array.set_val_eq, hne] using hvacant j (by omega) hj

private theorem shardKeysDistinct_set_presentN {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (i : Usize) (key old value : I64)
    (hdistinct : ShardKeysDistinctN N slots.val)
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    ShardKeysDistinctN N (slots.set i (some (key, value))).val := by
  have hold := getElem?_eq_some_of_index_usize_eq_okN slots i (some (key, old)) hslot
  have hi : i.val < slots.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hold
    simp at hold
  intro stored a b left right ha hb hleft hright
  by_cases hai : a = i.val
  · subst a
    have hnew := getElem?_set_eq_selfN slots.val i.val (some (key, value)) hi
    rw [Array.set_val_eq, hnew] at hleft
    have hstored : stored = key :=
      (Prod.mk.inj (Option.some.inj (Option.some.inj hleft.symm))).1
    by_cases hbi : b = i.val
    · exact hbi.symm
    · have hrightOld : slots.val[b]? = some (some (stored, right)) := by
        simpa [Array.set_val_eq, hbi] using hright
      exact hdistinct key i.val b old right (by simpa [slots.property] using hi) hb hold
        (by simpa [hstored] using hrightOld)
  · have hleftOld : slots.val[a]? = some (some (stored, left)) := by
      simpa [Array.set_val_eq, hai] using hleft
    by_cases hbi : b = i.val
    · subst b
      have hnew := getElem?_set_eq_selfN slots.val i.val (some (key, value)) hi
      rw [Array.set_val_eq, hnew] at hright
      have hstored : stored = key :=
        (Prod.mk.inj (Option.some.inj (Option.some.inj hright.symm))).1
      exact hdistinct key a i.val left old ha (by simpa [slots.property] using hi)
        (by simpa [hstored] using hleftOld) hold
    · have hrightOld : slots.val[b]? = some (some (stored, right)) := by
        simpa [Array.set_val_eq, hbi] using hright
      exact hdistinct stored a b left right ha hb hleftOld hrightOld

private theorem shardKeysDistinct_set_vacantN {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (first : Usize) (key value : I64)
    (hp : OccupiedPrefixN N slots.val) (hdistinct : ShardKeysDistinctN N slots.val)
    (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    ShardKeysDistinctN N (slots.set first (some (key, value))).val := by
  obtain ⟨k, hk, hoccupied, hvacant⟩ := hp
  have hfirstSlot := getElem?_eq_some_of_index_usize_eq_okN slots first none hslot
  have hfirst : first.val < N.val := by
    have hlen : first.val < slots.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hfirstSlot
      simp at hfirstSlot
    simpa [slots.property] using hlen
  have hkfirst : k = first.val := by
    have hkLe : k ≤ first.val := by
      by_contra h
      exact hoccupied first.val (Nat.zero_le _) (by omega)
        (by simpa [EmptyAt] using hfirstSlot)
    have hfirstLe : first.val ≤ k := by
      by_contra h
      exact hempty k (Nat.zero_le _) (by omega)
        (by simpa [EmptyAt] using hvacant k le_rfl (by omega))
    exact Nat.le_antisymm hkLe hfirstLe
  subst k
  have hlen : first.val < slots.val.length := by simpa [slots.property] using hfirst
  intro stored a b left right ha hb hleft hright
  by_cases hai : a = first.val
  · subst a
    have hnew := getElem?_set_eq_selfN slots.val first.val (some (key, value)) hlen
    rw [Array.set_val_eq, hnew] at hleft
    have hstored : stored = key :=
      (Prod.mk.inj (Option.some.inj (Option.some.inj hleft.symm))).1
    by_cases hbi : b = first.val
    · exact hbi.symm
    · have hrightOld : slots.val[b]? = some (some (key, right)) := by
        simpa [Array.set_val_eq, hbi, hstored] using hright
      by_cases hbBefore : b < first.val
      · exact (hnomatch b (Nat.zero_le _) hbBefore ⟨right, hrightOld⟩).elim
      · have hsuffix := hvacant b (by omega) hb
        rw [hsuffix] at hrightOld
        simp at hrightOld
  · have hleftOld : slots.val[a]? = some (some (stored, left)) := by
      simpa [Array.set_val_eq, hai] using hleft
    by_cases hbi : b = first.val
    · subst b
      have hnew := getElem?_set_eq_selfN slots.val first.val (some (key, value)) hlen
      rw [Array.set_val_eq, hnew] at hright
      have hstored : stored = key :=
        (Prod.mk.inj (Option.some.inj (Option.some.inj hright.symm))).1
      by_cases haBefore : a < first.val
      · exact (hnomatch a (Nat.zero_le _) haBefore
          ⟨left, by simpa [hstored] using hleftOld⟩).elim
      · have hsuffix := hvacant a (by omega) ha
        rw [hsuffix] at hleftOld
        simp at hleftOld
    · have hrightOld : slots.val[b]? = some (some (stored, right)) := by
        simpa [Array.set_val_eq, hbi] using hright
      exact hdistinct stored a b left right ha hb hleftOld hrightOld

private theorem tableInv_replacePresentN {N : Usize}
    (t : WordTableN N) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) N) (key old value : I64)
    (hinv : TableInvN t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    TableInvN (replacePresent t shard slots i key value) := by
  unfold TableInvN at hinv ⊢
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_okN t.shards shard (some slots) houter
  have hs : shard.val < 256 := by
    have hsLen : shard.val < t.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
      simp at houter'
    simpa [t.shards.property] using hsLen
  by_cases hss : s = shard.val
  · subst s
    have hslots : current = slots.set i (some (key, value)) := by
      simp [replacePresent, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    apply occupiedPrefix_set_presentN slots i key old value
    · apply hinv shard.val slots
      exact houter'
    · exact hslot
  · apply hinv s current
    simpa [replacePresent, Array.set_val_eq, hss] using hcurrent

private theorem tableInv_insertVacantN {N : Usize}
    (t : WordTableN N) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) N) (key value : I64)
    (hinv : TableInvN t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < N) (hempty : NoEmptyIn slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    TableInvN (insertVacant t shard slots first key value) := by
  unfold TableInvN at hinv ⊢
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_okN t.shards shard (some slots) houter
  have hs : shard.val < 256 := by
    have hsLen : shard.val < t.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
      simp at houter'
    simpa [t.shards.property] using hsLen
  by_cases hss : s = shard.val
  · subst s
    have hslots : current = slots.set first (some (key, value)) := by
      simp [insertVacant, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    obtain ⟨k, hk, hoccupied, hvacant⟩ := hinv shard.val slots houter'
    have hslot' := getElem?_eq_some_of_index_usize_eq_okN slots first none hslot
    have hfirstNat : first.val < N.val := hfirst
    have hkfirst : k = first.val := by
      have hkLe : k ≤ first.val := by
        by_contra h
        exact hoccupied first.val (Nat.zero_le _) (by omega) (by simpa [EmptyAt] using hslot')
      have hfirstLe : first.val ≤ k := by
        by_contra h
        exact hempty k (Nat.zero_le _) (by omega)
          (by simpa [EmptyAt] using hvacant k le_rfl (by omega))
      exact Nat.le_antisymm hkLe hfirstLe
    subst k
    exact occupiedPrefix_set_vacantN slots first key value hoccupied hvacant hfirst
  · apply hinv s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem tableInv_insertEmptyShardN {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (shard : Usize) (key value : I64)
    (hinv : TableInvN t) (hshard : shard < 256#usize) :
    TableInvN (insertVacant t shard (Array.repeat N none) 0#usize key value) := by
  unfold TableInvN at hinv ⊢
  intro s current hcurrent
  by_cases hss : s = shard.val
  · subst s
    have hs : shard.val < 256 := hshard
    have hcurrent' : current = (Array.repeat N none).set 0#usize (some (key, value)) := by
      simp [insertVacant, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    refine ⟨1, ?_, ?_, ?_⟩
    · exact Nat.succ_le_iff.mpr (by simpa using hN)
    · intro j _ hj
      have hj0 : j = 0 := by omega
      subst j
      intro hempty
      rw [EmptyAt, Array.set_val_eq] at hempty
      rw [Array.repeat_val] at hempty
      change ((List.replicate N.val (none : Option (I64 × I64))).set 0
        (some (key, value)))[0]? = some none at hempty
      have hb : 0 < (List.replicate N.val (none : Option (I64 × I64))).length := by
        simpa using (show 0 < N.val from hN)
      rw [List.getElem?_set_self hb] at hempty
      simp at hempty
    · intro j hj hjN
      have hj0 : j ≠ 0 := by omega
      have hjLen : j < (List.replicate N.val (none : Option (I64 × I64))).length := by
        simpa using hjN
      rw [Array.set_val_eq, Array.repeat_val]
      rw [List.getElem?_set_of_lt _ _ hjLen]
      change (if 0 = j then some (some (key, value)) else
        (List.replicate N.val (none : Option (I64 × I64)))[j]?) = some none
      rw [if_neg (Ne.symm hj0)]
      rw [List.getElem?_eq_getElem hjLen, List.getElem_replicate]
  · apply hinv s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem tableInv2_replacePresentN {N : Usize}
    (t : WordTableN N) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) N) (key old value : I64)
    (hinv : TableInv2N t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    TableInv2N (replacePresent t shard slots i key value) := by
  refine ⟨tableInv_replacePresentN t shard i slots key old value hinv.1 houter hslot, ?_⟩
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_okN t.shards shard (some slots) houter
  have hs : shard.val < 256 := by
    have hsLen : shard.val < t.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
      simp at houter'
    simpa [t.shards.property] using hsLen
  by_cases hss : s = shard.val
  · subst s
    have hslots : current = slots.set i (some (key, value)) := by
      simp [replacePresent, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    exact shardKeysDistinct_set_presentN slots i key old value
      (hinv.2 shard.val slots houter') hslot
  · apply hinv.2 s current
    simpa [replacePresent, Array.set_val_eq, hss] using hcurrent

private theorem tableInv2_insertVacantN {N : Usize}
    (t : WordTableN N) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) N) (key value : I64)
    (hinv : TableInv2N t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < N) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    TableInv2N (insertVacant t shard slots first key value) := by
  refine ⟨tableInv_insertVacantN t shard first slots key value hinv.1 houter hfirst hempty
    hslot, ?_⟩
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_okN t.shards shard (some slots) houter
  have hs : shard.val < 256 := by
    have hsLen : shard.val < t.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
      simp at houter'
    simpa [t.shards.property] using hsLen
  by_cases hss : s = shard.val
  · subst s
    have hslots : current = slots.set first (some (key, value)) := by
      simp [insertVacant, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    exact shardKeysDistinct_set_vacantN slots first key value
      (hinv.1 shard.val slots houter') (hinv.2 shard.val slots houter') hempty hnomatch hslot
  · apply hinv.2 s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem tableInv2_insertEmptyShardN {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (shard : Usize) (key value : I64)
    (hinv : TableInv2N t) (hshard : shard < 256#usize) :
    TableInv2N (insertVacant t shard (Array.repeat N none) 0#usize key value) := by
  refine ⟨tableInv_insertEmptyShardN hN t shard key value hinv.1 hshard, ?_⟩
  intro s current hcurrent
  by_cases hss : s = shard.val
  · subst s
    have hs : shard.val < 256 := hshard
    have hcurrent' : current = (Array.repeat N none).set 0#usize
        (some (key, value)) := by
      simp [insertVacant, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    apply shardKeysDistinct_set_vacantN (Array.repeat N none) 0#usize key value
    · refine ⟨0, Nat.zero_le _, ?_, ?_⟩
      · intro j _ hj
        omega
      · intro j _ hj
        rw [Array.repeat_val]
        have hjLen : j < (List.replicate N.val (none : Option (I64 × I64))).length := by
          simpa using hj
        rw [List.getElem?_eq_getElem hjLen, List.getElem_replicate]
    · intro stored i j left right hi hj hleft
      rw [Array.repeat_val] at hleft
      rw [List.getElem?_eq_getElem (by simpa using hi), List.getElem_replicate] at hleft
      simp at hleft
    · simp [NoEmptyIn]
    · simp [NoMatchIn]
    · apply index_usize_eq_ok_of_getElem?_eq_someN
      rw [Array.repeat_val]
      have hb : (0#usize).val <
          (List.replicate N.val (none : Option (I64 × I64))).length := by
        simpa using hN
      rw [List.getElem?_eq_getElem hb, List.getElem_replicate]
  · apply hinv.2 s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

/-- Physical no-duplicate preservation for every generic-`N` generated insertion branch. -/
theorem tableInv2N_insert_with {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (key value : I64) (hinv : TableInv2N t) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => match result.1 with
        | core.result.Result.Ok _ => TableInv2N result.2
        | core.result.Result.Err table.StorageError.ShardFull => result.2 = t ∧ TableInv2N result.2
  ⦄ := by
  let locate := generatedLocator t
  have hroute : GeneratedRoute t key (locate key) := generatedLocator_routes t key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, i⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      have hmouter : Array.index_mut_usize t.shards shard =
          ok (some slots, t.shards.set shard) := by
        simp [Array.index_mut_usize, houter]
      have hminner : Array.index_mut_usize slots i =
          ok (some (key, old), slots.set i) := by
        simp [Array.index_mut_usize, hslot]
      unfold table.Table.insert_with
      rw [hshard]
      simp only [bind_tc_ok]
      rw [hfind]
      simp [hmouter, hminner, table.WordMerge.Insts.Apeiron_wordTableMergeI64,
        table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge]
      exact tableInv2_replacePresentN t shard i slots key old
        (core.num.I64.wrapping_add value old) hinv houter hslot
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      obtain ⟨routedShard, hroutedShard, hshardBound⟩ := word_shard_index_spec t key
      have hshardEq : routedShard = shard :=
        Aeneas.Std.Result.ok_injective (hroutedShard.symm.trans hshard)
      subst routedShard
      have hshardLen : shard.val < t.shards.val.length := by
        rw [t.shards.property]
        exact hshardBound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec t.shards shard hshardLen)
      cases outer with
      | none =>
          rw [insert_with_empty_shard_exact t key value shard hshard hfind houter
            hN hshardBound]
          simp only [Aeneas.Std.WP.spec_ok]
          exact tableInv2_insertEmptyShardN hN t shard key
            (core.num.I64.wrapping_add value t.default) hinv hshardBound
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_okN t.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hinv.1 shard.val slots houterGet
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
          have hnomatch : NoMatchIn key slots.val 0 N.val := by
            simpa [FindPostN] using hpost
          by_cases hkFull : k = N.val
          · subst k
            rw [insert_with_full_exactN hN t key value shard slots hshard hfind houter hoccupied]
            simp [hinv]
          · have hkLt : k < N.val := by omega
            let first : Usize := ⟨k, by scalar_tac⟩
            have hfirst : first < N := hkLt
            have hslotGet : slots.val[first.val]? = some none := hvacant k le_rfl hkLt
            have hslot : Array.index_usize slots first = ok none :=
              index_usize_eq_ok_of_getElem?_eq_someN slots first none hslotGet
            have hnomatchPrefix : NoMatchIn key slots.val 0 first.val :=
              fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt)
            rw [insert_with_vacant_exact t key value shard first slots hshard hfind houter
              hfirst hoccupied hnomatchPrefix hslot]
            simp only [Aeneas.Std.WP.spec_ok]
            exact tableInv2_insertVacantN t shard first slots key
              (core.num.I64.wrapping_add value t.default) hinv houter hfirst hoccupied
              hnomatchPrefix hslot
/-! ## Exact reachable-prefix admission at general `N` -/
/-- Physical keys in the one state carried after a generated stream prefix are covered by that
    prefix's distinct routed keys. -/
def ReachablePrefixCoverageN {N : Usize} (kvs : Slice (I64 × I64)) (i : Nat)
    (state : WordTableN N) : Prop :=
  TableInv2N state ∧
    ∀ (shard : Usize) (slots : Array (Option (I64 × I64)) N) (k : Nat),
      Array.index_usize state.shards shard = ok (some slots) →
      k ≤ N.val → NoEmptyIn slots.val 0 k →
      ∀ (j : Nat) (key value : I64), j < k →
        slots.val[j]? = some (some (key, value)) →
        key ∈ routedStreamKeys kvs i shard

/-- The real generic-`N` empty constructor has no physical key outside the empty stream prefix. -/
theorem emptyWordTableN_reachablePrefixCoverageN {N : Usize} (default : I64)
    (kvs : Slice (I64 × I64)) :
    ReachablePrefixCoverageN kvs 0 (emptyWordTableN (N := N) default) := by
  set_option maxRecDepth 4096 in
  refine ⟨emptyWordTableN_tableInv2 default, ?_⟩
  intro shard slots k houter
  have houter' := getElem?_eq_some_of_index_usize_eq_okN
    (emptyWordTableN (N := N) default).shards shard (some slots) houter
  change (List.replicate 256
    (none : Option (Array (Option (I64 × I64)) N)))[shard.val]? =
      some (some slots) at houter'
  rw [List.getElem?_eq_getElem (by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
    simp at houter'), List.getElem_replicate] at houter'
  simp at houter'

private theorem routedStreamKeys_monoN (kvs : Slice (I64 × I64)) (i : Nat)
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

private theorem routedStreamKeys_currentN (kvs : Slice (I64 × I64)) (i : Usize)
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

private theorem present_reachablePrefixCoverageN {N : Usize} (hN : 0#usize < N)
    (kvs : Slice (I64 × I64)) (i : Nat)
    (state : WordTableN N) (key value : I64) (shard slot : Usize)
    (slots : Array (Option (I64 × I64)) N) (old : I64)
    (hcoverage : ReachablePrefixCoverageN kvs i state)
    (hi : i < (sliceEntries kvs).length)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 state shard key = ok (some slot))
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hslot : Array.index_usize slots slot = ok (some (key, old))) :
    ReachablePrefixCoverageN kvs (i + 1)
      (replacePresent state shard slots slot key
        (core.num.I64.wrapping_add value old)) := by
  have hwp := tableInv2N_insert_with hN state key value hcoverage.1
  rw [insert_with_present_exact state key value shard slot slots key old
    hshard hfind houter hslot] at hwp
  refine ⟨by simpa using hwp, ?_⟩
  intro s current k hcurrent hk hoccupied j stored storedValue hj hentry
  have houterGet := getElem?_eq_some_of_index_usize_eq_okN state.shards shard
    (some slots) houter
  have hshardBound : shard.val < state.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houterGet
    simp at houterGet
  by_cases hss : s.val = shard.val
  · have hs : s = shard := UScalar.eq_of_val_eq hss
    subst s
    have hcurrentGet := getElem?_eq_some_of_index_usize_eq_okN
      (replacePresent state shard slots slot key
        (core.num.I64.wrapping_add value old)).shards shard (some current) hcurrent
    have hcurrentEq : current = slots.set slot
        (some (key, core.num.I64.wrapping_add value old)) := by
      simp only [replacePresent] at hcurrentGet
      rw [Array.set_val_eq] at hcurrentGet
      rw [List.getElem?_set_eq_of_lt _ hshardBound] at hcurrentGet
      exact Option.some.inj (Option.some.inj hcurrentGet).symm
    subst current
    have hslotGet := getElem?_eq_some_of_index_usize_eq_okN slots slot
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
          rw [List.getElem?_eq_getElem (by simpa [slots.property] using (show slot.val < N.val by
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
    exact routedStreamKeys_monoN kvs i hi shard hold
  · have hne : shard.val ≠ s.val := Ne.symm hss
    have houterOld : Array.index_usize state.shards s = ok (some current) := by
      simpa [replacePresent, Array.index_usize, Array.set_val_eq, hne] using hcurrent
    exact routedStreamKeys_monoN kvs i hi s
      (hcoverage.2 s current k houterOld hk hoccupied j stored storedValue hj hentry)

private theorem vacant_reachablePrefixCoverageN {N : Usize} (hN : 0#usize < N)
    (kvs : Slice (I64 × I64)) (i : Usize)
    (state : WordTableN N) (key value : I64) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) N)
    (hcoverage : ReachablePrefixCoverageN kvs i.val state)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 state shard key = ok none)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hfirst : first < N) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    ReachablePrefixCoverageN kvs (i.val + 1)
      (insertVacant state shard slots first key
        (core.num.I64.wrapping_add value state.default)) := by
  have hwp := tableInv2N_insert_with hN state key value hcoverage.1
  rw [insert_with_vacant_exact state key value shard first slots hshard hfind houter
    hfirst hempty hnomatch hslot] at hwp
  refine ⟨by simpa using hwp, ?_⟩
  have hkeyRoute : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (emptyWordTable 0#i64) key = ok shard := by
    rw [← word_shard_index_independent_N state (emptyWordTable 0#i64) key]
    exact hshard
  have hkeyMember := routedStreamKeys_currentN kvs i key value hi hindex shard hkeyRoute
  intro s current k hcurrent hk hoccupied j stored storedValue hj hentry
  have houterGet := getElem?_eq_some_of_index_usize_eq_okN state.shards shard
    (some slots) houter
  have hshardBound : shard.val < state.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houterGet
    simp at houterGet
  by_cases hss : s.val = shard.val
  · have hs : s = shard := UScalar.eq_of_val_eq hss
    subst s
    have hcurrentGet := getElem?_eq_some_of_index_usize_eq_okN
      (insertVacant state shard slots first key
        (core.num.I64.wrapping_add value state.default)).shards shard (some current) hcurrent
    have hcurrentEq : current = slots.set first
        (some (key, core.num.I64.wrapping_add value state.default)) := by
      simp only [insertVacant] at hcurrentGet
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hcurrentGet
      exact Option.some.inj (Option.some.inj hcurrentGet).symm
    subst current
    have hslotGet := getElem?_eq_some_of_index_usize_eq_okN slots first none hslot
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
      exact routedStreamKeys_monoN kvs i.val hi shard
        (hcoverage.2 shard slots first.val houter (Nat.le_of_lt hfirst) hempty
          j stored storedValue hjBefore hentryOld)
  · have hne : shard.val ≠ s.val := Ne.symm hss
    have houterOld : Array.index_usize state.shards s = ok (some current) := by
      simpa [insertVacant, Array.index_usize, Array.set_val_eq, hne] using hcurrent
    exact routedStreamKeys_monoN kvs i.val hi s
      (hcoverage.2 s current k houterOld hk hoccupied j stored storedValue hj hentry)

private theorem emptyShard_reachablePrefixCoverageN {N : Usize} (hN : 0#usize < N)
    (kvs : Slice (I64 × I64)) (i : Usize)
    (state : WordTableN N) (key value : I64) (shard : Usize)
    (hcoverage : ReachablePrefixCoverageN kvs i.val state)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      state key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 state shard key = ok none)
    (houter : Array.index_usize state.shards shard = ok none)
    (hshardBound : shard < 256#usize) :
    ReachablePrefixCoverageN kvs (i.val + 1)
      (insertVacant state shard (Array.repeat N none) 0#usize key
        (core.num.I64.wrapping_add value state.default)) := by
  have hwp := tableInv2N_insert_with hN state key value hcoverage.1
  rw [insert_with_empty_shard_exact state key value shard hshard hfind houter
    hN hshardBound] at hwp
  refine ⟨by simpa using hwp, ?_⟩
  have hkeyRoute : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (emptyWordTable 0#i64) key = ok shard := by
    rw [← word_shard_index_independent_N state (emptyWordTable 0#i64) key]
    exact hshard
  have hkeyMember := routedStreamKeys_currentN kvs i key value hi hindex shard hkeyRoute
  intro s current k hcurrent hk hoccupied j stored storedValue hj hentry
  by_cases hss : s.val = shard.val
  · have hs : s = shard := UScalar.eq_of_val_eq hss
    subst s
    have hcurrentGet := getElem?_eq_some_of_index_usize_eq_okN
      (insertVacant state shard (Array.repeat N none) 0#usize key
        (core.num.I64.wrapping_add value state.default)).shards shard (some current) hcurrent
    have hcurrentEq : current = (Array.repeat N none).set 0#usize
        (some (key, core.num.I64.wrapping_add value state.default)) := by
      simp only [insertVacant] at hcurrentGet
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ (by
        rw [state.shards.property]
        exact hshardBound)] at hcurrentGet
      exact Option.some.inj (Option.some.inj hcurrentGet).symm
    subst current
    have hj0 : j = 0 := by
      by_contra hne
      have hjN : j < N.val := Nat.lt_of_lt_of_le hj hk
      have hentryOld : (List.replicate N.val (none : Option (I64 × I64)))[j]? =
          some (some (stored, storedValue)) := by
        simpa [Array.repeat_val, Array.set_val_eq, hne] using hentry
      rw [List.getElem?_eq_getElem (by simpa using hjN), List.getElem_replicate] at hentryOld
      simp at hentryOld
    subst j
    rw [Array.set_val_eq, Array.repeat_val] at hentry
    change ((List.replicate N.val (none : Option (I64 × I64))).set 0
      (some (key, core.num.I64.wrapping_add value state.default)))[0]? =
      some (some (stored, storedValue)) at hentry
    have hb : 0 < (List.replicate N.val (none : Option (I64 × I64))).length := by
      simpa using (show 0 < N.val from hN)
    rw [List.getElem?_set_self hb] at hentry
    have hstored : stored = key :=
      (Prod.mk.inj (Option.some.inj (Option.some.inj hentry))).1.symm
    simpa [hstored] using hkeyMember
  · have hne : shard.val ≠ s.val := Ne.symm hss
    have houterOld : Array.index_usize state.shards s = ok (some current) := by
      simpa [insertVacant, Array.index_usize, Array.set_val_eq, hne] using hcurrent
    exact routedStreamKeys_monoN kvs i.val hi s
      (hcoverage.2 s current k houterOld hk hoccupied j stored storedValue hj hentry)

/-- Bidirectional physical coverage for the unique states reached by successful generated prefixes
    at general `N`. -/
def ReachablePrefixExactN {N : Usize} (kvs : Slice (I64 × I64)) (i : Nat)
    (state : WordTableN N) : Prop :=
  ReachablePrefixCoverageN kvs i state ∧
    ∃ locate, Routes state locate ∧
      ∀ (shard : Usize) (key : I64),
        (∃ slot, locate key = some (shard, slot)) ↔ key ∈ routedStreamKeys kvs i shard

private theorem emptyWordTableN_routes_noneN {N : Usize} (default : I64) :
    Routes (emptyWordTableN (N := N) default) (fun _ => none) := by
  set_option maxRecDepth 4096 in
    intro key
    obtain ⟨shard, hshard, hbound⟩ := word_shard_index_spec (emptyWordTableN (N := N) default) key
    refine ⟨shard, hshard, ?_⟩
    unfold table.Table.find
    have houter : Array.index_usize (emptyWordTableN (N := N) default).shards shard = ok none := by
      unfold Array.index_usize
      rw [Array.getElem?_Usize_eq]
      have hget : (emptyWordTableN (N := N) default).shards.val[shard.val]? = some none := by
        change (List.replicate 256
          (none : Option (Array (Option (I64 × I64)) N)))[shard.val]? = some none
        rw [List.getElem?_eq_getElem (by simpa using hbound), List.getElem_replicate]
      rw [hget]
    rw [houter]
    simp

/-- The generated generic-`N` empty constructor has exactly the empty routed-key set. -/
theorem emptyWordTableN_reachablePrefixExactN {N : Usize} (default : I64)
    (kvs : Slice (I64 × I64)) : ReachablePrefixExactN kvs 0 (emptyWordTableN (N := N) default) := by
  set_option maxRecDepth 4096 in
    refine ⟨emptyWordTableN_reachablePrefixCoverageN default kvs,
      fun _ => none, emptyWordTableN_routes_noneN default, ?_⟩
    intro shard key
    simp [routedStreamKeys]

private theorem routedStreamKeys_succ_iffN (kvs : Slice (I64 × I64)) (i : Usize)
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

private theorem index_usize_repeat_zeroN {N : Usize} (hN : 0#usize < N) :
    Array.index_usize (Array.repeat N (none : Option (I64 × I64))) 0#usize = ok none := by
  apply index_usize_eq_ok_of_getElem?_eq_someN
  rw [Array.repeat_val]
  have hb : (0#usize).val < (List.replicate N.val (none : Option (I64 × I64))).length := by
    simpa using hN
  rw [List.getElem?_eq_getElem hb, List.getElem_replicate]

private theorem repeatN_find_noneN {N : Usize} (key : I64) :
    table.Table.find_slots core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      (Array.repeat N (none : Option (I64 × I64))) key = ok none := by
  obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec_generic (Array.repeat N (none : Option (I64 × I64))) key
      0#usize (Nat.zero_le _))
  cases found with
  | none => exact hfound
  | some i =>
      obtain ⟨hi, _, ⟨value, hmatch⟩, _⟩ := hpost
      rw [Array.repeat_val] at hmatch
      have hiLen : i.val < (List.replicate N.val (none : Option (I64 × I64))).length := by
        simpa [Array.repeat_val] using hi
      rw [List.getElem?_eq_getElem hiLen, List.getElem_replicate] at hmatch
      simp at hmatch

private theorem routes_insertEmptyShard_exactN {N : Usize} (hN : 0#usize < N)
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N) (key value : I64)
    (shard : Usize) (hroutes : Routes t locate)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard)
    (houter : Array.index_usize t.shards shard = ok none) (hshardBound : shard < 256#usize) :
    Routes (insertVacant t shard (Array.repeat N none) 0#usize key value)
      (insertLocator locate key shard 0#usize) := by
  intro x
  let next := insertVacant t shard (Array.repeat N none) 0#usize key value
  have hnewOuter : Array.index_usize next.shards shard =
      ok (some ((Array.repeat N none).set 0#usize (some (key, value)))) := by
    have hget : next.shards.val[shard.val]? =
        some (some ((Array.repeat N none).set 0#usize (some (key, value)))) := by
      dsimp [next, insertVacant]
      rw [Array.set_val_eq]
      have hsLen : shard.val < t.shards.val.length := by
        rw [t.shards.property]
        exact hshardBound
      exact List.getElem?_set_eq_of_lt _ hsLen
    exact index_usize_eq_ok_of_getElem?_eq_someN next.shards shard _ hget
  have hrepeatSlot : Array.index_usize (Array.repeat N
      (none : Option (I64 × I64))) 0#usize = ok none := index_usize_repeat_zeroN hN
  by_cases hx : x = key
  · subst x
    simp only [insertLocator]
    apply generatedRoute_some_of_find next key shard 0#usize
    · rw [← word_shard_index_state_independent t next key]
      exact hshard
    · unfold table.Table.find
      rw [hnewOuter]
      simp only [bind_tc_ok]
      exact find_slots_insertVacant_self (Array.repeat N none) 0#usize key value
        hN (fun j hj hjlt _ => (Nat.not_lt_of_ge hj) hjlt)
  · simp only [insertLocator, if_neg hx]
    have hroute := hroutes x
    cases hloc : locate x with
    | none =>
        rw [hloc] at hroute
        obtain ⟨s, hs, hf⟩ := hroute
        apply generatedRoute_none_of_find next x s
        · rw [← word_shard_index_state_independent t next x]
          exact hs
        · by_cases hsame : s = shard
          · subst s
            unfold table.Table.find
            rw [hnewOuter]
            simp only [bind_tc_ok]
            rw [find_slots_insertVacant_ne (Array.repeat N none) 0#usize key value x
              hrepeatSlot hx, repeatN_find_noneN]
          · have hne : shard.val ≠ s.val := fun h => hsame (UScalar.eq_of_val_eq h.symm)
            simpa [next, insertVacant, table.Table.find, Array.index_usize, Array.set_val_eq,
              hne] using hf
    | some p =>
        rcases p with ⟨s, i⟩
        rw [hloc] at hroute
        obtain ⟨oldSlots, oldValue, hs, hf, ho, hi⟩ := hroute
        have hsame : s ≠ shard := by
          intro heq
          subst s
          rw [houter] at ho
          simp at ho
        apply generatedRoute_some_of_find next x s i
        · rw [← word_shard_index_state_independent t next x]
          exact hs
        · have hne : shard.val ≠ s.val := fun h => hsame (UScalar.eq_of_val_eq h.symm)
          simpa [next, insertVacant, table.Table.find, Array.index_usize, Array.set_val_eq,
            hne] using hf

private theorem reachablePrefixCoverage_capacity_of_card_ltN {N : Usize}
    (kvs : Slice (I64 × I64)) (i : Nat) (state : WordTableN N)
    (hcoverage : ReachablePrefixCoverageN kvs i state) (shard : Usize)
    (hcard : (routedStreamKeys kvs i shard).card < N.val)
    (slots : Array (Option (I64 × I64)) N) (k : Nat)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hk : k ≤ N.val) (hoccupied : NoEmptyIn slots.val 0 k) : k < N.val := by
  have houterGet := getElem?_eq_some_of_index_usize_eq_okN state.shards shard
    (some slots) houter
  obtain ⟨keys, hkeysInjective, hkeysSlots⟩ :=
    occupiedPrefix_distinct_key_countN slots k hk hoccupied
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

private theorem insertLocator_exact_succN (kvs : Slice (I64 × I64)) (i : Usize)
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
  rw [routedStreamKeys_succ_iffN kvs i key value hi hindex shard routed hroute x]
  by_cases hx : x = key
  · subst x
    have hnot : key ∉ routedStreamKeys kvs i.val shard := by
      intro hm
      obtain ⟨slot, hslot⟩ := (hexact shard key).mpr hm
      rw [hlocate] at hslot
      simp at hslot
    simpa [insertLocator, hnot] using (@eq_comm Usize routed shard)
  · simp [insertLocator, hx, hexact shard x]

private theorem presentLocator_exact_succN (kvs : Slice (I64 × I64)) (i : Usize)
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
  rw [routedStreamKeys_succ_iffN kvs i key value hi hindex shard routed hroute x]
  constructor
  · intro h
    exact Or.inl ((hexact shard x).mp h)
  · rintro (h | ⟨rfl, rfl⟩)
    · exact (hexact shard x).mpr h
    · exact ⟨presentSlot, hlocate⟩

/-- A successful generated insertion preserves exact physical/prefix key-set equality at general
    `N`.  A repeated key succeeds even at load `N`; a fresh key requires strict capacity. -/
theorem reachablePrefixExact_insertN {N : Usize} (hN : 0#usize < N) (default : I64)
    (kvs : Slice (I64 × I64)) (i : Usize) (state : WordTableN N) (key value : I64)
    (hi : i.val < (sliceEntries kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hrep : ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)))
    (hexact : ReachablePrefixExactN kvs i.val state)
    (hadmit : ∀ routed,
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok routed →
        key ∈ routedStreamKeys kvs i.val routed ∨
          (routedStreamKeys kvs i.val routed).card < N.val) :
    ∃ next, table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 state () key value =
        ok (core.result.Result.Ok (), next) ∧
      ReadRep next (insertWith core.num.I64.wrapping_add key value
        (orderedFold default ((sliceEntries kvs).take i.val))) ∧
      ReachablePrefixExactN kvs (i.val + 1) next := by
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
        rw [← word_shard_index_independent_N state (emptyWordTable 0#i64) key]
        exact hshard
      refine ⟨next, hinsert, ?_, ?_, locate, ?_, ?_⟩
      · exact insert_with_present_readRep_preservation locate state next
          (orderedFold default ((sliceEntries kvs).take i.val)) key value shard slot slots old
          ((functionalRep_iff_readRep_of_routes locate state _ hroutes).mpr hrep)
          hroutes hlocate hshard hfind houter hslot hinsert
      · exact present_reachablePrefixCoverageN hN kvs i.val state key value shard slot slots old
          hcoverage hi hshard hfind houter hslot
      · exact routes_replacePresent locate state key
          (core.num.I64.wrapping_add value old) shard slot slots old hroutes houter hslot
      · exact presentLocator_exact_succN kvs i key value hi hindex locate shard slot hlocate
          hkeys hrouted
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      have hrouted : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok shard := by
        rw [← word_shard_index_independent_N state (emptyWordTable 0#i64) key]
        exact hshard
      have hnot : key ∉ routedStreamKeys kvs i.val shard := by
        intro hm
        obtain ⟨slot, hs⟩ := (hkeys shard key).mpr hm
        rw [hlocate] at hs
        simp at hs
      have hcard : (routedStreamKeys kvs i.val shard).card < N.val :=
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
          let next := insertVacant state shard (Array.repeat N none) 0#usize key
            (core.num.I64.wrapping_add value state.default)
          have hinsert := insert_with_empty_shard_exact state key value shard hshard hfind
            houter hN hshardBound
          have hroutes' : Routes next (insertLocator locate key shard 0#usize) :=
            routes_insertEmptyShard_exactN hN locate state key
              (core.num.I64.wrapping_add value state.default) shard hroutes hshard houter
              hshardBound
          refine ⟨next, hinsert, ?_, ?_, insertLocator locate key shard 0#usize, ?_, ?_⟩
          · apply (functionalRep_iff_readRep_of_routes (insertLocator locate key shard 0#usize)
              next _ hroutes').mp
            unfold FunctionalRep at ⊢
            have hfunctional : FunctionalRep locate state
                (orderedFold default ((sliceEntries kvs).take i.val)) :=
              (functionalRep_iff_readRep_of_routes locate state _ hroutes).mpr hrep
            rw [stateRead_insertEmptyShard locate state key value shard hlocate houter hN,
              hfunctional]
          · exact emptyShard_reachablePrefixCoverageN hN kvs i state key value shard hcoverage
              hi hindex hshard hfind houter hshardBound
          · exact routes_insertEmptyShard_exactN hN locate state key
              (core.num.I64.wrapping_add value state.default) shard hroutes hshard houter
              hshardBound
          · exact insertLocator_exact_succN kvs i key value hi hindex locate shard 0#usize
              hlocate hkeys hrouted
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_okN state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hcoverage.1.1 shard.val slots houterGet
          have hkLt : k < N.val := reachablePrefixCoverage_capacity_of_card_ltN kvs i.val state
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
          have hnomatch : NoMatchIn key slots.val 0 N.val := by simpa [FindPostN] using hpost
          let first : Usize := ⟨k, by scalar_tac⟩
          have hfirst : first < N := hkLt
          have hslot : Array.index_usize slots first = ok none :=
            index_usize_eq_ok_of_getElem?_eq_someN slots first none (hvacant k le_rfl hkLt)
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
          · exact vacant_reachablePrefixCoverageN hN kvs i state key value shard first slots
              hcoverage hi hindex hshard hfind houter hfirst hoccupied hnomatchPrefix hslot
          · exact routes_insertVacant locate state key
              (core.num.I64.wrapping_add value state.default) shard first slots hroutes hshard
              houter hfirst hnomatchPrefix hslot
          · exact insertLocator_exact_succN kvs i key value hi hindex locate shard first
              hlocate hkeys hrouted

private theorem reachablePrefixExact_card_le_occupiedPrefixN {N : Usize}
    (kvs : Slice (I64 × I64)) (i : Nat) (state : WordTableN N)
    (hexact : ReachablePrefixExactN kvs i state) (shard : Usize)
    (slots : Array (Option (I64 × I64)) N) (k : Nat)
    (houter : Array.index_usize state.shards shard = ok (some slots))
    (hvacant : ∀ j, k ≤ j → j < N.val → slots.val[j]? = some none) :
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
    have hslotGet := getElem?_eq_some_of_index_usize_eq_okN slots (slotOf x)
      (some (x.val, value)) hrouteSlot
    have hltN : (slotOf x).val < N.val := by
      by_contra h
      rw [List.getElem?_eq_none (by rw [slots.property]; exact Nat.not_lt.mp h)] at hslotGet
      simp at hslotGet
    by_contra h
    rw [hvacant (slotOf x).val (Nat.not_lt.mp h) hltN] at hslotGet
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
    for that step at general `N`. -/
theorem reachablePrefixExact_admissible_of_insert_successN {N : Usize} (hN : 0#usize < N)
    (_default : I64) (kvs : Slice (I64 × I64)) (i : Usize) (state next : WordTableN N)
    (key value : I64) (hexact : ReachablePrefixExactN kvs i.val state)
    (hsuccess : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 state () key value =
      ok (core.result.Result.Ok (), next)) :
    ∀ shard,
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
          (emptyWordTable 0#i64) key = ok shard →
        key ∈ routedStreamKeys kvs i.val shard ∨
          (routedStreamKeys kvs i.val shard).card < N.val := by
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
          rw [word_shard_index_independent_N state (emptyWordTable 0#i64) key]
          exact hrouted
        exact Aeneas.Std.Result.ok_injective (hshard.symm.trans hstateRoute)
      subst locatedShard
      exact Or.inl ((hkeys shard key).mp ⟨slot, hlocate⟩)
  | none =>
      rw [hlocate] at hroute
      obtain ⟨locatedShard, hshard, hfind⟩ := hroute
      have hstateRoute : table.Table.shard_index
          table.WordHome.Insts.Apeiron_wordTableHomeI64 state key = ok shard := by
        rw [word_shard_index_independent_N state (emptyWordTable 0#i64) key]
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
          rw [hempty, Finset.card_empty]
          simpa using hN
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_okN state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hcoverage.1.1 shard.val slots houterGet
          have hkLt : k < N.val := by
            by_contra h
            have hkEq : k = N.val := Nat.le_antisymm hk (Nat.not_lt.mp h)
            subst k
            have hfull := insert_with_full_exactN hN state key value shard slots hstateRoute
              hfind houter hoccupied
            rw [hfull] at hsuccess
            simp at hsuccess
          exact Nat.lt_of_le_of_lt
            (reachablePrefixExact_card_le_occupiedPrefixN kvs i.val state
              ⟨hcoverage, locate, hroutes, hkeys⟩ shard slots k houter hvacant)
            hkLt

end Apeiron.TableRefinement
