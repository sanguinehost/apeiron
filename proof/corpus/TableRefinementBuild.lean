import TableRefinementGeneric

/-! # Ordered build refinement for the executing fixed-shard word table -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- The ordinary list carried by a generated borrowed slice. -/
def sliceEntries (kvs : Slice (I64 × I64)) : List (I64 × I64) :=
  kvs.val

/-- The ordered canonical fold from the constant default map. -/
def orderedFold (default : I64) (kvs : List (I64 × I64)) : HMap I64 I64 :=
  kvs.foldl (fun state kv =>
    insertWith core.num.I64.wrapping_add kv.1 kv.2 state) (fun _ => default)

/-- The generated slice and its ordinary list have the same length. -/
theorem sliceEntries_length (kvs : Slice (I64 × I64)) :
    (sliceEntries kvs).length = (Slice.len kvs).val := by
  simp [sliceEntries, Slice.len]

/-- The local ordered fold is the canonical corpus build. -/
theorem orderedFold_eq_build (default : I64) (kvs : List (I64 × I64)) :
    orderedFold default kvs =
      Apeiron.Table.build core.num.I64.wrapping_add default kvs := rfl

/-- Physical facts sufficient for one successful generated insertion.  The three alternatives are
    a present key, a first vacant slot in an existing shard, and slot zero in an untouched shard.
    Construction and preservation of these facts is the residual physical-state invariant. -/
def BuildInsertPremises (t : WordTable) (h : HMap I64 I64) (key value : I64) : Prop :=
  ∃ locate, FunctionalRep locate t h ∧ Routes t locate ∧
    ((∃ shard i slots old,
        locate key = some (shard, i) ∧
        table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
        table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
          t shard key = ok (some i) ∧
        Array.index_usize t.shards shard = ok (some slots) ∧
        Array.index_usize slots i = ok (some (key, old))) ∨
      (∃ shard first slots,
        locate key = none ∧
        table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
        table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
          t shard key = ok none ∧
        Array.index_usize t.shards shard = ok (some slots) ∧
        first < 16#usize ∧ NoEmptyIn slots.val 0 first.val ∧
        NoMatchIn key slots.val 0 first.val ∧ Array.index_usize slots first = ok none) ∨
      (∃ shard,
        locate key = none ∧
        table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
        table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
          t shard key = ok none ∧
        Array.index_usize t.shards shard = ok none ∧ shard < 256#usize ∧
        Routes (insertVacant t shard (Array.repeat 16#usize none) 0#usize key
          (core.num.I64.wrapping_add value t.default))
          (insertLocator locate key shard 0#usize)))

/-- The physical one-step premises discharge the real generated insertion and preserve `ReadRep`. -/
theorem build_insert_premises_success (t : WordTable) (h : HMap I64 I64)
    (key value : I64) (hp : BuildInsertPremises t h key value) :
    ∃ t', table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') ∧
      ReadRep t' (insertWith core.num.I64.wrapping_add key value h) := by
  obtain ⟨locate, hfunctional, hroutes, hpresent | hvacant | hempty⟩ := hp
  · obtain ⟨shard, i, slots, old, hlocate, hshard, hfind, houter, hslot⟩ := hpresent
    let t' := replacePresent t shard slots i key
      (core.num.I64.wrapping_add value old)
    have hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') :=
      insert_with_present_exact t key value shard i slots key old hshard hfind houter hslot
    exact ⟨t', hinsert, insert_with_present_readRep_preservation locate t t' h key value
      shard i slots old hfunctional hroutes hlocate hshard hfind houter hslot hinsert⟩
  · obtain ⟨shard, first, slots, hlocate, hshard, hfind, houter, hfirst, hempty,
      hnomatch, hslot⟩ := hvacant
    let t' := insertVacant t shard slots first key
      (core.num.I64.wrapping_add value t.default)
    have hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') :=
      insert_with_vacant_exact t key value shard first slots hshard hfind houter hfirst
        hempty hnomatch hslot
    exact ⟨t', hinsert, insert_with_vacant_readRep_preservation_routed locate t t' h key
      value shard first slots hfunctional hroutes hlocate hshard hfind houter hfirst
      hempty hnomatch hslot hinsert⟩
  · obtain ⟨shard, hlocate, hshard, hfind, houter, hshardBound, hroutes'⟩ := hempty
    let t' := insertVacant t shard (Array.repeat 16#usize none) 0#usize key
      (core.num.I64.wrapping_add value t.default)
    have hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') :=
      insert_with_empty_shard_exact t key value shard hshard hfind houter (by decide)
        hshardBound
    refine ⟨t', hinsert, (functionalRep_iff_readRep_of_routes _ t' _ hroutes').mp ?_⟩
    unfold FunctionalRep at hfunctional ⊢
    rw [stateRead_insertEmptyShard locate t key value shard hlocate houter (by decide),
      hfunctional]

/-! ## Occupied-prefix physical invariant -/

/-- A populated shard consists of occupied slots followed by vacant slots. -/
def OccupiedPrefix (slots : List (Option (I64 × I64))) : Prop :=
  ∃ k, k ≤ 16 ∧ NoEmptyIn slots 0 k ∧
    ∀ j, k ≤ j → j < 16 → slots[j]? = some none

/-- Every populated physical shard has an occupied prefix and vacant suffix.  Untouched shards are
    represented by `none` and impose no slot condition. -/
def TableInv (t : WordTable) : Prop :=
  ∀ (shard : Nat) (slots : Array (Option (I64 × I64)) 16#usize),
    t.shards.val[shard]? = some (some slots) → OccupiedPrefix slots.val

/-- No key occurs at two physical positions in one populated shard. -/
def ShardKeysDistinct (slots : List (Option (I64 × I64))) : Prop :=
  ∀ (key : I64) (i j : Nat) (left right : I64),
    i < 16 → j < 16 →
    slots[i]? = some (some (key, left)) →
    slots[j]? = some (some (key, right)) → i = j

/-- The occupied-prefix invariant together with physical key uniqueness in every populated shard. -/
def TableInv2 (t : WordTable) : Prop :=
  TableInv t ∧ ∀ (shard : Nat) (slots : Array (Option (I64 × I64)) 16#usize),
    t.shards.val[shard]? = some (some slots) → ShardKeysDistinct slots.val

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

private theorem getElem?_set_eq_self {α : Type} (xs : List α) (i : Nat) (x : α)
    (hi : i < xs.length) : (xs.set i x)[i]? = some x := by
  rw [List.getElem?_eq_getElem (by simpa), List.getElem_set_self]

private theorem getElem?_set_eq_of_ne {α : Type} (xs : List α) (i j : Nat) (x : α)
    (hij : j ≠ i) : (xs.set i x)[j]? = xs[j]? := by
  simp [hij]

/-- The generated constructor establishes the physical occupied-prefix invariant. -/
theorem emptyWordTable_tableInv (default : I64) : TableInv (emptyWordTable default) := by
  set_option maxRecDepth 4096 in
  unfold TableInv
  intro shard slots h
  change (List.replicate 256
    (none : Option (Array (Option (I64 × I64)) 16#usize)))[shard]? = some (some slots) at h
  by_cases hs : shard < 256
  · rw [List.getElem?_eq_getElem (by simpa using hs), List.getElem_replicate] at h
    simp at h
  · rw [List.getElem?_eq_none (by simpa using Nat.not_lt.mp hs)] at h
    simp at h

/-- The generated constructor establishes occupied-prefix shape and physical key uniqueness. -/
theorem emptyWordTable_tableInv2 (default : I64) : TableInv2 (emptyWordTable default) := by
  set_option maxRecDepth 4096 in
    refine ⟨emptyWordTable_tableInv default, ?_⟩
    intro shard slots h
    change (List.replicate 256
      (none : Option (Array (Option (I64 × I64)) 16#usize)))[shard]? = some (some slots) at h
    by_cases hs : shard < 256
    · rw [List.getElem?_eq_getElem (by simpa using hs), List.getElem_replicate] at h
      simp at h
    · rw [List.getElem?_eq_none (by simpa using Nat.not_lt.mp hs)] at h
      simp at h

/-- An occupied prefix of length `k` with physical uniqueness has exactly `k` distinct keys,
    represented by an injective enumeration indexed by `Fin k`. -/
theorem occupiedPrefix_distinct_key_count
    (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat)
    (hk : k ≤ 16) (hoccupied : NoEmptyIn slots.val 0 k)
    (hdistinct : ShardKeysDistinct slots.val) :
    ∃ keys : Fin k → I64, Function.Injective keys ∧
      ∀ i : Fin k, ∃ value, slots.val[i.val]? = some (some (keys i, value)) := by
  have hexists : ∀ i : Fin k, ∃ key value,
      slots.val[i.val]? = some (some (key, value)) := by
    intro i
    have hi16 : i.val < 16 := Nat.lt_of_lt_of_le i.isLt hk
    have hiLen : i.val < slots.val.length := by simpa [slots.property] using hi16
    have hsome : ∃ entry, slots.val[i.val]? = some entry :=
      ⟨slots.val[i.val], List.getElem?_eq_getElem hiLen⟩
    obtain ⟨entry, hentry⟩ := hsome
    cases entry with
    | none =>
        exact (hoccupied i.val (Nat.zero_le _) i.isLt (by simpa [EmptyAt] using hentry)).elim
    | some pair =>
        rcases pair with ⟨key, value⟩
        exact ⟨key, value, hentry⟩
  classical
  let keys : Fin k → I64 := fun i => Classical.choose (hexists i)
  have hvalues : ∀ i : Fin k, ∃ value,
      slots.val[i.val]? = some (some (keys i, value)) := by
    intro i
    exact ⟨Classical.choose (Classical.choose_spec (hexists i)),
      Classical.choose_spec (Classical.choose_spec (hexists i))⟩
  refine ⟨keys, ?_, hvalues⟩
  intro i j hij
  obtain ⟨vi, hi⟩ := hvalues i
  obtain ⟨vj, hj⟩ := hvalues j
  apply Fin.ext
  exact hdistinct (keys i) i.val j.val vi vj
    (Nat.lt_of_lt_of_le i.isLt hk) (Nat.lt_of_lt_of_le j.isLt hk) hi (by simpa [hij] using hj)

private theorem occupiedPrefix_set_present
    (slots : Array (Option (I64 × I64)) 16#usize) (i : Usize) (key old value : I64)
    (hp : OccupiedPrefix slots.val)
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    OccupiedPrefix (slots.set i (some (key, value))).val := by
  obtain ⟨k, hk, hoccupied, hvacant⟩ := hp
  have hslot' := getElem?_eq_some_of_index_usize_eq_ok slots i (some (key, old)) hslot
  have hi : i.val < 16 := by
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

private theorem occupiedPrefix_set_vacant
    (slots : Array (Option (I64 × I64)) 16#usize) (k : Usize) (key value : I64)
    (hoccupied : NoEmptyIn slots.val 0 k.val)
    (hvacant : ∀ j, k.val ≤ j → j < 16 → slots.val[j]? = some none)
    (hk : k < 16#usize) :
    OccupiedPrefix (slots.set k (some (key, value))).val := by
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

private theorem shardKeysDistinct_set_present
    (slots : Array (Option (I64 × I64)) 16#usize) (i : Usize) (key old value : I64)
    (hdistinct : ShardKeysDistinct slots.val)
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    ShardKeysDistinct (slots.set i (some (key, value))).val := by
  have hold := getElem?_eq_some_of_index_usize_eq_ok slots i (some (key, old)) hslot
  have hi : i.val < slots.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hold
    simp at hold
  intro stored a b left right ha hb hleft hright
  by_cases hai : a = i.val
  · subst a
    have hnew := getElem?_set_eq_self slots.val i.val (some (key, value)) hi
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
      have hnew := getElem?_set_eq_self slots.val i.val (some (key, value)) hi
      rw [Array.set_val_eq, hnew] at hright
      have hstored : stored = key :=
        (Prod.mk.inj (Option.some.inj (Option.some.inj hright.symm))).1
      exact hdistinct key a i.val left old ha (by simpa [slots.property] using hi)
        (by simpa [hstored] using hleftOld) hold
    · have hrightOld : slots.val[b]? = some (some (stored, right)) := by
        simpa [Array.set_val_eq, hbi] using hright
      exact hdistinct stored a b left right ha hb hleftOld hrightOld

private theorem shardKeysDistinct_set_vacant
    (slots : Array (Option (I64 × I64)) 16#usize) (first : Usize) (key value : I64)
    (hp : OccupiedPrefix slots.val) (hdistinct : ShardKeysDistinct slots.val)
    (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    ShardKeysDistinct (slots.set first (some (key, value))).val := by
  obtain ⟨k, hk, hoccupied, hvacant⟩ := hp
  have hfirstSlot := getElem?_eq_some_of_index_usize_eq_ok slots first none hslot
  have hfirst : first.val < 16 := by
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
    have hnew := getElem?_set_eq_self slots.val first.val (some (key, value)) hlen
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
      have hnew := getElem?_set_eq_self slots.val first.val (some (key, value)) hlen
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

private theorem tableInv_replacePresent (t : WordTable) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (key old value : I64)
    (hinv : TableInv t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    TableInv (replacePresent t shard slots i key value) := by
  unfold TableInv at hinv ⊢
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_ok t.shards shard (some slots) houter
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
    apply occupiedPrefix_set_present slots i key old value
    · apply hinv shard.val slots
      exact houter'
    · exact hslot
  · apply hinv s current
    simpa [replacePresent, Array.set_val_eq, hss] using hcurrent

private theorem tableInv_insertVacant (t : WordTable) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (key value : I64)
    (hinv : TableInv t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < 16#usize) (hempty : NoEmptyIn slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    TableInv (insertVacant t shard slots first key value) := by
  unfold TableInv at hinv ⊢
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_ok t.shards shard (some slots) houter
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
    have hslot' := getElem?_eq_some_of_index_usize_eq_ok slots first none hslot
    have hfirstNat : first.val < 16 := hfirst
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
    exact occupiedPrefix_set_vacant slots first key value hoccupied hvacant hfirst
  · apply hinv s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem tableInv_insertEmptyShard (t : WordTable) (shard : Usize) (key value : I64)
    (hinv : TableInv t) (hshard : shard < 256#usize) :
    TableInv (insertVacant t shard (Array.repeat 16#usize none) 0#usize key value) := by
  unfold TableInv at hinv ⊢
  intro s current hcurrent
  by_cases hss : s = shard.val
  · subst s
    have hs : shard.val < 256 := hshard
    have hcurrent' : current = (Array.repeat 16#usize none).set 0#usize (some (key, value)) := by
      simp [insertVacant, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    refine ⟨1, by decide, ?_, ?_⟩
    · intro j _ hj
      have hj0 : j = 0 := by omega
      subst j
      simp [EmptyAt, Array.repeat_val, Array.set_val_eq]
    · intro j hj hj16
      have hj0 : j ≠ 0 := by omega
      have hjLen : j < (List.replicate 16 (none : Option (I64 × I64))).length := by simpa
      change ((List.replicate 16 (none : Option (I64 × I64))).set 0
        (some (key, value)))[j]? = some none
      rw [List.getElem?_set_of_lt _ _ hjLen, if_neg (Ne.symm hj0)]
      rw [List.getElem?_eq_getElem hjLen, List.getElem_replicate]
  · apply hinv s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem tableInv2_replacePresent (t : WordTable) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (key old value : I64)
    (hinv : TableInv2 t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    TableInv2 (replacePresent t shard slots i key value) := by
  refine ⟨tableInv_replacePresent t shard i slots key old value hinv.1 houter hslot, ?_⟩
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_ok t.shards shard (some slots) houter
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
    exact shardKeysDistinct_set_present slots i key old value
      (hinv.2 shard.val slots houter') hslot
  · apply hinv.2 s current
    simpa [replacePresent, Array.set_val_eq, hss] using hcurrent

private theorem tableInv2_insertVacant (t : WordTable) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (key value : I64)
    (hinv : TableInv2 t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < 16#usize) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    TableInv2 (insertVacant t shard slots first key value) := by
  refine ⟨tableInv_insertVacant t shard first slots key value hinv.1 houter hfirst hempty hslot,
    ?_⟩
  intro s current hcurrent
  have houter' := getElem?_eq_some_of_index_usize_eq_ok t.shards shard (some slots) houter
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
    exact shardKeysDistinct_set_vacant slots first key value
      (hinv.1 shard.val slots houter') (hinv.2 shard.val slots houter') hempty hnomatch hslot
  · apply hinv.2 s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem tableInv2_insertEmptyShard (t : WordTable) (shard : Usize) (key value : I64)
    (hinv : TableInv2 t) (hshard : shard < 256#usize) :
    TableInv2 (insertVacant t shard (Array.repeat 16#usize none) 0#usize key value) := by
  refine ⟨tableInv_insertEmptyShard t shard key value hinv.1 hshard, ?_⟩
  intro s current hcurrent
  by_cases hss : s = shard.val
  · subst s
    have hs : shard.val < 256 := hshard
    have hcurrent' : current = (Array.repeat 16#usize none).set 0#usize
        (some (key, value)) := by
      simp [insertVacant, Array.set_val_eq, hs] at hcurrent
      exact hcurrent.symm
    subst current
    apply shardKeysDistinct_set_vacant (Array.repeat 16#usize none) 0#usize key value
    · refine ⟨0, by decide, ?_, ?_⟩
      · intro j _ hj
        omega
      · intro j _ hj
        change (List.replicate 16 (none : Option (I64 × I64)))[j]? = some none
        have hjLen : j < (List.replicate 16 (none : Option (I64 × I64))).length := by
          simpa using hj
        rw [List.getElem?_eq_getElem hjLen, List.getElem_replicate]
    · intro stored i j left right hi hj hleft
      change (List.replicate 16 (none : Option (I64 × I64)))[i]? =
        some (some (stored, left)) at hleft
      rw [List.getElem?_eq_getElem (by simpa using hi), List.getElem_replicate] at hleft
      simp at hleft
    · simp [NoEmptyIn]
    · simp [NoMatchIn]
    · apply index_usize_eq_ok_of_getElem?_eq_some
      simp [Array.repeat_val]
  · apply hinv.2 s current
    simpa [insertVacant, Array.set_val_eq, hss] using hcurrent

private theorem repeat16_find_none (key : I64) :
    table.Table.find_slots core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      (Array.repeat 16#usize (none : Option (I64 × I64))) key = ok none := by
  obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec_generic (Array.repeat 16#usize (none : Option (I64 × I64))) key
      0#usize (Nat.zero_le _))
  cases found with
  | none => exact hfound
  | some i =>
      obtain ⟨hi, _, ⟨value, hmatch⟩, _⟩ := hpost
      have hiLen : i.val < (List.replicate 16 (none : Option (I64 × I64))).length := by simpa
      change (List.replicate 16 (none : Option (I64 × I64)))[i.val]? =
        some (some (key, value)) at hmatch
      rw [List.getElem?_eq_getElem hiLen, List.getElem_replicate] at hmatch
      simp at hmatch

private theorem routes_insertEmptyShard
    (locate : I64 → Option (Usize × Usize)) (t : WordTable) (key value : I64)
    (shard : Usize) (hroutes : Routes t locate)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (houter : Array.index_usize t.shards shard = ok none) (hshardBound : shard < 256#usize) :
    Routes (insertVacant t shard (Array.repeat 16#usize none) 0#usize key value)
      (insertLocator locate key shard 0#usize) := by
  intro x
  let next := insertVacant t shard (Array.repeat 16#usize none) 0#usize key value
  have hnewOuter : Array.index_usize next.shards shard =
      ok (some ((Array.repeat 16#usize none).set 0#usize (some (key, value)))) := by
    have hget : next.shards.val[shard.val]? =
        some (some ((Array.repeat 16#usize none).set 0#usize (some (key, value)))) := by
      dsimp [next, insertVacant]
      rw [Array.set_val_eq]
      have hsLen : shard.val < t.shards.val.length := by
        rw [t.shards.property]
        exact hshardBound
      exact List.getElem?_set_eq_of_lt _ hsLen
    exact index_usize_eq_ok_of_getElem?_eq_some next.shards shard _ hget
  have hrepeatSlot : Array.index_usize (Array.repeat 16#usize
      (none : Option (I64 × I64))) 0#usize = ok none := by
    apply index_usize_eq_ok_of_getElem?_eq_some
    simp [Array.repeat_val]
  by_cases hx : x = key
  · subst x
    simp only [insertLocator]
    apply generatedRoute_some_of_find next key shard 0#usize
    · rw [← word_shard_index_state_independent t next key]
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
        · rw [← word_shard_index_state_independent t next x]
          exact hs
        · by_cases hsame : s = shard
          · subst s
            unfold table.Table.find
            rw [hnewOuter]
            simp only [bind_tc_ok]
            rw [find_slots_insertVacant_ne (Array.repeat 16#usize none) 0#usize key value x
              hrepeatSlot hx, repeat16_find_none]
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

/-- An absent key in a physically full shard returns `ShardFull` with the state unchanged. -/
theorem insert_with_full_exact (t : WordTable) (key value : I64) (shard : Usize)
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
    (insert_with_loop0_full_spec_generic (t.shards.set shard) slots t.default key value
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
  have hget : t.shards.val[shard.val] = some slots := by
    rw [List.getElem?_eq_getElem hsLen] at houter'
    exact Option.some.inj houter'
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

/-- Every prefix state supplies the physical facts needed for its next generated insertion. -/
def AllBuildInsertsAdmitted (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTable) (key value : I64),
    i.val < (sliceEntries kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    BuildInsertPremises state
      (orderedFold default ((sliceEntries kvs).take i.val)) key value

/-- Fallback capacity contract on physical representatives of each canonical prefix fold.  Every
    populated shard has an occupied-prefix witness strictly shorter than 16.  The stronger
    stream-only result, `prefixShardCapacity_of_distinctKeysPerShard`, relating this load to the
    number of distinct routed keys in each prefix, remains to be proved. -/
def PrefixShardCapacity (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTable),
    i.val < (sliceEntries kvs).length →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    TableInv state ∧
      ∀ (shard : Nat) (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat),
        state.shards.val[shard]? = some (some slots) →
        k ≤ 16 → NoEmptyIn slots.val 0 k →
        (∀ j, k ≤ j → j < 16 → slots.val[j]? = some none) → k < 16

/-- Distinct stream keys in one generated word shard, independent of table state. -/
noncomputable def routedStreamKeys (kvs : Slice (I64 × I64)) (i : Nat)
    (shard : Usize) : Finset I64 := by
  classical
  exact (((sliceEntries kvs).take i).map Prod.fst).toFinset.filter fun key =>
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      (emptyWordTable 0#i64) key = ok shard

/-- Stream-only admission contract: every proper prefix has fewer than sixteen distinct keys routed
    to each physical shard. -/
def DistinctKeysPerShard (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Nat) (shard : Usize), i < (sliceEntries kvs).length →
    (routedStreamKeys kvs i shard).card < 16

/-- Additional physical/stream invariant needed because `ReadRep` alone permits duplicate or
    unrelated physical entries.  It states shape and uniqueness, and covers every occupied-prefix
    key by a distinct routed key from the corresponding stream prefix. -/
def PrefixShardKeyCoverage (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTable),
    i.val < (sliceEntries kvs).length →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    TableInv2 state ∧
      ∀ (shard : Usize) (slots : Array (Option (I64 × I64)) 16#usize) (k : Nat),
        Array.index_usize state.shards shard = ok (some slots) →
        k ≤ 16 → NoEmptyIn slots.val 0 k →
        ∀ (j : Nat) (key value : I64), j < k →
          slots.val[j]? = some (some (key, value)) →
          key ∈ routedStreamKeys kvs i.val shard

/-- Distinct routed stream keys imply strict physical prefix capacity once the explicit
    physical/stream coverage invariant is supplied. -/
theorem prefixShardCapacity_of_distinctKeysPerShard (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShard kvs)
    (hcoverage : PrefixShardKeyCoverage default kvs) : PrefixShardCapacity default kvs := by
  intro i state hi hrep
  obtain ⟨hinv, hcovered⟩ := hcoverage i state hi hrep
  refine ⟨hinv.1, ?_⟩
  intro shardNat slots k houter hk hoccupied _hvacant
  have hshardBound : shardNat < 256 := by
    have hlen : shardNat < state.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter
      simp at houter
    simpa [state.shards.property] using hlen
  let shard : Usize := ⟨shardNat, by scalar_tac⟩
  have houterIndex : Array.index_usize state.shards shard = ok (some slots) :=
    index_usize_eq_ok_of_getElem?_eq_some state.shards shard (some slots) houter
  obtain ⟨keys, hkeysInjective, hkeysSlots⟩ :=
    occupiedPrefix_distinct_key_count slots k hk hoccupied
      (hinv.2 shardNat slots houter)
  let intoStream : Fin k → { key // key ∈ routedStreamKeys kvs i.val shard } := fun j =>
    ⟨keys j, by
      obtain ⟨value, hslot⟩ := hkeysSlots j
      exact hcovered shard slots k houterIndex hk hoccupied j.val (keys j) value j.isLt hslot⟩
  have hIntoInjective : Function.Injective intoStream := by
    intro a b hab
    apply hkeysInjective
    exact Subtype.ext_iff.mp hab
  have hcard : k ≤ (routedStreamKeys kvs i.val shard).card := by
    simpa [intoStream] using Fintype.card_le_of_injective intoStream hIntoInjective
  exact Nat.lt_of_le_of_lt hcard (hdistinct i.val shard hi)

/-- Strict occupied-prefix capacity at every canonical prefix supplies all one-step admission
    premises required by the generated build loop. -/
theorem allBuildInsertsAdmitted_of_prefixShardCapacity (default : I64)
    (kvs : Slice (I64 × I64)) (hcapacity : PrefixShardCapacity default kvs) :
    AllBuildInsertsAdmitted default kvs := by
  intro i state key value hi _hindex hrep
  obtain ⟨hinv, hload⟩ := hcapacity i state hi hrep
  let locate := generatedLocator state
  have hroutes : Routes state locate := generatedLocator_routes state
  have hfunctional : FunctionalRep locate state
      (orderedFold default ((sliceEntries kvs).take i.val)) :=
    (generatedFunctionalRep_iff_readRep state
      (orderedFold default ((sliceEntries kvs).take i.val))).mpr hrep
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, slot⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      exact ⟨locate, hfunctional, hroutes, Or.inl
        ⟨shard, slot, slots, old, hlocate, hshard, hfind, houter, hslot⟩⟩
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
          have hroutes' : Routes next (insertLocator locate key shard 0#usize) :=
            routes_insertEmptyShard locate state key
              (core.num.I64.wrapping_add value state.default) shard hroutes hshard houter
              hshardBound
          exact ⟨locate, hfunctional, hroutes, Or.inr (Or.inr
            ⟨shard, hlocate, hshard, hfind, houter, hshardBound, hroutes'⟩)⟩
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok state.shards shard
            (some slots) houter
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hinv shard.val slots houterGet
          have hkLt : k < 16 := hload shard.val slots k houterGet hk hoccupied hvacant
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
          exact ⟨locate, hfunctional, hroutes, Or.inr (Or.inl
            ⟨shard, first, slots, hlocate, hshard, hfind, houter, hfirst, hoccupied,
              fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt), hslot⟩)⟩

/-- Read-boundary behavior required from one insertion when capacity refusal remains possible. -/
def BuildInsertResult (t : WordTable) (h : HMap I64 I64) (key value : I64) : Prop :=
  table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
    table.WordHome.Insts.Apeiron_wordTableHomeI64
    table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
    ⦃ result => match result.1 with
      | core.result.Result.Ok _ =>
          ReadRep result.2 (insertWith core.num.I64.wrapping_add key value h)
      | core.result.Result.Err table.StorageError.ShardFull => True ⦄

/-- The physical and read-boundary postcondition of one capacity-aware insertion. -/
def BuildInsertInvResult (t : WordTable) (h : HMap I64 I64) (key value : I64)
    (result : core.result.Result Unit table.StorageError × WordTable) : Prop :=
  match result.1 with
  | core.result.Result.Ok _ => TableInv result.2 ∧
      ReadRep result.2 (insertWith core.num.I64.wrapping_add key value h)
  | core.result.Result.Err table.StorageError.ShardFull => result.2 = t ∧ TableInv result.2

/-- The real generated insertion preserves the occupied-prefix invariant.  Success also preserves
    the canonical read representation; capacity refusal returns the unchanged state. -/
theorem tableInv_insert_with (t : WordTable) (h : HMap I64 I64) (key value : I64)
    (hinv : TableInv t) (hrep : ReadRep t h) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => BuildInsertInvResult t h key value result ⦄ := by
  let locate := generatedLocator t
  have hroutes : Routes t locate := generatedLocator_routes t
  have hfunctional : FunctionalRep locate t h :=
    (generatedFunctionalRep_iff_readRep t h).mpr hrep
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, i⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      have hp : BuildInsertPremises t h key value :=
        ⟨locate, hfunctional, hroutes, Or.inl
          ⟨shard, i, slots, old, hlocate, hshard, hfind, houter, hslot⟩⟩
      obtain ⟨t', hinsert, hrep'⟩ := build_insert_premises_success t h key value hp
      rw [hinsert]
      simp only [Aeneas.Std.WP.spec_ok, BuildInsertInvResult]
      refine ⟨?_, hrep'⟩
      have hexact := insert_with_present_exact t key value shard i slots key old
        hshard hfind houter hslot
      rw [hinsert] at hexact
      have ht' : t' = replacePresent t shard slots i key
          (core.num.I64.wrapping_add value old) := by
        exact (Prod.mk.inj (Aeneas.Std.Result.ok_injective hexact)).2
      subst t'
      exact tableInv_replacePresent t shard i slots key old
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
      obtain ⟨outer, houter, houterValue⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec t.shards shard hshardLen)
      cases outer with
      | none =>
          have houterNone : Array.index_usize t.shards shard = ok none := by
            exact houter
          let t' := insertVacant t shard (Array.repeat 16#usize none) 0#usize key
            (core.num.I64.wrapping_add value t.default)
          have hroutes' : Routes t' (insertLocator locate key shard 0#usize) :=
            routes_insertEmptyShard locate t key
              (core.num.I64.wrapping_add value t.default) shard hroutes hshard houterNone
              hshardBound
          have hp : BuildInsertPremises t h key value :=
            ⟨locate, hfunctional, hroutes, Or.inr (Or.inr
              ⟨shard, hlocate, hshard, hfind, houterNone, hshardBound, hroutes'⟩)⟩
          obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_success t h key value hp
          rw [hinsert]
          simp only [Aeneas.Std.WP.spec_ok, BuildInsertInvResult]
          refine ⟨?_, hrep'⟩
          have hexact := insert_with_empty_shard_exact t key value shard hshard hfind
            houterNone (by decide) hshardBound
          rw [hinsert] at hexact
          have hresult : result = t' := by
            exact (Prod.mk.inj (Aeneas.Std.Result.ok_injective hexact)).2
          subst result
          exact tableInv_insertEmptyShard t shard key
            (core.num.I64.wrapping_add value t.default) hinv hshardBound
      | some slots =>
          have houterSome : Array.index_usize t.shards shard = ok (some slots) := by
            exact houter
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok t.shards shard
            (some slots) houterSome
          obtain ⟨k, hk, hoccupied, hvacant⟩ := hinv shard.val slots houterGet
          have hfindSlots : table.Table.find_slots core.cmp.EqI64
              table.WordHome.Insts.Apeiron_wordTableHomeI64 slots key = ok none := by
            unfold table.Table.find at hfind
            rw [houterSome] at hfind
            simpa only [bind_tc_ok] using hfind
          obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
            (find_slots_spec_generic slots key 0#usize (Nat.zero_le _))
          unfold table.Table.find_slots at hfindSlots
          rw [hfindSlots] at hfound
          have hfoundNone : found = none := Aeneas.Std.Result.ok_injective hfound.symm
          subst found
          have hnomatch : NoMatchIn key slots.val 0 16 := by
            simpa [FindPostN] using hpost
          by_cases hkFull : k = 16
          · subst k
            rw [insert_with_full_exact t key value shard slots hshard hfind houterSome
              hoccupied]
            simp [BuildInsertInvResult, hinv]
          · have hkLt : k < 16 := by omega
            let first : Usize := ⟨k, by scalar_tac⟩
            have hfirst : first < 16#usize := hkLt
            have hslotGet : slots.val[first.val]? = some none :=
              hvacant k le_rfl hkLt
            have hslot : Array.index_usize slots first = ok none :=
              index_usize_eq_ok_of_getElem?_eq_some slots first none hslotGet
            have hp : BuildInsertPremises t h key value :=
              ⟨locate, hfunctional, hroutes, Or.inr (Or.inl
                ⟨shard, first, slots, hlocate, hshard, hfind, houterSome, hfirst,
                  hoccupied, fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt), hslot⟩)⟩
            obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_success t h key value hp
            rw [hinsert]
            simp only [Aeneas.Std.WP.spec_ok, BuildInsertInvResult]
            refine ⟨?_, hrep'⟩
            have hexact := insert_with_vacant_exact t key value shard first slots hshard
              hfind houterSome hfirst hoccupied
              (fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt)) hslot
            rw [hinsert] at hexact
            have hresult : result = insertVacant t shard slots first key
                (core.num.I64.wrapping_add value t.default) := by
              exact (Prod.mk.inj (Aeneas.Std.Result.ok_injective hexact)).2
            subst result
            exact tableInv_insertVacant t shard first slots key
              (core.num.I64.wrapping_add value t.default) hinv houterSome hfirst
              hoccupied hslot

/-- Physical no-duplicate preservation for every generated insertion branch.  Present writes keep
    the matching key in place, fresh writes use a search-proved absent key, and refusal is unchanged. -/
theorem tableInv2_insert_with (t : WordTable) (key value : I64) (hinv : TableInv2 t) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => match result.1 with
        | core.result.Result.Ok _ => TableInv2 result.2
        | core.result.Result.Err table.StorageError.ShardFull => result.2 = t ∧ TableInv2 result.2
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
      exact tableInv2_replacePresent t shard i slots key old
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
            (by decide) hshardBound]
          simp only [Aeneas.Std.WP.spec_ok]
          exact tableInv2_insertEmptyShard t shard key
            (core.num.I64.wrapping_add value t.default) hinv hshardBound
      | some slots =>
          have houterGet := getElem?_eq_some_of_index_usize_eq_ok t.shards shard
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
          have hnomatch : NoMatchIn key slots.val 0 16 := by
            simpa [FindPostN] using hpost
          by_cases hkFull : k = 16
          · subst k
            rw [insert_with_full_exact t key value shard slots hshard hfind houter hoccupied]
            simp [hinv]
          · have hkLt : k < 16 := by omega
            let first : Usize := ⟨k, by scalar_tac⟩
            have hfirst : first < 16#usize := hkLt
            have hslotGet : slots.val[first.val]? = some none := hvacant k le_rfl hkLt
            have hslot : Array.index_usize slots first = ok none :=
              index_usize_eq_ok_of_getElem?_eq_some slots first none hslotGet
            have hnomatchPrefix : NoMatchIn key slots.val 0 first.val :=
              fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hkLt)
            rw [insert_with_vacant_exact t key value shard first slots hshard hfind houter
              hfirst hoccupied hnomatchPrefix hslot]
            simp only [Aeneas.Std.WP.spec_ok]
            exact tableInv2_insertVacant t shard first slots key
              (core.num.I64.wrapping_add value t.default) hinv houter hfirst hoccupied
              hnomatchPrefix hslot

/-- Every generated insertion either preserves the canonical read representation or reports
    `ShardFull`; no physical invariant is needed merely to expose this result disjunction. -/
theorem buildInsertResult_of_readRep (t : WordTable) (h : HMap I64 I64) (key value : I64)
    (hrep : ReadRep t h) : BuildInsertResult t h key value := by
  let locate := generatedLocator t
  have hroutes : Routes t locate := generatedLocator_routes t
  have hfunctional : FunctionalRep locate t h :=
    (generatedFunctionalRep_iff_readRep t h).mpr hrep
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, i⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_success t h key value
        ⟨locate, hfunctional, hroutes, Or.inl
          ⟨shard, i, slots, old, hlocate, hshard, hfind, houter, hslot⟩⟩
      unfold BuildInsertResult
      rw [hinsert]
      simpa using hrep'
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
          let t' := insertVacant t shard (Array.repeat 16#usize none) 0#usize key
            (core.num.I64.wrapping_add value t.default)
          have hroutes' : Routes t' (insertLocator locate key shard 0#usize) :=
            routes_insertEmptyShard locate t key
              (core.num.I64.wrapping_add value t.default) shard hroutes hshard houter
              hshardBound
          obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_success t h key value
            ⟨locate, hfunctional, hroutes, Or.inr (Or.inr
              ⟨shard, hlocate, hshard, hfind, houter, hshardBound, hroutes'⟩)⟩
          unfold BuildInsertResult
          rw [hinsert]
          simpa using hrep'
      | some slots =>
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
          by_cases hvacant : ∃ j, j < 16 ∧ slots.val[j]? = some none
          · let k := Nat.find hvacant
            have hk : k < 16 := (Nat.find_spec hvacant).1
            have hslotGet : slots.val[k]? = some none := (Nat.find_spec hvacant).2
            have hnoempty : NoEmptyIn slots.val 0 k := by
              intro j _ hj hempty
              have hmin := Nat.find_min' hvacant
                ⟨hj.trans hk, by simpa [EmptyAt] using hempty⟩
              exact (Nat.not_lt_of_ge hmin hj).elim
            let first : Usize := ⟨k, by scalar_tac⟩
            have hfirst : first < 16#usize := hk
            have hslot : Array.index_usize slots first = ok none :=
              index_usize_eq_ok_of_getElem?_eq_some slots first none hslotGet
            obtain ⟨result, hinsert, hrep'⟩ := build_insert_premises_success t h key value
              ⟨locate, hfunctional, hroutes, Or.inr (Or.inl
                ⟨shard, first, slots, hlocate, hshard, hfind, houter, hfirst,
                  hnoempty, fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hk), hslot⟩)⟩
            unfold BuildInsertResult
            rw [hinsert]
            simpa using hrep'
          · have hfull : NoEmptyIn slots.val 0 16 := by
              intro j _ hj hempty
              exact hvacant ⟨j, hj, by simpa [EmptyAt] using hempty⟩
            unfold BuildInsertResult
            rw [insert_with_full_exact t key value shard slots hshard hfind houter hfull]
            simp

/-- Every prefix insertion either preserves the ordered fold or reports physical capacity refusal. -/
def AllBuildInsertResults (default : I64) (kvs : Slice (I64 × I64)) : Prop :=
  ∀ (i : Usize) (state : WordTable) (key value : I64),
    i.val < (sliceEntries kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    ReadRep state (orderedFold default ((sliceEntries kvs).take i.val)) →
    BuildInsertResult state
      (orderedFold default ((sliceEntries kvs).take i.val)) key value

/-- Every possible prefix state has the capacity-aware insertion result required by the generated
    build loop. -/
theorem allBuildInsertResults (default : I64) (kvs : Slice (I64 × I64)) :
    AllBuildInsertResults default kvs := by
  intro i state key value _ _ hrep
  exact buildInsertResult_of_readRep state
    (orderedFold default ((sliceEntries kvs).take i.val)) key value hrep

/-- The real generated loop either stops at `ShardFull` or returns the full ordered fold. -/
theorem table_build_loop_result_readRep (default : I64) (kvs : Slice (I64 × I64))
    (initial : WordTable) (start : Usize)
    (hstart : start.val ≤ (sliceEntries kvs).length)
    (hinitial : ReadRep initial (orderedFold default ((sliceEntries kvs).take start.val)))
    (hresults : AllBuildInsertResults default kvs) :
    table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
      initial start
      ⦃ result => result = core.result.Result.Err table.StorageError.ShardFull ∨
        ∃ final, result = core.result.Result.Ok final ∧
          ReadRep final (orderedFold default (sliceEntries kvs)) ⦄ := by
  unfold table.Table.build_loop
  apply loop.spec_decr_nat
    (measure := fun p : WordTable × Usize => (sliceEntries kvs).length - p.2.val)
    (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
      ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)))
    (post := fun result => result = core.result.Result.Err table.StorageError.ShardFull ∨
      ∃ final, result = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)))
  · intro p hp
    rcases p with ⟨state, i⟩
    obtain ⟨hi, hrep⟩ := hp
    dsimp only at hi hrep ⊢
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
      have hinsert := hresults i state key value hiList hindex hrep
      unfold BuildInsertResult at hinsert
      step
      cases r with
      | Ok resultUnit =>
          have hrep' := r_post
          simp only at hrep'
          have htake : (sliceEntries kvs).take (i.val + 1) =
              (sliceEntries kvs).take i.val ++ [(key, value)] := by
            rw [List.take_succ_eq_append_getElem hiList]
            congr
            exact key_post.symm
          step
          refine ⟨?_, ?_, ?_⟩
          · simpa [r_post] using Nat.succ_le_iff.mpr hiList
          · simpa [r_post, orderedFold, htake] using hrep'
          · simp only [r_post]
            omega
      | Err error =>
          cases error
          simp
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      right
      refine ⟨state, rfl, ?_⟩
      have hieq : i.val = (sliceEntries kvs).length := by
        rw [sliceEntries_length]
        exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
      simpa [hieq] using hrep
  · exact ⟨hstart, hinitial⟩

/-- The real generated word build is unconditional: it either reports physical capacity refusal or
    returns a table whose reads are the full canonical ordered fold. -/
theorem word_table_build_result_readRep (default : I64) (kvs : Slice (I64 × I64)) :
    word_table_build default kvs =
        ok (core.result.Result.Err table.StorageError.ShardFull) ∨
      ∃ final, word_table_build default kvs = ok (core.result.Result.Ok final) ∧
        ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
          (sliceEntries kvs)) := by
  have hinitial : ReadRep (emptyWordTable default) (orderedFold default []) := by
    apply (readRep_iff_Rep (emptyWordTable default) (orderedFold default [])).mpr
    intro key
    simpa [orderedFold] using emptyWordTable_read default key
  have hs := table_build_loop_result_readRep default kvs (emptyWordTable default) 0#usize
    (Nat.zero_le _) (by simpa using hinitial) (allBuildInsertResults default kvs)
  obtain ⟨result, hloop, hresult⟩ := Aeneas.Std.WP.spec_imp_exists hs
  unfold word_table_build table.Table.build
  rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64
      default () = ok (emptyWordTable default) by rfl]
  simp only [bind_tc_ok]
  rcases hresult with hfull | ⟨final, hok, hrep⟩
  · left
    rw [hloop, hfull]
  · right
    refine ⟨final, ?_, ?_⟩
    · rw [hloop, hok]
    · simpa [orderedFold_eq_build] using hrep

/-- The real generated loop consumes the slice in order and returns the canonical prefix fold.
    `AllBuildInsertsAdmitted` rules out `ShardFull`; without it the generated error branch stops. -/
theorem table_build_loop_readRep (default : I64) (kvs : Slice (I64 × I64))
    (initial : WordTable) (start : Usize)
    (hstart : start.val ≤ (sliceEntries kvs).length)
    (hinitial : ReadRep initial (orderedFold default ((sliceEntries kvs).take start.val)))
    (hadmitted : AllBuildInsertsAdmitted default kvs) :
    ∃ final,
      table.Table.build_loop core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs
        initial start = ok (core.result.Result.Ok final) ∧
      ReadRep final (orderedFold default (sliceEntries kvs)) := by
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 core.clone.CloneI64 () kvs state i)
      (initial, start)
      ⦃ r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)) ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : WordTable × Usize => (sliceEntries kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntries kvs).length ∧
        ReadRep p.1 (orderedFold default ((sliceEntries kvs).take p.2.val)))
      (post := fun r => ∃ final, r = core.result.Result.Ok final ∧
        ReadRep final (orderedFold default (sliceEntries kvs)))
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hrep⟩ := hp
      dsimp only at hi hrep ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        step
        have hiList : i.val < (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact hlt
        obtain ⟨state', hinsert, hrep'⟩ := build_insert_premises_success state
          (orderedFold default ((sliceEntries kvs).take i.val)) key value
          (hadmitted i state key value hiList (by
            have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
            unfold Slice.index_usize
            rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]) hrep)
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntries kvs).take (i.val + 1) =
            (sliceEntries kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post, orderedFold, htake] using hrep'
        · simp only [r_post]
          omega
      · rw [if_neg hlt]
        simp only [Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntries kvs).length := by
          rw [sliceEntries_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        refine ⟨state, rfl, ?_⟩
        simpa [hieq] using hrep
    · exact ⟨hstart, hinitial⟩
  obtain ⟨result, hresult, final, hfinal, hrep⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, hrep⟩

/-- The generated word build from its real empty constructor refines canonical ordered `build`
    whenever the explicit physical admission premise holds for every prefix. -/
theorem word_table_build_readRep (default : I64) (kvs : Slice (I64 × I64))
    (hadmitted : AllBuildInsertsAdmitted default kvs) :
    ∃ final, word_table_build default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) := by
  have hinitial : ReadRep (emptyWordTable default) (orderedFold default []) := by
    apply (readRep_iff_Rep (emptyWordTable default) (orderedFold default [])).mpr
    intro key
    simpa [orderedFold] using emptyWordTable_read default key
  obtain ⟨final, hloop, hrep⟩ := table_build_loop_readRep default kvs
    (emptyWordTable default) 0#usize (Nat.zero_le _) (by simpa using hinitial) hadmitted
  refine ⟨final, ?_, ?_⟩
  · unfold word_table_build table.Table.build
    rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64
        default () = ok (emptyWordTable default) by rfl]
    simpa using hloop
  · simpa [orderedFold_eq_build] using hrep

/-- Under strict occupied-prefix capacity for every canonical prefix representative, the real word
    build has only the successful branch and refines the canonical ordered build. -/
theorem word_table_build_readRep_of_prefixShardCapacity (default : I64)
    (kvs : Slice (I64 × I64)) (hcapacity : PrefixShardCapacity default kvs) :
    ∃ final, word_table_build default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) :=
  word_table_build_readRep default kvs
    (allBuildInsertsAdmitted_of_prefixShardCapacity default kvs hcapacity)

/-- The stream-only distinct-key bound yields successful generated build refinement under the
    explicit physical/stream coverage invariant. -/
theorem word_table_build_readRep_of_distinctKeysPerShard (default : I64)
    (kvs : Slice (I64 × I64)) (hdistinct : DistinctKeysPerShard kvs)
    (hcoverage : PrefixShardKeyCoverage default kvs) :
    ∃ final, word_table_build default kvs = ok (core.result.Result.Ok final) ∧
      ReadRep final (Apeiron.Table.build core.num.I64.wrapping_add default
        (sliceEntries kvs)) :=
  word_table_build_readRep_of_prefixShardCapacity default kvs
    (prefixShardCapacity_of_distinctKeysPerShard default kvs hdistinct hcoverage)

end Apeiron.TableRefinement
