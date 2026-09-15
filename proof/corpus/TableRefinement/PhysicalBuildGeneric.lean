import TableRefinement.CallbackAdmission
import Mathlib.Data.List.Count

/-! # Physical generic build refinement under explicit prefix admission

The admission condition quantifies only over states reached at an exact canonical prefix, keeps
the executing table's home value fixed, and requires the real insertion to return `Ok`.  The
callback insertion postcondition supplies physical preservation; admission does not assume it.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- Independently checkable physical capacity for one insertion.  It contains no execution result:
    the routed key is already present, or it is absent and its routed shard has a writable slot. -/
def InsertCapacityG {K V H : Type} {N : Usize}
    (HomeInst : table.Home H K) (t : table.Table K V H N) (key : K) : Prop :=
  ∃ shard, table.Table.shard_index HomeInst t key = ok shard ∧
    ((∃ slots i old, UniquePresentAtG t key shard i slots old) ∨
      (KeyAbsentG t key ∧
        ((0#usize < N ∧ Array.index_usize t.shards shard = ok none) ∨
          ∃ slots first, Array.index_usize t.shards shard = ok (some slots) ∧
            FirstVacantG slots first)))

/-- Every exact canonical prefix state has physical capacity for its next routed insertion. -/
def BuildCapacityG {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (mergeFn : V → V → V) (home : H) (default : V) (kvs : Slice (K × V)) : Prop :=
  ∀ (i : Usize) (state : table.Table K V H N) (key : K) (value : V),
    i.val < (sliceEntriesG kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    state.home = home →
    PhysicalInvG cmp HomeInst state
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) →
    InsertCapacityG HomeInst state key

/-- A bounded raw position is physically occupied. -/
def OccupiedPositionG {K V : Type} {N : Usize} (slots : Array (Option (K × V)) N)
    (position : Fin N.val) : Prop :=
  ∃ key value, Array.index_usize slots ⟨position.val, by scalar_tac⟩ =
    ok (some (key, value))

/-- Number of physically occupied positions in one raw shard. -/
noncomputable def OccupiedCountG {K V : Type} {N : Usize}
    (slots : Array (Option (K × V)) N) : Nat := by
  classical
  exact Fintype.card { position : Fin N.val // OccupiedPositionG slots position }

/-- Execution-independent count bound: capacity is positive and every populated shard in every
    represented proper prefix has strictly fewer occupied cells than its physical length. -/
def PrefixPhysicalCountBoundG {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (mergeFn : V → V → V) (home : H) (default : V) (kvs : Slice (K × V)) : Prop :=
  0#usize < N ∧
    ∀ (i : Usize) (state : table.Table K V H N),
      i.val < (sliceEntriesG kvs).length →
      state.home = home →
      PhysicalInvG cmp HomeInst state
        (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) →
      ∀ (shard : Usize) (slots : Array (Option (K × V)) N),
        Array.index_usize state.shards shard = ok (some slots) →
        OccupiedCountG slots < N.val

/-- Distinct stream keys in one callback-routed canonical prefix. -/
noncomputable def routedPrefixKeysG {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    (prefixLen : Nat) (shard : Usize) : Finset K := by
  classical
  exact (((sliceEntriesG kvs).take prefixLen).map Prod.fst).toFinset.filter fun key =>
    table.Table.shard_index HomeInst (emptyTableG (N := N) default home) key = ok shard

/-- Stream-only bound on distinct callback-routed keys in every proper prefix. -/
def DistinctRoutedKeysPerShardG {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V)) : Prop :=
  ∀ (prefixLen : Nat) (shard : Usize), prefixLen < (sliceEntriesG kvs).length →
    (routedPrefixKeysG (N := N) HomeInst default home kvs prefixLen shard).card < N.val

/-- No-ghost coverage: every occupied physical slot is a key in the corresponding callback-routed
    canonical prefix.  It covers all raw slots, independent of dense-prefix shape. -/
def PrefixPhysicalKeyCoverageG {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (mergeFn : V → V → V) (home : H) (default : V) (kvs : Slice (K × V)) : Prop :=
  ∀ (i : Usize) (state : table.Table K V H N),
    i.val < (sliceEntriesG kvs).length →
    state.home = home →
    PhysicalInvG cmp HomeInst state
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) →
    ∀ (shard slot : Usize) (slots : Array (Option (K × V)) N) (key : K) (value : V),
      Array.index_usize state.shards shard = ok (some slots) →
      Array.index_usize slots slot = ok (some (key, value)) →
      key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs i.val shard

/-- State-local provenance carried by the reachable build loop. -/
def PhysicalKeyCoverageAtG {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    (prefixLen : Nat) (state : table.Table K V H N) : Prop :=
  ∀ (shard slot : Usize) (slots : Array (Option (K × V)) N) (key : K) (value : V),
    Array.index_usize state.shards shard = ok (some slots) →
    Array.index_usize slots slot = ok (some (key, value)) →
    key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs prefixLen shard

/-- Routed prefix-key sets are monotone in the prefix length. -/
theorem routedPrefixKeysG_mono {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    {prefixLen nextLen : Nat} (hle : prefixLen ≤ nextLen) (shard : Usize) :
    routedPrefixKeysG (N := N) HomeInst default home kvs prefixLen shard ⊆
      routedPrefixKeysG (N := N) HomeInst default home kvs nextLen shard := by
  classical
  intro key hkey
  simp only [routedPrefixKeysG, Finset.mem_filter, List.mem_toFinset, List.mem_map] at hkey ⊢
  obtain ⟨⟨entry, hentry, rfl⟩, hroute⟩ := hkey
  refine ⟨⟨entry, ?_, rfl⟩, hroute⟩
  have hsmall : entry ∈ ((sliceEntriesG kvs).take nextLen).take prefixLen := by
    simpa [List.take_take, Nat.min_eq_left hle] using hentry
  exact List.mem_of_mem_take hsmall

/-- The exact next slice entry belongs to the next routed prefix at its generated shard. -/
theorem routedPrefixKeysG_next_key {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    (i : Usize) (key : K) (value : V) (shard : Usize)
    (hi : i.val < (sliceEntriesG kvs).length)
    (hindex : Slice.index_usize kvs i = ok (key, value))
    (hroute : table.Table.shard_index HomeInst (emptyTableG (N := N) default home) key =
      ok shard) :
    key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs (i.val + 1) shard := by
  classical
  have hget : (sliceEntriesG kvs)[i.val]? = some (key, value) := by
    unfold Slice.index_usize at hindex
    rw [Slice.getElem?_Usize_eq] at hindex
    split at hindex <;> simp_all [sliceEntriesG]
  have hentry : (sliceEntriesG kvs)[i.val] = (key, value) := by
    rw [List.getElem?_eq_getElem hi] at hget
    exact Option.some.inj hget
  have htake : (sliceEntriesG kvs).take (i.val + 1) =
      (sliceEntriesG kvs).take i.val ++ [(key, value)] := by
    rw [List.take_succ_eq_append_getElem hi, hentry]
  simp [routedPrefixKeysG, htake, hroute]

private theorem coverage_getElem?_of_index {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

private theorem coverage_index_bound {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    i.val < a.val.length := by
  have hget := coverage_getElem?_of_index a i x h
  by_contra hn
  rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at hget
  simp at hget

private theorem coverage_replace_entry_cases {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize) (slots : Array (Option (K × V)) N)
    (oldCell : Option (K × V)) (key : K) (new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok oldCell)
    (s j : Usize) (current : Array (Option (K × V)) N) (stored : K) (storedValue : V)
    (hs : Array.index_usize (replacePresentG t shard slots i key new).shards s =
      ok (some current))
    (hj : Array.index_usize current j = ok (some (stored, storedValue))) :
    (s = shard ∧ j = i ∧ stored = key) ∨
      ∃ oldSlots oldValue, Array.index_usize t.shards s = ok (some oldSlots) ∧
        Array.index_usize oldSlots j = ok (some (stored, oldValue)) := by
  have hshardBound := coverage_index_bound t.shards shard (some slots) houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots.set i (some (key, new)) := by
      have hs' := coverage_getElem?_of_index
        (replacePresentG t shard slots i key new).shards shard (some current) hs
      simp only [replacePresentG] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    by_cases hji : j = i
    · subst j
      have hnew := coverage_getElem?_of_index (slots.set i (some (key, new))) i
        (some (stored, storedValue)) hj
      have hi := coverage_index_bound slots i oldCell hslot
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi] at hnew
      have hpair : (stored, storedValue) = (key, new) :=
        Option.some.inj (Option.some.inj hnew.symm)
      exact Or.inl ⟨rfl, rfl, congrArg Prod.fst hpair⟩
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

private theorem physicalBuild_getElem?_of_index {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

/-- The real all-empty constructor has no occupied slot, hence satisfies state-local provenance
    at the empty prefix without any routing or capacity premise. -/
theorem emptyTableG_physicalKeyCoverageAtG {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V)) :
    PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs 0
      (emptyTableG (N := N) default home) := by
  set_option maxRecDepth 4096 in
  intro shard slot slots key value houter _
  have hget := physicalBuild_getElem?_of_index
    (emptyTableG (K := K) (N := N) default home).shards shard (some slots) houter
  change (List.replicate 256 (none : Option (Array (Option (K × V)) N)))[shard.val]? =
    some (some slots) at hget
  by_cases hbound : shard.val < 256
  · rw [List.getElem?_eq_getElem (by simpa using hbound), List.getElem_replicate] at hget
    simp at hget
  · rw [List.getElem?_eq_none (by simpa using Nat.not_lt.mp hbound)] at hget
    simp at hget

private theorem physicalKeyCoverageAtG_replacePresent {K V H : Type} {N : Usize}
    [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    {prefixLen nextLen : Nat} (hle : prefixLen ≤ nextLen)
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (oldCell : Option (K × V))
    (key : K) (new : V)
    (hcoverage : PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs prefixLen t)
    (hnew : key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs nextLen shard)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok oldCell) :
    PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs nextLen
      (replacePresentG t shard slots i key new) := by
  intro s j current stored storedValue hs hj
  rcases coverage_replace_entry_cases t shard i slots oldCell key new houter hslot
      s j current stored storedValue hs hj with hupdated | hold
  · rcases hupdated with ⟨rfl, rfl, rfl⟩
    exact hnew
  · obtain ⟨oldSlots, oldValue, hsOld, hjOld⟩ := hold
    exact routedPrefixKeysG_mono HomeInst default home kvs hle s
      (hcoverage s j oldSlots stored oldValue hsOld hjOld)

/-- Replacing a present key preserves state-local provenance at every extending prefix. -/
theorem physicalKeyCoverageAtG_replacePresent_present {K V H : Type} {N : Usize}
    [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    {prefixLen nextLen : Nat} (hle : prefixLen ≤ nextLen)
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (old new : V)
    (hcoverage : PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs prefixLen t)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs nextLen
      (replacePresentG t shard slots i key new) := by
  apply physicalKeyCoverageAtG_replacePresent HomeInst default home kvs hle t shard i slots
    (some (key, old)) key new hcoverage
  · exact routedPrefixKeysG_mono HomeInst default home kvs hle shard
      (hcoverage shard i slots key old houter hslot)
  · exact houter
  · exact hslot

/-- Writing a fresh key into a vacant populated slot preserves state-local provenance once the
    routed key is known to belong to the extending prefix. -/
theorem physicalKeyCoverageAtG_replacePresent_vacant {K V H : Type} {N : Usize}
    [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    {prefixLen nextLen : Nat} (hle : prefixLen ≤ nextLen)
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (new : V)
    (hcoverage : PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs prefixLen t)
    (hnew : key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs nextLen shard)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs nextLen
      (replacePresentG t shard slots i key new) :=
  physicalKeyCoverageAtG_replacePresent HomeInst default home kvs hle t shard i slots none
    key new hcoverage hnew houter hslot

private theorem coverage_insertEmpty_entry_cases {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V)
    (hN : 0#usize < N) (houter : Array.index_usize t.shards shard = ok none)
    (s j : Usize) (current : Array (Option (K × V)) N) (stored : K) (storedValue : V)
    (hs : Array.index_usize (insertEmptyShardG t shard key new).shards s = ok (some current))
    (hj : Array.index_usize current j = ok (some (stored, storedValue))) :
    (s = shard ∧ j = 0#usize ∧ stored = key) ∨
      ∃ oldSlots oldValue, Array.index_usize t.shards s = ok (some oldSlots) ∧
        Array.index_usize oldSlots j = ok (some (stored, oldValue)) := by
  have hshardBound := coverage_index_bound t.shards shard none houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = (Array.repeat N none).set 0#usize (some (key, new)) := by
      have hs' := coverage_getElem?_of_index
        (insertEmptyShardG t shard key new).shards shard (some current) hs
      simp only [insertEmptyShardG] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    have hjN : j.val < N.val := by
      simpa [((Array.repeat N none).set 0#usize (some (key, new))).property] using
        coverage_index_bound ((Array.repeat N none).set 0#usize (some (key, new))) j
          (some (stored, storedValue)) hj
    have hj0 : j = 0#usize := by
      by_contra hjne
      have hjneVal : j.val ≠ 0 := fun he => hjne (UScalar.eq_of_val_eq he)
      have hg := coverage_getElem?_of_index
        ((Array.repeat N none).set 0#usize (some (key, new))) j
        (some (stored, storedValue)) hj
      rw [Array.set_val_eq] at hg
      simp [Array.repeat_val, hjN, hjneVal] at hg
    subst j
    have hg := coverage_getElem?_of_index
      ((Array.repeat N none).set 0#usize (some (key, new))) 0#usize
      (some (stored, storedValue)) hj
    rw [Array.set_val_eq, Array.repeat_val,
      List.getElem?_set_self (by simpa using hN)] at hg
    have hpair : (stored, storedValue) = (key, new) :=
      Option.some.inj (Option.some.inj hg.symm)
    exact Or.inl ⟨rfl, rfl, congrArg Prod.fst hpair⟩
  · right
    have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [insertEmptyShardG, Array.index_usize, Array.set_val_eq, hne] using hs
    exact ⟨current, storedValue, hsOld, hj⟩

/-- Installing the first slot of an empty shard preserves state-local provenance once the routed
    key is known to belong to the extending prefix. -/
theorem physicalKeyCoverageAtG_insertEmptyShard {K V H : Type} {N : Usize}
    [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    {prefixLen nextLen : Nat} (hle : prefixLen ≤ nextLen)
    (t : table.Table K V H N) (shard : Usize) (key : K) (new : V)
    (hcoverage : PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs prefixLen t)
    (hnew : key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs nextLen shard)
    (hN : 0#usize < N) (houter : Array.index_usize t.shards shard = ok none) :
    PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs nextLen
      (insertEmptyShardG t shard key new) := by
  intro s j current stored storedValue hs hj
  rcases coverage_insertEmpty_entry_cases t shard key new hN houter
      s j current stored storedValue hs hj with hupdated | hold
  · rcases hupdated with ⟨rfl, rfl, rfl⟩
    exact hnew
  · obtain ⟨oldSlots, oldValue, hsOld, hjOld⟩ := hold
    exact routedPrefixKeysG_mono HomeInst default home kvs hle s
      (hcoverage s j oldSlots stored oldValue hsOld hjOld)

/-- Concrete empty stream at every element type. -/
def emptySliceG {α : Type} : Slice α := Slice.from [] (by simp)

/-- A strict occupied-cell count rules out the raw full-shard alternative. -/
private theorem occupiedCountG_not_full {K V : Type} {N : Usize}
    (slots : Array (Option (K × V)) N) (hcount : OccupiedCountG slots < N.val) :
    ¬ FullShardG slots := by
  classical
  intro hfull
  let allOccupied : Fin N.val →
      { position : Fin N.val // OccupiedPositionG slots position } := fun position =>
    ⟨position, by
      let index : Usize := ⟨position.val, by scalar_tac⟩
      have hindex : index < N := position.isLt
      obtain ⟨entry, hentry⟩ := hfull index hindex
      exact ⟨entry.1, entry.2, hentry⟩⟩
  have hinjective : Function.Injective allOccupied := by
    intro left right heq
    exact Subtype.ext_iff.mp heq
  have hcard : N.val ≤ OccupiedCountG slots := by
    simpa [OccupiedCountG, allOccupied] using
      Fintype.card_le_of_injective allOccupied hinjective
  omega

/-- Full-slot no-ghost coverage and the distinct routed-key bound imply the strict physical count
    bound.  `GlobalKeysDistinctG` supplies the injection from occupied positions to stream keys. -/
theorem prefixPhysicalCountBoundG_of_distinctRoutedKeysPerShardG
    {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (mergeFn : V → V → V) (home : H) (default : V) (kvs : Slice (K × V))
    (hN : 0#usize < N)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs)
    (hcoverage : PrefixPhysicalKeyCoverageG (N := N) cmp HomeInst mergeFn home default kvs) :
    PrefixPhysicalCountBoundG (N := N) cmp HomeInst mergeFn home default kvs := by
  classical
  refine ⟨hN, ?_⟩
  intro i state hi hhome hinv shard slots houter
  let Occupied := { position : Fin N.val // OccupiedPositionG slots position }
  let physicalKey : Occupied → K := fun position => Classical.choose position.property
  let physicalValue : (position : Occupied) → V := fun position =>
    Classical.choose (Classical.choose_spec position.property)
  have hphysicalSlot : ∀ position : Occupied,
      Array.index_usize slots ⟨position.val.val, by scalar_tac⟩ =
        ok (some (physicalKey position, physicalValue position)) := by
    intro position
    exact Classical.choose_spec (Classical.choose_spec position.property)
  let intoStream : Occupied →
      { key // key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs i.val shard } :=
    fun position => ⟨physicalKey position, hcoverage i state hi hhome hinv shard
      ⟨position.val.val, by scalar_tac⟩ slots (physicalKey position) (physicalValue position)
      houter (hphysicalSlot position)⟩
  have hIntoInjective : Function.Injective intoStream := by
    intro left right heq
    have hkeyEq : physicalKey left = physicalKey right := Subtype.ext_iff.mp heq
    have hpositions := hinv.2.1 shard ⟨left.val.val, by scalar_tac⟩ shard
      ⟨right.val.val, by scalar_tac⟩ slots slots (physicalKey left) (physicalValue left)
      (physicalValue right) houter (hphysicalSlot left) houter
      (by simpa [hkeyEq] using hphysicalSlot right)
    apply Subtype.ext
    apply Fin.ext
    exact congrArg UScalar.val hpositions.2
  have hcard : OccupiedCountG slots ≤
      (routedPrefixKeysG (N := N) HomeInst default home kvs i.val shard).card := by
    simpa [OccupiedCountG, Occupied, intoStream] using
      Fintype.card_le_of_injective intoStream hIntoInjective
  exact Nat.lt_of_le_of_lt hcard (hdistinct i.val shard hi)

/-- State-local provenance and physical key uniqueness bound the occupied positions of one reached
    shard by the distinct routed keys in its exact prefix. -/
private theorem occupiedCountG_lt_of_physicalKeyCoverageAtG
    {K V H : Type} {N : Usize} [DecidableEq K]
    (HomeInst : table.Home H K) (default : V) (home : H) (kvs : Slice (K × V))
    (prefixLen : Nat) (state : table.Table K V H N)
    (hinv : GlobalKeysDistinctG state)
    (hcoverage : PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs prefixLen state)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs)
    (hi : prefixLen < (sliceEntriesG kvs).length)
    (shard : Usize) (slots : Array (Option (K × V)) N)
    (houter : Array.index_usize state.shards shard = ok (some slots)) :
    OccupiedCountG slots < N.val := by
  classical
  let Occupied := { position : Fin N.val // OccupiedPositionG slots position }
  let physicalKey : Occupied → K := fun position => Classical.choose position.property
  let physicalValue : (position : Occupied) → V := fun position =>
    Classical.choose (Classical.choose_spec position.property)
  have hphysicalSlot : ∀ position : Occupied,
      Array.index_usize slots ⟨position.val.val, by scalar_tac⟩ =
        ok (some (physicalKey position, physicalValue position)) := by
    intro position
    exact Classical.choose_spec (Classical.choose_spec position.property)
  let intoStream : Occupied →
      { key // key ∈ routedPrefixKeysG (N := N) HomeInst default home kvs prefixLen shard } :=
    fun position => ⟨physicalKey position, hcoverage shard
      ⟨position.val.val, by scalar_tac⟩ slots (physicalKey position) (physicalValue position)
      houter (hphysicalSlot position)⟩
  have hIntoInjective : Function.Injective intoStream := by
    intro left right heq
    have hkeyEq : physicalKey left = physicalKey right := Subtype.ext_iff.mp heq
    have hpositions := hinv shard ⟨left.val.val, by scalar_tac⟩ shard
      ⟨right.val.val, by scalar_tac⟩ slots slots (physicalKey left) (physicalValue left)
      (physicalValue right) houter (hphysicalSlot left) houter
      (by simpa [hkeyEq] using hphysicalSlot right)
    apply Subtype.ext
    apply Fin.ext
    exact congrArg UScalar.val hpositions.2
  have hcard : OccupiedCountG slots ≤
      (routedPrefixKeysG (N := N) HomeInst default home kvs prefixLen shard).card := by
    simpa [OccupiedCountG, Occupied, intoStream] using
      Fintype.card_le_of_injective intoStream hIntoInjective
  exact Nat.lt_of_le_of_lt hcard (hdistinct prefixLen shard hi)

/-- The strict physical count bound derives every routed one-step capacity witness. -/
theorem buildCapacityG_of_prefixPhysicalCountBoundG
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (home : H) (default : V) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hcount : PrefixPhysicalCountBoundG (N := N) cmp HomeInst mergeFn home default kvs) :
    BuildCapacityG (N := N) cmp HomeInst mergeFn home default kvs := by
  intro i state key value hi _hindex hhome hinv
  have lawsState : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst state.home merge
      homeFn mergeFn := by simpa [hhome] using laws
  obtain ⟨shard, hroute, hpresent | habsent⟩ :=
    table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst state merge homeFn mergeFn
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key lawsState hinv
  · obtain ⟨slots, found, old, _, houter, hslot, hunique⟩ := hpresent
    exact ⟨shard, hroute, Or.inl ⟨slots, found, old, houter, hslot, hunique⟩⟩
  · obtain ⟨habsentKey, _, houter | ⟨slots, houter, hvacant | hfull⟩⟩ := habsent
    · exact ⟨shard, hroute, Or.inr ⟨habsentKey, Or.inl ⟨hcount.1, houter⟩⟩⟩
    · exact ⟨shard, hroute, Or.inr ⟨habsentKey, Or.inr ⟨slots, hvacant.choose, houter,
        hvacant.choose_spec⟩⟩⟩
    · have hstrict := hcount.2 i state hi hhome hinv shard slots houter
      exact (occupiedCountG_not_full slots hstrict hfull).elim

/-- Stream distinctness plus full-slot no-ghost coverage compose directly to build capacity. -/
theorem buildCapacityG_of_distinctRoutedKeysPerShardG
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (home : H) (default : V) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hN : 0#usize < N)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs)
    (hcoverage : PrefixPhysicalKeyCoverageG (N := N) cmp HomeInst mergeFn home default kvs) :
    BuildCapacityG (N := N) cmp HomeInst mergeFn home default kvs :=
  buildCapacityG_of_prefixPhysicalCountBoundG cmp cloneK cloneV HomeInst MergeInst merge homeFn
    mergeFn home default kvs laws
    (prefixPhysicalCountBoundG_of_distinctRoutedKeysPerShardG cmp HomeInst mergeFn home default
      kvs hN hdistinct hcoverage)

/-- Every real stream-prefix insertion is capacity-admissible at a physically represented state.
    The exact slice index and current home witness prevent admission from ranging over a different
    stream, route, or callback state. -/
def BuildAdmissibleG {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V) (HomeInst : table.Home H K)
    (MergeInst : table.Merge F V) (merge : F) (mergeFn : V → V → V) (home : H)
    (default : V) (kvs : Slice (K × V)) : Prop :=
  ∀ (i : Usize) (state : table.Table K V H N) (key : K) (value : V),
    i.val < (sliceEntriesG kvs).length →
    Slice.index_usize kvs i = ok (key, value) →
    state.home = home →
    PhysicalInvG cmp HomeInst state
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) →
    ∃ next, table.Table.insert_with cmp cloneV HomeInst MergeInst state merge key value =
      ok (core.result.Result.Ok (), next)

/-- Physical routed capacity discharges the operational success premise without assuming an
    insertion result. -/
theorem buildCapacityG_implies_buildAdmissibleG
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (home : H) (default : V) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hcapacity : BuildCapacityG (N := N) cmp HomeInst mergeFn home default kvs) :
    BuildAdmissibleG (N := N) cmp cloneV HomeInst MergeInst merge mergeFn home default kvs := by
  intro i state key value hi hindex hhome hinv
  obtain ⟨capacityShard, hcapacityRoute, hcapacityBranch⟩ :=
    hcapacity i state key value hi hindex hhome hinv
  have lawsState : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst state.home merge
      homeFn mergeFn := by simpa [hhome] using laws
  have hbranch :=
    table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst state merge homeFn mergeFn
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key lawsState hinv
  rcases hcapacityBranch with hpresentCapacity | habsentCapacity
  · rcases hbranch with ⟨actualShard, hactualRoute, hpresent | habsent⟩
    · have hshardEq : actualShard = capacityShard :=
        Aeneas.Std.Result.ok_injective (hactualRoute.symm.trans hcapacityRoute)
      subst actualShard
      obtain ⟨slots, found, old, hfind, houter, hslot, _⟩ := hpresent
      exact ⟨replacePresentG state capacityShard slots found key (mergeFn value old),
        insert_with_present_callback_exact cmp cloneK cloneV HomeInst MergeInst state merge
          homeFn mergeFn key key value old capacityShard found slots lawsState hcapacityRoute hfind
          houter hslot⟩
    · have hshardEq : actualShard = capacityShard :=
        Aeneas.Std.Result.ok_injective (hactualRoute.symm.trans hcapacityRoute)
      subst actualShard
      obtain ⟨habsentKey, _, _⟩ := habsent
      obtain ⟨slots, found, old, hunique⟩ := hpresentCapacity
      exact (habsentKey capacityShard found slots old hunique.1 hunique.2.1).elim
  · rcases hbranch with ⟨actualShard, hactualRoute, hpresent | habsent⟩
    · have hshardEq : actualShard = capacityShard :=
        Aeneas.Std.Result.ok_injective (hactualRoute.symm.trans hcapacityRoute)
      subst actualShard
      obtain ⟨slots, found, old, _, houter, hslot, _⟩ := hpresent
      exact (habsentCapacity.1 capacityShard found slots old houter hslot).elim
    · have hshardEq : actualShard = capacityShard :=
        Aeneas.Std.Result.ok_injective (hactualRoute.symm.trans hcapacityRoute)
      subst actualShard
      obtain ⟨_, hfind, _⟩ := habsent
      rcases habsentCapacity.2 with ⟨hN, houter⟩ | ⟨slots, first, houter, hfirst⟩
      · obtain ⟨routed, hrouted, hbound⟩ :=
          shard_index_callback_spec HomeInst state homeFn key (lawsState.home_total key)
        have hroutedEq : routed = capacityShard :=
          Aeneas.Std.Result.ok_injective (hrouted.symm.trans hcapacityRoute)
        subst routed
        have hshardBound : capacityShard < 256#usize := by simpa using hbound
        exact ⟨insertEmptyShardG state capacityShard key (mergeFn value state.default),
          insert_with_empty_shard_callback_exact cmp cloneK cloneV HomeInst MergeInst state merge
            homeFn mergeFn key value capacityShard lawsState hcapacityRoute hfind houter hN
            hshardBound⟩
      · obtain ⟨hfirstBound, hprefix, hslot⟩ := hfirst
        exact ⟨replacePresentG state capacityShard slots first key
            (mergeFn value state.default),
          insert_with_populated_vacant_callback_exact cmp cloneK cloneV HomeInst MergeInst state
            merge homeFn mergeFn key value capacityShard first slots lawsState hcapacityRoute hfind
            houter hfirstBound hprefix hslot⟩

/-- The concrete empty stream has a checked physical capacity witness at arbitrary types. -/
theorem emptySliceG_buildCapacityG {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (mergeFn : V → V → V) (home : H) (default : V) :
    BuildCapacityG (N := N) cmp HomeInst mergeFn home default
      (emptySliceG (α := K × V)) := by
  intro i state key value hi
  simp [sliceEntriesG, emptySliceG] at hi

private theorem insert_with_callback_home_preserved {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K)
    (cloneV : core.clone.Clone V) (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value : V)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h) (next : table.Table K V H N)
    (hnext : table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Ok (), next)) : next.home = t.home := by
  obtain ⟨result, hresult, houtcome⟩ := insert_with_callback_result_exact cmp cloneK cloneV
    HomeInst MergeInst t merge homeFn mergeFn h key value laws hinv
  have hresultEq : result = (core.result.Result.Ok (), next) :=
    Aeneas.Std.Result.ok_injective (hresult.symm.trans hnext)
  subst result
  rcases houtcome with ⟨shard, slots, i, old, _, hstate⟩ |
    ⟨shard, hN, houter, hstate⟩ | ⟨shard, slots, first, houter, hfirst, hstate⟩ | hstate
  · have htables := Prod.mk.inj hstate
    simpa [replacePresentG] using congrArg table.Table.home htables.2
  · have htables := Prod.mk.inj hstate
    simpa [insertEmptyShardG] using congrArg table.Table.home htables.2
  · have htables := Prod.mk.inj hstate
    simpa [replacePresentG] using congrArg table.Table.home htables.2
  · simp at hstate

/-- The generated loop carries both the physical refinement and state-local key provenance.  The
    routed-key bound excludes the only refusing branch at each actually reached prefix. -/
private theorem table_build_loop_physicalInv_of_stateLocalCoverage
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (initial : table.Table K V H N) (start : Usize)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hN : 0#usize < N)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitialHome : initial.home = home)
    (hinitial : PhysicalInvG cmp HomeInst initial
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take start.val)))
    (hinitialCoverage : PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs start.val
      initial) :
    ∃ final,
      table.Table.build_loop cmp cloneV HomeInst MergeInst cloneK merge kvs initial start =
        ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ∧
      final.home = home ∧
      PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs
        (sliceEntriesG kvs).length final := by
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body cmp cloneV HomeInst MergeInst cloneK merge
        kvs state i)
      (initial, start)
      ⦃ result => ∃ final, result = core.result.Result.Ok final ∧
        PhysicalInvG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ∧
        final.home = home ∧
        PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs
          (sliceEntriesG kvs).length final ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : table.Table K V H N × Usize => (sliceEntriesG kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntriesG kvs).length ∧ p.1.home = home ∧
        PhysicalInvG cmp HomeInst p.1
          (orderedFoldG mergeFn default ((sliceEntriesG kvs).take p.2.val)) ∧
        PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs p.2.val p.1)
      (post := fun result => ∃ final, result = core.result.Result.Ok final ∧
        PhysicalInvG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ∧
        final.home = home ∧
        PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs
          (sliceEntriesG kvs).length final)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hhome, hphysical, hcoverage⟩ := hp
      dsimp only at hi hhome hphysical hcoverage ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        simp only [laws.cloneK_identity, laws.cloneV_identity]
        step
        have hiList : i.val < (sliceEntriesG kvs).length := by
          rw [sliceEntriesG_length]
          exact hlt
        have hindex : Slice.index_usize kvs i = ok (key, value) := by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
        have lawsState : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst state.home merge
            homeFn mergeFn := by simpa [hhome] using laws
        obtain ⟨shard, hroute, hbranch⟩ :=
          table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst state merge homeFn
            mergeFn (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key
            lawsState hphysical
        have hrouteEmpty : table.Table.shard_index HomeInst
            (emptyTableG (N := N) default home) key = ok shard := by
          simpa [table.Table.shard_index, emptyTableG, hhome] using hroute
        have hnewMember := routedPrefixKeysG_next_key HomeInst default home kvs i key value shard
          hiList hindex hrouteEmpty
        obtain ⟨next, hinsert, hphysicalNext, hcoverageNext⟩ :
            ∃ next,
              table.Table.insert_with cmp cloneV HomeInst MergeInst state merge key value =
                ok (core.result.Result.Ok (), next) ∧
              PhysicalInvG cmp HomeInst next
                (insertWith mergeFn key value
                  (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val))) ∧
              PhysicalKeyCoverageAtG (N := N) HomeInst default home kvs (i.val + 1) next := by
          rcases hbranch with hpresent | habsent
          · obtain ⟨slots, found, old, hfind, houter, hslot, hunique⟩ := hpresent
            let next := replacePresentG state shard slots found key (mergeFn value old)
            refine ⟨next, ?_, ?_, ?_⟩
            · exact insert_with_present_callback_exact cmp cloneK cloneV HomeInst MergeInst state
                merge homeFn mergeFn key key value old shard found slots lawsState hroute hfind
                houter hslot
            · exact physicalInvG_replacePresent cmp cloneK cloneV HomeInst MergeInst state merge
                homeFn mergeFn _ key value old shard found slots lawsState hphysical
                ⟨houter, hslot, hunique⟩
            · exact physicalKeyCoverageAtG_replacePresent_present HomeInst default home kvs
                (Nat.le_succ i.val) state shard found slots key old (mergeFn value old) hcoverage
                houter hslot
          · obtain ⟨habsentKey, hfind, houterEmpty | ⟨slots, houter, hvacant | hfull⟩⟩ :=
              habsent
            · obtain ⟨routed, hrouted, hbound⟩ :=
                shard_index_callback_spec HomeInst state homeFn key (lawsState.home_total key)
              have hroutedEq : routed = shard :=
                Aeneas.Std.Result.ok_injective (hrouted.symm.trans hroute)
              subst routed
              let next := insertEmptyShardG state shard key (mergeFn value state.default)
              refine ⟨next, ?_, ?_, ?_⟩
              · exact insert_with_empty_shard_callback_exact cmp cloneK cloneV HomeInst MergeInst
                  state merge homeFn mergeFn key value shard lawsState hroute hfind houterEmpty hN
                  (by simpa using hbound)
              · exact physicalInvG_empty_shard cmp cloneK cloneV HomeInst MergeInst state merge
                  homeFn mergeFn _ key value shard lawsState hphysical habsentKey hroute hfind
                  houterEmpty hN
              · exact physicalKeyCoverageAtG_insertEmptyShard HomeInst default home kvs
                  (Nat.le_succ i.val) state shard key (mergeFn value state.default) hcoverage
                  hnewMember hN houterEmpty
            · obtain ⟨first, hfirst⟩ := hvacant
              let next := replacePresentG state shard slots first key
                (mergeFn value state.default)
              refine ⟨next, ?_, ?_, ?_⟩
              · exact insert_with_populated_vacant_callback_exact cmp cloneK cloneV HomeInst
                  MergeInst state merge homeFn mergeFn key value shard first slots lawsState
                  hroute hfind houter hfirst.1 hfirst.2.1 hfirst.2.2
              · exact physicalInvG_populated_vacant cmp cloneK cloneV HomeInst MergeInst state
                  merge homeFn mergeFn _ key value shard first slots lawsState hphysical
                  habsentKey hroute hfind houter hfirst
              · exact physicalKeyCoverageAtG_replacePresent_vacant HomeInst default home kvs
                  (Nat.le_succ i.val) state shard first slots key (mergeFn value state.default)
                  hcoverage hnewMember houter hfirst.2.2
            · have hstrict := occupiedCountG_lt_of_physicalKeyCoverageAtG HomeInst default home
                kvs i.val state hphysical.2.1 hcoverage hdistinct hiList shard slots houter
              exact (occupiedCountG_not_full slots hstrict hfull).elim
        have hhomeNext : next.home = home := by
          calc
            next.home = state.home := insert_with_callback_home_preserved cmp cloneK cloneV
              HomeInst MergeInst state merge homeFn mergeFn
              (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key value lawsState
              hphysical next hinsert
            _ = home := hhome
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntriesG kvs).take (i.val + 1) =
            (sliceEntriesG kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post] using hhomeNext
        · simpa [r_post, orderedFoldG, htake] using hphysicalNext
        · simpa [r_post] using hcoverageNext
        · simp only [r_post]
          omega
      · rw [if_neg hlt]
        simp only [Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntriesG kvs).length := by
          rw [sliceEntriesG_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        refine ⟨state, rfl, ?_, hhome, ?_⟩
        · simpa [hieq] using hphysical
        · simpa [hieq] using hcoverage
    · exact ⟨hstart, hinitialHome, hinitial, hinitialCoverage⟩
  obtain ⟨result, hresult, final, hfinal, hphysical, hhome, hcoverage⟩ :=
    Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, hphysical, hhome, hcoverage⟩

private theorem table_build_loop_physicalInv_of_buildAdmissibleG
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (initial : table.Table K V H N) (start : Usize)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hstart : start.val ≤ (sliceEntriesG kvs).length)
    (hinitialHome : initial.home = home)
    (hinitial : PhysicalInvG cmp HomeInst initial
      (orderedFoldG mergeFn default ((sliceEntriesG kvs).take start.val)))
    (hadmissible : BuildAdmissibleG (N := N) cmp cloneV HomeInst MergeInst merge mergeFn home
      default kvs) :
    ∃ final,
      table.Table.build_loop cmp cloneV HomeInst MergeInst cloneK merge kvs initial start =
        ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ∧
      final.home = home := by
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body cmp cloneV HomeInst MergeInst cloneK merge
        kvs state i)
      (initial, start)
      ⦃ result => ∃ final, result = core.result.Result.Ok final ∧
        PhysicalInvG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ∧
        final.home = home ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : table.Table K V H N × Usize => (sliceEntriesG kvs).length - p.2.val)
      (inv := fun p => p.2.val ≤ (sliceEntriesG kvs).length ∧ p.1.home = home ∧
        PhysicalInvG cmp HomeInst p.1
          (orderedFoldG mergeFn default ((sliceEntriesG kvs).take p.2.val)))
      (post := fun result => ∃ final, result = core.result.Result.Ok final ∧
        PhysicalInvG cmp HomeInst final (orderedFoldG mergeFn default (sliceEntriesG kvs)) ∧
        final.home = home)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hi, hhome, hphysical⟩ := hp
      dsimp only at hi hhome hphysical ⊢
      unfold table.Table.build_loop.body
      by_cases hlt : i < Slice.len kvs
      · rw [if_pos hlt]
        simp only [laws.cloneK_identity, laws.cloneV_identity]
        step
        have hiList : i.val < (sliceEntriesG kvs).length := by
          rw [sliceEntriesG_length]
          exact hlt
        have hindex : Slice.index_usize kvs i = ok (key, value) := by
          have hib : i.val < kvs.val.length := by simpa [Slice.len] using hlt
          unfold Slice.index_usize
          rw [Slice.getElem?_Usize_eq, List.getElem?_eq_getElem hib, ← key_post]
        obtain ⟨next, hinsert⟩ := hadmissible i state key value hiList hindex hhome hphysical
        have lawsState : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst state.home merge
            homeFn mergeFn := by simpa [hhome] using laws
        have hpost := insert_with_callback_admitted cmp cloneK cloneV HomeInst MergeInst state
          merge homeFn mergeFn
          (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key value lawsState
          hphysical
        rw [hinsert, Aeneas.Std.WP.spec_ok] at hpost
        have hphysicalNext : PhysicalInvG cmp HomeInst next
            (insertWith mergeFn key value
              (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val))) := by
          simpa [CallbackInsertPostG] using hpost
        have hhomeNext : next.home = home := by
          calc
            next.home = state.home := insert_with_callback_home_preserved cmp cloneK cloneV
              HomeInst MergeInst state merge homeFn mergeFn
              (orderedFoldG mergeFn default ((sliceEntriesG kvs).take i.val)) key value lawsState
              hphysical next hinsert
            _ = home := hhome
        rw [hinsert]
        simp only [bind_tc_ok]
        have htake : (sliceEntriesG kvs).take (i.val + 1) =
            (sliceEntriesG kvs).take i.val ++ [(key, value)] := by
          rw [List.take_succ_eq_append_getElem hiList]
          congr
          exact key_post.symm
        step
        refine ⟨?_, ?_, ?_, ?_⟩
        · simpa [r_post] using Nat.succ_le_iff.mpr hiList
        · simpa [r_post] using hhomeNext
        · simpa [r_post, orderedFoldG, htake] using hphysicalNext
        · simp only [r_post]
          omega
      · rw [if_neg hlt]
        simp only [Aeneas.Std.WP.spec_ok]
        have hieq : i.val = (sliceEntriesG kvs).length := by
          rw [sliceEntriesG_length]
          exact Nat.le_antisymm hi (Nat.not_lt.mp hlt)
        refine ⟨state, rfl, ?_, hhome⟩
        simpa [hieq] using hphysical
    · exact ⟨hstart, hinitialHome, hinitial⟩
  obtain ⟨result, hresult, final, hfinal, hphysical, hhome⟩ :=
    Aeneas.Std.WP.spec_imp_exists hs
  rw [hfinal] at hresult
  exact ⟨final, hresult, hphysical, hhome⟩

/-- Under explicit prefix/capacity admission, the real generated constructor and loop return the
    canonical ordered build and preserve the complete generic physical invariant. -/
theorem table_build_physicalInv_of_buildAdmissibleG {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K)
    (cloneV : core.clone.Clone V) (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hadmissible : BuildAdmissibleG (N := N) cmp cloneV HomeInst MergeInst merge mergeFn home
      default kvs) :
    ∃ final,
      table.Table.build N cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
        ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final
        (Apeiron.Table.build mergeFn default (sliceEntriesG kvs)) := by
  have hinitial : PhysicalInvG cmp HomeInst (emptyTableG (N := N) default home)
      (orderedFoldG mergeFn default []) := by
    have hconst : orderedFoldG mergeFn default ([] : List (K × V)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyTableG_physicalInv cmp cloneK cloneV HomeInst MergeInst default home merge homeFn
      mergeFn laws
  obtain ⟨final, hloop, hphysical, _⟩ :=
    table_build_loop_physicalInv_of_buildAdmissibleG cmp cloneK cloneV HomeInst MergeInst merge
      homeFn mergeFn default home kvs (emptyTableG (N := N) default home) 0#usize laws
      (Nat.zero_le _) rfl hinitial hadmissible
  unfold table.Table.build
  rw [table_new_eq_emptyTableG]
  simp only [bind_tc_ok]
  rw [hloop]
  exact ⟨final, rfl, hphysical⟩

/-- Generic routed-key distinctness and full-slot no-ghost coverage discharge the real build's
    physical/canonical result without an operational admission premise. -/
theorem table_build_physicalInv_of_distinctRoutedKeysPerShardG
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hN : 0#usize < N)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs)
    (hcoverage : PrefixPhysicalKeyCoverageG (N := N) cmp HomeInst mergeFn home default kvs) :
    ∃ final,
      table.Table.build N cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
        ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final
        (Apeiron.Table.build mergeFn default (sliceEntriesG kvs)) := by
  apply table_build_physicalInv_of_buildAdmissibleG cmp cloneK cloneV HomeInst MergeInst merge
    homeFn mergeFn default home kvs laws
  apply buildCapacityG_implies_buildAdmissibleG cmp cloneK cloneV HomeInst MergeInst merge
    homeFn mergeFn home default kvs laws
  exact buildCapacityG_of_distinctRoutedKeysPerShardG cmp cloneK cloneV HomeInst MergeInst merge
    homeFn mergeFn home default kvs laws hN hdistinct hcoverage

/-- Routed-key distinctness alone is the execution-independent capacity condition needed by the
    real nonempty generated build: state-local provenance is initialized by the constructor and
    preserved by each exact generated success branch. -/
theorem table_build_physicalInv_of_distinctRoutedKeysPerShardG_reachable
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hN : 0#usize < N)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs) :
    ∃ final,
      table.Table.build N cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
        ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final
        (Apeiron.Table.build mergeFn default (sliceEntriesG kvs)) := by
  have hinitial : PhysicalInvG cmp HomeInst (emptyTableG (N := N) default home)
      (orderedFoldG mergeFn default []) := by
    have hconst : orderedFoldG mergeFn default ([] : List (K × V)) = (fun _ => default) := rfl
    rw [hconst]
    exact emptyTableG_physicalInv cmp cloneK cloneV HomeInst MergeInst default home merge homeFn
      mergeFn laws
  obtain ⟨final, hloop, hphysical, _, _⟩ :=
    table_build_loop_physicalInv_of_stateLocalCoverage cmp cloneK cloneV HomeInst MergeInst merge
      homeFn mergeFn default home kvs (emptyTableG (N := N) default home) 0#usize laws hN
      hdistinct (Nat.zero_le _) rfl hinitial
      (emptyTableG_physicalKeyCoverageAtG HomeInst default home kvs)
  unfold table.Table.build
  rw [table_new_eq_emptyTableG]
  simp only [bind_tc_ok]
  rw [hloop]
  exact ⟨final, rfl, hphysical⟩

/-- The real generated build has a checked successful physical witness for the concrete empty
    stream at arbitrary key, value, home, merge, and capacity types. -/
theorem table_build_emptySliceG_physicalInv {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K)
    (cloneV : core.clone.Clone V) (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn) :
    ∃ final,
      table.Table.build N cmp cloneV HomeInst MergeInst cloneK merge default
          (emptySliceG (α := K × V)) home = ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final
        (Apeiron.Table.build mergeFn default
          (sliceEntriesG (emptySliceG (α := K × V)))) := by
  apply table_build_physicalInv_of_buildAdmissibleG cmp cloneK cloneV HomeInst MergeInst merge
    homeFn mergeFn default home (emptySliceG (α := K × V)) laws
  exact buildCapacityG_implies_buildAdmissibleG cmp cloneK cloneV HomeInst MergeInst merge
    homeFn mergeFn home default (emptySliceG (α := K × V)) laws
    (emptySliceG_buildCapacityG cmp HomeInst mergeFn home default)

/-- At zero capacity an empty stream returns the generated empty table, which retains the complete
    generic physical invariant. -/
theorem table_build_zero_of_empty {K V H F : Type} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hempty : (sliceEntriesG kvs).length = 0) :
    table.Table.build 0#usize cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
        ok (core.result.Result.Ok (emptyTableG (N := 0#usize) default home)) ∧
      PhysicalInvG cmp HomeInst (emptyTableG (N := 0#usize) default home)
        (Apeiron.Table.build mergeFn default (sliceEntriesG kvs)) := by
  have hentries : sliceEntriesG kvs = [] := List.eq_nil_of_length_eq_zero hempty
  constructor
  · unfold table.Table.build
    rw [table_new_eq_emptyTableG]
    simp only [bind_tc_ok]
    unfold table.Table.build_loop
    have hs : loop
        (fun (state, i) => table.Table.build_loop.body cmp cloneV HomeInst MergeInst cloneK merge
          kvs state i)
        (emptyTableG (N := 0#usize) default home, 0#usize)
        ⦃ r => r = core.result.Result.Ok (emptyTableG (N := 0#usize) default home) ⦄ := by
      apply loop.spec_decr_nat
        (measure := fun p : table.Table K V H 0#usize × Usize =>
          (sliceEntriesG kvs).length - p.2.val)
        (inv := fun p => p.1 = emptyTableG (N := 0#usize) default home ∧ p.2.val = 0)
      · intro p hp
        rcases p with ⟨state, i⟩
        obtain ⟨hstate, hi⟩ := hp
        dsimp only at hstate hi ⊢
        unfold table.Table.build_loop.body
        by_cases hlt : i < Slice.len kvs
        · exfalso
          have hsz : (Slice.len kvs).val = 0 := by
            rw [← sliceEntriesG_length]
            exact hempty
          have hlt' : i.val < (Slice.len kvs).val := hlt
          rw [hsz] at hlt'
          omega
        · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
          subst hstate
          rfl
      · exact ⟨rfl, rfl⟩
    obtain ⟨result, hresult, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
    rw [hpost] at hresult
    exact hresult
  · simpa [hentries, Apeiron.Table.build] using
      (emptyTableG_physicalInv cmp cloneK cloneV HomeInst MergeInst default home merge homeFn
        mergeFn laws)

/-- At zero capacity every nonempty stream reaches the first generated insertion and returns the
    exact `ShardFull` error without changing the constructor state. -/
theorem table_build_zero_refuses_of_nonempty {K V H F : Type} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hne : 0 < (sliceEntriesG kvs).length) :
    table.Table.build 0#usize cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
      ok (core.result.Result.Err table.StorageError.ShardFull) := by
  let initial := emptyTableG (K := K) (N := 0#usize) default home
  have hinitial : PhysicalInvG cmp HomeInst initial (fun _ => default) :=
    emptyTableG_physicalInv cmp cloneK cloneV HomeInst MergeInst default home merge homeFn mergeFn
      laws
  unfold table.Table.build
  rw [table_new_eq_emptyTableG]
  simp only [bind_tc_ok]
  unfold table.Table.build_loop
  have hs : loop
      (fun (state, i) => table.Table.build_loop.body cmp cloneV HomeInst MergeInst cloneK merge
        kvs state i)
      (initial, 0#usize)
      ⦃ r => r = core.result.Result.Err table.StorageError.ShardFull ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : table.Table K V H 0#usize × Usize =>
        (sliceEntriesG kvs).length - p.2.val)
      (inv := fun p => p.1 = initial ∧ p.2.val = 0)
    · intro p hp
      rcases p with ⟨state, i⟩
      obtain ⟨hstate, hi⟩ := hp
      dsimp only at hstate hi ⊢
      subst state
      unfold table.Table.build_loop.body
      have hlt : i < Slice.len kvs := by
        have hlen : 0 < (Slice.len kvs).val := by
          simpa [sliceEntriesG, Slice.len] using hne
        change i.val < (Slice.len kvs).val
        omega
      rw [if_pos hlt]
      simp only [laws.cloneK_identity, laws.cloneV_identity]
      step
      obtain ⟨shard, hroute, hpresent | habsent⟩ :=
        table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst initial merge homeFn
          mergeFn (fun _ => default) key laws hinitial
      · obtain ⟨slots, found, old, _, houter, hslot, _⟩ := hpresent
        have himpossible := emptyTableG_physicalKeyCoverageAtG HomeInst default home kvs shard
          found slots key old houter hslot
        simp [routedPrefixKeysG] at himpossible
      · rw [insert_with_zero_callback_exact cmp cloneV HomeInst MergeInst initial merge key value
          shard hroute habsent.2.1]
        simp
    · exact ⟨rfl, rfl⟩
  obtain ⟨result, hresult, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  rw [hpost] at hresult
  exact hresult

/-- Exact zero-capacity characterization: the generated generic build succeeds exactly for an
    empty stream and otherwise returns `ShardFull`. -/
theorem table_build_zero_ok_iff_empty {K V H F : Type} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn) :
    (∃ final,
      table.Table.build 0#usize cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
        ok (core.result.Result.Ok final)) ↔ (sliceEntriesG kvs).length = 0 := by
  constructor
  · rintro ⟨final, hfinal⟩
    by_contra hne
    have hpos : 0 < (sliceEntriesG kvs).length := Nat.pos_of_ne_zero hne
    rw [table_build_zero_refuses_of_nonempty cmp cloneK cloneV HomeInst MergeInst merge homeFn
      mergeFn default home kvs laws hpos] at hfinal
    have hbad := Aeneas.Std.Result.ok_injective hfinal
    cases hbad
  · intro hempty
    exact ⟨_, (table_build_zero_of_empty cmp cloneK cloneV HomeInst MergeInst merge homeFn mergeFn
      default home kvs laws hempty).1⟩

/-- A real generated `ShardFull` build result cannot coexist with the explicit prefix-admission
    witness.  This is one direction only: failure to supply that witness does not assert that the
    generated build refuses. -/
theorem table_build_shardFull_requires_not_buildAdmissibleG {K V H F : Type} {N : Usize}
    [DecidableEq K] (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K)
    (cloneV : core.clone.Clone V) (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hresult : table.Table.build N cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
      ok (core.result.Result.Err table.StorageError.ShardFull)) :
    ¬ BuildAdmissibleG (N := N) cmp cloneV HomeInst MergeInst merge mergeFn home default kvs := by
  intro hadmissible
  obtain ⟨final, hsuccess, _⟩ := table_build_physicalInv_of_buildAdmissibleG cmp cloneK cloneV
    HomeInst MergeInst merge homeFn mergeFn default home kvs laws hadmissible
  rw [hsuccess] at hresult
  simp at hresult

/-- Routed-key distinctness covers both capacities: at zero capacity it forces the stream empty;
    at positive capacity the reachable provenance theorem supplies the generated build witness. -/
theorem table_build_physicalInv_of_distinctRoutedKeysPerShardG_all
    {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (default : V) (home : H) (kvs : Slice (K × V))
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn)
    (hdistinct : DistinctRoutedKeysPerShardG (N := N) HomeInst default home kvs) :
    ∃ final,
      table.Table.build N cmp cloneV HomeInst MergeInst cloneK merge default kvs home =
        ok (core.result.Result.Ok final) ∧
      PhysicalInvG cmp HomeInst final
        (Apeiron.Table.build mergeFn default (sliceEntriesG kvs)) := by
  by_cases hN : 0#usize < N
  · exact table_build_physicalInv_of_distinctRoutedKeysPerShardG_reachable cmp cloneK cloneV
      HomeInst MergeInst merge homeFn mergeFn default home kvs laws hN hdistinct
  · have hzero : N = 0#usize := by scalar_tac
    subst N
    have hempty : (sliceEntriesG kvs).length = 0 := by
      by_contra hne
      have hpositive : 0 < (sliceEntriesG kvs).length := Nat.pos_of_ne_zero hne
      have hbad := hdistinct 0 0#usize hpositive
      simp at hbad
    exact ⟨_, (table_build_zero_of_empty cmp cloneK cloneV HomeInst MergeInst merge homeFn
      mergeFn default home kvs laws hempty).1,
      (table_build_zero_of_empty cmp cloneK cloneV HomeInst MergeInst merge homeFn mergeFn
        default home kvs laws hempty).2⟩

end Apeiron.TableRefinement
