import TableRefinementReachablePrefixExactN

/-! # Universal physical representation of the generated word table

`PhysicalInv` applies to every generated state at every fixed shard capacity.  It combines the
dense-prefix and full-key uniqueness invariant with home-shard placement and the canonical
extensional map.  The semantic component refines `Apeiron.Table.insertWith`; its read equations are
`Apeiron.Table.get_insertWith_self` and `Apeiron.Table.get_insertWith_ne`.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- Every stored full key is in the shard returned by the generated `WordHome` routing body. -/
def HomePlaced {N : Usize} (t : WordTableN N) : Prop :=
  ∀ (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (key value : I64),
    Array.index_usize t.shards shard = ok (some slots) →
    Array.index_usize slots i = ok (some (key, value)) →
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard

/-- Equal stored full keys identify one physical position across the complete 256-shard table. -/
def GlobalKeysDistinct {N : Usize} (t : WordTableN N) : Prop :=
  ∀ (leftShard leftSlot rightShard rightSlot : Usize)
      (leftSlots rightSlots : Array (Option (I64 × I64)) N) (key left right : I64),
    Array.index_usize t.shards leftShard = ok (some leftSlots) →
    Array.index_usize leftSlots leftSlot = ok (some (key, left)) →
    Array.index_usize t.shards rightShard = ok (some rightSlots) →
    Array.index_usize rightSlots rightSlot = ok (some (key, right)) →
    leftShard = rightShard ∧ leftSlot = rightSlot

/-- Universal physical shape: dense occupied prefixes, per-shard full-key uniqueness, home
    placement, and the resulting table-wide full-key uniqueness. -/
def PhysicalShape {N : Usize} (t : WordTableN N) : Prop :=
  TableInv2N t ∧ HomePlaced t ∧ GlobalKeysDistinct t

/-- Universal physical representation of the canonical total map `h`. -/
def PhysicalInv {N : Usize} (t : WordTableN N) (h : HMap I64 I64) : Prop :=
  PhysicalShape t ∧ ReadRep t h

private theorem getElem?_eq_some_of_index {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

private theorem index_of_getElem?_eq_some {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : a.val[i.val]? = some x) :
    Array.index_usize a i = ok x := by
  unfold Array.index_usize
  rw [Array.getElem?_Usize_eq, h]

/-- Home placement and per-shard uniqueness entail explicit table-wide uniqueness. -/
theorem globalKeysDistinct_of_tableInv2N_homePlaced {N : Usize} (t : WordTableN N)
    (hinv : TableInv2N t) (hhome : HomePlaced t) : GlobalKeysDistinct t := by
  intro ls li rs ri leftSlots rightSlots key left right hlo hli hro hri
  have hls := hhome ls li leftSlots key left hlo hli
  have hrs := hhome rs ri rightSlots key right hro hri
  have hs : ls = rs := Aeneas.Std.Result.ok_injective (hls.symm.trans hrs)
  subst rs
  have hslots : leftSlots = rightSlots := by
    exact Option.some.inj (Aeneas.Std.Result.ok_injective (hlo.symm.trans hro))
  subst rightSlots
  refine ⟨rfl, UScalar.eq_of_val_eq ?_⟩
  apply hinv.2 ls.val leftSlots (getElem?_eq_some_of_index t.shards ls (some leftSlots) hlo)
    key li.val ri.val left right
  · have h := getElem?_eq_some_of_index leftSlots li (some (key, left)) hli
    have : li.val < leftSlots.val.length := by
      by_contra hn
      rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
      simp at h
    simpa [leftSlots.property] using this
  · have h := getElem?_eq_some_of_index leftSlots ri (some (key, right)) hri
    have : ri.val < leftSlots.val.length := by
      by_contra hn
      rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
      simp at h
    simpa [leftSlots.property] using this
  · exact getElem?_eq_some_of_index leftSlots li (some (key, left)) hli
  · exact getElem?_eq_some_of_index leftSlots ri (some (key, right)) hri

/-- Exact decomposition of the universal representation into all physical clauses and the
    canonical extensional key/value relation. -/
theorem physicalInv_iff {N : Usize} (t : WordTableN N) (h : HMap I64 I64) :
    PhysicalInv t h ↔ TableInv2N t ∧ HomePlaced t ∧ GlobalKeysDistinct t ∧ ReadRep t h := by
  simp only [PhysicalInv, PhysicalShape, and_assoc]

/-- Every state has an extensional `ReadRep`; therefore existential `ReadRep` alone cannot
    characterize physical validity. -/
theorem readRep_exists {N : Usize} (t : WordTableN N) : ∃ h, ReadRep t h := by
  refine ⟨stateRead (generatedLocator t) t, ?_⟩
  intro key
  exact table_get_eq_stateRead t key

/-- A malformed one-slot state used to separate extensional readability from physical validity. -/
def offHomeTable : WordTableN 1#usize where
  default := 0#i64
  home := ()
  shards := (Array.repeat 256#usize none).set 1#usize
    (some ((Array.repeat 1#usize none).set 0#usize (some (0#i64, 1#i64))))

/-- The malformed witness stores key zero in shard one, although `WordHome` routes it to shard zero. -/
theorem offHomeTable_not_homePlaced : ¬ HomePlaced offHomeTable := by
  intro hhome
  have houter : Array.index_usize offHomeTable.shards 1#usize =
      ok (some ((Array.repeat 1#usize none).set 0#usize (some (0#i64, 1#i64)))) := by
    set_option maxRecDepth 4096 in
      simp [offHomeTable, Array.index_usize, Array.repeat_val, Array.set_val_eq]
  have hslot : Array.index_usize
      ((Array.repeat 1#usize none).set 0#usize (some (0#i64, 1#i64))) 0#usize =
      ok (some (0#i64, 1#i64)) := by
    simp [Array.index_usize, Array.repeat_val, Array.set_val_eq]
  have hone := hhome 1#usize 0#usize _ 0#i64 1#i64 houter hslot
  have hzero : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      offHomeTable 0#i64 = ok 0#usize := by
    simp [table.Table.shard_index, table.WordHome.Insts.Apeiron_wordTableHomeI64.home,
      HMod.hMod, UScalar.rem, IScalar.hcast]
    rw [u64_zero_mod_256, cast_u64_zero_usize]
  have : (1#usize : Usize) = 0#usize := Aeneas.Std.Result.ok_injective (hone.symm.trans hzero)
  simp at this

/-- Concrete counterexample to `PhysicalShape t ↔ ∃ h, ReadRep t h`: the right side holds for
    this off-home state and the left side does not. -/
theorem readRep_exists_without_physicalShape :
    (∃ h, ReadRep offHomeTable h) ∧ ¬ PhysicalShape offHomeTable := by
  refine ⟨readRep_exists offHomeTable, ?_⟩
  intro hshape
  exact offHomeTable_not_homePlaced hshape.2.1

/-- No entry is present in the physical state returned by the generated constructor. -/
theorem emptyWordTableN_homePlaced {N : Usize} (default : I64) :
    HomePlaced (emptyWordTableN (N := N) default) := by
  set_option maxRecDepth 4096 in
    intro shard i slots key value houter
    have h := getElem?_eq_some_of_index
      (emptyWordTableN (N := N) default).shards shard (some slots) houter
    change (List.replicate 256
      (none : Option (Array (Option (I64 × I64)) N)))[shard.val]? = some (some slots) at h
    rw [List.getElem?_eq_getElem (by
      by_contra hn
      rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
      simp at h), List.getElem_replicate] at h
    simp at h

/-- The real generated constructor establishes the universal physical representation at every
    `N`, including `N = 0`. -/
theorem emptyWordTableN_physicalInv {N : Usize} (default : I64) :
    PhysicalInv (emptyWordTableN (N := N) default) (fun _ => default) := by
  have hinv := emptyWordTableN_tableInv2 (N := N) default
  have hhome := emptyWordTableN_homePlaced (N := N) default
  exact ⟨⟨hinv, hhome, globalKeysDistinct_of_tableInv2N_homePlaced _ hinv hhome⟩,
    emptyWordTableN_readRep default⟩

private theorem homePlaced_replacePresent {N : Usize} (t : WordTableN N)
    (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (key old value : I64)
    (hhome : HomePlaced t) (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    HomePlaced (replacePresent t shard slots i key value) := by
  intro s j current stored storedValue hs hj
  have hshardBound : shard.val < t.shards.val.length := by
    have h := getElem?_eq_some_of_index t.shards shard (some slots) houter
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
    simp at h
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots.set i (some (key, value)) := by
      have hs' := getElem?_eq_some_of_index
        (replacePresent t shard slots i key value).shards shard (some current) hs
      simp only [replacePresent] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    by_cases hji : j = i
    · subst j
      have hnew := getElem?_eq_some_of_index (slots.set i (some (key, value))) i
        (some (stored, storedValue)) hj
      have hi : i.val < slots.val.length := by
        have h := getElem?_eq_some_of_index slots i (some (key, old)) hslot
        by_contra hn
        rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
        simp at h
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi] at hnew
      have hp : (stored, storedValue) = (key, value) :=
        Option.some.inj (Option.some.inj hnew.symm)
      have hk : stored = key := congrArg Prod.fst hp
      have hv : storedValue = value := congrArg Prod.snd hp
      subst stored
      subst storedValue
      rw [← word_shard_index_state_independent t
        (replacePresent t shard slots i key value) key]
      exact hhome shard i slots key old houter hslot
    · have hjiVal : j.val ≠ i.val := fun h => hji (UScalar.eq_of_val_eq h)
      have hjOld : Array.index_usize slots j = ok (some (stored, storedValue)) := by
        simpa [Array.index_usize, Array.set_val_eq, hjiVal] using hj
      rw [← word_shard_index_state_independent t
        (replacePresent t shard slots i key value) stored]
      exact hhome shard j slots stored storedValue houter hjOld
  · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [replacePresent, Array.index_usize, Array.set_val_eq, hne] using hs
    rw [← word_shard_index_state_independent t
      (replacePresent t shard slots i key value) stored]
    exact hhome s j current stored storedValue hsOld hj

private theorem homePlaced_insertVacant {N : Usize} (t : WordTableN N)
    (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (key value : I64)
    (hhome : HomePlaced t)
    (hroute : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    HomePlaced (insertVacant t shard slots i key value) := by
  intro s j current stored storedValue hs hj
  have hshardBound : shard.val < t.shards.val.length := by
    have h := getElem?_eq_some_of_index t.shards shard (some slots) houter
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
    simp at h
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots.set i (some (key, value)) := by
      have hs' := getElem?_eq_some_of_index
        (insertVacant t shard slots i key value).shards shard (some current) hs
      simp only [insertVacant] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    by_cases hji : j = i
    · subst j
      have hnew := getElem?_eq_some_of_index (slots.set i (some (key, value))) i
        (some (stored, storedValue)) hj
      have hi : i.val < slots.val.length := by
        have h := getElem?_eq_some_of_index slots i none hslot
        by_contra hn
        rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at h
        simp at h
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi] at hnew
      have hp : (stored, storedValue) = (key, value) :=
        Option.some.inj (Option.some.inj hnew.symm)
      have hk : stored = key := congrArg Prod.fst hp
      have hv : storedValue = value := congrArg Prod.snd hp
      subst stored
      subst storedValue
      rw [← word_shard_index_state_independent t (insertVacant t shard slots i key value) key]
      exact hroute
    · have hjiVal : j.val ≠ i.val := fun h => hji (UScalar.eq_of_val_eq h)
      have hjOld : Array.index_usize slots j = ok (some (stored, storedValue)) := by
        simpa [Array.index_usize, Array.set_val_eq, hjiVal] using hj
      rw [← word_shard_index_state_independent t
        (insertVacant t shard slots i key value) stored]
      exact hhome shard j slots stored storedValue houter hjOld
  · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [insertVacant, Array.index_usize, Array.set_val_eq, hne] using hs
    rw [← word_shard_index_state_independent t
      (insertVacant t shard slots i key value) stored]
    exact hhome s j current stored storedValue hsOld hj

private theorem homePlaced_insertEmptyShard {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (shard : Usize) (key value : I64) (hhome : HomePlaced t)
    (hroute : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard) (houter : Array.index_usize t.shards shard = ok none) :
    HomePlaced (insertVacant t shard (Array.repeat N none) 0#usize key value) := by
  intro s j current stored storedValue hs hj
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = (Array.repeat N none).set 0#usize (some (key, value)) := by
      have hs' := getElem?_eq_some_of_index
        (insertVacant t shard (Array.repeat N none) 0#usize key value).shards shard
        (some current) hs
      simp only [insertVacant] at hs'
      have hb : shard.val < t.shards.val.length := by
        have ho := getElem?_eq_some_of_index t.shards shard none houter
        by_contra hn
        rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at ho
        simp at ho
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hb] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    have hjN : j.val < N.val := by
      have hg := getElem?_eq_some_of_index
        ((Array.repeat N none).set 0#usize (some (key, value))) j
        (some (stored, storedValue)) hj
      by_contra hn
      rw [Array.set_val_eq, List.getElem?_eq_none (by simpa using Nat.not_lt.mp hn)] at hg
      simp at hg
    have hj0 : j = 0#usize := by
      by_contra hjne
      have hjneVal : j.val ≠ 0 := fun he => hjne (UScalar.eq_of_val_eq he)
      have hg := getElem?_eq_some_of_index
        ((Array.repeat N none).set 0#usize (some (key, value))) j
        (some (stored, storedValue)) hj
      rw [Array.set_val_eq] at hg
      simp [Array.repeat_val, hjN, hjneVal] at hg
    subst j
    have hg := getElem?_eq_some_of_index
      ((Array.repeat N none).set 0#usize (some (key, value))) 0#usize
      (some (stored, storedValue)) hj
    rw [Array.set_val_eq, Array.repeat_val, List.getElem?_set_self (by simpa using hN)] at hg
    have hp : (stored, storedValue) = (key, value) :=
      Option.some.inj (Option.some.inj hg.symm)
    have hk : stored = key := congrArg Prod.fst hp
    have hv : storedValue = value := congrArg Prod.snd hp
    subst stored
    subst storedValue
    rw [← word_shard_index_state_independent t
      (insertVacant t shard (Array.repeat N none) 0#usize key value) key]
    exact hroute
  · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [insertVacant, Array.index_usize, Array.set_val_eq, hne] using hs
    rw [← word_shard_index_state_independent t
      (insertVacant t shard (Array.repeat N none) 0#usize key value) stored]
    exact hhome s j current stored storedValue hsOld hj

/-- Home-shard placement is preserved by every branch of the real generated insertion. -/
theorem homePlaced_insert_with {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (key value : I64) (hhome : HomePlaced t) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => match result.1 with
        | core.result.Result.Ok _ => HomePlaced result.2
        | core.result.Result.Err table.StorageError.ShardFull => result.2 = t ∧ HomePlaced result.2
      ⦄ := by
  let locate := generatedLocator t
  have hroute : GeneratedRoute t key (locate key) := generatedLocator_routes t key
  cases hlocate : locate key with
  | some p =>
      rcases p with ⟨shard, i⟩
      rw [hlocate] at hroute
      obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
      rw [insert_with_present_exact t key value shard i slots key old hshard hfind houter hslot]
      simp only [Aeneas.Std.WP.spec_ok]
      exact homePlaced_replacePresent t shard i slots key old
        (core.num.I64.wrapping_add value old) hhome houter hslot
  | none =>
      rw [hlocate] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      obtain ⟨routed, hrouted, hbound⟩ := word_shard_index_spec t key
      have : routed = shard := Aeneas.Std.Result.ok_injective (hrouted.symm.trans hshard)
      subst routed
      have hlen : shard.val < t.shards.val.length := by rw [t.shards.property]; exact hbound
      obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
        (Array.index_usize_spec t.shards shard hlen)
      cases outer with
      | none =>
          rw [insert_with_empty_shard_exact t key value shard hshard hfind houter hN hbound]
          simp only [Aeneas.Std.WP.spec_ok]
          exact homePlaced_insertEmptyShard hN t shard key
            (core.num.I64.wrapping_add value t.default) hhome hshard houter
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
          have : found = none := Aeneas.Std.Result.ok_injective hfound.symm
          subst found
          have hnomatch : NoMatchIn key slots.val 0 N.val := by simpa [FindPostN] using hpost
          by_cases hvacant : ∃ j, j < N.val ∧ slots.val[j]? = some none
          · let firstNat := Nat.find hvacant
            have hfirstNat : firstNat < N.val := (Nat.find_spec hvacant).1
            have hslotGet : slots.val[firstNat]? = some none := (Nat.find_spec hvacant).2
            have hempty : NoEmptyIn slots.val 0 firstNat := by
              intro j _ hj hempty
              have hmin := Nat.find_min' hvacant
                ⟨hj.trans hfirstNat, by simpa [EmptyAt] using hempty⟩
              exact (Nat.not_lt_of_ge hmin hj).elim
            let first : Usize := ⟨firstNat, by scalar_tac⟩
            have hfirst : first < N := hfirstNat
            have hslot : Array.index_usize slots first = ok none :=
              index_of_getElem?_eq_some slots first none hslotGet
            rw [insert_with_vacant_exact t key value shard first slots hshard hfind houter
              hfirst hempty (fun j hj hjlt => hnomatch j hj (hjlt.trans hfirst)) hslot]
            simp only [Aeneas.Std.WP.spec_ok]
            exact homePlaced_insertVacant t shard first slots key
              (core.num.I64.wrapping_add value t.default) hhome hshard houter hslot
          · have hfull : NoEmptyIn slots.val 0 N.val := by
              intro j _ hj hempty
              exact hvacant ⟨j, hj, by simpa [EmptyAt] using hempty⟩
            rw [insert_with_full_exactN hN t key value shard slots hshard hfind houter hfull]
            simp [hhome]

/-- Combined physical and canonical postcondition for the real generated insertion. -/
def PhysicalInsertResult {N : Usize} (t : WordTableN N) (h : HMap I64 I64)
    (key value : I64)
    (result : core.result.Result Unit table.StorageError × WordTableN N) : Prop :=
  match result.1 with
  | core.result.Result.Ok _ =>
      PhysicalInv result.2 (insertWith core.num.I64.wrapping_add key value h)
  | core.result.Result.Err table.StorageError.ShardFull => result.2 = t ∧ PhysicalInv result.2 h

/-- At every positive `N`, successful real insertion preserves all physical clauses and refines
    canonical `insertWith`; `ShardFull` returns the identical physical representative. -/
theorem physicalInv_insert_with {N : Usize} (hN : 0#usize < N)
    (t : WordTableN N) (h : HMap I64 I64) (key value : I64) (hphysical : PhysicalInv t h) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => PhysicalInsertResult t h key value result ⦄ := by
  obtain ⟨shape, hread⟩ := hphysical
  obtain ⟨result, hr, hinv⟩ := Aeneas.Std.WP.spec_imp_exists
    (tableInv2N_insert_with hN t key value shape.1)
  obtain ⟨homeResult, hhr, hhome⟩ := Aeneas.Std.WP.spec_imp_exists
    (homePlaced_insert_with hN t key value shape.2.1)
  have heq : homeResult = result := Aeneas.Std.Result.ok_injective (hhr.symm.trans hr)
  subst homeResult
  obtain ⟨readResult, hrr, hreadResult⟩ := Aeneas.Std.WP.spec_imp_exists
    (buildInsertResult_of_readRepN hN t h key value hread)
  have heq : readResult = result := Aeneas.Std.Result.ok_injective (hrr.symm.trans hr)
  subst readResult
  rw [hr]
  cases result with
  | mk outcome next =>
      cases outcome with
      | Ok _ =>
          simp only [Aeneas.Std.WP.spec_ok, PhysicalInsertResult] at *
          exact ⟨⟨hinv, hhome,
            globalKeysDistinct_of_tableInv2N_homePlaced next hinv hhome⟩, hreadResult⟩
      | Err error =>
          cases error
          simp only [Aeneas.Std.WP.spec_ok, PhysicalInsertResult] at *
          refine ⟨hinv.1, ?_⟩
          simpa [hinv.1] using (show PhysicalInv t h from ⟨shape, hread⟩)

/-- At `N = 0` the generated insertion reaches the explicit early refusal and leaves every state
    unchanged. -/
theorem insert_with_zero_exact (t : WordTableN 0#usize) (key value : I64) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Err table.StorageError.ShardFull, t) := by
  have hroute := generatedLocator_routes t key
  cases hloc : generatedLocator t key with
  | some p =>
      rcases p with ⟨shard, i⟩
      rw [hloc] at hroute
      obtain ⟨slots, stored, _, _, _, hslot⟩ := hroute
      have hg := getElem?_eq_some_of_index slots i (some (key, stored)) hslot
      rw [List.getElem?_eq_none (by rw [slots.property]; simp)] at hg
      simp at hg
  | none =>
      rw [hloc] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      unfold table.Table.insert_with
      rw [hshard]
      simp only [bind_tc_ok]
      rw [hfind]
      simp

/-- Zero-capacity refusal preserves the universal representation without a success case. -/
theorem physicalInv_insert_with_zero (t : WordTableN 0#usize) (h : HMap I64 I64)
    (key value : I64) (hphysical : PhysicalInv t h) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => PhysicalInsertResult t h key value result ⦄ := by
  rw [insert_with_zero_exact t key value]
  exact ⟨rfl, hphysical⟩

/-- Universal insertion theorem, combining positive-capacity success/refusal with the exact
    zero-capacity refusal branch. -/
theorem physicalInv_insert_with_all {N : Usize} (t : WordTableN N) (h : HMap I64 I64)
    (key value : I64) (hphysical : PhysicalInv t h) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value
      ⦃ result => PhysicalInsertResult t h key value result ⦄ := by
  by_cases hzero : N.val = 0
  · have hN : N = 0#usize := UScalar.eq_of_val_eq hzero
    subst N
    exact physicalInv_insert_with_zero t h key value hphysical
  · have hpositive : 0#usize < N := Nat.pos_of_ne_zero hzero
    exact physicalInv_insert_with hpositive t h key value hphysical

end Apeiron.TableRefinement
