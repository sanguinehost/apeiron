import TableRefinementAdmissionGeneric

/-! # Replacement-map instance of the executing word table -/

namespace Apeiron.Joint.TableInstance

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word
open Apeiron.TableRefinement

/-- The selected last-write-wins merge, as an Aeneas callback instance. -/
def replacementMerge : table.Merge Unit I64 where
  merge := fun _ new _old => ok new

/-- The executing callback computes the selected new-first replacement function. -/
theorem replacementMerge_spec (new old : I64) :
    replacementMerge.merge () new old = ok new := by
  rfl

/-- The canonical merge selected by the replacement-map profile. -/
def replacement (new _old : I64) : I64 := new

/-- The executing instance remains the generated fixed-shard word table at `N = 16`. -/
abbrev ReplacementTable := WordTable

/-- Read/representation relation at the executing replacement-map instance. -/
def ReplacementReadRep (t : ReplacementTable) (h : HMap I64 I64) : Prop :=
  ReadRepG core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 t h

/-- The instance relation is the established executing-read relation, not the open physical `Rep`. -/
theorem replacementReadRep_iff_readRep (t : ReplacementTable) (h : HMap I64 I64) :
    ReplacementReadRep t h ↔ ReadRep t h := by
  rfl

/-- At `N = 16`, the instance relation agrees with the earlier scoped extensional relation. -/
theorem replacementReadRep_iff_scopedRep (t : ReplacementTable) (h : HMap I64 I64) :
    ReplacementReadRep t h ↔ Rep t h := by
  rw [replacementReadRep_iff_readRep, readRep_iff_Rep]

/-- The real generated `Table::new` initializes the constant canonical map. -/
theorem replacement_new_readRep (default : I64) :
    ∃ t, table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64
        default () = ok t ∧ ReplacementReadRep t (fun _ => default) := by
  refine ⟨emptyWordTable default, rfl, ?_⟩
  exact (replacementReadRep_iff_readRep _ _).mpr (emptyWordTableN_readRep default)

/-- Executing `get` is canonical `Apeiron.Table.get` under the instance relation. -/
theorem replacement_get {t : ReplacementTable} {h : HMap I64 I64}
    (hrep : ReplacementReadRep t h) (key : I64) :
    table.Table.get core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok (Apeiron.Table.get h key) := by
  exact hrep key

private theorem stateRead_replace_value (locate : I64 → Option (Usize × Usize))
    (t : ReplacementTable) (key value : I64) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (old : I64)
    (hlocate : locate key = some (shard, i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    stateRead locate (replacePresent t shard slots i key value) =
      insertWith replacement key value (stateRead locate t) := by
  have houter' : t.shards.val[shard.val]? = some (some slots) := by
    unfold Array.index_usize at houter
    split at houter <;> simp_all
  have hslot' : slots.val[i.val]? = some (some (key, old)) := by
    unfold Array.index_usize at hslot
    split at hslot <;> simp_all
  have hs : shard.val < 256 := by have := t.shards.property; grind
  have hi : i.val < 16 := by have := slots.property; grind
  have hsl : shard.val < t.shards.val.length := by simpa [t.shards.property] using hs
  have hil : i.val < slots.val.length := by simpa [slots.property] using hi
  have houterGet : t.shards.val[shard.val] = some slots := by
    rw [List.getElem?_eq_getElem hsl] at houter'
    exact Option.some.inj houter'
  have hslotGet : slots.val[i.val] = some (key, old) := by
    rw [List.getElem?_eq_getElem hil] at hslot'
    exact Option.some.inj hslot'
  funext x
  simp only [insertWith]
  by_cases hx : x = key
  · subst x
    simp [stateRead, replacePresent, replacement, hlocate, hs, hi]
  · rw [if_neg hx]
    simp only [stateRead]
    cases hloc : locate x with
    | none => simp [replacePresent]
    | some p =>
      rcases p with ⟨s, j⟩
      by_cases hss : s = shard
      · subst s
        by_cases hj : j = i
        · subst j
          simp [replacePresent, houterGet, hslotGet, hs, hi, Ne.symm hx]
        · have hji : j.val ≠ i.val := fun h => hj (UScalar.eq_of_val_eq h)
          simp [replacePresent, houterGet, hs, Array.set_val_eq, hji]
      · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
        simp [replacePresent, Array.set_val_eq, hne]

private theorem stateRead_insert_value (locate : I64 → Option (Usize × Usize))
    (t : ReplacementTable) (key value : I64) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize)
    (_hlocate : locate key = none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    stateRead (insertLocator locate key shard i)
        (insertVacant t shard slots i key value) =
      insertWith replacement key value (stateRead locate t) := by
  have houter' : t.shards.val[shard.val]? = some (some slots) := by
    unfold Array.index_usize at houter
    split at houter <;> simp_all
  have hslot' : slots.val[i.val]? = some none := by
    unfold Array.index_usize at hslot
    split at hslot <;> simp_all
  have hs : shard.val < 256 := by have := t.shards.property; grind
  have hi : i.val < 16 := by have := slots.property; grind
  have hsl : shard.val < t.shards.val.length := by simpa [t.shards.property] using hs
  have hil : i.val < slots.val.length := by simpa [slots.property] using hi
  have houterGet : t.shards.val[shard.val] = some slots := by
    rw [List.getElem?_eq_getElem hsl] at houter'
    exact Option.some.inj houter'
  have hslotGet : slots.val[i.val] = none := by
    rw [List.getElem?_eq_getElem hil] at hslot'
    exact Option.some.inj hslot'
  funext x
  simp only [insertWith]
  by_cases hx : x = key
  · subst x
    simp [stateRead, insertLocator, insertVacant, replacement, hs, hi]
  · rw [if_neg hx]
    simp only [stateRead, insertLocator, hx, ↓reduceIte]
    cases hloc : locate x with
    | none => simp [insertVacant]
    | some p =>
      rcases p with ⟨s, j⟩
      by_cases hss : s = shard
      · subst s
        by_cases hj : j = i
        · subst j
          simp [insertVacant, houterGet, hslotGet, hs, hi, Ne.symm hx]
        · have hji : j.val ≠ i.val := fun h => hj (UScalar.eq_of_val_eq h)
          simp [insertVacant, houterGet, hs, Array.set_val_eq, hji]
      · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
        simp [insertVacant, Array.set_val_eq, hne]

private theorem stateRead_insert_empty_value (locate : I64 → Option (Usize × Usize))
    (t : ReplacementTable) (key value : I64) (shard : Usize)
    (hroutes : Routes t locate)
    (houter : Array.index_usize t.shards shard = ok none) :
    stateRead (insertLocator locate key shard 0#usize)
        (insertVacant t shard (Array.repeat 16#usize none) 0#usize key value) =
      insertWith replacement key value (stateRead locate t) := by
  have houter' : t.shards.val[shard.val]? = some none := by
    unfold Array.index_usize at houter
    split at houter <;> simp_all
  have hsl : shard.val < t.shards.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at houter'
    simp at houter'
  have hs : shard.val < 256 := by simpa [t.shards.property] using hsl
  have houterGet : t.shards.val[shard.val] = none := by
    rw [List.getElem?_eq_getElem hsl] at houter'
    exact Option.some.inj houter'
  funext x
  simp only [insertWith]
  by_cases hx : x = key
  · subst x
    simp [stateRead, insertLocator, insertVacant, replacement, hs, Array.repeat_val,
      Array.set_val_eq]
  · rw [if_neg hx]
    simp only [stateRead, insertLocator, hx, ↓reduceIte]
    cases hloc : locate x with
    | none => simp [insertVacant]
    | some p =>
      rcases p with ⟨s, j⟩
      by_cases hss : s = shard
      · subst s
        have hroute := hroutes x
        rw [hloc] at hroute
        obtain ⟨oldSlots, oldValue, hshard, hfind, hold, hslot⟩ := hroute
        rw [houter] at hold
        simp at hold
      · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
        simp [insertVacant, Array.set_val_eq, hne]

private theorem insert_present_exact (t : ReplacementTable) (key value : I64)
    (shard i : Usize) (slots : Array (Option (I64 × I64)) 16#usize) (old : I64)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok (some i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge t () key value =
      ok (core.result.Result.Ok (), replacePresent t shard slots i key value) := by
  have hmouter : Array.index_mut_usize t.shards shard = ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have hminner : Array.index_mut_usize slots i = ok (some (key, old), slots.set i) := by
    simp [Array.index_mut_usize, hslot]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp [hmouter, hminner, replacementMerge, replacePresent]

private theorem loop_vacant_body_exact
    (back : Option (Array (Option (I64 × I64)) 16#usize) →
      Array (Option (Array (Option (I64 × I64)) 16#usize)) 256#usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (default key value : I64)
    (i : Usize) (hi : i < 16#usize) (hslot : Array.index_usize slots i = ok none) :
    table.Table.insert_with_loop0.body core.clone.CloneI64 replacementMerge back default ()
      key value slots i = ok (ControlFlow.done (core.result.Result.Ok (),
        back (some (slots.set i (some (key, value)))))) := by
  have hm : Array.index_mut_usize slots i = ok (none, slots.set i) := by
    simp [Array.index_mut_usize, hslot]
  have hu : Array.update (slots.set i none) i (some (key, value)) =
      ok (slots.set i (some (key, value))) := by
    have hb : i.val < (slots.set i none).length := by simpa using hi
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec (slots.set i none) i (some (key, value)) hb)
    rw [hp] at ha
    calc
      _ = ok ((slots.set i none).set i (some (key, value))) := ha
      _ = _ := by
        congr 1
        apply Aeneas.Std.Array.ext
        simp only [Array.set_val_eq, List.set_set]
  unfold table.Table.insert_with_loop0.body
  rw [if_pos hi, hslot]
  simp [replacementMerge, hm, hu, hi]

private theorem loop_vacant_exact
    (back : Option (Array (Option (I64 × I64)) 16#usize) →
      Array (Option (Array (Option (I64 × I64)) 16#usize)) 256#usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (default key value : I64)
    (first : Usize) (hfirst : first < 16#usize)
    (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    table.Table.insert_with_loop0 core.clone.CloneI64 replacementMerge back default () key value
      slots 0#usize ⦃ r => r = (core.result.Result.Ok (),
        back (some (slots.set first (some (key, value))))) ⦄ := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => first.val - i.val)
    (inv := fun i => InsertInvN key slots.val 0 first i)
    (post := fun r => r = (core.result.Result.Ok (),
      back (some (slots.set first (some (key, value))))))
  · intro i hi
    obtain ⟨hile, hstart, hnoempty, hnomatch'⟩ := hi
    by_cases hifirst : i = first
    · subst i
      rw [loop_vacant_body_exact back slots default key value first hfirst hslot]
      simp
    · have hilt : i.val < first.val := by
        have hne : i.val ≠ first.val := fun h => hifirst (UScalar.eq_of_val_eq h)
        omega
      have hiN : i < 16#usize := Nat.lt_trans hilt hfirst
      unfold table.Table.insert_with_loop0.body
      rw [if_pos hiN]
      step
      rcases o with _ | occupied
      · have hib : i.val < slots.val.length := by rw [slots.property]; exact hiN
        exact (hempty i.val hstart hilt (emptyAt_of_none hib o_post.symm)).elim
      · simp
        step
        refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [i1_post] using noEmptyIn_succ hnoempty (hempty i.val hstart hilt)
        · simpa [i1_post] using noMatchIn_succ hnomatch' (hnomatch i.val hstart hilt)
        · omega
  · refine ⟨Nat.zero_le _, le_rfl, ?_, ?_⟩
    · intro j hj hjlt _
      have hz : (0#usize).val = 0 := rfl
      omega
    · intro j hj hjlt _
      have hz : (0#usize).val = 0 := rfl
      omega

private theorem insert_vacant_exact (t : ReplacementTable) (key value : I64)
    (shard first : Usize) (slots : Array (Option (I64 × I64)) 16#usize)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < 16#usize) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge t () key value =
      ok (core.result.Result.Ok (), insertVacant t shard slots first key value) := by
  have hm : Array.index_mut_usize t.shards shard = ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (loop_vacant_exact (t.shards.set shard) slots t.default key value first hfirst hempty
      hnomatch hslot)
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
    let (result, shards) ← table.Table.insert_with_loop0 core.clone.CloneI64 replacementMerge
      (t.shards.set shard) t.default () key value slots 0#usize
    ok (result, { t with shards := shards })) = _
  rw [hr]
  simp [insertVacant]

private theorem insert_empty_exact (t : ReplacementTable) (key value : I64) (shard : Usize)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok none) (hbound : shard < 256#usize) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge t () key value =
      ok (core.result.Result.Ok (),
        insertVacant t shard (Array.repeat 16#usize none) 0#usize key value) := by
  let slots : Array (Option (I64 × I64)) 16#usize := Array.repeat 16#usize none
  have hslot : Array.index_usize slots 0#usize = ok none := by
    simp [slots, Array.index_usize, Array.repeat_val]
  have hmSlot : Array.index_mut_usize slots 0#usize = ok (none, slots.set 0#usize) := by
    simp [Array.index_mut_usize, hslot]
  have huSlot : Array.update (slots.set 0#usize none) 0#usize (some (key, value)) =
      ok (slots.set 0#usize (some (key, value))) := by
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec (slots.set 0#usize none) 0#usize (some (key, value)) (by decide))
    rw [hp] at ha
    calc
      _ = ok ((slots.set 0#usize none).set 0#usize (some (key, value))) := ha
      _ = _ := by congr 1
  have hmOuter : Array.index_mut_usize t.shards shard = ok (none, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have huOuter : Array.update (t.shards.set shard none) shard
      (some (slots.set 0#usize (some (key, value)))) =
      ok (t.shards.set shard (some (slots.set 0#usize (some (key, value))))) := by
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec (t.shards.set shard none) shard
        (some (slots.set 0#usize (some (key, value)))) (by simpa using hbound))
    rw [hp] at ha
    calc
      _ = ok ((t.shards.set shard none).set shard
          (some (slots.set 0#usize (some (key, value))))) := ha
      _ = _ := by
        congr 1
        apply Aeneas.Std.Array.ext
        simp only [Array.set_val_eq, List.set_set]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg (by decide), houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_pos (by rfl)]
  simp only [core.clone.impls.CloneI64.clone, replacementMerge, bind_tc_ok, massert]
  rw [if_pos (by decide), hmSlot]
  simp only [bind_tc_ok]
  change (do
    let slots2 ← Array.update (slots.set 0#usize none) 0#usize (some (key, value))
    massert (shard < 256#usize)
    let (o, back) ← Array.index_mut_usize t.shards shard
    let a ← Array.update (back o) shard (some slots2)
    ok (core.result.Result.Ok (), { t with shards := a })) = _
  rw [huSlot]
  simp only [bind_tc_ok, massert]
  rw [if_pos hbound, hmOuter]
  simp only [bind_tc_ok]
  rw [huOuter]
  simp [slots, insertVacant]

private theorem loop_full_exact
    (back : Option (Array (Option (I64 × I64)) 16#usize) →
      Array (Option (Array (Option (I64 × I64)) 16#usize)) 256#usize)
    (slots : Array (Option (I64 × I64)) 16#usize) (default key value : I64)
    (hfull : NoEmptyIn slots.val 0 16) :
    table.Table.insert_with_loop0 core.clone.CloneI64 replacementMerge back default () key value
      slots 0#usize ⦃ r => r = (core.result.Result.Err table.StorageError.ShardFull,
        back (some slots)) ⦄ := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => 16 - i.val)
    (inv := fun i => InsertFullInvN (N := 16#usize) slots.val 0 i)
    (post := fun r => r = (core.result.Result.Err table.StorageError.ShardFull,
      back (some slots)))
  · intro i hiInv
    obtain ⟨hile, hstart, hno⟩ := hiInv
    unfold table.Table.insert_with_loop0.body
    by_cases hi : i < 16#usize
    · rw [if_pos hi]
      step
      rcases o with _ | occupied
      · have hib : i.val < slots.val.length := by rw [slots.property]; exact hi
        exact (hfull i.val hstart hi (emptyAt_of_none hib o_post.symm)).elim
      · simp
        step
        have hin : i.val < 16 := hi
        have hnextBound : i.val + 1 ≤ 16 := by omega
        have hnextStart : 0 ≤ i.val + 1 := by omega
        have hdec : 16 - (i.val + 1) < 16 - i.val := by omega
        refine ⟨⟨by simpa [i1_post] using hnextBound, ?_, ?_⟩, ?_⟩
        · rw [i1_post]
          exact hnextStart
        · simpa [i1_post] using noEmptyIn_succ hno (hfull i.val hstart hi)
        · simpa [i1_post] using hdec
    · rw [if_neg hi, Aeneas.Std.WP.spec_ok]
  · change (0 : Nat) ≤ 16 ∧ 0 ≤ 0 ∧ NoEmptyIn slots.val 0 0
    exact ⟨by omega, by omega, fun j _ hj => by omega⟩

private theorem insert_full_exact (t : ReplacementTable) (key value : I64) (shard : Usize)
    (slots : Array (Option (I64 × I64)) 16#usize)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfull : NoEmptyIn slots.val 0 16) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge t () key value =
      ok (core.result.Result.Err table.StorageError.ShardFull, t) := by
  have hm : Array.index_mut_usize t.shards shard = ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (loop_full_exact (t.shards.set shard) slots t.default key value hfull)
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
    let (result, shards) ← table.Table.insert_with_loop0 core.clone.CloneI64 replacementMerge
      (t.shards.set shard) t.default () key value slots 0#usize
    ok (result, { t with shards := shards })) = _
  rw [hr]
  have houter' : t.shards.val[shard.val]? = some (some slots) := by
    unfold Array.index_usize at houter
    split at houter <;> simp_all
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

private theorem routes_empty_insert (locate : I64 → Option (Usize × Usize))
    (t : ReplacementTable) (key value : I64) (shard : Usize) (hroutes : Routes t locate)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (houter : Array.index_usize t.shards shard = ok none) (hbound : shard < 256#usize) :
    Routes (insertVacant t shard (Array.repeat 16#usize none) 0#usize key value)
      (insertLocator locate key shard 0#usize) := by
  intro x
  let next := insertVacant t shard (Array.repeat 16#usize none) 0#usize key value
  have hnewOuter : Array.index_usize next.shards shard =
      ok (some ((Array.repeat 16#usize none).set 0#usize (some (key, value)))) := by
    have hsLen : shard.val < t.shards.val.length := by
      rw [t.shards.property]
      exact hbound
    unfold Array.index_usize
    rw [Array.getElem?_Usize_eq]
    simp only [next, insertVacant, Array.set_val_eq]
    rw [List.getElem?_set_eq_of_lt _ hsLen]
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
            (by simp [Array.index_usize, Array.repeat_val]) hx]
          obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
            (find_slots_spec_generic (Array.repeat 16#usize
              (none : Option (I64 × I64))) x 0#usize (Nat.zero_le _))
          cases found with
          | none => exact hfound
          | some i =>
            obtain ⟨hi, _, ⟨stored, hmatch⟩, _⟩ := hpost
            rw [Array.repeat_val] at hmatch
            rw [List.getElem?_eq_getElem (by simpa using hi), List.getElem_replicate] at hmatch
            simp at hmatch
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

/-- The real generated replacement insertion refines canonical `HMap.insertWith replacement`;
    physical capacity refusal remains explicit. -/
theorem replacement_insert_with {t : ReplacementTable} {h : HMap I64 I64}
    (key value : I64) (hrep : ReplacementReadRep t h) :
    BuildInsertResultG core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge () replacement
      t h key value := by
  let locate := generatedLocator t
  have hroutes : Routes t locate := generatedLocator_routes t
  have hfunctional : FunctionalRep locate t h :=
    (generatedFunctionalRep_iff_readRep t h).mpr
      ((replacementReadRep_iff_readRep t h).mp hrep)
  have hroute := hroutes key
  cases hlocate : locate key with
  | some p =>
    rcases p with ⟨shard, i⟩
    rw [hlocate] at hroute
    obtain ⟨slots, old, hshard, hfind, houter, hslot⟩ := hroute
    have hinsert := insert_present_exact t key value shard i slots old hshard hfind houter hslot
    unfold BuildInsertResultG
    rw [hinsert]
    simp only [Aeneas.Std.WP.spec_ok]
    apply (functionalRep_iff_readRep_of_routes locate _ _
      (routes_replacePresent locate t key value shard i slots old hroutes houter hslot)).mp
    unfold FunctionalRep at hfunctional ⊢
    rw [stateRead_replace_value locate t key value shard i slots old hlocate houter hslot,
      hfunctional]
  | none =>
    rw [hlocate] at hroute
    obtain ⟨shard, hshard, hfind⟩ := hroute
    obtain ⟨routed, hrouted, hbound⟩ := word_shard_index_spec t key
    have : routed = shard := Aeneas.Std.Result.ok_injective (hrouted.symm.trans hshard)
    subst routed
    have hsLen : shard.val < t.shards.val.length := by rw [t.shards.property]; exact hbound
    obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.index_usize_spec t.shards shard hsLen)
    cases outer with
    | none =>
      have hinsert := insert_empty_exact t key value shard hshard hfind houter hbound
      unfold BuildInsertResultG
      rw [hinsert]
      simp only [Aeneas.Std.WP.spec_ok]
      let next := insertVacant t shard (Array.repeat 16#usize none) 0#usize key value
      have hroutes' : Routes next (insertLocator locate key shard 0#usize) :=
        routes_empty_insert locate t key value shard hroutes hshard houter hbound
      apply (functionalRep_iff_readRep_of_routes _ next _ hroutes').mp
      unfold FunctionalRep at hfunctional ⊢
      have hstate : stateRead (insertLocator locate key shard 0#usize) next =
          insertWith replacement key value (stateRead locate t) := by
        exact stateRead_insert_empty_value locate t key value shard hroutes houter
      rw [hstate, hfunctional]
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
      have hnone : found = none := Aeneas.Std.Result.ok_injective hfound.symm
      subst found
      have hnomatch : NoMatchIn key slots.val 0 16 := by simpa [FindPostN] using hpost
      by_cases hvacant : ∃ j, j < 16 ∧ slots.val[j]? = some none
      · let k := Nat.find hvacant
        have hk : k < 16 := (Nat.find_spec hvacant).1
        have hslotGet : slots.val[k]? = some none := (Nat.find_spec hvacant).2
        have hempty : NoEmptyIn slots.val 0 k := by
          intro j _ hj he
          exact (Nat.not_lt_of_ge (Nat.find_min' hvacant
            ⟨hj.trans hk, by simpa [EmptyAt] using he⟩) hj).elim
        let first : Usize := ⟨k, by scalar_tac⟩
        have hfirst : first < 16#usize := hk
        have hslot : Array.index_usize slots first = ok none := by
          unfold Array.index_usize
          rw [Array.getElem?_Usize_eq]
          rw [show first.val = k by rfl, hslotGet]
        have hinsert := insert_vacant_exact t key value shard first slots hshard hfind houter
          hfirst hempty (fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hk)) hslot
        unfold BuildInsertResultG
        rw [hinsert]
        simp only [Aeneas.Std.WP.spec_ok]
        have hroutes' := routes_insertVacant locate t key value shard first slots hroutes hshard
          houter hfirst (fun j hj hjlt => hnomatch j hj (Nat.lt_trans hjlt hk)) hslot
        apply (functionalRep_iff_readRep_of_routes _ _ _ hroutes').mp
        unfold FunctionalRep at hfunctional ⊢
        rw [stateRead_insert_value locate t key value shard first slots hlocate houter hslot,
          hfunctional]
      · have hfull : NoEmptyIn slots.val 0 16 := by
          intro j _ hj he
          exact hvacant ⟨j, hj, by simpa [EmptyAt] using he⟩
        unfold BuildInsertResultG
        rw [insert_full_exact t key value shard slots hshard hfind houter hfull]
        simp

/-- Concrete `ShardFull` witness for the replacement callback, with an unchanged generated state. -/
theorem replacement_shardFull_refusal :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge fullWordTable ()
      4096#i64 1#i64 =
      ok (core.result.Result.Err table.StorageError.ShardFull, fullWordTable) := by
  exact insert_full_exact fullWordTable 4096#i64 1#i64 0#usize fullSlots
    (by
      simp [table.Table.shard_index,
        table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
        UScalar.rem, IScalar.hcast]
      rw [u64_4096_mod_256, cast_u64_zero_usize])
    fullWordTable_find_4096
    (by
      set_option maxRecDepth 4096 in
        simp [fullWordTable, Array.index_usize, Array.repeat_val]) fullSlots_noEmpty

/-- The callback contract required by the generic ordered-build layer is inhabited. -/
theorem replacement_buildInsertResult (t : ReplacementTable) (h : HMap I64 I64)
    (key value : I64) (hrep : ReplacementReadRep t h) :
    BuildInsertResultG core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge () replacement
      t h key value := replacement_insert_with key value hrep

/-- The real generated ordered build either reports `ShardFull` or represents canonical
    `Apeiron.Table.build replacement`. -/
theorem replacement_build (default : I64) (kvs : Slice (I64 × I64)) :
    table.Table.build 16#usize core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge core.clone.CloneI64 ()
      default kvs () = ok (core.result.Result.Err table.StorageError.ShardFull) ∨
    ∃ final, table.Table.build 16#usize core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge core.clone.CloneI64 ()
        default kvs () = ok (core.result.Result.Ok final) ∧
      ReplacementReadRep final (Apeiron.Table.build replacement default (sliceEntriesG kvs)) := by
  have hinitial : ReplacementReadRep (emptyWordTable default)
      (orderedFoldG replacement default []) := by
    rw [replacementReadRep_iff_readRep]
    simpa [orderedFoldG, emptyWordTableN, emptyWordTable] using
      (emptyWordTableN_readRep default)
  have hs := table_build_loop_result_readRepG
    (cmp := core.cmp.EqI64) (cloneV := core.clone.CloneI64)
    (HomeInst := table.WordHome.Insts.Apeiron_wordTableHomeI64)
    (MergeInst := replacementMerge) (cloneK := core.clone.CloneI64) (merge := ())
    (mergeFn := replacement) (default := default) (kvs := kvs)
    (initial := emptyWordTable default) (start := 0#usize)
    cloneI64_identity cloneI64_identity (Nat.zero_le _) hinitial
    (fun _ state key value _ _ hrep => replacement_buildInsertResult state _ key value hrep)
  obtain ⟨result, hloop, hpost⟩ := Aeneas.Std.WP.spec_imp_exists hs
  unfold table.Table.build
  rw [show table.Table.new 16#usize table.WordHome.Insts.Apeiron_wordTableHomeI64 default () =
    ok (emptyWordTable default) by rfl]
  simp only [bind_tc_ok]
  rcases hpost with hfull | ⟨final, hok, hrep⟩
  · left
    rw [hloop, hfull]
  · right
    refine ⟨final, ?_, ?_⟩
    · rw [hloop, hok]
    · simpa [ReplacementReadRep, orderedFoldG, Apeiron.Table.build] using hrep

/-- A successful replacement build has the canonical ordered per-key read characterized by
    `Apeiron.Table.get_build_acc`. -/
theorem replacement_build_get_acc (default : I64) (kvs : Slice (I64 × I64))
    (final : ReplacementTable)
    (hsuccess : table.Table.build 16#usize core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 replacementMerge core.clone.CloneI64 ()
      default kvs () = ok (core.result.Result.Ok final)) (key : I64) :
    table.Table.get core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 final key =
      ok ((((sliceEntriesG kvs).filter (fun kv => kv.1 = key)).map Prod.snd).getLastD default) := by
  rcases replacement_build default kvs with hfull | ⟨built, hbuilt, hrep⟩
  · rw [hsuccess] at hfull
    simp at hfull
  · have hfinal : built = final := by
      exact core.result.Result.Ok.inj (Aeneas.Std.Result.ok_injective (hbuilt.symm.trans hsuccess))
    subst built
    rw [hrep key]
    congr 1
    unfold Apeiron.Table.build
    change Apeiron.Table.get
      ((sliceEntriesG kvs).foldl
        (fun m kv => insertWith replacement kv.1 kv.2 m) (fun _ => default)) key = _
    rw [Apeiron.Table.get_build_acc]
    exact Apeiron.Table.foldl_replace _ _

end Apeiron.Joint.TableInstance
