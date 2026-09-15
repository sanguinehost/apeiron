import TableRefinementBuildGeneric

/-! # Callback laws for the generated generic table

This module states the pure, total callback boundary needed to relate the generated generic Rust
table to `Apeiron.Table`.  The first discharge is the present-key branch of the real generated
`table.Table.insert_with` body.  It exposes routing and lookup as physical-state premises; no
success/refusal specification is assumed.
-/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-- Total denotations of every callback used by generic table construction and insertion.
    The merge equation fixes the source and canonical order to `new` then `old`. -/
structure CallbackLawsG {K V H F : Type} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (home : H) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V) : Prop where
  cmp_exact : ∀ a b,
    core.cmp.impls.PartialEqShared.eq cmp.partialEqInst a b = ok (decide (a = b))
  cloneK_identity : ∀ key, cloneK.clone key = ok key
  cloneV_identity : ∀ value, cloneV.clone value = ok value
  home_total : ∀ key, HomeInst.home home key = ok (homeFn key)
  merge_new_old : ∀ new old, MergeInst.merge merge new old = ok (mergeFn new old)

/-- The exact state update performed by a successful present-key branch. -/
def replacePresentG {K V H : Type} {N : Usize} (t : table.Table K V H N)
    (shard : Usize) (slots : Array (Option (K × V)) N) (i : Usize)
    (storedKey : K) (value : V) : table.Table K V H N :=
  { t with shards := t.shards.set shard (some (slots.set i (some (storedKey, value)))) }

/-- Primitive callback laws discharge the real generated present-key insertion branch.
    The result records the callback's `mergeFn new old`, so swapping the generated arguments is
    observable for noncommutative merge functions. -/
theorem insert_with_present_callback_exact {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (key storedKey : K) (value old : V) (shard i : Usize)
    (slots : Array (Option (K × V)) N)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (hfind : table.Table.find cmp HomeInst t shard key = ok (some i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (storedKey, old))) :
    table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Ok (),
        replacePresentG t shard slots i storedKey (mergeFn value old)) := by
  have hmouter : Array.index_mut_usize t.shards shard =
      ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have hminner : Array.index_mut_usize slots i =
      ok (some (storedKey, old), slots.set i) := by
    simp [Array.index_mut_usize, hslot]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp [hmouter, hminner, laws.cloneV_identity old, laws.merge_new_old value old,
    replacePresentG]

/-- The exact state produced by inserting an absent key into an untouched positive-capacity shard. -/
def insertEmptyShardG {K V H : Type} {N : Usize} (t : table.Table K V H N)
    (shard : Usize) (key : K) (value : V) : table.Table K V H N :=
  { t with shards := t.shards.set shard (some ((Array.repeat N none).set 0#usize
      (some (key, value)))) }

/-- Primitive callback laws discharge the real generated fresh-shard branch. -/
theorem insert_with_empty_shard_callback_exact {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (key : K) (value : V) (shard : Usize)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (hfind : table.Table.find cmp HomeInst t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok none)
    (hN : 0#usize < N) (hshardBound : shard < 256#usize) :
    table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Ok (),
        insertEmptyShardG t shard key (mergeFn value t.default)) := by
  have hNne : N ≠ 0#usize := ne_of_gt hN
  have hrepeat : Array.index_usize (Array.repeat N (none : Option (K × V))) 0#usize =
      ok none := by
    have hb : (0#usize).val < (Array.repeat N (none : Option (K × V))).length := by
      simpa using hN
    obtain ⟨x, hx, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.index_usize_spec (Array.repeat N (none : Option (K × V))) 0#usize hb)
    calc
      _ = ok x := hx
      _ = ok none := by rw [hp]; congr 1; simp [Array.repeat_val]
  have hmrepeat : Array.index_mut_usize (Array.repeat N (none : Option (K × V))) 0#usize =
      ok (none, (Array.repeat N none).set 0#usize) := by
    simp [Array.index_mut_usize, hrepeat]
  let merged := mergeFn value t.default
  let newSlots := (Array.repeat N (none : Option (K × V))).set 0#usize
    (some (key, merged))
  have hinner : Array.update ((Array.repeat N none).set 0#usize none) 0#usize
      (some (key, merged)) = ok newSlots := by
    have hb : (0#usize).val < ((Array.repeat N (none : Option (K × V))).set
        0#usize none).length := by simpa using hN
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec ((Array.repeat N none).set 0#usize none) 0#usize
        (some (key, merged)) hb)
    rw [hp] at ha
    calc
      _ = ok (((Array.repeat N none).set 0#usize none).set 0#usize
          (some (key, merged))) := ha
      _ = ok newSlots := by
        congr 1
        apply Aeneas.Std.Array.ext
        simp only [newSlots, Array.set_val_eq, List.set_set]
  have hmouter : Array.index_mut_usize t.shards shard = ok (none, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have houterUpdate : Array.update (t.shards.set shard none) shard (some newSlots) =
      ok (t.shards.set shard (some newSlots)) := by
    have hb : shard.val < (t.shards.set shard none).length := by simpa using hshardBound
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec (t.shards.set shard none) shard (some newSlots) hb)
    rw [hp] at ha
    calc
      _ = ok ((t.shards.set shard none).set shard (some newSlots)) := ha
      _ = ok (t.shards.set shard (some newSlots)) := by
        congr 1
        apply Aeneas.Std.Array.ext
        simp only [Array.set_val_eq, List.set_set]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg hNne, houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_pos (by rfl), laws.cloneV_identity t.default]
  simp only [bind_tc_ok]
  rw [laws.merge_new_old value t.default]
  simp only [bind_tc_ok, massert]
  rw [if_pos hN, hmrepeat]
  simp only [bind_tc_ok]
  change (do
      let slots2 ← Array.update ((Array.repeat N none).set 0#usize none) 0#usize
        (some (key, merged))
      massert (shard < 256#usize)
      let (o, back) ← Array.index_mut_usize t.shards shard
      let a ← Array.update (back o) shard (some slots2)
      ok (core.result.Result.Ok (), { t with shards := a })) = _
  rw [hinner]
  simp only [bind_tc_ok, massert]
  rw [if_pos hshardBound, hmouter]
  simp only [bind_tc_ok]
  rw [houterUpdate]
  simp [newSlots, merged, insertEmptyShardG]

/-- One vacant loop iteration performs the callback-defined fresh-value write. -/
theorem insert_with_loop0_vacant_body_callback_exact {K V F : Type} {N : Usize}
    (cloneV : core.clone.Clone V) (MergeInst : table.Merge F V)
    (back : Option (Array (Option (K × V)) N) →
      Array (Option (Array (Option (K × V)) N)) 256#usize)
    (slots : Array (Option (K × V)) N) (default : V) (merge : F)
    (mergeFn : V → V → V) (key : K) (value : V) (i : Usize)
    (hclone : cloneV.clone default = ok default)
    (hmerge : MergeInst.merge merge value default = ok (mergeFn value default))
    (hi : i < N) (hslot : Array.index_usize slots i = ok none) :
    table.Table.insert_with_loop0.body cloneV MergeInst back default merge key value slots i =
      ok (ControlFlow.done (core.result.Result.Ok (), back (some (slots.set i
        (some (key, mergeFn value default)))))) := by
  have hm : Array.index_mut_usize slots i = ok (none, slots.set i) := by
    simp [Array.index_mut_usize, hslot]
  have hu : Array.update (slots.set i none) i (some (key, mergeFn value default)) =
      ok (slots.set i (some (key, mergeFn value default))) := by
    have hb : i.val < (slots.set i none).length := by simpa using hi
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec (slots.set i none) i (some (key, mergeFn value default)) hb)
    rw [hp] at ha
    calc
      _ = ok ((slots.set i none).set i (some (key, mergeFn value default))) := ha
      _ = _ := by
        congr 1
        apply Aeneas.Std.Array.ext
        simp only [Array.set_val_eq, List.set_set]
  unfold table.Table.insert_with_loop0.body
  rw [if_pos hi, hslot]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_pos (by rfl), hclone]
  simp only [bind_tc_ok]
  rw [hmerge]
  simp only [bind_tc_ok, massert]
  rw [if_pos hi, hm]
  simp [hu]

/-- The generic vacant-slot loop reaches an explicitly identified first vacancy. -/
theorem insert_with_loop0_vacant_callback_exact {K V F : Type} {N : Usize}
    (cloneV : core.clone.Clone V) (MergeInst : table.Merge F V)
    (back : Option (Array (Option (K × V)) N) →
      Array (Option (Array (Option (K × V)) N)) 256#usize)
    (slots : Array (Option (K × V)) N) (default : V) (merge : F)
    (mergeFn : V → V → V) (key : K) (value : V) (first : Usize)
    (hclone : cloneV.clone default = ok default)
    (hmerge : MergeInst.merge merge value default = ok (mergeFn value default))
    (hfirst : first < N)
    (hprefix : ∀ j : Usize, j.val < first.val →
      ∃ entry, Array.index_usize slots j = ok (some entry))
    (hslot : Array.index_usize slots first = ok none) :
    Aeneas.Std.WP.spec
      (table.Table.insert_with_loop0 cloneV MergeInst back default merge key value slots 0#usize)
      (fun r => r = (core.result.Result.Ok (), back (some (slots.set first
        (some (key, mergeFn value default)))))) := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => first.val - i.val)
    (inv := fun i => i.val ≤ first.val)
    (post := fun r => r = (core.result.Result.Ok (), back (some (slots.set first
      (some (key, mergeFn value default))))))
  · intro i hile
    by_cases hifirst : i = first
    · subst i
      rw [insert_with_loop0_vacant_body_callback_exact cloneV MergeInst back slots default
        merge mergeFn key value first hclone hmerge hfirst hslot]
      simp
    · have hilt : i.val < first.val := by
        have hne : i.val ≠ first.val := fun h => hifirst (UScalar.eq_of_val_eq h)
        omega
      have hiN : i < N := Nat.lt_trans hilt hfirst
      obtain ⟨entry, hentry⟩ := hprefix i hilt
      unfold table.Table.insert_with_loop0.body
      rw [if_pos hiN, hentry]
      simp only [bind_tc_ok, core.option.Option.is_none]
      rw [if_neg (by simp)]
      step
      exact ⟨by simpa [i1_post] using Nat.succ_le_iff.mpr hilt, by omega⟩
  · simp

/-- A populated home shard with a first vacant slot takes the generated loop success branch. -/
theorem insert_with_populated_vacant_callback_exact {K V H F : Type} {N : Usize}
    [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (key : K) (value : V) (shard first : Usize) (slots : Array (Option (K × V)) N)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (hfind : table.Table.find cmp HomeInst t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < N)
    (hprefix : ∀ j : Usize, j.val < first.val →
      ∃ entry, Array.index_usize slots j = ok (some entry))
    (hslot : Array.index_usize slots first = ok none) :
    table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Ok (), replacePresentG t shard slots first key
        (mergeFn value t.default)) := by
  have hNne : N ≠ 0#usize := by
    intro hzero
    subst N
    simp at hfirst
  have hmouter : Array.index_mut_usize t.shards shard =
      ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have hloop := insert_with_loop0_vacant_callback_exact cloneV MergeInst
    (t.shards.set shard) slots t.default merge mergeFn key value first
    (laws.cloneV_identity t.default) (laws.merge_new_old value t.default)
    hfirst hprefix hslot
  obtain ⟨loopResult, hloopResult, hloopPost⟩ := Aeneas.Std.WP.spec_imp_exists hloop
  rw [hloopPost] at hloopResult
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg hNne, houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_neg (by simp), hmouter]
  simp only [bind_tc_ok]
  change (do
      let (r, a) ← table.Table.insert_with_loop0 cloneV MergeInst (t.shards.set shard)
        t.default merge key value slots 0#usize
      ok (r, { t with shards := a })) = _
  rw [hloopResult]
  simp [replacePresentG]

/-- The generic vacancy-search loop traverses a full shard and returns its storage unchanged.
    No clone or merge law is needed because the full path invokes neither callback. -/
theorem insert_with_loop0_full_callback_exact {K V F : Type} {N : Usize}
    (cloneV : core.clone.Clone V) (MergeInst : table.Merge F V)
    (back : Option (Array (Option (K × V)) N) →
      Array (Option (Array (Option (K × V)) N)) 256#usize)
    (slots : Array (Option (K × V)) N) (default : V) (merge : F)
    (key : K) (value : V)
    (hfull : ∀ i : Usize, i < N → ∃ entry, Array.index_usize slots i = ok (some entry)) :
    Aeneas.Std.WP.spec
      (table.Table.insert_with_loop0 cloneV MergeInst back default merge key value slots 0#usize)
      (fun r => r = (core.result.Result.Err table.StorageError.ShardFull, back (some slots))) := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => N.val - i.val)
    (inv := fun i => i.val ≤ N.val)
    (post := fun r =>
      r = (core.result.Result.Err table.StorageError.ShardFull, back (some slots)))
  · intro i hile
    unfold table.Table.insert_with_loop0.body
    by_cases hi : i < N
    · obtain ⟨entry, hentry⟩ := hfull i hi
      rw [if_pos hi, hentry]
      simp only [bind_tc_ok, core.option.Option.is_none]
      rw [if_neg (by simp)]
      step
      have hnextBound : i.val + 1 ≤ N.val := Nat.succ_le_iff.mpr hi
      have hdecrease : N.val - (i.val + 1) < N.val - i.val := by omega
      exact ⟨by simpa [i1_post] using hnextBound, by simpa [i1_post] using hdecrease⟩
    · rw [if_neg hi, Aeneas.Std.WP.spec_ok]
  · simp

/-- An absent key in a populated full shard returns `ShardFull` with the exact input table. -/
theorem insert_with_populated_full_callback_exact {K V H F : Type} {N : Usize}
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (key : K) (value : V)
    (shard : Usize) (slots : Array (Option (K × V)) N)
    (hN : 0#usize < N)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (hfind : table.Table.find cmp HomeInst t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfull : ∀ i : Usize, i < N → ∃ entry, Array.index_usize slots i = ok (some entry)) :
    table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Err table.StorageError.ShardFull, t) := by
  have hmouter : Array.index_mut_usize t.shards shard =
      ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have hloop := insert_with_loop0_full_callback_exact cloneV MergeInst
    (t.shards.set shard) slots t.default merge key value hfull
  obtain ⟨loopResult, hloopResult, hloopPost⟩ := Aeneas.Std.WP.spec_imp_exists hloop
  rw [hloopPost] at hloopResult
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg (ne_of_gt hN), houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_neg (by simp), hmouter]
  simp only [bind_tc_ok]
  change (do
      let (r, a) ← table.Table.insert_with_loop0 cloneV MergeInst (t.shards.set shard)
        t.default merge key value slots 0#usize
      ok (r, { t with shards := a })) = _
  rw [hloopResult]
  have houterGet : t.shards.val[shard.val]? = some (some slots) := by
    unfold Array.index_usize at houter
    rw [Array.getElem?_Usize_eq] at houter
    split at houter <;> simp_all
  have hshardLength : shard.val < t.shards.val.length := by
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at houterGet
    simp at houterGet
  have hset : t.shards.set shard (some slots) = t.shards := by
    apply Aeneas.Std.Array.ext
    simp only [Array.set_val_eq]
    apply List.ext_getElem?
    intro n
    rw [List.getElem?_set_of_lt' (some slots) t.shards.val hshardLength]
    split
    · rename_i hn
      subst n
      exact houterGet.symm
    · rfl
  simp [hset]

/-- At zero capacity an absent key refuses before clone or merge can be invoked. -/
theorem insert_with_zero_callback_exact {K V H F : Type}
    (cmp : core.cmp.Eq K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H 0#usize) (merge : F) (key : K) (value : V)
    (shard : Usize)
    (hshard : table.Table.shard_index HomeInst t key = ok shard)
    (hfind : table.Table.find cmp HomeInst t shard key = ok none) :
    table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value =
      ok (core.result.Result.Err table.StorageError.ShardFull, t) := by
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp

/-- A raw slot contains the queried canonical key. -/
def MatchesAtG {K V : Type} (key : K) (slots : List (Option (K × V))) (i : Nat) : Prop :=
  ∃ value, slots[i]? = some (some (key, value))

/-- No slot in `[start, stop)` contains the queried canonical key. -/
def NoMatchInG {K V : Type} (key : K) (slots : List (Option (K × V)))
    (start stop : Nat) : Prop :=
  ∀ i, start ≤ i → i < stop → ¬ MatchesAtG key slots i

def FindInvCallbackG {K V : Type} {N : Usize} (key : K)
    (slots : List (Option (K × V))) (start : Nat) (i : Usize) : Prop :=
  i.val ≤ N.val ∧ start ≤ i.val ∧ NoMatchInG key slots start i.val

def FindPostCallbackG {K V : Type} {N : Usize} (key : K)
    (slots : List (Option (K × V))) (start : Nat) : Option Usize → Prop
  | some i => i.val < N.val ∧ start ≤ i.val ∧ MatchesAtG key slots i.val ∧
      NoMatchInG key slots start i.val
  | none => NoMatchInG key slots start N.val

private theorem noMatchInG_succ {K V : Type} {key : K}
    {slots : List (Option (K × V))} {start i : Nat}
    (h : NoMatchInG key slots start i) (hnot : ¬ MatchesAtG key slots i) :
    NoMatchInG key slots start (i + 1) := by
  intro j hstart hj
  rcases Nat.lt_succ_iff_lt_or_eq.mp hj with hlt | rfl
  · exact h j hstart hlt
  · exact hnot

private theorem matchesAtG_of_get {K V : Type} {key stored : K} {value : V} {i : Nat}
    {slots : List (Option (K × V))} (hi : i < slots.length)
    (h : slots[i] = some (stored, value)) (heq : stored = key) : MatchesAtG key slots i := by
  exact ⟨value, by rw [List.getElem?_eq_getElem hi, h, heq]⟩

private theorem not_matchesAtG_of_none {K V : Type} {key : K} {i : Nat}
    {slots : List (Option (K × V))} (hi : i < slots.length) (h : slots[i] = none) :
    ¬ MatchesAtG key slots i := by
  rintro ⟨value, hvalue⟩
  rw [List.getElem?_eq_getElem hi, h] at hvalue
  simp at hvalue

private theorem not_matchesAtG_of_ne {K V : Type} {key stored : K} {value : V} {i : Nat}
    {slots : List (Option (K × V))} (hi : i < slots.length)
    (h : slots[i] = some (stored, value)) (hne : stored ≠ key) :
    ¬ MatchesAtG key slots i := by
  rintro ⟨other, hother⟩
  rw [List.getElem?_eq_getElem hi, h] at hother
  exact hne (Prod.mk.inj (Option.some.inj (Option.some.inj hother))).1

/-- Exact equality denotation proves the generated generic search is total and returns the first
    canonical matching key. -/
theorem find_slots_spec_callback_exact {K V : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (slots : Array (Option (K × V)) N) (key : K)
    (start : Usize) (hstart : start.val ≤ N.val)
    (hcmp : ∀ a b,
      core.cmp.impls.PartialEqShared.eq cmp.partialEqInst a b = ok (decide (a = b))) :
    Aeneas.Std.WP.spec (table.Table.find_slots_loop cmp slots key start)
      (fun result => FindPostCallbackG (N := N) key slots.val start.val result) := by
  unfold table.Table.find_slots_loop
  apply loop.spec_decr_nat
    (measure := fun i : Usize => N.val - i.val)
    (inv := fun i => FindInvCallbackG (N := N) key slots.val start.val i)
    (post := fun result => FindPostCallbackG (N := N) key slots.val start.val result)
  · intro i hi
    obtain ⟨hile, hstartle, hno⟩ := hi
    unfold table.Table.find_slots_loop.body
    by_cases hlt : i < N
    · rw [if_pos hlt]
      have hin : i.val < N.val := hlt
      have hib : i.val < slots.val.length := by rw [Array.length_eq]; exact hin
      step
      rcases o with _ | ⟨stored, value⟩
      · have ho : slots.val[i.val] = none := o_post.symm
        step
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [r_post] using noMatchInG_succ hno (not_matchesAtG_of_none hib ho)
        · omega
      · have ho : slots.val[i.val] = some (stored, value) := o_post.symm
        change Aeneas.Std.WP.spec (do
          let b ← core.cmp.impls.PartialEqShared.eq cmp.partialEqInst stored key
          if b = true then ok (ControlFlow.done (some i)) else do
            let i1 ← i + 1#usize
            ok (ControlFlow.cont i1)) _
        rw [hcmp stored key]
        simp only [bind_tc_ok]
        by_cases heq : stored = key
        · rw [if_pos (by simp [heq]), Aeneas.Std.WP.spec_ok]
          exact ⟨hin, hstartle, matchesAtG_of_get hib ho heq, hno⟩
        · rw [if_neg (by simp [heq])]
          step
          refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
          · omega
          · omega
          · simpa [i1_post] using noMatchInG_succ hno
              (not_matchesAtG_of_ne hib ho heq)
          · omega
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      have hge : N.val ≤ i.val := Nat.not_lt.mp hlt
      have hiN : i.val = N.val := by omega
      simpa [FindPostCallbackG, hiN] using hno
  · exact ⟨hstart, le_rfl, fun j hj hjlt => by omega⟩

/-- The exact raw shape needed by the generated vacant-slot and full-shard branches. -/
def FirstVacantG {K V : Type} {N : Usize} (slots : Array (Option (K × V)) N)
    (first : Usize) : Prop :=
  first < N ∧
    (∀ j : Usize, j.val < first.val →
      ∃ entry, Array.index_usize slots j = ok (some entry)) ∧
    Array.index_usize slots first = ok none

/-- Every in-bounds raw slot is occupied. -/
def FullShardG {K V : Type} {N : Usize} (slots : Array (Option (K × V)) N) : Prop :=
  ∀ i : Usize, i < N → ∃ entry, Array.index_usize slots i = ok (some entry)

private theorem callback_getElem?_of_index {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : Array.index_usize a i = ok x) :
    a.val[i.val]? = some x := by
  unfold Array.index_usize at h
  rw [Array.getElem?_Usize_eq] at h
  split at h <;> simp_all

private theorem callback_index_of_getElem? {α : Type} {N : Usize}
    (a : Array α N) (i : Usize) (x : α) (h : a.val[i.val]? = some x) :
    Array.index_usize a i = ok x := by
  unfold Array.index_usize
  rw [Array.getElem?_Usize_eq, h]

/-- A finite raw shard has a least vacant slot, or every in-bounds slot is occupied. -/
theorem firstVacant_or_fullShardG {K V : Type} {N : Usize}
    (slots : Array (Option (K × V)) N) :
    (∃ first, FirstVacantG slots first) ∨ FullShardG slots := by
  classical
  by_cases hvacant : ∃ k, k < N.val ∧ slots.val[k]? = some none
  · left
    let k := Nat.find hvacant
    have hk : k < N.val := (Nat.find_spec hvacant).1
    have hslotGet : slots.val[k]? = some none := (Nat.find_spec hvacant).2
    let first : Usize := ⟨k, by scalar_tac⟩
    refine ⟨first, hk, ?_, callback_index_of_getElem? slots first none hslotGet⟩
    intro j hj
    have hjN : j < N := Nat.lt_trans hj hk
    obtain ⟨slot, hslot, _⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.index_usize_spec slots j (by simpa using hjN))
    cases slot with
    | none =>
        have hget := callback_getElem?_of_index slots j none hslot
        have hmin := Nat.find_min' hvacant ⟨Nat.lt_trans hj hk, hget⟩
        exact (Nat.not_lt_of_ge hmin hj).elim
    | some entry => exact ⟨entry, hslot⟩
  · right
    intro i hi
    obtain ⟨slot, hslot, _⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.index_usize_spec slots i (by simpa using hi))
    cases slot with
    | none =>
        exact (hvacant ⟨i.val, hi, callback_getElem?_of_index slots i none hslot⟩).elim
    | some entry => exact ⟨entry, hslot⟩

/-- At start zero, the generated wrapper returns either the first matching physical key or exact
    no-match evidence for the entire shard. -/
theorem find_slots_callback_result_exact {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (slots : Array (Option (K × V)) N) (key : K)
    (hcmp : ∀ a b,
      core.cmp.impls.PartialEqShared.eq cmp.partialEqInst a b = ok (decide (a = b))) :
    ∃ result, table.Table.find_slots cmp HomeInst slots key = ok result ∧
      FindPostCallbackG (N := N) key slots.val 0 result := by
  have hspec := find_slots_spec_callback_exact cmp slots key 0#usize (Nat.zero_le _) hcmp
  simpa [table.Table.find_slots] using Aeneas.Std.WP.spec_imp_exists hspec

theorem find_slots_callback_some_exact {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (slots : Array (Option (K × V)) N) (key : K) (i : Usize)
    (hcmp : ∀ a b,
      core.cmp.impls.PartialEqShared.eq cmp.partialEqInst a b = ok (decide (a = b)))
    (hfind : table.Table.find_slots cmp HomeInst slots key = ok (some i)) :
    i < N ∧ (∃ value, Array.index_usize slots i = ok (some (key, value))) ∧
      NoMatchInG key slots.val 0 i.val := by
  obtain ⟨result, hresult, hpost⟩ :=
    find_slots_callback_result_exact cmp HomeInst slots key hcmp
  have heq : result = some i := Aeneas.Std.Result.ok_injective (hresult.symm.trans hfind)
  subst result
  obtain ⟨hi, _, ⟨value, hvalue⟩, hno⟩ := hpost
  exact ⟨hi, ⟨value, callback_index_of_getElem? slots i (some (key, value)) hvalue⟩, hno⟩

theorem find_slots_callback_none_exact {K V H : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K)
    (slots : Array (Option (K × V)) N) (key : K)
    (hcmp : ∀ a b,
      core.cmp.impls.PartialEqShared.eq cmp.partialEqInst a b = ok (decide (a = b)))
    (hfind : table.Table.find_slots cmp HomeInst slots key = ok none) :
    NoMatchInG key slots.val 0 N.val := by
  obtain ⟨result, hresult, hpost⟩ :=
    find_slots_callback_result_exact cmp HomeInst slots key hcmp
  have heq : result = none := Aeneas.Std.Result.ok_injective (hresult.symm.trans hfind)
  simpa [heq, FindPostCallbackG] using hpost

/-- Every stored key occupies the shard selected by the executing home callback. -/
def HomePlacedG {K V H : Type} {N : Usize} (HomeInst : table.Home H K)
    (t : table.Table K V H N) : Prop :=
  ∀ (shard i : Usize) (slots : Array (Option (K × V)) N) (key : K) (value : V),
    Array.index_usize t.shards shard = ok (some slots) →
    Array.index_usize slots i = ok (some (key, value)) →
    table.Table.shard_index HomeInst t key = ok shard

/-- Equal canonical keys identify one physical position across the full outer table. -/
def GlobalKeysDistinctG {K V H : Type} {N : Usize} (t : table.Table K V H N) : Prop :=
  ∀ (leftShard leftSlot rightShard rightSlot : Usize)
      (leftSlots rightSlots : Array (Option (K × V)) N) (key : K) (left right : V),
    Array.index_usize t.shards leftShard = ok (some leftSlots) →
    Array.index_usize leftSlots leftSlot = ok (some (key, left)) →
    Array.index_usize t.shards rightShard = ok (some rightSlots) →
    Array.index_usize rightSlots rightSlot = ok (some (key, right)) →
    leftShard = rightShard ∧ leftSlot = rightSlot

/-- Raw generic table invariant. Search and branch classification remain derived theorems. -/
def PhysicalInvG {K V H : Type} {N : Usize} (cmp : core.cmp.Eq K)
    (HomeInst : table.Home H K) (t : table.Table K V H N) (h : HMap K V) : Prop :=
  HomePlacedG HomeInst t ∧ GlobalKeysDistinctG t ∧ ReadRepG cmp HomeInst t h

/-- Concrete all-empty state returned by the generated generic constructor. -/
def emptyTableG {K V H : Type} {N : Usize} (default : V) (home : H) :
    table.Table K V H N where
  default := default
  home := home
  shards := Array.repeat 256#usize none

theorem table_new_eq_emptyTableG {K V H : Type} {N : Usize}
    (HomeInst : table.Home H K) (default : V) (home : H) :
    table.Table.new N HomeInst default home = ok (emptyTableG (N := N) default home) := rfl

/-- Total home denotation gives an in-bounds generated shard route. -/
theorem shard_index_callback_spec {K V H : Type} {N : Usize}
    (HomeInst : table.Home H K) (t : table.Table K V H N) (homeFn : K → U64) (key : K)
    (hhome : HomeInst.home t.home key = ok (homeFn key)) :
    ∃ shard, table.Table.shard_index HomeInst t key = ok shard ∧ shard.val < 256 := by
  have hs : Aeneas.Std.WP.spec (table.Table.shard_index HomeInst t key)
      (fun shard => shard.val < 256) := by
    unfold table.Table.shard_index
    rw [hhome]
    simp only [bind_tc_ok]
    step
    have hi : i1.val < 256 := by rw [i1_post]; omega
    have hfit : i1.val < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
    change (UScalar.cast .Usize i1).val < 256
    rw [UScalar.cast_val_mod_pow_of_inBounds_eq .Usize i1 hfit]
    exact hi
  obtain ⟨shard, hshard, hbound⟩ := Aeneas.Std.WP.spec_imp_exists hs
  exact ⟨shard, hshard, hbound⟩

set_option maxRecDepth 4096 in
private theorem emptyTableG_outer_none {K V H : Type} {N : Usize}
    (default : V) (home : H) (shard : Usize) (hbound : shard.val < 256) :
    Array.index_usize (emptyTableG (K := K) (N := N) default home).shards shard = ok none := by
  have hlength : shard.val < (emptyTableG (K := K) (N := N) default home).shards.length := by
    change shard.val < 256
    exact hbound
  obtain ⟨x, hx, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (Array.index_usize_spec (emptyTableG (K := K) (N := N) default home).shards
      shard hlength)
  have hslot :
      (emptyTableG (K := K) (N := N) default home).shards.val[shard.val] = none := by
    change (List.replicate 256 none)[shard.val] = none
    rw [List.getElem_replicate]
  calc
    _ = ok x := hx
    _ = ok none := by rw [hp, hslot]

private theorem successful_outer_index_bound {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard : Usize) (entry : Option (Array (Option (K × V)) N))
    (hindex : Array.index_usize t.shards shard = ok entry) : shard.val < 256 := by
  have hget : t.shards.val[shard.val]? = some entry := by
    unfold Array.index_usize at hindex
    rw [Array.getElem?_Usize_eq] at hindex
    split at hindex <;> simp_all
  have hlength : shard.val < t.shards.val.length := by
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at hget
    simp at hget
  simpa [t.shards.property] using hlength

/-- No physical slot in the table stores the queried key. -/
def KeyAbsentG {K V H : Type} {N : Usize} (t : table.Table K V H N) (key : K) : Prop :=
  ∀ (shard i : Usize) (slots : Array (Option (K × V)) N) (value : V),
    Array.index_usize t.shards shard = ok (some slots) →
    Array.index_usize slots i = ok (some (key, value)) → False

/-- The named physical position stores the key and is its unique occurrence in the table. -/
def UniquePresentAtG {K V H : Type} {N : Usize} (t : table.Table K V H N)
    (key : K) (shard i : Usize) (slots : Array (Option (K × V)) N) (value : V) : Prop :=
  Array.index_usize t.shards shard = ok (some slots) ∧
    Array.index_usize slots i = ok (some (key, value)) ∧
    ∀ (otherShard otherSlot : Usize) (otherSlots : Array (Option (K × V)) N)
        (otherValue : V),
      Array.index_usize t.shards otherShard = ok (some otherSlots) →
      Array.index_usize otherSlots otherSlot = ok (some (key, otherValue)) →
      otherShard = shard ∧ otherSlot = i

/-- Generated lookup reaches exactly one insertion branch: a unique present key, an absent key in
    an untouched shard, or an absent key in a populated shard that is vacant or full. -/
def TableBranchG {K V H : Type} {N : Usize} (cmp : core.cmp.Eq K)
    (HomeInst : table.Home H K) (t : table.Table K V H N) (key : K) : Prop :=
  ∃ shard, table.Table.shard_index HomeInst t key = ok shard ∧
    ((∃ slots i value,
        table.Table.find cmp HomeInst t shard key = ok (some i) ∧
        UniquePresentAtG t key shard i slots value) ∨
      (KeyAbsentG t key ∧ table.Table.find cmp HomeInst t shard key = ok none ∧
        (Array.index_usize t.shards shard = ok none ∨
          ∃ slots, Array.index_usize t.shards shard = ok (some slots) ∧
            ((∃ first, FirstVacantG slots first) ∨ FullShardG slots))))

private theorem successful_inner_index_bound {K V : Type} {N : Usize}
    (slots : Array (Option (K × V)) N) (i : Usize) (entry : Option (K × V))
    (hindex : Array.index_usize slots i = ok entry) : i < N := by
  have hget := callback_getElem?_of_index slots i entry hindex
  have hlength : i.val < slots.val.length := by
    by_contra hn
    rw [List.getElem?_eq_none (Nat.not_lt.mp hn)] at hget
    simp at hget
  simpa [slots.property] using hlength

/-- Primitive callback laws plus the raw physical invariant derive all generated lookup branches. -/
theorem table_branch_callback_exact {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h) :
    TableBranchG cmp HomeInst t key := by
  obtain ⟨hplaced, hdistinct, _⟩ := hinv
  obtain ⟨shard, hshard, hshardBound⟩ :=
    shard_index_callback_spec HomeInst t homeFn key (laws.home_total key)
  have houterBound : shard.val < t.shards.length := by
    simpa [t.shards.property] using hshardBound
  obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
    (Array.index_usize_spec t.shards shard houterBound)
  refine ⟨shard, hshard, ?_⟩
  cases outer with
  | none =>
      right
      have hfind : table.Table.find cmp HomeInst t shard key = ok none := by
        unfold table.Table.find
        rw [houter]
        simp
      refine ⟨?_, hfind, Or.inl houter⟩
      intro otherShard otherSlot otherSlots value hotherOuter hotherSlot
      have hotherRoute := hplaced otherShard otherSlot otherSlots key value
        hotherOuter hotherSlot
      have heq : otherShard = shard :=
        Aeneas.Std.Result.ok_injective (hotherRoute.symm.trans hshard)
      subst otherShard
      rw [houter] at hotherOuter
      simp at hotherOuter
  | some slots =>
      obtain ⟨result, hresult, hpost⟩ :=
        find_slots_callback_result_exact cmp HomeInst slots key laws.cmp_exact
      have hfind : table.Table.find cmp HomeInst t shard key = ok result := by
        unfold table.Table.find
        rw [houter]
        simpa only [bind_tc_ok] using hresult
      cases result with
      | some i =>
          left
          obtain ⟨hi, _, ⟨value, hget⟩, _⟩ := hpost
          have hslot := callback_index_of_getElem? slots i (some (key, value)) hget
          refine ⟨slots, i, value, hfind, houter, hslot, ?_⟩
          intro otherShard otherSlot otherSlots otherValue hotherOuter hotherSlot
          have hunique := hdistinct shard i otherShard otherSlot slots otherSlots key value
            otherValue houter hslot hotherOuter hotherSlot
          exact ⟨hunique.1.symm, hunique.2.symm⟩
      | none =>
          right
          have hno : NoMatchInG key slots.val 0 N.val := by
            simpa [FindPostCallbackG] using hpost
          refine ⟨?_, hfind, Or.inr ⟨slots, houter, firstVacant_or_fullShardG slots⟩⟩
          intro otherShard otherSlot otherSlots value hotherOuter hotherSlot
          have hotherRoute := hplaced otherShard otherSlot otherSlots key value
            hotherOuter hotherSlot
          have hshardEq : otherShard = shard :=
            Aeneas.Std.Result.ok_injective (hotherRoute.symm.trans hshard)
          subst otherShard
          have hslotsEq : otherSlots = slots := by
            have := Aeneas.Std.Result.ok_injective (hotherOuter.symm.trans houter)
            exact Option.some.inj this
          subst otherSlots
          have hi := successful_inner_index_bound slots otherSlot (some (key, value)) hotherSlot
          have hget := callback_getElem?_of_index slots otherSlot (some (key, value)) hotherSlot
          exact hno otherSlot.val (Nat.zero_le _) hi ⟨value, hget⟩

/-- Non-circular classification of the exact state/result pairs produced by generic insertion. -/
def CallbackInsertOutcomeG {K V H : Type} {N : Usize} (t : table.Table K V H N)
    (key : K) (value : V) (mergeFn : V → V → V)
    (result : core.result.Result Unit table.StorageError × table.Table K V H N) : Prop :=
  (∃ shard slots i old,
      UniquePresentAtG t key shard i slots old ∧
      result = (core.result.Result.Ok (),
        replacePresentG t shard slots i key (mergeFn value old))) ∨
  (∃ shard,
      0#usize < N ∧ Array.index_usize t.shards shard = ok none ∧
      result = (core.result.Result.Ok (),
        insertEmptyShardG t shard key (mergeFn value t.default))) ∨
  (∃ shard slots first,
      Array.index_usize t.shards shard = ok (some slots) ∧ FirstVacantG slots first ∧
      result = (core.result.Result.Ok (),
        replacePresentG t shard slots first key (mergeFn value t.default))) ∨
  result = (core.result.Result.Err table.StorageError.ShardFull, t)

/-- Every invariant generic table insertion executes one of the exact callback-defined updates, or
    returns `ShardFull` with the input table bit-for-bit unchanged. -/
theorem insert_with_callback_result_exact {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value : V)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h) :
    ∃ result,
      table.Table.insert_with cmp cloneV HomeInst MergeInst t merge key value = ok result ∧
      CallbackInsertOutcomeG t key value mergeFn result := by
  obtain ⟨shard, hshard, hpresent | habsent⟩ :=
    table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst t merge homeFn mergeFn
      h key laws hinv
  · obtain ⟨slots, i, old, hfind, houter, hslot, hunique⟩ := hpresent
    let result : core.result.Result Unit table.StorageError × table.Table K V H N :=
      (core.result.Result.Ok (), replacePresentG t shard slots i key (mergeFn value old))
    refine ⟨result, insert_with_present_callback_exact cmp cloneK cloneV HomeInst MergeInst
      t merge homeFn mergeFn key key value old shard i slots laws hshard hfind houter hslot, ?_⟩
    exact Or.inl ⟨shard, slots, i, old, ⟨houter, hslot, hunique⟩, rfl⟩
  · obtain ⟨_, hfind, houterEmpty | ⟨slots, houter, hvacant | hfull⟩⟩ := habsent
    · by_cases hN : 0#usize < N
      · have hshardBound : shard < 256#usize := by
          simpa using successful_outer_index_bound t shard none houterEmpty
        let result : core.result.Result Unit table.StorageError × table.Table K V H N :=
          (core.result.Result.Ok (), insertEmptyShardG t shard key (mergeFn value t.default))
        refine ⟨result, insert_with_empty_shard_callback_exact cmp cloneK cloneV HomeInst
          MergeInst t merge homeFn mergeFn key value shard laws hshard hfind houterEmpty hN
          hshardBound, ?_⟩
        exact Or.inr (Or.inl ⟨shard, hN, houterEmpty, rfl⟩)
      · have hzero : N = 0#usize := by
          scalar_tac
        subst N
        let result : core.result.Result Unit table.StorageError × table.Table K V H 0#usize :=
          (core.result.Result.Err table.StorageError.ShardFull, t)
        refine ⟨result, insert_with_zero_callback_exact cmp cloneV HomeInst MergeInst t merge
          key value shard hshard hfind, ?_⟩
        exact Or.inr (Or.inr (Or.inr rfl))
    · obtain ⟨first, hfirst, hprefix, hslot⟩ := hvacant
      let result : core.result.Result Unit table.StorageError × table.Table K V H N :=
        (core.result.Result.Ok (),
          replacePresentG t shard slots first key (mergeFn value t.default))
      refine ⟨result, insert_with_populated_vacant_callback_exact cmp cloneK cloneV HomeInst
        MergeInst t merge homeFn mergeFn key value shard first slots laws hshard hfind houter
        hfirst hprefix hslot, ?_⟩
      exact Or.inr (Or.inr (Or.inl ⟨shard, slots, first, houter,
        ⟨hfirst, hprefix, hslot⟩, rfl⟩))
    · by_cases hN : 0#usize < N
      · let result : core.result.Result Unit table.StorageError × table.Table K V H N :=
          (core.result.Result.Err table.StorageError.ShardFull, t)
        refine ⟨result, insert_with_populated_full_callback_exact cmp cloneV HomeInst MergeInst
          t merge key value shard slots hN hshard hfind houter hfull, ?_⟩
        exact Or.inr (Or.inr (Or.inr rfl))
      · have hzero : N = 0#usize := by
          scalar_tac
        subst N
        let result : core.result.Result Unit table.StorageError × table.Table K V H 0#usize :=
          (core.result.Result.Err table.StorageError.ShardFull, t)
        refine ⟨result, insert_with_zero_callback_exact cmp cloneV HomeInst MergeInst t merge
          key value shard hshard hfind, ?_⟩
        exact Or.inr (Or.inr (Or.inr rfl))

private theorem replacePresentG_entry_from_old {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (old new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old)))
    (s j : Usize) (current : Array (Option (K × V)) N) (stored : K) (storedValue : V)
    (hs : Array.index_usize (replacePresentG t shard slots i key new).shards s =
      ok (some current))
    (hj : Array.index_usize current j = ok (some (stored, storedValue))) :
    ∃ oldSlots oldValue,
      Array.index_usize t.shards s = ok (some oldSlots) ∧
      Array.index_usize oldSlots j = ok (some (stored, oldValue)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using successful_outer_index_bound t shard (some slots) houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots.set i (some (key, new)) := by
      have hs' := callback_getElem?_of_index
        (replacePresentG t shard slots i key new).shards shard (some current) hs
      simp only [replacePresentG] at hs'
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound] at hs'
      exact Option.some.inj (Option.some.inj hs').symm
    subst current
    by_cases hji : j = i
    · subst j
      have hnew := callback_getElem?_of_index (slots.set i (some (key, new))) i
        (some (stored, storedValue)) hj
      have hi : i.val < slots.val.length := by
        simpa [slots.property] using successful_inner_index_bound slots i (some (key, old)) hslot
      rw [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi] at hnew
      have hp : (stored, storedValue) = (key, new) :=
        Option.some.inj (Option.some.inj hnew.symm)
      have hk : stored = key := congrArg Prod.fst hp
      subst stored
      exact ⟨slots, old, houter, hslot⟩
    · have hjiVal : j.val ≠ i.val := fun h => hji (UScalar.eq_of_val_eq h)
      have hjOld : Array.index_usize slots j = ok (some (stored, storedValue)) := by
        simpa [Array.index_usize, Array.set_val_eq, hjiVal] using hj
      exact ⟨slots, storedValue, houter, hjOld⟩
  · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    have hsOld : Array.index_usize t.shards s = ok (some current) := by
      simpa [replacePresentG, Array.index_usize, Array.set_val_eq, hne] using hs
    exact ⟨current, storedValue, hsOld, hj⟩

/-- Replacing the value at the unique present-key position preserves physical placement. -/
theorem homePlacedG_replacePresent {K V H : Type} {N : Usize}
    (HomeInst : table.Home H K) (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (old new : V)
    (hhome : HomePlacedG HomeInst t)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    HomePlacedG HomeInst (replacePresentG t shard slots i key new) := by
  intro s j current stored storedValue hs hj
  obtain ⟨oldSlots, oldValue, hsOld, hjOld⟩ :=
    replacePresentG_entry_from_old t shard i slots key old new houter hslot
      s j current stored storedValue hs hj
  change table.Table.shard_index HomeInst t stored = ok s
  exact hhome s j oldSlots stored oldValue hsOld hjOld

/-- Replacing the value at a unique present-key position preserves global key uniqueness. -/
theorem globalKeysDistinctG_replacePresent {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (old new : V)
    (hdistinct : GlobalKeysDistinctG t)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    GlobalKeysDistinctG (replacePresentG t shard slots i key new) := by
  intro leftShard leftSlot rightShard rightSlot leftSlots rightSlots stored left right
    hleftOuter hleftSlot hrightOuter hrightSlot
  obtain ⟨leftOldSlots, leftOld, hleftOuterOld, hleftSlotOld⟩ :=
    replacePresentG_entry_from_old t shard i slots key old new houter hslot
      leftShard leftSlot leftSlots stored left hleftOuter hleftSlot
  obtain ⟨rightOldSlots, rightOld, hrightOuterOld, hrightSlotOld⟩ :=
    replacePresentG_entry_from_old t shard i slots key old new houter hslot
      rightShard rightSlot rightSlots stored right hrightOuter hrightSlot
  exact hdistinct leftShard leftSlot rightShard rightSlot leftOldSlots rightOldSlots stored
    leftOld rightOld hleftOuterOld hleftSlotOld hrightOuterOld hrightSlotOld

/-- The present-key exact update preserves both raw physical clauses of `PhysicalInvG`. -/
theorem physicalClausesG_replacePresent {K V H : Type} {N : Usize}
    (cmp : core.cmp.Eq K) (HomeInst : table.Home H K) (t : table.Table K V H N)
    (h : HMap K V) (key : K) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (old new : V)
    (hinv : PhysicalInvG cmp HomeInst t h)
    (hpresent : UniquePresentAtG t key shard i slots old) :
    HomePlacedG HomeInst (replacePresentG t shard slots i key new) ∧
      GlobalKeysDistinctG (replacePresentG t shard slots i key new) := by
  obtain ⟨hhome, hdistinct, _⟩ := hinv
  obtain ⟨houter, hslot, _⟩ := hpresent
  exact ⟨homePlacedG_replacePresent HomeInst t shard i slots key old new hhome houter hslot,
    globalKeysDistinctG_replacePresent t shard i slots key old new hdistinct houter hslot⟩

/-- Placement and uniqueness make an explicitly stored key observable through generated `get`. -/
theorem get_callback_of_uniquePresent {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (key : K) (shard i : Usize) (slots : Array (Option (K × V)) N) (value : V)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hhome : HomePlacedG HomeInst t) (hdistinct : GlobalKeysDistinctG t)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, value))) :
    table.Table.get cmp HomeInst t key = ok value := by
  have hroute := hhome shard i slots key value houter hslot
  obtain ⟨found, hfound, hpost⟩ :=
    find_slots_callback_result_exact cmp HomeInst slots key laws.cmp_exact
  cases found with
  | none =>
      have hno : NoMatchInG key slots.val 0 N.val := by
        simpa [FindPostCallbackG] using hpost
      have hi := successful_inner_index_bound slots i (some (key, value)) hslot
      have hget := callback_getElem?_of_index slots i (some (key, value)) hslot
      exact (hno i.val (Nat.zero_le _) hi ⟨value, hget⟩).elim
  | some found =>
      obtain ⟨_, _, ⟨foundValue, hfoundGet⟩, _⟩ := hpost
      have hfoundSlot := callback_index_of_getElem? slots found
        (some (key, foundValue)) hfoundGet
      have hsame := hdistinct shard found shard i slots slots key foundValue value
        houter hfoundSlot houter hslot
      have hfoundEq : found = i := hsame.2
      subst found
      have hvalueEq : foundValue = value := by
        have hp : (key, foundValue) = (key, value) := Option.some.inj
          (Aeneas.Std.Result.ok_injective (hfoundSlot.symm.trans hslot))
        exact congrArg Prod.snd hp
      subst foundValue
      have hfind : table.Table.find cmp HomeInst t shard key = ok (some i) := by
        unfold table.Table.find
        rw [houter]
        simpa only [bind_tc_ok] using hfound
      unfold table.Table.get
      rw [hroute]
      simp only [bind_tc_ok]
      rw [hfind]
      simp only [bind_tc_ok]
      rw [houter]
      simp only [bind_tc_ok]
      rw [hslot]
      simp

/-- Physical absence makes the generated read return the table default. -/
theorem get_callback_of_absent {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (key : K)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (habsent : KeyAbsentG t key) :
    table.Table.get cmp HomeInst t key = ok t.default := by
  obtain ⟨shard, hroute, hbound⟩ :=
    shard_index_callback_spec HomeInst t homeFn key (laws.home_total key)
  have houterBound : shard.val < t.shards.length := by
    simpa [t.shards.property] using hbound
  obtain ⟨outer, houter, _⟩ := Aeneas.Std.WP.spec_imp_exists
    (Array.index_usize_spec t.shards shard houterBound)
  cases outer with
  | none =>
      unfold table.Table.get table.Table.find
      rw [hroute]
      simp only [bind_tc_ok]
      rw [houter]
      simp
  | some slots =>
      obtain ⟨found, hfound, hpost⟩ :=
        find_slots_callback_result_exact cmp HomeInst slots key laws.cmp_exact
      cases found with
      | some i =>
          obtain ⟨_, _, ⟨foundValue, hfoundGet⟩, _⟩ := hpost
          have hfoundSlot := callback_index_of_getElem? slots i
            (some (key, foundValue)) hfoundGet
          exact (habsent shard i slots foundValue houter hfoundSlot).elim
      | none =>
          have hfind : table.Table.find cmp HomeInst t shard key = ok none := by
            unfold table.Table.find
            rw [houter]
            simpa only [bind_tc_ok] using hfound
          unfold table.Table.get
          rw [hroute]
          simp only [bind_tc_ok]
          rw [hfind]
          simp

private theorem replacePresentG_updated_position {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (old new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    Array.index_usize (replacePresentG t shard slots i key new).shards shard =
        ok (some (slots.set i (some (key, new)))) ∧
      Array.index_usize (slots.set i (some (key, new))) i = ok (some (key, new)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using successful_outer_index_bound t shard (some slots) houter
  have hi : i.val < slots.val.length := by
    simpa [slots.property] using successful_inner_index_bound slots i (some (key, old)) hslot
  constructor
  · apply callback_index_of_getElem?
    simp [replacePresentG, Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound]
  · apply callback_index_of_getElem?
    simp [Array.set_val_eq, List.getElem?_set_eq_of_lt _ hi]

private theorem replacePresentG_preserves_other_key {K V H : Type} {N : Usize}
    (t : table.Table K V H N) (shard i : Usize)
    (slots : Array (Option (K × V)) N) (key : K) (old new : V)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old)))
    (otherKey : K) (hne : otherKey ≠ key) (s j : Usize)
    (current : Array (Option (K × V)) N) (currentValue : V)
    (hs : Array.index_usize t.shards s = ok (some current))
    (hj : Array.index_usize current j = ok (some (otherKey, currentValue))) :
    ∃ updatedSlots,
      Array.index_usize (replacePresentG t shard slots i key new).shards s =
        ok (some updatedSlots) ∧
      Array.index_usize updatedSlots j = ok (some (otherKey, currentValue)) := by
  have hshardBound : shard.val < t.shards.val.length := by
    simpa [t.shards.property] using successful_outer_index_bound t shard (some slots) houter
  by_cases hss : s = shard
  · subst s
    have hcurrent : current = slots := by
      have := Aeneas.Std.Result.ok_injective (hs.symm.trans houter)
      exact Option.some.inj this
    subst current
    have hji : j ≠ i := by
      intro heq
      subst j
      have hp := Aeneas.Std.Result.ok_injective (hj.symm.trans hslot)
      have hpPair : (otherKey, currentValue) = (key, old) := Option.some.inj hp
      exact hne (congrArg Prod.fst hpPair)
    have hjiVal : j.val ≠ i.val := fun h => hji (UScalar.eq_of_val_eq h)
    refine ⟨slots.set i (some (key, new)), ?_, ?_⟩
    · apply callback_index_of_getElem?
      simp [replacePresentG, Array.set_val_eq, List.getElem?_set_eq_of_lt _ hshardBound]
    · simpa [Array.index_usize, Array.set_val_eq, hjiVal] using hj
  · have hneShard : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
    refine ⟨current, ?_, hj⟩
    simpa [replacePresentG, Array.index_usize, Array.set_val_eq, hneShard] using hs

/-- Exact reads from the present-key update agree with canonical `insertWith`. -/
theorem readRepG_replacePresent {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value old : V) (shard i : Usize)
    (slots : Array (Option (K × V)) N)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h)
    (hpresent : UniquePresentAtG t key shard i slots old) :
    ReadRepG cmp HomeInst (replacePresentG t shard slots i key (mergeFn value old))
      (insertWith mergeFn key value h) := by
  have hinvCopy := hinv
  have hpresentCopy := hpresent
  obtain ⟨hhome, hdistinct, hread⟩ := hinv
  obtain ⟨houter, hslot, _⟩ := hpresent
  let updated := replacePresentG t shard slots i key (mergeFn value old)
  have hphysical := physicalClausesG_replacePresent cmp HomeInst t h key shard i slots old
    (mergeFn value old) hinvCopy hpresentCopy
  obtain ⟨hhomeUpdated, hdistinctUpdated⟩ := hphysical
  obtain ⟨houterUpdated, hslotUpdated⟩ := replacePresentG_updated_position
    t shard i slots key old (mergeFn value old) houter hslot
  intro query
  by_cases hq : query = key
  · subst query
    have holdGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst t merge
      homeFn mergeFn key shard i slots old laws hhome hdistinct houter hslot
    have holdEq : old = h key := Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread key))
    have hnewGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst updated
      merge homeFn mergeFn key shard i (slots.set i (some (key, mergeFn value old)))
      (mergeFn value old) laws hhomeUpdated hdistinctUpdated houterUpdated hslotUpdated
    simpa [updated, insertWith, holdEq] using hnewGet
  · obtain ⟨oldShard, _, hpresentOld | habsentOld⟩ :=
      table_branch_callback_exact cmp cloneK cloneV HomeInst MergeInst t merge homeFn mergeFn
        h query laws hinvCopy
    · obtain ⟨oldSlots, oldIndex, oldValue, _, holdOuter, holdSlot, _⟩ := hpresentOld
      obtain ⟨updatedSlots, hupdatedOuter, hupdatedSlot⟩ :=
        replacePresentG_preserves_other_key t shard i slots key old (mergeFn value old)
          houter hslot query hq oldShard oldIndex oldSlots oldValue holdOuter holdSlot
      have holdGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst t merge
        homeFn mergeFn query oldShard oldIndex oldSlots oldValue laws hhome hdistinct
        holdOuter holdSlot
      have holdEq : oldValue = h query :=
        Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread query))
      have hnewGet := get_callback_of_uniquePresent cmp cloneK cloneV HomeInst MergeInst updated
        merge homeFn mergeFn query oldShard oldIndex updatedSlots oldValue laws hhomeUpdated
        hdistinctUpdated hupdatedOuter hupdatedSlot
      simpa [updated, insertWith, hq, holdEq] using hnewGet
    · obtain ⟨habsent, _, _⟩ := habsentOld
      have habsentUpdated : KeyAbsentG updated query := by
        intro s j current currentValue hs hj
        obtain ⟨oldSlots, oldValue, hsOld, hjOld⟩ :=
          replacePresentG_entry_from_old t shard i slots key old (mergeFn value old)
            houter hslot s j current query currentValue hs hj
        exact habsent s j oldSlots oldValue hsOld hjOld
      have holdGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst t merge
        homeFn mergeFn query laws habsent
      have hdefaultEq : t.default = h query :=
        Aeneas.Std.Result.ok_injective (holdGet.symm.trans (hread query))
      have hnewGet := get_callback_of_absent cmp cloneK cloneV HomeInst MergeInst updated merge
        homeFn mergeFn query laws habsentUpdated
      calc
        table.Table.get cmp HomeInst updated query = ok updated.default := hnewGet
        _ = ok t.default := rfl
        _ = ok (h query) := congrArg ok hdefaultEq
        _ = ok (insertWith mergeFn key value h query) := by
          simp [insertWith, hq]

/-- The exact present-key update establishes the full physical invariant for canonical insertion. -/
theorem physicalInvG_replacePresent {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (t : table.Table K V H N) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (h : HMap K V) (key : K) (value old : V) (shard i : Usize)
    (slots : Array (Option (K × V)) N)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst t.home merge homeFn mergeFn)
    (hinv : PhysicalInvG cmp HomeInst t h)
    (hpresent : UniquePresentAtG t key shard i slots old) :
    PhysicalInvG cmp HomeInst (replacePresentG t shard slots i key (mergeFn value old))
      (insertWith mergeFn key value h) := by
  obtain ⟨hphysical, hdistinct⟩ := physicalClausesG_replacePresent cmp HomeInst t h key
    shard i slots old (mergeFn value old) hinv hpresent
  exact ⟨hphysical, hdistinct, readRepG_replacePresent cmp cloneK cloneV HomeInst MergeInst t
    merge homeFn mergeFn h key value old shard i slots laws hinv hpresent⟩

/-- The real generic constructor establishes placement, uniqueness, and total default reads. -/
theorem emptyTableG_physicalInv {K V H F : Type} {N : Usize} [DecidableEq K]
    (cmp : core.cmp.Eq K) (cloneK : core.clone.Clone K) (cloneV : core.clone.Clone V)
    (HomeInst : table.Home H K) (MergeInst : table.Merge F V)
    (default : V) (home : H) (merge : F) (homeFn : K → U64) (mergeFn : V → V → V)
    (laws : CallbackLawsG cmp cloneK cloneV HomeInst MergeInst home merge homeFn mergeFn) :
    PhysicalInvG cmp HomeInst (emptyTableG (N := N) default home) (fun _ => default) := by
  refine ⟨?_, ?_, ?_⟩
  · intro shard i slots key value houter _
    have hbound := successful_outer_index_bound
      (emptyTableG (K := K) (N := N) default home) shard (some slots) houter
    have hempty := emptyTableG_outer_none (K := K) (N := N) default home shard hbound
    rw [houter] at hempty
    simp at hempty
  · intro leftShard leftSlot rightShard rightSlot leftSlots rightSlots key left right
      hleftOuter _ _ _
    have hbound := successful_outer_index_bound
      (emptyTableG (K := K) (N := N) default home) leftShard (some leftSlots) hleftOuter
    have hempty := emptyTableG_outer_none (K := K) (N := N) default home leftShard hbound
    rw [hleftOuter] at hempty
    simp at hempty
  · intro key
    obtain ⟨shard, hshard, hbound⟩ := shard_index_callback_spec HomeInst
      (emptyTableG (N := N) default home) homeFn key (laws.home_total key)
    have houter := emptyTableG_outer_none (K := K) (N := N) default home shard hbound
    unfold table.Table.get
    rw [hshard]
    simp only [bind_tc_ok]
    unfold table.Table.find
    rw [houter]
    simp [emptyTableG]

end Apeiron.TableRefinement
