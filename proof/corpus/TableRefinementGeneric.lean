import TableRefinement

/-! # Generic-`N` refinement of the executing fixed-shard word table -/

namespace Apeiron.TableRefinement

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

open Aeneas Aeneas.Std Result Apeiron.Table
open apeiron_word

/-! ## Generic-N present-key write preservation -/

abbrev WordTableN (N : Usize) := table.Table I64 I64 table.WordHome N

/-- A total functional denotation for a generated state, parameterized by a physical locator.
    Missing, out-of-range, empty, and mismatched locations denote the stored default. -/
def stateRead {N : Usize} (locate : I64 → Option (Usize × Usize)) (t : WordTableN N) :
    HMap I64 I64 := fun key =>
  match locate key with
  | none => t.default
  | some (shard, i) =>
    match t.shards.val[shard.val]? with
    | none | some none => t.default
    | some (some slots) =>
      match slots.val[i.val]? with
      | some (some (storedKey, value)) => if storedKey = key then value else t.default
      | _ => t.default

/-- Extensional representation through the total physical-state denotation. -/
def FunctionalRep {N : Usize} (locate : I64 → Option (Usize × Usize))
    (t : WordTableN N) (h : HMap I64 I64) : Prop := stateRead locate t = h

/-- A concrete locator that inhabits the present-key preservation premises. -/
def zeroLocator (key : I64) : Option (Usize × Usize) :=
  if key = 0#i64 then some (0#usize, 0#usize) else none

/-- The exact state update performed by the generated present-key branch. -/
def replacePresent {N : Usize} (t : WordTableN N) (shard : Usize)
    (slots : Array (Option (I64 × I64)) N) (i : Usize) (storedKey value : I64) :
    WordTableN N :=
  { t with shards := t.shards.set shard (some (slots.set i (some (storedKey, value)))) }

/-- The generated present-key branch updates the located slot with `merge new old`.
    The premises expose the successful generated callback/search/index operations; later physical
    invariants discharge them. -/
theorem insert_with_present_exact {N : Usize} (t : WordTableN N) (key value : I64)
    (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (storedKey old : I64)
    (hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok (some i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (storedKey, old))) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Ok (), replacePresent t shard slots i storedKey
        (core.num.I64.wrapping_add value old)) := by
  have hmouter : Array.index_mut_usize t.shards shard =
      ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  have hminner : Array.index_mut_usize slots i = ok (some (storedKey, old), slots.set i) := by
    simp [Array.index_mut_usize, hslot]
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp [hmouter, hminner, table.WordMerge.Insts.Apeiron_wordTableMergeI64,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge, replacePresent]

/-- Replacing a located present key is the canonical new-first function update.
    This is the pure state-denotation step consumed by the generated-body theorem above. -/
theorem stateRead_replacePresent {N : Usize} (locate : I64 → Option (Usize × Usize))
    (t : WordTableN N) (key value : I64) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) N) (old : I64)
    (hlocate : locate key = some (shard, i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    stateRead locate (replacePresent t shard slots i key
      (core.num.I64.wrapping_add value old)) =
      insertWith core.num.I64.wrapping_add key value (stateRead locate t) := by
  have houter' : t.shards.val[shard.val]? = some (some slots) := by
    unfold Array.index_usize at houter
    split at houter <;> simp_all
  have hslot' : slots.val[i.val]? = some (some (key, old)) := by
    unfold Array.index_usize at hslot
    split at hslot <;> simp_all
  have hshardBound : shard.val < 256 := by
    have := t.shards.property
    grind
  have hiBound : i.val < N.val := by
    have := slots.property
    grind
  have hshardLen : shard.val < t.shards.val.length := by
    rw [t.shards.property]
    exact hshardBound
  have hiLen : i.val < slots.val.length := by
    rw [slots.property]
    exact hiBound
  have houterGet : t.shards.val[shard.val] = some slots := by
    rw [List.getElem?_eq_getElem hshardLen] at houter'
    exact Option.some.inj houter'
  have hslotGet : slots.val[i.val] = some (key, old) := by
    rw [List.getElem?_eq_getElem hiLen] at hslot'
    exact Option.some.inj hslot'
  funext x
  simp only [insertWith]
  by_cases hx : x = key
  · subst x
    simp [stateRead, replacePresent, hlocate, houterGet, hslotGet, hshardBound, hiBound]
  · rw [if_neg hx]
    have hx' : key ≠ x := Ne.symm hx
    simp only [stateRead]
    cases hloc : locate x with
    | none => simp [replacePresent]
    | some p =>
      rcases p with ⟨s, j⟩
      simp only
      by_cases hs : s = shard
      · subst s
        by_cases hj : j = i
        · subst j
          simp [replacePresent, houterGet, hslotGet, hshardBound, hiBound, hx']
        · have hji : j.val ≠ i.val := by
            intro h
            apply hj
            apply UScalar.eq_of_val_eq
            exact h
          simp [replacePresent, houterGet, hshardBound, Array.set_val_eq, hji]
      · have hss : shard.val ≠ s.val := by
          intro h
          apply hs
          apply UScalar.eq_of_val_eq
          exact h.symm
        simp [replacePresent, Array.set_val_eq, hss]

/-- Generic-`N` present-key preservation for the real generated callback instances and body. -/
theorem insert_with_present_preserves {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N) (key value : I64)
    (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (old : I64)
    (hlocate : locate key = some (shard, i))
    (hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok (some i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    ∃ t', table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
        ok (core.result.Result.Ok (), t') ∧
      stateRead locate t' = insertWith core.num.I64.wrapping_add key value
        (stateRead locate t) := by
  refine ⟨replacePresent t shard slots i key (core.num.I64.wrapping_add value old), ?_, ?_⟩
  · exact insert_with_present_exact t key value shard i slots key old hshard hfind houter hslot
  · exact stateRead_replacePresent locate t key value shard i slots old hlocate houter hslot

/-- Successful present-key insertion preserves the functional representation at arbitrary `N`.
    The callback, search, and slot premises are explicit obligations of the future physical `Rep`. -/
theorem insert_with_present_rep_preservation {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t t' : WordTableN N) (h : HMap I64 I64)
    (key value : I64) (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (old : I64)
    (hrep : FunctionalRep locate t h)
    (hlocate : locate key = some (shard, i))
    (hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok (some i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old)))
    (hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Ok (), t')) :
    FunctionalRep locate t'
      (insertWith core.num.I64.wrapping_add key value h) := by
  have hexact := insert_with_present_exact t key value shard i slots key old
    hshard hfind houter hslot
  rw [hinsert] at hexact
  have ht' : t' = replacePresent t shard slots i key
      (core.num.I64.wrapping_add value old) := by
    exact (Prod.mk.inj (Aeneas.Std.Result.ok_injective hexact)).2
  subst t'
  unfold FunctionalRep at hrep ⊢
  rw [stateRead_replacePresent locate t key value shard i slots old hlocate houter hslot,
    hrep]

/-- The generic preservation premises are jointly inhabited by an actual generated write. -/
theorem insert_with_present_joint_witness (default first second : I64) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64
        (oneWordTable default first) () 0#i64 second =
        ok (core.result.Result.Ok (), replacedWordTable default first second) ∧
      FunctionalRep zeroLocator (replacedWordTable default first second)
        (insertWith core.num.I64.wrapping_add 0#i64 second
          (stateRead zeroLocator (oneWordTable default first))) := by
  have hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 (oneWordTable default first) 0#i64 =
      ok 0#usize := by
    simp [table.Table.shard_index,
      table.WordHome.Insts.Apeiron_wordTableHomeI64.home, HMod.hMod,
      UScalar.rem, IScalar.hcast]
    rw [u64_zero_mod_256, cast_u64_zero_usize]
  have houter : Array.index_usize (oneWordTable default first).shards 0#usize =
      ok (some (oneSlots default first)) := by
    set_option maxRecDepth 4096 in
      simp [oneWordTable, Array.index_usize, Array.repeat_val]
  have hslot : Array.index_usize (oneSlots default first) 0#usize =
      ok (some (0#i64, core.num.I64.wrapping_add first default)) := by
    set_option maxRecDepth 4096 in
      simp [oneSlots, Array.index_usize, Array.repeat_val]
  refine ⟨table_insert_present_zero default first second, ?_⟩
  exact insert_with_present_rep_preservation zeroLocator (oneWordTable default first)
    (replacedWordTable default first second) (stateRead zeroLocator (oneWordTable default first))
    0#i64 second 0#usize 0#usize (oneSlots default first)
    (core.num.I64.wrapping_add first default) rfl (by simp [zeroLocator]) hshard
    (oneWordTable_find default first) houter hslot (table_insert_present_zero default first second)

/-! ## Generic-N executing read refinement -/

def FindInvN {N : Usize} (key : I64) (slots : List (Option (I64 × I64)))
    (start : Nat) (i : Usize) : Prop :=
  i.val ≤ N.val ∧ start ≤ i.val ∧ NoMatchIn key slots start i.val

def FindPostN {N : Usize} (key : I64) (slots : List (Option (I64 × I64)))
    (start : Nat) : Option Usize → Prop
  | some i => i.val < N.val ∧ start ≤ i.val ∧ MatchesAt key slots i.val ∧
      NoMatchIn key slots start i.val
  | none => NoMatchIn key slots start N.val

/-- The generated bounded search is total and sound for every generated array length `N`. -/
theorem find_slots_spec_generic {N : Usize} (slots : Array (Option (I64 × I64)) N)
    (key : I64) (start : Usize) (hstart : start.val ≤ N.val) :
    table.Table.find_slots_loop core.cmp.EqI64 slots key start
      ⦃ r => FindPostN (N := N) key slots.val start.val r ⦄ := by
  unfold table.Table.find_slots_loop
  apply loop.spec_decr_nat
    (measure := fun i : Usize => N.val - i.val)
    (inv := fun i => FindInvN (N := N) key slots.val start.val i)
    (post := fun r => FindPostN (N := N) key slots.val start.val r)
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
        · simpa [r_post] using noMatchIn_succ hno (not_matchesAt_of_none hib ho)
        · omega
      · have ho : slots.val[i.val] = some (stored, value) := o_post.symm
        step
        split
        · rename_i heq
          have hk : stored = key := by
            rw [heq] at r_post
            exact of_decide_eq_true r_post.symm
          simp only [Aeneas.Std.WP.spec_ok]
          exact ⟨hin, hstartle, matchesAt_of_get hib ho hk, hno⟩
        · rename_i hne
          have hk : stored ≠ key := by
            intro hk
            exact hne (by simp [r_post, hk])
          step
          refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
          · omega
          · omega
          · simpa [i1_post] using noMatchIn_succ hno (not_matchesAt_of_ne hib ho hk)
          · omega
    · rw [if_neg hlt, Aeneas.Std.WP.spec_ok]
      have hiN : i.val = N.val := by
        have hge : N.val ≤ i.val := not_lt.mp hlt
        omega
      simpa [FindPostN, hiN] using hno
  · exact ⟨hstart, le_rfl, fun j hj hjlt => by omega⟩

/-- The bounded-search postcondition determines one result. -/
theorem findPostN_unique {N : Usize} {key : I64}
    {slots : List (Option (I64 × I64))} {start : Nat} {a b : Option Usize}
    (ha : FindPostN (N := N) key slots start a)
    (hb : FindPostN (N := N) key slots start b) : a = b := by
  cases a with
  | none =>
      cases b with
      | none => rfl
      | some j =>
          exact (ha j.val hb.2.1 hb.1 hb.2.2.1).elim
  | some i =>
      cases b with
      | none =>
          exact (hb i.val ha.2.1 ha.1 ha.2.2.1).elim
      | some j =>
          congr 1
          apply UScalar.eq_of_val_eq
          by_contra hne
          rcases lt_or_gt_of_ne hne with hij | hji
          · exact (hb.2.2.2 i.val ha.2.1 hij ha.2.2.1).elim
          · exact (ha.2.2.2 j.val hb.2.1 hji hb.2.2.1).elim

/-- Search results agree when the two slot arrays have the same matching positions for every key. -/
theorem find_slots_eq_of_matches {N : Usize}
    (left right : Array (Option (I64 × I64)) N)
    (key : I64) (hmatch : ∀ i, MatchesAt key left.val i ↔ MatchesAt key right.val i) :
    table.Table.find_slots core.cmp.EqI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64 left key =
      table.Table.find_slots core.cmp.EqI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64 right key := by
  obtain ⟨a, ha, hpa⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec_generic left key 0#usize (Nat.zero_le _))
  obtain ⟨b, hb, hpb⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec_generic right key 0#usize (Nat.zero_le _))
  have hpb' : FindPostN (N := N) key left.val 0 b := by
    cases b with
    | none =>
        intro i hi hlt hleft
        exact hpb i hi hlt ((hmatch i).mp hleft)
    | some i =>
        exact ⟨hpb.1, hpb.2.1, (hmatch i.val).mpr hpb.2.2.1,
          fun j hj hlt hleft => hpb.2.2.2 j hj hlt ((hmatch j).mp hleft)⟩
  unfold table.Table.find_slots
  rw [ha, hb, findPostN_unique hpa hpb']

/-- Filling a vacant slot with `key` does not affect searches for a different key. -/
theorem find_slots_insertVacant_ne {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (first : Usize) (key value x : I64)
    (hslot : Array.index_usize slots first = ok none) (hne : x ≠ key) :
    table.Table.find_slots core.cmp.EqI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        (slots.set first (some (key, value))) x =
      table.Table.find_slots core.cmp.EqI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64 slots x := by
  apply find_slots_eq_of_matches _ _ x
  intro j
  have hvacant : slots.val[first.val]? = some none := by
    unfold Array.index_usize at hslot
    cases hs : slots[first]? with
    | none => simp [hs] at hslot
    | some o =>
        rw [hs] at hslot
        have ho : o = none := Aeneas.Std.Result.ok_injective hslot
        simp [ho] at hs ⊢
  have hfirst : first.val < slots.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hvacant
    simp at hvacant
  by_cases hj : j = first.val
  · subst j
    have hchanged : (slots.val.set first.val (some (key, value)))[first.val]? =
        some (some (key, value)) := by
      have hb : first.val < (slots.val.set first.val (some (key, value))).length := by
        simpa using hfirst
      rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
    simp only [MatchesAt, Array.set_val_eq]
    constructor <;> intro h
    · rcases h with ⟨v, hv⟩
      rw [hchanged] at hv
      exact (hne (Prod.mk.inj (Option.some.inj (Option.some.inj hv))).1.symm).elim
    · rcases h with ⟨v, hv⟩
      rw [hvacant] at hv
      simp at hv
  · simp [MatchesAt, Array.set_val_eq, hj]

/-- Word routing always produces an in-bounds shard. -/
theorem word_shard_index_spec {N : Usize} (t : WordTableN N) (key : I64) :
    ∃ shard, table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      t key = ok shard ∧ shard.val < 256 := by
  have hs : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      t key ⦃ shard => shard.val < 256 ⦄ := by
    unfold table.Table.shard_index
    simp only [table.WordHome.Insts.Apeiron_wordTableHomeI64.home]
    step
    have hi : i.val < 256 := by rw [i_post]; omega
    have hfit : i.val < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
    change (UScalar.cast .Usize i).val < 256
    rw [UScalar.cast_val_mod_pow_of_inBounds_eq .Usize i hfit]
    exact hi
  obtain ⟨shard, hshard, hbound⟩ := Aeneas.Std.WP.spec_imp_exists hs
  exact ⟨shard, hshard, hbound⟩

/-- Word shard routing depends on the key, not on table storage or default state. -/
theorem word_shard_index_state_independent {N : Usize}
    (t t' : WordTableN N) (key : I64) :
    table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t' key := by
  simp [table.Table.shard_index, table.WordHome.Insts.Apeiron_wordTableHomeI64.home]

/-- The state-indexed route produced by the generated shard and search operations.
    A present route includes the physical facts needed to identify the value returned by `get`.
    The route is indexed by `t` because `Table.find` searches `t.shards`; it is not a function of
    the key alone across different states. -/
def GeneratedRoute {N : Usize} (t : WordTableN N) (key : I64) :
    Option (Usize × Usize) → Prop
  | none => ∃ shard,
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
      table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        t shard key = ok none
  | some (shard, i) => ∃ slots value,
      table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard ∧
      table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        t shard key = ok (some i) ∧
      Array.index_usize t.shards shard = ok (some slots) ∧
      Array.index_usize slots i = ok (some (key, value))

/-- A locator is valid for one generated state when it records that state's generated routes. -/
def Routes {N : Usize} (t : WordTableN N)
    (locate : I64 → Option (Usize × Usize)) : Prop :=
  ∀ key, GeneratedRoute t key (locate key)

/-- Every generated word-table state has a state-indexed locator obtained from its executing
    `shard_index` and `find` operations. -/
theorem generatedRoute_exists {N : Usize} (t : WordTableN N) (key : I64) :
    ∃ location, GeneratedRoute t key location := by
  obtain ⟨shard, hshard, hbound⟩ := word_shard_index_spec t key
  have hlen : shard.val < t.shards.val.length := by
    rw [t.shards.property]
    exact hbound
  have houter : Array.index_usize t.shards shard = ok t.shards.val[shard.val] := by
    obtain ⟨outer, hindex, hvalue⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.index_usize_spec t.shards shard hlen)
    rw [hvalue] at hindex
    exact hindex
  cases hs : t.shards.val[shard.val] with
  | none =>
      refine ⟨none, shard, hshard, ?_⟩
      unfold table.Table.find
      rw [houter, hs]
      simp
  | some slots =>
      obtain ⟨found, hfindSlots, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
        (find_slots_spec_generic slots key 0#usize (Nat.zero_le _))
      have hfind : table.Table.find core.cmp.EqI64
          table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok found := by
        unfold table.Table.find
        rw [houter, hs]
        simpa [table.Table.find_slots] using hfindSlots
      cases found with
      | none => exact ⟨none, shard, hshard, hfind⟩
      | some i =>
          obtain ⟨hi, _, ⟨value, hmatch⟩, _⟩ := hpost
          have hslot : Array.index_usize slots i = ok (some (key, value)) := by
            have hil : i.val < slots.val.length := by rw [slots.property]; exact hi
            rw [List.getElem?_eq_getElem hil] at hmatch
            have hvalue : slots.val[i.val] = some (key, value) := Option.some.inj hmatch
            obtain ⟨slot, hindex, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
              (Array.index_usize_spec slots i hil)
            rw [hpost, hvalue] at hindex
            exact hindex
          exact ⟨some (shard, i), slots, value, hshard, hfind, by simpa [hs] using houter,
            hslot⟩

/-- The generated shard and `none` search result determine the absent route. -/
theorem generatedRoute_none_of_find {N : Usize} (t : WordTableN N) (key : I64)
    (shard : Usize)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none) :
    GeneratedRoute t key none := ⟨shard, hshard, hfind⟩

/-- The generated shard and successful search index determine the present route, including the
    physical slot facts supplied by total generated routing. -/
theorem generatedRoute_some_of_find {N : Usize} (t : WordTableN N) (key : I64)
    (shard i : Usize)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok (some i)) :
    GeneratedRoute t key (some (shard, i)) := by
  obtain ⟨location, hroute⟩ := generatedRoute_exists t key
  cases location with
  | none =>
      obtain ⟨s, hs, hf⟩ := hroute
      have hse : s = shard := Aeneas.Std.Result.ok_injective (hs.symm.trans hshard)
      subst s
      rw [hf] at hfind
      simp at hfind
  | some p =>
      rcases p with ⟨s, j⟩
      obtain ⟨routedSlots, routedValue, hs, hf, ho, hi⟩ := hroute
      have hse : s = shard := Aeneas.Std.Result.ok_injective (hs.symm.trans hshard)
      subst s
      have hje : j = i := Option.some.inj (Aeneas.Std.Result.ok_injective (hf.symm.trans hfind))
      subst j
      exact ⟨routedSlots, routedValue, hshard, hfind, ho, hi⟩

/-- The locator derived from generated routing for this particular state. -/
noncomputable def generatedLocator {N : Usize} (t : WordTableN N) (key : I64) :
    Option (Usize × Usize) := Classical.choose (generatedRoute_exists t key)

theorem generatedLocator_routes {N : Usize} (t : WordTableN N) :
    Routes t (generatedLocator t) := fun key => Classical.choose_spec (generatedRoute_exists t key)

/-- The real generic-`N` generated `Table.get` equals the total physical-state denotation at every
    route produced by its generated `shard_index` and `find` operations. -/
theorem table_get_eq_stateRead_of_route {N : Usize} (locate : I64 → Option (Usize × Usize))
    (t : WordTableN N) (key : I64) (hroute : GeneratedRoute t key (locate key)) :
    table.Table.get core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok (stateRead locate t key) := by
  cases hloc : locate key with
  | none =>
      rw [hloc] at hroute
      obtain ⟨shard, hshard, hfind⟩ := hroute
      unfold table.Table.get
      rw [hshard]
      simp only [bind_tc_ok]
      rw [hfind]
      simp [stateRead, hloc]
  | some location =>
      rcases location with ⟨shard, i⟩
      rw [hloc] at hroute
      obtain ⟨slots, value, hshard, hfind, houter, hslot⟩ := hroute
      have houter' : t.shards.val[shard.val]? = some (some slots) := by
        unfold Array.index_usize at houter
        split at houter <;> simp_all
      have hslot' : slots.val[i.val]? = some (some (key, value)) := by
        unfold Array.index_usize at hslot
        split at hslot <;> simp_all
      unfold table.Table.get
      rw [hshard]
      simp only [bind_tc_ok]
      rw [hfind]
      simp [houter, hslot, stateRead, hloc, houter', hslot']

/-- Total read characterization: for every generic-`N` generated word-table state and every key,
    the real generated `get` equals `stateRead` at that state's generated locator. -/
theorem table_get_eq_stateRead {N : Usize} (t : WordTableN N) (key : I64) :
    table.Table.get core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok (stateRead (generatedLocator t) t key) :=
  table_get_eq_stateRead_of_route (generatedLocator t) t key (generatedLocator_routes t key)

/-- The corrected generic representation is stated at the executing read boundary.  Unlike
    `FunctionalRep`, it does not freeze a state-dependent search result into a locator shared by
    unrelated states. -/
def ReadRep {N : Usize} (t : WordTableN N) (h : HMap I64 I64) : Prop :=
  ∀ key, table.Table.get core.cmp.EqI64
    table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok (h key)

/-- For a locator routed in the current state, `FunctionalRep` and the corrected executing
    `ReadRep` are equivalent. -/
theorem functionalRep_iff_readRep_of_routes {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N) (h : HMap I64 I64)
    (hroutes : Routes t locate) : FunctionalRep locate t h ↔ ReadRep t h := by
  constructor
  · intro hfunctional key
    rw [table_get_eq_stateRead_of_route locate t key (hroutes key), hfunctional]
  · intro hread
    unfold FunctionalRep
    funext key
    have hget := table_get_eq_stateRead_of_route locate t key (hroutes key)
    rw [hread key] at hget
    exact Aeneas.Std.Result.ok_injective hget.symm

/-- `FunctionalRep` is sound in its corrected state-indexed form, using the locator generated from
    that same state. -/
theorem generatedFunctionalRep_iff_readRep {N : Usize} (t : WordTableN N)
    (h : HMap I64 I64) : FunctionalRep (generatedLocator t) t h ↔ ReadRep t h :=
  functionalRep_iff_readRep_of_routes (generatedLocator t) t h (generatedLocator_routes t)

/-- At `N = 16`, the corrected generic relation is exactly the existing extensional `Rep`, in
    both directions. -/
theorem readRep_iff_Rep (t : WordTable) (h : HMap I64 I64) : ReadRep t h ↔ Rep t h := by
  constructor
  · intro hread key
    rw [word_table_get, hread key]
    exact Aeneas.Std.WP.spec.ret rfl
  · intro hrep key
    obtain ⟨value, hget, hvalue⟩ := Aeneas.Std.WP.spec_imp_exists (hrep key)
    rw [hvalue] at hget
    simpa only [word_table_get] using hget

/-! ## Generic-N absent-key write preservation -/

/-- Extend a locator with the physical position assigned to a fresh key. -/
def insertLocator (locate : I64 → Option (Usize × Usize)) (key : I64)
    (shard i : Usize) : I64 → Option (Usize × Usize) := fun x =>
  if x = key then some (shard, i) else locate x

/-- The state update shared by the empty-shard and vacant-slot generated branches. -/
def insertVacant {N : Usize} (t : WordTableN N) (shard : Usize)
    (slots : Array (Option (I64 × I64)) N) (i : Usize) (key value : I64) :
    WordTableN N :=
  { t with shards := t.shards.set shard (some (slots.set i (some (key, value)))) }

/-- Replacing a present slot's value leaves every bounded search result unchanged. -/
theorem find_slots_replacePresent {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (i : Usize) (key old value x : I64)
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    table.Table.find_slots core.cmp.EqI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        (slots.set i (some (key, value))) x =
      table.Table.find_slots core.cmp.EqI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64 slots x := by
  apply find_slots_eq_of_matches _ _ x
  intro j
  have hold : slots.val[i.val]? = some (some (key, old)) := by
    unfold Array.index_usize at hslot
    cases hs : slots[i]? with
    | none => simp [hs] at hslot
    | some o =>
        rw [hs] at hslot
        have ho : o = some (key, old) := Aeneas.Std.Result.ok_injective hslot
        simp [ho] at hs ⊢
  have hi : i.val < slots.val.length := by
    by_contra h
    rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hold
    simp at hold
  by_cases hj : j = i.val
  · subst j
    have hnew : (slots.val.set i.val (some (key, value)))[i.val]? =
        some (some (key, value)) := by
      have hb : i.val < (slots.val.set i.val (some (key, value))).length := by
        simpa using hi
      rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
    simp only [MatchesAt, Array.set_val_eq]
    rw [hnew, hold]
    simp
  · simp [MatchesAt, Array.set_val_eq, hj]

/-- Filling a vacant slot leaves `find` unchanged for every different key. -/
theorem table_find_insertVacant_ne {N : Usize}
    (t : WordTableN N) (shard first routeShard : Usize)
    (slots : Array (Option (I64 × I64)) N) (key value x : I64)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots first = ok none) (hne : x ≠ key) :
    table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        (insertVacant t shard slots first key value) routeShard x =
      table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        t routeShard x := by
  unfold table.Table.find
  by_cases hs : routeShard = shard
  · subst routeShard
    have hget : t.shards.val[shard.val]? = some (some slots) := by
      unfold Array.index_usize at houter
      split at houter <;> simp_all
    have hsbound : shard.val < t.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hget
      simp at hget
    have hnew : Array.index_usize
        (insertVacant t shard slots first key value).shards shard =
        ok (some (slots.set first (some (key, value)))) := by
      have hset : (t.shards.val.set shard.val
          (some (slots.set first (some (key, value)))))[shard.val]? =
          some (some (slots.set first (some (key, value)))) := by
        have hb : shard.val < (t.shards.val.set shard.val
            (some (slots.set first (some (key, value))))).length := by simpa using hsbound
        rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
      have hshards : (insertVacant t shard slots first key value).shards =
          t.shards.set shard (some (slots.set first (some (key, value)))) := rfl
      rw [hshards]
      unfold Array.index_usize
      rw [Array.getElem?_Usize_eq]
      rw [Array.set_val_eq]
      rw [hset]
    rw [hnew, houter]
    simp [find_slots_insertVacant_ne slots first key value x hslot hne]
  · have hsv : shard.val ≠ routeShard.val := fun h => hs (UScalar.eq_of_val_eq h.symm)
    simp [insertVacant, Array.index_usize, Array.set_val_eq, hsv]

/-- Replacing a present value leaves `find` unchanged for every key and route shard. -/
theorem table_find_replacePresent {N : Usize}
    (t : WordTableN N) (shard i routeShard : Usize)
    (slots : Array (Option (I64 × I64)) N) (key old value x : I64)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        (replacePresent t shard slots i key value) routeShard x =
      table.Table.find core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        t routeShard x := by
  unfold table.Table.find
  by_cases hs : routeShard = shard
  · subst routeShard
    have hget : t.shards.val[shard.val]? = some (some slots) := by
      unfold Array.index_usize at houter
      split at houter <;> simp_all
    have hsbound : shard.val < t.shards.val.length := by
      by_contra h
      rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hget
      simp at hget
    have hnew : Array.index_usize
        (replacePresent t shard slots i key value).shards shard =
        ok (some (slots.set i (some (key, value)))) := by
      have hset : (t.shards.val.set shard.val
          (some (slots.set i (some (key, value)))))[shard.val]? =
          some (some (slots.set i (some (key, value)))) := by
        have hb : shard.val < (t.shards.val.set shard.val
            (some (slots.set i (some (key, value))))).length := by simpa using hsbound
        rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
      have hshards : (replacePresent t shard slots i key value).shards =
          t.shards.set shard (some (slots.set i (some (key, value)))) := rfl
      rw [hshards]
      unfold Array.index_usize
      rw [Array.getElem?_Usize_eq]
      rw [Array.set_val_eq]
      rw [hset]
    rw [hnew, houter]
    simp [find_slots_replacePresent slots i key old value x hslot]
  · have hsv : shard.val ≠ routeShard.val := fun h => hs (UScalar.eq_of_val_eq h.symm)
    simp [replacePresent, Array.index_usize, Array.set_val_eq, hsv]

/-- After filling the first vacant slot, bounded search finds the fresh key at that slot. -/
theorem find_slots_insertVacant_self {N : Usize}
    (slots : Array (Option (I64 × I64)) N) (first : Usize) (key value : I64)
    (hfirst : first < N) (hnomatch : NoMatchIn key slots.val 0 first.val) :
    table.Table.find_slots core.cmp.EqI64 table.WordHome.Insts.Apeiron_wordTableHomeI64
        (slots.set first (some (key, value))) key = ok (some first) := by
  obtain ⟨found, hfound, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
    (find_slots_spec_generic (slots.set first (some (key, value))) key 0#usize
      (Nat.zero_le _))
  have hi : first.val < slots.val.length := by rw [slots.property]; exact hfirst
  have hnew : (slots.val.set first.val (some (key, value)))[first.val]? =
      some (some (key, value)) := by
    have hb : first.val < (slots.val.set first.val (some (key, value))).length := by
      simpa using hi
    rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
  have hbefore : NoMatchIn key
      (slots.val.set first.val (some (key, value))) 0 first.val := by
    intro j hj hlt
    have hne : j ≠ first.val := by omega
    simpa [MatchesAt, hne] using hnomatch j hj hlt
  have htarget : FindPostN (N := N) key
      (slots.set first (some (key, value))).val 0 (some first) := by
    refine ⟨hfirst, Nat.zero_le _, ?_, by simpa [Array.set_val_eq] using hbefore⟩
    exact ⟨value, by simpa [Array.set_val_eq] using hnew⟩
  change table.Table.find_slots_loop core.cmp.EqI64
    (slots.set first (some (key, value))) key 0#usize = ok (some first)
  rw [hfound, findPostN_unique hpost htarget]

/-- A present-key update preserves all generated routes under the same locator. -/
theorem routes_replacePresent {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N)
    (key value : I64) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) N) (old : I64)
    (hroutes : Routes t locate)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old))) :
    Routes (replacePresent t shard slots i key value) locate := by
  intro x
  have hroute := hroutes x
  cases hloc : locate x with
  | none =>
      rw [hloc] at hroute
      obtain ⟨s, hs, hf⟩ := hroute
      apply generatedRoute_none_of_find _ _ s
      · rw [← word_shard_index_state_independent t
          (replacePresent t shard slots i key value) x]
        exact hs
      · rw [table_find_replacePresent t shard i s slots key old value x houter hslot]
        exact hf
  | some p =>
      rcases p with ⟨s, j⟩
      rw [hloc] at hroute
      obtain ⟨_, _, hs, hf, _, _⟩ := hroute
      apply generatedRoute_some_of_find _ _ s j
      · rw [← word_shard_index_state_independent t
          (replacePresent t shard slots i key value) x]
        exact hs
      · rw [table_find_replacePresent t shard i s slots key old value x houter hslot]
        exact hf

/-- A fresh-key vacant write preserves every old route and adds the new key's generated route. -/
theorem routes_insertVacant {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N)
    (key value : I64) (shard first : Usize)
    (slots : Array (Option (I64 × I64)) N)
    (hroutes : Routes t locate)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      t key = ok shard)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < N) (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    Routes (insertVacant t shard slots first key value)
      (insertLocator locate key shard first) := by
  intro x
  by_cases hx : x = key
  · subst x
    simp only [insertLocator]
    apply generatedRoute_some_of_find _ _ shard first
    · rw [← word_shard_index_state_independent t
        (insertVacant t shard slots first key value) key]
      exact hshard
    · unfold table.Table.find
      have hnewOuter : Array.index_usize
          (insertVacant t shard slots first key value).shards shard =
          ok (some (slots.set first (some (key, value)))) := by
        have hget : t.shards.val[shard.val]? = some (some slots) := by
          unfold Array.index_usize at houter
          split at houter <;> simp_all
        have hsbound : shard.val < t.shards.val.length := by
          by_contra h
          rw [List.getElem?_eq_none (Nat.not_lt.mp h)] at hget
          simp at hget
        have hset : (t.shards.val.set shard.val
            (some (slots.set first (some (key, value)))))[shard.val]? =
            some (some (slots.set first (some (key, value)))) := by
          have hb : shard.val < (t.shards.val.set shard.val
              (some (slots.set first (some (key, value))))).length := by
            simpa using hsbound
          rw [List.getElem?_eq_getElem hb, List.getElem_set_self]
        have hshards : (insertVacant t shard slots first key value).shards =
            t.shards.set shard (some (slots.set first (some (key, value)))) := rfl
        rw [hshards]
        unfold Array.index_usize
        rw [Array.getElem?_Usize_eq, Array.set_val_eq, hset]
      rw [hnewOuter]
      simp only [bind_tc_ok]
      exact find_slots_insertVacant_self slots first key value hfirst hnomatch
  · simp only [insertLocator, if_neg hx]
    have hroute := hroutes x
    cases hloc : locate x with
    | none =>
        rw [hloc] at hroute
        obtain ⟨s, hs, hf⟩ := hroute
        apply generatedRoute_none_of_find _ _ s
        · rw [← word_shard_index_state_independent t
            (insertVacant t shard slots first key value) x]
          exact hs
        · rw [table_find_insertVacant_ne t shard first s slots key value x houter hslot hx]
          exact hf
    | some p =>
        rcases p with ⟨s, j⟩
        rw [hloc] at hroute
        obtain ⟨_, _, hs, hf, _, _⟩ := hroute
        apply generatedRoute_some_of_find _ _ s j
        · rw [← word_shard_index_state_independent t
            (insertVacant t shard slots first key value) x]
          exact hs
        · rw [table_find_insertVacant_ne t shard first s slots key value x houter hslot hx]
          exact hf

/-- Writing a vacant slot denotes the canonical fresh-key update. -/
theorem stateRead_insertVacant {N : Usize} (locate : I64 → Option (Usize × Usize))
    (t : WordTableN N) (key value : I64) (shard i : Usize)
    (slots : Array (Option (I64 × I64)) N)
    (hlocate : locate key = none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok none) :
    stateRead (insertLocator locate key shard i)
        (insertVacant t shard slots i key
          (core.num.I64.wrapping_add value t.default)) =
      insertWith core.num.I64.wrapping_add key value (stateRead locate t) := by
  have houter' : t.shards.val[shard.val]? = some (some slots) := by
    unfold Array.index_usize at houter
    split at houter <;> simp_all
  have hslot' : slots.val[i.val]? = some none := by
    unfold Array.index_usize at hslot
    split at hslot <;> simp_all
  have hs : shard.val < 256 := by have := t.shards.property; grind
  have hi : i.val < N.val := by have := slots.property; grind
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
    simp [stateRead, insertLocator, insertVacant, hlocate, hs, hi]
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

/-- Populating an untouched shard denotes the same canonical fresh-key update. -/
theorem stateRead_insertEmptyShard {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t : WordTableN N)
    (key value : I64) (shard : Usize) (hlocate : locate key = none)
    (houter : Array.index_usize t.shards shard = ok none) (hN : 0 < N.val) :
    stateRead (insertLocator locate key shard 0#usize)
        (insertVacant t shard (Array.repeat N none) 0#usize key
          (core.num.I64.wrapping_add value t.default)) =
      insertWith core.num.I64.wrapping_add key value (stateRead locate t) := by
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
    simp [stateRead, insertLocator, insertVacant, hlocate, hs, hN,
      Array.repeat_val, Array.set_val_eq]
  · rw [if_neg hx]
    simp only [stateRead, insertLocator, hx, ↓reduceIte]
    cases hloc : locate x with
    | none => simp [insertVacant]
    | some p =>
      rcases p with ⟨s, j⟩
      by_cases hss : s = shard
      · subst s
        by_cases hj : j = 0#usize
        · subst j
          simp [insertVacant, hs, hN, houterGet, Array.repeat_val, Array.set_val_eq,
            Ne.symm hx]
        · have hj0 : j.val ≠ 0 := fun h => hj (UScalar.eq_of_val_eq h)
          by_cases hjN : j.val < N.val
          · have hjl : j.val <
                (List.replicate N.val (none : Option (I64 × I64))).length := by
              simpa using hjN
            have hnone : (List.replicate N.val (none : Option (I64 × I64)))[j.val]? =
                some none := by
              rw [List.getElem?_eq_getElem hjl, List.getElem_replicate]
            simp [insertVacant, hs, Array.repeat_val, Array.set_val_eq, hj0, houterGet,
              hnone]
          · have hnone : (List.replicate N.val (none : Option (I64 × I64)))[j.val]? =
                none := List.getElem?_eq_none (by simpa using hjN)
            simp [insertVacant, hs, Array.repeat_val, Array.set_val_eq, hj0, houterGet,
              hnone]
      · have hne : shard.val ≠ s.val := fun h => hss (UScalar.eq_of_val_eq h.symm)
        simp [insertVacant, Array.set_val_eq, hne]

/-- Exact generated `insert_with` body for an absent key whose shard is untouched. -/
theorem insert_with_empty_shard_exact {N : Usize} (t : WordTableN N) (key value : I64)
    (shard : Usize)
    (hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok none)
    (hN : 0#usize < N) (hshardBound : shard < 256#usize) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Ok (), insertVacant t shard (Array.repeat N none) 0#usize key
        (core.num.I64.wrapping_add value t.default)) := by
  have hNne : N ≠ 0#usize := ne_of_gt hN
  have hrepeat : Array.index_usize (Array.repeat N (none : Option (I64 × I64))) 0#usize =
      ok none := by
    have hb : (0#usize).val < (Array.repeat N (none : Option (I64 × I64))).length := by
      simpa using hN
    obtain ⟨x, hx, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.index_usize_spec (Array.repeat N (none : Option (I64 × I64))) 0#usize hb)
    calc
      _ = ok x := hx
      _ = ok none := by rw [hp]; congr 1; simp [Array.repeat_val]
  have hmrepeat : Array.index_mut_usize (Array.repeat N (none : Option (I64 × I64))) 0#usize =
      ok (none, (Array.repeat N none).set 0#usize) := by
    simp [Array.index_mut_usize, hrepeat]
  let merged := core.num.I64.wrapping_add value t.default
  let newSlots := (Array.repeat N (none : Option (I64 × I64))).set 0#usize
    (some (key, merged))
  have hinner : Array.update ((Array.repeat N none).set 0#usize none) 0#usize
      (some (key, merged)) = ok newSlots := by
    have hb : (0#usize).val < ((Array.repeat N (none : Option (I64 × I64))).set
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
  simp only [bind_tc_ok]
  simp only [core.option.Option.is_none]
  rw [if_pos (by rfl)]
  simp only [core.clone.impls.CloneI64.clone,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge, bind_tc_ok, massert]
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
  simp only [bind_tc_ok]
  simp only [massert]
  rw [if_pos hshardBound]
  simp only [bind_tc_ok]
  rw [hmouter]
  simp only [bind_tc_ok]
  rw [houterUpdate]
  simp [newSlots, merged, insertVacant]

/-- One vacant iteration clones the default, merges new-first, and writes the generated slot. -/
theorem insert_with_loop0_vacant_body_exact {N : Usize}
    (back : Option (Array (Option (I64 × I64)) N) →
      Array (Option (Array (Option (I64 × I64)) N)) 256#usize)
    (slots : Array (Option (I64 × I64)) N) (default key value : I64) (i : Usize)
    (hi : i < N) (hslot : Array.index_usize slots i = ok none) :
    table.Table.insert_with_loop0.body core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 back default () key value slots i =
      ok (ControlFlow.done (core.result.Result.Ok (), back (some (slots.set i
        (some (key, core.num.I64.wrapping_add value default)))))) := by
  have hm : Array.index_mut_usize slots i = ok (none, slots.set i) := by
    simp [Array.index_mut_usize, hslot]
  have hu : Array.update (slots.set i none) i
      (some (key, core.num.I64.wrapping_add value default)) =
      ok (slots.set i (some (key, core.num.I64.wrapping_add value default))) := by
    have hb : i.val < (slots.set i none).length := by simpa using hi
    obtain ⟨a, ha, hp⟩ := Aeneas.Std.WP.spec_imp_exists
      (Array.update_spec (slots.set i none) i
        (some (key, core.num.I64.wrapping_add value default)) hb)
    rw [hp] at ha
    calc
      _ = ok ((slots.set i none).set i
          (some (key, core.num.I64.wrapping_add value default))) := ha
      _ = _ := by
        congr 1
        apply Aeneas.Std.Array.ext
        simp only [Array.set_val_eq, List.set_set]
  unfold table.Table.insert_with_loop0.body
  rw [if_pos hi, hslot]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_pos (by rfl)]
  simp only [core.clone.impls.CloneI64.clone,
    table.WordMerge.Insts.Apeiron_wordTableMergeI64.merge, bind_tc_ok]
  simp only [massert]
  rw [if_pos hi, hm]
  simp [hu]

def InsertInvN (key : I64) (slots : List (Option (I64 × I64)))
    (start : Nat) (first : Usize) (i : Usize) : Prop :=
  i.val ≤ first.val ∧ start ≤ i.val ∧ NoEmptyIn slots start i.val ∧
    NoMatchIn key slots start i.val

/-- The generated loop reaches the first vacant slot and performs the exact fresh-key write. -/
theorem insert_with_loop0_vacant_spec {N : Usize}
    (back : Option (Array (Option (I64 × I64)) N) →
      Array (Option (Array (Option (I64 × I64)) N)) 256#usize)
    (slots : Array (Option (I64 × I64)) N) (default key value : I64)
    (start first : Usize) (hstart : start.val ≤ first.val) (hfirst : first < N)
    (hempty : NoEmptyIn slots.val start.val first.val)
    (hnomatch : NoMatchIn key slots.val start.val first.val)
    (hslot : Array.index_usize slots first = ok none) :
    table.Table.insert_with_loop0 core.clone.CloneI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 back default () key value slots start
      ⦃ r => r = (core.result.Result.Ok (), back (some (slots.set first
        (some (key, core.num.I64.wrapping_add value default))))) ⦄ := by
  unfold table.Table.insert_with_loop0
  apply loop.spec_decr_nat
    (measure := fun i : Usize => first.val - i.val)
    (inv := fun i => InsertInvN key slots.val start.val first i)
    (post := fun r => r = (core.result.Result.Ok (), back (some (slots.set first
      (some (key, core.num.I64.wrapping_add value default))))))
  · intro i hi
    obtain ⟨hile, hstartle, hnoempty, hnomatch'⟩ := hi
    by_cases hifirst : i = first
    · subst i
      rw [insert_with_loop0_vacant_body_exact back slots default key value first hfirst hslot]
      simp
    · have hilt : i.val < first.val := by
        have hne : i.val ≠ first.val := fun h => hifirst (UScalar.eq_of_val_eq h)
        omega
      have hiN : i < N := by exact Nat.lt_trans hilt hfirst
      unfold table.Table.insert_with_loop0.body
      rw [if_pos hiN]
      step
      rcases o with _ | occupied
      · have hib : i.val < slots.val.length := by rw [slots.property]; exact hiN
        have ho : slots.val[i.val] = none := o_post.symm
        exact (hempty i.val hstartle hilt (emptyAt_of_none hib ho)).elim
      · simp
        step
        refine ⟨⟨?_, ?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [i1_post] using noEmptyIn_succ hnoempty (hempty i.val hstartle hilt)
        · simpa [i1_post] using noMatchIn_succ hnomatch' (hnomatch i.val hstartle hilt)
        · omega
  · exact ⟨hstart, le_rfl, fun j hj hjlt => by omega, fun j hj hjlt => by omega⟩

def InsertFullInvN {N : Usize} (slots : List (Option (I64 × I64)))
    (start : Nat) (i : Usize) : Prop :=
  i.val ≤ N.val ∧ start ≤ i.val ∧ NoEmptyIn slots start i.val

/-- With no vacant slot, the generic loop reaches `i = N` and writes back unchanged storage. -/
theorem insert_with_loop0_full_spec_generic {N : Usize}
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
    (post := fun r => r = (core.result.Result.Err table.StorageError.ShardFull, back (some slots)))
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

/-- Exact generated `insert_with` body when the first vacant slot is in an existing shard. -/
theorem insert_with_vacant_exact {N : Usize} (t : WordTableN N) (key value : I64)
    (shard first : Usize) (slots : Array (Option (I64 × I64)) N)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64 t key =
      ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < N) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Ok (), insertVacant t shard slots first key
        (core.num.I64.wrapping_add value t.default)) := by
  have hNne : N ≠ 0#usize := by intro h; subst N; simp at hfirst
  have hm : Array.index_mut_usize t.shards shard = ok (some slots, t.shards.set shard) := by
    simp [Array.index_mut_usize, houter]
  obtain ⟨r, hr, hp⟩ := Aeneas.Std.WP.spec_imp_exists
    (insert_with_loop0_vacant_spec (t.shards.set shard) slots t.default key value
      0#usize first (Nat.zero_le _) hfirst hempty hnomatch hslot)
  rw [hp] at hr
  unfold table.Table.insert_with
  rw [hshard]
  simp only [bind_tc_ok]
  rw [hfind]
  simp only [bind_tc_ok]
  rw [if_neg hNne, houter]
  simp only [bind_tc_ok, core.option.Option.is_none]
  rw [if_neg (by simp), hm]
  simp only [bind_tc_ok]
  change (do
      let (result, shards) ← table.Table.insert_with_loop0 core.clone.CloneI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64 (t.shards.set shard)
        t.default () key value slots 0#usize
      ok (result, { t with shards := shards })) = _
  rw [hr]
  simp [insertVacant]

/-- Vacant-slot success preserves the functional denotation under the fresh-key locator. -/
theorem insert_with_vacant_rep_preservation {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t t' : WordTableN N) (h : HMap I64 I64)
    (key value : I64) (shard first : Usize) (slots : Array (Option (I64 × I64)) N)
    (hrep : FunctionalRep locate t h) (hlocate : locate key = none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots first = ok none)
    (hresult : t' = insertVacant t shard slots first key
      (core.num.I64.wrapping_add value t.default)) :
    FunctionalRep (insertLocator locate key shard first) t'
      (insertWith core.num.I64.wrapping_add key value h) := by
  subst t'
  unfold FunctionalRep at hrep ⊢
  rw [stateRead_insertVacant locate t key value shard first slots hlocate houter hslot, hrep]

/-- The corrected executing-read representation follows whenever the post-state routes the
    fresh-key locator. -/
theorem insert_with_vacant_readRep_preservation {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t t' : WordTableN N) (h : HMap I64 I64)
    (key value : I64) (shard first : Usize) (slots : Array (Option (I64 × I64)) N)
    (hrep : FunctionalRep locate t h) (hlocate : locate key = none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots first = ok none)
    (hresult : t' = insertVacant t shard slots first key
      (core.num.I64.wrapping_add value t.default))
    (hroutes : Routes t' (insertLocator locate key shard first)) :
    ReadRep t' (insertWith core.num.I64.wrapping_add key value h) := by
  apply (functionalRep_iff_readRep_of_routes _ _ _ hroutes).mp
  exact insert_with_vacant_rep_preservation locate t t' h key value shard first slots
    hrep hlocate houter hslot hresult

/-- Present-key preservation at the executing read boundary; the post-state route follows from the
    pre-state route and the exact generated update. -/
theorem insert_with_present_readRep_preservation {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t t' : WordTableN N) (h : HMap I64 I64)
    (key value : I64) (shard i : Usize) (slots : Array (Option (I64 × I64)) N) (old : I64)
    (hrep : FunctionalRep locate t h) (hroutes : Routes t locate)
    (hlocate : locate key = some (shard, i))
    (hshard : table.Table.shard_index
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok (some i))
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hslot : Array.index_usize slots i = ok (some (key, old)))
    (hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Ok (), t')) :
    ReadRep t' (insertWith core.num.I64.wrapping_add key value h) := by
  have hexact := insert_with_present_exact t key value shard i slots key old
    hshard hfind houter hslot
  rw [hinsert] at hexact
  have ht' : t' = replacePresent t shard slots i key
      (core.num.I64.wrapping_add value old) := by
    exact (Prod.mk.inj (Aeneas.Std.Result.ok_injective hexact)).2
  apply (functionalRep_iff_readRep_of_routes locate t' _ ?_).mp
  · exact insert_with_present_rep_preservation locate t t' h key value shard i slots old
      hrep hlocate hshard hfind houter hslot hinsert
  · rw [ht']
    exact routes_replacePresent locate t key (core.num.I64.wrapping_add value old)
      shard i slots old hroutes houter hslot

/-- Vacant-slot preservation at the executing read boundary; no post-state `Routes` premise is
    required because routing is preserved by the exact fresh-key update. -/
theorem insert_with_vacant_readRep_preservation_routed {N : Usize}
    (locate : I64 → Option (Usize × Usize)) (t t' : WordTableN N) (h : HMap I64 I64)
    (key value : I64) (shard first : Usize) (slots : Array (Option (I64 × I64)) N)
    (hrep : FunctionalRep locate t h) (hroutes : Routes t locate)
    (hlocate : locate key = none)
    (hshard : table.Table.shard_index table.WordHome.Insts.Apeiron_wordTableHomeI64
      t key = ok shard)
    (hfind : table.Table.find core.cmp.EqI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64 t shard key = ok none)
    (houter : Array.index_usize t.shards shard = ok (some slots))
    (hfirst : first < N) (hempty : NoEmptyIn slots.val 0 first.val)
    (hnomatch : NoMatchIn key slots.val 0 first.val)
    (hslot : Array.index_usize slots first = ok none)
    (hinsert : table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
      table.WordHome.Insts.Apeiron_wordTableHomeI64
      table.WordMerge.Insts.Apeiron_wordTableMergeI64 t () key value =
      ok (core.result.Result.Ok (), t')) :
    ReadRep t' (insertWith core.num.I64.wrapping_add key value h) := by
  have hexact := insert_with_vacant_exact t key value shard first slots hshard hfind
    houter hfirst hempty hnomatch hslot
  rw [hinsert] at hexact
  have ht' : t' = insertVacant t shard slots first key
      (core.num.I64.wrapping_add value t.default) := by
    exact (Prod.mk.inj (Aeneas.Std.Result.ok_injective hexact)).2
  apply (functionalRep_iff_readRep_of_routes _ t' _ ?_).mp
  · exact insert_with_vacant_rep_preservation locate t _ h key value shard first slots
      hrep hlocate houter hslot ht'
  · rw [ht']
    exact routes_insertVacant locate t key (core.num.I64.wrapping_add value t.default)
      shard first slots hroutes hshard houter hfirst hnomatch hslot

/-- The concrete one-slot write jointly witnesses absent routing, generated success, and denotation. -/
theorem insert_with_absent_joint_witness (default value : I64) :
    table.Table.insert_with core.cmp.EqI64 core.clone.CloneI64
        table.WordHome.Insts.Apeiron_wordTableHomeI64
        table.WordMerge.Insts.Apeiron_wordTableMergeI64
        (emptyWordTable default) () 0#i64 value =
        ok (core.result.Result.Ok (), oneWordTable default value) ∧
      FunctionalRep (insertLocator (fun _ => none) 0#i64 0#usize 0#usize)
        (oneWordTable default value)
        (insertWith core.num.I64.wrapping_add 0#i64 value
          (stateRead (fun _ => none) (emptyWordTable default))) := by
  refine ⟨table_insert_empty_zero default value, ?_⟩
  unfold FunctionalRep
  have houter : Array.index_usize (emptyWordTable default).shards 0#usize = ok none := by
    set_option maxRecDepth 4096 in
      simp [emptyWordTable, Array.index_usize, Array.repeat_val]
  change stateRead (insertLocator (fun _ => none) 0#i64 0#usize 0#usize)
      (insertVacant (emptyWordTable default) 0#usize (Array.repeat 16#usize none)
        0#usize 0#i64 (core.num.I64.wrapping_add value default)) = _
  exact stateRead_insertEmptyShard (fun _ => none) (emptyWordTable default) 0#i64 value
    0#usize rfl houter (by decide)


end Apeiron.TableRefinement
