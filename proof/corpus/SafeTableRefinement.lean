import ApeironWord
import Apeiron.Table.Basic

/-!
# The extracted safe table search refines its first-match semantics

`SafeTable.find_from` is the loop-based generated search. `loop.spec_decr_nat` supplies the total
measure `4 - index`, and the invariant says every position in `[start, index)` is not a key match.
`find_from_loop_spec` states the total postcondition: a returned `some j` is the first match at or
after `start`, and `none` means no match in `[start, 4)`. The proof uses the generated body's array
read and `I64` equality, which `step` decomposes through `Array.index_usize_spec` and the local
`eqI64_spec`. The canonical `Apeiron.Table` denotation and the `get`/`try_insert`/`remove` links are
the next step; this file proves the search semantics they consume.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false
set_option linter.unusedSimpArgs false
set_option linter.unusedVariables false
set_option maxHeartbeats 1000000

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result

variable {V : Type}

/-- A position holds the key: the slot is `some (key, _)`. -/
def MatchesAt (key : I64) (l : List (Option (I64 × V))) (i : ℕ) : Prop :=
  ∃ v, l[i]? = some (some (key, v))

/-- No position in `[a, b)` holds the key. -/
def NoMatchIn (key : I64) (l : List (Option (I64 × V))) (a b : ℕ) : Prop :=
  ∀ j, a ≤ j → j < b → ¬ MatchesAt key l j

/-- The loop invariant: the index is in range and no earlier position matches. -/
def Inv (key : I64) (l : List (Option (I64 × V))) (start : ℕ) (index : Usize) : Prop :=
  index.val ≤ 4 ∧ start ≤ index.val ∧ NoMatchIn key l start index.val

/-- The search's postcondition: the first match at or after `start`, or none. -/
def Post (key : I64) (l : List (Option (I64 × V))) (start : ℕ) (r : Option Usize) : Prop :=
  match r with
  | some j => j.val < 4 ∧ start ≤ j.val ∧ MatchesAt key l j.val ∧ NoMatchIn key l start j.val
  | none => NoMatchIn key l start 4

/-- The `I64` equality the generated body calls returns `decide`. -/
@[step] theorem eqI64_spec (a b : I64) :
    core.cmp.impls.PartialEqShared.eq core.cmp.EqI64.partialEqInst a b
      ⦃ r => r = decide (a = b) ⦄ := by
  change Aeneas.Std.WP.spec (ok (decide (a = b))) (fun r => r = decide (a = b))
  exact Aeneas.Std.WP.spec.ret rfl

/-- Extending a no-match range by one non-matching position keeps it a no-match range. -/
theorem noMatchIn_succ {key : I64} {l : List (Option (I64 × V))} {a i : ℕ}
    (h : NoMatchIn key l a i) (hnot : ¬ MatchesAt key l i) :
    NoMatchIn key l a (i + 1) := by
  intro j ha hji
  rcases Nat.lt_succ_iff_lt_or_eq.mp hji with hlt | heq
  · exact h j ha hlt
  · subst heq; exact hnot

/-- A slot read as `some (key, w)` is a match. -/
theorem matchesAt_of_get {key k : I64} {w : V} {l : List (Option (I64 × V))} {i : ℕ}
    (hi : i < l.length) (h : l[i] = some (k, w)) (heq : k = key) : MatchesAt key l i := by
  refine ⟨w, ?_⟩
  rw [List.getElem?_eq_getElem hi, h, heq]

/-- A slot read as `none` is not a match. -/
theorem not_matchesAt_of_get {key : I64} {l : List (Option (I64 × V))} {i : ℕ}
    (hi : i < l.length) (h : l[i] = none) : ¬ MatchesAt key l i := by
  rintro ⟨v, hv⟩
  rw [List.getElem?_eq_getElem hi] at hv
  simp [h] at hv

/-- A slot read as `some (k, _)` with `k ≠ key` is not a match. -/
theorem not_matchesAt_of_get_ne {key k : I64} {w : V} {l : List (Option (I64 × V))} {i : ℕ}
    (hi : i < l.length) (h : l[i] = some (k, w)) (hne : k ≠ key) : ¬ MatchesAt key l i := by
  rintro ⟨v, hv⟩
  rw [List.getElem?_eq_getElem hi] at hv
  simp only [Option.some.injEq] at hv
  rw [h] at hv
  simp only [Option.some.injEq, Prod.mk.injEq] at hv
  exact hne hv.1

/-- The generated loop search is total and returns the first key match in `[start, 4)`.

The proof applies `loop.spec_decr_nat` with measure `4 - index` and invariant `Inv`, then discharges
the body through the generated `Array.index_usize` read and `I64` equality. -/
theorem find_from_loop_spec (self : SafeTable I64 V 4#usize) (key : I64) (start : Usize)
    (hstart : start.val ≤ 4) :
    SafeTable.find_from_loop core.cmp.EqI64 self key start
      ⦃ r => Post key self.slots.val start.val r ⦄ := by
  unfold SafeTable.find_from_loop
  apply loop.spec_decr_nat
    (measure := fun (index : Usize) => 4 - index.val)
    (inv := fun index => Inv key self.slots.val start.val index)
    (post := fun r => Post key self.slots.val start.val r)
  · intro index hinv
    obtain ⟨hlen, hstartle, hno⟩ := hinv
    unfold SafeTable.find_from_loop.body
    by_cases hlt : index < 4#usize
    · rw [if_pos hlt]
      have hltn : index.val < 4 := hlt
      have hi : index.val < (self.slots.val).length := by
        rw [Array.length_eq]; exact hlt
      step
      rcases o with _ | ⟨stored, val⟩
      · have ho : (self.slots.val)[index.val] = none := o_post.symm
        step
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [r_post] using noMatchIn_succ hno (not_matchesAt_of_get hi ho)
        · omega
      · have ho : (self.slots.val)[index.val] = some (stored, val) := o_post.symm
        step
        split
        · rename_i hr
          have hkey : stored = key := by
            rw [hr] at r_post
            exact of_decide_eq_true r_post.symm
          simp only [Aeneas.Std.WP.spec_ok]
          exact ⟨hltn, hstartle, matchesAt_of_get hi ho hkey, hno⟩
        · rename_i hr
          have hkey : stored ≠ key := by
            intro h
            exact hr (by simp [r_post, h])
          step
          refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
          · omega
          · omega
          · simpa [index1_post] using noMatchIn_succ hno (not_matchesAt_of_get_ne hi ho hkey)
          · omega
    · rw [if_neg hlt]
      rw [Aeneas.Std.WP.spec_ok]
      show NoMatchIn key self.slots.val start.val 4
      have h4 : index.val = 4 := by
        have hge : 4 ≤ index.val := not_lt.mp hlt
        omega
      simpa [h4] using hno
  · refine ⟨hstart, le_refl _, ?_⟩
    intro j hsj hj
    omega

/-- The generated `find_from` (which calls `find_from_loop` at `start`) has the same total
first-match postcondition. -/
theorem find_from_spec (self : SafeTable I64 V 4#usize) (key : I64) (start : Usize)
    (hstart : start.val ≤ 4) :
    SafeTable.find_from core.cmp.EqI64 self key start
      ⦃ r => Post key self.slots.val start.val r ⦄ :=
  find_from_loop_spec self key start hstart

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

/-- The first index in `l` whose slot is `some (key, _)`, if any. -/
def findIdx (key : I64) : List (Option (I64 × V)) → Option ℕ
  | [] => none
  | x :: xs =>
      match x with
      | some (k, _) => if k = key then some 0 else (findIdx key xs).map (· + 1)
      | none => (findIdx key xs).map (· + 1)

theorem matchesAt_none_succ {key : I64} {xs : List (Option (I64 × V))} {j : ℕ} :
    MatchesAt key (none :: xs) (j + 1) ↔ MatchesAt key xs j := by
  simp only [MatchesAt, List.getElem?_cons_succ]

theorem matchesAt_some_succ {key k : I64} {w : V} {xs : List (Option (I64 × V))} {j : ℕ} :
    MatchesAt key (some (k, w) :: xs) (j + 1) ↔ MatchesAt key xs j := by
  simp only [MatchesAt, List.getElem?_cons_succ]

theorem matchesAt_some_zero {key k : I64} {w : V} {xs : List (Option (I64 × V))} :
    MatchesAt key (some (k, w) :: xs) 0 ↔ k = key := by
  simp only [MatchesAt, List.getElem?_cons_zero]
  constructor
  · rintro ⟨v, hv⟩
    simp only [Option.some.injEq, Prod.mk.injEq] at hv
    exact hv.1
  · intro hk
    exact ⟨w, by simp [hk]⟩

theorem noMatchIn_succ_iff {key : I64} {l : List (Option (I64 × V))} {x : Option (I64 × V)}
    {j : ℕ} (hcons : ∀ i, MatchesAt key (x :: l) (i + 1) ↔ MatchesAt key l i)
    (h0 : ¬ MatchesAt key (x :: l) 0) :
    NoMatchIn key (x :: l) 0 (j + 1) ↔ NoMatchIn key l 0 j := by
  constructor
  · intro h i _ hij
    exact fun hm => h (i + 1) (Nat.zero_le _) (Nat.succ_lt_succ hij) ((hcons i).mpr hm)
  · intro h i _ hij
    cases i with
    | zero => exact h0
    | succ i' =>
      rw [(hcons i')]
      exact h i' (Nat.zero_le _) (by omega)

theorem findIdx_spec (key : I64) (l : List (Option (I64 × V))) (i : ℕ) :
    findIdx key l = some i ↔ MatchesAt key l i ∧ NoMatchIn key l 0 i := by
  induction l generalizing i with
  | nil => simp [findIdx, MatchesAt]
  | cons x xs ih =>
    rcases x with _ | ⟨k, w⟩
    · simp only [findIdx]
      rw [Option.map_eq_some_iff]
      constructor
      · rintro ⟨j, hj, hji⟩
        obtain ⟨hm, hn⟩ := (ih j).mp hj
        subst hji
        exact ⟨matchesAt_none_succ.mpr hm,
          (noMatchIn_succ_iff (x := none) (fun i => matchesAt_none_succ)
            (by simp only [MatchesAt, List.getElem?_cons_zero]; rintro ⟨v, hv⟩; simp at hv)).mpr hn⟩
      · rintro ⟨hm, hn⟩
        cases i with
        | zero =>
          simp only [MatchesAt, List.getElem?_cons_zero] at hm
          rcases hm with ⟨v, hv⟩; simp at hv
        | succ j =>
          rw [matchesAt_none_succ] at hm
          rw [(noMatchIn_succ_iff (x := none) (fun i => matchesAt_none_succ)
            (by simp only [MatchesAt, List.getElem?_cons_zero]; rintro ⟨v, hv⟩; simp at hv))] at hn
          exact ⟨j, (ih j).mpr ⟨hm, hn⟩, rfl⟩
    · simp only [findIdx]
      by_cases hk : k = key
      · rw [if_pos hk]
        constructor
        · intro h
          rw [Option.some.injEq] at h
          subst h
          exact ⟨matchesAt_some_zero.mpr hk, fun j _ hj => by omega⟩
        · rintro ⟨hm, hn⟩
          cases i with
          | zero => rfl
          | succ j =>
            exact absurd (matchesAt_some_zero.mpr hk)
              (hn 0 (Nat.zero_le _) (Nat.succ_pos j))
      · rw [if_neg hk]
        rw [Option.map_eq_some_iff]
        constructor
        · rintro ⟨j, hj, hji⟩
          obtain ⟨hm, hn⟩ := (ih j).mp hj
          subst hji
          exact ⟨matchesAt_some_succ.mpr hm,
            (noMatchIn_succ_iff (x := some (k, w)) (fun i => matchesAt_some_succ)
              (fun hm => hk (matchesAt_some_zero.mp hm))).mpr hn⟩
        · rintro ⟨hm, hn⟩
          cases i with
          | zero => exact absurd (matchesAt_some_zero.mp hm) hk
          | succ j =>
            rw [matchesAt_some_succ] at hm
            rw [(noMatchIn_succ_iff (x := some (k, w)) (fun i => matchesAt_some_succ)
              (fun hm => hk (matchesAt_some_zero.mp hm)))] at hn
            exact ⟨j, (ih j).mpr ⟨hm, hn⟩, rfl⟩

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

theorem findIdx_none_iff (key : I64) (l : List (Option (I64 × V))) :
    findIdx key l = none ↔ NoMatchIn key l 0 l.length := by
  induction l with
  | nil => simp [findIdx, NoMatchIn]
  | cons x xs ih =>
    rcases x with _ | ⟨k, w⟩
    · simp only [findIdx]
      rw [Option.map_eq_none_iff, ih, List.length_cons]
      exact (noMatchIn_succ_iff (x := none) (fun i => matchesAt_none_succ)
        (by simp only [MatchesAt, List.getElem?_cons_zero]; rintro ⟨v, hv⟩; simp at hv)).symm
    · by_cases hk : k = key
      · simp only [findIdx]
        rw [if_pos hk]
        constructor
        · intro h; exact absurd h (by simp)
        · intro h
          exact absurd (matchesAt_some_zero.mpr hk)
            (h 0 (Nat.zero_le _) (Nat.succ_pos xs.length))
      · simp only [findIdx, hk, if_false]
        rw [Option.map_eq_none_iff, ih, List.length_cons]
        exact (noMatchIn_succ_iff (x := some (k, w)) (fun i => matchesAt_some_succ)
          (fun hm => hk (matchesAt_some_zero.mp hm))).symm

/-- The table the safe slots denote: the value stored at the first slot holding `k`. Absence is
`none`, matching the corpus `Apeiron.Table` default. -/
def tableHMap (t : SafeTable I64 V 4#usize) : Apeiron.Table.HMap I64 (Option V) :=
  fun k => (findIdx k t.slots.val).bind
    (fun i => (t.slots.val[i]?).bind (fun s => s.map Prod.snd))

@[step] theorem find_from_zero_spec (self : SafeTable I64 V 4#usize) (key : I64) :
    SafeTable.find_from core.cmp.EqI64 self key 0#usize
      ⦃ r => Post key self.slots.val 0 r ⦄ :=
  find_from_spec self key 0#usize (by decide)

theorem get_spec (self : SafeTable I64 V 4#usize) (key : I64) :
    SafeTable.get core.cmp.EqI64 self key ⦃ r => r = tableHMap self key ⦄ := by
  unfold SafeTable.get
  step
  rcases o with _ | index
  · rw [Aeneas.Std.WP.spec_ok]
    show none = tableHMap self key
    simp only [tableHMap]
    rw [(findIdx_none_iff key self.slots.val).mpr ?_]
    · rfl
    · have h4 : (4#usize).val = 4 := by decide
      have hlen : self.slots.val.length = 4 := by rw [Array.length_eq, h4]
      have ho : NoMatchIn key self.slots.val 0 4 := o_post
      rwa [hlen]
  · step
    obtain ⟨hlt4, _, hmatch, hno⟩ := o_post
    have hi : index.val < (self.slots.val).length := by
      have h4 : (4#usize).val = 4 := by decide
      have hp : self.slots.val.length = 4 := by rw [Array.length_eq, h4]
      omega
    rcases r with _ | ⟨stored, value⟩
    · rw [Aeneas.Std.WP.spec_ok]
      exact absurd hmatch (not_matchesAt_of_get hi r_post.symm)
    · refine Aeneas.Std.WP.spec.ret ?_
      have hfind : findIdx key self.slots.val = some index.val :=
        (findIdx_spec key self.slots.val index.val).mpr ⟨hmatch, hno⟩
      have hlook : (self.slots.val)[index.val]? = some (some (stored, value)) := by
        rw [List.getElem?_eq_getElem hi]; exact congrArg some r_post.symm
      simp [tableHMap, hfind, hlook]

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

/-- The lookup link to the corpus: the extracted `get` returns exactly the value the canonical
`Apeiron.Table.get` reads from `tableHMap`. -/
theorem get_spec_canonical (self : SafeTable I64 V 4#usize) (key : I64) :
    SafeTable.get core.cmp.EqI64 self key
      ⦃ r => r = Apeiron.Table.get (tableHMap self) key ⦄ := by
  simpa only [Apeiron.Table.get] using get_spec self key

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

/-- A slot is empty: it holds `none`. -/
def EmptyAt (l : List (Option (I64 × V))) (i : ℕ) : Prop := l[i]? = some none

/-- No slot in `[a, b)` is empty. -/
def NoEmptyIn (l : List (Option (I64 × V))) (a b : ℕ) : Prop :=
  ∀ j, a ≤ j → j < b → ¬ EmptyAt l j

/-- The empty-slot loop invariant. -/
def InvE (l : List (Option (I64 × V))) (start : ℕ) (index : Usize) : Prop :=
  index.val ≤ 4 ∧ start ≤ index.val ∧ NoEmptyIn l start index.val

/-- The empty-slot search postcondition. -/
def PostE (l : List (Option (I64 × V))) (start : ℕ) (r : Option Usize) : Prop :=
  match r with
  | some j => j.val < 4 ∧ start ≤ j.val ∧ EmptyAt l j.val ∧ NoEmptyIn l start j.val
  | none => NoEmptyIn l start 4

theorem noEmptyIn_succ {l : List (Option (I64 × V))} {a i : ℕ}
    (h : NoEmptyIn l a i) (hnot : ¬ EmptyAt l i) : NoEmptyIn l a (i + 1) := by
  intro j ha hji
  rcases Nat.lt_succ_iff_lt_or_eq.mp hji with hlt | heq
  · exact h j ha hlt
  · subst heq; exact hnot

theorem emptyAt_of_get {l : List (Option (I64 × V))} {i : ℕ}
    (hi : i < l.length) (h : l[i] = none) : EmptyAt l i := by
  rw [EmptyAt, List.getElem?_eq_getElem hi, h]

theorem not_emptyAt_of_get {l : List (Option (I64 × V))} {i : ℕ}
    (hi : i < l.length) {k : I64} {w : V} (h : l[i] = some (k, w)) : ¬ EmptyAt l i := by
  intro hv
  rw [EmptyAt, List.getElem?_eq_getElem hi, h] at hv
  simp at hv

theorem empty_from_loop_spec (self : SafeTable I64 V 4#usize) (start : Usize)
    (hstart : start.val ≤ 4) :
    SafeTable.empty_from_loop self start
      ⦃ r => PostE self.slots.val start.val r ⦄ := by
  unfold SafeTable.empty_from_loop
  apply loop.spec_decr_nat
    (measure := fun (index : Usize) => 4 - index.val)
    (inv := fun index => InvE self.slots.val start.val index)
    (post := fun r => PostE self.slots.val start.val r)
  · intro index hinv
    obtain ⟨hlen, hstartle, hno⟩ := hinv
    unfold SafeTable.empty_from_loop.body
    by_cases hlt : index < 4#usize
    · rw [if_pos hlt]
      have hltn : index.val < 4 := hlt
      have hi : index.val < (self.slots.val).length := by rw [Array.length_eq]; exact hlt
      step
      rcases o with _ | ⟨stored, val⟩
      · have ho : (self.slots.val)[index.val] = none := o_post.symm
        rw [Aeneas.Std.WP.spec_ok]
        exact ⟨hltn, hstartle, emptyAt_of_get hi ho, hno⟩
      · have ho : (self.slots.val)[index.val] = some (stored, val) := o_post.symm
        step
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · omega
        · omega
        · simpa [r_post] using noEmptyIn_succ hno (not_emptyAt_of_get hi ho)
        · omega
    · rw [if_neg hlt]
      rw [Aeneas.Std.WP.spec_ok]
      show NoEmptyIn self.slots.val start.val 4
      have hge : 4 ≤ index.val := not_lt.mp hlt
      have h4idx : index.val = 4 := by omega
      simpa [h4idx] using hno
  · refine ⟨hstart, le_refl _, ?_⟩
    intro j hsj hj
    omega

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

/-- The generated `empty_from` has the same total first-empty-slot postcondition. -/
theorem empty_from_spec (self : SafeTable I64 V 4#usize) (start : Usize)
    (hstart : start.val ≤ 4) :
    SafeTable.empty_from core.cmp.EqI64 self start
      ⦃ r => PostE self.slots.val start.val r ⦄ :=
  empty_from_loop_spec self start hstart

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

theorem matchesAt_set_self {l : List (Option (I64 × V))} {i : ℕ} {k : I64} {w : V}
    (hi : i < l.length) : MatchesAt k (l.set i (some (k, w))) i :=
  ⟨w, by rw [List.getElem?_set_self hi]⟩

theorem not_matchesAt_set_self {l : List (Option (I64 × V))} {i : ℕ} {key k : I64} {w : V}
    (hi : i < l.length) (hne : key ≠ k) : ¬ MatchesAt k (l.set i (some (key, w))) i := by
  rintro ⟨v, hv⟩
  rw [List.getElem?_set_self hi] at hv
  simp only [Option.some.injEq, Prod.mk.injEq] at hv
  exact hne hv.1

theorem matchesAt_set_ne {l : List (Option (I64 × V))} {i j : ℕ} {k : I64} {e : I64 × V}
    (hij : i ≠ j) : MatchesAt k (l.set i (some e)) j ↔ MatchesAt k l j := by
  simp only [MatchesAt, List.getElem?_set_ne hij]

theorem findIdx_set_some_self {l : List (Option (I64 × V))} {i : ℕ} {k : I64} {w : V}
    (hi : i < l.length) (hbefore : NoMatchIn k l 0 i) :
    findIdx k (l.set i (some (k, w))) = some i := by
  refine (findIdx_spec k (l.set i (some (k, w))) i).mpr ⟨matchesAt_set_self hi, ?_⟩
  intro m _ hm
  rw [matchesAt_set_ne (by omega : i ≠ m)]
  exact hbefore m (Nat.zero_le _) hm

theorem findIdx_set_some_ne {l : List (Option (I64 × V))} {i : ℕ} {key k : I64} {w : V}
    (hi : i < l.length) (hne : key ≠ k) (hnot : ¬ MatchesAt k l i) :
    findIdx k (l.set i (some (key, w))) = findIdx k l := by
  cases h : findIdx k l with
  | none =>
    rw [findIdx_none_iff] at h
    rw [findIdx_none_iff]
    intro m _ hm
    rw [List.length_set] at hm
    by_cases hmi : m = i
    · subst hmi; exact not_matchesAt_set_self hi hne
    · rw [matchesAt_set_ne (Ne.symm hmi)]
      exact h m (Nat.zero_le _) hm
  | some j =>
    rw [findIdx_spec] at h
    refine (findIdx_spec k (l.set i (some (key, w))) j).mpr ⟨?_, ?_⟩
    · by_cases hji : j = i
      · subst hji; exact absurd h.1 hnot
      · rw [matchesAt_set_ne (Ne.symm hji)]; exact h.1
    · intro m _ hm
      by_cases hmi : m = i
      · subst hmi; exact not_matchesAt_set_self hi hne
      · rw [matchesAt_set_ne (Ne.symm hmi)]
        exact h.2 m (Nat.zero_le _) hm

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

theorem tableHMap_set_self {t : SafeTable I64 V 4#usize} {i : Usize} {k : I64} {w : V}
    (hi : i.val < t.slots.val.length) (hbefore : NoMatchIn k t.slots.val 0 i.val) :
    tableHMap { t with slots := t.slots.set i (some (k, w)) } k = some w := by
  simp only [tableHMap, Array.set_val_eq]
  rw [findIdx_set_some_self hi hbefore]
  simp [List.getElem?_set_self hi]

/-- The slot-list readout `tableHMap` computes. -/
def lookupList (k : I64) (l : List (Option (I64 × V))) : Option V :=
  (findIdx k l).bind (fun i => (l[i]?).bind (fun s => s.map Prod.snd))

theorem tableHMap_eq_lookupList (t : SafeTable I64 V 4#usize) (k : I64) :
    tableHMap t k = lookupList k t.slots.val := rfl

theorem tableHMap_set_ne {t : SafeTable I64 V 4#usize} {i : Usize} {key k : I64} {w : V}
    (hi : i.val < t.slots.val.length) (hne : key ≠ k) (hnot : ¬ MatchesAt k t.slots.val i.val) :
    tableHMap { t with slots := t.slots.set i (some (key, w)) } k = tableHMap t k := by
  simp only [tableHMap_eq_lookupList, Array.set_val_eq]
  cases h : findIdx k t.slots.val with
  | none =>
    rw [lookupList, lookupList, findIdx_set_some_ne hi hne hnot, h]
    simp
  | some j =>
    have hj : MatchesAt k t.slots.val j := ((findIdx_spec k t.slots.val j).mp h).1
    have hji : j ≠ i.val := by
      intro hji; subst hji; exact hnot hj
    rw [lookupList, lookupList, findIdx_set_some_ne hi hne hnot, h]
    simp [List.getElem?_set_ne (Ne.symm hji)]

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

/-- The slot element `e` holds key `x`. -/
def IsElemMatch (x : I64) (e : Option (I64 × V)) : Prop := ∃ w, e = some (x, w)

/-- At most one position holds any given key. -/
def KeysUnique (l : List (Option (I64 × V))) : Prop :=
  ∀ k i j, MatchesAt k l i → MatchesAt k l j → i = j

/-- The stored count matches the number of occupied slots. -/
def WellFormed (t : SafeTable I64 V 4#usize) : Prop :=
  t.len.val = List.countP (fun e : Option (I64 × V) => e.isSome) t.slots.val ∧
    KeysUnique t.slots.val

theorem isElemMatch_none {x : I64} : ¬ IsElemMatch x (none : Option (I64 × V)) := by
  rintro ⟨w, hw⟩; simp at hw

theorem matchesAt_set_self_iff {l : List (Option (I64 × V))} {i : ℕ} {k : I64}
    {e : Option (I64 × V)} (hi : i < l.length) :
    MatchesAt k (l.set i e) i ↔ IsElemMatch k e := by
  rw [MatchesAt, List.getElem?_set_self hi]
  constructor
  · rintro ⟨w, hw⟩; exact ⟨w, Option.some.inj hw⟩
  · rintro ⟨w, rfl⟩; exact ⟨w, rfl⟩

theorem matchesAt_set_ne_iff {l : List (Option (I64 × V))} {i j : ℕ} {k : I64}
    {e : Option (I64 × V)} (hij : i ≠ j) :
    MatchesAt k (l.set i e) j ↔ MatchesAt k l j := by
  rw [MatchesAt, MatchesAt, List.getElem?_set_ne hij]

theorem findIdx_set_self {l : List (Option (I64 × V))} {i : ℕ} {k : I64}
    {e : Option (I64 × V)} (hi : i < l.length) (hm : IsElemMatch k e)
    (hbefore : NoMatchIn k l 0 i) :
    findIdx k (l.set i e) = some i := by
  refine (findIdx_spec k (l.set i e) i).mpr ⟨(matchesAt_set_self_iff hi).mpr hm, ?_⟩
  intro m _ hm'
  rw [matchesAt_set_ne_iff (by omega : i ≠ m)]
  exact hbefore m (Nat.zero_le _) hm'

theorem findIdx_set_ne {l : List (Option (I64 × V))} {i : ℕ} {k : I64}
    {e : Option (I64 × V)} (hi : i < l.length)
    (hold : ¬ MatchesAt k l i) (hnew : ¬ IsElemMatch k e) :
    findIdx k (l.set i e) = findIdx k l := by
  cases h : findIdx k l with
  | none =>
    rw [findIdx_none_iff] at h ⊢
    intro m _ hm
    rw [List.length_set] at hm
    by_cases hmi : m = i
    · subst hmi
      rw [matchesAt_set_self_iff hi]
      exact hnew
    · rw [matchesAt_set_ne_iff (Ne.symm hmi)]
      exact h m (Nat.zero_le _) hm
  | some j =>
    rw [findIdx_spec] at h
    refine (findIdx_spec k (l.set i e) j).mpr ⟨?_, ?_⟩
    · by_cases hji : j = i
      · subst hji; exact absurd h.1 hold
      · rw [matchesAt_set_ne_iff (Ne.symm hji)]; exact h.1
    · intro m _ hm
      by_cases hmi : m = i
      · subst hmi
        rw [matchesAt_set_self_iff hi]
        exact hnew
      · rw [matchesAt_set_ne_iff (Ne.symm hmi)]
        exact h.2 m (Nat.zero_le _) hm

theorem tableHMap_set_none {t : SafeTable I64 V 4#usize} {i : Usize} {k : I64}
    (hi : i.val < t.slots.val.length) (hnot : ¬ MatchesAt k t.slots.val i.val) :
    tableHMap { t with slots := t.slots.set i none } k = tableHMap t k := by
  simp only [tableHMap_eq_lookupList, Array.set_val_eq]
  cases h : findIdx k t.slots.val with
  | none =>
    rw [lookupList, lookupList,
      findIdx_set_ne hi hnot (isElemMatch_none), h]
    simp
  | some j =>
    have hj : MatchesAt k t.slots.val j := ((findIdx_spec k t.slots.val j).mp h).1
    have hji : j ≠ i.val := by
      intro hji; subst hji; exact hnot hj
    rw [lookupList, lookupList,
      findIdx_set_ne hi hnot (isElemMatch_none), h]
    simp [List.getElem?_set_ne (Ne.symm hji)]

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

theorem countP_isSome_le (l : List (Option (I64 × V))) :
    List.countP (fun e : Option (I64 × V) => e.isSome) l ≤ l.length :=
  List.countP_le_length

theorem countP_set_none {l : List (Option (I64 × V))} {i : ℕ} {x : I64 × V}
    (hi : i < l.length) (h : l[i] = some x) :
    List.countP (fun e : Option (I64 × V) => e.isSome) (l.set i none) =
      List.countP (fun e : Option (I64 × V) => e.isSome) l - 1 := by
  rw [List.countP_set hi, h]
  simp

theorem countP_set_some {l : List (Option (I64 × V))} {i : ℕ} {y : I64 × V}
    (hi : i < l.length) (h : l[i] = none) :
    List.countP (fun e : Option (I64 × V) => e.isSome) (l.set i (some y)) =
      List.countP (fun e : Option (I64 × V) => e.isSome) l + 1 := by
  rw [List.countP_set hi, h]
  simp

theorem wellFormed_len_le_four {t : SafeTable I64 V 4#usize} (hwf : WellFormed t) :
    t.len.val ≤ 4 := by
  obtain ⟨hlen, _⟩ := hwf
  rw [hlen]
  calc List.countP (fun e : Option (I64 × V) => e.isSome) t.slots.val
      ≤ t.slots.val.length := List.countP_le_length
    _ = 4 := by rw [Array.length_eq]; rfl

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

theorem lookupList_none {k : I64} {l : List (Option (I64 × V))} (h : findIdx k l = none) :
    lookupList k l = none := by
  simp [lookupList, h]

theorem tableHMap_none_of_noMatchIn (self : SafeTable I64 V 4#usize) (key : I64)
    (h : NoMatchIn key self.slots.val 0 4) : tableHMap self key = none := by
  rw [tableHMap_eq_lookupList]
  apply lookupList_none
  apply (findIdx_none_iff key self.slots.val).mpr
  rwa [Array.length_eq]

theorem tableHMap_some_of_get (self : SafeTable I64 V 4#usize) {key : I64} {i : ℕ} {w : V}
    (hmatches : MatchesAt key self.slots.val i) (hnomatch : NoMatchIn key self.slots.val 0 i)
    (hget : self.slots.val[i]? = some (some (key, w))) : tableHMap self key = some w := by
  rw [tableHMap_eq_lookupList, lookupList,
    (findIdx_spec key self.slots.val i).mpr ⟨hmatches, hnomatch⟩]
  simp [hget]

theorem not_matchesAt_of_emptyAt {k : I64} {l : List (Option (I64 × V))} {i : ℕ}
    (h : EmptyAt l i) : ¬ MatchesAt k l i := by
  intro hm; rcases hm with ⟨v, hv⟩
  rw [EmptyAt] at h; rw [hv] at h; simp at h

theorem tableHMap_set_eq_insertWith (self : SafeTable I64 V 4#usize) {i : Usize} {key : I64}
    {value : V} (hi : i.val < self.slots.val.length)
    (hbefore : NoMatchIn key self.slots.val 0 i.val)
    (hnot : ∀ x, x ≠ key → ¬ MatchesAt x self.slots.val i.val) :
    tableHMap { self with slots := self.slots.set i (some (key, value)) } =
      Apeiron.Table.insertWith (fun new _ => new) key (some value) (tableHMap self) := by
  funext x
  by_cases hx : x = key
  · subst hx
    rw [tableHMap_set_self hi hbefore]
    simp [Apeiron.Table.insertWith]
  · rw [tableHMap_set_ne hi (Ne.symm hx) (hnot x hx)]
    simp [Apeiron.Table.insertWith, hx]

@[step] theorem empty_from_zero_spec (self : SafeTable I64 V 4#usize) :
    SafeTable.empty_from core.cmp.EqI64 self 0#usize ⦃ r => PostE self.slots.val 0 r ⦄ :=
  empty_from_spec self 0#usize (by decide)

theorem noMatchIn_of_absent {key : I64} {l : List (Option (I64 × V))} (hno : NoMatchIn key l 0 4)
    (hlen : l.length = 4) (j : ℕ) : ¬ MatchesAt key l j := by
  by_cases hj : j < 4
  · exact hno j (Nat.zero_le _) hj
  · rintro ⟨v, hv⟩
    rw [List.getElem?_eq_none (by omega)] at hv
    simp at hv

theorem not_matchesAt_set_none {l : List (Option (I64 × V))} {i : ℕ} {k : I64} :
    ¬ MatchesAt k (l.set i none) i := by
  rintro ⟨v, hv⟩
  rw [List.getElem?_set] at hv
  split at hv <;> simp_all

theorem keysUnique_set_none {l : List (Option (I64 × V))} {i : ℕ} (huniq : KeysUnique l) :
    KeysUnique (l.set i none) := by
  intro k p q hp hq
  by_cases hpi : p = i
  · rw [hpi] at hp
    exact absurd hp not_matchesAt_set_none
  · rw [matchesAt_set_ne_iff (Ne.symm hpi)] at hp
    by_cases hqi : q = i
    · rw [hqi] at hq
      exact absurd hq not_matchesAt_set_none
    · rw [matchesAt_set_ne_iff (Ne.symm hqi)] at hq
      exact huniq k p q hp hq

theorem keysUnique_set_some {l : List (Option (I64 × V))} {i : ℕ} {key : I64} {value : V}
    (huniq : KeysUnique l) (hi : i < l.length)
    (hkey : ∀ j, j ≠ i → ¬ MatchesAt key l j) :
    KeysUnique (l.set i (some (key, value))) := by
  intro k p q hp hq
  by_cases hpi : p = i
  · rw [hpi] at hp
    by_cases hqi : q = i
    · rw [hpi, hqi]
    · rw [matchesAt_set_ne_iff (Ne.symm hqi)] at hq
      have hk : k = key := by
        rcases (matchesAt_set_self_iff hi).mp hp with ⟨w, hw⟩
        exact (Prod.mk.inj (Option.some.inj hw)).1.symm
      rw [hk] at hq
      exact absurd hq (hkey q hqi)
  · rw [matchesAt_set_ne_iff (Ne.symm hpi)] at hp
    by_cases hqi : q = i
    · rw [hqi] at hq
      have hk : k = key := by
        rcases (matchesAt_set_self_iff hi).mp hq with ⟨w, hw⟩
        exact (Prod.mk.inj (Option.some.inj hw)).1.symm
      rw [hk] at hp
      exact absurd hp (hkey p hpi)
    · rw [matchesAt_set_ne_iff (Ne.symm hqi)] at hq
      exact huniq k p q hp hq


theorem one_le_countP_of_get {l : List (Option (I64 × V))} {i : ℕ} {x : I64 × V}
    (hi : i < l.length) (h : l[i] = some x) :
    1 ≤ List.countP (fun e : Option (I64 × V) => e.isSome) l := by
  have h1 := List.countP_set (p := fun e : Option (I64 × V) => e.isSome) (l := l) (i := i)
    (a := some x) hi
  have hset : l.set i (some x) = l := by rw [← h]; exact List.set_getElem_self hi
  rw [h] at h1
  simp only [Option.isSome_some, if_true, Nat.sub_zero] at h1
  rw [hset] at h1
  omega

/-- The full postcondition of the extracted `try_insert`.

On success it records the displaced value, the canonical `insertWith` update and the length step: a
fresh insert (`old = none`) increments `len`, a replacement leaves it. On refusal it records the
exact `TableFull { key, value }` payload, that the key was absent, that no slot is empty, and that
the state is unchanged. The refusal branch cannot be met by an always-refuse body: such a body
would have to return the payload for a present key and for a table with a free slot, contradicting
`tableHMap self key = none` and `NoEmptyIn self.slots.val 0 4`. -/
def TryInsertPost (self : SafeTable I64 V 4#usize) (key : I64) (value : V) :
    (core.result.Result (Option V) (safe_table.TableFull I64 V) ×
      SafeTable I64 V 4#usize) → Prop
  | (core.result.Result.Ok old, t') =>
      old = tableHMap self key ∧
      tableHMap t' = Apeiron.Table.insertWith (fun new _ => new) key (some value) (tableHMap self) ∧
      WellFormed t' ∧
      (old = none → t'.len.val = self.len.val + 1) ∧
      (old ≠ none → t'.len.val = self.len.val)
  | (core.result.Result.Err e, t') =>
      e = ({ key, value } : safe_table.TableFull I64 V) ∧
      tableHMap self key = none ∧
      NoEmptyIn self.slots.val 0 4 ∧
      t' = self

theorem try_insert_spec (self : SafeTable I64 V 4#usize) (key : I64) (value : V)
    (hwf : WellFormed self) :
    SafeTable.try_insert core.cmp.EqI64 self key value ⦃ r =>
      TryInsertPost self key value r ⦄ := by
  unfold SafeTable.try_insert
  step
  rcases o with _ | index
  · step
    rcases r with _ | index
    · rw [Aeneas.Std.WP.spec_ok]
      refine ⟨rfl, tableHMap_none_of_noMatchIn self key o_post, r_post, rfl⟩
    · obtain ⟨hlt, _, hempty, _⟩ := r_post
      have hno : NoMatchIn key self.slots.val 0 4 := o_post
      have hbefore : NoMatchIn key self.slots.val 0 index.val := by
        intro j hj hjlt; exact hno j hj (by omega)
      have hi : index.val < self.slots.val.length := by
        rw [Array.length_eq]; simpa using hlt
      refine Aeneas.Std.WP.spec_bind (Pₘ := fun _ : Unit => index < 4#usize) ?_ ?_
      · exact Aeneas.Std.massert_spec (index < 4#usize) (by scalar_tac)
      · intro _ _
        refine Aeneas.Std.WP.spec_bind (Pₘ := fun r : Option (I64 × V) ×
            (Option (I64 × V) → Std.Array (Option (I64 × V)) 4#usize) =>
              r.1 = (↑self.slots)[↑index] ∧ r.2 = self.slots.set index) ?_ ?_
        · exact Aeneas.Std.Array.index_mut_usize_spec self.slots index hi
        · rintro ⟨o2, back⟩ ⟨ho2, hback⟩
          simp only [Prod.fst, Prod.snd] at ho2 hback
          have hbackid : back o2 = self.slots := by
            rw [hback, ho2]
            apply Aeneas.Std.Array.ext
            rw [Aeneas.Std.Array.set_val_eq]
            exact List.set_getElem_self hi
          refine Aeneas.Std.WP.spec_bind (Pₘ := fun a1 : Std.Array (Option (I64 × V)) 4#usize =>
            a1 = (back o2).set index (some (key, value))) ?_ ?_
          · exact Aeneas.Std.Array.update_spec (back o2) index (some (key, value))
              (by rw [hbackid]; exact hi)
          · intro a1 ha1
            refine Aeneas.Std.WP.spec_bind (Pₘ := fun z : Std.Usize =>
              z.val = self.len.val + (1#usize).val) ?_ ?_
            · refine Aeneas.Std.Usize.add_spec ?_
              have := wellFormed_len_le_four hwf
              scalar_tac
            · intro i hi'
              rw [Aeneas.Std.WP.spec_ok]
              have hempty' : self.slots.val[index.val] = none := by
                have h := hempty
                rw [EmptyAt, List.getElem?_eq_getElem hi] at h
                exact Option.some.inj h
              have hl : self.slots.val.length = 4 := by rw [Array.length_eq]; rfl
              have ha1' : a1 = self.slots.set index (some (key, value)) := by
                rw [ha1, hbackid]
              refine ⟨?_, ?_, ?_, ?_, ?_⟩
              · exact (tableHMap_none_of_noMatchIn self key hno).symm
              · rw [ha1']
                exact tableHMap_set_eq_insertWith self hi hbefore
                  (fun x _ => not_matchesAt_of_emptyAt hempty)
              · rw [ha1']
                unfold WellFormed
                rw [Array.set_val_eq]
                refine ⟨?_, ?_⟩
                · have hc : List.countP (fun e : Option (I64 × V) => e.isSome)
                      (self.slots.val.set index.val (some (key, value))) = i.val := by
                    rw [countP_set_some hi hempty', ← hwf.1]
                    exact hi'.symm
                  rw [hc]
                · exact keysUnique_set_some hwf.2 hi
                    (fun j _ => noMatchIn_of_absent hno hl j)
              · intro _
                simpa using hi'
              · intro h; exact absurd rfl h
  · obtain ⟨hlt4, _, hmatch, hno⟩ := o_post
    have hi : index.val < self.slots.val.length := by
      rw [Array.length_eq]; simpa using hlt4
    refine Aeneas.Std.WP.spec_bind (Pₘ := fun r : Option (I64 × V) ×
        (Option (I64 × V) → Std.Array (Option (I64 × V)) 4#usize) =>
          r.1 = (↑self.slots)[↑index] ∧ r.2 = self.slots.set index) ?_ ?_
    · exact Aeneas.Std.Array.index_mut_usize_spec self.slots index hi
    · rintro ⟨o1, back⟩ ⟨ho1, hback⟩
      rcases o1 with _ | ⟨stored, old⟩
      · refine Aeneas.Std.WP.spec.ret ?_
        exact absurd hmatch (not_matchesAt_of_get hi ho1.symm)
      · have hget : self.slots.val[index.val] = some (stored, old) := ho1.symm
        have hkey : stored = key := by
          rcases hmatch with ⟨w, hw⟩
          have h1 : self.slots.val[index.val]? = some (some (stored, old)) := by
            rw [List.getElem?_eq_getElem hi, hget]
          rw [hw] at h1
          exact (Prod.mk.inj (Option.some.inj (Option.some.inj h1))).1.symm
        refine Aeneas.Std.WP.spec.ret ?_
        simp only [Prod.snd] at hback ⊢
        rw [hback, hkey]
        refine ⟨?_, ?_, ?_, ?_, ?_⟩
        · have hlook : self.slots.val[index.val]? = some (some (key, old)) := by
            rw [List.getElem?_eq_getElem hi, hget, hkey]
          exact (tableHMap_some_of_get self hmatch hno hlook).symm
        · exact tableHMap_set_eq_insertWith self hi hno
            (fun x hx => not_matchesAt_of_get_ne hi hget (by rw [hkey]; exact Ne.symm hx))
        · unfold WellFormed
          rw [Array.set_val_eq]
          refine ⟨?_, ?_⟩
          · have hc : List.countP (fun e : Option (I64 × V) => e.isSome)
                (self.slots.val.set index.val (some (key, value))) = self.len.val := by
              rw [List.countP_set (p := fun e : Option (I64 × V) => e.isSome) hi, hget]
              simp only [Option.isSome_some, if_true]
              have h1 := one_le_countP_of_get hi hget
              rw [hwf.1]
              omega
            rw [hc]
          · exact keysUnique_set_some hwf.2 hi
              (fun j hj hm => hj (hwf.2 key j index.val hm hmatch))
        · intro h; exact absurd h (by simp)
        · intro _; rfl

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

theorem tableHMap_set_none_self (self : SafeTable I64 V 4#usize) {i : Usize} {key : I64}
    (hfind : findIdx key self.slots.val = some i.val) (huniq : KeysUnique self.slots.val) :
    tableHMap { self with slots := self.slots.set i none } key = none := by
  rw [tableHMap_eq_lookupList, Array.set_val_eq]
  apply lookupList_none
  rw [findIdx_none_iff]
  intro j _ hj
  rw [List.length_set] at hj
  by_cases hji : j = i.val
  · subst hji; exact not_matchesAt_set_none
  · rw [matchesAt_set_ne_iff (Ne.symm hji)]
    intro hmatchj
    obtain ⟨hmatchi, _⟩ := (findIdx_spec key self.slots.val i.val).mp hfind
    exact hji (huniq key j i.val hmatchj hmatchi)

theorem remove_spec (self : SafeTable I64 V 4#usize) (key : I64) (hwf : WellFormed self) :
    SafeTable.remove core.cmp.EqI64 self key ⦃ r =>
      (let (old, t') := r
       old = tableHMap self key ∧
       tableHMap t' = Apeiron.Table.insertWith (fun _ _ => none) key none (tableHMap self) ∧
       WellFormed t') ⦄ := by
  unfold SafeTable.remove
  step
  rcases o with _ | index
  · have hnone : tableHMap self key = none := tableHMap_none_of_noMatchIn self key o_post
    rw [Aeneas.Std.WP.spec_ok]
    refine ⟨?_, ?_, ?_⟩
    · exact hnone.symm
    · funext x
      by_cases hx : x = key
      · rw [hx]; rw [hnone]; simp [Apeiron.Table.insertWith]
      · simp [Apeiron.Table.insertWith, hx]
    · exact hwf
  · obtain ⟨hlt4, _, hmatch, hno⟩ := o_post
    have hi : index.val < self.slots.val.length := by
      rw [Array.length_eq]; simpa using hlt4
    have hfind : findIdx key self.slots.val = some index.val :=
      (findIdx_spec key self.slots.val index.val).mpr ⟨hmatch, hno⟩
    have htake : core.option.Option.take
        (self.slots.val)[index.val] = ((self.slots.val)[index.val], none) := by
      cases (self.slots.val)[index.val] <;> rfl
    refine Aeneas.Std.WP.spec_bind (Pₘ := fun r : Option (I64 × V) ×
        (Option (I64 × V) → Std.Array (Option (I64 × V)) 4#usize) =>
          r.1 = (↑self.slots)[↑index] ∧ r.2 = self.slots.set index) ?_ ?_
    · exact Aeneas.Std.Array.index_mut_usize_spec self.slots index hi
    · rintro ⟨o1, back⟩ ⟨ho1, hback⟩
      rcases o1 with _ | ⟨stored, value⟩
      · refine Aeneas.Std.WP.spec.ret ?_
        exact absurd hmatch (not_matchesAt_of_get hi ho1.symm)
      · have hget : self.slots.val[index.val] = some (stored, value) := ho1.symm
        have hkey : stored = key := by
          rcases hmatch with ⟨w, hw⟩
          have h1 : self.slots.val[index.val]? = some (some (stored, value)) := by
            rw [List.getElem?_eq_getElem hi, hget]
          rw [hw] at h1
          exact (Prod.mk.inj (Option.some.inj (Option.some.inj h1))).1.symm
        have hlen : 1 ≤ self.len.val := by
          rw [hwf.1]
          exact one_le_countP_of_get hi (by rw [hkey] at hget; exact hget)
        refine Aeneas.Std.WP.spec_bind (Pₘ := fun z : Std.Usize =>
          z.val = self.len.val - (1#usize).val ∧ (1#usize).val ≤ self.len.val) ?_ ?_
        · refine Aeneas.Std.Usize.sub_spec ?_
          scalar_tac
        · intro i hi'
          refine Aeneas.Std.WP.spec.ret ?_
          refine ⟨?_, ?_, ?_⟩
          · have hlook : self.slots.val[index.val]? = some (some (key, value)) := by
              rw [List.getElem?_eq_getElem hi, hget, hkey]
            exact (tableHMap_some_of_get self hmatch hno hlook).symm
          · simp only [Prod.snd] at hback ⊢
            rw [hback]
            funext x
            by_cases hx : x = key
            · rw [hx]
              change tableHMap { self with slots := self.slots.set index none } key =
                insertWith (fun _ _ => none) key none (tableHMap self) key
              rw [tableHMap_set_none_self self hfind hwf.2]
              simp [Apeiron.Table.insertWith]
            · change tableHMap { self with slots := self.slots.set index none } x =
                insertWith (fun _ _ => none) key none (tableHMap self) x
              rw [tableHMap_set_none (t := self) hi
                (not_matchesAt_of_get_ne hi hget (by rw [hkey]; exact Ne.symm hx))]
              simp [Apeiron.Table.insertWith, hx]
          · simp only [Prod.snd] at hback ⊢
            rw [hback]
            unfold WellFormed
            rw [Array.set_val_eq]
            refine ⟨?_, ?_⟩
            · rw [countP_set_none hi hget, ← hwf.1]
              exact hi'.1
            · exact keysUnique_set_none hwf.2

end safe_table
end apeiron_word

namespace apeiron_word
namespace safe_table

open Aeneas Aeneas.Std Result Apeiron.Table

variable {V : Type}

/-! ### The generated empty state, its denotation, and the constructor obligation

`emptyTable` is the capacity-four all-`none` state the bridge constructs directly. It is a model
witness for the generated `safe_table.SafeTable`, not a refinement of the Rust `SafeTable::new`:
`proof/extract/lib.rs` roots `word_safe_table4_get/insert/remove` but no `new`, and `regen.sh`'s
Charon selection omits it. The generic `SafeTable::<K, V, N>::new` body repeats
`[const { None }; N]`, whose `Option<(K, V)>: Copy` obligation Charon cannot resolve at the generic
type. Initializing this generated state from the actual Rust constructor therefore remains owed;
nothing below claims that link. -/

/-- The generated capacity-four state with every slot `none` and `len = 0`. -/
def emptyTable : SafeTable I64 V 4#usize where
  slots := Array.repeat 4#usize (none : Option (I64 × V))
  len := 0#usize

theorem not_matchesAt_replicate_none {n i : ℕ} {k : I64} :
    ¬ MatchesAt k (List.replicate n (none : Option (I64 × V))) i := by
  rintro ⟨v, hv⟩
  rw [List.getElem?_replicate] at hv
  split at hv <;> simp_all

theorem keysUnique_replicate_none (n : ℕ) :
    KeysUnique (List.replicate n (none : Option (I64 × V))) := by
  intro k i j hi _
  exact absurd hi (not_matchesAt_replicate_none (n := n) (i := i) (k := k))

/-- The generated all-`none` state is well formed: `len = 0` counts no occupied slot, and no key
matches anywhere. This is a model witness for the generated state, not a constructor refinement. -/
theorem emptyTable_wellFormed : WellFormed (emptyTable (V := V)) := by
  unfold WellFormed emptyTable
  constructor
  · simp [Array.repeat_val, List.countP_replicate]
  · simpa only [Array.repeat_val] using
      keysUnique_replicate_none (V := V) (4#usize).val

/-- The empty state denotes the empty table: reading any key is `none`. -/
theorem emptyTable_tableHMap (key : I64) :
    tableHMap (emptyTable (V := V)) key = none := by
  rw [tableHMap_eq_lookupList]
  apply lookupList_none
  rw [findIdx_none_iff]
  intro j _ _
  rw [emptyTable, Array.repeat_val]
  exact not_matchesAt_replicate_none

theorem emptyTable_emptyAt_zero : EmptyAt (emptyTable (V := V)).slots.val 0 := by
  rw [EmptyAt, emptyTable, Array.repeat_val, List.getElem?_replicate]
  rw [show (4#usize).val = 4 by decide]
  rfl

theorem not_noEmptyIn_emptyTable :
    ¬ NoEmptyIn (emptyTable (V := V)).slots.val 0 4 := by
  intro h
  exact h 0 (by omega) (by omega) emptyTable_emptyAt_zero

/-- The free-slot success branch is inhabited: inserting any absent key into the generated empty
state returns `Ok none`, a well-formed successor, and `len` one larger. -/
theorem try_insert_emptyTable_success (key : I64) (value : V) :
    ∃ t', SafeTable.try_insert core.cmp.EqI64 (emptyTable (V := V)) key value =
        ok (core.result.Result.Ok none, t') ∧
      WellFormed t' ∧
      t'.len.val = (emptyTable (V := V)).len.val + 1 := by
  obtain ⟨y, hy, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
    (try_insert_spec (emptyTable (V := V)) key value (emptyTable_wellFormed (V := V)))
  obtain ⟨res, t'⟩ := y
  rcases res with old | e
  · simp only [TryInsertPost] at hpost
    have hnone : old = none := by rw [hpost.1, emptyTable_tableHMap]
    rw [hnone] at hy hpost
    refine ⟨t', hy, hpost.2.2.1, ?_⟩
    exact hpost.2.2.2.1 rfl
  · simp only [TryInsertPost] at hpost
    exact absurd hpost.2.2.1 not_noEmptyIn_emptyTable

/-- The generated state with one key at slot 0. -/
def oneTable (k : I64) (v : V) : SafeTable I64 V 4#usize where
  slots := Array.make 4#usize [some (k, v), none, none, none] (by simp)
  len := 1#usize

theorem keysUnique_oneTable (k : I64) (v : V) :
    KeysUnique (oneTable k v).slots.val := by
  intro key i j hi hj
  rcases hi with ⟨wi, hwi⟩
  rcases hj with ⟨wj, hwj⟩
  simp only [oneTable, Array.make_val] at hwi hwj
  have hi4 : i < 4 := by
    have := (List.getElem?_eq_some_iff.mp hwi).1
    simpa using this
  have hj4 : j < 4 := by
    have := (List.getElem?_eq_some_iff.mp hwj).1
    simpa using this
  interval_cases i <;> interval_cases j <;> simp_all

theorem oneTable_wellFormed (k : I64) (v : V) : WellFormed (oneTable k v) := by
  unfold WellFormed
  constructor
  · simp [oneTable, Array.make_val]
  · exact keysUnique_oneTable k v

theorem oneTable_tableHMap_self (k : I64) (v : V) :
    tableHMap (oneTable k v) k = some v := by
  rw [tableHMap_eq_lookupList, lookupList]
  simp only [oneTable, Array.make_val]
  simp [findIdx]

/-- The present-key replacement branch is inhabited: inserting an already-present key returns its
old value as `Ok (some _)`, a well-formed successor, and the same `len`. -/
theorem try_insert_oneTable_replace (k : I64) (v value : V) :
    ∃ t', SafeTable.try_insert core.cmp.EqI64 (oneTable k v) k value =
        ok (core.result.Result.Ok (some v), t') ∧
      WellFormed t' ∧
      t'.len.val = (oneTable k v).len.val := by
  obtain ⟨y, hy, hpost⟩ := Aeneas.Std.WP.spec_imp_exists
    (try_insert_spec (oneTable k v) k value (oneTable_wellFormed k v))
  obtain ⟨res, t'⟩ := y
  rcases res with old | e
  · simp only [TryInsertPost] at hpost
    have hold : old = some v := by rw [hpost.1, oneTable_tableHMap_self]
    rw [hold] at hy hpost
    refine ⟨t', hy, hpost.2.2.1, ?_⟩
    exact hpost.2.2.2.2 (by simp)
  · simp only [TryInsertPost] at hpost
    have hnone : tableHMap (oneTable k v) k = none := hpost.2.1
    rw [oneTable_tableHMap_self] at hnone
    exact absurd hnone (by simp)

/-- An absent key in a table with no empty slot is refused with the original payload and the
unchanged state. The strengthened `try_insert_spec` characterises this branch. -/
theorem try_insert_refusal (self : SafeTable I64 V 4#usize) (key : I64) (value : V)
    (habsent : NoMatchIn key self.slots.val 0 4)
    (hfull : NoEmptyIn self.slots.val 0 4) :
    SafeTable.try_insert core.cmp.EqI64 self key value =
      ok (core.result.Result.Err { key, value }, self) := by
  have hfind : SafeTable.find_from core.cmp.EqI64 self key 0#usize = ok none := by
    obtain ⟨o, ho, hpost⟩ := Aeneas.Std.WP.spec_imp_exists (find_from_zero_spec self key)
    cases o with
    | none => exact ho
    | some j =>
      exact absurd hpost.2.2.1 (habsent j.val (Nat.zero_le _) hpost.1)
  have hempty : SafeTable.empty_from core.cmp.EqI64 self 0#usize = ok none := by
    obtain ⟨o, ho, hpost⟩ := Aeneas.Std.WP.spec_imp_exists (empty_from_zero_spec self)
    cases o with
    | none => exact ho
    | some j =>
      exact absurd hpost.2.2.1 (hfull j.val (Nat.zero_le _) hpost.1)
  unfold SafeTable.try_insert
  rw [hfind, hempty]
  simp

/-- A generated capacity-four state whose four slots hold the distinct keys `0, 1, 2, 3`. -/
def fullTable (k0 k1 k2 k3 : I64) (v0 v1 v2 v3 : V) : SafeTable I64 V 4#usize where
  slots := Array.make 4#usize
    [some (k0, v0), some (k1, v1), some (k2, v2), some (k3, v3)] (by simp)
  len := 4#usize

/-- The full table holds no empty slot. -/
theorem fullTable_noEmpty {k0 k1 k2 k3 : I64} {v0 v1 v2 v3 : V} :
    NoEmptyIn (fullTable k0 k1 k2 k3 v0 v1 v2 v3).slots.val 0 4 := by
  intro j _ _
  interval_cases j <;> simp [EmptyAt, fullTable, Array.make_val]

/-- The full table with keys `0, 1, 2, 3` does not hold key `5`. -/
theorem fullTable_noMatch_five {v0 v1 v2 v3 : V} :
    NoMatchIn 5#i64 (fullTable 0#i64 1#i64 2#i64 3#i64 v0 v1 v2 v3).slots.val 0 4 := by
  intro j _ _
  rintro ⟨w, hw⟩
  simp only [fullTable, Array.make_val] at hw
  have hj4 : j < 4 := by
    have := (List.getElem?_eq_some_iff.mp hw).1
    simpa using this
  interval_cases j <;> simp_all

/-- The full-table refusal branch is inhabited: an absent key in the four-slot table returns the
original payload and the unchanged state. -/
theorem try_insert_fullTable_refusal {v0 v1 v2 v3 : V} :
    SafeTable.try_insert core.cmp.EqI64
        (fullTable 0#i64 1#i64 2#i64 3#i64 v0 v1 v2 v3) 5#i64 v0 =
      ok (core.result.Result.Err { key := 5#i64, value := v0 },
        fullTable 0#i64 1#i64 2#i64 3#i64 v0 v1 v2 v3) :=
  try_insert_refusal _ _ _ fullTable_noMatch_five fullTable_noEmpty

end safe_table
end apeiron_word

