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
