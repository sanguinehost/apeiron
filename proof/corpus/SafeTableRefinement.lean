import ApeironWord

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
