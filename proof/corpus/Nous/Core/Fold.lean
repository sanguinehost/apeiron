import Mathlib.Tactic

/-!
# Indexed sequence traversals

Forward and reverse state threading over a list. The development is generic in the state type
`σ`, the element type `α` and the step `f : σ → α → σ`; the step takes the state first and the
input element second, matching the argument order of `List.foldl f init xs`.

`accumList f init xs` traverses `xs` from the first element to the last. The entry at index `i`
is the fold of the first `i + 1` elements, so the entry at the last index is the fold over the
whole input (`accumList_getLastD`). `mapAccum` packages these partial folds as an `Array`
together with the size law `results.size = xs.length`.

`accumListRight f init xs` traverses from the last element to the first. The entry at index `i`
is the fold of the elements from the last down to index `i`; the entry at index `0` is the right
fold over the whole input (`accumListRight_headD`), and the list is the reverse of the forward
partial folds of the reversed input (`accumListRight_eq_reverse`). `mapAccumRight` packages
these partial folds as an `Array` with the same size law.

Both traversals are total: no precondition is required, an empty input yields an empty result,
and nothing is discarded. Only the size law, the partial-fold agreement and the reversal
relation are claimed; no commutativity, ordering or performance claim is made.
-/

namespace Nous.Core

variable {σ α : Type*}

/-- Forward partial folds. `accumList f init xs` is the list whose entry at index `i` is the
fold of the first `i + 1` elements of `xs` from the state `init`, with the step `f` receiving
the state first and the element second. The empty input gives the empty list.

Cost: `xs.length` applications of `f`; O(`xs.length`) list cells and no auxiliary array. -/
def accumList (f : σ → α → σ) (init : σ) : List α → List σ
  | [] => []
  | x :: xs => f init x :: accumList f (f init x) xs

/-- `accumList f init xs` has the length of `xs`. -/
theorem accumList_length (f : σ → α → σ) (init : σ) (xs : List α) :
    (accumList f init xs).length = xs.length := by
  induction xs generalizing init with
  | nil => rfl
  | cons x xs ih => simp only [accumList, List.length_cons, ih]

/-- Each forward entry is the partial fold of the input prefix it indexes: entry `i` is the
fold of `xs.take (i + 1)`. -/
theorem accumList_getElem? (f : σ → α → σ) (init : σ) (xs : List α) (i : Nat) :
    (accumList f init xs)[i]? =
      if i < xs.length then some (List.foldl f init (xs.take (i + 1))) else none := by
  induction xs generalizing init i with
  | nil => simp [accumList]
  | cons x xs ih =>
      cases i with
      | zero => simp [accumList]
      | succ j =>
          simp only [accumList, List.getElem?_cons_succ, ih (f init x) j,
            List.take_succ_cons, List.foldl_cons, List.length_cons, Nat.succ_lt_succ_iff]

/-- The forward traversal agrees with `List.foldl` at its last entry, stated with the total
`getLastD` so that the empty input is the initial state on both sides. -/
theorem accumList_getLastD (f : σ → α → σ) (init : σ) (xs : List α) :
    (accumList f init xs).getLastD init = List.foldl f init xs := by
  induction xs generalizing init with
  | nil => rfl
  | cons x xs ih =>
      simp only [accumList, List.getLastD_cons, List.foldl_cons]
      exact ih (f init x)

/-- A nonempty list whose total last-element read is `b` has `getLast? = some b`; the total
`getLastD` and the partial `getLast?` agree on a nonempty list. -/
theorem getLast?_eq_some_of_getLastD (l : List α) (a b : α) (hne : l ≠ [])
    (h : l.getLastD a = b) : l.getLast? = some b := by
  cases hl : l.getLast? with
  | none => exact absurd (List.getLast?_eq_none_iff.mp hl) hne
  | some c =>
      have hc : c = b := by
        rw [← h, List.getLastD_eq_getLast?, hl]
        rfl
      rw [hc]

/-- The last forward partial fold is the left fold over the whole input, for a nonempty input. -/
theorem accumList_getLast?_of_ne_nil (f : σ → α → σ) (xs : List α) (init : σ) (h : xs ≠ []) :
    (accumList f init xs).getLast? = some (List.foldl f init xs) :=
  getLast?_eq_some_of_getLastD (accumList f init xs) init (List.foldl f init xs)
    (by
      rw [← List.length_pos_iff, accumList_length]
      exact List.length_pos_iff.mpr h)
    (accumList_getLastD f init xs)

/-- The last forward partial fold of a cons input is the left fold over that input. -/
theorem accumList_getLast?_cons (f : σ → α → σ) (init : σ) (x : α) (xs : List α) :
    (accumList f init (x :: xs)).getLast? = some (List.foldl f init (x :: xs)) :=
  accumList_getLast?_of_ne_nil f (x :: xs) init (by simp)

/-- Reverse partial folds. `accumListRight f init xs` is the list whose entry at index `i` is
the fold of the elements from the last down to index `i` of `xs` from the state `init`,
computed as the forward partial folds of the reversed input, reversed. The empty input gives
the empty list. -/
def accumListRight (f : σ → α → σ) (init : σ) (xs : List α) : List σ :=
  (accumList f init xs.reverse).reverse

/-- `accumListRight f init xs` has the length of `xs`. -/
theorem accumListRight_length (f : σ → α → σ) (init : σ) (xs : List α) :
    (accumListRight f init xs).length = xs.length := by
  simp only [accumListRight, List.length_reverse, accumList_length]

/-- The reverse traversal is the reverse of the forward traversal of the reversed input. No
commutativity of the step is required for this identity. -/
theorem accumListRight_eq_reverse (f : σ → α → σ) (init : σ) (xs : List α) :
    accumListRight f init xs = (accumList f init xs.reverse).reverse :=
  rfl

/-- The first reverse partial fold is the right fold over the whole input; because
`List.foldr` takes the element first, the step appears as `fun y acc => f acc y`. -/
theorem accumListRight_head?_cons (f : σ → α → σ) (init : σ) (x : α) (xs : List α) :
    (accumListRight f init (x :: xs)).head? =
      some (List.foldr (fun y acc => f acc y) init (x :: xs)) := by
  simp only [accumListRight, List.head?_reverse]
  rw [accumList_getLast?_of_ne_nil f (x :: xs).reverse init (by simp)]
  rw [List.foldl_reverse]

/-- The reverse traversal agrees with `List.foldr` at its first entry, stated with the total
`headD` so that the empty input is the initial state on both sides. -/
theorem accumListRight_headD (f : σ → α → σ) (init : σ) (xs : List α) :
    (accumListRight f init xs).headD init = List.foldr (fun y acc => f acc y) init xs := by
  rw [accumListRight, List.headD_eq_head?_getD, List.head?_reverse]
  rw [← List.getLastD_eq_getLast?, accumList_getLastD, List.foldl_reverse]

/-- Forward indexed traversal: the partial folds of `accumList`, packaged as an `Array` whose
size is the input length. -/
def mapAccum (f : σ → α → σ) (init : σ) (xs : List α) :
    { results : Array σ // results.size = xs.length } :=
  ⟨(accumList f init xs).toArray, by simp only [List.size_toArray, accumList_length]⟩

/-- Reverse indexed traversal: the partial folds of `accumListRight`, packaged as an `Array`
whose size is the input length. -/
def mapAccumRight (f : σ → α → σ) (init : σ) (xs : List α) :
    { results : Array σ // results.size = xs.length } :=
  ⟨(accumListRight f init xs).toArray, by simp only [List.size_toArray, accumListRight_length]⟩

/-- The forward result carries the size law as part of its type. -/
theorem mapAccum_size (f : σ → α → σ) (init : σ) (xs : List α) :
    (mapAccum f init xs).1.size = xs.length :=
  (mapAccum f init xs).2

/-- The reverse result carries the size law as part of its type. -/
theorem mapAccumRight_size (f : σ → α → σ) (init : σ) (xs : List α) :
    (mapAccumRight f init xs).1.size = xs.length :=
  (mapAccumRight f init xs).2

/-- The forward `Array` is the array of the partial-fold list. -/
theorem mapAccum_toList (f : σ → α → σ) (init : σ) (xs : List α) :
    (mapAccum f init xs).1.toList = accumList f init xs :=
  rfl

/-- The reverse `Array` is the array of the partial-fold list. -/
theorem mapAccumRight_toList (f : σ → α → σ) (init : σ) (xs : List α) :
    (mapAccumRight f init xs).1.toList = accumListRight f init xs :=
  rfl

/-- The last forward result of `mapAccum` is the left fold over the whole input. -/
theorem mapAccum_back?_cons (f : σ → α → σ) (init : σ) (x : α) (xs : List α) :
    (mapAccum f init (x :: xs)).1.back? = some (List.foldl f init (x :: xs)) := by
  show (accumList f init (x :: xs)).toArray.back? = _
  rw [List.back?_toArray, accumList_getLast?_cons]

/-- The first reverse result of `mapAccumRight` is the right fold over the whole input. -/
theorem mapAccumRight_getElem?_zero (f : σ → α → σ) (init : σ) (x : α) (xs : List α) :
    (mapAccumRight f init (x :: xs)).1[0]? =
      some (List.foldr (fun y acc => f acc y) init (x :: xs)) := by
  show (accumListRight f init (x :: xs)).toArray[0]? = _
  rw [List.getElem?_toArray, ← List.head?_eq_getElem?, accumListRight_head?_cons]

-- Boundary evaluations at a fixed step: the forward prefix folds and the reversed prefix folds.

#guard accumList (fun s a => s + a) 0 [1, 2, 3] = [1, 3, 6]

#guard accumListRight (fun s a => s + a) 0 [1, 2, 3] = [6, 5, 3]

#guard (mapAccum (fun s a => s + a) 0 [1, 2, 3]).1 = #[1, 3, 6]

#guard (mapAccumRight (fun s a => s + a) 0 [1, 2, 3]).1 = #[6, 5, 3]

#guard (mapAccum (fun s a => s + a) 0 ([] : List Nat)).1 = (#[] : Array Nat)

end Nous.Core
