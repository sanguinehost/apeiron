/-
Copyright (c) 2026 Apeiron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Apeiron
-/
import Mathlib.Data.List.OfFn
import Mathlib.Data.Fin.Tuple.Take
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Tactic

/-!
# Jacobi decoding reaches the greedy sequence exactly

Greedy decoding from a prompt `p : List V` with a deterministic next-token map `f : List V → V`
emits `greedy p f n`, the first `n` generated tokens. Writing `y_i` for the `i`-th generated token,
the sequential system is `y_i = f (p ++ [y_0, …, y_{i-1}])`.

Santilli, Casola, Noviello and Cinardi, *Accelerating Transformer Inference for Translation via
Parallel Decoding* (arXiv 2305.10427), §3.2 replaces this system by a Jacobi iteration on a window
`y : Fin m → V`:

`jacobiStep y i = f (p ++ (List.ofFn y).take i)`,

and §3.4 Proposition 1 states that the iteration yields the same result as greedy autoregressive
decoding in at most `m` parallel iterations, for any initialization, provided the stopping
condition `y_{k-1} - y_k = 0` (their equation (5)) holds. This file proves that statement, its
fixed-point stopping rule, and the guess-and-verify step of Leviathan, Kalman and Matias,
*Fast Inference from Transformers via Speculative Decoding* (arXiv 2211.17192), §2, in the
n-gram/lookahead form of Fu, Bailis, Stoica and Zhang, *Break the Sequential Dependency of LLM
Inference Using Lookahead Decoding* (arXiv 2402.02057), §2–§3.

* `jacobi_prefix_correct`: from any initialization, `k` iterations fix the first `min k m` window
  positions to the greedy tokens.
* `jacobi_exact`: `m` iterations give exactly the greedy window, from any initialization.
* `jacobi_fixed_point_iff`: a window is a fixed point of `jacobiStep` exactly when it is the greedy
  window, so stopping at a fixed point returns the greedy output.
* `verify_accept_prefix`: accepting the longest agreeing prefix of a draft `d` and appending one
  bonus token reproduces the greedy sequence, for every draft `d` of length at most `m`.
* `passes_bound`: with each verification pass emitting its accepted draft tokens plus one token,
  the number of passes to emit `m` tokens is at most `m` and equals `m` minus the accepted total.

The vocabulary `V` is an arbitrary type and `f` is deterministic and total, so every definition
and theorem below is total and holds at `m = 0`. The trust boundary is the Lean kernel check of
this file: no statement here covers a sampling decoder, a model whose next-token map depends on
keys and values beyond the emitted prefix, or a hardware implementation of `f`.
-/

namespace Nous
namespace Model

variable {V : Type*}

/-! ## Sequential greedy decoding -/

/-- `greedy p f n` is the list of the first `n` tokens greedily decoded from the prompt `p` with
next-token map `f`. Temporal cost: `n` applications of `f` and `n` list appends whose arguments
have length at most `n`, counted in list elements. Space: `O(n)` for the result and for the prompt
prefix handed to `f` at each step. Defined by recursion on `n`. -/
def greedy (p : List V) (f : List V → V) : ℕ → List V
  | 0 => []
  | n + 1 => greedy p f n ++ [f (p ++ greedy p f n)]

@[simp] theorem greedy_zero (p : List V) (f : List V → V) : greedy p f 0 = [] := rfl

theorem greedy_succ (p : List V) (f : List V → V) (n : ℕ) :
    greedy p f (n + 1) = greedy p f n ++ [f (p ++ greedy p f n)] := rfl

theorem greedy_length (p : List V) (f : List V → V) (n : ℕ) : (greedy p f n).length = n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [greedy_succ, ih]

/-- The first `j` greedy tokens are the length-`j` prefix of the first `n`, for `j ≤ n`. -/
theorem greedy_take_of_le (p : List V) (f : List V → V) {j n : ℕ} (h : j ≤ n) :
    (greedy p f n).take j = greedy p f j := by
  induction n with
  | zero =>
    have hj : j = 0 := Nat.le_zero.mp h
    subst hj
    rfl
  | succ n ih =>
    rcases Nat.lt_or_eq_of_le h with hlt | heq
    · have hj : j ≤ n := Nat.lt_succ_iff.mp hlt
      rw [greedy_succ,
        List.take_append_of_le_length (by rw [greedy_length]; exact hj)]
      exact ih hj
    · subst heq
      rw [greedy_succ]
      exact List.take_of_length_le (by simp [greedy_length])

/-! ## The greedy window and the Jacobi step -/

/-- The greedy window of width `m`: the tokens greedy decoding emits after the prompt. The identity
`ofFn_greedyWin` converts between the window and `greedy p f m`. -/
def greedyWin (p : List V) (f : List V → V) (m : ℕ) : Fin m → V :=
  fun i => f (p ++ greedy p f i.val)

/-- `List.ofFn` of the greedy window is the greedy sequence's first `m` tokens. -/
theorem ofFn_greedyWin (p : List V) (f : List V → V) (m : ℕ) :
    List.ofFn (greedyWin p f m) = greedy p f m := by
  induction m with
  | zero => rfl
  | succ m ih =>
    have hcast : (fun i : Fin m => greedyWin p f (m + 1) i.castSucc) = greedyWin p f m := by
      funext i
      simp [greedyWin]
    rw [List.ofFn_succ', hcast, ih, List.concat_eq_append]
    rfl

/-- One Jacobi update writes position `i` from the window's prefix of length `i`. Temporal cost
`O(m)` for the `ofFn`/`take` prefix plus one application of `f`; space `O(m)`. -/
def jacobiStep (p : List V) (f : List V → V) (m : ℕ) (y : Fin m → V) : Fin m → V :=
  fun i => f (p ++ (List.ofFn y).take i.val)

/-- `k` Jacobi updates from the initialization `y0`. -/
def jacobiIter (p : List V) (f : List V → V) (m : ℕ) (k : ℕ) (y0 : Fin m → V) : Fin m → V :=
  (jacobiStep p f m)^[k] y0

/-! ## Prefix correctness and exactness -/

/-- **Prefix correctness.** After `k` Jacobi iterations from any initialization `y0`, the first
`min k m` window tokens are the greedy tokens. Temporal cost of the induction: `O(k · m)`
`ofFn`/`take` work; space `O(m)`. -/
theorem jacobi_take (p : List V) (f : List V → V) (m : ℕ) (k : ℕ) (y0 : Fin m → V) :
    ∀ n ≤ min k m, (List.ofFn (jacobiIter p f m k y0)).take n = greedy p f n := by
  induction k with
  | zero =>
    intro n hn
    have h0 : n = 0 := Nat.le_zero.mp (le_trans hn (min_le_left 0 m))
    subst h0
    rfl
  | succ k ih =>
    intro n hn
    have hnm : n ≤ m := le_trans hn (min_le_right (k + 1) m)
    have hkey : Fin.take n hnm (jacobiStep p f m (jacobiIter p f m k y0))
        = Fin.take n hnm (greedyWin p f m) := by
      funext j
      rw [Fin.take_apply, Fin.take_apply]
      have hj : (Fin.castLE hnm j).val ≤ min k m := by
        have hjlt : (Fin.castLE hnm j).val < n := by
          rw [Fin.val_castLE]; exact j.isLt
        rcases le_total m k with hmk | hmk
        · rw [Nat.min_eq_right hmk]
          exact le_of_lt (lt_of_lt_of_le hjlt (le_trans hn (min_le_right (k + 1) m)))
        · rw [Nat.min_eq_left hmk]
          have : (Fin.castLE hnm j).val < k + 1 :=
            lt_of_lt_of_le hjlt (le_trans hn (min_le_left (k + 1) m))
          omega
      simp only [jacobiStep, greedyWin]
      rw [ih (Fin.castLE hnm j).val hj]
    calc (List.ofFn (jacobiIter p f m (k + 1) y0)).take n
        = (List.ofFn (jacobiStep p f m (jacobiIter p f m k y0))).take n := by
            simp only [jacobiIter, Function.iterate_succ_apply']
      _ = List.ofFn (Fin.take n hnm (jacobiStep p f m (jacobiIter p f m k y0))) :=
            (Fin.ofFn_take_eq_take_ofFn hnm _).symm
      _ = List.ofFn (Fin.take n hnm (greedyWin p f m)) := by rw [hkey]
      _ = (List.ofFn (greedyWin p f m)).take n := Fin.ofFn_take_eq_take_ofFn hnm _
      _ = (greedy p f m).take n := by rw [ofFn_greedyWin]
      _ = greedy p f n := greedy_take_of_le p f hnm

/-- **Prefix correctness, headline form.** After `k` iterations from any `y0`, the first
`min k m` tokens of the window equal greedy's first `min k m` tokens. -/
theorem jacobi_prefix_correct (p : List V) (f : List V → V) (m : ℕ) (k : ℕ) (y0 : Fin m → V) :
    (List.ofFn (jacobiIter p f m k y0)).take (min k m) = greedy p f (min k m) :=
  jacobi_take p f m k y0 (min k m) le_rfl

/-- **Exactness.** `m` Jacobi iterations give exactly the greedy window, for every `y0`. -/
theorem jacobi_exact (p : List V) (f : List V → V) (m : ℕ) (y0 : Fin m → V) :
    List.ofFn (jacobiIter p f m m y0) = greedy p f m := by
  have h := jacobi_take p f m m y0 m (by rw [min_self])
  have htake : (List.ofFn (jacobiIter p f m m y0)).take m = List.ofFn (jacobiIter p f m m y0) :=
    List.take_of_length_le (by simp)
  rw [← htake]
  exact h

/-! ## The fixed point is the greedy window -/

/-- The greedy window is a fixed point of `jacobiStep`. -/
theorem jacobiStep_greedyWin (p : List V) (f : List V → V) (m : ℕ) :
    jacobiStep p f m (greedyWin p f m) = greedyWin p f m := by
  funext i
  simp only [jacobiStep, greedyWin]
  rw [ofFn_greedyWin, greedy_take_of_le p f (le_of_lt i.isLt)]

/-- **Early exit is exact.** A window `y` is a fixed point of `jacobiStep` exactly when it is the
greedy window, i.e. exactly when its `List.ofFn` is `greedy p f m`. -/
theorem jacobi_fixed_point_iff (p : List V) (f : List V → V) (m : ℕ) (y : Fin m → V) :
    jacobiStep p f m y = y ↔ List.ofFn y = greedy p f m := by
  constructor
  · intro hy
    have hiter : ∀ k : ℕ, jacobiIter p f m k y = y := by
      intro k
      induction k with
      | zero => rfl
      | succ k ihk =>
        have hstep : jacobiIter p f m (k + 1) y = jacobiStep p f m (jacobiIter p f m k y) :=
          by simp only [jacobiIter, Function.iterate_succ_apply']
        rw [hstep, ihk, hy]
    have h := jacobi_take p f m m y m (by rw [min_self])
    rw [hiter m] at h
    rw [← List.take_of_length_le (l := List.ofFn y) (i := m) (by simp)]
    exact h
  · intro hy
    have hfun : y = greedyWin p f m :=
      List.ofFn_inj.mp (by rw [hy, ofFn_greedyWin])
    rw [hfun, jacobiStep_greedyWin]

/-! ## Draft verification -/

/-- `acceptFrom p f acc d` walks the draft `d` from the accepted greedy prefix `acc`: a draft token
equal to `f (p ++ acc)` is accepted and the walk continues; the first disagreement stops it. In
both cases one bonus token `f (p ++ acc)` is appended at the stop. Temporal cost: one comparison
and one append per draft token, the appended lists having length at most `|acc| + |d|`, counted in
list elements; space `O(|acc| + |d|)`. -/
def acceptFrom (p : List V) (f : List V → V) [DecidableEq V] : List V → List V → List V
  | acc, [] => acc ++ [f (p ++ acc)]
  | acc, x :: xs =>
      if x = f (p ++ acc) then acceptFrom p f (acc ++ [x]) xs
      else acc ++ [f (p ++ acc)]

/-- The tokens emitted by verifying the draft `d` against the greedy sequence: the longest
agreeing prefix of `d` plus one bonus token. -/
def accepted (p : List V) (f : List V → V) [DecidableEq V] (d : List V) : List V :=
  acceptFrom p f [] d

/-- Appending the next greedy token after a greedy prefix keeps the result a greedy prefix. -/
theorem greedy_snoc (p : List V) (f : List V → V) {n : ℕ} {acc : List V} {x : V}
    (h : acc = greedy p f n) (hx : x = f (p ++ acc)) :
    acc ++ [x] = greedy p f (n + 1) := by
  calc acc ++ [x] = greedy p f n ++ [x] := by rw [h]
    _ = greedy p f n ++ [f (p ++ acc)] := by rw [hx]
    _ = greedy p f n ++ [f (p ++ greedy p f n)] := by
          rw [congrArg (fun a => f (p ++ a)) h]
    _ = greedy p f (n + 1) := (greedy_succ p f n).symm

/-- **Verification preserves the greedy sequence.** Walking any draft `d` from a greedy prefix
returns a greedy prefix: `acceptFrom p f acc d = greedy p f (acceptFrom p f acc d).length`. -/
theorem acceptFrom_eq_greedy (p : List V) (f : List V → V) [DecidableEq V] (d : List V) :
    ∀ (acc : List V) (n : ℕ), acc = greedy p f n →
      acceptFrom p f acc d = greedy p f (acceptFrom p f acc d).length := by
  induction d with
  | nil =>
    intro acc n h
    have hlhs : acc ++ [f (p ++ acc)] = greedy p f (n + 1) := greedy_snoc p f h rfl
    have hlen : (acc ++ [f (p ++ acc)]).length = n + 1 := by
      rw [List.length_append, List.length_singleton, h, greedy_length]
    show acceptFrom p f acc [] = greedy p f (acceptFrom p f acc []).length
    rw [show acceptFrom p f acc [] = acc ++ [f (p ++ acc)] from rfl, hlen, hlhs]
  | cons x xs ih =>
    intro acc n h
    have hlhs : acc ++ [f (p ++ acc)] = greedy p f (n + 1) := greedy_snoc p f h rfl
    have hlen : (acc ++ [f (p ++ acc)]).length = n + 1 := by
      rw [List.length_append, List.length_singleton, h, greedy_length]
    show acceptFrom p f acc (x :: xs) =
      greedy p f (acceptFrom p f acc (x :: xs)).length
    rw [show acceptFrom p f acc (x :: xs) =
      (if x = f (p ++ acc) then acceptFrom p f (acc ++ [x]) xs
       else acc ++ [f (p ++ acc)]) from rfl]
    by_cases hx : x = f (p ++ acc)
    · rw [if_pos hx]
      exact ih (acc ++ [x]) (n + 1) (greedy_snoc p f h hx)
    · rw [if_neg hx, hlen, hlhs]

/-- **Verification emits the greedy sequence.** For every draft `d` (of length at most `m`),
`accepted p f d` is the greedy sequence: `accepted p f d = greedy p f (accepted p f d).length`.
Hence the emitted prompt-plus-accepted sequence is the greedy sequence, whatever the draft. -/
theorem verify_accept_prefix (p : List V) (f : List V → V) [DecidableEq V] (m : ℕ) (d : List V)
    (_hd : d.length ≤ m) :
    accepted p f d = greedy p f (accepted p f d).length :=
  acceptFrom_eq_greedy p f d [] 0 rfl

/-- **Verification emits the greedy sequence, prompt included.** The prompt-plus-emitted sequence
`p ++ accepted p f d` is the prompt plus a greedy prefix of length `(accepted p f d).length`. -/
theorem verify_accept_append (p : List V) (f : List V → V) [DecidableEq V] (m : ℕ) (d : List V)
    (hd : d.length ≤ m) :
    p ++ accepted p f d = p ++ greedy p f (accepted p f d).length :=
  congrArg (fun l => p ++ l) (verify_accept_prefix p f m d hd)

/-! ## Pass count -/

/-- In the verification model a pass emits its accepted draft tokens plus one token, so it always
emits at least one token. -/
theorem pass_emit_ge_one (P : ℕ) (acc : Fin P → ℕ) (i : Fin P) : 1 ≤ acc i + 1 :=
  Nat.le_add_left 1 (acc i)

/-- **Passes are at least the emitted tokens.** If each of `P` passes emits at least one token and
`m` tokens are emitted in total, then `P ≤ m`. -/
theorem passes_le_of_emit (P m : ℕ) (emit : Fin P → ℕ) (hpos : ∀ i, 1 ≤ emit i)
    (h : m = ∑ i : Fin P, emit i) : P ≤ m := by
  rw [h]
  calc P = ∑ _i : Fin P, (1 : ℕ) := by simp
    _ ≤ ∑ i : Fin P, emit i := Finset.sum_le_sum (fun i _ => hpos i)

/-- **Pass count identity.** With each pass `i` emitting `acc i + 1` tokens, the number of passes
for `m` tokens is `m` minus the total accepted draft tokens `∑ i, acc i`. -/
theorem passes_eq_of_emit (P m : ℕ) (acc : Fin P → ℕ)
    (h : m = ∑ i : Fin P, (acc i + 1)) : P = m - ∑ i : Fin P, acc i := by
  rw [h, Finset.sum_add_distrib]
  simp

/-- **Passes bound.** In the model where each verification pass emits `acc i + 1` tokens — its
accepted draft tokens plus one, hence at least one — emitting `m` tokens takes at most `m` passes
and exactly `m - ∑ i, acc i` passes. -/
theorem passes_bound (P m : ℕ) (acc : Fin P → ℕ) (h : m = ∑ i : Fin P, (acc i + 1)) :
    P ≤ m ∧ P = m - ∑ i : Fin P, acc i :=
  ⟨passes_le_of_emit P m (fun i => acc i + 1) (fun i => pass_emit_ge_one P acc i) h,
   passes_eq_of_emit P m acc h⟩

/-! ## Witnesses -/

namespace Witness

/-- The deterministic next-token map on `Bool` tokens that emits `true` while fewer than three
tokens are present and `false` afterwards. -/
def f3 : List Bool → Bool := fun l => l.length < 3

/-- The greedy sequence of `f3` from the empty prompt is `[true, true, true, false, …]`. -/
theorem greedy_witness : greedy ([] : List Bool) f3 4 = [true, true, true, false] := by
  decide

/-- **Non-vacuity of `jacobi_exact`.** From the initialization `(false, false, false)`, three Jacobi
iterations return the greedy window `[true, true, true]` exactly, whereas the initialization is not
a fixed point. -/
theorem jacobi_exact_witness :
    List.ofFn (jacobiIter ([] : List Bool) f3 3 3 (fun _ => false)) = [true, true, true] := by
  decide

/-- **Boundary case `m = 0`.** The width-zero window equals the greedy sequence of length zero,
from any initialization. -/
theorem jacobi_exact_zero (f : List V → V) (y0 : Fin 0 → V) :
    List.ofFn (jacobiIter ([] : List V) f 0 0 y0) = greedy ([] : List V) f 0 := by
  simp [jacobiIter]

end Witness

end Model
end Nous
