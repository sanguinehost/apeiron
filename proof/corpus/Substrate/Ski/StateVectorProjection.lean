import Substrate.Ski.PolyIotaTermAlgebra

/-!
# State-Vector Projection (Universal Algorithm Bridge — Phase 3)
@name Reaching Into the Stack
@child A running program keeps its values in a stack, newest on top. To grab the one you need, you just count down from the top. This proves the counting always lands on exactly the right value.
@scientist The registers live at each instruction in a Church cons-list state vector (head = most recent SSA value); reading the operand an instruction needs is projection at depth `d = k − i`, the composition `fst (snd^d state)` — no new combinators, just `fst`/`snd` derived from `S`/`K`/`I`.
@engineer Modeled as `List α`: `proj_zero_cons` returns the head, `proj_snd_cons` exposes the tail, `proj_succ_cons` steps the drop. The closed form — depth-`d` projection of a prepend-built `N`-element state = SSA index `N−1−d` — is the reading of those three, not a theorem here.

The Sanguine lifter (`interop/src/translator.rs`) emitted LLVM IR as a
sequence of instructions touching SSA registers. To make the lifted form
*executable* (per `theory/UNIVERSAL_ALGORITHM_BRIDGE.md` §4, Gap D), the
registers in scope at each instruction must be unified into a single
*state vector* that the next instruction reads. Neither artefact survives:
`interop/` went in `4556df7d` ("Delete all Rust") and `src/kernel/State.s`
in `f8efce70` ("Clean slate"), both 2026-05-31 — they are named here as
history, and no theorem below needs either.

This file discharges the projection contract for the **Church cons-list**
state-vector representation chosen in Phase 3 of the bridge, documented
in `src/kernel/State.s`:

  state = cons(Reg_{K-1})(cons(Reg_{K-2})(...cons(Reg_0)(nil)))

with head = most recently written SSA value (prepend convention). The
translator computes projection depth `d = k - i` at lift time, where `k`
is the current instruction index and `i` is the SSA index of the operand
it needs. The projection at depth `d` is

  fst (snd^d state)

— a composition of the existing `src/kernel/Data.s` primitives `fst` and
`snd` (themselves derived from `S_Comb`, `K_Comb`, `Identity` in
`Iota.s`). No new combinators are introduced.

## The model

Sanguine's substrate is untyped — every value is a `Comb`. The cons-list
state vector is recursively-typed (a cell is `nil` or `cons h t` where
`t` is itself a state vector). The cleanest Lean model is the native
`List α`:

* `cons h t` ≈ `h :: t`
* `fst (cons h t)` ≈ `(h :: t).head?` — returns `some h`
* `snd (cons h t)` ≈ `(h :: t).tail` — returns `t`
* `snd^d (cons … nil)` ≈ `List.drop d` applied to the same list
* Projection at depth `d` ≈ `(List.drop d state).head?`

## The contract

Two theorems anchor every projection the Phase 2 translator will emit:

* **`proj_zero_cons`** — depth-0 projection returns the head.
* **`proj_succ_cons`** — stripping one extra cell from the prefix is
  equivalent to stripping it off the tail directly.

Their composition gives the closed-form contract:

  proj_d (cons r_{N-1} (cons r_{N-2} ... (cons r_0 nil))) = some r_{N-1-d}

i.e. projecting at depth `d` from the head of an `N`-element
prepend-built state returns SSA value index `(N-1-d)`.

## What is proved here vs what is read onto the substrate

* **Proved (native):** the three projection equations on `List α`, each by `rfl` —
  `proj_zero_cons` (depth 0 returns the head), `proj_snd_cons` (one cell off exposes the tail),
  `proj_succ_cons` (`(h :: t).drop (n + 1) = t.drop n`).
* **Next:** the closed form
  `proj_d (cons r_{N-1} … nil) = some r_{N-1-d}` — depth-`d` projection of an `N`-element
  prepend-built state returns SSA index `N-1-d`. It is stated above as the composition of the
  three; no statement below quantifies over `N` or `d`, and the `@engineer` line says so.
* **Anchor:** that `List α` models the substrate's Church cons-list, that
  `fst` / `snd` on `Comb` are `head?` / `tail`, and that the lifter
  (`interop/src/translator.rs`) computes depth `d = k - i` at lift time. The substrate is
  untyped `Comb`; the model here is Lean's `List`, and no theorem welds the two.

-/

namespace Sanguine

universe u

variable {α : Type u}

-- ═══════════════════════════════════════════════════════════════════
-- THE PROJECTION CONTRACT
-- ═══════════════════════════════════════════════════════════════════

/-- **proj_zero_cons.** Depth-0 projection of a prepend-built state
    returns the most-recently-written SSA value. Corresponds to
    `fst(cons(h)(t)) = h` in the substrate. -/
theorem proj_zero_cons (h : α) (t : List α) :
    (h :: t).head? = some h := rfl

/-- **proj_snd_cons.** Stripping one cell from the head exposes the
    tail. Corresponds to `snd(cons(h)(t)) = t` in the substrate. -/
theorem proj_snd_cons (h : α) (t : List α) :
    (h :: t).tail = t := rfl

/-- **proj_succ_cons.** Applying `List.drop (n+1)` to `h :: t` equals
    applying `List.drop n` to `t`. By induction on `n` together with
    `proj_zero_cons`, this gives the closed-form projection contract:
    depth-`d` projection from a length-`N` prepend-built state returns
    SSA value index `(N-1-d)`. -/
theorem proj_succ_cons (n : Nat) (h : α) (t : List α) :
    (h :: t).drop (n + 1) = t.drop n := rfl

end Sanguine
