/-
Copyright (c) 2026 Apeiron. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Apeiron
-/
import Nous.Model.JacobiDecode
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic

/-!
# Self-speculative decoding with exact rollback

A deterministic engine step `f : S → T → S × T` maps a state and an input token to the next state
and the greedy next token. `greedyRun f s t n` is greedy decoding from state `s` with seed token `t`:
it emits `n` tokens and returns the state reached after consuming the seed and the first `n - 1`
emitted tokens, so the last emitted token is the seed of the next round.

A speculative round starts from `(s, t)` and a draft `d : List T` from an arbitrary drafter.
The verifier applies `f` to `t`, then to `d₀`, `d₁`, … and accepts the longest prefix of `d` on which
the emitted token equals the next draft token; at the first disagreement, or after the whole draft,
it emits the one produced token. `verifyAux` implements this and returns the state after the last
accepted input together with the emitted tokens. `spec_round_eq_greedy` shows that for every draft
the emitted tokens are `greedyRun`'s first `accepted + 1` tokens and the returned state is
`greedyRun`'s state at that length, so a wrong draft costs work, never output.

## Why exact rollback is exact undo

The engine's affine update writes `S̃ ← S̃ + R(k, d)` elementwise, where `R` is a deterministic
rounded product matrix of the key and the (deferred-decay) delta. Over an additive group the write is
reversed exactly by subtracting the same `R`: `(S + R) - R = S` (`undo_write`), so a rejected token's
state update is undone with no residue. The same holds for the scalar decay when the previous scalar
is stored before the pass (`restore_snapshot`). This is the property exact-integer arithmetic has and
IEEE floating-point addition lacks: in general `fl (fl (a + b) - b) ≠ a`, so a float write cannot be
undone by subtraction alone.

Engine sites this licenses:

* `src/engine/mixer.rs`, `delta_head_step_ref` / `delta_head_step_fast`, the rank-1 write pass
  `S̃ += k · d'ᵀ` (`delta_head_step_ref` lines 442–457): rejecting a speculative token subtracts the
  same rank-1 term elementwise.
* `src/engine/attn.rs`, `KvState::truncate(m)`: causal attention makes rows `0..m` independent of
  later appends, so rewinding the occupied counters to `m` is the append/truncate identity below
  (`truncate_append_length`, `truncate_length_add_append`); no cache row is copied or recomputed.

## What is not covered

* Renormalisation. When the deferred scalar `c` drops below `2⁻⁸`, `delta_head_step_ref` folds it
  into the state, `S̃ ← S̃ · c`, rounds, and resets `c ← 2³²`. That map is not inverted by the
  subtraction above; the engine must snapshot the head (state and scalar) before a speculative pass
  in which a renormalisation can occur, and restore the snapshot on rejection.
* Sampling decoders, dropouts, and any step whose next token is not a function of `(state, token)`.
* `f` itself. The model takes `f` as an arbitrary deterministic total function; nothing here proves
  that the Rust executor implements a particular `f`, nor anything about rustc/LLVM codegen, the
  kernel ABI, or the `unsafe` container.
* Runtime behaviour: the theorems are kernel checks over the model; they say nothing about timing.

## Relation to `JacobiDecode.verify_accept_prefix`

`JacobiDecode` models greedy decoding by a prefix map `g : List V → V`: its draft walk
`acceptFrom p g acc d` compares the next draft token with `g (p ++ acc)` and appends the one produced
token. The stateful engine step induces the same shape with the empty prompt. Define
`nextToken f s t acc` as the token emitted from the state that has consumed `t` and the accepted
prefix `acc`, with input the last token of `t :: acc`. Then `verifyAux`'s output tokens equal
`JacobiDecode.accepted [] (nextToken f s t) d` (`verifyAux_tokens_eq_accepted`), so
`verify_accept_prefix` applies and gives the token half of `spec_round_eq_greedy`
(`spec_round_tokens_eq_greedy_via_jacobi`); the bridge `greedyRun_tokens_eq_prefix` identifies
`greedy [] (nextToken f s t)` with `greedyRun`'s tokens, so the prefix-greedy sequence is the
stateful greedy sequence. The state half of `spec_round_eq_greedy` is proved directly.
-/

namespace Nous
namespace Model

variable {S T : Type*}

/-! ## The engine step and greedy decoding -/

/-- Next state of the deterministic step `f`. -/
def push (f : S → T → S × T) (s : S) (x : T) : S := (f s x).1

/-- Emitted token of the deterministic step `f`. -/
def emit (f : S → T → S × T) (s : S) (x : T) : T := (f s x).2

/-- `runState f s xs` consumes the inputs `xs` from the state `s`. Temporal cost `|xs|` state
updates; space `O(1)` beyond the inputs. -/
def runState (f : S → T → S × T) : S → List T → S
  | s, [] => s
  | s, x :: xs => runState f (push f s x) xs

@[simp] theorem runState_nil (f : S → T → S × T) (s : S) : runState f s [] = s := rfl

theorem runState_cons (f : S → T → S × T) (s : S) (x : T) (xs : List T) :
    runState f s (x :: xs) = runState f (push f s x) xs := rfl

/-- `lastInput t acc` is the token that feeds the step after the accepted prefix `acc`: the last
element of `acc`, or the seed `t` when `acc` is empty. -/
def lastInput (t : T) (acc : List T) : T := acc.getLastD t

@[simp] theorem lastInput_nil (t : T) : lastInput t [] = t := rfl

@[simp] theorem lastInput_singleton (t x : T) : lastInput t [x] = x := rfl

theorem lastInput_cons (t x : T) (xs : List T) : lastInput t (x :: xs) = lastInput x xs :=
  List.getLastD_cons

/-- Greedy decoding from state `s` with seed token `t`: `greedyRun f s t n` returns the final state
and the `n` emitted tokens. Temporal cost: `n` steps of `f` and `n` list conses, counted in tokens;
space `O(n)` for the token list, `O(1)` beyond it. Defined by recursion on `n`. -/
def greedyRun (f : S → T → S × T) (s : S) (t : T) : ℕ → S × List T
  | 0 => (s, [])
  | n + 1 =>
      ((greedyRun f (f s t).1 (f s t).2 n).1, (f s t).2 :: (greedyRun f (f s t).1 (f s t).2 n).2)

@[simp] theorem greedyRun_zero (f : S → T → S × T) (s : S) (t : T) :
    greedyRun f s t 0 = (s, []) := rfl

theorem greedyRun_succ (f : S → T → S × T) (s : S) (t : T) (n : ℕ) :
    greedyRun f s t (n + 1) =
      ((greedyRun f (f s t).1 (f s t).2 n).1,
        (f s t).2 :: (greedyRun f (f s t).1 (f s t).2 n).2) := rfl

/-- The `n` tokens emitted by greedy decoding from `(s, t)`. -/
def greedyTokens (f : S → T → S × T) (s : S) (t : T) (n : ℕ) : List T :=
  (greedyRun f s t n).2

/-- The state after `n` greedy tokens: it has consumed `t` and the first `n - 1` emitted tokens. -/
def greedyState (f : S → T → S × T) (s : S) (t : T) (n : ℕ) : S :=
  (greedyRun f s t n).1

theorem greedyRun_length (f : S → T → S × T) (s : S) (t : T) :
    ∀ n : ℕ, (greedyRun f s t n).2.length = n
  | 0 => rfl
  | n + 1 => by simp [greedyRun_succ, greedyRun_length f _ _ n]

/-! ## The speculative round -/

/-- One speculative pass. From `(s, t)` and the draft `d`, compare the token emitted by the step
against `d₀`; on agreement accept `d₀`, step again from the new state with input `d₀`, and continue;
at the first disagreement, or after the whole draft, emit the produced token and return the state
after the last accepted input. Temporal cost: one step and one comparison per draft token, counted in
tokens; space `O(|d|)` for the emitted list. -/
def verifyAux (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) : List T → S × List T
  | [] => ((f s t).1, [(f s t).2])
  | d0 :: rest =>
      if (f s t).2 = d0 then
        ((verifyAux f (f s t).1 d0 rest).1, d0 :: (verifyAux f (f s t).1 d0 rest).2)
      else ((f s t).1, [(f s t).2])

@[simp] theorem verifyAux_nil (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) :
    verifyAux f s t [] = ((f s t).1, [(f s t).2]) := rfl

theorem verifyAux_cons (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) (d0 : T)
    (rest : List T) :
    verifyAux f s t (d0 :: rest) =
      (if (f s t).2 = d0 then
        ((verifyAux f (f s t).1 d0 rest).1, d0 :: (verifyAux f (f s t).1 d0 rest).2)
      else ((f s t).1, [(f s t).2])) := rfl

/-- The state-and-tokens returned by a round. -/
def specRound (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) (d : List T) : S × List T :=
  verifyAux f s t d

/-- **Existence.** One speculative pass from `(s, t)` with any draft is greedy decoding from `(s, t)`
of some length: the state returned is `greedyRun`'s state at that length and the tokens are its
tokens. -/
theorem verifyAux_eq_greedyRun (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) (d : List T) :
    ∃ n : ℕ, verifyAux f s t d = greedyRun f s t n := by
  induction d generalizing s t with
  | nil => exact ⟨1, rfl⟩
  | cons d0 rest ih =>
      by_cases h : (f s t).2 = d0
      · obtain ⟨k, hk⟩ := ih (f s t).1 d0
        refine ⟨k + 1, ?_⟩
        rw [verifyAux_cons, if_pos h, hk, greedyRun_succ, ← h]
      · exact ⟨1, by rw [verifyAux_cons, if_neg h]; rfl⟩

/-- **Headline 1 (one round).** For every draft `d`, the tokens emitted by one speculative round are
greedy decoding's first `accepted + 1` tokens and the returned state is greedy decoding's state at
that length. -/
theorem spec_round_eq_greedy (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) (d : List T) :
    verifyAux f s t d = greedyRun f s t (verifyAux f s t d).2.length := by
  obtain ⟨n, hn⟩ := verifyAux_eq_greedyRun f s t d
  rw [hn, greedyRun_length]

/-- Token half of `spec_round_eq_greedy`. -/
theorem spec_round_tokens_eq_greedy (f : S → T → S × T) [DecidableEq T] (s : S) (t : T)
    (d : List T) :
    (verifyAux f s t d).2 = (greedyRun f s t (verifyAux f s t d).2.length).2 :=
  congrArg Prod.snd (spec_round_eq_greedy f s t d)

/-- State half of `spec_round_eq_greedy`. -/
theorem spec_round_state_eq_greedy (f : S → T → S × T) [DecidableEq T] (s : S) (t : T)
    (d : List T) :
    (verifyAux f s t d).1 = (greedyRun f s t (verifyAux f s t d).2.length).1 :=
  congrArg Prod.fst (spec_round_eq_greedy f s t d)

/-- A round always emits at least one token, so the seed of the next round is the last emitted
token. -/
theorem verifyAux_length_pos (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) (d : List T) :
    1 ≤ (verifyAux f s t d).2.length := by
  induction d generalizing s t with
  | nil => simp [verifyAux_nil]
  | cons d0 rest ih =>
      rw [verifyAux_cons]
      by_cases h : (f s t).2 = d0
      · rw [if_pos h]; simp
      · rw [if_neg h]; simp

/-! ## Iterated rounds -/

/-- `greedyRun` splits after `m ≥ 1` steps: the state and seed left by the first `m` steps continue
greedy decoding for the remaining `k` steps, and the token lists concatenate. -/
theorem greedyRun_append (f : S → T → S × T) (s : S) (t : T) (m : ℕ) (hm : 1 ≤ m) (k : ℕ) :
    greedyRun f s t (m + k) =
      ((greedyRun f (greedyRun f s t m).1 (lastInput t (greedyRun f s t m).2) k).1,
       (greedyRun f s t m).2 ++
         (greedyRun f (greedyRun f s t m).1 (lastInput t (greedyRun f s t m).2) k).2) := by
  induction m generalizing s t k with
  | zero => omega
  | succ m ih =>
      rcases Nat.eq_zero_or_pos m with hm0 | hmpos
      · subst hm0
        rw [show 0 + 1 + k = k + 1 from by omega, greedyRun_succ, greedyRun_succ]
        simp
      · have h1 : 1 ≤ m := hmpos
        rw [show m + 1 + k = (m + k) + 1 from by omega, greedyRun_succ]
        rw [ih (f s t).1 (f s t).2 h1 k, greedyRun_succ]
        simp only [lastInput_cons, List.cons_append]

/-- `specIter f D s t n` runs `n` speculative rounds from `(s, t)`, each round's draft produced by
the arbitrary drafter `D` at the round's starting state and seed; it returns the final state and the
concatenated emitted tokens. Temporal cost: `n` rounds, each as `verifyAux`. -/
def specIter (f : S → T → S × T) [DecidableEq T] (D : S → T → List T) (s : S) (t : T) :
    ℕ → S × List T
  | 0 => (s, [])
  | n + 1 =>
      ((specIter f D (verifyAux f s t (D s t)).1
          (lastInput t (verifyAux f s t (D s t)).2) n).1,
       (verifyAux f s t (D s t)).2 ++
         (specIter f D (verifyAux f s t (D s t)).1
           (lastInput t (verifyAux f s t (D s t)).2) n).2)

/-- **Headline 1 (all rounds).** `spec_decode_eq_greedy`: for any drafter `D`, the tokens emitted by
`n` speculative rounds are greedy decoding's first `emitted` tokens and the final state is greedy
decoding's state at that length. Drafter quality changes only how many tokens each round emits. -/
theorem spec_decode_eq_greedy (f : S → T → S × T) [DecidableEq T] (D : S → T → List T) :
    ∀ (s : S) (t : T) (n : ℕ),
      specIter f D s t n = greedyRun f s t (specIter f D s t n).2.length := by
  intro s t n
  induction n generalizing s t with
  | zero => simp [specIter]
  | succ n ih =>
      generalize hR : verifyAux f s t (D s t) = R
      have hpos : 1 ≤ R.2.length := hR ▸ verifyAux_length_pos f s t (D s t)
      have hRg : R = greedyRun f s t R.2.length := hR ▸ spec_round_eq_greedy f s t (D s t)
      have ih' := ih R.1 (lastInput t R.2)
      simp only [specIter]
      rw [hR]
      set Q := specIter f D R.1 (lastInput t R.2) n
      have hga : greedyRun f s t (R.2.length + Q.2.length) =
          ((greedyRun f R.1 (lastInput t R.2) Q.2.length).1,
           R.2 ++ (greedyRun f R.1 (lastInput t R.2) Q.2.length).2) := by
        have h := greedyRun_append f s t R.2.length hpos Q.2.length
        rw [← hRg] at h
        exact h
      rw [List.length_append]
      conv_lhs => rw [ih']
      rw [← hga]

/-! ## Connection to `JacobiDecode` (headline 4) -/

/-- The prefix map induced by the stateful step: from the state that has consumed `t` and the
accepted prefix `acc`, with input the last token of `t :: acc`, `nextToken f s t acc` is the next
token. This is `JacobiDecode`'s `g : List V → V` with the empty prompt. -/
def nextToken (f : S → T → S × T) (s : S) (t : T) (acc : List T) : T :=
  emit f (runState f s ((t :: acc).dropLast)) (lastInput t acc)

theorem nextToken_nil (f : S → T → S × T) (s : S) (t : T) :
    nextToken f s t [] = (f s t).2 := rfl

/-- One more accepted draft token shifts the state and the seed: the map induced from `(s, t)` at
input `y` is the map induced from `(push f s t, y)`. -/
theorem nextToken_shift (f : S → T → S × T) (s : S) (t y : T) (z : List T) :
    nextToken f (push f s t) y z = nextToken f s t (y :: z) := by
  unfold nextToken
  rw [← runState_cons]
  simp only [List.dropLast_cons_cons, lastInput_cons]

/-- `JacobiDecode.acceptFrom` with a prepended accumulator: prepending `pre` to the accumulator is
appending `pre` to the output and post-composing the next-token map with `pre ++ ·`. -/
theorem acceptFrom_pre (F : List T → T) [DecidableEq T] (pre : List T) :
    ∀ (acc d : List T),
      acceptFrom [] F (pre ++ acc) d = pre ++ acceptFrom [] (fun z => F (pre ++ z)) acc d := by
  intro acc d
  induction d generalizing acc with
  | nil => simp [acceptFrom, List.append_assoc]
  | cons x xs ih =>
      simp only [acceptFrom, List.nil_append]
      by_cases h : x = F (pre ++ acc)
      · simp only [if_pos h, List.append_assoc]
        exact ih (acc ++ [x])
      · simp only [if_neg h, List.append_assoc]

/-- **Headline 4 (token reuse).** The tokens emitted by one speculative round are exactly
`JacobiDecode.accepted [] (nextToken f s t) d`, the output of the prefix-map draft walk with the
empty prompt. -/
theorem verifyAux_tokens_eq_acceptFrom (f : S → T → S × T) [DecidableEq T] (s : S) (t : T) :
    ∀ d : List T, (verifyAux f s t d).2 = acceptFrom [] (nextToken f s t) [] d := by
  intro d
  induction d generalizing s t with
  | nil => rfl
  | cons d0 rest ih =>
      rw [verifyAux_cons]
      have hF : nextToken f s t [] = (f s t).2 := nextToken_nil f s t
      by_cases h : (f s t).2 = d0
      · rw [if_pos h, ih (f s t).1 d0]
        have hacc : acceptFrom [] (nextToken f s t) [] (d0 :: rest)
            = acceptFrom [] (nextToken f s t) [d0] rest := by
          rw [acceptFrom, if_pos (by rw [List.nil_append, hF]; exact h.symm), List.nil_append]
        have hF2 : (fun z => nextToken f s t ([d0] ++ z)) = nextToken f (f s t).1 d0 := by
          funext z
          rw [List.singleton_append, ← nextToken_shift f s t d0 z]
          rfl
        have hpre := acceptFrom_pre (nextToken f s t) [d0] [] rest
        rw [List.append_nil] at hpre
        rw [hacc, hpre, hF2, List.singleton_append]
      · rw [if_neg h]
        have hacc : acceptFrom [] (nextToken f s t) [] (d0 :: rest) = [(f s t).2] := by
          rw [acceptFrom, if_neg (by rw [List.nil_append, hF]; exact fun hh => h hh.symm)]
          simp only [List.nil_append, hF]
        rw [hacc]

theorem verifyAux_tokens_eq_accepted (f : S → T → S × T) [DecidableEq T] (s : S) (t : T)
    (d : List T) :
    (verifyAux f s t d).2 = accepted [] (nextToken f s t) d :=
  verifyAux_tokens_eq_acceptFrom f s t d

/-- The prefix-greedy `JacobiDecode.greedy` with the empty prompt and `nextToken` is the stateful
`greedyRun`'s token list. -/
theorem greedy_shift (p : List T) (g : List T → T) :
    ∀ n : ℕ, greedy p g n = greedy [] (fun z => g (p ++ z)) n
  | 0 => rfl
  | n + 1 => by
      rw [greedy_succ, greedy_shift p g n]
      rfl

theorem greedy_cons (p : List T) (g : List T → T) :
    ∀ n : ℕ, greedy p g (n + 1) = g p :: greedy (p ++ [g p]) g n
  | 0 => by simp [greedy_succ]
  | n + 1 => by
      nth_rewrite 1 [greedy_succ]
      conv_rhs => rw [greedy_succ]
      rw [greedy_cons p g n, List.cons_append]
      have harg : p ++ (g p :: greedy (p ++ [g p]) g n)
          = (p ++ [g p]) ++ greedy (p ++ [g p]) g n := by
        rw [← List.singleton_append, List.append_assoc]
      rw [harg]

theorem greedyRun_tokens_eq_prefix (f : S → T → S × T) (s : S) (t : T) :
    ∀ n : ℕ, (greedyRun f s t n).2 = greedy [] (nextToken f s t) n := by
  intro n
  induction n generalizing s t with
  | zero => rfl
  | succ n ih =>
      rw [greedyRun_succ]
      simp only [ih (f s t).1 (f s t).2]
      rw [greedy_cons ([] : List T) (nextToken f s t) n, nextToken_nil, List.nil_append]
      rw [greedy_shift [(f s t).2] (nextToken f s t) n]
      congr 1
      apply congrArg (fun H => greedy [] H n)
      funext z
      rw [List.singleton_append, ← nextToken_shift f s t (f s t).2 z]
      rfl

/-- **Headline 4.** The token half of `spec_round_eq_greedy` derived from
`JacobiDecode.verify_accept_prefix` and the stateful/prefix bridge. -/
theorem spec_round_tokens_eq_greedy_via_jacobi (f : S → T → S × T) [DecidableEq T] (s : S) (t : T)
    (d : List T) :
    (verifyAux f s t d).2 = (greedyRun f s t (verifyAux f s t d).2.length).2 := by
  have h := verifyAux_tokens_eq_accepted f s t d
  rw [h]
  nth_rewrite 1 [verify_accept_prefix [] (nextToken f s t) d.length d le_rfl]
  exact (greedyRun_tokens_eq_prefix f s t (accepted [] (nextToken f s t) d).length).symm

/-! ## Exact undo of the integer rank-1 write -/

section Undo

/-- Store the deferred scalar before a pass. -/
def snapshot {G : Type*} (c : G) : G := c

/-- **Headline 2 (scalar undo).** Restoring the stored scalar returns the previous value exactly. -/
theorem restore_snapshot {G : Type*} (c : G) : snapshot c = c := rfl

variable {G : Type*} [AddCommGroup G]
variable {m n : ℕ}

/-- The rank-1 write pass: add the deterministic rounded product matrix `R` elementwise to the state
`S̃`. Temporal cost `m · n` group additions; space `O(1)` beyond `S` and `R`. -/
def writeMat (S R : Fin m → Fin n → G) : Fin m → Fin n → G := S + R

/-- **Headline 2 (write undo).** Over an additive group, subtracting the same rounded products
reverses the write exactly: `(S + R) - R = S` pointwise. Applied to `S : Fin m → Fin n → ℤ` updated by
`S i j + r i j`, a rejected token's state update is undone with no residue. -/
theorem undo_write (S R : Fin m → Fin n → G) : writeMat S R - R = S := by
  ext i j
  simp [writeMat]

/-- The write pass on the pair `(state, scalar)`: the state receives `R`, the scalar is carried. -/
def writePass (S : Fin m → Fin n → G) (R : Fin m → Fin n → G) (c : G) :
    (Fin m → Fin n → G) × G :=
  (writeMat S R, c)

/-- **Headline 2 (combined).** Undoing a write pass restores the state and the stored scalar. -/
theorem undo_writePass (S R : Fin m → Fin n → G) (c : G) :
    writePass (writeMat S R - R) 0 (snapshot c) = (S, c) := by
  simp [writePass, writeMat, snapshot]

/-- Witness: over `ℤ` at `m = n = 2`, the update `S i j + r i j` reversed by subtracting `r`. -/
theorem undo_write_witness :
    (writeMat (m := 2) (n := 2) ![![(1 : ℤ), 2], ![3, 4]] ![![5, 6], ![7, 8]])
        - ![![5, 6], ![7, 8]] = ![![(1 : ℤ), 2], ![3, 4]] := by
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [writeMat]

end Undo

/-! ## KV list rollback -/

section KV

variable {V : Type*}

/-- `truncate l m` keeps the first `m` entries of the KV token list; it is the list shadow of
`KvState::truncate`, which rewinds the occupied counter to `m`. Temporal cost `m`; space `O(m)`. -/
def truncate (l : List V) (m : ℕ) : List V := l.take m

/-- **Headline 3 (truncate to the pre-append length).** Appending `extra` entries and truncating back
to the length before the append returns exactly the original list:
`(base ++ extra).take base.length = base`. -/
theorem truncate_append_length (base extra : List V) :
    truncate (base ++ extra) base.length = base :=
  List.take_append_length

/-- **Headline 3 (keep the accepted entries).** Truncating to `base.length + a` keeps exactly the `a`
accepted entries appended after `base`:
`(base ++ extra).take (base.length + a) = base ++ extra.take a`. -/
theorem truncate_length_add_append (base extra : List V) (a : ℕ) :
    truncate (base ++ extra) (base.length + a) = base ++ extra.take a :=
  List.take_length_add_append a

/-- Truncation is idempotent: truncating twice to `m` equals truncating once. -/
theorem truncate_idempotent (l : List V) (m : ℕ) : truncate (truncate l m) m = truncate l m := by
  simp [truncate, List.take_take]

/-- Appending one entry and truncating to the pre-append length drops it. -/
theorem truncate_append_singleton (base : List V) (x : V) :
    truncate (base ++ [x]) base.length = base :=
  List.take_append_length

end KV

/-! ## Witnesses -/

namespace RollbackWitness

/-- A deterministic step on `ℕ`: `f s t = (s + 1, s + t)`, so the state counts steps and the token
is the running sum. -/
def fW : ℕ → ℕ → ℕ × ℕ := fun s t => (s + 1, s + t)

/-- Greedy decoding of `fW` from `(0, 0)` emits `[0, 1, 3]` in three steps and reaches state `3`. -/
theorem greedy_witness : greedyRun fW 0 0 3 = (3, [0, 1, 3]) := by decide

/-- **Headline 5 (partly wrong draft).** From `(0, 0)` with draft `[0, 1, 9]`, the verifier accepts
`0, 1`, rejects `9`, emits the corrected token `3`, and returns the state `3`; the emitted tokens are
greedy's first `accepted + 1 = 3` tokens and the state is greedy's state at that length. -/
theorem spec_round_witness :
    verifyAux fW 0 0 [0, 1, 9] = (3, [0, 1, 3]) ∧
      verifyAux fW 0 0 [0, 1, 9] = greedyRun fW 0 0 (verifyAux fW 0 0 [0, 1, 9]).2.length := by
  decide

/-- **Headline 5 (wrong first token).** With draft `[7, 0, 0]` the first token already disagrees, so
the round emits only the corrected token `0` and returns the state `1`. -/
theorem spec_round_witness_first :
    verifyAux fW 0 0 [7, 0, 0] = (1, [0]) ∧
      (verifyAux fW 0 0 [7, 0, 0]).2 = (greedyRun fW 0 0 1).2 := by
  decide

/-- **Headline 5 (fully accepted draft).** With draft `[0, 1, 3]` the verifier accepts all three and
emits the one produced token `6`, so the round reaches state `4` with greedy's first four tokens. -/
theorem spec_round_witness_all :
    verifyAux fW 0 0 [0, 1, 3] = (4, [0, 1, 3, 6]) ∧
      (verifyAux fW 0 0 [0, 1, 3]).2 = (greedyRun fW 0 0 4).2 := by
  decide

end RollbackWitness

end Model
end Nous
