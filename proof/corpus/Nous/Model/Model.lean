import Apeiron.Table.Basic
import Apeiron.Trichotomy.AnyRing
import Nous.Model.State
import Nous.Model.Memory
import Nous.Model.Weighting
import Mathlib.Algebra.Field.Basic
import Mathlib.Tactic

/-!
# Model Assembly: Streaming State and Table Memory

This module defines the abstract sequential model assembly combining the streaming
state fold (`Nous.Model.State`) and the table memory update (`Nous.Model.Memory`).

## Operation Count in the Abstract Model

Per token step, the operation counts in the abstract model are:
- Streaming state update (`ringMul σ s v`):
  Expanding `(s.1 * v.1 + σ * s.2 * v.2, s.1 * v.2 + s.2 * v.1)` requires exactly
  4 ring multiplications, 1 scaling by `σ`, and 2 ring additions.
- Table memory update (`memoryStep σ k v M`):
  One `insertWith (ringMul σ) k v M` per token.

The probe-width bound `probe_width_bound` belongs to the executing face and is
not claimed here.

No attention mechanism, no channel decomposition, no concrete instances, and no
measured performance or empirical claim are asserted in this module.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Table
open Apeiron.Trichotomy.AnyRing

section Assembly

variable {R : Type} [CommRing R] {K : Type} [DecidableEq K]

/-- A single sequential model step threading both the ring-cell state and the table memory.

Binder order: the quadratic parameter `σ`, token key `k`, token value `v`, current state `s`,
and current table memory `M`. The token inputs `(k, v)` precede the accumulator states `(s, M)`.
-/
def step (σ : R) (k : K) (v : RingCell R) (s : RingCell R) (M : HMap K (RingCell R)) :
    RingCell R × HMap K (RingCell R) :=
  (ringMul σ s v, memoryStep σ k v M)

/-- Sequential fold of `memoryStep` across a list of key-value tokens starting from initial memory `M₀`. -/
def memoryFold (σ : R) (M₀ : HMap K (RingCell R)) (tokens : List (K × RingCell R)) :
    HMap K (RingCell R) :=
  tokens.foldl (fun M tok => memoryStep σ tok.1 tok.2 M) M₀

@[simp] theorem memoryFold_nil (σ : R) (M₀ : HMap K (RingCell R)) :
    memoryFold σ M₀ [] = M₀ :=
  rfl

/-- Run the model from an arbitrary initial state and memory accumulator. -/
def runFrom (σ : R) (s₀ : RingCell R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) :
    RingCell R × HMap K (RingCell R) :=
  tokens.foldl (fun acc tok => step σ tok.1 tok.2 acc.1 acc.2) (s₀, M₀)

@[simp] theorem runFrom_nil (σ : R) (s₀ : RingCell R) (M₀ : HMap K (RingCell R)) :
    runFrom σ s₀ M₀ [] = (s₀, M₀) :=
  rfl

/-- Run the model over a list of `(key, value)` tokens starting from `(ringOne, M₀)`.

Signature: `(σ : R) → HMap K (RingCell R) → List (K × RingCell R) → RingCell R × HMap K (RingCell R)`.
Threads the unit ring cell `ringOne` as initial state and `M₀` as initial table memory.
-/
def run (σ : R) (M₀ : HMap K (RingCell R)) (tokens : List (K × RingCell R)) :
    RingCell R × HMap K (RingCell R) :=
  runFrom σ ringOne M₀ tokens

@[simp] theorem run_nil (σ : R) (M₀ : HMap K (RingCell R)) :
    run σ M₀ [] = (ringOne, M₀) :=
  rfl

/-- Denotation lemma: the streaming state component of `run` equals `tokenFold` of the value stream. -/
theorem run_state_eq_tokenFold (σ : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) :
    (run σ M₀ tokens).1 = tokenFold σ (tokens.map Prod.snd) := by
  have h (s₀ : RingCell R) (M : HMap K (RingCell R)) (ts : List (K × RingCell R)) :
      (runFrom σ s₀ M ts).1 = (ts.map Prod.snd).foldl (ringMul σ) s₀ := by
    induction ts generalizing s₀ M with
    | nil => rfl
    | cons tok rest ih =>
      simp only [runFrom, List.foldl_cons, step]
      exact ih (ringMul σ s₀ tok.2) (memoryStep σ tok.1 tok.2 M)
  exact h ringOne M₀ tokens

/-- Denotation lemma: the table memory component of `run` equals the sequential fold of `memoryStep`. -/
theorem run_memory_eq_memoryFold (σ : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) :
    (run σ M₀ tokens).2 = memoryFold σ M₀ tokens := by
  have h (s₀ : RingCell R) (M : HMap K (RingCell R)) (ts : List (K × RingCell R)) :
      (runFrom σ s₀ M ts).2 = memoryFold σ M ts := by
    induction ts generalizing s₀ M with
    | nil => rfl
    | cons tok rest ih =>
      simp only [runFrom, memoryFold, List.foldl_cons, step]
      exact ih (ringMul σ s₀ tok.2) (memoryStep σ tok.1 tok.2 M)
  exact h ringOne M₀ tokens

/-- Denotation lemma: the table memory component of `run` expressed directly as `List.foldl`. -/
theorem run_memory_eq_foldl (σ : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) :
    (run σ M₀ tokens).2 = tokens.foldl (fun M tok => memoryStep σ tok.1 tok.2 M) M₀ :=
  run_memory_eq_memoryFold σ M₀ tokens

/-- Complete denotation: the model run is exactly the pair `(tokenFold, memoryFold)`. -/
theorem run_eq (σ : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) :
    run σ M₀ tokens = (tokenFold σ (tokens.map Prod.snd), memoryFold σ M₀ tokens) :=
  Prod.ext (run_state_eq_tokenFold σ M₀ tokens) (run_memory_eq_memoryFold σ M₀ tokens)

/-- Memory fold across concatenated token lists. -/
theorem memoryFold_append (σ : R) (M₀ : HMap K (RingCell R))
    (xs ys : List (K × RingCell R)) :
    memoryFold σ M₀ (xs ++ ys) = memoryFold σ (memoryFold σ M₀ xs) ys := by
  simp only [memoryFold, List.foldl_append]

/-- The state component of `run` splits multiplicatively across concatenated token lists. -/
theorem run_state_append (σ : R) (M₀ : HMap K (RingCell R))
    (xs ys : List (K × RingCell R)) :
    (run σ M₀ (xs ++ ys)).1 = ringMul σ (run σ M₀ xs).1 (run σ M₀ ys).1 := by
  simp only [run_state_eq_tokenFold, List.map_append]
  exact tokenFold_append σ (xs.map Prod.snd) (ys.map Prod.snd)

/-- The memory component of `run` sequences across concatenated token lists. -/
theorem run_memory_append (σ : R) (M₀ : HMap K (RingCell R))
    (xs ys : List (K × RingCell R)) :
    (run σ M₀ (xs ++ ys)).2 = memoryFold σ (run σ M₀ xs).2 ys := by
  simp only [run_memory_eq_memoryFold, memoryFold_append]

end Assembly

section Readout

variable {R : Type} [Field R] {K : Type} [DecidableEq K]

/-- Model readout of a ring cell as the channel weighting ratio.

Under the field division hypothesis matching `channelWeight`, the readout computes
the ratio of the upper channel coordinate to the total channel sum.
-/
def readout (r : R) (z : RingCell R) : R :=
  channelWeight r z

/-- Readout of the streaming state component of the model accumulator. -/
def readoutState (r : R) (state : RingCell R × HMap K (RingCell R)) : R :=
  readout r state.1

/-- Readout of the table memory at key `k` as the channel weighting ratio. -/
def readoutMemory (r : R) (k : K) (state : RingCell R × HMap K (RingCell R)) : R :=
  readout r (state.2 k)

/-- Readout of the state resulting from running the model over a token sequence. -/
def runReadout (σ : R) (r : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) : R :=
  readout r (run σ M₀ tokens).1

/-- Readout of the table memory at key `k` after running the model over a token sequence. -/
def runReadoutMemory (σ : R) (r : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) (k : K) : R :=
  readout r ((run σ M₀ tokens).2 k)

/-- The run state readout equals the channel weight of the token fold of values. -/
theorem runReadout_eq (σ : R) (r : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) :
    runReadout σ r M₀ tokens = channelWeight r (tokenFold σ (tokens.map Prod.snd)) := by
  simp only [runReadout, readout, run_state_eq_tokenFold]

/-- The run memory readout at key `k` equals the channel weight of `memoryFold` at key `k`. -/
theorem runReadoutMemory_eq (σ : R) (r : R) (M₀ : HMap K (RingCell R))
    (tokens : List (K × RingCell R)) (k : K) :
    runReadoutMemory σ r M₀ tokens k = channelWeight r ((memoryFold σ M₀ tokens) k) := by
  simp only [runReadoutMemory, readout, run_memory_eq_memoryFold]

/-- When `2 * z.1 ≠ 0`, the readout simplifies to `ringUpper r z / (2 * z.1)`. -/
theorem readout_eq {r : R} (z : RingCell R) (h : 2 * z.1 ≠ 0) :
    readout r z = ringUpper r z / (2 * z.1) :=
  channelWeight_eq z h

/-- Complementary lower channel readout ratio. -/
def lowerReadout (r : R) (z : RingCell R) : R :=
  lowerChannelWeight r z

/-- The readout and lower readout sum to unity when `2 * z.1 ≠ 0`. -/
theorem readout_add_lowerReadout {r : R} (z : RingCell R) (h : 2 * z.1 ≠ 0) :
    readout r z + lowerReadout r z = 1 :=
  channelWeight_add_lowerChannelWeight z h

end Readout

end Nous.Model
