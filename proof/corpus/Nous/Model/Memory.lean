import Apeiron.Table.Basic
import Apeiron.Trichotomy.AnyRing
import Mathlib.Logic.Function.Basic
import Mathlib.Tactic

/-!
# Table Memory Step Over an Arbitrary Commutative Ring

This module formalizes the table memory update step using the canonical abstract
map model `Apeiron.Table.HMap` with absence carried by a default value.

For a key `k : K`, value `v : RingCell R`, parameter `σ : R`, and map `M : HMap K (RingCell R)`,
the memory update step merges `v` into `M` at key `k` using the canonical ring multiplication:
`memoryStep σ k v M := insertWith (ringMul σ) k v M`.

The update is characterized as a standard function update via `memory_step_model`:
`memoryStep σ k v M = Function.update M k (ringMul σ v (M k))`.

Absence is represented by the canonical default model of `HMap K V := K → V`; no `Option`-based
absence API is introduced. The fixed probe-width bound is an obligation of the executing
(Rust/growable) face, named `probe_width_bound`, and is not claimed here.
-/

set_option linter.apeiron.notation false
set_option linter.apeiron.citation false

namespace Nous.Model

open Apeiron.Table
open Apeiron.Trichotomy.AnyRing

variable {R : Type} [CommRing R] {K : Type} [DecidableEq K]

/-- The memory update step: merge an incoming ring cell into a table using `ringMul σ`. -/
def memoryStep (σ : R) (k : K) (v : RingCell R) (M : HMap K (RingCell R)) : HMap K (RingCell R) :=
  insertWith (ringMul σ) k v M

/-- Characterization of `memoryStep` as a function update replacing `M k` with `ringMul σ v (M k)`. -/
theorem memory_step_model (σ : R) (k : K) (v : RingCell R) (M : HMap K (RingCell R)) :
    memoryStep σ k v M = Function.update M k (ringMul σ v (M k)) := by
  funext x
  simp only [memoryStep, insertWith, Function.update_apply]
  split_ifs with h
  · subst h
    rfl
  · rfl

/-- Reading back at the updated key yields the merged value `ringMul σ v (M k)`. -/
theorem memory_step_get_self (σ : R) (k : K) (v : RingCell R) (M : HMap K (RingCell R)) :
    get (memoryStep σ k v M) k = ringMul σ v (M k) :=
  get_insertWith_self (ringMul σ) k v M

/-- Reading at any distinct key `x ≠ k` is unchanged. -/
theorem memory_step_get_ne (σ : R) (k : K) (v : RingCell R) (M : HMap K (RingCell R))
    {x : K} (h : x ≠ k) :
    get (memoryStep σ k v M) x = get M x :=
  get_insertWith_ne (ringMul σ) k v M h

end Nous.Model
