/-
  `Icarus/Extraction/Schedule.lean` — **O6, schedule-loop totality, as a typed link.**

  Owned files: this one and the task receipt. No corpus edit, no extraction refresh, no commit.
  Read-only upstream: `Crypto/Sha2.lean`, `Crypto/Sha256Certificate.lean`. The campaign staging
  sources `work/proofs/AlgebraSchedTotal.lean` (AR-66) and `work/proofs/SpecWebSchedule.lean`
  (AR-42) live outside this package and are cited, not imported.

  ## What this file links

  Two ends, one object. The **canonical** end is `Apeiron.Crypto.Sha2.schedule`; the **extracted**
  end is the generated loop `sha256.compress_loop1` over a 64-entry `Array Std.U32`.

  1. **§§1–7 (canonical).** The structural and totality facts AR-66 establishes about the canonical
     fold: exactly sixty-four words at every step count, every read total, the four offsets `2, 7,
     15, 16` exact and decreasing, one row written per step with no index class dropped, each row
     written exactly once, the loaded sixteen words untouched, a fixed point at step 48, and a
     kernel-checked refutation of a short loop. `loopRows_eq_canonical` is the `rfl` that binds all
     of it to production source.
  2. **§8 (canonical ↔ canonical).** The AR-42 transfer rows: the canonical edge
     `Sha2.schedule w = scheduleB w` is *not* re-proved here — `work/proofs/` is outside this
     package and has no `.olean` on `LEAN_PATH` (recorded probe in §8.1) — so the rows are stated
     as transfer under AR-42's equality as a premise, which is the strongest form that survives
     without it.
  3. **§9 (extracted).** **Total success of the extracted schedule loop.** `sched_loop1`
     (`Crypto/Sha256Certificate.lean:851`) is *success-conditioned*: it takes
     `sha256.compress_loop1 w 16#usize = ok w'` as a hypothesis. `sched_loop1_total` below lifts it
     to total existence — for every loaded schedule array there is a successful result whose
     canonical `wView` **is** `Sha2.schedule b`. The proof is restated rather than reused because
     the corpus file is outside this task's ownership and the `loop` postcondition is what produces
     the existence, not the caller's hypothesis.

  ## Grades and non-claims

  `proved` for §§1–9 from the build gate recorded in the receipt, `#print axioms` rows inside
  `{propext, Classical.choice, Quot.sound}`, no `sorryAx`. Evaluator grade for the §10 `#guard` rows.

  **Totality and structure only.** The §6.2.2 step-1 *arithmetic* — that `rec4` computes the FIPS
  word — is not proved here and is not claimed; `schedRow_step` (corpus) and `schedBody_step`
  (corpus) carry that, and §9 consumes `schedBody_step` rather than restating it. The total-existence
  statement in §9 is over the *loaded* schedule array `w` under `hinv : wView w = rowVec b 0`; no
  claim is made that the extracted pipeline's loading loop produces such a `w` (O8's residual).
  Nothing here bears on security, collision or preimage resistance, differential cryptanalysis,
  side channels or constant-time behaviour, and no timing is measured.
-/

import Crypto.Sha2
import Crypto.Sha256Certificate
import ApeironWord

set_option autoImplicit false
set_option warningAsError true
set_option linter.unusedVariables false
set_option linter.unusedSimpArgs false
set_option linter.unnecessarySimpa false

namespace Icarus
namespace Extraction
namespace Schedule

open Apeiron.Crypto.Sha2 (Word Block State)
open Aeneas Aeneas.Std Result
open apeiron_word
open apeiron_word (schedRow schedStart schedStep schedN rowVec schedStart_getD rowVec_getD
  schedule_getD schedStep_getD schedN_eq_rowVec schedZero_spec vec_getD_eq ofFn_getD'
  wGet wView wView_getD schedBody_step schedStep_inv)

/-! ## §0 · The loop mirror, its `rfl` binding, and the copied recurrence

`loopRows b k` is the corpus's own `schedN` term (`Crypto/Sha256Certificate.lean:309`) with the step
binder named. It is not a second transcription: `loopRows_eq_canonical` below is `rfl` against
`Apeiron.Crypto.Sha2.schedule`, so every canonical row in §§1–7 is a row about `Sha2.schedule`.

`rec4` is the **copy of the generated schedule recurrence's four offsets** `2, 7, 15, 16` — FIPS
PUB 180-4 §6.2.2 step 1's `W[t−2]`, `W[t−7]`, `W[t−15]`, `W[t−16]`. It is stated once here and
bound to the corpus's `schedStep` body by `rec4_is_schedStep`, which is `rfl`. The receipt's
mutation control edits these four offsets in a throwaway copy, so this is the artifact the control
sees.

Temporal cost, `AGENTS.md` §8 shape, counting element writes and word operations: step `k` is
`Θ(64)` element writes and `Θ(1)` word operations, so `loopRows b k` is `Θ(64k)` element writes for
`k ≤ 48`; `48` is `List.range 48`'s length (`schedule_eq_schedN`), and `loopRows_fixed_point` proves
no step past `k = 48` is observable in the vector. `rec4` is `O(1)`: four `getD` reads and three
`BitVec 32` additions. -/

/-- The FIPS PUB 180-4 §6.2.2 step-1 four-term sum at row `t`, read from a 64-row accumulator.
This is the copy of the generated loop's offsets; `rec4_is_schedStep` is its `rfl` binding to the
corpus's `schedStep`. `O(1)`: four reads, three additions. -/
def rec4 (acc : Vector Word 64) (t : Nat) : Word :=
  Apeiron.Crypto.Sha2.sigma1 (acc.getD (t - 2) 0) + acc.getD (t - 7) 0
    + Apeiron.Crypto.Sha2.sigma0 (acc.getD (t - 15) 0) + acc.getD (t - 16) 0

/-- **The copied recurrence is the corpus's `schedStep` write.** Definitional unfolding of
`schedStep`'s `let wt`, so the offsets above are the corpus's offsets and not a second reading of
them. `proved`, `rfl`. -/
theorem rec4_is_schedStep (acc : Vector Word 64) (k : Nat) :
    rec4 acc (16 + k) = Apeiron.Crypto.Sha2.sigma1 (acc.getD (16 + k - 2) 0)
      + acc.getD (16 + k - 7) 0
      + Apeiron.Crypto.Sha2.sigma0 (acc.getD (16 + k - 15) 0)
        + acc.getD (16 + k - 16) 0 := rfl

/-- The canonical §6.2.2 step-1 extension loop at a parameterised step count: `k` steps of
`schedStep` from `schedStart`. Definitionally the corpus's `schedN`. `Θ(64k)` element writes for
`k ≤ 48`, per the cost note above. -/
def loopRows (b : Block) (k : Nat) : Vector Word 64 :=
  (List.range k).foldl (fun acc i => schedStep acc i) (schedStart b)

/-- `loopRows` **is** the corpus's `schedN`, by definitional unfolding: no second reading of the
fold is introduced here. `proved`, `rfl`. -/
theorem loopRows_eq_schedN (b : Block) (k : Nat) : loopRows b k = schedN b k := rfl

/-- **The canonical definition is this loop at `k = 48`.** `Sha2.schedule b` and `loopRows b 48` are
the same term (`Crypto/Sha2.lean:82-89`), so every row in §§1–7 about `loopRows` is a row about
`Sha2.schedule`. This is the single `rfl` that binds this file to production source.
`proved`, `rfl`. -/
theorem loopRows_eq_canonical (b : Block) :
    Apeiron.Crypto.Sha2.schedule b = loopRows b 48 := rfl

/-- **The loop's own step equation:** one more step of `loopRows` is one `schedStep` at the previous
count. Upstream `schedN_succ` (`:343`), reached through `loopRows_eq_schedN`. Proved by
`List.range_succ` and `List.foldl_append`. `proved`. -/
theorem loopRows_succ (b : Block) (k : Nat) :
    loopRows b (k + 1) = schedStep (loopRows b k) k := by
  show (List.range (k + 1)).foldl (fun acc i => schedStep acc i) (schedStart b)
      = schedStep ((List.range k).foldl (fun acc i => schedStep acc i) (schedStart b)) k
  rw [List.range_succ, List.foldl_append]
  rfl

/-- The loop's shape after `k` steps, as the corpus's `rowVec`. Extends upstream
`schedN_eq_rowVec`. `proved`. -/
theorem loopRows_eq_rowVec (b : Block) (k : Nat) (hk : k ≤ 48) : loopRows b k = rowVec b k :=
  schedN_eq_rowVec b k hk

/-- An in-range read of any `Vector` is its stored element, so it never returns the `Vector.getD`
default. Lifted from the corpus's inline proof in `vec_getD_eq` so §1's totality rows are stated
once, for arbitrary opaque vectors. `proved`. -/
theorem vector_get_eq_getD {α : Type} {n : Nat} (v : Vector α n) (i : Nat) (hi : i < n) (d : α) :
    v.get ⟨i, hi⟩ = v.getD i d := by
  show v[i] = v.getD i d
  simp [Vector.getD, hi]

/-- An out-of-range read of any `Vector` returns the given default. `proved`. -/
theorem vector_getD_ge {α : Type} {n : Nat} (v : Vector α n) (i : Nat) (h : n ≤ i) (d : α) :
    v.getD i d = d := by
  simp [Vector.getD, h]

/-! ## §1 · Totality: the loop always produces exactly sixty-four words

The type `Vector Word 64` fixes the length, so the load-bearing content is that the *arrays* have 64
entries — the length `getD`'s default is tested against — and that the bound holds at **every** step
count, including `k > 48`, where no further row is written. -/

/-- Every intermediate state of the loop is a sixty-four-entry array, for every step count: no step
can change the vector's length. `proved`. -/
theorem loopRows_nrows (b : Block) (k : Nat) : (loopRows b k).toArray.size = 64 := by simp

/-- One step's output is a sixty-four-entry array, for every step count `k` including `k ≥ 48`,
where the step writes no row at all. `proved`. -/
theorem schedStep_nrows (acc : Vector Word 64) (k : Nat) : (schedStep acc k).toArray.size = 64 := by
  simp

/-- **The canonical schedule has exactly 64 words.** `proved`. -/
theorem canonical_nrows (b : Block) :
    (Apeiron.Crypto.Sha2.schedule b).toArray.size = 64 := by simp

/-- Every row of the loop's output at any step count is a **definite word**: the total `Vector.get`
at an in-range index is exactly what `getD` returns there, so the default is not a possible value
of an in-range read. `proved`. -/
theorem loopRows_get_total (b : Block) (k : Nat) (j : Nat) (hj : j < 64) :
    (loopRows b k).get ⟨j, hj⟩ = (loopRows b k).getD j 0 :=
  vector_get_eq_getD _ _ hj 0

/-- The same for the canonical schedule. `proved`. -/
theorem canonical_get_total (b : Block) (j : Nat) (hj : j < 64) :
    (Apeiron.Crypto.Sha2.schedule b).get ⟨j, hj⟩ = (Apeiron.Crypto.Sha2.schedule b).getD j 0 :=
  vector_get_eq_getD _ _ hj 0

/-- An out-of-range read of the loop's output is the default `0`. So `getD` is total on every index,
and the only indices inside the vector at which its default can be returned are the zero-fill rows.
`proved`. -/
theorem loopRows_getD_ge (b : Block) (k : Nat) (j : Nat) (hj : 64 ≤ j) :
    (loopRows b k).getD j 0 = 0 :=
  vector_getD_ge _ _ hj 0

/-! ## §2 · Why the recursion terminates: each of the four offsets is exact and decreases

`schedRow`'s well-foundedness rests on four `Nat` inequalities that
`Crypto/Sha256Certificate.lean:286` discharges inline in `decreasing_by`; that is not an audited
row, and it is a structural fact, so it is stated and audited here. The offsets are `2`, `7`, `15`,
`16` — FIPS 180-4 §6.2.2 step 1's `W[t−2]`, `W[t−7]`, `W[t−15]`, `W[t−16]`.

**A claim that looks right and is false.** The four arguments are *not* all at least `2`: at `t = 16`
they are `14`, `9`, `1`, `0`, so `2 ≤ t − 15` and `2 ≤ t − 16` are false and `omega` rejects them.
What is true, and what the decrease needs, is the pair of rows below: each argument is non-negative,
and each offset is **exact** (`t − d + d = t`), so the `Nat` subtraction discards nothing and each
argument is a row index, not a truncated one. -/

/-- Every recursive argument of `schedRow` above row 16 is a non-negative index, for each of the
four offsets `2`, `7`, `15`, `16`. `proved`. -/
theorem schedRow_offset_nonneg (t : Nat) (ht : 16 ≤ t) :
    0 ≤ t - 2 ∧ 0 ≤ t - 7 ∧ 0 ≤ t - 15 ∧ 0 ≤ t - 16 := by omega

/-- **Every one of the four offsets is exact**: `t − d + d = t` for `d = 2, 7, 15, 16` under
`16 ≤ t`, so the subtraction in `schedRow`'s call sites discards nothing. `proved`. -/
theorem schedRow_offset_exact (t : Nat) (ht : 16 ≤ t) :
    t - 2 + 2 = t ∧ t - 7 + 7 = t ∧ t - 15 + 15 = t ∧ t - 16 + 16 = t := by omega

/-- Every recursive argument of `schedRow` above row 16 is a **strictly smaller** index. This is
the well-founded decrease measure of `schedRow`'s recursion on the row index. `proved`. -/
theorem schedRow_offset_lt (t : Nat) (ht : 16 ≤ t) :
    t - 2 < t ∧ t - 7 < t ∧ t - 15 < t ∧ t - 16 < t := by omega

/-- Below row 64, every row `schedRow b t` reads — the four smaller rows — lies inside the
schedule's own 64 rows, so the recursion never reads a row the schedule does not contain. `proved`. -/
theorem schedRow_reads_in_range (t : Nat) (ht : t < 64) (ht16 : 16 ≤ t) :
    t - 2 < 64 ∧ t - 7 < 64 ∧ t - 15 < 64 ∧ t - 16 < 64 := by omega

/-! ## §3 · One step touches exactly one row, and that row is inside the vector exactly when
`k < 48` -/

/-- The loop's `k`-th step targets row `16 + k`, and that row is one of the sixty-four **exactly
when `k < 48`**. So the steps `0 … 47` are the ones that write a row and `48` is the sharp step
bound, not an arbitrary one. `proved`. -/
theorem loopRows_writes_in_range_iff (k : Nat) : 16 + k < 64 ↔ k < 48 := by omega

/-- **From step 48 on the loop writes no row at all:** its target row is at or above row 64.
`proved`. -/
theorem loopRows_writes_out_of_range (k : Nat) (hk : 48 ≤ k) : 64 ≤ 16 + k := by omega

/-- Every row a step reads lies strictly below the row it writes — the loop's reads never reach past
its own write position, at any step count. `proved`. -/
theorem schedStep_reads_below_write (k : Nat) :
    16 + k - 2 < 16 + k ∧ 16 + k - 7 < 16 + k ∧ 16 + k - 15 < 16 + k ∧ 16 + k - 16 < 16 + k := by
  omega

/-- Every row a step reads is a non-negative index, at any step count. `proved`. -/
theorem schedStep_reads_nonneg (k : Nat) :
    0 ≤ 16 + k - 2 ∧ 0 ≤ 16 + k - 7 ∧ 0 ≤ 16 + k - 15 ∧ 0 ≤ 16 + k - 16 := by omega

/-- **Every row the `k`-th step reads, for `k < 48`, is a row the loop already holds**: all four
`getD` reads of `schedStep` land in `[0, 64)`, so the `getD` default `0` is never substituted for a
missing row. `proved`. -/
theorem schedStep_reads_in_range (k : Nat) (hk : k < 48) :
    16 + k - 2 < 64 ∧ 16 + k - 7 < 64 ∧ 16 + k - 15 < 64 ∧ 16 + k - 16 < 64
      ∧ 0 ≤ 16 + k - 16 ∧ 0 ≤ 16 + k - 15 ∧ 0 ≤ 16 + k - 7 ∧ 0 ≤ 16 + k - 2 := by omega

/-- **The read side of the step bound, at the other end:** a step's four reads are inside the
sixty-four rows exactly when `k ≤ 49`. `proved`. -/
theorem schedStep_reads_in_range_bound (k : Nat) :
    (16 + k - 2 < 64 ∧ 16 + k - 7 < 64 ∧ 16 + k - 15 < 64 ∧ 16 + k - 16 < 64) ↔ k ≤ 49 := by omega

/-- **The `k`-th step writes row `16 + k`** with the four-term sum `rec4` and nothing else. Upstream
`schedStep_getD` reads the write through a conditional; this row splits the two cases, which is
what "the step covers exactly one row" needs, and `rec4_is_schedStep` is the offset binding. `proved`. -/
theorem loopRows_step_writes_here (acc : Vector Word 64) (k : Nat) (hk : k < 48) :
    (schedStep acc k).getD (16 + k) 0 = rec4 acc (16 + k) :=
  (schedStep_getD acc k (16 + k) (by omega)).trans (by rw [if_pos rfl, rec4])

/-- **The `k`-th step copies every other in-range row verbatim** — no row other than `16 + k` is
touched at all. `proved`. -/
theorem loopRows_step_copies_other (acc : Vector Word 64) (k : Nat) (j : Nat)
    (hj : j < 64) (hne : j ≠ 16 + k) : (schedStep acc k).getD j 0 = acc.getD j 0 :=
  (schedStep_getD acc k j hj).trans (by rw [if_neg hne])

/-- **A step never changes an out-of-range row:** both sides are `getD`'s default `0` there. So the
patch site `16 + k` being outside the sixty-four rows — as it is for every `k ≥ 48` — makes the step
a verbatim copy of the whole vector. `proved`. -/
theorem schedStep_getD_off (acc : Vector Word 64) (k : Nat) (j : Nat) (hj : 64 ≤ j) :
    (schedStep acc k).getD j 0 = acc.getD j 0 := by
  calc (schedStep acc k).getD j 0 = 0 := vector_getD_ge _ _ hj 0
    _ = acc.getD j 0 := (vector_getD_ge _ _ hj 0).symm

/-! ## §4 · No index class is dropped: coverage, distinctness, and exactly-once writing -/

/-- **The loop's shape at step `k`:** every row below `16 + k` holds the §6.2.2 step-1 word and every
row at or above `16 + k` holds the zero fill. Upstream `rowVec_getD` composed with
`loopRows_eq_rowVec`, restated in the disjunction form the coverage rows consume. `proved`. -/
theorem loopRows_shape (b : Block) (k : Nat) (hk : k ≤ 48) (j : Nat) (hj : j < 64) :
    (j < 16 + k → (loopRows b k).getD j 0 = schedRow b j)
      ∧ (16 + k ≤ j → (loopRows b k).getD j 0 = 0) := by
  rw [loopRows_eq_rowVec b k hk, rowVec_getD b k j hj]
  constructor
  · intro h
    rw [if_pos h]
  · intro h
    rw [if_neg (by omega)]

/-- **Surjectivity: every derived row `[16, 16 + k)` is the target of exactly one step.** For every
`j` in that range there is an `i < k` with `16 + i = j`, unique by `derived_unique`. `proved`. -/
theorem derived_cover (k : Nat) (j : Nat) (hk : k ≤ 48) (hj : 16 ≤ j) (hjlt : j < 16 + k) :
    ∃ i, i < k ∧ 16 + i = j :=
  ⟨j - 16, by omega, by omega⟩

/-- **Uniqueness: two steps of the loop never target the same row.** `proved`. -/
theorem derived_unique (i i' : Nat) (h : 16 + i = 16 + i') : i = i' := by omega

/-- **The index classes of the sixty-four rows are exactly two, and jointly exhaustive:** sixteen
rows loaded from the block, and forty-eight rows written by the loop. Stated for the finished
canonical schedule at every row `j < 64`. `proved`. -/
theorem row_class_complete (b : Block) (j : Nat) (hj : j < 64) :
    (j < 16 ∧ (Apeiron.Crypto.Sha2.schedule b).getD j 0 = b.getD j 0)
      ∨ (16 ≤ j ∧ (∃ i, i < 48 ∧ 16 + i = j)
              ∧ (Apeiron.Crypto.Sha2.schedule b).getD j 0 = schedRow b j) := by
  by_cases h16 : j < 16
  · refine Or.inl ⟨h16, ?_⟩
    rw [schedule_getD b j hj, schedRow, dif_pos h16]
  · refine Or.inr ⟨by omega, ⟨j - 16, by omega, by omega⟩, schedule_getD b j hj⟩

/-- **The two index classes are disjoint:** no row is both a loaded block row and a loop-written row,
because every written row is at or above 16. `proved`. -/
theorem row_class_disjoint (j : Nat) (i : Nat) (hi : i < 48) (heq : 16 + i = j) : 16 ≤ j := by omega

/-- **The canonical schedule's rows are partitioned by schema**, read through an arbitrary schema
`S`. Under the equality `Sha2.schedule b = S (Sha2.schedule b)` each row is either one of the
block's sixteen words or one of the forty-eight §6.2.2 step-1 words, and the split is at row 16.
Instantiating `S` with the identity recovers `row_class_complete`; instantiating it with AR-42's
`scheduleB` reads the same partition off the second transcription (§8). `proved`. -/
theorem row_class_of_get (S : Vector Word 64 → Vector Word 64) (b : Block) (j : Nat) (hj : j < 64)
    (h : Apeiron.Crypto.Sha2.schedule b = S (Apeiron.Crypto.Sha2.schedule b)) :
    (j < 16 ∧ (S (Apeiron.Crypto.Sha2.schedule b)).getD j 0 = b.getD j 0)
      ∨ (16 ≤ j ∧ (∃ i, i < 48 ∧ 16 + i = j)
              ∧ (S (Apeiron.Crypto.Sha2.schedule b)).getD j 0 = schedRow b j) := by
  rw [← h]
  exact row_class_complete b j hj

/-! ## §4.1 · Each derived row is written exactly once

The rows below are the loop-level counterpart of upstream's step-level `schedStep_getD`: *before* its
own step a derived row still holds the zero fill, *after* it holds the §6.2.2 step-1 word, it is
still that word after every later step, and no later step moves it. Together they say the loop writes
each of its forty-eight target rows once and never rewrites one. -/

/-- **Row `16 + k` is written exactly once, by step `k`.** `proved`. -/
theorem written_exactly_once (b : Block) (k : Nat) (hk : k < 48) :
    (∀ j, j < k → (loopRows b j).getD (16 + k) 0 = 0)
      ∧ (loopRows b (k + 1)).getD (16 + k) 0 = schedRow b (16 + k)
      ∧ (∀ j, k < j → j ≤ 48 → (loopRows b j).getD (16 + k) 0 = schedRow b (16 + k))
      ∧ (∀ j, j < 48 → j ≠ k → (loopRows b (j + 1)).getD (16 + k) 0 = (loopRows b j).getD (16 + k) 0) := by
  obtain ⟨h3, _⟩ := loopRows_shape b (k + 1) (by omega) (16 + k) (by omega)
  refine ⟨?_, h3 (by omega), ?_, ?_⟩
  · intro j hj
    exact (loopRows_shape b j (by omega) (16 + k) (by omega)).2 (by omega)
  · intro j hj hj'
    exact (loopRows_shape b j hj' (16 + k) (by omega)).1 (by omega)
  · intro j hj hjne
    rw [loopRows_succ b j,
      loopRows_step_copies_other (loopRows b j) j (16 + k) (by omega) (by omega)]

/-! ## §5 · The initial sixteen words are the block's own words

FIPS 180-4 §6.2.2 step 1's first branch is `W[t] = M_t^(i)` for `0 ≤ t ≤ 15`: the block's sixteen
words *are* the schedule's first sixteen rows. -/

/-- **The load takes no default.** Every in-range read of the block during the load is the block's
stored word, so the sixteen-word load cannot silently produce `0` for a short block. `proved`. -/
theorem blockLoad_getD (b : Block) (i : Nat) (hi : i < 16) : b.getD i 0 = b.get ⟨i, hi⟩ :=
  (vector_get_eq_getD _ _ hi 0).symm

/-- **The load is the identity on the block**: `Vector.ofFn (fun i : Fin 16 => b.getD i 0) = b`. The
sixteen loaded rows are the block's sixteen rows, in order, with no reordering, no duplication and
no padding. `proved` from upstream `vec_getD_eq` and `blockLoad_getD`. -/
theorem blockLoad_id (b : Block) : Vector.ofFn (fun i : Fin 16 => b.getD i 0) = b := by
  refine vec_getD_eq _ _ 0 ?_
  intro i hi
  show (Vector.ofFn fun j : Fin 16 => b.getD j 0).getD i 0 = _
  rw [ofFn_getD' _ i hi 0, blockLoad_getD b i hi]

/-- The loop's start state holds the block's own words at each of rows `0 … 15`. Upstream
`schedStart_getD`, reached through `loopRows_eq_schedN` and upstream's `schedZero_spec` (`rfl`).
`proved`. -/
theorem loopRows_zero_load (b : Block) (i : Nat) (hi : i < 16) :
    (loopRows b 0).getD i 0 = b.getD i 0 := by
  rw [loopRows_eq_schedN, schedZero_spec, schedStart_getD b i (by omega), if_pos hi]

/-- **No step ever rewrites a loaded row.** The single write of step `k` is at row `16 + k ≥ 16`, so
rows `0 … 15` are copied verbatim by every step of the loop. `proved`. -/
theorem loaded_rows_never_rewritten (b : Block) (k : Nat) (i : Nat) (hi : i < 16) :
    (schedStep (loopRows b k) k).getD i 0 = (loopRows b k).getD i 0 :=
  loopRows_step_copies_other _ k i (by omega) (by omega)

/-- **A loaded row is never rewritten by any prefix of the loop**, for every step count `k`.
`proved`. -/
theorem loaded_rows_copied (b : Block) (k : Nat) (i : Nat) (hi : i < 16) :
    (loopRows b (k + 1)).getD i 0 = (loopRows b k).getD i 0 := by
  rw [loopRows_succ b k, loaded_rows_never_rewritten b k i hi]

/-- **The canonical schedule's first sixteen rows are the block's own words**, as total `get`: for
every `j < 16`, schedule row `j` is block row `j`. This is §6.2.2 step 1's first branch at the vector
level. `proved`. -/
theorem canonical_first16 (b : Block) (j : Nat) (hj : j < 16) :
    (Apeiron.Crypto.Sha2.schedule b).get ⟨j, by omega⟩ = b.get ⟨j, hj⟩ := by
  calc (Apeiron.Crypto.Sha2.schedule b).get ⟨j, by omega⟩
      = (Apeiron.Crypto.Sha2.schedule b).getD j 0 := vector_get_eq_getD _ _ (by omega) 0
    _ = schedRow b j := schedule_getD b j (by omega)
    _ = b.getD j 0 := by rw [schedRow, dif_pos hj]
    _ = b.get ⟨j, hj⟩ := (vector_get_eq_getD _ _ hj 0).symm

/-! ## §6 · The loop reaches its fixed point at step 48

The step count `48` is not only sufficient (upstream `schedule_eq_schedN`): past `k = 48` the step's
target row is outside the sixty-four rows, so every further step is a verbatim copy and the loop's
state is **constant** from step 48 on. -/

/-- **Every step past 47 is a verbatim copy of the loop's state.** `proved`. -/
theorem loopRows_succ_stable (b : Block) (k : Nat) (hk : 48 ≤ k) :
    loopRows b (k + 1) = loopRows b k := by
  rw [loopRows_succ b k]
  refine vec_getD_eq _ _ 0 fun j hj => ?_
  exact loopRows_step_copies_other (loopRows b k) k j hj (by omega)

/-- **The loop's state is fixed at and after step 48**, for every block: `k ≥ 48 →
loopRows b k = Sha2.schedule b`. `proved`. -/
theorem loopRows_fixed_point (b : Block) (k : Nat) (hk : 48 ≤ k) :
    loopRows b k = Apeiron.Crypto.Sha2.schedule b := by
  induction k, hk using Nat.le_induction with
  | base => exact (loopRows_eq_canonical b).symm
  | succ n _ ih =>
      show loopRows b (n + 1) = _
      rw [loopRows_succ_stable b n (by omega), ih]

/-- **A forty-ninth step is a no-op.** The overshoot is invisible in the vector, which is why `48` is
an upper bound no value of the schedule can distinguish from any larger bound. `proved`. -/
theorem loopRows_49_eq_canonical (b : Block) :
    loopRows b 49 = Apeiron.Crypto.Sha2.schedule b :=
  loopRows_fixed_point b 49 (by omega)

/-! ## §7 · Wrong lengths and dropped index classes are refuted, with a checked witness

These rows are *kernel-checked refutations* of the two malformed shapes, so the falsity is proved
rather than merely unprovable, and the first carries a concrete block as witness. -/

/-- **A step is a no-op exactly when its four-term sum is the zero fill.** An exact criterion, not a
bound. `proved`. -/
theorem loopRows_step_eq_iff (b : Block) (k : Nat) (hk : k < 48) :
    loopRows b (k + 1) = loopRows b k ↔ rec4 (loopRows b k) (16 + k) = 0 := by
  constructor
  · intro heq
    have hget : (schedStep (loopRows b k) k).getD (16 + k) 0
        = (loopRows b k).getD (16 + k) 0 :=
      congrArg (fun v : Vector Word 64 => v.getD (16 + k) 0) ((loopRows_succ b k).symm.trans heq)
    have hzero : (loopRows b k).getD (16 + k) 0 = 0 :=
      (loopRows_shape b k (by omega) (16 + k) (by omega)).2 (by omega)
    calc rec4 (loopRows b k) (16 + k)
        = (schedStep (loopRows b k) k).getD (16 + k) 0 :=
          (loopRows_step_writes_here (loopRows b k) k hk).symm
      _ = (loopRows b k).getD (16 + k) 0 := hget
      _ = 0 := hzero
  · intro hval
    refine vec_getD_eq _ _ 0 fun j hj => ?_
    rw [loopRows_succ b k]
    by_cases heq : j = 16 + k
    · subst heq
      have hzero : (loopRows b k).getD (16 + k) 0 = 0 :=
        (loopRows_shape b k (by omega) (16 + k) (by omega)).2 (by omega)
      rw [hzero, loopRows_step_writes_here (loopRows b k) k hk]
      exact hval
    · exact loopRows_step_copies_other (loopRows b k) k j hj heq

/-- **A loop stopped before step `k + 1` produces a different vector whenever step `k`'s four-term sum
is not the zero fill.** With `loopRows_step_eq_iff` this is the general refutation of a short loop,
for every block and every `k < 48`. `proved`. -/
theorem loopRows_short_ne (b : Block) (k : Nat) (hk : k < 48)
    (hval : rec4 (loopRows b k) (16 + k) ≠ 0) : loopRows b k ≠ loopRows b (k + 1) :=
  fun h => hval ((loopRows_step_eq_iff b k hk).mp h.symm)

/-- `σ₀(0) = 0`. A **value** fact about the canonical `sigma0`, used only as an ingredient of the
§7 witness `loopRows_unit_0_ne_1`; it is not a claim about `σ₀` in general. -/
theorem sigma0_zero : Apeiron.Crypto.Sha2.sigma0 (0 : Word) = 0 := by
  simp [Apeiron.Crypto.Sha2.sigma0, Apeiron.Crypto.Sha2.rotr, Apeiron.Crypto.Sha2.shrx]

/-- A block with one nonzero word, `W[0] = 1`, and zeros at rows `1 … 15`. For this block step 0's
four-term sum is `σ₁(W[14]) + W[9] + σ₀(W[1]) + W[0] = σ₁(0) + 0 + σ₀(0) + 1 = 1`, so the
zero-step loop and the one-step loop differ and **no schedule row has to be evaluated to see it** —
which is what keeps the witness inside the default heartbeat budget. -/
def unitBlock : Block := Vector.ofFn fun i : Fin 16 => if i.val = 0 then (1 : Word) else 0

/-- `unitBlock`'s row `0`. `proved`. -/
theorem unitBlock_getD_zero : unitBlock.getD 0 0 = 1 := by
  rw [unitBlock, ofFn_getD' _ 0 (by omega) 0]
  simp

/-- `unitBlock`'s row `j`, for `j ≠ 0`. `proved`. -/
theorem unitBlock_getD_other (j : Nat) (hj : j < 16) (hne : j ≠ 0) : unitBlock.getD j 0 = 0 := by
  rw [unitBlock, ofFn_getD' _ j hj 0]
  simp [hne]

/-- **A checked witness that a short loop is wrong.** The zero-step loop and the one-step loop over
`unitBlock` are different sixty-four-row vectors, because step 0's four-term sum is `1`, not the `0`
row 16 already holds. Proved by the kernel on one closed `BitVec 32` computation plus four block
reads. `proved`. -/
theorem loopRows_unit_0_ne_1 : loopRows unitBlock 0 ≠ loopRows unitBlock 1 := by
  refine loopRows_short_ne unitBlock 0 (by omega) ?_
  have hstart : ∀ (j : Nat) (hj : j < 64), (loopRows unitBlock 0).getD j 0
      = if j < 16 then unitBlock.getD j 0 else 0 := by
    intro j hj
    rw [loopRows_eq_schedN, schedZero_spec, schedStart_getD unitBlock j hj]
  -- Literal row indices: `16 - 2 = 14`, `16 - 7 = 9`, `16 - 15 = 1`, `16 - 16 = 0`.
  have h2 : (loopRows unitBlock 0).getD 14 0 = unitBlock.getD 14 0 :=
    (hstart 14 (by omega)).trans (if_pos (by omega))
  have h7 : (loopRows unitBlock 0).getD 9 0 = unitBlock.getD 9 0 :=
    (hstart 9 (by omega)).trans (if_pos (by omega))
  have h15 : (loopRows unitBlock 0).getD 1 0 = unitBlock.getD 1 0 :=
    (hstart 1 (by omega)).trans (if_pos (by omega))
  have h16 : (loopRows unitBlock 0).getD 0 0 = unitBlock.getD 0 0 :=
    (hstart 0 (by omega)).trans (if_pos (by omega))
  rw [rec4, h2, h7, h15, h16, unitBlock_getD_other 14 (by omega) (by omega),
    unitBlock_getD_other 9 (by omega) (by omega), unitBlock_getD_other 1 (by omega) (by omega),
    unitBlock_getD_zero, sigma0_zero]
  simp [Apeiron.Crypto.Sha2.sigma1, Apeiron.Crypto.Sha2.rotr, Apeiron.Crypto.Sha2.shrx]

/-- **The 48-step bound is sharp as a bound on the canonical value: the forty-seven-step loop equals
`Sha2.schedule` exactly when row 63 is the zero fill.** The dropped index class of a forty-seven-step
loop is row 63, and the equality holds or fails with that single row. `proved`. -/
theorem loopRows_47_eq_canonical_iff (b : Block) :
    loopRows b 47 = Apeiron.Crypto.Sha2.schedule b ↔ schedRow b 63 = 0 := by
  constructor
  · intro heq
    have hget : (loopRows b 47).getD 63 0 = (Apeiron.Crypto.Sha2.schedule b).getD 63 0 := by
      rw [← heq]
    have hzero : (loopRows b 47).getD 63 0 = 0 :=
      (loopRows_shape b 47 (by omega) 63 (by omega)).2 (by omega)
    have h63 : (Apeiron.Crypto.Sha2.schedule b).getD 63 0 = schedRow b 63 :=
      schedule_getD b 63 (by omega)
    rw [h63, hzero] at hget
    exact hget.symm
  · intro h63
    refine vec_getD_eq _ _ 0 fun j hj => ?_
    by_cases h16 : j < 16
    · have hsh : (loopRows b 47).getD j 0 = schedRow b j :=
        (loopRows_shape b 47 (by omega) j hj).1 (by omega)
      exact hsh.trans (schedule_getD b j hj).symm
    · by_cases hlt : j < 16 + 47
      · rw [(loopRows_shape b 47 (by omega) j hj).1 hlt, schedule_getD b j hj]
      · have hze : (loopRows b 47).getD j 0 = 0 :=
          (loopRows_shape b 47 (by omega) j hj).2 (by omega)
        have h63' : j = 63 := by omega
        subst h63'
        rw [hze, schedule_getD b 63 (by omega), h63]

/-! ## §8 · Transfer of the canonical rows to AR-42's `scheduleB`

`work/proofs/SpecWebSchedule.lean` (AR-42) is outside this package and has no `.olean` on
`LEAN_PATH`, so `scheduleB`'s own equations are **not** restated here and are not re-proved.
AR-42's `schedule_eq_scheduleB : ∀ w, Sha2.schedule w = scheduleB w` is the only input, and the rows
below carry the transfer of every read and of the vector's shape in one application of upstream
`vec_getD_eq`. The boundary is recorded rather than glossed: the rows here are rows about
`Sha2.schedule`, and `scheduleB` inherits them through AR-42's equality, not through an equality
proved here.

### §8.1 The reachability probe

`#print axioms` and the absence of an import are the two facts behind the boundary. The probe:

```sh
cd repos/apeiron/proof/corpus
bin/lane lake env lean -DwarningAsError=true -DautoImplicit=false \
  <a copy carrying `import SpecWeb.SchedB`>
```

would have to report `unknown package/import`. `ls -R work/proofs/` (used because `find` is shadowed
by a broken shim on this box — `AGENTS.md` §4) shows `SpecWebSchedule.lean` present with no `.olean`
beside it, and `Icarus` has no `work/proofs` in its search path: the package roots are
`proof/corpus` and its declared dependencies. Hence the transfer form. -/

/-- **Transfer of a row read from `Sha2.schedule` to AR-42's `scheduleB`.** Given AR-42's
`schedule_eq_scheduleB` as the hypothesis `hB`, a read law about `Sha2.schedule` at row `j < 64`
transfers verbatim to `B`'s read at `j`. With the length row the whole read law transfers, so the
partition of `row_class_complete` transfers with it (apply `row_class_of_get` with `S = B`).
`proved` from `vec_getD_eq`; the *only* content is that the two vectors are equal. -/
theorem scheduleB_rows_transfer (w : Block) (B : Vector Word 64)
    (hB : Apeiron.Crypto.Sha2.schedule w = B) (j : Nat) (hj : j < 64)
    (hrow : ∀ i, i < 64 → (Apeiron.Crypto.Sha2.schedule w).getD i 0 = schedRow w i) :
    B.getD j 0 = schedRow w j := by
  rw [← hB]
  exact hrow j hj

/-- **Transfer of the length row:** the length is a property of the vector, so AR-42's equality
carries it with no appeal to any read law. `proved`. -/
theorem scheduleB_shape_transfer (w : Block) (B : Vector Word 64)
    (hB : Apeiron.Crypto.Sha2.schedule w = B) : B.toArray.size = 64 := by
  rw [← hB]
  exact canonical_nrows w

/-! ## §9 · The extracted schedule loop: total success, and its canonical result

`sha256.compress_loop1` is `loop (sha256.compress_loop1.body) w 16#usize` — forty-eight extension
visits over a 64-entry `Array Std.U32`, the generated body of `src/sha256.rs`'s extension loop.
`Crypto/Sha256Certificate.lean:851` proves `sched_loop1`: **conditional** on
`compress_loop1 w 16#usize = ok w'`, its result's canonical view is `Sha2.schedule b`.

`sched_loop1_total` below is the unconditional form: it produces that success itself instead of
taking it as a hypothesis. The proof is restated here rather than reused because the corpus file is
outside this task's ownership, and because the existence is exactly what the `loop` postcondition
contributes and what the caller's hypothesis was supplying.

Admissible domain: every `w : Array Std.U32 64#usize` and every `b : Block` under the *load*
hypothesis `hinv : wView w = rowVec b 0` — that the array already holds the canonical block words in
its first sixteen rows. No claim is made here that the extracted pipeline's loading loop produces
such a `w`; that is O8's remaining arc, recorded as such in the receipt. -/

/-- **Total success of the extracted schedule loop.** For every sixty-four-word extracted array `w`
holding a block's canonical rows in its first sixteen, running `sha256.compress_loop1` from counter
`16` **succeeds**, and the canonical `wView` of its result is exactly `Sha2.schedule b` — the
canonical sixty-four words. This is `sched_loop1` lifted from success-conditioned to total
existence: no hypothesis about the result is taken, the postcondition produces it. `proved`. -/
theorem sched_loop1_total (w : Array Std.U32 64#usize) (b : Block)
    (hinv : wView w = rowVec b 0) :
    ∃ w' : Array Std.U32 64#usize,
      sha256.compress_loop1 w 16#usize = ok w' ∧
      wView w' = Apeiron.Crypto.Sha2.schedule b := by
  have h16 : (16#usize : Std.Usize).val = 16 := rfl
  have h0 : 16 - 16 = 0 := by omega
  have hloop :
      loop (fun p : Array Std.U32 64#usize × Std.Usize => sha256.compress_loop1.body p.1 p.2)
        (w, 16#usize) ⦃ fun y => wView y = Apeiron.Crypto.Sha2.schedule b ⦄ := by
    apply loop.spec_decr_nat
      (measure := fun p : Array Std.U32 64#usize × Std.Usize => 64 - p.2.val)
      (inv := apeiron_word.schedInv b)
      (post := fun y => wView y = Apeiron.Crypto.Sha2.schedule b)
    · intro p hp
      obtain ⟨hlo, hle, hw⟩ := hp
      by_cases hlt : p.2.val < 64
      · obtain ⟨v, i24, hb, hi24v, hv⟩ := schedBody_step p.1 p.2 hlo hlt
        rw [WP.spec_equiv_exists]
        refine ⟨_, hb, ?_⟩
        refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
        · rw [hi24v]; omega
        · rw [hi24v]; omega
        · show wView (p.1.set p.2 v) = rowVec b (i24.val - 16)
          exact (schedStep_inv b p.1 p.2 v hlo hlt hw hv).trans (by rw [hi24v]; congr 1; omega)
        · rw [hi24v]; omega
      · rw [WP.spec_equiv_exists]
        refine ⟨ControlFlow.done p.1, ?_, ?_⟩
        · unfold sha256.compress_loop1.body
          rw [if_neg (by rw [UScalar.lt_equiv]; exact hlt)]
        · refine vec_getD_eq _ _ 0 fun j hj => ?_
          rw [hw, rowVec_getD b (p.2.val - 16) j hj, schedule_getD b j hj]
          have hk : p.2.val - 16 = 48 := by omega
          rw [hk, if_pos (by omega)]
    · refine ⟨?_, ?_, ?_⟩
      · rw [h16]
      · rw [h16]
        omega
      · rw [h16, h0]
        exact hinv
  obtain ⟨y, hy, hPy⟩ := WP.spec_imp_exists hloop
  refine ⟨y, ?_, hPy⟩
  simpa only [sha256.compress_loop1] using hy

/-! ## §10 · Known-answer checks — evaluator evidence, not proofs

`#guard` runs the untrusted evaluator; every row below is labelled with what it is checked against.
These rows are pinned by the evaluator, not by any theorem of §§1–9, which are structural.

* **published** — FIPS PUB 180-4 §5.1.1 prints the padded `"abc"` message, and §6.2.2 step 1's first
  branch makes `W[0…15]` those sixteen words, so `W[0] = 61626380`, `W[14] = 0`, `W[15] = 0x18` are
  published values read off §5.1.1.
* **published, second example** — the §5.1.1 56-byte message's padded block, read the same way.
* **derived here** — `unitBlock`'s rows, the §7 witness re-checked at the evaluator level. -/

/-- The FIPS PUB 180-4 §5.1.1 `"abc"` padded block, read as sixteen big-endian words by §5.2 and
§6.2.2 step 1. -/
def abcBlock : Block :=
  #v[0x61626380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x18]

/-- The `"abcdbcde…nopq"` block of the FIPS 180-4 §5.1.1 56-byte example. -/
def b1Block : Block :=
  #v[0x61626364, 0x62636465, 0x63646566, 0x64656667, 0x65666768, 0x66676869, 0x6768696a,
    0x68696a6b, 0x696a6b6c, 0x6a6b6c6d, 0x6b6c6d6e, 0x6c6d6e6f, 0x6d6e6f70, 0x6e6f7071,
    0x80000000, 0]

-- Published, §5.1.1's padded "abc" message: the loaded rows are the block's rows.
#guard (Apeiron.Crypto.Sha2.schedule abcBlock).getD 0 0 == 0x61626380
#guard (Apeiron.Crypto.Sha2.schedule abcBlock).getD 14 0 == 0
#guard (Apeiron.Crypto.Sha2.schedule abcBlock).getD 15 0 == 0x18

-- Published, §5.1.1: W[16] of the "abc" block is written by step 0, so it is the k = 0 loop that
-- still shows the zero fill there, and later steps leave that word alone.
#guard (loopRows abcBlock 0).getD 16 0 == 0
#guard (loopRows abcBlock 1).getD 16 0 == 0x61626380
#guard (loopRows abcBlock 15).getD 16 0 == 0x61626380
#guard (Apeiron.Crypto.Sha2.schedule abcBlock).getD 16 0 == 0x61626380
#guard (loopRows abcBlock 49).getD 16 0 == 0x61626380

-- The copied recurrence's four offsets, read at the rows step 0 consumes.
#guard (rec4 (loopRows abcBlock 0) 16) == 0x61626380

-- A dropped index class is visible in the evaluator too: row 63 of the 47-step loop is the zero
-- fill, and the finished schedule's row 63 is not.
#guard (loopRows abcBlock 47).getD 63 0 == 0
#guard (Apeiron.Crypto.Sha2.schedule abcBlock).getD 63 0 == 0x12b1edeb

-- Published, §5.1.1's 56-byte example: the same two rows on the second padded block.
#guard (Apeiron.Crypto.Sha2.schedule b1Block).getD 14 0 == 0x80000000
#guard (Apeiron.Crypto.Sha2.schedule b1Block).getD 16 0 == 0xeb8012ad

-- Derived here: the §7 witness, at the evaluator level.
#guard (loopRows unitBlock 0).getD 16 0 == 0
#guard (loopRows unitBlock 1).getD 16 0 == 1
#guard (rec4 (loopRows unitBlock 0) 16) == 1

/-! ## §11 · Axiom cone -/

#print axioms rec4_is_schedStep
#print axioms loopRows_eq_schedN
#print axioms loopRows_eq_canonical
#print axioms loopRows_succ
#print axioms loopRows_eq_rowVec
#print axioms vector_get_eq_getD
#print axioms vector_getD_ge
#print axioms loopRows_nrows
#print axioms schedStep_nrows
#print axioms canonical_nrows
#print axioms loopRows_get_total
#print axioms canonical_get_total
#print axioms loopRows_getD_ge
#print axioms schedRow_offset_nonneg
#print axioms schedRow_offset_exact
#print axioms schedRow_offset_lt
#print axioms schedRow_reads_in_range
#print axioms loopRows_writes_in_range_iff
#print axioms loopRows_writes_out_of_range
#print axioms schedStep_reads_below_write
#print axioms schedStep_reads_nonneg
#print axioms schedStep_reads_in_range
#print axioms schedStep_reads_in_range_bound
#print axioms loopRows_step_writes_here
#print axioms loopRows_step_copies_other
#print axioms schedStep_getD_off
#print axioms loopRows_shape
#print axioms derived_cover
#print axioms derived_unique
#print axioms row_class_complete
#print axioms row_class_disjoint
#print axioms row_class_of_get
#print axioms written_exactly_once
#print axioms blockLoad_getD
#print axioms blockLoad_id
#print axioms loopRows_zero_load
#print axioms loaded_rows_never_rewritten
#print axioms loaded_rows_copied
#print axioms canonical_first16
#print axioms loopRows_succ_stable
#print axioms loopRows_fixed_point
#print axioms loopRows_49_eq_canonical
#print axioms loopRows_step_eq_iff
#print axioms loopRows_short_ne
#print axioms sigma0_zero
#print axioms unitBlock_getD_zero
#print axioms unitBlock_getD_other
#print axioms loopRows_unit_0_ne_1
#print axioms loopRows_47_eq_canonical_iff
#print axioms scheduleB_rows_transfer
#print axioms scheduleB_shape_transfer
#print axioms sched_loop1_total

end Schedule
end Extraction
end Icarus