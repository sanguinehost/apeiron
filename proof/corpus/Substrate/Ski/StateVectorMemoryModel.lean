import Substrate.Ski.StateVectorProjection
import Substrate.Collapse

/-!
# State-Vector Memory Model (Universal Algorithm Bridge — Phase 3, write side)

@name Memory Is Just a List
@child A computer's memory — putting a value in a slot, reading it back, making a new slot — is really just adding to and changing a list. Every memory move turns out to be a simple list operation, proven to behave correctly.
@scientist The LLVM memory ops (alloca, store, load, extractvalue, insertvalue, getelementptr) are each a direct operation over the cons-list state vector; this proves the write-side and the load-after-store / commutativity / length-preservation laws the translator schedules against.
@engineer `load_eq_getElem?` is the one identification the file rests on: `load d s` IS `s[d]?`. On it, `llvm_memory_is_list_indexing` collapses six opcodes to three operations (`cons`, `List.set`, `getElem?`, GEP as `Nat.+`), and every law below is a `List` lemma: the four McCarthy heap laws are `List.getElem?_set_self` / `List.getElem?_set_ne` / `List.set_set` / `List.set_comm`, the alloca group is `List.getElem?_cons_zero` / `List.getElem?_cons_succ` / `List.set_cons_zero`. No induction over `drop` or `set` survives.

The translator (`interop/src/translator_executable.rs`) emitted
LLVM IR's memory operations (`alloca`, `load`, `store`, `extractvalue`,
`insertvalue`, `getelementptr`) as **tagged Payload strings** that the
K=4 evaluator cannot reduce. The whole Rust harness was deleted in `4556df7d`
("Delete all Rust — remove the interop harness completely", 2026-05-31), and
`src/kernel/State.s` with `f8efce70` the same day; both are named below as
history. This file discharges the foundational
contract that operationalises those ops: each one is a direct
operation over the cons-list state vector documented in
`StateVectorProjection.lean`.

`StateVectorProjection.lean` covers the **read** side
(`load`, `extractvalue`) as projection via `(s.drop d).head?`. This file
covers the **write** side and the compositional properties that the
translator's pattern-matcher relies on:

* `alloca` prepends a fresh slot (becomes depth 0).
* `store` updates the slot at a given depth via `List.set`.
* `extractvalue` IS `load` (the LLVM aggregate-extraction op is the
  same combinator as register-read at a fixed depth).
* `insertvalue` IS `store` (same identification for the write side).
* `getelementptr` composes additively: a pointer + an offset is a
  pointer to a deeper position in the state vector.

The atomic operational properties (load-after-store same/other,
store commutativity, length preservation, alloca shift) are the
contract the translator schedules against. They are proven directly
on `List α`, without `axiom`, `sorry`, or `assume`.

## Why this matters

Per `theory/RUST_LIFT_GAP_AUDIT.md` (Sprint 0 audit, 2026-05-28; that
document deleted in `a8725ac7`, 2026-05-31, so the count is unrecheckable
from this tree), the
memory-op category dominates real Rust at ≈175k references repo-wide
(`store` 51k, `getelementptr` 46k, `alloca` 32k, plus `load` and
related). Without these operational properties, no Rust beyond pure
arithmetic can lift to executable form. With them, the translator
pattern-matcher can emit cons-list operations matching what the K=4
evaluator already reduces (via existing `fst`, `snd`, and the
S/K/Iota substrate).

Foundation, not feature: per Axiom XVI, this proof must land before
the translator's Phase 2 → Phase 4 extension can be safely written.

## What is proved here vs what is read onto LLVM

* **Proved (native):** `load_eq_getElem?` — the one identification the file rests on — and, on
  it, the McCarthy heap laws (`load_store_same`, `load_store_other`,
  `store_store_same_overwrites`, `store_store_swap`), the alloca group (`load_alloca`,
  `load_alloca_shift`, `alloca_length`, `store_zero_alloca`), `store_length`,
  `gepOffset_associative`, and the bundle `llvm_memory_is_list_indexing`. Each is a `List`
  lemma through the identification, not an induction over `drop` / `set`.
* **Proved (riding):** the streaming chord — `readMemory_is_streamFold`,
  `appendData_is_streaming`, `appendOne_is_streaming`, `memoryKey_is_foldKey` and
  `memoryAppend_exact` are `Collapse.stream_exact_general`, `Collapse.foldKey_is_streamFold`
  and `Collapse.streaming_exact` instantiated at the memory list; no new induction.
* **Anchor:** that these `List` operations ARE LLVM's `alloca` / `store` / `load`
  / `extractvalue` / `insertvalue` / `getelementptr`. Those six are `def`s posited here;
  `llvm_memory_is_list_indexing` proves the posits equal `cons` / `List.set` / `getElem?` /
  `Nat.+`, not that LLVM's opcodes carry those semantics. The cut into a real machine's memory
  sits at these definitions, and nothing below reads LLVM's reference or
  `interop/src/translator_executable.rs`.
* **Anchor:** the ≈175k memory-op references in real Rust (`store` 51k,
  `getelementptr` 46k, `alloca` 32k) — a count taken from `theory/RUST_LIFT_GAP_AUDIT.md`
  (Sprint 0 audit, 2026-05-28), read in, not measured or stated below.

-/

namespace Sanguine

universe u

variable {α : Type u}

-- ═══════════════════════════════════════════════════════════════════
-- THE WRITE-SIDE OPERATIONS
-- ═══════════════════════════════════════════════════════════════════

/-- **alloca.** Prepend a fresh slot carrying value `v`; the new slot
    is at depth 0. Matches the prepend convention of `src/kernel/State.s`:
    the most-recently-written SSA value sits at the head. -/
def alloca (v : α) (s : List α) : List α := v :: s

/-- **store.** Update the slot at depth `d` to value `v`. Out-of-bounds
    writes are no-ops (consistent with Lean's `List.set` total form). -/
def store (d : Nat) (v : α) (s : List α) : List α := s.set d v

/-- **load.** Projection at depth `d`. Identical to the contract
    proven in `StateVectorProjection.lean`; restated here for closure. -/
def load (d : Nat) (s : List α) : Option α := (s.drop d).head?

/-- **extractvalue.** LLVM's aggregate-field extraction at constant
    offset `d`. Identified with `load` at that depth — the cons-list
    state vector IS the flat aggregate, so the two ops collapse. -/
def extractvalue (d : Nat) (s : List α) : Option α := load d s

/-- **insertvalue.** LLVM's aggregate-field insertion at constant offset
    `d` with value `v`. Identified with `store` — same identification
    as `extractvalue` IS `load`. -/
def insertvalue (d : Nat) (v : α) (s : List α) : List α := store d v s

/-- **getelementptr (offset form).** Pointer arithmetic in a flat
    cons-list state collapses to additive depth composition. The
    GEP at base `p` with offset `o` is the same projection as
    depth-(p+o) in the underlying state. -/
def gepOffset (p o : Nat) : Nat := p + o

-- ═══════════════════════════════════════════════════════════════════
-- THE STATE VECTOR IS AN INDEXED LIST
-- ═══════════════════════════════════════════════════════════════════

/-- **`load` IS list indexing.** Dropping `d` and taking the head is `s[d]?`
    (`List.head?_drop`). The one identification this file rests on: with it,
    `(load, store) = (getElem?, List.set)` and the four McCarthy heap axioms below are
    `List`'s own lemmas, not four inductions over `drop`/`set`. -/
theorem load_eq_getElem? (d : Nat) (s : List α) : load d s = s[d]? := List.head?_drop

-- ═══════════════════════════════════════════════════════════════════
-- ATOMIC OPERATIONAL PROPERTIES
-- ═══════════════════════════════════════════════════════════════════

/-- **alloca produces a head load.** Loading depth 0 of an alloca
    returns the allocated value. `alloca` is `cons`, so this is
    `List.getElem?_cons_zero` through the identification. -/
theorem load_alloca (v : α) (s : List α) :
    load 0 (alloca v s) = some v := by
  rw [load_eq_getElem?]; exact List.getElem?_cons_zero

/-- **alloca shifts deeper reads by one.** A load at depth `d+1` of
    an alloca'd state reads the original state at depth `d`. This is
    the operational property that lets the translator reuse SSA names
    across alloca boundaries — `List.getElem?_cons_succ`. -/
theorem load_alloca_shift (d : Nat) (v : α) (s : List α) :
    load (d + 1) (alloca v s) = load d s := by
  rw [load_eq_getElem?, load_eq_getElem?]; exact List.getElem?_cons_succ

/-- **alloca increases length by exactly one.** Required by the
    translator's depth-arithmetic when scheduling post-alloca SSA
    references. -/
theorem alloca_length (v : α) (s : List α) :
    (alloca v s).length = s.length + 1 := List.length_cons ..

/-- **store at depth d, then load at d, returns the stored value**
    (when `d` is in bounds). This is the fundamental load-after-store
    equation that pattern-matching against pure dataflow relies on. -/
theorem load_store_same (d : Nat) (v : α) (s : List α)
    (h : d < s.length) :
    load d (store d v s) = some v := by
  rw [load_eq_getElem?]; exact List.getElem?_set_self h

/-- **store at depth d preserves loads at all other depths.** Independent
    writes commute with independent reads — the operational property
    that justifies parallel SSA scheduling. -/
theorem load_store_other (d e : Nat) (v : α) (s : List α)
    (hne : d ≠ e) :
    load d (store e v s) = load d s := by
  rw [load_eq_getElem?, load_eq_getElem?]; exact List.getElem?_set_ne (Ne.symm hne)

/-- **Store preserves length.** Updating a slot doesn't extend the
    state vector. (Out-of-bounds stores are also no-ops.) -/
theorem store_length (d : Nat) (v : α) (s : List α) :
    (store d v s).length = s.length := by
  show (List.set s d v).length = s.length
  simp

/-- **Store-then-store at the same depth: the second wins.** Sequence
    `store d v ; store d w` simplifies to `store d w` — the translator's
    dead-store elimination is sound. -/
theorem store_store_same_overwrites (d : Nat) (v w : α) (s : List α) :
    store d w (store d v s) = store d w s := List.set_set v

/-- **Store-then-store at different depths commute.** Sequence
    `store d v ; store e w` equals `store e w ; store d v` when
    `d ≠ e`. Justifies the translator reordering independent writes. -/
theorem store_store_swap (d e : Nat) (v w : α) (s : List α)
    (hne : d ≠ e) :
    store d v (store e w s) = store e w (store d v s) := List.set_comm w v (Ne.symm hne)

-- ═══════════════════════════════════════════════════════════════════
-- SIX OPCODES, THREE OPERATIONS
-- ═══════════════════════════════════════════════════════════════════

/-- **The LLVM memory layer IS `List` indexing.** Six opcodes, three operations: `alloca` is
    `cons`, `store` and `insertvalue` are `List.set`, `load` and `extractvalue` are `getElem?`,
    and `getelementptr` is `Nat` addition carried under the index. There is no aggregate layer
    and no pointer layer above the state vector — "register", "field" and "pointer + offset"
    are three names for the same depth. Everything the translator's pattern-matcher schedules
    against is `List` and `Nat`, which is why the heap laws above are `List`'s own lemmas. -/
theorem llvm_memory_is_list_indexing (d p o : Nat) (v : α) (s : List α) :
    alloca v s = v :: s
    ∧ store d v s = s.set d v
    ∧ load d s = s[d]?
    ∧ extractvalue d s = s[d]?
    ∧ insertvalue d v s = s.set d v
    ∧ load (gepOffset p o) s = s[p + o]? :=
  ⟨rfl, rfl, load_eq_getElem? d s, load_eq_getElem? d s, rfl, load_eq_getElem? (p + o) s⟩

/-- **Two GEPs collapse to one.** `gepOffset (gepOffset p o1) o2` is
    `gepOffset p (o1 + o2)`. Translator pass: any chain of GEPs
    folds into a single depth — associativity of `Nat.+`, nothing else. -/
theorem gepOffset_associative (p o1 o2 : Nat) :
    gepOffset (gepOffset p o1) o2 = gepOffset p (o1 + o2) :=
  Nat.add_assoc p o1 o2

-- ═══════════════════════════════════════════════════════════════════
-- ALLOCA + STORE COMPOSITION (THE RUST INIT PATTERN)
-- ═══════════════════════════════════════════════════════════════════

/-- **alloca-then-store-at-0 is alloca with the new value.** Rust's
    common pattern `let x = ...; x = v;` lowers to alloca + store-0,
    which collapses to `alloca v s`. Translator dead-init elimination. -/
theorem store_zero_alloca (v w : α) (s : List α) :
    store 0 v (alloca w s) = alloca v s := List.set_cons_zero

-- ═══════════════════════════════════════════════════════════════════
-- THE CHORD — STATE-VECTOR MEMORY IS THE STREAMING FOLD
-- ═══════════════════════════════════════════════════════════════════
-- The memory (`List α`) read forward and folded by a reader step `f` IS
-- `Collapse.streamFold f`. Appending data is the streaming law: the existing
-- memory, folded once into the running state, is never re-touched; the new
-- datum continues from that state. So forward reading IS the fold, and an
-- append pays only the appended flaw — `stream_exact_general` exactly. When the
-- reader's step is the commutative add, it is also order-free (`streaming_exact`).

-- Collapse's streaming law lives at universe 0 (`Type`), so the chord instantiates
-- the memory item/state at `Type` here. `MemItem` is the state-vector cell type.
variable {MemItem : Type} {σ : Type}

/-- **Reading memory forward IS the streaming fold.** Folding the memory `s` with
    reader step `f` from `init` is `Collapse.streamFold f init s` definitionally —
    the memory IS the accumulator, the read IS the fold. -/
theorem readMemory_is_streamFold (f : σ → MemItem → σ) (init : σ)
    (s : List MemItem) :
    s.foldl f init = Sanguine.Collapse.streamFold f init s := rfl

/-- **Appending data IS the streaming law (exact, order-free-irrelevant).** Folding
    the memory after appending `flaw` continues from the state after `rest`: the
    rest, folded once, is never re-scanned; the new datum pays only itself. This is
    `Collapse.stream_exact_general` over the memory list — forward-only, no
    commutativity, no re-touch. The license that an incremental read of the memory
    equals a full re-scan, exactly. -/
theorem appendData_is_streaming (f : σ → MemItem → σ) (init : σ)
    (rest flaw : List MemItem) :
    Sanguine.Collapse.streamFold f init (rest ++ flaw)
      = Sanguine.Collapse.streamFold f (Sanguine.Collapse.streamFold f init rest) flaw :=
  Sanguine.Collapse.stream_exact_general f init rest flaw

/-- **A single appended datum continues the fold from the memory's state.** The
    common case of `appendData_is_streaming`: writing one new value (the flaw / the
    surprise) folds it into the running state without re-touching the rest. Forward-
    only learning — no separate metric, the appended datum is the contribution. -/
theorem appendOne_is_streaming (f : σ → MemItem → σ) (init : σ)
    (rest : List MemItem) (v : MemItem) :
    Sanguine.Collapse.streamFold f init (rest ++ [v])
      = f (Sanguine.Collapse.streamFold f init rest) v := by
  rw [Sanguine.Collapse.stream_exact_general]
  rfl

/-- **The order-free memory key IS the commutative streaming key.** When the reader
    accumulates a contribution `c` additively into a commutative monoid, reading the
    memory forward equals `Collapse.foldKey c` — so the read is order-free
    (`streaming_order_free`) and removal is exact (`streaming_exact_removal`). The
    state-vector memory's add-accumulate IS the trainer's `(B,U)` ledger. -/
theorem memoryKey_is_foldKey {κ : Type} [AddCommMonoid κ] (c : MemItem → κ)
    (s : List MemItem) :
    s.foldl (fun acc x => acc + c x) 0 = Sanguine.Collapse.foldKey c s :=
  (Sanguine.Collapse.foldKey_is_streamFold c s).symm

/-- **Memory append is exact addition of the new datum's contribution** (commutative
    reader). `key (rest ++ flaw) = key rest + key flaw`: the memory's running key,
    certified once over `rest`, adds the appended flaw's contribution exactly and
    never re-touches the rest. The forward-only-learning chord, as algebra — the new
    datum's surprise is its own contribution, added in O(flaw). -/
theorem memoryAppend_exact {κ : Type} [AddCommMonoid κ] (c : MemItem → κ)
    (rest flaw : List MemItem) :
    Sanguine.Collapse.foldKey c (rest ++ flaw)
      = Sanguine.Collapse.foldKey c rest + Sanguine.Collapse.foldKey c flaw :=
  Sanguine.Collapse.streaming_exact c rest flaw

end Sanguine
