import ApeironWord

/-
  Cost model for the by-value builder receiver (`Kb::iff`, `Kb::loop_u`,
  `src/gpu/nn/mod.rs`).

  The extraction fix moves the builder out of the borrowed receiver with
  `core.mem.replace`, applies the closure by value and writes the result back.
  The rewrite changes the closure type, not the emitted instruction list: both
  plumbing forms append the same ops and allocate the same ids, so the rewrite
  is cost-neutral — one builder move per control-flow node, no instruction
  added, dropped or reordered. Model only; no hardware timing is claimed.
-/
namespace Perf

/-- The instruction kinds `Kb` appends; only identity and order matter here. -/
inductive KbOp where
  | selectionMerge
  | branchConditional
  | label
  | branch
  deriving DecidableEq, Repr

/-- Builder state: the appended ops and the next free id. -/
structure KbState where
  ops : List KbOp
  next : Nat
  deriving Repr

/-- `Kb::ins`: append one instruction. -/
def KbState.emit (s : KbState) (o : KbOp) : KbState :=
  { s with ops := s.ops ++ [o] }

/-- `Kb::id`: allocate one id. -/
def KbState.alloc (s : KbState) : Nat × KbState :=
  (s.next, { s with next := s.next + 1 })

/-- `core.mem.replace(self, Kb::default())`: the builder moves out and a
    default takes its place; the write-back `*self = k` discards the default. -/
def KbState.replace (s : KbState) : KbState × KbState :=
  (s, ⟨[], 0⟩)

/-- The borrowed-receiver form: `then(self)` mutates the borrowed builder. -/
def iffRef (s : KbState) (closureOps : List KbOp) : KbState :=
  let (_, s) := s.alloc
  let (_, s) := s.alloc
  let s := s.emit .selectionMerge
  let s := s.emit .branchConditional
  let s := s.emit .label
  let s := { s with ops := s.ops ++ closureOps }
  let s := s.emit .branch
  s.emit .label

/-- The by-value form: move out, apply the closure, write the result back. -/
def iffVal (s : KbState) (closureOps : List KbOp) : KbState :=
  let (_, s) := s.alloc
  let (_, s) := s.alloc
  let s := s.emit .selectionMerge
  let s := s.emit .branchConditional
  let s := s.emit .label
  let (moved, _default) := s.replace
  let moved := { moved with ops := moved.ops ++ closureOps }
  let s := moved
  let s := s.emit .branch
  s.emit .label

/-- The receiver rewrite appends the same ops and allocates the same ids. -/
theorem iff_val_eq_ref (s : KbState) (closureOps : List KbOp) :
    iffVal s closureOps = iffRef s closureOps := rfl

-- Non-vacuity: a closure emitting one op yields the five builder ops plus it,
-- and the two ids the branch pair allocates.
example :
    (iffVal ⟨[], 0⟩ [.selectionMerge]).ops.length = 6 ∧
    (iffVal ⟨[], 0⟩ [.selectionMerge]).next = 2 := by
  decide

end Perf
