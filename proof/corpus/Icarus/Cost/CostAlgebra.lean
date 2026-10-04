import Lean

set_option autoImplicit false
set_option warningAsError true

/-
SCOPE: proof/resource/CostAlgebra.lean, new. A composition/dominance algebra over per-primitive
resource records: the seven resources named in docs/epics/resource-guided-evolution.md
Specification "Resource semantics", a componentwise dominance order, the composition of a
sequence of dependent stages and of a graph of dependency layers, the strict `Nat` improvement
the candidate certificate compares, and two nonvacuous schedule witnesses of one computation.
Complete when the module kernel-checks with every headline audited and three seeded composition
defects reject.
REGISTER: algebra/refinement. No Rust source, no extraction, no inference implementation, no
canonical ring correspondence, no hardware timing.
CORPUS: no match. grep for `structure Cost`/`def Cost`/`resourceCost`/`criticalPath` over
/home/socol/Workspace/sanguine-apeiron/proof/Apeiron/, /home/socol/Workspace/apeiron/proof and
/home/socol/Workspace/apeiron/docs returned no file. Native grade of this module: proved.
REFERENCE: /home/socol/Workspace/apeiron/proof/DispatchAdmission.lean (the `Admission` shape:
`improves : ... afterCost x < beforeCost x`, the strict `Nat` cost this module's
`registeredCost` supplies) and
/home/socol/Workspace/apeiron/proof/resource/ActScaleResource.lean (per-primitive resource
accounting style: one actual body, its loop rank and its reachability witnesses).
GATE: python3 proof/resource/CostAlgebraGate.py --check --controls, cwd repository root, under
flock /tmp/opencode/apeiron-lane.lock and systemd-run MemoryMax=12G MemorySwapMax=0 with
LEAN_NUM_THREADS=1. Observed red before this file existed: `test -f proof/resource/CostAlgebra.lean`
exits 1. Budget 540 s for the whole invocation including the three controls.
GRADE: proved for the algebra, dominance and the two witnesses, under the closed logical cone
{propext, Classical.choice, Quot.sound} checked by `#resource_audit`. The per-stage counts are
declared example inputs of this module, grade `model`: they are not a measurement, and no
hardware speed follows. Unmeasured and unclaimed: latency, bandwidth, device dispatch cost,
compiler or codegen cost, and the cost of any specific kernel.
-/

namespace CostAlgebra

/-- The registered resource record of one primitive or one schedule. Each field counts one
resource named in the epic's resource semantics, in that field's own unit; a field never mixes two
resources, so a regression in one stays visible instead of being traded away.

- `ops`: arithmetic operations, one multiply-accumulate or scale unit per operation.
- `pack`: packing and copy bytes moved between layouts.
- `alloc`: buffer allocations.
- `xfer`: bytes uploaded to or read back from the device.
- `dispatch`: kernel dispatches and queue drains.
- `peak`: peak-live storage bytes held at one instant.
- `path`: dependency critical path, the longest chain of dependent stages.
-/
structure Cost where
  ops : Nat
  pack : Nat
  alloc : Nat
  xfer : Nat
  dispatch : Nat
  peak : Nat
  path : Nat
  deriving DecidableEq, Repr

/-- The record of doing nothing. -/
def zero : Cost := ⟨0, 0, 0, 0, 0, 0, 0⟩

/-- Dominance: a candidate dominates the baseline when no registered resource grows. The order is
componentwise on purpose; a single weighted sum hides a regression (`scalar_not_sufficient`). -/
def costLe (a b : Cost) : Prop :=
  a.ops ≤ b.ops ∧ a.pack ≤ b.pack ∧ a.alloc ≤ b.alloc ∧ a.xfer ≤ b.xfer ∧
    a.dispatch ≤ b.dispatch ∧ a.peak ≤ b.peak ∧ a.path ≤ b.path

instance : LE Cost where le := costLe

def costLt (a b : Cost) : Prop := costLe a b ∧ ¬ costLe b a

instance : LT Cost where lt := costLt

instance decidableLECost : DecidableLE Cost := fun a b =>
  inferInstanceAs (Decidable (a.ops ≤ b.ops ∧ a.pack ≤ b.pack ∧ a.alloc ≤ b.alloc ∧
    a.xfer ≤ b.xfer ∧ a.dispatch ≤ b.dispatch ∧ a.peak ≤ b.peak ∧ a.path ≤ b.path))

instance decidableLTCost : DecidableLT Cost := fun a b =>
  inferInstanceAs (Decidable (a ≤ b ∧ ¬ b ≤ a))

/-- Dependent composition: a stage that runs after another accumulates every registered
resource, so this is the sum the epic takes for disjoint dependent work. -/
def costAdd (a b : Cost) : Cost :=
  ⟨a.ops + b.ops, a.pack + b.pack, a.alloc + b.alloc, a.xfer + b.xfer,
    a.dispatch + b.dispatch, a.peak + b.peak, a.path + b.path⟩

instance : Add Cost where add := costAdd

/-- The dependent sum, unfolded, so that composition rewrites and its laws close. -/
@[simp] theorem add_apply (a b : Cost) : a + b = costAdd a b := rfl

/-- Concurrent composition of independent stages: additive resources still total, because the
work of both branches is performed, while the dependency critical path is the `max` of the two
branches, because independent branches overlap in time. Peak-live storage adds as well: buffers
held by both branches are live at the same instant under this premise. -/
def overlap (a b : Cost) : Cost :=
  ⟨a.ops + b.ops, a.pack + b.pack, a.alloc + b.alloc, a.xfer + b.xfer,
    a.dispatch + b.dispatch, a.peak + b.peak, max a.path b.path⟩

/-- Fieldwise order introduction. -/
theorem Cost.le_of_fields {a b : Cost} (h1 : a.ops ≤ b.ops) (h2 : a.pack ≤ b.pack)
    (h3 : a.alloc ≤ b.alloc) (h4 : a.xfer ≤ b.xfer) (h5 : a.dispatch ≤ b.dispatch)
    (h6 : a.peak ≤ b.peak) (h7 : a.path ≤ b.path) : a ≤ b :=
  ⟨h1, h2, h3, h4, h5, h6, h7⟩

theorem Cost.eq_of_fields {a b : Cost} (h1 : a.ops = b.ops) (h2 : a.pack = b.pack)
    (h3 : a.alloc = b.alloc) (h4 : a.xfer = b.xfer) (h5 : a.dispatch = b.dispatch)
    (h6 : a.peak = b.peak) (h7 : a.path = b.path) : a = b := by
  cases a; cases b
  cases h1; cases h2; cases h3; cases h4; cases h5; cases h6; cases h7
  rfl

instance : Std.IsPartialOrder Cost where
  le_refl := fun a => ⟨Nat.le_refl a.ops, Nat.le_refl a.pack, Nat.le_refl a.alloc,
    Nat.le_refl a.xfer, Nat.le_refl a.dispatch, Nat.le_refl a.peak, Nat.le_refl a.path⟩
  le_trans := by
    intro a b c h₁ h₂
    exact Cost.le_of_fields (Nat.le_trans h₁.1 h₂.1) (Nat.le_trans h₁.2.1 h₂.2.1)
      (Nat.le_trans h₁.2.2.1 h₂.2.2.1) (Nat.le_trans h₁.2.2.2.1 h₂.2.2.2.1)
      (Nat.le_trans h₁.2.2.2.2.1 h₂.2.2.2.2.1)
      (Nat.le_trans h₁.2.2.2.2.2.1 h₂.2.2.2.2.2.1)
      (Nat.le_trans h₁.2.2.2.2.2.2 h₂.2.2.2.2.2.2)
  le_antisymm := by
    intro a b h₁ h₂
    exact Cost.eq_of_fields (Nat.le_antisymm h₁.1 h₂.1) (Nat.le_antisymm h₁.2.1 h₂.2.1)
      (Nat.le_antisymm h₁.2.2.1 h₂.2.2.1) (Nat.le_antisymm h₁.2.2.2.1 h₂.2.2.2.1)
      (Nat.le_antisymm h₁.2.2.2.2.1 h₂.2.2.2.2.1)
      (Nat.le_antisymm h₁.2.2.2.2.2.1 h₂.2.2.2.2.2.1)
      (Nat.le_antisymm h₁.2.2.2.2.2.2 h₂.2.2.2.2.2.2)

instance : Std.LawfulOrderLT Cost where
  lt_iff := fun _ _ => Iff.rfl

/-- No cost exceeds the cost of doing nothing. -/
theorem zero_le (a : Cost) : zero ≤ a := by
  refine Cost.le_of_fields ?_ ?_ ?_ ?_ ?_ ?_ ?_
  all_goals simp [zero]

/-- Composition of dependent stages is associative: the grouping of a dependency chain does not
change its cost, so a graph may be assembled left- or right-associated. -/
theorem add_assoc (a b c : Cost) : a + b + c = a + (b + c) := by
  cases a; cases b; cases c
  simp [costAdd, Nat.add_assoc]

/-- Composition of dependent stages is commutative: disjoint work sums the same whatever order the
stages are listed in. -/
theorem add_comm (a b : Cost) : a + b = b + a := by
  cases a; cases b
  simp [costAdd, Nat.add_comm]

theorem add_zero (a : Cost) : a + zero = a := by
  cases a
  simp [costAdd, zero]

theorem add_swap (a b c : Cost) : b + (a + c) = a + (b + c) :=
  ((add_assoc b a c).symm.trans (congrArg (fun t => t + c) (add_comm b a))).trans (add_assoc a b c)

/-- Composition is monotone: replacing a stage by a no-more-expensive one does not raise the
composed cost of anything it is composed with. -/
theorem add_monotone {a b c : Cost} (h : a ≤ b) : a + c ≤ b + c := by
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩
  exact Cost.le_of_fields (Nat.add_le_add_right h1 _) (Nat.add_le_add_right h2 _)
    (Nat.add_le_add_right h3 _) (Nat.add_le_add_right h4 _) (Nat.add_le_add_right h5 _)
    (Nat.add_le_add_right h6 _) (Nat.add_le_add_right h7 _)

theorem add_monotone_left {a b c : Cost} (h : a ≤ b) : c + a ≤ c + b := by
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩
  exact Cost.le_of_fields (Nat.add_le_add_left h1 _) (Nat.add_le_add_left h2 _)
    (Nat.add_le_add_left h3 _) (Nat.add_le_add_left h4 _) (Nat.add_le_add_left h5 _)
    (Nat.add_le_add_left h6 _) (Nat.add_le_add_left h7 _)

/-- Concurrency of independent stages is associative, for the same reason as `add_assoc`. -/
theorem overlap_assoc (a b c : Cost) : overlap (overlap a b) c = overlap a (overlap b c) := by
  cases a; cases b; cases c
  simp [overlap, Nat.max_assoc, Nat.add_assoc]

theorem overlap_zero (a : Cost) : overlap a zero = a := by
  cases a
  simp [overlap, zero]

/-- The dependency critical path of two concurrent independent stages is the `max` of the two
branches, not their sum. -/
theorem overlap_path (a b : Cost) : (overlap a b).path = max a.path b.path := rfl

/-- Concurrency changes no additive resource: the `max` on the critical path is the whole
difference between the two compositions. -/
theorem overlap_fields (a b : Cost) :
    (overlap a b).ops = (a + b).ops ∧ (overlap a b).pack = (a + b).pack ∧
      (overlap a b).alloc = (a + b).alloc ∧ (overlap a b).xfer = (a + b).xfer ∧
      (overlap a b).dispatch = (a + b).dispatch ∧ (overlap a b).peak = (a + b).peak :=
  ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- Two independent branches with a nonzero critical path each overlap to a strictly shorter
chain than running them in sequence. -/
theorem overlap_path_lt_add (a b : Cost) (ha : 0 < a.path) (hb : 0 < b.path) :
    (overlap a b).path < (a + b).path := by
  simp only [overlap, costAdd, add_apply]
  omega

/-- Concurrency never costs more than sequence: the schedule-independent sum is an upper bound on
the `max` composition, so a candidate may always be compared in the sum. -/
theorem overlap_le_add (a b : Cost) : overlap a b ≤ a + b := by
  show costLe (overlap a b) (a + b)
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [costAdd, overlap] <;> omega

/-- Concurrency is monotone in each branch. -/
theorem overlap_monotone {a b c : Cost} (h : a ≤ b) : overlap a c ≤ overlap b c := by
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩
  exact Cost.le_of_fields (Nat.add_le_add_right h1 _) (Nat.add_le_add_right h2 _)
    (Nat.add_le_add_right h3 _) (Nat.add_le_add_right h4 _) (Nat.add_le_add_right h5 _)
    (Nat.add_le_add_right h6 _) (by simp [overlap] <;> omega)

theorem overlap_monotone_right {a b c : Cost} (h : a ≤ b) : overlap c a ≤ overlap c b := by
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩
  exact Cost.le_of_fields (Nat.add_le_add_left h1 _) (Nat.add_le_add_left h2 _)
    (Nat.add_le_add_left h3 _) (Nat.add_le_add_left h4 _) (Nat.add_le_add_left h5 _)
    (Nat.add_le_add_left h6 _) (by simp [overlap] <;> omega)

/-- The registered improvement relation: no resource grows and at least one resource shrinks.
The `≠` component is what separates an improvement from a tie. -/
def strictlyCheaper (a b : Cost) : Prop := a ≤ b ∧ a ≠ b

/-- A candidate that ties the baseline is not an improvement, so a non-strict comparison never
admits a candidate. -/
theorem tie_not_improvement (c : Cost) : ¬ strictlyCheaper c c := fun h => h.2 rfl

/-- A candidate that grows any registered resource is rejected, whatever it saves elsewhere: a
longer critical path is not paid for by fewer packing bytes. -/
theorem regression_not_improvement {a b : Cost} (h : a ≤ b) (grew : a.path > b.path) :
    ¬ strictlyCheaper a b := by
  rcases h with ⟨h1, h2, h3, h4, h5, h6, h7⟩
  omega

/-- Componentwise domination with a difference somewhere is a difference in a field. -/
theorem Cost.exists_strict {a b : Cost} (h : a ≤ b) (hne : a ≠ b) :
    a.ops < b.ops ∨ a.pack < b.pack ∨ a.alloc < b.alloc ∨ a.xfer < b.xfer ∨
      a.dispatch < b.dispatch ∨ a.peak < b.peak ∨ a.path < b.path := by
  rcases h with ⟨d1, d2, d3, d4, d5, d6, d7⟩
  by_cases e1 : a.ops = b.ops
  · by_cases e2 : a.pack = b.pack
    · by_cases e3 : a.alloc = b.alloc
      · by_cases e4 : a.xfer = b.xfer
        · by_cases e5 : a.dispatch = b.dispatch
          · by_cases e6 : a.peak = b.peak
            · by_cases e7 : a.path = b.path
              · exact (hne (Cost.eq_of_fields e1 e2 e3 e4 e5 e6 e7)).elim
              · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (by omega))))))
            · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (by omega))))))
          · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl (by omega)))))
        · exact Or.inr (Or.inr (Or.inr (Or.inl (by omega))))
      · exact Or.inr (Or.inr (Or.inl (by omega)))
    · exact Or.inr (Or.inl (by omega))
  · exact Or.inl (by omega)

/-- The registered `Nat` cost compared by the candidate certificate, in the shape
`proof/DispatchAdmission.lean` `Admission.improves` requires (`afterCost x < beforeCost x`). Every
weight is positive, so the scalar is monotone in every field. -/
def registeredCost (c : Cost) : Nat :=
  8 * c.ops + 64 * c.pack + 16 * c.alloc + 64 * c.xfer + 4 * c.dispatch + 64 * c.peak + c.path

/-- Domination in the componentwise order is sufficient for the strict `Nat` reduction the
candidate certificate demands. -/
theorem dominates_improves {a b : Cost} (h : strictlyCheaper a b) :
    registeredCost a < registeredCost b := by
  rcases h with ⟨hdom, hne⟩
  rcases Cost.exists_strict hdom hne with s
  rcases hdom with ⟨d1, d2, d3, d4, d5, d6, d7⟩
  simp only [registeredCost]
  rcases s with s1 | s2 | s3 | s4 | s5 | s6 | s7 <;> omega

/-- The scalar is not a substitute for the order: a candidate that lengthens the critical path can
still show a lower `registeredCost`, so selecting on the scalar alone would accept a
regression. -/
theorem scalar_not_sufficient :
    ∃ a b : Cost, registeredCost a < registeredCost b ∧ ¬ a ≤ b :=
  ⟨⟨0, 0, 0, 0, 0, 0, 4⟩, ⟨1, 0, 0, 0, 0, 0, 0⟩, by decide, by decide⟩

/-- One unit of the typed computation: a projection identified by its index, reading its weight
and packed-activation buffers and writing one output buffer. Buffer identifiers are indices; this
module fixes no buffer sizes and no instruction set. -/
structure Stage where
  id : Nat
  reads : List Nat
  writes : List Nat
  deriving DecidableEq, Repr

/-- One dependency layer: stages that may execute concurrently, each bound to its own registered
cost record. -/
abbrev Layer := List (Stage × Cost)

/-- A scheduled plan: the typed computation in dependency order, its schedule of dependency
layers, and the effects performed once and shared by every layer. -/
structure Plan where
  stages : List Stage
  layers : List Layer
  shared : Cost
  deriving DecidableEq

/-- The observable result of a plan: the ordered stages it computes. Two schedules of one
computation agree here, and only `planCost` may differ. -/
def observe (p : Plan) : List Stage := p.stages

/-- A plan binds one registered cost to each stage of its computation, each stage exactly once and
in order, so the composition can neither copy nor eliminate stateful work. Temporal cost is one
traversal of the layer list per `flatMap` and one `List.map`; spatial cost is O(1) words beyond
the traversal result. -/
def Scheduled (p : Plan) : Prop := (p.layers.flatMap id).map Prod.fst = p.stages

/-- The composed cost of a sequence of dependent stages: the fold of `+` over the list. Temporal
cost is n recursive steps in the number of stages n; spatial cost is O(1) words. -/
def seq (l : List Cost) : Cost := l.foldr (· + ·) zero

/-- The composed cost of one dependency layer: the fold of `overlap` over its stages. Temporal
cost is n recursive steps in the number of stages n; spatial cost is O(1) words. -/
def layerCost (l : Layer) : Cost := l.foldr (fun b acc => overlap b.2 acc) zero

/-- The composed cost of a plan: dependent layers add, concurrent stages inside a layer overlap,
and the shared effect is counted once. Temporal cost is n + k recursive steps in the number of
scheduled stages n and dependency layers k; spatial cost is one list of k layer costs. -/
def planCost (p : Plan) : Cost := (p.layers.map layerCost).foldr (· + ·) zero + p.shared

/-- The composed cost of a plan, under the field name the dominance statements read with. -/
def Plan.cost (p : Plan) : Cost := planCost p

/-- Two schedules with the same stage order, pointwise related. -/
def Pointwise {α : Type} (r : α → α → Prop) (a b : List α) : Prop :=
  match a, b with
  | [], [] => True
  | x :: xs, y :: ys => r x y ∧ Pointwise r xs ys
  | _, _ => False

/-- The fold of `+` is invariant under a permutation of its arguments: disjoint dependent work
sums the same whatever order the stages are listed in. -/
theorem seq_perm {a b : List Cost} (h : List.Perm a b) : seq a = seq b := by
  induction h with
  | nil => rfl
  | cons a h ih => exact congrArg (fun c => a + c) ih
  | swap a b l => simp only [seq, List.foldr_cons]; exact add_swap _ _ _
  | trans _ _ ih₁ ih₂ => exact ih₁.trans ih₂

theorem seq_append (a b : List Cost) : seq (a ++ b) = seq b + seq a := by
  induction a with
  | nil => exact (add_zero (seq b)).symm
  | cons c a ih =>
    show c + seq (a ++ b) = seq b + (c + seq a)
    exact (congrArg (fun t => c + t) ih).trans (add_swap _ _ _)

/-- Composition of a sequence is monotone in every stage. -/
theorem seq_monotone :
    ∀ {a b : List Cost}, Pointwise (· ≤ · : Cost → Cost → Prop) a b → seq a ≤ seq b
  | [], [], _ => zero_le _
  | x :: xs, y :: ys, h => by
    show x + seq xs ≤ y + seq ys
    exact Std.IsPreorder.le_trans _ _ _ (add_monotone (c := seq xs) h.1)
      (add_monotone_left (seq_monotone h.2))

/-- Composition of a layer is monotone in every stage. -/
theorem layerCost_monotone :
    ∀ {a b : Layer}, Pointwise (fun p q : Stage × Cost => p.2 ≤ q.2) a b → layerCost a ≤ layerCost b
  | [], [], _ => zero_le _
  | (_, c) :: xs, (_, d) :: ys, h => by
    show overlap c (layerCost xs) ≤ overlap d (layerCost ys)
    exact Std.IsPreorder.le_trans _ _ _ (overlap_monotone (c := layerCost xs) h.1)
      (overlap_monotone_right (layerCost_monotone h.2))

/-- The dependency critical path of a layer is the `max` over its stages: concurrent stages
overlap, and the chain the plan cannot shorten below is the longest branch. -/
theorem layer_critical_path (l : Layer) :
    (layerCost l).path = l.foldr (fun b acc => max b.2.path acc) 0 := by
  induction l with
  | nil => rfl
  | cons b l ih =>
    show max b.2.path (layerCost l).path = max b.2.path (l.foldr (fun c acc => max c.2.path acc) 0)
    rw [ih]

/-- The composed cost of a plan is monotone in its shared effect. -/
theorem planCost_mono_in_shared {p : Plan} {k : Cost} (h : p.shared ≤ k) :
    planCost p ≤ planCost { p with shared := k } := add_monotone_left h

/-- Adding work to the shared effect adds exactly its record to the plan: the shared effect enters
the composition once, so a duplicated shared effect is not expressible. -/
theorem planCost_shared_add (p : Plan) (k : Cost) :
    planCost { p with shared := p.shared + k } = planCost p + k := by
  cases p
  simp only [planCost]
  exact (add_assoc _ _ _).symm

/-- The registered example of the epic's sharing mechanism: three projections that each pack their
own activation, against one resident activation pack shared by all three. The counts below are
this module's declared inputs, not a measurement. -/
def proj0 : Stage := ⟨0, [0, 1], [8]⟩
def proj1 : Stage := ⟨1, [2, 3], [9]⟩
def proj2 : Stage := ⟨2, [4, 5], [10]⟩
def projections : List Stage := [proj0, proj1, proj2]

/-- One projection reading an already packed activation: 12 arithmetic operations, one dispatch,
critical path 3. -/
def dotCost : Cost := ⟨12, 0, 0, 0, 1, 0, 3⟩

/-- The same projection packing its own activation first: 3072 packing/copy bytes, one allocation,
3072 uploaded bytes, 3072 peak-live bytes, critical path 4. -/
def packDotCost : Cost := ⟨12, 3072, 1, 3072, 1, 3072, 4⟩

/-- One resident activation pack shared by the three projections: no arithmetic, and no dispatch
of its own because the pack rides in the launch that consumes it; critical path 1 as a dependency
of the layer. -/
def residentPack : Cost := ⟨0, 3072, 1, 3072, 0, 3072, 1⟩

/-- Baseline: each projection packs and runs on its own, one dependent layer each. -/
def perProjectionPlan : Plan where
  stages := projections
  layers := [[(proj0, packDotCost)], [(proj1, packDotCost)], [(proj2, packDotCost)]]
  shared := zero

/-- Candidate: one resident pack hoisted out of the projections, which then run as one concurrent
layer. -/
def residentPlan : Plan where
  stages := projections
  layers := [[(proj0, dotCost), (proj1, dotCost), (proj2, dotCost)]]
  shared := residentPack

/-- The same three projections run as three dependent layers, packing excluded. -/
def sequentialPlan : Plan where
  stages := projections
  layers := [[(proj0, dotCost)], [(proj1, dotCost)], [(proj2, dotCost)]]
  shared := zero

/-- The same three projections run as one concurrent layer, packing excluded. -/
def forkedPlan : Plan where
  stages := projections
  layers := [[(proj0, dotCost), (proj1, dotCost), (proj2, dotCost)]]
  shared := zero

theorem perProjection_scheduled : Scheduled perProjectionPlan := rfl
theorem resident_scheduled : Scheduled residentPlan := rfl
theorem sequential_scheduled : Scheduled sequentialPlan := rfl
theorem forked_scheduled : Scheduled forkedPlan := rfl

/-- The baseline composes to 36 arithmetic operations, 9216 packing/copy bytes, 3 allocations, 9216
uploaded bytes, 3 dispatches, 9216 peak-live bytes and critical path 12. -/
theorem perProjection_cost : perProjectionPlan.cost = ⟨36, 9216, 3, 9216, 3, 9216, 12⟩ := by
  decide

/-- The resident candidate composes to the same 36 arithmetic operations and 3 dispatches, with
3072 packing/copy bytes, 1 allocation, 3072 uploaded bytes, 3072 peak-live bytes and critical
path 4. -/
theorem resident_cost : residentPlan.cost = ⟨36, 3072, 1, 3072, 3, 3072, 4⟩ := by
  decide

/-- Three dependent layers of the same stages compose to critical path 9. -/
theorem sequential_cost : sequentialPlan.cost = ⟨36, 0, 0, 0, 3, 0, 9⟩ := by
  decide

/-- One concurrent layer of the same stages composes to critical path 3. -/
theorem forked_cost : forkedPlan.cost = ⟨36, 0, 0, 0, 3, 0, 3⟩ := by
  decide

/-- The two schedules of the projections compute the same thing. -/
theorem observations_equal : observe forkedPlan = observe sequentialPlan := rfl

/-- The two sharing candidates compute the same thing. -/
theorem equal_outputs : observe residentPlan = observe perProjectionPlan := rfl

/-- The resident candidate strictly dominates the baseline in the registered order. -/
theorem resident_dominates : strictlyCheaper residentPlan.cost perProjectionPlan.cost := by
  show residentPlan.cost ≤ perProjectionPlan.cost ∧ residentPlan.cost ≠ perProjectionPlan.cost
  decide

/-- Domination yields the strict `Nat` cost reduction the candidate certificate requires, on this
non-empty witness of registered improvement. -/
theorem resident_improves :
    registeredCost residentPlan.cost < registeredCost perProjectionPlan.cost :=
  dominates_improves resident_dominates

/-- The shared resident pack is what the improvement consists of. -/
theorem packing_reduction : residentPlan.cost.pack < perProjectionPlan.cost.pack := by
  decide

/-- Overlapping the three independent projections shortens the critical path and changes nothing
else: the nonvacuous witness of the `max` law against the sum. -/
theorem forked_critical_path_reduced : forkedPlan.cost < sequentialPlan.cost := by
  decide

end CostAlgebra

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"COST_ALGEBRA_AXIOM_REJECT {c}: {bad.toList}"
    logInfo m!"COST_ALGEBRA_AUDIT {c}: axioms {axioms.toList}"

#resource_audit CostAlgebra.add_apply
#resource_audit CostAlgebra.Cost.le_of_fields
#resource_audit CostAlgebra.Cost.eq_of_fields
#resource_audit CostAlgebra.zero_le
#resource_audit CostAlgebra.add_assoc
#resource_audit CostAlgebra.add_comm
#resource_audit CostAlgebra.add_zero
#resource_audit CostAlgebra.add_swap
#resource_audit CostAlgebra.add_monotone
#resource_audit CostAlgebra.add_monotone_left
#resource_audit CostAlgebra.overlap_assoc
#resource_audit CostAlgebra.overlap_zero
#resource_audit CostAlgebra.overlap_path
#resource_audit CostAlgebra.overlap_fields
#resource_audit CostAlgebra.overlap_path_lt_add
#resource_audit CostAlgebra.overlap_le_add
#resource_audit CostAlgebra.overlap_monotone
#resource_audit CostAlgebra.overlap_monotone_right
#resource_audit CostAlgebra.tie_not_improvement
#resource_audit CostAlgebra.regression_not_improvement
#resource_audit CostAlgebra.Cost.exists_strict
#resource_audit CostAlgebra.dominates_improves
#resource_audit CostAlgebra.scalar_not_sufficient
#resource_audit CostAlgebra.seq_perm
#resource_audit CostAlgebra.seq_append
#resource_audit CostAlgebra.seq_monotone
#resource_audit CostAlgebra.layerCost_monotone
#resource_audit CostAlgebra.layer_critical_path
#resource_audit CostAlgebra.planCost_mono_in_shared
#resource_audit CostAlgebra.planCost_shared_add
#resource_audit CostAlgebra.perProjection_scheduled
#resource_audit CostAlgebra.resident_scheduled
#resource_audit CostAlgebra.sequential_scheduled
#resource_audit CostAlgebra.forked_scheduled
#resource_audit CostAlgebra.perProjection_cost
#resource_audit CostAlgebra.resident_cost
#resource_audit CostAlgebra.sequential_cost
#resource_audit CostAlgebra.forked_cost
#resource_audit CostAlgebra.observations_equal
#resource_audit CostAlgebra.equal_outputs
#resource_audit CostAlgebra.resident_dominates
#resource_audit CostAlgebra.resident_improves
#resource_audit CostAlgebra.packing_reduction
#resource_audit CostAlgebra.forked_critical_path_reduced

run_cmd Lean.logInfo "COST_ALGEBRA_KERNEL_OK"
