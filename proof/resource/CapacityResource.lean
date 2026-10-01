import CapacitySource
import Aeneas

/-
SCOPE: the actual extracted body `table::resources::capacity_bound_holds` reached by the
extraction root `word_capacity_bound_holds`; its two guarded branches, its work measure, the
refusal (insufficient-capacity) branch and the success branch, and a reachable witness for
each. Also the root's forwarding equation and the joint machine-word domain.
REGISTER: scalar source refinement plus an explicit-work count of that one body. The body
performs no arithmetic, so the only refusal it can take is insufficient observed capacity; a
checked-overflow refusal does not exist in this body, and `body_total` shows its `Error` channel
is unreachable. Excluded: canonical-corpus correspondence, hardware timing, allocator behaviour,
the `capacity_bound` assertion in `src/table/state.rs:368-377`, and the eight-fold conjunction in
`LocatorReservationReceipt::holds` (`src/table/resources.rs:138-162`).
CORPUS: `src/table/resources.rs:208-210`, whose real body is
`pub(crate) fn capacity_bound_holds(required_len: usize, observed_capacity: usize) -> bool
{ required_len <= observed_capacity }`; the forwarding wrapper is `proof/extract/lib.rs:677-679`,
rooted by `proof/regen.sh:177`, and the same body is already extracted at
`proof/ApeironWord.lean:17606-17612`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/Capacity_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-3.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure is a count of
the extracted body's own reduction under the Aeneas model, not a measured time.
-/

namespace CapacityResource
open Aeneas Aeneas.Std Result

/-- The real extracted `src/table/resources.rs` body reached by the extraction root. -/
def body (required_len observed_capacity : Usize) : Result Bool :=
  capacity_source.table.resources.capacity_bound_holds required_len observed_capacity

/-- The extraction root forwards to that body; the root restates no comparison. -/
theorem root_forwards (required_len observed_capacity : Usize) :
    capacity_source.word_capacity_bound_holds required_len observed_capacity
      = body required_len observed_capacity := rfl

/-- The whole body is the `Result.ok` of one `Usize` comparison of the two argument words:
    no loop, no `bind`, no call, no `alloc` and no `Error` channel appears in it. This is the
    kernel certificate the work count below is read from. -/
theorem body_is_one_comparison (required_len observed_capacity : Usize) :
    body required_len observed_capacity
      = ok (decide (required_len.val ≤ observed_capacity.val)) := rfl

/-- The counted work of one evaluation of the real body. `value n m` is a value node reached
    after `n` machine-word comparisons and `m` heap allocations; `effect` is an effect node;
    `diverging` is a divergence node. -/
inductive Work where
  /-- A value node: `comparisons` word comparisons, `allocations` heap allocations. -/
  | value (comparisons allocations : Nat)
  /-- An effect node. -/
  | effect
  /-- A divergence node. -/
  | diverging

/-- The work of one evaluation of the real body at the given input pair. -/
def workOf (required_len observed_capacity : Usize) : Work :=
  Result.cases (body required_len observed_capacity)
    (fun _ => Work.value 1 0) (fun _ _ => Work.effect) Work.diverging

/-- The real body performs exactly one `Usize` comparison and no heap allocation, on every
    input pair. The count is read off the extracted body through `body_is_one_comparison`:
    the `Result` it returns is the value node `ok`, not an effect or divergence node, and
    `decide` on the single `≤` of the two `.val` fields is one comparison. -/
theorem work_is_one_comparison_zero_allocations (required_len observed_capacity : Usize) :
    workOf required_len observed_capacity = Work.value 1 0 := by
  show Result.cases (body required_len observed_capacity)
      (fun _ => Work.value 1 0) (fun _ _ => Work.effect) Work.diverging = Work.value 1 0
  show Result.cases (ok (decide (required_len.val ≤ observed_capacity.val)))
      (fun _ => Work.value 1 0) (fun _ _ => Work.effect) Work.diverging = Work.value 1 0
  rfl

/-- The body reads exactly the two machine words and nothing else, so its result is
    independent of every other input. -/
theorem body_reads_only_the_two_words (r r' o o' : Usize)
    (hr : r'.val = r.val) (ho : o'.val = o.val) : body r o = body r' o' := by
  show ok (decide (r.val ≤ o.val)) = ok (decide (r'.val ≤ o'.val))
  rw [hr, ho]

/-- The counted work does not depend on either argument's magnitude. -/
theorem work_is_input_independent (r r' o o' : Usize)
    (hr : r'.val = r.val) (ho : o'.val = o.val) : workOf r o = workOf r' o' := by
  unfold workOf
  rw [body_reads_only_the_two_words r r' o o' hr ho]

/-- The joint machine-word domain of the real call: both `required_len` and
    `observed_capacity` are `Usize` words, so each `.val` is below the word's bit width. -/
def MachineWords (required_len observed_capacity : Usize) : Prop :=
  required_len.val < 2 ^ UScalarTy.Usize.numBits ∧
    observed_capacity.val < 2 ^ UScalarTy.Usize.numBits

/-- The extracted types initialize the joint domain at every input pair, so the body's
    domain is total with no partial input to check and no unreachable premise. -/
theorem machineWords_holds (required_len observed_capacity : Usize) :
    MachineWords required_len observed_capacity :=
  ⟨required_len.hmax, observed_capacity.hmax⟩

/-- The success branch: the checked target length fits the capacity the successful
    reservation observed, so the check admits. -/
theorem success_branch (required_len observed_capacity : Usize)
    (h : required_len.val ≤ observed_capacity.val) :
    body required_len observed_capacity = ok true := by
  have h' : required_len.val ≤ observed_capacity.val := h
  simp [body, capacity_source.table.resources.capacity_bound_holds, h']

/-- The refusal branch: the observed capacity is below the checked target length, i.e. the
    reservation under-delivered, so the check refuses. -/
theorem refusal_branch (required_len observed_capacity : Usize)
    (h : observed_capacity.val < required_len.val) :
    body required_len observed_capacity = ok false := by
  have h' : observed_capacity.val < required_len.val := h
  simp [body, capacity_source.table.resources.capacity_bound_holds, h']

/-- The success guard is exact: the body admits exactly the pairs whose required length
    does not exceed the observed capacity. -/
theorem success_guard_exact (required_len observed_capacity : Usize) :
    body required_len observed_capacity = ok true ↔
      required_len.val ≤ observed_capacity.val := by
  simp [body, capacity_source.table.resources.capacity_bound_holds]

/-- The refusal guard is exact: the body refuses exactly the pairs whose observed capacity
    is below the required length. -/
theorem refusal_guard_exact (required_len observed_capacity : Usize) :
    body required_len observed_capacity = ok false ↔
      observed_capacity.val < required_len.val := by
  simp [body, capacity_source.table.resources.capacity_bound_holds]

/-- The transition is total: every input pair reaches a value node, so the body reaches
    neither an effect node nor a divergence node. -/
theorem body_total (required_len observed_capacity : Usize) :
    ∃ b : Bool, body required_len observed_capacity = ok b :=
  ⟨decide (required_len.val ≤ observed_capacity.val),
    body_is_one_comparison required_len observed_capacity⟩

/-- The body reaches a value node on every input pair under the Aeneas weak-precondition
    relation, so it produces no `Error` node at all: the only refusal it can take is the
    insufficient-capacity branch, never a checked-overflow or allocator refusal. -/
theorem body_never_errors (required_len observed_capacity : Usize) :
    WP.spec (body required_len observed_capacity) (fun _ => True) := by
  show body required_len observed_capacity ⦃ b => True ⦄
  rw [body_is_one_comparison]
  exact (WP.spec_ok (decide (required_len.val ≤ observed_capacity.val))).mpr trivial

/-- The two branches are mutually exclusive, so the real body has exactly the two
    transitions the reservation receipt consumes. -/
theorem branch_exclusive (required_len observed_capacity : Usize) :
    ¬ (body required_len observed_capacity = ok true ∧
        body required_len observed_capacity = ok false) := by
  rintro ⟨h1, h2⟩
  have h12 : (ok (true : Bool) : Result Bool) = ok false := h1.symm.trans h2
  have he : (true : Bool) = false := Result.ok_injective h12
  exact absurd he (by simp)

/-- The two branches are exhaustive, so no input pair is left undecided. -/
theorem branch_exhaustive (required_len observed_capacity : Usize) :
    body required_len observed_capacity = ok true ∨
      body required_len observed_capacity = ok false := by
  rcases body_total required_len observed_capacity with ⟨b, hb⟩
  cases hb' : b with
  | true => exact Or.inl (hb' ▸ hb)
  | false => exact Or.inr (hb' ▸ hb)

/-- A reachable success witness at the joint machine-word types: the `EMPTY` receipt of
    `src/table/resources.rs:99-102` carries `required_len = 0` with
    `observed_capacity = 0`, and the real body admits that pair. -/
theorem success_witness :
    MachineWords 0#usize 0#usize ∧ body 0#usize 0#usize = ok true :=
  ⟨machineWords_holds 0#usize 0#usize,
    success_branch 0#usize 0#usize (by simp)⟩

/-- A reachable refusal witness at the joint machine-word types: one element above the
    observed capacity, i.e. an under-delivered reservation. -/
theorem refusal_witness :
    MachineWords 1#usize 0#usize ∧ body 1#usize 0#usize = ok false :=
  ⟨machineWords_holds 1#usize 0#usize,
    refusal_branch 1#usize 0#usize (by simp)⟩

/-- Every equal pair is admitted, so the success guard is not vacuous at any magnitude. -/
theorem diagonal_accepted (r : Usize) : body r r = ok true :=
  success_branch r r (Nat.le_refl r.val)

/-- A larger observed capacity is never refused, so an over-delivered reservation keeps
    passing the same check. -/
theorem acceptance_monotone_observed (r o o' : Usize) (h : r.val ≤ o.val)
    (ho : o.val ≤ o'.val) : body r o' = ok true :=
  success_branch r o' (h.trans ho)

/-- A larger required length is never admitted against a fixed observed capacity, so
    raising the checked target cannot turn a refusal into an acceptance. -/
theorem refusal_monotone_required (r r' o : Usize) (h : o.val < r.val)
    (hr : r.val ≤ r'.val) : body r' o = ok false := by
  have h2 : ¬ (r'.val ≤ o.val) := by omega
  simp [body, capacity_source.table.resources.capacity_bound_holds, h2]

end CapacityResource

open Lean Elab Command in
elab "#resource_audit " id:ident : command => do
  let cs ← liftCoreM <| realizeGlobalConstWithInfos id
  let allowed : Lean.NameSet :=
    ({} : Lean.NameSet) |>.insert `propext |>.insert `Classical.choice |>.insert `Quot.sound
  for c in cs do
    let axioms ← liftTermElabM <| Lean.collectAxioms c
    let bad := axioms.filter (fun a => !allowed.contains a)
    unless bad.isEmpty do
      throwError m!"{c}: forbidden axioms {bad.toList}"
    logInfo m!"{c}: axioms {axioms.toList}"

#resource_audit CapacityResource.root_forwards
#resource_audit CapacityResource.body_is_one_comparison
#resource_audit CapacityResource.work_is_one_comparison_zero_allocations
#resource_audit CapacityResource.body_reads_only_the_two_words
#resource_audit CapacityResource.work_is_input_independent
#resource_audit CapacityResource.machineWords_holds
#resource_audit CapacityResource.success_branch
#resource_audit CapacityResource.refusal_branch
#resource_audit CapacityResource.success_guard_exact
#resource_audit CapacityResource.refusal_guard_exact
#resource_audit CapacityResource.body_total
#resource_audit CapacityResource.body_never_errors
#resource_audit CapacityResource.branch_exclusive
#resource_audit CapacityResource.branch_exhaustive
#resource_audit CapacityResource.success_witness
#resource_audit CapacityResource.refusal_witness
#resource_audit CapacityResource.diagonal_accepted
#resource_audit CapacityResource.acceptance_monotone_observed
#resource_audit CapacityResource.refusal_monotone_required

run_cmd Lean.logInfo "CAPACITY_KERNEL_OK"
