import PublishPreparedSource
import Aeneas

/-
SCOPE: the actual extracted body `table::resources::publish_prepared` reached by the extraction
root `word_stored_publish_prepared`; its two guarded publication transitions, its work measure (a
constant switch), its refusal branch, a reachable success witness, a reachable refusal witness, the
root's forwarding equation, and the joint totality of the transition over the whole extracted input
type.
REGISTER: source refinement of one straight-line body plus an explicit work count of that body.
The body is a two-arm `match` on `core::result::Result`: it moves the caller's exact success tuple
into `PreparedResult::Prepared`, or the concrete `StorageError` into `PreparedResult::Refused`. It
does not plan, reserve, read capacity, initialize, clone, copy or certify a predicate, so no
arithmetic refusal and no allocator refusal exists in it; `body_never_errors` shows its `Error`
channel is unreachable. Excluded: canonical-corpus correspondence, hardware timing, allocator
behaviour, the adapter that produces the envelope (`src/table/prepared.rs`), the well-formedness
predicates of `RebuildPlan`/`RebuildReserved`/the two receipts, and the relocation and commit cores
that consume `PreparedResult`.
CORPUS: `src/table/resources.rs:217-228`, whose real body is
`pub(crate) fn publish_prepared<R, V>(result: Result<PreparedEnvelope<R, V>, StorageError>) ->
PreparedResult<R, V> { match result { Ok((plan, storage, outcome)) => PreparedResult::Prepared
{ plan, storage, outcome }, Err(error) => PreparedResult::Refused(error) } }`; the forwarding
wrapper is `proof/extract/lib.rs:652-659`, rooted by `proof/regen.sh:175`, and the same body is
already extracted at `proof/ApeironWord.lean:20653-20666`.
REFERENCE: /home/socol/Workspace/apeiron/proof/ApeironWord.lean
GATE: `python3 proof/resource/PublishPrepared_gate.py --controls`, 540-second budget, lane
/tmp/opencode/apeiron-lane-1.lock.
GRADE: `proved` for the theorems below when that gate exits 0. The work measure is a count of the
extracted body's own reduction under the Aeneas model, not a measured time.
-/

namespace PublishPreparedResource
open Aeneas Aeneas.Std Result

/-- The published value the relocation and commit cores consume, at the concrete carrier and value
    word the extraction root is specialized at. -/
abbrev PublishedWord := publish_prepared_source.table.resources.PreparedResult
  (publish_prepared_source.table.keyrep.Stored Std.I64) Std.I64

/-- The stored-carrier word type the extraction root is specialized at. -/
abbrev Stored := publish_prepared_source.table.keyrep.Stored Std.I64

/-- The value word type the extraction root is specialized at. -/
abbrev Word := Std.I64

/-- The checked plan the adapter hands to publication. -/
abbrev Plan := publish_prepared_source.table.RebuildPlan

/-- The materialized reserved storage the adapter hands to publication. -/
abbrev Reserved := publish_prepared_source.table.RebuildReserved Stored Word

/-- The receipt pair and pending held carrier the adapter hands to publication. -/
abbrev Outcome := publish_prepared_source.table.resources.PreparedOutcome Stored

/-- The adapter's exact success tuple: the input the publication arm consumes. -/
abbrev Envelope := Plan × Reserved × Outcome

/-- The checked refusal type of the table. -/
abbrev StorageError := publish_prepared_source.table.StorageError

/-- The real extracted `src/table/resources.rs` body reached by the extraction root. -/
def body (result : core.result.Result Envelope StorageError) : Result PublishedWord :=
  publish_prepared_source.table.resources.publish_prepared result

/-- The extraction root forwards to that body; the root restates no match arm. -/
theorem root_forwards (result : core.result.Result Envelope StorageError) :
    publish_prepared_source.word_stored_publish_prepared result = body result := rfl

/-- One `core::result::Result` switch: each arm applies a single `PreparedResult` constructor to
    fields of the input and computes nothing. -/
def oneSwitch (result : core.result.Result Envelope StorageError) : Result PublishedWord :=
  match result with
  | core.result.Result.Ok t =>
      ok (publish_prepared_source.table.resources.PreparedResult.Prepared t.1 t.2.1 t.2.2)
  | core.result.Result.Err e =>
      ok (publish_prepared_source.table.resources.PreparedResult.Refused e)

/-- The kernel certificate the work count below is read from: the whole real body is `Result.ok` of
    one switch on the input's discriminant. No `loop`, no `bind`, no `alloc`, no call and no `Error`
    node appears in its reduction. -/
theorem body_is_one_constant_switch (result : core.result.Result Envelope StorageError) :
    body result = oneSwitch result := by
  cases result <;> rfl

/-- The counted work of one evaluation of the real body: `moves` value constructor applications,
    `comparisons` machine-word comparisons and `allocations` heap allocations. `effect` is an
    effect node and `diverging` a divergence node. -/
inductive Work where
  /-- A value node: `moves` constructor applications, `comparisons` word comparisons and
      `allocations` heap allocations. -/
  | value (moves comparisons allocations : Nat)
  /-- An effect node. -/
  | effect
  /-- A divergence node. -/
  | diverging

/-- The work of one evaluation of the real body at the given input. -/
def workOf (result : core.result.Result Envelope StorageError) : Work :=
  Result.cases (body result)
    (fun _ => Work.value 1 0 0) (fun _ _ => Work.effect) Work.diverging

/-- The closed form of the publication work: exactly one value move, no machine-word comparison
    and no heap allocation, on every input. The count is read off the extracted body through
    `body_is_one_constant_switch`: the returned node is the value `ok`, the switch reads only the
    input discriminant, and each arm applies one constructor without computing an argument. The
    body contains no `loop`, so it has no strictly decreasing rank to exhibit and this closed form
    is the whole measure. -/
theorem work_is_one_move_no_comparison_no_allocation
    (result : core.result.Result Envelope StorageError) :
    workOf result = Work.value 1 0 0 := by
  cases result with
  | Ok t =>
      show Result.cases
          (ok (publish_prepared_source.table.resources.PreparedResult.Prepared t.1 t.2.1 t.2.2))
          (fun _ => Work.value 1 0 0) (fun _ _ => Work.effect) Work.diverging
        = Work.value 1 0 0
      rfl
  | Err e =>
      show Result.cases
          (ok (publish_prepared_source.table.resources.PreparedResult.Refused e))
          (fun _ => Work.value 1 0 0) (fun _ _ => Work.effect) Work.diverging
        = Work.value 1 0 0
      rfl

/-- The work does not grow with the envelope: two success tuples of any plan length, reserved
    region count and receipt size cost the same single move. -/
theorem work_is_independent_of_the_envelope (t t' : Envelope) :
    workOf (core.result.Result.Ok t) = workOf (core.result.Result.Ok t') := rfl

/-- The two arms cost the same, so publication is not cheaper than refusal and refusal is not
    cheaper than publication: each moves exactly one value. -/
theorem work_is_independent_of_the_arm (t : Envelope) (e : StorageError) :
    workOf (core.result.Result.Ok t) = workOf (core.result.Result.Err e) := rfl

/-- The publication transition on the success guard: the caller's exact success tuple moves into
    `PreparedResult::Prepared`, plan, reserved storage and outcome field for field. -/
theorem publish_transition (plan : Plan) (storage : Reserved) (outcome : Outcome) :
    body (core.result.Result.Ok (plan, storage, outcome))
      = ok (publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome) :=
  rfl

/-- The refusal transition: the concrete `StorageError` reaches `PreparedResult::Refused` unchanged,
    so the failure cause survives publication. -/
theorem refusal_transition (e : StorageError) :
    body (core.result.Result.Err e)
      = ok (publish_prepared_source.table.resources.PreparedResult.Refused e) := rfl

/-- The model `Result.ok` is injective at every payload type. -/
theorem ok_injective' {α : Type} {p q : α} (h : (ok p : Result α) = ok q) : p = q :=
  (Result.ok.injEq (α := α) (a := p) (b := q)).mp h

/-- The two constructors of `PreparedResult` are disjoint, so no published value is both a
    publication and a refusal. -/
theorem published_arm_clash (plan : Plan) (storage : Reserved) (outcome : Outcome)
    (e : StorageError) :
    publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome
      ≠ publish_prepared_source.table.resources.PreparedResult.Refused e := by
  intro h
  cases h

/-- Two published tuples are equal exactly when their three components are. -/
theorem published_inj {plan : Plan} {storage : Reserved} {outcome : Outcome}
    {plan' : Plan} {storage' : Reserved} {outcome' : Outcome}
    (h : publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome
        = publish_prepared_source.table.resources.PreparedResult.Prepared plan' storage' outcome') :
    plan = plan' ∧ storage = storage' ∧ outcome = outcome' := by
  cases h
  exact ⟨rfl, rfl, rfl⟩

/-- An `Ok` input is never an `Err` input, in that order. -/
theorem ok_err_clash (t : Envelope) (e : StorageError)
    (h : core.result.Result.Ok t = core.result.Result.Err e) : False := by
  cases h

/-- An `Err` input is never an `Ok` input, in that order. -/
theorem err_ok_clash (t : Envelope) (e : StorageError)
    (h : core.result.Result.Err e = core.result.Result.Ok t) : False := by
  cases h

/-- The publication transition moves the exact tuple: it neither drops, reorders nor rewrites the
    plan, the reserved storage or the outcome, so two distinct success tuples cannot publish the
    same result. -/
theorem publish_preserves_the_exact_tuple (plan : Plan) (storage : Reserved) (outcome : Outcome)
    (plan' : Plan) (storage' : Reserved) (outcome' : Outcome) :
    body (core.result.Result.Ok (plan, storage, outcome))
      = body (core.result.Result.Ok (plan', storage', outcome')) →
    plan = plan' ∧ storage = storage' ∧ outcome = outcome' := fun h =>
  published_inj (ok_injective' h)

/-- The success guard is exact: the body publishes `Prepared` exactly on the `Ok` arm, carrying that
    arm's own three input components. So a refusal input never reaches the publication constructor,
    a success input always does, and the published components are the input components. -/
theorem publish_guard_exact (result : core.result.Result Envelope StorageError)
    (plan : Plan) (storage : Reserved) (outcome : Outcome) :
    body result = ok (publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome)
      ↔ ∃ t : Envelope, result = core.result.Result.Ok t ∧
        t.1 = plan ∧ t.2.1 = storage ∧ t.2.2 = outcome := by
  cases result with
  | Ok t =>
      constructor
      · intro h
        obtain ⟨h1, h2, h3⟩ := published_inj (ok_injective' h)
        exact ⟨t, rfl, h1, h2, h3⟩
      · rintro ⟨w, hw, h1, h2, h3⟩
        cases hw
        change (ok (publish_prepared_source.table.resources.PreparedResult.Prepared
            t.1 t.2.1 t.2.2) : Result PublishedWord)
          = ok (publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome)
        rw [h3, h2, h1]
  | Err e =>
      constructor
      · intro h
        exact absurd h (fun h' =>
          published_arm_clash plan storage outcome e
            (ok_injective' (h'.symm.trans (refusal_transition e))))
      · rintro ⟨t, ht, _, _, _⟩
        exact (err_ok_clash t e ht).elim

/-- The refusal guard is exact: the body publishes `Refused` exactly on the `Err` arm, carrying the
    input's own `StorageError` value. So a success input never reaches the refusal constructor, a
    refusal input always does, and the published cause is the input cause. -/
theorem refusal_guard_exact (result : core.result.Result Envelope StorageError)
    (e : StorageError) :
    body result = ok (publish_prepared_source.table.resources.PreparedResult.Refused e)
      ↔ result = core.result.Result.Err e := by
  cases result with
  | Ok t =>
      constructor
      · intro h
        exact absurd h (fun h' => published_arm_clash t.1 t.2.1 t.2.2 e (ok_injective' h'))
      · intro h
        exact (ok_err_clash t e h).elim
  | Err e' =>
      constructor
      · intro h
        have heq : publish_prepared_source.table.resources.PreparedResult.Refused e
            = publish_prepared_source.table.resources.PreparedResult.Refused e' :=
          ok_injective' (h.symm.trans (refusal_transition e'))
        cases heq
        rfl
      · intro h
        cases h
        exact refusal_transition _

/-- The success envelope a published value hands to the relocation and commit cores: the whole
    success tuple on the `Prepared` arm, and nothing at all on the refusal arm. -/
def envelopeOf : PublishedWord → Option (Plan × Reserved × Outcome)
  | publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome =>
      some (plan, storage, outcome)
  | publish_prepared_source.table.resources.PreparedResult.Refused _ => none

/-- The cause a published value reports: the `StorageError` on the refusal arm, and nothing at all
    on the publication arm. -/
def refusalOf : PublishedWord → Option StorageError
  | publish_prepared_source.table.resources.PreparedResult.Prepared _ _ _ => none
  | publish_prepared_source.table.resources.PreparedResult.Refused e => some e

/-- The refusal branch exposes no plan, no reserved storage, no reservation receipt, no locator
    receipt and no held carrier: the whole success envelope reads back as `none`, and the only
    thing the published value carries is the caller's own `StorageError`. -/
theorem refusal_publishes_only_the_error (e : StorageError) :
    WP.spec (body (core.result.Result.Err e))
      (fun p => envelopeOf p = none ∧ refusalOf p = some e) := by
  rw [refusal_transition]
  exact (WP.spec_ok _).mpr ⟨rfl, rfl⟩

/-- The publication branch exposes the whole success tuple and reports no refusal. -/
theorem publish_publishes_the_whole_envelope (t : Envelope) :
    WP.spec (body (core.result.Result.Ok t))
      (fun p => envelopeOf p = some t ∧ refusalOf p = none) := by
  rw [show body (core.result.Result.Ok t)
      = ok (publish_prepared_source.table.resources.PreparedResult.Prepared t.1 t.2.1 t.2.2) from rfl]
  exact (WP.spec_ok _).mpr ⟨rfl, rfl⟩

/-- The transition is total over the whole extracted input type: every input reaches a value node,
    so the body reaches neither an effect node nor a divergence node. -/
theorem body_total (result : core.result.Result Envelope StorageError) :
    ∃ p : PublishedWord, body result = ok p := by
  cases result with
  | Ok t =>
      exact ⟨publish_prepared_source.table.resources.PreparedResult.Prepared t.1 t.2.1 t.2.2, rfl⟩
  | Err e =>
      exact ⟨publish_prepared_source.table.resources.PreparedResult.Refused e, rfl⟩

/-- The body reaches a value node on every input under the Aeneas weak-precondition relation, so it
    produces no `Error` node at all: the only refusal it can take is the `Err` arm carrying a
    `StorageError`, never a checked-overflow or allocator refusal of its own. -/
theorem body_never_errors (result : core.result.Result Envelope StorageError) :
    WP.spec (body result) (fun _ => True) := by
  cases result with
  | Ok t =>
      rw [show body (core.result.Result.Ok t)
        = ok (publish_prepared_source.table.resources.PreparedResult.Prepared t.1 t.2.1 t.2.2) from rfl]
      exact (WP.spec_ok _).mpr trivial
  | Err e =>
      rw [show body (core.result.Result.Err e)
        = ok (publish_prepared_source.table.resources.PreparedResult.Refused e) from rfl]
      exact (WP.spec_ok _).mpr trivial

/-- The two arms are mutually exclusive: no input publishes both a success tuple and a refusal. -/
theorem arms_exclusive (result : core.result.Result Envelope StorageError)
    (plan : Plan) (storage : Reserved) (outcome : Outcome) (e : StorageError) :
    ¬ (body result = ok (publish_prepared_source.table.resources.PreparedResult.Prepared plan storage outcome)
      ∧ body result = ok (publish_prepared_source.table.resources.PreparedResult.Refused e)) := by
  cases result with
  | Ok t =>
      rintro ⟨h1, h2⟩
      exact published_arm_clash plan storage outcome e (ok_injective' (h1.symm.trans h2))
  | Err e' =>
      rintro ⟨h1, _⟩
      exact published_arm_clash plan storage outcome e'
        (ok_injective' (h1.symm.trans (refusal_transition e')))

/-- The two arms are exhaustive, and each one is reached from its own input shape: every input
    publishes either the success tuple or the refusal. This is the checkable form of "the match has
    exactly these two arms and no third". -/
theorem arms_exhaustive (result : core.result.Result Envelope StorageError) :
    (∃ t : Envelope, result = core.result.Result.Ok t ∧
      body result = ok (publish_prepared_source.table.resources.PreparedResult.Prepared
        t.1 t.2.1 t.2.2)) ∨
      (∃ e : StorageError, result = core.result.Result.Err e ∧
        body result = ok (publish_prepared_source.table.resources.PreparedResult.Refused e)) := by
  cases result with
  | Ok t => exact Or.inl ⟨t, rfl, rfl⟩
  | Err e => exact Or.inr ⟨e, rfl, rfl⟩

/-
Reachable witnesses.

The witnesses below are fully initialized values of the extracted envelope type: every numeric field
is a machine-word zero and every container is empty, so they are total terms and need no assumed
precondition. They witness reachability of each arm at the concrete stored carrier and `i64` value
word the production root uses. They do not claim that the production adapter produces exactly these
zero receipts.
-/

/-- The zero capacity bound. -/
def emptyBound : publish_prepared_source.table.resources.CapacityBound :=
  { required_len := 0#usize, observed_capacity := 0#usize }

/-- The zero region reservation: the region's single open-addressed slot-array bound, tags
    included. -/
def emptyRegionReservation : publish_prepared_source.table.resources.RegionReservation :=
  { slots := emptyBound }

/-- The 256-entry region array of a reservation receipt, every entry the zero reservation. -/
def emptyRegions : Array publish_prepared_source.table.resources.RegionReservation 256#usize :=
  Array.make 256#usize (List.replicate 256 emptyRegionReservation)
    (by simp only [List.length_replicate]; rfl)

/-- The zero reservation receipt of the materialized table storage. -/
def emptyReservation : publish_prepared_source.table.resources.ReservationReceipt :=
  { flat := emptyBound, directory := emptyBound, regions := emptyRegions }

/-- The zero locator reservation receipt. -/
def emptyLocator : publish_prepared_source.table.resources.LocatorReservationReceipt :=
  { parabolic_nodes := emptyBound, residue_nodes := emptyBound, identity_nodes := emptyBound,
    identity_edges := emptyBound, representatives := emptyBound, reverse_links := emptyBound,
    staging := emptyBound, retained_identity := emptyBound }

/-- The empty checked plan: no lanes, no entries, no spill loads. -/
def emptyPlan : Plan :=
  { lanes := 0#usize,
    entries := alloc.vec.Vec.from [] (by simp),
    spill_loads := alloc.vec.Vec.from [] (by simp) }

/-- The empty materialized reserved storage: no flat slot and no region. -/
def emptyReserved : Reserved :=
  { resident := { lanes := 0#usize, slots := alloc.vec.Vec.from [] (by simp) },
    regions := alloc.vec.Vec.from [] (by simp) }

/-- The zero outcome: the zero receipt pair and the pending held carrier for word zero. -/
def emptyOutcome : Outcome :=
  { reservation := emptyReservation, locator := emptyLocator,
    pending_held := { key := 0#i64 } }

/-- The fully initialized empty success envelope. -/
def emptyEnvelope : Envelope := (emptyPlan, emptyReserved, emptyOutcome)

/-- A reachable success witness: the fully initialized empty envelope is an `Ok` input the real body
    publishes unchanged, exposing that whole envelope and no refusal. -/
theorem success_witness :
    WP.spec (body (core.result.Result.Ok emptyEnvelope))
      (fun p => p = publish_prepared_source.table.resources.PreparedResult.Prepared
          emptyPlan emptyReserved emptyOutcome ∧
        envelopeOf p = some emptyEnvelope ∧ refusalOf p = none) := by
  rw [show body (core.result.Result.Ok emptyEnvelope)
      = ok (publish_prepared_source.table.resources.PreparedResult.Prepared
          emptyPlan emptyReserved emptyOutcome) from rfl]
  exact (WP.spec_ok _).mpr ⟨rfl, rfl, rfl⟩

/-- The success witness as an equation on the real body. -/
theorem success_witness_equation :
    body (core.result.Result.Ok emptyEnvelope)
      = ok (publish_prepared_source.table.resources.PreparedResult.Prepared
          emptyPlan emptyReserved emptyOutcome) := rfl

/-- A reachable refusal witness: a shard-full reservation failure is an `Err` input the real body
    refuses with that exact cause and with no envelope attached. -/
theorem refusal_witness :
    WP.spec (body (core.result.Result.Err
        publish_prepared_source.table.StorageError.ShardFull))
      (fun p => envelopeOf p = none ∧
        refusalOf p = some publish_prepared_source.table.StorageError.ShardFull) :=
  refusal_publishes_only_the_error publish_prepared_source.table.StorageError.ShardFull

/-- Every `StorageError` variant is a reachable refusal input, so no refusal cause is vacuous. -/
theorem refusal_witnesses_cover_every_variant (e : StorageError) :
    WP.spec (body (core.result.Result.Err e))
      (fun p => envelopeOf p = none ∧ refusalOf p = some e) :=
  refusal_publishes_only_the_error e

/-- The success witness reaches the extraction root, so the production-shaped call publishes its
    envelope. -/
theorem root_success_witness :
    publish_prepared_source.word_stored_publish_prepared (core.result.Result.Ok emptyEnvelope)
      = ok (publish_prepared_source.table.resources.PreparedResult.Prepared
          emptyPlan emptyReserved emptyOutcome) :=
  root_forwards _ ▸ success_witness_equation

/-- The refusal witness reaches the extraction root, so the production-shaped call refuses with the
    caller's own cause. -/
theorem root_refusal_witness :
    publish_prepared_source.word_stored_publish_prepared
        (core.result.Result.Err publish_prepared_source.table.StorageError.ShardFull)
      = ok (publish_prepared_source.table.resources.PreparedResult.Refused
          publish_prepared_source.table.StorageError.ShardFull) :=
  root_forwards _ ▸ refusal_transition publish_prepared_source.table.StorageError.ShardFull

end PublishPreparedResource

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

#resource_audit PublishPreparedResource.root_forwards
#resource_audit PublishPreparedResource.body_is_one_constant_switch
#resource_audit PublishPreparedResource.work_is_one_move_no_comparison_no_allocation
#resource_audit PublishPreparedResource.work_is_independent_of_the_envelope
#resource_audit PublishPreparedResource.work_is_independent_of_the_arm
#resource_audit PublishPreparedResource.publish_transition
#resource_audit PublishPreparedResource.refusal_transition
#resource_audit PublishPreparedResource.publish_preserves_the_exact_tuple
#resource_audit PublishPreparedResource.publish_guard_exact
#resource_audit PublishPreparedResource.refusal_guard_exact
#resource_audit PublishPreparedResource.refusal_publishes_only_the_error
#resource_audit PublishPreparedResource.publish_publishes_the_whole_envelope
#resource_audit PublishPreparedResource.body_total
#resource_audit PublishPreparedResource.body_never_errors
#resource_audit PublishPreparedResource.arms_exclusive
#resource_audit PublishPreparedResource.arms_exhaustive
#resource_audit PublishPreparedResource.success_witness
#resource_audit PublishPreparedResource.success_witness_equation
#resource_audit PublishPreparedResource.refusal_witness
#resource_audit PublishPreparedResource.refusal_witnesses_cover_every_variant
#resource_audit PublishPreparedResource.root_success_witness
#resource_audit PublishPreparedResource.root_refusal_witness

run_cmd Lean.logInfo "PUBLISH_PREPARED_KERNEL_OK"