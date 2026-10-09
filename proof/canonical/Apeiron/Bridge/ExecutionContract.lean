import Apeiron.Combinator.Verdict
import Apeiron.Decider.Stream

/-!
# Execution contracts for the evaluator and certificate reader

An evaluator can replace an arena while returning a term index. A failed allocation can also
change that arena. `Completion` carries the post-state in both cases, and keeps a refusal reason
separate from the canonical evaluator's `.fuel` answer.

`Evaluator.Refines` relates each admitted source request to the corresponding canonical request.
On success it requires the exact canonical result under the post-state representation relation.
On refusal it requires the declared refusal relation. The source representation, admission,
invariant and refusal relations remain obligations of an implementation.

`Reader.Refines` covers an entire input. Acceptance requires successful decoding, resource
admission and acceptance by `checkCertRecords`. A decoded, admitted input rejected by the source
must also be rejected by the canonical checker. Decode failure and resource exclusion therefore
cannot be reported as acceptance. Decoding includes the consumer's complete-input policy.

The canonical instances use a Boolean resource state: one successful call consumes its resource,
and a call without the resource refuses. They establish joint witnesses for these contracts;
they do not establish a representation relation for another implementation.

## Root

| from | here |
|---|---|
| `Comb.verdict_decided_exact` | `Evaluator.Refines.verdict_decided_exact` |
| `Decider.checkCertRecords_sound` | `Reader.Refines.accepted_sound` |

These interfaces introduce no algebraic operation. Their semantic conclusions inherit the roots
of the canonical evaluator and certificate checker. An external execution semantics and its
representation proof are required to instantiate them for a consumer.
-/

namespace Apeiron.Bridge.Execution

open Combinator Combinator.Comb Decider

/-- An observed execution, including its state after success or refusal.

    State Value Reason                       [state, returned value and refusal types] -/
inductive Completion (State Value Reason : Type) where
  | returned (state : State) (value : Value)
  | refused (state : State) (reason : Reason)
  deriving DecidableEq, Repr

/-- An evaluator entry and its arguments; normalization passes are explicit.

    Term                                     [the implementation's term type] -/
inductive Request (Term : Type) where
  | headReduce (term : Term)
  | normalize (passes : ℕ) (term : Term)
  | hasRedex (term : Term)
  | verdict (passes : ℕ) (left right : Term)
  deriving DecidableEq, Repr

/-- The three result types of evaluator entries.

    Term                                     [the implementation's term type] -/
inductive Value (Term : Type) where
  | term (term : Term)
  | redex (found : Bool)
  | verdict (answer : Verdict)
  deriving DecidableEq, Repr

/-- Evaluate a canonical request using the named canonical entry.

    Request Comb → Value Comb                 [a canonical request and its result] -/
def evaluate : Request Comb → Value Comb
  | .headReduce term => .term (Comb.headReduce term)
  | .normalize passes term => .term (Comb.normalize passes term)
  | .hasRedex term => .redex (Comb.hasRedex term)
  | .verdict passes left right => .verdict (Comb.verdict passes left right)

/-- Match request entries, pass counts and represented input terms.

    (rep : Term → Comb → Prop)                [the term representation at the initial state]
    Request Term → Request Comb → Prop        [matching source and canonical requests] -/
def Request.Related {Term : Type} (rep : Term → Comb → Prop) :
    Request Term → Request Comb → Prop
  | .headReduce source, .headReduce term => rep source term
  | .normalize count source, .normalize passes term => count = passes ∧ rep source term
  | .hasRedex source, .hasRedex term => rep source term
  | .verdict count left right, .verdict passes first second =>
      count = passes ∧ rep left first ∧ rep right second
  | _, _ => False

/-- Match result tags and scalar values, and represent returned terms in the final state.

    (rep : Term → Comb → Prop)                [the term representation at the final state]
    Value Term → Value Comb → Prop            [matching source and canonical results] -/
def Value.Related {Term : Type} (rep : Term → Comb → Prop) :
    Value Term → Value Comb → Prop
  | .term source, .term canonical => rep source canonical
  | .redex source, .redex found => source = found
  | .verdict source, .verdict answer => source = answer
  | _, _ => False

/-- The parameters supplied by an executing evaluator implementation.

    State Term RawResult Reason              [source state, terms, raw result and refusal types] -/
structure Evaluator (State Term RawResult Reason : Type) where
  /-- The invariant on source states, including states after refusal. -/
  invariant : State → Prop
  /-- A source term's canonical meaning in its current state. -/
  rep : State → Term → Comb → Prop
  /-- The domain on which the entry's success or declared refusal is specified. -/
  admitted : State → Request Term → Prop
  /-- Permitted refusal reasons and post-states for this request. -/
  refusal : State → Request Term → State → Reason → Prop
  /-- The executing source body, with its raw result. -/
  run : State → Request Term → RawResult
  /-- Decode the raw result without identifying resource refusal with canonical `.fuel`. -/
  observe : RawResult → Completion State (Value Term) Reason

/-- Exact successful results and declared refusals on represented, admitted inputs.

    (impl : Evaluator State Term RawResult Reason) [the executing evaluator interface]
    : Prop                                    [the implementation's proof obligation]
    ∀                                         [for every admitted represented request]
    →                                         [the hypotheses require the stated conclusion]
    ∧                                         [both conditions hold] -/
def Evaluator.Refines {State Term RawResult Reason : Type}
    (impl : Evaluator State Term RawResult Reason) : Prop :=
  ∀ before source canonical, impl.invariant before → impl.admitted before source →
    Request.Related (impl.rep before) source canonical →
    match impl.observe (impl.run before source) with
    | .returned after result =>
        impl.invariant after ∧ Value.Related (impl.rep after) result (evaluate canonical)
    | .refused after reason =>
        impl.invariant after ∧ impl.refusal before source after reason

/-- A represented, admitted and successful verdict inherits canonical decided exactness.

    (h : impl.Refines)                        [the execution contract is proved]
    impl.invariant before                    [the initial source state is valid]
    impl.admitted before (.verdict passes left right) [the source request is admitted]
    impl.rep before left first               [the left input represents the first term]
    impl.rep before right second             [the right input represents the second term]
    impl.observe (impl.run before (.verdict passes left right)) =
      .returned after (.verdict answer)       [the source execution returns this verdict]
    answer ≠ .fuel                           [the canonical computation was decided]
    answer = .equal ↔ Joinable first second   [equality is exactly a shared reduct]
    {passes : ℕ} {first second : Comb}        [the pass count and represented terms]
    {answer : Verdict}                       [the equal, distinct or fuel answer] -/
theorem Evaluator.Refines.verdict_decided_exact {State Term RawResult Reason : Type}
    {impl : Evaluator State Term RawResult Reason} (h : impl.Refines)
    {before after : State} {passes : ℕ} {left right : Term} {first second : Comb}
    {answer : Verdict} (hstate : impl.invariant before)
    (hadmit : impl.admitted before (.verdict passes left right))
    (hleft : impl.rep before left first) (hright : impl.rep before right second)
    (hrun : impl.observe (impl.run before (.verdict passes left right)) =
      .returned after (.verdict answer)) (hdecided : answer ≠ .fuel) :
    answer = .equal ↔ Joinable first second := by
  have hexact := h before (.verdict passes left right) (.verdict passes first second)
    hstate hadmit ⟨rfl, hleft, hright⟩
  rw [hrun] at hexact
  have heq : answer = Comb.verdict passes first second := hexact.2
  rw [heq] at hdecided ⊢
  exact Comb.verdict_decided_exact hdecided

/-- The parameters supplied by an executing complete-input certificate reader.

    State Input RawResult Reason             [source state, input, raw result and refusal types] -/
structure Reader (State Input RawResult Reason : Type) where
  /-- The invariant on source states, including states after refusal. -/
  invariant : State → Prop
  /-- Decode the entire source input according to its stated consumption policy. -/
  decode : State → Input → Option (List CertRecord)
  /-- The resource domain on which decoded input must match the canonical checker exactly. -/
  admitted : State → Input → Prop
  /-- Permitted refusal reasons and post-states. -/
  refusal : State → Input → State → Reason → Prop
  /-- The executing source reader and checker. -/
  run : State → Input → RawResult
  /-- A successful outcome is acceptance; all other outcomes retain their refusal reason. -/
  observe : RawResult → Completion State Unit Reason

/-- Exact complete-input acceptance with explicit decoding and resource refusal.

    (impl : Reader State Input RawResult Reason) [the executing reader interface]
    : Prop                                    [the implementation's proof obligation]
    ∀                                         [for every input in a valid initial state]
    →                                         [the hypotheses require the stated conclusion]
    ∃                                         [there is a decoded record sequence]
    ∧                                         [all stated conditions hold]
    List CertRecord                           [the decoded record sequence] -/
def Reader.Refines {State Input RawResult Reason : Type}
    (impl : Reader State Input RawResult Reason) : Prop :=
  ∀ before input, impl.invariant before →
    match impl.observe (impl.run before input) with
    | .returned after _ => impl.invariant after ∧ ∃ records,
        impl.decode before input = some records ∧ impl.admitted before input ∧
          checkCertRecords records = true
    | .refused after reason => impl.invariant after ∧ impl.refusal before input after reason ∧
        ∀ records, impl.decode before input = some records → impl.admitted before input →
          checkCertRecords records = false

/-- Source acceptance is exactly decoded, admitted canonical acceptance.

    (h : impl.Refines)                        [the execution contract is proved]
    impl.invariant before                    [the initial source state is valid]
    ∃ after, impl.observe (impl.run before input) = .returned after () [source acceptance]
    ↔ ∃ records, impl.decode before input = some records [exactly decoded input]
      ∧ impl.admitted before input            [within the source resource domain]
      ∧ checkCertRecords records = true        [accepted by the canonical checker] -/
theorem Reader.Refines.accepts_iff {State Input RawResult Reason : Type}
    {impl : Reader State Input RawResult Reason} (h : impl.Refines)
    {before : State} {input : Input} (hstate : impl.invariant before) :
    (∃ after, impl.observe (impl.run before input) = .returned after ()) ↔
      ∃ records, impl.decode before input = some records ∧ impl.admitted before input ∧
        checkCertRecords records = true := by
  have href := h before input hstate
  cases hout : impl.observe (impl.run before input) with
  | returned after value =>
      cases value
      rw [hout] at href
      exact ⟨fun _ => href.2, fun _ => ⟨after, rfl⟩⟩
  | refused after reason =>
      rw [hout] at href
      constructor
      · rintro ⟨_, hbad⟩
        contradiction
      · rintro ⟨records, hdecode, hadmit, hcheck⟩
        have hfalse := href.2.2 records hdecode hadmit
        rw [hcheck] at hfalse
        contradiction

/-- Successful source checking inherits canonical framing and certificate soundness.

    (h : impl.Refines)                        [the execution contract is proved]
    impl.invariant before                    [the initial source state is valid]
    impl.observe (impl.run before input) = .returned after () [source acceptance]
    ∃ records frames                         [a decoded stream and its frames exist]
    impl.decode before input = some records   [the decoder produces exactly these records]
    ∧ CertRecord.Frames records frames        [the canonical grammar frames them]
    ∧ ∀ frame ∈ frames                       [every frame has no satisfying assignment]
      ¬ ∃ ρ : Fin frame.columns → ℤ, SatLines ρ frame.raw [its system is unsatisfiable] -/
theorem Reader.Refines.accepted_sound {State Input RawResult Reason : Type}
    {impl : Reader State Input RawResult Reason} (h : impl.Refines)
    {before after : State} {input : Input} (hstate : impl.invariant before)
    (hrun : impl.observe (impl.run before input) = .returned after ()) :
    ∃ records frames, impl.decode before input = some records ∧
      CertRecord.Frames records frames ∧
      ∀ frame ∈ frames, ¬ ∃ ρ : Fin frame.columns → ℤ, SatLines ρ frame.raw := by
  obtain ⟨records, hdecode, _, hcheck⟩ := (h.accepts_iff hstate).mp ⟨after, hrun⟩
  obtain ⟨frames, hframes, hsound⟩ := checkCertRecords_sound hcheck
  exact ⟨records, frames, hdecode, hframes, hsound⟩

/-- A canonical evaluator with one consumable resource and a state-carrying refusal.

    Evaluator Bool Comb (Completion Bool (Value Comb) Unit) Unit [the witnessed interface] -/
def canonicalEvaluator : Evaluator Bool Comb (Completion Bool (Value Comb) Unit) Unit where
  invariant := fun _ => True
  rep := fun _ source canonical => source = canonical
  admitted := fun _ _ => True
  refusal := fun before _ after _ => before = false ∧ after = false
  run := fun available request =>
    if available then .returned false (evaluate request) else .refused false ()
  observe := id

/-- The canonical evaluator inhabits its execution contract.

    canonicalEvaluator.Refines                [all admitted requests satisfy the contract] -/
theorem canonicalEvaluator_refines : canonicalEvaluator.Refines := by
  intro before source canonical _ _ hrep
  cases before with
  | false => exact ⟨True.intro, rfl, rfl⟩
  | true =>
      refine ⟨True.intro, ?_⟩
      cases source <;> cases canonical <;>
        simp only [Request.Related, canonicalEvaluator] at hrep <;>
        simp_all [canonicalEvaluator, evaluate, Value.Related]

/-- The three refusal classes of the canonical reader witness. -/
inductive ReaderFailure where
  | decode
  | resource
  | rejected
  deriving DecidableEq, Repr

/-- A canonical complete-input reader with decoding and resource refusals.

    Reader Bool (Option (List CertRecord))    [resource state and already decoded optional input]
      (Completion Bool Unit ReaderFailure) ReaderFailure [raw outcomes and refusal reasons] -/
def canonicalReader : Reader Bool (Option (List CertRecord))
    (Completion Bool Unit ReaderFailure) ReaderFailure where
  invariant := fun _ => True
  decode := fun _ input => input
  admitted := fun available _ => available = true
  refusal := fun available input after reason => after = false ∧
    match reason with
    | .decode => input = none
    | .resource => input.isSome = true ∧ available = false
    | .rejected => ∃ records, input = some records ∧ available = true ∧
        checkCertRecords records = false
  run := fun available input =>
    match input with
    | none => .refused false .decode
    | some records =>
        if available then
          if checkCertRecords records then .returned false () else .refused false .rejected
        else .refused false .resource
  observe := id

/-- The canonical reader inhabits its execution contract.

    canonicalReader.Refines                   [all inputs satisfy the stated acceptance contract] -/
theorem canonicalReader_refines : canonicalReader.Refines := by
  intro before input _
  cases input with
  | none => simp [canonicalReader]
  | some records =>
      cases before with
      | false => simp [canonicalReader]
      | true =>
          cases hc : checkCertRecords records <;> simp [canonicalReader, hc]

/-- A complete zero-column contradiction for the canonical reader witness.

    List CertRecord                           [decoded records for a contradictory system] -/
private def readerFixture : List CertRecord :=
  [.header 0, .ge (-1) [], .cert, .op (.load 0), .op (.neg 0),
    .op (.meet 0 1), .op (.close 2)]

#guard canonicalEvaluator.run true (.headReduce (.app (.app .K .S) .K)) =
  .returned false (.term .S)
#guard canonicalEvaluator.run false (.headReduce .S) = .refused false ()
#guard canonicalEvaluator.run true (.verdict 0 .I (.app (.app .S .K) .K)) =
  .returned false (.verdict .equal)
#guard canonicalReader.run true (some readerFixture) = .returned false ()
#guard canonicalReader.run true none = .refused false .decode
#guard canonicalReader.run false (some readerFixture) = .refused false .resource
#guard canonicalReader.run true (some []) = .refused false .rejected

end Apeiron.Bridge.Execution
