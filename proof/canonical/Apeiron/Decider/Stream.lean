import Apeiron.Decider.Cert

/-!
# Certificate streams

@name Certificate Streams
@grade proved

A certificate batch may change its column count between systems. `CertFrame` packages that count
with one wire system and its certificate log. A stream is a list of these dependent frames.

`checkCertStream` executes `checkCert` on every frame. `CertStream.Replays` is the relational
statement that every frame is accepted, and `checkCertStream_eq_true_iff` identifies the executable
fold with that relation. `checkCertStream_sound` applies `checkCert_sound` at each frame, without a
shared assignment type between frames.

Both `close` and the finite-tree `residue` constructor terminate a frame. Their semantic checks
run during replay; framing alone checks the record boundary and coefficient counts.

## Root

| from | here |
|---|---|
| `checkCert_sound` | `CertFrame.accepted_sound`, `checkCertStream_sound` |
| `certStep_holds`, `crImpossible_empty` | `CertFrame.Accepted`, through `checkCert` |

The stream layer introduces no algebraic operation. Its soundness is the dependent-list closure of
the certificate theorem, whose roots are recorded in `Apeiron.Decider.Cert`.
-/

namespace Apeiron
namespace Decider

/-- One framed system and certificate, with its column count.

    CertFrame                               [one framed certificate system]

    So: systems with different column counts can occupy one certificate stream. -/
structure CertFrame where
  /-- The number of columns in this frame. -/
  columns : ℕ
  /-- The system rows read before the certificate. -/
  raw : List (Line columns)
  /-- The certificate operations following the system rows. -/
  log : List CertOp

/-- Acceptance of one frame by the canonical certificate checker.

    (frame : CertFrame)                     [one framed system and certificate]
    : Prop                                  [a statement that is true or false]

    So: the frame replays when `checkCert` accepts its log from an empty row list. -/
def CertFrame.Accepted (frame : CertFrame) : Prop :=
  checkCert frame.raw [] frame.log = true

/-- An accepted frame contains an unsatisfiable system.

    (frame : CertFrame)                     [one framed system and certificate]
    frame.Accepted                          [the canonical checker accepts the frame]
    ¬ ∃ ρ : Fin frame.columns → ℤ,          [there is no whole-number assignment]
      SatLines ρ frame.raw                  [that satisfies every system line]

    So: framing changes no certificate meaning; this is `checkCert_sound` at the frame's width. -/
theorem CertFrame.accepted_sound (frame : CertFrame) (h : frame.Accepted) :
    ¬ ∃ ρ : Fin frame.columns → ℤ, SatLines ρ frame.raw :=
  checkCert_sound frame.raw frame.log h

/-- A certificate stream is a finite sequence of dependent frames.

    CertStream                              [a finite stream of certificate frames] -/
abbrev CertStream := List CertFrame

/-- The relational replay statement for a certificate stream.

    (frames : CertStream)                   [a stream of framed certificates]
    : Prop                                  [a statement that is true or false]

    So: every frame in the stream is accepted by the canonical checker. -/
def CertStream.Replays (frames : CertStream) : Prop :=
  ∀ frame ∈ frames, frame.Accepted

/-- Execute the canonical checker over every frame in a stream.

    CertStream → Bool                       [a certificate stream and its acceptance bit]

    So: the stream is accepted exactly when every frame is accepted. -/
def checkCertStream : CertStream → Bool
  | [] => true
  | frame :: frames =>
      checkCert frame.raw [] frame.log && checkCertStream frames

/-- **Executable stream checking is exactly relational replay.**

    (frames : CertStream)                   [a stream of framed certificates]
    checkCertStream frames = true           [the executable stream checker accepts]
    ↔ frames.Replays                        [exactly when every frame replays]

    So: the recursive Boolean fold implements the framewise replay relation. -/
theorem checkCertStream_eq_true_iff (frames : CertStream) :
    checkCertStream frames = true ↔ frames.Replays := by
  induction frames with
  | nil => simp [checkCertStream, CertStream.Replays]
  | cons frame frames ih =>
      simp only [checkCertStream, Bool.and_eq_true, ih, CertStream.Replays,
        List.mem_cons, forall_eq_or_imp]
      rfl

/-- **An accepted certificate stream contains only unsatisfiable systems.**

    (frames : CertStream)                               [a stream of framed certificates]
    checkCertStream frames = true                       [the stream checker accepts]
    ∀ frame ∈ frames,                                   [for every frame in the stream]
      ¬ ∃ ρ : Fin frame.columns → ℤ, SatLines ρ frame.raw [its system has no assignment]

    So: `checkCert_sound` applies independently at each frame's column count. -/
theorem checkCertStream_sound (frames : CertStream) (h : checkCertStream frames = true) :
    ∀ frame ∈ frames, ¬ ∃ ρ : Fin frame.columns → ℤ, SatLines ρ frame.raw := by
  have hreplay : frames.Replays := (checkCertStream_eq_true_iff frames).mp h
  intro frame hframe
  exact frame.accepted_sound (hreplay frame hframe)

open Fragment

/-- One decoded record in a certificate stream. This layer excludes byte decoding, whitespace,
    fixed capacities, machine-word bounds and wide limbs.

    CertRecord                              [one record after text tokenization] -/
inductive CertRecord : Type
  | header (columns : ℕ)
  | ge (const : ℤ) (coeff : List ℤ)
  | eq (const : ℤ) (coeff : List ℤ)
  | cert
  | op (operation : CertOp)
  deriving DecidableEq, Repr

/-- Decode a row only when its coefficient count matches the header.

    (columns : ℕ)                       [the column count from the frame header]
    (const : ℤ)                          [the row's constant term]
    (coeff : List ℤ)                     [the row's coefficient tokens]
    : Option (Row columns)                  [the row, or refusal on the wrong count] -/
def decodeRecordRow (columns : ℕ) (const : ℤ) (coeff : List ℤ) : Option (Row columns) :=
  if h : coeff.length = columns then
    some ⟨const, fun i => coeff.get ⟨i.val, by omega⟩⟩
  else none

/-- The state of record framing.

    RecordState                             [a position in the record grammar] -/
inductive RecordState : Type
  | header (frames : CertStream)
  | rows (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
  | operations (frames : CertStream) (columns : ℕ)
      (raw : List (Line columns)) (log : List CertOp)

/-- One executable transition of the record grammar.

    RecordState → CertRecord              [the current state and next record]
    → Option RecordState                 [the next state, or refusal] -/
def recordTransition : RecordState → CertRecord → Option RecordState
  | .header frames, .header columns => some (.rows frames columns [])
  | .rows frames columns raw, .ge const coeff =>
      match decodeRecordRow columns const coeff with
      | some row => some (.rows frames columns (raw ++ [.ge row]))
      | none => none
  | .rows frames columns raw, .eq const coeff =>
      match decodeRecordRow columns const coeff with
      | some row => some (.rows frames columns (raw ++ [.eq row]))
      | none => none
  | .rows frames columns raw, .cert => some (.operations frames columns raw [])
  | .operations frames columns raw log, .op (.close i) =>
      some (.header (frames ++ [⟨columns, raw, log ++ [.close i]⟩]))
  | .operations frames columns raw log, .op (.residue cert) =>
      some (.header (frames ++ [⟨columns, raw, log ++ [.residue cert]⟩]))
  | .operations frames columns raw log, .op operation =>
      some (.operations frames columns raw (log ++ [operation]))
  | _, _ => none

/-- The relational one-record framing judgment.

    RecordStep before record after           [one admitted grammar step]
    →                                       [takes the left and gives the right] -/
inductive RecordStep : RecordState → CertRecord → RecordState → Prop
  | header (frames : CertStream) (columns : ℕ) :
      RecordStep (.header frames) (.header columns) (.rows frames columns [])
  | ge (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (const : ℤ) (coeff : List ℤ) (row : Row columns)
      (hrow : decodeRecordRow columns const coeff = some row) :
      RecordStep (.rows frames columns raw) (.ge const coeff)
        (.rows frames columns (raw ++ [.ge row]))
  | eq (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (const : ℤ) (coeff : List ℤ) (row : Row columns)
      (hrow : decodeRecordRow columns const coeff = some row) :
      RecordStep (.rows frames columns raw) (.eq const coeff)
        (.rows frames columns (raw ++ [.eq row]))
  | cert (frames : CertStream) (columns : ℕ) (raw : List (Line columns)) :
      RecordStep (.rows frames columns raw) .cert (.operations frames columns raw [])
  | load (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (i : ℕ) :
      RecordStep (.operations frames columns raw log) (.op (.load i))
        (.operations frames columns raw (log ++ [.load i]))
  | combo (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (i j : ℕ) (a b : ℤ) :
      RecordStep (.operations frames columns raw log) (.op (.combo i j a b))
        (.operations frames columns raw (log ++ [.combo i j a b]))
  | neg (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (i : ℕ) :
      RecordStep (.operations frames columns raw log) (.op (.neg i))
        (.operations frames columns raw (log ++ [.neg i]))
  | meet (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (i j : ℕ) :
      RecordStep (.operations frames columns raw log) (.op (.meet i j))
        (.operations frames columns raw (log ++ [.meet i j]))
  | tidy (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (i : ℕ) :
      RecordStep (.operations frames columns raw log) (.op (.tidy i))
        (.operations frames columns raw (log ++ [.tidy i]))
  | close (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (i : ℕ) :
      RecordStep (.operations frames columns raw log) (.op (.close i))
        (.header (frames ++ [⟨columns, raw, log ++ [.close i]⟩]))
  | residue (frames : CertStream) (columns : ℕ) (raw : List (Line columns))
      (log : List CertOp) (cert : ResidueCertificate) :
      RecordStep (.operations frames columns raw log) (.op (.residue cert))
        (.header (frames ++ [⟨columns, raw, log ++ [.residue cert]⟩]))

/-- Every successful executable transition is admitted by the relational grammar.

    {before after : RecordState}             [the states before and after one record]
    {record : CertRecord}                   [the record being read]
    recordTransition before record = some after [the executable transition succeeds]
    RecordStep before record after          [the relational grammar admits that step] -/
theorem recordTransition_sound {before after : RecordState} {record : CertRecord}
    (h : recordTransition before record = some after) : RecordStep before record after := by
  cases before with
  | header frames =>
      cases record with
      | header columns =>
          simp only [recordTransition, Option.some.injEq] at h
          cases h
          exact .header frames columns
      | ge => simp [recordTransition] at h
      | eq => simp [recordTransition] at h
      | cert => simp [recordTransition] at h
      | op => simp [recordTransition] at h
  | rows frames columns raw =>
      cases record <;> simp [recordTransition] at h
      · next const coeff =>
          cases hrow : decodeRecordRow columns const coeff with
          | none => simp [hrow] at h
          | some row =>
              simp [hrow] at h
              cases h
              exact .ge frames columns raw const coeff row hrow
      · next const coeff =>
          cases hrow : decodeRecordRow columns const coeff with
          | none => simp [hrow] at h
          | some row =>
              simp [hrow] at h
              cases h
              exact .eq frames columns raw const coeff row hrow
      · cases h; exact .cert frames columns raw
  | operations frames columns raw log =>
      cases record <;> simp [recordTransition] at h
      next operation =>
        cases operation <;> simp only [Option.some.injEq] at h
        · next i => cases h; exact .load frames columns raw log i
        · next i j a b => cases h; exact .combo frames columns raw log i j a b
        · next i => cases h; exact .neg frames columns raw log i
        · next i j => cases h; exact .meet frames columns raw log i j
        · next i => cases h; exact .tidy frames columns raw log i
        · next i => cases h; exact .close frames columns raw log i
        · next cert => cases h; exact .residue frames columns raw log cert

/-- Relational framing of a list of records.

    RecordSteps before records after         [zero or more admitted grammar steps]
    List                                    [a finite sequence of things, in order]
    →                                       [takes the left and gives the right] -/
inductive RecordSteps : RecordState → List CertRecord → RecordState → Prop
  | nil (state : RecordState) : RecordSteps state [] state
  | cons {before middle after : RecordState} {record : CertRecord} {records : List CertRecord}
      (head : RecordStep before record middle) (tail : RecordSteps middle records after) :
      RecordSteps before (record :: records) after

/-- Execute record framing with an explicit transition bound.

    ℕ → RecordState → List CertRecord    [fuel, initial state and remaining records]
    → Option CertStream                    [the completed frames, or refusal] -/
def parseCertRecordsAux : ℕ → RecordState → List CertRecord → Option CertStream
  | 0, _, _ => none
  | _ + 1, .header frames, [] => some frames
  | _ + 1, _, [] => none
  | fuel + 1, state, record :: records =>
      match recordTransition state record with
      | some next => parseCertRecordsAux fuel next records
      | none => none

/-- Successful bounded parsing follows the relational framing judgment.

    {fuel : ℕ} {state : RecordState}        [the transition bound and initial state]
    {records : List CertRecord}               [the records being parsed]
    {frames : CertStream}                    [the completed frames returned]
    parseCertRecordsAux fuel state records = some frames [the bounded parser succeeds]
    RecordSteps state records (.header frames) [the grammar reaches the returned boundary] -/
theorem parseCertRecordsAux_sound {fuel : ℕ} {state : RecordState}
    {records : List CertRecord} {frames : CertStream}
    (h : parseCertRecordsAux fuel state records = some frames) :
    RecordSteps state records (.header frames) := by
  induction fuel generalizing state records frames with
  | zero => simp [parseCertRecordsAux] at h
  | succ fuel ih =>
      cases records with
      | nil =>
          cases state with
          | header current =>
              rw [parseCertRecordsAux] at h
              cases h
              exact .nil _
          | rows => simp [parseCertRecordsAux] at h
          | operations => simp [parseCertRecordsAux] at h
      | cons record records =>
          rw [parseCertRecordsAux] at h
          cases ht : recordTransition state record with
          | none => simp [ht] at h
          | some next =>
              rw [ht] at h
              exact .cons (recordTransition_sound ht) (ih h)

/-- Parse a complete record stream with one transition available per record.

    (records : List CertRecord)              [the complete decoded record stream]
    : Option CertStream                     [the framed systems and logs, or refusal] -/
def parseCertRecords (records : List CertRecord) : Option CertStream :=
  match records with
  | [] => none
  | _ => parseCertRecordsAux (records.length + 1) (.header []) records

/-- The relational meaning of a complete record stream.

    (records : List CertRecord)              [the complete decoded record stream]
    (frames : CertStream)                   [the frames it contains]
    : Prop                                  [whether the grammar relates them] -/
def CertRecord.Frames (records : List CertRecord) (frames : CertStream) : Prop :=
  RecordSteps (.header []) records (.header frames)

/-- Successful complete parsing has the relational framing meaning.

    {records : List CertRecord} {frames : CertStream} [records and their returned frames]
    parseCertRecords records = some frames             [the complete parser succeeds]
    records.Frames frames                              [the relational grammar admits them] -/
theorem parseCertRecords_sound {records : List CertRecord} {frames : CertStream}
    (h : parseCertRecords records = some frames) : CertRecord.Frames records frames :=
  by
    cases records with
    | nil => simp [parseCertRecords] at h
    | cons record records => exact parseCertRecordsAux_sound h

/-- Parse and replay a complete decoded record stream.

    List CertRecord → Bool                 [a decoded stream and its acceptance bit] -/
def checkCertRecords (records : List CertRecord) : Bool :=
  match parseCertRecords records with
  | some frames => checkCertStream frames
  | none => false

/-- Accepted decoded records frame exactly and contain only unsatisfiable systems.

    {records : List CertRecord}              [the complete decoded record stream]
    checkCertRecords records = true          [framing and replay both accept]
    ∃ frames                              [there is a returned dependent frame stream]
      records.Frames frames ∧              [the relational grammar admits exactly those frames]
      ∀ frame ∈ frames                 [for every returned frame]
        ¬ ∃ ρ : Fin frame.columns → ℤ     [there is no satisfying assignment] -/
theorem checkCertRecords_sound {records : List CertRecord} (h : checkCertRecords records = true) :
    ∃ frames, CertRecord.Frames records frames ∧
      ∀ frame ∈ frames, ¬ ∃ ρ : Fin frame.columns → ℤ, SatLines ρ frame.raw := by
  unfold checkCertRecords at h
  cases hp : parseCertRecords records with
  | none => simp [hp] at h
  | some frames =>
      refine ⟨frames, parseCertRecords_sound hp, checkCertStream_sound frames ?_⟩
      simpa [hp] using h

/-- A one-column contradiction used by the parser and replay build guards.

    : List CertRecord                       [one complete decoded certificate frame] -/
private def framingFixture : List CertRecord :=
  [.header 1, .ge (-1) [0], .cert, .op (.load 0), .op (.neg 0),
    .op (.meet 0 1), .op (.close 2)]

#guard (parseCertRecords framingFixture).isSome
#guard checkCertRecords framingFixture
#guard parseCertRecords [] = none
#guard parseCertRecords [.header 1, .ge (-1) [], .cert, .op (.close 0)] = none
#guard parseCertRecords [.header 1, .ge (-1) [0], .op (.close 0)] = none
#guard parseCertRecords [.header 1, .ge (-1) [0]] = none
#guard parseCertRecords [.header 1, .cert, .ge (-1) [0], .op (.close 0)] = none
#guard parseCertRecords [.header 1, .ge (-1) [0], .cert, .op (.load 0)] = none
#guard parseCertRecords (framingFixture ++ [.ge 0 [0]]) = none
#guard checkCertRecords [.header 1, .eq (-1) [2], .cert,
  .op (.residue (synthesizeResidueCertificate 1
    (loadRows [Line.eq (⟨-1, ![2]⟩ : Row 1)])))]
#guard !checkCertRecords [.header 1, .eq (-1) [2], .cert, .op (.residue (.split []))]
#guard parseCertRecords [.header 0, .ge (-1) [], .cert,
  .op (.residue (.leaf 0)), .op (.load 0)] = none

end Decider
end Apeiron
