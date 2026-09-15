import Slang.Audit.Empty
import Slang.Derivation.Steps
import Slang.Derivation.Lexing
import Slang.Derivation.Selection
import Slang.Derivation.Validation
import Slang.Decode

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Derivation.DecodeTrace

open Slang Slang.Decode

/-- Requests name the recursive work of the decoder instead of evaluating it. -/
inductive Request where
  | utf8Fuel (fuel : Nat) (bytes : List UInt8)
  | lexing (q : Slang.Derivation.Lexing.Request)
  | selection (q : Slang.Derivation.Selection.Request)
  | validation (q : Slang.Derivation.Validation.Request)

/-- Answer type per request. -/
abbrev Answer : Request → Type
  | .utf8Fuel _ _ => Result (List Char)
  | .lexing q => Slang.Derivation.Lexing.Answer q
  | .selection q => Slang.Derivation.Selection.Answer q
  | .validation q => Slang.Derivation.Validation.Answer q

/-- The real implementation, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .utf8Fuel fuel bytes => Decode.decodeUtf8CharsFuel fuel bytes
  | .lexing q => Slang.Derivation.Lexing.evaluate q
  | .selection q => Slang.Derivation.Selection.evaluate q
  | .validation q => Slang.Derivation.Validation.evaluate q

/-- One unfolding of the implementation, exposing recursive calls as requests. -/
def body : (q : Request) → Layer Request Answer (Answer q)
  | .utf8Fuel fuel bytes =>
      match fuel, bytes with
      | 0, _ => .done (.error "source is not UTF-8")
      | _ + 1, [] => .done (.ok [])
      | fuel + 1, first :: rest =>
          let b0 := first.toNat
          if b0 < 0x80 then
            .ask (.utf8Fuel fuel rest) (fun r => .done (r.bind fun tail => .ok (Char.ofNat b0 :: tail)))
          else if b0 < 0xe0 then
            match rest with
            | [] => .done (.error "source is not UTF-8")
            | second :: rest =>
                let b1 := second.toNat
                let code := (b0 - 0xc0) * 0x40 + (b1 - 0x80)
                if b0 < 0xc2 || !Decode.isContinuationByte b1 then
                  .done (.error "source is not UTF-8")
                else
                  .ask (.utf8Fuel fuel rest)
                    (fun r => .done (r.bind fun tail => .ok (Char.ofNat code :: tail)))
          else if b0 < 0xf0 then
            match rest with
            | [] => .done (.error "source is not UTF-8")
            | second :: afterSecond =>
                match afterSecond with
                | [] => .done (.error "source is not UTF-8")
                | third :: rest =>
                    let b1 := second.toNat
                    let b2 := third.toNat
                    let code := (b0 - 0xe0) * 0x1000 + (b1 - 0x80) * 0x40 + (b2 - 0x80)
                    if !Decode.isContinuationByte b1 || !Decode.isContinuationByte b2 || code < 0x800 ||
                        (0xd800 ≤ code && code < 0xe000) then
                      .done (.error "source is not UTF-8")
                    else
                      .ask (.utf8Fuel fuel rest)
                        (fun r => .done (r.bind fun tail => .ok (Char.ofNat code :: tail)))
          else if b0 < 0xf5 then
            match rest with
            | [] => .done (.error "source is not UTF-8")
            | second :: afterSecond =>
                match afterSecond with
                | [] => .done (.error "source is not UTF-8")
                | third :: afterThird =>
                    match afterThird with
                    | [] => .done (.error "source is not UTF-8")
                    | fourth :: rest =>
                        let b1 := second.toNat
                        let b2 := third.toNat
                        let b3 := fourth.toNat
                        let code := (b0 - 0xf0) * 0x40000 + (b1 - 0x80) * 0x1000 +
                          (b2 - 0x80) * 0x40 + (b3 - 0x80)
                        if !Decode.isContinuationByte b1 || !Decode.isContinuationByte b2 ||
                            !Decode.isContinuationByte b3 || code < 0x10000 || 0x110000 ≤ code then
                          .done (.error "source is not UTF-8")
                        else
                          .ask (.utf8Fuel fuel rest)
                            (fun r => .done (r.bind fun tail => .ok (Char.ofNat code :: tail)))
          else .done (.error "source is not UTF-8")
  | .lexing q => .done (Slang.Derivation.Lexing.evaluate q)
  | .selection q => .done (Slang.Derivation.Selection.evaluate q)
  | .validation q => .done (Slang.Derivation.Validation.evaluate q)

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- The unfolding entails the implementation result. -/
theorem body_eq : ∀ q, evaluate q = fold evaluate (body q)
  | .utf8Fuel fuel bytes => by
      cases fuel with
      | zero => rfl
      | succ fuel =>
        cases bytes with
        | nil => rfl
        | cons first rest =>
          cases rest with
          | nil =>
            simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.DecodeTrace.body,
              Decode.decodeUtf8CharsFuel]
            repeat' first | rfl | split
            all_goals
              (rw [Slang.Derivation.fold_ask]
               simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.fold_done]
               rfl)
          | cons second afterSecond =>
            cases afterSecond with
            | nil =>
              simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.DecodeTrace.body,
                Decode.decodeUtf8CharsFuel]
              repeat' first | rfl | split
              all_goals
                (rw [Slang.Derivation.fold_ask]
                 simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.fold_done]
                 rfl)
            | cons third afterThird =>
              cases afterThird with
              | nil =>
                simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.DecodeTrace.body,
                  Decode.decodeUtf8CharsFuel]
                repeat' first | rfl | split
                all_goals
                  (rw [Slang.Derivation.fold_ask]
                   simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.fold_done]
                   rfl)
              | cons fourth rest =>
                simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.DecodeTrace.body,
                  Decode.decodeUtf8CharsFuel]
                repeat' first | rfl | split
                all_goals
                  (rw [Slang.Derivation.fold_ask]
                   simp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.fold_done]
                   rfl)
  | .lexing q => rfl
  | .selection q => rfl
  | .validation q => rfl

/-- A checked derivation of the real tokenizer's byte-level result. -/
theorem Run.sound {q : Request} {answer : Answer q}
    (derivation : Run q answer) : evaluate q = answer :=
  Slang.Derivation.call_sound evaluate body_eq derivation

/-- Transfer a `Lexing` derivation into the `DecodeTrace` lexing request. -/
theorem lexing_of_run {q : Slang.Derivation.Lexing.Request}
    {answer : Slang.Derivation.Lexing.Answer q}
    (derivation : Slang.Derivation.Lexing.Run q answer) :
    Run (.lexing q) answer :=
  Slang.Derivation.Derives.done_eq (Slang.Derivation.Lexing.Run.sound derivation)

/-- Transfer a `Selection` derivation into the `DecodeTrace` selection request. -/
theorem selection_of_run {q : Slang.Derivation.Selection.Request}
    {answer : Slang.Derivation.Selection.Answer q}
    (derivation : Slang.Derivation.Selection.Run q answer) :
    Run (.selection q) answer :=
  Slang.Derivation.Derives.done_eq (Slang.Derivation.Selection.Run.sound derivation)

/-- Transfer a `Validation` derivation into the `DecodeTrace` validation request. -/
theorem validation_of_run {q : Slang.Derivation.Validation.Request}
    {answer : Slang.Derivation.Validation.Answer q}
    (derivation : Slang.Derivation.Validation.Run q answer) :
    Run (.validation q) answer :=
  Slang.Derivation.Derives.done_eq (Slang.Derivation.Validation.Run.sound derivation)

/-- Raw bytes of the two-byte U+00E9 fixture. -/
abbrev multibyteBytes : List UInt8 := [0xc3, 0xa9]

/-- A concrete checked derivation of the decoder on the U+00E9 fixture. -/
theorem multibyte_run : Run (.utf8Fuel 3 multibyteBytes) (Except.ok ['é']) :=
  Derives.ask (Derives.done (Except.ok [])) (Derives.done (Except.ok ['é']))

/-- The real implementation result for the U+00E9 fixture. -/
theorem multibyte_result : evaluate (.utf8Fuel 3 multibyteBytes) = Except.ok ['é'] :=
  Run.sound multibyte_run

/-- A wrong answer for the U+00E9 fixture is not derivable. -/
theorem multibyte_wrong :
    ¬ Run (.utf8Fuel 3 multibyteBytes) (Except.error "source is not UTF-8") := by
  intro derivation
  have sound := Run.sound derivation
  rw [multibyte_result] at sound
  cases sound

#audit_empty Slang.Derivation.DecodeTrace.evaluate : (q : Slang.Derivation.DecodeTrace.Request) → Slang.Derivation.DecodeTrace.Answer q
#audit_empty Slang.Derivation.DecodeTrace.body_eq : ∀ (q : Slang.Derivation.DecodeTrace.Request), Slang.Derivation.DecodeTrace.evaluate q = Slang.Derivation.fold Slang.Derivation.DecodeTrace.evaluate (Slang.Derivation.DecodeTrace.body q)
#audit_empty Slang.Derivation.DecodeTrace.Run.sound : ∀ {q : Slang.Derivation.DecodeTrace.Request} {answer : Slang.Derivation.DecodeTrace.Answer q}, Slang.Derivation.DecodeTrace.Run q answer → Slang.Derivation.DecodeTrace.evaluate q = answer
#audit_empty Slang.Derivation.DecodeTrace.multibyte_run : Slang.Derivation.DecodeTrace.Run (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) (Except.ok ['é'])
#audit_empty Slang.Derivation.DecodeTrace.multibyte_result : Slang.Derivation.DecodeTrace.evaluate (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) = Except.ok ['é']
#audit_empty Slang.Derivation.DecodeTrace.multibyte_wrong : ¬Slang.Derivation.DecodeTrace.Run (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) (Except.error "source is not UTF-8")

end Slang.Derivation.DecodeTrace
