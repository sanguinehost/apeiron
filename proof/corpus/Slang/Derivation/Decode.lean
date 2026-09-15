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
  | decodeTokens (bytes : ByteArray) (tokens : List Token) (consumed : Nat)
  | decodeTable (bytes : ByteArray)
  | importedCoord (bytes : ByteArray)
  | closure (table std : ByteArray)

/-- Answer type per request. -/
abbrev Answer : Request → Type
  | .utf8Fuel _ _ => Result (List Char)
  | .lexing q => Slang.Derivation.Lexing.Answer q
  | .selection q => Slang.Derivation.Selection.Answer q
  | .validation q => Slang.Derivation.Validation.Answer q
  | .decodeTokens _ _ _ => Result CertifiedDecoded
  | .decodeTable _ => Result CertifiedDecoded
  | .importedCoord _ => Result ConstantDecl
  | .closure _ _ => Result CertifiedDecoded

/-- The real implementation, indexed by request. -/
def evaluate : (q : Request) → Answer q
  | .utf8Fuel fuel bytes => Decode.decodeUtf8CharsFuel fuel bytes
  | .lexing q => Slang.Derivation.Lexing.evaluate q
  | .selection q => Slang.Derivation.Selection.evaluate q
  | .validation q => Slang.Derivation.Validation.evaluate q
  | .decodeTokens bytes tokens consumed => Decode.decodeTokens bytes tokens consumed
  | .decodeTable bytes => Decode.decodeTable bytes
  | .importedCoord bytes => Decode.decodeImportedCoordCeiling bytes
  | .closure table std => Decode.decodeClosure table std

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
  | .decodeTokens bytes tokens consumed =>
      if consumed != bytes.size then
        .done (.error "tokenizer did not consume every source byte")
      else
        .ask (.validation (.delimiters tokens [])) (fun balanced =>
          if !balanced then
            .done (.error "unbalanced source declaration delimiter")
          else
            .ask (.selection (.scan (tokens.length + 1) tokens Decode.selectedConstantNames
                Decode.selectedFunctionNames)) (fun scanned =>
              match scanned with
              | .error message => .done (.error message)
              | .ok (constants, functions) =>
                  let constantNames := constants.map (·.name)
                  let functionNames := functions.map (·.name)
                  if constantNames.length != 8 then
                    .done (.error "selected table constants are missing or duplicated")
                  else if functionNames.length != Decode.selectedFunctionNames.length then
                    .done (.error "selected functions are missing or duplicated")
                  else
                    .ask (.validation (.distinct (constantNames ++ ["COORD_CEILING"])))
                      (fun constantsDistinct =>
                        .ask (.validation (.distinct functionNames)) (fun functionsDistinct =>
                          if !constantsDistinct || !functionsDistinct then
                            .done (.error "selected declaration names are not unique")
                          else
                            .ask (.validation (.spans bytes.size
                                (constants.map (·.span) ++ functions.map (·.span)))) (fun spansOk =>
                              if !spansOk then
                                .done (.error "selected declaration spans overlap or leave source bounds")
                              else
                                .ask (.validation (.resolveCalls functions)) (fun callsOk =>
                                  if !callsOk then
                                    .done (.error "unsupported or unresolved reachable call")
                                  else
                                    .ask (.validation (.resolveNames (constants ++ [{ name := "COORD_CEILING", value := .literal 8796093022208, span := ⟨0, 1⟩ }]) functions)) (fun namesOk =>
                                      if !namesOk then
                                        .done (.error "unresolved local or constant name")
                                      else
                                        let program : Program := { constants := constants, functions := functions, sourceBytes := bytes.size, consumedBytes := consumed }
                                        let certificate : Certificate := ⟨true, true, true, true, true⟩
                                        .done (.ok ⟨{ program := program, certificate := certificate }, ⟨rfl, rfl, rfl, rfl, rfl⟩⟩))))))))
  | .decodeTable bytes =>
      .ask (.lexing (.tokenize bytes)) (fun result =>
        match result with
        | .ok (tokens, consumed) => .ask (.decodeTokens bytes tokens consumed) (fun r => .done r)
        | .error message => .done (.error message))
  | .importedCoord bytes =>
      .ask (.lexing (.tokenize bytes)) (fun result =>
        match result with
        | .ok (tokens, consumed) =>
            if consumed = bytes.size then
              .ask (.selection (.scan (tokens.length + 1) tokens ["COORD_CEILING"] [])) (fun selected =>
                match selected with
                | .error message => .done (.error message)
                | .ok pair =>
                    match pair.1 with
                    | [] => .done (.error "COORD_CEILING is missing or duplicated")
                    | decl :: [] => .done (.ok decl)
                    | _ :: _ :: _ => .done (.error "COORD_CEILING is missing or duplicated"))
            else .done (.error "import tokenizer did not consume every source byte")
        | .error message => .done (.error message))
  | .closure table std =>
      .ask (.decodeTable table) (fun decoded =>
        match decoded with
        | .error message => .done (.error message)
        | .ok decoded =>
            .ask (.importedCoord std) (fun coord =>
              match coord with
              | .error message => .done (.error message)
              | .ok coord =>
                  match coord.value with
                  | .literal value =>
                      match Nat.beq value 8796093022208 with
                      | true => .done (.ok (Decode.appendImportedCoord decoded coord))
                      | false => .done (.error "COORD_CEILING has the wrong value")
                  | .name _ => .done (.error "COORD_CEILING has the wrong value")
                  | .call _ _ => .done (.error "COORD_CEILING has the wrong value")
                  | .memory _ => .done (.error "COORD_CEILING has the wrong value")
                  | .binary _ _ _ => .done (.error "COORD_CEILING has the wrong value")
                  | .conditional _ _ _ => .done (.error "COORD_CEILING has the wrong value")))

/-- A checked derivation of an answer for a layer. -/
abbrev Run (q : Request) (answer : Answer q) : Prop :=
  Derives body (body q) answer

/-- Human-readable unfolding of `Decode.decodeTokens`. -/
theorem decodeTokens_unfold_h (bytes : ByteArray) (tokens : List Token) (consumed : Nat) :
    Decode.decodeTokens bytes tokens consumed =
      (if (consumed != bytes.size) = true then
        Except.error "tokenizer did not consume every source byte"
       else if (!Decode.delimitersBalanced tokens []) = true then
        Except.error "unbalanced source declaration delimiter"
       else
        match Decode.scanSelected (tokens.length + 1) tokens
            Decode.selectedConstantNames Decode.selectedFunctionNames with
        | .error message => Except.error message
        | .ok (constants, functions) =>
            let constantNames := constants.map (·.name)
            let functionNames := functions.map (·.name)
            if (constantNames.length != 8) = true then
              Except.error "selected table constants are missing or duplicated"
            else if (functionNames.length != Decode.selectedFunctionNames.length) = true then
              Except.error "selected functions are missing or duplicated"
            else if (!Decode.distinct (constantNames ++ ["COORD_CEILING"]) ||
                !Decode.distinct functionNames) = true then
              Except.error "selected declaration names are not unique"
            else if (!Decode.spansValid bytes.size
                (constants.map (·.span) ++ functions.map (·.span))) = true then
              Except.error "selected declaration spans overlap or leave source bounds"
            else if (!Decode.resolveCalls functions) = true then
              Except.error "unsupported or unresolved reachable call"
            else if (!Decode.resolveNames (constants ++ [{ name := "COORD_CEILING", value := .literal 8796093022208, span := ⟨0, 1⟩ }]) functions) = true then
              Except.error "unresolved local or constant name"
            else
              Except.ok ⟨{ program := { constants := constants, functions := functions, sourceBytes := bytes.size, consumedBytes := consumed }, certificate := { exactConsumption := true, declarationBoundaries := true, selectedConstants := true, selectedNames := true, resolvedCalls := true } }, ⟨rfl, rfl, rfl, rfl, rfl⟩⟩) := by
  unfold Decode.decodeTokens
  dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
  by_cases h0 : (consumed != bytes.size) = true
  · rw [if_pos h0, if_pos h0] <;> rfl
  · rw [if_neg h0, if_neg h0]
    by_cases h1 : (!Decode.delimitersBalanced tokens []) = true
    · rw [if_pos h1, if_pos h1] <;> rfl
    · rw [if_neg h1, if_neg h1]
      cases hscan : Decode.scanSelected (tokens.length + 1) tokens
          Decode.selectedConstantNames Decode.selectedFunctionNames with
      | error message => rfl
      | ok pair => cases pair with | mk constants functions => dsimp only []

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
  | .decodeTokens bytes tokens consumed => by
      rw [show Slang.Derivation.DecodeTrace.evaluate (.decodeTokens bytes tokens consumed)
            = Decode.decodeTokens bytes tokens consumed from rfl]
      rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h]
      dsimp only [Slang.Derivation.DecodeTrace.body]
      repeat first
        | rw [Slang.Derivation.fold_ite]
        | rw [Slang.Derivation.fold_ask]
        | rw [Slang.Derivation.fold_done]
        | rw [Slang.Derivation.DecodeTrace.evaluate]
        | rw [Slang.Derivation.Selection.evaluate]
        | rw [Slang.Derivation.Validation.evaluate]
      cases hscan : Decode.scanSelected (tokens.length + 1) tokens
          Decode.selectedConstantNames Decode.selectedFunctionNames with
      | error message =>
          repeat first
            | rw [Slang.Derivation.fold_ite]
            | rw [Slang.Derivation.fold_ask]
            | rw [Slang.Derivation.fold_done]
            | rw [Slang.Derivation.DecodeTrace.evaluate]
            | rw [Slang.Derivation.Selection.evaluate]
            | rw [Slang.Derivation.Validation.evaluate]
      | ok pair =>
          cases pair with
          | mk constants functions =>
              repeat first
                | rw [Slang.Derivation.fold_ite]
                | rw [Slang.Derivation.fold_ask]
                | rw [Slang.Derivation.fold_done]
                | rw [Slang.Derivation.DecodeTrace.evaluate]
                | rw [Slang.Derivation.Selection.evaluate]
                | rw [Slang.Derivation.Validation.evaluate]
  | .decodeTable bytes => by
      rw [show Slang.Derivation.DecodeTrace.evaluate (.decodeTable bytes)
            = Decode.decodeTable bytes from rfl]
      dsimp only [Slang.Derivation.DecodeTrace.body]
      rw [Slang.Derivation.fold_ask]
      dsimp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.Lexing.evaluate]
      rw [Decode.decodeTable]
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      cases h : Decode.tokenize bytes with
      | error message => rfl
      | ok pair =>
          cases pair with
          | mk tokens consumed => rfl
  | .importedCoord bytes => by
      rw [show Slang.Derivation.DecodeTrace.evaluate (.importedCoord bytes)
            = Decode.decodeImportedCoordCeiling bytes from rfl]
      dsimp only [Slang.Derivation.DecodeTrace.body]
      rw [Slang.Derivation.fold_ask]
      dsimp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.Lexing.evaluate]
      rw [Decode.decodeImportedCoordCeiling]
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      cases h : Decode.tokenize bytes with
      | error message => rfl
      | ok pair =>
          cases pair with
          | mk tokens consumed =>
              dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
              by_cases hc : consumed = bytes.size
              · rw [if_pos hc, if_pos hc]
                rw [Slang.Derivation.fold_ask]
                dsimp only [Slang.Derivation.DecodeTrace.evaluate, Slang.Derivation.Selection.evaluate]
                cases hs : Decode.scanSelected (tokens.length + 1) tokens
                    ["COORD_CEILING"] [] with
                | error message => rfl
                | ok selected =>
                    cases selected with
                    | mk decls funcs =>
                        cases decls with
                        | nil => rfl
                        | cons decl rest =>
                            cases rest with
                            | nil => rfl
                            | cons decl2 rest2 => rfl
              · rw [if_neg hc, if_neg hc]
                rfl
  | .closure table std => by
      rw [show Slang.Derivation.DecodeTrace.evaluate (.closure table std)
            = Decode.decodeClosure table std from rfl]
      dsimp only [Slang.Derivation.DecodeTrace.body]
      rw [Slang.Derivation.fold_ask]
      dsimp only [Slang.Derivation.DecodeTrace.evaluate]
      rw [Decode.decodeClosure]
      dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
      cases h : Decode.decodeTable table with
      | error message => rfl
      | ok decoded =>
          dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
          rw [Slang.Derivation.fold_ask]
          dsimp only [Slang.Derivation.DecodeTrace.evaluate]
          cases hc : Decode.decodeImportedCoordCeiling std with
          | error message => rfl
          | ok coord =>
              dsimp only [Bind.bind, Monad.toBind, Except.instMonad, Except.bind]
              cases coord with
              | mk _ value _ =>
                  dsimp only []
                  cases value with
                  | literal v =>
                      dsimp only []
                      cases hv : Nat.beq v 8796093022208 with
                      | true => rfl
                      | false => rfl
                  | name n => rfl
                  | call n args => rfl
                  | memory n => rfl
                  | binary op left right => rfl
                  | conditional thenValue condition elseValue => rfl

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


/-- A checked derivation of the real decoder's result for every token list. -/
theorem decodeTokens_run : ∀ (bytes : ByteArray) (tokens : List Token) (consumed : Nat),
    Run (.decodeTokens bytes tokens consumed) (Decode.decodeTokens bytes tokens consumed) := by
  intro bytes tokens consumed
  unfold Run Slang.Derivation.DecodeTrace.body
  try dsimp only []
  by_cases h0 : (consumed != bytes.size) = true
  · rw [if_pos h0]
    exact Derives.done_eq (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_pos h0] <;> rfl)
  · rw [if_neg h0]
    refine Derives.ask (Slang.Derivation.DecodeTrace.validation_of_run
      (Slang.Derivation.Validation.delimiters_run tokens [])) ?_
    try dsimp only []
    by_cases h1 : (!Decode.delimitersBalanced tokens []) = true
    · rw [if_pos h1]
      exact Derives.done_eq (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_pos h1] <;> rfl)
    · rw [if_neg h1]
      refine Derives.ask (Slang.Derivation.DecodeTrace.selection_of_run
        (Slang.Derivation.Selection.scan_run (tokens.length + 1) tokens
          Decode.selectedConstantNames Decode.selectedFunctionNames)) ?_
      try dsimp only []
      cases hscan : Decode.scanSelected (tokens.length + 1) tokens
          Decode.selectedConstantNames Decode.selectedFunctionNames with
      | error message =>
          exact Derives.done_eq
            (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1, hscan] <;> rfl)
      | ok pair =>
          cases pair with
          | mk constants functions =>
              try dsimp only []
              by_cases h2 : ((constants.map (·.name)).length != 8) = true
              · rw [if_pos h2]
                exact Derives.done_eq
                  (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_pos h2])
              · rw [if_neg h2]
                by_cases h3 : ((functions.map (·.name)).length != Decode.selectedFunctionNames.length) = true
                · rw [if_pos h3]
                  exact Derives.done_eq
                    (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_neg h2, if_pos h3])
                · rw [if_neg h3]
                  refine Derives.ask (Slang.Derivation.DecodeTrace.validation_of_run
                    (Slang.Derivation.Validation.distinct_run ((constants.map (·.name)) ++ ["COORD_CEILING"]))) ?_
                  try dsimp only []
                  refine Derives.ask (Slang.Derivation.DecodeTrace.validation_of_run
                    (Slang.Derivation.Validation.distinct_run (functions.map (·.name)))) ?_
                  try dsimp only []
                  by_cases h4 : (!Decode.distinct ((constants.map (·.name)) ++ ["COORD_CEILING"]) || !Decode.distinct (functions.map (·.name))) = true
                  · rw [if_pos h4]
                    exact Derives.done_eq
                      (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_neg h2, if_neg h3, if_pos h4])
                  · rw [if_neg h4]
                    refine Derives.ask (Slang.Derivation.DecodeTrace.validation_of_run
                      (Slang.Derivation.Validation.spans_run bytes.size ((constants.map (·.span)) ++ (functions.map (·.span))))) ?_
                    try dsimp only []
                    by_cases h5 : (!Decode.spansValid bytes.size ((constants.map (·.span)) ++ (functions.map (·.span)))) = true
                    · rw [if_pos h5]
                      exact Derives.done_eq
                        (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_neg h2, if_neg h3, if_neg h4, if_pos h5])
                    · rw [if_neg h5]
                      refine Derives.ask (Slang.Derivation.DecodeTrace.validation_of_run
                        (Slang.Derivation.Validation.resolveCalls_run functions)) ?_
                      try dsimp only []
                      by_cases h6 : (!Decode.resolveCalls functions) = true
                      · rw [if_pos h6]
                        exact Derives.done_eq
                          (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_neg h2, if_neg h3, if_neg h4, if_neg h5, if_pos h6])
                      · rw [if_neg h6]
                        refine Derives.ask (Slang.Derivation.DecodeTrace.validation_of_run
                          (Slang.Derivation.Validation.resolveNames_run (constants ++ [{ name := "COORD_CEILING", value := .literal 8796093022208, span := ⟨0, 1⟩ }]) functions)) ?_
                        try dsimp only []
                        by_cases h7 : (!Decode.resolveNames (constants ++ [{ name := "COORD_CEILING", value := .literal 8796093022208, span := ⟨0, 1⟩ }]) functions) = true
                        · rw [if_pos h7]
                          exact Derives.done_eq
                            (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_neg h2, if_neg h3, if_neg h4, if_neg h5, if_neg h6, if_pos h7])
                        · rw [if_neg h7]
                          exact Derives.done_eq
                            (by rw [Slang.Derivation.DecodeTrace.decodeTokens_unfold_h, if_neg h0, if_neg h1]; simp only [hscan]; rw [if_neg h2, if_neg h3, if_neg h4, if_neg h5, if_neg h6, if_neg h7])

/-- A wrong answer for the empty decoder request is not derivable. -/
abbrev wrongDecoded : CertifiedDecoded :=
  ⟨{ program := { constants := [], functions := [], sourceBytes := 0, consumedBytes := 0 }, certificate := ⟨true, true, true, true, true⟩ }, ⟨rfl, rfl, rfl, rfl, rfl⟩⟩

theorem decodeTokens_wrong :
    ¬ Run (.decodeTokens (ByteArray.empty) [] 0) (.ok wrongDecoded) := by
  intro derivation
  have sound := Run.sound derivation
  have actual : evaluate (.decodeTokens (ByteArray.empty) [] 0) =
      .error "selected table constants are missing or duplicated" := rfl
  rw [actual] at sound
  cases sound

/-- A wrong answer for the empty table decoder request is not derivable. -/
theorem decodeTable_wrong :
    ¬ Run (.decodeTable (ByteArray.empty)) (.ok wrongDecoded) := by
  intro derivation
  have sound := Run.sound derivation
  have actual : evaluate (.decodeTable (ByteArray.empty)) =
      .error "selected table constants are missing or duplicated" := rfl
  rw [actual] at sound
  cases sound

/-- A wrong answer for the empty closure request is not derivable. -/
theorem closure_wrong :
    ¬ Run (.closure (ByteArray.empty) (ByteArray.empty)) (.ok wrongDecoded) := by
  intro derivation
  have sound := Run.sound derivation
  have actual : evaluate (.closure (ByteArray.empty) (ByteArray.empty)) =
      .error "selected table constants are missing or duplicated" := rfl
  rw [actual] at sound
  cases sound

#audit_empty Slang.Derivation.DecodeTrace.evaluate : (q : Slang.Derivation.DecodeTrace.Request) → Slang.Derivation.DecodeTrace.Answer q
#audit_empty Slang.Derivation.DecodeTrace.body_eq : ∀ (q : Slang.Derivation.DecodeTrace.Request), Slang.Derivation.DecodeTrace.evaluate q = Slang.Derivation.fold Slang.Derivation.DecodeTrace.evaluate (Slang.Derivation.DecodeTrace.body q)
#audit_empty Slang.Derivation.DecodeTrace.Run.sound : ∀ {q : Slang.Derivation.DecodeTrace.Request} {answer : Slang.Derivation.DecodeTrace.Answer q}, Slang.Derivation.DecodeTrace.Run q answer → Slang.Derivation.DecodeTrace.evaluate q = answer
#audit_empty Slang.Derivation.DecodeTrace.multibyte_run : Slang.Derivation.DecodeTrace.Run (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) (Except.ok ['é'])
#audit_empty Slang.Derivation.DecodeTrace.multibyte_result : Slang.Derivation.DecodeTrace.evaluate (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) = Except.ok ['é']
#audit_empty Slang.Derivation.DecodeTrace.multibyte_wrong : ¬Slang.Derivation.DecodeTrace.Run (Slang.Derivation.DecodeTrace.Request.utf8Fuel 3 Slang.Derivation.DecodeTrace.multibyteBytes) (Except.error "source is not UTF-8")

end Slang.Derivation.DecodeTrace

