import Slang.Decode

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source.Parsing

open Slang Slang.Decode

/-- A structural scan of the leading newline tokens. -/
inductive DropNewlinesScan : List Token → List Token → Prop where
  | done : DropNewlinesScan [] []
  | newline {token : Token} {rest result : List Token}
      (guard : token.kind = .newline) (tail : DropNewlinesScan rest result) :
      DropNewlinesScan (token :: rest) result
  | stop {token : Token} {rest : List Token} (guard : token.kind ≠ .newline) :
      DropNewlinesScan (token :: rest) (token :: rest)

namespace DropNewlinesScan

/-- A structural leading-newline scan entails the executable helper's result. -/
theorem sound {tokens result : List Token} (scan : DropNewlinesScan tokens result) :
    Decode.dropNewlines tokens = result := by
  induction scan with
  | done => rfl
  | newline guard _ tailSound =>
      unfold Decode.dropNewlines
      rw [if_pos guard]
      exact tailSound
  | stop guard =>
      unfold Decode.dropNewlines
      rw [if_neg guard]

end DropNewlinesScan

/-- A structural scan through one source line, retaining its newline in the suffix. -/
inductive TakeLineScan : List Token → List Token → List Token → Prop where
  | done : TakeLineScan [] [] []
  | newline {token : Token} {rest : List Token} (guard : token.kind = .newline) :
      TakeLineScan (token :: rest) [] (token :: rest)
  | next {token : Token} {rest line tail : List Token} (guard : token.kind ≠ .newline)
      (scan : TakeLineScan rest line tail) :
      TakeLineScan (token :: rest) (token :: line) tail

namespace TakeLineScan

/-- A structural line scan entails the executable helper's result. -/
theorem sound {tokens line tail : List Token} (scan : TakeLineScan tokens line tail) :
    Decode.takeLine tokens = (line, tail) := by
  induction scan with
  | done => rfl
  | newline guard =>
      unfold Decode.takeLine
      rw [if_pos guard]
  | next guard _ tailSound =>
      unfold Decode.takeLine
      rw [if_neg guard]
      dsimp only
      rw [tailSound]

end TakeLineScan

/-- The expression-parser obligation for one selected constant line. -/
inductive ConstantLineParse : List Token → Expr → List Token → Prop where
  | intro {tokens : List Token} {value : Expr} {leftover : List Token}
      (parsed : Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens =
        .ok (value, leftover)) : ConstantLineParse tokens value leftover

namespace ConstantLineParse

/-- A selected constant-line parse entails the executable expression-parser result. -/
theorem sound {tokens : List Token} {value : Expr} {leftover : List Token}
    (parse : ConstantLineParse tokens value leftover) :
    Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (value, leftover) := by
  cases parse
  assumption

end ConstantLineParse

/-- The function-parser obligation for one selected function declaration. -/
inductive SelectedFunctionParse : String → Nat → List Token → FunctionDecl → List Token → Prop where
  | intro {name : String} {start : Nat} {tokens : List Token} {decl : FunctionDecl}
      {after : List Token} (parsed : Decode.parseFunction name start tokens = .ok (decl, after)) :
      SelectedFunctionParse name start tokens decl after

namespace SelectedFunctionParse

/-- A selected function parse entails the executable declaration-parser result. -/
theorem sound {name : String} {start : Nat} {tokens : List Token} {decl : FunctionDecl}
    {after : List Token} (parse : SelectedFunctionParse name start tokens decl after) :
    Decode.parseFunction name start tokens = .ok (decl, after) := by
  cases parse
  assumption

end SelectedFunctionParse

/-- A token accepted by the grouped non-identifier branch of `Decode.scanSelected`. -/
inductive NonIdentifierToken : Token → Prop where
  | numeral (value : Nat) (span : Span) : NonIdentifierToken { kind := .numeral value, span }
  | lparen (span : Span) : NonIdentifierToken { kind := .lparen, span }
  | rparen (span : Span) : NonIdentifierToken { kind := .rparen, span }
  | lbrace (span : Span) : NonIdentifierToken { kind := .lbrace, span }
  | rbrace (span : Span) : NonIdentifierToken { kind := .rbrace, span }
  | lbracket (span : Span) : NonIdentifierToken { kind := .lbracket, span }
  | rbracket (span : Span) : NonIdentifierToken { kind := .rbracket, span }
  | comma (span : Span) : NonIdentifierToken { kind := .comma, span }
  | semi (span : Span) : NonIdentifierToken { kind := .semi, span }
  | newline (span : Span) : NonIdentifierToken { kind := .newline, span }
  | assign (span : Span) : NonIdentifierToken { kind := .assign, span }
  | lt (span : Span) : NonIdentifierToken { kind := .lt, span }
  | le (span : Span) : NonIdentifierToken { kind := .le, span }
  | gt (span : Span) : NonIdentifierToken { kind := .gt, span }
  | ge (span : Span) : NonIdentifierToken { kind := .ge, span }
  | other (value : Char) (span : Span) : NonIdentifierToken { kind := .other value, span }

namespace NonIdentifierToken

/-- A grouped non-identifier token composes with a successful recursive scan. -/
theorem sound {token : Token} (head : NonIdentifierToken token) {fuel : Nat}
    {rest : List Token} {constantNames functionNames : List String}
    {constants : List ConstantDecl} {functions : List FunctionDecl}
    (tail : Decode.scanSelected fuel rest constantNames functionNames = .ok (constants, functions)) :
    Decode.scanSelected (fuel + 1) (token :: rest) constantNames functionNames =
      .ok (constants, functions) := by
  cases head <;> exact tail

end NonIdentifierToken

/-- A successful structural traversal of the selected declaration scan. -/
inductive ScanSelected : Nat → List Token → List String → List String →
    List ConstantDecl → List FunctionDecl → Prop where
  | done {fuel : Nat} {constants functions : List String} :
      ScanSelected (fuel + 1) [] constants functions [] []
  | skipNonIdentifier {fuel : Nat} {token : Token} {rest : List Token}
      {constantNames functionNames : List String} {constants : List ConstantDecl}
      {functions : List FunctionDecl} (head : NonIdentifierToken token)
      (tail : ScanSelected fuel rest constantNames functionNames constants functions) :
      ScanSelected (fuel + 1) (token :: rest) constantNames functionNames
        constants functions
  | identEnd {fuel : Nat} {name : String} {span : Span} {constantNames functionNames : List String}
      {constants : List ConstantDecl} {functions : List FunctionDecl}
      (tail : ScanSelected fuel [] constantNames functionNames constants functions) :
      ScanSelected (fuel + 1) [{ kind := .ident name, span }] constantNames functionNames
        constants functions
  | skipOther {fuel : Nat} {name : String} {span : Span} {next : Token} {rest : List Token}
      {constantNames functionNames : List String} {constants : List ConstantDecl}
      {functions : List FunctionDecl} (notAssign : next.kind ≠ .assign)
      (notLparen : next.kind ≠ .lparen)
      (tail : ScanSelected fuel (next :: rest) constantNames functionNames constants functions) :
      ScanSelected (fuel + 1) ({ kind := .ident name, span } :: next :: rest)
        constantNames functionNames constants functions
  | skipConstant {fuel : Nat} {name : String} {span : Span} {next : Token}
      {rest : List Token} {constantNames functionNames : List String}
      {constants : List ConstantDecl} {functions : List FunctionDecl}
      (assign : next.kind = .assign) (notSelected : constantNames.contains name = false)
      (notDuplicate : Decode.selectedConstantNames.contains name = false)
      (tail : ScanSelected fuel (next :: rest) constantNames functionNames constants functions) :
      ScanSelected (fuel + 1) ({ kind := .ident name, span } :: next :: rest)
        constantNames functionNames constants functions
  | constant {fuel : Nat} {name : String} {span : Span} {next : Token} {rest line afterLine : List Token}
      {constantNames functionNames : List String} {value : Expr} {leftover : List Token}
      {constants : List ConstantDecl} {functions : List FunctionDecl} {stop : Nat}
      (assign : next.kind = .assign) (selected : constantNames.contains name = true)
      (lineScan : TakeLineScan rest line afterLine) (parse : ConstantLineParse line value leftover)
      (leftoverScan : DropNewlinesScan leftover [])
      (stopEq : stop = (line.getLast?.map (fun token => token.span.stop)).getD span.stop)
      (tail : ScanSelected fuel afterLine (constantNames.erase name) functionNames
        constants functions) :
      ScanSelected (fuel + 1) ({ kind := .ident name, span } :: next :: rest)
        constantNames functionNames
        ({ name, value, span := ⟨span.start, stop⟩ } :: constants) functions
  | skipFunction {fuel : Nat} {name : String} {span : Span} {next : Token}
      {rest : List Token} {constantNames functionNames : List String}
      {constants : List ConstantDecl} {functions : List FunctionDecl}
      (notAssign : next.kind ≠ .assign) (lparen : next.kind = .lparen)
      (notSelected : functionNames.contains name = false)
      (tail : ScanSelected fuel (next :: rest) constantNames functionNames constants functions) :
      ScanSelected (fuel + 1) ({ kind := .ident name, span } :: next :: rest)
        constantNames functionNames constants functions
  | function {fuel : Nat} {name : String} {span : Span} {next : Token} {rest after : List Token}
      {constantNames functionNames : List String} {constants : List ConstantDecl}
      {functions : List FunctionDecl} {decl : FunctionDecl}
      (notAssign : next.kind ≠ .assign) (lparen : next.kind = .lparen)
      (selected : functionNames.contains name = true)
      (parse : SelectedFunctionParse name span.start
        ({ kind := .lparen, span := ⟨span.stop, span.stop + 1⟩ } :: rest) decl after)
      (tail : ScanSelected fuel after constantNames (functionNames.erase name)
        constants functions) :
      ScanSelected (fuel + 1) ({ kind := .ident name, span } :: next :: rest)
        constantNames functionNames constants (decl :: functions)

namespace ScanSelected

/-- A structural selected-declaration scan entails the executable scan result. -/
theorem sound {fuel : Nat} {tokens : List Token} {constantNames functionNames : List String}
    {constants : List ConstantDecl} {functions : List FunctionDecl}
    (scan : ScanSelected fuel tokens constantNames functionNames constants functions) :
    Decode.scanSelected fuel tokens constantNames functionNames = .ok (constants, functions) := by
  have boolNeg : ∀ {b : Bool}, b = false → b ≠ true := by
    intro b guard
    rw [guard]
    exact Bool.false_ne_true
  induction scan with
  | done => rfl
  | skipNonIdentifier head _ tailSound => exact head.sound tailSound
  | identEnd _ tailSound =>
      dsimp only [Decode.scanSelected]
      exact tailSound
  | skipOther notAssign notLparen _ tailSound =>
      dsimp only [Decode.scanSelected]
      rw [if_neg notAssign, if_neg notLparen]
      exact tailSound
  | skipConstant assign notSelected notDuplicate _ tailSound =>
      dsimp only [Decode.scanSelected]
      rw [if_pos assign, if_neg (boolNeg notSelected), if_neg (boolNeg notDuplicate)]
      exact tailSound
  | constant assign selected lineScan parse leftoverScan stopEq _ tailSound =>
      dsimp only [Decode.scanSelected]
      rw [if_pos assign, if_pos selected, lineScan.sound]
      dsimp only
      change (Decode.parseExprFuel _ 0 _).bind _ = _
      rw [parse.sound]
      change (if Decode.dropNewlines _ != [] then _ else _) = _
      rw [leftoverScan.sound]
      dsimp only
      rw [tailSound, stopEq]
      rfl
  | skipFunction notAssign lparen notSelected _ tailSound =>
      dsimp only [Decode.scanSelected]
      rw [if_neg notAssign, if_pos lparen, if_neg (boolNeg notSelected)]
      exact tailSound
  | function notAssign lparen selected parse _ tailSound =>
      dsimp only [Decode.scanSelected]
      rw [if_neg notAssign, if_pos lparen, if_pos selected]
      rw [parse.sound]
      change (Decode.scanSelected _ _ _ _).bind _ = _
      rw [tailSound]
      rfl

end ScanSelected

end Slang.Source.Parsing
