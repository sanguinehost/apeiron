import Slang.Source.Lexing

set_option autoImplicit false
set_option warningAsError true

namespace Slang.Source

open Slang Slang.Decode

def DeclarationSpans (size : Nat) (spans : List Span) : Prop :=
  (spans.all fun span => span.start < span.stop && span.stop ≤ size) = true ∧
  spans.Pairwise fun left right => left.stop ≤ right.start

structure GrammarDerivation (source : SourceBytes) (tokens : List Token) (consumed : Nat)
    (decoded : CertifiedDecoded) : Prop where
  lexing : LexingDerivation source tokens consumed
  decoding : Decode.decodeTokens (encodeBytes source) tokens consumed = .ok decoded
  declarationSpans : DeclarationSpans source.length
    (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span))

namespace GrammarDerivation

theorem sound {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {decoded : CertifiedDecoded} (derivation : GrammarDerivation source tokens consumed decoded) :
    Decode.decodeTable (encodeBytes source) = .ok decoded := by
  simp only [Decode.decodeTable]
  rw [derivation.lexing.sound]
  exact derivation.decoding

theorem spansInside {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {decoded : CertifiedDecoded} (derivation : GrammarDerivation source tokens consumed decoded) :
    ((decoded.val.program.constants.map (·.span) ++
      decoded.val.program.functions.map (·.span)).all fun span =>
        span.start < span.stop && span.stop ≤ source.length) = true :=
  derivation.declarationSpans.1

theorem spansNonoverlapping {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {decoded : CertifiedDecoded} (derivation : GrammarDerivation source tokens consumed decoded) :
    (decoded.val.program.constants.map (·.span) ++
      decoded.val.program.functions.map (·.span)).Pairwise
        (fun left right => left.stop ≤ right.start) :=
  derivation.declarationSpans.2

end GrammarDerivation

structure ParsesDecl (source : SourceBytes) (tokens : List Token) (consumed : Nat)
    (header : Token) (bodyTokens : List Token) (decl : FunctionDecl) : Prop where
  lexing : LexingDerivation source tokens consumed
  tokenShape : tokens = header :: bodyTokens
  headerKind : header.kind = .ident decl.name
  parsed : Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
  spanInside : decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length

namespace ParsesDecl

theorem sound {source : SourceBytes} {tokens : List Token} {consumed : Nat} {header : Token}
    {bodyTokens : List Token} {decl : FunctionDecl}
    (derivation : ParsesDecl source tokens consumed header bodyTokens decl) :
    ∃ header bodyTokens,
      tokens = header :: bodyTokens ∧
      header.kind = .ident decl.name ∧
      Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, []) :=
  ⟨header, bodyTokens, derivation.tokenShape,
    derivation.headerKind, derivation.parsed⟩

theorem consumesSource {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {header : Token} {bodyTokens : List Token} {decl : FunctionDecl}
    (derivation : ParsesDecl source tokens consumed header bodyTokens decl) :
    consumed = source.length :=
  derivation.lexing.consumes

theorem consumesThroughClose {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {header : Token} {bodyTokens : List Token} {decl : FunctionDecl}
    (derivation : ParsesDecl source tokens consumed header bodyTokens decl) :
    Decode.parseFunction decl.name header.span.start bodyTokens =
      .ok (decl, []) :=
  derivation.parsed

theorem spanBounded {source : SourceBytes} {tokens : List Token} {consumed : Nat}
    {header : Token} {bodyTokens : List Token} {decl : FunctionDecl}
    (derivation : ParsesDecl source tokens consumed header bodyTokens decl) :
    decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length :=
  derivation.spanInside

end ParsesDecl

structure ParsesExpr (source : SourceBytes) (tokens : List Token) (consumed : Nat)
    (expression : Expr) : Prop where
  lexing : LexingDerivation source tokens consumed
  parsed : Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, [])

namespace ParsesExpr

theorem sound {source : SourceBytes} {tokens : List Token} {consumed : Nat} {expression : Expr}
    (derivation : ParsesExpr source tokens consumed expression) :
    Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens =
      .ok (expression, []) :=
  derivation.parsed

end ParsesExpr

end Slang.Source
