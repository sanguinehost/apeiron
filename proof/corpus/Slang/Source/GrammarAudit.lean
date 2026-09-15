import Slang.Audit.Empty
import Slang.Source.Grammar

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000

namespace Slang.Source.GrammarAudit

open Slang Slang.Decode Slang.Source

private def sourceByte (value : Nat) : SourceByte :=
  ⟨value % 256, Nat.mod_lt value (Nat.zero_lt_succ 255)⟩

private def sourceBytes (values : List Nat) : SourceBytes := values.map sourceByte

private def spansPairwise : List Token → Bool
  | [] => true
  | token :: rest =>
      rest.all (fun next => token.span.stop ≤ next.span.start) && spansPairwise rest

private theorem and_left_true (left right : Bool) (both : (left && right) = true) :
    left = true := by
  cases left with
  | false =>
      cases right <;> exact Bool.noConfusion both
  | true => rfl

private theorem and_right_true (left right : Bool) (both : (left && right) = true) :
    right = true := by
  cases left with
  | false =>
      cases right <;> exact Bool.noConfusion both
  | true => exact both

private theorem all_member_true {α : Type} (predicate : α → Bool) (values : List α)
    (all : values.all predicate = true) (value : α) (membership : value ∈ values) :
    predicate value = true := by
  induction values with
  | nil => exact nomatch membership
  | cons head tail ih =>
      cases membership with
      | head => exact and_left_true (predicate value) (tail.all predicate) all
      | tail _ member =>
          exact ih (and_right_true (predicate head) (tail.all predicate) all) member

private theorem spansPairwise_sound (tokens : List Token) (valid : spansPairwise tokens = true) :
    tokens.Pairwise (fun left right => left.span.stop ≤ right.span.start) := by
  induction tokens with
  | nil => exact List.Pairwise.nil
  | cons token rest ih =>
      apply List.Pairwise.cons
      · intro next membership
        exact of_decide_eq_true
          (all_member_true _ _
            (and_left_true _ _ valid) next membership)
      · exact ih (and_right_true _ _ valid)

def conditionalSource : SourceBytes :=
  sourceBytes
    [48, 32, 45, 32, 107, 32, 105, 102, 32, 107, 32, 60, 32, 48, 32, 101, 108,
      115, 101, 32, 107]

def conditionalTokens : List Token :=
  [⟨.numeral 0, ⟨0, 1⟩⟩, ⟨.other '-', ⟨2, 3⟩⟩, ⟨.ident "k", ⟨4, 5⟩⟩,
    ⟨.ident "if", ⟨6, 8⟩⟩, ⟨.ident "k", ⟨9, 10⟩⟩, ⟨.lt, ⟨11, 12⟩⟩,
    ⟨.numeral 0, ⟨13, 14⟩⟩, ⟨.ident "else", ⟨15, 19⟩⟩,
    ⟨.ident "k", ⟨20, 21⟩⟩]

def groupedConditional : Expr :=
  .conditional (.binary .sub (.literal 0) (.name "k"))
    (.binary .lt (.name "k") (.literal 0)) (.name "k")

def incorrectlyGroupedConditional : Expr :=
  .binary .sub (.literal 0)
    (.conditional (.name "k") (.binary .lt (.name "k") (.literal 0)) (.name "k"))

theorem conditional_derivation :
    ParsesExpr conditionalSource conditionalTokens 21 groupedConditional :=
  ⟨LexingDerivation.intro rfl ⟨rfl, spansPairwise_sound conditionalTokens rfl⟩ rfl, rfl⟩

theorem conditional_groups_after_subtraction :
    Decode.parseExprFuel (conditionalTokens.length * 4 + 16) 0 conditionalTokens =
      .ok (groupedConditional, []) :=
  conditional_derivation.sound

theorem conditional_not_grouped_as_subtraction_rhs :
    groupedConditional ≠ incorrectlyGroupedConditional := by
  intro equality
  exact Expr.noConfusion equality

def completeSource : SourceBytes :=
  sourceBytes
    [116, 97, 98, 108, 101, 83, 112, 114, 101, 97, 100, 40, 107, 41, 32, 123, 32,
      112, 111, 115, 32, 61, 32, 48, 32, 45, 32, 107, 32, 105, 102, 32, 107, 32,
      60, 32, 48, 32, 101, 108, 115, 101, 32, 107, 59, 32, 114, 101, 116, 117,
      114, 110, 32, 112, 111, 115, 59, 32, 125]

def completeTokens : List Token :=
  [⟨.ident "tableSpread", ⟨0, 11⟩⟩, ⟨.lparen, ⟨11, 12⟩⟩,
    ⟨.ident "k", ⟨12, 13⟩⟩, ⟨.rparen, ⟨13, 14⟩⟩, ⟨.lbrace, ⟨15, 16⟩⟩,
    ⟨.ident "pos", ⟨17, 20⟩⟩, ⟨.assign, ⟨21, 22⟩⟩, ⟨.numeral 0, ⟨23, 24⟩⟩,
    ⟨.other '-', ⟨25, 26⟩⟩, ⟨.ident "k", ⟨27, 28⟩⟩,
    ⟨.ident "if", ⟨29, 31⟩⟩, ⟨.ident "k", ⟨32, 33⟩⟩, ⟨.lt, ⟨34, 35⟩⟩,
    ⟨.numeral 0, ⟨36, 37⟩⟩, ⟨.ident "else", ⟨38, 42⟩⟩,
    ⟨.ident "k", ⟨43, 44⟩⟩, ⟨.semi, ⟨44, 45⟩⟩,
    ⟨.ident "return", ⟨46, 52⟩⟩, ⟨.ident "pos", ⟨53, 56⟩⟩,
    ⟨.semi, ⟨56, 57⟩⟩, ⟨.rbrace, ⟨58, 59⟩⟩]

def completeDecl : FunctionDecl :=
  { name := "tableSpread"
    parameters := ["k"]
    body := [.bind "pos" groupedConditional, .return_ (.name "pos")]
    span := ⟨0, 59⟩ }

def completeHeader : Token := ⟨.ident "tableSpread", ⟨0, 11⟩⟩

theorem complete_selected_body_derivation :
    ParsesDecl completeSource completeTokens 59 completeHeader completeTokens.tail
      completeDecl :=
  ⟨LexingDerivation.intro rfl ⟨rfl, spansPairwise_sound completeTokens rfl⟩ rfl,
    rfl, rfl, rfl, ⟨Nat.zero_lt_succ 58, Nat.le_refl 59⟩⟩

theorem complete_selected_body_zero_leftover :
    Decode.parseFunction completeDecl.name 0 completeTokens.tail = .ok (completeDecl, []) :=
  complete_selected_body_derivation.consumesThroughClose

def truncatedSource : SourceBytes :=
  sourceBytes
    [116, 97, 98, 108, 101, 83, 112, 114, 101, 97, 100, 40, 107, 41, 32, 123, 32,
      112, 111, 115, 32, 61, 32, 48, 32, 45, 32, 107, 32, 105, 102, 32, 107, 32,
      60, 32, 48, 32, 101, 108, 115, 101, 32, 107, 59]

def truncatedTokens : List Token := completeTokens.take 17

theorem truncated_selected_body_prefix_rejected :
    Decode.tokenize (encodeBytes truncatedSource) = .ok (truncatedTokens, 45) ∧
    (match Decode.parseFunction "tableSpread" 0 truncatedTokens.tail with
      | .error _ => true
      | .ok _ => false) = true :=
  ⟨rfl, rfl⟩

#audit_empty Slang.Decode.parseExprFuel : Nat → Nat → List Slang.Decode.Token → Slang.Decode.Result (Slang.Expr × List Slang.Decode.Token)
#audit_empty Slang.Decode.parseFunction : String → Nat → List Slang.Decode.Token → Slang.Decode.Result (Slang.FunctionDecl × List Slang.Decode.Token)
#audit_empty Slang.Decode.decodeTokens : ByteArray → List Slang.Decode.Token → Nat → Slang.Decode.Result Slang.Decode.CertifiedDecoded
#audit_empty Slang.Decode.decodeTable : ByteArray → Slang.Decode.Result Slang.Decode.CertifiedDecoded
#audit_empty Slang.Source.DeclarationSpans : Nat → List Slang.Span → Prop
#audit_empty Slang.Source.GrammarDerivation : Slang.Source.SourceBytes → List Slang.Decode.Token → Nat → Slang.Decode.CertifiedDecoded → Prop
#audit_empty Slang.Source.GrammarDerivation.mk : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.LexingDerivation source tokens consumed → Slang.Decode.decodeTokens (Slang.Source.encodeBytes source) tokens consumed = .ok decoded → Slang.Source.DeclarationSpans source.length (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span)) → Slang.Source.GrammarDerivation source tokens consumed decoded
#audit_empty Slang.Source.GrammarDerivation.lexing : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.GrammarDerivation source tokens consumed decoded → Slang.Source.LexingDerivation source tokens consumed
#audit_empty Slang.Source.GrammarDerivation.decoding : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.GrammarDerivation source tokens consumed decoded → Slang.Decode.decodeTokens (Slang.Source.encodeBytes source) tokens consumed = .ok decoded
#audit_empty Slang.Source.GrammarDerivation.declarationSpans : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.GrammarDerivation source tokens consumed decoded → Slang.Source.DeclarationSpans source.length (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span))
#audit_empty Slang.Source.GrammarDerivation.sound : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.GrammarDerivation source tokens consumed decoded → Slang.Decode.decodeTable (Slang.Source.encodeBytes source) = .ok decoded
#audit_empty Slang.Source.GrammarDerivation.spansInside : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.GrammarDerivation source tokens consumed decoded → ((decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span)).all fun span => span.start < span.stop && span.stop ≤ source.length) = true
#audit_empty Slang.Source.GrammarDerivation.spansNonoverlapping : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {decoded : Slang.Decode.CertifiedDecoded}, Slang.Source.GrammarDerivation source tokens consumed decoded → (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span)).Pairwise (fun left right => left.stop ≤ right.start)
#audit_empty Slang.Source.ParsesDecl : Slang.Source.SourceBytes → List Slang.Decode.Token → Nat → Slang.Decode.Token → List Slang.Decode.Token → Slang.FunctionDecl → Prop
#audit_empty Slang.Source.ParsesDecl.mk : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.LexingDerivation source tokens consumed → tokens = header :: bodyTokens → header.kind = .ident decl.name → Slang.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, []) → decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length → Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl
#audit_empty Slang.Source.ParsesDecl.lexing : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → Slang.Source.LexingDerivation source tokens consumed
#audit_empty Slang.Source.ParsesDecl.tokenShape : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → tokens = header :: bodyTokens
#audit_empty Slang.Source.ParsesDecl.headerKind : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → header.kind = .ident decl.name
#audit_empty Slang.Source.ParsesDecl.parsed : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → Slang.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
#audit_empty Slang.Source.ParsesDecl.spanInside : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length
#audit_empty Slang.Source.ParsesDecl.sound : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → ∃ header bodyTokens, tokens = header :: bodyTokens ∧ header.kind = .ident decl.name ∧ Slang.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
#audit_empty Slang.Source.ParsesDecl.consumesSource : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → consumed = source.length
#audit_empty Slang.Source.ParsesDecl.consumesThroughClose : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → Slang.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
#audit_empty Slang.Source.ParsesDecl.spanBounded : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {header : Slang.Decode.Token} {bodyTokens : List Slang.Decode.Token} {decl : Slang.FunctionDecl}, Slang.Source.ParsesDecl source tokens consumed header bodyTokens decl → decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length
#audit_empty Slang.Source.ParsesExpr : Slang.Source.SourceBytes → List Slang.Decode.Token → Nat → Slang.Expr → Prop
#audit_empty Slang.Source.ParsesExpr.mk : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {expression : Slang.Expr}, Slang.Source.LexingDerivation source tokens consumed → Slang.Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, []) → Slang.Source.ParsesExpr source tokens consumed expression
#audit_empty Slang.Source.ParsesExpr.lexing : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {expression : Slang.Expr}, Slang.Source.ParsesExpr source tokens consumed expression → Slang.Source.LexingDerivation source tokens consumed
#audit_empty Slang.Source.ParsesExpr.parsed : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {expression : Slang.Expr}, Slang.Source.ParsesExpr source tokens consumed expression → Slang.Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, [])
#audit_empty Slang.Source.ParsesExpr.sound : ∀ {source : Slang.Source.SourceBytes} {tokens : List Slang.Decode.Token} {consumed : Nat} {expression : Slang.Expr}, Slang.Source.ParsesExpr source tokens consumed expression → Slang.Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, [])
#audit_empty Slang.Source.GrammarAudit.conditionalSource : Slang.Source.SourceBytes
#audit_empty Slang.Source.GrammarAudit.conditionalTokens : List Slang.Decode.Token
#audit_empty Slang.Source.GrammarAudit.groupedConditional : Slang.Expr
#audit_empty Slang.Source.GrammarAudit.incorrectlyGroupedConditional : Slang.Expr
#audit_empty Slang.Source.GrammarAudit.conditional_derivation : Slang.Source.ParsesExpr Slang.Source.GrammarAudit.conditionalSource Slang.Source.GrammarAudit.conditionalTokens 21 Slang.Source.GrammarAudit.groupedConditional
#audit_empty Slang.Source.GrammarAudit.conditional_groups_after_subtraction : Slang.Decode.parseExprFuel (Slang.Source.GrammarAudit.conditionalTokens.length * 4 + 16) 0 Slang.Source.GrammarAudit.conditionalTokens = .ok (Slang.Source.GrammarAudit.groupedConditional, [])
#audit_empty Slang.Source.GrammarAudit.conditional_not_grouped_as_subtraction_rhs : Slang.Source.GrammarAudit.groupedConditional ≠ Slang.Source.GrammarAudit.incorrectlyGroupedConditional
#audit_empty Slang.Source.GrammarAudit.completeSource : Slang.Source.SourceBytes
#audit_empty Slang.Source.GrammarAudit.completeTokens : List Slang.Decode.Token
#audit_empty Slang.Source.GrammarAudit.completeDecl : Slang.FunctionDecl
#audit_empty Slang.Source.GrammarAudit.completeHeader : Slang.Decode.Token
#audit_empty Slang.Source.GrammarAudit.complete_selected_body_derivation : Slang.Source.ParsesDecl Slang.Source.GrammarAudit.completeSource Slang.Source.GrammarAudit.completeTokens 59 Slang.Source.GrammarAudit.completeHeader Slang.Source.GrammarAudit.completeTokens.tail Slang.Source.GrammarAudit.completeDecl
#audit_empty Slang.Source.GrammarAudit.complete_selected_body_zero_leftover : Slang.Decode.parseFunction Slang.Source.GrammarAudit.completeDecl.name 0 Slang.Source.GrammarAudit.completeTokens.tail = .ok (Slang.Source.GrammarAudit.completeDecl, [])
#audit_empty Slang.Source.GrammarAudit.truncatedSource : Slang.Source.SourceBytes
#audit_empty Slang.Source.GrammarAudit.truncatedTokens : List Slang.Decode.Token
#audit_empty Slang.Source.GrammarAudit.truncated_selected_body_prefix_rejected : Slang.Decode.tokenize (Slang.Source.encodeBytes Slang.Source.GrammarAudit.truncatedSource) = .ok (Slang.Source.GrammarAudit.truncatedTokens, 45) ∧ (match Slang.Decode.parseFunction "tableSpread" 0 Slang.Source.GrammarAudit.truncatedTokens.tail with | .error _ => true | .ok _ => false) = true

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Slang.Source.GrammarAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Slang.Source.GrammarAudit
