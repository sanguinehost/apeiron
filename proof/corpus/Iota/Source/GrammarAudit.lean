import Iota.Audit.Empty
import Iota.Source.Grammar

set_option autoImplicit false
set_option warningAsError true
set_option maxRecDepth 10000

namespace Iota.Source.GrammarAudit

open Iota Iota.Decode Iota.Source

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

macro "derive_take_while" : tactic =>
  `(tactic| solve
    | repeat first
      | exact Tokenizer.TakeWhileScan.nil
      | exact Tokenizer.TakeWhileScan.stop rfl
      | apply Tokenizer.TakeWhileScan.next rfl)

macro "derive_utf8_length" : tactic =>
  `(tactic| solve
    | repeat first
      | exact Tokenizer.Utf8LengthFold.done
      | apply Tokenizer.Utf8LengthFold.next)

macro "derive_digits" : tactic =>
  `(tactic| solve
    | repeat first
      | exact Tokenizer.DigitsFold.done
      | apply Tokenizer.DigitsFold.next)

macro "derive_ascii_utf8" : tactic =>
  `(tactic| repeat first
    | exact Tokenizer.Utf8Scan.done
    | apply Tokenizer.Utf8Scan.next (Tokenizer.Utf8Head.one _ _ (by decide)))

macro "derive_ascii_chars" : tactic =>
  `(tactic| repeat first
    | exact Tokenizer.CharsScan.done (Tokenizer.ReverseTokens.result _)
    | apply Tokenizer.CharsScan.nextNone (Tokenizer.LexHead.space _ _ _ rfl)
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.ident <;>
          first | rfl | decide | derive_take_while | derive_utf8_length
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.numeral <;>
          first | rfl | decide | derive_take_while | derive_utf8_length | derive_digits
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.lt <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.lparen <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.rparen <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.lbrace <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.rbrace <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.semi <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.assign <;> first | rfl | decide
    | apply Tokenizer.CharsScan.nextToken
      · apply Tokenizer.LexHead.other <;> first | rfl | decide)

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
  ⟨LexingDerivation.intro (by
      unfold conditionalSource conditionalTokens sourceBytes sourceByte
      refine Tokenizer.TokenizeDerivation.intro
        (chars := ['0', ' ', '-', ' ', 'k', ' ', 'i', 'f', ' ', 'k', ' ', '<', ' ', '0',
          ' ', 'e', 'l', 's', 'e', ' ', 'k']) ?_ ?_
      · derive_ascii_utf8
      · derive_ascii_chars)
    ⟨rfl, spansPairwise_sound conditionalTokens rfl⟩ rfl, rfl⟩

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
  ⟨LexingDerivation.intro (by
      unfold completeSource completeTokens sourceBytes sourceByte
      refine Tokenizer.TokenizeDerivation.intro
        (chars := ['t', 'a', 'b', 'l', 'e', 'S', 'p', 'r', 'e', 'a', 'd', '(', 'k', ')',
          ' ', '{', ' ', 'p', 'o', 's', ' ', '=', ' ', '0', ' ', '-', ' ', 'k', ' ', 'i',
          'f', ' ', 'k', ' ', '<', ' ', '0', ' ', 'e', 'l', 's', 'e', ' ', 'k', ';', ' ',
          'r', 'e', 't', 'u', 'r', 'n', ' ', 'p', 'o', 's', ';', ' ', '}']) ?_ ?_
      · derive_ascii_utf8
      · derive_ascii_chars)
    ⟨rfl, spansPairwise_sound completeTokens rfl⟩ rfl,
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

#audit_empty Iota.Decode.parseExprFuel : Nat → Nat → List Iota.Decode.Token → Iota.Decode.Result (Iota.Expr × List Iota.Decode.Token)
#audit_empty Iota.Decode.parseFunction : String → Nat → List Iota.Decode.Token → Iota.Decode.Result (Iota.FunctionDecl × List Iota.Decode.Token)
#audit_empty Iota.Decode.decodeTokens : ByteArray → List Iota.Decode.Token → Nat → Iota.Decode.Result Iota.Decode.CertifiedDecoded
#audit_empty Iota.Decode.decodeTable : ByteArray → Iota.Decode.Result Iota.Decode.CertifiedDecoded
#audit_empty Iota.Source.DeclarationSpans : Nat → List Iota.Span → Prop
#audit_empty Iota.Source.GrammarDerivation : Iota.Source.SourceBytes → List Iota.Decode.Token → Nat → Iota.Decode.CertifiedDecoded → Prop
#audit_empty Iota.Source.GrammarDerivation.mk : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.LexingDerivation source tokens consumed → Iota.Decode.decodeTokens (Iota.Source.encodeBytes source) tokens consumed = .ok decoded → Iota.Source.DeclarationSpans source.length (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span)) → Iota.Source.GrammarDerivation source tokens consumed decoded
#audit_empty Iota.Source.GrammarDerivation.lexing : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.GrammarDerivation source tokens consumed decoded → Iota.Source.LexingDerivation source tokens consumed
#audit_empty Iota.Source.GrammarDerivation.decoding : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.GrammarDerivation source tokens consumed decoded → Iota.Decode.decodeTokens (Iota.Source.encodeBytes source) tokens consumed = .ok decoded
#audit_empty Iota.Source.GrammarDerivation.declarationSpans : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.GrammarDerivation source tokens consumed decoded → Iota.Source.DeclarationSpans source.length (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span))
#audit_empty Iota.Source.GrammarDerivation.sound : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.GrammarDerivation source tokens consumed decoded → Iota.Decode.decodeTable (Iota.Source.encodeBytes source) = .ok decoded
#audit_empty Iota.Source.GrammarDerivation.spansInside : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.GrammarDerivation source tokens consumed decoded → ((decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span)).all fun span => span.start < span.stop && span.stop ≤ source.length) = true
#audit_empty Iota.Source.GrammarDerivation.spansNonoverlapping : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {decoded : Iota.Decode.CertifiedDecoded}, Iota.Source.GrammarDerivation source tokens consumed decoded → (decoded.val.program.constants.map (·.span) ++ decoded.val.program.functions.map (·.span)).Pairwise (fun left right => left.stop ≤ right.start)
#audit_empty Iota.Source.ParsesDecl : Iota.Source.SourceBytes → List Iota.Decode.Token → Nat → Iota.Decode.Token → List Iota.Decode.Token → Iota.FunctionDecl → Prop
#audit_empty Iota.Source.ParsesDecl.mk : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.LexingDerivation source tokens consumed → tokens = header :: bodyTokens → header.kind = .ident decl.name → Iota.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, []) → decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length → Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl
#audit_empty Iota.Source.ParsesDecl.lexing : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → Iota.Source.LexingDerivation source tokens consumed
#audit_empty Iota.Source.ParsesDecl.tokenShape : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → tokens = header :: bodyTokens
#audit_empty Iota.Source.ParsesDecl.headerKind : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → header.kind = .ident decl.name
#audit_empty Iota.Source.ParsesDecl.parsed : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → Iota.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
#audit_empty Iota.Source.ParsesDecl.spanInside : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length
#audit_empty Iota.Source.ParsesDecl.sound : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → ∃ header bodyTokens, tokens = header :: bodyTokens ∧ header.kind = .ident decl.name ∧ Iota.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
#audit_empty Iota.Source.ParsesDecl.consumesSource : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → consumed = source.length
#audit_empty Iota.Source.ParsesDecl.consumesThroughClose : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → Iota.Decode.parseFunction decl.name header.span.start bodyTokens = .ok (decl, [])
#audit_empty Iota.Source.ParsesDecl.spanBounded : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {header : Iota.Decode.Token} {bodyTokens : List Iota.Decode.Token} {decl : Iota.FunctionDecl}, Iota.Source.ParsesDecl source tokens consumed header bodyTokens decl → decl.span.start < decl.span.stop ∧ decl.span.stop ≤ source.length
#audit_empty Iota.Source.ParsesExpr : Iota.Source.SourceBytes → List Iota.Decode.Token → Nat → Iota.Expr → Prop
#audit_empty Iota.Source.ParsesExpr.mk : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {expression : Iota.Expr}, Iota.Source.LexingDerivation source tokens consumed → Iota.Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, []) → Iota.Source.ParsesExpr source tokens consumed expression
#audit_empty Iota.Source.ParsesExpr.lexing : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {expression : Iota.Expr}, Iota.Source.ParsesExpr source tokens consumed expression → Iota.Source.LexingDerivation source tokens consumed
#audit_empty Iota.Source.ParsesExpr.parsed : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {expression : Iota.Expr}, Iota.Source.ParsesExpr source tokens consumed expression → Iota.Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, [])
#audit_empty Iota.Source.ParsesExpr.sound : ∀ {source : Iota.Source.SourceBytes} {tokens : List Iota.Decode.Token} {consumed : Nat} {expression : Iota.Expr}, Iota.Source.ParsesExpr source tokens consumed expression → Iota.Decode.parseExprFuel (tokens.length * 4 + 16) 0 tokens = .ok (expression, [])
#audit_empty Iota.Source.GrammarAudit.conditionalSource : Iota.Source.SourceBytes
#audit_empty Iota.Source.GrammarAudit.conditionalTokens : List Iota.Decode.Token
#audit_empty Iota.Source.GrammarAudit.groupedConditional : Iota.Expr
#audit_empty Iota.Source.GrammarAudit.incorrectlyGroupedConditional : Iota.Expr
#audit_empty Iota.Source.GrammarAudit.conditional_derivation : Iota.Source.ParsesExpr Iota.Source.GrammarAudit.conditionalSource Iota.Source.GrammarAudit.conditionalTokens 21 Iota.Source.GrammarAudit.groupedConditional
#audit_empty Iota.Source.GrammarAudit.conditional_groups_after_subtraction : Iota.Decode.parseExprFuel (Iota.Source.GrammarAudit.conditionalTokens.length * 4 + 16) 0 Iota.Source.GrammarAudit.conditionalTokens = .ok (Iota.Source.GrammarAudit.groupedConditional, [])
#audit_empty Iota.Source.GrammarAudit.conditional_not_grouped_as_subtraction_rhs : Iota.Source.GrammarAudit.groupedConditional ≠ Iota.Source.GrammarAudit.incorrectlyGroupedConditional
#audit_empty Iota.Source.GrammarAudit.completeSource : Iota.Source.SourceBytes
#audit_empty Iota.Source.GrammarAudit.completeTokens : List Iota.Decode.Token
#audit_empty Iota.Source.GrammarAudit.completeDecl : Iota.FunctionDecl
#audit_empty Iota.Source.GrammarAudit.completeHeader : Iota.Decode.Token
#audit_empty Iota.Source.GrammarAudit.complete_selected_body_derivation : Iota.Source.ParsesDecl Iota.Source.GrammarAudit.completeSource Iota.Source.GrammarAudit.completeTokens 59 Iota.Source.GrammarAudit.completeHeader Iota.Source.GrammarAudit.completeTokens.tail Iota.Source.GrammarAudit.completeDecl
#audit_empty Iota.Source.GrammarAudit.complete_selected_body_zero_leftover : Iota.Decode.parseFunction Iota.Source.GrammarAudit.completeDecl.name 0 Iota.Source.GrammarAudit.completeTokens.tail = .ok (Iota.Source.GrammarAudit.completeDecl, [])
#audit_empty Iota.Source.GrammarAudit.truncatedSource : Iota.Source.SourceBytes
#audit_empty Iota.Source.GrammarAudit.truncatedTokens : List Iota.Decode.Token
#audit_empty Iota.Source.GrammarAudit.truncated_selected_body_prefix_rejected : Iota.Decode.tokenize (Iota.Source.encodeBytes Iota.Source.GrammarAudit.truncatedSource) = .ok (Iota.Source.GrammarAudit.truncatedTokens, 45) ∧ (match Iota.Decode.parseFunction "tableSpread" 0 Iota.Source.GrammarAudit.truncatedTokens.tail with | .error _ => true | .ok _ => false) = true

theorem nat_add_zero : (0 : Nat) + 0 = 0 := rfl
#audit_empty Iota.Source.GrammarAudit.nat_add_zero : (0 : Nat) + 0 = 0

end Iota.Source.GrammarAudit
