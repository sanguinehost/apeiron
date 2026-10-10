import Apeiron.Linter.Notation

/-!
# A bounded arithmetic parser

@name Arithmetic Parser
@grade proved

Multiplication is parsed inside addition, so `1 + 2 * 3` returns `add (lit 1) (mul (lit 2)
(lit 3))`. Each successful result carries the exact unconsumed suffix. Three independent grammar
relations state the primary, multiplicative and additive tiers.

This model covers natural literals, nested parentheses, multiplication and addition. Iota token
arenas, machine integers and the production parser remain separate refinement obligations.
-/

namespace Apeiron
namespace Parser
namespace Arithmetic

/-- One token in the arithmetic grammar.

    Token [one arithmetic input token] -/
inductive Token where
  | nat (n : Nat)
  | plus
  | times
  | lparen
  | rparen
  deriving DecidableEq, Repr

/-- One typed arithmetic syntax tree.

    Expr [one arithmetic expression tree] -/
inductive Expr where
  | lit (n : Nat)
  | add (left right : Expr)
  | mul (left right : Expr)
  deriving DecidableEq, Repr

/-- Number of constructors in an arithmetic tree.

    `Expr` [one arithmetic expression tree]
    `→` [takes the left and gives the right] -/
def Expr.nodes : Expr → Nat
  | .lit _ => 1
  | .add left right | .mul left right => left.nodes + right.nodes + 1

/-- Success with an exact suffix, syntax refusal, or exhausted fuel.

    Result [one bounded parser result] -/
inductive Result where
  | ok (tree : Expr) (rest : List Token)
  | syntax
  | fuel
  deriving DecidableEq, Repr

/-- The suffix does not start with multiplication.

    NotTimes input [input has no leading multiplication token]

    `List` [a finite sequence of things, in order]
    `→` [if the left holds so does the right] -/
inductive NotTimes : List Token → Prop where
  | nil : NotTimes []
  | nat (n : Nat) (rest : List Token) : NotTimes (.nat n :: rest)
  | plus (rest : List Token) : NotTimes (.plus :: rest)
  | lparen (rest : List Token) : NotTimes (.lparen :: rest)
  | rparen (rest : List Token) : NotTimes (.rparen :: rest)

/-- The suffix does not start with addition.

    NotPlus input [input has no leading addition token]

    `List` [a finite sequence of things, in order]
    `→` [if the left holds so does the right] -/
inductive NotPlus : List Token → Prop where
  | nil : NotPlus []
  | nat (n : Nat) (rest : List Token) : NotPlus (.nat n :: rest)
  | times (rest : List Token) : NotPlus (.times :: rest)
  | lparen (rest : List Token) : NotPlus (.lparen :: rest)
  | rparen (rest : List Token) : NotPlus (.rparen :: rest)

mutual
  /-- The primary grammar tier.

      PrimaryDerives input tree rest [input derives a primary tree and leaves rest]

      `List` [a finite sequence of things, in order]
      `Expr` [one arithmetic expression tree]
      `→` [if the left holds so does the right] -/
  inductive PrimaryDerives : List Token → Expr → List Token → Prop where
    | lit (n : Nat) (rest : List Token) : PrimaryDerives (.nat n :: rest) (.lit n) rest
    | group {inside rest : List Token} {tree : Expr} :
        ExprDerives inside tree (.rparen :: rest) → PrimaryDerives (.lparen :: inside) tree rest

  inductive MulTailDerives : Expr → List Token → Expr → List Token → Prop where
    | stop {left : Expr} {input : List Token} : NotTimes input →
        MulTailDerives left input left input
    | step {left right tree : Expr} {input middle rest : List Token} :
        PrimaryDerives input right middle → MulTailDerives (.mul left right) middle tree rest →
          MulTailDerives left (.times :: input) tree rest

  inductive MulDerives : List Token → Expr → List Token → Prop where
    | make {input middle rest : List Token} {first tree : Expr} :
        PrimaryDerives input first middle → MulTailDerives first middle tree rest →
          MulDerives input tree rest

  inductive ExprTailDerives : Expr → List Token → Expr → List Token → Prop where
    | stop {left : Expr} {input : List Token} : NotPlus input →
        ExprTailDerives left input left input
    | step {left right tree : Expr} {input middle rest : List Token} :
        MulDerives input right middle → ExprTailDerives (.add left right) middle tree rest →
          ExprTailDerives left (.plus :: input) tree rest

  inductive ExprDerives : List Token → Expr → List Token → Prop where
    | make {input middle rest : List Token} {first tree : Expr} :
        MulDerives input first middle → ExprTailDerives first middle tree rest →
          ExprDerives input tree rest
end

/-- A literal consumes its token.

    `≤` [is smaller than or equal to] -/
private theorem suffix_lit (n) (rest) : rest.length ≤ (Token.nat n :: rest).length := by simp
/-- Group delimiters preserve the contained derivation's suffix ordering.

    `≤` [is smaller than or equal to] -/
private theorem suffix_group {inside rest tree} (_ : ExprDerives inside tree (.rparen :: rest))
    (ih : (.rparen :: rest).length ≤ inside.length) : rest.length ≤ (.lparen :: inside).length := by
  simp only [List.length_cons] at *; omega
/-- A stopped multiplication tail consumes no token.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem suffix_mul_stop (left : Expr) {input} (_ : NotTimes input) :
    input.length ≤ input.length := by
  cases left <;> exact Nat.le_refl _
/-- A multiplication step composes its two suffix orderings.

    `≤` [is smaller than or equal to] -/
private theorem suffix_mul_step {left right tree input middle rest}
    (_ : PrimaryDerives input right middle)
    (_ : MulTailDerives (.mul left right) middle tree rest)
    (hp : middle.length ≤ input.length) (ht : rest.length ≤ middle.length) :
    rest.length ≤ (.times :: input).length := by
  simp only [List.length_cons]; omega
/-- A multiplication tier composes its primary and tail suffix orderings.

    `≤` [is smaller than or equal to] -/
private theorem suffix_mul_make {input middle rest first tree}
    (_ : PrimaryDerives input first middle) (_ : MulTailDerives first middle tree rest)
    (hp : middle.length ≤ input.length) (ht : rest.length ≤ middle.length) :
    rest.length ≤ input.length := by omega
/-- A stopped addition tail consumes no token.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem suffix_expr_stop (left : Expr) {input} (_ : NotPlus input) :
    input.length ≤ input.length := by
  cases left <;> exact Nat.le_refl _
/-- An addition step composes its two suffix orderings.

    `≤` [is smaller than or equal to] -/
private theorem suffix_expr_step {left right tree input middle rest}
    (_ : MulDerives input right middle)
    (_ : ExprTailDerives (.add left right) middle tree rest)
    (hm : middle.length ≤ input.length) (ht : rest.length ≤ middle.length) :
    rest.length ≤ (.plus :: input).length := by
  simp only [List.length_cons]; omega
/-- An expression tier composes its multiplication and tail suffix orderings.

    `≤` [is smaller than or equal to] -/
private theorem suffix_expr_make {input middle rest first tree}
    (_ : MulDerives input first middle) (_ : ExprTailDerives first middle tree rest)
    (hm : middle.length ≤ input.length) (ht : rest.length ≤ middle.length) :
    rest.length ≤ input.length := by omega

/-- A primary derivation leaves a suffix no longer than its input.

    `≤` [is smaller than or equal to] -/
theorem primary_suffix_length {input tree rest} (h : PrimaryDerives input tree rest) :
    rest.length ≤ input.length :=
  @PrimaryDerives.rec (fun i _ r _ => r.length ≤ i.length)
    (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) suffix_lit suffix_group
    (fun {left input} h => suffix_mul_stop left (input := input) h) suffix_mul_step
    suffix_mul_make (fun {left input} h => suffix_expr_stop left (input := input) h)
    suffix_expr_step suffix_expr_make input tree rest h

/-- A multiplication-tail derivation leaves a suffix no longer than its input.

    `≤` [is smaller than or equal to] -/
theorem mulTail_suffix_length {left input tree rest} (h : MulTailDerives left input tree rest) :
    rest.length ≤ input.length :=
  @MulTailDerives.rec (fun i _ r _ => r.length ≤ i.length)
    (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) suffix_lit suffix_group
    (fun {left input} h => suffix_mul_stop left (input := input) h) suffix_mul_step
    suffix_mul_make (fun {left input} h => suffix_expr_stop left (input := input) h)
    suffix_expr_step suffix_expr_make left input tree rest h

/-- A multiplication derivation leaves a suffix no longer than its input.

    `≤` [is smaller than or equal to] -/
theorem mul_suffix_length {input tree rest} (h : MulDerives input tree rest) :
    rest.length ≤ input.length :=
  @MulDerives.rec (fun i _ r _ => r.length ≤ i.length)
    (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) suffix_lit suffix_group
    (fun {left input} h => suffix_mul_stop left (input := input) h) suffix_mul_step
    suffix_mul_make (fun {left input} h => suffix_expr_stop left (input := input) h)
    suffix_expr_step suffix_expr_make input tree rest h

/-- An addition-tail derivation leaves a suffix no longer than its input.

    `≤` [is smaller than or equal to] -/
theorem exprTail_suffix_length {left input tree rest} (h : ExprTailDerives left input tree rest) :
    rest.length ≤ input.length :=
  @ExprTailDerives.rec (fun i _ r _ => r.length ≤ i.length)
    (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) suffix_lit suffix_group
    (fun {left input} h => suffix_mul_stop left (input := input) h) suffix_mul_step
    suffix_mul_make (fun {left input} h => suffix_expr_stop left (input := input) h)
    suffix_expr_step suffix_expr_make left input tree rest h

/-- An expression derivation leaves a suffix no longer than its input.

    `≤` [is smaller than or equal to] -/
theorem expr_suffix_length {input tree rest} (h : ExprDerives input tree rest) :
    rest.length ≤ input.length :=
  @ExprDerives.rec (fun i _ r _ => r.length ≤ i.length)
    (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) (fun _ i _ r _ => r.length ≤ i.length)
    (fun i _ r _ => r.length ≤ i.length) suffix_lit suffix_group
    (fun {left input} h => suffix_mul_stop left (input := input) h) suffix_mul_step
    suffix_mul_make (fun {left input} h => suffix_expr_stop left (input := input) h)
    suffix_expr_step suffix_expr_make input tree rest h

/-- Literal node-count case.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem nodes_lit (n) (r) :
    (Expr.lit n).nodes + r.length ≤ (Token.nat n :: r).length := by
  simp [Expr.nodes, Nat.add_comm]
/-- Group node-count case.

    `≤` [is smaller than or equal to] -/
private theorem nodes_group {i r t} (_ : ExprDerives i t (.rparen :: r))
    (ih : t.nodes + (.rparen :: r).length ≤ i.length) :
    t.nodes + r.length ≤ (.lparen :: i).length := by simp at *; omega
/-- Multiplication-tail stop node-count case.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem nodes_mt_stop (l : Expr) {i} (_ : NotTimes i) :
    l.nodes + i.length ≤ l.nodes + i.length := Nat.le_refl _
/-- Multiplication-tail step node-count case.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem nodes_mt_step {l q t i m r} (_hp : PrimaryDerives i q m)
    (_ : MulTailDerives (.mul l q) m t r) (ip : q.nodes + m.length ≤ i.length)
    (it : t.nodes + r.length ≤ (Expr.mul l q).nodes + m.length) :
    t.nodes + r.length ≤ l.nodes + (.times :: i).length := by
  simp only [Expr.nodes, List.length_cons] at it ⊢; omega
/-- Multiplication tier node-count case.

    `≤` [is smaller than or equal to] -/
private theorem nodes_m_make {i m r q t} (_ : PrimaryDerives i q m)
    (_ : MulTailDerives q m t r) (ip : q.nodes + m.length ≤ i.length)
    (it : t.nodes + r.length ≤ q.nodes + m.length) : t.nodes + r.length ≤ i.length := by omega
/-- Addition-tail stop node-count case.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem nodes_et_stop (l : Expr) {i} (_ : NotPlus i) :
    l.nodes + i.length ≤ l.nodes + i.length := Nat.le_refl _
/-- Addition-tail step node-count case.

    `Expr` [one arithmetic expression tree]
    `≤` [is smaller than or equal to] -/
private theorem nodes_et_step {l q t i m r} (_ : MulDerives i q m)
    (_ : ExprTailDerives (.add l q) m t r) (im : q.nodes + m.length ≤ i.length)
    (it : t.nodes + r.length ≤ (Expr.add l q).nodes + m.length) :
    t.nodes + r.length ≤ l.nodes + (.plus :: i).length := by
  simp only [Expr.nodes, List.length_cons] at it ⊢; omega
/-- Expression tier node-count case.

    `≤` [is smaller than or equal to] -/
private theorem nodes_e_make {i m r q t} (_ : MulDerives i q m)
    (_ : ExprTailDerives q m t r) (im : q.nodes + m.length ≤ i.length)
    (it : t.nodes + r.length ≤ q.nodes + m.length) : t.nodes + r.length ≤ i.length := by omega

/-- Primary constructors plus suffix tokens fit within input tokens.

    `≤` [is smaller than or equal to] -/
theorem primary_node_bound {i t r} (h : PrimaryDerives i t r) :
    t.nodes + r.length ≤ i.length :=
  @PrimaryDerives.rec (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length) nodes_lit nodes_group
    (fun {left input} h => nodes_mt_stop left (i := input) h) nodes_mt_step nodes_m_make
    (fun {left input} h => nodes_et_stop left (i := input) h) nodes_et_step nodes_e_make i t r h
/-- A multiplication tail adds at most its input tokens to its incoming tree.

    `≤` [is smaller than or equal to] -/
theorem mulTail_node_bound {l i t r} (h : MulTailDerives l i t r) :
    t.nodes + r.length ≤ l.nodes + i.length :=
  @MulTailDerives.rec (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length) nodes_lit nodes_group
    (fun {left input} h => nodes_mt_stop left (i := input) h) nodes_mt_step nodes_m_make
    (fun {left input} h => nodes_et_stop left (i := input) h) nodes_et_step nodes_e_make l i t r h
/-- Multiplication constructors plus suffix tokens fit within input tokens.

    `≤` [is smaller than or equal to] -/
theorem mul_node_bound {i t r} (h : MulDerives i t r) : t.nodes + r.length ≤ i.length :=
  @MulDerives.rec (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length) nodes_lit nodes_group
    (fun {left input} h => nodes_mt_stop left (i := input) h) nodes_mt_step nodes_m_make
    (fun {left input} h => nodes_et_stop left (i := input) h) nodes_et_step nodes_e_make i t r h
/-- An addition tail adds at most its input tokens to its incoming tree.

    `≤` [is smaller than or equal to] -/
theorem exprTail_node_bound {l i t r} (h : ExprTailDerives l i t r) :
    t.nodes + r.length ≤ l.nodes + i.length :=
  @ExprTailDerives.rec (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length) nodes_lit nodes_group
    (fun {left input} h => nodes_mt_stop left (i := input) h) nodes_mt_step nodes_m_make
    (fun {left input} h => nodes_et_stop left (i := input) h) nodes_et_step nodes_e_make l i t r h
/-- Expression constructors plus suffix tokens fit within input tokens.

    `≤` [is smaller than or equal to] -/
theorem expr_node_bound {i t r} (h : ExprDerives i t r) : t.nodes + r.length ≤ i.length :=
  @ExprDerives.rec (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length)
    (fun l i t r _ => t.nodes + r.length ≤ l.nodes + i.length)
    (fun i t r _ => t.nodes + r.length ≤ i.length) nodes_lit nodes_group
    (fun {left input} h => nodes_mt_stop left (i := input) h) nodes_mt_step nodes_m_make
    (fun {left input} h => nodes_et_stop left (i := input) h) nodes_et_step nodes_e_make i t r h

/-- A token-sized arena has capacity for every constructor of a derived expression.

    `Nat` [a natural number]
    `≤` [is smaller than or equal to]
    `→` [if the left holds so does the right] -/
theorem expr_capacity {i t r} (h : ExprDerives i t r) {slot capacity : Nat}
    (fits : slot + i.length ≤ capacity) : slot + t.nodes ≤ capacity := by
  have hn := expr_node_bound h
  omega

mutual
  /-- Parse one primary under finite fuel.

      Nat → List Token → Result [fuel and input give one bounded result]

      `List` [a finite sequence of things, in order]
      `→` [takes the left and gives the right] -/
  def parsePrimary : Nat → List Token → Result
    | 0, _ => .fuel
    | _ + 1, [] => .syntax
    | _ + 1, .nat n :: rest => .ok (.lit n) rest
    | fuel + 1, .lparen :: inside =>
        match parseExpr fuel inside with
        | .ok tree (.rparen :: rest) => .ok tree rest
        | .fuel => .fuel
        | _ => .syntax
    | _ + 1, _ => .syntax

  /-- Parse one multiplication tier under finite fuel. -/
  def parseMul : Nat → List Token → Result
    | 0, _ => .fuel
    | fuel + 1, input =>
        match parsePrimary fuel input with
        | .ok left rest => parseMulTail fuel left rest
        | result => result

  def parseMulTail : Nat → Expr → List Token → Result
    | 0, _, _ => .fuel
    | fuel + 1, left, .times :: input =>
        match parsePrimary fuel input with
        | .ok right rest => parseMulTail fuel (.mul left right) rest
        | result => result
    | _ + 1, left, input => .ok left input

  /-- Parse one addition tier under finite fuel. -/
  def parseExpr : Nat → List Token → Result
    | 0, _ => .fuel
    | fuel + 1, input =>
        match parseMul fuel input with
        | .ok left rest => parseExprTail fuel left rest
        | result => result

  def parseExprTail : Nat → Expr → List Token → Result
    | 0, _, _ => .fuel
    | fuel + 1, left, .plus :: input =>
        match parseMul fuel input with
        | .ok right rest => parseExprTail fuel (.add left right) rest
        | result => result
    | _ + 1, left, input => .ok left input
end

/-- Accept only a parse consuming the complete input.

    (fuel : Nat) (input : List Token) [the budget and complete token input]
    Result [the bounded result]

    `List` [a finite sequence of things, in order] -/
def parseComplete (fuel : Nat) (input : List Token) : Result :=
  match parseExpr fuel input with
  | .ok tree [] => .ok tree []
  | .fuel => .fuel
  | _ => .syntax

/-- Every successful parser tier derives the same tree and exact suffix.

    parsePrimary fuel input = .ok tree rest [primary success with its suffix]
    PrimaryDerives input tree rest [the primary grammar derives the same result]
    parseMul fuel input = .ok tree rest [multiplication success with its suffix]
    MulDerives input tree rest [the multiplication grammar derives the same result]
    parseExpr fuel input = .ok tree rest [addition success with its suffix]
    ExprDerives input tree rest [the addition grammar derives the same result]

    `Nat` [a natural number]
    `List` [a finite sequence of things, in order]
    `Expr` [one arithmetic expression tree]
    `∀` [for every value]
    `∧` [both are true]
    `→` [if the left holds so does the right] -/
theorem parse_sound : ∀ fuel,
    (∀ input tree rest, parsePrimary fuel input = .ok tree rest → PrimaryDerives input tree rest) ∧
    (∀ left input tree rest, parseMulTail fuel left input = .ok tree rest →
      MulTailDerives left input tree rest) ∧
    (∀ input tree rest, parseMul fuel input = .ok tree rest → MulDerives input tree rest) ∧
    (∀ left input tree rest, parseExprTail fuel left input = .ok tree rest →
      ExprTailDerives left input tree rest) ∧
    (∀ input tree rest, parseExpr fuel input = .ok tree rest → ExprDerives input tree rest) := by
  intro fuel
  induction fuel with
  | zero => simp [parsePrimary, parseMulTail, parseMul, parseExprTail, parseExpr]
  | succ fuel ih =>
      rcases ih with ⟨primarySound, mulTailSound, mulSound, exprTailSound, exprSound⟩
      constructor
      · intro input tree rest h
        cases input with
        | nil => simp [parsePrimary] at h
        | cons token input =>
            cases token with
            | nat n =>
                simp [parsePrimary] at h
                obtain ⟨rfl, rfl⟩ := h
                exact .lit n input
            | plus => simp [parsePrimary] at h
            | times => simp [parsePrimary] at h
            | rparen => simp [parsePrimary] at h
            | lparen =>
                simp only [parsePrimary] at h
                cases hp : parseExpr fuel input with
                | «syntax» => simp [hp] at h
                | fuel => simp [hp] at h
                | ok inner suffix =>
                    cases suffix with
                    | nil => simp [hp] at h
                    | cons close suffix =>
                        cases close with
                        | rparen =>
                            simp [hp] at h
                            obtain ⟨rfl, rfl⟩ := h
                            exact .group (exprSound _ _ _ hp)
                        | nat n => simp [hp] at h
                        | plus => simp [hp] at h
                        | times => simp [hp] at h
                        | lparen => simp [hp] at h
      constructor
      · intro left input tree rest h
        cases input with
        | nil => simp [parseMulTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop .nil
        | cons token input =>
            cases token with
            | times =>
                simp only [parseMulTail] at h
                cases hp : parsePrimary fuel input with
                | «syntax» => simp [hp] at h
                | fuel => simp [hp] at h
                | ok right middle =>
                    simp [hp] at h
                    exact .step (primarySound _ _ _ hp) (mulTailSound _ _ _ _ h)
            | nat n => simp [parseMulTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.nat n input)
            | plus => simp [parseMulTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.plus input)
            | lparen =>
                simp [parseMulTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.lparen input)
            | rparen =>
                simp [parseMulTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.rparen input)
      constructor
      · intro input tree rest h
        simp only [parseMul] at h
        cases hp : parsePrimary fuel input with
        | «syntax» => simp [hp] at h
        | fuel => simp [hp] at h
        | ok first middle =>
            simp [hp] at h
            exact .make (primarySound _ _ _ hp) (mulTailSound _ _ _ _ h)
      constructor
      · intro left input tree rest h
        cases input with
        | nil => simp [parseExprTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop .nil
        | cons token input =>
            cases token with
            | plus =>
                simp only [parseExprTail] at h
                cases hm : parseMul fuel input with
                | «syntax» => simp [hm] at h
                | fuel => simp [hm] at h
                | ok right middle =>
                    simp [hm] at h
                    exact .step (mulSound _ _ _ hm) (exprTailSound _ _ _ _ h)
            | nat n => simp [parseExprTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.nat n input)
            | times => simp [parseExprTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.times input)
            | lparen =>
                simp [parseExprTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.lparen input)
            | rparen =>
                simp [parseExprTail] at h; obtain ⟨rfl, rfl⟩ := h; exact .stop (.rparen input)
      · intro input tree rest h
        simp only [parseExpr] at h
        cases hm : parseMul fuel input with
        | «syntax» => simp [hm] at h
        | fuel => simp [hm] at h
        | ok first middle =>
            simp [hm] at h
            exact .make (mulSound _ _ _ hm) (exprTailSound _ _ _ _ h)

/-- Eventual exact success for a primary derivation.

    `Expr` [one arithmetic expression tree]
    `List` [a finite sequence of things, in order] -/
private abbrev PrimaryComplete (input : List Token) (tree : Expr) (rest : List Token) :=
  ∃ bound, ∀ fuel, bound ≤ fuel → parsePrimary fuel input = .ok tree rest
/-- Eventual exact success for a multiplication tail derivation.

    `Expr` [one arithmetic expression tree]
    `List` [a finite sequence of things, in order] -/
private abbrev MulTailComplete (left : Expr) (input : List Token) (tree : Expr)
    (rest : List Token) :=
  ∃ bound, ∀ fuel, bound ≤ fuel → parseMulTail fuel left input = .ok tree rest
/-- Eventual exact success for a multiplication derivation.

    `Expr` [one arithmetic expression tree]
    `List` [a finite sequence of things, in order] -/
private abbrev MulComplete (input : List Token) (tree : Expr) (rest : List Token) :=
  ∃ bound, ∀ fuel, bound ≤ fuel → parseMul fuel input = .ok tree rest
/-- Eventual exact success for an addition tail derivation.

    `Expr` [one arithmetic expression tree]
    `List` [a finite sequence of things, in order] -/
private abbrev ExprTailComplete (left : Expr) (input : List Token) (tree : Expr)
    (rest : List Token) :=
  ∃ bound, ∀ fuel, bound ≤ fuel → parseExprTail fuel left input = .ok tree rest
/-- Eventual exact success for an expression derivation.

    `Expr` [one arithmetic expression tree]
    `List` [a finite sequence of things, in order] -/
private abbrev ExprComplete (input : List Token) (tree : Expr) (rest : List Token) :=
  ∃ bound, ∀ fuel, bound ≤ fuel → parseExpr fuel input = .ok tree rest

/-- A literal derivation succeeds with one unit of fuel. -/
private theorem complete_lit (n) (rest) : PrimaryComplete (.nat n :: rest) (.lit n) rest :=
  ⟨1, by intro fuel hf; cases fuel <;> simp_all [parsePrimary]⟩
/-- A group adds one fuel unit to its contained expression bound. -/
private theorem complete_group {inside rest tree} (_ : ExprDerives inside tree (.rparen :: rest))
    (ih : ExprComplete inside tree (.rparen :: rest)) :
    PrimaryComplete (.lparen :: inside) tree rest := by
  obtain ⟨bound, ih⟩ := ih
  exact ⟨bound + 1, by intro fuel hf; cases fuel with
    | zero => omega
    | succ fuel => simp only [parsePrimary]; rw [ih fuel (by omega)]⟩
/-- A stopped multiplication tail succeeds with one unit of fuel. -/
private theorem complete_mul_stop {left input} (h : NotTimes input) :
    MulTailComplete left input left input :=
  ⟨1, by intro fuel hf; cases fuel with | zero => omega | succ fuel => cases h <;> rfl⟩
/-- A multiplication step uses one more than both child fuel bounds. -/
private theorem complete_mul_step {left right tree input middle rest}
    (_ : PrimaryDerives input right middle) (_ : MulTailDerives (.mul left right) middle tree rest)
    (ihp : PrimaryComplete input right middle)
    (iht : MulTailComplete (.mul left right) middle tree rest) :
    MulTailComplete left (.times :: input) tree rest := by
  obtain ⟨bp, ihp⟩ := ihp; obtain ⟨bt, iht⟩ := iht
  exact ⟨Nat.max bp bt + 1, by intro fuel hf; cases fuel with
    | zero => omega
    | succ fuel =>
      have hx := Nat.le_of_succ_le_succ hf
      simp only [parseMulTail]; rw [ihp fuel (Nat.le_trans (Nat.le_max_left _ _) hx)]
      exact iht fuel (Nat.le_trans (Nat.le_max_right _ _) hx)⟩
/-- A multiplication tier uses one more than its primary and tail bounds. -/
private theorem complete_mul_make {input middle rest first tree}
    (_ : PrimaryDerives input first middle) (_ : MulTailDerives first middle tree rest)
    (ihp : PrimaryComplete input first middle) (iht : MulTailComplete first middle tree rest) :
    MulComplete input tree rest := by
  obtain ⟨bp, ihp⟩ := ihp; obtain ⟨bt, iht⟩ := iht
  exact ⟨Nat.max bp bt + 1, by intro fuel hf; cases fuel with
    | zero => omega
    | succ fuel =>
      have hx := Nat.le_of_succ_le_succ hf
      simp only [parseMul]; rw [ihp fuel (Nat.le_trans (Nat.le_max_left _ _) hx)]
      exact iht fuel (Nat.le_trans (Nat.le_max_right _ _) hx)⟩
/-- A stopped addition tail succeeds with one unit of fuel. -/
private theorem complete_expr_stop {left input} (h : NotPlus input) :
    ExprTailComplete left input left input :=
  ⟨1, by intro fuel hf; cases fuel with | zero => omega | succ fuel => cases h <;> rfl⟩
/-- An addition step uses one more than both child fuel bounds. -/
private theorem complete_expr_step {left right tree input middle rest}
    (_ : MulDerives input right middle) (_ : ExprTailDerives (.add left right) middle tree rest)
    (ihm : MulComplete input right middle)
    (iht : ExprTailComplete (.add left right) middle tree rest) :
    ExprTailComplete left (.plus :: input) tree rest := by
  obtain ⟨bm, ihm⟩ := ihm; obtain ⟨bt, iht⟩ := iht
  exact ⟨Nat.max bm bt + 1, by intro fuel hf; cases fuel with
    | zero => omega
    | succ fuel =>
      have hx := Nat.le_of_succ_le_succ hf
      simp only [parseExprTail]; rw [ihm fuel (Nat.le_trans (Nat.le_max_left _ _) hx)]
      exact iht fuel (Nat.le_trans (Nat.le_max_right _ _) hx)⟩
/-- An expression tier uses one more than its multiplication and tail bounds. -/
private theorem complete_expr_make {input middle rest first tree}
    (_ : MulDerives input first middle) (_ : ExprTailDerives first middle tree rest)
    (ihm : MulComplete input first middle) (iht : ExprTailComplete first middle tree rest) :
    ExprComplete input tree rest := by
  obtain ⟨bm, ihm⟩ := ihm; obtain ⟨bt, iht⟩ := iht
  exact ⟨Nat.max bm bt + 1, by intro fuel hf; cases fuel with
    | zero => omega
    | succ fuel =>
      have hx := Nat.le_of_succ_le_succ hf
      simp only [parseExpr]; rw [ihm fuel (Nat.le_trans (Nat.le_max_left _ _) hx)]
      exact iht fuel (Nat.le_trans (Nat.le_max_right _ _) hx)⟩

/-- Every primary derivation succeeds with its exact tree and suffix above a finite fuel bound.

    `∃` [there is at least one]
    `∀` [for every value]
    `≤` [is smaller than or equal to]
    `→` [if the left holds so does the right] -/
theorem primary_complete {input tree rest} (h : PrimaryDerives input tree rest) :
    ∃ bound, ∀ fuel, bound ≤ fuel → parsePrimary fuel input = .ok tree rest :=
  @PrimaryDerives.rec (fun i t r _ => PrimaryComplete i t r)
    (fun l i t r _ => MulTailComplete l i t r) (fun i t r _ => MulComplete i t r)
    (fun l i t r _ => ExprTailComplete l i t r) (fun i t r _ => ExprComplete i t r)
    complete_lit complete_group complete_mul_stop complete_mul_step complete_mul_make
    complete_expr_stop complete_expr_step complete_expr_make input tree rest h

/-- Every multiplication-tail derivation succeeds exactly above a finite fuel bound.

    `∃` [there is at least one]
    `∀` [for every value]
    `≤` [is smaller than or equal to]
    `→` [if the left holds so does the right] -/
theorem mulTail_complete {left input tree rest} (h : MulTailDerives left input tree rest) :
    ∃ bound, ∀ fuel, bound ≤ fuel → parseMulTail fuel left input = .ok tree rest :=
  @MulTailDerives.rec (fun i t r _ => PrimaryComplete i t r)
    (fun l i t r _ => MulTailComplete l i t r) (fun i t r _ => MulComplete i t r)
    (fun l i t r _ => ExprTailComplete l i t r) (fun i t r _ => ExprComplete i t r)
    complete_lit complete_group complete_mul_stop complete_mul_step complete_mul_make
    complete_expr_stop complete_expr_step complete_expr_make left input tree rest h

/-- Every multiplication derivation succeeds with its exact tree and suffix above finite fuel.

    `∃` [there is at least one]
    `∀` [for every value]
    `≤` [is smaller than or equal to]
    `→` [if the left holds so does the right] -/
theorem mul_complete {input tree rest} (h : MulDerives input tree rest) :
    ∃ bound, ∀ fuel, bound ≤ fuel → parseMul fuel input = .ok tree rest :=
  @MulDerives.rec (fun i t r _ => PrimaryComplete i t r)
    (fun l i t r _ => MulTailComplete l i t r) (fun i t r _ => MulComplete i t r)
    (fun l i t r _ => ExprTailComplete l i t r) (fun i t r _ => ExprComplete i t r)
    complete_lit complete_group complete_mul_stop complete_mul_step complete_mul_make
    complete_expr_stop complete_expr_step complete_expr_make input tree rest h

/-- Every addition-tail derivation succeeds exactly above a finite fuel bound.

    `∃` [there is at least one]
    `∀` [for every value]
    `≤` [is smaller than or equal to]
    `→` [if the left holds so does the right] -/
theorem exprTail_complete {left input tree rest} (h : ExprTailDerives left input tree rest) :
    ∃ bound, ∀ fuel, bound ≤ fuel → parseExprTail fuel left input = .ok tree rest :=
  @ExprTailDerives.rec (fun i t r _ => PrimaryComplete i t r)
    (fun l i t r _ => MulTailComplete l i t r) (fun i t r _ => MulComplete i t r)
    (fun l i t r _ => ExprTailComplete l i t r) (fun i t r _ => ExprComplete i t r)
    complete_lit complete_group complete_mul_stop complete_mul_step complete_mul_make
    complete_expr_stop complete_expr_step complete_expr_make left input tree rest h

/-- Every expression derivation succeeds with its exact tree and suffix above finite fuel.

    `∃` [there is at least one]
    `∀` [for every value]
    `≤` [is smaller than or equal to]
    `→` [if the left holds so does the right] -/
theorem expr_complete {input tree rest} (h : ExprDerives input tree rest) :
    ∃ bound, ∀ fuel, bound ≤ fuel → parseExpr fuel input = .ok tree rest :=
  @ExprDerives.rec (fun i t r _ => PrimaryComplete i t r)
    (fun l i t r _ => MulTailComplete l i t r) (fun i t r _ => MulComplete i t r)
    (fun l i t r _ => ExprTailComplete l i t r) (fun i t r _ => ExprComplete i t r)
    complete_lit complete_group complete_mul_stop complete_mul_step complete_mul_make
    complete_expr_stop complete_expr_step complete_expr_make input tree rest h

/-- Primary bound.

    `Expr` [one tree]
    `List` [a finite sequence]
    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
private abbrev PB (i : List Token) (t : Expr) (r : List Token) :=
  ∀ f, 3 * i.length + 1 ≤ f → parsePrimary f i = .ok t r
/-- Multiplication-tail bound.

    `Expr` [one tree]
    `List` [a finite sequence]
    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
private abbrev MTB (l : Expr) (i : List Token) (t : Expr) (r : List Token) :=
  ∀ f, 3 * i.length + 1 ≤ f → parseMulTail f l i = .ok t r
/-- Multiplication bound.

    `Expr` [one tree]
    `List` [a finite sequence]
    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
private abbrev MB (i : List Token) (t : Expr) (r : List Token) :=
  ∀ f, 3 * i.length + 2 ≤ f → parseMul f i = .ok t r
/-- Addition-tail bound.

    `Expr` [one tree]
    `List` [a finite sequence]
    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
private abbrev ETB (l : Expr) (i : List Token) (t : Expr) (r : List Token) :=
  ∀ f, 3 * i.length + 1 ≤ f → parseExprTail f l i = .ok t r
/-- Expression bound.

    `Expr` [one tree]
    `List` [a finite sequence]
    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
private abbrev EB (i : List Token) (t : Expr) (r : List Token) :=
  ∀ f, 3 * i.length + 3 ≤ f → parseExpr f i = .ok t r

/-- Literal bound case. -/
private theorem bound_lit (n) (r) : PB (.nat n :: r) (.lit n) r := by
  intro f hf; cases f <;> simp_all [parsePrimary]
/-- Group bound case. -/
private theorem bound_group {i r t} (_ : ExprDerives i t (.rparen :: r))
    (ih : EB i t (.rparen :: r)) :
    PB (.lparen :: i) t r := by
  intro f hf
  cases f with
  | zero => omega
  | succ f => simp only [parsePrimary]; rw [ih f (by simp at hf; omega)]
/-- Multiplication-tail stop bound case.

    `Expr` [one arithmetic expression tree] -/
private theorem bound_mt_stop (l : Expr) {i} (h : NotTimes i) : MTB l i l i := by
  intro f hf; cases f with | zero => omega | succ f => cases h <;> rfl
/-- Multiplication-tail step bound case. -/
private theorem bound_mt_step {l q t i m r} (hp : PrimaryDerives i q m)
    (_ : MulTailDerives (.mul l q) m t r) (ip : PB i q m) (it : MTB (.mul l q) m t r) :
    MTB l (.times :: i) t r := by
  intro f hf; cases f with
  | zero => omega
  | succ f =>
    simp only [parseMulTail]; rw [ip f (by simp at hf; omega)]
    exact it f (by have hs := primary_suffix_length hp; simp at hf; omega)
/-- Multiplication tier bound case. -/
private theorem bound_m_make {i m r q t} (hp : PrimaryDerives i q m)
    (_ : MulTailDerives q m t r) (ip : PB i q m) (it : MTB q m t r) : MB i t r := by
  intro f hf; cases f with
  | zero => omega
  | succ f =>
    simp only [parseMul]; rw [ip f (by omega)]
    exact it f (by have hs := primary_suffix_length hp; omega)
/-- Addition-tail stop bound case.

    `Expr` [one arithmetic expression tree] -/
private theorem bound_et_stop (l : Expr) {i} (h : NotPlus i) : ETB l i l i := by
  intro f hf; cases f with | zero => omega | succ f => cases h <;> rfl
/-- Addition-tail step bound case. -/
private theorem bound_et_step {l q t i m r} (hm : MulDerives i q m)
    (_ : ExprTailDerives (.add l q) m t r) (im : MB i q m) (it : ETB (.add l q) m t r) :
    ETB l (.plus :: i) t r := by
  intro f hf; cases f with
  | zero => omega
  | succ f =>
    simp only [parseExprTail]; rw [im f (by simp at hf; omega)]
    exact it f (by have hs := mul_suffix_length hm; simp at hf; omega)
/-- Expression tier bound case. -/
private theorem bound_e_make {i m r q t} (hm : MulDerives i q m)
    (_ : ExprTailDerives q m t r) (im : MB i q m) (it : ETB q m t r) : EB i t r := by
  intro f hf; cases f with
  | zero => omega
  | succ f =>
    simp only [parseExpr]; rw [im f (by omega)]
    exact it f (by have hs := mul_suffix_length hm; omega)

/-- Primary derivations succeed above three times token length plus one.

    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
theorem primary_token_fuel {i t r} (h : PrimaryDerives i t r) :
    ∀ f, 3 * i.length + 1 ≤ f → parsePrimary f i = .ok t r :=
  @PrimaryDerives.rec (fun i t r _ => PB i t r) (fun l i t r _ => MTB l i t r)
    (fun i t r _ => MB i t r) (fun l i t r _ => ETB l i t r) (fun i t r _ => EB i t r)
    bound_lit bound_group (fun {left input} h => bound_mt_stop left (i := input) h) bound_mt_step
    bound_m_make (fun {left input} h => bound_et_stop left (i := input) h)
    bound_et_step bound_e_make i t r h
/-- Multiplication tails succeed above three times token length plus one.

    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
theorem mulTail_token_fuel {l i t r} (h : MulTailDerives l i t r) :
    ∀ f, 3 * i.length + 1 ≤ f → parseMulTail f l i = .ok t r :=
  @MulTailDerives.rec (fun i t r _ => PB i t r) (fun l i t r _ => MTB l i t r)
    (fun i t r _ => MB i t r) (fun l i t r _ => ETB l i t r) (fun i t r _ => EB i t r)
    bound_lit bound_group (fun {left input} h => bound_mt_stop left (i := input) h) bound_mt_step
    bound_m_make (fun {left input} h => bound_et_stop left (i := input) h)
    bound_et_step bound_e_make l i t r h
/-- Multiplications succeed above three times token length plus two.

    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
theorem mul_token_fuel {i t r} (h : MulDerives i t r) :
    ∀ f, 3 * i.length + 2 ≤ f → parseMul f i = .ok t r :=
  @MulDerives.rec (fun i t r _ => PB i t r) (fun l i t r _ => MTB l i t r)
    (fun i t r _ => MB i t r) (fun l i t r _ => ETB l i t r) (fun i t r _ => EB i t r)
    bound_lit bound_group (fun {left input} h => bound_mt_stop left (i := input) h) bound_mt_step
    bound_m_make (fun {left input} h => bound_et_stop left (i := input) h)
    bound_et_step bound_e_make i t r h
/-- Addition tails succeed above three times token length plus one.

    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
theorem exprTail_token_fuel {l i t r} (h : ExprTailDerives l i t r) :
    ∀ f, 3 * i.length + 1 ≤ f → parseExprTail f l i = .ok t r :=
  @ExprTailDerives.rec (fun i t r _ => PB i t r) (fun l i t r _ => MTB l i t r)
    (fun i t r _ => MB i t r) (fun l i t r _ => ETB l i t r) (fun i t r _ => EB i t r)
    bound_lit bound_group (fun {left input} h => bound_mt_stop left (i := input) h) bound_mt_step
    bound_m_make (fun {left input} h => bound_et_stop left (i := input) h)
    bound_et_step bound_e_make l i t r h
/-- Expressions succeed above three times token length plus three.

    `∀` [for every value]
    `≤` [is bounded by]
    `→` [if left then right] -/
theorem expr_token_fuel {i t r} (h : ExprDerives i t r) :
    ∀ f, 3 * i.length + 3 ≤ f → parseExpr f i = .ok t r :=
  @ExprDerives.rec (fun i t r _ => PB i t r) (fun l i t r _ => MTB l i t r)
    (fun i t r _ => MB i t r) (fun l i t r _ => ETB l i t r) (fun i t r _ => EB i t r)
    bound_lit bound_group (fun {left input} h => bound_mt_stop left (i := input) h) bound_mt_step
    bound_m_make (fun {left input} h => bound_et_stop left (i := input) h)
    bound_et_step bound_e_make i t r h

/-- Complete success derives a tree while leaving no token.

    parseComplete fuel input = .ok tree [] [the complete parser succeeds]
    ExprDerives input tree [] [the grammar derives the tree with empty suffix]

    `List` [a finite sequence of things, in order]
    `Expr` [one arithmetic expression tree]
    `→` [if the left holds so does the right] -/
theorem parseComplete_sound {fuel : Nat} {input : List Token} {tree : Expr}
    (h : parseComplete fuel input = .ok tree []) : ExprDerives input tree [] := by
  rw [parseComplete] at h
  cases hp : parseExpr fuel input with
  | «syntax» => simp [hp] at h
  | fuel => simp [hp] at h
  | ok parsed rest =>
      cases rest with
      | cons token rest => simp [hp] at h
      | nil =>
          simp [hp] at h
          subst parsed
          exact (parse_sound fuel).2.2.2.2 _ _ _ hp

/-- The multiplication-tail stop guard reaches the final suffix.

    `Expr` [an expression a law or spec row is allowed to state]
    `List` [a finite sequence of things, in order] -/
private theorem mulTail_suffix {left tree : Expr} {input rest : List Token}
    (h : MulTailDerives left input tree rest) : NotTimes rest := by
  refine MulTailDerives.rec (motive_1 := fun _ _ _ _ => True)
    (motive_2 := fun _ _ _ rest _ => NotTimes rest) (motive_3 := fun _ _ _ _ => True)
    (motive_4 := fun _ _ _ _ _ => True) (motive_5 := fun _ _ _ _ => True)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ h
  all_goals simp_all

/-- The addition-tail stop guard reaches the final suffix.

    `Expr` [an expression a law or spec row is allowed to state]
    `List` [a finite sequence of things, in order] -/
private theorem exprTail_suffix {left tree : Expr} {input rest : List Token}
    (h : ExprTailDerives left input tree rest) : NotPlus rest := by
  refine ExprTailDerives.rec (motive_1 := fun _ _ _ _ => True)
    (motive_2 := fun _ _ _ _ _ => True) (motive_3 := fun _ _ _ _ => True)
    (motive_4 := fun _ _ _ rest _ => NotPlus rest) (motive_5 := fun _ _ _ _ => True)
    ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ h
  all_goals simp_all

/-- A multiplication derivation leaves no multiplication continuation at its suffix.

    MulDerives input tree rest [a multiplication derivation and exact suffix]
    NotTimes rest [the suffix has no leading multiplication token]
    `Expr` [an expression a law or spec row is allowed to state]
    `List` [a finite sequence of things, in order]
    `→` [if the left holds so does the right] -/
theorem mul_suffix_not_times {input rest : List Token} {tree : Expr}
    (h : MulDerives input tree rest) : NotTimes rest := by
  cases h with
  | make _ tail => exact mulTail_suffix tail

/-- An addition derivation leaves no addition continuation at its suffix.

    ExprDerives input tree rest [an addition derivation and exact suffix]
    NotPlus rest [the suffix has no leading addition token]
    `Expr` [an expression a law or spec row is allowed to state]
    `List` [a finite sequence of things, in order]
    `→` [if the left holds so does the right] -/
theorem expr_suffix_not_plus {input rest : List Token} {tree : Expr}
    (h : ExprDerives input tree rest) : NotPlus rest := by
  cases h with
  | make _ tail => exact exprTail_suffix tail

/-- Multiplication binds inside addition on an inhabited input. -/
theorem parse_precedence :
    parseComplete 20 [.nat 1, .plus, .nat 2, .times, .nat 3] =
      .ok (.add (.lit 1) (.mul (.lit 2) (.lit 3))) [] := rfl

/-- Grouped additions form both children of the outer multiplication. -/
theorem parse_nested :
    parseComplete 30 [.lparen, .nat 1, .plus, .nat 2, .rparen, .times,
      .lparen, .nat 3, .plus, .nat 4, .rparen] =
      .ok (.mul (.add (.lit 1) (.lit 2)) (.add (.lit 3) (.lit 4))) [] := rfl

/-- Operator chains fold left at both precedence tiers. -/
theorem parse_left_chains :
    parseComplete 30 [.nat 1, .plus, .nat 2, .plus, .nat 3, .times, .nat 4, .times, .nat 5] =
      .ok (.add (.add (.lit 1) (.lit 2)) (.mul (.mul (.lit 3) (.lit 4)) (.lit 5))) [] := rfl

/-- A missing closing parenthesis is syntax refusal. -/
theorem parse_missing_close : parseComplete 10 [.lparen, .nat 7] = .syntax := rfl

/-- Zero fuel is reported separately from syntax refusal.

    `List` [a finite sequence of things, in order] -/
theorem parse_zero_fuel (input : List Token) : parseComplete 0 input = .fuel := rfl

end Arithmetic
end Parser
end Apeiron
