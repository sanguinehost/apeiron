import Apeiron.Linter.Registry

/-!
# The notation linter

Every symbol in a declaration's statement must carry a bracketed meaning in that
declaration's docstring. This runs during elaboration, on the parsed syntax tree, so it sees
the real `declSig` (the binders and the type, without the proof) and the real docstring
attached to the declaration. There is no text parsing of `theorem` keywords and no guessing
where a statement ends.

Modelled on `Mathlib.Linter.Style.docStringLinter`, which reaches the docstring the same way:
walk the syntax for `declModifiers`, and take the `docComment` node from it.

Mathlib runs style checks in two layers, and the reason is worth stating: a syntax-tree
linter is precise but blind to anything the parser discards, so trailing whitespace and
line length stay in a text-based pass (`Mathlib/Tactic/Linter/TextBased.lean`). We copy that
split. Gloss coverage is a syntax question and lives here; line length and prose filler are
text questions and stay in `scripts/check-style.py`.
-/

open Lean Elab Linter

namespace Apeiron.Linter

/-- Every symbol in a declaration's statement must carry a bracketed meaning in that
declaration's docstring. -/
register_option linter.apeiron.notation : Bool := {
  defValue := true
  descr := "every symbol in a statement carries a bracketed meaning in the docstring"
}

/-- Does `needle` occur in `haystack` at a word boundary on both sides?

Identifier characters and `.` count as boundary-breaking, so `mul` does not match inside
`Trichotomy.mul`, `normForm_mul`, or `mula`. -/
def containsToken (haystack needle : String) : Bool :=
  let isIdent (c : Char) : Bool := c.isAlphanum || c == '_' || c == '\'' || c == '.'
  let hs := haystack.toList
  let nd := needle.toList
  let n := nd.length
  let rec go (pre : Option Char) (rest : List Char) : Bool :=
    match rest with
    | [] => false
    | c :: cs =>
      let hit :=
        rest.take n == nd &&
        (match pre with | none => true | some p => !isIdent p) &&
        (match rest.drop n with
         | [] => true
         | d :: _ => !(d.isAlphanum || d == '_' || d == '\''))
      if hit then true else go (some c) cs
  go none hs

/-- Is `e` present in `text`, by its own matching rule? -/
def Entry.presentIn (e : Entry) (text : String) : Bool :=
  match e.kind with
  | MatchKind.token => containsToken text e.symbol
  | MatchKind.chars => ((text.splitOn e.symbol).length > 1)

/-- The left-hand side of every gloss line in a docstring.

A gloss line is `symbols   [meaning]`. We split on the LAST `[` because notation such as
`≃ₗ[ℝ]` contains brackets of its own, and the gloss is always the trailing bracket group.
A line without both brackets is prose, not a gloss, and contributes nothing. -/
def glossedText (doc : String) : String :=
  let lines := doc.splitOn "\n"
  let keep := lines.filterMap fun line =>
    if (line.splitOn "[").length > 1 && (line.splitOn "]").length > 1 then
      let parts := line.splitOn "["
      some (String.intercalate "[" (parts.dropLast))
    else none
  String.intercalate "\n" keep

/-- Pull the `declModifiers` nodes out of a command's syntax. Mathlib's `getDeclModifiers`. -/
partial def declModifierNodes : Syntax → Array Syntax
  | s@(.node _ kind args) =>
    (if kind == ``Parser.Command.declModifiers then #[s] else #[]) ++
      args.flatMap declModifierNodes
  | _ => #[]

/-- Pull the `declSig` nodes out of a command's syntax: the binders and the type, without
the proof. For `theorem f (x : T) : S := proof` this is `(x : T) : S`. -/
partial def declSigNodes : Syntax → Array Syntax
  | s@(.node _ kind args) =>
    (if kind == ``Parser.Command.declSig || kind == ``Parser.Command.optDeclSig then #[s]
     else #[]) ++ args.flatMap declSigNodes
  | _ => #[]

/-- The name a declaration is being given, for the error message. -/
partial def declIdName : Syntax → Option String
  | .node _ kind args =>
    if kind == ``Parser.Command.declId then
      args[0]? >>= fun i => i.getId.toString
    else args.findSome? declIdName
  | _ => none

@[inherit_doc linter.apeiron.notation]
def notationLinter : Linter where
  run := withSetOptionIn fun stx => do
    unless getLinterValue linter.apeiron.notation (← getLinterOptions) do return
    if (← get).messages.hasErrors then return
    let sigs := declSigNodes stx
    if sigs.isEmpty then return
    let mods := declModifierNodes stx
    let name := (declIdName stx).getD "this declaration"
    -- the statement as written, binders and type, without the proof
    let some sigText := (sigs[0]!).reprint | return
    -- the docstring, if the declaration has one
    let docText : String :=
      match mods[0]? with
      | none => ""
      | some m =>
        match (m.getArgs)[0]? with
        | none => ""
        | some d => (d.reprint).getD ""
    if docText.trimAscii.isEmpty then
      Linter.logLint linter.apeiron.notation sigs[0]!
        m!"`{name}` has no docstring, so nothing in its statement is glossed"
      return
    let glossed := glossedText docText
    for e in registry do
      if e.presentIn sigText && !e.presentIn glossed then
        Linter.logLint linter.apeiron.notation sigs[0]!
          m!"`{e.symbol}` appears in the statement of `{name}` but carries no [meaning] \
             in its docstring. Add a gloss line: `{e.symbol}` [{e.tag}]"

initialize addLinter notationLinter

end Apeiron.Linter
