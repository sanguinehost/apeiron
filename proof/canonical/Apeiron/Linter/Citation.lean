import Apeiron.Linter.Citations

/-!
# The citation linter

Every `@cite` key must resolve in `Apeiron.Linter.Citations`. A key that names nothing fails
the build, so a citation cannot quietly point at an entry that was never written.

The check runs on module docstrings and on declaration docstrings alike, so a source can be
attached to the one theorem it supports rather than only to the file.

What this catches is a key with no entry. What it cannot catch is an entry crediting the wrong
person, which is a reading question and stays with whoever writes the entry.
-/

open Lean Elab Linter

namespace Apeiron.Linter

/-- Every `@cite` key must resolve in the citation registry. -/
register_option linter.apeiron.citation : Bool := {
  defValue := true
  descr := "every @cite key resolves in Apeiron.Linter.Citations"
}

/-- The keys cited in a docstring: everything following an `@cite` on its own line. -/
def citedKeys (doc : String) : List String :=
  doc.splitOn "\n" |>.filterMap fun line =>
    let l := line.trimAscii.toString
    if l.startsWith "@cite " then
      let rest := (l.drop 6).trimAscii.toString
      if rest.isEmpty then none else some ((rest.takeWhile (fun c => c != ' ')).toString)
    else none

/-- Every docstring in a command: the module docstring, and each declaration's. -/
partial def docStrings : Syntax → Array Syntax
  | s@(.node _ kind args) =>
    let here :=
      if kind == ``Parser.Command.moduleDoc || kind == ``Parser.Command.docComment then #[s]
      else #[]
    here ++ args.flatMap docStrings
  | _ => #[]

@[inherit_doc linter.apeiron.citation]
def citationLinter : Linter where
  run := withSetOptionIn fun stx => do
    unless getLinterValue linter.apeiron.citation (← getLinterOptions) do return
    if (← get).messages.hasErrors then return
    for d in docStrings stx do
      let some text := d.reprint | continue
      for k in citedKeys text do
        unless knownSource k do
          Linter.logLint linter.apeiron.citation d
            m!"`@cite {k}` names no entry in `Apeiron.Linter.Citations`. Add the source there, \
               or fix the key."

initialize addLinter citationLinter

end Apeiron.Linter
