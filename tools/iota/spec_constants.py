"""Frozen requirements and fixtures for the Iota specification checker."""

import re


OBLIGATIONS = tuple(f"O{i:02d}" for i in range(1, 19))
SECTIONS = (
    "Required obligations",
    "Selected source and identity",
    "Declaration closure",
    "Grammar and source binding",
    "Source arena and parsed records",
    "State and result arenas",
    "Memo key and memo value",
    "Addressing faces and Sigma",
    "Memo storage and fold",
    "Pointwise invariant, success and refusal",
    "Canonical algebraic link",
    "Word and expression semantics",
    "Local state, calls, and control",
    "Memory and extents",
    "Checked execution outcomes and admission",
    "Required theorem shapes",
    "Termination and source resources",
    "Retired route",
    "Not covered",
    "Checker",
)
STAGES = tuple(f"W{i}" for i in range(21, 31))

SOURCE_BINDING_SPEC = (
    "`ParsesClosure bundle entryProgram` is indexed only by the table-entry program",
    "Its exact constant-name list has eight entries and excludes",
    "that program has nine selected constants and includes",
    "inductive ClosedBinding (bundle : SourceBundle) (value : CertifiedDecoded) : Prop where",
    "(entryBinding : ParsesClosure bundle entryDecoded.val.program)",
    "(tableImports : importPaths entryBytes = [cellPath])",
    "(cellImports : importPaths cellBytes = [stdImportPath])",
    "(coordDerivation : ImportedCoordDerivation stdBytes stdTokens stdConsumed coord)",
    "(composed : value = Decode.appendImportedCoord entryDecoded coord)",
    "theorem source_binding_of_closed_binding {value : Decode.CertifiedDecoded}",
    "(binding : ClosedBinding entryBundle value)",
    "theorem source_binding :\n  ∃ value : Decode.CertifiedDecoded,",
    "ClosedBinding entryBundle value",
    "MUST be removed, not renamed",
    "Whole-source `#guard` checks remain compiled regression checks",
    "The re-rooted memo-store contract is the active refinement",
)

SOURCE_BINDING_PLAN = (
    "The source headline remains open until the unconditional theorem",
    "ClosedBinding selectedBundle value",
    "closed program has all nine selected",
    "contradictory and MUST NOT remain under any",
    "The corrected helper remains only as `source_binding_of_closed_binding`",
    "Replacing the unconditional target with the conditional helper MUST fail",
    "Acceptance requires the unconditional `Iota.SourceAudit.source_binding` theorem",
    "The active refinement is the memo store",
)

# The re-rooted contract, as specification and plan text. `--check` requires the
# documents to state each element; a removed or weakened element is rejected.
REROOTED_SPEC = (
    "`SourceRef` is a file identity with a checked byte interval and a checked token interval",
    "No copied text, filename-only identity, unchecked offset or prefix parse.",
    "returns a fresh handle or an explicit arena-capacity refusal",
    "complete computation identity",
    "A digest or coordinate alone is not key equality.",
    "bounded result handle",
    "value without evidence is not a memo hit",
    "executing signature carries `Sigma`",
    "elliptic exact reversible packing",
    "parabolic carried complete key",
    "hyperbolic scramble/discard",
    "One home shard; no fallback probing.",
    "256 shards",
    "full-key comparison",
    "no growth, fallback container, silent eviction or unrelated-key replacement",
    "`Apeiron.Table.HMap MemoKey MemoValue`",
    "ordered fold of checked transitions",
    "reordering, skipping or synthesizing a transition is forbidden",
    "the non-storage refusals are distinct",
    "`Iota.Refinement.wordDenote`",
    "`Iota.Refinement.cellDenote`",
    "`Iota.Refinement.cellDenote_mul`",
    "`Iota.Refinement.cellNormForm_denote`",
    "`word_mul_denote` and any `ZMod` declaration MUST NOT enter the empty-cone chain",
)

REROOTED_PLAN = (
    "active refinement is the memo store",
    "`Apeiron.Table.HMap MemoKey MemoValue`",
    "complete computation identity",
    "State and result arenas",
    "the `tools/iota/refine.py` exact-declaration probe",
    "red until the theorem exists",
    "baseline green, mutation red, restored green",
)

# Tokens of the retired route. A document line that names one is rejected as
# active unless it also carries a retirement marker. `retired` and `rejected`
# are markers, so a retirement statement is admissible.
RETIRED_ROUTE_TOKENS = (
    "DecodeTrace.Run",  # retired trace-acceptance token
    "Derives",  # retired Layer/Derives token
    "Iota.Derivation",  # retired derivation module family
    "Iota/Derivation/",  # retired derivation directory
    "Iota.Generated.Trace",  # retired generated-trace module family
    "Iota/Generated/Trace/",  # retired generated-trace directory
    "named proof DAG",  # retired representation
    "proof DAG",  # retired representation
    "segment lemma",  # retired method
    "segment-lemma",  # retired method
    "Layer/Derives",  # retired tower
)
RETIREMENT_MARKERS = (
    "retired",
    "retire ",
    "rejected",
    "reject ",
    "superseded",
    "removed",
    "remove ",
    "no longer",
    "not the active",
    "historical",
    "sunset",
    "binned",
)

FORBIDDEN_SOURCE_BINDING_ACCEPTANCE = (
    "`source_binding` is a proved conditional headline",
    "The source stage is unblocked",
)

# The unconditional source-binding headline the specification freezes, as
# specification text. `--check` requires the document to state this shape; it does
# not require the working `proof/corpus/Iota/SourceAudit.lean` declaration.
# Declaration acceptance is the `refine.py` exact-declaration probe: stage `source`
# in `tools/iota/controls.json` binds `Iota.SourceAudit.source_binding` to the
# unconditional type, so substituting the conditional
# `source_binding_of_closed_binding` is rejected by that probe before the declared
# TABLE_PHI source mutation (step 1) runs.
SPEC_HEADLINE_SHAPE = (
    "theorem source_binding :",
    "∃ value : Decode.CertifiedDecoded",
    "Decode.decodeClosure (encodeBytes tableSource) (encodeBytes stdSource) = .ok value",
    "value.val.certificate.Valid",
    "ExpectedSelectedSyntax value.val.program = true",
    "ClosedBinding entryBundle value",
)

# Self-contained synthetic headline classifier. The full-name fixtures match the
# frozen `Iota.SourceAudit.source_binding` type; `--self-test` classifies these and
# never reads the repository artifact.
UNCONDITIONAL_HEADLINE_FIXTURE = (
    "theorem source_binding :\n"
    "  ∃ value : Iota.Decode.CertifiedDecoded,\n"
    "    Iota.Decode.decodeClosure "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.tableSource) "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = Except.ok value ∧\n"
    "    Iota.Decode.Certificate.Valid value.val.certificate ∧\n"
    "    Iota.SourceAudit.ExpectedSelectedSyntax value.val.program = true ∧\n"
    "    Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value := by\n"
    "  exact ⟨decoded, certificate, selected, closed⟩\n"
)
CONDITIONAL_HELPER_FIXTURE = (
    "theorem source_binding_of_closed_binding {value : Iota.Decode.CertifiedDecoded}\n"
    "    (binding : Iota.Source.ClosedBinding "
    "Iota.Generated.TableDerivation.entryBundle value)\n"
    "    (selectedSyntax : Iota.SourceAudit.ExpectedSelectedSyntax "
    "value.val.program = true) :\n"
    "    Iota.Decode.decodeClosure "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.tableSource) "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = Except.ok value ∧\n"
    "    Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value := by\n"
    "  exact ⟨binding, selectedSyntax⟩\n"
)
PREFIX_WITNESS_FIXTURE = (
    "theorem source_binding :\n"
    "  ∃ value : Iota.Decode.CertifiedDecoded,\n"
    "    Iota.Decode.decodeClosure "
    "(Iota.Source.encodeBytes (List.take 4096 "
    "Iota.Generated.TableDerivation.tableSource)) "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = Except.ok value ∧\n"
    "    Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value := by\n"
    "  exact witness\n"
)
EMPTY_WITNESS_FIXTURE = (
    "theorem source_binding :\n"
    "  ∃ value : Iota.Decode.CertifiedDecoded,\n"
    "    Iota.Decode.decodeClosure (Iota.Source.encodeBytes []) "
    "(Iota.Source.encodeBytes []) = Except.ok value ∧\n"
    "    Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value := by\n"
    "  exact witness\n"
)
FALLBACK_WITNESS_FIXTURE = (
    "theorem source_binding :\n"
    "  ∃ value : Iota.Decode.CertifiedDecoded,\n"
    "    Iota.Decode.decodeClosure "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.tableSource) "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = Except.ok value ∧\n"
    "    Iota.SourceAudit.ExpectedSelectedSyntax value.val.program = true ∧\n"
    "    value.val.program = Iota.Source.fallbackProgram ∧\n"
    "    Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value := by\n"
    "  exact witness\n"
)
GUARD_FIXTURE = (
    "#guard Iota.Generated.TableDerivation.decodeAccepts "
    "Iota.Generated.TableDerivation.tableSource\n"
)
UNCONDITIONAL_HEADLINE_REQUIRED = (
    "∃ value : Iota.Decode.CertifiedDecoded",
    "Iota.Decode.decodeClosure "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.tableSource) "
    "(Iota.Source.encodeBytes Iota.Generated.TableDerivation.stdSource) = Except.ok value",
    "Iota.Decode.Certificate.Valid value.val.certificate",
    "Iota.SourceAudit.ExpectedSelectedSyntax value.val.program = true",
    "Iota.Source.ClosedBinding Iota.Generated.TableDerivation.entryBundle value",
)
CONDITIONAL_HEADLINE_MARKERS = (
    "source_binding_of_closed_binding",
    "DecodeDerivation",
    "ParsesClosure",
    "→",
    "->",
)
PREFIX_HEADLINE_WORDS = ("take", "drop")
PREFIX_HEADLINE_SUBSTRINGS = ("prefix", "Prefix")
EMPTY_FALLBACK_HEADLINE_MARKERS = (
    "encodeBytes []",
    "fallback",
    "Fallback",
    "defaultProgram",
    "emptyProgram",
    "emptySource",
)

# A gate is a shell command line. Every command segment after splitting on the
# shell operators must begin with a known runner, a shell builtin, or a
# path-like executable token; a bare word such as `x` is not runnable.
RUNNERS = frozenset({"python3", "python", "lake", "cargo", "bash", "sh", "lean", "dune", "make"})
SHELL_BUILTINS = frozenset({"cd", "env", "exec", "time", "nice", "export", "set", "source", "."})
SCRIPT_SUFFIXES = (".py", ".sh", ".bash", ".pl", ".rb", ".js")
FILE_SUFFIXES = (".lean", ".md", ".py", ".sh", ".json", ".slang", ".toml", ".txt")
DOTTED_DECL = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)+")
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
# At least one termination statement and one resource-accounting statement must
# remain in the termination section body, so arbitrary prose cannot replace it.
TERMINATION_ELEMENTS = (
    "natural measures derived from decoded arguments",
    "source annotations alone are not premises",
    "Counts are not converted into one scalar budget",
)
# The only admissible non-stage prerequisite: the root stage's specification gate.
ROOT_PREREQUISITE = "this specification gate"

SOURCES = {
    "src/table/table.slang": "bf5d19e6eb5d127100e777dee054dcbebaebcd7c305fffc748f93e54b5525e1d",
    "src/kernel/std.slang": "e24b0e360f26794fab0fe1332c32455143d22d2059e871b8c012b4e8a89a6cae",
    "src/kernel/mem/cell.slang": "35e3e46c5b7820c67f9c92f2af07a16bf581964b90acf7c508c5b13570d9b3cc",
    "src/front/parser.slang": "e643bd85ca34b195888a0c64ec3662ce75c0c770d365b360285a9e0af2e4a882",
    "src/front/node.slang": "db096abda00e40ddf6114db7d8da8590b4e03e46390eb4fd38ce0477679b2d6c",
    "src/front/bridge.slang": "6d3aef63833ac4c2eaf99c71b044b7ae22346a7020c414ceb73ad03f5b4466c0",
    "src/emit/isa.slang": "da75ad47bbbbb4ddb8a22c2d5d2729910583f0e37fcec6a198aa7257dd5e79f9",
}
FUNCTIONS = (
    "tableSpread",
    "tableSlotKey",
    "tableSlotVal",
    "tableSlotSet",
    "tableSlotValSet",
    "tableInitGo",
    "tableInit",
    "tableCapacityOk",
    "tableFindSlotGo",
    "tableLookupSlotGo",
    "tableFind",
    "tableGet",
    "tableInsertWith",
)
CONSTANTS = (
    "TABLE_PRESENT",
    "TABLE_EMPTY",
    "TABLE_STRIDE",
    "MERGE_SET",
    "MERGE_MAP",
    "MERGE_ACC",
    "MERGE_GRAPH",
    "TABLE_PHI",
)
CANONICAL = {
    "HMap": "abbrev",
    "get": "def",
    "insertWith": "def",
    "build": "def",
    "get_insertWith_self": "theorem",
    "get_insertWith_ne": "theorem",
}
SPEC_PHRASES = (
    "64-bit wrapping multiplication",
    "do **not** call `cellGet` or `cellSet`",
    "**NO checked table refusal encoding**",
    "timeout is inconclusive",
    "never `return != -1`",
    "complete computation identity",
    "bounded result handle",
    "raw return word and the resulting state",
    "digest establishes byte equality only",
    "handwritten Lean algorithm",
    "complete brace-delimited bodies",
    "unsupported reachable construct MUST reject",
)
PLAN_PHRASES = (
    "FORBIDDEN: prove all Iota",
    "FORBIDDEN: implement a Rust evaluator",
)

OBLIGATION_HEADER = "| ID | Obligation | Required checked result |"
STAGE_HEADER = (
    "| Wave | Purpose | Owned files (maximum three) | Gate | "
    "Negative control | Opens / stop condition |"
)
OBLIGATION_COLUMNS = 3
STAGE_COLUMNS = 6
