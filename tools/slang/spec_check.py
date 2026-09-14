#!/usr/bin/env python3
"""Fail-closed consistency checker for the selected Slang semantics specification."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
SONG = ROOT.parent / "song"
CORPUS = ROOT.parent / "sanguine-apeiron" / "proof"
SPEC_PATH = ROOT / "docs/spec/slang-source-semantics.md"
PLAN_PATH = ROOT / "docs/plan/slang-source-refinement.md"
CANONICAL_MODULE = "Apeiron/Table/Basic.lean"
SONG_REV = "aeaf3baac434b2e3c01154d275bea65fdadc69bf"
CORPUS_REV = "8886641278c45dc0b2fe4ca5cc3a3d80ae19c46f"

OBLIGATIONS = tuple(f"O{i:02d}" for i in range(1, 19))
SECTIONS = (
    "Required obligations",
    "Selected source and identity",
    "Declaration closure",
    "Grammar and source binding",
    "Word and expression semantics",
    "Local state, calls, and control",
    "Memory and extents",
    "Checked execution outcomes and admission",
    "Representation and observation",
    "Required theorem shapes",
    "Termination and source resources",
    "Not covered",
    "Checker",
)
STAGES = tuple(f"W{i}" for i in range(21, 31))

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
PHRASES = (
    "64-bit wrapping multiplication",
    "do **not** call `cellGet` or `cellSet`",
    "**NO checked table refusal encoding**",
    "timeout is inconclusive",
    "never `return != -1`",
    "RepSong(memory, base, cap, default, O, h)",
    "observeRead",
    "raw return and resulting state",
    "digest establishes byte equality only",
    "handwritten Lean algorithm",
    "complete brace-delimited bodies",
    "unsupported reachable construct MUST reject",
    "FORBIDDEN: prove all Slang",
    "FORBIDDEN: implement a Rust evaluator",
)

OBLIGATION_HEADER = "| ID | Obligation | Required checked result |"
STAGE_HEADER = (
    "| Wave | Purpose | Owned files (maximum three) | Gate | "
    "Negative control | Opens / stop condition |"
)
OBLIGATION_COLUMNS = 3
STAGE_COLUMNS = 6


class CheckError(Exception):
    pass


@dataclass(frozen=True)
class Inputs:
    spec: str
    plan: str
    sources: dict[str, bytes]
    corpus: str
    spec_bytes: bytes = b""
    plan_bytes: bytes = b""
    spec_sha256: str = ""
    plan_sha256: str = ""


def fail(message: str) -> None:
    raise CheckError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_source(name: str, data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeError as exc:
        fail(f"source is not utf-8: song/{name}: {exc}")


def load() -> Inputs:
    try:
        raw = {name: (SONG / name).read_bytes() for name in SOURCES}
        for name, data in raw.items():
            decode_source(name, data)
        # The spec and plan are bound to their raw bytes, not to newline-normalized
        # text: a CRLF checkout must not decode into the same accepted document.
        spec_bytes = SPEC_PATH.read_bytes()
        plan_bytes = PLAN_PATH.read_bytes()
        return Inputs(
            spec=spec_bytes.decode("utf-8"),
            plan=plan_bytes.decode("utf-8"),
            sources=raw,
            corpus=(CORPUS / CANONICAL_MODULE).read_text(encoding="utf-8"),
            spec_bytes=spec_bytes,
            plan_bytes=plan_bytes,
            spec_sha256=sha256(spec_bytes),
            plan_sha256=sha256(plan_bytes),
        )
    except (OSError, UnicodeError) as exc:
        fail(f"unresolved file: {exc}")


def git_revision(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"repository identity unresolved: {path}: {exc}")
    return result.stdout.strip()


def require_once(text: str, needle: str, label: str) -> None:
    count = text.count(needle)
    if count != 1:
        fail(f"{label}: expected once, found {count}")


def row_cells(row: str, columns: int, label: str) -> list[str]:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    if len(cells) != columns:
        fail(f"{label}: expected {columns} columns, found {len(cells)}")
    return cells


def section_body(text: str, section: str) -> str:
    """Text between the `## section` heading and the next level-2 heading."""
    match = re.search(rf"^## {re.escape(section)}$", text, re.MULTILINE)
    if match is None:
        return ""
    rest = text[match.end():]
    following = re.search(r"^## ", rest, re.MULTILINE)
    if following is not None:
        rest = rest[: following.start()]
    return rest


def has_section_content(body: str) -> bool:
    """True when the body has a line that is neither blank, a heading, nor a bare comment."""
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.fullmatch(r"<!--.*-->", stripped):
            continue
        return True
    return False


def meaningful_text(text: str) -> bool:
    """True when the text carries a substantive word.

    Whitespace, HTML comments, punctuation and backticks alone are not content;
    the text must retain at least one alphanumeric character.
    """
    cleaned = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    return re.search(r"[^\W_]", cleaned) is not None


def strip_lean_comments(text: str) -> str:
    """Remove nested Lean block comments and line comments, preserving newlines.

    The result is searched for canonical declarations so that a declaration text
    that exists only inside a comment is not accepted as code.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    depth = 0
    while i < n:
        ch = text[i]
        if depth == 0 and ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                inner = text[i]
                out.append(inner)
                if inner == "\\" and i + 1 < n:
                    out.append(text[i + 1])
                    i += 2
                    continue
                i += 1
                if inner == '"':
                    break
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "-":
            depth += 1
            out.append("  ")
            i += 2
            continue
        if depth > 0 and ch == "-" and i + 1 < n and text[i + 1] == "/":
            depth -= 1
            out.append("  ")
            i += 2
            continue
        if depth > 0:
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            while i < n and text[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def detect_cycle(graph: dict[str, set[str]], label: str) -> None:
    color: dict[str, int] = {node: 0 for node in graph}
    stack: list[str] = []

    def visit(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for dep in sorted(graph[node]):
            if dep not in color:
                continue
            if color[dep] == 1:
                cycle = stack[stack.index(dep):] + [dep]
                fail(f"{label} cycle: " + " -> ".join(cycle))
            if color[dep] == 0:
                visit(dep)
        stack.pop()
        color[node] = 2

    for node in graph:
        if color[node] == 0:
            visit(node)


def check_markdown(inputs: Inputs) -> None:
    require_once(inputs.spec, OBLIGATION_HEADER, "obligations header")
    require_once(inputs.plan, STAGE_HEADER, "stage table header")
    for section in SECTIONS:
        count = len(re.findall(rf"^## {re.escape(section)}$", inputs.spec, re.MULTILINE))
        if count != 1:
            fail(f"section {section}: expected once, found {count}")
        if not has_section_content(section_body(inputs.spec, section)):
            fail(f"section {section}: body is empty")
    termination = section_body(inputs.spec, "Termination and source resources")
    for element in TERMINATION_ELEMENTS:
        if element not in termination:
            fail(f"termination section missing required element: {element}")
    for oid in OBLIGATIONS:
        rows = [line for line in inputs.spec.splitlines() if line.startswith(f"| {oid} |")]
        if len(rows) != 1:
            fail(f"obligation {oid}: expected one table row")
        cells = row_cells(rows[0], OBLIGATION_COLUMNS, f"obligation {oid}")
        if cells[0] != oid:
            fail(f"obligation {oid}: id column mismatch")
        if not meaningful_text(cells[1]):
            fail(f"obligation {oid}: empty obligation text")
        if not meaningful_text(cells[2]):
            fail(f"obligation {oid}: empty required checked result")
    for phrase in PHRASES[:12]:
        if phrase not in inputs.spec:
            fail(f"required statement missing: {phrase}")
    for phrase in PHRASES[12:]:
        if phrase not in inputs.plan:
            fail(f"required plan boundary missing: {phrase}")
    forbidden_budget = "fu" + "el"
    if re.search(rf"\b{forbidden_budget}\b", inputs.spec + inputs.plan, re.IGNORECASE):
        fail("generic scalar execution budget is forbidden in this specification route")


def check_sources(inputs: Inputs) -> None:
    if git_revision(SONG) != SONG_REV or SONG_REV not in inputs.spec:
        fail("Song repository identity mismatch")
    if git_revision(CORPUS.parent) != CORPUS_REV or CORPUS_REV not in inputs.spec:
        fail("canonical corpus identity mismatch")
    for name in SOURCES:
        if name not in inputs.sources:
            fail(f"source missing: song/{name}")

    table = decode_source("src/table/table.slang", inputs.sources["src/table/table.slang"])
    std = decode_source("src/kernel/std.slang", inputs.sources["src/kernel/std.slang"])
    cell = decode_source("src/kernel/mem/cell.slang", inputs.sources["src/kernel/mem/cell.slang"])
    if 'use "src/kernel/mem/cell.slang"' not in table:
        fail("table import edge unresolved")
    if 'use "src/kernel/std.slang"' not in cell:
        fail("cell import edge unresolved")
    if not re.search(r"^COORD_CEILING\s*=\s*8796093022208\s*$", std, re.MULTILINE):
        fail("COORD_CEILING definition unresolved")
    for name in CONSTANTS:
        if not re.search(rf"^{re.escape(name)}\s*=", table, re.MULTILINE):
            fail(f"selected constant unresolved: {name}")
        if f"`{name}`" not in inputs.spec:
            fail(f"selected constant absent from spec: {name}")
    for name in FUNCTIONS:
        if not re.search(rf"^{re.escape(name)}\([^\n]*\)\s*\{{", table, re.MULTILINE):
            fail(f"selected function unresolved: {name}")
        if f"`{name}`" not in inputs.spec:
            fail(f"selected function absent from spec: {name}")
    raw = ("return memory[" in table and "memory[base + TABLE_STRIDE * slot] = key;" in table)
    if not raw:
        fail("raw table memory access unresolved")

    loaded = set(inputs.sources)
    import_graph: dict[str, set[str]] = {}
    for name, data in inputs.sources.items():
        text = decode_source(name, data)
        imports = re.findall(r'^\s*use\s+"([^"]+)"', text, re.MULTILINE)
        for imported in imports:
            if not (SONG / imported).is_file():
                fail(f"unresolved import: song/{name} -> {imported}")
        import_graph[name] = {dep for dep in imports if dep in loaded}
    detect_cycle(import_graph, "source import")

    for name, expected in SOURCES.items():
        actual = sha256(inputs.sources[name])
        if actual != expected:
            fail(f"source digest mismatch: song/{name}: {actual}")
        row = f"`song/{name}` | `{expected}`"
        if row not in inputs.spec:
            fail(f"source identity row missing: song/{name}")


def canonical_references(text: str) -> set[str]:
    """Every `Apeiron.Table.<name>` reference, including the `{a,b}` shorthand."""
    names = set(re.findall(r"Apeiron\.Table\.([A-Za-z_][A-Za-z0-9_']*)", text))
    for group in re.findall(r"Apeiron\.Table\.\{([^}]*)\}", text):
        for part in group.split(","):
            part = part.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_']*", part):
                names.add(part)
    return names


def check_canonical(inputs: Inputs) -> None:
    module_path = CORPUS / CANONICAL_MODULE
    if not module_path.is_file():
        fail(f"canonical module file missing: {module_path}")
    require_once(inputs.spec, f"`{CANONICAL_MODULE}`", "canonical module reference")
    code = strip_lean_comments(inputs.corpus)
    if "namespace Apeiron\nnamespace Table" not in code:
        fail("canonical namespace unresolved")
    for name, kind in CANONICAL.items():
        pattern = rf"^{kind}\s+{re.escape(name)}\b"
        if not re.search(pattern, code, re.MULTILINE):
            fail(f"canonical target unresolved: Apeiron.Table.{name} ({kind})")
        require_once(inputs.spec, f"`Apeiron.Table.{name}`", f"canonical target {name}")
    referenced = canonical_references(inputs.spec + "\n" + inputs.plan)
    unresolved = sorted(referenced - set(CANONICAL))
    if unresolved:
        fail("unresolved canonical reference(s): " + ", ".join(unresolved))


def check_document_bytes(inputs: Inputs) -> None:
    """Bind the spec and plan to their raw bytes; reject normalized line endings."""
    for label, raw, text, digest in (
        ("spec", inputs.spec_bytes, inputs.spec, inputs.spec_sha256),
        ("plan", inputs.plan_bytes, inputs.plan, inputs.plan_sha256),
    ):
        if sha256(raw) != digest:
            fail(f"{label} digest drift from raw bytes")
        if raw.decode("utf-8") != text:
            fail(f"{label} text is not the raw-byte decoding")
        if b"\r" in raw:
            fail(f"{label} contains CR bytes; line endings are not normalized")


def relative_links(text: str) -> list[str]:
    """Repository-relative link targets, fragment stripped and external links skipped."""
    targets: list[str] = []
    for match in MARKDOWN_LINK.finditer(text):
        target = match.group(1).strip()
        if not target or target.startswith("#") or "://" in target:
            continue
        target = target.split("#", 1)[0]
        if target:
            targets.append(target)
    return targets


def planned_files(plan: str) -> set[str]:
    """Backticked paths declared in the plan's owned-files column."""
    planned: set[str] = set()
    for stage in STAGES:
        rows = [line for line in plan.splitlines() if line.startswith(f"| {stage} |")]
        if len(rows) != 1:
            continue
        cells = [cell.strip() for cell in rows[0].strip().strip("|").split("|")]
        if len(cells) != STAGE_COLUMNS:
            continue
        planned.update(re.findall(r"`([^`]+)`", cells[2]))
    return planned


def check_references(inputs: Inputs) -> None:
    """Resolve every relative doc link and every backticked module/declaration reference."""
    planned = planned_files(inputs.plan)
    for label, text, base in (
        ("spec", inputs.spec, SPEC_PATH.parent),
        ("plan", inputs.plan, PLAN_PATH.parent),
    ):
        for target in relative_links(text):
            if not (base / target).is_file():
                fail(f"{label} relative link unresolved: {target}")
        for token in re.findall(r"`([^`]+)`", text):
            if any(ch.isspace() for ch in token):
                continue
            if token.endswith(FILE_SUFFIXES):
                if token in planned or token == CANONICAL_MODULE:
                    continue
                if any((root / token).is_file() for root in (ROOT, CORPUS, SONG, ROOT.parent, base)):
                    continue
                fail(f"{label} module path unresolved: {token}")
            elif DOTTED_DECL.fullmatch(token):
                prefix, _, name = token.rpartition(".")
                if prefix != "Apeiron.Table" or name not in CANONICAL:
                    fail(f"{label} declaration reference unresolved: {token}")


def stage_row(plan: str, stage: str) -> str:
    rows = [line for line in plan.splitlines() if line.startswith(f"| {stage} |")]
    if len(rows) != 1:
        fail(f"stage {stage}: expected one task row")
    return rows[0]


def check_owned_files(cell: str, stage: str) -> None:
    entries = [entry for entry in re.split(r"[;,\s]+", cell) if entry]
    tokens: list[str] = []
    for entry in entries:
        if not re.fullmatch(r"`[^`]+`", entry):
            fail(f"stage {stage}: owned files must be backtick-quoted paths: {cell}")
        tokens.append(entry[1:-1])
    if not 1 <= len(tokens) <= 3:
        fail(f"stage {stage}: owned-file count is {len(tokens)}, expected 1..3")
    for token in tokens:
        if not token.strip():
            fail(f"stage {stage}: empty owned-file entry")


def gate_command_segments(gate: str) -> list[str]:
    """First word of every command segment in a shell command line."""
    first_words: list[str] = []
    for segment in re.split(r"&&|\|\||;|\|", gate):
        words = segment.strip().split()
        index = 0
        while index < len(words) and ENV_ASSIGNMENT.fullmatch(words[index]):
            index += 1
        if index < len(words):
            first_words.append(words[index])
    return first_words


def is_runnable_command(token: str) -> bool:
    """True for a known runner, a shell builtin, or a path to an executable script."""
    if token in RUNNERS or token in SHELL_BUILTINS:
        return True
    if token.endswith(SCRIPT_SUFFIXES):
        return True
    if token.startswith(("./", "../", "/")) or "/" in token:
        for candidate in (Path(token), ROOT / token, ROOT.parent / token):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return True
        return False
    return False


def check_gate(cell: str, stage: str) -> None:
    match = re.fullmatch(r"`([^`]+)`", cell)
    if match is None:
        fail(f"stage {stage}: gate missing")
    gate = match.group(1).strip()
    if not gate:
        fail(f"stage {stage}: gate is empty")
    segments = gate_command_segments(gate)
    if not segments:
        fail(f"stage {stage}: gate has no command")
    for token in segments:
        if not is_runnable_command(token):
            fail(f"stage {stage}: gate segment does not begin with a runnable command: {token}")


def check_negative_control(cell: str, stage: str) -> None:
    if "MUST" not in cell:
        fail(f"stage {stage}: negative control missing")
    if "reject" not in cell and "fail" not in cell:
        fail(f"stage {stage}: negative control has no rejection verb")


def prerequisite_dependencies(targets: str, stage: str) -> list[str]:
    """Stage ids named by the explicit `Requires` target list.

    Both `W<n>` and bare `<n>` stage numbers count, so a cycle cannot be hidden
    by writing the stage numbers without the `W` prefix. Ranges expand.
    """
    deps: list[str] = []

    def expand(match: re.Match[str]) -> str:
        low, high = int(match.group(1)), int(match.group(2))
        if high < low:
            fail(f"stage {stage}: malformed prerequisite range {match.group(0)}")
        deps.extend(f"W{number}" for number in range(low, high + 1))
        return " "

    remaining = re.sub(r"W?(\d+)\s*-\s*W?(\d+)", expand, targets)
    for match in re.finditer(r"W?(\d+)", remaining):
        deps.append(f"W{int(match.group(1))}")
    for dep in deps:
        if dep not in STAGES:
            fail(f"stage {stage}: unresolved prerequisite {dep}")
        if dep == stage:
            fail(f"stage {stage}: self prerequisite")
    if len(set(deps)) != len(deps):
        fail(f"stage {stage}: duplicate prerequisite")
    return deps


def check_prerequisite(cell: str, stage: str) -> list[str]:
    parts = re.split(r"\bStop\b", cell, maxsplit=1)
    head = parts[0]
    if "Requires" not in head:
        fail(f"stage {stage}: prerequisite missing")
    stop = parts[1] if len(parts) == 2 else ""
    if not meaningful_text(stop.strip(" .;:-")):
        fail(f"stage {stage}: stop condition missing")
    clause = head.split("Requires", 1)[1]
    targets = clause.split(".", 1)[0].strip().strip(" .;:-")
    deps = prerequisite_dependencies(targets, stage)
    if not deps and not (stage == STAGES[0] and targets == ROOT_PREREQUISITE):
        fail(f"stage {stage}: prerequisite does not name a stage")
    return deps


def check_plan(inputs: Inputs) -> None:
    dependencies: dict[str, list[str]] = {}
    for stage in STAGES:
        row = stage_row(inputs.plan, stage)
        cells = row_cells(row, STAGE_COLUMNS, f"stage {stage}")
        if not meaningful_text(cells[1]):
            fail(f"stage {stage}: purpose is empty")
        check_owned_files(cells[2], stage)
        check_gate(cells[3], stage)
        check_negative_control(cells[4], stage)
        dependencies[stage] = check_prerequisite(cells[5], stage)
    detect_cycle({stage: set(deps) for stage, deps in dependencies.items()}, "stage prerequisite")
    if inputs.plan.find("W21") > inputs.plan.find("W30"):
        fail("stage integration order reversed")


def validate(inputs: Inputs) -> None:
    check_document_bytes(inputs)
    check_markdown(inputs)
    check_sources(inputs)
    check_canonical(inputs)
    check_plan(inputs)
    check_references(inputs)


Mutation = tuple[str, Callable[[Inputs], Inputs]]


def spec_text(text: str) -> Callable[[Inputs], Inputs]:
    def mutate(i: Inputs) -> Inputs:
        raw = text.encode("utf-8")
        return replace(i, spec=text, spec_bytes=raw, spec_sha256=sha256(raw))

    return mutate


def plan_text(text: str) -> Callable[[Inputs], Inputs]:
    def mutate(i: Inputs) -> Inputs:
        raw = text.encode("utf-8")
        return replace(i, plan=text, plan_bytes=raw, plan_sha256=sha256(raw))

    return mutate


def sub_spec(old: str, new: str = "") -> Callable[[Inputs], Inputs]:
    return lambda i: spec_text(i.spec.replace(old, new, 1))(i)


def sub_plan(old: str, new: str = "") -> Callable[[Inputs], Inputs]:
    return lambda i: plan_text(i.plan.replace(old, new, 1))(i)


def sub_spec_all(old: str, new: str = "") -> Callable[[Inputs], Inputs]:
    return lambda i: spec_text(i.spec.replace(old, new))(i)


def document_crlf(which: str) -> Callable[[Inputs], Inputs]:
    """Re-encode one document with CRLF bytes without normalizing them."""

    def mutate(i: Inputs) -> Inputs:
        raw = i.spec_bytes if which == "spec" else i.plan_bytes
        raw = raw.replace(b"\n", b"\r\n")
        text = raw.decode("utf-8")
        if which == "spec":
            return replace(i, spec=text, spec_bytes=raw, spec_sha256=sha256(raw))
        return replace(i, plan=text, plan_bytes=raw, plan_sha256=sha256(raw))

    return mutate


def source_sub(path: str, old: str, new: str) -> Callable[[Inputs], Inputs]:
    def mutate(i: Inputs) -> Inputs:
        changed = dict(i.sources)
        changed[path] = changed[path].replace(old.encode("utf-8"), new.encode("utf-8"), 1)
        return replace(i, sources=changed)

    return mutate


def source_crlf(path: str) -> Callable[[Inputs], Inputs]:
    def mutate(i: Inputs) -> Inputs:
        changed = dict(i.sources)
        changed[path] = changed[path].replace(b"\n", b"\r\n")
        return replace(i, sources=changed)

    return mutate


def mutate_row_cell(row: str, index: int, value: str) -> str:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    cells[index] = value
    return "| " + " | ".join(cells) + " |"


def controls(inputs: Inputs) -> list[Mutation]:
    cases: list[Mutation] = []
    for oid in OBLIGATIONS:
        row = next(line for line in inputs.spec.splitlines() if line.startswith(f"| {oid} |"))
        cases.append((f"remove obligation {oid}", sub_spec(row + "\n")))
    for section in SECTIONS:
        cases.append((f"remove section {section}", sub_spec(f"## {section}", f"### {section}")))
    for stage in STAGES:
        row = stage_row(inputs.plan, stage)
        cases.append((f"remove stage {stage}", sub_plan(row + "\n")))
    for phrase in PHRASES[:12]:
        cases.append((f"remove statement {phrase[:24]}", sub_spec_all(phrase, "REMOVED")))
    for phrase in PHRASES[12:]:
        cases.append((f"remove boundary {phrase}", sub_plan(phrase, "REMOVED")))
    cases.extend(
        [
            ("change selected source digest", sub_spec(next(iter(SOURCES.values())), "0" * 64)),
            ("change Song revision", sub_spec(SONG_REV, "0" * 40)),
            ("change corpus revision", sub_spec(CORPUS_REV, "1" * 40)),
            ("remove table import", source_sub("src/table/table.slang", 'use "src/kernel/mem/cell.slang"', 'use "missing.slang"')),
            ("remove std import", source_sub("src/kernel/mem/cell.slang", 'use "src/kernel/std.slang"', 'use "missing.slang"')),
            ("change COORD_CEILING", source_sub("src/kernel/std.slang", "COORD_CEILING = 8796093022208", "COORD_CEILING = 7")),
            ("remove selected function", source_sub("src/table/table.slang", "tableGet(base, cap, key) {", "tableGone(base, cap, key) {")),
            ("remove selected constant", source_sub("src/table/table.slang", "TABLE_PHI =", "TABLE_GONE =")),
            ("remove canonical theorem", lambda i: replace(i, corpus=i.corpus.replace("theorem get_insertWith_ne", "theorem gone", 1))),
            ("remove canonical namespace", lambda i: replace(i, corpus=i.corpus.replace("namespace Table", "namespace Gone", 1))),
            ("remove W21 gate", sub_plan("`python3 tools/slang/source_semantics_check.py --check harness`", "MISSING")),
            ("remove W27 control", sub_plan("preservation/admission MUST fail", "control omitted")),
            ("introduce forbidden generic budget term", sub_spec("Counts are not converted", ("fu" + "el") + " is allowed. Counts are not converted")),
        ]
    )
    cases.extend(hardening_controls(inputs))
    return cases


def hardening_controls(inputs: Inputs) -> list[Mutation]:
    """Rejecting controls for the eight reported false greens and the reference checks."""
    cases: list[Mutation] = []

    obligation = next(
        line for line in inputs.spec.splitlines() if line.startswith("| O01 |")
    )
    cases.append(
        ("empty obligation text O01", sub_spec(obligation, mutate_row_cell(obligation, 1, "")))
    )
    cases.append(
        ("empty obligation result O01", sub_spec(obligation, mutate_row_cell(obligation, 2, "")))
    )
    cases.append(
        (
            "comment-only obligation result O01",
            sub_spec(obligation, mutate_row_cell(obligation, 2, "<!-- x -->")),
        )
    )

    termination_body = section_body(inputs.spec, "Termination and source resources")
    cases.append(("empty termination section body", sub_spec(termination_body, "\n")))

    w21 = stage_row(inputs.plan, "W21")
    cases.append(("empty W21 gate", sub_plan(w21, mutate_row_cell(w21, 3, "``"))))
    cases.append(
        (
            "four comma-separated owned files W21",
            sub_plan(w21, mutate_row_cell(w21, 2, "`a.py`, `b.py`, `c.py`, `d.py`")),
        )
    )
    cases.append(
        (
            "four owned files packed in one span W21",
            sub_plan(w21, mutate_row_cell(w21, 2, "`a.py,b.py`; `c.py`; `d.py`")),
        )
    )
    cases.append(
        (
            "cyclic prerequisite W21",
            sub_plan(w21, mutate_row_cell(w21, 5, "Requires W21. Stop if a mutation can survive.")),
        )
    )
    cases.append(
        (
            "missing stop condition W21",
            sub_plan(w21, mutate_row_cell(w21, 5, "Requires this specification gate.")),
        )
    )

    w22 = stage_row(inputs.plan, "W22")
    cases.append(
        (
            "unresolved prerequisite W99",
            sub_plan(w22, mutate_row_cell(w22, 5, "Requires W99. Stop on accepted prefixes.")),
        )
    )

    cases.append(
        (
            "canonical declaration inside comment",
            lambda i: replace(
                i,
                corpus=i.corpus.replace(
                    "abbrev HMap (K V : Type) : Type := K → V",
                    "/- abbrev HMap (K V : Type) : Type := K → V -/",
                    1,
                ),
            ),
        )
    )
    cases.append(
        (
            "nonexistent canonical reference",
            sub_spec("Canonical targets resolve in", "Canonical `Apeiron.Table.bogus` targets resolve in"),
        )
    )
    cases.append(
        (
            "nonexistent canonical module reference",
            sub_spec(f"`{CANONICAL_MODULE}`", "`Apeiron/Table/Gone.lean`"),
        )
    )
    cases.append(
        (
            "nonexistent canonical reference in plan",
            sub_plan("`Apeiron.Table.insertWith`", "`Apeiron.Table.bogus`"),
        )
    )
    cases.append(("crlf source bytes retain old digest", source_crlf("src/table/table.slang")))

    cases.append(
        (
            "unresolved source import",
            source_sub(
                "src/kernel/std.slang",
                "COORD_CEILING",
                'use "src/kernel/mem/gone.slang"\nCOORD_CEILING',
            ),
        )
    )
    cases.append(
        (
            "source import cycle",
            source_sub(
                "src/kernel/std.slang",
                "COORD_CEILING",
                'use "src/table/table.slang"\nCOORD_CEILING',
            ),
        )
    )

    # A gate must be an actual runnable command, not an arbitrary backticked word.
    cases.append(("non-command gate W21", sub_plan(w21, mutate_row_cell(w21, 3, "`x`"))))
    cases.append(
        ("nonexistent path gate W21", sub_plan(w21, mutate_row_cell(w21, 3, "`./missing-gate`")))
    )
    cases.append(
        ("non-executable path gate W21", sub_plan(w21, mutate_row_cell(w21, 3, "`docs/direction.md`")))
    )

    # The termination section must retain its termination/resource statements.
    cases.append(
        (
            "prose termination section body",
            sub_spec(
                termination_body,
                "\nThis section is intentionally prose and carries no required statement.\n",
            ),
        )
    )

    # A prerequisite cycle cannot be hidden behind bare stage numbers or prose.
    cases.append(
        (
            "bare-number prerequisite cycle W21",
            sub_plan(w21, mutate_row_cell(w21, 5, "Requires 30. Stop if a mutation can survive.")),
        )
    )
    cases.append(
        (
            "prose-only prerequisite W22",
            sub_plan(w22, mutate_row_cell(w22, 5, "Requires the final stage. Stop on accepted prefixes.")),
        )
    )

    # A required field of punctuation or backticks alone is not substantive.
    cases.append(
        ("punctuation-only obligation text O01", sub_spec(obligation, mutate_row_cell(obligation, 1, "`...`")))
    )
    cases.append(
        ("backtick-only purpose W21", sub_plan(w21, mutate_row_cell(w21, 1, "```")))
    )

    # Every relative doc link and backticked module/declaration reference resolves.
    cases.append(
        (
            "nonexistent relative doc link in plan",
            sub_plan(
                "[`../spec/slang-source-semantics.md`](../spec/slang-source-semantics.md)",
                "[`../spec/slang-source-semantics.md`](../spec/slang-source-gone.md)",
            ),
        )
    )
    cases.append(
        (
            "non-Apeiron.Table declaration reference in plan",
            sub_plan("`Apeiron.Table.insertWith`", "`Apeiron.Other.insertWith`"),
        )
    )
    cases.append(
        (
            "nonexistent module path reference in plan",
            sub_plan("`Apeiron.Table.insertWith`", "`Other/Module.lean`"),
        )
    )

    # The spec and plan are bound to raw bytes, not newline-normalized text.
    cases.append(("crlf spec bytes not normalized", document_crlf("spec")))
    cases.append(("crlf plan bytes not normalized", document_crlf("plan")))
    return cases


def run_controls(inputs: Inputs, verbose: bool) -> int:
    validate(inputs)
    cases = controls(inputs)
    survived: list[str] = []
    for index, (name, mutate) in enumerate(cases, 1):
        try:
            validate(mutate(inputs))
        except CheckError as exc:
            if verbose and index == 1:
                print(f"CONTROL RED: {name}: {exc}")
        else:
            survived.append(name)
    if survived:
        fail("controls survived: " + "; ".join(survived))
    print(f"CONTROLS PASS: {len(cases)}/{len(cases)} requirement mutations rejected")
    return len(cases)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--controls", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        inputs = load()
        if args.check:
            validate(inputs)
            print(
                f"CHECK PASS: {len(OBLIGATIONS)} obligations, {len(STAGES)} stages, "
                f"{len(SOURCES)} sources, {len(CANONICAL)} canonical targets"
            )
        elif args.controls:
            run_controls(inputs, True)
        else:
            count = run_controls(inputs, True)
            validate(inputs)
            print(f"SELF-TEST PASS: baseline accepted; {count} mutations rejected")
        return 0
    except CheckError as exc:
        print(f"SPEC CHECK FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
