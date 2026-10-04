#!/usr/bin/env python3
"""S8 claim/witness checker: a stated grade must map to a named, existing artifact.

Definition of a claim
=====================
A *claim* is a place in a governing document that assigns an evidence grade
(AGENTS.md section 5: ``proved``, ``tested``, ``measured``, ``model``,
``anchor``) to a result. This checker recognizes two forms:

  * **structured records** -- a markdown table row whose grade cell is a grade
    phrase, or a ``GRADE:``/``Grade:`` field. Every structured record must name
    a resolvable artifact.
  * **prose assertions** -- a sentence containing a grade predicate
    (``prove``/``proves``/``proved``/``proven``, ``tested``, ``measured``) that
    names at least one artifact. A sentence with no named artifact is general
    prose, not an evidence claim, and is not reported. Sentences are built
    across wrapped lines; a ``... proves:`` sentence owns the contiguous list
    that follows it.

Artifacts
=========
  * a **Lean declaration** -- a backticked identifier (contains ``_`` or ``.``)
    or an ``Apeiron.``-qualified name. It resolves when a declaration or
    namespace with that name (or its last dotted component) exists in the Lean
    sources under ``proof/`` -- the refinement package plus the pinned canonical
    corpus at ``proof/corpus/.lake/packages/apeiron`` -- or when it is named by
    an ``#audit`` line in ``proof/corpus/AxiomAudit.lean``. The three
    closed-cone axioms ``propext``, ``Classical.choice`` and ``Quot.sound``
    resolve by definition. A slash shorthand such as
    ``denote_wrapping_add/sub/mul`` expands against the first segment's trailing
    underscore (``denote_wrapping_sub``, ``denote_wrapping_mul``).
  * a **gate command** -- ``cargo <sub>``, ``lake build``, ``proof/regen.sh``,
    ``proof/install-patches.sh``, ``proof/storage-controls.py`` or a
    ``python3 <path>.py`` invocation. It resolves when the file the command
    drives exists: ``Cargo.toml`` for cargo, a lakefile for lake, the script
    itself otherwise.
  * a **file** -- a backticked path or a token with a source/document
    extension. It resolves when the file exists, trying the repository root,
    ``proof/``, ``proof/corpus/`` and ``tools/`` for a bare basename.

Grade requirements
==================
  * ``proved`` must name at least one resolvable Lean declaration.
  * ``tested`` and ``measured`` must name at least one resolvable gate command.
  * ``model``/``anchor`` must name at least one resolvable artifact of any kind.
  * every named artifact, whatever the grade, must resolve.

A claim that carries a grade but names no resolvable artifact is a finding.
The check is structural: it does not run Lean or cargo, and a resolvable name is
not the axiom-cone check itself (``lake build`` and ``AxiomAudit.lean`` do that).

Coverage and exclusions
=======================
Governing documents scanned: `the design notes`, `the design notes`,
`the design notes`, ``README.md``, ``proof/README.md`` and
`the storage-refinement note`. The first five are the AGENTS.md section 1
reading order. The epic is covered from its ``## Coordinate-keyed
counting-container contract`` heading to the next heading of the same or higher
level: the sections above it are a plan whose grade records are explicitly
superseded/historical, so they are not current claims, and covering them whole
reddens on pre-existing historical records rather than on anything new. A
covered heading that is absent is a structural error, never a silent pass: the
checker refuses to report green for a region it was told to scan and did not
find. Fenced code blocks are excluded (they show commands, not grade
assertions). A sentence under a plan/modal marker (``next``, ``will``,
``must``, ``open``, ``obligation``, ...) before its predicate is an intention,
not a claim, and is not reported.

Exit status: 0 when every claim names a resolvable artifact; 1 on any finding;
2 on a missing governing document, a missing covered section, a missing Lean
source population, or an I/O error. ``--json`` emits the machine-readable
report; ``--self-test`` runs throwaway-copy positive and negative controls under
``/tmp``.

Citation forms
==============
Citations are read from backticked spans only. A markdown link destination
(``[text](path)``) is not a citation for this checker and is neither checked nor
counted as coverage; a contract that relies on link destinations alone carries no
claim record here.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]

GOVERNING_DOCUMENTS = (
    "the design notes",
    "the design notes",
    "the design notes",
    "README.md",
    "proof/README.md",
    "the storage-refinement note",
)

# Governing documents covered only from a named heading to the next heading of the
# same or higher level. An epic file is long and its earlier sections carry
# superseded/historical grade records, so whole-file coverage would report those
# as findings and the gate would be red before any new claim is made. The key is
# the exact heading text: a renamed or deleted heading is reported as a missing
# covered section (exit 2), never as a document with nothing to find.
COVERED_SECTIONS = {
    "the storage-refinement note": ("## Coordinate-keyed counting-container contract",),
}

GRADES = ("proved", "tested", "measured", "model", "anchor")
CLOSED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

PINNED_CORPUS = Path("proof/corpus/.lake/packages/apeiron")
FILE_EXTENSIONS = (
    ".lean", ".rs", ".py", ".sh", ".md", ".toml", ".json", ".yaml", ".yml", ".txt"
)
KNOWN_PATH_HEADS = {
    "proof", "src", "tests", "tools", "docs", "target", "examples", "benches",
    "fuzz", "derive", "fold-abi",
}
EXCLUDE_DIR_NAMES = {".git", ".lake", "target", "node_modules", "__pycache__", ".venv"}

BACKTICK_RE = re.compile(r"`([^`]+)`")
LEAN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.'!?]*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+\S")
DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|noncomputable\s+|partial\s+|unsafe\s+|"
    r"@\[[^\]]*\]\s*)*(?:theorem|lemma|def|abbrev|structure|inductive|class|"
    r"opaque|instance)\s+([A-Za-z_][A-Za-z0-9_.'!?]*)",
    re.MULTILINE,
)
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z_][A-Za-z0-9_.'!?]*)", re.MULTILINE)
AUDIT_RE = re.compile(r"^\s*#audit\s+([A-Za-z0-9_.'!?]+)", re.MULTILINE)

GRADE_CELL_RE = re.compile(
    r"^(?:grade\s*:?\s*)?(" + "|".join(GRADES) + r")(?:\s+(?:at|in|for|about|only|with)\b.*)?$",
    re.I,
)
GRADE_FIELD_RE = re.compile(
    r"^(?:[-*]\s*)?(?:\*\*)?grade(?:\*\*)?\s*:?\s*(.+)$", re.I
)
TABLE_SEP_RE = re.compile(r"^[\s|:-]+$")

PROSE_PREDICATES = (
    (re.compile(r"\bproved\b", re.I), "proved"),
    (re.compile(r"\bproven\b", re.I), "proved"),
    (re.compile(r"\bproves\b", re.I), "proved"),
    (re.compile(r"\bprove\b", re.I), "proved"),
    (re.compile(r"\btested\b", re.I), "tested"),
    (re.compile(r"\bmeasured\b", re.I), "measured"),
)
DEFINITIONAL_RE = re.compile(r"only when|\ba claim is\b|\bclaims are\b", re.I)
# A predicate preceded by one of these is an intention, negation or obligation,
# not an assertion of evidence.
PLAN_OR_NEGATION_RE = re.compile(
    r"\b(?:not|never|neither|without|cannot|can't|isn't|aren't|wasn't|weren't|no"
    r"|must|should|shall|will|next|future|planned|remaining|targets?|required to"
    r"|to be|open|obligation)\b",
    re.I,
)

BARE_COMMAND_RE = re.compile(
    r"\b(?:cargo\s+(?:test|clippy|fmt|kani|build|check|bench)"
    r"|lake\s+build"
    r"|proof/regen\.sh"
    r"|proof/install-patches\.sh"
    r"|proof/storage-controls\.py)\b"
)


def rel(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


# --------------------------------------------------------------------------
# Lean declaration index
# --------------------------------------------------------------------------

def iter_lean_sources(root):
    """First-party Lean sources under proof/, plus the pinned corpus package.

    Every ``.lake`` directory is skipped while walking ``proof/``; the pinned
    canonical corpus is added explicitly so dependency trees (Mathlib and
    friends) are not parsed.
    """
    sources = []
    proof = root / "proof"
    if proof.is_dir():
        for dirpath, dirnames, filenames in os.walk(proof):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIR_NAMES]
            for name in filenames:
                if name.endswith(".lean"):
                    sources.append(Path(dirpath) / name)
    pinned = root / PINNED_CORPUS
    if pinned.is_dir():
        sources.extend(sorted(pinned.rglob("*.lean")))
    return sources


def build_declaration_index(root):
    declarations = set()
    namespaces = set()
    audits = set()
    for path in iter_lean_sources(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        declarations.update(DECL_RE.findall(text))
        namespaces.update(NAMESPACE_RE.findall(text))
        if path.name == "AxiomAudit.lean":
            audits.update(AUDIT_RE.findall(text))
    return declarations, namespaces, audits


def lean_resolves(name, declarations, namespaces, audits):
    if name in CLOSED_AXIOMS:
        return True
    short = name.split(".")[-1]
    return name in declarations or short in declarations or short in namespaces or short in audits


# --------------------------------------------------------------------------
# artifact classification and resolution
# --------------------------------------------------------------------------

def _identifier_artifact(token):
    if re.match(r"^v?\d", token):
        return None
    if token.isupper() and "_" in token:
        return None
    if LEAN_IDENT_RE.match(token) and ("_" in token or "." in token):
        return {"kind": "lean", "name": token}
    return None


def classify_artifacts(token):
    """Zero or more artifacts named by one backticked token."""
    token = token.strip()
    if not token:
        return []
    if token in CLOSED_AXIOMS:
        return [{"kind": "lean", "name": token}]
    if token.startswith(("cargo ", "lake ", "python3 ", "python ")):
        return [{"kind": "command", "name": token}]
    if token.endswith((".sh", ".py")):
        return [{"kind": "command", "name": token}]
    if "/" in token:
        head = token.split("/")[0]
        if token.endswith(FILE_EXTENSIONS) or head in KNOWN_PATH_HEADS:
            return [{"kind": "file", "name": token}]
        base = head
        if "_" not in base:
            return []
        prefix = base[: base.rfind("_") + 1]
        names = [base] + [prefix + segment for segment in token.split("/")[1:] if segment]
        artifacts = []
        for name in names:
            artifact = _identifier_artifact(name)
            if artifact:
                artifacts.append(artifact)
        return artifacts
    if token.endswith(FILE_EXTENSIONS):
        return [{"kind": "file", "name": token}]
    artifact = _identifier_artifact(token)
    return [artifact] if artifact else []


def collect_artifacts(text):
    found = {}
    for raw in BACKTICK_RE.findall(text):
        for artifact in classify_artifacts(raw):
            found[(artifact["kind"], artifact["name"])] = artifact
    for match in BARE_COMMAND_RE.finditer(text):
        name = match.group(0)
        found[("command", name)] = {"kind": "command", "name": name}
    return [found[key] for key in sorted(found)]


def file_resolves(root, name):
    name = name.strip().strip("`").rstrip(".,;:)")
    if not name:
        return False
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate.exists()
    searches = [root / name]
    if "/" not in name:
        searches += [
            root / "proof" / name,
            root / "proof" / "corpus" / name,
            root / "tools" / name,
        ]
    return any(path.exists() for path in searches)


def command_resolves(root, text):
    text = text.strip()
    if re.match(r"cargo\s+\w+", text):
        return (root / "Cargo.toml").is_file()
    if text.startswith("lake"):
        return (root / "proof" / "lakefile.lean").is_file() or (
            root / "proof" / "corpus" / "lakefile.lean"
        ).is_file()
    match = re.match(r"python3?\s+(\S+)", text)
    if match:
        return file_resolves(root, match.group(1))
    head = text.split()[0] if text.split() else text
    return file_resolves(root, head)


def resolve_artifact(root, artifact, declarations, namespaces, audits):
    kind, name = artifact["kind"], artifact["name"]
    if kind == "lean":
        return lean_resolves(name, declarations, namespaces, audits)
    if kind == "command":
        return command_resolves(root, name)
    return file_resolves(root, name)


# --------------------------------------------------------------------------
# claim extraction
# --------------------------------------------------------------------------

def covered_regions(relative, lines):
    """``(start, end, heading)`` index ranges this document contributes.

    A document absent from ``COVERED_SECTIONS`` is covered whole as
    ``(0, len(lines), "")``. A covered document contributes one region per named
    heading, running to the next heading of the same or higher level. ``start``
    is the 0-based index of the region's first line, so a claim found at
    relative line ``n`` is reported at file line ``start + n``.

    Returns an empty list when the document is listed in ``COVERED_SECTIONS`` and
    none of its headings is present; the caller reports that as a structural
    error instead of scanning nothing.
    """
    headings = COVERED_SECTIONS.get(relative)
    if not headings:
        return [(0, len(lines), "")]
    regions = []
    for index, line in enumerate(lines):
        match = HEADING_RE.match(line)
        if not match or line.strip() not in headings:
            continue
        level = len(match.group(1))
        end = len(lines)
        for cursor in range(index + 1, len(lines)):
            following = HEADING_RE.match(lines[cursor])
            if following and len(following.group(1)) <= level:
                end = cursor
                break
        regions.append((index, end, line.strip()))
    return regions


def make_claim(document, line, grade, kind, text, artifacts):
    return {
        "document": document,
        "line": line,
        "grade": grade,
        "kind": kind,
        "text": text.strip(),
        "artifacts": artifacts,
    }


def table_grade(cells):
    for cell in cells:
        match = GRADE_CELL_RE.match(cell.replace("*", "").strip())
        if match:
            return match.group(1).lower()
    return None


def iter_blocks(lines):
    """Group lines into prose and table blocks, skipping fenced code."""
    blocks = []
    in_fence = False
    fence_marker = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, None
            index += 1
            continue
        if in_fence or not stripped:
            index += 1
            continue
        if stripped.startswith("|") and stripped.count("|") >= 2:
            block = {"kind": "table", "lines": []}
            while index < len(lines):
                candidate = lines[index].strip()
                if candidate.startswith("|") and candidate.count("|") >= 2:
                    block["lines"].append((index + 1, lines[index]))
                    index += 1
                else:
                    break
            blocks.append(block)
            continue
        block = {"kind": "prose", "lines": []}
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate or candidate.startswith("|") or candidate.startswith(("```", "~~~")):
                break
            block["lines"].append((index + 1, lines[index]))
            index += 1
        blocks.append(block)
    return blocks


def sentence_spans(text):
    """Sentence spans that do not split inside dotted identifiers."""
    spans = []
    start = 0
    for position, character in enumerate(text):
        if character in ".!?":
            following = text[position + 1] if position + 1 < len(text) else ""
            if following == "" or following.isspace():
                spans.append((start, position + 1))
                start = position + 1
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def line_at(line_starts, offset):
    line = line_starts[0][1]
    for start, number in line_starts:
        if start <= offset:
            line = number
        else:
            break
    return line


def predicate_asserted(sentence, match):
    before = sentence[:match.start()]
    # A guard word inside a backticked artifact name ("no-such-gate.sh") is not
    # a negation of the predicate; blank the spans before matching.
    before = BACKTICK_RE.sub(" ", before)
    return not PLAN_OR_NEGATION_RE.search(before)


def is_list_block(block):
    first = block["lines"][0][1]
    return first.lstrip().startswith(("-", "*")) or first[:1].isspace()


def prose_claims(document, blocks, block_index):
    block = blocks[block_index]
    pieces = []
    line_starts = []
    offset = 0
    for line_number, raw in block["lines"]:
        text = raw.strip()
        line_starts.append((offset, line_number))
        pieces.append(text)
        offset += len(text) + 1
    joined = " ".join(pieces)

    claims = []
    for start, end in sentence_spans(joined):
        sentence = joined[start:end].strip()
        if not sentence or DEFINITIONAL_RE.search(sentence):
            continue
        grade = None
        for pattern, candidate in PROSE_PREDICATES:
            match = pattern.search(sentence)
            if match and predicate_asserted(sentence, match):
                grade = candidate
                break
        if grade is None:
            continue
        artifacts = collect_artifacts(sentence)
        if not artifacts:
            continue
        if joined.rstrip().endswith(":") and end >= len(joined.rstrip()) - 1:
            cursor = block_index + 1
            while cursor < len(blocks) and blocks[cursor]["kind"] == "prose" and is_list_block(blocks[cursor]):
                for _, raw in blocks[cursor]["lines"]:
                    for artifact in collect_artifacts(raw):
                        if artifact not in artifacts:
                            artifacts.append(artifact)
                cursor += 1
        claims.append(
            make_claim(document, line_at(line_starts, start), grade, "prose", sentence, artifacts)
        )
    return claims


def extract_claims(document, text):
    claims = []
    blocks = iter_blocks(text.splitlines())
    for block_index, block in enumerate(blocks):
        if block["kind"] == "table":
            for line_number, raw in block["lines"]:
                stripped = raw.strip()
                if TABLE_SEP_RE.match(stripped):
                    continue
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                grade = table_grade(cells)
                if grade:
                    claims.append(
                        make_claim(document, line_number, grade, "structured", raw,
                                   collect_artifacts(" ".join(cells)))
                    )
            continue
        field_lines = [
            (line_number, raw)
            for line_number, raw in block["lines"]
            if GRADE_FIELD_RE.match(raw.strip())
        ]
        if field_lines:
            for line_number, raw in field_lines:
                field = GRADE_FIELD_RE.match(raw.strip()).group(1).replace("*", "").strip()
                match = GRADE_CELL_RE.match(field)
                if match:
                    claims.append(
                        make_claim(document, line_number, match.group(1).lower(),
                                   "structured", raw, collect_artifacts(raw))
                    )
            continue
        claims.extend(prose_claims(document, blocks, block_index))
    return claims


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate_claim(root, claim, declarations, namespaces, audits):
    findings = []
    for artifact in claim["artifacts"]:
        artifact["resolved"] = resolve_artifact(
            root, artifact, declarations, namespaces, audits
        )
    resolved = [a for a in claim["artifacts"] if a.get("resolved")]
    resolved_lean = [a for a in resolved if a["kind"] == "lean"]
    resolved_command = [a for a in resolved if a["kind"] == "command"]
    location = f"{claim['document']}:{claim['line']}"

    for artifact in claim["artifacts"]:
        if not artifact.get("resolved"):
            findings.append(
                f"{location}: grade `{claim['grade']}` names unresolved "
                f"{artifact['kind']} `{artifact['name']}`"
            )
    if claim["grade"] == "proved" and not resolved_lean:
        findings.append(
            f"{location}: grade `proved` names no resolvable Lean theorem"
        )
    if claim["grade"] in ("tested", "measured") and not resolved_command:
        findings.append(
            f"{location}: grade `{claim['grade']}` names no resolvable gate command"
        )
    if claim["grade"] in ("model", "anchor") and not resolved:
        findings.append(
            f"{location}: grade `{claim['grade']}` names no resolvable artifact"
        )
    return findings


def build_report(root):
    root = Path(root).resolve()
    findings = []
    documents = []
    missing = []
    for relative in GOVERNING_DOCUMENTS:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
        else:
            documents.append((relative, path))

    if missing:
        return {
            "root": str(root),
            "documents": [relative for relative, _ in documents],
            "covered_sections": [],
            "missing_sections": [],
            "missing_documents": missing,
            "claims": [],
            "findings": [f"governing document missing: {name}" for name in missing],
            "ok": False,
            "error": True,
        }

    declarations, namespaces, audits = build_declaration_index(root)
    if not declarations:
        return {
            "root": str(root),
            "documents": [relative for relative, _ in documents],
            "covered_sections": [],
            "missing_sections": [],
            "missing_documents": [],
            "claims": [],
            "findings": ["no Lean declarations found under proof/ or the pinned corpus"],
            "ok": False,
            "error": True,
        }

    claims = []
    sections = []
    missing_sections = []
    for relative, path in documents:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            findings.append(f"{relative}: cannot read: {error}")
            continue
        lines = text.splitlines()
        regions = covered_regions(relative, lines)
        for heading in COVERED_SECTIONS.get(relative, ()):
            if not any(heading == region[2] for region in regions):
                missing_sections.append(f"{relative}: covered section `{heading}` not found")
        for start, end, heading in regions:
            sections.append(f"{relative}:{start + 1}" + (f" `{heading}`" if heading else " (whole file)"))
            for claim in extract_claims(relative, "\n".join(lines[start:end])):
                claim["line"] += start
                claim_findings = evaluate_claim(
                    root, claim, declarations, namespaces, audits
                )
                claim["ok"] = not claim_findings
                claims.append(claim)
                findings.extend(claim_findings)

    findings.extend(missing_sections)
    return {
        "root": str(root),
        "documents": [relative for relative, _ in documents],
        "covered_sections": sections,
        "missing_sections": missing_sections,
        "missing_documents": [],
        "declaration_count": len(declarations),
        "namespace_count": len(namespaces),
        "audit_count": len(audits),
        "claims": claims,
        "findings": findings,
        "ok": not findings and not missing_sections,
        "error": bool(missing_sections),
    }


def print_report(report):
    print(f"repo: {report['root']}")
    print(f"documents: {len(report['documents'])} scanned: {', '.join(report['documents'])}")
    for entry in report.get("covered_sections", []):
        if not entry.endswith("(whole file)"):
            print(f"covered: {entry}")
    if report.get("error"):
        print(f"\nERROR: {len(report['findings'])} structural problem(s)")
        for finding in report["findings"]:
            print(f"  - {finding}")
        return
    structured = sum(1 for claim in report["claims"] if claim["kind"] == "structured")
    prose = len(report["claims"]) - structured
    print(
        f"[claims] {len(report['claims'])} graded claims "
        f"({structured} structured, {prose} prose); "
        f"{report['declaration_count']} Lean declarations, "
        f"{report['namespace_count']} namespaces, "
        f"{report['audit_count']} #audit names indexed"
    )
    for claim in report["claims"]:
        status = "OK  " if claim["ok"] else "FAIL"
        print(f"  {status} {claim['document']}:{claim['line']} grade={claim['grade']} "
              f"kind={claim['kind']}")
    if report["findings"]:
        print(f"\nFAIL: {len(report['findings'])} unbacked claim(s)")
        for finding in report["findings"]:
            print(f"  - {finding}")
    else:
        print("\nOK: every graded claim names a resolvable artifact")


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

def _run_checker(root):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--json",
        "--root",
        str(root),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


EPIC_CONTRACT_SECTION = """# storage refinement

## Superseded spikes

Superseded records live above the contract section and are out of coverage.

## Coordinate-keyed counting-container contract

The constructor witness `word_table_new_rep` is proved.
"""


def _synthetic_tree(base):
    """A minimal tree whose one proved claim is backed."""
    _write(base / "Cargo.toml", "[workspace]\n")
    _write(base / "proof" / "regen.sh", "#!/bin/sh\nexit 0\n")
    _write(base / "proof" / "corpus" / "lakefile.lean", "import Lean\n")
    _write(
        base / "proof" / "corpus" / "TableRefinement.lean",
        "namespace Apeiron.TableRefinement\n\ntheorem word_table_new_rep : True := trivial\n",
    )
    _write(
        base / "proof" / "corpus" / "AxiomAudit.lean",
        "#audit word_table_new_rep\n",
    )
    _write(base / "docs" / "direction.md", "# direction\n")
    _write(
        base / "docs" / "kernel-boundary.md",
        "# boundary\n\nThe constructor witness `word_table_new_rep` is proved.\n",
    )
    _write(base / "docs" / "plan.md", "# handoff\n")
    _write(base / "README.md", "# readme\n")
    _write(base / "proof" / "README.md", "# proof\n")
    _write(base / "docs" / "epics" / "storage-refinement.md", EPIC_CONTRACT_SECTION)


def self_test():
    workdir = Path(tempfile.mkdtemp(prefix="apeiron-claims-selftest-"))
    results = []

    def record(name, expected, proc, must_contain=(), must_not_contain=()):
        try:
            report = json.loads(proc.stdout)
            findings = report.get("findings", [])
        except (ValueError, AttributeError):
            findings = ["<unparseable JSON output>"]
        text = "\n".join(findings)
        passed = proc.returncode == expected
        if expected == 0:
            passed = passed and not findings
        else:
            passed = passed and bool(findings)
        for needle in must_contain:
            passed = passed and needle in text
        for needle in must_not_contain:
            passed = passed and needle not in text
        results.append((name, expected, proc.returncode, passed, findings))

    try:
        base = workdir / "base"
        _synthetic_tree(base)

        record("positive: backed proved claim passes", 0, _run_checker(base))

        def fresh(name):
            target = workdir / name
            shutil.copytree(base, target)
            return target

        dotted = fresh("exact-dotted-declaration")
        dotted_source = dotted / "proof" / "Generated.lean"
        _write(dotted_source, "def table.Table.apply_insert : Nat := 0\n")
        with (dotted / "docs" / "kernel-boundary.md").open("a", encoding="utf-8") as handle:
            handle.write("\n`table.Table.apply_insert` is proved.\n")
        record("positive: exact dotted declaration passes", 0, _run_checker(dotted))
        _write(dotted_source, "def other.Table.apply_insert : Nat := 0\n")
        record("negative: different dotted declaration rejected", 1, _run_checker(dotted),
               ("table.Table.apply_insert",))
        dotted_source.unlink()
        record("negative: deleted dotted declaration rejected", 1, _run_checker(dotted),
               ("table.Table.apply_insert",))

        def unbacked_prose(tree):
            with (tree / "docs" / "kernel-boundary.md").open("a", encoding="utf-8") as handle:
                handle.write("\n`not_a_real_theorem` is proved.\n")

        def unbacked_structured(tree):
            with (tree / "docs" / "kernel-boundary.md").open("a", encoding="utf-8") as handle:
                handle.write("\n| item | status |\n| --- | --- |\n| thing | proved at stated scope |\n")

        def missing_command(tree):
            with (tree / "docs" / "kernel-boundary.md").open("a", encoding="utf-8") as handle:
                handle.write("\nThe gate `proof/no-such-gate.sh` is tested.\n")

        def missing_document(tree):
            (tree / "docs" / "direction.md").unlink()

        def epic_bad_citation(tree):
            with (tree / "docs" / "epics" / "storage-refinement.md").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(
                    "\nThe full-key oracle lives in the no-such-oracle note "
                    "and is tested.\n"
                )

        def epic_renamed_section(tree):
            path = tree / "docs" / "epics" / "storage-refinement.md"
            text = path.read_text(encoding="utf-8").replace(
                "## Coordinate-keyed counting-container contract",
                "## Counting container",
            )
            _write(path, text)

        def epic_superseded_record(tree):
            _write(
                tree / "docs" / "epics" / "storage-refinement.md",
                EPIC_CONTRACT_SECTION.replace(
                    "## Superseded spikes",
                    "## Superseded spikes\n\nThe removed head `no_such_removed_head` is proved.",
                ),
            )

        for name, mutate, expected, contains, excludes in (
            (
                "negative: unbacked proved claim rejected",
                unbacked_prose,
                1,
                ("not_a_real_theorem",),
                ("word_table_new_rep",),
            ),
            (
                "negative: structured proved with no artifact rejected",
                unbacked_structured,
                1,
                ("names no resolvable Lean theorem",),
                (),
            ),
            (
                "negative: missing gate command rejected",
                missing_command,
                1,
                ("proof/no-such-gate.sh",),
                (),
            ),
            (
                "negative: nonexistent path cited in the epic contract rejected",
                epic_bad_citation,
                1,
                ("the storage-refinement note", "the no-such-oracle note"),
                (),
            ),
            (
                "positive: superseded epic record above the contract not reported",
                epic_superseded_record,
                0,
                (),
                ("no_such_removed_head",),
            ),
            (
                "error: renamed epic contract section rejected",
                epic_renamed_section,
                2,
                ("## Coordinate-keyed counting-container contract",),
                (),
            ),
            (
                "error: missing governing document",
                missing_document,
                2,
                ("the design notes",),
                (),
            ),
        ):
            slug = "".join(ch if ch.isalnum() else "-" for ch in name)
            tree = fresh(slug)
            mutate(tree)
            record(name, expected, _run_checker(tree), contains, excludes)

        for name, expected, observed, passed, findings in results:
            verdict = "PASS" if passed else "FAIL"
            print(f"[self-test] {name}: expected exit {expected}, observed {observed} -> {verdict}")
            for finding in findings[:3]:
                print(f"            {finding}")

        failed = [name for name, _, _, passed, _ in results if not passed]
        print(
            f"\n[self-test] {len(results) - len(failed)}/{len(results)} "
            "controls behaved as required"
        )
        if failed:
            print("[self-test] FAILED: " + ", ".join(failed))
            return 1
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="repository root to check")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="report claims and findings")
    group.add_argument("--json", action="store_true", help="machine-readable report")
    group.add_argument("--self-test", action="store_true", help="run throwaway-copy controls")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    try:
        report = build_report(args.root)
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    if report.get("error"):
        return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
